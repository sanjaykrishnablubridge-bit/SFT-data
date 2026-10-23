"""SFT data-mix experiments, end to end. Resumable: finished runs/evals are skipped.

  python run.py                          # everything in the config
  python run.py --mixes control          # just the control
  python run.py --mixes 'add:*'          # one family of experiments
  python run.py --list                   # show components and expanded mixes, run nothing
  python run.py --smoke                  # tiny end-to-end check (~10 min)

Per mix: LR sweep (best LR by held-out val loss) -> final seeds at best LR ->
lm-eval on each. runs/report.md compares every mix to the control and to the
target model, with a noise threshold and a data/fairness audit.
"""
import argparse
import fnmatch
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

import data

SMOKE = {
    "token_budget": 1_048_576, "val_tokens": 65_536, "micro_batch_size": 2, "grad_accum": 2,
    "lr_grid": [3.0e-4, 1.0e-3], "final_seeds": [0], "eval_limit": 20,
}


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- training

def train(ctx, mix_name, fractions, lr, seed, save):
    out = ctx.root / "mixes" / data.safe_name(mix_name) / ("final" if save else "sweep") / f"lr{lr:g}_seed{seed}"
    if (out / "metrics.json").exists():
        return out, json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "mix.json").write_text(json.dumps(fractions, indent=2))
    sh([sys.executable, "train.py", "--config", ctx.args.config, "--mix", str(out / "mix.json"),
        "--lr", str(lr), "--seed", str(seed), "--out", str(out),
        "--override", json.dumps(ctx.override)] + (["--save"] if save else []))
    return out, json.loads((out / "metrics.json").read_text())


# ---------------------------------------------------------------- evaluation

def evaluate(ctx, model_path, out, chat):
    """lm-eval on every task in eval_groups. Returns {task: (score, stderr)}; cached."""
    cache = out / "scores.json"
    if cache.exists():
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    scores = {}
    for use_chat in (True, False):
        tasks = {t: m for g in ctx.cfg["eval_groups"].values() if g["chat"] == use_chat
                 for t, m in g["tasks"].items()}
        if not tasks or (use_chat and not chat):
            continue
        res_dir = out / ("lm_eval_chat" if use_chat else "lm_eval")
        cmd = ["lm_eval", "--model", "hf", "--model_args", f"pretrained={model_path},dtype=bfloat16",
               "--tasks", ",".join(tasks), "--batch_size", str(ctx.cfg["eval_batch_size"]),
               "--output_path", str(res_dir), "--seed", "0"]
        if use_chat:
            cmd += ["--apply_chat_template", "--fewshot_as_multiturn"]
        if ctx.cfg.get("eval_limit"):
            cmd += ["--limit", str(ctx.cfg["eval_limit"])]
        sh(cmd)
        res = json.loads(max(res_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime).read_text())
        for t, metric in tasks.items():
            r = res["results"][t]
            name, _, filt = metric.partition(",")
            scores[t] = (r[metric], r.get(f"{name}_stderr,{filt}") or 0.0)
    cache.write_text(json.dumps(scores, indent=2))
    return scores


# ---------------------------------------------------------------- pipeline

def run_mix(ctx, name, fractions, fixed_lr):
    cfg = ctx.cfg
    sweep = {}
    if fixed_lr is None:
        for lr in cfg["lr_grid"]:
            sweep[lr] = train(ctx, name, fractions, lr, cfg["sweep_seed"], save=False)[1]
        best = min(sweep, key=lambda lr: sweep[lr]["val_loss"])
    else:
        best = fixed_lr
    finals = []
    for seed in cfg["final_seeds"]:
        out, m = train(ctx, name, fractions, best, seed, save=True)
        s = None
        if not m["diverged"]:
            s = evaluate(ctx, out / "model", out, chat=True)
            if not ctx.args.keep_models and (out / "model").exists():
                shutil.rmtree(out / "model")
        finals.append((m, s))
    return {"fractions": fractions, "sweep": sweep, "best_lr": best, "finals": finals}


class Ctx:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--mixes", default="*", help="comma-separated globs over mix names (control always runs)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list", action="store_true", help="print components and mixes, then exit")
    ap.add_argument("--keep-models", action="store_true")
    args = ap.parse_args()

    ctx = Ctx()
    ctx.args = args
    ctx.cfg = yaml.safe_load(open(args.config))
    ctx.override = dict(SMOKE) if args.smoke else {}
    ctx.cfg.update(ctx.override)
    ctx.root = Path("runs_smoke" if args.smoke else "runs")
    ctx.root.mkdir(exist_ok=True)
    cfg = ctx.cfg

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    cat = data.catalog(cfg, tok)
    mixes = data.expand_mixes(cfg, cat)
    control = cfg["control"]
    pats = args.mixes.split(",")
    selected = [control] + [m for m in mixes if m != control and any(fnmatch.fnmatchcase(m, p) for p in pats)]

    if args.list:
        print(f"{'component':60} {'rows':>9} {'fits':>5} {'avg tok':>8} {'usable tokens':>14}")
        for c, v in cat.items():
            print(f"{c:60} {v['rows']:>9,} {v['fit_frac']:>5.0%} {v['avg_tokens']:>8.0f} {v['tokens']:>14,}")
        print(f"\n{len(mixes)} mixes ({len(selected)} selected):")
        for m in mixes:
            mark = "*" if m in selected else " "
            print(f" {mark} {m}  ({len(mixes[m])} components)")
        return

    refs = {}
    for r in cfg["references"]:
        refs[r["name"]] = evaluate(ctx, r["model"], ctx.root / "references" / data.safe_name(r["name"]), r["chat"])

    results = {}
    results[control] = run_mix(ctx, control, mixes[control], None)
    fixed = results[control]["best_lr"] if cfg["lr_policy"] == "control" else None
    for name in selected[1:]:
        results[name] = run_mix(ctx, name, mixes[name], fixed)
        write_report(ctx, cat, refs, results)  # refresh after every mix so partial results are usable
    write_report(ctx, cat, refs, results)
    print((ctx.root / "report.md").read_text())


# ---------------------------------------------------------------- report

def task_stats(finals, task):
    """(mean, standard error) over seeds, falling back to lm-eval's stderr for one seed."""
    vals = [s[task] for _, s in finals if s and task in s]
    if not vals:
        return None
    mean = statistics.mean(v for v, _ in vals)
    bench_se = statistics.mean(se for _, se in vals)
    seed_var = statistics.variance([v for v, _ in vals]) / len(vals) if len(vals) > 1 else 0.0
    return mean, math.sqrt(seed_var + bench_se ** 2 / len(vals))


def write_report(ctx, cat, refs, results):
    cfg = ctx.cfg
    control = cfg["control"]
    groups = cfg["eval_groups"]
    tasks = [t for g in groups.values() for t in g["tasks"]]
    target = cfg.get("target")
    L = [f"# SFT data-mix results on `{cfg['model']}`", "",
         f"Every mix: {cfg['token_budget']:,} training tokens, same recipe; LR policy `{cfg['lr_policy']}`; "
         f"seeds {cfg['final_seeds']}. Control: `{control}`.", ""]
    if target:
        L += [f"Target: `{target}` was trained with SFT **and DPO**; these runs are SFT only, "
              "so a gap on IFEval/chat is expected until DPO is added.", ""]
    if cfg.get("eval_limit"):
        L += [f"**SMOKE RUN: evals limited to {cfg['eval_limit']} items; scores are meaningless.**", ""]

    # scores table
    L += ["## Scores", "", "Mean over seeds (± standard error). Δ is vs the control; "
          "**▲/▼** = better/worse than control by more than 2× the combined noise.", ""]
    L += ["| model | " + " | ".join(tasks) + " | " + " | ".join(f"{g} avg" for g in groups) + " |",
          "|---|" + "---|" * (len(tasks) + len(groups))]
    for rname, sc in refs.items():
        cells = [f"{sc[t][0]:.3f}" if t in sc else "–" for t in tasks]
        gavg = [_gavg(sc, g) for g in groups.values()]
        L.append(f"| *{rname}* | " + " | ".join(cells) + " | " + " | ".join(gavg) + " |")
    ctrl = {t: task_stats(results[control]["finals"], t) for t in tasks}
    for name, r in results.items():
        cells = []
        for t in tasks:
            st = task_stats(r["finals"], t)
            if not st:
                cells.append("–")
                continue
            cell = f"{st[0]:.3f}±{st[1]:.3f}"
            if name != control and ctrl[t]:
                d = st[0] - ctrl[t][0]
                noise = 2 * math.sqrt(st[1] ** 2 + ctrl[t][1] ** 2)
                mark = " **▲**" if d > noise else " **▼**" if d < -noise else ""
                cell += f" ({d:+.3f}{mark})"
            cells.append(cell)
        means = {t: (task_stats(r["finals"], t) or (None,))[0] for t in tasks}
        gavg = []
        for g in groups.values():
            vs = [means[t] for t in g["tasks"] if means[t] is not None]
            gavg.append(f"{statistics.mean(vs):.3f}" if vs else "–")
        label = f"**{name}**" if name == control else name
        L.append(f"| {label} | " + " | ".join(cells) + " | " + " | ".join(gavg) + " |")

    if target in refs:
        L += ["", "## vs target", ""]
        for name, r in results.items():
            beat = [t for t in tasks if (st := task_stats(r["finals"], t)) and t in refs[target]
                    and st[0] > refs[target][t][0]]
            L.append(f"- `{name}` beats `{target}` on {len(beat)}/{len(tasks)}: {', '.join(beat) or '—'}")

    # LR
    L += ["", "## Learning rate", "",
          "Best LR per mix is chosen by loss on that mix's own held-out val split (never by benchmark).", ""]
    for name, r in results.items():
        if r["sweep"]:
            row = ", ".join(f"{lr:g}: " + ("diverged" if m["diverged"] else f"{m['val_loss']:.4f}")
                            for lr, m in r["sweep"].items())
            edge = r["best_lr"] in (min(cfg["lr_grid"]), max(cfg["lr_grid"]))
            L.append(f"- `{name}`: best {r['best_lr']:g} ({row})" + (" — **at grid edge, widen lr_grid**" if edge else ""))
        else:
            L.append(f"- `{name}`: {r['best_lr']:g} (control's best; lr_policy=control)")

    # audit
    L += ["", "## Data & fairness audit", ""]
    all_runs = [m for r in results.values() for m in list(r["sweep"].values()) + [m for m, _ in r["finals"]]]
    budgets = {m["train_tokens"] for m in all_runs}
    L.append(f"- Equal training tokens across every run: **{'PASS' if len(budgets) == 1 else 'FAIL'}** {sorted(budgets)}")
    cov = next((m.get("decontam_coverage") for m in all_runs if m.get("decontam_coverage")), None)
    if cov:
        L.append("- Decontamination coverage (13-gram overlap): " + "; ".join(f"{k}: {v}" for k, v in cov.items()))
    for name, r in results.items():
        m = r["finals"][0][0]
        comps = m["components"]
        contam = sum(c["dropped_contaminated"] for c in comps.values())
        long_ = sum(c["dropped_too_long"] for c in comps.values())
        rep = [f"{c} ×{v['repeats']:.1f}" for c, v in comps.items() if v["repeats"] > 1.01]
        line = (f"- `{name}`: {len(comps)} components, assistant (loss) tokens {m['loss_tokens']:,}, "
                f"dropped {contam:,} contaminated / {long_:,} too long")
        if rep:
            line += f"; **repeated data**: {', '.join(rep)}"
        L.append(line)
    if len(cfg["final_seeds"]) < 2:
        L.append("- Only one seed: ▲/▼ use benchmark stderr only, which ignores training noise. Use ≥2 seeds before trusting a win.")

    L += ["", "## Components", "", "| component | rows | fit in seq_len | avg tokens | usable tokens |", "|---|---|---|---|---|"]
    used = {c for r in results.values() for c in r["fractions"]}
    for c, v in cat.items():
        if c in used:
            L.append(f"| {c} | {v['rows']:,} | {v['fit_frac']:.0%} | {v['avg_tokens']:.0f} | {v['tokens']:,} |")
    (ctx.root / "report.md").write_text("\n".join(L) + "\n")
    (ctx.root / "results.json").write_text(json.dumps(
        {n: {"fractions": r["fractions"], "best_lr": r["best_lr"],
             "scores": [s for _, s in r["finals"]]} for n, r in results.items()}, indent=2))


def _gavg(sc, g):
    vs = [sc[t][0] for t in g["tasks"] if t in sc]
    return f"{statistics.mean(vs):.3f}" if vs else "–"


if __name__ == "__main__":
    main()
