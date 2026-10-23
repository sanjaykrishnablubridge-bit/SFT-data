"""SFT one mix at one LR for the fixed token budget (loss on assistant tokens only).

Writes <out>/metrics.json and, with --save, an eval-ready model to <out>/model.
Normally launched by run.py:
  python train.py --config configs/experiment.yaml --mix runs/x/mix.json --lr 3e-4 --seed 0 --out runs/x
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import data
from decontam import Decontaminator


@torch.no_grad()
def val_loss(model, ids, mask, mbs):
    """Mean loss over assistant tokens of the held-out blocks."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(ids), mbs):
        x, labels = batch(ids[i:i + mbs], mask[i:i + mbs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
            ignore_index=-100, reduction="sum")
        tot += loss.item()
        n += (labels[:, 1:] != -100).sum().item()
    model.train()
    return tot / max(n, 1)


def batch(ids, mask):
    x = torch.from_numpy(ids.astype(np.int64)).cuda()
    labels = x.masked_fill(~torch.from_numpy(mask).cuda(), -100)
    return x, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mix", required=True, help="JSON file: {component: fraction}")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--override", default="{}", help="JSON dict merged over the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg.update(json.loads(args.override))
    fractions = json.loads(Path(args.mix).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    dec = None
    if cfg.get("decontam", {}).get("enabled", True):
        dc = cfg["decontam"]
        dec = Decontaminator(dc["tasks"], dc.get("ngram", 13), dc.get("min_overlap", 0.5))
    d = data.build(cfg, fractions, tok, dec)
    train_ids, train_mask = d["train_ids"], d["train_mask"]

    # The seed only changes the order of the fixed training blocks.
    order = np.random.default_rng(args.seed).permutation(len(train_ids))
    train_ids, train_mask = train_ids[order], train_mask[order]
    torch.manual_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).cuda()
    model.config.use_cache = False

    mbs, accum = cfg["micro_batch_size"], cfg["grad_accum"]
    steps = len(train_ids) // (mbs * accum)
    warmup = max(1, int(cfg["warmup_frac"] * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=cfg["weight_decay"], fused=True)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, steps - warmup)
        floor = args.lr * cfg["min_lr_ratio"]
        return floor + 0.5 * (args.lr - floor) * (1 + math.cos(math.pi * p))

    v0 = val_loss(model, d["val_ids"], d["val_mask"], mbs)
    print(f"[{out}] lr={args.lr:g} seed={args.seed} steps={steps} "
          f"tokens={d['meta']['train_tokens']:,} loss_tokens={d['meta']['loss_tokens']:,} val0={v0:.4f}", flush=True)

    history, diverged, t0 = [], False, time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        lo = step * mbs * accum
        n_loss = int(train_mask[lo:lo + mbs * accum, 1:].sum())
        step_loss = 0.0
        for k in range(accum):
            x, labels = batch(train_ids[lo + k * mbs: lo + (k + 1) * mbs],
                              train_mask[lo + k * mbs: lo + (k + 1) * mbs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x).logits.float()
            # normalise by loss tokens in the whole step, so every assistant token weighs the same
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                ignore_index=-100, reduction="sum") / max(n_loss, 1)
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not math.isfinite(step_loss):
            diverged = True
            print("loss is not finite; stopping", flush=True)
            break
        if step % 20 == 0 or step == steps - 1:
            el = time.time() - t0
            history.append({"step": step, "loss": step_loss, "lr": lr_at(step)})
            print(f"step {step}/{steps} loss {step_loss:.4f} lr {lr_at(step):.2e} "
                  f"{(step + 1) * mbs * accum * cfg['seq_len'] / el:,.0f} tok/s", flush=True)

    v1 = float("inf") if diverged else val_loss(model, d["val_ids"], d["val_mask"], mbs)
    metrics = {"lr": args.lr, "seed": args.seed, "steps": steps, "val_loss_init": v0, "val_loss": v1,
               "diverged": diverged, "minutes": (time.time() - t0) / 60, "history": history,
               "decontam_coverage": dec.coverage if dec else None} | d["meta"]
    if args.save and not diverged:
        # Save in SmolLM2-Instruct format so lm-eval prompts it exactly like the target model.
        tok.chat_template = data.chat_template(cfg["system_prompt"])
        tok.eos_token = "<|im_end|>"
        model.generation_config.eos_token_id = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
        model.config.use_cache = True
        model.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done: val {v0:.4f} -> {v1:.4f}", flush=True)


if __name__ == "__main__":
    main()
