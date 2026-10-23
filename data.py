"""SFT data: sources -> components -> mixes -> packed, loss-masked token blocks.

A *source* is a Hugging Face dataset in the config. With `split_by: <column>`
it is broken into one *component* per value of that column (e.g. each
`source` inside smol-smoltalk or tulu-3), named "<source>/<value>".

A *mix* maps component globs to weights (a number, or "natural" = the
component's own token share). Every mix is built to exactly the same token
budget; components that run out are repeated and the repeat factor recorded.
"""
import fnmatch
import hashlib
import json
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from datasets import load_dataset

CACHE = Path("cache")
SMOLLM2_CHAT_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\n' + system_prompt + '<|im_end|>\n' }}{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def chat_template(system_prompt):
    """The SmolLM2-Instruct template, so our models are prompted exactly like the target."""
    literal = "'" + system_prompt.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return SMOLLM2_CHAT_TEMPLATE.replace("system_prompt", literal)


# ---------------------------------------------------------------- sources

@lru_cache(maxsize=None)
def _load(hf_id, subset, split):
    return load_dataset(hf_id, subset, split=split)


def _source_ds(src):
    return _load(src["hf_id"], src.get("subset"), src.get("split", "train"))


def rows(ds, idx):
    """Fetch many rows at once (much faster than ds[i] in a loop)."""
    batch = ds[idx]
    return [dict(zip(batch, vals)) for vals in zip(*batch.values())]


def to_messages(src, ex):
    """Normalise one example to a list of {role, content}."""
    if "prompt_field" in src:
        return [{"role": "user", "content": ex[src["prompt_field"]]},
                {"role": "assistant", "content": ex[src["response_field"]]}]
    return ex[src.get("messages_field", "messages")]


def catalog(cfg, tok):
    """{component: {source, value, rows, avg_tokens, tokens}}; cached on disk."""
    key = fingerprint([cfg["sources"], cfg["model"], cfg["seq_len"], cfg["system_prompt"]])
    path = CACHE / f"catalog_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {}
    for src in cfg["sources"]:
        ds = _source_ds(src)
        groups = _component_indices(src)
        for value, idx in groups.items():
            name = src["name"] if value is None else f"{src['name']}/{value}"
            sample = random.Random(0).sample(idx, min(300, len(idx)))
            n_tok = [len(t) for t in _render_tokenize(tok, cfg, [to_messages(src, r) for r in rows(ds, sample)])[0]]
            fits = [n for n in n_tok if n <= cfg["seq_len"]]
            avg = sum(fits) / len(fits) if fits else 0.0
            usable = len(idx) * len(fits) / len(n_tok)
            # "tokens" counts only examples that fit in seq_len, so natural weights are achievable
            out[name] = {"source": src["name"], "value": value, "rows": len(idx),
                         "fit_frac": round(len(fits) / len(n_tok), 3),
                         "avg_tokens": round(avg, 1), "tokens": int(avg * usable)}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


_INDICES = {}


def _component_indices(src):
    key = json.dumps(src, sort_keys=True)
    if key not in _INDICES:
        _INDICES[key] = _compute_indices(src)
    return _INDICES[key]


def _compute_indices(src):
    ds = _source_ds(src)
    col = src.get("split_by")
    if not col:
        return {None: list(range(len(ds)))}
    groups = defaultdict(list)
    for i, v in enumerate(ds[col]):
        groups[v].append(i)
    return dict(groups)


# ---------------------------------------------------------------- mixes

def resolve_mix(spec, cat):
    """{glob: weight|'natural'} -> {component: fraction}, fractions sum to 1."""
    weights = {}
    for pattern, w in spec.items():
        matched = [c for c in cat if fnmatch.fnmatchcase(c, pattern)]
        if not matched:
            raise ValueError(f"mix pattern {pattern!r} matches no component")
        if w == "natural":
            for c in matched:
                weights[c] = cat[c]["tokens"]
        else:
            # an explicit weight is split across the matched components by natural size
            total = sum(cat[c]["tokens"] for c in matched)
            for c in matched:
                weights[c] = float(w) * cat[c]["tokens"] / total
    z = sum(weights.values())
    return {c: w / z for c, w in sorted(weights.items()) if w > 0}


def expand_mixes(cfg, cat):
    """Named mixes from cfg['mixes'] plus generated experiments. Returns {name: fractions}."""
    mixes = {name: resolve_mix(spec, cat) for name, spec in cfg["mixes"].items()}
    exp = cfg.get("experiments") or {}

    def match(patterns):
        return [c for c in cat if any(fnmatch.fnmatchcase(c, p) for p in patterns)]

    if exp.get("leave_one_out"):
        base = mixes[exp["leave_one_out"]]
        for comp in base:
            rest = {c: f for c, f in base.items() if c != comp}
            z = sum(rest.values())
            mixes[f"loo:{comp}"] = {c: f / z for c, f in rest.items()}
    if exp.get("add"):
        a = exp["add"]
        base, frac = mixes[a["base"]], a["fraction"]
        for comp in match(a["candidates"]):
            m = {c: f * (1 - frac) for c, f in base.items()}
            m[comp] = m.get(comp, 0) + frac
            mixes[f"add:{comp}"] = m
    for comp in match(exp.get("alone") or []):
        mixes[f"alone:{comp}"] = {comp: 1.0}
    return mixes


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


# ---------------------------------------------------------------- tokenisation

def _render(messages, system_prompt):
    """ChatML text plus char spans that are trained on (assistant content + <|im_end|>)."""
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + list(messages)
    text, spans = "", []
    for m in messages:
        text += f"<|im_start|>{m['role']}\n"
        start = len(text)
        text += m["content"] + "<|im_end|>"
        if m["role"] == "assistant":
            spans.append((start, len(text)))
        text += "\n"
    return text, spans


def _render_tokenize(tok, cfg, batch):
    """Returns (ids_list, mask_list) where mask marks tokens that get loss."""
    rendered = [_render(m, cfg["system_prompt"]) for m in batch]
    enc = tok([t for t, _ in rendered], add_special_tokens=False, return_offsets_mapping=True)
    ids_out, mask_out = [], []
    for ids, offs, (_, spans) in zip(enc["input_ids"], enc["offset_mapping"], rendered):
        mask = [any(s <= a < e for s, e in spans) for a, _ in offs]
        ids_out.append(ids)
        mask_out.append(mask)
    return ids_out, mask_out


# ---------------------------------------------------------------- build

def build(cfg, fractions, tok, decontam=None):
    """Return dict(train_ids, train_mask, val_ids, val_mask [uint16/bool, shape (n, seq)], meta).

    Cached by everything that affects the tokens, so an LR sweep builds once.
    """
    seq = cfg["seq_len"]
    key = fingerprint([fractions, cfg["model"], seq, cfg["token_budget"], cfg["val_tokens"], cfg["data_seed"],
                 cfg["system_prompt"], cfg["sources"], bool(decontam), cfg.get("decontam", {})])
    path = CACHE / "data" / f"{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in ("train_ids", "train_mask", "val_ids", "val_mask")} | {
            "meta": json.loads(str(z["meta"]))}

    srcs = {s["name"]: s for s in cfg["sources"]}
    train_ex, val_ex, meta = [], [], {"components": {}}
    for comp, frac in fractions.items():
        src_name, _, value = comp.partition("/")
        src = srcs[src_name]
        idx = list(_component_indices(src)[value or None])
        random.Random(f"{cfg['data_seed']}:{comp}").shuffle(idx)
        q_train, q_val = frac * cfg["token_budget"], frac * cfg["val_tokens"]
        tr, va, got_tr, got_va, n_contam, n_long, used = [], [], 0, 0, 0, 0, 0
        ds = _source_ds(src)
        for start in range(0, len(idx), 1000):
            chunk = idx[start:start + 1000]
            msgs = [to_messages(src, r) for r in rows(ds, chunk)]
            if decontam:
                keep = [not decontam.hit(m) for m in msgs]
                n_contam += keep.count(False)
                msgs = [m for m, k in zip(msgs, keep) if k]
            for ids, mask in zip(*_render_tokenize(tok, cfg, msgs)):
                used += 1
                if len(ids) > seq:
                    n_long += 1
                    continue
                if got_va < q_val:
                    va.append((ids, mask)); got_va += len(ids)
                elif got_tr < q_train:
                    tr.append((ids, mask)); got_tr += len(ids)
            if got_tr >= q_train:
                break
        if not tr:
            raise ValueError(f"{comp}: no usable training examples")
        unique = got_tr
        while got_tr < q_train:  # component smaller than its quota: repeat it
            for ex in tr[:]:
                if got_tr >= q_train:
                    break
                tr.append(ex); got_tr += len(ex[0])
        train_ex += tr
        val_ex += va
        meta["components"][comp] = {
            "fraction": frac, "train_tokens": got_tr, "unique_train_tokens": unique,
            "repeats": got_tr / unique, "examples_seen": used,
            "dropped_contaminated": n_contam, "dropped_too_long": n_long}

    rng = random.Random(cfg["data_seed"])
    rng.shuffle(train_ex)
    rng.shuffle(val_ex)
    n_train = cfg["token_budget"] // seq
    n_val = max(1, cfg["val_tokens"] // seq)
    t_ids, t_mask = _pack(train_ex, seq, n_train)
    v_ids, v_mask = _pack(val_ex, seq, n_val)
    meta["train_tokens"] = int(t_ids.size)
    meta["loss_tokens"] = int(t_mask.sum())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_ids=t_ids, train_mask=t_mask, val_ids=v_ids, val_mask=v_mask,
             meta=json.dumps(meta))
    return {"train_ids": t_ids, "train_mask": t_mask, "val_ids": v_ids, "val_mask": v_mask, "meta": meta}


def _pack(examples, seq, n_blocks):
    """Concatenate whole examples into seq-length blocks (last partial block dropped).

    If the examples fall a little short of n_blocks (rounding), they are cycled.
    """
    ids, mask = [], []
    for i, m in examples:
        ids.extend(i); mask.extend(m)
    need = n_blocks * seq
    while len(ids) < need:
        ids += ids[:need - len(ids)]; mask += mask[:need - len(mask)]
    ids = np.array(ids[:need], dtype=np.uint16).reshape(n_blocks, seq)
    mask = np.array(mask[:need], dtype=bool).reshape(n_blocks, seq)
    return ids, mask


def fingerprint(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
