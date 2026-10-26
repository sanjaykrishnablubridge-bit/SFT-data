# SFT Data Size Decision: beating SmolLM2-135M

**Goal:** the best possible 135M instruct model, measured on the benchmarks in the SmolLM2 paper.
**Decision:** run a data-scaling sweep. Collect about 2× SmolLM2's SFT size, train at 0.25×, 0.5×, 1× and 2×, and let the results choose the size.

> **Baseline number.** SmolTalk is **1.1M samples** (instruction-response pairs), not 1.1M tokens (confirmed in Table 9 of the paper). For the 135M and 360M models, the SmolLM2 team used a filtered, simpler subset, `smol-smoltalk`, believed to be about 460K samples. Confirm that figure on the dataset card.
> Below, **N** means the sample count SmolLM2-135M was actually trained on.

---

## 1. Decision tree

The ✅ marks show the path we took.

```
ROOT: What is the SFT dataset for?
│
├─ A) Prove our data is better at the same size (controlled claim)
│     └─ Match N exactly.
│        Only isolates data quality if the base model is the same. Ours is a
│        different pretrained base, so this claim is muddied anyway.
│
└─ ✅ B) Ship the best 135M model and win most benchmarks
      │
      Q1: Is our base at least as strong as SmolLM2-135M base on knowledge
          benchmarks (MMLU, ARC, HellaSwag, PIQA)?
      │   These come from pretraining. SFT moves them by about ±1–2 points
      │   and often lowers them (alignment tax).
      │
      ├─ NO → Data size won't close the gap.
      │       Keep SFT moderate, aim it at benchmarks SFT does move
      │       (IFEval, MT-Bench, GSM8K, partly BBH), and flag the gap
      │       to the pretraining team.
      │
      └─ ✅ YES / CLOSE
            │
            Q2: Is SFT compute a binding constraint?
            │   At 135M, ~1M samples × ~1K tokens ≈ 1B tokens, a few GPU-hours.
            │
            ├─ YES (rare) → Match N; put everything into quality and mixture.
            │
            └─ ✅ NO
                  │
                  Q3: Does a 135M model keep gaining from more SFT data?
                  │   Unknown for our base. Small models saturate early and can
                  │   forget pretraining knowledge. SmolLM2 itself chose less,
                  │   simpler data for 135M.
                  │
                  └─ ✅ ACTION: run a data-scaling sweep
```

---

## 2. The sweep

1. Collect a pool of about **2N** samples. Decontaminate it against every eval set before any training.
2. Draw **nested** subsets from the pool, so each smaller run is contained in the larger one. Keep mixture proportions fixed across sizes.
3. Train each subset with the **same recipe** (LR, epochs, chat template). Evaluate with the **same harness, prompts and shot counts** as the paper.
4. Plot each benchmark against size and choose the size from the curve.

| Run | Size   | If N ≈ 460K | Why it's there                          |
|-----|--------|-------------|-----------------------------------------|
| S1  | 0.25 N | ≈ 115K      | Shows how early a 135M model saturates  |
| S2  | 0.5 N  | ≈ 230K      | Middle point for the curve shape        |
| S3  | 1 N    | ≈ 460K      | Same-size comparison with SmolLM2       |
| S4  | 2 N    | ≈ 920K      | Tests whether more data still helps     |

---

## 3. Reading the result

| What the curve shows                    | What we do                                                                 |
|-----------------------------------------|----------------------------------------------------------------------------|
| Still rising at 2N                      | Collect more data and extend the sweep.                                    |
| Flat after 1N                           | Stop adding volume. Spend effort on quality and mixture.                   |
| Knowledge scores fall as size grows     | Forgetting. Use less data, a lower LR or fewer epochs, or mix in some pretraining-style data. |

We report two results:
- **The 1N run**: the same-size comparison with SmolLM2.
- **The best run**: the headline model.

---

## 4. Levers that matter more than volume

- **Mixture per benchmark.** Constraint-following data for IFEval, short clean math chain-of-thought for GSM8K, good multi-turn chat for MT-Bench. Adjust the mix toward benchmarks we're losing before adding volume.
- **Difficulty matched to 135M.** Long, hard reasoning traces tend to hurt at this size.
- **Preference tuning.** SmolLM2 ran DPO after SFT. IFEval and MT-Bench gains often come from that stage.
- **Decontamination.** N-gram overlap filtering against every eval set, so the wins hold up.
- **Same eval harness.** SmolLM2 used lighteval. Harness, prompt or shot-count differences can move scores more than the data does.

---

## 5. Choosing the mixture ratios

There's no formula. Teams settle the mixture with **ablations**: start from a known-good mixture, change one thing at a time, and measure.

### Step 1: Start from a reference mixture

SmolTalk's composition (Table 9 of the SmolLM2 paper, 1.1M samples in total):

| Source | Samples | Share |
|---|---|---|
| MagPie-Ultra (general chat) | 431k | 39% |
| NuminaMath-CoT | 112k | 10% |
| Smol-Summarization | 101k | 9% |
| OpenHermes2.5 | 100k | 9% |
| APIGen-Function-Calling | 87.5k | 8% |
| Smol-Rewrite | 56.2k | 5% |
| Self-OSS-Starcoder2-Instruct | 50.7k | 4.6% |
| MetaMathQA | 50k | 4.6% |
| Smol-Constraints | 36.2k | 3.3% |
| SystemChats2.0 | 35.9k | 3.3% |
| Explore-Instruct-Rewriting | 32k | 2.9% |
| LongAlign + Everyday-Conversations | ~6k | 0.5% |

That mixture is for the 1.7B model. The 135M model was trained on the lighter subset (`smol-smoltalk`), which drops function calling, long-context data and the hardest MagPie-Ultra samples. **Use smol-smoltalk's composition as the starting point.** Load it from the Hugging Face hub and count samples per source.

### Step 2: Map benchmarks to data buckets

| Benchmark | Data bucket that moves it |
|---|---|
| IFEval | Constraint-following data (Smol-Constraints style) |
| MT-Bench | Multi-turn chat (MagPie, Everyday-Conversations) |
| GSM8K | Math chain-of-thought (MetaMathQA, simpler NuminaMath) |
| BBH | Reasoning and general instructions |
| MMLU / ARC / HellaSwag / PIQA | Mostly pretraining. Here the goal is to avoid hurting them |

Ablate at the bucket level (4–6 buckets), not the dataset level (13+ datasets). It needs far fewer runs.

### Step 3: Run ablations (cheap at 135M)

- **Add-one-in / leave-one-out.** Train on the reference mixture. Then double one bucket, or remove it, and retrain. Keep a change if it helps the target benchmarks without hurting the others. SmolLM2 built SmolTalk roughly this way.
- **Regression over random mixtures** (the RegMix idea). Train about 15–30 runs with randomly sampled bucket weights. Fit a regression of each benchmark score on the weights, then pick the weights that maximize the target.

### Step 4: Count tokens, not just samples

The training loss is spread over tokens, so a bucket's influence follows its **token** share. A math chain-of-thought sample can be 5× longer than a rewrite sample. Report both the sample share and the token share for every mixture.

### Order relative to the size sweep

1. **Settle the mixture first**, with ablations at a fixed size (for example 0.5N, to keep runs fast).
2. **Then run the size sweep** (0.25N → 2N) with that mixture held fixed.

If you change both at once, you can't tell which change moved the score.

---

## 6. Training pipeline: one mixed dataset, not dataset by dataset

Training on datasets in sequence (math, then chat, then constraints) causes catastrophic forgetting of the earlier ones. The standard approach, and what SmolLM2 did, is a **single SFT stage on one shuffled mixture**. Post-training stages are split by **training objective**, not by dataset:

```
Pretrained base
   └─► SFT    (one shuffled mixture of all datasets)
         └─► DPO   (preference pairs; SmolLM2 did this after SFT)
```

### SFT data pipeline

```
1. Load        each HF dataset
2. Normalize   convert every dataset to one schema:
               {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
3. Clean       drop empty or broken rows, filter by length, dedupe within and across sets
4. Decontam.   remove anything overlapping the eval benchmarks (n-gram match)
5. Weight      subsample large sets / upsample small ones to hit target ratios
               (cap repeats at ~2–3×, or the model memorizes them)
6. Merge       concatenate and shuffle into ONE dataset; keep a "source" column
7. Template    apply a chat template (SmolLM2 uses ChatML: <|im_start|>user ... <|im_end|>)
               The base model has no chat format yet; it is added here
8. Tokenize    and mask the loss so only assistant tokens are trained on
9. Pack/pad    packing several short samples per sequence speeds training a lot
10. Train      e.g. TRL's SFTTrainer
11. Eval       same harness and settings as the paper (lighteval)
```

Steps 5–6 with the `datasets` library (every dataset must already share the same columns from step 2):

```python
from datasets import concatenate_datasets

def take(ds, n, seed=0):
    ds = ds.shuffle(seed=seed)
    if n <= len(ds):
        return ds.select(range(n))
    reps = -(-n // len(ds))  # upsample by repeating, then trim
    return concatenate_datasets([ds] * reps).select(range(n))

mixture = concatenate_datasets([
    take(chat_ds,        int(0.40 * TOTAL)),
    take(math_ds,        int(0.15 * TOTAL)),
    take(constraints_ds, int(0.08 * TOTAL)),
    # ...
]).shuffle(seed=42)
```

### When staging is reasonable

A short final stage is sometimes used: for example, a last pass with a small amount of very high-quality data at a lower learning rate. Treat it as an experiment after you have a strong single-stage baseline, not as the main pipeline.

---

*SmolTalk's 1.1M total and per-source counts are from Table 9 of the SmolLM2 paper. The ~460K figure for smol-smoltalk is from memory; verify it on the dataset card before fixing N.*
