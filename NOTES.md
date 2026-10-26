# Working notes

Running log of state and decisions, committed so it survives instance changes
(GPU instances are throwaway; local Claude memory doesn't carry over).
Update it at each milestone. Newest entries first.

## Current status (2026-10-26)

- **Done:** environment set up; `run.py` runs on all GPUs; first experiment
  (control vs tulu3 vs half_half) finished. Results are in the log below.
- **Not kept:** trained model weights. `run.py` deletes each model after eval unless
  `--keep-models` is passed. To get one back, restore `runs/` and re-run that mix with
  `--keep-models` (only its final training re-runs, ~11 min per seed).
- **Copy off before the instance dies:** `runs/` (1.3 MB). Everything else rebuilds:
  `cache/` (tokenized data, ~500 MB), `.venv/` (setup.sh), the HF cache, `runs_smoke/`.
- **Waiting on the user:** the list of HF datasets for the ~1M-sample SFT pool, plus
  answers on the cleaning rules (see "Open questions").
- **Resume on a new instance:** clone → `bash setup.sh` → restore `runs/` → `source .venv/bin/activate`.

## Rules of thumb

- `runs/` is the results and is git-ignored. Copy it off before an instance dies;
  restore it on the next one and re-run the same command (finished work is skipped).
- Don't commit model weights to plain git. If a trained model needs to be kept,
  push it to the Hugging Face Hub (private repo) and load it from there.
- Weights are portable across GPU architectures. Only the environment
  (driver / CUDA / torch) and compiled extras are tied to the GPU.

## Log

### 2026-10-26: multi-GPU runs; first real experiment

- `run.py` now runs independent jobs (reference evals, mixes, LR-sweep points, seeds)
  concurrently, one job per GPU (`--gpus 0,1,2,3`; default is all visible). Each job's
  output goes to `train.log` / `lm_eval*.log` next to its results. File locks in
  `data.py` / `decontam.py` make concurrent jobs build each cache once.
- The smoke test passed on 4 GPUs: sweep → final → save → chat and non-chat eval all worked.
  Measured ~76k tok/s per GPU on Blackwell (vs ~40k on the RTX 6000 Ada), so a
  50M-token run takes ~11 min.
- Launched in tmux session `train`: `python run.py --mixes control,tulu3,half_half`
  with full settings (4-LR sweep, 2 seeds). Log: `runs/run.log`, report: `runs/report.md`.
  Question it answers: does swapping or adding tulu-3 beat smol-smoltalk (control)?
- **Result** (finished ~12:50 UTC, ~70 min wall; full table in `runs/report.md`, not in git):

  | | IFEval | GSM8K | knowledge avg | best LR |
  |---|---|---|---|---|
  | base | – | – | 0.476 | |
  | SmolLM2-135M-Instruct (target, SFT+DPO) | 0.220 | 0.020 | 0.452 | |
  | control (smol-smoltalk) | 0.176 | 0.018 | 0.449 | 1e-3 (grid edge) |
  | tulu3 | 0.172 | 0.017 | 0.462 | 3e-4 |
  | half_half | 0.164 | 0.022 | 0.468 | 3e-4 |

  - IFEval and GSM8K: no significant difference between mixes. GSM8K is ~2% for every
    model (floor), even with tulu-3's 50k GSM8K-style math, so it can't tell mixes apart.
  - Knowledge: tulu3 and half_half lose less to SFT (HellaSwag +0.025 ▲ for both; ARC-Easy
    +0.030 ▲ for half_half). **Confound:** control trained at LR 1e-3 vs 3e-4 for the
    others, and a higher LR alone can cause more forgetting. Re-run control at 3e-4 before
    crediting the data.
  - Gap to target on IFEval (0.176 vs 0.220) is plausibly DPO.

### 2026-10-26: plan to build a ~1M-sample SFT pool (see idea.md)

- N (smol-smoltalk train) = 460,341 samples, confirmed on the HF card (Apache-2.0).
  2N ≈ 920K, which fits the ~1M pool target. tulu-3-sft-mixture: 939,343 (ODC-BY).
- Decided: the user supplies the list of HF datasets. Each dataset goes through
  normalize → clean → decontam in its own parallel worker. No LLM judge for now.
  Output is local parquet only (git-ignored), and copying it off the instance is manual.
- ~1M is the size **after** cleaning. Mixture ratios are set later, using SmolTalk
  (Table 9, 1.1M, built for the 1.7B model) or smol-smoltalk as the reference.
- smol-smoltalk's actual composition, counted from the HF train split:

  | source | samples | share |
  |---|---|---|
  | smol-magpie-ultra-short | 270,838 | 58.8% |
  | self-oss-instruct | 48,071 | 10.4% |
  | openhermes-50k | 47,492 | 10.3% |
  | smol-contraints | 34,433 | 7.5% |
  | smollm-rewrite-30k | 26,657 | 5.8% |
  | smol-summarize-20k | 19,272 | 4.2% |
  | smol-summarize-5k | 4,749 | 1.0% |
  | longalign | 3,560 | 0.8% |
  | explore-instruct-rewrite | 3,017 | 0.7% |
  | everyday-conversations | 2,252 | 0.5% |

  It has **no math** (no NuminaMath or MetaMathQA), no function calling and no SystemChats.
  So the GSM8K bucket has no smol-smoltalk reference ratio; take it from SmolTalk instead.
- Note: these steps are CPU work (120 cores, 354 GB RAM here). The GPUs only help
  once model-based steps are added.

### 2026-10-26: new instance, environment set up

- Instance: 4× NVIDIA RTX PRO 6000 Blackwell (96 GB each, sm_120), driver 580.178.04
  (CUDA 13.0). README timings were measured on one RTX 6000 Ada, so expect different speeds.
- `bash setup.sh` ran clean: `.venv` (Python 3.12) from `requirements.lock` with
  torch 2.14.1+cu130, transformers 5.20.2, datasets 5.1.0, lm-eval 0.4.13.
  SmolLM2-135M and SmolLM2-135M-Instruct are prefetched. No `HF_TOKEN` set (both
  datasets are public; a token is needed only to push to the Hub).
- Checked: torch sees all 4 GPUs and runs bf16 on Blackwell.
- Training was single-GPU (`train.py` uses `.cuda()`); fixed later the same day by
  scheduling jobs across GPUs in `run.py` (see above).

## Open questions (data pool)

- Dataset list: the user will provide it.
- Cleaning rules: length cap (8192 vs tighter, e.g. 2048), English only?, keep multi-turn
  and system prompts?, drop boilerplate refusals?
- Dedup: exact + MinHash within each dataset, then one global cross-dataset pass. OK?
- Decontam: also add MT-Bench and BBH to `decontam.py`'s eval sets?
- Output: one parquet folder per dataset under `data/`, schema `{messages, source, id}`,
  plus a per-dataset drop report. Any extra columns (bucket label, original id)?

## Next

1. Re-run the control at LR 3e-4 to remove the LR confound in the knowledge result
   (2 runs, ~25 min on 4 GPUs).
2. Build the SFT pool once the dataset list arrives (CPU workers, no GPU needed).
3. README order: `loo:*` (which smol-smoltalk parts matter), then `add:*` candidates.
4. Later: DPO stage. The IFEval gap to the target (0.176 vs 0.220) is plausibly DPO.
