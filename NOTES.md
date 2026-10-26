# Working notes

Running log of state and decisions, committed so it survives instance changes
(GPU instances are throwaway; local Claude memory doesn't carry over).
Update it at each milestone. Newest entries first.

## Rules of thumb

- `runs/` is the results and is git-ignored. Copy it off before an instance dies;
  restore it on the next one and re-run the same command (finished work is skipped).
- Don't commit model weights to plain git. If a trained model needs to be kept,
  push it to the Hugging Face Hub (private repo) and load it from there.
- Weights are portable across GPU architectures. Only the environment
  (driver / CUDA / torch) and compiled extras are tied to the GPU.

## Log

### 2026-10-26: plan to build a ~1M-sample SFT pool (see idea.md)

- N (smol-smoltalk train) = 460,341 samples, confirmed on the HF card (Apache-2.0).
  2N ≈ 920K, which fits the ~1M pool target. tulu-3-sft-mixture: 939,343 (ODC-BY).
- Decided: the user supplies the list of HF datasets. Each dataset goes through
  normalize → clean → decontam in its own parallel worker. No LLM judge for now.
  Output is local parquet only (git-ignored), and copying it off the instance is manual.
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
- Nothing has been run yet: no `--list`, `--smoke` or experiments.
- Training is single-GPU (`train.py` uses `.cuda()`, i.e. GPU 0), so 3 GPUs sit idle.
  Option: run separate mixes in parallel with `CUDA_VISIBLE_DEVICES=N`. First check
  that concurrent runs don't collide in `runs/`.

## Next

1. `python run.py --list`, then `python run.py --smoke`.
2. `python run.py --mixes control` (see the README's suggested order).
