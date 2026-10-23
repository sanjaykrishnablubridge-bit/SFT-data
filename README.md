# SFT data-mix search for a 135M model

Goal: find the open-source SFT data mix that lets our 135M model beat
**SmolLM2-135M-Instruct**. While our base model pretrains, `SmolLM2-135M`
(base) stands in for it; the final mix gets confirmed on our own base.

## New instance

```bash
git clone <this repo> ~/sanjay && cd ~/sanjay
HF_TOKEN=... bash setup.sh          # tools, venv (pinned by requirements.lock), models, GPU check
source .venv/bin/activate
python run.py --list                # every data component and every planned mix
```

Copy `runs/` off the instance before it goes away (it's git-ignored; it is the results).

## How an experiment works

1. **Sources → components.** Each dataset in `configs/experiment.yaml` is split by its
   `source` column into components (`smolsmol/smol-magpie-ultra-short`, `tulu3/ai2-adapt-dev/...`).
2. **Mixes.** A mix is a set of component weights. The **control** is `smol-smoltalk` at its
   natural proportions: the SFT data SmolLM2-135M-Instruct was actually trained on.
   Generated experiments:
   - `loo:<c>`: control without component `c` → is `c` pulling its weight?
   - `add:<c>`: 20% of control's tokens replaced by candidate `c` → does `c` improve the mix?
   - `alone:<c>`: `c` by itself (screening only; narrow sets look bad here by design).
3. **Each mix is trained identically:** same token budget, same recipe, loss on assistant
   tokens only, SmolLM2-Instruct chat template. Examples sharing a 13-gram with any eval
   set are dropped. Each mix gets its own LR sweep; the best LR is picked by that mix's
   held-out val loss (never by benchmarks), then re-trained with several seeds.
4. **Eval** with lm-eval: IFEval and GSM8K through the chat template, plus knowledge
   tasks (HellaSwag, ARC, PIQA, Winogrande, OBQA) to catch SFT damage. The base model and
   SmolLM2-135M-Instruct are scored with the same harness as references.
5. **`runs/report.md`**: every mix vs control (▲/▼ only when the difference beats 2× the
   noise), vs the target, LR choices, and a data audit (equal tokens, contamination
   dropped, repeated data).

## Running

```bash
python run.py --smoke                        # tiny end-to-end check
python run.py --mixes control                # control (+ references) only
python run.py --mixes 'loo:*'                # which control components matter
python run.py --mixes 'add:tulu3/*'          # which tulu-3 components help
```

Everything is resumable: finished trainings/evals are skipped, so after an instance
dies just re-run the same command (with `runs/` restored).

Cost on one RTX 6000 Ada (measured): training ~40k tok/s, so ~21 min per 50M-token run;
full eval ~20 min per model. With the defaults (4-LR sweep + 2 seeds per mix) that's
~2.7 h per mix. For screening many candidates use `lr_policy: control` and
`final_seeds: [0]` (~40 min per mix), then re-run the promising ones with full settings.

Cost knobs: `token_budget`, `lr_policy`, `final_seeds`. Add candidates by appending to `sources:`, then reference them in
`mixes:` or `experiments:`.

## Suggested order

1. `--mixes control`: check the references reproduce sensibly and see how close
   control-SFT gets to SmolLM2-135M-Instruct (the gap left is roughly DPO + full budget).
2. `loo:*`: find control components that don't help.
3. `add:*` over candidate sources: find components that do.
4. Combine winners into a named mix in `mixes:`, compare to control with more seeds.
5. Re-run the winner and control on our own base (`model:`), then add DPO.

Caveats: numbers differ from the SmolLM2 paper (it used lighteval), so compare
only within this harness; and the target has DPO, these runs don't.
