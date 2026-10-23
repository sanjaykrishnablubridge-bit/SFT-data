#!/usr/bin/env bash
# One-shot setup for a fresh GPU instance. Safe to re-run.
#
#   bash setup.sh
#
# Optional env vars:
#   REPO_URL   git URL of this project; cloned/pulled into PROJECT_DIR if set
#   PROJECT_DIR  where the project lives (default: directory of this script)
#   HF_TOKEN   Hugging Face token (needed for gated/private datasets, or to push results)
#   GIT_NAME / GIT_EMAIL   git identity to configure
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

log "System packages"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
need=()
for p in git git-lfs tmux htop curl; do command -v "$p" >/dev/null || need+=("$p"); done
if [ ${#need[@]} -gt 0 ]; then
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${need[@]}"
fi
git lfs install --skip-repo >/dev/null

if [ -n "${GIT_NAME:-}" ]; then git config --global user.name "$GIT_NAME"; fi
if [ -n "${GIT_EMAIL:-}" ]; then git config --global user.email "$GIT_EMAIL"; fi

if [ -n "${REPO_URL:-}" ]; then
  log "Project code"
  if [ -d "$PROJECT_DIR/.git" ]; then
    git -C "$PROJECT_DIR" pull --ff-only
  else
    git clone "$REPO_URL" "$PROJECT_DIR"
  fi
fi
cd "$PROJECT_DIR"

log "uv + Python venv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv .venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

log "Python packages"
if [ -f requirements.lock ]; then
  uv pip install -r requirements.lock
else
  uv pip install -r requirements.txt
  uv pip freeze > requirements.lock
  echo "Wrote requirements.lock: commit it so every instance uses the same versions."
fi

if [ -n "${HF_TOKEN:-}" ]; then
  log "Hugging Face login"
  hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || huggingface-cli login --token "$HF_TOKEN"
fi

log "Shell config"
marker="# >>> smollm-testing >>>"
if ! grep -qF "$marker" ~/.bashrc; then
  cat >> ~/.bashrc <<EOF
$marker
export PATH="\$HOME/.local/bin:/usr/local/cuda/bin:\$PATH"
export HF_HOME="$HF_HOME"
[ -f "$PROJECT_DIR/.venv/bin/activate" ] && source "$PROJECT_DIR/.venv/bin/activate"
# <<< smollm-testing <<<
EOF
fi

log "Sanity check + model prefetch"
python - <<'EOF'
import torch, transformers, datasets, yaml
assert torch.cuda.is_available(), "CUDA not visible to torch"
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
print(f"transformers {transformers.__version__} | datasets {datasets.__version__}")
cfg = yaml.safe_load(open("configs/experiment.yaml"))
from huggingface_hub import snapshot_download
for m in dict.fromkeys([cfg["model"]] + [r["model"] for r in cfg["references"]]):
    snapshot_download(m)
    print(f"prefetched {m}")
import nltk; nltk.download("punkt_tab", quiet=True)  # needed by IFEval
EOF

log "Done. Open a new shell (or: source .venv/bin/activate), then:"
echo "  python run.py --list     # components and mixes"
echo "  python run.py --smoke    # quick end-to-end check"
echo "  python run.py            # full run (resumable); see README.md"
