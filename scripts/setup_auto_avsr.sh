#!/usr/bin/env bash
# Set up the Auto-AVSR (mpc001/auto_avsr) backend for lipsync-text.
#
# Hard requirements:
#   - active Python venv (script aborts if VIRTUAL_ENV isn't set so we never
#     install into system Python by accident)
#   - ffmpeg on PATH (Auto-AVSR's video decode path uses it via PyAV)
#
# What this does, exactly:
#   1. Confirm an active venv
#   2. Confirm ffmpeg exists
#   3. Clone https://github.com/mpc001/auto_avsr to third_party/auto_avsr
#      (idempotent — fetches latest if already cloned)
#   4. Install the dependencies their README lists, into the active venv
#      — no version pins, so existing pins from setup_usr2.sh (torch 2.6,
#      torchvision 0.21, torchaudio 2.6) are preserved
#   5. Verify the clone landed eval.py and the key Python modules import
#   6. Make checkpoints/ and print the manual checkpoint-download step
#
# Re-running this script is safe.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
REPO_DIR="$THIRD_PARTY/auto_avsr"
CKPT_DIR="$ROOT/checkpoints"
CKPT_PATH="$CKPT_DIR/auto_avsr.pth"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()    { printf '\033[32m[ ok ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
error() { printf '\033[31m[err ]\033[0m %s\n' "$*" >&2; }

# ----------------------------------------------------------------------------
bold "[1/6] Confirming active virtual environment"
# ----------------------------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
  error "No active venv (VIRTUAL_ENV is unset)."
  error "Fix: source .venv/bin/activate   (creating one with python3.13 -m venv .venv first if needed)"
  exit 1
fi
ok "venv: $VIRTUAL_ENV"
ok "python: $(python -V 2>&1)   ($(which python))"
ok "pip:    $(pip -V 2>&1 | awk '{print $1, $2}')"

# ----------------------------------------------------------------------------
bold "[2/6] Confirming ffmpeg is on PATH"
# ----------------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  error "ffmpeg not found."
  error "Fix: brew install ffmpeg   (macOS) / sudo apt-get install ffmpeg   (Linux)"
  exit 1
fi
ok "$(ffmpeg -version 2>&1 | head -1)"

# ----------------------------------------------------------------------------
bold "[3/6] Cloning mpc001/auto_avsr into $REPO_DIR"
# ----------------------------------------------------------------------------
mkdir -p "$THIRD_PARTY"
if [ -d "$REPO_DIR/.git" ]; then
  echo "       already cloned — fetching latest"
  git -C "$REPO_DIR" fetch --quiet origin
  git -C "$REPO_DIR" reset --quiet --hard origin/HEAD
else
  git clone --depth 1 https://github.com/mpc001/auto_avsr "$REPO_DIR"
fi
if [ ! -f "$REPO_DIR/eval.py" ]; then
  error "eval.py not found in $REPO_DIR. The upstream repo layout may have changed."
  error "Inspect manually: ls $REPO_DIR"
  exit 1
fi
ok "$REPO_DIR/eval.py present"

# ----------------------------------------------------------------------------
bold "[4/6] Installing dependencies from Auto-AVSR README"
# ----------------------------------------------------------------------------
# Their README lists these inline — no requirements.txt is shipped.
# No version pins here on purpose: torch / torchvision / torchaudio are
# already pinned by setup_usr2.sh (2.6 / 0.21 / 2.6), and `pip install` without
# --upgrade leaves satisfied packages alone, so the USR2 setup keeps working.
#
# Concretely, only `sentencepiece` is typically new after setup_usr2.sh.
pip install --quiet \
  torch torchvision torchaudio \
  pytorch-lightning sentencepiece av
ok "dependencies installed"

# ----------------------------------------------------------------------------
bold "[5/6] Verifying the install"
# ----------------------------------------------------------------------------
python - <<'PY'
import importlib, sys
required = ["torch", "torchaudio", "torchvision", "pytorch_lightning",
            "sentencepiece", "av"]
missing = []
versions = []
for mod in required:
    try:
        m = importlib.import_module(mod)
        versions.append(f"{mod}={getattr(m, '__version__', '?')}")
    except ImportError as e:
        missing.append(f"{mod}: {e}")
if missing:
    print("FAILED imports:")
    for m in missing:
        print("  -", m)
    sys.exit(1)
print("imports OK —", ", ".join(versions))
PY
ok "all required modules importable"

# ----------------------------------------------------------------------------
bold "[6/6] Preparing checkpoints directory"
# ----------------------------------------------------------------------------
mkdir -p "$CKPT_DIR"
if [ -f "$CKPT_PATH" ]; then
  ok "checkpoint already present: $CKPT_PATH ($(du -h "$CKPT_PATH" | cut -f1))"
else
  cat <<EOF

Next manual step — download an Auto-AVSR VSR checkpoint.

  Source of truth: https://github.com/mpc001/auto_avsr (README → Model Zoo).
  Direct download URLs rotate; treat the README table as authoritative.

  Recommended starting checkpoint:
      vsr_trlrs2lrs3vox2avsp_base.pth   (VSR, 20.3% WER on LRS3 per README)

  Save / rename the downloaded .pth to:
      $CKPT_PATH
  (or set LIPSYNC_AUTO_AVSR_CHECKPOINT=/path/to/your.pth before running inference)

Then run real inference:
  LIPSYNC_BACKEND=auto_avsr \\
      python scripts/run_inference.py path/to/sample.mp4 --mode vsr

EOF
fi

bold "Setup complete."
