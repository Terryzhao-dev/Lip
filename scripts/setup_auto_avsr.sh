#!/usr/bin/env bash
# Set up the Auto-AVSR (mpc001/auto_avsr) backend for lipsync-text.
#
# Mirrors setup_usr2.sh's shape. This script:
#   1. ffmpeg sanity check
#   2. Clones https://github.com/mpc001/auto_avsr to third_party/auto_avsr
#   3. Installs their requirements.txt
#   4. Creates the checkpoints/ directory and prints next-step download instructions
#
# Note on PyTorch: assumes a working PyTorch in the active venv (installed by
# setup_usr2.sh or manually). Auto-AVSR shares the same torch/torchaudio
# dependencies, so we don't reinstall here.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
REPO_DIR="$THIRD_PARTY/auto_avsr"
CKPT_DIR="$ROOT/checkpoints"
CKPT_PATH="$CKPT_DIR/auto_avsr.pth"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
warn()  { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
error() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; }

bold "[1/4] Checking ffmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  error "ffmpeg not found. Install: brew install ffmpeg (macOS) / apt-get install ffmpeg (Linux)"
  exit 1
fi
echo "  ok: $(ffmpeg -version | head -1)"

bold "[2/4] Cloning Auto-AVSR to $REPO_DIR"
mkdir -p "$THIRD_PARTY"
if [ -d "$REPO_DIR/.git" ]; then
  echo "  already cloned — fetching latest"
  git -C "$REPO_DIR" fetch --quiet origin
  git -C "$REPO_DIR" reset --quiet --hard origin/HEAD
else
  git clone --depth 1 https://github.com/mpc001/auto_avsr "$REPO_DIR"
fi
if [ ! -f "$REPO_DIR/demo.py" ]; then
  error "demo.py is missing from the clone at $REPO_DIR. Aborting."
  exit 1
fi

bold "[3/4] Installing Auto-AVSR requirements"
if [ -f "$REPO_DIR/requirements.txt" ]; then
  python3 -m pip install -r "$REPO_DIR/requirements.txt"
else
  warn "no requirements.txt found in $REPO_DIR — install manually if demo.py errors on import"
fi

bold "[4/4] Preparing checkpoints directory"
mkdir -p "$CKPT_DIR"
if [ -f "$CKPT_PATH" ]; then
  echo "  checkpoint already present: $CKPT_PATH"
else
  cat <<EOF

Next step — download an Auto-AVSR checkpoint manually.

  Source of truth: https://github.com/mpc001/auto_avsr (README → Models table).
  The checkpoint table holds the current download links; direct URLs rotate.

  Save / rename the downloaded .pth to:
        $CKPT_PATH
     (or set LIPSYNC_AUTO_AVSR_CHECKPOINT=/path/to/your.pth)

Then run:
  LIPSYNC_BACKEND=auto_avsr python scripts/run_inference.py path/to/sample.mp4

EOF
fi

bold "Setup complete."
