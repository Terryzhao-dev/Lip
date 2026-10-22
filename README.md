# Lip

Lip reading on the desktop. Press ⌘⇧L, mouth a sentence at the webcam, and the text lands on your clipboard. Recognition is done by USR 2.0 (Haliassos et al., ICLR 2026), a pretrained visual speech recognition model. LLMs only come in optionally, to clean up punctuation.

Longer term I want to fine-tune it on my own recordings, which is what the data collection mode is for. Auto-AVSR also works as a backend (through its `eval.py`); AV-HuBERT is stubbed but not wired up yet.

Note: the USR 2.0 checkpoint links change, so I don't hardcode them. Get the current one from the "Pretrained Models" table in the [ahaliassos/usr2](https://github.com/ahaliassos/usr2) README.

## Quick start

```bash
# 1. Project setup
git clone https://github.com/Terryzhao-dev/Lip.git && cd Lip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg                          # webcam capture + USR2 video decode

# 2. Install USR 2.0 backend (clones repo + installs PyTorch + deps)
bash scripts/setup_usr2.sh

# 3. Download a checkpoint manually from the USR2 GitHub README's checkpoint table.
#
#    The official table is the source of truth; treat any hardcoded checkpoint
#    URL in this repo as potentially stale. Authors rotate Drive/HF links.
#
#    Steps:
#      a. Open https://github.com/ahaliassos/usr2
#      b. Scroll to:  Pretrained Models → Fine-tuned (full model) → High-resource
#      c. Click the download link for:  Base+ LRS3+Vox2   (recommended first run)
#         (or "Huge LRS2+LRS3+Vox2+AVS" if you want max accuracy and don't mind
#          a multi-GB download)
#      d. Save / rename the downloaded .pth to:
#           checkpoints/usr2_base_plus.pth
#         OR export LIPSYNC_CHECKPOINT=/path/to/your.pth to point elsewhere.

# 4. Real inference on one video
LIPSYNC_BACKEND=usr2 \
  LIPSYNC_CHECKPOINT=checkpoints/usr2_base_plus.pth \
  python scripts/run_inference.py path/to/sample.mp4

# 5. Live hotkey app
python src/main.py
# Press ⌘⇧L → mouth a phrase → press again (or wait 8s).
# Transcript lands on your clipboard + macOS notification.
```

**Smoke-test the wire-up without a checkpoint (mock backend):**

```bash
LIPSYNC_BACKEND=mock python scripts/run_inference.py any.mp4
```

Mock returns a recognizable placeholder string so you can confirm the rest of the app works before pulling down 1–4 GB of model weights.

## Cloud API (FastAPI)

```bash
# Install API deps (one-time)
pip install fastapi 'uvicorn[standard]' python-multipart pydantic

# Local server
python scripts/serve.py                 # 127.0.0.1:8000
python scripts/serve.py --host 0.0.0.0  # LAN / cloud
```

Endpoints:

| Method | Path | Notes |
|---|---|---|
| GET  | `/healthz` | liveness probe |
| GET  | `/models`  | `{available: [...], default: "usr2"}` |
| POST | `/transcribe-video` | `multipart/form-data` with `file`, `model`, `mode` |
| POST | `/log-correction`   | JSON body, appends a gold-quality reviewed Entry to `data/manifest.jsonl` |

Example:

```bash
curl -F file=@/tmp/me_5s.mp4 \
     -F model=usr2 -F mode=avsr \
     http://localhost:8000/transcribe-video | jq

# {
#   "raw_transcript": "I ALSO ALLOWED MY NAME SERILY",
#   "model_name": "usr2",
#   "checkpoint_name": "baseplus_low_resource_lrs3vox2.pth",
#   "mode": "av",
#   "runtime_ms": 20891,
#   "mouth_crop_path": null,
#   "error": null
# }

curl -X POST http://localhost:8000/log-correction \
     -H 'content-type: application/json' \
     -d '{
       "video_path": "data/clips/20260519T143300Z.mp4",
       "raw_prediction": "I ALSO ALLOWED MY NAME SERILY",
       "corrected_transcript": "I also allowed my name, surely.",
       "model_name": "usr2",
       "checkpoint_name": "baseplus_low_resource_lrs3vox2.pth",
       "mode": "avsr",
       "fps": 25, "resolution": "1280x720"
     }' | jq
```

## Programmatic interface

```python
from pathlib import Path
import avsr

result = avsr.transcribe(
    Path("clip.mp4"),
    mode="vsr",       # "vsr" / "avsr" / "asr"  (also "v"/"a"/"av")
    model="auto_avsr" # "usr2" / "auto_avsr" / "avhubert" / "mock"
)
print(result.text, result.model_name, result.checkpoint_name, result.runtime_ms)

# Registry
print(avsr.available_models())
```

## Modes

Mode is controlled by env vars (`src/config.py` is the source of truth).

| Mode | Set | Behavior |
|---|---|---|
| **Transcribe-only (default)** | nothing | Record → AVSR → display → discard clip. Nothing saved. |
| **Collect for fine-tuning** | `LIPSYNC_COLLECT=1` | Same flow, but after transcription a review dialog opens. Saving writes the clip and your corrected transcript to `data/manifest.jsonl`. See `TRAINING_PLAN.md`. |
| **LLM rescore** | `LIPSYNC_RESCORE=1` | AVSR text is post-processed by `rescore.py:polish()` (Gemini, punctuation cleanup only, never adds words). |

Modality:

| Set | Behavior |
|---|---|
| `LIPSYNC_MODALITY=av` (default) | audio + visual (best accuracy when audio is captured) |
| `LIPSYNC_MODALITY=v` | visual only, for silent mouthing |
| `LIPSYNC_MODALITY=a` | audio only |

Backend:

```bash
LIPSYNC_BACKEND=usr2         # default
LIPSYNC_BACKEND=mock         # placeholder, for testing without weights
LIPSYNC_BACKEND=auto_avsr    # needs the Auto-AVSR repo + checkpoint
LIPSYNC_BACKEND=avhubert     # skeleton (not yet wired up)
```

## Architecture

```
                 ┌──────────────────┐
                 │  recorder.py     │  ffmpeg AVFoundation, mp4 with audio
                 │  ⌘⇧L hotkey      │  (or scripts/run_inference.py video.mp4)
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐    src/config.py picks backend
                 │  avsr.py         │ ─▶ MockBackend / USR2Backend
                 │  Backend         │     / AutoAVSRBackend / AVHuBERTBackend
                 │  interface       │
                 └────────┬─────────┘
                          ▼
            ┌───────── result.text ─────────┐
            ▼                                ▼
   ┌──────────────────┐                ┌──────────────────┐
   │  rescore.py      │ (optional)     │  output.py        │
   │  LLM polish      │ ─────────────▶ │  clipboard + notif│
   └──────────────────┘                └──────────────────┘
                                              │
                                              ▼ (only if LIPSYNC_COLLECT=1)
                                       ┌──────────────────┐
                                       │  review.py +     │
                                       │  dataset.py      │
                                       │  manifest.jsonl  │
                                       └──────────────────┘
```

## macOS permissions

- **Camera, Microphone**: auto-prompt on first capture.
- **Accessibility**: required by `pynput` for the ⌘⇧L global hotkey. System Settings → Privacy & Security → Accessibility → add and restart your terminal/IDE.

To pick a specific audio device (e.g. AirPods):

```bash
ffmpeg -f avfoundation -list_devices true -i ""
export LIPSYNC_FFMPEG_AUDIO_DEV=<index>
```

## Switching backends

Auto-AVSR runs through its own `eval.py` CLI (`export LIPSYNC_BACKEND=auto_avsr`). AV-HuBERT is only scaffolded: implement `load()` and `transcribe()` in the relevant class in `src/avsr.py`, then `export LIPSYNC_BACKEND=avhubert`.

## Optional: fine-tuning on your own data

Data collection is **opt-in** now (`LIPSYNC_COLLECT=1`), not the primary workflow. When enabled, sessions flow through the manifest schema described below and you eventually fine-tune on top of USR2's checkpoint. Detailed plan in [TRAINING_PLAN.md](./TRAINING_PLAN.md).

Manifest layout (gitignored):

```
data/
├── clips/<session_id>.mp4         raw 640×480 video (+ audio)
├── rois/<session_id>.mp4          96×96 grayscale mouth crop
├── landmarks/<session_id>.npz     478-point face mesh per frame
├── audio/<session_id>.wav         16 kHz mono PCM
└── manifest.jsonl                 append-only, latest row per session_id wins
```

Pipeline commands when collecting:

```bash
LIPSYNC_COLLECT=1 python src/main.py   # record + review
python src/preprocess.py               # mouth ROI + landmarks + audio extract
python src/split.py --by session       # train/val/test (speaker-disjoint when multi-speaker)
python scripts/verify_pipeline.py      # integrity gate before any training
```

Schema is in `src/dataset.py:Entry`; full field list and validation rules described inline there.

## Role of Gemini / Grok / MiniMax

These are **support models**, not the recognition system. Only used for:
- Punctuation polish / capitalization (`rescore.py:polish`).
- N-best selection from beam-search output (`rescore.py:rescore_nbest`), constrained to picking one of the existing hypotheses, never free-text generation.
- Code generation / project automation during development.

If you fine-tune any LLM, it's only as a text post-processor on top of AVSR output. Never as the core recognition model.

## Limitations

- USR2 inference is invoked via subprocess against their `demo.py`. Adds 1–3 s of Python startup overhead per call. If we hit a latency wall, switch to in-process import.
- Mouth crop in `src/preprocess.py` doesn't do face alignment / rotation normalization, just a centre crop around the lip centroid. Good enough as a baseline.
- Single-speaker assumption everywhere. `speaker_id` field exists for future multi-speaker support but defaults to `"default"`.
- Hold-to-record / streaming inference / menu-bar app: not done yet.
