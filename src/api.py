"""FastAPI app for cloud-hosted lip-to-text inference + correction logging.

Endpoints
---------
POST /transcribe-video
    multipart/form-data:
        file        — uploaded video (required)
        model       — registry name; default "auto_avsr" (overridable)
        mode        — "vsr" | "avsr" | "asr" (also "v"/"a"/"av")
    JSON response:
        raw_transcript, model_name, checkpoint_name, mode, runtime_ms,
        mouth_crop_path (nullable), error (nullable)

POST /log-correction
    JSON body matching CorrectionPayload (see schema below). Appends a
    gold-quality reviewed Entry to data/manifest.jsonl.

GET /healthz
    Liveness probe; returns {"ok": true}.

GET /models
    Lists registry names.

Local test:
    pip install fastapi 'uvicorn[standard]' python-multipart
    python scripts/serve.py
    curl -F file=@/tmp/me_5s.mp4 -F model=usr2 -F mode=avsr http://localhost:8000/transcribe-video
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import avsr
import config
import dataset

app = FastAPI(title="lipsync-text", version="0.2.0")


# ---- request / response models ---------------------------------------------

class TranscribeResponse(BaseModel):
    raw_transcript: str | None
    model_name: str
    checkpoint_name: str | None
    mode: str
    runtime_ms: int | None
    mouth_crop_path: str | None
    error: str | None = None


class CorrectionPayload(BaseModel):
    """Schema for POST /log-correction.

    Mirrors the user's correction-logging contract:
      - original video path
      - mouth crop path
      - raw prediction
      - corrected transcript
      - model / checkpoint
      - mode
      - fps / resolution
      - timestamp (server-assigned via dataset.append)
    """
    video_path: str = Field(..., description="Absolute or data/-relative path to the raw clip")
    mouth_crop_path: str | None = None
    raw_prediction: str | None = None
    corrected_transcript: str
    model_name: str
    checkpoint_name: str | None = None
    mode: str
    fps: int | None = None
    resolution: str | None = None
    speaker_id: str | None = None
    language: str | None = "en"


class CorrectionResponse(BaseModel):
    session_id: str
    ok: bool


# ---- endpoints --------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/models")
def models() -> dict[str, Any]:
    return {"available": avsr.available_models(), "default": config.AVSR_BACKEND}


@app.post("/transcribe-video", response_model=TranscribeResponse)
async def transcribe_video(
    file: UploadFile = File(...),
    model: str = Form(default=config.AVSR_BACKEND),
    mode: str = Form(default=config.AVSR_MODALITY),
) -> TranscribeResponse:
    if model not in avsr.available_models():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}. Pick one of {avsr.available_models()}.",
        )

    # Persist the upload to a temp file so backends (subprocess-based)
    # can read it from disk.
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        clip_path = Path(tmp.name)

    try:
        t0 = time.monotonic()
        result = avsr.transcribe(clip_path, mode=mode, model=model)
        # Backend should set runtime_ms; fall back to wall-clock if it didn't.
        runtime_ms = result.runtime_ms or int((time.monotonic() - t0) * 1000)
        return TranscribeResponse(
            raw_transcript=result.text,
            model_name=result.model_name,
            checkpoint_name=result.checkpoint_name,
            mode=result.mode,
            runtime_ms=runtime_ms,
            mouth_crop_path=result.mouth_crop_path,
            error=None,
        )
    except avsr.NotInstalled as e:
        return TranscribeResponse(
            raw_transcript=None,
            model_name=model,
            checkpoint_name=None,
            mode=mode,
            runtime_ms=None,
            mouth_crop_path=None,
            error=f"not_installed: {e}",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        return TranscribeResponse(
            raw_transcript=None,
            model_name=model,
            checkpoint_name=None,
            mode=mode,
            runtime_ms=None,
            mouth_crop_path=None,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        clip_path.unlink(missing_ok=True)


@app.post("/log-correction", response_model=CorrectionResponse)
def log_correction(payload: CorrectionPayload) -> CorrectionResponse:
    session_id = dataset.new_session_id()
    entry = dataset.Entry(
        session_id=session_id,
        speaker_id=payload.speaker_id or config.SPEAKER_ID,
        raw_clip_path=payload.video_path,
        mouth_roi_path=payload.mouth_crop_path,
        fps=payload.fps,
        resolution=payload.resolution,
        audio_present=(payload.mode in ("avsr", "av", "audiovisual", "audio-visual")),
        language=payload.language or "en",
        pseudo_transcript=payload.raw_prediction,
        pseudo_label_model=payload.model_name,
        pseudo_label_checkpoint=payload.checkpoint_name,
        mode=payload.mode,
        transcript_human=payload.corrected_transcript,
        review_status="reviewed",
        label_quality="gold",
    )
    dataset.append(entry)
    return CorrectionResponse(session_id=session_id, ok=True)
