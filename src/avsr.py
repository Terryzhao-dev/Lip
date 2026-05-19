"""Visual / audio-visual speech recognition backend interface.

Common contract:
    from avsr import transcribe
    result = transcribe(Path("clip.mp4"), mode="vsr", model="auto_avsr")
    print(result.text, result.model_name, result.checkpoint_name)

Selection precedence:
    explicit `model=` kwarg → LIPSYNC_BACKEND env var → "usr2" default.
    explicit `mode=` kwarg  → LIPSYNC_MODALITY env var → "av" default.

Model registry (see _BACKENDS):
    usr2       — USR 2.0 (working, real subprocess against demo.py)
    auto_avsr  — Auto-AVSR (real subprocess wiring; needs repo + checkpoint installed)
    avhubert   — AV-HuBERT (research placeholder; raises NotInstalled until wired)
    mock       — returns a placeholder string; for smoke tests
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import config

# Internal canonical modes; the API/CLI also accept the friendlier aliases below.
Mode = Literal["v", "a", "av"]
_MODE_ALIASES: dict[str, Mode] = {
    "v": "v", "visual": "v", "vsr": "v",
    "a": "a", "audio": "a", "asr": "a",
    "av": "av", "audiovisual": "av", "audio-visual": "av", "avsr": "av",
}


def normalize_mode(mode: str | None) -> Mode:
    if mode is None:
        return _MODE_ALIASES.get(config.AVSR_MODALITY, "av")
    key = mode.lower()
    if key not in _MODE_ALIASES:
        raise ValueError(f"Unknown mode {mode!r}. Allowed: {sorted(set(_MODE_ALIASES))}")
    return _MODE_ALIASES[key]


@dataclass
class AVSRResult:
    """Common return shape across all backends."""
    text: str
    model_name: str
    checkpoint_name: str | None = None
    mode: Mode = "av"
    runtime_ms: int | None = None
    mouth_crop_path: str | None = None
    confidence: float | None = None
    error: str | None = None


class NotInstalled(RuntimeError):
    """Raised when the configured backend is selected but its dependencies are missing."""


class Backend(Protocol):
    name: str
    def load(self) -> None: ...
    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult: ...


# ---- Mock backend -----------------------------------------------------------

class MockBackend:
    """Returns a placeholder string. For testing the rest of the app without
    needing a multi-GB checkpoint installed."""

    name = "mock"

    def load(self) -> None:
        pass

    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult:
        return AVSRResult(
            text=f"[mock backend — set LIPSYNC_BACKEND=usr2 (or auto_avsr) and "
                 f"install the checkpoint for real transcription. video={video_path.name}]",
            model_name="mock",
            checkpoint_name="mock-v0",
            mode=mode,
        )


# ---- USR 2.0 backend (subprocess against their demo.py) --------------------

class USR2Backend:
    """USR 2.0 (Haliassos et al., ICLR 2026). Working.

    Setup: bash scripts/setup_usr2.sh
    """

    name = "usr2"
    _RESULT_RE = re.compile(r"^\s*Result\s*:\s*(.+?)\s*$", re.MULTILINE)

    def load(self) -> None:
        repo = config.USR2_REPO_DIR
        if not repo.exists():
            raise NotInstalled(
                f"USR2 repo not found at {repo}.\nFix: bash scripts/setup_usr2.sh"
            )
        if not (repo / "demo.py").exists():
            raise NotInstalled(
                f"USR2 demo.py missing at {repo}/demo.py.\n"
                f"Fix: rm -rf {repo} && bash scripts/setup_usr2.sh"
            )
        ckpt = config.AVSR_CHECKPOINT_PATH
        if ckpt is None:
            raise NotInstalled(
                "No checkpoint configured. Set LIPSYNC_CHECKPOINT=/path/to/your.pth "
                "or save a fine-tuned checkpoint at checkpoints/usr2_base_plus.pth."
            )
        if not ckpt.exists():
            raise NotInstalled(
                f"Checkpoint not found at {ckpt}.\n"
                "Download from https://github.com/ahaliassos/usr2 → Pretrained Models "
                "→ Fine-tuned (full model) → High-resource → Base+ LRS3+Vox2. "
                f"Save / rename the .pth to:\n  {ckpt}"
            )
        try:
            import torch  # noqa: F401
        except ImportError:
            raise NotInstalled("PyTorch not installed. Run: bash scripts/setup_usr2.sh")

    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult:
        self.load()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        repo = config.USR2_REPO_DIR
        ckpt = config.AVSR_CHECKPOINT_PATH
        assert ckpt is not None
        shim = Path(__file__).resolve().parent.parent / "scripts" / "usr2_shim.py"
        cmd = [
            sys.executable, str(shim),
            f"video={video_path.resolve()}",
            f"model.pretrained_model_path={ckpt.resolve()}",
            f"modality={mode}",
            "detector=mediapipe",
            f"model/backbone={config.USR2_BACKBONE}",
        ]
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=900)
        runtime_ms = int((time.monotonic() - t0) * 1000)
        if proc.returncode != 0:
            raise RuntimeError(
                f"USR2 demo.py exited {proc.returncode}.\n"
                f"Command: cd {repo} && {' '.join(cmd)}\n"
                f"STDERR:\n{proc.stderr[-1500:]}\n"
                f"STDOUT (tail):\n{proc.stdout[-500:]}"
            )
        text = self._extract_transcript(proc.stdout)
        if not text:
            raise RuntimeError(
                "USR2 demo.py exited 0 but no 'Result :' line found.\n"
                f"STDOUT:\n{proc.stdout[-1500:]}"
            )
        return AVSRResult(
            text=text,
            model_name="usr2",
            checkpoint_name=ckpt.name,
            mode=mode,
            runtime_ms=runtime_ms,
        )

    def _extract_transcript(self, stdout: str) -> str:
        matches = self._RESULT_RE.findall(stdout)
        return matches[-1].strip() if matches else ""


# ---- Auto-AVSR backend (subprocess against their demo.py) -------------------

class AutoAVSRBackend:
    """Auto-AVSR (Ma et al., ICASSP 2023).
    https://github.com/mpc001/auto_avsr

    Setup: bash scripts/setup_auto_avsr.sh
    Then download a pretrained .pth listed in their README and either save to
    checkpoints/auto_avsr.pth or set LIPSYNC_AUTO_AVSR_CHECKPOINT=/path.

    Their demo.py is Hydra-based with key=value overrides — same shape as USR2.
    If their CLI / output format differs from this implementation, the error
    message will surface the exact mismatch and the file/line to tweak.
    """

    name = "auto_avsr"
    # Best-effort match for common VSR demo output patterns.
    _RESULT_RE = re.compile(
        r"(?:^\s*Result\s*:\s*|hyp\s*=\s*|prediction\s*:\s*|transcript\s*:\s*)(.+?)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def load(self) -> None:
        repo = config.AUTO_AVSR_REPO_DIR
        if not repo.exists():
            raise NotInstalled(
                f"Auto-AVSR repo not found at {repo}.\n"
                f"Fix: bash scripts/setup_auto_avsr.sh"
            )
        if not (repo / "demo.py").exists():
            raise NotInstalled(
                f"Auto-AVSR demo.py missing at {repo}/demo.py.\n"
                f"Fix: rm -rf {repo} && bash scripts/setup_auto_avsr.sh"
            )
        ckpt = config.AUTO_AVSR_CHECKPOINT_PATH
        if ckpt is None or not ckpt.exists():
            raise NotInstalled(
                f"Auto-AVSR checkpoint not found ({ckpt}).\n"
                "Pick one from https://github.com/mpc001/auto_avsr (README → Models). "
                "Save it to checkpoints/auto_avsr.pth or set LIPSYNC_AUTO_AVSR_CHECKPOINT."
            )
        try:
            import torch  # noqa: F401
        except ImportError:
            raise NotInstalled("PyTorch not installed. Run: bash scripts/setup_auto_avsr.sh")

    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult:
        self.load()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        repo = config.AUTO_AVSR_REPO_DIR
        ckpt = config.AUTO_AVSR_CHECKPOINT_PATH
        assert ckpt is not None
        # Auto-AVSR uses data.modality={video,audio,audiovisual}, file=<path>.
        auto_modality = {"v": "video", "a": "audio", "av": "audiovisual"}[mode]
        cmd = [
            sys.executable, "demo.py",
            f"file={video_path.resolve()}",
            f"pretrained_model_path={ckpt.resolve()}",
            f"data.modality={auto_modality}",
            "detector=mediapipe",
        ]
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=900)
        runtime_ms = int((time.monotonic() - t0) * 1000)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Auto-AVSR demo.py exited {proc.returncode}.\n"
                f"Command: cd {repo} && {' '.join(cmd)}\n"
                f"STDERR:\n{proc.stderr[-1500:]}\n"
                f"STDOUT (tail):\n{proc.stdout[-500:]}\n\n"
                "NOTE: Auto-AVSR's CLI may have rotated. Compare these args against "
                "their current demo.py docstring; adjust src/avsr.py:AutoAVSRBackend.transcribe."
            )
        text = self._extract_transcript(proc.stdout)
        if not text:
            raise RuntimeError(
                "Auto-AVSR demo.py exited 0 but no transcript line found.\n"
                f"STDOUT:\n{proc.stdout[-1500:]}\n\n"
                "Adjust AutoAVSRBackend._RESULT_RE in src/avsr.py to match their actual format."
            )
        return AVSRResult(
            text=text,
            model_name="auto_avsr",
            checkpoint_name=ckpt.name,
            mode=mode,
            runtime_ms=runtime_ms,
        )

    def _extract_transcript(self, stdout: str) -> str:
        matches = self._RESULT_RE.findall(stdout)
        return matches[-1].strip() if matches else ""


# ---- AV-HuBERT placeholder (research/benchmark branch) ----------------------

class AVHuBERTBackend:
    """AV-HuBERT (Shi et al., ICLR 2022).
    Research/benchmark branch. Deliberately not wired up yet — see TRAINING_PLAN.md
    for when we add this. https://github.com/facebookresearch/av_hubert
    """

    name = "avhubert"

    def load(self) -> None:
        raise NotInstalled(
            "avhubert backend is a research placeholder, not yet wired up. "
            "See TRAINING_PLAN.md — this is the AV-HuBERT benchmark branch."
        )

    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult:
        self.load()
        raise NotInstalled("avhubert backend not implemented.")


# ---- Registry + dispatch ----------------------------------------------------

_BACKENDS: dict[str, type[Backend]] = {
    "mock": MockBackend,
    "usr2": USR2Backend,
    "auto_avsr": AutoAVSRBackend,
    "avhubert": AVHuBERTBackend,
}

_instances: dict[str, Backend] = {}


def available_models() -> list[str]:
    """Names registered in the model registry."""
    return sorted(_BACKENDS)


def get_backend(model: str | None = None) -> Backend:
    """Resolve a Backend instance for `model` (defaults to config.AVSR_BACKEND)."""
    name = model or config.AVSR_BACKEND
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown model {name!r}. Pick one of {available_models()}."
        )
    if name not in _instances:
        _instances[name] = _BACKENDS[name]()
    return _instances[name]


def transcribe(
    video_path: Path,
    mode: str | None = None,
    model: str | None = None,
) -> AVSRResult:
    """Top-level inference call.

    mode  : "vsr" / "avsr" / "asr" (also accepts "v" / "a" / "av").
    model : "usr2" / "auto_avsr" / "avhubert" / "mock".
    """
    canonical_mode = normalize_mode(mode)
    backend = get_backend(model)
    return backend.transcribe(video_path, canonical_mode)
