"""Launch the lipsync-text FastAPI server.

    python scripts/serve.py                  # binds 127.0.0.1:8000
    python scripts/serve.py --host 0.0.0.0   # bind all interfaces (LAN/cloud)
    python scripts/serve.py --port 9000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on src/ changes")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT / "src")] if args.reload else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
