#!/usr/bin/env python3
"""Command-line entry point: replay data/capture.json through the operator."""

from pathlib import Path
import sys

# Allow ``python -I /workspace/replay_capture.py ...`` where isolated mode may
# not put the repository root on sys.path automatically.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from event_voxel.replay import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
