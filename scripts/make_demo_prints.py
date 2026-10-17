#!/usr/bin/env python3
"""Regenerate the synthetic demo prints shipped in afis/demo_prints/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afis.demo import demo_dir, write_demo_prints  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else demo_dir()
    files = write_demo_prints(out)
    print(f"Wrote {len(files)} demo prints to {out}")
