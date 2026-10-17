"""
Bundled demo prints.

`afis/demo_prints/` holds synthetic fingerprints made with `afis.synthetic`
(6 fingers x 4 impressions, named demo-<finger>_<impression>.png). They are
not real people's fingerprints, so they can be shipped and shared freely.
Regenerate them with  `python scripts/make_demo_prints.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

DEMO_PATTERNS = ["right_loop", "whorl", "left_loop", "arch", "tented_arch", "whorl"]
DEMO_FIRST_SEED = 2027
DEMO_IMPRESSIONS = 4


def demo_dir() -> Path:
    # Works from a source checkout, an installed package and a PyInstaller bundle.
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    for candidate in (Path(__file__).resolve().parent / "demo_prints", base / "afis" / "demo_prints"):
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent / "demo_prints"


def demo_files() -> List[Path]:
    return sorted(demo_dir().glob("demo-*_*.png"))


def demo_people(count: int = 6) -> List[Tuple[str, List[str]]]:
    """[("demo-01", [path of impression 1, 2, ...]), ...] for the first `count` demo fingers."""
    people = {}
    for p in demo_files():
        finger = p.stem.split("_")[0]
        people.setdefault(finger, []).append(str(p))
    if not people:
        raise FileNotFoundError(f"Demo prints are missing from {demo_dir()}")
    return [(pid, sorted(paths)) for pid, paths in sorted(people.items())][:count]


def write_demo_prints(folder: Path) -> List[Path]:
    """(Re)generate the demo prints deterministically into `folder`."""
    import cv2

    from .synthetic import generate_impressions, generate_master

    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for f, pattern in enumerate(DEMO_PATTERNS, 1):
        master = generate_master(DEMO_FIRST_SEED + f - 1, pattern)
        for i, img in enumerate(generate_impressions(master, DEMO_IMPRESSIONS), 1):
            path = folder / f"demo-{f:02d}_{i}.png"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            written.append(path)
    return written
