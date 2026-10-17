import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = Path(os.environ.get("AFIS_SAMPLES", ROOT / "samples"))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real per-user data folder, key or settings from a test."""
    monkeypatch.setenv("AFIS_HOME", str(tmp_path / "afis_home"))
    monkeypatch.delenv("AFIS_DB", raising=False)
    monkeypatch.delenv("AFIS_KEY", raising=False)
    yield


def sample(name: str) -> Path:
    p = SAMPLES / name
    if not p.exists():
        pytest.skip(f"sample {name} not available (see samples/README.md)")
    return p


def demo(finger: int, impression: int) -> str:
    """Path of a bundled synthetic demo print, e.g. demo(1, 2) -> demo-01_2.png."""
    from afis.demo import demo_dir
    p = demo_dir() / f"demo-{finger:02d}_{impression}.png"
    assert p.exists(), f"bundled demo print missing: {p}"
    return str(p)


def ridge_pattern(angle_deg: float, wavelength: float = 9.0, size: int = 300) -> np.ndarray:
    th = np.radians(angle_deg)
    y, x = np.mgrid[0:size, 0:size]
    u = -x * np.sin(th) + y * np.cos(th)
    img = (128 + 100 * np.cos(2 * np.pi * u / wavelength)).astype(np.uint8)
    img[:40] = 200
    img[:, :40] = 200
    return img
