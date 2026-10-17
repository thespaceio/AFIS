"""
Synthetic fingerprints: lawful demo, test and evaluation data.

Real fingerprint databases are biometric data and their licences usually
forbid redistribution, so AFIS ships with a generator instead. It follows
the approach of SFinGe (Cappelli, Maio & Maltoni, 2000):

1. A finger-shaped foreground mask.
2. An orientation field from the zero-pole model of Sherlock & Monro (1993):
   cores and deltas are placed according to the pattern class (arch, tented
   arch, left/right loop, whorl), plus a smooth random perturbation.
3. A ridge-wavelength map (about 8-10 px, i.e. ~500 dpi).
4. Ridges "grown" from random seed noise by repeatedly applying Gabor
   filters tuned to the local orientation and wavelength. Where ridges
   collide they end or fork, so every seed produces different minutiae.
5. Rendering: dark ridges on a light background with pressure, contrast
   and sensor noise.

`generate_impressions` then makes several captures of the same finger
(placement, rotation, skin distortion, pressure, partial capture) using
`afis.evaluation.synthetic_impression`.

Synthetic prints are cleaner and more regular than real ones. Accuracy
measured on them shows that the system works end to end; it says little
about accuracy on real-world captures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

PATTERNS = ("arch", "tented_arch", "left_loop", "right_loop", "whorl")


@dataclass
class MasterFinger:
    """A synthetic finger: the ideal ridge pattern before any capture effects."""
    ridges: np.ndarray            # float32 in [-1, 1], ridges positive
    mask: np.ndarray              # bool foreground
    orientation: np.ndarray       # radians in [0, pi), direction along ridges
    pattern: str
    cores: List[Tuple[int, int]]
    deltas: List[Tuple[int, int]]
    seed: int


def _finger_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Rounded-top, slightly flattened ellipse, like a finger pressed on glass."""
    cy, cx = h * rng.uniform(0.50, 0.56), w * rng.uniform(0.47, 0.53)
    ry, rx = h * rng.uniform(0.40, 0.46), w * rng.uniform(0.36, 0.43)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = (y - cy) / ry
    # Flatter at the bottom (the finger continues off the sensor).
    dy = np.where(dy > 0, dy * 0.8, dy)
    return (((x - cx) / rx) ** 2 + dy ** 2) <= 1.0


def _singularities(pattern: str, h: int, w: int, rng: np.random.Generator):
    """Core and delta positions (x, y) for each pattern class."""
    j = lambda s: rng.uniform(-s, s)  # noqa: E731 - small jitter helper
    cx, cy = w * (0.5 + j(0.05)), h * (0.42 + j(0.05))
    if pattern == "arch":
        return [], []
    if pattern == "tented_arch":
        return [(cx, cy)], [(cx + j(4), cy + h * rng.uniform(0.12, 0.16))]
    if pattern == "left_loop":
        return [(cx, cy)], [(cx - w * rng.uniform(0.18, 0.26), cy + h * rng.uniform(0.25, 0.32))]
    if pattern == "right_loop":
        return [(cx, cy)], [(cx + w * rng.uniform(0.18, 0.26), cy + h * rng.uniform(0.25, 0.32))]
    if pattern == "whorl":
        gap = h * rng.uniform(0.06, 0.10)
        cores = [(cx + j(6), cy - gap / 2), (cx + j(6), cy + gap / 2)]
        dy = cy + h * rng.uniform(0.25, 0.30)
        deltas = [(cx - w * rng.uniform(0.25, 0.32), dy), (cx + w * rng.uniform(0.25, 0.32), dy)]
        return cores, deltas
    raise ValueError(f"Unknown pattern '{pattern}'. Use one of {PATTERNS}")


def _orientation_field(pattern: str, h: int, w: int, cores, deltas, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    z = x + 1j * y
    if pattern == "arch":
        # Ridges run across the finger and bulge upwards in the middle.
        bulge = rng.uniform(0.6, 1.0)
        u = (x - w / 2) / (w / 2)
        v = np.clip((h * 0.75 - y) / h, 0, 1)
        theta = np.arctan(-bulge * 2.2 * u * v * np.exp(-u * u))
    else:
        # Sherlock-Monro: theta = 1/2 (sum arg(z - delta) - sum arg(z - core)).
        theta = np.zeros_like(x)
        for dx_, dy_ in deltas:
            theta += np.angle(z - (dx_ + 1j * dy_))
        for cx_, cy_ in cores:
            theta -= np.angle(z - (cx_ + 1j * cy_))
        theta *= 0.5
        if len(cores) == 1 and len(deltas) == 1:
            # Keep the field horizontal far from the loop.
            theta += 0.0
    # Smooth random perturbation in the doubled-angle domain.
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 40)
    noise *= np.radians(rng.uniform(4, 9)) / (np.abs(noise).max() + 1e-9)
    theta = theta + noise
    return np.mod(theta, np.pi)


def _gabor_kernel(theta: float, wavelength: float) -> np.ndarray:
    """Even-symmetric Gabor kernel whose stripes run along direction `theta`."""
    sigma = 0.55 * wavelength
    half = int(np.ceil(2.6 * sigma))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    # Coordinate across the ridges (normal to theta).
    n = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma)) * np.cos(2 * np.pi * n / wavelength)
    return (g - g.mean()).astype(np.float32)


def generate_master(seed: int, pattern: Optional[str] = None, size: Tuple[int, int] = (400, 320),
                    iterations: int = 14, angle_bins: int = 24) -> MasterFinger:
    """Grow one synthetic finger. The same seed always gives the same finger."""
    rng = np.random.default_rng(seed)
    h, w = size
    if pattern is None:
        # Rough natural frequencies: loops most common, then whorls, then arches.
        pattern = str(rng.choice(PATTERNS, p=[0.05, 0.05, 0.32, 0.32, 0.26]))
    mask = _finger_mask(h, w, rng)
    cores, deltas = _singularities(pattern, h, w, rng)
    orientation = _orientation_field(pattern, h, w, cores, deltas, rng)

    wavelength = rng.uniform(8.3, 9.7)
    bins = np.round(orientation / np.pi * angle_bins).astype(np.int32) % angle_bins
    kernels = [_gabor_kernel(k * np.pi / angle_bins, wavelength) for k in range(angle_bins)]

    # Sparse random seeds; each seed set leads to a different ridge topology.
    img = np.zeros((h, w), np.float32)
    n_seeds = int(rng.integers(40, 70))
    ys, xs = rng.integers(0, h, n_seeds), rng.integers(0, w, n_seeds)
    img[ys, xs] = rng.choice([-1.0, 1.0], n_seeds)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)

    for _ in range(iterations):
        out = np.zeros_like(img)
        for k, kern in enumerate(kernels):
            sel = bins == k
            if sel.any():
                out[sel] = cv2.filter2D(img, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT)[sel]
        # Soft clipping keeps ridges at a constant amplitude.
        img = np.tanh(out / (np.percentile(np.abs(out), 90) + 1e-6) * 1.6)

    img[~mask] = 0
    to_int = lambda pts: [(int(round(px)), int(round(py))) for px, py in pts]  # noqa: E731
    return MasterFinger(img.astype(np.float32), mask, orientation, pattern, to_int(cores), to_int(deltas), seed)


def render(master: MasterFinger, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Turn the ideal ridge map into a grayscale 'scanner' image (dark ridges, light background)."""
    rng = rng or np.random.default_rng(master.seed + 1)
    h, w = master.ridges.shape
    # Pressure varies smoothly over the finger; light pressure fades ridges near the edge.
    pressure = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 30)
    pressure = 0.75 + 0.25 * pressure / (np.abs(pressure).max() + 1e-9)
    edge = cv2.GaussianBlur(master.mask.astype(np.float32), (0, 0), 6)
    ridge = np.clip(master.ridges * pressure, -1, 1)
    ink = (ridge + 1) / 2 * edge                     # 0 = no ink, 1 = full ridge
    img = 235 - 190 * ink
    img += rng.normal(0, 5, img.shape)
    img = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0.7)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_impressions(master: MasterFinger, count: int, seed: Optional[int] = None,
                         strength: float = 1.0) -> List[np.ndarray]:
    """`count` captures of the same finger. The first one is the undistorted rendering."""
    from .evaluation import synthetic_impression

    rng = np.random.default_rng(master.seed * 7919 + 17 if seed is None else seed)
    base = render(master, rng)
    images = [base]
    while len(images) < count:
        images.append(synthetic_impression(render(master, rng), rng, strength))
    return images


def generate_dataset(fingers: int, impressions: int, first_seed: int = 1000,
                     patterns: Optional[Sequence[str]] = None) -> List[Tuple[int, int, np.ndarray]]:
    """[(finger_index, impression_index, image), ...] with FVC-style numbering from 1."""
    out = []
    for f in range(fingers):
        pattern = patterns[f % len(patterns)] if patterns else None
        master = generate_master(first_seed + f, pattern)
        for i, img in enumerate(generate_impressions(master, impressions)):
            out.append((f + 1, i + 1, img))
    return out
