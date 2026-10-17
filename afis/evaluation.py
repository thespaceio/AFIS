"""
Accuracy evaluation: genuine / impostor scores, FAR, FRR and the equal error rate.

FAR (false accept rate)  = impostor pairs scoring >= threshold
FRR (false reject rate)  = genuine pairs scoring  <  threshold
EER (equal error rate)   = the point where FAR == FRR (lower is better)

`synthetic_impression` creates extra impressions of a real print (rotation,
translation, skin elasticity, pressure, noise, partial capture). It is useful
for testing when only a few real prints are available, but results on real
multi-impression databases such as FVC2002 are the meaningful benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import cv2
import numpy as np


@dataclass
class ErrorRates:
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    eer: float
    eer_threshold: float
    genuine: np.ndarray
    impostor: np.ndarray

    def far_at(self, threshold: float) -> float:
        return float(np.mean(self.impostor >= threshold)) if len(self.impostor) else 0.0

    def frr_at(self, threshold: float) -> float:
        return float(np.mean(self.genuine < threshold)) if len(self.genuine) else 0.0

    def threshold_for_far(self, target_far: float) -> float:
        """Lowest threshold whose FAR is at or below the target."""
        ok = self.thresholds[self.far <= target_far]
        return float(ok.min()) if len(ok) else float(self.thresholds.max())

    def summary(self) -> Dict[str, float]:
        t1 = self.threshold_for_far(0.01)
        t0 = self.threshold_for_far(0.0)
        return {
            "genuine_pairs": int(len(self.genuine)),
            "impostor_pairs": int(len(self.impostor)),
            "eer": round(self.eer, 4),
            "eer_threshold": round(self.eer_threshold, 4),
            "frr_at_far_1pct": round(self.frr_at(t1), 4),
            "threshold_far_1pct": round(t1, 4),
            "frr_at_far_0": round(self.frr_at(t0), 4),
            "threshold_far_0": round(t0, 4),
            "genuine_mean": round(float(np.mean(self.genuine)), 4) if len(self.genuine) else 0.0,
            "impostor_mean": round(float(np.mean(self.impostor)), 4) if len(self.impostor) else 0.0,
            "impostor_max": round(float(np.max(self.impostor)), 4) if len(self.impostor) else 0.0,
        }


def error_rates(genuine: Sequence[float], impostor: Sequence[float]) -> ErrorRates:
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    thresholds = np.unique(np.concatenate([g, i, [0.0, 1.0001]]))
    far = np.array([np.mean(i >= t) if len(i) else 0.0 for t in thresholds])
    frr = np.array([np.mean(g < t) if len(g) else 0.0 for t in thresholds])
    k = int(np.argmin(np.abs(far - frr)))
    # Interpolate between the two thresholds that bracket the crossing.
    eer = float((far[k] + frr[k]) / 2)
    return ErrorRates(thresholds, far, frr, eer, float(thresholds[k]), g, i)


def plot_report(rates: ErrorRates, path: str, title: str = "AFIS") -> None:
    """Score distributions + DET-style curve, saved as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, max(1e-3, float(max(rates.genuine.max(initial=0), rates.impostor.max(initial=0)))), 30)
    ax1.hist(rates.impostor, bins=bins, alpha=0.75, label=f"Impostor ({len(rates.impostor)})", color="#FF3B30")
    ax1.hist(rates.genuine, bins=bins, alpha=0.75, label=f"Genuine ({len(rates.genuine)})", color="#00CC6A")
    ax1.axvline(rates.eer_threshold, color="#333", ls="--", lw=1, label=f"EER threshold {rates.eer_threshold:.3f}")
    ax1.set_xlabel("Match score")
    ax1.set_ylabel("Pairs")
    ax1.set_title("Score distributions")
    ax1.legend(frameon=False)

    ax2.plot(rates.thresholds, rates.far * 100, label="FAR", color="#FF3B30")
    ax2.plot(rates.thresholds, rates.frr * 100, label="FRR", color="#00CC6A")
    ax2.axvline(rates.eer_threshold, color="#333", ls="--", lw=1)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Error rate (%)")
    ax2.set_title(f"FAR / FRR  (EER = {rates.eer * 100:.2f}%)")
    ax2.set_xlim(0, min(1, float(rates.thresholds.max())))
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def synthetic_impression(image: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """A plausible new impression of the same finger."""
    h, w = image.shape
    pad = int(0.2 * max(h, w))
    img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(np.median(image)))
    H, W = img.shape

    # Skin elasticity: a smooth random displacement field (a few pixels).
    amp = 3.0 * strength
    dx = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 25)
    dy = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 25)
    dx *= amp / (np.abs(dx).max() + 1e-9)
    dy *= amp / (np.abs(dy).max() + 1e-9)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    img = cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Placement: rotation and translation.
    angle = rng.uniform(-25, 25) * strength
    tx, ty = rng.uniform(-25, 25, 2) * strength
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=int(np.median(image)))

    # Pressure: dry (thin ridges) or wet (thick ridges).
    k = np.ones((2, 2), np.uint8)
    p = rng.uniform()
    if p < 0.25:
        img = cv2.dilate(img, k)      # dry: dark ridges shrink
    elif p > 0.75:
        img = cv2.erode(img, k)       # wet: dark ridges grow

    # Contrast, blur and sensor noise.
    img = img.astype(np.float32) * rng.uniform(0.8, 1.15) + rng.uniform(-15, 15)
    if rng.uniform() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
    img += rng.normal(0, 6 * strength, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Partial capture: crop back to the original size with a random offset.
    oy, ox = rng.integers(int(pad * 0.6), int(pad * 1.4), 2)
    return img[oy:oy + h, ox:ox + w]
