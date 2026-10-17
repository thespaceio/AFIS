"""
Fingerprint preprocessing and enhancement.

Pipeline (each step is a public method so the GUI can show it):

    raw image
      -> normalise          zero mean / unit variance
      -> segment            foreground (finger) mask from local variance
      -> orientation        ridge direction per pixel (doubled-angle smoothing)
      -> frequency          ridge frequency (1 / ridge spacing)
      -> enhance            oriented Gabor filters tuned to the ridge frequency
      -> binarise           ridges = 1, valleys = 0 (inside the mask)
      -> skeletonise        one-pixel-wide ridge lines

Conventions used everywhere in the package:
    * image coordinates: x to the right, y down
    * `orientation` is the direction *along* the ridges in [0, pi)

References:
    Hong, Wan & Jain (1998) "Fingerprint image enhancement: algorithm and
    performance evaluation", IEEE TPAMI 20(8).
    P. Kovesi, "MATLAB and Octave Functions for Computer Vision" (ridge filtering).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


@dataclass
class PreprocessConfig:
    block_size: int = 16                 # segmentation block size (px)
    segment_threshold: float = 0.1       # min std of normalised block to count as finger
    gradient_sigma: float = 1.0          # smoothing before gradients
    block_sigma: float = 7.0             # window for the structure tensor
    orient_smooth_sigma: float = 7.0     # smoothing of the doubled-angle field
    freq_block_size: int = 32            # block size for frequency estimation
    freq_window: int = 5                 # 1-D smoothing window for the ridge signature
    min_wavelength: float = 5.0          # px, valid ridge spacing range at ~500 dpi
    max_wavelength: float = 15.0
    gabor_kx: float = 0.65               # Gabor sigma = k / frequency
    gabor_ky: float = 0.65
    angle_step_deg: float = 3.0          # orientation quantisation for the filter bank
    ridge_threshold: float = 0.0         # enhanced response below this = ridge


@dataclass
class PreprocessResult:
    """All intermediate images, kept for visualisation and feature extraction."""
    raw: np.ndarray
    normalized: np.ndarray
    mask: np.ndarray
    orientation: np.ndarray
    coherence: np.ndarray
    frequency: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray
    skeleton: np.ndarray
    quality: float
    median_wavelength: float
    timings: Dict[str, float] = field(default_factory=dict)
    ridge_fraction: float = 1.0          # share of finger blocks with a clear, plausible ridge spacing

    def stages(self) -> Dict[str, np.ndarray]:
        """8-bit images for display, in pipeline order."""
        def to_u8(a: np.ndarray) -> np.ndarray:
            a = a.astype(np.float32)
            lo, hi = float(a.min()), float(a.max())
            return np.zeros_like(a, np.uint8) if hi <= lo else ((a - lo) / (hi - lo) * 255).astype(np.uint8)

        orient_vis = (self.orientation / np.pi * 179).astype(np.uint8)
        hsv = np.dstack([orient_vis, np.full_like(orient_vis, 200), (self.coherence * 255).astype(np.uint8)])
        orient_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        orient_rgb[~self.mask] = 0

        return {
            "Raw": self.raw,
            "Normalised": to_u8(self.normalized),
            "Segmentation mask": (self.mask * 255).astype(np.uint8),
            "Orientation field": orient_rgb,
            "Enhanced (Gabor)": to_u8(-self.enhanced * self.mask),
            "Binary ridges": (self.binary * 255).astype(np.uint8),
            "Skeleton": (self.skeleton * 255).astype(np.uint8),
        }


class PreprocessingPipeline:
    """Turns a grayscale fingerprint image into an enhanced ridge skeleton."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._filter_cache: Dict[float, list] = {}
        self.stats: Dict[str, Any] = {"processed_images": 0, "processing_times": [], "quality_scores": []}

    # ------------------------------------------------------------------ public
    def preprocess(self, raw_image: np.ndarray) -> PreprocessResult:
        if not isinstance(raw_image, np.ndarray) or raw_image.ndim != 2:
            raise ValueError("Input must be a 2-D grayscale numpy array")

        t: Dict[str, float] = {}
        t0 = time.perf_counter()

        norm = self.normalise(raw_image)
        t["normalise"] = time.perf_counter() - t0

        s = time.perf_counter()
        mask = self.segment(norm)
        if mask.sum() < 0.05 * mask.size:
            raise ValueError("No fingerprint area found in the image")
        # Re-normalise using only the finger area so background doesn't skew contrast.
        fg = norm[mask]
        norm = (norm - fg.mean()) / (fg.std() + 1e-9)
        t["segment"] = time.perf_counter() - s

        s = time.perf_counter()
        orientation, coherence = self.orientation(norm)
        t["orientation"] = time.perf_counter() - s

        s = time.perf_counter()
        frequency, median_wl, ridge_fraction = self.frequency(norm, orientation, mask, with_fraction=True)
        t["frequency"] = time.perf_counter() - s

        s = time.perf_counter()
        enhanced = self.enhance(norm, orientation, 1.0 / median_wl)
        t["enhance"] = time.perf_counter() - s

        s = time.perf_counter()
        binary = (enhanced < self.config.ridge_threshold) & mask
        binary = self._clean_binary(binary)
        skeleton = skeletonize(binary)
        t["skeleton"] = time.perf_counter() - s

        quality = self.quality(mask, coherence, ridge_fraction)
        total = time.perf_counter() - t0
        t["total"] = total

        self.stats["processed_images"] += 1
        self.stats["processing_times"].append(total)
        self.stats["quality_scores"].append(quality)

        return PreprocessResult(
            raw=raw_image, normalized=norm, mask=mask, orientation=orientation,
            coherence=coherence, frequency=frequency, enhanced=enhanced,
            binary=binary, skeleton=skeleton, quality=quality,
            median_wavelength=median_wl, timings=t, ridge_fraction=ridge_fraction,
        )

    # ------------------------------------------------------------------ stages
    @staticmethod
    def normalise(image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float64)
        return (img - img.mean()) / (img.std() + 1e-9)

    def segment(self, norm: np.ndarray) -> np.ndarray:
        """Foreground mask: blocks whose local standard deviation is high enough."""
        b = self.config.block_size
        h, w = norm.shape
        hb, wb = int(np.ceil(h / b)), int(np.ceil(w / b))
        padded = np.zeros((hb * b, wb * b))
        padded[:h, :w] = norm
        block_std = padded.reshape(hb, b, wb, b).std(axis=(1, 3))
        block_mask = block_std > self._segment_cut(block_std)

        # Smooth the block mask: close gaps, drop islands, fill holes.
        block_mask = ndimage.binary_closing(block_mask, iterations=2)
        block_mask = ndimage.binary_opening(block_mask, iterations=1)
        labels, n = ndimage.label(block_mask)
        if n > 1:
            sizes = ndimage.sum(block_mask, labels, range(1, n + 1))
            block_mask = labels == (int(np.argmax(sizes)) + 1)
        block_mask = ndimage.binary_fill_holes(block_mask)

        mask = np.kron(block_mask, np.ones((b, b), dtype=bool))[:h, :w]
        # Soften the blocky edge.
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), b / 2) > 0.5
        return mask

    def _segment_cut(self, block_std: np.ndarray) -> float:
        """Threshold between background and finger blocks (Otsu on block std, with a floor)."""
        vals = block_std.ravel()
        hist, edges = np.histogram(vals, bins=64)
        centers = (edges[:-1] + edges[1:]) / 2
        w0 = np.cumsum(hist)
        w1 = w0[-1] - w0
        m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
        m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
        between = w0 * w1 * (m0 - m1) ** 2
        otsu = centers[int(np.argmax(between))]
        return max(self.config.segment_threshold, min(otsu, 0.5 * vals.max()))

    def orientation(self, norm: np.ndarray):
        """Ridge orientation in [0, pi) and coherence in [0, 1]."""
        c = self.config
        smoothed = cv2.GaussianBlur(norm, (0, 0), c.gradient_sigma)
        gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

        gxx = cv2.GaussianBlur(gx * gx, (0, 0), c.block_sigma)
        gyy = cv2.GaussianBlur(gy * gy, (0, 0), c.block_sigma)
        gxy = cv2.GaussianBlur(gx * gy, (0, 0), c.block_sigma)

        # Doubled-angle representation avoids the 0 / pi wrap-around problem.
        denom = np.sqrt(4 * gxy ** 2 + (gxx - gyy) ** 2) + 1e-12
        sin2 = 2 * gxy / denom
        cos2 = (gxx - gyy) / denom
        sin2 = cv2.GaussianBlur(sin2, (0, 0), c.orient_smooth_sigma)
        cos2 = cv2.GaussianBlur(cos2, (0, 0), c.orient_smooth_sigma)

        gradient_dir = 0.5 * np.arctan2(sin2, cos2)          # across the ridges
        orientation = np.mod(gradient_dir + np.pi / 2, np.pi)  # along the ridges
        coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-12)
        return orientation, np.clip(coherence, 0, 1)

    def frequency(self, norm: np.ndarray, orientation: np.ndarray, mask: np.ndarray, with_fraction: bool = False):
        """
        Block-wise ridge frequency; returns (frequency map, median wavelength in px),
        plus the fraction of finger blocks with a valid wavelength if `with_fraction`.
        """
        c = self.config
        b = c.freq_block_size
        h, w = norm.shape
        freq = np.zeros_like(norm)
        wavelengths = []
        blocks = 0

        for r in range(0, h - b + 1, b // 2):
            for col in range(0, w - b + 1, b // 2):
                if mask[r:r + b, col:col + b].mean() < 0.8:
                    continue
                blocks += 1
                wl = self._block_wavelength(norm[r:r + b, col:col + b],
                                            orientation[r:r + b, col:col + b])
                if wl:
                    wavelengths.append(wl)
                    freq[r:r + b, col:col + b] = 1.0 / wl

        median_wl = float(np.median(wavelengths)) if wavelengths else 9.0
        freq[(freq == 0) & mask] = 1.0 / median_wl
        freq[~mask] = 0
        if with_fraction:
            return freq, median_wl, (len(wavelengths) / blocks if blocks else 0.0)
        return freq, median_wl

    def _block_wavelength(self, block: np.ndarray, orient_block: np.ndarray) -> Optional[float]:
        """Rotate the block so ridges are vertical, project, and measure peak spacing."""
        c = self.config
        # Mean orientation of the block via doubled angles.
        ang = 0.5 * np.arctan2(np.mean(np.sin(2 * orient_block)), np.mean(np.cos(2 * orient_block)))
        b = block.shape[0]
        # Rotate so that ridges become vertical (ridge direction -> +y axis).
        rot = cv2.getRotationMatrix2D((b / 2, b / 2), np.degrees(ang) - 90, 1.0)
        rotated = cv2.warpAffine(block, rot, (b, b), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        crop = int(b * 0.15)
        rotated = rotated[crop:b - crop, crop:b - crop]

        signature = rotated.sum(axis=0)
        signature = ndimage.uniform_filter1d(signature, c.freq_window)
        # Local maxima (light valleys between dark ridges).
        dil = ndimage.grey_dilation(signature, size=c.freq_window)
        peaks = np.where((signature == dil) & (signature > signature.mean()))[0]
        if len(peaks) < 2:
            return None
        wl = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
        if c.min_wavelength <= wl <= c.max_wavelength:
            return float(wl)
        return None

    def enhance(self, norm: np.ndarray, orientation: np.ndarray, frequency: float) -> np.ndarray:
        """Oriented Gabor filtering. Ridges come out negative, valleys positive."""
        bank = self._gabor_bank(round(frequency, 3))
        step = np.radians(self.config.angle_step_deg)
        idx = np.round(orientation / step).astype(int) % len(bank)

        img = norm.astype(np.float32)
        enhanced = np.zeros_like(img)
        for k, kernel in enumerate(bank):
            sel = idx == k
            if not sel.any():
                continue
            response = cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            enhanced[sel] = response[sel]
        return enhanced

    def _gabor_bank(self, frequency: float) -> list:
        if frequency in self._filter_cache:
            return self._filter_cache[frequency]
        c = self.config
        sx, sy = c.gabor_kx / frequency, c.gabor_ky / frequency
        size = int(np.round(3 * max(sx, sy)))
        y, x = np.mgrid[-size:size + 1, -size:size + 1].astype(np.float32)
        bank = []
        for theta in np.arange(0, np.pi, np.radians(c.angle_step_deg)):
            # u runs across the ridges, v along them.
            u = -x * np.sin(theta) + y * np.cos(theta)
            v = x * np.cos(theta) + y * np.sin(theta)
            kernel = np.exp(-0.5 * (u ** 2 / sx ** 2 + v ** 2 / sy ** 2)) * np.cos(2 * np.pi * frequency * u)
            kernel -= kernel.mean()            # zero DC: flat areas give no response
            kernel /= np.abs(kernel).sum()
            bank.append(kernel.astype(np.float32))
        self._filter_cache[frequency] = bank
        return bank

    @staticmethod
    def _clean_binary(binary: np.ndarray) -> np.ndarray:
        """Remove tiny ridge fragments and fill tiny holes before thinning."""
        labels, n = ndimage.label(binary)
        if n:
            sizes = ndimage.sum(binary, labels, range(1, n + 1))
            keep = np.zeros(n + 1, dtype=bool)
            keep[1:] = sizes >= 20
            binary = keep[labels]
        holes = ndimage.binary_fill_holes(binary) & ~binary
        labels, n = ndimage.label(holes)
        if n:
            sizes = ndimage.sum(holes, labels, range(1, n + 1))
            small = np.zeros(n + 1, dtype=bool)
            small[1:] = sizes < 10
            binary = binary | small[labels]
        return binary

    @staticmethod
    def quality(mask: np.ndarray, coherence: np.ndarray, ridge_fraction: float = 1.0) -> float:
        """
        0..1 quality from finger area, ridge-flow coherence and how much of the
        finger shows a clear ridge spacing. Noise has low coherence; smooth blobs
        or photos are coherent but have no plausible ridge spacing.
        """
        if not mask.any():
            return 0.0
        area = min(1.0, mask.mean() / 0.4)
        coh = float(np.mean(coherence[mask]))
        return float(np.clip((0.2 * area + 0.8 * coh) * (0.3 + 0.7 * ridge_fraction), 0, 1))

    def get_statistics(self) -> Dict[str, Any]:
        times = self.stats["processing_times"]
        q = self.stats["quality_scores"]
        return {
            "processed_images": self.stats["processed_images"],
            "avg_processing_time": float(np.mean(times)) if times else 0.0,
            "avg_quality_score": float(np.mean(q)) if q else 0.0,
        }
