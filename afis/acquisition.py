"""Loading and validating fingerprint images."""

from __future__ import annotations

import os
from typing import Dict, Union

import cv2
import numpy as np

SUPPORTED_FORMATS = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".pgm"}


class AcquisitionError(ValueError):
    """The image can't be used as a fingerprint."""


class FingerprintAcquisition:
    """
    Loads a grayscale fingerprint without distorting it.

    The original version stretched every image to 500 x 500, which changes the
    ridge spacing and the geometry of the minutiae. Images are now kept at their
    native resolution (about 500 dpi for FVC / most optical scanners) and only
    scaled down, with the aspect ratio preserved, if they're very large.
    """

    def __init__(self, min_dimension: int = 150, max_dimension: int = 1000, min_contrast: float = 15.0):
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.min_contrast = min_contrast
        self.stats: Dict[str, int] = {"images_loaded": 0, "images_rejected": 0}

    def load(self, source: Union[str, os.PathLike, np.ndarray, bytes]) -> np.ndarray:
        """Load from a file path, encoded image bytes, or a numpy array."""
        if isinstance(source, np.ndarray):
            image = source
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(source, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise AcquisitionError("Could not decode the image data")
        else:
            path = os.fspath(source)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_FORMATS:
                raise AcquisitionError(f"Unsupported format '{ext}'. Use one of: {', '.join(sorted(SUPPORTED_FORMATS))}")
            # imdecode handles non-ASCII Windows paths that cv2.imread can't.
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise AcquisitionError(f"Could not decode image: {path}")

        image = np.ascontiguousarray(image.astype(np.uint8))
        self._validate(image)
        image = self._limit_size(image)
        self.stats["images_loaded"] += 1
        return image

    # Kept for compatibility with the original API.
    load_fingerprint = load

    def _validate(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        if min(h, w) < self.min_dimension:
            self._reject(f"Image too small ({w}x{h}); need at least {self.min_dimension}px on each side")
        if float(image.std()) < self.min_contrast:
            self._reject(f"Image contrast too low (std {image.std():.1f} < {self.min_contrast})")

    def _reject(self, message: str) -> None:
        self.stats["images_rejected"] += 1
        raise AcquisitionError(message)

    def _limit_size(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        if max(h, w) <= self.max_dimension:
            return image
        s = self.max_dimension / max(h, w)
        return cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    def get_statistics(self) -> Dict[str, int]:
        return dict(self.stats)
