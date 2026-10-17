"""Loading and validating fingerprint images."""

from __future__ import annotations

import os
from typing import Dict, Union

import numpy as np

# OpenCV reads this limit lazily on the first decode; it stops decompression bombs
# from allocating gigabytes before our own checks run.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "40000000")

import cv2  # noqa: E402

SUPPORTED_FORMATS = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".pgm"}
MAX_FILE_BYTES = 40 * 1024 * 1024       # refuse absurdly large files before decoding them
MAX_PIXELS = 40_000_000                 # e.g. a crafted PNG that decompresses to gigabytes


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
                code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
                image = cv2.cvtColor(image, code)
        elif isinstance(source, (bytes, bytearray)):
            if len(source) > MAX_FILE_BYTES:
                raise AcquisitionError("Image data is too large")
            image = self._decode(bytes(source), "the image data")
        else:
            path = os.fspath(source)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_FORMATS:
                raise AcquisitionError(f"Unsupported file type '{ext or 'none'}'. "
                                       f"Use one of: {', '.join(sorted(SUPPORTED_FORMATS))}")
            size = os.path.getsize(path)
            if size == 0:
                raise AcquisitionError(f"The file is empty: {os.path.basename(path)}")
            if size > MAX_FILE_BYTES:
                raise AcquisitionError(f"The file is too large ({size / 1e6:.0f} MB; limit "
                                       f"{MAX_FILE_BYTES // 1_000_000} MB)")
            # Read bytes ourselves: imdecode handles non-ASCII Windows paths that cv2.imread can't.
            with open(path, "rb") as fh:
                image = self._decode(fh.read(), os.path.basename(path))

        if image.ndim != 2 or image.size == 0:
            raise AcquisitionError("Expected a grayscale or colour image")
        if image.dtype != np.uint8:
            image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        image = np.ascontiguousarray(image.astype(np.uint8))
        self._validate(image)
        image = self._limit_size(image)
        self.stats["images_loaded"] += 1
        return image

    @staticmethod
    def _decode(data: bytes, name: str) -> np.ndarray:
        """Decode by content (not extension) with a pixel-count guard."""
        buf = np.frombuffer(data, np.uint8)
        try:
            image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        except cv2.error as e:
            raise AcquisitionError(f"Could not read {name}: the file is damaged or not an image") from e
        if image is None:
            raise AcquisitionError(f"Could not read {name}: the file is damaged or not a supported image")
        if image.size > MAX_PIXELS:
            raise AcquisitionError(f"{name} is too large ({image.shape[1]}x{image.shape[0]} pixels)")
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
