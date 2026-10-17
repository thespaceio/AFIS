"""Drawing helpers shared by the GUI, the CLI and the evaluation scripts (OpenCV only)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .features import Minutia

# BGR colours
ENDING = (60, 200, 60)        # green
BIFURCATION = (235, 140, 30)  # blue
CORE = (40, 40, 230)          # red
DELTA = (0, 200, 230)         # yellow
MATCH = (0, 204, 106)
WEAK = (36, 165, 245)          # amber: pairings found for a non-match


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def draw_minutiae(image: np.ndarray, minutiae: Iterable[Minutia],
                  cores: Sequence = (), deltas: Sequence = (), scale: int = 2) -> np.ndarray:
    """Minutiae as circles with a direction tick; cores as red squares, deltas as yellow triangles."""
    out = to_bgr(image)
    if scale != 1:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for m in minutiae:
        colour = ENDING if m.minutiae_type == "ending" else BIFURCATION
        cx, cy = m.x * scale, m.y * scale
        cv2.circle(out, (cx, cy), 4 * scale // 2 + 2, colour, 2, cv2.LINE_AA)
        ex = int(cx + 9 * scale * np.cos(m.orientation))
        ey = int(cy + 9 * scale * np.sin(m.orientation))
        cv2.line(out, (cx, cy), (ex, ey), colour, 2, cv2.LINE_AA)
    for x, y in cores:
        cv2.rectangle(out, (x * scale - 9, y * scale - 9), (x * scale + 9, y * scale + 9), CORE, 2)
    for x, y in deltas:
        pts = np.array([[x * scale, y * scale - 11], [x * scale - 10, y * scale + 8],
                        [x * scale + 10, y * scale + 8]], np.int32)
        cv2.polylines(out, [pts], True, DELTA, 2, cv2.LINE_AA)
    return out


def draw_match(img1: np.ndarray, min1: List[Minutia], img2: np.ndarray, min2: List[Minutia],
               pairs: List[Tuple[int, int]], gap: int = 20, is_match: bool = True) -> np.ndarray:
    """Two prints side by side with lines joining matched minutiae (amber if the pair didn't match)."""
    colour = MATCH if is_match else WEAK
    a = draw_minutiae(img1, min1, scale=1)
    b = draw_minutiae(img2, min2, scale=1)
    h = max(a.shape[0], b.shape[0])
    canvas = np.full((h, a.shape[1] + gap + b.shape[1], 3), 255, np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    off = a.shape[1] + gap
    canvas[: b.shape[0], off:] = b
    for i, j in pairs:
        p = (min1[i].x, min1[i].y)
        q = (min2[j].x + off, min2[j].y)
        cv2.line(canvas, p, q, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 5, colour, 2, cv2.LINE_AA)
        cv2.circle(canvas, q, 5, colour, 2, cv2.LINE_AA)
    return canvas


def montage(images: List[np.ndarray], labels: Optional[List[str]] = None, cols: int = 4,
            cell: int = 260) -> np.ndarray:
    """Grid of images with captions (used for the README and reports)."""
    tiles = []
    for k, img in enumerate(images):
        t = to_bgr(img)
        s = cell / max(t.shape[:2])
        t = cv2.resize(t, (int(t.shape[1] * s), int(t.shape[0] * s)), interpolation=cv2.INTER_AREA)
        tile = np.full((cell + 28, cell, 3), 255, np.uint8)
        y0, x0 = (cell - t.shape[0]) // 2, (cell - t.shape[1]) // 2
        tile[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        if labels:
            cv2.putText(tile, labels[k], (6, cell + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)
