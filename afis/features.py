"""
Minutiae and global feature extraction from a ridge skeleton.

Minutiae
    * Candidates come from the crossing number (CN) of each skeleton pixel:
      CN = 1 -> ridge ending, CN = 3 -> bifurcation.
    * Each minutia gets a full direction in [0, 2*pi) by tracing the ridge:
      endings point out of the ridge end, bifurcations point into the fork.
    * Spurious minutiae are removed: outside the finger, short spurs, short
      ridge islands, broken ridges (two facing endings) and bridges
      (two close bifurcations).

Global features
    * Singular points (cores and deltas) from the Poincare index of the
      orientation field, with differences wrapped to (-pi/2, pi/2] because a
      ridge orientation is only defined modulo pi.
    * A simple pattern class (arch / loop / whorl) from the singular points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

# 8-neighbourhood in circular (clockwise) order, as (dy, dx), starting north.
RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


@dataclass
class Minutia:
    """A ridge ending or bifurcation. Coordinates in pixels (x right, y down)."""
    x: int
    y: int
    orientation: float          # direction in radians, [0, 2*pi)
    minutiae_type: str          # 'ending' or 'bifurcation'
    confidence: float = 1.0

    def __post_init__(self):
        self.x, self.y = int(self.x), int(self.y)
        self.orientation = float(self.orientation) % (2 * np.pi)
        if self.minutiae_type not in ("ending", "bifurcation"):
            raise ValueError(f"Invalid minutia type: {self.minutiae_type}")
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    def distance_to(self, other: "Minutia") -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def angle_difference(self, other: "Minutia") -> float:
        d = abs(self.orientation - other.orientation) % (2 * np.pi)
        return float(min(d, 2 * np.pi - d))

    def to_list(self) -> list:
        return [self.x, self.y, round(self.orientation, 5), self.minutiae_type, round(self.confidence, 3)]

    @classmethod
    def from_list(cls, data) -> "Minutia":
        return cls(x=data[0], y=data[1], orientation=data[2], minutiae_type=data[3], confidence=data[4])


@dataclass
class FeatureConfig:
    border_margin: int = 10            # px inside the finger mask
    trace_length: int = 12             # px traced to measure minutia direction
    spur_length: int = 12              # ending -> junction shorter than this = spur
    island_length: int = 18            # ending -> ending shorter than this = short ridge
    broken_ridge_distance: float = 14  # two facing endings closer than this = broken ridge
    bridge_distance: float = 10        # two bifurcations closer than this = bridge
    duplicate_distance: float = 5
    max_minutiae: int = 150
    singular_step: int = 8             # grid step for the Poincare index
    singular_smooth_sigma: float = 4.0 # extra smoothing (in grid cells) before Poincare


@dataclass
class _Candidate:
    y: int
    x: int
    kind: str
    angle: float = 0.0
    traces: List[Tuple[str, int, Tuple[int, int]]] = field(default_factory=list)  # (stop, length, end)
    remove: bool = False


class FeatureExtractor:
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.stats: Dict[str, Any] = {"images_processed": 0, "minutiae_counts": []}

    # ================================================================ minutiae
    def extract_minutiae(self, skeleton: np.ndarray, orientation: np.ndarray,
                         mask: np.ndarray, coherence: Optional[np.ndarray] = None) -> List[Minutia]:
        sk = skeleton.astype(bool)
        cn = self.crossing_number(sk)

        inner = ndimage.binary_erosion(mask, iterations=self.config.border_margin, border_value=0)
        ys, xs = np.nonzero(sk & ((cn == 1) | (cn == 3)))

        cands: List[_Candidate] = []
        for y, x in zip(ys, xs):
            kind = "ending" if cn[y, x] == 1 else "bifurcation"
            c = _Candidate(int(y), int(x), kind)
            self._describe(sk, cn, c)
            c.remove = not inner[y, x]
            cands.append(c)

        self._remove_spurious(cands, orientation)

        minutiae = []
        for c in cands:
            if c.remove:
                continue
            conf = float(coherence[c.y, c.x]) if coherence is not None else 1.0
            minutiae.append(Minutia(c.x, c.y, c.angle, c.kind, conf))

        minutiae.sort(key=lambda m: m.confidence, reverse=True)
        minutiae = minutiae[: self.config.max_minutiae]

        self.stats["images_processed"] += 1
        self.stats["minutiae_counts"].append(len(minutiae))
        return minutiae

    @staticmethod
    def crossing_number(sk: np.ndarray) -> np.ndarray:
        """CN = 0.5 * sum |P(k) - P(k+1)| around the 8-neighbourhood."""
        p = np.pad(sk.astype(np.int8), 1)
        h, w = sk.shape
        ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING]
        cn = sum(np.abs(ring[k] - ring[(k + 1) % 8]) for k in range(8)) // 2
        return np.where(sk, cn, 0)

    def _trace(self, sk, cn, start: Tuple[int, int], first: Tuple[int, int], limit: int):
        """Follow a ridge from `start` through `first` until a junction, an end, or `limit` px."""
        h, w = sk.shape
        visited = {start, first}
        cur = first
        length = 1
        while length < limit:
            y, x = cur
            if cn[y, x] >= 3:
                return "junction", length, cur
            if cn[y, x] == 1:
                return "end", length, cur
            nbrs = [(y + dy, x + dx) for dy, dx in RING
                    if 0 <= y + dy < h and 0 <= x + dx < w and sk[y + dy, x + dx]
                    and (y + dy, x + dx) not in visited]
            if not nbrs:
                return "end", length, cur
            # Prefer 4-connected steps (handles skeleton staircases).
            nbrs.sort(key=lambda p: abs(p[0] - y) + abs(p[1] - x))
            visited.update(nbrs)
            cur = nbrs[0]
            length += 1
        return "length", length, cur

    def _branch_starts(self, sk, y, x) -> List[Tuple[int, int]]:
        """One starting pixel per ridge branch leaving (y, x)."""
        h, w = sk.shape
        vals = [bool(sk[y + dy, x + dx]) if 0 <= y + dy < h and 0 <= x + dx < w else False for dy, dx in RING]
        starts = []
        for k in range(8):
            if vals[k] and not vals[k - 1]:
                # A run of ridge neighbours starts at k; prefer its 4-connected member.
                run = [k]
                j = (k + 1) % 8
                while vals[j] and j != k:
                    run.append(j)
                    j = (j + 1) % 8
                best = min(run, key=lambda i: abs(RING[i][0]) + abs(RING[i][1]))
                starts.append((y + RING[best][0], x + RING[best][1]))
        return starts

    def _describe(self, sk, cn, c: _Candidate) -> None:
        """Trace the branches of a candidate and compute its direction."""
        L = self.config.trace_length
        starts = self._branch_starts(sk, c.y, c.x)
        vectors = []
        for s in starts:
            stop, length, end = self._trace(sk, cn, (c.y, c.x), s, L)
            c.traces.append((stop, length, end))
            vectors.append((end[1] - c.x, end[0] - c.y))   # (dx, dy)

        if c.kind == "ending" and vectors:
            dx, dy = vectors[0]
            c.angle = float(np.arctan2(-dy, -dx)) % (2 * np.pi)       # out of the ridge end
        elif c.kind == "bifurcation" and len(vectors) == 3:
            ang = [np.arctan2(dy, dx) for dx, dy in vectors]
            best, stem = None, 0
            for i in range(3):
                for j in range(i + 1, 3):
                    d = abs((ang[i] - ang[j] + np.pi) % (2 * np.pi) - np.pi)
                    if best is None or d < best:
                        best, stem = d, 3 - i - j
            dx, dy = vectors[stem]
            c.angle = float(np.arctan2(-dy, -dx)) % (2 * np.pi)       # away from the stem, into the fork
        else:
            c.remove = True   # malformed junction (CN 3 with an unusual neighbourhood)

    def _remove_spurious(self, cands: List[_Candidate], orientation: np.ndarray) -> None:
        cfg = self.config
        by_pos = {(c.y, c.x): c for c in cands}

        for c in cands:
            if c.kind != "ending" or not c.traces:
                continue
            stop, length, end = c.traces[0]
            other = by_pos.get(end)
            if stop == "junction" and length < cfg.spur_length:
                c.remove = True                               # spur
                junction = self._nearest(cands, end, "bifurcation", 2)
                if junction:
                    junction.remove = True
            elif stop == "end" and length < cfg.island_length:
                c.remove = True                               # short ridge / dot
                if other:
                    other.remove = True

        endings = [c for c in cands if c.kind == "ending" and not c.remove]
        for i, a in enumerate(endings):
            for b in endings[i + 1:]:
                d = np.hypot(a.x - b.x, a.y - b.y)
                if d >= cfg.broken_ridge_distance:
                    continue
                facing = abs((a.angle - b.angle) % (2 * np.pi) - np.pi) < np.radians(50)
                if facing:                                    # broken ridge
                    a.remove = b.remove = True

        bifs = [c for c in cands if c.kind == "bifurcation" and not c.remove]
        for i, a in enumerate(bifs):
            for b in bifs[i + 1:]:
                if np.hypot(a.x - b.x, a.y - b.y) < cfg.bridge_distance:
                    a.remove = b.remove = True                # bridge

        kept = [c for c in cands if not c.remove]
        for i, a in enumerate(kept):
            if a.remove:
                continue
            for b in kept[i + 1:]:
                if not b.remove and np.hypot(a.x - b.x, a.y - b.y) < cfg.duplicate_distance:
                    b.remove = True

    @staticmethod
    def _nearest(cands, pos, kind, radius) -> Optional[_Candidate]:
        y, x = pos
        best, bd = None, radius + 1
        for c in cands:
            if c.kind == kind and not c.remove:
                d = max(abs(c.y - y), abs(c.x - x))
                if d < bd:
                    best, bd = c, d
        return best

    # ================================================================ global
    def extract_global_features(self, orientation: np.ndarray, mask: np.ndarray,
                                wavelength: float = 0.0, quality: float = 0.0) -> Dict[str, Any]:
        cores, deltas = self.find_singular_points(orientation, mask)
        return {
            "core_points": cores,
            "delta_points": deltas,
            "pattern_type": self.classify_pattern(cores, deltas),
            "ridge_wavelength": round(float(wavelength), 2),
            "quality": round(float(quality), 3),
            "image_size": [int(mask.shape[0]), int(mask.shape[1])],
        }

    def find_singular_points(self, orientation: np.ndarray, mask: np.ndarray):
        """Cores (+1/2) and deltas (-1/2) from the Poincare index on a coarse grid."""
        s = self.config.singular_step
        # Block-average the orientation using doubled angles, then smooth.
        c2 = cv2.resize(np.cos(2 * orientation), None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
        s2 = cv2.resize(np.sin(2 * orientation), None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
        sig = self.config.singular_smooth_sigma
        c2 = cv2.GaussianBlur(c2, (0, 0), sig)
        s2 = cv2.GaussianBlur(s2, (0, 0), sig)
        theta = 0.5 * np.arctan2(s2, c2)

        small_mask = cv2.resize(mask.astype(np.uint8), (theta.shape[1], theta.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        inner = ndimage.binary_erosion(small_mask, iterations=2, border_value=0)

        # Closed path around each cell (counter-clockwise on screen).
        path = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)][::-1]
        h, w = theta.shape
        index = np.zeros_like(theta)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if not inner[i, j]:
                    continue
                angles = [theta[i + dy, j + dx] for dy, dx in path]
                total = 0.0
                for k in range(8):
                    d = angles[(k + 1) % 8] - angles[k]
                    d = (d + np.pi / 2) % np.pi - np.pi / 2   # wrap to [-pi/2, pi/2)
                    total += d
                index[i, j] = total / np.pi                    # half-turns around the cell

        # With y pointing down the screen, a core gives -1 half-turn and a delta +1
        # (verified on FVC2002 loop prints, where the core is visible).
        cores = self._cluster(np.argwhere(index < -0.5), s)
        deltas = self._cluster(np.argwhere(index > 0.5), s)
        return cores, deltas

    @staticmethod
    def _cluster(cells: np.ndarray, step: int) -> List[List[int]]:
        if len(cells) == 0:
            return []
        groups: List[List[np.ndarray]] = []
        for c in cells:
            for g in groups:
                if np.abs(g[0] - c).max() <= 2:
                    g.append(c)
                    break
            else:
                groups.append([c])
        out = []
        for g in groups:
            cy, cx = np.mean(g, axis=0)
            out.append([int(cx * step + step / 2), int(cy * step + step / 2)])
        return out

    @staticmethod
    def classify_pattern(cores, deltas) -> str:
        nc, nd = len(cores), len(deltas)
        if nc >= 2 or nd >= 2:
            return "whorl"
        if nc == 1:
            return "loop"
        if nc == 0 and nd == 0:
            return "arch"
        return "unknown"

    # ================================================================ helpers
    def extract_all(self, result) -> Tuple[List[Minutia], Dict[str, Any]]:
        """Run on a PreprocessResult."""
        minutiae = self.extract_minutiae(result.skeleton, result.orientation, result.mask, result.coherence)
        glob = self.extract_global_features(result.orientation, result.mask,
                                            result.median_wavelength, result.quality)
        glob["num_minutiae"] = len(minutiae)
        return minutiae, glob

    def get_statistics(self) -> Dict[str, Any]:
        counts = self.stats["minutiae_counts"]
        return {
            "images_processed": self.stats["images_processed"],
            "avg_minutiae": float(np.mean(counts)) if counts else 0.0,
        }
