"""
Minutiae matching.

1. Local structures: every minutia is described by its k nearest neighbours
   (distance, direction of the neighbour seen from the minutia, and relative
   minutia direction). These values don't change when the finger is moved or
   rotated, so they can be compared between two impressions directly.
2. The most similar local structures give candidate alignments. Each
   candidate is a rigid transform (rotation about the minutia + translation)
   that maps one minutia onto its partner.
3. Each alignment is applied to all minutiae; pairs within a distance and
   angle tolerance are matched one-to-one. The best alignment is refined
   with a least-squares fit (Kabsch) on the matched pairs.
4. Score = m^2 / (n1 * n2), where m is the number of matched minutiae and
   n1, n2 are the minutiae counts inside the overlapping area of the two
   prints. 0 = nothing in common, 1 = identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .features import Minutia
from .template import FingerprintTemplate


@dataclass
class MatcherConfig:
    neighbours: int = 6                  # k nearest neighbours per local structure
    local_dist_tol: float = 10.0         # px
    local_angle_tol: float = np.radians(25)
    candidates: int = 12                 # alignments tried
    match_dist_tol: float = 12.0         # px after alignment
    match_angle_tol: float = np.radians(25)
    min_overlap_count: int = 18          # floor for n1, n2 in the score (stops tiny overlaps scoring high)
    refine_iterations: int = 2
    threshold: float = 0.18              # accept if score >= threshold; chosen with scripts/evaluate.py


def _wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class FingerprintMatcher:
    def __init__(self, config: Optional[MatcherConfig] = None):
        self.config = config or MatcherConfig()
        self.stats: Dict[str, list | int] = {"verifications": 0, "identifications": 0, "match_scores": []}

    # ------------------------------------------------------------------ public
    def match_templates(self, t1: FingerprintTemplate, t2: FingerprintTemplate) -> Tuple[float, Dict]:
        score, details = self.match_minutiae(t1.minutiae, t2.minutiae)
        self.stats["match_scores"].append(score)
        return score, details

    def verify(self, query: FingerprintTemplate, reference: FingerprintTemplate,
               threshold: Optional[float] = None) -> Tuple[bool, float, Dict]:
        thr = self.config.threshold if threshold is None else threshold
        score, details = self.match_templates(query, reference)
        self.stats["verifications"] += 1
        return score >= thr, score, details

    def identify(self, query: FingerprintTemplate, references: Sequence[FingerprintTemplate],
                 top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Best score per person, highest first."""
        best: Dict[str, Tuple[float, Dict]] = {}
        for ref in references:
            score, details = self.match_templates(query, ref)
            details = dict(details, fingerprint_id=ref.fingerprint_id)
            if ref.person_id not in best or score > best[ref.person_id][0]:
                best[ref.person_id] = (score, details)
        self.stats["identifications"] += 1
        ranked = sorted(((pid, s, d) for pid, (s, d) in best.items()), key=lambda r: r[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------ core
    def match_minutiae(self, m1: List[Minutia], m2: List[Minutia]) -> Tuple[float, Dict]:
        empty = {"matched_minutiae": 0, "pairs": [], "total_minutiae_1": len(m1),
                 "total_minutiae_2": len(m2), "alignment": None}
        if len(m1) < 3 or len(m2) < 3:
            return 0.0, dict(empty, reason="insufficient_minutiae")

        a = self._as_array(m1)
        b = self._as_array(m2)
        types_a = np.array([m.minutiae_type == "ending" for m in m1])
        types_b = np.array([m.minutiae_type == "ending" for m in m2])

        sim = self._local_similarity(a, b)
        order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]

        best = (0, [], None)
        tried = 0
        for i, j in order:
            if sim[i, j] <= 0 or tried >= self.config.candidates:
                break
            tried += 1
            rot = _wrap(b[j, 2] - a[i, 2])
            R = self._rot(rot)
            t = b[j, :2] - R @ a[i, :2]
            pairs = self._pair(self._apply(a, R, t, rot), b)
            if len(pairs) > len(best[1]):
                best = (len(pairs), pairs, (R, t, rot))

        if best[2] is None:
            return 0.0, dict(empty, reason="no_alignment")

        _, pairs, (R, t, rot) = best
        for _ in range(self.config.refine_iterations):
            if len(pairs) < 3:
                break
            R2, t2, rot2 = self._kabsch(a[[p[0] for p in pairs], :2], b[[p[1] for p in pairs], :2])
            new_pairs = self._pair(self._apply(a, R2, t2, rot2), b)
            if len(new_pairs) < len(pairs):
                break
            pairs, R, t, rot = new_pairs, R2, t2, rot2

        moved = self._apply(a, R, t, rot)
        n1 = self._count_inside(moved[:, :2], b[:, :2])
        n2 = self._count_inside(b[:, :2], moved[:, :2])
        floor = self.config.min_overlap_count
        m = len(pairs)
        score = (m * m) / (max(n1, floor, m) * max(n2, floor, m))

        same_type = sum(types_a[i] == types_b[j] for i, j in pairs)
        details = {
            "matched_minutiae": m,
            "pairs": [(int(i), int(j)) for i, j in pairs],
            "overlap_minutiae_1": int(n1),
            "overlap_minutiae_2": int(n2),
            "total_minutiae_1": len(m1),
            "total_minutiae_2": len(m2),
            "same_type_pairs": int(same_type),
            "alignment": {"rotation_deg": round(float(np.degrees(rot)), 1),
                          "dx": round(float(t[0]), 1), "dy": round(float(t[1]), 1)},
        }
        return float(min(1.0, score)), details

    # ------------------------------------------------------------------ steps
    @staticmethod
    def _as_array(ms: List[Minutia]) -> np.ndarray:
        return np.array([(m.x, m.y, m.orientation) for m in ms], dtype=np.float64)

    def _local_structures(self, a: np.ndarray):
        """(n, k, 3) array of [distance, radial angle, relative direction] to k nearest neighbours."""
        k = min(self.config.neighbours, len(a) - 1)
        d = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1])
        np.fill_diagonal(d, np.inf)
        nn = np.argsort(d, axis=1)[:, :k]
        rows = np.arange(len(a))[:, None]
        dist = d[rows, nn]
        dx = a[nn, 0] - a[:, None, 0]
        dy = a[nn, 1] - a[:, None, 1]
        radial = _wrap(np.arctan2(dy, dx) - a[:, None, 2])
        relative = _wrap(a[nn, 2] - a[:, None, 2])
        return np.stack([dist, radial, relative], axis=-1)

    def _local_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fraction of neighbours that agree, for every minutia pair (n1 x n2)."""
        la, lb = self._local_structures(a), self._local_structures(b)
        c = self.config
        # Compare every neighbour of i with every neighbour of j: shape (n1, n2, k1, k2)
        dd = np.abs(la[:, None, :, None, 0] - lb[None, :, None, :, 0])
        dr = np.abs(_wrap(la[:, None, :, None, 1] - lb[None, :, None, :, 1]))
        do = np.abs(_wrap(la[:, None, :, None, 2] - lb[None, :, None, :, 2]))
        ok = (dd < c.local_dist_tol) & (dr < c.local_angle_tol) & (do < c.local_angle_tol)
        cost = np.where(ok, dd / c.local_dist_tol + dr / c.local_angle_tol + do / c.local_angle_tol, np.inf)
        # Greedy-free approximation: each neighbour of i counts if it has some partner in j.
        agree = np.isfinite(cost.min(axis=3)).sum(axis=2)
        k = max(1, min(la.shape[1], lb.shape[1]))
        return agree / k

    @staticmethod
    def _rot(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    @staticmethod
    def _apply(a: np.ndarray, R: np.ndarray, t: np.ndarray, rot: float) -> np.ndarray:
        out = a.copy()
        out[:, :2] = a[:, :2] @ R.T + t
        out[:, 2] = (a[:, 2] + rot) % (2 * np.pi)
        return out

    def _pair(self, a: np.ndarray, b: np.ndarray) -> List[Tuple[int, int]]:
        """One-to-one pairs within tolerance, closest first."""
        c = self.config
        d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
        da = np.abs(_wrap(a[:, None, 2] - b[None, :, 2]))
        valid = (d <= c.match_dist_tol) & (da <= c.match_angle_tol)
        ii, jj = np.nonzero(valid)
        order = np.argsort(d[ii, jj] + 5 * da[ii, jj])
        used_a, used_b, pairs = set(), set(), []
        for k in order:
            i, j = int(ii[k]), int(jj[k])
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                pairs.append((i, j))
        return pairs

    @staticmethod
    def _kabsch(p: np.ndarray, q: np.ndarray):
        """Best rotation R and translation t with R @ p + t ~= q."""
        mp, mq = p.mean(axis=0), q.mean(axis=0)
        H = (p - mp).T @ (q - mq)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        t = mq - R @ mp
        return R, t, float(np.arctan2(R[1, 0], R[0, 0]))

    def _count_inside(self, pts: np.ndarray, region_pts: np.ndarray, margin: float = 15.0) -> int:
        """How many of `pts` fall inside the convex hull of `region_pts` (plus a margin)."""
        if len(region_pts) < 3:
            return len(pts)
        hull = cv2.convexHull(region_pts.astype(np.float32))
        return int(sum(cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -margin for x, y in pts))

    def get_statistics(self) -> Dict:
        scores = self.stats["match_scores"]
        return {
            "verifications": self.stats["verifications"],
            "identifications": self.stats["identifications"],
            "avg_match_score": float(np.mean(scores)) if scores else 0.0,
        }
