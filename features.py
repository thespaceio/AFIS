import cv2
import numpy as np
import sqlite3
import pickle
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
import warnings




# ============================================================================
# 3. MINUTIA DATA STRUCTURE
# ============================================================================

@dataclass
class Minutia:
    """
    Represents a single minutia point extracted from a fingerprint.

    Attributes:
        x: X-coordinate (column) in the image
        y: Y-coordinate (row) in the image
        orientation: Direction of the minutia in radians (0 to 2π)
        minutiae_type: Type of minutia - 'ending' or 'bifurcation'
        confidence: Quality/confidence score between 0 and 1
    """
    x: int
    y: int
    orientation: float
    minutiae_type: str  # 'ending' or 'bifurcation'
    confidence: float = 1.0

    def __post_init__(self):
        # Normalize orientation to [0, 2π)
        self.orientation = self.orientation % (2 * np.pi)

        # Validate type
        if self.minutiae_type not in ('ending', 'bifurcation'):
            raise ValueError(f"Invalid minutia type: {self.minutiae_type}")

        # Clamp confidence
        self.confidence = max(0.0, min(1.0, self.confidence))

    def distance_to(self, other: 'Minutia') -> float:
        """Euclidean distance to another minutia."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_difference(self, other: 'Minutia') -> float:
        """
        Minimum angular difference between orientations (radians).
        Returns value in [0, π].
        """
        diff = abs(self.orientation - other.orientation) % (2 * np.pi)
        return min(diff, 2 * np.pi - diff)

    def to_tuple(self) -> tuple:
        """Convert to a simple tuple for serialization."""
        return (self.x, self.y, self.orientation, self.minutiae_type, self.confidence)

    @classmethod
    def from_tuple(cls, data: tuple) -> 'Minutia':
        """Create a Minutia from a tuple (as produced by to_tuple)."""
        return cls(x=data[0], y=data[1], orientation=data[2],
                   minutiae_type=data[3], confidence=data[4])


# ============================================================================
# 4. FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """
    Extracts minutiae and global features from a preprocessed fingerprint.
    Uses the skeleton image and orientation field to locate ridge endings
    and bifurcations. Also identifies singular points (core/delta) and
    classifies the fingerprint pattern.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature extractor with configuration.

        Args:
            config: Optional dictionary with parameters:
                - 'minutia_min_distance': minimum distance between minutiae (default 8)
                - 'border_margin': distance from border to ignore (default 15)
                - 'ending_cn': crossing number for ridge ending (default 1)
                - 'bifurcation_cn': crossing number for bifurcation (default 3)
                - 'max_minutiae': maximum number of minutiae to keep (default 150)
        """
        self.config = {
            'minutia_min_distance': 8,
            'border_margin': 15,
            'ending_cn': 1,
            'bifurcation_cn': 3,
            'max_minutiae': 150,
            'quality_threshold': 0.3,
        }
        if config:
            self.config.update(config)

        # 8-neighbour offsets (clockwise from top-left)
        self.neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0),
            (1, -1), (0, -1)
        ]

        # Statistics
        self.stats = {
            'images_processed': 0,
            'avg_minutiae': 0,
            'minutiae_counts': []
        }

    def extract_minutiae(self, skeleton: np.ndarray, orientation: np.ndarray) -> List[Minutia]:
        """
        Extract all minutiae points from the skeleton image.

        Args:
            skeleton: Binary skeleton image (255 = ridge, 0 = background)
            orientation: Orientation field (radians)

        Returns:
            List of Minutia objects, sorted by confidence
        """
        rows, cols = skeleton.shape
        minutiae = []

        # Pad the skeleton to handle borders easily
        padded = np.pad(skeleton, pad_width=1, mode='constant', constant_values=0)

        # Scan every pixel of the original skeleton
        for y in range(rows):
            for x in range(cols):
                if skeleton[y, x] == 0:
                    continue  # not a ridge pixel

                # Get 8 neighbours from padded image
                neighbours = []
                for dy, dx in self.neighbor_offsets:
                    ny, nx = y + 1 + dy, x + 1 + dx  # +1 because of padding
                    neighbours.append(1 if padded[ny, nx] > 0 else 0)

                # Compute crossing number (CN)
                cn = 0
                for i in range(8):
                    if neighbours[i] == 1 and neighbours[(i + 1) % 8] == 0:
                        cn += 1

                # Determine minutia type based on CN
                mtype = None
                if cn == self.config['ending_cn']:
                    mtype = 'ending'
                elif cn == self.config['bifurcation_cn']:
                    mtype = 'bifurcation'
                else:
                    continue  # not a minutia

                # Calculate orientation at this point (from orientation field)
                orient = orientation[y, x]

                # Refine orientation for endings (follow ridge direction)
                if mtype == 'ending':
                    orient = self._refine_ending_orientation(skeleton, x, y, orient)

                # Estimate confidence (based on local ridge quality)
                confidence = self._estimate_minutia_confidence(skeleton, x, y, neighbours)

                # Create minutia object
                minutiae.append(Minutia(
                    x=x, y=y,
                    orientation=orient,
                    minutiae_type=mtype,
                    confidence=confidence
                ))

        # Remove spurious minutiae
        minutiae = self._filter_spurious_minutiae(minutiae, rows, cols)

        # Sort by confidence and limit count
        minutiae.sort(key=lambda m: m.confidence, reverse=True)
        if len(minutiae) > self.config['max_minutiae']:
            minutiae = minutiae[:self.config['max_minutiae']]

        # Update statistics
        self.stats['images_processed'] += 1
        self.stats['minutiae_counts'].append(len(minutiae))
        self.stats['avg_minutiae'] = np.mean(self.stats['minutiae_counts'])

        return minutiae

    def _refine_ending_orientation(self, skeleton: np.ndarray, x: int, y: int,
                                   initial_orient: float, trace_length: int = 10) -> float:
        """
        For a ridge ending, trace the ridge backward to get a more accurate direction.
        """
        # Simple implementation: walk along the ridge for a few steps
        # Use the direction opposite to the ridge continuation
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        # Find the neighbor that is also a ridge pixel (the ridge continues in one direction)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]:
                if skeleton[ny, nx] > 0:
                    # Found the continuation direction
                    # The ending orientation should point away from this direction
                    angle = np.arctan2(-dy, -dx)  # opposite direction
                    return angle % (2 * np.pi)
        # Fallback to orientation field
        return initial_orient

    def _estimate_minutia_confidence(self, skeleton: np.ndarray, x: int, y: int,
                                     neighbours: List[int]) -> float:
        """
        Estimate confidence based on local ridge clarity.
        Simple heuristic: number of ridge neighbours and proximity to other minutiae.
        """
        # Base confidence
        conf = 0.7

        # Check local ridge continuity – count ridge neighbours in a small window
        window_size = 5
        half = window_size // 2
        window = skeleton[max(0, y - half):min(skeleton.shape[0], y + half + 1),
                 max(0, x - half):min(skeleton.shape[1], x + half + 1)]
        ridge_pixels = np.sum(window > 0)
        # If the window is mostly empty, confidence drops
        if ridge_pixels < 3:
            conf *= 0.5
        elif ridge_pixels < 5:
            conf *= 0.8

        return min(conf, 1.0)

    def _filter_spurious_minutiae(self, minutiae: List[Minutia],
                                  img_height: int, img_width: int) -> List[Minutia]:
        """
        Remove minutiae that are too close to the image border or too close to each other.
        """
        if not minutiae:
            return []

        # 1. Remove border minutiae
        border = self.config['border_margin']
        valid = []
        for m in minutiae:
            if (border <= m.x < img_width - border and
                    border <= m.y < img_height - border):
                valid.append(m)

        # 2. Remove duplicates or very close minutiae (keep the one with higher confidence)
        valid.sort(key=lambda m: m.confidence, reverse=True)
        filtered = []
        for m in valid:
            too_close = False
            for kept in filtered:
                if m.distance_to(kept) < self.config['minutia_min_distance']:
                    too_close = True
                    break
            if not too_close:
                filtered.append(m)

        return filtered

    def extract_global_features(self, skeleton: np.ndarray,
                                orientation: np.ndarray) -> Dict[str, Any]:
        """
        Extract global features: core/delta points, pattern type, ridge count, etc.

        Args:
            skeleton: Binary skeleton image
            orientation: Orientation field

        Returns:
            Dictionary containing:
                - 'core_points': list of (x, y) tuples for detected cores
                - 'delta_points': list of (x, y) tuples for detected deltas
                - 'pattern_type': string ('arch', 'loop', 'whorl', 'tented_arch')
                - 'ridge_density': average number of ridges per unit area
                - 'num_minutiae': total minutiae count (will be filled later)
        """
        rows, cols = skeleton.shape

        # 1. Find singular points (core & delta) using Poincaré index
        cores, deltas = self._find_singular_points(orientation)

        # 2. Classify pattern based on singular points
        pattern = self._classify_pattern(cores, deltas)

        # 3. Compute ridge density (approx)
        ridge_pixels = np.sum(skeleton > 0)
        total_pixels = rows * cols
        ridge_density = ridge_pixels / total_pixels if total_pixels > 0 else 0

        return {
            'core_points': cores,
            'delta_points': deltas,
            'pattern_type': pattern,
            'ridge_density': float(ridge_density),
            'image_size': (rows, cols),
            'num_minutiae': 0  # placeholder, will be updated after extraction
        }

    def _find_singular_points(self, orientation: np.ndarray,
                              block_size: int = 16) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Detect core and delta points using the Poincaré index method.

        Args:
            orientation: Orientation field (radians)
            block_size: Size of the block for Poincaré calculation

        Returns:
            (list_of_cores, list_of_deltas) as (x, y) coordinates.
        """
        rows, cols = orientation.shape
        cores = []
        deltas = []

        # We'll compute Poincaré index on a grid of points
        step = max(2, block_size // 2)
        for y in range(step, rows - step, step):
            for x in range(step, cols - step, step):
                # Define a closed path around (x, y) using 8 points on a circle
                radius = block_size // 2
                angles = []
                for i in range(8):
                    theta = i * (2 * np.pi / 8)
                    nx = int(x + radius * np.cos(theta))
                    ny = int(y + radius * np.sin(theta))
                    if 0 <= nx < cols and 0 <= ny < rows:
                        angles.append(orientation[ny, nx])
                    else:
                        break
                else:  # all 8 points inside image
                    poincare = 0
                    for i in range(8):
                        diff = angles[(i + 1) % 8] - angles[i]
                        # Wrap difference to [-π, π]
                        diff = (diff + np.pi) % (2 * np.pi) - np.pi
                        poincare += diff
                    poincare /= (2 * np.pi)

                    # Poincaré index ≈ 0.5 → core, ≈ -0.5 → delta
                    if abs(poincare - 0.5) < 0.25:
                        cores.append((x, y))
                    elif abs(poincare + 0.5) < 0.25:
                        deltas.append((x, y))

        # Remove duplicates (cluster nearby points)
        cores = self._cluster_points(cores, distance=block_size)
        deltas = self._cluster_points(deltas, distance=block_size)

        return cores, deltas

    def _cluster_points(self, points: List[Tuple[int, int]], distance: int) -> List[Tuple[int, int]]:
        """Average points that are too close together."""
        if not points:
            return []
        # Simple clustering: keep the first and average others within distance
        clusters = []
        used = [False] * len(points)
        for i in range(len(points)):
            if used[i]:
                continue
            cluster = [points[i]]
            used[i] = True
            for j in range(i + 1, len(points)):
                if not used[j]:
                    dx = points[i][0] - points[j][0]
                    dy = points[i][1] - points[j][1]
                    if dx * dx + dy * dy < distance * distance:
                        cluster.append(points[j])
                        used[j] = True
            # Compute centroid
            avg_x = int(np.mean([p[0] for p in cluster]))
            avg_y = int(np.mean([p[1] for p in cluster]))
            clusters.append((avg_x, avg_y))
        return clusters

    def _classify_pattern(self, cores: List[Tuple[int, int]],
                          deltas: List[Tuple[int, int]]) -> str:
        """
        Classify fingerprint pattern based on number of cores/deltas.
        Simple heuristic, not exhaustive.
        """
        num_cores = len(cores)
        num_deltas = len(deltas)

        if num_cores == 0 and num_deltas == 0:
            return 'arch'
        elif num_cores == 1 and num_deltas == 0:
            return 'loop'
        elif num_cores == 1 and num_deltas == 1:
            # Check relative positions to distinguish loop vs whorl
            # If core and delta are far apart, likely loop; if close, maybe whorl?
            # For simplicity, we'll just call it loop (can be improved)
            return 'loop'  # or 'whorl' if you have a better heuristic
        elif num_cores == 2 and num_deltas == 2:
            return 'whorl'
        elif num_cores == 0 and num_deltas == 1:
            return 'tented_arch'
        else:
            return 'unknown'

    def extract_all(self, skeleton: np.ndarray,
                    orientation: np.ndarray) -> Tuple[List[Minutia], Dict[str, Any]]:
        """
        Convenience method to extract both minutiae and global features.
        Returns:
            (minutiae_list, global_features)
        """
        minutiae = self.extract_minutiae(skeleton, orientation)
        global_features = self.extract_global_features(skeleton, orientation)
        global_features['num_minutiae'] = len(minutiae)
        return minutiae, global_features

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about extracted features."""
        stats = self.stats.copy()
        if stats['minutiae_counts']:
            stats['min_minutiae'] = np.min(stats['minutiae_counts'])
            stats['max_minutiae'] = np.max(stats['minutiae_counts'])
        return stats