from template import TemplateDatabase, FingerprintTemplate
from features import FeatureExtractor
from features import Minutia
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
# 7. FINGERPRINT MATCHER
# ============================================================================

class FingerprintMatcher:
    """
    Compares two fingerprint templates and computes a similarity score.
    Implements both 1:1 verification and 1:N identification.
    Uses a combination of minutiae-based matching and global feature filtering.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the matcher with configuration parameters.

        Args:
            config: Optional dictionary with parameters:
                - 'spatial_tolerance': maximum distance for minutiae match (pixels) – default 12
                - 'angular_tolerance': maximum orientation difference (radians) – default 0.5 (≈28°)
                - 'min_matches_ratio': minimum ratio of matches to total minutiae – default 0.35
                - 'max_refinement_iterations': iterations for alignment refinement – default 3
                - 'identification_candidate_limit': max candidates to return – default 10
        """
        self.config = {
            'spatial_tolerance': 12,  # pixels
            'angular_tolerance': 0.5,  # radians (~28 degrees)
            'min_matches_ratio': 0.35,  # minimum match score to be considered a match
            'max_refinement_iterations': 3,
            'identification_candidate_limit': 10
        }
        if config:
            self.config.update(config)

        # Statistics
        self.stats = {
            'verifications': 0,
            'identifications': 0,
            'match_scores': []
        }

    def match_templates(self, template1: FingerprintTemplate,
                        template2: FingerprintTemplate) -> Tuple[float, Dict]:
        """
        Compare two fingerprint templates and return a similarity score.

        Args:
            template1: First template (query)
            template2: Second template (reference)

        Returns:
            (score, details) where score is between 0 and 1, and details is a dict
            containing matching information (e.g., number of matched minutiae, alignment).
        """
        # Quick reject based on pattern type if they are known and different
        if (template1.get_pattern_type() != 'unknown' and
                template2.get_pattern_type() != 'unknown' and
                template1.get_pattern_type() != template2.get_pattern_type()):
            return 0.0, {'reason': 'pattern_mismatch'}

        # Quick reject if one template has very few minutiae
        if len(template1.minutiae) < 3 or len(template2.minutiae) < 3:
            return 0.0, {'reason': 'insufficient_minutiae'}

        # Perform minutiae-based matching
        score, details = self._match_minutiae(template1.minutiae, template2.minutiae)

        # Optionally incorporate global features into score (e.g., ridge density similarity)
        if score > 0:
            # Boost score slightly if ridge densities are similar
            rd1 = template1.global_features.get('ridge_density', 0)
            rd2 = template2.global_features.get('ridge_density', 0)
            density_sim = 1.0 - min(abs(rd1 - rd2), 0.3) / 0.3
            score = 0.9 * score + 0.1 * density_sim

        # Cap at 1.0
        score = min(score, 1.0)

        # Log score
        self.stats['match_scores'].append(score)

        return score, details

    def _match_minutiae(self, minutiae1: List[Minutia],
                        minutiae2: List[Minutia]) -> Tuple[float, Dict]:
        """
        Core minutiae matching algorithm using geometric alignment.
        Tries several candidate alignments based on minutiae pairs.
        """
        # Convert minutiae to numpy arrays for easier manipulation
        m1 = np.array([(m.x, m.y, m.orientation) for m in minutiae1])
        m2 = np.array([(m.x, m.y, m.orientation) for m in minutiae2])

        best_score = 0.0
        best_alignment = None
        best_matches = []

        # Number of candidate minutiae to try as reference points
        # Use at most 15 from each set to keep complexity manageable
        n1 = min(len(m1), 15)
        n2 = min(len(m2), 15)

        # For each pair of potential reference minutiae
        for i in range(n1):
            for j in range(n2):
                # Compute transformation aligning minutia i of set1 to minutia j of set2
                dx = m2[j, 0] - m1[i, 0]
                dy = m2[j, 1] - m1[i, 1]
                dtheta = m2[j, 2] - m1[i, 2]
                # Normalize angle difference to [-π, π]
                dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

                # Apply transformation to all minutiae in set1
                transformed = self._transform_minutiae(m1, dx, dy, dtheta)

                # Count matching minutiae
                matches, matched_pairs = self._count_matches(transformed, m2)

                # Score = 2 * matches / (|A| + |B|)
                score = 2.0 * matches / (len(m1) + len(m2))

                if score > best_score:
                    best_score = score
                    best_alignment = (dx, dy, dtheta)
                    best_matches = matched_pairs

        # Optional: refine alignment using the best set of matches
        if best_score > 0.1 and len(best_matches) >= 3:
            refined_score, refined_matches = self._refine_alignment(
                m1, m2, best_matches, best_alignment)
            if refined_score > best_score:
                best_score = refined_score
                best_matches = refined_matches

        details = {
            'score': best_score,
            'matched_minutiae': len(best_matches),
            'total_minutiae_1': len(m1),
            'total_minutiae_2': len(m2),
            'alignment': best_alignment
        }

        return best_score, details

    def _transform_minutiae(self, minutiae: np.ndarray, dx: float, dy: float,
                            dtheta: float) -> np.ndarray:
        """
        Apply translation and rotation to a set of minutiae.
        Input: array of shape (n, 3) where columns are x, y, theta.
        Returns transformed array.
        """
        if len(minutiae) == 0:
            return minutiae

        transformed = minutiae.copy()
        cos_t = np.cos(dtheta)
        sin_t = np.sin(dtheta)

        # Rotate and translate
        transformed[:, 0] = minutiae[:, 0] * cos_t - minutiae[:, 1] * sin_t + dx
        transformed[:, 1] = minutiae[:, 0] * sin_t + minutiae[:, 1] * cos_t + dy
        # Rotate angles
        transformed[:, 2] = (minutiae[:, 2] + dtheta) % (2 * np.pi)

        return transformed

    def _count_matches(self, set1: np.ndarray, set2: np.ndarray) -> Tuple[int, List]:
        """
        Count matching minutiae between two sets after alignment.
        Uses nearest neighbor search with tolerance.
        Returns (number_of_matches, list_of_matched_pairs).
        Each matched pair is (idx1, idx2).
        """
        if len(set1) == 0 or len(set2) == 0:
            return 0, []

        matches = []
        used2 = [False] * len(set2)

        # For each minutia in set1, find closest in set2 within tolerances
        for i in range(len(set1)):
            best_dist = float('inf')
            best_j = -1
            for j in range(len(set2)):
                if used2[j]:
                    continue
                # Spatial distance
                dx = set1[i, 0] - set2[j, 0]
                dy = set1[i, 1] - set2[j, 1]
                dist = np.hypot(dx, dy)
                if dist > self.config['spatial_tolerance']:
                    continue
                # Angular difference
                da = abs(set1[i, 2] - set2[j, 2])
                da = min(da, 2 * np.pi - da)
                if da > self.config['angular_tolerance']:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j != -1:
                matches.append((i, best_j))
                used2[best_j] = True

        return len(matches), matches

    def _refine_alignment(self, m1: np.ndarray, m2: np.ndarray,
                          initial_matches: List, initial_alignment: Tuple) -> Tuple[float, List]:
        """
        Refine alignment using the set of matched points to compute a better transformation.
        Uses least squares to estimate optimal rigid transformation.
        """
        if len(initial_matches) < 3:
            return 0.0, []

        # Extract matched point pairs
        pts1 = np.array([m1[i][:2] for i, _ in initial_matches])
        pts2 = np.array([m2[j][:2] for _, j in initial_matches])

        # Compute optimal rotation and translation using Procrustes analysis
        # Center the points
        mean1 = np.mean(pts1, axis=0)
        mean2 = np.mean(pts2, axis=0)
        pts1_centered = pts1 - mean1
        pts2_centered = pts2 - mean2

        # Compute covariance matrix
        H = pts1_centered.T @ pts2_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = mean2 - R @ mean1

        # Extract rotation angle from R
        dtheta = np.arctan2(R[1, 0], R[0, 0])

        # Apply refined transformation
        transformed = self._transform_minutiae(m1, t[0], t[1], dtheta)

        # Count matches with refined alignment
        matches, matched_pairs = self._count_matches(transformed, m2)
        score = 2.0 * matches / (len(m1) + len(m2))

        return score, matched_pairs

    def verify(self, template1: FingerprintTemplate,
               template2: FingerprintTemplate,
               threshold: float = None) -> Tuple[bool, float, Dict]:
        """
        1:1 verification.

        Args:
            template1: Query template
            template2: Reference template
            threshold: Score threshold for acceptance (if None, uses config)

        Returns:
            (is_match, score, details)
        """
        if threshold is None:
            threshold = self.config['min_matches_ratio']

        score, details = self.match_templates(template1, template2)
        is_match = score >= threshold

        self.stats['verifications'] += 1

        return is_match, score, details

    def identify(self, query_template: FingerprintTemplate,
                 database_templates: List[FingerprintTemplate],
                 threshold: float = None,
                 top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """
        1:N identification against a list of reference templates.

        Args:
            query_template: Template to identify
            database_templates: List of all templates in database
            threshold: Minimum score to consider a match (if None, uses config)
            top_k: Maximum number of candidates to return (if None, uses config)

        Returns:
            List of tuples (person_id, score, details) sorted by score descending.
            If a match is found, the first element is the matched identity.
        """
        if threshold is None:
            threshold = self.config['min_matches_ratio']
        if top_k is None:
            top_k = self.config['identification_candidate_limit']

        # Quick filter: first match by pattern type to reduce candidates
        # We'll still compute scores for all, but we can use pattern type as early filter
        # For simplicity, we compute scores for all
        scores = []
        for ref in database_templates:
            score, details = self.match_templates(query_template, ref)
            scores.append((ref.person_id, score, details))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        self.stats['identifications'] += 1

        return scores[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Return matching statistics."""
        stats = self.stats.copy()
        if stats['match_scores']:
            stats['avg_match_score'] = np.mean(stats['match_scores'])
            stats['min_match_score'] = np.min(stats['match_scores'])
            stats['max_match_score'] = np.max(stats['match_scores'])
        return stats