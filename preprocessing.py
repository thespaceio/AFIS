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
# 2. PREPROCESSING PIPELINE
# ============================================================================


class PreprocessingPipeline:
    """
    Complete fingerprint image preprocessing and enhancement pipeline.
    Transforms raw fingerprint images into clean, enhanced ridge maps suitable
    for feature extraction. The pipeline consists of sequential operations that
    progressively clean and clarify the fingerprint pattern.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessing pipeline with configuration parameters.

        Args:
            config: Optional dictionary of configuration parameters. If None,
                   default values are used.
        """
        # Default configuration
        self.config = {
            # Noise reduction parameters
            'denoise_strength': 10,  # h parameter for non-local means
            'denoise_template_size': 7,  # Size of template patch
            'denoise_search_size': 21,  # Size of search window

            # Contrast enhancement
            'clahe_clip_limit': 2.0,  # Contrast limiting for CLAHE
            'clahe_tile_size': (8, 8),  # Tile size for CLAHE

            # Orientation estimation
            'orientation_block_size': 16,  # Block size for orientation field
            'orientation_smooth_sigma': 1.5,  # Gaussian smoothing sigma

            # Frequency estimation
            'frequency_block_size': 32,  # Block size for frequency map

            # Gabor filter parameters
            'gabor_kernel_size': 21,  # Size of Gabor kernels
            'gabor_sigma': 4.0,  # Standard deviation
            'gabor_gamma': 0.5,  # Spatial aspect ratio
            'gabor_psi': 0,  # Phase offset

            # Binarization
            'adaptive_block_size': 11,  # Block size for adaptive threshold
            'adaptive_c': 2,  # Constant subtracted from mean

            # Morphological operations
            'morph_kernel_size': 3,  # Size of structuring element

            # Quality thresholds
            'min_ridge_frequency': 0.05,  # Minimum valid ridge frequency
            'max_ridge_frequency': 0.2,  # Maximum valid ridge frequency
        }

        # Update with user configuration if provided
        if config:
            self.config.update(config)

        # Internal state tracking
        self.processing_stats = {
            'processed_images': 0,
            'processing_times': [],
            'quality_scores': []
        }

        # Precompute Gabor filter bank for efficiency
        self._init_gabor_filters()

    def _init_gabor_filters(self) -> None:
        """Initialize a bank of Gabor filters at different orientations."""
        self.gabor_filters = []
        kernel_size = self.config['gabor_kernel_size']
        sigma = self.config['gabor_sigma']
        gamma = self.config['gabor_gamma']
        psi = self.config['gabor_psi']

        # Create filters at 8 different orientations (0 to 157.5 degrees in steps of 22.5)
        for theta in np.arange(0, np.pi, np.pi / 8):
            kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma,
                theta,
                1.0,  # Lambda (wavelength) - will be adjusted per frequency
                gamma,
                psi,
                ktype=cv2.CV_32F
            )
            # Normalize kernel to prevent excessive brightness
            kernel /= 1.5 * np.sum(np.abs(kernel))
            self.gabor_filters.append((theta, kernel))

    def preprocess(self, raw_image: np.ndarray) -> Dict[str, Any]:
        """
        Apply complete preprocessing pipeline to a fingerprint image.

        Args:
            raw_image: Grayscale fingerprint image

        Returns:
            Dictionary containing all intermediate processing results

        Raises:
            ValueError: If input image is invalid
        """
        # Validate input
        if not isinstance(raw_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if len(raw_image.shape) != 2:
            raise ValueError("Input must be a 2D grayscale image")

        start_time = datetime.now()

        try:
            # Step 1: Noise reduction
            denoised = self._reduce_noise(raw_image)

            # Step 2: Contrast normalization
            normalized = self._normalize_contrast(denoised)

            # Step 3: Orientation field estimation
            orientation_field = self._estimate_orientation(normalized)

            # Step 4: Frequency map estimation
            frequency_map = self._estimate_frequency(normalized, orientation_field)

            # Step 5: Ridge enhancement using Gabor filters
            enhanced = self._enhance_ridges(normalized, orientation_field, frequency_map)

            # Step 6: Binarization (convert to black/white)
            binary = self._binarize_image(enhanced)

            # Step 7: Morphological cleanup
            cleaned = self._clean_binary_image(binary)

            # Step 8: Skeletonization (thinning to single-pixel width)
            skeleton = self._skeletonize_image(cleaned)

            # Step 9: Calculate quality metrics
            quality_score = self._calculate_quality_score(
                enhanced, skeleton, orientation_field, frequency_map
            )

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats['processed_images'] += 1
            self.processing_stats['processing_times'].append(processing_time)
            self.processing_stats['quality_scores'].append(quality_score)

            # Compile all results
            results = {
                'raw': raw_image,
                'denoised': denoised,
                'normalized': normalized,
                'orientation': orientation_field,
                'frequency': frequency_map,
                'enhanced': enhanced,
                'binary': binary,
                'cleaned': cleaned,
                'skeleton': skeleton,
                'quality_score': quality_score,
                'processing_time': processing_time
            }

            return results

        except Exception as e:
            # Log error and re-raise
            error_msg = f"Preprocessing failed: {str(e)}"
            self._log_error(error_msg, raw_image.shape)
            raise RuntimeError(error_msg) from e

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction while preserving ridge structures.

        Args:
            image: Input grayscale image

        Returns:
            Denoised image
        """
        # Apply non-local means denoising (preserves edges better than Gaussian)
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=self.config['denoise_strength'],
            templateWindowSize=self.config['denoise_template_size'],
            searchWindowSize=self.config['denoise_search_size']
        )

        # Additional median filter for salt-and-pepper noise
        # Note: kernel size must be odd
        denoised = cv2.medianBlur(denoised, 3)

        return denoised

    def _normalize_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image contrast using adaptive histogram equalization.

        Args:
            image: Input grayscale image

        Returns:
            Contrast-normalized image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_tile_size']
        )
        normalized = clahe.apply(image)

        # Additional global normalization to ensure full dynamic range
        normalized = cv2.normalize(
            normalized,
            None,
            alpha=0,  # Minimum value
            beta=255,  # Maximum value
            norm_type=cv2.NORM_MINMAX
        )

        return normalized.astype(np.uint8)

    def _estimate_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate ridge orientation field using gradient information.

        Args:
            image: Contrast-normalized grayscale image

        Returns:
            Orientation field in radians (same shape as input)
        """
        rows, cols = image.shape
        block_size = self.config['orientation_block_size']

        # Compute image gradients using Sobel operator
        # dx: gradient in x-direction, dy: gradient in y-direction
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Initialize orientation field
        orientation = np.zeros((rows, cols), dtype=np.float64)

        # Calculate orientation block by block
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                # Define block boundaries
                i_end = min(i + block_size, rows)
                j_end = min(j + block_size, cols)

                # Extract gradient blocks
                block_dx = dx[i:i_end, j:j_end]
                block_dy = dy[i:i_end, j:j_end]

                # Compute orientation using gradient information
                # Gxx, Gyy, Gxy are sums of squared gradients
                Gxx = np.sum(block_dx ** 2)
                Gyy = np.sum(block_dy ** 2)
                Gxy = np.sum(block_dx * block_dy)

                # Avoid division by zero
                if Gxx - Gyy == 0 and Gxy == 0:
                    theta = 0
                else:
                    # Orientation calculation formula
                    theta = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy) + (np.pi / 2)

                # Assign orientation to entire block
                orientation[i:i_end, j:j_end] = theta

        # Smooth orientation field using Gaussian blur
        # This creates a more coherent flow field
        sigma = self.config['orientation_smooth_sigma']
        orientation = cv2.GaussianBlur(orientation, (5, 5), sigma)

        return orientation

    def _estimate_frequency(self, image: np.ndarray,
                            orientation: np.ndarray) -> np.ndarray:
        """
        Estimate ridge frequency (distance between ridges).

        Args:
            image: Enhanced grayscale image
            orientation: Orientation field in radians

        Returns:
            Frequency map (inverse of ridge spacing)
        """
        rows, cols = image.shape
        block_size = self.config['frequency_block_size']
        frequency = np.zeros((rows, cols), dtype=np.float64)

        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                i_end = min(i + block_size, rows)
                j_end = min(j + block_size, cols)

                # Extract image and orientation blocks
                block_image = image[i:i_end, j:j_end]
                block_orientation = orientation[i:i_end, j:j_end]

                # Estimate frequency for this block
                freq = self._estimate_block_frequency(block_image, block_orientation)

                # Assign frequency to entire block
                frequency[i:i_end, j:j_end] = freq

        return frequency

    def _estimate_block_frequency(self, block: np.ndarray,
                                  orientation: np.ndarray) -> float:
        """
        Estimate ridge frequency for a single block.

        Args:
            block: Image block
            orientation: Orientation block

        Returns:
            Estimated ridge frequency
        """
        block_rows, block_cols = block.shape

        # Use center of block for orientation reference
        center_row, center_col = block_rows // 2, block_cols // 2
        local_orientation = orientation[center_row, center_col]

        # Calculate perpendicular direction (across ridges)
        perp_angle = local_orientation + (np.pi / 2)

        # Sample intensity profile along perpendicular direction
        profile = []
        max_samples = min(block_cols, 50)  # Limit sampling length

        for k in range(max_samples):
            # Calculate sample position
            offset = k - (max_samples // 2)
            x = center_col + int(offset * np.cos(perp_angle))
            y = center_row + int(offset * np.sin(perp_angle))

            # Check bounds and add to profile
            if 0 <= x < block_cols and 0 <= y < block_rows:
                profile.append(block[y, x])

        # Need enough samples for reliable frequency estimation
        if len(profile) < 20:
            return 0.1  # Default frequency

        # Find peaks in intensity profile (ridge centers)
        peaks = []
        for idx in range(1, len(profile) - 1):
            if profile[idx] > profile[idx - 1] and profile[idx] > profile[idx + 1]:
                peaks.append(idx)

        # Need at least 2 peaks to estimate frequency
        if len(peaks) < 2:
            return 0.1  # Default frequency

        # Calculate distances between consecutive peaks
        peak_distances = np.diff(peaks)

        # Remove outliers (distances too small or too large)
        mean_distance = np.mean(peak_distances)
        std_distance = np.std(peak_distances)
        valid_distances = [d for d in peak_distances
                           if mean_distance - std_distance < d < mean_distance + std_distance]

        if len(valid_distances) == 0:
            return 0.1

        # Average valid distances gives ridge spacing
        avg_distance = np.mean(valid_distances)

        # Frequency is inverse of distance (with safety check)
        if avg_distance > 0:
            frequency = 1.0 / avg_distance
            # Constrain to reasonable range
            frequency = max(self.config['min_ridge_frequency'],
                            min(frequency, self.config['max_ridge_frequency']))
            return frequency
        else:
            return 0.1  # Fallback frequency

    def _enhance_ridges(self, image: np.ndarray,
                        orientation: np.ndarray,
                        frequency: np.ndarray) -> np.ndarray:
        """
        Enhance ridge structures using Gabor filter bank.

        Args:
            image: Normalized grayscale image
            orientation: Orientation field
            frequency: Frequency map

        Returns:
            Ridge-enhanced image
        """
        rows, cols = image.shape
        enhanced = np.zeros_like(image, dtype=np.float32)

        # Apply Gabor filters adaptively across the image
        kernel_size = self.config['gabor_kernel_size']
        half_kernel = kernel_size // 2

        # Process in a grid pattern (every 8th pixel for efficiency)
        step_size = 8

        for i in range(0, rows, step_size):
            for j in range(0, cols, step_size):
                # Get local orientation and frequency
                local_orientation = orientation[i, j]
                local_frequency = frequency[i, j]

                # Skip if frequency is invalid
                if (local_frequency < self.config['min_ridge_frequency'] or
                        local_frequency > self.config['max_ridge_frequency']):
                    continue

                # Find the closest precomputed Gabor filter orientation
                best_filter = None
                min_angle_diff = float('inf')

                for theta, kernel in self.gabor_filters:
                    angle_diff = min(abs(local_orientation - theta),
                                     abs(local_orientation - theta - np.pi),
                                     abs(local_orientation - theta + np.pi))

                    if angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                        best_filter = kernel

                if best_filter is None:
                    continue

                # Define region for filter application
                i_start = max(0, i - half_kernel)
                i_end = min(rows, i + half_kernel + 1)
                j_start = max(0, j - half_kernel)
                j_end = min(cols, j + half_kernel + 1)

                # Extract region from image
                region = image[i_start:i_end, j_start:j_end].astype(np.float32)

                # Extract corresponding part of kernel
                k_start_i = max(0, half_kernel - i)
                k_start_j = max(0, half_kernel - j)
                k_end_i = k_start_i + (i_end - i_start)
                k_end_j = k_start_j + (j_end - j_start)

                kernel_region = best_filter[k_start_i:k_end_i, k_start_j:k_end_j]

                # Apply filter to region
                filtered_region = cv2.filter2D(region, cv2.CV_32F, kernel_region)

                # Add to enhanced image
                enhanced[i_start:i_end, j_start:j_end] += filtered_region

        # Normalize enhanced image to 0-255 range
        enhanced = cv2.normalize(
            enhanced,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)

        return enhanced

    def _binarize_image(self, enhanced_image: np.ndarray) -> np.ndarray:
        """
        Convert enhanced grayscale image to binary (black/white).

        Args:
            enhanced_image: Ridge-enhanced grayscale image

        Returns:
            Binary image (255 for ridges, 0 for valleys)
        """
        # Use adaptive thresholding to handle varying illumination
        binary = cv2.adaptiveThreshold(
            enhanced_image,
            255,  # Maximum value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config['adaptive_block_size'],
            self.config['adaptive_c']
        )

        return binary

    def _clean_binary_image(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Clean binary image using morphological operations.

        Args:
            binary_image: Binary fingerprint image

        Returns:
            Cleaned binary image
        """
        kernel_size = self.config['morph_kernel_size']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Close operation: fills small holes in ridges
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Open operation: removes small spurs from ridges
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

    def _skeletonize_image(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Thin binary ridges to single-pixel width (skeletonization).

        Args:
            binary_image: Cleaned binary image

        Returns:
            Skeletonized image
        """
        # For educational purposes, implement a simple thinning algorithm
        # In production, you'd use cv2.ximgproc.thinning() or skimage.morphology.skeletonize

        skeleton = binary_image.copy()
        rows, cols = skeleton.shape

        # Zhang-Suen thinning algorithm (simplified version)
        changed = True
        while changed:
            changed = False

            # Pass 1: Mark pixels for deletion
            markers = np.zeros_like(skeleton, dtype=np.uint8)

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if skeleton[i, j] == 255:  # Ridge pixel
                        # Get 8-neighborhood (clockwise from top-left)
                        p2, p3, p4 = skeleton[i - 1, j], skeleton[i - 1, j + 1], skeleton[i, j + 1]
                        p5, p6, p7 = skeleton[i + 1, j + 1], skeleton[i + 1, j], skeleton[i + 1, j - 1]
                        p8, p9 = skeleton[i, j - 1], skeleton[i - 1, j - 1]

                        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                        neighbor_count = sum(1 for n in neighbors if n == 255)

                        # Conditions for deletion (preserves connectivity)
                        if 2 <= neighbor_count <= 6:
                            transitions = 0
                            for k in range(8):
                                if neighbors[k] == 255 and neighbors[(k + 1) % 8] == 0:
                                    transitions += 1

                            if transitions == 1:
                                markers[i, j] = 1

            # Delete marked pixels
            skeleton[markers == 1] = 0

            if np.any(markers):
                changed = True

        # Final cleanup: remove isolated pixels
        skeleton = self._remove_isolated_pixels(skeleton)

        return skeleton

    def _remove_isolated_pixels(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Remove isolated pixels from skeleton.

        Args:
            skeleton: Thinned ridge image

        Returns:
            Cleaned skeleton
        """
        cleaned = skeleton.copy()
        rows, cols = cleaned.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if cleaned[i, j] == 255:
                    # Count neighbors
                    neighbors = [
                        cleaned[i - 1, j - 1], cleaned[i - 1, j], cleaned[i - 1, j + 1],
                        cleaned[i, j - 1], cleaned[i, j + 1],
                        cleaned[i + 1, j - 1], cleaned[i + 1, j], cleaned[i + 1, j + 1]
                    ]

                    # Remove pixel if it has no neighbors (isolated)
                    if sum(neighbors) == 0:
                        cleaned[i, j] = 0

        return cleaned

    def _calculate_quality_score(self, enhanced: np.ndarray,
                                 skeleton: np.ndarray,
                                 orientation: np.ndarray,
                                 frequency: np.ndarray) -> float:
        """
        Calculate quality score for processed fingerprint.

        Args:
            enhanced: Enhanced grayscale image
            skeleton: Skeletonized ridges
            orientation: Orientation field
            frequency: Frequency map

        Returns:
            Quality score between 0.0 and 1.0
        """
        scores = []

        # 1. Contrast score (based on enhanced image)
        contrast_score = np.std(enhanced) / 128.0  # Normalize to ~0-1
        scores.append(min(contrast_score, 1.0))

        # 2. Ridge continuity score (based on skeleton)
        ridge_pixels = np.sum(skeleton == 255)
        total_pixels = skeleton.size
        ridge_density = ridge_pixels / total_pixels

        # Ideal ridge density is around 0.3-0.4
        if 0.2 <= ridge_density <= 0.5:
            continuity_score = 1.0 - abs(ridge_density - 0.35) / 0.15
        else:
            continuity_score = 0.2
        scores.append(continuity_score)

        # 3. Orientation coherence score
        orientation_variance = np.var(orientation)
        # Low variance indicates coherent orientation field
        coherence_score = 1.0 / (1.0 + orientation_variance)
        scores.append(coherence_score)

        # 4. Frequency consistency score
        valid_freq_mask = (frequency >= self.config['min_ridge_frequency']) & \
                          (frequency <= self.config['max_ridge_frequency'])
        valid_freq_ratio = np.sum(valid_freq_mask) / frequency.size
        scores.append(valid_freq_ratio)

        # Weighted average of all scores
        weights = [0.3, 0.3, 0.2, 0.2]  # Contrast and continuity are most important
        quality_score = np.average(scores, weights=weights)

        return float(quality_score)

    def _log_error(self, error_message: str, image_shape: Tuple[int, int]) -> None:
        """
        Log preprocessing errors.

        Args:
            error_message: Description of error
            image_shape: Shape of image that caused error
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'image_shape': image_shape,
            'config': self.config
        }

        error_file = 'preprocessing_errors.json'
        try:
            with open(error_file, 'a') as f:
                json.dump(error_entry, f)
                f.write('\n')
        except:
            pass  # Silently fail if logging fails

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get preprocessing pipeline statistics.

        Returns:
            Dictionary of processing statistics
        """
        stats = self.processing_stats.copy()

        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['min_processing_time'] = np.min(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0
            stats['min_processing_time'] = 0
            stats['max_processing_time'] = 0

        if stats['quality_scores']:
            stats['avg_quality_score'] = np.mean(stats['quality_scores'])
            stats['min_quality_score'] = np.min(stats['quality_scores'])
            stats['max_quality_score'] = np.max(stats['quality_scores'])
        else:
            stats['avg_quality_score'] = 0
            stats['min_quality_score'] = 0
            stats['max_quality_score'] = 0

        return stats

    def quick_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Simplified preprocessing for quick visualization.

        Args:
            image: Input fingerprint image

        Returns:
            Enhanced grayscale image (skips skeletonization)
        """
        denoised = self._reduce_noise(image)
        normalized = self._normalize_contrast(denoised)
        orientation = self._estimate_orientation(normalized)
        frequency = self._estimate_frequency(normalized, orientation)
        enhanced = self._enhance_ridges(normalized, orientation, frequency)

        return enhanced

# ============================================================================
# [CONTINUATION - Next class would be FeatureExtractor]
# ============================================================================