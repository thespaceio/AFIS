#!/usr/bin/env python3
"""
AFIS Core Module - Contains all core classes for fingerprint recognition
Author: AFIS Developer
Date: 2026
"""

# ============================================================================
# IMPORT MODULES
# ============================================================================

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

warnings.filterwarnings('ignore')


# ============================================================================
# 1. FINGERPRINT ACQUISITION
# ============================================================================

class FingerprintAcquisition:
    """
    Handles fingerprint image loading, validation, and preprocessing setup.
    This is the first stage of the AFIS pipeline - converting physical/digital
    fingerprint images into standardized, validated formats for processing.
    """

    def __init__(self, standard_size: Tuple[int, int] = (500, 500)):
        """
        Initialize the acquisition module.

        Args:
            standard_size: Target dimensions for all fingerprint images (height, width)
        """
        self.standard_size = standard_size
        self.supported_formats = {'.tif', '.bmp', '.png', '.jpg', '.jpeg'}
        self.quality_thresholds = {
            'min_contrast': 20,  # Minimum standard deviation of pixel intensities
            'min_dimension': 200,  # Minimum image dimension in pixels
            'max_dimension': 1000  # Maximum image dimension (for resizing)
        }

        # Acquisition statistics
        self.stats = {
            'images_loaded': 0,
            'images_rejected': 0,
            'rejection_reasons': defaultdict(int)
        }

    def load_fingerprint(self, image_path: str) -> np.ndarray:
        """
        Load and validate a fingerprint image from file.

        Args:
            image_path: Path to fingerprint image file

        Returns:
            Grayscale image as numpy array (standardized size)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image fails validation checks
            IOError: If image cannot be read/decoded
        """
        # Check file existence and format
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}. Supported: {self.supported_formats}")

        # Load image in grayscale
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"Failed to decode image: {image_path}")
        except Exception as e:
            raise IOError(f"Error reading image {image_path}: {str(e)}")

        # Validate image quality
        self._validate_image(image, image_path)

        # Standardize size if needed
        if image.shape != self.standard_size:
            image = self._resize_image(image)

        # Update statistics
        self.stats['images_loaded'] += 1

        # Log successful acquisition
        self._log_acquisition(image_path, image.shape, True)

        return image

    def _validate_image(self, image: np.ndarray, image_path: str) -> None:
        """
        Perform comprehensive validation on fingerprint image.

        Args:
            image: Grayscale image array
            image_path: Source path for logging

        Raises:
            ValueError: If image fails any validation check
        """
        # Check image dimensions
        height, width = image.shape

        if height < self.quality_thresholds['min_dimension']:
            self._log_rejection('small_height', image_path, height)
            raise ValueError(f"Image too small: height={height} < {self.quality_thresholds['min_dimension']}")

        if width < self.quality_thresholds['min_dimension']:
            self._log_rejection('small_width', image_path, width)
            raise ValueError(f"Image too small: width={width} < {self.quality_thresholds['min_dimension']}")

        # Check intensity range
        if np.min(image) < 0 or np.max(image) > 255:
            self._log_rejection('invalid_intensity', image_path, (np.min(image), np.max(image)))
            raise ValueError(f"Invalid intensity range: [{np.min(image)}, {np.max(image)}]. Expected [0, 255]")

        # Check contrast (standard deviation of pixel intensities)
        contrast = np.std(image)
        if contrast < self.quality_thresholds['min_contrast']:
            self._log_rejection('low_contrast', image_path, contrast)
            raise ValueError(f"Insufficient contrast: {contrast:.1f} < {self.quality_thresholds['min_contrast']}")

        # Check for completely blank/empty images
        if np.all(image == image[0, 0]):
            self._log_rejection('uniform_image', image_path, None)
            raise ValueError("Image is completely uniform (no fingerprint pattern)")

        # Check for reasonable fingerprint area (not too much background)
        unique_values = np.unique(image)
        if len(unique_values) < 10:  # Too few distinct intensities
            self._log_rejection('low_variety', image_path, len(unique_values))
            raise ValueError(f"Image lacks intensity variety: only {len(unique_values)} distinct values")

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to standard dimensions while preserving aspect ratio when possible.

        Args:
            image: Original image array

        Returns:
            Resized image array
        """
        height, width = image.shape

        # If image is too large, first resize to reasonable dimensions
        if height > self.quality_thresholds['max_dimension'] or width > self.quality_thresholds['max_dimension']:
            scale = self.quality_thresholds['max_dimension'] / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Resize to standard size
        return cv2.resize(image, self.standard_size[::-1], interpolation=cv2.INTER_CUBIC)

    def _log_acquisition(self, image_path: str, image_shape: Tuple[int, int], success: bool) -> None:
        """
        Log acquisition attempt for audit trail.

        Args:
            image_path: Path to image file
            image_shape: Dimensions of loaded image
            success: Whether acquisition was successful
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'image_shape': image_shape,
            'success': success,
            'standard_size': self.standard_size
        }

        # Append to log file
        log_file = 'acquisition_log.json'
        try:
            with open(log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        except:
            pass  # Silently fail if logging fails

    def _log_rejection(self, reason: str, image_path: str, details: Any) -> None:
        """
        Log image rejection with reason.

        Args:
            reason: Category of rejection
            image_path: Path to rejected image
            details: Additional information about rejection
        """
        self.stats['images_rejected'] += 1
        self.stats['rejection_reasons'][reason] += 1

        rejection_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'reason': reason,
            'details': str(details)
        }

        # Append to rejection log
        rejection_file = 'rejection_log.json'
        try:
            with open(rejection_file, 'a') as f:
                json.dump(rejection_entry, f)
                f.write('\n')
        except:
            pass

    def load_multiple_fingerprints(self, directory_path: str) -> Dict[str, np.ndarray]:
        """
        Load all fingerprint images from a directory.

        Args:
            directory_path: Path to directory containing fingerprint images

        Returns:
            Dictionary mapping filenames to loaded images
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")

        images = {}

        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)

            # Skip directories and non-image files
            if not os.path.isfile(filepath):
                continue

            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in self.supported_formats:
                continue

            try:
                image = self.load_fingerprint(filepath)
                images[filename] = image
                print(f" Loaded: {filename} ({image.shape})")
            except Exception as e:
                print(f" Skipped {filename}: {str(e)}")

        print(f"\nLoaded {len(images)} images from {directory_path}")
        print(f"Statistics: {self.stats['images_loaded']} loaded, "
              f"{self.stats['images_rejected']} rejected")

        return images

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get acquisition statistics.

        Returns:
            Dictionary of acquisition statistics
        """
        stats = self.stats.copy()
        stats['total_attempted'] = stats['images_loaded'] + stats['images_rejected']

        if stats['total_attempted'] > 0:
            stats['success_rate'] = stats['images_loaded'] / stats['total_attempted']
        else:
            stats['success_rate'] = 0.0

        return stats

    def create_test_image(self, pattern_type: str = 'loop') -> np.ndarray:
        """
        Create a synthetic fingerprint image for testing.

        Args:
            pattern_type: Type of fingerprint pattern ('loop', 'whorl', 'arch')

        Returns:
            Synthetic fingerprint image
        """
        height, width = self.standard_size

        # Base image with gradient
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2

        # Create different patterns
        if pattern_type == 'loop':
            # Left loop pattern
            pattern = np.sin(x * 0.03) * np.cos(y * 0.02) * 127 + 128
        elif pattern_type == 'whorl':
            # Whorl/spiral pattern
            angle = np.arctan2(y - center_y, x - center_x)
            radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            pattern = np.sin(angle * 2 + radius * 0.1) * 127 + 128
        elif pattern_type == 'arch':
            # Arch pattern
            pattern = np.sin(x * 0.02) * 127 + 128
        else:
            # Default: simple ridges
            pattern = np.sin(x * 0.05) * 127 + 128

        # Add some noise
        noise = np.random.normal(0, 10, (height, width))
        pattern = np.clip(pattern + noise, 0, 255).astype(np.uint8)

        # Add some ridge-like texture
        for i in range(0, height, 15):
            pattern[i:i + 3, :] = pattern[i:i + 3, :] * 0.7  # Darker ridges

        return pattern

    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save an image to file.

        Args:
            image: Image array to save
            output_path: Destination path
        """
        cv2.imwrite(output_path, image)
        print(f"Image saved to: {output_path}")

# ============================================================================
# [CONTINUATION - Next class would be PreprocessingPipeline]
# ============================================================================
