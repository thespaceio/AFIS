from preprocessing import PreprocessingPipeline
from acquisition import FingerprintAcquisition
from matching import FingerprintMatcher
from template import TemplateDatabase, FingerprintTemplate
from features import FeatureExtractor
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
# 8. AFIS SYSTEM – MAIN INTEGRATION CLASS
# ============================================================================

class AFISSystem:
    """
    Complete Automated Fingerprint Identification System.
    Integrates acquisition, preprocessing, feature extraction, template storage,
    and matching into a unified API.
    """

    def __init__(self, db_path: str = 'fingerprint_templates.db'):
        """
        Initialize the AFIS system with all components.

        Args:
            db_path: Path to SQLite database for template storage
        """
        self.acquisition = FingerprintAcquisition()
        self.preprocessor = PreprocessingPipeline()
        self.extractor = FeatureExtractor()
        self.database = TemplateDatabase(db_path)
        self.matcher = FingerprintMatcher()

        # Internal state
        self.last_processing_results = None  # For debugging/visualization

    def enroll(self, image_path: str, person_id: str,
               finger_position: int = 1) -> Optional[FingerprintTemplate]:
        """
        Enroll a fingerprint image into the system.

        Args:
            image_path: Path to fingerprint image
            person_id: Identifier of the person (e.g., 'user123')
            finger_position: Optional finger position code (e.g., 1=right thumb)

        Returns:
            FingerprintTemplate object if successful, None otherwise
        """
        try:
            print(f"📥 Enrolling fingerprint for {person_id}...")

            # 1. Acquire image
            raw_image = self.acquisition.load_fingerprint(image_path)

            # 2. Preprocess
            results = self.preprocessor.preprocess(raw_image)
            self.last_processing_results = results

            skeleton = results['skeleton']
            orientation = results['orientation']

            # 3. Extract features
            minutiae, global_features = self.extractor.extract_all(skeleton, orientation)

            if len(minutiae) < 5:
                print(f"⚠️  Warning: Only {len(minutiae)} minutiae found. Quality may be low.")
                if len(minutiae) < 3:
                    print("❌ Enrollment failed: insufficient minutiae.")
                    return None

            # 4. Create template
            fingerprint_id = f"{person_id}_{finger_position}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            template = FingerprintTemplate(
                fingerprint_id=fingerprint_id,
                person_id=person_id,
                minutiae=minutiae,
                global_features=global_features
            )

            # 5. Save to database
            success = self.database.save_template(template)
            if not success:
                print("❌ Failed to save template to database.")
                return None

            print(f"✅ Enrollment successful: {len(minutiae)} minutiae, pattern: {global_features['pattern_type']}")
            return template

        except Exception as e:
            print(f"❌ Enrollment error: {e}")
            return None

    def verify(self, query_image_path: str, claimed_id: str,
               threshold: float = None) -> Tuple[bool, float, Dict]:
        """
        1:1 verification: Does this fingerprint belong to claimed_id?

        Args:
            query_image_path: Path to fingerprint image to verify
            claimed_id: Person ID to verify against
            threshold: Match score threshold (if None, uses matcher's default)

        Returns:
            (is_match, score, details)
        """
        try:
            print(f"🔍 Verifying fingerprint for {claimed_id}...")

            # Create template from query image
            query_template = self._create_template_from_image(query_image_path, "temp_query")
            if query_template is None:
                return False, 0.0, {"error": "Failed to create template from query image"}

            # Retrieve reference templates for claimed person
            ref_templates = self.database.load_templates_by_person(claimed_id)
            if not ref_templates:
                print(f"⚠️  No templates found for {claimed_id}")
                return False, 0.0, {"error": "No reference templates found"}

            # Match against the best reference (usually the first one, but could average)
            best_score = 0.0
            best_details = None
            for ref in ref_templates:
                is_match, score, details = self.matcher.verify(
                    query_template, ref, threshold)
                if score > best_score:
                    best_score = score
                    best_details = details

            is_match = best_score >= (threshold or self.matcher.config['min_matches_ratio'])

            print(f"✅ Verification result: {'MATCH' if is_match else 'NO MATCH'} (score: {best_score:.3f})")
            return is_match, best_score, best_details

        except Exception as e:
            print(f"❌ Verification error: {e}")
            return False, 0.0, {"error": str(e)}

    def identify(self, query_image_path: str, top_k: int = 5,
                 threshold: float = None) -> List[Tuple[str, float, Dict]]:
        """
        1:N identification: Find the person(s) most likely to match the query.

        Args:
            query_image_path: Path to fingerprint image to identify
            top_k: Number of top candidates to return
            threshold: Minimum score to consider a match (if None, uses matcher's default)

        Returns:
            List of tuples (person_id, score, details) sorted by score descending.
        """
        try:
            print(f"🔎 Identifying fingerprint...")

            # Create template from query image
            query_template = self._create_template_from_image(query_image_path, "temp_identify")
            if query_template is None:
                return []

            # Load all reference templates
            all_templates = self.database.load_all_templates()
            if not all_templates:
                print("⚠️  Database is empty.")
                return []

            # Perform identification
            results = self.matcher.identify(query_template, all_templates,
                                            threshold=threshold, top_k=top_k)

            print(f"✅ Identification complete. Top candidate: {results[0][0]} (score: {results[0][1]:.3f})")
            return results

        except Exception as e:
            print(f"❌ Identification error: {e}")
            return []

    def _create_template_from_image(self, image_path: str,
                                    template_id: str) -> Optional[FingerprintTemplate]:
        """
        Helper: Create a template from an image without saving to database.
        Used for verification and identification queries.
        """
        try:
            raw_image = self.acquisition.load_fingerprint(image_path)
            results = self.preprocessor.preprocess(raw_image)
            self.last_processing_results = results

            skeleton = results['skeleton']
            orientation = results['orientation']

            minutiae, global_features = self.extractor.extract_all(skeleton, orientation)

            if len(minutiae) < 3:
                return None

            # Create a temporary template (not stored)
            template = FingerprintTemplate(
                fingerprint_id=template_id,
                person_id="query",
                minutiae=minutiae,
                global_features=global_features
            )
            return template

        except Exception as e:
            print(f"Error creating template from {image_path}: {e}")
            return None

    def delete_person(self, person_id: str) -> int:
        """Delete all templates belonging to a person."""
        return self.database.delete_person_templates(person_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        stats = self.database.get_statistics()
        stats['preprocessing_stats'] = self.preprocessor.get_statistics()
        stats['extractor_stats'] = self.extractor.get_statistics()
        stats['matcher_stats'] = self.matcher.get_statistics()
        return stats

    def visualize_last_processing(self):
        """
        Visualize the last processed image and its stages.
        Useful for debugging and demonstration.
        """
        if self.last_processing_results is None:
            print("No processing results to visualize.")
            return

        # Import matplotlib conditionally
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        results = self.last_processing_results
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        titles = ['Raw', 'Denoised', 'Normalized', 'Enhanced',
                  'Binary', 'Cleaned', 'Skeleton', 'Orientation']

        images = [
            results['raw'],
            results['denoised'],
            results['normalized'],
            results['enhanced'],
            results['binary'],
            results['cleaned'],
            results['skeleton'],
            results['orientation']
        ]

        for ax, img, title in zip(axes.flat, images, titles):
            if title == 'Orientation':
                ax.imshow(img, cmap='hsv')
            else:
                ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_minutiae(self, image_path: str = None, template: FingerprintTemplate = None):
        """
        Visualize minutiae on the fingerprint image.
        If image_path is provided, processes that image; otherwise uses the last processed results.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed.")
            return

        if image_path is not None:
            raw = self.acquisition.load_fingerprint(image_path)
            results = self.preprocessor.preprocess(raw)
            skeleton = results['skeleton']
            orientation = results['orientation']
            minutiae, _ = self.extractor.extract_all(skeleton, orientation)
        elif template is not None:
            # Need the original image for visualization. If we have template only, we cannot show original.
            print("Visualization requires the original image. Pass image_path.")
            return
        elif self.last_processing_results is not None:
            raw = self.last_processing_results['raw']
            minutiae = self.extractor.extract_minutiae(
                self.last_processing_results['skeleton'],
                self.last_processing_results['orientation']
            )
        else:
            print("No image or processing results available.")
            return

        # Convert to color for drawing
        img_color = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        for m in minutiae:
            color = (0, 255, 0) if m.minutiae_type == 'ending' else (255, 0, 0)
            cv2.circle(img_color, (m.x, m.y), 3, color, -1)
            # Draw orientation line
            end_x = int(m.x + 10 * np.cos(m.orientation))
            end_y = int(m.y + 10 * np.sin(m.orientation))
            cv2.line(img_color, (m.x, m.y), (end_x, end_y), color, 1)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title(f"Minutiae: {len(minutiae)} points (green=ending, blue=bifurcation)")
        plt.axis('off')
        plt.show()