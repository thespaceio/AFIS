"""AFISSystem: one object that runs the whole pipeline and talks to the database."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .acquisition import FingerprintAcquisition
from .database import TemplateDatabase
from .features import FeatureExtractor, Minutia
from .matching import FingerprintMatcher
from .preprocessing import PreprocessingPipeline, PreprocessResult
from .template import FingerprintTemplate

ImageSource = Union[str, bytes, np.ndarray]

FINGER_NAMES = {
    0: "Unknown", 1: "Right thumb", 2: "Right index", 3: "Right middle", 4: "Right ring", 5: "Right little",
    6: "Left thumb", 7: "Left index", 8: "Left middle", 9: "Left ring", 10: "Left little",
}


class EnrollmentError(ValueError):
    """The print isn't good enough to enrol."""


@dataclass
class ProcessedPrint:
    image: np.ndarray
    result: PreprocessResult
    minutiae: List[Minutia]
    global_features: Dict[str, Any]

    def as_template(self, person_id: str = "query", fingerprint_id: str = "query",
                    finger_position: int = 0) -> FingerprintTemplate:
        return FingerprintTemplate(fingerprint_id, person_id, self.minutiae, self.global_features, finger_position)


class AFISSystem:
    MIN_ENROL_MINUTIAE = 12
    MIN_ENROL_QUALITY = 0.35

    def __init__(self, db_path: str = "afis_templates.db", key: Optional[bytes] = None, store_images: bool = True):
        """
        Args:
            db_path: SQLite file (':memory:' for a throw-away database).
            key: Fernet key; defaults to AFIS_KEY or a key file next to the database.
            store_images: keep an encrypted copy of each enrolment image so the GUI
                can show matched minutiae side by side. Set False to store templates only.
        """
        self.acquisition = FingerprintAcquisition()
        self.preprocessor = PreprocessingPipeline()
        self.extractor = FeatureExtractor()
        self.matcher = FingerprintMatcher()
        self.database = TemplateDatabase(db_path, key)
        self.store_images = store_images
        self.last: Optional[ProcessedPrint] = None

    # ------------------------------------------------------------------ pipeline
    def process(self, source: ImageSource) -> ProcessedPrint:
        image = self.acquisition.load(source)
        result = self.preprocessor.preprocess(image)
        minutiae, global_features = self.extractor.extract_all(result)
        self.last = ProcessedPrint(image, result, minutiae, global_features)
        return self.last

    # ------------------------------------------------------------------ operations
    def enroll(self, source: ImageSource, person_id: str, finger_position: int = 0) -> FingerprintTemplate:
        person_id = person_id.strip()
        if not person_id:
            raise EnrollmentError("Person ID cannot be empty")
        p = self.process(source)
        if p.result.quality < self.MIN_ENROL_QUALITY:
            raise EnrollmentError(f"Image quality too low ({p.result.quality:.2f}). Try a clearer print.")
        if len(p.minutiae) < self.MIN_ENROL_MINUTIAE:
            raise EnrollmentError(f"Only {len(p.minutiae)} minutiae found (need {self.MIN_ENROL_MINUTIAE}). "
                                  "Try a larger or clearer print.")
        fid = f"{person_id}-{finger_position}-{uuid.uuid4().hex[:8]}"
        template = p.as_template(person_id, fid, finger_position)
        png = cv2.imencode(".png", p.image)[1].tobytes() if self.store_images else None
        self.database.save_template(template, png)
        return template

    def verify(self, source: ImageSource, claimed_id: str,
               threshold: Optional[float] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """1:1 - does this print belong to `claimed_id`? Returns (is_match, score, details)."""
        refs = self.database.load_templates_by_person(claimed_id)
        if not refs:
            return False, 0.0, {"error": f"No templates enrolled for '{claimed_id}'"}
        query = self.process(source).as_template()
        best_score, best_details = 0.0, {"matched_minutiae": 0, "pairs": []}
        for ref in refs:
            _, score, details = self.matcher.verify(query, ref, threshold)
            if score >= best_score:
                best_score, best_details = score, dict(details, fingerprint_id=ref.fingerprint_id)
        thr = self.matcher.config.threshold if threshold is None else threshold
        is_match = best_score >= thr
        self.database.log("VERIFY", claimed_id, best_details.get("fingerprint_id"),
                          f"{'MATCH' if is_match else 'NO MATCH'} score={best_score:.3f}")
        return is_match, best_score, best_details

    def identify(self, source: ImageSource, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """1:N - ranked (person_id, score, details); details['is_match'] marks scores above threshold."""
        refs = self.database.load_all_templates()
        if not refs:
            return []
        query = self.process(source).as_template()
        ranked = self.matcher.identify(query, refs, top_k)
        thr = self.matcher.config.threshold
        for _, score, details in ranked:
            details["is_match"] = score >= thr
        top = ranked[0]
        self.database.log("IDENTIFY", top[0] if top[2]["is_match"] else None, top[2].get("fingerprint_id"),
                          f"top score={top[1]:.3f}")
        return ranked

    def delete_person(self, person_id: str) -> int:
        return self.database.delete_person_templates(person_id)

    def reference_image(self, fingerprint_id: str) -> Optional[np.ndarray]:
        png = self.database.load_image(fingerprint_id)
        return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE) if png else None

    def get_statistics(self) -> Dict[str, Any]:
        stats = self.database.get_statistics()
        stats["preprocessing_stats"] = self.preprocessor.get_statistics()
        stats["extractor_stats"] = self.extractor.get_statistics()
        stats["matcher_stats"] = self.matcher.get_statistics()
        stats["threshold"] = self.matcher.config.threshold
        return stats

    def close(self) -> None:
        self.database.close()
