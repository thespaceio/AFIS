"""AFIS - Automated Fingerprint Identification System (educational / portfolio project)."""

from .features import FeatureExtractor, Minutia
from .matching import FingerprintMatcher, MatcherConfig
from .preprocessing import PreprocessingPipeline, PreprocessResult
from .system import AFISSystem, EnrollmentError
from .template import FingerprintTemplate

__version__ = "2.1.0"

__all__ = [
    "AFISSystem", "EnrollmentError", "FeatureExtractor", "FingerprintMatcher", "FingerprintTemplate",
    "MatcherConfig", "Minutia", "PreprocessResult", "PreprocessingPipeline", "__version__",
]
