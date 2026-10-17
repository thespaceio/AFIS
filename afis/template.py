"""
Fingerprint templates.

A template holds only what matching needs: the minutiae and a few global
features. It is stored as JSON (never pickle, which can execute code when a
tampered database is loaded).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .features import Minutia

TEMPLATE_VERSION = 3


@dataclass
class FingerprintTemplate:
    fingerprint_id: str
    person_id: str
    minutiae: List[Minutia]
    global_features: Dict[str, Any] = field(default_factory=dict)
    finger_position: int = 0
    created_at: Optional[str] = None
    template_hash: Optional[str] = None
    version: int = TEMPLATE_VERSION

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.template_hash is None:
            self.template_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """
        SHA-256 over the identity and the minutiae.

        Binding the person and template IDs into the hash means a template
        can't be moved to another person's record without detection.
        Version-2 templates (AFIS 2.0) hashed the minutiae only.
        """
        minutiae = [m.to_list() for m in self.minutiae]
        if self.version < 3:
            payload = minutiae
        else:
            payload = {"fingerprint_id": self.fingerprint_id, "person_id": self.person_id,
                       "finger_position": self.finger_position, "minutiae": minutiae}
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def is_intact(self) -> bool:
        return self.template_hash == self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fingerprint_id": self.fingerprint_id,
            "person_id": self.person_id,
            "finger_position": self.finger_position,
            "minutiae": [m.to_list() for m in self.minutiae],
            "global_features": self.global_features,
            "created_at": self.created_at,
            "template_hash": self.template_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FingerprintTemplate":
        return cls(
            fingerprint_id=data["fingerprint_id"],
            person_id=data["person_id"],
            minutiae=[Minutia.from_list(m) for m in data["minutiae"]],
            global_features=data.get("global_features", {}),
            finger_position=int(data.get("finger_position", 0)),
            created_at=data.get("created_at"),
            template_hash=data.get("template_hash"),
            version=int(data.get("version", 2)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "FingerprintTemplate":
        return cls.from_dict(json.loads(text))

    def get_minutiae_count(self) -> int:
        return len(self.minutiae)

    def get_pattern_type(self) -> str:
        return self.global_features.get("pattern_type", "unknown")
