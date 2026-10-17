"""
Template storage: SQLite, encrypted at rest, with an audit log.

* Templates (and the optional enrolment image used for match visualisation)
  are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) before they touch
  the database. Without the key the database holds only person IDs and
  counts.
* The key comes from the AFIS_KEY environment variable, or from a key file
  next to the database (created on first run with owner-only permissions).
* Every enrolment, deletion and match decision is written to `audit_log`.
* All SQL uses parameters, never string formatting.
"""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken

from .template import FingerprintTemplate


class TemplateIntegrityError(Exception):
    """A stored template failed decryption or its hash check."""


def load_or_create_key(db_path: str) -> bytes:
    env = os.environ.get("AFIS_KEY")
    if env:
        return env.encode()
    if db_path == ":memory:":
        return Fernet.generate_key()
    key_path = Path(db_path).with_suffix(".key")
    if key_path.exists():
        return key_path.read_bytes().strip()
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    try:
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # Windows: permissions are handled by the user profile
    return key


class TemplateDatabase:
    def __init__(self, db_path: str = "afis_templates.db", key: Optional[bytes] = None):
        self.db_path = db_path
        self._fernet = Fernet(key or load_or_create_key(db_path))
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    # ------------------------------------------------------------------ schema
    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS templates (
                fingerprint_id  TEXT PRIMARY KEY,
                person_id       TEXT NOT NULL,
                finger_position INTEGER DEFAULT 0,
                pattern_type    TEXT,
                minutiae_count  INTEGER,
                quality         REAL,
                template_blob   BLOB NOT NULL,      -- encrypted JSON
                image_blob      BLOB,               -- encrypted PNG (optional)
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_person ON templates (person_id);
            CREATE TABLE IF NOT EXISTS audit_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT DEFAULT CURRENT_TIMESTAMP,
                action         TEXT NOT NULL,
                person_id      TEXT,
                fingerprint_id TEXT,
                details        TEXT
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------ audit
    def log(self, action: str, person_id: str = None, fingerprint_id: str = None, details: str = None) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (action, person_id, fingerprint_id, details) VALUES (?, ?, ?, ?)",
            (action, person_id, fingerprint_id, details),
        )
        self.conn.commit()

    def audit_entries(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT timestamp, action, person_id, fingerprint_id, details FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ crud
    def save_template(self, template: FingerprintTemplate, image_png: Optional[bytes] = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO templates
               (fingerprint_id, person_id, finger_position, pattern_type, minutiae_count, quality,
                template_blob, image_blob, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                template.fingerprint_id,
                template.person_id,
                template.finger_position,
                template.get_pattern_type(),
                template.get_minutiae_count(),
                template.global_features.get("quality"),
                self._fernet.encrypt(template.to_json().encode()),
                self._fernet.encrypt(image_png) if image_png else None,
                template.created_at,
            ),
        )
        self.conn.commit()
        self.log("ENROLL", template.person_id, template.fingerprint_id,
                 f"{template.get_minutiae_count()} minutiae, {template.get_pattern_type()}")

    def _decode(self, row) -> FingerprintTemplate:
        try:
            template = FingerprintTemplate.from_json(self._fernet.decrypt(row["template_blob"]).decode())
        except InvalidToken as e:
            raise TemplateIntegrityError(
                f"Cannot decrypt template {row['fingerprint_id']} (wrong key or tampered data)") from e
        if not template.is_intact():
            raise TemplateIntegrityError(f"Template {row['fingerprint_id']} failed its integrity check")
        return template

    def load_template(self, fingerprint_id: str) -> Optional[FingerprintTemplate]:
        row = self.conn.execute("SELECT * FROM templates WHERE fingerprint_id = ?", (fingerprint_id,)).fetchone()
        return self._decode(row) if row else None

    def load_templates_by_person(self, person_id: str) -> List[FingerprintTemplate]:
        rows = self.conn.execute("SELECT * FROM templates WHERE person_id = ?", (person_id,)).fetchall()
        return [self._decode(r) for r in rows]

    def load_all_templates(self) -> List[FingerprintTemplate]:
        return [self._decode(r) for r in self.conn.execute("SELECT * FROM templates").fetchall()]

    def load_image(self, fingerprint_id: str) -> Optional[bytes]:
        row = self.conn.execute("SELECT image_blob FROM templates WHERE fingerprint_id = ?",
                                (fingerprint_id,)).fetchone()
        if not row or row["image_blob"] is None:
            return None
        try:
            return self._fernet.decrypt(row["image_blob"])
        except InvalidToken:
            return None

    def delete_template(self, fingerprint_id: str) -> bool:
        row = self.conn.execute("SELECT person_id FROM templates WHERE fingerprint_id = ?",
                                (fingerprint_id,)).fetchone()
        cur = self.conn.execute("DELETE FROM templates WHERE fingerprint_id = ?", (fingerprint_id,))
        self.conn.commit()
        if cur.rowcount:
            self.log("DELETE", row["person_id"] if row else None, fingerprint_id)
        return cur.rowcount > 0

    def delete_person_templates(self, person_id: str) -> int:
        cur = self.conn.execute("DELETE FROM templates WHERE person_id = ?", (person_id,))
        self.conn.commit()
        self.log("DELETE_PERSON", person_id, None, f"{cur.rowcount} template(s)")
        return cur.rowcount

    def list_templates(self) -> List[Dict[str, Any]]:
        """Metadata only (no decryption) for tables in the GUI."""
        rows = self.conn.execute(
            """SELECT fingerprint_id, person_id, finger_position, pattern_type, minutiae_count, quality, created_at
               FROM templates ORDER BY person_id, created_at"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_statistics(self) -> Dict[str, Any]:
        c = self.conn
        total = c.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
        persons = c.execute("SELECT COUNT(DISTINCT person_id) FROM templates").fetchone()[0]
        avg = c.execute("SELECT AVG(minutiae_count) FROM templates").fetchone()[0] or 0
        dist = {r[0]: r[1] for r in c.execute("SELECT pattern_type, COUNT(*) FROM templates GROUP BY pattern_type")}
        return {"total_templates": total, "unique_persons": persons,
                "avg_minutiae_count": round(avg, 1), "pattern_distribution": dist}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
