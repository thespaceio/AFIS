"""
Template storage: SQLite, encrypted at rest, with a tamper-evident audit log.

* Templates (and the optional enrolment image used for match visualisation)
  are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) before they touch
  the database. Without the key the database holds only person IDs, finger
  numbers and counts.
* The key comes from the AFIS_KEY environment variable, or from a key file
  next to the database, created on first run with owner-only permissions.
  A known value encrypted with the key is stored in the database, so a
  wrong or missing key is reported at once instead of failing later.
* Each decrypted template must carry a valid SHA-256 hash that covers its
  person ID, template ID and minutiae, and must match the row it was read
  from, so templates can't be swapped between people.
* Every enrolment, deletion and match decision is written to `audit_log`.
  Each entry carries an HMAC of its contents and of the previous entry, so
  editing, deleting or reordering entries breaks the chain, and the
  encrypted chain head stored in `meta` reveals entries cut from the end
  (`verify_audit`). Someone holding the key can still rewrite history, so
  for strong accountability export the log to separate storage.
* Deleted rows are overwritten on disk (`PRAGMA secure_delete`).
* All SQL uses parameters, never string formatting.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken

from .template import FingerprintTemplate

SCHEMA_VERSION = 3
_KEY_CHECK = b"AFIS key check v1"


class TemplateIntegrityError(Exception):
    """A stored template failed decryption or its integrity checks."""


class AFISKeyError(Exception):
    """Base class for encryption-key problems."""


class KeyMismatchError(AFISKeyError):
    """The key can't decrypt this database."""


class MissingKeyError(AFISKeyError):
    """The database holds encrypted data but its key file is gone."""


def key_path_for(db_path: str) -> Path:
    return Path(db_path).with_suffix(".key")


def _write_private(path: Path, data: bytes) -> None:
    """Create `path` with owner-only permissions; fail if it already exists."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def load_or_create_key(db_path: str, allow_create: bool = True) -> bytes:
    env = os.environ.get("AFIS_KEY")
    if env:
        return env.strip().encode()
    if db_path == ":memory:":
        return Fernet.generate_key()
    key_path = key_path_for(db_path)
    if key_path.exists():
        return key_path.read_bytes().strip()
    if not allow_create:
        raise MissingKeyError(
            f"The database {db_path} contains encrypted data but its key file {key_path} is missing. "
            "Restore the key file (or set AFIS_KEY). Without it the stored templates can't be read.")
    key = Fernet.generate_key()
    _write_private(key_path, key)
    return key


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TemplateDatabase:
    def __init__(self, db_path: str = "afis_templates.db", key: Optional[bytes] = None):
        self.db_path = db_path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA secure_delete = ON")
            self._create_schema()
            has_data = self._has_encrypted_data()
            if key is None:
                key = load_or_create_key(db_path, allow_create=not has_data)
            try:
                self._fernet = Fernet(key)
            except (ValueError, TypeError) as e:
                raise KeyMismatchError("The encryption key is not a valid Fernet key "
                                       "(check AFIS_KEY or the .key file)") from e
            self._check_key()
            self._audit_key = hmac.new(key if isinstance(key, bytes) else key.encode(),
                                       b"afis-audit-chain", hashlib.sha256).digest()
        except Exception:
            self.conn.close()
            raise
        if db_path != ":memory:" and os.name == "posix":
            try:
                os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass

    # ------------------------------------------------------------------ schema
    def _create_schema(self) -> None:
        with self._lock:
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
                CREATE TABLE IF NOT EXISTS meta (
                    name  TEXT PRIMARY KEY,
                    value BLOB
                );
                """
            )
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(audit_log)")}
            if "entry_hash" not in cols:      # upgrade an AFIS 2.0 database
                self.conn.execute("ALTER TABLE audit_log ADD COLUMN entry_hash TEXT")
            self.conn.execute("INSERT OR REPLACE INTO meta (name, value) VALUES ('schema_version', ?)",
                              (str(SCHEMA_VERSION),))
            self.conn.commit()

    def _has_encrypted_data(self) -> bool:
        with self._lock:
            if self.conn.execute("SELECT 1 FROM meta WHERE name = 'key_check'").fetchone():
                return True
            return self.conn.execute("SELECT 1 FROM templates LIMIT 1").fetchone() is not None

    def _check_key(self) -> None:
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE name = 'key_check'").fetchone()
            if row is not None:
                try:
                    ok = self._fernet.decrypt(row["value"]) == _KEY_CHECK
                except InvalidToken:
                    ok = False
                if not ok:
                    raise KeyMismatchError(f"The encryption key does not match the database {self.db_path}.")
                return
            # Older database without a key check: test the key on one template.
            sample = self.conn.execute("SELECT template_blob FROM templates LIMIT 1").fetchone()
            if sample is not None:
                try:
                    self._fernet.decrypt(sample["template_blob"])
                except InvalidToken as e:
                    raise KeyMismatchError(
                        f"The encryption key does not match the database {self.db_path}.") from e
            self.conn.execute("INSERT INTO meta (name, value) VALUES ('key_check', ?)",
                              (self._fernet.encrypt(_KEY_CHECK),))
            self.conn.commit()

    # ------------------------------------------------------------------ audit
    def _entry_hash(self, prev: str, timestamp: str, action: str, person_id, fingerprint_id, details) -> str:
        payload = json.dumps([prev, timestamp, action, person_id, fingerprint_id, details], separators=(",", ":"))
        return hmac.new(self._audit_key, payload.encode(), hashlib.sha256).hexdigest()

    def log(self, action: str, person_id: str = None, fingerprint_id: str = None, details: str = None,
            commit: bool = True) -> None:
        with self._lock:
            prev = self.conn.execute(
                "SELECT entry_hash FROM audit_log WHERE entry_hash IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
            ts = _now()
            h = self._entry_hash(prev[0] if prev else "", ts, action, person_id, fingerprint_id, details)
            self.conn.execute(
                "INSERT INTO audit_log (timestamp, action, person_id, fingerprint_id, details, entry_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, action, person_id, fingerprint_id, details, h),
            )
            head = json.dumps({"hash": h, "count": self._chain_length()}).encode()
            self.conn.execute("INSERT OR REPLACE INTO meta (name, value) VALUES ('audit_head', ?)",
                              (self._fernet.encrypt(head),))
            if commit:
                self.conn.commit()

    def _chain_length(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM audit_log WHERE entry_hash IS NOT NULL").fetchone()[0]

    def audit_entries(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, timestamp, action, person_id, fingerprint_id, details FROM audit_log "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def verify_audit(self) -> Dict[str, Any]:
        """
        Re-compute the audit chain.

        Returns {"ok", "checked", "legacy", "first_bad_id"}. `legacy` counts
        entries written by AFIS 2.0, before the chain existed.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, timestamp, action, person_id, fingerprint_id, details, entry_hash "
                "FROM audit_log ORDER BY id").fetchall()
            head_row = self.conn.execute("SELECT value FROM meta WHERE name = 'audit_head'").fetchone()
        prev, checked, legacy = "", 0, 0
        for r in rows:
            if r["entry_hash"] is None:
                if checked:          # an unhashed entry after the chain started = inserted later
                    return {"ok": False, "checked": checked, "legacy": legacy, "first_bad_id": r["id"]}
                legacy += 1
                continue
            expected = self._entry_hash(prev, r["timestamp"], r["action"], r["person_id"],
                                        r["fingerprint_id"], r["details"])
            if not hmac.compare_digest(expected, r["entry_hash"]):
                return {"ok": False, "checked": checked, "legacy": legacy, "first_bad_id": r["id"]}
            prev = r["entry_hash"]
            checked += 1
        if head_row is not None or checked:
            try:
                head = json.loads(self._fernet.decrypt(head_row["value"])) if head_row else {}
            except (InvalidToken, ValueError):
                head = {}
            if head.get("hash") != (prev or None) or head.get("count") != checked:
                # Entries were removed from the end (or the head was tampered with).
                return {"ok": False, "checked": checked, "legacy": legacy, "first_bad_id": None,
                        "truncated": True}
        return {"ok": True, "checked": checked, "legacy": legacy, "first_bad_id": None}

    # ------------------------------------------------------------------ crud
    def save_template(self, template: FingerprintTemplate, image_png: Optional[bytes] = None) -> None:
        with self._lock:
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
            self.log("ENROLL", template.person_id, template.fingerprint_id,
                     f"{template.get_minutiae_count()} minutiae, {template.get_pattern_type()}", commit=False)
            self.conn.commit()

    def _decode(self, row) -> FingerprintTemplate:
        fid = row["fingerprint_id"]
        try:
            data = json.loads(self._fernet.decrypt(row["template_blob"]).decode())
            if not data.get("template_hash"):
                raise TemplateIntegrityError(f"Template {fid} has no integrity hash")
            template = FingerprintTemplate.from_dict(data)
        except InvalidToken as e:
            raise TemplateIntegrityError(f"Cannot decrypt template {fid} (wrong key or tampered data)") from e
        except (ValueError, KeyError, TypeError) as e:
            raise TemplateIntegrityError(f"Template {fid} is malformed: {e}") from e
        if not template.is_intact():
            raise TemplateIntegrityError(f"Template {fid} failed its integrity check")
        if template.fingerprint_id != fid or template.person_id != row["person_id"]:
            raise TemplateIntegrityError(f"Template {fid} does not belong to this record (moved or swapped)")
        return template

    def _query(self, sql: str, params: Tuple = ()) -> list:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def load_template(self, fingerprint_id: str) -> Optional[FingerprintTemplate]:
        rows = self._query("SELECT * FROM templates WHERE fingerprint_id = ?", (fingerprint_id,))
        return self._decode(rows[0]) if rows else None

    def load_templates_by_person(self, person_id: str) -> List[FingerprintTemplate]:
        return [self._decode(r) for r in self._query("SELECT * FROM templates WHERE person_id = ?", (person_id,))]

    def load_all_templates(self) -> List[FingerprintTemplate]:
        return [self._decode(r) for r in self._query("SELECT * FROM templates")]

    def check_integrity(self) -> Dict[str, Any]:
        """Decrypt and verify every template. Returns {"checked", "bad": [(fingerprint_id, reason)]}."""
        bad, n = [], 0
        for r in self._query("SELECT * FROM templates"):
            n += 1
            try:
                self._decode(r)
            except TemplateIntegrityError as e:
                bad.append((r["fingerprint_id"], str(e)))
        return {"checked": n, "bad": bad}

    def load_image(self, fingerprint_id: str) -> Optional[bytes]:
        rows = self._query("SELECT image_blob FROM templates WHERE fingerprint_id = ?", (fingerprint_id,))
        if not rows or rows[0]["image_blob"] is None:
            return None
        try:
            return self._fernet.decrypt(rows[0]["image_blob"])
        except InvalidToken:
            return None

    def delete_template(self, fingerprint_id: str) -> bool:
        with self._lock:
            row = self.conn.execute("SELECT person_id FROM templates WHERE fingerprint_id = ?",
                                    (fingerprint_id,)).fetchone()
            cur = self.conn.execute("DELETE FROM templates WHERE fingerprint_id = ?", (fingerprint_id,))
            if cur.rowcount:
                self.log("DELETE", row["person_id"] if row else None, fingerprint_id, commit=False)
            self.conn.commit()
            return cur.rowcount > 0

    def delete_person_templates(self, person_id: str) -> int:
        with self._lock:
            cur = self.conn.execute("DELETE FROM templates WHERE person_id = ?", (person_id,))
            if cur.rowcount:
                self.log("DELETE_PERSON", person_id, None, f"{cur.rowcount} template(s)", commit=False)
            self.conn.commit()
            return cur.rowcount

    def delete_all_templates(self) -> int:
        """Remove every template and image (the audit log is kept and records the action)."""
        with self._lock:
            cur = self.conn.execute("DELETE FROM templates")
            n = cur.rowcount
            self.log("DELETE_ALL", None, None, f"{n} template(s)", commit=False)
            self.conn.commit()
            self.conn.execute("VACUUM")
            return n

    def list_templates(self) -> List[Dict[str, Any]]:
        """Metadata only (no decryption) for tables in the GUI."""
        rows = self._query(
            """SELECT fingerprint_id, person_id, finger_position, pattern_type, minutiae_count, quality,
                      created_at, image_blob IS NOT NULL AS has_image
               FROM templates ORDER BY person_id, created_at""")
        return [dict(r) for r in rows]

    def persons(self) -> List[str]:
        return [r[0] for r in self._query("SELECT DISTINCT person_id FROM templates ORDER BY person_id")]

    def get_statistics(self) -> Dict[str, Any]:
        total = self._query("SELECT COUNT(*) FROM templates")[0][0]
        persons = self._query("SELECT COUNT(DISTINCT person_id) FROM templates")[0][0]
        avg = self._query("SELECT AVG(minutiae_count) FROM templates")[0][0] or 0
        dist = {r[0]: r[1] for r in self._query("SELECT pattern_type, COUNT(*) FROM templates GROUP BY pattern_type")}
        return {"total_templates": total, "unique_persons": persons,
                "avg_minutiae_count": round(avg, 1), "pattern_distribution": dist}

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
