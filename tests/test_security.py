"""Encryption, key management, integrity, audit chain and destructive actions."""
import json
import os
import sqlite3
import stat

import numpy as np
import pytest
from cryptography.fernet import Fernet

from afis.config import Settings, data_dir, default_db_path
from afis.database import (KeyMismatchError, MissingKeyError, TemplateDatabase, TemplateIntegrityError,
                           key_path_for)
from afis.features import Minutia
from afis.system import AFISSystem, EnrollmentError, validate_person_id
from afis.template import FingerprintTemplate

from conftest import demo


def _template(fid="alice-1", pid="alice", n=20, seed=4):
    rng = np.random.default_rng(seed)
    ms = [Minutia(int(rng.uniform(40, 300)), int(rng.uniform(40, 300)), rng.uniform(0, 2 * np.pi),
                  rng.choice(["ending", "bifurcation"])) for _ in range(n)]
    return FingerprintTemplate(fid, pid, ms, {"pattern_type": "loop", "quality": 0.8})


# ---------------------------------------------------------------- keys
def test_key_file_created_owner_only(tmp_path):
    db = str(tmp_path / "a.db")
    TemplateDatabase(db).close()
    kp = key_path_for(db)
    assert kp.exists() and Fernet(kp.read_bytes())          # a valid key
    if os.name == "posix":
        assert stat.S_IMODE(kp.stat().st_mode) == 0o600
        assert stat.S_IMODE(os.stat(db).st_mode) == 0o600


def test_wrong_key_rejected_at_open(tmp_path):
    db = str(tmp_path / "a.db")
    TemplateDatabase(db, Fernet.generate_key()).close()
    with pytest.raises(KeyMismatchError):
        TemplateDatabase(db, Fernet.generate_key())


def test_invalid_key_material_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("AFIS_KEY", "not-a-key")
    with pytest.raises(KeyMismatchError):
        TemplateDatabase(str(tmp_path / "a.db"))


def test_missing_key_file_is_not_silently_replaced(tmp_path):
    db = str(tmp_path / "a.db")
    d = TemplateDatabase(db)
    d.save_template(_template())
    d.close()
    key_path_for(db).unlink()
    with pytest.raises(MissingKeyError):
        TemplateDatabase(db)
    assert not key_path_for(db).exists()                    # no new key was written


def test_env_key_takes_precedence(tmp_path, monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv("AFIS_KEY", key.decode())
    db = str(tmp_path / "a.db")
    TemplateDatabase(db).close()
    assert not key_path_for(db).exists()
    TemplateDatabase(db, key).close()                       # same key works


def test_legacy_database_without_key_check_is_upgraded(tmp_path):
    key = Fernet.generate_key()
    db = str(tmp_path / "a.db")
    d = TemplateDatabase(db, key)
    d.save_template(_template())
    d.conn.execute("DELETE FROM meta WHERE name = 'key_check'")
    d.conn.commit()
    d.close()
    with pytest.raises(KeyMismatchError):                   # wrong key detected via a template
        TemplateDatabase(db, Fernet.generate_key())
    d = TemplateDatabase(db, key)
    assert d.conn.execute("SELECT 1 FROM meta WHERE name = 'key_check'").fetchone()
    d.close()


# ---------------------------------------------------------------- templates
def test_nothing_sensitive_in_plaintext(tmp_path):
    db = str(tmp_path / "a.db")
    afis = AFISSystem(db, key=Fernet.generate_key())
    t = afis.enroll(demo(1, 1), "alice")
    afis.close()
    raw = open(db, "rb").read()
    assert b'"minutiae"' not in raw and b"\x89PNG" not in raw
    first = t.minutiae[0]
    assert json.dumps(first.to_list()).encode() not in raw


def test_swapped_template_detected(tmp_path):
    d = TemplateDatabase(str(tmp_path / "a.db"), Fernet.generate_key())
    d.save_template(_template("alice-1", "alice", seed=1))
    d.save_template(_template("bob-1", "bob", seed=2))
    # An attacker with write access moves Bob's encrypted template into Alice's row.
    bob_blob = d.conn.execute("SELECT template_blob FROM templates WHERE person_id='bob'").fetchone()[0]
    d.conn.execute("UPDATE templates SET template_blob = ? WHERE person_id='alice'", (bob_blob,))
    with pytest.raises(TemplateIntegrityError, match="does not belong"):
        d.load_templates_by_person("alice")
    report = d.check_integrity()
    assert report["checked"] == 2 and [fid for fid, _ in report["bad"]] == ["alice-1"]


def test_relabelled_person_detected(tmp_path):
    d = TemplateDatabase(str(tmp_path / "a.db"), Fernet.generate_key())
    d.save_template(_template("alice-1", "alice"))
    d.conn.execute("UPDATE templates SET person_id = 'mallory'")
    with pytest.raises(TemplateIntegrityError):
        d.load_templates_by_person("mallory")


def test_template_hash_covers_identity():
    t = _template()
    data = t.to_dict()
    data["person_id"] = "mallory"
    assert not FingerprintTemplate.from_dict(data).is_intact()


def test_version2_templates_still_load():
    t = _template()
    data = t.to_dict()
    data["version"] = 2
    legacy = FingerprintTemplate.from_dict(data)
    legacy.template_hash = legacy.compute_hash()            # hash as AFIS 2.0 computed it (minutiae only)
    again = FingerprintTemplate.from_json(legacy.to_json())
    assert again.version == 2 and again.is_intact()


# ---------------------------------------------------------------- audit log
def test_audit_chain_detects_edits_deletions_and_truncation(tmp_path):
    d = TemplateDatabase(str(tmp_path / "a.db"), Fernet.generate_key())
    for k in range(4):
        d.log("VERIFY", f"p{k}", None, f"score={k / 10:.3f}")
    assert d.verify_audit() == {"ok": True, "checked": 4, "legacy": 0, "first_bad_id": None}

    d.conn.execute("UPDATE audit_log SET details = 'score=0.999' WHERE id = 2")
    res = d.verify_audit()
    assert not res["ok"] and res["first_bad_id"] == 2


def test_audit_chain_detects_deleted_middle_entry(tmp_path):
    d = TemplateDatabase(str(tmp_path / "a.db"), Fernet.generate_key())
    for k in range(4):
        d.log("VERIFY", f"p{k}")
    d.conn.execute("DELETE FROM audit_log WHERE id = 2")
    assert d.verify_audit()["first_bad_id"] == 3


def test_audit_chain_detects_truncation_and_insertion(tmp_path):
    d = TemplateDatabase(str(tmp_path / "a.db"), Fernet.generate_key())
    for k in range(3):
        d.log("VERIFY", f"p{k}")
    d.conn.execute("DELETE FROM audit_log WHERE id = 3")
    assert d.verify_audit().get("truncated")

    d2 = TemplateDatabase(str(tmp_path / "b.db"), Fernet.generate_key())
    d2.log("ENROLL", "a")
    d2.conn.execute("INSERT INTO audit_log (timestamp, action) VALUES ('2020-01-01', 'FORGED')")
    assert not d2.verify_audit()["ok"]


def test_legacy_audit_entries_are_tolerated(tmp_path):
    db = str(tmp_path / "old.db")
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                                action TEXT NOT NULL, person_id TEXT, fingerprint_id TEXT, details TEXT);
        INSERT INTO audit_log (action, person_id) VALUES ('ENROLL', 'old');
    """)
    con.commit()
    con.close()
    d = TemplateDatabase(db, Fernet.generate_key())
    d.log("VERIFY", "new")
    res = d.verify_audit()
    assert res["ok"] and res["legacy"] == 1 and res["checked"] == 1


def test_operations_are_audited(tmp_path):
    afis = AFISSystem(str(tmp_path / "a.db"), key=Fernet.generate_key())
    t = afis.enroll(demo(1, 1), "alice")
    afis.verify(demo(1, 2), "alice")
    afis.identify(demo(1, 3))
    afis.delete_template(t.fingerprint_id)
    actions = [e["action"] for e in reversed(afis.database.audit_entries())]
    assert actions == ["ENROLL", "VERIFY", "IDENTIFY", "DELETE"]
    assert afis.check_integrity()["audit"]["ok"]


# ---------------------------------------------------------------- destructive actions
def test_delete_all_requires_typed_confirmation(tmp_path):
    afis = AFISSystem(str(tmp_path / "a.db"), key=Fernet.generate_key())
    afis.enroll(demo(1, 1), "alice")
    for wrong in ("", "delete", "yes", None):
        with pytest.raises(ValueError):
            afis.delete_all(wrong)
    assert afis.get_statistics()["total_templates"] == 1
    assert afis.delete_all("DELETE") == 1
    assert afis.get_statistics()["total_templates"] == 0
    assert afis.database.audit_entries()[0]["action"] == "DELETE_ALL"


def test_deleted_data_is_overwritten_on_disk(tmp_path):
    db = str(tmp_path / "a.db")
    d = TemplateDatabase(db, Fernet.generate_key())
    d.save_template(_template(), b"x" * 5000)
    blob = d.conn.execute("SELECT template_blob FROM templates").fetchone()[0]
    d.delete_template("alice-1")
    d.close()
    assert bytes(blob[:40]) not in open(db, "rb").read()


# ---------------------------------------------------------------- input validation
@pytest.mark.parametrize("pid", ["alice", "Jane Doe", "j.doe@example.org", "O'Neil", "user_01", "Adébáyọ̀", "Ọlá-Olúwa", "张伟"])
def test_valid_person_ids(pid):
    import unicodedata
    assert validate_person_id(f"  {pid} ") == unicodedata.normalize("NFC", pid)


@pytest.mark.parametrize("pid", ["", "   ", "../etc/passwd", "a\nb", "x" * 65, "<script>", "-rf"])
def test_invalid_person_ids(pid):
    with pytest.raises(EnrollmentError):
        validate_person_id(pid)


def test_sql_injection_is_inert(tmp_path):
    afis = AFISSystem(str(tmp_path / "a.db"), key=Fernet.generate_key())
    afis.enroll(demo(1, 1), "alice")
    ok, _, details = afis.verify(demo(1, 2), "x' OR '1'='1")
    assert not ok and "error" in details
    assert afis.delete_person("x' OR '1'='1") == 0
    assert afis.get_statistics()["total_templates"] == 1


# ---------------------------------------------------------------- paths & settings
def test_data_dir_is_private(tmp_path):
    d = data_dir()
    assert d == tmp_path / "afis_home" and d.is_dir()
    if os.name == "posix":
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
    assert default_db_path() == str(d / "afis_templates.db")


def test_afis_db_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AFIS_DB", str(tmp_path / "custom" / "x.db"))
    assert default_db_path() == str(tmp_path / "custom" / "x.db")
    assert (tmp_path / "custom").is_dir()


def test_settings_roundtrip_and_sanitising(tmp_path):
    s = Settings(threshold=0.25, store_images=False, accepted_notice=True)
    s.save(tmp_path)
    assert Settings.load(tmp_path) == s
    (tmp_path / "settings.json").write_text('{"threshold": 99, "unknown": 1}')
    assert Settings.load(tmp_path).threshold == 0.95
    (tmp_path / "settings.json").write_text("{not json")
    assert Settings.load(tmp_path) == Settings()


def test_gitignore_excludes_secrets():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    for pattern in ("*.db", "*.key", "settings.json", "samples/*"):
        assert pattern in text


def test_legacy_afis_folder_is_reused_and_tightened(tmp_path, monkeypatch):
    monkeypatch.delenv("AFIS_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    legacy = tmp_path / ".afis"
    legacy.mkdir(mode=0o755)
    (legacy / "afis_templates.db").write_bytes(b"")
    assert data_dir() == legacy
    if os.name == "posix":
        assert stat.S_IMODE(legacy.stat().st_mode) == 0o700
