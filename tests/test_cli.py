"""Command-line interface: commands, exit codes and confirmations."""
import pytest

from afis import __version__
from afis.cli import main

from conftest import demo


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cli.db")


def run(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out + out.err


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and __version__ in capsys.readouterr().out


def test_full_cli_flow(db, capsys):
    code, out = run(capsys, "--db", db, "enroll", "alice", demo(1, 1), "--finger", "2")
    assert code == 0 and "Enrolled alice" in out
    run(capsys, "--db", db, "enroll", "bob", demo(2, 1))

    code, out = run(capsys, "--db", db, "verify", "alice", demo(1, 2))
    assert code == 0 and "MATCH" in out
    code, out = run(capsys, "--db", db, "verify", "bob", demo(1, 2))
    assert code == 1 and "NO MATCH" in out
    code, out = run(capsys, "--db", db, "verify", "nobody", demo(1, 2))
    assert code == 2 and "No templates" in out

    code, out = run(capsys, "--db", db, "identify", demo(2, 3))
    assert code == 0 and "Best candidate: bob" in out
    code, out = run(capsys, "--db", db, "identify", demo(3, 2))      # not enrolled
    assert code == 1

    code, out = run(capsys, "--db", db, "list")
    assert code == 0 and "alice" in out and "bob" in out
    code, out = run(capsys, "--db", db, "stats")
    assert "templates        : 2" in out
    code, out = run(capsys, "--db", db, "check")
    assert code == 0 and "intact" in out
    code, out = run(capsys, "--db", db, "audit")
    assert "ENROLL" in out and "VERIFY" in out


def test_analyse(capsys):
    code, out = run(capsys, "analyse", demo(2, 1))
    assert code == 0 and "minutiae" in out and "whorl" in out


def test_errors_have_exit_code_2(db, capsys, tmp_path):
    code, out = run(capsys, "--db", db, "enroll", "alice", str(tmp_path / "missing.png"))
    assert code == 2 and "not found" in out.lower()
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image")
    code, out = run(capsys, "--db", db, "enroll", "alice", str(bad))
    assert code == 2 and "could not read" in out.lower()
    code, out = run(capsys, "--db", db, "enroll", "../evil", demo(1, 1))
    assert code == 2
    code, out = run(capsys, "--db", db, "identify", demo(1, 1))
    assert code == 2 and "empty" in out


def test_delete_asks_for_confirmation(db, capsys, monkeypatch):
    run(capsys, "--db", db, "enroll", "alice", demo(1, 1))
    monkeypatch.setattr("builtins.input", lambda _: "n")
    code, out = run(capsys, "--db", db, "delete", "alice")
    assert code == 1 and "Cancelled" in out
    monkeypatch.setattr("builtins.input", lambda _: "y")
    code, out = run(capsys, "--db", db, "delete", "alice")
    assert code == 0 and "Deleted 1" in out


def test_delete_all_needs_typed_delete(db, capsys, monkeypatch):
    run(capsys, "--db", db, "enroll", "alice", demo(1, 1))
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    code, _ = run(capsys, "--db", db, "delete-all")
    assert code == 1
    monkeypatch.setattr("builtins.input", lambda _: "DELETE")
    code, out = run(capsys, "--db", db, "delete-all")
    assert code == 0 and "Deleted 1" in out
    code, out = run(capsys, "--db", db, "delete-all", "--yes")
    assert code == 0 and "Deleted 0" in out


def test_wrong_key_is_a_clean_error(db, capsys, monkeypatch):
    from cryptography.fernet import Fernet
    run(capsys, "--db", db, "enroll", "alice", demo(1, 1))
    monkeypatch.setenv("AFIS_KEY", Fernet.generate_key().decode())
    code, out = run(capsys, "--db", db, "stats")
    assert code == 2 and "does not match" in out


def test_demo_and_default_location(capsys, tmp_path):
    code, out = run(capsys, "demo")
    assert code == 0 and "demo-06" in out
    assert (tmp_path / "afis_home" / "afis_templates.db").exists()
    code, out = run(capsys, "identify", demo(5, 4))
    assert code == 0 and "demo-05" in out
    code, out = run(capsys, "info")
    assert str(tmp_path / "afis_home") in out


def test_threshold_override_and_settings(capsys, tmp_path):
    from afis.config import Settings
    run(capsys, "demo")
    Settings(threshold=0.9).save(tmp_path / "afis_home")
    code, _ = run(capsys, "verify", "demo-01", demo(1, 2))
    assert code == 1                                   # 0.9 is too strict for this pair
    code, _ = run(capsys, "--threshold", "0.18", "verify", "demo-01", demo(1, 2))
    assert code == 0


def test_audit_export(db, capsys, tmp_path):
    run(capsys, "--db", db, "enroll", "alice", demo(1, 1))
    out_csv = tmp_path / "audit.csv"
    code, _ = run(capsys, "--db", db, "audit", "--export", str(out_csv))
    lines = out_csv.read_text().splitlines()
    assert code == 0 and lines[0].startswith("id,timestamp,action") and "ENROLL" in lines[1]
