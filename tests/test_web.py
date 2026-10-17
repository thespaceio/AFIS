"""The browser demo: its Python glue (run natively here) and the site build."""
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "scripts"))

import afis_web  # noqa: E402
import build_web  # noqa: E402


@pytest.fixture(scope="module")
def info():
    return afis_web.init()


def test_init_enrols_demo_people(info):
    assert [p["id"] for p in info["people"]] == ["demo-01", "demo-02", "demo-03", "demo-04"]
    assert info["enrolled"] == [p["id"] for p in info["people"]]
    assert len(info["prints"]) == 16 and info["threshold"] == pytest.approx(0.18)
    assert "not been validated" in info["note"]


def test_identify_finds_the_right_person(info):
    for person in info["people"]:
        for probe in person["probes"]:
            r = afis_web.identify(probe)
            assert r["candidate"] == person["id"] == r["truth"], probe
            assert r["picture"].startswith("data:image/png;base64,")


def test_verify_genuine_and_impostor(info):
    assert afis_web.verify("demo-01_3", "demo-01")["is_match"]
    r = afis_web.verify("demo-01_3", "demo-02")
    assert not r["is_match"] and r["truth"] == "demo-01"
    assert afis_web.verify("demo-01_3", "nobody")["error"]


def test_threshold_and_analyse(info):
    try:
        afis_web.set_threshold(0.99)
        assert afis_web.identify("demo-03_2")["candidate"] is None
        with pytest.raises(ValueError):
            afis_web.set_threshold(0)
    finally:
        afis_web.set_threshold(info["threshold"])
    a = afis_web.analyse("demo-02_1")
    assert a["pattern"] == "whorl" and a["minutiae"] == a["endings"] + a["bifurcations"] > 12
    assert len(a["stages"]) == 7


def test_only_bundled_demo_prints_are_accepted(info):
    for bad in ("../../etc/passwd", "/etc/passwd", "demo-99_1", ""):
        with pytest.raises(ValueError, match="Unknown demo print"):
            afis_web.identify(bad)


def test_engine_zip_has_no_gui(tmp_path):
    build_web.engine_zip(tmp_path / "afis.zip")
    names = zipfile.ZipFile(tmp_path / "afis.zip").namelist()
    assert "afis_web.py" in names and "afis/system.py" in names
    assert sum(n.startswith("afis/demo_prints/") for n in names) == 16
    assert not [n for n in names if "/gui/" in n or "__pycache__" in n]


def test_pyodide_package_list_matches_worker():
    worker = (ROOT / "web" / "worker.js").read_text()
    for name in build_web.PYODIDE_PACKAGES:
        assert f'"{name}"' in worker
