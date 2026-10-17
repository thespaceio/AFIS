import json

import numpy as np
import pytest
from cryptography.fernet import Fernet

from afis.acquisition import AcquisitionError, FingerprintAcquisition
from afis.database import TemplateDatabase, TemplateIntegrityError
from afis.features import FeatureExtractor, Minutia
from afis.matching import FingerprintMatcher
from afis.preprocessing import PreprocessingPipeline
from afis.system import AFISSystem, EnrollmentError
from afis.template import FingerprintTemplate

from conftest import ridge_pattern, sample


# ---------------------------------------------------------------- preprocessing
@pytest.mark.parametrize("angle", [0, 30, 60, 90, 135])
def test_orientation_and_wavelength_on_synthetic_ridges(angle):
    r = PreprocessingPipeline().preprocess(ridge_pattern(angle))
    est = np.degrees(np.median(r.orientation[r.mask])) % 180
    assert min(abs(est - angle), 180 - abs(est - angle)) < 3
    assert 8 <= r.median_wavelength <= 10


def test_skeleton_follows_dark_ridges():
    path = sample("101_1.tif")
    img = FingerprintAcquisition().load(path)
    r = PreprocessingPipeline().preprocess(img)
    assert img[r.skeleton].mean() < img[r.mask].mean() - 40


# ---------------------------------------------------------------- features
def test_crossing_number_ending_and_bifurcation():
    sk = np.zeros((20, 20), bool)
    sk[10, 2:15] = True                 # a horizontal ridge ending at x=14
    sk[3:10, 8] = True                  # a branch joining it from above at x=8
    cn = FeatureExtractor.crossing_number(sk)
    assert cn[10, 14] == 1 and cn[10, 2] == 1
    assert cn[10, 8] == 3


def test_minutia_validation_and_serialisation():
    m = Minutia(5, 7, 7.0, "ending", 1.4)
    assert 0 <= m.orientation < 2 * np.pi and m.confidence == 1.0
    assert Minutia.from_list(m.to_list()).x == 5
    with pytest.raises(ValueError):
        Minutia(1, 1, 0, "loop")


def test_real_print_is_a_loop_with_a_core():
    afis = AFISSystem(":memory:")
    p = afis.process(sample("101_1.tif"))
    assert p.global_features["pattern_type"] == "loop"
    assert len(p.global_features["core_points"]) == 1
    assert 12 <= len(p.minutiae) <= 80


# ---------------------------------------------------------------- matching
def _random_minutiae(rng, n=35):
    return [Minutia(int(rng.uniform(40, 300)), int(rng.uniform(40, 300)), rng.uniform(0, 2 * np.pi),
                    rng.choice(["ending", "bifurcation"])) for _ in range(n)]


def _moved(ms, angle_deg, dx, dy):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return [Minutia(int(round(c * m.x - s * m.y + dx)), int(round(s * m.x + c * m.y + dy)),
                    m.orientation + a, m.minutiae_type) for m in ms]


def test_identical_minutiae_score_one():
    ms = _random_minutiae(np.random.default_rng(1))
    score, d = FingerprintMatcher().match_minutiae(ms, ms)
    assert score == pytest.approx(1.0) and d["matched_minutiae"] == len(ms)


def test_rotated_and_shifted_print_still_matches():
    ms = _random_minutiae(np.random.default_rng(2))
    score, d = FingerprintMatcher().match_minutiae(ms, _moved(ms, 35, 60, -20))
    assert score > 0.8
    assert abs(d["alignment"]["rotation_deg"] - 35) < 3


def test_unrelated_prints_score_low():
    rng = np.random.default_rng(3)
    scores = [FingerprintMatcher().match_minutiae(_random_minutiae(rng), _random_minutiae(rng))[0] for _ in range(20)]
    assert max(scores) < FingerprintMatcher().config.threshold


# ---------------------------------------------------------------- storage
def _template():
    return FingerprintTemplate("alice-1", "alice", _random_minutiae(np.random.default_rng(4), 20),
                               {"pattern_type": "loop", "quality": 0.8})


def test_template_json_roundtrip_and_integrity():
    t = _template()
    t2 = FingerprintTemplate.from_json(t.to_json())
    assert t2.is_intact() and len(t2.minutiae) == 20
    data = json.loads(t.to_json())
    data["minutiae"][0][0] += 5
    assert not FingerprintTemplate.from_dict(data).is_intact()


def test_database_encrypts_templates(tmp_path):
    key = Fernet.generate_key()
    db_file = str(tmp_path / "t.db")
    db = TemplateDatabase(db_file, key)
    db.save_template(_template(), b"png-bytes")
    raw = db.conn.execute("SELECT template_blob FROM templates").fetchone()[0]
    assert b"minutiae" not in raw and b"alice" not in raw
    assert db.load_template("alice-1").person_id == "alice"
    assert db.load_image("alice-1") == b"png-bytes"
    assert [e["action"] for e in db.audit_entries()] == ["ENROLL"]
    db.close()

    with pytest.raises(TemplateIntegrityError):
        TemplateDatabase(db_file, Fernet.generate_key()).load_template("alice-1")


def test_tampered_blob_is_rejected(tmp_path):
    db = TemplateDatabase(str(tmp_path / "t.db"), Fernet.generate_key())
    db.save_template(_template())
    db.conn.execute("UPDATE templates SET template_blob = ?", (b"gAAAAAtampered",))
    with pytest.raises(TemplateIntegrityError):
        db.load_all_templates()


# ---------------------------------------------------------------- acquisition
def test_acquisition_rejects_bad_images():
    acq = FingerprintAcquisition()
    with pytest.raises(AcquisitionError):
        acq.load(np.full((300, 300), 128, np.uint8))          # no contrast
    with pytest.raises(AcquisitionError):
        acq.load(ridge_pattern(0, size=100))                   # too small
    with pytest.raises(FileNotFoundError):
        acq.load("does/not/exist.png")


# ---------------------------------------------------------------- end to end
def test_enroll_verify_identify(tmp_path):
    afis = AFISSystem(str(tmp_path / "afis.db"), key=Fernet.generate_key())
    afis.enroll(sample("101_1.tif"), "alice", 2)
    afis.enroll(sample("102_1.tif"), "bob", 2)

    ok, score, _ = afis.verify(sample("101_2.tif"), "alice")
    assert ok and score > afis.matcher.config.threshold
    ok, _, _ = afis.verify(sample("101_2.tif"), "bob")
    assert not ok

    ranked = afis.identify(sample("102_2.tif"))
    assert ranked[0][0] == "bob" and ranked[0][2]["is_match"]
    assert afis.verify(sample("101_2.tif"), "nobody")[2]["error"]
    assert afis.delete_person("alice") == 1


def test_enroll_rejects_empty_id():
    with pytest.raises(EnrollmentError):
        AFISSystem(":memory:").enroll(sample("101_1.tif"), "   ")
