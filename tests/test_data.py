"""Image loading edge cases and the synthetic fingerprint generator."""
import cv2
import numpy as np
import pytest

from afis.acquisition import AcquisitionError, FingerprintAcquisition
from afis.demo import demo_files, demo_people, write_demo_prints
from afis.matching import FingerprintMatcher
from afis.synthetic import PATTERNS, generate_impressions, generate_master
from afis.system import AFISSystem, EnrollmentError

from conftest import demo


# ---------------------------------------------------------------- acquisition
def test_rejects_empty_and_corrupt_files(tmp_path):
    acq = FingerprintAcquisition()
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(AcquisitionError, match="empty"):
        acq.load(empty)
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"II*\x00" + b"\x00" * 100)
    with pytest.raises(AcquisitionError):
        acq.load(corrupt)
    truncated = tmp_path / "trunc.png"
    truncated.write_bytes(open(demo(1, 1), "rb").read()[:200])
    with pytest.raises(AcquisitionError):
        acq.load(truncated)


def test_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "print.gif"
    p.write_bytes(open(demo(1, 1), "rb").read())
    with pytest.raises(AcquisitionError, match="Unsupported"):
        FingerprintAcquisition().load(p)


def test_decodes_by_content_and_unicode_paths(tmp_path):
    src = cv2.imread(demo(1, 1), cv2.IMREAD_GRAYSCALE)
    p = tmp_path / "empreinte_ñ_指纹.tif"            # PNG data with a .tif name, non-ASCII path
    p.write_bytes(open(demo(1, 1), "rb").read())
    assert np.array_equal(FingerprintAcquisition().load(p), src)


def test_accepts_colour_alpha_and_16bit_arrays():
    g = cv2.imread(demo(1, 1), cv2.IMREAD_GRAYSCALE)
    acq = FingerprintAcquisition()
    assert acq.load(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)).shape == g.shape
    assert acq.load(cv2.cvtColor(g, cv2.COLOR_GRAY2BGRA)).shape == g.shape
    assert acq.load(g.astype(np.uint16) * 257).dtype == np.uint8
    assert acq.load(open(demo(1, 1), "rb").read()).shape == g.shape


def test_large_images_are_downscaled():
    g = cv2.imread(demo(1, 1), cv2.IMREAD_GRAYSCALE)
    big = cv2.resize(g, None, fx=4, fy=4)
    assert max(FingerprintAcquisition().load(big).shape) == 1000


@pytest.mark.parametrize("kind", ["noise", "blobs", "checker"])
def test_non_fingerprints_fail_the_quality_gate(kind):
    rng = np.random.default_rng(1)
    if kind == "noise":
        img = rng.integers(0, 255, (320, 320), dtype=np.uint8)
    elif kind == "blobs":
        img = np.clip(cv2.GaussianBlur(rng.normal(0, 1, (320, 320)), (0, 0), 10) * 2000 + 128, 0, 255).astype(np.uint8)
    else:
        y, x = np.mgrid[0:320, 0:320]
        img = ((((x // 20) + (y // 20)) % 2) * 200 + 20).astype(np.uint8)
    with pytest.raises((EnrollmentError, AcquisitionError, ValueError)):
        AFISSystem(":memory:").enroll(img, "not-a-finger")


# ---------------------------------------------------------------- synthetic generator
def test_generator_is_deterministic():
    a, b = generate_master(42, "whorl"), generate_master(42, "whorl")
    assert np.array_equal(a.ridges, b.ridges)
    assert not np.array_equal(a.ridges, generate_master(43, "whorl").ridges)


@pytest.mark.parametrize("pattern,expected", [("arch", "arch"), ("left_loop", "loop"),
                                              ("right_loop", "loop"), ("whorl", "whorl")])
def test_generated_patterns_are_classified(pattern, expected):
    afis = AFISSystem(":memory:")
    img = generate_impressions(generate_master(7, pattern), 1)[0]
    assert afis.process(img).global_features["pattern_type"] == expected


def test_genuine_scores_exceed_impostor_scores():
    afis, matcher = AFISSystem(":memory:"), FingerprintMatcher()
    t = {}
    for f in range(3):
        for i, img in enumerate(generate_impressions(generate_master(500 + f), 3)):
            t[(f, i)] = afis.process(img).as_template()
    genuine = [matcher.match_templates(t[(f, 0)], t[(f, i)])[0] for f in range(3) for i in (1, 2)]
    impostor = [matcher.match_templates(t[(a, 0)], t[(b, 0)])[0] for a in range(3) for b in range(3) if a < b]
    assert min(genuine) > max(impostor)


def test_bundled_demo_prints_match_the_generator(tmp_path):
    files = demo_files()
    assert len(files) == 24 and len(demo_people()) == 6
    regenerated = write_demo_prints(tmp_path)
    same = sum(np.array_equal(cv2.imread(str(a), 0), cv2.imread(str(b), 0)) for a, b in zip(files, regenerated))
    # Identical with the pinned library versions; allow tiny drift across OpenCV/numpy releases.
    assert same >= 20 or all(np.abs(cv2.imread(str(a), 0).astype(int) - cv2.imread(str(b), 0)).mean() < 3
                             for a, b in zip(files, regenerated))


def test_unknown_pattern_rejected():
    with pytest.raises(ValueError):
        generate_master(1, "spiral")
    assert len(PATTERNS) == 5


def test_demo_prints_are_well_separated():
    """The curated demo set must give clear results with the default matcher."""
    import itertools
    afis, matcher = AFISSystem(":memory:"), FingerprintMatcher()
    t = {(pid, k): afis.process(p).as_template() for pid, paths in demo_people() for k, p in enumerate(paths)}
    gen = [matcher.match_templates(t[a], t[b])[0] for a, b in itertools.combinations(t, 2) if a[0] == b[0]]
    imp = [matcher.match_templates(t[a], t[b])[0] for a, b in itertools.combinations(t, 2) if a[0] != b[0]]
    # The search checks one argument order; the matcher is not exactly symmetric, hence 0.12.
    assert min(gen) >= 0.24 and max(imp) <= 0.12
