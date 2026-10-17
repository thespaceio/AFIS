#!/usr/bin/env python3
"""
Measure AFIS accuracy on a folder of fingerprint images.

Images must follow the FVC naming convention  <finger>_<impression>.<ext>
(e.g. 101_1.tif, 101_2.tif ...). Protocol (as in FVC):
  * genuine pairs:  every pair of impressions of the same finger
  * impostor pairs: first impression of each finger vs first impression of every other finger

Several folders can be given; finger IDs are prefixed with the folder name so
fingers from different databases never count as genuine pairs.

Examples
    python scripts/evaluate.py data/FVC2002/DB1_B
    python scripts/evaluate.py data/FVC2002/DB?_B --all-impostors      # all four FVC2002 B sets
    python scripts/evaluate.py --synthetic 100 --impressions 8          # generated fingers
    python scripts/evaluate.py samples --augment 6 --mirror --all-impostors

Interpreting the numbers: with N impostor pairs the smallest FAR that can be
measured is about 1/N, and the 95% upper bound on a FAR measured as 0 is about
3/N ("rule of three"). The summary reports both.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afis.acquisition import SUPPORTED_FORMATS, FingerprintAcquisition  # noqa: E402
from afis.system import AFISSystem  # noqa: E402
from afis.evaluation import error_rates, plot_report, synthetic_impression  # noqa: E402
from afis.features import FeatureExtractor  # noqa: E402
from afis.matching import FingerprintMatcher  # noqa: E402
from afis.preprocessing import PreprocessingPipeline  # noqa: E402
from afis.template import FingerprintTemplate  # noqa: E402


def load_fingers(folders, augment: int, mirror: bool, seed: int, synthetic: int = 0, impressions: int = 8):
    acq = FingerprintAcquisition()
    fingers = defaultdict(list)
    prefix = len(folders) > 1
    for folder in folders:
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in SUPPORTED_FORMATS:
                continue
            finger = (f"{folder.name}/" if prefix else "") + p.stem.split("_")[0]
            fingers[finger].append((p.name, acq.load(str(p))))
    if synthetic:
        from afis.synthetic import generate_dataset
        for f, i, img in generate_dataset(synthetic, impressions, first_seed=seed * 1000):
            fingers[f"syn{f:04d}"].append((f"syn{f:04d}_{i}", img))
    if mirror:
        for f in list(fingers):
            fingers[f + "m"] = [(n + " (mirrored)", cv2.flip(img, 1)) for n, img in fingers[f]]
    if augment:
        rng = np.random.default_rng(seed)
        for f, imps in fingers.items():
            base = list(imps)
            for k in range(augment):
                name, img = base[k % len(base)]
                imps.append((f"{name} (synthetic {k + 1})", synthetic_impression(img, rng)))
    return fingers


def multi_enrolment(templates, k: int, matcher) -> dict:
    """Error rates when each finger is enrolled with its first k impressions (best score over the gallery)."""
    by_finger = defaultdict(list)
    for (f, i), t in sorted(templates.items()):
        by_finger[f].append(t)
    galleries = {f: ts[:k] for f, ts in by_finger.items() if len(ts) > k}
    probes = {f: by_finger[f][k:] for f in galleries}
    genuine, impostor = [], []
    for f, ps in probes.items():
        for p in ps:
            genuine.append(max(matcher.match_templates(p, g)[0] for g in galleries[f]))
            # One impostor attempt per probe against each other finger's gallery.
            for f2, gal in galleries.items():
                if f2 != f:
                    impostor.append(max(matcher.match_templates(p, g)[0] for g in gal))
    r = error_rates(genuine, impostor)
    thr = matcher.config.threshold
    return {"gallery_size": k, "genuine_attempts": len(genuine), "impostor_attempts": len(impostor),
            "eer": round(r.eer, 4), "far_at_current_threshold": round(r.far_at(thr), 5),
            "frr_at_current_threshold": round(r.frr_at(thr), 4),
            "frr_at_far_1pct": round(r.frr_at(r.threshold_for_far(0.01)), 4),
            "rates_by_threshold": r.summary()["rates_by_threshold"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", type=Path, nargs="*", help="folders of <finger>_<impression> images")
    ap.add_argument("--synthetic", type=int, default=0, help="add this many generated fingers")
    ap.add_argument("--impressions", type=int, default=8, help="impressions per generated finger")
    ap.add_argument("--title", default=None, help="title for the report")
    ap.add_argument("--gallery", type=int, default=0,
                    help="also report multi-impression enrolment: the first K impressions of each finger are "
                         "enrolled, the rest are probes, and a probe's score is its best score over a gallery")
    ap.add_argument("--augment", type=int, default=0, help="synthetic impressions to add per finger")
    ap.add_argument("--mirror", action="store_true", help="add mirrored prints as extra fingers")
    ap.add_argument("--all-impostors", action="store_true",
                    help="use every cross-finger pair as an impostor pair (more pairs for small sets)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    if not args.folders and not args.synthetic:
        ap.error("give at least one folder or --synthetic N")
    fingers = load_fingers(args.folders, args.augment, args.mirror, args.seed, args.synthetic, args.impressions)
    if not fingers:
        sys.exit(f"No images found in {', '.join(map(str, args.folders))}")
    title = args.title or ("AFIS evaluation - " + (", ".join(f.name for f in args.folders) or
                                                    f"{args.synthetic} synthetic fingers"))

    pre, ext, matcher = PreprocessingPipeline(), FeatureExtractor(), FingerprintMatcher()
    templates = {}
    failed = []
    qualities = []
    low_quality = 0
    t0 = time.perf_counter()
    for f, imps in fingers.items():
        for idx, (name, img) in enumerate(imps):
            try:
                r = pre.preprocess(img)
                m, g = ext.extract_all(r)
                templates[(f, idx)] = FingerprintTemplate(f"{f}-{idx}", f, m, g)
                qualities.append(r.quality)
                if r.quality < AFISSystem.MIN_ENROL_QUALITY or len(m) < AFISSystem.MIN_ENROL_MINUTIAE:
                    low_quality += 1
            except Exception as e:  # noqa: BLE001 - report and continue
                failed.append((name, str(e)))
    t_extract = time.perf_counter() - t0

    genuine, impostor = [], []
    t0 = time.perf_counter()
    genuine_matched = []
    for f, imps in fingers.items():
        idxs = [i for i in range(len(imps)) if (f, i) in templates]
        for i, j in itertools.combinations(idxs, 2):
            score, details = matcher.match_templates(templates[(f, i)], templates[(f, j)])
            genuine.append(score)
            genuine_matched.append(details.get("matched_minutiae", 0))
    if args.all_impostors:
        keys = sorted(templates)
        impostor_pairs = [(a, b) for a, b in itertools.combinations(keys, 2) if a[0] != b[0]]
    else:
        firsts = [(f, min(i for (ff, i) in templates if ff == f)) for f in fingers if any(k[0] == f for k in templates)]
        impostor_pairs = list(itertools.combinations(firsts, 2))
    for a, b in impostor_pairs:
        impostor.append(matcher.match_templates(templates[a], templates[b])[0])
    t_match = time.perf_counter() - t0

    rates = error_rates(genuine, impostor)
    summary = rates.summary()
    if args.gallery:
        summary["multi_enrolment"] = multi_enrolment(templates, args.gallery, matcher)
    n_imgs = sum(len(v) for v in fingers.values())
    thr = matcher.config.threshold
    summary.update({
        "title": title,
        "fingers": len(fingers),
        "images": n_imgs,
        "failed_to_process": len(failed),
        "below_enrolment_gate": low_quality,
        "failure_to_enrol_rate": round(low_quality / max(1, n_imgs), 4),
        "quality_median": round(float(np.median(qualities)), 3) if qualities else None,
        "minutiae_median": float(np.median([len(t.minutiae) for t in templates.values()])) if templates else None,
        "genuine_matched_median": float(np.median(genuine_matched)) if genuine_matched else None,
        "min_measurable_far": round(1 / max(1, len(impostor)), 6),
        "far_upper_95_if_zero": round(3 / max(1, len(impostor)), 6),
        "avg_extract_seconds": round(t_extract / max(1, n_imgs), 3),
        "avg_match_ms": round(1000 * t_match / max(1, len(genuine) + len(impostor)), 2),
        "current_threshold": thr,
        "far_at_current_threshold": round(rates.far_at(thr), 4),
        "frr_at_current_threshold": round(rates.frr_at(thr), 4),
        "far_at_0.25": round(rates.far_at(0.25), 4),
        "frr_at_0.25": round(rates.frr_at(0.25), 4),
    })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(args.out / "scores.npz", genuine=np.array(genuine), impostor=np.array(impostor))
    plot_report(rates, str(args.out / "score_report.png"), title=title)

    print(json.dumps(summary, indent=2))
    for name, err in failed:
        print(f"  could not process {name}: {err}")
    print(f"\nSaved {args.out / 'summary.json'}, {args.out / 'scores.npz'} and {args.out / 'score_report.png'}")


if __name__ == "__main__":
    main()
