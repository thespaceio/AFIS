#!/usr/bin/env python3
"""
Measure AFIS accuracy on a folder of fingerprint images.

Images must follow the FVC naming convention  <finger>_<impression>.<ext>
(e.g. 101_1.tif, 101_2.tif ...). Protocol (as in FVC):
  * genuine pairs:  every pair of impressions of the same finger
  * impostor pairs: first impression of each finger vs first impression of every other finger

Examples
    python scripts/evaluate.py data/FVC2002/DB1_B
    python scripts/evaluate.py samples --augment 6        # add 6 synthetic impressions per print
    python scripts/evaluate.py samples --augment 6 --mirror --all-impostors
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afis.acquisition import SUPPORTED_FORMATS, FingerprintAcquisition  # noqa: E402
from afis.evaluation import error_rates, plot_report, synthetic_impression  # noqa: E402
from afis.features import FeatureExtractor  # noqa: E402
from afis.matching import FingerprintMatcher  # noqa: E402
from afis.preprocessing import PreprocessingPipeline  # noqa: E402
from afis.template import FingerprintTemplate  # noqa: E402


def load_fingers(folder: Path, augment: int, mirror: bool, seed: int):
    acq = FingerprintAcquisition()
    fingers = defaultdict(list)
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        finger = p.stem.split("_")[0]
        fingers[finger].append((p.name, acq.load(str(p))))
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--augment", type=int, default=0, help="synthetic impressions to add per finger")
    ap.add_argument("--mirror", action="store_true", help="add mirrored prints as extra fingers")
    ap.add_argument("--all-impostors", action="store_true",
                    help="use every cross-finger pair as an impostor pair (more pairs for small sets)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    fingers = load_fingers(args.folder, args.augment, args.mirror, args.seed)
    if not fingers:
        sys.exit(f"No images found in {args.folder}")

    pre, ext, matcher = PreprocessingPipeline(), FeatureExtractor(), FingerprintMatcher()
    templates = {}
    failed = []
    t0 = time.perf_counter()
    for f, imps in fingers.items():
        for idx, (name, img) in enumerate(imps):
            try:
                r = pre.preprocess(img)
                m, g = ext.extract_all(r)
                templates[(f, idx)] = FingerprintTemplate(f"{f}-{idx}", f, m, g)
            except Exception as e:  # noqa: BLE001 - report and continue
                failed.append((name, str(e)))
    t_extract = time.perf_counter() - t0

    genuine, impostor = [], []
    t0 = time.perf_counter()
    for f, imps in fingers.items():
        idxs = [i for i in range(len(imps)) if (f, i) in templates]
        for i, j in itertools.combinations(idxs, 2):
            genuine.append(matcher.match_templates(templates[(f, i)], templates[(f, j)])[0])
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
    n_imgs = sum(len(v) for v in fingers.values())
    summary.update({
        "fingers": len(fingers),
        "images": n_imgs,
        "failed_to_process": len(failed),
        "avg_extract_seconds": round(t_extract / max(1, n_imgs), 3),
        "avg_match_ms": round(1000 * t_match / max(1, len(genuine) + len(impostor)), 2),
        "current_threshold": matcher.config.threshold,
        "far_at_current_threshold": round(rates.far_at(matcher.config.threshold), 4),
        "frr_at_current_threshold": round(rates.frr_at(matcher.config.threshold), 4),
    })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_report(rates, str(args.out / "score_report.png"), title=f"AFIS evaluation - {args.folder.name}")

    print(json.dumps(summary, indent=2))
    for name, err in failed:
        print(f"  could not process {name}: {err}")
    print(f"\nSaved {args.out / 'summary.json'} and {args.out / 'score_report.png'}")


if __name__ == "__main__":
    main()
