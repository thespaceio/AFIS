#!/usr/bin/env python3
"""
Grid-search matcher parameters on a *tuning* set of fingerprint folders.

Templates are extracted once; every matcher configuration then re-scores all
genuine pairs and every cross-finger impostor pair within each folder. Scores
from different folders (sensors) are pooled, never compared with each other.

Keep the tuning data separate from the data you report results on, e.g. tune
on FVC2000 / FVC2004 "B" sets and report on FVC2002 "B" with evaluate.py.

    python scripts/tune.py data/FVC2000/DB1_B data/FVC2004/DB1_B ... --out tune.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afis.acquisition import SUPPORTED_FORMATS, FingerprintAcquisition  # noqa: E402
from afis.evaluation import error_rates  # noqa: E402
from afis.features import FeatureExtractor  # noqa: E402
from afis.matching import FingerprintMatcher, MatcherConfig  # noqa: E402
from afis.preprocessing import PreprocessingPipeline  # noqa: E402
from afis.template import FingerprintTemplate  # noqa: E402

GRID = {
    "match_dist_tol": [12.0, 15.0, 18.0],
    "match_angle_tol_deg": [25, 35],
    "candidates": [12, 30],
    "local_dist_tol": [10.0, 14.0],
    "neighbours": [6, 8],
    "min_overlap_count": [12, 18],
}


def extract(folder: Path):
    acq, pre, ext = FingerprintAcquisition(), PreprocessingPipeline(), FeatureExtractor()
    fingers = defaultdict(list)
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        try:
            m, g = ext.extract_all(pre.preprocess(acq.load(str(p))))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            continue
        f = p.stem.split("_")[0]
        fingers[f].append(FingerprintTemplate(p.stem, f, m, g))
    return fingers


def pairs(fingers):
    gen = [(a, b) for imps in fingers.values() for a, b in itertools.combinations(imps, 2)]
    flat = [t for imps in fingers.values() for t in imps]
    imp = [(a, b) for a, b in itertools.combinations(flat, 2) if a.person_id != b.person_id]
    return gen, imp


def evaluate(cfg: MatcherConfig, sets, impostor_sample: int, rng):
    m = FingerprintMatcher(cfg)
    g_scores, i_scores = [], []
    for gen, imp in sets:
        if impostor_sample and len(imp) > impostor_sample:
            imp = [imp[k] for k in rng.choice(len(imp), impostor_sample, replace=False)]
        g_scores += [m.match_minutiae(a.minutiae, b.minutiae)[0] for a, b in gen]
        i_scores += [m.match_minutiae(a.minutiae, b.minutiae)[0] for a, b in imp]
    return error_rates(g_scores, i_scores)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--impostor-sample", type=int, default=1500, help="impostor pairs per folder (0 = all)")
    ap.add_argument("--grid", type=str, default=None, help="JSON dict overriding the default grid")
    ap.add_argument("--shard", type=str, default="0/1", help="run only configurations k with k %% n == i (i/n)")
    ap.add_argument("--out", type=Path, default=Path("results/tune.csv"))
    args = ap.parse_args()
    grid = dict(GRID, **(json.loads(args.grid) if args.grid else {}))

    sets = []
    for folder in args.folders:
        t0 = time.perf_counter()
        fingers = extract(folder)
        gen, imp = pairs(fingers)
        sets.append((gen, imp))
        print(f"{folder}: {sum(map(len, fingers.values()))} templates, {len(gen)} genuine, "
              f"{len(imp)} impostor pairs ({time.perf_counter() - t0:.0f}s)", flush=True)

    keys = list(grid)
    rows = []
    shard, n_shards = map(int, args.shard.split("/"))
    for k, values in enumerate(itertools.product(*(grid[k] for k in keys))):
        if k % n_shards != shard:
            continue
        params = dict(zip(keys, values))
        cfg = MatcherConfig(**{k: v for k, v in params.items() if k != "match_angle_tol_deg"})
        cfg.match_angle_tol = np.radians(params["match_angle_tol_deg"])
        t0 = time.perf_counter()
        r = evaluate(cfg, sets, args.impostor_sample, np.random.default_rng(0))
        t_far01 = r.threshold_for_far(0.001)
        t_far1 = r.threshold_for_far(0.01)
        row = dict(params, eer=round(r.eer, 4), eer_threshold=round(r.eer_threshold, 4),
                   frr_at_far_1pct=round(r.frr_at(t_far1), 4), threshold_far_1pct=round(t_far1, 4),
                   frr_at_far_01pct=round(r.frr_at(t_far01), 4), threshold_far_01pct=round(t_far01, 4),
                   seconds=round(time.perf_counter() - t0, 1),
                   rates_by_threshold=r.summary()["rates_by_threshold"])
        rows.append(row)
        print(json.dumps(row), flush=True)

    if not rows:
        print("No configurations in this shard.")
        return
    rows.sort(key=lambda r: (r["frr_at_far_1pct"], r["eer"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "rates_by_threshold"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\nBest 10 (by FRR at FAR = 1%, then EER):")
    for r in rows[:10]:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
