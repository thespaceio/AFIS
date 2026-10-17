#!/usr/bin/env python3
"""
Regenerate the synthetic demo prints shipped in afis/demo_prints/.

    python scripts/make_demo_prints.py            # write the prints for the seeds in afis/demo.py
    python scripts/make_demo_prints.py --search   # find new seeds (prints DEMO_SEEDS to paste in)

The demo set is curated: --search keeps a candidate finger only if its own
impressions match each other well and every impression scores low against
every impression of the fingers already chosen, so the demo shows clear
results. This makes the demo prints easier than real data; see the README
for accuracy on real fingerprints.
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afis.demo import DEMO_IMPRESSIONS, DEMO_PATTERNS, demo_dir, write_demo_prints  # noqa: E402
from afis.matching import FingerprintMatcher  # noqa: E402
from afis.synthetic import generate_impressions, generate_master  # noqa: E402
from afis.system import AFISSystem  # noqa: E402


def search(min_genuine: float, max_impostor: float, start: int) -> list:
    afis, matcher = AFISSystem(":memory:"), FingerprintMatcher()
    chosen, seeds = [], []
    seed = start
    for pattern in DEMO_PATTERNS:
        while True:
            seed += 1
            imgs = generate_impressions(generate_master(seed, pattern), DEMO_IMPRESSIONS)
            try:
                temps = [afis.process(i).as_template() for i in imgs]
            except ValueError:
                continue
            if min(len(t.minutiae) for t in temps) < AFISSystem.MIN_ENROL_MINUTIAE:
                continue
            gen = min(matcher.match_minutiae(a.minutiae, b.minutiae)[0] for a, b in itertools.combinations(temps, 2))
            if gen < min_genuine:
                continue
            imp = max((matcher.match_minutiae(a.minutiae, b.minutiae)[0]
                       for other in chosen for a in temps for b in other), default=0.0)
            if imp > max_impostor:
                continue
            print(f"{pattern:<12} seed {seed}: lowest genuine {gen:.3f}, highest impostor {imp:.3f}")
            chosen.append(temps)
            seeds.append(seed)
            break
    return seeds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", type=Path, default=None)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--min-genuine", type=float, default=0.25)
    ap.add_argument("--max-impostor", type=float, default=0.10)
    ap.add_argument("--start", type=int, default=3000)
    args = ap.parse_args()
    if args.search:
        print("DEMO_SEEDS =", search(args.min_genuine, args.max_impostor, args.start))
    else:
        out = args.out or demo_dir()
        files = write_demo_prints(out)
        print(f"Wrote {len(files)} demo prints to {out}")
