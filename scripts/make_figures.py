#!/usr/bin/env python3
"""Regenerate docs/pipeline.png (every processing stage) from a bundled synthetic demo print."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afis.demo import demo_dir  # noqa: E402
from afis.system import AFISSystem  # noqa: E402
from afis.visualization import draw_minutiae, montage  # noqa: E402


def main(out: Path) -> None:
    afis = AFISSystem(":memory:")
    p = afis.process(str(demo_dir() / "demo-01_1.png"))
    stages = p.result.stages()
    images = [img if img.ndim == 2 else img[:, :, ::-1] for img in stages.values()]
    g = p.global_features
    images.append(draw_minutiae(p.image, p.minutiae, g["core_points"], g["delta_points"], scale=1))
    labels = [f"{k + 1}. {name}" for k, name in enumerate(stages)] + [f"{len(stages) + 1}. Minutiae + singular points"]
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "pipeline.png"), montage(images, labels, cols=4, cell=230))
    print("saved", out / "pipeline.png")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs"))
