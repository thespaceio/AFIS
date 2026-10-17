"""
Glue between the browser demo (web/worker.js) and the AFIS engine.

The demo runs the same `afis` package as the desktop app, inside Pyodide. It
only ever sees the bundled synthetic demo prints: there is no upload, and the
encrypted database lives in memory and disappears when the tab is closed.
Every function returns plain JSON-able data; images are PNG data URLs.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from cryptography.fernet import Fernet

from afis import __version__
from afis.demo import demo_files, demo_people
from afis.system import AFISSystem
from afis.visualization import draw_match, draw_minutiae

NOTE = ("Scores are probabilistic evidence, not proof of identity. AFIS is an educational system "
        "and has not been validated for forensic or law-enforcement decisions.")

_afis: Optional[AFISSystem] = None
_files: Dict[str, Path] = {}


def _png(img: np.ndarray, max_side: int = 0) -> str:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def _bgr_png(img: np.ndarray, max_side: int = 0) -> str:
    """For drawings that are already BGR (the visualization helpers)."""
    return _png(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), max_side)


def _path(name: str) -> Path:
    if name not in _files:
        raise ValueError(f"Unknown demo print: {name}")
    return _files[name]


def init(progress=None) -> dict:
    """Create an in-memory, encrypted database and enrol impression 1 of each demo person."""
    global _afis
    _files.clear()
    _files.update({p.stem: p for p in demo_files()})
    _afis = AFISSystem(":memory:", key=Fernet.generate_key())
    people = _afis.enroll_demo(progress=progress)
    return {
        "version": __version__,
        "threshold": _afis.threshold,
        "people": [{"id": pid, "enrolled": Path(paths[0]).stem, "probes": [Path(p).stem for p in paths[1:]]}
                   for pid, paths in demo_people()],
        "prints": sorted(_files),
        "enrolled": people,
        "note": NOTE,
    }


def set_threshold(value: float) -> float:
    _afis.threshold = value
    return _afis.threshold


def analyse(name: str) -> dict:
    p = _afis.process(str(_path(name)))
    g = p.global_features
    return {
        "name": name,
        "quality": round(float(p.result.quality), 3),
        "minutiae": len(p.minutiae),
        "endings": sum(m.minutiae_type == "ending" for m in p.minutiae),
        "bifurcations": sum(m.minutiae_type != "ending" for m in p.minutiae),
        "pattern": str(g.get("pattern_type", "unknown")),
        "cores": len(g.get("core_points", [])),
        "deltas": len(g.get("delta_points", [])),
        "overlay": _bgr_png(draw_minutiae(p.image, p.minutiae, g.get("core_points", []), g.get("delta_points", []))),
        "stages": [{"name": k, "image": _png(v, 360)} for k, v in p.result.stages().items()],
    }


def _picture(details: dict, is_match: bool) -> Optional[str]:
    fid = details.get("fingerprint_id")
    if not fid or _afis.last is None:
        return None
    ref_t = _afis.database.load_template(fid)
    ref_img = _afis.reference_image(fid)
    if ref_t is None or ref_img is None:
        return None
    return _bgr_png(draw_match(_afis.last.image, _afis.last.minutiae, ref_img, ref_t.minutiae,
                               details.get("pairs", []), is_match=is_match))


def identify(name: str) -> dict:
    ranked = _afis.identify(str(_path(name)), top_k=5)
    rows: List[dict] = [{"person": pid, "score": round(float(score), 4), "is_match": bool(d["is_match"]),
                         "matched": int(d.get("matched_minutiae", 0))} for pid, score, d in ranked]
    top = ranked[0] if ranked else None
    return {
        "probe": name,
        "threshold": _afis.threshold,
        "candidates": rows,
        "candidate": top[0] if top and top[2]["is_match"] else None,
        "picture": _picture(top[2], top[2]["is_match"]) if top else None,
        "reference": f"{top[0]}_1" if top else None,     # the demo enrols impression 1
        "truth": name.split("_")[0],
    }


def verify(name: str, person: str) -> dict:
    ok, score, details = _afis.verify(str(_path(name)), person)
    return {
        "probe": name,
        "person": person,
        "threshold": _afis.threshold,
        "is_match": bool(ok),
        "score": round(float(score), 4),
        "matched": int(details.get("matched_minutiae", 0)),
        "error": details.get("error"),
        "picture": None if "error" in details else _picture(details, ok),
        "truth": name.split("_")[0],
    }

