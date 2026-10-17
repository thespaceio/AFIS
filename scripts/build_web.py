#!/usr/bin/env python3
"""
Build the browser demo (web/) into a static site that can be served from anywhere,
e.g. GitHub Pages.

    python scripts/build_web.py --pyodide /path/to/pyodide --out site

--pyodide is an unpacked Pyodide release (https://github.com/pyodide/pyodide/releases,
the full `pyodide-<version>.tar.bz2`). Only the runtime and the packages AFIS needs
are copied, so the demo is self-hosted and makes no requests to third-party CDNs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PACKAGE = ROOT / "afis"

PAGE_FILES = ["index.html", "app.js", "worker.js", "style.css"]
PYODIDE_CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
# Must match PACKAGES in web/worker.js; dependencies are added from the lock file.
PYODIDE_PACKAGES = ["numpy", "scipy", "scikit-image", "opencv-python", "cryptography", "sqlite3"]


def version() -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', (PACKAGE / "__init__.py").read_text())
    return m.group(1) if m else "dev"


def package_closure(lock: dict, names) -> list:
    packages = lock["packages"]
    norm = {k.lower().replace("_", "-"): k for k in packages}
    seen, todo = set(), list(names)
    while todo:
        key = norm[todo.pop().lower().replace("_", "-")]
        if key not in seen:
            seen.add(key)
            todo.extend(packages[key].get("depends", []))
    return sorted(seen)


def copy_pyodide(src: Path, dest: Path) -> int:
    missing = [f for f in PYODIDE_CORE if not (src / f).exists()]
    if missing:
        sys.exit(f"{src} does not look like an unpacked Pyodide release (missing {', '.join(missing)})")
    dest.mkdir(parents=True)
    for f in PYODIDE_CORE:
        shutil.copy2(src / f, dest / f)
    lock = json.loads((src / "pyodide-lock.json").read_text())
    total = 0
    for name in package_closure(lock, PYODIDE_PACKAGES):
        f = lock["packages"][name]["file_name"]
        shutil.copy2(src / f, dest / f)
        total += (dest / f).stat().st_size
    return total + sum((dest / f).stat().st_size for f in PYODIDE_CORE)


def engine_zip(dest: Path) -> None:
    """The afis package (without the desktop GUI) plus the browser glue module."""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PACKAGE.rglob("*")):
            rel = f.relative_to(ROOT)
            if f.is_dir() or "__pycache__" in rel.parts or rel.parts[1] == "gui":
                continue
            if f.suffix in (".py", ".png"):
                z.write(f, rel.as_posix())
        z.write(WEB / "afis_web.py", "afis_web.py")


def build(pyodide: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ver = version()
    for f in PAGE_FILES:
        text = (WEB / f).read_text(encoding="utf-8")
        if f == "index.html":
            text = text.replace('<html lang="en">', f'<html lang="en" data-version="{ver}">', 1)
        (out / f).write_text(text, encoding="utf-8")
    shutil.copy2(PACKAGE / "gui" / "icon.png", out / "icon.png")
    (out / "prints").mkdir()
    for p in sorted((PACKAGE / "demo_prints").glob("demo-*.png")):
        shutil.copy2(p, out / "prints" / p.name)
    engine_zip(out / "afis.zip")
    size = copy_pyodide(pyodide, out / "pyodide")
    (out / ".nojekyll").write_text("")
    print(f"Built AFIS {ver} browser demo in {out} (Pyodide runtime + packages: {size / 1e6:.0f} MB)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pyodide", type=Path, required=True, help="unpacked Pyodide release folder")
    ap.add_argument("--out", type=Path, default=ROOT / "site", help="output folder (default: site/)")
    args = ap.parse_args(argv)
    build(args.pyodide.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
