"""
Where AFIS keeps its data, and the user's settings.

Default data folder (owner-only permissions):
    Linux / BSD   $XDG_DATA_HOME/afis      (usually ~/.local/share/afis)
    macOS         ~/Library/Application Support/AFIS
    Windows       %APPDATA%\\AFIS

An existing ~/.afis folder from AFIS 2.0 is used as-is so nothing is lost.

Overrides (environment variables):
    AFIS_HOME   data folder
    AFIS_DB     database file (the key file sits next to it)
    AFIS_KEY    Fernet key, instead of the key file
"""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DB_NAME = "afis_templates.db"
SETTINGS_NAME = "settings.json"


def _platform_dir() -> Path:
    home = Path.home()
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming") / "AFIS"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "AFIS"
    return Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share") / "afis"


def secure_mkdir(path: Path) -> Path:
    """Create a folder readable only by the current user (no-op permissions on Windows)."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(path, stat.S_IRWXU)
        except OSError:
            pass
    return path


def data_dir() -> Path:
    env = os.environ.get("AFIS_HOME")
    if env:
        return secure_mkdir(Path(env).expanduser())
    legacy = Path.home() / ".afis"
    if legacy.is_dir() and (legacy / DB_NAME).exists():
        return secure_mkdir(legacy)        # created 0755 by AFIS 2.0: tighten it
    return secure_mkdir(_platform_dir())


def default_db_path() -> str:
    env = os.environ.get("AFIS_DB")
    if env:
        path = Path(env).expanduser()
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(data_dir() / DB_NAME)


@dataclass
class Settings:
    """User preferences, saved as JSON next to the database."""
    threshold: float = 0.18          # match threshold (see README > Accuracy)
    store_images: bool = True        # keep an encrypted copy of enrolment images for match pictures
    accepted_notice: bool = False    # the first-run responsible-use notice has been acknowledged

    @classmethod
    def load(cls, folder: Path) -> "Settings":
        path = Path(folder) / SETTINGS_NAME
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in raw.items() if k in known})
        s.threshold = min(0.95, max(0.01, float(s.threshold)))
        s.store_images = bool(s.store_images)
        s.accepted_notice = bool(s.accepted_notice)
        return s

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / (SETTINGS_NAME + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, folder / SETTINGS_NAME)
