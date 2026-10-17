#!/usr/bin/env python3
"""
Regenerate the README screenshots: drives the GUI off-screen with sample prints.

    QT_QPA_PLATFORM=offscreen python scripts/gui_screenshots.py samples docs
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.fernet import Fernet  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from afis.gui.app import MainWindow  # noqa: E402
from afis.gui.style import STYLE  # noqa: E402


def main(samples: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    os.environ["AFIS_KEY"] = Fernet.generate_key().decode()
    app = QApplication([])
    app.setStyleSheet(STYLE)
    win = MainWindow(os.path.join(tempfile.mkdtemp(), "demo.db"))
    win.resize(1360, 900)
    win.show()

    def settle():
        for _ in range(5):
            win.pool.waitForDone()
            app.processEvents()

    def shot(name):
        settle()
        win.grab().save(str(out / name))
        print("saved", out / name)

    s = lambda n: str(samples / n)  # noqa: E731

    # Analyse
    win.show_page(0)
    win.pages[0].drop.set_path(s("101_1.tif"))
    shot("gui_analyse.png")

    # Enroll two people
    enroll = win.pages[1]
    win.show_page(1)
    for person, img, finger in (("tolu.adeyemi", "102_1.tif", 2), ("faith.alabi", "101_1.tif", 2)):
        enroll.drop.set_path(s(img))
        enroll.person.setText(person)
        enroll.finger.setCurrentIndex(finger)
        enroll.enroll()
        settle()
    shot("gui_enroll.png")

    # Verify: genuine, then impostor
    verify = win.pages[2]
    win.show_page(2)
    verify.drop.set_path(s("101_2.tif"))
    verify.person.setEditText("faith.alabi")
    verify.verify()
    shot("gui_verify_match.png")
    verify.drop.set_path(s("102_2.tif"))
    verify.person.setEditText("faith.alabi")
    verify.verify()
    shot("gui_verify_reject.png")

    # Identify
    ident = win.pages[3]
    win.show_page(3)
    ident.drop.set_path(s("102_2.tif"))
    ident.identify()
    shot("gui_identify.png")

    win.show_page(4)
    shot("gui_database.png")
    win.show_page(5)
    shot("gui_audit.png")
    win.close()


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples"), Path(sys.argv[2] if len(sys.argv) > 2 else "docs"))
