#!/usr/bin/env python3
"""
Regenerate the README screenshots by driving the real GUI with the bundled
synthetic demo prints (no real fingerprints are shown).

    python scripts/gui_screenshots.py docs                      # off-screen
    QT_QPA_PLATFORM=xcb python scripts/gui_screenshots.py docs  # under X / Xvfb (nicer fonts)
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.fernet import Fernet  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from afis.demo import demo_dir  # noqa: E402
from afis.gui import app as gui  # noqa: E402
from afis.gui.style import STYLE  # noqa: E402


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    os.environ["AFIS_KEY"] = Fernet.generate_key().decode()
    app = QApplication([])
    app.setStyleSheet(STYLE)
    win = gui.MainWindow(os.path.join(tempfile.mkdtemp(), "demo.db"))
    win.resize(1360, 900)
    win.show()
    d = lambda f, i: str(demo_dir() / f"demo-{f:02d}_{i}.png")  # noqa: E731

    def settle():
        for _ in range(8):
            win.pool.waitForDone()
            app.processEvents()

    def shot(cls, name):
        win.show_page(win.pages.index(win.page(cls)))
        settle()
        win.grab().save(str(out / name))
        print("saved", out / name)

    shot(gui.HomePage, "gui_home_empty.png")
    win.page(gui.HomePage).load_demo()
    settle()
    shot(gui.HomePage, "gui_home.png")

    win.page(gui.AnalysePage).drop.set_path(d(2, 1))
    shot(gui.AnalysePage, "gui_analyse.png")

    enroll = win.page(gui.EnrollPage)
    enroll.drop.set_path(d(1, 2))
    enroll.person.setText("Adébáyọ̀ Ọlá")
    enroll.finger.setCurrentIndex(7)
    enroll.enroll()
    settle()
    shot(gui.EnrollPage, "gui_enroll.png")

    verify = win.page(gui.VerifyPage)
    win.show_page(win.pages.index(verify))
    verify.drop.set_path(d(2, 3))
    verify.person.setEditText("demo-02")
    verify.verify()
    shot(gui.VerifyPage, "gui_verify_match.png")
    verify.person.setEditText("demo-05")
    verify.verify()
    shot(gui.VerifyPage, "gui_verify_reject.png")

    ident = win.page(gui.IdentifyPage)
    ident.drop.set_path(d(3, 2))
    ident.identify()
    shot(gui.IdentifyPage, "gui_identify.png")

    shot(gui.DatabasePage, "gui_database.png")
    audit = win.page(gui.AuditPage)
    audit.verify_chain()
    shot(gui.AuditPage, "gui_audit.png")
    win.page(gui.SettingsPage).check()
    shot(gui.SettingsPage, "gui_settings.png")
    win.close()


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs"))
