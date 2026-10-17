"""Smoke test: the GUI builds and runs a full enrol -> verify -> identify cycle off-screen."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from cryptography.fernet import Fernet  # noqa: E402

from conftest import sample  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_enroll_verify_identify(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AFIS_KEY", Fernet.generate_key().decode())
    from afis.gui.app import MainWindow

    win = MainWindow(str(tmp_path / "gui.db"))

    def settle():
        for _ in range(5):
            win.pool.waitForDone()
            app.processEvents()

    enroll, verify, identify, database = win.pages[1], win.pages[2], win.pages[3], win.pages[4]
    enroll.drop.set_path(str(sample("101_1.tif")))
    enroll.person.setText("alice")
    enroll.enroll()
    settle()
    assert "Enrolled alice" in enroll.message.text()
    assert database.table.rowCount() == 1

    verify.drop.set_path(str(sample("101_2.tif")))
    verify.person.setEditText("alice")
    verify.verify()
    settle()
    assert "MATCH" in verify.result.verdict.text() and "NO" not in verify.result.verdict.text()

    identify.drop.set_path(str(sample("101_2.tif")))
    identify.identify()
    settle()
    assert identify.table.item(0, 1).text() == "alice"
    win.close()
