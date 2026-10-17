"""
The desktop app, driven off-screen: every page, the demo flow, error handling,
protected deletes and startup with a wrong key.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from cryptography.fernet import Fernet  # noqa: E402

from conftest import demo, sample  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def dialogs(monkeypatch):
    """Record message boxes instead of blocking on them; answers can be preset."""
    from afis.gui import app as gui
    seen = {"warn": [], "ask": [], "answer": True, "text": ("DELETE", True)}
    monkeypatch.setattr(gui.MainWindow, "warn", lambda self, text: seen["warn"].append(text))
    monkeypatch.setattr(gui.MainWindow, "ask", lambda self, text, title="": (seen["ask"].append(text),
                                                                              seen["answer"])[1])
    monkeypatch.setattr(gui.QInputDialog, "getText", staticmethod(lambda *a, **k: seen["text"]))
    monkeypatch.setattr(gui.MainWindow, "open_folder", lambda self, folder: seen.setdefault("opened", folder))
    return seen


@pytest.fixture
def win(app, tmp_path, monkeypatch, dialogs):
    monkeypatch.setenv("AFIS_KEY", Fernet.generate_key().decode())
    from afis.gui.app import MainWindow
    w = MainWindow(str(tmp_path / "gui.db"))
    yield w
    w.close()


def settle(app, win):
    for _ in range(10):
        win.pool.waitForDone()
        app.processEvents()


def pages(win):
    from afis.gui import app as gui
    return {cls.__name__: win.page(cls) for cls in (gui.HomePage, gui.AnalysePage, gui.EnrollPage, gui.VerifyPage,
                                                     gui.IdentifyPage, gui.DatabasePage, gui.AuditPage,
                                                     gui.SettingsPage)}


def test_every_page_opens(app, win):
    for i in range(len(win.pages)):
        win.show_page(i)
        app.processEvents()
        assert win.stack.currentIndex() == i
    assert win.menuBar().actions()[0].text() == "&File"


def test_demo_identify_verify_flow(app, win, dialogs):
    p = pages(win)
    db = p["DatabasePage"]
    assert db.empty.isVisibleTo(db) and not db.table.isVisibleTo(db)

    p["HomePage"].load_demo()
    settle(app, win)
    assert "Enrolled 6 demo people" in p["HomePage"].demo_msg.text()
    assert db.table.rowCount() == 6 and p["HomePage"].stats.values["People"].text() == "6"
    assert p["VerifyPage"].person.count() == 6

    ident = p["IdentifyPage"]
    assert ident.drop.set_path(demo(3, 2))
    ident.identify()
    settle(app, win)
    assert ident.table.item(0, 1).text() == "demo-03"
    assert "CANDIDATE: demo-03" in ident.result.verdict.text()
    assert ident.result.picture.pixmap() is not None and not ident.result.picture.pixmap().isNull()

    ver = p["VerifyPage"]
    ver.drop.set_path(demo(1, 3))
    ver.person.setEditText("demo-01")
    ver.verify()
    settle(app, win)
    assert "MATCH — consistent with demo-01" in ver.result.verdict.text()
    ver.person.setEditText("demo-02")
    ver.verify()
    settle(app, win)
    assert "NO MATCH" in ver.result.verdict.text()
    ver.person.setEditText("nobody")
    ver.verify()
    settle(app, win)
    assert "No templates enrolled for 'nobody'" in ver.result.verdict.text()
    assert not dialogs["warn"]

    audit = p["AuditPage"]
    audit.refresh()
    actions = {audit.table.item(r, 1).text() for r in range(audit.table.rowCount())}
    assert {"ENROLL", "IDENTIFY", "VERIFY"} <= actions
    audit.verify_chain()
    assert "intact" in audit.chain.text()


def test_enroll_and_analyse(app, win, dialogs):
    p = pages(win)
    enroll = p["EnrollPage"]
    enroll.enroll()
    assert "Choose a fingerprint" in enroll.message.text()
    enroll.drop.set_path(demo(2, 1))
    enroll.enroll()
    assert "Enter a person ID" in enroll.message.text()
    enroll.person.setText("../bad")
    enroll.enroll()
    settle(app, win)
    assert "✗" in enroll.message.text()
    enroll.person.setText("Adébáyọ̀ Ọlá")
    enroll.enroll()
    settle(app, win)
    assert "Enrolled Adébáyọ̀ Ọlá" in enroll.message.text()

    analyse = p["AnalysePage"]
    analyse.drop.set_path(demo(2, 2))           # selecting a file runs the analysis
    settle(app, win)
    assert analyse.stats.values["Pattern"].text() == "Whorl"
    assert analyse.stage_grid.count() == len(win.afis.last.result.stages()) == 7


def test_invalid_files_are_reported(app, win, dialogs, tmp_path):
    p = pages(win)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image at all")
    assert not p["AnalysePage"].drop.set_path(str(bad))
    assert "could not be read" in dialogs["warn"][-1]
    assert not p["EnrollPage"].drop.set_path(str(tmp_path / "missing.tif"))
    assert "not found" in dialogs["warn"][-1]
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert not p["IdentifyPage"].drop.set_path(str(txt))
    assert "not a supported image type" in dialogs["warn"][-1]
    assert p["IdentifyPage"].drop.path is None


def test_identify_with_empty_database(app, win, dialogs):
    ident = pages(win)["IdentifyPage"]
    ident.identify()
    assert "Choose a fingerprint" in dialogs["warn"][-1]
    ident.drop.set_path(demo(1, 1))
    ident.identify()
    settle(app, win)
    assert "database is empty" in ident.result.verdict.text()


def test_protected_deletes(app, win, dialogs):
    p = pages(win)
    p["HomePage"].load_demo()
    settle(app, win)
    db = p["DatabasePage"]

    db.table.setCurrentCell(-1, -1)
    db.delete_selected()
    assert "Select a row" in dialogs["warn"][-1]

    db.filter.setText("demo-02")
    visible = [r for r in range(db.table.rowCount()) if not db.table.isRowHidden(r)]
    assert len(visible) == 1
    db.table.selectRow(visible[0])
    dialogs["answer"] = False                    # user clicks Cancel
    db.delete_selected()
    assert "demo-02" in dialogs["ask"][-1] and db.table.rowCount() == 6
    dialogs["answer"] = True
    db.delete_selected()
    assert db.table.rowCount() == 5
    db.filter.clear()

    db.table.selectRow(0)
    db.delete_selected_template()
    assert db.table.rowCount() == 4

    settings = p["SettingsPage"]
    dialogs["text"] = ("delete", True)           # wrong word: nothing happens
    settings.delete_all()
    assert db.table.rowCount() == 4 and "Nothing was deleted" in dialogs["warn"][-1]
    dialogs["text"] = ("DELETE", False)          # cancelled
    settings.delete_all()
    assert db.table.rowCount() == 4
    dialogs["text"] = ("DELETE", True)
    settings.delete_all()
    assert win.afis.get_statistics()["total_templates"] == 0 and db.empty.isVisibleTo(db)


def test_settings_persist(app, win, tmp_path):
    from afis.config import Settings
    s = pages(win)["SettingsPage"]
    s.threshold.setValue(0.27)
    s.store_images.setChecked(False)
    saved = Settings.load(tmp_path)
    assert saved.threshold == 0.27 and saved.store_images is False
    assert win.afis.threshold == pytest.approx(0.27) and win.afis.store_images is False
    s.check()
    assert "✓ Templates: 0 of 0" in s.check_result.text()


def test_startup_with_wrong_key_shows_error(app, tmp_path, monkeypatch):
    from afis.gui import app as gui
    from afis.system import AFISSystem
    db = tmp_path / "afis_home" / "afis_templates.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    AFISSystem(str(db), key=Fernet.generate_key()).close()
    monkeypatch.setenv("AFIS_KEY", Fernet.generate_key().decode())
    shown = []
    monkeypatch.setattr(gui.QMessageBox, "critical", staticmethod(lambda *a: shown.append(a[2])))
    assert gui.main(["afis-gui", "--accept-notice"]) == 2
    assert "does not match" in shown[0]


def test_first_run_notice(app, tmp_path, monkeypatch):
    from afis.config import Settings
    from afis.gui import app as gui
    s = Settings()
    monkeypatch.setattr(gui.QMessageBox, "exec", lambda self: gui.QMessageBox.Cancel)
    assert gui.show_notice(None, s, tmp_path) is False and not s.accepted_notice
    monkeypatch.setattr(gui.QMessageBox, "exec", lambda self: gui.QMessageBox.Ok)
    assert gui.show_notice(None, s, tmp_path) is True
    assert Settings.load(tmp_path).accepted_notice


def test_fvc_sample_flow(app, win):
    """Same flow on real FVC2002 prints when they are available locally."""
    p = pages(win)
    enroll = p["EnrollPage"]
    enroll.drop.set_path(str(sample("101_1.tif")))
    enroll.person.setText("alice")
    enroll.enroll()
    settle(app, win)
    assert "Enrolled alice" in enroll.message.text()
    ident = p["IdentifyPage"]
    ident.drop.set_path(str(sample("101_2.tif")))
    ident.identify()
    settle(app, win)
    assert ident.table.item(0, 1).text() == "alice"
