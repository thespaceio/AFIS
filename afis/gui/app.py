"""
AFIS desktop application (PySide6).

    python -m afis.gui          or      afis-gui   (after `pip install -e .`)

The database lives in ~/.afis/afis_templates.db unless AFIS_DB is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QAction, QFont, QIcon, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import __version__
from ..system import FINGER_NAMES, AFISSystem
from ..visualization import draw_match, draw_minutiae
from .style import STYLE
from .widgets import Card, DropArea, ImageView, StatGrid, Task, page_header


def default_db_path() -> str:
    env = os.environ.get("AFIS_DB")
    if env:
        return env
    folder = Path.home() / ".afis"
    folder.mkdir(exist_ok=True)
    return str(folder / "afis_templates.db")


def app_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor("#00CC6A"), 4)
    p.setPen(pen)
    for r in (8, 15, 22, 29):
        p.drawArc(32 - r, 36 - r, 2 * r, 2 * r, 20 * 16, 140 * 16)
    p.end()
    return QIcon(pix)


# ====================================================================== pages
class BasePage(QWidget):
    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.win = window
        self.afis: AFISSystem = window.afis
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setObjectName("root")
        self.lay = QVBoxLayout(inner)
        self.lay.setContentsMargins(28, 24, 28, 24)
        self.lay.setSpacing(16)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def refresh(self) -> None:
        """Called when the page is shown or the database changes."""


class AnalysePage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // PIPELINE", "Analyse a fingerprint",
                                       "See every processing stage, the minutiae and the pattern."))
        top = QHBoxLayout()
        self.drop = DropArea()
        self.drop.setFixedWidth(300)
        self.drop.fileSelected.connect(self.run)
        top.addWidget(self.drop)

        self.stats = StatGrid(["Quality", "Minutiae", "Endings", "Bifurcations", "Pattern", "Processing time"], 3)
        self.overlay = ImageView("Minutiae overlay appears here", 300)
        right = Card("Minutiae")
        legend = QLabel("<span style='color:#3CC83C'>●</span> ending &nbsp; "
                        "<span style='color:#1E8CEB'>●</span> bifurcation &nbsp; "
                        "<span style='color:#E62828'>■</span> core &nbsp; "
                        "<span style='color:#E6C800'>▲</span> delta")
        legend.setObjectName("muted")
        right.body.addWidget(self.stats)
        right.body.addWidget(self.overlay, 1)
        right.body.addWidget(legend)
        top.addWidget(right, 1)
        self.lay.addLayout(top)

        self.stage_card = Card("Processing stages")
        self.stage_grid = QGridLayout()
        self.stage_grid.setSpacing(10)
        self.stage_card.body.addLayout(self.stage_grid)
        self.lay.addWidget(self.stage_card)
        self.lay.addStretch(1)

    def run(self, path: str) -> None:
        self.win.run_task(self.afis.process, self.show_result, path, busy_text="Processing…")

    def show_result(self, p) -> None:
        g = p.global_features
        endings = sum(m.minutiae_type == "ending" for m in p.minutiae)
        self.stats.set("Quality", f"{p.result.quality:.2f}")
        self.stats.set("Minutiae", len(p.minutiae))
        self.stats.set("Endings", endings)
        self.stats.set("Bifurcations", len(p.minutiae) - endings)
        self.stats.set("Pattern", g["pattern_type"].title())
        self.stats.set("Processing time", f"{p.result.timings['total'] * 1000:.0f} ms")
        self.overlay.set_image(draw_minutiae(p.image, p.minutiae, g["core_points"], g["delta_points"]))

        while self.stage_grid.count():
            item = self.stage_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        stages = p.result.stages()
        for k, (name, img) in enumerate(stages.items()):
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            view = ImageView("", 170)
            view.set_image(img if img.ndim == 2 else img[:, :, ::-1].copy())
            cap = QLabel(f"{k + 1}. {name}")
            cap.setObjectName("muted")
            cap.setAlignment(Qt.AlignCenter)
            v.addWidget(view)
            v.addWidget(cap)
            self.stage_grid.addWidget(box, k // 4, k % 4)
        self.win.status(f"Processed in {p.result.timings['total'] * 1000:.0f} ms · ridge spacing "
                        f"{p.result.median_wavelength:.1f}px")


class EnrollPage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // ENROLL", "Enroll a fingerprint",
                                       "Add a person's print to the encrypted template database."))
        row = QHBoxLayout()
        self.drop = DropArea()
        self.drop.setFixedWidth(320)
        row.addWidget(self.drop)

        form = Card("Details")
        self.person = QLineEdit()
        self.person.setPlaceholderText("Person ID, e.g. faith.alabi")
        self.finger = QComboBox()
        for k, name in FINGER_NAMES.items():
            self.finger.addItem(name, k)
        self.button = QPushButton("Enroll fingerprint")
        self.button.setObjectName("primary")
        self.button.clicked.connect(self.enroll)
        self.message = QLabel("")
        self.message.setObjectName("verdictInfo")
        self.message.setWordWrap(True)
        self.message.hide()
        self.preview = ImageView("", 320)
        for w in (QLabel("Person ID"), self.person, QLabel("Finger"), self.finger, self.button, self.message):
            form.body.addWidget(w)
        form.body.addWidget(self.preview, 1)
        row.addWidget(form, 1)
        self.lay.addLayout(row)
        self.lay.addStretch(1)

    def enroll(self) -> None:
        person = self.person.text().strip()
        if not self.drop.path:
            return self.show_msg("Choose a fingerprint image first.", "verdictBad")
        if not person:
            return self.show_msg("Enter a person ID.", "verdictBad")
        self.win.run_task(self.afis.enroll, self.done, self.drop.path, person, self.finger.currentData(),
                          button=self.button, busy_text="Enrolling…", on_error=lambda e: self.show_msg(e, "verdictBad"))

    def done(self, template) -> None:
        g = template.global_features
        self.show_msg(f"✓ Enrolled {template.person_id} · {FINGER_NAMES[template.finger_position]} · "
                      f"{len(template.minutiae)} minutiae · {g.get('pattern_type')} · quality {g.get('quality'):.2f}",
                      "verdictOk")
        p = self.afis.last
        self.preview.set_image(draw_minutiae(p.image, p.minutiae, g["core_points"], g["delta_points"]))
        self.win.database_changed()

    def show_msg(self, text: str, kind: str) -> None:
        self.message.setObjectName(kind)
        self.message.setText(text)
        self.message.style().polish(self.message)
        self.message.show()


class MatchView(Card):
    """Verdict banner, score bar and the side-by-side match picture."""

    def __init__(self, title: str):
        super().__init__(title)
        self.verdict = QLabel("Run a search to see the result")
        self.verdict.setObjectName("verdictInfo")
        self.verdict.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.score_text = QLabel("")
        self.score_text.setObjectName("muted")
        self.picture = ImageView("", 320)
        for w in (self.verdict, self.bar, self.score_text):
            self.body.addWidget(w)
        self.body.addWidget(self.picture, 1)

    def display(self, ok: bool, headline: str, score: float, threshold: float, detail: str) -> None:
        self.verdict.setObjectName("verdictOk" if ok else "verdictBad")
        self.verdict.setText(headline)
        self.verdict.style().polish(self.verdict)
        self.bar.setValue(int(min(1.0, score) * 1000))
        self.score_text.setText(f"Score {score:.3f}  ·  threshold {threshold:.2f}  ·  {detail}")


def match_picture(afis: AFISSystem, details: dict) -> Optional[np.ndarray]:
    fid = details.get("fingerprint_id")
    if not fid or afis.last is None:
        return None
    ref_t = afis.database.load_template(fid)
    ref_img = afis.reference_image(fid)
    if ref_t is None or ref_img is None:
        return draw_minutiae(afis.last.image, afis.last.minutiae)
    return draw_match(afis.last.image, afis.last.minutiae, ref_img, ref_t.minutiae, details.get("pairs", []))


class VerifyPage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // VERIFY · 1:1", "Verify an identity",
                                       "Is this fingerprint really the person they claim to be?"))
        row = QHBoxLayout()
        left = QVBoxLayout()
        self.drop = DropArea()
        self.drop.setFixedWidth(320)
        self.person = QComboBox()
        self.person.setEditable(True)
        self.person.lineEdit().setPlaceholderText("Claimed person ID")
        self.button = QPushButton("Verify")
        self.button.setObjectName("primary")
        self.button.clicked.connect(self.verify)
        left.addWidget(self.drop, 1)
        left.addWidget(QLabel("Claimed identity"))
        left.addWidget(self.person)
        left.addWidget(self.button)
        row.addLayout(left)
        self.result = MatchView("Result")
        row.addWidget(self.result, 1)
        self.lay.addLayout(row)

    def refresh(self) -> None:
        current = self.person.currentText()
        people = sorted({r["person_id"] for r in self.afis.database.list_templates()})
        self.person.clear()
        self.person.addItems(people)
        if current:
            self.person.setEditText(current)

    def verify(self) -> None:
        person = self.person.currentText().strip()
        if not self.drop.path or not person:
            return self.win.warn("Choose an image and a claimed identity first.")
        self.win.run_task(self.afis.verify, self.done, self.drop.path, person, button=self.button,
                          busy_text="Verifying…")

    def done(self, res) -> None:
        ok, score, details = res
        thr = self.afis.matcher.config.threshold
        if "error" in details:
            self.result.display(False, details["error"], 0.0, thr, "")
            return
        person = self.person.currentText().strip()
        head = f"✓ MATCH — verified as {person}" if ok else f"✗ NO MATCH — not {person}"
        self.result.display(ok, head, score, thr, f"{details.get('matched_minutiae', 0)} matched minutiae")
        self.result.picture.set_image(match_picture(self.afis, details))


class IdentifyPage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // IDENTIFY · 1:N", "Identify a fingerprint",
                                       "Search every enrolled person for the best match."))
        row = QHBoxLayout()
        left = QVBoxLayout()
        self.drop = DropArea()
        self.drop.setFixedWidth(320)
        self.button = QPushButton("Search database")
        self.button.setObjectName("primary")
        self.button.clicked.connect(self.identify)
        left.addWidget(self.drop, 1)
        left.addWidget(self.button)
        row.addLayout(left)

        right = QVBoxLayout()
        self.result = MatchView("Best candidate")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Rank", "Person", "Score", "Matched minutiae"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(200)
        right.addWidget(self.result, 1)
        right.addWidget(self.table)
        row.addLayout(right, 1)
        self.lay.addLayout(row)

    def identify(self) -> None:
        if not self.drop.path:
            return self.win.warn("Choose a fingerprint image first.")
        self.win.run_task(self.afis.identify, self.done, self.drop.path, 5, button=self.button,
                          busy_text="Searching…")

    def done(self, ranked) -> None:
        thr = self.afis.matcher.config.threshold
        self.table.setRowCount(0)
        if not ranked:
            self.result.display(False, "The database is empty — enroll someone first.", 0, thr, "")
            return
        for k, (pid, score, d) in enumerate(ranked):
            self.table.insertRow(k)
            for c, val in enumerate([f"{k + 1}{' ✓' if d['is_match'] else ''}", pid, f"{score:.3f}",
                                     str(d.get("matched_minutiae", 0))]):
                item = QTableWidgetItem(val)
                if d["is_match"]:
                    item.setForeground(QColor("#00CC6A"))
                self.table.setItem(k, c, item)
        pid, score, d = ranked[0]
        head = f"✓ IDENTIFIED as {pid}" if d["is_match"] else "✗ No match above the threshold"
        self.result.display(d["is_match"], head, score, thr, f"{d.get('matched_minutiae', 0)} matched minutiae")
        self.result.picture.set_image(match_picture(self.afis, d))


class DatabasePage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // DATABASE", "Enrolled templates",
                                       "Templates and images are encrypted at rest (Fernet). Only metadata is listed here."))
        self.stats = StatGrid(["Templates", "People", "Avg. minutiae", "Database file"], 4)
        self.lay.addWidget(self.stats)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Match threshold"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.9)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(2)
        self.threshold.setValue(self.afis.matcher.config.threshold)
        self.threshold.setToolTip("Higher = fewer false accepts, more false rejects. Default chosen by scripts/evaluate.py")
        self.threshold.valueChanged.connect(self.set_threshold)
        controls.addWidget(self.threshold)
        controls.addStretch(1)
        self.delete_btn = QPushButton("Delete selected person")
        self.delete_btn.setObjectName("danger")
        self.delete_btn.clicked.connect(self.delete_selected)
        controls.addWidget(self.delete_btn)
        self.lay.addLayout(controls)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Person", "Finger", "Pattern", "Minutiae", "Quality", "Enrolled"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(320)
        self.lay.addWidget(self.table, 1)

    def set_threshold(self, value: float) -> None:
        self.afis.matcher.config.threshold = float(value)
        self.win.status(f"Match threshold set to {value:.2f}")

    def refresh(self) -> None:
        rows = self.afis.database.list_templates()
        s = self.afis.database.get_statistics()
        self.stats.set("Templates", s["total_templates"])
        self.stats.set("People", s["unique_persons"])
        self.stats.set("Avg. minutiae", s["avg_minutiae_count"])
        self.stats.set("Database file", Path(self.afis.database.db_path).name)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [row["person_id"], FINGER_NAMES.get(row["finger_position"], "?"), row["pattern_type"] or "",
                    str(row["minutiae_count"]), f"{(row['quality'] or 0):.2f}", (row["created_at"] or "")[:16].replace("T", " ")]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))

    def delete_selected(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return self.win.warn("Select a row first.")
        person = self.table.item(r, 0).text()
        if QMessageBox.question(self, "Delete person", f"Delete every template for '{person}'?") == QMessageBox.Yes:
            n = self.afis.delete_person(person)
            self.win.status(f"Deleted {n} template(s) for {person}")
            self.win.database_changed()


class AuditPage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // AUDIT", "Audit log",
                                       "Every enrolment, verification, identification and deletion is recorded."))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time (UTC)", "Action", "Person", "Details"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(420)
        self.lay.addWidget(self.table, 1)

    def refresh(self) -> None:
        entries = self.afis.database.audit_entries(500)
        self.table.setRowCount(len(entries))
        colours = {"ENROLL": "#00CC6A", "DELETE": "#FF3B30", "DELETE_PERSON": "#FF3B30"}
        for r, e in enumerate(entries):
            for c, v in enumerate([e["timestamp"], e["action"], e["person_id"] or "—", e["details"] or ""]):
                item = QTableWidgetItem(str(v))
                if c == 1 and e["action"] in colours:
                    item.setForeground(QColor(colours[e["action"]]))
                self.table.setItem(r, c, item)


# ====================================================================== window
class MainWindow(QMainWindow):
    PAGES = [("Analyse", AnalysePage), ("Enroll", EnrollPage), ("Verify", VerifyPage),
             ("Identify", IdentifyPage), ("Database", DatabasePage), ("Audit log", AuditPage)]

    def __init__(self, db_path: Optional[str] = None):
        super().__init__()
        self.afis = AFISSystem(db_path or default_db_path())
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(1)          # one pipeline run at a time
        self._tasks = set()                     # keep tasks alive until their signals are delivered
        self.setWindowTitle("AFIS — Automated Fingerprint Identification System")
        self.setWindowIcon(app_icon())
        self.resize(1280, 860)

        root = QWidget()
        root.setObjectName("root")
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        sv = QVBoxLayout(sidebar)
        sv.setContentsMargins(0, 0, 0, 12)
        brand = QLabel("AFIS")
        brand.setObjectName("brand")
        sub = QLabel("FINGERPRINT ID")
        sub.setObjectName("brandSub")
        sv.addWidget(brand)
        sv.addWidget(sub)

        self.stack = QStackedWidget()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.pages = []
        for k, (name, cls) in enumerate(self.PAGES):
            page = cls(self)
            self.pages.append(page)
            self.stack.addWidget(page)
            btn = QPushButton(name)
            btn.setObjectName("nav")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, i=k: self.show_page(i))
            self.group.addButton(btn, k)
            sv.addWidget(btn)
        sv.addStretch(1)
        footer = QLabel(f"v{__version__} · encrypted store")
        footer.setObjectName("muted")
        footer.setStyleSheet("padding: 0 16px; font-size: 11px;")
        sv.addWidget(footer)

        h.addWidget(sidebar)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

        self.show_page(0)
        self.database_changed()

    # navigation / status
    def show_page(self, index: int) -> None:
        self.group.button(index).setChecked(True)
        self.stack.setCurrentIndex(index)
        self.pages[index].refresh()

    def status(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)

    def warn(self, text: str) -> None:
        QMessageBox.warning(self, "AFIS", text)

    def database_changed(self) -> None:
        for p in self.pages:
            if isinstance(p, (VerifyPage, DatabasePage, AuditPage)):
                p.refresh()
        s = self.afis.database.get_statistics()
        self.status(f"{s['total_templates']} template(s) · {s['unique_persons']} people · {self.afis.database.db_path}")

    # background work
    def run_task(self, fn: Callable, on_done: Callable, *args, button: Optional[QPushButton] = None,
                 busy_text: str = "Working…", on_error: Optional[Callable[[str], None]] = None, **kwargs) -> None:
        if button:
            button.setEnabled(False)
        self.statusBar().showMessage(busy_text)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        task = Task(fn, *args, **kwargs)
        task.setAutoDelete(False)
        self._tasks.add(task)

        def finish():
            self._tasks.discard(task)
            QApplication.restoreOverrideCursor()
            if button:
                button.setEnabled(True)

        def ok(result):
            finish()
            self.statusBar().showMessage("Done", 4000)
            on_done(result)
            for p in self.pages:
                if isinstance(p, AuditPage):
                    p.refresh()

        def fail(message):
            finish()
            self.statusBar().showMessage(f"Error: {message}", 10000)
            (on_error or self.warn)(message)

        task.signals.done.connect(ok)
        task.signals.failed.connect(fail)
        self.pool.start(task)

    def closeEvent(self, event):  # noqa: N802
        self.pool.waitForDone(3000)
        self.afis.close()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AFIS")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
