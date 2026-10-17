"""
AFIS desktop application (PySide6).

    afis-gui            (installed)       or      python -m afis.gui

Data is kept in the per-user AFIS folder (see afis/config.py) unless AFIS_DB
or AFIS_HOME is set.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QThreadPool, Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import __version__
from ..config import Settings, default_db_path
from ..database import AFISKeyError, key_path_for
from ..demo import demo_dir
from ..system import FINGER_NAMES, AFISSystem
from ..visualization import draw_match, draw_minutiae
from .style import STYLE
from .widgets import Card, DropArea, ImageView, StatGrid, Task, page_header

PROBABILISTIC_NOTE = ("Scores are probabilistic evidence, not proof of identity. AFIS is an educational system "
                      "and has not been validated for forensic or law-enforcement decisions.")

RESPONSIBLE_USE = (
    "<p><b>Before you start</b></p>"
    "<p>AFIS is an educational fingerprint-recognition system. Its match scores are "
    "<b>probabilistic</b>: genuine prints can be rejected and different people can occasionally "
    "match. It has <b>not</b> been validated for forensic, legal, employment or law-enforcement "
    "decisions and must not be used as the sole basis for any decision about a person.</p>"
    "<p>Fingerprints are sensitive biometric data. Only enrol people who have given informed "
    "consent, keep the database and its key private, and delete records you no longer need. "
    "Data protection laws (for example GDPR, BIPA or the NDPR) may apply to you.</p>"
    "<p>To explore safely, use <b>Load demo data</b> on the Home page: the demo prints are "
    "synthetic and belong to no one.</p>")


def app_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor("#00CC6A"), 4))
    for r in (8, 15, 22, 29):
        p.drawArc(32 - r, 36 - r, 2 * r, 2 * r, 20 * 16, 140 * 16)
    p.end()
    return QIcon(pix)


def make_table(headers, min_height: int = 240) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setAlternatingRowColors(True)
    t.setMinimumHeight(min_height)
    return t


def note_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("muted")
    lab.setWordWrap(True)
    lab.setStyleSheet("font-size: 12px;")
    return lab


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

    def drop_area(self) -> DropArea:
        drop = DropArea()
        drop.setMinimumWidth(260)
        drop.setMaximumWidth(340)
        drop.setMaximumHeight(430)
        drop.fileRejected.connect(self.win.warn)
        return drop


class HomePage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // WELCOME", "Fingerprint identification, step by step",
                                       "Enhance prints, extract minutiae, and enrol, verify or identify people "
                                       "against an encrypted local database."))
        self.stats = StatGrid(["Templates", "People", "Avg. minutiae"], 3)
        self.lay.addWidget(self.stats)

        start = Card("Quick start")
        steps = QLabel(
            "<ol style='margin-left:-20px; line-height:150%'>"
            "<li><b>Load demo data</b> below: it enrols six synthetic people (demo-01 … demo-06).</li>"
            "<li>Open <b>Identify</b>, click <i>Browse…</i> and pick a demo print such as "
            "<code>demo-03_2.png</code>. AFIS should name demo-03.</li>"
            "<li>Open <b>Verify</b> and check a print against a claimed identity.</li>"
            "<li>Use <b>Analyse</b> to watch every processing stage, then <b>Enroll</b> your own images.</li>"
            "</ol>")
        steps.setWordWrap(True)
        steps.setTextFormat(Qt.RichText)
        start.body.addWidget(steps)
        row = QHBoxLayout()
        self.demo_btn = QPushButton("Load demo data")
        self.demo_btn.setObjectName("primary")
        self.demo_btn.setToolTip("Enrol impression 1 of each synthetic demo finger")
        self.demo_btn.clicked.connect(self.load_demo)
        folder_btn = QPushButton("Open demo prints folder")
        folder_btn.clicked.connect(lambda: self.win.open_folder(demo_dir()))
        row.addWidget(self.demo_btn)
        row.addWidget(folder_btn)
        row.addStretch(1)
        start.body.addLayout(row)
        self.demo_msg = note_label("")
        start.body.addWidget(self.demo_msg)
        self.lay.addWidget(start)

        about = Card("Good to know")
        about.body.addWidget(note_label(
            "• Supported images: PNG, TIFF, BMP, JPEG, PGM. Grayscale scans at about 500 dpi work best.<br>"
            "• Templates and images are encrypted on this computer. Nothing is uploaded anywhere.<br>"
            "• " + PROBABILISTIC_NOTE))
        self.lay.addWidget(about)
        self.lay.addStretch(1)

    def refresh(self) -> None:
        s = self.afis.database.get_statistics()
        self.stats.set("Templates", s["total_templates"])
        self.stats.set("People", s["unique_persons"])
        self.stats.set("Avg. minutiae", s["avg_minutiae_count"])

    def load_demo(self) -> None:
        self.win.run_task(self.afis.enroll_demo, self.demo_done, button=self.demo_btn,
                          busy_text="Enrolling demo prints…")

    def demo_done(self, people) -> None:
        DropArea.start_dir = str(demo_dir())
        self.demo_msg.setText(f"✓ Enrolled {len(people)} demo people ({people[0]} … {people[-1]}). "
                              "Impressions 2–4 of each are not enrolled: use them on Verify and Identify.")
        self.win.database_changed()


class AnalysePage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // PIPELINE", "Analyse a fingerprint",
                                       "See every processing stage, the minutiae and the pattern."))
        top = QHBoxLayout()
        self.drop = self.drop_area()
        self.drop.fileSelected.connect(self.run)
        top.addWidget(self.drop, 0, Qt.AlignTop)

        self.stats = StatGrid(["Quality", "Minutiae", "Endings", "Bifurcations", "Pattern", "Processing time"], 3)
        self.overlay = ImageView("Choose an image: the minutiae overlay appears here", 300)
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
        self.win.run_task(self.afis.process, self.show_result, path, busy_text="Processing…",
                          on_error=self.failed)

    def failed(self, message: str) -> None:
        self.stats.reset()
        self.overlay.set_image(None)
        self.overlay.setText("Could not analyse this image")
        self.win.warn(message)

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
        for k, (name, img) in enumerate(p.result.stages().items()):
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
        self.drop = self.drop_area()
        row.addWidget(self.drop, 0, Qt.AlignTop)

        form = Card("Details")
        self.person = QLineEdit()
        self.person.setPlaceholderText("Person ID, e.g. jane.doe")
        self.person.setMaxLength(64)
        self.person.returnPressed.connect(self.enroll)
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
        form.body.addWidget(note_label("Only enrol people who have agreed to it. Letters, digits, spaces and "
                                       ". _ - @ ' are allowed in IDs."))
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
                          button=self.button, busy_text="Enrolling…",
                          on_error=lambda e: self.show_msg(f"✗ {e}", "verdictBad"))

    def done(self, template) -> None:
        g = template.global_features
        self.show_msg(f"✓ Enrolled {template.person_id} · {FINGER_NAMES[template.finger_position]} · "
                      f"{len(template.minutiae)} minutiae · {g.get('pattern_type')} · quality {g.get('quality'):.2f}",
                      "verdictOk")
        p = self.afis.last
        if p is not None:
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
        self.score_text.setWordWrap(True)
        self.picture = ImageView("", 260)
        for w in (self.verdict, self.bar, self.score_text):
            self.body.addWidget(w)
        self.body.addWidget(self.picture, 1)
        self.body.addWidget(note_label(PROBABILISTIC_NOTE))

    def display(self, ok: bool, headline: str, score: float, threshold: float, detail: str) -> None:
        self.verdict.setObjectName("verdictOk" if ok else "verdictBad")
        self.verdict.setText(headline)
        self.verdict.style().polish(self.verdict)
        self.bar.setValue(int(min(1.0, score) * 1000))
        self.score_text.setText(f"Score {score:.3f}  ·  threshold {threshold:.2f}" + (f"  ·  {detail}" if detail else ""))

    def info(self, text: str) -> None:
        self.verdict.setObjectName("verdictInfo")
        self.verdict.setText(text)
        self.verdict.style().polish(self.verdict)
        self.bar.setValue(0)
        self.score_text.setText("")
        self.picture.set_image(None)


def match_picture(afis: AFISSystem, details: dict, is_match: bool = True) -> Optional[np.ndarray]:
    fid = details.get("fingerprint_id")
    if not fid or afis.last is None:
        return None
    ref_t = afis.database.load_template(fid)
    ref_img = afis.reference_image(fid)
    if ref_t is None or ref_img is None:
        return draw_minutiae(afis.last.image, afis.last.minutiae)
    return draw_match(afis.last.image, afis.last.minutiae, ref_img, ref_t.minutiae, details.get("pairs", []),
                      is_match=is_match)


class VerifyPage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // VERIFY · 1:1", "Verify an identity",
                                       "Does this fingerprint match the person it is claimed to be?"))
        row = QHBoxLayout()
        left = QVBoxLayout()
        self.drop = self.drop_area()
        self.person = QComboBox()
        self.person.setEditable(True)
        self.person.lineEdit().setPlaceholderText("Claimed person ID")
        self.button = QPushButton("Verify")
        self.button.setObjectName("primary")
        self.button.clicked.connect(self.verify)
        left.addWidget(self.drop)
        left.addWidget(QLabel("Claimed identity"))
        left.addWidget(self.person)
        left.addWidget(self.button)
        left.addStretch(1)
        row.addLayout(left)
        self.result = MatchView("Result")
        row.addWidget(self.result, 1)
        self.lay.addLayout(row)

    def refresh(self) -> None:
        current = self.person.currentText()
        self.person.blockSignals(True)
        self.person.clear()
        self.person.addItems(self.afis.database.persons())
        self.person.setEditText(current)
        self.person.blockSignals(False)

    def verify(self) -> None:
        person = self.person.currentText().strip()
        if not self.drop.path or not person:
            return self.win.warn("Choose an image and a claimed identity first.")
        self._claimed = person
        self.win.run_task(self.afis.verify, self.done, self.drop.path, person, button=self.button,
                          busy_text="Verifying…")

    def done(self, res) -> None:
        ok, score, details = res
        thr = self.afis.threshold
        if "error" in details:
            self.result.info(details["error"] + ". Enroll this person first, or pick a name from the list.")
            return
        person = self._claimed
        head = f"✓ MATCH — consistent with {person}" if ok else f"✗ NO MATCH — does not match {person}"
        self.result.display(ok, head, score, thr, f"{details.get('matched_minutiae', 0)} matched minutiae")
        self.result.picture.set_image(match_picture(self.afis, details, ok))


class IdentifyPage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // IDENTIFY · 1:N", "Identify a fingerprint",
                                       "Search every enrolled person for the best match."))
        row = QHBoxLayout()
        left = QVBoxLayout()
        self.drop = self.drop_area()
        self.button = QPushButton("Search database")
        self.button.setObjectName("primary")
        self.button.clicked.connect(self.identify)
        left.addWidget(self.drop)
        left.addWidget(self.button)
        left.addStretch(1)
        row.addLayout(left)

        right = QVBoxLayout()
        self.result = MatchView("Best candidate")
        self.table = make_table(["Rank", "Person", "Score", "Matched minutiae"], 150)
        self.table.setMaximumHeight(190)
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
        thr = self.afis.threshold
        self.table.setRowCount(0)
        if not ranked:
            self.result.info("The database is empty. Enroll someone first, or use Load demo data on the Home page.")
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
        head = f"✓ CANDIDATE: {pid}" if d["is_match"] else "✗ No candidate above the threshold"
        self.result.display(d["is_match"], head, score, thr, f"{d.get('matched_minutiae', 0)} matched minutiae")
        self.result.picture.set_image(match_picture(self.afis, d, d["is_match"]))


class DatabasePage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // DATABASE", "Enrolled templates",
                                       "Templates and images are encrypted at rest (Fernet). "
                                       "Only metadata is listed here."))
        self.stats = StatGrid(["Templates", "People", "Avg. minutiae"], 3)
        self.lay.addWidget(self.stats)

        controls = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by person…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self.apply_filter)
        controls.addWidget(self.filter, 1)
        self.delete_one = QPushButton("Delete selected template")
        self.delete_one.setObjectName("danger")
        self.delete_one.clicked.connect(self.delete_selected_template)
        self.delete_btn = QPushButton("Delete selected person")
        self.delete_btn.setObjectName("danger")
        self.delete_btn.clicked.connect(self.delete_selected)
        controls.addWidget(self.delete_one)
        controls.addWidget(self.delete_btn)
        self.lay.addLayout(controls)

        self.empty = QLabel("No templates yet. Enroll a fingerprint, or use Load demo data on the Home page.")
        self.empty.setObjectName("verdictInfo")
        self.empty.setWordWrap(True)
        self.lay.addWidget(self.empty)
        self.table = make_table(["Person", "Finger", "Pattern", "Minutiae", "Quality", "Enrolled (UTC)"], 320)
        self.lay.addWidget(self.table, 1)
        self._ids = []

    def refresh(self) -> None:
        rows = self.afis.database.list_templates()
        s = self.afis.database.get_statistics()
        self.stats.set("Templates", s["total_templates"])
        self.stats.set("People", s["unique_persons"])
        self.stats.set("Avg. minutiae", s["avg_minutiae_count"])
        self.empty.setVisible(not rows)
        self.table.setVisible(bool(rows))
        self.table.setRowCount(len(rows))
        self._ids = [r["fingerprint_id"] for r in rows]
        for r, row in enumerate(rows):
            vals = [row["person_id"], FINGER_NAMES.get(row["finger_position"], "?"), row["pattern_type"] or "",
                    str(row["minutiae_count"]), f"{(row['quality'] or 0):.2f}",
                    (row["created_at"] or "")[:16].replace("T", " ")]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.apply_filter(self.filter.text())

    def apply_filter(self, text: str) -> None:
        text = text.strip().lower()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            self.table.setRowHidden(r, bool(text) and text not in item.text().lower())

    def _selected(self) -> Optional[int]:
        r = self.table.currentRow()
        if r < 0 or self.table.isRowHidden(r):
            self.win.warn("Select a row in the table first.")
            return None
        return r

    def delete_selected(self) -> None:
        r = self._selected()
        if r is None:
            return
        person = self.table.item(r, 0).text()
        n = sum(1 for k in range(self.table.rowCount()) if self.table.item(k, 0).text() == person)
        if self.win.ask(f"Delete all {n} template(s) for '{person}'?\n\nThis cannot be undone.", "Delete person"):
            n = self.afis.delete_person(person)
            self.win.status(f"Deleted {n} template(s) for {person}")
            self.win.database_changed()

    def delete_selected_template(self) -> None:
        r = self._selected()
        if r is None:
            return
        person, finger = self.table.item(r, 0).text(), self.table.item(r, 1).text()
        if self.win.ask(f"Delete this template ({person}, {finger})?\n\nThis cannot be undone.", "Delete template"):
            self.afis.delete_template(self._ids[r])
            self.win.status(f"Deleted one template for {person}")
            self.win.database_changed()


class AuditPage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // AUDIT", "Audit log",
                                       "Every enrolment, verification, identification and deletion is recorded. "
                                       "Entries are chained with an HMAC so edits can be detected."))
        row = QHBoxLayout()
        verify = QPushButton("Verify log integrity")
        verify.clicked.connect(self.verify_chain)
        export = QPushButton("Export CSV…")
        export.clicked.connect(self.export_csv)
        self.chain = QLabel("")
        self.chain.setObjectName("muted")
        row.addWidget(verify)
        row.addWidget(export)
        row.addWidget(self.chain, 1)
        self.lay.addLayout(row)
        self.table = make_table(["Time (UTC)", "Action", "Person", "Details"], 420)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.lay.addWidget(self.table, 1)

    def refresh(self) -> None:
        entries = self.afis.database.audit_entries(500)
        self.table.setRowCount(len(entries))
        colours = {"ENROLL": "#00CC6A", "DELETE": "#FF3B30", "DELETE_PERSON": "#FF3B30", "DELETE_ALL": "#FF3B30"}
        for r, e in enumerate(entries):
            ts = (e["timestamp"] or "").replace("T", " ").replace("+00:00", "")
            for c, v in enumerate([ts, e["action"], e["person_id"] or "—", e["details"] or ""]):
                item = QTableWidgetItem(str(v))
                if c == 1 and e["action"] in colours:
                    item.setForeground(QColor(colours[e["action"]]))
                self.table.setItem(r, c, item)

    def verify_chain(self) -> None:
        a = self.afis.database.verify_audit()
        if a["ok"]:
            text = f"✓ Chain of {a['checked']} entries is intact"
            if a["legacy"]:
                text += f" ({a['legacy']} older entries predate the chain)"
        elif a.get("truncated"):
            text = "✗ Entries were removed from the end of the log"
        else:
            text = f"✗ Chain broken at entry {a['first_bad_id']}: the log was edited"
        self.chain.setText(text)
        self.chain.setStyleSheet(f"color: {'#00CC6A' if a['ok'] else '#FF6A61'}; font-weight: 600;")

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export audit log", "afis_audit_log.csv", "CSV (*.csv)")
        if not path:
            return
        entries = list(reversed(self.afis.database.audit_entries(1_000_000)))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "timestamp", "action", "person_id", "fingerprint_id", "details"])
            w.writeheader()
            w.writerows(entries)
        self.win.status(f"Exported {len(entries)} entries to {path}")


class SettingsPage(BasePage):
    def __init__(self, window):
        super().__init__(window)
        self.lay.addWidget(page_header("AFIS // SETTINGS", "Settings & data",
                                       "Matching threshold, privacy options and where your data lives."))
        match = Card("Matching")
        row = QHBoxLayout()
        row.addWidget(QLabel("Match threshold"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.90)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(2)
        self.threshold.setValue(self.afis.threshold)
        self.threshold.valueChanged.connect(self.set_threshold)
        reset = QPushButton("Reset to default (0.18)")
        reset.clicked.connect(lambda: self.threshold.setValue(0.18))
        row.addWidget(self.threshold)
        row.addWidget(reset)
        row.addStretch(1)
        match.body.addLayout(row)
        match.body.addWidget(note_label(
            "Higher = fewer false accepts but more false rejects. The default was chosen from the evaluation "
            "described in the README; re-run scripts/evaluate.py on data like yours before relying on it."))
        self.lay.addWidget(match)

        privacy = Card("Privacy")
        self.store_images = QCheckBox("Keep an encrypted copy of enrolment images (used to draw match pictures)")
        self.store_images.setChecked(self.afis.store_images)
        self.store_images.toggled.connect(self.set_store_images)
        privacy.body.addWidget(self.store_images)
        privacy.body.addWidget(note_label("Turn this off to store minutiae templates only (data minimisation). "
                                          "It applies to new enrolments."))
        self.lay.addWidget(privacy)

        data = Card("Data")
        db = self.afis.database.db_path
        self.paths = QLabel(f"Database: <code>{db}</code><br>Key: <code>{self.win.key_location()}</code>")
        self.paths.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.paths.setWordWrap(True)
        data.body.addWidget(self.paths)
        data.body.addWidget(note_label("Back up the database and the key together and keep the key private. "
                                       "Without the key the templates can't be read; with it, anyone can."))
        row = QHBoxLayout()
        open_btn = QPushButton("Open data folder")
        open_btn.clicked.connect(lambda: self.win.open_folder(Path(db).parent))
        check_btn = QPushButton("Check integrity")
        check_btn.clicked.connect(self.check)
        self.wipe = QPushButton("Delete all templates…")
        self.wipe.setObjectName("danger")
        self.wipe.clicked.connect(self.delete_all)
        row.addWidget(open_btn)
        row.addWidget(check_btn)
        row.addStretch(1)
        row.addWidget(self.wipe)
        data.body.addLayout(row)
        self.check_result = note_label("")
        data.body.addWidget(self.check_result)
        self.lay.addWidget(data)
        self.lay.addStretch(1)

    def set_threshold(self, value: float) -> None:
        self.afis.threshold = float(value)
        self.win.settings.threshold = round(float(value), 2)
        self.win.save_settings()
        self.win.status(f"Match threshold set to {value:.2f}")

    def set_store_images(self, on: bool) -> None:
        self.afis.store_images = bool(on)
        self.win.settings.store_images = bool(on)
        self.win.save_settings()

    def check(self) -> None:
        res = self.afis.check_integrity()
        t, a = res["templates"], res["audit"]
        lines = [f"{'✓' if not t['bad'] else '✗'} Templates: {t['checked'] - len(t['bad'])} of {t['checked']} "
                 "decrypted and passed their integrity check"]
        lines += [f"    {fid}: {why}" for fid, why in t["bad"][:5]]
        lines.append("✓ Audit log chain intact" if a["ok"] else "✗ Audit log has been altered")
        self.check_result.setText("\n".join(lines))
        self.check_result.setStyleSheet(
            f"color: {'#00CC6A' if not t['bad'] and a['ok'] else '#FF6A61'}; font-size: 12px;")

    def delete_all(self) -> None:
        n = self.afis.database.get_statistics()["total_templates"]
        if n == 0:
            return self.win.warn("The database is already empty.")
        text, accepted = QInputDialog.getText(
            self, "Delete all templates",
            f"This permanently deletes all {n} template(s) and stored images.\n"
            "The audit log is kept. Type DELETE to confirm:")
        if not accepted:
            return
        if text.strip() != "DELETE":
            return self.win.warn("Nothing was deleted (you need to type DELETE).")
        deleted = self.afis.delete_all("DELETE")
        self.win.status(f"Deleted {deleted} template(s)")
        self.win.database_changed()


# ====================================================================== window
class MainWindow(QMainWindow):
    PAGES = [("Home", HomePage), ("Analyse", AnalysePage), ("Enroll", EnrollPage), ("Verify", VerifyPage),
             ("Identify", IdentifyPage), ("Database", DatabasePage), ("Audit log", AuditPage),
             ("Settings", SettingsPage)]

    def __init__(self, db_path: Optional[str] = None, settings: Optional[Settings] = None):
        super().__init__()
        db_path = db_path or default_db_path()
        self.settings_dir = Path(db_path).parent
        self.settings = settings or Settings.load(self.settings_dir)
        self.afis = AFISSystem(db_path, store_images=self.settings.store_images, threshold=self.settings.threshold)
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)          # one pipeline run at a time
        self._tasks = set()                     # keep tasks alive until their signals are delivered
        if not DropArea.start_dir:
            DropArea.start_dir = str(demo_dir())
        self.setWindowTitle("AFIS — Automated Fingerprint Identification System")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(960, 640)
        self._fit_to_screen()

        root = QWidget()
        root.setObjectName("root")
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
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
            btn.setToolTip(f"{name}  (Ctrl+{k + 1})")
            btn.clicked.connect(lambda _=False, i=k: self.show_page(i))
            self.group.addButton(btn, k)
            sv.addWidget(btn)
            act = QAction(self)
            act.setShortcut(QKeySequence(f"Ctrl+{k + 1}"))
            act.triggered.connect(lambda _=False, i=k: self.show_page(i))
            self.addAction(act)
        sv.addStretch(1)
        footer = QLabel(f"v{__version__} · encrypted · offline")
        footer.setObjectName("muted")
        footer.setStyleSheet("padding: 0 16px; font-size: 11px;")
        sv.addWidget(footer)

        h.addWidget(sidebar)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self._build_menu()

        self.show_page(0)
        self.database_changed()

    def _fit_to_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 860)
            return
        avail = screen.availableGeometry()
        self.resize(min(1280, int(avail.width() * 0.92)), min(860, int(avail.height() * 0.92)))

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        act = file_menu.addAction("Open data folder")
        act.triggered.connect(lambda: self.open_folder(Path(self.afis.database.db_path).parent))
        act = file_menu.addAction("Open demo prints folder")
        act.triggered.connect(lambda: self.open_folder(demo_dir()))
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        help_menu = bar.addMenu("&Help")
        act = help_menu.addAction("Quick start")
        act.triggered.connect(lambda: self.show_page(0))
        act = help_menu.addAction("Responsible use")
        act.triggered.connect(lambda: QMessageBox.information(self, "Responsible use", RESPONSIBLE_USE))
        act = help_menu.addAction("About AFIS")
        act.triggered.connect(self.about)

    def about(self) -> None:
        QMessageBox.about(self, "About AFIS",
                          f"<h3>AFIS {__version__}</h3>"
                          "<p>Automated Fingerprint Identification System — an educational, offline "
                          "fingerprint enhancement and matching tool.</p>"
                          "<p>By Faith Alabi (SPACEIO). MIT licence.<br>"
                          "<a href='https://github.com/thespaceio/AFIS'>github.com/thespaceio/AFIS</a></p>"
                          f"<p>{PROBABILISTIC_NOTE}</p>")

    # navigation / status
    def show_page(self, index: int) -> None:
        self.group.button(index).setChecked(True)
        self.stack.setCurrentIndex(index)
        self.pages[index].refresh()

    def page(self, cls):
        return next(p for p in self.pages if isinstance(p, cls))

    def status(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)

    def warn(self, text: str) -> None:
        QMessageBox.warning(self, "AFIS", text)

    def ask(self, text: str, title: str = "AFIS") -> bool:
        box = QMessageBox(QMessageBox.Warning, title, text, QMessageBox.Yes | QMessageBox.Cancel, self)
        box.setDefaultButton(QMessageBox.Cancel)
        return box.exec() == QMessageBox.Yes

    def open_folder(self, folder: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def key_location(self) -> str:
        import os
        return "AFIS_KEY environment variable" if os.environ.get("AFIS_KEY") else str(
            key_path_for(self.afis.database.db_path))

    def save_settings(self) -> None:
        try:
            self.settings.save(self.settings_dir)
        except OSError as e:
            self.status(f"Could not save settings: {e}")

    def database_changed(self) -> None:
        for p in self.pages:
            if isinstance(p, (HomePage, VerifyPage, DatabasePage, AuditPage)):
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
            self.page(AuditPage).refresh()

        def fail(message):
            finish()
            self.statusBar().showMessage(f"Error: {message}", 10000)
            (on_error or self.warn)(message)
            self.page(AuditPage).refresh()

        task.signals.done.connect(ok)
        task.signals.failed.connect(fail)
        self.pool.start(task)

    def closeEvent(self, event):  # noqa: N802
        self.pool.waitForDone(5000)
        self.afis.close()
        super().closeEvent(event)


def show_notice(parent: Optional[QWidget], settings: Settings, settings_dir: Path) -> bool:
    """First-run responsible-use notice. Returns False if the user declines."""
    if settings.accepted_notice:
        return True
    box = QMessageBox(QMessageBox.Information, "AFIS — responsible use", RESPONSIBLE_USE,
                      QMessageBox.Ok | QMessageBox.Cancel, parent)
    box.button(QMessageBox.Ok).setText("I understand")
    box.button(QMessageBox.Cancel).setText("Quit")
    if box.exec() != QMessageBox.Ok:
        return False
    settings.accepted_notice = True
    try:
        settings.save(settings_dir)
    except OSError:
        pass
    return True


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--version" in argv:
        print(f"AFIS {__version__}")
        return 0
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("AFIS")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI" if sys.platform == "win32" else app.font().family(), 10))

    db_path = default_db_path()
    settings = Settings.load(Path(db_path).parent)
    if "--accept-notice" not in argv and not show_notice(None, settings, Path(db_path).parent):
        return 0
    try:
        win = MainWindow(db_path, settings)
    except AFISKeyError as e:
        QMessageBox.critical(None, "AFIS — cannot open the database",
                             f"{e}\n\nDatabase: {db_path}\n\nIf you moved the database, copy its .key file "
                             "next to it. To start fresh instead, set AFIS_HOME to an empty folder.")
        return 2
    except Exception as e:  # noqa: BLE001 - last-resort message for non-technical users
        QMessageBox.critical(None, "AFIS — startup error", f"AFIS could not start:\n\n{e}\n\nDatabase: {db_path}")
        return 2
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
