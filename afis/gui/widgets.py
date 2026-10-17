"""Reusable Qt widgets for the AFIS GUI."""

from __future__ import annotations

import os
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

IMAGE_FILTER = "Fingerprint images (*.tif *.tiff *.bmp *.png *.jpg *.jpeg *.pgm)"
IMAGE_EXTS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".pgm")


def to_pixmap(image: np.ndarray) -> QPixmap:
    """numpy (gray or BGR) -> QPixmap."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        h, w = image.shape
        q = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        q = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class ImageView(QLabel):
    """Shows an image scaled to fit, keeping its aspect ratio."""

    def __init__(self, placeholder: str = "", min_height: int = 220):
        super().__init__(placeholder)
        self._pix: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("muted")
        self.setWordWrap(True)

    def set_image(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            self._pix = None
            self.clear()
            return
        self._pix = to_pixmap(image)
        self._rescale()

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class DropArea(QFrame):
    """Drag-and-drop or browse for a fingerprint image."""

    fileSelected = Signal(str)
    fileRejected = Signal(str)
    start_dir: str = ""          # shared by every DropArea: last folder used (or the demo prints)

    def __init__(self, prompt: str = "Drop a fingerprint image here"):
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.path: Optional[str] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        self.view = ImageView(f"{prompt}\nor click Browse", 240)
        self.name = QLabel("")
        self.name.setObjectName("muted")
        self.name.setAlignment(Qt.AlignCenter)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.name)
        lay.addWidget(browse, 0, Qt.AlignCenter)

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a fingerprint image", DropArea.start_dir,
                                              IMAGE_FILTER + ";;All files (*)")
        if path:
            self.set_path(path)

    def set_path(self, path: str) -> bool:
        """Show a preview of `path`; returns False (and emits fileRejected) if it can't be read."""
        name = os.path.basename(path)
        img = None
        if not os.path.isfile(path):
            problem = f"File not found: {name}"
        elif not path.lower().endswith(IMAGE_EXTS):
            problem = f"'{name}' is not a supported image type ({', '.join(IMAGE_EXTS)})"
        else:
            try:
                img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
            except (cv2.error, OSError, ValueError):
                img = None
            problem = None if img is not None else f"'{name}' could not be read: the file is damaged or not an image"
        if problem:
            self.clear()
            self.fileRejected.emit(problem)
            return False
        self.path = path
        DropArea.start_dir = os.path.dirname(path)
        self.view.set_image(img)
        self.name.setText(name)
        self.name.setToolTip(path)
        self.fileSelected.emit(path)
        return True

    def clear(self) -> None:
        self.path = None
        self.view.set_image(None)
        self.view.setText("Drop a fingerprint image here\nor click Browse")
        self.name.setText("")

    # drag and drop
    def dragEnterEvent(self, e):  # noqa: N802
        urls = e.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.setProperty("hover", True)
            self.style().polish(self)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):  # noqa: N802
        self.setProperty("hover", False)
        self.style().polish(self)

    def dropEvent(self, e):  # noqa: N802
        self.setProperty("hover", False)
        self.style().polish(self)
        self.set_path(e.mimeData().urls()[0].toLocalFile())


class Card(QFrame):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 14, 16, 16)
        self.body.setSpacing(10)
        if title:
            t = QLabel(title)
            t.setStyleSheet("font-weight: 700; font-size: 15px;")
            self.body.addWidget(t)


class StatGrid(QWidget):
    """Label / value tiles."""

    def __init__(self, labels: list, cols: int = 3):
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.values = {}
        for k, label in enumerate(labels):
            tile = QFrame()
            tile.setObjectName("card")
            v = QVBoxLayout(tile)
            v.setContentsMargins(12, 10, 12, 10)
            val = QLabel("–")
            val.setObjectName("statValue")
            lab = QLabel(label)
            lab.setObjectName("statLabel")
            v.addWidget(val)
            v.addWidget(lab)
            grid.addWidget(tile, k // cols, k % cols)
            self.values[label] = val

    def set(self, label: str, value) -> None:
        self.values[label].setText(str(value))

    def reset(self) -> None:
        for v in self.values.values():
            v.setText("–")


def page_header(eyebrow: str, title: str, subtitle: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 6)
    lay.setSpacing(4)
    for text, name in ((eyebrow, "eyebrow"), (title, "title"), (subtitle, "subtitle")):
        lab = QLabel(text)
        lab.setObjectName(name)
        lab.setWordWrap(True)
        lay.addWidget(lab)
    return w


def hbox(*widgets, stretch_last: bool = False) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for i, x in enumerate(widgets):
        lay.addWidget(x, 1 if (stretch_last and i == len(widgets) - 1) else 0)
    return w


# ---------------------------------------------------------------------- background work
class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    """Run `fn` on the thread pool; emit done(result) or failed(message)."""

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.signals.failed.emit(str(e) or type(e).__name__)
        else:
            self.signals.done.emit(result)
