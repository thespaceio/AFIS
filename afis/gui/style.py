"""Qt style sheet: Th3 SpaceIO palette (deep navy, terminal green, threat red)."""

NAVY = "#0A1628"
PANEL = "#10203A"
PANEL_2 = "#182B4A"
BLACK = "#05080D"
GREEN = "#00CC6A"
RED = "#FF3B30"
AMBER = "#F5A524"
GRAY = "#A7B0BC"
LINE = "#26354F"
WHITE = "#FFFFFF"

STYLE = f"""
* {{
    font-family: "Space Grotesk", "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    color: {WHITE};
}}
QMainWindow, QWidget#root {{ background: {NAVY}; }}
QWidget#sidebar {{ background: {BLACK}; border-right: 1px solid {LINE}; }}
QLabel#brand {{ font-size: 18px; font-weight: 700; padding: 18px 16px 2px 16px; }}
QLabel#brandSub {{ color: {GREEN}; font-family: "JetBrains Mono", "Consolas", monospace; font-size: 11px;
                  letter-spacing: 2px; padding: 0 16px 16px 16px; }}
QPushButton#nav {{
    text-align: left; padding: 11px 16px; border: none; border-radius: 8px; margin: 2px 10px;
    color: {GRAY}; background: transparent; font-weight: 500;
}}
QPushButton#nav:hover {{ background: {PANEL}; color: {WHITE}; }}
QPushButton#nav:checked {{ background: rgba(0, 204, 106, 0.15); color: {GREEN}; }}

QLabel#eyebrow {{ color: {GREEN}; font-family: "JetBrains Mono", "Consolas", monospace; font-size: 11px; letter-spacing: 2px; }}
QLabel#title {{ font-size: 26px; font-weight: 700; }}
QLabel#subtitle {{ color: {GRAY}; }}
QLabel#muted {{ color: {GRAY}; }}
QLabel#statValue {{ font-size: 22px; font-weight: 700; }}
QLabel#statLabel {{ color: {GRAY}; font-size: 12px; }}

QFrame#card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#drop {{ background: {BLACK}; border: 2px dashed {LINE}; border-radius: 12px; }}
QFrame#drop[hover="true"] {{ border-color: {GREEN}; }}

QLineEdit, QComboBox, QDoubleSpinBox {{
    background: {BLACK}; border: 1px solid {LINE}; border-radius: 8px; padding: 8px 10px; min-height: 20px;
}}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border-color: {GREEN}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {LINE}; selection-background-color: {PANEL_2}; }}

QPushButton {{
    background: {PANEL_2}; border: 1px solid {LINE}; border-radius: 8px; padding: 9px 16px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {GRAY}; }}
QPushButton:disabled {{ color: #5b6779; border-color: {PANEL_2}; }}
QPushButton#primary {{ background: {GREEN}; color: {BLACK}; border: none; }}
QPushButton#primary:hover {{ background: #12e07c; }}
QPushButton#primary:disabled {{ background: #1d4a35; color: #5b8f74; }}
QPushButton#danger {{ background: transparent; color: {RED}; border: 1px solid rgba(255, 59, 48, 0.5); }}
QPushButton#danger:hover {{ background: rgba(255, 59, 48, 0.12); }}

QLabel#verdictOk {{ background: rgba(0, 204, 106, 0.14); border: 1px solid {GREEN}; border-radius: 10px;
                    color: {GREEN}; font-size: 20px; font-weight: 700; padding: 14px; }}
QLabel#verdictBad {{ background: rgba(255, 59, 48, 0.14); border: 1px solid {RED}; border-radius: 10px;
                     color: #FF6A61; font-size: 20px; font-weight: 700; padding: 14px; }}
QLabel#verdictInfo {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; color: {GRAY};
                      font-size: 15px; padding: 14px; }}

QTableWidget {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; gridline-color: {LINE};
               alternate-background-color: #0d1b32; selection-background-color: {PANEL_2}; }}
QHeaderView::section {{ background: {BLACK}; color: {GRAY}; border: none; border-bottom: 1px solid {LINE};
                        padding: 8px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
QTableWidget::item {{ padding: 6px; }}

QProgressBar {{ background: {BLACK}; border: 1px solid {LINE}; border-radius: 6px; height: 12px; text-align: center; }}
QProgressBar::chunk {{ background: {GREEN}; border-radius: 5px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {BLACK}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px; min-height: 30px; }}
QScrollBar:horizontal {{ background: {BLACK}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QStatusBar {{ background: {BLACK}; color: {GRAY}; border-top: 1px solid {LINE}; }}
QDialog, QMessageBox, QInputDialog, QFileDialog {{ background: {PANEL}; }}
QMessageBox QLabel {{ color: {WHITE}; }}
QMenuBar {{ background: {BLACK}; border-bottom: 1px solid {LINE}; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
QMenuBar::item:selected {{ background: {PANEL_2}; }}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {PANEL_2}; }}
QCheckBox {{ spacing: 10px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border: 1px solid {LINE}; border-radius: 4px; background: {BLACK}; }}
QCheckBox::indicator:checked {{ background: {GREEN}; border-color: {GREEN}; }}
QTreeView, QListView {{ background: {BLACK}; border: 1px solid {LINE}; }}
QToolTip {{ background: {PANEL}; border: 1px solid {LINE}; color: {WHITE}; padding: 6px; }}
"""
