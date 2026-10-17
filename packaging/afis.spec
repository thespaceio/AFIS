# PyInstaller spec: one folder with two programs sharing the same libraries.
#   AFIS  (AFIS.exe / AFIS.app)  desktop app, no console window
#   afis  (afis-cli.exe)         command-line tool
# Build:  pyinstaller packaging/afis.spec --noconfirm
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent
ICON = str(ROOT / "packaging" / "afis.png")
WIN = sys.platform == "win32"
MAC = sys.platform == "darwin"

datas = [
    (str(ROOT / "afis" / "demo_prints"), "afis/demo_prints"),
    (str(ROOT / "afis" / "gui" / "icon.png"), "afis/gui"),
    (str(ROOT / "LICENSE"), "."),
]
datas += collect_data_files("skimage", includes=["**/*.pyi"])   # lazy-loader stubs

excludes = ["matplotlib", "tkinter", "IPython", "pytest", "pandas", "PyQt5", "PyQt6", "PySide2",
            "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore",
            "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization"]

a = Analysis(
    [str(ROOT / "packaging" / "afis_gui_entry.py"), str(ROOT / "packaging" / "afis_cli_entry.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=["afis.gui.app", "afis.cli", "skimage.morphology"],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

# Analysis puts one entry script per program in a.scripts (order as above).
gui_script = [s for s in a.scripts if s[0] == "afis_gui_entry"]
cli_script = [s for s in a.scripts if s[0] == "afis_cli_entry"]
common = [s for s in a.scripts if s[0] not in ("afis_gui_entry", "afis_cli_entry")]

gui = EXE(pyz, common + gui_script, [], exclude_binaries=True, name="AFIS", console=False,
          icon=ICON, upx=False)
cli = EXE(pyz, common + cli_script, [], exclude_binaries=True, name="afis-cli" if WIN else "afis",
          console=True, icon=ICON, upx=False)
coll = COLLECT(gui, cli, a.binaries, a.datas, name="AFIS", upx=False)

if MAC:
    app = BUNDLE(coll, name="AFIS.app", icon=ICON, bundle_identifier="io.thespace.afis",
                 info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": "2.1.0"})
