#!/usr/bin/env bash
# Install AFIS from this source folder for the current user (Linux / macOS).
#
#   git clone https://github.com/thespaceio/AFIS.git && cd AFIS && ./install.sh
#
# Creates a private virtual environment (no system packages are touched),
# installs AFIS into it, and adds `afis-gui` / `afis` launchers plus an
# application-menu entry on Linux. Remove with ./uninstall.sh.
# Prefer not to build from source? Download the ready-made bundle from
# https://github.com/thespaceio/AFIS/releases instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${AFIS_VENV:-${XDG_DATA_HOME:-$HOME/.local/share}/afis-app}"
BIN="$HOME/.local/bin"
PY="${PYTHON:-}"

if [[ -z "$PY" ]]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
      PY="$c"; break
    fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "error: Python 3.9 or newer is required."
  echo "  Debian / Ubuntu / Kali:  sudo apt install python3 python3-venv"
  echo "  macOS:                   brew install python   (or https://www.python.org/downloads/)"
  exit 1
fi
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
  echo "error: $PY has no 'venv' module. On Debian / Ubuntu / Kali run:  sudo apt install python3-venv"
  exit 1
fi

echo "Using $("$PY" --version) at $(command -v "$PY")"
echo "Creating virtual environment in $VENV"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
echo "Installing AFIS and its dependencies (this downloads ~200 MB the first time)..."
"$VENV/bin/python" -m pip install --quiet "$HERE"

mkdir -p "$BIN"
ln -sf "$VENV/bin/afis" "$BIN/afis"
ln -sf "$VENV/bin/afis-gui" "$BIN/afis-gui"

if [[ "$(uname -s)" == "Linux" ]]; then
  APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
  mkdir -p "$APPS" "$ICONS"
  cp "$HERE/afis/gui/icon.png" "$ICONS/afis.png"
  cat > "$APPS/afis.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=AFIS
GenericName=Fingerprint Identification
Comment=Analyse, enrol, verify and identify fingerprints (educational)
Exec=$VENV/bin/afis-gui
Icon=afis
Terminal=false
Categories=Education;Science;Security;
Keywords=fingerprint;biometrics;minutiae;
EOF
  command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" >/dev/null 2>&1 || true

  LDCONFIG="$(command -v ldconfig || echo /sbin/ldconfig)"
  missing=()
  if [[ -x "$LDCONFIG" ]]; then          # skip the check if ldconfig isn't available
    for lib in libEGL.so.1 libxkbcommon-x11.so.0 libxcb-cursor.so.0 libfontconfig.so.1; do
      "$LDCONFIG" -p 2>/dev/null | grep -q "$lib" || missing+=("$lib")
    done
  fi
  if ((${#missing[@]})); then
    echo
    echo "note: Qt needs these system libraries, which were not found: ${missing[*]}"
    echo "      sudo apt install libegl1 libxkbcommon-x11-0 libxcb-cursor0 libfontconfig1"
  fi
fi

echo
"$VENV/bin/afis" --version
echo "Installed. Start the app from your application menu or run:  afis-gui"
echo "Command line:  afis --help"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "(add $BIN to your PATH, e.g.  echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc)";;
esac
