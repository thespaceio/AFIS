#!/usr/bin/env bash
# Install the prebuilt AFIS bundle for the current user (no root needed).
#
#   tar xzf AFIS-*-linux-x86_64.tar.gz && cd AFIS && ./install.sh
#
# Installs to ~/.local/opt/afis, adds `afis-gui` and `afis` to ~/.local/bin and
# an "AFIS" entry to the application menu. Your fingerprint database is NOT
# stored here; it lives in ~/.local/share/afis (see `afis info`).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${AFIS_PREFIX:-$HOME/.local/opt/afis}"
BIN="$HOME/.local/bin"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"

if [[ ! -x "$HERE/AFIS" ]]; then
  echo "error: run this script from the extracted AFIS folder (AFIS executable not found)" >&2
  exit 1
fi

check_libs() {
  # Qt needs a few X11/Wayland libraries that minimal installs (e.g. Kali light) may lack.
  local missing=()
  local LDCONFIG
  LDCONFIG="$(command -v ldconfig || echo /sbin/ldconfig)"
  [[ -x "$LDCONFIG" ]] || return 0
  for lib in libEGL.so.1 libxkbcommon.so.0 libxkbcommon-x11.so.0 libxcb-cursor.so.0 libfontconfig.so.1 libdbus-1.so.3; do
    "$LDCONFIG" -p 2>/dev/null | grep -q "$lib" || missing+=("$lib")
  done
  if ((${#missing[@]})); then
    echo
    echo "note: these system libraries were not found: ${missing[*]}"
    echo "      On Debian / Ubuntu / Kali install them with:"
    echo "        sudo apt install libegl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libfontconfig1 libdbus-1-3"
    echo "      On Fedora: sudo dnf install mesa-libEGL libxkbcommon-x11 xcb-util-cursor fontconfig dbus-libs"
    echo
  fi
}

echo "Installing AFIS to $PREFIX"
mkdir -p "$PREFIX" "$BIN" "$APPS" "$ICONS"
if [[ "$HERE" != "$PREFIX" ]]; then
  rm -rf "$PREFIX.new"
  cp -a "$HERE" "$PREFIX.new"
  rm -rf "$PREFIX"
  mv "$PREFIX.new" "$PREFIX"
fi
ln -sf "$PREFIX/AFIS" "$BIN/afis-gui"
ln -sf "$PREFIX/afis" "$BIN/afis"
cp "$PREFIX/_internal/afis/gui/icon.png" "$ICONS/afis.png"
cat > "$APPS/afis.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=AFIS
GenericName=Fingerprint Identification
Comment=Analyse, enrol, verify and identify fingerprints (educational)
Exec=$PREFIX/AFIS
Icon=afis
Terminal=false
Categories=Education;Science;Security;
Keywords=fingerprint;biometrics;minutiae;
EOF
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" >/dev/null 2>&1 || true

check_libs
echo "Done. Start AFIS from your application menu, or run:  afis-gui"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "(add $BIN to your PATH to use the afis / afis-gui commands from a terminal)";;
esac
