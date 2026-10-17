#!/usr/bin/env bash
# Remove AFIS for the current user.
#
#   ./uninstall.sh            remove the program, launchers and menu entry (keeps your data)
#   ./uninstall.sh --purge    also delete the fingerprint database, key and settings
set -euo pipefail

PREFIX="${AFIS_PREFIX:-$HOME/.local/opt/afis}"
VENV="${AFIS_VENV:-${XDG_DATA_HOME:-$HOME/.local/share}/afis-app}"
BIN="$HOME/.local/bin"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
DATA="${AFIS_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/afis}"

for f in "$BIN/afis" "$BIN/afis-gui" "$APPS/afis.desktop" "$ICONS/afis.png"; do
  [[ -e "$f" || -L "$f" ]] && rm -f "$f" && echo "removed $f"
done
for d in "$PREFIX" "$VENV"; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "removed $d"
done
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" >/dev/null 2>&1 || true

if [[ "${1:-}" == "--purge" ]]; then
  for d in "$DATA" "$HOME/.afis"; do
    [[ -d "$d" ]] || continue
    echo
    echo "This permanently deletes the AFIS database, its encryption key and settings in:"
    echo "  $d"
    read -r -p "Type DELETE to confirm: " answer
    if [[ "$answer" == "DELETE" ]]; then
      rm -rf "$d" && echo "removed $d"
    else
      echo "kept $d"
    fi
  done
else
  [[ -d "$DATA" ]] && echo "Your data was kept in $DATA (run with --purge to delete it)."
fi
echo "AFIS has been uninstalled."
