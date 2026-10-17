"""PyInstaller entry point for the desktop app."""
import sys

from afis.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
