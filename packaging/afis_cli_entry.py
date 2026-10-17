"""PyInstaller entry point for the command-line tool."""
import sys

from afis.cli import main

if __name__ == "__main__":
    sys.exit(main())
