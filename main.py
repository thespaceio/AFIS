#!/usr/bin/env python3
"""Start the AFIS command-line tool:  python main.py   (or: python -m afis)"""
import sys

from afis.cli import main

if __name__ == "__main__":
    sys.exit(main())
