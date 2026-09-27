"""Compatibility entry point: python3 extract.py TRACE [--gbrain]."""

import sys
from memory.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["extract"] + sys.argv[1:]))
