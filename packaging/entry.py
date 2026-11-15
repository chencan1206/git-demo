"""Frozen-app entry point.

Kept separate from ``plant_hsi.app`` so PyInstaller has a plain script to
analyse; it only forwards to the real entry point.
"""
from __future__ import annotations

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from plant_hsi.app import main

    sys.exit(main())
