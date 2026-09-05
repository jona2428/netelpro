"""Pytest bootstrap configuration for Straylight.

Inserts the straylight package parent directory (project root) into sys.path
so pytest works seamlessly from the project root or subdirectories with zero install.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Parent of the 'straylight' package directory is the project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
