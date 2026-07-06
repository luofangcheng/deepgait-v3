"""DeepGait v3 — L4 algorithm core.

Mirrors v1's L4 layer but routes through the v4.x 4-stage pipeline:

- ``pawprint`` — Stage 1 (fTIR → PawPrint)
- ``calibration`` / ``triangulation`` / ``fusion`` — Stage 2 (ChArUco + DLT + ankle-to-paw)
- ``metrics`` — Stage 3 (32 CatWalk params)
- ``report`` — Stage 4 (Excel/CSV/HTML/PNG)
- ``io`` — L2 I/O concerns that hang off core (e.g. PawPrint serializer)
- ``adapters`` — v1-shaped function aliases for GUI compatibility
"""
from __future__ import annotations

__version__ = "0.1.0.dev0"