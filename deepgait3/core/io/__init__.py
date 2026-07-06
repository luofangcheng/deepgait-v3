"""L2 I/O concerns that are core-coupled.

Currently holds the PawPrint serializer (moved from v2 ``dynamics_v04/serializer.py``
since serialization is an I/O concern, not an algorithm concern). Future I/O
modules for the other stages will land here.
"""
from .pawprint_serializer import save_pawprints, save_pawprint_database

__all__ = ["save_pawprints", "save_pawprint_database"]