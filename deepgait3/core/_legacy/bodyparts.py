"""Bodypart definitions for deepgait (12-point scheme, matching VGL).

VGL 12 bodyparts:
    Nose, Butt,
    FrontRight1, FrontRight2,   # toe, heel
    FrontLeft1,  FrontLeft2,
    HindRight1,  HindRight2,
    HindLeft1,   HindLeft2,
    MidPointRight, MidPointLeft

Each paw has toe (1) + heel (2).  Body axis: Nose -> CoM -> Butt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class Paw:
    """A single paw's bodypart names."""
    name: str           # e.g. "RightHind"
    toe: str            # e.g. "HindRight1"
    heel: str           # e.g. "HindRight2"
    side: str           # "left" | "right"
    limb: str           # "fore" | "hind"


# ---------------------------------------------------------------------------
# 12-point VGL scheme
# ---------------------------------------------------------------------------

BODYPARTS_12 = [
    "Nose",
    "Butt",
    "FrontRight1", "FrontRight2",
    "FrontLeft1",  "FrontLeft2",
    "HindRight1",  "HindRight2",
    "HindLeft1",   "HindLeft2",
    "MidPointRight", "MidPointLeft",
]

PAWS = [
    Paw("RightFore",  "FrontRight1", "FrontRight2", "right", "fore"),
    Paw("LeftFore",   "FrontLeft1",  "FrontLeft2",  "left",  "fore"),
    Paw("RightHind",  "HindRight1",  "HindRight2",  "right", "hind"),
    Paw("LeftHind",   "HindLeft1",   "HindLeft2",   "left",  "hind"),
]

BODY_AXIS = {
    "nose": "Nose",
    "butt": "Butt",
    "mid_right": "MidPointRight",
    "mid_left": "MidPointLeft",
}


def get_paw_bodyparts(paw: Paw) -> tuple[str, str]:
    """Return (toe, heel) names for a paw."""
    return paw.toe, paw.heel


def get_all_paw_names() -> list[str]:
    """Return all paw toe+heel names (8 total)."""
    names = []
    for p in PAWS:
        names.extend([p.toe, p.heel])
    return names


def get_paw_by_bodypart(name: str) -> Paw | None:
    """Given a toe or heel name, return the parent Paw."""
    for p in PAWS:
        if name in (p.toe, p.heel):
            return p
    return None
