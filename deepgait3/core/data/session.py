"""Recording session model — 5-camera video set integrity.

A *recording session* is one acquisition event that produces a fixed
set of role-tagged MP4 files (bottom + left + right + top). This module
defines the filesystem layout and provides pure-stdlib helpers to:

  * Resolve the canonical path for a role inside a session
    (``<videos_root>/<animal_id>/<ts>/<role>.mp4``).
  * Scan ``videos_root`` for all session directories and build
    :class:`SessionInfo` records.
  * Verify a session is complete (no missing roles).

Zero Qt dependency. Designed to be called from GUI, CLI, or tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


REQUIRED_ROLES: tuple = ("bottom", "left", "right", "top")
"""Roles every valid recording session must contain."""


@dataclass
class SessionInfo:
    """In-memory description of a single recording session."""

    animal_id: str
    session_ts: str
    roles_present: Set[str] = field(default_factory=set)
    files: Dict[str, Path] = field(default_factory=dict)
    total_size_bytes: int = 0

    @property
    def is_complete(self) -> bool:
        return self.roles_missing == set()

    @property
    def roles_missing(self) -> Set[str]:
        return set(REQUIRED_ROLES) - self.roles_present


class SessionStore:
    """Filesystem-backed store of recording sessions.

    Layout::

        <videos_root>/
            <animal_id>/
                <session_ts>/
                    bottom.mp4
                    left.mp4
                    right.mp4
                    top.mp4
    """

    REQUIRED_ROLES: tuple = REQUIRED_ROLES

    def __init__(self, videos_root: Path) -> None:
        self.videos_root = Path(videos_root)

    # ------------------------------------------------------------------
    # Path resolution (writer side)
    # ------------------------------------------------------------------
    def recorded_path_for(
        self, animal_id: str, role: str, ts: str,
    ) -> Path:
        """Return ``<videos_root>/<animal_id>/<ts>/<role>.mp4``.

        Creates parent directories so the caller can hand the path
        straight to :class:`cv2.VideoWriter`.
        """
        safe_aid = (animal_id or "session").strip() or "session"
        d = self.videos_root / safe_aid / ts
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{role}.mp4"

    # ------------------------------------------------------------------
    # Scan / verify (reader side)
    # ------------------------------------------------------------------
    def list_sessions(self) -> List[SessionInfo]:
        """Return all sessions found under ``videos_root``.

        Skips any directory that has no second-level ``<ts>`` child
        and any MP4 whose stem is not in :data:`REQUIRED_ROLES`.
        """
        out: List[SessionInfo] = []
        if not self.videos_root.is_dir():
            return out
        for animal_dir in sorted(self.videos_root.iterdir()):
            if not animal_dir.is_dir():
                continue
            animal_id = animal_dir.name
            for ts_dir in sorted(animal_dir.iterdir()):
                if not ts_dir.is_dir():
                    continue
                info = self._build_session(animal_id, ts_dir.name, ts_dir)
                if info is not None:
                    out.append(info)
        return out

    def verify_session(self, animal_id: str, ts: str) -> Set[str]:
        """Return the set of :data:`REQUIRED_ROLES` missing from session.

        Empty set means the session is complete.
        """
        d = self.videos_root / animal_id / ts
        if not d.is_dir():
            return set(self.REQUIRED_ROLES)
        present = {
            p.stem
            for p in d.glob("*.mp4")
            if p.stem in self.REQUIRED_ROLES
        }
        return set(self.REQUIRED_ROLES) - present

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_session(
        self, animal_id: str, ts: str, ts_dir: Path,
    ) -> Optional[SessionInfo]:
        roles: Set[str] = set()
        files: Dict[str, Path] = {}
        total = 0
        for p in ts_dir.glob("*.mp4"):
            if p.stem in self.REQUIRED_ROLES:
                roles.add(p.stem)
                files[p.stem] = p
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        if not roles:
            # No recognised role files; treat the directory as not a
            # session and skip silently.
            return None
        return SessionInfo(
            animal_id=animal_id,
            session_ts=ts,
            roles_present=roles,
            files=files,
            total_size_bytes=total,
        )


__all__ = [
    "REQUIRED_ROLES",
    "SessionInfo",
    "SessionStore",
]
