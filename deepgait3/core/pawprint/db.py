"""SQLite storage for footprint trial results.

Database layout
───────────────
  trial             — one row per experiment
  footprint_cycle   — one row per detected print cycle
  footprint_frame   — per-frame data within a cycle
  mouse_roi         — per-frame mouse bounding boxes
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List

from .models import TrialResult, FootprintCycle, FrameRecord, MouseRoi

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trial (
    trial_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    mouse_id     TEXT    NOT NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    input_dir    TEXT    NOT NULL,
    num_frames   INTEGER NOT NULL,
    frame_width  INTEGER NOT NULL,
    frame_height INTEGER NOT NULL,
    fps          REAL    NOT NULL DEFAULT 60.0,
    px_per_mm    REAL    NOT NULL DEFAULT 1.92,
    roi_pad      INTEGER NOT NULL DEFAULT 50,
    tau_paw      REAL    NOT NULL DEFAULT 10.0
);

CREATE TABLE IF NOT EXISTS mouse_roi (
    roi_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id     INTEGER NOT NULL REFERENCES trial(trial_id),
    frame        INTEGER NOT NULL,
    tight_x1     INTEGER NOT NULL,
    tight_y1     INTEGER NOT NULL,
    tight_x2     INTEGER NOT NULL,
    tight_y2     INTEGER NOT NULL,
    expanded_x1  INTEGER NOT NULL,
    expanded_y1  INTEGER NOT NULL,
    expanded_x2  INTEGER NOT NULL,
    expanded_y2  INTEGER NOT NULL,
    area_px      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS footprint_cycle (
    cycle_id                 INTEGER PRIMARY KEY,
    trial_id                 INTEGER NOT NULL REFERENCES trial(trial_id),
    touchdown_frame          INTEGER NOT NULL,
    liftoff_frame            INTEGER NOT NULL,
    peak_area_frame          INTEGER NOT NULL,
    peak_intensity_frame     INTEGER NOT NULL,
    duration_s               REAL    NOT NULL,
    max_area_mm2             REAL    NOT NULL,
    max_area_px              INTEGER NOT NULL,
    centroid_at_peak_x_mm    REAL    NOT NULL,
    centroid_at_peak_y_mm    REAL    NOT NULL,
    bbox_peak_x1             INTEGER NOT NULL,
    bbox_peak_y1             INTEGER NOT NULL,
    bbox_peak_x2             INTEGER NOT NULL,
    bbox_peak_y2             INTEGER NOT NULL,
    loading_duration_s       REAL    NOT NULL,
    weight_bearing_duration_s REAL   NOT NULL,
    unloading_duration_s     REAL    NOT NULL,
    touchdown_intensity      REAL    NOT NULL,
    liftoff_intensity        REAL    NOT NULL,
    is_clean_liftoff         INTEGER NOT NULL,
    n_frames                 INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS footprint_frame (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id        INTEGER NOT NULL REFERENCES footprint_cycle(cycle_id),
    frame           INTEGER NOT NULL,
    time_s          REAL    NOT NULL,
    area_mm2        REAL    NOT NULL,
    area_px         INTEGER NOT NULL,
    centroid_x_mm   REAL    NOT NULL,
    centroid_y_mm   REAL    NOT NULL,
    bbox_x1         INTEGER NOT NULL,
    bbox_y1         INTEGER NOT NULL,
    bbox_x2         INTEGER NOT NULL,
    bbox_y2         INTEGER NOT NULL,
    mean_intensity  REAL    NOT NULL,
    peak_intensity  REAL    NOT NULL,
    mean_pressure   REAL    NOT NULL,
    peak_pressure   REAL    NOT NULL,
    is_peak_area    INTEGER NOT NULL,
    is_peak_intensity INTEGER NOT NULL,
    png_path         TEXT
);

CREATE INDEX IF NOT EXISTS idx_mouse_roi_trial ON mouse_roi(trial_id, frame);
CREATE INDEX IF NOT EXISTS idx_cycle_trial ON footprint_cycle(trial_id);
CREATE INDEX IF NOT EXISTS idx_cycle_td ON footprint_cycle(touchdown_frame);
CREATE INDEX IF NOT EXISTS idx_frame_cycle ON footprint_frame(cycle_id, frame);
"""


def create_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the database and ensure the schema exists."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def save_trial(conn: sqlite3.Connection, result: TrialResult) -> int:
    """Persist a full TrialResult and return its trial_id."""
    cur = conn.execute(
        """INSERT INTO trial (mouse_id, input_dir, num_frames,
           frame_width, frame_height, fps, px_per_mm, roi_pad, tau_paw)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (result.mouse_id, result.input_dir, result.num_frames,
         result.frame_width, result.frame_height, result.fps,
         result.px_per_mm, result.roi_pad, result.tau_paw),
    )
    trial_id = cur.lastrowid

    # mouse ROIs
    for roi in result.mouse_rois:
        conn.execute(
            """INSERT INTO mouse_roi
               (trial_id, frame, tight_x1, tight_y1, tight_x2, tight_y2,
                expanded_x1, expanded_y1, expanded_x2, expanded_y2, area_px)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trial_id, roi.frame,
             roi.tight_xyxy[0], roi.tight_xyxy[1],
             roi.tight_xyxy[2], roi.tight_xyxy[3],
             roi.expanded_xyxy[0], roi.expanded_xyxy[1],
             roi.expanded_xyxy[2], roi.expanded_xyxy[3],
             roi.area_px),
        )

    # cycles + per-frame records
    for c in result.cycles:
        conn.execute(
            """INSERT INTO footprint_cycle
               (cycle_id, trial_id, touchdown_frame, liftoff_frame,
                peak_area_frame, peak_intensity_frame, duration_s,
                max_area_mm2, max_area_px, centroid_at_peak_x_mm,
                centroid_at_peak_y_mm, bbox_peak_x1, bbox_peak_y1,
                bbox_peak_x2, bbox_peak_y2, loading_duration_s,
                weight_bearing_duration_s, unloading_duration_s,
                touchdown_intensity, liftoff_intensity, is_clean_liftoff,
                n_frames)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?)""",
            (c.cycle_id, trial_id, c.touchdown_frame, c.liftoff_frame,
             c.peak_area_frame, c.peak_intensity_frame, c.duration_s,
             c.max_area_mm2, c.max_area_px, c.centroid_at_peak_x_mm,
             c.centroid_at_peak_y_mm,
             c.bbox_at_peak_xyxy[0], c.bbox_at_peak_xyxy[1],
             c.bbox_at_peak_xyxy[2], c.bbox_at_peak_xyxy[3],
             c.loading_duration_s, c.weight_bearing_duration_s,
             c.unloading_duration_s,
             c.touchdown_intensity, c.liftoff_intensity,
             int(c.is_clean_liftoff), c.n_frames),
        )
        for fr in c.frames:
            conn.execute(
                """INSERT INTO footprint_frame
                   (cycle_id, frame, time_s, area_mm2, area_px,
                    centroid_x_mm, centroid_y_mm,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    mean_intensity, peak_intensity, mean_pressure,
                    peak_pressure, is_peak_area, is_peak_intensity,
                    png_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (c.cycle_id, fr.frame, fr.time_s, fr.area_mm2, fr.area_px,
                 fr.centroid_x_mm, fr.centroid_y_mm,
                 fr.bbox_x1, fr.bbox_y1, fr.bbox_x2, fr.bbox_y2,
                 fr.mean_intensity, fr.peak_intensity,
                 fr.mean_pressure, fr.peak_pressure,
                 int(fr.is_peak_area), int(fr.is_peak_intensity),
                 fr.png_path or None),
            )

    conn.commit()
    return int(trial_id)


__all__ = ["create_db", "save_trial"]
