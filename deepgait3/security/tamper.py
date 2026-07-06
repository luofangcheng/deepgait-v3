"""Tamper response policy (Layer 3, security).

Anti-debug and integrity probes only *detect* — :mod:`tamper` decides
what to do about it. Per kb/20 §2 ("反调试的最佳实践：不立即反应，
而是在关键时刻出错"), the policy must be configurable so the same
detection signal can produce different responses at different call
sites:

* ``low``     — log only (default; suitable for probes run on startup)
* ``medium``  — degrade to read-only / trial-mode
* ``high``    — refuse to load Cython algorithms and refuse to export
* ``hard``    — terminate the process immediately (drastic; risky)

References
----------
* kb/20_security_anti_tamper.md §2 "respond_to_debug_detection"
"""
from __future__ import annotations

import enum
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict


logger = logging.getLogger(__name__)


class TamperLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HARD = "hard"


class TamperAction(str, enum.Enum):
    LOG = "log"
    DEGRADE = "degrade"
    REFUSE = "refuse"
    TERMINATE = "terminate"


@dataclass
class TamperPolicy:
    """Map detection severity → action.

    Attributes
    ----------
    level_overrides : dict
        Optional per-level override. Default mapping follows kb/20:

        * low     → log
        * medium  → log + degrade
        * high    → log + refuse
        * hard    → log + refuse + terminate
    on_degrade : callable, optional
        Fired when the action is ``degrade`` (e.g. switch to trial mode).
    on_refuse : callable, optional
        Fired when the action is ``refuse`` (e.g. raise in business code).
    """

    level_overrides: Dict[TamperLevel, TamperAction] = None  # type: ignore[assignment]
    on_degrade: Callable[[], None] = None
    on_refuse: Callable[[], None] = None

    def __post_init__(self) -> None:
        if self.level_overrides is None:
            self.level_overrides = {
                TamperLevel.LOW:    TamperAction.LOG,
                TamperLevel.MEDIUM: TamperAction.DEGRADE,
                TamperLevel.HIGH:   TamperAction.REFUSE,
                TamperLevel.HARD:   TamperAction.TERMINATE,
            }

    def choose_action(self, severity: str) -> TamperAction:
        try:
            level = TamperLevel(severity)
        except ValueError:
            level = TamperLevel.LOW
        return self.level_overrides.get(level, TamperAction.LOG)

    def execute(self, action: TamperAction) -> None:
        if action == TamperAction.LOG:
            logger.warning("TamperPolicy: tamper signal (log-only)")
        elif action == TamperAction.DEGRADE:
            logger.error("TamperPolicy: tampering detected — DEGRADING")
            if self.on_degrade is not None:
                try:
                    self.on_degrade()
                except Exception:
                    logger.exception("TamperPolicy: on_degrade raised")
        elif action == TamperAction.REFUSE:
            logger.error("TamperPolicy: tampering detected — REFUSING")
            if self.on_refuse is not None:
                try:
                    self.on_refuse()
                except Exception:
                    logger.exception("TamperPolicy: on_refuse raised")
        elif action == TamperAction.TERMINATE:
            logger.critical("TamperPolicy: tampering detected — TERMINATING")
            # Flush logs, then exit. Hard exit avoids giving the attacker
            # further time to inspect memory.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0xDEAD)
        else:
            logger.error("TamperPolicy: unknown action %r", action)


def respond_to_detection(
    severity: str = "warning",
    policy: TamperPolicy = None,  # type: ignore[assignment]
) -> TamperAction:
    """Convenience entry point used by business code.

    Returns the :class:`TamperAction` that was chosen (useful for tests).
    """
    if policy is None:
        policy = TamperPolicy()
    action = policy.choose_action(severity)
    policy.execute(action)
    return action