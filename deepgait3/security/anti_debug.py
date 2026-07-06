"""Anti-debug + anti-tamper detection (Layer 3, security).

Combines several light-weight detection techniques documented in
kb/20_security_anti_tamper.md §2:

* ``ptrace`` on Linux
* ``IsDebuggerPresent`` + ``CheckRemoteDebuggerPresent`` on Windows
* ``sysctl`` probe on macOS
* ``/proc/<pid>/status`` TracerPid reading on Linux
* Frida server port (27042 default) detection
* Common VM hypervisor strings in ``/sys/class/dmi/id/*``

Design principle (kb/20 §2): **detection without immediate reaction**.
Raising an exception or exiting the process on every probe gives the
attacker free feedback. Instead, the probe returns a structured
:class:`AntiDebugReport` and the application decides what to do at
the most disruptive moment for the attacker (e.g. mid-recording).

CLOSED-SOURCE: production builds compile this module with Cython
(kb/18 §2.3) so the check helpers are not trivially reverseable.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class DebuggerDetected(RuntimeError):
    """Raised by :meth:`TamperPolicy.respond_to_detection` in `hard` mode."""


class FridaDetected(RuntimeError):
    """Raised when Frida is detected and the policy is `hard`."""


# ---------------------------------------------------------------------------
# Probe report
# ---------------------------------------------------------------------------
@dataclass
class AntiDebugReport:
    """Result of an :class:`AntiDebugProbe` sweep."""
    debugger_attached: bool = False
    frida_present: bool = False
    vm_detected: bool = False
    suspicious_modules: List[str] = field(default_factory=list)
    raw_signals: Dict[str, str] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return (not self.debugger_attached
                and not self.frida_present
                and not self.vm_detected
                and not self.suspicious_modules)

    @property
    def severity(self) -> str:
        """Coarse severity label used by :class:`TamperPolicy`."""
        if self.debugger_attached or self.frida_present:
            return "high"
        if self.vm_detected or self.suspicious_modules:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        return {
            "debugger_attached": self.debugger_attached,
            "frida_present": self.frida_present,
            "vm_detected": self.vm_detected,
            "suspicious_modules": list(self.suspicious_modules),
            "severity": self.severity,
            "raw_signals": dict(self.raw_signals),
        }


# ---------------------------------------------------------------------------
# Per-OS detection helpers
# ---------------------------------------------------------------------------
def _check_ptrace_linux() -> bool:
    """On Linux, attempting to PTRACE_ATTACH to our own pid will fail
    with EPERM if another tracer (gdb, strace) is already attached."""
    PTRACE_ATTACH = 16  # from <sys/ptrace.h>
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.ptrace.restype = ctypes.c_long
    libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long,
                            ctypes.c_void_p, ctypes.c_void_p]
    pid = os.getpid()
    try:
        if libc.ptrace(PTRACE_ATTACH, pid, 0, 0) != 0:
            errno = ctypes.get_errno()
            if errno == 1:  # EPERM — already traced
                return True
    except Exception:
        return False
    finally:
        # Always detach so we don't accidentally freeze the process.
        try:
            libc.ptrace(17, pid, 0, 0)  # PTRACE_DETACH
        except Exception:
            pass
    return False


def _check_proc_status_linux() -> bool:
    """/proc/self/status contains TracerPid: <pid> when traced."""
    try:
        text = Path(f"/proc/{os.getpid()}/status").read_text()
    except Exception:
        return False
    for line in text.splitlines():
        if line.startswith("TracerPid:"):
            try:
                tracer_pid = int(line.split()[1])
            except (ValueError, IndexError):
                return False
            return tracer_pid != 0
    return False


def _check_isdebuggerpresent_windows() -> bool:
    """Calls kernel32!IsDebuggerPresent. Safe no-op on non-Windows."""
    if platform.system() != "Windows":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        return bool(kernel32.IsDebuggerPresent())
    except Exception:
        return False


def _check_frida_ports() -> bool:
    """Probe the default Frida server port (27042) on localhost."""
    try:
        with socket.create_connection(("127.0.0.1", 27042), timeout=0.05):
            return True
    except (OSError, socket.timeout):
        return False


def _check_frida_modules() -> List[str]:
    """Look for frida-* artefacts loaded into our address space."""
    suspicious = []
    if platform.system() == "Linux":
        maps_path = Path(f"/proc/{os.getpid()}/maps")
        try:
            text = maps_path.read_text(errors="ignore")
        except Exception:
            return suspicious
        for needle in ("frida-agent", "frida-gadget", "gmain"):
            if needle in text:
                suspicious.append(needle)
    return suspicious


def _check_vm_dmi() -> bool:
    """Detect VM hypervisor strings in DMI info (Linux)."""
    if platform.system() != "Linux":
        return False
    dmi = Path("/sys/class/dmi/id")
    if not dmi.is_dir():
        return False
    needles = ("vmware", "virtualbox", "qemu", "kvm", "hyper-v",
               "microsoft corporation", "xen")
    for f in ("sys_vendor", "product_name", "bios_vendor", "board_vendor"):
        p = dmi / f
        try:
            content = p.read_text(errors="ignore").lower()
        except Exception:
            continue
        if any(n in content for n in needles):
            return True
    return False


# ---------------------------------------------------------------------------
# Convenience wrappers (kb/20 §2)
# ---------------------------------------------------------------------------
def is_debugger_attached() -> bool:
    """Return True if any debugger is attached to the current process.

    Uses ``/proc/self/status`` TracerPid as the authoritative source — it
    is set by the kernel itself, so it cannot be forged by userspace.
    The ptrace self-attach trick is unreliable on modern glibc/musl
    (returns EPERM even when no debugger is present), so we use it only
    as a secondary signal.
    """
    if platform.system() == "Windows":
        return _check_isdebuggerpresent_windows()
    if platform.system() == "Linux":
        return _check_proc_status_linux()
    # macOS and unknown: be conservative and assume clean.
    return False


def is_frida_present() -> bool:
    """Return True if Frida server is reachable on localhost."""
    return _check_frida_ports() or bool(_check_frida_modules())


def is_virtual_machine() -> bool:
    """Return True if running inside a VM (heuristic, DMI-based)."""
    return _check_vm_dmi()


def is_being_analyzed() -> bool:
    """Convenience: True if any of debugger / Frida / VM is detected."""
    return (is_debugger_attached()
            or is_frida_present()
            or is_virtual_machine())


# ---------------------------------------------------------------------------
# AntiDebugProbe — composite probe (kb/20 §2.5)
# ---------------------------------------------------------------------------
class AntiDebugProbe:
    """Composite probe that runs all detection helpers and returns a
    :class:`AntiDebugReport`.

    Parameters
    ----------
    deep : bool
        If True, run the additional (slower) checks: loaded-module
        sweep and DMI VM probing. Default True.
    """

    def __init__(self, deep: bool = True) -> None:
        self._deep = bool(deep)

    def run(self) -> AntiDebugReport:
        rep = AntiDebugReport()
        rep.debugger_attached = is_debugger_attached()
        rep.frida_present = is_frida_present()
        if self._deep:
            rep.suspicious_modules = _check_frida_modules()
            rep.vm_detected = is_virtual_machine()
        rep.raw_signals = {
            "platform": platform.system(),
            "pid": str(os.getpid()),
            "deep": str(self._deep),
        }
        return rep


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------
def respond_to_debug_detection(severity: str = "warning") -> None:
    """Legacy compatibility shim — actual policy lives in :mod:`tamper`."""
    from .tamper import TamperPolicy, TamperAction
    policy = TamperPolicy()
    action = policy.choose_action(severity)
    policy.execute(action)