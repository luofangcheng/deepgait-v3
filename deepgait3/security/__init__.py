"""Security subpackage — Layer 3 of the deepgait v2 architecture.

Modules
-------
* ``anti_debug``   — detect ptrace, IDA, x64dbg, Frida, sandbox
* ``integrity``    — SHA-256 self-hash + HMAC chain audit log
* ``tamper``       — tamper response policy

CLOSED-SOURCE NOTE: in production these modules are Cython-compiled
(kb/18_cython_nuitka.md §2.3) so the detection logic is not reverse
engineerable. The Python source here is the **debug build**.
"""
from .anti_debug import (
    AntiDebugProbe,
    AntiDebugReport,
    DebuggerDetected,
    FridaDetected,
    is_debugger_attached,
    is_frida_present,
    is_virtual_machine,
    is_being_analyzed,
    respond_to_debug_detection,
)
from .integrity import (
    IntegrityVerifier,
    IntegrityError,
    ModuleHashMismatch,
    compute_module_hashes,
)
from .tamper import TamperPolicy, TamperLevel, TamperAction, respond_to_detection

__all__ = [
    "AntiDebugProbe",
    "AntiDebugReport",
    "DebuggerDetected",
    "FridaDetected",
    "is_debugger_attached",
    "is_frida_present",
    "is_virtual_machine",
    "is_being_analyzed",
    "respond_to_debug_detection",
    "IntegrityVerifier",
    "IntegrityError",
    "ModuleHashMismatch",
    "compute_module_hashes",
    "TamperPolicy",
    "TamperLevel",
    "TamperAction",
    "respond_to_detection",
]