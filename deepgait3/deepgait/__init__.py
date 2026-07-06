"""DEPRECATED: ``deepgait3.deepgait`` has been flattened.

The legacy sub-packages that lived under ``deepgait3.deepgait`` have been
moved to the ``deepgait3`` top level::

    deepgait3.core._legacy   (old algorithms, pending migration)
    deepgait3.gui            (application layer)
    deepgait3.hardware       (camera drivers, DLC runner)
    deepgait3.io             (HDF5, NWB, BIDS I/O)
    deepgait3.utils          (shared utilities)
    deepgait3.license        (license enforcement)
    deepgait3.security       (anti-tamper)

Importing this package still works (via re-exports), but emits a
:class:`DeprecationWarning`.  Update your code to import from the new
locations directly.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "deepgait3.deepgait is deprecated; import from deepgait3.{gui,core._legacy,...} directly",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export top-level subpackages so ``from deepgait3.deepgait.gui import X``
# still resolves for any external scripts that haven't been updated.
from deepgait3 import gui           # noqa: F401
from deepgait3 import hardware      # noqa: F401
from deepgait3 import io            # noqa: F401
from deepgait3 import utils         # noqa: F401
from deepgait3 import license       # noqa: F401
from deepgait3 import security      # noqa: F401
