"""Legacy v2.0 algorithm modules pending migration to the new 4-stage pipeline.

Modules here are actively used by the GUI but have not yet been refactored
into the proper ``core/{pawprint,calibration,triangulation,fusion,metrics,report}``
structure. Each module should be migrated one at a time by:

1. Extracting pure algorithm functions into the appropriate ``core/`` subpackage.
2. Updating GUI callers to use the new API.
3. Removing the old module from ``_legacy/``.

When a module is fully migrated, its corresponding Cython artefacts
(``.c`` sources and ``.so`` compiled objects) can also be removed.
"""
