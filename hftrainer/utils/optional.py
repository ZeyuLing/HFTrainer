"""Helpers for optional, method-specific dependencies.

The core package must stay importable when heavyweight model stacks are not
installed.  Method adapters call :func:`require_modules` at the point where a
backend is actually constructed and report the exact extra that provides it.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType


def require_modules(
    modules: Iterable[str],
    *,
    feature: str,
    install_hint: str,
) -> dict[str, ModuleType]:
    """Import optional modules or raise one actionable error.

    Args:
        modules: Fully-qualified module names to import.
        feature: Human-readable feature name used in the error message.
        install_hint: Command that installs the optional dependency set.

    Returns:
        Mapping from the requested module name to the imported module.
    """

    imported: dict[str, ModuleType] = {}
    missing: list[str] = []
    for module_name in modules:
        try:
            imported[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # A missing transitive package is part of the optional environment
            # just as much as a missing top-level package. Keep non-import
            # runtime failures untouched, but aggregate ModuleNotFoundError into
            # the same actionable installation hint.
            missing.append(exc.name or module_name)

    if missing:
        names = ', '.join(missing)
        raise ImportError(
            f"{feature} requires optional modules that are not installed: {names}. "
            f"Install the supported backend with: {install_hint}"
        )
    return imported
