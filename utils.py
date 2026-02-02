"""
Backwards-compatibility shim.

Legacy helper code was moved to `libs/utils/legacy_utils.py` to avoid confusion with the
`libs/utils/` package.
"""

from __future__ import annotations

import importlib
from typing import Any

_IMPL: Any = None


def _load_impl() -> Any:
    global _IMPL
    if _IMPL is None:
        _IMPL = importlib.import_module("libs.utils.legacy_utils")
    return _IMPL


def __getattr__(name: str) -> Any:  # noqa: D401
    return getattr(_load_impl(), name)


def __dir__() -> list[str]:
    impl = _load_impl()
    return sorted(set(globals().keys()) | set(dir(impl)))
