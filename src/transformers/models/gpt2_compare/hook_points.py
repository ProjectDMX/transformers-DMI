"""Compatibility shim for hook points (moved to monitoring.hook_points)."""

from monitoring.hook_points import (  # noqa: F401
    HookPoint,
    HookedRootModule,
    HookFunction,
    NamesFilter,
    LensHandle,
)

__all__ = [
    "HookPoint",
    "HookedRootModule",
    "HookFunction",
    "NamesFilter",
    "LensHandle",
]
