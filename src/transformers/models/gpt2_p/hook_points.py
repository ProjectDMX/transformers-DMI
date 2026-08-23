"""Compatibility shim for DMI hook points."""

from dmi.hooks.point import (  # noqa: F401
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
