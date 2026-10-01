"""Chebyshev Lanczos configuration and explicit sensitivity sampling."""

from importlib import import_module

_OWNERS = {"Lanczos": "method", "SensitivitySampling": "method", "LanczosResult": "records"}
__all__ = list(_OWNERS)


def __getattr__(name):
    if name not in _OWNERS:
        raise AttributeError(name)
    return getattr(import_module(f".{_OWNERS[name]}", __name__), name)
