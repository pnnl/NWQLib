"""QLS method on the original A and b, and its explicit verification against an independent reference."""

from importlib import import_module

_OWNERS = {
    "QLS": "method",
    "QLSAnalysis": "primary_records",
    "QLSVerification": "verification",
    "QLS_POLYNOMIAL_KAPPA_FLOOR": "constants",
    "QLS_TARGET_MARGIN": "constants",
}
__all__ = list(_OWNERS)


def __getattr__(name):
    if name not in _OWNERS:
        raise AttributeError(name)
    return getattr(import_module(f".{_OWNERS[name]}", __name__), name)
