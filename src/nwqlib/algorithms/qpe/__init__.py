"""QPE estimators QCELS, SPE, RFE and RWPE, with their analysis and verification records."""

from importlib import import_module

_OWNERS = {"QCELS": "method", "SPE": "method", "RFE": "method", "RWPE": "method",
           "QPEAnalysis": "records", "QPEVerification": "records"}
__all__ = list(_OWNERS)


def __getattr__(name):
    if name not in _OWNERS:
        raise AttributeError(name)
    return getattr(import_module("." + _OWNERS[name], __name__), name)
