"""The built-in Methods and the two exceptions a caller catches around plan, solve and compare, imported on first access.

Records, options and functions of one family import from that family's
subpackage, such as `nwqlib.algorithms.qhd`. The Method protocol imports from
`nwqlib.algorithms.protocol` and the registry from `nwqlib.algorithms.registry`.
`_METHOD_OWNERS` is also the list that `registry.builtin_registrations` reads.
"""

from importlib import import_module

# The same configured-type owners supply exports and builtin discovery. Required
# scientific configuration is never guessed to construct an inventory instance.
_METHOD_OWNERS = {
    "QHD": "qhd.method",
    "ADAPT": "gcim.adapt",
    "FixedGCIM": "gcim.fixed_basis",
    "QFD": "qfd.method",
    "Lanczos": "lanczos.method",
    "QLS": "qls.method",
    "QCELS": "qpe.method",
    "SPE": "qpe.method",
    "RFE": "qpe.method",
    "RWPE": "qpe.method",
    "LCHS": "lchs.method",
    "ExpectationMethod": "expectation",
}
_OWNERS = {**_METHOD_OWNERS, "ApplicabilityError": "protocol", "ResourceLimitError": "protocol"}
__all__ = list(_OWNERS)


def __getattr__(name):
    if name not in _OWNERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_OWNERS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
