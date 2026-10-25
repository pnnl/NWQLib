"""Quantum filter diagonalization in a fixed product-formula basis."""

from importlib import import_module

_OWNERS = {"QFD": "method", "QFDResult": "records", "QFDReconstruction": "records",
           "QFDEvolutionDiagnostics": "records", "chemistry_evolution_terms": "chemistry"}
__all__ = list(_OWNERS)


def __getattr__(name):
    if name not in _OWNERS:
        raise AttributeError(name)
    return getattr(import_module(f".{_OWNERS[name]}", __name__), name)
