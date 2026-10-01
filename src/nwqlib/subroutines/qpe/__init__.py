"""Coherent QPE construction; statistical methods live in algorithms.qpe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coherent import CoherentQPECircuit, build_coherent_qpe_circuit

__all__ = ["CoherentQPECircuit", "build_coherent_qpe_circuit"]


def __getattr__(name):
    if name in {"CoherentQPECircuit", "build_coherent_qpe_circuit"}:
        from importlib import import_module
        return getattr(import_module(".coherent", __name__), name)
    raise AttributeError(name)
