"""Bounded numerical inputs.

Native numerical dependencies load on selection.
"""

from importlib import import_module

from .access import InputManifest, ProductManifest

__all__ = ["InputManifest", "ProductManifest"]
_INPUT_EXPORTS = [
    "OperatorInput", "PeriodicStencil", "operator_input", "ingest_dense",
    "ingest_pauli", "ingest_pauli_masks", "pauli_table", "ingest_sparse", "ingest_periodic_stencil",
]
__all__ += _INPUT_EXPORTS
_ORDERED_EXPORTS = ["FermionTerms", "fermion_table", "ingest_fermion", "FactorizedOperatorProduct"]
__all__ += _ORDERED_EXPORTS
_REFINEMENT_EXPORTS = ["RefinementOption", "RefinementReceipt", "OperatorFactReport", "refine_operator_facts"]
__all__ += _REFINEMENT_EXPORTS
_DF_EXPORTS = ["DFManifest", "FactorizedHamiltonian", "DFConversionReceipt", "DFConversion", "ingest_df"]
__all__ += _DF_EXPORTS


def __getattr__(name):
    if name in _DF_EXPORTS:
        return getattr(import_module(".df", __name__), name)
    if name in _REFINEMENT_EXPORTS:
        return getattr(import_module(".refinement", __name__), name)
    if name in ("FermionTerms", "fermion_table", "ingest_fermion"):
        return getattr(import_module("._fermion", __name__), name)
    if name == "FactorizedOperatorProduct":
        return getattr(import_module("._factorized", __name__), name)
    if name in __all__:
        return getattr(import_module(".inputs", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
