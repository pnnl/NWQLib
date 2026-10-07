"""GCiM methods (FixedGCIM and ADAPT) and their chemistry and sector helpers, imported on first access.

Discovery and fixed planning do not import native providers or chemistry.
"""

from importlib import import_module

_EXPORTS = {
    **dict.fromkeys(
        ("FixedGCIM", "FixedGCIMBasis", "FixedGCIMResult", "ProjectedPencil", "ScalarEstimate"),
        "fixed_basis",
    ),
    "ADAPT": "adapt",
    **dict.fromkeys(("ADAPTResult", "AdaptRound"), "adapt_records"),
    "AdaptVerificationOptions": "adapt_verification",
    **dict.fromkeys(
        (
            "GCIMChemistryProblemData",
            "build_gcim_chemistry_problem",
            "chemistry_reference_diagnostic",
            "chemistry_reference_report_section",
            "closed_shell_reference_occupations",
            "correlation_fraction",
            "qubit_operator_to_sparse_pauli",
            "CHEMISTRY_EXTRA_MESSAGE",
        ),
        "chemistry",
    ),
    "sector_expectations": "sector",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
