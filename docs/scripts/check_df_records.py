"""Fresh-process supplied-DF conversion/FixedGCIM scan import-attempt audit."""

import argparse

import builtins
import importlib.abc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def supplied_df_scan():
    """Convert supplied two-orbital factors, plan FixedGCIM and scan it with the conversion cost.

    The factors satisfy V=((3,-4),(4,3))/5, w=(1,2), T=-B^2/2 and E0=3. The occupation
    column |0011> has Rayleigh quotient 3556/625 Ha, a trial energy rather than a ground claim.
    No geometry, decomposition, reference solve or native execution runs. Imports stay inside
    this function so the audit guard sees them.
    """
    import numpy as np
    import nwqlib
    from nwqlib.algorithms import FixedGCIM
    from nwqlib.core import Basis, InputRef, Scope, Source, Unit
    from nwqlib.operators import ingest_df
    from nwqlib.problems.inputs import ingest_occupation
    from nwqlib.resources import ResourceContext
    from nwqlib.search import Candidate, Objective

    energy = Unit(symbol="Ha", dimension="energy")
    source = InputRef(
        identity="rational-four-mode-audit",
        representation="supplied factor dataset",
        source=Source(
            name="hand algebra",
            version="1",
            domain="two spatial orbitals",
            reference="V=((3,-4),(4,3))/5; w=(1,2); T=-B^2/2; E0=3",
        ),
    )
    df = ingest_df(
        -np.array([[73.0, -36.0], [-36.0, 52.0]]) / 50,
        ((np.array([[3.0, -4.0], [4.0, 3.0]]) / 5, np.array([1.0, 2.0])),),
        constant_energy=3.0,
        orbital_basis=Basis(identity="two supplied orbitals", dimension=2, ordering="p=0,1"),
        energy_unit=energy,
        source=source,
    )
    converted = df.to_pauli(max_bytes=16_000_000, max_products=4_000_000)
    # The tuple is in increasing wire order, so (1,1,0,0) denotes ket |0011>.
    problem = nwqlib.Eigenproblem(
        A=converted.operator,
        unit=energy,
        scope=Scope(domain="converted four-mode Hamiltonian; explicit occupation trial column"),
    )
    occupation = ingest_occupation((1, 1, 0, 0), num_qubits=4)
    selected = nwqlib.plan(problem, method=FixedGCIM(basis=(occupation,)), seed=7)
    # Ranking includes the actual conversion occurrence as prior work.
    searched = nwqlib.scan(
        (Candidate(selected, prior_work=(converted.work_fact,)),),
        objectives=(Objective(kind="logical_width"),),
        context=ResourceContext(batch_schedule="serial"),
    )
    return converted, searched, searched.comparison.rows[0].estimate


def main(argv=None):
    """Exercise finite supplied-factor conversion and scan while blocking chemistry and SDK
    dependencies.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; this audit must fail")
    args = parser.parse_args(argv)
    from docs.scripts.check_core_records import BLOCKED

    blocked = tuple(name for name in BLOCKED if not name.startswith("nwqlib.")) + (
        "openfermion",
        "pyscf",
        "qiskit_nature",
        "nwqlib.algorithms.gcim.chemistry",
        "nwqlib.backends.qiskit_aer",
        "nwqlib.backends.ibm_runtime",
        "nwqlib.backends.ionq",
        "nwqlib.backends.nexus",
        "nwqlib.backends.nwqsim",
        "nwqlib.backends.nwqec",
        "matplotlib",
        "pandas",
        "IPython",
        "nbformat",
        "jupyter",
    )
    attempts = []

    def forbidden(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)

    def record(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("blocked DF dependency import: " + name)

    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        record(name)
        for item in fromlist or ():
            record(name + "." + item)
        return original(name, globals, locals, fromlist, level)

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            record(fullname)
            return None

    # Intercept both import entry points and record attempts before raising.
    # The final assertion must also catch forbidden imports swallowed by callers.
    builtins.__import__ = guarded
    sys.meta_path.insert(0, Blocker())
    if args.poison:
        try:
            __import__("pyscf")
        except ImportError:
            pass
    from nwqlib.operators import DFConversionReceipt
    from nwqlib.search import SearchSelection

    converted, searched, resources = supplied_df_scan()
    receipt = DFConversionReceipt.model_validate_json(converted.receipt.model_dump_json())
    assert receipt.output == converted.operator.manifest
    selection = SearchSelection.model_validate_json(searched.selection.model_dump_json())
    selection.validate_comparison(searched.comparison)
    assert selection.prior_work[0][0].evidence.artifact == receipt.content_id
    assert resources.construction_id == searched.select(0).plan.construction.content_id
    loaded = [name for name in sys.modules if forbidden(name)]
    assert not (attempts or loaded), f"DF isolation violation: attempts={attempts}, loaded={loaded}"
    print(
        "Supplied DF conversion, receipt, FixedGCIM resources/scan passed; attempts=[], loaded=[]"
    )


if __name__ == "__main__":
    main()
