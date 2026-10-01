#!/usr/bin/env python3
"""Fresh-process real two-method scientist workflow and swallowed-import control."""

import argparse

import builtins
import importlib.abc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def stored_context():
    """Explicit synthetic declarations; missing coverage and time remain unknown."""
    from datetime import datetime, timezone
    from nwqlib.backends import AER_STATEVECTOR_TARGET
    from nwqlib.backends.profiles import Allocation, DeviceConfiguration, DeviceProfile
    from nwqlib.core import Limit, Source, Unit
    from nwqlib.evidence import Evidence
    from nwqlib.resources import ResourceContext

    source = Source(
        name="synthetic tiny energy device",
        version="1",
        domain="example declaration, not measured hardware calibration",
        reference="docs/scripts/check_scientist_records.py",
    )
    recorded = datetime(2026, 9, 1, tzinfo=timezone.utc)
    expires = datetime(2026, 10, 1, tzinfo=timezone.utc)
    # Describe a synthetic runtime/build/precision combination. The source
    # label prevents this example declaration from masquerading as calibration.
    configuration = DeviceConfiguration(
        name="synthetic local Aer grant",
        version="1",
        target=AER_STATEVECTOR_TARGET,
        hardware=source,
        runtime=source.revise(name=AER_STATEVECTOR_TARGET.name),
        build=source,
        compiler=source,
        precision="complex128",
        representation="statevector",
    )
    evidence = Evidence(kind="external_specification", source=source)
    profile = DeviceProfile(
        configuration=configuration, recorded_at=recorded, valid_until=expires, evidence=evidence
    )
    # Attach a one-MiB capacity grant to that exact configuration. This is
    # a stock constraint, not cumulative memory traffic or a runtime estimate.
    allocation = Allocation(
        name="synthetic 1 MiB logical-device grant",
        configuration_id=configuration.content_id,
        locations=("logical_device",),
        topology="single_device",
        limits=(
            Limit(
                stage="execution",
                metric="memory",
                unit=Unit(symbol="byte", dimension="bytes"),
                kind="capacity_stock",
                value=1048576,
                scope="logical_device",
            ),
        ),
        recorded_at=recorded,
        valid_until=expires,
        evidence=evidence,
    )
    # Freeze the assessment time within the supplied validity window.
    return dict(
        profile=profile,
        allocation=allocation,
        context=ResourceContext(batch_schedule="serial"),
        assessed_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
    )


def child(poison=False):
    """Trace actual compare/scan/rescore operations while prohibiting repeated input, planning or
    native work.
    """
    from docs.scripts.check_core_records import BLOCKED

    forbidden_prefixes = tuple(name for name in BLOCKED if not name.startswith("nwqlib.")) + (
        "nwqlib.backends.qiskit_aer",
        "nwqlib.backends.ibm_runtime",
        "nwqlib.backends.ionq",
        "nwqlib.backends.nexus",
        "nwqlib.backends.nwqsim",
        "nwqlib.backends.nwqec",
        "nwqlib.backends.qre",
        "openfermion",
        "pyscf",
        "qiskit_nature",
        "matplotlib",
        "pandas",
        "IPython",
        "nbformat",
        "jupyter",
    )
    attempts = []

    def forbidden(name):
        return any(name == p or name.startswith(p + ".") for p in forbidden_prefixes)

    def record(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("blocked SDK/provider import: " + name)

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
    if poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass
    import nwqlib
    from nwqlib.algorithms import FixedGCIM, Lanczos
    from nwqlib.algorithms.qpe import QCELS
    from nwqlib.core import Scope
    from nwqlib.problems.inputs import ingest_vector
    from nwqlib.resources import ResourceContext
    from nwqlib.search import Candidate, Objective, SearchSelection

    # Select three energy methods on one fixed matrix whose two supplied
    # preparations span the complete trial space. No reference solve runs.
    initial = ingest_vector([1.0, 0.0])
    planned = nwqlib.compare(
        nwqlib.Eigenproblem(A=[[1.0, 0.25], [0.25, 2.0]], unit="Ha", scope=Scope(domain="fixed energy comparison 1")),
        methods=(
            Lanczos(initial_state=initial, krylov_dimension=2),
            FixedGCIM(basis=(initial, ingest_vector([0.0, 1.0]))),
            QCELS(initial_state=initial),
        ),
        context=ResourceContext(batch_schedule="serial"),
        seed=7,
    )
    assert tuple(row.method.descriptor.method for row in planned.rows) == (
        "chebyshev_lanczos", "fixed_gcim", "qcels")
    assert all(row.plan is not None for row in planned.rows)
    plans = tuple(row.plan for row in planned.rows)
    assert all(plan.problem.content_id == planned.problem.content_id for plan in plans)
    context = stored_context()
    events = []

    def blocked(*args, **kwargs):
        events.append("forbidden repeated work")
        raise AssertionError(events[-1])

    import socket
    import nwqlib.blocks as blocks
    import nwqlib.execution as execution
    from nwqlib.operators.inputs import OperatorInput

    for plan in plans:
        type(plan.method).plan = blocked
    OperatorInput.pauli_terms = blocked
    blocks.lower_qiskit = blocked
    execution.prepare = execution.submit = blocked
    socket.socket = blocked
    # Explicit operations may fold/assess. They must not replan or scan inputs.
    assert (
        nwqlib.estimate(plans[0], context=context["context"]).construction_id
        == plans[0].construction.content_id
    )
    search_context = {key: value for key, value in context.items() if key != "allocation"}
    searched = nwqlib.scan(
        tuple(Candidate(plan, allocation=context["allocation"]) for plan in plans),
        objectives=(Objective(kind="logical_width"),),
        **search_context,
    )
    comparison = searched.comparison
    points = sum(len(row.estimate.assessments) for row in comparison.rows)
    assert points > 0 and searched.selection.work.assessed_points == points
    import nwqlib.backends as backends
    import nwqlib.resources as resources

    import nwqlib.search as search

    backends.assess = resources.estimate = search.estimate_plan = blocked
    row = comparison.select(0)
    assert row is comparison.rows[0] and row.plan is plans[0]
    rescored = nwqlib.scan(comparison, objectives=searched.selection.objectives)
    assert rescored.comparison is comparison
    assert rescored.selection.work.assessed_points == rescored.selection.work.base_folds == 0
    assert rescored.selection.work.reused_points == points
    assert rescored.selection.values == searched.selection.values
    assert rescored.selection.nondominated == searched.selection.nondominated
    stored_search = SearchSelection.model_validate_json(rescored.selection.model_dump_json())
    stored_search.validate_comparison(comparison)
    loaded_modules = [name for name in sys.modules if forbidden(name)]
    if attempts or loaded_modules or events:
        raise AssertionError(
            f"scientist isolation violation: attempts={attempts}, loaded={loaded_modules}, events={events}"
        )
    print(
        "Actual root planning/comparison/scan/selection readback: zero SDK, chemistry, notebook or repeated input/native work."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; this audit must fail")
    args = parser.parse_args(argv)
    child(args.poison)


if __name__ == "__main__":
    main()
