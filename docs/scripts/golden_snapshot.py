#!/usr/bin/env python3
"""Generate and compare deterministic JSON snapshots for maintainer checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _lchs_report():
    from nwqlib import LinearDynamics, solve
    from nwqlib.algorithms import LCHS

    problem = LinearDynamics(A=[[0.4, 0.05], [0.05, 0.5]], initial_state=[1.0, 0.25], time=0.3)
    return solve(problem, method=LCHS(), execution="classical", seed=7).report()


def _lchs_inhomogeneous_report():
    from nwqlib import LinearDynamics, solve
    from nwqlib.algorithms import LCHS

    problem = LinearDynamics(
        A=[[0.35, 0.05j], [0.02j, 0.45]], initial_state=[1.0, 0.25], source=[0.1, -0.05j], time=0.4
    )
    # Two explicit Duhamel nodes keep this snapshot small; this is the chosen
    # finite source approximation, not a reference or a default-accuracy claim.
    return solve(problem, method=LCHS(duhamel_nodes=2), execution="classical", seed=7).report()


def _qpe_plan():
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms import SPE

    return plan(
        Eigenproblem(A=[[0.2, 0.0], [0.0, 0.7]]),
        method=SPE(initial_state=[1, 0], overlap_lower_bound=1.),
        seed=7,
    ).to_record()


def _lanczos_plan():
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms import Lanczos

    return plan(
        Eigenproblem(A=[[1, 0], [0, -1]]),
        method=Lanczos(initial_state=[1, 1], krylov_dimension=2),
        seed=7,
    ).to_record()


def _gcim_plan():
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms import FixedGCIM

    return plan(
        Eigenproblem(A=[[1, 0], [0, -1]]), method=FixedGCIM(basis=([1, 1], [1, 1j])), seed=7
    ).to_record()


def _adapt_plan():
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms import ADAPT
    from nwqlib.operators import ingest_pauli

    pool = (ingest_pauli((("Y", 1j),), num_qubits=1),)
    return plan(
        Eigenproblem(A=[[1, 0], [0, -1]]),
        method=ADAPT(initial_state=[1, 1], pool=pool, max_iterations=1),
        seed=7,
    ).to_record()


def _expectation_plan():
    from nwqlib import Expectation, plan
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.operators import ingest_pauli

    # <psi|(I+Z)|psi>/<psi|psi> for the physical vector (2,2) is 8/8 = 1.
    problem = Expectation(
        state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)
    )
    return plan(problem, method=ExpectationMethod(), seed=7).to_record()


def _qls_report():
    from nwqlib import LinearSystem, solve
    from nwqlib.algorithms import QLS

    return solve(
        LinearSystem(A=[[1.0, 0.0], [0.0, 2.0]], b=[1.0, 0.25]),
        method=QLS(),
        execution="classical",
        seed=7,
    ).report()


def _qhd_report():
    import sympy as sp
    from nwqlib import Optimization, solve
    from nwqlib.algorithms import QHD

    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x**2, variables=(x,), bounds=((-1.0, 1.0),))
    return solve(problem, method=QHD(), execution="classical", seed=7).report()


def _lchs_resource_snapshot(operation, *, qsp=False):
    from nwqlib import LinearDynamics, NormSquared, estimate, plan, prepare
    from nwqlib.algorithms.lchs import LCHS, ProviderConfig
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    from nwqlib.resources import ResourceContext

    method = LCHS(
        hamiltonian_evolution_backend="qsp_block_encoding" if qsp else "dense_exact",
        k_quadrature=ProviderConfig(
            implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": -1}
        ),
    )
    selected = plan(
        LinearDynamics(A=np.diag([0.1, 0.2]), initial_state=[1.0, 0.25], time=0.1),
        method=method,
        output=NormSquared(),
        seed=7,
    )
    width = sum(register.width for register in selected.construction.program.registers)
    if width > 13:
        raise ValueError("snapshot exceeds its declared 13-qubit native scope")
    if operation == "formula":
        return estimate(selected, context=ResourceContext(basis="cx")).model_dump(mode="json")
    if operation == "sample":
        return sample_resources(selected, max_qubits=13)
    if operation != "full":
        raise ValueError("unknown snapshot resource operation")
    prepared = prepare(selected)
    try:
        return prepared.inspect_resources()
    finally:
        prepared.run.close()


def _method_schemas():
    from nwqlib.algorithms.registry import builtin_registrations, options_schema

    # Required scientific inputs stay required in their actual Method schema;
    # reading defaults never constructs dummy states, bases or operator pools.
    return {row.source.name: options_schema(row) for row in builtin_registrations()}


ARTIFACT_BUILDERS = {
    "reports/lchs.json": _lchs_report,
    "reports/lchs_inhomogeneous.json": _lchs_inhomogeneous_report,
    "plans/qpe.json": _qpe_plan,
    "plans/chebyshev_lanczos.json": _lanczos_plan,
    "plans/fixed_gcim.json": _gcim_plan,
    "plans/adapt_gcim.json": _adapt_plan,
    "plans/finite_pauli_expectation.json": _expectation_plan,
    "reports/qls.json": _qls_report,
    "reports/qhd.json": _qhd_report,
    "resource_estimates/lchs_full.json": lambda: _lchs_resource_snapshot("full"),
    "resource_estimates/lchs_sample.json": lambda: _lchs_resource_snapshot("sample"),
    "resource_estimates/lchs_formula.json": lambda: _lchs_resource_snapshot("formula"),
    "resource_estimates/lchs_qsp_sample.json": lambda: _lchs_resource_snapshot("sample", qsp=True),
    "resource_estimates/lchs_qsp_formula.json": lambda: _lchs_resource_snapshot(
        "formula", qsp=True
    ),
}


def _roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return type(a) is type(b) and a == b


def generate(out_dir: Path) -> int:
    from itertools import count
    from unittest.mock import patch
    from uuid import UUID

    for relative, builder in ARTIFACT_BUILDERS.items():
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep the complete provenance graph while fixing this artifact's
        # run/snapshot/attempt/analysis identities and fixture timestamps. These are
        # not timing evidence. Production UUIDs/clocks remain unchanged;
        # numerical seeds belong to the individual scientific fixtures.
        identifiers = count(1)
        ticks = count(0)
        with (
            patch("nwqlib.blocks.encoding.uuid4", side_effect=lambda: UUID(int=next(identifiers))),
            patch("nwqlib.blocks.selection.uuid4", side_effect=lambda: UUID(int=next(identifiers))),
            patch(
                "nwqlib._prepared_execution.uuid4", side_effect=lambda: UUID(int=next(identifiers))
            ),
            patch("nwqlib.core.analysis.uuid4", side_effect=lambda: UUID(int=next(identifiers))),
            patch("nwqlib.execution.uuid4", side_effect=lambda: UUID(int=next(identifiers))),
            patch("nwqlib._prepared_execution._now", return_value="2026-01-01T00:00:00+00:00"),
            patch(
                "nwqlib._prepared_execution.perf_counter", side_effect=lambda: float(next(ticks))
            ),
        ):
            value = _roundtrip(builder())
        target.write_text(
            json.dumps(value, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        print(f"wrote {relative}")
    for name, data in _method_schemas().items():
        relative = f"methods/{name}.json"
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_roundtrip(data), indent=2, sort_keys=False), encoding="utf-8")
        print(f"wrote {relative}")
    return 0


def _diff(a: Any, b: Any, path: str, diffs: list[tuple[str, str]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            key_path = f"{path}.{key}"
            if key not in a:
                diffs.append((key_path, f"key only in B (value {b[key]!r})"))
            elif key not in b:
                diffs.append((key_path, f"key only in A (value {a[key]!r})"))
            else:
                _diff(a[key], b[key], key_path, diffs)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"list length {len(a)} != {len(b)}"))
            return
        for index, (item_a, item_b) in enumerate(zip(a, b)):
            _diff(item_a, item_b, f"{path}[{index}]", diffs)
        return
    if not _values_equal(a, b):
        diffs.append((path, f"{a!r} != {b!r}"))


def compare(dir_a: Path, dir_b: Path) -> int:
    for directory in (dir_a, dir_b):
        if not directory.is_dir():
            print(f"FAIL {directory}: snapshot root is not an existing directory")
            print("COMPARE: FAILED")
            return 1
    files_a = {str(path.relative_to(dir_a)) for path in dir_a.rglob("*.json")}
    files_b = {str(path.relative_to(dir_b)) for path in dir_b.rglob("*.json")}
    if not files_a or not files_b:
        print("FAIL: each snapshot must contain at least one JSON artifact")
        print("COMPARE: FAILED")
        return 1
    failed = files_a != files_b
    for name in sorted(files_a - files_b):
        print(f"FAIL {name}: artifact only in {dir_a}")
    for name in sorted(files_b - files_a):
        print(f"FAIL {name}: artifact only in {dir_b}")

    for name in sorted(files_a & files_b):
        a = json.loads((dir_a / name).read_text(encoding="utf-8"))
        b = json.loads((dir_b / name).read_text(encoding="utf-8"))
        diffs: list[tuple[str, str]] = []
        _diff(a, b, "$", diffs)
        if diffs:
            failed = True
            for path, message in diffs:
                print(f"FAIL {name}: {path}: {message}")
        else:
            print(f"OK   {name}")
    if failed:
        print("COMPARE: FAILED")
        return 1
    print("COMPARE: CLEAN")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", metavar="OUT_DIR")
    group.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    args = parser.parse_args()
    if args.generate:
        return generate(Path(args.generate))
    return compare(Path(args.compare[0]), Path(args.compare[1]))


if __name__ == "__main__":
    raise SystemExit(main())
