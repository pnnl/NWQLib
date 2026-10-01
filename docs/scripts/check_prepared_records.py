#!/usr/bin/env python3
"""Fresh-process SDK import-attempt audit of the real planning/observation path."""

import argparse

import builtins
import importlib.abc
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def expectation_plan(method=None):
    """Plan <psi|(I+Z)|psi>/<psi|psi> for the physical vector (2,2): 8/8 = 1.

    Imports stay inside this function so the audit guard sees them.
    """
    import nwqlib
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import Expectation

    problem = Expectation(
        state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)
    )
    return nwqlib.plan(problem, method=method or ExpectationMethod(), seed=7)


def child(poison=False):
    from docs.scripts.check_core_records import BLOCKED
    # Run imports its standard-library journal support even in memory.
    # This boundary excludes vendor SDKs, not sqlite3 or platform file locks.
    forbidden_prefixes = tuple(item for item in BLOCKED if item not in ("nwqlib.algorithms", "nwqlib.execution"))
    attempts = []
    def forbidden(name):
        # Permit configuration modules whose imports are inert. SDK imports
        # and native adapter activation remain blocked by the audit.
        inert = ("nwqlib.backends.capabilities", "nwqlib.backends.connection", "nwqlib.backends.nwqsim",
                 "nwqlib.backends.slurm")
        if name == "nwqlib.backends" or any(name == prefix or name.startswith(prefix + ".") for prefix in inert):
            return False
        return any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    def record(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("blocked SDK import: " + name)
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
    builtins.__import__ = guarded
    sys.meta_path.insert(0, Blocker())
    if poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass
    from nwqlib.subroutines.state_preparation import decompose_state_to_mps
    import numpy as np
    # A one-qubit compression/reconstruction has no SVD or native action.
    # The package boundary must keep this NumPy-only scientific consumer.
    compressed = decompose_state_to_mps([3., 4.])
    np.testing.assert_allclose(compressed.to_statevector(), [.6, .8], rtol=0, atol=1e-15)
    from nwqlib.core import Source
    from nwqlib.execution import ObservationChunk, ObservationView, PauliValue, RegisterMap, Run
    plan = expectation_plan()
    experiment = plan.experiments[0]
    realized = plan.resolve(experiment.name)
    with Run(plan) as run:
        assert run.checkpoint({"next_index": 0}) == 1
        assert run.checkpoint_state == {"next_index": 0}
        assert run.trace.jobs == run.trace.preparations == 0
    # Remote Run construction now creates a durable journal. This in-memory
    # audit checks only inert configuration interchange, without opening runs.
    from nwqlib.backends.nwqsim import NWQSimBackend
    configuration = NWQSimBackend(executable="/not-installed/nwqlib-nwqsim", spool="/not-created/spool",
        max_input_bytes=1024, max_output_bytes=1024, max_buffer_bytes=1024)
    assert NWQSimBackend.model_validate_json(configuration.model_dump_json()) == configuration
    from nwqlib.backends.slurm import NWQSimSlurmBackend, SlurmProfile
    slurm = NWQSimSlurmBackend(profile=SlurmProfile(cluster="explicit", account="supplied",
        walltime="00:05:00", nodes=1, ranks_per_node=1, threads=1,
        executable=configuration.executable, spool=configuration.spool, build_identity="supplied-source-revision"),
        accounting_since="2026-09-10T00:00:00", accounting_until="2026-09-11T00:00:00",
        max_input_bytes=1024, max_output_bytes=1024, max_buffer_bytes=1024,
        max_response_bytes=1024, timeout_seconds=2)
    assert NWQSimSlurmBackend.model_validate_json(slurm.model_dump_json()) == slurm
    from nwqlib.evidence.error_model import ErrorModel
    model = ErrorModel.model_validate_json(plan.error_model.model_dump_json())
    assert model == plan.error_model
    assert model.subject_id == plan.problem.content_id
    assert model.output_id == plan.output.content_id
    assert model.construction_id == plan.construction.content_id
    assert model.terms and all(term.fact.fact.availability == "unknown" for term in model.terms)
    from nwqlib.core.planning import Plan
    assert Plan.record_identity(plan.to_record()) == plan.content_id
    chunk = ObservationChunk(
        run_id="offline", plan_id=plan.content_id, realization_id=realized.content_id,
        prepared_id="sha256:" + "0" * 64, experiment=experiment.name, setting=experiment.setting, bindings=(),
        quantum_layout=(RegisterMap(name="system", bits=(0,)),), classical_layout=(),
        attempt="offline-example", job="offline-example", chunk="0", observation=experiment.observation,
        population="unconditional", returned_shots=None, trajectories=1,
        values=(PauliValue(label="Z", value=0.),),
        source=Source(name="supplied offline interchange example", version="1", domain="not an execution receipt",
                      reference="analytical one-qubit <I>=1, <Z>=0"),
    )
    view = ObservationView(chunks=(chunk, chunk))
    assert len(ObservationView.model_validate_json(view.model_dump_json()).chunks) == 1
    from nwqlib.execution import EstimateValue
    from nwqlib.algorithms.expectation import ExpectationMethod
    estimate_plan = expectation_plan(method=ExpectationMethod(estimate_precision=.1))
    assert Plan.record_identity(estimate_plan.to_record()) == estimate_plan.content_id
    estimate_spec = estimate_plan.experiments[0].observation
    estimate_chunk = chunk.revise(plan_id=estimate_plan.content_id,
        realization_id=estimate_plan.resolve(estimate_plan.experiments[0].name).content_id,
        experiment=estimate_plan.experiments[0].name, setting=estimate_plan.experiments[0].setting,
        observation=estimate_spec, trajectories=None, values=(EstimateValue(
            estimate_id=estimate_spec.estimate.content_id, value=2.25, standard_error=.125,
            uncertainty_unavailable=None, uncertainty_source=chunk.source,
            requested_options_json='{"resilience_level":0}', provider_metadata_json='{}'),))
    assert ObservationView.model_validate_json(ObservationView(chunks=(estimate_chunk,)).model_dump_json()).chunks == (estimate_chunk,)
    assert all(term.fact.fact.availability == "unknown" for term in estimate_plan.error_model.terms)
    assert plan.output.kind == "normalized_expectation"
    loaded = [name for name in sys.modules if forbidden(name)]
    if attempts or loaded:
        raise AssertionError(f"SDK isolation violation: attempts={attempts}, loaded={loaded}")
    print("SDK-free current planning, checkpoint, configuration/error metadata and observation interchange passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--child", action="store_true", help="run the isolated audit directly")
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; requires --child")
    args = parser.parse_args(argv)
    if args.poison and not args.child:
        parser.error("--poison requires --child")
    if args.child:
        child(args.poison)
        return
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))))
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    clean = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if clean.returncode:
        raise RuntimeError(clean.stdout + clean.stderr)
    print(clean.stdout, end="")
    poisoned = subprocess.run(command + ["--poison"], cwd=ROOT, env=env, text=True, capture_output=True)
    if poisoned.returncode == 0 or "attempts=['qiskit']" not in poisoned.stderr:
        raise AssertionError("caught-import control failed:\n" + poisoned.stderr)
    print("Caught blocked import control rejected as required.")


if __name__ == "__main__":
    main()
