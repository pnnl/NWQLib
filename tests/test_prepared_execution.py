"""Actual one-qubit preparation, native caches, noise and cumulative failures."""

import warnings

import numpy as np
import pytest

import nwqlib
from nwqlib._prepared_execution import PreparationNotRebuilt, Run, prepare_experiment, submit_experiment
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.backends import AerBackend
from nwqlib.backends import qiskit_aer as aer
from nwqlib.execution import ExecutionLimits
from test_expectation_current import problem
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend


def base_plan(*, shots=None):
    return nwqlib.plan(problem(state=[1.0, 1.0]), method=ExpectationMethod(), shots=shots, seed=7)


@pytest.mark.parametrize('boundary,grid_points', [('periodic', 16), ('dirichlet', 4), ('periodic', 6)])
def test_qubit_cap_precedes_readout_size_and_explains_binary_applicability(boundary, grid_points, tmp_path, monkeypatch):
    import sympy as sp
    import nwqlib._prepared_execution as owner
    from nwqlib.algorithms.qhd import QHD, UniformState
    from nwqlib.problems import Optimization

    x, y = sp.symbols('x y', real=True)
    problem = Optimization(objective=x*x + y*y, variables=(x, y), bounds=((-1., 1.), (-1., 1.)))
    method = QHD(boundary=boundary, num_grid_points=grid_points, num_steps=1,
                 total_time=.01, initial_state=UniformState())
    chosen = nwqlib.plan(problem, method=method, seed=7)
    cap = 20 if grid_points == 16 else 2
    limits = ExecutionLimits(max_simulation_qubits=cap)
    with Run(chosen, limits=limits, directory=tmp_path / 'refused') as run:
        with monkeypatch.context() as patch:
            patch.setattr(owner, 'readout_shape', lambda **kw: pytest.fail('readout size preceded qubit admission'))
            with pytest.raises(ValueError, match=f'max_simulation_qubits={cap}') as refused:
                method.prepare(chosen, run=run)
        message = str(refused.value)
        assert f'{2 * grid_points} qubits' in message and 'encoding="binary"' in message
        if boundary == 'periodic' and grid_points == 16:
            assert '8 qubits' in message
        else:
            assert 'changes' in message and 'grid' in message and 'boundary' in message
        assert run.trace.preparations == 0
    if boundary == 'periodic' and grid_points == 16:
        binary = nwqlib.plan(problem, method=method.revise(encoding='binary'), seed=7)
        prepared = nwqlib.prepare(binary, limits=limits, directory=tmp_path / 'binary', progress=False)
        with prepared.run:
            assert prepared.circuits and all(c.num_qubits == 8 for c in prepared.circuits)


def test_qubit_cap_does_not_offer_qhd_encoding_for_other_families(tmp_path):
    from nwqlib.operators import ingest_pauli

    chosen = nwqlib.plan(nwqlib.Expectation(state=[1, 0, 0, 0],
        observable=ingest_pauli((("ZZ", 1.),), num_qubits=2)), method=ExpectationMethod())
    with Run(chosen, limits=ExecutionLimits(max_simulation_qubits=1), directory=tmp_path / 'run') as run:
        with pytest.raises(ValueError, match='max_simulation_qubits=1') as refused:
            chosen.method.prepare(chosen, run=run)
        assert 'binary' not in str(refused.value)
        assert run.trace.preparations == 0


def test_native_cache_delta_and_idle_release_follow_actual_preparations(tmp_path, monkeypatch):
    import nwqlib._run_archive as archive
    from qiskit import qpy
    import gc
    import weakref

    count = 4  # preparations; a per-preparation visit count differs from a cumulative one

    visits, encodings = [], []
    entry, dump = archive._cache_entry, qpy.dump

    def visited(token, value, files, name):
        visits.append(token)
        return entry(token, value, files, name)

    def encoded(*args, **kwargs):
        encodings.append(1)
        return dump(*args, **kwargs)

    monkeypatch.setattr(archive, "_cache_entry", visited)
    monkeypatch.setattr(qpy, "dump", encoded)
    with Run(selected(), directory=tmp_path / "delta") as run:
        handles = [prepare_experiment(run.plan.resolve("counts"), run=run) for _ in range(count)]
        assert len(visits) == count and all(token[0] == "handle" for token in visits)
        assert len(encodings) == count  # Empty one-qubit measurement circuits, no definitions.
        before = len(visits), len(encodings)
        with monkeypatch.context() as patch:
            patch.setattr(run._state["handles"], "items", lambda: pytest.fail("checkpoint visited native history"))
            for cache in run._state["definitions"]:
                patch.setattr(cache, "items", lambda: pytest.fail("checkpoint visited definition history"))
            run.checkpoint({"unchanged": True})
        assert (len(visits), len(encodings)) == before
        refused = run.release_native()
        assert not refused["released"] and len(refused["kept"]) == count
        # Real tiny Aer acquisitions make the first two handles fully idle.
        for handle in handles[:2]:
            run.collect(submit_experiment(handle, run=run))
        stored = tuple(handles)
        circuits = tuple(handle.inspect_circuit() for handle in stored)
        native_refs = tuple(weakref.ref(handle._native) for handle in handles[:2])
        release = run.release_native()
        assert release["released"] == tuple(handle.record.content_id for handle in handles[:2])
        assert tuple(run._state["handles"].values()) == stored
        assert all(handle._native is None for handle in handles[:2])
        gc.collect()
        assert all(reference() is None for reference in native_refs)
        with monkeypatch.context() as patch:
            patch.setattr(aer, "_prepare_aer_execution", lambda *a, **k: pytest.fail("released handle rebuilt"))
            patch.setattr(aer, "_submit_aer_execution", lambda *a, **k: pytest.fail("inspection acquired"))
            inspected = tuple(handle.inspect_circuit() for handle in stored)
        assert inspected == circuits
        inspected[0].x(0)
        assert stored[0].inspect_circuit() == circuits[0]
        assert run.trace.preparations == count and sum(event.shots for event in run.trace.events) == 14
        assert (len(visits), len(encodings)) == before


def test_native_cache_file_failure_restores_prior_pointer_and_charge(tmp_path, monkeypatch):
    from qiskit import qpy

    with Run(selected(), directory=tmp_path / "rollback") as run:
        first = prepare_experiment(run.plan.resolve("counts"), run=run)
        previous_handles = tuple(run._state["handles"].values())
        files = run._state["archive_files"]
        before = run._state["external_bytes"], set(files._written_files)
        dump = qpy.dump

        def interrupted(*args, **kwargs):
            dump(*args, **kwargs)
            raise OSError("interrupted new QPY")

        monkeypatch.setattr(qpy, "dump", interrupted)
        with pytest.raises(OSError, match="new QPY"):
            prepare_experiment(run.plan.resolve("counts"), run=run)
        assert tuple(run._state["handles"].values()) == previous_handles == (first,)
        assert (run._state["external_bytes"], files._written_files) == before
        assert len(run.prepared_artifacts) == 1 and run.trace.preparations == 2
        assert len(tuple(run.directory.glob("*.qpy"))) == 1


def test_actual_aer_cache_prune_deletes_only_its_row_after_commit(tmp_path, monkeypatch):
    import sqlite3
    from nwqlib._run_archive import write_caches

    with Run(base_plan(), directory=tmp_path / "prune") as run:
        prepare_experiment(run.plan.resolve("expectation"), run=run)
        owner = next(iter(run._state["backend_context"]["aer_preparations"].values()))
        assert owner._blocks
        original = dict(owner._blocks)
        deleted_keys = {run._state["cache_keys"][("aer", target, key)]
                        for target, item in run._state["backend_context"]["aer_preparations"].items()
                        for key in item._blocks}
        journal = run._state["journal"]
        before = run._state["data_bytes"], journal.usage["data_bytes"]
        journal.connection.execute("CREATE TRIGGER refuse_prune BEFORE DELETE ON records "
                                   "BEGIN SELECT RAISE(ABORT, 'interrupt prune'); END")
        owner.keep_sources(())  # The real Aer owner records each expired source.
        with pytest.raises(sqlite3.IntegrityError, match="interrupt prune"):
            write_caches(run)
        assert dict(owner._blocks) == original
        assert (run._state["data_bytes"], journal.usage["data_bytes"]) == before
        journal.connection.execute("DROP TRIGGER refuse_prune")
        # The caller must reopen after a persistence error, keeping the exact
        # prior cache. Continue that original source below, without reacquisition.
        path = run.directory
    with nwqlib.load_run(path, backend=AerBackend()) as run:
        owner = next(iter(run._state["backend_context"]["aer_preparations"].values()))
        owner.keep_sources(())
        write_caches(run)
        current = {key for key, _, _ in run._state["journal"].rows("cache")}
        assert not current & deleted_keys and not owner._blocks
        assert not any(("cache", key) in run._state["data_sizes"] for key in deleted_keys)
        connection = run._state["journal"].connection
        expected = connection.execute("SELECT sum(length(CAST(payload AS BLOB))) FROM records").fetchone()[0]
        expected += connection.execute("SELECT coalesce(sum(length(payload)),0) FROM binary").fetchone()[0]
        assert run._state["journal"].usage["data_bytes"] == expected
        assert run.trace.preparations == 1 and not run.trace.events


@pytest.mark.parametrize("observer_fails", [False, True])
def test_progress_uses_published_work_and_never_repeats_acquisition(tmp_path, observer_fails):
    from nwqlib.core.planning import RandomStreams

    InjectedBackend.supports_synchronous = True
    chosen = selected()
    expected_rng = RandomStreams.restore(chosen.randomness)
    expected_rng.next_seed()  # One actual selected acquisition, including its preparation.
    notices = []

    def observe(stage, done, total):
        notices.append((stage, done, total, tuple(name for name, _ in InjectedBackend.calls)))
        if observer_fails and stage == "collect":
            raise RuntimeError("observer unavailable")

    prepared = nwqlib.prepare(chosen, backend=InjectedBackend(), directory=tmp_path / "progress",
                              progress=observe)
    with nwqlib.submit(prepared) as run:
        result = run.wait()
        assert result.counts == 7 and run.rng.snapshot() == expected_rng.snapshot()
        assert [name for name, _ in InjectedBackend.calls].count("prepare") == 1
        assert [name for name, _ in InjectedBackend.calls].count("launch") == 1
        assert notices[0][:3] == ("input", 1, 1)
        before, after = [row for row in notices if row[0] == "prepare"]
        assert before[1:] == (None, None, ())
        assert after[1:] == (1, None, ("prepare",))
        (collected,) = [row for row in notices if row[0] == "collect"]
        assert collected[1:3] == (len(result.data.observations.chunks), None)
        assert collected[3].count("launch") == 1
        if observer_fails:
            assert len(run.warnings) == 1 and "observer unavailable" in run.warnings[0]
            assert run._state["progress"] is False
        else:
            assert [row[:3] for row in notices[-2:]] == [("analyze", 0, 1), ("analyze", 1, 1)]
            assert not run.warnings


def test_default_progress_updates_one_live_kernel_display_and_scripts_stay_quiet(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import IPython
    import IPython.display

    displays, updates = [], []

    class Handle:
        def update(self, value, *, raw):
            assert raw is True
            updates.append(value["text/plain"])

    def display(value, *, raw, display_id):
        assert raw is True and display_id is True
        displays.append(value["text/plain"])
        return Handle()

    monkeypatch.setattr(IPython.display, "display", display)
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    InjectedBackend.supports_synchronous = True
    prepared = nwqlib.prepare(selected(), backend=InjectedBackend(), directory=tmp_path / "kernel")
    with nwqlib.submit(prepared) as run:
        assert run.wait().counts == 7
    assert displays == ["Input accepted"]
    assert "Prepared artifacts: 1" in updates and "Collected observations: 1" in updates
    assert updates[-1] == "Analysis finished"
    displays.clear()
    updates.clear()
    # Terminal IPython is also a script environment for this purpose.
    for index, shell in enumerate((None, SimpleNamespace())):
        monkeypatch.setattr(IPython, "get_ipython", lambda: shell)
        with Run(selected(), backend=InjectedBackend(), directory=tmp_path / f"quiet-{index}") as run:
            run.progress("input", 1, 1)
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    with Run(selected(), backend=InjectedBackend(), progress=False, directory=tmp_path / "disabled") as run:
        run.progress("input", 1, 1)
    assert not displays and not updates


def test_prepared_resource_receiver_inspects_only_the_selected_live_native(monkeypatch):
    from qiskit import QuantumCircuit
    from nwqlib._prepared_execution import Prepared

    run = Run(base_plan())
    first = prepare_experiment(run.plan.resolve("expectation"), run=run)
    second = prepare_experiment(run.plan.resolve("expectation"), run=run)
    prepared = Prepared(run)
    monkeypatch.setattr(QuantumCircuit, "copy", lambda *a, **k: pytest.fail("raw inventory copied native data"))
    import copy
    monkeypatch.setattr(copy, "deepcopy", lambda *a, **k: pytest.fail("raw inventory deep-copied native data"))
    import qiskit
    from qiskit_aer import AerSimulator
    monkeypatch.setattr(qiskit, "transpile", lambda *a, **k: pytest.fail("raw inventory compiled"))
    monkeypatch.setattr(AerSimulator, "run", lambda *a, **k: pytest.fail("raw inventory simulated"))
    inventory = prepared.inspect_resources(index=1)
    # State (1, 1)/sqrt(2) needs one H; the exact expectation is one Aer save.
    assert inventory["operations"] == {"h": 1, "save_expval": 1}
    assert inventory["total_operations"] == 2 and inventory["depth"] == 1
    assert inventory["circuit"] == (
        f"prepared circuit {second.record.content_id}; not executed by inspection")
    assert first.record.content_id != second.record.content_id
    assert run.trace.jobs == 0
    with pytest.raises(ValueError, match="no prepared quantum"):
        prepared.inspect_resources(index=2)
    run.close()
    with pytest.raises(ValueError, match="closed"):
        prepared.inspect_resources()


def same_native(actual, original):
    if isinstance(original, np.ndarray):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        np.testing.assert_array_equal(actual, original)
    elif isinstance(original, dict):
        assert actual.keys() == original.keys()
        for key in original:
            same_native(actual[key], original[key])
    elif isinstance(original, (tuple, list)):
        assert len(actual) == len(original)
        for a, b in zip(actual, original, strict=True):
            same_native(a, b)
    else:
        assert actual == original


@pytest.mark.parametrize("preparations", [0, 2])
def test_noise_input_and_cached_native_preparations_reopen_without_serializing_again(
    tmp_path, monkeypatch, preparations
):
    """Count noise-model serialization and poison repeated QPY writes after reopening the
    original preparation cache.
    """
    from qiskit import qpy
    from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error

    model = NoiseModel()
    model.add_all_qubit_readout_error(ReadoutError([[0.9, 0.1], [0.2, 0.8]]))
    model.add_all_qubit_quantum_error(amplitude_damping_error(0.25), ["ry"])
    expected = model.to_dict()
    encoded, decoded = [], []
    import nwqlib._run_archive as archive
    encode, decode = NoiseModel.to_dict, archive._load_noise_model

    def write(self, serializable=False):
        encoded.append(self)
        return encode(self, serializable=serializable)

    def read(value):
        decoded.append(value)
        return decode(value)

    monkeypatch.setattr(NoiseModel, "to_dict", write)
    monkeypatch.setattr(archive, "_load_noise_model", read)
    monkeypatch.setattr(NoiseModel, "from_dict", lambda *a: pytest.fail("deprecated noise loader"))
    backend = AerBackend.from_noise_model(model)
    metadata = AerBackend.model_validate_json(backend.model_dump_json())
    assert not encoded and not decoded and metadata.noise_model_id == backend.noise_model_id
    assert (
        backend.model_copy()._bound_noise_model() is model and backend.revise()._bound_noise_model() is model
    )
    path = tmp_path / "noise"
    with Run(base_plan(shots=8), backend=backend, directory=path) as run:
        for _ in range(preparations):
            handle = prepare_experiment(run.plan.resolve("group_0"), run=run)
            assert handle.record.backend_configuration_id == backend.content_id
            assert "explicit noise model" in handle.record.target.domain
        run.checkpoint({"prepared": preparations})
        assert run.trace.jobs == 0 and len(encoded) == 1 and not decoded
    with nwqlib.load_run(path, backend=metadata) as run:
        assert len(decoded) == 1
        # The archive keeps the basis gates beside to_dict (_run_archive._noise_model_data).
        same_native(decoded[0], {**expected, "basis_gates": sorted(model.basis_gates)})
        same_native(encode(run.backend._bound_noise_model()), expected)
        with monkeypatch.context() as patch:
            patch.setattr(
                qpy, "dump", lambda *a, **k: pytest.fail("unchanged loaded native cache serialized again")
            )
            run.checkpoint({"loaded": True})
        assert len(encoded) == len(decoded) == 1 and run.trace.jobs == 0
        assert len(tuple(path.glob("*noise-model.json"))) == 1
    # load_run bound the rebuilt model to the reopened Run's own copy of the
    # backend, so the configuration passed to it is still unbound.
    with Run(base_plan(shots=8), backend=metadata) as run:
        with pytest.raises(ValueError, match="noise model is unbound"):
            prepare_experiment(run.plan.resolve("group_0"), run=run)
        assert not run.trace.jobs
    with Run(base_plan(), backend=backend) as run:
        with pytest.raises(ValueError, match="exact readouts require a noiseless"):
            prepare_experiment(run.plan.resolve("expectation"), run=run)
        assert not run.trace.preparations
    with Run(base_plan()) as run:
        handle = prepare_experiment(run.plan.resolve("expectation"), run=run)
        assert "explicit noise model" not in handle.record.target.domain


def test_backend_notice_is_once_per_saved_run_and_warning_error_precedes_work(tmp_path, monkeypatch):
    monkeypatch.setattr(InjectedBackend, "qualification_notice", "supplied offline limitation", raising=False)
    path = tmp_path / "notice"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Run(selected(), backend=InjectedBackend(), directory=path) as run:
            assert not caught
            prepare_experiment(run.plan.resolve("counts"), run=run)
            prepare_experiment(run.plan.resolve("counts"), run=run)
        with nwqlib.load_run(path, backend=InjectedBackend(), method=CountsMethod) as run:
            prepare_experiment(run.plan.resolve("counts"), run=run)
        assert [str(item.message) for item in caught] == ["supplied offline limitation"]
    InjectedBackend.calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with Run(selected(), backend=InjectedBackend(), directory=tmp_path / "refused") as run:
            with pytest.raises(UserWarning, match="supplied offline"):
                prepare_experiment(run.plan.resolve("counts"), run=run)
            assert not run.trace.preparations and not run.trace.events and not InjectedBackend.calls


def test_prepared_reuse_keeps_same_native_object_and_failed_actual_exposure(monkeypatch, tmp_path):
    InjectedBackend.supports_synchronous = True
    run = Run(
        selected(),
        backend=InjectedBackend(),
        limits=ExecutionLimits(max_total_shots=21),
        directory=tmp_path / "reuse",
    )
    handle = prepare_experiment(run.plan.resolve("counts"), run=run)
    native = handle._native
    first = submit_experiment(handle, run=run)
    second = submit_experiment(handle, run=run)
    assert first.histogram().index_list() == second.histogram().index_list()
    assert first.histogram().weights.tolist() == second.histogram().weights.tolist()
    assert first.acquisition_key != second.acquisition_key
    assert handle._native is native and run.trace.preparations == 1
    InjectedBackend.fail = True
    with pytest.raises(RuntimeError, match="native read failed"):
        submit_experiment(handle, run=run)
    assert run.exposure["completed"]["shots"] == 14 and run.exposure["uncertain"]["shots"] == 7
    assert run.trace.submissions[-1].status == "uncertain"
    with pytest.raises(ValueError, match="max_total_shots"):
        submit_experiment(handle, run=run)
    assert sum(name == "launch" for name, _ in InjectedBackend.calls) == 3
    assert sum(name == "prepare" for name, _ in InjectedBackend.calls) == 1
    run.close()


def test_actual_compiler_configuration_and_receipt_cannot_be_swapped(monkeypatch):
    run = Run(base_plan())
    handle = prepare_experiment(run.plan.resolve("expectation"), run=run)
    assert set(handle.record.native_basis) == set(handle._native.simulator.target.operation_names)
    # G counts the native evolution operations of the coherent readout, not its save instruction.
    assert handle.record.native_operations == sum(not item.operation.name.startswith("save_")
                                                  for item in handle._native.circuit.data)
    changed = AerBackend().revise(noise_model_id="different actual configuration")
    other = Run(run.plan, backend=changed)
    monkeypatch.setattr(
        aer, "_submit_aer_execution", lambda *a: pytest.fail("foreign compiler reached execution")
    )
    with pytest.raises(ValueError, match="backend"):
        submit_experiment(handle, run=other)
    assert other.trace.jobs == 0


@pytest.mark.parametrize("channel", ["readout", "sx_damping"])
def test_bound_noise_applies_after_translation_to_effective_basis(channel):
    from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error

    model = NoiseModel()
    if channel == "readout":
        model.add_all_qubit_readout_error(ReadoutError([[0.0, 1.0], [0.0, 1.0]]))
    else:
        model.add_all_qubit_quantum_error(amplitude_damping_error(1.0), ["sx"])
    plan = nwqlib.plan(problem((("Z", 1.0),), state=[1.0, 1.0]), method=ExpectationMethod(), shots=16, seed=7)
    result = nwqlib.solve(plan, backend=AerBackend.from_noise_model(model))
    # Readout maps both bits to one. Damping after the final translated SX
    # resets to zero; attaching noise only to the original preparation's name
    # would miss that channel entirely and leave the mean near zero.
    assert result.value == (-1.0 if channel == "readout" else 1.0)
    assert result.data.observations.chunks[0].returned_shots == 16
    (receipt,) = result.data.receipts
    assert "noise-target basis translation" in receipt.compiler.name
    assert "u" not in receipt.native_basis
    exact = nwqlib.solve(problem((("Z", 1.0),), state=[1.0, 1.0]), method=ExpectationMethod())
    assert exact.value == pytest.approx(0.0, abs=1e-13, rel=0)


def test_close_releases_private_caches_but_keeps_result_data_and_spent_exposure():
    prepared = nwqlib.prepare(base_plan())
    run = prepared.run
    (native,) = prepared.circuits
    native.clear()  # Inspection owns a copy, not the actual submitted circuit.
    result = nwqlib.submit(prepared).wait()
    assert result.value == pytest.approx(1.0, abs=1e-13, rel=0)
    before = run.trace
    run.close()
    assert run.trace == before and run.result is result and result.data.observations.chunks
    assert result.analyze().value == result.value
    assert run._state["method_context"] is None and run._state["handles"] == {}
    assert run._state["definitions"] == ({}, {}) and run._state["backend_context"] == {}
    with pytest.raises(ValueError, match="closed"):
        prepared.circuits


def test_direct_preparation_limit_is_a_run_limit_checked_before_any_preparation(monkeypatch):
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems import inputs
    from nwqlib.resources.records import ResourceContext

    # A four-amplitude direct PREP of magnitude/phase form, above a lowered
    # Run limit of two amplitudes.
    plan = nwqlib.plan(Expectation(state=[0.6, 0.0, 0.0, 0.8], observable=ingest_pauli((("ZZ", 1.0),), num_qubits=2)),
                       method=ExpectationMethod(), seed=7)
    # Planning and the circuit-free estimate accept it, because the direct CX
    # law needs no synthesis. Each component costs 2*(2**2-2) = 4 CX slots.
    report = nwqlib.estimate(plan, context=ResourceContext(basis="cx"))
    components = report.quantity("preparation_components").fact.value
    cx = report.quantity("cx")
    assert components.denominator == cx.fact.value.denominator == 1 and components.numerator >= 1
    assert cx.interpretation == "upper_bound" and cx.fact.value.numerator == 4 * components.numerator

    calls = []
    original = inputs.prepare_qiskit

    def recorded(state, **kwargs):
        calls.append(kwargs.get("max_direct_amplitudes"))
        return original(state, **kwargs)

    monkeypatch.setattr(inputs, "prepare_qiskit", recorded)
    run = Run(plan, limits=ExecutionLimits(max_direct_amplitudes=2))
    with pytest.raises(ValueError, match="4 amplitudes, exceeding the Run's max_direct_amplitudes=2"):
        plan.method.prepare(plan, run=run)
    assert run.trace.preparations == 0 and calls == []
    # A deliberate increase lets the same Run proceed, and the Run's limit
    # reaches the synthesis instead of the prepare_qiskit default.
    run.extend_limits(max_direct_amplitudes=4)
    plan.method.prepare(plan, run=run)
    assert run.trace.preparations == 1 and calls == [4] and len(run.limit_amendments) == 1
    run.close()


def test_direct_amplitude_limit_covers_transforms_adapt_queries_and_standalone_lowering(monkeypatch):
    import sys
    from test_adapt_primary import plan_for
    from nwqlib.blocks.lowering import lower_qiskit
    from nwqlib.blocks.selection import direct_preparation_amplitudes, select_preparation, transform_block
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems.inputs import state_input

    # A control or adjoint synthesizes through its base PREP constructor.
    base = select_preparation("prep", state_input([0.6, 0.0, 0.0, 0.8]))
    both = transform_block("both", base, control=True, adjoint=True)
    assert direct_preparation_amplitudes(base) == direct_preparation_amplitudes(both) == 4
    # ADAPT query blocks declare their two-amplitude direct reference.
    queries = [block for block in plan_for().blocks if block.record.semantics.kind == "unknown"]
    assert queries and {direct_preparation_amplitudes(block) for block in queries} == {2}
    # Standalone lowering refuses the same size before importing the SDK.
    plan = nwqlib.plan(Expectation(state=[0.6, 0.0, 0.0, 0.8], observable=ingest_pauli((("ZZ", 1.0),), num_qubits=2)),
                       method=ExpectationMethod(), seed=7)
    monkeypatch.setitem(sys.modules, "qiskit", None)
    with pytest.raises(ValueError, match="4 amplitudes, exceeding max_direct_amplitudes=2"):
        lower_qiskit(plan.construction, blocks=plan.blocks, max_direct_amplitudes=2)


def test_lchs_vector_prep_uses_the_direct_tree_and_the_run_limit():
    from nwqlib import LinearDynamics
    from nwqlib.algorithms.lchs import LCHS, ProviderConfig
    from nwqlib.blocks.selection import direct_preparation_amplitudes

    method = LCHS(k_quadrature=ProviderConfig(
        implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": -1}))
    plan = nwqlib.plan(LinearDynamics(A=np.diag([.2, .4]), initial_state=[.6, .8], time=.1),
                       method=method, execution="quantum")
    # Both LCHS vector PREP leaves declare qiskit.direct, the direct
    # magnitude/phase tree, so they count at their padded amplitude counts.
    counts = {block.record.signature.name: direct_preparation_amplitudes(block) for block in plan.blocks}
    assert counts["initial_prep"] == 2 and counts["coefficient_prep"] == counts["coefficient_inverse"] >= 2
    limit = counts["coefficient_prep"] - 1
    run = Run(plan, limits=ExecutionLimits(max_direct_amplitudes=limit))
    with pytest.raises(ValueError, match=f"exceeding the Run's max_direct_amplitudes={limit}"):
        plan.method.prepare(plan, run=run)
    assert run.trace.preparations == 0
    run.close()


# Two real four-qubit Pauli Hamiltonians with different supports and signs.
ALL_SETTINGS_HAMILTONIANS = {
    "ring": (("IIII", .1), ("ZZII", .3), ("IXXI", -.2), ("IIZZ", .25), ("XIIX", .15)),
    "mixed": (("ZIII", .4), ("XYZI", -.3), ("IYYX", .2), ("ZZZZ", .05)),
}


def settings_values(result):
    """Experiment names and returned outcomes, without Run, attempt or job identities."""
    return [(chunk.experiment, chunk.histogram().index_list(), chunk.histogram().weights.tolist())
            for chunk in result.data.observations.chunks]


@pytest.mark.parametrize("hamiltonian,durable", [("ring", False), ("mixed", True)])
def test_prepare_all_settings_exposes_every_circuit_and_submit_reuses_them(
    hamiltonian, durable, tmp_path, monkeypatch
):
    from nwqlib import Eigenproblem
    from nwqlib.algorithms.qpe import QCELS
    from nwqlib.blocks import lowering
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation
    from nwqlib.resources.records import ResourceContext

    target = ingest_pauli(ALL_SETTINGS_HAMILTONIANS[hamiltonian], num_qubits=4)
    method = QCELS(initial_state=ingest_occupation("0101", num_qubits=4), num_times=3, tau=.2, grid_size=64)
    selected = nwqlib.plan(Eigenproblem(A=target), method=method, shots=16, seed=7)
    names = tuple(experiment.name for experiment in selected.experiments)
    assert len(names) > 1
    # The durable case names the local backend explicitly, the other uses the default.
    directory, backend = (tmp_path / "all", AerBackend()) if durable else (None, None)
    prepared = nwqlib.prepare(selected, backend=backend, settings="all", directory=directory)
    run = prepared.run
    assert prepared.setting_names == names and len(prepared.circuits) == len(names)
    # Prepared.circuit(i) copies circuit i of the same index order, and changing
    # the copy leaves the prepared circuit unchanged.
    assert all(prepared.circuit(index) == circuit for index, circuit in enumerate(prepared.circuits))
    prepared.circuit(0).clear()
    assert prepared.circuit(0) == prepared.circuits[0] and len(prepared.circuit(0).data)
    # Nothing is submitted by preparation.
    assert run.trace.jobs == 0 and run.trace.events == ()
    # The fold counts each controlled Suzuki step by its exact CX law
    # (qpe/powers.py::suzuki_step_cx) times its repetitions and the shots of
    # its setting. The occupation preparation, H, the identity phase and the
    # feedback rotation add no CX, so transpiling every prepared circuit
    # without optimization and weighting it by its setting's shots must give
    # the same total.
    options = dict(basis_gates=["cx", "u"], optimization_level=0)
    compiled = [prepared.inspect_resources(index=index, transpile_options=options)["operations"].get("cx", 0)
                * selected.resolve(name).resolved_observation(selected)[1].shots
                for index, name in enumerate(prepared.setting_names)]
    folded = nwqlib.estimate(selected, context=ResourceContext(basis="cx")).quantity("cx")
    assert folded.interpretation == "exact" and folded.fact.value.denominator == 1
    assert sum(compiled) == folded.fact.value.numerator > 0
    prepared_trace = run.trace
    assert prepared_trace.preparations == len(names)

    def forbidden(*args, **kwargs):
        raise AssertionError("submission lowered a setting that was already prepared")

    monkeypatch.setattr(lowering, "_lower_qiskit", forbidden)
    if durable:
        # Stop before submission, then continue the saved Run.
        run.close()
        run = nwqlib.load_run(directory, backend=AerBackend())
        result = run.wait()
    else:
        result = nwqlib.submit(prepared).wait()
    with run:
        trace = run.trace
        assert trace.preparations == len(names) and trace.jobs == len(names)
        assert trace.local_prepared_ids == prepared_trace.local_prepared_ids
        assert trace.construction_work_reserved == prepared_trace.construction_work_reserved
        assert trace.synthesis_work_reserved == prepared_trace.synthesis_work_reserved
    monkeypatch.undo()
    # The default path creates each setting's workflow row, and draws its
    # seed, in the same Plan order, so it acquires the same histograms.
    default = nwqlib.solve(selected)
    assert ([receipt.runtime.seed for receipt in result.data.receipts]
            == [receipt.runtime.seed for receipt in default.data.receipts])
    assert settings_values(result) == settings_values(default)


@pytest.mark.parametrize("family", ["rwpe", "lanczos", "adapt"])
def test_prepare_all_settings_refuses_outcome_dependent_controllers(family, tmp_path):
    from math import pi
    from nwqlib import Eigenproblem
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.algorithms.lanczos import Lanczos, SensitivitySampling
    from nwqlib.algorithms.qpe import RWPE
    from nwqlib.operators import ingest_pauli

    problem, method = {
        "rwpe": (Eigenproblem(A=[[.4, 0.], [0., -.7]]), RWPE(initial_state=[1., 0.], max_steps=3, tau=.2)),
        "lanczos": (Eigenproblem(A=[[1., 1.], [1., 1.]]), Lanczos(
            initial_state=[1., 0.], krylov_dimension=2,
            sampling=SensitivitySampling(total_shots=100, pilot_fraction=.2))),
        "adapt": (Eigenproblem(A=ingest_pauli((("Z", 1.),), num_qubits=1)), ADAPT(
            initial_state=(1, 1), pool=(ingest_pauli((("Y", 1j),), num_qubits=1),), theta=pi / 8,
            max_iterations=2)),
    }[family]
    selected = nwqlib.plan(problem, method=method, shots=1 if family == "rwpe" else None, seed=7)
    # The refusal comes before the Run (Method.prepare_all_refusal), so the
    # folder stays free for the default preparation.
    with pytest.raises(ValueError, match="cannot prepare every setting in advance") as refused:
        nwqlib.prepare(selected, settings="all", directory=tmp_path / "run")
    assert "settings='first' prepares" in str(refused.value)
    assert not (tmp_path / "run").exists()
    nwqlib.prepare(selected, directory=tmp_path / "run").run.close()


def test_prepare_all_settings_refuses_a_remote_backend_before_any_run(tmp_path):
    from nwqlib.backends.ionq import IonQBackend

    backend = IonQBackend(device="qpu.forte-1", max_input_bytes=65536, max_response_bytes=65536)
    with pytest.raises(ValueError, match="needs local Aer.*'ionq' is none of them"):
        nwqlib.prepare(base_plan(shots=8), backend=backend, settings="all", directory=tmp_path / "remote")
    assert not (tmp_path / "remote").exists()


def test_prepare_all_settings_failure_at_a_later_setting_submits_nothing_on_resume(tmp_path, monkeypatch):
    from nwqlib.blocks import lowering
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation

    # ZZ, XX and YY need three measurement bases, so the Plan has three settings.
    selected = nwqlib.plan(
        Expectation(state=[1., 0., 0., 1.],
                    observable=ingest_pauli((("ZZ", 1.), ("XX", .5), ("YY", .25)), num_qubits=2)),
        method=ExpectationMethod(), shots=8, seed=7)
    assert len(selected.experiments) == 3
    original, calls = lowering._lower_qiskit, []

    def fails_second(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("lowering stopped")
        return original(*args, **kwargs)

    monkeypatch.setattr(lowering, "_lower_qiskit", fails_second)
    with pytest.raises(RuntimeError, match="lowering stopped"):
        nwqlib.prepare(selected, settings="all", directory=tmp_path / "run")
    monkeypatch.undo()
    # The second setting keeps its charge without a receipt, and the Run never
    # repeats a preparation attempt, so it cannot complete. Resuming refuses
    # before it spends shots on the first setting.
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend()) as run:
        assert run.trace.preparations == 2 and len(run.prepared_artifacts) == 1
        with pytest.raises(PreparationNotRebuilt, match="'group_1' failed or was interrupted and no replacement"):
            run.resume()
        assert run.trace.jobs == 0 and run.trace.preparations == 2


class PrepareWithoutSettings(ExpectationMethod):
    """An extension Method whose prepare override predates the settings keyword."""

    def prepare(self, plan, *, run):
        return super().prepare(plan, run=run)


def test_prepare_hook_without_the_settings_keyword_serves_the_default_and_refuses_all(tmp_path):
    # The hook receives settings only for an explicit "all", which the
    # inherited prepare_all_refusal refuses before any Run (docs/algorithm_protocol.md).
    selected = nwqlib.plan(problem(state=[1.0, 1.0]), method=PrepareWithoutSettings(), shots=8, seed=7)
    prepared = nwqlib.prepare(selected)
    with prepared.run:
        assert prepared.setting_names == (selected.experiments[0].name,)
    with pytest.raises(ValueError, match="PrepareWithoutSettings.prepare does not accept the settings keyword"):
        nwqlib.prepare(selected, settings="all", directory=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_template_hook_reaches_only_a_declaring_adapter_and_keeps_row_seeds():
    import dataclasses
    from typing import ClassVar
    import numpy as np
    import nwqlib
    from nwqlib._prepared_execution import Run, _template_seed
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib.operators import ingest_pauli

    received = []

    class TemplateAer(AerBackend):
        prepares_from_templates: ClassVar[bool] = True

        def prepare(self, circuit, *, construction_id, backend_context, template_seed, **arguments):
            received.append((construction_id, template_seed, backend_context is run._state["backend_context"]))
            return dataclasses.replace(super().prepare(circuit, **arguments), template_seed=template_seed)

    class ForgedSeedAer(AerBackend):
        def prepare(self, circuit, **arguments):
            return dataclasses.replace(super().prepare(circuit, **arguments), template_seed=1)

    state = np.array([1.0, 1j, 0.5, -0.5]) / np.linalg.norm([1.0, 1j, 0.5, -0.5])
    problem = nwqlib.Expectation(state=state, observable=ingest_pauli((("XY", 1.0), ("ZZ", 0.5)), num_qubits=2))
    plan = nwqlib.plan(problem, method=ExpectationMethod(), shots=64, seed=7)
    outcomes = {}
    for backend in (AerBackend(), TemplateAer()):
        with Run(plan, backend=backend) as run:
            run.resume()
            outcomes[type(backend).__name__] = (run.result.value, run.prepared_artifacts)
    (plain_value, plain), (value, receipts) = outcomes["AerBackend"], outcomes["TemplateAer"]
    # The seed derives from the Plan's randomness and the construction, not from
    # the row-seed stream, so row seeds and the sampled value are unchanged.
    assert value == plain_value and [r.runtime.seed for r in receipts] == [r.runtime.seed for r in plain]
    assert all(r.template_seed is None for r in plain)
    expected = [_template_seed(plan.randomness, r.construction_id) for r in receipts]
    assert [r.template_seed for r in receipts] == expected
    assert received == [(r.construction_id, seed, True) for r, seed in zip(receipts, expected)]
    with Run(plan, backend=ForgedSeedAer()) as run, pytest.raises(ValueError, match="template seed"):
        run.resume()


def test_static_path_funds_reductions_from_the_method_before_native_work(tmp_path, monkeypatch):
    """The inherited prepare and execute take a reduction's allowance from ``Method.reduction_allowance``.

    A test-local reducer registers ``2 << width`` work units on a static LCHS
    Plan whose one experiment is a trajectory with a reduction point. One
    unit less than that is refused before the preparation charge and, on a
    reopened Run, before submission; the full allowance prepares, and the
    reopened Run acquires once and reduces once.
    """
    from nwqlib import LinearDynamics
    from nwqlib.algorithms.lchs import LCHS, ProviderConfig
    from nwqlib.core import planning
    from nwqlib.core.planning import Experiment, ObservationPoint, ObservationSpec, Reducer, ReducerOutput

    reduced, asked, shortfall = [], [], [1]

    def execute(state, parameters, bindings):
        reduced.append(state.shape)
        return (np.array([np.vdot(state, state).real]),)

    def reduction_allowance(self, plan, point, *, observation, width, run):
        asked.append((point.experiment, width, observation.positions[0].id))
        return (2 << width) - shortfall[0]

    monkeypatch.setitem(planning.READOUT_REDUCERS, "norm", Reducer(
        shape=lambda parameters: (ReducerOutput("float64", (1,)),), work=lambda parameters, width: 2 << width,
        execute=execute))
    monkeypatch.setattr(LCHS, "reduction_allowance", reduction_allowance, raising=False)
    monkeypatch.setattr(LCHS, "analyze", lambda self, plan, data, *, settings: data)
    method = LCHS(k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
                                              parameters={"num_qubits": 2, "lsb_position": -1}))
    base = nwqlib.plan(LinearDynamics(A=np.diag([.2, .4]), initial_state=[1., 0.], time=.1), method=method,
                       execution="quantum", seed=7)
    schedule = ObservationSpec(kind="trajectory", positions=(ObservationPoint(id="end", kind="reduction",
                                                                             reducer="norm"),))
    chosen = base.revise(experiments=(Experiment(name="setting_0", setting="setting_0", observation=schedule),))
    chosen = chosen._bind(blocks=base.blocks, **base._native)
    refused = r"reductions through point 'end' register (\d+) work units, more than the remaining allowance"
    with Run(chosen, progress=False) as run:
        with monkeypatch.context() as patch:
            patch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: pytest.fail("native lowering started"))
            with pytest.raises(ValueError, match=refused):
                chosen.method.prepare(chosen, run=run)
        assert run.trace.preparations == 0
    shortfall[0] = 0
    with Run(chosen, directory=tmp_path / "run", progress=False) as run:
        chosen.method.prepare(chosen, run=run)
        assert run.trace.preparations == 1
    shortfall[0] = 1
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as run:
        with pytest.raises(ValueError, match=refused):
            run.plan.method.execute(run.plan, run=run)
        assert run.trace.jobs == 0 and reduced == []
    shortfall[0] = 0
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as run:
        data = run.plan.method.execute(run.plan, run=run)
        assert run.trace.jobs == 1
    width = asked[0][1]
    assert reduced == [(1 << width,)] and [chunk.point for chunk in data.observations.chunks] == ["end"]
    assert asked == [("setting_0", width, "end")] * 4
