"""Actual one-qubit preparation, native caches, noise and cumulative failures."""

import numpy as np
import pytest

import nwqlib
from nwqlib._prepared_execution import PreparationNotRebuilt, Run, prepare_experiment, submit_experiment
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.backends import AerBackend
from nwqlib.backends import qiskit_aer as aer
from nwqlib.execution import ExecutionLimits
from test_expectation_current import problem
from test_run_lifecycle import InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend


def base_plan(*, shots=None):
    return nwqlib.plan(problem(state=[1.0, 1.0]), method=ExpectationMethod(), shots=shots, seed=7)


def test_qubit_cap_precedes_readout_size_and_explains_binary_applicability(tmp_path, monkeypatch):
    import sympy as sp
    import nwqlib._prepared_execution as owner
    from nwqlib.algorithms.qhd import QHD, UniformState
    from nwqlib.problems import Optimization

    x, y = sp.symbols('x y', real=True)
    problem = Optimization(objective=x*x + y*y, variables=(x, y), bounds=((-1., 1.), (-1., 1.)))
    method = QHD(boundary='periodic', num_grid_points=16, num_steps=1,
                 total_time=.01, initial_state=UniformState())
    chosen = nwqlib.plan(problem, method=method, seed=7)
    cap = 20
    limits = ExecutionLimits(max_simulation_qubits=cap)
    with Run(chosen, limits=limits, directory=tmp_path / 'refused') as run:
        with monkeypatch.context() as patch:
            patch.setattr(owner, 'readout_shape', lambda **kw: pytest.fail('readout size preceded qubit admission'))
            with pytest.raises(ValueError, match=f'max_simulation_qubits={cap}') as refused:
                method.prepare(chosen, run=run)
        message = str(refused.value)
        assert '32 qubits' in message and 'encoding="binary"' in message
        assert '8 qubits' in message
        assert run.trace.preparations == 0
    binary = nwqlib.plan(problem, method=method.revise(encoding='binary'), seed=7)
    prepared = nwqlib.prepare(binary, limits=limits, directory=tmp_path / 'binary', progress=False)
    with prepared.run:
        assert prepared.circuits and all(c.num_qubits == 8 for c in prepared.circuits)


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


def test_progress_uses_published_work_and_never_repeats_acquisition(tmp_path):
    from nwqlib.core.planning import RandomStreams

    InjectedBackend.supports_synchronous = True
    chosen = selected()
    expected_rng = RandomStreams.restore(chosen.randomness)
    expected_rng.next_seed()  # One actual selected acquisition, including its preparation.
    notices = []

    def observe(stage, done, total):
        notices.append((stage, done, total, tuple(name for name, _ in InjectedBackend.calls)))
        if stage == "collect":
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
        assert len(run.warnings) == 1 and "observer unavailable" in run.warnings[0]
        assert run._state["progress"] is False


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


@pytest.mark.parametrize("preparations", [2])
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


# Two real four-qubit Pauli Hamiltonians with different supports and signs.
def settings_values(result):
    """Experiment names and returned outcomes, without Run, attempt or job identities."""
    return [(chunk.experiment, chunk.histogram().index_list(), chunk.histogram().weights.tolist())
            for chunk in result.data.observations.chunks]


def test_prepare_all_settings_exposes_every_circuit_and_submit_reuses_them(tmp_path, monkeypatch):
    from nwqlib import Eigenproblem
    from nwqlib.algorithms.qpe import QCELS
    from nwqlib.blocks import lowering
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation
    from nwqlib.resources.records import ResourceContext

    target = ingest_pauli((("ZIII", .4), ("XYZI", -.3), ("IYYX", .2), ("ZZZZ", .05)), num_qubits=4)
    method = QCELS(initial_state=ingest_occupation("0101", num_qubits=4), num_times=3, tau=.2, grid_size=64)
    selected = nwqlib.plan(Eigenproblem(A=target), method=method, shots=16, seed=7)
    names = tuple(experiment.name for experiment in selected.experiments)
    assert len(names) > 1
    # The saved Run names the local backend explicitly.
    directory = tmp_path / "all"
    prepared = nwqlib.prepare(selected, backend=AerBackend(), settings="all", directory=directory)
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
    # Stop before submission, then continue the saved Run.
    run.close()
    run = nwqlib.load_run(directory, backend=AerBackend())
    result = run.wait()
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
