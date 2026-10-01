"""Actual saved arrays, source associations and explicit Result reanalysis."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import nwqlib
from nwqlib.algorithms.qhd import QHD
from nwqlib.backends import qiskit_aer
from nwqlib.problems import Optimization


@pytest.fixture
def supplied_qhd(monkeypatch):
    raw = np.zeros(8, dtype=np.complex128)
    raw[0], raw[1], raw[2], raw[4] = 0.5j, 0.5, -0.5j, -0.5
    calls = []

    def acquire(native):
        calls.append(native)
        return SimpleNamespace(
            raw_output={"statevector": raw},
            metadata={
                "native_job_id": "supplied-eight-amplitudes",
                "statevector_semantics": "pre_final_measurement",
            },
        )

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", acquire)
    x = sp.Symbol("x")
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    plan = nwqlib.plan(
        problem, method=QHD(num_grid_points=3, num_steps=1, total_time=0.1, keep_state=True), seed=7
    )
    return SimpleNamespace(plan=plan, raw=raw, calls=calls)


def test_qhd_saved_array_and_reanalysis_keep_actual_coordinates(tmp_path, monkeypatch, supplied_qhd):
    case = supplied_qhd
    result = nwqlib.solve(case.plan)
    assert len(case.calls) == 1 and result.valid_mass == 0.75 and result.invalid_mass == 0.25
    path = result.save(tmp_path / "result")

    def forbidden(*a, **k):
        raise AssertionError("load attempted planning, native work or publication")

    monkeypatch.setattr(QHD, "plan", forbidden)
    monkeypatch.setattr(qiskit_aer, "_prepare_aer_execution", forbidden)
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", forbidden)
    import nwqlib.artifacts as artifacts

    monkeypatch.setattr(artifacts, "sha256", forbidden)
    restored = nwqlib.load_result(path)
    handle = restored.data.artifact(restored.artifact)
    assert type(handle.array) is np.ndarray and isinstance(handle.array.base, np.memmap)
    assert not handle.array.flags.writeable
    np.testing.assert_array_equal(handle.array, case.raw)
    again = restored.analyze()
    assert again.data is restored.data and again.marginals.array.tolist() == [[0.25, 0.25, 0.25]]
    assert again.valid_mass == 0.75 and again.invalid_mass == 0.25 and again.candidate_indices == (1,)
    assert again.objective == pytest.approx(0.04, abs=1e-15, rel=0)
    assert again.origin.invocation_id != result.origin.invocation_id
    assert again.contribution_ids == result.contribution_ids and len(case.calls) == 1


@pytest.mark.parametrize("damage", ["shape", "dtype", "missing"])
def test_saved_array_layout_corruption_rejects_without_scientific_reanalysis(
    tmp_path, monkeypatch, supplied_qhd, damage
):
    result = nwqlib.solve(supplied_qhd.plan)
    path = result.save(tmp_path / damage)
    saved = json.loads((path / "result.json").read_text())
    array = path / saved["data"]["artifacts"][0]["array"]
    if damage == "missing":
        array.unlink()
    else:
        with array.open("wb") as stream:
            np.save(
                stream,
                np.zeros(
                    4 if damage == "shape" else 8, dtype=np.complex128 if damage == "shape" else np.float64
                ),
            )
    monkeypatch.setattr(QHD, "analyze", lambda *a, **k: pytest.fail("damaged archive reached analysis"))
    with pytest.raises((ValueError, FileNotFoundError)):
        nwqlib.load_result(path)
    assert len(supplied_qhd.calls) == 1


def test_interrupted_result_snapshot_removes_its_folder_and_keeps_live_array(
    tmp_path, monkeypatch, supplied_qhd
):
    result = nwqlib.solve(supplied_qhd.plan)
    original = result.data.artifact(result.artifact).array
    import nwqlib._choice_archive as owner

    write = owner.ArchiveFiles.write_array

    def interrupted(files, *args):
        write(files, *args)
        raise OSError("interrupted explicit snapshot")

    monkeypatch.setattr(owner.ArchiveFiles, "write_array", interrupted)
    with pytest.raises(OSError, match="interrupted explicit snapshot"):
        result.save(tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert result.data.artifact(result.artifact).array is original and not original.flags.writeable
    np.testing.assert_array_equal(original, supplied_qhd.raw)


def test_equal_saved_bytes_preserve_distinct_acquisition_provenance(supplied_qhd):
    first = nwqlib.solve(supplied_qhd.plan)
    second = nwqlib.solve(supplied_qhd.plan)
    a, b = first.artifact, second.artifact
    assert a.digest == b.digest and a.content_id != b.content_id and a.acquisition != b.acquisition
    np.testing.assert_array_equal(first.data.artifact(a).array, second.data.artifact(b).array)
    with pytest.raises(ValueError, match="not stored"):
        first.data.artifact(b)


@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_artifact_publication_canonicalizes_endian_before_identity(dtype):
    from hashlib import sha256
    import struct
    from nwqlib.artifacts import ArrayOutput, ArtifactStore
    from nwqlib.core import Basis, Source

    if dtype == "complex128":
        expected = struct.pack("<6d", 1.0, -2.0, 3.0, 0.5, -0.0, 0.0)
        array = np.asarray([1 - 2j, 0j, 3 + 0.5j, 0j, complex(-0.0, 0.0), 0j], dtype=">c16")[::2]
    else:
        expected = struct.pack("<3d", 1.0, -2.5, -0.0)
        array = np.asarray([1.0, 7.0, -2.5, 7.0, -0.0, 7.0], dtype=">f8")[::2]
    output = ArrayOutput(
        name="supplied",
        kind="vector",
        basis=Basis(identity="three", dimension=3, ordering="increasing coordinate"),
        frame="physical",
        global_phase="physical",
        dtype=dtype,
    )
    source = Source(name="supplied values", version="1", domain="test", reference="struct.pack oracle")
    identity = "sha256:" + "a" * 64
    provenance = dict(
        plan_id=identity,
        realization_id=identity,
        construction_id=identity,
        producer_id=identity,
        acquisition=("run", "attempt", "job", "0"),
        source=source,
    )
    store = ArtifactStore(max_bytes=2 * len(expected))
    handle = store._publish(array, output=output, provenance=provenance)
    assert handle.manifest.digest == "sha256:" + sha256(expected).hexdigest()
    assert handle.manifest.encoding == f"{dtype}-le-c" and handle.manifest.data_bytes == len(expected)
    assert handle.array.dtype.str == ("<c16" if dtype == "complex128" else "<f8")
    assert handle.array.tobytes() == expected
    array[0] = 99
    assert handle.array.tobytes() == expected and not handle.array.flags.writeable
    assert store.data_bytes == len(expected)
    # A handed-over private array is kept, not copied, and can no longer be written.
    private = np.frombuffer(expected, dtype=handle.array.dtype).copy()
    kept = store._publish(private, output=output, private=True,
                          provenance=dict(provenance, acquisition=("run", "attempt", "job", "1")))
    assert np.shares_memory(kept.array, private) and kept.manifest.digest == handle.manifest.digest
    with pytest.raises(ValueError, match="read-only"):
        private[0] = 99
    with pytest.raises(ValueError):
        kept.array.flags.writeable = True
    if dtype == "float64":
        with pytest.raises(ValueError, match="global_phase"):
            ArrayOutput.model_validate(dict(output.model_dump(exclude_computed_fields=True),
                                            global_phase="modulo_global_phase"))
    # Finiteness is checked in blocks: a nonfinite last entry of a vector longer
    # than one block, and of the last row of an operator, still rejects.
    for kind, size in (("vector", 3 * 2**16 + 5), ("operator", 300)):
        wide = ArrayOutput(name="wide", kind=kind, frame="physical", global_phase="physical", dtype=dtype,
                           basis=Basis(identity="wide", dimension=size, ordering="increasing coordinate"))
        values = np.zeros(wide.shape, dtype=handle.array.dtype)
        values.reshape(-1)[-1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            ArtifactStore(max_bytes=wide.data_bytes)._publish(values, output=wide, provenance=provenance)


def test_saved_source_text_does_not_choose_executable_owner(tmp_path, monkeypatch, supplied_qhd):
    result = nwqlib.solve(supplied_qhd.plan)
    path = result.save(tmp_path / "data")
    saved = json.loads((path / "result.json").read_text())
    saved["selection"]["method"] = "os.system"
    (path / "result.json").write_text(json.dumps(saved))
    import importlib

    monkeypatch.setattr(
        importlib, "import_module", lambda *a: pytest.fail("saved source selected arbitrary import")
    )
    with pytest.raises(ValueError, match="Method|method"):
        nwqlib.load_result(path)


def test_saved_physical_input_layout_is_checked_before_reuse(tmp_path, monkeypatch):
    from nwqlib.algorithms import LCHS
    from nwqlib.problems import inputs

    result = nwqlib.solve(nwqlib.LinearDynamics(A=np.eye(2), initial_state=[1., 0.], time=0),
                          method=LCHS(), execution="classical")
    path = result.save(tmp_path / "initial")
    saved = json.loads((path / "result.json").read_text())
    filename = saved["selection"]["selected"]["problem"]["inputs"]["initial_state"]["data"]["physical"]
    monkeypatch.setattr(inputs, "normalize_physical_vector_with_scale",
                        lambda *a: pytest.fail("load normalized the input again"))
    loaded = nwqlib.load_result(path)
    np.testing.assert_array_equal(loaded.solution, [1., 0.])
    with (path / filename).open("wb") as stream:
        np.save(stream, np.array([1., 0., 5.]))  # Same dtype, wrong native dimension.
    with pytest.raises(ValueError, match="layout"):
        nwqlib.load_result(path)


def test_saved_native_layout_preserves_compact_and_declared_inputs(tmp_path, monkeypatch):
    from scipy import sparse
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.operators import inputs as operators
    from nwqlib.problems import inputs as states

    state_inputs = [states.ingest_vector([1., 2.]), states.ingest_vector([1j, 2.]),
                    states.ingest_vector([0., 0.]), states.ingest_product([[0., 0.], [1., 0.]]),
                    states.ingest_product([[1j, 1.], [1., 0.]]),
                    states.ingest_occupation("10", num_qubits=2)]
    state_inputs += [states.StateInput.from_record(state_inputs[0].to_record())]
    dense = operators.ingest_dense(np.diag([1., 2.]))
    operator_inputs = [dense, operators.ingest_dense(np.diag([1j, 2j])),
                       operators.ingest_sparse(sparse.csr_matrix(np.diag([1., 2.]))),
                       operators.ingest_sparse(sparse.csc_array(np.diag([1j, 2j]))),
                       operators.ingest_pauli([("Z", 1.)], num_qubits=1),
                       operators.ingest_periodic_stencil(num_qubits=2, mass=.2, diffusion=.1),
                       operators.OperatorInput.from_record(dense.to_record())]
    files = ArchiveFiles(tmp_path, 100_000)
    state_data = [files.write_state(f"state-{i}", state) for i, state in enumerate(state_inputs)]
    operator_data = [files.write_operator(f"operator-{i}", op) for i, op in enumerate(operator_inputs)]
    # Sharing is retained by the archive owner, not copied by layout validation.
    repeated = files.write_operator("same", dense)
    assert repeated["array"] == operator_data[0]["array"]
    monkeypatch.setattr(operators, "_digest", lambda *a: pytest.fail("load rehashed native input"))
    monkeypatch.setattr(states, "normalize_physical_vector_with_scale", lambda *a: pytest.fail("load normalized"))
    reopened = ArchiveFiles(tmp_path, 100_000)
    for original, data in zip(state_inputs, state_data, strict=True):
        restored = reopened.read_state(data)
        assert restored.to_record() == original.to_record()
        for name in ("_physical", "_direction"):
            a, b = getattr(original, name), getattr(restored, name)
            if a is None:
                assert b is None
            else:
                np.testing.assert_array_equal(a, b)
                assert a.dtype == b.dtype and not b.flags.writeable
    for original, data in zip(operator_inputs, operator_data, strict=True):
        assert reopened.read_operator(data).to_record() == original.to_record()
    assert reopened.read_operator(repeated)._data is reopened.read_operator(operator_data[0])._data
    wrong = dict(operator_data[0], array=state_data[0]["physical"])
    with pytest.raises(ValueError, match="layout"):
        reopened.read_operator(wrong)
    wrong = dict(operator_data[4], x=operator_data[4]["coefficients"])
    with pytest.raises(ValueError, match="layout"):
        reopened.read_operator(wrong)
    wrong = dict(operator_data[2], sparse_type="csc_matrix")
    with pytest.raises(ValueError, match="layout"):
        reopened.read_operator(wrong)


def _with_fault(result, *, chunk=None, event=None, item=None, receipts=None, artifacts=None):
    """Return a copy of result attached to its RunData with exactly the given changes.

    The one observation chunk, its attempt and its submission item take the
    given fields, and the attempt and the Result name the revised chunk, so
    the fault passes Result._attach and reaches save_result.
    """
    from dataclasses import replace
    from nwqlib.execution import ObservationView

    data = result.data
    (old,) = data.observations.chunks
    new = old.revise(**chunk) if chunk else old
    events = tuple(e.revise(**{"observation_id": new.content_id, **(event or {})}) if e.attempt == old.attempt else e
                   for e in data.trace.events)
    submissions = data.trace.submissions if item is None else tuple(
        s.revise(items=tuple(i.revise(**item) if i.attempt == old.attempt else i for i in s.items))
        for s in data.trace.submissions)
    observations = ObservationView(chunks=(new,))
    forged = replace(data, observations=observations, trace=data.trace.revise(events=events, submissions=submissions),
                     receipts=data.receipts if receipts is None else receipts,
                     artifacts=data.artifacts if artifacts is None else artifacts)
    ids = tuple(new.content_id if identity == old.content_id else identity for identity in result.contribution_ids)
    return result.revise(observation_id=observations.content_id, contribution_ids=ids)._attach(result.plan, forged)


@pytest.fixture(scope="module")
def hadamard_counts():
    """A legal counts Result of the reference Hadamard Method, another receipt of its Plan and one of another Plan."""
    from _hadamard_method import HadamardPauliExpectation
    from nwqlib.operators import ingest_pauli

    problem = nwqlib.Expectation(state=[1.0, 1j], observable=ingest_pauli((("Y", 1.0),), num_qubits=1))
    selected = nwqlib.plan(problem, method=HadamardPauliExpectation(), shots=8, seed=7)
    other = nwqlib.plan(problem, method=HadamardPauliExpectation(), shots=16, seed=7)
    return SimpleNamespace(result=nwqlib.solve(selected), again=nwqlib.solve(selected).data.receipts[0],
                           foreign=nwqlib.solve(other).data.receipts[0])


_JOIN = "differs from its actual preparation and completed attempt"


def _join_faults(case):
    """Map each fault name to RunData changes that break exactly one join, and the refusal it must raise."""
    from nwqlib.ir.expressions import Binding

    (chunk,) = case.result.data.observations.chunks
    receipts = case.result.data.receipts
    moved = lambda layout: tuple(register.revise(name="moved") for register in layout)  # noqa: E731
    return {
        "attempt_missing": (dict(chunk=dict(attempt="another-attempt")), _JOIN),
        # An attempt names its observation exactly when it is completed (ConsumptionEvent).
        "attempt_not_completed": (dict(event=dict(status="uncertain", observation_id=None)), _JOIN),
        "attempt_names_another_observation": (dict(event=dict(observation_id="sha256:" + "0" * 64)), _JOIN),
        "attempt_names_another_receipt": (dict(receipts=receipts + (case.again,),
            event=dict(prepared_id=case.again.content_id), item=dict(prepared_id=case.again.content_id)), _JOIN),
        "realization": (dict(chunk=dict(realization_id="sha256:" + "1" * 64)), _JOIN),
        "readout": (dict(chunk=dict(observation=chunk.observation.revise(shots=9))), _JOIN),
        "bindings": (dict(chunk=dict(bindings=(Binding(parameter="steps", value=1),))), _JOIN),
        "quantum_layout": (dict(chunk=dict(quantum_layout=moved(chunk.quantum_layout))), _JOIN),
        "classical_layout": (dict(chunk=dict(classical_layout=moved(chunk.classical_layout))), _JOIN),
        "population": (dict(chunk=dict(population="unknown")), _JOIN),
        "plan": (dict(chunk=dict(plan_id="sha256:" + "2" * 64)), _JOIN),
        "run": (dict(chunk=dict(run_id="another-run")), _JOIN),
        "returned_shots": (dict(event=dict(returned_shots=7)), _JOIN),
        "submission_job": (dict(chunk=dict(job="another-job")), "job/item association"),
        "receipt_from_another_plan": (dict(receipts=receipts + (case.foreign,)), "preparation belongs to another Plan"),
        "duplicate_receipts": (dict(receipts=receipts + receipts), "receipts must be distinct"),
        "attempt_without_receipt": (dict(receipts=()), "no original preparation receipt"),
    }


@pytest.mark.parametrize("fault", [
    "attempt_missing", "attempt_not_completed", "attempt_names_another_observation", "attempt_names_another_receipt",
    "realization", "readout", "bindings", "quantum_layout", "classical_layout", "population", "plan", "run",
    "returned_shots", "submission_job", "receipt_from_another_plan", "duplicate_receipts", "attempt_without_receipt",
])
def test_save_refuses_an_observation_that_differs_from_its_attempt_or_receipt(
    tmp_path, monkeypatch, hadamard_counts, fault
):
    """Saving is where a Result's observations meet their attempts and receipts.

    Loading does not repeat these joins, so each RunData below differs from a
    legal one in exactly one join and must be refused before any file is
    written. The Hadamard Result's own validate_data repeats the submission
    join, so it is disabled here and each refusal comes from the shared owner
    (saved_evidence._validate_recorded_data). The same data rebuilt without a
    fault still saves and loads.
    """
    from _hadamard_method import HadamardExpectationResult, HadamardPauliExpectation

    monkeypatch.setattr(HadamardExpectationResult, "validate_data", lambda self, data: None)
    rebuilt = _with_fault(hadamard_counts.result)
    loaded = nwqlib.load_result(rebuilt.save(tmp_path / "legal"), method=HadamardPauliExpectation)
    assert loaded.content_id == rebuilt.content_id and loaded.value == hadamard_counts.result.value
    changes, message = _join_faults(hadamard_counts)[fault]
    forged = _with_fault(hadamard_counts.result, **changes)
    with pytest.raises(ValueError, match=message):
        forged.save(tmp_path / fault)
    assert not (tmp_path / fault).exists()


@pytest.mark.parametrize("fault,message", [
    ("array_without_manifest", "missing its requested array"),
    ("duplicate_manifests", "require distinct manifests"),
    ("manifest_from_another_plan", "array belongs to another Plan"),
])
def test_save_refuses_an_array_without_its_one_manifest(tmp_path, supplied_qhd, fault, message):
    """Each array a saved observation names needs one manifest of the same Plan in the saved inventory."""
    result = nwqlib.solve(supplied_qhd.plan)
    (handle,) = result.data.artifacts
    if fault == "array_without_manifest":
        artifacts = ()
    elif fault == "duplicate_manifests":
        artifacts = (handle, handle)
    else:
        other = nwqlib.plan(supplied_qhd.plan.problem,
                            method=QHD(num_grid_points=3, num_steps=1, total_time=0.2, keep_state=True), seed=7)
        artifacts = (handle, *nwqlib.solve(other).data.artifacts)
    _with_fault(result).save(tmp_path / "legal")
    with pytest.raises(ValueError, match=message):
        _with_fault(result, artifacts=artifacts).save(tmp_path / fault)
    assert not (tmp_path / fault).exists()
