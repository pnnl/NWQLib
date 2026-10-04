"""Small persistence failures retain operation context and scientific data."""

import shutil
import sqlite3

import numpy as np
import pytest

from nwqlib._choice_archive import ArchiveFiles
from nwqlib._run_journal import LocalJournal


def test_array_truncation_names_entry_without_scanning_valid_payload(tmp_path):
    files = ArchiveFiles(tmp_path, 10_000)
    name = files.write_array("array.npy", np.arange(8.))
    valid = ArchiveFiles(tmp_path, 10_000).read_array(name)
    assert not valid.flags.writeable and not valid.flags.owndata
    path = tmp_path / name
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError) as caught:
        ArchiveFiles(tmp_path, 10_000).read_array(name)
    assert str(path) in " ".join(caught.value.__notes__)


def test_journal_lock_and_lost_folder_have_owned_context(tmp_path):
    folder = tmp_path / "durable"
    folder.mkdir()
    path = folder / "run.sqlite"
    journal = LocalJournal(path, 4096, create=True)
    try:
        with pytest.raises(BlockingIOError) as caught:
            LocalJournal(path, 4096, create=False)
        assert str(path) in " ".join(caught.value.__notes__)
        journal.commit((("example", "before", 1),))
        prior = journal.usage.copy()
        shutil.rmtree(folder)
        with pytest.raises(sqlite3.Error) as caught:
            journal.commit((("example", "after", 2),))
        assert str(path) in " ".join(caught.value.__notes__)
        assert journal._data_bytes == prior["data_bytes"]
    finally:
        journal.close()


def test_noise_archive_preserves_parameters_channels_and_qubit_associations(tmp_path, monkeypatch):
    import nwqlib as nw
    from nwqlib.algorithms import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib._prepared_execution import Run
    from nwqlib._run_archive import _load_noise_model, _noise_model_data
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
    from test_prepared_execution import same_native

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.rz(.37, 0)
    error = QuantumError(circuit)
    # Basis gates other than Aer's default select another lowering target,
    # which NoiseModel.to_dict does not record.
    model = NoiseModel(basis_gates=["cx", "u"])
    model.add_all_qubit_quantum_error(error, ["x"])
    model.add_quantum_error(error, ["h"], [1])
    model.add_all_qubit_readout_error(ReadoutError([[.9, .1], [.3, .7]]))
    model.add_readout_error(ReadoutError([[.8, .2], [.4, .6]]), [1])
    saved = model.to_dict()
    monkeypatch.setattr(NoiseModel, "from_dict", lambda *a: pytest.fail("deprecated loader called"))
    restored = _load_noise_model(_noise_model_data(model))
    same_native(restored.to_dict(), saved)
    assert restored.basis_gates == model.basis_gates
    quantum = restored._default_quantum_errors["x"]
    expected = np.diag(np.exp(1j*np.array([-.37/2, .37/2]))) @ (np.array([[1, 1], [1, -1]])/np.sqrt(2))
    np.testing.assert_allclose(Operator(quantum.circuits[0]).data, expected, atol=2e-15, rtol=0)
    assert quantum.id == error.id
    plan = nw.plan(nw.Expectation(state=[1., 0.], observable=[[1., 0.], [0., -1.]]),
        method=ExpectationMethod(), shots=8, seed=7)
    backend = AerBackend.from_noise_model(model)
    with Run(plan, backend=backend) as baseline:
        expected_result = baseline.wait()
    with Run(plan, backend=backend, directory=tmp_path / "noise") as run:
        run.checkpoint({"cached": True})
    metadata = AerBackend.model_validate_json(backend.model_dump_json())
    with nw.load_run(tmp_path / "noise", backend=metadata) as run:
        same_native(run.backend._bound_noise_model().to_dict(), saved)
        # Aer's NoiseModel equality also compares basis gates.
        assert run.backend._bound_noise_model() == model
        assert run.trace.jobs == 0
        actual = run.wait()
        assert actual.value == expected_result.value
        assert sum(chunk.returned_shots for chunk in actual.data.observations.chunks) == 8
        assert sum(chunk.returned_shots for chunk in expected_result.data.observations.chunks) == 8


def test_noise_loader_rejects_unrepresented_conditions():
    from nwqlib._run_archive import _load_noise_model
    with pytest.raises(ValueError, match="instruction fields"):
        _load_noise_model({"basis_gates": ["x"], "errors": [dict(type="qerror", operations=["x"], probabilities=[1.],
            instructions=[[dict(name="x", qubits=[0], conditional=1)]])]})
