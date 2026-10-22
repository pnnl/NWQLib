"""Exact dense-unitary synthesis and its use for controlled gates.

Qiskit 2.6.0 snaps the Weyl coordinates of a two-qubit unitary to a special
class within average gate fidelity 1e-9, which gives entry errors of the
order of t for exp(-i t G) at small t, and its quantum Shannon decomposition
drops rotations of at most 1e-10 rad. Its ``UnitaryGate.control`` keeps a
synthesis of the controlled matrix that ``numpy.allclose(atol=1e-7,
rtol=1e-5)`` accepts. The expected values below are formed independently
of the synthesis: SciPy matrix exponentials, Kronecker products and
controlled matrices assembled entry by entry.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary

from nwqlib.subroutines._dense_synthesis import dense_unitary_circuit
from nwqlib.subroutines.qiskit_compat import controlled

UNIT_ROUNDOFF = 2.0**-53


def _hermitian(rng, dimension):
    x = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    return (x + x.conj().T) / 2


def _controlled_matrix(unitary, controls, state):
    """Controlled matrix with the controls as the low-order qubits, as in Qiskit's ControlledGate."""
    dimension = unitary.shape[0]
    matrix = np.eye(dimension << controls, dtype=complex)
    selected = [(row << controls) | state for row in range(dimension)]
    matrix[np.ix_(selected, selected)] = unitary
    return matrix


def _cx(circuit):
    return circuit.count_ops().get("cx", 0)


@pytest.mark.parametrize(("num_qubits", "family"), [(1, "evolution"), (2, "evolution"), (3, "evolution"),
                                                   (4, "evolution"), (5, "evolution"), (5, "kronecker"),
                                                   (6, "kronecker")])
def test_dense_unitary_circuit_is_exact_to_rounding(num_qubits, family):
    # Qiskit's own definition errs by 0.75 t to 1.6 t at two qubits for t up
    # to 3e-5 and by up to about 1e-8 at three qubits for t near 1e-9 to
    # 1e-8, which the evolutions exp(-i t G) over t in [1e-10, 10] detect.
    # A Kronecker product A (x) B with B on the two lowest qubits gives A.2
    # blocks that need at most two CX with the diagonal they receive, so the
    # batched A.2 loop of _apply_a2 restarts after them inside a batch.
    # Keeping the stale diagonal after such a block, or a frame that the
    # restart discards, raised the largest entry error to 0.31 to 0.91 for
    # each of the seeds below and either change. The bound is 64 unit
    # roundoffs times the dimension, about four times the largest value
    # measured, 14 for the evolutions on widths 1 to 5 and 17 over 30 seeds
    # of each Kronecker shape.
    dimension = 1 << num_qubits
    bound = 64 * UNIT_ROUNDOFF * dimension
    if family == "evolution":
        rng = np.random.default_rng(20260924 + num_qubits)
        generators = [_hermitian(rng, dimension) for _ in range(2)]
        cases = [(time, expm(-1j * time * generator)) for generator in generators
                 for time in np.logspace(-10, 1, 12)]
    else:
        cases = [(seed, np.kron(random_unitary(dimension // 4, seed=2 * seed).data,
                                random_unitary(4, seed=2 * seed + 1).data))
                 for seed in {5: (0, 2), 6: (6, 20)}[num_qubits]]
    for label, unitary in cases:
        error = np.abs(Operator(dense_unitary_circuit(unitary)).data - unitary).max()
        assert error <= bound, (label, error)


def test_generic_cx_count_matches_block_zxz_law():
    # Krol and Al-Ars, arXiv:2403.13692v2, abstract and Sec. 5.3:
    # (22/48) 4**n - (3/2) 2**n + 5/3 CX.
    num_qubits = 4
    unitary = random_unitary(1 << num_qubits, seed=num_qubits).data
    circuit = dense_unitary_circuit(unitary)
    law = (22 * 4**num_qubits - 72 * 2**num_qubits + 80) // 48
    assert _cx(circuit) == law
    assert set(circuit.count_ops()) <= {"u", "cx", "rz", "h"}


def test_near_special_two_qubit_unitary_keeps_its_entangling_part():
    # Coordinates of 1e-9, far outside the 2**-46 rounding window, still need their CX.
    x, y, z = (np.array(p, dtype=complex) for p in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]))
    canonical = lambda a, b, c: expm(1j * (a * np.kron(x, x) + b * np.kron(y, y) + c * np.kron(z, z)))  # noqa: E731
    left = np.kron(random_unitary(2, seed=1).data, random_unitary(2, seed=2).data)
    right = np.kron(random_unitary(2, seed=3).data, random_unitary(2, seed=4).data)
    for coordinates, cx in (((1e-9, 1e-10, 1e-11), 3), ((1e-9, 0, 0), 2), ((np.pi / 4 - 1e-9, 0, 0), 2)):
        unitary = left @ canonical(*coordinates) @ right
        circuit = dense_unitary_circuit(unitary)
        assert _cx(circuit) == cx, coordinates
        assert np.abs(Operator(circuit).data - unitary).max() <= 64 * UNIT_ROUNDOFF * 4, coordinates


def test_block_diagonal_input_is_demultiplexed_directly():
    # A controlled two-qubit unitary on three qubits is block diagonal in its
    # top qubit. Demultiplexing gives one multiplexor of 4 CX and two blocks
    # of at most three CX, the bound (25/96) 4**3 - 2**3 + 4/3 = 10 of the
    # controlled census. A generic three-qubit synthesis would take 19.
    target = random_unitary(4, seed=11).data
    unitary = np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), target]])
    circuit = dense_unitary_circuit(unitary)
    assert _cx(circuit) <= 10
    assert np.abs(Operator(circuit).data - unitary).max() <= 64 * UNIT_ROUNDOFF * 8
    assert len(dense_unitary_circuit(np.exp(0.7j) * np.eye(8)).data) == 0


def _a2_block_by_block(blocks):
    """Weyl frames of the A.2 blocks computed one block at a time, the definition that _apply_a2 batches."""
    from nwqlib.subroutines import _dense_synthesis as ds

    def cx_count(theta):
        return ds._cx_counts(*ds._reduced_weyl(theta)[2:])[0]

    frames, delta = [], None
    for index, block in enumerate(blocks):
        matrix = ds._polar(block if delta is None else block @ np.diag(delta))
        frame, delta = ds._weyl(matrix[None]), None
        if index + 1 < len(blocks) and cx_count(frame[3]) > 2:
            candidate, reduced = ds._two_qubit_up_to_diagonal(matrix)
            reduced_frame = ds._weyl(reduced[None])
            if cx_count(reduced_frame[3]) <= 2:
                delta, frame = candidate, reduced_frame
        frames.append(frame)
    return [np.concatenate([frame[part] for frame in frames]) for part in range(4)]


def test_a2_batches_restart_after_a_block_that_keeps_its_own_circuit():
    # _apply_a2 assumes that every block of a batch passes its diagonal on and
    # discards the rest of the batch after a block that does not. A local
    # block times the incoming Delta = diag(1, 1, e^{i psi}, e^{-i psi}), a
    # controlled RZ, needs at most two CX, so it passes nothing, and the next
    # block must start without a diagonal. Batches hold 1, 1, 1, 1, 2, 3, 4,
    # 6, 9, ... blocks after a restart, so the local block at 2 ends a
    # one-block batch, and those at 25 and 57 cut short a batch of 9 and one
    # of 10 blocks, whose later blocks are recomputed. Keeping a stale
    # diagonal or a discarded frame changes the frames by O(1).
    from nwqlib.subroutines._dense_synthesis import _apply_a2

    rng = np.random.default_rng(21)
    blocks = [random_unitary(4, seed=rng).data for _ in range(64)]
    for position in (2, 25, 57):
        blocks[position] = np.kron(random_unitary(2, seed=rng).data, random_unitary(2, seed=rng).data)
    expected = _a2_block_by_block(blocks)
    actual = _apply_a2(np.array(blocks))
    for part, (got, want) in enumerate(zip(actual, expected)):
        assert np.abs(got - want).max() <= 1e-12, part


@pytest.mark.parametrize("controls,state", [(2, 1)])
def test_controlled_composite_uses_exact_dense_definition(controls, state):
    from nwqlib.subroutines.block_encoding.core import build_block_encoding

    rng = np.random.default_rng(5)
    unitary = expm(-1j * 1e-6 * _hermitian(rng, 4))
    branch = QuantumCircuit(2, global_phase=0.4)
    branch.append(UnitaryGate(unitary), [0, 1])
    # Qiskit's own route errs by about 1e-6 on the first composite. For
    # A = diag(1, 1 - 1e-12) the one-ancilla dense dilation differs from a
    # local gate by about 1.4e-6, and Qiskit's synthesis drops that part.
    dilation = build_block_encoding(np.diag([1.0, 1.0 - 1e-12]), implementation="dense_dilation").circuit
    for composite, matrix in ((branch, np.exp(0.4j) * unitary), (dilation, Operator(dilation).data)):
        gate = controlled(composite.to_gate(), controls, ctrl_state=state)
        expected = _controlled_matrix(matrix, controls, state)
        assert np.abs(Operator(gate).data - expected).max() <= 1e-13


def test_exact_dense_unitaries_make_basis_lowering_exact():
    # NWQ-Sim, Nexus and IonQ lower with transpile at optimization level 0,
    # which would synthesize these near-identity unitaries with errors of
    # about 1e-6 and 1e-9.
    from qiskit import transpile
    from nwqlib.subroutines.qiskit_compat import exact_dense_unitaries

    rng = np.random.default_rng(9)
    two = expm(-1j * 1e-6 * _hermitian(rng, 4))
    three = expm(-1j * 1e-8 * _hermitian(rng, 8))
    inner = QuantumCircuit(2)
    inner.append(UnitaryGate(two), [0, 1])
    circuit = QuantumCircuit(3, global_phase=0.2)
    circuit.append(UnitaryGate(two), [1, 2])
    circuit.append(UnitaryGate(three), [0, 1, 2])
    circuit.append(inner.to_gate(), [2, 0])
    expected = Operator(circuit).data
    for basis in (["u", "cx"], ["rx", "ry", "rz", "cx"]):
        lowered = transpile(exact_dense_unitaries(circuit), basis_gates=basis, optimization_level=0)
        assert np.abs(Operator(lowered).data - expected).max() <= 1e-13, basis
    plain = QuantumCircuit(2)
    plain.cx(0, 1)
    assert exact_dense_unitaries(plain) is plain


@pytest.mark.parametrize("controls", [2])
def test_directly_controlled_unitary_is_exact_to_rounding(controls):
    # Qiskit 2.6.0's UnitaryGate.control keeps its synthesis of the
    # controlled matrix within numpy.allclose(atol=1e-7, rtol=1e-5). Here it
    # erred by 2.5e-11 to 9.0e-11 per entry already at t = 1e-10, and by up
    # to 1.4e-7 at t = 1e-6 for other random generators.
    rng = np.random.default_rng(30 + controls)
    for num_qubits in (1, 2):
        generator = _hermitian(rng, 1 << num_qubits)
        for time in (1e-10, 1e-6, 1e-2, 1.0):
            unitary = expm(-1j * time * generator)
            for state in (0, (1 << controls) - 1):
                gate = controlled(UnitaryGate(unitary), controls, ctrl_state=state)
                expected = _controlled_matrix(unitary, controls, state)
                bound = 64 * UNIT_ROUNDOFF * expected.shape[0]
                assert np.abs(Operator(gate).data - expected).max() <= bound, (num_qubits, time, state)
    # Lowering to a gate basis keeps the exact definition.
    from qiskit import transpile

    unitary = expm(-1e-6j * _hermitian(rng, 4))
    circuit = QuantumCircuit(2 + controls)
    circuit.append(controlled(UnitaryGate(unitary), controls, ctrl_state=0), range(2 + controls))
    lowered = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    expected = _controlled_matrix(unitary, controls, 0)
    assert np.abs(Operator(lowered).data - expected).max() <= 64 * UNIT_ROUNDOFF * expected.shape[0]


def test_dense_qpe_admission_charges_the_controlled_synthesis():
    # Planning charges each dense power the exact synthesis of its controlled
    # matrix before anything is built. With seven system qubits the
    # synthesis alone needs 11 * 256**3 + (8**2 + 5*8 + 256) * 256**2 = 2.08e8
    # work units, above an explicit max_work of 1e8. With six it needs 2.9e7.
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms.qpe import QCELS

    rng = np.random.default_rng(16)
    with pytest.raises(ValueError, match="max_work"):
        plan(Eigenproblem(A=_hermitian(rng, 128)), method=QCELS(initial_state=np.eye(128)[0], max_work=10**8), seed=7)
    from test_synthesis_admission import _controlled_synthesis_work

    chosen = plan(Eigenproblem(A=_hermitian(rng, 64)), method=QCELS(initial_state=np.eye(64)[0], max_work=10**8), seed=7)
    block = next(b for b in chosen.blocks if b.record.signature.name == "power_p1")
    # The eigendecomposition that the first constructed power computes,
    # 8 D**3 + 32 D**2 units, one product for the spectral power, D**3, and
    # 8 D**2, then the synthesis on seven qubits.
    assert block.record.construction_work == 9 * 64**3 + 40 * 64**2 + _controlled_synthesis_work(7)


def _dense_synthesis_work(num_qubits):
    """Work law of one ``dense_unitary_circuit`` on m qubits, 113 M**3 / 4 + (m**2 + 16 m + 512) M**2.

    It is written out here so that a test states the boundary it expects
    independently of the library's ``dense_synthesis_size``.
    """
    size = 1 << num_qubits
    return 113 * size**3 // 4 + (num_qubits**2 + 16 * num_qubits + 512) * size**2


def _counted_syntheses(monkeypatch):
    """Record the qubit count of every exact dense synthesis made after this call."""
    from nwqlib.subroutines import _dense_synthesis

    calls = []
    original = _dense_synthesis.dense_unitary_circuit

    def counted(matrix):
        calls.append(np.asarray(matrix).shape[0].bit_length() - 1)
        return original(matrix)

    monkeypatch.setattr(_dense_synthesis, "dense_unitary_circuit", counted)
    return calls


@pytest.mark.parametrize("num_qubits", [4])
def test_dense_synthesis_size_bounds_the_kept_instructions_and_the_traced_peak(num_qubits):
    # The kept-byte law allots 256 bytes to each of at most 11 * 4**m / 8
    # instructions, the census bound for any input. A Haar-random input used
    # 45 to 84 percent of these slots for m = 2 to 5, and a block-diagonal
    # input takes the demultiplexing path instead. The working-byte law
    # bounds the Python-side peak of one call, which tracemalloc measured at
    # 0.41 to 0.63 of the law for Haar-random and near-identity inputs and
    # 0.25 to 0.43 for block-diagonal ones, m = 3 to 9.
    import tracemalloc
    from nwqlib.subroutines._dense_synthesis import dense_synthesis_size

    size = 1 << num_qubits
    generic = random_unitary(size, seed=40 + num_qubits).data
    block = np.zeros((size, size), dtype=complex)
    block[: size // 2, : size // 2] = random_unitary(size // 2, seed=50 + num_qubits).data
    block[size // 2 :, size // 2 :] = random_unitary(size // 2, seed=60 + num_qubits).data
    _, working, kept = dense_synthesis_size(num_qubits)
    for matrix in (generic, block):
        circuit = dense_unitary_circuit(matrix)
        assert kept >= 256 * len(circuit.data) + 16384
        tracemalloc.start()
        try:
            dense_unitary_circuit(matrix)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak <= working


def test_basis_lowering_admits_its_syntheses_before_the_first_one(monkeypatch):
    # The rewrite synthesizes each distinct dense matrix once. Here that is
    # two three-qubit unitaries, one of them also nested in a composite, and
    # one two-qubit unitary. A one-qubit unitary keeps Qiskit's exact
    # definition. A charge that refuses stops the rewrite before any
    # synthesis starts. A Run's charge receives their summed law once,
    # before the first synthesis.
    from nwqlib.subroutines.qiskit_compat import dense_synthesis_widths, exact_dense_unitaries

    first, second = random_unitary(8, seed=31).data, random_unitary(8, seed=32).data
    inner = QuantumCircuit(3)
    inner.append(UnitaryGate(first), [0, 1, 2])
    circuit = QuantumCircuit(3)
    circuit.append(UnitaryGate(first), [0, 1, 2])
    circuit.append(inner.to_gate(), [2, 1, 0])
    circuit.append(UnitaryGate(second), [0, 1, 2])
    circuit.append(UnitaryGate(random_unitary(4, seed=33).data), [0, 1])
    circuit.append(UnitaryGate(random_unitary(2, seed=34).data), [2])
    assert sorted(dense_synthesis_widths(circuit)) == [2, 3, 3]
    need = 2 * _dense_synthesis_work(3) + _dense_synthesis_work(2)
    calls = _counted_syntheses(monkeypatch)

    def refuse(work, operation):
        raise ValueError(f"{operation} refused")

    with pytest.raises(ValueError, match="refused"):
        exact_dense_unitaries(circuit, charge=refuse)
    assert calls == []
    exact_dense_unitaries(circuit)
    assert sorted(calls) == [2, 3, 3]
    charges = []
    exact_dense_unitaries(circuit, charge=lambda work, operation: charges.append(work))
    assert charges == [need]


def test_noisy_aer_run_admits_dense_synthesis_against_its_limit(monkeypatch):
    # A noise model's basis omits Aer's unitary instruction, so preparing
    # this supplied state circuit synthesizes its three-qubit unitary. The
    # Run's max_synthesis_work refuses one unit below that synthesis before
    # it starts and admits it at the law.
    import nwqlib
    from qiskit_aer.noise import NoiseModel
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import state_input

    state = QuantumCircuit(3)
    state.append(UnitaryGate(random_unitary(8, seed=35).data), [0, 1, 2])
    problem = Expectation(state=state_input(state), observable=ingest_pauli((("ZZZ", 1.0),), num_qubits=3))
    chosen = nwqlib.plan(problem, method=ExpectationMethod(), shots=16, seed=7)
    backend = AerBackend.from_noise_model(NoiseModel(basis_gates=["cx", "rz", "sx", "x"]))
    need = _dense_synthesis_work(3)
    calls = _counted_syntheses(monkeypatch)
    with pytest.raises(ValueError, match="max_synthesis_work"):
        nwqlib.prepare(chosen, backend=backend, limits=ExecutionLimits(max_synthesis_work=need - 1))
    assert calls == []
    prepared = nwqlib.prepare(chosen, backend=backend, limits=ExecutionLimits(max_synthesis_work=need))
    prepared.run.close()
    assert calls == [3]


def test_coherent_qpe_admits_its_powers_and_their_controlled_synthesis():
    # The unitarity check and the m - 1 squarings of a D-square unitary cost
    # D**3 work units each, and each of the m powers adds the exact synthesis
    # of its controlled matrix on log2(D) + 1 qubits. The limit is compared
    # before the check.
    from nwqlib.subroutines.qpe import build_coherent_qpe_circuit
    from test_synthesis_admission import _controlled_synthesis_work

    unitary = random_unitary(4, seed=36).data
    need = 3 * (4**3 + _controlled_synthesis_work(3))
    with pytest.raises(ValueError, match="max_work"):
        build_coherent_qpe_circuit(unitary, num_phase_qubits=3, max_work=need - 1)
    with pytest.raises(ValueError, match="max_bytes"):
        build_coherent_qpe_circuit(unitary, num_phase_qubits=3, max_bytes=1000)
    qpe = build_coherent_qpe_circuit(unitary, num_phase_qubits=3, max_work=need)
    assert qpe.powers == (1, 2, 4)
