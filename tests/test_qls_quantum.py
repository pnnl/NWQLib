"""Actual QLS selected bodies, native original-coordinate readout and storage."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import nwqlib
from nwqlib import (
    LinearSystem,
    Solution,
    StateVector,
    NormSquared,
    QuadraticForm,
    NormalizedExpectation,
)
from nwqlib.algorithms.qls import QLS
from nwqlib.algorithms.qls import quantum, method as owner
from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan
from nwqlib.blocks import select_block_encoding
from nwqlib.blocks.selection import SelectedBlock
from nwqlib.operators import ingest_dense
from nwqlib.subroutines.block_encoding import BlockEncoding
from test_qls_primary import MATRIX, RHS, forbidden, aligned


def choose(*, A=MATRIX, b=RHS, solver="qsvt_inverse", output=None, shots=None, **fields):
    if output is None:
        output = (
            Solution()
            if solver == "qsvt_inverse"
            else StateVector(normalization="unit", global_phase="modulo_global_phase")
        )
    selected = nwqlib.plan(
        LinearSystem(A=A, b=b),
        method=QLS(
            solver=solver,
            encoded_solution_norm_estimate=None if solver == "qsvt_inverse" else 1.5,
            **fields,
        ),
        output=output,
        shots=shots,
        seed=7,
    )
    assert selected.reconstruction.width <= 13
    return selected


def compact_key(rec, setting, ones):
    """The count key of a setting's measured register with the physical bits ``ones`` set.

    The setting measures its selectors and coordinates into the prefix
    ``quantum._counts_bits`` names, in the shared layout of
    ``quantum._counts_width`` bits.
    """
    observed = quantum._counts_bits(rec, setting)
    return format(sum(1 << observed.index(bit) for bit in ones), f"0{quantum._counts_width(rec)}b")


def selector_ones(rec):
    return [bit for bit, value in rec.success + rec.conditions if value]


def test_complex_three_coordinate_native_uses_original_svd_once(monkeypatch):
    original = np.linalg.svd
    seen = []

    def svd(matrix, *args, **kwargs):
        if matrix.shape == (3, 3) and np.array_equal(matrix, MATRIX/2):
            seen.append("original")
        elif matrix.shape == (4, 4) and any(np.array_equal(matrix[:3, :3], value)
                for value in (MATRIX, MATRIX/2)):
            raise AssertionError("padded SVD manufactured scale")
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", svd)
    selected = choose()
    assert selected.reconstruction.width == 6
    result = nwqlib.solve(selected)
    expected = np.linalg.solve(MATRIX, RHS)
    assert np.linalg.norm(result.x - expected) / np.linalg.norm(expected) < 0.01
    assert seen == ["original"]
    base = selected._native["base"]
    assert base.svd is selected._native["svd"]
    base.validate()
    wrong = replace(base, original=ingest_dense(MATRIX + np.eye(3)))
    with pytest.raises(ValueError, match="original input binding"):
        wrong.validate()
    wrong_source = replace(base.selected._payload, source=np.eye(4))
    wrong = replace(
        base,
        selected=SelectedBlock.bind(
            base.selected.record, payload=wrong_source, constructor=base.selected._constructor
        ),
    )
    with pytest.raises(ValueError, match="exact selected input"):
        wrong.validate()


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


@pytest.mark.parametrize("A", [
    np.array([[1.1, 0.1], [0.1, 0.9]]),  # Hermitian inverse route
    np.array([[1.5, 0.4], [-0.2, 1.1]]),  # Hermitian dilation of a general A
], ids=["hermitian", "general"])
def test_quantum_solution_and_scalars_keep_physical_scale_sign_and_phase(A):
    # Method-error-bounded class. Automatic dense selection puts every eigenvalue or
    # singular value of A/alpha in the polynomial domain, so the relative error of x is
    # at most epsilon_inv (docs/algorithms/qls.md, Inverse polynomial accuracy). The
    # added 1e-9 allows for the phase-fit term, about 1e-15, and statevector rounding.
    epsilon = QLS().epsilon_inv
    allowance = epsilon + 1e-9
    b0 = np.array([1.0, 0.25])
    # RHS scaling b -> c b gives x -> c x, including a negative sign and a complex phase.
    for c in (-3.0, 2.5 * np.exp(1j * np.pi / 3), 1e-3j):
        x = nwqlib.solve(LinearSystem(A=A, b=c * b0), method=QLS(), seed=7).x
        assert _relative_error(x, np.linalg.solve(A, c * b0)) <= allowance
    # Simultaneous scaling (A, b) -> (c A, c b) leaves x unchanged. A negative c keeps a
    # Hermitian A Hermitian, so each instance stays on its route.
    for c in (-2.0, 1e-4, 1e4):
        x = nwqlib.solve(LinearSystem(A=c * A, b=c * b0), method=QLS(), seed=7).x
        assert _relative_error(x, np.linalg.solve(A, b0)) <= allowance
    # Physical scalars of x for a complex b. With ||x_hat - x|| <= epsilon ||x||, the
    # quadratic form differs by at most ||O|| (2 epsilon + epsilon**2) ||x||**2, and the
    # normalized expectation by 2 ||O|| epsilon / (1 - epsilon) (docs/mathematics.md,
    # Proposition 30, with delta = epsilon ||x|| and ||x_hat|| >= (1 - epsilon) ||x||).
    b = 2.5 * np.exp(1j * np.pi / 3) * b0
    x = np.linalg.solve(A, b)
    O = np.array([[0.3, 0.1 - 0.2j], [0.1 + 0.2j, -0.2]])
    norm_O, mass = np.linalg.norm(O, 2), float(np.vdot(x, x).real)
    quadratic = float(np.vdot(x, O @ x).real)
    cases = (
        (QuadraticForm(observable=O), quadratic, norm_O * (2 * epsilon + epsilon**2) * mass),
        (NormSquared(), mass, (2 * epsilon + epsilon**2) * mass),
        (NormalizedExpectation(observable=O), quadratic / mass, 2 * norm_O * epsilon / (1 - epsilon)),
    )
    for output, expected, bound in cases:
        value = nwqlib.solve(LinearSystem(A=A, b=b), method=QLS(), output=output, seed=7).value
        assert abs(value - expected) <= bound + 1e-9 * abs(expected)


def test_compared_accuracy_settings_solve_within_their_own_inverse_tolerance():
    from nwqlib.resources import ResourceContext

    A, b = np.array([[1.1, 0.1], [0.1, 0.9]]), np.array([1.0, 0.25])
    exact = np.array([25 / 28, 5 / 28])  # Closed form in docs/algorithms/qls.md.
    epsilons = (0.1, 0.01, 0.001)
    comparison = nwqlib.compare(LinearSystem(A=A, b=b), methods=tuple(QLS(epsilon_inv=e) for e in epsilons),
                                seed=7, context=ResourceContext(basis="cx"))
    degrees, cx = [], []
    for row, epsilon in zip(comparison.rows, epsilons, strict=True):
        result = nwqlib.solve(row)
        assert result.plan.content_id == row.plan.content_id
        # Method-error-bounded class, as in the scaling test above: the relative error is
        # at most the row's own epsilon_inv, whose higher-degree phases run on Aer.
        assert _relative_error(result.x, exact) <= epsilon + 1e-9
        degrees.append(row.plan.reconstruction.degree)
        cx.append(row.estimate.quantity("cx").fact.value.value)
    # docs/algorithms/qls.md: a smaller epsilon_inv raises the degree and the CX count.
    # Only the ordering is asserted, because CX counts depend on the Qiskit version.
    assert degrees == sorted(degrees) and degrees[0] < degrees[-1]
    assert cx == sorted(cx) and cx[0] < cx[-1]


@pytest.mark.parametrize("solver", ["shortcut_native_svp"])
def test_original_three_coordinate_shortcut_matches_padded_native_model(solver):
    selected = choose(solver=solver)
    assert selected.reconstruction.width in (8, 9)
    result = nwqlib.solve(selected)
    classical = nwqlib.solve(
        selected.problem,
        method=selected.method,
        output=selected.output,
        execution="classical",
        seed=7,
    )
    np.testing.assert_allclose(
        aligned(result.value, classical.value), classical.value, rtol=1e-11, atol=1e-11
    )
    assert result.algorithm_success_mass == pytest.approx(
        classical.algorithm_success_mass, abs=2e-11
    )
    assert result.value.shape == (3,) and result.physical_scale is None


@pytest.mark.parametrize("family", ["multiplexed_pauli", "compact_pauli"])
def test_planned_encoding_archive_keeps_constructor_phases_and_frames(
    tmp_path, monkeypatch, family
):
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    if family == "compact_pauli":
        # 2 I - X keeps Pauli access; its decomposition is the saved operator's
        # own term table, so the archive names problem.A instead of copying it.
        from nwqlib.operators import ingest_pauli

        selected = choose(A=ingest_pauli((("I", 2.0), ("X", -1.0)), num_qubits=1), b=[1.0, 1j], kappa=3.0)
    else:
        # I, X, Y and Z terms, so the saved packed table exercises every label letter.
        A = np.array([[2.0, -1.0 + 0.5j], [-1.0 - 0.5j, 1.5]])
        selected = choose(A=A, b=[1.0, 1j], block_encoding_implementation=family)
    saved = save_plan(selected, ArchiveFiles(tmp_path, 4_000_000))
    decomposition = saved["selected"]["encoding"].get("decomposition")
    assert (decomposition is not None and decomposition.get("reference") == "problem.A") == (
        family == "compact_pauli")
    monkeypatch.setattr(QLS, "plan", forbidden)
    monkeypatch.setattr(owner, "select_polynomial", forbidden)
    monkeypatch.setattr("nwqlib.subroutines.block_encoding.core.plan_block_encoding", forbidden)
    monkeypatch.setattr("nwqlib.subroutines.qsp.phases.solve_symmetric_qsp_phases", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    loaded = load_plan(saved, ArchiveFiles(tmp_path, 4_000_000))
    assert loaded.content_id == selected.content_id
    assert loaded._native["encoding"].record == selected._native["encoding"].record
    # The saved Pauli table decodes to the selected labels and coefficients.
    assert loaded._native["encoding"]._payload.decomposition == selected._native["encoding"]._payload.decomposition
    assert loaded.reconstruction.phase_solution == selected.reconstruction.phase_solution
    result = nwqlib.solve(loaded)
    expected = np.linalg.solve(A, [1.0, 1j])
    np.testing.assert_allclose(result.x, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("q", [1, 2, 3])
def test_periodic_stencil_and_product_rhs_solve_like_their_dense_arrays(q):
    from nwqlib.operators import PeriodicStencil, operator_input
    from nwqlib.problems import ingest_product

    # The PeriodicStencil operator m I + nu (2I - S - S^T) and the product state, built
    # densely with qubit 0 as the least significant index bit. The product b has no
    # coordinate symmetry, so a misplaced qubit changes the solution. At q = 1 both
    # shifts are X, so A = (m + 2 nu) I - 2 nu X and the two wraps share one coefficient.
    mass, diffusion = 0.5, 0.25
    pairs = [[np.cos(0.3 + 0.4 * j), np.sin(0.3 + 0.4 * j) * np.exp(0.2j * j)] for j in range(q)]
    b = np.ones(1)
    for pair in pairs:
        b = np.kron(pair, b)
    identity = np.eye(2**q)
    shift = np.roll(identity, 1, axis=0)
    dense = mass * identity + diffusion * (2 * identity - shift - shift.T)
    expected = np.linalg.solve(dense, b)
    compact = LinearSystem(A=operator_input(PeriodicStencil(q, mass=mass, diffusion=diffusion)),
                           b=ingest_product(pairs))
    # Method-error-bounded class: the analytic endpoints m and m + 4 nu put the spectrum
    # of A/alpha in the polynomial domain, so the relative error is at most epsilon_inv
    # (docs/algorithms/qls.md, Inverse polynomial accuracy). The added 1e-9 allows for
    # the phase-fit term and statevector rounding.
    allowance = QLS().epsilon_inv + 1e-9
    for problem in (compact, LinearSystem(A=dense, b=b)):
        result = nwqlib.solve(problem, method=QLS(), seed=7)
        assert result.plan.reconstruction.width <= 11
        assert _relative_error(result.x, expected) <= allowance


def test_two_qubit_pauli_a_solves_like_its_matrix():
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.operators import ingest_pauli

    # Labels that differ under qubit reversal; qubit 0 is the rightmost letter.
    terms = (("II", 2.0), ("ZX", 0.5), ("XI", 0.3), ("IY", 0.2))
    matrix = SparsePauliOp.from_list(list(terms)).to_matrix()
    b = np.array([1.0, 0.5j, -0.25, 0.1])
    expected = np.linalg.solve(matrix, b)
    # QLS computes no singular values of a Pauli A, so the user supplies
    # K >= alpha/sigma_min(A), with alpha the coefficient one-norm.
    alpha = sum(abs(c) for _, c in terms)
    kappa = 1.0001 * alpha / np.linalg.svd(matrix, compute_uv=False)[-1]
    # Method-error-bounded class: K puts the spectrum of A/alpha in the polynomial domain,
    # so the relative error is at most epsilon_inv. The added 1e-9 allows for the
    # phase-fit term and statevector rounding.
    allowance = QLS().epsilon_inv + 1e-9
    for A in (ingest_pauli(terms, num_qubits=2), SparsePauliOp.from_list(list(terms))):
        x = nwqlib.solve(LinearSystem(A=A, b=b), method=QLS(kappa=kappa), seed=7).x
        assert _relative_error(x, expected) <= allowance


def test_supplied_complex_global_phase_survives_control_inverse_and_archive(tmp_path, monkeypatch):
    """A supplied unitary has inverse A-dagger, exposing lost global phase through control,
    adjoint and archive paths.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    circuit = QuantumCircuit(1)
    circuit.ry(0.4, 0)
    circuit.global_phase = 0.37
    matrix = Operator(circuit).data
    operator = ingest_dense(matrix)
    encoding = select_block_encoding(
        "supplied",
        BlockEncoding(
            circuit=circuit,
            alpha=1.0,
            num_ancillas=0,
            system_qubits=1,
            error_bound=0.0,
            implementation="supplied",
            metadata={},
        ),
        operator=operator,
    )
    selected = choose(A=operator, b=[1.0, 1j], encoding=encoding)
    assert selected.reconstruction.width == 4
    assert selected.reconstruction.alpha == 1.
    result = nwqlib.solve(selected)
    expected = matrix.conj().T @ np.array([1.0, 1j])
    np.testing.assert_allclose(result.x, expected, rtol=0.01, atol=0.01)
    result.save(tmp_path / "result")
    monkeypatch.setattr(QLS, "plan", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr("nwqlib.subroutines.qsp.phases.solve_symmetric_qsp_phases", forbidden)
    loaded = nwqlib.load_result(tmp_path / "result")
    np.testing.assert_array_equal(loaded.x, result.x)
    assert loaded.plan.method.encoding is loaded.plan._native["encoding"]
    assert loaded.plan.method.encoding._payload.circuit == encoding._payload.circuit


def test_padded_normalized_counts_take_the_prefix_mass_from_an_unrotated_later_group(monkeypatch):
    """With a rotated first group, the unrotated second group supplies the physical-prefix mass.

    The zero-extended observable is 0.5 (IZ + ZZ + IX + ZX). Each group sees
    r selected shots at original coordinate 0, three at original coordinate
    1 and two at the dummy coordinate 3. Every label of either group has
    parity +1 at coordinate 0 and -1 at coordinate 1, and the two labels of
    each group have opposite parities at coordinate 3, so each unconditional
    group mean is (r - 3)/(r + 5), and the physical-prefix fraction of the
    unrotated group is (r1 + 3)/(r1 + 5). Coordinate 1 is decoded from the
    measured positions of the compact register, not from the physical wires.
    """
    from nwqlib.backends import qiskit_aer

    observable = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    selected = choose(output=NormalizedExpectation(observable=observable), shots=12)
    rec = selected.reconstruction
    assert [s.label for s in rec.settings] == ["ZX", "ZZ"] and rec.groups == (("IX", "ZX"), ("IZ", "ZZ"))
    calls = []

    def acquire(prepared, **kwargs):
        returned = 5 + len(calls)
        setting = rec.settings[len(calls)]
        calls.append(returned)
        counts = {compact_key(rec, setting, selector_ones(rec)): returned,
                  compact_key(rec, setting, selector_ones(rec) + [rec.coordinates[0]]): 3,
                  compact_key(rec, setting, selector_ones(rec) + list(rec.coordinates)): 2}
        return SimpleNamespace(raw_output={"counts": counts}, metadata={"native_job_id": str(returned)})

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", acquire)
    result = nwqlib.solve(selected)
    r0, r1 = calls
    mass = next(c for c in result.data.observations.chunks if c.experiment == "group_1")
    assert result.mass_contribution_id == mass.content_id and result.returned_shots == r1 + 5
    assert result.algorithm_success_mass == 1.0 and result.physical_slice_mass == (r1 + 3) / (r1 + 5)
    assert result.value == pytest.approx(((r0 - 3) / (r0 + 5) + (r1 - 3) / (r1 + 5)) / ((r1 + 3) / (r1 + 5)),
                                         rel=1e-14, abs=0)


def test_sampled_register_position_reads_the_wire_its_layout_names():
    """Each position of a compact count register holds the physical wire ``quantum._counts_bits`` names.

    With diagonal A and b = e_2 the solution is a multiple of e_2, so every
    selected shot has coordinate bit 1 set and the ``ZI`` parity is -1 on each
    of them. The ``ZI`` group measures its selectors and coordinate bit 1
    only, a register that is not a prefix of the physical wires, so a
    position that measured another wire would publish a different value.
    """
    from nwqlib.operators import ingest_pauli

    selected = choose(A=np.diag([1.0, 1.5, 2.0, 2.5]), b=[0.0, 0.0, 1.0, 0.0], shots=200,
                      output=NormalizedExpectation(observable=ingest_pauli((("ZI", 1.0),), num_qubits=2)))
    rec = selected.reconstruction
    observed = quantum._counts_bits(rec, rec.settings[0])
    assert observed != tuple(range(len(observed))) and observed[-1] == rec.coordinates[1]
    result = nwqlib.solve(selected)
    assert result.groups[0].selected_shots > 0
    assert result.value == -1.0


def _reference_slice(selected):
    """The bound Plan circuit in Statevector, with the physical slice by explicit bit enumeration."""
    from qiskit.quantum_info import Statevector
    from nwqlib.blocks.lowering import lower_qiskit

    _, construction = selected.resolve("projected_moments")._selected_construction(selected)
    chosen = {record.content_id for record in construction.selections}
    circuit = lower_qiskit(construction, blocks=tuple(
        block for block in selected.blocks if block.record.content_id in chosen)).circuit
    state = Statevector(circuit).data
    rec = selected.reconstruction
    algorithm = physical = 0.0
    vector = np.zeros(1 << len(rec.coordinates), dtype=complex)
    for index, amplitude in enumerate(state):
        if any((index >> bit) & 1 != value for bit, value in rec.success):
            continue
        algorithm += abs(amplitude) ** 2
        if any((index >> bit) & 1 != value for bit, value in rec.conditions):
            continue
        coordinate = sum(((index >> bit) & 1) << place for place, bit in enumerate(rec.coordinates))
        if coordinate < rec.original_dimension:
            vector[coordinate] = amplitude
            physical += abs(amplitude) ** 2
    return vector, algorithm, physical


def _pauli_matrix(label):
    one = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
           "Z": np.diag([1, -1])}
    matrix = np.eye(1)
    for axis in label:  # qubit zero is the rightmost letter, the last Kronecker factor
        matrix = np.kron(matrix, one[axis])
    return matrix


@pytest.mark.parametrize("case", ["padded_inverse", "unpadded_pauli", "non_hermitian", "shortcut_dilation"])
def test_exact_qls_outputs_match_an_independent_selected_circuit_reference(case):
    """Projected p, p_alg and q against the same bound circuit in Statevector.

    The slice uses the actual success, condition and coordinate maps, and
    each Pauli matrix is built independently from its label. The receipts
    exclude supplied-state operations, so this uses the unit-scale regression
    contract atol=2e-12, rtol=0 for raw masses and q with C <= 1, propagated
    to the normalized value by the quotient rule
    ``[Q + (|q|/p) P]/(p - P) + u |q_hat/p_hat|`` and to physical outputs by
    the recovery. It is not a native error certificate. The unpadded case
    gives a Pauli observable as input, and its terms are read back by label.
    """
    from nwqlib._quantum_readout import pair_value
    from nwqlib.operators import ingest_pauli

    dense = np.diag([0.4, -0.2, 0.1]) + 0.05 * (np.eye(3, k=1) + np.eye(3, k=-1))
    A, b, solver, observable = MATRIX, RHS, "qsvt_inverse", dense
    if case == "unpadded_pauli":
        A, b = np.diag([1.0, -2.0, 1.5, 0.8]), [1.0, 0.5, 0.25, -0.5]
        observable = ingest_pauli((("II", 0.1), ("ZX", -0.3), ("XI", 0.25), ("YY", 0.2), ("IZ", 0.0)),
                                  num_qubits=2)
    elif case == "non_hermitian":
        A, b = np.array([[1.5, 0.4], [-0.2, 1.1]]), [1.0, 0.5]
        observable = np.array([[0.3, 0.1], [0.1, -0.2]])
    elif case == "shortcut_dilation":
        solver = "shortcut_dilation"
    outputs = [NormalizedExpectation(observable=observable)]
    if solver == "qsvt_inverse":
        outputs += [QuadraticForm(observable=observable), NormSquared()]
    tolerance = 2e-12
    for output in outputs:
        selected = choose(A=A, b=b, solver=solver, output=output)
        result = nwqlib.solve(selected)
        rec = selected.reconstruction
        if case == "non_hermitian":
            assert rec.conditions == ((rec.coordinates[-1] + 1, 1),)
        if case == "shortcut_dilation":
            assert (rec.width - 1, 1) in rec.success
        vector, algorithm, physical = _reference_slice(selected)
        (point,) = selected.experiments[0].observation.positions
        import json
        terms = json.loads(point.parameters)["terms"]
        assert sum(abs(c) for _, c in terms) <= 1
        matrix = sum((c * _pauli_matrix(label) for label, c in terms), np.zeros((len(vector),) * 2))
        numerator = float(np.vdot(vector, matrix @ vector).real)
        if isinstance(observable, np.ndarray) and not isinstance(output, NormSquared):
            padded = np.zeros((len(vector),) * 2)
            padded[:len(observable), :len(observable)] = observable
            assert np.abs(matrix - padded).max() < 1e-15
        elif not isinstance(output, NormSquared):
            rows = (("II", 0.1), ("ZX", -0.3), ("XI", 0.25), ("YY", 0.2))
            assert np.abs(matrix - sum(c * _pauli_matrix(label) for label, c in rows)).max() < 1e-15
        stats = result.reduction
        p, q = float(pair_value(stats.physical_mass)), float(pair_value(stats.numerator))
        assert abs(p - physical) <= tolerance and abs(float(pair_value(stats.success_mass)) - algorithm) <= tolerance
        assert abs(q - numerator) <= tolerance
        assert physical > 0.1
        assert result.physical_slice_mass == p and result.algorithm_success_mass == float(
            pair_value(stats.success_mass))
        if rec.recovery is not None:
            # The published scale is Gamma*sqrt(p), with the root split by the
            # parity of p's binary exponent (even on the padded fixture, odd on
            # the unpadded and non-Hermitian ones).
            # Its square is Gamma**2 times the reference physical mass.
            gamma2 = rec.recovery.as_float() ** 2
            assert abs(result.physical_scale.squared_as_float() - gamma2 * physical) <= (
                gamma2 * tolerance * (1 + 1e-12))
        if isinstance(output, NormalizedExpectation):
            bound = (tolerance + abs(numerator) / physical * tolerance) / (physical - tolerance) + 2**-53 * abs(result.value)
            assert abs(result.value - numerator / physical) <= bound
            continue
        gamma2 = rec.recovery.as_float() ** 2
        target = numerator if isinstance(output, QuadraticForm) else physical
        assert abs(result.value - gamma2 * target) <= gamma2 * tolerance * (1 + 1e-12)


def test_grouped_sampled_setting_shares_one_count_table_within_its_sampling_radius():
    """Two qubit-wise commuting terms share one counts setting with ``shots`` returned shots.

    With C_g = sum |c_j| over the group and n_g its selected shots, the
    estimate lies within ``r = sqrt(2 log(2/delta) sum_g C_g**2/n_g)`` of the
    ideal value with probability at least 1 - delta; the fixed seed makes
    this a deterministic regression at delta = 1e-6. The classical model
    evaluates the same selected polynomial.
    """
    from math import log, sqrt as root
    from nwqlib.operators import ingest_pauli

    rows = (("ZI", 0.3), ("ZZ", -0.2), ("II", 0.1))
    observable = ingest_pauli(rows, num_qubits=2)
    output = NormalizedExpectation(observable=observable)
    A, b = np.diag([1.0, -2.0, 1.5, 0.8]), [1.0, 0.5, 0.25, -0.5]
    selected = choose(A=A, b=b, output=output, shots=4000)
    assert [e.name for e in selected.experiments] == ["group_0"]
    assert selected.reconstruction.groups == (("ZI", "ZZ"),)
    assert selected.reconstruction.grouping_comparisons == 1
    result = nwqlib.solve(selected)
    (group,) = result.groups
    assert group.returned_shots == 4000 and group.population == "success_conditional"
    assert result.data.observations.chunks[0].returned_shots == 4000
    classical = nwqlib.solve(selected.problem, method=selected.method, output=NormalizedExpectation(
        observable=sum(c * _pauli_matrix(label) for label, c in rows).real), execution="classical", seed=7)
    radius = root(2 * log(2 / 1e-6) * 0.5**2 / group.selected_shots)
    assert abs(result.value - classical.value) <= radius


def test_samples_follow_the_solution_distribution_in_original_coordinates():
    from math import log, sqrt
    from nwqlib import Samples

    # b = A x_t fixes the exact outcome distribution |x_i|**2/||x||**2 = (.6, .3, .1) of a
    # general, padded system, so any exchange of coordinates moves it by at least .2.
    A = np.array([[1.0, 0.2, 0.0], [0.1, 1.5, 0.3], [0.0, -0.2, 2.0]])
    x = np.array([np.sqrt(0.6), 1j * np.sqrt(0.3), -np.sqrt(0.1)])
    result = nwqlib.solve(LinearSystem(A=A, b=A @ x), method=QLS(), output=Samples(), shots=4000, seed=11)
    indices, counts = result.samples.indices.array, result.samples.counts.array
    assert set(indices) <= {0, 1, 2}
    n = int(counts.sum())
    assert n == result.physical_selected_shots
    empirical = np.zeros(3)
    empirical[indices] = counts / n
    # Statistical class with a fixed seed. Given the n selected shots, the
    # Bretagnolle-Huber-Carol inequality bounds the total variation distance by
    # sqrt((3 log 2 + log(1/delta))/(2 n)) except with probability delta = 1e-6. The
    # computed direction is within 2 epsilon_inv of x/||x|| in L2, which bounds the total
    # variation distance between the two outcome distributions by 2 epsilon_inv.
    radius = sqrt((3 * log(2) + log(1e6)) / (2 * n)) + 2 * QLS().epsilon_inv
    assert radius < 0.2
    assert 0.5 * np.abs(empirical - [0.6, 0.3, 0.1]).sum() <= radius


def test_run_archive_keeps_already_constructed_queries(tmp_path, monkeypatch):
    selected = choose(A=np.array([[1.0, 0.1j], [0.2, 2.0]]), b=[1.0, 1j])
    prepared = nwqlib.prepare(selected)
    result = nwqlib.submit(prepared).wait()
    run = prepared.run
    keys = tuple(run._state["method_context"]["qls_base_gates"])
    assert keys
    run.save(tmp_path / "run")
    monkeypatch.setattr(quantum, "_native_encoding", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    loaded = nwqlib.load_run(tmp_path / "run", backend=run.backend)
    assert tuple(loaded._state["method_context"]["qls_base_gates"]) == keys
    again = loaded.resume().wait()
    np.testing.assert_array_equal(again.x, result.x)


@pytest.mark.parametrize(
    "solver,general,a_queries,a_controls,rhs_queries",
    [
        ("qsvt_inverse", False, 1, 0, 0),
        ("qsvt_inverse", True, 2, 1, 0),
        ("shortcut_native_svp", False, 1, 1, 2),
        ("shortcut_dilation", False, 2, 2, 4),
    ],
)
def test_actual_query_projector_and_phase_populations_match_independent_counts(
    solver, general, a_queries, a_controls, rhs_queries
):
    """Walk actual query calls and compare solver-specific multiplicities, controls and phase
    counts with independent expectations.
    """
    from nwqlib.ir import BlockCall, Sequence

    selected = choose(
        A=np.array([[1.0, 0.1], [0.0, 0.5]]) if general else np.diag([1.0, 0.5]),
        b=[1.0, 1j],
        solver=solver,
        epsilon_inv=0.1,
        block_encoding_implementation="dense_dilation",
    )
    r = selected.reconstruction
    program = selected.construction.program
    definitions = {d.id: d.node for d in program.definitions}
    expressions = {e.id: e.value.value for e in program.expressions}
    calls = []

    def walk(name):
        node = definitions[name]
        if isinstance(node, Sequence):
            for child in node.children:
                walk(child)
        elif isinstance(node, BlockCall):
            calls.append(
                (
                    node.signature,
                    {
                        a.parameter: getattr(
                            expressions[a.value.expression],
                            "value",
                            expressions[a.value.expression],
                        )
                        for a in node.arguments
                    },
                )
            )

    walk(program.root)
    query = next(s for s in selected.construction.selections if s.signature.name.startswith("original_A"))
    assert query.semantics.base_semantics == selected._native["encoding"].record.semantics
    aq = [row for row in calls if row[0].startswith("original_A_query")]
    rhs = [row for row in calls if row[0].startswith("rhs_query")]
    assert len(aq) == a_queries * r.degree and all(row[0].endswith(f"_c{a_controls}") for row in aq)
    assert len(rhs) == rhs_queries * r.degree
    assert sum(name.startswith("rz_") for name, _ in calls) == r.degree + 1
    assert sum(name.startswith("global_phase") for name, _ in calls) == 1
    if general or solver == "shortcut_dilation":
        assert sum(args["adjoint"] for _, args in aq) == r.degree
        assert {args["control_state"] for _, args in aq} == {
            0,
            1,
        }  # outer dilation bit precedes fixed-zero augmentation bit
    else:
        assert sum(args["adjoint"] for _, args in aq) == r.degree // 2
    report = nwqlib.estimate(selected)
    count = report.quantity("calls").fact.value
    assert count.numerator / count.denominator == len(calls)


def _supplied_child(circuit, *, ancillas, alpha, error):
    """Select a supplied child encoding together with its declared original operator."""
    from qiskit.quantum_info import Operator

    step = 1 << ancillas
    operator = ingest_dense(alpha * Operator(circuit).data[::step, ::step])
    encoding = select_block_encoding(
        "child",
        BlockEncoding(
            circuit=circuit,
            alpha=alpha,
            num_ancillas=ancillas,
            system_qubits=circuit.num_qubits - ancillas,
            error_bound=error,
            implementation="supplied",
            metadata={},
        ),
        operator=operator,
    )
    return operator, encoding


def _lowered_forward_query_block(selected):
    """Lower only the selected forward query of the actual Program and return its encoded block."""
    from qiskit.quantum_info import Operator
    from nwqlib.blocks.lowering import lower_qiskit
    from nwqlib.blocks.records import SelectedConstruction
    from nwqlib.ir import Definition, Sequence as IRSequence
    from nwqlib.subroutines.block_encoding import block_encoding_top_left

    program = selected.construction.program
    allocations = tuple(d.id for d in program.definitions if d.id.startswith("allocate_"))
    root = Definition(
        id="forward_query_only", node=IRSequence(children=(*allocations, "encoding_forward"))
    )
    query = SelectedConstruction(
        program=program.revise(root=root.id, definitions=program.definitions + (root,)),
        selections=selected.construction.selections,
    )
    unitary = Operator(lower_qiskit(query, blocks=selected.blocks).circuit).data
    # Pair, signal, child and shortcut ancillas are the low-order wires below the system.
    return block_encoding_top_left(unitary, num_ancillas=selected.reconstruction.coordinates[0])


def test_shortcut_queries_keep_child_ancillas_complex_block_and_error_premise():
    """The selected G_t query and its dilation act on a supplied complex child as in Dalzell
    arXiv:2406.12086v2, App. A; the expected blocks come from the child's own unitary.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    a, n = 1, 1
    child = QuantumCircuit(a + n)
    child.global_phase = 0.19
    child.rx(0.51, a)
    child.rz(0.2, a)
    child.ry(0.6, 0)
    child.cx(0, 1)
    child.ry(0.41, 0)
    encoded = Operator(child).data[:: 1 << a, :: 1 << a]
    operator, encoding = _supplied_child(child, ancillas=a, alpha=2.0, error=0.02)
    rhs = np.zeros(1 << n, dtype=complex)
    rhs[:2] = [np.sqrt(0.3), 1j * np.sqrt(0.7)]
    # A_t = (A/alpha) (+) |0><0|/t, b' = (b, 1)/sqrt(2) and G_t = (I - |b'><b'|) A_t,
    # with t = 1.5 from choose().
    augmented = np.zeros((2 << n, 2 << n), dtype=complex)
    augmented[: 1 << n, : 1 << n] = encoded
    augmented[1 << n, 1 << n] = 1 / 1.5
    b_prime = np.zeros(2 << n, dtype=complex)
    b_prime[: 1 << n] = rhs / np.sqrt(2)
    b_prime[1 << n] = 1 / np.sqrt(2)
    g_t = (np.eye(2 << n) - np.outer(b_prime, b_prime.conj())) @ augmented
    # G_t is not normal, so a projector applied on the wrong side changes the block.
    assert np.max(np.abs(g_t.conj().T @ g_t - g_t @ g_t.conj().T)) >= 1.0e-2
    zero = np.zeros_like(g_t)
    targets = {
        "shortcut_native_svp": g_t,
        "shortcut_dilation": np.block([[zero, g_t], [g_t.conj().T, zero]]),
    }
    for solver, target in targets.items():
        # kappa=4 bounds the child's condition number, which is at most 1.15 here.
        selected = choose(
            A=operator, b=rhs, solver=solver, encoding=encoding, kappa=4.0, epsilon_inv=0.1
        )
        rec = selected.reconstruction
        # Wire order: pair and signal, the child ancillas, two shortcut ancillas, the
        # system, the augmentation bit and, for the dilation, one more bit.
        assert rec.encoding_ancillas == a
        assert rec.coordinates == tuple(range(a + 4, a + 4 + n))
        assert rec.width == a + 5 + n + (solver == "shortcut_dilation")
        # The child's alpha and original-A error premise reach every query unchanged.
        assert rec.alpha == 2.0 and rec.encoding_error == 0.02
        premises = {
            (s.semantics.base_semantics.alpha, s.semantics.base_semantics.epsilon)
            for s in selected.construction.selections
            if s.signature.name.startswith("original_A_query")
        }
        assert premises == {(2.0, 0.02)}
        # Binary64 products of these few dozen unit-norm gates stay near 1e-14. A wrong
        # projector, control value, phase or wire moves entries by O(0.1).
        np.testing.assert_allclose(
            _lowered_forward_query_block(selected), target, rtol=0, atol=2.0e-12
        )


@pytest.mark.parametrize(
    ("solver", "hermitian", "route"),
    [("qsvt_inverse", False, "gatewise"), ("qsvt_inverse", False, "auto"), ("shortcut_native_svp", False, "auto"),
     ("shortcut_dilation", False, "auto"), ("shortcut_dilation", False, "whole_matrix"), ("qsvt_inverse", True, "auto")],
)
def test_controlled_dense_query_synthesis_is_admitted_at_planning(solver, hermitian, route):
    """Refuse the exact synthesis of a controlled dense query, and Qiskit's control of it, before the original SVD.

    A controlled query of the dense dilation of a 16-dimensional system
    synthesizes its unitary on log2(16) + 1 = 5 qubits, once for the forward
    and once for the adjoint specialization, and Qiskit then controls the
    synthesized gates with one control, or two for ``shortcut_dilation``.
    On the whole-matrix route, which "auto" takes for one control, each
    specialization instead synthesizes the controlled matrix on five qubits
    plus the controls. Planning charges twice the laws of the route. The
    Hermitian inverse queries the dilation without controls, and its
    construction synthesizes nothing.
    """
    from test_dense_synthesis import _dense_synthesis_work
    from test_synthesis_admission import _controlled_synthesis_work, _gatewise_control_work

    rng = np.random.default_rng(4)
    A = np.eye(16) + 0.2 * rng.normal(size=(16, 16))
    if hermitian:
        A = (A + A.T) / 2
    controls = 2 if solver == "shortcut_dilation" else 1
    if route == "whole_matrix" or (route == "auto" and controls == 1):
        need = 2 * _controlled_synthesis_work(5 + controls)
    else:
        need = 2 * (_dense_synthesis_work(5) + _gatewise_control_work(5, controls))

    def planned(max_work):
        return choose(A=A, b=np.ones(16), solver=solver, epsilon_inv=0.1,
                      block_encoding_implementation="dense_dilation", max_work=max_work,
                      dense_control_route=route)

    if hermitian:
        planned(need - 1)
        return
    with pytest.raises(ValueError, match="QLS controlled dense-query synthesis exceeds max_work"):
        planned(need - 1)
    planned(need)


@pytest.mark.parametrize(("general", "syntheses", "route"), [(True, 2, "gatewise"), (True, 2, "auto"),
                                                            (False, 0, "auto")])
def test_dense_query_syntheses_of_a_run_match_the_planning_charge(monkeypatch, general, syntheses, route):
    """Count the exact syntheses and control steps of a whole inverse Run against the two of each that planning charges.

    For the general A the controlled query of the dense dilation acts on
    two qubits with one control, and "auto" synthesizes the controlled
    matrix on three qubits instead. The Hermitian A is queried without a
    control and synthesizes nothing. The shortcut solvers, whose two readout
    settings reuse the specializations of the first preparation, are
    counted in ``test_synthesis_admission.py``.
    """
    from test_dense_synthesis import _counted_syntheses
    from test_synthesis_admission import _counted_dense_controls

    selected = choose(A=np.array([[1.0, 0.1], [0.0, 0.5]]) if general else np.diag([1.0, 0.5]), b=[1.0, 1j],
                      epsilon_inv=0.1, block_encoding_implementation="dense_dilation", dense_control_route=route)
    calls = _counted_syntheses(monkeypatch)
    controls = _counted_dense_controls(monkeypatch)
    nwqlib.solve(selected, progress=False)
    assert calls == [3 if route == "auto" else 2] * syntheses
    assert controls == [(1, (2,))] * syntheses
