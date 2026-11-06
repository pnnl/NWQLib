"""Tests for the block-encoding subroutines.

The embedding-identity test checks the convention in docs/conventions.md,
"Block encodings and QSP": for every construction and
registered instance shape, the circuit's all-zero-ancilla block reproduces
``A / alpha`` at 1e-12. No wall-clock assertions anywhere; integer anchors are
construction-exact quantities only (term counts, ancilla counts, gate counts
of the built circuit).
"""

from __future__ import annotations


import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator, SparsePauliOp


from nwqlib.subroutines.block_encoding import (
    BandSpecification,
    block_encoding_top_left,
    build_block_encoding,
    build_block_encoding_from_plan,
)
from nwqlib.subroutines._multiplexors import (
    append_dependency_projected_unitaries,
    projected_unitary_resource_law,
)
from nwqlib.backends.resources import (
    block_encoding_per_query_cx,
    dense_unitary_cx_qsd_upper_bound,
)
from nwqlib.subroutines.block_encoding import plan_block_encoding
from nwqlib.subroutines.qiskit_compat import controlled, inverse_realized_gate


def _second_difference(num_qubits,*,diffusivity=1.,spacing=1.,boundary="periodic"):
    size=1<<num_qubits
    shift=np.roll(np.eye(size),1,axis=0) if boundary=="periodic" else np.eye(size,k=1)
    return diffusivity/spacing**2*(2*np.eye(size)-shift-shift.T)


def _dense_from_bands(specification: BandSpecification) -> np.ndarray:
    """Materialize the circulant a band specification describes (test oracle)."""

    dimension = 2**specification.num_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    columns = np.arange(dimension)
    for offset, coefficient in zip(specification.offsets, specification.coefficients):
        matrix[(columns + offset) % dimension, columns] = coefficient
    return matrix


def _encoded_block(record) -> np.ndarray:
    """Return the all-zero-ancilla block of the built circuit's unitary."""

    unitary = Operator(record.circuit).data
    return block_encoding_top_left(unitary, num_ancillas=record.num_ancillas)


def _hermitian_2x2() -> np.ndarray:
    return np.array([[1.0, 2.0 - 1.0j], [2.0 + 1.0j, -1.0]], dtype=complex)


def _random_dense(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(size=(dimension, dimension))


@pytest.mark.parametrize("control_state", (0,))
def test_realized_inverse_preserves_open_control_and_relative_phase(monkeypatch, control_state) -> None:
    from qiskit.circuit.library import UnitaryGate

    unitary = np.exp(0.23j) * np.asarray([[np.cos(0.31), -np.sin(0.31)], [np.sin(0.31), np.cos(0.31)]])
    forward = controlled(UnitaryGate(unitary), 1, ctrl_state=control_state)

    def forbidden(*args, **kwargs):
        raise AssertionError("inverse entered control synthesis")

    monkeypatch.setattr(UnitaryGate, "control", forbidden)
    inverse = inverse_realized_gate(forward)
    expected = np.eye(4, dtype=complex)
    indices = np.arange(control_state, 4, 2)
    expected[np.ix_(indices, indices)] = unitary.conj().T
    np.testing.assert_allclose(Operator(inverse).data, expected, rtol=0.0, atol=2.0e-14)


_FIVE_BAND_SPEC = BandSpecification(
    offsets=(-2, -1, 0, 1, 2),
    coefficients=(0.3, -1.0, 2.5, -1.0, 0.3 + 0.1j),
    num_qubits=3,
)

# Every construction and registered instance shape (plan Section 7 item 1).
# pauli_lcu shapes are registered up to 2 system qubits: the reused lcu/core
# SELECT synthesizes eagerly through Qiskit controlled-unitary definitions,
# whose rounding accumulates to ~3e-12 at 8 terms x 3 system qubits — above
# the machine-precision class. banded and dense_dilation cover 3-qubit shapes
# at 1e-12; the cross-construction test still reconstructs the 3-qubit
# instance through pauli_lcu under a documented synthesis-noise margin.
_EMBEDDING_CASES = {
    "pauli_lcu_hermitian_1q": lambda: (
        build_block_encoding(_hermitian_2x2(), implementation="pauli_lcu"),
        _hermitian_2x2(),
    ),
    "pauli_lcu_heat_2q": lambda: (
        build_block_encoding(_second_difference(2), implementation="pauli_lcu"),
        _second_difference(2),
    ),
    "banded_spec_tridiagonal_2q": lambda: (
        build_block_encoding(
            BandSpecification(offsets=(-1, 0, 1), coefficients=(-1.0, 2.0, -1.0), num_qubits=2)
        ),
        _dense_from_bands(
            BandSpecification(offsets=(-1, 0, 1), coefficients=(-1.0, 2.0, -1.0), num_qubits=2)
        ),
    ),
    "banded_dense_detected_heat_3q": lambda: (
        build_block_encoding(
            _second_difference(3, diffusivity=0.7, spacing=0.25),
            implementation="banded",
        ),
        _second_difference(3, diffusivity=0.7, spacing=0.25),
    ),
    "banded_single_band_complex_3q": lambda: (
        build_block_encoding(BandSpecification(offsets=(1,), coefficients=(0.5j,), num_qubits=3)),
        _dense_from_bands(BandSpecification(offsets=(1,), coefficients=(0.5j,), num_qubits=3)),
    ),
    "banded_five_band_complex_3q": lambda: (
        build_block_encoding(_FIVE_BAND_SPEC),
        _dense_from_bands(_FIVE_BAND_SPEC),
    ),
    "dense_dilation_heat_2q": lambda: (
        build_block_encoding(_second_difference(2), implementation="dense_dilation"),
        _second_difference(2),
    ),
    "dense_dilation_random_nonhermitian_2q": lambda: (
        build_block_encoding(_random_dense(4, seed=7), implementation="dense_dilation"),
        _random_dense(4, seed=7),
    ),
    "dense_dilation_random_1q": lambda: (
        build_block_encoding(_random_dense(2, seed=11), implementation="dense_dilation"),
        _random_dense(2, seed=11),
    ),
}


@pytest.mark.parametrize("case", sorted(_EMBEDDING_CASES))
def test_embedding_identity_reproduces_scaled_matrix(case: str) -> None:
    """Machine-precision identity: the encoded block equals A / alpha at 1e-12."""

    record, matrix = _EMBEDDING_CASES[case]()
    block = _encoded_block(record)
    assert np.max(np.abs(block - matrix / record.alpha)) <= 1.0e-12
    assert record.error_bound <= 1.0e-12


def test_cross_construction_agreement_on_tridiagonal() -> None:
    """All three constructions reconstruct the same A; alphas hit analytic values.

    2-qubit periodic heat matrix (kappa=h=1): all three encoded blocks agree
    with A/alpha at machine precision. At N=4 the analytic alphas provably
    coincide at 4.0: banded = |2|+|-1|+|-1| = 4 kappa/h^2; pauli_lcu = 4 from
    II(2), IX(-1), XX(-1); dense_dilation = ||A||_2 = 2 - 2 cos(pi) = 4.

    3-qubit instance: the alphas genuinely differ — banded stays 4, pauli_lcu
    is 5 from III(2), IIX(-1), IXX/IYY/XXX/XYY(+-1/2) -> 2+1+4*(1/2), and
    dense_dilation stays 4 (max_k 2 - 2 cos(2 pi k/8) at k=4).
    """

    small = _second_difference(2)
    for name in ("pauli_lcu", "banded", "dense_dilation"):
        record = build_block_encoding(small, implementation=name)
        assert record.alpha == pytest.approx(4.0, rel=1.0e-12, abs=0)
        assert np.max(np.abs(_encoded_block(record) - small / record.alpha)) <= 1.0e-12

    matrix = _second_difference(3)
    records = {
        name: build_block_encoding(matrix, implementation=name)
        for name in ("pauli_lcu", "banded", "dense_dilation")
    }
    assert records["banded"].alpha == pytest.approx(4.0, rel=1.0e-15, abs=0)
    assert records["pauli_lcu"].alpha == pytest.approx(5.0, rel=1.0e-12, abs=0)
    assert records["pauli_lcu"].implementation == "multiplexed_pauli"
    assert records["dense_dilation"].alpha == pytest.approx(4.0, rel=1.0e-12, abs=0)
    assert records["pauli_lcu"].alpha != records["banded"].alpha
    for name in ("banded", "dense_dilation"):
        block = _encoded_block(records[name])
        assert np.max(np.abs(block - matrix / records[name].alpha)) <= 1.0e-12
    # pauli_lcu at 8 terms x 3 system qubits carries ~3e-12 of Qiskit eager
    # controlled-unitary synthesis rounding (measured); 1e-10 is a ~30x-margin
    # tripwire on the same exact identity, not a method tolerance.
    pauli_block = _encoded_block(records["pauli_lcu"])
    assert np.max(np.abs(pauli_block - matrix / records["pauli_lcu"].alpha)) <= 1.0e-10


def test_f3_qsd_bound_selects_the_realized_lower_cx_route_near_tie() -> None:
    matrix = SparsePauliOp.from_list(
        [
            ("XX", 1.1804 + 0.0995j),
            ("IY", 0.8638 - 0.001j),
            ("YY", 0.29 + 0.2632j),
        ]
    ).to_matrix()

    selected = build_block_encoding(matrix)
    assert selected.implementation == "dense_dilation"
    # Two address qubits: 18 SELECT CX, 2 coefficient-phase CX, and
    # 2 * (2**2 - 2) positive-amplitude PREP CX.
    assert selected.metadata["predicted_per_query_cx"] == {
        "multiplexed_pauli": 24.0,
        "dense_dilation": 20.0,
    }

    built = {
        name: build_block_encoding(matrix, implementation=name)
        for name in ("pauli_lcu", "dense_dilation")
    }
    cx_counts = {
        name: transpile(
            encoding.circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
        )
        .count_ops()
        .get("cx", 0)
        for name, encoding in built.items()
    }
    # Realized QSD output is content- and platform-dependent; the contract is
    # the registered upper bound plus the ordering below.
    assert cx_counts["dense_dilation"] <= int(
        selected.metadata["predicted_per_query_cx"]["dense_dilation"]
    )
    assert cx_counts["dense_dilation"] < cx_counts["pauli_lcu"]


def test_dense_unitary_cx_qsd_upper_bound_values_and_domain() -> None:
    assert [dense_unitary_cx_qsd_upper_bound(n) for n in range(1, 6)] == [
        0,
        3,
        20,
        100,
        444,
    ]
    with pytest.raises(ValueError, match="total_qubits must be at least 1"):
        dense_unitary_cx_qsd_upper_bound(0)


def test_pauli_top_left_is_exact_with_complex_phases_and_identity_padding() -> None:
    """Three physical terms exercise the fourth address state's identity pad."""

    terms = (("II", 0.5j), ("XY", -1.0 + 0.25j), ("YZ", 0.75 - 0.5j))
    matrix = SparsePauliOp.from_list(list(terms)).to_matrix()
    record = build_block_encoding(matrix, implementation="pauli_lcu")
    assert record.metadata["pauli_term_count"] == 3
    assert record.metadata["padded_table_size"] == 4
    np.testing.assert_allclose(
        record.alpha * _encoded_block(record), matrix, rtol=0.0, atol=1.0e-12
    )


def test_default_pauli_and_circulant_selection_keep_small_nonzero_terms():
    matrix = np.diag([1e-13, 2e-13]).astype(complex)
    matrix[0, 1] = 3e-14j
    record = build_block_encoding(matrix)
    np.testing.assert_allclose(block_encoding_top_left(Operator(record.circuit).data,
        num_ancillas=record.num_ancillas), matrix/record.alpha, rtol=2e-13, atol=2e-13)
    near = np.diag([1., 1.+1e-13]).astype(complex)
    plan = plan_block_encoding(near)
    assert plan.implementation != "banded"
    assert plan.decomposition.atol == 0
    assert any(0 < abs(term.coefficient) < 1e-12 for term in plan.decomposition.terms)


def test_dense_dilation_planning_admits_the_singular_values_of_its_norm(monkeypatch):
    from nwqlib.subroutines.block_encoding import core

    d = 64
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((d, d)) + 1j*rng.standard_normal((d, d))
    # The intake scan, 16 d**2 = 65,536 units, and the singular values of
    # np.linalg.norm(A, 2): ceil(4 d**3/3) = 349,526 for the bidiagonal
    # reduction and 32 d (d + 1) = 133,120 for the rest of the SVD without
    # vectors. A full SVD with both frames would be charged 8 d**3.
    law = 548_182
    plan = plan_block_encoding(matrix, implementation="dense_dilation", max_work=law)
    assert plan.alpha == pytest.approx(np.linalg.norm(matrix, 2), rel=1e-14, abs=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted spectral norm")

    with monkeypatch.context() as guarded:
        guarded.setattr(np.linalg, "norm", forbidden)
        with pytest.raises(ValueError, match="block-encoding normalization exceeds max_work"):
            plan_block_encoding(matrix, implementation="dense_dilation", max_work=law - 1)
        # The bidiagonal reduction alone makes ceil(4 d**3/3) = 349,526 multiply-adds.
        with pytest.raises(ValueError, match="block-encoding normalization exceeds max_work"):
            plan_block_encoding(matrix, implementation="dense_dilation", max_work=300_000)
    # A caller's full SVD, whose frames the completion reuses, stays charged 8 d**3.
    with pytest.raises(ValueError, match="block-encoding normalization exceeds max_work"):
        core._plan_block_encoding(matrix, implementation="dense_dilation", max_bytes=10**9,
            max_work=16*d*d + 8*d**3 - 1, dense_norm=forbidden)


def test_dense_dilation_completion_is_unitary_and_admitted_at_its_work():
    """The completed dilation of a complex 8 x 8 matrix is unitary to rounding and holds A/alpha in its corner.

    Its completion is admitted at 9 d**3 work, the SVD and the complement product, and refused one unit
    below, before the SVD. No product of the 2d-square dilation with its adjoint is formed.
    """
    d = 8
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    plan = plan_block_encoding(matrix, implementation="dense_dilation")
    with pytest.raises(ValueError, match="completion work"):
        build_block_encoding_from_plan(plan, max_work=9 * d**3 - 1)
    unitary = Operator(build_block_encoding_from_plan(plan, max_work=9 * d**3).circuit).data
    assert np.max(np.abs(unitary @ unitary.conj().T - np.eye(2 * d))) <= 1e-13
    assert np.max(np.abs(block_encoding_top_left(unitary, num_ancillas=1) - matrix / plan.alpha)) <= 1e-14


def test_pauli_dense_dilation_admits_the_singular_values_of_its_norm(monkeypatch):
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    d = 64
    decomposition = PauliDecomposition(
        terms=(PauliTerm(label="XYZIIX", coefficient=.5), PauliTerm(label="ZZIYII", coefficient=.3j),
               PauliTerm(label="IIIIII", coefficient=.2)),
        input_dimension=d, operator_dimension=d, num_qubits=6, atol=0.)
    table = core._admit_pauli_plan(decomposition, held_bytes=0, max_bytes=10**9, max_work=10**9, classify=False)
    # The Pauli planning, the expansion of 3 terms to the dense matrix (3 d**2)
    # and the singular values of np.linalg.norm(A, 2), ceil(4 d**3/3) +
    # 32 d (d + 1) = 482,646 units.
    law = table + 3*d*d + 482_646
    plan = plan_block_encoding(decomposition, implementation="dense_dilation", max_work=law)
    dense = decomposition.to_sparse_pauli_op().to_matrix()
    assert plan.alpha == pytest.approx(np.linalg.norm(dense, 2), rel=1e-14, abs=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted dense conversion")

    monkeypatch.setattr(PauliDecomposition, "to_sparse_pauli_op", forbidden)
    with pytest.raises(ValueError, match="block-encoding Pauli dense conversion exceeds max_work"):
        plan_block_encoding(decomposition, implementation="dense_dilation", max_work=law - 1)
    # The bidiagonal reduction alone makes ceil(4 d**3/3) = 349,526 multiply-adds.
    with pytest.raises(ValueError, match="block-encoding Pauli dense conversion exceeds max_work"):
        plan_block_encoding(decomposition, implementation="dense_dilation", max_work=300_000)


def test_pauli_plan_checks_its_own_arrays_and_the_build_checks_the_tables():
    """Planning 4097 terms on 20 qubits fits 1 MB and 5,000,000 work, while its SELECT tables do not.

    Planning holds the 4097 labels and coefficients (147,492 bytes), two
    packed masks (65,552), the 8192 letter codes and their comparison and
    the extracted mask words (147,488), and does 4,505,660 work units. The
    tables of the circuit, 64 * 8192 * 36 = 18,874,368 bytes and
    32 * 8192 * 400 = 104,857,600 work units, are refused by the build
    before it forms them.
    """
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    n, m = 20, 4097
    terms = tuple(PauliTerm(label=format(i, f"0{n}b").translate(str.maketrans("01", "IX")), coefficient=1.0)
                  for i in range(1, m + 1))
    decomposition = PauliDecomposition(terms=terms, input_dimension=1 << n, operator_dimension=1 << n,
                                       num_qubits=n, atol=0.0)
    plan = plan_block_encoding(decomposition, implementation="multiplexed_pauli", max_bytes=1_000_000,
                               max_work=5_000_000)
    assert (plan.implementation, plan.num_ancillas) == ("multiplexed_pauli", 13)
    assert plan == plan_block_encoding(decomposition, implementation="multiplexed_pauli")
    with pytest.raises(ValueError, match="Pauli planning work 4505660 exceeds max_work=4505659"):
        plan_block_encoding(decomposition, implementation="multiplexed_pauli", max_work=4_505_659)
    with pytest.raises(ValueError, match="Pauli planning arrays"):
        plan_block_encoding(decomposition, implementation="multiplexed_pauli", max_bytes=360_531)
    plan_block_encoding(decomposition, implementation="multiplexed_pauli", max_bytes=360_532)

    def forbidden(*args, **kwargs):
        raise AssertionError("tables formed before their check")

    with pytest.MonkeyPatch.context() as guarded:
        guarded.setattr(core, "_pauli_lcu_encoding", forbidden)
        with pytest.raises(ValueError, match="construction tables"):
            build_block_encoding_from_plan(plan, max_bytes=1_000_000)
        with pytest.raises(ValueError, match="construction exceeds max_work"):
            build_block_encoding_from_plan(plan, max_work=5_000_000)


@pytest.mark.parametrize(
    ("table", "expected_controls"),
    (
        (("I", "Y"), 1),
        (("I", "X", "Y", "Z"), 2),
    ),
)
def test_dependency_projected_ucg_basis_scaling(
    table: tuple[str, ...], expected_controls: int
) -> None:
    """Pinned UCG law equals Qiskit-2.4 basis CX at two support widths."""

    local = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]).astype(complex),
    }
    controls = QuantumRegister(expected_controls, "control")
    target = QuantumRegister(1, "target")
    circuit = QuantumCircuit(controls, target)
    projection = append_dependency_projected_unitaries(
        circuit,
        target[0],
        list(controls),
        [local[label] for label in table],
    )
    assert projection.effective_control_count == expected_controls
    basis = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
    )
    law = projected_unitary_resource_law(expected_controls)
    assert basis.count_ops().get("cx", 0) == law["basis_cx_gates"]


@pytest.mark.parametrize(
    "specification",
    (
        BandSpecification(offsets=(-1, 0, 1), coefficients=(-1.0, 2.0, -1.0), num_qubits=2),
        _FIVE_BAND_SPEC,
    ),
)
def test_banded_multiplexor_law_tracks_measured_basis_scaling(
    specification: BandSpecification,
) -> None:
    """The structural law stays within 4% at address widths two and three."""

    plan = plan_block_encoding(specification)
    record = build_block_encoding(specification)
    basis = transpile(
        record.circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
    )
    measured = basis.count_ops().get("cx", 0)
    predicted = block_encoding_per_query_cx(
        plan.to_dict(),
        implementation="banded",
        system_qubits=plan.system_qubits,
    )
    assert abs(predicted - measured) / measured <= 0.04


def test_dense_normalization_bound_covers_rounded_zero_residual() -> None:
    from fractions import Fraction

    matrix = np.diag([0.1, 0.2]).astype(complex)
    alpha = 0.3
    plan = plan_block_encoding(matrix, implementation="dense_dilation", normalization=alpha)
    record = build_block_encoding_from_plan(plan)
    block = _encoded_block(record)
    assert np.array_equal(matrix - alpha * block, np.zeros_like(matrix))
    exact_error = max(
        abs(Fraction(float(matrix[i, i].real)) - Fraction(alpha) * Fraction(float(block[i, i].real)))
        for i in range(2)
    )
    assert exact_error > 0
    assert Fraction(record.error_bound) >= exact_error
    assert record.error_bound == plan.error_bound

    exact = plan_block_encoding(np.diag([1.0, 2.0]), implementation="dense_dilation", normalization=2.0)
    assert build_block_encoding_from_plan(exact).error_bound == 0.0


def test_dense_contraction_check_uses_the_required_singular_values() -> None:
    matrix = np.full((2, 2), 1.0e-15)

    def build(normalization, source=matrix):
        return build_block_encoding_from_plan(
            plan_block_encoding(source, implementation="dense_dilation", normalization=normalization)
        )

    with pytest.raises(ValueError, match="below"):
        build(1.5e-15)

    legal = build(3.0e-15)
    np.testing.assert_allclose(
        _encoded_block(legal), matrix / legal.alpha, atol=np.finfo(float).eps, rtol=0.0
    )

    # A computed SVD norm can round a few ulps below the largest entry
    # magnitude, the exact norm here. An alpha one ulp below it is admitted
    # by the entry tests and the singular-value test alike, and the dilation
    # stays exact. A deficit of 1e-11, beyond rounding, is refused with both
    # raw values.
    diagonal = np.diag([1.0, 0.5]).astype(complex)
    rounded = build(float(np.nextafter(1.0, 0.0)), diagonal)
    np.testing.assert_allclose(
        rounded.alpha * _encoded_block(rounded), diagonal, atol=2 * np.finfo(float).eps, rtol=0.0
    )
    unitary = Operator(rounded.circuit).data
    assert np.max(np.abs(unitary @ unitary.conj().T - np.eye(4))) <= 4 * np.finfo(float).eps
    with pytest.raises(ValueError, match=r"alpha 0\.99999999999 is below .* entry magnitude is 1\.0"):
        build(1.0 - 1.0e-11, diagonal)


@pytest.mark.parametrize("scale", [1.0, 1.0e-12, 1.0e-13, 1.0e-12j])
def test_dense_banded_detection_bound_has_one_owner(scale: complex) -> None:
    """The detection window, the bands and alpha scale with A, and the dropped band of ``matrix`` stays in the bound.

    Scales 1e-12 and 1e-13 place the bands of ``stencil`` at and below 1e-12 in absolute terms. The imaginary scale
    makes it non-Hermitian, so it takes the unpaired layout, whose alpha still scales with the magnitude of the factor.
    """
    from dataclasses import replace

    matrix = scale * np.diag([2.0, 2.0 + 1.0e-13]).astype(complex)
    matrix[1, 0] = scale * 1.0e-14
    plan = plan_block_encoding(matrix, implementation="banded")
    ideal = _dense_from_bands(plan.source)
    independent_error = float(np.linalg.norm(matrix - ideal, 2))
    assert independent_error > 0.0
    assert independent_error <= plan.error_bound
    assert build_block_encoding_from_plan(plan).error_bound == plan.error_bound
    # Composition may add a child error in the same operator frame.
    augmented = replace(plan, error_bound=2.0 * plan.error_bound)
    assert build_block_encoding_from_plan(augmented).error_bound == augmented.error_bound

    exact = plan_block_encoding(ideal)
    assert exact.error_bound == 0.0
    assert build_block_encoding_from_plan(exact).error_bound == 0.0

    # Each column of a circulant holds every band once, so alpha = sum_b |beta_b| is the matrix 1-norm, 2 + 1 + 1 at
    # scale 1. rel=1e-15 covers the rounding of a sum of three terms, as in the cross-construction test above.
    stencil = scale * _second_difference(2)
    assert np.linalg.norm(stencil, 1) == pytest.approx(4.0 * abs(scale), rel=1.0e-15, abs=0)
    encoded = plan_block_encoding(stencil, implementation="banded")
    assert encoded.alpha == pytest.approx(4.0 * abs(scale), rel=1.0e-15, abs=0)
    assert encoded.error_bound == 0.0  # Exact: entries of one wrapped diagonal are equal products of scale.
    # The 1e-12 is the machine-precision identity of test_embedding_identity_reproduces_scaled_matrix. A/alpha has no units.
    block = _encoded_block(build_block_encoding_from_plan(encoded))
    assert np.max(np.abs(block - stencil / encoded.alpha)) <= 1.0e-12
    # The diagonal of this matrix varies by 0.5 |scale|, far above the window of 2.5e-12 |scale|, so it is no circulant.
    with pytest.raises(ValueError):
        plan_block_encoding(scale * np.diag([2.0, 2.5]).astype(complex), implementation="banded")


def test_hermitian_circulant_within_roundoff_keeps_its_three_structure_facts():
    """A Fourier-built kinetic operator, Hermitian entry by entry, is a circulant only within roundoff.

    Each band pair reads two diagonals with independent roundoff, so without a projection ``beta_{-b} = conj(beta_b)``
    fails, leaving an unpaired block, no involution and an unknown target. For a Hermitian A,
    ``A - Pi(C) = ((A - C) + (A - C)^dagger)/2`` keeps ``||A - Pi(C)||_2`` within the detector's bound. The three facts
    have separate proofs: the target from the input's stored entries, the Hermitian block and the involution from the
    exactly conjugate stored pairs. Dense matrices and the shared-route premise are the independent consumers. The
    circuit has about 400 gates on 8 qubits, so at about ``400*2**-53`` per unit of alpha, ``alpha*1e-13`` covers the
    statevector rounding of the block.
    """
    from nwqlib.subroutines.block_encoding.core import _detect_banded_pauli_structure
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm, decompose_matrix_to_pauli
    from nwqlib.subroutines.qsp.evolution import (build_qsp_evolution_encoding, plan_shared_qsvt_evolution,
                                                  prepare_qsp_evolution)

    size = 16
    fourier = np.fft.fft(np.eye(size), norm="ortho")
    kinetic = fourier.conj().T @ np.diag(np.fft.fftfreq(size, d=1 / size)**2 / 2) @ fourier
    kinetic = (kinetic + kinetic.conj().T) / 2
    assert np.array_equal(kinetic, kinetic.conj().T)
    facts = ("target_operator_is_hermitian", "encoded_operator_is_hermitian", "full_unitary_involution")
    plan = plan_block_encoding(kinetic)
    assert plan.implementation == "banded" and plan.error_bound > 0  # premise: not an exact circulant
    assert all(plan.detail[name] is True for name in facts)
    assert np.linalg.norm(kinetic - _dense_from_bands(plan.source), 2) <= plan.error_bound
    bands = dict(zip((offset % size for offset in plan.source.offsets), plan.source.coefficients))
    assert all(value == bands[-offset % size].conjugate() for offset, value in bands.items())
    encoding = build_block_encoding_from_plan(plan)
    assert encoding.error_bound == plan.error_bound and all(encoding.metadata[name] is True for name in facts)
    unitary = Operator(encoding.circuit).data
    assert np.abs(unitary @ unitary - np.eye(len(unitary))).max() <= 1e-12
    block = block_encoding_top_left(unitary, num_ancillas=encoding.num_ancillas)
    assert np.linalg.norm(encoding.alpha * block - kinetic, 2) <= plan.error_bound + encoding.alpha * 1e-13
    shared = plan_shared_qsvt_evolution(plan, prepare_qsp_evolution(tau=plan.alpha * .1, epsilon=1e-4),
                                        evolution_time=.1, amplification="raw")
    assert shared.signal_access == "direct" and shared.hermitian_target_premise == "construction"
    build_qsp_evolution_encoding(encoding, evolution_time=.1, epsilon=1e-4, require_hermitian_evidence=True)
    decomposition = decompose_matrix_to_pauli(kinetic)
    assert all(term.coefficient.imag == 0 for term in decomposition.terms)
    pauli = plan_block_encoding(decomposition)
    assert pauli.implementation == "banded" and all(pauli.detail[name] is True for name in facts)
    assert np.linalg.norm(kinetic - _dense_from_bands(pauli.source), 2) <= pauli.error_bound
    # A band that projects to exact zero leaves the specification, so these stay usable paired encodings of I. For Y at
    # dimension 2, offset 1 is N/2 and keeps Re(beta_1) = 0. For IY + XY at dimension 4, beta_1 = beta_3 = 1e-15j, so
    # beta_1 + conj(beta_3) cancels as a pair. Both terms are within the detection window of a unit I, and the dense
    # form of each input is the preservation case. ||A - I||_2 is the independent comparison for the bound.
    tiny, pauli_x, pauli_y = 1e-15, np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])
    for terms, matrix in ((("I", 1.), ("Y", tiny)), np.eye(2) + tiny * pauli_y), (
            (("II", 1.), ("IY", tiny), ("XY", tiny)), np.eye(4) + tiny * (np.kron(np.eye(2), pauli_y) + np.kron(pauli_x, pauli_y))):
        assert np.array_equal(matrix, matrix.conj().T)
        input_form = PauliDecomposition(terms=tuple(PauliTerm(label=label, coefficient=value) for label, value in terms),
                                        input_dimension=len(matrix), operator_dimension=len(matrix),
                                        num_qubits=len(terms[0][0]), atol=0.)
        # Premise: the detector keeps these roundoff bands, so the projection is what removes them.
        assert len(_detect_banded_pauli_structure(input_form)[0].offsets) == len(terms)
        for source in (input_form, matrix):
            cancelled = plan_block_encoding(source)
            assert cancelled.implementation == "banded"
            assert cancelled.source.offsets == (0,) and cancelled.source.coefficients == (1,)
            assert cancelled.detail["periodic_gate_counts"]["paired_labels"] is True
            assert all(cancelled.detail[name] is True for name in facts)
            assert np.linalg.norm(matrix - np.eye(len(matrix)), 2) <= cancelled.error_bound
            built = build_block_encoding_from_plan(cancelled)
            assert all(built.metadata[name] is True for name in facts)
            unitary = Operator(built.circuit).data
            assert np.abs(unitary @ unitary - np.eye(len(unitary))).max() <= 1e-12
            block = block_encoding_top_left(unitary, num_ancillas=built.num_ancillas)
            assert np.linalg.norm(built.alpha * block - matrix, 2) <= cancelled.error_bound + built.alpha * 1e-13
    # A nonzero coefficient stays however small: this is exact-zero removal, not a threshold.
    small = plan_block_encoding(PauliDecomposition(
        terms=(PauliTerm(label="I", coefficient=1.), PauliTerm(label="X", coefficient=tiny)),
        input_dimension=2, operator_dimension=2, num_qubits=1, atol=0.))
    assert small.source.offsets == (0, 1) and small.source.coefficients == (1, tiny)
    assert small.detail["periodic_gate_counts"]["paired_labels"] is True and small.error_bound == 0.0
    # Legal neighbors. A true non-Hermitian circulant is not projected, and QSP evolution refuses it. Its symmetrized
    # input, which that refusal recommends, then passes. A supplied specification keeps its stored values.
    shift = np.roll(np.eye(8), 1, axis=0)
    skew = np.eye(8, dtype=complex) + (.5 + .2j) * shift + (.3 - .1j) * shift.T
    unpaired = plan_block_encoding(skew)
    assert all(unpaired.detail[name] is False for name in facts) and unpaired.error_bound == 0.0
    with pytest.raises(ValueError, match="requires Hermitian target and encoded operators"):
        build_qsp_evolution_encoding(build_block_encoding_from_plan(unpaired), evolution_time=.1, epsilon=1e-4)
    assert all(plan_block_encoding((skew + skew.conj().T) / 2).detail[name] is True for name in facts)
    given = BandSpecification(offsets=(-1, 1), coefficients=(1 + 1e-15j, 1 + 0j), num_qubits=3)
    assert plan_block_encoding(given).source is given
    assert plan_block_encoding(given).detail["encoded_operator_is_hermitian"] is False


@pytest.mark.parametrize("perturbed", [False, True])
def test_circulant_frobenius_certificate_rounds_outward(perturbed):
    from fractions import Fraction
    from nwqlib.subroutines.block_encoding.core import _detect_banded_pauli_structure
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    eps = np.finfo(float).eps
    coefficient = complex(1+3*eps, 18*eps) if perturbed else 1.
    decomposition = PauliDecomposition(
        terms=(PauliTerm(label="IX", coefficient=1.), PauliTerm(label="XX", coefficient=coefficient)),
        input_dimension=4, operator_dimension=4, num_qubits=2, atol=0.)
    bands, bound = _detect_banded_pauli_structure(decomposition)
    x = np.array([[0., 1.], [1., 0.]])
    target = np.kron(np.eye(2), x) + coefficient * np.kron(x, x)
    implied = _dense_from_bands(bands)
    # Disjoint Pauli supports and disjoint shifts make these entries exact.
    # Rational squared differences give an independent Frobenius comparison.
    square = sum(((Fraction(float(a.real))-Fraction(float(b.real)))**2
                  + (Fraction(float(a.imag))-Fraction(float(b.imag)))**2
                  for a, b in zip(target.flat, implied.flat)), Fraction())
    assert Fraction(bound)**2 >= square
    assert (bound == 0.) is (not perturbed)
    # Automatic planning keeps this roundoff circulant from either public entry. The Pauli
    # input's bound is the Frobenius bound above, and the dense input's is the spectral-norm
    # bound of its own entrywise residual.
    pauli, dense = plan_block_encoding(decomposition), plan_block_encoding(target)
    assert pauli.implementation == dense.implementation == "banded"
    assert pauli.source.offsets == dense.source.offsets
    np.testing.assert_allclose(_dense_from_bands(dense.source), implied, rtol=0, atol=64*eps)
    assert pauli.error_bound == bound
    assert dense.error_bound >= np.linalg.norm(target - _dense_from_bands(dense.source), 2)
    assert (dense.error_bound == 0.) is (not perturbed)


def test_circulant_check_stops_at_the_first_mismatching_chunk(monkeypatch):
    """A non-circulant sum is refused after the traces of its first 16 terms, and a circulant passes every chunk.

    The 1000 distinct nonidentity Z strings on 10 qubits share flip mask 0, so the candidate is a multiple
    of the identity and predicts a zero coefficient for each of them. The circulant with bands at offsets
    0, 1 and -3 on 5 qubits decomposes into more than 16 Pauli terms, so its check runs several chunks and
    recovers the three bands.
    """
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm, decompose_matrix_to_pauli

    table = core._shift_pauli_trace_table
    rows = []

    def counted(**kwargs):
        rows.append(len(kwargs["flips"]))
        return table(**kwargs)

    monkeypatch.setattr(core, "_shift_pauli_trace_table", counted)
    labels = [format(i, "010b").translate(str.maketrans("01", "IZ")) for i in range(1, 1001)]
    diagonal = PauliDecomposition(terms=tuple(PauliTerm(label=label, coefficient=1.0) for label in labels),
                                  input_dimension=1024, operator_dimension=1024, num_qubits=10, atol=0.0)
    with pytest.raises(ValueError, match="not a certified circulant"):
        core._detect_banded_pauli_structure(diagonal)
    assert rows == [16]
    rows.clear()
    first = np.zeros(32, dtype=complex)
    first[[0, 1, 29]] = [0.5, 0.25 - 0.5j, 0.75]
    circulant = decompose_matrix_to_pauli(np.array([np.roll(first, k) for k in range(32)]).T, atol=0.0, rtol=0.0)
    bands, bound = core._detect_banded_pauli_structure(circulant)
    assert len(circulant.terms) > 16 and rows[:2] == [16, 32] and sum(rows) == len(circulant.terms)
    assert bands.offsets == (-3, 0, 1) and bound <= 1e-14
    np.testing.assert_allclose(bands.coefficients, (0.75, 0.5, 0.25 - 0.5j), rtol=0, atol=1e-15)


def _controlled_matrix(matrix, num_controls, state):
    """The matrix that applies ``matrix`` when the controls, qubits 0 to k - 1, hold ``state`` (test oracle)."""
    result = np.eye(matrix.shape[0] << num_controls, dtype=complex)
    rows = [state + (index << num_controls) for index in range(matrix.shape[0])]
    result[np.ix_(rows, rows)] = matrix
    return result


def _basis_cx(gate):
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, circuit.qubits)
    return transpile(circuit, basis_gates=["cx", "u"], optimization_level=0).count_ops().get("cx", 0)


def _real_pauli_terms():
    # Five terms with negative signs: three address qubits, three padded addresses.
    return (("XZI", .7), ("ZIY", -.4), ("YYZ", -.25), ("III", .1), ("IXX", -.3))


def _query_case(case):
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    if case == "negative_scalar_pauli":
        # A 1x1 operator has no system qubit to carry its sign.
        matrix = np.array([[-1.]])
        return plan_block_encoding(matrix, implementation="multiplexed_pauli"), matrix
    if case == "dense_dilation":
        # A Hermitian 2x2 matrix whose dilation has no involution proof; its queries take the opaque fallback.
        matrix = np.array([[.4, .1], [.1, -.2]])
        return plan_block_encoding(matrix, implementation="dense_dilation"), matrix
    if case.endswith("pauli"):
        terms = _real_pauli_terms() if case == "real_pauli" else (
            ("XZI", .7 + .2j), ("ZIY", -.4), ("YYZ", -.25j), ("IXX", .3 - .1j))
        decomposition = PauliDecomposition(
            terms=tuple(PauliTerm(label=label, coefficient=value) for label, value in terms),
            input_dimension=8, operator_dimension=8, num_qubits=3, atol=0.)
        return (plan_block_encoding(decomposition, implementation="multiplexed_pauli"),
                SparsePauliOp.from_list(list(terms)).to_matrix())
    # A Hermitian specification with two complex pairs, a negative identity, a
    # negative shift by N/2 and two padded labels, the non-Hermitian five
    # bands, or beta I with beta not real, which needs no shift but keeps the
    # phase beta/|beta| of its block.
    specification = {"paired_band": BandSpecification(offsets=(-2, -1, 0, 1, 2, 4),
                                                      coefficients=(.3 - .2j, .5 + .4j, -1.1, .5 - .4j, .3 + .2j, -.6),
                                                      num_qubits=3),
                     "unpaired_band": _FIVE_BAND_SPEC,
                     "scalar_band": BandSpecification(offsets=(0,), coefficients=(.6 + .8j,), num_qubits=2)}[case]
    return plan_block_encoding(specification), _dense_from_bands(specification)


@pytest.mark.parametrize(("case", "involution"), [("real_pauli", True), ("complex_pauli", False),
                                                  ("paired_band", True), ("unpaired_band", False),
                                                  ("scalar_band", False), ("negative_scalar_pauli", True),
                                                  ("dense_dilation", None)])
def test_structured_queries_are_exact_and_hermitian_inputs_are_full_involutions(case, involution, monkeypatch):
    """Native and opaque queries: full unitary, block, controls and adjoint against independent matrices.

    A real Pauli sum and a Hermitian band specification give a unitary that is
    its own adjoint and inverse on the whole space (not only a Hermitian
    block); the complex and non-Hermitian encodings are not involutions. The
    scalar cases have no shift or no system qubit: `(0.6 + 0.8i) I` keeps
    its phase, which controls turn into a relative phase, and the 1x1 matrix
    -1 keeps its exact sign. The
    controlled query applies U only for the stated control value and the
    identity otherwise, global phase included, and the adjoint applies
    U^dagger. The recorded CX count can exceed the emitted circuit only by
    its recorded bound part (PREP slot bounds and multi-controlled X counts
    without ancillas); the multiplexors, diagonals and QFT pair are exact.
    A count that needs an X with more than 64 controls, as the paired labels
    and the scalar sign do with 66 external controls, is unknown (None).

    The dense dilation is controlled as an opaque circuit: its involution
    stays unproved and its CX count unknown. Controlling it needs the exact
    synthesis of its 2-qubit unitary, so with `max_work = 1` the forward and
    the adjoint query of the built encoding are both refused before any
    control or reversal starts.

    Coarse array bytes are checked before anything is copied or built
    (`core._admit_native_query` and `core._opaque_matrix_bytes`).
    A native query needs `H + (4 + 2A) E` bytes with
    `E = 64 * 2**(a + k) * (n + 16)` and `H = m (n + 16)` for the m stored terms,
    plus the 8x8 complex128 matrix (1024 bytes) that a plan of the same
    terms made from their dense matrix keeps. Both real Pauli plans, whose
    SELECT has projected UCGs, are involutions, so their adjoint query is
    the forward query, checked with A = 0 and built without a reversal.
    They are refused one byte below before any UCG exists or builds its
    definition, and built at the bound. A supplied
    circuit appends a two-qubit unitary three times (s = 256 bytes each),
    and the estimate for its adjoint query is 8s per occurrence, 6144
    bytes. It is refused one byte below before the gate conversion and
    built at the bound with the exact adjoint action.
    """
    from qiskit.circuit.library import UCGate, UnitaryGate

    from nwqlib.subroutines.block_encoding import (BlockEncoding, block_encoding_query_cx,
                                                   build_block_encoding_query)

    plan, matrix = _query_case(case)
    unitary = Operator(build_block_encoding_query(plan)).data
    assert np.max(np.abs(block_encoding_top_left(unitary, num_ancillas=plan.num_ancillas)
                         - matrix / plan.alpha)) <= 1e-12
    assert plan.detail["full_unitary_involution"] is involution
    identity = np.eye(len(unitary))
    if involution is not None:
        assert bool(np.max(np.abs(unitary - unitary.conj().T)) <= 1e-12) is involution
        assert bool(np.max(np.abs(unitary @ unitary - identity)) <= 1e-12) is involution
    for controls, state in ((1, 0), (1, 1), (2, 2)):
        for adjoint in (False, True):
            query = build_block_encoding_query(plan, num_controls=controls, ctrl_state=state, adjoint=adjoint)
            target = unitary.conj().T if adjoint else unitary
            assert np.max(np.abs(Operator(query).data - _controlled_matrix(target, controls, state))) <= 1e-12
    if involution:
        with monkeypatch.context() as sentinels:
            sentinels.setattr("nwqlib.subroutines.qiskit_compat.inverse_realized_gate",
                              lambda *args, **kwargs: pytest.fail("an involution's adjoint query was reversed"))
            build_block_encoding_query(plan, num_controls=1, adjoint=True)

    def forbidden(*args, **kwargs):
        pytest.fail("the query copied or built data before its limit check")

    if involution is None:
        assert block_encoding_query_cx(plan, num_controls=1) is None
        built = build_block_encoding_from_plan(plan)
        product = np.diag([1, 1j, -1, -1j])
        leaf = UnitaryGate(product)
        repeated = QuantumCircuit(2)
        for _ in range(3):
            repeated.append(leaf, repeated.qubits)
        supplied = BlockEncoding(circuit=repeated, alpha=1., num_ancillas=1, system_qubits=1, error_bound=0.,
                                 implementation="supplied", metadata={})
        with monkeypatch.context() as sentinels:
            sentinels.setattr("nwqlib.subroutines.qiskit_compat.controlled", forbidden)
            sentinels.setattr("nwqlib.subroutines.qiskit_compat.inverse_realized_gate", forbidden)
            for adjoint in (False, True):
                with pytest.raises(ValueError, match="opaque block-encoding query synthesis"):
                    build_block_encoding_query(built, num_controls=1, adjoint=adjoint, max_work=1)
            sentinels.setattr(QuantumCircuit, "to_gate", forbidden)
            # (5 + 3) s for the adjoint of each occurrence of s = 16 * 4**2 bytes.
            need = 3 * 8 * 256
            with pytest.raises(ValueError, match="max_bytes"):
                build_block_encoding_query(supplied, adjoint=True, max_bytes=need - 1)
        reverse = build_block_encoding_query(supplied, adjoint=True, max_bytes=need)
        assert np.max(np.abs(Operator(reverse).data - np.linalg.matrix_power(product, 3).conj().T)) <= 1e-12
        return
    if case == "real_pauli":
        # The dense twin keeps its 8x8 complex128 matrix as the plan source.
        dense = plan_block_encoding(matrix, implementation="multiplexed_pauli")
        for twin, source_bytes in ((plan, 0), (dense, 16 * 8 * 8)):
            n, a = twin.system_qubits, twin.num_ancillas
            held = len(twin.decomposition.terms) * (n + 16) + source_bytes
            for controls, adjoint in ((0, False), (1, True)):
                need = held + 4 * 64 * (1 << (a + controls)) * (n + 16)
                with monkeypatch.context() as sentinels:
                    sentinels.setattr(UCGate, "__init__", forbidden)
                    sentinels.setattr(UCGate, "_define", forbidden)
                    with pytest.raises(ValueError, match="max_bytes"):
                        build_block_encoding_query(twin, num_controls=controls, adjoint=adjoint,
                                                   max_bytes=need - 1)
                build_block_encoding_query(twin, num_controls=controls, adjoint=adjoint, max_bytes=need)
    assert (block_encoding_query_cx(plan, num_controls=66) is None) is (case in ("paired_band",
                                                                                  "negative_scalar_pauli"))
    census = plan.detail["query_census"]
    for controls, bound in ((0, census["forward_bound_cx"]), (1, census["controlled_bound_cx"])):
        law = block_encoding_query_cx(plan, num_controls=controls)
        emitted = _basis_cx(build_block_encoding_query(plan, num_controls=controls))
        assert 0 <= law - emitted <= bound
    assert (census["forward_cx"], census["controlled_cx"]) == (
        block_encoding_query_cx(plan), block_encoding_query_cx(plan, num_controls=1))


def _node_values_oracle(kind, half_width, spacing, width):
    """Node value of every address label from the label mappings and completions (independent oracle)."""
    values = []
    for label in range(1 << width):
        if kind in ("rotation_table", "twos_complement_pauli"):
            j = label - (1 << width) if label >> (width - 1) else label
            if kind == "rotation_table" and abs(j) > half_width:
                j = 0
        else:
            magnitude = label & ((1 << (width - 1)) - 1)
            sign = -1 if label >> (width - 1) else 1
            j = sign * (min(magnitude, half_width) if kind == "comparator" else magnitude)
        values.append(spacing * j)
    return np.array(values)


@pytest.mark.parametrize(("half_width", "spacing"), [(1, .25), (2, .25), (3, .1)])
def test_uniform_node_recipes_encode_signed_nodes_as_full_involutions(half_width, spacing):
    """Each node recipe encodes h j on its physical labels with its documented completion.

    The physical labels, the duplicate zero, the signed endpoints and the
    padding all appear in the block. The full unitary is an involution, the
    controlled query is exact for both control values, and the recorded CX
    counts bound the emitted circuits within their recorded bound parts.

    The error bound of each encoding covers what its stored data change. The
    rotation table stores rounded angles, which move even the zero node:
    fl(pi) lies about 1.2246e-16 below pi, so the stored angle 2 arccos 0
    gives the block value cos(fl(pi)/2), about 6.12e-17, where 0 is
    intended. The comparator and the Pauli recipes have exact blocks j/N for
    a completion extent N (J, 2**p or 2**p - 1), and their stored alpha is
    h N rounded, which the spacing 0.1 makes inexact; the bound is then
    |h N - alpha| rounded up.
    """
    from fractions import Fraction

    from nwqlib.subroutines.block_encoding import (build_uniform_node_encoding, build_uniform_node_query,
                                                   uniform_node_labels, uniform_node_recipes, uniform_node_values)

    recipes = uniform_node_recipes(half_width, spacing)
    assert [recipe.kind for recipe in recipes] == [
        "rotation_table", "comparator", "twos_complement_pauli", "signed_magnitude_pauli"]
    for recipe in recipes:
        width = recipe.address_qubits
        assert 1 << (width - 1) <= 2 * half_width < 1 << width
        expected = _node_values_oracle(recipe.kind, half_width, spacing, width)
        nodes = np.arange(-half_width, half_width + 1)
        np.testing.assert_array_equal(expected[uniform_node_labels(recipe)], spacing * nodes)
        np.testing.assert_allclose(uniform_node_values(recipe), expected, rtol=0, atol=0)
        assert recipe.physical_radius == spacing * half_width
        assert recipe.alpha >= np.max(np.abs(expected))
        encoding = build_uniform_node_encoding(recipe)
        if recipe.kind == "rotation_table":
            assert 6.1e-17 * recipe.alpha <= encoding.error_bound <= 1e-14 * recipe.alpha
        else:
            extent = {"comparator": half_width, "twos_complement_pauli": 1 << (width - 1),
                      "signed_magnitude_pauli": (1 << (width - 1)) - 1}[recipe.kind]
            exact = abs(Fraction(spacing) * extent - Fraction(recipe.alpha))
            assert Fraction(encoding.error_bound) >= exact
            assert encoding.error_bound <= np.nextafter(float(exact), np.inf)
        unitary = Operator(encoding.circuit).data
        step = 1 << recipe.encoding_ancillas
        assert np.max(np.abs(unitary[::step, ::step] - np.diag(expected / recipe.alpha))) <= 1e-12
        assert np.max(np.abs(unitary @ unitary - np.eye(len(unitary)))) <= 1e-12
        assert np.max(np.abs(unitary - unitary.conj().T)) <= 1e-12
        for state in (0, 1):
            query = build_uniform_node_query(recipe, num_controls=1, ctrl_state=state)
            assert np.max(np.abs(Operator(query).data - _controlled_matrix(unitary, 1, state))) <= 1e-12
        census = recipe.query_census
        for controls, prefix in ((0, "forward"), (1, "controlled")):
            emitted = _basis_cx(build_uniform_node_query(recipe, num_controls=controls))
            assert 0 <= census[prefix + "_cx"] - emitted <= census[prefix + "_bound_cx"]


def test_uniform_node_recipes_and_materializers_admit_their_sizes_first(monkeypatch):
    """Node normalizations and materialized grids are admitted before anything is built.

    Finite h and J can still overflow binary64. With J = 2 and h = 0.4 times
    the largest binary64 value, the radius 2h is finite but the
    two's-complement and signed-magnitude normalizations 4h and 3h are not,
    so only the rotation table and the comparator remain; h = 1e308 makes
    the radius itself overflow. For J = 2**60, so p = 61, the
    signed-magnitude coefficient (2**61 - 1)/2 has no binary64 value and
    that recipe is left out, while the two's-complement coefficients, all
    powers of two, stay exact. Under the default limits the label and value
    grids of J = 2**30 (2**31 + 1 and 2**32 entries) and an angle table with
    40 controls (2**40 entries) need far more than 10 GB, and each request
    is refused before NumPy forms a grid or Qiskit a circuit.

    The two's-complement Pauli recipe for J = 3 encodes an integer diagonal,
    an involution, so its adjoint query is the forward query and needs
    `H + 4E` bytes with `E = 64 * 2**a * (r + 16)`, the m terms of the
    stored plan `H = m (r + 16)` counted once between the node and
    `build_block_encoding_query`, plus `(a + 8) // 8` bytes for `2**a`. It
    is built at that limit and refused one byte below.
    """
    import sys

    from nwqlib.subroutines.block_encoding import (build_diagonal_table_query, build_uniform_node_query,
                                                   uniform_node_labels, uniform_node_recipes, uniform_node_values)

    kept = uniform_node_recipes(2, .4 * sys.float_info.max)
    assert [recipe.kind for recipe in kept] == ["rotation_table", "comparator"]
    assert all(np.isfinite(recipe.alpha) for recipe in kept)
    with pytest.raises(ValueError, match="radius"):
        uniform_node_recipes(2, 1e308)
    assert [recipe.kind for recipe in uniform_node_recipes(1 << 60, 1.)] == [
        "rotation_table", "comparator", "twos_complement_pauli"]
    pauli = next(recipe for recipe in uniform_node_recipes(3, .25) if recipe.kind == "twos_complement_pauli")
    r, a = pauli.address_qubits, pauli.plan.num_ancillas
    need = pauli.pauli_terms * (r + 16) + 4 * 64 * (1 << a) * (r + 16) + (a + 8) // 8
    build_uniform_node_query(pauli, adjoint=True, max_bytes=need)
    with pytest.raises(ValueError, match="max_bytes"):
        build_uniform_node_query(pauli, adjoint=True, max_bytes=need - 1)

    def forbidden(*args, **kwargs):
        pytest.fail("a node grid or circuit was built before its size was admitted")

    table = uniform_node_recipes(1 << 30, 1.)[0]
    monkeypatch.setattr(np, "arange", forbidden)
    monkeypatch.setattr(QuantumCircuit, "__init__", forbidden)
    for request in (lambda: uniform_node_labels(table), lambda: uniform_node_values(table),
                    lambda: build_uniform_node_query(table),
                    lambda: build_diagonal_table_query([.5], 1., num_controls=40)):
        with pytest.raises(ValueError, match="bytes, exceeding the physical memory"):
            request()


def _real_pauli(terms, n):
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    decomposition = PauliDecomposition(terms=tuple(PauliTerm(label=label, coefficient=value) for label, value in terms),
                                       input_dimension=1 << n, operator_dimension=1 << n, num_qubits=n, atol=0.)
    return (plan_block_encoding(decomposition, implementation="multiplexed_pauli"),
            SparsePauliOp.from_list(list(terms)).to_matrix())


def _check_composite(plan, target):
    """Block within the plan bound, involution when recorded, exact control and adjoint, CX within the census.

    The census CX of one forward and one singly controlled query are upper bounds from the child laws, so the
    transpiled count of the built query must not exceed them when they are known.
    """
    from nwqlib.subroutines.block_encoding import build_access_query, build_joint_encoding

    built = build_joint_encoding(plan)
    unitary = Operator(built.circuit).data
    block = block_encoding_top_left(unitary, num_ancillas=built.num_ancillas)
    assert np.linalg.norm(plan.alpha * block - target, 2) <= plan.error_bound + 1e-12
    if plan.full_unitary_involution:
        assert np.max(np.abs(unitary @ unitary - np.eye(len(unitary)))) <= 1e-10
        assert np.max(np.abs(unitary - unitary.conj().T)) <= 1e-10
    queries = {state: build_access_query(plan, num_controls=1, ctrl_state=state) for state in (0, 1)}
    for state, query in queries.items():
        assert np.max(np.abs(Operator(query).data - _controlled_matrix(unitary, 1, state))) <= 1e-10
    assert np.max(np.abs(Operator(build_access_query(plan, adjoint=True)).data - unitary.conj().T)) <= 1e-10
    for query, bound in ((build_access_query(plan), plan.query_census["forward_cx"]),
                         (queries[1], plan.query_census["controlled_cx"])):
        if bound is not None:
            assert _basis_cx(query) <= bound


def test_joint_generators_and_contraction_encode_their_targets():
    """Separate parts, single A, the convex contraction and a product encode their targets.

    The targets are S = D_Q (x) (D_K (x) L_0 + I (x) H_0), the single-A S, X_a = (1-q_a) I - q_a L_0/alpha and
    the noncommuting product L_0 H_0. The oracles are assembled independently with numpy (system order: node
    register low, application register, child system high). Real Pauli children, node tables and the real
    combine PREP propagate a full involution; the single-A construction is an involution for a dense child
    without one. A Hermitian paired band with a time factor checks the H-only branch. The product's factors
    both leave garbage on their ancillas, so its block is L_0 H_0 only with independent ancilla registers.
    """
    from nwqlib.subroutines.block_encoding import (plan_block_encoding_product, plan_convex_contraction,
                                                   plan_joint_generator, plan_single_a_generator,
                                                   uniform_node_recipes, uniform_node_values)
    l_plan, l0 = _real_pauli((("X", .5), ("Z", .2)), 1)
    h_plan, h0 = _real_pauli((("Y", .3), ("X", -.1)), 1)
    recipe = uniform_node_recipes(1, .7)[0]
    joint = plan_joint_generator(nodes=recipe, l_child=l_plan, h_child=h_plan, time_factors=(1., .25))
    q, d_k = np.diag([1., .25]), np.diag(uniform_node_values(recipe))
    assert joint.full_unitary_involution is True
    # The rotation-table node query carries the combine control, so a controlled joint query controls it twice.
    assert joint.query_census["controlled_cx"] is not None
    assert joint.alpha == pytest.approx(recipe.alpha * l_plan.alpha + h_plan.alpha, rel=1e-15)
    _check_composite(joint, np.kron(l0, np.kron(q, d_k)) + np.kron(h0, np.kron(q, np.eye(len(d_k)))))
    # The comparator core carries the combine control and the external one, k = 2 (M7.4 at the node owner).
    comparator = uniform_node_recipes(1, .7)[1]
    joint = plan_joint_generator(nodes=comparator, l_child=l_plan, h_child=h_plan, time_factors=(1., .25))
    d_c = np.diag(uniform_node_values(comparator))
    assert joint.query_census["controlled_cx"] is not None
    _check_composite(joint, np.kron(l0, np.kron(q, d_c)) + np.kron(h0, np.kron(q, np.eye(len(d_c)))))

    band = BandSpecification(offsets=(-1, 1), coefficients=(.3 - .1j, .3 + .1j), num_qubits=2)
    shift = np.roll(np.eye(4), 1, axis=0)
    h_only = plan_joint_generator(h_child=plan_block_encoding(band), time_factors=(1., .25))
    assert h_only.full_unitary_involution is True
    _check_composite(h_only, np.kron((.3 + .1j) * shift + (.3 - .1j) * shift.T, q))

    a = np.array([[.2 + .1j, .3], [-.1j, -.4 + .2j]])
    k = np.array([-.5, 0., .5, 0.])
    single = plan_single_a_generator(a_child=plan_block_encoding(a, implementation="dense_dilation"),
                                     node_coefficients=k - 1j * np.array([1., 1., 1., 0.]),
                                     radius=float(np.hypot(.5, 1.)), time_factors=(1., .5))
    assert single.full_unitary_involution is True
    q = np.diag([1., .5])
    _check_composite(single, np.kron((a + a.conj().T) / 2, np.kron(q, np.diag(k)))
                     + np.kron((a - a.conj().T) / 2j, np.kron(q, np.diag([1., 1., 1., 0.]))))

    times = np.array([1., .3, 0., .6])
    contraction = plan_convex_contraction(l_child=l_plan, alpha=.9, time_factors=times)
    assert contraction.alpha == 1.0 and contraction.full_unitary_involution is True
    _check_composite(contraction, sum(np.kron((1 - t) * np.eye(2) - t * l0 / .9, np.diag(np.eye(4)[i]))
                                      for i, t in enumerate(times)))
    with pytest.raises(ValueError, match="cover"):
        plan_convex_contraction(l_child=l_plan, alpha=.5 * l_plan.alpha)

    product = plan_block_encoding_product(l_plan, h_plan)
    assert np.linalg.norm(l0 @ h0 - h0 @ l0, 2) > .1
    _check_composite(product, l0 @ h0)


class _Application:
    def __init__(self, kind, time, mass):
        import math

        self.kind, self.elapsed_time = kind, time
        self.mass = type("Mass", (), {})()
        self.mass.mantissa, self.mass.exponent = math.frexp(mass)


@pytest.mark.parametrize("layout", ["twos_complement", "signed_magnitude", "table"])
def test_factorized_source_preparation_matches_the_finite_sum(layout):
    """The source LCU circuit with the factorized PREP, the coupled phase and kind-controlled inputs.

    The evolution is the exact exp(-i T S) of the joint generator, so the check isolates PREP, phases and the
    never-undone state preparation: the projected output times gamma equals the independently formed finite
    sum over two unequal source times, noncommuting L_0, H_0 and a relative input phase. The uniform layouts
    take the label mapping of the rotation-table (two's complement) or the comparator (signed magnitude) node
    recipe, whose coupled phase is applied per label bit; the table layout applies it as one table. The table
    layout also prepares an initial state |0> with an empty circuit, with no source and an exact-zero kernel
    coefficient, which stays a structural zero without an omission allowance.
    """
    import math
    import scipy.linalg
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector
    from nwqlib.algorithms.lchs.providers import lifted_magnitudes
    from nwqlib.subroutines.block_encoding import BlockEncoding, uniform_node_labels, uniform_node_recipes
    from nwqlib.subroutines.block_encoding.source_prep import (build_source_lcu_circuit,
                                                               plan_factorized_source_preparation)
    from nwqlib.subroutines.state_preparation import build_qiskit_state_preparation

    l0, h0 = np.array([[.2, .15], [.15, -.1]]), np.array([[.1, -.2j], [.2j, -.3]])
    c_l, c_h, shift, final, h = .35, -.4, .2, 1.1, .6
    k = h * np.array([-1., 0., 1.])
    recipe = None if layout == "table" else next(r for r in uniform_node_recipes(1, h) if r.mapping == layout)
    eye = np.eye(2)

    def check(apps, coefficients, *, initial, source):
        magnitudes, lift = lifted_magnitudes(coefficients)
        records = [_Application(kind, t, wq * np.linalg.norm(z) * np.exp(shift * t)) for kind, t, wq, z in apps]
        prep = plan_factorized_source_preparation(
            applications=records, kernel_magnitudes=magnitudes, kernel_lift=lift,
            kernel_arguments=np.angle(coefficients), kernel_nodes=None if recipe else k, node_recipe=recipe,
            scalar_phase=(c_l + shift, c_h))
        assert prep.phase_recipe == layout
        labels = np.arange(3) if recipe is None else uniform_node_labels(recipe)
        d_k = np.zeros(1 << prep.kernel_qubits)
        d_k[labels] = k
        q = np.diag(np.array(prep.application_times) / final)
        joint = np.kron(l0, np.kron(q, np.diag(d_k))) + np.kron(h0, np.kron(q, np.eye(len(d_k))))
        width = prep.kernel_qubits + prep.application_qubits + 1
        circuit = QuantumCircuit(width)
        circuit.append(UnitaryGate(scipy.linalg.expm(-1j * final * joint)), circuit.qubits)
        exact = BlockEncoding(circuit=circuit, alpha=1., num_ancillas=0, system_qubits=width, error_bound=0.,
                              implementation="exact_evolution_oracle", metadata={})
        lcu = build_source_lcu_circuit(prep, exact, initial=initial, source=source)
        output = Statevector(lcu).data[::1 << (prep.kernel_qubits + prep.application_qubits)]
        expected = sum(wq * np.linalg.norm(z) * np.exp(shift * t) * c
                       * scipy.linalg.expm(-1j * t * (kj * ((c_l + shift) * eye + l0) + c_h * eye + h0))
                       @ (z / np.linalg.norm(z)) for _, t, wq, z in apps for kj, c in zip(k, coefficients))
        gamma = math.ldexp(prep.recovery_mantissa, prep.recovery_exponent)
        assert np.linalg.norm(gamma * output - expected) <= 1e-12 * np.linalg.norm(expected)
        return prep

    u0, b = np.array([.6, .8j]) * 1.7, np.array([1 - .5j, .3]) * .9
    x, w = np.polynomial.legendre.leggauss(2)
    apps = [("initial", final, 1., u0)] + [("source", final - r, wq, b) for r, wq in
                                           zip(final / 2 * (x + 1), final / 2 * w)]
    check(apps, np.array([.4 - .7j, -1.1 + .2j, .3 + .5j]), initial=build_qiskit_state_preparation(u0).circuit,
          source=build_qiskit_state_preparation(b).circuit)
    if layout == "table":
        prep = check([("initial", final, 1., np.array([1.3, 0.]))], np.array([.4 - .7j, 0., .3 + .5j]),
                     initial=QuantumCircuit(1), source=None)
        assert prep.omitted_applications == () and prep.kernel_amplitudes[1] == 0.


def test_saved_analytic_kernel_prep_prepares_the_low_somma_kernel_state_with_controls_and_inverse(monkeypatch):
    """The analytic Low-Somma kernel PREP, rebuilt from its saved record, prepares the kernel state it claims.

    ``data/lchs_analytic_kernel_prep.json`` is the LCHS archive record of one prepared query for J = 1,
    h = 1/2, gamma = 1, margin eta = 1/4 and tolerance 1e-3 on the comparator nodes. Planning is phase-free and
    must equal the saved plan, whose degree n and sample count M follow G19/G20 at lambda = hJ = 1/2: with
    rho = 1 + sqrt(2) and M_rho = (2/sqrt 3) exp(1/32), n is the least even degree with
    tail(n) = 2 M_rho rho**-n/(rho - 1) <= tol/2, and M the least power of two above n with
    alias(n, M) = 2 M_rho (1 + 2n) rho**-(2M - n)/(1 - rho**-2M) <= tol - tail(n). Reading the record and
    building its queries calls no phase solver, coefficient acquisition or probability evaluation.

    The forward query maps |0> to psi_f = sum_j f(hj)|label(j)>/norm with f(k) = exp(-k**2/8)/sqrt(1 + k**2),
    the signed-magnitude labels {0: 0, 1: 1, -1: 3} on two address qubits (label 2 is padding) and all work
    qubits zero, within the preparation's kernel_error; its amplitude on nonzero work states is at most the
    amplification error. With one control the active block is the forward query and the inactive block the
    identity, for both control values, and the adjoint is the conjugate transpose, uncontrolled and
    controlled. No global phase is aligned; 1e-10 covers statevector rounding of a few hundred gates on at
    most seven qubits. The transpiled CX counts are at most the census of each query.
    """
    import json
    import math
    from pathlib import Path
    from nwqlib.algorithms.lchs import archive
    from nwqlib.subroutines.block_encoding import analytic_prep as ap
    from nwqlib.subroutines.block_encoding.nodes import uniform_node_recipes

    rounding = 1e-10
    recipe = next(r for r in uniform_node_recipes(1, .5) if r.kind == "comparator")
    planned = ap.plan_analytic_kernel_preparation(recipe, gamma=1., approximation_tolerance=1e-3, margin=.25,
                                                  max_rounds=4, max_bytes=32 * 2**20)
    rho, m_rho = 1 + math.sqrt(2), 2 / math.sqrt(3) * math.exp(1 / 32)

    def tail(n):
        return 2 * m_rho * rho**-n / (rho - 1)

    def alias(n, samples):
        return 2 * m_rho * (1 + 2 * n) * rho ** -(2 * samples - n) / (1 - rho ** (-2 * samples))
    n = next(n for n in range(0, 100, 2) if tail(n) <= 1e-3 / 2)
    samples = next(s for s in (1 << b for b in range(1, 20)) if s > n and alias(n, s) <= 1e-3 - tail(n))
    assert (planned.degree, planned.samples) == (n, samples)
    # libm exp/pow differences only.
    assert planned.tail_bound == pytest.approx(tail(n), rel=1e-12)
    assert planned.alias_bound == pytest.approx(alias(n, samples), rel=1e-12)

    for name in ("solve_wx_qsp_phases", "_streamed_probabilities", "_acquire_coefficients"):
        monkeypatch.setattr(ap, name, lambda *a, _name=name, **k: pytest.fail(f"rebuild called {_name}"))
    saved = json.loads((Path(__file__).parent / "data" / "lchs_analytic_kernel_prep.json").read_text())
    prepared = archive._SelectedData(None, None).read(saved)
    assert prepared.plan == planned

    work = prepared.plan.work_qubits
    f = [math.exp(-k * k / 8) / math.sqrt(1 + k * k) for k in (0., .5, -.5)]
    target = np.zeros(1 << (work + 2), complex)
    for label, value in zip((0, 1, 3), f):
        target[label << work] = value / math.sqrt(sum(v * v for v in f))
    forward = Operator(ap.build_analytic_kernel_query(prepared)).data
    assert np.linalg.norm(forward[:, 0] - target) <= prepared.kernel_error
    garbage = np.linalg.norm(forward[:, 0].reshape(-1, 1 << work)[:, 1:])
    assert garbage <= prepared.error_terms["amplification"] + rounding
    for state in (0, 1):
        gate = Operator(ap.build_analytic_kernel_query(prepared, num_controls=1, ctrl_state=state)).data
        active, inactive = np.arange(state, len(gate), 2), np.arange(1 - state, len(gate), 2)
        assert np.max(np.abs(gate[np.ix_(active, active)] - forward)) <= rounding
        assert np.max(np.abs(gate[np.ix_(inactive, inactive)] - np.eye(len(forward)))) <= rounding
        assert np.max(np.abs(gate[np.ix_(active, inactive)])) <= rounding
    adjoint = Operator(ap.build_analytic_kernel_query(prepared, adjoint=True)).data
    assert np.max(np.abs(adjoint - forward.conj().T)) <= rounding
    controlled_query = Operator(ap.build_analytic_kernel_query(prepared, num_controls=1)).data
    controlled_adjoint = Operator(ap.build_analytic_kernel_query(prepared, num_controls=1, adjoint=True)).data
    assert np.max(np.abs(controlled_adjoint - controlled_query.conj().T)) <= rounding
    for name, controls in (("forward", 0), ("controlled", 1)):
        gate = ap.build_analytic_kernel_query(prepared, num_controls=controls)
        circuit = QuantumCircuit(gate.num_qubits)
        circuit.append(gate, circuit.qubits)
        cx = transpile(circuit, basis_gates=["cx", "u"], optimization_level=0).count_ops().get("cx", 0)
        assert 0 < cx <= prepared.census[name]["cx"]


def test_analytic_kernel_mp1_pass_is_admitted_before_it_runs(monkeypatch):
    """MP1 holds one batch of ten float64 arrays over min(M, max(d + 1, 4096)) nodes, the stored target and
    its partial sums (analytic_prep._mp1_bytes). With M = 8193 nodes that exceeds the coefficient acquisition,
    so a limit admitting the acquisition and not MP1 is refused before MP1 runs, and a limit admitting MP1
    gives the probabilities of the default limit."""
    from dataclasses import replace
    from nwqlib._limits import ResourceLimitError
    from nwqlib.subroutines.block_encoding import analytic_prep as ap
    from nwqlib.subroutines.block_encoding.nodes import uniform_node_recipes
    from nwqlib.subroutines.qsp.phases import _QSPPhaseConvergenceError

    recipe = next(r for r in uniform_node_recipes(4096, 1e-3) if r.kind == "comparator")
    planned = ap.plan_analytic_kernel_preparation(recipe, gamma=1., approximation_tolerance=1e-3, margin=.25,
                                                  max_rounds=64, max_bytes=32 * 2**20)
    reference = ap.prepare_analytic_kernel(planned)
    needed = ap._mp1_bytes(planned.degree + 1, planned.degree + 1, 2 * 4096 + 1)
    acquisition = ap._acquisition_bytes(planned.degree, planned.samples)
    assert acquisition < needed - 1
    with monkeypatch.context() as patch:
        patch.setattr(ap, "_streamed_probabilities", lambda *a: pytest.fail("MP1 ran before admission"))
        with pytest.raises(ValueError, match="MP1"):
            ap.prepare_analytic_kernel(replace(planned, max_bytes=needed - 1))
    admitted = ap.prepare_analytic_kernel(replace(planned, max_bytes=needed))
    assert (admitted.target_probability, admitted.physical_probability) == (reference.target_probability,
                                                                            reference.physical_probability)
    # A phase stop at max_fft is a limit refusal that keeps the attempts, and a target of 1e-30, far below what
    # these binary64 attempts reach, stops on the ideal rule as a numerical failure.
    tight = replace(planned, residual_tolerance=1e-30)
    with pytest.raises(_QSPPhaseConvergenceError) as ruled:
        ap.prepare_analytic_kernel(tight)
    assert not isinstance(ruled.value, ResourceLimitError)
    first = ruled.value.evidence.fft_lengths[0]
    with pytest.raises(ResourceLimitError) as capped:
        ap.prepare_analytic_kernel(replace(tight, max_fft=2 * first))
    assert capped.value.evidence.fft_lengths == (first, 2 * first)


def test_band_offsets_fit_int64_storage_and_long_range_angles_keep_their_phase():
    """An offset is stored when it fits int64: at q = 64 the representative 2**63 is refused and 2**63 - 1
    kept. For a paired band b = 3 * 2**37 + 12345 at q = 40, each label's diagonal phase plus the RZ half
    angles of a basis state k gives phase_l + 2 pi (b_l k mod N)/N modulo 2 pi, the scalar Fourier factor of
    S**b_l, so a reduction of the angles cannot change a label's sign."""
    from nwqlib.operators import ingest_band
    from nwqlib.subroutines.block_encoding.banded import _band_angle_tables, _band_diagonal_phases, _band_layout

    with pytest.raises(ValueError, match="int64"):
        ingest_band([2**63], [1.0], num_qubits=64)
    assert ingest_band([2**63 - 1], [1.0], num_qubits=64).band().offsets == (2**63 - 1,)
    q, b = 40, 3 * 2**37 + 12345
    layout = _band_layout(ingest_band([0, b, -b], [0.5, 0.25 + 0.1j, 0.25 - 0.1j], num_qubits=q).band())
    tables = _band_angle_tables(layout)
    phases = _band_diagonal_phases(layout, tables)
    assert sorted(layout.offsets) == [-b, b] or sorted(set(layout.offsets)) == [-b, 0, b]
    rng = np.random.default_rng(2)
    for k in [0, (1 << q) - 1, *map(int, rng.integers(0, 1 << q, size=8, dtype=np.int64))]:
        for label, offset in enumerate(layout.offsets):
            combined = phases[label] + sum((2 * ((k >> j) & 1) - 1) * tables[j][label] / 2 for j in range(q))
            exact = layout.phases[label] + 2 * np.pi * ((offset * k) % (1 << q)) / (1 << q)
            assert abs((combined - exact + np.pi) % (2 * np.pi) - np.pi) < 1e-11


def test_dense_auto_selects_the_dilation_from_the_term_count_when_prep_alone_costs_more(monkeypatch):
    """A dense matrix with more than d**2/2 nonzero Pauli terms is planned as a dense dilation without forming them.

    A random Hermitian matrix on 5 qubits has all 1024 Pauli coefficients
    nonzero, so the PREP pair of the Pauli encoding alone needs
    2*(1024 - 2) CX, above the dense dilation's QSD count on 6 qubits. The
    plan has the spectral norm and error bound of the explicit dense
    dilation and records the term count and that PREP count. Its words are
    counted, never packed. A tridiagonal matrix keeps the full comparison
    and its Pauli terms.
    """
    from nwqlib.subroutines.block_encoding import core

    rng = np.random.default_rng(4)
    b = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    a = (b + b.conj().T) / 2
    explicit = plan_block_encoding(a, implementation="dense_dilation")
    with monkeypatch.context() as guard:
        guard.setattr(core, "PauliTerm", lambda **_: pytest.fail("Pauli term formed"))
        guard.setattr(core, "_coefficient_masks", lambda *_, **__: pytest.fail("Pauli words packed"))
        selected = plan_block_encoding(a)
    assert selected.implementation == "dense_dilation" and selected.decomposition is None
    assert (selected.alpha, selected.error_bound) == (explicit.alpha, explicit.error_bound)
    assert selected.detail["term_count"] == 1024 and selected.detail["prep_pair_cx"] == 2 * (1024 - 2)
    assert 2 * (1024 - 2) > core._dense_dilation_predicted_cx(5)
    shift = np.diag(np.ones(31), 1)
    tridiagonal = plan_block_encoding(2 * np.eye(32) - shift - shift.T)
    assert tridiagonal.implementation == "multiplexed_pauli" and tridiagonal.decomposition.terms


def test_supplied_query_counts_the_matrix_table_of_each_top_level_ucgate():
    """An uncontrolled adjoint query of a supplied UCGate is checked at 8 * 64 * L bytes (core._opaque_matrix_bytes).

    L is the number of 2x2 matrices the gate keeps in its table: Qiskit's
    table with ``mux_simp=False``, the shorter table that ``mux_simp=True``
    keeps after it removes a repeated control, and the padded table of
    NWQLib's realized multiplexer. Each query is refused one byte below
    before any gate conversion, and built at the bound with the exact
    adjoint action. An instruction that only shares the name ``multiplexer``
    holds no table and counts nothing.
    """
    from unittest.mock import patch

    from qiskit.circuit import Gate
    from qiskit.circuit.library import UCGate

    from nwqlib.subroutines._multiplexors import append_uniformly_controlled_unitaries
    from nwqlib.subroutines.block_encoding import BlockEncoding, build_block_encoding_query
    from nwqlib.subroutines.block_encoding.core import _opaque_matrix_bytes

    x, s = np.array([[0, 1], [1, 0]], complex), np.diag([1, 1j])
    plain = QuantumCircuit(2)
    plain.append(UCGate([x, s], mux_simp=False), [0, 1])
    simplified = QuantumCircuit(3)
    simplified.append(UCGate([x, s, x, s], mux_simp=True), [0, 1, 2])
    realized = QuantumCircuit(3)
    append_uniformly_controlled_unitaries(realized, realized.qubits[0], realized.qubits[1:], [x, s, x])
    for circuit, matrices in ((plain, 2), (simplified, 2), (realized, 4)):
        (gate,) = (item.operation for item in circuit.data)
        assert len(gate.params) == matrices
        need = 8 * 64 * matrices
        supplied = BlockEncoding(circuit=circuit, alpha=1., num_ancillas=circuit.num_qubits - 1, system_qubits=1,
                                 error_bound=0., implementation="supplied", metadata={})
        with patch.object(QuantumCircuit, "to_gate", side_effect=AssertionError("conversion before the check")):
            with pytest.raises(ValueError, match=f"needs about {need} bytes"):
                build_block_encoding_query(supplied, adjoint=True, max_bytes=need - 1)
        query = build_block_encoding_query(supplied, adjoint=True, max_bytes=need)
        assert np.max(np.abs(Operator(query).data - Operator(circuit).data.conj().T)) <= 1e-12
    named = Gate("multiplexer", 1, [0.3])
    named.definition = QuantumCircuit(1)
    named.definition.rz(0.3, 0)
    other = QuantumCircuit(2)
    other.append(named, [0])
    other.append(UCGate([x, s], mux_simp=False), [0, 1])
    assert _opaque_matrix_bytes(other) == 2 * 64
