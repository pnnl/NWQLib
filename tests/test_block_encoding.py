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


def test_pauli_dense_dilation_admits_the_singular_values_of_its_norm(monkeypatch):
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    d = 64
    decomposition = PauliDecomposition(
        terms=(PauliTerm(label="XYZIIX", coefficient=.5), PauliTerm(label="ZZIYII", coefficient=.3j),
               PauliTerm(label="IIIIII", coefficient=.2)),
        input_dimension=d, operator_dimension=d, num_qubits=6, atol=0.)
    table = core._admit_pauli_plan(decomposition, held_bytes=0, max_bytes=10**9, max_work=10**9, classify=False)
    # The Pauli tables, the expansion of 3 terms to the dense matrix (3 d**2)
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


@pytest.mark.parametrize("scale", [1.0])
def test_dense_banded_detection_bound_has_one_owner(scale: float) -> None:
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
