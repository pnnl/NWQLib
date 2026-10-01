"""Independent selected QSP composition, recovery, phase and resource relations."""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
import builtins
import numpy as np
import pytest
import scipy.linalg
from qiskit import QuantumCircuit,transpile
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import UnitaryGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator,SparsePauliOp
from qiskit_aer import AerSimulator
from conftest import _encoded_block
from nwqlib.operators._pauli import _pauli_masks
import nwqlib.algorithms.lchs.compiled_selection as compiled_module
import nwqlib.algorithms.lchs.native as native_module
import nwqlib.subroutines.block_encoding.core as block_encoding_core
from nwqlib.subroutines.block_encoding import block_encoding_top_left,build_block_encoding,build_block_encoding_from_plan,plan_block_encoding
from nwqlib.subroutines.pauli_decomposition import PauliDecomposition,PauliTerm,decompose_matrix_to_pauli
from nwqlib.subroutines.qsp import build_control_diagonal_generator_encoding,compiled_select_resource_law
from nwqlib.subroutines.qsp.evolution import build_qsp_evolution_encoding
_PanicException=type('PanicException',(BaseException,),{'__module__':'pyo3_runtime'})

def _encoded_block_via_statevector(circuit, num_ancillas: int) -> np.ndarray:
    """Column-wise encoded-block extraction for deep compiled circuits.

    ``Operator(...)`` on the compiled SELECT resolves the full unitary
    column-by-column through Python and is prohibitively slow past ~8
    qubits; flattening once and running the Aer statevector per system
    basis column extracts the same all-zero-ancilla block exactly.
    """

    assert circuit.num_qubits <= 13
    flat = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    simulator = AerSimulator(method="statevector")
    system_dimension = 2 ** (flat.num_qubits - num_ancillas)
    rows = np.arange(system_dimension) << num_ancillas
    block = np.zeros((system_dimension, system_dimension), dtype=complex)
    for column in range(system_dimension):
        prepared = QuantumCircuit(flat.num_qubits)
        for bit in range(flat.num_qubits - num_ancillas):
            if (column >> bit) & 1:
                prepared.x(num_ancillas + bit)
        prepared.compose(flat, inplace=True)
        prepared.save_statevector()
        vector = np.asarray(simulator.run(prepared).result().get_statevector())
        block[:, column] = vector[rows]
    return block



def _seeded_l_h(dimension: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded generic-spectrum Hermitian pair (CI trap list: no degenerate
    spectra for two-qubit synthesis; eigenvalue gaps are O(0.1), far from
    any decision threshold)."""

    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(
        rng.standard_normal((dimension, dimension))
        + 1j * rng.standard_normal((dimension, dimension))
    )
    l_part = basis @ np.diag(np.linspace(0.08, 0.5, dimension)) @ basis.conj().T
    # Matrix multiplication can leave an antisymmetric rounding component.
    # This fixture specifies an exactly Hermitian target for eigenvalue QSP.
    l_part = (l_part + l_part.conj().T) / 2
    h_base = rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal(
        (dimension, dimension)
    )
    h_part = 0.15 * (h_base + h_base.conj().T) / 2.0
    return l_part, h_part



@pytest.mark.parametrize("angle", (1.0e-11, 1.0e-15))
def test_compiled_coefficient_phase_preserves_tiny_request(angle) -> None:
    from nwqlib.algorithms.lchs.native import _compiled_coefficient_phase_gate

    gate = _compiled_coefficient_phase_gate(np.asarray([1.0, np.exp(1.0j * angle)]), 4)
    circuit = QuantumCircuit(2)
    circuit.append(gate, circuit.qubits)
    diagonal = np.diag(Operator(circuit.decompose(reps=10)).data)
    assert (diagonal[1] / diagonal[0]).imag / angle == pytest.approx(1.0, rel=1.0e-10, abs=0.0)
    assert _compiled_coefficient_phase_gate(np.asarray([1.0, 2.0]), 4) is None
    assert _compiled_coefficient_phase_gate(np.asarray([1.0, complex(-0.0, -0.0)]), 4) is None



def test_control_diagonal_generator_encoding_composition() -> None:
    """Machine-precision identity: the encoded block is exactly
    ``(kron(L, D_L) + kron(H, D_H)) / alpha`` with
    ``alpha = max|D_L| alpha_L + max|D_H| alpha_H`` (the control-register
    generalization of the scalar-seam alpha)."""

    l_matrix = np.array([[0.5, 0.1], [0.1, 0.3]], dtype=complex)
    h_matrix = np.array([[0.2, 0.05j], [-0.05j, -0.1]], dtype=complex)
    encoding_l = build_block_encoding(l_matrix, implementation="dense_dilation")
    encoding_h = build_block_encoding(h_matrix, implementation="dense_dilation")
    d_l = np.array([0.8, -1.3, 0.0, 0.4])
    d_h = np.array([1.0, 1.0, 0.0, 1.0])

    joint = build_control_diagonal_generator_encoding(
        encoding_l, encoding_h, l_diagonal=d_l, h_diagonal=d_h
    )
    assert joint.alpha == pytest.approx(
        1.3 * encoding_l.alpha + 1.0 * encoding_h.alpha, rel=1.0e-12, abs=0
    )
    assert joint.num_ancillas == max(encoding_l.num_ancillas, encoding_h.num_ancillas) + 2
    assert joint.system_qubits == 2 + 1
    block = _encoded_block(joint.circuit, joint.num_ancillas)
    target = np.kron(l_matrix, np.diag(d_l)) + np.kron(h_matrix, np.diag(d_h))
    assert np.max(np.abs(joint.alpha * block - target)) <= 1.0e-12

    # Single-term degenerations mirror the scalar seam's cases.
    single_l = build_control_diagonal_generator_encoding(encoding_l, None, l_diagonal=d_l)
    block_l = _encoded_block(single_l.circuit, single_l.num_ancillas)
    assert single_l.alpha == pytest.approx(1.3 * encoding_l.alpha, rel=1.0e-12, abs=0)
    assert np.max(np.abs(single_l.alpha * block_l - np.kron(l_matrix, np.diag(d_l)))) <= 1.0e-12
    single_h = build_control_diagonal_generator_encoding(
        encoding_l, encoding_h, l_diagonal=np.zeros(4), h_diagonal=d_h
    )
    block_h = _encoded_block(single_h.circuit, single_h.num_ancillas)
    assert np.max(np.abs(single_h.alpha * block_h - np.kron(h_matrix, np.diag(d_h)))) <= 1.0e-12

    with pytest.raises(ValueError, match="vanished"):
        build_control_diagonal_generator_encoding(encoding_l, None, l_diagonal=np.zeros(4))
    with pytest.raises(ValueError, match="power-of-two"):
        build_control_diagonal_generator_encoding(encoding_l, None, l_diagonal=np.ones(3))
    with pytest.raises(ValueError, match="same length"):
        build_control_diagonal_generator_encoding(
            encoding_l, encoding_h, l_diagonal=np.ones(4), h_diagonal=np.ones(2)
        )
    with pytest.raises(ValueError, match="must be real"):
        build_control_diagonal_generator_encoding(
            encoding_l, None, l_diagonal=np.array([1.0j, 0.0])
        )
    with pytest.raises(ValueError, match="exactly when"):
        build_control_diagonal_generator_encoding(
            encoding_l, encoding_h, l_diagonal=np.ones(4), h_diagonal=None
        )



def _blockwise_expm_reference(
    l_matrix: np.ndarray,
    h_matrix: np.ndarray,
    d_l: np.ndarray,
    d_h: np.ndarray,
    evolution_time: float,
) -> np.ndarray:
    """Independent scipy reference: ``sum_b kron(exp(-i T' (d1_b L + d2_b H)), |b><b|)``."""

    branch_count = d_l.size
    reference = np.zeros(
        (l_matrix.shape[0] * branch_count, l_matrix.shape[0] * branch_count), dtype=complex
    )
    for branch in range(branch_count):
        projector = np.zeros((branch_count, branch_count), dtype=complex)
        projector[branch, branch] = 1.0
        evolution = scipy.linalg.expm(
            -1.0j * evolution_time * (d_l[branch] * l_matrix + d_h[branch] * h_matrix)
        )
        reference += np.kron(evolution, projector)
    return reference



def test_compiled_select_matches_blockwise_expm_three_system_qubits_banded() -> None:
    """Evidence 1 at 3 system qubits, M = 2: the periodic heat matrix routes
    through the banded child (H = 0, single-child compiled SELECT).

    Method-error-bounded class: the encoded block matches an independent
    blockwise ``scipy.linalg.expm`` reference within the evolution encoding's
    documented operator-norm error bound.
    """

    shift=np.roll(np.eye(8),1,axis=0)
    heat=2*np.eye(8)-shift-shift.T
    encoding = build_block_encoding(heat)
    assert encoding.implementation == "banded"
    d_l = np.array([0.6, -0.35])
    joint = build_control_diagonal_generator_encoding(encoding, None, l_diagonal=d_l)
    evolution = build_qsp_evolution_encoding(
        joint,
        evolution_time=0.4,
        epsilon=1.0e-2,
    )
    block = _encoded_block_via_statevector(evolution.circuit, evolution.num_ancillas)
    reference = _blockwise_expm_reference(heat, np.zeros_like(heat), d_l, np.zeros(2), 0.4)
    assert float(np.linalg.norm(block - reference, 2)) <= evolution.error_bound



def test_qsp_oaa_raw_block_contract_and_compensated_metadata() -> None:
    """The generic block stays raw while recovery removes only its known deficit.

    Method-error-bounded class: the raw block is compared with ``lambda V``
    using the residual-only bound; the unchanged generic BlockEncoding
    contract compares the same block with ``V`` using deficit plus residual.
    """

    l_matrix = np.array([[0.5, 0.1], [0.1, 0.3]], dtype=complex)
    h_matrix = np.array([[0.2, 0.05j], [-0.05j, -0.1]], dtype=complex)
    encoding_l = build_block_encoding(l_matrix, implementation="dense_dilation")
    encoding_h = build_block_encoding(h_matrix, implementation="dense_dilation")
    d_l = np.array([1.7, -0.6])
    d_h = np.ones(2)
    evolution_time = 0.5
    joint = build_control_diagonal_generator_encoding(
        encoding_l,
        encoding_h,
        l_diagonal=d_l,
        h_diagonal=d_h,
    )
    evolution = build_qsp_evolution_encoding(
        joint,
        evolution_time=evolution_time,
        epsilon=1.0e-5,
    )
    block = _encoded_block_via_statevector(
        evolution.circuit,
        evolution.num_ancillas,
    )
    reference = _blockwise_expm_reference(
        l_matrix,
        h_matrix,
        d_l,
        d_h,
        evolution_time,
    )
    metadata = evolution.metadata
    amplitude = float(metadata["oaa_amplitude"])
    lambda_oaa = 3.0 * amplitude - 4.0 * amplitude**3
    deficit = float(metadata["oaa_amplitude_deficit"])
    residual_bound = float(metadata["oaa_residual_error_bound"])

    assert deficit == pytest.approx(1.0 - lambda_oaa, rel=1.0e-12, abs=0)
    assert metadata["recovery_scale"] == pytest.approx(1.0 / lambda_oaa, rel=1.0e-12, abs=0)
    assert metadata["compensated_recovery_error_bound"] == pytest.approx(
        residual_bound / lambda_oaa,
        rel=1.0e-12, abs=0,
    )
    assert evolution.error_bound == pytest.approx(
        deficit + residual_bound,
        rel=1.0e-15, abs=0,
    )
    assert float(np.linalg.norm(block - lambda_oaa * reference, 2)) <= residual_bound
    assert float(np.linalg.norm(block - reference, 2)) <= evolution.error_bound



def _count_structural_operations(circuit) -> Counter:
    """Count named operations recursively through gate definitions.

    ControlledGate wrappers are transparent (their base gate is visited);
    multiplexed rotations (``ucry``) are counted as leaves; branch units
    (``generator_branch_*``) are counted and then unrolled so the rotations
    inside them are reached; dense ``UnitaryGate`` leaves are not synthesized.
    Inverted units keep their name prefix (qiskit appends ``_dg``), so
    prefix matching covers the OAA's reversed pass. A branch that the
    whole-matrix route controls is a gate named ``c_`` and the branch name.
    """

    counts: Counter = Counter()

    def visit_operation(operation) -> None:
        if isinstance(operation, ControlledGate):
            visit_operation(operation.base_gate)
            return
        name = operation.name
        if name.startswith("ucry"):
            counts["ucry"] += 1
            counts["ucry_angles"] += len(operation.params)
            return
        for prefix in ("generator_branch_l", "generator_branch_h"):
            if name.startswith(prefix) or name.startswith("c_" + prefix):
                counts[prefix] += 1
                break
        if isinstance(operation, UnitaryGate):
            return
        definition = getattr(operation, "definition", None)
        if definition is not None:
            for instruction in definition.data:
                visit_operation(instruction.operation)

    for instruction in circuit.data:
        visit_operation(instruction.operation)
    return counts



@pytest.mark.parametrize("route", ["gatewise", "whole_matrix"])
def test_compiled_select_gate_count_law_conformance(route) -> None:
    """The analytic law matches implementation-derived counts (seeded
    instance): ucry gates, per-child branch applications, rotation angles,
    and the recorded query count are the exact law values on both dense
    control routes."""

    l_matrix, h_matrix = _seeded_l_h(2, 31)
    encoding_l = build_block_encoding(l_matrix, implementation="dense_dilation")
    encoding_h = build_block_encoding(h_matrix, implementation="dense_dilation")
    d_l = np.array([0.7, -0.4, 1.1, 0.0])
    d_h = np.array([1.0, 1.0, 1.0, 0.0])
    joint = build_control_diagonal_generator_encoding(
        encoding_l, encoding_h, l_diagonal=d_l, h_diagonal=d_h, dense_control_route=route
    )
    evolution = build_qsp_evolution_encoding(
        joint,
        evolution_time=0.6,
        epsilon=1.0e-3,
    )
    expansion = evolution.metadata["jacobi_anger"]
    law = compiled_select_resource_law(
        control_qubits=2,
        cos_degree=expansion["cos_degree"],
        sin_degree=expansion["sin_degree"],
    )
    assert evolution.metadata["block_encoding_queries"] == law["block_encoding_queries"]

    counts = _count_structural_operations(evolution.circuit)
    assert counts["ucry"] == law["multiplexed_rotation_gates"]
    assert counts["ucry_angles"] == law["multiplexed_rotation_angles"]
    assert (
        counts["generator_branch_l"] + counts["generator_branch_h"] == law["child_encoding_calls"]
    )
    assert counts["generator_branch_l"] == counts["generator_branch_h"]
    # Each branch unit carries exactly one multiplexed rotation and one
    # child-encoding call (verified once at the joint-encoding level).
    joint_counts = _count_structural_operations(joint.circuit)
    assert joint_counts["generator_branch_l"] == 1
    assert joint_counts["generator_branch_h"] == 1
    assert joint_counts["ucry"] == 2
    assert (
        joint.num_ancillas - max(encoding_l.num_ancillas, encoding_h.num_ancillas)
        == law["generator_ancillas_beyond_children"]
    )

    # Single-child law (H = 0): one child call and one multiplexed rotation
    # per query.
    single = build_control_diagonal_generator_encoding(encoding_l, None, l_diagonal=d_l)
    single_evolution = build_qsp_evolution_encoding(
        single,
        evolution_time=0.6,
        epsilon=1.0e-3,
    )
    single_expansion = single_evolution.metadata["jacobi_anger"]
    single_law = compiled_select_resource_law(
        control_qubits=2,
        cos_degree=single_expansion["cos_degree"],
        sin_degree=single_expansion["sin_degree"],
        child_count=1,
    )
    single_counts = _count_structural_operations(single_evolution.circuit)
    assert single_counts["ucry"] == single_law["multiplexed_rotation_gates"]
    assert single_counts["ucry_angles"] == single_law["multiplexed_rotation_angles"]

    with pytest.raises(ValueError, match="child_count"):
        compiled_select_resource_law(control_qubits=2, cos_degree=2, sin_degree=1, child_count=3)
    with pytest.raises(ValueError, match="nonnegative"):
        compiled_select_resource_law(control_qubits=-1, cos_degree=2, sin_degree=1)



def test_compiled_select_rotation_count_scaling_trend() -> None:
    """Scaling trend pin: multiplexed rotation angles per joint-encoding
    application double with each added control qubit (2 * 2**n_c for two
    children), in the law and in the constructed circuits alike."""

    l_matrix, h_matrix = _seeded_l_h(2, 41)
    encoding_l = build_block_encoding(l_matrix, implementation="dense_dilation")
    encoding_h = build_block_encoding(h_matrix, implementation="dense_dilation")
    rng = np.random.default_rng(42)
    angle_counts = []
    for control_qubits in (1, 2, 3):
        branches = 2**control_qubits
        joint = build_control_diagonal_generator_encoding(
            encoding_l,
            encoding_h,
            l_diagonal=rng.uniform(-1.0, 1.0, size=branches),
            h_diagonal=np.ones(branches),
        )
        counts = _count_structural_operations(joint.circuit)
        law = compiled_select_resource_law(
            control_qubits=control_qubits, cos_degree=1, sin_degree=1
        )
        per_query_angles = law["multiplexed_rotation_angles"] // law["block_encoding_queries"]
        assert counts["ucry_angles"] == per_query_angles == 2 * branches
        angle_counts.append(counts["ucry_angles"])
    assert [second / first for first, second in zip(angle_counts, angle_counts[1:])] == [
        2.0,
        2.0,
    ]



def test_compiled_qsp_dense_child_normalizes_the_kept_operator() -> None:
    """Compare the pruned dense child with an independently built encoding of the kept matrix and
    its own norm.
    """
    labels = [a + b + c for a in "IXYZ" for b in "IXYZ" for c in "IXYZ"]
    rng = np.random.default_rng(1234)
    coefficients = np.zeros(64)
    coefficients[0] = 10.0
    term_ids = rng.choice(np.arange(1, 64), size=45, replace=False)
    coefficients[term_ids[:35]] = rng.normal(scale=0.05, size=35)
    coefficients[term_ids[35:]] = rng.choice([-1, 1], size=10) * 9.0e-12
    l_part = SparsePauliOp(labels, coefficients).to_matrix()

    plan, _, _, _ = compiled_module._compiled_qsp_part_plans(
        l_part,
        np.zeros_like(l_part),
        l_norm=float(np.linalg.norm(l_part, 2)),
        max_bytes=256*1024**2,
    )
    kept = np.asarray(plan.decomposition.to_sparse_pauli_op().to_matrix())
    baseline = replace(plan_block_encoding(kept), error_bound=plan.error_bound)
    encoding = native_module._build_compiled_qsp_part_encoding(plan)
    baseline_encoding = build_block_encoding_from_plan(baseline)

    assert plan.implementation == baseline.implementation == "dense_dilation"
    assert plan.alpha == pytest.approx(np.linalg.norm(kept, 2), abs=1.0e-14)
    assert plan.alpha == pytest.approx(baseline.alpha, abs=1.0e-14)
    # Tiny coefficients are extracted from a matrix assembled from 46 terms
    # including an order-ten identity. Its binary64 reconstruction allowance
    # is absolute in those input units, not relative to the removed 9e-11.
    assert plan.error_bound == pytest.approx(sum(abs(coefficients[term_ids[35:]])), rel=0, abs=1e-12)
    np.testing.assert_allclose(
        block_encoding_top_left(
            Operator(encoding.circuit).data, num_ancillas=encoding.num_ancillas
        ),
        kept / plan.alpha,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        Operator(encoding.circuit).data, Operator(baseline_encoding.circuit).data
    )
    assert encoding.circuit.count_ops() == baseline_encoding.circuit.count_ops()
    assert encoding.circuit.depth() == baseline_encoding.circuit.depth()



def test_compiled_qsp_pruned_circulant_keeps_banded_route_without_dense_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove a tiny noncirculant term and poison dense reconstruction to preserve the kept
    banded route.
    """
    shift = np.roll(np.eye(4), 1, axis=0)
    kept = np.eye(4) + 0.2 * (shift + shift.conj().T)
    l_part = kept + 5.0e-13 * np.diag([1.0, -1.0, 1.0, -1.0])
    decompositions = 0
    original = compiled_module.decompose_matrix_to_pauli

    def counted(*args, **kwargs):
        nonlocal decompositions
        decompositions += 1
        return original(*args, **kwargs)

    def reject_dense_reconstruction(*_args, **_kwargs):
        raise AssertionError("kept structured child reconstructed densely")

    monkeypatch.setattr(compiled_module, "decompose_matrix_to_pauli", counted)
    monkeypatch.setattr(PauliDecomposition, "to_sparse_pauli_op", reject_dense_reconstruction)
    plan, _, _, _ = compiled_module._compiled_qsp_part_plans(
        l_part,
        np.zeros_like(l_part),
        l_norm=float(np.linalg.norm(l_part, 2)),
    )
    baseline = replace(plan_block_encoding(kept), error_bound=plan.error_bound)
    encoding = native_module._build_compiled_qsp_part_encoding(plan)
    baseline_encoding = build_block_encoding_from_plan(baseline)

    assert decompositions == 2
    assert plan.implementation == baseline.implementation == "banded"
    assert plan.alpha == baseline.alpha == pytest.approx(1.4, rel=2e-15, abs=0)
    assert plan.detail["periodic_gate_counts"] == baseline.detail["periodic_gate_counts"]
    np.testing.assert_allclose(
        block_encoding_top_left(
            Operator(encoding.circuit).data, num_ancillas=encoding.num_ancillas
        ),
        kept / plan.alpha,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        Operator(encoding.circuit).data, Operator(baseline_encoding.circuit).data
    )
    assert encoding.circuit.count_ops() == baseline_encoding.circuit.count_ops()
    assert encoding.circuit.depth() == baseline_encoding.circuit.depth()



@pytest.mark.parametrize("scale", (1.0e-9, 1.0, 1.0e9))
def test_compiled_qsp_banded_detection_preserves_small_kept_bands(scale: float) -> None:
    shift = np.roll(np.eye(4), 1, axis=0)
    l_part = scale * (1.0e-9 * np.eye(4) + 5.0e-13 * (shift + shift.T))
    plan, _, _, _ = compiled_module._compiled_qsp_part_plans(
        l_part,
        np.zeros_like(l_part),
        l_norm=float(np.linalg.norm(l_part, 2)),
    )
    encoding = native_module._build_compiled_qsp_part_encoding(plan)
    encoded = plan.alpha * block_encoding_top_left(
        Operator(encoding.circuit).data,
        num_ancillas=encoding.num_ancillas,
    )
    residual = float(np.linalg.norm(encoded - l_part, 2))

    assert plan.implementation == "banded"
    assert plan.source.offsets == (-1, 0, 1)
    expected_alpha = scale * 1.001e-9
    assert plan.alpha == pytest.approx(
        expected_alpha,
        rel=128.0 * np.finfo(float).eps,
        abs=0.0,
    )
    assert residual <= plan.error_bound + 128.0 * np.finfo(float).eps * np.linalg.norm(l_part, 2)



def test_compact_pauli_banded_detection_is_stable_and_never_enumerates_basis_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scalar_coefficient = 1.0541424899789856e-16 - 9.304680447082048e-16j
    scalar = PauliDecomposition(
        terms=(PauliTerm(label="I", coefficient=scalar_coefficient),),
        input_dimension=2,
        operator_dimension=2,
        num_qubits=1,
        atol=0.0,
    )
    paired_bands = {
        0: -8.474037969048256e-16,
        2: -1.3240588951029617e-16,
        1: 7.392697615555309e-16 + 1.6428117448960486e-16j,
        3: 7.392697615555309e-16 - 1.6428117448960486e-16j,
    }
    identity = np.eye(4)
    paired_matrix = sum(
        coefficient * np.roll(identity, offset, axis=0)
        for offset, coefficient in paired_bands.items()
    )
    paired = decompose_matrix_to_pauli(paired_matrix, atol=0.0, rtol=0.0)
    identity_8 = np.eye(8)
    shift_8 = np.roll(identity_8, 1, axis=0)
    full = decompose_matrix_to_pauli(
        identity_8 + 1.0e-7 * (shift_8 + shift_8.T), atol=0.0, rtol=0.0
    )
    missing_support = replace(
        full,
        terms=tuple(term for term in full.terms if term.label not in {"IXX", "IYY"}),
    )
    bands_32 = {
        0: 0.21568799735641267,
        8: 0.0007630092465915088 + 0.004466123708038179j,
        24: 0.0007630092465915088 - 0.004466123708038179j,
        5: 6.171335672566716e-10 + 6.56964366542855e-12j,
        27: 6.171335672566716e-10 - 6.56964366542855e-12j,
        1: 0.8805173539490923 + 0.5943918628427985j,
        31: 0.8805173539490923 - 0.5943918628427985j,
        16: 1.1466181098717675e-09,
    }
    identity_32 = np.eye(32)
    full_32 = decompose_matrix_to_pauli(
        sum(
            coefficient * np.roll(identity_32, offset, axis=0)
            for offset, coefficient in bands_32.items()
        ),
        atol=0.0,
        rtol=0.0,
    )
    missing_group = replace(
        full_32,
        terms=tuple(term for term in full_32.terms if _pauli_masks(term.label)[0] != 11),
    )

    for decomposition, operator, offsets, alpha in (
        (scalar, scalar_coefficient * np.eye(2), (0,), abs(scalar_coefficient)),
        (paired, paired_matrix, (-1, 0, 1, 2), sum(map(abs, paired_bands.values()))),
        (
            missing_support,
            missing_support.to_sparse_pauli_op().to_matrix(),
            (-1, 0, 1),
            1.0000002,
        ),
        (
            missing_group,
            missing_group.to_sparse_pauli_op().to_matrix(),
            None,
            None,
        ),
    ):
        if offsets is None:
            # A circulant's first column contains its shift coefficients.
            # Read the kept operator, since deleting a Pauli group can
            # introduce shifts absent from the original circulant input.
            dimension = operator.shape[0]
            offsets = tuple(
                sorted(
                    index - dimension if index > dimension // 2 else index
                    for index, value in enumerate(operator[:, 0])
                    if value != 0.0
                )
            )
            alpha = sum(abs(value) for value in operator[:, 0])
        # These rounded supplied tables may differ from a circulant by a
        # nonzero exact-rational residual. Approximation is explicitly chosen
        # here; automatic exact compact structure is checked below.
        plan = plan_block_encoding(decomposition, implementation="banded")
        reconstructed = sum(
            coefficient * np.roll(np.eye(operator.shape[0]), offset, axis=0)
            for offset, coefficient in zip(
                plan.source.offsets, plan.source.coefficients, strict=True
            )
        )
        floating_allowance = 128.0 * np.finfo(float).eps * np.linalg.norm(operator, 2)

        assert plan.implementation == "banded"
        assert plan.source.offsets == offsets
        assert plan.alpha == pytest.approx(
            alpha,
            rel=128.0 * np.finfo(float).eps,
            abs=0.0,
        )
        assert np.linalg.norm(operator - reconstructed, 2) <= (
            plan.error_bound + floating_allowance
        )

    num_qubits = 80
    decomposition = PauliDecomposition(
        terms=(PauliTerm(label="I" * num_qubits, coefficient=2.5),),
        input_dimension=1 << num_qubits,
        operator_dimension=1 << num_qubits,
        num_qubits=num_qubits,
        atol=0.0,
    )

    def bounded_range(*args):
        values = builtins.range(*args)
        if len(values) > 4 * num_qubits:
            raise AssertionError("compact Pauli planning enumerated basis states")
        return values

    monkeypatch.setattr(block_encoding_core, "range", bounded_range, raising=False)
    plan = plan_block_encoding(decomposition)

    assert plan.implementation == "banded"
    assert plan.source.offsets == (0,)
    assert plan.alpha == 2.5



def test_lcu_select_synthesizes_original_branches_once(monkeypatch) -> None:
    import nwqlib.subroutines.lcu.core as lcu_core

    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    unitary, _ = np.linalg.qr(matrix)

    original_controlled = lcu_core.controlled
    calls = []

    def counted(gate, *args, **kwargs):
        calls.append(Operator(gate).data)
        return original_controlled(gate, *args, **kwargs)

    monkeypatch.setattr(lcu_core, "controlled", counted)
    data = lcu_core.prepare_lcu_data([0.5j, -0.5], [unitary, np.eye(2)])
    lcu_core.build_lcu_select(data)
    assert len(calls) == 2
    for actual, expected in zip(calls, data.phase_adjusted_unitaries, strict=True):
        np.testing.assert_array_equal(actual, expected)



# An ordinary exception receives the branch note; a native panic derives from
# BaseException only and must propagate unchanged. Neither is retried.
@pytest.mark.parametrize("error_type", [QiskitError, _PanicException])
def test_controlled_branch_gate_propagates_native_failure_once(
    monkeypatch,
    error_type,
) -> None:
    import nwqlib.subroutines.lcu.core as lcu_core

    failure = error_type("TwoQubitWeylDecomposition: failed to diagonalize M2")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise failure

    monkeypatch.setattr(lcu_core, "controlled", fail)
    with pytest.raises(error_type) as caught:
        lcu_core._controlled_branch_gate(np.eye(2), num_control_qubits=1, branch_index=0)
    assert caught.value is failure
    assert len(calls) == 1
    if isinstance(failure, Exception):
        assert failure.__notes__ == [
            "LCU branch 0, matrix shape (2, 2), controls=1; NWQLib does not retry the synthesis."
        ]


@pytest.mark.parametrize('scale',[1e-9,1.,1e9])
def test_selected_qsp_pruned_h_and_near_antihermitian_l_keep_original_scale(scale,monkeypatch):
    from types import SimpleNamespace
    from nwqlib import LinearDynamics,plan
    from nwqlib.algorithms.lchs import LCHS,ProviderConfig
    from nwqlib.subroutines.qsp import evolution
    method=LCHS(hamiltonian_evolution_backend='qsp_block_encoding',
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    l_part=scale*np.diag([1e-9,2e-9])
    noise=scale*1e-26*np.array([[0.,1.],[1.,0.]])
    chosen=plan(LinearDynamics(A=l_part+1j*noise,initial_state=[1,0],time=.1/scale),method=method)
    selected=chosen._native['native_data'].qsp_plan
    assert selected['part_plans'][1].implementation=='exact_zero'
    assert selected['part_plans'][2]==pytest.approx(scale*1e-26,rel=3e-15,abs=0)
    assert sum(r.width for r in chosen.construction.program.registers)<=13
    seen=[]
    def capture(encoding,**kwargs):
        assert kwargs['_prepared'] is selected['prepared_evolution']
        assert encoding.metadata['target_operator_is_hermitian'] is True
        assert encoding.metadata['encoded_operator_is_hermitian'] is True
        seen.append(encoding.error_bound)
        return SimpleNamespace(circuit=encoding.circuit)
    monkeypatch.setattr(evolution,'build_qsp_evolution_encoding',capture)
    native_module.build_selected_evolution(selected)
    clean=dict(selected,physical_h_diagonal=np.zeros_like(selected['physical_h_diagonal']))
    native_module.build_selected_evolution(clean)
    assert seen[0]-seen[1]==pytest.approx(scale*1e-26,rel=1e-10,abs=0)
    monkeypatch.setattr(evolution,'prepare_qsp_evolution',lambda **kw: pytest.fail('pruned L reached phase fitting'))
    with pytest.raises(ValueError,match='nonzero Hermitian part L'):
        plan(LinearDynamics(A=scale*1e-26*np.eye(2)+1j*l_part,initial_state=[1,0],time=.1/scale),method=method)
