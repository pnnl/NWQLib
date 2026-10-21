"""Independent selected QSP composition, recovery, phase and resource relations."""
from __future__ import annotations
from collections import Counter
import numpy as np
import pytest
import scipy.linalg
from qiskit import QuantumCircuit,transpile
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from conftest import _encoded_block
import nwqlib.algorithms.lchs.native as native_module
from nwqlib.subroutines.block_encoding import build_block_encoding
from nwqlib.subroutines.qsp import build_control_diagonal_generator_encoding,compiled_select_resource_law
from nwqlib.subroutines.qsp.evolution import build_qsp_evolution_encoding
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
