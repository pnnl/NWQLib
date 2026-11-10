"""Independent selected QSP composition, recovery, phase and resource relations."""
from __future__ import annotations
from collections import Counter
from functools import reduce
from itertools import product
import numpy as np
import pytest
import scipy.linalg
from qiskit import QuantumCircuit,transpile
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from conftest import _encoded_block
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
    """A fully pruned H keeps its mass in the generator error; a pruned L is refused before any phase work.

    H = scale*1e-26 X is below the relative Pauli cutoff 1e-12 of the
    largest coefficient of L = scale*diag(1e-9, 2e-9), so the joint
    generator has no H child. The omitted D_Q (x) I (x) H has norm at most
    ||H|| = scale*1e-26, which must enter delta_S once at every scale, and
    the selected row's child term is T_e*delta_S. With L below the cutoff
    instead, planning refuses before coherent selection or phase fitting.
    """
    from nwqlib import LinearDynamics,plan
    from nwqlib.algorithms.lchs import LCHS,ProviderConfig,coherent_selection
    method=LCHS(ham_evo='qsp_block_encoding',
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    l_part=scale*np.diag([1e-9,2e-9])
    noise=scale*1e-26*np.array([[0.,1.],[1.,0.]])
    chosen=plan(LinearDynamics(A=l_part+1j*noise,initial_state=[1,0],time=.1/scale),method=method)
    prepared=chosen._native['native_data'].qsp_plan
    generator=prepared.generator
    assert 'h_child' not in generator.components and generator.target_operator_is_hermitian is True
    assert generator.error_terms['omitted_part']==pytest.approx(scale*1e-26,rel=3e-15,abs=0)
    assert generator.error_bound-generator.error_terms['homogeneous']==pytest.approx(scale*1e-26,rel=1e-10,abs=0)
    assert prepared.error['child']==prepared.evolution_time*generator.error_bound
    assert sum(r.width for r in chosen.construction.program.registers)<=13
    monkeypatch.setattr(coherent_selection,'select_coherent_route',
                        lambda *a,**kw: pytest.fail('pruned L reached coherent selection'))
    with pytest.raises(ValueError,match='nonzero Hermitian part L'):
        plan(LinearDynamics(A=scale*1e-26*np.eye(2)+1j*l_part,initial_state=[1,0],time=.1/scale),method=method)


def test_compiled_qsp_planning_checks_kept_children_not_select_tables():
    """Planning a dense L + iH on the QSP backend checks the Pauli transforms and the kept children.

    For the tridiagonal L = 2I - S - S^T and H = 0.3(S + S^T) on 7 qubits,
    S the shift, the Pauli transforms keep a small part of the d**2
    possible words, and planning forms no SELECT table. The Plan is made
    under max_bytes=4_000_000, far below the 50,561,024 bytes of two
    full-support children with their SELECT tables. 2,110,000 bytes cover
    the 128*d**2 = 2,097,152 bytes held for L, H, their coefficients and the
    dense reserve and the labels of the kept words of L, and refuse their
    planning arrays.
    """
    from nwqlib import LinearDynamics, NormSquared, plan
    from nwqlib.algorithms.lchs import LCHS

    d = 1 << 7
    shift = np.diag(np.ones(d - 1), 1)
    a = 2 * np.eye(d) - shift - shift.T + 0.3j * (shift + shift.T)
    state = np.zeros(d, complex)
    state[0] = 1

    def make(max_bytes):
        return plan(LinearDynamics(A=a, initial_state=state, time=0.5),
                    method=LCHS(ham_evo="qsp_block_encoding", max_bytes=max_bytes),
                    output=NormSquared(), shots=100, seed=1)

    assert make(4_000_000).method.max_bytes == 4_000_000
    with pytest.raises(ValueError, match="block-encoding Pauli planning arrays"):
        make(2_110_000)


@pytest.mark.parametrize("first, second, needed, operation", (
    ("band", "band", 36, "compact LCHS generator children"),
    ("pauli", "band", 206, "block-encoding Pauli planning arrays"),
    ("band", "pauli", 242, "block-encoding Pauli planning arrays")))
def test_compact_children_charge_the_first_child_once_under_either_byte_limit(first, second, needed, operation,
                                                                               monkeypatch):
    """The second compact child is planned beside the first child's m(q+16) bytes, counted once.

    On q = 2, the centered band parts keep two bands and the Pauli part three words, so a completed first child
    holds 36 or 54 bytes. The band planner forms no array, so two band children need exactly the 36 held bytes,
    and a Pauli child needs its planning arrays beside its labels and the held bytes. The same number of bytes
    suffices as the physical memory under max_bytes=None and as an explicit cap, and one byte less refuses both.
    """
    from types import SimpleNamespace
    import nwqlib._limits as limits
    from nwqlib.algorithms.lchs.compiled_selection import _compiled_qsp_compact_part_plans
    from nwqlib.operators import ingest_band, ingest_pauli
    parts = {"pauli": ingest_pauli((("XI", .2), ("IX", .1), ("YY", .3)), num_qubits=2),
             "band": ingest_band((-1, 0, 1), (-.3, .7, -.3), num_qubits=2)}
    hamiltonian = ingest_band((-1, 1), (-.2j, .2j), num_qubits=2) if second == "band" else parts["pauli"]
    generator = SimpleNamespace(num_qubits=2, dissipative=parts[first], hamiltonian=hamiltonian)
    for memory in (needed, needed - 1):
        monkeypatch.setattr(limits, "physical_memory_bytes", lambda memory=memory: memory)
        for cap in (None, memory):
            if memory == needed:
                assert all(_compiled_qsp_compact_part_plans(generator, unitary=False, max_bytes=cap,
                                                            max_select_work=None)[:2])
            else:
                with pytest.raises(ValueError, match=f"{operation} needs about {needed} bytes"):
                    _compiled_qsp_compact_part_plans(generator, unitary=False, max_bytes=cap, max_select_work=None)


def test_compiled_qsp_children_plan_their_kept_words_and_reuse_a_settled_dense_part(monkeypatch):
    """Each child of a dense L + iH encodes its kept operator, and a settled dense child forms no Pauli term.

    On three qubits a part with more than d**2/2 = 32 kept Pauli words is a
    dense dilation from the word count alone. A full-support part is then
    planned from the part itself. A word pruned at the shared cutoff 1e-12
    of the largest coefficient leaves the dense child the matrix of the
    kept words, and its error bound adds the pruned magnitude. An H below
    the cutoff is the exact-zero child whose mass is returned. A periodic
    band stays banded, and a tridiagonal part keeps its Pauli words. Each
    child's block, built from its Plan, is its kept operator over alpha. The
    work of the full-support pair is (64*q + 64)*d**2 for the two transforms,
    the pruning and both intake scans, plus the singular values of H, and
    one unit less is refused.
    """
    from qiskit.quantum_info import Operator
    from nwqlib._linalg_laws import singular_values_work
    from nwqlib.algorithms.lchs.compiled_selection import _compiled_qsp_part_plans
    from nwqlib.subroutines import pauli_decomposition
    from nwqlib.subroutines.block_encoding.core import block_encoding_top_left, build_block_encoding_from_plan

    q, d = 3, 8
    built = Counter()
    initialize = pauli_decomposition.PauliTerm.__init__

    def counted(self, **kwargs):
        built["terms"] += 1
        initialize(self, **kwargs)
    monkeypatch.setattr(pauli_decomposition.PauliTerm, "__init__", counted)

    def children(l_part, h_part, limit=None):
        built.clear()
        return _compiled_qsp_part_plans(l_part, h_part, l_norm=float(np.linalg.norm(l_part, 2)), max_select_work=limit)

    def assert_block(plan, operator):
        circuit = build_block_encoding_from_plan(plan).circuit
        block = block_encoding_top_left(Operator(circuit).data, num_ancillas=plan.num_ancillas)
        np.testing.assert_allclose(plan.alpha * block, operator, atol=1e-13, rtol=0)

    one = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
           "Z": np.diag([1, -1])}
    labels = ["".join(letters) for letters in product("IXYZ", repeat=q)]
    word = {label: np.kron(np.kron(one[label[0]], one[label[1]]), one[label[2]]) for label in labels}

    l_part, h_part = _seeded_l_h(d, 5)
    l_plan, h_plan, omitted, work = children(l_part, h_part)
    assert built["terms"] == 0 and omitted == 0.0
    assert l_plan.source is l_part and h_plan.source is h_part
    assert (l_plan.implementation, l_plan.decomposition, l_plan.alpha) == (
        "dense_dilation", None, float(np.linalg.norm(l_part, 2)))
    assert work == (64*q + 64)*d*d + singular_values_work(d)
    with pytest.raises(ValueError, match="exceeds max_work"):
        children(l_part, h_part, work - 1)
    assert children(l_part, h_part, work)[3] == work
    for plan, part in ((l_plan, l_part), (h_plan, h_part)):
        assert_block(plan, part)

    # 41 kept words of L, one word at 1e-16 of the identity coefficient 1.
    kept = np.eye(d) + sum((j % 7 + 1) / 64 * word[label] for j, label in enumerate(labels[1:41]))
    l_plan, h_plan, omitted, _ = children(kept + 1e-16 * word[labels[41]], h_part)
    assert built["terms"] == 0 and l_plan.implementation == "dense_dilation"
    assert l_plan.error_bound >= 1e-16 and h_plan.source is h_part
    assert_block(l_plan, kept)

    h_small = 1e-15 * h_part
    l_plan, h_plan, omitted, _ = children(l_part, h_small)
    assert built["terms"] == 0 and h_plan.implementation == "exact_zero"
    assert omitted == pytest.approx(sum(abs(np.trace(word[label] @ h_small)) / d for label in labels), rel=1e-12)

    shift = np.roll(np.eye(d), 1, axis=1)
    tridiagonal = 0.3 * (np.diag(np.ones(d - 1), 1) + np.diag(np.ones(d - 1), -1))
    l_plan, h_plan, _, _ = children(2 * np.eye(d) - shift - shift.T, tridiagonal)
    assert l_plan.implementation == "banded"
    assert l_plan.detail["structure_detection"] == "detected_from_pauli_decomposition"
    assert h_plan.implementation == "multiplexed_pauli"
    assert {term.label for term in h_plan.decomposition.terms} == {
        label for label in labels if abs(np.trace(word[label] @ tridiagonal)) > 1e-12}
    assert_block(l_plan, 2 * np.eye(d) - shift - shift.T)
    assert_block(h_plan, tridiagonal)


def test_compiled_qsp_dense_children_reuse_the_counted_words_and_the_unpruned_part(monkeypatch):
    """A dense child is not transformed again or rebuilt from terms when its matrix is already at hand.

    On four qubits a child with more than d**2/2 = 128 kept words is dense
    by that count. With all 256 words of L and H each part is transformed
    once. A 180-word L whose other words include one below the shared cutoff
    adds one inverse transform and no forward one. Neither forms a Pauli
    term, packs words or converts terms to a matrix, the auto records carry
    the kept count, and the unpruned children match plan_block_encoding of
    their parts, with the pruned mass named in the error evaluation. Those
    coefficients are random normals, so the transform of
    the 180-word part has rounding-level words at absent positions, which
    the cutoff prunes too, and the record still counts the 180 kept words. A
    64-word L with dyadic coefficients has no such words. Its terms are
    compared, and the comparison selects the dense dilation. Unpruned, it
    uses the part itself. With one more word pruned, its kept operator is
    rebuilt from the terms once, normalized by its own norm, and the ledger
    adds the m*d**2 rebuild and the singular values.
    """
    from qiskit.quantum_info import Operator
    from nwqlib._linalg_laws import singular_values_work
    from nwqlib.algorithms.lchs.compiled_selection import _compiled_qsp_part_plans
    from nwqlib.operators import _pauli
    from nwqlib.subroutines import pauli_decomposition
    from nwqlib.subroutines.block_encoding import core, plan_block_encoding
    from nwqlib.subroutines.block_encoding.core import block_encoding_top_left, build_block_encoding_from_plan

    q, d = 4, 16
    rng = np.random.default_rng(17)
    one = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
           "Z": np.diag([1, -1])}
    labels = ["".join(letters) for letters in product("IXYZ", repeat=q)]
    order = ["IIII", *(labels[k] for k in rng.permutation(np.arange(1, d * d)))]

    def word(label):
        return reduce(np.kron, (one[letter] for letter in label)).astype(complex)

    def pauli_sum(values, tiny=None):
        """Return a part with the first len(values) words, plus the next word at ``tiny`` times the largest, and its kept operator."""
        kept = sum(value * word(label) for value, label in zip(values, order))
        extra = 0 if tiny is None else tiny * np.abs(values).max() * word(order[len(values)])
        return kept + extra, kept

    calls = Counter()

    def counted(name, function):
        def traced(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return traced
    for owner, name in ((_pauli, "_block_transform"), (core, "_block_transform"), (_pauli, "_inverse_block_transform"),
                        (core, "_coefficient_masks"), (pauli_decomposition.PauliDecomposition, "to_sparse_pauli_op")):
        monkeypatch.setattr(owner, name, counted(name, getattr(owner, name)))
    initialize = pauli_decomposition.PauliTerm.__init__

    def counted_term(self, **kwargs):
        calls["PauliTerm"] += 1
        initialize(self, **kwargs)
    monkeypatch.setattr(pauli_decomposition.PauliTerm, "__init__", counted_term)

    def children(l_part, h_part, *, l_norm):
        calls.clear()
        l_plan, h_plan, _, work = _compiled_qsp_part_plans(l_part, h_part, l_norm=l_norm, max_select_work=None)
        return l_plan, h_plan, work

    def assert_block(plan, operator):
        circuit = build_block_encoding_from_plan(plan).circuit
        block = block_encoding_top_left(Operator(circuit).data, num_ancillas=plan.num_ancillas)
        np.testing.assert_allclose(plan.alpha * block, operator, atol=1e-13, rtol=0)

    l_part, _ = pauli_sum(rng.normal(size=256))
    h_part, _ = pauli_sum(rng.normal(size=256))
    l_norm = float(np.linalg.norm(l_part, 2))
    l_plan, h_plan, _ = children(l_part, h_part, l_norm=l_norm)
    assert calls == Counter({"_block_transform": 2})
    for plan, part, normalization in ((l_plan, l_part, l_norm), (h_plan, h_part, None)):
        public = plan_block_encoding(part, normalization=normalization)
        assert plan.source is part and plan.requested_implementation == "auto"
        assert (plan.alpha, plan.error_bound) == (public.alpha, public.error_bound)
        for key in ("term_count", "prep_pair_cx", "selection_reason", "normalization_records"):
            assert plan.detail[key] == public.detail[key]
        assert plan.detail["term_count"] == 256
        assert plan.detail["error_evaluation"] == (
            f"{public.detail['error_evaluation']}, plus the pruned Pauli-coefficient mass")
    assert_block(l_plan, l_part)

    pruned_part, kept = pauli_sum(rng.normal(size=180), tiny=1e-14)
    l_plan, _, _ = children(pruned_part, np.zeros((d, d)), l_norm=float(np.linalg.norm(pruned_part, 2)))
    assert calls == Counter({"_block_transform": 2, "_inverse_block_transform": 1})
    assert l_plan.implementation == "dense_dilation" and l_plan.detail["term_count"] == 180
    assert l_plan.detail["selection_reason"].startswith("auto selected dense_dilation: the PREP pair of the 180-term")
    assert l_plan.alpha == pytest.approx(np.linalg.norm(kept, 2), rel=1e-13)
    assert l_plan.error_bound >= np.abs(pruned_part - kept).max()
    assert_block(l_plan, kept)

    dyadic = (np.arange(64) % 7 + 1) / 64
    l_part, _ = pauli_sum(dyadic)
    l_norm = float(np.linalg.norm(l_part, 2))
    l_plan, _, unpruned_work = children(l_part, np.zeros((d, d)), l_norm=l_norm)
    assert l_plan.implementation == "dense_dilation" and l_plan.decomposition.terms
    assert calls["to_sparse_pauli_op"] == 0 and calls["PauliTerm"] >= 64 and l_plan.source is l_part
    public = plan_block_encoding(l_part, normalization=l_norm)
    assert (l_plan.alpha, l_plan.error_bound) == (public.alpha, public.error_bound)
    np.testing.assert_array_equal(l_plan.source, public.source)
    assert_block(l_plan, l_part)

    pruned_part, kept = pauli_sum(dyadic, tiny=1e-14)
    l_plan, _, pruned_work = children(pruned_part, np.zeros((d, d)), l_norm=l_norm)
    assert l_plan.implementation == "dense_dilation" and calls["to_sparse_pauli_op"] == 1
    assert l_plan.alpha == pytest.approx(np.linalg.norm(kept, 2), rel=1e-13)
    assert l_plan.error_bound >= np.abs(pruned_part - kept).max()
    assert pruned_work - unpruned_work == 64 * d * d + singular_values_work(d)
    assert_block(l_plan, kept)
