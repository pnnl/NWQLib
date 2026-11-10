"""Independent certificate, lowering, resolution, and resource tests."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from _lchs_suzuki_reference import sorted_pauli_lie_1_reference, sorted_pauli_suzuki_2_reference
from conftest import lchs_dense_quadrature
from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs import LCHS, SampledPairs, Trotter
from nwqlib.algorithms.lchs.native import (
    resolve_lcu_select_implementation,
)
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.algorithms.lchs.sampled_native import (
    build_pair_circuit,
    build_pair_template,
    pair_circuit_resource_law,
    pair_key,
    pair_select_plan,
)
from nwqlib.algorithms.lchs.select_synthesis import (
    build_multiplexed_product_formula_select,
    build_structured_product_formula_select,
)
from nwqlib.algorithms.lchs.time_independent_terms import (
    _TrotterPauliDecomposition,
    _build_product_formula_select_plan,
    generate_lchs_product_formula_select_plan,
)
from nwqlib.subroutines._multiplexors import (
    affine_angle_table_values,
    product_formula_select_resource_law,
)


def _signed_options(
    address_qubits: int,
    *,
    select: str = "auto",
    preparation: str = "direct",
    resource_tier: str = "measured",
) -> LCHS:
    return LCHS(
        lchs_kernel=ProviderConfig(implementation="cauchy_density"),
        k_quadrature=ProviderConfig(
            implementation="signed_binary_uniform",
            parameters={"num_qubits": address_qubits, "lsb_position": -2},
        ),
        ham_evo=Trotter(steps=1),
        lcu_select=select,
        lcu_prep=preparation,
    )


def _mixed_problem() -> LinearDynamics:
    return LinearDynamics(
        A=np.array([[0.2, -0.1], [-0.1, 0.3]], dtype=complex),
        initial_state=np.array([1.0, 0.25], dtype=complex),
        time=0.1,
    )


def _heat_problem() -> LinearDynamics:
    # Independent finite Dirichlet second difference; no PDE toy factory.
    matrix=.1*(2*np.eye(4)-np.eye(4,k=1)-np.eye(4,k=-1))
    return LinearDynamics(A=matrix,initial_state=[1,0,0,0],time=.1)


def _signed_address_structure(address_qubits: int) -> dict[str, object]:
    return {
        "affine": True,
        "kind": "twos_complement_affine",
        "num_qubits": address_qubits,
        "sign_bit": address_qubits - 1,
        "sign_coefficient": float(-(1 << (address_qubits - 1))),
        "unsigned_terms": tuple(
            {"qubit": qubit, "coefficient": float(1 << qubit)}
            for qubit in range(address_qubits - 1)
        ),
    }


@pytest.mark.parametrize(
    ("problem_factory", "address_qubits", "order"),
    ((_mixed_problem, 2, 1), (_mixed_problem, 2, 2), (_mixed_problem, 4, 2), (_heat_problem, 3, 2)),
)
def test_structured_and_multiplexed_select_match_independent_dense_blocks(
    problem_factory: object,
    address_qubits: int,
    order: int,
) -> None:
    problem = problem_factory()
    options = _signed_options(address_qubits).revise(ham_evo=Trotter(steps=1, order=order))
    quadrature, *parts = lchs_dense_quadrature(problem.A.dense_array(), problem.elapsed_time, options)
    data = generate_lchs_product_formula_select_plan(
        quadrature=quadrature,
        parts=tuple(parts),
        final_time=problem.elapsed_time,
        method=options,
    )
    plan = data.plan
    assert plan.structure_certificate["eligible"] is True
    structured_circuit = build_structured_product_formula_select(
        plan, num_system_qubits=int(np.log2(problem.dimension)),
    )
    structured = np.asarray(Operator(structured_circuit).data)
    law = product_formula_select_resource_law(plan, implementation="structured")
    assert structured_circuit.decompose(reps=10).count_ops().get("cx", 0) == law["select_basis_cx_count"]
    multiplexed = np.asarray(
        Operator(
            build_multiplexed_product_formula_select(
                plan,
                num_system_qubits=int(np.log2(problem.dimension)),
            )
        ).data
    )
    system_dimension = problem.dimension
    machine_tolerance = 64.0 * structured.shape[0] * np.finfo(float).eps
    matrix = problem.A.dense_array()
    l_part, h_part = (matrix + matrix.conj().T) / 2, (matrix - matrix.conj().T) / (2j)
    reference = sorted_pauli_lie_1_reference if order == 1 else sorted_pauli_suzuki_2_reference
    for branch in range(plan.padded_node_count):
        indices = [branch + (basis << address_qubits) for basis in range(system_dimension)]
        # The signed address uses lsb=2^-2; the Cauchy coefficients are real
        # positive. Reconstruct from the original A and mathematical product,
        # independently of the selected occurrence schedule and angle tables.
        signed = branch if branch < 1 << (address_qubits - 1) else branch - (1 << address_qubits)
        expected = reference(l_part, h_part, k_value=signed / 4,
            final_time=problem.elapsed_time, reps=1)
        assert np.max(np.abs(structured[np.ix_(indices, indices)] - expected)) <= (
            machine_tolerance
        )
        assert np.max(np.abs(multiplexed[np.ix_(indices, indices)] - expected)) <= (
            machine_tolerance
        )
    assert np.max(np.abs(structured - multiplexed)) <= machine_tolerance
    for affine, generic in zip(
        plan.structured_generator_payload["occurrence_affine_tables"],
        plan.occurrence_angle_tables.array,
        strict=True,
    ):
        reconstructed = affine_angle_table_values(
            offset=affine.offset,
            coefficients=affine.coefficients,
        )
        assert np.max(np.abs(np.asarray(reconstructed) - np.asarray(generic))) <= (affine.tolerance)


def test_structured_eligibility_is_produced_before_resolution() -> None:
    condition = "structure certificate reconstruction does not match generic angle tables"
    decomposition = _TrotterPauliDecomposition(
        num_qubits=1,
        identity_label="I",
        l_coefficients={"X": 1.0},
        h_coefficients={"X": 0.0},
    )
    nodes = np.array([0.0, 0.75, -2.0, -1.0])
    elapsed_times = np.full(4, 0.1)
    address = _signed_address_structure(2)
    plan, _, _ = _build_product_formula_select_plan(
        decomposition=decomposition,
        k_values=nodes,
        elapsed_times=elapsed_times,
        coefficients=np.ones(4, dtype=complex),
        branch_to_node=(0, 1, 2, 3),
        method=_signed_options(2),
        address_structure=address,
    )
    certificate = plan.structure_certificate
    assert certificate["eligible"] is False
    assert any(condition in reason for reason in certificate["rejection_reasons"])
    resolution = resolve_lcu_select_implementation(
        _signed_options(2),
        backend="trotter",
        select_plan=plan,
    )
    assert resolution == "multiplexor"
    with pytest.raises(ValueError, match=condition):
        resolve_lcu_select_implementation(
            _signed_options(2, select="structured"),
            backend="trotter",
            select_plan=plan,
        )


# Sampled LCHS pair circuits (sampled_native). Every expected value comes from
# scipy.linalg.expm products of dense Pauli matrices in the stated schedule
# order, from the exact branch exponential, or from direct expansion of the
# pair state, never from the circuit builder.

UNIT_ROUNDOFF = 2.0**-53
PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SDG = np.diag([1, -1j])

# Mixed kinds, unequal times, identity terms in both children.
L_MIXED = {"XI": 0.6, "ZZ": -0.4, "II": 0.15}
H_MIXED = {"ZZ": 0.3, "IY": 0.5, "YX": -0.25, "II": -0.2}
# Homogeneous equal-time pair: IY and XX are H-only labels, so their angles are equal.
L_HOMOGENEOUS = {"XI": 0.6, "ZZ": -0.4}
H_HOMOGENEOUS = {"ZZ": 0.3, "IY": 0.5, "XX": -0.25}


def dense(label):
    """Pauli matrix of a Qiskit high-to-low label."""
    matrix = np.ones((1, 1), dtype=complex)
    for factor in label:
        matrix = np.kron(matrix, PAULI[factor])
    return matrix


def rotation_preparation(angles, phase):
    """Circuit and vector of exp(i*phase) prod_j RZ(b_j) RY(a_j) on qubit j, from separate constructions."""
    circuit = QuantumCircuit(len(angles), global_phase=phase)
    for qubit, (a, b) in enumerate(angles):
        circuit.ry(a, qubit)
        circuit.rz(b, qubit)
    vector = np.ones(1, dtype=complex)
    for a, b in reversed(angles):
        vector = np.kron(vector, np.diag([np.exp(-0.5j * b), np.exp(0.5j * b)]) @ [np.cos(a / 2), np.sin(a / 2)])
    return circuit, np.exp(1j * phase) * vector


PREP_INITIAL, PSI_INITIAL = rotation_preparation([(0.7, 0.3), (1.9, -0.8)], 0.4)
PREP_SOURCE, PSI_SOURCE = rotation_preparation([(2.3, 1.2), (0.5, 0.9)], -1.3)
PREPARATIONS = (PREP_INITIAL, PREP_SOURCE)
DIRECTIONS = (PSI_INITIAL, PSI_SOURCE)


def product_formula(l_terms, h_terms, time, node, steps, order):
    """Ordered expm product of r Lie-1 or Suzuki-2 steps over the sorted labels, and its factor count."""
    width = len(next(iter({**l_terms, **h_terms})))
    identity = "I" * width
    labels = sorted((set(l_terms) | set(h_terms)) - {identity})
    coefficient = {label: node * l_terms.get(label, 0.0) + h_terms.get(label, 0.0) for label in labels}
    if order == 1:
        sequence = [(label, time / steps) for label in labels]
    else:
        half = [(label, time / (2 * steps)) for label in labels[:-1]]
        sequence = [*half, (labels[-1], time / steps), *reversed(half)]
    step = np.eye(2**width, dtype=complex)
    for label, duration in sequence:
        step = expm(-1j * duration * coefficient[label] * dense(label)) @ step
    unitary = np.eye(2**width, dtype=complex)
    for _ in range(steps):
        unitary = step @ unitary
    phase = np.exp(-1j * time * (node * l_terms.get(identity, 0.0) + h_terms.get(identity, 0.0)))
    return phase * unitary, len(sequence) * steps


def exact_evolution(l_terms, h_terms, time, node):
    """exp(-i*t*(k*L_c + H_c)) of the full branch generator."""
    width = len(next(iter({**l_terms, **h_terms})))
    generator = np.zeros((2**width, 2**width), dtype=complex)
    for label, value in l_terms.items():
        generator += node * value * dense(label)
    for label, value in h_terms.items():
        generator += value * dense(label)
    return expm(-1j * time * generator)


def expected_state(chi_x, chi_y, theta, term_label, copies):
    """Pre-measurement state from the pair state (|0>chi_x + |1>chi_y)/sqrt(2) and the readout matrices."""
    q = len(term_label)
    stride = 2**copies
    pair = np.zeros(2**q * stride, dtype=complex)
    pair[0::stride] = chi_x / np.sqrt(2)
    pair[1::stride] = chi_y / np.sqrt(2)
    system = np.ones((1, 1), dtype=complex)
    for axis in term_label:
        system = np.kron(system, {"X": HADAMARD, "Y": HADAMARD @ SDG}.get(axis, PAULI["I"]))
    branch = HADAMARD @ np.diag([1, np.exp(1j * theta)])
    return np.kron(system, np.kron(np.eye(stride // 2), branch)) @ pair


def measured_scores(pair):
    """E[s_a*s_P] and E[s_a] from the exact outcome probabilities, read through the circuit's measure wiring."""
    circuit = pair.circuit
    probabilities = np.abs(Statevector(circuit.remove_final_measurements(inplace=False)).data) ** 2
    wiring = {circuit.find_bit(item.qubits[0]).index: circuit.find_bit(item.clbits[0]).index
              for item in circuit.data if item.operation.name == "measure"}
    outcomes = np.array([sum(((index >> qubit) & 1) << bit for qubit, bit in wiring.items())
                         for index in range(probabilities.size)], dtype=np.int64)

    def sign(mask):
        return 1.0 - 2.0 * (np.bitwise_count(outcomes & mask) & 1)

    return (float(probabilities @ sign(pair.branch_mask | pair.parity_mask)),
            float(probabilities @ sign(pair.branch_mask)))


def roundoff_tolerance(circuit, oracle_factors):
    """32u per applied gate of the decomposed circuit and per oracle factor.

    To first order, applying one gate of width d <= 4 to a unit vector in
    binary64 adds at most sqrt(2)*u*sqrt(d) from rounding its entries plus
    sqrt(2)*gamma_(d+1)*sqrt(d) from the complex inner products, about 17u
    with u = 2**-53, and one expm factor product on the 4-dimensional system
    adds the same order. Allowing 32u per applied operation, counting every
    gate of the fully decomposed circuit and every oracle factor, gives a
    tolerance of order 1e-12 for these circuits. A probability expectation
    then differs by at most about twice the state error.
    """
    gates = sum(circuit.remove_final_measurements(inplace=False).decompose(reps=8).count_ops().values())
    return 32 * UNIT_ROUNDOFF * (gates + oracle_factors + 8)


def check_law(pair, plan, kinds, term_label, implementation, copies):
    """The law equals count_ops after one-level expansion of the SELECT multiplexors, without transpilation."""
    law = pair_circuit_resource_law(plan, kinds=kinds, term_label=term_label,
                                    implementation=implementation, control_copies=copies)
    expanded = pair.circuit.decompose(gates_to_decompose=["ucrz", "crz", "nwqlib_phase_diagonal"], reps=2)
    counts = dict(expanded.count_ops())
    preparations = {name: counts.pop(name) for name in list(counts) if "prep_" in name}
    assert counts == law["gate_counts"]
    assert sum(preparations.values()) == law["preparation_count"]
    assert sum(count for name, count in preparations.items() if name.startswith("cprep_")) == (
        law["controlled_preparation_count"])
    assert (pair.circuit.num_qubits, pair.circuit.num_clbits) == (law["qubits"], law["clbits"])
    return law


def mixed_case(implementation, steps=3, order=1):
    times, nodes, kinds, theta = (0.9, 0.35), (0.7, -1.3), (0, 1), 1.1
    plan = pair_select_plan(L_MIXED, H_MIXED, times=times, nodes=nodes, steps=steps, order=order,
                            implementation=implementation)
    branches = [product_formula(L_MIXED, H_MIXED, t, k, steps, order) for t, k in zip(times, nodes)]
    chi = [unitary @ DIRECTIONS[kind] for (unitary, _), kind in zip(branches, kinds)]
    factors = sum(count for _, count in branches)
    return plan, times, nodes, kinds, theta, chi, factors


@pytest.mark.parametrize("implementation", ["multiplexor", "structured"])
def test_mixed_pair_matches_ordered_expm_and_phase_aware_score(implementation):
    plan, times, nodes, kinds, theta, (chi_x, chi_y), factors = mixed_case(implementation)
    term = "XY"
    pair = build_pair_circuit(plan, preparations=PREPARATIONS, kinds=kinds, theta=theta, term_label=term,
                              implementation=implementation)
    tolerance = roundoff_tolerance(pair.circuit, factors)
    state = Statevector(pair.circuit.remove_final_measurements(inplace=False)).data
    assert np.max(np.abs(state - expected_state(chi_x, chi_y, theta, term, 1))) <= tolerance

    cross = np.vdot(chi_x, dense(term) @ chi_y)
    assert abs(cross.imag) > 0.05
    score_q, score_n = measured_scores(pair)
    assert abs(score_q - (np.exp(1j * theta) * cross).real) <= 3 * tolerance
    assert abs(score_q - (np.exp(-1j * theta) * cross).real) > 0.05
    assert abs(score_n - (np.exp(1j * theta) * np.vdot(chi_x, chi_y)).real) <= 3 * tolerance

    # The ordered expm product differs from the exact branch exponential, so
    # agreement above is with the product formula, not with U(t, k).
    for time, node, kind, chi in zip(times, nodes, kinds, (chi_x, chi_y)):
        bias = np.linalg.norm(chi - exact_evolution(L_MIXED, H_MIXED, time, node) @ DIRECTIONS[kind])
        assert bias > 1e3 * tolerance
    check_law(pair, plan, kinds, term, implementation, 1)


def test_direct_pair_select_plan_checks_the_tables_it_receives():
    """The public helper checks its own L_c and H_c tables and certifies affine tables for the structured SELECT."""
    arguments = dict(times=(0.9, 0.35), nodes=(0.7, -1.3), steps=1, order=1, implementation="structured")
    with pytest.raises(ValueError, match="finite and real"):
        pair_select_plan({"X": 1j}, {}, **arguments)
    plan = pair_select_plan({"X": 1.0}, {"Z": 2.0}, **arguments)
    assert plan.occurrence_schedule == ("X", "Z") and plan.structure_certificate["eligible"]
    # The Z row is the Lie multiplier 2 times t_b times its H coefficient 2.
    assert np.allclose(plan.occurrence_angle_tables.array[1], [4.0 * 0.9, 4.0 * 0.35])
    multiplexor = pair_select_plan({"X": 1.0}, {"Z": 2.0}, **(arguments | {"implementation": "multiplexor"}))
    assert np.array_equal(multiplexor.occurrence_angle_tables.array, plan.occurrence_angle_tables.array)
    # Exact NumPy integers are step and repetition counts.
    assert pair_select_plan({"X": 1.0}, {"Z": 2.0}, **(arguments | {"steps": np.int64(1)})) == plan
    assert pair_select_plan({"X": 1.0}, {"Z": 2.0}, **(arguments | {"repetitions": np.int32(1)})) == plan


def test_a_multiplexor_pair_plan_builds_only_the_multiplexor():
    """A structured build of a plan without a certificate is refused, and its multiplexor is the structured plan's."""
    common = dict(times=(0.8, 0.8), nodes=(0.4, -0.9), steps=1, order=1)
    bare, plan = (pair_select_plan(L_HOMOGENEOUS, H_HOMOGENEOUS, implementation=implementation, **common)
                  for implementation in ("multiplexor", "structured"))
    arguments = dict(preparations=PREPARATIONS, kinds=(0, 0), theta=-0.7, term_label="ZX")
    with pytest.raises(ValueError, match="eligible structure certificate"):
        build_pair_circuit(bare, implementation="structured", **arguments)
    built, direct = (build_pair_circuit(item, implementation="multiplexor", **arguments).circuit
                     .remove_final_measurements(inplace=False) for item in (bare, plan))
    assert np.max(np.abs(Operator(built).data - Operator(direct).data)) == 0.0


@pytest.mark.parametrize("implementation", ["multiplexor", "structured"])
def test_sampled_pair_row_templates_bind_to_their_direct_circuits(implementation):
    """Each row's bound template of a pair SELECT is its circuit."""
    import nwqlib
    from nwqlib import NormSquared
    from nwqlib.algorithms.lchs.sampled_native import pair_template_size
    from nwqlib.algorithms.lchs.sampled_plan import _row_inputs, _row_words, construct_pair, point_row
    from nwqlib.operators import ingest_pauli

    linear = LinearDynamics(A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
                            initial_state=[.6, .8j], time=.7)
    chosen = nwqlib.plan(linear, method=LCHS(estimator=SampledPairs(repeats=2),
                                             ham_evo=Trotter(steps=2),
                                             lcu_select=implementation),
                         output=NormSquared(), shots=12, seed=11)
    block = next(item for item in chosen.blocks if item.record.signature.name == "lchs_pair")
    payload, rec, _ = block._payload
    word = _row_words(payload, rec)
    points = tuple(chosen._static_points())
    rows_of = {point.key: point_row(point) for point in points}
    templates = chosen.method.row_templates(chosen, points)
    for template in templates:
        inputs = _row_inputs(payload, rec.pair_rows, word, rows_of[template.rows[0][0]])
        entries, crz, size = pair_template_size(pair_key(prepares=payload.prepares_inputs, **inputs)[0])
        assert (entries, crz) == (len(template.circuit.data), template.circuit.count_ops().get("crz", 0))
        assert template._retained_bytes == size
        for key, values in template.rows:
            bound = template.circuit.assign_parameters(dict(zip(template.parameters, values)))
            direct = construct_pair(block, (("row", rows_of[key]),), None)
            assert np.max(np.abs(Operator(bound).data - Operator(direct).data)) <= 1e-12


# One of each form of `_append_correction`, as (order, phase, x, z, angle). Low
# orders put a frame around a CRZ outside the SELECT, emitted even for angle 0.
# Order 0 adds nothing, a higher order adds controlled Paulis and a phase when it
# has one, and a label word of identity leaves only one phase.
TEMPLATE_CORRECTIONS = (
    ((2, 0, 0b01, 0b10, .3), (3, 2, 0b11, 0b00, 0.)),
    ((0, 0, 0, 0, 0.), (4, 1, 0b10, 0b11, 0.)),
    ((4, 0, 0b01, 0b00, 0.), (2, 1, 0, 0, .4)),
)


def _template_case(case, implementation):
    """Return the SELECT plan, the `pair_key` arguments and the input preparations of one tiny pair."""
    tables, times, nodes, steps, order = (L_MIXED, H_MIXED), (.9, .35), (.7, -1.3), 3, 1
    arguments = dict(kinds=(0, 1), theta=1.1, term_label="XY", implementation=implementation)
    preparations = PREPARATIONS
    if case == "repetition blocks and control copies":
        order, arguments["control_copies"] = 2, 3
    elif case == "no inputs":
        preparations = None
        arguments.update(kinds=(0, 0), term_label="II")
    elif case == "omitted angles":
        # XI has opposite nodes, so its mean angle is 0, and IY has equal times, so
        # its difference is 0. With no identity term there is no diagonal, and the
        # first kept angle takes slot 0, which counts like any other slot.
        tables, times, nodes = ({"XI": .6}, {"IY": .5}), (.8, .8), (.4, -.4)
        arguments.update(kinds=(0, 0), term_label="ZI")
    elif case == "outside corrections":
        steps = 1
        arguments.update(corrections=TEMPLATE_CORRECTIONS, control_copies=2)
    elif case == "frame-free label":
        # A multiplexor occurrence of Z emits four entries, against the one UCRZ
        # that a separate SELECT holds, so that SELECT's check runs low here.
        tables, steps, preparations = ({"Z": .6}, {"Z": .1}), 8, None
        arguments.update(kinds=(0, 0), term_label="Z")
    plan = pair_select_plan(*tables, times=times, nodes=nodes, steps=steps, order=order, implementation=implementation)
    return plan, arguments, preparations


@pytest.mark.parametrize("implementation", ["multiplexor", "structured"])
@pytest.mark.parametrize("case", ["mixed inputs", "repetition blocks and control copies", "no inputs", "omitted angles",
                                  "outside corrections", "frame-free label"])
def test_pair_template_size_counts_its_circuit_and_is_checked_before_the_walk(case, implementation, monkeypatch):
    """The count from a pair's key equals the circuit it builds, and that circuit's allowance is its limit.

    Independent relation: the built circuit's own entries N and CRZ entries C give
    16384 + 256*N + 1024*C (sampled_native.pair_template_size derives N and C from
    the key alone and states what the allowance leaves out). One byte below it
    refuses with the SELECT never walked, and at it the same circuit is built.
    """
    from nwqlib.algorithms.lchs import sampled_native

    plan, arguments, preparations = _template_case(case, implementation)
    key, values = pair_key(plan, prepares=preparations is not None, **arguments)
    template = dict(parameters=tuple(ParameterVector("slot", len(values))), preparations=preparations, **arguments)
    built = build_pair_template(plan, **template)
    crz = built.count_ops().get("crz", 0)
    allowance = 16384 + 256*len(built.data) + 1024*crz
    with monkeypatch.context() as patched:
        patched.setattr(sampled_native, "_walk_select", lambda *a, **k: pytest.fail("the SELECT was walked"))
        with pytest.raises(ValueError, match="sampled pair template needs about .* exceeding max_bytes"):
            build_pair_template(plan, max_bytes=allowance - 1, **template)
    assert len(build_pair_template(plan, max_bytes=allowance, **template).data) == len(built.data)
    assert sampled_native.pair_template_size(key) == (len(built.data), crz, allowance)
    if case == "outside corrections":
        assert sum(entry[0] == "crz" for entry in key[0]) == 2


@pytest.mark.parametrize("term", ["ZX", "II"])
def test_homogeneous_equal_time_pair_uses_uncontrolled_h_rotations(term):
    times, nodes, kinds, theta, steps = (0.8, 0.8), (0.4, -0.9), (0, 0), -0.7, 2
    branches = [product_formula(L_HOMOGENEOUS, H_HOMOGENEOUS, t, k, steps, 2) for t, k in zip(times, nodes)]
    chi_x, chi_y = (unitary @ PSI_INITIAL for unitary, _ in branches)
    laws = {}
    for implementation in ("multiplexor", "structured"):
        plan = pair_select_plan(L_HOMOGENEOUS, H_HOMOGENEOUS, times=times, nodes=nodes, steps=steps, order=2,
                                implementation=implementation)
        pair = build_pair_circuit(plan, preparations=PREPARATIONS, kinds=kinds, theta=theta, term_label=term,
                                  implementation=implementation)
        tolerance = roundoff_tolerance(pair.circuit, sum(count for _, count in branches))
        state = Statevector(pair.circuit.remove_final_measurements(inplace=False)).data
        assert np.max(np.abs(state - expected_state(chi_x, chi_y, theta, term, 1))) <= tolerance
        score_q, _ = measured_scores(pair)
        assert abs(score_q - (np.exp(1j * theta) * np.vdot(chi_x, dense(term) @ chi_y)).real) <= 3 * tolerance
        laws[implementation] = check_law(pair, plan, kinds, term, implementation, 1)
    # Suzuki-2 over IY, XI, XX, ZZ has two IY and two XX occurrences per step.
    assert laws["structured"]["equal_angle_occurrence_count"] == 4 * steps
    assert laws["multiplexor"]["gate_counts"]["cx"] - laws["structured"]["gate_counts"]["cx"] == 2 * 4 * steps


@pytest.mark.parametrize("copies", [2, 3])
@pytest.mark.parametrize("implementation", ["multiplexor", "structured"])
def test_control_copies_keep_the_pair_action_and_return_clean(copies, implementation):
    plan, _, _, kinds, theta, (chi_x, chi_y), factors = mixed_case(implementation, steps=2, order=2)
    pair = build_pair_circuit(plan, preparations=PREPARATIONS, kinds=kinds, theta=theta, term_label="YZ",
                              implementation=implementation, control_copies=copies)
    tolerance = roundoff_tolerance(pair.circuit, factors)
    state = Statevector(pair.circuit.remove_final_measurements(inplace=False)).data
    assert np.max(np.abs(state - expected_state(chi_x, chi_y, theta, "YZ", copies))) <= tolerance
    by_copy = state.reshape(4, 2 ** (copies - 1), 2)
    assert np.max(np.abs(by_copy[:, 1:, :])) <= tolerance
    law = check_law(pair, plan, kinds, "YZ", implementation, copies)
    single = pair_circuit_resource_law(plan, kinds=kinds, term_label="YZ", implementation=implementation)
    assert law["gate_counts"]["cx"] - single["gate_counts"]["cx"] == 2 * (copies - 1)
