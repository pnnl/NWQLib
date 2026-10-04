"""Natural input, physical phase, source association and compact-data witnesses."""

from dataclasses import FrozenInstanceError, replace
from math import ldexp, sqrt

import numpy as np
import pytest
from scipy import sparse

from nwqlib.operators import (
    OperatorInput, PeriodicStencil, ingest_pauli, ingest_pauli_masks,
    operator_input, pauli_table,
)
from nwqlib.problems.inputs import (
    PhysicalScale, StateInput, compose_recovery, ingest_occupation, ingest_product,
    state_input,
)


def test_dense_pauli_transform_independent_trace_and_safe_average():
    from nwqlib.operators._pauli import _half_sum, pauli_coefficients
    matrices = dict(I=np.eye(2), X=np.array([[0, 1], [1, 0]]),
                    Y=np.array([[0, -1j], [1j, 0]]), Z=np.diag([1., -1.]))
    h = .2*np.kron(matrices["I"], matrices["Y"])-.3*np.kron(matrices["X"], matrices["Z"])
    coefficients = dict(pauli_coefficients(h))
    for left, a in matrices.items():
        for right, b in matrices.items():
            assert coefficients.get(left+right, 0.) == pytest.approx(np.trace(np.kron(a, b)@h)/4, rel=0, abs=1e-16)
    limit, tiny = np.finfo(float).max, np.nextafter(0., 1.)
    np.testing.assert_array_equal(_half_sum(np.array([limit, tiny]), np.array([limit, tiny])), [limit, tiny])


@pytest.mark.parametrize("source", [lambda: [[2, 1j], [-1j, -3]],
    lambda: np.array([[2, 1j], [-1j, -3]]),
    lambda: sparse.csr_array([[2, 1j], [-1j, -3]]),
    lambda: sparse.csc_matrix([[2, 1j], [-1j, -3]])])
def test_natural_operator_snapshot_and_original_action(source, monkeypatch):
    value = source()
    monkeypatch.setattr(sparse.csr_array, "toarray", lambda *a, **k: pytest.fail("implicit dense conversion"))
    operator = operator_input(value)
    identity = operator.reference
    if isinstance(value, list):
        value[0][1] = 9
    elif sparse.issparse(value):
        value.data[:] = 9
    else:
        value[:] = 9
    # [2*1+i*2i, -i*1-3*2i] = [0,-7i], independent of the storage form.
    np.testing.assert_array_equal(operator.matvec(np.array([1, 2j])), [0, -7j])
    assert operator.entry(0, 1) == 1j and operator.structure == "hermitian"
    assert operator_input(operator) is operator and operator.reference == identity
    with pytest.raises(FrozenInstanceError):
        operator.structure = "general"
    with pytest.raises(TypeError, match="factory"):
        replace(operator, structure="general")


def test_numeric_conversion_domain_and_exact_structure():
    integer = operator_input([[1, 2], [0, 3]])
    assert integer.dense_array().dtype == np.dtype("float64")
    assert integer.structure == "general"
    near = operator_input([[1., 1e-16], [0., 2.]])
    assert near.structure == "general" and near.entry(0, 1) == 1e-16
    for bad in ([[1, 2], [3]], [[float("nan")]]):
        with pytest.raises((ValueError, TypeError)):
            operator_input(bad)
    duplicate = sparse.csr_array((np.array([1, 2]), np.array([0, 0]), np.array([0, 2, 2])), shape=(2, 2))
    with pytest.raises(ValueError, match="canonical"):
        operator_input(duplicate)
    sparse_integer = operator_input(sparse.csr_array([[1, 0], [0, 2]], dtype=np.int16))
    assert sparse_integer.manifest.dtype == "float64"


def test_unknown_protocols_and_size_refuse_before_conversion(monkeypatch):
    import nwqlib.operators.inputs as inputs
    import nwqlib.problems.inputs as states
    class Unknown:
        def __array__(self, *args, **kwargs):
            pytest.fail("unknown array protocol")
        def __iter__(self):
            pytest.fail("unknown iterator")
    for adapter in (operator_input, state_input):
        with pytest.raises(TypeError):
            adapter(Unknown())
    monkeypatch.setattr(inputs, "_freeze_array", lambda *a: pytest.fail("copy before byte admission"))
    monkeypatch.setattr(states, "_freeze_array", lambda *a: pytest.fail("copy before byte admission"))
    for adapter, value in ((operator_input, [[1, 0], [0, 1]]),
                           (state_input, [3, 4j]), (ingest_product, [[1, 0], [0, 1]])):
        with pytest.raises(ValueError, match="max_bytes"):
            adapter(value, max_bytes=1)


def test_state_physical_phase_normalizes_once_and_metadata_cannot_rebind(monkeypatch):
    import nwqlib.problems.inputs as states
    calls = []
    original = states.normalize_physical_vector_with_scale
    def normalize(value):
        calls.append(value.shape)
        return original(value)
    monkeypatch.setattr(states, "normalize_physical_vector_with_scale", normalize)
    source = np.array([3, 4j])
    state = state_input(source)
    source[:] = 0
    assert state.preparation.physical_scale.as_float() == 5
    np.testing.assert_array_equal(state.physical_vector(), [3, 4j])
    np.testing.assert_allclose(state._direction, [.6, .8j], rtol=0, atol=2e-16)
    scaled = state_input(np.array([-6j, 8]))
    assert scaled.preparation.physical_scale.as_float() == 10
    np.testing.assert_allclose(scaled._direction, [-.6j, .8], rtol=0, atol=2e-16)
    assert state_input(state) is state and calls == [(2,), (2,)]
    for array in (state._physical, state._direction):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    stored = StateInput.from_record(state.to_record())
    assert stored.to_record() == state.to_record() and calls == [(2,), (2,)]
    for access in (stored.physical_vector, lambda: stored.entry(0)):
        with pytest.raises(ValueError, match="native data is unavailable"):
            access()
    changed = state.to_record()
    changed["preparation"]["input"]["identity"] = "another state"
    with pytest.raises(ValueError, match="identity or basis"):
        StateInput.from_record(changed)


def test_operator_metadata_never_restores_native_access(monkeypatch):
    import nwqlib.operators.inputs as inputs
    operator = operator_input([[1, 0], [0, 2]])
    monkeypatch.setattr(inputs, "_digest", lambda *a: pytest.fail("metadata rehash"))
    restored = OperatorInput.from_record(operator.to_record())
    assert restored.reference == operator.reference and restored.basis == operator.basis
    assert restored.to_record() == operator.to_record()
    for access in (restored.dense_array, restored.pauli_terms, restored.fermion_terms,
                   restored.periodic_stencil, restored.sparse_entries,
                   lambda: restored.entry(0, 0), lambda: restored.matvec(None)):
        with pytest.raises(ValueError, match="native data is unavailable"):
            access()


def test_zero_subnormal_and_compact_state_scales():
    zero = state_input([0, 0])
    assert zero.preparation.physical_scale.as_float() == 0 and zero._direction is None
    tiny = ldexp(1., -1074)
    vector = state_input([tiny, tiny])
    scale = vector.preparation.physical_scale
    assert ldexp(scale.mantissa, scale.exponent + 1074) == pytest.approx(sqrt(2), rel=3e-16, abs=0)
    product = ingest_product([[tiny, tiny], [tiny, tiny]])
    scale = product.preparation.physical_scale
    # (sqrt(2)*2**-1074)**2 = 2**-2147; compare outside underflow.
    assert ldexp(scale.mantissa, scale.exponent + 2147) == pytest.approx(1, rel=9e-16, abs=0)
    bits = ingest_occupation("10", num_qubits=2)
    assert bits._physical.tolist() == [1, 0]
    assert bits.basis.dimension == 4 and bits.preparation.physical_scale.as_float() == 1
    assert compose_recovery(2., (3., 4.)).as_float() == 1.5
    with pytest.raises(ValueError, match="unit norm"):
        PhysicalScale(mantissa=.75, exponent=1, evidence="supplied_unitary_contract")


def test_compact_hundred_qubit_inputs_never_allocate_system_data(monkeypatch):
    monkeypatch.setattr(np, "zeros", lambda *a, **k: pytest.fail("system zero array"))
    monkeypatch.setattr(np, "eye", lambda *a, **k: pytest.fail("system matrix"))
    stencil = operator_input(PeriodicStencil(100, .1, .1, .05))
    basis = ingest_occupation((0,)*100, num_qubits=100)
    product = ingest_product([[1, 0]]*100)
    assert stencil.basis.dimension == basis.basis.dimension == product.basis.dimension == 1 << 100
    assert stencil.manifest.access == () and stencil.periodic_stencil().potential == .05
    assert product._physical.shape == product._direction.shape == (100, 2)
    with pytest.raises(ValueError, match="max_bytes"):
        operator_input(PeriodicStencil(10**10, 1, 1), max_bytes=64)


def test_pauli_residue_word_order_and_expansion_bound(monkeypatch):
    operator = ingest_pauli([("IX", 1), ("IX", 2**-55), ("IX", -1), ("YI", -2), ("II", 3)], num_qubits=2)
    assert operator.pauli_terms().labels() == (("IX", 2**-55), ("YI", -2), ("II", 3))
    np.testing.assert_array_equal(operator.matvec(np.array([1., 0., 0., 0.])), [3, 2**-55, -2j, 0])
    a = pauli_table([("X"+"I"*99, 1)], num_qubits=100)
    b = pauli_table([("Y"+"I"*99, 1)], num_qubits=100)
    assert a.product(b).labels() == (("Z"+"I"*99, 1j),)
    native = ingest_pauli_masks(a.x, a.z, a.coefficients, num_qubits=100)
    assert native.reference == ingest_pauli(a.labels(), num_qubits=100).reference
    monkeypatch.setattr(np, "empty", lambda *a, **k: pytest.fail("product allocation before size check"))
    with pytest.raises(ValueError, match="max_bytes"):
        a.product(b, max_bytes=1)


def test_grouping_and_action_product_limits_have_legal_counterparts(monkeypatch):
    from nwqlib.operators import _pauli

    # Count the candidate rows that the compatibility kernels actually
    # evaluate. Grouping charges each tile before comparing it
    # (PauliTerms.group), so a refusal evaluates no candidate beyond its cap.
    evaluated = {"rows": 0, "cap": None}

    def counted(kernel):
        def compare(xa, za, xb, zb):
            evaluated["rows"] += len(xb)
            if evaluated["cap"] is not None and evaluated["rows"] > evaluated["cap"]:
                pytest.fail("grouping compared candidates beyond max_comparisons")
            return kernel(xa, za, xb, zb)
        return compare

    monkeypatch.setattr(_pauli, "qwc", counted(_pauli.qwc))
    monkeypatch.setattr(_pauli, "anticommutes", counted(_pauli.anticommutes))

    def refused(table, strategy, cap, message):
        evaluated.update(rows=0, cap=cap)
        with pytest.raises(ValueError, match=message):
            table.group(strategy=strategy, max_comparisons=cap)
        evaluated["cap"] = None

    table = pauli_table([("XX", 1), ("YY", -2), ("ZZ", 3), ("XI", 4)], num_qubits=2)
    # Every evaluated candidate tile is counted in full, including groups
    # after the first compatible one (PauliTerms.group, "comparison_count").
    evaluated["rows"] = 0
    qwc = table.group(strategy="qwc", max_comparisons=6)
    assert (qwc.groups, qwc.comparison_count, evaluated["rows"]) == (((0, 3), (1,), (2,)), 6, 6)
    evaluated["rows"] = 0
    commuting = table.group(strategy="commuting", max_comparisons=6)
    assert (commuting.groups, commuting.comparison_count, evaluated["rows"]) == (((0, 1, 2), (3,)), 6, 6)
    refused(table, "qwc", 5, "max_comparisons=5")
    refused(table, "commuting", 5, "max_comparisons=5")
    # One QWC group needs L-1 comparisons and is admitted at that cap.
    labels = [format(k, "011b").replace("0", "I").replace("1", "Z") for k in range(1, 2001)]
    diagonal = pauli_table([(label, 1) for label in labels], num_qubits=11)
    grouped = diagonal.group(strategy="qwc", max_comparisons=1999)
    assert (len(grouped.groups), grouped.groups[0][:3], grouped.comparison_count) == (1, (0, 1, 2), 1999)
    distinct = pauli_table([("X", 1), ("Y", 1), ("Z", 1)], num_qubits=1)
    refused(distinct, "qwc", 2, "max_comparisons=2")
    operator = operator_input([[1, 2], [3, 4]])
    np.testing.assert_array_equal(operator.matvec(np.array([1., -1.]), max_products=4), [-1, -1])
    with pytest.raises(ValueError, match="max_products"):
        operator.matvec(None, max_products=3)


def test_packed_product_phase_and_compatibility_match_explicit_matrices():
    from nwqlib.operators._pauli import anticommutes, qwc

    local = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
             "Z": np.diag([1., -1.])}
    single = {}
    for a in "IXYZ":
        for b in "IXYZ":
            left, right = pauli_table([(a, 1)], num_qubits=1), pauli_table([(b, 1)], num_qubits=1)
            ((label, phase),) = left.product(right).labels()
            np.testing.assert_array_equal(local[a] @ local[b], phase * local[label])
            single[a, b] = label, phase
            commute = np.array_equal(local[a] @ local[b], local[b] @ local[a])
            assert bool(anticommutes(left.x[0], left.z[0], right.x[0], right.z[0])) is not commute
            assert bool(qwc(left.x[0], left.z[0], right.x[0], right.z[0])) is (a == b or "I" in (a, b))
    # Two-word tables against the qubitwise product of the explicit factors.
    rng = np.random.default_rng(3)
    q = 70
    left = ["".join(rng.choice(list("IXYZ"), q)) for _ in range(5)]
    right = ["".join(rng.choice(list("IXYZ"), q)) for _ in range(4)]
    product = pauli_table([(p, 1) for p in left], num_qubits=q).product(
        pauli_table([(p, 1) for p in right], num_qubits=q)).labels()
    for index, (p, r) in enumerate((p, r) for p in left for r in right):
        factors = [single[a, b] for a, b in zip(p, r, strict=True)]
        expected = complex(np.prod([phase for _, phase in factors]))
        assert product[index] == ("".join(label for label, _ in factors), expected)
    words = left + right
    table = pauli_table([(p, 1) for p in words], num_qubits=q)
    for i, word in enumerate(words):
        odd = [sum(a != "I" != b != a for a, b in zip(word, other, strict=True)) % 2 == 1 for other in words]
        same = [all(a == b or "I" in (a, b) for a, b in zip(word, other, strict=True)) for other in words]
        assert anticommutes(table.x[i], table.z[i], table.x, table.z).tolist() == odd
        assert qwc(table.x[i], table.z[i], table.x, table.z).tolist() == same


def test_known_qiskit_inputs_preserve_phase_without_simulation():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    from qiskit.quantum_info import SparsePauliOp, Statevector
    native = SparsePauliOp(["Y", "-iX"], [1, 2])
    operator = operator_input(native)
    assert operator.pauli_terms().labels() == (("Y", 1), ("X", -2j))
    native.coeffs[:] = 0
    assert operator.pauli_terms().labels() == (("Y", 1), ("X", -2j))
    state = state_input(Statevector([.6, .8j]))
    np.testing.assert_array_equal(state.physical_vector(), [.6, .8j])
    definition = QuantumCircuit(1, global_phase=.19)
    definition.ry(.37, 0)
    gate = Gate("shared", 1, [])
    gate._definition = definition
    circuit = QuantumCircuit(1, global_phase=.23)
    circuit.append(gate, [0], copy=False)
    circuit.append(gate, [0], copy=False)
    supplied = state_input(circuit)
    definition.clear()
    circuit.clear()
    first, second = (instruction.operation for instruction in supplied._native.data)
    assert first is second and first is not gate
    assert supplied._native.global_phase == .23 and first._definition.global_phase == .19
    assert first._definition.data[0].operation.params == [.37]
    assert supplied.preparation.physical_scale.as_float() == 1
    assert supplied.manifest.payload_bytes == supplied.preparation.payload_bytes > 0


def test_sparse_pauli_op_conversion_charges_its_own_table_before_masks(monkeypatch):
    from qiskit.quantum_info import SparsePauliOp
    # L = 16W+16 bytes per term. The converted table (L) stays alive through
    # the four L-byte copies of the mask conversion, so the exact law is 5RL
    # plus the dimension integer. Charging the SDK's 2q boolean bytes per term
    # instead of the converted table would admit the 2-qubit case below its
    # peak and reject the 20-qubit case at it. Each term also carries the
    # identity JSON share 6q + 360 of its Plan records.
    small = SparsePauliOp(["XY", "ZI"], [1.0, 0.5 - 0.25j])
    wide = SparsePauliOp(["X" + "I" * 19], [2.0])
    for native, labels in ((small, (("XY", 1.0), ("ZI", 0.5 - 0.25j))), (wide, (("X" + "I" * 19, 2.0),))):
        q = native.num_qubits
        law = len(native) * (5 * 32 + 6 * q + 360) + (q + 8) // 8
        admitted = operator_input(native, max_bytes=law)
        assert admitted.reference == ingest_pauli(labels, num_qubits=native.num_qubits).reference
        with monkeypatch.context() as guard:
            guard.setattr(np, "zeros", lambda *a, **k: pytest.fail("mask allocation before its byte check"))
            with pytest.raises(ValueError, match="SparsePauliOp conversion needs"):
                operator_input(native, max_bytes=law - 1)


def test_pauli_admission_charges_the_identity_json_of_its_records():
    import json
    from nwqlib.algorithms.gcim.fixed_basis import PauliTerm
    from nwqlib.core.records import Complex128

    def compact(value):
        # A nested record enters its parent's identity JSON without content_id.
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", exclude_computed_fields=True)
        return len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())

    # The share is three copies of J = 2q + 120 bytes, two identity JSON
    # copies and the records' own data. J holds the widest per-term layout,
    # a term record with a 24-character binary64 repr and its separator plus
    # the label and coefficient again in a readout spec and 4 bytes of list
    # brackets. It also holds an ADAPT pool term with an imaginary
    # coefficient.
    coefficient = -1.2345678901234567e-100
    assert len(repr(coefficient)) == 24
    for q in (1, 12, 13, 20, 64, 65, 200):
        label = ("XYZ" * q)[:q]
        widest = compact(PauliTerm(label=label, coefficient=coefficient))
        assert widest + 1 + (q + 3) + 25 + 4 == 2 * q + 120
        pool_term = [label, Complex128(real=0.0, imag=coefficient).model_dump(mode="json",
                                                                              exclude_computed_fields=True)]
        assert compact(pool_term) + 1 <= 2 * q + 120
    # Label and mask admission admit a table at the law with the share and
    # reject it one byte below. The law without the share rejects.
    for q, count in ((12, 3), (20, 2), (13, 4)):
        w = (q + 63) // 64
        rows = tuple(("I" * i + "X" + "I" * (q - i - 1), 1.0 + i) for i in range(count))
        share = count * 3 * (2 * q + 120)
        law = count * (q + 4 * (16 * w + 16)) + share
        assert ingest_pauli(rows, num_qubits=q, max_bytes=law).pauli_terms().labels() == rows
        for below in (law - 1, law - share):
            with pytest.raises(ValueError, match="raw Pauli input exceeds max_bytes"):
                ingest_pauli(rows, num_qubits=q, max_bytes=below)
        table = pauli_table(rows, num_qubits=q)
        masks = table.x, table.z, table.coefficients
        law = count * 4 * (16 * w + 16) + share + (q + 8) // 8
        assert ingest_pauli_masks(*masks, num_qubits=q, max_bytes=law).reference == \
            ingest_pauli(rows, num_qubits=q).reference
        for below in (law - 1, law - share):
            with pytest.raises(ValueError, match="Pauli mask conversion needs"):
                ingest_pauli_masks(*masks, num_qubits=q, max_bytes=below)


def test_direct_preparation_limit_precedes_selected_synthesis(monkeypatch):
    from types import SimpleNamespace
    from nwqlib.problems.inputs import prepare_qiskit
    from nwqlib.subroutines.state_preparation import direct
    calls = []
    def synthesize(direction, **kwargs):
        calls.append(direction)
        return SimpleNamespace(circuit="selected native construction")
    monkeypatch.setattr(direct, "_build_normalized_state_preparation", synthesize)
    state = state_input([3, 4j])
    with pytest.raises(ValueError, match="max_direct_amplitudes"):
        prepare_qiskit(state, max_direct_amplitudes=1)
    assert calls == []
    result = prepare_qiskit(state, max_direct_amplitudes=2)
    assert result.circuit == "selected native construction" and calls[0] is state._direction
