"""Natural input, physical phase, source association and compact-data witnesses."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy import sparse

from nwqlib.operators import (
    OperatorInput, PeriodicStencil, ingest_pauli, ingest_pauli_masks, operator_input, pauli_table,
)
from nwqlib.problems.inputs import StateInput, ingest_occupation, ingest_product, state_input


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


def test_an_exact_zero_has_one_identity_and_every_other_component_keeps_its_bits():
    """IEEE 754 -0.0 and +0.0 are one real number, so a sign of zero must not change what an input is
    (``_freeze_numbers``). The expected arrays are written out with +0.0, with a subnormal and a negative
    nonzero whose bits must survive, for the four inputs that store caller bytes."""
    from nwqlib.operators import ingest_dense, ingest_sparse
    from nwqlib.problems.inputs import ingest_vector

    tiny = 5e-324
    dense = np.array([[1.5, -0.0], [-0.0, -tiny]])
    written = np.array([[1.5, 0.0], [0.0, -tiny]])
    assert ingest_dense(dense).manifest.reference == ingest_dense(written).manifest.reference
    assert ingest_dense(dense).dense_array().tobytes() == written.tobytes()
    assert ingest_dense(np.asfortranarray(dense)).dense_array().tobytes() == written.tobytes()
    assert ingest_dense(dense).dense_array().dtype == np.dtype("float64")
    parts = np.array([[complex(-0.0, 2.0), complex(3.0, -0.0)], [complex(-0.0, -0.0), complex(-tiny, -tiny)]])
    clean = np.array([[complex(0.0, 2.0), complex(3.0, 0.0)], [complex(0.0, 0.0), complex(-tiny, -tiny)]])
    assert ingest_dense(parts).manifest.reference == ingest_dense(clean).manifest.reference
    assert ingest_dense(parts).dense_array().tobytes() == clean.tobytes()
    assert ingest_dense(parts).dense_array().dtype == np.dtype("complex128")

    structure = (np.array([0, 1]), np.array([0, 1, 2]))
    stored = sparse.csr_array((np.array([-0.0, -tiny]), *structure), shape=(2, 2))
    canonical = sparse.csr_array((np.array([0.0, -tiny]), *structure), shape=(2, 2))
    assert ingest_sparse(stored).manifest.reference == ingest_sparse(canonical).manifest.reference
    assert ingest_sparse(stored).sparse_entries()[0].tobytes() == canonical.data.tobytes()

    assert ingest_vector(np.array([1.0, -0.0])).manifest.reference == ingest_vector(np.array([1.0, 0.0])).manifest.reference
    assert ingest_vector(parts[0]).manifest.reference == ingest_vector(clean[0]).manifest.reference
    imaginary = ingest_vector(np.array([complex(3.0, -0.0), 1j]))
    assert imaginary.manifest.reference == ingest_vector(np.array([3 + 0j, 1j])).manifest.reference
    assert (ingest_product(np.array([[1.0, -0.0]])).manifest.reference
            == ingest_product(np.array([[1.0, 0.0]])).manifest.reference)

    # A -0.0 needs the writable copy beside the snapshot, and +0.0 only the snapshot, each against the limit.
    size = written.nbytes
    ingest_dense(written, max_bytes=size)
    with pytest.raises(ValueError, match="negative zero normalization"):
        ingest_dense(dense, max_bytes=size)
    ingest_dense(dense, max_bytes=2 * size)


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
    # The limit takes NumPy integers and None, and memory never changes the blocks or the count: the
    # reserve M(16W+16) + 32W + 64tW + 64t is 160 + 512 = 672 bytes for M = t = 4 and W = 1.
    assert table.group(strategy="qwc", max_comparisons=np.int64(6)) == qwc == table.group(strategy="qwc")
    assert table.group(strategy="qwc", max_bytes=672) == qwc
    with pytest.raises(ValueError, match="Pauli grouping"):
        table.group(strategy="qwc", max_bytes=671)
    operator = operator_input([[1, 2], [3, 4]])
    np.testing.assert_array_equal(operator.matvec(np.array([1., -1.]), max_products=4), [-1, -1])
    with pytest.raises(ValueError, match="max_products"):
        operator.matvec(None, max_products=3)


def test_pauli_matvec_peak_fits_its_byte_check():
    """The traced peak of a Pauli matvec stays within the bytes that its check counts.

    tracemalloc follows NumPy's allocations and Python's containers. On 12
    qubits, row j of 1000 has the letters of ``j*2654435761`` in base 4, so
    about 900 flip groups form and ``apply_terms`` keeps a member list for
    each. The float64 vector is converted to complex128 inside the action.
    Above the entry, with the vector and the operator already held, the peak
    measured 388,456 bytes on CPython 3.12, more than the 339,016 that
    counting the grouping as ``24M + 16g + 8`` bytes allows.
    """
    import tracemalloc
    from nwqlib.operators.inputs import _matvec_requirements

    q = 12
    operator = ingest_pauli(tuple(("".join("IXYZ"[(j * 2654435761 >> (2 * k)) & 3] for k in reversed(range(q))),
                                   (-1.) ** j / (j + 1)) for j in range(1, 1001)), num_qubits=q)
    vector = np.full(1 << q, 2.0 ** (-q / 2))
    tracemalloc.start()
    try:
        entry = tracemalloc.get_traced_memory()[0]
        operator.matvec(vector)
        peak = tracemalloc.get_traced_memory()[1] - entry
    finally:
        tracemalloc.stop()
    assert peak <= _matvec_requirements(operator)[1]


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


def test_observable_moment_acts_unscaled_and_scales_only_after_overflow(monkeypatch):
    """A dense observable with entries 4 acts without a scaled copy and gives the moment
    of the physical vector bit for bit. With entries 1.5e308 the unscaled action on the
    framed vector (norm 1/2) overflows, the scaled copy acts once, and the recovered
    moment 1.5e308 * (16 * 2**-6)**2 = 9.375e306 matches the direct NumPy value."""
    from nwqlib.operators import ingest_dense
    from nwqlib.operators import inputs as inputs_module
    from nwqlib.problems.scalars import physical_vector_statistics

    original = inputs_module._scaled_observable
    copies = []
    monkeypatch.setattr(inputs_module, "_scaled_observable",
                        lambda *args: copies.append(args[1]) or original(*args))
    rng = np.random.default_rng(4)
    matrix = rng.uniform(-4, 4, size=(8, 8)) + 1j * rng.uniform(-4, 4, size=(8, 8))
    matrix = matrix + matrix.conj().T
    solution = rng.normal(size=8) + 1j * rng.normal(size=8)
    _, _, statistics = physical_vector_statistics(solution, observable=ingest_dense(matrix))
    assert statistics[1].value == np.vdot(solution, matrix @ solution).real
    assert copies == []
    # Dividing the framed vector [0.5, 3 * 2**-1073] by 2**3 would round its
    # subnormal entry, so only the unscaled action avoids the operator copy.
    _, _, statistics = physical_vector_statistics(np.array([1.0, 3 * 2.0**-1072]),
                                                  observable=ingest_dense(4 * np.eye(2)))
    assert statistics[1].value == 4.0
    assert copies == []
    large, solution = np.full((16, 16), 1.5e308), np.full(16, 2.0**-6)
    _, _, statistics = physical_vector_statistics(solution, observable=ingest_dense(large))
    assert statistics[1].value == pytest.approx(np.vdot(solution, large @ solution).real, rel=1e-15)
    assert statistics[1].value == pytest.approx(9.375e306, rel=1e-15)
    assert copies == [1024]


@pytest.mark.parametrize("factor, copies", ((1.0, 0), (0.25, 0), (1e-200, 1)))
def test_observable_moment_copies_the_operator_only_beyond_exact_vector_scaling(monkeypatch, factor, copies):
    """O = factor * IZ in dense, CSR and Pauli form. For O = IZ the operator acts unscaled.
    For 0.25 IZ, e = -1, the exact scaling moves onto a copy of the vector. A legitimate
    1e-200 IZ, e = -664, lies beyond that range and forms the scaled operator copy. Each
    physical moment matches v^dagger O v computed directly."""
    from nwqlib.operators import ingest_dense, ingest_pauli, ingest_sparse
    from nwqlib.operators import inputs as inputs_module
    from nwqlib.problems.scalars import physical_vector_statistics

    original, made = inputs_module._scaled_observable, []
    monkeypatch.setattr(inputs_module, "_scaled_observable", lambda *args: made.append(1) or original(*args))
    solution = np.array([0.75 + 0.3j, -0.2 + 0.5j, 1.2 + 0.4j, 0.1j])
    matrix = np.diag(factor * np.array([1.0, -1.0, 1.0, -1.0]))
    want = np.vdot(solution, matrix @ solution).real
    for observable in (ingest_dense(matrix), ingest_sparse(sparse.csr_matrix(matrix)),
                       ingest_pauli([("IZ", factor)], num_qubits=2)):
        made.clear()
        _, _, statistics = physical_vector_statistics(solution, observable=observable)
        assert statistics[1].value == pytest.approx(want, rel=1e-15, abs=0)
        assert len(made) == copies


def test_integer_parameters_accept_numpy_integers_and_refuse_out_of_domain_values():
    """Widths, ladder operators, entry indices and control states take NumPy integers as
    the equal Python integers, and still refuse a value outside their domain."""
    from nwqlib.operators import ingest_band, ingest_dense, ingest_pauli, ingest_periodic_stencil
    from nwqlib.operators._fermion import fermion_table
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.subroutines.block_encoding.core import _control_state

    two = np.int64(2)
    assert ingest_pauli([("XZ", 0.5)], num_qubits=two).manifest == ingest_pauli([("XZ", 0.5)], num_qubits=2).manifest
    assert (ingest_band([0, 1], [1.0, 0.5], num_qubits=two).manifest
            == ingest_band([0, 1], [1.0, 0.5], num_qubits=2).manifest)
    assert (ingest_periodic_stencil(two, 1.0, 0.5).manifest
            == ingest_periodic_stencil(2, 1.0, 0.5).manifest)
    stencil = PeriodicStencil(two, 1.0, 0.5)
    assert stencil == PeriodicStencil(2, 1.0, 0.5) and type(stencil.num_qubits) is int
    dense = ingest_dense(np.arange(4.0).reshape(2, 2))
    assert dense.entry(np.int64(1), np.int32(0)) == 2.0
    numpy_terms = fermion_table([(1.0, ((np.int64(1), np.int64(1)), (np.int64(0), np.int64(0))))], num_modes=two)
    python_terms = fermion_table([(1.0, ((1, 1), (0, 0)))], num_modes=2)
    assert (numpy_terms.modes.tolist(), numpy_terms.actions.tolist()) == (python_terms.modes.tolist(),
                                                                           python_terms.actions.tolist())
    assert _control_state(np.int64(2), 2) == 2
    for call, message in ((lambda: PeriodicStencil(np.int64(0), 1.0, 0.5), "num_qubits"),
                          (lambda: dense.entry(2, 0), "entry indices"),
                          (lambda: fermion_table([(1.0, ((0, np.int64(2)),))], num_modes=2), "Fermion action"),
                          (lambda: _control_state(np.int64(4), 2), "control state")):
        with pytest.raises(ValueError, match=message):
            call()


def test_pauli_table_arrays_are_new_views_of_immutable_storage(tmp_path):
    """Changing the dtype of an array read from a Pauli table
    leaves the table, its operator's identity and its action unchanged, for a table built in memory
    and for one loaded from an archive, which keeps the archive's read-only mapping without a copy. The
    storage cannot be made writable, no attribute can be assigned, and pickling gives an equal table."""
    import pickle
    import warnings

    from nwqlib._choice_archive import ArchiveFiles

    fresh = ingest_pauli([("XY", 0.5), ("ZI", -1.5), ("II", 2.0)], num_qubits=2)
    files = ArchiveFiles(tmp_path, None)
    loaded = files.read_operator(files.write_operator("one", fresh))
    state = np.array([1.0, 2.0j, -0.5, 0.25])
    action = fresh.matvec(state)
    for operator, mapped in ((fresh, False), (loaded, True)):
        table, identity = operator.pauli_terms(), operator.manifest.content_id
        labels = table.labels()
        for name in ("x", "z", "coefficients"):
            exported = getattr(table, name)
            assert np.shares_memory(exported, getattr(table, name)) and exported is not getattr(table, name)
            owner = exported
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            assert (type(owner) is np.memmap) is mapped
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                exported.dtype = np.int8
            with pytest.raises(ValueError):
                owner.setflags(write=True)
            with pytest.raises(AttributeError):
                setattr(table, name, exported)
        assert table.labels() == labels and operator.manifest.content_id == identity
        assert np.array_equal(operator.matvec(state), action)
        restored = pickle.loads(pickle.dumps(table))
        assert restored.labels() == labels and restored.num_qubits == 2
        assert all(np.array_equal(getattr(restored, name), getattr(table, name))
                   for name in ("x", "z", "coefficients"))


def test_immutable_input_storage_is_shared_under_independent_headers():
    """A dense input backed by immutable bytes is stored without a copy, yet changing the dtype or shape of the
    caller's array, or of an array the operator returns, leaves the operator's entries unchanged."""
    import warnings

    from nwqlib.operators import ingest_dense

    for change in ("dtype", "shape"):
        supplied = np.frombuffer(np.array([[1.0, 2.0], [3.0, 4.0]]).tobytes(), dtype=np.float64).reshape(2, 2)
        operator = ingest_dense(supplied)
        identity = operator.manifest.content_id
        returned = operator.dense_array()
        assert np.shares_memory(returned, supplied) and returned is not supplied
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for array in (supplied, returned):
                if change == "dtype":
                    array.dtype = np.int64
                else:
                    array.shape = (4,)
        assert operator.entry(1, 0) == 3.0 and operator.manifest.content_id == identity
        assert (operator.dense_array().dtype, operator.dense_array().shape) == (np.float64, (2, 2))


def test_frozen_array_exports_are_views_with_their_own_headers():
    """np.asarray of a FrozenArray shares its elements without a copy but not its header, so changing the
    exported array's dtype leaves the record's value and hash unchanged."""
    import warnings

    from nwqlib.core.records import FrozenArray

    frozen = FrozenArray(np.array([-0.5, 1.5, 2.0]))
    before = hash(frozen)
    exported = np.asarray(frozen)
    assert np.shares_memory(exported, frozen.array) and exported is not np.asarray(frozen)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exported.dtype = np.int64
    assert frozen.tolist() == [-0.5, 1.5, 2.0] and frozen[0] == -0.5 and hash(frozen) == before


def test_frozen_array_storage_cannot_be_made_writable():
    """Public construction stores the elements in a bytes object, so setflags(write=True) on the base of a
    view is refused, here for a QHD Plan's support table, whose objective is -0.5. The same
    holds for int64, float64, an input with -0.0, a Fortran-order input and big-endian inputs, after
    pickling and after JSON. A C-order view of a canonical bytes object is shared without a copy, with
    its own offset and shape, and a big-endian infinity is refused."""
    import pickle

    import sympy as sp
    from nwqlib import Optimization, plan
    from nwqlib.algorithms.qhd import QHD
    from nwqlib.algorithms.qhd.method import objective_at
    from nwqlib.core.records import FrozenArray

    def unlock(frozen):
        owner = frozen.array
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        owner.setflags(write=True)

    x = sp.Symbol("x", real=True)
    r = plan(Optimization(objective=x, variables=(x,), bounds=((-1.0, 1.0),)),
             method=QHD(encoding="binary", boundary="periodic", num_grid_points=4, num_steps=1)).reconstruction
    table = r.support_values[0].values
    start = objective_at(r, (1,), 4)
    with pytest.raises(ValueError):
        unlock(table)
    assert start == -0.5
    inputs = (np.arange(6, dtype=np.int64).reshape(2, 3), np.linspace(-1, 1, 5), np.array([1.0, -0.0, 2.0]),
              np.asfortranarray(np.arange(6.0).reshape(2, 3)), np.arange(4.0).astype(">f8"),
              np.arange(4).astype(">i8"))
    for values in inputs:
        first = FrozenArray(values)
        for frozen in (first, pickle.loads(pickle.dumps(first)), FrozenArray.from_json(first.to_json())):
            with pytest.raises(ValueError):
                unlock(frozen)
            assert frozen == first and hash(frozen) == hash(first) and np.array_equal(frozen.array, values)
            assert frozen.array.dtype.str in ("<f8", "<i8") and frozen.array.flags.c_contiguous
            if frozen.array.dtype.kind == "f":
                assert not np.signbit(frozen.array[frozen.array == 0]).any()
    window = np.frombuffer(np.arange(8.0).tobytes(), dtype="<f8")[2:6]
    shared = FrozenArray(window)
    assert np.shares_memory(shared.array, window) and shared.tolist() == [2.0, 3.0, 4.0, 5.0]
    assert shared == FrozenArray(np.arange(2.0, 6.0)) and hash(shared) == hash(FrozenArray(np.arange(2.0, 6.0)))
    with pytest.raises(ValueError, match="finite"):
        FrozenArray(np.array([1.0, np.inf]).astype(">f8"))


@pytest.mark.parametrize("convert", [
    lambda limits: operator_input(np.array([[2.0, 1j], [-1j, -3.0]]), **limits),
    lambda limits: state_input(np.array([1.0, 1j]), **limits),
])
def test_input_conversions_default_to_the_physical_memory(convert, monkeypatch):
    """Without max_bytes, a conversion is limited by the physical memory, read when it runs.

    The least admitted memory T is found by raising the mocked memory to each
    requirement that a refusal reports. The default then refuses at T - 1 and
    accepts at T, as an explicit max_bytes of T - 1 and of T, a NumPy integer, does.
    """
    import re
    from nwqlib import _limits

    memory = 1
    while True:
        monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: memory)
        try:
            convert({})
            break
        except ValueError as error:
            need = int(re.search(r"needs about (\d+) bytes, exceeding the physical memory", str(error))[1])
            assert need > memory
            memory = need
    monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: memory - 1)
    with pytest.raises(ValueError, match="physical memory"):
        convert({})
    with pytest.raises(ValueError, match=f"max_bytes={memory - 1}"):
        convert(dict(max_bytes=memory - 1))
    convert(dict(max_bytes=np.int64(memory)))


def test_a_raw_pauli_refusal_names_the_byte_limit_it_used(monkeypatch):
    """A raw Pauli table is refused by its term count, and the refusal names the physical memory or max_bytes."""
    from nwqlib import _limits

    monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: 1)
    with pytest.raises(ValueError, match=r"raw Pauli input exceeds the physical memory \(1 bytes\)"):
        ingest_pauli([("XI", 1.0), ("ZZ", -0.5)], num_qubits=2)
    with pytest.raises(ValueError, match="raw Pauli input exceeds max_bytes=1"):
        ingest_pauli([("XI", 1.0), ("ZZ", -0.5)], num_qubits=2, max_bytes=1)
