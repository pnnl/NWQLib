"""Exact pool and compiler relations for the current primary ADAPT path, on small fixtures."""

from dataclasses import replace
import numpy as np
import pytest
import scipy.linalg
from qiskit import transpile
from qiskit.quantum_info import Operator, SparsePauliOp
import nwqlib.subroutines.fermionic_pool as fermionic_pool
import nwqlib.subroutines.fermionic_circuits as matrix_elements
from nwqlib.algorithms.gcim import sector_expectations
from nwqlib.subroutines.fermionic_pool import (
    FermionicGenerator,
    apply_generator,
    apply_generator_exponential,
    apply_spin_squared,
)
from _fermionic_references import (
    generator_matrix,
    generator_pauli,
    spin_squared_operator,
    spin_z_operator,
    total_particle_number_operator,
)


def _generator(index: int, *, label: str = "XI") -> FermionicGenerator:
    return FermionicGenerator(
        family="custom",
        spatial_indices=(),
        pool_index=index,
        num_qubits=2,
        fermion_terms=(),
        pauli_terms=tuple((SparsePauliOp.from_list([(label, 1.0j)])).to_list()),
    )


def test_sector_helper_matches_sparse_reference() -> None:
    rng = np.random.default_rng(818)
    state = rng.normal(size=16) + 1.0j * rng.normal(size=16)
    state = state / np.linalg.norm(state)
    expected = {
        "particle_number": float(
            np.real(np.vdot(state, total_particle_number_operator(4) @ state))
        ),
        "spin_z": float(np.real(np.vdot(state, spin_z_operator(4) @ state))),
        "spin_squared": float(np.real(np.vdot(state, spin_squared_operator(4) @ state))),
    }

    observed = sector_expectations(state, num_qubits=4)
    for key, value in expected.items():
        assert observed[key] == pytest.approx(value, abs=1.0e-12)


def test_matrix_free_generator_action_and_exponential_match_dense_reference() -> None:
    generator = FermionicGenerator(
        family="single",
        spatial_indices=(0, 1),
        pool_index=0,
        num_qubits=2,
        fermion_terms=(),
        pauli_terms=tuple(
            (SparsePauliOp.from_list([("XI", 0.2j), ("YI", -0.3j), ("ZI", 0.4j)])).to_list()
        ),
    )
    rng = np.random.default_rng(91)
    state = rng.normal(size=4) + 1.0j * rng.normal(size=4)
    dense = generator_matrix(generator)

    # Unnormalized vectors need a relative roundoff allowance and an absolute zero floor.
    np.testing.assert_allclose(
        apply_generator(generator, state),
        dense @ state,
        rtol=1.0e-13,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        apply_generator_exponential(generator, state, 0.31),
        scipy.linalg.expm(0.31 * dense) @ state,
        rtol=1.0e-13,
        atol=1e-12,
    )
    # The classical ADAPT Plan acts with the packed rows of the same generator.
    from nwqlib.algorithms.gcim.adapt_actions import pack_action_rows, packed_exponential

    np.testing.assert_allclose(
        packed_exponential(pack_action_rows(generator.pauli_terms, 2), state, 0.31),
        scipy.linalg.expm(0.31 * dense) @ state,
        rtol=1.0e-13,
        atol=1e-12,
    )


@pytest.mark.parametrize(("theta", "steps"), [(0.0, 0), (0.2, 1), (10 / 9, 2), (1.3, 3)])
def test_exponential_work_law_counts_the_taylor_pauli_actions(monkeypatch, theta, steps) -> None:
    from types import SimpleNamespace
    from nwqlib.algorithms.gcim import adapt_actions
    from nwqlib.algorithms.gcim.adapt_records import _exponential_action_work
    from nwqlib.operators._pauli import pauli_action_requirements

    terms = tuple(
        (SparsePauliOp.from_list([("XY", 0.4j), ("ZI", -0.3j), ("YY", 0.2j)])).to_list()
    )
    member = SimpleNamespace(terms=terms)
    actions = []
    action = adapt_actions.apply_terms

    def counted(table, state, **kwargs):
        actions.append(len(table))
        return action(table, state, **kwargs)

    monkeypatch.setattr(adapt_actions, "apply_terms", counted)
    d = 4
    adapt_actions.packed_exponential(adapt_actions.pack_action_rows(terms, 2), np.ones(d, dtype=complex), theta)
    degree = fermionic_pool.GENERATOR_TAYLOR_DEGREE
    # Every Horner term applies the whole three-row table once. The Pauli
    # 1-norm is 0.9, so a step count is max(1, ceil(1.8 theta)), and none at
    # theta = 0. At theta = 10/9, 1.8 theta rounds to exactly 2, where a
    # count formed in another order could round up to 3.
    assert actions == [3] * (degree * steps)
    # The law (adapt_records._exponential_action_work) charges each executed
    # action the shared single-column allowance of pauli_action_requirements,
    # a scaling and an addition per Horner term, one copy per step and ten
    # normalization passes, which also cover the copy that replaces the
    # steps at theta = 0.
    shared = pauli_action_requirements(d, 3, min(3, d), complex_input=True)[1]
    assert _exponential_action_work(member, theta, d) == (
        len(actions) * (shared + 2 * d) + steps * d + 10 * d
    )


def test_classical_adapt_acts_with_packed_tables_and_admits_each_whole_query(monkeypatch) -> None:
    """Classical ADAPT's Pauli actions use the Plan's packed tables, and a query is priced whole.

    The small one-qubit classical fixture performs 97 generator, H0 and
    insertion actions, every one on a table packed once per Plan
    (adapt_actions.build_action_tables) and none from labels. Its two-factor
    energy-and-gradient query at angles (0.2, -0.7) needs 5,939 work units
    under the shared action law (adapt_actions.query_work with
    adapt_records._energy_gradient_work): it is admitted at max_products =
    5,939 and refused at 5,938. Label-form actions or the former per-term
    action charge fail.
    """
    from collections import Counter
    import nwqlib
    from nwqlib.algorithms.gcim import adapt_actions, adapt_acquisition
    from nwqlib.operators._pauli import PauliTerms
    from test_adapt_primary import plan_for

    counts = Counter()
    for module in (adapt_actions, adapt_acquisition, fermionic_pool):
        original = module.apply_terms

        def counted(table, *args, _original=original, **kwargs):
            counts["packed" if isinstance(table, PauliTerms) else "labels"] += 1
            return _original(table, *args, **kwargs)

        monkeypatch.setattr(module, "apply_terms", counted)
    plan = plan_for(execution="classical")
    # One H0 row and three generator rows, each also packed as an
    # insertion row: R = 1 + 2*3 = 7 rows of 32 bytes, packed at
    # R(q + 4) = 35 work units (adapt_actions.build_action_tables).
    tables = plan._native["inputs"].cache["action_tables"]
    assert (tables["resident_bytes"], tables["packing_work"]) == (224, 35)
    nwqlib.solve(plan, progress=False)
    assert counts == Counter(packed=97)

    point = adapt_acquisition._point(plan, "energy", right=((0, 0.2), (1, -0.7)))
    experiment, program = plan._resolve_selection(point.experiment, point.bindings)
    admitted = plan.method.revise(max_products=5939)
    (kernel,) = admitted.selected_kernels(plan, experiment, program)
    assert kernel.invocation_work == 5939
    with pytest.raises(ValueError, match="ADAPT classical query needs 5939 scalar products, exceeding "
                                         "max_products=5938. Later queries"):
        plan.method.revise(max_products=5938).selected_kernels(plan, experiment, program)


def test_sector_work_law_counts_two_linear_spin_actions(monkeypatch) -> None:
    import nwqlib.operators.inputs as inputs
    from nwqlib.algorithms.gcim.sector import _sector_work

    passes = []
    action = inputs.apply_terms

    def counted(rows, state, **kwargs):
        passes.append(max(1, len(rows)))
        return action(rows, state, **kwargs)

    monkeypatch.setattr(inputs, "apply_terms", counted)
    q = 8
    rng = np.random.default_rng(3)
    sector_expectations(rng.normal(size=1 << q) + 1.0j * rng.normal(size=1 << q), num_qubits=q)
    # S_- S_+ is applied as two sequential sums of 2q Pauli terms each, so
    # its cost is linear in q, and each is charged by the shared action law.
    from nwqlib.operators._pauli import pauli_action_requirements

    n = 1 << q
    assert len(passes) == 2 and sum(passes) == 4 * q
    actions = sum(pauli_action_requirements(n, rows, min(rows, n))[1] for rows in passes)
    assert _sector_work(q) == n * (9 * (q // 2) + 15) + q * n // 2 + 4 * q + 2 + actions


def test_matrix_free_spin_squared_action_matches_validation_matrix() -> None:
    rng = np.random.default_rng(72)
    state = rng.normal(size=16) + 1.0j * rng.normal(size=16)
    # The action scales with this unnormalized input, including possible zero components.
    np.testing.assert_allclose(
        apply_spin_squared(state, num_qubits=4),
        spin_squared_operator(4) @ state,
        rtol=1.0e-13,
        atol=1.0e-12,
    )


def test_structured_generator_preserves_identity_global_phase() -> None:
    identity = FermionicGenerator(
        family="custom",
        spatial_indices=(),
        pool_index=0,
        num_qubits=2,
        fermion_terms=(),
        pauli_terms=tuple((SparsePauliOp.from_list([("II", 0.5j)])).to_list()),
    )
    theta = 0.37
    observed = Operator(matrix_elements.build_generator_circuit(identity, theta)).data
    np.testing.assert_allclose(observed, np.exp(0.5j * theta) * np.eye(4), rtol=0.0, atol=1.0e-12)


def test_exact_compiler_constructs_large_compact_inputs_with_bounded_local_work(
    monkeypatch,
) -> None:
    import nwqlib.subroutines.fermionic_circuits as compiler

    def no_full_matrix(*args, **kwargs):
        raise AssertionError("compact generator construction must not materialize a full operator")

    monkeypatch.setattr(SparsePauliOp, "to_matrix", no_full_matrix)
    observed_shapes = []
    original_expm = compiler.la.expm

    def local_expm(matrix):
        observed_shapes.append(matrix.shape)
        assert max(matrix.shape) <= 5
        return original_expm(matrix)

    monkeypatch.setattr(compiler.la, "expm", local_expm)
    commuting = fermionic_pool._generator(
        family="single",
        spatial_indices=(0, 39),
        pool_index=0,
        num_qubits=80,
        base_terms=fermionic_pool._single_base_terms(0, 39),
    )
    assert compiler.build_generator_circuit(commuting, 0.23, controlled=True).num_qubits == 81
    assert not observed_shapes
    for family, indices in (
        ("double_singlet", (0, 0, 17, 39)),
        ("double_singlet", (0, 17, 29, 39)),
        ("double_triplet", (0, 17, 29, 39)),
        ("double_triplet", (0, 17, 17, 39)),
    ):
        base = (
            fermionic_pool._double_singlet_base_terms
            if family == "double_singlet"
            else fermionic_pool._double_triplet_base_terms
        )
        generator = fermionic_pool._generator(
            family=family,
            spatial_indices=indices,
            pool_index=0,
            num_qubits=80,
            base_terms=base(*indices),
        )
        start = len(observed_shapes)
        circuit = compiler.build_generator_circuit(generator, 0.23, controlled=True)
        assert circuit.num_qubits == 81
        shared = bool(set(indices[:2]).intersection(indices[2:]))
        if shared:
            assert len(observed_shapes) > start
            active = sorted({mode for _, ops in generator.fermion_terms for mode, _ in ops})
            # Each active mode crosses exactly mode - destination spectators in each direction. Routing is unconditional in both control branches.
            expected_fswaps = 2 * sum(mode - destination for destination, mode in enumerate(active))
            assert circuit.count_ops()["swap"] == expected_fswaps
        else:
            assert len(observed_shapes) == start


def test_exact_compiler_rejects_nondefault_noncommuting_terms_before_local_work(
    monkeypatch,
) -> None:
    import nwqlib.subroutines.fermionic_circuits as compiler

    def no_exponential(*args, **kwargs):
        raise AssertionError("unsupported records must fail before local exponentiation")

    monkeypatch.setattr(compiler.la, "expm", no_exponential)

    generator = fermionic_pool.enumerate_spin_adapted_gsd_pool(3)[8]
    duplicate = replace(
        generator, fermion_terms=generator.fermion_terms + generator.fermion_terms[:1]
    )
    changed = replace(
        generator,
        pauli_terms=tuple(
            (generator_pauli(generator) + SparsePauliOp.from_list([("XIIIII", 0.01j)])).to_list()
        ),
    )
    custom = replace(
        _generator(0),
        pauli_terms=tuple((SparsePauliOp.from_list([("XI", 1.0j), ("ZI", 1.0j)])).to_list()),
    )
    for invalid in (duplicate, changed, custom):
        with pytest.raises(
            ValueError,
            match="custom generator|Pauli representation must agree",
        ):
            compiler.build_generator_circuit(invalid, 0.23)


def test_exact_compiler_controlled_identity_preserves_relative_phase() -> None:
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    generator = replace(
        _generator(0), pauli_terms=tuple((SparsePauliOp.from_list([("II", 0.5j)])).to_list())
    )
    expected = np.eye(8, dtype=complex)
    expected[1::2, 1::2] *= np.exp(0.5j * 0.37)
    observed = build_generator_circuit(generator, 0.37, controlled=True)
    np.testing.assert_allclose(Operator(observed).data, expected, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("label", ("XI", "YI", "ZI", "XX", "YY", "ZZ", "XYZ", "III"))
@pytest.mark.parametrize("theta", (1.0e-11, 0.37))
def test_exact_compiler_controls_only_the_pauli_rotation(label, theta) -> None:
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    generator = replace(_generator(0, label=label), num_qubits=len(label))
    pauli = generator_pauli(generator).paulis[0].to_matrix()
    dimension = len(pauli)
    exact = np.cos(theta) * np.eye(dimension) + 1j * np.sin(theta) * pauli
    expected = np.eye(2 * dimension, dtype=complex)
    expected[1::2, 1::2] = exact
    circuit = build_generator_circuit(generator, theta, controlled=True)
    observed = Operator(circuit).data
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-12)
    if theta < 1.0e-10:
        # The imaginary first-order action detects an omitted phase/rotation even below the whole-operator tolerance.
        np.testing.assert_allclose(
            observed.imag / theta, expected.imag / theta, rtol=1.0e-5, atol=1.0e-8
        )
    lowered = transpile(circuit, basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=7)
    weight = sum(c != "I" for c in label)
    assert lowered.count_ops().get("cx", 0) <= 2 * weight
