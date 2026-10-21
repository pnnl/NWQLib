"""Exact pool and compiler relations for the current primary ADAPT path, on small fixtures."""

from dataclasses import replace
import numpy as np
import pytest
import scipy.linalg
from qiskit import transpile
from qiskit.quantum_info import Operator, SparsePauliOp
import nwqlib.subroutines.fermionic_pool as fermionic_pool
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


@pytest.mark.parametrize("label", ("YI", "XYZ"))
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
