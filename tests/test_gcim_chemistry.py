"""Chemistry-input anchors for ADAPT-GCiM."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from _h2_sto3g_reference import (
    H2_STO3G_JW_TERMS,
    H2_STO3G_M7_ROUNDING,
    H2_STO3G_M7_TERMS,
)
from nwqlib.algorithms.gcim import (
    build_gcim_chemistry_problem,
    chemistry_reference_diagnostic,
    qubit_operator_to_sparse_pauli,
)
from nwqlib._prepared_execution import Run
from nwqlib.core.planning import RandomStreams
from _fermionic_references import (
    jw_double_excitation_generator,
    jw_single_excitation_generator,
)


def _require_chemistry_extras() -> None:
    pytest.importorskip("openfermion")
    pytest.importorskip("pyscf")


def _sparse_pauli_dict(operator) -> dict[str, float]:
    return {
        str(pauli): float(np.real_if_close(coefficient))
        for pauli, coefficient in zip(operator.paulis, operator.coeffs, strict=True)
    }


def test_h2_pyscf_openfermion_pipeline_matches_stored_table_and_published_m7() -> None:
    _require_chemistry_extras()

    data = build_gcim_chemistry_problem(
        "H 0 0 0; H 0 0 0.7414",
        basis="sto-3g",
        unit="Angstrom",
        name="H2/STO-3G chemistry input anchor",
        reference_methods=("mp2", "ccsd"),
    )

    assert data.num_qubits == 4
    assert data.metadata["active_space"] is None
    assert data.reference_occupations == (1, 1, 0, 0)
    np.testing.assert_array_equal(data.adapt_method().initial_state._physical, (1, 1, 0, 0))
    panel = data.reference_panel
    assert panel["label"] == "application context, not a validation gate"
    # PySCF energies are iterative-solver convergence anchors; platform BLAS
    # variance makes 1e-8 the intended tolerance class.
    assert panel["e_hf"] == pytest.approx(-1.116684387085341, rel=0.0, abs=1.0e-8)
    assert panel["e_mp2"] == pytest.approx(-1.1298551535553099, rel=0.0, abs=1.0e-8)
    assert panel["e_ccsd"] == pytest.approx(-1.1372703406409217, rel=0.0, abs=1.0e-8)
    assert data.metadata["chemistry_reference_panel"] == panel

    terms = _sparse_pauli_dict(data.hamiltonian)
    electronic_terms = dict(terms)
    electronic_terms["IIII"] -= data.nuclear_repulsion_energy
    assert set(electronic_terms) == {label for label, _ in H2_STO3G_JW_TERMS}
    for label, expected in H2_STO3G_JW_TERMS:
        assert electronic_terms[label] == pytest.approx(expected, rel=0.0, abs=1.0e-8)
    # Independent published anchor at its printed precision; M7 includes the
    # nuclear repulsion in the identity term, like the total Hamiltonian here.
    for label, printed in H2_STO3G_M7_TERMS:
        assert terms[label] == pytest.approx(printed, rel=0.0, abs=H2_STO3G_M7_ROUNDING + 1.0e-9)

    exact_total_ground = float(np.linalg.eigvalsh(data.hamiltonian.to_matrix())[0])
    problem = data.eigenproblem()
    method = data.adapt_method()
    plan = method.plan(
        problem,
        output=problem.default_output(),
        execution="classical",
        shots=None,
        rng=RandomStreams(7),
    )
    result = Run(plan).wait(timeout=30, poll_interval=0)
    assert all(row.residual_norm is None for row in result.history)
    assert result.eigenvalue == pytest.approx(exact_total_ground, rel=0.0, abs=1.0e-6)
    chemistry_refs = chemistry_reference_diagnostic(data.metadata, energy=result.eigenvalue)
    assert chemistry_refs["reference_panel"] == panel
    assert chemistry_refs["correlation_fraction"] == (
        (result.eigenvalue - panel["e_hf"]) / (panel["e_ccsd"] - panel["e_hf"])
    )
    assert chemistry_refs["correlation_fraction"] == pytest.approx(1.0, rel=0.0, abs=1e-4)
    assert plan.reconstruction.pool[result.selected[0]].family == "double_singlet"

    # Actual chemistry reference production remains occupation/order preserving.
    # Little-endian basis index: each occupied mode j contributes 2**j.
    reference = np.zeros(2**data.num_qubits, dtype=complex)
    reference[sum(2**mode for mode, bit in enumerate(data.reference_occupations) if bit)] = 1.0
    expectation = float(
        np.vdot(reference, data.hamiltonian.to_matrix(sparse=True) @ reference).real
    )
    assert expectation == pytest.approx(data.rhf_energy, rel=0.0, abs=1.0e-8)
    np.testing.assert_allclose(
        Statevector.from_instruction(data.reference_preparation).data, reference, rtol=0.0, atol=0.0
    )


def test_h4_closed_shell_reference_expectation_equals_rhf_energy() -> None:
    _require_chemistry_extras()

    data = build_gcim_chemistry_problem(
        "H 0 0 0; H 0 0 0.75; H 0 0 1.5; H 0 0 2.25",
        basis="sto-3g",
        unit="Angstrom",
        name="H4/STO-3G closed-shell reference anchor",
    )

    assert data.num_qubits == 8
    # PySCF energies are iterative-solver convergence anchors; platform BLAS
    # variance makes 1e-8 the intended tolerance class.
    assert data.reference_panel["e_hf"] == pytest.approx(
        -2.103290822999874, rel=0.0, abs=1.0e-8
    )
    # For a computational basis state |n>, <n|Z_j|n> = (-1)**n_j and every
    # string containing X or Y has zero expectation, so the sum below over the
    # I/Z strings is <n|H|n>. H uses the converged RHF orbitals, in which the
    # closed-shell determinant's energy is exactly E_HF, so SCF convergence
    # does not enter. Each discarded term with |c| <= coefficient_cutoff = 1e-12
    # moves the sum by at most 1e-12. On eight modes at most 120 number-diagonal
    # fermionic terms (8 one-body, 112 two-body) and 256 I/Z strings can
    # contribute, below 4e-10 in total. Floating-point roundoff in the
    # coefficients, this sum and E_HF stays below 1e-12 at this size.
    paulis = data.hamiltonian.paulis
    diagonal = ~paulis.x.any(axis=1)
    signs = (-1.0) ** (paulis.z[diagonal] @ np.array(data.reference_occupations))
    expectation = float(np.sum(signs * data.hamiltonian.coeffs[diagonal].real))
    assert expectation == pytest.approx(data.rhf_energy, rel=0.0, abs=1.0e-9)


def test_rhf_orbital_signs_do_not_change_the_hamiltonian(monkeypatch) -> None:
    _require_chemistry_extras()
    import pyscf.scf

    # Stretched linear H4: its MOs alternate inversion parity, so flipping one
    # MO changes the sign of integrals with an odd count of that index. In H2
    # every nonzero integral has an even count of each index.
    geometry = "; ".join(f"H 0 0 {2.0 * index}" for index in range(4))
    reference = build_gcim_chemistry_problem(geometry, basis="sto-3g")
    original = pyscf.scf.RHF

    def rhf_with_flipped_orbital(mol):
        # An MO column's sign is arbitrary, and another LAPACK build may return
        # the opposite one. Energies and densities are unchanged.
        mean_field = original(mol)
        kernel = mean_field.kernel

        def kernel_then_flip(*args, **kwargs):
            energy = kernel(*args, **kwargs)
            mean_field.mo_coeff = mean_field.mo_coeff * np.array([1.0, -1.0, 1.0, 1.0])
            return energy

        mean_field.kernel = kernel_then_flip
        return mean_field

    monkeypatch.setattr(pyscf.scf, "RHF", rhf_with_flipped_orbital)
    flipped = build_gcim_chemistry_problem(geometry, basis="sto-3g")

    assert flipped.rhf_energy == pytest.approx(reference.rhf_energy, rel=0.0, abs=1.0e-12)
    # The two builds differ only in floating-point summation order, far below
    # 1e-8 Hartree. An orbital-sign-dependent Hamiltonian changes coefficients
    # of this molecule by O(0.1) Hartree.
    left, right = (_sparse_pauli_dict(data.hamiltonian) for data in (reference, flipped))
    assert max(abs(left.get(label, 0.0) - right.get(label, 0.0)) for label in left | right) < 1.0e-8


def test_lih_cas22_active_space_matches_pyscf_casci() -> None:
    _require_chemistry_extras()

    data = build_gcim_chemistry_problem(
        "Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        unit="Angstrom",
        name="LiH/STO-3G CAS(2,2) anchor",
        active_space=(2, 2),
        reference_methods=("casci",),
    )

    assert data.n_spatial_orbitals == 2
    assert data.num_qubits == 4
    assert data.num_electrons == 2
    assert data.reference_occupations == (1, 1, 0, 0)
    assert data.metadata["active_space"] == {
        "n_active_electrons": 2,
        "n_active_orbitals": 2,
        "system_spatial_orbitals": 6,
        "core_energy": pytest.approx(-6.804012298302049, rel=0.0, abs=1.0e-8),
    }
    panel = data.reference_panel
    assert panel["mp2"]["status"] == "not_requested"
    assert panel["ccsd"]["status"] == "not_requested"
    assert panel["casci"]["status"] == "computed"
    assert panel["e_casci"] == pytest.approx(-7.8621288334385895, rel=0.0, abs=1.0e-8)

    lowest = float(np.linalg.eigvalsh(data.hamiltonian.to_matrix())[0])
    assert lowest == pytest.approx(panel["e_casci"], rel=0.0, abs=1.0e-8)
    diagnostic = chemistry_reference_diagnostic(data.metadata, energy=lowest)
    assert diagnostic is not None
    assert diagnostic["active_space_casci_correlation_fraction"] == pytest.approx(
        1.0,
        rel=0.0,
        abs=1.0e-8,
    )
    no_reference = build_gcim_chemistry_problem(
        "Li 0 0 0; H 0 0 1.6",
        active_space=(2, 2),
    )
    assert no_reference.reference_panel["casci"]["status"] == "not_requested"
    # Same RHF frame, with the iterative chemistry tolerance for separate RHF runs.
    np.testing.assert_allclose(
        data.hamiltonian.to_matrix(),
        no_reference.hamiltonian.to_matrix(),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_spin_nonzero_input_is_rejected() -> None:
    _require_chemistry_extras()

    with pytest.raises(ValueError, match="spin=0"):
        build_gcim_chemistry_problem(
            "H 0 0 0; H 0 0 0.7414",
            basis="sto-3g",
            unit="Angstrom",
            spin=2,
        )


def test_openfermion_jw_matches_hand_rolled_e4_e5_generators() -> None:
    _require_chemistry_extras()
    from openfermion.ops import FermionOperator
    from openfermion.transforms import jordan_wigner

    single_fermion = FermionOperator(((2, 1), (0, 0)), 1.0)
    single_fermion += FermionOperator(((0, 1), (2, 0)), -1.0)
    single_openfermion = qubit_operator_to_sparse_pauli(
        jordan_wigner(single_fermion),
        num_qubits=4,
    )
    single_native = jw_single_excitation_generator(2, 0, num_qubits=4)
    np.testing.assert_allclose(
        single_openfermion.to_matrix(),
        single_native.to_matrix(),
        rtol=0.0,
        atol=1.0e-12,
    )

    double_fermion = FermionOperator(((2, 1), (3, 1), (0, 0), (1, 0)), 1.0)
    double_fermion += FermionOperator(((1, 1), (0, 1), (3, 0), (2, 0)), -1.0)
    double_openfermion = qubit_operator_to_sparse_pauli(
        jordan_wigner(double_fermion),
        num_qubits=4,
    )
    double_native = jw_double_excitation_generator(2, 3, 0, 1, num_qubits=4)
    np.testing.assert_allclose(
        double_openfermion.to_matrix(),
        double_native.to_matrix(),
        rtol=0.0,
        atol=1.0e-12,
    )
