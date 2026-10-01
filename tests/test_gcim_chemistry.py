"""Chemistry-input anchors for ADAPT-GCiM."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from _h2_sto3g_reference import (
    H2_STO3G_JW_TERMS,
    H2_STO3G_M7_ROUNDING,
    H2_STO3G_M7_TERMS,
    H2_STO3G_NUCLEAR_REPULSION,
)
from nwqlib.algorithms.gcim import (
    CHEMISTRY_EXTRA_MESSAGE,
    build_gcim_chemistry_problem,
    chemistry_reference_diagnostic,
    correlation_fraction,
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


@pytest.fixture
def chemistry_events(monkeypatch):
    """Record chemistry calls with synthetic solvers, including an orbital rotation that exposes
    wrong basis reuse.
    """
    from nwqlib.algorithms.gcim import chemistry

    events = []

    class MeanField:
        converged = True
        e_tot = -0.5

        def __init__(self):
            self.mo_coeff = np.eye(2)

        def kernel(self):
            events.append("rhf")
            return self.e_tot

        def get_hcore(self):
            return np.diag([-1.0, 1.0])

    class Reference:
        converged = True
        e_tot = -0.6

        def __init__(self, name):
            self.name = name

        def kernel(self):
            events.append(self.name)
            return (-0.1,)

    class CASCI(Reference):
        def __init__(self, mean_field, *_):
            super().__init__("casci")
            self.mo_coeff = mean_field.mo_coeff

        def kernel(self):
            events.append("casci")
            self.mo_coeff = np.array([[0.0, -1.0], [1.0, 0.0]])
            return (self.e_tot,)

        def get_h1eff(self, mo_coeff=None):
            events.append("h1eff")
            orbitals = self.mo_coeff if mo_coeff is None else mo_coeff
            return orbitals.T @ np.diag([-1.0, 1.0]) @ orbitals, 0.7

        def get_h2eff(self, mo_coeff=None):
            events.append("h2eff")
            return np.zeros((2, 2, 2, 2))

    mol = SimpleNamespace(nelectron=2, energy_nuc=lambda: 0.7, atom_symbol=lambda _: "H")
    extras = SimpleNamespace(
        gto=SimpleNamespace(M=lambda **_: mol),
        scf=SimpleNamespace(RHF=lambda _: MeanField()),
        mp=SimpleNamespace(MP2=lambda _: Reference("mp2")),
        cc=SimpleNamespace(CCSD=lambda _: Reference("ccsd")),
        mcscf=SimpleNamespace(CASCI=CASCI),
        ao2mo=SimpleNamespace(
            kernel=lambda *_: np.zeros((2, 2, 2, 2)),
            restore=lambda _, integrals, __: np.asarray(integrals),
        ),
        FermionOperator=object,
        jordan_wigner=lambda _: SimpleNamespace(terms={(): 1.0}),
    )
    monkeypatch.setattr(chemistry, "_require_chemistry_extras", lambda: extras)
    monkeypatch.setattr(
        chemistry,
        "_assemble_spin_orbital_fermion_operator",
        lambda *_, **__: SimpleNamespace(terms={(): 1.0}),
    )
    return events


@pytest.mark.parametrize("active_space", [None, (2, 2), (np.int64(2), np.int64(2))])
def test_default_input_and_reports_do_not_solve_references_or_allocate_amplitudes(
    chemistry_events,
    monkeypatch,
    active_space,
) -> None:
    from nwqlib.algorithms.gcim import chemistry

    def forbidden(*_, **__):
        pytest.fail("chemistry input/report construction allocated reference amplitudes")

    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    data = build_gcim_chemistry_problem("fake geometry", active_space=active_space)
    events = ["rhf"] if active_space is None else ["rhf", "h1eff", "h2eff"]
    assert chemistry_events == events
    assert data.reference_occupations == (1, 1, 0, 0)
    assert data.adapt_method().initial_state.preparation.implementation == "qiskit.occupation"
    circuit = data.reference_preparation
    assert circuit.count_ops() == {"x": 2}
    assert [circuit.find_bit(item.qubits[0]).index for item in circuit.data] == [0, 1]
    diagnostic = chemistry_reference_diagnostic(data.metadata, energy=-0.5)
    section = chemistry.chemistry_reference_report_section(diagnostic)
    # The display consumes the stored panel and starts no chemistry solve.
    assert chemistry_events == events
    assert "correlation_fraction" not in diagnostic
    assert "active_space_casci_correlation_fraction" not in diagnostic
    for method in ("mp2", "ccsd") + (() if active_space is None else ("casci",)):
        assert data.reference_panel[f"e_{method}"] is None
        assert data.reference_panel[method]["status"] == "not_requested"
        assert any(f"E_{method.upper()}: not_requested" in line for line in section.lines)


@pytest.mark.parametrize("method", ["mp2", "ccsd", "casci"])
def test_explicit_reference_runs_once_and_casci_rotation_does_not_change_input(
    chemistry_events,
    method,
) -> None:
    data = build_gcim_chemistry_problem(
        "fake geometry",
        active_space=(2, 2),
        reference_methods=(method, method.upper()),
    )
    assert chemistry_events == ["rhf", "h1eff", "h2eff", method]
    assert data.reference_panel[method]["status"] == "computed"
    assert data.reference_panel[f"e_{method}"] == -0.6
    assert np.array_equal(data.one_body_integrals, np.diag([-1.0, 1.0]))
    assert data.metadata["orbital_frame"] == (
        "RHF molecular orbitals before reference calculations; each orbital's first AO "
        "coefficient above 1e-08 of its largest magnitude is positive"
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"reference_methods": ("unknown",)}, "unsupported chemistry reference_methods"),
        ({"reference_methods": ("casci",)}, "requires active_space"),
    ],
)
def test_reference_method_admission_precedes_chemistry_work(monkeypatch, kwargs, message) -> None:
    from nwqlib.algorithms.gcim import chemistry

    def forbidden():
        pytest.fail("invalid reference request reached chemistry dependencies")

    monkeypatch.setattr(chemistry, "_require_chemistry_extras", forbidden)
    with pytest.raises(ValueError, match=message):
        build_gcim_chemistry_problem("fake geometry", **kwargs)


def test_chemistry_import_guard_message_names_optional_extra() -> None:
    assert "nwqlib[chemistry]" in CHEMISTRY_EXTRA_MESSAGE
    assert "openfermion" in CHEMISTRY_EXTRA_MESSAGE
    assert "pyscf" in CHEMISTRY_EXTRA_MESSAGE


def test_scientific_cardinalities_reject_before_unneeded_chemistry_work(
    chemistry_events,
    monkeypatch,
) -> None:
    from nwqlib.algorithms.gcim import chemistry

    with pytest.raises(ValueError, match="molecule electron count"):
        build_gcim_chemistry_problem("fake geometry", active_space=(4, 2))
    assert chemistry_events == []

    def forbidden():
        pytest.fail("invalid active-space counts reached chemistry dependencies")

    monkeypatch.setattr(chemistry, "_require_chemistry_extras", forbidden)
    for active_space in ((2.9, 2.9), (2, 2.9), (True, 2)):
        with pytest.raises(ValueError, match="integer"):
            build_gcim_chemistry_problem("fake geometry", active_space=active_space)
        with pytest.raises(ValueError, match="integer"):
            chemistry._validate_active_space(
                active_space, total_electrons=4, system_spatial_orbitals=4
            )
    assert chemistry._validate_active_space(
        (np.int64(2), np.int64(2)), total_electrons=4, system_spatial_orbitals=4
    ) == (2, 2)
    assert chemistry.closed_shell_reference_occupations(
        n_spatial_orbitals=np.int64(2), num_electrons=np.int64(2)
    ) == (1, 1, 0, 0)
    for counts in ((2.9, 2), (2, 2.9)):
        with pytest.raises(ValueError, match="integer"):
            chemistry.closed_shell_reference_occupations(
                n_spatial_orbitals=counts[0], num_electrons=counts[1]
            )


def test_correlation_fraction_formula_on_synthetic_inputs() -> None:
    assert correlation_fraction(-1.15, -1.0, -1.2) == pytest.approx(0.75, rel=0, abs=2e-15)
    with pytest.raises(ValueError, match="E_CCSD - E_HF"):
        correlation_fraction(-1.0, -1.0, -1.0)


def test_stored_h2_table_matches_published_m7_to_its_printed_digits() -> None:
    # The stored electronic table plus 1/R must reproduce every printed
    # coefficient of Kyriienko arXiv:1901.09988, doi:10.1038/s41534-019-0239-7,
    # Eq. (M7), including the qubit
    # placement and
    # signs of the four XY terms. The allowance is the half unit of the sixth
    # printed decimal plus 1e-9 for the binary64 table and the Bohr constant.
    stored = dict(H2_STO3G_JW_TERMS)
    stored["IIII"] += H2_STO3G_NUCLEAR_REPULSION
    assert set(stored) == {label for label, _ in H2_STO3G_M7_TERMS}
    for label, printed in H2_STO3G_M7_TERMS:
        assert stored[label] == pytest.approx(printed, rel=0.0, abs=H2_STO3G_M7_ROUNDING + 1.0e-9)


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


def test_active_space_input_rejections() -> None:
    _require_chemistry_extras()

    geometry = "H 0 0 0; H 0 0 0.7414"
    with pytest.raises(ValueError, match="even active electron"):
        build_gcim_chemistry_problem(geometry, active_space=(1, 2))
    with pytest.raises(ValueError, match="active-orbital capacity"):
        build_gcim_chemistry_problem(geometry, active_space=(6, 2))
    with pytest.raises(ValueError, match="molecular orbital count"):
        build_gcim_chemistry_problem(geometry, active_space=(2, 3))
    with pytest.raises(ValueError, match="core and active orbitals"):
        build_gcim_chemistry_problem(geometry, active_space=(0, 2))


def test_heh_cation_charge_knob_builds_closed_shell_problem() -> None:
    _require_chemistry_extras()

    data = build_gcim_chemistry_problem(
        "He 0 0 0; H 0 0 0.772",
        basis="sto-3g",
        unit="Angstrom",
        charge=1,
        name="HeH+/STO-3G charge-knob coverage",
        reference_methods=(),
    )

    assert data.metadata["charge"] == 1
    assert data.num_electrons == 2
    assert data.num_qubits == 4
    assert data.reference_occupations == (1, 1, 0, 0)
    # Sanity bound, not a physics anchor: a bound cation RHF energy is negative.
    assert data.reference_panel["e_hf"] < 0.0


def test_coefficient_cutoff_drops_pauli_terms_monotonically() -> None:
    _require_chemistry_extras()

    geometry = "H 0 0 0; H 0 0 0.7414"
    counts = []
    for cutoff in (1.0e-12, 1.0e-6, 1.0e-1):
        data = build_gcim_chemistry_problem(
            geometry,
            basis="sto-3g",
            unit="Angstrom",
            coefficient_cutoff=cutoff,
            reference_methods=(),
        )
        counts.append(len(data.hamiltonian.paulis))

    # Loosening the cutoff can only remove terms; the H2/STO-3G JW table has
    # |coefficient| ~ 4.5e-2 entries, so the 1e-1 cutoff must strictly drop.
    assert counts[0] >= counts[1] >= counts[2]
    assert counts[2] < counts[0]


def test_spin_nonzero_input_is_rejected() -> None:
    _require_chemistry_extras()

    with pytest.raises(ValueError, match="spin=0"):
        build_gcim_chemistry_problem(
            "H 0 0 0; H 0 0 0.7414",
            basis="sto-3g",
            unit="Angstrom",
            spin=2,
        )


def test_reference_panel_records_mp2_ccsd_failures_without_blocking_build(monkeypatch) -> None:
    """Inject a thrown MP2 error and nonconverged CCSD result without discarding the successfully
    built Hamiltonian.
    """
    _require_chemistry_extras()
    from pyscf import cc, mp

    class BrokenMP2:
        def __init__(self, mean_field):
            self.mean_field = mean_field

        def kernel(self):
            raise RuntimeError("mp2 boom")

    class NonConvergedCCSD:
        converged = False
        e_tot = None

        def __init__(self, mean_field):
            self.mean_field = mean_field

        def kernel(self):
            return 0.0, None, None

    monkeypatch.setattr(mp, "MP2", BrokenMP2)
    monkeypatch.setattr(cc, "CCSD", NonConvergedCCSD)

    data = build_gcim_chemistry_problem(
        "H 0 0 0; H 0 0 0.7414",
        basis="sto-3g",
        unit="Angstrom",
        reference_methods=("mp2", "ccsd"),
    )

    panel = data.reference_panel
    assert panel["e_mp2"] is None
    assert panel["mp2"]["status"] == "failed"
    assert "mp2 boom" in panel["mp2"]["failure_reason"]
    assert panel["e_ccsd"] is None
    assert panel["ccsd"]["status"] == "non_converged"
    assert "did not converge" in panel["ccsd"]["failure_reason"]


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


def test_partial_chemistry_panel_renders_missing_reference() -> None:
    from nwqlib.algorithms.gcim import chemistry as gcim_chemistry

    diagnostic = chemistry_reference_diagnostic(
        {
            "chemistry_reference_panel": {
                "label": "application context, not a validation gate",
                "e_hf": -1.0,
                "mp2": {"status": "not_requested"},
            }
        },
        energy=0.0,
    )
    section = gcim_chemistry.chemistry_reference_report_section(diagnostic)
    assert section.title == "Chemistry References"
    assert any("E_CCSD" in line for line in section.lines)


def test_reference_energy_records_mp2_missing_converged_as_computed() -> None:
    import types
    from nwqlib.algorithms.gcim import chemistry as gcim_chemistry

    class Solver:
        e_tot = -1.2

        def kernel(self):
            return (-0.2,)

    mean_field = types.SimpleNamespace(e_tot=-1.0)
    mp2 = gcim_chemistry._reference_energy(
        "mp2",
        Solver,
        lambda solver: solver.kernel()[0],
        mean_field,
        missing_converged_ok=True,
    )
    ccsd = gcim_chemistry._reference_energy(
        "ccsd",
        Solver,
        lambda solver: solver.kernel()[0],
        mean_field,
        missing_converged_ok=False,
    )
    assert mp2["mp2"]["status"] == "computed"
    assert ccsd["ccsd"]["status"] == "non_converged"


def test_zero_correlation_panel_remains_undefined_and_renderable() -> None:
    from nwqlib.algorithms.gcim import chemistry as gcim_chemistry

    diagnostic = gcim_chemistry.chemistry_reference_diagnostic(
        {"chemistry_reference_panel": {"e_hf": -2.807, "e_ccsd": -2.807}},
        energy=-2.807,
    )
    assert "correlation_fraction" not in diagnostic
    assert diagnostic["correlation_fraction_note"] == "undefined: E_CCSD == E_HF"
    section = gcim_chemistry.chemistry_reference_report_section(diagnostic)
    assert any("undefined" in line for line in section.lines)


def test_numpy_bool_convergence_flags_are_respected() -> None:
    import types
    from nwqlib.algorithms.gcim import chemistry as gcim_chemistry

    class _FakeSolver:
        converged = np.bool_(False)
        e_tot = -1.23

    record = gcim_chemistry._reference_energy(
        "ccsd",
        lambda: _FakeSolver(),
        lambda solver: -0.1,
        types.SimpleNamespace(e_tot=-1.0),
        missing_converged_ok=False,
    )
    assert record["e_ccsd"] is None
    assert record["ccsd"]["status"] == "non_converged"
