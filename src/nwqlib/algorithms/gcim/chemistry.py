"""Chemistry input preparation for GCiM and ADAPT-GCiM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from nwqlib._validation import integer
from nwqlib.reporting import ReportSection
from nwqlib.subroutines._registry_utils import package_versions

CHEMISTRY_EXTRA_MESSAGE = (
    "GCiM chemistry input requires optional dependencies openfermion and pyscf; "
    'install them with pip install "nwqlib[chemistry]".'
)
# Coefficients below this fraction of an MO's largest AO magnitude do not set its
# sign. Symmetry zeros carry roundoff near 1e-16 of the scale; see
# docs/ENGINEERING_CONSTANTS.md.
ORBITAL_PHASE_RELATIVE_THRESHOLD = 1.0e-8


@dataclass(frozen=True, kw_only=True)
class GCIMChemistryProblemData:
    """Molecular Hamiltonian, reference state and integrals from `build_gcim_chemistry_problem`.

    `build_gcim_chemistry_problem` returns it. `eigenproblem()` gives the
    `Eigenproblem`, and `adapt_method(...)` an `ADAPT` method with the
    matching reference state. Energies and integral coefficients are in
    Hartree. The builder uses the converged closed-shell RHF molecular orbitals
    and interleaved alpha/beta spin modes. Each orbital's sign is fixed so
    that its first significant AO coefficient is positive. This removes the
    dependence on the eigensolver's sign choice, which could flip the
    generators that contain the orbital and change a fixed-angle ADAPT-GCIM
    trajectory. It fixes signs only. Degenerate orbitals can still be
    returned in another rotation of their subspace, and binary64 rounding
    can still decide between generators whose gradients tie in exact
    arithmetic ([GCiM guide](../../algorithms/gcim.md#adaptive-generator-coordinates)).
    In tensor shapes below, n is `n_spatial_orbitals`. Requested reference
    calculations give application context and do not certify the solver.
    The fields below are read-only.

    Attributes:
        hamiltonian: Jordan-Wigner SparsePauliOp for the selected full or active space,
            including its constant offset and the requested coefficient cutoff.
        reference_preparation: Computational-basis circuit preparing the recorded
            closed-shell occupations from zero. Inspecting it runs no chemistry solve.
        reference_occupations: Zero/one occupation per qubit in interleaved spatial-orbital
            alpha/beta order, which `adapt_method` uses as the reference state.
        n_spatial_orbitals: Number of selected spatial orbitals, reduced to the active
            orbital count when an active space was requested.
        num_qubits: Twice n_spatial_orbitals for the Jordan-Wigner spin-orbital register.
        num_electrons: Electrons in the selected Hamiltonian space, which is
            the active electron count when frozen-core reduction is used.
        nuclear_repulsion_energy: Original molecule's nuclear repulsion contribution.
        constant_energy: Scalar offset before Jordan-Wigner mapping: nuclear repulsion
            for the full space, or the returned frozen-core energy for an active space.
            This is not necessarily the entire final Pauli-identity coefficient.
        rhf_energy: Converged full-molecule RHF total energy, including nuclear repulsion.
        one_body_integrals: Spatial-orbital tensor of shape (n,n) in the RHF orbital basis, or
            the active-space effective one-body tensor with frozen-core contributions.
        two_body_integrals: Spatial-orbital tensor of shape (n,n,n,n), obtained by
            transposing the chemist-ordered integrals with axes (0,2,3,1). After spin
            expansion, entries multiply a_p^dagger a_q^dagger a_r a_s with factor 1/2.
        reference_panel: RHF and explicitly requested MP2/CCSD/CASCI values/statuses.
            Unrequested or failed reference energies stay None with their status/reason.
        metadata: Geometry and basis, charge, ordering, cutoff, active space
            and dependency versions of this construction. `to_dict` omits the
            large integral tensors.
    """

    hamiltonian: SparsePauliOp
    reference_preparation: QuantumCircuit
    reference_occupations: tuple[int, ...]
    n_spatial_orbitals: int
    num_qubits: int
    num_electrons: int
    nuclear_repulsion_energy: float
    constant_energy: float
    rhf_energy: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    reference_panel: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def eigenproblem(self):
        """Return the `Eigenproblem` of `hamiltonian`, in Hartree with its constant offset."""
        from nwqlib.problems.records import Eigenproblem

        return Eigenproblem(A=self.hamiltonian, unit="Hartree")

    def adapt_method(self, **settings):
        """Return an `ADAPT` method with this molecule's reference state and the spin-adapted pool.

        Args:
            **settings (object): Further `ADAPT` arguments. They override the defaults
                set here: `initial_state` (the occupation state of
                `reference_occupations`), `pool="spin_adapted_sd"` and
                `n_spatial_orbitals`.

        Returns:
            method (ADAPT): The configured method.
        """
        from nwqlib.problems.inputs import ingest_occupation
        from .adapt import ADAPT

        reference = ingest_occupation(self.reference_occupations, num_qubits=self.num_qubits)
        return ADAPT(
            **dict(
                initial_state=reference,
                pool="spin_adapted_sd",
                n_spatial_orbitals=self.n_spatial_orbitals,
            )
            | settings
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like summary of the fields, without the integral tensors.

        The summary holds the orbital, qubit and electron counts, the
        nuclear-repulsion, constant and RHF energies, the reference panel,
        the reference occupations, the Pauli term count of `hamiltonian` and
        `metadata`.
        """

        return {
            "n_spatial_orbitals": self.n_spatial_orbitals,
            "num_qubits": self.num_qubits,
            "num_electrons": self.num_electrons,
            "nuclear_repulsion_energy": self.nuclear_repulsion_energy,
            "constant_energy": self.constant_energy,
            "rhf_energy": self.rhf_energy,
            "reference_panel": dict(self.reference_panel),
            "reference_occupations": list(self.reference_occupations),
            "pauli_term_count": len(self.hamiltonian.paulis),
            "metadata": dict(self.metadata),
        }


def build_gcim_chemistry_problem(
    geometry: Any,
    *,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    name: str | None = None,
    coefficient_cutoff: float = 1.0e-12,
    active_space: tuple[int, int] | None = None,
    reference_methods: tuple[str, ...] = (),
) -> GCIMChemistryProblemData:
    """Build a closed-shell molecular Hamiltonian for GCiM and ADAPT from a geometry.

    PySCF performs RHF and the molecular-orbital integral transforms, NWQLib
    assembles the spin-orbital fermionic operator, and OpenFermion performs
    only the Jordan-Wigner transform. It needs the `chemistry` extra,
    `pip install "nwqlib[chemistry]"`. MP2, CCSD or active-space CASCI
    references run only when `reference_methods` requests them. Integrals
    are in the RHF molecular-orbital basis, with each orbital's first AO coefficient above
    `1e-8` of its largest magnitude made positive.

    The Hamiltonian is
    `E_c + sum h_pq a_p^dagger a_q + 1/2 sum h_pqrs a_p^dagger a_q^dagger a_r a_s`
    over interleaved spin orbitals, mapped by Jordan-Wigner with spin orbital
    j on qubit j.

    Args:
        geometry: PySCF atom specification, for example `"H 0 0 0; H 0 0 0.74"`.
        basis: PySCF basis name.
        charge: Total molecular charge.
        spin: `2S`. Only 0, closed-shell RHF, is supported.
        unit: Unit of the coordinates, `"Angstrom"` or `"Bohr"`.
        name: Optional label stored in `metadata`.
        coefficient_cutoff: Absolute cutoff in Hartree. Fermionic terms and
            final Pauli coefficients at or below it are dropped, and an
            imaginary part at or below it is set to zero.
        active_space: Optional `(n_active_electrons, n_active_orbitals)`.
            For a molecule of N electrons the lowest
            `(N - n_active_electrons)/2` RHF orbitals are frozen as doubly
            occupied core, and PySCF CASCI supplies the effective one- and
            two-body integrals and the core energy.
        reference_methods: Any of `"mp2"`, `"ccsd"` and `"casci"` to run as
            application context. `"casci"` needs `active_space`.

    Returns:
        data (GCIMChemistryProblemData): The Hamiltonian, reference state and
            integrals of the selected full or active space.

    Raises:
        ValueError: For an open-shell or odd-electron molecule, an invalid
            active space, an unsupported reference method, `"casci"` without
            `active_space`, a negative cutoff or an unconverged RHF.
        ImportError: If PySCF or OpenFermion is not installed.

    Examples:
        H2 in the STO-3G basis at 0.74 Angstrom maps to four qubits. ADAPT
        with the default spin-adapted pool returns -1.137284 Hartree, the
        lowest eigenvalue of `data.hamiltonian` by exact diagonalization,
        below the RHF energy of -1.116759 Hartree:

        >>> from nwqlib import solve
        >>> from nwqlib.algorithms.gcim import build_gcim_chemistry_problem
        >>> data = build_gcim_chemistry_problem("H 0 0 0; H 0 0 0.74")
        >>> print(data.num_qubits, data.reference_occupations)
        4 (1, 1, 0, 0)
        >>> print(round(data.rhf_energy, 6))
        -1.116759
        >>> result = solve(data.eigenproblem(), method=data.adapt_method(), seed=7)
        >>> print(round(result.eigenvalue, 6))
        -1.137284
    """

    methods = tuple(str(method).lower() for method in reference_methods)
    if "casci" in methods and active_space is None:
        raise ValueError("CASCI reference method requires active_space")
    if spin != 0:
        raise ValueError("GCiM chemistry input currently supports closed-shell RHF only (spin=0)")
    if coefficient_cutoff < 0.0:
        raise ValueError("coefficient_cutoff must be non-negative")
    if active_space is not None:
        active_space = _active_space_counts(active_space)
    extras = _require_chemistry_extras()

    mol = extras.gto.M(
        atom=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
        verbose=0,
    )
    if mol.nelectron % 2:
        raise ValueError("closed-shell RHF chemistry input requires an even electron count")
    if active_space is not None:
        _validate_active_space(active_space, total_electrons=int(mol.nelectron))

    mean_field = extras.scf.RHF(mol)
    rhf_energy = float(mean_field.kernel())
    if not bool(mean_field.converged):
        raise ValueError("PySCF RHF did not converge for the supplied geometry")
    # LAPACK fixes each MO only up to sign, and the sign can differ between
    # builds. Flipping an orbital flips generators such as A -> -A, and a
    # fixed-angle ADAPT-GCIM product exp(theta A) then follows another trajectory.
    # Reference calculations reuse the same frame through mean_field.
    mo_coefficients = _fixed_orbital_phase(np.asarray(mean_field.mo_coeff))
    mean_field.mo_coeff = mo_coefficients
    system_spatial_orbitals = int(mo_coefficients.shape[1])
    nuclear_repulsion = float(mol.energy_nuc())
    casci = None
    if active_space is None:
        n_spatial_orbitals = system_spatial_orbitals
        num_electrons = int(mol.nelectron)
        one_body = np.asarray(mo_coefficients.T @ mean_field.get_hcore() @ mo_coefficients)
        eri_mo = extras.ao2mo.kernel(mol, mo_coefficients)
        two_body = np.asarray(
            extras.ao2mo.restore(1, eri_mo, n_spatial_orbitals).transpose(0, 2, 3, 1)
        )
        constant_energy = nuclear_repulsion
        active_space_metadata = None
    else:
        n_active_electrons, n_active_orbitals = _validate_active_space(
            active_space,
            total_electrons=int(mol.nelectron),
            system_spatial_orbitals=system_spatial_orbitals,
        )
        casci = extras.mcscf.CASCI(mean_field, n_active_orbitals, n_active_electrons)
        h1eff, core_energy = casci.get_h1eff(mo_coefficients)
        h2eff = casci.get_h2eff(mo_coefficients)
        n_spatial_orbitals = n_active_orbitals
        num_electrons = n_active_electrons
        one_body = np.asarray(h1eff)
        two_body = np.asarray(
            extras.ao2mo.restore(1, h2eff, n_active_orbitals).transpose(0, 2, 3, 1)
        )
        constant_energy = float(core_energy)
        active_space_metadata = {
            "n_active_electrons": n_active_electrons,
            "n_active_orbitals": n_active_orbitals,
            "system_spatial_orbitals": system_spatial_orbitals,
            "core_energy": constant_energy,
        }

    # Extract RHF-frame integrals before reference kernels can rotate orbitals.
    reference_panel = _chemistry_reference_panel(
        extras,
        mean_field,
        rhf_energy,
        reference_methods=methods,
        casci=casci,
    )
    if active_space_metadata is not None:
        reference_panel["casci"].update(active_space_metadata)

    one_spin, two_spin = _spin_orbital_integrals(one_body, two_body)
    fermion_operator = _assemble_spin_orbital_fermion_operator(
        extras.FermionOperator,
        one_spin,
        two_spin,
        constant_energy=constant_energy,
        coefficient_cutoff=coefficient_cutoff,
    )
    qubit_operator = extras.jordan_wigner(fermion_operator)
    num_qubits = 2 * n_spatial_orbitals
    hamiltonian = qubit_operator_to_sparse_pauli(
        qubit_operator,
        num_qubits=num_qubits,
        coefficient_cutoff=coefficient_cutoff,
    )
    occupations = closed_shell_reference_occupations(
        n_spatial_orbitals=n_spatial_orbitals,
        num_electrons=num_electrons,
    )
    reference_circuit = QuantumCircuit(num_qubits, name="RHF")
    for mode, occupied in enumerate(occupations):
        if occupied:
            reference_circuit.x(mode)
    metadata = {
        "name": name or f"{mol.atom_symbol(0)}.../{basis} GCiM chemistry input",
        "geometry": geometry,
        "basis": basis,
        "charge": int(charge),
        "spin": int(spin),
        "unit": unit,
        "integral_convention": "MO two-body tensor h[p,q,r,s] = (ps|qr)",
        "orbital_frame": (
            "RHF molecular orbitals before reference calculations; each orbital's first AO "
            f"coefficient above {ORBITAL_PHASE_RELATIVE_THRESHOLD:g} of its largest magnitude is positive"
        ),
        "fermion_operator_term_count": len(fermion_operator.terms),
        "pauli_term_count": len(hamiltonian.paulis),
        "nuclear_repulsion_energy": nuclear_repulsion,
        "constant_energy": constant_energy,
        "rhf_energy": rhf_energy,
        "chemistry_reference_panel": reference_panel,
        "active_space": active_space_metadata,
        "dependency_versions": _chemistry_versions(),
    }

    return GCIMChemistryProblemData(
        hamiltonian=hamiltonian,
        reference_preparation=reference_circuit,
        reference_occupations=occupations,
        n_spatial_orbitals=n_spatial_orbitals,
        num_qubits=num_qubits,
        num_electrons=num_electrons,
        nuclear_repulsion_energy=nuclear_repulsion,
        constant_energy=constant_energy,
        rhf_energy=rhf_energy,
        reference_panel=reference_panel,
        one_body_integrals=one_body,
        two_body_integrals=two_body,
        metadata=metadata,
    )


def correlation_fraction(energy: float, e_hf: float, e_ccsd: float) -> float:
    """Return the correlation fraction `(E - E_HF) / (E_CCSD - E_HF)` of an energy.

    The fraction gives application context, not a bound on the error of E.
    `chemistry_reference_diagnostic` uses it with the stored reference
    energies.

    Args:
        energy: The energy E to compare.
        e_hf: The Hartree-Fock energy, in the unit of E.
        e_ccsd: The CCSD energy, or another reference energy such as CASCI,
            in the unit of E.

    Returns:
        fraction (float): `(E - E_HF) / (E_CCSD - E_HF)`.

    Raises:
        ValueError: If `E_CCSD - E_HF` is zero.
    """

    denominator = float(e_ccsd) - float(e_hf)
    if denominator == 0.0:
        raise ValueError("E_CCSD - E_HF must be nonzero")
    return (float(energy) - float(e_hf)) / denominator


def chemistry_reference_diagnostic(
    metadata: Mapping[str, Any],
    *,
    energy: float | None,
) -> dict[str, Any] | None:
    """Return the stored chemistry reference energies and correlation fractions for an energy.

    It reads `metadata["chemistry_reference_panel"]`, which
    `build_gcim_chemistry_problem` stores, and adds the full-space CCSD
    correlation fraction `(E - E_HF)/(E_CCSD - E_HF)` and the active-space
    CASCI fraction `(E - E_HF)/(E_CASCI - E_HF)` when the needed energies
    are present. A zero denominator is recorded as the note
    `"undefined: E_CCSD == E_HF"` or `"undefined: E_CASCI == E_HF"`. It runs
    no calculation. The values give application context, not a bound on the
    error of E or an identification of the ground state. The
    [GCiM guide](../../algorithms/gcim.md#adaptive-generator-coordinates)
    prints the report section of such a diagnostic.

    Args:
        metadata: The `metadata` of a `GCIMChemistryProblemData`.
        energy: The energy E to compare, for example `result.eigenvalue`, or
            None.

    Returns:
        diagnostic (dict | None): The reference panel and the fractions, or
            None when `metadata` holds no reference panel.
    """

    panel = metadata.get("chemistry_reference_panel")
    if not isinstance(panel, Mapping):
        return None
    diagnostic: dict[str, Any] = {
        "label": "application context, not a validation gate",
        "reference_panel": dict(panel),
    }
    e_hf = panel.get("e_hf")
    e_ccsd = panel.get("e_ccsd")
    if energy is not None and e_hf is not None and e_ccsd is not None:
        # A zero-correlation panel (E_CCSD == E_HF) has no defined fraction;
        # the panel is context, not a gate, so record why instead of failing
        # a completed run during result assembly.
        if float(e_ccsd) == float(e_hf):
            diagnostic["correlation_fraction_note"] = "undefined: E_CCSD == E_HF"
        else:
            diagnostic["correlation_fraction"] = correlation_fraction(
                energy,
                float(e_hf),
                float(e_ccsd),
            )
    e_casci = panel.get("e_casci")
    if energy is not None and e_hf is not None and e_casci is not None:
        if float(e_casci) == float(e_hf):
            diagnostic["active_space_casci_correlation_fraction_note"] = (
                "undefined: E_CASCI == E_HF"
            )
        else:
            diagnostic["active_space_casci_correlation_fraction"] = correlation_fraction(
                energy,
                float(e_hf),
                float(e_casci),
            )
    return diagnostic


def chemistry_reference_report_section(diagnostic: Mapping[str, Any]) -> ReportSection:
    """Format a chemistry reference diagnostic as a "Chemistry References" report section.

    The section lists E_HF, E_MP2 and E_CCSD, plus E_CASCI for an active
    space, and the correlation fractions that the diagnostic holds. A
    missing energy shows its recorded status and reason, or `unavailable`.
    `result.report()` does not include this section, so print or store it
    beside the report.

    Args:
        diagnostic: A diagnostic from `chemistry_reference_diagnostic`.

    Returns:
        section (ReportSection): The formatted lines in `section.lines` and
            the diagnostic in `section.data`.
    """

    panel = diagnostic["reference_panel"]
    lines = [
        "label: application context, not a validation gate",
        _energy_line("E_HF", panel.get("e_hf"), None),
    ]
    lines.append(_energy_line("E_MP2", panel.get("e_mp2"), panel.get("mp2")))
    lines.append(_energy_line("E_CCSD", panel.get("e_ccsd"), panel.get("ccsd")))
    if "casci" in panel:
        lines.append(_energy_line("E_CASCI", panel.get("e_casci"), panel.get("casci")))
    if "correlation_fraction" in diagnostic:
        lines.append(
            f"full-space CCSD correlation fraction: {diagnostic['correlation_fraction']:.12g}"
        )
    if "correlation_fraction_note" in diagnostic:
        lines.append(
            f"full-space CCSD correlation fraction: {diagnostic['correlation_fraction_note']}"
        )
    if "active_space_casci_correlation_fraction" in diagnostic:
        lines.append(
            "active-space CASCI correlation fraction: "
            f"{diagnostic['active_space_casci_correlation_fraction']:.12g}"
        )
    if "active_space_casci_correlation_fraction_note" in diagnostic:
        lines.append(
            "active-space CASCI correlation fraction: "
            f"{diagnostic['active_space_casci_correlation_fraction_note']}"
        )
    return ReportSection(
        title="Chemistry References",
        lines=tuple(lines),
        data=dict(diagnostic),
    )


def _energy_line(label: str, energy: Any, status_record: Any) -> str:
    """Format one reference energy, or its recorded status and reason when it is absent."""
    if energy is not None:
        return f"{label}: {float(energy):.12g}"
    if isinstance(status_record, Mapping):
        status = status_record.get("status", "unavailable")
        reason = status_record.get("failure_reason", "no detail recorded")
        return f"{label}: {status} ({reason})"
    return f"{label}: unavailable"


def closed_shell_reference_occupations(
    *,
    n_spatial_orbitals: int,
    num_electrons: int,
) -> tuple[int, ...]:
    """Return the closed-shell RHF occupation bitstring in interleaved spin-orbital order.

    Spin orbitals `2k` and `2k + 1` are the alpha and beta orbitals of
    spatial orbital k, and the lowest `num_electrons/2` spatial orbitals are
    doubly occupied.

    Args:
        n_spatial_orbitals: Positive number of spatial orbitals.
        num_electrons: Nonnegative even number of electrons, at most
            `2*n_spatial_orbitals`.

    Returns:
        occupations (tuple[int, ...]): `2*n_spatial_orbitals` zeros and ones.

    Raises:
        ValueError: For an odd electron count or more electrons than the
            orbitals can hold.

    Examples:
        >>> from nwqlib.algorithms.gcim import closed_shell_reference_occupations
        >>> closed_shell_reference_occupations(n_spatial_orbitals=3, num_electrons=2)
        (1, 1, 0, 0, 0, 0)
    """

    n_spatial_orbitals = integer(n_spatial_orbitals, "n_spatial_orbitals", 1)
    num_electrons = integer(num_electrons, "num_electrons", 0)
    if num_electrons % 2:
        raise ValueError("closed-shell occupations require a non-negative even electron count")
    electron_pairs = num_electrons // 2
    if electron_pairs > n_spatial_orbitals:
        raise ValueError("num_electrons exceeds the closed-shell orbital capacity")
    occupations = [0] * (2 * n_spatial_orbitals)
    for orbital in range(electron_pairs):
        occupations[2 * orbital] = 1
        occupations[2 * orbital + 1] = 1
    return tuple(occupations)


def _active_space_counts(active_space: tuple[int, int]) -> tuple[int, int]:
    """Return ``(n_active_electrons, n_active_orbitals)`` for a closed-shell active space.

    The electron count must be even and fit in the active orbitals. This
    check needs no molecule, so it runs before PySCF is imported.
    """
    try:
        n_active_electrons, n_active_orbitals = active_space
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "active_space must be a tuple (n_active_electrons, n_active_orbitals)"
        ) from exc
    n_active_electrons = integer(n_active_electrons, "active_space electron count", 0)
    n_active_orbitals = integer(n_active_orbitals, "active_space orbital count", 1)
    if n_active_electrons % 2:
        raise ValueError("active_space requires an even active electron count")
    if n_active_electrons > 2 * n_active_orbitals:
        raise ValueError("active_space active electrons exceed active-orbital capacity")
    return n_active_electrons, n_active_orbitals


def _validate_active_space(
    active_space: tuple[int, int],
    *,
    total_electrons: int,
    system_spatial_orbitals: int | None = None,
) -> tuple[int, int]:
    """Check an active space against the molecule and return its counts.

    The active electrons cannot exceed the molecule's electrons. Once the
    RHF orbital count is known, the ``(total_electrons -
    n_active_electrons)/2`` doubly occupied core orbitals plus the active
    orbitals must fit in it, as PySCF CASCI places the core below the active
    orbitals.
    """
    n_active_electrons, n_active_orbitals = _active_space_counts(active_space)
    if n_active_electrons > total_electrons:
        raise ValueError("active_space active electrons cannot exceed the molecule electron count")
    if system_spatial_orbitals is None:
        return n_active_electrons, n_active_orbitals
    if n_active_orbitals > system_spatial_orbitals:
        raise ValueError("active_space active orbitals exceed the molecular orbital count")
    n_core_orbitals = (total_electrons - n_active_electrons) // 2
    if n_core_orbitals + n_active_orbitals > system_spatial_orbitals:
        raise ValueError("active_space core and active orbitals exceed the molecular orbital count")
    return n_active_electrons, n_active_orbitals


def qubit_operator_to_sparse_pauli(
    qubit_operator: Any,
    *,
    num_qubits: int,
    coefficient_cutoff: float = 1.0e-12,
) -> SparsePauliOp:
    """Convert an OpenFermion `QubitOperator` to a Qiskit `SparsePauliOp`.

    In a Qiskit label qubit 0 is the rightmost character, so OpenFermion
    qubit j becomes label position `num_qubits - 1 - j`. Terms with
    magnitude at or below `coefficient_cutoff` are dropped, an imaginary part
    at or below it is set to zero, and equal labels are then combined by
    `SparsePauliOp.simplify` with the same absolute tolerance.

    Args:
        qubit_operator: The OpenFermion `QubitOperator`.
        num_qubits: Positive number of qubits.
        coefficient_cutoff: Nonnegative absolute cutoff on coefficients.

    Returns:
        operator (SparsePauliOp): The converted operator, or the zero
            operator on `num_qubits` qubits when no term remains.

    Raises:
        ValueError: If `num_qubits` is not positive, `coefficient_cutoff` is
            negative, or a term acts on a qubit outside `num_qubits`.
    """

    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")
    if coefficient_cutoff < 0.0:
        raise ValueError("coefficient_cutoff must be non-negative")

    pauli_terms: list[tuple[str, complex | float]] = []
    for term, coefficient in qubit_operator.terms.items():
        coefficient = complex(coefficient)
        if abs(coefficient) <= coefficient_cutoff:
            continue
        label = ["I"] * num_qubits
        for qubit, pauli in term:
            if qubit < 0 or qubit >= num_qubits:
                raise ValueError("OpenFermion QubitOperator term exceeds num_qubits")
            label[num_qubits - 1 - qubit] = pauli
        value: complex | float
        if abs(coefficient.imag) <= coefficient_cutoff:
            value = float(coefficient.real)
        else:
            value = coefficient
        pauli_terms.append(("".join(label), value))
    if not pauli_terms:
        pauli_terms = [("I" * num_qubits, 0.0)]
    return SparsePauliOp.from_list(pauli_terms).simplify(atol=coefficient_cutoff)


@dataclass(frozen=True)
class _ChemistryExtras:
    """The OpenFermion and PySCF objects this module uses, imported on first use."""

    FermionOperator: Any
    jordan_wigner: Any
    ao2mo: Any
    cc: Any
    gto: Any
    mcscf: Any
    mp: Any
    scf: Any


def _require_chemistry_extras() -> _ChemistryExtras:
    """Import the optional chemistry dependencies, or raise ``ImportError`` with the install hint.

    The imports happen here, not at module import, so this module and the
    rest of the GCiM package import without PySCF or OpenFermion.
    """
    try:
        from openfermion.ops import FermionOperator
        from openfermion.transforms import jordan_wigner
        from pyscf import ao2mo, cc, gto, mcscf, mp, scf
    except ImportError as exc:
        raise ImportError(CHEMISTRY_EXTRA_MESSAGE) from exc
    return _ChemistryExtras(
        FermionOperator=FermionOperator,
        jordan_wigner=jordan_wigner,
        ao2mo=ao2mo,
        cc=cc,
        gto=gto,
        mcscf=mcscf,
        mp=mp,
        scf=scf,
    )


def _chemistry_reference_panel(
    extras: _ChemistryExtras,
    mean_field: Any,
    rhf_energy: float,
    *,
    reference_methods: tuple[str, ...],
    casci: Any | None,
) -> dict[str, Any]:
    """Run only requested molecular reference methods and preserve missing or failed results as
    context.
    """
    panel: dict[str, Any] = {
        "label": "application context, not a validation gate",
        "e_hf": float(rhf_energy),
        "hf": {"status": "computed", "energy": float(rhf_energy)},
    }
    if "mp2" in reference_methods:
        panel.update(_mp2_reference(extras, mean_field))
    else:
        panel.update(
            {
                "e_mp2": None,
                "mp2": {
                    "status": "not_requested",
                    "failure_reason": "MP2 reference method not requested",
                },
            }
        )
    if "ccsd" in reference_methods:
        panel.update(_ccsd_reference(extras, mean_field))
    else:
        panel.update(
            {
                "e_ccsd": None,
                "ccsd": {
                    "status": "not_requested",
                    "failure_reason": "CCSD reference method not requested",
                },
            }
        )
    if casci is not None:
        if "casci" in reference_methods:
            panel.update(
                _reference_energy(
                    "casci",
                    lambda: casci,
                    lambda solver: solver.kernel()[0] - rhf_energy,
                    mean_field,
                    missing_converged_ok=False,
                )
            )
        else:
            panel.update(
                {
                    "e_casci": None,
                    "casci": {
                        "status": "not_requested",
                        "failure_reason": "CASCI reference method not requested",
                    },
                }
            )
    return panel


def _mp2_reference(extras: _ChemistryExtras, mean_field: Any) -> dict[str, Any]:
    # PySCF 2.13 MP2 does not expose ``converged``; absence means "not
    # reported", not failure. CCSD below does expose it and must pass.
    return _reference_energy(
        "mp2",
        lambda: extras.mp.MP2(mean_field),
        lambda solver: solver.kernel()[0],
        mean_field,
        missing_converged_ok=True,
    )


def _ccsd_reference(extras: _ChemistryExtras, mean_field: Any) -> dict[str, Any]:
    return _reference_energy(
        "ccsd",
        lambda: extras.cc.CCSD(mean_field),
        lambda solver: solver.kernel()[0],
        mean_field,
        missing_converged_ok=False,
    )


def _reference_energy(
    key: str,
    make_solver: Any,
    correlation_energy_from_kernel: Any,
    mean_field: Any,
    *,
    missing_converged_ok: bool,
) -> dict[str, Any]:
    """Run one reference solver and record its energy or the reason it is absent.

    A reference panel is application context. A failed or non-converged
    reference is therefore recorded with its status and reason, and the
    Hamiltonian build continues. A solver that exposes a false convergence
    flag never contributes an energy.
    """
    label = key.upper()
    try:
        solver = make_solver()
        correlation_energy = correlation_energy_from_kernel(solver)
        converged = getattr(solver, "converged", None)
        # bool() so numpy-bool convergence flags cannot slip past an
        # identity comparison and feed a garbage energy into the panel.
        not_converged = (converged is None and not missing_converged_ok) or (
            converged is not None and not bool(converged)
        )
        if not_converged:
            return {
                f"e_{key}": None,
                key: {
                    "status": "non_converged",
                    "failure_reason": f"PySCF {label} did not converge",
                },
            }
        total = solver.e_tot
        if total is None:
            total = mean_field.e_tot + correlation_energy
        return {
            f"e_{key}": float(total),
            key: {
                "status": "computed",
                "energy": float(total),
                "correlation_energy": float(correlation_energy),
            },
        }
    except Exception as exc:  # pragma: no cover - exercised by monkeypatch.
        return {
            f"e_{key}": None,
            key: {"status": "failed", "failure_reason": str(exc)},
        }


def _fixed_orbital_phase(mo_coefficients: np.ndarray) -> np.ndarray:
    """Make each MO's first AO coefficient above the relative threshold positive.

    The rule reads the first significant coefficient rather than the largest one,
    because symmetry-equivalent atoms give largest magnitudes that differ only
    by roundoff. It fixes signs only; a degenerate orbital subspace can still be
    returned in another rotation.
    """
    magnitudes = np.abs(mo_coefficients)
    significant = magnitudes > ORBITAL_PHASE_RELATIVE_THRESHOLD * magnitudes.max(axis=0)
    first = np.argmax(significant, axis=0)
    return mo_coefficients * np.sign(mo_coefficients[first, np.arange(mo_coefficients.shape[1])])


def _spin_orbital_integrals(
    one_body: np.ndarray,
    two_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand spatial-orbital integrals to the interleaved spin-orbital basis."""
    n_spatial = int(one_body.shape[0])
    if one_body.shape != (n_spatial, n_spatial):
        raise ValueError("one_body integrals must be a square matrix")
    if two_body.shape != (n_spatial, n_spatial, n_spatial, n_spatial):
        raise ValueError("two_body integrals must be a rank-4 spatial-orbital tensor")

    n_spin = 2 * n_spatial
    one_spin = np.zeros((n_spin, n_spin), dtype=complex)
    two_spin = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=complex)
    # Spin orbital 2k is alpha and 2k+1 is beta. The one-body term conserves
    # spin. In h[p,q,r,s] = (ps|qr) each electron keeps its spin from s to p
    # and from r to q. Only the four blocks with spins (sigma,tau,tau,sigma),
    # two same-spin and two mixed-spin, are nonzero.
    one_spin[0::2, 0::2] = one_body
    one_spin[1::2, 1::2] = one_body
    two_spin[0::2, 0::2, 0::2, 0::2] = two_body
    two_spin[1::2, 1::2, 1::2, 1::2] = two_body
    two_spin[0::2, 1::2, 1::2, 0::2] = two_body
    two_spin[1::2, 0::2, 0::2, 1::2] = two_body
    return one_spin, two_spin


def _assemble_spin_orbital_fermion_operator(
    FermionOperator: Any,
    one_spin: np.ndarray,
    two_spin: np.ndarray,
    *,
    constant_energy: float,
    coefficient_cutoff: float,
) -> Any:
    """Build ``E_c + sum h_pq a_p^dagger a_q + 1/2 sum h_pqrs a_p^dagger a_q^dagger a_r a_s``.

    The tensor is ``h_pqrs = (ps|qr)`` in chemist notation. The unrestricted
    sum counts each electron pair twice, so the two-body term carries 1/2.
    Terms at or below ``coefficient_cutoff`` in
    magnitude are omitted. The Python loop visits all ``n_spin**4`` entries.
    """
    n_spin = int(one_spin.shape[0])
    fermion_operator = FermionOperator((), constant_energy)
    for p in range(n_spin):
        for q in range(n_spin):
            coefficient = complex(one_spin[p, q])
            if abs(coefficient) > coefficient_cutoff:
                fermion_operator += FermionOperator(((p, 1), (q, 0)), coefficient)
            for r in range(n_spin):
                for s in range(n_spin):
                    coefficient = 0.5 * complex(two_spin[p, q, r, s])
                    if abs(coefficient) > coefficient_cutoff:
                        fermion_operator += FermionOperator(
                            ((p, 1), (q, 1), (r, 0), (s, 0)),
                            coefficient,
                        )
    return fermion_operator


def _chemistry_versions() -> dict[str, str]:
    return {
        package: version or "unknown"
        for package, version in package_versions(("openfermion", "pyscf")).items()
    }


__all__ = [
    "CHEMISTRY_EXTRA_MESSAGE",
    "GCIMChemistryProblemData",
    "build_gcim_chemistry_problem",
    "chemistry_reference_diagnostic",
    "chemistry_reference_report_section",
    "closed_shell_reference_occupations",
    "correlation_fraction",
    "qubit_operator_to_sparse_pauli",
]
