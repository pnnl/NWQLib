"""Actual ADAPT processed inputs and immutable controller results."""

from __future__ import annotations
from math import fsum
from typing import Annotated, Literal
from pydantic import Field, model_validator
from nwqlib.core.records import (
    Record,
    InputRef,
    Complex128,
    PositiveInt,
    Real,
    Source,
    Text,
)
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Plan
from nwqlib.operators.access import Count
from nwqlib.problems.inputs import StatePreparationSpec
from nwqlib.ir.expressions import number
from .fixed_basis import PauliArrays, PauliTerm, ProjectedPencil

PROCESSING_SOURCE = Source(
    name="adapt.processed_inputs",
    version="2",
    domain="explicit symmetry and coefficient processing",
    reference="nwqlib.algorithms.gcim.adapt_inputs",
)


def label_cache(plan):
    """Return the dictionary that holds a Plan's decoded commutator rows and label sets.

    A bound Plan uses its ``AdaptInputs.cache``, which sampled circuit
    construction reads too, so each row is decoded once per Plan. A Plan
    without native inputs uses an ``adapt`` entry of its private cache.
    Neither is saved.
    """
    native = plan._native
    return native["inputs"].cache if native else plan._cache.setdefault("adapt", {})


def commutator_rows(pool, cache):
    """Return each member's ``[H, A]`` rows ``((label, coefficient), ...)``, decoded once.

    ``cache`` is the dictionary of ``label_cache`` or ``AdaptInputs.cache``.
    A member without a commutator, as in a classical Plan, gives an empty
    tuple.
    """
    if "commutators" not in cache:
        cache["commutators"] = tuple(
            () if member.commutator is None else member.commutator.rows() for member in pool
        )
    return cache["commutators"]


def screen_labels(pool, cache):
    """Return the sorted distinct commutator labels of the pool, which ``groups`` index."""
    if "screen_labels" not in cache:
        cache["screen_labels"] = tuple(
            sorted({label for rows in commutator_rows(pool, cache) for label, _ in rows})
        )
    return cache["screen_labels"]


def energy_labels(labels, cache):
    """Return the sorted nonidentity Hamiltonian ``labels``, which ``energy_groups`` index, once per ``cache``."""
    if "energy_labels" not in cache:
        cache["energy_labels"] = tuple(sorted(labels))
    return cache["energy_labels"]


def active_screen_labels(pool, cache, removed, limit):
    """Return the sorted commutator labels of the members not in ``removed``, and their set.

    The union is formed once per selected chain, keyed by the removed pool
    indices, so every query of a screening round and its reads share it. At
    most ``limit + 1`` removed sets are kept; one run meets at most
    ``max_selections + 1``. ``cache`` is not saved.
    """
    store = cache.setdefault("active_screen_labels", {})
    key = frozenset(removed)
    if key not in store:
        if len(store) > limit:
            store.clear()
        rows = commutator_rows(pool, cache)
        labels = tuple(
            sorted({label for index, terms in enumerate(rows) if index not in key for label, _ in terms})
        )
        store[key] = (labels, frozenset(labels))
    return store[key]


class AdaptPoolMember(Record):
    """One admitted pool generator: its raw and processed identities, terms and ``[H, A]``.

    ``commutator`` depends on this member's processed generator, the
    reconstruction's processed Hamiltonian, ``processing_source`` and the
    Pauli cutoff. The record asserts that relation. Portable validation does
    not recompute either ordered product. Only quantum Plans form it:
    classical screening applies the generator to the state directly.

    Attributes:
        original_input: Identity of the generator as supplied, before the
            anti-Hermitian projection.
        processed_input: Identity of the processed Pauli generator that queries use.
        symmetry: Evidence of the anti-Hermitian projection (``AdaptSymmetry``).
        pool_index: Position in the admitted pool.
        family: Pool family name. A supplied operator has family ``custom``.
        spatial_indices: Spatial orbitals of the generator, empty for a
            supplied operator.
        spin_orbital_indices: Spin orbitals of a reference-dependent member, or None.
        terms: Processed ``(label, coefficient)`` Pauli terms of ``A``, all
            coefficients purely imaginary.
        fermion_terms: ``(coefficient, ((mode, action), ...))`` fermionic terms
            of a ``FermionicGenerator``, empty for qubit-excitation families
            and supplied operators.
        commutator: Real Pauli terms of the Hermitian ``[H, A]`` whose expectation
            is the screening gradient, after the Pauli cutoff, as ``PauliArrays``.
            None when the Plan is classical and forms no commutator.
        commutator_dropped_l1: Sum of the absolute ``[H, A]`` coefficients removed
            by the cutoff, in the Hamiltonian's energy unit. Zero when no
            commutator is formed.
    """

    original_input: InputRef
    processed_input: InputRef
    symmetry: "AdaptSymmetry"
    pool_index: Count
    family: Text
    spatial_indices: tuple[Count, ...]
    spin_orbital_indices: tuple[Count, ...] | None
    terms: tuple[tuple[Text, Complex128], ...]
    fermion_terms: tuple[tuple[Complex128, tuple[tuple[Count, Literal[0, 1]], ...]], ...]
    commutator: PauliArrays | None = None
    commutator_dropped_l1: Annotated[Real, Field(ge=0)] = 0.0

    @model_validator(mode="after")
    def _member(self):
        """Require an anti-Hermitian Pauli processed input and a known original representation.

        A ``FermionicGenerator``, from a named or an explicit pool, has an
        original identity of representation ``fermionic_generator``. It must
        carry ``RAW_GENERATOR_SOURCE`` and an identity in the format that
        ``_generator_reference`` writes, ``sha256:`` followed by 64 lowercase
        hex digits. A supplied operator must be Pauli, dense, CSR or CSC data.
        """
        from nwqlib.subroutines.fermionic_pool import RAW_GENERATOR_SOURCE

        if self.commutator is None and self.commutator_dropped_l1 != 0:
            raise ValueError("a member without a commutator has no dropped commutator mass")
        if (
            self.symmetry.target != "anti-Hermitian"
            or self.processed_input.representation != "pauli"
        ):
            raise ValueError("pool member requires its anti-Hermitian processed Pauli relation")
        if self.original_input.representation == "fermionic_generator":
            identity = self.original_input.identity
            if (
                self.original_input.source != RAW_GENERATOR_SOURCE
                or not identity.startswith("sha256:")
                or len(identity) != 71
                or any(c not in "0123456789abcdef" for c in identity[7:])
            ):
                raise ValueError("raw generator reference requires the versioned finite encoding")
        elif self.original_input.representation not in ("pauli", "dense", "csr", "csc"):
            raise ValueError(
                "original generator must identify admitted matrix, Pauli or fermionic metadata"
            )
        return self


class AdaptSymmetry(Record):
    """Raw symmetry-projection evidence; no approximation is hidden as zero.

    A generator must be anti-Hermitian for ``exp(theta A)`` to be unitary and a
    Hamiltonian Hermitian for real energies. Admission projects each generator
    onto its anti-Hermitian part within ``input_symmetry_tolerance`` and records
    the correction, so the processed operator, not the raw input, is the
    declared scientific target. A Hamiltonian must already be exactly
    Hermitian, and its record shows a zero correction. The Frobenius norms
    of a Pauli sum are ``2**(q/2)`` times the coefficient 2-norm, because Pauli
    strings are orthogonal under the trace inner product.

    Attributes:
        target: ``Hermitian`` for a Hamiltonian, ``anti-Hermitian`` for a generator.
        projection: ``(H + H.adjoint)/2`` or ``(A - A.adjoint)/2``, matching ``target``.
        input_symmetry_tolerance: Relative Frobenius correction admitted, zero for
            exact symmetry.
        correction_applied: Whether the projection changed any coefficient.
        correction_component_max: Largest real or imaginary component of the
            correction, in the operator's units.
        correction_frobenius_norm: Frobenius norm of the correction, or None
            when it overflows binary64.
        correction_frobenius_norm_status: ``finite`` or ``overflow``.
        relative_frobenius_correction: Correction norm divided by the input
            norm, or None when that ratio underflows.
        relative_frobenius_correction_status: ``finite`` or ``underflow``.
        certificate_target: Fixed ``processed_operator``. Later evidence refers
            to the processed operator, not to the raw input.
    """

    target: Literal["Hermitian", "anti-Hermitian"]
    projection: Text
    input_symmetry_tolerance: Annotated[Real, Field(ge=0)]
    correction_applied: Annotated[bool, Field(strict=True)]
    correction_component_max: Annotated[Real, Field(ge=0)]
    correction_frobenius_norm: Annotated[Real, Field(ge=0)] | None
    correction_frobenius_norm_status: Literal["finite", "overflow"]
    relative_frobenius_correction: Annotated[Real, Field(ge=0)] | None
    relative_frobenius_correction_status: Literal["finite", "underflow"]
    certificate_target: Literal["processed_operator"]

    @model_validator(mode="after")
    def _domain(self):
        """Require the evidence fields to describe one consistent projection.

        The projection formula must match ``target``, each norm must be
        present exactly when its status is ``finite``, an applied correction
        must be nonzero and within a nonzero tolerance, and the absence of a
        correction must be recorded with exact zeros.
        """
        expected = "(H + H.adjoint)/2" if self.target == "Hermitian" else "(A - A.adjoint)/2"
        if self.projection != expected:
            raise ValueError("symmetry projection differs from its target")
        if (self.correction_frobenius_norm_status == "finite") != (
            self.correction_frobenius_norm is not None
        ):
            raise ValueError("correction norm availability differs from its status")
        if (self.relative_frobenius_correction_status == "finite") != (
            self.relative_frobenius_correction is not None
        ):
            raise ValueError("relative correction availability differs from its status")
        if self.correction_applied:
            if (
                self.correction_component_max <= 0
                or self.correction_frobenius_norm == 0
                or self.relative_frobenius_correction == 0
                or self.input_symmetry_tolerance == 0
                or self.relative_frobenius_correction is not None
                and self.relative_frobenius_correction > self.input_symmetry_tolerance
            ):
                raise ValueError(
                    "an applied symmetry correction must be nonzero and admitted by its tolerance"
                )
        elif (
            self.correction_component_max != 0
            or self.correction_frobenius_norm != 0
            or self.relative_frobenius_correction != 0
        ):
            raise ValueError("no symmetry correction requires exact zero correction evidence")
        return self


class AdaptReconstruction(Record):
    """Bounded finite laws used by live execution and SDK-free report reload.

    It stores the processed Hamiltonian terms, pool members with their
    commutators, readout groups and work laws that every later query,
    analysis and verification reads. Report reload uses these records without
    importing Qiskit or rebuilding the pool. Only quantum Plans store
    commutators, and only quantum Plans with shots store readout groups.

    Attributes:
        schema_version: Record layout version.
        reference: Preparation record of the reference state.
        original_hamiltonian: Identity of the Hamiltonian as supplied.
        processed_hamiltonian: Identity of the processed Hamiltonian the queries use.
        terms: Kept Pauli terms of the processed Hamiltonian, including the
            identity term. Empty when a classical plan keeps a dense or
            sparse matrix.
        pool: Admitted pool members in pool-index order.
        pool_name: Named pool family, or ``explicit`` for a supplied pool.
        max_selections: ``min(max_iterations, pool size)``, the longest chain.
        num_qubits: System register width.
        hamiltonian_action_work: Work of one Hamiltonian action from
            ``operators.inputs._matvec_requirements``. Its Pauli case uses
            ``operators._pauli.pauli_action_requirements``, which charges
            the grouped pass, a possible ordered retry and the 3L
            preprocessing visits (the AND, popcount and rotation scans that
            precede the grouped tiles), with ``g = min(L, N)`` as the upper
            bound on distinct flip masks, valid without inspecting the
            masks. The value is the one computed at planning under the
            archive format's law.
        hamiltonian_dropped_l1: Sum of the absolute Hamiltonian coefficients
            removed by ``pauli_coefficient_cutoff``, in the energy unit.
        groups: QWC readout groups for sampled screening. Each group is a
            tuple of indices into the sorted distinct commutator labels of the
            pool (``screen_labels``). Empty unless the Plan samples.
        energy_groups: QWC readout groups for sampled energy queries, as
            indices into the sorted nonidentity Hamiltonian labels
            (``energy_labels``). Empty unless the Plan samples.
        processing: Human-readable processing qualifiers.
        hamiltonian_symmetry: Symmetry evidence of the Hamiltonian.
        processing_source: Versioned source of the processing rules.
        reference_action_work: Known preparation work of the reference, or
            None when a supplied preparation's work is unknown.
        reference_action_bytes: Known bytes of that preparation.
        host_identity_shift: For a classical plan that keeps a dense or
            sparse matrix, the mean diagonal ``c = trace(H)/d`` that pair
            queries remove from H before they project, so their ``h``
            scalars are ``<left|H - cI|right>`` and the solve adds c back.
            Zero when ``terms`` holds the Hamiltonian, whose identity term
            is then c.
        offset_free_action_work: Work that removing c from the stored matrix
            adds to one pair query's H action, from
            ``fixed_basis._offset_free_requirements``.
        offset_free_action_bytes: Bytes of the shifted sparse operator and its
            setup scratch, from the same law.
    """

    schema_version: Literal[4] = 4
    reference: StatePreparationSpec
    original_hamiltonian: InputRef
    processed_hamiltonian: InputRef
    terms: tuple[PauliTerm, ...]
    pool: tuple[AdaptPoolMember, ...]
    pool_name: Text
    max_selections: PositiveInt
    num_qubits: PositiveInt
    hamiltonian_action_work: Count
    hamiltonian_dropped_l1: Annotated[Real, Field(ge=0)]
    groups: tuple[tuple[Count, ...], ...]
    energy_groups: tuple[tuple[Count, ...], ...]
    processing: tuple[Text, ...]
    hamiltonian_symmetry: AdaptSymmetry
    processing_source: Source
    reference_action_work: Count | None = None
    reference_action_bytes: Count = 0
    host_identity_shift: Real = 0.0
    offset_free_action_work: Count = 0
    offset_free_action_bytes: Count = 0

    @model_validator(mode="after")
    def _one_identity_source(self):
        """A Pauli table carries its own identity term, so only a matrix plan has a host shift."""
        if self.terms and self.host_identity_shift != 0:
            raise ValueError("a Pauli-table reconstruction takes c from its identity term")
        return self

    @property
    def identity_shift(self):
        """Identity coefficient c that the projected solve adds back to the Ritz values."""
        if not self.terms:
            return self.host_identity_shift
        return fsum(t.coefficient for t in self.terms if t.label == "I" * self.num_qubits)

    @property
    def nonidentity_terms(self):
        return tuple(t for t in self.terms if t.label != "I" * self.num_qubits)

    @property
    def nonidentity_coefficients(self):
        return tuple(t.coefficient for t in self.nonidentity_terms)


def chain(values, side):
    """Return the active chain ``((pool_index, theta), ...)`` of side ``l`` or ``r`` of a point.

    ``values`` maps parameter names to bound values. Only the first
    ``{side}_count`` slots are active (``adapt._parameters``).
    """
    return tuple(
        (int(number(values[f"{side}_pool_{i}"])), float(number(values[f"{side}_theta_{i}"])))
        for i in range(int(number(values[f"{side}_count"])))
    )


def _exponential_action_work(member, theta, dimension):
    """Return the work charged for one classical normalized ``exp(theta A) @ vector``.

    With ``m = GENERATOR_TAYLOR_DEGREE = 18``, ``s = _taylor_steps(
    coefficients, theta)`` (including its zero and underflow branches) and
    ``W_A`` the shared packed Pauli action of the member's ``T`` stored rows
    (``adapt_actions.action_requirements``), the envelope is

        W_exp = s [d + m (W_A + 2d)] + 10d.

    Each of the s steps first copies a vector, and each Horner term
    performs one Pauli action, one scaling and one addition. The final 10d is
    the normalization allowance in ``adapt_acquisition._vector`` and the
    zero-step copy; it remains conservative when a suffix is used without
    normalization. Packing is absent from ``W_A`` because it has its
    once-per-Plan charge (``adapt_actions.build_action_tables``). The
    per-action group construction and rotated coefficients remain present on
    every action: prepacking masks does not cache flip-group dictionaries or
    rotated coefficients. This law and ``adapt_actions.packed_exponential``
    both take the step count from ``fermionic_pool._taylor_steps`` with the
    member's stored coefficients, so the law cannot miss a step at a rounding
    boundary. ADAPT's query and verification work laws share this function.
    These are logical pass allowances, not CPU-time or accuracy bounds.
    """
    from nwqlib.subroutines.fermionic_pool import GENERATOR_TAYLOR_DEGREE, _taylor_steps
    from .adapt_actions import action_requirements

    steps = _taylor_steps((complex(c.real, c.imag) for _, c in member.terms), theta)
    action = action_requirements(dimension, len(member.terms))[1]
    return steps * (dimension + GENERATOR_TAYLOR_DEGREE * (action + 2 * dimension)) + 10 * dimension


def _energy_gradient_work(members, thetas, dimension, hamiltonian_work):
    """Return the work charged for one classical energy-and-gradient evaluation.

    The units are those of ``_exponential_action_work``: one amplitude of
    ``d = dimension`` visited by one elementwise pass or reduction.
    ``W_Aj`` is the shared packed Pauli action of generator j with ``T_j``
    stored rows (``adapt_actions.action_requirements``), substituted into
    the existing derivative graph without changing that graph. ``W_H``
    (``hamiltonian_work``) is one Hamiltonian action. With
    the degree ``m = GENERATOR_TAYLOR_DEGREE`` and the step counts ``s_j`` of
    ``_taylor_steps``, each nonzero-step derivative Horner term makes two A
    actions, two elementwise operations for the value and four for the
    tangent, each step copies the value and tangent bases, and the kernel
    starts with a copy and a zero fill:

        W_pair,j = 2d + s_j [2d + m (2 W_Aj + 6d)]   (s_j > 0),
                   d + W_Aj                          (s_j = 0, theta_j = 0),
                   2d                                (s_j = 0, theta_j != 0).

    The last branch differentiates the locally constant identity
    approximation caused by underflow; at zero angle the limiting tangent is
    ``A @ state``. The unnormalized reverse Taylor action has
    ``W_back,j = s_j [d + m (W_Aj + 2d)]`` for ``s_j > 0`` and ``d`` for
    ``s_j = 0``. Each forward factor normalization takes nine passes, each
    pullback a dot, scale, subtraction and two real-component divisions, five
    passes, then one tangent dot, and the final energy dot one pass. The
    driver does not propagate an unused adjoint past factor 1. Thus

        W_EG = W_H + d + sum_{j=1..k} (W_pair,j + 15d) + sum_{j=2..k} W_back,j.

    The generator charges count array passes and action kernels, excluding
    scalar angle operations, Python calls and term-selector metadata visits.
    The Hamiltonian charge uses its caller-supplied work law. There are
    ``2m sum(s_j)`` forward generator actions, ``m sum_{j>=2} s_j`` backward
    actions, one more A action for each zero-angle factor with zero steps and
    one H action. Reference materialization on a cache miss is added by the
    caller. Returns ``(work, steps)`` with the precomputed step counts.
    """
    from nwqlib.subroutines.fermionic_pool import GENERATOR_TAYLOR_DEGREE, _taylor_steps
    from .adapt_actions import action_requirements

    m, d = GENERATOR_TAYLOR_DEGREE, dimension
    steps = [
        _taylor_steps((complex(c.real, c.imag) for _, c in member.terms), theta)
        for member, theta in zip(members, thetas, strict=True)
    ]
    work = hamiltonian_work + d
    for j, (member, theta, s) in enumerate(zip(members, thetas, steps, strict=True)):
        action = action_requirements(d, len(member.terms))[1]
        if s > 0:
            work += 2 * d + s * (2 * d + m * (2 * action + 6 * d))
        elif theta == 0:
            work += d + action
        else:
            work += 2 * d
        work += 15 * d
        if j:
            work += s * (d + m * (action + 2 * d)) if s > 0 else d
    return work, tuple(steps)


class AdaptRound(Record):
    """One projected analysis and the adaptive decision that followed it.

    Rounds record the values that actually drove each decision, so a later
    cutoff reanalysis of the Result cannot rewrite them.

    Attributes:
        iteration: Number of selected generators in the basis of this round.
        selected: Pool indices of those generators in selection order.
        theta: Their angles at this analysis, in radians.
        energy: Lowest projected Ritz value of this round, or None.
        gradient_norm: Euclidean norm of the screened product-state gradients, or
            None when no screening followed.
        gradients: ``(pool_index, <[H, A_i]>)`` pairs for the unselected generators.
        winner: Pool index selected next, or None when no screening followed or
            the gradient-norm floor stopped the controller.
        residual_norm: Full-space residual ``||(H - E) psi||`` of the normalized Ritz
            state. It is recorded only under ``residual_norm`` stopping and is None
            otherwise.
        flat_count: Consecutive small energy changes counted after this round.
        contribution_ids: Observation chunks first collected after the previous
            round's screening and up to this round's screening, including any
            optimizer energy queries in between.
        basis_realization: Realization identity of the query that supplied the
            diagonal of the full selected product, which the completed matrix
            stage must contain: the shared full-chain query of an exact quantum
            run, whose screening values the later screen reuses, or the diagonal
            pair query otherwise.
        analysis_attempt: Controller analysis attempt that produced this round.
        optimizer_attempts: Cumulative optimizer invocations at this round.
        optimizer_rounds: Cumulative completed BFGS iterations at this round.
        energy_queries: Cumulative logical energy queries charged to the optimizer
            allowance.
        sampled_failure: Sampled-pencil failure or enclosure excursion code, or None.
        sampled_enclosure_violation: Distance of the sampled Ritz value outside the
            processed Pauli enclosure, or None.
    """

    schema_version: Literal[2] = 2
    iteration: Count
    selected: tuple[Count, ...]
    theta: tuple[Real, ...]
    energy: Real | None
    gradient_norm: Real | None = None
    gradients: tuple[tuple[Count, Real], ...] = ()
    winner: Count | None = None
    residual_norm: Real | None = None
    flat_count: Count = 0
    contribution_ids: tuple[str, ...] = ()
    basis_realization: Text | None = None
    analysis_attempt: Count = 0
    optimizer_attempts: Count = 0
    optimizer_rounds: Count = 0
    energy_queries: Count = 0
    sampled_failure: Text | None = None
    sampled_enclosure_violation: Real | None = None


class ADAPTResult(Result):
    """Adaptive projected estimate, its trajectory and its controller outcome.

    ``selected``/``theta`` describe the controller's latest coordinates, while
    ``value_selected``/``value_theta`` describe the basis that produced
    ``eigenvalue`` and ``pencil``. They differ when a generator was selected
    or optimized after the last projection, and reanalysis always uses the
    value basis.

    Attributes:
        eigenvalue: Lowest projected Ritz value of the processed Hamiltonian, or
            None. It establishes neither ground identity nor adaptive coverage.
        selected: Pool indices chosen so far, in selection order.
        theta: Current angles of those generators, in radians.
        value_selected: Pool indices of the basis behind ``eigenvalue``.
        value_theta: Angles of the basis behind ``eigenvalue``.
        history: One ``AdaptRound`` per completed projected analysis.
        pencil: Projected pencil of the value basis, or None.
        stop_reason: Controller outcome. Scientific stops are ``gradient_norm_floor``,
            ``flat_counter``, ``residual_norm_threshold``, ``pool_exhausted``,
            ``max_iterations``, ``optimization_evaluation_budget_exhausted``,
            ``no_usable_overlap_subspace`` and ``invalid_sampled_evidence``.
            Execution states are ``running``, ``cancelled``, ``partial_observation``,
            ``pending_preparation``, ``pending_acquisition``, ``uncertain_preparation``,
            ``uncertain_native_intent``, ``backend_failed`` and ``not_started``.
        pending: Labels of queries or statuses that blocked further progress.
        optimizer_attempts: Number of optimizer invocations started.
        optimizer_rounds: Completed BFGS iterations over all invocations.
        optimizer_restarts: Records of BFGS restarts from a saved incumbent after resume.
        energy_queries: Logical energy queries charged to ``optimize_max_evaluations``.
        analysis_attempts: Projected analyses started by the controller.
        analysis_cutoff: Overlap-eigenvalue cutoff of this analysis.
        target_identification: Fixed statement of what the value does not establish.
    """

    eigenvalue: Real | None
    selected: tuple[Count, ...]
    theta: tuple[Real, ...]
    value_selected: tuple[Count, ...]
    value_theta: tuple[Real, ...]
    history: tuple[AdaptRound, ...]
    pencil: ProjectedPencil | None
    stop_reason: Text
    pending: tuple[Text, ...] = ()
    optimizer_attempts: Count = 0
    optimizer_rounds: Count = 0
    optimizer_restarts: tuple[dict, ...] = ()
    energy_queries: Count = 0
    analysis_attempts: Count = 0
    analysis_cutoff: Real
    target_identification: Text = (
        "processed-input projected estimate; ground identity and adaptive coverage not established"
    )

    def _summary_lines(self):
        lines = [
            f"Ritz eigenvalue: {self._scalar_text(self.eigenvalue)}{self._unit_text()}",
            self.target_identification,
            f"ADAPT stop: {self.stop_reason}; {len(self.selected)} selected generators",
        ]
        if self.pending:
            lines.append(f"Partial data: {len(self.pending)} pending queries")
        if self.pencil is not None:
            lines.extend(self.pencil._sampled_summary())
        return lines

    def validate_plan(self, plan):
        """Associate the reported value and pencil with the exact parameter sequence that
        produced them.
        """
        self._validate_common_plan(plan, Plan)
        if self.construction_id != plan.construction.content_id or len(self.selected) != len(
            self.theta
        ):
            raise ValueError(
                "ADAPT result differs from its actual selected construction/trajectory"
            )
        for selected, theta in (
            (self.selected, self.theta),
            (self.value_selected, self.value_theta),
            *((row.selected, row.theta) for row in self.history),
        ):
            if (
                len(selected) != len(theta)
                or len(set(selected)) != len(selected)
                or len(selected) > plan.reconstruction.max_selections
                or any(i >= len(plan.reconstruction.pool) for i in selected)
            ):
                raise ValueError("ADAPT selected sequence differs from its admitted pool")
        if self.pencil is not None and self.eigenvalue != (
            self.pencil.eigenvalues[0] if self.pencil.eigenvalues else None
        ):
            raise ValueError("ADAPT value differs from its actual projected pencil")
        if self.pencil is not None:
            from .adapt_acquisition import basis_chains

            size = len(basis_chains(tuple(zip(self.value_selected, self.value_theta, strict=True))))
            # An empty admitted overlap subspace has no Ritz vector. Its H/S
            # matrices still describe the complete attempted basis.
            coefficient_count = size if self.pencil.eigenvalues else 0
            if (
                len(self.pencil.hamiltonian) != size
                or len(self.pencil.overlap) != size
                or len(self.pencil.coefficients) != coefficient_count
                or self.pencil.overlap_cutoff != self.analysis_cutoff
            ):
                raise ValueError("ADAPT pencil differs from its original basis or analysis cutoff")
            self.pencil._validate_sampled_enclosure(plan.reconstruction,
                sampled=plan.execution == "quantum" and plan.shots is not None)

    def projected_diagnostics(self):
        from nwqlib.evidence.verification import ProjectedDiagnostics

        p = self.pencil
        return (
            ProjectedDiagnostics(None, None, None, None)
            if p is None
            else ProjectedDiagnostics(
                p.overlap,
                p.overlap_eigenvalues,
                p.overlap_normalization_error,
                p.projected_backward_error,
            )
        )
