"""ADAPT processed inputs and the immutable results of the ADAPT iteration."""

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
from .fixed_basis import (
    GroupMoment,
    PairPreparation,
    PauliArrays,
    PauliTerm,
    ProjectedPencil,
    ResolvedAcquisition,
    SampledPairVariance,
    ShotAllocation,
)

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
    without bound inputs uses an ``adapt`` entry of its private cache.
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
    most ``limit + 1`` removed sets are kept. One run meets at most
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
    """One checked pool generator, with its raw and processed content hashes, terms and ``[H, A]``.

    ``commutator`` depends on this member's processed generator, the
    reconstruction's processed Hamiltonian, ``processing_source`` and the
    Pauli cutoff. The record asserts that relation. Portable validation does
    not recompute either ordered product. Only quantum Plans form it,
    because classical screening applies the generator to the state directly.

    Attributes:
        original_input: Content hash of the generator as supplied, before the
            anti-Hermitian projection.
        processed_input: Content hash of the processed Pauli generator that queries use.
        symmetry: Evidence of the anti-Hermitian projection (``AdaptSymmetry``).
        pool_index: Position in the checked pool.
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
    """Raw symmetry-projection evidence.

    No approximation is hidden as zero.
    A generator must be anti-Hermitian for ``exp(theta A)`` to be unitary and a
    Hamiltonian Hermitian for real energies. The input check projects each generator
    onto its anti-Hermitian part within ``input_symmetry_tolerance`` and records
    the correction, so the processed operator, not the raw input, is the
    declared scientific target. A Hamiltonian must already be exactly
    Hermitian, and its record shows a zero correction. The Frobenius norms
    of a Pauli sum are ``2**(q/2)`` times the coefficient 2-norm, because Pauli
    strings are orthogonal under the trace inner product.

    Attributes:
        target: ``Hermitian`` for a Hamiltonian, ``anti-Hermitian`` for a generator.
        projection: ``(H + H.adjoint)/2`` or ``(A - A.adjoint)/2``, matching ``target``.
        input_symmetry_tolerance: Largest relative Frobenius correction allowed, zero for
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
    """Bounded finite records and work formulas used by live execution and by report reload without Qiskit.

    It stores the processed Hamiltonian terms, pool members with their
    commutators, measurement groups and work formulas that every later query,
    analysis and verification reads. Report reload uses these records without
    importing Qiskit or rebuilding the pool. Only quantum Plans of the
    default growth store commutators, and only quantum Plans with shots
    store measurement groups.

    Attributes:
        schema_version: Record layout version.
        reference: Preparation record of the reference state.
        original_hamiltonian: Content hash of the Hamiltonian as supplied.
        processed_hamiltonian: Content hash of the processed Hamiltonian the queries use.
        terms: Kept Pauli terms of the processed Hamiltonian, including the
            identity term. Empty when a classical plan keeps a dense or
            sparse matrix.
        pool: Checked pool members in pool-index order.
        pool_name: Named pool family, or ``explicit`` for a supplied pool.
        max_selections: The longest chain: ``min(max_iterations, pool size)``,
            or ``BoundedRitzGrowth.max_chain_length`` for bounded growth.
        num_qubits: System register width.
        hamiltonian_action_work: Work of one Hamiltonian action from
            ``operators.inputs._matvec_requirements``. Its Pauli case uses
            ``operators._pauli.pauli_action_requirements``, which counts
            the grouped pass, a possible ordered retry and the 3L
            preprocessing visits (the AND, popcount and rotation scans that
            precede the grouped tiles), with ``g = min(L, N)`` as the upper
            bound on distinct flip masks, valid without inspecting the
            masks. The value is the one computed at planning under the
            archive format's work formula.
        hamiltonian_dropped_l1: Sum of the absolute Hamiltonian coefficients
            removed by ``pauli_coefficient_cutoff``, in the energy unit.
        groups: QWC measurement groups for sampled screening. Each group is a
            tuple of indices into the sorted distinct commutator labels of the
            pool (``screen_labels``). Empty unless the Plan samples.
        energy_groups: QWC measurement groups for sampled energy queries and
            grouped pencil settings, as indices into the sorted nonidentity
            Hamiltonian labels (``energy_labels``). Empty unless the Plan
            samples.
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
            setup scratch, from the same formula.
        acquisition: The requested and resolved pencil measurement choices
            (``ResolvedAcquisition``). Its real proof covers the processed
            Hamiltonian, the reference and every admitted pool generator, so
            every basis the run can build keeps the resolved quadratures. The
            proof is exact for ``processed_hamiltonian``, the target that the
            cutoff recorded in ``processing`` and ``hamiltonian_dropped_l1``
            defines, and says nothing about the imaginary parts of matrix
            elements of ``original_hamiltonian``. With a scalar processed
            Hamiltonian, quantum planning resolves the choices for bounded
            growth when ``max_basis_size>1``, because evaluating extensions
            can need overlaps. Default growth and bounded growth with
            ``max_basis_size=1`` measure nothing and resolve no choice.
    """

    schema_version: Literal[5] = 5
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
    acquisition: ResolvedAcquisition

    @model_validator(mode="after")
    def _one_identity_source(self):
        """A Pauli table carries its own identity term, so only a matrix plan sets ``host_identity_shift``."""
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
    ``{side}_count`` positions are active (``adapt._parameters``).
    """
    return tuple(
        (int(number(values[f"{side}_pool_{i}"])), float(number(values[f"{side}_theta_{i}"])))
        for i in range(int(number(values[f"{side}_count"])))
    )


def _exponential_action_work(member, theta, dimension):
    """Return the work counted for one classical normalized ``exp(theta A) @ vector``.

    With ``m = GENERATOR_TAYLOR_DEGREE = 18``, ``s = _taylor_steps(
    coefficients, theta)`` (including its zero and underflow branches) and
    ``W_A`` the shared packed Pauli action of the member's ``T`` stored rows
    (``adapt_actions.action_requirements``), the bound is

        W_exp = s [d + m (W_A + 2d)] + 10d.

    Each of the s steps first copies a vector, and each Horner term
    performs one Pauli action, one scaling and one addition. The final 10d is
    the normalization allowance in ``adapt_acquisition._vector`` and the
    zero-step copy. It remains conservative when a suffix is used without
    normalization. Packing is absent from ``W_A`` because it is counted once
    per Plan (``adapt_actions.build_action_tables``). The
    per-action group construction and rotated coefficients remain present on
    every action: prepacking masks does not cache flip-group dictionaries or
    rotated coefficients. This formula and ``adapt_actions.packed_exponential``
    both take the step count from ``fermionic_pool._taylor_steps`` with the
    member's stored coefficients, so the formula cannot miss a step at a
    rounding boundary. ADAPT's query and verification work formulas share
    this function.
    These are logical pass allowances, not CPU-time or accuracy bounds.
    """
    from nwqlib.subroutines.fermionic_pool import GENERATOR_TAYLOR_DEGREE, _taylor_steps
    from .adapt_actions import action_requirements

    steps = _taylor_steps((complex(c.real, c.imag) for _, c in member.terms), theta)
    action = action_requirements(dimension, len(member.terms))[1]
    return steps * (dimension + GENERATOR_TAYLOR_DEGREE * (action + 2 * dimension)) + 10 * dimension


def _energy_gradient_work(members, thetas, dimension, hamiltonian_work):
    """Return the work counted for one classical energy-and-gradient evaluation.

    The units are those of ``_exponential_action_work``: one amplitude of
    ``d = dimension`` visited by one elementwise pass or reduction.
    ``W_Aj`` is the shared packed Pauli action of generator j with ``T_j``
    stored rows (``adapt_actions.action_requirements``). ``W_H``
    (``hamiltonian_work``) is one Hamiltonian action. With
    the degree ``m = GENERATOR_TAYLOR_DEGREE`` and the step counts ``s_j`` of
    ``_taylor_steps``, each nonzero-step derivative Horner term makes two A
    actions, two elementwise operations for the value and four for the
    tangent, each step copies the value and tangent bases, and the routine
    starts with a copy and a zero fill:

        W_pair,j = 2d + s_j [2d + m (2 W_Aj + 6d)]   (s_j > 0),
                   d + W_Aj                          (s_j = 0, theta_j = 0),
                   2d                                (s_j = 0, theta_j != 0).

    The last branch differentiates the locally constant identity
    approximation caused by underflow. At zero angle the limiting tangent is
    ``A @ state``. The unnormalized reverse Taylor action has
    ``W_back,j = s_j [d + m (W_Aj + 2d)]`` for ``s_j > 0`` and ``d`` for
    ``s_j = 0``. Each forward factor normalization takes nine passes, each
    pullback a dot, scale, subtraction and two real-component divisions, five
    passes, then one tangent dot, and the final energy dot one pass. The
    driver does not propagate an unused adjoint past factor 1. Thus

        W_EG = W_H + d + sum_{j=1..k} (W_pair,j + 15d) + sum_{j=2..k} W_back,j.

    The generator terms count array passes and action calls, excluding
    scalar angle operations, Python calls and term-selector metadata visits.
    The Hamiltonian term uses its caller-supplied work formula. There are
    ``2m sum(s_j)`` forward generator actions, ``m sum_{j>=2} s_j`` backward
    actions, one more A action for each zero-angle factor with zero steps and
    one H action. Reference construction on a cache miss is added by the
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


class BoundedCandidate(Record):
    """One extension that a `BoundedRitzGrowth` round examined, with its reason and, when scored, its score.

    `AdaptRound.candidates` holds them in the order the round examined them
    ([`BoundedRitzGrowth`][nwqlib.algorithms.gcim.adapt.BoundedRitzGrowth]).
    The energy quantities are in the unit of the `Eigenproblem` and are
    formed without the identity coefficient c of the processed Hamiltonian
    `A = cI + A0`, which cancels from them. The fields below are read-only.

    Attributes:
        chain: The extension's `(pool index, theta)` items in the order they
            act, its parent chain followed by one generator.
        reason: `"accepted"` for the extension added to the basis,
            `"ranked"` for another scored one, `"below_independence_cutoff"`
            when `eta` was at most the independence cutoff,
            `"nonfinite_score"` when a score quantity was not finite,
            `"duplicate"` for a chain already in the basis,
            `"depth_unknown"` and `"above_depth_limit"` for an extension
            that `basis_depth_limit` excluded, and `"candidate"` for an
            extension chosen for scoring whose matrix elements the run had
            not finished obtaining when it stopped. With shots, an
            evaluated extension that the sampled eligibility rule of
            [`BoundedRitzGrowth`][nwqlib.algorithms.gcim.adapt.BoundedRitzGrowth]
            does not pass has `"screening_noise_unknown"` when an overlap
            setting has fewer than two shots or no finite nonnegative
            variance, or when one set of counts serves two different overlap
            entries, `"screening_metric_unresolved"` when
            `metric_epsilon` is at least 1/2 or a needed quantity is not
            finite, and `"screening_independence_unresolved"` when
            `eta_margin` is at most `independence_cutoff`. The last one
            does not show that the extension depends on the basis states.
        depth_bound: Under `basis_depth_limit`, the extension's CX count
            bound, the reference's count plus that of every generator
            occurrence, or None when it is unknown, no limit is set or the
            chain is already in the basis.
        depth_reason: Why `depth_bound` is unknown, or None.
        eta: Squared norm of the extension's part outside the kept overlap
            subspace of the current basis, or None when it was not computed.
            With shots it is the estimate from the sampled overlaps, which
            can be negative and is not clipped.
        coupling: `rho = |<q|A - E|psi>|` for that part q, normalized, and
            the current lowest Ritz state psi and value E, or None.
        gap: `delta = <q|A|q> - E`, formed from `A0` and the offset-free Ritz
            value, or None.
        score: `(sqrt(delta**2 + 4 rho**2) - delta)/2`, a lower bound on the
            energy decrease that adding q to the kept subspace gives, or
            None. It is not a prediction of the next solve's energy. With
            shots it is computed from sampled matrix elements, and the
            sampled eligibility rule does not control the noise of the
            Hamiltonian elements or of the Ritz pair, so it is a raw
            estimate, not a validated energy decrease.
        independence_cutoff: With shots, the floor that `eta_margin` must
            exceed, which is `BoundedRitzGrowth.independence_cutoff` or, when
            that is None, ADAPT's `overlap_cutoff`. Without shots it is None,
            as are the fields below.
        multiplier: With shots, the fixed multiplier k = 3 of the sampled
            eligibility rule.
        metric_beta: With shots, `beta_S = k sqrt(nu_S)`, where `nu_S` is
            the largest sum over one row of the current overlap matrix of
            the variance proxies of its off-diagonal entries. Each measured
            real or imaginary part contributes the larger of its sample-mean
            variance and `1/(N + 1)**2` for its N shots. The diagonal and
            an imaginary part that real-only measurement leaves out
            contribute nothing.
        metric_epsilon: With shots, `beta_S` divided by the smallest overlap
            eigenvalue that the current solve kept. The extension is scored
            only when it is below 1/2.
        eta_sigma: With shots, the first-order spread proxy of `eta` from
            the variance proxies of the current overlap entries and of the
            extension's overlaps.
        extension_noise: With shots, `b_s`, the sum over the extension's
            overlaps of their real and imaginary variance proxies, each
            weighted by the squared norm of the matching row of T, the matrix
            that maps coordinates in the kept subspace of the current solve
            to the basis with `T^dagger S T = I`.
        eta_resolution: With shots, `R = k eta_sigma + (d + metric_epsilon
            x)**2/(1 - metric_epsilon)` with `d = k sqrt(extension_noise)`
            and x the norm of the extension's coordinates in the kept
            subspace. The second term bounds the change of `eta` beyond
            first order when the norm conditions in the GCiM guide hold, and
            its inputs here are variance proxies. It is a conditional
            empirical resolution, not a confidence bound.
        eta_margin: With shots, `eta - eta_resolution`. The extension is
            scored only when it exceeds `independence_cutoff`.
        overlap_cutoff: With shots, the overlap cutoff of the current solve.
        kept_rank: With shots, the number of overlap directions that the
            current solve kept.
    """

    chain: tuple[tuple[Count, Real], ...]
    reason: Literal["accepted", "ranked", "below_independence_cutoff", "nonfinite_score", "duplicate",
                    "depth_unknown", "above_depth_limit", "candidate", "screening_noise_unknown",
                    "screening_metric_unresolved", "screening_independence_unresolved"]
    depth_bound: Count | None = None
    depth_reason: Text | None = None
    eta: Real | None = None
    coupling: Annotated[Real, Field(ge=0)] | None = None
    gap: Real | None = None
    score: Annotated[Real, Field(ge=0)] | None = None
    independence_cutoff: Annotated[Real, Field(gt=0, lt=1)] | None = None
    multiplier: PositiveInt | None = None
    metric_beta: Annotated[Real, Field(ge=0)] | None = None
    metric_epsilon: Annotated[Real, Field(ge=0)] | None = None
    eta_sigma: Annotated[Real, Field(ge=0)] | None = None
    extension_noise: Annotated[Real, Field(ge=0)] | None = None
    eta_resolution: Annotated[Real, Field(ge=0)] | None = None
    eta_margin: Real | None = None
    overlap_cutoff: Annotated[Real, Field(gt=0)] | None = None
    kept_rank: PositiveInt | None = None


class AdaptRound(Record):
    """One projected analysis of an `ADAPT` run and the selection that followed it.

    `ADAPTResult.history` holds one per completed projected analysis. A round
    keeps the values that drove its decision, so a later reanalysis of the
    Result at another cutoff does not change it. With `BoundedRitzGrowth`
    a round's basis is the reference and the chains that earlier rounds
    added (`ADAPTResult.basis`), `selected`, `theta` and `gradients` are
    empty and `winner` is None, and `candidates` records the extensions the
    round examined. The fields below are read-only.

    Attributes:
        iteration: Number of selected generators in the basis of this
            round, or with `BoundedRitzGrowth` the number of chains added
            before it.
        selected: Pool indices of those generators in selection order.
        theta: Their angles at this analysis, in radians.
        energy: Lowest projected Ritz value of this round, or None.
        energy_decrease: The previous round's `energy` minus this one, or
            None for the first round or when either is unavailable, in both
            growth modes. It is the actual change of the solved value,
            negative when the energy rose, for example because the overlap
            cutoff kept other directions of the enlarged basis or sampling
            noise moved the estimate.
        kept_rank: Number of overlap directions that this round's solve
            kept above the overlap cutoff, the rank of its projected
            problem, in both growth modes, and 0 when the solve kept none.
        gradient_norm: Euclidean norm of the screened product-state
            gradients, or None when no screening followed.
        gradients: `(pool_index, <[H, A_i]>)` pairs for the unselected
            generators.
        winner: Pool index selected next, or None when no screening followed
            or the gradient-norm floor stopped the run.
        residual_norm: Full-space residual `||(H - E) psi||` of the normalized
            Ritz state. It is recorded only under `"residual_norm"` stopping
            and is None otherwise.
        flat_count: Consecutive small energy changes counted after this round.
        contribution_ids: Content hashes of the data first collected after
            the previous round's screening and up to this round's screening,
            including any optimizer energy queries in between.
        basis_realization: Content hash of a query that identifies the full
            product state of this round's basis, or None. It is the shared
            query on the full product state in an exact quantum run, whose
            screening values the next screening reuses, the diagonal pair
            query in a classical run, the first-group diagonal setting of the
            full product in a grouped sampled run, or the query of the first
            nonidentity Pauli term of the diagonal pair in a termwise sampled
            run. With `BoundedRitzGrowth` it is the diagonal pair query of
            the chain added last. The matrix stage of the round, which
            obtains every matrix element of the current basis, must contain
            this query. The value is None before any generator is selected
            and in an exact quantum run whose Hamiltonian has no nonidentity
            term.
        analysis_attempt: Analysis attempt of the run that produced this round.
        optimizer_attempts: Cumulative optimizer invocations at this round.
        optimizer_rounds: Cumulative completed BFGS iterations at this round.
        energy_queries: Cumulative energy queries counted against
            `optimize_max_evaluations`.
        sampled_failure: Sampled-pencil failure or enclosure excursion code,
            or None.
        sampled_enclosure_violation: Distance of the sampled Ritz value
            outside the Pauli L1 enclosure of the processed operator, or None.
        candidates: With `BoundedRitzGrowth`, the
            [`BoundedCandidate`][nwqlib.algorithms.gcim.adapt_records.BoundedCandidate]
            extensions that the round after this analysis examined, in
            examination order, the added one marked `"accepted"`. Empty
            otherwise.
    """

    schema_version: Literal[2] = 2
    iteration: Count
    selected: tuple[Count, ...]
    theta: tuple[Real, ...]
    energy: Real | None
    energy_decrease: Real | None = None
    kept_rank: Count
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
    candidates: tuple[BoundedCandidate, ...] = ()


class ADAPTResult(Result):
    """Smallest projected eigenvalue from an `ADAPT` run, with its history and stop reason.

    [`solve`][nwqlib.scientist.solve] returns it for an `ADAPT` method, and
    `load_result` reopens a saved one. The answer is `eigenvalue`, the lowest
    projected Ritz value of the processed Hamiltonian in the basis given by
    `value_selected` and `value_theta`, in the unit of the `Eigenproblem`. It
    establishes neither the ground state nor coverage of the adaptive search.
    `stop_reason` says why the run ended, `history` holds one round per
    projected analysis, and `pencil` holds the projected matrices. `selected`
    and `theta` describe the latest generators and angles, while
    `value_selected` and `value_theta` describe the basis behind `eigenvalue`
    and `pencil`. They differ when a generator was selected or optimized
    after the last projection, and reanalysis always uses the value basis.
    With [`BoundedRitzGrowth`][nwqlib.algorithms.gcim.adapt.BoundedRitzGrowth]
    the explicit chains `basis` and `value_basis` take their place, and
    `selected`, `theta`, `value_selected` and `value_theta` are empty.
    `print(result)` shows the value, its scope and the stop reason. The
    fields below are read-only. The fields of
    [`Result`][nwqlib.core.analysis.Result] are present too.

    Attributes:
        eigenvalue: Lowest projected Ritz value of the processed Hamiltonian,
            or None. It establishes neither the ground state nor coverage of
            the adaptive search.
        selected: Pool indices chosen so far, in selection order.
        theta: Current angles of those generators, in radians.
        value_selected: Pool indices of the basis behind `eigenvalue`.
        value_theta: Angles of the basis behind `eigenvalue`.
        history: One [`AdaptRound`][nwqlib.algorithms.gcim.adapt_records.AdaptRound]
            per completed projected analysis.
        pencil: The [`ProjectedPencil`][nwqlib.algorithms.gcim.fixed_basis.ProjectedPencil]
            of the value basis, or None.
        stop_reason: Why the run stopped. Scientific stops are
            `gradient_norm_floor`, `flat_counter`, `residual_norm_threshold`,
            `pool_exhausted`, `max_iterations`,
            `optimization_evaluation_budget_exhausted`,
            `no_usable_overlap_subspace` and `invalid_sampled_evidence`.
            With `BoundedRitzGrowth` the scientific stops are
            `max_basis_size`, `max_iterations`, `flat_counter`,
            `basis_depth_limit`, `parent_window_exhausted`,
            `candidate_scans_exhausted`, `no_admissible_candidate`,
            `noise_inconclusive` (with shots, no evaluated extension of the
            last round passed the sampled eligibility rule, so the growth
            ends on its current basis), `no_usable_overlap_subspace` without
            shots and `invalid_sampled_evidence` with shots, all operational
            limits rather than evidence of convergence. When a round finds
            no extension to evaluate, `basis_depth_limit` takes precedence over
            `parent_window_exhausted`, and that over
            `candidate_scans_exhausted`, each describing only the examined
            extensions ([`BoundedRitzGrowth`][nwqlib.algorithms.gcim.adapt.BoundedRitzGrowth]).
            Execution states are `running`, `cancelled`, `not_started`,
            `partial_observation` (data that a decision needs are missing or
            unusable), `pending_preparation` and `pending_acquisition` (a
            circuit preparation or a backend job has not finished),
            `uncertain_preparation` (an upload or compilation may or may not
            have completed), `uncertain_native_intent` (a recorded backend
            submission may or may not have reached the backend),
            `backend_failed` and `native_depth_refused` (the next circuit's
            native two-qubit depth exceeds `ADAPT.max_native_two_qubit_depth`
            or is unavailable, so it was not submitted, and `pending` holds
            the refusal, which names the circuit's experiment and preparation
            record, the limit and the depth or why it is unavailable).
        pending: Labels of queries or statuses that blocked further progress.
        optimizer_attempts: Number of optimizer invocations started.
        optimizer_rounds: Completed BFGS iterations over all invocations.
        optimizer_restarts: Records of BFGS restarts from the saved best
            point after the run was resumed.
        energy_queries: Energy queries counted against
            `optimize_max_evaluations`.
        analysis_attempts: Projected analyses started by the run, the pilot
            analysis of `ADAPT.sampling` included.
        analysis_cutoff: Overlap-eigenvalue cutoff of this analysis.
        group_moments: For a grouped run with shots, the moments of every
            count measurement of the value basis's pencil
            ([`GroupMoment`][nwqlib.algorithms.gcim.fixed_basis.GroupMoment]),
            at the positions of that basis, as in `FixedGCIMResult`. The
            first group's overlap moments and their covariance with H0
            belong to the same counts. Empty otherwise.
        sampled_pair_variances: For a grouped run with shots, the pooled
            sample-mean variances and overlap–H0 covariance of each pair of
            the value basis
            ([`SampledPairVariance`][nwqlib.algorithms.gcim.fixed_basis.SampledPairVariance]).
            Empty otherwise.
        pair_preparations: For a value basis whose pencil entries come from
            joint states, computed exactly by quantum execution without shots
            or measured with grouped shots, the construction of the joint
            state of each off-diagonal pair
            ([`PairPreparation`][nwqlib.algorithms.gcim.fixed_basis.PairPreparation]).
            Under `pair_preparation="vacuum_reference"` the pairs of the
            reference have the route `"vacuum_reference"` and every other
            pair the `"prefix"` route with its shared blocks. Under
            `"vacuum_restore"` every pair, those of the reference included,
            has the route `"vacuum_restore"`, whose circuit for the
            reference is that of `"vacuum_reference"`. Each pair's vacuum
            phase correction is determined by its recorded route, the pair's
            generator chains and the Plan's pool, through the generator
            circuits built for those chains or, after the Run is reopened,
            the generator circuits restored with it.
            Exact execution prepares joint states also when
            `matrix_measurement="termwise"` is requested, because that choice
            applies only to shots. Termwise measurement with shots prepares
            no joint state, because its Hadamard tests place each Pauli term
            between the two controlled preparations, so the field is then
            empty. It is also empty for classical execution, for a Plan that
            measures nothing and when `pencil` is None. A one-state value
            basis has no off-diagonal pair.
        basis: With `BoundedRitzGrowth`, the chains of the basis so far in
            basis order, the reference `()` first. Each chain is a tuple of
            `(pool index, theta)` items in the order they act. Empty
            otherwise.
        value_basis: With `BoundedRitzGrowth`, the chains of the basis
            behind `eigenvalue` and `pencil`, the first chains of `basis`.
            Empty otherwise.
        frontier_cursor: With `BoundedRitzGrowth`, `(parent position, next
            pool index)` of each parent chain of the last examination, with
            positions in `basis`. The pairs are in cyclic position order
            ending with the parent of the last examined extension, after
            which a later round continues the rotation. Empty otherwise.
        shot_allocation: With `ADAPT.sampling`, the
            [`ShotAllocation`][nwqlib.algorithms.gcim.fixed_basis.ShotAllocation]
            of the pilot and main stages that measure the value basis again
            after its growth has stopped, decided once before the first
            main-stage measurement, with the shots of every grouped setting
            of that basis in its setting order, by pair, quadrature and
            group. When that basis has no setting to measure, both stages
            have zero shots, the per-setting lists are empty and
            `effective_fraction` is `"not_applicable"`, while the shots of
            the growth stay spent. Once the main stage has been analyzed,
            `eigenvalue`, `pencil`, `group_moments`,
            `sampled_pair_variances` and `contribution_ids` come from the
            main-stage measurements alone, `group_moments` in the setting
            order of the decision, while `history` keeps the analyses that
            grew the basis. Loading the Result and
            `result.analyze(...)` read the decision as saved. None without
            `sampling` and before the decision.
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
    group_moments: tuple[GroupMoment, ...] = ()
    sampled_pair_variances: tuple[SampledPairVariance, ...] = ()
    pair_preparations: tuple[PairPreparation, ...] = ()
    basis: tuple[tuple[tuple[Count, Real], ...], ...] = ()
    value_basis: tuple[tuple[tuple[Count, Real], ...], ...] = ()
    frontier_cursor: tuple[tuple[Count, Count], ...] = ()
    shot_allocation: ShotAllocation | None = None
    target_identification: Text = (
        "processed-input projected estimate; ground identity and adaptive coverage not established"
    )

    def _summary_lines(self):
        lines = [
            f"Ritz eigenvalue: {self._scalar_text(self.eigenvalue)}{self._unit_text()}",
            self.target_identification,
            f"ADAPT stop: {self.stop_reason}; "
            + (f"{len(self.basis)} basis chains" if self.basis else f"{len(self.selected)} selected generators"),
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
        acquisition = plan.reconstruction.acquisition
        if (self.group_moments or self.sampled_pair_variances) and acquisition.matrix_measurement != "grouped":
            raise ValueError("ADAPT group statistics belong only to a grouped Plan with shots")
        if self.pair_preparations and (
            acquisition.pair_preparation is None or acquisition.matrix_measurement == "termwise"
        ):
            raise ValueError("ADAPT pair preparations belong only to the joint states of exact "
                             "or grouped quantum measurement")
        if self.shot_allocation is not None and self.shot_allocation.sampling != plan.method.sampling:
            raise ValueError("ADAPT shot allocation belongs to another Plan's PencilSampling")
        growth = plan.method.basis_growth
        if growth is not None:
            # A bounded basis is the explicit chain list, not a function of
            # one selected chain, and the value basis is its first chains.
            if (self.selected or self.value_selected or not self.basis or self.basis[0] != ()
                    or self.value_basis != self.basis[:len(self.value_basis)]
                    or any(len(chain) > growth.max_chain_length or index >= len(plan.reconstruction.pool)
                           for chain in self.basis for index, _ in chain)):
                raise ValueError("ADAPT bounded basis differs from its admitted pool and chain limit")
        elif self.basis or self.value_basis or self.frontier_cursor:
            raise ValueError("ADAPT explicit chains belong only to BoundedRitzGrowth")
        if self.pencil is not None:
            from .adapt_acquisition import basis_chains

            size = (len(self.value_basis) if growth is not None
                    else len(basis_chains(tuple(zip(self.value_selected, self.value_theta, strict=True)))))
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
        """Return the stored Gram matrix and solve diagnostics for projected checks.

        It reads the stored `pencil` and runs no new solve.
        `result.verify(checks=ProjectedVerificationOptions(...))` uses it.

        Returns:
            diagnostics (ProjectedDiagnostics): Its `overlap`, `spectrum`,
                `normalization` and `backward_error` are the pencil's
                `overlap`, `overlap_eigenvalues`, `overlap_normalization_error`
                and `projected_backward_error`, the last on the physical
                pencil (H, S). All four are None when there is no pencil.
        """
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
