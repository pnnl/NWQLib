"""Chebyshev moment reconstruction and scientific results."""

from typing import Annotated, Literal
from pydantic import Field
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Plan
from nwqlib.core.records import ContentID, InputRef, Nonnegative, PositiveInt, Real, Record, Text
from nwqlib.problems.inputs import PhysicalScale


class MomentSetting(Record):
    """degree, experiment and point bind one signed parity or positive reflection readout.

    Degree k is read after walk_steps=floor(k/2) walk steps (Kirby et al.,
    arXiv 2208.00567v4, Eq. (24)). readout is ``reflection`` for even k,
    ``select`` for odd k and ``classical`` for the host recurrence. qubits
    are the wires the readout uses, and histogram_width is the admitted bit
    width of a returned outcome key.

    Exact-probability settings share one acquisition: every exact setting
    names the one trajectory experiment and its own observation point, which
    reads the walk state after walk_steps steps through the even or odd
    view. Its qubits are the point's marginal and histogram_width their
    number. A sampled setting names its own experiment and no point, and its
    histogram_width is the full index and system width of a count outcome.
    Classical settings name their host scalar and no point.

    Attributes:
        degree: Moment degree k.
        experiment: Experiment that acquires the degree.
        point: Trajectory point ID of an exact setting, None for sampled and classical settings.
        walk_steps: floor(k/2).
        readout: ``reflection``, ``select`` or ``classical``.
        qubits: Wires the readout reads.
        histogram_width: Admitted bit width of a returned outcome key.
    """

    degree: PositiveInt
    experiment: Text
    point: Text | None = None
    walk_steps: int
    readout: Literal["select", "reflection", "classical"]
    qubits: tuple[int, ...]
    histogram_width: int


class LanczosReconstruction(Record):
    """Everything analysis needs to turn moments into physical Ritz values.

    The selected operator's packed Pauli table fixes the nonidentity term
    order. Center and alpha define H = center I + alpha K. Analysis derives
    each used SELECT address's coefficient sign and system-support mask from
    that table. Matrix-input plans use their stored affine frame without a
    Pauli readout table. Algebraic known moments are independent of acquired
    degree settings.

    A normalized Ritz value x maps to the physical value center+alpha*x.
    Padding contributes zero without postselection. The readout table is
    readout.LanczosReadout, derived from the table that operator names.

    Attributes:
        schema_version: Version of this record layout.
        center: Identity shift of the frame, in the operator's units.
        alpha: Scale of the frame, in the operator's units. Zero marks a scalar operator.
        krylov_dimension: Trial dimension m.
        num_system_qubits: System qubits n.
        num_index_qubits: SELECT index qubits, zero for at most one term.
        known: (degree, value) pairs of algebraically known moments, always including mu_0=1.
        settings: One MomentSetting per acquired degree, in ascending degree. Exact settings share one trajectory acquisition and name their points.
        subspace: Content identity of the selected ChebyshevSubspace.
        physical_scale: Scale of the supplied reference state before normalization.
        operator: Identity of the operator access whose moments are read. For Pauli input it names the packed Pauli table that fixes the SELECT address order.
        spectral_lower: Lower enclosure endpoint, center-alpha up to outward rounding, not a computed eigenvalue.
        spectral_upper: Upper enclosure endpoint, center+alpha up to outward rounding, not a computed eigenvalue.
        enclosure_source: ``pauli_l1`` for the centered coefficient L1 norm or ``scaled_gershgorin`` for scaled Gershgorin row intervals, or the equal column intervals of Hermitian CSC storage.
    """

    schema_version: Literal[4] = 4
    center: Real
    alpha: Real
    krylov_dimension: PositiveInt
    num_system_qubits: int
    num_index_qubits: int
    known: tuple[tuple[int, Real], ...]
    settings: tuple[MomentSetting, ...]
    subspace: InputRef
    physical_scale: PhysicalScale
    operator: InputRef | None = None
    spectral_lower: Real | None = None
    spectral_upper: Real | None = None
    enclosure_source: Literal["pauli_l1", "scaled_gershgorin"] = "pauli_l1"

    @property
    def known_moments(self):
        """Known moments as a dict from degree to value."""
        return dict(self.known)

    @property
    def moment_indices(self):
        """Acquired degrees in ascending order, aligned with settings."""
        return tuple(setting.degree for setting in self.settings)


class MomentStatistics(Record):
    """Sample mean and second moment of one acquired Chebyshev moment, with its shots.

    `LanczosResult.statistics` pairs each acquired degree with one. The
    property `variance` is the unbiased sample-mean variance
    `(second_moment - mean**2)/(shots - 1)`, an empirical value only, and
    `None` without shots or for one shot. The fields below are read-only.

    Attributes:
        mean: Mean of the per-shot values of the moment.
        second_moment: Mean of their squares.
        shots: Shots behind the mean, or `None` for a numerical marginal.
    """

    mean: Real
    second_moment: Real
    shots: PositiveInt | None

    @property
    def variance(self):
        if self.shots is None:
            return None
        return (self.second_moment - self.mean**2) / (self.shots - 1) if self.shots > 1 else None


class LanczosResult(Result):
    """Projected estimate of the smallest eigenvalue from a `Lanczos` run.

    [`solve`][nwqlib.scientist.solve] returns it for a `Lanczos` method, and
    `load_result` reopens a saved one. The answer is `eigenvalue`, the lowest
    physical Ritz value `center + alpha*x`, in the unit of the
    `Eigenproblem`. Here `center` and `alpha` shift and scale `A` so that
    `(A - center*I) / alpha` has its spectrum in [-1, 1] (Oumarou et al.,
    arXiv:2603.15552v1, Section 2, Eqs. (1) and (4)), and x is the lowest
    Ritz value of that rescaled operator. `eigenvalue` is `None` when no
    Ritz value is available, and `failure` then gives the reason. The value
    is a projected estimate with unresolved error sources, and it does not
    establish the smallest full-space eigenvalue. `print(result)` shows it
    with its cutoff and rank notes, and `result.analyze(...)` recomputes it
    from the same moments with other `overlap_*` settings. The fields below
    are read-only. The fields of [`Result`][nwqlib.core.analysis.Result] are
    present too.

    `result.plan.reconstruction` stores `center`, `alpha` and the trial
    dimension m as `krylov_dimension`, so `overlap` and `hamiltonian` are m
    by m, in raw Chebyshev coordinates of the trial basis rather than the
    full Hilbert space. Missing moments are `None`. Moments shared by several
    pencil entries make those entries correlated, and `moment_variances` are
    empirical values, not confidence bounds.

    Attributes:
        eigenvalue: Lowest physical Ritz value `center + alpha*x` of the
            pencil after the Gram eigenvalues at or below `cutoff` are
            discarded, or `None` when unavailable.
        eigenvalues: Physical Ritz values of the kept pencil in ascending order.
        coefficients: S-normalized lowest Ritz vector in raw Chebyshev coordinates.
        moments: `mu_0` to `mu_(2m-1)`, with `mu_k = <psi|T_k(K)|psi>` for the
            normalized initial state, used by the analysis, `None` where
            missing.
        moment_variances: Empirical sample-mean variance per moment, 0 for
            known moments and `None` when unavailable.
        hamiltonian: Physical projected matrix `center*S + alpha*K_proj`,
            where
            `K_proj_ij = (mu_(i+j+1) + mu_|i+j-1| + mu_|i-j+1| + mu_|i-j-1|)/4`
            is K projected onto the trial space, or `None` when not
            representable.
        overlap: Raw Chebyshev Gram matrix `S_ij = (mu_(i+j) + mu_|i-j|)/2`.
            Its diagonal need not equal one.
        kept_rank: Number of kept directions, the overlap eigenvalues above
            the cutoff.
        projected_backward_error: Dimensionless backward error
            `||K_proj c - x S c|| / ((||K_proj||_F + |x| ||S||_F) ||c||)` of
            the lowest pair (x, c) of the normalized pencil (K_proj, S) that
            the solve uses, with `K_proj = (H - center S) / alpha` and
            `x = (E - center) / alpha`. Adding `c_I I` to a Pauli input, or
            scaling it by a positive factor, leaves K_proj and this value
            unchanged in exact arithmetic. For a matrix input the Gershgorin
            enclosure that sets center and alpha is widened by a roundoff
            allowance proportional to `|center|`, so an offset `c_I` changes
            alpha and the value by a relative amount of order
            `eps |c_I| / alpha`. FixedGCIM and ADAPT
            state theirs on the physical pencil (H, S) instead, so the two
            values are not comparable.
        overlap_normalization_error: Defect in `c^dagger S c = 1` for the returned coefficients.
        overlap_spectrum: Eigenvalues of S before thresholding.
        cutoff: Overlap-eigenvalue cutoff that the analysis applied.
        cutoff_source: Rule that set the cutoff.
        gram_sampling_bound: `m*e` with `e = sqrt(2*log(2*r/delta)/n_min)`,
            where m is `krylov_dimension`, r counts the sampled moments that
            enter S, n_min is the smallest of their shot counts and delta is
            `analysis_failure_probability`. With probability at least
            `1 - delta` it bounds the spectral norm of the Gram sampling
            error, provided the shots are independent outcomes in [-1, 1] and
            the measurements are unbiased (Hoeffding (1963),
            doi:10.1080/01621459.1963.10500830, Theorem 2, Eq. (2.6), p. 16,
            made two-sided as in Eq. (1.4), p. 13, with a union bound over the
            r moments). [Proposition 12](../../mathematics.md#r12) derives
            `||Delta S||_2 <= ||Delta S||_F <= m*e` from the moment errors.
            It is independent of the cutoff policy. It does not bound the
            error of `eigenvalue`.
        empirical_gram_noise_frobenius_rms: Estimate, from sample variances,
            of the root-mean-square Frobenius norm of the same Gram sampling
            error, without a coverage claim. It does not bound the error of
            `eigenvalue`.
        failure: Reason no Ritz value is available, or None.
        missing: Moment degrees without data.
        statistics: Pooled per-degree moment statistics that entered the analysis.
        physical_scale: Physical scale of the supplied reference state.
        subspace: Reference, by content hash, to the Chebyshev trial subspace that the Plan chose.
        selected_construction_ids: Content hash of the circuit construction behind each entry of `contribution_ids`, in the same order.
        analysis_cutoff: Cutoff requested for this analysis, or None for the default rule.
        analysis_cutoff_policy: Empirical or confidence policy requested for this analysis.
        analysis_noise_multiplier: Empirical-noise multiplier used by this analysis.
        analysis_failure_probability: Tail probability used for this analysis's Gram sampling bound.
        target_identification: Statement that the value is a projected estimate, not an identified ground state.
    """

    schema_version: Literal[3] = 3
    eigenvalue: Real | None
    eigenvalues: tuple[Real, ...] | None
    coefficients: tuple[Real, ...] | None
    moments: tuple[Real | None, ...]
    moment_variances: tuple[Real | None, ...]
    hamiltonian: tuple[tuple[Real, ...], ...] | None
    overlap: tuple[tuple[Real, ...], ...] | None
    kept_rank: int | None
    projected_backward_error: Nonnegative | None
    overlap_normalization_error: Nonnegative | None
    overlap_spectrum: tuple[Real, ...] | None
    cutoff: Annotated[Real, Field(gt=0)] | None
    cutoff_source: Literal[
        "scalar_operator",
        "not_evaluated",
        "user_fixed",
        "deterministic_default",
        "hoeffding_gram_bound",
        "unavailable_sampling_population",
        "empirical_gram_rms",
        "unavailable_empirical_variance",
    ]
    gram_sampling_bound: Nonnegative | None = None
    empirical_gram_noise_frobenius_rms: Nonnegative | None = None
    failure: Text | None
    missing: tuple[int, ...]
    statistics: tuple[tuple[int, MomentStatistics], ...]
    physical_scale: PhysicalScale
    subspace: InputRef
    selected_construction_ids: tuple[ContentID, ...]

    def projected_diagnostics(self):
        """Return the stored Gram matrix and solve diagnostics for projected checks.

        It reads stored fields and runs no reconstruction or solve.
        `result.verify(checks=ProjectedVerificationOptions(...))` uses it.

        Returns:
            diagnostics (ProjectedDiagnostics): Its `overlap`, `spectrum`,
                `normalization` and `backward_error` are this Result's
                `overlap`, `overlap_spectrum`, `overlap_normalization_error`
                and `projected_backward_error`, in raw Chebyshev coordinates.
        """
        from nwqlib.evidence.verification import ProjectedDiagnostics

        return ProjectedDiagnostics(
            self.overlap,
            self.overlap_spectrum,
            self.overlap_normalization_error,
            self.projected_backward_error,
        )

    analysis_cutoff: Annotated[Real, Field(gt=0)] | None = None
    analysis_cutoff_policy: Literal["empirical", "confidence"] = "empirical"
    analysis_noise_multiplier: Annotated[Real, Field(gt=0)] = 1.0
    analysis_failure_probability: Annotated[Real, Field(gt=0, lt=1)] = 0.05
    target_identification: Text = (
        "projected Ritz estimate; smallest full-space eigenvalue not established"
    )

    def _summary_lines(self):
        """Return the printed summary: the Ritz value, its scope, and cutoff and rank notes."""
        lines = [
            f"Ritz eigenvalue: {self._scalar_text(self.eigenvalue)}{self._unit_text()}",
            self.target_identification,
        ]
        if self.missing:
            lines.append(f"Partial data: {len(self.missing)} missing moments")
        if self.failure:
            lines.append("Analysis unavailable: " + self.failure)
        if self.cutoff_source == "empirical_gram_rms":
            lines.append("Exploratory Gram noise cutoff. No sampling-confidence or Ritz-energy guarantee.")
        if self.gram_sampling_bound is not None:
            lines.append(f"Conditional Gram sampling bound: {self._scalar_text(self.gram_sampling_bound)} "
                         f"at failure probability {self.analysis_failure_probability:g}")
        if self.kept_rank is not None and self.overlap_spectrum is not None and self.kept_rank < len(self.overlap_spectrum):
            lines.append(f"Gram rank after {self.cutoff_source}: {self.kept_rank}/{len(self.overlap_spectrum)}")
        return lines

    def validate_plan(self, plan):
        """Check that a stored or reloaded Result agrees with its Plan, without a solve.

        The Result must name the Plan's construction and subspace. For a
        nonscalar operator its moments and variances must equal those rebuilt
        from its stored statistics and the Plan's known moments. Missing
        degrees must be exactly those without data. The eigenvalue must be the first stored Ritz value, and no
        Ritz value may exist with missing moments. The cutoff source, cutoff,
        Gram bound and empirical RMS must equal what reconstruct would choose
        on the same branch: the scalar operator, missing data, or a
        recomputation from stored populations in O(m) scalar work. No Gram
        matrix, eigensolve or state application is repeated. Raises
        ValueError on the first disagreement.
        """
        from .numerical import _moment_payload, _resolve_overlap_cutoff
        import numpy as np

        self._validate_common_plan(plan, Plan)
        rec = plan.reconstruction
        if self.construction_id != plan.construction.content_id or self.subspace != rec.subspace:
            raise ValueError(
                "Lanczos result differs from its actual selected construction/subspace"
            )
        m = rec.krylov_dimension
        statistics = dict(self.statistics)
        if len(statistics) != len(self.statistics):
            raise ValueError("Lanczos statistics repeat a moment degree")
        moments, variances = _moment_payload(rec, statistics)
        if rec.alpha != 0 and (self.moments != moments or self.moment_variances != variances):
            raise ValueError("Lanczos moments differ from their stored scalar populations")
        if len(self.moments) != 2 * m or self.missing != tuple(
            i for i, v in enumerate(self.moments) if v is None
        ):
            raise ValueError("Lanczos result differs from its original moment population")
        if self.eigenvalue is not None and (
            not self.eigenvalues or self.eigenvalue != self.eigenvalues[0]
        ):
            raise ValueError("eigenvalue differs from the reconstructed Ritz spectrum")
        if rec.alpha and self.missing and self.eigenvalue is not None:
            raise ValueError("missing moments cannot produce a Ritz estimate")
        # Follow reconstruct's branch order using this analysis's requested
        # cutoff. A later explicit analyze(overlap_cutoff=...) is lawful.
        if rec.alpha == 0:
            if self.eigenvalue != rec.center:
                raise ValueError("Lanczos identity eigenvalue differs from its exact constant")
            source, cutoff = "scalar_operator", self.analysis_cutoff
        elif self.missing:
            source, cutoff = "not_evaluated", self.analysis_cutoff
        else:
            # Recompute O(m) scalar evidence from its actual stored populations.
            # No Gram matrix, eigensolve or state application is repeated.
            sampled = plan.shots is not None
            known = rec.known_moments
            cutoff, evidence = _resolve_overlap_cutoff(self.analysis_cutoff, m, sampled=sampled,
                variances=np.array([np.nan if v is None else v for v in variances]) if sampled else None,
                moment_shots=[0 if k in known else statistics[k].shots for k in range(2*m)],
                failure_probability=self.analysis_failure_probability, policy=self.analysis_cutoff_policy,
                noise_multiplier=self.analysis_noise_multiplier)
            source = evidence["source"]
            if (self.gram_sampling_bound != evidence["gram_sampling_bound"]
                    or self.empirical_gram_noise_frobenius_rms != evidence["empirical_gram_noise_frobenius_rms"]):
                raise ValueError("Lanczos Gram noise evidence differs from its stored populations")
            if cutoff is None and not self.failure:
                raise ValueError("unavailable sampling cutoff requires an unavailable failed solve")
        if self.cutoff_source != source or self.cutoff != cutoff:
            raise ValueError("Lanczos cutoff/source differs from its actual analysis branch")
        if (rec.alpha == 0 or self.missing) and (
            self.gram_sampling_bound is not None or self.empirical_gram_noise_frobenius_rms is not None
        ):
            raise ValueError("unevaluated Gram noise cannot supply sampling evidence")

    def energy_endpoint(self):
        """Return this Result's EnergyEndpoint for verify_energy_shift.

        The endpoint carries the Ritz value, the identities of the original
        operator and reference preparation, the Chebyshev subspace rule and
        Krylov dimension, and the payload of the selected operator access.
        verify_energy_shift uses them to establish A_target-A_baseline=shift*I,
        unless that relation is asserted, before it compares two such values.
        """
        from nwqlib.evidence.energy_shift import EnergyEndpoint, _operator_payload

        plan = self.plan
        self.validate_plan(plan)
        operator = plan._native["operator"]
        return EnergyEndpoint(
            plan_id=self.plan_id,
            result_id=self.content_id,
            construction_id=self.construction_id,
            operator_id=plan.problem.A.reference.identity,
            basis=plan.problem.basis,
            unit=plan.output.frame(plan.problem).unit,
            preparations=(plan._native["reference"].preparation.content_id,),
            subspace_rule="chebyshev",
            order=plan.reconstruction.krylov_dimension,
            sector=plan.problem.sector,
            frame=plan.output.frame(plan.problem),
            **_operator_payload(operator, max_bytes=plan.method.max_bytes),
            value=self.eigenvalue,
            analyzer=None if self.origin is None else self.origin.analyzer,
        )
