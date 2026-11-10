"""QLS records of the chosen polynomial, the physical reconstruction on the original input, and the results."""

from math import sqrt
from typing import Annotated, Literal
from pydantic import Field
from nwqlib.algorithms.protocol import AlgorithmDescriptor
from nwqlib.artifacts import ArtifactManifest
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Plan
from nwqlib.core.records import (
    ContentID,
    FrozenArray,
    InputRef,
    Nonnegative,
    PositiveInt,
    Real,
    Record,
    Source,
    Text,
)
from nwqlib.execution import KernelApplication
from nwqlib.operators.access import Count
from nwqlib.problems.inputs import PhysicalScale
from nwqlib._validation import NUMERICAL_RELATION_RTOL
from nwqlib._quantum_readout import ReadoutSetting


METHOD = Source(
    name="qls",
    version="3",
    domain="original-coordinate Ax=b",
    reference=(
        "selected inverse Chebyshev QSVT and Dalzell arXiv:2406.12086v2 kernel-reflection methods"
    ),
)
DESCRIPTOR = AlgorithmDescriptor(
    method="qls",
    version="3",
    problem_families=("linear_system",),
    output_families=(
        "solution",
        "state_vector",
        "norm_squared",
        "quadratic_form",
        "normalized_expectation",
        "samples",
    ),
    access_families=("dense", "pauli", "periodic_stencil", "bound_block_encoding"),
    resource_coverage=("actual selected preparation, encoding, QSVT phase/projector/query body",),
    evidence_coverage=(
        "original spectral facts, selected polynomial and acquired original-coordinate output",
    ),
    limitations=(
        "complete propagated physical error and native rounding remain unknown",
        "shortcut supplies a unit direction; classical norm models do not supply physical x",
    ),
    references=(METHOD,),
    maintenance="NWQLib",
)


class InversePolynomial(Record):
    """Chosen odd inverse polynomial and the evidence of its relative residual.

    Attributes:
        kind: Discriminator, always ``"inverse"``.
        coefficients: Chebyshev coefficients of the relative minimax
            approximation ``P ~ 1/(polynomial_kappa x)``, before division by
            ``rescale``, with zero even entries.
        rescale: Positive divisor that puts ``P`` in the QSP amplitude
            domain. Physical recovery multiplies by it again.
        ideal_degree: Smallest odd degree whose minimax relative residual
            meets ``epsilon_inv``, decided in binary64. The table's degree
            is larger when its numerical check needed a refinement.
        minimax_error: Relative residual ``sech(m beta)`` of the exact
            minimax polynomial of the table's degree ``2m - 1``, with
            ``beta = 2 atanh(1/polynomial_kappa)``.
        residual_check_value: The value that planning compared with
            ``epsilon_inv``, ``minimax_error + 2F + R``, computed in binary64
            as ``(minimax_error + F) + (F + R)``. Planning uses the table
            only when it is at most ``epsilon_inv``. ``F`` is the binary64
            evaluation of a Chebyshev-zero bound on the largest difference
            between ``polynomial_kappa x P(x)`` for these coefficients and
            for the exact minimax polynomial, and ``R`` is the evaluation
            resolution, four machine epsilons times the bit length of the
            number of evaluation points. The second ``F`` and ``R`` allow
            for the error of that evaluation. The value has no arithmetic
            enclosure. In the cases checked it exceeds
            ``max |polynomial_kappa x P(x) - 1|`` over
            ``1/polynomial_kappa <= |x| <= 1``, but no proof shows that it
            always does
            ([Inverse polynomial accuracy](../../algorithms/qls.md#inverse-polynomial-accuracy)).
    """

    kind: Literal["inverse"] = "inverse"
    coefficients: tuple[Real, ...]
    rescale: Annotated[Real, Field(gt=0)]
    ideal_degree: PositiveInt
    minimax_error: Annotated[Real, Field(ge=0, lt=1)]
    residual_check_value: Annotated[Real, Field(ge=0, lt=1)]

    @property
    def degree(self):
        return len(self.coefficients) - 1


class ReflectionPolynomial(Record):
    """Chosen even kernel-reflection polynomial of a shortcut solver.

    Attributes:
        kind: Discriminator, always ``"reflection"``.
        coefficients: Chebyshev coefficients of Dalzell's ``K``
            (arXiv:2406.12086v2, App. B.3, Eq. (62)), before division by
            ``rescale``, with zero odd entries.
        rescale: Positive divisor that puts ``K`` in the QSP amplitude
            domain. The Eq. (17) mass window is divided by its square.
        ell: Half-degree from Dalzell Eq. (6). The degree is ``2 ell``.
        eta: Kernel-reflection parameter, ``epsilon_inv / sqrt(2)``.
    """

    kind: Literal["reflection"] = "reflection"
    coefficients: tuple[Real, ...]
    rescale: Annotated[Real, Field(gt=0)]
    ell: PositiveInt
    eta: Annotated[Real, Field(gt=0, le=1)]

    @property
    def degree(self):
        return 2 * self.ell


Polynomial = Annotated[InversePolynomial | ReflectionPolynomial, Field(discriminator="kind")]


class QLSWork(Record):
    """Operation counts and peak array bytes of one explicit classical QLS model evaluation.

    ``host_planning.selected_work`` computes these at planning time, from
    the dimensions, the chosen polynomial degree and the norm-model choice,
    after the polynomial is chosen and before the classical model runs.
    Planning compares ``size_units`` with ``max_work`` and
    ``workspace_bytes`` with ``max_bytes``. Search rows evaluate a
    classical probability model. Planned trials and queries describe the
    quantum work that this model plans, and no job is run. Quantum
    execution records only the two dimensions.

    Attributes:
        original_dimension: Dimension of the original equation.
        system_dimension: Dimension on which the polynomial acts, doubled by
            a Hermitian dilation.
        augmented_dimension: Dimension of Dalzell's augmented ``G_t``, or 0.
        dilation_dimension: Dimension of the Hermitian dilation of ``G_t``,
            or 0.
        target_solves: Encoded reference solves made by a norm model.
        svd_calls: Dense SVD calls of one evaluation, namely the SVD of
            ``G_t`` for ``shortcut_native_svp`` and the original SVD when
            the evaluation computes it because planning did not.
        eigh_calls: Dense Hermitian eigendecompositions of one evaluation,
            namely that of the dilation of ``G_t`` for ``shortcut_dilation``
            and the original eigensystem of a Hermitian ``A`` when the
            evaluation computes it because planning did not.
        matvecs: Matrix-vector products, namely the two basis changes of
            the chosen action, the row product ``b'† A_t`` of a shortcut and
            the coordinate matvec of the linear norm model.
        search_rows: Upper bound on the rows of the norm model (grid
            candidates, search rounds or sequence candidates).
        planned_trials: Upper bound on the norm model's planned trials.
        planned_queries: Upper bound on the norm model's planned queries.
        vector_reductions: Norm and inner-product reductions over vectors.
        observable_actions: Observable applications for scalar outputs.
        workspace_bytes: Largest peak bytes of the arrays held at once
            during the preparation, the numerical action and the output
            statistics, together with original factors that the evaluation
            computes, excluding the input data, factors that the Plan
            already holds and the internal workspace of external libraries.
        size_units: Scalar operation count of the work formula in
            ``selected_work``, compared with ``max_work``. It is not measured
            time.
    """

    original_dimension: PositiveInt
    system_dimension: PositiveInt
    augmented_dimension: Count = 0
    dilation_dimension: Count = 0
    target_solves: Count = 0
    svd_calls: Count = 0
    eigh_calls: Count = 0
    matvecs: Count = 0
    search_rows: Count = 0
    planned_trials: Count = 0
    planned_queries: Count = 0
    vector_reductions: Count = 0
    observable_actions: Count = 0
    workspace_bytes: Count = 0
    size_units: Count = 0


class QLSReconstruction(Record):
    """Spectral facts of the original A, and the padding and embedding that planning chose.

    kappa_be is the encoded inverse-gap parameter, computed from the
    smallest original singular value with a minimum of one, or supplied by
    the caller. polynomial_kappa is the parameter that the polynomial uses.
    It includes the domain floor and, for automatic periodic selection, an
    outward bound on the rounded encoded gap. The condition number of the
    original matrix never includes that floor or padding.

    Attributes:
        original_operator: Reference, with content hash, of the original A.
        original_rhs: Reference, with content hash, of the original physical b.
        original_dimension: Original equation dimension.
        padded_dimension: Coordinate dimension after padding.
        system_dimension: Dimension on which the chosen polynomial acts.
        embedding: ``hermitian_dilation`` for the Hermitian dilation that
            ``qsvt_inverse`` uses for a non-Hermitian A, otherwise ``none``.
        rhs_scale: Original RHS physical norm in binary-scale form.
        alpha: Normalization of the chosen encoding, strictly positive.
        alpha_source: Source of alpha, a computation or a supplied
            assumption.
        kappa_be: Encoded inverse-gap parameter. Automatic spectral
            selection uses ``max(1, alpha/sigma_min)`` with the original
            smallest singular endpoint. Automatic compact periodic
            selection uses ``alpha/m``, where m is the positive mass.
            Otherwise this is the accepted caller-supplied bound.
        polynomial_kappa: Parameter that the polynomial uses,
            including the domain floor ``1.01``. Outside automatic compact
            periodic selection it is ``max(kappa_be, 1.01)``. Automatic
            compact periodic selection also uses an upward-rounded
            inverse-gap parameter covering both the original spectrum
            and the rounded encoding.
        kappa_source: Source of the bound on the encoding domain, a
            computation or a supplied assumption.
        condition_number: Original sigma_max/sigma_min, if independently known.
        sigma_min: Original smallest singular endpoint, if known.
        sigma_max: Original largest singular endpoint, if known.
        spectral_method: Computation that gave the endpoints, or the scope
            of the supplied assumption.
        spectral_calls: Counts of the spectral work done during planning.
        polynomial: Coefficients of the chosen inverse or reflection
            polynomial and the evidence of its fit.
        phase_solution: Complete vector of the chosen QSVT phases, not a
            request for a new fit.
        phase_error: Coefficient one-norm of the difference between the
            polynomial rebuilt from the phases and ``polynomial/rescale``. In
            exact arithmetic it bounds ``sup |Re P_phases - polynomial/rescale|``
            over ``[-1, 1]``; its binary64 evaluation is a numerical
            estimate without an enclosure. None means unavailable.
        phase_tolerance: The limit that synthesis applied to ``phase_error``:
            the Method's ``phase_tol``, or its default rule
            (``quantum._phase_tolerance``). None for classical execution,
            which has no phase synthesis.
        phase_fft_lengths: FFT lengths that the canonical phase synthesis
            tried, in order; the last one gave ``phase_solution``.
        encoding_family: Family of the chosen encoding, if applicable.
        encoding_ancillas: Ancillas required by that encoding.
        encoding_error: Bound relative to the encoding's declared operator target.
        t: Chosen shortcut norm parameter, not the recovered physical
            solution norm.
        t_source: ``not_applicable`` for the inverse, ``user`` for a supplied
            numeric t, or the classical norm model ``grid``,
            ``noisy_binary_search`` or ``linear_kappa_sequence``.
        width: Width of the chosen circuit. Zero for a classical-only
            construction.
        coordinates: Ordered coordinate qubits.
        success: Algorithmic success conditions as (bit, value) pairs.
        conditions: Additional projection conditions as (bit, value) pairs.
        settings: Readout settings. An exact scalar output has
            the one ``projected_moments`` reduction. A sampled Pauli output
            has one counts setting per qubit-wise commuting group, labeled by
            the group's accumulated basis, plus the computational-basis
            ``physical_mass`` setting of a padded normalized output whose
            group bases all contain X or Y. A sampled NormSquared output
            has its ``physical_mass`` setting. Vector and Samples outputs
            have ``setting_0``.
        groups: Member Pauli labels of each group setting, in the order of
            the group settings and first-fit order within a group. Each
            distinct nonzero non-identity label of the observable is in one
            group.
        grouping_comparisons: Number of qubit-wise mask comparisons
            evaluated during grouping, counted against ``max_work``.
        observable_input: Reference to a finite Pauli observable of a scalar
            output. Analysis reads its labels and coefficients from the
            observable's packed term table, so the table is not copied into
            the reconstruction.
        observable_labels: Non-identity labels with nonzero coefficient of a
            dense observable's ``P O P`` Pauli table, in table order, or ().
        observable_coefficients: Real coefficients of those labels, in the
            same order, or None.
        observable_identity: Coefficient of the identity label in that dense
            table, zero when the table has none.
        recovery: Composed physical-output scale. Absent for unit-only shortcuts.
        work: Operation counts and peak array bytes of the classical
            model (``QLSWork``). A quantum Plan records only its dimensions.
    """

    original_operator: InputRef
    original_rhs: InputRef
    original_dimension: PositiveInt
    padded_dimension: PositiveInt
    system_dimension: PositiveInt
    embedding: Literal["none", "hermitian_dilation"]
    rhs_scale: PhysicalScale
    alpha: Annotated[Real, Field(gt=0)]
    alpha_source: Text
    kappa_be: Annotated[Real, Field(ge=1)]
    polynomial_kappa: Annotated[Real, Field(gt=1)]
    kappa_source: Text
    condition_number: Annotated[Real, Field(ge=1)] | None
    sigma_min: Annotated[Real, Field(gt=0)] | None
    sigma_max: Annotated[Real, Field(gt=0)] | None
    spectral_method: Text
    spectral_calls: tuple[tuple[Text, Count], ...] = ()
    polynomial: Polynomial
    phase_solution: tuple[Real, ...] = ()
    phase_error: Nonnegative | None = None
    phase_tolerance: Annotated[Real, Field(gt=0)] | None = None
    phase_fft_lengths: tuple[PositiveInt, ...] = ()
    encoding_family: Text | None = None
    encoding_ancillas: Count = 0
    encoding_error: Nonnegative | None = None
    t: Annotated[Real, Field(ge=1)] | None = None
    t_source: Literal[
        "not_applicable", "user", "grid", "noisy_binary_search", "linear_kappa_sequence"
    ]
    width: Count = 0
    coordinates: tuple[Count, ...] = ()
    success: tuple[tuple[Count, Count], ...] = ()
    conditions: tuple[tuple[Count, Count], ...] = ()
    settings: tuple[ReadoutSetting, ...] = ()
    groups: tuple[tuple[Text, ...], ...] = ()
    grouping_comparisons: Count = 0
    observable_input: InputRef | None = None
    observable_labels: tuple[Text, ...] = ()
    observable_coefficients: FrozenArray | None = None
    observable_identity: Real = 0.0
    recovery: PhysicalScale | None = None
    work: QLSWork

    @property
    def degree(self):
        return self.polynomial.degree


class QLSSamples(Record):
    """Sampled output of a QLS run: the observed original coordinates and their counts, as two int64 arrays.

    `QLSAnalysis.samples` holds it for a `Samples` output. For m indices,
    analysis copies the two int64 arrays that the sample reduction returns
    into immutable storage one at a time, holding at most 24m data bytes
    while it does. The two stored arrays occupy 16m data bytes. These byte
    counts exclude input data held by the caller, workspace of earlier
    reductions and Python object overhead.
    Building the record does not check these bytes against a memory limit.

    Attributes:
        indices: Distinct original-coordinate indices below the original
            dimension, increasing. Padding coordinates are excluded.
        counts: Positive returned counts of those indices, each at most
            ``MAX_COUNT``.
    """

    indices: FrozenArray
    counts: FrozenArray

    @property
    def total(self):
        """Sum of the counts as a Python integer."""
        return int(self.counts.array.sum())


class QLSProjectedMoments(Record):
    """Saved statistics of the exact scalar readout of a QLS run (`shots=None`): the three masses and the projected moment.

    `QLSAnalysis.reduction` holds it.
    Each scalar is a canonical ``(mantissa, exponent)`` pair, in which zero
    is ``(0.0, 0)`` and a nonzero mantissa has absolute value in [1/2, 1).

    Attributes:
        kernel: Versioned name of the computation of the masses and the
            moment, such as `"pow2-mass-pauli-moment/1"`.
        complete_mass: Computed sum of squared amplitude magnitudes over the
            full saved state, including all ``populations[0]`` amplitudes.
        success_mass: Success-only mass ``p_alg``, computed as the sum of squared
            magnitudes of the ``populations[1]`` amplitudes whose success
            selectors match. This selection can include branches that fail the
            additional conditions and coordinates added by padding.
        physical_mass: Physical-slice mass p over the ``populations[2]``
            coordinates whose success and condition selectors match and whose
            original-coordinate index is below the original dimension.
        numerator: Projected moment ``q = v^dagger O v`` of the Pauli form
            of the observable on the physical slice v, not a moment over the
            whole success block. Zero for a NormSquared output.
        numerator_radius: Upper bound, rounded upward, on the absolute rounding
            error in ``numerator`` from its classical evaluation using the Pauli
            form of the observable and the computed physical slice. It assumes binary64
            round-to-nearest arithmetic with gradual underflow and finite
            intermediate results. Errors in the prepared state and in converting
            the original observable to its Pauli form are separate.
        lost_state_components: Number of initially nonzero real or imaginary
            parts of physical-slice amplitudes that round to zero when rescaled
            by a power of two for the moment calculation. Real and imaginary
            parts are counted separately. Default ``0``.
        lost_coefficients: Number of initially nonzero coefficients of
            non-identity Pauli terms that round to zero when rescaled by a power
            of two for the moment calculation. Default ``0``.
        populations: Numbers of amplitudes in the full saved state, the
            success-only selection and the physical slice, in that order. The
            last is the original dimension. These counts include zero amplitudes.
        contribution_id: Content hash of the `ObservationChunk` that holds
            the saved reduction.
    """

    kernel: Text
    complete_mass: tuple[Real, int]
    success_mass: tuple[Real, int]
    physical_mass: tuple[Real, int]
    numerator: tuple[Real, int]
    numerator_radius: tuple[Real, int]
    lost_state_components: Count = 0
    lost_coefficients: Count = 0
    populations: tuple[Count, Count, Count]
    contribution_id: ContentID


class QLSGroupMoments(Record):
    """Shot counts and weighted moments of one sampled measurement setting of a QLS run.

    `QLSAnalysis.groups` holds one per setting.

    Attributes:
        name: Setting name.
        labels: Member Pauli labels of a group, in first-fit order. Empty for
            a mass setting.
        basis: Accumulated measurement basis, qubit zero rightmost.
        population: ``success_conditional`` (weighted values over the shots
            whose success and condition selectors match), ``unconditional``
            (that selector indicator times the weighted values over all
            returned shots, the identity term assigned to the first group) or
            ``physical_prefix`` (the computational-basis mass setting).
        returned_shots: Shots returned for the setting.
        selected_shots: Returned shots whose success and condition selectors
            match (and, for the mass setting, whose original-coordinate index
            is below the original dimension).
        mean: Weighted outcome mean, or the selected fraction of the mass
            setting. None when the setting has no shots to average.
        second_moment: Weighted second moment, for a group.
        variance: Unbiased variance of the mean, for a group of at least two shots.
        contribution_id: Content hash of the `ObservationChunk` that holds
            the setting's counts.
    """

    name: Text
    labels: tuple[Text, ...] = ()
    basis: Text
    population: Literal["success_conditional", "unconditional", "physical_prefix"]
    returned_shots: Count
    selected_shots: Count
    mean: Real | None = None
    second_moment: Real | None = None
    variance: Real | None = None
    contribution_id: ContentID


class QLSAnalysis(Result):
    """Solution or requested output of `A x = b` from a `QLS` run.

    [`solve`][nwqlib.scientist.solve] returns it for a `QLS` method, and
    `load_result` reopens a saved one. For the default `Solution` output
    the answer is `x`, the physical solution in the original coordinates
    with its scale and phase, in the problem's `unit`. `value` gives the
    requested output of any kind, the saved array, the `Samples` arrays or
    the scalar `scalar_value`. The shortcut solvers give a unit direction
    without the physical magnitude, so `x` is unavailable for them. When a
    requested quantity cannot be recovered, `unavailable` gives the reason.
    `epsilon_inv` does not bound the error of these values. Check a result
    with `result.verify(checks=QLSVerification(...))`. `result.analyze()`
    takes no settings, so another polynomial needs a new Plan. The fields
    below are read-only. The fields of
    [`Result`][nwqlib.core.analysis.Result] are present too.

    An exact scalar output (`shots=None`) saves the statistics of its one
    readout in `reduction`: the complete mass (the squared norm) of the
    saved state, the success-only mass p_alg, the physical-slice mass p and
    the projected moment q. A
    `NormalizedExpectation` reports `q/p`, a `QuadraticForm` `Gamma**2 q`
    and a `NormSquared` `Gamma**2 p`, each with `norm_squared` equal to
    `Gamma**2 p`, where Gamma is the physical recovery scale. A
    `NormSquared` reduction saves q as zero. All these values share the
    readout's `mass_contribution_id`.

    Attributes:
        scalar_value: Requested scalar of a `NormSquared`, `QuadraticForm`
            or `NormalizedExpectation` output, physical or unit-normalized as
            that output defines it, or `None`.
        unavailable: Reason a requested scalar or array is unavailable, or
            `None`.
        norm_squared: Recovered physical norm squared `||x||**2`, when
            available. The shortcut solvers never set it.
        physical_scale: Physical scale of the obtained vector, from the
            amplitudes or the classical model, not inferred from counts.
        physical_scale_unavailable: Why the physical magnitude cannot be
            recovered, for example for a shortcut's unit direction.
        numerator: Observable numerator before division by the mass.
        numerator_frame: Normalization of `numerator`, `"unit"` for a
            normalized expectation or `"physical"` for a quadratic form.
        algorithm_success_mass: Success-only mass p_alg, a probability,
            empirical for counts.
        physical_slice_mass: Mass p of the success-and-physical slice, a
            fraction of the whole state, or of all returned shots for counts.
            `None` when no measurement tests the original-coordinate prefix,
            a padded sampled quadratic form whose group bases all contain X
            or Y. For an unpadded output without condition bits, the two
            masses describe the same outcomes, and the exact and sampled
            readouts report equal values.
        mass_contribution_id: Content hash of the measurement that supplies
            those masses.
        artifact: Description of the saved solution array. Printing the
            result does not load it.
        applications: Evaluations of the classical model and their recorded
            work.
        execution: `"classical"`, the polynomial model evaluated
            classically, or `"quantum"`, the circuits run on a backend.
        submitted_shots: Shots requested for the mass measurement, when
            sampled.
        returned_shots: Shots returned by that measurement, before either
            selection.
        algorithm_selected_shots: Counts that satisfy algorithmic success.
        physical_selected_shots: Counts that also satisfy the conditions and
            lie in original coordinates.
        samples: For a `Samples` output, the distinct original-coordinate
            indices (`samples.indices`) and their counts (`samples.counts`),
            not all returned shots. `None` for other outputs.
        reduction: Saved statistics of the exact scalar readout.
        groups: Weighted moments of each sampled measurement setting, with
            its Pauli labels, basis and returned and selected shots.
    """

    scalar_value: Real | None = None
    unavailable: Text | None = None
    norm_squared: Nonnegative | None = None
    physical_scale: PhysicalScale | None = None
    physical_scale_unavailable: Text | None = None
    numerator: Real | None = None
    numerator_frame: Literal["physical", "unit"] | None = None
    algorithm_success_mass: Real | None = None
    physical_slice_mass: Real | None = None
    mass_contribution_id: ContentID | None = None
    artifact: ArtifactManifest | None = None
    applications: tuple[KernelApplication, ...] = ()
    execution: Literal["classical", "quantum"]
    submitted_shots: Count | None = None
    returned_shots: Count | None = None
    algorithm_selected_shots: Count | None = None
    physical_selected_shots: Count | None = None
    samples: QLSSamples | None = None
    reduction: QLSProjectedMoments | None = None
    groups: tuple[QLSGroupMoments, ...] = ()

    def _summary_lines(self):
        """Return the report summary lines of this result.

        They give the requested array, sample counts or scalar, the solver,
        the polynomial and phase components (`_component_lines`), a warning
        when the spectral assumption was supplied rather than checked during
        planning, and any reason a quantity is unavailable.
        """
        kind = None if self._plan is None else self._plan.output.kind
        if self.artifact is not None:
            label = "Array output" if kind is None else "Physical solution" if kind == "solution" else "State vector"
            lines = [label + ": " + self._array_text(self.artifact)]
            if self.artifact.output.frame == "unit":
                lines.append("Normalized direction; physical solution magnitude is not supplied by these amplitudes")
            if acquisition := self._array_acquisition_text(self.artifact):
                lines.append(acquisition)
        elif kind == "samples":
            stored = 0 if self.samples is None else len(self.samples.indices.array)
            lines = [f"Samples: {stored} stored original-coordinate indices; returned shots={self.returned_shots}"]
            lines.append(f"Selected counts: algorithm branch={self.algorithm_selected_shots}; "
                         f"physical coordinates={self.physical_selected_shots}")
            lines.append(f"Observed selection masses: algorithm branch={self._scalar_text(self.algorithm_success_mass)}; "
                         f"physical slice={self._scalar_text(self.physical_slice_mass)}")
        else:
            lines = [f"{kind or 'Scalar value'}: {self._scalar_text(self.scalar_value)}{self._unit_text()}"]
        if self._plan is not None:
            lines.append(f"QLS solver: {self._plan.method.solver}; selected finite polynomial model")
            rec = self._plan.reconstruction
            lines += _component_lines(rec, self._plan.method.epsilon_inv)
            if rec.sigma_min is None:
                lines.append("At selection, the supplied spectral premise was not independently checked; "
                             "inverse accuracy is conditional on alpha and kappa covering the original spectrum.")
        if self.unavailable:
            lines.append(self.unavailable)
        if self.physical_scale_unavailable:
            lines.append(self.physical_scale_unavailable)
        return lines

    @property
    def value(self):
        """The requested output, the saved array, the `Samples` arrays or the scalar."""
        if self.artifact is not None:
            return self.data.artifact(self.artifact).array
        if self.plan.output.kind == "samples":
            return self.samples
        return self.scalar_value

    @property
    def x(self):
        """The physical solution `x` in the original coordinates, a complex array.

        Available for `Solution` and physical `StateVector` outputs, and
        `None` when the array is unavailable, with the reason in
        `unavailable`. A unit `StateVector`, the only vector output of the
        shortcut solvers, carries no physical magnitude.

        Raises:
            ValueError: For any other output.
        """
        if self.plan.output.kind != "solution" and not (
            self.plan.output.kind == "state_vector" and self.plan.output.normalization == "physical"
        ):
            raise ValueError("x requires a physical solution output")
        return self.value

    @property
    def alpha(self):
        """The encoding normalization `alpha` chosen at planning."""
        return self.plan.reconstruction.alpha

    @property
    def kappa(self):
        """The encoded inverse-gap parameter `kappa_be`, not the original condition number.

        With `kappa="auto"` it is `max(1, alpha / sigma_min)` from the
        computed singular endpoint. A supplied numeric `kappa` is kept as
        given, after a check that it covers that value when `sigma_min` is
        known.
        """
        return self.plan.reconstruction.kappa_be

    def validate_plan(self, plan):
        """Check the output normalization, the physical mass and the frame of the saved array against the chosen QLS route.

        When the success and physical subsets coincide, the exact and sampled
        routes compute one mass for both, so the two reported masses of an
        unpadded output without conditions are equal.
        """
        self._validate_common_plan(plan, Plan)
        validate_selection(plan)
        if self.construction_id != plan._construction_id or self.execution != plan.execution:
            raise ValueError("QLS result differs from its selected construction/execution")
        rec = plan.reconstruction
        whole = rec.embedding == "none" and rec.original_dimension == rec.padded_dimension
        a, p = self.algorithm_success_mass, self.physical_slice_mass
        shared = self.reduction is not None or self.returned_shots is not None
        if whole and (
            (a is None) != (p is None)
            or a is not None
            and (p != a if shared else abs(p - a) > NUMERICAL_RELATION_RTOL * max(1.0, a))
        ):
            raise ValueError("whole-branch QLS masses must describe the same population")
        if (self.reduction is not None) != (plan.execution == "quantum" and plan.shots is None
                                            and plan.output.kind in {"norm_squared", "quadratic_form",
                                                                     "normalized_expectation"}):
            raise ValueError("QLS exact scalar outputs carry exactly their saved reduction statistics")
        if (self.samples is not None) != (plan.output.kind == "samples"):
            raise ValueError("QLS sample arrays belong to a Samples output")
        if plan.output.kind == "norm_squared" and self.scalar_value != self.norm_squared:
            raise ValueError("QLS norm result differs from its observed norm")
        if (
            plan.output.kind in {"solution", "state_vector", "samples"}
            and self.scalar_value is not None
        ):
            raise ValueError("QLS vector/sample output cannot carry a scalar value")
        if plan.method.solver != "qsvt_inverse" and (
            self.physical_scale is not None or self.norm_squared is not None
        ):
            raise ValueError("shortcut cannot publish an unacquired physical magnitude")
        if self.execution == "classical" and self.norm_squared != (
            None if self.physical_scale is None else self.physical_scale.squared_as_float()
        ):
            raise ValueError("QLS classical norm differs from its physical scale")
        from .method import _array_output

        expected = _array_output(plan.problem, plan.output)
        manifest = self.artifact
        if manifest is not None:
            if manifest.plan_id != self.plan_id or manifest.construction_id != self.construction_id:
                raise ValueError("QLS artifact differs from its actual selection")
            if manifest.output != expected:
                raise ValueError(
                    "QLS artifact differs from its requested original-coordinate frame"
                )


def _component_lines(rec, epsilon_inv):
    """Return the summary lines of the polynomial and phase components of ``rec``, from its stored values alone.

    Each quantity keeps its own meaning, and nothing here decides whether the
    result is accurate. Write k = ``polynomial_kappa``, s = ``rescale``, delta
    = ``phase_error`` and e = ``residual_check_value``. The checks e and delta
    have no arithmetic enclosure, so every combination is nominal.

    For an inverse polynomial, Hermitian or dilated spectra in ``[1/k, 1]``
    give ``y = (A/alpha)^-1 b_hat`` with ``1 <= ||y|| <= k``, and recovery by
    ``k s`` turns a phase perturbation of operator norm delta into a relative
    error ``k s delta/||y||`` (``verification._comparisons``). The actual
    ``||y||`` needs a reference solve that a summary must not run, so the
    line states the dependence and evaluates its worst case ``||y|| = 1``.
    That value can exceed ``epsilon_inv`` for a valid computation: it is not
    the error of this result.

    A shortcut returns a unit direction, so its phase term is relative to the
    ideal selected vector v of norm m: if ``delta < m``, the two normalized
    vectors differ by at most ``2 delta/m``. Dalzell arXiv:2406.12086v2,
    Appendix C, Theorem 1 gives ``m**2 s**2 >= L = sin^2(2 theta) ((1 - eta)/(1 + eta))**2``
    with ``theta = arctan(||y||/t)``. Its ``sin^2(2 theta)`` is
    ``4 t^2 n^2/(t^2 + n^2)^2`` at ``n = ||y||``, which rises and then falls
    in n, so over ``[1, k]`` its minimum is at an endpoint. That gives the
    term ``2 s delta/sqrt(L)``, which is conditional on the spectral premise
    and covers only the phase synthesis relative to the selected polynomial.
    """
    polynomial, delta = rec.polynomial, rec.phase_error
    s, k = polynomial.rescale, rec.polynomial_kappa
    if polynomial.kind == "inverse":
        lines = [f"Inverse polynomial: target epsilon_inv={epsilon_inv:.3g}; numerical check value "
                 f"{polynomial.residual_check_value:.3g}"]
    else:
        lines = [f"Kernel-reflection polynomial: eta={polynomial.eta:.3g}; "
                 f"t={rec.t_source if rec.t is None else format(rec.t, '.6g')}"]
    if rec.phase_tolerance is None:
        return lines + ["Phase synthesis: not applicable to the classical polynomial model"]
    lines.append(f"Phase synthesis: residual {delta:.3g} (numerical estimate, no arithmetic enclosure) "
                 f"against target {rec.phase_tolerance:.3g}; polynomial_kappa={k:.6g}, rescale={s:.6g}")
    if polynomial.kind == "inverse":
        lines.append(f"Relative inverse error relation, given the spectral assumption: at most "
                     f"{polynomial.residual_check_value:.3g} + {k * s * delta:.3g}/||y||, ||y|| in [1, {k:.6g}], "
                     f"so at most {polynomial.residual_check_value + k * s * delta:.3g} (nominal worst case, "
                     "not the error of this result); encoding, execution and sampling errors are not included")
    elif rec.t is not None:
        from .norm_search import _success_center, _success_window

        center = min(_success_center(1.0, rec.t), _success_center(k, rec.t))
        lower = _success_window(center, polynomial.eta)[0]
        lines.append(f"Unit-direction relation, given the spectral assumption: the phase synthesis adds at most "
                     f"{2 * s * delta / sqrt(lower):.3g} relative to the selected polynomial, for ||y|| in "
                     f"[1, {k:.6g}] and t={rec.t:.6g}; the polynomial's own direction error is not included")
    return lines


def validate_selection(plan):
    """Require a Plan whose reconstruction matches its original inputs and solver.

    The reconstruction names the original ``A`` and ``b``, belongs to the
    configured solver and keeps any supplied ``alpha`` and ``kappa``.
    Classical execution plans one evaluation of the polynomial model.
    """
    from .method import QLS

    if (
        type(plan) is not Plan
        or type(plan.method) is not QLS
        or not isinstance(plan.reconstruction, QLSReconstruction)
    ):
        raise ValueError("QLS requires its actual configured Method and selected reconstruction")
    rec = plan.reconstruction
    if (
        rec.original_operator != plan.problem.A.reference
        or rec.original_rhs != plan.problem.b.manifest.reference
        or rec.original_dimension != plan.problem.dimension
    ):
        raise ValueError("QLS selection differs from its original A,b input association")
    if (rec.polynomial.kind == "inverse") != (plan.method.solver == "qsvt_inverse"):
        raise ValueError("QLS polynomial belongs to another configured solver")
    if plan.method.alpha != "auto" and plan.method.alpha != rec.alpha:
        raise ValueError("QLS changed its supplied alpha")
    if plan.method.kappa != "auto" and plan.method.kappa != rec.kappa_be:
        raise ValueError("QLS changed its supplied kappa")
    if plan.execution == "classical" and (len(plan.construction.kernels) != 1 or len(plan.experiments) != 1):
        raise ValueError("classical QLS selects one actual polynomial-model invocation")
