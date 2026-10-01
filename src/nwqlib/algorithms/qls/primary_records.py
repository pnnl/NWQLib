"""Original-input QLS selection, physical reconstruction and attached results."""

from typing import Annotated, Literal
from pydantic import Field, model_validator
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
from nwqlib._validation import NUMERICAL_RELATION_RTOL, validate_normalized_mass
from nwqlib._quantum_readout import ReadoutSetting


METHOD = Source(
    name="qls",
    version="2",
    domain="original-coordinate Ax=b",
    reference=(
        "selected inverse Chebyshev QSVT and Dalzell arXiv:2406.12086v2 kernel-reflection methods"
    ),
)
DESCRIPTOR = AlgorithmDescriptor(
    method="qls",
    version="2",
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
    """Selected odd inverse polynomial and the evidence of its fit.

    Attributes:
        kind: Discriminator, always ``"inverse"``.
        coefficients: Chebyshev coefficients of ``P ~ 1/(polynomial_kappa x)``,
            before division by ``rescale``, with zero even entries.
        rescale: Positive divisor that puts ``P`` in the QSP amplitude
            domain. Physical recovery multiplies by it again.
        candidate_degrees: Every odd degree actually fitted, in search order,
            including failed candidates.
        certificate: Norming bound on ``max |polynomial_kappa x P(x) - 1|``
            over ``1/polynomial_kappa <= |x| <= 1``, evaluated in binary64.
        certificate_basis: How ``certificate`` was obtained.
        certificate_grid_points: Size of the affine Chebyshev certificate grid.
        lsq_node_count: Least-squares node count of the accepted fit.
    """

    kind: Literal["inverse"] = "inverse"
    coefficients: tuple[Real, ...]
    rescale: Annotated[Real, Field(gt=0)]
    candidate_degrees: tuple[PositiveInt, ...]
    certificate: Nonnegative
    certificate_basis: Literal["affine_chebyshev_residual_norming"]
    certificate_grid_points: PositiveInt
    lsq_node_count: PositiveInt

    @property
    def degree(self):
        return len(self.coefficients) - 1


class ReflectionPolynomial(Record):
    """Selected even kernel-reflection polynomial of a shortcut solver.

    Attributes:
        kind: Discriminator, always ``"reflection"``.
        coefficients: Chebyshev coefficients of Dalzell's ``K``
            (arXiv:2406.12086v2, App. B.3, Eq. (62)), before division by
            ``rescale``, with zero odd entries.
        rescale: Positive divisor that puts ``K`` in the QSP amplitude
            domain. The Eq. (17) mass window is divided by its square.
        ell: Half-degree from Dalzell Eq. (6). The degree is ``2 ell``.
        eta: Kernel-reflection parameter, ``epsilon_inv / sqrt(2)``.
        fit_points: Chebyshev nodes used to recover the coefficients.
    """

    kind: Literal["reflection"] = "reflection"
    coefficients: tuple[Real, ...]
    rescale: Annotated[Real, Field(gt=0)]
    ell: PositiveInt
    eta: Annotated[Real, Field(gt=0, le=1)]
    fit_points: PositiveInt

    @property
    def degree(self):
        return 2 * self.ell


Polynomial = Annotated[InversePolynomial | ReflectionPolynomial, Field(discriminator="kind")]


class QLSWork(Record):
    """Operation counts and peak live-array bytes of one explicit classical QLS model evaluation.

    ``host_planning.selected_work`` computes these at planning time, from
    the dimensions, the selected polynomial degree and the norm-model
    choice, after polynomial selection and before the classical model runs.
    Planning compares ``size_units`` with ``max_work`` and
    ``workspace_bytes`` with ``max_bytes``. Search rows evaluate a
    classical probability model. Planned trials and queries describe that
    model's proposed quantum work, never actual jobs. Quantum execution
    records only the two dimensions.

    Attributes:
        original_dimension: Dimension of the original equation.
        system_dimension: Dimension on which the polynomial acts, doubled by
            a Hermitian dilation.
        augmented_dimension: Dimension of Dalzell's augmented ``G_t``, or 0.
        dilation_dimension: Dimension of the Hermitian dilation of ``G_t``,
            or 0.
        target_solves: Encoded reference solves made by a norm model.
        svd_calls: Dense SVD calls of one evaluation: the SVD of ``G_t``
            for ``shortcut_native_svp``, and the original SVD when the
            evaluation acquires it because planning did not.
        eigh_calls: Dense Hermitian eigendecompositions of one evaluation:
            that of the dilation of ``G_t`` for ``shortcut_dilation``, and
            the original eigensystem of a Hermitian ``A`` when the
            evaluation acquires it because planning did not.
        matrix_products: Dense matrix-matrix products. Zero, because
            ``G_t`` is formed by a rank-one update.
        matvecs: Matrix-vector products: the two basis changes of the
            selected action, the row product ``b'† A_t`` of a shortcut and
            the coordinate matvec of the linear norm model.
        search_rows: Envelope on norm-model rows (grid candidates, search
            rounds or sequence candidates).
        planned_trials: Envelope on the norm model's planned trials.
        planned_queries: Envelope on the norm model's planned queries.
        vector_reductions: Norm and inner-product reductions over vectors.
        observable_actions: Observable applications for scalar outputs.
        workspace_bytes: Largest peak of live arrays among the preparation,
            the numerical action and the output statistics, together with
            original factors that the evaluation acquires, excluding stored
            input data, factors already owned by the Plan and native
            library workspace.
        size_units: Scalar operation count of the size law in
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
    matrix_products: Count = 0
    matvecs: Count = 0
    search_rows: Count = 0
    planned_trials: Count = 0
    planned_queries: Count = 0
    vector_reductions: Count = 0
    observable_actions: Count = 0
    workspace_bytes: Count = 0
    size_units: Count = 0


class QLSReconstruction(Record):
    """Original spectral facts and the actual selected padded/embedded relation.

    kappa_be is alpha/sigma_min when numerically known, or a user premise.
    polynomial_kappa additionally applies the explicitly named domain floor.
    The original matrix condition number never includes that floor or padding.

    Attributes:
        original_operator: Identity/reference of the original A.
        original_rhs: Identity/reference of the original physical b.
        encoding_operator: Operator represented by the selected encoding.
        original_dimension: Original equation dimension.
        padded_dimension: Coordinate dimension after admitted padding.
        system_dimension: Dimension on which the selected polynomial acts.
        embedding: Original or Hermitian-dilation representation.
        rhs_scale: Original RHS physical norm in binary-scale form.
        alpha: Actual encoding normalization, strictly positive.
        alpha_source: Source or premise selecting alpha.
        kappa_be: Encoding alpha divided by known sigma_min, or supplied bound.
        polynomial_kappa: Polynomial-domain parameter including its named floor.
        kappa_source: Source or premise selecting the encoding-domain bound.
        condition_number: Original sigma_max/sigma_min, if independently known.
        sigma_min: Existing original smallest singular endpoint, if known.
        sigma_max: Existing original largest singular endpoint, if known.
        spectral_method: Actual endpoint computation or supplied-premise scope.
        spectral_calls: Counts of spectral work already performed at selection.
        polynomial: Actual inverse/reflection coefficients and fit evidence.
        phase_solution: Complete selected QSVT phase vector, not a new fit request.
        phase_error: Norming bound on the phase-fit residual
            ``sup |Re P_phases - polynomial/rescale|`` over ``[-1, 1]``.
            None means unavailable.
        phase_evaluations: Recorded phase-solver evaluations.
        encoding_id: Identity of the actual selected encoding, when present.
        encoding_family: Selected implementation family, if applicable.
        encoding_ancillas: Ancillas required by that encoding.
        encoding_error: Bound relative to the encoding's declared operator target.
        preparation_id: Identity of the actual RHS preparation.
        t: Selected shortcut norm parameter, not recovered physical solution norm.
        t_source: Explicit supplied value or selected classical norm model.
        width: Actual selected circuit width; zero for a host-only construction.
        coordinates: Ordered coordinate qubits.
        success: Algorithmic success predicates as bit/value pairs.
        conditions: Additional selected projection predicates as bit/value pairs.
        settings: Actual readout acquisitions. An exact scalar output has
            the one ``projected_moments`` reduction; a sampled Pauli output
            has one counts setting per qubit-wise commuting group, labeled by
            the group's accumulated basis, plus the unrotated
            ``physical_mass`` setting of a padded normalized output whose
            group bases all contain X or Y; a sampled NormSquared output
            has its ``physical_mass`` setting; vector and Samples outputs
            have ``setting_0``.
        groups: Member Pauli labels of each group setting, in the order of
            the group settings and first-fit order within a group. Each
            distinct nonzero non-identity label of the observable is in one
            group.
        grouping_comparisons: Actual evaluated qubit-wise comparison masks
            of the grouping, charged to ``max_work``.
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
        recovery: Composed physical-output scale; absent for unit-only shortcuts.
        work: Operation counts and peak live-array bytes of the classical
            model (``QLSWork``). A quantum Plan records only its dimensions.
    """

    original_operator: InputRef
    original_rhs: InputRef
    encoding_operator: InputRef
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
    phase_evaluations: Count = 0
    encoding_id: ContentID | None = None
    encoding_family: Text | None = None
    encoding_ancillas: Count = 0
    encoding_error: Nonnegative | None = None
    preparation_id: ContentID | None = None
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

    @model_validator(mode="after")
    def _relations(self):
        """Require compatible original/padded dimensions, polynomial parity and physical readout
        projectors.
        """
        expected = self.padded_dimension * (2 if self.embedding == "hermitian_dilation" else 1)
        if self.original_dimension > self.padded_dimension or self.system_dimension != expected:
            raise ValueError(
                "QLS reconstruction differs from its original padding/embedding dimensions"
            )
        if (self.sigma_min is None) != (self.sigma_max is None):
            raise ValueError("original singular endpoints must be jointly available")
        if self.sigma_min is not None and (
            self.sigma_min > self.sigma_max
            or self.condition_number != self.sigma_max / self.sigma_min
        ):
            raise ValueError("condition estimate must use the original singular endpoints")
        if self.polynomial_kappa < self.kappa_be:
            raise ValueError("polynomial domain does not cover the selected encoding condition")
        p = self.polynomial
        if len(p.coefficients) != p.degree + 1 or p.degree < 1:
            raise ValueError("QLS polynomial degree differs from its coefficient table")
        if p.kind == "inverse" and (p.degree % 2 != 1 or not p.candidate_degrees):
            raise ValueError("inverse polynomial requires its actual odd-degree selection")
        if any(c != 0 for c in p.coefficients[0 if p.kind == "inverse" else 1 :: 2]):
            raise ValueError("QLS polynomial parity differs from its method")
        if self.phase_solution and len(self.phase_solution) != p.degree + 1:
            raise ValueError("QLS requires its complete selected phase vector")
        if self.width:
            bits = self.coordinates + tuple(bit for bit, _ in self.success + self.conditions)
            if sorted(bits) != list(range(self.width)) or any(
                value not in (0, 1) for _, value in self.success + self.conditions
            ):
                raise ValueError(
                    "QLS selectors and coordinates must partition actual circuit width"
                )
            if self.original_dimension > 1 << len(self.coordinates):
                raise ValueError("original QLS output does not fit its coordinate register")
        if self.t is not None and self.t > self.polynomial_kappa:
            raise ValueError("shortcut t exceeds its selected polynomial domain")
        if len(self.groups) != sum(setting.name.startswith("group_") for setting in self.settings):
            raise ValueError("QLS groups differ from their group readout settings")
        coefficients = () if self.observable_coefficients is None else self.observable_coefficients.array
        if len(self.observable_labels) != len(coefficients):
            raise ValueError("QLS observable labels differ from their coefficients")
        return self


class QLSSamples(Record):
    """Selected original-coordinate samples as two int64 arrays.

    Analysis adopts the fresh, C-contiguous int64 arrays returned by
    ``reduce_sample_arrays`` through
    ``FrozenArray._from_owned_canonical_array``. For m stored indices, the
    adopted arrays occupy 16m data bytes and the local validation phase
    peaks at 17m data bytes while one Boolean comparison mask is live.
    These array-data terms exclude caller-held input data, earlier
    reduction workspace and Python object overhead. No working-memory
    admission is made at this construction.

    Attributes:
        indices: Distinct original-coordinate indices below the original
            dimension, increasing; dummy coordinates are excluded.
        counts: Positive returned counts of those indices, each at most
            ``MAX_COUNT``.
    """

    indices: FrozenArray
    counts: FrozenArray

    @model_validator(mode="after")
    def _arrays(self):
        import numpy as np

        indices, counts = self.indices.array, self.counts.array
        if (indices.dtype != np.int64 or counts.dtype != np.int64 or indices.ndim != 1
                or indices.shape != counts.shape or np.any(counts <= 0) or np.any(indices < 0)
                or np.any(indices[1:] <= indices[:-1])):
            raise ValueError("QLS samples need increasing distinct int64 indices and positive int64 counts")
        return self

    @property
    def total(self):
        """Sum of the counts as a Python integer."""
        return sum(map(int, self.counts.array))


class QLSProjectedMoments(Record):
    """Saved scalar statistics of the one exact projected reduction of a scalar output.

    Each scalar is a canonical ``(mantissa, exponent)`` pair: zero is
    ``(0.0, 0)``, a nonzero mantissa has absolute value in [1/2, 1).

    Attributes:
        kernel: Mass and moment kernel of the reduction.
        complete_mass: Complete native norm of the saved state, over all
            ``populations[0]`` amplitudes.
        success_mass: Success-only mass p_alg, over the ``populations[1]``
            amplitudes whose success selectors match; it can include
            condition-failing branches and dummy coordinates.
        physical_mass: Physical-slice mass p over the ``populations[2]``
            coordinates whose success and condition selectors match and whose
            original-coordinate index is below the original dimension.
        numerator: Projected moment ``q = v^dagger O v`` of the stored Pauli
            observable on the physical slice v, not a full selected-block
            moment; zero for a NormSquared output.
        numerator_radius: Upward host-contraction radius of the numerator
            against the stored observable on the computed slice.
        lost_state_components: Framed slice components that underflowed.
        lost_coefficients: Framed coefficients that underflowed.
        populations: Complete, success-only and physical population sizes.
        contribution_id: The acquisition's one point chunk.
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
    """Returned population and weighted moments of one sampled counts setting.

    Attributes:
        name: Setting name.
        labels: Member Pauli labels of a group, in first-fit order; empty for
            a mass setting.
        basis: Accumulated measurement basis, qubit zero rightmost.
        population: ``success_conditional`` (weighted values over the shots
            whose success and condition selectors match), ``unconditional``
            (that selector indicator times the weighted values over all
            returned shots, the identity term assigned to the first group) or
            ``physical_prefix`` (the unrotated mass setting).
        returned_shots: Actual returned shots.
        selected_shots: Returned shots whose success and condition selectors
            match (and, for the mass setting, whose original-coordinate index
            is below the original dimension).
        mean: Weighted outcome mean, or the selected fraction of the mass
            setting; None without a population.
        second_moment: Weighted second moment, for a group.
        variance: Unbiased variance of the mean, for a group of at least two shots.
        contribution_id: The setting's acquisition.
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
    requested output of any kind, the stored array, the `Samples` arrays or
    the scalar `scalar_value`. The shortcut solvers give a unit direction
    without the physical magnitude, so `x` is unavailable for them. When a
    requested quantity cannot be recovered, `unavailable` gives the reason.
    `epsilon_inv` does not bound the error of these values. Check a result
    with `result.verify(checks=QLSVerification(...))`. `result.analyze()`
    takes no settings, so another polynomial needs a new Plan. The fields
    below are read-only. The fields of
    [`Result`][nwqlib.core.analysis.Result] are present too.

    An exact scalar output (`shots=None`) saves the statistics of its one
    readout in `reduction`: the complete norm of the saved state, the
    success-only mass
    p_alg, the physical-slice mass p and the projected moment q. A
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
        artifact: Description of the stored solution array. Printing the
            result does not load it.
        applications: Classical model applications and their recorded work.
        execution: `"classical"`, the polynomial model evaluated on the
            host, or `"quantum"`, the circuits run on a backend.
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
        a warning when the spectral premise was supplied rather than checked
        at selection, and any reason a quantity is unavailable.
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
        """The requested output, the stored array, the `Samples` arrays or the scalar."""
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
        """The encoded gap parameter `kappa_be`, not the original condition number.

        With `kappa="auto"` it is `max(1, alpha / sigma_min)` from the
        computed singular endpoint. A supplied numeric `kappa` is kept as
        given, after a check that it covers that value when `sigma_min` is
        known.
        """
        return self.plan.reconstruction.kappa_be

    @model_validator(mode="after")
    def _masses(self):
        """Check the record-level mass and scale relations.

        Mass validation uses the producing acquisition and the selected
        mass-reduction kernel. The complete native population is checked
        before projection. Success and physical masses are subpopulations,
        and their subset relation includes host reduction error without
        assuming either mass is one.

        Without the producing receipt this record checks finiteness,
        nonnegativity and population identity only: the published masses of
        an exact reduction are the binary64 values of its saved pairs, of
        nested populations, from its one acquisition; sampled masses come
        from nested integer counts. The subset relation with host error and
        the complete-norm bound of the producing receipt, its qualified
        state error when available and its probability window otherwise, are
        checked by ``_quantum_readout.validate_saved_masses`` at reduction
        and analysis. For an amplitude or host-kernel acquisition the
        physical mass may exceed the algorithm mass only within
        ``NUMERICAL_RELATION_RTOL``. A physical scale and its unavailability
        reason exclude each other, and the mass acquisition must be one of
        the result's contributions.
        """
        algorithm, physical = self.algorithm_success_mass, self.physical_slice_mass
        for value in (algorithm, physical):
            validate_normalized_mass(value)
        if self.reduction is not None:
            from nwqlib._quantum_readout import pair_float

            reduction = self.reduction
            complete, success, selected = reduction.populations
            if (
                not 0 <= selected <= success <= complete
                or algorithm != pair_float(reduction.success_mass)
                or physical != pair_float(reduction.physical_mass)
                or self.mass_contribution_id != reduction.contribution_id
            ):
                raise ValueError("QLS masses differ from their saved reduction statistics")
        elif self.returned_shots is not None and self.algorithm_selected_shots is not None:
            if not (
                (self.physical_selected_shots is None
                 or self.physical_selected_shots <= self.algorithm_selected_shots)
                and self.algorithm_selected_shots <= self.returned_shots
            ):
                raise ValueError("QLS physical counts exceed their algorithm subset")
        elif (
            algorithm is not None
            and physical is not None
            and physical - algorithm > NUMERICAL_RELATION_RTOL * max(1.0, algorithm)
        ):
            raise ValueError("QLS physical mass exceeds its algorithm subset")
        if self.physical_scale is not None and self.physical_scale_unavailable is not None:
            raise ValueError("physical scale cannot be both known and unavailable")
        if (
            self.mass_contribution_id is not None
            and self.mass_contribution_id not in self.contribution_ids
        ):
            raise ValueError("QLS mass must identify its actual contributing acquisition")
        if any(item.contribution_id not in self.contribution_ids for item in self.groups):
            raise ValueError("QLS group statistics must identify their actual acquisitions")
        return self

    def validate_plan(self, plan):
        """Bind output normalization, physical mass and artifact frames to the selected QLS
        route.

        Identical success and physical populations share one computed mass
        on the exact and sampled routes, so the two published masses of an
        unpadded, unconditioned output are equal.
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


def validate_selection(plan):
    """Require a Plan whose reconstruction still matches its original selection.

    Analysis, loading and verification call this before using a Plan. The
    reconstruction must name the original ``A`` and ``b``, belong to the
    configured solver and keep any supplied ``alpha`` and ``kappa``. A
    quantum Plan must also carry its selected phase, query and readout
    body. A mismatch raises rather than reselecting, so stored data are
    never interpreted under a different selection.
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
    if plan.execution == "quantum":
        from .quantum import validate_selected_body

        validate_selected_body(plan)
    elif len(plan.construction.kernels) != 1 or len(plan.experiments) != 1:
        raise ValueError("classical QLS selects one actual polynomial-model invocation")
