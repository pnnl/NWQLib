"""Selected LCHS reconstruction and original-coordinate scientific results."""

from typing import Literal

from pydantic import model_validator

from nwqlib._quantum_readout import ReadoutSetting
from nwqlib.algorithms.protocol import AlgorithmDescriptor
from nwqlib.artifacts import ArtifactManifest
from nwqlib.core.analysis import Result
from nwqlib.core.records import ContentID, FrozenArray, Nonnegative, PositiveInt, Real, Record, Source, Text
from nwqlib.execution import KernelApplication
from nwqlib.operators.access import Count
from nwqlib.problems.inputs import PhysicalScale

METHOD = Source(name="lchs", version="3", domain="constant du/dt=-A u+b",
    reference="An, Childs and Lin, arXiv:2312.03916v2 Eqs. (4), (6)-(7), (61), (186); finite tail-budget inversion; Trefethen ATAP ISBN 978-1-61197-239-9, Theorem19.3, n=Q-1")
DESCRIPTOR = AlgorithmDescriptor(method="lchs", version="3",
    problem_families=("linear_dynamics",),
    output_families=("solution", "state_vector", "norm_squared", "quadratic_form", "normalized_expectation", "samples"),
    access_families=("dense", "periodic_stencil"),
    resource_coverage=("selected native recipe and declared host numerical work",),
    evidence_coverage=("physical reconstruction and applicable conditional component bounds",),
    limitations=("dense SELECT uses classically exponentiated branches",
                 "construction tolerance does not bound total physical output error",
                 "floating-point and provider/model errors remain separate"), references=(METHOD,), maintenance="NWQLib")


class LCHSReconstruction(Record):
    """Actual recovery and coordinate mapping, never a second scientific input.

    Attributes:
        elapsed_time: Selected final time minus initial time.
        dimension: Original physical coordinate dimension.
        encoded_dimension: Dimension including any coherent dummy coordinates.
        mode: "quadrature" (finite LCHS sum), "unitary" (a quantum Plan
            whose L is exactly zero), "initial" (zero elapsed time, so the
            initial vector is the solution) or "zero" (zero initial state and
            no nonzero source acting over a positive time, so the solution is
            zero).
        has_source: Whether the original problem includes a constant source.
        initial_scale: Original initial-vector norm in binary physical-scale form.
        source_scale: Original source-vector scale, when a source is selected.
        recovery: Physical scale that multiplies the success-projected
            amplitudes. For LCHS evolution it composes the coefficient 1-norm,
            any QSP amplitude factor, without a constant source the
            initial-state norm, and the power of two 2**e that the
            coefficients omit when the PSD recovery exp(shift*T) exceeds
            binary64. A no-evolution readout uses the initial-state
            norm alone. None for classical host Plans and for no-evolution
            Plans answered from the ingested input. A unit-normalized output
            does not depend on it.
        success_bits: Actual success-projector qubit indices.
        system_bits: Ordered encoded coordinate qubits.
        terms: Selected observable Pauli labels and coefficients.
        settings: Actual acquisitions: the one exact ``projected_moments``
            reduction, the amplitude or Samples readout, or one counts
            setting per qubit-wise commuting group (its label is the group's
            accumulated basis), the physical-mass counts setting of a
            NormSquared output and, for a padded normalized output, the
            unrotated ``physical_mass`` setting.
        groups: Member labels of each group setting, in first-fit order,
            aligned with the settings named ``group_<g>``.
        grouping_comparisons: Evaluated candidate-mask comparisons of sampled QWC
            grouping, charged to max_readout_work.
        select_work: Sum of SELECT construction work recorded at planning, including
            its separately admitted decomposition, selection and classification
            charges under the construction-stage max_select_work limits.
        coefficient_l1_norm: One-norm ``sum_j |c_j|`` of the coefficient table
            this Plan selects. Without a constant source, and in every
            classical Plan, the table holds the kernel quadrature
            coefficients of one operator application, so the value is
            dimensionless. A quantum Plan with a constant source prepares the
            source branch layout instead, whose coefficients carry the
            initial-state norm or a Duhamel weight times the source norm,
            and the PSD growth of their application, so the value is in the
            solution's unit. When the PSD recovery exp(shift*T) exceeds
            binary64 this is the one-norm of the stored coefficients, which
            omit its power of two 2**e.
        physical_branches: Nonpadding entries of that table, the k nodes of
            one application, or for a quantum constant-source Plan the k
            nodes of the initial application and of every Duhamel node.
        padded_branches: Power-of-two address-space size of that table,
            including padding. A classical Plan runs no SELECT, and the value
            is the address size its kernel table would need.
        selected_backend: Hamiltonian-evolution implementation chosen once.
        selected_select: Concrete SELECT implementation or identity route.
        psd_premise: "numerical" (the scale-aware eigenvalue check of
            _numerical_psd_decision passed, after any shift),
            "analytic_periodic" (PSD by construction for a PeriodicStencil)
            or "unavailable".
        psd_shift: Applied operator shift, with physical growth restored elsewhere.
        l_norm: Selected norm of the processed Cartesian L component.
        kernel_approximation_bound: Kernel-integral approximation bound. This is
            the cutoff tail for the ACL arXiv:2312.03916v2 Eq.(7) kernel and
            also includes the approximate-identity error for Low-Somma
            arXiv:2508.19238v2. It bounds the operator
            norm of the propagator error for a unit input vector, before the
            PSD growth factor. The fact ``kernel_approximation`` in
            ``Plan.facts``, which a classical Result also carries in
            ``Result.facts``, is the physical L2 bound instead, this field
            times the input norm, the Duhamel weight and
            ``exp(psd_shift * t)`` of each operator application, summed over
            the applications. It is not total error.
        quadrature_bound: Available k-quadrature component bound, in the same
            unit-input operator frame. The fact ``k_quadrature``, published
            the same way, is its physical L2 bound, weighted and summed as
            above. Not total error.
        source_nodes: Selected Duhamel time nodes.
        source_weights: Matching Duhamel time-quadrature weights.
        step_counts: Selected product-formula repetitions where applicable.
        classical_work: Existing classical work counters/proxies, not CPU timing.
    """

    schema_version: Literal[4] = 4
    elapsed_time: Nonnegative
    dimension: PositiveInt
    encoded_dimension: PositiveInt
    mode: Literal["quadrature", "initial", "zero", "unitary"] = "quadrature"
    has_source: bool = False
    initial_scale: PhysicalScale
    source_scale: PhysicalScale | None = None
    recovery: PhysicalScale | None = None
    success_bits: tuple[Count, ...] = ()
    system_bits: tuple[Count, ...] = ()
    terms: tuple[tuple[Text, Real], ...] = ()
    settings: tuple[ReadoutSetting, ...] = ()
    groups: tuple[tuple[Text, ...], ...] = ()
    grouping_comparisons: Count = 0
    select_work: Count = 0
    coefficient_l1_norm: Nonnegative = 0.
    physical_branches: Count = 0
    padded_branches: PositiveInt = 1
    selected_backend: Text
    selected_select: Text
    psd_premise: Literal["numerical", "analytic_periodic", "unavailable"] = "unavailable"
    psd_shift: Real = 0.
    l_norm: Nonnegative = 0.
    kernel_approximation_bound: Nonnegative | None = None
    quadrature_bound: Nonnegative | None = None
    source_nodes: tuple[Real, ...] = ()
    source_weights: tuple[Real, ...] = ()
    step_counts: tuple[Count, ...] = ()
    classical_work: tuple[tuple[Text, Count], ...] = ()

    @model_validator(mode="after")
    def _coordinates(self):
        if self.encoded_dimension < self.dimension:
            raise ValueError("encoded dimension cannot truncate original coordinates")
        if len(self.source_nodes) != len(self.source_weights):
            raise ValueError("Duhamel nodes and weights differ in length")
        if not self.has_source and (self.source_scale is not None or self.source_nodes):
            raise ValueError("homogeneous selection cannot carry source acquisition")
        if self.system_bits:
            width = len(self.success_bits) + len(self.system_bits)
            if self.success_bits + self.system_bits != tuple(range(width)):
                raise ValueError("LCHS uses low success bits followed by ordered system bits")
            if 1 << len(self.system_bits) != self.encoded_dimension:
                raise ValueError("system width differs from the selected embedding")
        return self


class LCHSSamples(Record):
    """Sampled output of an LCHS run: the observed original coordinates and their counts, as two int64 arrays.

    `LCHSAnalysis.samples` holds it for a `Samples` output. Analysis adopts
    the fresh, C-contiguous int64 arrays that the sample reduction returns,
    without copying them. For m stored indices, the adopted arrays occupy
    16m data bytes and the local validation phase peaks at 17m data bytes
    while one Boolean comparison mask is live. These array-data terms
    exclude caller-held input data, earlier reduction workspace and Python
    object overhead. Building the record does not check these bytes against
    a memory limit.

    Attributes:
        indices: Distinct original-coordinate indices below the original
            dimension, increasing. Padding coordinates are excluded.
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
            raise ValueError("LCHS samples need increasing distinct int64 indices and positive int64 counts")
        return self

    @property
    def total(self):
        """Sum of the counts as a Python integer."""
        return sum(map(int, self.counts.array))


class LCHSProjectedMoments(Record):
    """Saved statistics of the exact scalar readout of an LCHS run (`shots=None`): the three masses and the projected moment.

    `LCHSAnalysis.reduction` holds it.
    Each scalar is a canonical ``(mantissa, exponent)`` pair: zero is
    ``(0.0, 0)``, a nonzero mantissa has absolute value in [1/2, 1).

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
            success-and-physical coordinates.
        numerator: Projected moment ``q = v^dagger O v`` of the stored Pauli
            observable on the physical slice v, not a full selected-block
            moment. Zero for a NormSquared output.
        numerator_radius: Upper bound, rounded upward, on the absolute rounding
            error in ``numerator`` from its classical evaluation using the stored
            observable and the computed physical slice. It assumes binary64
            round-to-nearest arithmetic with gradual underflow and finite
            intermediate results. Errors in the prepared state and in converting
            the original observable to the stored one are separate.
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


class LCHSGroupMoments(Record):
    """Shot counts and weighted moments of one sampled measurement setting of an LCHS run.

    `LCHSAnalysis.groups` holds one per setting.

    Attributes:
        name: Setting name.
        labels: Member Pauli labels of a group, in first-fit order. Empty for
            a mass setting.
        basis: Accumulated measurement basis, qubit zero rightmost.
        population: ``success_conditional`` (weighted values over the
            success-selected shots), ``unconditional`` (success indicator
            times weighted values over all returned shots, the identity term
            assigned to the first group) or ``physical_prefix`` (the
            unrotated mass setting).
        returned_shots: Shots returned for the setting.
        selected_shots: Returned shots whose success selectors match (and,
            for the mass setting, whose coordinate is physical).
        mean: Weighted outcome mean, or the selected fraction of the mass
            setting. None without a population.
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


class LCHSAnalysis(Result):
    """Solution or requested output of `du/dt = -A u + b` from an `LCHS` run.

    [`solve`][nwqlib.scientist.solve] returns it for an `LCHS` method, and
    `load_result` reopens a saved one. For the default `Solution` output the
    answer is `solution`, the physical solution u(T) in the original
    coordinates, with its norm and phase, in the problem's `unit`. A
    `StateVector` output is in `state_vector`. `NormSquared`,
    `QuadraticForm` and `NormalizedExpectation` give the scalar `value`, and
    `Samples` gives `samples`. Counts do not recover a complex vector. When
    the output cannot be recovered, `unavailable` gives the reason, for
    example a normalized output at zero physical mass. `print(result)`
    shows the value with the construction tolerance and the kernel bounds,
    each labeled as not a total error bound. `result.analyze()` takes no
    settings. The fields below are read-only. The fields of
    [`Result`][nwqlib.core.analysis.Result] are present too.

    The kernel-integral and k-quadrature bounds are facts of the Plan
    (`result.plan.facts`), and a classical Result also carries them in
    `result.facts`. They are physical L2 bounds of error components, not a
    bound on the total error of the output. No reference solution is
    computed unless `result.verify` is called.

    Attributes:
        value: Requested scalar of a `NormSquared`, `QuadraticForm` or
            `NormalizedExpectation` output, physical or unit-normalized as
            that output defines it. `None` for vector and `Samples` outputs
            and when unavailable.
        unavailable: Reason the requested output cannot be recovered, or
            `None`.
        norm_squared: Recovered physical norm squared `||u(T)||**2`, when
            available.
        physical_scale: Physical scale of the obtained vector, when
            representable.
        numerator: Observable value, the normalized ratio for a
            `NormalizedExpectation` and the recovered physical quadratic form
            for a `QuadraticForm`.
        numerator_frame: Normalization of `numerator`, `"unit"` for the
            normalized ratio or `"physical"` for the quadratic form.
        artifact: Description of the stored solution or state array.
            Printing the result does not load the array, and `solution` and
            `state_vector` read it.
        applications: Numerical operator applications of the run, each with
            its error-component facts.
        submitted_shots: Shots requested for a sampled output, if
            applicable.
        returned_shots: Shots returned, before selection of the physical
            coordinates.
        selected_shots: Returned shots kept for the physical output. For
            `Samples`, the sum of the sample counts.
        samples: For a `Samples` output, the distinct original-coordinate
            indices (`samples.indices`, increasing, padding coordinates
            excluded) and their positive counts (`samples.counts`), both
            int64 arrays. They do not cover all submitted shots.
        reduction: Saved statistics of the exact scalar readout
            (`shots=None`): the complete, success and physical-slice masses
            and the projected moment, as mantissa-exponent pairs.
        groups: Weighted moments of each sampled measurement setting, with
            its Pauli labels, basis and returned and selected shots.
    """

    value: Real | None = None
    unavailable: Text | None = None
    norm_squared: Nonnegative | None = None
    physical_scale: PhysicalScale | None = None
    numerator: Real | None = None
    numerator_frame: Literal["physical", "unit"] | None = None
    artifact: ArtifactManifest | None = None
    applications: tuple[KernelApplication, ...] = ()
    submitted_shots: Count | None = None
    returned_shots: Count | None = None
    selected_shots: Count | None = None
    samples: LCHSSamples | None = None
    reduction: LCHSProjectedMoments | None = None
    groups: tuple[LCHSGroupMoments, ...] = ()

    @model_validator(mode="after")
    def _selected(self):
        if self.samples is not None and self.selected_shots != self.samples.total:
            raise ValueError("LCHS selected shots differ from the sum of the sample counts")
        return self

    def _summary_lines(self):
        """Describe the Result from stored records only, without loading arrays.

        Every error-related line states its scope. The construction tolerance
        and the kernel bounds are labeled as not total physical-output error,
        and the layered MPS circuit error is reported as not evaluated.
        """
        plan = self._plan
        kind = None if plan is None else plan.output.kind
        if self.artifact is not None:
            label = "Array output" if kind is None else "Physical solution" if kind == "solution" else "State vector"
            lines = [label + ": " + self._array_text(self.artifact)]
            if acquisition := self._array_acquisition_text(self.artifact):
                lines.append(acquisition)
        elif kind in {"solution", "state_vector"}:
            label = "Physical solution" if kind == "solution" else "State vector"
            frame = "physical" if kind == "solution" else plan.output.normalization
            phase = "physical" if kind == "solution" else plan.output.global_phase
            description = f"shape=({plan.problem.dimension},), {frame}, phase={phase}"
            if unit := self._unit_text():
                description += "; unit=" + unit.strip()
            state = plan.problem.initial_state
            array = None
            if (self.unavailable is None and plan.reconstruction.mode in {"initial", "zero"}
                    and self._data is not None and not self._data.observations.chunks
                    and state.reference.representation == "vector"):
                # Ingested vectors alone own resident physical/unit arrays.
                # Positive-norm normalization preserves their physical phase.
                array = state._direction if frame == "unit" else state._physical
            text = (description + "; no acquired array" if array is None else
                    self._resident_array_text(array, description) + "; supplied initial vector")
            lines = [label + ": " + text]
        elif kind == "samples":
            stored = 0 if self.samples is None else len(self.samples.indices.array)
            lines = [f"Samples: {stored} distinct indices; selected shots={self.selected_shots}; "
                     f"returned shots={self.returned_shots}"]
        else:
            lines = [f"{kind or 'Scalar value'}: {self._scalar_text(self.value)}{self._unit_text()}"]
        if self.unavailable:
            lines.append(self.unavailable)
        if plan is not None:
            rec = plan.reconstruction
            lines.append(f"LCHS mode: {rec.mode}; elapsed time={self._scalar_text(rec.elapsed_time)}")
            if rec.mode in {"quadrature", "unitary"}:
                lines.append(f"Realization: {rec.selected_backend}; SELECT={rec.selected_select}")
                if rec.selected_backend == "dense_exact":
                    lines.append("Dense realization uses classically exponentiated branches")
                lines.append(f"Construction tolerance: {self._scalar_text(plan.method.approximation_tolerance)}; not a total error bound")
                lines.append(f"Selected kernel bounds per unit input before PSD growth: "
                             f"approximation={self._scalar_text(rec.kernel_approximation_bound)}; "
                             f"quadrature={self._scalar_text(rec.quadrature_bound)}; not total physical-output error")
                lines.append(f"PSD premise: {rec.psd_premise}; shift={self._scalar_text(rec.psd_shift)}")
                if rec.has_source:
                    lines.append(f"Source quadrature: {len(rec.source_nodes)} nodes; {rec.physical_branches} physical branches")
                if rec.classical_work:
                    work = dict(rec.classical_work)
                    lines.append(f"Classical work proxy: {work['size_units']}; vector rotations={work.get('vector_rotations',0)}; matrix-power products={work.get('matrix_power_products',0)}")
                for block in plan.construction.selections:
                    if block.choice == 'mps_circuit':
                        parameters = {item.parameter:item.value for item in block.cost_parameters}
                        lines.append(f"{block.signature.name}: MPS layers={parameters['mps_layers']}, max stored bond={parameters['mps_max_stored_bond']}; core bytes={parameters['mps_core_bytes']}; construction work={block.construction_work}; SDK workspace unknown")
                        lines.append("Layered MPS circuit error: not evaluated; TT-SVD discarded weight is a separate compression quantity")
        return lines

    @property
    def solution(self):
        """The physical solution u(T) of a `Solution` output, a complex array.

        It is in the original coordinates and keeps the physical norm and
        phase. With no evolution (`time` equal to `initial_time`, or a zero
        input) it is the supplied initial vector.

        Raises:
            AttributeError: If the requested output was not `Solution`.
            ValueError: If the array is unavailable, with the reason.
        """
        if self.plan.output.kind != "solution":
            raise AttributeError("solution was not the selected output")
        if self.artifact is None:
            if self.plan.reconstruction.mode in {"initial","zero"} and not self.data.observations.chunks:
                return self.plan.problem.initial_state.physical_vector()
            raise ValueError(self.unavailable or "solution array is unavailable")
        return self.data.artifact(self.artifact).array

    @property
    def state_vector(self):
        """The state vector of a `StateVector` output, a complex array.

        It is in the original coordinates, normalized and phased as the
        `StateVector` output requests.

        Raises:
            AttributeError: If the requested output was not `StateVector`.
            ValueError: If the array is unavailable, with the reason.
        """
        if self.plan.output.kind != "state_vector":
            raise AttributeError("state_vector was not the selected output")
        if self.artifact is None:
            if self.unavailable is None and self.plan.reconstruction.mode in {"initial","zero"} and not self.data.observations.chunks:
                state = self.plan.problem.initial_state
                return state._direction if self.plan.output.normalization=="unit" else state.physical_vector()
            raise ValueError(self.unavailable or "state array is unavailable")
        return self.data.artifact(self.artifact).array

    def validate_plan(self, plan):
        """Reject a Result that does not match its Plan's construction, output kind or coordinates."""
        super().validate_plan(plan)
        if plan.method.descriptor.method != "lchs" or self.construction_id != plan.construction.content_id:
            raise ValueError("LCHS result requires its actual selected construction")
        output = plan.output.kind
        if output in {"solution", "state_vector", "samples"} and self.value is not None:
            raise ValueError("array/sample output cannot carry an LCHS scalar value")
        if output == "norm_squared" and self.value != self.norm_squared:
            raise ValueError("LCHS norm output differs from its observed norm squared")
        if output == "normalized_expectation" and self.norm_squared == 0 and self.value is not None:
            raise ValueError("normalized expectation is undefined for the zero physical vector")
        if self.artifact is not None and (
            self.artifact.plan_id != self.plan_id or self.artifact.construction_id != self.construction_id
            or self.artifact.output.basis != plan.problem.basis
        ):
            raise ValueError("LCHS array must preserve its actual acquisition and original coordinates")
