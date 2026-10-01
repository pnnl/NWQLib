"""QPE records: the planned schedule and powers, samples, results and their consistency checks.

QPEReconstruction stores everything analysis reads (time step, queries,
powers, Pauli terms). validate_selection and validate_selected_body check at
every load and before execution or analysis that these stored values agree
with the Program that acquires the data, so data are never analyzed with
parameters they were not acquired with. QPESample and QPEAnalysis hold the
reduced ancilla means and the estimate.
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import isfinite, pi
from typing import Annotated, Literal
from pydantic import Field, StrictBool, StrictInt, model_validator
from nwqlib._validation import NUMERICAL_RELATION_RTOL
from nwqlib.algorithms.protocol import AlgorithmDescriptor
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Plan
from nwqlib.core.records import (
    ContentID,
    Float64,
    FrozenArray,
    InputRef,
    Nonnegative,
    PositiveInt,
    Real,
    Record,
    Source,
    Text,
)
from nwqlib.operators.access import Count
from nwqlib.problems.inputs import StatePreparationSpec
from nwqlib.ir import Binding

REFERENCES = (
    Source(
        name="QCELS finite complex least squares",
        version="2",
        domain="finite single-mode complex fit; no global-optimum or interval guarantee",
        reference="Ding and Lin, arXiv:2211.11973v2 Eq.(2); complex amplitude eliminated, dimensionless finite grid and bracket search",
    ),
    Source(
        name="Basic Gaussian random-walk phase estimation",
        version="2",
        domain="one Bernoulli datum per continuous-time query, Gaussian moment matching",
        reference="Granade and Wiebe, arXiv:2208.04526v1 Eq. (7) and Algorithm 1; feedback sign derived from the likelihood",
    ),
    Source(
        name="Statistical phase estimation by Fourier CDF thresholding",
        version="2",
        domain="importance-sampled ACDF with caller-declared ground overlap and controlled-evolution access",
        reference="Wan, Berta and Campbell, arXiv:2110.12071v2 PDF Eqs. (A1)-(A2), HTML (16)-(17), Algorithm 1 threshold; evolution error belongs to the selected backend",
    ),
    Source(
        name="Randomized Fourier estimation",
        version="2",
        domain="uniform random integer powers and largest sampled Fourier coefficient",
        reference="Kshirsagar, Katabarwa and Johnson, Quantum 8, 1531 (2024), arXiv:2209.11322v3 Sec. 2 Eqs. (5)-(7) and Algorithm 1",
    ),
)


class QPEQuery(Record):
    """One logical quadrature of the signal z_p and the acquisition that observes it.

    ``phase_shift`` is the logical quadrature s, with mean Re(exp(i*s)*z_p):
    0 for Re z_p and -pi/2 for Im z_p. A sampled Hadamard test executes it
    as its ancilla phase before the final Hadamard. An exact static quantum
    Plan executes no phase: its queries point to one point of the shared
    trajectory Experiment, which saves the ancilla X (s = 0) or Y
    (s = -pi/2) expectation, and the two quadratures of one power share that
    point and its one acquisition. For RWPE ``phase_shift`` is a placeholder,
    and the bound feedback parameter is the phase actually used.

    ``multiplicity`` records how many random SPE or RFE draws share this
    query. QCELS and RWPE queries have multiplicity one. An exact query is
    evaluated once. With counts, each SPE or RFE query contributes its mean
    over the shots actually returned, weighted by its original random-power
    draw multiplicity. At least one valid shot is required for every count
    query. Under outcome-independent sampling with stationary independent
    shots, the expected weighted contribution is the multiplicity times the
    selected query signal, and its shot variance scales as the square of the
    multiplicity divided by the received population. The Result records
    requested and received populations and whether the requested exposure
    was complete. Sampling bounds use those populations and keep the
    uncertainty from drawing Fourier powers separate. Complete requested
    exposure preserves the separate equal-exposure draw distribution. A
    short return uses a different shot-noise distribution. The finite CDF
    crossing and Fourier peak do not provide an energy confidence interval.

    A counts batch requests ``shots * multiplicity`` repetitions
    (records.validate_selection), and analysis uses each query's received
    population (method._static_estimate, method.received_fourier_exposure).

    Attributes:
        schema_version: Record layout version.
        experiment: Unique query name ``query_<i>``.
        power: Integer power p of U for the static estimators. For RWPE the
            real relative evolution time 1/sigma_k in units of tau.
        phase_shift: Logical quadrature s in radians. 0 means Re z_p and
            -pi/2 means Im z_p.
        multiplicity: Number of random SPE or RFE draws represented by this
            query.
        acquisition: Name of the Experiment that acquires this query: the
            query name for a sampled or RWPE query, ``trajectory`` for the
            exact static quantum trajectory and ``overlaps`` for the
            classical static host kernel.
        point: Trajectory point ID whose ancilla X or Y expectation gives
            this query, or None when its Experiment observes only this query.
    """

    schema_version: Literal[3] = 3
    experiment: Text
    power: StrictInt | Real
    phase_shift: Real
    multiplicity: PositiveInt = 1
    acquisition: Text
    point: Text | None = None


class QPECommonStep(Record):
    """The one ordered second-order step shared by every exact-trajectory product-formula power.

    A common selection binds one ordered step to every requested power by
    its integer grid and cumulative counts. Its exact targets are derived
    from the stored time unit, and its acceptance records the completed
    emitted-parameter recheck at every prefix. Each requested power keeps
    its cumulative count in ``QPEPower.steps`` and its published complete
    bound in ``QPEPower.total_error``, and every power's allowance is the
    Method's ``controlled_power_error_budget``
    (``trotterization.error_budget.select_common_step``). The selection is
    stored here rather than as a ``TrotterStepSelection`` because the Plan
    identity needs a Record, and the per-power counts and bounds already
    live in the ``QPEPower`` entries.

    For a common-grid integer power p, the target time is the exact product
    ``p*val(tau)`` of p and the stored binary64 time unit, while
    ``evolution_time`` is its binary64 display and the implemented
    nonidentity evolution reaches ``steps*val(step_time)``, whose exact
    discrepancy contributes to the recorded bound.

    Attributes:
        formula_order: Order of the shared ordered product-formula step.
            Common QPE trajectories use the symmetric second-order Suzuki
            formula.
        tau: Stored finite positive binary64 time unit. The exact target time
            at integer power p is p times its represented value.
        g: Greatest common divisor of the positive requested integer powers.
            It identifies the exact target-time grid.
        m: Positive integer subdivisions of g times the represented time
            unit. The ideal common step is g*val(tau)/m.
        step_time: Stored binary64 parameter h_hat used by the emitted shared
            step. Its represented value can differ from the ideal rational
            step.
        step_count: Number r(p_max) of shared steps through the last
            requested power. This is a sufficient common-grid count and need
            not be the independently smallest count for that power.
        evolution_time: Binary64 display of the largest common-grid target
            time, whose exact value is defined by the maximum power and
            stored tau. The actual cumulative emitted time is separately
            ``r_max*val(h_hat)``.
        error_budget: Effective product-formula allowance for the full common
            trajectory after the other structural contributions are reserved
            at every queried power:
            ``min_(p>0) (epsilon_p-P_p-T_p-I_p-A_p)*r_max/r_p``, rounded
            downward.
        bound_value: Upward product-formula bound
            ``W_up*r_max*abs(val(h_hat))**3`` for the actually emitted common
            step. It is not the complete controlled-power error.
        recheck_outcome: Outcome of comparing every complete emitted-prefix
            subtotal with its corresponding allowance. Accepted common
            selections have all powers passed.
    """

    formula_order: Literal[2] = 2
    tau: Annotated[Real, Field(gt=0)]
    g: PositiveInt
    m: PositiveInt
    step_time: Annotated[Real, Field(gt=0)]
    step_count: PositiveInt
    evolution_time: Annotated[Real, Field(gt=0)]
    error_budget: Nonnegative
    bound_value: Nonnegative
    recheck_outcome: Literal["accepted"] = "accepted"


class QPEPower(Record):
    """One distinct power of U and the construction selected for it.

    The three error fields are operator-norm bounds on the controlled
    evolution, not estimator bias. Only the product-formula backend has
    them. Exact static trajectories use nonnegative integer power positions
    on one common grid. Sampled static settings select each integer power
    independently, and RWPE selects each continuous power independently.
    Each record gives its selected count and complete structural error at
    the exact represented target time. Native synthesis and execution
    roundoff are separate.

    Attributes:
        power: Integer power p, or the RWPE relative time.
        backend: ``dense_exact``, ``trotter_error_budgeted`` or ``classical``
            (host kernel, no circuit).
        evolution_time: Nearest binary64 display of the nonnegative target
            time ``p*val(tau)`` for this integer power, whose exact value is
            determined by p and tau. An independently selected sampled or
            RWPE product-formula time is ``abs(p)*tau``. None for unitary
            input.
        steps: Cumulative number of applications of the selected common step
            through this power position on the exact trajectory. Zero power has
            zero steps. For a sampled or RWPE power the number r of its own
            repeated second-order steps at its time.
        pruning_error: Upward bound on ``d_drop*abs(t_p)``. The exact
            target is ``p*val(tau)`` for an integer power and
            ``val(p)*val(tau)`` for an RWPE relative time.
        evolution_error: Upward subtotal for the selected evolution relative
            to the kept Hamiltonian at ``t_p``, including product-formula
            error, represented-time displacement and applicable
            identity-phase and rotation-angle formation errors.
        total_error: Upward rounding of the exact pruning plus evolution
            subtotal at this power's exact target time, including emitted
            parameter formation and checked against its controlled-evolution
            allowance before publication.
        selected_definition: Content ID of the construction the Program
            applies for this power. Exact-trajectory product-formula powers
            name the shared step, and sampled and RWPE product-formula powers
            their own step block. A dense exact trajectory names the block of the gap
            B^(p_i - p_(i-1)) of its selected base B (exp(-i*tau*H), or the
            polar factor V of a unitary input) that precedes this power's
            position, so several powers can name one gap block; sampled and
            RWPE dense Plans name the power's own dense block. None for
            power 0, for the classical route and for a product formula
            without nonidentity Pauli terms.
    """

    power: StrictInt | Real
    backend: Literal["classical", "dense_exact", "trotter_error_budgeted"]
    evolution_time: Nonnegative | None
    steps: Count
    pruning_error: Nonnegative | None
    evolution_error: Nonnegative | None
    total_error: Nonnegative | None
    selected_definition: ContentID | None = None

    @model_validator(mode="after")
    def _bounds(self):
        """Check the necessary outward-composition interval of finite error fields.

        For exact nonnegative parts x,y, monotonic upward rounding gives
        max(up(x),up(y)) <= up(x+y) <= up(up(x)+up(y)). The upper endpoint
        can be infinity even when the recorded total is finite. Dense and
        host powers carry no error fields, and power 0 has no construction.
        """
        from fractions import Fraction
        from nwqlib.subroutines.trotterization.error_budget import _upward_float

        values = (self.pruning_error, self.evolution_error, self.total_error)
        if self.backend == "trotter_error_budgeted":
            if any(value is None for value in values) or not (
                max(self.pruning_error, self.evolution_error)
                <= self.total_error
                <= _upward_float(Fraction(self.pruning_error) + Fraction(self.evolution_error))
            ):
                raise ValueError("selected power must keep the outward pruning plus evolution error")
        elif any(value is not None for value in values):
            raise ValueError("dense and host powers have no product-formula error terms")
        if self.power == 0 and (self.steps or self.selected_definition is not None):
            raise ValueError("zero power is an identity with no selected power construction")
        return self


class QPEReconstruction(Record):
    """Everything analysis needs to interpret the data: inputs, tau, schedule and powers.

    Attributes:
        schema_version: Record layout version.
        target: Operator actually used, after power-of-two padding.
        original_target: Operator of the supplied problem.
        original_preparation: Initial state as supplied.
        method_id: Content ID of the configured Method.
        preparation: Preparation of the reference state actually used.
        target_kind: ``hamiltonian`` (U = exp(-i*tau*H)) or ``unitary``.
        tau: Time step of U, None for unitary input.
        tau_selection: How tau was chosen: ``supplied``, ``scaled absolute
            row-sum bound``, ``Pauli coefficient l1 bound``, ``zero
            Hamiltonian`` or ``unitary input``.
        spectral_radius_bound: R with every |E| <= R, in operator units, or
            None when unavailable.
        aliasing_bound_sufficient: True when tau*R < pi, so integer powers
            identify tau*E on one branch, or for unitary input.
        queries: Planned logical quadratures in first-occurrence order.
        powers: One entry per distinct query power, sorted.
        pauli_labels: Kept nonidentity Pauli labels for product-formula
            powers, in input order, which fixes the product-formula term
            order. Qiskit order, with qubit 0 rightmost (docs/inputs.md).
        pauli_coefficients: Real coefficients c_j of those labels in the same
            order, as one float64 array (None without kept terms).
        identity_coefficient: Identity coefficient c_I split off for
            product-formula powers and applied on the control: on the exact
            trajectory through phase increments targeting
            ``-c_I*(p_k-p_(k-1))*val(tau)`` between adjacent powers, and for
            sampled and RWPE powers as the whole phase ``-p*tau*c_I``. It is 0 for dense powers, whose matrix contains
            it.
        pruned_mass: l1 mass of the pruned nonidentity terms, bounded upward
            (``error_budget.upper_dropped_mass``) for every estimator.
        pair_commutation_checks: Pairwise commutation tests the one census
            performed.
        nested_commutation_checks: Nested commutation tests the one census
            performed, zero for ``relaxed_prefix``.
        bound_variant: Selected Pauli-triangle expression, "exact_census" for
            the full pair/triple indicator structure or "relaxed_prefix" for
            the second-order suffix relaxation. None without a census.
        coefficient_arithmetic: Numerical evaluation used for the supplied
            coefficient, "outward_float64_scaled" for scaled binary64 upper
            products and sums with rational rescaling. None without a census.
        bound_coefficient: The census's upward coefficient W_up as an exact
            (numerator, denominator) pair, None without a census.
        census_block: Contraction block size actually passed to the census's
            coefficient evaluation, None without a census.
        common_step: The shared step of an exact static product-formula
            trajectory, or None for independent sampled/RWPE powers and
            when no positive common-grid power is present.
        selected_base: For unitary input, the admitted operator V = polar(A)
            of the target A, the single base of every power, gap and
            spectral consumer. None for Hamiltonian input.
        numerical_work: Admitted host-kernel work, 0 for quantum execution.
        numerical_payload: Admitted host-kernel bytes, 0 for quantum execution.
        convention: Kernels use principal radians, and the public Eigenphase
            is converted to turns once.
    """

    schema_version: Literal[3] = 3
    target: InputRef
    original_target: InputRef
    original_preparation: InputRef
    method_id: ContentID
    preparation: StatePreparationSpec
    target_kind: Literal["hamiltonian", "unitary"]
    tau: Annotated[Real, Field(gt=0)] | None
    tau_selection: Text
    spectral_radius_bound: Nonnegative | None
    aliasing_bound_sufficient: StrictBool
    queries: tuple[QPEQuery, ...]
    powers: tuple[QPEPower, ...]
    pauli_labels: tuple[Text, ...] = ()
    pauli_coefficients: FrozenArray | None = None
    identity_coefficient: Real = 0.0
    pruned_mass: Nonnegative = 0.0
    pair_commutation_checks: Count = 0
    nested_commutation_checks: Count = 0
    bound_variant: Literal["exact_census", "relaxed_prefix"] | None = None
    coefficient_arithmetic: Literal["outward_float64_scaled"] | None = None
    bound_coefficient: tuple[StrictInt, StrictInt] | None = None
    census_block: PositiveInt | None = None
    common_step: QPECommonStep | None = None
    selected_base: InputRef | None = None
    numerical_work: Count
    numerical_payload: Count
    convention: Literal["kernels use principal radians; public Eigenphase uses turns once"] = (
        "kernels use principal radians; public Eigenphase uses turns once"
    )

    @property
    def pauli_terms(self):
        """Kept (label, coefficient) pairs in product-formula order."""
        if self.pauli_coefficients is None:
            return ()
        return tuple(zip(self.pauli_labels, self.pauli_coefficients.array.tolist(), strict=True))

    @model_validator(mode="after")
    def _target(self):
        """Keep the relations that analysis and archives read without rechecking.

        Hamiltonian input has tau and unitary input has none, and only
        unitary input has a selected polar base. Query names are unique and
        the power inventory equals the distinct query powers. An exact
        trajectory's common-grid power keeps the display
        ``float(p*Fraction(tau))`` of its exact target time and the cumulative
        count ``(p/g)*m``; an independently selected sampled or RWPE
        product-formula time is ``abs(p)*tau``. Each Pauli label matches the
        register width, with its coefficient at the same position and the
        identity coefficient stored separately. The census provenance fields
        are present together.
        """
        from fractions import Fraction
        from functools import reduce
        from math import gcd

        if (self.target_kind == "hamiltonian") != (self.tau is not None):
            raise ValueError("only Hamiltonian QPE has a positive evolution time unit")
        if (self.target_kind == "unitary") != (self.selected_base is not None):
            raise ValueError("unitary QPE records its selected polar base, and only unitary QPE")
        if len({q.experiment for q in self.queries}) != len(self.queries):
            raise ValueError("QPE query names must be unique")
        if tuple(sorted({q.power for q in self.queries})) != tuple(p.power for p in self.powers):
            raise ValueError("power inventory must match the finite query/candidate inventory")
        common = self.common_step
        if common is not None:
            positive = [p.power for p in self.powers if p.power]
            if (common.tau != self.tau or not positive
                    or any(type(p.power) is not int or p.power < 0 for p in self.powers)
                    or common.g != reduce(gcd, positive)
                    or common.step_count != max(positive) // common.g * common.m
                    or common.evolution_time != float(max(positive) * Fraction(self.tau))):
                raise ValueError("QPE common step differs from its integer power grid")
        for power in self.powers:
            if self.tau is None:
                expected = None
            elif common is not None or power.backend != "trotter_error_budgeted" and type(power.power) is int:
                expected = float(abs(power.power) * Fraction(self.tau))
            else:
                expected = abs(power.power) * self.tau
            if power.evolution_time != expected:
                raise ValueError("power time must display the exact target time of its power")
            if common is not None and power.backend == "trotter_error_budgeted" and (
                    power.steps != power.power // common.g * common.m):
                raise ValueError("common-grid power steps must be the cumulative count (p/g)*m")
        width = self.preparation.basis.dimension.bit_length() - 1
        count = 0 if self.pauli_coefficients is None else self.pauli_coefficients.array.shape
        if (count != 0 and (count != (len(self.pauli_labels),) or not self.pauli_labels)
                or count == 0 and self.pauli_labels):
            raise ValueError("Pauli labels and coefficients must have the same length and order")
        if any(
            len(label) != width or set(label) - set("IXYZ") or not set(label) - {"I"}
            for label in self.pauli_labels
        ):
            raise ValueError(
                "nonidentity Pauli terms must match width and keep identity separately"
            )
        census = (self.bound_variant, self.coefficient_arithmetic, self.bound_coefficient,
                  self.census_block)
        if any(value is None for value in census) != all(value is None for value in census):
            raise ValueError("QPE census provenance fields are recorded together")
        if self.bound_coefficient is not None and (self.bound_coefficient[1] < 1
                                                   or self.bound_coefficient[0] < 0):
            raise ValueError("QPE census coefficient must be a nonnegative exact ratio")
        return self


class QCELSFit(Record):
    """Outcome of the QCELS finite search. There is no statistical uncertainty model.

    Attributes:
        objective: Always the amplitude-eliminated complex least squares.
        effective_grid: Grid size G of numerical.qcels_grid_size.
        evaluations: Objective evaluations performed, at most
            (1 + QCELS_BRACKET_EVALUATIONS)*G.
        residual: Smallest objective value found, mean |z_p - a*exp(-i*p*theta)|**2.
        interval_method: Always ``unavailable``.
    """

    objective: Literal["complex_least_squares"] = "complex_least_squares"
    effective_grid: PositiveInt
    evaluations: PositiveInt
    residual: Nonnegative
    interval_method: Literal["unavailable"] = "unavailable"


class QPEInterval(Record):
    """Nominal model interval of one estimator, not a composed uncertainty or coverage bound.

    Only RWPE produces one, from its Gaussian model.

    Attributes:
        low: Lower endpoint in ``frame``.
        high: Upper endpoint in ``frame``.
        level: Nominal model level, numerical.QPE_INTERVAL_CONFIDENCE.
        method: Interval construction, ``rwpe_gaussian_credible``.
        frame: Unwrapped phase turns centered on the reported phase, or
            energy in the requested operator units.
        interpretation: Scope text, prefixed by any identification_note.
    """

    low: Real
    high: Real
    level: Literal[0.95] = 0.95
    method: Text
    frame: Literal["unwrapped turns around reported phase", "energy in requested operator units"]
    interpretation: Text

    @model_validator(mode="after")
    def _ordered(self):
        """Require low <= high in the unwrapped frame."""
        if self.low > self.high:
            raise ValueError("interval endpoints must remain ordered in their unwrapped frame")
        return self


class QPESample(Record):
    """One measured ancilla mean with the power, phase and observations it came from.

    ``mean`` is the ancilla <Z> in [-1, 1]. For counts it equals
    (zeros - ones)/(zeros + ones). ``raw_mean`` keeps an exact non-count mean
    whose magnitude exceeded one within its roundoff window, and ``mean`` is
    then the corresponding endpoint. The guide section "Meaning of the result"
    states this convention.

    Attributes:
        experiment: Query name.
        power: Power p, or RWPE relative time, of that query.
        phase_shift: Ancilla phase in radians actually applied, the bound
            feedback for RWPE, or the logical quadrature that the classical
            host kernel evaluated. None for a trajectory point, which reads
            its quadrature from the ancilla X or Y expectation without an
            executed phase.
        mean: Re(exp(i*s)*z_p) as measured, in [-1, 1].
        raw_mean: Original exact mean when it lay just outside [-1, 1].
        zeros: Returned shots with outcome 0, None for exact readout.
        ones: Returned shots with outcome 1, None for exact readout.
        contributions: Content IDs of the observation chunks used. The two
            quadratures of one trajectory point name the same point chunk,
            which is one acquisition, not two.
        point: Trajectory point ID of the value, or None.
    """

    experiment: Text
    power: StrictInt | Real
    phase_shift: Real | None
    mean: Annotated[Real, Field(ge=-1, le=1)]
    raw_mean: Real | None = None
    zeros: Count | None = None
    ones: Count | None = None
    contributions: tuple[ContentID, ...]
    point: Text | None = None

    @model_validator(mode="after")
    def _population(self):
        """Require consistent counts, a mean equal to the count ratio and valid raw_mean use."""
        if (self.point is None) == (self.phase_shift is None):
            raise ValueError("a sample has an executed phase or a trajectory point")
        if self.raw_mean is not None:
            if (self.zeros is not None or self.ones is not None or not 1 < abs(self.raw_mean)
                    or self.mean != (1.0 if self.raw_mean > 0 else -1.0)):
                raise ValueError("raw_mean requires a non-count numerical endpoint adjustment")
        if (self.zeros is None) != (self.ones is None):
            raise ValueError("binary counts keep both frequencies")
        if self.zeros is not None:
            total = self.zeros + self.ones
            if total <= 0 or self.mean != (self.zeros - self.ones) / total:
                raise ValueError("count mean must use the actual positive returned population")
        if len(set(self.contributions)) != len(self.contributions) or not self.contributions:
            raise ValueError("sample requires distinct actual contributions")
        return self

    @property
    def shots(self):
        return None if self.zeros is None else self.zeros + self.ones


def bounded_mean(mean, window=NUMERICAL_RELATION_RTOL):
    """Return (mean, raw_mean) for a finite non-count ancilla expectation.

    A value in [-1, 1] returns (value, None). A value whose magnitude
    exceeds one by at most ``window`` returns the nearer endpoint and the
    original value. Anything else raises ValueError. An exact trajectory
    expectation uses its circuit's roundoff window
    (PreparedArtifact.probability_window), and host scalars use the fixed
    NUMERICAL_RELATION_RTOL.
    """
    if not isfinite(mean) or abs(mean) > 1 + window:
        raise ValueError("QPE binary expectation is outside its numerical endpoint window")
    if -1 <= mean <= 1:
        return mean, None
    return (1.0 if mean > 0 else -1.0), mean


def identification_note(plan):
    """Return a warning when the stored bounds do not establish what the estimate identifies, else None.

    The note states which premise is unverified. It does not claim the
    estimate is wrong.

    - RWPE: the basic walk can move the mean at most
      prior_std/(sqrt(e) - sqrt(e-1)), about 2.96*prior_std, from the prior
      mean (Granade and Wiebe arXiv:2208.04526v1, Eq. (8), p. 5, which
      prints 2.95).
      The note appears when |prior_mean| + tau*R reaches that distance or R
      is unknown.
    - SPE: the filter premise of numerical.spe_filter_domain must hold and
      tau*R + delta <= pi/2 (numerical.spe_phase_radius). Unitary input
      stores no tau or R, so it always receives this note. When both
      conditions hold, the branch check below still applies.
    - Static Hamiltonian estimators: integer powers identify tau*E only
      modulo 2*pi. The note appears unless tau*R < pi.
    """
    r = plan.reconstruction
    if plan.method.estimator == "rwpe":
        from math import e, sqrt

        # Eq. (8) sums every possible Eq. (7a) shift, sum_k sigma_k/sqrt(e),
        # a geometric series in sqrt((e-1)/e). This checks the prior's reach,
        # independently of a wrapped integer-power phase branch.
        reach = plan.method.prior_std/(sqrt(e)-sqrt(e-1))
        if r.tau is None or r.spectral_radius_bound is None:
            return "RWPE target support is unverified against the basic walk's finite reach."
        radius = r.tau*r.spectral_radius_bound
        if abs(plan.method.prior_mean)+radius >= reach:
            return ("RWPE target support is not contained in the basic walk's finite reach "
                    "from the selected Gaussian prior. The model interval need not cover the target.")
        return None
    if plan.method.estimator == "spe":
        from .numerical import spe_filter_domain

        delta, degree_ok = spe_filter_domain(plan.method.fourier_degree, plan.method.filter_beta,
                                             plan.method.overlap_lower_bound)
        if delta is None or not degree_ok:
            return ("SPE filter support is unverified: the selected beta and degree do not "
                    "establish the WBC arXiv:2110.12071v2 Theorem 3 approximation premise at "
                    "the declared overlap.")
        if (r.tau is None or r.spectral_radius_bound is None
                or r.tau*r.spectral_radius_bound > pi/2-delta):
            return ("SPE filter support is unverified: the spectral radius plus the filter's "
                    "transition half-width must not exceed pi/2 for this scan.")
    if r.target_kind != "hamiltonian" or r.aliasing_bound_sufficient:
        return None
    return ("Hamiltonian energy branch is unverified: integer-power observations identify energy "
            "only modulo 2*pi/tau, and the selected bound does not establish tau*E in [-pi, pi). "
            "The phase remains a modulo-one estimate.")


def validate_selected_body(plan):
    """Check stored mathematical parameters and their actual IR uses.

    Analysis reads powers, times, phases and readout from the reconstruction,
    while execution runs the Program. This check joins every stored value to
    the IR node that realizes it: register lifetimes, Hadamard and
    preparation wiring, each signed power block and its repeat count, the
    controlled identity phase, each quadrature or feedback angle and the
    ancilla measurement. A Plan whose stored parameters disagree with its
    Program is rejected before execution or analysis, so data cannot be
    interpreted with parameters it was not acquired with. For the classical
    route it checks the host kernel's inputs, scalar labels and size laws.

    The checks run in this order, each raising ValueError on the first
    mismatch: _validate_host_body for classical execution, otherwise
    _validate_native_wiring, _validate_power_nodes and
    _validate_query_bodies. method._quantum_program lists the node ids.
    """
    rec, program = plan.reconstruction, plan.construction.program
    nodes = {definition.id: definition.node for definition in program.definitions}
    if plan.execution == "classical":
        _validate_host_body(plan, nodes)
        return
    # For native queries, check the actual register lifetimes and operand
    # wiring before trusting stored mathematical power metadata.
    selected = {item.content_id: item for item in plan.construction.selections}
    expressions = {expression.id: expression.value for expression in program.expressions}
    width = rec.preparation.basis.dimension.bit_length() - 1
    counts = plan.shots is not None
    route = plan_route(plan)
    _validate_native_wiring(plan, nodes, selected, width, counts)
    lengths = _validate_power_nodes(plan, nodes, expressions, selected, width, route)
    _validate_query_bodies(plan, nodes, expressions, width, route, lengths)


def _validate_host_body(plan, nodes):
    """Check the classical Program: one host kernel with the stored inputs, labels and size law.

    The work and byte laws are recomputed here with the formulas of
    method._host_construction, so a stored Plan cannot declare a different
    host workload from the one its inputs imply.
    """
    from nwqlib.ir import ClassicalStage
    from .method import HOST

    rec, program = plan.reconstruction, plan.construction.program
    count = len(rec.queries)
    pointwise = plan.method.estimator == "rwpe"
    dimension = rec.preparation.basis.dimension
    labels = ("z",) if pointwise else tuple(f"z_{index}" for index in range(count))
    work = (
        8 * dimension**3
        + 16 * dimension**2
        + 16 * dimension * (1 if pointwise else len(rec.powers))
        + 64 * count
    )
    payload = 160 * dimension**2 + 128 * dimension + 64 * count
    root = nodes[program.root]
    if (
        plan.construction.selections
        or len(plan.construction.kernels) != 1
        or any(
            power.backend != "classical" or power.selected_definition is not None or power.steps
            for power in rec.powers
        )
    ):
        raise ValueError("QPE classical selection differs from its host power inventory")
    kernel = plan.construction.kernels[0]
    if (
        not isinstance(root, ClassicalStage)
        or root.boundary != "host"
        or root.kernel != kernel.name
        or root.implementation != HOST
        or kernel.implementation != HOST
        or kernel.inputs != (rec.target, rec.preparation.input)
        or kernel.scalars != labels
        or kernel.scalar_frames != ("unit",) * len(labels)
        or kernel.invocation_work != rec.numerical_work
        or rec.numerical_work != work
        or rec.numerical_payload != payload
        or len(kernel.workspace) != 1
        or (
            kernel.workspace[0].location,
            kernel.workspace[0].purpose,
            kernel.workspace[0].bytes,
            kernel.workspace[0].source,
        )
        != ("host", "workspace", payload, HOST)
        or kernel.construction_work != len(labels) + 1 + (count if pointwise else 0)
    ):
        raise ValueError(
            "QPE classical declaration differs from its stored inputs/scalars/workspace"
        )


def _validate_native_wiring(plan, nodes, selected, width, counts):
    """Check registers, their allocation and release, the Hadamard and the preparation.

    The ancilla is one clean qubit and the system register has ``width``
    qubits. A one-bit classical ``readout`` exists only for count
    acquisition. The Hadamard acts on the ancilla, and the preparation block
    acts on the system register and prepares exactly the stored reference.
    """
    from nwqlib.ir import Allocate, BlockCall, Release

    rec, program = plan.reconstruction, plan.construction.program
    registers = (("ancilla", 1, "clean_ancilla"),) + (
        (("system", width, "system"),) if width else ()
    )
    if tuple(
        (register.name, register.width, register.role) for register in program.registers
    ) != registers or tuple(
        (value.name, value.dtype, value.width) for value in program.classical
    ) != ((("readout", "bits", 1),) if counts else ()):
        raise ValueError("QPE register inventory differs from its selected ancilla/system readout")
    for name, _, _ in registers:
        allocation, release = nodes.get("allocate_" + name), nodes.get("release_" + name)
        if (
            not isinstance(allocation, Allocate)
            or allocation.wire != name
            or not isinstance(release, Release)
            or release.wire != name
        ):
            raise ValueError("QPE register lifetime differs from its selected query wiring")
    hadamard = nodes.get("h")
    if (
        not isinstance(hadamard, BlockCall)
        or hadamard.signature != "hadamard"
        or hadamard.arguments
        or tuple((port.port, port.wire) for port in hadamard.ports) != (("register", "ancilla"),)
    ):
        raise ValueError("QPE Hadamard must act on its control ancilla")
    if width:
        preparation = nodes.get("prepare")
        if (
            not isinstance(preparation, BlockCall)
            or preparation.signature != "preparation"
            or preparation.arguments
            or tuple((port.port, port.wire) for port in preparation.ports)
            != (("system", "system"),)
        ):
            raise ValueError("QPE preparation must act on its selected system register")
        prepared = next(
            (
                definition
                for definition in selected.values()
                if definition.signature.name == preparation.signature
            ),
            None,
        )
        if (
            prepared is None
            or prepared.controlled
            or prepared.adjoint
            or prepared.base_selection_id is not None
            or prepared.semantics.preparation != rec.preparation
        ):
            raise ValueError("QPE selected preparation differs from its stored requested reference")


def plan_route(plan):
    """Return the Plan's acquisition route: classical, trajectory, sampled or rwpe."""
    if plan.execution == "classical":
        return "classical"
    if plan.method.estimator == "rwpe":
        return "rwpe"
    return "trajectory" if plan.shots is None else "sampled"


def _expected_block(definition, expected, name, rec, kind):
    """Check one selected power block's implementation, input, name and cost parameters."""
    from nwqlib.ir.expressions import number

    parameters = {binding.parameter: number(binding.value) for binding in definition.cost_parameters}
    base = rec.target if rec.selected_base is None or kind == "suzuki_step" else rec.selected_base
    if (
        (definition.implementation.name, definition.implementation.version) != ("qpe." + kind, "1")
        or definition.signature.target != definition.implementation
        or definition.controlled
        or definition.adjoint
        or definition.base_selection_id is not None
        or definition.semantics.input != base
        or definition.semantics.basis != rec.preparation.basis
        or definition.signature.name != name
        or parameters != expected
        or len(definition.cost_parameters) != len(expected)
    ):
        raise ValueError("QPE selected power parameters/input differ from its stored reconstruction")


def _validate_call(nodes, name, width):
    """Require the controlled power call ``name`` on the ancilla control and system register."""
    from nwqlib.ir import BlockCall

    call = nodes.get(name)
    if (
        not isinstance(call, BlockCall)
        or call.signature != name
        or call.arguments
        or tuple((port.port, port.wire) for port in call.ports)
        != (("control", "ancilla"),) + ((("system", "system"),) if width else ())
    ):
        raise ValueError("QPE selected power call differs from its stored construction")


def _validate_phase(nodes, expressions, tag, angle):
    """Require the control phase ``tag`` to apply P(angle) on the ancilla with that exact angle."""
    from nwqlib.ir import BlockCall, Constant, ExprRef
    from nwqlib.ir.expressions import number

    call, expression = nodes.get(tag), expressions.get(tag)
    if (
        not isinstance(call, BlockCall)
        or call.signature != "identity_phase"
        or len(call.arguments) != 1
        or call.arguments[0].parameter != "angle"
        or not isinstance(call.arguments[0].value, ExprRef)
        or call.arguments[0].value.expression != tag
        or not isinstance(expression, Constant)
        or number(expression.value) != angle
        or tuple((port.port, port.wire) for port in call.ports) != (("register", "ancilla"),)
    ):
        raise ValueError("QPE identity power phase differs from its selected time")


def _validate_power_nodes(plan, nodes, expressions, selected, width, route):
    """Check the power constructions against the stored QPEPower entries and return the gap lengths.

    Exact static trajectory. For a product formula, ``gap_<p>`` applies the
    identity phase increment P(-(p-p_prev)*tau*c_I) (the binary64 value the
    common-step recheck certifies) and repeats the shared ``common_step``
    block steps(p) - steps(p_prev) times; that block stores the common
    step time. A dense trajectory gap applies the block ``power_p<k>`` of
    the gap k = p - p_prev. Each dense block stores its exponent (and tau
    for a Hamiltonian) and acts on the stored target, or on the selected
    polar base for unitary input.

    Sampled static settings and RWPE. Each power's ``selected_<p>`` applies
    its own independently selected block, repeated ``steps`` times with
    signed step time ``evolution_time/steps``, and the control phase
    P(-p*tau*c_I) for a product formula, or its dense block of p.

    Returns:
        {power: number of bound logical operations its gap adds}, for the
        trajectory's point positions.
    """
    from nwqlib.ir import Repeat, Sequence
    from .powers import _power_name

    rec = plan.reconstruction
    blocks = {definition.content_id: definition for definition in selected.values()}
    records = {power.power: power for power in rec.powers}
    for power in rec.powers:
        if power.backend == "classical":
            raise ValueError("native QPE cannot keep a classical power")
        if (
            plan.method.controlled_power_backend != "auto"
            and power.backend != plan.method.controlled_power_backend
        ):
            raise ValueError("QPE power backend differs from its selected option")
        if power.backend == "dense_exact" and power.steps:
            raise ValueError("dense QPE power has no repeated Suzuki steps")
    lengths = {}
    if route in ("rwpe", "sampled"):
        for power in rec.powers:
            required = power.power != 0 and (power.backend == "dense_exact" or bool(rec.pauli_labels))
            if required != (power.selected_definition is not None):
                raise ValueError("QPE nonzero selected power construction is missing or extraneous")
            children = []
            if power.backend == "trotter_error_budgeted":
                if rec.target_kind != "hamiltonian" or bool(power.steps) != required:
                    raise ValueError("Suzuki power requires its Hamiltonian step selection")
                angle = -power.power * rec.tau * rec.identity_coefficient
                if not isfinite(angle):
                    raise ValueError("QPE identity phase must remain finite")
                if angle:
                    tag = f"identity_{power.power}"
                    _validate_phase(nodes, expressions, tag, angle)
                    children.append(tag)
            if required:
                name = _power_name(power.power)
                kind = "dense_power" if power.backend == "dense_exact" else "suzuki_step"
                expected = {"power": power.power}
                if kind == "dense_power" and rec.tau is not None:
                    expected["tau"] = rec.tau
                elif kind == "suzuki_step":
                    expected.update(steps=power.steps, step_time=power.evolution_time / power.steps)
                _expected_block(blocks[power.selected_definition], expected, name, rec, kind)
                _validate_call(nodes, name, width)
                if power.steps:
                    repeat = nodes.get(name + "_repeat")
                    if not isinstance(repeat, Repeat) or repeat.body != name or repeat.count != power.steps:
                        raise ValueError("QPE repeated power differs from selected step multiplicity")
                    name += "_repeat"
                children.append(name)
            sequence = nodes.get(f"selected_{power.power}")
            if not isinstance(sequence, Sequence) or sequence.children != tuple(children):
                raise ValueError("QPE selected power IR differs from its stored construction")
        return lengths
    common = rec.common_step
    product = any(power.backend == "trotter_error_budgeted" for power in rec.powers)
    step = None
    if product and common is None and any(power.power for power in rec.powers):
        raise ValueError("exact-trajectory product-formula powers require their common step selection")
    if product and rec.pauli_labels and common is not None:
        step = next((d for d in blocks.values() if d.signature.name == "common_step"), None)
        if step is None:
            raise ValueError("QPE common-grid powers require their shared step block")
        _expected_block(step, {"step_time": common.step_time}, "common_step", rec, "suzuki_step")
        _validate_call(nodes, "common_step", width)
    previous = 0
    for power in rec.powers:
        p = power.power
        children, length = [], 0
        if power.backend == "trotter_error_budgeted":
            if rec.target_kind != "hamiltonian":
                raise ValueError("Suzuki power requires its Hamiltonian step selection")
            expected = None if step is None or not p else step.content_id
            if power.selected_definition != expected:
                raise ValueError("QPE product-formula power must name the shared step")
            angle = -(p - previous) * rec.tau * rec.identity_coefficient
            if not isfinite(angle):
                raise ValueError("QPE identity phase must remain finite")
            if angle:
                _validate_phase(nodes, expressions, f"identity_{p}", angle)
                children.append(f"identity_{p}")
                length += 1
            delta = power.steps - (records[previous].steps if previous in records else 0)
            if delta < 0 or delta and p == previous:
                raise ValueError("QPE gap repeats differ from the cumulative common-step counts")
            # An empty kept generator applies its common step as no gate.
            if delta and step is not None:
                repeat = nodes.get(f"common_step_{p}")
                if not isinstance(repeat, Repeat) or repeat.body != "common_step" or repeat.count != delta:
                    raise ValueError("QPE repeated power differs from selected step multiplicity")
                children.append(f"common_step_{p}")
                length += delta
        else:
            k = p - previous
            if (k != 0) != (power.selected_definition is not None) or k < 0:
                raise ValueError("QPE nonzero selected power construction is missing or extraneous")
            if k:
                name = _power_name(k)
                expected = {"power": k} if rec.tau is None else {"power": k, "tau": rec.tau}
                _expected_block(blocks[power.selected_definition], expected, name, rec, "dense_power")
                _validate_call(nodes, name, width)
                children.append(name)
                length += 1
        gap = nodes.get(f"gap_{p}")
        if children and (not isinstance(gap, Sequence) or gap.children != tuple(children)):
            raise ValueError("QPE gap IR differs from its stored construction")
        lengths[p] = length
        previous = p
    return lengths


def _validate_query_bodies(plan, nodes, expressions, width, route, lengths):
    """Check the acquisition of every query against its stored association and readout.

    Trajectory: one Experiment ``trajectory`` whose batch acquires once
    (``observation_kind="trajectory"``) the body allocate, prepare, H, the
    gap nodes in power order and release, with one point ``power_<p>`` per
    distinct power at the boundary after the preparation, the H and the
    gaps up to p, reading the ancilla X and Y labels; every query names
    that Experiment and its power's point. Sampled and RWPE: each query's
    batch runs allocate, prepare, H, ``selected_<p>``, ``feedback_<i>``, H,
    the measurement and release, with the stored constant quadrature (the
    symbolic ``feedback`` parameter for RWPE) and the Plan's shots times the
    query's multiplicity as count repetitions.
    """
    from nwqlib.ir import (
        BlockCall,
        Constant,
        ExprRef,
        Measure,
        MeasurementBatch,
        ParameterRef,
        Sequence,
    )
    from nwqlib.ir.expressions import number
    from .method import METHOD, trajectory_labels

    rec = plan.reconstruction
    allocate = ("allocate_ancilla",) + (("allocate_system",) if width else ())
    release = ("release_ancilla",) + (("release_system",) if width else ())
    prepare = ("prepare",) if width else ()
    if route == "trajectory":
        (experiment,) = plan.experiments
        batch = nodes.get(experiment.batch)
        readout = experiment.readout
        body = None if not isinstance(batch, MeasurementBatch) else nodes.get(batch.body)
        position, points = len(prepare) + 1, []
        for power in rec.powers:
            position += lengths[power.power]
            points.append((f"power_{power.power}", position, "pauli_expectation",
                           trajectory_labels(width)))
        # The expected gap nodes follow from the stored powers, not from the
        # nodes that happen to be present.
        gaps = tuple(f"gap_{power}" for power, length in lengths.items() if length)
        if (
            experiment.name != "trajectory"
            or not isinstance(batch, MeasurementBatch)
            or experiment.setting_index != 0
            or len(batch.settings) != 1
            or batch.axes
            or batch.settings[0].bindings
            or batch.settings[0].label != "trajectory"
            or batch.settings[0].metadata.format != METHOD
            or batch.settings[0].metadata.data != rec.target
            or batch.repetitions != 1
            or batch.observation_kind != "trajectory"
            or readout is None
            or tuple((p.id, p.position, p.kind, p.labels) for p in readout.positions) != tuple(points)
            or any(p.view is not None for p in readout.positions)
            or readout.population != "unconditional"
            or not isinstance(body, Sequence)
            or body.children != (*allocate, *prepare, "h", *gaps, *release)
            or any(q.acquisition != "trajectory" or q.point != f"power_{q.power}"
                   or q.phase_shift not in (0, -pi / 2) for q in rec.queries)
        ):
            raise ValueError("QPE trajectory differs from its stored powers, points and queries")
        return
    for index, (query, experiment) in enumerate(zip(rec.queries, plan.experiments, strict=True)):
        batch = nodes.get(experiment.batch)
        readout = experiment.readout
        if (
            query.acquisition != query.experiment
            or query.point is not None
            or not isinstance(batch, MeasurementBatch)
            or experiment.setting_index != 0
            or len(batch.settings) != 1
            or batch.axes
            or batch.settings[0].bindings
            or batch.settings[0].label != query.experiment
            or batch.settings[0].metadata.format != METHOD
            or batch.settings[0].metadata.data != rec.target
            or batch.repetitions != plan.shots * query.multiplicity
            or batch.observation_kind != "counts"
            or readout is None
            or readout.amplitudes is not None
            or readout.position is not None
            or readout.positions
            or readout.labels
            or readout.qubits != ()
            or readout.population != "unconditional"
        ):
            raise ValueError("QPE acquisition/readout differs from its selected query and options")
        body = nodes.get(batch.body)
        measurement = nodes.get(f"measure_{index}")
        if not isinstance(measurement, Measure) or (
            measurement.wire,
            measurement.result,
            measurement.basis,
        ) != ("ancilla", "readout", "computational"):
            raise ValueError("QPE measurement must keep the computational ancilla population")
        rotation = nodes.get(f"feedback_{index}")
        expression = expressions.get(
            "feedback_value" if plan.method.estimator == "rwpe" else f"quadrature_{index}"
        )
        valid_phase = (
            isinstance(expression, ParameterRef) and expression.parameter == "feedback"
            if plan.method.estimator == "rwpe"
            else isinstance(expression, Constant) and number(expression.value) == query.phase_shift
        )
        expected_body = (*allocate, *prepare, "h", f"selected_{query.power}", f"feedback_{index}",
                         "h", f"measure_{index}", *release)
        if (
            not isinstance(body, Sequence)
            or body.children != expected_body
            or not valid_phase
            or not isinstance(rotation, BlockCall)
            or rotation.signature != "feedback"
            or len(rotation.arguments) != 1
            or rotation.arguments[0].parameter != "angle"
            or tuple((port.port, port.wire) for port in rotation.ports)
            != (("register", "ancilla"),)
            or not isinstance(rotation.arguments[0].value, ExprRef)
            or rotation.arguments[0].value.expression
            != ("feedback_value" if plan.method.estimator == "rwpe" else f"quadrature_{index}")
        ):
            raise ValueError("QPE query power/phase differs from its stored native experiment")


def validate_selection(plan):
    """Check that a stored Plan is internally consistent before it is executed or analyzed.

    The checks cover the concrete Method type, the original problem and
    initial state, tau, the padded register size, the Hamiltonian or
    unitary kind, the aliasing flag tau*R < pi, the RWPE time schedule
    1/sigma_k and single shot, the SPE and RFE pairing of real and imaginary
    queries with their draw multiplicities (distinct settings, multiplicity
    above one only for SPE and RFE, counts repetitions of shots times
    multiplicity), each power's error allowance
    and, through validate_selected_body, the Program itself. A mismatch
    raises ValueError.
    """
    from .method import QCELS, SPE, RFE, RWPE
    from nwqlib.problems import Eigenproblem, SpectralEstimation

    if (
        type(plan) is not Plan
        or type(plan.method) not in (QCELS, SPE, RFE, RWPE)
        or not isinstance(plan.problem, (Eigenproblem, SpectralEstimation))
    ):
        raise ValueError("QPE requires its original scientific Problem and configured Method")
    r, m = plan.reconstruction, plan.method
    if type(r) is not QPEReconstruction or r.method_id != m.content_id:
        raise ValueError("QPE reconstruction differs from Method")
    original = plan.problem.A if isinstance(plan.problem, Eigenproblem) else plan.problem.operator
    initial = (
        m.initial_state if isinstance(plan.problem, Eigenproblem) else plan.problem.initial_state
    )
    if original.reference != r.original_target or initial.reference != r.original_preparation:
        raise ValueError("QPE selected input differs from the original target or preparation")
    if m.tau is not None and m.tau != r.tau:
        raise ValueError("QPE selected time differs from Method.tau")
    if r.preparation.basis.dimension != 1 << (plan.problem.dimension - 1).bit_length():
        raise ValueError("QPE selected register differs from the original padded dimension")
    if plan.problem.dimension == r.preparation.basis.dimension and (
        r.target != original.reference or r.preparation != initial.preparation
    ):
        raise ValueError("unpadded QPE must keep the original target and preparation")
    kind = (
        "unitary"
        if isinstance(plan.problem, SpectralEstimation) and plan.problem.unitary is not None
        else "hamiltonian"
    )
    if r.target_kind != kind:
        raise ValueError("QPE estimator differs from its Hamiltonian/unitary domain")
    sufficient = (kind == "unitary" or r.spectral_radius_bound is not None
                  and r.tau * r.spectral_radius_bound < pi)
    if r.aliasing_bound_sufficient != sufficient:
        raise ValueError("QPE domain evidence differs from its selected spectral bound and time")
    if m.estimator == "rwpe":
        from .numerical import rwpe_scale
        expected_times = tuple(1/rwpe_scale(m.prior_std, step) for step in range(max(1, m.max_steps)))
        if (kind != "hamiltonian" or tuple(q.power for q in r.queries) != expected_times
                or plan.execution == "quantum" and plan.shots != 1):
            raise ValueError("RWPE requires its continuous Hamiltonian times and one shot per update")
    elif any(type(query.power) is not int for query in r.queries):
        raise ValueError("static QPE estimators require integer powers")
    if m.estimator not in {"spe", "rfe"} and any(q.multiplicity != 1 for q in r.queries):
        raise ValueError("only SPE and RFE queries can combine random draws")
    if m.estimator in {"spe", "rfe"} and len(
        {(q.power, q.phase_shift) for q in r.queries}
    ) != len(r.queries):
        raise ValueError("Fourier acquisition settings must be distinct")
    if m.estimator == "rfe":
        if sum(q.multiplicity for q in r.queries) != 2 * m.num_samples:
            raise ValueError("RFE query population differs from its selected sample count")
        for real, imag in zip(r.queries[::2], r.queries[1::2], strict=True):
            if (real.power != imag.power or not 0 <= real.power < m.num_frequencies
                    or real.phase_shift != 0 or imag.phase_shift != -pi / 2
                    or real.multiplicity != imag.multiplicity):
                raise ValueError("RFE queries must pair real and imaginary tests of each drawn power")
    if m.estimator == "spe":
        if sum(q.multiplicity for q in r.queries) != 2 * m.num_samples:
            raise ValueError("SPE query population differs from its selected sample count")
        for real, imag in zip(r.queries[::2], r.queries[1::2], strict=True):
            if (real.power != imag.power or real.power < 1 or real.power % 2 != 1
                    or real.power > 2*m.fourier_degree+1
                    or real.phase_shift != 0 or imag.phase_shift != -pi/2
                    or real.multiplicity != imag.multiplicity):
                raise ValueError("SPE queries must pair real and imaginary tests of each drawn odd frequency")
    route = plan_route(plan)
    if route == "trajectory":
        expected = ("trajectory",)
    elif route == "classical" and m.estimator != "rwpe":
        expected = ("overlaps",)
        if any(q.acquisition != "overlaps" or q.point is not None for q in r.queries):
            raise ValueError("classical QPE queries belong to the one host-kernel Experiment")
    else:
        expected = tuple(q.experiment for q in r.queries)
        if any(q.acquisition != q.experiment or q.point is not None for q in r.queries):
            raise ValueError("a sampled QPE query is its own Experiment")
    if tuple(e.name for e in plan.experiments) != expected:
        raise ValueError("QPE experiments differ from selected query schedule")
    if any(
        p.backend == "trotter_error_budgeted" and p.total_error > m.controlled_power_error_budget
        for p in r.powers
    ):
        raise ValueError("controlled-power error exceeds selected allowance")
    selected = {item.content_id for item in plan.construction.selections}
    if any(
        p.selected_definition is not None and p.selected_definition not in selected
        for p in r.powers
    ):
        raise ValueError("power inventory must bind actual selected definitions")
    validate_selected_body(plan)


def query_at(plan, experiment, bindings=()):
    """Return (power, ancilla phase) for a named query, using the bound RWPE feedback."""
    selected = next((q for q in plan.reconstruction.queries if q.experiment == experiment), None)
    if selected is None:
        raise ValueError("QPE point is outside its finite query inventory")
    if plan.method.estimator != "rwpe":
        return selected.power, selected.phase_shift
    from nwqlib.ir.expressions import number

    values = {b.parameter: b.value for b in bindings}
    return selected.power, float(number(values["feedback"]))


class RWPEGaussian(Record):
    """Moment-matched Gaussian posterior N(mean, standard_deviation**2) for phi = -tau*E.

    Attributes:
        mean: Posterior mean of the unwrapped phase in radians.
        standard_deviation: Posterior width in radians,
            numerical.rwpe_scale(prior_std, processed steps).
    """

    mean: Real
    standard_deviation: Annotated[Real, Field(gt=0)]


class QPEExposure(Record):
    """Requested and received population of one count-mode SPE or RFE query.

    Attributes:
        experiment: Query name; its power and quadrature are the Plan's query
            and its contributing count identities are its samples'.
        multiplicity: Original random-draw multiplicity m.
        requested: Requested shots ``plan.shots * m``.
        received: Shots actually returned, 1 <= received <= requested.
    """

    experiment: Text
    multiplicity: PositiveInt
    requested: PositiveInt
    received: PositiveInt


class QPEAnalysis(Result):
    """QPE Result: the estimate, its scope and the samples it was computed from.

    Construction and uncertainty refer to the original Plan. The receipts of
    the contributing observations record the points actually executed.

    Attributes:
        estimator: ``qcels``, ``spe``, ``rfe`` or ``rwpe``.
        estimator_value: Kernel value, energy in operator units for a
            Hamiltonian or principal eigenphase in radians for a unitary.
            None when no estimate was identified.
        phase_turns: Eigenphase of U in turns, in [0, 1).
        interval: Nominal model interval, RWPE only.
        samples: Reduced ancilla means, one per acquired value.
        missing: Names of planned queries without data.
        complete: True when all planned data are present and an estimate
            exists. For RWPE, every planned step has been processed.
        gaussian: RWPE posterior moments, None for the other estimators.
        processed_steps: RWPE updates applied.
        controller_work: RWPE controller size units charged.
        stop_reason: Why acquisition stopped, why no estimate is identified,
            or, for QCELS, that the fit is aliased.
        analysis_settings: Reanalysis settings, such as a changed grid_size.
        fit: QCELS search record, None for the other estimators.
        exposure: Requested and received population of every count-mode SPE
            or RFE query, empty otherwise.
        requested_exposure_complete: Whether every count query returned its
            requested population, computed from all queries and separate from
            ``complete``; None without count-mode Fourier queries.
        minimum_effective_shots_per_draw: The exact rational min_q n_q/m_q as
            (numerator, denominator), an effective exposure rather than a
            fractional physical shot count; None without count queries.
        rfe_coordinate_variance_upper: Upward binary64 publication of the
            exact received-count proxy A = sum_u (m_u/M)**2 max(1/n_uR,
            1/n_uI), an upper bound on each real and imaginary coordinate
            proxy of the sampled Fourier coefficients in units of squared
            ancilla expectation (for SPE, 4*S**2*A bounds the CDF proxy with
            its weight sum S). It concerns the conditional shot contribution
            under independent stationary shots, not the random-draw term, and
            gives no energy confidence interval.
    """

    estimator: Literal["qcels", "spe", "rfe", "rwpe"]
    estimator_value: Real | None
    phase_turns: Annotated[Real, Field(ge=0, lt=1)] | None
    interval: QPEInterval | None
    samples: tuple[QPESample, ...]
    missing: tuple[Text, ...]
    complete: StrictBool
    gaussian: RWPEGaussian | None = None
    processed_steps: Count = 0
    controller_work: Count = 0
    stop_reason: Text | None = None
    analysis_settings: tuple[Binding, ...] = ()
    fit: QCELSFit | None = None
    exposure: tuple[QPEExposure, ...] = ()
    requested_exposure_complete: StrictBool | None = None
    minimum_effective_shots_per_draw: tuple[PositiveInt, PositiveInt] | None = None
    rfe_coordinate_variance_upper: Nonnegative | None = None

    def _summary_lines(self):
        """Report lines that print each estimate together with the scope limits it carries."""
        if self._plan is None:
            lines = [f"{self.estimator} estimator value: {self._scalar_text(self.estimator_value)}; output frame not attached"]
        elif self._plan.output.kind == "eigenphase":
            lines = [f"Eigenphase: {self._scalar_text(self.phase_turns)} turns"]
        else:
            lines = [f"Eigenvalue estimate: {self._scalar_text(self.estimator_value)}{self._unit_text()}"]
        lines.append(f"{self.estimator}: selected prepared population; ground identity not established")
        if self._plan is not None and self.estimator == "spe":
            from .numerical import spe_filter_domain

            lines.append(f"SPE declared ground-overlap lower bound: {self._plan.method.overlap_lower_bound:g}")
            method = self._plan.method
            delta, supported = spe_filter_domain(method.fourier_degree, method.filter_beta,
                                                 method.overlap_lower_bound)
            if supported:
                lines.append(f"Fourier filter transition half-width: {delta:.6g} radians. "
                             f"Conditional filter error bound: {3*method.overlap_lower_bound/8:.6g}.")
            lines.append(f"Finite Fourier draws: {method.num_samples}. The grid spacing does not "
                         "bound filter or sampling error.")
        if self._plan is not None:
            note = identification_note(self._plan)
            if note is not None:
                lines.append(note)
        if self.estimator == "qcels":
            lines.append("Finite complex-fit estimate; uncertainty interval unavailable for this acquisition/model.")
            lines.append("Finite search does not establish a global optimum, unique mode or mixed-spectrum identification.")
            if self.fit is not None:
                lines.append(f"Complex least-squares residual: {self.fit.residual:.6g}; "
                             f"effective grid: {self.fit.effective_grid}; "
                             f"objective evaluations: {self.fit.evaluations}")
        if self.estimator == "rfe":
            lines.append("Largest sampled Fourier coefficient. Uncertainty interval unavailable.")
            lines.append("The precision guarantee of Kshirsagar, Katabarwa and Johnson "
                         "arXiv:2209.11322v3 requires an eigenstate and sufficient independent "
                         "samples.")
        if self.interval is not None:
            interval = self.interval
            lines.append(f"Nominal {interval.method} interval [{self._scalar_text(interval.low)}, "
                         f"{self._scalar_text(interval.high)}]; {interval.frame}; {interval.interpretation}")
        if not self.complete:
            lines.append(f"Partial data: {len(self.missing)} missing queries")
        if self.stop_reason:
            lines.append("Stop: " + self.stop_reason)
        return lines

    def validate_plan(self, plan):
        """Check stored scalar/query meaning without another estimator or grid."""
        self._validate_common_plan(plan, Plan)
        validate_selection(plan)
        from nwqlib.problems import Eigenphase

        if (
            self.construction_id != plan.construction.content_id
            or self.estimator != plan.method.estimator
        ):
            raise ValueError("QPE result method/construction differs from its exact Plan")
        from .method import _analysis_configuration, _QCELS_ANALYSIS_SOURCE, _ANALYSIS_SOURCE
        from .numerical import QCELS_BRACKET_EVALUATIONS, qcels_grid_size

        settings = {item.parameter: item.value.value if isinstance(item.value, Float64) else item.value
                    for item in self.analysis_settings}
        if len(settings) != len(self.analysis_settings):
            raise ValueError("QPE analysis settings require distinct parameter names")
        config = _analysis_configuration(plan.method, settings)
        if self.estimator == "qcels":
            if self.origin is None or self.origin.analyzer != _QCELS_ANALYSIS_SOURCE:
                raise ValueError("unsupported QCELS analysis revision; use its original source/environment")
            if self.fit is not None:
                powers = tuple(q.power for q in plan.reconstruction.queries)
                size = qcels_grid_size(config.grid_size, max(powers)-min(powers))
                # G grid evaluations plus at most G refined local minima.
                if (self.fit.effective_grid != size
                        or self.fit.evaluations > (1 + QCELS_BRACKET_EVALUATIONS)*size):
                    raise ValueError("QCELS fit differs from its actual finite analysis grid")
        elif self.origin is None or self.origin.analyzer != _ANALYSIS_SOURCE:
            raise ValueError("unsupported QPE analysis revision; use its original source/environment")
        phase_output = isinstance(plan.output, Eigenphase)
        if self.estimator_value is not None:
            if plan.reconstruction.tau is None and not -pi <= self.estimator_value < pi:
                raise ValueError("unitary QPE estimator value requires its principal-radian branch")
            expected = (
                energy_phase_turns(self.estimator_value, plan.reconstruction.tau)
                if plan.reconstruction.tau is not None
                else (self.estimator_value / (2 * pi)) % 1.0
            )
            if self.phase_turns != expected:
                raise ValueError("QPE estimator scalar and phase disagree")
        frame = (
            "unwrapped turns around reported phase"
            if phase_output
            else "energy in requested operator units"
        )
        if self.interval is not None and self.interval.frame != frame:
            raise ValueError("QPE interval frame differs from its requested output")
        note = identification_note(plan)
        if self.interval is not None and note is not None and not self.interval.interpretation.startswith(note):
            raise ValueError("QPE interval must retain its unverified identification domain")
        if self.estimator == "rwpe":
            from .numerical import rwpe_scale
            if (self.processed_steps > plan.method.max_steps
                    or self.gaussian is not None and self.gaussian.standard_deviation
                       != rwpe_scale(plan.method.prior_std, self.processed_steps)):
                raise ValueError("RWPE Gaussian width differs from its processed one-bit updates")
        elif self.gaussian is not None or self.processed_steps:
            raise ValueError("only RWPE has a Gaussian controller state")
        queries = {query.experiment: query for query in plan.reconstruction.queries}
        contributions = set(self.contribution_ids)
        for sample in self.samples:
            query = queries.get(sample.experiment)
            if (
                query is None
                or sample.power != query.power
                or sample.point != query.point
                or query.point is None and self.estimator != "rwpe"
                and sample.phase_shift != query.phase_shift
                or not set(sample.contributions) <= contributions
            ):
                raise ValueError("QPE sample differs from its stored query/contribution inventory")
        counted = (self.estimator in {"spe", "rfe"} and plan.shots is not None
                   and not self.missing)
        if counted != bool(self.exposure) or (self.requested_exposure_complete is None) == counted:
            raise ValueError("QPE exposure facts belong exactly to a count-mode Fourier Result without missing queries")
        for item in self.exposure:
            query = queries.get(item.experiment)
            requested = None if query is None else plan.shots * query.multiplicity
            if (query is None or item.multiplicity != query.multiplicity
                    or item.requested != requested or not 1 <= item.received <= requested):
                raise ValueError(
                    f"QPE query {item.experiment} records {item.received} received shots "
                    f"for an admitted request of {requested}")

    @property
    def phase(self):
        return self.phase_turns

    @property
    def eigenvalue(self):
        if self.plan.reconstruction.tau is None:
            raise AttributeError("a unitary-only input has no Hamiltonian eigenvalue conversion")
        return self.estimator_value

    @property
    def value(self):
        from nwqlib.problems import Eigenphase

        return self.phase_turns if isinstance(self.plan.output, Eigenphase) else self.eigenvalue

    @model_validator(mode="after")
    def _analysis(self):
        """Enforce the estimator-specific result shape.

        QCELS keeps its fit record and has no interval. Only RWPE keeps
        Gaussian posterior moments. An absent estimate has no phase or
        interval, and a complete analysis has no missing query.
        """
        if self.estimator == "qcels":
            if self.interval is not None:
                raise ValueError("QCELS uncertainty interval is unavailable for the complex fit")
            if self.estimator_value is not None and self.fit is None:
                raise ValueError("QCELS estimate requires its finite complex-fit evidence")
        elif self.fit is not None:
            raise ValueError("complex-fit evidence belongs only to QCELS")
        if (
            (self.estimator_value is None) != (self.phase_turns is None)
            or self.estimator_value is None
            and self.interval is not None
        ):
            raise ValueError("absent estimate has no phase or interval")
        if self.complete and (self.missing or self.estimator_value is None):
            raise ValueError("complete analysis requires an estimate without missing queries")
        if self.gaussian is not None and self.estimator != "rwpe":
            raise ValueError("only RWPE keeps Gaussian posterior moments")
        return self


def energy_phase_turns(value, tau):
    """Return the eigenphase of U = exp(-i*tau*H) for energy ``value``, in turns in [0, 1).

    The phase angle is -tau*value, reduced to the principal branch
    [-pi, pi) and divided by 2*pi, then taken modulo one.
    """
    angle = -value * tau
    if not isfinite(angle):
        raise ValueError("energy phase intermediate is outside the finite scalar domain")
    principal = (angle + pi) % (2 * pi) - pi
    return (principal / (2 * pi)) % 1.0


def public_estimate(estimate, *, phase_output, tau=None):
    """Return (value, phase_turns, interval) for one kernel estimate, converting units once.

    ``value`` is the kernel value unchanged, energy for a Hamiltonian or
    principal eigenphase in radians for a unitary. ``phase_turns`` is the
    eigenphase of U in [0, 1), from energy_phase_turns for a Hamiltonian or
    phase/(2*pi) modulo one for a unitary. For Eigenphase output the
    interval endpoints are scaled to turns, by -tau/(2*pi) for energy, which
    reverses their order, or by 1/(2*pi) for radians. They are then shifted
    so that the scaled value lands on phase_turns, which keeps the interval
    unwrapped around the reported phase. For Eigenvalue output the interval
    stays in energy units. A missing estimate returns (None, None, None).
    """
    if estimate is None or estimate.value is None:
        return None, None, None
    phase = (
        (estimate.phase / (2 * pi)) % 1.0
        if tau is None
        else energy_phase_turns(estimate.value, tau)
    )
    interval = None
    if estimate.interval_low is not None:
        low, high = estimate.interval_low, estimate.interval_high
        if phase_output:
            scale = 1 / (2 * pi) if tau is None else -tau / (2 * pi)
            center = estimate.value * scale
            a, b = low * scale, high * scale
            low, high = min(a, b) + phase - center, max(a, b) + phase - center
        interval = QPEInterval(
            low=low,
            high=high,
            method=estimate.interval_method,
            frame="unwrapped turns around reported phase"
            if phase_output
            else "energy in requested operator units",
            interpretation=estimate.interpretation,
        )
    if not isfinite(estimate.value):
        raise ValueError("estimator value is outside finite scalar domain")
    return float(estimate.value), float(phase), interval


METHOD = Source(
    name="qpe",
    version="1",
    domain="four selected phase/energy estimators from scalar Hadamard observations",
    reference="nwqlib.algorithms.qpe. QCELS, SPE, RFE and RWPE numerical kernels.",
)

DESCRIPTOR = AlgorithmDescriptor(
    method=METHOD.name,
    version=METHOD.version,
    problem_families=("eigenproblem", "spectral_estimation"),
    output_families=("eigenphase", "eigenvalue"),
    access_families=("dense", "pauli"),
    resource_coverage=(
        "selected signed powers and actual finite observation populations",
        "explicit dense host size and compact Pauli product construction",
    ),
    evidence_coverage=("method-specific local intervals and common observation lineage",),
    limitations=(
        "single-mode fit/threshold/posterior does not establish a ground state or mixed-spectrum accuracy",
        "dense powers are synthesized exactly to rounding, while native floating-point error remains unbounded; explicit comparison is separate",
    ),
    references=REFERENCES,
    maintenance="NWQLib",
)


class QPEVerification(Record):
    """Explicit nominal spectral comparison in the estimator's target frame.

    SPE compares the lowest spectral cluster. Other estimators compare the
    largest prepared cluster. Numerical grouping does not prove degeneracy.

    Attributes:
        tolerance: Accepted component discrepancy in the requested output
            unit, the Problem's energy unit for an ``Eigenvalue`` output and
            turns of circular distance for an ``Eigenphase`` output.
        group_atol: Absolute tolerance for numerical spectral clustering, in
            the unit of the compared eigenvalues. For a Hamiltonian target it
            is in the Problem's energy unit and compares eigenvalues of H.
            For a unitary target it is in radians and compares the principal
            differences of eigenphase angles. It does not follow the output
            unit, so an ``Eigenphase`` output of a Hamiltonian still groups
            by energy.
        minimum_overlap: Required projector weight, defaulting to SPE's declared eta or .9 otherwise.
        materialize_preparation: Explicitly allow simulation of a supplied preparation to obtain its reference state.
        reuse_only: Require matching existing spectrum and reference data instead of computing them.
        max_bytes: Cap on known reference arrays.
        max_work: Cap on declared reference work.
    """

    tolerance: Nonnegative
    group_atol: Nonnegative = 1e-8
    minimum_overlap: Annotated[Real, Field(ge=0, le=1)] | None = None
    """Required projector weight, defaulting to SPE's declared eta or .9 otherwise."""
    materialize_preparation: StrictBool = False
    reuse_only: StrictBool = False
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_work: PositiveInt = 1_000_000_000

    @property
    def source(self):
        return Source(
            name="qpe.nominal_cluster_reference",
            version="3",
            domain="explicit nominal spectral cluster and QR projector",
            reference="numerical grouping/projector overlap; no exact degeneracy or ground-identity proof",
        )

    def verification_checks(self, result):
        """Return the ``component_error`` and ``overlap_deficit`` CheckSpecs of these options.

        ``verification.verification_checks`` defines both checks.
        """
        from .verification import verification_checks

        return verification_checks(self, result)
