"""Finite binary populations, conditional inference and affine readout correction.

A population has n0 zero and n1 one outcomes of a +/-1 variable, with sample
mean z=(n0-n1)/n and unbiased sample-mean variance 4*n0*n1/[n^2(n-1)]. The
selectable inferences are:

- ``hoeffding``: radius sqrt(2*log(2/alpha)/n). Hoeffding (1963),
  doi:10.1080/01621459.1963.10500830, Theorem 1,
  Eq. (2.3), p. 15, gives P(mean-mu >= t) <= exp(-2nt^2) for outcomes in
  [0,1]. The map x -> (x+1)/2 takes [-1,1] to [0,1] and halves every
  deviation (the substitution t -> t/(b-a) stated below Eq. (2.4)), so on
  [-1,1] the bound is exp(-nt^2/2). Adding the lower tail as in Eq. (1.4),
  p. 13, and solving 2exp(-nt^2/2)=alpha for t gives the radius.
- ``anytime_hoeffding``: the same radius at alpha_n=alpha/[n(n+1)] for every
  n, a union bound over n because sum_n 1/[n(n+1)]=1 (derived here).
- ``beta``: the conjugate posterior Beta(prior_alpha+n0, prior_beta+n1) for
  p=P(bit=0), mapped to 2p-1, with the equal-tail interval at alpha/2 and
  1-alpha/2. The lower endpoint inverts the regularized incomplete beta
  I_x(a,b) at alpha/2 and the upper endpoint inverts its complement
  1-I_x(a,b) at alpha/2. When a+b exceeds MAX_BETA_SHAPE_TOTAL the interval
  is unavailable, while the posterior mean and variance remain available.

Here alpha=delta/F, with delta the options' failure_probability and F the
number of predeclared settings, so all F intervals hold jointly with
probability at least 1-delta. The affine readout model z=a*mu+b,
a=(z0-z1)/2, b=(z0+z1)/2, with z0 and z1 the calibration means, is the
standard single-bit model, derived in docs/algorithms/expectation.md.

These operations consume supplied sufficient counts. They never acquire data,
infer independence from identifiers, or turn a statistical model into a total
physical guarantee. SciPy is loaded only for the selected scalar Beta quantiles.
"""

from itertools import product
from math import isfinite, log, nextafter, sqrt
from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from nwqlib.core.records import ContentID, Nonnegative, Rational, Real, Record, Source, Text
from nwqlib.operators.access import Count
from ._work import DEFAULT_MAX_INTEGER_BITS, ExactArithmetic


# Binary64 exactly represents every integer through 2**53. Reserve one endpoint
# for n+1 in the time-uniform schedule and keep all posterior count inputs exact.
MAX_BINARY_COUNT = (1 << 53) - 1
BinaryCount = Annotated[Count, Field(le=MAX_BINARY_COUNT)]

# Largest Beta shape total a+b whose equal-tail interval is published. The
# endpoint error of SciPy's betaincinv and betainccinv grows about in
# proportion to a+b. Against a 50-digit mpmath reference, the sampled
# posteriors stayed below 1e-3 posterior standard deviations up to
# a+b = 5e11 and reached 1.2e-3 at 1e12 (docs/ENGINEERING_CONSTANTS.md,
# "Binary inference representation").
MAX_BETA_SHAPE_TOTAL = 5e11

BINARY_SOURCE = Source(name="binary_inference", version="1",
    domain="supplied +/-1 sufficient counts; conditional statistical models and scalar binary64 evaluation",
    reference="Hoeffding doi:10.1080/01621459.1963.10500830 for [-1,1]; alpha_n=alpha/(n*(n+1)); Beta(alpha+n0,beta+n1)")


class BinaryInferenceOptions(Record):
    """Statistical model that turns measured binary counts into estimates and intervals.

    Build it with keyword arguments, for example
    `BinaryInferenceOptions(method="hoeffding", sampling_model="iid_bernoulli")`,
    and pass it as `ExpectationMethod(inference=...)` or to
    `result.analyze(inference=...)`. Every argument is optional, and the
    default returns empirical points without an interval. It applies to
    measured counts, so any other choice needs positive shots. It changes
    neither what is measured nor the physical accuracy, and its intervals
    describe sampling under the stated model only.

    Below, a population has n0 zero and n1 one outcomes, `n = n0 + n1`,
    delta is `failure_probability`, and `alpha = delta/F` is its share for
    each of the F predeclared settings, so that all F intervals hold jointly
    with probability at least `1 - delta`.

    Attributes:
        method: Default `"point"`, the empirical mean and variance without an
            interval. `"hoeffding"` gives a fixed-time interval of radius
            `sqrt(2*log(2/alpha)/n)` (Hoeffding (1963),
            doi:10.1080/01621459.1963.10500830, Theorem 1, Eq. (2.3), p. 15,
            rescaled to outcomes in [-1, 1] and made two-sided as in
            Eq. (1.4), p. 13). `"anytime_hoeffding"` gives the same radius at
            `alpha_n = alpha/(n*(n + 1))` for every n, a union bound over n.
            `"beta"` gives the posterior `Beta(prior_alpha + n0, prior_beta + n1)`
            of `p = P(bit = 0)`, mapped to `2*p - 1`, with an equal-tail
            interval whose credibility is not frequentist coverage. That
            interval is unavailable when the shape total exceeds `5e11`, and
            the posterior mean and variance remain
            ([Engineering constants](../../ENGINEERING_CONSTANTS.md#binary-inference-representation)).
        failure_probability: Default `0.05`, strictly between 0 and 1. Family
            failure probability delta, divided equally across the
            predeclared science and calibration settings.
        sampling_model: Default `"unknown"`. Assumed model of the shots, which
            NWQLib does not verify: `"iid_bernoulli"`,
            `"constant_conditional_mean"` or `"unknown"`. `"unknown"` keeps
            the empirical values and gives no interval.
            `"constant_conditional_mean"` does not supply the Beta likelihood.
        independent_populations: Default `False`. `True` declares the measured
            groups, or with mitigation the science and calibration settings,
            statistically independent. The variance sum across settings and a
            product Beta model need this assumption, which NWQLib does not
            deduce from the data.
        prior_alpha: Default `1.0`, positive. Beta prior shape for
            `p = P(bit = 0)`, added to the zero count n0.
        prior_beta: Default `1.0`, positive. Beta prior shape added to the one
            count n1.

    Raises:
        ValueError: If `method="beta"` is combined with
            `sampling_model="constant_conditional_mean"`, or if a method other
            than `"point"` has a `failure_probability` whose complement
            `1 - failure_probability` is not representable strictly between 0
            and 1.
    """

    method: Literal["point", "hoeffding", "anytime_hoeffding", "beta"] = "point"
    failure_probability: Annotated[Real, Field(gt=0, lt=1)] = 0.05
    sampling_model: Literal["unknown", "iid_bernoulli", "constant_conditional_mean"] = "unknown"
    independent_populations: StrictBool = False
    prior_alpha: Annotated[Real, Field(gt=0)] = 1.0
    prior_beta: Annotated[Real, Field(gt=0)] = 1.0

    @model_validator(mode="after")
    def _model(self):
        if self.method == "beta" and self.sampling_model == "constant_conditional_mean":
            raise ValueError("Beta inference requires an iid Bernoulli likelihood, not only a conditional mean")
        if self.method != "point" and not 0 < 1.0 - self.failure_probability < 1:
            raise ValueError("selected interval probability must be representable strictly between zero and one")
        return self


class BinaryReadoutMitigation(Record):
    """Readout calibration that corrects each measured parity with a fitted single-bit channel.

    Build it with keyword arguments, for example
    `BinaryReadoutMitigation(calibration_shots=512)`, and pass it as
    `ExpectationMethod(mitigation=...)` together with positive shots.
    `calibration_shots` is the only required argument. Each non-identity
    term is then measured by its own parity circuit, which collects the
    term's parity on its pivot, the highest-index qubit on which the term
    acts. Each pivot gets two calibration circuits, which prepare 0 and 1.
    For K terms and P distinct pivots, a complete run uses `K + 2*P`
    circuits and `K*shots + 2*P*calibration_shots` shots.

    The model `z = a*mu + b` maps the parity mean mu to the observed pivot
    mean z. The calibration means z0 and z1 give `a = (z0 - z1)/2` and
    `b = (z0 + z1)/2`, and the corrected mean is `mu = (z - b)/a`. The model
    assumes a stationary readout channel, transfer of the calibration to the
    science circuits and correct calibration preparation, which NWQLib does
    not verify. Mitigation excludes Beta inference.

    Attributes:
        calibration_shots: Required. Positive number of shots, at most
            `2**53 - 1`, of each zero and each one calibration circuit.
        minimum_contrast: Default `0.05`, in (0, 1]. Lower limit on
            `abs(a) = abs(z0 - z1)/2`. Below it the correction, and therefore
            the result's `value`, is unavailable. A negative contrast of
            larger magnitude is accepted. The default is an adjustable
            conditioning threshold
            ([Engineering constants](../../ENGINEERING_CONSTANTS.md#binary-inference-representation)).
    """

    calibration_shots: Annotated[BinaryCount, Field(gt=0)]
    minimum_contrast: Annotated[Real, Field(gt=0, le=1)] = 0.05


class BinaryPopulation(Record):
    """Actual counted data after reuse of a frozen preparation is removed.

    source_ids are distinct counted-data identities; observation_ids name their
    counted observations in that order. reused_observation_ids keep further
    observations of the same frozen sources without increasing sample size.
    preparation_ids keep the receipts associated with each counted source,
    including replay receipts. One hardware preparation may supply many fresh
    acquisitions, and many same-seed preparations may supply one frozen source.
    Supplied non-quantum data can omit preparation_ids. Equal counts from
    distinct sources remain distinct data, not proof of IID.
    """

    name: Text
    zeros: BinaryCount
    ones: BinaryCount
    source_ids: tuple[ContentID, ...]
    observation_ids: tuple[ContentID, ...]
    reused_observation_ids: tuple[ContentID, ...] = ()
    preparation_ids: tuple[tuple[ContentID, ...], ...] = ()

    @model_validator(mode="after")
    def _population(self):
        if self.zeros + self.ones > MAX_BINARY_COUNT:
            raise ValueError("binary inference accumulated count exceeds its exact binary64 integer domain")
        if (len(self.source_ids) != len(self.observation_ids)
                or len(set(self.source_ids)) != len(self.source_ids)
                or len(set(self.observation_ids)) != len(self.observation_ids)
                or len(set(self.reused_observation_ids)) != len(self.reused_observation_ids)
                or set(self.observation_ids) & set(self.reused_observation_ids)
                or (self.zeros + self.ones > 0 and not self.source_ids)):
            raise ValueError("binary populations require distinct actual source/observation identities")
        if self.preparation_ids and (len(self.preparation_ids) != len(self.source_ids)
                or any(len(set(group)) != len(group) for group in self.preparation_ids)):
            raise ValueError("binary source preparation associations must match the counted sources")
        return self


class BinaryInterval(Record):
    """Interval for one binary mean, separate from its empirical variance and its point estimate.

    `BinaryEstimate.interval` and `BinaryCorrection.interval` hold it.
    Binary64 evaluation is not a certified numerical enclosure. The fields
    below are read-only.

    Attributes:
        kind: `"fixed_time"` (Hoeffding endpoint coverage),
            `"time_uniform"` (anytime Hoeffding coverage) or `"bayesian"`
            (Beta posterior credibility), from the inference method.
        probability: Family probability `1 - delta`, with `delta` the
            options' `failure_probability`, strictly between 0 and 1.
        status: `"conditional"` with both endpoints, `"unavailable"`, or
            `"empty"` after a justified intersection with the physical
            parameter set, which is not missing data.
        lower: Lower endpoint, `None` unless `status` is `"conditional"`.
        upper: Upper endpoint, `None` unless `status` is `"conditional"`.
        reason: Text recorded with the interval, including why it is
            unavailable or empty.
        assumptions: The assumptions the interval depends on.
        family_size: Number F of predeclared settings, positive. Each
            interval is evaluated at `alpha = delta/F`, so all F hold jointly
            with probability at least `1 - delta`.
    """

    kind: Literal["fixed_time", "time_uniform", "bayesian"]
    probability: Annotated[Real, Field(gt=0, lt=1)]
    status: Literal["conditional", "unavailable", "empty"]
    lower: Real | None
    upper: Real | None
    reason: Text
    assumptions: tuple[Text, ...]
    family_size: Annotated[Count, Field(gt=0)]

    @model_validator(mode="after")
    def _bounds(self):
        if self.status == "conditional":
            if self.lower is None or self.upper is None or self.lower > self.upper:
                raise ValueError("available binary interval requires ordered finite endpoints")
        elif self.lower is not None or self.upper is not None:
            raise ValueError("unavailable or empty interval cannot contain numeric endpoints")
        return self


class BinaryEstimate(Record):
    """Quantities derived from the counts of one binary setting, with the empirical and posterior roles kept apart.

    `ExpectationStatistics.populations` holds one per measured label or
    setting, as the [Expectation
    guide](../../algorithms/expectation.md) describes. The fields below are
    read-only.

    Attributes:
        population: The counted data: `zeros` (n0) and `ones` (n1) of the
            distinct counted sources, with their identifiers.
        options_id: Content hash of the `BinaryInferenceOptions` used.
        source: The `Source` of the binary inference.
        raw_mean: Sample mean `z = (n0 - n1)/n` in [-1, 1], `None` when
            `n = 0`.
        empirical_variance: Exact rational sample-mean variance
            `4*n0*n1/(n**2*(n - 1))`, `None` for `n <= 1`.
        point: The Beta posterior mean of `2p - 1` for `beta` inference,
            `None` when its assumptions fail, and `raw_mean` otherwise.
        posterior_variance: Beta posterior variance of `2p - 1`, which is
            not a sampling variance.
        interval: The [`BinaryInterval`][nwqlib.evidence.binary.BinaryInterval]
            of the selected inference, `None` for point inference. A
            missing interval keeps its kind and the reason.
        unavailable: Reasons for quantities that could not be formed.
    """

    population: BinaryPopulation
    options_id: ContentID
    source: Source = BINARY_SOURCE
    raw_mean: Annotated[Real, Field(ge=-1, le=1)] | None
    empirical_variance: Rational | None
    point: Annotated[Real, Field(ge=-1, le=1)] | None
    posterior_variance: Annotated[Nonnegative, Field(le=1)] | None
    interval: BinaryInterval | None
    unavailable: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def _domains(self):
        """Reject stored values that do not follow from the counts they claim.

        raw_mean must be (n0-n1)/n and exist exactly when n>0.
        empirical_variance must equal 4*n0*n1/[n^2(n-1)] as an exact
        rational, checked by cross-multiplication, and exists only for n>1.
        A conditional interval must lie in [-1,1], the domain of a binary mean.
        """
        n = self.population.zeros + self.population.ones
        if (self.raw_mean is None) != (n == 0):
            raise ValueError("binary sample mean availability differs from its actual population")
        if n and self.raw_mean != (self.population.zeros - self.population.ones) / n:
            raise ValueError("binary sample mean differs from its exact counted population")
        if self.empirical_variance is not None:
            variance = self.empirical_variance
            if (n <= 1 or variance.numerator < 0
                    or variance.numerator * (n * n * (n - 1))
                    != variance.denominator * (4 * self.population.zeros * self.population.ones)):
                raise ValueError("binary empirical variance differs from its counted definition")
        if self.interval is not None and self.interval.status == "conditional" and (
                self.interval.lower < -1 or self.interval.upper > 1):
            raise ValueError("a binary mean parameter lies in [-1,1]")
        return self


class BinaryCorrection(Record):
    """Readout-corrected mean of one science term from the affine calibration model, with its local derivatives.

    `ExpectationStatistics.corrections` holds one per science term when
    readout mitigation is used. The model is `z = a*mu + b` with
    `a = (z0 - z1)/2` and `b = (z0 + z1)/2`, where z0 and z1 are the
    calibration means ([Expectation guide](../../algorithms/expectation.md)).
    The fields below are read-only.

    Attributes:
        science_id: Content hash of the science setting's counted population.
        zero_id: Content hash of the zero-state calibration population.
        one_id: Content hash of the one-state calibration population.
        options_id: Content hash of the options used.
        contrast: The contrast a, in [-1, 1], or `None`.
        offset: The offset b, in [-1, 1], or `None`.
        point: The corrected mean mu, a signed finite-sample estimator, not
            a physical probability or a bounded Pauli mean. It can lie
            outside [-1, 1].
        derivatives: The Jacobian of `point` in `(z, z0, z1)`. Its use in a
            variance is a delta-method approximation.
        interval_image: Image of the calibration box before intersection
            with the physical mean domain [-1, 1].
        interval: That intersection, a
            [`BinaryInterval`][nwqlib.evidence.binary.BinaryInterval].
        unavailable: Reasons for quantities that could not be formed.
        assumptions: The assumptions of the correction.
    """

    science_id: ContentID
    zero_id: ContentID
    one_id: ContentID
    options_id: ContentID
    contrast: Annotated[Real, Field(ge=-1, le=1)] | None
    offset: Annotated[Real, Field(ge=-1, le=1)] | None
    point: Real | None
    derivatives: tuple[Real, Real, Real] | None
    interval_image: tuple[Real, Real] | None
    interval: BinaryInterval | None
    unavailable: tuple[Text, ...]
    assumptions: tuple[Text, ...]

    @model_validator(mode="after")
    def _availability(self):
        if (self.point is None) != (self.derivatives is None):
            raise ValueError("available affine correction requires its complete local Jacobian")
        if self.point is not None and (self.contrast is None or self.contrast == 0 or self.offset is None):
            raise ValueError("available affine correction requires a nonzero contrast and finite offset")
        if self.interval_image is not None and self.interval_image[0] > self.interval_image[1]:
            raise ValueError("calibration box image endpoints must be ordered")
        if self.interval is not None and self.interval.status == "conditional" and (
                self.interval.lower < -1 or self.interval.upper > 1):
            raise ValueError("corrected physical-mean parameter interval must lie in [-1,1]")
        return self


def _kind(options):
    """Return the BinaryInterval kind of an interval-producing inference method."""
    return {"hoeffding": "fixed_time", "anytime_hoeffding": "time_uniform", "beta": "bayesian"}[options.method]


def _interval(options, family_size, *, reason, assumptions, bounds=None, empty=False):
    """Build the BinaryInterval of the selected method at family probability 1-delta.

    bounds gives a conditional interval, empty an empty one, and neither an
    unavailable one whose reason explains why.
    """
    probability = 1.0 - options.failure_probability
    if not 0 < probability < 1:
        raise ValueError("selected interval probability is not representable strictly between zero and one")
    return BinaryInterval(kind=_kind(options), probability=probability,
        status="empty" if empty else "conditional" if bounds is not None else "unavailable",
        lower=None if bounds is None else bounds[0], upper=None if bounds is None else bounds[1],
        reason=reason, assumptions=assumptions, family_size=family_size)


def infer_binary(population, *, options, family_size, fixed_time, fixed_time_reason,
                 applicability_reason, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
    """Infer one selected finite population without acquisition or count replay.

    fixed_time and applicability_reason are facts supplied by the actual
    acquisition owner. They are not inferred from counts or options labels.
    A direct caller is responsible for their scientific validity/provenance.

    Every interval uses alpha=delta/F, with delta the options'
    failure_probability and F=family_size, so that a union bound over the F
    predeclared science and calibration settings of one Plan gives joint
    probability at least 1-delta. The ``hoeffding`` and ``anytime_hoeffding``
    radii and the Beta posterior are those of the module docstring.

    Args:
        population: BinaryPopulation with the counts n0 and n1.
        options: BinaryInferenceOptions selecting the method and its premises.
        family_size: Number F of predeclared settings sharing delta.
        fixed_time: Whether the data are one complete original acquisition,
            the premise of the fixed-time Hoeffding interval.
        fixed_time_reason: Why fixed_time holds or fails.
        applicability_reason: Reason the interval is unavailable for this
            population, such as missing receipts, or None.
        max_integer_bits: Integer width limit of the exact rational arithmetic.

    Returns:
        BinaryEstimate with the raw mean (n0-n1)/n, the exact sample-mean
        variance 4*n0*n1/[n^2(n-1)] for n>1, the point (the posterior mean
        for Beta), the Beta posterior variance of 2p-1, the interval, and the
        reasons for any unavailable quantity.
    """
    if type(population) is not BinaryPopulation or type(options) is not BinaryInferenceOptions:
        raise TypeError("binary inference requires concrete population and options records")
    if type(family_size) is not int or not 1 <= family_size <= MAX_BINARY_COUNT:
        raise ValueError("binary inference requires a finite predeclared family size")
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    n = population.zeros + population.ones
    mean = point = posterior_variance = variance = interval = None
    unavailable = []
    if n:
        raw = arithmetic.divide(arithmetic.fraction(population.zeros - population.ones), arithmetic.fraction(n))
        mean = point = float(raw)
        if n > 1:
            numerator = arithmetic.integer_product(arithmetic.integer_product(4, population.zeros), population.ones)
            denominator = arithmetic.integer_product(arithmetic.integer_product(n, n), n - 1)
            v = arithmetic.divide(arithmetic.fraction(numerator), arithmetic.fraction(denominator))
            variance = Rational(numerator=v.numerator, denominator=v.denominator)
    else:
        unavailable.append("no nonempty observed population")
    if options.method != "point":
        assumptions = (
            "fixed target and the selected sampling model are unverified premises",
            "finite family was declared before observing outcomes; no cross-Plan winner coverage",
            "scalar binary64 interval evaluation is not a certified arithmetic bound",
        )
        reason = applicability_reason
        if not n:
            reason = "no nonempty observed population"
        elif options.sampling_model == "unknown":
            reason = "sampling model is unspecified"
        elif options.method == "hoeffding" and not fixed_time:
            reason = fixed_time_reason
        elif options.method == "beta" and family_size > 1 and not options.independent_populations:
            reason = "joint Beta family needs the explicitly selected independent-population/product-prior model"
        # alpha=delta/F for delta=failure_probability and F predeclared
        # settings, so all F intervals hold jointly with probability >=1-delta.
        alpha = options.failure_probability / family_size
        if alpha == 0:
            reason = "family error probability is unrepresentable in binary64"
        if reason is not None:
            interval = _interval(options, family_size, reason=reason, assumptions=assumptions)
            if options.method == "beta":
                point = None
                unavailable.append("selected posterior point unavailable: " + reason)
        elif options.method == "beta":
            # Conjugate update: prior Beta(prior_alpha, prior_beta) on p=P(bit=0)
            # and n0 zeros, n1 ones give Beta(prior_alpha+n0, prior_beta+n1).
            # Each addition must change both summands in binary64. Otherwise a
            # positive prior or count would silently vanish. The tail must
            # satisfy 0 < alpha/2 < 1-alpha/2 < 1 in binary64, which keeps
            # alpha/2 above 2**-54 (about 5.6e-17). That is the tail range of
            # the endpoint accuracy measurement behind MAX_BETA_SHAPE_TOTAL,
            # whose smallest tail is 6e-17.
            a, b = options.prior_alpha + population.zeros, options.prior_beta + population.ones
            total = a + b
            tail = alpha / 2
            represented = (isfinite(total) and total > 0 and total != a and total != b and 0 < tail < 1 - tail < 1
                and (not population.zeros or (a != options.prior_alpha and a != float(population.zeros)))
                and (not population.ones or (b != options.prior_beta and b != float(population.ones))))
            if not represented:
                reason = "Beta posterior shapes or tail probabilities cannot keep the selected positive inputs in binary64"
                point = None
                unavailable.append(reason)
                interval = _interval(options, family_size, reason=reason, assumptions=assumptions)
            else:
                bounds = None
                if total <= MAX_BETA_SHAPE_TOTAL:
                    from scipy.special import betainccinv, betaincinv

                    # The upper endpoint solves 1-I_x(a,b) = alpha/2 directly.
                    # Passing 1-alpha/2 to betaincinv would round it to a
                    # multiple of 2**-53, an error of up to 2**-54 in the upper
                    # tail, which doubles a tail just above 2**-54.
                    lo, hi = float(betaincinv(a, b, tail)), float(betainccinv(a, b, tail))
                    if not (isfinite(lo) and isfinite(hi) and 0 <= lo <= hi <= 1):
                        raise ValueError("selected Beta kernel returned invalid posterior endpoints")
                    bounds = (2 * lo - 1, 2 * hi - 1)
                p = a / total
                point = 2 * p - 1
                # Var(2p-1)=4ab/[(a+b)^2 (a+b+1)] under Beta(a, b).
                posterior_variance = 4 * p * (b / total) / (total + 1)
                if not isfinite(posterior_variance) or posterior_variance <= 0:
                    reason = "positive Beta posterior variance is unrepresentable in binary64"
                    point = posterior_variance = None
                    unavailable.append(reason)
                    interval = _interval(options, family_size, reason=reason, assumptions=assumptions)
                elif bounds is None:
                    interval = _interval(options, family_size, assumptions=assumptions,
                        reason=f"Beta shape total a+b exceeds {MAX_BETA_SHAPE_TOTAL:g}, the largest total whose "
                            "quantile endpoints were validated to 1e-3 posterior standard deviations")
                else:
                    interval = _interval(options, family_size,
                        reason="equal-tail Beta posterior in p=P(bit=0), mapped by 2p-1; credibility is not coverage",
                        assumptions=(*assumptions, "iid Bernoulli likelihood and selected positive Beta prior"),
                        bounds=bounds)
        else:
            # On [-1,1], P(abs(z-mu)>=t)<=2exp(-n t^2/2) (Hoeffding 1963,
            # doi:10.1080/01621459.1963.10500830,
            # Theorem 1, Eq. (2.3), p. 15, rescaled as in the module docstring). Setting
            # the right side to alpha gives t=sqrt(2*log(2/alpha)/n). The anytime
            # option uses alpha/[n(n+1)] in place of alpha.
            # log(n)+log(n+1) avoids constructing n*(n+1) in binary64.
            logarithm = log(2.0) - log(alpha)
            if options.method == "anytime_hoeffding":
                logarithm += log(n) + log(n + 1)
            radius = sqrt(2 * logarithm / n)
            lower, upper = max(-1.0, mean - radius), min(1.0, mean + radius)
            interval = _interval(options, family_size, assumptions=assumptions,
                reason=("Hoeffding endpoint for the prescribed complete acquisition; intersection with [-1,1]"
                    if options.method == "hoeffding" else
                    "union bound with alpha_n=alpha/[n(n+1)] at every n; intersection with [-1,1]"),
                bounds=(lower, upper))
    return BinaryEstimate(population=population, options_id=options.content_id,
        raw_mean=mean, empirical_variance=variance, point=point, posterior_variance=posterior_variance,
        interval=interval, unavailable=tuple(unavailable))


def _float(value, arithmetic, *, side=None):
    """Convert a bounded exact scalar, with outward rational endpoint rounding."""
    try:
        result = float(value)
    except OverflowError:
        return None
    if not isfinite(result) or (result == 0 and value != 0):
        return None
    if side == "lower" and not arithmetic.le(arithmetic.fraction(result), value):
        result = nextafter(result, float("-inf"))
    elif side == "upper" and not arithmetic.le(value, arithmetic.fraction(result)):
        result = nextafter(result, float("inf"))
    return result if isfinite(result) else None


def correct_binary(science, zero, one, *, options, inference, applicability_reason,
                   max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
    """Correct one mean and jointly propagate its explicitly supplied rectangle.

    Model. A stationary single-bit readout channel maps a true +/-1 mean mu
    to the observed mean z=a*mu+b. Preparing the pivot in zero (mu=+1) and
    in one (mu=-1) gives the calibration means z0=a+b and z1=b-a, so
    a=(z0-z1)/2 is the contrast and b=(z0+z1)/2 the offset.

    Calibration fields keep distinct data identities even when their values
    happen to agree. Shared-population derivatives are composed by the
    existing linear_variance owner, and this function never assumes their
    independence. Stored inference options and interval families must agree
    before their rectangle can be composed. Correction never changes its
    statistical model.

    The corrected point is mu=(z-b)/a=(2z-z0-z1)/(z0-z1), with derivatives
    (2/d, -(1+mu)/d, (mu-1)/d) in (z, z0, z1) for d=z0-z1. mu is
    linear-fractional, hence quasilinear wherever its denominator keeps one
    sign. Once the joint rectangle's denominator interval excludes zero, the
    extremes over the rectangle therefore lie at its eight corners. That
    image is kept before its intersection with [-1,1].

    Args:
        science: BinaryEstimate of the science setting (mean z).
        zero: BinaryEstimate of the zero calibration at the same pivot (z0).
        one: BinaryEstimate of the one calibration at the same pivot (z1).
        options: BinaryReadoutMitigation with the minimum contrast.
        inference: BinaryInferenceOptions that produced the three estimates.
        applicability_reason: Reason correction is unavailable, or None.
        max_integer_bits: Integer width limit of the exact rational arithmetic.

    Returns:
        BinaryCorrection with the contrast a, offset b, corrected point mu,
        its three derivatives, the corner image of the joint rectangle, that
        image intersected with [-1,1], and the reasons for any unavailable part.
    """
    if any(type(value) is not BinaryEstimate for value in (science, zero, one)):
        raise TypeError("binary correction requires three actual binary estimates")
    if type(inference) is not BinaryInferenceOptions:
        raise TypeError("binary correction requires concrete inference options")
    if type(options) is not BinaryReadoutMitigation or inference.method == "beta":
        raise ValueError("binary affine mitigation has no selected Beta posterior model")
    if any(value.options_id != inference.content_id for value in (science, zero, one)):
        raise ValueError("binary correction inference options differ from its stored estimates")
    intervals = tuple(value.interval for value in (science, zero, one))
    if inference.method == "point":
        compatible = all(interval is None for interval in intervals)
    else:
        compatible = all(interval is not None and interval.kind == _kind(inference)
            and interval.probability == 1.0 - inference.failure_probability for interval in intervals)
        compatible = compatible and len({interval.family_size for interval in intervals}) == 1
    if not compatible:
        raise ValueError("binary correction inference intervals differ in kind, probability or family")
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    assumptions = (
        "same stationary affine readout channel transfers from zero/one calibration to the science pivot",
        "calibration preparation and spectator/gate effects are unverified; fit bias is not removed by sampling intervals",
        "Jacobian variance is a first-order delta-method approximation, not total physical variance",
        "an explicit backend noise model or hardware calibration does not by itself establish error reduction",
    )
    point = contrast = offset = derivatives = interval = image = None
    unavailable = []
    missing = any(value.raw_mean is None for value in (science, zero, one))
    reason = applicability_reason or ("missing nonempty science or calibration population" if missing else None)
    if not missing:
        z, z0, z1 = (arithmetic.fraction(value.raw_mean) for value in (science, zero, one))
        d = arithmetic.subtract(z0, z1)
        a = arithmetic.divide(d, arithmetic.fraction(2))
        b = arithmetic.divide(arithmetic.add(z0, z1), arithmetic.fraction(2))
        contrast, offset = _float(a, arithmetic), _float(b, arithmetic)
        if not arithmetic.le(arithmetic.fraction(options.minimum_contrast), arithmetic.absolute(a)):
            reason = reason or "absolute calibration contrast is below the selected minimum"
        if reason is None:
            mu = arithmetic.divide(arithmetic.subtract(z, b), a)
            jacobian = (arithmetic.divide(arithmetic.fraction(2), d),
                arithmetic.divide(-arithmetic.add(arithmetic.fraction(1), mu), d),
                arithmetic.divide(arithmetic.subtract(mu, arithmetic.fraction(1)), d))
            values = tuple(_float(value, arithmetic) for value in (mu, *jacobian))
            if any(value is None for value in values):
                reason = "affine corrected point or Jacobian is unrepresentable in binary64"
            else:
                point, derivatives = values[0], values[1:]
    if reason is not None:
        unavailable.append(reason)
    selected_interval = science.interval
    if selected_interval is not None:
        family = selected_interval.family_size
        box = intervals
        interval_reason = reason
        if any(value is None or value.status != "conditional" for value in box):
            interval_reason = interval_reason or "selected science/calibration joint rectangle is unavailable"
        if interval_reason is None:
            bounds = tuple((arithmetic.fraction(value.lower), arithmetic.fraction(value.upper)) for value in box)
            # The union bound makes the three intervals hold jointly, so on that
            # event d=z0-z1 lies in [z0_low-z1_high, z0_high-z1_low].
            dlow = arithmetic.subtract(bounds[1][0], bounds[2][1])
            dhigh = arithmetic.subtract(bounds[1][1], bounds[2][0])
            if dlow <= 0 <= dhigh:
                interval_reason = "joint calibration denominator interval contains zero"
            else:
                corners = []
                for z, z0, z1 in product(*bounds):
                    numerator = arithmetic.subtract(arithmetic.subtract(arithmetic.multiply(2, z), z0), z1)
                    corners.append(arithmetic.divide(numerator, arithmetic.subtract(z0, z1)))
                low = high = corners[0]
                for corner in corners[1:]:
                    if arithmetic.le(corner, low):
                        low = corner
                    if arithmetic.le(high, corner):
                        high = corner
                lo, hi = _float(low, arithmetic, side="lower"), _float(high, arithmetic, side="upper")
                if lo is None or hi is None:
                    interval_reason = "joint calibration interval image is unrepresentable in binary64"
                else:
                    image = (lo, hi)
                    clipped = (max(-1.0, lo), min(1.0, hi))
                    empty = clipped[0] > clipped[1]
                    interval = _interval(inference, family,
                        reason="joint rectangle image intersected with the physical mean domain [-1,1]",
                        assumptions=(*selected_interval.assumptions, *assumptions),
                        bounds=None if empty else clipped, empty=empty)
        if interval_reason is not None:
            interval = _interval(inference, family, reason=interval_reason,
                assumptions=(*selected_interval.assumptions, *assumptions))
    return BinaryCorrection(science_id=science.population.content_id,
        zero_id=zero.population.content_id, one_id=one.population.content_id, options_id=options.content_id,
        contrast=contrast, offset=offset, point=point, derivatives=derivatives,
        interval_image=image, interval=interval, unavailable=tuple(unavailable), assumptions=assumptions)
