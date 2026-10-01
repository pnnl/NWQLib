"""Explicit original-input QLS comparisons, sharing one reference solve."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import isfinite, sqrt
from typing import Literal
from pydantic import model_validator
from nwqlib.core.records import Float64, Nonnegative, PositiveInt, Record, Source, Text, Unit
from nwqlib.evidence.error_model import CheckDomain, CheckSpec, ErrorFrame, FramedFact
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.evidence.verification import witness_check_facts
from nwqlib.execution import KernelApplication, VerificationReceipt, verification_invocation
from nwqlib.ir import Binding
from .constants import QLS_SPECTRAL_COVERAGE_TOLERANCE


class QLSVerification(Record):
    """Reference comparisons for a `QLS` Result, with bounded reference work.

    Build it with keyword arguments and pass it to
    `result.verify(checks=...)`, for example
    `result.verify(checks=QLSVerification(comparisons=("spectral_domain",)))`.
    `comparisons` is the only required argument. The call returns
    `(receipt, facts)`, with one nonnegative dimensionless fact per
    comparison, in the given order, ready for
    `Certificate.with_verification` with the same options. The receipt also
    keeps the companion values, such as raw discrepancies and error
    budgets. The [QLS guide](../../algorithms/qls.md#explicit-verification-and-evidence)
    shows a complete check.

    All comparisons share one solve of the original `A/alpha` with the
    normalized b, and a spectral-only request solves nothing. Singular
    endpoints that the Plan already computed are reused. A comparison that
    needs only the norm of the solution, such as `"eq17"`, reuses the
    encoded reference norm that a grid or noisy-search norm model recorded
    from its own solve, which is that model's intermediate value, not
    independent evidence. A new spectral or solution reference needs a
    dense A, and compact Pauli or sparse input is never converted to a
    dense matrix. Without `relative_tolerance`, `"inverse_relative_error"`
    reports the raw relative error divided by the error budget
    `epsilon_inv + polynomial_kappa s delta / ||y_ref||`, with delta the
    phase-fit residual bound (zero for classical execution), s the
    polynomial rescale and `y_ref = (A/alpha)^-1 b/||b||`. A value of 0.386
    then means 0.386 of that budget, not 38.6% physical inverse error. A
    missing phase bound leaves that budget unknown. The automatic mass
    windows and the default `5*epsilon_inv` direction window are heuristics.
    These checks do not prove a bound on the total physical error or
    establish confidence coverage.

    Attributes:
        name: Default `"reference"`. Prefix of every reported fact. Each
            check is named `name + "." + comparison`.
        comparisons: Required. Distinct comparisons, at least one.
            `"spectral_domain"` checks that `alpha` and `kappa_be` cover the
            original singular endpoints. `"inverse_relative_error"` and
            `"inverse_success"` compare the `"qsvt_inverse"` output and its
            success mass with the shared reference solve.
            `"shortcut_direction"` compares a shortcut's unit direction
            modulo global phase. `"eq17"` compares a shortcut's success mass
            with Dalzell's Eq. (17) window (arXiv:2406.12086v2). The inverse
            comparisons need an inverse Plan, and the shortcut comparisons a
            shortcut Plan.
        relative_tolerance: Default `None`. Nonnegative threshold on the
            relative L2 inverse error. `None` reports the error divided by
            the method's error budget, checked against 1.
        direction_tolerance: Default `None`. Nonnegative threshold on the
            phase-aligned L2 direction error. `None` uses `5*epsilon_inv`.
        probability_tolerance: Default `None`. Nonnegative absolute
            threshold on the mass comparisons. `None` reports the
            discrepancy divided by a heuristic window, checked against 1.
        spectral_tolerance: Default `1e-9`, nonnegative. Window on the
            dimensionless spectral-domain deficit, wide enough for ordinary
            rounding of an automatic `kappa`. It is a numerical window, not a
            spectral theorem.
        max_work: Default `1e9` (`1_000_000_000`). Upper limit on the known
            reference work, checked before any solve.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Upper
            limit on the known reference arrays, checked before any solve.

    Raises:
        ValueError: If `comparisons` is empty or repeats a name.
    """

    name: Text = "reference"
    comparisons: tuple[
        Literal[
            "spectral_domain",
            "inverse_relative_error",
            "inverse_success",
            "shortcut_direction",
            "eq17",
        ],
        ...,
    ]
    relative_tolerance: Nonnegative | None = None
    direction_tolerance: Nonnegative | None = None
    probability_tolerance: Nonnegative | None = None
    spectral_tolerance: Nonnegative = QLS_SPECTRAL_COVERAGE_TOLERANCE
    max_work: PositiveInt = 1_000_000_000
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES

    @model_validator(mode="after")
    def _choices(self):
        if not self.comparisons or len(set(self.comparisons)) != len(self.comparisons):
            raise ValueError("verification requires distinct selected comparisons")
        return self

    @property
    def source(self):
        return Source(
            name="qls.reference." + self.name,
            version="2",
            domain="explicit original-input comparison; numerical method allowances and heuristic mass windows",
            reference="one shared solve(A/alpha,b_direction); missing original spectrum only when explicitly requested",
        )

    def verification_checks(self, result):
        """Return one CheckSpec per selected comparison, named ``name + "." + comparison``.

        Each check compares the comparison's top-level fact with the
        threshold that ``_threshold`` selects. With an explicit tolerance the
        fact is the raw discrepancy in its own metric. Without one, the
        inverse error and the mass comparisons report the raw discrepancy
        divided by its allowance, and the threshold 1 passes a discrepancy
        within that allowance. ``shortcut_direction`` keeps its raw
        phase-aligned discrepancy and, without ``direction_tolerance``,
        compares it with ``5 * epsilon_inv``. Every value is nonnegative and
        dimensionless. Each CheckSpec carries these options' identity, so a
        fact answers only this selection. When no vector comparison runs on
        a Result whose grid or noisy-search model recorded its encoded
        reference norm, each check other than ``spectral_domain`` names that
        recorded value as its reference: the model's own intermediate, not
        independent evidence.
        """
        plan = result.plan
        result.validate_plan(plan)
        _admit_solver(plan, self)
        recorded = (_recorded_reference_norm(result) is not None
                    and not {"inverse_relative_error", "shortcut_direction"} & set(self.comparisons))
        fields = dict(claim_id=plan.output.content_id, source=self.source, options_id=self.content_id,
            domain=CheckDomain(lower=0.), prerequisites=(), experiments=0,
            access=("existing QLS output vector or observed algorithm success mass",),
            classical_work="discrepancy, norms and allowance ratio on the original coordinates",
            data_description="scalar discrepancies, allowances and numerical call counts")
        return tuple(CheckSpec(name=self.name + "." + comparison,
                               frame=_frame(plan, self, self.name + "." + comparison),
                               threshold=_threshold(plan, self, comparison),
                               reference_work=RECORDED_REFERENCE_WORK
                               if recorded and comparison != "spectral_domain" else self.source.reference,
                               **fields)
                     for comparison in self.comparisons)


def verify(plan, result, *, checks):
    """Run one explicit ``QLSVerification`` and return ``(receipt, facts)``.

    ``facts`` holds one receipt-witnessed fact per selected comparison, in
    the order of ``checks.verification_checks(result)``, ready for
    ``Certificate.with_verification`` with the same options. The receipt
    also keeps the companion facts, such as raw discrepancies and
    allowances.
    """
    if type(checks) is not QLSVerification:
        raise TypeError("checks requires one explicit QLSVerification")
    return _verify(plan, result, checks)


RECORDED_REFERENCE_WORK = ("encoded reference norm recorded by the selected norm model's own solve; "
                           "the model's own intermediate, not independent evidence")


def _recorded_reference_norm(result):
    """Return the encoded reference norm that the Result's own classical evaluation recorded, or None.

    A grid or noisy-search norm model records ``encoded_reference_norm``
    in its application arguments (``host_planning.application_arguments``).
    Other routes made no such solve and return None.
    """
    for application in result.applications or ():
        for binding in application.arguments:
            if binding.parameter == "encoded_reference_norm":
                return binding.value.value
    return None


def _admit_solver(plan, choice):
    """Reject a comparison that belongs to the other QLS solver."""
    inverse = plan.method.solver == "qsvt_inverse"
    for name in choice.comparisons:
        if name != "spectral_domain" and (name in {"inverse_relative_error", "inverse_success"}) != inverse:
            raise ValueError("QLS comparison belongs to a different solver")


def _metric(choice, name):
    """Return ``(metric, heuristic)`` for the fact ``name`` of this selection.

    A top-level comparison fact takes the metric of its comparison. The
    inverse error and the mass comparisons report a ratio to their
    allowance unless the options supply an explicit tolerance, and the
    mass ratio is heuristic. A companion fact, whose name has one more
    dotted part, takes that last part as its metric.
    """
    comparison = name.removeprefix(choice.name + ".").split(".")[0]
    heuristic = comparison in {"inverse_success", "eq17"} and choice.probability_tolerance is None
    if name.count(".") > choice.name.count(".") + 1:
        return name.rsplit(".", 1)[1], heuristic
    if comparison == "spectral_domain":
        return "spectral_domain_deficit", heuristic
    if comparison == "shortcut_direction":
        return "phase_aligned_l2", heuristic
    if comparison == "inverse_relative_error":
        explicit = choice.relative_tolerance is not None
        return ("relative_l2" if explicit else "relative_error_over_method_allowance"), heuristic
    return ("heuristic_mass_window_discrepancy_ratio" if heuristic else "absolute_error"), heuristic


def _frame(plan, choice, name):
    """Return the dimensionless ErrorFrame of fact ``name`` on the original A and b."""
    return ErrorFrame(
        quantity=name,
        metric=_metric(choice, name)[0],
        unit=Unit(symbol="1", dimension="dimensionless"),
        scope=plan.problem.evidence_scope,
        conditioning="original A,b and actual selected polynomial/output acquisition",
    )


def _threshold(plan, choice, comparison):
    """Return the check threshold of one comparison.

    ``spectral_domain`` uses ``spectral_tolerance`` and ``shortcut_direction``
    uses ``direction_tolerance``, or ``5 * epsilon_inv`` without it. The
    inverse error and the mass comparisons use their explicit tolerance, or
    1 for their ratio to the allowance.
    """
    if comparison == "spectral_domain":
        return choice.spectral_tolerance
    if comparison == "shortcut_direction":
        explicit = choice.direction_tolerance
        return 5 * plan.method.epsilon_inv if explicit is None else explicit
    explicit = choice.relative_tolerance if comparison == "inverse_relative_error" else choice.probability_tolerance
    return 1.0 if explicit is None else explicit


def _target(plan, result, choice):
    """Admit the requested comparisons and return the acquired vector they compare.

    Inverse comparisons need an inverse Plan, and shortcut comparisons a
    shortcut Plan. Mass comparisons need an observed algorithm mass from a
    nonempty population. Vector comparisons need the Result's own stored
    vector in its declared frame, which keeps physical scale and phase for
    the inverse and is a unit vector modulo global phase for a shortcut. Returns
    ``(vector, manifest)``, or ``(None, None)`` when no vector is compared.
    """
    _admit_solver(plan, choice)
    inverse = plan.method.solver == "qsvt_inverse"
    vector = any(
        name in choice.comparisons for name in ("inverse_relative_error", "shortcut_direction")
    )
    mass = any(name in choice.comparisons for name in ("inverse_success", "eq17"))
    if mass:
        if result.algorithm_success_mass is None or result.mass_contribution_id is None:
            raise ValueError("verification requires an existing observed algorithm mass")
        if result.returned_shots is not None and result.returned_shots <= 0:
            raise ValueError("verification requires its actual nonempty returned shot population")
    if not vector:
        return None, None
    manifest = result.artifact
    frame, phase = ("physical", "physical") if inverse else ("unit", "modulo_global_phase")
    if (
        manifest is None
        or manifest.output.name != "solution"
        or manifest.output.kind != "vector"
        or manifest.output.basis != plan.problem.basis
        or manifest.output.frame != frame
        or manifest.output.global_phase != phase
    ):
        raise ValueError("verification requires its existing physical-phase or modulo-phase vector")
    return result.data.artifact(manifest).array, manifest


def _verify(plan, result, choice):
    """Run the explicitly requested spectral, inverse or direction reference comparisons under
    their work caps.
    """
    import numpy as np
    from nwqlib._linalg_laws import singular_values_work
    from nwqlib._numerics import stable_vector_norm
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery
    from nwqlib._numerics import normalized_matrix
    from .numerical import _rhs_direction

    checks = choice.verification_checks(result)
    target, manifest = _target(plan, result, choice)
    rec, d = plan.reconstruction, plan.problem.dimension
    bounds = (rec.sigma_min, rec.sigma_max)
    reused = all(value is not None for value in bounds)
    # A comparison that consumes only ||(A/alpha)^-1 b_hat|| reuses the norm
    # that the grid or noisy-search model recorded from its own encoded
    # solve. That value is the model's own intermediate, not independent
    # evidence. Vector comparisons and routes without a recorded norm solve.
    recorded = _recorded_reference_norm(result)
    vector = any(name in {"inverse_relative_error", "shortcut_direction"} for name in choice.comparisons)
    solve = any(name != "spectral_domain" for name in choice.comparisons) and (
        vector or recorded is None)
    spectrum = "spectral_domain" in choice.comparisons and not reused
    # Size laws for d = original dimension, checked before any reference
    # work. The shared solve charges d^3 for the dense solve of A/alpha, 8 d^2
    # for normalization and RHS expansion and 128 d for the vector
    # differences, norms and phase inner product. Its bytes allow three
    # complex d x d slots, two for the normalized copy and the LU copy that
    # np.linalg.solve factorizes and one to spare, and 48 complex d-vectors.
    # A missing spectrum charges the work of the SVD without vectors below
    # (_linalg_laws.singular_values_work) plus 3 d^2 units of margin for
    # NumPy's input conversion around the call. Its bytes are two complex
    # d x d slots plus the values.
    work = ((d**3 + 8 * d * d + 128 * d) * int(solve)
            + (singular_values_work(d) + 3 * d * d) * int(spectrum))
    data_bytes = 16 * ((3 * d * d + 48 * d) * int(solve) + (2 * d * d + d) * int(spectrum))
    if work > choice.max_work or data_bytes > choice.max_bytes:
        raise ValueError("selected QLS reference exceeds max_work or max_bytes")
    if (solve or spectrum) and plan.problem.A.reference.representation != "dense":
        raise ValueError(
            "QLS verification requires original dense input access for the selected missing "
            "spectral or solution reference; compact inputs are not implicitly densified. "
            "Use spectral_domain alone when this Plan already has singular endpoints, or "
            "explicitly select a dense-input Plan within its work and byte limits."
        )
    matrix = plan.problem.A.dense_array() if solve or spectrum else None
    counts = dict(
        solve_attempts=0,
        solve_completed=0,
        svd_attempts=0,
        svd_completed=0,
        norm_attempts=0,
        norm_completed=0,
        vector_differences=0,
        phase_inner_products=0,
        rhs_expansions=0,
        selected_spectrum_reuses=int(reused),
        recorded_reference_norm_reuses=int(not solve and recorded is not None
                                           and any(name != "spectral_domain" for name in choice.comparisons)),
        known_work=work,
        known_bytes=data_bytes,
    )
    parameters = dict(
        alpha=rec.alpha,
        kappa_be=rec.kappa_be,
        polynomial_kappa=rec.polynomial_kappa,
        rescale=rec.polynomial.rescale,
        epsilon_inv=plan.method.epsilon_inv,
        dimension=d,
    )
    parameters["kappa_source." + rec.kappa_source] = 1
    if spectrum:
        counts["svd_attempts"] += 1
        singular = np.linalg.svd(matrix, compute_uv=False)
        counts["svd_completed"] += 1
        bounds = float(singular[-1]), float(singular[0])
    # All requested solution comparisons share one solve of the normalized
    # original matrix. Spectral-only checks do not acquire that reference vector.
    reference = norm = discrepancy = None
    if counts["recorded_reference_norm_reuses"]:
        norm = recorded
        parameters["encoded_reference_norm"] = norm
    if solve:
        rhs = _rhs_direction(plan.problem.b)
        counts["rhs_expansions"] = int(plan.problem.b.preparation.implementation != "qiskit.direct")
        encoded = normalized_matrix(matrix, rec.alpha)
        counts["solve_attempts"] += 1
        reference = np.linalg.solve(encoded, rhs)
        counts["solve_completed"] += 1
        counts["norm_attempts"] += 1
        norm = stable_vector_norm(reference)
        counts["norm_completed"] += 1
        if not isfinite(norm) or norm <= 0:
            norm = None
        else:
            parameters["encoded_reference_norm"] = norm
    # Inverse error preserves physical phase and scale. Direction-only
    # shortcut comparison may align global phase before computing its L2 error.
    if target is not None and norm is not None:
        if "inverse_relative_error" in choice.comparisons:
            # Map the physical x to the frame of the reference solve,
            # y = (A/alpha)^{-1} b_hat = alpha x / ||b||. With ||b|| = m 2**e,
            # PhysicalScale(0.5, 1 - e) is 2**-e, so the composed factor is
            # alpha / (m 2**e) without forming ||b|| in binary64.
            rhs_scale = plan.problem.b.preparation.physical_scale
            scale = compose_recovery(
                (rec.alpha, rhs_scale.mantissa),
                PhysicalScale(mantissa=0.5, exponent=1 - rhs_scale.exponent),
            )
            actual = scale.apply_vector(target)
            if actual is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    difference = actual - reference
                counts["vector_differences"] += 1
        else:
            unit = reference / norm
            counts["phase_inner_products"] += 1
            phase = np.vdot(unit, target)
            actual = target * (np.conj(phase) / abs(phase)) if abs(phase) else target
            difference = actual - unit
            counts["vector_differences"] += 1
        if actual is not None:
            counts["norm_attempts"] += 1
            absolute = stable_vector_norm(difference)
            counts["norm_completed"] += 1
            if isfinite(absolute) and (absolute != 0 or not np.any(difference)):
                discrepancy = (
                    absolute / norm if "inverse_relative_error" in choice.comparisons else absolute
                )
                if not isfinite(discrepancy) or discrepancy == 0 and absolute != 0:
                    discrepancy = None
    values = _comparisons(
        plan,
        result,
        choice,
        bounds=bounds,
        norm=norm,
        discrepancy=discrepancy,
        parameters=parameters,
    )
    # Label empirical mass windows separately from numerical error criteria,
    # including the actual threshold used for each comparison.
    metrics = []
    for name, value in values.items():
        heuristic = _metric(choice, name)[1]
        frame = _frame(plan, choice, name)
        fact = Fact(
            quantity=name,
            unit=frame.unit,
            scope=frame.scope,
            availability="unknown" if value is None else "concrete",
            value=None if value is None else Float64(value=value),
            reason="required finite reference, phase certificate or valid mass allowance unavailable"
            if value is None
            else None,
            evidence=None
            if value is None
            else Evidence(
                kind="empirical_prediction" if heuristic else "numerical_estimate",
                source=choice.source,
            ),
        )
        metrics.append(FramedFact(frame=frame, bindings=(), fact=fact))
    for name in choice.comparisons:
        parameters["threshold." + name] = _threshold(plan, choice, name)
    application = KernelApplication(
        name=choice.name,
        implementation=choice.source,
        arguments=tuple(
            Binding(
                parameter=name, value=Float64(value=value) if isinstance(value, float) else value
            )
            for name, value in (*parameters.items(), *counts.items())
        ),
        facts=tuple(metrics),
    )
    receipt = VerificationReceipt(
        invocation_id=verification_invocation(),
        plan_id=plan.content_id,
        result_id=result.content_id,
        construction_id=plan._construction_id,
        artifact_ids=() if manifest is None else (manifest.content_id,),
        reference=choice.source,
        options_id=choice.content_id,
        applications=(application,),
    )
    return receipt, witness_check_facts(receipt, result, checks)


def _comparisons(plan, result, options, *, bounds, norm, discrepancy, parameters):
    """Nominal method criteria and separate raw/heuristic facts, without replay.

    NWQLib's derivations, with ``y = (A/alpha)^{-1} b_hat``, ``k`` the
    polynomial domain parameter, ``s`` the rescale, ``e = epsilon_inv`` and
    ``delta`` the phase-fit residual bound (zero for the classical model).
    All allowances require the spectral premise to hold within the
    spectral tolerance.

    - ``inverse_relative_error``: the executed polynomial is
      ``P/s + r`` with ``|r| <= delta``. Since ``|k x P(x) - 1| <= e`` gives
      ``|k P(x) - 1/x| <= e/abs(x)`` on the domain, the ideal polynomial
      contributes norm error at most ``e*||y||``. Recovery multiplies the
      phase residual by ``k s``, giving ``e + k*s*delta/||y||`` relatively.
    - ``inverse_success``: the ideal branch mass is ``p = (||y||/(k s))^2``
      and the executed branch differs from ``y/(k s)`` by at most
      ``a = e*||y||/(k*s) + delta``, so ``|mass - p| <= (2 sqrt(p) + a) a``. The shot
      term is four predicted-rate standard deviations over the returned
      shots.
    - ``eq17``: Dalzell's Eq. (17) window (arXiv:2406.12086v2) for the
      selected ``eta`` and ``t``, divided by ``s^2`` for the rescaled
      polynomial, with each endpoint's amplitude widened by ``delta``, plus
      ``QLS_EQ17_ABSOLUTE_WINDOW`` and the same shot term without a variance
      floor.

    These are component allowances under the stated premises, not
    certified total-error or confidence bounds.
    """
    from nwqlib.problems.inputs import compose_recovery
    from .constants import QLS_EQ17_ABSOLUTE_WINDOW, QLS_INVERSE_BINOMIAL_VARIANCE_FLOOR
    from .norm_search import _success_center, _success_window

    rec = plan.reconstruction
    values = {}
    # An explicit numeric kappa is an admitted premise, not its proof. Only
    # compatible selected/acquired numerical evidence can contradict it here;
    # absence never requests a hidden spectrum computation.
    lower, upper = (None, None) if bounds is None else bounds
    lower_deficit = None if lower is None else max(0.0, 1.0 - (lower / rec.alpha) * rec.kappa_be)
    upper_deficit = None if upper is None else max(0.0, upper / rec.alpha - 1.0)
    covered = all(
        value is None or value <= options.spectral_tolerance
        for value in (lower_deficit, upper_deficit)
    )
    deficit = (
        max(lower_deficit, upper_deficit)
        if lower_deficit is not None and upper_deficit is not None
        else None
    )
    delta = 0.0 if plan.execution == "classical" else rec.phase_error
    s, k, e = rec.polynomial.rescale, rec.polynomial_kappa, plan.method.epsilon_inv
    # The polynomial certificate is relative to each inverse component.
    # The ideal postselected amplitude has norm ||y||/(k*s).
    amplitude = (e * (norm / k) / s + delta
                 if norm is not None and covered and delta is not None else None)
    if amplitude is not None and (not isfinite(amplitude) or amplitude <= 0):
        amplitude = None
    relative_allowance = (
        e + (k / norm) * (s * delta) if norm is not None and covered and delta is not None else None
    )

    def put(name, value):
        values[name] = value if value is not None and isfinite(value) else None

    def ratio(raw, allowance):
        return (
            raw / allowance if raw is not None and allowance is not None and allowance > 0 else None
        )

    def shot_allowance(predicted, floor=0.0):
        """Return four standard deviations of an observed Bernoulli success fraction.

        The value is ``4 sqrt(max(p (1 - p), floor) / N)`` for predicted rate
        ``p`` and ``N`` returned shots. It is 0 without shot sampling and
        ``None`` when ``p`` lies outside ``[0, 1]``.
        """
        if result.returned_shots is None:
            return 0.0  # Source-selected absence of shot sampling, not total error zero.
        if predicted is None or not 0 <= predicted <= 1:
            return None  # A branch squared norm outside [0,1] is not a Bernoulli rate.
        return 4 * sqrt(max(predicted * (1 - predicted), floor) / result.returned_shots)

    for comparison in options.comparisons:
        name = options.name + "." + comparison
        value = None
        if comparison == "spectral_domain":
            value = deficit
            put(name + ".lower_deficit", lower_deficit)
            put(name + ".upper_deficit", upper_deficit)
        elif comparison == "shortcut_direction":
            value = discrepancy
        elif comparison == "inverse_relative_error":
            if discrepancy is not None:
                parameters["relative_error"] = discrepancy
            if options.relative_tolerance is not None:
                value = discrepancy
            else:
                put(name + ".raw_relative_error", discrepancy)
                put(name + ".method_allowance", relative_allowance)
                value = ratio(discrepancy, values[name + ".method_allowance"])
                if values[name + ".method_allowance"] is not None:
                    parameters["method_allowance"] = values[name + ".method_allowance"]
        elif comparison == "inverse_success" and norm is not None:
            predicted = compose_recovery(norm, (1.0, k), (1.0, s)).squared_as_float()
            raw = None if predicted is None else abs(result.algorithm_success_mass - predicted)
            put(name + ".predicted_mass", predicted)
            if predicted is not None:
                parameters["predicted_mass"] = predicted
            if options.probability_tolerance is not None:
                value = raw
            else:
                deterministic = (
                    (2 * sqrt(predicted) + amplitude) * amplitude
                    if predicted is not None and amplitude is not None
                    else None
                )
                if deterministic == 0:
                    deterministic = (
                        None  # Positive e cannot establish an exact zero allowance by underflow.
                    )
                stochastic = shot_allowance(predicted, QLS_INVERSE_BINOMIAL_VARIANCE_FLOOR)
                put(name + ".raw_discrepancy", raw)
                put(name + ".deterministic_allowance", deterministic)
                put(name + ".shot_allowance", stochastic)
                deterministic = values[name + ".deterministic_allowance"]
                allowance = (
                    deterministic + stochastic
                    if deterministic is not None and stochastic is not None
                    else None
                )
                value = ratio(raw, allowance)
        elif comparison == "eq17" and norm is not None:
            if plan.execution == "quantum":
                t = rec.t
            else:
                from .method import _validate_application

                _validate_application(plan, result.applications)
                t = next(
                    argument.value.value
                    for application in result.applications
                    for argument in application.arguments
                    if argument.parameter == "t_value"
                )
            center = _success_center(norm, t)
            lower, upper = (bound / s / s for bound in _success_window(center, rec.polynomial.eta))
            predicted = center / s / s
            parameters.update(
                t_value=t, predicted_mass=predicted, window_lower=lower, window_upper=upper
            )
            put(name + ".predicted_mass", predicted)
            put(name + ".window_lower", lower)
            put(name + ".window_upper", upper)
            raw = max(
                0.0, lower - result.algorithm_success_mass, result.algorithm_success_mass - upper
            )
            if options.probability_tolerance is not None:
                value = raw
            else:
                widened_lower = widened_upper = deterministic = None
                if covered and delta is not None:
                    # Reuse exact selected endpoints when there is no phase-fit
                    # discrepancy; sqrt followed by square would add rounding.
                    # Numerical sqrt/square rounding must not shrink the
                    # base interval that a nonnegative delta only widens.
                    widened_lower = (
                        lower if delta == 0 else min(lower, max(0.0, sqrt(lower) - delta) ** 2)
                    )
                    widened_upper = upper if delta == 0 else max(upper, (sqrt(upper) + delta) ** 2)
                    deterministic = (
                        lower - widened_lower
                        if result.algorithm_success_mass < lower
                        else widened_upper - upper
                        if result.algorithm_success_mass > upper
                        else 0.0
                    )
                stochastic = shot_allowance(predicted)
                put(name + ".raw_discrepancy", raw)
                put(name + ".phase_lower", widened_lower)
                put(name + ".phase_upper", widened_upper)
                put(name + ".deterministic_allowance", deterministic)
                put(name + ".numerical_allowance", QLS_EQ17_ABSOLUTE_WINDOW)
                put(name + ".shot_allowance", stochastic)
                allowance = (
                    deterministic + QLS_EQ17_ABSOLUTE_WINDOW + stochastic
                    if deterministic is not None and stochastic is not None
                    else None
                )
                value = ratio(raw, allowance)
        put(name, value)
    return values
