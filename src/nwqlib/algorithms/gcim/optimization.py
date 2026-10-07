"""Product-energy BFGS, the classical Taylor-model gradient and established finite generalized/insertion shifts.

The controller supplies every energy query, owns checkpoints and counts the
original global query allowance. This module submits no circuit itself.
"""

from math import fsum, isfinite, pi
from types import MappingProxyType
import numpy as np
import scipy.optimize as spo
from nwqlib.execution import ExecutionMode
from nwqlib.operators.access import DEFAULT_INPUT_BYTES
from nwqlib.subroutines.fermionic_circuits import (
    _require_default_double,
    _plan_generator_circuit,
    _GeneratorCircuitPlan,
    _UnsupportedDefaultGenerator,
)
from nwqlib.subroutines.fermionic_pool import (
    _active_occupation_blocks,
    _check_generator_work,
    _snapshot_generator,
)


def taylor_action_derivative(action, state, theta, steps, degree):
    """Return the fixed-step scaled Taylor action on ``state`` and its derivative in ``theta``.

    This differentiates all scaled Taylor steps with the step count fixed.
    ``action`` is the admitted generator action. The degree and step count
    must come from the same owner as the forward exponential
    (``fermionic_pool.GENERATOR_TAYLOR_DEGREE`` and ``_taylor_steps``). The
    value's multiply then add order matches ``_scaled_taylor_step`` with the
    same action. For a fixed step count ``s`` and ``h = theta/s``, each step is
    the Horner form of ``P_m(hA) = sum_{r<=m} (hA)**r/r!`` and its derivative
    in theta adds ``A v/(s r) + (h/r) A dv`` per Horner term. The derivative
    of the continuous polynomial model holds within each interval where
    ``_taylor_steps`` is constant. Step-count changes are piecewise
    boundaries, and the integer step selector itself is not differentiated.
    With zero steps the model is the identity: at a zero angle the
    derivative is the limit from the nonzero-angle branch, ``A state``, and at
    a nonzero angle whose steps underflow to zero the selected identity
    approximation is locally constant, so the derivative is zero. A zero
    generator has zero derivative in either branch.
    """
    if steps < 1:
        # The derivative at zero is the limit from the nonzero-angle branch.
        derivative = action(state) if theta == 0 else np.zeros_like(state)
        return state.copy(), derivative
    h = theta / steps
    value = state.copy()
    derivative = np.zeros_like(state)
    for _ in range(steps):
        base, dbase = value, derivative
        value, derivative = base.copy(), dbase.copy()
        for order in range(degree, 0, -1):
            av = action(value)
            ad = action(derivative)
            value = base + (h / order) * av
            derivative = dbase + av / (steps * order) + (h / order) * ad
    return value, derivative


def fixed_taylor_action(action, state, theta, steps, degree):
    """Apply the fixed-step Taylor polynomial of ``exp(theta A)`` to ``state``, unnormalized.

    With the forward factor's step count this is ``F_j(theta)``. A
    real-coefficient Taylor polynomial in anti-Hermitian A satisfies
    ``F_j^dagger = F_j(-theta_j)`` for a fixed step count, so the reverse sweep
    of ``normalized_taylor_energy_gradient`` applies it at ``-theta_j``. It is
    an adjoint, not an exact inverse, and its result is never normalized.
    """
    if steps == 0:
        return state.copy()
    value = state
    h = theta / steps
    for _ in range(steps):
        base = value
        value = base.copy()
        for order in range(degree, 0, -1):
            work = action(value)
            work *= h / order
            work += base
            value = work
    return value


def normalized_taylor_energy_gradient(
    reference, generators, theta, *, action, step_count, normalize,
    h0_action, degree,
):
    """Evaluate the normalized Taylor product energy and its real-parameter gradient.

    Store each factor's normalized output, raw norm and raw tangent, then
    apply the real normalization pullback in reverse. Adjoint Taylor actions
    use the forward factor's fixed step count and are left unnormalized.
    Energy and gradient share one Hamiltonian action.

    Derivation. Let H be the Hermitian operator applied by ``h0_action``
    and psi_0 the normalized reference. For j = 1..k in the order
    ``_vector`` applies the chain, set ``r_j = F_j(theta_j) psi_{j-1}``,
    ``n_j = ||r_j||`` and ``psi_j = r_j/n_j``. The objective is
    ``E(theta) = Re(psi_k^dagger H psi_k)``. Its last normalization is
    part of the computational graph, and the terminal adjoint is
    ``lambda_k = H psi_k``.

    Differentiating ``r/||r||`` gives the real normalization pullback
    ``lambda~_j = (lambda_j - psi_j Re(psi_j^dagger lambda_j))/n_j``.
    Then ``g_j = 2 Re<lambda~_j, F'_j psi_{j-1}>`` and
    ``lambda_{j-1} = F_j^dagger lambda~_j``. For the anti-Hermitian
    generators and real-coefficient Taylor factors,
    ``F_j^dagger = F_j(-theta_j)`` at the same fixed step count. The
    reverse state is never normalized, and the unused adjoint past
    factor 1 is not propagated.

    The gradient differentiates the continuous normalized polynomial
    model on each interval with fixed selected Taylor step counts.
    It does not differentiate binary64 rounding or the integer step
    selector. A binary64 evaluation of ``c0 + <H-c0 I>`` can round
    differently from direct ``<H>``. The caller therefore reports the
    energy evaluated with the supplied action and binds this gradient
    to that same objective.

    Cases. ``k = 0`` evaluates one reference energy and returns shape
    ``(0,)``. At ``theta = 0, s = 0`` the raw tangent is A times the input, at
    nonzero underflow-to-zero steps it is zero, and at a zero generator it
    is zero. A nonfinite or zero factor norm rejects.

    Cost. One energy-and-gradient evaluation stores the normalized forward
    states and raw parameter tangents, uses one Hamiltonian action, and
    differentiates each fixed-step Taylor factor and normalization.
    Admission includes both forward derivative actions and the unnormalized
    reverse actions (``adapt_records._energy_gradient_work``).
    """
    k = len(theta)
    if len(generators) != k:
        raise ValueError("one generator is required per angle")
    states = [np.asarray(reference, dtype=np.complex128)]
    tangents = []
    norms = np.empty(k, dtype=np.float64)
    steps = np.empty(k, dtype=np.int64)
    for j, (generator, angle) in enumerate(zip(generators, theta, strict=True)):
        steps[j] = step_count(generator, float(angle))
        raw, tangent = taylor_action_derivative(
            lambda v: action(generator, v), states[-1], float(angle),
            int(steps[j]), degree,
        )
        state, norm = normalize(raw)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("a Taylor factor needs finite positive norm")
        states.append(state)
        tangents.append(tangent)
        norms[j] = norm
        del raw, tangent, state
    last = states[-1]
    adjoint = h0_action(last)
    energy = float(np.vdot(last, adjoint).real)
    gradient = np.empty(k, dtype=np.float64)
    for j in range(k - 1, -1, -1):
        state = states[j + 1]
        pull = adjoint - state * float(np.vdot(state, adjoint).real)
        np.divide(pull.real, norms[j], out=pull.real)
        np.divide(pull.imag, norms[j], out=pull.imag)
        gradient[j] = 2.0 * float(np.vdot(pull, tangents[j]).real)
        if j:
            adjoint = fixed_taylor_action(
                lambda v: action(generators[j], v), pull, -float(theta[j]),
                int(steps[j]), degree,
            )
    return energy, gradient


class _EvaluationBudgetExhausted(Exception):
    """Raised by the controller's ``evaluate`` when a new energy query would exceed ``optimize_max_evaluations``."""


def _generator_shift_rule(
    generator,
    *,
    execution_mode=ExecutionMode.STATEVECTOR,
    compilation_plan=None,
    max_bytes=DEFAULT_INPUT_BYTES,
    max_products=1_000_000_000,
):
    """Resolve generalized shifts from small active occupation blocks once.

    The frequencies are eigenvalue gaps of -iA. Fermionic permutations
    preserve that spectrum, including occupied spectators. Only blocks up to
    dimension eight on at most eight active modes are admitted here. Pauli
    insertion shifts after the complete generator are valid without termwise
    commutation. Custom commuting generators use that rule without a
    spectral construction. Under SHOTS the insertion rule is returned
    without building the generalized rule. Its proxy K*sum(w^2), for K
    energy evaluations with signed coefficients w under uniform shot
    allocation, is smaller than the generalized rule's for every supported
    default GSD index class, and the returned mapping reports it. This is a
    worst-case variance/cost proxy with a common Hamiltonian-group factor,
    not an optimal allocation or a confidence guarantee.

    Generalized shift rule. Writing ``exp(theta A) = exp(i theta G)`` with
    ``G = -iA``, the energy is a trigonometric polynomial whose frequencies
    are the R distinct positive eigenvalue gaps of G (Wierichs et al.,
    arXiv:2107.12390v3, Sec. 2.1, Eqs. (1)-(3), and Sec. 3, Eq. (15)). Its odd
    part gives ``E'(0) = sum_mu w_mu (E(x_mu) - E(-x_mu))`` with
    ``sum_mu 2 sin(Omega_l x_mu) w_mu = Omega_l`` for each of the R
    frequencies, ``mu = 1..R`` (Sec. 3.2, Eqs. (16)-(18), and App. B.1 for
    unequal gaps). The shifts
    ``x_mu = (2 mu - 1) pi / (2 Omega_max)`` reduce to those of Eq. (24) when
    the gaps are equidistant.

    Pauli insertion rule. For a Pauli string, ``exp(i s pi/4 P) = (I + i s P)/sqrt(2)``
    and inserting that factor gives ``<[O, i P]> = E(+) - E(-)``. This is the
    two-eigenvalue rule of Schuld et al., arXiv:1811.11184v1, Sec. III.A,
    Theorem 1, Eq. (8), and Eqs. (13)-(14). NWQLib's step is where the factor
    goes. For ``A = i sum_k c_k P_k``, ``A`` commutes with ``exp(theta A)``, so
    ``dE/dtheta = <chi|[O, A]|chi>`` with ``chi`` the state after the complete
    exponential and ``O`` the observable seen from there. Inserting each
    ``P_k`` factor after the complete exponential then gives
    ``E' = sum_k c_k (E(+) - E(-))`` with no commutation needed among the
    ``P_k``. This is the derivative decomposition of Sec. III.B,
    Eqs. (16)-(17), with the Pauli terms as the unitaries.

    Outside SHOTS, the rule with fewer energy evaluations is chosen, with the
    signed weight square sum breaking ties.

    Returns:
        A read-only mapping. ``method`` is ``pauli_insertion_shift``, whose
        ``terms`` list each nonidentity ``pauli_label`` with its real
        ``coefficient`` ``c_k`` and ``term_index``, or ``generalized_shift``,
        with the ``shifts`` ``x_mu`` in radians, the ``weights`` ``w_mu`` and
        the ``frequencies``. Both give ``evaluations``, the number of energy
        queries per derivative, and the ``coefficient_convention`` that
        ``optimize_product`` applies.
    """
    if compilation_plan is None:
        generator = _snapshot_generator(generator, max_bytes=max_bytes, max_products=max_products)
    elif (
        type(compilation_plan) is not _GeneratorCircuitPlan
        or compilation_plan.generator is not generator
    ):
        raise ValueError("shift rule requires its exact admitted generator/compiler association")
    rows, q = generator.pauli_terms, generator.num_qubits
    # Untuned allowances for the insertion-rule table: per Pauli term its
    # q-character label and about 128 bytes of mapping entries, and q + 8
    # operations to read and copy it, plus a fixed 512 bytes and 32 operations.
    _check_generator_work(
        max_bytes,
        max_products,
        payload_bytes=len(rows) * (q + 128) + 512,
        work=len(rows) * (q + 8) + 32,
    )
    if any(c.real != 0 for _, c in rows):
        raise ValueError("shift rule requires an anti-Hermitian generator")
    elementary = tuple(
        MappingProxyType(
            {"pauli_label": p, "coefficient": float(complex(c).imag), "term_index": index}
        )
        for index, (p, c) in enumerate(generator.pauli_terms)
        if c != 0 and set(p) != {"I"}
    )
    selection_basis = (
        "uniform_allocation_variance_cost_proxy"
        if execution_mode is ExecutionMode.SHOTS
        else "energy_evaluation_count"
    )
    squared_weights = 2 * fsum(term["coefficient"] * term["coefficient"] for term in elementary)
    evaluations = 2 * len(elementary)
    insertion = {
        "method": "pauli_insertion_shift",
        "terms": elementary,
        "evaluations": evaluations,
        "signed_weight_square_sum": squared_weights if isfinite(squared_weights) else None,
        "uniform_allocation_variance_cost_proxy": evaluations * squared_weights
        if isfinite(evaluations * squared_weights)
        else None,
        "selection_basis": selection_basis,
        "coefficient_convention": "each term contributes coefficient times (energy_plus - energy_minus)",
    }
    if execution_mode is ExecutionMode.SHOTS:
        # All supported default GSD index classes select insertion under this proxy.
        # Recompare the cost if the exact compiler adds another generator class.
        return MappingProxyType(insertion)
    try:
        if compilation_plan is None or compilation_plan.kind == "commuting":
            _require_default_double(generator, max_bytes=max_bytes, max_products=max_products)
    except _UnsupportedDefaultGenerator:
        # Only an inapplicable default-family recipe can select this existing
        # insertion alternative. Budget/dependency failures propagate unchanged.
        plan = compilation_plan or _plan_generator_circuit(
            generator, max_bytes=max_bytes, max_products=max_products
        )
        if plan.kind != "commuting":
            raise ValueError("measured optimization requires a default GSD or commuting generator")
        return MappingProxyType(insertion)
    if compilation_plan is not None and compilation_plan.occupation_blocks:
        active, blocks = compilation_plan.active_modes, compilation_plan.occupation_blocks
    else:
        active, blocks = _active_occupation_blocks(
            generator, max_bytes=max_bytes, max_products=max_products
        )
    entries = sum(len(indices) ** 2 for indices, _ in blocks)
    eigen_count = sum(len(indices) for indices, _ in blocks)
    # Bytes: 64 per block entry for -i*B and eigvalsh workspace copies of
    # complex128 data, and 32 per eigenvalue. Work: k**3 per dense Hermitian
    # eigensolve of a k-by-k block, one pass over the entries, and a sort of
    # the eigenvalues.
    _check_generator_work(
        max_bytes,
        max_products,
        payload_bytes=64 * entries + 32 * eigen_count,
        work=sum(len(indices) ** 3 for indices, _ in blocks)
        + entries
        + eigen_count * (eigen_count.bit_length() + 1),
    )
    eigenvalues = [value for _, block in blocks for value in np.linalg.eigvalsh(-1j * block)]
    max_block = max((len(indices) for indices, _ in blocks), default=0)
    scale = max((abs(value) for value in eigenvalues), default=0.0)
    # Engineering merging window for repeated eigenvalues of bounded Hermitian blocks.
    tolerance = 16 * np.finfo(float).eps * max_block * scale

    def distinct(values):
        result = []
        for value in sorted(values):
            if not result or value - result[-1] > tolerance:
                result.append(float(value))
        return result

    spectrum = distinct(eigenvalues)
    # Every pair of distinct eigenvalues gives one candidate gap: 32 bytes
    # each, and a sort of the candidates.
    candidates = len(spectrum) * (len(spectrum) - 1) // 2
    _check_generator_work(
        max_bytes,
        max_products,
        payload_bytes=32 * candidates,
        work=candidates * (candidates.bit_length() + 2),
    )
    frequencies = distinct(
        spectrum[j] - left
        for i, left in enumerate(spectrum)
        for j in range(i + 1, len(spectrum))
        if spectrum[j] - left > tolerance
    )
    if not frequencies:
        return MappingProxyType(insertion)
    f = len(frequencies)
    # The f-by-f system sum_mu 2 sin(Omega_l x_mu) w_mu = Omega_l: 64 bytes per
    # matrix entry for the matrix and its LU copy, 128 per frequency for the
    # vectors, and f**3 for the dense solve plus the entrywise setup.
    _check_generator_work(
        max_bytes, max_products, payload_bytes=64 * f * f + 128 * f, work=f**3 + 8 * f * f + 8 * f
    )
    shifts = (2 * np.arange(len(frequencies)) + 1) * pi / (2 * max(frequencies))
    weights = np.linalg.solve(2 * np.sin(np.outer(frequencies, shifts)), frequencies)
    generalized = {
        "method": "generalized_shift",
        "shifts": tuple(map(float, shifts)),
        "weights": tuple(map(float, weights)),
        "frequencies": tuple(frequencies),
        "evaluations": 2 * len(shifts),
        "max_block_dimension": max_block,
        "active_mode_count": len(active),
        "spectral_roundoff_tolerance": tolerance,
        "signed_weight_square_sum": 2 * float(weights @ weights),
        "uniform_allocation_variance_cost_proxy": 4 * len(shifts) * float(weights @ weights),
        "selection_basis": selection_basis,
        "coefficient_convention": "each shift contributes weight times (energy_plus - energy_minus)",
    }
    return MappingProxyType(
        min(
            (insertion, generalized),
            key=lambda rule: (rule["evaluations"], rule["signed_weight_square_sum"]),
        )
    )


def optimize_product(*, options, execution, selected, starting, evaluate, completed_round, rules):
    """Run one fresh BFGS minimization and return ``(message, success)``.

    The caller keeps the best point and the query record. An exhausted
    evaluation allowance returns ``("evaluation_budget_exhausted", False)``.

    Zheng et al. (2024), arXiv:2312.07691v3, run their optimization rounds with
    SciPy BFGS (Table III caption, p. 8) on the ansatz parameters every m
    iterations (Appendix H). Classical (THEORY) execution evaluates the energy
    and its gradient together, one query per parameter point
    (``normalized_taylor_energy_gradient``), and BFGS reads both from that one
    evaluation. Circuit modes supply the measured gradient from the
    generalized or Pauli insertion shift rule of each selected generator.
    ``options.optimize_rounds_n`` caps the BFGS iterations. The caller's
    ``evaluate`` charges every new parameter point to the global allowance and
    raises when it is exhausted. Analytic gradients bound neither the number
    of BFGS iterations nor the number of objective points.
    """
    initial = np.asarray(starting, dtype=float)

    def objective(values):
        return evaluate(tuple(map(float, values)), role="objective")

    def jacobian(values):
        """Return the measured gradient ``dE/dtheta_j`` for every selected angle.

        Each component uses the rule of its generator
        (``_generator_shift_rule``). The generalized rule gives
        ``sum_mu w_mu (E(theta + x_mu e_j) - E(theta - x_mu e_j))``, and the
        insertion rule gives ``sum_k c_k (E_k(+) - E_k(-))``, where
        ``E_k(+-)`` is the energy with the Pauli factor
        ``exp(+-i pi/4 P_k)`` inserted after the ``j``-th exponential. Every
        energy comes from ``evaluate``, which reuses an already collected
        point and charges only new ones.
        """
        # ``evaluate`` charges only points that are not yet collected.
        # Reserving the full shift count in advance would reject a derivative
        # whose points were all collected already.
        derivatives = []
        for position, generator in enumerate(selected):
            rule = rules[generator.pool_index]
            terms = []
            if rule["method"] == "generalized_shift":
                for shift, weight in zip(rule["shifts"], rule["weights"], strict=True):
                    energies = []
                    for sign in (1, -1):
                        shifted = np.array(values, copy=True)
                        shifted[position] += sign * shift
                        energies.append(evaluate(tuple(map(float, shifted)), role="derivative"))
                    terms.append(weight * (energies[0] - energies[1]))
            else:
                for term in rule["terms"]:
                    energies = [
                        evaluate(
                            tuple(map(float, values)),
                            insertion=(position, term["term_index"], sign),
                            role="derivative",
                        )
                        for sign in (1, -1)
                    ]
                    terms.append(term["coefficient"] * (energies[0] - energies[1]))
            derivatives.append(fsum(terms))
        return np.asarray(derivatives)

    def energy_and_gradient(values):
        return evaluate(tuple(map(float, values)), role="objective", gradient=True)

    try:
        if execution != "classical":
            objective(initial)
        result = spo.minimize(
            energy_and_gradient if execution == "classical" else objective,
            initial,
            method="BFGS",
            jac=True if execution == "classical" else jacobian,
            callback=lambda _: completed_round(),
            options={"maxiter": options.optimize_rounds_n},
        )
        return str(result.message), bool(result.success)
    except _EvaluationBudgetExhausted:
        return "evaluation_budget_exhausted", False
