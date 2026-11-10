"""Independent contracts for LCHS kernel and quadrature providers."""

from __future__ import annotations
from nwqlib import LinearDynamics, plan, solve
from nwqlib.algorithms.lchs import LCHS

import math

import numpy as np
import pytest

from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.algorithms.lchs.providers import (
    LOW_SOMMA_PROFILE_ID,
    LOW_SOMMA_UNBUDGETED_STAGES,
    LCHS_KERNEL_IMPLEMENTATIONS,
    LCHSProblemContext,
    _resolve_coefficient_context,
    resolve_provider_config,
)


def _context() -> LCHSProblemContext:
    return LCHSProblemContext(
        final_time=0.7,
        epsilon=0.13,
        l_norm=0.4,
    )


@pytest.mark.parametrize("order", [1, 7, 64])
def test_gauss_rules_integrate_polynomials_without_a_dense_companion(order, monkeypatch):
    from nwqlib.algorithms.lchs.providers import composite_gauss_grid
    from nwqlib.algorithms.lchs.inhomogeneous_theory import duhamel_quadrature

    def dense_companion(*args, **kwargs):
        raise AssertionError("quadrature constructed a dense companion matrix")

    monkeypatch.setattr(np.polynomial.legendre, "legcompanion", dense_companion)
    nodes, weights = composite_gauss_grid(interval_count_each_side=2, h1=.75, node_count=order)
    times, time_weights = duhamel_quadrature(.7, order)
    assert np.all(np.diff(nodes) > 0) and np.all(weights > 0)
    assert np.all(np.diff(times) > 0) and np.all(time_weights > 0)
    # Gauss order n integrates degree <= 2n-1 exactly in real arithmetic.
    # Normalize the coordinate so powers stay within [-1, 1]. The tolerance
    # covers node/weight formation and the finite sums at these small orders.
    for degree in sorted(set(range(min(12, 2 * order))) | {2 * order - 2, 2 * order - 1}):
        expected = 0. if degree % 2 else 3. / (degree + 1)
        assert abs(np.dot(weights, (nodes / 1.5)**degree) - expected) <= 1e-12
        assert abs(np.dot(time_weights, (times / .7)**degree) - .7 / (degree + 1)) <= 1e-12


def _signed_request(*, num_qubits: int = 3, lsb_position: int = -1) -> ProviderConfig:
    return ProviderConfig(
        implementation="signed_binary_uniform",
        parameters={"num_qubits": num_qubits, "lsb_position": lsb_position},
    )


def _independent_eq7(beta: float, k_value: float) -> complex:
    normalization = 2.0 * math.pi * math.exp(-(2.0**beta))
    kernel = 1.0 / (normalization * np.exp((1.0 + 1.0j * k_value) ** beta))
    return kernel / (1.0 - 1.0j * k_value)


def _independent_cauchy(k_value: float) -> complex:
    return complex(1.0 / (math.pi * (1.0 + k_value**2)))


def _independent_low_somma_coefficient(
    k_value: float,
    *,
    gamma: float,
    c_value: float,
) -> complex:
    return complex(
        np.exp(c_value - 1.0j * k_value * c_value)
        * math.exp(-(k_value**2 + 1.0) / (4.0 * gamma**2))
        / (math.pi * (1.0 + k_value**2))
    )


def test_provider_parameters_keep_their_mathematical_domains():
    request = ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": 1.0})
    with pytest.raises(ValueError, match="must lie in"):
        resolve_provider_config(request, LCHS_KERNEL_IMPLEMENTATIONS, slot="test_provider")


@pytest.mark.parametrize("beta", (0.3, 0.8, 0.95))
@pytest.mark.parametrize("k_value", (-2.0, -0.25, 0.0, 0.75, 3.0))
def test_near_optimal_eq7_provider_matches_independent_closed_form(
    beta: float,
    k_value: float,
) -> None:
    provider = LCHS_KERNEL_IMPLEMENTATIONS["near_optimal_eq7"]
    actual = provider.coefficient(
        k_value,
        {"kernel_parameters": {"beta": beta}, "derived_parameters": {}},
    )
    assert actual == pytest.approx(
        _independent_eq7(beta, k_value),
        rel=2.0e-15,
        abs=0.0,
    )


@pytest.mark.parametrize(
    ("epsilon", "c_value", "final_time", "l_norm"),
    (
        (0.8, 1.0, 0.1, 0.4),
        (0.5, 0.5, 0.7, 0.4),
        (0.2, 2.0, 0.3, 1.2),
    ),
)
def test_low_somma_paper_profile_grid_matches_independent_closed_form(
    epsilon: float,
    c_value: float,
    final_time: float,
    l_norm: float,
) -> None:
    context = LCHSProblemContext(
        final_time=final_time,
        epsilon=epsilon,
        l_norm=l_norm,
    )
    plan = _resolve_coefficient_context(
        lchs_kernel=ProviderConfig(
            implementation="low_somma_f2",
            parameters={"c": c_value},
        ),
        k_quadrature=ProviderConfig(implementation="symmetric_uniform_trapezoid"),
        problem_context=context,
    )
    epsilon_lchs = epsilon / 3.0
    epsilon_quad = epsilon / 3.0
    gamma = (1.0 / c_value) * math.sqrt(
        c_value + math.log((1.0 + 1.0 / (2.0 * math.pi)) / epsilon_lchs)
    )
    radius = 2.0 * c_value * gamma**2
    h_max = math.pi / (
        final_time * l_norm / 2.0
        + math.log(64.0 * math.exp(3.0 * c_value / 2.0) / (15.0 * epsilon_quad))
    )
    ratio = radius / h_max
    # These calibration points stay at least 0.16 from an integer ceiling boundary.
    assert abs(ratio - round(ratio)) > 0.16
    index_limit = math.ceil(ratio)
    step = radius / index_limit
    expected_nodes = tuple(float(index * step) for index in range(-index_limit, index_limit + 1))
    expected_coefficients = tuple(
        step
        * _independent_low_somma_coefficient(
            node,
            gamma=gamma,
            c_value=c_value,
        )
        for node in expected_nodes
    )
    padded_count = 1 << (len(expected_nodes) - 1).bit_length()
    record = plan.record()

    derived = dict(plan.derived_parameters)
    # The provider evaluates the same logarithm as 3c/2+log(64/(15*epsilon_quad)),
    # so h_max may differ by a few ulps; 1e-15 is about 4.5 ulps.
    assert derived.pop("h_max") == pytest.approx(h_max, rel=1.0e-15, abs=0.0)
    assert derived == {
        "J": index_limit,
        "R": radius,
        "allowance_kernel_approximation": epsilon_lchs,
        "allowance_k_quadrature": epsilon_quad,
        "epsilon_lchs": epsilon_lchs,
        "epsilon_quad": epsilon_quad,
        "gamma": gamma,
        "h": step,
    }
    np.testing.assert_array_equal(plan.nodes, expected_nodes)
    assert all(plan.nodes[index] == -plan.nodes[-index - 1] for index in range(index_limit))
    assert plan.nodes[index_limit] == 0.0
    np.testing.assert_array_equal(plan.base_weights, (step,) * len(expected_nodes))
    np.testing.assert_allclose(
        plan.coefficients,
        expected_coefficients,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert step <= h_max
    assert len(plan.nodes) == 2 * index_limit + 1
    assert plan.quadrature.padding_count == padded_count - len(expected_nodes)
    assert plan.quadrature.address_structure == {
        "affine": False,
        "kind": "symmetric_uniform_trapezoid_with_identity_padding",
        "padding_count": padded_count - len(expected_nodes),
        "physical_node_count": len(expected_nodes),
    }
    assert record["lchs_kernel_fixed_parameters"] == {"j": 2, "y": 1}
    assert record["kernel_parameter_selection"] == {
        "c": "user",
        "profile": LOW_SOMMA_PROFILE_ID,
    }
    assert record["approximate_lchs_error_bound"] == epsilon_lchs
    assert record["quadrature_error_bound"] == epsilon_quad
    assert record["solution_error_certificate_status"] == "incomplete"
    assert record["unbudgeted_error_stages"] == list(LOW_SOMMA_UNBUDGETED_STAGES)


def test_default_provider_pair_matches_pre_provider_construction_at_roundoff() -> None:
    """Build the Gauss panels and kernel coefficients independently to check the selected finite
    quadrature recipe.
    """
    beta = 0.8
    multiplier = 1.25
    context = _context()
    plan = _resolve_coefficient_context(
        lchs_kernel=ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": beta}),
        k_quadrature=ProviderConfig(
            implementation="composite_gauss",
            parameters={"truncation_multiplier": multiplier},
        ),
        problem_context=context,
    )

    from scipy.special import lambertw

    normalization = 2 * math.pi * math.exp(-(2**beta))
    tail_budget = context.epsilon / 2
    x = float(lambertw(2 / (beta * normalization * tail_budget)).real)
    theoretical_range = (x / math.cos(beta * math.pi / 2)) ** (1 / beta)
    range_k = multiplier * theoretical_range
    panel_count_each_side = plan.quadrature.interval_count_each_side
    node_count = plan.quadrature.node_count
    h1 = range_k / panel_count_each_side
    base_nodes, base_weights = np.polynomial.legendre.leggauss(node_count)
    expected_nodes = []
    for shifted_index in range(2 * panel_count_each_side):
        panel_index = -panel_count_each_side + shifted_index
        lower_end = panel_index * h1
        expected_nodes.extend(0.5 * h1 * base_nodes + lower_end + 0.5 * h1)
    expected_weights = np.tile(
        0.5 * h1 * base_weights,
        2 * panel_count_each_side,
    )
    expected_coefficients = []
    expected_l1 = 0.0
    for node, weight in zip(expected_nodes, expected_weights, strict=True):
        coefficient = weight * _independent_eq7(beta, float(node))
        expected_l1 += float(abs(coefficient))
        expected_coefficients.append(complex(coefficient))

    # Lambert W and the logarithmic production inverse differ by rounding.
    assert plan.quadrature.range_k == pytest.approx(range_k, rel=2e-14, abs=0)
    assert plan.quadrature.interval_count_each_side == panel_count_each_side
    assert plan.quadrature.node_count == node_count
    np.testing.assert_allclose(plan.nodes, expected_nodes, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(plan.base_weights, expected_weights, rtol=2e-14, atol=0)
    np.testing.assert_allclose(
        plan.coefficients,
        np.asarray(expected_coefficients),
        rtol=1.0e-13,
        atol=0.0,
    )
    assert plan.coefficient_l1_norm == pytest.approx(
        expected_l1,
        rel=1.0e-13,
        abs=0.0,
    )


@pytest.mark.parametrize(
    ("num_qubits", "lsb_position"),
    ((1, -2), (2, 0), (3, -1), (4, 2)),
)
def test_signed_binary_nodes_and_affine_certificate_are_exact(
    num_qubits: int,
    lsb_position: int,
) -> None:
    """Powers-of-two weights give exact signed-binary nodes, including the negative sign-bit
    contribution.
    """
    plan = _resolve_coefficient_context(
        lchs_kernel=ProviderConfig(implementation="cauchy_density"),
        k_quadrature=_signed_request(
            num_qubits=num_qubits,
            lsb_position=lsb_position,
        ),
        problem_context=_context(),
    )
    state_count = 2**num_qubits
    transition = 2 ** (num_qubits - 1)
    delta_k = float(2.0**lsb_position)
    expected = tuple(
        float((state if state < transition else state - state_count) * delta_k)
        for state in range(state_count)
    )

    np.testing.assert_array_equal(plan.nodes, expected)
    np.testing.assert_array_equal(plan.base_weights, (delta_k,) * state_count)
    assert plan.quadrature.node_order == "computational_basis"
    assert plan.quadrature.physical_node_count == state_count
    assert plan.quadrature.padding_count == 0
    certificate = plan.quadrature.address_structure
    for state, node in enumerate(plan.nodes):
        reconstructed = sum(
            term["coefficient"]
            for term in certificate["unsigned_terms"]
            if (state >> term["qubit"]) & 1
        )
        if (state >> certificate["sign_bit"]) & 1:
            reconstructed += certificate["sign_coefficient"]
        assert reconstructed == node


@pytest.mark.parametrize(
    ("kernel_request", "quadrature_request", "certificate_status"),
    (
        (
            ProviderConfig(implementation="near_optimal_eq7"),
            ProviderConfig(implementation="composite_gauss"),
            "certified",
        ),
        (
            ProviderConfig(implementation="near_optimal_eq7"),
            _signed_request(),
            "uncertified",
        ),
        (
            ProviderConfig(implementation="cauchy_density"),
            _signed_request(),
            "uncertified",
        ),
        (
            ProviderConfig(implementation="low_somma_f2"),
            ProviderConfig(implementation="symmetric_uniform_trapezoid"),
            "incomplete",
        ),
    ),
)
def test_every_allowed_pair_has_independent_coefficients_and_status(
    kernel_request: ProviderConfig,
    quadrature_request: ProviderConfig,
    certificate_status: str,
) -> None:
    """Independent kernel formulas check coefficient values while provider-pair status remains
    scoped to its evidence.
    """
    plan = _resolve_coefficient_context(
        lchs_kernel=kernel_request,
        k_quadrature=quadrature_request,
        problem_context=_context(),
    )
    beta = float(plan.resolved_lchs_kernel.parameters.get("beta", 0.0))
    gamma = float(plan.derived_parameters.get("gamma", 0.0))
    c_value = float(plan.resolved_lchs_kernel.parameters.get("c", 0.0))
    expected_coefficients = []
    for node, weight in zip(plan.nodes.tolist(), plan.base_weights.tolist(), strict=True):
        if kernel_request.implementation == "near_optimal_eq7":
            kernel_value = _independent_eq7(beta, node)
        elif kernel_request.implementation == "cauchy_density":
            kernel_value = _independent_cauchy(node)
        else:
            kernel_value = _independent_low_somma_coefficient(
                node,
                gamma=gamma,
                c_value=c_value,
            )
        expected_coefficients.append(weight * kernel_value)

    if kernel_request.implementation == "near_optimal_eq7":
        # The oracle forms w = (1 + i k)**beta with Python's complex power and the plan with NumPy's. Each
        # rounds w by a few units in the last place, and exp(-w) has relative condition |w|, so the two
        # agree within 8u(1 + |w|) at each node.
        w = np.abs((1.0 + 1.0j*plan.nodes)**beta)
        relative = np.abs(plan.coefficients - expected_coefficients)/np.abs(expected_coefficients)
        assert np.all(relative <= 8*2.0**-53*(1.0 + w))
    if kernel_request.implementation == "low_somma_f2":
        np.testing.assert_allclose(
            plan.coefficients,
            expected_coefficients,
            rtol=2.0e-15,
            atol=0.0,
        )
    if kernel_request.implementation != "cauchy_density":
        assert plan.coefficient_l1_norm == pytest.approx(
            sum(float(abs(value)) for value in expected_coefficients),
            rel=2.0e-15,
            abs=0.0,
        )
    else:
        assert np.array_equal(plan.coefficients, expected_coefficients)
        assert plan.coefficient_l1_norm == sum(float(abs(value)) for value in expected_coefficients)
    assert plan.quadrature.physical_node_count == len(plan.nodes)
    expected_padding = (
        (1 << (len(plan.nodes) - 1).bit_length()) - len(plan.nodes)
        if kernel_request.implementation == "low_somma_f2"
        else 0
    )
    assert plan.quadrature.padding_count == expected_padding
    assert plan.compatibility.certificate_status == certificate_status
    assert plan.record()["solution_error_certificate_status"] == certificate_status


def test_cauchy_composite_gauss_cannot_inherit_eq7_bounds() -> None:
    with pytest.raises(ValueError, match="range and node-count laws are"):
        _resolve_coefficient_context(
            lchs_kernel=ProviderConfig(implementation="cauchy_density"),
            k_quadrature=ProviderConfig(implementation="composite_gauss"),
            problem_context=_context(),
        )


def test_uncertified_pair_suppresses_solution_error_certificate():
    choice = plan(
        LinearDynamics(A=np.diag([0.2, 0.3]), initial_state=[1.0, 0.0], time=0.2),
        method=LCHS(k_quadrature=_signed_request(num_qubits=2, lsb_position=-1)),
        execution="classical",
        seed=7,
    )
    rec = choice.reconstruction
    assert (
        choice._native["native_data"].coefficient_plan.compatibility.certificate_status
        == "uncertified"
    )
    assert rec.kernel_approximation_bound is None and rec.quadrature_bound is None
    result = solve(choice)
    assessment = result.assess(absolute_tolerance=0.1)
    assert assessment.status == "INCONCLUSIVE"
    assert "algorithmic_approximation" in assessment.remaining
    assert all(
        f.fact.value is None
        for app in result.applications
        for f in app.facts
        if f.fact.quantity in {"kernel_approximation", "k_quadrature"}
    )


def test_default_pair_primary_record_preserves_provider_values():
    choice = plan(
        LinearDynamics(A=np.diag([0.2, 0.3]), initial_state=[1.0, 0.0], time=0.1),
        method=LCHS(approx_tol=0.5),
        execution="classical",
        seed=7,
    )
    rec = choice.reconstruction
    quadrature = choice._native["native_data"].quadrature
    beta, eps, h = 0.75, 0.5, quadrature.h1
    # The independent Lambert-W inverse of the ACL tail bound
    # (arXiv:2312.03916v2) fixes the
    # cutoff. Each of the two construction errors receives half the budget.
    from scipy.special import lambertw
    cbeta = 2 * math.pi * math.exp(-(2**beta))
    root = float(lambertw(2 / (beta * cbeta * (eps / 2))).real)
    k = (root / math.cos(beta * math.pi / 2)) ** (1 / beta)
    q = quadrature.node_count
    x = math.cos(beta*math.pi/2)*k**beta
    truncation = 2/(beta*cbeta) * math.exp(-x)/x
    # ATAP ISBN 978-1-61197-239-9 uses n+1 nodes: the actual Q-point rule has
    # exponent -2*(Q-1).
    rho = 1.8/h + math.sqrt(1 + (1.8/h)**2)
    bound = k*64/(15*cbeta*.1)*math.exp(.9*.1*.3)*rho**(-2*(q-1))/(rho*rho-1)
    # The positive moderate expression and production log form differ only by
    # binary64 evaluation; 1e-12 allows their elementary-function roundoff.
    assert rec.kernel_approximation_bound == pytest.approx(truncation, rel=1e-12, abs=0)
    assert rec.quadrature_bound == pytest.approx(bound, rel=1e-12, abs=0)
    assert rec.kernel_approximation_bound <= eps / 2
    assert rec.quadrature_bound <= eps / 2
    assert rec.kernel_approximation_bound + rec.quadrature_bound <= eps
    assert rec.coefficient_l1_norm == pytest.approx(sum(abs(c) for c in quadrature.coefficients), rel=1e-12, abs=0)
    assert quadrature.total_node_count == len(quadrature.coefficients) == 2*q*quadrature.interval_count_each_side
    assert quadrature.range_k == pytest.approx(k, rel=1e-14, abs=0)
    assert quadrature.h1 == pytest.approx(h, rel=1e-14, abs=0)
    assert q >= 2


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_quadrature_work_limit_refuses_before_grid_allocation(execution, monkeypatch):
    """The composite-Gauss grid is built only after its complete work charge fits."""
    from nwqlib.algorithms.lchs import providers

    problem = LinearDynamics(A=[[.4, .15], [.05, .25]], initial_state=[1, 0], time=.1)
    with monkeypatch.context() as guard:
        guard.setattr(providers, "roots_legendre", lambda *a: pytest.fail("unadmitted grid allocation"))
        with pytest.raises(ValueError, match=r"exceeding LCHS\.max_quadrature_work=1\."):
            plan(problem, method=LCHS(max_quadrature_work=1), execution=execution, seed=7)


def test_closed_tail_bound_keeps_small_cutoff_domain_and_positive_underflow():
    from nwqlib.algorithms.lchs.providers import eq7_tail_bound
    beta, k = .9, .2  # The K>=1 premise of ACL arXiv:2312.03916v2, Eq.(62), is intentionally false.
    c = math.cos(beta*math.pi/2)
    cb = 2*math.pi*math.exp(-2**beta)
    expected = 2/(beta*cb)*math.exp(-c*k**beta)/(c*k**beta)
    assert eq7_tail_bound(beta, k) == pytest.approx(expected, rel=2e-15, abs=0)
    assert eq7_tail_bound(beta, 1e6) == np.nextafter(0., 1.)  # Positive bound rounded upward.


def _noncommuting_parts():
    """A PSD L and Hermitian H on one qubit with ||[L, H]|| = 0.127, and ||L||."""
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]], complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    return 0.6 * np.eye(2) + 0.2 * z + 0.1 * x, 0.15 * x + 0.25 * y, 0.6 + math.hypot(0.2, 0.1)


def _finite_sum_error(plan, l_part, h_part, time):
    """Operator-norm error of sum_j c_j exp(-i t (k_j L + H)) against exp(-t (L + iH)), both by expm."""
    from scipy.linalg import expm
    total = sum(c * expm(-1j * time * (k * l_part + h_part)) for k, c in zip(plan.nodes, plan.coefficients))
    return float(np.linalg.norm(total - expm(-time * (l_part + 1j * h_part)), 2))


def test_apel_berntson_provider_bounds_noncommuting_exponential() -> None:
    """The Apel-Berntson sum (arXiv:2610.03669v1) meets its component bounds for noncommuting L and H.

    Unequal explicit allowances must each bound their own component, and the
    finite coefficient mass must stay within its Corollary 9 bound of beta_R. The
    reference exponentials are independent of the provider.
    """
    l_part, h_part, l_norm = _noncommuting_parts()
    allowances = {"kernel_approximation": 5e-7, "k_discretization": 2e-7, "k_truncation": 1e-7}
    plan = _resolve_coefficient_context(
        lchs_kernel=ProviderConfig(implementation="apel_berntson"),
        k_quadrature=ProviderConfig(implementation="tanh_sinh"),
        problem_context=LCHSProblemContext(final_time=0.7, epsilon=1e-6, l_norm=l_norm,
                                           component_allowances=allowances),
    )
    derived = plan.derived_parameters
    assert plan.quadrature.approximate_lchs_error_bound == allowances["kernel_approximation"]
    assert derived["discretization_bound"] <= allowances["k_discretization"]
    assert derived["truncation_bound"] <= allowances["k_truncation"]
    bound = plan.quadrature.approximate_lchs_error_bound + plan.quadrature.quadrature_error_bound
    for time in (0.35, 0.7):
        assert _finite_sum_error(plan, l_part, h_part, time) <= bound
    assert abs(plan.coefficient_l1_norm - derived["beta_R"]) <= derived["beta_discretization_bound"]
    assert plan.compatibility.certificate_status == "certified"


@pytest.fixture(scope="module")
def discrete_weight_artifact():
    """One bounded fixed-node preparation: 16 nodes, b = 0.6, M = 64, 16 directions."""
    from nwqlib.algorithms.lchs.coefficient_artifact import prepare_lchs_discrete_weights
    return prepare_lchs_discrete_weights(
        nodes=0.5 * (np.arange(16) - 7.5), support_endpoint=0.6, residual_allowance=2e-2,
        residual_modes=64, polygon_directions=16, solver_time_limit=30.0)


def _artifact_plan(artifact, final_time):
    _, _, l_norm = _noncommuting_parts()
    return _resolve_coefficient_context(
        lchs_kernel=ProviderConfig(implementation="discrete_weights",
                                   parameters={"artifact_id": artifact.content_id}),
        k_quadrature=ProviderConfig(implementation="artifact_nodes"),
        problem_context=LCHSProblemContext(final_time=final_time, epsilon=2e-2, l_norm=l_norm),
        coefficient_artifact=artifact,
    )


def _independent_residual_bound(nodes, weights, b, modes):
    """Local Wiener-norm bound of the weights on [0, b], with Gauss-Legendre cosine coefficients."""
    nodes, weights = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=complex)
    points, quadrature_weights = np.polynomial.legendre.leggauss(400)
    points, quadrature_weights = 0.5 * b * (points + 1), 0.5 * b * quadrature_weights
    eta = np.exp(-1j * np.outer(points, nodes)) @ weights - np.exp(-points)
    cosine = np.array([(2 / b) * np.sum(quadrature_weights * eta * np.cos(m * np.pi * points / b))
                       for m in range(modes + 1)])
    slope_zero = abs(1 - 1j * np.sum(nodes * weights))
    slope_end = abs(np.exp(-b) - 1j * np.sum(nodes * weights * np.exp(-1j * nodes * b)))
    curvature = b * np.sum(nodes**2 * np.abs(weights)) + 1 - np.exp(-b)
    return (abs(cosine[0]) / 2 + np.sum(np.abs(cosine[1:]))
            + 2 * b / (np.pi**2 * modes) * (slope_zero + slope_end + curvature))


def test_discrete_weight_artifact_bounds_noncommuting_exponential(discrete_weight_artifact) -> None:
    """Prepared weights meet the residual bound recomputed from them, and only within their support.

    The local Wiener-norm bound on [0, b] holds for every t with
    t*||L|| <= b, also when L and H do not commute. T_e*||L|| = 0.577 lies
    inside b = 0.6 and 0.75*||L|| = 0.618 does not. At t = 0 the error is
    |sum_j w_j - 1|, so the bound covers it. The recorded bound is compared
    with one whose cosine coefficients come from 400-point Gauss-Legendre
    quadrature instead of closed-form moments, for the prepared weights and
    on the short support b = 1e-18. There zero weights have error one at
    t = 0, and the unit weight at kappa = 0 is exact up to O(b). The
    tolerance 1e-11 is far above the binary64 error of either sum, about
    1e-13, and far below the tail floor 2b(1-exp(-b))/(pi**2 M) = 8.6e-4 at
    b = 0.6.
    """
    from nwqlib.algorithms.lchs.coefficient_artifact import _residual_record
    l_part, h_part, _ = _noncommuting_parts()
    plan = _artifact_plan(discrete_weight_artifact, 0.7)
    residual = discrete_weight_artifact.residual
    bound = plan.quadrature.approximate_lchs_error_bound
    assert bound == residual.residual_bound + residual.rounding_estimate <= 2e-2
    assert residual.residual_bound == pytest.approx(
        _independent_residual_bound(discrete_weight_artifact.nodes, discrete_weight_artifact.weights, 0.6, 64),
        rel=0, abs=1e-11)
    assert abs(sum(discrete_weight_artifact.weights) - 1) <= bound
    for weights in (np.zeros(1), np.ones(1)):
        short = _residual_record(np.zeros(1), weights, 1e-18, 1, 2e-2)
        assert short.residual_bound == pytest.approx(_independent_residual_bound(np.zeros(1), weights, 1e-18, 1),
                                                     rel=0, abs=1e-11)
        assert abs(weights.sum() - 1) <= short.residual_bound + short.rounding_estimate
    for time in (0.175, 0.35, 0.7):
        assert _finite_sum_error(plan, l_part, h_part, time) <= bound
    # Recursive and exact (fsum) summation of 16 moduli differ by at most 15 unit roundoffs.
    assert plan.coefficient_l1_norm == pytest.approx(discrete_weight_artifact.coefficient_mass,
                                                     rel=16 * 2.0**-53, abs=0)
    with pytest.raises(ValueError, match=r"covers T_e\*B_Ls <= 0.6"):
        _artifact_plan(discrete_weight_artifact, 0.75)


def test_discrete_weight_artifact_reloads_and_is_used_without_a_solver(discrete_weight_artifact,
                                                                       monkeypatch) -> None:
    """A JSON save and reload keeps every weight bit and the bound, and using the weights runs no solver."""
    import scipy.optimize
    from nwqlib.algorithms.lchs.coefficient_artifact import LCHSDiscreteWeightArtifact
    reloaded = LCHSDiscreteWeightArtifact.model_validate_json(discrete_weight_artifact.model_dump_json())
    assert reloaded.content_id == discrete_weight_artifact.content_id
    assert [(w.real.hex(), w.imag.hex()) for w in reloaded.weights] == [
        (w.real.hex(), w.imag.hex()) for w in discrete_weight_artifact.weights]
    assert reloaded.residual == discrete_weight_artifact.residual
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *args, **kwargs: pytest.fail("solver called"))
    plan = _artifact_plan(reloaded, 0.7)
    np.testing.assert_array_equal(plan.nodes, reloaded.nodes)
    np.testing.assert_array_equal(plan.coefficients, reloaded.weights)


@pytest.mark.parametrize("case, count", (
    ("default_pair", 0), ("apel_berntson", None), ("discrete_weights", 16), ("empty", 0), ("one_zero_value", 1),
))
def test_sampled_held_bytes_charge_16_bytes_per_kernel_value(case, count, discrete_weight_artifact) -> None:
    """`sampled_plan._held_bytes` charges each kernel value of a rule as one complex128 of 16 bytes.

    Every other array here is empty and there are no inputs, so H is the
    kernel-value charge alone. The default composite-Gauss rule has no kernel
    values and charges nothing. The tanh-sinh and artifact_nodes rules supply
    an array and charge their node count (None here means the rule's own
    node_count). A single zero value still counts as one value, because the
    charge depends on the array's length and not on its entries.
    """
    from types import SimpleNamespace as Fields
    from nwqlib.algorithms.lchs.sampled_plan import _held_bytes

    def grid_rule(kernel, quadrature):
        context = LCHSProblemContext(final_time=0.7, epsilon=2e-2, l_norm=_noncommuting_parts()[2])
        return _resolve_coefficient_context(lchs_kernel=ProviderConfig(implementation=kernel),
                                            k_quadrature=ProviderConfig(implementation=quadrature),
                                            problem_context=context).quadrature

    rule = {
        "default_pair": lambda: grid_rule("near_optimal_eq7", "composite_gauss"),
        "apel_berntson": lambda: grid_rule("apel_berntson", "tanh_sinh"),
        "discrete_weights": lambda: _artifact_plan(discrete_weight_artifact, 0.7).quadrature,
        "empty": lambda: Fields(kernel_values=np.zeros(0, complex)),
        "one_zero_value": lambda: Fields(kernel_values=np.zeros(1, complex)),
    }[case]()
    expected = 16 * (rule.node_count if count is None else count)
    assert rule.kernel_values is None or rule.kernel_values.nbytes == expected
    nothing = np.zeros(0)
    raw = Fields(nodes=nothing, base_weights=nothing, coefficients=nothing, quadrature=rule)
    data = Fields(quadrature=Fields(k_nodes=nothing, coefficients=nothing, coefficient_plan=raw),
                  coefficient_plan=Fields(coefficients=nothing))
    problem = Fields(A=None, initial_state=None, source=None)
    common = Fields(generator=Fields(dissipative=None, hamiltonian=None))
    assert _held_bytes(problem, None, common, data) == expected


def test_discrete_weight_preparation_charges_before_building_its_program(monkeypatch) -> None:
    """The byte charge and the tail floor refuse before any constraint matrix or solve."""
    import scipy.optimize
    import scipy.sparse
    from nwqlib.algorithms.lchs.coefficient_artifact import prepare_lchs_discrete_weights
    monkeypatch.setattr(scipy.sparse, "csr_array", lambda *args, **kwargs: pytest.fail("matrix built"))
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *args, **kwargs: pytest.fail("solver called"))
    request = dict(nodes=np.linspace(-3, 3, 16), support_endpoint=0.6, residual_modes=64,
                   polygon_directions=16, solver_time_limit=1.0)
    with pytest.raises(ValueError, match="discrete-weight linear program"):
        prepare_lchs_discrete_weights(residual_allowance=2e-2, max_bytes=1_000_000, **request)
    # The tail term alone is 2b(1-exp(-b))/(pi**2 M) = 8.6e-4 at M = 64.
    with pytest.raises(ValueError, match="tail bound alone"):
        prepare_lchs_discrete_weights(residual_allowance=8e-4, **request)


def test_discrete_weight_preparation_limits_bytes_and_solver_time_only(monkeypatch) -> None:
    """J = 2048 nodes, M = 256 modes and D = 32 directions pass every check whose bytes fit, up to the program.

    The program has about 3.4e7 matrix entries and about 2.2 GB of known arrays, below the 10 GB default.
    The explicit solver time limit, not a work count, bounds the solve.
    """
    from nwqlib.algorithms.lchs import coefficient_artifact

    class Reached(Exception):
        pass

    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(coefficient_artifact, "_residual_affine_maps", stop)
    with pytest.raises(Reached):
        coefficient_artifact.prepare_lchs_discrete_weights(
            nodes=np.linspace(-3, 3, 2048), support_endpoint=0.6, residual_allowance=2e-2, residual_modes=256,
            polygon_directions=32, solver_time_limit=1.0)


@pytest.mark.parametrize("request_", (ProviderConfig(implementation="apel_berntson"),
                                      ProviderConfig(implementation="low_somma_f2", parameters={"c": 1.0})))
def test_continuous_kernel_proposal_mean_matches_scalar_exponential(request_) -> None:
    """The mean of the returned weight over the guarded proposal is the guarded kernel integral.

    For U uniform on [0, 1], node(U) has density q_R/(1-2*guard) on
    [-R_g, R_g] with R_g = tan((1-2*guard)*atan(R)), so the mean of
    weight(k)*exp(-i x k) equals int_{-R_g}^{R_g} g(k) exp(-i x k) dk with no
    further factor. The right side integrates a closed form of g directly in
    k. scipy.integrate.quad reports absolute errors below 1e-9 for both
    sides here. The tolerance 1e-8 lies far below the change
    2*guard*|integral| >= 5e-6 that a weight for the unguarded proposal would
    add. For a scalar L = lam >= 0 and x = t*lam, the mean is within kernel
    plus guard error of exp(-x). The weight cap is attained at k = 0.
    """
    from scipy.integrate import quad
    from nwqlib.algorithms.lchs.providers import resolve_lchs_continuous_kernel
    plan = resolve_lchs_continuous_kernel(lchs_kernel=request_, kernel_allowance=1e-3, guard_allowance=1e-4)
    derived = plan.derived_parameters
    if request_.implementation == "apel_berntson":
        def kernel(k):
            ratio = math.log((plan.radius + k) / (plan.radius - k))
            return (math.exp((1 - derived["mu"]) * derived["Delta"]) / (math.pi * (1 + k * k))
                    * np.exp(-1j * derived["Delta"] / math.pi * ratio))
    else:
        def kernel(k):
            return _independent_low_somma_coefficient(k, gamma=derived["gamma"], c_value=1.0)
    guarded_radius = math.tan((1 - 2 * plan.guard) * math.atan(plan.radius))

    def integral(function, lower, upper):
        return complex(*(quad(lambda v, imaginary: (function(v).imag if imaginary else function(v).real),
                              lower, upper, args=(flag,), limit=4000, epsabs=1e-12)[0] for flag in (False, True)))

    assert plan.guard_omission_bound <= 1e-4
    for x in (0.5, 2.0):
        mean = integral(lambda u: plan.weight(plan.node(u)) * np.exp(-1j * x * plan.node(u)), 0, 1)
        guarded = integral(lambda k: kernel(k) * np.exp(-1j * x * k), -guarded_radius, guarded_radius)
        assert abs(mean - guarded) <= 1e-8
        assert abs(mean - math.exp(-x)) <= plan.kernel_error_bound + plan.guard_omission_bound
    # Binary64 rounding of the exponentials is about 1e-14 here, far below 2*guard.
    assert abs(plan.weight(0.0)) == pytest.approx(plan.importance_weight_bound, rel=1e-12)
    if request_.implementation == "apel_berntson":
        # The guard 2.4e-19 rounds node(0) and node(1) to -R and R, where the kernel is undefined.
        with pytest.raises(ValueError, match="numerical range"):
            resolve_lchs_continuous_kernel(lchs_kernel=request_, kernel_allowance=1e-3, guard_allowance=1e-18)


def test_public_lchs_helpers_apply_a_work_limit_only_when_given():
    """The Cartesian check and the coefficient plan of a 512-dimensional A need no work limit.

    The eigenvalue check of L costs d**3 = 134,217,728 work units at
    d = 512. cartesian_decomposition runs it without a limit and refuses one
    unit below it when that limit is given. resolve_lchs_coefficient_plan
    runs the same check with the default LCHS, which sets no work limit.
    """
    from nwqlib import LinearDynamics
    from nwqlib.algorithms.lchs import cartesian_decomposition, resolve_lchs_coefficient_plan

    d = 512
    shift = np.diag(np.ones(d - 1), 1)
    a = 2 * np.eye(d) - shift - shift.T + 0.3j * (shift + shift.T)
    l_part, _ = cartesian_decomposition(a)
    np.testing.assert_array_equal(l_part, 2 * np.eye(d) - shift - shift.T)
    with pytest.raises(ValueError, match=f"max_spectral_work={d**3 - 1}"):
        cartesian_decomposition(a, max_spectral_work=d**3 - 1)
    state = np.zeros(d)
    state[0] = 1
    coefficients = resolve_lchs_coefficient_plan(LinearDynamics(A=a, initial_state=state, time=0.5))
    assert len(coefficients.nodes) == len(coefficients.coefficients) > 0


@pytest.mark.parametrize("matrix, providers", (
    (np.diag([.25, .5]), {"lchs_kernel": ProviderConfig(implementation="cauchy_density"),
                          "k_quadrature": ProviderConfig(implementation="signed_binary_uniform",
                                                         parameters={"num_qubits": 2, "lsb_position": -1})}),
    (np.diag([.25, .5]), {"lchs_kernel": ProviderConfig(implementation="apel_berntson"),
                          "k_quadrature": ProviderConfig(implementation="tanh_sinh")}),
    (1j * np.diag([.25, .5]), {}),
), ids=("signed_binary", "tanh_sinh_kernel_values", "zero_l_one_node"))
def test_independent_plans_compare_by_exact_field_values(matrix, providers) -> None:
    """Two resolves of one input are equal plans, and one changed array element or field makes them unequal."""
    from dataclasses import replace
    from nwqlib.algorithms.lchs import resolve_lchs_coefficient_plan

    problem = LinearDynamics(A=matrix, initial_state=[1., 0.], time=.25)
    left, right = (resolve_lchs_coefficient_plan(problem, **providers) for _ in range(2))
    assert left is not right and left == right and not left != right
    assert left.quadrature == right.quadrature
    coefficients = left.coefficients.copy()
    coefficients[-1] += 2.0**-40
    assert replace(left, coefficients=coefficients) != left
    weights = left.quadrature.base_weights.copy()
    weights[0] = np.nextafter(weights[0], np.inf)
    assert replace(left.quadrature, base_weights=weights) != left.quadrature
    assert replace(left, quadrature=replace(left.quadrature, base_weights=weights)) != left
    if left.quadrature.kernel_values is not None:
        assert replace(left.quadrature, kernel_values=None) != left.quadrature
    assert replace(left, recovery_exponent=left.recovery_exponent + 1) != left
    assert left != left.quadrature and left != tuple(left.coefficients)


def test_saved_plan_reopens_an_equal_coefficient_plan(tmp_path) -> None:
    from nwqlib import load_result
    from nwqlib.algorithms.lchs import resolve_lchs_coefficient_plan

    chosen = plan(LinearDynamics(A=np.diag([.2, .5]), initial_state=[1., 0.], time=1.), method=LCHS(),
                  execution="quantum")
    reopened = load_result(solve(chosen).save(tmp_path / "saved")).plan
    assert resolve_lchs_coefficient_plan(reopened) == resolve_lchs_coefficient_plan(chosen)


def test_provider_records_copy_and_pickle_as_equal_read_only_records() -> None:
    """An LCHS with a provider parameter and a resolved provider survive pickle and deepcopy unchanged.

    The LCHS Method and the resolved provider record taken from a coefficient plan
    come back equal, with their parameters a read-only view in sorted order. The
    coefficient plan itself is not round-tripped here.
    """
    import copy
    import pickle
    from types import MappingProxyType
    from nwqlib.algorithms.lchs import resolve_lchs_coefficient_plan

    kernel = ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": .7})
    method = LCHS(lchs_kernel=kernel, k_quadrature="composite_gauss")
    resolved = resolve_lchs_coefficient_plan(
        LinearDynamics(A=np.diag([.25, .5]), initial_state=[1., 0.], time=.25),
        lchs_kernel=ProviderConfig(implementation="cauchy_density"),
        k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
                                    parameters={"num_qubits": 2, "lsb_position": -1})).resolved_k_quadrature
    for copied in (pickle.loads(pickle.dumps(method)), copy.deepcopy(method)):
        assert copied == method and copied.content_id == method.content_id
        assert dict(copied.lchs_kernel.parameters) == {"beta": .7}
        assert type(copied.lchs_kernel.parameters) is MappingProxyType
        with pytest.raises(TypeError):
            copied.lchs_kernel.parameters["beta"] = .5
    for copied in (pickle.loads(pickle.dumps(resolved)), copy.deepcopy(resolved)):
        assert copied == resolved and copied.to_dict() == resolved.to_dict()
        assert list(copied.parameters) == list(resolved.parameters) == ["lsb_position", "num_qubits"]
        assert type(copied.parameters) is MappingProxyType
