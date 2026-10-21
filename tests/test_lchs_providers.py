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
        "epsilon_lchs": epsilon_lchs,
        "epsilon_quad": epsilon_quad,
        "gamma": gamma,
        "h": step,
    }
    assert plan.nodes == expected_nodes
    assert all(plan.nodes[index] == -plan.nodes[-index - 1] for index in range(index_limit))
    assert plan.nodes[index_limit] == 0.0
    assert plan.base_weights == (step,) * len(expected_nodes)
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

    assert plan.nodes == expected
    assert plan.base_weights == (delta_k,) * state_count
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
    for node, weight in zip(plan.nodes, plan.base_weights, strict=True):
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

    if kernel_request.implementation != "cauchy_density":
        np.testing.assert_allclose(
            plan.coefficients,
            expected_coefficients,
            rtol=2.0e-15,
            atol=0.0,
        )
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
        method=LCHS(approximation_tolerance=0.5),
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
    problem = LinearDynamics(A=[[.4, .15], [.05, .25]], initial_state=[1, 0], time=.1)
    leggauss = np.polynomial.legendre.leggauss
    with monkeypatch.context() as guard:
        guard.setattr(np.polynomial.legendre, "leggauss", lambda *a: pytest.fail("unadmitted grid allocation"))
        with pytest.raises(ValueError, match=r"exceeding LCHS\.max_quadrature_work=1\."):
            plan(problem, method=LCHS(max_quadrature_work=1), execution=execution, seed=7)
    assert np.polynomial.legendre.leggauss is leggauss


def test_closed_tail_bound_keeps_small_cutoff_domain_and_positive_underflow():
    from nwqlib.algorithms.lchs.providers import eq7_tail_bound
    beta, k = .9, .2  # The K>=1 premise of ACL arXiv:2312.03916v2, Eq.(62), is intentionally false.
    c = math.cos(beta*math.pi/2)
    cb = 2*math.pi*math.exp(-2**beta)
    expected = 2/(beta*cb)*math.exp(-c*k**beta)/(c*k**beta)
    assert eq7_tail_bound(beta, k) == pytest.approx(expected, rel=2e-15, abs=0)
    assert eq7_tail_bound(beta, 1e6) == np.nextafter(0., 1.)  # Positive bound rounded upward.
