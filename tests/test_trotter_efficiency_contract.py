"""Explicit fixed-PF certificates at the selected schedule, with actual work caps."""

import pytest

from nwqlib import LinearDynamics, plan, solve
from nwqlib.algorithms.lchs import LCHS, LCHSRefinement, Trotter, TrotterErrorBudgeted
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.subroutines.trotterization import error_budget


def selected(*, execution="classical"):
    return plan(
        LinearDynamics(
            A=[[0.45, 0.08j], [0.03j, 0.7]],
            initial_state=[1.0, 0.2],
            time=0.2,
        ),
        method=LCHS(
            approx_tol=0.7,
            lchs_kernel=ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": 0.8}),
            k_quadrature=ProviderConfig(
                implementation="composite_gauss", parameters={"truncation_multiplier": 1.0}
            ),
            ham_evo=Trotter(steps=2),
        ),
        execution=execution,
        seed=7,
    )


def arguments(receipt):
    return {item.parameter: item.value for item in receipt.applications[0].arguments}


@pytest.fixture(scope="module")
def fixed_result():
    return solve(selected(execution="quantum"))


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_plan_and_refinement_choose_the_census_expression_from_the_same_nodes(monkeypatch, execution):
    """The budgeted plan and the refinement of its saved node table weight the same K nodes.

    The union labels X, Y and Z anticommute pairwise, so N(A) > 0, and every k-node weights them
    differently. The threshold is moved to the plan's own selection count N(A)*max(w, K) and one below
    it, so the plan takes the full expression and then the relaxation, without a large table. Each time
    the refinement makes the same choice from the same count.
    """
    choose = error_budget.census_structure
    calls = []

    def spy(*args, **kwargs):
        census = choose(*args, **kwargs)
        calls.append((kwargs.get("weightings", 1), census.selection_count, census.variant))
        return census

    def budgeted():
        return plan(LinearDynamics(A=[[0.45, 0.08j], [0.03j, 0.7]], initial_state=[1.0, 0.2], time=0.2),
                    method=LCHS(approx_tol=0.7,
                                lchs_kernel=ProviderConfig(implementation="near_optimal_eq7",
                                                           parameters={"beta": 0.8}),
                                k_quadrature=ProviderConfig(implementation="composite_gauss",
                                                            parameters={"truncation_multiplier": 1.0}),
                                ham_evo=TrotterErrorBudgeted(order=2)),
                    execution=execution, seed=7)

    monkeypatch.setattr(error_budget, "census_structure", spy)
    budgeted()
    weightings, count, _ = calls[0]
    assert weightings > 1 and count > 0
    for threshold, variant in ((count, "exact_census"), (count - 1, "relaxed_prefix")):
        monkeypatch.setattr(error_budget, "EXACT_CENSUS_WORD_TESTS", threshold)
        calls.clear()
        solve(budgeted()).verify(checks=LCHSRefinement(components=("fixed_pf",)))
        assert len(calls) >= 2 and set(calls) == {(weightings, count, variant)}


def test_fixed_trotter_does_not_compute_unrequested_commutator_bounds(monkeypatch):
    monkeypatch.setattr(
        error_budget,
        "_bound_coefficient_evaluation",
        lambda *a, **k: pytest.fail("default fixed PF acquired a structural certificate"),
    )
    monkeypatch.setattr(
        error_budget,
        "dense_trotter_bound_coefficient",
        lambda *a, **k: pytest.fail("default fixed PF acquired dense validation"),
    )
    result = solve(selected())
    assert result.applications
    assert all(
        next(fact for fact in app.facts if fact.fact.quantity == "trotter_synthesis").fact.value
        is None
        for app in result.applications
    )
    assessment = result.assess(absolute_tolerance=0.1)
    assert (
        "algorithmic_approximation" in assessment.remaining and assessment.status == "INCONCLUSIVE"
    )
    assert result.report()["result"] == result.model_dump(mode="json")


def test_fixed_structural_certificate_refines_evidence_without_reselecting_or_mutating(
    monkeypatch, fixed_result
):
    """Explicit commutator evidence must use the fixed schedule and reject a contradictory dense
    bound before publication.
    """
    from nwqlib.algorithms.lchs import time_independent_terms as numerical

    result = fixed_result
    choice = result.plan
    before = result.model_dump_json()
    schedule = choice.reconstruction.step_counts
    checks = LCHSRefinement(components=("fixed_pf",), dense_validation=True)
    monkeypatch.setattr(
        numerical,
        "generate_lchs_product_formula_select_plan",
        lambda *a, **k: pytest.fail("refinement reselected a PF schedule"),
    )
    receipt, facts = result.verify(checks=checks)
    assert choice.reconstruction.step_counts == schedule and result.model_dump_json() == before
    bound = next(
        fact.fact.value.value for fact in facts if fact.fact.quantity == "trotter_synthesis"
    )
    assert bound > 0
    assert arguments(receipt)["dense_bound_completed"] == len(
        choice._native["native_data"].quadrature.k_nodes
    )
    assert any(
        fact.fact.value.value > 0
        for fact in receipt.applications[0].facts
        if fact.fact.quantity.endswith("dense_synthesis")
    )
    assert result.assess(absolute_tolerance=0.1, facts=facts).status == "INCONCLUSIVE"
    monkeypatch.setattr(error_budget, "dense_trotter_bound_coefficient", lambda *a, **k: 1e6)
    monkeypatch.setattr(
        "nwqlib.algorithms.lchs.refinement._publish",
        lambda *a, **k: pytest.fail("contradictory dense evidence created a receipt"),
    )
    with pytest.raises(RuntimeError, match="does not upper-bound dense validation"):
        result.verify(checks=checks)
    assert result.model_dump_json() == before and choice.reconstruction.step_counts == schedule


def test_fixed_refinement_checks_its_concrete_limit_before_expensive_work(
    monkeypatch, fixed_result
):
    from nwqlib.algorithms.lchs import time_independent_terms as numerical

    result = fixed_result

    def forbidden(*args, **kwargs):
        pytest.fail("refinement crossed its unadmitted numerical boundary")

    from nwqlib.subroutines.trotterization import error_budget

    # Refinement reads the stored node table and never decomposes.
    # max_dense_work rejects before the requested dense commutators.
    monkeypatch.setattr(numerical, "_trotter_pauli_decomposition", forbidden)
    monkeypatch.setattr(error_budget, "dense_trotter_bound_coefficient", forbidden)
    monkeypatch.setattr("nwqlib.algorithms.lchs.refinement._publish", forbidden)
    with pytest.raises(ValueError, match="max_dense_work"):
        result.verify(checks=LCHSRefinement(components=("fixed_pf",), max_dense_work=1, dense_validation=True))
