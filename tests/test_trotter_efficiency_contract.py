"""Explicit fixed-PF certificates at the selected schedule, with actual work caps."""

import numpy as np
import pytest

from nwqlib import LinearDynamics, plan, solve
from nwqlib.algorithms.lchs import LCHS, LCHSRefinement
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.subroutines.trotterization import error_budget


def selected(*, execution="classical", identity=False):
    return plan(
        LinearDynamics(
            A=np.eye(2) if identity else [[0.45, 0.08j], [0.03j, 0.7]],
            initial_state=[1.0, 0.0] if identity else [1.0, 0.2],
            time=0.2,
        ),
        method=LCHS(
            approximation_tolerance=0.7,
            lchs_kernel=ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": 0.8}),
            k_quadrature=ProviderConfig(
                implementation="composite_gauss", parameters={"truncation_multiplier": 1.0}
            ),
            hamiltonian_evolution_backend="trotter",
            trotter_steps=2,
        ),
        execution=execution,
        seed=7,
    )


def arguments(receipt):
    return {item.parameter: item.value for item in receipt.applications[0].arguments}


@pytest.fixture(scope="module")
def fixed_result():
    return solve(selected(execution="quantum"))


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
    assert result.references == "not_run"
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


@pytest.mark.parametrize("control", ["max_steps", "max_structural_work", "max_dense_work"])
def test_fixed_refinement_checks_its_concrete_limit_before_expensive_work(
    control, monkeypatch, fixed_result
):
    from nwqlib.algorithms.lchs import time_independent_terms as numerical

    result = fixed_result

    def forbidden(*args, **kwargs):
        pytest.fail("refinement crossed its unadmitted numerical boundary")

    from nwqlib.subroutines.trotterization import error_budget

    # Refinement reads the stored node table and never decomposes. max_steps
    # rejects before any work, max_structural_work before the table read and
    # the census, and max_dense_work before the requested dense commutators.
    monkeypatch.setattr(numerical, "_trotter_pauli_decomposition", forbidden)
    options = {control: 1}
    if control == "max_structural_work":
        # One unit below the stored-table read W_read = sum_k (p_k + 2).
        table = numerical.stored_pf_nodes(result.plan._native["native_data"])
        options[control] = numerical.stored_node_read_work(len(node["union_indices"]) for node in table.nodes) - 1
    if control == "max_dense_work":
        monkeypatch.setattr(error_budget, "dense_trotter_bound_coefficient", forbidden)
        options["dense_validation"] = True
    else:
        monkeypatch.setattr(numerical, "_node_bound_coefficients", forbidden)
    monkeypatch.setattr("nwqlib.algorithms.lchs.refinement._publish", forbidden)
    message = control + (" before reading its node table" if control == "max_structural_work" else "")
    with pytest.raises(ValueError, match=message):
        result.verify(checks=LCHSRefinement(components=("fixed_pf",), **options))


def test_fixed_identity_certificate_preserves_selected_repetitions_and_zero_bound():
    choice = selected(execution="quantum", identity=True)
    result = solve(choice)
    schedule = choice.reconstruction.step_counts
    _, facts = result.verify(checks=LCHSRefinement(components=("fixed_pf",)))
    # Fixed selection keeps two repetitions; the traceless generator vanishes
    # and each actual branch only carries its exact identity phase.
    assert schedule and set(schedule) == {2}
    assert choice.reconstruction.step_counts == schedule
    assert (
        next(fact.fact.value.value for fact in facts if fact.fact.quantity == "trotter_synthesis")
        == 0.0
    )
