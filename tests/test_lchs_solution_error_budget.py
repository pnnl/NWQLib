"""Stage-bound composition contracts for LCHS solution-level error accounting."""

from __future__ import annotations

import numpy as np
import pytest

from nwqlib.algorithms.lchs.solution_error_budget import (
    LCHSApplicationRecord,
    application_stage_bounds,
    circuit_stage_bounds,
)


RAW_BOUNDS = {
    "approximate_lchs_error_bound": 0.25,
    "quadrature_error_bound": 0.125,
}
INCOMPLETE_PROFILE = {
    **RAW_BOUNDS,
    "solution_error_certificate_status": "incomplete",
    "kernel_parameter_selection": {"profile": "paper_closed_form_f2_1"},
    "unbudgeted_error_stages": [
        "trotter_synthesis",
        "lcu_coefficient_preparation",
        "initial_state_preparation",
    ],
}


def _application(
    *,
    start_time: float = 0.0,
    elapsed_time: float = 1.0,
    weight: float = 2.0,
    recovery_scale: float = 3.0,
    synthesis_error_bound: float | None = 0.125,
) -> LCHSApplicationRecord:
    return LCHSApplicationRecord(
        start_time=start_time,
        elapsed_time=elapsed_time,
        weight=weight,
        recovery_scale=recovery_scale,
        synthesis_error_bound=synthesis_error_bound,
    )


def _classical(*, raw_record=RAW_BOUNDS, **overrides):
    arguments = {
        "applications": (_application(),),
        "backend": "dense_exact",
        "psd_premise_satisfied": True,
    }
    arguments.update(overrides)
    return application_stage_bounds(raw_record, **arguments)


def _circuit(*, raw_record=RAW_BOUNDS, **overrides):
    arguments = {
        "applications": (_application(),),
        "backend": "dense_exact",
        "inhomogeneous": False,
        "psd_premise_satisfied": True,
        "gamma": 1.0,
        "delta_lcu": 0.0,
        "input_preparation_output_error_bound": 0.0,
    }
    arguments.update(overrides)
    return circuit_stage_bounds(raw_record, **arguments)


def test_stage_bound_manifest_and_composition() -> None:
    """Each stage is its own physical L2 term with an independently derived weight.

    Application a has weight w_a and PSD recovery scale r_a. Operator-level
    truncation and k-quadrature bounds e contribute sum_a w_a*r_a*e, and node
    synthesis bounds s_a contribute sum_a w_a*r_a*s_a. The prepared coefficient
    state enters PREP and PREP-dagger, so its amplitude error contributes
    2*gamma*delta_lcu, and it scales the compensated QSP bound once by gamma.
    Dyadic inputs make every expected sum exact in binary64.
    """
    applications = (
        _application(weight=2.0, recovery_scale=3.0, synthesis_error_bound=0.125),
        _application(
            start_time=0.5,
            elapsed_time=0.5,
            weight=0.5,
            recovery_scale=4.0,
            synthesis_error_bound=0.25,
        ),
    )
    truncation = 2.0 * 3.0 * 0.25 + 0.5 * 4.0 * 0.25
    k_quadrature = 2.0 * 3.0 * 0.125 + 0.5 * 4.0 * 0.125
    synthesis = 2.0 * 3.0 * 0.125 + 0.5 * 4.0 * 0.25

    kernel_stages = {"kernel_approximation": (truncation, None), "k_quadrature": (k_quadrature, None)}
    for backend in ("dense_exact", "qsp_block_encoding"):
        classical = _classical(applications=applications, backend=backend)
        assert list(classical) == ["kernel_approximation", "k_quadrature"]
        assert classical == kernel_stages
    for backend in ("trotter", "trotter_error_budgeted"):
        classical = _classical(applications=applications, backend=backend)
        assert list(classical) == ["kernel_approximation", "k_quadrature", "trotter_synthesis"]
        assert classical == {**kernel_stages, "trotter_synthesis": (synthesis, None)}

    circuit = _circuit(
        applications=applications,
        backend="trotter_error_budgeted",
        inhomogeneous=True,
        gamma=3.0,
        delta_lcu=0.125,
        input_preparation_output_error_bound=0.25,
        duhamel_quadrature_error_bound=0.5,
    )
    assert list(circuit) == [
        "kernel_approximation",
        "k_quadrature",
        "trotter_synthesis",
        "duhamel_quadrature",
        "lcu_coefficient_preparation",
        "initial_state_preparation",
    ]
    assert circuit == {
        **kernel_stages,
        "trotter_synthesis": (synthesis, None),
        "duhamel_quadrature": (0.5, None),
        "lcu_coefficient_preparation": (2.0 * 3.0 * 0.125, None),
        "initial_state_preparation": (0.25, None),
    }

    dense = _circuit()
    assert list(dense) == [
        "kernel_approximation",
        "k_quadrature",
        "lcu_coefficient_preparation",
        "initial_state_preparation",
    ]
    assert dense["kernel_approximation"] == (2.0 * 3.0 * 0.25, None)

    qsp = _circuit(backend="qsp_block_encoding", gamma=3.0, compensated_recovery_error_bound=0.125)
    assert list(qsp) == [
        "kernel_approximation",
        "k_quadrature",
        "qsp_synthesis",
        "lcu_coefficient_preparation",
        "initial_state_preparation",
    ]
    assert qsp["qsp_synthesis"] == (3.0 * 0.125, None)


def test_stage_bound_suppression_and_invalid_inputs() -> None:
    """A failed premise or missing stage suppresses its scope, and exact zero differs from
    underflowed nonzero error.
    """
    kernel_values = {"kernel_approximation": 2.0 * 3.0 * 0.25, "k_quadrature": 2.0 * 3.0 * 0.125}

    # The PSD premise is shared by every classical stage; circuit preparation
    # stages do not depend on it.
    assert _classical(backend="trotter", psd_premise_satisfied=False) == {
        name: (None, "uncertified_psd_premise")
        for name in ("kernel_approximation", "k_quadrature", "trotter_synthesis")
    }
    assert _circuit(backend="trotter", psd_premise_satisfied=False, gamma=2.0, delta_lcu=0.125) == {
        "kernel_approximation": (None, "uncertified_psd_premise"),
        "k_quadrature": (None, "uncertified_psd_premise"),
        "trotter_synthesis": (None, "uncertified_psd_premise"),
        "lcu_coefficient_preparation": (2.0 * 2.0 * 0.125, None),
        "initial_state_preparation": (0.0, None),
    }
    application_cases = (
        ((), "missing_stage:applications"),
        ((_application(weight=np.inf),), "nonfinite_stage:applications"),
        ((_application(recovery_scale=0.0),), "unusable_stage:applications"),
    )
    for applications, reason in application_cases:
        assert _classical(applications=applications) == {
            "kernel_approximation": (None, reason),
            "k_quadrature": (None, reason),
        }
        circuit = _circuit(applications=applications)
        assert circuit["kernel_approximation"] == (None, reason)
        assert circuit["initial_state_preparation"] == (0.0, None)

    stage_cases = (
        ({"raw_record": {"quadrature_error_bound": 0.125}}, "kernel_approximation", "missing_stage:kernel_approximation"),
        (
            {"raw_record": {**RAW_BOUNDS, "approximate_lchs_error_bound": np.inf}},
            "kernel_approximation",
                "nonfinite_stage:kernel_approximation",
        ),
        (
            {"raw_record": {**RAW_BOUNDS, "approximate_lchs_error_bound": -1.0}},
            "kernel_approximation",
            "unusable_stage:kernel_approximation",
        ),
        (
            {"raw_record": {**RAW_BOUNDS, "approximate_lchs_error_bound": 1.0e308}},
            "kernel_approximation",
                "nonfinite_stage:kernel_approximation",
        ),
        ({"unusable_stages": {"k_quadrature"}}, "k_quadrature", "unusable_stage:k_quadrature"),
    )
    for overrides, stage, reason in stage_cases:
        other = "k_quadrature" if stage == "kernel_approximation" else "kernel_approximation"
        assert _classical(**overrides) == {stage: (None, reason), other: (kernel_values[other], None)}
    assert _classical(
        backend="trotter",
        applications=(_application(synthesis_error_bound=None),),
    ) == {
        "kernel_approximation": (kernel_values["kernel_approximation"], None),
        "k_quadrature": (kernel_values["k_quadrature"], None),
        "trotter_synthesis": (None, "missing_stage:trotter_synthesis"),
    }
    circuit_cases = (
        ({"gamma": None}, "lcu_coefficient_preparation"),
        ({"input_preparation_output_error_bound": None}, "initial_state_preparation"),
        ({"backend": "qsp_block_encoding"}, "qsp_synthesis"),
        ({"inhomogeneous": True}, "duhamel_quadrature"),
    )
    for overrides, stage in circuit_cases:
        circuit = _circuit(**overrides)
        assert circuit[stage] == (None, f"missing_stage:{stage}")
        assert circuit["kernel_approximation"] == (kernel_values["kernel_approximation"], None)

    # A registered incomplete profile keeps its own caps and names each
    # unbudgeted stage.
    assert _circuit(raw_record=INCOMPLETE_PROFILE, backend="trotter") == {
        "kernel_approximation": (kernel_values["kernel_approximation"], None),
        "k_quadrature": (kernel_values["k_quadrature"], None),
        **{
            name: (None, f"unbudgeted_stage:paper_closed_form_f2_1:{name}")
            for name in INCOMPLETE_PROFILE["unbudgeted_error_stages"]
        },
    }

    smallest = np.nextafter(0.0, 1.0)
    assert _classical(
        applications=(_application(weight=smallest, recovery_scale=1.0),),
        raw_record={"approximate_lchs_error_bound": 0.5, "quadrature_error_bound": 0.0},
    ) == {
        "kernel_approximation": (None, "unusable_stage:kernel_approximation"),
        "k_quadrature": (0.0, None),
    }

    with pytest.raises(ValueError, match="off-path"):
        _classical(unusable_stages={"qsp_synthesis"})
    with pytest.raises(ValueError, match="selected path"):
        _circuit(unusable_stages={"qsp_synthesis"})
    for bounds in (_classical, _circuit):
        for stages in ([], ["unregistered_stage"]):
            with pytest.raises(ValueError, match="registered unbudgeted_error_stages"):
                bounds(raw_record={**INCOMPLETE_PROFILE, "unbudgeted_error_stages": stages})
        with pytest.raises(ValueError, match="named parameter profile"):
            bounds(raw_record={**INCOMPLETE_PROFILE, "kernel_parameter_selection": {}})


@pytest.mark.parametrize("matrix,initial,time", [
    ([[0.4, 0.15], [0.05, 0.25]], [20.0, 0.0], 0.1),  # PSD L, no shift: the input norm 20 scales the bound.
    ([[-2.0, 0.1], [0.1, 0.2]], [1.0, 0.0], 1.0),  # Negative spectrum: exp(psd_shift*t) scales it.
])
def test_kernel_bound_fields_are_unit_input_and_published_facts_are_physical(matrix, initial, time):
    """The reconstruction bounds are per unit input before PSD growth, the Result facts physical."""
    from nwqlib import LinearDynamics, solve
    from nwqlib.algorithms import LCHS

    result = solve(LinearDynamics(A=matrix, initial_state=initial, time=time), method=LCHS(),
                   execution="classical")
    rec = result.plan.reconstruction
    weight = np.linalg.norm(initial) * np.exp(rec.psd_shift * time)
    facts = {fact.fact.quantity: fact.fact.value.value for fact in result.facts
             if fact.fact.quantity in {"kernel_approximation", "k_quadrature"}}
    assert facts["kernel_approximation"] == pytest.approx(weight * rec.kernel_approximation_bound, rel=1e-13)
    assert facts["k_quadrature"] == pytest.approx(weight * rec.quadrature_bound, rel=1e-13)
    assert "Selected kernel bounds per unit input before PSD growth:" in str(result)
