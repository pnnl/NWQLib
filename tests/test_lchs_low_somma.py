"""Actual selected Low–Somma (arXiv:2508.19238v2) PF body and explicit incomplete component bounds.

Independent coefficient/grid formulas live in test_lchs_providers. This file
uses the SELECT built during public preparation, not a second selection.
"""

import numpy as np
import pytest
import scipy.linalg

from nwqlib import LinearDynamics, plan, solve
from nwqlib.algorithms.lchs import LCHS


def test_low_somma_coefficient_rounding_is_refused_at_planning():
    """A shift c whose coefficient 1-norm defeats binary64 is refused, a moderate one kept.

    At c = 50 the Low-Somma coefficients have alpha about 1.2e16, so the
    binary64 finite sum carries rounding of order u*alpha, about 1.3 of the
    input norm, far above the published 2*epsilon/3 kernel and quadrature
    components. Near the allowance 0.1*epsilon = 1e-3, u*alpha is about
    3.4e-3 at c = 42 and 7.7e-4 at c = 40, which pins the threshold within a
    factor of 4.4. At c = 30, u*alpha is about 5e-7 and the classical sum
    still agrees with expm within the published bound.
    """
    from nwqlib.algorithms.lchs import ProviderConfig

    target = LinearDynamics(A=[[.4, .15], [.05, .25]], initial_state=[1., 0.], time=.1)

    def shifted(c):
        return LCHS(lchs_kernel=ProviderConfig(implementation="low_somma_f2", parameters={"c": c}),
                    k_quadrature=ProviderConfig(implementation="symmetric_uniform_trapezoid"))

    with pytest.raises(ValueError, match=r"u\*alpha = 1\.\d+ .* above 0\.1\*approximation_tolerance = 0\.001"):
        plan(target, method=shifted(50.), execution="classical")
    with pytest.raises(ValueError, match=r"u\*alpha = 0\.003\d* .* above 0\.1\*approximation_tolerance"):
        plan(target, method=shifted(42.), execution="classical")
    edge = plan(target, method=shifted(40.), execution="classical").reconstruction.coefficient_l1_norm
    assert 5e-4 < 2.0**-53*edge <= 1e-3
    result = solve(target, method=shifted(30.), execution="classical")
    alpha = result.plan.reconstruction.coefficient_l1_norm
    assert 1e9 < alpha and 2.0**-53*alpha <= 1e-3
    exact = scipy.linalg.expm(-np.array([[.4, .15], [.05, .25]])*.1) @ np.array([1., 0.])
    bound = next(item.fact.value.value for item in result.facts if item.fact.quantity == "algorithmic_approximation")
    assert np.linalg.norm(np.asarray(result.solution) - exact) <= bound
