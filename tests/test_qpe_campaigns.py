"""Bounded two-qubit seeded sampling and resolution regression relations.

These comparisons concern the stated instances and do not certify interval
coverage or multi-component identification.
"""

from math import pi, sqrt
import numpy as np
import pytest
from nwqlib import Eigenproblem, SpectralEstimation, plan, solve
from nwqlib.algorithms.qpe import QCELS, SPE, RFE, RWPE

ANCHOR = {
    "qcels": (QCELS, dict(num_times=8)),
    "spe": (SPE, dict(num_samples=64, fourier_degree=8, overlap_lower_bound=1.)),
    "rfe": (RFE, dict(num_samples=49, num_frequencies=25)),
    "rwpe": (RWPE, dict(max_steps=24)),
}


def campaign_plan(
    estimator, *, execution="classical", seed=17, matrix=None, phase=False, **settings
):
    method_type, defaults = ANCHOR[estimator]
    fields = {**defaults, **settings}
    matrix = np.diag([0.25, -0.5]) if matrix is None else matrix
    problem = (
        SpectralEstimation(unitary=matrix, initial_state=[0.0, 1.0])
        if phase
        else Eigenproblem(A=matrix)
    )
    method = method_type(initial_state=None if phase else [0.0, 1.0], **fields)
    return plan(
        problem,
        method=method,
        execution=execution,
        shots=(1 if estimator == "rwpe" else 4096) if execution == "quantum" else None,
        seed=seed,
    )


@pytest.mark.parametrize("method", ["qcels", "spe", "rfe"])
def test_qpe_counts_resolve_an_independent_eigenvalue(method):
    """Each sampled public estimator resolves H|1> = -0.5|1> at its grid and sampling scale."""
    tau = 0.9 * pi / ((3 if method == "spe" else 1) * 0.5)
    selected = campaign_plan(method, seed=0, execution="quantum")
    # QCELS refines minima bracketed by two neighboring cells of [-pi,pi).
    # SPE takes the first crossing on [-pi/2,pi/2), and RFE returns the
    # nearest Fourier cell on this single-mode seeded schedule.
    if method == "qcels":
        grid_window = 4*pi/(tau*selected.method.grid_size)
    elif method == "spe":
        grid_window = pi/(tau*selected.method.grid_size)
    else:
        grid_window = pi/(tau*selected.method.num_frequencies)
    # This fixed sampling allowance is a regression window for the fixture,
    # not an estimator confidence interval or a coverage guarantee.
    sampling_window = 4/(tau*sqrt(4096))
    result = solve(selected)
    assert abs(result.value + .5) <= grid_window + sampling_window


def test_existing_host_resolution_recipes():
    coarse = solve(campaign_plan("qcels", num_times=2))
    fine = solve(campaign_plan("qcels", num_times=4))
    # Both exact single-mode fits may reach roundoff; query count does not
    # establish a strict improvement in an already resolved value.
    assert coarse.value == pytest.approx(-0.5, abs=2e-13, rel=0)
    assert fine.value == pytest.approx(-0.5, abs=2e-13, rel=0)
    spe_matrix = np.diag([0.0, -0.7])
    coarse = solve(
        campaign_plan(
            "spe", seed=3, matrix=spe_matrix, num_samples=4, fourier_degree=3, grid_size=512
        )
    )
    fine = solve(
        campaign_plan(
            "spe", seed=3, matrix=spe_matrix, num_samples=8, fourier_degree=3, grid_size=512
        )
    )
    # The approved method stream changes the old sampled schedule. Both
    # workloads may already resolve the same cell; more queries do not imply
    # strict point-error improvement. Check the actual single-mode resolution.
    cell = 2 * pi / (0.9 * pi / (3 * 0.7) * 512)
    assert abs(coarse.value + 0.7) <= cell / 2 and abs(fine.value + 0.7) <= cell / 2
    hamiltonian = np.diag([.6, 2.4])
    coarse = solve(
        campaign_plan("rwpe", seed=1, matrix=hamiltonian, tau=1., max_steps=12)
    )
    fine = solve(
        campaign_plan("rwpe", seed=1, matrix=hamiltonian, tau=1., max_steps=16)
    )
    # More one-bit updates shrink the assumed Gaussian width, not every
    # realized point error. The same saved seeds preserve the common prefix.
    assert [(s.power, s.phase_shift, s.mean) for s in fine.samples[:12]] == [
        (s.power, s.phase_shift, s.mean) for s in coarse.samples]
    assert fine.gaussian.standard_deviation/coarse.gaussian.standard_deviation == pytest.approx(
        ((np.e-1)/np.e)**2, rel=2e-15, abs=0)
