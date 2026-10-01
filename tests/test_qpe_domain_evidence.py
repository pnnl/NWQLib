"""Filter domains, finite walk reach and the measure of reused Fourier draws."""

from math import asin, pi, sqrt

import numpy as np
import pytest
from scipy.special import lambertw

from nwqlib import Eigenproblem, plan, solve
from nwqlib.algorithms.qpe import RFE, RWPE, SPE
from nwqlib.algorithms.qpe import numerical
from nwqlib.algorithms.qpe.records import identification_note


def test_spe_auto_time_respects_filter_transition_and_notes_supplied_violation():
    eta, beta = .15, 1.
    delta = asin(sqrt(float(lambertw(8/(pi*eta**2)).real)/(4*beta)))
    problem = Eigenproblem(A=np.diag([0., 1.]))
    method = SPE(initial_state=np.sqrt([eta, 1-eta]), overlap_lower_bound=eta, filter_beta=beta)
    selected = plan(problem, method=method, seed=17)
    assert selected.reconstruction.tau == pytest.approx(.9*(pi/2-delta), rel=2e-15)
    assert identification_note(selected) is None
    unsafe = plan(problem, method=method.revise(tau=.3*pi), seed=17)
    assert "transition half-width" in identification_note(unsafe)
    unresolved = plan(problem, method=method.revise(fourier_degree=1), seed=17)
    assert "approximation premise" in identification_note(unresolved)


def test_spe_wrap_crossing_is_unidentified_instead_of_an_endpoint_energy():
    eta, beta, degree = .3, 6., 11
    weights = numerical.spe_fourier_weights(degree, beta)
    powers = 2*np.arange(degree+1)+1
    counts = np.maximum(1, np.rint(512*weights/weights.sum()).astype(int))
    samples = eta+(1-eta)*np.exp(-1j*.49*pi*powers)
    # This legal finite draw histogram approximates the infinite-sample
    # two-level CDF. Its first value exceeds the requested ground threshold.
    first = .5+2*weights.sum()*np.dot(counts, np.imag(np.exp(-.5j*pi*powers)*samples))/counts.sum()
    assert first > eta/2
    result = numerical.spe(powers, samples, tau=1., grid_size=64,
        fourier_degree=degree, filter_beta=beta, overlap_lower_bound=eta, multiplicities=counts)
    assert result.value is None and "scan boundary" in result.stop_reason


def test_grouped_draws_keep_fourier_draw_weights_and_pooled_shot_population():
    powers, samples, counts = (0, 1, 3), (1, 1, 1j), (20, 10, 1)
    grid = 2*pi*np.arange(8)/8
    coefficients = [sum(n*z*np.exp(-1j*k*x) for k, z, n in zip(powers, samples, counts))/sum(counts)
                    for x in grid]
    expected = (grid[int(np.argmax(np.abs(coefficients)))]+pi)%(2*pi)-pi
    actual = numerical.rfe(powers, samples, tau=None, num_frequencies=8, multiplicities=counts)
    assert actual.phase == pytest.approx(expected, abs=1e-15)
    assert actual.phase != numerical.rfe(powers, samples, tau=None, num_frequencies=8).phase
    problem = Eigenproblem(A=np.diag([.1, .2]))
    method = RFE(initial_state=[1, 0], num_samples=12, num_frequencies=2)
    exact = plan(problem, method=method, seed=17)
    sampled = plan(problem, method=method, shots=5, seed=17)
    assert len(exact.reconstruction.queries) == 4
    assert sum(q.multiplicity for q in exact.reconstruction.queries) == 24
    # Shots pool the same draw schedule: one query per setting, whose batch
    # requests shots times its multiplicity. The exact queries instead share
    # the trajectory's points.
    def draws(queries):
        return tuple((q.experiment, q.power, q.phase_shift, q.multiplicity) for q in queries)
    assert draws(sampled.reconstruction.queries) == draws(exact.reconstruction.queries)
    assert {q.acquisition for q in exact.reconstruction.queries} == {"trajectory"}
    nodes = {d.id: d.node for d in sampled.construction.program.definitions}
    assert ([nodes[e.batch].repetitions for e in sampled.experiments]
            == [5*q.multiplicity for q in sampled.reconstruction.queries])


def _upward_fraction(x):
    from fractions import Fraction
    from math import inf, nextafter

    y = float(x)
    return nextafter(y, inf) if Fraction(y) < x else y


def pooling_tolerances(M, factor, Q=2.0):
    """Return (T_SPE, T_RFE) for comparing separate and pooled Fourier arrays.

    Pooling preserves the exact linear estimator for the same complete count
    populations. Array comparisons use an absolute forward-error allowance
    for mean formation, weighting and accumulation, and do not require equal
    decisions at numerical ties.

    Result. Compare arrays with an absolute forward-error bound. Exact
    equality is unnecessary, and an all-zero witness would not detect a
    dropped multiplicity.

    Condition on the schedule. Let M=sum(m_j) and U be its distinct-power
    count. Each separate draw returns s shots in each quadrature. For the
    detector use s=8 and compute (2*n0-n)/n with Python-integer numerator
    arithmetic. Pool the same realized counts by integer addition before
    division. Both paths use the same grid, parameters and computed complex
    phase factors E_jl. Require finite arithmetic, gradual underflow and
    (M+8)u<1. Put

        gamma_k = k u/(1-k u),
        Q = max_(j,l) (abs(Re E_jl)+abs(Im E_jl)).

    Treating the common computed phase factors as data avoids an unnecessary
    libm accuracy premise. A simple detector may first check Q<=2 and use 2.
    No grid-size multiplier is needed for an entrywise deterministic bound.

    For SPE let f be the common binary64 value 2*sum(weights)/M computed by
    the source. Sufficient tolerances are

        T_SPE = 2 gamma_(M+8) (1/2 + abs(f) M Q),
        T_RFE = 4 gamma_(M+8) Q.

    T_RFE bounds complex modulus and therefore works with a complex
    assert_allclose using rtol=0. For one repeated power M=m.

    Derivation. The exact rational count means satisfy sum_r x_r=m*x_pool and
    the corresponding imaginary identity. Each real or imaginary mean has
    magnitude at most one. For these small normal-range counts, its correctly
    rounded division has relative error at most u, with zero exact. The
    imaginary complex product uses two real products and an addition. SPE
    also forms f*m and accumulates at most M terms into 1/2. Eight
    pre-accumulation/weight roundings conservatively cover both arithmetic
    paths. Each path has forward error at most gamma_(M+8)*(1/2+abs(f)*M*Q)
    relative to the same exact linear expression in the stored phase
    factors. The pooled path has U<=M additions. Add both errors.

    For RFE, normalized exact weights sum to one. Each output component on
    each path has error at most gamma_(M+8)*Q, including input means, m/M
    and complex products. Adding sides and converting the two component
    bounds to modulus costs 2*sqrt(2), bounded by the stated factor 4. These
    are array bounds, not equality of first-crossing or argmax decisions at
    ties.

    These bounds do not cover an arbitrary alternative mean-formation path
    or overflow. The pooling identity itself is stated where queries record
    their multiplicity (records.QPEQuery).
    """
    from fractions import Fraction

    if type(M) is not int or M < 1 or M+8 >= 2**53:
        raise ValueError("invalid detector draw count")
    gamma = Fraction(M+8, 2**53-M-8)
    scale = Fraction(1, 2) + abs(Fraction(factor))*M*Fraction(Q)
    return _upward_fraction(2*gamma*scale), _upward_fraction(4*gamma*Fraction(Q))


@pytest.mark.parametrize("tau", (None, 1.))
@pytest.mark.parametrize("powers,pairs", (
    # Multiplicities 3, 3, 2 and 1 over both quadratures.
    ((1, 3, 1, 5, 3, 1, 7, 5, 3),
     ((8, 1), (2, 7), (5, 4), (6, 3), (1, 8), (4, 2), (3, 6), (7, 5), (0, 4))),
    # The real draw means of power 1 cancel exactly, its imaginary ones do not.
    ((1, 1, 3), ((6, 3), (2, 2), (8, 8))),
))
def test_pooled_draw_counts_give_the_separate_draw_spe_cdf(powers, pairs, tau):
    """Pooling the same eight-shot draws keeps the SPE ACDF array within T_SPE (pooling_tolerances)."""
    shots, degree, beta = 8, 3, 6.
    coordinates = np.linspace(-pi/2, pi/2, 129, endpoint=False)
    separate = [complex((2*real - shots)/shots, (2*imag - shots)/shots) for real, imag in pairs]
    pooled = {}
    for power, (real, imag) in zip(powers, pairs):
        count, zeros_real, zeros_imag = pooled.get(power, (0, 0, 0))
        pooled[power] = (count + 1, zeros_real + real, zeros_imag + imag)
    means = [complex((2*real - count*shots)/(count*shots), (2*imag - count*shots)/(count*shots))
             for count, real, imag in pooled.values()]
    multiplicities = [count for count, _, _ in pooled.values()]
    assert len(pooled) < len(powers) and 1 in multiplicities
    Q = max(np.max(np.abs(np.cos(power*coordinates)) + np.abs(np.sin(power*coordinates)))
            for power in powers)
    assert Q <= 2
    factor = 2*float(np.sum(numerical.spe_fourier_weights(degree, beta)))/len(powers)
    tolerance, _ = pooling_tolerances(len(powers), factor)
    options = dict(tau=tau, fourier_degree=degree, filter_beta=beta)
    expected = numerical.spe_cdf(powers, separate, coordinates, **options)
    actual = numerical.spe_cdf(tuple(pooled), means, coordinates, multiplicities=multiplicities, **options)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    # Dropping the multiplicities moves the array far outside the allowance.
    dropped = numerical.spe_cdf(tuple(pooled), means, coordinates, **options)
    assert np.max(np.abs(dropped - expected)) > 1e6*tolerance


def test_rwpe_public_estimate_and_out_of_reach_interval_disclosure():
    problem = Eigenproblem(A=np.diag([.1, .2]))
    method = RWPE(initial_state=[1, 0], tau=1., prior_std=.25, max_steps=20)
    result = solve(problem, method=method, execution="classical", seed=7)
    # For this seed the nominal 95-percent interval, about 0.01 wide,
    # contains the eigenvalue 0.1.
    assert result.interval.low <= .1 <= result.interval.high
    assert identification_note(result.plan) is None
    unreachable = solve(Eigenproblem(A=np.diag([4., 5.])),
        method=RWPE(initial_state=[1, 0], tau=3.), execution="classical", seed=7)
    note = identification_note(unreachable.plan)
    assert "finite reach" in note
    assert unreachable.interval.interpretation.startswith(note)
