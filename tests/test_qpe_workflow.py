"""Independent tiny numerical relations for QCELS, SPE, RFE and RWPE.

The former stochastic/native campaigns are not a prerequisite for these
falsifiers. Counts below are supplied populations, never new backend shots.
"""

from math import cos, pi, sin

import numpy as np
import pytest

from nwqlib.algorithms.qpe import SPE
from nwqlib.algorithms.qpe import numerical
from nwqlib.algorithms.qpe.records import public_estimate


def ideal_overlap(power, *, phases, weights):
    return sum(
        weight * complex(cos(power * phase), sin(power * phase))
        for phase, weight in zip(phases, weights, strict=True)
    )


def test_qcels_nonzero_phase_and_symmetric_two_component_center():
    powers = (0, 1, 2, 3, 4)
    phase, split = 0.37, 0.09
    for weights, phases in (((1.0,), (phase,)), ((0.5, 0.5), (phase - split, phase + split))):
        samples = [ideal_overlap(power, phases=phases, weights=weights) for power in powers]
        fitted = numerical.qcels(powers, samples, tau=None, grid_size=64)
        # Equal mixture = exp(i*m*phase)*cos(m*split); all cosines are
        # positive, so its fitted phase equals the midpoint (not either mode).
        assert fitted.value == pytest.approx(phase, abs=2e-8, rel=0)
        assert fitted.phase == pytest.approx(phase, abs=2e-8, rel=0)
        assert fitted.interval_method is None
        assert fitted.fit.interval_method == "unavailable"
        if len(phases) == 2:
            assert min(abs(fitted.value - mode) for mode in phases) > 0.08
        energy = numerical.qcels(powers, samples, tau=0.25, grid_size=64)
        assert energy.value == pytest.approx(-phase / 0.25, abs=8e-8, rel=0)


def test_qcels_residual_and_sparse_schedule_use_complex_signal():
    # Original sparse-time witness: adjacent samples wind more than pi.
    # Analytic exp(-i E*tau*p) supplies an oracle independent of phase unwrap.
    powers = (0, *np.linspace(1, 500, 16, dtype=int))
    for scale in (1., 1e12, 1e14):
        energy, tau = -.7*scale, .2/scale
        samples = [complex(cos(energy*tau*p), -sin(energy*tau*p)) for p in powers]
        fit = numerical.qcels(powers, samples, tau=tau, grid_size=64)
        assert fit.value/scale == pytest.approx(-.7, abs=2e-12, rel=0)
        assert fit.fit.effective_grid == 4001
        assert fit.fit.evaluations <= 65*4001
        # Exact single mode permits zero residual, up to arithmetic roundoff.
        assert fit.fit.residual < 1e-24
        assert fit.interval_low is fit.interval_high is fit.interval_method is None
    # Repeated powers are observations, not an invitation to invent phases.
    duplicate = numerical.qcels((0, 1, 1, 2), (1, np.exp(.4j), np.exp(.4j), np.exp(.8j)),
                               tau=None, grid_size=64)
    assert duplicate.value == pytest.approx(.4, abs=2e-13, rel=0)
    alias = numerical.qcels((0, 2, 4), np.exp(.4j*np.array((0, 2, 4))), tau=None, grid_size=64)
    assert abs(np.exp(2j*alias.value)-np.exp(.8j)) < 1e-12
    assert "aliased" in alias.stop_reason and alias.interval_low is None
    for powers, samples in (((0, 1, 2), (0, 0, 0)), ((0, 1, 2), (1, 0, 0)),
                            ((3, 3), (1, 1j))):
        flat = numerical.qcels(powers, samples, tau=None, grid_size=64)
        assert flat.value is None and "flat" in flat.stop_reason
        assert public_estimate(flat, phase_output=True) == (None, None, None)
    with pytest.raises(ValueError, match="finite complex"):
        numerical.qcels((0, 1), (1, np.nan), tau=None, grid_size=64)


@pytest.mark.parametrize("tau", [None, 0.25])
def test_spe_fourier_estimator_matches_independent_chebyshev_filter(tau):
    from scipy.special import iv

    degree, beta = 4, 3.
    bessel = np.exp(-beta)*iv(np.arange(degree+1), beta)
    # WBC arXiv:2110.12071v2, PDF Eq. (A3), HTML Eq. (18), followed by P=(1+Q)/2,
    # gives an independent Chebyshev
    # representation. Evaluate it at sin(x-y) for two spectral components.
    coefficients = np.zeros(2*degree+2)
    coefficients[0] = .5
    factor = np.sqrt(2*beta/pi)
    coefficients[1] = factor*bessel[0]
    for j in range(1, degree+1):
        coefficients[2*j+1] += factor*bessel[j]*(-1)**j/(2*j+1)
        coefficients[2*j-1] -= factor*bessel[j]*(-1)**j/(2*j-1)
    coordinates = np.array([-.9, -.5, .1, .6, 1.1])
    locations, population = (-.45, .65), (.3, .7)
    expected = sum(weight*np.polynomial.chebyshev.chebval(np.sin(coordinates-location), coefficients)
                   for weight, location in zip(population, locations, strict=True))
    weights = numerical.spe_fourier_weights(degree, beta)
    observed = np.zeros_like(coordinates)
    for j, probability in enumerate(weights/np.sum(weights)):
        power = 2*j+1
        sample = sum(weight*np.exp((1j if tau is None else -1j)*power*location)
                     for weight, location in zip(population, locations, strict=True))
        observed += probability*numerical.spe_cdf((power,), (sample,), coordinates,
            tau=tau, fourier_degree=degree, filter_beta=beta)
    # Short recurrences and sums in these two representations accumulate libm
    # and Bessel roundoff. The 2e-14 absolute window is far below sign defects.
    np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-14)


def test_spe_cdf_targets_the_low_energy_jump_of_a_mixed_spectrum():
    degree, beta = 12, 20.
    weights = numerical.spe_fourier_weights(degree, beta)
    # A fixed stratified population removes seed variation from the CDF
    # counterexample. These are scalar observations, not quantum acquisitions.
    multiplicities = np.rint(10_000*weights/np.sum(weights)).astype(int)
    powers = np.repeat(2*np.arange(degree+1)+1, multiplicities)
    samples = .3*np.exp(-1j*powers*(-.65)) + .7*np.exp(-1j*powers*.55)
    result = numerical.spe(powers, samples, tau=1., grid_size=512,
        fourier_degree=degree, filter_beta=beta, overlap_lower_bound=.3)
    # eta/2=.15 is the midpoint of the .3 ground-space jump. The gap is1.2,
    # so its smoothing tail is negligible here. The .01 window includes one
    # grid cell (pi/512) and the sub-percent stratification discrepancy.
    assert result.value == pytest.approx(-.65, rel=0, abs=.01)
    assert result.interval_low is result.interval_high is None


@pytest.mark.parametrize("grid_size", [4096])
def test_unitary_spe_reports_one_principal_phase_across_a_phase_sweep(grid_size):
    from nwqlib import SpectralEstimation, solve
    from nwqlib.problems import Eigenphase

    # angle_principal rounds x + pi, so it can return a number near a grid
    # point x rather than x itself. On the default 4096-point grid the phase
    # in turns then differs from the turns of x at 745 points. This sweep puts
    # the first crossing on such points. A value
    # taken from the raw coordinate would then differ from the phase, and
    # validate_plan would reject the Result.
    for phi in np.linspace(-1.4, 1.4, 15):
        unitary = np.diag([np.exp(1j*phi), np.exp(1j*(phi+1.2))])
        result = solve(SpectralEstimation(unitary=unitary, initial_state=[1., 0.]),
                       method=SPE(overlap_lower_bound=1., grid_size=grid_size),
                       output=Eigenphase(), execution="classical", seed=3)
        assert result.phase == (result.estimator_value/(2*pi)) % 1.0
        # The exact eigenstate signal gives C~(phi) = 1/2 exactly and an
        # increasing ACDF through phi, so the first crossing is the first
        # grid point at or above phi, less than one cell pi/grid_size away.
        assert 0 <= result.estimator_value - phi <= pi/grid_size + 1e-15


def test_unitary_spe_returns_the_highest_energy_of_exp_minus_i_tau_h():
    """SPE targets the lowest principal eigenphase, which for exp(-i tau H) is the highest energy."""
    import scipy.linalg
    from nwqlib import SpectralEstimation, solve
    from nwqlib.problems import Eigenphase

    energies, tau = np.array([-0.3, 0.45]), 1.0
    state = np.array([1.0, 1.0]) / np.sqrt(2)

    def turns(energy):
        return (-tau * energy / (2 * pi)) % 1.0

    def distance(a, b):
        d = abs(a - b) % 1.0
        return min(d, 1.0 - d)

    hamiltonian = solve(SpectralEstimation(hamiltonian=np.diag(energies), initial_state=state),
                        method=SPE(overlap_lower_bound=0.5, tau=tau), output=Eigenphase(),
                        execution="classical", seed=3)
    unitary = solve(SpectralEstimation(unitary=scipy.linalg.expm(-1j * tau * np.diag(energies)),
                                       initial_state=state),
                    method=SPE(overlap_lower_bound=0.5), output=Eigenphase(), execution="classical", seed=3)
    # The two phases lie 0.12 turns apart, far beyond the estimator's
    # 0.003-turn discrepancy on this population.
    assert distance(hamiltonian.phase, turns(-0.3)) < 0.01 < 0.1 < distance(hamiltonian.phase, turns(0.45))
    assert distance(unitary.phase, turns(0.45)) < 0.01 < 0.1 < distance(unitary.phase, turns(-0.3))


def test_spe_draw_distribution_and_quadratures_follow_the_filter():
    from scipy.special import iv

    class Draws:
        def choice(self, size, count, *, p):
            expected = np.array([iv(0, 1.5)+iv(1, 1.5),
                                 (iv(1, 1.5)+iv(2, 1.5))/3, iv(2, 1.5)/5])
            expected /= np.sum(expected)
            np.testing.assert_allclose(p, expected, rtol=2e-15, atol=0)
            assert (size, count) == (3, 3)
            return np.array([2, 0, 2])

    method = SPE(fourier_degree=2, filter_beta=1.5, num_samples=3, overlap_lower_bound=.3)
    schedule = numerical.planned_power_schedule(method, .2, Draws())
    assert schedule == tuple((power, phase) for power in (5, 1, 5) for phase in (0., -pi/2))


def test_rfe_largest_fourier_peak_has_correct_sign_and_mixed_spectrum_target():
    # Two orthogonal Fourier modes have exact DFT magnitudes .55 and .45.
    # The dominant mode is not the ground energy or a CDF threshold crossing.
    energies, weights = (2*pi*4/49, -2*pi*3/49), (.55, .45)
    powers = tuple(range(49))
    samples = [
        ideal_overlap(power, phases=tuple(-energy for energy in energies), weights=weights)
        for power in powers
    ]
    result = numerical.rfe(powers, samples, tau=1., num_frequencies=49)
    assert result.value == pytest.approx(energies[0], rel=0, abs=2e-14)
    assert result.phase == pytest.approx(-energies[0], rel=0, abs=2e-14)
    assert result.interval_low is result.interval_high is result.interval_method is None


def test_aliasing_and_public_unwrapped_turns_boundary():
    phase = -pi + 0.02
    powers = (0, 1, 2, 3, 4)
    samples = [ideal_overlap(power, phases=(phase,), weights=(1.0,)) for power in powers]
    result = numerical.qcels(powers, samples, tau=None, grid_size=64)
    value, turns, interval = public_estimate(result, phase_output=True)
    assert value == pytest.approx(phase, abs=1e-14)
    assert turns == pytest.approx(phase / (2 * pi) % 1.0, abs=1e-14)
    assert interval is None
    shifted = numerical.qcels(powers, samples, tau=1.0, grid_size=64)
    assert shifted.value == pytest.approx(-phase, abs=1e-14)
    # RFE covers the same full phase period, including a point near its cut.
    rfe_samples = [
        ideal_overlap(power, phases=(-pi + 0.02,), weights=(1.0,)) for power in range(64)
    ]
    rfe = numerical.rfe(range(64), rfe_samples, tau=1.0, num_frequencies=64)
    assert abs(rfe.value - (pi - 0.02)) <= pi/64


@pytest.mark.parametrize("datum", [0, 1])
def test_rwpe_one_bit_update_matches_independent_gaussian_integration(datum):
    from scipy.integrate import quad

    mean, std = .37, .63
    time, feedback = numerical.rwpe_choose(mean, std)

    def likelihood(y):
        # Hadamard interference gives this amplitude, independently of the
        # closed-form Gaussian update and its feedback-sign convention.
        phase = time*(mean+std*y)+feedback
        p0 = abs(1+complex(cos(phase), sin(phase)))**2/4
        return (p0 if datum == 0 else 1-p0)*np.exp(-y*y/2)/np.sqrt(2*pi)

    moments = [quad(lambda y: y**order*likelihood(y), -np.inf, np.inf,
                    epsabs=1e-12, epsrel=1e-12)[0] for order in range(3)]
    posterior_mean = mean+std*moments[1]/moments[0]
    posterior_std = std*np.sqrt(moments[2]/moments[0]-(moments[1]/moments[0])**2)
    assert numerical.rwpe_update(mean, std, datum) == pytest.approx(posterior_mean, rel=0, abs=2e-12)
    assert numerical.rwpe_scale(std, 1) == pytest.approx(posterior_std, rel=0, abs=2e-12)
    assert time == 1/std
    estimated = numerical.rwpe_estimate(mean, std, tau=.25)
    assert estimated.value == -mean/.25
    assert estimated.interval_low < estimated.value < estimated.interval_high


def test_hamiltonian_phase_interval_reverses_and_recenters_energy_bounds():
    estimate = numerical.NumericalEstimate(
        value=0.4,
        phase=-0.2,
        interval_low=0.3,
        interval_high=0.7,
        interval_method="supplied asymmetric energy interval",
        interpretation="test exact frame conversion",
    )
    raw, turns, interval = public_estimate(estimate, phase_output=True, tau=0.5)
    assert raw == 0.4
    assert turns == pytest.approx(1 - 0.2 / (2 * pi), abs=1e-15)
    assert interval.low == pytest.approx(1 - 0.35 / (2 * pi), abs=1e-15)
    assert interval.high == pytest.approx(1 - 0.15 / (2 * pi), abs=1e-15)
