"""Independent QLS numerical/subroutine witnesses; public consumers live in primary/quantum tests.

The larger historical reference cases remain explicit test selectors. This
file is not a finite native acceptance command or permission to run a campaign.
"""

import math
from fractions import Fraction
import numpy as np
import pytest
from qiskit.quantum_info import Operator
from _qsp_references import (
    cks_inverse_coefficients, snwpb_error_for_degree, snwpb_inverse_values,
    snwpb_mindegree_for_error,
)
from nwqlib.algorithms.qls import numerical as qls_numerical
from nwqlib.algorithms.qls.constants import QLS_TARGET_MARGIN
import nwqlib.subroutines.qsp.inverse as qsp_inverse
from nwqlib.subroutines.block_encoding import block_encoding_top_left, build_block_encoding
from nwqlib.subroutines.qsp import (
    build_real_chebyshev_encoding, chebyshev_polynomial_sup_bound, inverse_degree_law_bound,
    solve_symmetric_qsp_phases,
)
from nwqlib.subroutines.qsp.shortcut import _realize_kernel_reflection, plan_kernel_reflection

def _normed_qsp_target(coefficients: tuple[float, ...]) -> tuple[np.ndarray, float]:
    scale = chebyshev_polynomial_sup_bound(coefficients) * (1.0 + QLS_TARGET_MARGIN)
    return np.asarray(coefficients) / scale, scale



def test_selected_fit_degrees_stay_under_the_degree_law_bound() -> None:
    """Selected degrees stay under the registered degree-law bound.

    The fit's approximation certificate separately owns its error guarantee.
    """

    for kappa_be, epsilon_inv in ((4.0, 1.0e-2), (10.0, 1.0e-2), (10.0, 1.0e-3)):
        fit = qsp_inverse._fit_inverse_chebyshev(kappa_be, epsilon_inv)
        bound = inverse_degree_law_bound(kappa_be, epsilon_inv)
        assert fit.degree <= bound



def test_fit_integer_degree_anchors_with_recorded_margin() -> None:
    """Integer degree anchors, valid per the platform-quantities rule.

    Boundary margins (measured): at (kappa 4, eps 1e-2) the next-lower odd
    degree 21 fails its certificate by 1.9% of epsilon; at (kappa 8, eps
    1e-3) degree 61 fails by 24% — both astronomically above float64
    least-squares cross-platform noise, so the selected integers cannot
    flip between platforms.
    """

    fit = qsp_inverse._fit_inverse_chebyshev(4.0, 1.0e-2)
    assert fit.degree == 23 and fit.kappa == 4.0
    # The domain parameter must leave 1/kappa <= |x| <= 1 a positive width.
    with pytest.raises(ValueError, match="positive width"):
        qsp_inverse._fit_inverse_chebyshev(1.0, 1.0e-2)
    grid = np.linspace(0.25, 1.0, 25 * fit.degree)
    values = np.polynomial.chebyshev.chebval(grid, fit.coefficients)
    assert np.max(np.abs(4.0 * grid * values - 1.0)) <= 1.0e-2
    assert qsp_inverse._fit_inverse_chebyshev(8.0, 1.0e-3).degree == 63



def test_cks_cross_check_agrees_within_summed_certificates() -> None:
    """[CKS] (arXiv:1511.02306v2) closed form and the fit agree on D within their certificates."""

    kappa_be, epsilon_inv = 4.0, 1.0e-2
    fit = qsp_inverse._fit_inverse_chebyshev(kappa_be, epsilon_inv)
    cks = cks_inverse_coefficients(kappa_be, epsilon_inv)
    domain = np.linspace(1.0 / kappa_be, 1.0, 25 * (len(cks) - 1))
    fit_values = kappa_be * np.polynomial.chebyshev.chebval(domain, fit.coefficients)
    cks_values = np.polynomial.chebyshev.chebval(domain, cks)
    cks_certificate = float(np.max(np.abs(domain * cks_values - 1.0)))
    assert cks_certificate <= epsilon_inv  # the paper form holds its own bound
    disagreement = float(np.max(np.abs(domain * (fit_values - cks_values))))
    assert disagreement <= fit.certificate + cks_certificate + 1.0e-12



def test_fit_stays_primary_against_snwpb_frontier_and_sup() -> None:
    """The certified fit is at least as good as the optimality references.

    Under this pipeline's x-weighted certificate the LSQ fit needs no more
    degree than the [SNWPB] (arXiv:2507.15537v1) absolute-error-optimal
    family, and its
    sup_[-1,1] (hence the rescale, hence success probability) is the
    smallest of the three constructions — the recorded reason the fit
    stays primary (QLS contracts Sec. 2).
    """

    kappa_be, epsilon_inv = 10.0, 1.0e-2
    fit = qsp_inverse._fit_inverse_chebyshev(kappa_be, epsilon_inv)
    coefficients = np.asarray(fit.coefficients)
    # On D, |x p(x) - 1| = |x| |p(x) - 1/x| <= |p(x) - 1/x| because |x| <= 1.
    # The absolute-error degree of [SNWPB] arXiv:2507.15537v1, Eq. (8),
    # therefore bounds the search.
    search_limit = snwpb_mindegree_for_error(epsilon_inv, kappa_be)
    domain = np.linspace(1.0 / kappa_be, 1.0, 25 * search_limit)
    fit_values = np.polynomial.chebyshev.chebval(domain, coefficients)
    assert np.max(np.abs(kappa_be * domain * fit_values - 1.0)) <= epsilon_inv
    for reference_degree in range(1, search_limit + 1, 2):
        values = snwpb_inverse_values(domain, degree=reference_degree, kappa_be=kappa_be)
        if np.max(np.abs(domain * values - 1.0)) <= epsilon_inv:
            break
    else:
        pytest.fail("[SNWPB] arXiv:2507.15537v1 reference misses its own Eq. (8) degree")
    # The reference equioscillates, so its dense-grid maximum of |P - 1/x| reads the
    # [SNWPB] arXiv:2507.15537v1, Eq. (7), error to grid resolution (its
    # Eq. (26) 0.2% class).
    measured = float(np.max(np.abs(values - 1.0 / domain)))
    closed_form = snwpb_error_for_degree(reference_degree, kappa_be)
    assert measured == pytest.approx(closed_form, rel=5.0e-3, abs=0)
    assert fit.degree <= reference_degree

    # For d < N, a degree-d polynomial satisfies
    # sup_[-1,1] |p| <= max_j |p(cos(pi j/N))| / cos(d pi/(2N)) (Ehlich and Zeller,
    # doi:10.1007/BF01111276),
    # so the fit side is an upper bound and each reference side a sampled lower bound.
    n = 25 * max(fit.degree, reference_degree)
    nodes = np.cos(np.pi * np.arange(n + 1) / n)
    fit_sup = float(np.max(np.abs(np.polynomial.chebyshev.chebval(nodes, coefficients))))
    fit_sup /= np.cos(fit.degree * np.pi / (2 * n))
    # Odd parity covers x < 0. The quotient form loses accuracy as x approaches 0.
    positive = nodes[nodes >= 1.0e-3]
    snwpb = snwpb_inverse_values(positive, degree=reference_degree, kappa_be=kappa_be)
    cks = np.polynomial.chebyshev.chebval(positive, cks_inverse_coefficients(kappa_be, epsilon_inv))
    assert fit_sup < float(np.max(np.abs(snwpb))) / kappa_be
    assert fit_sup < float(np.max(np.abs(cks))) / kappa_be


def test_inverse_residual_norming_covers_unsampled_endpoint(monkeypatch):
    from nwqlib.subroutines.qsp import inverse
    from nwqlib.subroutines.qsp.phases import chebyshev_grid
    fit = qsp_inverse._fit_inverse_chebyshev(2., .9, max_degree=1)
    assert fit.degree == 1
    coefficient = fit.coefficients[1]
    # r(x)=2*c*x^2-1 is monotone on [.5,1], hence |r|'s maximum is an endpoint.
    exact_max = max(abs(.5*coefficient-1), abs(2*coefficient-1))
    nodes = .25*chebyshev_grid(fit.certificate_grid_points) + .75
    sampled = max(abs(2*coefficient*nodes**2-1))
    assert sampled < exact_max <= fit.certificate <= .9
    # Removing the norming factor would make this same certificate false.
    # This reuses the bounded degree-one fit; no native QSVT or large grid.
    monkeypatch.setattr(inverse, "chebyshev_norming_sup_bound",
                        lambda value, **kwargs: value)
    bad = qsp_inverse._fit_inverse_chebyshev(2., .9, max_degree=1)
    assert bad.certificate < exact_max



def test_kernel_reflection_polynomial_eq6_and_bounds() -> None:
    kr = _realize_kernel_reflection(plan_kernel_reflection(4.0, 1.0e-2))
    assert kr.kr_ell == int(np.ceil(4.0 * np.log(2.0 / 1.0e-2) / 2.0))
    grid = np.linspace(0.0, 1.0, 2001)
    values = np.polynomial.chebyshev.chebval(grid, kr.coefficients)
    assert np.polynomial.chebyshev.chebval(0.0, kr.coefficients) == pytest.approx(1.0, abs=1e-12)
    off_kernel = values[grid >= 1.0 / 4.0]
    assert np.max(off_kernel) <= -1.0 + 4.0e-2 / 1.01 + 1.0e-12
    assert np.min(off_kernel) >= -1.0 - 1.0e-12
    normed, _ = _normed_qsp_target(kr.coefficients)
    phases = solve_symmetric_qsp_phases(normed)
    assert phases.max_residual <= 1.0e-12



def test_real_pass_right_singular_semantics_on_non_normal_blocks() -> None:
    """Permanent review-spike guard: even -> Vp(S)V^dag, odd -> Wp(S)V^dag."""

    rng = np.random.default_rng(20260703)
    matrix = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    vector = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    non_normal = (np.eye(2) - np.outer(vector, vector.conj())) @ matrix
    encoding = build_block_encoding(non_normal, implementation="dense_dilation")
    encoded = block_encoding_top_left(
        Operator(encoding.circuit).data, num_ancillas=encoding.num_ancillas
    )
    left, singular_values, right_h = np.linalg.svd(encoded)
    assert np.max(np.abs(encoded.conj().T @ encoded - encoded @ encoded.conj().T)) >= 1.0e-2

    even = _realize_kernel_reflection(plan_kernel_reflection(2.0, 0.1))
    even_target, _ = _normed_qsp_target(even.coefficients)
    even_phases = solve_symmetric_qsp_phases(even_target)
    even_block = block_encoding_top_left(
        Operator(build_real_chebyshev_encoding(encoding, even_phases.phases)).data,
        num_ancillas=encoding.num_ancillas + 2,
    )
    even_expected = (
        right_h.conj().T
        @ np.diag(np.polynomial.chebyshev.chebval(singular_values, even_target))
        @ right_h
    )
    assert np.max(np.abs(even_block - even_expected)) <= 1.0e-12

    odd_coefficients = np.zeros(4)
    odd_coefficients[1] = 0.2
    odd_coefficients[3] = 0.1
    odd_phases = solve_symmetric_qsp_phases(odd_coefficients)
    odd_block = block_encoding_top_left(
        Operator(build_real_chebyshev_encoding(encoding, odd_phases.phases)).data,
        num_ancillas=encoding.num_ancillas + 2,
    )
    odd_expected = (
        left @ np.diag(np.polynomial.chebyshev.chebval(singular_values, odd_coefficients)) @ right_h
    )
    assert np.max(np.abs(odd_block - odd_expected)) <= 1.0e-12



def test_inverse_fit_checks_degree_one_before_doubling() -> None:
    assert qsp_inverse._fit_inverse_chebyshev(1.2, 0.2).degree == 1



def test_auto_kappa_resolves_small_magnitudes_without_squaring() -> None:
    values = np.concatenate([np.geomspace(1.0e-5, 1.0, 64), -np.geomspace(1.0e-5, 1.0, 64)])
    matrix = np.diag(values).astype(complex)
    sigma_min, sigma_max, method = qls_numerical._extreme_singular_values(matrix, hermitian=True)
    # Full magnitudes retain the small eigenvalues of an indefinite matrix.
    np.testing.assert_allclose(
        (sigma_min, sigma_max),
        (1.0e-5, 1.0),
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    assert method == "eigvalsh"



@pytest.mark.parametrize("method", ["shortcut_native_svp", "shortcut_dilation"])
def test_shortcut_theory_applies_polynomial_without_materializing_operator(monkeypatch, method) -> None:
    """An even polynomial acts on one start vector without another dense operator."""

    # G_t from the rank-one update equals the explicit complement-projector
    # product (I - v v^dagger) A_t, v = b'/||b'|| (numerical.projected_augmented_inplace),
    # within the scoped regression tolerance rtol=0,
    # atol=2e-12 at unit scale and original dimension two.
    from nwqlib.subroutines.dense_matrices import dalzell_augmented_matrix
    encoded = np.array([[.5, .2j], [-.1, .3]])
    rhs = np.array([.6, .8j])
    b_prime = np.concatenate((rhs, [1., 0.])) / np.sqrt(2.)
    b_prime /= np.linalg.norm(b_prime)
    projector = np.eye(4) - np.outer(b_prime, b_prime.conj())
    np.testing.assert_allclose(qls_numerical.shortcut_matrix(encoded, rhs, t_value=1.5),
                               projector @ dalzell_augmented_matrix(encoded, 1.5), rtol=0, atol=2e-12)
    # Several row tiles give the same product as one tile.
    for tile in (1, 3):
        tiled = qls_numerical.projected_augmented_inplace(
            dalzell_augmented_matrix(encoded, 1.5).astype(complex), b_prime, tile=tile)
        np.testing.assert_allclose(tiled, projector @ dalzell_augmented_matrix(encoded, 1.5), rtol=0, atol=2e-12)
    matrix = np.array([[.3, .2j, .1, 0], [.1, .4, 0, .2],
                       [0, .1, .3, 0], [.2, 0, 0, .1]], dtype=complex)
    coefficients = np.array([.2, 0, -.3, 0, .4])
    gram = matrix.conj().T @ matrix
    start = np.eye(4, dtype=complex)[:, 2]
    expected = (.2 * np.eye(4) - .3 * (2 * gram - np.eye(4))
                + .4 * (8 * gram @ gram - 8 * gram + np.eye(4))) @ start
    original_svd = np.linalg.svd

    class ActionMatrix(np.ndarray):
        def __matmul__(self, other):
            if self.ndim == 2 and np.ndim(other) == 2:
                raise AssertionError("polynomial action materialized a dense matrix product")
            return super().__matmul__(other)

        def conj(self, *args, **kwargs):
            if self.ndim == 2:
                raise AssertionError("polynomial action copied a conjugated dense matrix")
            return super().conj(*args, **kwargs)

    def action_svd(*args, **kwargs):
        left, singular_values, right_h = original_svd(*args, **kwargs)
        return left, singular_values, right_h.view(ActionMatrix)

    original_eigh = np.linalg.eigh

    def action_eigh(*args, **kwargs):
        values, vectors = original_eigh(*args, **kwargs)
        return values, vectors.view(ActionMatrix)

    monkeypatch.setattr(np.linalg, "svd", action_svd)
    monkeypatch.setattr(np.linalg, "eigh", action_eigh)
    branch, probability = qls_numerical.shortcut_polynomial_action(matrix, coefficients,
        method=method, system_dimension=2)
    # Independent T_2 and T_4 recurrences on G†G, with ample binary64 roundoff
    # margin. The lower block of the even polynomial of the dilation
    # [[0, G], [G†, 0]] is the same V K(Sigma) V† e_n.
    np.testing.assert_allclose(branch, expected[:2], rtol=1.e-13, atol=1.e-14)
    assert probability == pytest.approx(float(np.vdot(expected[:2], expected[:2]).real), rel=1.e-13, abs=1.e-14)



@pytest.mark.parametrize("kappa", [5., 20.])
def test_noisy_binary_search_paper_arithmetic_anchors(kappa):
    """Dalzell arXiv:2406.12086v2, Sec.5.2 trial/query formulas; deterministic model work is not execution."""
    from nwqlib.algorithms.qls.norm_search import _noisy_binary_search_norm
    _, source, metadata = _noisy_binary_search_norm(kappa_be=kappa, encoded_norm=4.)
    ladder_size = math.ceil(math.log2(kappa))+1
    rounds = math.ceil(math.log(ladder_size)/math.log(1.5))
    repetitions = math.ceil(72*math.log(40*rounds))
    per_trial = math.ceil(kappa*math.log(2/math.sqrt(1/8))/2)
    assert source == "noisy_binary_search"
    assert len(metadata["candidate_ladder"]) == ladder_size
    assert metadata["search_rounds"] == rounds
    assert metadata["repetitions_per_estimate"] == repetitions
    assert metadata["queries_per_trial"] == per_trial
    assert metadata["planned_trials_total"] == repetitions*rounds
    assert metadata["planned_queries_total"] == 2*repetitions*rounds*per_trial
    assert len(metadata["rounds"]) == rounds


@pytest.mark.parametrize("kappa", [1.01, 5., 20., 64.])
def test_noisy_binary_search_final_set_contains_encoded_norm(kappa):
    """Dalzell arXiv:2406.12086v2, Sec.5.2, keeps the ladder half containing the crossing
    t^2/(t^2+||x||^2)=1/2 at t=||x||."""
    from nwqlib.algorithms.qls.norm_search import _noisy_binary_search_norm
    # Encoded norms lie in [1, kappa]. Rungs and their one-ulp neighbours put the
    # probe comparison at the threshold, where rounding decides the outcome.
    rungs = sorted({2.**k for k in range(int(math.log2(kappa))+1)} | {kappa})
    norms = {1., kappa} | {math.sqrt(a*b) for a, b in zip(rungs, rungs[1:])}
    for rung in rungs:
        norms |= {rung, math.nextafter(rung, 0.), math.nextafter(rung, math.inf)}
    # The computed probability has relative error below 4u, and its t-slope is
    # 1/(2t) at the crossing, so a flipped comparison moves a kept end by < 4u*t.
    # The tolerance 4*eps = 8u leaves a factor of two.
    tolerance = 4*np.finfo(float).eps
    for norm in sorted(n for n in norms if 1. <= n <= kappa):
        chosen, _, metadata = _noisy_binary_search_norm(kappa_be=kappa, encoded_norm=norm)
        final = metadata["final_active_t"]
        assert final[0] <= norm*(1+tolerance) and final[-1] >= norm*(1-tolerance), norm
        assert final[0] <= chosen <= final[-1]


def test_linear_kappa_sequence_paper_arithmetic_anchors():
    """Dalzell arXiv:2406.12086v2, Sec.5.3 decreasing-sigma sequence and actual represented trial totals."""
    from nwqlib.algorithms.qls.norm_search import _linear_kappa_sequence_norm
    kappa=20.
    _, source, metadata = _linear_kappa_sequence_norm(kappa_be=kappa,
        left=np.eye(2), values=np.array([1., .25]), alpha=1., rhs=np.array([1.,1.]))
    length=math.ceil(math.log2(kappa))
    assert source == "linear_kappa_sequence"
    assert metadata["sequence_length"] == length
    assert metadata["visited_sigmas"] == [1.] + [2.**-step for step in range(1,length+1)]
    trials=queries=0
    base=math.ceil(100*math.log(80))
    for record in metadata["steps"]:
        step=record["step"]
        per_candidate=base*(1+length-step)
        per_trial=math.ceil(2.**step*math.log(80)/2)
        assert record["trials_per_candidate"] == per_candidate
        assert record["queries_per_trial"] == per_trial
        assert record["planned_trials"] == 4*per_candidate
        assert record["planned_queries"] == 8*per_candidate*per_trial
        trials += 4*per_candidate
        queries += 8*per_candidate*per_trial
    assert metadata["planned_trials_total"] == trials
    assert metadata["planned_queries_total"] == queries
    assert metadata["steps"][-1]["f_squared"] == 0.  # Last sigma is the exact dyadic reciprocal.
    # diag(1,1/4)^-1 (1,1)/sqrt2 = (1,4)/sqrt2.
    assert metadata["steps"][-1]["step_norm"] == pytest.approx(math.sqrt(17/2),rel=1e-13,abs=1e-14)
    # A complex frame weights |u_j^dagger b_hat|**2, with the conjugate on the frame.
    rng = np.random.default_rng(5)
    left, _, _ = np.linalg.svd(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    direction = rng.normal(size=4) + 1j * rng.normal(size=4)
    expected = np.abs(left.conj().T @ (direction / np.linalg.norm(direction))) ** 2
    np.testing.assert_allclose(qls_numerical.linear_model_weights(left, direction), expected, rtol=0, atol=2e-12)


def _exact_ceil_log2(value):
    """Smallest integer k with 2**k >= value, by exact rational comparison."""
    ratio = Fraction(value)
    k = ratio.numerator.bit_length() - ratio.denominator.bit_length()
    while Fraction(2)**k < ratio:
        k += 1
    while Fraction(2)**(k - 1) >= ratio:
        k -= 1
    return k


def test_ladder_length_is_exact_at_and_beside_powers_of_two():
    """The model ladder and the planning envelope both have ceil(log2 kappa)+1 rungs.

    Rounded logarithms err on both sides of 2**e. log2 returns e one ulp
    above 2**e, and log(x, 2) returns more than 29 at 2**29. The reference
    compares exact rationals with powers of two.
    """
    from nwqlib.algorithms.qls.host_planning import search_envelope
    from nwqlib.algorithms.qls.norm_search import _norm_search_candidate_ladder
    values = [1.01, 3., 20., 1.e4]
    for e in range(1, 64):
        power = 2.**e
        values += [math.nextafter(power, 0.), power, math.nextafter(power, math.inf)]
    for kappa in values:
        rungs = _exact_ceil_log2(kappa) + 1
        assert len(_norm_search_candidate_ladder(kappa)) == rungs, kappa
        assert search_envelope(kappa, "grid")[0] == rungs, kappa


def test_shortcut_eq17_window_arithmetic_and_poisson_anchor():
    """Independent arithmetic of Dalzell arXiv:2406.12086v2, Eq.17, keeps the eta-squared upper correction."""
    from nwqlib.algorithms.qls.norm_search import _success_center, _success_window
    from nwqlib.subroutines.qsp.shortcut import dalzell_eta_from_precision
    eta=.1/math.sqrt(2)
    selected_eta=dalzell_eta_from_precision(.1)
    assert 2*selected_eta**2==pytest.approx(.1**2,rel=2e-15,abs=0.)
    # For four-point unscaled Poisson and normalized all-ones RHS,
    # A^-1 b = (1,3/2,3/2,1); alpha=2+2cos(pi/5).
    alpha=2+2*math.cos(math.pi/5)
    norm=alpha*math.sqrt(13/2)
    t=8.
    center=4*t*t*norm*norm/(t*t+norm*norm)**2
    assert _success_center(norm,t) == pytest.approx(center,rel=1e-14,abs=1e-15)
    lower,upper=_success_window(center,eta)
    assert lower == pytest.approx(center*(1-eta)**2/(1+eta)**2,rel=1e-14,abs=1e-15)
    assert upper == pytest.approx(center+4*eta**2/(1+eta)**2,rel=1e-14,abs=1e-15)
    assert upper < center+eta  # Falsifies the former eta^2 -> eta defect.


def test_degree_convergence_error_strictly_shrinks():
    """Preserved fixed diagonal fit convergence; no claim for arbitrary inputs."""
    matrix=np.diag([1.,1/8])
    rhs=np.array([1.,1.])/math.sqrt(2)
    reference=np.array([1.,8.])/math.sqrt(2)
    errors=[]
    degrees=[]
    for epsilon in (.1,.01,.001):
        fit=qsp_inverse._fit_inverse_chebyshev(8.,epsilon)
        branch,_=qls_numerical.inverse_polynomial_action(np.eye(2),np.diag(matrix),rhs,alpha=1.,coefficients=fit.coefficients)
        errors.append(float(np.linalg.norm(8*branch-reference)/np.linalg.norm(reference)))
        degrees.append(fit.degree)
        assert errors[-1]<=epsilon
    assert errors[0]>errors[1]>errors[2]
    assert degrees[0]<degrees[1]<degrees[2]
