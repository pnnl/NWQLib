"""Independent QLS numerical/subroutine witnesses; public consumers live in primary/quantum tests.

The larger historical reference cases remain explicit test selectors. This
file is not a finite native acceptance command or permission to run a campaign.
"""

from fractions import Fraction
import math
import re
import numpy as np
import pytest
from qiskit.quantum_info import Operator
from _qsp_references import cks_inverse_coefficients, restricted_inverse_residual_bound
from nwqlib.algorithms.qls import numerical as qls_numerical
from nwqlib.algorithms.qls.constants import QLS_TARGET_MARGIN
from nwqlib.subroutines.block_encoding import block_encoding_top_left, build_block_encoding
from nwqlib.subroutines.qsp import build_real_chebyshev_encoding, chebyshev_polynomial_sup_bound, solve_wx_qsp_phases
from nwqlib.subroutines.qsp.inverse import _inverse_chebyshev, _table
from nwqlib.subroutines.qsp.shortcut import _realize_kernel_reflection, plan_kernel_reflection

def _normed_qsp_target(coefficients: tuple[float, ...]) -> tuple[np.ndarray, float]:
    scale = chebyshev_polynomial_sup_bound(coefficients) * (1.0 + QLS_TARGET_MARGIN)
    return np.asarray(coefficients) / scale, scale


def _chebyshev_t(m: int, t: Fraction) -> Fraction:
    previous, current = Fraction(1), t
    for _ in range(m - 1):
        previous, current = current, 2 * t * current - previous
    return current if m else Fraction(1)


def test_minimax_inverse_matches_exact_rational_reference_and_its_relative_residual() -> None:
    """The stored kappa=4 table equals the exact minimax polynomial off its DCT nodes.

    With a = 1/4 and m = 6, P(x) = (1 - T_6(t(x^2))/T_6(17/15))/(4x),
    t(y) = (1 + a^2 - 2y)/(1 - a^2), is evaluated in rational arithmetic at
    x = 3/10, which is not one of the 12 Chebyshev zeros of the construction.
    """
    inverse = _inverse_chebyshev(4.0, 0.1)
    assert (inverse.degree, inverse.ideal_degree) == (11, 11)
    assert all(c == 0 for c in inverse.coefficients[::2])
    a, x = Fraction(1, 4), Fraction(3, 10)
    exact = (1 - _chebyshev_t(6, (1 + a * a - 2 * x * x) / (1 - a * a))
             / _chebyshev_t(6, (1 + a * a) / (1 - a * a))) / (4 * x)
    assert abs(np.polynomial.chebyshev.chebval(0.3, inverse.coefficients) - float(exact)) <= 1e-15
    assert inverse.minimax_error == pytest.approx(float(1 / _chebyshev_t(6, Fraction(17, 15))), rel=1e-14)
    grid = np.linspace(0.25, 1.0, 4001)
    residual = np.max(np.abs(4.0 * grid * np.polynomial.chebyshev.chebval(grid, inverse.coefficients) - 1.0))
    assert inverse.minimax_error <= residual + 1e-15 and residual <= inverse.residual_check_value <= 0.1
    with pytest.raises(ValueError, match="kappa must exceed 1"):
        _inverse_chebyshev(1.0, 0.1)


def test_minimax_inverse_degree_boundary_and_numerical_refinement() -> None:
    """The degree changes exactly where sech(m beta) crosses epsilon_inv; a marginal table is refined.

    For kappa = 4, sech(6 beta) = 2/((5/3)^6 + (3/5)^6). At kappa = 43/32
    the degree-1 minimax residual is 825/2873, and with epsilon_inv 2e-15
    above it the check value of the degree-1 table, which includes
    4 eps log2(12) = 3.6e-15, exceeds the tolerance. The review witness is
    a binary64 epsilon at which a degree-1 coefficient from another
    evaluation exceeds the tolerance by exactly 2**-58 at x = 1, while the
    coefficient computed here meets it. The returned tables of either degree
    are checked against an exact rational upper bound on their residual. At
    kappa = 3 the epsilon just above sech(3 beta) = 16/65 leaves the
    degree-5 table no room. At kappa = 100 and epsilon_inv = 1e-12 the
    formation term of the refined table exceeds that of the first, and the
    refinement reserve covers it. At 1e-14 the formation term alone exceeds
    the tolerance.
    """
    delta = float(Fraction(2) / (Fraction(5, 3) ** 6 + Fraction(3, 5) ** 6))
    above, below = _inverse_chebyshev(4.0, delta * (1 + 1e-9)), _inverse_chebyshev(4.0, delta * (1 - 1e-9))
    assert (above.degree, above.ideal_degree, below.degree, below.ideal_degree) == (11, 11, 13, 13)
    kappa, witness = Fraction(43, 32), float.fromhex("0x1.260c4beaf89b0p-2")
    archived, computed = Fraction(8627849756020611, 2**53), Fraction(_table(43 / 32, 1, None)[0][1])
    assert restricted_inverse_residual_bound([archived], kappa) - Fraction(witness) == Fraction(1, 2**58)
    assert restricted_inverse_residual_bound([computed], kappa) <= Fraction(witness)
    for epsilon in (825 / 2873 + 2e-15, witness):
        marginal = _inverse_chebyshev(43 / 32, epsilon)
        assert marginal.ideal_degree == 1 <= marginal.degree and marginal.degree % 2 == 1
        odd = [Fraction(c) for c in marginal.coefficients[1::2]]
        assert restricted_inverse_residual_bound(odd, kappa) <= Fraction(marginal.residual_check_value)
        assert marginal.residual_check_value <= epsilon
    boundary = _inverse_chebyshev(3.0, 0.24615384615384617)
    assert boundary.degree == 7 and boundary.residual_check_value <= 0.24615384615384617
    refined = _inverse_chebyshev(100.0, 1e-12)
    assert refined.degree > refined.ideal_degree and refined.residual_check_value <= 1e-12
    with pytest.raises(ValueError, match="unresolved"):
        _inverse_chebyshev(100.0, 1e-14)


def test_unresolved_inverse_check_reports_its_compared_value_for_a_table_that_meets_the_tolerance() -> None:
    """At kappa = 1.01 and epsilon_inv = 3e-15 the degree-13 table meets the tolerance, which its check cannot decide.

    The refusal reports sech(7 beta), the formation term and the resolution
    4 eps log2(60) of M = 60 evaluation points, and the check value formed
    from them. An exact rational bound puts the residual of the same table
    below 3e-15. At epsilon_inv = 1e-14 the check accepts this table, which
    records the value that the check compared.
    """
    with pytest.raises(ValueError, match="unresolved") as refused:
        _inverse_chebyshev(1.01, 3e-15)
    reported = re.search(r"degree 13 .* = \((\S+) \+ (\S+)\) \+ \((\S+) \+ (\S+)\) = (\S+) exceeds",
                         str(refused.value))
    delta, formation, repeated, resolution, value = map(float, reported.groups())
    coefficients, *terms = _table(1.01, 7, None)
    assert delta == pytest.approx(1 / math.cosh(14 * math.atanh(1 / 1.01)), rel=1e-12)
    assert formation == repeated and [formation, resolution] == terms
    assert resolution == 4 * np.finfo(float).eps * (60).bit_length()
    assert value == (delta + formation) + (formation + resolution) > 3e-15
    odd = [Fraction(c) for c in coefficients[1::2]]
    assert restricted_inverse_residual_bound(odd, Fraction(1.01)) <= Fraction(3e-15)
    accepted = _inverse_chebyshev(1.01, 1e-14)
    assert accepted.coefficients == tuple(coefficients) and accepted.residual_check_value == value


def test_cks_cross_check_agrees_within_summed_residual_bounds() -> None:
    """[CKS] (arXiv:1511.02306v2) closed form and the minimax table agree on D within their residuals."""

    kappa_be, epsilon_inv = 4.0, 1.0e-2
    inverse = _inverse_chebyshev(kappa_be, epsilon_inv)
    cks = cks_inverse_coefficients(kappa_be, epsilon_inv)
    domain = np.linspace(1.0 / kappa_be, 1.0, 25 * (len(cks) - 1))
    table_values = kappa_be * np.polynomial.chebyshev.chebval(domain, inverse.coefficients)
    cks_values = np.polynomial.chebyshev.chebval(domain, cks)
    cks_residual = float(np.max(np.abs(domain * cks_values - 1.0)))
    assert cks_residual <= epsilon_inv  # the paper form holds its own bound
    disagreement = float(np.max(np.abs(domain * (table_values - cks_values))))
    assert disagreement <= inverse.residual_check_value + cks_residual + 1.0e-12


def test_kernel_reflection_polynomial_eq6_eq62_and_bounds() -> None:
    """The DCT table equals Dalzell Eq. (62) in rational arithmetic at x = 3/10 and keeps K(0) = 1."""
    kr = _realize_kernel_reflection(plan_kernel_reflection(4.0, 1.0e-2))
    ell = int(np.ceil(4.0 * np.log(2.0 / 1.0e-2) / 2.0))
    assert kr.kr_ell == ell and all(c == 0 for c in kr.coefficients[1::2])
    delta, x = Fraction(1, 4), Fraction(3, 10)
    filter_delta = 1 / _chebyshev_t(ell, (1 + delta * delta) / (1 - delta * delta))
    filter_x = _chebyshev_t(ell, (1 + delta * delta - 2 * x * x) / (1 - delta * delta)) * filter_delta
    exact = (2 * filter_x - 1 + filter_delta) / (1 + filter_delta)
    assert abs(np.polynomial.chebyshev.chebval(0.3, kr.coefficients) - float(exact)) <= 1e-14
    grid = np.linspace(0.0, 1.0, 2001)
    values = np.polynomial.chebyshev.chebval(grid, kr.coefficients)
    assert np.polynomial.chebyshev.chebval(0.0, kr.coefficients) == pytest.approx(1.0, abs=1e-12)
    off_kernel = values[grid >= 1.0 / 4.0]
    assert np.max(off_kernel) <= -1.0 + 4.0e-2 / 1.01 + 1.0e-12
    assert np.min(off_kernel) >= -1.0 - 1.0e-12
    normed, _ = _normed_qsp_target(kr.coefficients)
    assert solve_wx_qsp_phases(normed).residual_sup_bound <= 1.0e-12


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
    even_phases = solve_wx_qsp_phases(even_target)
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
    odd_phases = solve_wx_qsp_phases(odd_coefficients)
    odd_block = block_encoding_top_left(
        Operator(build_real_chebyshev_encoding(encoding, odd_phases.phases)).data,
        num_ancillas=encoding.num_ancillas + 2,
    )
    odd_expected = (
        left @ np.diag(np.polynomial.chebyshev.chebval(singular_values, odd_coefficients)) @ right_h
    )
    assert np.max(np.abs(odd_block - odd_expected)) <= 1.0e-12



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



@pytest.mark.parametrize("kappa", [20.])
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


@pytest.mark.parametrize("kappa", [64.])
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
