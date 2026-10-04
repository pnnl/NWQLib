"""Independent Chebyshev, sensitivity and physical-spectrum relations."""

from fractions import Fraction

import numpy as np
import pytest
from nwqlib.algorithms.lanczos import Lanczos, numerical
from nwqlib.algorithms.lanczos.records import MomentStatistics
from nwqlib.core.planning import RandomStreams
from nwqlib.problems.records import Eigenproblem
from nwqlib.operators import ingest_pauli


def selected(*, terms=(("I", 1.0), ("X", -1.0), ("Z", 0.5)), m=2):
    q = len(terms[0][0])
    problem = Eigenproblem(A=ingest_pauli(terms, num_qubits=q))
    method = Lanczos(initial_state=[1] + [0] * ((1 << q) - 1), krylov_dimension=m)
    return method.plan(
        problem,
        output=problem.default_output(),
        execution="quantum",
        shots=None,
        rng=RandomStreams(7),
    )


def kernel(plan, moments):
    statistics = {
        k: MomentStatistics(mean=v, second_moment=1.0, shots=None) for k, v in moments.items()
    }
    return numerical.reconstruct(plan.reconstruction, statistics, cutoff=None, sampled=False)


def test_signed_centered_pencil_from_independent_moments():
    result = kernel(selected(), {1: 1 / 3, 2: 1 / 9, 3: -7 / 27})
    np.testing.assert_allclose(result["overlap"], [[1, 1 / 3], [1 / 3, 5 / 9]], rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        result["hamiltonian"], [[1.5, 7 / 6], [7 / 6, 5 / 6]], rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(
        result["eigenvalues"], [1 - np.sqrt(5) / 2, 1 + np.sqrt(5) / 2], rtol=0, atol=2e-14
    )


def test_analytical_readout_moments_reconstruct_sector_pencil():
    from nwqlib.algorithms.lanczos.readout import decode_histogram
    from nwqlib.problems.inputs import ingest_occupation

    # H = X0 + Z0 + Z1 + Z2 + Z3 on |1110> stays in basis indices
    # (14, 15), where H = [[-2, 1], [1, -4]]. Direct pair recurrence
    # gives raw moments (1, -2, 5, -16), hence the Chebyshev moments
    # (1, -2/5, -3/5, 86/125) for H/5.
    labels = ("IIIX", "IIIZ", "IIZI", "IZII", "ZIII")
    problem = Eigenproblem(
        A=ingest_pauli(((label, 1) for label in labels), num_qubits=4)
    )
    state = ingest_occupation((0, 1, 1, 1), num_qubits=4)
    plan = Lanczos(initial_state=state, krylov_dimension=2).plan(
        problem,
        output=problem.default_output(),
        execution="quantum",
        shots=None,
        rng=RandomStreams(7),
    )
    rec = plan.reconstruction
    assert (rec.center, rec.alpha, rec.num_system_qubits, rec.num_index_qubits) == (0, 5, 4, 3)
    assert [s.histogram_width for s in rec.settings] == [7, 3, 7]
    # Integer ratios specify exact analytical readout marginals. The odd
    # settings use label-controlled Pauli parity, the even setting reflection.
    weights = (
        {112: 1, 120: 1, 113: 2, 114: 2, 115: 2, 116: 2},
        {0: 1, 1: 4},
        {112: 49, 120: 1, 113: 162, 121: 8, 114: 2, 122: 8, 115: 2, 123: 8, 116: 2, 124: 8},
    )
    statistics = {}
    expected_moments = (-2 / 5, -3 / 5, 86 / 125)
    for degree, (setting, histogram, expected) in enumerate(
        zip(rec.settings, weights, expected_moments, strict=True), start=1
    ):
        marginal = {bits: weight / sum(histogram.values()) for bits, weight in histogram.items()}
        stats = decode_histogram(plan._native["readout"], setting, marginal, counts=False)
        assert setting.degree == degree
        assert stats.mean == pytest.approx(expected, rel=0, abs=2e-15)
        assert stats.shots is None
        statistics[setting.degree] = stats
    result = numerical.reconstruct(rec, statistics, cutoff=None, sampled=False)
    # Direct sector algebra gives det(Hp-E*S)=(E²+6E+7)/25.
    np.testing.assert_allclose(
        result["overlap"], [[1, -2 / 5], [-2 / 5, 1 / 5]], rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(result["hamiltonian"], [[-2, 1], [1, -16 / 25]], rtol=0, atol=2e-15)
    assert result["eigenvalue"] == pytest.approx(-3 - np.sqrt(2), rel=0, abs=2e-13)


def test_missing_moment_is_partial_and_identity_needs_no_solve(monkeypatch):
    """A missing moment makes the result partial; analysis never acquires it again.

    The exact trajectory reads mu_1, mu_2 and mu_3 as three point chunks of
    one acquisition. Without the point chunk of mu_2, exactly degree 2 is
    unacquired, every other moment equals the full analysis bitwise, and no
    backend preparation or simulation runs. The chunks of mu_1 and mu_3 carry
    the same one-point readout. Swapping their point and chunk key alone
    keeps the probability arrays' old acquisition and is refused as a
    provenance mismatch. Swapping the point, the chunk key and the arrays'
    manifests while each chunk keeps its own boundary is refused by the
    receipt's point boundary. These checks establish that a chunk's labels
    agree with its receipt, not that its payload belongs to its point: no
    per-chunk payload digest is recorded, so a relabel that also moves the
    boundary is not among the refused cases.
    """
    import dataclasses
    import nwqlib.algorithms.lanczos.numerical as owner
    from nwqlib._prepared_execution import Run
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.execution import ObservationView, point_chunk_key

    plan = selected()
    run = Run(plan)
    full = run.wait(timeout=5, poll_interval=0)
    data = run.data
    for name in ("_prepare_aer_execution", "_submit_aer_execution"):
        monkeypatch.setattr(aer, name, lambda *a, _name=name, **k: pytest.fail(f"{_name} during analysis"))
    monkeypatch.setattr(
        owner, "_solve_projected_pencil", lambda *a, **k: pytest.fail("unexpected solve")
    )
    kept = tuple(chunk for chunk in data.observations.chunks if chunk.point != "moment_2")
    assert len(kept) == len(data.observations.chunks) - 1 == 2
    missing = plan.method.analyze(
        plan, dataclasses.replace(data, observations=ObservationView(chunks=kept)), settings={}
    )
    assert missing.eigenvalue is None and missing.missing == (2,) and missing.moments[2] is None
    assert [missing.moments[k] for k in (0, 1, 3)] == [full.moments[k] for k in (0, 1, 3)]
    swap = {"moment_1": "moment_3", "moment_3": "moment_1"}

    def relabelled(chunk):
        key = point_chunk_key(chunk.chunk.rsplit("/", 1)[0], swap[chunk.point])
        with pytest.raises(ValueError, match="provenance differs"):
            chunk.revise(point=swap[chunk.point], chunk=key)
        arrays, acquisition = chunk.values[0], (*chunk.acquisition_key[:-1], key)
        manifests = {name: getattr(arrays, name).revise(acquisition=acquisition)
                     for name in ("probabilities", "indices") if getattr(arrays, name) is not None}
        return chunk.revise(point=swap[chunk.point], chunk=key, values=(arrays.revise(**manifests),))

    swapped = tuple(relabelled(chunk) if chunk.point in swap else chunk for chunk in data.observations.chunks)
    with pytest.raises(ValueError, match="receipt's point boundary"):
        plan.method.analyze(
            plan, dataclasses.replace(data, observations=ObservationView(chunks=swapped)), settings={}
        )
    partial = kernel(selected(), {1: 1 / 3})
    assert partial["eigenvalue"] is None and partial["missing"] == (2, 3)
    scalar = kernel(selected(terms=(("I", 3.0),)), {})
    assert scalar["eigenvalue"] == 3 and not scalar["missing"]


def test_regularized_sensitivity_rotates_the_kept_projector():
    from nwqlib._projected_eigensolver import _solve_projected_pencil

    mu = np.array([1.0, 0.0, -0.5, 0.0])
    h, s = numerical._projected_matrices(mu, 2)
    result, _, failure, _ = _solve_projected_pencil(
        h, s, overlap_eigenvalue_cutoff=0.5, _sampled_overlap=True, _keep_overlap_eigenvectors=True
    )
    assert failure is None and result.kept_overlap_rank == 1
    derivative = numerical._moment_energy_derivatives(result, h, s, 0.0, 1.0, 0.5)
    # S=[[1,t],[t,1/4]] rotates the kept vector by (0,4/3).
    # K=[[t,1/4],[1/4,t/4]] gives 1 + 2*(1/4)*(4/3)=5/3.
    assert derivative @ [0.0, 1.0, 0.0, -2.0] == pytest.approx(5 / 3, rel=0, abs=2e-14)


def test_shared_moment_covariance_and_rank_policy():
    # S00=mu0; S01=S10=mu1; S11=(mu0+mu2)/2. Correlated
    # repeated entries give E||delta S||F^2=2*v1+v2/4, not v1+v2/4.
    cutoff, record = numerical._resolve_overlap_cutoff(
        None, 2, sampled=True, variances=np.array([0.0, 0.04, 0.16, 0.01]),
        moment_shots=[0, 100, 200, 1], policy="confidence"
    )
    assert record["empirical_gram_noise_frobenius_rms"] == pytest.approx(np.sqrt(0.12), rel=0, abs=2e-16)
    assert record["source"] == "hoeffding_gram_bound"
    # Two sampled moments enter S. With m=2 and n_min=100, the
    # two-sided union bound at cutoff/(2m) spends exactly delta=.05.
    assert 4*np.exp(-100*(cutoff/4)**2/2) == pytest.approx(.05, rel=2e-14, abs=0)
    assert numerical._allocate(101, np.array([1e308, 1e308])).sum() == 101
    # Scalar cancellation after overflowing intermediate multiplication.
    values = numerical.restore_ritz_values(np.array([2.0]), -1.7e308, 1e308)
    assert values[0] == pytest.approx(3e307, rel=2e-15, abs=0)


def test_sampling_cutoff_does_not_treat_zero_sample_variance_as_certainty():
    kwargs = dict(sampled=True, variances=np.zeros(4), moment_shots=[0, 100, 400, 1],
                  policy="confidence")
    cutoff, _ = numerical._resolve_overlap_cutoff(None, 2, **kwargs)
    assert cutoff > .1
    refined, _ = numerical._resolve_overlap_cutoff(None, 2,
        **dict(kwargs, moment_shots=[0, 400, 1600, 1]))
    assert refined == pytest.approx(cutoff/2, rel=2e-15, abs=0)
    fixed, evidence = numerical._resolve_overlap_cutoff(1e-5, 2, **kwargs)
    assert fixed == 1e-5 and evidence["source"] == "user_fixed"
    # mu_1 affects H only for m=1. Its shot count must not regularize S00=1.
    single, _ = numerical._resolve_overlap_cutoff(None, 1, sampled=True, moment_shots=[0, 1],
                                                 policy="confidence")
    assert single == numerical.DEFAULT_OVERLAP_EIGENVALUE_CUTOFF


def test_exploratory_cutoff_keeps_noise_scale_separate_from_confidence():
    variance = np.array([0., .04, .16, .01])
    actual, evidence = numerical._resolve_overlap_cutoff(None, 2, sampled=True,
        variances=variance, moment_shots=[0, 100, 200, 1], noise_multiplier=.5)
    assert actual == pytest.approx(.5*np.sqrt(2*.04+.16/4), rel=2e-15)
    assert evidence["source"] == "empirical_gram_rms"
    assert evidence["gram_sampling_bound"] == pytest.approx(2*np.sqrt(2*np.log(4/.05)/100))
    zero, evidence = numerical._resolve_overlap_cutoff(None, 2, sampled=True,
        variances=np.zeros(4), moment_shots=[0, 100, 200, 1])
    assert zero == numerical.DEFAULT_OVERLAP_EIGENVALUE_CUTOFF
    assert evidence["gram_sampling_bound"] > .1


@pytest.mark.parametrize(
    "weights,check_spectrum",
    [([0.1**2, 0.2**2, 0.3**2, 0.0001**2], False), ([1.0, 1.0, 1.0, 1.0], True)],
)
def test_large_identity_shift_keeps_the_normalized_pencil(weights, check_spectrum):
    terms = (("ZI", 0.4), ("IZ", 0.6), ("ZZ", 0.1))
    base = selected(terms=terms, m=4)
    shifted = selected(terms=(*terms, ("II", 1e8)), m=4)
    # These four values follow directly from Pauli diagonal signs. The small
    # ground-overlap weights stress covariance; the balanced case owns accuracy.
    energies = np.array([1.1, -0.3, 0.1, -0.9])
    weights = np.array(weights) / sum(weights)
    moments = {
        degree: float(weights @ np.cos(degree * np.arccos(energies / 1.1)))
        for degree in range(1, 8)
    }
    result = kernel(base, moments)
    translated = kernel(shifted, moments)
    assert result["coefficients"] == translated["coefficients"]
    assert result["kept_rank"] == translated["kept_rank"] == 4
    if check_spectrum:
        np.testing.assert_allclose(
            result["eigenvalues"], [-0.9, -0.3, 0.1, 1.1], rtol=0, atol=1e-12
        )
    assert translated["eigenvalue"] - 1e8 == pytest.approx(
        result["eigenvalue"], rel=0, abs=2 * np.spacing(1e8)
    )


def test_actual_pilot_floor_and_legal_nonbinding_allocation():
    allocation = np.array([2, 18, 10])
    returned = np.array([6, 6, 6])
    result, floor, count = numerical._apply_pilot_floor(allocation, returned)
    assert count == 1
    np.testing.assert_array_equal(floor, [6, 6, 6])
    # Every setting receives its floor; the remaining 30-18=12 shots follow the
    # donors' surplus above the floor, (18-6):(10-6)=3:1, giving 9 and 3.
    np.testing.assert_array_equal(result, [6, 15, 9])
    # A legal allocation is preserved; the floor must not force uniform shots.
    allowed = np.array([6, 14, 10])
    result, _, count = numerical._apply_pilot_floor(allowed, returned)
    np.testing.assert_array_equal(result, allowed)
    assert count == 0


def test_sensitivity_weights_match_independent_quadratic_root():
    selection = selected(terms=(("X", 1.0), ("Y", 1.0), ("Z", 1.0)), m=2)
    counts = ((45, 35, 20), (21, 39, 0), (32, 16, 32))
    statistics = tuple(
        MomentStatistics(
            mean=(p - n) / (p + n + z), second_moment=(p + n) / (p + n + z), shots=p + n + z
        )
        for p, n, z in counts
    )
    weights, evidence = numerical._sensitivity_weights(
        # This oracle differentiates the full-rank quadratic root. Select that
        # exploratory rank explicitly, independently of automatic sampling admission.
        selection.reconstruction, statistics, 7.0, 2.0, selection.method.revise(overlap_cutoff=1e-4)
    )
    assert evidence["fallback_reason"] is None
    # det(K-kappa*S)=a*kappa^2+b*kappa+c. Implicit differentiation
    # is independent of the production coefficient/projector derivative.
    u, v, w = 0.1, -0.3, 0.2
    a = (1 + v) / 2 - u * u
    b = u * (1 + v) / 2 - (3 * u + w) / 4
    c = u * (3 * u + w) / 4 - ((1 + v) / 2) ** 2
    kappa = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    da = np.array([-2 * u, 0.5, 0.0])
    db = np.array([(1 + v) / 2 - 0.75, u / 2, -0.25])
    dc = np.array([(6 * u + w) / 4, -(1 + v) / 2, u / 4])
    derivatives = -2 * (kappa * kappa * da + kappa * db + dc) / (2 * a * kappa + b)
    variances = np.array(
        [
            (p * (1 - mu) ** 2 + n * (-1 - mu) ** 2 + z * mu**2) / (p + n + z - 1)
            for (p, n, z), mu in zip(counts, (u, v, w), strict=True)
        ]
    )
    np.testing.assert_allclose(weights, abs(derivatives) * np.sqrt(variances), rtol=0, atol=1e-13)


@pytest.mark.parametrize("representation", ["dense", "csr", "csc"])
def test_classical_moments_do_not_depend_on_an_identity_offset(representation):
    from scipy import sparse
    from nwqlib.operators.inputs import ingest_dense, ingest_sparse

    # A0 has entries on the 2**-20 grid, so A0 + cI is exact for c on that
    # grid with |c| <= 2**30, and so are A0_ii + c - (center + c) =
    # A0_ii - center and the division by the power of two alpha. With the
    # offset removed from the stored diagonal before scaling,
    # K = (A - center*I)/alpha is then bitwise the same matrix for every c,
    # and so are the moments. Subtracting (center/alpha)*v after the product
    # would leave roundoff of size u*|c|/alpha in each action.
    rng = np.random.default_rng(11)
    d = 16
    a0 = rng.normal(size=(d, d))
    a0 = np.round((a0 + a0.T) / 2 / np.linalg.norm(a0, 2) * 2**20) / 2**20
    psi = rng.normal(size=d)
    psi /= np.linalg.norm(psi)
    center, alpha = .25, 4.

    def moments(c):
        a = a0 + c * np.eye(d)
        operator = (ingest_dense(a) if representation == "dense"
                    else ingest_sparse(getattr(sparse, f"{representation}_matrix")(a)))
        return numerical.chebyshev_moments(operator, psi, m=8, center=center + c, alpha=alpha)

    np.testing.assert_array_equal(moments(1e6), moments(0.))


def test_projected_backward_error_is_stated_on_the_normalized_pencil():
    """Lanczos reports the backward error of its normalized pencil (K, S), not of (H, S)."""
    from nwqlib import solve

    rng = np.random.default_rng(11)
    M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = 3.0 * np.eye(4) + 0.25 * (M + M.conj().T)
    psi0 = tuple(rng.normal(size=4) + 1j * rng.normal(size=4))

    def backward(h, s, c, e):
        residual = np.linalg.norm(h @ c - e * s @ c)
        return residual / ((np.linalg.norm(h) + abs(e) * np.linalg.norm(s)) * np.linalg.norm(c))

    result = solve(Eigenproblem(A=H), method=Lanczos(initial_state=psi0, krylov_dimension=3), shots=400, seed=5)
    rec = result.plan.reconstruction
    S, physical = np.array(result.overlap), np.array(result.hamiltonian)
    c = np.array(result.coefficients)
    K = (physical - rec.center * S) / rec.alpha
    x = (result.eigenvalue - rec.center) / rec.alpha
    assert result.projected_backward_error == pytest.approx(backward(K, S, c, x), rel=1e-6)
    # The physical-pencil value differs, so the two frames are distinguishable here.
    assert abs(backward(physical, S, c, result.eigenvalue) - result.projected_backward_error) > 1e-3


def rational_sqrt_interval(x, precision=160):
    """Rational lower and upper bounds of sqrt(x) from an integer square root."""
    from math import isqrt

    x = Fraction(x)
    if x < 0:
        raise ValueError("negative square")
    if not x:
        return Fraction(0), Fraction(0)
    e = (x.numerator.bit_length()-x.denominator.bit_length())//2+1
    scale = Fraction(2)**(precision-e)
    y = x*scale*scale
    root = isqrt(y.numerator//y.denominator)
    exact = root*root*y.denominator == y.numerator
    return Fraction(root)/scale, Fraction(root+int(not exact))/scale


def check_gershgorin_union(A, stored_lo, stored_hi):
    """Compare stored endpoints with the exact Gershgorin union of the stored Hermitian matrix A.

    The target is l_* = min_i(a_ii - r_i), h_* = max_i(a_ii + r_i) with
    r_i = sum_(j != i) |a_ij|. Rational square-root intervals [l_-, l_+] and
    [h_-, h_+] enclose l_* and h_*, and E_ref, the largest distance from the
    floating reference endpoints to those intervals, bounds the reference
    rounding. With n = d-1, s the largest absolute real or imaginary input
    component and S >= max_i(|a_ii| + r_i), the small-fixture production
    widening is E_prod = gamma_(32n+128) S + 32 s tiny + 16 eta for
    (32n+128)u <= 1/4, under round-to-nearest, gradual underflow, finite
    values and a one-ulp hypot. The signed discrepancies then satisfy
    -E_ref <= l_hat - l_stored <= E_prod + E_ref and
    -E_ref <= h_stored - h_hat <= E_prod + E_ref, evaluated as rationals,
    and stored_lo <= l_- and stored_hi >= h_+ establish enclosure of the
    exact union on the fixture. The owner is _eigen_inputs.gershgorin_frame.
    """
    from math import fsum, hypot

    Q = Fraction
    d = len(A)
    diag = [Q(float(A[i,i].real)) for i in range(d)]
    rlo, rhi, rfloat = [], [], []
    scale = max(max(abs(Q(float(z.real))), abs(Q(float(z.imag)))) for z in A.flat)
    for i in range(d):
        lower = upper = Q(0)
        magnitudes = []
        for j in range(d):
            if i == j:
                continue
            re, im = Q(float(A[i,j].real)), Q(float(A[i,j].imag))
            lo, hi = rational_sqrt_interval(re*re+im*im)
            lower += lo
            upper += hi
            magnitudes.append(hypot(float(re), float(im)))
        rlo.append(lower)
        rhi.append(upper)
        rfloat.append(fsum(magnitudes))
    lminus = min(c-r for c,r in zip(diag,rhi))
    lplus = min(c-r for c,r in zip(diag,rlo))
    hminus = max(c+r for c,r in zip(diag,rlo))
    hplus = max(c+r for c,r in zip(diag,rhi))
    lf = Q(min(float(c)-r for c,r in zip(diag,rfloat)))
    hf = Q(max(float(c)+r for c,r in zip(diag,rfloat)))
    Eref = max(abs(lf-lminus),abs(lf-lplus),abs(hf-hminus),abs(hf-hplus))
    n, u = d-1, Q(1,2**53)
    k = 32*n+128
    if k*u > Q(1,4):
        raise ValueError("outside this small-fixture comparison law")
    S = max(abs(c)+r for c,r in zip(diag,rhi))
    Eprod = (k*u/(1-k*u))*S+32*scale*Q(1,2**1022)+16*Q(1,2**1074)
    gl, gh = lf-Q(stored_lo), Q(stored_hi)-hf
    assert -Eref <= gl <= Eprod+Eref
    assert -Eref <= gh <= Eprod+Eref
    assert Q(stored_lo) <= lminus and Q(stored_hi) >= hplus
    return gl, gh, Eprod+Eref


@pytest.mark.parametrize("representation", ["dense", "csr", "csc"])
def test_classical_gershgorin_frame_encloses_the_spectrum_with_empty_rows(representation):
    """The recorded frame of _eigen_inputs.gershgorin_frame encloses the Gershgorin union and the spectrum.

    Rows 0, 3 and 6 are empty (leading, middle and trailing), so the
    compressed reduction must give them zero radius without reading past the
    stored values. The eigenvalues come from an independent dense eigvalsh.
    The stored interval is compared with an independently computed
    Gershgorin union of the stored matrix. Rational square-root intervals
    bound rounding in the floating reference. The signed comparison allows
    that reference error and the derived production widening, and a
    separate interval check establishes enclosure on the fixture. The
    reference includes empty rows and excludes every diagonal entry from its
    radius.

    The seven-dimensional fixture's outer intervals are wide enough that
    the radius a start-pointer reduction error assigns to an empty leading
    or middle row changes neither returned endpoint. In the five-dimensional fixture with only
    a_11 = a_33 = 2 and a_13 = a_31 = 1 (zero-based), the true intervals
    are {0}, [1,3], {0}, [1,3], {0}, so the exact hull is [0,3]; a nonzero
    radius on the empty middle row moves the lower endpoint below 0. With
    d = 5, n = d-1 = 4, k = 32n+128 = 256, input scale 2 and S = 3, the
    small-fixture allowance of the comparison above is
    gamma_256*3 + 64 tiny + 16 eta.
    """
    from scipy import sparse

    rng = np.random.default_rng(3)
    d = 7
    block = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    matrix = block + block.conj().T
    matrix[[0, 3, 6], :] = 0
    matrix[:, [0, 3, 6]] = 0
    # The comparison law needs exact Hermiticity, a real diagonal and finite entries.
    assert np.array_equal(matrix, matrix.conj().T) and not np.diag(matrix).imag.any()
    assert np.isfinite(matrix).all()
    A = matrix if representation == "dense" else getattr(sparse, f"{representation}_matrix")(matrix)
    problem = Eigenproblem(A=A)
    plan = Lanczos(krylov_dimension=2).plan(
        problem, output=problem.default_output(), execution="classical", shots=None,
        rng=RandomStreams(5),
    )
    rec = plan.reconstruction
    spectrum = np.linalg.eigvalsh(matrix)
    assert rec.enclosure_source == "scaled_gershgorin"
    assert rec.spectral_lower <= spectrum[0] and spectrum[-1] <= rec.spectral_upper
    check_gershgorin_union(matrix, rec.spectral_lower, rec.spectral_upper)
    if representation == "dense":
        return
    Q = Fraction
    matrix = np.zeros((5, 5))
    matrix[1, 1] = matrix[3, 3] = 2.0
    matrix[1, 3] = matrix[3, 1] = 1.0
    problem = Eigenproblem(
        A=getattr(sparse, f"{representation}_matrix")(matrix)
    )
    plan = Lanczos(krylov_dimension=2).plan(
        problem,
        output=problem.default_output(),
        execution="classical",
        shots=None,
        rng=RandomStreams(5),
    )
    rec = plan.reconstruction
    # Exact Gershgorin hull [0,3]. n=4, scale=2, S=3.
    u = Q(1, 2**53)
    ku = 256 * u
    error = 3 * ku / (1 - ku) + 64 * Q(1, 2**1022) + 16 * Q(1, 2**1074)
    assert -error <= Q(rec.spectral_lower) <= 0
    assert 3 <= Q(rec.spectral_upper) <= 3 + error
