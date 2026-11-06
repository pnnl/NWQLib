"""Size formulas (laws) of the dense linear-algebra routines that several methods call.

The formulas cover SciPy's matrix exponential and its action on a vector, the
singular values that numpy.linalg.norm(ord=2) and
numpy.linalg.svd(compute_uv=False) compute without singular vectors, the
eigenvalues and eigenvectors of numpy.linalg.eigh and the least-squares
solution of numpy.linalg.lstsq. A caller checks the counts from these
formulas against its work and byte limits before it calls the routine. Work
uses the units of the library's dense formulas, n**3 for a product of two n-square matrices, n**2 for
a dense matrix-vector product and one unit per stored entry that a sparse
product with a vector visits. A multiply-add, a division or an elementwise
operation such as an absolute value or a comparison counts as one unit.
Bytes count the arrays of known size that the routine holds at once and
exclude the rest of the process. The exponential formulas follow the
algorithms of SciPy 1.18.1, and the other formulas follow the LAPACK drivers
that NumPy 2.5.2 calls (gesdd, heevd or syevd, and gelsd). Revisit a formula
when its routine changes. The module also provides seeded_norm_estimates,
which fixes the random draws of SciPy's norm estimates.
"""

from contextlib import contextmanager
from math import ceil, isfinite, log2

# Largest 1-norm of a matrix that expm_requirements admits. To pick the Pade
# degree and the squarings, both SciPy kernels compute norms of M**k for k up
# to 10 and of |M|**k for k up to 27, each at most ||M||_1**k, so this limit
# keeps every one of them below 2**999. Above it a norm can overflow. SciPy
# 1.18.1's scipy.linalg.expm then derives its squaring count from an
# infinite value and did not return within 20 s for a 3 x 3 input of norm
# 2**39, and scipy.sparse.linalg.expm raises (docs/dependency_issues.md).
# Registered in ENGINEERING_CONSTANTS.
MAX_EXPONENTIAL_NORM = 2.0**37

# Condition (3.13) of Al-Mohy and Higham, SIAM J. Sci. Comput. 33, 488
# (2011), doi:10.1137/100788860, in
# scipy.sparse.linalg._expm_multiply._condition_3_13 for one
# vector, m_max = 55 and ell = 2. At or below this 1-norm expm_multiply picks
# its parameters from the 1-norm alone. Above it, it estimates the 1-norms of
# the powers 2 to 9 with onenormest. The product is evaluated in SciPy's
# order, a = 2 ell p_max (p_max + 3) = 352 times b = theta_55/55, which gives
# 63.36. Multiplying 352 by 9.9 first rounds one ulp higher, and at a norm
# equal to that higher value SciPy estimates the norms while the law would
# charge no estimation.
_EXPM_MULTIPLY_NORM_ONLY = 2 * 2 * 8 * (8 + 3) * (9.9 / 55)


@contextmanager
def seeded_norm_estimates():
    """Run a SciPy call with NumPy's global generator seeded, then restore the caller's state.

    SciPy 1.18.1's onenormest (scipy/sparse/linalg/_onenormest.py) draws the
    +-1 starting columns and their resamples from NumPy's legacy global
    generator (np.random.randint). scipy.sparse.linalg.expm calls it from
    order 200, and expm_multiply for a shifted 1-norm above 63.36, condition
    (3.13) of Al-Mohy and Higham (2011), doi:10.1137/100788860. Without
    this seeding such a call advances the caller's random stream, and the
    estimated norms, and with them the parameters SciPy selects and the
    rounding of its result, depend on the caller's state. During the call
    the generator starts from seed 0, so the same input gives the same draws
    and the same result, and on exit the generator returns to the caller's
    state. The generator is shared by the whole process, so another thread
    that draws from it during the call interleaves with these draws.
    """
    import numpy as np

    state = np.random.get_state()
    np.random.seed(0)
    try:
        yield
    finally:
        np.random.set_state(state)


def expm_requirements(dimension, norm):
    """Return (work, bytes) of one dense matrix exponential of an n-square M with ||M||_1 <= norm.

    Both scipy.linalg.expm and scipy.sparse.linalg.expm follow Algorithm 5.1
    of Al-Mohy and Higham, SIAM J. Matrix Anal. Appl. 31, 970 (2009),
    doi:10.1137/09074721X. They form M**2, M**4, M**6, M**8 and M**10 to
    choose the Pade degree (for large n they estimate the norms of the last
    two instead), then three products and one LU solve with n right-hand
    sides for the degree-13 quotient, then s squarings. The squaring count
    is ceil(log2(eta/4.25)), with eta at most ||M||_1, plus a backward-error
    term that adds squarings only while 2**-s ||M||_1 exceeds 2**2.44, so
    s <= max(0, ceil(log2(||M||_1/4.25))).
    The formula counts ``ceil(log2(norm))`` squarings when ``norm > 1`` and
    zero otherwise, using the supplied upper bound ``norm`` on ``||M||_1``.
    For ``norm > 4.25``, this count is two or three larger than
    ``ceil(log2(norm/4.25))`` in exact arithmetic.
    With c the counted squarings above, the formula counts (10 + c) n**3
    units for eight products, the solve at 4/3 n**3 units and c squarings.
    It adds 128 n**2 units for the 79 products of |M| with a vector in the
    backward-error tests of orders 3 to 13, their absolute values and norms,
    and 34 scaled sums of the Pade evaluation.
    The routines hold at most 20 complex n-square arrays at once (M, the
    identity, the five powers, four scaled powers, two partial products, U,
    V, their sum and difference, the LU factors, the solution and a squared
    copy), 320 n**2 bytes, and 65536 bytes cover small objects.
    Traced peaks of scipy.sparse.linalg.expm stayed at or below 17.2 complex
    n-square arrays for n from 32 to 256. scipy.linalg.expm allocates
    6 n**2 + 4 n complex entries of workspace and its n-square output.

    Args:
        dimension: n, the order of M.
        norm: An upper bound on ||M||_1.

    Returns:
        The work in units and the bytes of the arrays held at once.

    Raises:
        ValueError: norm is not finite or exceeds MAX_EXPONENTIAL_NORM.
    """
    if not (isfinite(norm) and norm <= MAX_EXPONENTIAL_NORM):
        raise ValueError(f"matrix exponential norm {norm!r} exceeds the kernel limit {MAX_EXPONENTIAL_NORM!r}")
    squarings = ceil(log2(norm)) if norm > 1 else 0
    return (10 + squarings)*dimension**3 + 128*dimension**2, 320*dimension**2 + 65536


def expm_multiply_requirements(dimension, entries, norm):
    """Return (work, bytes) of one scipy.sparse.linalg.expm_multiply(G, v) for a vector v.

    G is an n-square sparse matrix with at most ``entries`` stored entries,
    and norm bounds ||G - (trace(G)/n) I||_1, the shifted 1-norm that SciPy
    uses. SciPy 1.18.1 implements Algorithm 3.2 of Al-Mohy and Higham
    (2011), doi:10.1137/100788860, in _expm_multiply_simple. It shifts G
    by its mean diagonal and takes the exact 1-norm, 5 (entries + n) units.
    It then applies s times m terms of the Taylor series, one product of G
    with a vector each. The parameter search of _fragment_3_1 includes the
    pair m = 55, s = ceil(alpha/9.9) with alpha at most the norm, so it
    applies at most 55 ceil(norm/9.9) products and none for a zero norm. Above the 1-norm of
    condition (3.13), 63.36 for one vector, it also estimates the 1-norms of
    the powers 2 to 9 with onenormest. Each estimate of the p-th power makes
    at most six forward and five transposed products of a two-column block,
    22 p products with a vector, 968 over p = 2 to 9. Each product with a
    vector costs ``entries`` units plus 7 n for the scaling, norms and sums
    around it. The bytes allow 64 per stored entry of the shifted copy and
    of the identity, and 32 complex n-vectors for the Taylor terms and the
    onenormest blocks.

    Args:
        dimension: n, the order of G.
        entries: An upper bound on the stored entries of G.
        norm: An upper bound on ||G - (trace(G)/n) I||_1.

    Returns:
        The work in units and the bytes of the arrays held at once.
    """
    if not (isfinite(norm) and norm >= 0):
        raise ValueError(f"expm_multiply generator norm {norm!r} must be finite and nonnegative")
    products = 55*ceil(norm/9.9) if norm > 0 else 0
    if norm > _EXPM_MULTIPLY_NORM_ONLY:
        products += 968
    return products*(entries + 7*dimension) + 5*(entries + dimension), 64*(entries + dimension) + 512*dimension


def singular_values_work(dimension):
    """Return the work units of the singular values of a d-square matrix without its singular vectors.

    numpy.linalg.norm(ord=2) and numpy.linalg.svd(compute_uv=False) run
    LAPACK's gesdd with JOBZ='N'. For a square matrix it first reduces the
    matrix to bidiagonal form (zgebrd, or dgebrd for real input). LAPACK's
    operation count for that step is 16 d**2 (d - d/3) real flops for
    complex input and 4 d**2 (d - d/3) for real input, ceil(4 d**3/3)
    multiply-adds in either case. Forming its 2d Householder vectors adds
    about 2 d**2 units. gesdd scans the input once for its scaling and NumPy
    copies it, d**2 units each. The singular values of the bidiagonal matrix
    then come from the dqds algorithm (dbdsdc, dlasdq and dlasq1 to dlasq6),
    whose transforms make, per element, one division, one addition, one
    multiply-add, one multiplication and two comparisons, six units.
    dlasq2 reports its division count, which stayed at or below 4.12 d**2
    for random complex and real matrices and for matrices with clustered or
    graded singular values of order 4 to 128 (macOS Accelerate LAPACK,
    2026-09-25), so the dqds stage took at most about 25 d**2 units. The
    formula counts 32 d (d + 1) units for these steps together, which also covers
    their terms linear in d. The library's dense formulas count 8 d**3 for a
    full SVD, which also forms both singular frames. This route forms
    neither.

    Args:
        dimension: d, the order of the matrix.

    Returns:
        The work in units.
    """
    return (4 * dimension**3 + 2) // 3 + 32 * dimension * (dimension + 1)


def hermitian_eigensystem_work(dimension):
    """Return the work units of the eigenvalues and eigenvectors of a d-square Hermitian matrix.

    numpy.linalg.eigh runs LAPACK's zheevd, or dsyevd for real input, with
    JOBZ='V'. The driver reduces the matrix to real tridiagonal form, about
    (2/3) d**3 multiply-adds, computes the eigenvectors of the tridiagonal
    matrix and applies the reduction's Householder reflectors to them
    (zunmtr or dormtr), d**2 (d - 1) multiply-adds. Above order 25 the
    tridiagonal eigenvectors come from divide and conquer, whose merge
    products take at most (2/3) d**3 multiply-adds when no eigenvalue
    deflates. At order 25 and below they come from implicit QL or QR
    iteration (zsteqr or dsteqr). Each rotation updates two columns of the
    eigenvector matrix at four units per row, so two iterations per
    eigenvalue over the shrinking active block take about 4 d**3 units. The
    cubic work is therefore about 6 d**3 units or less on either route. The
    formula counts 8 d**3, the count of a full SVD in the library's dense
    formulas, and 32 d**2 for the terms of lower order: the input copy and
    scans, the Householder vectors, and the rotation parameters or the
    secular equations of the tridiagonal step.

    Args:
        dimension: d, the order of the matrix.

    Returns:
        The work in units.
    """
    return 8 * dimension**3 + 32 * dimension**2


def complex_schur_work(dimension):
    """Return the estimated work units of the complex Schur form and Schur vectors of a d-square matrix.

    scipy.linalg.schur(output="complex") runs LAPACK's zgees with
    JOBVS='V'. It reduces the matrix to upper Hessenberg form (zgehrd),
    about (5/3) d**3 multiply-adds, forms the unitary factor of that
    reduction (zunghr), about (2/3) d**3, then updates the Schur form T
    and vectors Z by QR iteration (zhseqr). LAPACK can use multishift QR
    for larger blocks. The estimate models this stage in single-shift
    sweeps that update T across its full width and update Z.
    A sweep on an active block of order k applies
    k - 1 reflectors of order two. Each updates about d entry pairs of T and
    d of Z, at about four units per pair, so a sweep costs about 8 d k
    units. The number of sweeps depends on how fast eigenvalues deflate
    (LAPACK Working Note 121, https://www.netlib.org/lapack/lawnspdf/lawn121.pdf),
    and two sweeps per eigenvalue, the assumption of
    ``hermitian_eigensystem_work``, give about 8 d**3 units, so about
    10.3 d**3 in all. Published rough guides are larger. Golub and Van Loan
    (Matrix Computations, 4th ed., Johns Hopkins University Press, 2013,
    ISBN 978-1-4214-0794-4, Sec. 7.5.6, p. 391) count about 25 d**3 real
    flops for the real Schur form with its orthogonal factor, about two
    Francis steps per deflation, and Eigen's ComplexSchur documentation
    (https://libeigen.gitlab.io/eigen/docs-master/classEigen_1_1ComplexSchur.html)
    gives about 25 d**3 complex flops with the unitary factor. These are
    about 12.5 d**3 multiply-adds in each algorithm's scalar arithmetic.
    The formula counts 13 d**3, which covers the nominal estimates
    and about 2.7 sweeps per eigenvalue in the count above, and 32 d**2 for
    the permutation balancing, copies, shifts and reflector generation. It
    is an estimate, not a bound on the iterations.

    Args:
        dimension: d, the order of the matrix.

    Returns:
        The work in units.
    """
    return 13 * dimension**3 + 32 * dimension**2


def qr_work(rows, columns):
    """Return the work units of the reduced QR factorization of an m x c matrix with m >= c.

    numpy.linalg.qr in its default reduced mode runs LAPACK's zgeqrf, or
    dgeqrf for real input, and then forms the m x c factor Q with zungqr or
    dorgqr. LAPACK's operation counts for the two steps are each
    c**2 (m - c/3) multiply-adds, so together they stay below 2 m c**2.
    The formula counts 2 m c**2 for them and 8 m c for NumPy's copy of the
    input, the copy of Q it returns, the extraction of R and the
    Householder scalars.

    Args:
        rows: m, the number of rows.
        columns: c, the number of columns, at most m.

    Returns:
        The work in units.
    """
    return 2 * rows * columns**2 + 8 * rows * columns
