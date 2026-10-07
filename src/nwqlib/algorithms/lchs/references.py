"""Explicit constant-generator references and their actual numerical work.

ACL denotes An, Childs and Lin, https://arxiv.org/html/2312.03916v2.
Its Eq. (2) specializes to exp(-A*T)u0 + integral exp(-A*(T-s))b ds.
These binary64 evaluations are references, not certified rounding bounds.
"""

from nwqlib._linalg_laws import MAX_EXPONENTIAL_NORM, expm_requirements, seeded_norm_estimates
from nwqlib.operators.access import _check_bytes


def dense_work(checks, counts, *, work, peak_bytes):
    """Admit accumulated dense-kernel size units and simultaneous known arrays.

    Matrix products/decompositions use D**3 units, matvecs D**2 and vector
    operations D. Vendor workspace, FLOPs and elapsed time remain unknown.
    """
    _check_bytes(peak_bytes, checks.max_bytes, "LCHS reference arrays")
    total = counts.get("dense_work", 0) + work
    if total > checks.max_dense_work:
        raise ValueError("LCHS reference exceeds max_dense_work before numerical work")
    counts["dense_work"] = total
    counts["known_peak_bytes"] = max(counts.get("known_peak_bytes", 0), peak_bytes)


def _exponential(matrix, held_bytes, *, checks, counts):
    """Return expm(matrix) with its work admitted first, or None when it is unavailable.

    The exponential is unavailable when the 1-norm of the matrix exceeds
    _linalg_laws.MAX_EXPONENTIAL_NORM or when the result has a nonfinite
    entry, for example when ||A|| T overflows it. held_bytes counts the
    caller's arrays that stay alive during the call.

    For a triangular input, SciPy 1.18.1's scipy.linalg.expm recomputes the
    first superdiagonal after each squaring as t (exp(l2) - exp(l1))/(l2 - l1)
    (scipy/linalg/src/_matfuncs_expm.c), which cancels when two neighboring
    diagonal entries are close but distinct. The closed form of a diagonal A
    with eigenvalues 40 and 1e-12 then had a relative error of 5.4e-6.
    scipy.sparse.linalg.expm recomputes that superdiagonal with Eq. (10.42) of
    Higham's Functions of Matrices (doi:10.1137/1.9780898717778), as Code
    Fragment 2.1 of Al-Mohy and Higham (SIAM J. Matrix Anal. Appl. 31, 970
    (2009), doi:10.1137/09074721X) prescribes, and stays accurate there. On
    stiff inputs that are not triangular, however, it forms the Pade
    quotient as (V - U)^{-1}(V + U), where scipy.linalg.expm uses
    I + 2 (V - U)^{-1} U, and its closed form was several hundred times less
    accurate (2.3e-13 against 3.1e-16 at ||A|| T of about 1e4). A triangular
    matrix therefore goes to scipy.sparse.linalg.expm, a lower triangular one
    through its transpose because that kernel detects only upper
    triangularity, and every other matrix to scipy.linalg.expm. The
    measurements and their inputs are in docs/dependency_issues.md, section
    "Triangular matrix exponential".

    The work and bytes of either kernel are those of
    _linalg_laws.expm_requirements.

    From n = 200 scipy.sparse.linalg.expm estimates the norms of the matrix
    powers with onenormest, which draws from NumPy's global random
    generator, so the call runs inside _linalg_laws.seeded_norm_estimates.
    """
    import numpy as np
    import scipy.linalg
    import scipy.sparse.linalg

    n = len(matrix)
    # n**2 units for the 1-norm and 4 n**2 for the two triangularity tests,
    # which copy one triangle each. Bytes: M and one copy.
    dense_work(checks, counts, work=5*n*n, peak_bytes=32*n*n + held_bytes)
    norm = float(np.linalg.norm(matrix, 1))
    if not norm <= MAX_EXPONENTIAL_NORM:
        return None
    work, data_bytes = expm_requirements(n, norm)
    dense_work(checks, counts, work=work, peak_bytes=data_bytes + held_bytes)
    upper = not np.tril(matrix, -1).any()
    lower = not upper and not np.triu(matrix, 1).any()
    counts["expm_attempts"] += 1
    with np.errstate(all="ignore"):
        if upper or lower:
            with seeded_norm_estimates():
                exponential = scipy.sparse.linalg.expm(matrix.T if lower else matrix)
            if lower:
                exponential = exponential.T
        else:
            exponential = scipy.linalg.expm(matrix)
    counts["expm_completed"] += 1
    return exponential if np.isfinite(exponential).all() else None


def expm_reference(matrix, initial, elapsed, *, checks, counts):
    """Return exp(-A*T) @ u0, or None when the exponential is unavailable (see _exponential)."""
    d = len(initial)
    # 2*d**2 units for scaling A by T and the matvec. Bytes: A, -A*T and six
    # complex128 d-vectors.
    dense_work(checks, counts, work=2*d*d, peak_bytes=16*(2*d*d+6*d))
    # ACL arXiv:2312.03916v2, Eq. (2), b=0 and time-independent A:
    # u(T)=exp(-A*T)u0.
    propagator = _exponential(-matrix * elapsed, 16*(d*d+6*d), checks=checks, counts=counts)
    if propagator is None:
        return None
    counts["matvec_attempts"] += 1
    reference = propagator @ initial
    counts["matvec_completed"] += 1
    return reference


def closed_form_reference(matrix, initial, source, elapsed, *, checks, counts):
    """Return exp(-A*T)u0 + integral_0^T exp(-A*s) ds b, or None when it is unavailable.

    The reference is ACL arXiv:2312.03916v2, Eq. (2), for constant A and b.
    Both terms come from one exponential of the (d+1)-square augmented matrix
    M = [[-A*T, b*T], [0, 0]]. For k >= 1, M**k has blocks (-A*T)**k and
    (-A*T)**(k-1) b*T in its first block row, so expm(M) = [[exp(-A*T),
    phi_1(-A*T) b*T], [0, 1]] with phi_1(X) = sum_{j>=0} X**j/(j+1)!. Since
    phi_1(-A*T)*T = integral_0^T exp(-A*s) ds, the top-right block is the
    Duhamel term of Eq. (2), which equals A^{-1}(I - exp(-A*T))b when A is
    invertible and stays defined when A is singular. Forming I - exp(-A*T)
    explicitly would cancel about log10(1/(||A||*T)) significant digits when
    ||A||*T is small. The augmented exponential has no such subtraction and
    no linear solve with A, and _exponential evaluates it without the
    cancellation of SciPy's triangular branch.
    """
    import numpy as np

    d = len(initial)
    # 2*(d+1)**2 units for forming and scaling M, d**2 for the matvec and d
    # for the final sum. Bytes: A, M and eight complex128 d-vectors.
    dense_work(checks, counts, work=2*(d+1)**2+d*d+d, peak_bytes=16*(d*d+(d+1)**2+8*d))
    augmented = np.zeros((d+1, d+1), dtype=complex)
    augmented[:d, :d] = -matrix*elapsed
    augmented[:d, d] = source*elapsed
    exponential = _exponential(augmented, 16*(d*d+8*d), checks=checks, counts=counts)
    if exponential is None:
        return None
    # The same exponential supplies the homogeneous term, with physical phase.
    counts["matvec_attempts"] += 1
    homogeneous = exponential[:d, :d] @ initial
    counts["matvec_completed"] += 1
    return homogeneous+exponential[:d, d]


def ivp_reference(matrix, initial, source, elapsed, *, checks, counts):
    """Integrate du/dt = -A u + b with RK45 on the real-packed complex state.

    Every right-hand-side call is counted and capped by max_rhs_calls before
    it runs, and SciPy's reported evaluation count must equal that count.
    The caller's rtol and atol control the local error only, so the result is
    an independent numerical reference, not a certified solution.
    """
    import numpy as np
    import scipy.integrate

    d = len(initial)
    dense_work(checks, counts, work=6*d+1, peak_bytes=16*d*d+128*d+8)
    if elapsed == 0:
        return initial

    def rhs(_time, packed):
        if counts["rhs_attempts"] >= checks.max_rhs_calls:
            raise ValueError("IVP reference RHS evaluation cap exhausted")
        dense_work(checks, counts, work=d*d+7*d, peak_bytes=16*d*d+224*d+8)
        counts["rhs_attempts"] += 1
        state = packed[:d] + 1j*packed[d:]
        counts["matvec_attempts"] += 1
        # ACL arXiv:2312.03916v2, Eq. (1): du/dt=-A*u+b. Pack Re(u) and Im(u),
        # preserving the signed complex equation, with no normalization or
        # phase alignment.
        derivative = -(matrix @ state) + source
        counts["matvec_completed"] += 1
        result = np.concatenate((derivative.real, derivative.imag))
        counts["rhs_completed"] += 1
        return result

    packed = np.concatenate((initial.real, initial.imag))
    counts["ivp_attempts"] += 1
    # SciPy solve_ivp, method RK45: Dormand-Prince 5(4), with local error
    # atol+rtol*abs(y). These caller controls are not global solution error.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    result = scipy.integrate.solve_ivp(rhs, (0.,elapsed), packed, method="RK45", rtol=checks.rtol,
        atol=checks.atol, t_eval=(elapsed,), dense_output=False)
    counts["ivp_completed"] += 1
    if not result.success:
        raise RuntimeError(f"solve_ivp failed: {result.message}")
    if result.y.shape != (2*d,1):
        raise ValueError("IVP reference must return its single requested terminal vector")
    if result.nfev != counts["rhs_attempts"]:
        raise ValueError("IVP reported evaluations differ from actual reference RHS calls")
    return result.y[:d,0] + 1j*result.y[d:,0]
