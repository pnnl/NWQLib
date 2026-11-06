"""Quadrature data, dense terms and product-formula plans for time-independent LCHS.

This term generator supplies the homogeneous ``b(t) = 0`` specialization.
The constant-source construction adds source-state and elapsed-time quadrature
above the same ``A = L + iH`` decomposition and ``k``-quadrature machinery.

ACL is An, Childs, and Lin, arXiv:2312.03916v2, and CSTWZ is Childs, Su,
Tran, Wiebe, and Zhu, doi:10.1103/PhysRevX.11.011020. The Cartesian
decomposition A = L + iH is ACL Eqs. (3)-(4), and the kernel, the integral
discretization and the pointwise decay of the quadrature that this module
uses are ACL Eq. (7), Eq. (61) and Eqs. (185)-(186). The quadrature rules
(``providers.py``) integrate that decay using
E1(x) <= exp(-x)/x and select the cutoff K by scalar bisection, and their
automatic Q-point grid uses Trefethen,
Approximation Theory and Approximation Practice, ISBN 978-1-61197-239-9,
Theorem 19.3, with n = Q - 1. This module uses the bounds and counts of
providers.py without recomputing them. Pocrnic, Johnson, Katabarwa, and
Wiebe, arXiv:2506.20760v2, Lemmas 3-4, give constant-factor rules for K
and Q instead (Lemma 3 solves the ACL Eq. (62) truncation bound for K with
the Lambert W function), and their Section IV, Eqs. (61)-(65), compiles
SELECT as one simulation of the effective Hamiltonian
sum_j |j><j| (x) (k_j L + H). The qsp_block_encoding backend follows that
idea through subroutines/qsp but builds the generator from separate L and H
encodings, so the constants of their Lemmas 7-8, which assume one block
encoding of A, do not apply. CSTWZ Propositions 9 and 10, Eqs. (120)-(121),
bound the product-formula steps.
"""

from __future__ import annotations

from nwqlib._limits import ResourceLimitError, exceeds, optional_limit

from dataclasses import dataclass, replace
from functools import cached_property
from itertools import cycle, islice
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from nwqlib._validation import integer
from nwqlib.core.records import FrozenArray
from nwqlib.algorithms.lchs.time_independent_common import (
    as_square_matrix,
    validate_final_time,
)
from nwqlib.algorithms.lchs.native import (
    LCHS_TROTTER_EPSILON_FRACTION,
)
from nwqlib.algorithms.lchs.solution_error_budget import (
    DEFAULT_PSD_TOLERANCE,
    _numerical_psd_decision,
    psd_recovery,
    psd_recovery_exponent,
    psd_recovery_part,
    scaled_sum,
    signed_shift,
)
from nwqlib.algorithms.lchs.providers import (
    LCHSCoefficientPlan,
    LCHSProblemContext,
    _resolve_coefficient_context,
    lifted_magnitudes,
)

if TYPE_CHECKING:
    from nwqlib.algorithms.lchs.method import LCHS
    from nwqlib.algorithms.lchs.select_synthesis import LCHSProductFormulaSelectPlan
    from nwqlib.subroutines.trotterization import TrotterStepSelection


def _cartesian_parts(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Form L = (A + A†)/2 and H = -0.5j*(A - A†) from one conjugate-transpose workspace.

    The adjoint is a new C-ordered conjugated array, so both parts are
    private new arrays: L is formed first, then A - A† overwrites the
    adjoint (which does not overlap A) and becomes H. Scaling in place
    performs the same elementary operations and gives the same rounded
    finite values as the expressions 0.5*(A + A†) and -0.5j*(A - A†). With
    A still in memory, this holds 48 d**2 bytes of complex d-square arrays.
    """
    adjoint = np.conjugate(array.T, order="C")
    l_part = np.add(array, adjoint)
    np.subtract(array, adjoint, out=adjoint)
    l_part *= 0.5
    adjoint *= -0.5j
    return l_part, adjoint


def _adopt_read_only(array: np.ndarray) -> np.ndarray:
    """Mark a privately owned new array read-only without copying it.

    Only arrays that own their data are adopted, so a caller's input or a
    view of another owner is never frozen in place.
    """
    if not array.flags.owndata:
        raise ValueError("only a privately owned array can be adopted read-only")
    array.flags.writeable = False
    return array


def _hermitian_endpoints(l_part: np.ndarray, *, real: bool, max_spectral_work: int | None) -> tuple[float, float]:
    """Return (lambda_min, lambda_max) of the Hermitian part L, from its exact structure or one eigensolve.

    One count of the nonzero entries of L and of its diagonal decides,
    without a tolerance, whether L is zero or diagonal. A zero L has the
    endpoints (0, 0). A diagonal L has the smallest and largest diagonal
    entry: the diagonal (a + conj(a))/2 of L = (A + A†)/2 is exactly real, so
    these endpoints are exact. Any other L takes one eigenvalues-only solve,
    admitted first as d**3 work units against max_spectral_work. With
    ``real`` (A was real) L is real symmetric and its real part goes to the
    real symmetric solver. Both solvers are backward stable, so the
    endpoints are eigensolver values within rounding
    (solution_error_budget._numerical_psd_decision).
    """
    diagonal = np.diagonal(l_part)
    nonzero = np.count_nonzero(l_part)
    if nonzero == np.count_nonzero(diagonal):
        if not nonzero:
            return 0.0, 0.0
        values = diagonal.real
        return float(values.min()), float(values.max())
    d = l_part.shape[0]
    if exceeds(d**3, max_spectral_work):
        raise ResourceLimitError(
            f"LCHS eigensolve exceeds max_spectral_work: it needs {d**3} work units (d**3 with d={d}), "
            f"max_spectral_work={max_spectral_work}. Raise max_spectral_work to at least {d**3}.")
    eigenvalues = np.linalg.eigvalsh(l_part.real if real and np.iscomplexobj(l_part) else l_part)
    return float(eigenvalues[0]), float(eigenvalues[-1])


def cartesian_decomposition(
    matrix: Any,
    *,
    eigcheck: bool = True,
    psd_tolerance: float = DEFAULT_PSD_TOLERANCE,
    max_bytes=None,
    max_spectral_work=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Hermitian parts `(L, H)` of `A = L + iH` used by LCHS.

    `L = (A + A^dagger)/2` and `H = (A - A^dagger)/(2i)`, as in An, Childs
    and Lin, arXiv:2312.03916v2, Eqs. (3)-(4). LCHS needs L positive
    semidefinite (PSD), which keeps `||exp(-A t)|| <= 1` (same paper,
    Lemma 21, Eq. (162)). With `eigcheck=True` the function also checks that
    condition. `eigcheck=False` only decomposes. It does not establish
    the PSD condition or replace the check of `LCHS`, which tests its own
    eigenvalue endpoints.

    Args:
        matrix (array_like): Finite square matrix A.
        eigcheck (bool): Check that L is PSD within `psd_tolerance`.
        psd_tolerance (float): Default `1e-12`. Relative window of the
            check. An eigenvalue of L at or above `-psd_tolerance*||L||_2`
            passes.
        max_bytes (int | None): Default `None`, the physical memory of this computer.
            Upper limit on the arrays of the check, `64 d**2 + 16 d` bytes
            for dimension d, tested before they are formed. The count covers
            the known NumPy arrays, not the eigensolver's own LAPACK
            workspace.
        max_spectral_work (int | None): Default `None`, no limit. Upper
            limit on the eigenvalue check, counted as `d**3` work units,
            which are not timings, only when L is neither zero nor
            diagonal. The endpoints of a zero or diagonal L are read from
            its diagonal, and for an A of real dtype they come from the
            real symmetric L.

    Returns:
        parts (tuple[numpy.ndarray, numpy.ndarray]): L and H as complex
            arrays.

    Raises:
        ValueError: If A is not a finite square matrix, or, with `eigcheck`,
            if L has an eigenvalue below `-psd_tolerance*||L||_2` or the
            check exceeds `max_spectral_work` or `max_bytes`.
    """
    from nwqlib.operators.access import _check_bytes

    # Form the Cartesian parts L=(A+A†)/2 and H=(A−A†)/(2i) from one
    # conjugate-transpose workspace (_cartesian_parts). With A live, the parts
    # hold 48 d**2 bytes. The eigenvalue check of an L that is neither zero
    # nor diagonal (_hermitian_endpoints) is admitted first: its known NumPy
    # arrays, A, L, H, the solver's internal d-square copy and the internal
    # and public eigenvalue vectors, are
    # B_cartesian = 64 d**2 + 16 d bytes, and its work is d**3 against
    # max_spectral_work. The eigensolver's queried LAPACK workspace
    # Q_N(d) = 16 l_N + 8 r_N + I i_N (JOBZ=N) is not charged: this is a
    # declared known-array logical workspace, not a complete workspace cap.
    # These units are admission proxies, not timings or equal-cost CPU
    # operations.
    if eigcheck:
        # Read the input once in its own dtype: a real dtype gives a real
        # symmetric L for the eigenvalue check.
        matrix = np.asarray(matrix)
        real = not np.iscomplexobj(matrix)
    array = as_square_matrix(matrix)
    d = array.shape[0]
    if eigcheck:
        _check_bytes(64*d*d + 16*d, max_bytes, "LCHS Cartesian parts and eigenvalue check")
    l_part, h_part = _cartesian_parts(array)

    if eigcheck:
        lower, upper = _hermitian_endpoints(l_part, real=real,
            max_spectral_work=optional_limit(max_spectral_work, "max_spectral_work"))
        _numerical_psd_decision(
            lambda_min=lower,
            lambda_max=upper,
            psd_tolerance=float(psd_tolerance),
            make_l_psd=False,
        )
    return l_part, h_part


def _node_eigensystem(k_value: float, l_part: np.ndarray, h_part: np.ndarray):
    """Return (eigenvalues, vectors) of the Hermitian node generator A_k = k*L + H.

    Apply all elapsed-time branches of a fixed Hermitian generator from one
    eigensystem. For A_k=kL+H=V diag(lambda) V†, its time-t action is
    V(exp(-it lambda) * (V†v)). Each source application uses its own elapsed
    time and PSD recovery part. The classical sum processes one node at a
    time and accumulates vectors. Explicit dense branch matrices require a matrix product at each
    time. Numerical eigensystem, phase and multiplication errors are separate
    from the ideal finite-sum approximation.

    The generator is formed by one multiply and one add and must be finite.
    This replaces the quick norm check made before scipy.linalg.expm, which
    this route does not use. Columns may rotate within a degenerate
    eigenspace without changing the matrix function, and no inverse
    eigenvalue gap enters its operator error (ACL arXiv:2312.03916v2,
    Eq. (70) defines the branch). A Hermitian eigensolver is normwise
    backward stable (LAPACK Users' Guide, 3rd ed.,
    doi:10.1137/1.9780898719604, Sec. 4.7): with
    rho = ||A V - V Lambda||, eta = ||V†V - I|| < 1 and
    r = eta/(1 + sqrt(1 - eta)), beta = rho + r(||A|| + ||Lambda||), and
    phase and formation errors eps_phase and eps_form,
    ||U_hat(t) - exp(-itA)|| <= |t| beta + 2r + r**2 + (1 + eta) eps_phase + eps_form.
    No reorthonormalization is performed.
    """
    generator = np.multiply(float(k_value), l_part)
    generator += h_part
    if not np.isfinite(generator).all():
        raise ValueError("LCHS node generator k*L + H must be finite")
    return np.linalg.eigh(generator)


def _branch_unitary(eigensystem, elapsed: float, dimension: int) -> np.ndarray:
    """Return U_k(t) = (V * exp(-i t lambda)) @ V† from one node eigensystem.

    A zero elapsed time returns the identity directly, not the rounded V V†.
    The phase products must be finite. The dense product costs D**3 per
    branch. Forming the phase and scaling the columns add 2*D**2 + 3*D.
    """
    if elapsed == 0:
        return np.eye(dimension, dtype=complex)
    eigenvalues, vectors = eigensystem
    phase = np.exp(-1j * float(elapsed) * eigenvalues)
    if not np.isfinite(phase).all():
        raise ValueError("LCHS branch phase products must be finite")
    return (vectors * phase) @ vectors.conj().T


def spectral_lchs_sum(l_part, h_part, nodes, coefficients, applications, counts=None):
    """Return sum_k c_k V_k [sum_a w_a (exp(-i t_a Lambda_k) * (V_k† v_a))], streamed over nodes.

    applications contains (elapsed time, already scaled weight, physical
    vector). The weight includes psd_recovery_part, not its carried 2**e.
    Each node's eigensystem is formed once and released before the next,
    each distinct input vector is projected once per node (a cache of at
    most the initial and source inputs, scoped to this call and keyed by the
    vector's identity), the phases of every application are accumulated in
    the eigenbasis and one final V action is made per node. The
    node-coefficient accumulation order is kept. An exactly zero L shares
    one H eigensystem across nodes. If L and H are both zero or every
    elapsed time is zero, every action is the identity and no eigensystem is
    formed. ``counts`` receives eigh_calls and matvecs.

    The dense exact route reuses one Hermitian eigensystem for each distinct
    kL+H across source elapsed times. Classical evaluation applies its phases to
    vectors without forming each dense propagator.
    """
    counts = {} if counts is None else counts
    counts.setdefault("eigh_calls", 0)
    counts.setdefault("matvecs", 0)
    result = np.zeros(h_part.shape[0], dtype=np.complex128)
    if not (np.any(l_part) or np.any(h_part)) or not any(elapsed for elapsed, _, _ in applications):
        for coefficient in coefficients:
            combined = np.zeros_like(result)
            for _, weight, vector in applications:
                combined += weight * vector
            result += coefficient * combined
        return result
    shared = None
    zero_l = not np.any(l_part)
    for node, coefficient in zip(nodes, coefficients, strict=True):
        if zero_l:
            if shared is None:
                shared = _node_eigensystem(0.0, l_part, h_part)
                counts["eigh_calls"] += 1
            eigenvalues, vectors = shared
        else:
            eigenvalues, vectors = _node_eigensystem(node, l_part, h_part)
            counts["eigh_calls"] += 1
        combined = np.zeros_like(result)
        projected = {}
        for elapsed, weight, vector in applications:
            transformed = projected.get(id(vector))
            if transformed is None:
                transformed = np.conjugate(vectors.T @ np.conjugate(vector))
                projected[id(vector)] = transformed
                counts["matvecs"] += 1
            combined += weight * np.exp(-1j * elapsed * eigenvalues) * transformed
        result += coefficient * (vectors @ combined)
        counts["matvecs"] += 1
    return result


def _spectral_host_requirements(dimension, *, nodes, applications, inputs, eigensystems):
    """Return (W_host, phase bytes) of spectral_lchs_sum for D, K nodes, A applications and r inputs.

    With one dense matrix product D**3 units and one matvec D**2 units
    (_linalg_laws) and one pointwise operation or reduction input one visit,

        W_host = D + K_eig*(8*D**3 + 32*D**2 + 2*D**2)
                   + K*((r+1)*D**2 + (2*r+3)*D + 6*A*D),

    the output zero fill, each eigensystem with its two generator passes,
    the r input projections and one output projection, two conjugations per
    input, the combined zero fill and the final coefficient multiply/add, and
    six pointwise visits per application (phase argument, exponential,
    weight and coordinate products and accumulation). K_eig is K, one for an
    exactly zero L, and zero for identity actions. An elementary exponential
    counts as one visit, not one hardware operation. The dense classical
    action reuses each Hermitian eigensystem across elapsed times and projects each
    distinct input once per generator.

    Bytes, excluding the caller's inputs held in memory (B_held):
    max(32*D**2 + 16*D, 48*D**2 + 32*D, 16*D**2 + (160 + 32*r)*D): the
    generator formation, the eigensolver's internal copy, public vectors and
    both eigenvalue vectors with the result, and the final phase with the
    eigenvectors, cached projections, combined vector, conjugate temporaries
    and phase products. The queried LAPACK workspace Q_V(D) of NumPy's zheevd
    is not counted, because this is a declared known-array logical workspace,
    not a complete workspace limit. These units are proxies for the limit
    checks, not timings or equal-cost CPU operations.
    """
    D, K, A, r = dimension, nodes, applications, inputs
    work = D + eigensystems*(8*D**3 + 34*D**2) + K*((r+1)*D**2 + (2*r+3)*D + 6*A*D)
    return work, max(32*D*D + 16*D, 48*D*D + 32*D, 16*D*D + (160 + 32*r)*D)


def _spectral_branch_requirements(dimension, *, eigensystems, branches):
    """Return (work, bytes) of forming explicit branch matrices from node eigensystems.

    W = K_eig*(8*D**3 + 32*D**2 + 2*D**2) + B*(D**3 + 2*D**2 + 3*D) for B
    physical nonzero-time branches: each eigensystem with its generator
    passes, and per branch the dense product (V*phase) @ V.conj().T, column
    scaling and conjugation, and three visits per entry of the phase. The
    synthesis, control and any preparation-matrix work are added by the
    caller (W_synthesis, W_source). Branches built as explicit circuits
    additionally form a dense matrix and add the cost of their chosen
    synthesis. Bytes before synthesis are the peak of the matrix expression,
    64*D**2 + 24*D (V, the scaled matrix, conjugated V, result and
    phase/eigenvalue vectors). The caller adds the inputs in memory, cached
    eigensystems, synthesis working bytes and completed branch circuits. The
    queried LAPACK workspace is not counted (a declared known-array logical
    workspace). These units are proxies for the
    limit checks, not timings or equal-cost CPU operations.
    """
    D = dimension
    return eigensystems*(8*D**3 + 34*D**2) + branches*(D**3 + 2*D**2 + 3*D), 64*D*D + 24*D


# Absolute imaginary-residue window of the product-formula census only, the
# window of its operator validation (error_budget._validated_terms). Pruning
# and every other check of a combined node read that node's relative cutoff
# (combine_aligned). A node above unit scale can therefore pass its own
# cutoff and still be refused by the census. Registered in
# ENGINEERING_CONSTANTS with its revisit condition.
_PAULI_COEFFICIENT_ATOL = 1.0e-12


@dataclass(frozen=True, kw_only=True)
class _TrotterPauliDecomposition:
    """Pauli coefficients of L and H, computed once and reused for every k-node.

    The node generator k*L + H has Pauli coefficient k*l_P + h_P for label P,
    so one decomposition of each part serves all nodes. Both dictionaries
    list their labels in lexical (I < X < Y < Z) order, as
    operators._pauli.pauli_coefficients emits them.
    """

    num_qubits: int
    identity_label: str
    l_coefficients: Mapping[str, complex]
    h_coefficients: Mapping[str, complex]

    @cached_property
    def aligned_rows(self) -> tuple[tuple[str, complex, complex, bool, bool], ...]:
        """Merge the lexically ordered L and H supports once, as (label, h, l, has_l, is_identity) rows.

        A two-way merge of the m_L + m_H = s ordered keys makes at most s - 1
        string comparisons of at most q characters, and the union has u rows.
        It is counted as W_align = (q+4)*s + (q+2)*u logical visits
        (_coefficient_preparation_work). No hash or sort is repeated at a node.
        A row absent from H carries h = 0.0 and one absent from L has
        has_l False, so combine_aligned reproduces the dictionary arithmetic.
        """
        l_items = tuple(self.l_coefficients.items())
        h_items = tuple(self.h_coefficients.items())
        rows = []
        i = j = 0
        while i < len(l_items) or j < len(h_items):
            if j == len(h_items) or (i < len(l_items) and l_items[i][0] < h_items[j][0]):
                label, l_value = l_items[i]
                rows.append((label, 0.0, l_value, True, label == self.identity_label))
                i += 1
            elif i == len(l_items) or h_items[j][0] < l_items[i][0]:
                label, h_value = h_items[j]
                rows.append((label, h_value, 0.0, False, label == self.identity_label))
                j += 1
            else:
                label = l_items[i][0]
                rows.append((label, h_items[j][1], l_items[i][1], True, label == self.identity_label))
                i += 1
                j += 1
        return tuple(rows)


@dataclass(frozen=True, kw_only=True)
class _CombinedTrotterTerms:
    """Pauli form of one node generator k*L + H after pruning.

    identity_coefficient is the real coefficient c_I of the identity label,
    applied as the branch phase exp(-i*t*c_I). traceless_terms are the kept
    non-identity (label, coefficient) pairs in sorted label order.
    pruned_labels records the labels removed by the node's coefficient_cutoff
    (combine_aligned), and pruned_l1_mass is a finite binary64 upper bound
    d_up on the summed moduli of their coefficients
    (error_budget.upper_dropped_mass), from which each application reports
    its own pruning bound. The cutoff is also the node's imaginary-residue
    window. kept_indices and pruned_indices are the positions of the kept
    and dropped rows in the decomposition's aligned union
    (_TrotterPauliDecomposition.aligned_rows), recorded while the rows are
    combined, so the saved node table needs no label hashing.
    """

    identity_coefficient: float
    traceless_terms: tuple[tuple[str, complex], ...]
    pruned_labels: tuple[str, ...]
    pruned_l1_mass: float
    kept_indices: tuple[int, ...] = ()
    pruned_indices: tuple[int, ...] = ()
    coefficient_cutoff: float = 0.0


@dataclass(frozen=True, kw_only=True)
class _LCHSTrotterNodeRecord:
    """LCHS-only product-formula accounting for one quadrature node.

    All bounds are operator-norm bounds on one node evolution
    exp(-i*t*(k*L+H)), before the node's LCU coefficient, the physical input
    norm and the PSD recovery are applied.

    A pf_bound_value computed from the census (the anticommuting Pauli pairs
    and nested commutation indicators of error_budget.coefficient_up) uses a
    coefficient evaluated with
    scaled binary64 products and sums rounded upward, followed by exact
    rational rescaling. bound_variant identifies the expression being
    enclosed.

    Attributes:
        pf_bound_value: Commutator bound of CSTWZ doi:10.1103/PhysRevX.11.011020,
            Props. 9-10, for the kept terms at the chosen step count, or None
            when it was not evaluated.
        pruned_l1_mass: Finite upper bound on the summed moduli of the pruned
            Pauli coefficients (error_budget.upper_dropped_mass).
        combined_bound_value: up(pruning + pf_bound_value), with the pruning
            bound up(|t|*pruned_l1_mass) (error_budget.upper_combined_error),
            or None.
        step_count: Product-formula repetitions for this node. Padding and
            zero-time address slots have zero, and so does a budgeted node or a
            classical action whose kept generator is empty. Fixed-step SELECT
            address slots keep the common count, with zero angles for an empty
            generator.
        selection: The TrotterStepSelection of the budgeted selector, or None.
        bound_value_status: How pf_bound_value was obtained, for example
            ``structural_upper_bound``, ``not_evaluated`` or ``not_applicable``.
        dense_bound_value: Optional dense-matrix evaluation of the same bound
            plus the pruning term, used only to check the structural bound.
        pauli_term_count: Kept Pauli terms used by an evaluated fixed-step bound.
        pair_commutation_checks: Pauli pair commutation tests newly performed
            for that bound. The first node evaluated from a shared ordered
            structure carries the structure's tests and later nodes and
            applications zero.
        nested_commutation_checks: Nested commutation tests newly performed
            for that bound, counted as the pair tests are.
        bound_variant: Pauli-triangle expression of pf_bound_value,
            "exact_census" or "relaxed_prefix", or None when no census
            produced it (no kept term, not evaluated or not applicable).
    """

    pf_bound_value: float | None
    pruned_l1_mass: float
    combined_bound_value: float | None
    step_count: int
    selection: TrotterStepSelection | None
    bound_value_status: str = "not_evaluated"
    dense_bound_value: float | None = None
    pauli_term_count: int = 0
    pair_commutation_checks: int = 0
    nested_commutation_checks: int = 0
    bound_variant: str | None = None

    @property
    def bound_method(self) -> str | None:
        """Return ``pauli_triangle`` for a structural bound, otherwise None."""
        if self.bound_value_status == "structural_upper_bound":
            return "pauli_triangle"
        return None

    @property
    def algebraic_work_counts(self) -> dict[str, int]:
        """Return the Pauli term and commutation-check counts of this node's bound.

        A budgeted node reads them from its TrotterStepSelection. A fixed-step
        node reads the counts stored on this record.
        """
        selection = self.selection
        return {
            "pauli_terms": (
                self.pauli_term_count if selection is None else selection.pauli_term_count
            ),
            "pair_commutation_checks": (
                self.pair_commutation_checks
                if selection is None
                else selection.pair_commutation_checks
            ),
            "nested_commutation_checks": (
                self.nested_commutation_checks
                if selection is None
                else selection.nested_commutation_checks
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the reported per-node record.

        Unless the optional dense check ran, dense_bound_value and
        dense_bound_method are None and dense_bound_status and
        structural_vs_dense_validation_status read ``not_evaluated``. When it
        ran, the structural bound was already compared with it in
        _fixed_trotter_certificate_records, which raises if the structural
        value is smaller by more than its 64-ulp rounding window, so the
        comparison status reads ``structural_upper_bound_confirmed``.
        """
        return {
            "pf_bound_value": self.pf_bound_value,
            "pruned_l1_mass": self.pruned_l1_mass,
            "combined_bound_value": self.combined_bound_value,
            "step_count": self.step_count,
            "bound_method": self.bound_method,
            "bound_value_status": self.bound_value_status,
            "bound_variant": self.bound_variant,
            "dense_bound_value": self.dense_bound_value,
            "dense_bound_method": "dense" if self.dense_bound_value is not None else None,
            "dense_bound_status": (
                "dense_validation_value"
                if self.dense_bound_value is not None
                else "not_evaluated"
            ),
            "structural_vs_dense_validation_status": (
                "structural_upper_bound_confirmed"
                if self.dense_bound_value is not None
                else "not_evaluated"
            ),
            "algebraic_work_counts": self.algebraic_work_counts,
        }


def _pauli_coefficients(matrix: np.ndarray) -> tuple[int, dict[str, complex]]:
    """Return (qubit count, {label: coefficient}) of the exact Pauli expansion.

    Zero tolerances keep every nonzero coefficient. Pruning is applied later
    per node, where its mass is added to the synthesis bound.
    """
    from nwqlib.subroutines.pauli_decomposition import decompose_matrix_to_pauli
    decomposition = decompose_matrix_to_pauli(matrix, atol=0.0, rtol=0.0)
    return decomposition.num_qubits, {
        term.label: complex(term.coefficient) for term in decomposition.terms
    }


def _trotter_pauli_decomposition(
    l_part: np.ndarray,
    h_part: np.ndarray,
) -> _TrotterPauliDecomposition:
    """Decompose the prepared L and H once for all product-formula nodes."""
    l_qubits, l_coefficients = _pauli_coefficients(l_part)
    h_qubits, h_coefficients = _pauli_coefficients(h_part)
    if l_qubits != h_qubits:
        raise ValueError("L and H Pauli decompositions must use the same qubit count")
    return _TrotterPauliDecomposition(
        num_qubits=l_qubits,
        identity_label="I" * l_qubits,
        l_coefficients=l_coefficients,
        h_coefficients=h_coefficients,
    )


def _pauli_decomposition_requirements(dimension):
    """Return (work, peak bytes) of _trotter_pauli_decomposition for a d-by-d L and H.

    The formula and its derivation are stated at _admit_pauli_decomposition.
    With q = log2(d), N = d**2 and h = q//2, the half-label widths are (h,)
    for even q and (h, q-h) for odd q. The work is (34*q+28)*N + 2*H and the
    peak bytes (104+2*q)*N + T, with T the sum of j*4**j and H the sum of
    (j+1)*4**j over those widths. Under an explicit max_select_work=1_000_000_000,
    the decomposition stage accepts d = 1024 (q = 10), with 385,888,256
    work units and 130,028,544 bytes. At d = 2048 (q = 11), its
    1,686,179,840 work units exceed that limit. Later SELECT stages set
    aside their own work and bytes within the shared limits.
    """
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("dimension must be a positive power of two")
    q = dimension.bit_length() - 1
    n = dimension * dimension
    h = q // 2
    widths = (h,) if q % 2 == 0 else (h, q - h)
    half_chars = sum(width * 4**width for width in widths)
    half_work = sum((width + 1) * 4**width for width in widths)
    work = (34 * q + 28) * n + 2 * half_work
    peak_bytes = (104 + 2 * q) * n + half_chars
    return work, peak_bytes


def _admit_pauli_decomposition(dimension: int, *, max_bytes, max_select_work) -> tuple[int, int | None]:
    """Check both dense Pauli transforms against the limits and return their work and the SELECT work remaining.

    The inputs are already allocated complex128 matrices of power-of-two
    order d. Put q = log2(d), N = d**2 and h = q//2. The half-label table
    widths are (h,) for even q and (h, q-h) for odd q. Define T as the sum
    of j*4**j over those widths and H as the sum of (j+1)*4**j.

    One transform level evaluates eight real half-sums over N/4 entries.
    Each half-sum uses four safety-test operations and at most three
    arithmetic operations per entry. Four negated blocks add N operations
    per level, giving at most 15*q*N numerical visits per matrix. The
    zero, finiteness and output-index scans add 6*N visits. An output row
    uses q character writes, q first-hash visits and eight coefficient or
    record visits. With at most N rows for each of L and H, the total work
    set aside is (34*q+28)*N + 2*H.

    L and H are transformed sequentially. During H's extraction, both
    inputs, the preceding level and the final coefficient level occupy
    64*N bytes. The index vector adds 8*N, both logical output tables add
    2*N*(q+16), and the half-label characters add T. The resulting
    (104+2*q)*N + T byte bound also covers the transform temporaries
    and the scalar and one-qubit cases. Both final reshapes are views.

    Work uses numerical-entry, character and coefficient-record visits,
    with indexing bookkeeping bundled into its operation. It is a size
    formula rather than a CPU-time or hardware-FLOP count. Bytes cover explicit
    array and logical table contents, excluding Python container overhead
    and process RSS. Other arrays of the caller are counted by the code that
    holds them.
    The transform omits exact zeros and performs no threshold pruning.

    These units are proxies for the limit checks, not timings or equal-cost CPU
    operations. The formula is the block identity of
    operators._pauli.pauli_coefficients counted by
    _pauli_decomposition_requirements.

    An explicit max_select_work=1_000_000_000 accepts this stage through
    d = 1024 (385,888,256 work units and 130,028,544 bytes), while d = 2048
    needs 1,686,179,840 work units. This is a decomposition-stage boundary, not
    a guarantee that a complete dense Plan at q = 10 fits every limit.

    Args:
        dimension: Positive power-of-two order d after any padding.
        max_bytes: Limit on this stage's known simultaneous payload bytes.
        max_select_work: Shared decomposition and SELECT work limit.

    Returns:
        ``(work, remaining)``: the work set aside for the two transforms and
        the extraction, and the limit minus it, or None without a limit.
        Work and bytes equal to their limits are accepted.
    """
    from nwqlib.operators.access import _check_bytes
    work, peak_bytes = _pauli_decomposition_requirements(dimension)
    _check_bytes(peak_bytes, max_bytes, "LCHS Pauli decomposition")
    max_select_work = optional_limit(max_select_work, "max_select_work")
    if exceeds(work, max_select_work):
        raise ResourceLimitError(
            f"LCHS Pauli decomposition needs {work} units of max_select_work, above the limit "
            f"{max_select_work}; it is the first stage charged to max_select_work, "
            f"so this stage needs that limit to be at least {work}")
    return work, None if max_select_work is None else max_select_work - work


# This serves LCHS only. QPE's controlled powers make their own identity and
# pruning selection in algorithms/qpe/powers.py.
def _combined_trotter_terms(
    decomposition: _TrotterPauliDecomposition,
    *,
    k_value: float,
) -> _CombinedTrotterTerms:
    """Form H + k L, separate its identity phase and record the coefficient mass removed by pruning.

    The identity coefficient c_I depends on k through k*tr(L)/2**q, so
    exp(-i*t*c_I) differs between branches. It is a relative phase inside
    the LCU sum, not a global phase, and it is kept as a per-branch phase.
    Pauli coefficients of the Hermitian L and H are real up to decomposition
    rounding. An identity coefficient whose imaginary part exceeds the
    node's cutoff raises here. Other labels are checked where their rotation
    angles are formed.

    This function computes the node's dropped mass. The cutoff test runs before
    the identity is separated, so an identity coefficient it drops belongs
    in the mass. The mass is a finite binary64 upper bound on the sum of
    abs(real(c)) + abs(imag(c)) over precisely the dropped
    coefficients, their correctly rounded sum moved up two units in the last
    place when more than one is nonzero (error_budget.upper_dropped_mass). That
    sum bounds the sum of their moduli. It is not the exact sum of complex
    magnitudes.
    """
    return combine_aligned(decomposition.aligned_rows, k_value)


def combine_aligned(rows, k) -> _CombinedTrotterTerms:
    """Combine and prune one k-node from the aligned L/H rows, in their lexical order.

    For each (label, h, l, has_l, is_identity) row the value is
    complex(h + k*l) when L has the label and complex(h) otherwise, the
    original operand order, so an H-only row is not multiplied by an
    artificial zero and an L-only row keeps the addition to 0.0.

    One cutoff serves the whole node, PAULI_COEFFICIENT_RTOL times the node's
    operand scale max_l max(|h_l|, |k*l_l|) over all its labels, the identity
    included, and 0 for an all-zero node. It is homogeneous,
    cutoff(s*H, s*L, k) = s*cutoff(H, L, k), so a change of unit A -> s*A,
    T -> T/s keeps the same terms. Taking the operands, not |h + k*l|, keeps
    the roundoff residual of an exact cancellation below the cutoff. It can
    still remove a real small term beside a large one, including a rotation
    beside a large identity, so a fixed-step route reports the mass it dropped
    (_trotter_pruning_bound) and a budgeted route charges it to its allowance.

    A value whose modulus is at most the cutoff is dropped. A kept identity
    is the real branch phase coefficient. Other kept values are the traceless
    terms. The kept coefficients and their order equal those of combining
    the dictionaries and sorting the union. The dropped values are passed one
    by one to error_budget.upper_dropped_mass, whose result is a certified
    upper bound on the summed moduli of exactly the dropped coefficients.

    Work per node, in scalar visits (_coefficient_preparation_work): one
    visit per union row for the operand scale, two arithmetic visits per L
    row, at most six conversion, comparison, output and reduction visits per
    union row, one tuple-copy visit per output row and four scalar identity
    checks, 2*m_L + 8*u + 4, plus the mass stream's 10*u + 5. These units are
    proxies for the limit checks, not timings or equal-cost CPU operations.
    """
    from nwqlib.subroutines.pauli_decomposition import PAULI_COEFFICIENT_RTOL
    from nwqlib.subroutines.trotterization.error_budget import upper_dropped_mass

    kept, pruned, kept_indices, pruned_indices = [], [], [], []
    identity = 0.0
    cutoff = PAULI_COEFFICIENT_RTOL*max(
        (max(abs(h), abs(k*l_value) if has_l else 0.0) for _, h, l_value, has_l, _ in rows), default=0.0)

    def dropped_values():
        nonlocal identity
        for position, (label, h, l_value, has_l, is_identity) in enumerate(rows):
            value = complex(h + k*l_value) if has_l else complex(h)
            magnitude = abs(value)
            if magnitude <= cutoff:
                pruned.append(label)
                pruned_indices.append(position)
                yield value
            elif is_identity:
                if abs(value.imag) > cutoff:
                    raise ValueError(
                        "Hamiltonian coefficients must be real for product-formula "
                        "synthesis; identity term has coefficient "
                        f"{value} (build the operator from real coefficients "
                        "or simplify away numerical imaginary parts)"
                    )
                identity = float(value.real)
            else:
                kept.append((label, value))
                kept_indices.append(position)

    mass_up = upper_dropped_mass(dropped_values())
    return _CombinedTrotterTerms(
        identity_coefficient=identity,
        traceless_terms=tuple(kept),
        pruned_labels=tuple(pruned),
        pruned_l1_mass=mass_up,
        kept_indices=tuple(kept_indices),
        pruned_indices=tuple(pruned_indices),
        coefficient_cutoff=cutoff,
    )


def _coefficient_preparation_work(q, m_l, m_h, union, nodes):
    """Return (W_align, W_nodes): alignment once and combination of ``nodes`` distinct k-nodes.

    With s = m_L + m_H and u the union size including identity,
    W_align = (q+4)*s + (q+2)*u and each node costs 2*m_L + 8*u + 4
    combination visits, including its operand-scale scan (combine_aligned),
    plus 10*u + 5 for its streamed upper dropped mass (upper_dropped_mass, at
    most 10 visits per dropped coefficient and five per node), so
    W_nodes = K*(2*m_L + 18*u + 9). Classical planning counts
    W_coeff = W_dec + W_align + W_nodes. Zero nodes add no node work. These
    units are proxies for the limit checks, not timings or equal-cost CPU operations.
    """
    s = m_l + m_h
    return (q + 4)*s + (q + 2)*union, nodes*(2*m_l + 18*union + 9)


def _node_key(k_value) -> str:
    """Return the node-cache key of one k-node, its exact binary64 value.

    A node-value cache is local to one decomposition, cutoff and bound
    policy. It is keyed by the exact binary64 k value, so -0.0 and 0.0 stay
    distinct, and never by a kept count, an unordered label set or a
    display string (a physical branch index is not a common node index
    across applications either).
    """
    return float(k_value).hex()


def _distinct_combined_nodes(
    decomposition: _TrotterPauliDecomposition,
    k_values,
) -> dict[str, _CombinedTrotterTerms]:
    """Combine and prune each distinct k-node once, keyed by _node_key in first-occurrence order.

    Every application of a node reuses this one combined record (identity
    coefficient, kept terms, pruned labels and upper pruned mass) at its own
    elapsed time. No pruning bound is recorded for the node alone.
    """
    nodes: dict[str, _CombinedTrotterTerms] = {}
    for k_value in k_values:
        key = _node_key(k_value)
        if key not in nodes:
            nodes[key] = _combined_trotter_terms(decomposition, k_value=float(k_value))
    return nodes


def _node_cache_bytes(nodes: int, union_count: int, num_qubits: int) -> int:
    """Return B_cache = 128*K_n*(u+1)*(q+1) + 8192*K_n for K_n cached combined nodes.

    u is the union-label count including identity and q the system width.
    The first term is the existing LCHS per-node label/value bound,
    counted once per distinct node. The second sets aside room for each
    node's scalar bound coefficient and metadata. A saved W_up has at most 5376 magnitude
    bits under the finite coefficient_up reductions, and its two integer
    components take less than 1536 bytes, within that scalar allowance. This
    is an explicit engineering allowance for the Python objects, not a heap bound.
    """
    return 128*nodes*(union_count+1)*(num_qubits+1) + 8192*nodes


@dataclass(frozen=True, kw_only=True)
class LCHSProductFormulaNodes:
    """The product-formula coefficient table of a Plan, saved once per exact k-node.

    Per-application elapsed times, step counts and routes are saved outside
    this table, in host_actions["applications"] for classical execution and in
    the SELECT plan for a quantum Plan. Angles are not saved. They do not
    determine the coefficients at zero elapsed time and do not
    recover the discarded coefficients or their certified mass, so a
    fixed-step Plan keeps this table even when no certificate was requested.

    The complex coefficients keep the imaginary-residue checks of the
    classical action and of the census. The identity coefficient and the upper mass are
    the saved binary64 values, never recomputed from pruned_indices, angles
    or rounded moduli.

    Attributes:
        num_qubits: System width q of every label.
        labels: Shared lexical union of the L and H labels, including the
            identity when present (_TrotterPauliDecomposition.aligned_rows).
        keys: Exact _node_key strings of the distinct nodes, in
            first-occurrence order of the quadrature grid.
        grid_to_node: Read-only int64 array, the node index of each original
            quadrature position.
        nodes: One read-only mapping per distinct node, with
            ``union_indices`` (read-only int64, the kept nonidentity rows in
            lexical order), ``coefficients`` (read-only complex128,
            the values combine_aligned returned, parallel to union_indices),
            ``identity_coefficient`` (binary64 generator coefficient c_I, so
            an application's identity phase is -elapsed*c_I),
            ``pruned_l1_mass`` (binary64 upper dropped mass d_up) and
            ``pruned_indices`` (read-only int64 diagnostic dropped rows,
            including the identity if it was dropped).
    """

    num_qubits: int
    labels: tuple[str, ...]
    keys: tuple[str, ...]
    grid_to_node: np.ndarray
    nodes: tuple[Mapping[str, Any], ...]


def _selected_node_table(decomposition, combined_nodes, k_values) -> LCHSProductFormulaNodes:
    """Return the node table of the combined nodes over the quadrature grid k_values.

    Each node's arrays are formed from the kept and dropped row positions
    that combine_aligned recorded, in their recorded order, and made
    read-only. The labels are the decomposition's aligned union. Every grid
    position must name a combined node. The kept Pauli coefficients of the
    Hermitian L and H are real up to decomposition rounding, and an
    imaginary residue above the node's own cutoff (combine_aligned) is
    refused here, once per node, so the rotations formed from the table at
    each application (host_pf._rotation_sequence) read their real parts
    directly.
    """
    labels = tuple(row[0] for row in decomposition.aligned_rows)
    grid_keys = tuple(_node_key(k) for k in k_values)
    keys = tuple(dict.fromkeys(grid_keys))
    position = {key: index for index, key in enumerate(keys)}

    def frozen(values, dtype):
        array = np.array(values, dtype=dtype)
        array.flags.writeable = False
        return array

    nodes = []
    for key in keys:
        combined = combined_nodes[key]
        coefficients = frozen([value for _, value in combined.traceless_terms], np.complex128)
        if np.any(np.abs(coefficients.imag) > combined.coefficient_cutoff):
            raise ValueError("Hamiltonian coefficients must be real for Pauli rotations")
        nodes.append(MappingProxyType(dict(
            union_indices=frozen(combined.kept_indices, np.int64),
            coefficients=coefficients,
            identity_coefficient=float(combined.identity_coefficient),
            pruned_l1_mass=float(combined.pruned_l1_mass),
            pruned_indices=frozen(combined.pruned_indices, np.int64),
        )))
    return LCHSProductFormulaNodes(
        num_qubits=decomposition.num_qubits, labels=labels, keys=keys,
        grid_to_node=frozen([position[key] for key in grid_keys], np.int64), nodes=tuple(nodes))


def _stored_node_table_bytes(table: LCHSProductFormulaNodes) -> int:
    """Return the logical size of a saved node table, counted once where it is held.

    q*u for the shared labels, B_nodes = 24*sum_k p_k + 128*K for the node
    coefficient and union-index rows and their scalar metadata, and the
    records 8*sum_k d_k for the diagnostic pruned indices and 8*N_grid for
    the grid-to-node mapping. Reading rows of the table in place copies nothing.
    """
    return _node_table_bytes(table.num_qubits, len(table.labels),
        [len(node["union_indices"]) for node in table.nodes],
        [len(node["pruned_indices"]) for node in table.nodes], len(table.grid_to_node))


def _node_table_bytes(q, union_count, kept_counts, dropped_counts, grid_count) -> int:
    """Return q*u + 24*sum_k p_k + 128*K + 8*sum_k d_k + 8*N_grid (_stored_node_table_bytes)."""
    return (q*union_count + 24*sum(kept_counts) + 128*len(kept_counts)
            + 8*sum(dropped_counts) + 8*grid_count)


def stored_node_read_work(kept_counts, *, pruned_counts=()):
    """Logical read visits for the distinct nodes of the saved table.

    W_read = sum_k (p_k + 2) over the nodes actually read in one invocation,
    with p_k the node's kept nonidentity rows. One visit to a row read in
    place includes associating its index with its label. Two scalar visits
    read c_I and d_up per node. Optional diagnostic rows cost one visit each.
    Conversion, validation, copying and later arithmetic are counted
    separately. These units are proxies for the limit checks, not timings or equal-cost
    CPU operations.
    """
    return sum(p + 2 for p in kept_counts) + sum(pruned_counts)


def borrowed_node_rows(shared_labels, node):
    """Yield the (label, coefficient) pairs of one saved node without copying."""
    for index, coefficient in zip(
        node["union_indices"], node["coefficients"], strict=True
    ):
        yield shared_labels[index], coefficient


def borrowed_node_scalars(node):
    """Return the saved generator identity coefficient and upper dropped mass."""
    return node["identity_coefficient"], node["pruned_l1_mass"]


class _BorrowedRows:
    """A repeatable view, with a length, of one saved node's (label, coefficient) rows.

    Each iteration reads the arrays in memory again (borrowed_node_rows),
    so a caller that checks emptiness and traverses its input more than
    once, such as host_pf._rotation_sequence, makes no copy.
    """

    __slots__ = ("labels", "node")

    def __init__(self, labels, node):
        self.labels, self.node = labels, node

    def __len__(self):
        return len(self.node["union_indices"])

    def __iter__(self):
        return borrowed_node_rows(self.labels, self.node)


def stored_pf_nodes(data) -> LCHSProductFormulaNodes:
    """Return the saved node table of a product-formula LCHSData, for classical actions or SELECT."""
    table = (data.host_actions["nodes"] if data.host_actions is not None
             else None if data.select_data is None else data.select_data.nodes)
    if type(table) is not LCHSProductFormulaNodes:
        raise ValueError("this LCHS Plan has no stored product-formula node table")
    return table


def _union_nonidentity_labels(decomposition: _TrotterPauliDecomposition) -> tuple[str, ...]:
    """Return the sorted union of the nonidentity labels of L and H."""
    return tuple(sorted(
        label for label in set(decomposition.l_coefficients) | set(decomposition.h_coefficients)
        if label != decomposition.identity_label
    ))


def _node_bound_coefficients(
    labels: Sequence[str],
    num_qubits: int,
    node_terms: Mapping[str, Any],
    keys: Sequence[str],
    *,
    order: int,
    max_work: int | None,
    max_bytes: int | None,
    census_held: int,
    limit_owner: str = "LCHS",
    work_field: str = "max_select_work",
    field_work_offset: int = 0,
    work_used: int = 0,
    linear_and_other_work: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return each listed node's Pauli-triangle W_up from one shared structure that passed the limit checks.

    The shared basis ``labels`` is the sorted union of the nonidentity
    labels of L and H (_union_nonidentity_labels), and node_terms maps each
    key to its kept (label, coefficient) rows in that order: a combined
    node's traceless_terms, or the rows of a saved node table read in place
    (_BorrowedRows). The K listed nodes form a K x p magnitude matrix over
    the union, with exact zeros at the labels a node omits.
    error_budget.coefficient_up contracts every row against one ordered pair
    structure. A zero magnitude removes its label's contributions exactly,
    so each row's exact coefficient is that of a separate census on the
    node's kept terms in their order with the same expression (CSTWZ
    doi:10.1103/PhysRevX.11.011020, Eqs. (120)-(121)). Its outward W_up can
    differ from a separate census in the last bits, because the rounding
    factors and reduction trees include the zero entries.

    Selection. The pair scan of the union counts N(A) = sum_i e_i (p-i-1)
    and chooses the expression once for all K nodes from N(A)*max(w, K),
    w = ceil(q/64) (error_budget.census_structure with weightings=K):
    "exact_census" at most error_budget.EXACT_CENSUS_WORD_TESTS,
    "relaxed_prefix" above it. The nested masks of the exact expression
    are formed once per tile of pairs and reused by the K node
    contractions, which make K*N(A) weight products beside the w*N(A)
    word tests, so the count is the larger of the two. All K nodes take the
    same expression. No node is contracted to choose, and the limits never
    choose the expression.

    Limits. Before the scan, the label packing p*q and the w*p(p-1)/2 pair
    tests, with the caller's prior and other work, are checked against
    ``max_work``, and the masks with ``census_held`` against ``max_bytes``
    (None: the physical memory). After the scan and before any nested test,
    the work G + K*V of the chosen expression (error_budget.census_work with
    nodes=K) and its error_budget.census_bytes with nodes=K plus
    ``census_held`` are checked. ``census_held`` counts the other LCHS
    arrays held together with the census, never its own buffers. A refusal
    names ``limit_owner(work_field=...)`` or ``limit_owner(max_bytes=...)``
    and the value it needs (LCHS(max_select_work=...), the default, for
    SELECT selection and the classical actions of host_pf,
    LCHSRefinement(max_structural_work=...) for fixed-step refinement).
    field_work_offset is the work of that field counted before ``max_work``
    was handed here, so each stated requirement is a value of the field.

    Work: p*q label packing plus G + K*V, returned as ledger["work"]. The
    caller adds its own preparation and application work. The first listed
    node's evaluation carries the structure's pair and nested tests, the
    others zero. These units are proxies for the limit checks, not timings
    or equal-cost CPU operations.

    Returns:
        ({key: _BoundCoefficientEvaluation}, ledger) with ledger keys
        ``work``, ``bytes``, ``variant``, ``pair_checks``,
        ``nested_checks``, ``word_tests``, ``selection_count``,
        ``structures`` and ``contractions``. ``word_tests`` is the w*N(A)
        logical nested word tests, ``selection_count`` the N(A)*max(w, K)
        that chose the expression, and ``nested_checks`` the nested tests
        made, zero for "relaxed_prefix".
    """
    from nwqlib._limits import exceeds, memory_limit
    from nwqlib.subroutines.trotterization import error_budget as eb

    keys = tuple(keys)
    ledger = dict(work=0, bytes=0, variant=None, pair_checks=0, nested_checks=0, word_tests=0,
                  selection_count=0, structures=0, contractions=0)
    if not keys:
        return {}, ledger
    labels = tuple(labels)
    p, q, nodes = len(labels), num_qubits, len(keys)
    words = (q + 63) // 64
    prior = work_used + linear_and_other_work + p * q
    limit = memory_limit(max_bytes)

    def admit(stage, work, size):
        reasons = []
        if exceeds(work, max_work):
            reasons.append(f"{limit_owner}({work_field}=...) to at least {work + field_work_offset} "
                           f"(now {max_work + field_work_offset})")
        if exceeds(size, limit):
            reasons.append(f"{limit_owner}(max_bytes=...) to at least {size} (now {max_bytes})"
                           if max_bytes is not None else
                           f"memory: it needs {size} bytes, above the physical memory ({limit} bytes)")
        if reasons:
            raise ResourceLimitError(
                f"LCHS census {stage} exceeds its allowance. Raise " + " and ".join(reasons)
                + f". The bytes include {census_held} bytes held by LCHS. These figures admit the "
                "census only. Later LCHS stages are admitted separately")

    admit("pair scan", prior + words * (p * (p - 1) // 2),
          census_held + eb.census_bytes(p, q, 0, order=1, variant="exact_census"))
    census = eb.census_structure(labels, q, order, weightings=nodes, max_bytes=max_bytes)
    exact2 = order == 2 and census.variant == "exact_census"
    need_work = prior + eb.census_work(p, q, order=order, variant=census.variant, pairs=census.pairs,
                                       nested=census.nested_tests if exact2 else 0, nodes=nodes)
    need_bytes = census_held + eb.census_bytes(p, q, census.pairs, order=order, variant=census.variant,
                                               nodes=nodes, tile=census.nested_tile)
    admit(f"({census.variant}, {nodes} nodes)", need_work, need_bytes)
    index = {label: position for position, label in enumerate(labels)}
    magnitudes = np.zeros((nodes, p), dtype=np.float64)
    for row, key in zip(magnitudes, keys, strict=True):
        for label, coefficient in node_terms[key]:
            value = complex(coefficient)
            # Same absolute imaginary-residue admission as the census's
            # operator validation (error_budget._validated_terms), not the
            # node's relative cutoff (see _PAULI_COEFFICIENT_ATOL).
            if abs(value.imag) > _PAULI_COEFFICIENT_ATOL:
                raise ValueError(
                    "Hamiltonian coefficients must be real for Trotter error bounds; "
                    f"term {label!r} has coefficient {value} (build the operator from "
                    "real coefficients or simplify away numerical imaginary parts)"
                )
            row[index[label]] = abs(value.real)
    coefficients = eb.coefficient_up(magnitudes, census.rows, order=order, variant=census.variant,
                                     x=census.x, z=census.z)
    del magnitudes
    nested_checks = census.nested_tests if exact2 else 0
    evaluations = {
        key: eb._BoundCoefficientEvaluation(
            coefficient=coefficient,
            pauli_term_count=len(node_terms[key]),
            pair_commutation_checks=census.pair_tests if first else 0,
            nested_commutation_checks=nested_checks if first else 0,
            bound_variant=census.variant,
            nested_word_tests=census.word_tests,
        )
        for first, (key, coefficient) in zip((True,) + (False,) * (nodes - 1),
                                             zip(keys, coefficients, strict=True), strict=True)
    }
    ledger.update(work=need_work - work_used - linear_and_other_work, bytes=need_bytes,
                  variant=census.variant, pair_checks=census.pair_tests, nested_checks=nested_checks,
                  word_tests=census.word_tests, selection_count=census.selection_count, structures=1,
                  contractions=nodes)
    return evaluations, ledger


def _first_use(evaluations: Mapping[str, Any], key: str, used: set[str]):
    """Return a node's evaluation with its checks on first use and zero checks on reuse.

    A scalar selection from an existing W_up performs no new commutation
    test, so only the first application of a node reports the counts.
    """
    evaluation = evaluations.get(key)
    if evaluation is None or key not in used:
        used.add(key)
        return evaluation
    return replace(evaluation, pair_commutation_checks=0, nested_commutation_checks=0)


def _selection_with_remaining_allowance(evaluation, *, time, remaining, order):
    """Select steps for a node coefficient that already passed the limit checks, against the remainder R.

    At R = 0 only an exactly zero formula numerator (W_up = 0 or t = 0) is
    accepted, with the existing one-step convention for a zero theorem term
    of a nonempty generator. Otherwise PruningBudgetExhausted is raised. A
    positive R uses the exact step inversion of _selection_from_evaluation.
    No commutator scan runs here.
    """
    from nwqlib.subroutines.trotterization import error_budget as eb

    if remaining == 0.0:
        if time != 0.0 and evaluation.coefficient != 0:
            raise eb.PruningBudgetExhausted
        return eb.TrotterStepSelection(
            formula_order=order,
            evolution_time=float(time),
            error_budget=0.0,
            bound_value=0.0,
            step_count=1,
            pauli_term_count=evaluation.pauli_term_count,
            pair_commutation_checks=evaluation.pair_commutation_checks,
            nested_commutation_checks=evaluation.nested_commutation_checks,
            bound_variant=evaluation.bound_variant,
        )
    return eb._selection_from_evaluation(evaluation, time=time, error_budget=remaining, order=order)


def _budgeted_trotter_node_record(
    combined: _CombinedTrotterTerms,
    evaluation,
    *,
    final_time: float,
    error_budget: float,
    order: int,
) -> _LCHSTrotterNodeRecord:
    """Select one application of a node against the PF budget left after its reported pruning bound.

    Dropping Pauli terms whose moduli sum to at most d_up
    (combined.pruned_l1_mass) changes the node evolution exp(-i*t*(k*L+H))
    by at most |t|*d_up in operator norm: Duhamel's formula gives
    ||exp(-i*t*X) - exp(-i*t*Y)|| <= |t|*||X - Y|| for Hermitian X and Y,
    and ||X - Y|| <= d_up because every Pauli string has unit norm. The
    reported pruning bound is P = up(|t|*d_up)
    (error_budget.upper_pruning_error). If P exceeds the allowance epsilon
    the node is refused. Otherwise R = down(epsilon - P)
    (error_budget._remaining_budget_after_pruning), the node's cached W_up
    selects the smallest step count whose bound E_r = W_up*|t|**(o+1)/r**o
    is at most R, E = up(E_r) <= R, and the reported total
    T = up(P + E) (error_budget.upper_combined_error) satisfies
    ``|t|*d_up + E_r <= P + E <= T <= epsilon``. The order argument, the Method's
    trotter_order, selects the Lie-1 bound (CSTWZ
    doi:10.1103/PhysRevX.11.011020, Prop. 9, Eq. (120)) or the Suzuki-2
    bound (Prop. 10, Eq. (121)).

    Args:
        combined: The node's shared combined terms (_distinct_combined_nodes).
        evaluation: The node's W_up from the census (_node_bound_coefficients), with
            zero checks on reuse (_first_use). None when no term is kept.
        final_time: This application's elapsed time.
        error_budget: The per-node synthesis allowance epsilon.
        order: Product-formula order.

    Returns:
        The node record. Its combined_bound_value is T, and an empty kept
        generator has zero steps and T = P.
    """
    from nwqlib.subroutines.trotterization.error_budget import (
        PruningBudgetExhausted,
        _remaining_budget_after_pruning,
        upper_combined_error,
        upper_pruning_error,
    )

    budget = float(error_budget)
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError(f"LCHS per-node error budget must be positive; got {error_budget!r}")
    prune_cost = upper_pruning_error(final_time, combined.pruned_l1_mass)
    if prune_cost > budget:
        raise ValueError(
            "LCHS pruned l1 mass exceeds the per-node synthesis budget: "
            f"published pruning bound={prune_cost!r}, pruned_l1={combined.pruned_l1_mass!r}, "
            f"budget={budget!r}"
        )
    if not combined.traceless_terms:
        return _LCHSTrotterNodeRecord(
            pf_bound_value=0.0,
            pruned_l1_mass=combined.pruned_l1_mass,
            combined_bound_value=upper_combined_error(prune_cost, 0.0),
            step_count=0,
            selection=None,
            bound_value_status="structural_upper_bound",
        )
    if evaluation is None:
        raise ValueError("a kept LCHS product-formula node needs its admitted bound coefficient")
    remaining = _remaining_budget_after_pruning(total_budget=budget, pruning_error=prune_cost)
    try:
        selection = _selection_with_remaining_allowance(
            evaluation, time=final_time, remaining=remaining, order=order)
    except PruningBudgetExhausted:
        raise ValueError(
            "LCHS pruned l1 mass consumes the entire per-node synthesis "
            f"budget ({budget!r}) while the kept PF theorem term is nonzero"
        ) from None
    return _LCHSTrotterNodeRecord(
        pf_bound_value=float(selection.bound_value),
        pruned_l1_mass=combined.pruned_l1_mass,
        combined_bound_value=upper_combined_error(prune_cost, selection.bound_value),
        step_count=int(selection.step_count),
        selection=selection,
        bound_value_status=selection.bound_value_status,
        bound_variant=selection.bound_variant,
    )


def _fixed_trotter_certificate_records(
    *,
    nodes: LCHSProductFormulaNodes,
    applications: Sequence[tuple[float, Sequence[int]]],
    method: LCHS,
    dense_validation: bool = False,
    max_bytes=None,
    max_dense_work=None,
    max_structural_work=None,
    max_steps=None,
    counts=None,
    held_bytes: int = 0,
) -> tuple[list[_LCHSTrotterNodeRecord], ...]:
    """Explicitly bound already fixed node schedules of several applications, without step selection.

    CSTWZ, Phys.Rev.X11,011020 (2021), doi:10.1103/PhysRevX.11.011020,
    Props.9-10, Eqs.(120)-(121):
    W_up*t**(p+1)/r**p bounds r fixed order-p steps. Each application is an
    (elapsed_time, step_counts) pair with one saved step count per
    quadrature position. W depends on the ordered kept coefficients, order
    and bound variant but not on t, r or the application, so every distinct
    k-node of the saved node table (LCHSProductFormulaNodes) is bounded
    once (_node_bound_coefficients) and each application evaluates it at its
    own (elapsed_time, step_count). Pauli commutator triangle sums bound the
    spectral relation. Optional dense evaluation checks those sums, not a
    newly chosen circuit, and its dense coefficient is also formed once per
    node.

    Fixed-step refinement evaluates the saved step counts using the saved
    coefficients and pruning mass. The coefficient table is read once per
    distinct k-node. Reading a row in place costs one logical visit, and
    reading the identity coefficient and upper dropped mass
    costs two visits per node, W_read = sum_k (p_k + 2)
    (stored_node_read_work). There is no alignment, combination, pruning or
    upper-mass computation in a read. The table is counted in the memory
    held by the phase that reads it (_stored_node_table_bytes). Nothing is copied.

    Each record reports P = up(|t|*d_up), the formula bound
    E = up(W_up*|t|**(p+1)/r**p) and T = up(P + E)
    (error_budget.upper_pruning_error, upper_combined_error). The saved
    steps are kept, so T may exceed a requested allowance.

    Structural work, with p_k a node's kept count, K distinct nodes and M
    applications: W_read + p*q + G + K*V + sum_nodes(16*p_k*(q+1))
    + 16*M*K, with G and V of error_budget.census_work. The 16*p_k*(q+1) term is this function's conservative
    preparation allowance for the census magnitudes and labels, once per
    node. It is not a decomposition or a second read of the saved rows.
    Bytes: the data already held and the saved table, plus the census
    (_node_bound_coefficients). No decomposition is counted here. These units
    are proxies for the limit checks, not timings or equal-cost CPU operations. All
    node work accumulates in the supplied counters, and first evaluations
    carry the census tests.

    max_structural_work governs this explicitly requested structural
    refinement independently of max_dense_work. Each limit set to None
    sets no limit, and max_bytes=None uses the physical memory.

    Returns:
        One record list per application, in quadrature-position order.
    """
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.subroutines.trotterization import error_budget as trotter_error_budget

    from nwqlib._linalg_laws import singular_values_work
    from nwqlib.operators.access import _check_bytes
    max_dense_work = optional_limit(max_dense_work, "max_dense_work")
    max_structural_work = optional_limit(max_structural_work, "max_structural_work")
    max_steps = optional_limit(max_steps, "max_steps")
    counts = {} if counts is None else counts
    for name in ("structural_work", "stored_node_read_work", "dense_work", "known_peak_bytes",
                 "structural_bound_attempts", "structural_bound_completed", "pair_commutation_checks",
                 "nested_commutation_checks", "dense_bound_attempts", "dense_bound_completed",
                 "dense_term_matrices", "dense_matrix_products", "dense_spectral_norms"):
        counts.setdefault(name, 0)
    order = int(method.trotter_order)
    if type(nodes) is not LCHSProductFormulaNodes:
        raise ValueError("fixed certificates read an existing stored node table")
    grid = nodes.grid_to_node
    schedules = []
    for elapsed_time, step_counts in applications:
        elapsed_time = validate_final_time(elapsed_time)
        if len(step_counts) != len(grid):
            raise ValueError("fixed certificate schedule must match the k-node count")
        steps = tuple(integer(step, "fixed certificate step", 0) for step in step_counts)
        if any(exceeds(step, max_steps) for step in steps):
            raise ValueError("fixed certificate steps exceed max_steps")
        schedules.append((elapsed_time, steps))
    q = nodes.num_qubits
    keys = tuple(nodes.keys[int(index)] for index in grid)
    table = dict(zip(nodes.keys, nodes.nodes, strict=True))
    # Work: one acquisition of each distinct node's stored rows and scalars.
    read_work = stored_node_read_work(len(node["union_indices"]) for node in nodes.nodes)
    if exceeds(counts["structural_work"]+read_work, max_structural_work):
        raise ValueError("fixed certificate exceeds max_structural_work before reading its node table")
    # Bytes: the borrowed table stays in the held payload of this phase.
    held = held_bytes+_stored_node_table_bytes(nodes)
    _check_bytes(held, max_bytes, "fixed PF stored node table")
    counts["structural_work"] += read_work
    counts["stored_node_read_work"] += read_work
    shared = tuple(label for label in nodes.labels if set(label) != {"I"})
    rows = {key: _BorrowedRows(nodes.labels, node) for key, node in table.items()}
    for elapsed_time, steps in schedules:
        if elapsed_time and any(len(rows[key]) and step == 0 for key, step in zip(keys, steps, strict=True)):
            raise ValueError("nonzero fixed certificate nodes require a positive step count")
    census_keys = tuple(key for key in nodes.keys if len(rows[key])
                        and any(elapsed_time for elapsed_time, _ in schedules))
    label_work = sum(16*len(rows[key])*(q+1) for key in census_keys)
    scalar_work = 16*len(schedules)*len(nodes.keys)
    if exceeds(counts["structural_work"]+label_work+scalar_work, max_structural_work):
        raise ValueError("fixed certificate exceeds max_structural_work before Pauli commutators")
    evaluations, ledger = _node_bound_coefficients(
        shared, q, rows, census_keys, order=order,
        max_work=max_structural_work, max_bytes=max_bytes, census_held=held,
        limit_owner="LCHSRefinement", work_field="max_structural_work",
        work_used=counts["structural_work"], linear_and_other_work=label_work+scalar_work)
    counts["structural_work"] += label_work+scalar_work+ledger["work"]
    counts["known_peak_bytes"] = max(counts["known_peak_bytes"], ledger["bytes"], held)
    counts["structural_bound_attempts"] += len(census_keys)
    counts["structural_bound_completed"] += len(census_keys)
    counts["pair_commutation_checks"] += ledger["pair_checks"]
    counts["nested_commutation_checks"] += ledger["nested_checks"]
    # Dense commutator evaluation is an explicit bounded reference path.
    # It checks structural evidence without choosing a new circuit schedule.
    # Its coefficient is independent of time and steps, so it is formed once
    # per distinct node and its matrices are released before the next node.
    dense_coefficients = {}
    if dense_validation:
        d = 1 << q
        for key in census_keys:
            p = len(rows[key])
            # Bytes: 12 complex128 d-by-d arrays, whatever p. The dense pass
            # forms one term matrix at a time, beside the suffix sum, the
            # commutator products of that term and the copy that its spectral
            # norm factors (error_budget._tail_commutator_norms). Its traced
            # NumPy peak was 5.7 such arrays for q = 7 and 10 or 40 terms.
            peak = 192*d*d
            # Work per term: 2 matrix products and 1 spectral norm (order 1) or
            # 6 products and 2 norms (order 2), d**3 per product, plus 24*d**2
            # entrywise operations. Each norm is np.linalg.norm(ord=2) in
            # trotterization/error_budget.py::_spectral_norm, which computes
            # singular values alone (_linalg_laws.singular_values_work). The
            # counters below record the same populations.
            products = 2 if order == 1 else 6
            work = products*p*d**3+order*p*singular_values_work(d)+24*p*d*d
            _check_bytes(held+peak, max_bytes, "fixed PF dense commutator arrays")
            if exceeds(counts["dense_work"]+work, max_dense_work):
                raise ValueError("fixed certificate exceeds max_dense_work before dense commutators")
            counts["dense_work"] += work
            counts["known_peak_bytes"] = max(counts["known_peak_bytes"], held+peak)
            counts["dense_bound_attempts"] += 1
            dense_coefficients[key] = trotter_error_budget.dense_trotter_bound_coefficient(
                SparsePauliOp.from_list(list(rows[key])), order=order)
            counts["dense_bound_completed"] += 1
            counts["dense_term_matrices"] += p
            counts["dense_matrix_products"] += products*p
            counts["dense_spectral_norms"] += order*p
    used: set[str] = set()
    results = []
    for elapsed_time, steps in schedules:
        records: list[_LCHSTrotterNodeRecord] = []
        for key, step_count in zip(keys, steps, strict=True):
            _, mass = borrowed_node_scalars(table[key])
            prune_cost = trotter_error_budget.upper_pruning_error(elapsed_time, mass)
            if not len(rows[key]) or elapsed_time == 0:
                total = trotter_error_budget.upper_combined_error(prune_cost, 0.0)
                records.append(
                    _LCHSTrotterNodeRecord(
                        pf_bound_value=0.0,
                        pruned_l1_mass=mass,
                        combined_bound_value=total,
                        step_count=int(step_count),
                        selection=None,
                        bound_value_status="structural_upper_bound",
                        dense_bound_value=total if dense_validation else None,
                    )
                )
                continue
            if step_count < 1:
                raise ValueError("nonzero fixed certificate nodes require a positive step count")
            evaluation = _first_use(evaluations, key, used)
            # W t**(order + 1)/r**order evaluated exactly and rounded upward, so a
            # nonzero bound is positive (trotterization.error_budget._finite_bound).
            structural = trotter_error_budget._finite_bound(
                evaluation.coefficient, elapsed_time, order, int(step_count))
            dense = None
            if dense_validation:
                dense = trotter_error_budget._finite_bound(
                    dense_coefficients[key], elapsed_time, order, int(step_count))
                # The Pauli-triangle sum is an upper bound on the dense commutator
                # norms. A 64-ulp window relative to max(1, structural, dense)
                # keeps binary64 rounding in either evaluation from reading as a
                # violation of that relation.
                comparison_scale = max(1.0, structural, dense)
                if structural + 64.0 * np.finfo(float).eps * comparison_scale < dense:
                    raise RuntimeError("structural Trotter bound does not upper-bound dense validation")
            records.append(
                _LCHSTrotterNodeRecord(
                    pf_bound_value=structural,
                    pruned_l1_mass=mass,
                    combined_bound_value=trotter_error_budget.upper_combined_error(prune_cost, structural),
                    step_count=int(step_count),
                    selection=None,
                    bound_value_status="structural_upper_bound",
                    dense_bound_value=(None if dense is None
                                       else trotter_error_budget.upper_combined_error(prune_cost, dense)),
                    pauli_term_count=evaluation.pauli_term_count,
                    pair_commutation_checks=evaluation.pair_commutation_checks,
                    nested_commutation_checks=evaluation.nested_commutation_checks,
                    bound_variant=evaluation.bound_variant,
                )
            )
        results.append(records)
    return tuple(results)


def _lchs_trotter_error_budget_per_node(method: LCHS) -> float:
    """Return the per-node synthesis allowance, ``LCHS_TROTTER_EPSILON_FRACTION * method.approximation_tolerance``.

    It bounds one node evolution in operator norm. The weighted sum over
    nodes and the physical scaling are applied later, so it is not a bound
    on the solution error.
    """
    return LCHS_TROTTER_EPSILON_FRACTION * float(method.approximation_tolerance)


def _unusable_kernel_stage_names(quadrature: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the kernel stages whose recorded bound cannot be used.

    Outside the exact L = 0 reduction, the tail and quadrature bounds are
    positive analytic quantities. A recorded None (no bound for this pair) or
    an exact zero (only possible by rounding) must therefore not enter the
    physical propagation as zero error, so the stage is reported unusable.
    """

    if quadrature.get('solution_error_certificate_status') == 'exact_algebraic':
        return ()
    owners = (
        ("kernel_approximation", "approximate_lchs_error_bound"),
        ("k_quadrature", "quadrature_error_bound"),
    )
    return tuple(
        stage
        for stage, raw_key in owners
        if raw_key in quadrature
        and (quadrature[raw_key] is None or float(quadrature[raw_key]) == 0.0)
    )


def _coefficient_weighted_bound(coefficients: np.ndarray, node_bounds):
    """Return sum_j |c_j|*e_j as a PhysicalScale, over the raw coefficients (_trotter_budget_quadrature_terms)."""
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery
    magnitudes, lift = lifted_magnitudes(coefficients)
    unlift = PhysicalScale(mantissa=0.5, exponent=1 - lift)
    return scaled_sum([
        compose_recovery(float(magnitude), unlift, bound)
        for magnitude, bound in zip(magnitudes, node_bounds, strict=True)
        if magnitude > 0 and bound > 0
    ])


def _trotter_pruning_bound(dropped_masses, *, coefficients: np.ndarray, elapsed_time: float):
    """Return sum_j |c_j|*|t|*d_j, one application's operator bound on its deleted Pauli terms.

    Deleting a term set of coefficient mass d_j from a node generator G_j
    changes its evolution by ||exp(-i*t*G_j) - exp(-i*t*(G_j - D_j))|| <= |t|*d_j
    (Duhamel, with ||D_j|| <= d_j), and the identity counts because it can be
    a relative phase between LCU branches. The triangle inequality over the
    LCU sum then gives this value. solution_error_budget.pruning_stage applies
    the application's |w|*||v||*exp(shift*t) once, in place of any gamma, and t
    is the application's own elapsed time, not always T. It bounds pruning
    alone: a fixed step count leaves the product-formula error unknown, so
    this is not a bound on the total error.
    """
    from nwqlib.subroutines.trotterization.error_budget import upper_pruning_error
    return _coefficient_weighted_bound(coefficients, [upper_pruning_error(elapsed_time, mass) for mass in dropped_masses])


def _trotter_budget_quadrature_terms(
    selections: list[_LCHSTrotterNodeRecord | TrotterStepSelection],
    *,
    coefficients: np.ndarray,
    method: LCHS,
) -> dict[str, Any]:
    """Recorded budget terms for the error-budgeted backend's selections.

    This is the one place where the recorded schedule and bound terms are
    computed. The
    SELECT plan, the classical Pauli actions and explicit refinement pass
    their per-node records here, so equal node records give equal values.
    The synthesis bound sum_j |c_j|*e_j is the triangle inequality over the
    LCU sum, with e_j the node's PF plus pruning bound and c_j the supplied
    coefficients, the raw stored ones for the solution bound. It is an
    operator-norm bound on the ideal finite sum before physical input and
    PSD recovery weighting, which solution_error_budget applies once. It is
    recorded as a PhysicalScale: each term composes the lifted magnitude
    2**k*|c_j| of ``providers.lifted_magnitudes``, the inverse lift 2**-k
    and e_j, and ``scaled_sum`` adds the terms, so a positive term below
    the binary64 range stays positive instead of rounding to zero, and the
    lifted one-norm never stands in for the raw one. Readers compose it
    with the physical factors before any binary64 conversion.

    trotter_bound_variants lists every Pauli-triangle expression the records
    used. trotter_algebraic_work_counts sums the records' newly performed
    tests: a node's shared census tests are carried by its first record
    only, so reused coefficients are not counted again.
    """
    from nwqlib.subroutines.trotterization import (
        TROTTER_BOUND_REFERENCE,
    )

    records = [
        selection
        if isinstance(selection, _LCHSTrotterNodeRecord)
        else _LCHSTrotterNodeRecord(
            pf_bound_value=float(selection.bound_value),
            pruned_l1_mass=0.0,
            combined_bound_value=float(selection.bound_value),
            step_count=int(selection.step_count),
            selection=selection,
            bound_value_status=selection.bound_value_status,
            bound_variant=selection.bound_variant,
        )
        for selection in selections
    ]
    bound_values = [record.combined_bound_value for record in records]
    evaluated = all(bound is not None for bound in bound_values)
    dense_bound_values = [record.dense_bound_value for record in records]
    dense_evaluated = all(bound is not None for bound in dense_bound_values)
    trotter_synthesis_error_bound = _coefficient_weighted_bound(coefficients, bound_values) if evaluated else None
    work_counts = [record.algebraic_work_counts for record in records]
    terms = {
        "trotter_formula_order": int(method.trotter_order),
        "trotter_step_counts": [record.step_count for record in records],
        "trotter_bound_value_max": (
            float(max(bound for bound in bound_values if bound is not None))
            if evaluated
            else None
        ),
        "trotter_synthesis_error_bound": trotter_synthesis_error_bound,
        "trotter_bound_reference": TROTTER_BOUND_REFERENCE,
        "trotter_bound_method": "pauli_triangle" if evaluated else None,
        "trotter_bound_variants": sorted({record.bound_variant for record in records
                                          if record.bound_variant is not None}),
        "trotter_bound_status": (
            "structural_upper_bound" if evaluated else "not_evaluated"
        ),
        "trotter_dense_bound_value_max": (
            float(max(bound for bound in dense_bound_values if bound is not None))
            if dense_evaluated
            else None
        ),
        "trotter_dense_bound_method": "dense" if dense_evaluated else None,
        "trotter_dense_bound_status": (
            "dense_validation_value" if dense_evaluated else "not_evaluated"
        ),
        "trotter_structural_vs_dense_validation_status": (
            "structural_upper_bound_confirmed" if dense_evaluated else "not_evaluated"
        ),
        "trotter_algebraic_work_counts": {
            key: sum(counts[key] for counts in work_counts)
            for key in (
                "pauli_terms",
                "pair_commutation_checks",
                "nested_commutation_checks",
            )
        },
        "trotter_node_records": [record.to_dict() for record in records],
    }
    if method.hamiltonian_evolution_backend == "trotter_error_budgeted":
        terms["trotter_error_budget_per_node"] = _lchs_trotter_error_budget_per_node(method)
    return terms


@dataclass(frozen=True, kw_only=True)
class _PreparedDecomposition:
    """Cartesian parts after padding and the PSD decision.

    Attributes:
        l_part: Encoded L, shifted by psd_shift*I when a shift was applied.
        h_evolution: Encoded H, unchanged by the shift.
        conversion: Smallest eigenvalue of L before and after the shift,
            whether a shift was applied, the shift, make_l_psd and
            psd_tolerance, as later reported.
        lambda_min_before: Smallest eigenvalue of the unshifted encoded L.
        l_norm: Spectral norm of the prepared L, max(|lambda_min|, |lambda_max|)
            after the shift.
        numerical_psd_premise_satisfied: Result of the numerical PSD rule.
    """

    l_part: np.ndarray
    h_evolution: np.ndarray
    conversion: Mapping[str, Any]
    lambda_min_before: float
    l_norm: float
    numerical_psd_premise_satisfied: bool


def _prepare_decomposition(
    *,
    matrix: np.ndarray,
    method: LCHS,
    padded_dimension: int | None = None,
    max_spectral_work: int | None = None,
    parts: bool = True,
) -> _PreparedDecomposition:
    """Return an LCHS-ready matrix decomposition and PSD conversion metadata.

    With ``parts=False`` only the coefficient endpoints are needed: L is
    formed without H, a padded or a shifted array, and l_part and
    h_evolution are None. The endpoints, padding zero and PSD rule are the
    same.

    The endpoints of L (_hermitian_endpoints) come in one of three ways. A
    zero or diagonal L gives them exactly from its diagonal, with no
    eigensolve. A real A gives a real symmetric L, solved by the real
    symmetric eigensolver. Any other L takes one complex Hermitian
    eigensolve. The numerical PSD rule
    (solution_error_budget._numerical_psd_decision) either accepts L, or
    shifts it by s = -lambda_min + window, or raises when make_l_psd is False.
    Because L is Hermitian, max(|lambda_min|, |lambda_max|) after the shift is
    its spectral norm up to eigensolver rounding, and it is the ||L|| used to
    choose the quadrature. Callers restore the physical growth exp(s*T).
    """

    # matrix is the admitted dense input: ingest_dense checked it square and
    # finite, froze it and kept a real matrix in float64.
    real = not np.iscomplexobj(matrix)
    if parts:
        l_part, h_evolution = _cartesian_parts(np.asarray(matrix, dtype=complex))
    elif real:
        l_part = np.add(matrix, matrix.T)
        l_part *= 0.5
        h_evolution = None
    else:
        adjoint = np.conjugate(matrix.T, order="C")
        l_part = np.add(matrix, adjoint)
        del adjoint
        l_part *= 0.5
        h_evolution = None
    min_l_eigenvalue, max_l_eigenvalue = _hermitian_endpoints(l_part, real=real,
                                                              max_spectral_work=max_spectral_work)
    dimension = matrix.shape[0]
    target_dimension = dimension if padded_dimension is None else integer(
        padded_dimension, "padded_dimension", dimension)
    if not parts:
        l_part = None
    if target_dimension > dimension:
        # A_pad=A⊕0 implies L_pad=L⊕0 and H_pad=H⊕0 (ACL arXiv:2312.03916v2,
        # Eq. (4)).
        # Add the known zero eigenvalue to the original endpoints. No second
        # eigensolve of the padded matrix runs. The PSD shift then acts on every
        # encoded coordinate, including the dummy block.
        min_l_eigenvalue = min(min_l_eigenvalue, 0.0)
        max_l_eigenvalue = max(max_l_eigenvalue, 0.0)
    if parts and target_dimension > dimension:
        # Pad the two parts sequentially so each old array dies before the
        # next allocation: 48 d**2 + 16 D**2, then 32 d**2 + 32 D**2 bytes,
        # each at most 64 D**2.
        padded = np.zeros((target_dimension, target_dimension), dtype=complex)
        padded[:dimension, :dimension] = l_part
        l_part = padded
        padded = np.zeros((target_dimension, target_dimension), dtype=complex)
        padded[:dimension, :dimension] = h_evolution
        h_evolution = padded
        del padded
    decision = _numerical_psd_decision(
        lambda_min=min_l_eigenvalue,
        lambda_max=max_l_eigenvalue,
        psd_tolerance=float(method.psd_tolerance),
        make_l_psd=method.make_l_psd,
    )
    psd_shift = decision.shift
    if parts:
        if psd_shift > 0.0:
            # The shift touches only the diagonal, with the same finite entries
            # as L + shift*I. Untouched off-diagonal signed zeros are not
            # rewritten to +0.0 as the full identity sum would.
            diagonal = np.diag_indices(target_dimension)
            l_part[diagonal] += psd_shift
        l_part = _adopt_read_only(l_part)
        h_evolution = _adopt_read_only(h_evolution)

    min_l_eigenvalue_after = float(min_l_eigenvalue + psd_shift)
    max_l_eigenvalue_after = float(max_l_eigenvalue + psd_shift)
    l_norm = max(abs(min_l_eigenvalue_after), abs(max_l_eigenvalue_after))
    # The numerical PSD rule only adds a nonnegative shift. It does not take
    # a positive smallest eigenvalue out as known scalar decay, so the dense
    # recipe records scalar_decay = 0 and keeps its original grid.
    conversion = {
        "min_l_eigenvalue_before_psd_conversion": min_l_eigenvalue,
        "min_l_eigenvalue_after_psd_conversion": min_l_eigenvalue_after,
        "psd_correction_applied": psd_shift > 0.0,
        "psd_shift": float(psd_shift),
        "scalar_decay": 0.0,
        "psd_premise_evidence": "numerical",
        "make_l_psd": method.make_l_psd,
        "psd_tolerance": float(method.psd_tolerance),
    }
    return _PreparedDecomposition(
        l_part=l_part,
        h_evolution=h_evolution,
        conversion=conversion,
        lambda_min_before=min_l_eigenvalue,
        l_norm=float(l_norm),
        numerical_psd_premise_satisfied=decision.premise_satisfied,
    )


@dataclass(frozen=True, kw_only=True)
class LCHSQuadratureData:
    """Quadrature-plan scalars that every LCHS recipe shares, without system arrays.

    It carries the node grid, the coefficients and their counts (``M``,
    ``Q``, interval counts) and the PSD facts of the processed L. It holds no
    array of the system dimension, so a compact generator and a dense one
    use the same record. The dense recipe keeps its padded L and H in
    ``selection.DenseLCHSInputs``.

    Attributes:
        conversion: PSD-shift metadata. ``psd_shift`` is the nonnegative
            shift s_+ and ``scalar_decay`` the nonnegative known decay c
            (``solution_error_budget.signed_shift``), so the signed shift is s = s_+ - c and the original
            physical propagator is exp(s*t) times the propagator of A + s*I.
            ``psd_premise_evidence`` names the evidence for the PSD premise
            of L + s*I, which is ``"numerical"`` (the dense eigensolve rule),
            ``"certified_bound"`` (exact structural bounds of a compact
            input) or ``"user_assertion"`` (a supplied encoding, or a lower bound
            of L stated to ``ingest_cartesian`` that improves on the structural
            one). The dense
            recipe adds the computed smallest eigenvalue before and after the
            shift, the compact recipe the lower and upper bounds it used.
        h1: Gauss panel width in k, or None for a non-Gauss/unitary rule.
        range_k: Cutoff K of the rule as built and of its available
            component bounds, or the maximum node magnitude for a signed-binary
            grid (zero for the exact unitary reduction). In composite Gauss, K
            specifies the panel interval [-K, K]. The Gauss nodes are interior.
        interval_count_each_side: Gauss panels per half-axis, symmetric-uniform integer
            half-range J, or zero for signed/unitary rules.
        node_count: Gauss points per panel Q, or total points for non-Gauss rules.
        total_node_count: Mathematical node count M, excluding the padding address
            slots added later for the LCU.
        k_nodes: Real array of shape (M,) in the node order of the quadrature rule.
        coefficients: Complex array of shape (M,) before physical PSD-shift compensation.
        coefficient_l1_norm: Sum of these uncompensated coefficient magnitudes.
        l_norm: Bound on ||L + s*I|| used to select the grid: the dense
            recipe's numerical spectral-norm estimate, or the compact recipe's
            structural bound ``operators.cartesian.ScalarCompensation.processed_norm``.
            No additional eigensolve is implied by reading it.
        psd_premise_satisfied: Whether L + s*I is known to be positive
            semidefinite, by the evidence ``conversion["psd_premise_evidence"]``
            names. The numerical rule's outcome is not a formal PSD
            certificate.
        coefficient_plan: The LCHSCoefficientPlan from which these arrays were
            taken, with its kernel and quadrature configuration, nodes, weights, coefficients
            and component bounds.
    """

    conversion: Mapping[str, Any]
    h1: float | None
    range_k: float
    interval_count_each_side: int
    node_count: int
    total_node_count: int
    k_nodes: np.ndarray
    coefficients: np.ndarray
    coefficient_l1_norm: float
    l_norm: float
    psd_premise_satisfied: bool
    coefficient_plan: LCHSCoefficientPlan


def _quadrature_data_from_plan(
    *,
    l_norm: float,
    conversion: Mapping[str, Any],
    psd_premise_satisfied: bool,
    final_time: float,
    method: LCHS,
    max_bytes=None,
    max_quadrature_work=None,
    providers=None,
    work_ledger=None,
    dense_select_slot_limit=None,
) -> LCHSQuadratureData:
    """Assemble quadrature data from the processed-L norm bound, the PSD facts and the coefficient plan.

    A zero ``l_norm`` means L + s*I is exactly zero, so the unitary reduction
    of a single node replaces the k integral. ``providers`` is a
    ``selection.ProviderEntry`` of the coherent provider frontier, whose
    kernel, quadrature and private seams (``free_half_width``,
    ``strip_margin``) are used; None uses the Method's pair. A supplied
    ``work_ledger`` receives the admitted quadrature work, and
    ``dense_select_slot_limit`` is the slot limit of a quantum dense_exact route
    (``providers._resolve_coefficient_context``).
    """

    if l_norm == 0.:
        from .providers import _unitary_coefficient_plan
        coefficient_plan = _unitary_coefficient_plan(method)
    else:
        seams = {} if providers is None else dict(free_half_width=providers.free_half_width,
                                                  strip_margin=providers.strip_margin)
        coefficient_plan = _resolve_coefficient_context(
        lchs_kernel=method.lchs_kernel if providers is None else providers.lchs_kernel,
        k_quadrature=method.k_quadrature if providers is None else providers.k_quadrature,
        **seams,
        work_ledger=work_ledger,
        dense_select_slot_limit=dense_select_slot_limit,
        problem_context=LCHSProblemContext(
            final_time=float(final_time),
            epsilon=float(method.approximation_tolerance),
            l_norm=l_norm,
        ),
        max_bytes=max_bytes,
        max_quadrature_work=max_quadrature_work,
        )
    quadrature_plan = coefficient_plan.quadrature
    return LCHSQuadratureData(
        conversion=conversion,
        h1=quadrature_plan.h1,
        range_k=quadrature_plan.range_k,
        interval_count_each_side=quadrature_plan.interval_count_each_side,
        node_count=quadrature_plan.node_count,
        total_node_count=quadrature_plan.physical_node_count,
        k_nodes=np.asarray(coefficient_plan.nodes, dtype=float),
        coefficients=np.asarray(coefficient_plan.coefficients, dtype=complex),
        coefficient_l1_norm=coefficient_plan.coefficient_l1_norm,
        l_norm=l_norm,
        psd_premise_satisfied=psd_premise_satisfied,
        coefficient_plan=coefficient_plan,
    )


def lchs_quadrature_summary(
    quadrature_data: LCHSQuadratureData,
    *,
    method: LCHS,
    he_backend: str,
    final_time: float,
) -> dict[str, Any]:
    """Assemble the shared homogeneous LCHS quadrature dictionary.

    K, Q, M, h1 and the component bounds come from the coefficient plan's record.
    operator_scale is the PSD recovery exp(shift*T), and
    lcu_coefficient_l1_norm is operator_scale times the uncompensated
    coefficient 1-norm. Both are None when binary64 cannot represent them.
    matrix_norm, prepared_matrix_norm and h_norm are None here. Explicit
    refinement evaluates those spectral norms on request.
    """

    operator_scale = psd_recovery(float(quadrature_data.conversion["psd_shift"]), final_time)
    lcu_norm = None if operator_scale is None else operator_scale*quadrature_data.coefficient_l1_norm
    l_norm = quadrature_data.l_norm
    coefficient_plan = quadrature_data.coefficient_plan
    provider_record = coefficient_plan.record()
    trotter_formula = (
        {
            "trotter_formula": "lie_1_sorted_pauli_product"
            if method.trotter_order == 1
            else "suzuki_2_sorted_pauli_product"
        }
        if he_backend == "trotter"
        else {}
    )
    return {
        "beta": coefficient_plan.resolved_lchs_kernel.parameters.get("beta"),
        "epsilon": float(method.approximation_tolerance),
        "truncation_multiplier": (
            coefficient_plan.resolved_k_quadrature.parameters.get("truncation_multiplier")
        ),
        "K": float(quadrature_data.range_k),
        "h1": quadrature_data.h1,
        "Q": int(quadrature_data.node_count),
        "M": quadrature_data.total_node_count,
        "interval_count_each_side": int(quadrature_data.interval_count_each_side),
        "coefficient_l1_norm": quadrature_data.coefficient_l1_norm,
        "matrix_norm": None,
        "prepared_matrix_norm": None,
        "l_norm": l_norm,
        "h_norm": None,
        "hamiltonian_evolution_backend": he_backend,
        "trotter_steps": int(method.trotter_steps if he_backend == "trotter" else 0),
        **trotter_formula,
        "operator_scale": operator_scale,
        "lcu_coefficient_l1_norm": lcu_norm if lcu_norm is None or isfinite(lcu_norm) else None,
        **dict(quadrature_data.conversion),
        **provider_record,
    }


@dataclass(frozen=True, kw_only=True)
class LCHSProductFormulaSelectData:
    """Homogeneous product-formula plan plus shared LCU and report records.

    Attributes:
        coefficients: Physical complex LCU coefficients in mathematical node order,
            including any PSD-shift compensation.
        plan: Compact product-formula occurrence tables, branch mapping and
            phases that SELECT construction and its resource formulas use.
        quadrature: Shared kernel-quadrature, conversion and per-node synthesis metadata.
            Reading these records does not construct the SELECT circuit.
        numerical_psd_premise_satisfied: Outcome of the numerical PSD check of
            the shared prepared L under its configured tolerance.
        nodes: Saved coefficient table, once per k-node
            (LCHSProductFormulaNodes), read by verification and refinement.
    """

    coefficients: np.ndarray
    plan: LCHSProductFormulaSelectPlan
    quadrature: Mapping[str, Any]
    numerical_psd_premise_satisfied: bool
    nodes: LCHSProductFormulaNodes


def _affine_address_coefficients(
    address_structure: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return two's-complement bit coefficients after strict metadata checks."""

    num_qubits = address_structure.get("num_qubits")
    sign_bit = address_structure.get("sign_bit")
    if type(num_qubits) is not int or num_qubits < 1:
        raise ValueError("affine address certificate requires a positive num_qubits")
    if sign_bit != num_qubits - 1:
        raise ValueError("affine address certificate sign_bit must be the high bit")
    coefficients: list[float | None] = [None] * num_qubits
    for term in address_structure.get("unsigned_terms", ()):
        if not isinstance(term, Mapping):
            raise ValueError("affine address unsigned terms must be mappings")
        qubit = term.get("qubit")
        if type(qubit) is not int or not 0 <= qubit < num_qubits - 1:
            raise ValueError("affine address unsigned-term qubit is out of range")
        if coefficients[qubit] is not None:
            raise ValueError("affine address certificate repeats an unsigned qubit")
        coefficients[qubit] = float(term.get("coefficient"))
    coefficients[sign_bit] = float(address_structure.get("sign_coefficient"))
    if any(value is None for value in coefficients):
        raise ValueError("affine address certificate must cover every address bit")
    result = tuple(float(value) for value in coefficients)
    if not np.all(np.isfinite(result)):
        raise ValueError("affine address coefficients must be finite")
    return result


def _attach_affine_pauli_structure(
    plan: LCHSProductFormulaSelectPlan,
    *,
    decomposition: _TrotterPauliDecomposition,
    occurrence_template: tuple[tuple[str, float], ...],
    padded_times: np.ndarray,
    address_structure: Mapping[str, Any],
) -> LCHSProductFormulaSelectPlan:
    """Attach the affine-Pauli certificate when address metadata permits.

    For occurrence (P, m) of the product-formula template, with r steps of
    length t/r, the rotation angle at address j is
    theta_j = m*(t/r)*(h_P + k_j*l_P). On the signed-binary grid
    k_j = sum_b w_b*x_b(j) is affine in the address bits x_b, with bit
    weights w_b from the two's-complement address structure, so
    theta_j = m*(t/r)*h_P + sum_b m*(t/r)*l_P*w_b*x_b(j). Structured SELECT
    implements this as at most one fixed rotation plus one singly controlled
    rotation per address bit with a nonzero coefficient
    (subroutines._multiplexors.append_affine_rotation), instead of a
    uniformly controlled rotation over all 2**a angles. Each generic table is reconstructed from these affine
    coefficients and compared entrywise, and any failed assumption is recorded
    as a rejection reason instead of raising. This construction is NWQLib's.
    """
    from nwqlib.algorithms.lchs.select_synthesis import _validate_affine_pauli_domain
    from nwqlib.subroutines._multiplexors import certify_affine_angle_table

    if not address_structure.get("affine", False):
        return plan
    # A structured address needs uniform steps/time and a fully occupied
    # address basis. Padding and branch-dependent schedules can break affinity.
    reasons: list[str] = []
    if len(set(plan.branch_step_counts)) != 1:
        reasons.append("fixed-step product formula requires a uniform step count")
    if (
        plan.physical_node_count != plan.padded_node_count
        or any(node is None for node in plan.branch_to_node)
    ):
        reasons.append("physical nodes must fill the address basis without padding")
    if len(set(float(value) for value in padded_times)) != 1:
        reasons.append("structured affine occurrences require a uniform elapsed time")

    try:
        address_coefficients = _affine_address_coefficients(address_structure)
    except (TypeError, ValueError) as exc:
        reasons.append(f"address values are not affine in computational-basis bits: {exc}")
        address_coefficients = ()
    if address_coefficients and (1 << len(address_coefficients)) != plan.padded_node_count:
        reasons.append("affine address certificate does not cover the SELECT control basis")
        address_coefficients = ()

    # Reconstruct every generic angle table from affine address coefficients.
    # The residual check catches pruning that changes the purported structure.
    affine_tables = []
    max_residual = 0.0
    max_tolerance = 0.0
    steps = plan.branch_step_counts[0] if plan.branch_step_counts else 0
    elapsed_time = float(padded_times[0]) if padded_times.size else 0.0
    if steps <= 0 and len(plan.occurrence_angle_tables):
        reasons.append("structured Pauli occurrences require a positive fixed step count")
    if address_coefficients and steps > 0:
        repeated_template = tuple(
            islice(cycle(occurrence_template), len(plan.occurrence_angle_tables))
        )
        for (label, multiplier), generic_table in zip(
            repeated_template,
            plan.occurrence_angle_tables.array,
            strict=True,
        ):
            l_coefficient = complex(decomposition.l_coefficients.get(label, 0.0))
            h_coefficient = complex(decomposition.h_coefficients.get(label, 0.0))
            if l_coefficient.imag != 0.0 or h_coefficient.imag != 0.0:
                reasons.append(
                    f"affine Pauli structure rejected unsupported term {label!r}: "
                    "Pauli coefficients must be real"
                )
                break
            factor = float(multiplier * elapsed_time / steps)
            affine = certify_affine_angle_table(
                generic_table,
                offset=factor * h_coefficient.real,
                coefficients=tuple(
                    factor * l_coefficient.real * value
                    for value in address_coefficients
                ),
            )
            affine_tables.append(affine)
            max_residual = max(max_residual, affine.max_residual)
            max_tolerance = max(max_tolerance, affine.tolerance)

    pauli_terms = tuple(
        (label, decomposition.h_coefficients.get(label, 0.0) + decomposition.l_coefficients.get(label, 0.0))
        for label in plan.pauli_labels
    )
    try:
        _validate_affine_pauli_domain(pauli_terms)
    except (TypeError, ValueError) as exc:
        reasons.append(f"affine Pauli structure rejected unsupported terms: {exc}")

    reconstruction_failed = any(
        affine.max_residual > affine.tolerance for affine in affine_tables
    )
    if reconstruction_failed and any(plan.pruned_l1_mass):
        reasons.append("pruning does not preserve affine occurrence angle tables")
    if reconstruction_failed:
        reasons.append(
            "structure certificate reconstruction does not match generic angle tables"
        )
    if len(affine_tables) != len(plan.occurrence_angle_tables):
        reasons.append("affine occurrence tables do not cover every occurrence")

    certificate = MappingProxyType(
        {
            "eligible": not reasons,
            "rejection_reasons": tuple(dict.fromkeys(reasons)),
            "reconstructed_table_max_residual": float(max_residual),
            "reconstruction_tolerance": float(max_tolerance),
            "symbolic_check_count": len(plan.occurrence_angle_tables)
            * len(address_coefficients),
            "explicit_table_entry_checks": len(plan.occurrence_angle_tables)
            * plan.padded_node_count,
        }
    )
    payload = MappingProxyType({"occurrence_affine_tables": tuple(affine_tables)})
    return replace(
        plan,
        structure_certificate=certificate,
        structured_generator_payload=payload,
    )


def _repetition_blocks(branch_step_counts: Sequence[int]) -> tuple[list[int], tuple[int, ...]]:
    """Group formula repetitions into blocks with a fixed set of active branches.

    Multiplexed SELECT advances every branch through one shared schedule, and
    branch j is active in repetitions 0, ..., r_j - 1. Between two consecutive
    distinct nonzero step counts the active set does not change, so one angle
    table per occurrence serves the whole block. For step counts (3, 5, 0, 8)
    the blocks start at repetitions 0, 3 and 5 and repeat 3, 2 and 3 times.

    Returns:
        The first repetition index of each block, and the number of
        repetitions in each block. Their total equals max(branch_step_counts).
    """
    block_ends = sorted(set(branch_step_counts) - {0})
    block_starts = [0, *block_ends[:-1]] if block_ends else []
    block_repetitions = tuple(
        end - start for start, end in zip(block_starts, block_ends, strict=True)
    )
    return block_starts, block_repetitions


def _build_product_formula_select_plan(
    *,
    decomposition: _TrotterPauliDecomposition,
    k_values: np.ndarray,
    elapsed_times: np.ndarray,
    coefficients: np.ndarray,
    branch_to_node: tuple[int | None, ...],
    method: LCHS,
    address_structure: Mapping[str, Any],
    max_steps: int | None = None,
    max_bytes=None,
    max_select_work=None,
    charged_before: int = 0,
    grid_k_values: Sequence[float] | None = None,
    work_ledger: dict | None = None,
    coefficient_phases: Sequence[float] | None = None,
) -> tuple[LCHSProductFormulaSelectPlan, tuple[_LCHSTrotterNodeRecord, ...], LCHSProductFormulaNodes]:
    """Build the shared homogeneous/inhomogeneous product-formula plan.

    The coefficient phase of each address slot is arg(coefficient), or the
    supplied ``coefficient_phases``: the raw coefficient arguments of the
    homogeneous plan, or the phases of a source layout, whose coefficient
    state is kept as amplitudes and phases (source_selection).

    ``max_select_work`` is what remains of the caller's limit after the
    ``charged_before`` work it already counted (the Pauli decomposition), so
    a refusal states the value that limit needs and its current value,
    ``max_select_work + charged_before``. A supplied ``work_ledger`` mapping
    receives, under ``"work"``, the max_select_work that this selection
    counted (its selection, census and angle-table work), so the caller can
    carry the remainder to later stages.

    The plan tables cover the power-of-two SELECT address slots. The returned
    records follow the supplied address-slot tables one-to-one, so callers
    weight each record by the coefficient of the same address slot. The
    returned node table (LCHSProductFormulaNodes) keeps each distinct node's
    coefficients once, over the quadrature grid (grid_k_values for a source
    layout, otherwise the positions that branch_to_node names), for later
    verification and refinement. Added padding address slots have zero
    coefficient and zero steps and stay inside the plan.

    For B compressed repetition blocks, W template occurrences and S padded
    address slots, the float64 angle table contains E = B*W*S entries. The fill loop
    forms one S-entry row at a time, checks each angle for finiteness and
    canonicalizes signed zero. A private
    ownership transfer makes the filled buffer read-only without copying it,
    so table construction and storage each require 8*E data bytes
    beside selection_bytes, and the fill adds at most 48*S bytes of row
    scratch. Optional affine validation adds at most three
    float64 S-entry row arrays, giving 8*E + 24*S for that phase. The check
    counts both allowances. These are
    known array-data terms, not a process RSS bound.

    Each address slot j evolves k_j*L + H for its elapsed time t_j with r_j
    steps of the Lie-1 or Suzuki-2 template. Steps are fixed (trotter) or
    chosen per address slot against the per-node budget
    (trotter_error_budgeted). The angle of occurrence (P, m) at address slot
    j is m*t_j*c_P(k_j)/r_j, where m is the Qiskit template multiplier. It
    is 2 for every Lie-1 term. In Suzuki-2 it is 2 for the middle label and 1
    for each of the two occurrences of every other label. The saved value is
    therefore the RZ angle, twice the exponent. An address slot whose r_j
    repetitions are used up gets angle zero (see _repetition_blocks).
    """
    from nwqlib.algorithms.lchs.select_synthesis import (
        LCHSProductFormulaSelectPlan,
        product_formula_occurrence_template,
    )

    he_backend = method.hamiltonian_evolution_backend
    k_table = np.asarray(k_values, dtype=float).reshape(-1)
    time_table = np.asarray(elapsed_times, dtype=float).reshape(-1)
    coefficient_table = np.asarray(coefficients, dtype=complex).reshape(-1)
    slot_count = int(coefficient_table.size)
    if not (slot_count == k_table.size == time_table.size == len(branch_to_node)):
        raise ValueError("product-formula branch tables must have equal length")
    if slot_count < 1:
        raise ValueError("product-formula branch tables must be non-empty")
    padded_node_count = 1 << max(0, slot_count - 1).bit_length()
    from nwqlib.operators.access import _check_bytes
    terms = len(set(decomposition.l_coefficients) | set(decomposition.h_coefficients))
    q = decomposition.num_qubits
    budgeted = he_backend == "trotter_error_budgeted"
    padded_branch_to_node = branch_to_node + (None,) * (padded_node_count - slot_count)
    # One combined record per distinct physical k-node (exact binary64 k),
    # shared by every slot and application that uses that node. Padding
    # slots have no node.
    node_keys = [
        _node_key(k_value) if node is not None else None
        for k_value, node in zip(np.pad(k_table, (0, padded_node_count - slot_count)),
                                 padded_branch_to_node, strict=True)
    ]
    # The stored node table covers the original quadrature grid: grid_k_values
    # when given (a source layout, whose branch_to_node names physical
    # branches), otherwise the positions that branch_to_node names.
    if grid_k_values is None:
        grid_k: list[float | None] = [None]*(1 + max(node for node in branch_to_node if node is not None))
        for k_value, node in zip(k_table, branch_to_node, strict=True):
            if node is not None:
                grid_k[node] = float(k_value)
        if any(k_value is None for k_value in grid_k):
            raise ValueError("product-formula slots must name every quadrature position")
    else:
        grid_k = [float(k_value) for k_value in grid_k_values]
    distinct = tuple(dict.fromkeys([key for key in node_keys if key is not None]
                                   + [_node_key(k_value) for k_value in grid_k]))
    # All node coefficient combinations and the shared census precede the
    # compact occurrence/angle table. Work, with u = terms the union label
    # count including identity, K distinct nodes and S padded slots: the
    # one-time support alignment W_align, sum_nodes((q+2)*u + 10*u + 5) for
    # each node's combination allowance, operand-scale scan and streamed upper
    # dropped mass (_coefficient_preparation_work) and 16*S before the census,
    # then p*q + G + K_n*V for the admitted shared census when budgeted
    # (_node_bound_coefficients). These units are admission proxies, not
    # timings or equal-cost CPU operations.
    align_work, _ = _coefficient_preparation_work(
        q, len(decomposition.l_coefficients), len(decomposition.h_coefficients), terms, 0)
    selection_work = (align_work + len(distinct)*((q+2)*terms + 10*terms + 5)
                      + 16*padded_node_count)
    # Zero is a legal budget here. It is what _admit_pauli_decomposition
    # leaves when the decomposition used the whole limit, and every selection
    # then exceeds it.
    max_select_work = None if max_select_work is None else integer(max_select_work, "max_select_work", 0)
    if exceeds(selection_work, max_select_work):
        raise ResourceLimitError(
            f"product-formula branch selection needs {selection_work} units of max_select_work, "
            f"above the {max_select_work} left after {charged_before} already charged; this stage "
            f"needs max_select_work to be at least {selection_work+charged_before} "
            f"(now {max_select_work+charged_before})")
    # Bytes: the combined node cache once per distinct node (_node_cache_bytes),
    # the padded k/time/coefficient tables at 32 bytes per slot, and a fixed
    # 8192-byte envelope for each slot's node record and step selection
    # (registered in ENGINEERING_CONSTANTS).
    selection_bytes = (_node_cache_bytes(len(distinct), terms, q)
                       + padded_node_count*(32+8192))
    _check_bytes(selection_bytes, max_bytes, "product-formula branch selection")
    padded_k = np.zeros(padded_node_count, dtype=float)
    padded_times = np.zeros(padded_node_count, dtype=float)
    padded_coefficients = np.zeros(padded_node_count, dtype=complex)
    padded_k[:slot_count] = k_table
    padded_times[:slot_count] = time_table
    padded_coefficients[:slot_count] = coefficient_table
    padded_phases = (np.angle(padded_coefficients) if coefficient_phases is None
                     else np.concatenate((np.asarray(coefficient_phases, dtype=float),
                                          np.zeros(padded_node_count-slot_count))))

    pauli_labels = tuple(
        label
        for label in sorted(set(decomposition.l_coefficients) | set(decomposition.h_coefficients))
        if label != decomposition.identity_label
    )
    occurrence_template = product_formula_occurrence_template(
        pauli_labels,
        order=method.trotter_order,
    )
    occurrence_schedule = tuple(label for label, _ in occurrence_template)

    combined_nodes = _distinct_combined_nodes(
        decomposition, [k for k, key in zip(padded_k, node_keys, strict=True) if key is not None] + grid_k)
    table_bytes = _node_table_bytes(
        q, terms, [len(combined_nodes[key].kept_indices) for key in distinct],
        [len(combined_nodes[key].pruned_indices) for key in distinct], len(grid_k))
    selection_bytes += table_bytes
    _check_bytes(selection_bytes, max_bytes, "product-formula stored node table")
    node_table = _selected_node_table(decomposition, combined_nodes, grid_k)
    evaluations: dict[str, Any] = {}
    if budgeted:
        # Every stored node with kept nonidentity terms is weighted once any elapsed time is
        # nonzero, the population that the classical plan (host_pf) and the
        # refinement of the saved table (_fixed_trotter_certificate_records)
        # weight, so all three choose the expression from the same K.
        moving = any(key is not None and elapsed_time != 0.0
                     for key, elapsed_time in zip(node_keys, padded_times, strict=True))
        census_keys = tuple(key for key in distinct if moving and combined_nodes[key].traceless_terms)
        # The census coexists with the cached nodes, the padded tables, the
        # decomposition's logical label/coefficient tables and the live dense
        # L and H (32*d**2 bytes).
        census_held = selection_bytes + (q+16)*(
            len(decomposition.l_coefficients)+len(decomposition.h_coefficients)) + 32*(1 << q)**2
        evaluations, ledger = _node_bound_coefficients(
            _union_nonidentity_labels(decomposition), q,
            {key: combined_nodes[key].traceless_terms for key in census_keys},
            census_keys, order=int(method.trotter_order),
            max_work=max_select_work, max_bytes=max_bytes, census_held=census_held,
            limit_owner="LCHS", work_field="max_select_work",
            field_work_offset=charged_before,
            linear_and_other_work=selection_work)
        selection_work += ledger["work"]
    empty = _CombinedTrotterTerms(identity_coefficient=0.0, traceless_terms=(), pruned_labels=(),
                                  pruned_l1_mass=0.0)
    combined_terms = [empty if key is None else combined_nodes[key] for key in node_keys]
    used: set[str] = set()
    slot_records: list[_LCHSTrotterNodeRecord] = []
    for slot, (key, elapsed_time) in enumerate(zip(node_keys, padded_times, strict=True)):
        if key is None or elapsed_time == 0.0:
            record = _LCHSTrotterNodeRecord(
                pf_bound_value=0.0,
                pruned_l1_mass=0.0,
                combined_bound_value=0.0,
                step_count=0,
                selection=None,
                bound_value_status="not_applicable",
            )
        elif budgeted:
            record = _budgeted_trotter_node_record(
                combined_nodes[key],
                _first_use(evaluations, key, used),
                final_time=float(elapsed_time),
                error_budget=_lchs_trotter_error_budget_per_node(method),
                order=method.trotter_order,
            )
        else:
            record = _LCHSTrotterNodeRecord(
                pf_bound_value=None,
                pruned_l1_mass=combined_nodes[key].pruned_l1_mass,
                combined_bound_value=None,
                # Fixed-step plans keep their common repetition schedule even
                # when one branch generator is zero. That branch's angles are zero.
                step_count=int(method.trotter_steps),
                selection=None,
            )
        if max_steps is not None and record.step_count > max_steps:
            raise ResourceLimitError(f"a product-formula node needs {record.step_count} steps, above the step limit "
                                     f"{max_steps} (LCHS.max_trotter_steps)")
        slot_records.append(record)

    branch_step_counts = tuple(record.step_count for record in slot_records)
    max_step_count = max(branch_step_counts, default=0)
    block_starts, block_repetitions = _repetition_blocks(branch_step_counts)
    node_coefficients = {key: dict(combined.traceless_terms) for key, combined in combined_nodes.items()}
    term_coefficients = [{} if key is None else node_coefficients[key] for key in node_keys]
    table_entries = len(occurrence_template)*len(block_repetitions)*padded_node_count
    # Selection stores W*B*S angles (W template occurrences, B blocks, S
    # padded slots) in one float64 array, 8 bytes per stored angle. Emitting the W*R
    # occurrences of all R repetitions happens later, at lowering. The table
    # is frozen by ownership transfer (FrozenArray._from_owned_canonical_array,
    # which states what that leaves writable). A copy into protected bytes
    # storage would hold 16 bytes per angle at the handoff. Construction and storage both hold
    # 8 bytes per angle. The fill forms one table row at a time with at most
    # 48 bytes of row scratch per slot (a complex coefficient row and float64
    # and mask rows). Affine row checking adds three S-entry float64 rows.
    angle_peak_bytes = 8 * table_entries + (48 * padded_node_count if table_entries else 0)
    if (
        table_entries
        and he_backend == "trotter"
        and method.lcu_select_implementation in ("auto", "structured")
        and address_structure.get("affine", False)
    ):
        angle_peak_bytes += 24 * padded_node_count
    _check_bytes(
        selection_bytes + angle_peak_bytes,
        max_bytes,
        "product-formula angle tables",
    )
    table_work = selection_work+8*table_entries+len(occurrence_template)*padded_node_count
    if exceeds(table_work, max_select_work):
        raise ResourceLimitError(
            f"product-formula angle-table selection needs {table_work} units of max_select_work, "
            f"above the {max_select_work} left after {charged_before} already charged; this stage "
            f"needs max_select_work to be at least {table_work+charged_before} "
            f"(now {max_select_work+charged_before})")
    angle_tables = np.zeros(
        (len(block_starts) * len(occurrence_template), padded_node_count),
        dtype="<f8",
    )
    # One row of S slot angles at a time, in the order (repetition block,
    # occurrence, slot): angle = multiplier*t*c.real/steps on an active slot
    # and 0.0 otherwise, evaluated left to right as in the scalar formula.
    # The first slot that fails a check raises its own error, the
    # imaginary-coefficient check before the finite-angle check.
    steps = np.array(branch_step_counts, dtype=float)
    slot_cutoffs = np.array([combined.coefficient_cutoff for combined in combined_terms])
    row = 0
    for repetition in block_starts:
        active = (repetition < steps) & (steps > 0)
        for label, multiplier in occurrence_template:
            coefficient = np.array([by_label.get(label, 0.0) for by_label in term_coefficients], dtype=complex)
            imaginary = np.abs(coefficient.imag) > slot_cutoffs
            angle = np.zeros(padded_node_count)
            with np.errstate(over="ignore", invalid="ignore"):
                np.divide(multiplier * padded_times * coefficient.real, steps, out=angle, where=active)
            failed = imaginary | ~np.isfinite(angle)
            if failed.any():
                slot = int(np.flatnonzero(failed)[0])
                if imaginary[slot]:
                    raise ValueError(
                        "Hamiltonian coefficients must be real for product-formula "
                        f"synthesis; term {label!r} has coefficient {complex(coefficient[slot])}"
                    )
                raise ValueError("FrozenArray float64 elements must be finite")
            angle_tables[row] = angle + 0.0
            row += 1
    angle_tables = FrozenArray._from_owned_canonical_array(angle_tables)
    plan = LCHSProductFormulaSelectPlan(
        pauli_labels=pauli_labels,
        occurrence_schedule=occurrence_schedule,
        occurrence_angle_tables=angle_tables,
        occurrence_block_repetitions=block_repetitions,
        branch_step_counts=branch_step_counts,
        identity_phases=tuple(
            float(-elapsed_time * combined.identity_coefficient) if node is not None else 0.0
            for combined, elapsed_time, node in zip(
                combined_terms,
                padded_times,
                padded_branch_to_node,
                strict=True,
            )
        ),
        coefficient_phases=tuple(float(value) for value in padded_phases),
        physical_node_count=sum(node is not None for node in padded_branch_to_node),
        padded_node_count=padded_node_count,
        branch_to_node=padded_branch_to_node,
        pruning_decisions=tuple(
            combined.pruned_labels if node is not None else ()
            for combined, node in zip(
                combined_terms,
                padded_branch_to_node,
                strict=True,
            )
        ),
        pruned_l1_mass=tuple(
            combined.pruned_l1_mass if node is not None else 0.0
            for combined, node in zip(
                combined_terms,
                padded_branch_to_node,
                strict=True,
            )
        ),
        formula_order=int(method.trotter_order),
        repetitions=max_step_count,
        structure_certificate=None,
        structured_generator_payload=None,
    )
    if he_backend == "trotter" and method.lcu_select_implementation in ("auto", "structured"):
        plan = _attach_affine_pauli_structure(
            plan,
            decomposition=decomposition,
            occurrence_template=occurrence_template,
            padded_times=padded_times,
            address_structure=address_structure,
        )
    if work_ledger is not None:
        work_ledger["work"] = selection_work+8*table_entries+len(occurrence_template)*padded_node_count
    return plan, tuple(slot_records[:slot_count]), node_table


def generate_lchs_product_formula_select_plan(
    *,
    quadrature: LCHSQuadratureData,
    parts: tuple[np.ndarray, np.ndarray],
    final_time: float,
    method: LCHS,
    max_steps: int | None = None,
    max_bytes=None,
    max_select_work=None,
    work_ledger: dict | None = None,
) -> LCHSProductFormulaSelectData:
    """Build one common product-formula SELECT plan for a homogeneous circuit of a dense A.

    A supplied ``work_ledger`` mapping receives, under ``"work"``, the
    max_select_work counted by the Pauli decomposition and the selection.

    Every k-node evolves for the full time T. The LCU coefficients returned
    include exp(s*T) for the signed shift s, divided by
    2**psd_recovery_exponent(s, T) when that factor is outside the normal
    binary64 range. The SELECT coefficient phases are the arguments of the
    raw c_j, because the positive factor does not change them and a
    subnormal part multiplied by it can round by up to order one. The
    recorded synthesis bound weights each node's bound by the uncompensated
    |c_j|, because solution_error_budget applies the PSD recovery once per
    application. ``quadrature`` and ``parts``, the padded L (after the
    shift) and H whose order gives the dimension, are those of the dense
    recipe, ``selection.select_dense(method, problem, backend=...)``, as
    its ``quadrature`` and ``dense.l_part``, ``dense.h_part``.
    """

    he_backend = method.hamiltonian_evolution_backend
    # A valid dense_exact or QSP Method must not receive a product-formula plan.
    if he_backend not in ("trotter", "trotter_error_budgeted"):
        raise ValueError(
            "product-formula SELECT planning serves the product-formula "
            f"backends ('trotter', 'trotter_error_budgeted'); got {he_backend!r}"
        )
    l_part, h_part = parts
    dimension = l_part.shape[0]
    selection = {}
    charged, remaining_work = _admit_pauli_decomposition(dimension, max_bytes=max_bytes,
                                                         max_select_work=max_select_work)
    decomposition = _trotter_pauli_decomposition(l_part, h_part)
    shift = signed_shift(quadrature.conversion)
    operator_scale = (
        psd_recovery_part(shift, final_time, psd_recovery_exponent(shift, final_time))
        if shift != 0.0
        else 1.0
    )
    raw = np.asarray(quadrature.coefficients, dtype=complex)
    coefficients = operator_scale * raw
    plan, slot_records, node_table = _build_product_formula_select_plan(
        decomposition=decomposition,
        k_values=np.asarray(quadrature.k_nodes, dtype=float),
        elapsed_times=np.full(quadrature.total_node_count, final_time),
        coefficients=coefficients,
        branch_to_node=tuple(range(quadrature.total_node_count)),
        method=method,
        address_structure=quadrature.coefficient_plan.quadrature.address_structure,
        max_steps=max_steps,max_bytes=max_bytes,max_select_work=remaining_work,charged_before=charged,
        work_ledger=selection,coefficient_phases=np.angle(raw),
    )
    if work_ledger is not None:
        work_ledger["work"] = charged+selection["work"]
    summary = {
        **lchs_quadrature_summary(
            quadrature,
            method=method,
            he_backend=he_backend,
            final_time=final_time,
        ),
        **_trotter_budget_quadrature_terms(
            slot_records,
            coefficients=quadrature.coefficients,
            method=method,
        ),
    }
    return LCHSProductFormulaSelectData(
        coefficients=coefficients,
        plan=plan,
        quadrature=summary,
        numerical_psd_premise_satisfied=quadrature.psd_premise_satisfied,
        nodes=node_table,
    )


__all__ = [
    "LCHSProductFormulaSelectData",
    "LCHSQuadratureData",
    "cartesian_decomposition",
    "generate_lchs_product_formula_select_plan",
    "lchs_quadrature_summary",
]
