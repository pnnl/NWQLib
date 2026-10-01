"""Quadrature data, dense terms and product-formula plans for time-independent LCHS.

This term generator supplies the homogeneous ``b(t) = 0`` specialization.
The constant-source construction adds source-state and elapsed-time quadrature
above the same ``A = L + iH`` decomposition and ``k``-quadrature machinery.

ACL is An, Childs, and Lin, arXiv:2312.03916v2, and CSTWZ is Childs, Su,
Tran, Wiebe, and Zhu, doi:10.1103/PhysRevX.11.011020. The Cartesian
decomposition A = L + iH is ACL Eqs. (3)-(4), and the kernel, the integral
discretization and the pointwise decay of the quadrature that this module
consumes are ACL Eq. (7), Eq. (61) and Eqs. (185)-(186). The quadrature
provider (``providers.py``) integrates that decay using
E1(x) <= exp(-x)/x and selects the cutoff K by scalar bisection, and its
automatic Q-point grid uses Trefethen,
Approximation Theory and Approximation Practice, ISBN 978-1-61197-239-9,
Theorem 19.3, with n = Q - 1. This module consumes the provider's bounds
and counts without recomputing them. Pocrnic, Johnson, Katabarwa, and
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

from nwqlib._limits import DEFAULT_MAX_BYTES

from dataclasses import dataclass, replace
from functools import cached_property
from itertools import cycle, islice
from math import fsum, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from nwqlib._validation import integer
from nwqlib.core.records import FrozenArray
from nwqlib.algorithms.lchs.time_independent_common import (
    as_square_matrix,
    is_power_of_two,
    validate_final_time,
)
from nwqlib.algorithms.lchs.native import (
    LCHS_TROTTER_EPSILON_FRACTION,
    resolve_hamiltonian_evolution_backend,
)
from nwqlib.algorithms.lchs.solution_error_budget import (
    DEFAULT_PSD_TOLERANCE,
    _numerical_psd_decision,
    psd_recovery,
    psd_recovery_exponent,
    psd_recovery_part,
)
from nwqlib.algorithms.lchs.providers import (
    LCHSCoefficientPlan,
    LCHSProblemContext,
    _resolve_coefficient_context,
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
    A still live, this holds 48 d**2 bytes of complex d-square arrays.
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


def cartesian_decomposition(
    matrix: Any,
    *,
    eigcheck: bool = True,
    psd_tolerance: float = DEFAULT_PSD_TOLERANCE,
    max_bytes=DEFAULT_MAX_BYTES,
    max_spectral_work=100_000_000,
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
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Upper limit on the arrays of the check, `64 d**2 + 16 d` bytes
            for dimension d, tested before they are formed. The count covers
            the known NumPy arrays, not the eigensolver's own LAPACK
            workspace.
        max_spectral_work (int): Upper limit on the eigenvalue check of a
            nonzero L, counted as `d**3` work units, which are not timings.

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
    # hold 48 d**2 bytes. The eigenvalue check of a nonzero L is admitted
    # first: its known NumPy arrays, A, L, H, the solver's internal d-square
    # copy and the internal and public eigenvalue vectors, are
    # B_cartesian = 64 d**2 + 16 d bytes, and its work is d**3 against
    # max_spectral_work. The eigensolver's queried LAPACK workspace
    # Q_N(d) = 16 l_N + 8 r_N + I i_N (JOBZ=N) is not charged: this is a
    # declared known-array logical workspace, not a complete workspace cap.
    # These units are admission proxies, not timings or equal-cost CPU
    # operations.
    array = as_square_matrix(matrix)
    d = array.shape[0]
    if eigcheck:
        _check_bytes(64*d*d + 16*d, max_bytes, "LCHS Cartesian parts and eigenvalue check")
    l_part, h_part = _cartesian_parts(array)

    if eigcheck:
        if np.any(l_part):
            spectral_cap = integer(max_spectral_work, "max_spectral_work", 1)
            if d**3 > spectral_cap:
                raise ValueError(
                    f"LCHS eigensolve exceeds max_spectral_work: it needs {d**3} work units (d**3 with d={d}), "
                    f"max_spectral_work={spectral_cap}. Raise max_spectral_work to at least {d**3}.")
            eigenvalues = np.linalg.eigvalsh(l_part)
            lower, upper = float(eigenvalues[0].real), float(eigenvalues[-1].real)
        else:
            lower = upper = 0.0
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
    time and PSD recovery part. The host sum streams nodes and accumulates
    vectors. Explicit dense branch matrices require a matrix product at each
    time. Numerical eigensystem, phase and multiplication errors are separate
    from the ideal finite-sum approximation.

    The generator is formed by one multiply and one add and must be finite;
    this replaces the matrix-exponential norm tripwire of scipy.linalg.expm,
    which this route does not use. Columns may rotate within a degenerate
    eigenspace without changing the matrix function, and no inverse
    eigenvalue gap enters its operator error (ACL arXiv:2312.03916v2,
    Eq. (70) defines the branch). A Hermitian eigensolver is normwise
    backward stable (LAPACK Users' Guide, 3rd ed., Sec. 4.7): with
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
    branch; forming the phase and scaling the columns add 2*D**2 + 3*D.
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
    vector); the weight includes psd_recovery_part, not its carried 2**e.
    Each node's eigensystem is formed once and released before the next,
    each distinct input vector is projected once per node (a cache of at
    most the initial and source inputs, scoped to this call and keyed by the
    vector's identity), the phases of every application are accumulated in
    the eigenbasis and one final V action is made per node. The
    node-coefficient accumulation order is kept. An exactly zero L shares
    one H eigensystem across nodes; if L and H are both zero or every
    elapsed time is zero, every action is the identity and no eigensystem is
    formed. ``counts`` receives eigh_calls and matvecs.

    The dense exact route reuses one Hermitian eigensystem for each distinct
    kL+H across source elapsed times. Host evaluation applies its phases to
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
    counts as one kernel visit, not one hardware operation. Dense host action
    reuses each Hermitian eigensystem across elapsed times and projects each
    distinct input once per generator.

    Bytes, excluding the caller's live inputs (B_held):
    max(32*D**2 + 16*D, 48*D**2 + 32*D, 16*D**2 + (160 + 32*r)*D): the
    generator formation, the eigensolver's internal copy, public vectors and
    both eigenvalue vectors with the result, and the final phase with the
    eigenvectors, cached projections, combined vector, conjugate temporaries
    and phase products. The queried LAPACK workspace Q_V(D) of NumPy's zheevd
    is not charged: this is a declared known-array logical workspace, not a
    complete workspace cap. These units are admission proxies, not timings or
    equal-cost CPU operations.
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
    selected synthesis, control and any preparation-matrix work are added by
    the caller (W_synthesis, W_source). Explicit native branches additionally
    form a dense matrix and pay their selected synthesis cost. Bytes before
    synthesis are the matrix-expression frontier 64*D**2 + 24*D (V, the
    scaled matrix, conjugated V, result and phase/eigenvalue vectors); the
    caller adds live inputs, cached eigensystems, synthesis working bytes and
    completed branch circuits. The queried LAPACK workspace is not charged
    (a declared known-array logical workspace). These units are admission
    proxies, not timings or equal-cost CPU operations.
    """
    D = dimension
    return eigensystems*(8*D**3 + 34*D**2) + branches*(D**3 + 2*D**2 + 3*D), 64*D*D + 24*D


# Absolute pruning and imaginary-residue cutoff for the combined H + k*L Pauli
# coefficients. Registered in ENGINEERING_CONSTANTS with its revisit condition.
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
        string comparisons of at most q characters, and the union has u rows;
        it is charged W_align = (q+4)*s + (q+2)*u logical visits
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
    pruned_labels records the labels removed by _PAULI_COEFFICIENT_ATOL, and
    pruned_l1_mass is a finite binary64 upper bound d_up on the summed
    moduli of their stored coefficients (error_budget.upper_dropped_mass),
    from which each application publishes its own pruning bound.
    kept_indices and pruned_indices are the positions of the kept and
    dropped rows in the decomposition's aligned union
    (_TrotterPauliDecomposition.aligned_rows), recorded while the rows are
    combined, so the stored node table needs no label hashing.
    """

    identity_coefficient: float
    traceless_terms: tuple[tuple[str, complex], ...]
    pruned_labels: tuple[str, ...]
    pruned_l1_mass: float
    kept_indices: tuple[int, ...] = ()
    pruned_indices: tuple[int, ...] = ()


@dataclass(frozen=True, kw_only=True)
class _LCHSTrotterNodeRecord:
    """LCHS-only product-formula accounting for one quadrature node.

    All bounds are operator-norm bounds on one node evolution
    exp(-i*t*(k*L+H)), before the node's LCU coefficient, the physical input
    norm and the PSD recovery are applied.

    Attributes:
        pf_bound_value: Commutator bound of CSTWZ doi:10.1103/PhysRevX.11.011020,
            Props. 9-10, for the kept terms at the selected step count, or None
            when it was not evaluated.
        pruned_l1_mass: Finite upper bound on the summed moduli of the pruned
            Pauli coefficients (error_budget.upper_dropped_mass).
        combined_bound_value: up(pruning + pf_bound_value), with the pruning
            bound up(|t|*pruned_l1_mass) (error_budget.upper_combined_error),
            or None.
        step_count: Product-formula repetitions for this node. Padding and
            zero-time slots have zero, and so does a budgeted node or a
            classical action whose kept generator is empty. Fixed-step SELECT
            slots keep the common count, with zero angles for an empty
            generator.
        selection: The TrotterStepSelection of the budgeted selector, or None.
        bound_value_status: How pf_bound_value was obtained, for example
            ``structural_upper_bound``, ``not_evaluated`` or ``not_applicable``.
        dense_bound_value: Optional dense-matrix evaluation of the same bound
            plus the pruning term, used only to check the structural bound.
        pauli_term_count: Kept Pauli terms used by an evaluated fixed-step bound.
        pair_commutation_checks: Pauli pair commutation tests newly performed
            for that bound; the first node evaluated from a shared ordered
            structure carries the structure's tests and later nodes and
            applications zero.
        nested_commutation_checks: Nested commutation tests newly performed
            for that bound, counted as the pair tests are.
        bound_variant: Pauli-triangle expression of pf_bound_value,
            "exact_census" or "relaxed_prefix", or None when no census
            produced it (no kept term, not evaluated or not applicable).
        coefficient_arithmetic: Numerical evaluation of that coefficient,
            "outward_float64_scaled", or None under the same conditions.
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
    coefficient_arithmetic: str | None = None

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
            "coefficient_arithmetic": self.coefficient_arithmetic,
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
    per node, where its mass is charged to the synthesis bound.
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

    The law and its derivation are stated at _admit_pauli_decomposition.
    With q = log2(d), N = d**2 and h = q//2, the half-label widths are (h,)
    for even q and (h, q-h) for odd q; the work is (34*q+28)*N + 2*H and the
    peak bytes (104+2*q)*N + T, with T the sum of j*4**j and H the sum of
    (j+1)*4**j over those widths. At the default max_select_work=100_000_000
    the decomposition work is 87,570,944 at d = 512 and 385,888,256 at
    d = 1024.
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


def _admit_pauli_decomposition(dimension: int, *, max_bytes, max_select_work) -> int:
    """Admit both dense Pauli transforms and return the SELECT work remaining.

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
    record visits. With at most N rows for each of L and H, the total
    reservation is (34*q+28)*N + 2*H.

    L and H are transformed sequentially. During H's extraction, both
    inputs, the preceding level and the final coefficient level occupy
    64*N bytes. The index vector adds 8*N, both logical output tables add
    2*N*(q+16), and the half-label characters add T. The resulting
    (104+2*q)*N + T byte envelope also covers the transform temporaries
    and the scalar and one-qubit cases. Both final reshapes are views.

    Work uses numerical-entry, character and coefficient-record visits,
    with indexing bookkeeping bundled into its operation. It is a size
    law rather than a CPU-time or hardware-FLOP count. Bytes cover explicit
    array and logical table payloads, excluding Python container overhead
    and process RSS. Other caller-owned arrays are charged by their owner.
    The transform omits exact zeros and performs no threshold pruning.

    These units are admission proxies, not timings or equal-cost CPU
    operations. The law is the block identity of
    operators._pauli.pauli_coefficients counted by
    _pauli_decomposition_requirements.

    Args:
        dimension: Positive power-of-two order d after any selected padding.
        max_bytes: Limit on this stage's known simultaneous payload bytes.
        max_select_work: Shared decomposition and SELECT work limit.

    Returns:
        The work limit minus the two-transform and extraction reservation.
        Work and bytes equal to their limits are admitted.
    """
    from nwqlib.operators.access import _check_bytes
    work, peak_bytes = _pauli_decomposition_requirements(dimension)
    _check_bytes(peak_bytes, max_bytes, "LCHS Pauli decomposition")
    max_select_work = integer(max_select_work, "max_select_work", 1)
    if work > max_select_work:
        raise ValueError(
            f"LCHS Pauli decomposition needs {work} units of max_select_work, above the limit "
            f"{max_select_work}; it is the first stage charged to LCHS(max_select_work=...), "
            f"so this stage needs that field to be at least {work}")
    return max_select_work - work


# This serves LCHS only. QPE's controlled powers make their own identity and
# pruning selection in algorithms/qpe/powers.py.
def _combined_trotter_terms(
    decomposition: _TrotterPauliDecomposition,
    *,
    k_value: float,
) -> _CombinedTrotterTerms:
    """Form H + k L, separate its identity phase and record the coefficient mass removed by
    pruning.

    The identity coefficient c_I depends on k through k*tr(L)/2**q, so
    exp(-i*t*c_I) differs between branches. It is a relative phase inside
    the LCU sum, not a global phase, and it is kept as a per-branch phase.
    Pauli coefficients of the Hermitian L and H are real up to decomposition
    rounding. An identity coefficient whose imaginary part exceeds
    _PAULI_COEFFICIENT_ATOL raises here. Other labels are checked where their
    rotation angles are formed.

    This function owns the node's dropped mass. The cutoff test runs before
    the identity is separated, so an identity coefficient it drops belongs
    in the mass. The mass is a finite binary64 upper bound on the sum of
    abs(real(c)) + abs(imag(c)) over precisely the dropped stored
    coefficients, accumulated upward (error_budget.upper_dropped_mass); that
    sum bounds the sum of their moduli. It is not the exact sum of complex
    magnitudes.
    """
    return combine_aligned(decomposition.aligned_rows, k_value, _PAULI_COEFFICIENT_ATOL)


def combine_aligned(rows, k, cutoff) -> _CombinedTrotterTerms:
    """Combine and prune one k-node from the aligned L/H rows, in their lexical order.

    For each (label, h, l, has_l, is_identity) row the value is
    complex(h + k*l) when L has the label and complex(h) otherwise, the
    original operand order, so an H-only row is not multiplied by an
    artificial zero and an L-only row keeps the addition to 0.0. A value
    whose modulus is at most the cutoff is dropped; a kept identity is the
    real branch phase coefficient; other kept values are the traceless
    terms. The kept coefficients and their order equal those of combining
    the dictionaries and sorting the union. The dropped values stream into
    error_budget.upper_dropped_mass, whose result is a certified upper bound
    on the summed moduli of exactly the dropped stored coefficients.

    Work per node, in scalar visits (_coefficient_preparation_work): two
    arithmetic visits per L row, at most six conversion, comparison,
    emission and reduction visits per union row, one tuple-copy visit per
    emitted row and four scalar identity checks, 2*m_L + 7*u + 4, plus the
    mass stream's 16*u + 4. These units are admission proxies, not timings
    or equal-cost CPU operations.
    """
    from nwqlib.subroutines.trotterization.error_budget import upper_dropped_mass

    kept, pruned, kept_indices, pruned_indices = [], [], [], []
    identity = 0.0

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
    )


def _coefficient_preparation_work(q, m_l, m_h, union, nodes):
    """Return (W_align, W_nodes): alignment once and combination of ``nodes`` distinct k-nodes.

    With s = m_L + m_H and u the union size including identity,
    W_align = (q+4)*s + (q+2)*u and each node costs 2*m_L + 7*u + 4
    combination visits plus 16*u + 4 for its streamed upper dropped mass
    (upper_dropped_mass, at most 16 visits per dropped coefficient and four
    per node), so W_nodes = K*(2*m_L + 23*u + 8). The host selection charges
    W_coeff = W_dec + W_align + W_nodes. Zero nodes add no node work. These
    units are admission proxies, not timings or equal-cost CPU operations.
    """
    s = m_l + m_h
    return (q + 4)*s + (q + 2)*union, nodes*(2*m_l + 23*union + 8)


def _node_key(k_value) -> str:
    """Return the node-cache key of one k-node, its exact stored binary64 value.

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
    elapsed time; no pruning charge is stored under the node alone.
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
    The first term is the existing LCHS per-node label/value envelope,
    counted once per distinct node; the second reserves each node's scalar
    bound coefficient and metadata. A stored W_up has at most 5376 magnitude
    bits under the finite coefficient_up reductions, and its two integer
    components take less than 1536 bytes, within that scalar allowance. This
    is an explicit native-object engineering allowance, not a heap bound.
    """
    return 128*nodes*(union_count+1)*(num_qubits+1) + 8192*nodes


@dataclass(frozen=True, kw_only=True)
class LCHSProductFormulaNodes:
    """The selected product-formula coefficient table, stored once per exact k-node.

    Store the selected coefficient table once per k node and the elapsed
    time, step count and route per application. Angles are not stored:
    they do not determine the coefficients at zero elapsed time and do not
    recover the discarded coefficients or their certified mass, so a
    fixed-step Plan keeps this table even when no certificate was requested.

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
            selected lexical order), ``coefficients`` (read-only complex128,
            the values combine_aligned returned, parallel to union_indices),
            ``identity_coefficient`` (binary64 generator coefficient c_I, so
            an application's identity phase is -elapsed*c_I),
            ``pruned_l1_mass`` (binary64 upper dropped mass d_up) and
            ``pruned_indices`` (read-only int64 diagnostic dropped rows,
            including the identity if it was dropped).

    The complex coefficients keep the imaginary-residue checks of the action
    and census consumers. The identity coefficient and the upper mass are
    the saved binary64 values, never recomputed from pruned_indices, angles
    or rounded moduli.
    """

    num_qubits: int
    labels: tuple[str, ...]
    keys: tuple[str, ...]
    grid_to_node: np.ndarray
    nodes: tuple[Mapping[str, Any], ...]


def _selected_node_table(decomposition, combined_nodes, k_values) -> LCHSProductFormulaNodes:
    """Return the stored node table of the combined nodes over the quadrature grid k_values.

    Each node's arrays are formed from the kept and dropped row positions
    that combine_aligned recorded, in their selected order, and made
    read-only; the labels are the decomposition's aligned union. Every grid
    position must name a combined node.
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
        nodes.append(MappingProxyType(dict(
            union_indices=frozen(combined.kept_indices, np.int64),
            coefficients=frozen([value for _, value in combined.traceless_terms], np.complex128),
            identity_coefficient=float(combined.identity_coefficient),
            pruned_l1_mass=float(combined.pruned_l1_mass),
            pruned_indices=frozen(combined.pruned_indices, np.int64),
        )))
    return LCHSProductFormulaNodes(
        num_qubits=decomposition.num_qubits, labels=labels, keys=keys,
        grid_to_node=frozen([position[key] for key in grid_keys], np.int64), nodes=tuple(nodes))


def _stored_node_table_bytes(table: LCHSProductFormulaNodes) -> int:
    """Return the logical payload of a stored node table, counted once where it is held.

    q*u for the shared labels, B_nodes = 24*sum_k p_k + 128*K for the node
    coefficient and union-index rows and their scalar metadata, and the
    selected records 8*sum_k d_k for the diagnostic pruned indices and
    8*N_grid for the grid-to-node mapping. Borrowing the table copies nothing.
    """
    return _node_table_bytes(table.num_qubits, len(table.labels),
        [len(node["union_indices"]) for node in table.nodes],
        [len(node["pruned_indices"]) for node in table.nodes], len(table.grid_to_node))


def _node_table_bytes(q, union_count, kept_counts, dropped_counts, grid_count) -> int:
    """Return q*u + 24*sum_k p_k + 128*K + 8*sum_k d_k + 8*N_grid (_stored_node_table_bytes)."""
    return (q*union_count + 24*sum(kept_counts) + 128*len(kept_counts)
            + 8*sum(dropped_counts) + 8*grid_count)


def stored_node_read_work(kept_counts, *, pruned_counts=()):
    """Logical acquisition visits for distinct resident selected nodes.

    W_read = sum_k (p_k + 2) over the nodes actually read in one invocation,
    with p_k the node's selected nonidentity rows. One borrowed row visit
    includes index/reference association. Two scalar visits read c_I and
    d_up per node. Optional diagnostic rows cost one visit each.
    Conversion, validation, copying and downstream arithmetic have separate
    charges. These units are admission proxies, not timings or equal-cost
    CPU operations.
    """
    return sum(p + 2 for p in kept_counts) + sum(pruned_counts)


def borrowed_node_rows(shared_labels, node):
    """Yield existing selected label/coefficient associations without recoding."""
    for index, coefficient in zip(
        node["union_indices"], node["coefficients"], strict=True
    ):
        yield shared_labels[index], coefficient


def borrowed_node_scalars(node):
    """Return the saved generator identity coefficient and upper dropped mass."""
    return node["identity_coefficient"], node["pruned_l1_mass"]


class _BorrowedRows:
    """A repeatable sized view of one stored node's (label, coefficient) rows.

    Each iteration borrows the resident arrays again (borrowed_node_rows),
    so a consumer that checks emptiness and traverses its input more than
    once, such as host_pf._rotation_sequence, pays no copy.
    """

    __slots__ = ("labels", "node")

    def __init__(self, labels, node):
        self.labels, self.node = labels, node

    def __len__(self):
        return len(self.node["union_indices"])

    def __iter__(self):
        return borrowed_node_rows(self.labels, self.node)


def stored_pf_nodes(data) -> LCHSProductFormulaNodes:
    """Return the stored node table of a product-formula LCHSData, host or SELECT."""
    table = (data.host_actions["nodes"] if data.host_actions is not None
             else None if data.select_data is None else data.select_data.nodes)
    if type(table) is not LCHSProductFormulaNodes:
        raise ValueError("this LCHS Plan has no stored product-formula node table")
    return table


def _lchs_census_choice(p, q, nodes, E, N, *, order, max_work, max_bytes, census_held,
                        limit_owner, work_field, field_work_offset=0,
                        linear_and_other_work=0, work_used=0):
    """Choose the shared census variant and block for ``nodes`` node contractions, or refuse.

    One ordered structure over the p shared nonidentity labels serves every
    distinct active k-node. With w = ceil(q/64), P = p(p-1)/2, E pairs, N
    nested tests and F <= N triples,

        G = wP + 1_D*wN,   V = E (order 1), p+2E (order 2, relaxed_prefix),
                               E+F (order 2, exact_census),

    and the shared work is G + K_n*V: census_work prices one structural
    scan plus one node contraction, and each further node adds V. The
    structure bytes are census_bytes with F_cap = J before pairs or N after
    pairs for exact_census at order 2 and zero otherwise, and census_held
    counts the LCHS populations that coexist with the census, never the
    census's own buffers. exact_census is chosen when its complete work and
    storage fit, otherwise relaxed_prefix at order 2; the result is
    (variant, block, need_work, census bytes). linear_and_other_work
    carries the preparation and reserved later work charged by the same
    limit, and work_used the prior noncensus work. These units are
    admission proxies, not timings or equal-cost CPU operations.

    Node restriction bytes (_node_bound_coefficients). With E, F the shared
    pair/triple counts, E', F' a node's counts and b the actual block, the
    shared law (error_budget.census_bytes) is

        B0 = B_held + 80p + 16pw + 32E + 48F + H_chunks + max(R_row, C_contract),
        w = ceil(q/64),  H_chunks = 384[p+1+I_exact2(E+1)],
        R_row = 32pw+64p+32w+65536,  C_contract = 56p+64b+48H(b)+65536.

    32E+48F already funds the builders' overlapping original chunks and
    concatenated tables; after the builders finish only 16E+24F of shared
    table data remains. Boolean selection of rows from a two-dimensional
    array also creates an intp index array, 8E' or 8F' bytes on the checked
    64-bit NumPy stack, in addition to the selected output. A pair-mask
    expression can transiently hold three E-byte Boolean arrays, a
    triple-mask expression three F-byte arrays, and the mask remains while
    the row-index array and selected output coexist. The simultaneously
    live table/selection bytes are 16E+24F for an all-kept node, and for a
    partial node 16E+24F+max(3E, E+24E') at pair selection,
    16E+24F+16E'+max(3F, F+32F') at triple selection and at most
    16E+24F+16E'+24F' plus the contraction law at remap/contraction. The
    pair stage is at most 41E+24F, the triple stage at most 32E+57F and the
    contraction at most 32E+48F. For the pair stage the excess over
    32E+48F is at most 9E-24F <= 9E, for the triple stage at most 9F, and
    the stages are sequential, hence the maximum. A simple sufficient
    addition to the complete B0, reusing its funded double-table storage, is

        R_restrict = 9 max(E, F),   B_LCHS = B0 + R_restrict.

    It is a sufficient envelope, not the minimum possible allocation law.
    Nodes are sequential, so the maximum over nodes applies, never their
    sum. The choice has only E and the cap F_cap = N before the triple
    census, so it adds 9*max(E, F_cap) to census_held at every candidate,
    refusal candidates included; for E = F = 0 it adds zero. Position
    construction, the kept mask and the magnitude arrays fit the existing
    80p input and 56p contraction allowances, and the remap's iterator
    overhead belongs to H0.

    Block. The largest fitting block of choose_census_block is replaced by
    65536 when it exceeds 65536 and the complete envelope at 65536 also
    fits, and the recomputed 65536 envelope is returned. Lowering b reduces
    64b but can increase 48H(b), so the clamp checks the entire law. The
    block changes how the outward partial sums are grouped: W_up remains an
    upper bound under coefficient_up's arithmetic premises but need not be
    bitwise equal to the result at another block, and a separate node
    census agrees at the same b because filtering preserves row order. On
    F = N label sets (two commuting classes that anticommute across, every
    nested test succeeding) with two nodes contracted under max_bytes =
    10**10, the admitted bytes at q = 6, 7, 8 were 6,412,128, 28,330,064
    and 190,055,472 at blocks 48,640, 65,536 and 65,536, against traced
    peaks of 3,194,384, 19,435,392 and 153,234,976 with every label kept
    and 4,246,451, 22,177,139 and 178,333,379 when the first node omits the
    first label and the second node the last (CPython 3.12.14, NumPy
    2.5.2). This qualifies the selected implementation on the checked
    CPython/NumPy stack with the shared H0 convention; it is not a
    process-RSS bound or a qualification of another NumPy implementation.

    A refusal names the limit that governs it as ``limit_owner(work_field=...)``
    and ``limit_owner(max_bytes=...)``. field_work_offset is the work of
    that field charged before ``max_work`` was handed here (the field's
    value is max_work + field_work_offset), so each stated requirement is a
    value of the field itself.
    """
    from nwqlib.subroutines.trotterization.error_budget import (
        census_bytes, census_sizes, census_work, choose_census_block,
    )

    if nodes < 1:
        raise ValueError("no active nodes need a census")
    P, _ = census_sizes(p)
    word_count = (q+63)//64
    variants = ("exact_census",) if order == 1 else ("exact_census", "relaxed_prefix")
    rejected = []
    for variant in variants:
        exact2 = order == 2 and variant == "exact_census"
        F_cap = N if exact2 else 0
        G = word_count*(P+(N if exact2 else 0))
        first = census_work(p, q, order=order, variant=variant,
                            pairs=E, nested=N if exact2 else 0, triples=F_cap)
        V = first-G
        need_work = work_used + linear_and_other_work + first + (nodes-1)*V
        # One node's Boolean selection (masks, intp row index, selected
        # rows) beyond the double-table reserve of census_bytes.
        reserve = 9*max(E, F_cap)
        held = census_held + reserve
        fit = choose_census_block(p, q, E, F_cap, max_bytes,
                                  order=order, variant=variant, held=held)
        if need_work <= max_work and fit is not None:
            block, size = fit
            clamped = census_bytes(p, q, E, F_cap, 65536, order=order,
                                   variant=variant, held=held)
            if block > 65536 and clamped <= max_bytes:
                block, size = 65536, clamped
            return variant, block, need_work, size
        rejected.append(_census_refusal_part(
            variant, need_work, max_work, fit is not None,
            census_bytes(p, q, E, F_cap, 1, order=order, variant=variant, held=held),
            max_bytes, census_held, reserve, limit_owner, work_field, field_work_offset))
    raise ValueError(
        "LCHS census has no admitted variant: " + "; ".join(rejected) + ". These numbers "
        "admit the census stage only; later LCHS stages are admitted separately"
    )


def _census_refusal_part(variant, need_work, max_work, bytes_fit, block1_bytes, max_bytes,
                         census_held, reserve, limit_owner, work_field, field_work_offset):
    """Return one refused census candidate's requirement in the governing Method fields.

    The byte figure is the block-1 envelope, one checked candidate of
    choose_census_block, not the smallest envelope over its candidates.
    """
    reasons = []
    if need_work > max_work:
        reasons.append(f"raise {limit_owner}({work_field}=...) to at least "
                       f"{need_work+field_work_offset} (now {max_work+field_work_offset})")
    if not bytes_fit:
        held = f"{census_held} bytes held by LCHS"
        if reserve:
            held += f" and a {reserve}-byte node restriction reserve"
        reasons.append(f"no checked contraction block fits {limit_owner}(max_bytes={max_bytes}), "
                       f"and the checked block-1 envelope is {block1_bytes} bytes including {held}")
    return f"{variant}: " + " and ".join(reasons)


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
    max_work: int,
    max_bytes: int,
    census_held: int,
    limit_owner: str = "LCHS",
    work_field: str = "max_select_work",
    field_work_offset: int = 0,
    work_used: int = 0,
    linear_and_other_work: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return each listed node's Pauli-triangle W_up from one admitted shared structure.

    The shared basis ``labels`` is the sorted union of the nonidentity
    labels of L and H (_union_nonidentity_labels), and node_terms maps each
    key to its kept (label, coefficient) rows in that order: a combined
    node's traceless_terms, or the borrowed rows of a stored node table
    (_BorrowedRows). Deleting zero-magnitude (pruned) labels preserves the relative order
    of the remaining labels, so restricting the union's pair and triple rows
    to a node's kept labels evaluates the same expression as a separate
    census on that node's kept terms (CSTWZ doi:10.1103/PhysRevX.11.011020,
    Eqs. (120)-(121), via error_budget.coefficient_up). The restriction
    keeps row order, so the outward reductions are those of that separate
    census at the same block.

    Admission (_lchs_census_choice) happens before any packing: the
    complete conservative candidates use E = P and N = J; if neither fits,
    the pair-only stage is admitted with the order-one pair law and refused
    before packing if even that fails. After the pairs exist, the actual E
    and N = nested_test_count select exact_census or relaxed_prefix before
    any triple test, against the work of all K_n node contractions and the
    bytes of the cached combined nodes in census_held. The shared pairs and
    triples are released after the last contraction.

    Restriction. A node keeping every label borrows the shared pair and
    triple tables. Otherwise its rows are the Boolean-masked copies of the
    shared rows, remapped in place with
    ``np.take(position, sub, out=sub, mode="clip")``. NumPy's default
    ``mode="raise"`` buffers the full output even with ``out`` (7,200,328
    temporary bytes for a 7,200,000-byte ``sub`` on NumPy 2.5.2, against
    232 bytes with ``mode="clip"``). Clipping changes no
    index here because the masks select only kept labels, so every selected
    old index lies in [0, p) and position[old_index] lies in the new node
    range. The node's copies are deleted after its evaluation is recorded,
    so the previous node's copies are not live while the next node's are
    built. _lchs_census_choice admits the selection populations as
    9*max(E, F) bytes on top of the shared census law, reusing that law's
    double-table reserve.

    This count assumes no Boolean temporary elision. The gathered masks are
    owning, writable, C-contiguous one-byte `np.bool_` arrays. NumPy 2.5.2's
    release-build elision threshold is 262,144 bytes, with 32 bytes in a
    `Py_DEBUG` build. Reuse also requires an exact uniquely referenced array,
    compatible shape and casting, and approval of the interpreter call stack.
    Reaching the size threshold alone does not guarantee reuse. The remap uses
    matching native eight-byte integer arrays and `mode="clip"` on indices
    already proved in range.

    Refusals name ``limit_owner(work_field=...)`` and
    ``limit_owner(max_bytes=...)``, the limits that govern this census
    (LCHS(max_select_work=...), the default, for SELECT selection and the
    classical host actions of host_pf, LCHSRefinement(
    max_structural_work=...) for fixed-step refinement). field_work_offset
    is the work of that field charged before ``max_work`` was handed here.

    Work: p*q label packing plus G + K_n*V, returned as ledger["work"]; the
    caller adds its own preparation and application work. The first listed
    node's evaluation carries the structure's pair and nested tests, the
    others zero. These units are admission proxies, not timings or
    equal-cost CPU operations.

    Returns:
        ({key: _BoundCoefficientEvaluation}, ledger) with ledger keys
        ``work``, ``bytes``, ``variant``, ``block``, ``pair_checks``,
        ``nested_checks``, ``structures`` and ``contractions``.
    """
    from nwqlib.subroutines.trotterization import error_budget as eb

    keys = tuple(keys)
    ledger = dict(work=0, bytes=0, variant=None, block=None, pair_checks=0, nested_checks=0,
                  structures=0, contractions=0)
    if not keys:
        return {}, ledger
    labels = tuple(labels)
    p, q = len(labels), num_qubits
    packing = p*q
    P, J = eb.census_sizes(p)
    try:
        _lchs_census_choice(p, q, len(keys), P, J, order=order, max_work=max_work,
                            max_bytes=max_bytes, census_held=census_held, limit_owner=limit_owner,
                            work_field=work_field, field_work_offset=field_work_offset,
                            linear_and_other_work=linear_and_other_work+packing, work_used=work_used)
    except ValueError:
        # Pair-only stage: masks and the complete pair scan, priced by the
        # order-one census law with E = P.
        pair_work = (work_used + linear_and_other_work + packing
                     + eb.census_work(p, q, order=1, variant="exact_census", pairs=P))
        pair_fit = eb.choose_census_block(p, q, P, 0, max_bytes, order=1,
                                          variant="exact_census", held=census_held)
        if pair_work > max_work or pair_fit is None:
            block1 = eb.census_bytes(p, q, P, 0, 1, order=1, variant="exact_census",
                                     held=census_held)
            requirement = _census_refusal_part(
                "pair stage", pair_work, max_work, pair_fit is not None, block1, max_bytes,
                census_held, 0, limit_owner, work_field, field_work_offset)
            raise ValueError(
                f"LCHS census exceeds its allowance at the {requirement}. These figures admit "
                "the pair stage only; the complete census and later LCHS stages are admitted "
                "separately"
            ) from None
    x, z = eb.pack_labels(labels, q)
    pairs, pair_checks = eb.pair_structure(x, z)
    nested = eb.nested_test_count(pairs, p) if order == 2 else 0
    variant, block, need_work, need_bytes = _lchs_census_choice(
        p, q, len(keys), len(pairs), nested, order=order, max_work=max_work, max_bytes=max_bytes,
        census_held=census_held, limit_owner=limit_owner, work_field=work_field,
        field_work_offset=field_work_offset, linear_and_other_work=linear_and_other_work+packing,
        work_used=work_used)
    triples, nested_checks = None, 0
    if order == 2 and variant == "exact_census":
        triples, nested_checks = eb.triple_structure(x, z, pairs)
    del x, z
    index = {label: position for position, label in enumerate(labels)}
    evaluations: dict[str, Any] = {}
    first = True
    for key in keys:
        rows = node_terms[key]
        kept = np.zeros(p, dtype=bool)
        magnitudes = np.zeros(p, dtype=np.float64)
        for label, coefficient in rows:
            value = complex(coefficient)
            # Same absolute imaginary-residue admission as the census's
            # operator validation (error_budget._validated_terms).
            if abs(value.imag) > _PAULI_COEFFICIENT_ATOL:
                raise ValueError(
                    "Hamiltonian coefficients must be real for Trotter error bounds; "
                    f"term {label!r} has coefficient {value} (build the operator from "
                    "real coefficients or simplify away numerical imaginary parts)"
                )
            kept[index[label]] = True
            magnitudes[index[label]] = abs(value.real)
        node_pairs, node_triples = pairs, triples
        if not kept.all():
            position = np.cumsum(kept, dtype=np.intp) - 1
            node_pairs = pairs[kept[pairs[:, 0]] & kept[pairs[:, 1]]]
            # Every selected index is a kept label, so clip changes none and
            # avoids take's full output buffer of mode="raise".
            np.take(position, node_pairs, out=node_pairs, mode="clip")
            if triples is not None:
                node_triples = triples[
                    kept[triples[:, 0]] & kept[triples[:, 1]] & kept[triples[:, 2]]
                ]
                np.take(position, node_triples, out=node_triples, mode="clip")
            del position
        coefficient = eb.coefficient_up(magnitudes[kept], node_pairs, order=order, variant=variant,
                                        triples=node_triples, block=block)
        evaluations[key] = eb._BoundCoefficientEvaluation(
            coefficient=coefficient,
            pauli_term_count=len(rows),
            pair_commutation_checks=pair_checks if first else 0,
            nested_commutation_checks=nested_checks if first else 0,
            bound_variant=variant,
            coefficient_arithmetic=eb.COEFFICIENT_ARITHMETIC,
        )
        first = False
        del node_pairs, node_triples
    ledger.update(work=need_work - work_used - linear_and_other_work, bytes=need_bytes,
                  variant=variant, block=block, pair_checks=pair_checks,
                  nested_checks=nested_checks, structures=1, contractions=len(keys))
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
    """Select steps for an already admitted node coefficient against the remainder R.

    At R = 0 only an exactly zero formula numerator (W_up = 0 or t = 0) is
    accepted, with the existing one-step convention for a zero theorem term
    of a nonempty generator; otherwise PruningBudgetExhausted is raised. A
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
            coefficient_arithmetic=evaluation.coefficient_arithmetic,
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
    """Select one application of a node against the PF budget left after its published pruning bound.

    Dropping Pauli terms whose moduli sum to at most d_up
    (combined.pruned_l1_mass) changes the node evolution exp(-i*t*(k*L+H))
    by at most |t|*d_up in operator norm: Duhamel's formula gives
    ||exp(-i*t*X) - exp(-i*t*Y)|| <= |t|*||X - Y|| for Hermitian X and Y,
    and ||X - Y|| <= d_up because every Pauli string has unit norm. The
    published pruning bound is P = up(|t|*d_up)
    (error_budget.upper_pruning_error). If P exceeds the allowance epsilon
    the node is refused; otherwise R = down(epsilon - P)
    (error_budget._remaining_budget_after_pruning), the node's cached W_up
    selects the smallest step count whose bound E_r = W_up*|t|**(o+1)/r**o
    is at most R, E = up(E_r) <= R, and the published total
    T = up(P + E) (error_budget.upper_combined_error) satisfies
    |t|*d + E_r <= P + E <= T <= epsilon. The order argument, the Method's
    trotter_order, selects the Lie-1 bound (CSTWZ
    doi:10.1103/PhysRevX.11.011020, Prop. 9, Eq. (120)) or the Suzuki-2
    bound (Prop. 10, Eq. (121)).

    Args:
        combined: The node's shared combined terms (_distinct_combined_nodes).
        evaluation: The node's admitted W_up (_node_bound_coefficients), with
            zero checks on reuse (_first_use); None when no term is kept.
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
        coefficient_arithmetic=selection.coefficient_arithmetic,
    )


def _fixed_trotter_certificate_records(
    *,
    nodes: LCHSProductFormulaNodes,
    applications: Sequence[tuple[float, Sequence[int]]],
    method: LCHS,
    dense_validation: bool = False,
    max_bytes=DEFAULT_MAX_BYTES,
    max_dense_work=100_000_000,
    max_structural_work=100_000_000,
    max_steps=100_000,
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
    k-node of the stored node table (LCHSProductFormulaNodes) is bounded
    once (_node_bound_coefficients) and each application evaluates it at its
    own (elapsed_time, step_count). Pauli commutator triangle sums bound the
    spectral relation; optional dense evaluation checks those sums, not a
    new selected circuit, and its dense coefficient is also formed once per
    node.

    Fixed-step refinement evaluates the saved step counts using the saved
    coefficients and pruning mass. The selected coefficient table is read
    once per distinct k-node: a borrowed selected row costs one logical
    visit, and reading the identity coefficient and upper dropped mass
    costs two visits per node, W_read = sum_k (p_k + 2)
    (stored_node_read_work). There is no alignment, combination, pruning or
    upper-mass computation in a read. The table remains in the consuming
    phase's held payload (_stored_node_table_bytes); nothing is copied.

    Each record publishes P = up(|t|*d_up), the formula bound
    E = up(W_up*|t|**(p+1)/r**p) and T = up(P + E)
    (error_budget.upper_pruning_error, upper_combined_error); the saved
    steps are kept, so T may exceed a requested allowance.

    Structural work, with p_k a node's kept count, K distinct nodes and M
    applications: W_read + p*q + G + sum_nodes(16*p_k*(q+1) + V_k)
    + 16*M*K. The 16*p_k*(q+1) term is the consumer's conservative
    preparation allowance for the census magnitudes and labels, once per
    node; it is not a decomposition or a second stored-row acquisition.
    Bytes: the held payload and the stored table, plus the census structure
    (_lchs_census_choice). No decomposition is charged here. These units
    are admission proxies, not timings or equal-cost CPU operations. All
    node work accumulates in the supplied counters, and first evaluations
    carry the census tests.

    Returns:
        One record list per application, in quadrature-position order.
    """
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.subroutines.trotterization import error_budget as trotter_error_budget

    from nwqlib._linalg_laws import singular_values_work
    from nwqlib.operators.access import _check_bytes
    max_dense_work = integer(max_dense_work, "max_dense_work", 1)
    max_structural_work = integer(max_structural_work, "max_structural_work", 1)
    max_steps = integer(max_steps, "max_steps", 1)
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
        if any(step > max_steps for step in steps):
            raise ValueError("fixed certificate steps exceed max_steps")
        schedules.append((elapsed_time, steps))
    q = nodes.num_qubits
    keys = tuple(nodes.keys[int(index)] for index in grid)
    table = dict(zip(nodes.keys, nodes.nodes, strict=True))
    # Work: one acquisition of each distinct node's stored rows and scalars.
    read_work = stored_node_read_work(len(node["union_indices"]) for node in nodes.nodes)
    if counts["structural_work"]+read_work > max_structural_work:
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
    if counts["structural_work"]+label_work+scalar_work > max_structural_work:
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
            # Bytes: 24 complex128 d-by-d arrays per term (term matrices,
            # partial sums and commutator products).
            peak = 384*p*d*d
            # Work per term: 2 matrix products and 1 spectral norm (order 1) or
            # 6 products and 2 norms (order 2), d**3 per product, plus 24*d**2
            # entrywise operations. Each norm is np.linalg.norm(ord=2) in
            # trotterization/error_budget.py::_spectral_norm, which computes
            # singular values alone (_linalg_laws.singular_values_work). The
            # counters below record the same populations.
            products = 2 if order == 1 else 6
            work = products*p*d**3+order*p*singular_values_work(d)+24*p*d*d
            _check_bytes(held+peak, max_bytes, "fixed PF dense commutator arrays")
            if counts["dense_work"]+work > max_dense_work:
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
                    coefficient_arithmetic=evaluation.coefficient_arithmetic,
                )
            )
        results.append(records)
    return tuple(results)


def _lchs_trotter_error_budget_per_node(method: LCHS) -> float:
    """Return the per-node synthesis allowance, 0.1 times approximation_tolerance.

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


def _trotter_budget_quadrature_terms(
    selections: list[_LCHSTrotterNodeRecord | TrotterStepSelection],
    *,
    coefficients: np.ndarray,
    method: LCHS,
) -> dict[str, Any]:
    """Recorded budget terms for the error-budgeted backend's selections.

    Single arithmetic owner of the recorded schedule and bound terms. The
    SELECT plan, the classical Pauli actions and explicit refinement pass
    their per-node records here, so equal node records give equal values.
    The synthesis bound sum_j |c_j|*e_j is the triangle inequality over the
    LCU sum, with e_j the node's PF plus pruning bound. It is an operator-norm
    bound on the ideal finite sum before physical input and PSD recovery
    weighting, which solution_error_budget applies once.

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
            coefficient_arithmetic=selection.coefficient_arithmetic,
        )
        for selection in selections
    ]
    bound_values = [record.combined_bound_value for record in records]
    evaluated = all(bound is not None for bound in bound_values)
    dense_bound_values = [record.dense_bound_value for record in records]
    dense_evaluated = all(bound is not None for bound in dense_bound_values)
    trotter_synthesis_error_bound = (
        float(
            fsum(
                abs(coefficient) * bound
                for coefficient, bound in zip(
                    coefficients,
                    bound_values,
                    strict=True,
                )
                if bound is not None
            )
        )
        if evaluated
        else None
    )
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
    if resolve_hamiltonian_evolution_backend(method) == "trotter_error_budgeted":
        terms["trotter_error_budget_per_node"] = _lchs_trotter_error_budget_per_node(method)
    return terms


@dataclass(frozen=True, kw_only=True)
class _PreparedDecomposition:
    """Cartesian parts after padding and the PSD decision.

    Attributes:
        l_part: Encoded L, shifted by psd_shift*I when a shift was selected.
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

    With ``parts=False`` only the coefficient endpoints are needed: one
    adjoint and L are formed, the adjoint is released before the solve, no
    H, padded or shifted array is formed, and l_part and h_evolution are
    None. The endpoints, padding zero and PSD rule are the same.

    One Hermitian eigensolve of L supplies both endpoints. The numerical PSD
    rule (solution_error_budget._numerical_psd_decision) either accepts L, or
    shifts it by s = -lambda_min + window, or raises when make_l_psd is False.
    Because L is Hermitian, max(|lambda_min|, |lambda_max|) after the shift is
    its spectral norm up to eigensolver rounding, and it is the ||L|| used by
    the quadrature selector. Callers restore the physical growth exp(s*T).
    """

    array = as_square_matrix(matrix)
    if parts:
        l_part, h_evolution = _cartesian_parts(array)
    else:
        adjoint = np.conjugate(array.T, order="C")
        l_part = np.add(array, adjoint)
        del adjoint
        l_part *= 0.5
        h_evolution = None
    if not np.any(l_part):
        min_l_eigenvalue = max_l_eigenvalue = 0.0
    else:
        if max_spectral_work is not None and matrix.shape[0]**3 > max_spectral_work:
            raise ValueError(
                f"LCHS eigensolve exceeds max_spectral_work: it needs {matrix.shape[0]**3} work units "
                f"(d**3 with d={matrix.shape[0]}), max_spectral_work={max_spectral_work}. "
                f"Raise max_spectral_work to at least {matrix.shape[0]**3}.")
        eigenvalues = np.linalg.eigvalsh(l_part)
        min_l_eigenvalue = float(eigenvalues[0].real)
        max_l_eigenvalue = float(eigenvalues[-1].real)
    dimension = matrix.shape[0]
    target_dimension = dimension if padded_dimension is None else integer(
        padded_dimension, "padded_dimension", dimension)
    if not parts:
        l_part = None
    if target_dimension > dimension:
        # A_pad=A⊕0 implies L_pad=L⊕0 and H_pad=H⊕0 (ACL arXiv:2312.03916v2,
        # Eq. (4)).
        # Add the known zero eigenvalue to the original endpoints; no second
        # eigensolve of the padded matrix. The PSD shift then acts on every
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
    conversion = {
        "min_l_eigenvalue_before_psd_conversion": min_l_eigenvalue,
        "min_l_eigenvalue_after_psd_conversion": min_l_eigenvalue_after,
        "psd_correction_applied": psd_shift > 0.0,
        "psd_shift": float(psd_shift),
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
    """Quadrature-plan scalars shared by dense terms and representative sampling.

    This carries everything dense term realization needs except the term
    matrices, so planning and representative resource sampling can reuse the
    node grid and counts (``M``, ``Q``, interval counts) without constructing
    all ``M`` matrices.

    Attributes:
        l_part: N-by-N Hermitian L after the selected PSD conversion and any recorded
            dimension padding, used in the node generator k*L+H.
        h_part: Matching Hermitian H used in exp(-i*T*(k*L+H)).
        conversion: Decomposition, dimension and PSD-shift metadata; the original
            physical propagator requires exp(shift*T) compensation when shift is positive.
        h1: Actual Gauss panel width in k, or None for a non-Gauss/unitary rule.
        range_k: Selected finite cutoff scale recorded by the provider.
        effective_range_k: Realized cutoff used by the provider's component bounds;
            it need not equal the largest interior Gauss node magnitude.
        interval_count_each_side: Gauss panels per half-axis, symmetric-uniform integer
            half-range J, or zero for signed/unitary rules.
        node_count: Gauss points per panel Q, or total points for non-Gauss rules.
        total_node_count: Mathematical node count M, excluding later LCU address padding.
        k_nodes: Real array of shape (M,) in the provider's stored node order.
        coefficients: Complex array of shape (M,) before physical PSD-shift compensation.
        coefficient_l1_norm: Sum of these uncompensated coefficient magnitudes.
        l_norm: Numerical spectral-norm estimate of the prepared L used to select the
            grid; no additional eigensolve is implied by reading it.
        lambda_min_before_psd_conversion: Computed smallest Hermitian-L eigenvalue before
            the selected shift, retaining the original dissipativity information.
        numerical_psd_premise_satisfied: Outcome of numerical PSD admission for the
            prepared matrix under its scale-aware tolerance, not a formal PSD certificate.
        coefficient_plan: Same provider configuration, node/weight, coefficient and
            scoped-bound owner from which these numerical arrays were obtained.
    """

    l_part: np.ndarray
    h_part: np.ndarray
    conversion: Mapping[str, Any]
    h1: float | None
    range_k: float
    effective_range_k: float
    interval_count_each_side: int
    node_count: int
    total_node_count: int
    k_nodes: np.ndarray
    coefficients: np.ndarray
    coefficient_l1_norm: float
    l_norm: float
    lambda_min_before_psd_conversion: float
    numerical_psd_premise_satisfied: bool
    coefficient_plan: LCHSCoefficientPlan


def _quadrature_data_from_plan(
    *,
    prepared: _PreparedDecomposition,
    final_time: float,
    method: LCHS,
    max_bytes=DEFAULT_MAX_BYTES,
    max_quadrature_work=100_000_000,
) -> LCHSQuadratureData:
    """Assemble quadrature data from its resolved decomposition and provider plan."""

    if prepared.l_norm == 0.:
        from .providers import _unitary_coefficient_plan
        coefficient_plan = _unitary_coefficient_plan(method)
    else:
        coefficient_plan = _resolve_coefficient_context(
        lchs_kernel=method.lchs_kernel,
        k_quadrature=method.k_quadrature,
        problem_context=LCHSProblemContext(
            final_time=float(final_time),
            epsilon=float(method.approximation_tolerance),
            l_norm=prepared.l_norm,
        ),
        max_bytes=max_bytes,
        max_quadrature_work=max_quadrature_work,
        )
    quadrature_plan = coefficient_plan.quadrature
    return LCHSQuadratureData(
        l_part=prepared.l_part,
        h_part=prepared.h_evolution,
        conversion=prepared.conversion,
        h1=quadrature_plan.h1,
        range_k=quadrature_plan.range_k,
        effective_range_k=quadrature_plan.effective_range_k,
        interval_count_each_side=quadrature_plan.interval_count_each_side,
        node_count=quadrature_plan.node_count,
        total_node_count=quadrature_plan.physical_node_count,
        k_nodes=np.asarray(coefficient_plan.nodes, dtype=float),
        coefficients=np.asarray(coefficient_plan.coefficients, dtype=complex),
        coefficient_l1_norm=coefficient_plan.coefficient_l1_norm,
        l_norm=prepared.l_norm,
        lambda_min_before_psd_conversion=prepared.lambda_min_before,
        numerical_psd_premise_satisfied=prepared.numerical_psd_premise_satisfied,
        coefficient_plan=coefficient_plan,
    )


def generate_lchs_quadrature(
    *,
    matrix: np.ndarray,
    final_time: float,
    method: LCHS,
    max_bytes=DEFAULT_MAX_BYTES,
    padded_dimension=None,
    max_spectral_work=100_000_000,
    max_quadrature_work=100_000_000,
) -> LCHSQuadratureData:
    """Generate the homogeneous LCHS quadrature plan without term unitaries.

    Steps: A = L + iH (ACL arXiv:2312.03916v2, Eqs. (3)-(4)), optional zero
    padding to padded_dimension, the numerical PSD decision and shift, then
    the provider pair's nodes and coefficients for T and ||L||. No
    exp(-i*T*(k*L+H)) is formed. The returned coefficients do not yet
    include exp(shift*T).
    """

    from nwqlib.operators.access import _check_bytes
    dimension = matrix.shape[0]
    target_dimension = dimension if padded_dimension is None else integer(
        padded_dimension, "padded_dimension", dimension)
    # Enclosing law 64*D**2 + 128*D: A, L and the conjugate reused as H hold
    # at most 48 d**2 bytes, sequential padding peaks at 48 d**2 + 16 D**2
    # and then 32 d**2 + 32 D**2, and the eigenvalues-only solve on the
    # original dimension adds its d-square working copy, so four complex
    # D-square arrays bound the known arrays; 128*D covers the O(D) diagonal
    # and spectral vectors. The queried eigensolver workspace Q_N(d) is not
    # charged (a declared known-array workspace, not a complete cap).
    _check_bytes(64*target_dimension**2 + 128*target_dimension, max_bytes, "LCHS Cartesian/PSD arrays")
    max_spectral_work = integer(max_spectral_work, "max_spectral_work", 1)
    prepared = _prepare_decomposition(
        matrix=matrix,
        method=method,
        padded_dimension=target_dimension,
        max_spectral_work=max_spectral_work,
    )
    return _quadrature_data_from_plan(
        prepared=prepared,
        final_time=final_time,
        method=method,
        max_bytes=max_bytes,
        max_quadrature_work=max_quadrature_work,
    )


def lchs_quadrature_summary(
    quadrature_data: LCHSQuadratureData,
    *,
    method: LCHS,
    he_backend: str,
    final_time: float,
) -> dict[str, Any]:
    """Assemble the shared homogeneous LCHS quadrature dictionary.

    K, Q, M, h1 and the component bounds come from the provider record.
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
        "effective_K": quadrature_data.effective_range_k,
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
            including any selected PSD-shift compensation.
        plan: Actual compact product-formula occurrence tables, branch mapping and
            phases consumed by SELECT construction and its resource laws.
        quadrature: Shared provider, conversion and per-node synthesis metadata;
            reading these records does not construct the SELECT circuit.
        numerical_psd_premise_satisfied: Outcome of the shared prepared-L numerical
            PSD admission rule under its configured tolerance.
        nodes: Stored selected coefficient table once per k-node
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
    coefficients and compared entrywise, and any failed premise is recorded
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
    max_bytes=DEFAULT_MAX_BYTES,
    max_select_work=100_000_000,
    grid_k_values: Sequence[float] | None = None,
    work_ledger: dict | None = None,
) -> tuple[LCHSProductFormulaSelectPlan, tuple[_LCHSTrotterNodeRecord, ...], LCHSProductFormulaNodes]:
    """Build the shared homogeneous/inhomogeneous product-formula plan.

    A supplied ``work_ledger`` mapping receives, under ``"work"``, the
    max_select_work this selection charged (its selection, census and
    angle-table work), so the caller can carry the remainder to later stages.

    The plan tables cover the power-of-two SELECT slots. The returned records
    follow the supplied slot tables one-to-one, so callers weight each record
    by the coefficient of the same slot. The returned stored node table
    (LCHSProductFormulaNodes) keeps each distinct node's selected
    coefficients once, over the quadrature grid (grid_k_values for a source
    layout, otherwise the positions that branch_to_node names), for later
    verification and refinement. Added padding slots have zero
    coefficient and zero steps and stay inside the plan.

    For B compressed repetition blocks, W template occurrences and S padded
    slots, the float64 angle table contains E = B*W*S entries. The fill loop
    checks each angle for finiteness and canonicalizes signed zero. A private
    ownership transfer freezes the filled buffer and publishes its read-only
    view, so table construction and storage each require 8*E data bytes
    beside selection_bytes. Optional affine validation adds at most three
    float64 S-entry row arrays, giving 8*E + 24*S for that phase. These are
    known array-data terms, not a process RSS bound.

    Each slot j evolves k_j*L + H for its elapsed time t_j with r_j steps of
    the Lie-1 or Suzuki-2 template. Steps are fixed (trotter) or selected per
    slot against the per-node budget (trotter_error_budgeted). The angle of
    occurrence (P, m) at slot j is m*t_j*c_P(k_j)/r_j, where m is the Qiskit
    template multiplier. It is 2 for every Lie-1 term. In Suzuki-2 it is 2 for
    the middle label and 1 for each of the two occurrences of every other
    label. The stored value is therefore the RZ angle, twice the exponent.
    A slot whose r_j repetitions are used up gets angle zero (see
    _repetition_blocks).
    """
    from nwqlib.algorithms.lchs.select_synthesis import (
        LCHSProductFormulaSelectPlan,
        product_formula_occurrence_template,
    )

    he_backend = resolve_hamiltonian_evolution_backend(method)
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
    # one-time support alignment W_align, sum_nodes((q+1)*u + 16*u + 4) for
    # each node's combination allowance and streamed upper dropped mass
    # (_coefficient_preparation_work) and 16*S before the census, then
    # p*q + G + K_n*V for the admitted shared census when budgeted
    # (_node_bound_coefficients). These units are admission proxies, not
    # timings or equal-cost CPU operations.
    align_work, _ = _coefficient_preparation_work(
        q, len(decomposition.l_coefficients), len(decomposition.h_coefficients), terms, 0)
    selection_work = (align_work + len(distinct)*((q+1)*terms + 16*terms + 4)
                      + 16*padded_node_count)
    # Zero is a legal budget here. It is what _admit_pauli_decomposition
    # leaves when the decomposition used the whole limit, and every selection
    # then exceeds it.
    max_select_work = integer(max_select_work, "max_select_work", 0)
    # Planning hands this builder the remainder of method.max_select_work
    # after the Pauli decomposition (parameters.py and
    # generate_lchs_product_formula_select_plan), so the difference is the
    # field's work charged before this selection. A direct call with an
    # unrelated limit gets figures relative to that limit.
    charged_before = method.max_select_work - max_select_work
    if selection_work > max_select_work:
        raise ValueError(
            f"product-formula branch selection needs {selection_work} units of max_select_work, "
            f"above the {max_select_work} left after {charged_before} already charged; this stage "
            f"needs LCHS(max_select_work=...) to be at least {selection_work+charged_before} "
            f"(now {method.max_select_work})")
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
        census_keys = tuple(dict.fromkeys(
            key for key, elapsed_time in zip(node_keys, padded_times, strict=True)
            if key is not None and elapsed_time != 0.0 and combined_nodes[key].traceless_terms
        ))
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
                # when one branch generator is zero; that branch's angles are zero.
                step_count=int(method.trotter_steps),
                selection=None,
            )
        if max_steps is not None and record.step_count > max_steps:
            raise ValueError("selected product-formula steps exceed the declared circuit envelope")
        slot_records.append(record)

    branch_step_counts = tuple(record.step_count for record in slot_records)
    max_step_count = max(branch_step_counts, default=0)
    block_starts, block_repetitions = _repetition_blocks(branch_step_counts)
    node_coefficients = {key: dict(combined.traceless_terms) for key, combined in combined_nodes.items()}
    term_coefficients = [{} if key is None else node_coefficients[key] for key in node_keys]
    table_entries = len(occurrence_template)*len(block_repetitions)*padded_node_count
    # Selection stores W*B*L angles (W template occurrences, B blocks, L
    # slots) in one float64 array, 8 bytes per stored angle. Emitting the W*R
    # occurrences of all R repetitions happens later, at lowering. The table
    # is frozen by ownership transfer, so construction and storage both hold
    # 8 bytes per angle; affine row checking adds three S-entry float64 rows.
    angle_peak_bytes = 8 * table_entries
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
    if table_work > max_select_work:
        raise ValueError(
            f"product-formula angle-table selection needs {table_work} units of max_select_work, "
            f"above the {max_select_work} left after {charged_before} already charged; this stage "
            f"needs LCHS(max_select_work=...) to be at least {table_work+charged_before} "
            f"(now {method.max_select_work})")
    angle_tables = np.zeros(
        (len(block_starts) * len(occurrence_template), padded_node_count),
        dtype="<f8",
    )
    row = 0
    for repetition in block_starts:
        for label, multiplier in occurrence_template:
            for slot, (coefficients_by_label, elapsed_time, steps) in enumerate(zip(
                term_coefficients,
                padded_times,
                branch_step_counts,
                strict=True,
            )):
                coefficient = complex(coefficients_by_label.get(label, 0.0))
                if abs(coefficient.imag) > _PAULI_COEFFICIENT_ATOL:
                    raise ValueError(
                        "Hamiltonian coefficients must be real for product-formula "
                        f"synthesis; term {label!r} has coefficient {coefficient}"
                    )
                angle = (
                    multiplier * elapsed_time * coefficient.real / steps
                    if repetition < steps and steps > 0
                    else 0.0
                )
                if not isfinite(angle):
                    raise ValueError("FrozenArray float64 elements must be finite")
                angle_tables[row, slot] = angle + 0.0
            row += 1
    angle_tables = FrozenArray._from_owned_canonical_array(angle_tables)
    plan = LCHSProductFormulaSelectPlan(
        pauli_labels=pauli_labels,
        occurrence_schedule=occurrence_schedule,
        occurrence_angle_tables=angle_tables,
        occurrence_block_repetitions=block_repetitions,
        branch_step_counts=branch_step_counts,
        max_step_count=max_step_count,
        identity_phases=tuple(
            float(-elapsed_time * combined.identity_coefficient) if node is not None else 0.0
            for combined, elapsed_time, node in zip(
                combined_terms,
                padded_times,
                padded_branch_to_node,
                strict=True,
            )
        ),
        coefficient_phases=tuple(float(np.angle(value)) for value in padded_coefficients),
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
    matrix: np.ndarray | None = None,
    final_time: float,
    method: LCHS,
    _quadrature_plan: LCHSQuadratureData | None = None,
    max_steps: int | None = None,
    max_bytes=DEFAULT_MAX_BYTES,
    max_select_work=100_000_000,
    work_ledger: dict | None = None,
) -> LCHSProductFormulaSelectData:
    """Build one common product-formula SELECT plan for a homogeneous circuit.

    A supplied ``work_ledger`` mapping receives, under ``"work"``, the
    max_select_work charged by the Pauli decomposition and the selection.

    Every k-node evolves for the full time T. The LCU coefficients returned
    include exp(shift*T), divided by 2**psd_recovery_exponent(shift, T) when
    that factor exceeds binary64. The recorded synthesis bound weights each
    node's bound by the uncompensated |c_j|, because solution_error_budget
    applies the PSD recovery once per application. ``matrix`` is needed only
    when no selected quadrature plan is supplied; otherwise the plan's
    encoded L and H give the dimension.
    """

    he_backend = resolve_hamiltonian_evolution_backend(method)
    # A valid dense_exact or QSP Method must not receive a product-formula plan.
    if he_backend not in ("trotter", "trotter_error_budgeted"):
        raise ValueError(
            "product-formula SELECT planning serves the product-formula "
            f"backends ('trotter', 'trotter_error_budgeted'); got {he_backend!r}"
        )
    if matrix is None and _quadrature_plan is None:
        raise ValueError("product-formula SELECT planning needs a matrix or a selected quadrature plan")
    dimension = (matrix if _quadrature_plan is None else _quadrature_plan.l_part).shape[0]
    if not is_power_of_two(dimension):
        raise ValueError(
            "product-formula SELECT planning requires a matrix dimension that is a power of "
            "two (plan() pads a non-power-of-two LinearDynamics automatically)"
        )
    quadrature_data = _quadrature_plan or generate_lchs_quadrature(
        matrix=matrix, final_time=final_time, method=method,
        max_bytes=method.max_bytes, max_spectral_work=method.max_spectral_work,
        max_quadrature_work=method.max_quadrature_work,
    )
    selection = {}
    remaining_work = _admit_pauli_decomposition(dimension, max_bytes=max_bytes,
                                                max_select_work=max_select_work)
    decomposition = _trotter_pauli_decomposition(
        quadrature_data.l_part,
        quadrature_data.h_part,
    )
    shift = quadrature_data.conversion["psd_shift"]
    operator_scale = (
        psd_recovery_part(shift, final_time, psd_recovery_exponent(shift, final_time))
        if shift > 0.0
        else 1.0
    )
    coefficients = operator_scale * np.asarray(
        quadrature_data.coefficients,
        dtype=complex,
    )
    plan, slot_records, node_table = _build_product_formula_select_plan(
        decomposition=decomposition,
        k_values=np.asarray(quadrature_data.k_nodes, dtype=float),
        elapsed_times=np.full(quadrature_data.total_node_count, final_time),
        coefficients=coefficients,
        branch_to_node=tuple(range(quadrature_data.total_node_count)),
        method=method,
        address_structure=quadrature_data.coefficient_plan.quadrature.address_structure,
        max_steps=max_steps,max_bytes=max_bytes,max_select_work=remaining_work,work_ledger=selection,
    )
    if work_ledger is not None:
        work_ledger["work"] = max_select_work-remaining_work+selection["work"]
    quadrature = {
        **lchs_quadrature_summary(
            quadrature_data,
            method=method,
            he_backend=he_backend,
            final_time=final_time,
        ),
        **_trotter_budget_quadrature_terms(
            slot_records,
            coefficients=quadrature_data.coefficients,
            method=method,
        ),
    }
    return LCHSProductFormulaSelectData(
        coefficients=coefficients,
        plan=plan,
        quadrature=quadrature,
        numerical_psd_premise_satisfied=(quadrature_data.numerical_psd_premise_satisfied),
        nodes=node_table,
    )


__all__ = [
    "LCHSProductFormulaSelectData",
    "LCHSQuadratureData",
    "cartesian_decomposition",
    "generate_lchs_product_formula_select_plan",
    "generate_lchs_quadrature",
    "lchs_quadrature_summary",
]
