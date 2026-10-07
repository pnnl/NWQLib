"""Actual original spectrum, polynomial selection and bounded classical work."""

from dataclasses import dataclass, replace
from math import ceil, fsum, log
import numpy as np
from nwqlib._linalg_laws import hermitian_eigensystem_work, singular_values_work
from nwqlib._numerics import binary_scaled_matrix
from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib._validation import finite_real
from nwqlib.blocks.encoding import _select_planned_encoding
from nwqlib.blocks.selection import SelectedBlock
from nwqlib.core.records import Float64, InputRef
from nwqlib.ir import Binding
from nwqlib.operators import ingest_dense
from nwqlib.operators.access import _check_bytes, refuse_known_need
from nwqlib.problems.inputs import ingest_vector
from .constants import QLS_POLYNOMIAL_KAPPA_FLOOR, QLS_SPECTRAL_PREMISE_RTOL, QLS_TARGET_MARGIN
from .norm_search import _ceil_log2
from .primary_records import InversePolynomial, ReflectionPolynomial, QLSWork


@dataclass(frozen=True)
class OriginalSVD:
    """Original-input singular frames, computed once and reused.

    Dense quantum selection computes this SVD of the original ``A`` once.
    Endpoint selection, dense completion and the archive reuse it, so no
    padded or repeated decomposition is needed. Classical dense QLS acquires
    it for a general original ``A`` when the selected inverse or linear norm
    model needs factors (``_original_factors``). ``require_source`` rejects
    frames that belong to a different input, including a same-shape one,
    and frames that are writable.

    Attributes:
        source: Input reference of the original ``A``.
        left: Left singular vectors ``U``, read-only.
        singular: Singular values in descending order, in the original scale.
        right_h: Conjugate-transposed right singular vectors ``V^dagger``,
            read-only.
    """

    source: InputRef
    left: np.ndarray
    singular: np.ndarray
    right_h: np.ndarray

    def require_source(self, operator):
        d = operator.basis.dimension
        if (
            self.source != operator.reference
            or self.left.shape != (d, d)
            or self.singular.shape != (d,)
            or self.right_h.shape != (d, d)
            or any(value.flags.writeable for value in (self.left, self.singular, self.right_h))
        ):
            raise ValueError("selected SVD differs from its original input binding")

    method = "original_svd"
    calls = "svd_calls"

    def endpoints(self):
        """Return the original ``(sigma_min, sigma_max)`` of the stored singular values."""
        return float(self.singular[-1]), float(self.singular[0])

    def frames(self):
        """Return ``(left, values, right_h)`` for ``numerical.inverse_polynomial_action``."""
        return self.left, self.singular, self.right_h


@dataclass(frozen=True)
class OriginalEigensystem:
    """Signed eigenvalues and eigenvectors of an exactly Hermitian original ``A``, acquired once.

    Classical dense QLS acquires this factorization for a Hermitian original
    when the selected inverse or linear norm model needs factors
    (``_original_factors``). Planning endpoints, the inverse polynomial
    action and the norm-model weights reuse it. Signed eigenvalues are
    needed for the odd inverse. ``require_source`` rejects factors that
    belong to a different input and factors that are writable.

    Attributes:
        source: Input reference of the original ``A``.
        values: Signed eigenvalues in ascending order, in the original scale.
        vectors: Eigenvectors ``V`` as columns, read-only.
    """

    source: InputRef
    values: np.ndarray
    vectors: np.ndarray
    method = "original_eigh"
    calls = "eigh_calls"

    def require_source(self, operator):
        d = operator.basis.dimension
        if (
            self.source != operator.reference
            or self.values.shape != (d,)
            or self.vectors.shape != (d, d)
            or any(value.flags.writeable for value in (self.values, self.vectors))
        ):
            raise ValueError("selected eigensystem differs from its original input binding")

    def endpoints(self):
        """Return ``(min |lambda|, max |lambda|)``, the original singular endpoints."""
        magnitudes = np.abs(self.values)
        return float(np.min(magnitudes)), float(np.max(magnitudes))

    def frames(self):
        """Return ``(left, values, right_h)`` for ``numerical.inverse_polynomial_action``.

        ``right_h`` is None: the right frame of a Hermitian eigensystem is
        ``vectors`` itself, so no conjugate-transposed copy is stored.
        """
        return self.vectors, self.values, None


def original_factor_laws(d, hermitian):
    """Return ``(F, P, B_F)`` of one original full factorization of a d-square ``A``.

    Classical QLS charges an original spectral factorization at the stage
    that acquires it and reuses its immutable factors for the inverse
    action or linear norm model:

    ========================  ================  ===================  ====================
    Original factorization    Work ``F(d)``     Persistent ``P(d)``  Acquisition ``B_F(d)``
    ========================  ================  ===================  ====================
    General full SVD          ``8 d**3``        ``32 d**2 + 8 d``    ``96 d**2 + 32 d``
    Hermitian full ``eigh``   ``8 d**3+32 d**2``  ``16 d**2 + 8 d``  ``64 d**2 + 32 d``
    ========================  ================  ===================  ====================

    The Hermitian acquisition envelope has four complex matrix slots for
    scaling, driver-layout input and returned eigenvectors, plus vector
    scratch. The general envelope is the law of ``_original_svd``.
    Both exclude native LAPACK workspace. The Hermitian work is
    ``_linalg_laws.hermitian_eigensystem_work(d)``.
    """
    if hermitian:
        return hermitian_eigensystem_work(d), 16 * d * d + 8 * d, 64 * d * d + 32 * d
    return 8 * d**3, 32 * d * d + 8 * d, 96 * d * d + 32 * d


def _original_eigensystem(operator, method):
    """Compute the one admitted ``eigh`` of an exactly Hermitian original dense ``A``.

    Bytes and work of ``original_factor_laws(d, hermitian=True)`` are
    admitted before the call. The matrix is scaled by a power of two for the
    decomposition, and the signed eigenvalues are scaled back exactly.
    Nonfinite output is rejected, and the factors are made read-only.
    """
    d = operator.basis.dimension
    work, _, data = original_factor_laws(d, True)
    _check_bytes(data, method.max_bytes, "original QLS eigensystem arrays")
    if work > method.max_work:
        raise ValueError(
            f"original QLS eigensystem exceeds max_work: it needs {work} work units for dimension {d}, "
            f"QLS.max_work={method.max_work}. Raise QLS.max_work to at least {work}.")
    matrix, exponent = binary_scaled_matrix(operator.dense_array())
    values, vectors = np.linalg.eigh(matrix)
    np.ldexp(values, exponent, out=values)
    for array in (values, vectors):
        if not np.isfinite(array).all():
            raise ValueError("original QLS eigensystem produced nonfinite data")
        array.flags.writeable = False
    return OriginalEigensystem(operator.reference, values, vectors)


def _original_factors(operator, method):
    """Acquire the original factorization that classical factor consumers reuse.

    Exact admitted Hermiticity metadata (``operator.structure``) selects one
    full ``eigh``. Any other dense ``A`` takes ``_original_svd``.
    """
    if operator.structure == "hermitian":
        return _original_eigensystem(operator, method)
    return _original_svd(operator, method)


def classical_factor_consumer(method):
    """Whether a classical evaluation needs original factors: the inverse or the linear norm model."""
    return method.solver == "qsvt_inverse" or method.encoded_solution_norm_estimate == "linear_kappa_sequence"


def _original_svd(operator, method):
    """Compute the one admitted SVD of the original dense ``A``.

    Bytes (``96 d^2 + 32 d``) and work (``8 d^3``) of
    ``original_factor_laws(d, hermitian=False)`` are admitted before the
    call. The matrix is scaled by a power of two for the decomposition, and
    the singular values are scaled back exactly. Nonfinite output is
    rejected, and the frames are made read-only so a Plan cannot mutate
    them later.
    """
    d = operator.basis.dimension
    # 96 d^2 bytes are six complex128 d x d slots, an allowance for the
    # power-of-two-scaled copy, its component-maximum temporaries, the SVD's
    # input copy, U and V^dagger. 32 d covers the singular values. 8 d^3 is
    # the SVD work law of dense completion (docs/ENGINEERING_CONSTANTS.md,
    # DEFAULT_MAX_BLOCK_WORK).
    work, _, data = original_factor_laws(d, False)
    _check_bytes(data, method.max_bytes, "original QLS SVD arrays")
    if work > method.max_work:
        raise ValueError("original QLS SVD exceeds max_work")
    matrix, exponent = binary_scaled_matrix(operator.dense_array())
    left, singular, right = np.linalg.svd(matrix)
    np.ldexp(singular, exponent, out=singular)
    for array in (left, singular, right):
        if not np.isfinite(array).all():
            raise ValueError("original QLS SVD produced nonfinite data")
        array.flags.writeable = False
    return OriginalSVD(operator.reference, left, singular, right)


def _admit_query_synthesis(method, original, dimension, *, encoding=None, rhs=None):
    """Charge the exact syntheses that the controlled queries make, and Qiskit's control of them, to ``max_work`` and ``max_bytes``.

    The dense dilation of the encoded ``dimension``-square operator is one
    ``UnitaryGate`` on m = log2(dimension) + 1 qubits. A controlled query
    (``quantum._query_circuit``) passes it to ``qiskit_compat.controlled``
    on the route that ``method.dense_control_route`` selects for its
    controls. The gate-wise route replaces it by
    ``_dense_synthesis.dense_unitary_circuit`` before Qiskit controls each
    synthesized gate, and the whole-matrix route synthesizes the controlled
    matrix on m + k qubits for k controls (``controlled_synthesis_size``).
    The inverse of a non-Hermitian
    ``A``, through its Hermitian dilation, and ``shortcut_native_svp`` call
    the query of ``A`` with one control, and ``shortcut_dilation`` with two,
    each forward and adjoint. Lowering builds each of these two
    specializations once per Run and reuses it (``blocks/lowering.py``), so
    two syntheses of distinct matrices, and on the gate-wise route two
    control steps, are charged,
    their work summed and their circuits kept together. A supplied native
    encoding (``encoding.native``) takes the gate-wise route on every
    ``dense_control_route`` and is charged from the dense unitaries that
    ``qiskit_compat.dense_synthesis_widths`` and ``dense_control_counts``
    find in its circuit.

    The shortcuts also control the RHS preparation, with the control count
    of the ``A`` query. ``shortcut_native_svp`` builds it forward and
    adjoint, and ``shortcut_dilation`` both again for each value of the
    dilation qubit. A supplied preparation circuit (``qiskit.supplied``) is
    charged for the dense unitaries in each of these specializations, on
    the gate-wise route. Other preparations and encodings hold no dense
    unitary.

    The inverse of a Hermitian ``A`` calls the query without controls, so its
    construction synthesizes nothing, and a backend that lowers the circuit
    to a gate basis charges the synthesis to the Run's
    ``max_synthesis_work``.
    """
    from nwqlib.subroutines._dense_synthesis import (admit_dense_syntheses, gatewise_control_counts,
                                                     select_dense_control_route)

    controls = 2 if method.solver == "shortcut_dilation" else 1
    widths, steps, controlled_widths = [], [], []
    if not (method.solver == "qsvt_inverse" and original.structure == "hermitian"):
        if encoding is not None and encoding.record.implementation.name == "encoding.native":
            from nwqlib.subroutines.qiskit_compat import dense_control_counts, dense_synthesis_widths

            circuit = encoding._payload.circuit
            widths += 2 * list(dense_synthesis_widths(circuit))
            steps += 2 * [dense_control_counts(circuit, controls)]
        elif encoding is None or encoding.record.implementation.domain == "dense_dilation":
            if select_dense_control_route(method.dense_control_route, controls) == "whole_matrix":
                controlled_widths += 2 * [dimension.bit_length() + controls]
            else:
                widths += 2 * [dimension.bit_length()]
                steps += 2 * [gatewise_control_counts(dimension.bit_length(), controls)]
    if method.solver != "qsvt_inverse" and rhs is not None and rhs.preparation.implementation == "qiskit.supplied":
        from nwqlib.subroutines.qiskit_compat import dense_control_counts, dense_synthesis_widths

        specializations = 2 if method.solver == "shortcut_native_svp" else 4
        widths += specializations * list(dense_synthesis_widths(rhs._native))
        steps += specializations * [dense_control_counts(rhs._native, controls)]
    admit_dense_syntheses(widths, controls=steps, controlled_qubit_counts=controlled_widths,
                          max_work=method.max_work, max_bytes=method.max_bytes,
                          operation="QLS controlled dense-query synthesis")


def _covers_spectral_estimate(bound, estimate):
    """Compare positive finite estimates without an absolute floor or overflow."""
    return estimate <= bound or (estimate - bound) / estimate <= QLS_SPECTRAL_PREMISE_RTOL


def _kappa_premise_refusal(route, alpha, alpha_meaning):
    """Refusal message for a route without original singular values, so ``kappa="auto"`` cannot apply.

    The encoded gap parameter is ``alpha / sigma_min(A)`` (``_spectrum``),
    so the message states that bound with the ``alpha`` the route knows.
    """
    value = "" if alpha is None else f" = {alpha:.6g}"
    return (f"{route} needs a numeric kappa in place of kappa='auto'. Pass QLS(kappa=K) with K at least "
            f"alpha / sigma_min(A), where alpha{value} is {alpha_meaning} and sigma_min(A) is the smallest "
            "singular value of A. This bound is not the condition number sigma_max(A) / sigma_min(A), which "
            "is smaller whenever alpha exceeds sigma_max(A).")


def input_access_refusal(operator, execution):
    """Refusal message for a CSR, CSC or classical Pauli ``A``, or None for another input.

    Classical QLS reads dense ``A`` only (``method._admit``). Quantum QLS
    reads dense, Pauli or periodic-stencil access of ``A``, or a supplied
    encoding bound to it (``select_inputs``). A dense copy holds all d**2
    entries, so the message offers one only for a matrix whose dense copy
    fits, and the dense route's own ``max_bytes`` and ``max_work`` limits
    still apply to it.
    """
    representation = operator.reference.representation
    d = operator.basis.dimension
    if representation == "pauli" and execution == "classical":
        return (f"classical QLS reads dense A only and has no route for a Pauli A (dimension {d}). If a dense "
                f"{d} x {d} copy fits within the max_bytes of QLS, pass the dense matrix of the Pauli sum, "
                "for a Qiskit SparsePauliOp A.to_matrix(). Otherwise plan quantum execution with QLS(kappa=...).")
    if representation not in {"csr", "csc"}:
        return None
    matrix = f"a {representation.upper()} matrix A (dimension {d})"
    dense = (f"If a dense {d} x {d} copy of A fits in memory, A.toarray() gives dense input, and the dense "
             "route then applies the max_bytes and max_work limits of QLS to it.")
    if execution == "classical":
        return f"classical QLS reads dense A only and has no route for {matrix}. {dense}"
    return (f"QLS has no route for {matrix}. Quantum QLS reads dense, Pauli or periodic-stencil access of A, "
            "or an encoding bound to A. Pass a Pauli sum as a Qiskit SparsePauliOp with QLS(kappa=...), the "
            "periodic operator m I + d (2I - S - S†), with S the cyclic shift and m > 0, as "
            "nwqlib.operators.PeriodicStencil(q, mass=m, diffusion=d), or an encoding bound to A as "
            f"QLS(encoding=...). {dense}")


def _spectrum(operator, method, *, alpha=None, factors=None):
    """Resolve alpha and the encoded singular-value gap from supplied premises or bounded original-matrix work.

    Endpoints come from already acquired original factors (the quantum
    dense SVD, or the classical SVD or Hermitian eigensystem that the
    selected inverse or linear norm model reuses), from one bounded dense
    eigenvalue or singular-value computation, or not at all when ``alpha``
    and ``kappa`` are both supplied premises. Compact inputs never acquire
    a spectrum here and need a numeric ``kappa``. With endpoints known,
    ``alpha`` defaults to ``sigma_max`` and ``kappa_be`` to
    ``max(1, alpha / sigma_min)``. Supplied premises must cover the
    endpoints within the relative window ``QLS_SPECTRAL_PREMISE_RTOL``.
    ``polynomial_kappa`` adds the separate domain floor
    ``QLS_POLYNOMIAL_KAPPA_FLOOR``. The original condition number
    ``sigma_max / sigma_min`` is reported without either change.

    When dense Hermitian endpoint selection is needed and no reusable
    factors are available, compute all eigenvalues of the binary-scaled
    original matrix with ``eigvalsh`` and take the minimum and maximum
    absolute values. Admission charges ``4*d**3 + 8*d**2`` work units and
    ``64*d**2 + 48*d`` known buffer bytes, excluding stored input and native
    LAPACK workspace. Exact admitted Hermiticity metadata selects this
    branch at every dimension, including ``d = 1``. General matrices keep
    the SVD-values law. The method string and ``spectral_calls`` name the
    factorization actually used.
    """
    events = dict(svd_calls=0, eigh_calls=0, eigvalsh_calls=0)
    lower = upper = None
    source = "user"
    if factors is not None:
        factors.require_source(operator)
        lower, upper = factors.endpoints()
        source, events[factors.calls] = factors.method, 1
    elif alpha is None and method.alpha == "auto" or method.kappa == "auto":
        if operator.reference.representation != "dense":
            raise ApplicabilityError(_kappa_premise_refusal(
                "QLS with a supplied encoding of non-dense A", alpha, "the normalization of that encoding"))
        from .numerical import _extreme_singular_values

        d = operator.basis.dimension
        hermitian = operator.structure == "hermitian"
        if hermitian:
            # 4 d^3 is the eigenvalues-only dense law. 8 d^2 covers component
            # maxima and reductions, binary scaling and lower-order endpoint visits
            # without a second Hermiticity test. Four complex matrix slots
            # cover the scaled matrix, input-layout copy and scaling/scan
            # frontiers, and three complex vector slots the eigenvalues,
            # their magnitudes and endpoint scratch. The caller's input and
            # opaque LAPACK workspace are excluded.
            work, data = 4 * d**3 + 8 * d * d, 64 * d * d + 48 * d
        else:
            # The SVD without vectors (_linalg_laws.singular_values_work)
            # plus 8 d^2 for the scaling and the magnitudes. Bytes, in
            # complex128 slots: 5 d^2, an allowance for the power-of-two-
            # scaled copy, its scaling temporaries and the SVD input, and
            # 3 d for the singular values.
            work, data = singular_values_work(d) + 8 * d * d, 16 * (5 * d * d + 3 * d)
        _check_bytes(data, method.max_bytes, "original QLS spectral arrays")
        if work > method.max_work:
            raise ValueError("original QLS spectral selection exceeds max_work")
        lower, upper, source = _extreme_singular_values(
            operator.dense_array(), hermitian=hermitian, events=events)
    if lower is not None:
        lower, upper = (
            finite_real(lower, "minimum singular value"),
            finite_real(upper, "maximum singular value"),
        )
        if lower <= 0 or upper <= 0:
            raise ValueError("QLS requires an invertible original matrix")
    alpha = (upper if method.alpha == "auto" else method.alpha) if alpha is None else alpha
    alpha = finite_real(alpha, "selected QLS alpha")
    if upper is not None and not _covers_spectral_estimate(alpha, upper):
        raise ValueError("supplied alpha is below the original numerical spectral norm")
    # The encoded gap parameter is alpha/sigma_min, which can differ from
    # the original matrix condition number sigma_max/sigma_min.
    # At a unitary endpoint both SVD estimates can round above alpha. Keep the
    # raw endpoints, but the encoded gap parameter's mathematical domain is >=1.
    required = None if lower is None else max(1., finite_real(alpha / lower, "selected QLS kappa"))
    kappa = required if method.kappa == "auto" else method.kappa
    if required is not None and not _covers_spectral_estimate(kappa, required):
        raise ValueError("supplied kappa does not cover the original selected spectrum")
    return dict(
        alpha=alpha,
        alpha_source="original_spectral_norm" if method.alpha == "auto" else "user",
        kappa_be=kappa,
        kappa_source="original_spectrum" if method.kappa == "auto" else "user",
        polynomial_kappa=max(kappa, QLS_POLYNOMIAL_KAPPA_FLOOR),
        condition_number=None if lower is None else upper / lower,
        sigma_min=lower,
        sigma_max=upper,
        spectral_method=source,
        spectral_calls=tuple(events.items()),
    )


def select_inputs(problem, method, *, execution):
    """Select an actual encoding while preserving original A,b and their scale.

    The routes are tried in this order. A supplied ``SelectedBlock`` must be
    bound to the original ``A`` and owns its ``alpha``. A compact periodic stencil uses
    ``select_periodic_encoding``. Dense input with power-of-two dimension
    and exactly Hermitian entries uses the requested or automatic family.
    Any other dense input uses the dense dilation, whose quantum route
    computes the original SVD once. Classical dense selection acquires the
    original SVD, or one ``eigh`` of a Hermitian original, only when the
    selected inverse or linear norm model needs factors and planning needs
    spectral endpoints (``classical_factor_consumer``). Endpoints then come
    from those factors, and the classical evaluation reuses them. Compact
    Pauli input keeps Pauli access and requires a numeric ``kappa``. A
    non-power-of-two dense system is padded with the positive block
    ``alpha * I`` and ``b`` with zeros, so the solution occupies the original
    coordinate slice and the padding adds no singular value below
    ``sigma_min`` or above ``alpha``. When the selected encoding's ``alpha``
    differs from the spectral estimate, ``kappa_be`` is recomputed from the
    encoding's ``alpha``. The original Problem is never changed.

    Returns:
        ``(encoding, encoded_operator, rhs, svd, spectrum)``: the selected
        encoding block, the operator it represents (padded when needed),
        the matching RHS, the original factors (``OriginalSVD``, or
        ``OriginalEigensystem`` for a classical Hermitian original) or
        ``None``, and the spectral record fields of ``QLSReconstruction``.
    """
    from nwqlib.subroutines.block_encoding.core import plan_block_encoding

    original, rhs = problem.A, problem.b
    rhs._require_data()
    d = problem.dimension
    padded = 1 << max(1, (d - 1).bit_length())
    svd = None
    scale = rhs.preparation.physical_scale
    if scale is None or scale.mantissa == 0:
        raise ApplicabilityError("QLS preparation requires nonzero physical b access")
    if rhs.preparation.blocker and not (d != padded and rhs.reference.representation == "vector"):
        raise ApplicabilityError(rhs.preparation.blocker)
    # A supplied encoding owns its normalization and must refer to original A.
    # Other routes select a concrete construction from the available input access.
    if method.encoding is not None:
        encoding = method.encoding
        if not isinstance(encoding, SelectedBlock) or encoding.record.implementation.name not in {
            "encoding.native",
            "encoding.planned",
        }:
            raise ValueError(
                "QLS encoding must be an actual original-input-bound selected encoding"
            )
        semantics = encoding.record.semantics
        if semantics.input != original.reference or semantics.basis != original.basis:
            raise ValueError("QLS encoding differs from original A")
        if d != padded:
            raise ApplicabilityError(
                "a supplied encoding needs its own coherent non-power-of-two extension"
            )
        if method.block_encoding_implementation != "auto":
            raise ValueError("supplied encoding conflicts with a constructor override")
        if method.alpha != "auto" and method.alpha != semantics.alpha:
            raise ValueError("supplied encoding and requested alpha differ")
        if (execution == "classical" and method.kappa == "auto" and classical_factor_consumer(method)
                and original.reference.representation == "dense"):
            # The encoding owns alpha, so only kappa needs endpoints. The
            # classical inverse or linear model reuses the same factors.
            svd = _original_factors(original, method)
        spectrum = _spectrum(original, method, alpha=semantics.alpha, factors=svd)
        spectrum["alpha_source"] = "supplied_encoding"
        encoded_operator = original
    elif original.reference.representation == "periodic_stencil":
        from .periodic import select_periodic_encoding

        encoding, spectrum = select_periodic_encoding(original, method)
        encoded_operator = original
    elif original.reference.representation == "dense":
        force_dense = d != padded or original.structure != "hermitian"
        selected = None
        if force_dense and method.block_encoding_implementation not in {"auto", "dense_dilation"}:
            raise ApplicabilityError(
                "general/padded QLS uses the explicitly normalized dense dilation"
            )
        # Before the first dense phase, the byte needs that the dimensions
        # and the selected route fix are combined by their maximum: the dense
        # intake 192D**2+32D (every dense route plans through it), the dense
        # quantum completion 384D**2+32D when it is selected, and the dense
        # Pauli transform 192D**2+64D**2(q+16) when pauli_lcu or
        # multiplexed_pauli is requested. Under auto the Pauli transform
        # depends on the circulant detection, so it is not included.
        known_bytes = 192 * padded * padded + 32 * padded
        if execution == "quantum" and (force_dense or method.block_encoding_implementation == "dense_dilation"):
            known_bytes = max(known_bytes, 384 * padded * padded + 32 * padded)
        elif method.block_encoding_implementation in ("pauli_lcu", "multiplexed_pauli"):
            qubits = padded.bit_length() - 1
            known_bytes = max(known_bytes, 192 * padded * padded + 64 * padded * padded * (qubits + 16))
        refuse_known_need("QLS", "max_bytes", method.max_bytes, known_bytes,
                          "the selected polynomial, norming grid and QSP phase solve")
        if execution == "quantum" and (force_dense or method.block_encoding_implementation == "dense_dilation"):
            # 384 p^2 + 32 p bytes and 9 p^3 work are the completion laws of
            # _admit_dense_input(construction=True, selected_svd=True) in
            # subroutines/block_encoding/core.py (DEFAULT_MAX_BLOCK_WORK
            # registry row). Checking them here makes a completion that the
            # native build would refuse fail before the original SVD runs.
            _check_bytes(384*padded*padded+32*padded, method.max_bytes,
                "selected dense QLS completion arrays")
            if 9*padded**3 > method.max_work:
                raise ValueError(
                    f"selected dense QLS completion exceeds max_work: it needs {9*padded**3} work units "
                    f"(9*p**3 with p={padded}), QLS.max_work={method.max_work}. "
                    f"Raise QLS.max_work to at least {9*padded**3}.")
            _admit_query_synthesis(method, original, padded)
            svd = _original_svd(original, method)
        elif execution == "quantum":
            from nwqlib.subroutines.block_encoding.core import _plan_block_encoding

            def dense_norm(matrix):
                nonlocal svd
                svd = _original_svd(original, method)
                return float(svd.singular[0])

            selected = _plan_block_encoding(original.dense_array(),
                implementation=method.block_encoding_implementation,
                normalization=None if method.alpha == "auto" else method.alpha,
                max_bytes=method.max_bytes, max_work=method.max_work, dense_norm=dense_norm)
            if selected.implementation == "dense_dilation" and svd is None:
                svd = _original_svd(original, method)
        elif classical_factor_consumer(method) and (method.alpha == "auto" or method.kappa == "auto"):
            # Compute original frames during planning only when planning
            # needs spectral information. Supplied alpha and kappa defer the
            # single factorization to the admitted classical evaluation.
            svd = _original_factors(original, method)
        spectrum = _spectrum(original, method, factors=svd,
            alpha=None if selected is None else selected.alpha)
        encoded_operator = original
        matrix = original.dense_array()
        # Pad A with positive alpha-valued dummy coordinates and b with zeros.
        # The original solution then occupies the same physical coordinate slice.
        if d != padded:
            # The padded complex matrix and its admitted snapshot (16 p^2
            # bytes each), and the padded RHS and its snapshot (16 p each).
            _check_bytes(
                32 * padded * padded + 32 * padded, method.max_bytes, "QLS positive dummy padding"
            )
            matrix = np.zeros((padded, padded), dtype=complex)
            matrix[:d, :d] = original.dense_array()
            matrix[d:, d:] = spectrum["alpha"] * np.eye(padded - d)
            encoded_operator = ingest_dense(matrix, max_bytes=method.max_bytes)
            vector = np.zeros(padded, dtype=complex)
            vector[:d] = rhs.physical_vector()
            rhs = ingest_vector(vector, max_bytes=method.max_bytes)
        family = "dense_dilation" if force_dense else method.block_encoding_implementation
        # 192 p^2 bytes is the matrix term of the dense intake law of
        # _admit_dense_input (192 D^2 + 32 D bytes), which
        # plan_block_encoding also applies to the padded matrix.
        _check_bytes(192 * padded * padded, method.max_bytes, "QLS encoding structural selection")
        if selected is None:
            selected = plan_block_encoding(
                matrix, implementation=family, normalization=spectrum["alpha"],
                max_bytes=method.max_bytes, max_work=method.max_work,
            )
        if selected.implementation == "dense_dilation":
            selected = replace(selected, source=encoded_operator.dense_array())
        if method.alpha != "auto" and selected.alpha != method.alpha:
            raise ValueError("selected encoding cannot realize the supplied alpha")
        actual = selected.alpha
        if actual != spectrum["alpha"]:
            required = max(1., actual / spectrum["sigma_min"])
            kappa = required if method.kappa == "auto" else method.kappa
            if not _covers_spectral_estimate(kappa, required):
                raise ValueError("supplied kappa does not cover the selected encoding")
            spectrum.update(kappa_be=kappa, polynomial_kappa=max(kappa, QLS_POLYNOMIAL_KAPPA_FLOOR))
        spectrum["alpha"], spectrum["alpha_source"] = actual, "selected_encoding"
        if (spectrum["sigma_max"] is not None
                and not _covers_spectral_estimate(actual, spectrum["sigma_max"])):
            raise ValueError("selected encoding alpha is below the original numerical spectral norm")
        # The selected original SVD belongs to the family payload. Native dense
        # completion consumes its analytic padding without another padded SVD.
        if isinstance(selected.source, np.ndarray):
            selected.source.flags.writeable = False
        encoding = _select_planned_encoding("base_encoding", selected, operator=encoded_operator,
            dense_svd_selected=execution == "quantum" and svd is not None)
    # Keep compact Pauli access and require a numeric gap premise. Do not
    # form a dense matrix merely to infer a condition number.
    elif "pauli_terms" in original.manifest.access:
        from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

        if method.kappa == "auto":
            # The Pauli encoding selected below has this coefficient 1-norm
            # as its alpha (subroutines/block_encoding/core.py,
            # _plan_block_encoding).
            alpha = fsum(abs(c) for c in original.pauli_terms().coefficients)
            raise ApplicabilityError(_kappa_premise_refusal(
                "compact Pauli QLS", alpha, "the Pauli coefficient 1-norm of A"))
        terms = original.pauli_terms().labels(max_bytes=method.max_bytes)
        decomposition = PauliDecomposition(
            terms=tuple(PauliTerm(label=label, coefficient=c) for label, c in terms),
            input_dimension=d,
            operator_dimension=d,
            num_qubits=(d - 1).bit_length(),
            atol=0.0,
        )
        family = (
            "multiplexed_pauli"
            if method.block_encoding_implementation == "auto"
            else method.block_encoding_implementation
        )
        if family != "multiplexed_pauli" and family != "pauli_lcu":
            raise ApplicabilityError(
                "compact Pauli QLS keeps Pauli access; select a bound encoding for another family"
            )
        selected = plan_block_encoding(decomposition, implementation=family,
            max_bytes=method.max_bytes, max_work=method.max_work)
        if method.alpha != "auto" and selected.alpha != method.alpha:
            raise ValueError("Pauli encoding differs from requested normalization")
        spectrum = _spectrum(original, method, alpha=selected.alpha)
        spectrum["alpha_source"] = "selected_encoding"
        encoding = _select_planned_encoding("base_encoding", selected, operator=original)
        encoded_operator = original
    else:
        raise ApplicabilityError(
            input_access_refusal(original, execution)
            or "QLS needs explicit dense/Pauli/periodic access or an original-bound encoding"
        )
    if (
        execution == "quantum"
        and encoding.record.construction_work is not None
        and encoding.record.construction_work > method.max_work
    ):
        raise ValueError("selected QLS encoding construction exceeds max_work")
    # Every controlled query is checked here. The query of an automatically
    # selected or supplied encoding, and a supplied RHS circuit, are checked
    # here for the first time. The explicit and general dense routes were
    # already checked before their SVD, without the supplied RHS, which this
    # check adds.
    if execution == "quantum":
        _admit_query_synthesis(method, original, encoded_operator.basis.dimension, encoding=encoding, rhs=rhs)
    return encoding, encoded_operator, rhs, svd, spectrum


def select_polynomial(method, kappa):
    """Return the selected ``InversePolynomial`` or ``ReflectionPolynomial`` for domain parameter ``kappa``.

    ``qsvt_inverse`` fits the odd inverse polynomial of
    ``_fit_inverse_chebyshev`` to the target ``epsilon_inv``. The shortcut
    solvers realize Dalzell's even kernel-reflection polynomial
    (arXiv:2406.12086v2) with ``eta = epsilon_inv / sqrt(2)``. Both records
    keep the unrescaled Chebyshev coefficients together with
    ``rescale = s``, the norming bound on ``sup |P|`` times
    ``1 + QLS_TARGET_MARGIN``. The phase solver targets ``P/s``, and the
    inverse route's physical recovery multiplies by ``s`` again. The
    shortcut routes return a unit direction, so there ``s`` enters only the
    success-mass prediction.
    """
    from nwqlib.subroutines.qsp.inverse import _fit_inverse_chebyshev
    from nwqlib.subroutines.qsp.shortcut import (
        plan_kernel_reflection,
        _realize_kernel_reflection,
        dalzell_eta_from_precision,
        _KR_CONSTRUCTION_GRID_POINTS,
    )
    from nwqlib.subroutines.qsp.phases import chebyshev_polynomial_sup_bound

    controls = dict(
        max_degree=method.max_degree, max_work=method.max_work, max_bytes=method.max_bytes
    )
    if method.solver == "qsvt_inverse":
        candidates = []
        fit = _fit_inverse_chebyshev(
            kappa, method.epsilon_inv, candidate_degrees=candidates, **controls
        )
        coefficients = fit.coefficients
        record = InversePolynomial
        fields = dict(
            candidate_degrees=tuple(candidates),
            certificate=fit.certificate,
            certificate_basis="affine_chebyshev_residual_norming",
            certificate_grid_points=fit.certificate_grid_points,
            lsq_node_count=fit.lsq_node_count,
        )
    else:
        kr = _realize_kernel_reflection(
            plan_kernel_reflection(kappa, dalzell_eta_from_precision(method.epsilon_inv)),
            **controls,
        )
        coefficients = kr.coefficients
        record = ReflectionPolynomial
        fields = dict(
            ell=kr.kr_ell,
            eta=kr.kr_eta,
            fit_points=max(_KR_CONSTRUCTION_GRID_POINTS, 8 * (kr.degree + 1)),
        )
    # Rescale the selected polynomial into the QSP amplitude domain, keeping
    # that factor for later recovery of the physical solution. With
    # s = (norming bound) * (1 + QLS_TARGET_MARGIN), max|P/s| < 1 on [-1, 1],
    # the boundedness condition of [MRTC] Theorems 9-10
    # (arXiv:2105.02859v5, App. A.1).
    rescale = chebyshev_polynomial_sup_bound(
        coefficients, max_degree=method.max_degree, max_bytes=method.max_bytes
    ) * (1 + QLS_TARGET_MARGIN)
    return record(coefficients=coefficients, rescale=rescale, **fields)


def search_envelope(kappa, source):
    """Return ``(rows, trials, queries)`` envelopes of a norm-search model before it runs.

    The formulas are those of ``norm_search.py`` (Dalzell arXiv:2406.12086v2,
    Eqs. (24), (25), (28) and (29), with ``eta = sqrt(1/8)`` for the noisy
    search and ``ln(2/0.025) = ln(80)`` for the linear sequence), evaluated
    from ``kappa`` alone so that planning can admit the model's work. Execution
    rejects realized accounting above this envelope. The ladder length uses
    the models' exact ``_ceil_log2``, so the envelope and the model count the
    same rungs even when ``kappa`` lies one ulp from a power of two.
    """
    steps = _ceil_log2(kappa)
    ladder = steps + 1
    if source == "grid":
        trials = ceil(100 * log(20 * ladder)) * ladder
        return ladder, trials, 0  # The grid model records trials only, with no query estimate.
    if source == "noisy_binary_search":
        rounds = ceil(log(ladder) / log(1.5))
        trials = ceil(72 * log(40 * rounds)) * rounds
        return rounds, trials, 2 * trials * ceil(kappa * log(2 / (1 / 8) ** 0.5) / 2)
    if source == "linear_kappa_sequence":
        base = ceil(100 * log(80))
        trials = 4 * base * steps * (steps + 1) // 2
        # Every step j has four candidates and overhead steps+1-j, so the
        # trials sum to 4*base*steps*(steps+1)/2. The factor 8 in the queries
        # is Eq. (25)'s factor 2 (U_A and U_A^dagger) times four candidates.
        queries = sum(
            8 * base * (steps + 1 - j) * ceil(2.0**j * log(80) / 2) for j in range(1, steps + 1)
        )
        return 4 * steps, trials, queries
    return 0, 0, 0


def selected_work(matrix, *, alpha, embedded, method, polynomial, kappa, t_source, output, outputs,
                  hermitian, factors_held):
    """Return the ``QLSWork`` record of one explicit classical model evaluation.

    Classical QLS charges an original spectral factorization at the stage
    that acquires it and reuses its immutable factors for the inverse
    action or linear norm model. Shortcut action separately factors its
    augmented projected matrix. Rank-one projector application has
    quadratic work, and Hermitian dilations are written into one allocated
    array. Workspace includes factors acquired during this invocation and
    excludes factors already owned by the Plan and opaque LAPACK workspace.

    Symbols: ``d`` is
    the original dimension, ``n = d*(2 if embedded else 1)`` the semantic
    system dimension, ``a = 2n`` for a shortcut and ``h = 2a`` for
    ``shortcut_dilation``, ``m`` the polynomial degree,
    ``s = ceil(log2(kappa))`` for the linear model and ``tgt`` one for a
    grid or noisy-search target solve. ``f`` is one only when a factor
    consumer (the inverse, or the linear norm model) needs factors that
    planning did not acquire (``factors_held``). ``F(d)``, ``P(d)`` and
    ``B_F(d)`` are ``original_factor_laws``. ``E(x) = 8 x**3 + 32 x**2`` is
    ``_linalg_laws.hermitian_eigensystem_work``. In the polynomial-visit
    units the selected kernels are

    - ``W_inverse = f F(d) + m d + 2 d**2``;
    - ``W_shortcut = f F(d) + tgt n**3 + W_G + (8 a**3 + m a + 2 a**2)``
      for ``shortcut_native_svp`` or ``+ (W_D + E(h) + m h + 2 h**2)`` for
      ``shortcut_dilation``, with the rank-one assembly
      ``W_G = 4 a**2 + n**2 + 12 a`` and the dilation assembly
      ``W_D = h**2 + 2 a**2``;
    - ``W_linear,extra = d**2 + 12 d + 4 s d`` for the linear model, whose
      original factorization is already in ``f F(d)`` or planning.

    ``size_units`` adds the selected observable work, physical
    materialization, direction preparation, the vector reductions and, for
    a shortcut, the componentwise normalization ``parts d**2 int(alpha!=1)``.
    The factor-based inverse forms neither an encoded matrix copy nor a 2d
    matrix or RHS embedding. The polynomial term ``m d`` uses the
    whole-Clenshaw-visit convention.

    ``workspace_bytes`` follows these explicit peaks, with
    complex ceilings even for real input and ``r = min(a, PROJECTOR_ROW_TILE)``:
    ``inverse_action = 208 d``, ``augmentation = 16 a**2 + 16 r a + 64 a``,
    ``native_action = max(augmentation, 48 a**2 + 192 a + 32 n)``,
    ``dilation_action = max(augmentation, 16 a**2 + 16 h**2,
    16 a**2 + 32 h**2 + 152 h + 32 n)`` and ``linear_action = 128 d``. For a
    shortcut, ``scaled_copy = 16 d**2 int(alpha != 1)``, ``encoded_resident
    = 16 n**2 if embedded else scaled_copy``, ``preparation = max(scaled_copy
    + 16 n**2, encoded_resident + 16 (n + d)) if embedded else scaled_copy``,
    ``rhs_workspace = 16 n if embedded else 0``, ``coefficient_frontier =
    16 (m + 1)``, ``reference_frontier = 16 n tgt`` and ``numerical =
    encoded_resident + rhs_workspace + coefficient_frontier +
    reference_frontier + max(selected_action, linear_action if linear else 0)``.
    For the inverse, ``numerical = 16 (m + 1) + 208 d``. ``statistics =
    16 returned_branch_dimension + 16 d + 16 d int(quadratic_form) +
    observable_bytes + 40 d int(materialize_physical)``, where the inverse
    returns its d-entry original-coordinate branch directly. Then
    ``workspace_bytes = max(B_F if acquired else 0, (P if acquired else 0) +
    max(preparation, numerical, statistics))``. Factors already owned by the
    Plan are excluded here. Planning adds their persistent ``P(d)`` to this
    field when it compares the resident arrays with ``max_bytes``. These are
    conservative known-array ceilings, not minimal simultaneous allocation
    equalities, and they infer no allocator traffic or RSS.
    """
    from .numerical import PROJECTOR_ROW_TILE

    d = len(matrix)
    n = d * (2 if embedded else 1)
    shortcut = method != "qsvt_inverse"
    a = 2 * n if shortcut else 0
    h = 2 * a if method == "shortcut_dilation" else 0
    m = polynomial.degree
    target = int(shortcut and t_source in {"grid", "noisy_binary_search"})
    linear = int(t_source == "linear_kappa_sequence")
    steps = _ceil_log2(kappa) if linear else 0
    acquire = int((not shortcut or bool(linear)) and not factors_held)
    factor_work, factor_bytes, acquisition_bytes = original_factor_laws(d, hermitian)
    # Actual decompositions of one evaluation: the acquired original
    # factorization (SVD, or eigh of a Hermitian original), the SVD of G_t
    # for the native SVP shortcut and the eigh of H(G_t) for the dilation.
    svd = acquire * int(not hermitian) + int(method == "shortcut_native_svp")
    eigh = acquire * int(hermitian) + int(method == "shortcut_dilation")
    rows, trials, queries = search_envelope(kappa, t_source)
    # Two basis-change matvecs of the selected action, the row product
    # b'^dagger A_t of the rank-one projector and the linear model's
    # coordinate matvec. G_t is formed without a dense matrix product.
    matvecs = 2 + int(shortcut) + linear
    observable = output.kind in {"quadratic_form", "normalized_expectation"}
    if observable:
        from nwqlib.operators.inputs import _scaled_observable_requirements
        observable_bytes, observable_work = _scaled_observable_requirements(output.observable)
    else:
        observable_bytes = observable_work = 0
    materialize = any(o.frame == "physical" for o in outputs)
    # Branch mass, plus one reduction charged for an embedded inverse. That
    # inverse computes its slice mass on the same d entries as the branch
    # mass, so the extra reduction is a conservative allowance. Selected norm
    # and its subnormal refinement. Target norm. Full-branch/reference
    # normalization refinements and phase dot. Selected observable dot.
    # Model RHS/step reductions. Six finite/nonzero/lost-component reductions
    # at most in apply_vector.
    reductions = (
        1
        + int(embedded)
        + 2
        + target
        + int(observable)
        + linear * (2 + steps)
        + 6 * int(materialize)
    )
    parts = 2 if matrix.dtype.kind == "c" else 1
    normalization_work = parts * d * d * int(alpha != 1.0) * int(shortcut)
    direction_work = 4 * d
    # Per real/imaginary part: frexp, mantissa product, exponent addition,
    # ldexp, and four finite/lost-mask ufunc visits. Two input masks.
    materialization_work = 18 * d * int(materialize)
    if not shortcut:
        size = acquire * factor_work + m * d + 2 * d * d
    else:
        assembly = 4 * a * a + n * n + 12 * a
        # The eigh of H(G_t) with vectors, E(h), or the full SVD of G_t, 8 a^3.
        decomposition = (hermitian_eigensystem_work(h) if h else 8 * a**3)
        p = h or a
        action = decomposition + m * p + 2 * p * p + (h * h + 2 * a * a if h else 0)
        size = acquire * factor_work + target * n**3 + assembly + action
    size += linear * (d * d + 12 * d + 4 * steps * d)
    size += (
        reductions * n
        + observable_work
        + normalization_work
        + direction_work
        + materialization_work
    )

    coefficient_frontier = 16 * (m + 1)
    if not shortcut:
        preparation = 0
        numerical = coefficient_frontier + 208 * d
    else:
        r = min(a, PROJECTOR_ROW_TILE)
        scaled_copy = 16 * d * d * int(alpha != 1.0)
        encoded_resident = 16 * n * n if embedded else scaled_copy
        preparation = (max(scaled_copy + 16 * n * n, encoded_resident + 16 * (n + d))
                       if embedded else scaled_copy)
        rhs_workspace = 16 * n if embedded else 0
        reference_frontier = 16 * n * target
        augmentation = 16 * a * a + 16 * r * a + 64 * a
        if h:
            selected_action = max(augmentation, 16 * a * a + 16 * h * h,
                                  16 * a * a + 32 * h * h + 152 * h + 32 * n)
        else:
            selected_action = max(augmentation, 48 * a * a + 192 * a + 32 * n)
        numerical = (encoded_resident + rhs_workspace + coefficient_frontier + reference_frontier
                     + max(selected_action, 128 * d if linear else 0))
    statistics = (
        16 * (n if shortcut else d)
        + 16 * d
        + 16 * d * int(output.kind == "quadratic_form")
        + observable_bytes
        + 40 * d * int(materialize)
    )
    workspace = max(
        acquisition_bytes if acquire else 0,
        (factor_bytes if acquire else 0) + max(preparation, numerical, statistics),
    )
    return QLSWork(
        original_dimension=d,
        system_dimension=n,
        augmented_dimension=a,
        dilation_dimension=h,
        target_solves=target,
        svd_calls=svd,
        eigh_calls=eigh,
        matvecs=matvecs,
        search_rows=rows,
        planned_trials=trials,
        planned_queries=queries,
        vector_reductions=reductions,
        observable_actions=int(observable),
        workspace_bytes=workspace,
        size_units=size,
    )


def application_arguments(reconstruction, *, t_value=1.0, rows=None, trials=None, queries=None,
                          reference_norm=None):
    """Return the argument bindings that a classical model application must report.

    The bindings are the selected ``alpha``, ``kappa_be``,
    ``polynomial_kappa``, ``rescale``, ``degree`` and ``t_value``, the work
    counts of ``QLSWork``, and a ``t_source.<name>`` flag. A grid or
    noisy-search norm model also records ``encoded_reference_norm``, the
    norm ``||(A/alpha)^-1 b_hat||`` of its one encoded reference solve, so
    that verification can reuse that intermediate instead of solving again.
    Execution calls this with the realized ``t``, rows, trials, queries and
    reference norm, and ``_validate_application`` calls it with the
    selected values (the reference norm then has the placeholder 1), so a
    stored observation is checked against the same field list that
    produced it.
    """
    r, w = reconstruction, reconstruction.work
    values = dict(
        alpha=r.alpha,
        kappa_be=r.kappa_be,
        polynomial_kappa=r.polynomial_kappa,
        rescale=r.polynomial.rescale,
        degree=r.polynomial.degree,
        t_value=t_value,
        target_solves=w.target_solves,
        svd_calls=w.svd_calls,
        eigh_calls=w.eigh_calls,
        matvecs=w.matvecs,
        search_rows=w.search_rows if rows is None else rows,
        planned_trials=w.planned_trials if trials is None else trials,
        planned_queries=w.planned_queries if queries is None else queries,
    )
    if w.target_solves:
        values["encoded_reference_norm"] = 1.0 if reference_norm is None else reference_norm
    values["t_source." + r.t_source] = 1
    return tuple(
        Binding(parameter=name, value=value if type(value) is int else Float64(value=float(value)))
        for name, value in values.items()
    )
