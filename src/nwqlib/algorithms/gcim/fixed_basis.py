"""Fixed GCiM: normalized preparations, exact pair reductions or grouped sampled counts, one H/S solve.

H=cI+H0 is reconstructed as Hproj=H0proj+cS from the same overlap
observations. The solve uses H0proj and adds c to its Ritz values, so the
identity term does not enter the ill-conditioned projection. Complex
off-diagonals and their conjugate partners are kept.

The method is the discretized Hill-Wheeler problem of Zheng et al., Phys. Rev.
Research 5, 023200 (2023), Eq. (13), with matrix elements Eqs. (14)-(15) and
their Pauli-sum form Eqs. (33)-(34) (numbering of arXiv:2212.09205v1). The
generating functions are caller-supplied normalized states rather than the
paper's states ``exp(Gamma(Z_p))|Phi>`` at discretized generator coordinates
``Z_p`` (Eq. (10), discretized in Eq. (17)).
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import fsum, isfinite
from typing import Annotated, ClassVar, Literal
import numpy as np
from pydantic import Field, StrictBool, model_validator
from nwqlib.algorithms.protocol import Method, AlgorithmDescriptor, ApplicabilityError
from nwqlib.algorithms._eigen_inputs import (
    eigen_operator,
    eigen_state,
    state_direction,
    preparation_requirements,
    DEFAULT_CONVERSION_WORK,
)
from nwqlib.algorithms._eigen_support import matched_chunks, host_construction
from nwqlib._projected_eigensolver import (
    _sampled_pencil_diagnostics,
    _solve_projected_pencil,
    entrywise_gram_allowance,
    gram_formation_allowance,
)
from nwqlib.blocks import (
    SelectedConstruction,
    select_preparation,
    select_pauli_readout,
    select_signed_pauli,
    transform_block,
)
from nwqlib.core.analysis import Result, capture_analysis_origin
from nwqlib.evidence.error_model import exact_readout_sampling
from nwqlib.core.planning import Plan, Experiment, ReadoutDetails
from nwqlib.core.records import (
    FrozenArray,
    Source,
    Complex128,
    ContentID,
    InputRef,
    Nonnegative,
    PositiveInt,
    Real,
    Record,
    Text,
)
from nwqlib.operators.access import Count, _check_bytes
from nwqlib.operators.inputs import ingest_pauli, pauli_table
from nwqlib.problems.inputs import StatePreparationSpec, ingest_vector
from nwqlib.problems.records import StateData
from nwqlib.ir import (
    Allocate,
    BlockCall,
    ClassicalValue,
    Definition,
    Measure,
    MeasurementBatch,
    MetadataRef,
    PortMap,
    Register,
    Release,
    Sequence,
    Setting,
)
from nwqlib.ir.validation import admitted_program
from .pencil import (
    _projected_solve_bytes,
    _projected_solve_work,
    assemble_pair_pencil,
    pair_acquisitions,
    pair_count,
    physical_hamiltonian,
)
# Importing the pair reducer registers block_matrix_elements, so a Plan or
# Result holding pair reductions validates in any process that loads GCiM.
from . import pair_reducer  # noqa: E402,F401

METHOD = Source(
    name="fixed_gcim",
    version="4",
    domain="projected Hermitian eigenvalue",
    reference="Zheng et al., Phys. Rev. Research 5, 023200 (2023), arXiv:2212.09205v1, Eq. (13)",
)
DESCRIPTOR = AlgorithmDescriptor(
    method=METHOD.name,
    version=METHOD.version,
    problem_families=("eigenproblem",),
    output_families=("eigenvalue",),
    access_families=("pauli", "dense", "csr", "csc"),
    references=(METHOD,),
    limitations=("projected estimate does not establish smallest full-space eigenvalue",),
)


# Relative window for the sampled Ritz-versus-enclosure plausibility check,
# scaled by the largest enclosure endpoint (docs/ENGINEERING_CONSTANTS.md).
# It is an engineering allowance near the enclosure edge, not calibrated to
# the eigensolve error, and not a statistical or spectral bound.
SAMPLED_PENCIL_ENCLOSURE_RTOL = 1e-8


class FixedGCIMBasis(Record):
    """Record that names a `FixedGCIM` trial basis as an `Eigenproblem` subspace.

    Build it with keyword arguments from the preparation records of the
    trial states, for example
    `FixedGCIMBasis(preparations=tuple(state_input(v).preparation for v in basis))`
    with `state_input` from `nwqlib.problems`. `preparations` is the only
    required argument. It holds the ordered content hashes and physical
    scales of the states, not their data. Pass its `reference` as
    `Eigenproblem(subspace=...)` to request exactly this trial basis.
    `FixedGCIM` then rejects a basis that differs from it.

    Attributes:
        preparations: Required, nonempty. One preparation record per trial
            state, in basis order.
        convention: Default `"normalized prepared columns"`, the only
            accepted value. Each state enters as its normalized preparation
            column.

    Examples:
        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import FixedGCIM, FixedGCIMBasis
        >>> from nwqlib.problems import state_input
        >>> basis = ([1, 1], [1, 1j])
        >>> preparations = tuple(state_input(v).preparation for v in basis)
        >>> record = FixedGCIMBasis(preparations=preparations)
        >>> problem = Eigenproblem(A=[[1, 0], [0, -1]], subspace=record.reference)
        >>> result = solve(problem, method=FixedGCIM(basis=basis), seed=7)
        >>> print(round(result.eigenvalue, 10))
        -1.0
    """

    preparations: Annotated[tuple[StatePreparationSpec, ...], Field(min_length=1)]
    convention: Literal["normalized prepared columns"] = "normalized prepared columns"

    @property
    def reference(self):
        """Reference to this basis by content hash, for `Eigenproblem(subspace=...)`."""
        return InputRef(
            identity=self.content_id, representation="fixed normalized GCiM basis", source=METHOD
        )


class PauliTerm(Record):
    """One real Pauli term of an admitted operator or of an ADAPT commutator ``[H, A]``.

    Operator tables include the identity term.

    Attributes:
        label: Qiskit Pauli label, qubit 0 rightmost.
        coefficient: Real coefficient in the problem's energy unit.
    """

    label: Text
    coefficient: Real


PAULI_LETTERS = "IXYZ"


class PauliArrays(Record):
    """Real-coefficient Pauli terms stored as one letter-code table and one coefficient array.

    ``labels`` is an int64 array of shape ``(rows, q)``. Entry ``[r, j]`` is
    the code of character ``j`` of row ``r``'s label string, with ``I = 0``,
    ``X = 1``, ``Y = 2`` and ``Z = 3``. Character ``j`` acts on qubit
    ``q - 1 - j``, so the rightmost character is qubit 0, as in every Pauli
    label string of the library. ``coefficients`` is the float64 array of
    shape ``(rows,)`` whose entry ``r`` belongs to row ``r``. Rows are distinct
    and in increasing label-string order. The codes increase with the letters,
    so that order is the lexicographic order of the code rows. Each row costs
    ``8q + 8`` raw bytes.

    Attributes:
        labels: Letter-code table described above.
        coefficients: Real coefficient of each row.
    """

    labels: FrozenArray
    coefficients: FrozenArray

    @model_validator(mode="after")
    def _layout(self):
        """Require the dtypes and shapes above, codes in 0..3 and strictly increasing rows."""


        labels, coefficients = self.labels.array, self.coefficients.array
        if (
            labels.dtype != np.int64
            or labels.ndim != 2
            or labels.shape[1] < 1
            or coefficients.dtype != np.float64
            or coefficients.shape != labels.shape[:1]
        ):
            raise ValueError("Pauli arrays need an int64 (rows, q) code table and float64 (rows,) coefficients")
        if labels.size and (labels.min() < 0 or labels.max() > 3):
            raise ValueError("Pauli letter codes must be 0 (I), 1 (X), 2 (Y) or 3 (Z)")
        if len(labels) > 1:
            differ = labels[1:] != labels[:-1]
            first = differ.argmax(axis=1)
            rows = np.arange(len(first))
            if not differ.any(axis=1).all() or not (
                labels[1:][rows, first] > labels[:-1][rows, first]
            ).all():
                raise ValueError("Pauli array rows must be distinct and in increasing label order")
        return self

    @classmethod
    def from_rows(cls, rows, num_qubits):
        """Build the arrays from ``(label, real coefficient)`` rows in increasing label order."""


        codes = {letter: code for code, letter in enumerate(PAULI_LETTERS)}
        rows = tuple(rows)
        labels = np.array(
            [[codes[letter] for letter in label] for label, _ in rows], dtype=np.int64
        ).reshape(len(rows), num_qubits)
        coefficients = np.array([float(c) for _, c in rows], dtype=np.float64)
        return cls(labels=FrozenArray(labels), coefficients=FrozenArray(coefficients))

    def rows(self):
        """Return the terms as ``((label, coefficient), ...)`` in stored order."""
        return tuple(
            ("".join(PAULI_LETTERS[code] for code in row), coefficient)
            for row, coefficient in zip(
                self.labels.array.tolist(), self.coefficients.array.tolist(), strict=True
            )
        )


class SampledGroup(Record):
    """One qubit-wise-commuting system basis and its indices into the reconstruction's nonidentity rows.

    Attributes:
        basis: The group's basis word of ``num_qubits`` letters, I where
            no member acts, qubit zero rightmost as in the Pauli labels.
        indices: Positions of the member rows in
            ``FixedGCIMReconstruction.nonidentity_terms``.
    """

    basis: Text
    indices: tuple[Count, ...]


class FixedGCIMReconstruction(Record):
    """Pauli table and basis size that turn the observed scalars back into ``H`` and ``S``.

    The identity coefficient ``c`` is not measured. The acquired scalars give
    ``H0`` without it, the recorded ``H`` adds ``c S`` from the acquired
    overlaps, and the solve adds ``c`` to the Ritz values.

    Attributes:
        schema_version: Record layout version.
        terms: The admitted operator's real Pauli terms as one letter-code
            table and one coefficient array (``PauliArrays``, rows in
            increasing label order), the identity row included. The exact
            pair reductions act with its nonidentity rows. Classical plans
            store an empty table because they read ``H0`` directly.
        basis_size: Number of trial states ``m``.
        num_qubits: System register width.
        host_identity_shift: For a classical plan, the mean diagonal
            ``c = trace(A)/d`` that the host kernel removes from ``A`` before
            it projects, so its ``h`` scalars are ``V^dagger (A - cI) V``. For
            Pauli input this is the identity coefficient. Zero for a quantum
            plan, whose ``c`` is the identity term of ``terms``.
        covariance: Fixed statement of the identity-term rule above.
        groups: For a sampled Plan with nonidentity terms, the recorded
            deterministic first-fit qubit-wise-commuting partition of the
            nonidentity rows (``PauliTerms.group(strategy="qwc")``), fixed
            before the measurements. Group zero supplies the overlap of each
            off-diagonal quadrature. Empty for exact and classical Plans.
    """

    schema_version: Literal[3] = 3
    terms: PauliArrays
    basis_size: PositiveInt
    num_qubits: PositiveInt
    host_identity_shift: Real = 0.0
    covariance: Literal["H_identity=c*S uses the same overlap observations"] = (
        "H_identity=c*S uses the same overlap observations"
    )
    groups: tuple[SampledGroup, ...] = ()

    @model_validator(mode="after")
    def _one_identity_source(self):
        """A quantum table carries its own identity term, so only a classical plan has a host shift."""
        if len(self.terms.coefficients.array) and self.host_identity_shift != 0:
            raise ValueError("a Pauli-table reconstruction takes c from its identity term")
        return self

    @model_validator(mode="after")
    def _sampled_partition(self):
        """Require recorded groups to partition the nonidentity rows into qubit-wise-commuting bases."""
        if not self.groups:
            return self
        rows = self.nonidentity_terms
        indices = [i for group in self.groups for i in group.indices]
        if sorted(indices) != list(range(len(rows))):
            raise ValueError("GCIM groups must partition the nonidentity rows exactly once")
        for group in self.groups:
            if not group.indices or len(group.basis) != self.num_qubits:
                raise ValueError("GCIM group has an empty membership or wrong basis width")
            basis = ["I"] * self.num_qubits
            for i in group.indices:
                for bit, letter in enumerate(rows[i][0]):
                    if letter != "I":
                        if basis[bit] not in ("I", letter):
                            raise ValueError("GCIM group is not qubit-wise commuting")
                        basis[bit] = letter
            if group.basis != "".join(basis):
                raise ValueError("GCIM group basis differs from its Pauli rows")
        return self

    def _identity_row(self):
        """Return the index of the all-identity row, or None."""
        rows = np.flatnonzero(~self.terms.labels.array.any(axis=1))
        return int(rows[0]) if rows.size else None

    @property
    def identity_shift(self):
        row = self._identity_row()
        return self.host_identity_shift if row is None else float(self.terms.coefficients.array[row])

    @property
    def nonidentity_terms(self):
        """The nonidentity rows as ``((label, coefficient), ...)`` in stored order."""
        row = self._identity_row()
        return tuple(item for index, item in enumerate(self.terms.rows()) if index != row)

    @property
    def nonidentity_coefficients(self):
        """The nonidentity coefficients in stored order, without forming labels."""
        row = self._identity_row()
        coefficients = self.terms.coefficients.array
        return tuple(float(c) for index, c in enumerate(coefficients.tolist()) if index != row)


class ScalarEstimate(Record):
    """Pooled mean of one weighted group of Pauli terms in a sampled `FixedGCIM` run.

    `FixedGCIMResult.estimates` holds one per group setting. Each count
    measurement of the setting measures `Y = sum_k c_k Z_k` from shared
    shots. `value` is the shot-weighted mean of the measurements' means and
    need not lie in [-1, 1]. `chunk_mean_variances` contains
    `(mean(Y**2) - mean(Y)**2)/(N - 1)` for N > 1 shots and None for N = 1.
    The group moments in `FixedGCIMResult.group_moments` keep the raw
    moments and flag a negative raw variance. Independent measurements pool
    their variances with squared shot weights. The overlap and physical-H
    covariance are recorded in the group and pair statistics. These are
    sample-mean variance estimates, not a joint bound on the pencil or a
    confidence interval for the projected eigenvalue. The fields below are
    read-only.

    Attributes:
        experiment: Name of the group setting.
        value: Shot-weighted mean of the measurements' means of Y.
        contribution_ids: Content hashes of the contributing count data.
        weights: Fraction of the returned shots in each contribution.
        chunk_mean_variances: The sample-mean variance of Y of each
            contribution, or None for a one-shot contribution.
        sampled_shots: Total returned shots.
    """

    schema_version: Literal[2] = 2
    experiment: Text
    value: Real
    contribution_ids: tuple[ContentID, ...]
    weights: tuple[Real, ...]
    chunk_mean_variances: tuple[Real | None, ...]
    sampled_shots: Count


class GroupMoment(Record):
    """Moments of one independently measured counts histogram.

    Y is the coefficient-weighted group parity, including the ancilla sign
    on an off-diagonal pair. The optional X is that ancilla sign and is
    recorded only by group zero. Variances and covariance concern sample
    means. A one-shot measurement has unavailable variance. A negative raw
    variance is published with variance_negative=True and is never clipped.
    label_means preserves each term's first moment from the same histogram.

    With N returned shots of one fixed setting and the unbiased sample
    covariance divided again by N, the sample-mean variances are
    ``V_Y = (mean(Y**2) - mean(Y)**2)/(N - 1)``,
    ``V_X = (1 - mean(X)**2)/(N - 1)`` and
    ``C_XY = (mean(XY) - mean(X) mean(Y))/(N - 1)`` for N > 1.

    Attributes:
        experiment: The setting's experiment name.
        contribution_id: Content hash of the count data these moments come from.
        pair: ``(i, j)`` with ``i <= j``.
        quadrature: ``real`` (ancilla X) or ``imag`` (ancilla Y).
        group: Index into ``FixedGCIMReconstruction.groups``.
        shots: Returned shots N of that count data.
        label_means: The mean of each member's ``Z_k`` in group order.
        mean: ``mean(Y)``.
        second: ``mean(Y**2)``.
        mean_variance: ``V_Y``, or None for N = 1.
        overlap_mean: ``mean(X)`` for the overlap-supplying group of an
            off-diagonal pair, otherwise None.
        overlap_cross: ``mean(XY)`` of that group, otherwise None.
        overlap_variance: ``V_X`` of that group, otherwise None.
        mean_covariance: ``C_XY`` of that group, otherwise None.
        variance_negative: Whether ``V_Y`` or ``V_X`` is negative.
    """

    experiment: Text
    contribution_id: ContentID
    pair: tuple[Count, Count]
    quadrature: Literal["real", "imag"]
    group: Count
    shots: Count
    label_means: tuple[Real, ...]
    mean: Real
    second: Real
    mean_variance: Real | None
    overlap_mean: Real | None
    overlap_cross: Real | None
    overlap_variance: Real | None
    mean_covariance: Real | None
    variance_negative: StrictBool


class SampledPairVariance(Record):
    """Independent real/imaginary sample-mean variances, including shared-shot covariance.

    Entries are (real, imaginary). A diagonal overlap is algebraic one,
    with zero variance. None means a required measurement has one shot.
    Negative estimates stay raw and set variance_negative. These estimates
    are not simultaneous confidence bounds or a Ritz-value certificate.

    Under independent measurements of different settings, each quadrature
    has ``V_H0 = sum_g V_(Y_g)`` and
    ``V_H = sum_g V_(Y_g) + c_I**2 V_X + 2 c_I C_(X,Y_0)``. The diagonal
    uses ``V_S = 0`` and covariance zero because its solved overlap is
    algebraic. When ``c_I = 0``, an unavailable overlap variance does not make
    the physical-H variance unavailable. Independent real and imaginary
    settings give ``E|error|**2 = V_real + V_imag``.

    Attributes:
        pair: ``(i, j)`` with ``i <= j``.
        h0: Variance of the offset-free Hamiltonian entry per quadrature.
        overlap: Variance of the overlap entry per quadrature.
        covariance: Overlap–H0 sample-mean covariance per quadrature.
        physical_h: Variance of ``H0 + c_I S`` per quadrature.
        variance_negative: Whether any h0, overlap or physical_h entry is negative.
    """

    pair: tuple[Count, Count]
    h0: tuple[Real | None, Real | None]
    overlap: tuple[Real | None, Real | None]
    covariance: tuple[Real | None, Real | None]
    physical_h: tuple[Real | None, Real | None]
    variance_negative: StrictBool


class PairEstimate(Record):
    """The reported ``H0_ij`` and ``S_ij`` of one exact pair and the reductions they were pooled from.

    Each contribution is one exact readout (``block_matrix_elements``) of the
    pair's phase-faithful joint state (or of its diagonal system state) and
    counts once in the pooling. Its two raw complex scalars are the
    offset-free Hamiltonian entry and the overlap entry in the requested
    ``(left, right)`` orientation. A
    diagonal pair's overlap is its raw squared norm, kept as evidence while
    assembly uses unit diagonal overlap. The individual Pauli-term
    transitions are not recorded and cannot be recovered from the weighted
    payload, so a different Hamiltonian truncation needs new measurements.

    Attributes:
        experiment: The pair's experiment name.
        pair: ``(left, right)`` with ``left <= right``.
        hamiltonian: Pooled offset-free ``H0_ij``.
        overlap: Pooled ``S_ij``.
        contribution_ids: Content hashes of the contributing exact readouts.
        weights: Share of each contribution in the pooled `hamiltonian` and
            `overlap`.
        hamiltonian_bound: Outward error allowance for the reported offset-free
            Hamiltonian entry ``H0_ij`` under the state-error model of the
            preparation records. It combines that model with bounds for applying
            H0 to the right state, taking its complex inner product with the left
            state, and forming the weighted average of repeated acquisitions of
            the same pair. None when a preparation record has an unresolved
            exclusion or no applicable state budget.
            When a supplied matrix is synthesized under control, this
            allowance also assumes the exact-to-rounding convention of
            [exact dense synthesis](../../development/dense_synthesis.md).
            It does not add an independently certified synthesis residual.
        overlap_bound: The corresponding allowance for the reported overlap,
            with state, contraction and pooling terms. The solved diagonal
            is algebraic one and contributes zero to the Gram allowance.
            When a supplied matrix is synthesized under control, this
            allowance also assumes the exact-to-rounding convention of
            [exact dense synthesis](../../development/dense_synthesis.md).
            It does not add an independently certified synthesis residual.
    """

    experiment: Text
    pair: tuple[Count, Count]
    hamiltonian: Complex128
    overlap: Complex128
    contribution_ids: tuple[ContentID, ...]
    weights: tuple[Real, ...]
    hamiltonian_bound: Nonnegative | None
    overlap_bound: Nonnegative | None


class ProjectedPencil(Record):
    """Projected matrices H and S of one trial basis and the diagnostics of their solve.

    `FixedGCIMResult.pencil` and `ADAPTResult.pencil` hold it. The matrices
    are the measured values, completed by complex conjugation, without
    averaging, diagonal jitter or positive-semidefinite repair. The solve
    keeps the overlap eigenvectors whose eigenvalues exceed `overlap_cutoff`
    and diagonalizes H in that subspace (canonical orthogonalization,
    Epperly, Lin and Nakatsukasa, arXiv:2110.07492v2, Algorithm 1.1, and
    Zheng et al., arXiv:2312.07691v3, Appendix F, Eqs. (F1)-(F3)). No
    full-space state residual is computed. The fields below are read-only.

    Attributes:
        hamiltonian: Projected Hamiltonian ``H = H0 + c_I S`` of shape (m, m)
            in the problem's energy unit, from ``H0`` without the identity
            term ``c_I I`` and from ``S``, completed from its upper triangle
            by conjugation. The solve used ``H0`` and added
            ``c_I`` to its Ritz values.
        overlap: Gram matrix ``S`` of shape (m, m), completed the same way.
        overlap_eigenvalues: Full ascending spectrum of ``S``, before the cutoff.
        eigenvalues: Ascending Ritz values of the kept subspace.
        coordinate_vectors: Rows index the m basis states and columns the Ritz
            pairs. Each column is an S-normalized coefficient vector.
        kept_rank: Number of overlap directions above ``overlap_cutoff``.
        overlap_cutoff: Absolute overlap-eigenvalue threshold used by this solve.
        overlap_filter: `"deterministic_gram"` refuses a negative overlap mode
            beyond roundoff. `"sampled_positive_subspace"` keeps only positive
            modes above the cutoff of a sampled, possibly indefinite S.
        overlap_condition_number: Largest over smallest kept overlap eigenvalue.
        projected_residual: `norm2(Hc - ESc)` for the lowest Ritz pair.
        projected_backward_error: Dimensionless residual scaled by
            `(normF(H) + abs(E) normF(S)) norm2(c)`, on the physical pencil
            (H, S) in the Problem's energy unit, with `H = H0 + c_I S` when
            the solve removed an identity coefficient c_I. Rescaling the
            energy unit leaves it unchanged, but adding an identity term to
            the operator changes `normF(H)` and `abs(E)` and therefore the
            value. Lanczos states its value on the normalized pencil instead.
        overlap_normalization_error: `abs(c^dagger S c - 1)` for the lowest Ritz pair.
        failure_reason: Solver refusal code, or None when the solve completed.
        sampled_enclosure: For sampled quantum data, the outward-rounded
            interval `[c - sum_k |c_k|, c + sum_k |c_k|]` that contains the
            spectrum of the processed operator `c I + sum_k c_k P_k`, or None.
        sampled_enclosure_violation: Distance of the lowest Ritz value outside that
            enclosure, zero inside it, or None when unavailable.
        sampled_enclosure_passed: Whether the violation lies within
            `sampled_enclosure_rtol` times the largest enclosure endpoint
            magnitude.
        sampled_enclosure_rtol: Relative window used by that comparison.
        sampled_failure: Solver refusal or `"ritz_outside_operator_enclosure"`.
        processing: Fixed description of the processing applied to the raw pencil.
    """

    hamiltonian: tuple[tuple[Complex128, ...], ...]
    overlap: tuple[tuple[Complex128, ...], ...]
    overlap_eigenvalues: tuple[Real, ...]
    eigenvalues: tuple[Real, ...]
    coordinate_vectors: tuple[tuple[Complex128, ...], ...]
    kept_rank: Count
    overlap_cutoff: Annotated[Real, Field(gt=0)]
    overlap_filter: Literal["deterministic_gram", "sampled_positive_subspace"]
    overlap_condition_number: Annotated[Real, Field(ge=1)] | None
    projected_residual: Nonnegative | None
    projected_backward_error: Nonnegative | None
    overlap_normalization_error: Nonnegative | None
    failure_reason: Text | None
    # A noisy pencil can violate the processed operator enclosure while its
    # own eigensolve is numerically valid. These are not solver failures.
    sampled_enclosure: tuple[Real, Real] | None = None
    sampled_enclosure_violation: Nonnegative | None = None
    sampled_enclosure_passed: StrictBool | None = None
    sampled_enclosure_rtol: Nonnegative | None = None
    sampled_failure: Text | None = None
    processing: Literal["raw Hermitian completion; canonical positive-subspace projection"] = (
        "raw Hermitian completion; canonical positive-subspace projection"
    )

    @property
    def coefficients(self):
        """Coefficient vector of the lowest Ritz pair, the first column of `coordinate_vectors`.

        It is an empty tuple when no Ritz value exists. In the coordinates of
        all columns, the kept pencil is `(diag(eigenvalues), I)` up to the
        reported solve and normalization errors. Reading it performs no new
        projection or eigensolve.
        """
        return tuple(row[0] for row in self.coordinate_vectors) if self.eigenvalues else ()

    def _sampled_summary(self):
        if self.overlap_filter != "sampled_positive_subspace":
            return []
        status = ("not assessed against" if self.sampled_enclosure_passed is None else
                  "within numerical window of" if self.sampled_enclosure_passed else "outside numerical window of")
        return [f"Sampled Ritz estimate: {status} processed Pauli enclosure {self.sampled_enclosure}; "
                f"violation={self.sampled_enclosure_violation}, relative window={self.sampled_enclosure_rtol}. "
                "Physical accuracy and ground identity not established."]

    def _validate_sampled_enclosure(self, reconstruction, *, sampled):
        """Recompute the sampled enclosure diagnostic from this pencil and require equality.

        A saved or reloaded pencil could otherwise carry diagnostic fields
        that belong to another Ritz value or operator. Deterministic pencils
        must carry no enclosure fields.
        """
        if self.overlap_filter != ("sampled_positive_subspace" if sampled else "deterministic_gram"):
            raise ValueError("pencil overlap filter differs from its selected observations")
        bounds = processed_pauli_enclosure(reconstruction) if sampled else None
        if self.sampled_enclosure != bounds:
            raise ValueError("sampled pencil enclosure differs from its processed operator")
        value = self.eigenvalues[0] if self.eigenvalues else None
        violation = None if bounds is None or value is None else max(bounds[0]-value, value-bounds[1], 0.)
        passed = None if violation is None else violation <= SAMPLED_PENCIL_ENCLOSURE_RTOL*max(map(abs, bounds))
        failure = None if bounds is None else self.failure_reason or (
            "ritz_outside_operator_enclosure" if passed is False else None)
        if (self.sampled_enclosure_rtol != (None if bounds is None else SAMPLED_PENCIL_ENCLOSURE_RTOL)
                or self.sampled_enclosure_violation != (violation if violation is None or isfinite(violation) else None)
                or self.sampled_enclosure_passed != passed or self.sampled_failure != failure):
            raise ValueError("sampled enclosure diagnostic differs from its actual Ritz estimate")


def sampled_pencil_fields(diagnostics):
    """Return the ``sampled_*`` ProjectedPencil fields of a sampled enclosure diagnostic, or an empty dict when there is none.

    A nonfinite violation is stored as None.
    """
    if diagnostics is None:
        return {}
    violation = diagnostics["enclosure_violation"]
    return dict(sampled_enclosure=tuple(diagnostics["enclosure_bounds"]),
        sampled_enclosure_violation=violation if violation is None or isfinite(violation) else None,
        sampled_enclosure_passed=diagnostics["enclosure_passed"],
        sampled_enclosure_rtol=diagnostics["enclosure_relative_tolerance"],
        sampled_failure=diagnostics["failure_reason"])


def processed_pauli_enclosure(reconstruction):
    """Return an outward-rounded interval that contains the processed operator's spectrum.

    For ``A = cI + sum_k c_k P_k`` each Pauli string has unit operator norm, so
    the spectrum lies in ``[c - sum_k |c_k|, c + sum_k |c_k|]`` by the triangle
    inequality. Every Ritz value of an exact pencil lies inside this interval,
    because a Rayleigh quotient of ``A`` cannot leave ``[lambda_min, lambda_max]``.
    Each rounding step moves outward so binary64 evaluation cannot shrink
    it. ``fsum`` rounds to nearest, so the radius is moved up one more unit
    in the last place after the sum. No full-space action is needed.
    Returns None when an endpoint is not finite.
    """
    try:
        with np.errstate(over="ignore"):
            radius = fsum(float(np.nextafter(abs(c), np.inf)) for c in reconstruction.nonidentity_coefficients)
            if radius:
                radius = float(np.nextafter(radius, np.inf))
            center = reconstruction.identity_shift
            bounds = (float(np.nextafter(center-radius, -np.inf)), float(np.nextafter(center+radius, np.inf)))
    except OverflowError:
        return None
    return bounds if all(map(isfinite, bounds)) else None


# Shown with ``negative_deterministic_overlap_eigenvalue`` when an
# off-diagonal pair has no overlap bound. The solve then receives zero input
# allowance and refuses when min_k s_hat_k < -tau_ro, with
# tau_ro = B*eps*max_k |s_hat_k| the eigensolver roundoff allowance. The
# negative-mode check precedes rank filtering, so overlap_cutoff cannot
# change it.
UNQUALIFIED_GRAM_REFUSAL = (
    "The computed Gram matrix has a negative eigenvalue beyond the solver's roundoff allowance, "
    "and an off-diagonal preparation has no qualified state-error bound. Retry with sampled "
    "acquisition using shots, use an ADAPT construction whose native preparations have qualified "
    "receipts, or remove redundant trial states from the fixed basis. Increasing overlap_cutoff "
    "does not change this negative-eigenvalue check."
)


def _exact_overlap_allowance(basis_size, pair_bounds):
    """Return the Gram-error allowance of an overlap acquired by exact pair reductions, or None.

    ``pair_bounds[(i, j)]`` is the outward entry bound ``E^S_ij`` of each
    acquired off-diagonal overlap (``pair_reducer.entry_bounds``, pooled by
    ``pooled_bound`` over repeated acquisitions), from the actual receipt of
    each pair. With solved diagonal S fixed to one, ``E^S_ii = 0``. Then

        ||S_hat - S||_2 <= max_i sum_(j != i) E^S_ij <= (B - 1) max_(i<j) E^S_ij,

    where B is the full basis size. Shared acquisition error requires no
    statistical independence assumption or union factor. This supplies
    acquisition/assembly error only, leaving trial-space, truncation,
    eigensolve and ground-identification sources separate. An unresolved
    receipt delta of any off-diagonal pair makes the allowance unavailable
    (None), not zero.
    """
    errors = np.zeros((basis_size, basis_size))
    for (left, right), bound in pair_bounds.items():
        if left == right:
            continue
        if bound is None:
            return None
        errors[left, right] = errors[right, left] = bound
    return entrywise_gram_allowance(errors)


def _mean_diagonal(operator, *, max_bytes):
    """Return ``c = trace(A)/d``, the identity component a classical projection removes from A.

    For Pauli input this is the identity coefficient, exactly the identity
    component of the Pauli expansion. For dense and CSR/CSC input it is the
    mean of the stored diagonal, formed as for eigen_operator's mean-trace
    padding. Other representations give zero and keep their action. Any c
    gives the same Ritz values in exact arithmetic, since the solve adds it
    back. The mean diagonal is the mean eigenvalue, so it lies in
    [lambda_min, lambda_max] and ``||A - cI||_2 <= lambda_max - lambda_min``,
    a bound that an identity offset of A does not change.
    """
    representation = operator.reference.representation
    d = operator.basis.dimension
    if representation == "pauli":
        identity = "I" * (d.bit_length() - 1)
        return fsum(c.real for label, c in operator.pauli_terms().labels(max_bytes=max_bytes)
                    if label == identity)
    if representation in ("dense", "csr", "csc"):
        return fsum(operator._data.diagonal().real / d)
    return 0.0


def _offset_free_requirements(operator, count, shift):
    """Return the (bytes, work) that ``_offset_free_actions`` adds to ``count`` plain actions.

    Zero shift, Pauli input and other representations add nothing. Dense
    input copies every row of A once in blocks, d**2 entry copies, and
    subtracts c from d diagonal entries. Its one refilled row block and the
    block product are d-by-b arrays of ``_classical_projection_requirements``.

    A Run creates its shifted canonical CSR/CSC payload once, shares its
    original pattern only when every diagonal is present, and charges that
    payload throughout later actions. Setup work is charged once and each
    action visits the shifted pattern, including inserted diagonals. With h
    missing diagonals the extra work is ``W_setup + count*h`` on the first
    acquisition and ``count*h`` on a cache hit, relative to the
    ``count*(z + d)`` of the plain actions. The bytes are
    ``_shifted_sparse_requirements``. This charge includes the setup, so a
    caller that reuses a Run's shifted operator is charged conservatively.
    Before the scan, h is bounded by d
    from counts alone.
    """
    if not shift:
        return 0, 0
    representation = operator.reference.representation
    d = operator.basis.dimension
    if representation == "dense":
        return 0, d * d + d
    if representation in ("csr", "csc"):
        size, setup, h = _shifted_sparse_requirements(operator)
        return size, setup + count * h
    return 0, 0


def _nonidentity_table(operator):
    """Return the packed nonidentity rows of a Pauli operator, whose sum is ``A - c_I I``.

    The identity row has zero flip and phase words. The kept rows stay in
    stored order with their coefficients, and no label is materialized.
    """
    from nwqlib.operators._pauli import PauliTerms

    table = operator.pauli_terms()
    keep = table.x.any(axis=1) | table.z.any(axis=1)
    return PauliTerms._snapshot(table.num_qubits, table.x[keep], table.z[keep], table.coefficients[keep])


def _shifted_sparse(operator, shift, *, max_bytes):
    """Return ``A - shift*I`` of a canonical CSR/CSC operator as one read-only sparse matrix, built once.

    Apply the operator after subtracting its identity shift from the stored
    diagonal. A Run may reuse one shifted CSR/CSC value array. Stored
    diagonal entries receive one binary64 subtraction, and missing diagonal
    entries become the exactly represented negative shift. The sparse
    pattern is shared only when it already contains every diagonal position.
    Multiplication order can differ from a blockwise application even though
    the shifted entries are identical.

    A first scan finds the diagonal positions and counts the h missing
    diagonals, the output is admitted, then a second pass fills it. With z
    stored entries, d lines and index itemsize I the persistent payload is
    ``B_shifted = 16z`` when h = 0 and the immutable indices and pointers are
    shared, and ``(16 + I)(z + h) + I(d + 1)`` when h > 0. The scan is
    array-vectorized, so its line-index and mask populations, ``(I + 1)z``
    bytes, and for h > 0 the union's concatenated values, indices, lines and
    sort order join the setup scratch (``_shifted_sparse_requirements``).
    """
    from scipy import sparse

    matrix = operator._data
    representation = operator.reference.representation
    d = operator.basis.dimension
    indptr, indices = matrix.indptr, matrix.indices
    lines = np.repeat(np.arange(d, dtype=indices.dtype), np.diff(indptr))
    on_diagonal = indices == lines
    present = np.zeros(d, dtype=bool)
    present[lines[on_diagonal]] = True
    missing = np.flatnonzero(~present).astype(indices.dtype)
    _check_bytes(_shifted_sparse_requirements(operator, len(missing))[0], max_bytes, "shifted sparse operator")
    values = matrix.data.astype(np.complex128, copy=True)
    values[on_diagonal] -= shift
    kind = sparse.csr_matrix if representation == "csr" else sparse.csc_matrix
    if not len(missing):
        values.flags.writeable = False
        shifted = kind((values, indices, indptr), shape=matrix.shape, copy=False)
        # The constructor may downcast the index arrays. Assigning the
        # originals shares the immutable pattern without a copy.
        shifted.indices, shifted.indptr = indices, indptr
        return shifted
    all_lines = np.concatenate((lines, missing))
    all_indices = np.concatenate((indices, missing))
    all_values = np.concatenate((values, np.full(len(missing), -shift, dtype=np.complex128)))
    del lines, values
    order = np.lexsort((all_indices, all_lines))
    counts = np.bincount(all_lines, minlength=d)
    pointers = np.zeros(d + 1, dtype=indices.dtype)
    np.cumsum(counts, out=pointers[1:])
    result_values = all_values[order]
    result_values.flags.writeable = False
    result_indices = all_indices[order]
    shifted = kind((result_values, result_indices, pointers), shape=matrix.shape, copy=False)
    shifted.indices, shifted.indptr = result_indices, pointers
    return shifted


def _shifted_sparse_requirements(operator, missing=None):
    """Return ``(bytes, setup work, missing diagonals)`` of ``_shifted_sparse``.

    ``missing`` is the counted h, or None for the count-only bound h <= d.
    Bytes are the persistent payload ``B_shifted`` plus the setup scratch of
    the vectorized scan and, for h > 0, of the union. One sufficient visit
    law for the two-pass scanner is ``W_setup = 4z + 6d + 3h``.
    """
    matrix = operator._data
    d = operator.basis.dimension
    z = matrix.nnz
    itemsize = matrix.indices.dtype.itemsize
    h = d if missing is None else missing
    persistent = 16 * z if not h else (16 + itemsize) * (z + h) + itemsize * (d + 1)
    scratch = (itemsize + 1) * z + d + itemsize * h
    if h:
        scratch += (16 + 16 + 2 * itemsize + 8) * (z + h) + 8 * (d + 1)
    return persistent + scratch + 65536, 4 * z + 6 * d + 3 * h, h


def _offset_free_actions(operator, columns, shift, out, *, max_bytes, max_products, shifted=None,
                         limit_name="max_products"):
    """Fill ``out`` with ``(A - shift*I) V`` for the columns of V, with the shift removed from A first.

    ``columns`` and ``out`` are complex128 arrays of shape ``(d, k)`` that do
    not overlap. Forming ``A v`` and subtracting ``shift*v`` afterwards would
    leave the rounding of the diagonal term, about u*|shift|*|v_i| in entry
    i, which the ill-conditioned projected solve then amplifies. Removing the
    shift from the stored operator rounds only ``A_ii - shift``, with
    relative error u of the smaller difference.

    - Pauli: one grouped ``apply_terms`` call of the nonidentity terms, whose
      sum is ``A - shift*I`` exactly, on the whole block, writing into
      ``out``. Each group's diagonal is computed once per coordinate tile
      and shared by all k columns.
    - Zero shift, or a representation ``_mean_diagonal`` does not shift:
      ``operator.matvec`` of each column.
    - Dense: blocks of at most k rows are copied, their diagonal entries
      shifted and multiplied by all columns at once, so A is copied once and
      a block never exceeds one d-by-k array.
    - CSR/CSC: one product with the shifted sparse operator of
      ``_shifted_sparse``, built here or passed as ``shifted`` by a Run that
      reuses it, and no dense copy of A is made
      (``_offset_free_requirements``).
    """
    representation = operator.reference.representation
    d = operator.basis.dimension
    if representation == "pauli":
        from nwqlib.operators._pauli import apply_terms

        return apply_terms(_nonidentity_table(operator), columns, num_qubits=d.bit_length() - 1, out=out)
    if not shift or representation not in ("dense", "csr", "csc"):
        for index in range(columns.shape[1]):
            out[:, index] = operator.matvec(columns[:, index], max_bytes=max_bytes, max_products=max_products,
                                            limit_name=limit_name)
        return out
    count = columns.shape[1]
    matrix = operator._data
    if representation == "dense":
        # One row-block buffer is refilled, so a single block is live.
        block = np.empty((count, d), dtype=complex)
        for start in range(0, d, count):
            stop = min(d, start + count)
            rows = block[: stop - start]
            rows[...] = matrix[start:stop]
            rows[np.arange(stop - start), np.arange(start, stop)] -= shift
            out[start:stop] = rows @ columns
        return out
    if shifted is None:
        shifted = _shifted_sparse(operator, shift, max_bytes=max_bytes)
    out[...] = shifted @ columns
    return out


def _classical_projection_requirements(operator, columns, shift, preparation_bytes, basis_size):
    """Return the ``(bytes, work)`` of one classical projection over ``columns`` distinct states.

    Classical GCiM fills normalized trial columns V and shifted actions
    W=(A−cI)V in two preallocated complex arrays. Their payload is 32bN bytes
    for b columns of length N. Conjugating inner products form H₀=V†W and
    S=V†V without an additional state block. The projection allowance adds
    the b-by-b matrices and the actual preparation, operator and action
    workspace. The stored physical Hamiltonian is H=H₀+cS.

    With ``k = columns`` distinct states of length N and ``m = basis_size``,
    the stages are sequential and the peak is their maximum,
    ``max{16kN + B_preparation, B_action stage, 32kN + 512m² + B_dot}``
    (``B_Gram = 32kN + 512m² + B_dot``, with ``B_dot = 0`` for
    ``np.vdot`` of contiguous columns, which conjugates inside the
    reduction). ``B_preparation`` is the largest known preparation of one
    column. The action stage of Pauli input is the shared block action law
    ``pauli_action_requirements`` at k columns with a statically complex
    input (its input and output are V and W) and the count-only flip bound
    ``g = min(L, N)``, plus ``64L`` bytes for the kernel's packed
    nonidentity table (``_nonidentity_table``: the masked row copies and
    their immutable snapshot, 32 bytes per row each). Other input holds V
    and W plus ``32kN`` bytes for two further d-by-k arrays: the refilled
    dense row block and its product, or SciPy's C-ordered copy of the
    column block and the sparse block product, or the per-column route's
    action. The operator action's own workspace and the sparse blocks of
    ``_offset_free_requirements`` are added. ``512m²`` is
    ``_projected_solve_bytes``, which covers the small matrices and their
    temporary Hermitian parts. Work is the block action (Pauli) or k plain
    actions plus the shift removal, and ``2 m² N`` scalar products for the
    two Gram matrices of m² conjugating dots each.
    """
    from nwqlib.operators._pauli import pauli_action_requirements
    from nwqlib.operators.inputs import _matvec_requirements

    d = operator.basis.dimension
    k = columns
    if operator.reference.representation == "pauli":
        table = operator.pauli_terms()
        terms = int(np.count_nonzero(table.x.any(axis=1) | table.z.any(axis=1)))
        action_bytes, action_work = pauli_action_requirements(
            d, terms, min(terms, d), columns=k, complex_input=True)
        action_bytes += 64 * terms
    else:
        _, matvec_bytes, matvec_work = _matvec_requirements(operator)
        shift_bytes, shift_work = _offset_free_requirements(operator, k, shift)
        action_bytes = 64 * k * d + matvec_bytes + shift_bytes
        action_work = k * matvec_work + shift_work
    stage_bytes = max(
        16 * k * d + preparation_bytes,
        action_bytes,
        32 * k * d + _projected_solve_bytes(basis_size),
    )
    return stage_bytes, action_work + 2 * basis_size * basis_size * d


def _coefficient_mass(coefficients):
    """Return an outward upper bound on ``C1 = sum |c_t|`` of the given coefficients, as a float.

    Each rounding step moves upward, so binary64 evaluation cannot shrink it.
    """
    with np.errstate(over="ignore"):
        mass = fsum(float(np.nextafter(abs(complex(c)), np.inf)) for c in coefficients)
    return float(np.nextafter(mass, np.inf)) if mass else 0.0


def pair_reduction_point(identity, coefficients, q, *, diagonal, ancilla=True, point="pair"):
    """Return the ``block_matrix_elements`` reduction point of one exact pair query.

    ``identity`` is the admitted content identity of the Pauli operator
    whose nonidentity ``coefficients`` (in stored order) form H0, and ``q``
    the system width. ``diagonal`` selects the diagonal system-state layout,
    with the ancilla at bit 0 or, without ``ancilla``, the system register
    alone. ``point`` is the point identity.
    """
    from nwqlib.core.planning import ObservationPoint
    from .pair_reducer import REDUCER, pair_parameters

    terms = len(coefficients)
    parameters = pair_parameters(left=0, right=0 if diagonal else 1, qubits=q, identity=identity, terms=terms,
                                 flips=min(terms, 1 << q), c1=_coefficient_mass(coefficients).hex(),
                                 ancilla=ancilla)
    return ObservationPoint(id=point, kind="reduction", reducer=REDUCER, parameters=parameters)


def _pair_table(operator):
    """Bind the operator's packed nonidentity table for the Plan's pair reductions."""
    from .pair_reducer import bind_table

    return bind_table(operator.reference.identity, lambda: _nonidentity_table(operator))


def _pair_work_total(plan):
    """Return the summed registered work of the Plan's pair reductions.

    The sum depends only on the Plan's immutable experiments, so it is parsed
    once per Plan object and kept in the Plan's derived-value cache. Later
    calls, such as one ``reduction_allowance`` call per point, read it
    without decoding the M pair parameters again.
    """
    import json
    from .pair_reducer import pair_work

    if "gcim_pair_work_total" not in plan._cache:
        plan._cache["gcim_pair_work_total"] = sum(
            pair_work(json.loads(point.parameters))
            for experiment in plan.experiments if experiment.readout is not None
            for point in experiment.readout.positions if point.kind == "reduction")
    return plan._cache["gcim_pair_work_total"]


def _exact_pair_construction(basis, operator, q, basis_record):
    """Return the blocks, definitions and experiments of the exact per-pair acquisition.

    Each off-diagonal pair ``(i, j)``, ``i < j``, prepares one phase-faithful
    joint state: H on the ancilla, X on the ancilla, ``U_i`` controlled on
    it, X on the ancilla and ``U_j`` controlled on it, so the ancilla's zero
    branch holds ``phi_i`` and its one branch ``phi_j``. Each diagonal pair
    prepares its system state once. Every experiment is one trajectory whose
    end point is the registered ``block_matrix_elements`` reduction of that
    pair, returning ``(H0_ij, S_ij)``. The ancilla is native bit 0.

    The ancilla flips use the selected extension of the one-qubit basis
    state |1>, whose direct constructor is exactly X on the full ancilla
    space. The controlled basis preparations keep their selected composite
    definitions and global phases. Receipt qualification follows the
    operations and parameters of the resulting native circuit.
    """
    from nwqlib.core.planning import ObservationPoint
    from .pair_reducer import REDUCER, pair_parameters

    b = len(basis)
    table = _nonidentity_table(operator)
    terms = len(table)
    flips = min(terms, 1 << q)
    c1 = _coefficient_mass(table.coefficients.tolist()).hex()
    h_state = ingest_vector(np.array([2**-0.5, 2**-0.5], dtype=complex))
    h = select_preparation("ancilla_h", h_state)
    x_state = ingest_vector(np.array([0.0, 1.0], dtype=complex))
    x = select_preparation("ancilla_x", x_state)
    forward, prepared = [], []
    for index, state in enumerate(basis):
        base = select_preparation(f"basis_{index}", state)
        prepared.append(base)
        if b > 1:
            forward.append(transform_block(f"forward_{index}", base, control=True))
    blocks = [h, x, *forward, *prepared]
    ancilla_blocks = (h, x)
    definitions = [
        Definition(id="allocate_ancilla", node=Allocate(wire="ancilla")),
        Definition(id="allocate_system", node=Allocate(wire="system")),
        Definition(id="release_ancilla", node=Release(wire="ancilla")),
        Definition(id="release_system", node=Release(wire="system")),
    ]
    for block in blocks:
        ports = tuple(
            PortMap(port=p.name,
                    wire="ancilla" if p.name == "control" or block in ancilla_blocks else "system")
            for p in block.record.signature.quantum
        )
        definitions.append(Definition(id=block.record.signature.name,
                                      node=BlockCall(signature=block.record.signature.name, ports=ports)))
    metadata = MetadataRef(format=METHOD, data=basis_record.reference)
    experiments = []
    for left, right in pair_acquisitions(b, terms):
        name = f"pair_{left}_{right}"
        if left == right:
            children = ["allocate_ancilla", "allocate_system", f"basis_{left}"]
        else:
            children = ["allocate_ancilla", "allocate_system", "ancilla_h", "ancilla_x",
                        f"forward_{left}", "ancilla_x", f"forward_{right}"]
        children += ["release_ancilla", "release_system"]
        body, batch = f"body_{name}", f"batch_{name}"
        definitions.append(Definition(id=body, node=Sequence(children=tuple(children))))
        definitions.append(Definition(id=batch, node=MeasurementBatch(
            body=body, settings=(Setting(label=name, metadata=metadata),), repetitions=1,
            observation_kind="trajectory")))
        parameters = pair_parameters(left=left, right=right, qubits=q, identity=operator.reference.identity,
                                     terms=terms, flips=flips, c1=c1)
        point = ObservationPoint(id="pair", kind="reduction", reducer=REDUCER, parameters=parameters)
        experiments.append(Experiment(name=name, batch=batch, setting_index=0,
                                      readout=ReadoutDetails(positions=(point,))))
    return blocks, definitions, experiments


def _check_projected_solve_work(b, cap, what):
    """Refuse a ``b``-by-``b`` projected solve whose work ``_projected_solve_work(b)`` exceeds ``cap``.

    The refusal names the needed work and the FixedGCIM field that caps it. A
    sampled Plan also checks its grouped analysis against the same field.
    """
    work = _projected_solve_work(b)
    if work > cap:
        raise ValueError(
            f"{what} needs {work} work units for basis size {b}, exceeding max_analysis_work={cap}. "
            f"Raise FixedGCIM.max_analysis_work to at least {work} or reduce the basis size."
        )


def sampled_groups(table, q, method):
    """Return the recorded QWC partition of the nonidentity rows of ``table``.

    The rows are partitioned by the deterministic first-fit owner
    ``PauliTerms.group(strategy="qwc")``, admitted by its own comparison and
    array law against ``max_bytes`` and ``max_analysis_work`` before any
    setting is expanded. Each group's basis carries the single non-I letter
    of its members on every qubit of the union of their supports.
    """
    rows = tuple((p, c) for p, c in table.rows() if p != "I" * q)
    if not rows:
        return ()
    packed = pauli_table(rows, num_qubits=q, max_bytes=method.max_bytes)
    partition = packed.group(strategy="qwc", max_bytes=method.max_bytes,
                             max_comparisons=method.max_analysis_work, limit_name="FixedGCIM.max_analysis_work")
    answer = []
    for indices in partition.groups:
        basis = ["I"] * q
        for j in indices:
            for k, letter in enumerate(rows[j][0]):
                if letter != "I":
                    if basis[k] not in ("I", letter):
                        raise ValueError("group is not QWC")
                    basis[k] = letter
        answer.append(SampledGroup(basis="".join(basis), indices=indices))
    return tuple(answer)


def sampled_settings(b, groups):
    """Enumerate ``(name, i, j, quadrature, group)`` of the declared sampled settings.

    An off-diagonal pair ``i < j`` has one setting per quadrature and system
    group, ``max(1, G)`` of each. Group zero supplies the overlap. A
    diagonal pair has one real setting per group. For M required pairs with
    D diagonal ones and G nonempty groups the count is
    ``2(M - D) max(1, G) + DG``, which is ``b**2 G`` for the complete
    b-state pencil.
    """
    for i in range(b):
        for j in range(i, b):
            for quad in (("real",) if i == j else ("real", "imag")):
                for g in range(len(groups) if i == j else max(1, len(groups))):
                    yield f"group_{i}_{j}_{quad}_{g}", i, j, quad, g


def _sampled_construction(basis, q, basis_record, groups, shots):
    """Return the blocks, definitions and experiments of the grouped sampled acquisition.

    An off-diagonal pair ``(i, j)`` prepares the exact-pair joint state:
    H on the ancilla, X on the ancilla, ``U_i`` controlled on it, X on the
    ancilla and ``U_j`` controlled on it, giving
    ``chi_ij = (|0> phi_i + |1> phi_j)/sqrt(2)``. For
    ``z_k = <phi_i|P_k|phi_j>``, direct multiplication of the two ancilla
    blocks gives ``<X_a P_k> = Re z_k`` and ``<Y_a P_k> = Im z_k``, with a
    positive Y sign. The X readout applies H and the Y readout S^dagger then
    H to the ancilla. A diagonal pair prepares ``phi_i`` directly and
    measures the system group only. Each group's system readout rotates X
    positions by H, Y positions by S^dagger then H, and leaves I and Z.

    The ancilla is native and classical bit zero. System qubit k is
    classical bit k + 1, so a displayed key reads ``system_bits phase``. Two
    shared classical declarations of common width ``n + 1`` keep every
    setting's layout equal. A diagonal setting leaves the ancilla classical
    bit at its initial zero and measures the full system register.
    Off-diagonal settings measure both registers. Every acquisition starts
    fresh with ``shots`` repetitions.
    """
    h = select_preparation("ancilla_h", ingest_vector(np.array([2**-0.5, 2**-0.5], dtype=complex)))
    x = select_preparation("ancilla_x", ingest_vector(np.array([0.0, 1.0], dtype=complex)))
    reads = [
        select_pauli_readout(f"read_{axis}", select_signed_pauli(f"axis_{axis}", ingest_pauli(((axis, 1.0),), num_qubits=1)))
        for axis in ("X", "Y")
    ]
    prepared = [select_preparation(f"basis_{i}", state) for i, state in enumerate(basis)]
    # A one-state basis has only its diagonal pair, which needs no controlled
    # preparation, as in the exact pair construction.
    forward = [transform_block(f"forward_{i}", base, control=True) for i, base in enumerate(prepared)] if len(basis) > 1 else []
    tails = [
        select_pauli_readout(f"system_read_{g}", select_signed_pauli(f"basis_read_{g}", ingest_pauli(((group.basis, 1.0),), num_qubits=q)))
        for g, group in enumerate(groups)
    ]
    blocks = [h, x, *reads, *prepared, *forward, *tails]
    definitions = [
        Definition(id=f"{verb}_{wire}", node=cls(wire=wire))
        for verb, cls in (("allocate", Allocate), ("release", Release))
        for wire in ("ancilla", "system")
    ]
    ancilla = {block.record.signature.name for block in (h, x, *reads)}
    for block in blocks:
        name = block.record.signature.name
        definitions.append(Definition(id=name, node=BlockCall(signature=name, ports=tuple(
            PortMap(port=p.name, wire="ancilla" if p.name == "control" or name in ancilla else "system")
            for p in block.record.signature.quantum))))
    definitions += [
        Definition(id="measure_phase", node=Measure(wire="ancilla", result="phase")),
        Definition(id="measure_system", node=Measure(wire="system", result="system_bits")),
    ]
    metadata = MetadataRef(format=METHOD, data=basis_record.reference)
    experiments = []
    for name, i, j, quad, g in sampled_settings(len(basis), groups):
        children = ["allocate_ancilla", "allocate_system"]
        children += ([f"basis_{i}"] if i == j else
                     ["ancilla_h", "ancilla_x", f"forward_{i}", "ancilla_x", f"forward_{j}"])
        if groups:
            children.append(f"system_read_{g}")
        if i != j:
            children += ["read_X" if quad == "real" else "read_Y", "measure_phase"]
        children += ["measure_system", "release_ancilla", "release_system"]
        body, batch = f"body_{name}", f"batch_{name}"
        definitions += [
            Definition(id=body, node=Sequence(children=tuple(children))),
            Definition(id=batch, node=MeasurementBatch(
                body=body, settings=(Setting(label=name, metadata=metadata),), repetitions=shots,
                observation_kind="counts")),
        ]
        experiments.append(Experiment(name=name, batch=batch, setting_index=0, readout=ReadoutDetails()))
    return blocks, definitions, experiments


def group_requirements(c, terms, entries):
    """Return the logical ``(bytes, work)`` of reducing one count table of ``entries`` stored bins.

    With c shared classical bits, ``w = ceil(c/64)`` packed words, m stored
    entries (explicit zero-count bins included) and ``L_g`` group terms,

        B_g = 8(w + 11)m + 8L_g,
        W_g = m{c + 28 + (4w + 4)L_g} + L_g{(c - 1)w + 1}.

    The first ``8(w + 1)m`` bytes are the decoded arrays. An additional 80m
    covers floating weights, X, Y, parity conversions and moment products,
    which occur in successive phases, and 8L_g holds the label means. Work
    covers key decoding, population checks, wordwise AND/popcount/parity
    passes, the coefficient accumulation, label dots and moment dots. The
    last term is the mask construction scanning each label once per word.
    These are conservative kernel/pass units. Python integer, list and
    record headers, decoder object temporaries, opaque NumPy workspace and
    RSS are outside this logical model, and the stored CountBin JSON
    reservation is a distinct stored-byte model.
    """
    w = (c + 63) // 64
    return 8 * (w + 11) * entries + 8 * terms, entries * (c + 4 + (4 * w + 4) * terms + 24) + terms * ((c - 1) * w + 1)


def sampled_analysis_requirements(b, n, groups, shots):
    """Bound one grouped analysis of one acquisition per declared setting.

    Each of G nonempty groups occurs b**2 times. With L total terms,
    c=n+1, w=ceil(c/64), and m=min(shots, 2**c), group_requirements
    supplies W_g and the live workspace 80*m+8*L_g. Summing cached
    decoded arrays, taking the largest workspace, and adding contribution,
    label and projected-solve payloads gives

        B = 8*(w+1)*b**2*G*m + 80*m + 8*max_g L_g
            + 8*b**2*(L+16*G) + (n+16)*L + 512*b**2,
        W = b**2*sum_g W_g + L*c + 32*b**2 + 16*b**3.

    The laws increase with bin population, so substitution of m bounds
    each pass. readout_shape sets m and _decoded_chunk enforces it on
    backend output, including stored zero-count bins. The common c-bit
    envelope also covers the unused phase bit of a diagonal setting.
    read_groups still admits actual populations on every analysis pass,
    including repeated or supplied chunks. Units and exclusions match
    group_requirements and read_groups. An empty partition is the
    scalar-identity shortcut and has no grouped pass.
    """
    if not groups:
        return 0, 0
    c = n + 1
    w = (c + 63) // 64
    bins = shots if c >= shots.bit_length() else 1 << c
    terms = sum(len(group.indices) for group in groups)
    per_group_work = 0
    workspace = 0
    for group in groups:
        size, work = group_requirements(c, len(group.indices), bins)
        per_group_work += work
        workspace = max(workspace, size - 8 * (w + 1) * bins)
    copies = b * b
    size = (8 * (w + 1) * copies * len(groups) * bins + workspace
            + 8 * copies * (terms + 16 * len(groups)) + (n + 16) * terms
            + _projected_solve_bytes(b))
    work = copies * per_group_work + terms * c + _projected_solve_work(b)
    return size, work


def sampled_analysis_work(b, n, terms, groups, shots):
    """Return the grouped-analysis work A(G) from counts known before grouping.

    For basis size b, L = ``terms`` nonidentity terms, n system qubits and
    s = ``shots`` per setting, write c=n+1, w=ceil(c/64) and m=min(s,2^c).
    Summing ``group_requirements(c, L_g, m)`` over the G = ``groups``
    groups, using sum L_g = L, gives the work of
    ``sampled_analysis_requirements``:

        A(G) = b²{m[(c+28)G + (4w+4)L] + L[(c−1)w+1]} + Lc + S_b,

    with S_b = 32b²+16b³ the projected solve (``_projected_solve_work``).
    For L > 0, 1 <= G <= L and A is monotone in G, so A(1) is a necessary
    analysis-work floor and A(L) a sufficient analysis envelope before
    grouping. After grouping A(G) is the exact admission charge. Identity
    terms do not enter L. Sampled FixedGCIM checks grouping, the projected
    solve and grouped analysis against ``max_analysis_work``. These are
    phase ceilings, so the required common limit is their maximum,
    ``R = max(L(L-1)/2, S_b, A(L))``, not their sum.
    """
    if not terms:
        return 0
    c = n + 1
    w = (c + 63) // 64
    bins = shots if c >= shots.bit_length() else 1 << c
    return (
        b*b * (bins*((c + 28)*groups + (4*w + 4)*terms)
               + terms*((c - 1)*w + 1))
        + terms*c + _projected_solve_work(b)
    )


def reduce_group(chunk, rec, setting, rows=None):
    """Return the ``GroupMoment`` of one count chunk, or None for a zero population.

    With ``a = (-1)**phase`` and ``p_k`` the system parity of the non-I
    positions of label k after the group's basis rotation, one shot
    contributes ``Z_k = a p_k`` off diagonal and ``Z_k = p_k`` on the
    diagonal, and ``Y_g = sum_(k in g) c_k Z_k``. All counts belong to the
    unconditional population, so selected shots equal returned shots. The
    common width, the diagonal's unused phase bit and the exact count total
    against ``returned_shots`` are checked, and packed masks cover every
    word of the key.
    The label means satisfy ``mean(Y_g) = sum_(k in g) c_k mean(Z_k)``
    mathematically, with ordinary reduction rounding in its evaluation.
    Each H0 quadrature is formed by summing the pooled weighted group
    means ``mean(Y_g)``.
    """
    name, i, j, quad, g = setting
    group = rec.groups[g] if rec.groups else SampledGroup(basis="I" * rec.num_qubits, indices=())
    histogram = chunk.histogram()
    if histogram.width != rec.num_qubits + 1:
        raise ValueError("GCIM counts must match the shared phase/system classical layout")
    words = histogram.packed_indices()
    counts = histogram.weights
    total = sum(map(int, counts))
    if total != chunk.returned_shots:
        raise ValueError("GCIM count population differs from returned_shots")
    if total == 0:
        return None
    if i == j and np.any(words[:, 0] & np.uint64(1)):
        raise ValueError("GCIM diagonal setting has a nonzero unused phase bit")
    weights = counts.astype(float) / total
    x = 1.0 - 2.0 * (words[:, 0] & np.uint64(1)).astype(float)
    y = np.zeros(len(counts), float)
    means = []
    rows = rec.nonidentity_terms if rows is None else rows
    for index in group.indices:
        label, coefficient = rows[index]
        parity = np.zeros(len(counts), np.uint8)
        # System qubit k is classical bit k + 1. The ancilla sign follows.
        for word in range(words.shape[1]):
            mask = 0
            for bit, letter in enumerate(reversed(label)):
                physical = bit + 1
                if letter != "I" and physical // 64 == word:
                    mask |= 1 << (physical % 64)
            parity ^= (np.bitwise_count(words[:, word] & np.uint64(mask)) & 1).astype(np.uint8)
        z = 1.0 - 2.0 * parity.astype(float)
        if i != j:
            z *= x
        means.append(float(np.dot(weights, z)))
        y += coefficient * z
    mean = float(np.dot(weights, y))
    second = float(np.dot(weights, y * y))
    variance = None if total == 1 else (second - mean * mean) / (total - 1)
    supplies = i != j and g == 0
    xm = float(np.dot(weights, x)) if supplies else None
    xy = float(np.dot(weights, x * y)) if supplies else None
    vx = (None if total == 1 else (1.0 - xm * xm) / (total - 1)) if supplies else None
    cov = (None if total == 1 else (xy - xm * mean) / (total - 1)) if supplies else None
    return GroupMoment(
        experiment=name, contribution_id=chunk.content_id, pair=(i, j), quadrature=quad, group=g,
        shots=total, label_means=tuple(means), mean=mean, second=second, mean_variance=variance,
        overlap_mean=xm, overlap_cross=xy, overlap_variance=vx, mean_covariance=cov,
        variance_negative=any(v is not None and v < 0 for v in (variance, vx)),
    )


def pooled_statistics(rec, moments):
    """Pool the group moments per setting and reconstruct the pair entries and their variances.

    Repeated acquisitions a of one setting pool with ``w_a = N_a/sum_b N_b``:
    the mean is ``sum_a w_a mean_a``, the variance ``sum_a w_a**2 V_a`` and
    the covariance ``sum_a w_a**2 C_a``, because overlap and H0 use the same
    weights here. The quadratures are ``H0_ij,q = sum_g mean(Y_ij,q,g)`` and
    ``S_ij,q = mean(X_ij,q,0)``, with ``S_ii = 1`` and zero imaginary
    diagonal. The variances follow ``SampledPairVariance``. Pooling
    uncertainty is conditional on an independent-acquisition model. Shared
    deterministic simulator seeds do not prove independence. Returns
    ``(estimates, missing, pairs, variances)``, with empty pairs and
    variances when a setting is missing.
    """
    by_name = {s[0]: [] for s in sampled_settings(rec.basis_size, rec.groups)}
    for row in moments:
        by_name[row.experiment].append(row)
    estimates, pooled = [], {}
    for name, rows in by_name.items():
        if not rows:
            continue
        n = sum(r.shots for r in rows)
        weights = tuple(r.shots / n for r in rows)

        def average(field, squared=False, rows=rows, weights=weights):
            values = [getattr(r, field) for r in rows]
            if any(v is None for v in values):
                return None
            return fsum((w * w if squared else w) * v for w, v in zip(weights, values, strict=True))

        pooled[name] = tuple(average(field, squared=k >= 2) for k, field in enumerate(
            ("mean", "overlap_mean", "mean_variance", "overlap_variance", "mean_covariance")))
        estimates.append(ScalarEstimate(
            experiment=name, value=pooled[name][0], contribution_ids=tuple(r.contribution_id for r in rows),
            weights=weights, chunk_mean_variances=tuple(r.mean_variance for r in rows), sampled_shots=n))
    missing = tuple(name for name in by_name if name not in pooled)
    if missing:
        return tuple(estimates), missing, {}, ()
    pairs, variances, c = {}, [], rec.identity_shift

    def vsum(values):
        return None if any(v is None for v in values) else fsum(values)

    for i in range(rec.basis_size):
        for j in range(i, rec.basis_size):
            h, s, vh, vs, cov, physical = [], [], [], [], [], []
            for quad in ("real", "imag"):
                if i == j and quad == "imag":
                    for column in (h, s, vh, vs, cov, physical):
                        column.append(0.0)
                    continue
                count = len(rec.groups) if i == j else max(1, len(rec.groups))
                values = [pooled[f"group_{i}_{j}_{quad}_{g}"] for g in range(count)]
                h.append(fsum(v[0] for v in values))
                s.append(1.0 if i == j else values[0][1])
                vh.append(vsum([v[2] for v in values]))
                vs.append(0.0 if i == j else values[0][3])
                cov.append(0.0 if i == j else values[0][4])
                physical.append(None if vh[-1] is None or (c != 0 and (vs[-1] is None or cov[-1] is None))
                                else vh[-1] + (c * c * vs[-1] + 2 * c * cov[-1] if c else 0.0))
            pairs[i, j] = (complex(*h), complex(*s))
            variances.append(SampledPairVariance(
                pair=(i, j), h0=tuple(vh), overlap=tuple(vs), covariance=tuple(cov), physical_h=tuple(physical),
                variance_negative=any(v is not None and v < 0 for v in (*vh, *vs, *physical))))
    return tuple(estimates), missing, pairs, tuple(variances)


def read_groups(plan, data):
    """Admit one complete grouped analysis pass, then reduce every matched count chunk.

    ``matched_chunks`` checks each contribution's association before use.
    The pass is admitted before the first histogram decode, from the stored
    bin count ``len(chunk.values)`` of each chunk (explicit zero-count bins
    included). ``ObservationChunk.histogram()`` caches its arrays on the
    chunk, so every decoded table may stay alive after the pass:

        B_analysis = sum_a 8(w + 1)m_a + max_a(80m_a + 8L_a)
                     + 8 sum_a(L_a + 16) + (n + 16)L + 512b**2,

    where the third term is a logical payload allowance for the
    contribution scalars and indices, and the fourth the once-decoded
    nonidentity label and coefficient payload. The projected solve's own
    allowance is added. The pass work ``sum_a W_a + L(n + 1) + 32b**2 +
    16b**3`` is checked against ``max_analysis_work``. Repeated acquisitions
    enter these sums separately. Stored observations and the immutable
    reconstruction keep their own admission, and this is not a bound on the
    whole Python object graph.
    """
    rec = plan.reconstruction
    expected = {s[0]: s for s in sampled_settings(rec.basis_size, rec.groups)}
    moments, used, decoded, workspace, records = [], 0, 0, 0, 0
    c = rec.num_qubits + 1
    w = (c + 63) // 64
    matched = tuple(chunk for chunk, _ in matched_chunks(plan, data))
    for chunk in matched:
        s = expected[chunk.experiment]
        terms = len(rec.groups[s[-1]].indices) if rec.groups else 0
        bins = len(chunk.values)
        size, work = group_requirements(c, terms, bins)
        decoded += 8 * (w + 1) * bins
        workspace = max(workspace, size - 8 * (w + 1) * bins)
        records += 8 * (terms + 16)
        used += work
    label_payload = (rec.num_qubits + 16) * len(rec.nonidentity_coefficients)
    _check_bytes(decoded + workspace + records + label_payload + _projected_solve_bytes(rec.basis_size),
                 plan.method.max_bytes, "GCIM grouped count arrays and scalar payload")
    used += len(rec.nonidentity_coefficients) * (rec.num_qubits + 1) + _projected_solve_work(rec.basis_size)
    if used > plan.method.max_analysis_work:
        raise ValueError(f"GCIM grouped analysis needs {used} work units, exceeding "
                         f"max_analysis_work={plan.method.max_analysis_work}")
    rows = rec.nonidentity_terms
    for chunk in matched:
        row = reduce_group(chunk, rec, expected[chunk.experiment], rows)
        if row is not None:
            moments.append(row)
    return tuple(moments)


class FixedGCIMResult(Result):
    """Smallest projected eigenvalue from a `FixedGCIM` run, with its projected matrices.

    [`solve`][nwqlib.scientist.solve] returns it for a `FixedGCIM` method,
    and `load_result` reopens a saved one. The answer is `eigenvalue`, the
    lowest kept Ritz value of the projected pencil, in the unit of the
    `Eigenproblem`. It is None when data are missing or the solve was
    refused. It is a projected estimate, not a certified full-space ground
    energy. `pencil` holds the projected matrices H and S and the solve
    diagnostics. `print(result)` shows the value and its scope, and
    `result.analyze(overlap_cutoff=...)` solves the same matrix elements at
    another cutoff without new measurement. Whenever a Result is attached to
    its Plan and data, as after loading or reanalysis, NWQLib rejects a value
    that differs from its own pencil, a pencil built under another
    measurement policy, and any estimate formed from incomplete data. The
    fields below are read-only. The fields of
    [`Result`][nwqlib.core.analysis.Result] are present too.

    Attributes:
        eigenvalue: Lowest kept Ritz value in the problem's energy unit, or
            None when data are missing or the solve was refused. It is not a
            certified full-space ground energy.
        missing: Names of planned measurements with no returned data. A
            nonempty tuple means no pencil was solved.
        pairs: Exact pair measurements: pooled `H0_ij` and `S_ij` with their
            reductions and entry error bounds. Empty for sampled and
            classical Plans.
        estimates: For a sampled Plan, the pooled weighted group means
            ([`ScalarEstimate`][nwqlib.algorithms.gcim.fixed_basis.ScalarEstimate])
            with their contributing count data, weights and the estimated
            variance of each contribution's mean.
        group_moments: For a sampled Plan, the moments of every count
            measurement, in measurement order.
        sampled_pair_variances: For a complete sampled pencil, the pooled
            sample-mean variances and overlap–H0 covariance of each pair.
        pencil: The [`ProjectedPencil`][nwqlib.algorithms.gcim.fixed_basis.ProjectedPencil]
            with the projected matrices and solve diagnostics, or None for
            partial data or a scalar identity target.
        analysis_cutoff: Overlap-eigenvalue cutoff used by this analysis.
        target_identification: Fixed statement of what the value does not establish.
    """

    eigenvalue: Real | None
    missing: tuple[Text, ...]
    schema_version: Literal[4] = 4
    estimates: tuple[ScalarEstimate, ...]
    pairs: tuple[PairEstimate, ...] = ()
    group_moments: tuple[GroupMoment, ...] = ()
    sampled_pair_variances: tuple[SampledPairVariance, ...] = ()
    pencil: ProjectedPencil | None
    analysis_cutoff: Annotated[Real, Field(gt=0)]
    target_identification: Text = (
        "projected estimate; smallest full-space eigenvalue not established"
    )

    def _summary_lines(self):
        lines = [
            f"Ritz eigenvalue: {self._scalar_text(self.eigenvalue)}{self._unit_text()}",
            self.target_identification,
        ]
        if self.missing:
            lines.append(f"Partial data: {len(self.missing)} missing transition measurements")
        elif self.pencil is not None and self.pencil.failure_reason:
            lines.append("Analysis unavailable: " + self.pencil.failure_reason)
            if (self.pencil.failure_reason == "negative_deterministic_overlap_eigenvalue"
                    and any(p.overlap_bound is None for p in self.pairs if p.pair[0] != p.pair[1])):
                lines.append(UNQUALIFIED_GRAM_REFUSAL)
        if self.pencil is not None:
            lines.extend(self.pencil._sampled_summary())
        return lines

    def validate_plan(self, plan: Plan) -> None:
        """Reject this Result unless its value and pencil follow from ``plan``.

        The checks, in order: the same Plan and construction, no value from
        partial observations, the settings of a sampled Plan with nonidentity
        terms equal to those of its recorded group partition, group
        statistics only on such a Plan, no exact pair reductions on such a
        Plan, the identity coefficient as the value of a
        quantum Plan whose operator is a multiple of the identity, a pencil of
        the Plan's basis size and analysis cutoff, a value equal to the
        pencil's lowest Ritz value, the overlap filter of the Plan's
        acquisition (``sampled_positive_subspace`` exactly when shots were
        taken), and sampled enclosure fields recomputed from the pencil. No
        acquisition or eigensolve is repeated.

        Args:
            plan: The Plan this Result claims to analyze.

        Raises:
            ValueError: If any check fails.
        """
        self._validate_common_plan(plan, Plan)
        if self.construction_id != plan.construction.content_id:
            raise ValueError("GCIM result differs from its actual selected construction")
        if self.missing and (self.eigenvalue is not None or self.pencil is not None):
            raise ValueError("partial observations cannot produce a projected estimate")
        rec = plan.reconstruction
        # A sampled Plan with nonidentity terms acquires exactly the settings
        # of its recorded group partition.
        if plan.execution == "quantum" and plan.shots is not None and rec.nonidentity_coefficients and (
            not rec.groups
            or tuple(e.name for e in plan.experiments)
            != tuple(s[0] for s in sampled_settings(rec.basis_size, rec.groups))
        ):
            raise ValueError("GCIM sampled Plan differs from its declared group settings")
        grouped = plan.execution == "quantum" and plan.shots is not None and bool(rec.nonidentity_coefficients)
        if not grouped and (rec.groups or self.group_moments or self.sampled_pair_variances):
            raise ValueError("GCIM group statistics belong only to a sampled Plan with nonidentity terms")
        if grouped and self.pairs:
            raise ValueError("a sampled GCIM Result holds no exact pair reductions")
        # Empty quantum Pauli remainder is the selected algebraic cI branch.
        # Classical reconstruction has no Pauli table, so emptiness there is
        # not evidence of a scalar operator.
        if (
            plan.execution == "quantum"
            and not plan.reconstruction.nonidentity_coefficients
            and self.eigenvalue != plan.reconstruction.identity_shift
        ):
            raise ValueError("GCIM identity eigenvalue differs from its exact constant")
        if self.pencil is not None:
            p = self.pencil
            m = plan.reconstruction.basis_size
            if (
                len(p.hamiltonian) != m
                or len(p.overlap) != m
                or p.overlap_cutoff != self.analysis_cutoff
            ):
                raise ValueError("GCIM pencil differs from its basis or analysis settings")
            if self.eigenvalue != (p.eigenvalues[0] if p.eigenvalues else None):
                raise ValueError("GCIM eigenvalue differs from its projected spectrum")
            p._validate_sampled_enclosure(plan.reconstruction, sampled=plan.execution == "quantum" and plan.shots is not None)

    def energy_endpoint(self):
        """Return this Result's ``EnergyEndpoint`` for energy-shift verification.

        The endpoint carries the projected value, the selected operator's
        payload (Pauli terms or matrix entries) and the identities of the
        normalized basis preparations. The ``H -> H + cI`` check of
        ``nwqlib.evidence.energy_shift`` compares two such endpoints without
        new acquisition.
        """
        from nwqlib.evidence.energy_shift import EnergyEndpoint, _operator_payload

        plan = self.plan
        operator = plan._native["operator"]
        return EnergyEndpoint(
            plan_id=self.plan_id,
            result_id=self.content_id,
            construction_id=self.construction_id,
            operator_id=plan.problem.A.reference.identity,
            basis=plan.problem.basis,
            unit=plan.output.frame(plan.problem).unit,
            preparations=tuple(state.preparation.content_id for state in plan._native["basis"]),
            subspace_rule="fixed_preparations",
            order=plan.reconstruction.basis_size,
            sector=plan.problem.sector,
            frame=plan.output.frame(plan.problem),
            **_operator_payload(operator, max_bytes=plan.method.max_bytes),
            value=self.eigenvalue,
            analyzer=None if self.origin is None else self.origin.analyzer,
        )

    def projected_diagnostics(self):
        """Return the stored Gram matrix and solve diagnostics for projected checks.

        It reads the stored `pencil` and runs no new solve.
        `result.verify(checks=ProjectedVerificationOptions(...))` uses it.

        Returns:
            diagnostics (ProjectedDiagnostics): Its `overlap`, `spectrum`,
                `normalization` and `backward_error` are the pencil's
                `overlap`, `overlap_eigenvalues`, `overlap_normalization_error`
                and `projected_backward_error`, the last on the physical
                pencil (H, S). All four are None when there is no pencil.
        """
        from nwqlib.evidence.verification import ProjectedDiagnostics

        p = self.pencil
        return (
            ProjectedDiagnostics(None, None, None, None)
            if p is None
            else ProjectedDiagnostics(
                p.overlap,
                p.overlap_eigenvalues,
                p.overlap_normalization_error,
                p.projected_backward_error,
            )
        )


def solve_pencil(hamiltonian, overlap, *, cutoff, sampled, enclosure=None, identity_shift=0.0,
                 overlap_input_tolerance=0.0):
    """Solve one acquired FixedGCIM pencil and return it as a ``ProjectedPencil``.

    ``hamiltonian`` is ``H0``, the acquired projection without the identity
    term ``identity_shift*I``. The solve uses ``(H0, S)`` and adds the shift
    to the Ritz values afterwards (``_solve_projected_pencil``), and the
    record stores ``H = H0 + identity_shift*S`` in the problem's energy
    unit. The overlap is the Gram matrix of normalized columns and is used as
    acquired. Lanczos instead solves its moment pencil in the normalized frame
    ``K = (H - center I)/alpha`` and restores the Ritz values afterwards. The
    pencil arrives Hermitian by conjugate completion, so the solver's
    symmetrization is disabled and the stored matrices are the acquired ones.
    ``sampled`` selects the indefinite-overlap policy of
    ``_solve_projected_pencil``, and ``overlap_input_tolerance`` is the
    derived error of a deterministic Gram matrix that its negative-mode test
    admits. ``enclosure`` enables the sampled plausibility diagnostic.
    ``overlap_condition_number`` is the ratio of the largest to the smallest
    kept overlap eigenvalue, or None when no direction is kept or the ratio
    is not finite.
    """
    result, spectrum, failure, _ = _solve_projected_pencil(
        hamiltonian,
        overlap,
        overlap_eigenvalue_cutoff=cutoff,
        symmetrize_matrices=False,
        overlap_input_tolerance=overlap_input_tolerance,
        identity_shift=identity_shift,
        _sampled_overlap=sampled,
    )
    kept = [] if spectrum is None else [float(x) for x in spectrum if x > cutoff]
    condition = max(kept) / min(kept) if kept else None
    if condition is not None and not isfinite(condition):
        condition = None

    def matrix_record(matrix):
        return tuple(
            tuple(Complex128(real=float(z.real), imag=float(z.imag)) for z in row) for row in matrix
        )

    diagnostics = None if not sampled or enclosure is None else _sampled_pencil_diagnostics(
        eigensolver=result, solver_failure_reason=failure,
        enclosure=enclosure, relative_tolerance=SAMPLED_PENCIL_ENCLOSURE_RTOL)
    pencil = ProjectedPencil(
        hamiltonian=matrix_record(physical_hamiltonian(hamiltonian, overlap, identity_shift)),
        overlap=matrix_record(overlap),
        overlap_eigenvalues=() if spectrum is None else tuple(float(x) for x in spectrum),
        eigenvalues=() if result is None else tuple(float(x) for x in result.eigenvalues),
        coordinate_vectors=() if result is None else matrix_record(result.eigenvectors),
        kept_rank=0 if result is None else result.kept_overlap_rank,
        overlap_cutoff=cutoff,
        overlap_filter="sampled_positive_subspace" if sampled else "deterministic_gram",
        overlap_condition_number=condition,
        projected_residual=None if result is None else result.residual_norm,
        projected_backward_error=None
        if result is None
        else result.generalized_eigenpair_backward_error,
        overlap_normalization_error=None if result is None else result.overlap_normalization_error,
        failure_reason=failure,
        **sampled_pencil_fields(diagnostics),
    )
    return pencil


class FixedGCIM(Method):
    """Generator-coordinate (GCiM) method for the smallest eigenvalue in a fixed trial basis.

    Build it with keyword arguments and pass it as `method=`, for example
    `solve(Eigenproblem(A=matrix), method=FixedGCIM(basis=(phi_1, phi_2)))`.
    `basis` is the only required argument. The result is a
    [`FixedGCIMResult`][nwqlib.algorithms.gcim.fixed_basis.FixedGCIMResult]
    whose `eigenvalue` is the lowest Ritz value of the projected problem
    `H f = E S f`, with `H_ij = <phi_i|A|phi_j>` and `S_ij = <phi_i|phi_j>`
    for the normalized trial states phi_i. This is the discretized
    Hill-Wheeler problem of Zheng et al., Phys. Rev. Research 5, 023200
    (2023), arXiv:2212.09205v1, Eq. (13), with matrix elements
    Eqs. (14)-(15). The trial states are the supplied states, normalized,
    instead of the paper's states `exp(Gamma(Z_p))|Phi>` at discretized
    generator coordinates `Z_p` (Eq. (10), discretized in Eq. (17)). A
    projected Ritz value does not certify the full-space ground energy.

    The call chooses how H and S are obtained, for M basis pairs of which D
    are diagonal:

    - Default, exact quantum: one phase-faithful joint state per
      off-diagonal pair and one system state per diagonal pair, each reduced
      when it is measured to the offset-free `H0_ij` and `S_ij`, so
      `(M - D) + D` measurements.
    - `shots=n`: the same joint state per off-diagonal pair, with its
      ancilla measured in X and Y together with each qubit-wise commuting
      group of system terms, and each diagonal pair's system state per group,
      so `2(M - D) max(1, G) + DG` settings for G groups, n shots each. Each
      count table supplies its group's weighted Hamiltonian mean and, for
      the first group, the overlap and its covariance with that mean.
    - `execution="classical"`: `V^dagger (A - cI) V` and `V^dagger V`
      computed directly, with `c = trace(A)/d`.

    Every path solves without the identity term and adds c to the Ritz
    values afterwards. Sampled Ritz values are raw estimates, and the
    enclosure diagnostic in `result.pencil` does not certify their physical
    accuracy. `overlap_cutoff` can be changed after the run with
    `result.analyze(overlap_cutoff=...)`, which reuses the same matrix
    elements. The [GCiM guide](../../algorithms/gcim.md) describes the matrix
    elements, their error bounds and the solve.

    Attributes:
        basis: Required. Trial states phi_i, a nonempty tuple of states in a
            form that `StateData` accepts. Each is normalized.
        overlap_cutoff: Default `1e-12`, positive. Overlap eigenvectors with
            eigenvalues above it are kept for the solve.
        max_basis_size: Default `64`. Largest accepted number of trial
            states, checked before pairs are formed.
        max_experiments: Default `100_000`. Largest number of measurement
            settings: exact pair reductions, or sampled group settings.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Limit on
            the known bytes of conversion, the pair table, the grouped count
            analysis and the projected solve, and on each exact pair
            reduction's workspace, checked when that pair is prepared.
        max_analysis_work: Default `1_000_000_000`. Limit on the work of the
            projected matrices and eigensolve, of the comparisons that form
            qubit-wise commuting groups, and of each pass of the grouped count
            analysis. Sampled planning checks the worst-case pass for one
            measurement per setting before it builds circuits, and analysis
            checks the stored counts again before decoding them. The default
            is ten times the other work defaults because that planning bound
            is conservative ([Engineering constants](../../ENGINEERING_CONSTANTS.md),
            row "Sampled FixedGCIM analysis admission").
        max_classical_products: Default `100_000_000`. Limit on classical
            operator-vector products and on the summed classical work of the
            exact pair reductions.
        input_conversion: Default `"auto"`, which keeps the accepted input
            access. `"dense_pauli"` permits explicit dense-to-Pauli
            conversion.
        max_conversion_work: Default `100_000_000`. Limit on the work of an
            explicitly chosen operator conversion.
        max_admission_steps: Default `1_000_000`. Upper limit on the
            planning work of checking each `Program` that the method builds
            (NWQLib's description of a circuit as named steps). It caps the
            number of stored fields of a `Program` and the work units of one
            check of it. Summing a `Program`'s resource counts may use up to
            24 times this value. A larger `Program` is refused with a
            ValueError that names this field, the refused stage and its
            count. Raising the limit permits a larger check and changes no
            quantum operation
            ([planning work limit](../../development/program_checks.md#planning-work-limit)).

    Examples:
        `H = ZZ + 0.5*(XI + IX)` on two qubits has lowest eigenvalue
        `-sqrt(2) = -1.4142135623...`, which this three-state basis
        recovers on Aer with exact readout:

        >>> from qiskit.quantum_info import SparsePauliOp
        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import FixedGCIM
        >>> H = SparsePauliOp.from_list([("ZZ", 1), ("XI", 0.5), ("IX", 0.5)])
        >>> basis = ([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0])
        >>> result = solve(Eigenproblem(A=H), method=FixedGCIM(basis=basis), seed=7)
        >>> print(round(result.eigenvalue, 10))
        -1.4142135624
    """

    schema_version: Literal[2] = 2
    basis: Annotated[tuple[StateData, ...], Field(min_length=1)]
    overlap_cutoff: Annotated[Real, Field(gt=0)] = 1e-12
    # max_basis_size, max_experiments, max_analysis_work and
    # max_classical_products are untuned workload ceilings on derived counts,
    # checked before expansion (docs/ENGINEERING_CONSTANTS.md). Revisit for an
    # explicitly intended larger basis or classical workload.
    max_basis_size: PositiveInt = 64
    max_experiments: PositiveInt = 100_000
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_analysis_work: PositiveInt = 1_000_000_000
    max_classical_products: PositiveInt = 100_000_000
    input_conversion: Literal["auto", "dense_pauli"] = "auto"
    max_conversion_work: PositiveInt = DEFAULT_CONVERSION_WORK
    max_admission_steps: PositiveInt = 1_000_000
    result_type: ClassVar[type] = FixedGCIMResult

    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR

    def plan(self, problem, *, output, execution, shots, rng):
        """Select normalized trial columns and their complex H/S matrix-element acquisitions."""
        if execution == "classical":
            if shots is not None:
                raise ValueError("classical projected matrices do not use shots")
            return self._plan_classical(problem, output, rng)
        # Bound both pair expansion and projected solve before selecting circuit blocks.
        # The byte and work laws below are the "Fixed GCIM projected size law"
        # of docs/ENGINEERING_CONSTANTS.md.
        b = len(self.basis)
        if b > self.max_basis_size:
            raise ValueError(f"basis has {b} states, more than FixedGCIM.max_basis_size={self.max_basis_size}, before pair expansion")
        operator = eigen_operator(
            problem,
            output,
            max_bytes=self.max_bytes,
            input_conversion=self.input_conversion,
            max_conversion_work=self.max_conversion_work,
        )
        basis = tuple(eigen_state(state, problem, max_bytes=self.max_bytes) for state in self.basis)
        basis_record = FixedGCIMBasis(preparations=tuple(state.preparation for state in basis))
        if problem.subspace is not None and problem.subspace != basis_record.reference:
            raise ApplicabilityError("requested subspace differs from this fixed normalized basis")
        q = operator.basis.dimension.bit_length() - 1
        from .adapt_inputs import _packed_to_pauli_arrays, pauli_array_conversion_requirements

        table = operator.pauli_terms()
        # The conversion to the recorded letter-code table is admitted before it
        # allocates, with the 64L bytes of the packed nonidentity table that the
        # exact pair reductions later hold (_nonidentity_table).
        _check_bytes(pauli_array_conversion_requirements(len(table), q)[1] + 64 * len(table), self.max_bytes,
                     "GCIM Pauli term table")
        input_terms = _packed_to_pauli_arrays(table.x, table.z, table.coefficients.real, q)
        nonidentity = len(input_terms.coefficients.array) - int((~input_terms.labels.array.any(axis=1)).any())
        # The projected solve and the one-group analysis floor are known
        # requirements before the actual grouping and setting construction.
        solve_need = _projected_solve_work(b)
        if shots and nonidentity:
            analysis_floor = sampled_analysis_work(b, q, nonidentity, 1, shots)
            if max(solve_need, analysis_floor) > self.max_analysis_work:
                raise ValueError(
                    f"FixedGCIM known analysis requirements need at least "
                    f"{max(solve_need, analysis_floor)} work units, exceeding "
                    f"max_analysis_work={self.max_analysis_work}. "
                    "Increase FixedGCIM.max_analysis_work or reduce the basis or operator."
                )
        # A sampled Plan records its QWC partition before settings expansion.
        # The scalar-identity shortcut acquires no pencil and needs no groups.
        groups = sampled_groups(input_terms, q, self) if shots and nonidentity else ()
        e = ((b * b * len(groups) if shots else pair_count(b, nonidentity)) if nonidentity else 0)
        if e > self.max_experiments:
            raise ValueError(
                f"GCIM acquisition population {e} exceeds max_experiments={self.max_experiments} "
                "before expansion"
            )
        # Bytes: the projected-solve allowance of the b-by-b pencil plus an
        # untuned 256 bytes per setting for its name, Setting, body/batch
        # Definitions and Experiment record, and for a sampled Plan the
        # G*n + 8L logical basis and index bytes of its group description.
        # 256E is an untuned record allowance, not a derived Python-heap bound.
        _check_bytes(
            _projected_solve_bytes(b) + 256 * e
            + sum(q + 8 * len(group.indices) for group in groups),
            self.max_bytes,
            "GCIM selected pairs and projected workspace",
        )
        _check_projected_solve_work(b, self.max_analysis_work, "projected solve")
        if e and shots:
            analysis_bytes, analysis_work = sampled_analysis_requirements(b, q, groups, shots)
            _check_bytes(analysis_bytes, self.max_bytes,
                         "GCIM planned grouped analysis")
            known = max(solve_need, analysis_work)
            if known > self.max_analysis_work:
                raise ValueError(
                    f"GCIM planned grouped analysis needs at most {known} work units "
                    f"for one acquisition per setting, the known complete analysis "
                    f"requirement, exceeding max_analysis_work={self.max_analysis_work}. "
                    f"Raise FixedGCIM.max_analysis_work to at least {known} or reduce "
                    "shots, the basis size or the grouped observable."
                )
        blocks, definitions, experiments = [], [], []
        if e and shots:
            blocks, definitions, experiments = _sampled_construction(basis, q, basis_record, groups, shots)
            registers = (
                Register(name="ancilla", width=1, role="clean_ancilla"),
                Register(name="system", width=q),
            )
            classical = (
                ClassicalValue(name="phase", dtype="bits", width=1),
                ClassicalValue(name="system_bits", dtype="bits", width=q),
            )
        elif e:
            blocks, definitions, experiments = _exact_pair_construction(basis, operator, q, basis_record)
            registers = (
                Register(name="ancilla", width=1, role="clean_ancilla"),
                Register(name="system", width=q),
            )
            classical = ()
        else:
            registers, classical = (), ()
        definitions.append(
            Definition(
                id="root",
                node=Sequence(children=tuple(exp.batch for exp in experiments)),
            )
        )
        program = admitted_program(
            "FixedGCIM.max_admission_steps",
            self.max_admission_steps,
            root="root",
            definitions=tuple(definitions),
            registers=registers,
            classical=classical,
            signatures=tuple(block.record.signature for block in blocks),
        )
        construction = SelectedConstruction(
            program=program,
            selections=tuple(block.record for block in blocks),
        )
        # Keep identity terms for H_identity = c S rather than acquiring them again.
        reconstruction = FixedGCIMReconstruction(terms=input_terms, basis_size=b, num_qubits=q, groups=groups)
        from types import SimpleNamespace
        from nwqlib.algorithms._eigen_support import eigen_error_model

        # eigen_error_model reads only the output, problem and construction,
        # so the Plan is validated once with its error model.
        components = SimpleNamespace(output=output, problem=problem, construction=construction)
        plan = Plan(
            problem=problem,
            method=self,
            output=output,
            execution=execution,
            shots=shots,
            randomness=rng.snapshot(),
            construction=construction,
            experiments=tuple(experiments),
            reconstruction=reconstruction,
            assumptions=(
                "normalized preparation columns",
                "H_identity=c*S uses the same overlap observations",
                "projected estimate does not establish smallest full-space eigenvalue",
            ),
            error_model=eigen_error_model(components, METHOD),
        )
        if e and not shots:
            # The summed registered work of the pair reductions is recorded
            # once in the Plan's cache. reduction_allowance admits it against
            # max_classical_products, the Method's host-work limit, when a
            # pair is prepared, and funds each point from this ledger.
            _pair_work_total(plan)
            return plan._bind(blocks=blocks, operator=operator, basis=basis,
                              pair_table=_pair_table(operator))
        return plan._bind(blocks=blocks, operator=operator, basis=basis)

    def reduction_allowance(self, plan, point, *, observation, width, run):
        """Admit the summed pair work and a point's workspace, and return the remaining allowance.

        The summed registered work of all the Plan's pair reductions, computed
        once per Plan object and cached, is admitted against
        ``max_classical_products`` first: a Plan whose sum exceeds the cap is
        refused here, before the point's preparation charge and native work. The workspace gate then refuses a
        point whose ``pair_bytes`` exceeds ``max_bytes``. The allowance
        returned for one point is the cap less the registered work of every
        other pair reduction of the Plan. Each preparation is admitted
        separately against this value, and the cumulative limit holds because
        the whole population is admitted, when each pair is acquired once. A
        repeated acquisition of a pair registers its work again, which this
        Plan-level ledger does not count. The ledger is read from the Plan's
        cache, never changed, so the hook may be asked several times for one
        point without recomputing the population.
        """
        import json
        from .pair_reducer import pair_bytes, pair_work

        pair_work_total = _pair_work_total(plan)
        if pair_work_total > self.max_classical_products:
            raise ValueError(
                f"GCIM pair reductions exceed max_classical_products: registered work {pair_work_total}, "
                f"max_classical_products={self.max_classical_products}. "
                f"Raise FixedGCIM.max_classical_products to at least {pair_work_total}."
            )
        own = 0
        for item in observation.positions:
            if item.kind == "reduction":
                parameters = json.loads(item.parameters)
                _check_bytes(pair_bytes(parameters), self.max_bytes, "GCIM pair reduction workspace")
                own += pair_work(parameters)
        return int(max(0, self.max_classical_products - (pair_work_total - own)))

    def _plan_classical(self, problem, output, rng):
        """Declare bounded host work for H0 = V-dagger (A - cI) V and S = V-dagger V.

        ``c = trace(A)/d`` (``_mean_diagonal``) is removed from A before the
        product (``_offset_free_actions``) and added back after the solve. The
        law is ``_classical_projection_requirements``: one block action of
        the distinct state columns (Pauli input) or one operator action per
        distinct column with the extra work of removing c (other input),
        ``2 b**2 d`` scalar products for the two Gram matrices, and any known
        preparation work. States are distinct by object identity, so a
        repeated input shares its column and action. Custom preparation work
        that cannot be counted stays declared as unknown instead of being
        guessed.
        """
        b = len(self.basis)
        if b > self.max_basis_size:
            raise ValueError(f"basis has {b} states, more than FixedGCIM.max_basis_size={self.max_basis_size}, before projected acquisition")
        operator = eigen_operator(problem, output, max_bytes=self.max_bytes, as_pauli=False)
        basis = tuple(
            eigen_state(s, problem, max_bytes=self.max_bytes, pad=False) for s in self.basis
        )
        basis_record = FixedGCIMBasis(preparations=tuple(s.preparation for s in basis))
        if problem.subspace is not None and problem.subspace != basis_record.reference:
            raise ApplicabilityError("requested subspace differs from this fixed normalized basis")
        d = operator.basis.dimension
        unique = tuple({id(s): s for s in basis}.values())
        preparation = tuple(preparation_requirements(s) for s in unique)
        shift = _mean_diagonal(operator, max_bytes=self.max_bytes)
        known_bytes, work = _classical_projection_requirements(
            operator, len(unique), shift, max(size for size, _ in preparation), b)
        work += sum(w for _, w in preparation if w is not None)
        _check_bytes(known_bytes, self.max_bytes, "classical projected matrices")
        _check_projected_solve_work(b, self.max_analysis_work, "projected solve")
        if work > self.max_classical_products:
            raise ValueError("classical projected matrices exceed max_classical_products")
        labels = tuple(
            f"{matrix}_{i}_{j}_{part}"
            for matrix in ("h", "s")
            for i in range(b)
            for j in range(i, b)
            for part in ("real", "imag")
        )
        construction, experiments = host_construction(
            METHOD,
            (operator.reference, *(s.reference for s in basis)),
            labels,
            work=work,
            frames=tuple("physical" if name.startswith("h_") else "unit" for name in labels),
            admission=("FixedGCIM.max_admission_steps", self.max_admission_steps),
            description="classical H0=V†(A-cI)V with c=trace(A)/d and S=V†V on normalized "
            "preparation columns; "
            + (
                "custom preparation work unknown"
                if any(w is None for _, w in preparation)
                else "known preparation work included"
            ),
        )
        rec = FixedGCIMReconstruction(
            terms=PauliArrays.from_rows((), max(1, (d - 1).bit_length())), basis_size=b,
            num_qubits=max(1, (d - 1).bit_length()),
            host_identity_shift=shift,
        )
        from types import SimpleNamespace
        from nwqlib.algorithms._eigen_support import eigen_error_model

        # eigen_error_model reads only the output, problem and construction,
        # so the Plan is validated once with its error model.
        components = SimpleNamespace(output=output, problem=problem, construction=construction)
        plan = Plan(
            problem=problem,
            method=self,
            output=output,
            execution="classical",
            shots=None,
            randomness=rng.snapshot(),
            construction=construction,
            experiments=experiments,
            reconstruction=rec,
            assumptions=("normalized preparation columns; projected estimate only",),
            error_model=eigen_error_model(components, METHOD),
        )
        return plan._bind(
            blocks=self._host_blocks(plan, operator, basis),
            operator=operator,
            basis=basis,
        )

    def _host_blocks(self, plan, operator, basis):
        """Bind the projected-matrix kernel without materializing its trial vectors yet."""
        from nwqlib.blocks.kernels import BoundKernel, KernelOutput
        from nwqlib.execution import ScalarValue

        (kernel,) = plan.construction.kernels
        rec = plan.reconstruction

        def invoke():
            """Apply A - cI once to the block of normalized columns and publish the upper-triangular H0/S scalars."""
            from nwqlib._projected_eigensolver import _hermitian_part

            b = rec.basis_size
            d = operator.basis.dimension
            positions = {}
            for state in basis:
                positions.setdefault(id(state), (len(positions), state))
            # The byte law is _classical_projection_requirements. Its
            # preparation stage is checked by state_direction itself when it
            # runs, as is each plain operator action of the per-column route.
            _check_bytes(
                _classical_projection_requirements(operator, len(positions), rec.identity_shift, 0, b)[0],
                self.max_bytes,
                "classical GCIM projections",
            )
            if kernel.invocation_work > self.max_classical_products:
                raise ValueError("classical GCIM projections exceed max_classical_products")
            # Contiguous columns let np.vdot conjugate inside the reduction,
            # so no conjugate copy of V is formed.
            v = np.empty((d, len(positions)), dtype=np.complex128, order="F")
            for column, state in positions.values():
                v[:, column] = state_direction(state, max_bytes=self.max_bytes)
            w = np.empty_like(v, order="F")
            _offset_free_actions(operator, v, rec.identity_shift, w,
                                 max_bytes=self.max_bytes, max_products=self.max_classical_products,
                                 limit_name="FixedGCIM.max_classical_products")
            index = [positions[id(s)][0] for s in basis]
            # Full H0 and S are formed before their Hermitian parts, so both
            # independently rounded triangles are averaged.
            h = np.empty((b, b), dtype=np.complex128)
            s = np.empty((b, b), dtype=np.complex128)
            for i, left in enumerate(index):
                for j, right in enumerate(index):
                    h[i, j] = np.vdot(v[:, left], w[:, right])
                    s[i, j] = np.vdot(v[:, left], v[:, right])
            del v, w
            h = _hermitian_part(h)
            s = _hermitian_part(s)
            scalars = []
            for label, frame in zip(kernel.scalars, kernel.scalar_frames, strict=True):
                matrix, i, j, part = label.split("_")
                value = (h if matrix == "h" else s)[int(i), int(j)]
                scalars.append(
                    ScalarValue(
                        label=label,
                        value=float(value.real if part == "real" else value.imag),
                        frame=frame,
                    )
                )
            return KernelOutput(
                plan_id=plan.content_id,
                selected_kernel_id=kernel.content_id,
                scalars=tuple(scalars),
                physical_scale=None,
                physical_scale_unavailable="no recovery applies: the pencil uses normalized basis columns "
                "and H is already in the Problem's energy unit",
            )

        return (BoundKernel._bind(plan, kernel, invoke),)

    def analyze(self, plan, data, *, settings):
        """Reconstruct the acquired Hermitian pencil and solve its usable overlap subspace."""
        cutoff = settings.get("overlap_cutoff", self.overlap_cutoff)
        if (
            not isinstance(cutoff, (int, float))
            or not 0 < cutoff < float("inf")
        ):
            raise ValueError("overlap_cutoff must be finite and positive")
        rec = plan.reconstruction
        if plan.execution == "classical":
            chunks = tuple(chunk for chunk, _ in matched_chunks(plan, data))
            if len(chunks) != 1:
                raise ValueError("classical projection requires its one complete H/S acquisition")
            scalars = {item.label: item.value for item in chunks[0].values}
            b = rec.basis_size
            _check_bytes(_projected_solve_bytes(b), self.max_bytes, "GCIM projected analysis")
            _check_projected_solve_work(b, self.max_analysis_work, "GCIM projected analysis")
            # Recover the lower triangle by conjugation from the acquired upper triangle.
            matrices = {}
            for name in ("h", "s"):
                matrix = np.empty((b, b), dtype=complex)
                for i in range(b):
                    for j in range(i, b):
                        value = complex(
                            scalars[f"{name}_{i}_{j}_real"], scalars[f"{name}_{i}_{j}_imag"]
                        )
                        matrix[i, j], matrix[j, i] = value, value.conjugate()
                matrices[name] = matrix
            # The host kernel forms S from inner products of length d, so its
            # eigenvalues can lie below zero by the Gram formation error.
            trace = fsum(float(matrices["s"][i, i].real) for i in range(b))
            pencil = solve_pencil(
                matrices["h"], matrices["s"], cutoff=cutoff, sampled=False,
                identity_shift=rec.identity_shift,
                overlap_input_tolerance=gram_formation_allowance(plan.problem.dimension, trace),
            )
            return FixedGCIMResult(
                plan_id=plan.content_id,
                construction_id=plan.construction.content_id,
                observation_id=data.observations.content_id,
                contribution_ids=(chunks[0].content_id,),
                eigenvalue=pencil.eigenvalues[0] if pencil.eigenvalues else None,
                missing=(),
                estimates=(),
                pencil=pencil,
                analysis_cutoff=cutoff,
                origin=capture_analysis_origin(
                    analyzer=METHOD, method_id=self.content_id, dependencies=("numpy", "scipy")
                ),
                facts=exact_readout_sampling(plan, data.observations, random_draws=False),
            )._attach(plan, data)
        if plan.shots is None and rec.nonidentity_coefficients:
            return self._analyze_pairs(plan, data, cutoff)
        if rec.nonidentity_coefficients:
            return self._analyze_groups(plan, data, cutoff)
        # A scalar identity target has no acquisition, so its value is the
        # identity coefficient and no overlap is calculated.
        tuple(matched_chunks(plan, data))
        return FixedGCIMResult(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=(),
            eigenvalue=rec.identity_shift,
            origin=capture_analysis_origin(
                analyzer=METHOD, method_id=self.content_id, dependencies=("numpy", "scipy")
            ),
            missing=(),
            estimates=(),
            pencil=None,
            analysis_cutoff=cutoff,
            facts=exact_readout_sampling(plan, data.observations, random_draws=False),
        )._attach(plan, data)

    def _analyze_groups(self, plan, data, cutoff):
        """Reduce every group count acquisition, pool the settings and solve the sampled pencil.

        ``read_groups`` admits the pass and reduces each matched chunk,
        ``pooled_statistics`` pools repeated acquisitions and reconstructs
        ``(H0_ij, S_ij)`` with their variances, and a complete pencil is
        solved with the positive-subspace policy, ``identity_shift = c_I``,
        and the processed-operator enclosure as a numerical diagnostic of the
        sampled Ritz estimate. The recorded ``H`` is ``H0 + c_I S``.
        """
        rec = plan.reconstruction
        moments = read_groups(plan, data)
        estimates, missing, pairs, variances = pooled_statistics(rec, moments)
        pencil = None
        if not missing:
            h, s = assemble_pair_pencil(rec.basis_size, pairs)
            pencil = solve_pencil(h, s, cutoff=cutoff, sampled=True, enclosure=processed_pauli_enclosure(rec),
                                  identity_shift=rec.identity_shift)
        return FixedGCIMResult(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=tuple(row.contribution_id for row in moments),
            eigenvalue=pencil.eigenvalues[0] if pencil and pencil.eigenvalues else None,
            missing=missing,
            estimates=estimates,
            pencil=pencil,
            analysis_cutoff=cutoff,
            group_moments=moments,
            sampled_pair_variances=variances,
            origin=capture_analysis_origin(
                analyzer=METHOD, method_id=self.content_id, dependencies=("numpy", "scipy")
            ),
            facts=exact_readout_sampling(plan, data.observations, random_draws=False),
        )._attach(plan, data)

    def _analyze_pairs(self, plan, data, cutoff):
        """Pool each exact pair's reductions, assemble the unit-diagonal pencil and solve it.

        Every chunk is one ``block_matrix_elements`` reduction returning
        ``(H0_ij, S_ij)``. Its entry bounds come from its producing receipt's
        saved-state error (``PreparedArtifact.saved_state_error``) through
        ``pair_reducer.entry_bounds`` with the ordered-action envelope, and
        repeated acquisitions of one pair pool with population weights under
        ``pooled_bound``, separately for H0 and S. The deterministic Gram
        filter admits the maximum row sum of the off-diagonal overlap bounds
        (``_exact_overlap_allowance``). When that allowance is unavailable
        it admits no input error beyond the solver's own roundoff.
        """
        import json
        from .pair_reducer import entry_bounds, pooled_bound

        rec = plan.reconstruction
        b = rec.basis_size
        terms = len(rec.nonidentity_coefficients)
        pairs = {(left, right): [] for left, right in pair_acquisitions(b, terms)}
        receipts = {receipt.content_id: receipt for receipt in data.receipts}
        ids = []
        for chunk, _ in matched_chunks(plan, data):
            (point,) = chunk.observation.positions
            parameters = json.loads(point.parameters)
            pair = (parameters["left"], parameters["right"])
            reduced = [value for value in chunk.values if value.kind == "reduced" and value.component == 0]
            if pair not in pairs or len(reduced) != 1 or len(reduced[0].real) != 2:
                raise ValueError("GCIM needs its pair reduction's two complex scalars")
            (value,) = reduced
            h0 = complex(value.real[0], value.imaginary[0])
            overlap = complex(value.real[1], value.imaginary[1])
            receipt = receipts.get(chunk.prepared_id)
            # The reducer's entry bounds cover the host contraction of the saved
            # state, so only its amplitude-readout exclusion is resolved here.
            # The exact pair entries cancel the common global phase, but the
            # input array can still carry a shared host phase correction, so
            # the entry law uses the saved-state budget.
            delta = None if receipt is None else receipt.saved_state_error(("amplitude-derived masses",))[0]
            bounds = entry_bounds(delta, float.fromhex(parameters["c1"]), 1 << parameters["qubits"], terms,
                                  diagonal=pair[0] == pair[1])
            pairs[pair].append((chunk.content_id, h0, overlap, bounds))
            ids.append(chunk.content_id)
        estimates, values, overlap_bounds = [], {}, {}
        for (left, right), rows in pairs.items():
            if not rows:
                continue
            count = len(rows)
            weights = tuple(1 / count for _ in rows)
            if count == 1:
                h0, overlap = rows[0][1], rows[0][2]
            else:
                h0 = complex(fsum(w * r[1].real for w, r in zip(weights, rows)),
                             fsum(w * r[1].imag for w, r in zip(weights, rows)))
                overlap = complex(fsum(w * r[2].real for w, r in zip(weights, rows)),
                                  fsum(w * r[2].imag for w, r in zip(weights, rows)))
            populations = [1] * count
            h_bound = pooled_bound([r[1] for r in rows], [None if r[3] is None else r[3][1] for r in rows],
                                   populations)
            s_bound = pooled_bound([r[2] for r in rows], [None if r[3] is None else r[3][0] for r in rows],
                                   populations)
            values[(left, right)] = (h0, overlap)
            overlap_bounds[(left, right)] = s_bound
            estimates.append(PairEstimate(
                experiment=f"pair_{left}_{right}", pair=(left, right),
                hamiltonian=Complex128(real=h0.real, imag=h0.imag),
                overlap=Complex128(real=overlap.real, imag=overlap.imag),
                contribution_ids=tuple(r[0] for r in rows), weights=weights,
                hamiltonian_bound=h_bound, overlap_bound=s_bound,
            ))
        missing = tuple(f"pair_{left}_{right}" for (left, right), rows in pairs.items() if not rows)
        pencil = value = None
        if not missing:
            _check_bytes(_projected_solve_bytes(b), self.max_bytes, "GCIM projected analysis")
            _check_projected_solve_work(b, self.max_analysis_work, "GCIM projected analysis")
            h0, s = assemble_pair_pencil(b, values)
            allowance = _exact_overlap_allowance(b, overlap_bounds)
            pencil = solve_pencil(h0, s, cutoff=cutoff, sampled=False, identity_shift=rec.identity_shift,
                                  overlap_input_tolerance=0.0 if allowance is None else allowance)
            value = pencil.eigenvalues[0] if pencil.eigenvalues else None
        return FixedGCIMResult(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=tuple(ids),
            eigenvalue=value,
            origin=capture_analysis_origin(
                analyzer=METHOD, method_id=self.content_id, dependencies=("numpy", "scipy")
            ),
            missing=missing,
            estimates=(),
            pairs=tuple(estimates),
            pencil=pencil,
            analysis_cutoff=cutoff,
            facts=exact_readout_sampling(plan, data.observations, random_draws=False),
        )._attach(plan, data)

    def save_archive(self, plan, files):
        """Save the selected inputs so loading can rebind them without replanning.

        The configured basis and the selected basis are stored separately.
        ``selected_basis_indices`` records which positions shared one state
        object, because classical acquisition materializes a repeated input once
        and a reload must reproduce that binding. Quantum plans also store their
        selected native blocks. See docs/saved_evidence.md.
        """
        from nwqlib.blocks._archive import write_blocks

        selected = plan._native["basis"]
        positions, unique, indices = {}, [], []
        for state in selected:
            key = id(state)
            if key not in positions:
                positions[key] = len(unique)
                unique.append(files.write_state(f"selected_basis_{len(unique)}", state))
            indices.append(positions[key])
        return dict(
            format="fixed_gcim/6",
            plan=plan.to_record(),
            problem=files.write_problem(plan.problem),
            output=files.write_output(plan.output),
            method=self.model_dump(mode="json", exclude_computed_fields=True),
            basis=[files.write_state(f"basis_{i}", state) for i, state in enumerate(self.basis)],
            selected_basis=unique,
            selected_basis_indices=indices,
            operator=files.write_operator("selected_operator", plan._native["operator"]),
            blocks=write_blocks(plan.blocks, files) if plan.execution == "quantum" else None,
        )

    def verify(self, plan, result, *, checks):
        from nwqlib.evidence.energy_shift import EnergyShiftOptions, verify_energy_shift
        from nwqlib.evidence.verification import verify_projected

        if type(checks) is EnergyShiftOptions:
            return verify_energy_shift(result, options=checks)
        return verify_projected(result, options=checks)

    @classmethod
    def load_archive(cls, saved, files):
        """Rebind the saved Plan, operator, basis and blocks without selection or acquisition.

        Loading reads the stored construction instead of calling ``plan``, so a
        reopened result keeps its original selected blocks even if planning
        rules change. A classical plan rebinds its host kernel to the restored
        states, with no new operator action at load time.
        """
        from nwqlib._choice_archive import unsupported_archive_format
        from nwqlib.blocks._archive import read_blocks

        if saved.get("format") != "fixed_gcim/6":
            raise unsupported_archive_format("FixedGCIM archive", saved.get("format"), "fixed_gcim/6")
        fields = dict(saved["method"])
        fields["basis"] = tuple(files.read_state(s) for s in saved["basis"])
        method = cls(**fields)
        plan = files.read_plan(
            saved["plan"],
            problem=files.read_problem(saved["problem"]),
            method=method,
            output=files.read_output(saved["output"]),
            reconstruction=FixedGCIMReconstruction.model_validate(saved["plan"]["reconstruction"]),
        )
        unique = tuple(files.read_state(s) for s in saved["selected_basis"])
        basis = tuple(unique[i] for i in saved["selected_basis_indices"])
        operator = files.read_operator(saved["operator"])
        blocks = (
            read_blocks(saved["blocks"], plan.construction.selections, files)
            if plan.execution == "quantum"
            else method._host_blocks(plan, operator, basis)
            if plan.experiments
            else ()
        )
        return plan._bind(
            blocks=blocks,
            operator=operator,
            basis=basis,
            **({"pair_table": _pair_table(operator)}
               if plan.execution == "quantum" and plan.shots is None and plan.experiments else {}),
        )
