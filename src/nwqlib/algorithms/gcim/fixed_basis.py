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
    transform_block,
)
from nwqlib.blocks.selection import select_pauli_group_basis
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
    Rational,
    Real,
    Record,
    Text,
)
from nwqlib.operators.access import Count, _check_bytes
from nwqlib.operators.inputs import _basis, pauli_table
from nwqlib.problems.inputs import StatePreparationSpec, ingest_vector
from nwqlib.problems.records import StateData
from nwqlib.ir import (
    Allocate,
    BlockCall,
    ClassicalValue,
    Definition,
    Expression,
    ExprRef,
    Measure,
    MeasurementBatch,
    MetadataRef,
    Parameter,
    ParameterRef,
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
from .pencil_sampling import pilot_split
# Importing the pair reducer registers block_matrix_elements, so a Plan or
# Result holding pair reductions validates in any process that loads GCiM.
from . import pair_reducer  # noqa: E402,F401

METHOD = Source(
    name="fixed_gcim",
    version="5",
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


class PencilAcquisition(Record):
    """How `FixedGCIM`, `ADAPT` and `QFD` measure the projected matrices H and S of their trial states.

    Build it with keyword arguments and pass it as `FixedGCIM(acquisition=...)`,
    `ADAPT(acquisition=...)` or `QFD(acquisition=...)`. Every field has a
    default. The choices change how `H_ij = <phi_i|A|phi_j>` and
    `S_ij = <phi_i|phi_j>` are measured, not their ideal values: one basis
    has the same ideal H and S under every supported combination, while the
    circuits, the number of measurement settings and the finite-shot
    covariance differ.

    Write `A = c I + sum_k c_k P_k` with real `c_k` and partition the L
    nonidentity Pauli words into G qubit-wise commuting groups. For a pair
    `i < j` the grouped measurement prepares the joint state
    `chi_ij = (|0>|phi_i> + |1>|phi_j>)/sqrt(2)` of one ancilla and the
    system. With `z = <phi_i|P|phi_j>`, the ancilla Pauli operators give
    `<X_a P> = Re z` and `<Y_a P> = Im z`, with a positive imaginary sign
    (Huggins et al., arXiv:1909.09114v2, Sec. II.A, Eqs. (4)-(10)). One
    setting measures the ancilla in the X basis (H, then a computational
    measurement) or in the Y basis (S^dagger, then H) together with one
    group's system basis (H for X, S^dagger then H for Y, nothing for Z). The
    ancilla sign times each system parity estimates one term, and the first
    group's ancilla sign alone estimates the overlap. A diagonal pair
    prepares `phi_i` alone, because `S_ii = 1` and `H_ii` is real. For b trial
    states the complete pencil has `b**2 G` settings with both parts and
    `b(b+1)G/2` with real parts only. Reversing a pair conjugates its
    entries.

    Attributes:
        matrix_measurement: Default `"grouped"`, the joint-state measurement
            above. `"termwise"` is `ADAPT`'s comparison construction: one
            Hadamard test per pair, Pauli term and part, `b**2 L + b(b-1)`
            settings with both parts and `bL + b(b-1)(L+1)/2` with real parts
            only. It requires `pair_preparation="controlled"`. `FixedGCIM`
            and `QFD` measure grouped joint states only. Only measurement
            with shots uses this field.
        pair_preparation: Default `"prefix"`. When the ordered preparations
            of a pair begin with exactly the same blocks (the same reference
            preparation, generators, binary64 angles, qubits and phases),
            those blocks run once without control and only the remaining
            blocks are controlled on the ancilla, which gives the same joint
            state. Otherwise, or with `"controlled"`, both complete
            preparations are controlled. `FixedGCIM` treats each supplied
            state as one block, so only two identical preparations share a
            prefix. `QFD` uses the powers of its one step V instead: a pair
            runs the reference and `V**b` once without control, for the
            power b of smaller absolute value (of the smaller index on a
            tie), and controls `V**(q - b)` for the other power q. Because
            powers of V commute, this prepares the same joint state as
            controlling both preparations.
            `"vacuum_reference"` and `"vacuum_restore"` are for `ADAPT`,
            whose states are `phi_i = Q_i B|vac>` for the reference
            preparation B and a generator chain `Q_i` that maps the
            all-zero state `|vac>` to `exp(i gamma_i)|vac>` with a known
            vacuum phase `gamma_i` (Yoshioka et al., arXiv:2407.14431v3,
            Fig. 1b and pp. 2-3). In the pairs they construct, both
            control B instead of the generators and run every chain
            completely, without a shared prefix. For the pair of the
            reference and `phi_j`, `"vacuum_reference"` applies the
            Hadamard gate to the ancilla, B controlled on ancilla value 1,
            `Q_j` without control, B controlled on ancilla value 0 and the
            phase gate `diag(1, exp(i gamma_j))` on the ancilla. Every
            other pair uses the `"prefix"` construction. `"vacuum_restore"`
            orders the two chains of each pair lexicographically by their
            `(pool index, angle)` items, as Python compares tuples, `Q_L`
            first and `Q_R` second, which can reverse their basis order. It
            applies the Hadamard gate, B controlled on 1, `Q_R`,
            the adjoint of `Q_L`, B controlled on 0, `Q_L` and
            `diag(1, exp(i(gamma_R - gamma_L)))`, all without control except
            B, so it runs `m_R + 2 m_L` generator exponentials for chains of
            `m_L` and `m_R` generators. The zero branch stays at the vacuum,
            with the phase `exp(i(gamma_R - gamma_L))`, until B prepares the
            reference on it and `Q_L` makes it `phi_L`, while the other
            branch holds the adjoint of `Q_L` applied to `phi_R` until the
            final `Q_L` restores `phi_R`. This adjoint reverses the circuit
            already built for `Q_L` and inverts each of its gates. When the
            tuple order reverses the basis order, the pair's entries of H
            and S are conjugated back to basis order. Both give the same
            joint state as `"controlled"`, with the phase correction in the
            circuit for real-only and complex quadratures alike, so H and S, the
            settings and the count reduction are unchanged. `ADAPT` planning
            accepts them when the Pauli terms of every pool generator show
            that its exponential maps `|vac>` to itself up to a phase
            (`ADAPT`'s `acquisition`), and raises `ApplicabilityError`
            otherwise. `FixedGCIM` and `QFD` raise `ApplicabilityError` for
            both values in quantum execution, exact or with shots, unless
            the Plan measures nothing, because a `FixedGCIM` state is one
            supplied preparation and the step V of `QFD` has no known
            vacuum phase. Quantum execution, with or without shots, uses
            this field, and the construction of every pair whose entries
            come from a joint state is recorded. Termwise measurement with
            shots prepares no joint state, so it records none.
        quadratures: Default `"auto"`. `"complex"` measures the real and
            imaginary part of every off-diagonal pair. `"real"` measures real
            parts only and records the imaginary parts as exact zeros, which
            requires a proof that A and every state the Method can prepare are
            real in the computational basis, so that `H` and `S` are real.
            The proof reads the stored inputs and needs every nonzero Pauli
            coefficient of A to be real with an even number of Y letters in
            its word, and every trial state to be an occupation state, a
            product state whose stored amplitudes are all real or a vector
            whose stored amplitudes are all real. `ADAPT` needs its reference
            to qualify and every generator of its pool to have purely
            imaginary Pauli coefficients on words with an odd number of Y
            letters, so that each real-angle exponential is a real matrix. A
            supplied circuit never qualifies. For `ADAPT`, A is the
            Hamiltonian after its `pauli_coefficient_cutoff`, so the proof
            says nothing about the imaginary parts of matrix elements of the
            operator before that cutoff. No proof covers the time-evolved
            states of `QFD`, which are complex in general even for a real A,
            so `QFD` rejects `"real"`. `"auto"` chooses `"real"` exactly
            when the proof holds and `"complex"` otherwise. Planning decides
            once, and resuming or reanalyzing a run keeps that choice. Only
            measurement with shots uses this field. Exact and classical
            execution, and a Plan that measures nothing, leave the requested
            value without effect, so H and S keep both parts. `FixedGCIM`,
            `QFD` and the default growth of `ADAPT` measure nothing when A is
            a multiple of the identity, while `ADAPT` with bounded Ritz
            growth still measures overlaps for such an A whenever its basis
            can hold more than the reference.
            Real-only measurement does not show that an unmeasured imaginary
            part is zero on a noisy device.

    Raises:
        ValueError: If `matrix_measurement="termwise"` is combined with any
            `pair_preparation` other than `"controlled"`.
    """

    matrix_measurement: Literal["grouped", "termwise"] = "grouped"
    pair_preparation: Literal["prefix", "controlled", "vacuum_reference", "vacuum_restore"] = "prefix"
    quadratures: Literal["auto", "complex", "real"] = "auto"

    @model_validator(mode="after")
    def _termwise_controlled(self):
        """The termwise Hadamard test controls each complete preparation, so it has no prefix form."""
        if self.matrix_measurement == "termwise" and self.pair_preparation != "controlled":
            raise ValueError("matrix_measurement='termwise' controls each complete preparation; "
                             "use pair_preparation='controlled'")
        return self


class PencilSampling(Record):
    """A shot budget that `FixedGCIM`, `QFD` or `ADAPT` splits between a uniform pilot stage and a main stage chosen from it.

    Build it with keyword arguments, for example
    `PencilSampling(total_shots=12000, pilot_fraction=0.2)`, and pass it as
    `FixedGCIM(sampling=...)`, `QFD(sampling=...)` or `ADAPT(sampling=...)`.
    Both arguments are required, and every method needs quantum execution
    for it. For `FixedGCIM` and `QFD` it replaces `shots=`, which must then
    be omitted. The basis, the grouped settings, their circuits and the
    projected pencil stay those of `shots=`. Only the shots of each setting
    change. [`ADAPT`][nwqlib.algorithms.gcim.adapt.ADAPT] keeps a positive
    `shots=` for the queries that grow its basis and measures the final
    basis again in the two stages with this separate budget. A quantum
    ADAPT Plan that measures nothing also accepts `shots=None` or termwise
    measurement and runs no stage.

    For a complete pencil with m > 0 settings and B = `total_shots`, the
    pilot stage takes `B_p = min(B - 2m, max(2m, floor(f B)))` shots, with
    the exact rational value of the binary64 number f = `pilot_fraction`, and
    the main stage the other `B - B_p`, so each stage can give every setting
    at least two shots. The pilot gives every setting `floor(B_p/m)` shots
    and the first `B_p mod m` settings one more. Its pencil is solved at
    `overlap_cutoff`, and one decision then fixes the main-stage shots N_s of
    each setting s from v_s, the pilot estimate of the per-shot variance of
    the first-order change that one shot of s makes in the lowest Ritz
    value. Over real-valued N_s of at least 2 that sum to `B - B_p`, the
    proportional rule with that floor, `N_s = max(2, sqrt(v_s/lambda))` with
    lambda > 0 set by the sum, minimizes the local variance
    `sum_s v_s/N_s` that a linear model at the pilot solution predicts. The
    integer N_s round it deterministically. Settings at the floor get 2,
    the others their real-valued N_s rounded down, and the remaining shots
    go one each to the settings above the floor where one more shot lowers
    the local variance the most, ties in setting order. They are at least
    2, spend `B - B_p` exactly and need not be the integer minimum of the
    local variance. The local variance is no confidence bound. It
    leaves out the noise of the pilot estimates, a change of kept rank at
    the cutoff, near-degenerate Ritz values and device bias.
    When the pilot solve fails, a pilot Ritz value or coefficient is not
    finite, the two lowest offset-free Ritz values of the pilot solve
    (`ProjectedPencil.offset_free_eigenvalues`) are equal, a v_s is unusable
    or every v_s is zero, every setting gets the uniform split of `B - B_p`
    instead. The allocation counts every shot at the same cost, whatever its
    circuit. The main stage measures every setting afresh, and only its
    measurements form the final H and S. The pilot measurements stay in the
    Run, and their shots count toward its totals like the main ones. The
    result's `shot_allocation`
    ([`ShotAllocation`][nwqlib.algorithms.gcim.fixed_basis.ShotAllocation])
    records the decision, and the
    [GCiM guide](../../algorithms/gcim.md#pilot-and-main-shot-allocation)
    describes the two stages.

    For `FixedGCIM` and `QFD`, planning refuses a budget below four shots
    per setting and gives m in its message. It also checks the grouped
    analysis of one stage against `max_analysis_work` and `max_bytes`
    before any shot, using the larger stage budget `max(B_p, B - B_p)` for
    either stage because the main shots are not known at planning, so it
    can refuse a budget whose actual allocation would fit. A refusal names
    the limit to raise or the parameters to reduce. `ADAPT` knows its final
    basis only after its growth, so its planning makes both checks before
    the first query on the largest basis the Plan can reach. With m* grouped
    settings in that basis, `total_shots` must be at least 4m*, and the
    grouped analysis of one stage of `total_shots` shots on it is checked
    against `max_bytes` and `max_products`. The split and the allocation of
    an ADAPT run use the m settings of the basis it ends with, and m <= m*.
    For `FixedGCIM` and `QFD`, because the main stage chooses its shots from
    the pilot, `prepare(plan, settings="all")` is rejected, and
    `prepare(plan)` prepares the next setting.

    Attributes:
        total_shots: Required. Positive integer number of shots of both
            stages and all settings, at least 4m for m settings. `ADAPT`
            requires at least 4m* for the m* settings of the largest basis
            its Plan can reach.
        pilot_fraction: Required, a finite number strictly between 0 and 1.
            Requested share of `total_shots` for the pilot stage, adjusted so
            that each stage keeps at least two shots per setting.
    """

    total_shots: PositiveInt
    pilot_fraction: Annotated[Real, Field(gt=0, lt=1)]


class PairPreparation(Record):
    """The selected construction of the joint state of one off-diagonal pair.

    Every route prepares `(|0>|phi_i> + |1>|phi_j>)/sqrt(2)` with the phases
    of the two preparations kept, for `ADAPT` possibly with `phi_i` and
    `phi_j` exchanged (`pair`). `ResolvedAcquisition.pairs` of a
    `FixedGCIM` or `QFD` Plan holds one per off-diagonal pair, selected at
    planning, and `ADAPTResult.pair_preparations` holds one per off-diagonal
    pair of the value basis when its pencil entries come from joint states.
    A record describes the construction and does not show that
    observations of the pair were obtained.

    Attributes:
        pair: `(i, j)` with `i < j`, positions in the basis of the pencil.
            `ADAPT` orders the two chains of the pair lexicographically by
            their `(pool index, angle)` items, as Python compares tuples,
            which can reverse these positions, and then conjugates the
            pair's entries of H and S back to basis order.
        route: `"prefix"` when the first `shared_blocks` preparation blocks
            of both states run once without control, `"controlled"` when
            both complete preparations are controlled on the ancilla.
            The two `ADAPT` routes control only the reference preparation B
            ([`PencilAcquisition`][nwqlib.algorithms.gcim.fixed_basis.PencilAcquisition]).
            `"vacuum_reference"` is the construction for the pair of the
            reference and one other state, which runs B twice under control
            and that state's generator chain once without control.
            `"vacuum_restore"` runs B twice under control and, without
            control, the chain that comes second in the tuple order that
            `pair` describes, the adjoint of the first chain and the first
            chain, so the first chain runs twice, once as its adjoint. Both
            apply the known vacuum phase on the ancilla.
        shared_blocks: Number of leading blocks that run once without
            control: 1 for two identical `FixedGCIM` preparations, the
            reference plus the common leading generators for `ADAPT`, and
            the reference plus the `abs(b)` uses of V or its adjoint for the
            shared power b of `QFD`. Zero exactly for every route other
            than `"prefix"`.
    """

    pair: tuple[Count, Count]
    route: Literal["prefix", "controlled", "vacuum_reference", "vacuum_restore"]
    shared_blocks: Count = 0

    @model_validator(mode="after")
    def _route(self):
        if self.pair[0] >= self.pair[1] or (self.route == "prefix") != (self.shared_blocks > 0):
            raise ValueError("pair preparation needs i < j and shared blocks exactly for the prefix route")
        return self


class ResolvedAcquisition(Record):
    """The requested and the resolved `PencilAcquisition` of one Plan, fixed when the Plan is made.

    `FixedGCIMReconstruction.acquisition` and `AdaptReconstruction.acquisition`
    hold it, and for `QFD` the `FixedGCIMReconstruction` in
    `QFDReconstruction.pencil`. Running, resuming and reanalyzing the Plan
    read these values and never decide them again. A choice that the Plan's
    execution does not use is None: classical execution and a Plan that
    measures nothing use none, exact readout only `pair_preparation`, and
    measurement with shots all three. A requested `"real"` that the
    execution does not use has no effect, and H and S keep both parts. A
    requested `"vacuum_reference"` or `"vacuum_restore"` that the execution
    does not use is kept in `requested` with `pair_preparation` None. When
    the execution uses it, only an `ADAPT` Plan resolves it, and only when
    every pool generator passes the vacuum rule of `ADAPT`'s `acquisition`,
    a check of the generators' Pauli terms.

    Attributes:
        requested: The Method's `PencilAcquisition`.
        matrix_measurement: The measurement of the matrix elements, or None.
        pair_preparation: The pair construction rule, one of the four
            values of `PencilAcquisition.pair_preparation`, or None.
        quadratures: `"complex"` or `"real"` as measured, or None.
        real_proof: `"proved"` when the stored inputs named in `scope` prove
            that the measured Hamiltonian and every preparable state are
            real, `"unknown"` when they do not, and None when no proof was
            needed because `"complex"` was requested or quadratures are not
            used.
        reason: Why `quadratures` has its value, naming the first input that
            failed the proof, or None.
        scope: The inputs the proof covered, or None. Its Hamiltonian is the
            operator the Plan measures. For `ADAPT` that is the Hamiltonian
            after `pauli_coefficient_cutoff`, whose recorded truncation
            defines it, and a proof of it says nothing about the imaginary
            parts of matrix elements of the operator before that cutoff.
        pairs: The construction of every off-diagonal pair of a quantum
            `FixedGCIM` or `QFD` Plan, selected at planning before any
            observation exists, so a record does not show that observations
            of its pair were obtained. Empty for classical execution, for a
            Plan that measures nothing and for `ADAPT`, whose pairs arise
            during the run and whose joint-state constructions
            `ADAPTResult.pair_preparations` reports.
    """

    requested: PencilAcquisition
    matrix_measurement: Literal["grouped", "termwise"] | None = None
    pair_preparation: Literal["prefix", "controlled", "vacuum_reference", "vacuum_restore"] | None = None
    quadratures: Literal["complex", "real"] | None = None
    real_proof: Literal["proved", "unknown"] | None = None
    reason: Text | None = None
    scope: Text | None = None
    pairs: tuple[PairPreparation, ...] = ()

    @model_validator(mode="after")
    def _real_requires_proof(self):
        """A real-only pencil records unmeasured imaginary parts as exact zeros, which only a proof justifies."""
        if self.quadratures == "real" and self.real_proof != "proved":
            raise ValueError("real quadratures require a structural real proof")
        return self


def real_hamiltonian_reason(rows):
    """Return why the Pauli rows do not prove ``A* = A``, or None when they do.

    ``rows`` are the ``(label, coefficient)`` pairs of the canonical Hermitian
    Pauli table, identity included. Complex conjugation in the computational
    basis gives ``P* = (-1)**n_Y P`` for a Pauli word with ``n_Y`` letters Y,
    so ``A* = sum_k conj(c_k) (-1)**n_Y(k) P_k`` equals A exactly when every
    nonzero coefficient is real and its word has an even number of Y. Only
    exact zeros are skipped, never small coefficients. One pass over the
    stored rows, no matrix.
    """
    for label, coefficient in rows:
        value = complex(coefficient)
        if value == 0:
            continue
        if value.imag != 0:
            return f"Hamiltonian coefficient of {label} is not real"
        if label.count("Y") % 2:
            return f"Hamiltonian word {label} has an odd number of Y"
    return None


def real_state_reason(state):
    """Return why a selected state's stored representation does not prove a real prepared target, or None when it does.

    An occupation state is a computational basis state with phase +1. A
    product state qualifies when every stored local amplitude has an exact
    zero imaginary part, and a vector with the phase-faithful
    ``qiskit.direct`` preparation when every stored physical amplitude has
    one. Dividing a real nonzero vector by its positive norm keeps it real,
    and zero padding keeps it real. The raw stored amplitudes are scanned,
    not the normalized direction, because the division can round a nonzero
    imaginary part to zero: ``(2**1022 + 2**-1022 i, 1)`` normalizes to a
    direction whose first imaginary part underflows. A real dtype needs no
    scan. Otherwise ``np.count_nonzero`` of the imaginary view visits each
    stored amplitude once without a full temporary array. The proof reads
    the stored data, never the representation name, so a state rebuilt from
    its metadata record, which holds none, is unknown in every
    representation. A supplied circuit and a matrix-product-state circuit
    are unknown too, and no circuit is formed or simulated. The proof
    concerns the ideal target. It assigns no error to the floating-point
    synthesis or the device.
    """
    implementation = state.preparation.implementation
    if implementation not in ("qiskit.occupation", "qiskit.product", "qiskit.direct"):
        return f"{implementation} preparation of {state.reference.identity} has no structural real proof"
    physical = state._physical
    if physical is None:
        return f"{state.reference.representation} state {state.reference.identity} holds no stored data"
    if physical.dtype.kind == "c" and np.count_nonzero(physical.imag):
        return f"{state.reference.representation} state {state.reference.identity} has a stored complex amplitude"
    return None


def real_generator_reason(generator):
    """Return why a pool generator's Pauli sum does not give real exponentials, or None when it does.

    ``A = sum_k a_k P_k`` is real when every nonzero ``a_k`` is purely
    imaginary and its word has an odd number of Y, since
    ``(a P)* = conj(a) (-1)**n_Y P = a P``. A real anti-Hermitian A is real
    skew-symmetric, so ``exp(theta A)`` is a real orthogonal matrix for real
    theta, and so is every product of such factors. An
    anti-Hermitian check alone does not prove it, because ``i I`` passes it
    and has a complex exponential. Exact zeros only.
    """
    for label, coefficient in generator.pauli_terms:
        value = complex(coefficient)
        if value == 0:
            continue
        if value.real != 0 or label.count("Y") % 2 == 0:
            return f"pool generator {generator.pool_index} term {label} is not real"
    return None


def _no_vacuum_action():
    """The vacuum-action reason of the FixedGCIM and QFD states, which never have a known vacuum phase."""
    return ("a FixedGCIM state is one supplied preparation and the QFD step V has no known vacuum phase, so "
            "neither has a generator chain after the reference with a known action on the all-zero state")


def resolve_acquisition(requested, *, execution, shots, acquires, real_reason, scope, pairs=(),
                        vacuum_reason=_no_vacuum_action):
    """Resolve a Method's ``PencilAcquisition`` once, at planning, into its ``ResolvedAcquisition``.

    ``acquires`` is False when the Plan measures nothing, as for a scalar
    identity operator. Only measurement with shots that acquires uses
    quadratures. Classical execution, exact readout and a Plan that measures
    nothing keep the request unchanged in ``requested`` with no resolved
    quadratures or proof, and their consumers compute both parts of H and S.
    ``real_reason()`` returns None when the structural real proof of the
    whole preparable family holds and otherwise its first failing input,
    and it runs only when quadratures are used and not requested as
    ``"complex"``. The proof is exact for the target that ``scope`` names,
    the operator the Plan measures. A truncation recorded before the proof,
    such as ADAPT's ``pauli_coefficient_cutoff``, defines that target, so
    the proof and its success ``reason`` say nothing about imaginary entries
    of the operator before truncation. ``pairs`` are the pair constructions
    of a ``FixedGCIM`` or ``QFD`` Plan. Where quadratures are used, an explicit
    ``"real"`` without a proof raises ``ApplicabilityError``, so a Plan never
    claims a real pencil it cannot justify.

    A Plan that prepares joint states, exactly or with shots, uses
    ``pair_preparation``, and there a requested ``"vacuum_reference"`` or
    ``"vacuum_restore"`` calls ``vacuum_reason()``. It returns None when
    every generator chain after the reference preparation maps the vacuum
    to itself with a known phase, and otherwise why not. A reason raises
    ``ApplicabilityError``, before any count or construction depends on the
    rule. The default reason refuses, because a FixedGCIM state and the QFD
    step have no such chain. Their pair rules (``fixed_pair_preparations``,
    ``qfd.construction.macro_uses``) read every value other than
    ``"prefix"`` as controlled pairs, so without this refusal they would
    plan controlled pairs under a vacuum label. ADAPT passes its structural
    eligibility of the admitted pool (``adapt._admit_generator_routes``).
    Classical execution and a Plan that measures nothing do not use the
    rule and keep the request without a check.
    """
    if execution != "quantum" or not acquires:
        return ResolvedAcquisition(requested=requested)
    if requested.pair_preparation in ("vacuum_reference", "vacuum_restore"):
        reason = vacuum_reason()
        if reason is not None:
            raise ApplicabilityError(f"pair_preparation={requested.pair_preparation!r} needs a known vacuum "
                                     f"action of every generator chain after the reference: {reason}")
    if shots is None:
        return ResolvedAcquisition(requested=requested, pair_preparation=requested.pair_preparation, pairs=pairs)
    proof = reason = None
    quadratures = "complex"
    if requested.quadratures != "complex":
        reason = real_reason()
        proof = "unknown" if reason else "proved"
        if reason is None:
            quadratures, reason = "real", "the structural real proof holds for every input in scope"
        elif requested.quadratures == "real":
            raise ApplicabilityError(f"quadratures='real' needs a structural real proof: {reason}")
    return ResolvedAcquisition(
        requested=requested, matrix_measurement=requested.matrix_measurement,
        pair_preparation=requested.pair_preparation, quadratures=quadratures, real_proof=proof,
        reason=reason, scope=scope if proof else None, pairs=pairs)


class PauliTerm(Record):
    """One real Pauli term of a processed operator or of an ADAPT commutator ``[H, A]``.

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

    The identity coefficient ``c`` is not measured. The obtained scalars give
    ``H0`` without it, the recorded ``H`` adds ``c S`` from the obtained
    overlaps, and the solve adds ``c`` to the Ritz values. A QFD Plan holds
    one as ``QFDReconstruction.pencil``.

    Attributes:
        schema_version: Record layout version.
        terms: The processed operator's real Pauli terms as one letter-code
            table and one coefficient array (``PauliArrays``, rows in
            increasing label order), the identity row included. The exact
            pair reductions act with its nonidentity rows. Classical plans
            store an empty table because they read ``H0`` directly.
        basis_size: Number of trial states ``m``.
        num_qubits: System register width.
        host_identity_shift: For a classical plan, the mean diagonal
            ``c = trace(A)/d`` that the classical routine removes from ``A`` before
            it projects, so its ``h`` scalars are ``V^dagger (A - cI) V``. For
            Pauli input this is the identity coefficient. Zero for a quantum
            plan, whose ``c`` is the identity term of ``terms``.
        covariance: Fixed statement of the identity-term rule above.
        groups: For a sampled Plan with nonidentity terms, the recorded
            deterministic first-fit qubit-wise-commuting partition of the
            nonidentity rows (``PauliTerms.group(strategy="qwc")``), fixed
            before the measurements. Group zero supplies the overlap of each
            off-diagonal quadrature. Empty for exact and classical Plans.
        acquisition: The requested and resolved measurement choices
            (``ResolvedAcquisition``), with the construction of every
            off-diagonal pair of a quantum Plan.
    """

    schema_version: Literal[4] = 4
    terms: PauliArrays
    basis_size: PositiveInt
    num_qubits: PositiveInt
    host_identity_shift: Real = 0.0
    covariance: Literal["H_identity=c*S uses the same overlap observations"] = (
        "H_identity=c*S uses the same overlap observations"
    )
    groups: tuple[SampledGroup, ...] = ()
    acquisition: ResolvedAcquisition

    @model_validator(mode="after")
    def _one_identity_source(self):
        """A quantum table carries its own identity term, so only a classical plan sets ``host_identity_shift``."""
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

    def settings(self):
        """Return the planned sampled settings ``(name, i, j, quadrature, group)`` of the complete pencil."""
        return tuple(sampled_settings(pencil_pairs(self.basis_size), len(self.groups),
                                      self.acquisition.quadratures or "complex"))

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
    """Pooled mean of one weighted group of Pauli terms in a sampled `FixedGCIM` or `QFD` run.

    `FixedGCIMResult.estimates` and `QFDResult.estimates` hold one per group
    setting. Each count measurement of the setting measures
    `Y = sum_k c_k Z_k` from shared shots. `value` is the shot-weighted mean
    of the measurements' means and need not lie in [-1, 1].
    `chunk_mean_variances` contains the sample-mean variance `V_Y` of each
    measurement (`GroupMoment.mean_variance`) for N > 1 shots and None for
    N = 1. The group moments in `FixedGCIMResult.group_moments` and
    `QFDResult.group_moments` keep the raw moments of each measurement with
    these variances. Independent measurements pool
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
    means. A one-shot measurement has unavailable variances and covariance
    (None). label_means preserves each term's first moment from the same
    histogram.

    With N returned shots of one fixed setting, the sample-mean variances
    and covariance are the unbiased sample covariance divided again by N,
    for N > 1. With ``w_a = n_a/N`` the weight of the ``n_a`` shots in
    stored bin a and the residuals ``dY_a = Y_a - mean(Y)`` and
    ``dX_a = X_a - mean(X)``, they are evaluated as
    ``V_Y = sum_a (w_a dY_a) dY_a/(N - 1)``,
    ``V_X = sum_a (w_a dX_a) dX_a/(N - 1)`` and
    ``C_XY = sum_a (w_a dY_a) dX_a/(N - 1)``. In exact arithmetic these
    equal ``(mean(Y**2) - mean(Y)**2)/(N - 1)``,
    ``(1 - mean(X)**2)/(N - 1)`` and
    ``(mean(XY) - mean(X) mean(Y))/(N - 1)``. In binary64 every term of
    ``V_Y`` and ``V_X`` is a product of two numbers of the same sign, so
    neither variance is negative, while ``C_XY`` keeps its sign. The
    rounding of the mean can still give a constant outcome a tiny positive
    variance. Every numeric field must be finite, so a reduction whose
    moments overflow raises an error instead of reporting them unavailable.

    `ADAPT` measures each pair of generator chains once in a fixed chain
    order and reuses the counts in later bases. When a basis lists the two
    chains in the opposite order, the reported imaginary moments are those
    of the conjugate entry: `label_means`, `mean` and `overlap_mean` change
    sign, while `second`, `overlap_cross`, the variances and the covariance
    are unchanged.

    Attributes:
        experiment: The setting's experiment name. For `ADAPT` it is `pair`
            for an off-diagonal setting or `diagonal`.
        contribution_id: Content hash of the count data these moments come from.
        pair: ``(i, j)`` with ``i <= j``, positions in the basis of the pencil.
        quadrature: ``real`` (ancilla X) or ``imag`` (ancilla Y).
        group: Index of the qubit-wise commuting group of nonidentity terms,
            into ``FixedGCIMReconstruction.groups`` or
            ``AdaptReconstruction.energy_groups``.
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
    settings give ``E|error|**2 = V_real + V_imag``. When the Plan measures
    real parts only (``ResolvedAcquisition.quadratures == "real"``), the
    imaginary H0 and S entries of an off-diagonal pair are exact zeros of a
    real pencil, not measurements, and their entries here are zero.

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

    Each contribution is one exact reduction (``block_matrix_elements``) of
    the pair's phase-faithful joint state (or of its diagonal system state)
    to two raw complex scalars, the offset-free Hamiltonian entry and the
    overlap entry in the requested ``(left, right)`` orientation, and counts
    once in the pooling. A diagonal pair's overlap is its raw squared norm,
    kept as evidence while assembly uses unit diagonal overlap. The
    individual Pauli-term transitions are not recorded and cannot be
    recovered from the weighted sum, so a different Hamiltonian truncation
    needs new measurements.

    Attributes:
        experiment: The pair's experiment name.
        pair: ``(left, right)`` with ``left <= right``.
        hamiltonian: Pooled offset-free ``H0_ij``.
        overlap: Pooled ``S_ij``.
        contribution_ids: Content hashes of the contributing exact reductions.
        weights: Share of each contribution in the pooled `hamiltonian` and
            `overlap`.
        hamiltonian_bound: Outward error allowance for the reported offset-free
            Hamiltonian entry ``H0_ij`` under the state-error model of the
            preparation records. It combines that model with bounds for applying
            H0 to the right state, taking its complex inner product with the left
            state, and forming the weighted average of repeated computations of
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
    hold the obtained upper-triangle entries, completed by complex
    conjugation, except that quantum execution sets every diagonal overlap
    entry to one under the normalized-basis convention. Classical FixedGCIM
    computes both triangles and averages them before recording the upper
    triangle. No diagonal jitter or positive-semidefinite repair is applied.
    The solve
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
        offset_free_eigenvalues: The lowest one or two Ritz values that the
            solve computed from ``(H0, S)``, before ``c_I`` is added, in
            ascending order, one at `kept_rank` 1 and two at a larger rank.
            The first one or two `eigenvalues` are these values plus
            ``c_I``, rounded once each. They are therefore equal for
            ``c_I = 0``, and two Ritz values that round to the same physical
            value stay distinct here. Empty when the solve failed. These are
            the energies that the pilot allocation of `PencilSampling` uses
            for its per-setting variance estimates v_s
            (`ShotAllocation.variances`) and its exact-tie test, and
            `ShotAllocation.gap` is their rounded difference `e1 - e0`.
        coordinate_vectors: Rows index the m basis states and columns the Ritz
            pairs. Each column is an S-normalized coefficient vector.
        kept_rank: Number of overlap directions above ``overlap_cutoff``.
        overlap_cutoff: Absolute overlap-eigenvalue threshold used by this solve.
        overlap_filter: `"deterministic_gram"` rejects a negative overlap mode
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
        failure_reason: Code of the solver's rejection, or None when the solve completed.
        sampled_enclosure: For sampled quantum data, the outward-rounded
            interval `[c - sum_k |c_k|, c + sum_k |c_k|]` that contains the
            spectrum of the processed operator `c I + sum_k c_k P_k`, or None.
        sampled_enclosure_violation: Distance of the lowest Ritz value outside that
            enclosure, zero inside it, or None when unavailable.
        sampled_enclosure_passed: Whether the violation lies within
            `sampled_enclosure_rtol` times the largest enclosure endpoint
            magnitude.
        sampled_enclosure_rtol: Relative window used by that comparison.
        sampled_failure: Solver rejection code or `"ritz_outside_operator_enclosure"`.
        processing: Fixed description of the processing applied to the raw pencil.
    """

    hamiltonian: tuple[tuple[Complex128, ...], ...]
    overlap: tuple[tuple[Complex128, ...], ...]
    overlap_eigenvalues: tuple[Real, ...]
    eigenvalues: tuple[Real, ...]
    offset_free_eigenvalues: tuple[Real, ...] = ()
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

    @model_validator(mode="after")
    def _offset_free_length(self):
        if len(self.offset_free_eigenvalues) != min(2, len(self.eigenvalues)):
            raise ValueError("offset_free_eigenvalues must hold the lowest one or two Ritz values of the solve")
        return self

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


class ShotAllocation(Record):
    """The pilot decision of a `PencilSampling` Run: the two stage budgets, the shots of every setting and the pilot estimates.

    `FixedGCIMResult.shot_allocation`, `QFDResult.shot_allocation` and
    `ADAPTResult.shot_allocation` hold it. It is decided once, after the
    last pilot measurement and before the first main-stage measurement, and
    saved with the Run. Resuming, loading and `result.analyze(...)` then
    read it as decided and never solve the pilot again or choose other
    shots, so a new `overlap_cutoff` changes the main pencil only. An
    allocation interrupted before its decision was saved is repeated from
    the pilot measurements after `run.resume(reanalyze=True)`, without
    measuring again. `pilot_counts`, `main_counts` and `variances` are in
    the setting order of the pencil. For `FixedGCIM` and `QFD` it is the
    order of the Plan's experiments. For `ADAPT` it is the order of the
    grouped settings of the final basis, by pair, quadrature and group, in
    which `ADAPTResult.group_moments` lists the records of an analyzed main
    stage. The fields below are read-only.

    Attributes:
        sampling: The requested `PencilSampling`.
        pilot_shots: Shots of the pilot stage, B_p of `PencilSampling`, or 0
            when no setting is measured.
        main_shots: Shots of the main stage, `total_shots - B_p` when
            settings are measured, or 0 when none is.
        effective_fraction: `B_p/total_shots` as an exact ratio in lowest
            terms (a `Rational` record), or `"not_applicable"` when the
            pencil has no setting to measure, so that both stages have zero
            shots and `pilot_counts`, `main_counts` and `variances` are
            empty. For `FixedGCIM` and `QFD` this happens when the operator
            is a multiple of the identity. For `ADAPT` it depends on the
            final basis rather than on the operator, because bounded growth
            on a multiple of the identity measures overlaps once its basis
            has two or more states. The shots that the ADAPT growth already
            spent stay counted when both stages have zero shots.
        pilot_counts: Shots of each setting in the pilot stage.
        main_counts: Shots of each setting in the main stage, each at least
            2, summing to `main_shots`.
        variances: For each setting, the pilot estimate v_s of the per-shot
            variance of the first-order change that one shot of the setting
            makes in the lowest Ritz value, in squared energy units, or None
            where the pilot moments give no estimate: a setting with one
            pilot shot, a negative variance estimate or an overflow. A
            negative v_s from rounding is kept as computed and makes the
            main stage uniform.
            Empty when the pilot solve failed or no setting is measured.
        fallback_reason: Why the main stage takes the uniform split, or None
            when its shots follow `variances`: `"pilot_solve_failed"` (the
            pilot pencil's `failure_reason` gives the solver's code),
            `"pilot_solution_nonfinite"`, `"pilot_lowest_ritz_degenerate"`
            (the two lowest offset-free Ritz values are equal),
            `"pilot_variance_unusable"` or `"pilot_variances_all_zero"`. An
            all-zero estimate does not mean zero uncertainty.
        local_variance: `sum_s v_s/N_s` over the main-stage shots N_s, the
            variance of the lowest Ritz value that a linear model at the
            pilot solution predicts for the main stage, or None when a v_s
            is missing or unusable. It is not a confidence bound. It leaves
            out the noise of the pilot estimates, a change of kept rank at
            the cutoff, near-degenerate Ritz values and device bias.
        gap: The rounded difference `e1 - e0` of the two lowest offset-free
            Ritz values of the pilot solve
            (`ProjectedPencil.offset_free_eigenvalues`), or None at kept rank
            1, after a failed solve or without a pilot. No tolerance is
            applied to it, and a nonzero gap does not show that the lowest
            value is simple.
        gap_frame: Fixed statement that `gap` belongs to the offset-free
            values, not to the physical `eigenvalues`.
        pilot: The pilot [`ProjectedPencil`][nwqlib.algorithms.gcim.fixed_basis.ProjectedPencil],
            solved at the Method's `overlap_cutoff`, whose offset-free
            values, coefficients and kept rank the decision used, with the
            physical Ritz values and enclosure diagnostic of its own solve.
            None when no setting is measured.
    """

    sampling: PencilSampling
    pilot_shots: Count
    main_shots: Count
    effective_fraction: Rational | Literal["not_applicable"]
    pilot_counts: tuple[PositiveInt, ...]
    main_counts: tuple[PositiveInt, ...]
    variances: tuple[Real | None, ...]
    fallback_reason: Text | None
    local_variance: Nonnegative | None
    gap: Nonnegative | None
    gap_frame: Literal["rounded difference of the two lowest offset-free Ritz values of the pilot solve"] = (
        "rounded difference of the two lowest offset-free Ritz values of the pilot solve")
    pilot: ProjectedPencil | None

    def _summary(self, pencil):
        """Return the line that `print(result)` shows for this decision beside the main ``pencil``."""
        if self.pilot is None:
            return ("Shot allocation: not applicable, because the operator is a multiple of the identity and needs "
                    "no shots")
        rule = "pilot allocation" if self.fallback_reason is None else f"uniform main stage ({self.fallback_reason})"
        main = "no" if pencil is None else pencil.kept_rank
        return (f"Shot allocation: {rule}; {self.pilot_shots} pilot and {self.main_shots} main shots, kept rank "
                f"{self.pilot.kept_rank} in the pilot and {main} in the main pencil. The local variance comes from a "
                "linear model at the pilot solution and is not a confidence bound")


def _check_shot_allocation(allocation, missing, plan, label):
    """Require a Result's ``ShotAllocation`` exactly on a ``PencilSampling`` Plan, once main-stage data exist.

    The decision must name the Plan's ``PencilSampling`` and one main count
    per experiment. Without it the Result predates the decision, so it holds
    no main-stage data and lists every experiment in ``missing``.
    """
    sampling = plan.method.sampling
    if allocation is not None and (sampling is None or allocation.sampling != sampling
                                   or len(allocation.main_counts) != len(plan.experiments)):
        raise ValueError(f"{label} shot allocation belongs to another Plan's PencilSampling")
    if sampling is not None and allocation is None and set(missing) != {e.name for e in plan.experiments}:
        raise ValueError(f"{label} main-stage data need the shot allocation decided after the pilot stage")


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
    """Return the Gram-error allowance of an overlap obtained by exact pair reductions, or None.

    ``pair_bounds[(i, j)]`` is the outward entry bound ``E^S_ij`` of each
    obtained off-diagonal overlap (``pair_reducer.entry_bounds``, pooled by
    ``pooled_bound`` over repeated computations), from the preparation
    record of each pair. With solved diagonal S fixed to one, ``E^S_ii = 0``.
    Then

        ||S_hat - S||_2 <= max_i sum_(j != i) E^S_ij <= (B - 1) max_(i<j) E^S_ij,

    where B is the full basis size. The bound needs no statistical
    independence assumption or union factor. It covers only the error of
    obtaining and assembling the overlap, leaving trial-space, truncation,
    eigensolve and ground-identification sources separate. An unresolved
    state-error bound in the preparation record of any off-diagonal pair
    makes the allowance unavailable (None), not zero.
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

    A Run creates its shifted canonical CSR/CSC matrix once, shares its
    original pattern only when every diagonal is present, and counts that
    matrix's bytes for all later actions. Setup work is counted once and each
    action visits the shifted pattern, including inserted diagonals. With h
    missing diagonals the extra work is ``W_setup + count*h`` on the first
    computation and ``count*h`` on a cache hit, relative to the
    ``count*(z + d)`` of the plain actions. The bytes are
    ``_shifted_sparse_requirements``. This count includes the setup, so it
    overestimates the cost for a caller that reuses a Run's shifted operator.
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
    stored order with their coefficients, and no label string is formed.
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
    diagonals, the output size is checked against ``max_bytes``, then a
    second pass fills it. With z stored entries, d lines and index itemsize I
    the persistent storage is ``B_shifted = 16z`` when h = 0 and the
    immutable indices and pointers are shared, and
    ``(16 + I)(z + h) + I(d + 1)`` when h > 0. The scan is array-vectorized,
    so its line-index and mask arrays, ``(I + 1)z`` bytes, and for h > 0 the
    union's concatenated values, indices, lines and sort order join the
    setup scratch (``_shifted_sparse_requirements``).
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
    Bytes are the persistent storage ``B_shifted`` plus the setup scratch of
    the vectorized scan and, for h > 0, of the union. One sufficient count of
    element visits for the two-pass scan is ``W_setup = 4z + 6d + 3h``.
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
    W=(A−cI)V in two preallocated complex arrays. Together they hold 32bN
    bytes for b columns of length N. Conjugating inner products form H₀=V†W
    and S=V†V without an additional state block. The byte count of the
    projection adds the b-by-b matrices and the preparation, operator and
    action workspace. The stored physical Hamiltonian is H=H₀+cS.

    With ``k = columns`` distinct states of length N and ``m = basis_size``,
    the stages are sequential and the peak is their maximum,
    ``max{16kN + B_preparation, B_action stage, 32kN + 512m² + B_dot}``
    (``B_Gram = 32kN + 512m² + B_dot``, with ``B_dot = 0`` for
    ``np.vdot`` of contiguous columns, which conjugates inside the
    reduction). ``B_preparation`` is the largest known preparation of one
    column. The action stage of Pauli input is the shared block-action
    formula ``pauli_action_requirements`` at k columns with an input known
    to be complex (its input and output are V and W) and the count-only flip
    bound ``g = min(L, N)``, plus ``64L`` bytes for the routine's packed
    nonidentity table (``_nonidentity_table``, the masked row copies and
    their read-only copy, 32 bytes per row each). Other input holds V
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

    ``identity`` is the content hash of the processed Pauli operator
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


def fixed_pair_preparations(basis, rule):
    """Return the ``PairPreparation`` of every off-diagonal pair of a FixedGCIM basis under ``rule``.

    Each supplied state is one opaque preparation block, so under
    ``"prefix"`` a pair shares a prefix only when its two preparations are
    the same block: equal preparation records and, for a supplied circuit,
    the same stored circuit. A supplied circuit's preparation record names a
    declared identifier, which does not show that two circuits apply the
    same gates and global phase. Equal records of every other preparation
    name equal stored data, from which the circuit is built
    deterministically. No algebraic equivalence is searched.
    """
    pairs = []
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            left, right = basis[i], basis[j]
            same = (rule == "prefix" and left.preparation == right.preparation
                    and (left.preparation.implementation != "qiskit.supplied" or left._native is right._native))
            pairs.append(PairPreparation(pair=(i, j), route="prefix" if same else "controlled",
                                         shared_blocks=int(same)))
    return tuple(pairs)


def _fixed_pair_recipes(basis, pairs):
    """Return the system preparation blocks of a FixedGCIM basis and the ``_joint_state_blocks`` recipe of every pair.

    State i is the one block ``basis_i``. ``pairs`` are the off-diagonal
    ``PairPreparation`` records (``fixed_pair_preparations``). A prefix pair
    joins two identical preparations, so their one block is the common
    prefix and both controlled remainders are empty. A controlled pair has
    an empty prefix, block i on the ancilla's zero branch and block j on its
    one branch. A diagonal pair runs its block alone.
    """
    prepared = [select_preparation(f"basis_{i}", state) for i, state in enumerate(basis)]
    recipes = {(i, i): ((i,), (), ()) for i in range(len(basis))}
    for item in pairs:
        i, j = item.pair
        recipes[i, j] = ((i,), (), ()) if item.route == "prefix" else ((), (i,), (j,))
    return prepared, recipes


def _joint_state_blocks(prepared, recipes, controlled=None):
    """Return the blocks, their definitions and each pair's preparation children, built from per-pair recipes.

    The block-IR producer of the joint states of FixedGCIM and QFD, exact
    and sampled. ``prepared`` lists the uncontrolled system blocks, and
    ``recipes[i, j]`` for ``i <= j`` is ``(common, left, right)``: positions
    in ``prepared`` of the blocks that run once without control, of those
    controlled on the ancilla's zero branch and of those controlled on its
    one branch, each list in execution order. With C, L and R the products
    of the three lists, an off-diagonal pair runs C on the system, H on the
    ancilla, then, when L is nonempty, X, the controlled blocks of L and X
    again, and then the controlled blocks of R. This prepares
    ``chi_ij = (|0> L C|0> + |1> R C|0>)/sqrt(2)``, so ``phi_i = L C|0>`` and
    ``phi_j = R C|0>``. The global phase of C multiplies both branches and
    cancels in every ``<phi_i|P|phi_j>``. The controlled form of block k is
    ``controlled[k]`` when the caller supplies it, a block with a first
    one-qubit port ``control`` and the system port, and otherwise the
    selected composite definition ``forward_k`` of block k. Either way the
    controlled block's global phase becomes the relative phase of its
    branch. The recipe generator owns the phases: a common prefix is exact
    only when both complete preparations begin, in execution order, with
    the same blocks, phases included. A diagonal pair ``(i, i)`` runs its
    common blocks alone, with no ancilla gate. The ancilla blocks are the
    selected one-qubit preparations of ``|+>`` and ``|1>``, whose direct
    constructors are exactly H and X. Returns ``(blocks, definitions,
    children)`` with ``children[i, j]`` the definition names in order. The
    ancilla is qubit 0 of each circuit.
    """
    h = select_preparation("ancilla_h", ingest_vector(np.array([2**-0.5, 2**-0.5], dtype=complex)))
    x = select_preparation("ancilla_x", ingest_vector(np.array([0.0, 1.0], dtype=complex)))
    supplied = controlled or {}
    forward = {k: supplied[k] if k in supplied else transform_block(f"forward_{k}", prepared[k], control=True)
               for k in sorted({k for _, left, right in recipes.values() for k in (*left, *right)})}
    names = {k: block.record.signature.name for k, block in forward.items()}
    blocks = [h, x, *forward.values(), *prepared]
    definitions = [
        Definition(id=f"{verb}_{wire}", node=cls(wire=wire))
        for verb, cls in (("allocate", Allocate), ("release", Release))
        for wire in ("ancilla", "system")
    ]
    for block in blocks:
        name = block.record.signature.name
        definitions.append(Definition(id=name, node=BlockCall(signature=name, ports=tuple(
            PortMap(port=p.name, wire="ancilla" if p.name == "control" or block in (h, x) else "system")
            for p in block.record.signature.quantum))))
    children = {}
    for (i, j), (common, left, right) in recipes.items():
        children[i, j] = [prepared[k].record.signature.name for k in common]
        if i != j:
            children[i, j].append("ancilla_h")
            if left:
                children[i, j] += ["ancilla_x", *(names[k] for k in left), "ancilla_x"]
            children[i, j] += [names[k] for k in right]
    return blocks, definitions, children


def _exact_pair_construction(prepared, recipes, b, operator, q, metadata, controlled=None):
    """Return the blocks, definitions and experiments of the exact per-pair computation.

    Each off-diagonal pair ``(i, j)``, ``i < j``, of the b basis states
    prepares its phase-faithful joint state from its recipe
    (``_joint_state_blocks`` with ``prepared``, ``recipes`` and
    ``controlled``), so the ancilla's zero branch holds ``phi_i`` and its
    one branch ``phi_j``. Each diagonal pair prepares its system state once.
    Every experiment is one trajectory whose end point is the registered
    ``block_matrix_elements`` reduction of that pair, returning
    ``(H0_ij, S_ij)``, and its setting carries ``metadata``. Whether the
    preparation record supplies a state-error bound depends on the
    operations and parameters of the resulting Qiskit circuit.
    """
    from nwqlib.core.planning import ObservationPoint
    from .pair_reducer import REDUCER, pair_parameters

    table = _nonidentity_table(operator)
    terms = len(table)
    flips = min(terms, 1 << q)
    c1 = _coefficient_mass(table.coefficients.tolist()).hex()
    blocks, definitions, children = _joint_state_blocks(prepared, recipes, controlled)
    experiments = []
    for left, right in pair_acquisitions(b, terms):
        name = f"pair_{left}_{right}"
        sequence = ["allocate_ancilla", "allocate_system", *children[left, right], "release_ancilla",
                    "release_system"]
        body, batch = f"body_{name}", f"batch_{name}"
        definitions.append(Definition(id=body, node=Sequence(children=tuple(sequence))))
        definitions.append(Definition(id=batch, node=MeasurementBatch(
            body=body, settings=(Setting(label=name, metadata=metadata),), repetitions=1,
            observation_kind="trajectory")))
        parameters = pair_parameters(left=left, right=right, qubits=q, identity=operator.reference.identity,
                                     terms=terms, flips=flips, c1=c1)
        point = ObservationPoint(id="pair", kind="reduction", reducer=REDUCER, parameters=parameters)
        experiments.append(Experiment(name=name, batch=batch, setting_index=0,
                                      readout=ReadoutDetails(positions=(point,))))
    return blocks, definitions, experiments


def _check_projected_solve_work(b, cap, what, owner="FixedGCIM"):
    """Reject a ``b``-by-``b`` projected solve whose work ``_projected_solve_work(b)`` exceeds ``cap``.

    The error message names the needed work and the field of the Method
    class ``owner`` that limits it. A sampled Plan also checks its grouped
    analysis against the same field.
    """
    work = _projected_solve_work(b)
    if work > cap:
        raise ValueError(
            f"{what} needs {work} work units for basis size {b}, exceeding max_analysis_work={cap}. "
            f"Raise {owner}.max_analysis_work to at least {work} or reduce the basis size."
        )


def _pencil_admission(method, q, b, terms, nonidentity, shots, quadratures, label="GCIM", sampling=None):
    """Check the known requirements of a b-state pencil before its settings exist and return ``(groups, e)``.

    ``method`` is the planning FixedGCIM or QFD, whose limit fields the
    checks read and whose class name the messages give, ``terms`` the
    recorded ``PauliArrays`` with ``nonidentity`` nonidentity rows on ``q``
    system qubits, ``shots`` the shots per setting or None for the exact
    pair route and ``quadratures`` the resolved quadratures. ``groups`` is
    the recorded QWC partition of a sampled Plan, empty otherwise, and
    ``e`` the number of settings: exact pair reductions or grouped
    measurements. ``label`` starts the messages of the population, byte and
    planned-analysis checks.

    The projected solve and the one-group analysis floor are known
    requirements before the actual grouping and setting construction.
    Planning charges the grouped-pass law one chunk of m = min(s, 2**(n+1))
    bins for each of the K settings of every group on the P = b(b+1)/2
    pairs (copies=K). The planned work of G groups exceeds that of one
    group by K(G - 1)[m(c + 44) + 23], so for L > 0 and 1 <= G <= L one
    group of all L terms bounds it from below before grouping. Only that
    work is used, and the single group need not be a valid partition.
    Grouping, the projected solve and the grouped analysis are checked one
    after another against max_analysis_work, so the limit they need is the
    largest of their works, not their sum.

    Under ``sampling``, the Method's ``PencilSampling``, ``shots`` is the
    two-shot marker of a counts Plan and no stage's tables are known, so
    neither check reads it as a setting's shots. The floor charges each of
    the K settings two bins, the stage minimum, and stays a work floor: for
    every grouping that the budget admits, the stage bound below exceeds it
    by ``(U - 2K)(c + 44) + (4w + 4)(U L_max - 2KL) + 23K(G - 1)``, with no
    oriented view. The m = e settings are split by ``pilot_split``, which
    refuses a budget below 4m, and the planned check charges both stages
    with the stage-budget form of ``grouped_pass_requirements`` at
    ``S = max(B_p, B_main)``, since that bound grows with S.
    """
    owner = type(method).__name__
    solve_need = _projected_solve_work(b)
    if shots and nonidentity:
        per_group, pairs = sampled_setting_count(b, 1, quadratures), b * (b + 1) // 2
        bins = 2 if sampling else min(shots, 1 << (q + 1))
        _, analysis_floor = grouped_pass_requirements(
            q, ((nonidentity, bins),), nonidentity, b, settings=per_group, pairs=pairs, oriented=False,
            copies=per_group)
        if max(solve_need, analysis_floor) > method.max_analysis_work:
            raise ValueError(
                f"{owner} known analysis requirements need at least "
                f"{max(solve_need, analysis_floor)} work units, exceeding "
                f"max_analysis_work={method.max_analysis_work}. "
                f"Increase {owner}.max_analysis_work or reduce the basis or operator."
            )
    # A sampled Plan records its QWC partition before settings expansion.
    # The scalar-identity shortcut acquires no pencil and needs no groups.
    groups = sampled_groups(terms, q, method) if shots and nonidentity else ()
    e = ((sampled_setting_count(b, len(groups), quadratures) if shots else pair_count(b, nonidentity))
         if nonidentity else 0)
    if e > method.max_experiments:
        raise ValueError(
            f"{label} acquisition population {e} exceeds max_experiments={method.max_experiments} "
            "before expansion"
        )
    if sampling:
        stage = max(pilot_split(sampling.total_shots, sampling.pilot_fraction, e))
    # Bytes: the projected-solve allowance of the b-by-b pencil plus an
    # untuned 256 bytes per setting for its name, Setting, body/batch
    # Definitions and Experiment record, and for a sampled Plan the
    # G*n + 8L logical basis and index bytes of its group description.
    # 256E is an untuned record allowance, not a derived Python-heap bound.
    _check_bytes(
        _projected_solve_bytes(b) + 256 * e
        + sum(q + 8 * len(group.indices) for group in groups),
        method.max_bytes,
        f"{label} selected pairs and projected workspace",
    )
    _check_projected_solve_work(b, method.max_analysis_work, "projected solve", owner)
    if e and sampling:
        # Each descriptor holds the common bins H that a table of one stage
        # of `stage` shots can store, and C = K G = M (grouped_pass_requirements).
        analysis_bytes, analysis_work = grouped_pass_requirements(
            q, [(len(group.indices), min(stage, 1 << (q + 1))) for group in groups], nonidentity, b,
            settings=per_group * len(groups), pairs=pairs, oriented=False, copies=per_group, shot_total=stage)
        for field, need, limit in (("max_bytes", analysis_bytes, method.max_bytes),
                                   ("max_analysis_work", max(solve_need, analysis_work), method.max_analysis_work)):
            if need > limit:
                unit = "data bytes" if field == "max_bytes" else "work units"
                raise ValueError(
                    f"{label} planned grouped analysis of one PencilSampling stage of at most {stage} shots needs "
                    f"at most {need} {unit}, exceeding {field}={limit}. This conservative planning bound over "
                    "the stage budget can refuse an allocation that would fit. Before any shot, raise "
                    f"{owner}.{field} to at least {need} or reduce PencilSampling.total_shots, the basis size "
                    "or the grouped observable.")
    elif e and shots:
        # One (L_g, m) descriptor per group, repeated for its K settings.
        # The law increases with every bin count, readout_shape sets m and
        # _decoded_chunk enforces it on backend output, zero-count bins
        # included, and the common c-bit width covers the unused phase bit
        # of a diagonal setting. read_groups checks the observed bins of
        # every later pass.
        analysis_bytes, analysis_work = grouped_pass_requirements(
            q, [(len(group.indices), bins) for group in groups], sum(len(group.indices) for group in groups),
            b, settings=per_group * len(groups), pairs=pairs, oriented=False, copies=per_group)
        _check_bytes(analysis_bytes, method.max_bytes, f"{label} planned grouped analysis")
        known = max(solve_need, analysis_work)
        if known > method.max_analysis_work:
            raise ValueError(
                f"{label} planned grouped analysis needs at most {known} work units "
                f"for one acquisition per setting, the known complete analysis "
                f"requirement, exceeding max_analysis_work={method.max_analysis_work}. "
                f"Raise {owner}.max_analysis_work to at least {known} or reduce "
                "shots, the basis size or the grouped observable."
            )
    return groups, e


def sampled_groups(table, q, method):
    """Return the recorded QWC partition of the nonidentity rows of ``table``.

    The rows are partitioned by the deterministic first-fit grouping
    ``PauliTerms.group(strategy="qwc")``, whose comparison and array counts
    are checked against ``max_bytes`` and ``max_analysis_work`` of
    ``method``, the planning FixedGCIM or QFD, before any setting is
    expanded. Each group's basis carries the single non-I letter of its
    members on every qubit of the union of their supports.
    """
    rows = tuple((p, c) for p, c in table.rows() if p != "I" * q)
    if not rows:
        return ()
    packed = pauli_table(rows, num_qubits=q, max_bytes=method.max_bytes)
    partition = packed.group(strategy="qwc", max_bytes=method.max_bytes,
                             max_comparisons=method.max_analysis_work,
                             limit_name=f"{type(method).__name__}.max_analysis_work")
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


def pencil_pairs(b):
    """Yield the upper-triangle pairs ``(i, j)``, ``i <= j``, of a complete b-state pencil, row by row."""
    for i in range(b):
        for j in range(i, b):
            yield i, j


def sampled_settings(pairs, group_count, quadratures="complex"):
    """Enumerate ``(name, i, j, quadrature, group)`` of the grouped settings of the requested pairs.

    An off-diagonal pair ``i < j`` has one setting per measured quadrature
    and system group, ``max(1, G)`` groups each. Group zero supplies the
    overlap. Under ``quadratures="real"`` an off-diagonal pair measures only
    its real part, because a structurally real pencil has
    ``Im H_ij = Im S_ij = 0`` exactly. A diagonal pair has one
    real setting per group, since ``S_ii = 1`` and ``H_ii`` is real. For M
    pairs with D diagonal ones and Q measured quadratures the count is
    ``Q(M - D) max(1, G) + DG``, which for the complete b-state pencil is
    ``b**2 G`` with Q = 2 and ``b(b+1)G/2`` with Q = 1.
    """
    parts = ("real",) if quadratures == "real" else ("real", "imag")
    for i, j in pairs:
        for quad in ("real",) if i == j else parts:
            for g in range(group_count if i == j else max(1, group_count)):
                yield f"group_{i}_{j}_{quad}_{g}", i, j, quad, g


def sampled_setting_count(b, group_count, quadratures="complex"):
    """Return the number of ``sampled_settings`` of the complete b-state pencil without enumerating them.

    ``bG + Q b(b-1)/2 max(1, G)`` with Q measured off-diagonal quadratures,
    which for ``G >= 1`` is ``b**2 G`` (Q = 2) and ``b(b+1)G/2`` (Q = 1).
    """
    return b * group_count + (1 if quadratures == "real" else 2) * (b * (b - 1) // 2) * max(1, group_count)


def _sampled_construction(prepared, recipes, q, metadata, groups, shots, settings, controlled=None, sampling=None):
    """Return the blocks, definitions, experiments and stage Program fields of the grouped sampled measurements.

    An off-diagonal pair ``(i, j)`` prepares the exact-pair joint state
    ``chi_ij = (|0> phi_i + |1> phi_j)/sqrt(2)`` from its recipe
    (``_joint_state_blocks`` with ``prepared``, ``recipes`` and
    ``controlled``). For ``z_k = <phi_i|P_k|phi_j>``, direct multiplication of the two ancilla
    blocks gives ``<X_a P_k> = Re z_k`` and ``<Y_a P_k> = Im z_k``, with a
    positive Y sign. The X readout applies H and the Y readout S^dagger then
    H to the ancilla. A diagonal pair prepares ``phi_i`` directly and
    measures the system group only. Each group's system readout rotates X
    positions by H, Y positions by S^dagger then H, and leaves I and Z.
    Every readout is the exact primitive basis change of
    ``select_pauli_group_basis`` on the one port ``system`` of its register,
    for the one-letter ancilla label X or Y or the group's basis word (qubit
    0 rightmost). With V = H and V = H S^dagger, ``V^dagger Z V`` is X and Y
    without a scalar phase, so measuring Z after V measures the label.

    The ancilla is qubit 0 and classical bit 0. System qubit k is
    classical bit k + 1, so a displayed key reads ``system_bits phase``. Two
    shared classical declarations of common width ``n + 1`` keep every
    setting's layout equal. A diagonal setting leaves the ancilla classical
    bit at its initial zero and measures the full system register.
    Off-diagonal settings measure both registers. Every measurement starts
    fresh with ``shots`` repetitions, and each setting carries ``metadata``.
    ``settings`` are the planned ``sampled_settings``.

    Under a ``PencilSampling`` ``sampling`` every batch repeats instead
    ``shot_count``, the open Program parameter ``shots`` (at least the
    two-shot stage minimum), and the open parameter ``stage`` (0 pilot, 1
    main) marks the stage of a point, so ``Plan.resolve`` binds both for
    each measurement of each stage and the two stages of one setting are
    distinct points with the same batch, counts readout and repetitions
    expression. The last item returned holds these declarations as Program
    fields, and is empty for fixed ``shots``.
    """
    fields = {}
    if sampling is not None:
        shots = ExprRef(expression="shot_count")
        fields = dict(parameters=(Parameter(name="stage", domain="integer", lower=0, upper=1),
                                  Parameter(name="shots", domain="integer", lower=2, upper=sampling.total_shots)),
                      expressions=(Expression(id="shot_count", value=ParameterRef(parameter="shots")),))
    reads = [select_pauli_group_basis(f"read_{axis}", axis, basis=_basis(2)) for axis in ("X", "Y")]
    tails = [select_pauli_group_basis(f"system_read_{g}", group.basis, basis=_basis(1 << q))
             for g, group in enumerate(groups)]
    blocks, definitions, children = _joint_state_blocks(prepared, recipes, controlled)
    for block in (*reads, *tails):
        name = block.record.signature.name
        definitions.append(Definition(id=name, node=BlockCall(signature=name, ports=tuple(
            PortMap(port=p.name, wire="ancilla" if block in reads else "system")
            for p in block.record.signature.quantum))))
    blocks += [*reads, *tails]
    definitions += [
        Definition(id="measure_phase", node=Measure(wire="ancilla", result="phase")),
        Definition(id="measure_system", node=Measure(wire="system", result="system_bits")),
    ]
    experiments = []
    for name, i, j, quad, g in settings:
        sequence = ["allocate_ancilla", "allocate_system", *children[i, j]]
        if groups:
            sequence.append(f"system_read_{g}")
        if i != j:
            sequence += ["read_X" if quad == "real" else "read_Y", "measure_phase"]
        sequence += ["measure_system", "release_ancilla", "release_system"]
        body, batch = f"body_{name}", f"batch_{name}"
        definitions += [
            Definition(id=body, node=Sequence(children=tuple(sequence))),
            Definition(id=batch, node=MeasurementBatch(
                body=body, settings=(Setting(label=name, metadata=metadata),), repetitions=shots,
                observation_kind="counts")),
        ]
        experiments.append(Experiment(name=name, batch=batch, setting_index=0, readout=ReadoutDetails()))
    return blocks, definitions, experiments, fields


def group_requirements(c, terms, entries):
    """Return the logical ``(bytes, work)`` of reducing one count table of ``entries`` stored bins.

    With c shared classical bits, ``w = ceil(c/64)`` packed words, m stored
    entries (explicit zero-count bins included) and ``L_g`` group terms,

        B_g = 8(w + 15)m + 8L_g,
        W_g = m{c + 44 + (4w + 4)L_g} + L_g{(c - 1)w + 1}.

    The first ``8(w + 1)m`` bytes are the decoded arrays. An additional 80m
    covers floating weights, X, Y, parity conversions and moment products,
    which occur in successive phases, 32m the four centered vectors
    ``dY``, ``dX``, ``w dY`` and ``w dX`` of the variance evaluation
    (``reduce_group``), and 8L_g holds the label means. Work covers key
    decoding, shot-count checks, wordwise AND/popcount/parity passes, the
    coefficient accumulation, label dots and moment dots, and 16m of it
    bounds the 10m + 6 operations of the two residual subtractions, the two
    weight products and the three centered dots with their divisions. The
    last term is the mask construction scanning each label once per word.
    These are conservative counts of routine calls and array passes. Python
    integer, list and record headers, decoder object temporaries, opaque
    NumPy workspace and RSS are outside this logical model, and the bytes
    set aside for the saved CountBin JSON are counted by a separate model.
    """
    w = (c + 63) // 64
    return 8 * (w + 15) * entries + 8 * terms, entries * (c + 44 + (4 * w + 4) * terms) + terms * ((c - 1) * w + 1)


def grouped_pass_requirements(num_qubits, chunks, terms, basis_size, *, settings, pairs, oriented, copies=1,
                              shot_total=None):
    """Return the logical ``(bytes, work)`` of one grouped analysis pass over count chunks.

    ``chunks`` lists ``(L_a, m_a)`` descriptors of the chunks of the pass:
    the term count of a chunk's group and its stored bins, explicit
    zero-count bins included. ``copies``, a positive integer k, repeats
    every descriptor k times without expanding the list, so J descriptors
    describe C = kJ chunks. ``settings`` is the number M of settings the
    pass requests, ``pairs`` the number P of distinct upper-triangle pairs
    with at least one of them, and ``oriented`` whether the pass builds a
    second, oriented ``GroupMoment`` for every chunk. With c = n + 1,
    w = ceil(c/64), L nonidentity terms, T = sum_a L_a over the C chunks and
    C_eff = max(C, M),

        B = sum_a 8(w + 1)m_a + max_a(112m_a + 8L_a)
            + 8 sum_a(L_a + 16) + (n + 16)L + 512b**2
            + o 8(T + 16C) + 8(8C_eff + 24M + 37P),
        W = sum_a W_a + L(n + 1) + 32b**2 + 16b**3
            + o(T + 2C) + 23C_eff + 22P,

    with o = 1 for an oriented view. The first line of each comes from
    ``group_requirements``: decoded arrays summed over the chunks, the
    largest workspace of one reduction (zero without chunks), a logical
    allowance for each canonical contribution's scalars and indices, the
    once-decoded labels and coefficients, and the b-state projected solve.
    The oriented view holds each contribution's L_a label means and 16
    scalar and index slots, and orienting an imaginary row negates at most
    L_a label means, its H0 mean and its overlap mean. ``pooled_statistics``
    and pair assembly hold, in 8-byte slots, 8 per chunk (row, weight,
    variance and contribution references and weighted scratch), 24 per
    setting (the shared setting key, five pooled scalars, the shot total,
    the estimate's value, shot count and name, the group index, two scratch
    views and a pair-descriptor reserve) and 37 per pair (the pair key, the
    H0 and S components and result references, the variance record, six
    two-element reconstruction lists and the returned aggregate). Their work
    is 17C + 16P arithmetic and writes (weights, weighted first moments,
    variance and covariance fields, sums across groups, physical-H
    variances and the H0 and S writes) plus 6C + 6P availability and
    negative-variance scans. With missing data C can be below M. Pooling
    then forms at most min(C, M) settings and C rows before it reports the
    missing settings, which C_eff covers.

    Repetition multiplies the decoded arrays, the canonical contributions
    and sum_a W_a by k, with C and T. The workspace maximum is that of the
    listed descriptors, zero for an empty list, and the label and
    projected-solve terms do not depend on the chunks. M, P and o are the
    caller's complete values, so C_eff = max(kJ, M), not k max(J, M).
    Runtime passes list their actual chunks with k = 1. FixedGCIM planning
    lists one ``(L_g, m)`` per group with k = K, the settings of each group
    (``sampled_setting_count`` with one group: b**2 with both parts and
    b(b+1)/2 with real parts only), M = KG, P = b(b+1)/2 and o = 0. These
    describe one chunk of m = min(s, 2**c) bins per setting of the complete
    pencil, and the law then takes O(G) steps.

    The law holds for any pass: the complete pencil or a subset of its
    pairs, real-only or complex quadratures, unequal shot counts and
    repeated chunks of one setting. It is monotone in every m_a, so upper
    bounds on the bins of tables not yet acquired bound the pass. The
    allowance covers the decoded arrays of the pass population, one
    reduction workspace, the canonical contributions, the oriented view
    when built, pooling and pair assembly, the labels and the projected
    solve. It is a logical model in the units of ``group_requirements``,
    not the process's resident memory, and covers neither other passes nor
    the history of the Run.
    ``adapt_acquisition._keep_current`` releases the cached moments of
    chains outside the current basis and active queries. Saved observations
    and the immutable reconstruction are checked separately.

    ``shot_total=S`` bounds one fresh measurement stage of S shots whose
    count tables do not exist yet, as the planning of ``PencilSampling``
    needs, where the shots of each setting follow from the pilot. Its
    premises: every setting of the pass has exactly one count table, the
    stage's shots sum to S, and ``readout_shape`` and ``_decoded_chunk``
    keep the stored bins of a table of N_a shots at m_a <= min(N_a, 2**c),
    explicit zero bins included. Then sum_a m_a <= U, max_a m_a <= H and
    sum_a L_a m_a <= L_max U with H = min(S, 2**c), U = min(S, M 2**c) and
    L_max the largest listed L_a. The caller lists every descriptor with the
    bins H and makes C = kJ equal to M: one ``(L_g, H)`` per group with
    k = K for G >= 1 groups, and one ``(0, H)`` with k = M and P the
    off-diagonal pairs for a pencil without nonidentity terms. The decoded
    arrays are then charged for U = min(S, k sum_J H) bins, the bin-free
    work ``L_a{(c - 1)w + 1}`` of each descriptor is multiplied by k, and the
    bin work ``U{c + 44 + (4w + 4)L_max}`` is added once, so that

        B = 8(w + 1)U + 112H + 8L_max + 8(T + 16M) + (n + 16)L + 512b**2
            + o 8(T + 16M) + 8(32M + 37P),
        W = U{c + 44 + (4w + 4)L_max} + T{(c - 1)w + 1} + L(n + 1)
            + 32b**2 + 16b**3 + o(T + 2M) + 23M + 22P,

    with the workspace maximum and L_max not multiplied by k and the other
    terms as above. Each replaced term is an upper bound, so the result
    covers unequal shots and group sizes but can exceed the pass that an
    actual allocation produces. It bounds the logical payload and work of
    that one stage, not the Run's history, the Python heap, the resident
    memory or the time, and neither repeated tables of one setting nor two
    stages pooled together, which the runtime check of the actual chunks
    still covers. Actual sparse bin counts do not belong in this form. It is
    an input of the planning check, not a user setting.

    Raises:
        ValueError: If ``copies`` is not a positive integer.
    """
    if type(copies) is not int or copies < 1:
        raise ValueError("copies must be a positive integer")
    c = num_qubits + 1
    w = (c + 63) // 64
    decoded = workspace = records = work = count = labels = binned = longest = 0
    for size_terms, bins in chunks:
        size, cost = group_requirements(c, size_terms, bins)
        decoded += 8 * (w + 1) * bins
        workspace = max(workspace, size - 8 * (w + 1) * bins)
        records += 8 * (size_terms + 16)
        work += cost if shot_total is None else group_requirements(c, size_terms, 0)[1]
        count += 1
        labels += size_terms
        binned += bins
        longest = max(longest, size_terms)
    decoded, records, work, count, labels = (copies * x for x in (decoded, records, work, count, labels))
    if shot_total is not None:
        stored = min(shot_total, copies * binned)
        decoded = 8 * (w + 1) * stored
        work += group_requirements(c, longest, stored)[1] - group_requirements(c, longest, 0)[1]
    pooled, view = max(count, settings), int(oriented)
    return (decoded + workspace + records + (num_qubits + 16) * terms + _projected_solve_bytes(basis_size)
            + view * 8 * (labels + 16 * count) + 8 * (8 * pooled + 24 * settings + 37 * pairs),
            work + terms * c + _projected_solve_work(basis_size)
            + view * (labels + 2 * count) + 23 * pooled + 22 * pairs)


def reduce_group(chunk, setting, *, rows, members, num_qubits):
    """Return the ``GroupMoment`` of one count chunk, or None when it holds no shots.

    ``setting`` is ``(experiment, i, j, quadrature, group)``, ``rows`` the
    nonidentity ``(label, coefficient)`` Pauli rows and ``members`` the row
    indices of the setting's group, empty for the overlap-only setting of
    an operator without nonidentity terms. With ``a = (-1)**phase`` and
    ``p_k`` the system parity of the non-I positions of label k after the
    group's basis rotation, one shot contributes ``Z_k = a p_k`` off
    diagonal and ``Z_k = p_k`` on the diagonal, and
    ``Y_g = sum_(k in g) c_k Z_k``. All counts are
    unconditional, without postselection, so the kept shots equal the
    returned shots. The common width ``num_qubits + 1``, the diagonal's
    unused phase bit and the exact count total against ``returned_shots``
    are checked, and packed masks cover every word of the key. The label
    means satisfy ``mean(Y_g) = sum_(k in g) c_k mean(Z_k)`` mathematically,
    with ordinary reduction rounding in its evaluation. Each H0 quadrature
    is formed by summing the pooled weighted group means ``mean(Y_g)``.
    FixedGCIM, QFD and ADAPT share this reduction and its layout: ancilla
    at classical bit 0, system qubit k at bit k + 1.
    """
    name, i, j, quad, g = setting
    histogram = chunk.histogram()
    if histogram.width != num_qubits + 1:
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
    for index in members:
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
    supplies = i != j and g == 0
    xm = float(np.dot(weights, x)) if supplies else None
    xy = float(np.dot(weights, x * y)) if supplies else None
    variance = vx = cov = None
    if total > 1:
        # Weighting each residual before the dot keeps every variance term
        # nonnegative and avoids squaring a large residual first.
        dy = y - mean
        wy = weights * dy
        variance = float(np.dot(wy, dy)) / (total - 1)
        if supplies:
            dx = x - xm
            wx = weights * dx
            vx = float(np.dot(wx, dx)) / (total - 1)
            cov = float(np.dot(wy, dx)) / (total - 1)
    return GroupMoment(
        experiment=name, contribution_id=chunk.content_id, pair=(i, j), quadrature=quad, group=g,
        shots=total, label_means=tuple(means), mean=mean, second=second, mean_variance=variance,
        overlap_mean=xm, overlap_cross=xy, overlap_variance=vx, mean_covariance=cov,
    )


def pooled_statistics(settings, moments, identity_shift, *, pools=None):
    """Pool the group moments per setting and reconstruct the pair entries and their variances.

    ``settings`` are the planned ``(name, i, j, quadrature, group)`` of the
    pairs to reconstruct (``sampled_settings``), and each moment joins the
    setting with its ``(pair, quadrature, group)``. Repeated measurements a
    of one setting are pooled with ``w_a = N_a/sum_b N_b``: the mean is
    ``sum_a w_a mean_a``, the variance ``sum_a w_a**2 V_a`` and the
    covariance ``sum_a w_a**2 C_a``, because overlap and H0 use the same
    weights here. The quadratures are
    ``H0_ij,q = sum_g mean(Y_ij,q,g)`` and ``S_ij,q = mean(X_ij,q,0)``, with
    ``S_ii = 1``. A quadrature without settings, the imaginary part of a
    diagonal pair or of an off-diagonal pair measured real-only, is an exact
    zero with zero variance. The variances follow ``SampledPairVariance``
    with identity coefficient ``identity_shift``. Pooling uncertainty is
    conditional on an independent-measurement model. Shared deterministic
    simulator seeds do not prove independence. Returns
    ``(estimates, missing, pairs, variances)`` in setting and pair order,
    with empty pairs and variances when a setting is missing.

    A dict ``pools`` receives, for each setting with data, its key
    ``(i, j, quadrature, group)`` and the pooled ``(N, V_Y, V_X, C_XY)``:
    the total shots N and the pooled variances and covariance above, V_X
    and C_XY None outside the overlap role. The pilot shot allocation reads
    them (``pencil_sampling.influence_variances``) instead of pooling the
    moments again.
    """
    names = {(i, j, quad, g): name for name, i, j, quad, g in settings}
    by_key = {key: [] for key in names}
    for row in moments:
        by_key[(*row.pair, row.quadrature, row.group)].append(row)
    estimates, pooled = [], {}
    for key, rows in by_key.items():
        if not rows:
            continue
        n = sum(r.shots for r in rows)
        weights = tuple(r.shots / n for r in rows)

        def average(field, squared=False, rows=rows, weights=weights):
            values = [getattr(r, field) for r in rows]
            if any(v is None for v in values):
                return None
            return fsum((w * w if squared else w) * v for w, v in zip(weights, values, strict=True))

        pooled[key] = tuple(average(field, squared=k >= 2) for k, field in enumerate(
            ("mean", "overlap_mean", "mean_variance", "overlap_variance", "mean_covariance")))
        if pools is not None:
            pools[key] = (n, *pooled[key][2:])
        estimates.append(ScalarEstimate(
            experiment=names[key], value=pooled[key][0], contribution_ids=tuple(r.contribution_id for r in rows),
            weights=weights, chunk_mean_variances=tuple(r.mean_variance for r in rows), sampled_shots=n))
    missing = tuple(name for key, name in names.items() if key not in pooled)
    if missing:
        return tuple(estimates), missing, {}, ()
    measured = {}
    for i, j, quad, g in names:
        measured.setdefault((i, j), {}).setdefault(quad, []).append(g)
    pairs, variances, c = {}, [], identity_shift

    def vsum(values):
        return None if any(v is None for v in values) else fsum(values)

    for (i, j), parts in measured.items():
        h, s, vh, vs, cov, physical = [], [], [], [], [], []
        for quad in ("real", "imag"):
            if quad not in parts:
                for column in (h, s, vh, vs, cov, physical):
                    column.append(0.0)
                continue
            values = [pooled[i, j, quad, g] for g in parts[quad]]
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


def read_groups(chunks, settings, *, members, rows, num_qubits, basis_size, max_bytes, max_analysis_work,
                label="GCIM"):
    """Check one complete grouped analysis pass against the limits, then reduce every count chunk.

    ``chunks`` are the count chunks of the pass, each already checked
    against its Plan (``matched_chunks``) and named by the experiment of
    one of ``settings``, the planned ``(name, i, j, quadrature, group)``
    settings of the pencil (``sampled_settings``). ``members[g]`` lists the
    indices into ``rows``, the nonidentity ``(label, coefficient)`` rows,
    of group g's terms, and ``num_qubits`` is the system width n. The pass
    is checked before the first histogram decode, from the stored bin
    count ``len(chunk.values)`` of each chunk (explicit zero-count bins
    included), by the per-chunk law of ``grouped_pass_requirements`` for
    the M settings on their P pairs, L = ``len(rows)`` terms and the
    ``basis_size`` solve: bytes against ``max_bytes`` and work against
    ``max_analysis_work``. Repeated measurements enter its sums separately.
    The moments are returned in chunk order as reduced, without a second
    oriented view, for ``pooled_statistics``. ``label`` starts the messages
    of the two checks.
    """
    expected = {s[0]: s for s in settings}
    size, used = grouped_pass_requirements(
        num_qubits, ((len(members[expected[chunk.experiment][-1]]), len(chunk.values)) for chunk in chunks),
        len(rows), basis_size, settings=len(expected), pairs=len({s[1:3] for s in expected.values()}),
        oriented=False)
    _check_bytes(size, max_bytes, f"{label} grouped count arrays and scalar payload")
    if used > max_analysis_work:
        raise ValueError(f"{label} grouped analysis needs {used} work units, exceeding "
                         f"max_analysis_work={max_analysis_work}")
    moments = []
    for chunk in chunks:
        setting = expected[chunk.experiment]
        row = reduce_group(chunk, setting, rows=rows, members=members[setting[-1]], num_qubits=num_qubits)
        if row is not None:
            moments.append(row)
    return tuple(moments)


class FixedGCIMResult(Result):
    """Smallest projected eigenvalue from a `FixedGCIM` run, with its projected matrices.

    [`solve`][nwqlib.scientist.solve] returns it for a `FixedGCIM` method,
    and `load_result` reopens a saved one. The answer is `eigenvalue`, the
    lowest kept Ritz value of the projected pencil, in the unit of the
    `Eigenproblem`. It is None when data are missing or the solver
    rejected the pencil. It is a projected estimate, not a certified full-space ground
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
            None when data are missing or the solver rejected the pencil. It is not a
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
            partial data or a scalar identity target. Under `PencilSampling`
            it is formed from the main-stage measurements only.
        analysis_cutoff: Overlap-eigenvalue cutoff used by this analysis.
        target_identification: Fixed statement of what the value does not establish.
        shot_allocation: Under `PencilSampling`, the pilot decision
            ([`ShotAllocation`][nwqlib.algorithms.gcim.fixed_basis.ShotAllocation])
            as it was decided, or None before it. Then `estimates`,
            `group_moments`, `sampled_pair_variances`, `pencil`, `missing` and
            `contribution_ids` concern the main stage only, and a Result from
            before the decision lists every setting in `missing`. The pilot
            measurements stay in the Run's observations. None without
            `PencilSampling`.
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
    shot_allocation: ShotAllocation | None = None

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
        if self.shot_allocation is not None:
            lines.append(self.shot_allocation._summary(self.pencil))
        return lines

    def validate_plan(self, plan: Plan) -> None:
        """Reject this Result unless its value and pencil follow from ``plan``.

        The checks, in order: the same Plan and construction, no value from
        partial observations, the settings of a sampled Plan with nonidentity
        terms equal to those of its recorded group partition and resolved
        quadratures, group statistics only on such a Plan, no exact pair
        reductions on such a Plan, the identity coefficient as the value of a
        quantum Plan whose operator is a multiple of the identity, a pencil of
        the Plan's basis size and analysis cutoff, a value equal to the
        pencil's lowest Ritz value, the overlap filter of the Plan's
        measurements (``sampled_positive_subspace`` exactly when shots were
        taken), sampled enclosure fields recomputed from the pencil, and a
        `shot_allocation` exactly on a `PencilSampling` Plan, naming its
        sampling and one main count per experiment, required once any
        setting has main-stage data. No measurement or eigensolve is
        repeated.

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
        # of its recorded group partition and resolved quadratures.
        if plan.execution == "quantum" and plan.shots is not None and rec.nonidentity_coefficients and (
            not rec.groups
            or tuple(e.name for e in plan.experiments) != tuple(s[0] for s in rec.settings())
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
        _check_shot_allocation(self.shot_allocation, self.missing, plan, "GCIM")

    def energy_endpoint(self):
        """Return this Result's ``EnergyEndpoint`` for energy-shift verification.

        The endpoint carries the projected value, the operator's data
        (Pauli terms or matrix entries) and the content hashes of the
        normalized basis preparations. The ``H -> H + cI`` check of
        ``nwqlib.evidence.energy_shift`` compares two such endpoints without
        new measurements.
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
                 overlap_input_tolerance=0.0, symmetrize_matrices=False):
    """Solve one obtained GCiM pencil and return it as a ``ProjectedPencil``.

    ``hamiltonian`` is ``H0``, the obtained projection without the identity
    term ``identity_shift*I``. The solve uses ``(H0, S)`` and adds the shift
    to the Ritz values afterwards (``_solve_projected_pencil``), and the
    record stores ``H = H0 + identity_shift*S`` in the problem's energy
    unit, with the lowest one or two Ritz values from before the shift in
    ``offset_free_eigenvalues``. The overlap is the Gram matrix of
    normalized columns and is used as obtained. Lanczos instead solves its
    moment pencil in the normalized frame
    ``K = (H - center I)/alpha`` and restores the Ritz values afterwards. The
    stored matrices are always the obtained ones. ``symmetrize_matrices``
    asks the solver to solve the Hermitian parts after its Hermitian check,
    the choice that a Method option such as ADAPT's ``symmetrize_matrices``
    makes. FixedGCIM keeps the default False, because its pencil arrives
    Hermitian by conjugate completion.
    ``sampled`` selects the indefinite-overlap policy of
    ``_solve_projected_pencil``, and ``overlap_input_tolerance`` is the
    derived error of a deterministic Gram matrix that its negative-mode test
    allows. ``enclosure``, the ``processed_pauli_enclosure`` of the measured
    operator, enables the sampled plausibility diagnostic, whose comparison
    the ``sampled_*`` fields record. ``overlap_condition_number`` is the
    ratio of the largest to the smallest kept overlap eigenvalue, or None
    when no direction is kept or the ratio is not finite.
    """
    result, spectrum, failure, _ = _solve_projected_pencil(
        hamiltonian,
        overlap,
        overlap_eigenvalue_cutoff=cutoff,
        symmetrize_matrices=symmetrize_matrices,
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
        offset_free_eigenvalues=() if result is None else result.offset_free_eigenvalues,
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


def pair_reduction_allowance(method, plan, observation, label="GCIM"):
    """Check the summed pair work and one point's workspace, and return the point's remaining work allowance.

    The exact pair route of FixedGCIM and QFD calls it from the Method
    ``method``'s ``reduction_allowance``, and the limits are its
    ``max_classical_products`` and ``max_bytes``. The summed registered work
    of all the Plan's pair reductions, computed once per Plan object and
    cached, is checked against ``max_classical_products`` first. A Plan
    whose sum exceeds the limit is rejected here, before the point's
    preparation work is counted and before any circuit runs. The workspace
    check then rejects a point whose ``pair_bytes`` exceeds ``max_bytes``.
    The allowance returned for one point is the limit less the registered
    work of every other pair reduction of the Plan. Each preparation is
    checked separately against this value, and the cumulative limit holds
    because the work of all pairs was checked together, when each pair is
    computed once. A repeated computation of a pair registers its work
    again, which this Plan-level total does not count. The total is read
    from the Plan's cache, never changed, so the check may run several
    times for one point without recomputing the sum. ``label`` starts the
    messages.
    """
    import json
    from .pair_reducer import pair_bytes, pair_work

    pair_work_total = _pair_work_total(plan)
    if pair_work_total > method.max_classical_products:
        raise ValueError(
            f"{label} pair reductions exceed max_classical_products: registered work {pair_work_total}, "
            f"max_classical_products={method.max_classical_products}. "
            f"Raise {type(method).__name__}.max_classical_products to at least {pair_work_total}."
        )
    own = 0
    for item in observation.positions:
        if item.kind == "reduction":
            parameters = json.loads(item.parameters)
            _check_bytes(pair_bytes(parameters), method.max_bytes, f"{label} pair reduction workspace")
            own += pair_work(parameters)
    return int(max(0, method.max_classical_products - (pair_work_total - own)))


def exact_pair_pencil(method, plan, data, rec, cutoff, label="GCIM"):
    """Pool each exact pair's reductions, assemble the unit-diagonal pencil and solve it.

    ``rec`` is the Plan's ``FixedGCIMReconstruction``, its own for
    FixedGCIM and ``QFDReconstruction.pencil`` for QFD, and the limits are
    the ``max_bytes`` and ``max_analysis_work`` of ``method``. Every
    chunk is one ``block_matrix_elements`` reduction returning
    ``(H0_ij, S_ij)``. Its entry bounds come from the saved-state error of
    the preparation record that produced it
    (``PreparedArtifact.saved_state_error``) through
    ``pair_reducer.entry_bounds`` with the ordered-action error bound, and
    repeated computations of one pair are pooled with the weights
    ``n_a/sum n_a`` of ``pooled_bound``, separately for H0 and S. The
    deterministic Gram filter allows the maximum row sum of the
    off-diagonal overlap bounds (``_exact_overlap_allowance``). When that
    allowance is unavailable it allows no input error beyond the solver's
    own roundoff. Returns ``(estimates, missing, ids, pencil)``: the
    ``PairEstimate`` of every obtained pair, the names of the pairs without
    data, the content hashes of the chunks and the solved pencil, which is
    None when a pair is missing. ``label`` starts the messages.
    """
    import json
    from .pair_reducer import entry_bounds, pooled_bound

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
            raise ValueError(f"{label} needs its pair reduction's two complex scalars")
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
    pencil = None
    if not missing:
        _check_bytes(_projected_solve_bytes(b), method.max_bytes, f"{label} projected analysis")
        _check_projected_solve_work(b, method.max_analysis_work, f"{label} projected analysis",
                                    type(method).__name__)
        h0, s = assemble_pair_pencil(b, values)
        allowance = _exact_overlap_allowance(b, overlap_bounds)
        pencil = solve_pencil(h0, s, cutoff=cutoff, sampled=False, identity_shift=rec.identity_shift,
                              overlap_input_tolerance=0.0 if allowance is None else allowance)
    return tuple(estimates), missing, tuple(ids), pencil


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
      to the offset-free `H0_ij` and `S_ij` as soon as it is prepared, so
      `(M - D) + D` measurements.
    - `shots=n`: the same joint state per off-diagonal pair, with its
      ancilla measured in X and Y together with each qubit-wise commuting
      group of system terms, and each diagonal pair's system state per group,
      so `Q(M - D) max(1, G) + DG` settings for G groups, n shots each. Q is
      2, or 1 when `acquisition` resolves real-only quadratures, which the
      default `"auto"` does exactly when A and every basis state are proved
      real. Each count table supplies its group's weighted Hamiltonian mean
      and, for the first group, the overlap and its covariance with that
      mean.
    - `sampling=PencilSampling(...)` instead of `shots=`: the same settings,
      measured in a uniform pilot stage and again in a main stage whose
      shots per setting the pilot pencil decides, with only the main-stage
      measurements in H and S.
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
        max_classical_products: Default `100_000_000`. Limit on the counted
            classical work of basis preparation and projection, and on the
            summed classical work of the exact pair reductions.
        input_conversion: Default `"auto"`, which keeps Pauli input as it
            is and, for quantum execution, converts an explicit dense matrix
            of original dimension at most 16 to Pauli form. Classical
            execution uses the supplied matrix representation. `"dense_pauli"`
            also permits the conversion of sparse matrices and of larger dense
            matrices, within `max_conversion_work` and `max_bytes`.
        max_conversion_work: Default `100_000_000`. Limit on the counted
            work of operator-to-Pauli conversion, including the conversion
            that `input_conversion="auto"` selects.
        max_admission_steps: Default `10_000_000`. Upper limit on the
            planning work of checking each `Program` that the method builds
            (NWQLib's description of a circuit as named steps). It caps the
            number of stored fields of a `Program` and the work units of one
            check of it. Summing a `Program`'s resource counts may use up to
            24 times this value. A larger `Program` is rejected with a
            ValueError that names this field, the planning stage that exceeded
            the limit and its count. Raising the limit permits a larger check and changes no
            quantum operation
            ([planning work limit](../../development/program_checks.md#planning-work-limit)).
        acquisition: Default `PencilAcquisition()`. How quantum execution
            measures H and S
            ([`PencilAcquisition`][nwqlib.algorithms.gcim.fixed_basis.PencilAcquisition]).
            FixedGCIM measures grouped joint states only, so
            `matrix_measurement="termwise"` is rejected. Each basis state is
            one supplied preparation, with no generator chain whose action
            on the all-zero state is known, so quantum planning raises
            `ApplicabilityError` for `pair_preparation="vacuum_reference"`
            and `"vacuum_restore"`. Classical execution and a scalar
            operator, which prepare no joint state, keep the request
            without effect. The Plan records the resolved choices in
            `plan.reconstruction.acquisition`.
        max_native_two_qubit_depth: Default `None`, which requests no depth
            limit. `None` is neither a depth of 0 nor evidence that the
            circuits suit a device. A nonnegative integer D requires each
            submitted quantum circuit, every grouped count setting and every
            exact pair reduction, to have a native two-qubit depth of at most
            D, with D itself allowed. This depth is that of the circuit the
            backend prepared for execution, with the basis-state
            preparations, their controlled forms, the measurement basis
            changes and the backend's qubit routing, computed once when the
            circuit is prepared
            ([`PreparedArtifact.native_depth`][nwqlib.execution.PreparedArtifact]).
            Each circuit is checked just before its submission, and one whose
            depth exceeds D or is unavailable is not submitted: the run raises
            `nwqlib.backends.inspection.NativeDepthCapRefusal`, which names
            the setting, the limit and the depth or why it is unavailable.
            Of the current backends only IBM Runtime computes this depth, for
            its compiled circuits, and only when its options keep the
            provider's dynamical decoupling and twirling off and
            qiskit-ibm-runtime 0.49.0 is installed
            (`nwqlib.backends.IBMRuntimeBackend`). Otherwise the depth is
            unavailable and an integer limit refuses the first circuit, as it
            does on every other backend, Aer included. The limit also refuses
            a circuit whose preparation recorded a version of nwqlib, Qiskit
            or qiskit-ibm-runtime other than the installed one, as in a Run
            reopened in another environment, and then the refusal has its
            `provenance` set. Such a Run stays resumable. Resumed where the
            recorded versions are installed, it checks the same circuit's
            depth against D again and submits it when the depth meets the
            limit. In the installed environment, a new Run prepares its
            circuits again, and their preparation records then decide the
            limit. Execution without the limit is another option. Settings
            submitted earlier keep their counts and shots in the Run, and
            `plan.method.analyze(plan, run.data, settings={})` returns a
            Result that lists the settings without data in `missing` and has
            no pencil. The limit changes no circuit, setting or Hamiltonian
            term. No value suits every device, so choose D for the device and
            the circuits, for example from the depth recorded for a setting
            prepared by `nwqlib.prepare` (`prepared.run.prepared_artifacts`).
            Under `sampling` the limit applies to every pilot and main
            circuit in the same way, and the settings without main-stage
            data are the `missing` ones.
        sampling: Default `None`, which measures every setting with the
            `shots=` of the call. A
            [`PencilSampling`][nwqlib.algorithms.gcim.fixed_basis.PencilSampling]
            replaces `shots=` with a pilot stage and a main stage whose
            shots per setting the pilot decides, and needs quantum
            execution. The Result's `shot_allocation` records that decision.
            `nwqlib.estimate(plan)` then reports the totals of the whole
            Plan, such as its settings and shots, as unavailable, because
            each setting is measured in a pilot stage and again in a main
            stage whose shots the pilot decides. One prepared setting, whose
            stage and shots are fixed, is estimated normally from its
            preparation record. The shot budget of a Run is
            `PencilSampling.total_shots`.

    Raises:
        ValueError: If `acquisition` requests termwise measurement.

    Examples:
        `H = ZZ + 0.5*(XI + IX)` on two qubits has lowest eigenvalue
        `-sqrt(2) = -1.4142135623...`, which this three-state basis
        recovers on Aer with exact readout (`shots=None`, in which the
        simulator computes the matrix elements from its statevector without
        shot sampling, subject to numerical rounding):

        >>> from qiskit.quantum_info import SparsePauliOp
        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import FixedGCIM
        >>> H = SparsePauliOp.from_list([("ZZ", 1), ("XI", 0.5), ("IX", 0.5)])
        >>> basis = ([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0])
        >>> result = solve(Eigenproblem(A=H), method=FixedGCIM(basis=basis), seed=7)
        >>> print(round(result.eigenvalue, 10))
        -1.4142135624
    """

    schema_version: Literal[3] = 3
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
    max_admission_steps: PositiveInt = 10_000_000
    acquisition: PencilAcquisition = PencilAcquisition()
    max_native_two_qubit_depth: Count | None = None
    sampling: PencilSampling | None = None
    result_type: ClassVar[type] = FixedGCIMResult

    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR

    @model_validator(mode="after")
    def _grouped_only(self):
        """FixedGCIM has only the grouped joint-state construction."""
        if self.acquisition.matrix_measurement != "grouped":
            raise ValueError("FixedGCIM measures grouped joint states; matrix_measurement='termwise' is "
                             "ADAPT's comparison construction")
        return self

    def plan(self, problem, *, output, execution, shots, rng):
        """Choose normalized trial columns and the computation of their complex H/S matrix elements."""
        if self.sampling is not None:
            if shots is not None:
                raise ValueError("FixedGCIM.sampling sets the shots of every setting and stage; omit shots=")
            if execution != "quantum":
                raise ValueError("FixedGCIM.sampling measures counts on a quantum register; "
                                 "use execution='quantum'")
            # Plan.shots marks a counts Plan at the two-shot stage minimum.
            # The stages bind each setting's actual shots (pencil_workflow).
            shots = 2
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
        # The measurement choices are resolved once, before any count or
        # construction depends on them. The real proof scans the recorded
        # Pauli rows and every basis state's stored data.
        acquisition = resolve_acquisition(
            self.acquisition, execution=execution, shots=shots, acquires=bool(nonidentity),
            real_reason=lambda: real_hamiltonian_reason(input_terms.rows()) or next(
                filter(None, map(real_state_reason, basis)), None),
            scope=f"recorded Hamiltonian Pauli terms and all {b} basis states",
            pairs=fixed_pair_preparations(basis, self.acquisition.pair_preparation) if nonidentity else ())
        quadratures = acquisition.quadratures or "complex"
        groups, e = _pencil_admission(self, q, b, input_terms, nonidentity, shots, quadratures,
                                      sampling=self.sampling)
        blocks, definitions, experiments, staged = [], [], [], {}
        metadata = MetadataRef(format=METHOD, data=basis_record.reference)
        if e and shots:
            settings = tuple(sampled_settings(pencil_pairs(b), len(groups), quadratures))
            blocks, definitions, experiments, staged = _sampled_construction(
                *_fixed_pair_recipes(basis, acquisition.pairs), q, metadata, groups, shots, settings,
                sampling=self.sampling)
            registers = (
                Register(name="ancilla", width=1, role="clean_ancilla"),
                Register(name="system", width=q),
            )
            classical = (
                ClassicalValue(name="phase", dtype="bits", width=1),
                ClassicalValue(name="system_bits", dtype="bits", width=q),
            )
        elif e:
            blocks, definitions, experiments = _exact_pair_construction(
                *_fixed_pair_recipes(basis, acquisition.pairs), b, operator, q, metadata)
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
            **staged,
        )
        construction = SelectedConstruction(
            program=program,
            selections=tuple(block.record for block in blocks),
        )
        # Keep identity terms for H_identity = c S rather than acquiring them again.
        reconstruction = FixedGCIMReconstruction(terms=input_terms, basis_size=b, num_qubits=q, groups=groups,
                                                 acquisition=acquisition)
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

    def before_submit(self, plan, prepared, *, run):
        """Refuse a circuit whose native two-qubit depth exceeds `max_native_two_qubit_depth` or is unavailable, before the Run records its submission."""
        from nwqlib.backends.inspection import check_native_depth_cap

        check_native_depth_cap(prepared, self.max_native_two_qubit_depth)

    def reduction_allowance(self, plan, point, *, observation, width, run):
        """Check the summed pair work and a point's workspace, and return the remaining allowance (``pair_reduction_allowance``)."""
        return pair_reduction_allowance(self, plan, observation)

    def prepare_all_refusal(self):
        """PencilSampling rejects ``settings="all"``, because the main stage takes its shots from the pilot."""
        if self.sampling is not None:
            from .pencil_workflow import prepare_all_refusal

            return prepare_all_refusal("FixedGCIM")
        return super().prepare_all_refusal()

    def query_catalog_reason(self):
        """Under PencilSampling, why the Plan's experiments are templates of the pilot and main points (``pencil_workflow.CATALOG_REASON``), else None.

        Every point that the stages measure is ``Plan.resolve`` of one
        experiment with ``stage`` and ``shots`` bound, and its construction
        keeps the experiment's batch, counts readout and repetitions
        expression ``shot_count``.
        """
        if self.sampling is None:
            return None
        from .pencil_workflow import CATALOG_REASON

        return CATALOG_REASON

    def prepare(self, plan, *, run, settings="first"):
        """Prepare the next pilot or main setting under PencilSampling (``pencil_workflow``), else the static path.

        Under PencilSampling ``settings`` is always ``"first"``, because
        ``prepare_all_refusal`` rejects ``"all"`` before a Run exists.
        """
        if self.sampling is None:
            return super().prepare(plan, run=run, settings=settings)
        from .pencil_workflow import prepare_stages

        return prepare_stages(plan, run=run, rec=plan.reconstruction, label="GCIM")

    def execute(self, plan, *, run):
        """Run the pilot and main stages of ``pencil_workflow`` under PencilSampling, else the static path."""
        if self.sampling is None:
            return super().execute(plan, run=run)
        from .pencil_workflow import execute_stages

        return execute_stages(plan, run=run, rec=plan.reconstruction, label="GCIM")

    def recover_analysis(self, plan, *, run):
        """Clear an interrupted PencilSampling allocation or final analysis for an explicit retry (``pencil_workflow``)."""
        if self.sampling is None:
            return super().recover_analysis(plan, run=run)
        from .pencil_workflow import recover_stages

        return recover_stages(plan, run=run)

    def _plan_classical(self, problem, output, rng):
        """Declare bounded classical work for H0 = V-dagger (A - cI) V and S = V-dagger V.

        ``c = trace(A)/d`` (``_mean_diagonal``) is removed from A before the
        product (``_offset_free_actions``) and added back after the solve. The
        work formula is ``_classical_projection_requirements``, which counts
        one block action of the distinct state columns (Pauli input) or one
        operator action per distinct column with the extra work of removing c
        (other input), ``2 b**2`` inner products of length d for the two Gram
        matrices, counted as ``2 b**2 d`` scalar work units, and any known
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
            acquisition=ResolvedAcquisition(requested=self.acquisition),
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
        """Bind the classical projected-matrix routine without computing its trial vectors yet."""
        from nwqlib.blocks.kernels import BoundKernel, KernelOutput
        from nwqlib.execution import ScalarValue

        (kernel,) = plan.construction.kernels
        rec = plan.reconstruction

        def invoke():
            """Apply A - cI once to the block of normalized columns and return the upper-triangular H0/S scalars."""
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
        """Reconstruct the obtained Hermitian pencil and solve its usable overlap subspace."""
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
        allocation = None
        if self.sampling is not None:
            from .pencil_workflow import frozen_allocation

            allocation = frozen_allocation(plan, data)
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
            shot_allocation=allocation,
        )._attach(plan, data)

    def _analyze_groups(self, plan, data, cutoff):
        """Reduce the counts of every group measurement, pool the settings and solve the sampled pencil.

        ``read_groups`` checks the pass against the limits and reduces each matched chunk,
        ``pooled_statistics`` pools repeated measurements and reconstructs
        ``(H0_ij, S_ij)`` with their variances, and a complete pencil is
        solved with the positive-subspace policy, ``identity_shift = c_I``,
        and the processed-operator enclosure as a numerical diagnostic of the
        sampled Ritz estimate. The recorded ``H`` is ``H0 + c_I S``. Under
        PencilSampling the pass reads the main-stage chunks only, and the
        Result carries the frozen ``ShotAllocation`` of the saved iteration
        state, so neither the pilot solve nor the allocation is repeated.
        """
        rec = plan.reconstruction
        settings = rec.settings()
        chunks = tuple(chunk for chunk, _ in matched_chunks(plan, data))
        allocation = None
        if self.sampling is not None:
            from .pencil_workflow import frozen_allocation, stage_chunks

            allocation, chunks = frozen_allocation(plan, data), stage_chunks(chunks, 1)
        moments = read_groups(
            chunks, settings,
            members=[group.indices for group in rec.groups] or [()], rows=rec.nonidentity_terms,
            num_qubits=rec.num_qubits, basis_size=rec.basis_size, max_bytes=plan.method.max_bytes,
            max_analysis_work=plan.method.max_analysis_work)
        estimates, missing, pairs, variances = pooled_statistics(settings, moments, rec.identity_shift)
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
            shot_allocation=allocation,
        )._attach(plan, data)

    def _analyze_pairs(self, plan, data, cutoff):
        """Pool each exact pair's reductions, assemble the unit-diagonal pencil and solve it (``exact_pair_pencil``)."""
        estimates, missing, ids, pencil = exact_pair_pencil(self, plan, data, plan.reconstruction, cutoff)
        value = pencil.eigenvalues[0] if pencil and pencil.eigenvalues else None
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
        """Save the chosen inputs so loading can rebind them without planning again.

        The configured basis and the selected basis are stored separately.
        ``selected_basis_indices`` records which positions shared one state
        object, because the classical computation builds a repeated input once
        and a reload must reproduce that binding. Quantum plans also store their
        selected circuit blocks. See docs/saved_evidence.md.
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
            format="fixed_gcim/7",
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
        """Rebind the saved Plan, operator, basis and blocks without choosing them again or running circuits.

        Loading reads the stored construction instead of calling ``plan``, so a
        reopened result keeps its original selected blocks even if planning
        rules change. A classical plan rebinds its classical routine to the restored
        states, with no new operator action at load time.
        """
        from nwqlib._choice_archive import unsupported_archive_format
        from nwqlib.blocks._archive import read_blocks

        if saved.get("format") != "fixed_gcim/7":
            raise unsupported_archive_format("FixedGCIM archive", saved.get("format"), "fixed_gcim/7")
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
