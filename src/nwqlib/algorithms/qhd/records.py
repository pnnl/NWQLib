"""Immutable selected QHD numerical data and observed finite-box results.

``QHDReconstruction`` holds everything that planning fixed once, the objective
support tables, the initial amplitudes, the compiled step blocks
(``QHDBlock`` for one-hot, ``QHDBinaryBlock`` for binary) and the phase and
approximation ledgers with the lower-range omissions (``QHDRangeOmissions``).
Execution, analysis and loading read it instead of evaluating the objective
again. ``QHDAnalysis`` is the analyzed readout, the best observed candidate and
the most probable point with its tie window.
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

from hashlib import sha256
from typing import Annotated, Literal
from pydantic import Field, StrictBool, model_validator
from nwqlib.algorithms.protocol import AlgorithmDescriptor
from nwqlib.artifacts import ArtifactManifest
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Plan
from nwqlib.core.records import (
    ContentID,
    FrozenArray,
    InputRef,
    Nonnegative,
    PositiveInt,
    Real,
    Record,
    Source,
    Text,
)
from nwqlib.execution import KernelApplication
from nwqlib.operators.access import Count

from .validation import _NORMAL_MIN

# Resolution of the most-probable selection (QHDAnalysis.mode_status), which
# RefinementLevel.mode_status and ALEvaluation.mode_status copy.
ModeStatus = Literal["resolved", "unresolved", "unavailable"]

# Every field of this Source enters the content identity of QHD Plans and
# saved results, so its text is part of the record format. Leng et al.
# equation numbers cited in this package refer to arXiv:2303.01471v1.
METHOD = Source(
    name="qhd",
    version="2",
    domain="one-hot Dirichlet or periodic, or binary periodic, box optimization; best observed candidate and most "
    "probable point",
    reference="arXiv:2303.01471v1",
)


def objective_reference(expression, variables, bounds):
    """Return the input reference that hashes the objective text, variable order and bounds.

    ``expression`` and ``variables`` are the stored ``srepr`` strings. The
    SHA-256 identity binds a Plan to exactly this objective on this box, and
    the text is never evaluated.
    """
    identity = sha256((expression + repr(variables) + repr(bounds)).encode()).hexdigest()
    return InputRef(
        identity="sympy:" + identity, representation="admitted_sympy_expression", source=METHOD
    )


def table_extrema(values):
    """Return ``(minimum, maximum, maximum magnitude)`` of a finite nonempty float64 array by selection.

    For a finite nonempty binary64 array, ``np.max`` and ``np.min`` select
    input values without arithmetic. Thus
    ``Fraction(float(np.max(a))) == Fraction(max(a))``, and likewise for min
    and maximum absolute value. Signed zeros can have different selected
    sign bits, but their exact ``Fraction`` is zero and their numerical
    range is unchanged. A NaN is invalid and must already have been
    rejected. The extrema are cached with the immutable array, bound to the
    same table content and shape. Three extrema do not determine an
    absolute sum.
    """
    import numpy as np

    a = np.asarray(values, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    return lo, hi, max(abs(lo), abs(hi))


class SupportValues(Record):
    """Objective values of one support group on its grid tuples, evaluated once at planning time.

    Store the finite float64 entries of one additive support group in C
    order, with the first support variable most significant. The array and
    its extrema are immutable selected data. The cached minimum, maximum and
    maximum magnitude are exact selections from the same array.

    Attributes:
        support: Sorted variable indices of one additive objective group.
        values: ``FrozenArray`` whose ``.array`` is the read-only
            one-dimensional float64 array of the ``K**len(support)`` finite
            objective values, first support variable most significant
            (``compiler.QHDCompiler._objective_grid_values``).
        minimum: The smallest entry of ``values`` (``table_extrema``).
        maximum: The largest entry of ``values``.
        magnitude: The largest absolute entry of ``values``,
            ``max(|minimum|, |maximum|)``.
    """

    support: tuple[Count, ...]
    values: FrozenArray
    minimum: Real
    maximum: Real
    magnitude: Nonnegative

    @classmethod
    def tabulate(cls, support, values):
        """Return the record of one support table with its extrema selected from ``values`` (``table_extrema``)."""
        table = FrozenArray(values)
        minimum, maximum, magnitude = table_extrema(table.array)
        return cls(support=support, values=table, minimum=minimum, maximum=maximum, magnitude=magnitude)

class QHDWalshPhase(Record):
    """Upper allowances, in radians, for the identity phases of a Plan's Walsh diagonal blocks.

    The four scalars of ``binary.WalshPhaseTerms`` that
    ``binary.compile_binary_steps`` returns for the selected binary
    constructions, stored unchanged at planning. Kept-state phase admission
    reads them (``method._admit_compiled_phase``).

    Attributes:
        count: B, the number of Walsh blocks, each of which assigns its
            identity phase to the circuit's global phase.
        identity_sum: Y, the sum of the absolute computed identity phases.
        formation: F_W, the summed formation allowances of the identity
            phases against the exact table or model identity.
        reconstruction: R_W, the summed per-entry allowances of the emitted
            phases against the exact phases of the same identity phase and
            kept rotations.
    """

    count: Count
    identity_sum: Nonnegative
    formation: Nonnegative
    reconstruction: Nonnegative


class QHDBlock(Record):
    """One compiled step block, either a fused XX+YY hopping or a number-projector phase.

    A projector is stored by its qubits, not expanded into the ``2**s``
    Pauli-Z strings of ``prod n``.

    Attributes:
        kind: ``kinetic`` for one XX+YY hopping link or ``number_projector``
            for one product of number operators.
        time_step: Duration of this block.
        time: Midpoint schedule time of the step that owns the block.
        support: Two linked qubits, or the projector's qubits.
        coefficient: Pauli coefficient of each of XX and YY, or the weighted
            objective value of the projector.
        angle: Duration times the one-hot matrix element. The hopping
            circuit is ``exp(-i time_step * coefficient * (XX + YY))`` and
            the projector circuit is ``exp(-i angle (prod n - I/2**s))``.
        provider: Selected native construction.
        cx: CX count of that construction before routing.
    """

    kind: Literal["kinetic", "number_projector"]
    time_step: Real
    time: Real
    support: tuple[Count, ...]
    coefficient: Real
    angle: Real
    provider: Text | None = None
    cx: Count = 0


class QHDBinaryBlock(Record):
    """One compiled block of a binary-encoded step: a kinetic conjugation or a support-table phase.

    The block stores its exponent and resolved synthesis, not its angles.
    The construction rebuilds the phase table from the grid (kinetic) or the
    stored support table (potential) and synthesizes it again with the same
    exponent and choice (``binary.PhaseTable.synthesize``), which also gives
    the ``cx`` and ``rotations`` recorded here.

    Attributes:
        kind: ``"binary_kinetic"`` applies ``F^dagger exp(-i exponent diag(E)) F``
            on one variable's register, with F the selected QFT and E the
            Fourier-mode energies of ``QHD.kinetic_model``.
            ``"binary_potential"`` applies ``exp(-i exponent diag(v))`` for
            one support table v.
        time_step: Duration of this block, dt or dt/2 for a second-order
            potential half.
        time: Midpoint schedule time of the step that owns the block.
        variables: The kinetic block's variable, or the potential block's
            sorted support.
        exponent: ``time_step`` times the step's kinetic or potential weight
            (``schedules.step_weights``), so that ``exponent * E`` is a phase
            in radians.
        synthesis: Resolved diagonal synthesis, ``"dense_diagonal"`` or
            ``"walsh_rotations"`` (``binary.BinarySynthesis``).
        cx: CX count of the block's construction at optimization level 0
            before routing, both QFTs included for a kinetic block.
        rotations: Rotation gates of the construction before angle-specific
            simplification: three phase gates per controlled phase of each
            QFT, one Rz per Walsh string, or the ``2**n - 1`` multiplexor
            angles of a nonzero dense diagonal on n qubits, of which
            exact-zero angles are omitted in the circuit.
    """

    kind: Literal["binary_kinetic", "binary_potential"]
    time_step: Real
    time: Real
    variables: tuple[Count, ...]
    exponent: Real
    synthesis: Literal["dense_diagonal", "walsh_rotations"]
    cx: Count
    rotations: Count


class QHDTableOmission(Record):
    """The lower-range data of one support table (``QHDRangeOmissions``).

    Attributes:
        subnormal_entries: Finite nonzero entries below ``2**-1022`` in
            magnitude. The stored table keeps them unchanged.
        walsh_coefficients: Nonidentity Walsh coefficients whose
            normalization would fall below ``2**-1022`` and that the binary
            construction therefore uses as zero (``binary.walsh_admission``).
            Zero unless a selected Walsh block uses the table's coefficients.
        walsh_identity: Whether the identity coefficient was used as zero
            for the same reason.
        walsh_identity_charge: The exact ``d_0 = |s_0|/N`` of an omitted
            identity coefficient, rounded upward. The Walsh identity
            formation of the ``identity_phase`` entry charges it times
            ``|X|`` for every Walsh block occurrence of the table.
        walsh_charge: The exact sum of ``d_m = |s_m|/N`` over the omitted
            nonidentity coefficients, rounded upward. The ``angle_formation``
            entry charges it times ``|X|`` for every Walsh block occurrence
            of the table.
    """

    subnormal_entries: Count
    walsh_coefficients: Count
    walsh_identity: StrictBool
    walsh_identity_charge: Nonnegative
    walsh_charge: Nonnegative


class QHDRangeOmissions(Record):
    """What planning omitted because its arithmetic would fall below the normal binary64 range, with the charges.

    Planning keeps every coefficient, phase and gate parameter that it forms
    zero or at least ``2**-1022`` in magnitude. A contribution whose
    required product would be nonzero and smaller is omitted, and the
    omission is charged against its intended action in one entry of the
    error ledger (``resources.circuit_resources``). The owners named below
    derive the charges, which are exact sums rounded upward once. The
    lowering of a dense diagonal forms its own angles, which belong to the
    later ``diagonal_wrap`` entry. Zero counts and charges mean that nothing
    was omitted, and an omitted value is never reported as an exact zero.
    Kinetic coefficients, energies, hopping terms and diagonal phases outside
    the range are refused instead of omitted
    (``validation._kinetic_range_error``). A binary kinetic block's
    rotations, dense phase entries and identity phase follow the same
    product rule as a potential block's (``binary.PhaseTable.synthesize``)
    and are counted with them.

    Why a range rule, and why omission. The phase and angle ledgers bound
    rounding with the relative model ``fl(z) = z (1 + delta)``,
    ``|delta| <= u``, which holds for results in the normal binary64 range.
    A result below ``2**-1022`` rounds with an absolute error of up to
    ``2**-1075`` that no relative bound covers (Higham, *Accuracy and
    Stability of Numerical Algorithms*, 2nd ed., doi:10.1137/1.9780898718027,
    Theorem 2.2 and Eq. (2.8)). Planning therefore requires every product,
    quotient and power-of-two scaling that a ledger charges, and every
    coefficient and gate parameter, to be zero or in that range
    (``validation._normal_range``). Quantities formed from the
    objective or the initial state can fall below it on ordinary problems,
    such as the far tail of a narrow Gaussian well or of a narrow Gaussian
    initial state, and refusing those Plans would make such problems
    unplannable.
    Such a contribution is omitted instead. Because
    ``||exp(-i a G) - I|| <= |a| ||G||`` for a Hermitian G, omitting an
    evolution factor ``exp(-i a G)`` is charged ``|a| ||G||``. A structured
    preparation chain cut short is charged an upper bound on the distance of
    its prepared vector from the chain's target
    (``initial_state.chain_selection``). Both charges are computed exactly
    and rounded upward.

    Attributes:
        policy: The omission policy, part of the Plan's identity, so a saved
            selection made under another policy is not silently reused.
        tables: One record per support table, in ``support_values`` order.
        projectors: One-hot number-projector occurrences omitted together
            with their identity events, both second-order potential halves
            counted (``potential.PotentialCompiler.select_occurrences``).
        projector_charge: The sum of ``(1 - 2**-s) |t b v|`` over them,
            already part of the ``rotation_pruning`` entry and of
            ``pruning_error_bound``.
        rotations: Binary Walsh rotations of the selected constructions
            omitted because ``2 x c_m`` would fall below the range.
        rotation_charge: The sum of ``|x c_m|`` over them, part of the
            ``rotation_pruning`` entry and of ``pruning_error_bound``.
        dense_entries: Binary dense phase entries of the selected
            constructions set to zero for the same reason. An entry is not
            a gate.
        dense_charge: The sum over dense block occurrences of the largest
            omitted ``|x v_z|``, part of the ``rotation_pruning`` entry and of
            ``pruning_error_bound``.
        identity_phases: Binary Walsh identity phases ``-x c_0`` emitted as
            zero.
        identity_phase_charge: The sum of ``|x c_0|`` over them, part of the
            Walsh identity formation of the ``identity_phase`` entry.
        identity_events: Omitted identity events of the phase ledger: the
            identities of the omitted projectors and the objective-constant
            contributions of the compiled routes, recorded as zero events
            that keep the ledger's contribution count, or the one constant
            phase ``-c sum_k dt b_k`` that the ``schrodinger`` and
            ``split_step`` flavors restore for a kept state.
        identity_charge: The sum of ``2**-s |t b v|`` over the omitted
            projectors and the exact omitted constant phases, added to the
            phase ledger (``identity_phase`` entry and kept-state phase
            admission).
        chains: Structured one-hot preparation, per register: the links of
            the chain without the cutoff, the links emitted and the cutoff
            charge, part of the ``state_preparation`` entry
            (``initial_state.chain_selection``). Empty for other
            preparations.
    """

    policy: Literal["lower_range_omission/1"] = "lower_range_omission/1"
    tables: tuple[QHDTableOmission, ...]
    projectors: Count = 0
    projector_charge: Nonnegative = 0.0
    rotations: Count = 0
    rotation_charge: Nonnegative = 0.0
    dense_entries: Count = 0
    dense_charge: Nonnegative = 0.0
    identity_phases: Count = 0
    identity_phase_charge: Nonnegative = 0.0
    identity_events: Count = 0
    identity_charge: Nonnegative = 0.0
    chains: tuple[tuple[Count, Count, Nonnegative], ...] = ()


class QHDReconstruction(Record):
    """QHD data fixed at planning time and reused by execution, analysis and loading.

    It holds the objective text, its support tables, the compiled schedule
    and the phase and approximation ledgers. physical_phase is the omitted
    identity phase that the native circuit and the ir_product reference
    restore. dropped_angle_sum sums the removed rotation angles, and the two
    error bounds are operator-norm bounds on the emitted product, none of
    them an objective-gap bound. size_units count array entries and
    actions, not CPU instructions.

    Attributes:
        method_id: Content identity of the QHD Method that made the selection.
        expression: ``srepr`` of the original objective.
        symbolic_variables: ``srepr`` of each variable, in coordinate order.
        support_values: Objective tables by variable support.
        constant: Sum of the objective terms with no variable.
        initial_amplitudes: Normalized nonnegative amplitude vector of the
            Method's initial state for each variable, K entries each, from
            the one evaluation at planning (``initial_state.evaluate``).
            Each vector is immutable float64, in the selected entries and axis
            order. ``method._initial_state_bytes`` counts each distinct data
            buffer and its array and container headers once. The
            native builder prepares these vectors, and the classical kernel
            forms the tensor product of them for a general product state. For
            the uniform state and the kinetic ground state on the periodic
            grid the classical kernel fills ``1/sqrt(K**d)`` directly instead
            (``initial_state.restricted_state``).
        initial_state_error: Construction error bound of the classical start
            vector in units of u (``initial_state.restricted_state_error``),
            the start term of the host state budget. It is 2 for the direct
            uniform fill.
        steps: Ordered blocks of every step, ``QHDBlock`` records for the
            one-hot encoding and ``QHDBinaryBlock`` records for the binary
            encoding. Empty when neither native execution nor the
            ``ir_product`` flavor needs them.
        step_weights: ``(t, kinetic_weight, potential_weight)`` per step,
            with t the midpoint ``fl((k + 1/2) dt)`` of step k and
            ``dt = fl(total_time/num_steps)``. Under the Method's
            ``"midpoint"`` coefficient rule the weights are ``a(t)`` and
            ``b(t)``. Under ``"integrated"`` they are the step averages
            ``A/dt`` and ``B/dt`` of the schedule's interval integrals over
            the step. The record requires what the range admission of
            planning (``schedules.step_weights``) guarantees under both
            rules. Every weight is a positive normal binary64 number, and so
            is the first midpoint ``dt/2``. The midpoints do not decrease,
            because rounding is monotone. Planning places each midpoint
            strictly between its binary64 step ends ``fl(k dt)`` and
            ``fl((k + 1) dt)`` only under the midpoint rule, which the record
            does not store, and it compares no midpoint with total_time. At
            the default limits the midpoints are also strictly increasing and
            below total_time T: ``max_bytes = 10**10`` and ``_STEP_BYTES`` of
            768 bytes per step allow at most about 1.3e7 steps, and with a
            normal dt and the exact ``k + 1/2`` the relative rounding bound
            gives the last midpoint at most ``T (1 - 1/(2 N)) (1 + u)**2 < T``
            and keeps consecutive midpoints distinct for ``N < 2**51``.
            ``max_work`` and ``max_bytes`` have no upper bound, however. At
            ``N = T = 2**52 + 4`` the integrated rows repeat a midpoint and end
            with a midpoint equal to T, and at T = 3.1 with
            ``N = 17 * 2**48`` the last midpoint is 3.1000000000000005. With
            raised limits such Plans pass the admissions that precede the row
            builder, although they cannot practically be built. The record
            therefore requires neither
            strictly increasing midpoints nor an upper bound.
        compact_schedule_selected: Whether ``steps`` were compiled.
        physical_phase: Total omitted identity phase in radians, accumulated
            with Neumaier's compensation on the one-hot encoding
            (``compiler.QHDCompiler._record_global_phase``) and with
            ``math.fsum`` on the binary encoding
            (``binary.compile_binary_steps``).
            ``method._compiled_phase_allowance`` bounds its rounding against
            the exact identity angle of the stored data.
        phase_sources: Omitted phase by source (kinetic diagonal, constant
            objective and projector identity for one-hot, the constant
            objective for binary, whose diagonal blocks keep their own
            identity phases).
        dropped_angle_sum: Sum of absolute angles removed by the threshold:
            one-hot block angles, or binary ``Rz`` angles.
        dropped_count: Number of nonzero rotations removed by the threshold.
        pruning_error_bound: Bound on ``||U - U_0||_2`` for the emitted
            product U and the product ``U_0`` with every removed rotation
            restored, by telescoping over the removed rotations. The
            threshold's removals are charged here as below, and the
            lower-range omissions by their exact charges
            (``range_omissions``). A one-hot
            block ``exp(-i G)`` with generator norm at most its absolute
            angle contributes that angle (the hopping ``XX + YY`` has
            eigenvalues within ``[-2, 2]`` and its angle is twice
            ``time_step * coefficient``, a projector's traceless generator has
            eigenvalues within ``(-1, 1)`` times its angle), so the bound is
            the sum of the dropped angles. A binary ``Rz(theta)`` differs from
            I by ``2 |sin(theta/4)| <= |theta|/2``, so the bound is half of
            it. The dropped binary64 angles are summed exactly, the binary sum
            is halved exactly, the range charges are added exactly, and the
            result is rounded upward once, so the recorded value is never
            below the exact bound and is 0.0 exactly when nothing is removed.
        aqft_error_bound: Bound on ``||U - U_0||_2`` for the emitted product U
            and the product ``U_0`` with exact QFTs,
            ``min(2, 2 N_s d e_F)`` with ``e_F = binary.qft_error_bound`` per
            QFT, because a matched pair of truncated transforms around an
            exact unitary diagonal errs by at most ``2 e_F``
            (``||F~^dagger D F~ - F^dagger D F|| <= ||F~^dagger - F^dagger|| + ||F~ - F||``)
            and each of the ``N_s`` steps has one conjugation per variable in
            either product order. Zero for exact QFTs and for one-hot.
        range_omissions: The contributions omitted below the normal binary64
            range and their charges (``QHDRangeOmissions``).
        walsh_phase: The identity-phase allowances of the Walsh diagonal
            blocks (``QHDWalshPhase``) that ``binary.compile_binary_steps``
            returned for a binary Plan with compiled steps, None for every
            other Plan.
        host_state_error: The classical kernel's first-order 2-norm state
            error budget delta (``method._host_state_error``), computed once
            at planning for a classical Plan and None for a native one.
        host_mass_window: The admitted ``|mass - 1|`` of the classical
            kernel's final state, ``state_mass_window`` of delta and ``K**d``
            (``method._host_probability_window``), None for a native Plan.
        host_tie_window: The Plan's ceiling on the classical kernel's
            most-probable tie window, ``_host_window`` of delta
            (``method._host_tie_window``), None for a native Plan.
        width: Register width, ``d*K`` for one-hot and ``d*b`` for binary.
        restricted_dimension: Valid grid size ``K**d``.
        support_evaluations: Grid tuples evaluated across all support tables.
        size_units: Selected host array and action units for classical
            execution, zero otherwise.
        workspace_bytes: Known host workspace for classical execution, zero
            otherwise.
    """

    method_id: ContentID
    expression: Text
    symbolic_variables: tuple[Text, ...]
    support_values: tuple[SupportValues, ...]
    constant: Real
    initial_amplitudes: tuple[FrozenArray, ...]
    initial_state_error: Nonnegative
    steps: tuple[tuple[Annotated[QHDBlock | QHDBinaryBlock, Field(discriminator="kind")], ...], ...]
    step_weights: tuple[tuple[Nonnegative, Nonnegative, Nonnegative], ...]
    compact_schedule_selected: StrictBool
    physical_phase: Real
    phase_sources: tuple[tuple[Text, Real], ...]
    dropped_angle_sum: Nonnegative
    dropped_count: Count
    pruning_error_bound: Nonnegative
    aqft_error_bound: Annotated[Real, Field(ge=0, le=2)]
    range_omissions: QHDRangeOmissions
    walsh_phase: QHDWalshPhase | None = None
    host_state_error: Nonnegative | None = None
    host_mass_window: Nonnegative | None = None
    host_tie_window: Nonnegative | None = None
    width: PositiveInt
    restricted_dimension: PositiveInt
    support_evaluations: Count
    size_units: Count
    workspace_bytes: Count

    @model_validator(mode="after")
    def _one_initial_vector_per_variable(self):
        """Require one stored nonnegative float64 initial vector of K entries per variable.

        The native builder, the classical start vector, the preparation
        charge of the error ledger and the rotation census read that shape.
        """
        d = len(self.symbolic_variables)
        # K**d grid points for either encoding (width d*K one-hot, d*b binary).
        if len(self.initial_amplitudes) != d or any(
            v.array.dtype.str != "<f8" or v.array.ndim != 1 or len(v) ** d != self.restricted_dimension
            or not (v.array >= 0).all()
            for v in self.initial_amplitudes
        ):
            raise ValueError("QHD initial amplitudes need one nonnegative float64 vector of K entries per variable")
        return self

    @model_validator(mode="after")
    def _host_windows_together(self):
        """Require the host state budget and its two windows together or none of them.

        Planning stores all three for a classical Plan and none for a
        native one.
        """
        present = {value is not None for value in (self.host_state_error, self.host_mass_window,
                                                   self.host_tie_window)}
        if len(present) != 1:
            raise ValueError("QHD host state budget, mass window and tie window are stored together")
        return self

    @model_validator(mode="after")
    def _admitted_step_rows(self):
        """Require the step rows that planning's range admission gives every Plan (``schedules.step_weights``).

        Every weight and the first midpoint are positive normal binary64
        numbers (``validation._normal_range``), and the midpoints do not
        decrease. The attribute ``step_weights`` states why the record
        requires no more.
        """
        # The first midpoint dt/2 is normal, and every later one is at least its predecessor.
        earlier = _NORMAL_MIN
        for time, kinetic, potential in self.step_weights:
            if kinetic < _NORMAL_MIN or potential < _NORMAL_MIN or time < earlier:
                raise ValueError("QHD step weights must be positive normal numbers, and the step midpoints "
                                 "positive normal numbers in step order")
            earlier = time
        return self

    @model_validator(mode="after")
    def _range_omissions_match_the_tables(self):
        """Require one omission record per support table and no chains or one per variable, each within its links.

        ``QHDRangeOmissions`` defines its tables in ``support_values`` order
        and its chains per register, and a record of another shape would
        report omissions for tables or registers that the Plan does not have.
        """
        omitted, d = self.range_omissions, len(self.symbolic_variables)
        if (len(omitted.tables) != len(self.support_values) or len(omitted.chains) not in (0, d)
                or any(emitted > links for links, emitted, _charge in omitted.chains)):
            raise ValueError(f"QHD range omissions need one record per support table and none or {d} chains, each "
                             "emitting at most its links")
        return self


class QHDAnalysis(Result):
    """Readout of a QHD run: best observed point, most probable point and probability masses.

    [`solve`][nwqlib.scientist.solve] returns it for a `QHD` method, and `load_result`
    reopens a saved one. The answer is `candidate`, the valid grid point with positive
    observed weight that has the least evaluated objective, and `objective`, the
    objective there from the stored support tables (the objective's monomials grouped
    by the variables they contain, each group tabulated on the grid, plus a constant),
    in the Problem's objective unit.
    `most_probable_coordinates` is the grid point where the evolution concentrated
    probability, which the augmented-Lagrangian and refinement layers read by default.
    The two points can differ. Both are None when no valid outcome was observed, and
    neither establishes a continuous or global optimum. `print(result)` shows both
    points, the mode status and the valid probability. The fields below are read-only.
    The fields of [`Result`][nwqlib.core.analysis.Result] are present too. The guide's
    [readout](../../algorithms/qhd.md#readout) section explains both points and the tie
    window.

    The candidate minimizes the evaluated binary64 objective among valid points with
    positive observed weight, with the smallest grid index on objective ties. For exact
    readout these are positive-probability points, and for counts they are sampled
    points. Exact probability readout does not make objective evaluation exact or
    guarantee that every grid point has positive probability. The most probable point
    describes concentration of the observed distribution under the tie rule of the Mode
    status note below. Its fields are unavailable exactly when the candidate is.

    Attributes:
        value: Objective at the candidate, read from the stored support tables, or None
            when no valid outcome was observed. The same as `objective`.
        candidate_indices: Grid index of the candidate for each variable.
        candidate_coordinates: Candidate grid point in the original coordinates. The
            same as `candidate`.
        candidate_probability: Unconditional probability of the candidate.
        most_probable_indices: Grid index for each variable of the most probable valid
            point. Among the valid points with positive observed probability, it is the
            lexicographically smallest index tuple whose probability lies within
            `most_probable_tie_window` of the computed maximum. Counts compare pooled
            integer counts exactly.
        most_probable_coordinates: That point in the original coordinates.
        most_probable_probability: Its unconditional probability.
        most_probable_objective: Objective at that point, read from the stored support
            tables.
        most_probable_deficit: Computed maximum probability minus
            `most_probable_probability`, at most the tie window. A positive value means
            a point within the window was chosen over the computed maximum. Other points
            can lie within the window even when it is zero, so the selection does not
            identify a unique mode.
        most_probable_tie_window: Derived bound on the error of a computed difference of
            two probabilities on this readout path, 0 for counts, or None when no bound
            was derived. Classical `split_step` evolution derives it from the state
            budget it observed on its trajectory, at most the Plan's window.
        most_probable_tie_window_unavailable: Why no tie window was derived, for example
            a preparation record outside the roundoff derivation. The selection then
            compared the computed probabilities exactly, so roundoff can decide between
            points that the model makes equal.
        probability_maximizer_indices: Lexicographically first valid grid-index tuple
            attaining the largest positive observed weight. Exact readout compares the
            computed probabilities. Counts compare pooled integers before division by
            `returned_shots`. None without a positive valid outcome.
        probability_maximizer_coordinates: Coordinates of the computed probability
            maximizer on the Plan's grid, or None with its indices.
        probability_maximizer_probability: Unconditional computed probability, or
            empirical frequency rounded once from its pooled count and `returned_shots`,
            at the computed probability maximizer. None with its indices.
        probability_maximizer_objective: Evaluated objective from the Plan's support
            tables at the computed probability maximizer. None with its indices.
        mode_status: Resolution of the most-probable selection among positive observed
            valid points, `resolved`, `unresolved` or `unavailable`. With an available
            window, resolved means only one point passes the maximum-to-point tie test,
            and unresolved means another point passes it. Counts use equality of pooled
            integer counts. Unavailable means no positive valid outcome or no derived
            numerical window. This status supplies no sampling-error or optimization
            guarantee.
        expected_objective: Objective mean conditional on a valid outcome.
        marginals: Unconditional probability of each grid index per variable, a
            `FrozenArray` whose `.array` is the read-only float64 array of shape (d, K)
            in variable order and grid order. Each row sums to `valid_mass`. Analysis
            accumulates it from the observed points, and for classical execution reads
            it from the marginals that classical evolution returns.
        valid_mass: Probability of outcomes that encode a grid point, one excitation per
            register for one-hot and every outcome for binary. The same as
            `valid_probability`.
        invalid_mass: Probability of all other register outcomes, zero for binary.
        observed_mass: Total observed probability.
        valid_count: For counts, the number of returned outcomes that encode a grid
            point, summed as integers over all returned count data. None for exact
            readout. With
            positive `returned_shots`, `valid_mass` is `valid_count/returned_shots`
            rounded once.
        returned_shots: For counts, the returned shots, summed as integers, which can be
            fewer than the Plan requested. None for exact readout.
        missing: Reasons the observation is incomplete or yields no valid candidate.
        applications: Records of the classical evolution's numerical call, with its
            arguments and raw values, for classical execution.
        artifact: Saved final state, when `keep_state` requested it.

    Mode status:
        Write O for the valid points with positive observed weight, q_i for the stored
        probability of point i (for counts the exact empirical frequency `N_i/R` of the
        pooled integer count N_i and the integer `returned_shots` R, compared through
        N_i), M for the largest q_i over O, m for the computed probability maximizer
        (the `probability_maximizer_*` fields) and s for the tie representative (the
        `most_probable_*` fields). Under an available window W the representative is the
        lexicographically first point passing the maximum-to-point tie test
        `fl(M - q_i) <= W`, with `fl` the binary64 subtraction, and for counts the
        lexicographically first point with the largest pooled count, W = 0. Without a
        window the stored probabilities are compared exactly, so s = m. `mode_status` is
        `unavailable` when O is empty or W is unavailable, `unresolved` when W is
        available and a point of O other than s passes the same test, and `resolved`
        otherwise, that is when exactly one point passes it. A zero deficit occurs both
        at a well-separated maximum and at a tie, so the status is evaluated from the
        observed weights together with the selection and cannot be reconstructed from
        the deficit.

    Guarantee:
        Suppose W bounds the pairwise errors on the points of O defined in the
        Mode status note, with ``p_i`` the reference probability at point i and
        ``abs((q_i - q_j) - (p_i - p_j)) <= W`` for every compared i, j in O.
        If exactly one
        point passes the test, it is m, hence s = m, and `fl(q_m - q_t) > W` for every
        other t. Rounding is monotone and fixes the stored binary64 W, so an exact
        difference at most W would round to at most W, and the rejection implies the
        exact real difference `q_m - q_t > W`, so `p_m - p_t >= q_m - q_t - W > 0`.
        `resolved` therefore certifies a unique maximum of the stored q and, under the
        stated pairwise error model, a unique maximum of p on O. Exact subtraction,
        including Sterbenz's condition `q_t >= q_m/2`, suffices for this conclusion but
        is not necessary for it. Rounding can make the status conservatively unresolved,
        since a rounded subtraction can accept an exact gap slightly above W, and it
        cannot make it falsely resolved under the pairwise assumption. For counts, a
        single largest integer N_i gives a unique maximum of the exact empirical
        frequencies N_i/R, and a count tie is unresolved. Neither result is a confidence
        statement about an underlying distribution. `unresolved` says that this window
        does not separate the representative from every competitor. It does not prove
        equal probabilities, multiple exact modes, a large numerical error or an
        inferior objective. `resolved` says nothing about optimization quality, a
        continuous minimizer, physical-model discrepancy, hardware bias or finite-shot
        uncertainty. The population is O. If an exact readout completely represents a
        valid grid distribution, an omitted known-zero q_i cannot exceed the positive
        maximum, so m is also a computed maximizer on the full valid grid, but a
        certificate about p on O extends to omitted points only when their pairwise
        error bounds apply and `fl(M - 0) > W`, namely M > W. An incomplete observation
        does not make omitted probabilities known zero.
    """

    value: Real | None
    candidate_indices: tuple[Count, ...] | None
    candidate_coordinates: tuple[Real, ...] | None
    candidate_probability: Nonnegative | None
    most_probable_indices: tuple[Count, ...] | None
    most_probable_coordinates: tuple[Real, ...] | None
    most_probable_probability: Nonnegative | None
    most_probable_objective: Real | None
    most_probable_deficit: Nonnegative | None
    most_probable_tie_window: Nonnegative | None
    most_probable_tie_window_unavailable: Text | None
    probability_maximizer_indices: tuple[Count, ...] | None
    probability_maximizer_coordinates: tuple[Real, ...] | None
    probability_maximizer_probability: Nonnegative | None
    probability_maximizer_objective: Real | None
    mode_status: ModeStatus
    expected_objective: Real | None
    marginals: FrozenArray
    valid_mass: Nonnegative
    invalid_mass: Nonnegative
    observed_mass: Nonnegative
    valid_count: Count | None
    returned_shots: Count | None
    missing: tuple[Text, ...]
    applications: tuple[KernelApplication, ...] = ()
    artifact: ArtifactManifest | None = None

    def _summary_lines(self):
        """Return the display lines from the stored fields.

        The concentration lines give the computed probability maximizer, the
        tie representative with its deficit and window, and the mode status,
        shown also when the deficit is zero, because a zero deficit occurs
        both at a separated maximum and at a tie. Without a positive valid
        outcome one line replaces both points. The status line of counts
        names the empirical frequencies it compares. The unavailable status
        gives the stored reason: why no tie window was derived, when the
        computed probabilities were then compared exactly, or the missing
        population. Exact readout shows the concentration lines before the
        candidate and counts after it.
        """
        mode = self._mode_lines()
        exact_readout = self._plan is not None and self._plan.shots is None
        if self.candidate_coordinates is None:
            candidate = "Candidate: unavailable (no positive valid outcome)"
        else:
            if self._plan is None:
                meaning = "least evaluated objective among observed valid points"
            elif exact_readout:
                meaning = ("grid minimum over positive-probability points in the observed population, "
                           "using evaluated binary64 objectives")
            else:
                meaning = "least evaluated objective among observed valid samples"
            candidate = (
                f"Candidate ({meaning}): {self._values_text(self.candidate_coordinates)}; objective: "
                f"{self._scalar_text(self.value)}{self._unit_text()}"
            )
        lines = [*mode, candidate] if exact_readout else [candidate, *mode]
        lines.append(f"Valid probability: {self._scalar_text(self.valid_mass)}; "
                     "best observed finite-grid candidate, not a global optimum")
        if self.missing:
            lines.append(f"Partial data: {len(self.missing)} missing populations")
        return lines

    def _mode_lines(self):
        """Return the maximizer, tie-representative and mode-status lines of ``_summary_lines``."""
        if self.probability_maximizer_indices is None:
            points = ["Computed probability maximizer and tie representative: unavailable (no positive valid "
                      "outcome)."]
        else:
            points = [
                f"Computed probability maximizer: {self._values_text(self.probability_maximizer_coordinates)}. "
                f"Probability: {self._scalar_text(self.probability_maximizer_probability)}.",
                f"Tie representative: {self._values_text(self.most_probable_coordinates)}. "
                f"Probability: {self._scalar_text(self.most_probable_probability)}. "
                f"Deficit: {self._scalar_text(self.most_probable_deficit)}. "
                f"Window: {self._scalar_text(self.most_probable_tie_window)}.",
            ]
        if self.mode_status == "unavailable":
            reason = (self.most_probable_tie_window_unavailable if self.probability_maximizer_indices is not None
                      else "; ".join(self.missing))
            status = f"Mode status: unavailable ({reason})."
        elif self.returned_shots is not None:
            status = f"Empirical mode status: {self.mode_status}. Counts give no proof of the population mode."
        elif self.mode_status == "resolved":
            status = "Mode status: resolved among positive observed valid points."
        else:
            status = "Mode status: unresolved. Another positive observed valid point passes the tie test."
        return [*points, status]

    @property
    def candidate(self):
        """Candidate point in the original coordinates, the same as ``candidate_coordinates``."""
        return self.candidate_coordinates

    @property
    def objective(self):
        """Original objective at the candidate, the same as ``value``."""
        return self.value

    @property
    def valid_probability(self):
        """Unconditional probability of the outcomes that encode a grid point, the same as ``valid_mass``."""
        return self.valid_mass

    @property
    def position_mean(self):
        """Mean grid coordinate of each variable conditional on a valid outcome, or None without valid mass.

        ``<x_j> = sum_i x_j(i) m_j(i) / valid_mass``, with m_j the stored unconditional
        marginal of variable j and x_j(i) the Plan's grid coordinates. Liu et al.
        arXiv:2607.16996v1 Eq. (94) define the mean over the whole final state. Dividing by
        ``valid_mass`` conditions it on a valid outcome, NWQLib's choice, because an invalid
        one-hot outcome encodes no position. It is computed on demand from the marginals and
        the attached Plan's grid only, with no evolution, decoding, objective evaluation or
        measurement, and is not stored. The mean and ``position_standard_deviation``
        describe the observed distribution. They are no criterion for agreement of two
        states or for optimization success, because two distributions can share mean and
        variance at total-variation distance 1. For example, {-1, +1} with probability 1/2
        each and {-2, 0, +2} with {1/8, 3/4, 1/8} both have mean 0 and variance 1, and put
        probability 0 and 3/4 on the point 0. On the periodic grid the coordinates are those
        of the chart ``[lower, upper)``, so both moments depend on where the period is cut.
        Mass split between ``x_0`` and ``x_(K-1)``, which are neighbors through the wrap
        link, gives a mean near the middle of the box, where it may have no mass.
        """
        moments = self._position_moments()
        return None if moments is None else moments[0]

    @property
    def position_standard_deviation(self):
        """Standard deviation of each variable's grid coordinate conditional on a valid outcome, or None.

        ``sigma_j = sqrt(sum_i (x_j(i) - <x_j>)**2 m_j(i) / valid_mass)`` with ``<x_j>``
        from ``position_mean``, conditioned on a valid outcome as the mean is. Liu et al.
        arXiv:2607.16996v1 Eq. (94) define the standard deviation over the whole final
        state. The deviations are scaled by their largest magnitude before squaring, so
        coordinates far from the origin do not overflow. None when ``valid_mass`` is zero.
        """
        moments = self._position_moments()
        return None if moments is None else moments[1]

    def _position_moments(self):
        """Return ``(means, standard_deviations)`` from the marginals and the Plan's grid, or None."""
        from math import fsum, sqrt
        from .method import _grid

        if self.valid_mass == 0:
            return None
        grid = _grid(self.plan)
        means, deviations = [], []
        for j, row in enumerate(self.marginals.array.tolist()):
            coordinates = [grid.grid_value(j, i) for i in range(len(row))]
            # <x_j> = sum_i x_j(i) m_j(i) / valid_mass
            mean = fsum(x * m for x, m in zip(coordinates, row, strict=True)) / self.valid_mass
            # sigma_j**2 = s**2 sum_i ((x_j(i) - <x_j>)/s)**2 m_j(i) / valid_mass
            # with s the largest |x_j(i) - <x_j>|, so no square overflows.
            spread = max(abs(x - mean) for x in coordinates)
            variance = 0.0 if spread == 0 else fsum(
                ((x - mean) / spread) ** 2 * m for x, m in zip(coordinates, row, strict=True)
            ) / self.valid_mass
            means.append(mean)
            deviations.append(spread * sqrt(variance))
        return tuple(means), tuple(deviations)

    @staticmethod
    def pooled_summation_roundoff(chunks):
        """First-order roundoff of pooling the bin values of ``chunks`` into QHD's masses.

        Analysis divides each of the M stored bin values by the chunk count (one rounding),
        adds it into running sums (at most M additions per value) and forms the valid mass
        with one ``fsum`` (one more rounding), so each mass moves by at most ``(M + 2)*u``,
        with ``u = 2**-53``. Each chunk's own total lies within the window of its
        preparation record, which bounds the simulator and readout roundoff. This allowance
        widens the mass check only. The tie window of the most probable point has its own
        pooling term ``gamma_C = C u/(1 - C u)`` for the C roundings of each point's pooled
        probability.

        M counts the stored entries of each probabilities or counts chunk, which include
        stored zero values, so M is not the number of nonzero values. Another chunk adds its
        number of stored values.
        """
        from nwqlib._validation import UNIT_ROUNDOFF

        entries = sum(chunk.histogram().entries if chunk.observation.kind in ("probabilities", "counts")
                      else len(chunk.values) for chunk in chunks)
        return (entries + 2) * UNIT_ROUNDOFF

    def validate_analysis_masses(self, data):
        """Bound the observed masses by the roundoff window of the executions that produced them.

        Analysis calls this on the masses it forms, including masses decoded
        from a stored state, which carry no scalar for the chunk-receipt
        check. Masses decoded from a saved statevector use the receipt's
        ``saved_state_probability_window``, which includes the host
        correction of the saved state. Direct probability and count chunks
        keep ``probability_window``. An unassessed saved-state correction
        gives no finite upper-mass budget, so the masses are then checked
        for nonnegativity only (``validate_normalized_mass`` with None).
        """
        from nwqlib._validation import NUMERICAL_RELATION_RTOL, validate_normalized_mass

        receipts = {receipt.content_id: receipt for receipt in data.receipts}
        windows = [receipts[chunk.prepared_id].saved_state_probability_window
                   if chunk.observation.kind == "amplitudes" else receipts[chunk.prepared_id].probability_window
                   for chunk in data.observations.chunks if chunk.prepared_id in receipts]
        if None in windows:
            for mass in (self.valid_mass, self.invalid_mass, self.observed_mass):
                validate_normalized_mass(mass, None)
            return
        # A host receipt counts no instructions and keeps the fixed floor. The
        # classical kernel records its derived expm_multiply window in its
        # application (method._application) instead.
        windows.extend(binding.value.value for chunk in data.observations.chunks
                       for application in chunk.applications for binding in application.arguments
                       if binding.parameter == "probability_window")
        window = max(windows, default=NUMERICAL_RELATION_RTOL)
        window += self.pooled_summation_roundoff(data.observations.chunks)
        for mass in (self.valid_mass, self.invalid_mass, self.observed_mass):
            validate_normalized_mass(mass, window)

    def number_sector_width(self):
        """Return the register width ``d*K`` once the result is known to hold complete-register counts.

        The shared number-sector check reads every register bit of every
        count, so it applies only to native execution with a counts readout,
        and only to the one-hot encoding, whose valid sector has one
        excitation per register. Every binary outcome is a grid point.
        """
        plan = self.plan
        self.validate_plan(plan)
        if plan.method.encoding != "one_hot":
            raise ValueError("the QHD number-sector check applies to the one-hot encoding. Every binary "
                             "outcome encodes a grid point")
        if (
            plan.execution != "quantum"
            or not plan.experiments
            or plan.resolve("qhd").resolved_observation(plan)[1].kind != "counts"
        ):
            raise ValueError("QHD number-sector check requires selected complete-register counts")
        return plan.reconstruction.width

    def number_sector_observations(self):
        """Return the analyzed count chunks after the same checks that analysis applies.

        No objective is evaluated and no state is evolved.
        """
        from .method import _admit_observations

        self.number_sector_width()
        _admit_observations(self.plan, self.data.observations, result=self)
        return self.data.observations.chunks

    @model_validator(mode="after")
    def _lineage(self):
        """Require candidate, marginals and valid/invalid masses to describe one consistent
        observed grid population.
        """
        from math import fsum, isclose
        from nwqlib._validation import NUMERICAL_RELATION_RTOL as tolerance

        if not isclose(
            self.valid_mass + self.invalid_mass,
            self.observed_mass,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError("QHD valid/invalid masses must partition observed mass")
        marginals = self.marginals.array
        if marginals.dtype.str != "<f8" or marginals.ndim != 2 or not len(marginals) or marginals.shape[1] < 2:
            raise ValueError("QHD marginals require a nonempty rectangular float64 grid with K >= 2")
        if not (marginals >= 0).all():
            raise ValueError("QHD marginals are nonnegative probabilities")
        if any(
            not isclose(fsum(row), self.valid_mass, rel_tol=tolerance, abs_tol=tolerance)
            for row in marginals
        ):
            raise ValueError("each QHD marginal must sum to the unconditional valid mass")
        available = self.value is not None
        if any(
            (value is not None) != available
            for value in (
                self.candidate_indices,
                self.candidate_coordinates,
                self.candidate_probability,
                self.expected_objective,
            )
        ):
            raise ValueError("QHD candidate fields require joint availability")
        if available and (
            self.valid_mass <= 0
            or not 0 < self.candidate_probability <= self.valid_mass
            or len(self.candidate_indices) != len(marginals)
            or len(self.candidate_coordinates) != len(marginals)
            or any(
                i >= len(row) for i, row in zip(self.candidate_indices, marginals, strict=True)
            )
        ):
            raise ValueError("QHD candidate must belong to the observed valid population")
        if available and (
            self.expected_objective < self.value
            or any(
                self.candidate_probability > row[i]
                for i, row in zip(self.candidate_indices, marginals, strict=True)
            )
        ):
            raise ValueError(
                "QHD candidate probability and mean violate the observed population domain"
            )
        if not available and self.valid_mass != 0:
            raise ValueError("a nonempty valid QHD population requires its observed candidate")
        # The most probable point shares the candidate's population. Its
        # probability is copied from the same points that the marginals sum,
        # and rounded addition of nonnegative values is monotone
        # (fl(x + y) >= x for x, y >= 0), so every marginal entry of the point
        # is at least its probability with no allowance. The candidate order
        # uses the subtraction predicate of the selection. The candidate
        # probability is at most the computed maximum M, and rounded
        # subtraction is monotone, so p_c - p_mode <= M - p_mode <= window.
        # Counts divide integers by one shot total, which is monotone too.
        mode = (self.most_probable_indices, self.most_probable_coordinates,
                self.most_probable_probability, self.most_probable_objective, self.most_probable_deficit)
        if any((value is not None) != available for value in mode):
            raise ValueError("QHD most-probable fields require joint availability with the candidate")
        window, reason = self.most_probable_tie_window, self.most_probable_tie_window_unavailable
        if (window is not None or reason is not None) != available or (
            window is not None and reason is not None
        ):
            raise ValueError("QHD most-probable selection records its tie window or why it is unavailable")
        # The candidate minimizes the stored objective over the positive
        # population that contains the most probable point, and both values
        # come from the same table lookup (method.objective_at), so the
        # comparison is exact.
        if available and self.most_probable_objective < self.value:
            raise ValueError("QHD most probable objective lies below the candidate's, which minimizes over the "
                             "same observed population")
        if available:
            probability, tie = self.most_probable_probability, 0.0 if window is None else window
            if (
                len(self.most_probable_indices) != len(marginals)
                or len(self.most_probable_coordinates) != len(marginals)
                or any(i >= len(row) for i, row in zip(self.most_probable_indices, marginals, strict=True))
                or not 0 < probability <= self.valid_mass
            ):
                raise ValueError("QHD most probable point must belong to the observed valid population")
            if (
                self.most_probable_deficit > tie
                or any(probability > row[i] for i, row in zip(self.most_probable_indices, marginals, strict=True))
                or (self.candidate_probability > probability and self.candidate_probability - probability > tie)
            ):
                raise ValueError("QHD most probable probability violates its window or observed population")
        # The computed probability maximizer m shares that population and has
        # the largest stored weight M, the lexicographically first index on
        # exact ties. Its marginal entries bound its probability as they bound
        # the representative's, M is at least the candidate's and the
        # representative's probability, and its table objective is at least
        # the candidate's. The representative's deficit is fl(M - p_s) of the
        # same two stored values, and for counts both are the largest pooled
        # count over R, so the deficit is 0. A binary64 difference of distinct
        # values is never zero, so the nonnegative deficit also gives M >= p_s.
        # m passes the tie test and the representative s is the first point
        # that passes it, so s <= m, with
        # s = m when the status is resolved, without a window (exact
        # comparison) and for counts (W = 0 admits equal integers only).
        maximizer = (self.probability_maximizer_indices, self.probability_maximizer_coordinates,
                     self.probability_maximizer_probability, self.probability_maximizer_objective)
        if any((value is not None) != available for value in maximizer):
            raise ValueError("QHD probability-maximizer fields require joint availability with the candidate")
        if (self.mode_status == "unavailable") != (not available or window is None):
            raise ValueError("QHD mode status is unavailable exactly without a positive valid outcome or a tie "
                             "window")
        if available:
            top, indices = self.probability_maximizer_probability, self.probability_maximizer_indices
            if (
                len(indices) != len(marginals)
                or len(self.probability_maximizer_coordinates) != len(marginals)
                or any(i >= len(row) for i, row in zip(indices, marginals, strict=True))
                or not 0 < top <= self.valid_mass
                or any(top > row[i] for i, row in zip(indices, marginals, strict=True))
            ):
                raise ValueError("QHD probability maximizer must belong to the observed valid population")
            if (
                top < self.candidate_probability
                or self.probability_maximizer_objective < self.value
                or self.most_probable_deficit != top - self.most_probable_probability
            ):
                raise ValueError("QHD probability maximizer must hold the largest observed weight, from which the "
                                 "representative's deficit is computed")
            if self.most_probable_indices > indices or indices != self.most_probable_indices and (
                self.mode_status != "unresolved" or self.returned_shots is not None
            ):
                raise ValueError("QHD tie representative is the first point that passes the tie test, and one "
                                 "other than the probability maximizer requires an unresolved status")
        if self.artifact is not None and self.artifact.plan_id != self.plan_id:
            raise ValueError("QHD result state artifact differs from its Plan")
        return self

    def validate_plan(self, plan):
        """Check the marginal shape and the coordinates and objectives of the candidate, the most probable point
        and the probability maximizer against the Plan.
        """
        from .method import _grid, objective_at

        self._validate_common_plan(plan, Plan)
        if self.construction_id != plan.construction.content_id:
            raise ValueError("QHD aggregate result differs from its base Plan construction")
        if (self.artifact is not None and self.artifact.construction_id
                != plan.resolve("qhd").selected_construction(plan).content_id):
            raise ValueError("QHD result state artifact differs from its selected construction")
        grid = _grid(plan)
        if self.marginals.array.shape != (grid.num_variables, grid.num_grid_points):
            raise ValueError("QHD analysis marginal shape differs from its selected grid")
        for name, indices, coordinates, value in (
            ("candidate", self.candidate_indices, self.candidate_coordinates, self.value),
            ("most-probable", self.most_probable_indices, self.most_probable_coordinates,
             self.most_probable_objective),
            ("probability-maximizer", self.probability_maximizer_indices, self.probability_maximizer_coordinates,
             self.probability_maximizer_objective),
        ):
            if indices is None:
                continue
            if coordinates != tuple(grid.grid_value(j, i) for j, i in enumerate(indices)):
                raise ValueError(f"QHD {name} coordinates differ from selected grid indices")
            if value != objective_at(plan.reconstruction, indices, grid.num_grid_points):
                raise ValueError(f"QHD {name} objective differs from its admitted support table")


DESCRIPTOR = AlgorithmDescriptor(
    method=METHOD.name,
    version=METHOD.version,
    problem_families=("optimization",),
    output_families=("optimization_candidate",),
    access_families=("admitted_sympy_expression",),
    resource_coverage=("compact selected native schedule", "restricted sparse THEORY size",
                       "split-step transform THEORY size"),
    evidence_coverage=("observed grid-point population, candidate and most probable point",),
    limitations=(
        "box-only one-hot Dirichlet or periodic, or binary periodic, discretization",
        "one-hot periodic grid requires an even K of at least 4, the binary encoding K = 2**b",
        "best observed candidate is not a global optimum",
        "one selected THEORY evolution; no automatic companion or grid minimum",
        "native exact probabilities contain 2**(d*K) one-hot or K**d binary entries; THEORY stores K**d amplitudes",
    ),
    references=(METHOD,),
    maintenance="NWQLib",
)


class QHDVerification(Record):
    """Options of an explicit QHD check: grid minimum, or fidelity against a reference evolution.

    Build it with keyword arguments, for example
    `QHDVerification(comparisons=("grid_minimum",))`, and pass it to
    `result.verify(checks=...)`, which returns `(receipt, facts)`: a
    `VerificationReceipt` that records the check, its raw values and the numerical calls
    it made, and a tuple of `FramedFact` records that cite it. `comparisons` is the only
    required argument. `solve` never runs these checks. Replaying the original classical
    evolution checks consistency, not independent accuracy, and the receipt names what
    produced the result and the reference it compares. The guide's
    [explicit verification](../../algorithms/qhd.md#explicit-verification) section
    describes what each comparison checks and what it does not establish. The default
    tolerances are untuned, and the
    [engineering constants](../../ENGINEERING_CONSTANTS.md#explicit-workflow-and-reference-controls)
    register them.

    Attributes:
        comparisons: Required. Distinct checks to run, a nonempty tuple of
            `"grid_minimum"`, `"schrodinger_fidelity"` and `"ir_product_fidelity"`.
            `grid_minimum` evaluates the original objective on every grid point. Each
            fidelity choice runs one restricted evolution against the stored state.
        objective_gap_tolerance: Default `1e-12`. Pass threshold, in objective units,
            for the candidate's evaluated gap to the least reevaluated binary64 value.
        schrodinger_infidelity_tolerance: Default `0.1`, between 0 and 1. Pass threshold
            for the infidelity against the restricted Schrodinger evolution.
        ir_product_infidelity_tolerance: Default `1e-9`, between 0 and 1. Pass threshold
            for the computed infidelity against the numerical reference of the compiled
            product. Binary phase reconstruction and reference-evolution error are not
            propagated into this tolerance decision.
        minimum_tolerance: Default `0.0`. Window around the least reevaluated binary64
            objective value that `minimum_success_mass` uses. Zero counts ties in the
            evaluated table. It counts exactly the mathematical minimizers when the
            evaluations are exact and this tolerance is zero.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Positive limit on
            the byte allowances charged for this check, separate from the QHD
            Method's limits. For observed counts or probabilities without a kept
            state, the `grid_minimum` charge omits histogram construction or
            loading, decoding and passing-weight storage at every readout width.
            This limit therefore does not bound the `minimum_success_mass` step.
        max_work: Default `1_000_000_000`. Positive limit on the known work of this
            check, separate from the QHD Method's limits.

    Raises:
        ValueError: If `comparisons` is empty or repeats a choice.
    """

    comparisons: tuple[Literal["grid_minimum", "schrodinger_fidelity", "ir_product_fidelity"], ...]
    # Untuned windows of explicit comparisons, in objective units for the gap
    # and the minimizer window and dimensionless for infidelity. The gap and
    # infidelity thresholds are the thresholds of the verification checks and
    # of the receipt's ``within_tolerance`` facts, and minimum_tolerance=0
    # counts ties at the least reevaluated binary64 value in ``minimum_success_mass``. Registered in
    # docs/ENGINEERING_CONSTANTS.md ("QHDVerification tolerances"). Revisit
    # with the intended objective scale, state and schedule.
    objective_gap_tolerance: Nonnegative = 1e-12
    schrodinger_infidelity_tolerance: Annotated[Real, Field(ge=0, le=1)] = 0.1
    ir_product_infidelity_tolerance: Annotated[Real, Field(ge=0, le=1)] = 1e-9
    minimum_tolerance: Nonnegative = 0.0
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_work: PositiveInt = 1_000_000_000

    @model_validator(mode="after")
    def _choices(self):
        if not self.comparisons or len(set(self.comparisons)) != len(self.comparisons):
            raise ValueError("verification requires distinct selected comparisons")
        return self

    @property
    def source(self):
        """Source record of these comparisons. Each receipt adds its producer relations."""
        return Source(
            name="qhd.explicit_reference",
            version="3",
            domain="selected finite-grid/state comparison",
            reference="original-objective evaluation or explicit restricted evolution; independence depends on the original producer; not a global optimum certificate",
        )

    def verification_checks(self, result):
        """Return one CheckSpec per selected comparison, as ``verification.verification_checks`` defines."""
        from .verification import verification_checks

        return verification_checks(self, result)
