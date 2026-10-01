"""SDK-free execution declarations and explicit prepared/run entry points."""

from nwqlib._limits import DEFAULT_MAX_BYTES, DEFAULT_MAX_DIRECT_AMPLITUDES

from collections.abc import Mapping
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import AwareDatetime, Field, PrivateAttr, StrictBool, StrictInt, model_validator

from nwqlib._validation import (
    AER_STATEVECTOR_ROUNDOFF_PER_OPERATION,
    MAX_COUNT,
    NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION,
    exact_probability_window,
    native_state_error,
    validate_normalized_mass,
)
from nwqlib._phase_product import corrected_mass_window, corrected_state_error
from nwqlib.core.planning import MAX_RUNTIME_SEED, ObservationSpec, Realization, RuntimeOptions, reducer_outputs
from nwqlib.core.records import ContentID, Nonnegative, PositiveInt, Real, Record, Source, Text
from nwqlib.artifacts import ArtifactManifest, ReadoutArray, UnavailableOutput, probability_readout
from nwqlib.evidence.error_model import FramedFact
from nwqlib.ir import Binding
from nwqlib.operators.access import Count
from nwqlib.problems.inputs import PhysicalScale

if TYPE_CHECKING:
    from nwqlib._prepared_execution import PreparedHandle, Run, prepare_experiment, submit_experiment, submit_detached, refresh_submissions
    from nwqlib._remote_preparation import refresh_preparation

TimeScope = Literal["selected_acquisition", "acquisition_overhead", "native_call_wall"]

__all__ = ["ExecutionMode"]


class ExecutionMode(str, Enum):
    """Supported execution modes for algorithm runs.

    The execution mode is part of the public result metadata so reports can
    distinguish exact simulation, measured execution, provider estimates and
    non-circuit theory paths.
    """

    STATEVECTOR = "statevector"
    SHOTS = "shots"
    ESTIMATED = "estimated"
    THEORY = "theory"


class RegisterMap(Record):
    """Declared register name and least-significant-first native bit positions.

    Attributes:
        name: Register name in the associated selected/native layout.
        bits: Ordered global bit positions belonging to that register.
    """

    name: Text
    bits: tuple[Count, ...]


class LogicalPreparationReceipt(Record):
    """Logical-lowering facts for the parent's selected construction, stored without replay.

    These are the counts that lowering already measured, kept so that inspection
    and reopening never lower again. dynamic_visits counts admitted dynamic IR
    visits, not native gates. reserved_work is the admitted dynamic/definition
    size law, not the outer run reservation or measured CPU/RSS.
    defined_selections are actually built
    native definition templates, including cached bases, not specialization or
    operation counts. Exact call arguments remain in the selected Realization.
    """

    dynamic_visits: Count
    reserved_work: Count
    defined_selections: tuple[ContentID, ...]


class CountsSampling(Record):
    """Actual producer semantics, separate from an assumed statistical model.

    fixed_seed resets the producer's sampling stream to seed on every call.
    fresh obtains a new physical acquisition; its job coordinates identify the
    data, without establishing IID or stationarity. unknown supplies neither
    premise. RuntimeOptions.seed can select compilation alone on hardware.
    """

    kind: Literal["fixed_seed", "fresh", "unknown"]
    seed: Annotated[Count, Field(le=MAX_RUNTIME_SEED)] | None = None

    @model_validator(mode="after")
    def _seed_role(self):
        if (self.kind == "fixed_seed") != (self.seed is not None):
            raise ValueError("only fixed-seed sampling has an effective sampling seed")
        return self


# Derived per-instruction roundoff constant of each exact statevector target,
# keyed by PreparedArtifact.target.name (_validation.py derives both).
_ROUNDOFF_PER_OPERATION = {
    "aer_statevector": AER_STATEVECTOR_ROUNDOFF_PER_OPERATION,
    "nwqsim_cpu_sv": NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION,
}


class PreparedArtifact(Record):
    """Portable receipt for a factory-owned native snapshot, not executable bytes.

    snapshot names a fresh in-process owned circuit; it is not a verified file
    digest. Target/compiler sources are actual preparation-time snapshots.
    The exact small Realization survives failed attempts and journal restoration;
    no completed observation or default-point inference is needed to recover it.
    Inspection of this record is allocation-free with respect to native data.

    Attributes:
        execution: quantum_circuit or host_kernel preparation kind.
        plan_id: Original Plan identity.
        realization_id: Exact concrete experiment realization identity.
        realization: Concrete selected arguments and experiment, kept even before any result exists.
        construction_id: Selected construction used for this preparation.
        snapshot: Factory-owned native snapshot identity, not a payload digest.
        target: Actual target identity at preparation time.
        compiler: Actual compiler/preparation source.
        runtime: Effective runtime seed and native preparation options.
        observation: Exact selected readout semantics.
        quantum_layout: Selected logical quantum register layout.
        classical_layout: Selected logical classical register layout.
        native_basis: Actual native gate vocabulary after preparation.
        environment: Actual preparation-process dependency versions.
        preparation_time: Recorded preparation timestamp.
        construction_work_reserved: Counted quantum construction work charged before execution.
        logical: Optional actual logical-lowering receipt; None for a host-only preparation.
        native_operations: Prepared operation count, or None when unavailable.
            For a coherent statevector readout on Aer, a trajectory included,
            it counts the native evolution operations that execute, including
            every executed forward and inverse view operation, and no save
            instruction: a save evaluates or copies the current state and
            inserts no state error into the continuing state. It is not
            multiplied by the number of saved labels or points, and one
            whole-trajectory count gives a conservative window for every
            point. On Aer, a body with control flow and sampled counts count
            every instruction of the prepared native circuit. Other adapters
            state their own count.
        host_preparation_work_reserved: Counted host setup work charged by this preparation.
        selected_kernel_id: Exact host declaration identity, or None for native circuit execution.
        transformation: Description of the actual selected-to-native transformation.
        population: unconditional, native_conditioned or unknown measurement-population meaning.
        payload: Optional native-byte locator, not the bytes themselves.
        native_quantum_layout: Actual compiled quantum register layout.
        native_classical_layout: Actual compiled classical register layout.
        logical_to_native: Ordered logical-to-native wire map.
        backend_configuration_id: Exact connection configuration identity, or None for host execution.
        counts_sampling: Actual count-source sampling contract, or None for non-counts readouts.
        provider_options_json: Saved effective provider settings without credentials.
        probability_window_exclusions: Labels of the native operations, readout path,
            simulator version or non-default compiler optimization level whose effect
            the derivation of ``probability_window`` does not bound. An empty tuple means the whole execution lies inside it.
            None means the preparing backend did not assess it.
        body_length: Number of bound logical operations of the selected body, the
            largest trajectory boundary, recorded for a trajectory; None otherwise.
        boundaries: Resolved boundary of each trajectory point in declaration
            order, with the end-position shorthand resolved before native
            preparation; empty for other readouts.
        template_seed: Seed of the per-construction native template that the
            adapter prepared this circuit from, as the preparation hook
            supplied it (``_prepared_execution._template_seed``); None when the
            adapter used no template.
        statevector_roundoff: The ``(t, e)`` envelope of the host phase
            correction of this execution's saved statevectors
            (``_phase_product.phase_product_envelope``), the componentwise
            maxima over the receipt's save keys, charged to every statevector
            consumer the receipt covers (``saved_state_error`` and
            ``saved_state_probability_window``). The pair ``(t, e)`` bounds
            the output phase product on saved statevectors. ``(0.0, 0.0)``
            denotes an assessed zero-error operation, including no product or
            multiplication by the represented factor ``(1.0, ±0.0)`` under
            the host arithmetic model. ``None`` means the host correction was
            not assessed.
    """

    schema_version: Literal[8] = 8
    execution: Literal["quantum_circuit", "host_kernel"] = "quantum_circuit"
    plan_id: ContentID
    realization_id: ContentID
    realization: Realization
    construction_id: ContentID
    snapshot: Text
    target: Source
    compiler: Source
    runtime: RuntimeOptions
    observation: ObservationSpec
    quantum_layout: tuple[RegisterMap, ...]
    classical_layout: tuple[RegisterMap, ...]
    native_basis: tuple[Text, ...]
    environment: tuple[Source, ...]
    preparation_time: Text
    construction_work_reserved: Count
    logical: LogicalPreparationReceipt | None
    native_operations: Count | None
    host_preparation_work_reserved: Count = 0
    selected_kernel_id: ContentID | None = None
    transformation: Text
    population: Literal["unconditional", "native_conditioned", "unknown"]
    payload: "PayloadRef | None" = None
    native_quantum_layout: tuple[RegisterMap, ...] = ()
    native_classical_layout: tuple[RegisterMap, ...] = ()
    logical_to_native: tuple[Count, ...] = ()
    backend_configuration_id: ContentID | None = None
    counts_sampling: CountsSampling | None = None
    provider_options_json: Text | None = None
    probability_window_exclusions: tuple[Text, ...] | None = None
    body_length: Count | None = None
    boundaries: tuple[Count, ...] = ()
    template_seed: Count | None = None
    statevector_roundoff: tuple[Nonnegative, Nonnegative] | None = None

    @property
    def probability_window(self) -> float:
        """Admitted binary64 roundoff between one and an exact probability population of this
        execution. It follows the native operation count, the width and the executing
        simulator's per-instruction constant, see ``exact_probability_window``. Host and
        unknown counts keep the fixed floor."""
        per_operation = _ROUNDOFF_PER_OPERATION.get(self.target.name, AER_STATEVECTOR_ROUNDOFF_PER_OPERATION)
        return exact_probability_window(self.native_operations, len(self.logical_to_native),
                                        per_operation=per_operation)

    def state_error(self, resolved=()):
        """Return ``(delta, None)`` for this execution's final-state budget, or ``(None, reason)``.

        delta is ``native_state_error`` of the native operation count with the
        target's derived per-instruction constant, a first-order bound on
        ``||psi_hat - exp(i phi) psi||_2`` against the unitary native circuit.
        It describes distance modulo one common global phase. It does not
        bound the exact prefix phase, the accumulation of the phase ledger
        (``backends.qiskit_aer._trajectory_phase_ledger``), the subtraction
        forming its correction angle, or a phase-defined amplitude; a saved
        statevector's consumers use ``saved_state_error``. It exists only
        when the derivation covers the whole execution, that is, for a
        circuit on a target with a derived constant, a known operation count
        and assessed ``probability_window_exclusions`` that are all in
        ``resolved``, the labels whose effect the caller's own readout bounds.
        Unlike ``probability_window``, an unknown count or target gets no
        default constant and no floor. The budget enters the bound on
        differences of point probabilities
        (``_validation.probability_difference_window``, used by
        ``algorithms.qhd.method._readout_window`` for the most-probable tie
        window). The Aer constant that ``probability_window`` takes for an
        unknown target has no derivation for that target, and the
        ``NUMERICAL_RELATION_RTOL`` floor that it keeps for an unknown count
        is an input convention for a population total, not a derived bound, so
        neither can stand in for delta.
        """
        per_operation = _ROUNDOFF_PER_OPERATION.get(self.target.name)
        if self.execution != "quantum_circuit" or per_operation is None:
            return None, f"no derived per-instruction roundoff for target {self.target.name}"
        if self.native_operations is None:
            return None, "unknown native operation count"
        if self.probability_window_exclusions is None:
            return None, "roundoff exclusions were not assessed"
        unresolved = sorted(set(self.probability_window_exclusions) - set(resolved))
        if unresolved:
            return None, "outside the roundoff derivation: " + ", ".join(unresolved)
        return native_state_error(self.native_operations, per_operation), None

    def saved_state_error(self, resolved=()):
        """Return ``(delta_saved, reason)``: ``state_error(resolved)`` after the host correction of a saved state.

        ``delta_saved = delta + t (1 + delta) + e`` with ``(t, e)`` the
        receipt's ``statevector_roundoff``
        (``_phase_product.corrected_state_error``). Like ``state_error`` it
        is a distance modulo one common global phase, conditional on the
        native first-order model. It returns the native unavailable reason
        unchanged when delta is unavailable, and an unavailable reason for
        the missing premise when the host correction was not assessed. Every
        consumer of a saved statevector uses it; direct native probability
        and Pauli saves keep ``state_error``, since their buffers undergo no
        host correction.
        """
        if self.statevector_roundoff is None:
            return None, "saved-state host correction was not assessed"
        delta, reason = self.state_error(resolved)
        return corrected_state_error(delta, self.statevector_roundoff), reason

    @property
    def saved_state_probability_window(self):
        """``probability_window`` propagated through the host correction of a saved state, or None.

        ``omega_saved = omega + (2t + t**2)(1 + omega) + 2(1 + t)(1 + omega) e
        + e**2`` with ``(t, e)`` the receipt's ``statevector_roundoff``
        (``_phase_product.corrected_mass_window``). An unassessed correction
        gives None, not the uncorrected window, so it yields no finite
        mass-check budget.
        """
        if self.statevector_roundoff is None:
            return None
        return corrected_mass_window(self.probability_window, self.statevector_roundoff)

    @model_validator(mode="after")
    def _execution_kind(self):
        """Keep host and quantum receipts disjoint so neither can fabricate the other's facts.

        A host receipt names its selected kernel and carries no lowering, layout,
        native basis, payload, backend or population other than unconditional. A
        quantum receipt carries its logical lowering receipt and no host fields.
        Counts receipts must state their actual sampling semantics.
        """
        if (self.observation.kind == "counts") != (self.counts_sampling is not None):
            raise ValueError("counts preparations require their actual sampling semantics")
        if self.realization.plan_id != self.plan_id or self.realization.content_id != self.realization_id:
            raise ValueError("prepared receipt must keep its exact Plan and Realization")
        if self.execution == "host_kernel":
            if (self.selected_kernel_id is None or self.logical is not None or self.native_operations is not None
                    or self.quantum_layout or self.classical_layout or self.native_basis
                    or self.construction_work_reserved or self.observation.kind != "host_scalars"
                    or self.population != "unconditional" or self.payload is not None
                    or self.native_quantum_layout or self.native_classical_layout or self.logical_to_native
                    or self.backend_configuration_id is not None or self.provider_options_json is not None
                    or self.probability_window_exclusions is not None or self.template_seed is not None):
                raise ValueError("host preparation cannot fabricate quantum lowering or population")
        elif (self.logical is None or self.selected_kernel_id is not None
              or self.host_preparation_work_reserved or self.observation.kind == "host_scalars"):
            raise ValueError("quantum preparation requires its native logical receipt")
        if self.observation.kind == "trajectory":
            if self.body_length is None or self.boundaries != self.observation.boundaries(self.body_length):
                raise ValueError("a trajectory receipt records its body length and every resolved point boundary")
        elif self.body_length is not None or self.boundaries:
            raise ValueError("only a trajectory receipt records point boundaries")
        return self


class PayloadRef(Record):
    """Local payload key, format and byte count; no content authentication.

    key locates one saved input or an already identified result artifact.
    Loading these metadata does not load a provider SDK.
    """

    format: Text
    key: Text
    bytes: Count


class PauliValue(Record):
    """One exact Pauli expectation. Its unit bound depends on the executed circuit's
    roundoff and is checked with the preparation receipt, see
    ObservationChunk.validate_unit_bound."""

    kind: Literal["pauli"] = "pauli"
    label: Text
    value: Real


class CountBin(Record):
    """One raw outcome count. In bits, classical bit 0 is the rightmost character."""

    kind: Literal["count"] = "count"
    bits: Text
    count: Count


class EstimateValue(Record):
    """A finite signed provider estimate of one original weighted observable.

    estimate_id binds the original coefficients and requested precision. The
    reported standard_error is statistical uncertainty with the provider's
    stated meaning; it is not total physical error or a certified bound.
    provider_metadata_json preserves returned primitive/PUB settings and usage;
    requested_options_json records the options actually passed to the primitive.
    Unreported effective settings, extra shots and billing remain unknown.
    """

    kind: Literal["estimate"] = "estimate"
    estimate_id: ContentID
    value: Real
    standard_error: Nonnegative | None
    uncertainty_unavailable: Text | None
    uncertainty_source: Source
    requested_options_json: Text
    provider_metadata_json: Text

    @model_validator(mode="after")
    def _uncertainty(self):
        if (self.standard_error is None) != (self.uncertainty_unavailable is not None):
            raise ValueError("provider estimate requires reported standard error or its explicit unavailable reason")
        return self


class ProbabilityArrays(Record):
    """The stored form of one probability marginal: array manifests and scalar summaries.

    Probability readouts store their numerical values in binary arrays. The
    observation chunk contains the array manifests, outcome layout and scalar
    population summaries. Dense probabilities use outcome positions as
    indices: ``probabilities`` has shape ``(D,)``, ``D = 2**width``, and every
    outcome exists by position, including exact zeros. Sparse probabilities
    use sorted unique integer indices: ``indices`` has shape ``(s,)`` for a
    width of at most 64, or ``(s, W)`` packed words with ``W =
    ceil(width/64)``, word zero least significant, and ``probabilities`` holds
    the ``s`` strictly positive values. The first selected probability qubit is
    least significant in the outcome index. The arrays are published with the
    chunk's acquisition (``artifacts.ReadoutArray``); the summaries are
    computed once at publication from the bytes that are hashed
    (``artifacts.probability_readout``), which also checks the values, so
    loading checks this layout and the manifests' associations and rereads no
    value. The stored-entry count D of a dense marginal and its nonzero count
    s are different quantities. The unit endpoint compares ``mass`` with one
    plus the receipt's window (``ObservationChunk.validate_unit_bound``).

    Attributes:
        encoding: dense or sparse.
        width: Number of observed qubits.
        entries: Stored entries: ``2**width`` dense, ``s`` sparse.
        nonzero: Number of nonzero stored values.
        mass: ``math.fsum`` of the stored values.
        maximum: Largest stored value, zero when nothing is stored.
        probabilities: Manifest of the float64 values.
        indices: Manifest of the uint64 indices of a sparse marginal, else None.
    """

    kind: Literal["probability_arrays"] = "probability_arrays"
    encoding: Literal["dense", "sparse"]
    width: PositiveInt
    entries: Count
    nonzero: Count
    mass: Nonnegative
    maximum: Nonnegative
    probabilities: ArtifactManifest
    indices: ArtifactManifest | None = None

    @model_validator(mode="after")
    def _layout(self):
        """Require the manifests' components and shapes that the encoding, width and entry counts imply."""
        words = -(-self.width // 64)
        values = self.probabilities.output
        indices = None if self.indices is None else self.indices.output
        dense = self.encoding == "dense"
        if (type(values) is not ReadoutArray or values.component != "probabilities"
                or values.shape != (self.entries,) or self.nonzero > self.entries
                or (dense and (indices is not None or self.entries != 1 << self.width))
                or (not dense and (type(indices) is not ReadoutArray or indices.component != "indices"
                                   or indices.shape != ((self.entries,) if words == 1 else (self.entries, words))
                                   or self.nonzero != self.entries))):
            raise ValueError("probability arrays differ from their encoding, width and entry counts")
        return self


class ScalarValue(Record):
    """A selected scalar statistic with explicit frame and availability.

    The selected host/amplitude declaration and method own its mathematical
    domain; encoded_branch may denote a selected normalized quantum mass or
    an unnormalized host branch norm, according to that declaration.
    """

    kind: Literal["scalar"] = "scalar"
    label: Text
    value: Real | None
    frame: Literal["physical", "unit", "encoded_branch"] = "physical"
    unavailable: Text | None = None

    @model_validator(mode="after")
    def _availability(self):
        if (self.value is None) != (self.unavailable is not None):
            raise ValueError("unrepresentable scalar requires an explicit unavailable reason")
        return self


class KernelApplication(Record):
    """One selected numerical application and its bounded scalar evidence.

    arguments keep method-owned parameters, including selected schedule
    counts. Facts keep their scientific frames and missing-value evidence.
    Names and source are declarations; neither can load an implementation.
    """

    name: Text
    implementation: Source
    arguments: tuple[Binding, ...]
    facts: tuple[FramedFact, ...] = ()

    @model_validator(mode="after")
    def _arguments(self):
        if len({item.parameter for item in self.arguments}) != len(self.arguments):
            raise ValueError("application argument names must be distinct")
        return self


def verification_invocation():
    """Identify one explicitly performed verification; loading never calls this."""
    return str(uuid4())


class VerificationReceipt(Record):
    """An explicit verification's raw facts, numerical calls and admitted work.

    artifact_ids identify the live target arrays actually read. Application
    facts have declared evidence; returned check facts may cite this completed
    receipt. Application facts state the actual selected numerical work.
    Receipt absence cannot establish whether
    a reference ran. invocation_id distinguishes actual producing calls; None
    means manual or otherwise unknown invocation provenance. Load never creates
    an invocation or proves that two calls were statistically independent.
    """

    schema_version: Literal[2] = 2
    plan_id: ContentID
    invocation_id: Text | None = None
    result_id: ContentID
    construction_id: ContentID
    artifact_ids: tuple[ContentID, ...]
    reference: Source
    options_id: ContentID
    applications: tuple[KernelApplication, ...]

    @model_validator(mode="after")
    def _lineage(self):
        """Require distinct targets and application names, and application facts with declared or no evidence.

        A fact witnessed by this receipt can exist only after the receipt does, so
        an application fact inside it may carry declared evidence or none.
        """
        if len(set(self.artifact_ids)) != len(self.artifact_ids):
            raise ValueError("verification target artifacts must be distinct")
        if len({item.name for item in self.applications}) != len(self.applications):
            raise ValueError("verification application names must be distinct")
        if any(fact.fact.evidence is not None and fact.fact.evidence.status != "declared"
               for application in self.applications for fact in application.facts):
            raise ValueError("raw verification facts must precede their witnessed receipt")
        return self


class ReducedValues(Record):
    """One output component of an acquisition-time reduction, flattened in C order.

    A float64 component stores ``real``, a complex128 component ``real`` and
    ``imaginary``, and an int64 component ``integers``. The component's
    dtype and shape are those its reducer registers
    (``core.planning.reducer_outputs``), which the point chunk checks.
    """

    kind: Literal["reduced"] = "reduced"
    component: Count
    real: tuple[Real, ...] = ()
    imaginary: tuple[Real, ...] = ()
    integers: tuple[StrictInt, ...] = ()


Statistic = Annotated[PauliValue | CountBin | ProbabilityArrays | ScalarValue | EstimateValue | ReducedValues,
                      Field(discriminator="kind")]

# MAX_COUNT (imported from _validation, which defines the count domain) is the
# largest count and requested shot number an ObservationChunk admits, the
# int64 maximum: ObservationChunk.histogram returns counts as int64 weights.


def _probability_mapping(outcomes, width):
    """The ``(values, indices)`` arrays of a mapping from bit-string key to probability.

    Keys have ``width`` characters of 0 and 1, the rightmost character bit 0
    (the first observed qubit); values are floats or integers, stored as
    float64. Indices are 1-D uint64 for a width of at most 64, else
    ``(entries, W)`` packed words, word 0 least significant.
    ``artifacts.probability_readout`` checks the values, sorts the indices and
    chooses the stored encoding.
    """
    keys, weights = list(outcomes), list(outcomes.values())
    if any(type(key) is not str or len(key) != width or set(key) - set("01") for key in keys):
        raise ValueError("histogram keys must match the declared little-endian readout width")
    if any(type(weight) is bool or not isinstance(weight, (int, float)) for weight in weights):
        raise ValueError("probabilities are real numbers")
    words = max(1, -(-width // 64))
    if words == 1:
        indices = np.array([int(key, 2) for key in keys], dtype=np.uint64)
    else:
        mask = (1 << 64) - 1
        indices = np.array([[(int(key, 2) >> (64 * word)) & mask for word in range(words)] for key in keys],
                           dtype=np.uint64).reshape(-1, words)
    return np.array(weights, dtype=np.float64), indices


class Histogram:
    """Outcome indices and weights of one probabilities or counts chunk.

    ``ObservationChunk.histogram()`` returns it. The layout below is the
    reader contract that consumers and later storage build on. All arrays are
    read-only.

    - ``width``: the readout width w, the number of observed qubits
      (probabilities) or of classical bits over the whole classical layout
      (counts).
    - ``entries``: the number of stored entries, counting any stored zero
      weight, so it is not the number of nonzero weights.
    - ``weights``: a 1-D array with one weight per entry, float64 for
      probabilities and int64 for counts.
    - ``indices()``: for w <= 64, a 1-D uint64 array with one outcome index
      per entry. Bit 0 of an index is the first observed qubit
      (probabilities) or classical bit 0 (counts), which is the rightmost
      character of the chunk's bit-string key. For w > 64 it raises
      ValueError naming the width.
    - ``packed_indices()``: for every w, a 2-D uint64 array of shape
      ``(entries, words)`` with ``words = max(1, ceil(w/64))``. Word 0 holds
      the least significant 64 bits and the padding bits above w are zero.
    - ``index_list()``: the outcome indices as Python integers, for every w.

    Entries keep the order in which the chunk stores them; nothing is sorted.
    A probability chunk stores a dense marginal by outcome position, every
    outcome including exact zeros, or a sparse marginal with strictly
    increasing indices of its nonzero values (``ProbabilityArrays``), so its
    entries are in increasing index order; a dense marginal's indices are
    formed when they are first requested. For a histogram obtained from a
    validated count chunk, every weight and the exact total are in
    ``[0, 2**63 - 1]``, and the total equals ``returned_shots`` and does not
    exceed the requested shots. Because the weights are nonnegative, an int64
    sum of this single chunk's weights cannot overflow. Totals combined across
    acquisitions require an overflow-safe accumulation. An empty chunk gives
    ``indices()`` of shape ``(0,)`` and ``packed_indices()`` of shape
    ``(0, words)``.

    Attributes:
        width: Readout width w.
        entries: Number of stored entries.
        weights: Read-only 1-D float64 probabilities or int64 counts.
    """

    __slots__ = ("width", "_entries", "_weights", "_packed", "_source")

    def __init__(self, width, integers, weights):
        words = max(1, -(-width // 64))
        if words == 1:
            packed = np.array(integers, dtype=np.uint64).reshape(-1, 1)
        else:
            mask = (1 << 64) - 1
            packed = np.array([[(value >> (64 * word)) & mask for word in range(words)] for value in integers],
                              dtype=np.uint64).reshape(-1, words)
        # Views of read-only owners cannot be made writeable again, and the
        # object refuses assignment, so a Histogram that a chunk keeps and
        # hands to every reader stays unchanged. The reshape above returns a
        # view, so its owner is marked too.
        if weights.base is not None:
            weights = weights.copy()
        for array in (packed.base, packed, weights):
            if array is not None:
                array.flags.writeable = False
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "_entries", len(weights))
        object.__setattr__(self, "_weights", weights.view())
        object.__setattr__(self, "_packed", packed.view())
        object.__setattr__(self, "_source", None)

    @classmethod
    def _from_arrays(cls, width, indices, weights):
        """The Histogram of stored readout arrays, as views without a copy.

        ``indices`` is None for a dense marginal, a 1-D uint64 array, or the
        ``(entries, words)`` packed words. The arrays are marked read-only
        before their views are taken, so the views cannot be made writeable.
        """
        histogram = object.__new__(cls)
        object.__setattr__(histogram, "width", width)
        object.__setattr__(histogram, "_entries", len(weights))
        object.__setattr__(histogram, "_source", None)
        histogram._wrap(indices, weights)
        return histogram

    @classmethod
    def _from_source(cls, width, entries, source):
        """A Histogram whose arrays ``source()`` returns, as ``(indices, weights)``, on first access.

        ``width`` and ``entries`` come from the chunk's summaries, so reading
        them reads no payload and forms no index.
        """
        histogram = object.__new__(cls)
        object.__setattr__(histogram, "width", width)
        object.__setattr__(histogram, "_entries", entries)
        object.__setattr__(histogram, "_weights", None)
        object.__setattr__(histogram, "_packed", None)
        object.__setattr__(histogram, "_source", source)
        return histogram

    def _wrap(self, indices, weights):
        for array in (indices, weights):
            if array is not None and array.flags.writeable:
                array.flags.writeable = False
        packed = None if indices is None else indices.reshape(-1, 1) if indices.ndim == 1 else indices.view()
        object.__setattr__(self, "_weights", weights.view())
        object.__setattr__(self, "_packed", packed)

    def _load(self):
        """Read the arrays of a Histogram built from a payload source, once."""
        if self._source is not None:
            self._wrap(*self._source())
            object.__setattr__(self, "_source", None)

    def __setattr__(self, name, value):
        raise AttributeError("a Histogram is read-only")

    @property
    def entries(self):
        return self._entries

    @property
    def weights(self):
        self._load()
        return self._weights

    def indices(self):
        """Return the 1-D uint64 outcome indices; the width must be at most 64."""
        if self.width > 64:
            raise ValueError(f"readout width {self.width} exceeds one 64-bit index; use packed_indices()")
        return self.packed_indices()[:, 0]

    def packed_indices(self):
        """Return the ``(entries, words)`` uint64 outcome indices, word 0 least significant."""
        self._load()
        if self._packed is None:
            # A dense marginal stores outcome j at position j.
            packed = np.arange(self._entries, dtype=np.uint64).reshape(-1, 1)
            packed.base.flags.writeable = False
            object.__setattr__(self, "_packed", packed.view())
        return self._packed

    def index_list(self):
        """Return the outcome indices as Python integers, for every width."""
        self._load()
        if self._packed is None:
            return list(range(self._entries))
        if self._packed.shape[1] == 1:
            return self._packed[:, 0].tolist()
        return [sum(word << (64 * position) for position, word in enumerate(row)) for row in self._packed.tolist()]


class ObservationChunk(Record):
    """One immutable acquisition's sufficient statistics, never per-shot copies.

    Distinct attempt/chunk identities stay distinct even with equal values.
    Identity is provenance, not evidence that random acquisitions are independent.

    Probabilities and counts are read through ``histogram()``, whose layout
    ``Histogram`` states, and built with ``from_histogram``. Counts are
    stored as JSON ``CountBin`` records. Every count, the exact total and the
    requested shots are at most ``MAX_COUNT = 2**63 - 1``, the int64 maximum
    of the reader's count weights, and a chunk outside that domain is
    rejected. A probability marginal stores its values in binary arrays: the
    chunk holds one ``ProbabilityArrays`` record with the array manifests and
    the scalar summaries computed at publication. The arrays live in the
    artifact store of the Run or Result that holds the chunk. A chunk keeps a
    private reference to that store, which is not a field, and publication,
    reopening a Run (``load_run``), loading a Result (``load_result``) and
    the chunk's own ``revise`` and copies set or carry it; ``histogram()``
    reads the arrays through it on first use, once per payload for the
    store's lifetime. A chunk rebuilt from JSON by any other path has no such
    reference, and its ``histogram()`` refuses with the loader to use.

    A trajectory acquisition stores one chunk per observation point. The
    chunk names its ``point`` and resolved ``boundary``, so a saved value maps
    to its body, point and receipt after reload. A trajectory stores one full
    declaration in its preparation receipt and one row per point, with only
    that point's readout declaration and the shared declaration identity on
    the row. Receipt validation and point indexing visit the schedule once,
    each chunk validates its own declaration and values, and ``readout()``
    returns the stored one-point spec. Storage and validation are linear in
    the total admitted declarations and payloads, with no per-point copy or
    rescan of the full schedule. The one-point spec is
    ``ObservationSpec.point_observation`` of the point, and
    ``trajectory_id`` is the content identity of the full declaration. The
    join with the receipt (``validate_unit_bound`` and ``Run._write``)
    checks both. Point chunks share their actual
    acquisition's completion, preparation, shot and native-work population:
    the chunks of one acquisition share its run, attempt and job, and differ
    in ``chunk`` (``point_chunk_key``). A missing point makes its dependent
    quantity incomplete. It never triggers a replay of the shared prefix in
    analysis or loading.

    Attributes:
        run_id: Original acquisition Run identity.
        execution: quantum_circuit or host_kernel contribution kind.
        plan_id: Exact selected Plan identity.
        realization_id: Concrete selected experiment arguments.
        prepared_id: Actual prepared artifact used by the acquisition.
        experiment: Original experiment name.
        setting: Actual readout or batch setting.
        bindings: Original ordered experiment argument values.
        quantum_layout: Selected quantum register order for interpreting coordinates.
        classical_layout: Selected classical register order for interpreting count labels.
        attempt: Identity of the original reserved acquisition attempt.
        job: Original acquisition/job identity, not a new reduction ID.
        chunk: Original item/slot within that acquisition.
        observation: Selected readout declaration against which returned
            statistics are checked; for a trajectory point, that point's
            one-point readout.
        population: Actual unconditional, native_conditioned or unknown population.
        returned_shots: Returned raw count total, or None when no raw-shot total is available.
        trajectories: Actual trajectory count, or None when unavailable/not applicable.
        values: Sufficient statistics in the original label and physical-frame
            association: for probabilities one ``ProbabilityArrays`` record,
            for counts one ``CountBin`` per stored outcome. Probabilities and
            counts are read with ``histogram()``.
        source: Actual backend/kernel provenance.
        selected_kernel_id: Host declaration identity, or None for quantum contributions.
        physical_scale: Positive factor that restores physical magnitude to
            this chunk's unit-normalized data, or None. Its meaning belongs to
            the Method. For LCHS and the QLS ``qsvt_inverse`` solver it is the
            norm of the output vector, and analysis multiplies the unit
            solution direction by it. For Expectation, Lanczos and QPE it is
            the norm of the supplied input state. Expectation's physical
            quadratic form uses it, and the eigenvalue Methods record it with
            the input, since an energy does not depend on the state's norm.
            QHD, FixedGCIM and ADAPT host chunks carry None, because their
            scalars need no recovery, a QLS shortcut chunk carries None
            because the shortcut recovers no physical norm, and quantum
            circuit chunks other than amplitude readout carry None.
        physical_scale_unavailable: Why ``physical_scale`` is None, because the
            scale is unavailable or no recovery applies. Host and amplitude
            chunks set exactly one of the two fields.
        artifacts: Manifests of arrays from this same acquisition.
        applications: Actual host operator/application receipts.
        unavailable: Named outputs that could not be produced.
        point: ID of the trajectory point whose values this chunk holds, or
            None for a readout without a point schedule.
        boundary: Resolved boundary of that point in the selected body, or None.
        trajectory_id: Content identity of the trajectory declaration that
            the point belongs to, held by the preparation receipt, or None.
    """

    schema_version: Literal[3] = 3
    run_id: Text
    execution: Literal["quantum_circuit", "host_kernel"] = "quantum_circuit"
    plan_id: ContentID
    realization_id: ContentID
    prepared_id: ContentID
    experiment: Text
    setting: Text
    bindings: tuple[Binding, ...]
    quantum_layout: tuple[RegisterMap, ...]
    classical_layout: tuple[RegisterMap, ...]
    attempt: Text
    job: Text
    chunk: Text
    observation: ObservationSpec
    population: Literal["unconditional", "native_conditioned", "unknown"]
    returned_shots: Count | None
    trajectories: Count | None
    values: tuple[Statistic, ...]
    source: Source
    selected_kernel_id: ContentID | None = None
    physical_scale: PhysicalScale | None = None
    physical_scale_unavailable: Text | None = None
    artifacts: tuple[ArtifactManifest, ...] = ()
    applications: tuple[KernelApplication, ...] = ()
    unavailable: tuple[UnavailableOutput, ...] = ()
    point: Text | None = None
    boundary: Count | None = None
    trajectory_id: ContentID | None = None
    __slots__ = ("_histogram", "_store", "__weakref__")

    def readout(self):
        """The readout whose statistics this chunk holds, its ``observation``.

        For a trajectory point chunk that is the point's one-point readout,
        which the chunk stores, so nothing is rebuilt.
        """
        return self.observation

    def _carry_payload(self, other):
        """Give ``other`` this chunk's payload source and Histogram when it holds the same values."""
        if other is self or other.values != self.values:
            return other
        for name in ("_histogram", "_store"):
            try:
                value = object.__getattribute__(self, name)
            except AttributeError:
                continue
            object.__setattr__(other, name, value)
        return other

    def revise(self, **changes):
        """Validate a new revision; one with the same values keeps this chunk's payload source."""
        return self._carry_payload(super().revise(**changes))

    def model_copy(self, *, update=None, deep=False):
        """Copy or revise; a copy with the same values keeps this chunk's payload source."""
        return self._carry_payload(super().model_copy(update=update, deep=deep))

    def __copy__(self):
        return self._carry_payload(super().__copy__())

    def __deepcopy__(self, memo=None):
        # The store is shared, not copied: an artifact store refuses copying.
        return self._carry_payload(super().__deepcopy__(memo))

    def __getstate__(self):
        """Pickle a probability chunk with its stored arrays, which an unpickled chunk reads from memory."""
        state = super().__getstate__()
        payload = self._payload()
        if payload:
            state = dict(state, nwqlib_probability_arrays=tuple(array for _, array in payload))
        return state

    def __setstate__(self, state):
        arrays = state.get("nwqlib_probability_arrays")
        super().__setstate__({key: value for key, value in state.items() if key != "nwqlib_probability_arrays"})
        if arrays is not None:
            indices, values = (None, *arrays) if len(arrays) == 1 else arrays
            object.__setattr__(self, "_histogram", Histogram._from_arrays(self.values[0].width, indices, values))

    def _manifests(self):
        """The manifests of every array this chunk names: its ``artifacts`` and its probability arrays."""
        if self.readout().kind != "probabilities":
            return self.artifacts
        arrays = self.values[0]
        return self.artifacts + tuple(item for item in (arrays.indices, arrays.probabilities) if item is not None)

    def _holds_payload(self):
        """Whether this probability chunk holds its arrays in memory or a store to read them from."""
        return self.readout().kind == "probabilities" and any(
            hasattr(self, name) for name in ("_histogram", "_store"))

    def _payload(self):
        """The ``(manifest, array)`` pairs of this probability chunk's arrays, empty when it holds none.

        The arrays are the stored ones, read from memory or once through the
        chunk's store; a saved Result writes them beside its other arrays.
        """
        if not self._holds_payload():
            return ()
        arrays, histogram = self.values[0], self.histogram()
        pairs = ((arrays.probabilities, histogram.weights),)
        if arrays.indices is not None:
            packed = histogram.packed_indices()
            pairs = ((arrays.indices, packed[:, 0] if len(arrays.indices.output.shape) == 1 else packed),) + pairs
        return pairs

    def _attach_store(self, store):
        """Record ``store`` as the payload source of a probability chunk rebuilt from stored JSON.

        Run reopening and Result loading call this for each chunk they
        rebuild; the arrays are read only when ``histogram()`` is first called.
        """
        if self.readout().kind == "probabilities":
            object.__setattr__(self, "_store", store)
        return self

    @classmethod
    def _from_probability_arrays(cls, values, indices=None, *, store=None, **fields):
        """Build a probability chunk from its arrays, publishing them to ``store`` when given.

        The arrays are checked and their stored encoding chosen by
        ``artifacts.probability_readout``, which also computes the summaries
        from the bytes that are hashed. The caller hands the arrays over: a
        published array is kept without a copy when it owns its data
        (``ArtifactStore._publish`` with ``private=True``). Without a store the
        manifests carry the digest of the same canonical bytes and the chunk
        keeps the arrays in memory. Either way the chunk keeps its Histogram,
        so reading it right after publication reads no store.
        """
        from nwqlib.artifacts import ArtifactManifest as Manifest
        observation = fields["observation"]
        if not isinstance(observation, ObservationSpec):
            observation = fields["observation"] = ObservationSpec.model_validate(observation)
        width = len(observation.qubits)
        indices, values, summary = probability_readout(values, indices, width=width)
        provenance = dict(plan_id=fields.get("plan_id"), realization_id=fields.get("realization_id"),
                          construction_id=fields.get("prepared_id"), producer_id=observation.content_id,
                          acquisition=tuple(fields.get(name) for name in ("run_id", "attempt", "job", "chunk")),
                          source=fields.get("source"))
        manifests, arrays = {}, {}
        for component, array in (("indices", indices), ("probabilities", values)):
            if array is None:
                continue
            output = ReadoutArray(component=component, dtype="uint64" if component == "indices" else "float64",
                                  shape=array.shape)
            if store is not None:
                handle = store._publish(array, output=output, provenance=provenance, private=True)
                manifests[component], arrays[component] = handle.manifest, handle._array
            else:
                array.flags.writeable = False
                digest = "sha256:" + sha256(memoryview(array).cast("B")).hexdigest()
                manifests[component] = Manifest(output=output, digest=digest, data_bytes=output.data_bytes,
                                                encoding=output.encoding, **provenance)
                arrays[component] = array
        del indices, values
        record = ProbabilityArrays(width=width, probabilities=manifests["probabilities"],
                                   indices=manifests.get("indices"), **summary)
        chunk = cls(values=(record,), **fields)
        object.__setattr__(chunk, "_histogram", Histogram._from_arrays(width, arrays.get("indices"),
                                                                       arrays["probabilities"]))
        if store is not None:
            object.__setattr__(chunk, "_store", store)
        return chunk

    @classmethod
    def from_histogram(cls, outcomes, /, **fields):
        """Build a probabilities or counts chunk from its outcomes.

        Validation is that of every other construction path, including the
        ``MAX_COUNT`` count domain. A probability chunk keeps its arrays in
        memory, with manifests that carry the digest of their bytes; its
        stored encoding (dense or sparse) and its canonical sorted order are
        those of ``artifacts.probability_readout``.

        Args:
            outcomes: Either a mapping from bit-string key to weight, with keys
                spelled as the chunk stores them (rightmost character bit 0,
                no register separators), or a pair ``(indices, weights)`` in the
                layout of ``Histogram``: 1-D indices for a width of at most 64
                or ``(entries, words)`` packed indices for every width, integer
                counts or float64 probabilities. Empty arrays of shape ``(0,)``
                or ``(0, words)`` describe an empty chunk whatever their dtype.
            **fields: The other ObservationChunk fields except ``values``.
                ``observation`` selects counts or probabilities. The width is
                the number of its qubits (probabilities) or of the bits of
                ``classical_layout`` (counts).

        Raises:
            TypeError: When ``values`` is also supplied.
            ValueError: For another readout kind, indices outside the width or
                in another layout, weights of another type or length, or any
                check of the chunk itself.
        """
        if "values" in fields:
            raise TypeError("from_histogram builds values from its outcomes")
        observation = fields.get("observation")
        if not isinstance(observation, ObservationSpec):
            observation = fields["observation"] = ObservationSpec.model_validate(observation)
        counts = observation.kind == "counts"
        if not counts and observation.kind != "probabilities":
            raise ValueError("from_histogram builds probabilities or counts chunks")
        layout = tuple(item if isinstance(item, RegisterMap) else RegisterMap.model_validate(item)
                       for item in fields.get("classical_layout", ()))
        fields["classical_layout"] = layout
        width = sum(len(item.bits) for item in layout) if counts else len(observation.qubits)
        words = max(1, -(-width // 64))
        if isinstance(outcomes, Mapping):
            if counts:
                return cls(values=tuple(CountBin(bits=key, count=weight) for key, weight in outcomes.items()),
                           **fields)
            return cls._from_probability_arrays(*_probability_mapping(outcomes, width), **fields)
        indices, weights = (np.asarray(item) for item in outcomes)
        if indices.size == 0 and weights.shape == (0,) and indices.shape in ((0,), (0, words)):
            # An empty array has no values whose type could matter, for example np.asarray([]).
            indices, weights = np.zeros((0, words), dtype=np.uint64), np.zeros(0, dtype=np.int64 if counts else np.float64)
            if words == 1:
                indices = indices[:, 0]
        if (indices.dtype.kind not in "iu" or indices.ndim not in (1, 2) or (indices.ndim == 1 and width > 64)
                or (indices.ndim == 2 and indices.shape[1] != words) or (indices < 0).any()):
            raise ValueError(f"indices of a width-{width} readout must be 1-D (width at most 64) "
                             f"or (entries, {words}) packed nonnegative integers")
        if weights.ndim != 1 or len(weights) != len(indices) or (
                weights.dtype.kind not in "iu" if counts else weights.dtype != np.float64):
            raise ValueError("weights must be one integer count or float64 probability per index")
        if not counts:
            # probability_readout checks the width, order and duplicates.
            return cls._from_probability_arrays(np.array(weights, dtype=np.float64),
                                                np.array(indices[:, 0] if indices.ndim == 2 and words == 1 else indices,
                                                         dtype=np.uint64), **fields)
        integers = (indices.tolist() if indices.ndim == 1 else
                    [sum(int(word) << (64 * position) for position, word in enumerate(row)) for row in indices.tolist()])
        if any(value >> width for value in integers):
            raise ValueError(f"an outcome index exceeds the readout width {width}")
        return cls(values=tuple(CountBin(bits=format(index, f"0{width}b") if width else "", count=weight)
                                for index, weight in zip(integers, weights.tolist(), strict=True)), **fields)

    def histogram(self):
        """Return the outcomes of this probabilities or counts chunk in the ``Histogram`` layout.

        The chunk builds its Histogram on the first call and returns the same
        read-only object afterwards. The slot that holds it is not a field, so
        equality, identity and serialization never see it. A probability chunk
        rebuilt from stored JSON returns a Histogram whose ``width`` and
        ``entries`` come from its summaries; its arrays are read through the
        artifact store of its Run or Result on the first access to the weights
        or indices (``ArtifactStore.get``: one hydration per payload for the
        store's lifetime, with no value or summary recheck).

        Raises:
            ValueError: For any other readout kind, or for a probability chunk
                that has neither its arrays nor a store to read them from.
        """
        try:
            return object.__getattribute__(self, "_histogram")
        except AttributeError:
            pass
        observation = self.readout()
        counts = observation.kind == "counts"
        if not counts and observation.kind != "probabilities":
            raise ValueError(f"a {observation.kind} observation has no histogram")
        if counts:
            weights = np.array([item.count for item in self.values], dtype=np.int64)
            histogram = Histogram(self._readout_width(), [int(item.bits, 2) if item.bits else 0
                                                          for item in self.values], weights)
        else:
            try:
                store = object.__getattribute__(self, "_store")
            except AttributeError:
                raise ValueError("this probability chunk was rebuilt without its payload source; read it from the "
                                 "Run or Result that holds it, reopened with nwqlib.load_run or "
                                 "nwqlib.load_result") from None
            arrays = self.values[0]
            # The width and entry count come from the summaries; the arrays
            # are read on the first access to weights or indices.
            histogram = Histogram._from_source(arrays.width, arrays.entries, lambda: (
                None if arrays.indices is None else store.get(arrays.indices).array,
                store.get(arrays.probabilities).array))
        object.__setattr__(self, "_histogram", histogram)
        return histogram

    def _readout_width(self):
        """Classical width of counts, or the number of observed qubits of probabilities."""
        observation = self.readout()
        if observation.kind == "counts":
            return sum(len(item.bits) for item in self.classical_layout)
        return len(observation.qubits)

    @property
    def acquisition_key(self):
        """``(run_id, attempt, job, chunk)``, the identity of one acquisition's data.

        A Run stores, deduplicates and joins observations under this key. Two
        chunks with equal values but different keys are different acquisitions,
        except that the point chunks of one trajectory acquisition share its
        run, attempt and job and differ only in ``chunk``.
        """
        return self.run_id, self.attempt, self.job, self.chunk

    @model_validator(mode="after")
    def _statistics(self):
        """Admit readout values, populations and artifact provenance for the declared execution
        kind.
        """
        # A reduction point's one-point readout is the singleton trajectory of
        # that entire point (ObservationSpec.point_readout_fields); a chunk
        # can carry no other trajectory declaration. The canonical
        # comparison walks one constant-field wrapper and its own point, not
        # the parent schedule. A point declared with position None is
        # resolved against the receipt (validate_unit_bound). Source: the
        # NWQLib reduction-declaration derivation (its local relation block).
        has_point = self.point is not None
        if (has_point != (self.boundary is not None)
                or has_point != (self.trajectory_id is not None)):
            raise ValueError("a point chunk requires its point, boundary, and trajectory identity together")

        observation = self.readout()
        reduction_point = None
        if observation.kind == "trajectory":
            if not has_point or len(observation.positions) != 1:
                raise ValueError("a chunk can carry only its singleton reduction declaration")
            candidate = observation.positions[0]
            if candidate.kind != "reduction" or candidate.id != self.point:
                raise ValueError("a trajectory wrapper must declare this chunk's reduction point")
            expected_fields = ObservationSpec.point_readout_fields(candidate)
            if any(getattr(observation, name) != value for name, value in expected_fields.items()):
                raise ValueError("a reduction chunk requires the canonical one-point declaration")
            reduction_point = candidate
        reduction = reduction_point is not None

        if has_point and observation.kind in {"counts", "host_scalars", "estimated_observable"}:
            raise ValueError("this readout kind is not a trajectory point")
        if has_point and (self.execution != "quantum_circuit"
                          or self.population != "unconditional"
                          or observation.population != "unconditional"
                          or observation.shots != 0
                          or self.returned_shots is not None or self.trajectories != 1):
            raise ValueError("a trajectory point describes one unconditional exact acquisition")

        declared_position = reduction_point.position if reduction else observation.position
        if has_point and declared_position is not None and declared_position != self.boundary:
            raise ValueError("chunk boundary differs from its declared trajectory point")
        if self.execution == "host_kernel":
            if (observation.kind != "host_scalars" or self.selected_kernel_id is None
                    or (self.physical_scale is None) != (self.physical_scale_unavailable is not None)
                    or self.returned_shots is not None or self.trajectories is not None
                    or self.quantum_layout or self.classical_layout or self.population != "unconditional"):
                raise ValueError("host observation cannot fabricate quantum populations")
            if (any(type(item) is not ScalarValue for item in self.values)
                    or tuple(item.label for item in self.values) != observation.labels):
                raise ValueError("host statistics must match the exact selected ordered names")
            if len({a.output.name for a in self.artifacts + self.unavailable}) != len(self.artifacts) + len(self.unavailable):
                raise ValueError("duplicate produced/unavailable host output name")
            if any(a.acquisition != self.acquisition_key or a.plan_id != self.plan_id
                   or a.realization_id != self.realization_id or a.producer_id != self.selected_kernel_id
                   for a in self.artifacts):
                raise ValueError("artifact provenance differs from this exact acquisition")
            return self
        if observation.kind == "host_scalars" or self.selected_kernel_id is not None or self.applications:
            raise ValueError("quantum observation cannot carry host execution evidence")
        # A reduction point holds one ReducedValues record per registered
        # output component, of that component's dtype and size, from one
        # unconditional numerical trajectory.
        if reduction:
            outputs = reducer_outputs(reduction_point.reducer, reduction_point.parameters)
            if (self.returned_shots is not None or self.trajectories != 1 or self.population != "unconditional"
                    or self.physical_scale is not None or self.physical_scale_unavailable is not None
                    or self.artifacts or self.unavailable or len(self.values) != len(outputs)):
                raise ValueError("a reduction point holds its registered output components from one trajectory")
            for index, (item, output) in enumerate(zip(self.values, outputs, strict=True)):
                sizes = {"float64": (output.items, 0, 0), "complex128": (output.items, output.items, 0),
                         "int64": (0, 0, output.items)}[output.dtype]
                if (type(item) is not ReducedValues or item.component != index
                        or (len(item.real), len(item.imaginary), len(item.integers)) != sizes):
                    raise ValueError("reduced values differ from their registered output component")
            return self
        # Amplitude readout describes one numerical trajectory with an explicit
        # physical scale and output artifact, not a measured-shot population.
        if observation.kind == "amplitudes":
            declaration = observation.amplitudes
            from nwqlib.amplitudes import validate_amplitude_masses
            validate_amplitude_masses(declaration,self.values)
            if (self.physical_scale is None) != (self.physical_scale_unavailable is not None):
                raise ValueError("amplitude physical scale needs exactly one value or unavailable reason")
            if self.returned_shots is not None or self.trajectories != 1 or self.population != "unconditional":
                raise ValueError("amplitudes describe one unconditional numerical trajectory")
            if len(self.artifacts) + len(self.unavailable) != 1:
                raise ValueError("amplitude acquisition produces exactly its selected vector or explicit absence")
            if any(a.output != declaration.output for a in self.artifacts + self.unavailable):
                raise ValueError("amplitude output differs from its exact declaration")
            if any(a.acquisition != self.acquisition_key or a.plan_id != self.plan_id
                   or a.realization_id != self.realization_id or a.producer_id != declaration.content_id
                   or a.construction_id != declaration.construction_id or a.source != declaration.source for a in self.artifacts):
                raise ValueError("amplitude artifact provenance differs from its exact acquisition")
            if sorted(bit for reg in self.quantum_layout for bit in reg.bits) != list(range(declaration.width)):
                raise ValueError("amplitude logical layout differs from the selected width")
            return self
        if self.physical_scale is not None or self.physical_scale_unavailable is not None or self.artifacts or self.unavailable:
            raise ValueError("quantum scalar observations cannot carry amplitude artifacts")
        # A provider-managed estimate supplies its own uncertainty information.
        # It cannot fabricate an exact trajectory or a raw-shot count.
        if observation.kind == "estimated_observable":
            if (self.returned_shots is not None or self.trajectories is not None or self.population != "unconditional"
                    or len(self.values) != 1 or type(self.values[0]) is not EstimateValue
                    or self.values[0].estimate_id != observation.estimate.content_id):
                raise ValueError("provider estimate must bind its weighted observable without exact trajectories or raw counts")
            return self
        if observation.kind == "probabilities":
            # Publication checked the values and computed the summaries from
            # the bytes that were hashed (artifacts.probability_readout).
            # Loading checks the declared layout and the associations of the
            # manifests with this acquisition, without rereading the values.
            if self.returned_shots is not None or self.trajectories != 1:
                raise ValueError("exact simulator observation describes one trajectory, not measured shots")
            if len(self.values) != 1 or type(self.values[0]) is not ProbabilityArrays:
                raise ValueError("a probability readout stores one ProbabilityArrays record")
            arrays = self.values[0]
            if arrays.width != len(observation.qubits):
                raise ValueError("probability arrays differ from the declared readout width")
            producer = observation.content_id
            for manifest in (arrays.probabilities, arrays.indices):
                if manifest is not None and (
                        manifest.acquisition != self.acquisition_key or manifest.plan_id != self.plan_id
                        or manifest.realization_id != self.realization_id
                        or manifest.construction_id != self.prepared_id or manifest.producer_id != producer
                        or manifest.source != self.source):
                    raise ValueError("probability array provenance differs from this exact acquisition")
            return self
        # Direct scalar/count readouts must preserve the declared statistic type,
        # unique outcome keys and actual returned population.
        expected = {"counts": CountBin, "pauli_expectation": PauliValue}[observation.kind]
        if any(type(item) is not expected for item in self.values):
            raise ValueError("statistic kind differs from the requested observation")
        keys = [item.label if isinstance(item, PauliValue) else item.bits for item in self.values]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate statistic keys within one chunk")
        # The count domain: every count and the exact total are in
        # [0, MAX_COUNT], the total equals returned_shots and does not exceed
        # the requested shots, 1 <= shots <= MAX_COUNT. The total is formed
        # with Python integers, so it cannot wrap.
        if observation.kind == "counts":
            if not 1 <= observation.shots <= MAX_COUNT:
                raise ValueError(f"requested shots must be in [1, {MAX_COUNT}]")
            if any(item.count > MAX_COUNT for item in self.values):
                raise ValueError(f"a count exceeds the int64 readout limit {MAX_COUNT}")
            total = sum(item.count for item in self.values)  # Python integers
            if (self.returned_shots is None or self.returned_shots != total
                    or not 0 <= total <= observation.shots or self.trajectories is not None):
                raise ValueError(
                    "counts require the exact returned total within requested shots "
                    "and no exact-trajectory population"
                )
        elif self.returned_shots is not None or self.trajectories != 1:
            raise ValueError("exact simulator observation describes one trajectory, not measured shots")
        if observation.kind == "pauli_expectation" and not set(keys) <= set(observation.labels):
            raise ValueError("returned Pauli labels differ from requested labels")
        if observation.kind == "counts":
            width = self._readout_width()
            if any(len(key) != width or set(key) - set("01") for key in keys):
                raise ValueError("histogram keys must match the declared little-endian readout width")
        return self

    def declares_readout_of(self, receipt):
        """Whether this chunk holds the readout its preparation receipt declares.

        A point chunk holds the one-point readout of its point in the
        receipt's trajectory, whose identity is its ``trajectory_id``; any
        other chunk holds the receipt's observation itself. The receipt's
        point lookup is indexed, so K point chunks join in O(K) lookups.
        """
        if self.point is None:
            return self.trajectory_id is None and receipt.observation == self.observation
        declaration = receipt.observation
        try:
            point = declaration.point(self.point)
        except ValueError:
            return False
        return (declaration.content_id == self.trajectory_id
                and declaration._point_readout(point) == self.observation)

    def validate_unit_bound(self, receipt):
        """Admit exact probabilities, Pauli expectations and branch masses that exceed their
        unit endpoint only within the binary64 roundoff window of the executed circuit.

        The window follows the receipt's native operation count, so this check joins the
        chunk to its own preparation receipt. Alone, the chunk admits nonnegative
        probabilities and masses and finite expectations. Counts and host scalars have
        no roundoff endpoint here.

        The window is derived from the receipt rather than stored on the chunk. A
        stored copy would change the record format and add derived data that every
        reader would have to check against its receipt. The cost is that each place
        where a new chunk meets its receipt must call this method. The shared places
        are the decoding of backend output and the validation of a standalone Result
        before it is saved (``saved_evidence._validate_recorded_data``). Reopening a
        Run or loading a Result reads chunks that passed one of them, and it does
        not call this method again. Method result validators may add their own
        calls, as QLS does for its masses. A new producer of chunks that bypasses
        both places must add the call. The window
        is defined in the ``exact_probability_window`` paragraph of
        docs/ENGINEERING_CONSTANTS.md.

        A trajectory point chunk also joins its receipt's trajectory
        declaration: the same ``trajectory_id``, the declared point's one-point
        readout and its resolved boundary. A reduction point has no endpoint
        here: host contractions of saved states have their own accumulation
        error, which the receipt's probability/Pauli readout term does not
        certify. Amplitude-derived masses use the receipt's
        ``saved_state_probability_window``, which includes the host correction
        of the saved state, and are checked for nonnegativity only when that
        correction was not assessed.
        """
        if receipt.content_id != self.prepared_id:
            raise ValueError("an observation's unit bound requires its own preparation receipt")
        if self.point is not None:
            if not self.declares_readout_of(receipt):
                raise ValueError("a point chunk requires its receipt's trajectory declaration and declared point")
            if receipt.boundaries[receipt.observation.point_index(self.point)] != self.boundary:
                raise ValueError("chunk boundary differs from its receipt's resolved point boundary")
        bound = 1 + receipt.probability_window
        kind = self.readout().kind
        # The saved total is math.fsum of the published values, so this is
        # the exact predicate fsum(p_j) <= fl(1+w) (artifacts.probability_readout).
        if kind == "probabilities" and self.values[0].mass > bound:
            raise ValueError("probability marginal exceeds unit population beyond the roundoff window "
                             "of its executed circuit")
        if kind == "pauli_expectation" and any(abs(item.value) > bound for item in self.values):
            raise ValueError("exact Pauli expectation exceeds unit magnitude beyond the roundoff window "
                             "of its executed circuit")
        if kind == "amplitudes":
            # Amplitude masses are formed from the host-corrected saved state;
            # an unassessed correction (None) gives no finite upper bound.
            window = receipt.saved_state_probability_window
            for item in self.values:
                validate_normalized_mass(item.value, window)


class ObservationView(Record):
    """Bounded supplied chunks; identical fetches deduplicate only by acquisition ID."""

    schema_version: Literal[2] = 2
    chunks: tuple[ObservationChunk, ...] = ()

    @model_validator(mode="after")
    def _deduplicate(self):
        """Keep one copy of a chunk fetched twice, and reject different data under one key.

        Equal values under different acquisition keys stay separate chunks,
        because they come from separate acquisitions.
        """
        unique = {}
        for item in self.chunks:
            prior = unique.get(item.acquisition_key)
            if prior is not None and prior != item:
                raise ValueError("conflicting payload under the same immutable acquisition identity")
            unique[item.acquisition_key] = item
        object.__setattr__(self, "chunks", tuple(unique.values()))
        return self


class TimingObservation(Record):
    """Supplied scoped seconds, independent of event outcome or output validity.

    exact names an observed duration, possibly time until failure. right_censored
    names a lower bound for an explicitly continuing target, with a required
    reason/source identifying the cutoff. It is never an exact residual or a
    coverage claim. Native synchronous calls emit exact terminal durations only.
    """

    scope: TimeScope
    seconds: Nonnegative
    censoring: Literal["exact", "right_censored"] = "exact"
    source: Source
    reason: Text | None = None

    @model_validator(mode="after")
    def _censoring(self):
        if self.censoring == "right_censored" and self.reason is None:
            raise ValueError("right-censored time requires its continuing target/cutoff reason")
        return self


class ExecutionLimits(Record):
    """Execution-only inclusive limits, independent of symbolic planning.

    max_data_bytes bounds stored run records and numerical arrays, not process
    RSS. Simulator memory covers the native state and known working arrays.
    max_total_circuits independently caps preparation and circuit acquisitions.

    Attributes:
        max_simulation_qubits: Maximum selected width admitted before local native simulation.
        simulator_memory_mb: Native simulator memory allowance in MiB.
        max_total_circuits: Independent inclusive cap on quantum preparations and on circuit acquisition attempts.
        max_total_shots: Inclusive total reserved raw-shot budget, including uncertain attempts.
        max_data_bytes: Cap on recorded Run data and numerical payloads, not total process RSS.
        max_completion_metadata_bytes: Additional JSON metadata allowance per completed acquisition, apart from declared numeric values, arrays and host application receipts that the kernel declares.
        max_direct_amplitudes: Largest amplitude count of a declared direct magnitude/phase state preparation that the Run synthesizes.
        max_synthesis_work: Largest total work, in the units of ``_dense_synthesis.dense_synthesis_size``, of the exact dense-unitary syntheses that the Run makes while it prepares circuits, over all its preparations. It counts each synthesis that a backend makes when it lowers a prepared circuit to a gate basis. The Run keeps the circuit of each such synthesis, keyed by its matrix, until the Run is closed (``Run._exact_dense_unitaries``), so each distinct matrix is synthesized and counted once, and again after the Run is reopened. It also counts the syntheses and Qiskit's control of them (``_dense_synthesis.gatewise_control_size``) when lowering controls a transformed block. Each amount is checked against the remaining allowance and reserved on the preparation's charge before its first synthesis starts. A preparation that the check refuses stays charged as a preparation but leaves its experiment unprepared, so a later call can prepare it once the limit is raised. Syntheses that a Method charges to its own limits at planning are not counted.
    """

    schema_version: Literal[5] = 5
    # The defaults are untuned local choices that a caller raises explicitly for a
    # larger intended workload. docs/ENGINEERING_CONSTANTS.md registers them under
    # "Explicit workflow and reference controls" (qubits, memory, circuits,
    # shots, direct amplitudes and synthesis work), "Shared byte default"
    # (max_data_bytes) and "Local journal scalar admission" (completion
    # metadata). A 20-qubit complex128 statevector needs 16 MiB. The qubit cap
    # applies to Aer and the local NWQ-Sim backend. Aer receives
    # simulator_memory_mb as max_memory_mb, its cap on quantum-state storage.
    # The synthesis work of 10**9 units, 0.65 to 1.1 s on one core, admits the
    # synthesis of one dense unitary on 8 qubits (5.2 * 10**8 units) in a Run
    # and refuses one on 9 qubits (4.0 * 10**9 units).
    max_simulation_qubits: PositiveInt = 20
    simulator_memory_mb: PositiveInt = 1024
    max_total_circuits: PositiveInt = 512
    max_total_shots: PositiveInt = 1_000_000
    max_data_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_completion_metadata_bytes: PositiveInt = 65_536
    max_direct_amplitudes: PositiveInt = DEFAULT_MAX_DIRECT_AMPLITUDES
    max_synthesis_work: PositiveInt = 1_000_000_000


class LimitAmendment(Record):
    """One committed increase and the known counters immediately before it.

    sequence orders amendments; recorded_at is an aware wall-clock timestamp,
    not an elapsed-time measure. old/new contain complete execution caps.
    circuit_preparations includes failed and successful-unused preparations;
    circuit_attempts includes reserved and uncertain circuit acquisitions.
    raw_shots counts explicitly prescribed shots, not unknown sampling inside
    managed provider estimates. stored_data_bytes excludes pending_output_bytes;
    both describe this Run's accounted data, never physical memory/RSS.
    synthesis_work is the dense synthesis work reserved against
    max_synthesis_work.
    """

    schema_version: Literal[2] = 2
    sequence: PositiveInt
    recorded_at: AwareDatetime
    old: ExecutionLimits
    new: ExecutionLimits
    circuit_preparations: Count
    circuit_attempts: Count
    raw_shots: Count
    stored_data_bytes: Count
    pending_output_bytes: Count
    synthesis_work: Count

    @model_validator(mode="after")
    def _increase(self):
        """Require at least one cap to grow and none to shrink, with counters inside the old caps."""
        names = ExecutionLimits.model_fields.keys() - {"schema_version", "parent_id"}
        changes = tuple(getattr(self.new, name) - getattr(self.old, name) for name in names)
        if min(changes) < 0 or max(changes) == 0:
            raise ValueError("limit amendment requires an actual increase without decreases")
        if (max(self.circuit_preparations, self.circuit_attempts) > self.old.max_total_circuits
                or self.raw_shots > self.old.max_total_shots
                or self.stored_data_bytes + self.pending_output_bytes > self.old.max_data_bytes
                or self.synthesis_work > self.old.max_synthesis_work):
            raise ValueError("pre-amendment counters exceed the old execution limits")
        return self


def _validate_limit_amendments(amendments, *, current, initial=None):
    """Validate an ordered cap chain without reading or reproducing execution."""
    previous = initial
    counts = (0, 0, 0, 0)
    for sequence, amendment in enumerate(amendments, 1):
        if amendment.sequence != sequence or previous is not None and amendment.old != previous:
            raise ValueError("limit amendment history has invalid ordering or old/new continuity")
        updated = (amendment.circuit_preparations, amendment.circuit_attempts, amendment.raw_shots,
                   amendment.synthesis_work)
        if any(new < old for old, new in zip(counts, updated, strict=True)):
            raise ValueError("limit amendment history resets cumulative execution counters")
        previous, counts = amendment.new, updated
    if previous is not None and current != previous:
        raise ValueError("current execution limits differ from their amendment history")


class ConsumptionEvent(Record):
    """Execution exposure is separate from scientific contributions.

    status is reserved before the call and then becomes completed (with its
    observation), uncertain or failed. Started quantum failures keep the full reserved shots as
    uncertain exposure, because the provider may have executed or billed them.
    A durably recorded synchronous host exception is failed, with its invocation
    and work still charged. A process interruption without that record is uncertain.
    A synchronous adapter cannot determine actual provider billing after failure.
    evaluations counts requested non-count observable readouts, not simulator
    trajectories; a trajectory schedule of K points is one evaluation.
    A completed trajectory attempt's observation_id is the content identity
    of the ObservationView of its point chunks in schedule order, which binds
    the complete collection.
    provider_managed_sampling marks unknown sampling cost behind
    a provider estimate; shots then counts zero explicitly prescribed raw shots,
    without claiming zero physical sampling or a provider-spend bound.
    assessment_id is bound before native work. timing measures its named scope,
    separately from started/finished lifecycle timestamps and scientific output.
    """

    schema_version: Literal[2] = 2
    attempt: Text
    execution: Literal["quantum_circuit", "host_kernel"] = "quantum_circuit"
    prepared_id: ContentID
    status: Literal["reserved", "completed", "uncertain", "failed"]
    submission: Text | None = None
    evaluations: Count
    shots: Count
    started: Text
    finished: Text | None = None
    returned_shots: Count | None = None
    failure: Text | None = None
    observation_id: ContentID | None = None
    host_invocations: Count = 0
    host_work: Count = 0
    data_bytes_reserved: Count = 0
    assessment_id: ContentID | None = None
    timing: TimingObservation | None = None
    checkpoint_sequence: PositiveInt | None = None
    artifacts_reserved: Count = 0
    provider_managed_sampling: StrictBool = False

    @model_validator(mode="after")
    def _receipt(self):
        """Tie each status to the evidence it needs and keep host and quantum attempts apart.

        Only a completed event names an observation. Only a finished host
        exception can be failed, since a quantum call that raised may still have
        run. A host attempt is one invocation with no submission, shots or
        evaluations, and a quantum attempt has a submission and no host work.
        """
        if (self.status == "completed") != (self.observation_id is not None):
            raise ValueError("only completed events require an immutable observation receipt identity")
        if self.status == "failed" and (self.execution != "host_kernel" or self.finished is None
                                        or self.failure is None):
            raise ValueError("failed events require a finished host exception; quantum exposure stays uncertain")
        if self.execution == "host_kernel":
            if (self.submission is not None or self.evaluations or self.shots or self.provider_managed_sampling
                    or self.returned_shots is not None or self.host_invocations != 1):
                raise ValueError("one host attempt is one invocation and no quantum event")
        elif self.submission is None or self.host_invocations or self.host_work:
            raise ValueError("quantum acquisitions require a submission and cannot reserve host work")
        return self


class JobLocator(Record):
    """A provider/scheduler job identity with credential-free account context."""

    provider: Text
    job_id: Text
    account: Text | None = None
    instance: Text | None = None
    project: Text | None = None
    region: Text | None = None
    cluster: Text | None = None
    process_id: PositiveInt | None = None
    host: Text | None = None


class PendingPreparation(Record):
    """Original local data and remote upload/compile lifecycle, without shots.

    The SDK circuit data lives in local_data, so revisions never hash or parse
    its contents. A known upload or compile ID is stored before further work.
    """

    preparation_id: Text
    workflow_item: Text | None = None
    run_id: Text
    realization: Realization
    runtime: RuntimeOptions
    snapshot: Text
    construction_id: ContentID
    bindings: tuple[Binding, ...]
    observation: ObservationSpec
    items: Count
    setting: Text
    logical: LogicalPreparationReceipt
    quantum_layout: tuple[RegisterMap, ...]
    classical_layout: tuple[RegisterMap, ...]
    local_data: PayloadRef
    status: Literal["local", "upload_intent", "upload_uncertain", "uploaded", "compile_intent",
                    "compile_uncertain", "compile_pending", "completed", "failed", "cancelled"] = "local"
    upload_id: Text | None = None
    input_ref_json: Text | None = None
    locator: JobLocator | None = None
    prepared_id: ContentID | None = None
    failure: Text | None = None
    provider_status: Text | None = None
    cancel_requested: Text | None = None


def point_chunk_key(result_key, point):
    """The ``chunk`` key of one trajectory point's chunk, ``"<result_key>/<point>"``.

    The point chunks of one acquisition share its run, attempt and job and
    differ in this key.
    """
    return f"{result_key}/{point}"


class SubmissionItem(Record):
    """One acquisition's exact position in the submitted result population.

    item identifies the submitted PUB/circuit. result_key is the decoder's
    stable chunk key, not a flattened histogram. Optional child job and
    provider result identifiers are part of the provider association.
    """

    schema_version: Literal[2] = 2
    attempt: Text
    prepared_id: ContentID
    item: Count
    result_key: Text = "0"
    child_job: Text | None = None
    provider_result_id: Text | None = None

    def validate_observation(self, chunk, *, run_id, locator):
        """Check that ``chunk`` came from this item: same run, attempt, preparation, job and result key.

        The job is the item's child job when the provider reported one, and the
        submission's job otherwise. A trajectory point chunk's key is
        ``point_chunk_key(result_key, point)``.
        """
        key = self.result_key if chunk.point is None else point_chunk_key(self.result_key, chunk.point)
        if (chunk.run_id != run_id or self.attempt != chunk.attempt or self.prepared_id != chunk.prepared_id
                or locator is None or chunk.job != (self.child_job or locator.job_id)
                or chunk.chunk != key):
            raise ValueError("quantum observation differs from its actual job/item association")


class SubmissionRecord(Record):
    """One admitted physical request, separate from its acquisition items.

    An intent reserves one possible job even if its acknowledgement is lost.
    Provider billing and local native invocation evidence are not inferred
    from request count.

    A detached or reconciled submission moves from intent to acknowledged (its
    locator is known) and then to completed, failed or cancelled as the provider
    reports. A synchronous submission goes from intent straight to completed,
    with its locator. An intent found on reopen, a call that raised, or a
    provider that reports an uncertain state makes it uncertain.
    A terminal submission sets results_consumed when its remaining observations
    are published or its unpublished output can no longer be retrieved. This
    flag stops further retrieval and releases pending output capacity. A failed
    submission with consumed results can still have uncertain attempts and
    charged execution exposure.
    """

    schema_version: Literal[2] = 2
    submission_id: Text
    run_id: Text
    backend: Source
    items: tuple[SubmissionItem, ...]
    status: Literal["intent", "acknowledged", "completed", "failed", "cancelled", "uncertain"]
    started: Text
    locator: JobLocator | None = None
    finished: Text | None = None
    provider_status: Text | None = None
    failure: Text | None = None
    cancel_requested: Text | None = None
    refresh_failure: Text | None = None
    native_invocations: Count | None = None
    native_simulations: Count | None = None
    timing: TimingObservation | None = None
    results_consumed: StrictBool = False

    @model_validator(mode="after")
    def _association(self):
        """Require distinct item coordinates, a locator where the status requires one and none for an intent.

        Attempts, item indices and result keys must each be distinct, so one
        returned result maps to one attempt. An intent has no locator yet, and an
        acknowledged, completed or cancelled submission has one. A failed or
        uncertain submission may lack one, because its acknowledgement may never
        have arrived. Only a terminal submission can have its results consumed.
        """
        if not self.items:
            raise ValueError("a submission requires its acquisition items")
        if len({item.attempt for item in self.items}) != len(self.items):
            raise ValueError("one submission cannot repeat an acquisition attempt")
        if len({item.item for item in self.items}) != len(self.items) or len({item.result_key for item in self.items}) != len(self.items):
            raise ValueError("submission items require distinct provider/result coordinates")
        if self.status in {"acknowledged", "completed", "cancelled"} and self.locator is None:
            raise ValueError("acknowledged submission requires its actual job locator")
        if self.status == "intent" and self.locator is not None:
            raise ValueError("an unacknowledged intent cannot contain a known job locator")
        if self.results_consumed and self.status not in {"completed", "failed", "cancelled"}:
            raise ValueError("only terminal provider results can be fully consumed")
        return self


class ExecutionTrace(Record):
    """One run's bounded preparation charges and attempt history.

    local_prepared_ids names successful preparations performed in this run.
    A used portable receipt may instead come from another authorized run;
    its use incurs execution exposure, not a new local preparation charge.
    limits and limit_amendments capture this snapshot's cap/history prefix;
    absent limits means this standalone trace has no supplied cap metadata.

    Attributes:
        run_id: Original Run identity.
        plan_id: Exact selected Plan identity.
        preparations: Number of actual local quantum preparation attempts.
        construction_work_reserved: Quantum construction work charged in this Run.
        events: Ordered immutable attempt records, including failed and uncertain work.
        submissions: Original quantum submission records and per-item association.
        host_preparations: Number of local host setup attempts.
        host_preparation_work_reserved: Counted host setup work charged in this Run.
        host_invocations: Number of reserved host invocations, including interrupted work.
        host_work_reserved: Counted host execution work charged before invocation.
        data_bytes_reserved: Output/data reservation total across attempts.
        data_bytes: Recorded Run data bytes at this snapshot.
        synthesis_work_reserved: Exact dense synthesis work reserved against ``max_synthesis_work`` in this Run.
        local_prepared_ids: Successful preparations actually performed locally in this Run.
        cancel_requested: Explicit cancellation request, or None; a request does not prove provider cancellation.
        termination_reason: Method/controller termination explanation, or None while absent.
        limits: Limits at this snapshot, or None for a standalone trace without cap metadata.
        limit_amendments: Original ordered history of explicit limit increases.
    """

    schema_version: Literal[4] = 4
    run_id: Text
    plan_id: ContentID
    preparations: Count
    construction_work_reserved: Count
    events: tuple[ConsumptionEvent, ...]
    submissions: tuple[SubmissionRecord, ...] = ()
    host_preparations: Count = 0
    host_preparation_work_reserved: Count = 0
    host_invocations: Count = 0
    host_work_reserved: Count = 0
    data_bytes_reserved: Count = 0
    data_bytes: Count = 0
    synthesis_work_reserved: Count = 0
    local_prepared_ids: tuple[ContentID, ...] = ()
    cancel_requested: Text | None = None
    termination_reason: Text | None = None
    limits: ExecutionLimits | None = None
    limit_amendments: tuple[LimitAmendment, ...] = ()
    _acquisitions: dict = PrivateAttr(default_factory=dict)

    @property
    def jobs(self) -> int:
        """Reserved physical quantum requests, including unresolved intents."""
        return len(self.submissions)

    def validate_observation(self, chunk):
        """Join one quantum result to its actual job/item in constant time."""
        if chunk.execution == "host_kernel":
            return
        submission, item = self._acquisition(chunk.attempt)
        item.validate_observation(chunk, run_id=self.run_id, locator=submission.locator)

    def _acquisition(self, attempt):
        member = self._acquisitions.get(attempt)
        if member is None:
            raise ValueError("quantum observation lacks its actual submission item")
        return member

    @model_validator(mode="after")
    def _host_reservations(self):
        """Reconcile limits, submissions and host reservations against the actual attempt
        inventory.
        """
        _validate_limit_amendments(self.limit_amendments, current=self.limits)
        if self.limits is not None and self.data_bytes > self.limits.max_data_bytes:
            raise ValueError("captured data bytes exceed this trace's execution limit")
        if self.limits is not None and self.synthesis_work_reserved > self.limits.max_synthesis_work:
            raise ValueError("reserved synthesis work exceeds this trace's execution limit")
        if self.limit_amendments:
            last = self.limit_amendments[-1]
            if (last.circuit_preparations > self.preparations - self.host_preparations
                    or last.circuit_attempts > sum(event.execution == "quantum_circuit" for event in self.events)
                    or last.raw_shots > sum(event.shots for event in self.events)
                    or last.synthesis_work > self.synthesis_work_reserved):
                raise ValueError("limit amendment counters exceed this trace's actual population")
        self._acquisitions = {}
        if len({s.submission_id for s in self.submissions}) != len(self.submissions):
            raise ValueError("trace submission identities must be distinct")
        for submission in self.submissions:
            if submission.run_id != self.run_id:
                raise ValueError("trace submission belongs to another run")
            for item in submission.items:
                if item.attempt in self._acquisitions:
                    raise ValueError("acquisition appears in more than one submission")
                self._acquisitions[item.attempt] = submission, item
        attempts = set()
        for event in self.events:
            if event.attempt in attempts:
                raise ValueError("trace acquisition attempts must be distinct")
            attempts.add(event.attempt)
            if event.execution == "quantum_circuit":
                member = self._acquisitions.get(event.attempt)
                if (member is None or member[0].submission_id != event.submission
                        or member[1].prepared_id != event.prepared_id):
                    raise ValueError("quantum acquisition differs from its submission item")
        if set(self._acquisitions) != {e.attempt for e in self.events if e.execution == "quantum_circuit"}:
            raise ValueError("submission item inventory differs from actual quantum acquisitions")
        if (len(set(self.local_prepared_ids)) != len(self.local_prepared_ids)
                or len(self.local_prepared_ids) > self.preparations
                or self.host_preparations > self.preparations
                or self.host_invocations != sum(event.host_invocations for event in self.events)
                or self.host_work_reserved != sum(event.host_work for event in self.events)
                or self.data_bytes_reserved != sum(event.data_bytes_reserved for event in self.events)
):
            raise ValueError("trace preparation/host reservations disagree with actual attempted events")
        return self


class RunFailed(RuntimeError):
    """A terminal outcome or an unavailable recovery from Run continuation.

    The stage and status describe the outcome being reported. A recovery error
    reports an unavailable attempt outcome with status uncertain, even when its
    submission is terminal because the output is irretrievable. ``locator`` and
    ``directory`` preserve the original recovery coordinates; ``failure`` is
    recorded failure text, not an invented local exception. ``trace`` and
    ``exposure`` capture the existing attempts and charges without another
    backend call. Recovery with status uncertain does not claim that execution
    failed. The attempt field identifies the original acquisition, whose record
    and charged exposure are available in the trace and exposure snapshots.

    Attributes:
        stage: ``prepare``, ``execute`` or ``recovery``.
        status: Observed outcome, such as ``failed``, ``cancelled`` or ``uncertain``.
        locator: Original provider job or remote-preparation locator, or None when
            none was acknowledged.
        failure: Recorded failure text, or None.
        directory: Durable Run folder, or None for an in-memory Run.
        trace: ExecutionTrace snapshot at the time of the error.
        exposure: Read-only exposure snapshot, as returned by ``Run.exposure``.
        attempt: Attempt identity for a recovery error, otherwise None.
    """

    def __init__(self, *, stage, status, locator, failure, directory, trace, exposure, attempt=None):
        """Keep the recovery coordinates and build one message naming each of them.

        ``exposure`` is copied into read-only mappings, so a caller cannot edit
        the snapshot it inspects. A recovery error adds the remedy: inspect the
        saved evidence or cancel the Run, and start a new Run to execute the
        Plan again, because resubmitting in place is never offered.
        """
        from types import MappingProxyType

        self.stage, self.status = stage, status
        self.locator, self.failure = locator, failure
        self.attempt = attempt
        self.directory, self.trace = directory, trace
        self.exposure = MappingProxyType({
            name: MappingProxyType(dict(values)) for name, values in exposure.items()
        })
        job = "" if locator is None else f"; original job {locator.job_id}"
        cause = "" if failure is None else f": {failure}"
        recovery = "" if directory is None else f"; original run is saved at {directory}"
        unavailable = (
            f"; cannot recover the outcome of original attempt {attempt}. "
            "Inspect the saved evidence or cancel this Run. "
            "To execute the Plan again, start a new Run. "
            "The original attempt remains uncertain and its execution exposure stays charged. "
            "Nothing is resubmitted in this Run."
            if stage == "recovery" else ""
        )
        super().__init__(f"run {stage} {status}{cause}{job}{recovery}{unavailable}")


__all__ += [
    "RegisterMap", "LogicalPreparationReceipt", "PreparedArtifact", "PauliValue", "CountBin", "ProbabilityArrays",
    "PayloadRef", "CountsSampling", "EstimateValue", "ReducedValues",
    "Histogram", "ObservationChunk", "ObservationView", "ConsumptionEvent", "ExecutionTrace", "TimingObservation", "ExecutionLimits", "LimitAmendment",
    "JobLocator", "SubmissionItem", "SubmissionRecord", "RunFailed",
    "ScalarValue", "KernelApplication", "VerificationReceipt",
    "Run", "PreparedHandle", "PendingPreparation", "prepare_experiment", "refresh_preparation",
    "submit_experiment", "submit_detached", "refresh_submissions",
]


def __getattr__(name):
    """Resolve the lifecycle names on first access.

    ``_prepared_execution`` and ``_remote_preparation`` import this module, so
    importing them here at module level would be circular.
    """
    if name == "refresh_preparation":
        from nwqlib._remote_preparation import refresh_preparation
        return refresh_preparation
    if name in {"Run", "PreparedHandle", "prepare_experiment", "submit_experiment", "submit_detached", "refresh_submissions"}:
        from importlib import import_module
        return getattr(import_module("nwqlib._prepared_execution"), name)
    raise AttributeError(name)
