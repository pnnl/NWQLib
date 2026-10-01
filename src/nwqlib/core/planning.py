"""Portable Plan/Realization ownership, with method-owned concrete record types.

Interchange never loads a class from a Source. Callers deserialize with the
same module-level concrete class used to construct the Plan.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import (Field, InstanceOf, PrivateAttr, SerializeAsAny, StrictInt, computed_field, field_serializer,
                      field_validator, model_validator)

from nwqlib._validation import MAX_COUNT
from nwqlib.amplitudes import AmplitudeReadout
from nwqlib.blocks.records import SelectedConstruction, SelectedKernel
from nwqlib.core.records import ContentID, Real, Record, Text
from nwqlib.evidence.error_model import ErrorModel, FramedFact
from nwqlib.ir import Binding, MeasurementBatch
from nwqlib.ir.expressions import number
from nwqlib.ir.validation import _Admission
from nwqlib.operators.access import Count
from nwqlib.problems.records import Accuracy


class ObservableEstimateSpec(Record):
    """One requested finite real Pauli sum, in its original physical units.

    observable_id identifies the original physical observable; labels and
    coefficients keep the actual weighted operator, including identity terms.
    precision is a requested standard-error target in the same coefficient
    units, not a bound on physical error or provider cost.

    Attributes:
        observable_id: Identity of the original physical observable.
        labels: Distinct equal-width Pauli labels, including identity contributions.
        coefficients: Finite real weights in the original observable units, aligned with labels.
        precision: Requested provider standard-error target in those units; not total physical error.
    """

    observable_id: ContentID
    labels: tuple[Text, ...]
    coefficients: tuple[Real, ...]
    precision: Annotated[Real, Field(gt=0)]

    @model_validator(mode="after")
    def _observable(self):
        if (not self.labels or len(self.labels) != len(self.coefficients)
                or len(set(self.labels)) != len(self.labels)
                or len({len(label) for label in self.labels}) != 1
                or any(set(label) - set("IXYZ") for label in self.labels)):
            raise ValueError("estimated observable requires matching distinct finite real Pauli terms")
        return self


@dataclass(frozen=True)
class ReducerOutput:
    """One output component of a registered acquisition-time reduction.

    ``shape`` is the component's array shape, whose product is its number of
    logical output values. ``index_width`` is the bit width of the packed
    outcome index stored beside each value of an indexed component, and zero
    for an unindexed component. The payload reservation of the component is
    ``payload_bytes`` below.
    """

    dtype: Literal["float64", "int64", "complex128"]
    shape: tuple[int, ...]
    index_width: int = 0

    def __post_init__(self):
        if (self.dtype not in ("float64", "int64", "complex128") or type(self.shape) is not tuple
                or any(type(size) is not int or size < 0 for size in self.shape)
                or type(self.index_width) is not int or self.index_width < 0
                or (self.index_width and self.dtype == "complex128")):
            raise ValueError("reducer output needs a float64, int64 or complex128 dtype, a nonnegative integer "
                             "shape and a nonnegative index width; an indexed component stores an 8-byte value")

    @property
    def items(self):
        """Number of logical output values, the product of ``shape``."""
        return prod(self.shape)

    @property
    def payload_bytes(self):
        """Numerical payload bytes of the component under the NWQLib array payload law.

        The array law is ``8*sum(unindexed_real L_k) + 16*sum(complex L_k) +
        8*sum(indexed (W_k+1)*s_k)`` with ``W_k=(width_k+63)//64``. An
        unindexed float64 or int64 value occupies eight bytes, a complex128
        value sixteen. An indexed component stores an eight-byte value column
        and ``W`` packed eight-byte index words per entry. A reduction reserves
        its registered output's actual dtype and shape, including any indexed
        components, so two complex128 output scalars reserve 32 numerical
        bytes. The component's shape counts its stored entries.
        """
        if self.index_width:
            return 8 * ((self.index_width + 63) // 64 + 1) * self.items
        return (16 if self.dtype == "complex128" else 8) * self.items


@dataclass(frozen=True)
class Reducer:
    """Registered acquisition-time reduction: its shape, byte, work and execution functions.

    The registry interface supplies the validated output shape/dtypes, the
    host work of one reduction and its execution. Unknown reducers fail
    shape resolution before native work. A built-in reducer registers itself
    in ``READOUT_REDUCERS`` when its module is imported, and that module is
    listed in ``BUILTIN_REDUCER_MODULES`` so that ``registered_reducer``
    resolves its name in a process that has not imported it. A user reducer
    is added to ``READOUT_REDUCERS`` directly, under a name that no built-in
    reducer uses. Each reducer's shape is
    registered before it can be admitted.

    The functions a reducer registers are:

    - ``shape(parameters)``: the output components, whose byte function is
      ``ReducerOutput.payload_bytes`` (the reservation of the published
      output, which ``point_array_reservation`` and ``_admit_readout`` use).
    - ``work(parameters, width)``: the host work of one reduction of a saved
      state of ``width`` qubits, as a nonnegative integer in the unit that
      the calling Method's work allowance counts. The execution interface
      (``admit_reductions``) takes the calling Method's remaining work
      allowance and refuses a schedule whose summed registered work exceeds
      it, before acquisition. No shared reduction cap exists.
    - ``execute(state, parameters, bindings)``: the reduction itself. It
      receives the read-only complex128 simulator state saved at its point
      (qubit zero least significant), the validated read-only parameter
      mapping and the experiment's immutable argument bindings, and returns
      one array per registered component with that component's dtype and
      shape (``execute_reduction`` checks them). The saved outputs keep the
      bindings with them in their point chunk. A reducer registered with
      ``receives_context=True`` is also called with the keyword ``context``,
      a read-only ``ReductionContext`` of the producing preparation receipt,
      so that its acquisition-time validation can consult that receipt's
      window, exclusions and qualified state error before its outputs are
      published. The context's ``state_error`` is the modulo-phase budget for
      a reducer registered ``phase_invariant`` and unavailable, with its
      reason, for any other; ``modulo_phase_state_error`` carries the
      modulo-phase budget for both (``ReductionContext``).

    Attributes:
        shape: Function of the validated parameter mapping that returns the
            output components as a nonempty tuple of ``ReducerOutput``. It
            describes the published output only; the reduction's private
            state or action workspace is admitted by the owner that allocates it.
        work: Registered host work function, or None when no execution is registered.
        execute: Registered execution function, or None when no execution is
            registered. A reducer is executable when both are registered.
        phase_invariant: True when the reducer's outputs are unchanged when
            the saved state is multiplied by one common global phase, and
            False otherwise, the default. It is set at registration and read
            from the registry (``registered_reducer(name).phase_invariant``).
            It declares whether an intermediate reduction point of this
            reducer may be served without a record of the saved state's
            global phase at that point, and whether its context carries the
            modulo-phase budget as its ``state_error`` (``ReductionContext``).
        receives_context: True when ``execute`` takes the keyword
            ``context``; False, the default, calls it with the three
            positional arguments only.
        state_error_resolutions: Receipt exclusion labels whose effects this
            reducer's own readout arithmetic bounds. The execution interface passes
            them to the producing receipt's saved_state_error method when it builds the
            ReductionContext. The default resolves no labels.
    """

    shape: Callable[[Mapping[str, Any]], tuple[ReducerOutput, ...]]
    work: Callable[[Mapping[str, Any], int], int] | None = None
    execute: Callable[..., tuple] | None = None
    phase_invariant: bool = False
    receives_context: bool = False
    state_error_resolutions: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.phase_invariant) is not bool or type(self.receives_context) is not bool:
            raise ValueError("a reducer declares phase_invariant and receives_context as booleans")
        if (type(self.state_error_resolutions) is not tuple
                or any(type(label) is not str or not label
                       for label in self.state_error_resolutions)):
            raise ValueError("state-error resolutions must be a tuple of nonempty labels")
        if (self.work is None) != (self.execute is None) or any(
                function is not None and not callable(function) for function in (self.work, self.execute)):
            raise ValueError("an executable reducer registers both its work and its execution function")

    @property
    def executable(self):
        """Whether both the work and the execution function are registered."""
        return self.execute is not None


@dataclass(frozen=True)
class ReductionContext:
    """Read-only facts of the preparation receipt that produced a reduction's saved state.

    A reducer registered with ``receives_context=True`` receives it as the
    keyword ``context`` (``execute_reduction``).

    A saved-state reduction receives a qualified native estimate modulo
    global phase when its declared outputs are invariant under that phase.
    Phase-defined state uncertainty is unavailable unless a separate
    phase-error model supplies it. The context reports that reason. A host
    phase correction adds its finite multiplication and modulus error to every
    consumer of the corrected array. Projected mass checks can use the
    modulo-phase estimate even when other diagnostics from the same reducer
    depend on phase. If native premises remain unavailable, they use the
    propagated probability-window convention and report the original
    exclusions.

    The receipt's state-error budget (``PreparedArtifact.state_error``)
    bounds the computed state only up to one common global phase. It does
    not bound the exact prefix phase, the accumulation of the phase ledger,
    the subtraction forming its correction angle, or a phase-defined
    amplitude, and no finite phase-defined estimate follows from adding a
    host term to it. A saved state shared by a phase-sensitive and a
    phase-invariant reducer is corrected once, and both see the same array
    and the same receipt-wide host envelope
    (``PreparedArtifact.statevector_roundoff``): the invariant consumer
    cannot omit the product charge because its own registration did not
    request the phase correction.

    Attributes:
        prepared_id: Content identity of the producing preparation receipt.
        probability_window: That receipt's
            ``saved_state_probability_window``, its probability window
            propagated through the host correction of the saved state
            (``_phase_product.corrected_mass_window``), or None when that
            correction was not assessed, which gives no finite mass-check
            budget. It does not certify a host contraction of the saved
            state, which has its own accumulation error.
        probability_window_exclusions: That receipt's
            ``probability_window_exclusions``: the operations, readout path,
            simulator version or non-default compiler optimization level whose
            effect the window does not bound; empty
            when the whole execution lies inside it, None when not assessed.
        state_error: The qualified state budget for the reducer's declared
            phase semantics. For a reducer registered ``phase_invariant`` it
            is ``modulo_phase_state_error``; for any other reducer it is None,
            since the phase-defined budget is unavailable. It is also None
            when a native premise remains unavailable. A reducer uses its
            probability-window convention when this value is unavailable. The
            receipt's original exclusions remain available in
            probability_window_exclusions.
        state_error_reason: Why ``state_error`` is None: the phase-defined
            reason for a reducer that is not registered phase invariant, the
            receipt's native or host unavailable reason otherwise; None when
            ``state_error`` is available.
        modulo_phase_state_error: The receipt's ``saved_state_error`` after
            resolving the labels declared by the selected reducer
            (``_phase_product.corrected_state_error``), a distance modulo one
            common global phase with the host product charge included, or
            None when a native premise or the host assessment is unavailable.
            A phase-sensitive reducer may use it only for outputs
            independently shown invariant, as the projected mass checks do.
    """

    prepared_id: str
    probability_window: float | None
    probability_window_exclusions: tuple[str, ...] | None
    state_error: float | None = None
    state_error_reason: str | None = None
    modulo_phase_state_error: float | None = None


# Name -> Reducer. A built-in reducer registers itself here when its module is
# imported; a user reducer is added directly. Look names up with
# registered_reducer, which also resolves built-in reducers whose module the
# process has not imported yet.
READOUT_REDUCERS: dict[str, Reducer] = {}

# The module of each built-in reducer, so that a reducer name in a saved
# record resolves in a process that has not imported that module. A later
# built-in reducer adds its module here. A listed module imports with the base
# dependencies alone, since any failed import would replace every unknown-name
# refusal with that ImportError, and it looks up no reducer while it is being
# imported, since that lookup would see the module partially initialized.
# Built-in reducer names are reserved: importing a listed module, which any
# lookup miss does, replaces a user reducer registered under a built-in name.
BUILTIN_REDUCER_MODULES: tuple[str, ...] = (
    "nwqlib._quantum_readout",
    "nwqlib.algorithms.gcim.pair_reducer",
)


def registered_reducer(name):
    """Return the reducer registered under ``name``, or None when there is none.

    On a miss, each module of ``BUILTIN_REDUCER_MODULES`` is imported (an
    already imported module is not imported again, and a module registers its
    reducers once, at import) and the registry is consulted again.
    """
    reducer = READOUT_REDUCERS.get(name)
    if reducer is None:
        from importlib import import_module

        for module in BUILTIN_REDUCER_MODULES:
            import_module(module)
        reducer = READOUT_REDUCERS.get(name)
    return reducer


def reducer_outputs(name, parameters):
    """Resolve a registered reducer's output components from its validated parameters.

    An unknown reducer, a shape function that fails, or a result that is not
    a nonempty tuple of ``ReducerOutput`` rejects before native work: an
    unresolved output shape is never counted as zero.
    """
    reducer = registered_reducer(name)
    if reducer is None:
        raise ValueError(
            f"reduction {name!r} has no registered output shape in this process; unknown reducers are "
            "refused before native work. An extension Method registers its reducers when its module is "
            "imported, so load a saved Result that uses one with load_result(path, method=...)")
    try:
        outputs = reducer.shape(MappingProxyType(json.loads(parameters)))
    except Exception as error:
        raise ValueError(f"reduction {name!r} could not resolve its output shape: {error}") from error
    if type(outputs) is not tuple or not outputs or any(type(item) is not ReducerOutput for item in outputs):
        raise ValueError(f"reduction {name!r} must resolve a nonempty tuple of ReducerOutput")
    return outputs


def admit_reductions(observation, *, width, allowance):
    """Admit a trajectory's reductions before acquisition and return their registered host work.

    Each reduction point needs a reducer registered with its execution and
    work functions, and unindexed output components (an indexed component
    needs an array payload, which the point chunk's JSON values do not
    provide). The work of the schedule is the sum of ``work(parameters,
    width)`` over its reduction points, checked against ``allowance`` point
    by point. ``allowance`` is the calling Method's remaining work allowance
    (the Method owns the ledger it comes from); a schedule with reductions
    and no allowance is refused. Nothing here runs a reduction.
    """
    total = 0
    for point in observation.positions:
        if point.kind != "reduction":
            continue
        outputs = reducer_outputs(point.reducer, point.parameters)
        reducer = registered_reducer(point.reducer)
        if not reducer.executable:
            raise ValueError(f"reduction {point.reducer!r} has no registered execution")
        if any(output.index_width for output in outputs):
            raise ValueError(f"reduction {point.reducer!r} has an indexed output component, which needs an array "
                             "payload")
        if allowance is None:
            raise ValueError(f"reduction {point.reducer!r} at point {point.id!r} needs the calling Method's "
                             "remaining work allowance")
        work = reducer.work(MappingProxyType(json.loads(point.parameters)), width)
        if type(work) is not int or work < 0:
            raise ValueError(f"reduction {point.reducer!r} must register a nonnegative integer work")
        total += work
        if total > allowance:
            raise ValueError(f"reductions through point {point.id!r} register {total} work units, more than the "
                             f"remaining allowance {allowance} of the calling Method; refused before acquisition")
    return total


def execute_reduction(point, state, *, bindings, context=None):
    """Run the registered reduction of ``point`` once on the state saved at that point.

    ``context`` is the ``ReductionContext`` of the producing receipt. A
    reducer registered with ``receives_context`` receives it as the keyword
    ``context``, and the call refuses without one; any other reducer is
    called with the state, parameters and bindings only.

    The state can be shared with other points at the same boundary, so the
    reducer receives a view whose own array and every array it is based on
    are read-only, and no copy of the state is made: the reducer cannot
    write the state through that view or re-enable writing on it. A
    registered reducer must not change the writability of the arrays its
    view is based on. The returned arrays must match the registered
    components' dtypes and shapes; each is returned read-only.
    """
    import numpy as np

    outputs = reducer_outputs(point.reducer, point.parameters)
    owner = base = np.asarray(state)
    while isinstance(base, np.ndarray):
        base.flags.writeable = False
        base = base.base
    reducer = registered_reducer(point.reducer)
    extra = {}
    if reducer.receives_context:
        if not isinstance(context, ReductionContext):
            raise ValueError(f"reduction {point.reducer!r} needs the ReductionContext of its producing receipt")
        extra["context"] = context
    result = reducer.execute(owner.view(), MappingProxyType(json.loads(point.parameters)), bindings, **extra)
    if type(result) is not tuple or len(result) != len(outputs):
        raise ValueError(f"reduction {point.reducer!r} returned another number of components than it registered")
    arrays = []
    for array, output in zip(result, outputs, strict=True):
        array = np.array(array, copy=True)
        if array.dtype != np.dtype(output.dtype) or array.shape != output.shape:
            raise ValueError(f"reduction {point.reducer!r} returned a component differing from its registered "
                             "dtype and shape")
        array.flags.writeable = False
        arrays.append(array)
    return tuple(arrays)


def _canonical_parameters(value):
    """Canonical compact sorted-key JSON text of one reducer parameter mapping."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class ReadoutView(Record):
    """Optional reference to the selected coherent readout tail and its bound wire mapping.

    Its inverse restores the continuation state in ideal arithmetic. The
    receipt includes every executed forward and inverse operation.

    Attributes:
        tail: Definition ID of the selected coherent readout tail.
        inverse: Definition ID of the selected exact inverse of that tail,
            which reverses and adjoints the tail's actual operation sequence.
        wires: Bound logical wires of the tail, in the tail's own port order.
    """

    tail: Text
    inverse: Text
    wires: tuple[Count, ...]

    @model_validator(mode="after")
    def _mapping(self):
        if not self.wires or len(set(self.wires)) != len(self.wires):
            raise ValueError("a readout view needs its inverse and a nonempty distinct wire mapping")
        return self


class ObservationPoint(Record):
    """One observation point of an exact trajectory.

    A point observes the selected coherent body at one boundary, optionally
    through a reversible readout view. Its readout kind is
    ``pauli_expectation`` with ``labels``, ``probabilities`` with ``qubits``,
    ``amplitudes`` with its ``AmplitudeReadout`` output/projection declaration
    (the kind name alone does not determine its selected dimension, frame,
    masses or output bytes), or ``reduction`` with a registered ``reducer``
    name and its ``parameters``.

    Attributes:
        id: Point identity, unique within its trajectory.
        position: Number of bound logical operations preceding an observation
            point. Zero observes the initial state, and the body length
            observes its final state. Resolve an end-position shorthand before
            native preparation. Lowering preserves this boundary when it
            expands operations and, on a backend that executes readout views,
            inserts them. None is the
            end-position shorthand.
        kind: pauli_expectation, probabilities, amplitudes or reduction.
        labels: Ordered Pauli labels for this point, with one character per
            logical qubit and qubit zero rightmost. Labels are distinct within
            a point. The same label at another point denotes another returned
            value.
        qubits: Ordered logical qubits of this point's probability marginal,
            least significant first in each returned outcome. A reversible
            view acts before the marginal is saved and is undone before
            continuation.
        amplitudes: The existing amplitude output/projection declaration of
            an amplitude point.
        reducer: Registered reducer name of a reduction point.
        parameters: Canonical sorted-key compact JSON object of the reducer
            parameters; a mapping is accepted and canonicalized. Other kinds
            keep ``{}``.
        view: Optional reference to the selected coherent readout tail and
            its bound wire mapping. Its inverse restores the continuation
            state in ideal arithmetic. The receipt includes every executed
            forward and inverse operation.
    """

    id: Text
    position: Count | None = None
    kind: Literal["pauli_expectation", "probabilities", "amplitudes", "reduction"]
    labels: tuple[Text, ...] = ()
    qubits: tuple[Count, ...] = ()
    amplitudes: AmplitudeReadout | None = None
    reducer: Text | None = None
    parameters: str = "{}"
    view: ReadoutView | None = None

    @field_validator("parameters", mode="before")
    @classmethod
    def _parameters(cls, value):
        """Admit a JSON object with a deterministic identity: canonical sorted-key compact text."""
        if isinstance(value, Mapping):
            try:
                return _canonical_parameters(dict(value))
            except (TypeError, ValueError) as error:
                raise ValueError("reducer parameters must be a JSON-encodable mapping") from error
        if type(value) is not str:
            raise ValueError("reducer parameters must be a JSON object")
        try:
            decoded = json.loads(value)
        except ValueError as error:
            raise ValueError("reducer parameters must be a JSON object") from error
        if type(decoded) is not dict or _canonical_parameters(decoded) != value:
            raise ValueError("reducer parameters must be canonical sorted-key compact JSON of an object")
        return value

    @model_validator(mode="after")
    def _point(self):
        """Admit only the detail fields that the point's readout kind uses.

        Pauli labels are nonempty, distinct and use IXYZ; a probability
        marginal names at least one wire (``q_k >= 1``), each once; an
        amplitude point carries its declaration; a reduction names its reducer.
        """
        reduction = self.reducer is not None or self.parameters != "{}"
        if self.kind == "pauli_expectation":
            if (not self.labels or len(set(self.labels)) != len(self.labels) or self.qubits
                    or self.amplitudes is not None or reduction):
                raise ValueError("a Pauli point requires distinct labels and no marginal, amplitude or reducer")
            if any(set(label) - set("IXYZ") for label in self.labels):
                raise ValueError("Pauli labels use IXYZ")
        elif self.kind == "probabilities":
            if (not self.qubits or len(set(self.qubits)) != len(self.qubits) or self.labels
                    or self.amplitudes is not None or reduction):
                raise ValueError("a probability point requires distinct marginal qubits (q_k >= 1) "
                                 "and no labels, amplitude or reducer")
        elif self.kind == "amplitudes":
            if self.amplitudes is None or self.labels or self.qubits or reduction:
                raise ValueError("an amplitude point requires its declaration and no labels, marginal or reducer")
        elif self.reducer is None or self.labels or self.qubits or self.amplitudes is not None:
            raise ValueError("a reduction point requires a reducer name and no labels, marginal or amplitude")
        return self


def point_items(point, *, width=None, max_items=None, max_items_source=None):
    """Return the number of logical items ``L_k`` one trajectory point produces.

    This is the one per-point item count that the JSON-envelope reservation
    and the array reservation both use. By the NWQLib admission law, for one
    declared trajectory,
    ``N_items = sum_k L_k``. A Pauli point contributes its number of distinct
    requested labels. A probability marginal on ``q_k`` selected wires
    contributes ``2**q_k`` logical outcomes regardless of its eventual sparse
    stored-entry count. An amplitude point contributes the dimension of its
    selected output declaration. A reduction contributes the output
    cardinality computed by its registered shape function from the validated
    parameters. Two distinct views at one boundary contribute both payloads.
    Positions are neither repetitions nor shots.

    For a marginal and nonnegative remaining item capacity ``M_items``, the
    point rejects when ``q_k >= M_items.bit_length()`` before forming
    ``1 << q_k``; the existing nonempty-marginal domain requires ``q_k >= 1``.

    Args:
        point: The ObservationPoint.
        width: Total logical quantum width, or None when unresolved. A
            resolved width checks full-width labels, marginal and view wires
            and the amplitude declaration.
        max_items: Remaining item capacity, or None for no capacity check.
        max_items_source: Optional clause naming where ``max_items`` comes from.
    """
    source = "" if max_items_source is None else f", {max_items_source}"
    if width is not None:
        if any(len(label) != width for label in point.labels):
            raise ValueError("Pauli labels must match the entire logical circuit width")
        if any(qubit >= width for qubit in point.qubits):
            raise ValueError("probability qubits must belong to the circuit")
        if point.view is not None and any(wire >= width for wire in point.view.wires):
            raise ValueError("readout view wires must belong to the circuit")
    if point.kind == "pauli_expectation":
        items = len(point.labels)
    elif point.kind == "probabilities":
        # 2**bit_length(m) > m, so q >= max(1, bit_length(m)) implies 2**q > m.
        if max_items is not None and len(point.qubits) >= max(1, max_items.bit_length()):
            raise ValueError(f"trajectory point {point.id!r}: a probability marginal on {len(point.qubits)} qubits "
                             f"has 2**{len(point.qubits)} entries, more than max_items={max_items}{source}")
        items = 1 << len(point.qubits)
    elif point.kind == "amplitudes":
        if width is not None and point.amplitudes.width != width:
            raise ValueError("amplitude declaration must match the unmeasured logical layout")
        items = point.amplitudes.output.basis.dimension
    else:
        items = sum(output.items for output in reducer_outputs(point.reducer, point.parameters))
    if max_items is not None and items > max_items:
        raise ValueError(f"trajectory point {point.id!r} can have {items} items, more than max_items={max_items}{source}")
    return items


def point_array_reservation(point, *, width=None):
    """Pre-acquisition numerical payload bytes of one point whose values are array-backed.

    By the NWQLib array payload law, for array-backed components,
    actual numerical payload bytes are ``8*sum(unindexed_real L_k) +
    16*sum(complex L_k) + 8*sum(indexed (W_k+1)*s_k)``. Before acquisition,
    each dense or adaptively sparse marginal reserves ``8*2**q_k`` payload
    bytes. The adaptive encoding chooses indexed storage only when
    ``8*(W_k+1)*s_k < 8*2**q_k``, otherwise dense storage, and records the
    actual encoding on completion. A reduction reserves its registered
    output's actual dtype and shape (``ReducerOutput.payload_bytes``). So a
    Pauli point reserves eight bytes per real value, a marginal ``8*2**q_k``,
    an amplitude point sixteen bytes per complex128 amplitude. Pauli/scalar
    values that still use JSON keep their JSON prototype envelope, which
    ``_prepared_execution._admit_readout`` charges now; this array
    reservation is the law for the array representation.
    """
    if point.kind == "reduction":
        return sum(output.payload_bytes for output in reducer_outputs(point.reducer, point.parameters))
    items = point_items(point, width=width)
    return (16 if point.kind == "amplitudes" else 8) * items


class ReadoutDetails(Record):
    """Method-owned readout details; a selected IR batch owns acquisition counts/kind.

    The one-endpoint readout (``position``, ``labels``, ``qubits``) is the
    one-point case of a trajectory. A populated ``positions`` belongs only to
    the ``trajectory`` kind, which leaves the one-endpoint fields empty, so a
    Plan never declares two competing schedules.

    Attributes:
        amplitudes: Optional selected amplitude output/projection declaration.
        estimate: Optional weighted-observable provider estimate specification.
        position: Number of bound logical operations preceding an observation
            point. Zero observes the initial state, and the body length
            observes its final state. Resolve an end-position shorthand before
            native preparation. Lowering preserves this boundary when it
            expands operations and, on a backend that executes readout views,
            inserts them. None is the
            end-position shorthand.
        labels: Ordered Pauli or host-scalar labels defining returned value association.
        qubits: Explicit measured/marginal wire positions in least-significant-first order.
        population: Requested unconditional or native_conditioned measurement population.
        padding: Meaning of padding and which returned coordinates belong to the selected system.
        positions: Ordered observation-point declarations for one exact
            trajectory. Each point identifies a boundary in the selected
            coherent body, its readout kind, and its ordered labels or
            marginal qubits. A point may name a selected reversible readout
            view. Points share one preparation and acquisition. Their order
            and payload sizes are part of the Plan.
    """

    amplitudes: AmplitudeReadout | None = None
    estimate: ObservableEstimateSpec | None = None
    position: Count | None = None
    labels: tuple[Text, ...] = ()
    qubits: tuple[Count, ...] = ()
    population: Literal["unconditional", "native_conditioned"] = "unconditional"
    padding: Text = "no padding; all returned bit patterns belong to the declared registers"
    positions: tuple[ObservationPoint, ...] = ()
    __slots__ = ("_point_index",)

    @model_validator(mode="after")
    def _schedule(self):
        """Admit one ordered point schedule: unique point IDs and nondecreasing boundaries.

        The end-position shorthand (None) resolves to the body length, the
        largest boundary, so a point with an explicit boundary cannot follow it.
        """
        if not self.positions:
            return self
        if len({point.id for point in self.positions}) != len(self.positions):
            raise ValueError("trajectory point IDs must be unique")
        previous = 0
        for point in self.positions:
            if point.position is not None and (previous is None or point.position < previous):
                raise ValueError("trajectory boundaries must be nondecreasing; an explicit boundary cannot "
                                 "follow the end shorthand")
            previous = point.position
        if (self.position is not None or self.labels or self.qubits or self.amplitudes is not None
                or self.estimate is not None):
            raise ValueError("a trajectory schedule and a one-endpoint readout declaration cannot both be populated")
        return self

    def boundaries(self, body_length):
        """Resolve every point's boundary against the selected body length.

        Requires ``0 <= position <= body_length`` for each point and resolves
        the end-position shorthand to ``body_length``. Resolution precedes
        native preparation.
        """
        if type(body_length) is not int or body_length < 0:
            raise ValueError("body length must be a nonnegative integer")
        resolved = tuple(body_length if point.position is None else point.position for point in self.positions)
        if any(boundary > body_length for boundary in resolved):
            raise ValueError(f"a trajectory point lies beyond the selected body of {body_length} operations")
        return resolved

    def point(self, point_id):
        """Return the declared point with this ID."""
        return self.positions[self.point_index(point_id)]

    def point_index(self, point_id):
        """Return the schedule position of the point with this ID.

        The ID-to-position map is built on the first call, in one pass, and
        kept in a slot that equality, identity and serialization do not see,
        so joining K point chunks to one declaration takes O(K) lookups.
        """
        try:
            index = object.__getattribute__(self, "_point_index")
        except AttributeError:
            index = {point.id: position for position, point in enumerate(self.positions)}
            object.__setattr__(self, "_point_index", index)
        try:
            return index[point_id]
        except (KeyError, TypeError):
            raise ValueError(f"trajectory declares no point {point_id!r}") from None


class ObservationSpec(ReadoutDetails):
    """Explicit native readout; bit/qubit positions are least significant first.

    Pauli observations occur before the selected logical instruction (None is
    the end). Counts use the Program's computational measurement positions;
    probability marginals occur after final measurement removal. No state dump.

    An exact trajectory observation declares non-destructive readouts of one
    selected coherent evolution. Its raw-shot count is zero. Each result is
    associated with its point ID and label or outcome. Saved values are
    reductions of the simulator state at those points. Sampled observations
    describe the Program's actual measurement populations and use their
    declared positive shot counts.

    A trajectory declares deterministic non-destructive observations of the
    selected coherent body, with admitted populations. Zero raw shots does not
    admit measurement-conditioned, reset, postselected, noisy-trajectory or
    outcome-dependent evolution under this contract. Existing host,
    amplitude-output, sampled and provider-estimate populations keep their own
    meanings.

    Coalescing requires the same selected body, boundary, bound view, readout
    kind and population. Two references share a saved scalar only when they
    also select the same label, outcome/amplitude declaration or reducer with
    the same parameters. A union of distinct requested labels still contains
    one item per distinct label. Preserve every point-to-result association
    when sharing a datum. Admission counts every declared point's
    items, ``N_items = sum_k L_k`` (``point_items``).

    Attributes:
        kind: Selected pauli_expectation, counts, probabilities, host_scalars, amplitudes, estimated_observable or trajectory readout.
        shots: Positive raw shot request for counts; zero for readouts without a prescribed raw-shot population, including a trajectory.
    """

    kind: Literal["pauli_expectation", "counts", "probabilities", "host_scalars", "amplitudes", "estimated_observable",
                  "trajectory"]
    shots: Count = 0

    @model_validator(mode="after")
    def _readout(self):
        """Admit only the detail fields that each readout kind uses.

        - ``trajectory``: at least one point, zero shots and the unconditional
          population; the one-endpoint fields stay empty. An empty observation
          request is algebraic work with no acquisition, so it declares no
          experiment.
        - ``estimated_observable``: an ``estimate``, zero shots, and no
          amplitudes, labels, qubits or position, on the unconditional final
          state.
        - ``amplitudes``: an ``amplitudes`` declaration and the same
          exclusions.
        - ``host_scalars``: distinct labels naming the statistics, with no
          qubits, shots or position, and the unconditional population.
        - ``pauli_expectation``: distinct IXYZ labels and an optional
          position, with no qubits or shots.
        - ``counts``: ``1 <= shots <= 2**63 - 1`` (the int64 count domain,
          ``_validation.MAX_COUNT``), no labels, qubits or position, and the
          unconditional population. The Program's classical layout fixes the
          bits.
        - ``probabilities``: distinct qubits for the marginal, with no
          labels, shots or position.

        Only a trajectory carries ``positions``.
        """
        if self.kind == "trajectory":
            if not self.positions:
                raise ValueError("a trajectory needs at least one observation point; an empty request is "
                                 "algebraic work with no acquisition")
            if self.shots:
                raise ValueError("a trajectory has zero raw shots")
            if self.population != "unconditional":
                raise ValueError("a trajectory observes deterministic coherent evolution; measurement-conditioned "
                                 "populations are not admitted")
            return self
        if self.positions:
            raise ValueError("observation points belong only to trajectory readout")
        if self.kind == "estimated_observable":
            if (self.estimate is None or self.amplitudes is not None or self.labels or self.qubits or self.shots
                    or self.position is not None or self.population != "unconditional"):
                raise ValueError("estimated observable requires its weighted operator/precision and unconditional final state")
            return self
        if self.estimate is not None:
            raise ValueError("weighted estimate declaration belongs only to estimated-observable readout")
        if self.kind == "amplitudes":
            if self.amplitudes is None or self.labels or self.qubits or self.shots or self.position is not None or self.population != "unconditional":
                raise ValueError("amplitudes require their explicit declaration and unconditional trajectory")
            return self
        if self.amplitudes is not None:
            raise ValueError("amplitude declaration belongs only to amplitude readout")
        if self.kind == "host_scalars":
            if (not self.labels or len(set(self.labels)) != len(self.labels) or self.qubits
                    or self.shots or self.position is not None or self.population != "unconditional"):
                raise ValueError("host scalars require distinct physical statistics, without quantum readout")
        elif self.kind == "pauli_expectation":
            if not self.labels or len(set(self.labels)) != len(self.labels) or self.qubits or self.shots:
                raise ValueError("Pauli readout requires distinct labels and no shots/qubit marginal")
            if any(set(label) - set("IXYZ") for label in self.labels):
                raise ValueError("Pauli labels use IXYZ")
        elif self.labels or self.position is not None:
            raise ValueError("instruction position and labels belong to Pauli readout")
        elif self.kind == "counts":
            if not 1 <= self.shots <= MAX_COUNT or self.qubits or self.population != "unconditional":
                raise ValueError(
                    f"counts require 1 <= shots <= {MAX_COUNT}, "
                    "the unconditional population, and the Program's classical layout"
                )
        elif not self.qubits or len(set(self.qubits)) != len(self.qubits) or self.shots:
            raise ValueError("probabilities require distinct qubits and no shots")
        return self

    def point_observation(self, point_id):
        """The one-point readout of a trajectory point with a statistic record type.

        Pauli, probability and amplitude points map to the existing readout
        kinds, so their saved values keep those kinds' records and checks. A
        reduction point's readout is the one-point trajectory of that point,
        and its saved values are ``ReducedValues`` records.
        """
        return self._point_readout(self.point(point_id))

    def point_observations(self):
        """Map every point ID to its one-point readout, in one pass over the schedule."""
        return {point.id: self._point_readout(point) for point in self.positions}

    @classmethod
    def point_readout_fields(cls, point):
        """The field tree of one point's one-point readout, without constructing an ObservationSpec.

        This is the mapping of ``point_observation``, which builds its spec
        from these fields, so a byte bound of the stored spec can be formed
        from the same fields without constructing the spec.
        It accepts the already visited point object, avoiding an ID lookup.
        Pauli, probability and amplitude points take their kind, labels,
        qubits, amplitude declaration and the unconditional population, and a
        Pauli point also its position. Their point ID, view and non-Pauli
        position stay associated only through the receipt, point ID,
        boundary and trajectory identity. A reduction point's readout is the
        singleton trajectory of that entire point, so it copies the point's
        ID, declared position, reducer, parameters and view. Every other
        field keeps its default.
        """
        fields = {name: field.default for name, field in cls.model_fields.items() if name != "kind"}
        if point.kind == "reduction":
            fields.update(kind="trajectory", positions=(point,), population="unconditional")
            return fields
        fields.update(kind=point.kind, labels=point.labels, qubits=point.qubits,
                      amplitudes=point.amplitudes, population="unconditional",
                      position=point.position if point.kind == "pauli_expectation" else None)
        return fields

    def _point_readout(self, point):
        return ObservationSpec(**self.point_readout_fields(point))

    def unsupported_schedule(self):
        """Describe this observation's multi-point schedule, views and reductions, or return None.

        Adapters name this description when they reject a schedule they do not
        execute. Existing single-point readouts return None.
        """
        if self.kind != "trajectory":
            return None
        features = [f"a {len(self.positions)}-point trajectory schedule"]
        views = sum(point.view is not None for point in self.positions)
        if views:
            features.append(f"{views} readout view(s)")
        reducers = sorted({point.reducer for point in self.positions if point.kind == "reduction"})
        if reducers:
            features.append("reduction(s) " + ", ".join(map(repr, reducers)))
        return ", ".join(features)

    def reject_unsupported_schedule(self, backend):
        """Reject a trajectory schedule before native construction or submission, naming ``backend``.

        Preserve each adapter's supported single-point readouts. Reject new
        multi-point schedules, views and reductions that it does not execute
        before native construction or submission, naming the backend and
        unsupported feature.
        """
        feature = self.unsupported_schedule()
        if feature is not None:
            raise ValueError(f"{backend} does not execute {feature}; it executes the existing single-endpoint "
                             "readout kinds only")


def readout_shape(*, kind, details, width, classical_width, repetitions=None, max_items=None,
                  max_items_source=None):
    """Return the number of sufficient-statistic entries a readout produces (a bound for counts), or None.

    The entries are one value per estimated observable, per amplitude of
    the declared output, per host scalar label or per Pauli label, ``2**k``
    probabilities for a k-qubit marginal, and for counts the bound
    ``min(repetitions, 2**classical_width)`` on the number of distinct
    bitstrings. A trajectory produces ``N_items = sum_k L_k`` over its points
    (``point_items``), checked against the remaining capacity point by point.
    The same call checks that the detail fields agree with the layout.

    width is the sum of all selected circuit quantum registers, and counts use
    the actual classical bits layout. This does not select/prepare a Program or
    require execution-only positive repetitions or an instruction position.

    Args:
        kind: Readout kind, or None for a node without readout.
        details: ReadoutDetails or ObservationSpec of the readout.
        width: Total quantum width, or None when unresolved.
        classical_width: Total classical bit width, or None when unresolved.
        repetitions: Shots for counts, or None when unresolved.
        max_items: Optional admitted entry limit, checked without forming ``2**k`` or ``2**classical_width``.
        max_items_source: Optional clause that a refusal appends to name where
            ``max_items`` comes from, such as a Run's ``max_data_bytes``.

    Returns:
        The entry count, or None when counts lack a resolved classical width
        or repetition count, or when ``kind`` is None.
    """
    for size in (width, classical_width, repetitions, max_items):
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError("readout dimensions/populations require nonnegative integers or unknown")
    if kind != "trajectory" and details.positions:
        raise ValueError("observation points belong only to trajectory readout")
    if (kind == "estimated_observable") != (details.estimate is not None):
        raise ValueError("estimated readout kind and weighted declaration must agree")
    if kind in {"counts", "probabilities"} and details.position is not None:
        raise ValueError("instruction position belongs only to Pauli readout")
    if kind == "trajectory":
        if not details.positions or details.position is not None or details.labels or details.qubits or details.amplitudes is not None:
            raise ValueError("a trajectory declares its nonempty point schedule and no competing one-endpoint readout")
        if classical_width or details.population != "unconditional":
            raise ValueError("a trajectory observes deterministic coherent evolution without measurement registers")
        items, remaining = 0, max_items
        for point in details.positions:
            count = point_items(point, width=width, max_items=remaining, max_items_source=max_items_source)
            items += count
            if remaining is not None:
                remaining -= count
        return items
    if kind == "estimated_observable":
        if (classical_width or details.labels or details.qubits or details.amplitudes is not None
                or details.position is not None or details.population != "unconditional"
                or width is not None and any(len(label) != width for label in details.estimate.labels)):
            raise ValueError("estimated Pauli sum must match the unmeasured logical circuit width")
        items = 1
    elif kind == "amplitudes":
        declaration = details.amplitudes
        if declaration is None or width != declaration.width or classical_width:
            raise ValueError("amplitude declaration must match the unmeasured logical layout")
        items = declaration.output.basis.dimension
    elif kind == "host_scalars":
        if width or classical_width or details.qubits or details.position is not None or details.population != "unconditional":
            raise ValueError("host readout has no quantum/classical register population")
        items = len(details.labels)
    elif kind == "pauli_expectation":
        if (not details.labels or len(set(details.labels)) != len(details.labels)
                or any(set(label) - set("IXYZ") for label in details.labels) or details.qubits):
            raise ValueError("Pauli readout requires distinct IXYZ labels without a qubit marginal")
        if width is not None and any(len(label) != width for label in details.labels):
            raise ValueError("Pauli labels must match the entire logical circuit width")
        items = len(details.labels)
    elif kind == "probabilities":
        if not details.qubits or len(set(details.qubits)) != len(details.qubits) or details.labels:
            raise ValueError("probabilities require distinct circuit qubits without Pauli labels")
        if width is not None and max(details.qubits) >= width:
            raise ValueError("probability qubits must belong to the circuit")
        # 2**bit_length(m) > m, so k >= max(1, bit_length(m)) implies 2**k > m.
        # This rejects before forming 2**k for a wide marginal.
        if max_items is not None and len(details.qubits) >= max(1, max_items.bit_length()):
            source = "" if max_items_source is None else f", {max_items_source}"
            raise ValueError(f"a probability marginal on {len(details.qubits)} qubits has 2**{len(details.qubits)} "
                             f"entries, more than max_items={max_items}{source}")
        items = 1 << len(details.qubits)
    elif kind == "counts":
        if details.labels or details.qubits or details.population != "unconditional":
            raise ValueError("counts require the actual unconditional classical layout")
        if classical_width == 0:
            raise ValueError("counts require explicit Program measurement registers")
        if classical_width is None or repetitions is None:
            return None
        # At most min(shots, 2**classical_width) distinct bitstrings. When
        # classical_width >= bit_length(shots), 2**classical_width > shots, so
        # the minimum is shots and the power is never formed.
        items = repetitions if classical_width >= repetitions.bit_length() else min(repetitions, 1 << classical_width)
    elif kind is None:
        return None
    else:
        raise ValueError("unsupported readout kind")
    if max_items is not None and items > max_items:
        source = "" if max_items_source is None else f", {max_items_source}"
        raise ValueError(f"the {kind} readout can have {items} sufficient-statistic entries, more than "
                         f"max_items={max_items}{source}")
    return items


class Experiment(Record):
    """Explicit batch selector with details, or a direct non-batch observation.

    Attributes:
        name: Unique experiment name inside the Plan.
        setting: Readout setting label of a direct experiment. None for a batch experiment, whose batch setting supplies it.
        observation: Declared readout of a direct experiment. None for a batch experiment.
        readout: Method-owned readout details of a batch experiment. The batch itself owns kind and repetitions.
        batch: Definition ID of the selected MeasurementBatch, or None for a direct experiment.
        setting_index: Index of the selected setting within that batch.
    """

    name: Text
    setting: Text | None = None
    observation: ObservationSpec | None = None
    readout: ReadoutDetails | None = None
    batch: Text | None = None
    setting_index: Count | None = None

    @model_validator(mode="after")
    def _selector(self):
        if (self.batch is None) != (self.setting_index is None):
            raise ValueError("batch selection needs both batch and setting_index")
        if self.batch is None:
            if self.setting is None or self.observation is None or self.readout is not None:
                raise ValueError("non-batch experiment requires setting and observation only")
        elif self.setting is not None or self.observation is not None or self.readout is None:
            raise ValueError("batch experiment requires readout details; IR owns setting/kind/shots")
        return self

    def _resolved_observation(self, admission, *, construction_id: ContentID | None = None) -> tuple[str, ObservationSpec]:
        """Resolve the admitted Program, binding batch amplitudes to its selected construction.

        ``construction_id`` is required for a probability batch carrying an
        amplitude declaration. Preparation, analysis and restoration supply
        the same selected child identity; direct readouts keep their declaration.
        """
        if self.batch is None:
            return self.setting, self.observation
        batch = admission.nodes[self.batch]
        if batch.observation_kind is None or batch.repetitions is None:
            raise ValueError("batch observation kind and repetitions must be known before prepare")
        count = admission.integer(batch.repetitions,
                                  admission.expressions(admission.binding_map(admission.p.bindings)),
                                  "acquisition repetitions")
        if count is None or count < 1 or (batch.observation_kind != "counts" and count != 1):
            raise ValueError("prepare requires positive counts repetitions or exactly one observable evaluation")
        details = self.readout.model_dump(exclude_computed_fields=True, exclude={"parent_id", "schema_version"})
        kind = batch.observation_kind
        if kind == "probabilities" and self.readout.amplitudes is not None:
            if construction_id is None:
                raise ValueError("amplitude batch readout requires the selected construction identity")
            kind = "amplitudes"
            details["amplitudes"] = self.readout.amplitudes.revise(construction_id=construction_id)
        return batch.settings[0].label, ObservationSpec(
            kind=kind, shots=count if kind == "counts" else 0, **details)


class RandomState(Record):
    """Resolved entropy and exact PCG64 states; no array or live generator.

    Attributes:
        entropy: Root nonnegative seed entropy used by NumPy SeedSequence.
        spawn_key: Child-stream path under the original root seed.
        method_state: Serialized PCG64 state for scientific selection decisions.
        backend_state: Serialized independent PCG64 stream used to generate backend seeds.
    """

    entropy: int | tuple[int, ...]
    spawn_key: tuple[int, ...] = ()
    method_state: Text
    backend_state: Text

    @model_validator(mode="after")
    def _states(self):
        entropy = (self.entropy,) if type(self.entropy) is int else self.entropy
        if not entropy or any(type(value) is not int or value < 0 for value in entropy):
            raise ValueError("random entropy must contain nonnegative integers")
        for text in (self.method_state, self.backend_state):
            value = json.loads(text)
            if value.get("bit_generator") != "PCG64":
                raise ValueError("random streams require PCG64 states")
        return self


class RandomStreams:
    """Separate method and backend draws from one recorded SeedSequence.

    The root SeedSequence spawns two independent PCG64 streams. The method
    stream serves scientific selection (for example a default random
    initialization). The backend stream only generates execution seeds. Keeping
    them apart means that the number of backend seeds drawn never shifts the
    method's draws, so one seed reproduces the same selection however often
    it is executed. ``snapshot`` records the exact generator states in the
    Plan, and ``restore`` resumes them without replaying earlier draws.
    """

    def __init__(self, seed=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        import numpy as np
        self._initialize(np.random.SeedSequence(seed))

    def _initialize(self, sequence):
        import numpy as np
        self.entropy, self.spawn_key = sequence.entropy, tuple(sequence.spawn_key)
        method, backend = sequence.spawn(2)
        self.method = np.random.Generator(np.random.PCG64(method))
        self.backend = np.random.Generator(np.random.PCG64(backend))

    @classmethod
    def from_sequence(cls, sequence):
        result = object.__new__(cls)
        result._initialize(sequence)
        return result

    @classmethod
    def restore(cls, snapshot):
        """Resume both streams at the saved PCG64 states, without replaying earlier draws."""
        if type(snapshot) is not RandomState:
            raise TypeError("restore requires an admitted RandomState")
        import numpy as np
        result = object.__new__(cls)
        result.entropy, result.spawn_key = snapshot.entropy, snapshot.spawn_key
        for name, text in (("method", snapshot.method_state), ("backend", snapshot.backend_state)):
            # The seed 0 only creates the object. The saved state replaces it.
            generator = np.random.Generator(np.random.PCG64(0))
            generator.bit_generator.state = json.loads(text)
            setattr(result, name, generator)
        return result

    def snapshot(self):
        return RandomState(entropy=self.entropy, spawn_key=self.spawn_key,
            method_state=json.dumps(self.method.bit_generator.state, sort_keys=True, separators=(",", ":")),
            backend_state=json.dumps(self.backend.bit_generator.state, sort_keys=True, separators=(",", ":")))

    def next_seed(self):
        """Draw one backend seed in [0, MAX_RUNTIME_SEED], the uint32 domain of RuntimeOptions."""
        return int(self.backend.integers(0, 2**32, dtype="uint32"))


class Plan(Record):
    """One immutable selected experiment, with its original live input access.

    Private native blocks and caches are bound once by the method. The portable
    record is computed only on request and cannot reconstruct live inputs.

    A Plan is the selection a Method made once for one Problem, output,
    execution mode, shot request and seed. It holds the construction,
    experiments, reconstruction data and error model. A comparison or search
    row holds at most one Plan (a blocked row holds none), and selecting a
    row returns that original row with the same Plan object. Preparation,
    execution, analysis, saving and loading all consume this selection and
    never run the Method's selection again. Any change to a scientific
    choice, including shots or a Program binding that conflicts with the
    selected one, needs a new Plan with a new identity. This is what lets a
    Result, its observations and an original resource forecast name the
    selection they belong to by ``plan_id``. If selection could be repeated
    silently, data could end up attached to a construction it was not
    acquired from. A Run's ExecutionLimits, the settings of a later
    analysis and a later assessment criterion are not Plan fields, so they
    do not change the identity. The public contract is docs/scientist.md,
    "Execute the selected row".

    The Plan keeps one recent resolution and one recent selected construction.
    Each slot recognizes its requested bindings and the selected Program's
    canonical bindings. A miss replaces that slot, and a later request for an
    evicted point repeats its normal resolution and admission. Both slots share
    existing Plan declarations and may share the same selected Program.
    The two-slot cap bounds memo cardinality. Storage per slot depends on the
    selected graph, bindings and readout declarations.

    Attributes:
        problem: Original scientific input, including its admitted access representation.
        method: Immutable configured Method that selected this experiment.
        output: Requested scientific quantity and its physical frame.
        execution: Selected ``quantum`` or ``classical`` computational route.
        shots: Requested positive shots per sampled acquisition; None selects exact readout.
        selection_accuracy: Requested selection criterion, not an achieved error guarantee.
        randomness: Root seed and named stream states after selection.
        construction: Selected Program, native block declarations and host-kernel contracts.
        experiments: Ordered experiment definitions with unique names and readout meanings.
        reconstruction: Method-owned data needed to interpret the selected observations.
        error_model: Known and unavailable error sources; None means no selected model.
        facts: Framed facts established during selection with their stated conditions.
        assumptions: Mathematical or experimental premises kept without inventing validation.
        requirements: Unmet operational or scientific requirements reported by the Method.
    """

    schema_version: Literal[7] = 7
    problem: SerializeAsAny[InstanceOf[Record]]
    method: SerializeAsAny[InstanceOf[Record]]
    output: SerializeAsAny[InstanceOf[Record]]
    execution: Literal["quantum", "classical"]
    shots: Count | None = None
    selection_accuracy: Accuracy | None = None
    randomness: RandomState
    construction: SelectedConstruction
    experiments: tuple[Experiment, ...] = ()
    reconstruction: Any = None
    error_model: ErrorModel | None = None
    facts: tuple[FramedFact, ...] = ()
    assumptions: tuple[Text, ...] = ()
    requirements: tuple[Text, ...] = ()
    _cache: dict = PrivateAttr(default_factory=dict)
    _blocks: tuple = PrivateAttr(default=())
    _native: dict = PrivateAttr(default_factory=dict)
    _bound: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _selection(self):
        if len({item.name for item in self.experiments}) != len(self.experiments):
            raise ValueError("experiment names must be unique within the Plan")
        if self.shots == 0:
            raise ValueError(f"shots must be a positive int or None for exact readout, got {self.shots!r}")
        if not all(callable(getattr(self.method, name, None)) for name in ("plan", "execute", "analyze", "error_model")):
            raise TypeError("Plan requires its configured scientific Method")
        return self

    @field_serializer("problem", "method", "output")
    def _describe(self, value):
        return dict(type=f"{type(value).__module__}.{type(value).__qualname__}",
                    fields=value.model_dump(mode="json", exclude_computed_fields=True))

    def to_record(self):
        """Project selected metadata; never acquire or serialize native data."""
        return self.model_dump(mode="json", exclude_computed_fields=True)

    def __str__(self):
        return (f"Plan(method={type(self.method).__name__}, problem={type(self.problem).__name__}, "
                f"output={type(self.output).__name__}, execution={self.execution}, "
                f"shots={self.shots}, experiments={len(self.experiments)})")

    @computed_field
    @property
    def content_id(self) -> str:
        # The identity covers the portable fields only. Problem, Method and
        # output enter with their concrete type names, so equal fields of
        # different classes stay distinct. Bound native blocks and caches are
        # private and excluded, which lets an archive restore the same
        # bindings under the same identity. Cached because the Plan is frozen.
        if "identity" not in self._cache:
            self._cache["identity"] = self.record_identity(self.to_record())
        return self._cache["identity"]

    @classmethod
    def record_identity(cls, fields):
        """Hash the v7 portable fields without constructing native inputs.

        The saved-evidence reader calls this on stored metadata to check a
        saved Plan identity without constructing the Plan or its native inputs.
        """
        data = dict(format="nwqlib.plan/7", type=f"{cls.__module__}.{cls.__qualname__}", fields=fields)
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @property
    def blocks(self):
        return self._blocks

    def _bind(self, *, blocks=(), **native):
        """Supported Method hook: bind the selected blocks and named live inputs once.

        Return this Plan without compiling, acquiring, or reconstructing its inputs.
        Archive readers restore the same selected bindings before returning the Plan.
        """
        if self._bound:
            raise ValueError("a Plan's live data is already bound")
        self._blocks = tuple(blocks)
        self._native = dict(native)
        self._bound = True
        return self

    def __eq__(self, other):
        """Compare concrete type and portable identity. Bound native data is ignored."""
        return type(self) is type(other) and self.content_id == other.content_id

    @property
    def _experiment_map(self):
        if "experiments" not in self._cache:
            self._cache["experiments"] = MappingProxyType({item.name: item for item in self.experiments})
        return self._cache["experiments"]

    @property
    def _parameter_map(self):
        if "parameters" not in self._cache:
            self._cache["parameters"] = MappingProxyType({item.name: item for item in self.construction.program.parameters})
        return self._cache["parameters"]

    @property
    def _binding_map(self):
        if "bindings" not in self._cache:
            self._cache["bindings"] = MappingProxyType({item.parameter: item.value for item in self.construction.program.bindings})
        return self._cache["bindings"]

    @property
    def _program_nodes(self):
        return self._program_admission.nodes

    @property
    def _program_admission(self):
        if "program_admission" not in self._cache:
            admission = _Admission(self.construction.program)
            admission.setup()
            self._cache["program_admission"] = admission
            self._cache["has_batches"] = admission.contains_batch[admission.p.root]
        return self._cache["program_admission"]

    @property
    def _construction_id(self):
        if "construction_id" not in self._cache:
            self._cache["construction_id"] = self.construction.content_id
        return self._cache["construction_id"]

    def resolve(self, experiment: str, *, bindings: tuple[Binding, ...] = ()) -> "Realization":
        """Bind only admitted Program choices, without changing the scientific selection.

        Args:
            experiment: Name of one of this Plan's experiments.
            bindings: Values for Program parameters that the Plan and the
                selected setting leave unbound, such as a range-axis point.
                A value that differs from an existing binding rejects.

        Repeated requests for a point reuse its selection while that point occupies
        the recent-resolution slot and the bindings match its stored or canonical
        bindings in order.
        An evicted point is resolved and admitted again.
        The memo affects reuse, while the immutable Plan and effective bindings
        determine the selected result.

        Returns:
            The Realization naming this Plan, the experiment and the complete
                effective bindings in parameter-name order. Nothing is prepared.
        """
        _, program = self._resolve_selection(experiment, bindings)
        return Realization(plan_id=self.content_id, experiment=experiment, bindings=program.bindings)

    def _effective_point(self, experiment, bindings):
        """Return the experiment and the merged parameter values of one point.

        The values merge, in order, the Plan's Program bindings, the selected
        setting's bindings and the explicit ``bindings``. Each is admitted in
        its declared domain. A name bound twice must have equal values, so a
        point can add choices but never override one. Overriding needs a new
        Plan. Every range axis must then have an on-grid value, and the
        Method's ``validate_point`` checks its own relations. No selected
        record is constructed here.
        """
        selected = self._experiment_map.get(experiment)
        if selected is None:
            raise ValueError("experiment is not admitted by this Plan")
        if len({item.parameter for item in bindings}) != len(bindings):
            raise ValueError("duplicate realization binding")
        nodes = self._program_nodes
        batch = None
        setting_bindings = ()
        if selected.batch is not None:
            batch = nodes.get(selected.batch)
            if not isinstance(batch, MeasurementBatch):
                raise ValueError("experiment must select a declared MeasurementBatch")
            if selected.setting_index >= len(batch.settings):
                raise ValueError("setting_index must select a kept setting")
            setting_bindings = batch.settings[selected.setting_index].bindings
        elif self._cache["has_batches"]:
            raise ValueError("MeasurementBatch requires an explicit batch selector")
        values = dict(self._binding_map)
        for item in (*setting_bindings, *bindings):
            parameter = self._parameter_map.get(item.parameter)
            if parameter is None:
                raise ValueError("realization binds an undeclared Program parameter")
            parameter.admit(item.value)
            if item.parameter in values and number(values[item.parameter]) != number(item.value):
                raise ValueError("conflicting Plan/Setting/explicit bindings require a new Plan revision")
            values.setdefault(item.parameter, item.value)
        if batch is not None:
            for axis in batch.axes:
                value = values.get(axis.parameter)
                if (type(value) is not int or not axis.start <= value < axis.stop
                        or (value - axis.start) % axis.step):
                    raise ValueError("provide a selected value inside each range axis")
        self.method.validate_point(self, selected, MappingProxyType(values))
        return selected, values

    def _resolve_selection(self, experiment, bindings):
        """Reuse the latest resolution and its canonical binding alias.

        This Plan keeps one resolution pair. A miss releases this slot before
        resolving the requested point. The stored pair owns its selected
        Experiment and Program, with their shared children.
        """
        raw = tuple(bindings)
        cached = self._cache.get("resolution_memo")
        if cached is not None:
            if experiment == cached[0] and (
                raw == cached[1] or raw == cached[2][1].bindings
            ):
                return cached[2]
        self._cache.pop("resolution_memo", None)
        del cached
        resolved = self._resolve_selection_once(experiment, raw)
        self._cache["resolution_memo"] = (resolved[0].name, raw, resolved)
        return resolved

    def _resolve_selection_once(self, experiment, bindings):
        """Use one merge rule and admit evidence inputs before constructing the point."""
        selected, values = self._effective_point(experiment, bindings)
        specialized = self._specialize_experiment(selected, MappingProxyType(values))
        if (type(specialized) is not Experiment or any(
                getattr(specialized, field) != getattr(selected, field)
                for field in ("name", "setting", "batch", "setting_index"))):
            raise ValueError("point readout must keep its declared experiment selector")
        selected = specialized
        return selected, self._selected_program(selected, values)

    def _specialize_experiment(self, experiment, values):
        """Method-owned bounded readout derivation; static selections are unchanged.

        The actual Method owns this scientific relation. The selected Experiment
        travels with the same Program through preparation, assessment and analysis.
        """
        return self.method.specialize_experiment(self, experiment, values)

    def _selected_kernels(self, experiment, program):
        """Method-owned bounded host declarations at the admitted point.

        A changed declaration keeps its template as parent and the same native
        input/implementation identities. Its actual scalar/work declaration is
        part of SelectedConstruction, consumed by the existing fold and host.
        """
        return self.method.selected_kernels(self, experiment, program)

    def _selected_program(self, selected, values):
        """Construct and validate the complete effective point after admission."""
        program = self.construction.program
        batch = self._program_nodes[selected.batch] if selected.batch is not None else None
        if batch is not None:
            from nwqlib.ir.selection import select_experiment
            if "program_id" not in self._cache:
                self._cache["program_id"] = program.content_id
            return select_experiment(self._program_admission, selected.batch, selected.setting_index,
                {axis.parameter: values[axis.parameter] for axis in batch.axes},
                bindings=tuple(Binding(parameter=k, value=values[k]) for k in sorted(values)),
                parent_id=self._cache["program_id"])
        elif values != dict(self._binding_map):
            program = program.bind(**values)
        ordered = tuple(sorted(program.bindings, key=lambda item: item.parameter))
        if ordered != program.bindings:
            program = program.revise(bindings=ordered)
        return program

    def _selected_construction(self, experiment, bindings):
        """Reuse the latest construction and its canonical binding alias.

        The single construction slot can share its Program with the resolution
        slot when both describe that point. A miss releases this slot before
        deriving the new construction. Its normal admission runs on each miss.
        """
        raw = tuple(bindings)
        cached = self._cache.get("construction_memo")
        if cached is not None:
            if experiment == cached[0] and (
                raw == cached[1] or raw == cached[2][1].program.bindings
            ):
                return cached[2]
        self._cache.pop("construction_memo", None)
        del cached
        resolved = self._selected_construction_once(experiment, raw)
        self._cache["construction_memo"] = (resolved[0].name, raw, resolved)
        return resolved

    def _selected_construction_once(self, experiment, bindings):
        """Derive the construction of one admitted point without reselecting it.

        Selected definitions and encodings are reused from the Plan, filtered
        to the point's Program. A Method may change a host-kernel declaration
        at a point only as a revision of its selected template with the same
        name, implementation, inputs and dependencies, so the point cannot
        substitute another kernel or input. When nothing changes, the Plan's
        own construction object is returned.
        """
        selected, program = self._resolve_selection(experiment, bindings)
        kernels = self._selected_kernels(selected, program)
        if type(kernels) is not tuple:
            raise TypeError("selected host declarations require an immutable tuple")
        if kernels != self.construction.kernels:
            if "kernels" not in self._cache:
                self._cache["kernels"] = {item.name: item for item in self.construction.kernels}
            for kernel in kernels:
                template = self._cache["kernels"].get(kernel.name) if type(kernel) is SelectedKernel else None
                if template is None or (kernel != template and (
                        kernel.parent_id != template.content_id or any(
                            getattr(kernel, field) != getattr(template, field)
                            for field in ("name", "implementation", "inputs", "dependencies")))):
                    raise ValueError("point host declaration differs from its exact selected template/input")
        if program is self.construction.program and kernels == self.construction.kernels:
            return selected, self.construction
        if "selections" not in self._cache:
            self._cache["selections"] = {item.signature.name: item for item in self.construction.selections}
            encodings = {}
            for encoding in self.construction.encodings:
                encodings.setdefault(encoding.root, []).append(encoding)
            self._cache["encodings"] = encodings
        construction = SelectedConstruction(parent_id=self._construction_id, program=program,
            selections=tuple(self._cache["selections"][signature.name] for signature in program.signatures),
            kernels=kernels,
            encodings=tuple(encoding for definition in program.definitions
                            for encoding in self._cache["encodings"].get(definition.id, ())))
        return selected, construction


class Realization(Record):
    """Reference to one immutable Plan and its admitted experiment/binding choices.

    The actual Plan is stored once by the run/planning context. Loading this
    small record alone does not verify the referenced Plan: validate_plan is the
    explicit pair boundary. No N-experiment Plan is embedded in N realizations.
    """

    plan_id: ContentID
    experiment: Text
    bindings: tuple[Binding, ...] = ()

    @model_validator(mode="after")
    def _choices(self):
        if len({item.parameter for item in self.bindings}) != len(self.bindings):
            raise ValueError("duplicate realization binding")
        return self

    def _selected_program(self, plan: Plan):
        """Validate the pair once and return its experiment and selected Program."""
        if self.plan_id != plan.content_id:
            raise ValueError("realization requires its exact immutable Plan")
        selected, program = plan._resolve_selection(self.experiment, self.bindings)
        if self.bindings != program.bindings:
            raise ValueError("Realization must store canonical effective bindings; use Plan.resolve")
        return selected, program

    def validate_plan(self, plan: Plan) -> Experiment:
        """Validate this reference/choice pair without traversing the experiment table."""
        return self._selected_program(plan)[0]

    def _selected_construction(self, plan: Plan):
        if self.plan_id != plan.content_id:
            raise ValueError("realization requires its exact immutable Plan")
        selected, construction = plan._selected_construction(self.experiment, self.bindings)
        if self.bindings != construction.program.bindings:
            raise ValueError("Realization must store canonical effective bindings; use Plan.resolve")
        return selected, construction

    def selected_construction(self, plan: Plan) -> SelectedConstruction:
        """Derive the selected graph using the same pair-boundary merge rule."""
        return self._selected_construction(plan)[1]

    def resolved_observation(self, plan: Plan) -> tuple[str, ObservationSpec]:
        """Resolve IR acquisition using the effective point, before native reservation.

        The point is resolved through the Plan's memo and its Program's
        stored Readiness, so repeated calls for the point that occupies the
        recent-resolution slot (and, for a batch amplitude readout, the
        recent-construction slot) repeat neither the Method's point relations
        nor the Program admission. An evicted point is resolved and admitted
        again. A caller that already holds the point's Experiment and
        SelectedConstruction calls ``Experiment._resolved_observation`` with
        an ``_Admission`` of that construction's Program
        (``_Admission.admitted``) and its identity.
        """
        selected, program = self._selected_program(plan)
        if selected.batch is None:
            return selected.setting, selected.observation
        admission = _Admission(program)
        admission.admitted().require_ready()
        construction_id = (self.selected_construction(plan).content_id
                           if (admission.nodes[selected.batch].observation_kind == "probabilities"
                               and selected.readout.amplitudes is not None) else None)
        return selected._resolved_observation(admission, construction_id=construction_id)


# Backend seeds are drawn as uint32 (RandomStreams.next_seed) and
# RuntimeOptions admits exactly that domain, with no offset or wraparound.
# Registered in ENGINEERING_CONSTANTS.md. Revisit with a changed backend seed
# domain.
MAX_RUNTIME_SEED = 2**32 - 1


class RuntimeOptions(Record):
    """Actual uint32 backend seed saved with one preparation."""

    seed: Annotated[StrictInt, Field(ge=0, le=MAX_RUNTIME_SEED)]
