"""Bounded physical state inputs and an explicit selected Qiskit preparation seam."""

from dataclasses import dataclass, field
from math import frexp, ldexp
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, StrictInt, model_validator

from nwqlib._limits import DEFAULT_MAX_DIRECT_AMPLITUDES
from nwqlib._numerics import _occupation_bits, normalize_state_vector_with_scale, normalize_physical_vector_with_scale
from nwqlib.core.records import Basis, InputRef, Record, Real, Source, Text
from nwqlib._validation import finite_real
from nwqlib.operators.access import Count, DEFAULT_INPUT_BYTES, InputManifest, _check_bytes
from nwqlib.operators.inputs import _array_input, _basis, _digest, _freeze_array, _manifest


class PhysicalScale(Record):
    """Numerical nonnegative scale = mantissa * 2**exponent; never squared.

    Evidence distinguishes an observed norm, composed method recovery, and a
    supplied unitary premise (not a norm measurement). Numerical evidence is
    not certified exact arithmetic. Binary
    scaling keeps product norms outside the representable float64 range.

    Each scale has one representation. A nonzero scale has ``0.5 <= mantissa < 1``
    as returned by ``math.frexp``, and zero is ``(0, 0)``. The unit norm is
    ``(0.5, 1)``.

    Attributes:
        mantissa: Binary mantissa, 0 or in [0.5, 1).
        exponent: Power of two, 0 for a zero scale.
        evidence: ``floating_point_norm`` for a computed norm, ``composed_floating_point_recovery`` for a product of method factors, or ``supplied_unitary_contract`` for the unit norm a supplied preparation circuit promises.
    """

    mantissa: Annotated[Real, Field(ge=0, lt=1)]
    exponent: StrictInt
    evidence: Literal["floating_point_norm", "composed_floating_point_recovery", "supplied_unitary_contract"] = "floating_point_norm"

    @model_validator(mode="after")
    def _normalized(self):
        if self.evidence == "supplied_unitary_contract" and (self.mantissa, self.exponent) != (0.5, 1):
            raise ValueError("supplied unitary evidence requires the canonical unit norm")
        if self.mantissa == 0:
            if self.exponent != 0:
                raise ValueError("zero scale has exponent zero")
        elif self.mantissa < 0.5:
            raise ValueError("nonzero scale requires 0.5 <= mantissa < 1")
        return self

    def as_float(self) -> float | None:
        """Return a representable norm, or None for overflow/underflow."""
        try:
            value = ldexp(self.mantissa, self.exponent)
        except OverflowError:
            return None
        return None if value == 0 and self.mantissa != 0 else value

    def squared_as_float(self) -> float | None:
        """Return representable physical norm squared; underflow is unavailable."""
        try:
            value = ldexp(self.mantissa * self.mantissa, 2 * self.exponent)
        except OverflowError:
            return None
        return None if value == 0 and self.mantissa != 0 else value

    def apply_vector(self, vector):
        """Scale finite complex128 components, or return None if any is lost.

        Per-component binary decomposition avoids an underflowing mantissa
        product before an exponent can recover a representable value. This
        allocates one complex output plus real mantissa/exponent scratch, all
        linear in the supplied vector. It does not normalize or change phase.
        """
        if type(vector) is not np.ndarray or vector.dtype != np.dtype("complex128") or vector.ndim != 1:
            raise ValueError("physical scaling requires a complex128 vector")
        if not np.isfinite(vector).all():
            raise ValueError("physical scaling requires finite components")
        result = np.empty_like(vector)
        limits = np.finfo(float)
        # Outside the entire binary64 exponent span no nonzero binary64 input
        # can produce a representable nonzero product; avoid huge integer casts.
        span = limits.maxexp - limits.minexp + limits.nmant + 2
        if abs(self.exponent) > span and np.any(vector != 0):
            return None
        if self.mantissa == 0 or not np.any(vector != 0):
            result.fill(0)
            return result
        for values, output in ((vector.real, result.real), (vector.imag, result.imag)):
            mantissas, exponents = np.frexp(values)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                np.ldexp(mantissas * self.mantissa, exponents + self.exponent, out=output)
            if not np.isfinite(output).all() or np.any((output == 0) & (values != 0)):
                return None
        return result


def compose_recovery(*factors):
    """Compose positive binary recovery without a full reciprocal or quotient.

    Each factor is a PhysicalScale, a positive finite real scalar, or a pair
    (numerator, denominator) whose operands must each be positive and finite.
    A binary mantissa/exponent pair is represented only by PhysicalScale.
    Current consumers pass a bounded short list; this is scalar arithmetic,
    not a persisted expression evaluator or a certified error bound.
    """
    # (0.5, 1) is the empty product 1. Each factor multiplies the mantissas
    # and adds the exponents, and frexp returns the mantissa to [0.5, 1). The
    # exponent is a Python integer, so no intermediate product overflows or
    # underflows binary64.
    mantissa, exponent = 0.5, 1
    for factor in factors:
        if isinstance(factor, PhysicalScale):
            m, e = factor.mantissa, factor.exponent
            if m == 0:
                raise ValueError("recovery factors must be strictly positive")
        else:
            numerator, denominator = factor if isinstance(factor, tuple) else (factor, 1.0)
            numerator = finite_real(numerator, "recovery numerator")
            denominator = finite_real(denominator, "recovery denominator")
            if numerator <= 0 or denominator <= 0:
                raise ValueError("recovery ratio operands must each be strictly positive")
            mn, en = frexp(numerator)
            md, ed = frexp(denominator)
            m, e = mn / md, en - ed
        mantissa, adjustment = frexp(mantissa * m)
        exponent += e + adjustment
    return PhysicalScale(mantissa=mantissa, exponent=exponent,
                         evidence="composed_floating_point_recovery")


class StatePreparationSpec(Record):
    """Executable construction availability, without constructing a circuit.

    Phase is physical: dividing by a positive norm keeps global phase.
    Ancilla and approximation facts default to unknown. A selected native
    construction supplies its actual facts; explicit declarations remain legal.
    Inverse/control availability refers to the selected unitary construction.

    Attributes:
        input: Identity of the admitted state input.
        basis: Its computational basis and coordinate order.
        physical_scale: Norm of the physical input, or None when unknown.
        normalization: The fixed convention that the circuit prepares the input divided by its positive norm.
        required_ancillas: Extra qubits the construction needs, 0 for every selected construction, None when unknown.
        inverse_available: Whether the selected unitary may be inverted.
        control_available: Whether the selected unitary may be controlled.
        implementation: Selected constructor, or None when no quantum preparation exists.
        approximation: Algorithmic approximation statement, or None when unknown.
        work_law: Text form of the construction's cost law.
        items: Stored numerical items the construction reads, such as amplitudes, qubit factors, occupation bits or circuit instructions.
        payload_bytes: Known logical bytes of the construction, the item bytes of a StateInput (checked by ``prepare_qiskit``) or the layered construction envelope for ``qiskit.mps_circuit``.
        work: Construction size units, such as q*2**q for a direct vector, q for product or occupation input, the instruction count for a supplied circuit, or the layered construction envelope for ``qiskit.mps_circuit``.
        blocker: Why no preparation exists, or None.
    """

    input: InputRef
    basis: Basis
    physical_scale: PhysicalScale | None
    normalization: Literal["physical input; prepare input divided by positive norm"] = "physical input; prepare input divided by positive norm"
    required_ancillas: Count | None = None
    inverse_available: bool
    control_available: bool
    implementation: Literal["qiskit.direct", "qiskit.product", "qiskit.occupation", "qiskit.supplied", "qiskit.mps_circuit"] | None
    approximation: Text | None = None
    work_law: Text
    items: Count | None
    payload_bytes: Count | None
    work: Count | None
    blocker: Text | None = None

    @model_validator(mode="after")
    def _availability(self):
        if self.implementation is None:
            if self.blocker is None or self.inverse_available or self.control_available:
                raise ValueError("unavailable preparation requires a blocker and no inverse/control claim")
        elif (self.blocker is not None or self.physical_scale is None
              or self.physical_scale.mantissa == 0 or self.required_ancillas != 0
              or any(value is None for value in (self.items, self.payload_bytes, self.work))):
            raise ValueError("selected preparation requires nonzero scale and known construction limits")
        return self


@dataclass(frozen=True, eq=False, init=False, slots=True)
class StateInput:
    """Factory-owned physical data and its correlated preparation specification.

    Fields cannot be independently replaced. Exported record revisions remain
    standalone declarations and do not change this native owner.

    Attributes:
        manifest: Input identity, dimensions and checked metadata.
        preparation: Scale and selected construction specification.
        _physical: Private physical amplitudes or occupation bits.
        _direction: Private normalized vector/factors, or None for zero/opaque input.
        _native: Private supplied circuit snapshot, or None for other input forms.
    """

    manifest: InputManifest
    preparation: StatePreparationSpec
    _physical: object = field(repr=False)
    _direction: object = field(repr=False)
    _native: object = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError("native inputs require an ingestion/declaration factory; field replacement is unsupported")

    @classmethod
    def _from_admitted(cls, manifest, preparation, physical, direction, native=None):
        instance = object.__new__(cls)
        object.__setattr__(instance, "manifest", manifest)
        object.__setattr__(instance, "preparation", preparation)
        object.__setattr__(instance, "_physical", physical)
        object.__setattr__(instance, "_direction", direction)
        object.__setattr__(instance, "_native", native)
        return instance

    @property
    def basis(self):
        return self.manifest.basis

    @property
    def reference(self):
        return self.manifest.reference

    def to_record(self):
        """Describe this handle without copying or normalizing numerical data."""
        return dict(format="nwqlib.state_input/2",
                    manifest=self.manifest.model_dump(mode="json", exclude_computed_fields=True),
                    preparation=self.preparation.model_dump(mode="json", exclude_computed_fields=True))

    @classmethod
    def from_record(cls, record):
        """Read inert metadata; numerical or circuit access is not recreated."""
        if (type(record) is not dict or set(record) != {"format", "manifest", "preparation"}
                or record["format"] != "nwqlib.state_input/2"):
            raise ValueError("unsupported state input record")
        manifest = InputManifest.model_validate(record["manifest"])
        preparation = StatePreparationSpec.model_validate(record["preparation"])
        if preparation.input != manifest.reference or preparation.basis != manifest.basis:
            raise ValueError("state preparation metadata differs from its input identity or basis")
        return cls._from_admitted(manifest, preparation, None, None)

    def _require_data(self):
        if self._physical is None and self._native is None:
            raise ValueError("state native data is unavailable; a descriptive record does not restore data access")

    def entry(self, index: int):
        """Read one explicit physical vector entry; compact inputs stay compact."""
        self._require_data()
        if self.manifest.reference.representation != "vector" or "entries" not in self.manifest.access:
            raise ValueError("physical vector entries require an ingested vector")
        if type(index) is not int or not 0 <= index < self.manifest.basis.dimension:
            raise ValueError("entry index must lie in the state dimension")
        return complex(self._physical[index])

    def physical_vector(self):
        """Access immutable physical amplitudes; compact inputs stay compact."""
        self._require_data()
        if self.manifest.reference.representation != "vector" or "entries" not in self.manifest.access:
            raise ValueError("physical vector access requires an ingested vector")
        return self._physical


def _spec(manifest, scale, implementation, *, items, payload_bytes, work, law, blocker=None):
    zero = scale.mantissa == 0
    blocker = "zero physical input is not quantum-preparable" if zero else blocker
    unavailable = blocker is not None
    return StatePreparationSpec(
        input=manifest.reference, basis=manifest.basis, physical_scale=scale,
        inverse_available=not unavailable, control_available=not unavailable,
        implementation=None if unavailable else implementation,
        required_ancillas=None if unavailable else 0,
        approximation=None if unavailable else "no algorithmic approximation; floating synthesis roundoff not bounded",
        work_law=law, items=items, payload_bytes=payload_bytes, work=work,
        blocker=blocker,
    )


def ingest_vector(vector, *, max_bytes=DEFAULT_INPUT_BYTES) -> StateInput:
    """Admit a small physical float64/complex128 vector, including zero data.

    Nonzero data normalizes once through the shared stable BLAS kernel. Both
    physical bytes and normalized direction are kept (O(D)); metadata
    exports and preparation do not normalize or hash again.
    """
    vector = _array_input(vector, ndim=1, max_bytes=max_bytes, extra_bytes_per_item=16)
    dimension = vector.size
    if dimension < 1:
        raise ValueError("physical state dimension must be positive")
    physical = _freeze_array(vector)
    if not np.isfinite(physical).all():
        raise ValueError("state amplitudes must be finite")
    if np.any(physical != 0):
        direction, _, norm_scale = normalize_physical_vector_with_scale(physical)
        direction = _freeze_array(direction)
    else:
        direction, norm_scale = None, (0.0, 0)
    scale = PhysicalScale(mantissa=norm_scale[0], exponent=norm_scale[1])
    manifest = _manifest("vector", dimension, physical.shape, str(physical.dtype),
                         physical.nbytes + (0 if direction is None else direction.nbytes),
                         _digest("vector", physical.shape, (physical,)),
                         ("entries",),
                         ("finite", "physical norm computed"), "O(D) admission; two O(D) native payloads")
    spec = _spec(manifest, scale, "qiskit.direct", items=dimension, payload_bytes=dimension * 16,
                 blocker=("no selected zero-qubit quantum preparation" if dimension == 1 else
                          "dimension is not a power of two; no selected quantum preparation"
                          if dimension & (dimension - 1) else None),
                 work=(dimension.bit_length() - 1) * dimension,
                 law="direct magnitude/phase synthesis O(q*2**q) work, O(2**q) live numerical storage")
    return StateInput._from_admitted(manifest, spec, physical, direction)


def ingest_product(amplitudes, *, max_bytes=DEFAULT_INPUT_BYTES) -> StateInput:
    """Admit shape (q,2) amplitudes with row j belonging to qubit j.

    Physical tensor order is row q-1 tensor ... tensor row 0. Non-unit factors
    and complex/global phase survive. No 2**q vector is allocated.
    """
    amplitudes = _array_input(amplitudes, ndim=2, max_bytes=max_bytes, extra_bytes_per_item=16)
    if amplitudes.shape[0] < 1 or amplitudes.shape[1] != 2:
        raise ValueError("product amplitudes must have shape (positive qubits, 2)")
    physical = _freeze_array(amplitudes)
    if not np.isfinite(physical).all():
        raise ValueError("product amplitudes must be finite")
    if np.any(np.all(physical == 0, axis=1)):
        direction, mantissa, exponent = None, 0.0, 0
    else:
        direction = np.empty(physical.shape, dtype=complex)
        mantissa, exponent = 0.5, 1
        for qubit, pair in enumerate(physical):
            direction[qubit], _, (local_mantissa, local_exponent) = normalize_state_vector_with_scale(pair)
            mantissa, adjustment = frexp(mantissa * local_mantissa)
            exponent += local_exponent + adjustment
        direction = _freeze_array(direction)
    scale = PhysicalScale(mantissa=mantissa, exponent=exponent)
    qubits = physical.shape[0]
    manifest = _manifest("product", 1 << qubits, physical.shape, str(physical.dtype),
                         physical.nbytes + (0 if direction is None else direction.nbytes),
                         _digest("product", physical.shape, (physical,)),
                         (),
                         ("finite", "physical norm computed"), "O(q) ingestion, storage and single-qubit synthesis")
    spec = _spec(manifest, scale, "qiskit.product", items=qubits, payload_bytes=qubits * 32,
                 work=qubits, law="q single-qubit preparations; O(q) work/storage; no full state")
    return StateInput._from_admitted(manifest, spec, physical, direction)


def ingest_occupation(occupations, *, num_qubits: int, max_bytes=DEFAULT_INPUT_BYTES) -> StateInput:
    """Admit q0-first bits: string '10' means qubits (1,0), printed ket |01>.

    Input is a sized string/list/tuple; width and exact 0/1 values are checked
    before circuit allocation. No electron, spin or sector inference is made.
    """
    if type(num_qubits) is not int or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    # One uint8 per occupation bit, plus two dimension-integer slots as in
    # ingest_periodic_stencil. The dimension 2**q has q + 1 bits, that is
    # (q + 8) // 8 bytes. The factor two has no recorded derivation.
    _check_bytes(num_qubits + 2 * ((num_qubits + 8) // 8), max_bytes, "occupation input")
    if type(occupations) not in (str, tuple, list) or len(occupations) != num_qubits:
        raise ValueError("occupation input must be a sized bit sequence matching num_qubits")
    bits = _occupation_bits(occupations, "occupations")
    physical = _freeze_array(np.asarray(bits, dtype=np.uint8))
    manifest = _manifest("occupation", 1 << num_qubits, physical.shape, "uint8", physical.nbytes,
                         _digest("occupation", physical.shape, (physical,)), (),
                         ("exact bits and width",), "O(q) ingestion and X-gate synthesis; no full state")
    spec = _spec(manifest, PhysicalScale(mantissa=0.5, exponent=1), "qiskit.occupation",
                 items=num_qubits, payload_bytes=num_qubits, work=num_qubits,
                 law="at most q X gates; O(q) work/storage; no full state")
    return StateInput._from_admitted(manifest, spec, physical, None)


def bind_preparation_circuit(circuit, *, reference: InputRef, basis: Basis,
                             max_bytes=DEFAULT_INPUT_BYTES) -> StateInput:
    """Bind supplied U|0> through one bounded stored-data snapshot.

    The circuit reference is declared, never a native content digest. Admission
    preserves physical phase and checks supported unitary execution structure;
    it does not measure a norm, synthesize, simulate, or expose classical entries.
    Snapshot census excludes SDK synthesis and native allocator overhead.
    """
    from nwqlib._optional import optional_import
    optional_import("qiskit", extra="qiskit")
    from qiskit import QuantumCircuit
    from nwqlib.blocks._qiskit_intake import snapshot_circuit

    if type(circuit) is not QuantumCircuit:
        raise TypeError("supplied preparation requires an ordinary QuantumCircuit")
    if not isinstance(reference, InputRef) or reference.representation != "circuit":
        raise ValueError("supplied preparation requires a circuit reference")
    if (not isinstance(basis, Basis) or basis.dimension & (basis.dimension - 1)
            or basis.dimension.bit_length() - 1 != circuit.num_qubits):
        raise ValueError("supplied preparation width must match its power-of-two basis")
    if circuit.num_clbits or circuit.num_parameters:
        raise ValueError("supplied preparation requires no classical bits or free parameters")
    native, data_bytes = snapshot_circuit(circuit, max_bytes=max_bytes)
    manifest = InputManifest(reference=reference, basis=basis, identity_status="declared",
        dtype=None, shape=None, byte_order=None, payload_bytes=data_bytes, access=(),
        checked=("width matches power-of-two basis", "no classical bits or free parameters",
                 "finite bound native parameters", "top-level gates or gate-based annotations",
                 "stored definitions only; barriers omitted"),
        work_law="supplied native circuit; logical stored-data snapshot census in preparation spec")
    spec = StatePreparationSpec(input=reference, basis=basis,
        physical_scale=PhysicalScale(mantissa=.5, exponent=1, evidence="supplied_unitary_contract"),
        required_ancillas=0, inverse_available=True, control_available=True,
        implementation="qiskit.supplied", approximation=None,
        work_law="memo-preserving stored-data copy; supplied unitary premise; numerical/synthesis error and native allocation overhead unknown",
        items=len(native.data), payload_bytes=data_bytes, work=len(native.data))
    return StateInput._from_admitted(manifest, spec, None, None, native)


@dataclass(frozen=True, eq=False)
class QiskitPreparation:
    """Actual selected circuit and its input-bound specification; no execution.

    Args:
        spec: Specification consumed to construct this circuit.
        circuit: Native unitary preparation circuit, supporting inverse/control.
    """

    spec: StatePreparationSpec
    circuit: object


def prepare_qiskit(state: StateInput, *, max_bytes=DEFAULT_INPUT_BYTES,
                   max_direct_amplitudes=DEFAULT_MAX_DIRECT_AMPLITUDES) -> QiskitPreparation:
    """Explicit synthesis only, after nonzero/access/size admission.

    Uses the already normalized direction, preserving its global phase. The
    returned native circuit may be composed, inverted, or made into a controlled
    gate. This does not submit, transpile, or simulate it.

    A supplied circuit is snapshotted again on every call. The stored
    snapshot is shared by every later preparation of this input, and the
    caller owns the returned circuit and may extend it in place, so the
    stored circuit is never returned itself.

    Inside a Run, the native constructor of a selected preparation receives
    ``ExecutionLimits.max_direct_amplitudes`` through lowering, and the Run
    checks it for the whole Plan before its first preparation.
    """
    state._require_data()
    spec = state.preparation
    if type(max_direct_amplitudes) is not int or max_direct_amplitudes < 1:
        raise ValueError("max_direct_amplitudes must be a positive integer")
    if spec.input != state.manifest.reference or spec.basis != state.manifest.basis:
        raise ValueError("preparation must keep its admitted input identity and basis")
    if spec.blocker is not None or spec.implementation is None:
        raise ValueError(spec.blocker or "executable state preparation unavailable")
    # The direct magnitude/phase construction has O(q*2**q) arithmetic even
    # when its input data fit in memory. This finite synthesis limit does not
    # constrain compact preparations or legal input/Plan dimension metadata.
    # The default 65_536 = 2**16 amplitudes (_limits.py) is registered in
    # ENGINEERING_CONSTANTS.md, "Explicit input operation defaults".
    if spec.implementation == "qiskit.direct" and state.basis.dimension > max_direct_amplitudes:
        raise ValueError(f"direct preparation has {state.basis.dimension} amplitudes, "
                         f"exceeding max_direct_amplitudes={max_direct_amplitudes}")
    _check_bytes(spec.payload_bytes, max_bytes, "state preparation data")
    if spec.implementation == "qiskit.supplied":
        from nwqlib.blocks._qiskit_intake import snapshot_circuit
        circuit, _ = snapshot_circuit(state._native, max_bytes=max_bytes)
        return QiskitPreparation(spec, circuit)
    from nwqlib._optional import optional_import
    optional_import("qiskit", extra="qiskit")
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.state_preparation.direct import _build_normalized_state_preparation

    if spec.implementation == "qiskit.direct":
        circuit = _build_normalized_state_preparation(
            state._direction, input_norm=spec.physical_scale.as_float(), register_name="system",
        ).circuit
    else:
        qubits = spec.basis.dimension.bit_length() - 1
        circuit = QuantumCircuit(qubits, name="state_preparation")
        if spec.implementation == "qiskit.occupation":
            for qubit, bit in enumerate(state._physical):
                if bit:
                    circuit.x(qubit)
        else:
            for qubit, pair in enumerate(state._direction):
                local = _build_normalized_state_preparation(pair, input_norm=1.0, register_name="system")
                circuit.compose(local.circuit, qubits=[qubit], inplace=True)
    return QiskitPreparation(spec, circuit)


def state_input(value, *, max_bytes=DEFAULT_INPUT_BYTES) -> StateInput:
    """Admit known physical amplitudes or a supplied preparation circuit.

    QuantumCircuit means its unitary action on the all-zero state. Its native
    data is copied without simulation or synthesis and receives a fresh source
    identity; reusing an existing StateInput preserves that identity.

    NWQLib computes no content digest of circuit data, so it has no basis to
    assert that two circuits prepare the same input. A fresh identity per
    admission never makes that assertion. A caller who knows that two
    circuits are the same input passes one identity to
    ``bind_preparation_circuit``.
    """
    _check_bytes(0, max_bytes)
    if isinstance(value, StateInput):
        return value
    if type(value) in (np.ndarray, list, tuple):
        return ingest_vector(value, max_bytes=max_bytes)
    module, name = type(value).__module__, type(value).__name__
    if module == "qiskit.quantum_info.states.statevector" and name == "Statevector":
        from qiskit.quantum_info import Statevector
        if type(value) is Statevector:
            return ingest_vector(value.data, max_bytes=max_bytes)
    if module == "qiskit.circuit.quantumcircuit" and name == "QuantumCircuit":
        from qiskit import QuantumCircuit
        from uuid import uuid4
        if type(value) is QuantumCircuit:
            # 16 bytes per qubit, the same per-qubit allowance that the snapshot
            # census charges a circuit, plus the dimension integer 2**q of
            # (q + 8) // 8 bytes, checked before the basis record is built.
            _check_bytes(16 * value.num_qubits + (value.num_qubits + 8) // 8, max_bytes,
                         "circuit width metadata")
            reference = InputRef(identity="circuit:" + str(uuid4()), representation="circuit",
                source=Source(name="supplied QuantumCircuit", version="1",
                    domain="declared unitary preparation on the all-zero input",
                    reference="immutable stored-data snapshot; no numerical action or native content hash"))
            return bind_preparation_circuit(value, reference=reference,
                basis=_basis(1 << value.num_qubits), max_bytes=max_bytes)
    raise TypeError("state input requires numerical vector data, Statevector, QuantumCircuit or StateInput")
