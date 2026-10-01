"""Backend capability declarations.

A target states what it supports at three levels. Coarse capabilities name an
execution kind. ``readouts``, ``artifacts`` and ``program_nodes`` name the
exact observation kinds and inputs it accepts. ``instructions`` names the exact
primitive or selected kernel, transform and width. For each of these
collections, None means unknown and an empty tuple means none, so profile assessment
(``backends.assessment``) can tell an undeclared subset from an unsupported one.
A coarse capability never certifies a readout or instruction that the target
does not list.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Literal

from pydantic import StrictBool, model_validator

from nwqlib.core.records import Record, Source, Text
from nwqlib.operators.access import Count


class BackendCapability(str, Enum):
    """Execution features that algorithms may require from a backend."""

    STATEVECTOR = "statevector"
    COUNTS = "counts"
    EXPECTATION = "expectation"
    ESTIMATED_OBSERVABLE = "estimated_observable"
    QASM_EXPORT = "qasm_export"
    HARDWARE_SUBMIT = "hardware_submit"
    NOISE_MODEL = "noise_model"
    NATIVE_GATE_TARGET = "native_gate_target"
    HOST_KERNEL = "host_kernel"


class InstructionSupport(Record):
    """A concrete logical primitive or selected kernel, including its transform.

    max_qubits bounds the complete instruction width, including controls.
    Kernel support matches the exact implementation Source, never its name alone.
    """

    primitive: Literal["x", "h", "z", "sdg", "cx", "mc_z", "phase"] | None = None
    implementation: Source | None = None
    controlled: StrictBool = False
    adjoint: StrictBool = False
    max_qubits: Count

    @model_validator(mode="after")
    def _instruction(self):
        if (self.primitive is None) == (self.implementation is None):
            raise ValueError("declare exactly one primitive or selected implementation")
        return self


class BackendTarget(Record):
    """What a simulator, cloud provider or export target supports, for profile assessment.

    Pass it as `target=` to a
    [`DeviceConfiguration`][nwqlib.backends.profiles.DeviceConfiguration].
    Start from a built-in target such as
    [`AER_STATEVECTOR_TARGET`][nwqlib.backends.targets.AER_STATEVECTOR_TARGET]
    and change fields with `revise`, or build one with keyword arguments,
    for example `BackendTarget(name="my_target", provider="my_provider")`.
    `name` and `provider` are required. For each optional collection,
    `None` means unknown and an empty tuple means none, so profile
    assessment can tell an undeclared subset from an unsupported one. A
    coarse capability never certifies a readout or instruction that the
    target does not list. The [profiles guide](../profiles.md) shows how
    these declarations decide the capability check.

    Attributes:
        name: Required. Backend or target name.
        provider: Required. Provider family, such as `"qiskit_aer"` or
            `"ionq"`.
        capabilities: Default `None`, unknown. Coarse execution
            capabilities, such as `"statevector"`, `"counts"` or
            `"noise_model"`.
        description: Default `""`. Text for reports.
        native_basis_gates: Default `()`. Native or preferred basis gates
            when known, in the given order.
        max_qubits: Default `None`. Advertised or configured qubit limit.
        artifacts: Default `None`, unknown. Accepted input forms, such as
            `"selected_construction"` (a Plan's construction), which profile
            assessment requires.
        readouts: Default `None`, unknown. Supported observation kinds among
            `"pauli_expectation"`, `"counts"`, `"probabilities"`,
            `"host_scalars"`, `"amplitudes"`, `"estimated_observable"` and
            `"trajectory"`, declared separately from `capabilities`.
            `"trajectory"` is the exact schedule with several observation
            points, and each point also needs its own readout kind listed.
        readout_features: Default `None`, unknown. Trajectory features the
            target executes: `"multi_position"` (several observation points
            of one coherent state in one execution) and `"views"`
            (reversible basis changes before a Pauli or probability point,
            undone exactly afterwards).
        reducers: Default `None`, unknown. The registered reducers that the
            target runs at a trajectory point, which turn a saved state into
            statistics during execution: a tuple of reducer names, or
            `"registered"` for every reducer in
            `nwqlib.core.planning.READOUT_REDUCERS`.
        host_dependencies: Default `None`, unknown. Packages available to
            classical routines on this target. Profile assessment requires
            every dependency that a chosen routine declares to be listed.
        instructions: Default `None`, unknown. Exact instruction subset,
            including controls and adjoints, as `InstructionSupport`
            records, each naming one primitive gate or one implementation
            `Source` and its width.
        program_nodes: Default `None`, unknown. Supported `Program` node
            kinds.

    Raises:
        ValueError: If a collection declares a value twice.
    """

    name: Text
    provider: Text
    capabilities: tuple[BackendCapability, ...] | None = None
    description: str = ""
    native_basis_gates: tuple[Text, ...] = ()
    max_qubits: Count | None = None
    artifacts: tuple[Text, ...] | None = None
    readouts: tuple[Literal["pauli_expectation", "counts", "probabilities", "host_scalars", "amplitudes",
                            "estimated_observable", "trajectory"], ...] | None = None
    readout_features: tuple[Literal["multi_position", "views"], ...] | None = None
    reducers: tuple[Text, ...] | Literal["registered"] | None = None
    host_dependencies: tuple[Text, ...] | None = None
    instructions: tuple[InstructionSupport, ...] | None = None
    program_nodes: tuple[Text, ...] | None = None

    @model_validator(mode="after")
    def _sets(self):
        """Reject duplicate declarations and store set-like fields in sorted order.

        Sorting gives equal declarations one content identity, so a stored
        target compares equal however its sets were written. The basis gate
        tuple keeps its given order, and instruction entries sort by identity.
        """
        for name in ("capabilities", "native_basis_gates", "artifacts", "readouts", "readout_features",
                     "reducers", "program_nodes"):
            values = getattr(self, name)
            if values is not None and not isinstance(values, str):
                if len(set(values)) != len(values):
                    raise ValueError(f"duplicate {name} declaration")
                if name != "native_basis_gates":
                    object.__setattr__(self, name, tuple(sorted(values)))
        if self.instructions is not None:
            identities = [item.content_id for item in self.instructions]
            if len(set(identities)) != len(identities):
                raise ValueError("duplicate instruction support")
            object.__setattr__(self, "instructions", tuple(sorted(self.instructions, key=lambda item: item.content_id)))
        return self

    def missing_capabilities(
        self, required: Iterable[BackendCapability]
    ) -> frozenset[BackendCapability]:
        """Return required capabilities that this target does not support."""

        return frozenset(required) - frozenset(self.capabilities or ())


def capability_set(*capabilities: BackendCapability) -> frozenset[BackendCapability]:
    """Build an immutable backend capability set."""

    return frozenset(capabilities)


def require_backend_capabilities(
    backend_target: BackendTarget,
    required: Iterable[BackendCapability],
    *,
    context: str,
) -> None:
    """Raise an actionable error when a target misses required capabilities."""

    required_set = frozenset(required)
    missing = backend_target.missing_capabilities(required_set)
    if not missing:
        return
    required_names = ", ".join(sorted(capability.value for capability in required_set))
    missing_names = ", ".join(sorted(capability.value for capability in missing))
    available_names = ", ".join(
        sorted(capability.value for capability in backend_target.capabilities or ())
    ) if backend_target.capabilities is not None else "unknown"
    raise ValueError(
        f"{context} requires backend capabilities [{required_names}], "
        f"but backend target {backend_target.name!r} is missing [{missing_names}]. "
        f"Available capabilities: [{available_names}]."
    )


def unsupported_readout(target, observation):
    """Name the first part of ``observation`` that ``target`` does not declare, or return None.

    A trajectory needs the ``trajectory`` readout, the ``multi_position``
    feature, each point's readout kind, the ``views`` feature for a point with
    a view, and each reducer by name, or any registered reducer on a target
    that declares ``reducers="registered"``. The ``views`` feature covers Pauli and
    probability points; a view on an amplitude or reduction point is named
    as unsupported on every target. An unknown (None) or absent declaration
    supports nothing.
    """
    readouts = getattr(target, "readouts", None) or ()
    features = getattr(target, "readout_features", None) or ()
    reducers = getattr(target, "reducers", None) or ()
    if observation.kind not in readouts:
        return f"readout {observation.kind!r}"
    if observation.kind != "trajectory":
        return None
    if "multi_position" not in features:
        return f"a {len(observation.positions)}-point trajectory schedule"
    for point in observation.positions:
        if point.kind == "reduction":
            if reducers == "registered":
                from nwqlib.core.planning import registered_reducer
                if registered_reducer(point.reducer) is None:
                    return f"unregistered reduction {point.reducer!r} at point {point.id!r}"
            elif point.reducer not in reducers:
                return f"reduction {point.reducer!r} at point {point.id!r}"
        elif point.kind not in readouts:
            return f"readout {point.kind!r} at point {point.id!r}"
        if point.view is not None and ("views" not in features
                                       or point.kind not in {"pauli_expectation", "probabilities"}):
            return (f"the readout view of {point.kind} point {point.id!r}; "
                    "remove view= from that point to observe it without a view")
    return None
