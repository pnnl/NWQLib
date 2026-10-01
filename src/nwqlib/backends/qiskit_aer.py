"""Qiskit Aer lowering, readout insertion and result translation.

This module owns how a selected logical circuit becomes an Aer-executable
circuit. Definitions are expanded until Aer's native classes remain. Pauli
readouts are inserted at the original instruction position, and probability
and amplitude saves follow the circuit after final measurements are removed.
A trajectory schedule inserts every point's saves at its original bound
boundary, before lowering, and executes nothing after the last save.
A final measurement is removed only when no later classical instruction reads
its bit.
``_submit_aer_execution`` is the single place where Aer results become NWQLib
raw outputs, with the key order described in ``nwqlib.backends.connection``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Barrier, ControlFlowOp, Gate, Instruction, Measure, Reset
    from qiskit.circuit.library import UCGate, UnitaryGate
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.quantum_info import Pauli
    from qiskit.transpiler.passes import Decompose
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import SaveExpectationValue, SaveProbabilities, SaveStatevector
except ModuleNotFoundError as error:
    if error.name in {"qiskit", "qiskit_aer"}:
        error.add_note("Aer execution requires its extra: pip install 'nwqlib[aer]'.")
    raise

from nwqlib._validation import integer
from nwqlib.backends.capabilities import BackendTarget
from nwqlib.backends.results import BackendRunResult
from nwqlib.backends.targets import AER_COUNTS_TARGET, AER_STATEVECTOR_TARGET
from nwqlib.execution import ExecutionMode


# Engineering constant: decompose() repetitions that flatten nested
# instructions (StatePreparation, controlled customs) into Aer-executable
# gates. Execution stops earlier when only native operations remain; selected
# explicit-analysis callers still use this fixed decomposition budget.
AER_EXECUTION_DECOMPOSE_REPS = 10


class _AerDecompose(Decompose):
    """Use Qiskit's decomposition, stopping at Aer-native classes and names."""

    def __init__(self, target, *, exact_synthesis=None):
        """Record Aer's native operation classes by name, from the simulator's target.

        ``expanded`` reports whether the last pass decomposed anything, which
        lets ``_lower_aer_circuit`` stop as soon as only native operations remain.
        ``exact_synthesis`` is the Run's ``_exact_dense_unitaries``, which
        synthesizes the dense unitaries that a target without the unitary
        instruction needs, each distinct matrix once per Run object, and
        reserves each new synthesis against the Run's ``max_synthesis_work``.
        With None, lowering calls ``qiskit_compat.exact_dense_unitaries``
        without a synthesis cache or charge.
        """
        super().__init__(apply_synthesis=True)
        self.native_types = {
            name: operation if isinstance(operation, type) else operation.base_class
            for name, operation in zip(target.operation_names, target.operations, strict=True)
        }
        self.expanded = False
        self.exact_synthesis = exact_synthesis

    def is_native(self, operation):
        """Whether Aer executes ``operation`` directly, matched by name and class.

        Matching the class as well as the name keeps a custom gate that happens
        to be named like an Aer gate (for example ``x``) from reaching Aer's
        name-based dispatcher with its own definition hidden.
        """
        # Aer consumes the full matrix table, not Qiskit's diagonal omission or
        # simplified-control mapping. Those representations require their definition.
        if isinstance(operation, UCGate) and (
            operation.up_to_diagonal or len(operation.params) != 1 << (operation.num_qubits - 1)
        ):
            return False
        return (operation.name == "barrier" and type(operation) is Barrier) or (
            getattr(operation, "base_class", type(operation))
            is self.native_types.get(operation.name)
        )

    def _should_decompose(self, node):
        """Decompose every operation that ``is_native`` rejects, and record that the pass changed something."""
        if isinstance(node.op, ControlFlowOp):
            self.expanded |= not self.is_native(node.op)
            return True  # Qiskit applies this same policy inside each block.
        if self.is_native(node.op):
            return False
        self.expanded = True
        return True


def _lower_aer_circuit(circuit, decompose):
    """Lower one circuit with one owned operation copy and no count proxy.

    Aer applies a ``UnitaryGate`` as its matrix. A noise model's gate basis
    omits that instruction, and decomposing it would use Qiskit's inexact
    dense synthesis, so dense unitaries are first replaced by their exact
    synthesis, through the Run's synthesis cache when ``decompose``
    carries the Run's ``_exact_dense_unitaries`` (``_AerDecompose``).
    """
    if "unitary" not in decompose.native_types:
        from nwqlib.subroutines.qiskit_compat import exact_dense_unitaries

        circuit = (decompose.exact_synthesis or exact_dense_unitaries)(circuit)
    dag = circuit_to_dag(circuit, copy_operations=True)
    repetitions = 0
    for _ in range(AER_EXECUTION_DECOMPOSE_REPS):
        decompose.expanded = False
        dag = decompose.run(dag)
        repetitions += 1
        if not decompose.expanded:
            break
    else:
        # The final allowed pass may have completed lowering. Check its output
        # without another rewrite: a custom Gate named "x" must never reach
        # Aer's name-based dispatcher with its H definition still hidden.
        pending = [dag]
        while pending:
            for node in pending.pop().op_nodes():
                if not decompose.is_native(node.op):
                    raise ValueError(
                        f"Aer lowering requires native operations within {AER_EXECUTION_DECOMPOSE_REPS} passes; "
                        f"unresolved {type(node.op).__name__} {node.op.name!r}"
                    )
                if isinstance(node.op, ControlFlowOp):
                    pending.extend(circuit_to_dag(block, copy_operations=False) for block in node.op.blocks)
    return dag_to_circuit(dag, copy_operations=False), repetitions


class _AerPreparation:
    """One solve's bound top-level Gate definitions, shared by identity.

    Keep unique lowered blocks, never expanded submission circuits. The caller
    must keep its logical gates unchanged and discard this owner after the solve.
    Parameter and definition payloads may be shared with those source gates;
    immutability is the caller's lifetime contract, not a deep-copy guarantee.
    Strong source references prevent reuse of an expired object's integer id.
    """

    def __init__(self):
        self._blocks: dict[int, tuple[Gate, Instruction]] = {}

    def keep_sources(self, sources):
        """Release definitions outside the caller's current immutable gate set.

        Prepared circuits keep their own instructions. Run keeps static
        definitions and the previous/current parameter points, including during
        a failed preparation, instead of accumulating every optimizer trial.
        """
        stored = {id(source) for source in sources}
        for key in tuple(self._blocks):
            if key not in stored:
                del self._blocks[key]

    def prepare(self, circuit, decompose):
        """Return a new container whose non-native top-level gates share cached lowered definitions.

        Each distinct source gate is lowered once and wrapped as an Instruction
        with the same name, parameters and label, so instruction positions and
        observation boundaries are unchanged. A ``UnitaryGate`` wrapper is
        named ``dense_unitary`` instead, because Qiskit's ``circuit_to_dag``
        turns an instruction named ``unitary`` with a matrix parameter back
        into a ``UnitaryGate`` and would discard the exact lowered definition.
        """
        prepared = circuit.copy_empty_like()
        for item in circuit.data:
            operation = item.operation
            if isinstance(operation, Gate) and not decompose.is_native(operation):
                key = id(operation)
                if key not in self._blocks:
                    block = QuantumCircuit(operation.num_qubits)
                    block.append(operation, block.qubits, copy=False)
                    lowered, _ = _lower_aer_circuit(block, decompose)
                    # Instruction also accepts matrix-valued gate parameters.
                    name = "dense_unitary" if isinstance(operation, UnitaryGate) else operation.name
                    shared = Instruction(name, operation.num_qubits, 0,
                                         operation.params, label=operation.label)
                    shared.definition = lowered
                    self._blocks[key] = (operation, shared)
                operation = self._blocks[key][1]
            prepared.append(operation, item.qubits, item.clbits, copy=False)
        return prepared


def _remove_final_measurements(circuit: QuantumCircuit, *, inplace: bool = False) -> QuantumCircuit:
    """Remove final readout, preserving custom names and classical dependencies."""

    collisions = []
    consumed = set()
    for index in range(len(circuit.data) - 1, -1, -1):
        item = circuit.data[index]
        operation = item.operation
        if ((operation.name == "measure" and not isinstance(operation, Measure))
                or (operation.name == "barrier" and not isinstance(operation, Barrier))
                or (isinstance(operation, Measure) and not consumed.isdisjoint(item.clbits))):
            collisions.append(index)
        if not isinstance(operation, Measure):
            # Qiskit's removal pass follows quantum successors. A measurement
            # can instead feed a later conditional on another qubit. Composite
            # classical instructions are conservatively protected until lowered.
            consumed.update(item.clbits)
    if not collisions:
        if not inplace:
            return circuit.remove_final_measurements(inplace=False)
        circuit.remove_final_measurements(inplace=True)
        return circuit
    # Reuse the copy needed by native removal; temporary labels never reach
    # resource inventory or execution. No definitions are lowered here.
    protected = circuit if inplace else circuit.copy()
    names = set(circuit.count_ops())
    original_operations = {}
    for index in collisions:
        item = protected.data[index]
        name = f"_nwqlib_final_operation_{index}"
        while name in names:
            name += "_"
        names.add(name)
        original_operations[name] = item.operation
        # A renamed native Measure still has a native opcode in Qiskit's DAG.
        # Use an opaque temporary wrapper and restore the owned operation after
        # removal; no definition copy or synthesis is needed at this boundary.
        operation = Instruction(name, item.operation.num_qubits, item.operation.num_clbits, [])
        protected.data[index] = item.replace(operation=operation)
    protected.remove_final_measurements(inplace=True)
    for index, item in enumerate(protected.data):
        if item.operation.name in original_operations:
            protected.data[index] = item.replace(operation=original_operations[item.operation.name])
    return protected


def _prepare_execution_circuit(
    circuit: QuantumCircuit,
    *,
    decompose: _AerDecompose,
    add_save_statevector: bool,
    probability_qubits: tuple[int, ...] | None = None,
    remove_final_measurements: bool = False,
) -> tuple[QuantumCircuit, dict[str, Any], int | None, bool]:
    """Return an Aer-executable circuit and preprocessing metadata.

    Qiskit Aer does not always execute composite library instructions such as
    ``StatePreparation`` or controlled dense ``UnitaryGate`` directly.
    Preparation therefore decomposes an owned execution copy and leaves the
    caller's logical circuit unchanged.
    """

    execution_circuit, repetitions = _lower_aer_circuit(circuit, decompose)
    before = 0
    if remove_final_measurements:
        before = execution_circuit.count_ops().get("measure", 0)
        if before:
            # Composite Instructions can expose final measurements only after
            # lowering. This is an owned execution copy; logical inventory
            # remains separate. Pauli observations were inserted before lowering.
            _remove_final_measurements(execution_circuit, inplace=True)
    save_statevector_added = False
    if add_save_statevector:
        execution_circuit.save_statevector()
        save_statevector_added = True
    elif probability_qubits is not None:
        execution_circuit.save_probabilities(qubits=probability_qubits)
    operations = execution_circuit.count_ops()
    native_removed = before > operations.get("measure", 0)
    # Counts inside branches/loops are execution-dependent. Do not turn a
    # missing top-level Measure into a claim of zero dynamic measurements.
    measurements = None if execution_circuit.has_control_flow_op() else operations.get("measure", 0)
    metadata = {
        "preprocessing": "decompose",
        "decompose_reps": repetitions,
        "save_statevector_added": save_statevector_added,
        "logical_operation_names": tuple(sorted(circuit.count_ops())),
        "execution_operation_names": tuple(sorted(operations)),
    }
    return execution_circuit, metadata, measurements, native_removed


@dataclass(frozen=True)
class _PreparedAerExecution:
    """Internal prepared native circuit; the caller owns its mutation lifetime.

    ``trajectory`` is the save plan of a trajectory schedule
    (``_trajectory_saves``), or None for a single-endpoint readout.
    """

    simulator: object
    circuit: QuantumCircuit
    backend_target: BackendTarget
    shots: int | None
    pauli_expectation_readout: tuple[int, tuple[str, ...]] | None
    probability_qubits: tuple[int, ...] | None
    metadata: dict
    trajectory: tuple | None = None


@dataclass(frozen=True)
class _TrajectorySave:
    """One saved datum of a trajectory: its boundary, save kind, result key and payload.

    ``payload`` is the Pauli label of an expectation save, the ordered qubits
    of a probability save, or None for a statevector save. ``view`` is the
    ``(tail, inverse)`` definition pair of a readout view, or None.
    """

    boundary: int
    kind: str
    key: str
    payload: object
    view: tuple | None = None


def _trajectory_saves(observation, boundaries):
    """Plan the saves of a trajectory schedule and associate each point with them.

    Points are visited in schedule order with their resolved ``boundaries``.
    Identical requests share one saved datum: the same Pauli label at one
    boundary, the same marginal qubits at one boundary, and one statevector
    per boundary for amplitude and reduction points. A request through a
    readout view is a different datum from one without it, and from one
    through another view. Save keys are ``nwqlib_t<n>`` in first-use order,
    so the same schedule and boundaries always give the same keys, which QPY
    restoration relies on. The saves are returned in insertion order: at
    each boundary the saves without a view, then each view's saves, the
    views in first-use order; a view executes only for a Pauli or
    probability point.

    Returns:
        ``(saves, points)``: the saves in boundary order, and per point in
        schedule order ``(point_id, kind, keys)``, where ``keys`` is a tuple
        of ``(label, key)`` pairs for a Pauli point and one key otherwise.
    """
    saves, keys, points, groups = [], {}, [], {}

    def key(boundary, kind, payload, view=None):
        identity = boundary, kind, payload, view
        if identity not in keys:
            keys[identity] = f"nwqlib_t{len(keys)}"
            groups.setdefault((boundary, view), len(groups))
            saves.append(_TrajectorySave(boundary, kind, keys[identity], payload, view))
        return keys[identity]

    for point, boundary in zip(observation.positions, boundaries, strict=True):
        view = None if point.view is None else (point.view.tail, point.view.inverse)
        if view is not None and point.kind not in ("pauli_expectation", "probabilities"):
            raise ValueError(f"Aer executes a readout view only for a Pauli or probability point, not {point.id!r}")
        if point.kind == "pauli_expectation":
            points.append((point.id, point.kind, tuple((label, key(boundary, "pauli", label, view))
                                                       for label in point.labels)))
        elif point.kind == "probabilities":
            points.append((point.id, point.kind, key(boundary, "probabilities", tuple(point.qubits), view)))
        else:
            points.append((point.id, point.kind, key(boundary, "statevector", None)))
    # Boundaries are nondecreasing in schedule order; the saves without a
    # view precede the view groups at their boundary.
    order = {group: (group[0], group[1] is not None, index) for group, index in groups.items()}
    saves.sort(key=lambda save: order[(save.boundary, save.view)])
    return tuple(saves), tuple(points)


def _aer_operation_counts(circuit):
    """Return ``(native_operations, save_instructions)`` of a lowered coherent Aer circuit.

    ``native_operations`` is the receipt's numerical operation count ``G``:
    the native evolution operations to which the per-operation state-roundoff
    model applies (``_validation.py``, consumed by
    ``PreparedArtifact.probability_window`` and ``state_error``). A save
    evaluates a statistic of the current state, or copies the state, and
    stores the result. Rounding in that evaluation changes the saved
    statistic; it does not insert another local state error into the
    continuing state, so a save adds zero to ``G``. Readout arithmetic is
    represented separately by ``exact_readout_roundoff``. Barrier and Delay
    are state identities on this noiseless target. The save inventory is an
    instruction count, not a runtime-work estimate: a marginal can require a
    pass over the state and exponential output storage.

    Premise: the lowered, deterministic coherent native circuit. Instruction
    types are compared, not names, because a real gate can have a name that
    starts with ``save_``. Source: the NWQLib Aer gate-accounting derivation.
    """
    from qiskit.circuit import Barrier, Delay
    from qiskit_aer.library.save_instructions.save_data import SaveData

    if circuit.has_control_flow_op():
        raise ValueError("a coherent trajectory operation count needs a fixed native body")
    gates = 0
    saves = 0
    for item in circuit.data:
        operation = item.operation
        if isinstance(operation, SaveData):
            saves += 1
        elif not isinstance(operation, (Barrier, Delay)):
            gates += 1
    return gates, saves


def _aer_state_threads():
    """The positive state-update thread cap that Aer statevector simulations are given.

    The value, the larger of the logical CPU count and a positive
    ``OMP_NUM_THREADS``, is not a bound on Aer's automatic threading, whose
    maximum is the OpenMP maximum of the executing task. It is used as an
    explicit cap: ``_prepare_aer_execution`` passes it to Aer as
    ``max_parallel_threads``, and the Aer executor takes the minimum of an
    explicit positive cap and its automatic OpenMP maximum (one without
    OpenMP), so the state-update worker count does not exceed it, apart from
    Aer's explicit debugging overrides, which the statevector target does not
    enable. The trajectory memory check uses the same value
    (``connection._run_state_threads`` resolves it once per open Run). The cap can
    reduce parallelism. It does not price transport, fusion workspace,
    thread stacks or process RSS.
    """
    import os

    threads = os.cpu_count() or 1
    value = os.environ.get("OMP_NUM_THREADS", "")
    if value.strip().isdigit() and int(value) > 0:
        threads = max(threads, int(value))
    return threads


def _admit_aer_saved_buffers(observation, *, width, memory_mb, state_threads, additional_peak_bytes):
    """Admit the simulation-phase numerical arrays of an Aer trajectory before native lowering; return their bytes.

    For distinct probability save ``i`` of marginal width ``k_i >= 1``,
    ``save_probabilities`` produces a dense float64 vector of length
    ``N_i = 2**k_i``, and Aer keeps the saved results until completion, so
    the retained numerical payload is ``B_prob = 8 * sum_i 2**k_i``. Let
    ``S`` count distinct saved statevectors of amplitude and reduction
    points, ``E`` the distinct saved expectations and ``w`` the complete
    native width. The general marginal kernel of the Aer 0.17.2 CPU
    statevector implementation additionally allocates, per participating
    worker, a private probability vector and a uint64 index array of
    ``2**k_i`` entries each, a transient bounded by
    ``W_marginal = max_i (g_i * 16 * t_i * 2**k_i)`` with ``g_i`` indicating
    the general kernel rather than its sorted full-register fast path and
    ``t_i`` a positive upper bound on the state-update threads. The
    conservative simulation-phase envelope of the numerical arrays is

        B_sim,num = (1+S) * 16 * 2**w + B_prob + 8*E + W_marginal,

    checked against ``simulator_memory_mb * 2**20``. It uses the final
    retained-save population together with the largest transient. Declared
    positions are used before lowering resolves the end shorthand, which can
    overcount an eventual shared save: a safe early bound. Every power is
    refused before it is formed beyond the remaining budget.

    Distinct boundaries, ordered wire lists or readout views are different
    saved data. Premises: CPU, double precision, this kernel, one coherent
    shot, and the stated thread bound; not established for GPU, distributed or chunked
    simulation, or another Aer release. The complete transport, allocator,
    OpenMP stack, fusion workspace and host decode objects are not covered;
    ``additional_peak_bytes`` is an internal qualified allowance for them,
    and zero tests only the named numerical arrays. Source: the NWQLib
    saved-marginal memory derivation.
    """
    if observation.kind != "trajectory":
        raise ValueError("saved-buffer admission requires a trajectory")
    if (type(width) is not int or width < 0
            or type(memory_mb) is not int or memory_mb < 1
            or type(state_threads) is not int or state_threads < 1
            or type(additional_peak_bytes) is not int or additional_peak_bytes < 0):
        raise ValueError("invalid native memory-admission inputs")
    limit = memory_mb << 20
    used = 0

    def charge(amount, what):
        nonlocal used
        remaining = limit - used
        if amount > remaining:
            raise ValueError(
                f"{what}: counted={used}, requested={amount}, "
                f"remaining={remaining}, minimum={used + amount} bytes; "
                f"simulator_memory_mb={memory_mb} ({limit} bytes)"
            )
        used += amount

    def charge_power(coefficient, exponent, what):
        # Reject before forming an unaffordable power or product.
        capacity = (limit - used) // coefficient
        if exponent >= capacity.bit_length():
            raise ValueError(
                f"{what}: counted={used}, requested={coefficient}*2**{exponent}, "
                f"remaining={limit - used}, "
                f"minimum={used}+{coefficient}*2**{exponent} bytes; "
                f"simulator_memory_mb={memory_mb} ({limit} bytes)"
            )
        charge(coefficient << exponent, what)

    charge(additional_peak_bytes, "additional qualified peak storage")
    charge_power(16, width, "live complex128 state")
    state_positions = set()
    probability_saves = set()
    pauli_saves = set()
    largest_general_marginal = None
    for point in observation.positions:
        # A request through a readout view is distinct saved data.
        view = None if point.view is None else (point.view.tail, point.view.inverse)
        if view is not None and point.kind not in {"probabilities", "pauli_expectation"}:
            raise ValueError(f"Aer executes a readout view only for a Pauli or probability point, not {point.id!r}")
        if point.kind in {"amplitudes", "reduction"}:
            if point.position not in state_positions:
                charge_power(16, width, f"saved state for point {point.id!r}")
                state_positions.add(point.position)
        elif point.kind == "probabilities":
            identity = (point.position, point.qubits, view)
            if identity in probability_saves:
                continue
            k = len(point.qubits)
            if not 1 <= k <= width or any(q >= width for q in point.qubits):
                raise ValueError("probability wires must belong to the native layout")
            charge_power(8, k, f"saved probabilities for point {point.id!r}")
            probability_saves.add(identity)
            full_sorted = k == width and all(q == j for j, q in enumerate(point.qubits))
            if not full_sorted:
                largest_general_marginal = max(largest_general_marginal or 0, k)
        elif point.kind == "pauli_expectation":
            for label in point.labels:
                identity = (point.position, label, view)
                if identity not in pauli_saves:
                    charge(8, f"saved expectation for point {point.id!r}")
                    pauli_saves.add(identity)
        else:
            raise ValueError(f"unsupported trajectory point kind {point.kind!r}")
    if largest_general_marginal is not None:
        charge_power(16 * state_threads, largest_general_marginal,
                     "temporary probability and uint64 index arrays")
    return used


def _trajectory_phase_ledger(selected, keys, decompose, preparation):
    """Return the prefix phase ``Phi_k`` of each named statevector save, before lowering.

    ``selected`` is the bound circuit with the saves (and any view tails and
    inverses) inserted; the phase of a save is that of every instruction
    before it.

    Aer applies an executed circuit's global phase once, before its first
    instruction, so every saved state carries the phase ``theta_save`` of
    the whole lowered circuit, not that of its own prefix. Write the
    lowering of top-level instruction ``j`` as ``C_j = exp(i*phi_j) U_j``,
    ``U_j`` the emitted native sequence, and let ``phi_0`` be the logical
    circuit's own top-level phase, which the circuit-prefix convention
    includes in every prefix. The phase of boundary ``k`` is ``Phi_k = phi_0
    + sum_(j<=k) phi_j`` (modulo ``2*pi``), and the saved array is multiplied
    by ``exp(i*(Phi_k - theta_save))``; ``theta_save`` is the executed
    circuit's global phase. The receipt's native budget is modulo one common
    global phase and does not bound the exact prefix phase, this ledger's
    accumulation or the subtraction forming the correction angle, so a
    phase-defined output has no state-error budget
    (``core.planning.ReductionContext``). The multiplication by the actual
    stored factor has the finite modulo-phase envelope of
    ``_phase_product.phase_product_envelope``, which the receipt records
    (``_statevector_roundoff``) and every consumer of the corrected array
    carries.

    The ledger is produced by the same boundary-preserving lowering that
    executes: each top-level block of ``preparation`` is lowered once, and its
    wrapper's lowered definition carries ``phi_j``, which the final
    decomposition adds to the circuit phase. A native instruction adds zero.
    Another non-native instruction is lowered alone to read its phase.
    Source: the NWQLib intermediate-phase derivation (the prefix-phase ledger
    over the actually inserted circuit).
    """
    from qiskit_aer.library.save_instructions.save_data import SaveData

    shared = {id(block) for _, block in preparation._blocks.values()}
    total, ledger = float(selected.global_phase), {}
    for item in selected.data:
        operation = item.operation
        if isinstance(operation, SaveData):
            if operation.label in keys:
                ledger[operation.label] = total
            continue
        if decompose.is_native(operation):
            phase = 0.0
        elif id(operation) in shared:
            phase = float(operation.definition.global_phase)
        else:
            block = QuantumCircuit(operation.num_qubits, operation.num_clbits)
            block.append(operation, block.qubits, block.clbits)
            phase = float(_lower_aer_circuit(block, decompose)[0].global_phase)
        total += phase
    return ledger


def _phase_factor(angle):
    """The ``(real, imag)`` pair of the binary64 factor ``exp(i*angle)`` that corrects one saved state.

    It is formed once, at preparation, and stored in the native metadata, so
    submission and a restored preparation multiply by the same represented
    factor whose envelope the receipt records.
    """
    factor = complex(np.exp(1j * angle))
    return factor.real, factor.imag


def _statevector_roundoff(prepared):
    """Receipt-wide ``(t, e)`` envelope of the stored phase factors of ``prepared``.

    Each corrected save key's factor has its envelope
    ``_phase_product.phase_product_envelope(factor, D)`` with
    ``D = 2**native_width``; a save with no multiplication has ``(0, 0)``.
    The componentwise maxima over the receipt's save keys give one
    conservative pair, charged to every statevector consumer the receipt
    covers, including an uncorrected save when another key caused the
    maximum. Charges are never summed across shared consumers, and the
    multiplication is never applied twice. A preparation without stored
    factors, every non-trajectory readout included, multiplies no saved state
    and gives ``(0.0, 0.0)``. This is O(number of distinct saved keys)
    scalar work, with no scan or copy of a state.
    """
    from nwqlib._phase_product import phase_product_envelope

    dimension = 1 << prepared.circuit.num_qubits
    envelopes = [phase_product_envelope(complex(*factor), dimension)
                 for factor in prepared.metadata.get("statevector_phase_factors", {}).values()]
    return (max((t for t, _ in envelopes), default=0.0), max((e for _, e in envelopes), default=0.0))


def _restore_aer_readout(circuit, observation, boundaries=()):
    """Restore only backend-owned save classes lost by QPY, at their native positions.

    Prepared Run lowering inserts these top-level, default-subtype readouts.
    Logical user inputs cannot supply Aer directives. Keep the saved parameters
    and wire order; neither lower the circuit again nor move the observation to
    its earlier logical position. A trajectory's saves are planned again from
    its observation and the receipt's resolved ``boundaries``
    (``_trajectory_saves``), so every save, of every kind, is identified by
    its key, class, payload and wires.
    """
    if observation.kind == "trajectory":
        planned, _ = _trajectory_saves(observation, boundaries)
        saves, wires = [], []
        for save in planned:
            if save.kind == "pauli":
                saves.append(SaveExpectationValue(Pauli(save.payload), label=save.key))
                wires.append(tuple(circuit.qubits))
            elif save.kind == "probabilities":
                saves.append(SaveProbabilities(len(save.payload), label=save.key))
                wires.append(tuple(circuit.qubits[index] for index in save.payload))
            else:
                saves.append(SaveStatevector(circuit.num_qubits, label=save.key))
                wires.append(tuple(circuit.qubits))
    elif observation.kind == "pauli_expectation":
        saves = [SaveExpectationValue(Pauli(label), label=f"nwqlib_exp_{index}")
                 for index, label in enumerate(observation.labels)]
        wires = [tuple(circuit.qubits)] * len(saves)
    elif observation.kind == "probabilities":
        saves = [SaveProbabilities(len(observation.qubits))]
        wires = [tuple(circuit.qubits[index] for index in observation.qubits)]
    elif observation.kind == "amplitudes":
        saves = [SaveStatevector(circuit.num_qubits)]
        wires = [tuple(circuit.qubits)]
    elif observation.kind == "counts":
        return
    else:
        raise ValueError(f"unsupported saved Aer readout kind: {observation.kind!r}")

    def payload(parameters):
        """Normalize a saved expectation payload to ``((label, (re, im)), ...)`` for exact comparison.

        QPY can return lists or tuples and real scalar types that differ from
        the ones written. An expectation payload is a list of (Pauli string,
        (real coefficient, imaginary coefficient)). The scalars are compared
        exactly, and no operator action or tolerance applies.
        """
        if not isinstance(parameters, (tuple, list)):
            raise ValueError("saved Aer readout has an invalid Pauli payload")
        normalized = []
        for entry in parameters:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("saved Aer readout has an invalid Pauli payload")
            label, coefficients = entry
            if (not isinstance(label, str) or not isinstance(coefficients, (tuple, list))
                    or len(coefficients) != 2
                    or any(not isinstance(value, (int, float, complex, np.number)) for value in coefficients)):
                raise ValueError("saved Aer readout has an invalid Pauli payload")
            normalized.append((label, tuple(complex(value) for value in coefficients)))
        return tuple(normalized)

    expected = {(save.name, save.label): (save, qubits) for save, qubits in zip(saves, wires, strict=True)}
    seen = set()
    for index, item in enumerate(circuit.data):
        operation = item.operation
        key = operation.name, operation.label
        save, expected_qubits = expected.get(key, (None, None))
        if save is None:
            continue
        if (key in seen or operation.num_qubits != save.num_qubits
                or item.qubits != expected_qubits or item.clbits
                or payload(operation.params) != payload(save.params)):
            raise ValueError("saved Aer readout does not match its observation")
        seen.add(key)
        if type(operation) is Instruction:
            # QPY preserves these payloads but not the Aer class/_subtype.
            save.params = operation.params
            circuit.data[index] = item.replace(operation=save)
    if seen != expected.keys():
        raise ValueError("saved Aer circuit is missing its observation readout")


# Aer operations whose effect on the state norm the exact-probability roundoff
# derivation (_validation.py) does not bound. Supplied matrix, state or channel
# data carry their own unitarity or normalization defect. A mid-circuit
# measurement or reset renormalizes the state, and control flow makes the
# executed instruction count data dependent.
_WINDOW_EXCLUDED_OPERATIONS = frozenset({
    "unitary", "diagonal", "multiplexer", "initialize", "set_statevector",
    "kraus", "quantum_channel", "qerror_loc", "roerror", "measure", "reset",
    "if_else", "for_loop", "while_loop", "switch_case",
})
# Aer gates whose matrix rounds a sum of angles inside a complex exponential.
_PHASE_SUM_OPERATIONS = frozenset({"u", "u2", "u3", "cu", "cu2", "cu3", "mcu", "mcu2", "mcu3"})
# The Aer release whose fusion pass, kernels and gate matrices the derivation
# was read against. Another installed release is listed as an exclusion.
_ROUNDOFF_CHECKED_AER_VERSION = "0.17.2"


def _phase_sum(params):
    """Sum of |parameters| of a phase-sum gate, or infinity when one is not a bound real number."""
    try:
        return sum(abs(float(value)) for value in params)
    except (TypeError, ValueError):
        return float("inf")


def _probability_window_exclusions(circuit, operation_names, *, amplitudes):
    """Return the sorted labels of what the exact-probability derivation does not bound.

    ``operation_names`` are the top-level operation names of ``circuit``. A
    phase-sum gate is listed as ``"<name> phase sum"`` only when the sum of its
    |parameters| exceeds ``PHASE_SUM_LIMIT``, so the parameters are read only
    when such a gate is present. An amplitude readout adds the
    ``"amplitude-derived masses"`` label, because NWQLib forms its branch
    masses outside the derived readout term. An installed Aer release other
    than the checked one adds its own label.
    """
    from importlib.metadata import version
    from nwqlib._validation import PHASE_SUM_LIMIT

    excluded = set(operation_names) & _WINDOW_EXCLUDED_OPERATIONS
    if not _PHASE_SUM_OPERATIONS.isdisjoint(operation_names):
        for item in circuit.data:
            if item.name in _PHASE_SUM_OPERATIONS and _phase_sum(item.params) > PHASE_SUM_LIMIT:
                excluded.add(f"{item.name} phase sum")
    if amplitudes:
        excluded.add("amplitude-derived masses")
    if version("qiskit-aer") != _ROUNDOFF_CHECKED_AER_VERSION:
        excluded.add("unchecked qiskit-aer version")
    return tuple(sorted(excluded))


def _readout_population(circuit, save_type, labels):
    """Classify the history before each requested native save, never the later circuit.

    This is structural trajectory provenance, not a proof of state purity.
    Unresolved preceding control flow is unknown; preceding measurement/reset
    can condition the saved trajectory. No simulation or gate expansion occurs.
    """
    history = "unconditional"
    saved = {}
    for item in circuit.data:
        operation = item.operation
        if isinstance(operation, save_type) and operation.label in labels:
            if operation.label in saved:
                return "unknown"
            saved[operation.label] = history
        if isinstance(operation, ControlFlowOp):
            history = "unknown"
        elif isinstance(operation, (Measure, Reset)) and history != "unknown":
            history = "native_conditioned"
    if saved.keys() != labels or len(set(saved.values())) != 1:
        return "unknown"
    return next(iter(saved.values()))


def _prepare_aer_execution(
    backend_target: BackendTarget,
    circuit: QuantumCircuit,
    *,
    shots: int | None,
    seed: int | None,
    pauli_expectation_readout: tuple[int, tuple[str, ...]] | None = None,
    probability_qubits: tuple[int, ...] | None = None,
    trajectory: tuple | None = None,
    phase_keys: frozenset = frozenset(),
    views: dict | None = None,
    preparation: _AerPreparation | None = None,
    noise_model=None,
    simulator_memory_mb: int = 1024,
    state_threads: int | None = None,
    exact_synthesis=None,
) -> _PreparedAerExecution:
    """Finish native lowering and readout insertion without submitting a job.

    A statevector target returns amplitudes unless one readout selector is
    given. ``probability_qubits`` requests only the native probability marginal
    on those qubits, ordered least significant first, and its mapping omits
    exact zero entries. ``pauli_expectation_readout=(position, labels)``
    observes system-width Pauli labels before the instruction at ``position`` of
    the original circuit. A position equal to the circuit length observes after
    all of its instructions, including measurements.

    ``trajectory``, the ``(saves, points)`` plan of ``_trajectory_saves``, evaluates all selected point
    readouts while advancing one coherent simulator state. Each save is
    inserted before the instruction at its boundary of the original bound
    circuit, before lowering, so lowering cannot move a gate across it that
    acts on the saved wires, and no prefix is constructed again. Pauli saves
    (one ``save_expectation_value`` per label, each with its own key) leave
    the state unchanged. Nothing after the last save is executed, so the
    native operation count covers the circuit through the final observation.
    ``phase_keys`` names the statevector saves that serve a phase-sensitive
    point; each gets the phase of its own prefix (``_trajectory_phase_ledger``).
    ``views`` maps a readout view's tail and inverse definitions to their
    logical circuits. At a view's boundary the tail is applied, the view's
    saves follow, and the exact inverse of the same selected sequence is
    applied before continuation; the final observation needs no inverse when
    nothing follows it (``save_rotated_marginal`` of the readout-view
    derivation). Multiple views at one boundary each start from the restored
    continuation state.
    ``preparation`` may share bound definitions that its caller keeps immutable.
    A persistent prepared-handle caller privately owns those source definitions
    and the fresh logical container; this function creates its execution circuit.
    ``state_threads`` is the positive state-update thread cap of the
    statevector target, passed to Aer as ``max_parallel_threads`` and used by
    the trajectory memory admission (``_admit_aer_saved_buffers``); None
    resolves ``_aer_state_threads()``.
    ``exact_synthesis`` synthesizes the dense unitaries on a noise target
    whose gate basis omits the unitary instruction (``_AerDecompose``), and
    the noiseless targets apply them as matrices.
    """

    simulator_memory_mb = integer(simulator_memory_mb, "simulator_memory_mb", 1)
    if backend_target.name == AER_STATEVECTOR_TARGET.name:
        # Aer documents zero_threshold (default 1e-10) as the cutoff for
        # truncating small values in result data. Every exact readout on this
        # target (amplitudes, probability marginals and Pauli expectations)
        # sets it to zero, so the truncation cannot remove a small nonzero
        # value and readout accuracy does not depend on Aer's default.
        # backends/connection.py records the effective value in the preparation
        # metadata, and restore_native_data reuses it.
        # The explicit thread cap cannot raise the executor's automatic
        # OpenMP maximum; it is the cap the trajectory memory check uses.
        simulator = AerSimulator(
            method="statevector",
            device="CPU",
            precision="double",
            seed_simulator=seed,
            max_memory_mb=simulator_memory_mb,
            zero_threshold=0.0,
            max_parallel_threads=integer(_aer_state_threads() if state_threads is None else state_threads,
                                         "state_threads", 1),
        )
        decompose = _AerDecompose(simulator.target, exact_synthesis=exact_synthesis)
        # Preserve shared source identities before final-measurement removal
        # makes its copy. Gate substitution preserves instruction positions.
        if trajectory is not None and preparation is None:
            # A trajectory's phase ledger reads each top-level block's lowered phase.
            preparation = _AerPreparation()
        observed = circuit if preparation is None else preparation.prepare(circuit, decompose)
        ledger = {}
        if pauli_expectation_readout is not None:
            position, labels = pauli_expectation_readout
            selected = observed.copy_empty_like()
            selected.data = observed.data[:position]
            for index, label in enumerate(labels):
                selected.save_expectation_value(Pauli(label), selected.qubits, label=f"nwqlib_exp_{index}")
            selected.data.extend(observed.data[position:])
            observed = selected
        elif trajectory is not None:
            saves = trajectory[0]
            tails = {name: preparation.prepare(tail, decompose) for name, tail in (views or {}).items()}
            selected = observed.copy_empty_like()
            start = 0
            for index, save in enumerate(saves):
                group = save.boundary, save.view
                if index == 0 or (saves[index - 1].boundary, saves[index - 1].view) != group:
                    selected.data.extend(observed.data[start:save.boundary])
                    start = save.boundary
                    if save.view is not None:
                        selected.compose(tails[save.view[0]], inplace=True, copy=False)
                if save.kind == "pauli":
                    selected.save_expectation_value(Pauli(save.payload), selected.qubits, label=save.key)
                elif save.kind == "probabilities":
                    selected.save_probabilities([selected.qubits[qubit] for qubit in save.payload], label=save.key)
                else:
                    selected.save_statevector(label=save.key)
                last = index + 1 == len(saves)
                if save.view is not None and not last and (saves[index + 1].boundary, saves[index + 1].view) != group:
                    selected.compose(tails[save.view[1]], inplace=True, copy=False)
            observed = selected
            if phase_keys:
                ledger = _trajectory_phase_ledger(selected, phase_keys, decompose, preparation)
        # Insert observations before DAG-based removal can reorder independent gates.
        simulation_circuit = _remove_final_measurements(observed)
        # Custom gates sharing the name "measure" survive logical removal, so
        # their counts cancel here. This delta describes removal only, never
        # the measurements that become visible after native lowering.
        removed_final_measurements = circuit.count_ops().get("measure", 0) > simulation_circuit.count_ops().get("measure", 0)
        execution_circuit, preprocessing_metadata, remaining_measurements, native_removed = _prepare_execution_circuit(
            simulation_circuit,
            decompose=decompose,
            add_save_statevector=probability_qubits is None and pauli_expectation_readout is None and trajectory is None,
            probability_qubits=probability_qubits,
            remove_final_measurements=True,
        )
        removed_final_measurements |= native_removed
        conditioned = None if remaining_measurements is None else remaining_measurements > 0
        readout_population = None
        if trajectory is not None:
            readout_population = _readout_population(
                execution_circuit, (SaveExpectationValue, SaveProbabilities, SaveStatevector),
                {save.key for save in trajectory[0]})
        elif pauli_expectation_readout is not None:
            labels = {f"nwqlib_exp_{index}" for index in range(len(pauli_expectation_readout[1]))}
            readout_population = _readout_population(execution_circuit, SaveExpectationValue, labels)
        elif probability_qubits is not None:
            readout_population = _readout_population(execution_circuit, SaveProbabilities, {"probabilities"})
        metadata = {
            "seed_simulator": seed,
            "removed_final_measurements": removed_final_measurements,
            "execution_measurement_count": remaining_measurements,
            "statevector_is_measurement_conditioned": conditioned,
            "readout_population": readout_population,
            "statevector_semantics": (
                "unknown_control_flow" if conditioned is None else
                "measurement_conditioned" if conditioned else "pre_final_measurement"
            ),
            "execution_preprocessing": preprocessing_metadata,
            "probability_qubits": probability_qubits,
            "pauli_expectation_readout": pauli_expectation_readout,
            "simulation_trajectory_count": 1,
            # The explicit thread cap, reapplied on restoration.
            "simulator_max_parallel_threads": simulator.options.max_parallel_threads,
            # Phase-sensitive saved states: the (real, imag) pair of the
            # factor exp(i*(Phi_k - theta_save)) per save key, formed once
            # here. Submission multiplies by this stored factor, also after
            # restoration, and the receipt's envelope is that of this factor.
            **({"statevector_phase_factors": {
                key: _phase_factor(phase - float(execution_circuit.global_phase))
                for key, phase in ledger.items()}} if ledger else {}),
            "probability_window_exclusions": _probability_window_exclusions(
                execution_circuit, preprocessing_metadata["execution_operation_names"],
                amplitudes=(any(save.kind == "statevector" for save in trajectory[0]) if trajectory is not None
                            else probability_qubits is None and pauli_expectation_readout is None)),
        }
    else:
        shots = integer(shots, "shots", 1)
        if not circuit.num_clbits:
            raise ValueError("shot-based execution requires at least one measurement")
        simulator = AerSimulator(seed_simulator=seed, max_memory_mb=simulator_memory_mb,
                                 **({"noise_model": noise_model} if noise_model is not None else {}))
        decompose = _AerDecompose(simulator.target, exact_synthesis=exact_synthesis)
        translate_basis = noise_model is not None and "u" not in decompose.native_types
        if translate_basis:
            # StatePreparation reaches canonical U, whose primitive definition
            # cannot be decomposed further. Noise selects a narrower gate basis;
            # admit U only as an intermediate, then translate to that real basis.
            from qiskit.circuit.library import UGate
            decompose.native_types["u"] = UGate
        execution_circuit, preprocessing_metadata, measurement_count, _ = _prepare_execution_circuit(
            circuit if preparation is None else preparation.prepare(circuit, decompose),
            decompose=decompose,
            add_save_statevector=False,
        )
        if translate_basis:
            from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
            from qiskit.transpiler.passes import BasisTranslator
            execution_circuit = BasisTranslator(SessionEquivalenceLibrary, list(simulator.target.operation_names))(execution_circuit)
            preprocessing_metadata.update(native_basis_translation="Qiskit BasisTranslator for the selected noise-model target",
                execution_operation_names=tuple(sorted(execution_circuit.count_ops())))
        if measurement_count is None:
            # Presence can be known without inferring a branch/loop execution
            # count. Inspect lowered block syntax once, never unroll a loop.
            pending = [execution_circuit]
            inspected = set()
            has_measurement = False
            while pending and not has_measurement:
                block = pending.pop()
                if id(block) in inspected:
                    continue
                inspected.add(id(block))
                for item in block.data:
                    if isinstance(item.operation, Measure):
                        has_measurement = True
                        break
                    if isinstance(item.operation, ControlFlowOp):
                        pending.extend(item.operation.blocks)
        else:
            has_measurement = measurement_count > 0
        if not has_measurement:
            raise ValueError("shot-based execution requires at least one measurement")
        metadata = {
            "seed_simulator": seed,
            "execution_preprocessing": preprocessing_metadata,
            "execution_measurement_count": measurement_count,
        }
    return _PreparedAerExecution(
        simulator, execution_circuit, backend_target, shots,
        pauli_expectation_readout, probability_qubits, metadata, trajectory,
    )


def _submit_aer_execution(prepared: _PreparedAerExecution) -> BackendRunResult:
    """Submit the already native artifact; no lowering, inventory or readout edits."""
    circuit = prepared.circuit
    # Exact readouts, a trajectory schedule of several points included, run
    # one pure-state trajectory (ENGINEERING_CONSTANTS.md, "Budgets and
    # mechanical bounds", row "Aer STATEVECTOR trajectory count"). Counts use
    # the selected shot budget.
    job = prepared.simulator.run(circuit, shots=prepared.shots or 1)
    result = job.result()
    if prepared.trajectory is not None:
        # Values are keyed by point, and a Pauli value by (point, label):
        # the same label at two points is two values.
        mode = ExecutionMode.STATEVECTOR
        data = result.data(0)
        saves, association = prepared.trajectory
        factors, states = prepared.metadata.get("statevector_phase_factors", {}), {}
        points = {}
        for point, kind, keys in association:
            if kind == "pauli_expectation":
                points[point] = {"pauli_expectations": {label: float(data[key]) for label, key in keys}}
            elif kind == "probabilities":
                # As for one marginal: the dense float64 buffer, qubits[0]
                # the least significant index bit, handed to publication.
                points[point] = {"probabilities": data[keys]}
            else:
                if keys not in states:
                    # A save carries the phase of the whole executed circuit;
                    # a phase-sensitive point gets its own prefix's phase.
                    # The correction multiplies the result's own array in
                    # place by the factor stored at preparation, once per
                    # save key, so no second state is held and every point
                    # sharing the key reads the same corrected array.
                    states[keys] = data[keys].data
                    if keys in factors:
                        states[keys] *= complex(*factors[keys])
                points[point] = {"statevector": states[keys]}
        raw_output = {"trajectory": points}
    elif prepared.backend_target.name == AER_COUNTS_TARGET.name:
        # Qiskit count keys already put classical bit 0 rightmost, with a space
        # between registers. The common decoder removes those spaces.
        raw_output = {"counts": {key: int(value) for key, value in result.get_counts(circuit).items()}}
        mode = ExecutionMode.SHOTS
    else:
        mode = ExecutionMode.STATEVECTOR
        if prepared.pauli_expectation_readout is not None:
            raw_output = {"pauli_expectations": {
                label: float(result.data(0)[f"nwqlib_exp_{index}"])
                for index, label in enumerate(prepared.pauli_expectation_readout[1])
            }}
        elif prepared.probability_qubits is not None:
            # Aer indexes the marginal with probability_qubits[j] as bit j, the
            # common probability-index convention. Its dense float64 buffer is
            # handed to publication, which keeps it without a copy on the dense
            # branch and forms indices only on the sparse branch.
            raw_output = {"probabilities": result.data(circuit)["probabilities"]}
        else:
            raw_output = {"statevector": result.get_statevector(circuit).data}
    return BackendRunResult(
        execution_mode=mode,
        backend_target=prepared.backend_target,
        raw_output=raw_output,
        metadata=dict(prepared.metadata, native_job_id=job.job_id()),
    )
