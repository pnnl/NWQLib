"""Qiskit construction of the QHD circuit that planning chose, from the data recorded in its Plan.

The circuit is built from the data of the Plan, the reconstruction's initial
amplitudes, support tables, step rows, compiled schedule and physical phase,
with the grid rebuilt from the Problem's bounds and the Method's options. The
objective is not evaluated, and the one-hot blocks are derived from the
recorded data with the arithmetic of planning, so a loaded Plan yields the
circuit whose resources planning counted.
"""

from .grid import OneHotGrid
from .initial_state import chain_links
from .onehot_schedule import derivation_bytes
from nwqlib.blocks.selection import NATIVE_POSITION_BYTES, _no_arguments
from nwqlib.subroutines.hamiltonian_evolution.pauli_ir import (
    PauliEvolutionBlock,
    PauliEvolutionTerm,
)

# Coarse bytes per Aer native position of a whole QHD circuit at the peak of its
# construction and lowering, with the gate definition and the flattened circuit
# together. Fresh-process footprint growth was 972 to 1208 bytes per position for
# one-hot K = 20 and 26 up to 10**7 positions, and 1604 to 2740 for binary K = 256
# and K = 64, d = 2 (Qiskit 2.6.0, Aer 0.18.0). These are estimates, not bounds.
ONEHOT_POSITION_BYTES = 1024
BINARY_POSITION_BYTES = 2048


def raw_blocks(reconstruction, method, grid):
    """Yield the evolution blocks of a one-hot Plan one at a time, in circuit order.

    The blocks of each step are derived from the Plan's tables, step rows
    and projector providers (``onehot_schedule.OneHotSchedule.blocks``),
    with one tile of table entries in memory at a time. A kinetic block
    becomes its XX and YY terms on its pair with its coefficient, and keeps
    its pair and angle. A projector block keeps its support and angle, so
    the direct ``ir_product`` classical evaluation reads both kinds without
    parsing labels. The circuit construction and the ``ir_product``
    classical evaluation (``theory._run_ir_product``) therefore hold one
    block at a time instead of a copy of every block.
    """
    from .onehot_schedule import OneHotSchedule

    schedule = OneHotSchedule.of(reconstruction, method, grid)
    for step in range(len(reconstruction.step_weights)):
        for block in schedule.blocks(step):
            if block.kind == "kinetic":
                left, right = block.support
                terms = tuple(
                    PauliEvolutionTerm(
                        pauli=f"{axis}{left}{axis}{right}", coefficient=block.coefficient
                    )
                    for axis in "xy"
                )
            else:
                terms = ()
            yield PauliEvolutionBlock(
                kind=block.kind,
                terms=terms,
                time_step=block.time_step,
                time=block.time,
                support=block.support,
                angle=block.angle,
            )


def _state_preparation_positions(width):
    """Return the Aer native positions of Qiskit's ``StatePreparation`` on ``width`` qubits, 2**(width+1) - width - 2.

    Measured on dense and one-hot vectors of 2, 3, 4 and 6 qubits (Qiskit
    2.6.0, Aer 0.18.0). A uniform vector, whose rotations vanish, has only
    ``width`` positions, which is small next to the rest of a binary circuit.
    """
    return (2 << width) - width - 2


def _onehot_positions(reconstruction, method, d):
    """Return the Aer native positions of a one-hot QHD circuit from the Plan's block counts.

    A hopping block lowers to 14 positions. A projector block on s qubits is
    one multi-controlled phase, or for the diagonal provider
    (``structured_number_projector_provider``) 2**s - 2 CX and 2**s - 1 RZ.
    The structured preparation has d X and one CRY and one CX per emitted
    link (``initial_state.chain_links``). This count equals the lowered circuit
    at K = 4, 8 and 20 with Qiskit 2.6.0 and Aer 0.18.0.
    """
    schedule = reconstruction.onehot
    providers = {p.support_size: p.provider for p in schedule.providers}
    positions = 14 * schedule.hopping_blocks + sum(
        count * ((2 << s) - 3 if providers[s] == "diagonal_synthesis" else 1)
        for s, count in schedule.projector_blocks)
    if method.init_prep == "structured":
        positions += d + 2 * sum(chain_links(vector) for vector in reconstruction.initial_amplitudes)
    elif method.init_prep == "qiskit_state_preparation":
        positions += d * _state_preparation_positions(method.num_grid_points)
    return positions


def _onehot_native_bytes(reconstruction, method, d):
    """Return a coarse estimate of the bytes of building and lowering a one-hot QHD circuit.

    The circuit counts ``ONEHOT_POSITION_BYTES`` per position
    (``_onehot_positions``). Each ``StatePreparation`` keeps its ``2**K`` complex
    amplitudes, and the blocks of each step are derived with the transient
    storage of ``onehot_schedule.derivation_bytes``. Planning records the
    estimate as the preparation workspace (``method.QHD._select_native``),
    the Run compares it with ``max_bytes`` (``_construction_memory``), and
    ``construct_qhd`` checks it before it allocates the circuit for a caller
    outside a Run.
    """
    schedule = reconstruction.onehot
    k = method.num_grid_points
    preparation = 16 * d * (1 << k) if method.init_prep == "qiskit_state_preparation" else 0
    return (ONEHOT_POSITION_BYTES * _onehot_positions(reconstruction, method, d) + preparation
            + derivation_bytes(k, d, (p.support_size for p in schedule.providers)))


def _binary_positions(reconstruction, method, d):
    """Return the Aer native positions of a binary QHD circuit from the Plan's block counts.

    The structured preparation has d b H gates. A kinetic block adds its
    forward and inverse QFTs, one position for each H, CP and swap, in place
    of the pair's CX and rotations that its recorded counts include. A
    dense diagonal lowers to exactly its CX and rotations, and an all-zero
    table to none. A Walsh diagonal has at most its CX and rotations, since
    Aer keeps a weight-two parity rotation as one RZZ and the recorded count
    includes its two CX.
    """
    k = method.num_grid_points
    bits = k.bit_length() - 1
    cutoff = method.binary_synthesis.aqft_cutoff
    m = bits - 1 if cutoff is None else min(cutoff, bits - 1)
    cp = m * bits - m * (m + 1) // 2
    swaps = bits // 2 if method.binary_synthesis.qft_bit_reversal == "swap" else 0
    positions = 0
    if method.init_prep == "structured":
        positions = d * bits
    elif method.init_prep == "qiskit_state_preparation":
        positions = d * _state_preparation_positions(bits)
    pair = 2 * (bits + cp + swaps) - (4 * cp + 6 * swaps + 6 * cp)
    for group in reconstruction.steps:
        for block in group:
            positions += block.cx + block.rotations + (pair if block.kind == "binary_kinetic" else 0)
    return positions


def _binary_native_bytes(reconstruction, method, d):
    """Return a coarse estimate of the bytes of a binary QHD circuit and one block's construction arrays.

    The circuit counts ``BINARY_POSITION_BYTES`` per position
    (``_binary_positions``), 16384 bytes for its circuit and register objects,
    the allowance of the dense syntheses
    (``_dense_synthesis.controlled_synthesis_size``), and for
    ``StatePreparation`` its d*K complex amplitudes. These fixed allowances
    count twice, for the source and its appended copy.
    Both callers exclude the resource-only recipe first
    (``method.QHD._select_binary_native``, ``construct_qhd``).

    The scratch is 65536+(320+L(n_max))*E_max plus
    256*n_max*(1+digits(max(0,d*b-1))) for labels/qubit lists. It permits
    padded floats, wrapping/transform arrays, Gray indices and controls,
    schedule pairs, and the two Walsh tolist lists. L is
    _integer_storage.
    """
    from .binary import _integer_storage

    k = method.num_grid_points
    bits = k.bit_length() - 1
    amplitudes = 16 * d * k if method.init_prep == "qiskit_state_preparation" else 0
    widest = max((bits * len(block.variables) for group in reconstruction.steps for block in group),
                 default=bits)
    largest = 1 << widest
    scratch = (65536 + (320 + _integer_storage(widest)) * largest
               + 256 * widest * (1 + len(str(max(0, d * bits - 1)))))
    return (BINARY_POSITION_BYTES * _binary_positions(reconstruction, method, d)
            + 2 * (16384 + amplitudes) + scratch)


def _binary_held_bytes(reconstruction, method, d):
    """Return the bytes that a binary construction holds beside its model, and the model's reservation.

    The held bytes are the Plan's source records and ``_binary_native_bytes``.
    The reservation is that of ``binary._model_reservation``, ``(model, latest,
    build, use)``.
    """
    from .binary import _model_reservation
    from .method import _binary_source_reservation

    k = method.num_grid_points
    specs = tuple((int(t.values.array.size), len(t.support)) for t in reconstruction.support_values)
    source_bytes, _ = _binary_source_reservation(d, k, specs, method.num_steps, method.trotter_order)
    parts = _model_reservation(
        d, k.bit_length() - 1, specs,
        cutoff=method.binary_synthesis.aqft_cutoff,
        swaps=method.binary_synthesis.qft_bit_reversal == "swap",
        potential_walsh=method.binary_synthesis.potential != "dense_diagonal")
    return source_bytes + _binary_native_bytes(reconstruction, method, d), parts


def _binary_workspace_bytes(reconstruction, method, d):
    """Return the coarse bytes of a binary construction, its held bytes, model and largest use."""
    held, parts = _binary_held_bytes(reconstruction, method, d)
    return held + parts[0] + parts[1] + max(parts[2:])


def _construction_memory(reconstruction, method, d):
    """Return ``(native_bytes, definition_bytes)`` of the QHD block for ``SelectedBlock.bind``.

    The Plan's preparation workspace is split so that the Run's comparison with
    ``max_bytes`` sees the same total. The flattened circuit that a prepared
    handle holds until it is released counts ``NATIVE_POSITION_BYTES`` per
    position, and the rest of the estimate belongs to the gate definition that the
    Run holds. Planning and archive rebinding share this function. A
    resource-only recipe builds no circuit, so it has no estimate.
    """
    if method.init_prep == "none":
        return None, None
    if method.encoding == "binary":
        total, positions = (_binary_workspace_bytes(reconstruction, method, d),
                            _binary_positions(reconstruction, method, d))
    else:
        total, positions = (_onehot_native_bytes(reconstruction, method, d),
                            _onehot_positions(reconstruction, method, d))
    native = NATIVE_POSITION_BYTES * positions
    return native, total - native


def construct_qhd(block, arguments, method_context):
    """Build the QHD circuit from the Plan's blocks.

    The circuit prepares the Method's initial state in every variable register
    from the per-variable vectors that planning saved
    (``QHDReconstruction.initial_amplitudes``) with the selected recipe. The
    saved vectors are the only evaluation of the initial state, so the
    circuit, the classical evaluation and the preparation entry of the error
    sources use the same amplitudes, and loading a saved Plan evaluates
    nothing. For the uniform state, and for the kinetic ground state on the
    periodic grid, the classical evaluation fills ``1/sqrt(K**d)`` directly
    (``QHDReconstruction.initial_amplitudes``).
    One-hot (``d*K`` qubits): the structured amplitude chain or Qiskit
    ``StatePreparation`` (``initial_state.append_initial_state``), then each
    fused XX+YY hopping and number-projector block in the Plan's order
    (``raw_blocks``). The recorded projector providers must be those of the
    tables' support sizes, and the emitted blocks must have the recorded
    counts by kind and support size (``QHDOneHotSchedule``), or the
    construction raises, because a circuit with other blocks would not be
    the circuit that the Plan's CX and rotation counts describe. Its byte
    estimate (``_onehot_native_bytes``) is checked against ``max_bytes``
    before the circuit is allocated, also for a loaded Plan. Binary
    (``d*b`` qubits): the structured H layer or Qiskit ``StatePreparation``
    (``initial_state.append_binary_initial_state``), then the Plan's binary
    blocks (``append_binary_steps``). Adding the recorded ``physical_phase`` to
    the circuit's global phase restores the identity components that the
    compiler omitted, so the circuit's phase agrees with the product formula
    of the full Hamiltonian, stencil diagonal and constant objective included.
    """
    _no_arguments(arguments)
    import numpy as np
    from qiskit import QuantumCircuit
    from .initial_state import append_binary_initial_state, append_initial_state
    from nwqlib.subroutines.hamiltonian_evolution import append_pauli_evolution_block

    reconstruction, options, bounds, variables = block._payload
    if block.record.blocker or options.init_prep == "none":
        raise ValueError("init_prep='none' is resource-only")
    grid = OneHotGrid(variables, bounds, options.num_grid_points, options.boundary_points,
                      options.boundary)
    if options.encoding == "binary":
        from .binary import _construction_cache_capacity

        native_held, parts = _binary_held_bytes(reconstruction, options, grid.num_variables)
        _construction_cache_capacity(
            options.max_bytes, held_bytes=native_held,
            model_bytes=parts[0], latest_bytes=parts[1], build_bytes=parts[2], use_bytes=parts[3])
    else:
        from nwqlib._limits import check_memory

        check_memory(_onehot_native_bytes(reconstruction, options, grid.num_variables), options.max_bytes,
                     "QHD one-hot circuit construction")
    circuit = QuantumCircuit(reconstruction.width, name="qhd")
    amplitudes = tuple(np.asarray(vector, dtype=float) for vector in reconstruction.initial_amplitudes)
    if options.encoding == "binary":
        bits = options.num_grid_points.bit_length() - 1
        append_binary_initial_state(circuit, grid, bits, amplitudes, recipe=options.init_prep)
        append_binary_steps(circuit, reconstruction, options, grid, held_bytes=native_held)
    else:
        from collections import Counter

        from .onehot_schedule import projector_providers

        recorded = reconstruction.onehot
        if recorded.providers != projector_providers(reconstruction.support_values):
            raise ValueError("QHD one-hot projector providers differ from those of the Plan's support sizes")
        append_initial_state(circuit, grid, amplitudes, recipe=options.init_prep)
        # Emitted blocks by support size, with 0 for the hopping blocks.
        emitted = Counter()
        for compact in raw_blocks(reconstruction, options, grid):
            append_pauli_evolution_block(circuit, compact)
            emitted[0 if compact.kind == "kinetic" else len(compact.support)] += 1
        if +emitted != +Counter({0: recorded.hopping_blocks, **dict(recorded.projector_blocks)}):
            raise ValueError("QHD one-hot blocks differ from the Plan's recorded block counts")
    circuit.global_phase += reconstruction.physical_phase
    return circuit


def append_binary_steps(circuit, reconstruction, method, grid, *, held_bytes):
    """Append the Plan's binary blocks of every step, in order, to a circuit on the ``d*b`` register.

    A potential block synthesizes its support table's diagonal on the qubits
    of its variables in increasing order, local bit ``t b + l`` on qubit
    ``j_t b + l`` (``binary.address_table``). A kinetic block on variable j
    applies the selected QFT, its phase diagonal and the exact inverse QFT on
    qubits ``j b`` to ``j b + b - 1`` (``binary.kinetic_table``). A dense
    diagonal goes to ``append_control_diagonal_phases``, which keeps the
    table's identity phase in its gate. A Walsh string with mask m becomes
    ``Rz(theta_m)`` on the parity of its qubits
    (``apply_pauli_rotation`` of the Z string, ``exp(-i theta Z.../2)``), and the
    block's identity phase ``-x c_0`` (``DiagonalConstruction.identity_phase``)
    is added to the circuit's global phase. Each block's synthesis is rebuilt
    from the recorded exponent and choice (``binary.BinaryModel``) and must
    reproduce the recorded CX and rotation counts, or the construction
    raises. Planning formed the Plan's CX and rotation counts from those
    block counts (``method.QHD._select_binary_native``), and a circuit with
    other counts would not be the circuit that the Plan's counts describe.
    The physical phase of the constant objective is not added here.

    The caller counts the Plan's records, the completed QHD circuit graph
    and one block's construction scratch in held_bytes before circuit
    allocation, and the binary model checks those bytes, together with its
    own arrays, metadata and construction workspace, against ``max_bytes``.
    The graph allowance covers the definitions and parameters that this
    function creates, while subsequent SDK synthesis and process RSS have
    separate scope.
    """
    from .binary import BinaryModel

    model = BinaryModel(grid, method, reconstruction.support_values, held_bytes=held_bytes)
    bits = model.bits
    # CX and rotations of the emitted QFT pair, which the recorded counts of a
    # kinetic block include and model.construction, the phase diagonal alone,
    # does not. Qiskit 2.6.0 defines a controlled phase by two CX and three
    # phase gates and a swap by three CX.
    pair = (2 * sum(3 if g[0] == "swap" else 2 if g[0] == "cp" else 0 for g in model.forward),
            2 * 3 * sum(g[0] == "cp" for g in model.forward))

    def gates(sequence, offset):
        for gate in sequence:
            if gate[0] == "h":
                circuit.h(offset + gate[1])
            elif gate[0] == "cp":
                circuit.cp(gate[1], offset + gate[2], offset + gate[3])
            else:
                circuit.swap(offset + gate[1], offset + gate[2])

    for step in reconstruction.steps:
        for block in step:
            built = model.construction(block.kind, block.variables, block.exponent, block.synthesis)
            if block.kind == "binary_kinetic":
                offset = block.variables[0] * bits
                gates(model.forward, offset)
                append_diagonal(circuit, built, [offset + level for level in range(bits)])
                gates(model.inverse, offset)
                expected = pair
            else:
                append_diagonal(circuit, built, [j * bits + level for j in block.variables for level in range(bits)])
                expected = (0, 0)
            if (block.cx - built.cx, block.rotations - built.rotations) != expected or (
                    built.synthesis != block.synthesis):
                raise ValueError("QHD binary block differs from its recorded synthesis")


def append_diagonal(circuit, built, qubits):
    """Append one ``binary.DiagonalConstruction`` on ``qubits``, local address bit r on ``qubits[r]``.

    A dense construction goes to ``append_control_diagonal_phases`` with its
    phases, which adds no gate for an all-zero table and otherwise keeps
    the identity phase inside its gate. A Walsh construction adds its
    identity phase to the circuit's global phase and applies
    ``Rz(theta_m) = exp(-i theta_m Z/2)`` on the parity of each kept string
    (``apply_pauli_rotation`` of the Z string: one Rz for weight one, one
    ``RZZ`` of two CX for weight two, and a CX ladder of ``2 (w - 1)`` CX
    otherwise).
    """
    from nwqlib.subroutines._multiplexors import append_control_diagonal_phases
    from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import apply_pauli_rotation

    if built.phases is not None:
        append_control_diagonal_phases(circuit, [circuit.qubits[q] for q in qubits], built.phases)
        return
    circuit.global_phase += built.identity_phase
    for mask, angle in zip(built.masks.tolist(), built.angles.tolist(), strict=True):
        label = "".join(f"z{qubits[r]}" for r in range(len(qubits)) if (mask >> r) & 1)
        apply_pauli_rotation(circuit, label, angle)
