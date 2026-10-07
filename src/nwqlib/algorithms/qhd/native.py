"""Native Qiskit construction of the selected QHD circuit from its stored blocks.

The circuit is built from stored Plan data, the reconstruction's initial
amplitudes, compiled step blocks, support tables (binary) and physical phase,
with the grid rebuilt from the Problem's bounds and the Method's options. The
objective is not evaluated and the schedule is not recompiled, so a loaded Plan
yields the circuit that planning priced.
"""

from .grid import OneHotGrid
from nwqlib.blocks.selection import _no_arguments
from nwqlib.subroutines.hamiltonian_evolution.pauli_ir import (
    PauliEvolutionBlock,
    PauliEvolutionTerm,
)


def raw_blocks(reconstruction):
    """Yield the evolution blocks of the stored ``QHDBlock`` records one at a time, in circuit order.

    A kinetic record becomes its XX and YY terms on the stored pair with the
    stored coefficient, and keeps its pair and angle. A projector record keeps
    its support and angle, so the direct ``ir_product`` kernel reads both
    kinds without parsing labels. Each block is rebuilt when it is consumed,
    so the circuit builder and the classical ``ir_product`` kernel
    (``theory._run_ir_product``) hold one block at a time instead of a copy
    of every stored block.
    """
    for group in reconstruction.steps:
        for block in group:
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


def _binary_native_bytes(reconstruction, method, d):
    """Reserve the known QHD circuit graph and one block's construction scratch.

    Count C circuit containers, Q local wire positions, G gate/instruction
    positions, A qubit-argument positions and V scalar-parameter positions.
    The qualified graph allowance is 65536+4096*C+512*Q+2048*G+32*A+64*V
    on CPython 3.12.14 and Qiskit 2.5.2. Two graph allowances permit source
    and appended-copy coexistence. These rates price named objects and
    storage, not process RSS or arbitrary SDK synthesis. Revisit when the
    builder, the copying of gate definitions, the parameter representation
    or the SDK version changes, or a measured workload falls outside the
    checked range (docs/ENGINEERING_CONSTANTS.md).

    A nonzero dense n-qubit diagonal has E=2**n entries. It adds n+1
    circuits of total width q=n+n*(n+1)//2, 2*E+n-2 gate positions,
    q+3*E-5 arguments and 3*E-2 parameters. Its outer and multiplexor
    gates store E+(E-1) phase parameters, and up to E-1 Rz parameters
    appear in their definitions. A Walsh diagonal uses at most cx+r
    instructions, 2*cx+r arguments and r parameters. Remove the QFT pair's
    CX/rotation counts before applying that Walsh law. Each kinetic block
    also adds the actual forward/inverse H, CP and swap populations.

    Structured preparation contributes d*b H gates. StatePreparation
    contributes d gates and d*K complex parameters, with its later lazy
    synthesis outside this constructor. Both callers exclude the resource-only
    recipe first (``method.QHD._select_binary_native``, ``construct_qhd``).
    A zero dense block adds no graph but keeps its padding scratch charge.

    The scratch is 65536+(320+L(n_max))*E_max plus
    256*n_max*(1+digits(max(0,d*b-1))) for labels/qubit lists. It permits
    padded floats, wrapping/transform arrays, Gray indices and controls,
    schedule pairs, and the two Walsh tolist populations. L is
    _integer_storage. Built-in immutable SDK-singleton definitions and
    prior process-global caches are outside the new graph population.
    """
    from .binary import _integer_storage

    k = method.num_grid_points
    bits = k.bit_length() - 1
    cutoff = method.binary_synthesis.aqft_cutoff
    m = bits - 1 if cutoff is None else min(cutoff, bits - 1)
    cp = m * bits - m * (m + 1) // 2
    swaps = bits // 2 if method.binary_synthesis.qft_bit_reversal == 'swap' else 0
    pair_cx, pair_rot = 4 * cp + 6 * swaps, 6 * cp
    circuits, wires, gates, args, params = 1, d * bits, 0, 0, 0
    if method.initial_state_preparation == 'structured':
        gates += d * bits
        args += d * bits
    elif method.initial_state_preparation == 'qiskit_state_preparation':
        gates += d
        args += d * bits
        params += d * k
    largest = k
    widest = bits
    for group in reconstruction.steps:
        for block in group:
            n = bits * len(block.variables)
            e = 1 << n
            largest, widest = max(largest, e), max(widest, n)
            kinetic = block.kind == 'binary_kinetic'
            if kinetic:
                gates += 2 * (bits + cp + swaps)
                args += 2 * (bits + 2 * cp + 2 * swaps)
                params += 2 * cp
            cx = block.cx - (pair_cx if kinetic else 0)
            rotations = block.rotations - (pair_rot if kinetic else 0)
            if block.synthesis == 'dense_diagonal':
                if rotations:
                    local_wires = n + n * (n + 1) // 2
                    circuits += n + 1
                    wires += local_wires
                    gates += 2 * e + n - 2
                    args += local_wires + 3 * e - 5
                    params += 3 * e - 2
            else:
                gates += cx + rotations
                args += 2 * cx + rotations
                params += rotations
    graph = 65536 + 4096 * circuits + 512 * wires + 2048 * gates + 32 * args + 64 * params
    scratch = (65536 + (320 + _integer_storage(widest)) * largest
               + 256 * widest * (1 + len(str(max(0, d * bits - 1)))))
    return 2 * graph + scratch


def construct_qhd(block, arguments, method_context):
    """Build the QHD circuit from the Plan's stored blocks.

    The circuit prepares the Method's initial state in every variable register
    from the per-variable vectors that planning stored
    (``QHDReconstruction.initial_amplitudes``) with the selected recipe. The
    stored vectors are the one evaluation of the initial state, so the
    circuit, the classical kernel and the preparation charge of the error
    ledger use the same amplitudes, and loading a saved Plan evaluates
    nothing. For the uniform state, and for the kinetic ground state on the
    periodic grid, the classical kernel fills ``1/sqrt(K**d)`` directly
    (``QHDReconstruction.initial_amplitudes``).
    One-hot (``d*K`` qubits): the structured amplitude chain or Qiskit
    ``StatePreparation`` (``initial_state.append_initial_state``), then each
    fused XX+YY hopping and number-projector block in stored order. Binary
    (``d*b`` qubits): the structured H layer or Qiskit ``StatePreparation``
    (``initial_state.append_binary_initial_state``), then the stored binary
    blocks (``append_binary_steps``). Adding the stored ``physical_phase`` to
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
    if block.record.blocker or options.initial_state_preparation == "none":
        raise ValueError("initial_state_preparation='none' is resource-only")
    grid = OneHotGrid(variables, bounds, options.num_grid_points, options.include_boundary_points,
                      options.boundary)
    if options.encoding == "binary":
        from .binary import _construction_cache_capacity, _model_reservation
        from .method import _binary_source_reservation

        specs = tuple((int(t.values.array.size), len(t.support))
                      for t in reconstruction.support_values)
        source_bytes, _ = _binary_source_reservation(
            grid.num_variables, options.num_grid_points, specs,
            options.num_steps, options.trotter_order)
        native_held = source_bytes + _binary_native_bytes(reconstruction, options, grid.num_variables)
        parts = _model_reservation(
            grid.num_variables, options.num_grid_points.bit_length() - 1, specs,
            cutoff=options.binary_synthesis.aqft_cutoff,
            swaps=options.binary_synthesis.qft_bit_reversal == "swap",
            potential_walsh=options.binary_synthesis.potential != "dense_diagonal")
        _construction_cache_capacity(
            options.max_bytes, held_bytes=native_held,
            model_bytes=parts[0], latest_bytes=parts[1], build_bytes=parts[2], use_bytes=parts[3])
    circuit = QuantumCircuit(reconstruction.width, name="qhd")
    amplitudes = tuple(np.asarray(vector, dtype=float) for vector in reconstruction.initial_amplitudes)
    if options.encoding == "binary":
        bits = options.num_grid_points.bit_length() - 1
        append_binary_initial_state(circuit, grid, bits, amplitudes, recipe=options.initial_state_preparation)
        append_binary_steps(circuit, reconstruction, options, grid, held_bytes=native_held)
    else:
        append_initial_state(circuit, grid, amplitudes, recipe=options.initial_state_preparation)
        for compact in raw_blocks(reconstruction):
            append_pauli_evolution_block(circuit, compact)
    circuit.global_phase += reconstruction.physical_phase
    return circuit


def append_binary_steps(circuit, reconstruction, method, grid, *, held_bytes):
    """Append the stored binary blocks of every step, in order, to a circuit on the ``d*b`` register.

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
    from the stored exponent and choice (``binary.BinaryModel``) and must
    reproduce the recorded CX and rotation counts, or the construction
    raises. Planning priced the Plan's CX and rotation laws from those counts
    (``method.QHD._select_binary_native``), and a circuit with other counts
    would not be the circuit that those laws describe. The physical
    phase of the constant objective is not added here.

    The caller reserves the source Plan records, the completed QHD circuit
    graph and one block's construction scratch in held_bytes before circuit
    allocation, and the binary model admits that reservation together with
    its own arrays, metadata and construction workspace. The graph allowance
    covers stored definitions and parameters created by this builder, while
    subsequent SDK synthesis and process RSS have separate scope.
    """
    from .binary import BinaryModel

    model = BinaryModel(grid, method, reconstruction.support_values, held_bytes=held_bytes)
    bits = model.bits
    # CX and rotations of the emitted QFT pair, which the recorded counts of a
    # kinetic block include and model.construction, the phase diagonal alone,
    # does not. Qiskit 2.5.2 defines a controlled phase by two CX and three
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
    phases, which emits no gate for an all-zero table and otherwise keeps
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
