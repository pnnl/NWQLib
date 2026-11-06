"""Qiskit compatibility layer for controlled and inverted gates.

Every controlled-gate construction in the library calls :func:`controlled`,
so the next Qiskit signature change is a one-line fix here instead of a
per-site sweep. :func:`controlled` also makes the gates it controls exact
to rounding, because Qiskit 2.6.0 approximates dense unitaries in both of
its routes. It controls a composite gate by unrolling its definition,
which synthesizes the dense ``UnitaryGate`` blocks it meets there
inexactly (``subroutines/_dense_synthesis.py``), and it controls a
``UnitaryGate`` by a synthesis of the controlled matrix that it accepts
within a tolerance. :func:`exact_dense_unitaries` replaces the dense
unitaries of a whole circuit before a backend compiles it.
:func:`inverse_realized_gate` inverts a gate by reversing the circuit it
was built from, because Qiskit's own ``inverse`` of a controlled
``UnitaryGate`` synthesizes the gate again. A complete, unsimplified
``UCGate`` can keep its adjoint matrix table for execution as a
``multiplexer`` instruction.
Callers that will control or decompose the inverse select the reversed
forward definition with ``native_ucg=False``.
"""

from __future__ import annotations

from typing import Any

from qiskit.circuit import ControlledGate, Gate, Instruction, QuantumCircuit
from qiskit.circuit.library import UCGate, UnitaryGate

from nwqlib._limits import ResourceLimitError


def controlled(gate: Any, num_ctrl_qubits: int, *, ctrl_state: int | str | None = None,
               route: str = "gatewise") -> Any:
    """Return ``gate`` controlled on ``num_ctrl_qubits`` qubits, equal to the controlled matrix to rounding.

    A ``UnitaryGate`` becomes a ``ControlledGate`` named ``c_dense_unitary``
    whose definition is ``_dense_synthesis.controlled_unitary_circuit``,
    the exact synthesis of the whole controlled matrix. A base matrix that
    is not unitary to rounding is synthesized as its unitary polar factor
    (``controlled_unitary_circuit``). Qiskit 2.6.0's own
    ``UnitaryGate.control`` synthesizes the same matrix by quantum Shannon
    decomposition, keeps the result when ``numpy.allclose`` with
    ``atol=1e-7`` and ``rtol=1e-5`` accepts it, and otherwise falls back to
    an ``Isometry`` circuit, which can raise on valid input (RX(1e-7) with
    three controls). The exact synthesis takes at most
    ``(25/96) 4**m - 2**m + 4/3`` CX on ``m >= 3`` qubits in total
    (``_dense_synthesis`` module docstring). In tests with one to five
    controls on up to seven qubits it took as many CX as Qiskit or fewer,
    except for one system qubit with two controls, where Qiskit took 8 and
    the exact synthesis 10. Controlling each gate of the synthesized base
    unitary instead, the route below, would take about four times as many
    CX with one control. The dense LCU SELECT branches, the dense QPE powers
    and the coherent QPE powers use this whole-matrix synthesis. The base
    gate stays the ``UnitaryGate``, as in Qiskit, so ``inverse()`` of the
    result follows Qiskit's ``ControlledGate.inverse`` and synthesizes the
    adjoint again through Qiskit's inexact ``UnitaryGate.control``.
    :func:`inverse_realized_gate` reverses the exact definition instead.

    Any other gate takes the route that ``route`` selects for
    ``num_ctrl_qubits`` controls (``_dense_synthesis.select_dense_control_route``,
    ``"gatewise"`` by default). On the gate-wise route Qiskit unrolls the
    definition and controls each gate. When that definition contains a
    ``UnitaryGate`` on two or more qubits, the unrolling goes through
    Qiskit's own synthesis of the unitary, which is not exact to rounding
    (the ``_dense_synthesis`` module docstring gives the mechanism). This
    function first replaces each such unitary, at any depth, by a gate whose
    definition is ``_dense_synthesis.dense_unitary_circuit``. The
    constructions that call this function count the work of each synthesis
    against their limits before construction with
    ``_dense_synthesis.dense_synthesis_size``, one for each distinct unitary
    that :func:`dense_synthesis_widths` finds in the gate, and Qiskit's
    control of the synthesized gates with
    ``_dense_synthesis.gatewise_control_size``, for every occurrence that
    :func:`dense_control_counts` finds.

    On the whole-matrix route a gate whose definition is one dense
    ``UnitaryGate`` on all its qubits, with a global phase, becomes the
    ``c_dense_unitary`` gate of that phase times the matrix, as a
    ``UnitaryGate`` does. For another gate the controls are distributed
    over the instructions of its definition. Each instruction that is such a
    dense unitary is controlled through its whole controlled matrix, the
    consecutive runs of other instructions are controlled gate-wise, the
    global phase becomes a phase gate on the controls, and X gates around
    the result give an open ``ctrl_state``. A gate with no such instruction,
    or one whose definition holds a free parameter, takes the gate-wise
    route, which keeps the parameter bindable. A one-qubit ``UnitaryGate``
    inside a composite keeps Qiskit's exact definition and its control on
    both routes. :func:`dense_control_charges`
    lists the syntheses that either route makes, and it chooses the route by
    the same test (``_whole_matrix_definition``), so the counted syntheses
    are the ones this function makes. The LCHS ``dense_exact`` SELECT, the
    controlled QLS query of a dense dilation and the joint generator of
    ``build_control_diagonal_generator_encoding`` choose the route from
    their ``dense_control_route`` option, and a supplied circuit, whose
    matrix is not known and which may hold operations without one, always
    takes the gate-wise route. This function checks no limit itself.

    Qiskit 2.x deprecates the default of ``annotated``, and Qiskit 3.0
    changes it. ``annotated=False`` keeps the synthesized controlled gate on
    both series.
    """

    if isinstance(gate, UnitaryGate):
        return _controlled_unitary(gate, num_ctrl_qubits, ctrl_state)
    if route != "gatewise":
        from nwqlib.subroutines._dense_synthesis import select_dense_control_route

        if select_dense_control_route(route, num_ctrl_qubits) == "whole_matrix":
            whole = _controlled_whole_matrix(gate, num_ctrl_qubits, ctrl_state)
            if whole is not None:
                return whole
    gate = _exact_dense_definitions(gate, {})
    return gate.control(num_ctrl_qubits, ctrl_state=ctrl_state, annotated=False)


def _dense_matrix(operation: Any) -> Any:
    """Return the matrix of a dense unitary that the whole-matrix route controls, or None.

    That is a ``UnitaryGate`` on two or more qubits, or a gate whose
    definition is only such a gate on all its qubits in order, whose global
    phase multiplies the matrix. A parameterized phase gives None.
    """
    import numpy as np

    if isinstance(operation, UnitaryGate):
        return operation.to_matrix() if operation.num_qubits >= 2 else None
    if getattr(operation, "_standard_gate", None) is not None or operation.num_qubits < 2:
        return None
    definition = getattr(operation, "definition", None)
    if definition is None or len(definition.data) != 1:
        return None
    item = definition.data[0]
    inner = item.operation
    if (not isinstance(inner, UnitaryGate) or inner.num_qubits != operation.num_qubits or item.clbits
            or [definition.find_bit(qubit).index for qubit in item.qubits] != list(range(inner.num_qubits))):
        return None
    try:
        phase = float(definition.global_phase)
    except TypeError:
        return None
    return np.exp(1j * phase) * inner.to_matrix()


def _whole_matrix_definition(gate: Any) -> Any:
    """Return the definition over which the whole-matrix route distributes the controls of ``gate``, or None for the gate-wise route.

    For a gate that :func:`_dense_matrix` rejects, :func:`controlled` and
    :func:`dense_control_charges` both choose the route with this function
    before any synthesis starts. The distributed form needs a definition
    without classical bits or free parameters that holds at least one dense
    unitary that :func:`_dense_matrix` accepts. The whole-matrix route
    returns a gate without parameters, so a free parameter in the global
    phase or in any instruction, nested gates included, could no longer be
    bound. Such a gate takes the gate-wise route in both functions, and
    Qiskit's control of it keeps the parameters. Qiskit stores a global
    phase without free parameters as a ``float``, which becomes the angle of
    the phase gate on the controls.
    """
    definition = getattr(gate, "definition", None)
    if (definition is None or getattr(gate, "_standard_gate", None) is not None or definition.num_clbits
            or definition.num_parameters):
        return None
    if not any(_dense_matrix(item.operation) is not None for item in definition.data):
        return None
    return definition


def _controlled_whole_matrix(gate: Any, num_ctrl_qubits: int, ctrl_state: int | str | None) -> Any:
    """Return ``gate`` controlled on the whole-matrix route, or None when the gate takes the gate-wise route.

    The distributed form applies ``C(g_1) ... C(g_r)`` for the definition
    ``g_1 ... g_r``, which equals the controlled gate because a control
    distributes over a product. The phase of the definition is a phase
    gate on the controls, multi-controlled for two or more.
    """
    from qiskit.circuit.library import PhaseGate

    matrix = _dense_matrix(gate)
    if matrix is not None:
        return _controlled_unitary(UnitaryGate(matrix, label=gate.label, check_input=False), num_ctrl_qubits,
                                   ctrl_state)
    definition = _whole_matrix_definition(gate)
    if definition is None:
        return None
    k = num_ctrl_qubits
    state = (1 << k) - 1 if ctrl_state is None else int(ctrl_state, 2) if isinstance(ctrl_state, str) else ctrl_state
    if type(state) is not int or not 0 <= state < 1 << k:
        raise ValueError(f"control state {ctrl_state!r} does not fit {k} controls")
    controls = list(range(k))
    open_controls = [qubit for qubit in controls if not state >> qubit & 1]
    circuit = QuantumCircuit(k + gate.num_qubits, name="c_" + gate.name)
    if open_controls:
        circuit.x(open_controls)
    run = QuantumCircuit(gate.num_qubits)

    def flush():
        nonlocal run
        if run.data:
            circuit.append(controlled(run.to_gate(label=gate.name), k), [*controls, *range(k, k + gate.num_qubits)])
            run = QuantumCircuit(gate.num_qubits)

    for item in definition.data:
        qubits = [definition.find_bit(qubit).index for qubit in item.qubits]
        dense = _dense_matrix(item.operation)
        if dense is None:
            run.append(item.operation, qubits)
            continue
        flush()
        inner = UnitaryGate(dense, label=item.operation.label, check_input=False)
        circuit.append(_controlled_unitary(inner, k, None), [*controls, *(k + qubit for qubit in qubits)])
    flush()
    phase = float(definition.global_phase)
    if phase:
        circuit.append(PhaseGate(phase) if k == 1 else PhaseGate(phase).control(k - 1), controls)
    if open_controls:
        circuit.x(open_controls)
    result = Gate("c_" + gate.name, k + gate.num_qubits, [], label=gate.label)
    result.definition = circuit
    return result


def dense_control_charges(operation: Any, num_controls: int, route: str) -> tuple[tuple[int, ...], tuple[int, ...],
                                                                                  tuple[int, int]]:
    """Return the syntheses that :func:`controlled` makes when it adds ``num_controls`` controls to ``operation`` on ``route``.

    The result is ``(widths, controlled_widths, counts)``. ``widths`` are the
    qubit counts of the syntheses of the gate-wise route
    (:func:`dense_synthesis_widths`), ``controlled_widths`` those, controls
    included, of the controlled matrices of the whole-matrix route, and
    ``counts`` the ``(gates, instructions)`` of Qiskit's control of
    the synthesized gates (:func:`dense_control_counts`), zero when no gate
    is controlled that way. ``_dense_synthesis.admit_dense_syntheses`` takes
    the three. Reading a definition can make Qiskit build it.
    """
    if isinstance(operation, UnitaryGate):
        return (), (operation.num_qubits + num_controls,), (0, 0)
    if route != "gatewise":
        from nwqlib.subroutines._dense_synthesis import select_dense_control_route

        if select_dense_control_route(route, num_controls) == "whole_matrix":
            if _dense_matrix(operation) is not None:
                return (), (operation.num_qubits + num_controls,), (0, 0)
            definition = _whole_matrix_definition(operation)
            if definition is not None:
                widths, controlled_widths, totals = [], [], [0, 0]
                for item in definition.data:
                    if _dense_matrix(item.operation) is not None:
                        controlled_widths.append(item.operation.num_qubits + num_controls)
                        continue
                    item_widths, counts, _ = dense_gatewise_charges(item.operation, num_controls)
                    widths += item_widths
                    totals = [total + count for total, count in zip(totals, counts, strict=True)]
                return tuple(widths), tuple(controlled_widths), tuple(totals)
    widths, counts, _ = dense_gatewise_charges(operation, num_controls)
    return widths, (), counts


def _controlled_unitary(gate: UnitaryGate, num_ctrl_qubits: int, ctrl_state: int | str | None) -> ControlledGate:
    """Return ``gate`` controlled through the exact synthesis of its closed-control matrix.

    The gate is built first, so Qiskit rejects an invalid control count or
    control state before the synthesis starts. Its definition is assigned
    afterwards because the ``ControlledGate`` constructor deep-copies a
    definition passed to it. As in Qiskit's ``UnitaryGate.control``, the
    definition has closed controls, and the ``definition`` property adds the
    X gates of an open ``ctrl_state``.
    """
    from nwqlib.subroutines._dense_synthesis import controlled_unitary_circuit

    matrix = gate.to_matrix()
    result = ControlledGate("c_dense_unitary", gate.num_qubits + num_ctrl_qubits, [matrix],
                            num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state, base_gate=gate)
    result.definition = controlled_unitary_circuit(matrix, num_ctrl_qubits)
    return result


def exact_dense_unitaries(circuit: Any, *, cache: dict[Any, QuantumCircuit] | None = None) -> Any:
    """Return ``circuit`` with every dense ``UnitaryGate``, at any depth, synthesized exactly.

    Aer applies a ``UnitaryGate`` as its matrix, but a backend that compiles
    a circuit to a gate basis would synthesize it with Qiskit's own inexact
    synthesis (the ``_dense_synthesis`` module docstring gives the
    mechanism). The NWQ-Sim and Nexus preparations, IonQ with its QIS
    gateset and IBM at optimization levels 0 and 1 therefore call this
    function before they compile, and so does Aer compilation when a noise
    model's gate basis omits the unitary instruction. The circuit comes back
    unchanged when it contains no ``UnitaryGate`` on two or more qubits.
    Control-flow blocks and annotated operations are not rewritten. The
    syntheses this call makes are those of :func:`dense_synthesis_widths`,
    or with ``cache`` the matrices of ``dense_unitary_census(circuit,
    cached=True)`` that the cache lacks, so a caller can size them with
    ``_dense_synthesis.dense_synthesis_size`` before calling.

    ``cache`` maps the content of a dense matrix (:func:`dense_matrix_key`)
    to its synthesized circuit. A matrix found there is not synthesized
    again, and each gate receives a copy of the stored circuit, so a caller
    that edits a definition leaves the cache unchanged. The call
    synthesizes once each distinct matrix that the cache lacks, and stores
    its circuit. ``dense_unitary_circuit`` is deterministic, so a stored
    circuit equals a new synthesis of the same matrix. The backends reach
    this function through ``Run._exact_dense_unitaries``, which checks the
    memory of the syntheses, passes the Run's synthesis cache during a
    preparation of the Run and counts their work.
    """
    rewritten = _rewrite_circuit(circuit, {}, cache)
    return circuit if rewritten is None else rewritten


def dense_matrix_key(matrix: Any) -> tuple[tuple[int, ...], str, bytes]:
    """Return the key of a dense matrix in a synthesis cache: its shape, dtype and bytes.

    The key compares matrix content, so equal matrices in different gate
    objects share one entry. Each preparation lowers the Plan to new gate
    objects, so a dense unitary that it repeats reaches the cache as a new
    object with the same content.
    """
    return matrix.shape, matrix.dtype.str, matrix.tobytes()


def dense_synthesis_widths(operation: Any) -> tuple[int, ...]:
    """Return the qubit count of each dense unitary that the exact rewrite of ``operation`` would synthesize.

    ``operation`` is a circuit or a single operation. The walk follows the
    rules of :func:`exact_dense_unitaries` and of the composite branch of
    :func:`controlled` without synthesizing anything. A ``UnitaryGate`` on
    two or more qubits counts once per distinct matrix and label, which is
    how the rewrite keys it. Standard gates, one-qubit unitaries and
    control-flow blocks count nothing. A ``UnitaryGate`` that
    :func:`controlled` receives directly is outside this count, because it
    goes through ``_dense_synthesis.controlled_unitary_circuit``, whose size
    formula is ``controlled_synthesis_size``. Reading a definition can make
    Qiskit build it, as the rewrite itself would.
    """
    return tuple(_dense_unitary_widths(operation, cached=False).values())


def dense_unitary_census(operation: Any, *, cached: bool = False,
                         built: bool = False) -> tuple[dict[Any, int], dict[int, int], Any]:
    """Return the two populations of dense unitaries in ``operation`` from one walk.

    The result is ``(widths, occurrences, unbuilt)``. ``widths`` maps the key
    of each distinct dense ``UnitaryGate`` on two or more qubits to its qubit
    count, the population that the exact rewrite synthesizes once each
    (:func:`dense_synthesis_widths`). ``occurrences`` counts those gates by
    width with their multiplicity through nested definitions, the population
    whose synthesized gates Qiskit's control unrolls once for every place
    where they occur (:func:`dense_control_counts`). The two differ whenever a
    matrix occurs more than once, so they are kept apart. Without ``cached``
    a unitary is keyed by its matrix bytes and label, as one rewrite keys it,
    and with ``cached`` by :func:`dense_matrix_key`, as a synthesis cache
    keys it. Each gate object is keyed once. Standard gates, one-qubit
    unitaries and control-flow blocks count nothing.

    Reading ``definition`` can make Qiskit build it. With ``built`` the walk
    reads only definitions that are already built (``_definition``), as
    circuit intake does (``blocks._qiskit_intake``), so planning builds
    nothing. An operation whose class builds its definition lazily
    (overrides ``_define``) and has not built it, or that is not an
    ``Instruction`` but has a definition, is skipped, and the walk goes on
    with the rest. ``unbuilt`` is the first such operation, and the two
    populations then hold the known part: the dense unitaries outside the
    skipped definitions, which the construction synthesizes and controls as
    well. Otherwise ``unbuilt`` is None.
    """
    from collections import Counter

    widths: dict[Any, int] = {}
    memo: dict[int, Counter] = {}
    unbuilt: list[Any] = []
    # Holding each visited object keeps its id from being reused during the walk.
    held: list[Any] = []

    def definition(item: Any) -> Any:
        if not built:
            return getattr(item, "definition", None)
        if isinstance(item, Instruction):
            if item._definition is None and type(item)._define is not Instruction._define:
                unbuilt.append(item)
            return item._definition
        if getattr(type(item), "definition", None) is not None:
            unbuilt.append(item)
        return None

    def occurrences(item: Any) -> Counter:
        key = id(item)
        if key in memo:
            return memo[key]
        held.append(item)
        if isinstance(item, UnitaryGate):
            memo[key] = Counter()
            if item.num_qubits >= 2:
                matrix = item.to_matrix()
                widths.setdefault(dense_matrix_key(matrix) if cached else
                                  ("dense_unitary", matrix.tobytes(), item.label), item.num_qubits)
                memo[key] = Counter({item.num_qubits: 1})
            return memo[key]
        memo[key] = Counter()
        if getattr(item, "_standard_gate", None) is None:
            inner = definition(item)
            memo[key] = Counter() if inner is None else walk(inner)
        return memo[key]

    def walk(circuit: Any) -> Counter:
        total: Counter = Counter()
        for instruction in circuit.data:
            if not (instruction.is_standard_gate() or instruction.is_control_flow()):
                total += occurrences(instruction.operation)
        return total

    total = walk(operation) if isinstance(operation, QuantumCircuit) else occurrences(operation)
    return widths, dict(total), unbuilt[0] if unbuilt else None


def _dense_unitary_widths(operation: Any, *, cached: bool) -> dict[Any, int]:
    """Map the key of each distinct dense unitary in ``operation`` to its qubit count (:func:`dense_unitary_census`)."""
    return dense_unitary_census(operation, cached=cached)[0]


def dense_gatewise_charges(operation: Any, num_controls: int, *,
                           built: bool = False) -> tuple[tuple[int, ...], tuple[int, int], Any]:
    """Return ``(widths, counts, unbuilt)`` of the gate-wise control of ``operation`` with ``num_controls`` controls, from one walk.

    ``widths`` are those of :func:`dense_synthesis_widths` and ``counts``
    the pair of :func:`dense_control_counts`, the two populations of
    :func:`dense_unitary_census`, which also defines ``built`` and
    ``unbuilt``. ``_dense_synthesis.admit_dense_syntheses`` takes
    ``widths`` and ``counts``.
    """
    widths, occurrences, unbuilt = dense_unitary_census(operation, built=built)
    counts = (0, 0) if isinstance(operation, UnitaryGate) else gatewise_control_totals(occurrences, num_controls)
    return tuple(widths.values()), counts, unbuilt


def unbuilt_definition(operation: Any) -> str:
    """Describe ``operation``, the ``unbuilt`` of :func:`dense_unitary_census`, in the reason that a planning count is unknown."""
    return f"supplied operation {operation.name} ({operation.num_qubits} qubits): definition not built at planning"


def refuse_unbuilt_limit(operation: Any, limit: str) -> None:
    """Refuse the explicit work ``limit``, written as ``Method.field=value``, that the controlled synthesis of ``operation`` leaves uncounted at planning.

    ``operation`` is the ``unbuilt`` of :func:`dense_unitary_census`. Planning
    cannot count its synthesis without building its definition, so the
    requested limit cannot be checked, and the Plan is refused before any
    later work.
    """
    raise ResourceLimitError(f"controlled synthesis of {unbuilt_definition(operation)}, so it cannot be counted against "
                             f"{limit}. Expand the supplied circuit before it is ingested, for example with decompose(), so "
                             "that its definitions are built, or set that limit to None")


def gatewise_control_totals(occurrences: dict[int, int], num_controls: int) -> tuple[int, int]:
    """Return (gates, instructions) of Qiskit's control of the synthesized dense unitaries counted in ``occurrences``.

    Each occurrence of a width adds the pair of
    ``_dense_synthesis.gatewise_control_counts`` for that width.
    """
    from nwqlib.subroutines._dense_synthesis import gatewise_control_counts

    totals = [0, 0]
    for width, count in occurrences.items():
        for index, value in enumerate(gatewise_control_counts(width, num_controls)):
            totals[index] += count * value
    return tuple(totals)


def dense_control_counts(operation: Any, num_controls: int) -> tuple[int, int]:
    """Return (gates, instructions) of Qiskit's control of the exact syntheses in ``operation``.

    ``operation`` is a gate that :func:`controlled` receives with
    ``num_controls`` controls, or a circuit that becomes one. Qiskit unrolls
    the definition of the gate to its basis and controls each gate
    (``qiskit/circuit/_add_control.py``), so a dense ``UnitaryGate`` on two
    or more qubits is unrolled, after its exact synthesis, once for every
    place where it occurs. The count therefore uses the occurrences of
    :func:`dense_unitary_census`, where :func:`dense_synthesis_widths`
    counts each distinct synthesis once (:func:`gatewise_control_totals`).
    The other gates of the definition are controlled too and lie outside
    this count, and so does a gate that an earlier control already
    unrolled. A ``UnitaryGate`` that :func:`controlled` receives directly
    goes through ``_dense_synthesis.controlled_unitary_circuit`` and counts
    nothing here. Reading a definition can make Qiskit build it.
    """
    return dense_gatewise_charges(operation, num_controls)[1]




def _exact_dense_definitions(operation: Any, visited: dict[Any, tuple[Any, Any]],
                             cache: dict[Any, QuantumCircuit] | None = None) -> Any:
    """Return ``operation`` with every dense ``UnitaryGate`` in its definition tree synthesized exactly.

    An operation whose tree has no ``UnitaryGate`` on two or more qubits is
    returned unchanged, so standard gates and cached definitions stay as
    Qiskit built them. A one-qubit ``UnitaryGate`` is kept too, because
    Qiskit defines it by exact ZYZ angles. A dense ``UnitaryGate`` becomes a
    gate named ``dense_unitary`` whose definition is the exact synthesis. A
    changed composite becomes a plain gate, or instruction, with the same
    name, label and width and the rewritten definition. ``visited`` maps
    the ``id`` of a visited composite, or the bytes and label of a dense
    matrix, to the pair (original, result). Qiskit copies instructions into
    each definition, so keying dense matrices by their bytes synthesizes a
    repeated unitary once per call. Storing the original keeps each ``id``
    valid while the map exists. ``cache`` is the synthesis cache of
    :func:`exact_dense_unitaries`, or None.
    """
    if isinstance(operation, UnitaryGate):
        if operation.num_qubits < 2:
            return operation
        matrix = operation.to_matrix()
        key = ("dense_unitary", matrix.tobytes(), operation.label)
        if key not in visited:
            from nwqlib.subroutines._dense_synthesis import dense_unitary_circuit

            exact = Gate("dense_unitary", operation.num_qubits, [], label=operation.label)
            if cache is None:
                exact.definition = dense_unitary_circuit(matrix)
            else:
                stored = dense_matrix_key(matrix)
                if stored not in cache:
                    cache[stored] = dense_unitary_circuit(matrix)
                exact.definition = cache[stored].copy()
            visited[key] = (operation, exact)
        return visited[key][1]
    if getattr(operation, "_standard_gate", None) is not None:
        return operation
    key = id(operation)
    if key in visited:
        return visited[key][1]
    definition = getattr(operation, "definition", None)
    exact = None if definition is None else _rewrite_circuit(definition, visited, cache)
    result = operation
    if exact is not None:
        # A plain gate keeps its signature. A library subclass's own
        # parameters describe its original definition, so the rewritten
        # gate exposes the parameters left in the new definition.
        params = (list(operation.params) if operation.base_class in (Gate, Instruction)
                  else list(exact.parameters))
        if isinstance(operation, Gate):
            result = Gate(operation.name, operation.num_qubits, params, label=operation.label)
        else:
            result = Instruction(operation.name, operation.num_qubits, operation.num_clbits, params,
                                 label=operation.label)
        result.definition = exact
    visited[key] = (operation, result)
    return result


def _rewrite_circuit(circuit: Any, visited: dict[Any, tuple[Any, Any]],
                     cache: dict[Any, QuantumCircuit] | None = None) -> Any:
    """Return a copy of ``circuit`` with its dense unitaries synthesized exactly, or None if none changed.

    Standard gates and control-flow operations are kept without building
    their Python objects or entering their blocks.
    """
    items = list(circuit.data)
    replaced = []
    for item in items:
        new = None
        if not (item.is_standard_gate() or item.is_control_flow()):
            old = item.operation
            new = _exact_dense_definitions(old, visited, cache)
            new = None if new is old else new
        replaced.append(new)
    if all(new is None for new in replaced):
        return None
    exact = circuit.copy_empty_like()
    for new, item in zip(replaced, items):
        exact._append(item if new is None else item.replace(operation=new))
    return exact


def inverse_realized_gate(gate: Gate, *, native_ucg: bool = True) -> Gate:
    """Return the adjoint of ``gate`` from its UCG table or from its reversed definition.

    Reversing a definition reuses the UCG syntheses that Qiskit has already
    built for it, also in copies made from that built definition. A copy
    made before Qiskit built the definition synthesizes on its own. Keeping
    the adjoint table of a complete UCG avoids synthesis only while later
    steps leave that UCG as a ``multiplexer`` instruction.

    With ``native_ucg=True``, a complete, unsimplified ``UCGate`` becomes
    the UCG of its adjoint table, or a ``UnitaryGate`` for zero controls.
    The address order, target and control order, padding and label are
    preserved. Orthogonality of the control projectors gives
    ``M(U)^dagger = M(U^dagger)`` for arbitrary stored matrices,
    where ``M(U) = sum_c |c><c| (x) U_c``.
    For finite binary64 entries, conjugate transposition is exact, and the
    represented forward and adjoint pair has defect
    ``max_c ||U_c^dagger U_c - I||_2``.
    This is an inverse identity only for unitary table blocks. Aer executes
    it as its ``multiplexer`` instruction only if later transformations
    leave the complete table intact and the target accepts that instruction.
    Aer 0.18.0 applies each control address's block of the supplied table
    directly (``QubitVector::apply_multiplexer``,
    https://github.com/Qiskit/qiskit-aer/blob/0.18.0/src/simulators/statevector/qubitvector.hpp#L1305-L1340),
    so the table reaches Aer as its ``multiplexer`` instruction. A backend
    that decomposes the gate needs a check of the circuit it builds. This
    costs one two-by-two conjugate transpose and copy per table entry,
    plus the constructor's existing validation, and forms no dense
    operator. Qiskit's ``UCGate`` accepts only one target qubit, so the
    adjoint table allocates ``64 * 2**k`` new array bytes for ``k``
    controls.

    Set ``native_ucg=False`` before building an inverse that will be
    controlled or decomposed. Its UCG leaves then use the reversed forward
    definition. An existing ``ControlledGate`` always selects that route
    for its definition and base metadata. Reversing selected factors avoids
    an independent synthesis of the adjoint table. It does not certify
    synthesis error, phase-rounding error or backend execution error.

    A ``UnitaryGate`` uses its stored matrix's conjugate transpose without
    another unitarity check. Standard gates use their own Qiskit inverses.
    Other definitions are reversed recursively, with their global phases
    negated. Open controlled gates use the closed definition and restore
    the control state once. Reversing a controlled definition avoids
    Qiskit's ``base_gate.inverse().control(...)`` resynthesis. A gate with
    no definition falls back to its own inverse. Shared operations are
    inverted once per invocation and effective ``native_ucg`` mode. The
    cache holds the original operations until the traversal finishes.

    Args:
        gate: Gate whose table or built definition is adjointed.
        native_ucg: Preserve eligible UCG tables for execution as a
            ``multiplexer`` instruction. Pass
            False when the caller knows the result will be controlled or
            decomposed. This option does not inspect a backend target.
    """
    memo = {}

    def inverse(operation, *, native_ucg):
        if isinstance(operation, ControlledGate):
            native_ucg = False
        key = (id(operation), native_ucg)
        if key not in memo:
            memo[key] = (operation, _construct_inverse(operation, native_ucg=native_ucg, invert=inverse))
        return memo[key][1]

    return inverse(gate, native_ucg=native_ucg)


def _construct_inverse(gate, *, native_ucg, invert):
    """Build one gate's adjoint, using ``invert`` to reuse adjoints of shared children."""
    if isinstance(gate, UnitaryGate):
        # The admitted matrix adjoint needs no repeated unitarity checks or synthesis.
        return UnitaryGate(gate.to_matrix().conj().T, check_input=False, label=gate.label)
    if (
        native_ucg
        and isinstance(gate, UCGate)
        and not gate.up_to_diagonal
        and gate.simp_contr[0] is False
    ):
        # The adjoint of a complete multiplexer is the multiplexer of the adjoint table.
        adjoint_table = [matrix.conj().T.copy() for matrix in gate.params]
        if gate.num_qubits == 1:
            return UnitaryGate(adjoint_table[0], check_input=False, label=gate.label)
        inverse = UCGate(adjoint_table, up_to_diagonal=False, mux_simp=False)
        inverse.label = gate.label
        return inverse
    if gate.base_class.inverse.__module__.startswith("qiskit.circuit.library.standard_gates."):
        # Native inverses preserve the gate type and parameter arity required by later control and basis translation.
        return gate.inverse()
    closed_gate = gate
    if isinstance(gate, ControlledGate) and gate.ctrl_state != (1 << gate.num_ctrl_qubits) - 1:
        # Open-control definitions include X conjugations, so invert the closed definition before restoring the control state.
        closed_gate = gate.to_mutable()
        closed_gate.ctrl_state = (1 << gate.num_ctrl_qubits) - 1
    definition = closed_gate.definition
    if definition is None:
        return gate.inverse()
    inverse_definition = definition.copy_empty_like()
    inverse_definition.global_phase = -definition.global_phase
    for instruction in reversed(definition.data):
        inverse_definition.append(
            invert(instruction.operation, native_ucg=native_ucg),
            instruction.qubits,
            instruction.clbits,
        )
    if isinstance(gate, ControlledGate):
        base_inverse = invert(gate.base_gate, native_ucg=False)
        # A generic inverse must not reuse a native name interpreted by the transpiler.
        return ControlledGate(
            f"{closed_gate.name}_dg", gate.num_qubits, base_inverse.params,
            label=gate.label, num_ctrl_qubits=gate.num_ctrl_qubits,
            definition=inverse_definition, ctrl_state=gate.ctrl_state, base_gate=base_inverse,
        )
    # A plain gate keeps its parameters. A library subclass's own parameters describe its
    # original definition, so the inverse exposes the free parameters left in the reversed
    # definition.
    params = list(gate.params) if gate.base_class is Gate else list(inverse_definition.parameters)
    inverse = Gate(f"{gate.name}_dg", gate.num_qubits, params, label=gate.label)
    inverse.definition = inverse_definition
    return inverse
