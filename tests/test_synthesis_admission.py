"""Admission of Qiskit's control of the exact dense synthesis and of the syntheses that a Run repeats.

``qiskit_compat.controlled`` replaces each dense unitary of a composite gate
by its exact synthesis and then lets Qiskit control every synthesized gate,
which unrolls the synthesis at every place where it occurs.
The expected work below is written out independently of the library's
laws. The gate census bounds of the synthesis are restated, and the
instructions that Qiskit emits for each gate kind are counted from the
installed Qiskit, not read from the stored tables.
"""

from functools import lru_cache

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary

from nwqlib.subroutines.qiskit_compat import controlled, dense_control_counts, dense_synthesis_widths


def _census(num_qubits):
    """Gate bounds of one exact synthesis on m qubits: U, CX, RZ and H."""
    m = num_qubits
    if m == 1:
        return {"u": 1, "cx": 0, "rz": 0, "h": 0}
    return {"u": 7 * 4 ** (m - 2), "cx": (25 * 4**m - 72 * 2**m + 32) // 48,
            "rz": (3 * 4**m - 12 * 2**m) // 8, "h": (4**m - 16) // 24}


@lru_cache(maxsize=None)
def _emitted_instructions(kind, controls):
    """Instructions that the installed Qiskit emits when it controls one gate of ``kind``."""
    body = QuantumCircuit(2)
    if kind == "u":
        body.u(0.3, 0.7, 1.1, 0)
    elif kind == "cx":
        body.cx(0, 1)
    elif kind == "rz":
        body.rz(0.3, 0)
    elif kind == "h":
        body.h(0)
    else:
        body.global_phase = 0.4
    gate = Gate("g", 2, [])
    gate.definition = body
    return tuple(gate.control(controls, annotated=False)._definition.data)


def _emitted(kind, controls):
    """Number of instructions that Qiskit emits for one gate of ``kind`` with ``controls`` controls."""
    return len(_emitted_instructions(kind, controls))


def _heavy(kind, controls):
    """Emitted instructions that hold angles or are Python objects rather than parameterless standard gates."""
    return sum(1 for item in _emitted_instructions(kind, controls)
               if not item.is_standard_gate() or item.operation.params)


def _controlled_census(num_qubits):
    """Gate bounds of one whole-matrix synthesis on m qubits, controls included.

    The controlled matrix is demultiplexed at the top into one multiplexed
    RZ of 2**(m-1) rotations and 2**(m-1) CX and two (m-1)-qubit blocks.
    """
    m = num_qubits
    if m <= 2:
        return _census(m)
    block, half = _census(m - 1), 2 ** (m - 1)
    return {"u": 2 * block["u"], "cx": 2 * block["cx"] + half, "rz": 2 * block["rz"] + half, "h": 2 * block["h"]}


def _controlled_synthesis_work(num_qubits):
    """Work law of one whole-matrix synthesis on m qubits, 11 M**3 + (m**2 + 5 m + 256) M**2."""
    size = 1 << num_qubits
    return 11 * size**3 + (num_qubits**2 + 5 * num_qubits + 256) * size**2


def _census_control(census, controls):
    """(gates, instructions, heavy) of Qiskit's control of a circuit with gate census ``census`` and a global phase."""
    return (sum(census.values()) + 1,
            sum(count * _emitted(kind, controls) for kind, count in census.items()) + _emitted("phase", controls),
            sum(count * _heavy(kind, controls) for kind, count in census.items()) + _heavy("phase", controls))


def _gatewise_control(num_qubits, controls):
    """(gates, instructions, heavy) of Qiskit's control of one synthesis, with its global phase."""
    return _census_control(_census(num_qubits), controls)


def _gatewise_control_work(num_qubits, controls, occurrences=1, census=None):
    """Work of Qiskit's control of ``occurrences`` syntheses in one call, 2048 gates + 16 instructions."""
    gates, instructions, _ = _census_control(_census(num_qubits) if census is None else census, controls)
    return occurrences * (2048 * gates + 16 * instructions)


def _counted_dense_controls(monkeypatch):
    """Record every ``qiskit_compat.controlled`` call on a composite that holds a dense unitary.

    Each entry is the control count and the sorted widths of the distinct
    dense unitaries that the call synthesizes. Callers import ``controlled``
    when they run, so the patched function sees every call made after this.
    """
    from nwqlib.subroutines import qiskit_compat

    calls = []
    original = qiskit_compat.controlled

    def counted(gate, num_ctrl_qubits, **options):
        if not isinstance(gate, UnitaryGate):
            widths = qiskit_compat.dense_synthesis_widths(gate)
            if widths:
                calls.append((num_ctrl_qubits, tuple(sorted(widths))))
        return original(gate, num_ctrl_qubits, **options)

    monkeypatch.setattr(qiskit_compat, "controlled", counted)
    return calls


@lru_cache(maxsize=None)
def _controlled_twice(kind):
    """(gates, instructions) of controlling once more a gate of ``kind`` that Qiskit controlled once.

    The gates are those that Qiskit unrolls from the once-controlled gate,
    and the instructions those that the second control emits.
    """
    from qiskit.circuit._add_control import EFFICIENTLY_CONTROLLED_GATES, _unroll_gate

    body = QuantumCircuit(2)
    if kind == "u":
        body.u(0.3, 0.7, 1.1, 0)
    elif kind == "cx":
        body.cx(0, 1)
    elif kind == "ry":
        body.ry(0.3, 0)
    elif kind == "h":
        body.h(0)
    else:
        body.global_phase = 0.4
    gate = Gate("g", 2, [])
    gate.definition = body
    once = gate.control(1, annotated=False)
    unrolled = _unroll_gate(once, EFFICIENTLY_CONTROLLED_GATES).definition
    return sum(unrolled.count_ops().values()), len(once.control(1, annotated=False)._definition.data)


def _twice_controlled_work(num_qubits):
    """Work of the second one-control step over one synthesis that a first one-control step unrolled.

    The law prices an RZ at the costliest rotation, which one control turns
    into a CU, so the RZ entries use a controlled RY here. The global phase
    and the two X gates around an open first control are one gate and one
    instruction each.
    """
    census = _census(num_qubits)
    pairs = {kind: _controlled_twice("ry" if kind == "rz" else kind) for kind in census}
    gates = 3 + sum(census[kind] * pairs[kind][0] for kind in census)
    instructions = 3 + sum(census[kind] * pairs[kind][1] for kind in census)
    return 2048 * gates + 16 * instructions


def test_twice_controlled_dense_counts_bound_the_second_control():
    # A two-child QSP generator controls each dense child on its combine
    # qubit, and the parity control of a pass then controls that output
    # again. The second step unrolls the first one's gates and controls each
    # of them, and the law bounds both counts for a Haar-random child.
    from qiskit.circuit._add_control import EFFICIENTLY_CONTROLLED_GATES, _unroll_gate
    from nwqlib.algorithms.lchs.compiled_selection import _twice_controlled_dense_counts
    from nwqlib.subroutines._dense_synthesis import gatewise_control_size

    for m in (2, 3):
        child = QuantumCircuit(m)
        child.append(UnitaryGate(random_unitary(1 << m, seed=44 + m).data), child.qubits)
        child.global_phase = 0.2
        once = controlled(child.to_gate(), 1, ctrl_state=0)
        unrolled = sum(_unroll_gate(once, EFFICIENTLY_CONTROLLED_GATES).definition.count_ops().values())
        twice = len(once.control(1, annotated=False)._definition.data)
        gates, instructions, heavy = counts = _twice_controlled_dense_counts(m)
        assert unrolled <= gates and twice <= instructions and heavy <= instructions
        assert gatewise_control_size(*counts)[0] == _twice_controlled_work(m)


def test_gatewise_control_tables_match_installed_qiskit():
    # Every entry of the stored tables, and the fixed counts beside them, is
    # recomputed with the installed Qiskit. A failure means that Qiskit
    # controls one of these gate kinds differently, and the table and
    # docs/dependency_issues.md need revisiting.
    from nwqlib.subroutines._dense_synthesis import CONTROLLED_RZ_INSTRUCTIONS, CONTROLLED_U_INSTRUCTIONS

    assert len(CONTROLLED_U_INSTRUCTIONS) == len(CONTROLLED_RZ_INSTRUCTIONS) == 64
    for k in range(1, 65):
        assert _emitted("u", k) == CONTROLLED_U_INSTRUCTIONS[k - 1], k
        assert _emitted("rz", k) == CONTROLLED_RZ_INSTRUCTIONS[k - 1], k
        assert (_emitted("cx", k), _emitted("h", k), _emitted("phase", k)) == (1, 7, 1), k


def test_gatewise_heavy_counts_match_installed_qiskit():
    # The kept-byte law allows more memory for an emitted instruction that
    # holds angles or is a Python object. Their count per gate kind is
    # recomputed with the installed Qiskit for every control count.
    from nwqlib.subroutines._dense_synthesis import _controlled_heavy_instructions

    for k in range(1, 65):
        for kind in ("u", "rz", "cx", "h", "phase"):
            assert _controlled_heavy_instructions(kind, k) == _heavy(kind, k), (kind, k)


@pytest.mark.parametrize("num_qubits,controls", [(2, 1), (2, 3), (3, 1), (3, 2), (4, 1), (4, 5)])
def test_gatewise_control_law_bounds_the_built_control_step(num_qubits, controls):
    # Qiskit's control of a composite that holds one Haar-random unitary
    # emits, for each gate of the exact synthesis, the instructions of its
    # kind, and one for the global phase. With one control every kind emits
    # a fixed count, so the relation is exact. With more, a U gate with
    # special angles emits fewer than a general one. The law also replaces
    # the actual census by its bound, so it can only count more.
    from nwqlib.subroutines._dense_synthesis import gatewise_control_counts

    composite = QuantumCircuit(num_qubits)
    composite.append(UnitaryGate(random_unitary(1 << num_qubits, seed=40 + num_qubits).data), composite.qubits)
    composite.global_phase = 0.25
    built = controlled(composite.to_gate(), controls)
    synthesized = next(iter(built.base_gate.definition.data)).operation.definition
    actual = {kind: synthesized.count_ops().get(kind, 0) for kind in ("u", "cx", "rz", "h")}
    emitted = len(built._definition.data)
    expected = sum(count * _emitted(kind, controls) for kind, count in actual.items()) + 1
    assert emitted == expected if controls == 1 else emitted <= expected
    gates, instructions, _ = counts = gatewise_control_counts(num_qubits, controls)
    assert counts == _gatewise_control(num_qubits, controls)
    assert emitted <= instructions and sum(actual.values()) < gates


def test_dense_control_counts_follow_every_occurrence():
    # Qiskit unrolls a dense unitary once for every place it occurs, while
    # the exact rewrite synthesizes each distinct matrix once. Here the
    # three-qubit unitary occurs three times, once inside a nested
    # composite, the two-qubit unitary once and a one-qubit unitary, which
    # Qiskit defines exactly, counts nothing. A UnitaryGate controlled
    # directly takes the whole-matrix synthesis and counts nothing either.
    first = UnitaryGate(random_unitary(8, seed=41).data)
    inner = QuantumCircuit(3)
    inner.append(first, [0, 1, 2])
    inner.append(UnitaryGate(random_unitary(4, seed=42).data), [0, 1])
    circuit = QuantumCircuit(3)
    circuit.append(first, [0, 1, 2])
    circuit.append(first, [2, 1, 0])
    circuit.append(inner.to_gate(), [0, 1, 2])
    circuit.append(UnitaryGate(random_unitary(2, seed=43).data), [1])
    assert sorted(dense_synthesis_widths(circuit)) == [2, 3]
    for k in (1, 2):
        three, two = _gatewise_control(3, k), _gatewise_control(2, k)
        assert dense_control_counts(circuit.to_gate(), k) == tuple(3 * a + b for a, b in zip(three, two))
        assert dense_control_counts(circuit, k) == dense_control_counts(circuit.to_gate(), k)
    assert dense_control_counts(first, 2) == (0, 0, 0)


def test_synthesis_cache_reuses_the_circuit_that_a_new_synthesis_makes(monkeypatch):
    # The cache keys a matrix by its content, so an equal matrix in a new
    # gate object with another label is neither synthesized nor charged
    # again. The transposed two-qubit matrix is new and is charged. The
    # synthesis is deterministic, so the reused circuit equals a new
    # synthesis of the matrix. Each gate receives its own copy.
    from nwqlib.subroutines._dense_synthesis import dense_unitary_circuit
    from nwqlib.subroutines.qiskit_compat import dense_matrix_key, exact_dense_unitaries
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work

    three, two = random_unitary(8, seed=48).data, random_unitary(4, seed=49).data
    first = QuantumCircuit(3)
    first.append(UnitaryGate(three), [0, 1, 2])
    first.append(UnitaryGate(two), [0, 1])
    second = QuantumCircuit(3)
    second.append(UnitaryGate(three.copy(), label="again"), [0, 1, 2])
    second.append(UnitaryGate(two.T.copy()), [1, 2])
    cache, charges = {}, []
    calls = _counted_syntheses(monkeypatch)
    exact_first = exact_dense_unitaries(first, charge=lambda work, operation: charges.append(work), cache=cache)
    exact_second = exact_dense_unitaries(second, charge=lambda work, operation: charges.append(work), cache=cache)
    assert sorted(calls) == [2, 2, 3]
    assert charges == [_dense_synthesis_work(3) + _dense_synthesis_work(2), _dense_synthesis_work(2)]
    reused = exact_second.data[0].operation.definition
    assert reused == dense_unitary_circuit(three) == exact_first.data[0].operation.definition
    assert reused is not cache[dense_matrix_key(three)]
    reused.x(0)
    assert cache[dense_matrix_key(three)] == dense_unitary_circuit(three)
    assert exact_dense_unitaries(second, charge=lambda work, operation: charges.append(work), cache=cache)
    assert len(charges) == 2 and sorted(calls) == [2, 2, 3]


def test_run_synthesizes_each_basis_unitary_once_and_again_after_reopening(tmp_path, monkeypatch):
    # NWQ-Sim lowers every prepared circuit to U and CX. The three
    # experiments of this Plan hold the same three-qubit unitary, in gate
    # objects that each preparation lowers anew. The Run's synthesis cache
    # keys it by its matrix, so the second preparation reuses the circuit and
    # reserves nothing. A reopened Run starts with an empty cache, so its
    # first preparation synthesizes the unitary again against the same
    # cumulative max_synthesis_work, which, set one unit below two
    # syntheses, refuses it. The refused preparation stays charged but leaves its static
    # item unprepared, so after the limit is raised the next call prepares it.
    # A backend called directly with the Run, outside its preparations, is
    # held to the same limit without reserving anything or using the cache.
    import nwqlib
    from nwqlib import Expectation
    from nwqlib._prepared_execution import Run, _prepare_static_item
    from nwqlib._run_archive import load as load_run
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import nwqsim
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import state_input
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work
    from test_nwqsim_targets import configured, description

    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    matrix = random_unitary(8, seed=46).data
    state = QuantumCircuit(3)
    state.append(UnitaryGate(matrix), [0, 1, 2])
    problem = Expectation(state=state_input(state),
                          observable=ingest_pauli((("ZZZ", 1.0), ("XXX", 0.5), ("YYY", 0.25)), num_qubits=3))
    selected = nwqlib.plan(problem, method=ExpectationMethod(), shots=16, seed=7)
    first, second, third = selected.experiments
    need = _dense_synthesis_work(3)
    calls = _counted_syntheses(monkeypatch)
    journal = tmp_path / "run"
    limits = ExecutionLimits(max_data_bytes=10**7, max_synthesis_work=2 * need - 1)
    with Run(selected, backend=configured(tmp_path), limits=limits, directory=journal) as run:
        handle = _prepare_static_item(selected, run, first)
        assert calls == [3] and run.trace.synthesis_work_reserved == need

        def direct(unitary):
            circuit = QuantumCircuit(3)
            circuit.append(UnitaryGate(unitary), range(unitary.shape[0].bit_length() - 1))
            circuit.measure_all()
            return run.backend.prepare(circuit, observation=handle.record.observation, runtime=handle.record.runtime,
                                       position=len(circuit.data), source_definitions=(), run=run, snapshot="direct")

        # need - 1 units remain. A new two-qubit synthesis fits and is not
        # cached. The three-qubit matrix that the first preparation cached is
        # not read from the cache, so it needs a new synthesis and is refused
        # before that starts.
        direct(random_unitary(4, seed=51).data)
        with pytest.raises(ValueError, match="rest of this Run's max_synthesis_work"):
            direct(matrix)
        assert calls == [3, 2] and len(run._state["synthesis_cache"]) == 1
        # A later preparation still takes the matrix from the cache.
        _prepare_static_item(selected, run, second)
        assert calls == [3, 2]
        assert (run.trace.synthesis_work_reserved, run.trace.preparations) == (need, 2)
    with load_run(journal, backend=configured(tmp_path)) as run:
        with pytest.raises(ValueError, match="max_synthesis_work"):
            _prepare_static_item(selected, run, third)
        assert calls == [3, 2]
        assert (run.trace.synthesis_work_reserved, run.trace.preparations) == (need, 3)
        run.extend_limits(max_synthesis_work=2 * need)
        prepared = _prepare_static_item(selected, run, third)
        assert calls == [3, 2, 3]
        trace = run.trace
        assert (trace.synthesis_work_reserved, trace.preparations) == (2 * need, 4)
        assert trace.limit_amendments[0].synthesis_work == need
    with load_run(journal, backend=configured(tmp_path)) as run:
        assert (run.trace.synthesis_work_reserved, run.trace.preparations) == (2 * need, 4)
        assert _prepare_static_item(selected, run, third).record.content_id == prepared.record.content_id
    assert calls == [3, 2, 3]


def test_noisy_aer_lowering_takes_its_syntheses_from_the_run_cache(monkeypatch):
    # A noise model's basis omits Aer's unitary instruction, so Aer lowering
    # synthesizes the dense unitaries of the supplied state. The state holds
    # one three-qubit matrix twice, under two labels, which one rewrite keys
    # apart. The Run's synthesis cache keys the matrix by its content, so the
    # Run synthesizes and reserves it once, within a limit of one synthesis.
    # Closing the Run releases the cached circuit while the Run and its
    # prepared handle are still referenced, and the reservation stays.
    import gc
    import weakref

    import nwqlib
    from qiskit_aer.noise import NoiseModel
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import state_input
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work

    matrix = random_unitary(8, seed=50).data
    state = QuantumCircuit(3)
    state.append(UnitaryGate(matrix), [0, 1, 2])
    state.append(UnitaryGate(matrix, label="twice"), [2, 1, 0])
    problem = Expectation(state=state_input(state),
                          observable=ingest_pauli((("ZZZ", 1.0), ("XXX", 0.5)), num_qubits=3))
    chosen = nwqlib.plan(problem, method=ExpectationMethod(), shots=16, seed=7)
    backend = AerBackend.from_noise_model(NoiseModel(basis_gates=["cx", "rz", "sx", "x"]))
    need = _dense_synthesis_work(3)
    calls = _counted_syntheses(monkeypatch)
    prepared = nwqlib.prepare(chosen, backend=backend, limits=ExecutionLimits(max_synthesis_work=need))
    run = prepared.run
    cached = [weakref.ref(circuit) for circuit in run._state["synthesis_cache"].values()]
    run.close()
    gc.collect()
    assert calls == [3] and run.trace.synthesis_work_reserved == need
    assert len(cached) == 1 and all(ref() is None for ref in cached)


def test_refused_synthesis_leaves_the_controller_checkpoint_unprepared(tmp_path, monkeypatch):
    # An adaptive controller finds the preparation of its current checkpoint
    # through preparation_for_checkpoint and never repeats it. A preparation
    # that max_synthesis_work refused made no synthesis, so the checkpoint
    # names no preparation, before and after reopening, and the controller
    # prepares again once the limit is raised.
    import nwqlib
    from nwqlib import Expectation
    from nwqlib._prepared_execution import Run, prepare_experiment
    from nwqlib._run_archive import load as load_run
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import nwqsim
    from nwqlib.core.planning import RuntimeOptions
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import state_input
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work
    from test_nwqsim_targets import configured, description

    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    state = QuantumCircuit(2)
    state.append(UnitaryGate(random_unitary(4, seed=47).data), [0, 1])
    problem = Expectation(state=state_input(state), observable=ingest_pauli((("ZZ", 1.0),), num_qubits=2))
    selected = nwqlib.plan(problem, method=ExpectationMethod(), shots=16, seed=7)
    point = selected.resolve(selected.experiments[0].name)
    need = _dense_synthesis_work(2)
    calls = _counted_syntheses(monkeypatch)
    journal = tmp_path / "run"
    limits = ExecutionLimits(max_data_bytes=10**7, max_synthesis_work=need - 1)
    with Run(selected, backend=configured(tmp_path), limits=limits, directory=journal) as run:
        sequence = run.checkpoint({"step": 1})
        with pytest.raises(ValueError, match="max_synthesis_work"):
            prepare_experiment(point, run=run, runtime=RuntimeOptions(seed=5))
        assert calls == [] and run.preparation_for_checkpoint(sequence) is None
    with load_run(journal, backend=configured(tmp_path)) as run:
        assert run.preparation_for_checkpoint(sequence) is None and run.trace.preparations == 1
        run.extend_limits(max_synthesis_work=need)
        prepared = prepare_experiment(point, run=run, runtime=RuntimeOptions(seed=5))
        assert calls == [2]
        identity = run.preparation_for_checkpoint(sequence)
        assert run._state["preparation_charges"][identity].prepared_id == prepared.record.content_id
    with load_run(journal, backend=configured(tmp_path)) as run:
        assert run.preparation_for_checkpoint(sequence) == identity
        assert (run.trace.synthesis_work_reserved, run.trace.preparations) == (need, 2)


def test_run_admits_the_controlled_transform_of_a_supplied_basis_before_its_synthesis(monkeypatch):
    # An off-diagonal FixedGCIM pair controls each basis preparation through
    # a transformed block; the first setting, a diagonal pair, prepares its
    # basis state uncontrolled, so every setting is prepared here. Lowering
    # controls a supplied circuit once per Run, which synthesizes its
    # two-qubit unitary exactly and lets Qiskit control the synthesized
    # gates. The Run charges both against max_synthesis_work before the
    # synthesis starts, one unit below refuses and the laws admit.
    import nwqlib
    from nwqlib.algorithms.gcim import FixedGCIM
    from nwqlib.execution import ExecutionLimits
    from nwqlib.problems.records import Eigenproblem
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work

    basis = QuantumCircuit(2)
    basis.append(UnitaryGate(random_unitary(4, seed=48).data), [0, 1])
    selected = nwqlib.plan(Eigenproblem(A=np.diag([1.0, -1.0, 0.5, -0.5])),
                           method=FixedGCIM(basis=(basis, [1.0, 0.0, 0.0, 0.0])), execution="quantum", seed=7)
    need = _dense_synthesis_work(2) + _gatewise_control_work(2, 1)
    calls = _counted_syntheses(monkeypatch)
    with pytest.raises(ValueError, match="max_synthesis_work"):
        nwqlib.prepare(selected, limits=ExecutionLimits(max_synthesis_work=need - 1), settings="all")
    assert calls == []
    prepared = nwqlib.prepare(selected, limits=ExecutionLimits(max_synthesis_work=need), settings="all")
    assert calls == [2] and prepared.run.trace.synthesis_work_reserved == need
    prepared.run.close()


@pytest.mark.parametrize("solver,controls,specializations,route",
                         [("shortcut_native_svp", 1, 2, "gatewise"), ("shortcut_native_svp", 1, 2, "auto"),
                          ("shortcut_dilation", 2, 4, "auto")])
def test_qls_shortcut_admits_the_supplied_rhs_syntheses_it_controls(monkeypatch, solver, controls,
                                                                     specializations, route):
    # A Dalzell shortcut (arXiv:2406.12086v2) controls the RHS preparation,
    # forward and adjoint,
    # and shortcut_dilation both again for each value of its dilation qubit.
    # Each controlled specialization synthesizes the dense unitary of this
    # supplied circuit again and Qiskit controls the synthesized gates, on
    # every route. Planning charges them with the two controlled queries of
    # the dense dilation of A on three qubits, which "auto" controls through
    # the whole controlled matrix with one control and gate-wise with two,
    # and a whole Run makes exactly these syntheses and control steps. The
    # coarse epsilon_inv keeps the kernel-reflection fit, which planning
    # admits after the syntheses, below the smallest of these requirements,
    # so max_work equal to the requirement admits the Plan.
    import nwqlib
    from nwqlib import LinearSystem, NormalizedExpectation
    from nwqlib.algorithms.qls import QLS
    from nwqlib.problems.inputs import state_input
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work

    rhs = QuantumCircuit(2)
    rhs.append(UnitaryGate(random_unitary(4, seed=49).data), [0, 1])
    problem = LinearSystem(A=np.diag([1.0, 0.8, 0.6, 0.5]) + 0.05, b=state_input(rhs))
    output = NormalizedExpectation(observable=np.diag([1.0, -1.0, 0.5, 0.0]))
    whole = route == "auto" and controls == 1
    query = (_controlled_synthesis_work(3 + controls) if whole
             else _dense_synthesis_work(3) + _gatewise_control_work(3, controls))
    need = 2 * query + specializations * (_dense_synthesis_work(2) + _gatewise_control_work(2, controls))

    def planned(max_work):
        method = QLS(solver=solver, encoded_solution_norm_estimate=1.5, epsilon_inv=0.5,
                     block_encoding_implementation="dense_dilation", max_work=max_work,
                     dense_control_route=route)
        return nwqlib.plan(problem, method=method, output=output, seed=7)

    with pytest.raises(ValueError, match="QLS controlled dense-query synthesis exceeds max_work"):
        planned(need - 1)
    selected = planned(need)
    calls = _counted_syntheses(monkeypatch)
    steps = _counted_dense_controls(monkeypatch)
    nwqlib.solve(selected, progress=False)
    assert sorted(calls) == [2] * specializations + ([3 + controls] * 2 if whole else [3, 3])
    assert sorted(steps) == [(controls, (2,))] * specializations + [(controls, (3,))] * 2


def test_adapt_admits_the_controlled_synthesis_of_a_supplied_reference(monkeypatch):
    # ADAPT's Hadamard-test and joint-state queries control the reference preparation, and
    # the Run's method context builds that controlled gate once. For a
    # supplied reference with a two-qubit unitary that is one exact
    # synthesis and one control step, which planning charges to
    # max_products. Explicit verification synthesizes the uncontrolled
    # reference again and charges it to its own max_products before the
    # synthesis starts.
    from nwqlib._prepared_execution import Run
    from nwqlib.algorithms.gcim import AdaptVerificationOptions
    from nwqlib.operators import ingest_pauli
    from test_adapt_primary import plan_for
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work

    reference = QuantumCircuit(2)
    reference.append(UnitaryGate(random_unitary(4, seed=50).data), [0, 1])
    options = dict(A=ingest_pauli((("ZI", 1.0), ("IZ", 0.5)), num_qubits=2), initial_state=reference,
                   pool_rows=((("YI", 1j),),))
    need = _dense_synthesis_work(2) + _gatewise_control_work(2, 1)
    with pytest.raises(ValueError, match="ADAPT controlled reference synthesis exceeds max_work"):
        plan_for(max_products=need - 1, **options)
    selected = plan_for(max_products=need, **options)
    calls = _counted_syntheses(monkeypatch)
    result = Run(selected).wait(timeout=60, poll_interval=0)
    assert calls == [2]
    calls.clear()
    with pytest.raises(ValueError, match="supplied reference circuit"):
        result.verify(checks=AdaptVerificationOptions(name="r", comparisons=("residual",),
                                                      max_products=_dense_synthesis_work(2) - 1))
    assert calls == []
    result.verify(checks=AdaptVerificationOptions(name="r", comparisons=("residual",)))
    assert calls == [2]
