"""Tiny external Method through the shared selected-experiment lifecycle."""

from types import SimpleNamespace
from typing import ClassVar, Literal

import numpy as np
import pytest

from nwqlib.algorithms.protocol import AlgorithmDescriptor, Method
from nwqlib.blocks.records import SelectedConstruction
from nwqlib.core.analysis import Result, capture_analysis_origin
from nwqlib.core.planning import Experiment, ObservationSpec, Plan, RandomStreams
from nwqlib.core.records import Record, Source
from nwqlib.execution import CountsSampling, ExecutionLimits
from nwqlib.ir import Allocate, ClassicalValue, Definition, Measure, Program, Register, Sequence
from nwqlib.scientist import plan, prepare, solve, submit


class ScalarOutput(Record):
    quantity: str = "Z expectation"


class ToyProblem(Record):
    width: int = 1

    def default_output(self):
        return ScalarOutput()


class ToyResult(Result):
    value: float


class ToyMethod(Method):
    scale: float = 2.
    calls: ClassVar[list] = []

    @property
    def descriptor(self):
        return AlgorithmDescriptor(method="external.toy", version="1")

    def plan(self, problem, *, output, execution, shots, rng):
        self.calls.append("plan")
        rng.method.integers(100)  # Must not consume the separate backend stream.
        program = Program(root="body", registers=(Register(name="q", width=problem.width),),
            classical=(ClassicalValue(name="c", dtype="bits", width=problem.width),),
            definitions=(Definition(id="body", node=Sequence(children=("allocate", "measure"))),
                Definition(id="allocate", node=Allocate(wire="q")),
                Definition(id="measure", node=Measure(wire="q", result="c"))))
        return Plan(problem=problem, method=self, output=output, execution=execution,
            shots=shots, randomness=rng.snapshot(), construction=SelectedConstruction(program=program, selections=()),
            experiments=(Experiment(name="z", setting="computational", observation=ObservationSpec(kind="counts", shots=shots)),))

    def analyze(self, selected, data, *, settings):
        self.calls.append("analyze")
        chunk, = data.observations.chunks
        histogram = chunk.histogram()
        counts = dict(zip(histogram.index_list(), histogram.weights.tolist()))
        mean = (counts.get(0, 0)-counts.get(1, 0))/chunk.returned_shots
        return ToyResult(plan_id=selected.content_id, construction_id=selected.construction.content_id,
            observation_id=data.observations.content_id, contribution_ids=(chunk.content_id,),
            value=self.scale*mean,
            origin=capture_analysis_origin(analyzer=self.descriptor.source, method_id=self.content_id))


class StubBackend(Record):
    kind: Literal["stub"] = "stub"
    calls: ClassVar[list] = []
    fail: ClassVar[bool] = False

    def target_for(self, observation):
        return SimpleNamespace(readouts=("counts",))

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot):
        from nwqlib.backends.connection import NativePreparation
        self.calls.append(("prepare", runtime.seed))
        source = Source(name="supplied", version="1", domain="stubbed native boundary", reference="test")
        native = SimpleNamespace(circuit=circuit, shots=observation.shots, seed=runtime.seed)
        return NativePreparation(native=native, target=source, compiler=source, native_basis=("measure",),
            environment=(), quantum_layout=(), classical_layout=(), logical_to_native=(0,), operations=1,
            population="unconditional", transformation="supplied", counts_sampling=CountsSampling(kind="fixed_seed", seed=runtime.seed))

    def submit(self, native, *, submission_id, run):
        self.calls.append(("submit", native.shots, native.seed))
        assert run.trace.events[-1].status == "reserved"  # Intent is visible before launch.
        assert run._state["workflow_items"]["z"].attempt == run.trace.events[-1].attempt
        if self.fail:
            raise OSError("supplied acquisition failure")
        return SimpleNamespace(raw_output={"counts": {"0": 3, "1": 1}}, metadata={"native_job_id": submission_id})


@pytest.fixture
def stub(monkeypatch):
    from nwqlib.blocks import lowering
    ToyMethod.calls.clear()
    StubBackend.calls.clear()
    StubBackend.fail = False

    class Bits(tuple):
        def __new__(cls, name, bits):
            obj = super().__new__(cls, bits)
            obj.name = name
            return obj

    def lower(construction, **kwargs):
        width = construction.program.registers[0].width
        q, c = Bits("q", range(width)), Bits("c", range(width))
        circuit = SimpleNamespace(data=("measure",), qregs=(q,), cregs=(c,), num_qubits=width,
                                  find_bit=lambda bit: SimpleNamespace(index=bit))
        return lowering.LogicalCircuit(construction.content_id, circuit, (("q", tuple(q)),),
                                       (("c", tuple(c)),), 3, 1, ())
    monkeypatch.setattr(lowering, "_lower_qiskit", lower)
    return StubBackend()


def test_selected_plan_executes_without_replanning_and_reanalysis_shares_data(stub):
    problem, method = ToyProblem(), ToyMethod()
    selected = plan(problem, method=method, shots=4, seed=19)
    prepared = prepare(selected, backend=stub)
    assert ToyMethod.calls == ["plan"] and len(StubBackend.calls) == 1
    run = submit(prepared)
    result = run.wait()
    assert result.value == 1.  # Supplied 3:1 histogram gives <Z>=1/2, scale=2.
    assert result.plan is selected and selected.problem is problem and selected.method is method
    expected_backend = np.random.Generator(np.random.PCG64(np.random.SeedSequence(19).spawn(2)[1]))
    expected_seed = int(expected_backend.integers(0, 2**32, dtype="uint32"))
    assert StubBackend.calls == [("prepare", expected_seed), ("submit", 4, expected_seed)]
    before = tuple(StubBackend.calls)
    changed = result.analyze()
    assert changed is not result and changed.data is result.data and changed.value == result.value
    assert changed.origin.invocation_id != result.origin.invocation_id
    result.report()
    assert tuple(StubBackend.calls) == before and ToyMethod.calls.count("plan") == 1
    with pytest.raises(ValueError, match="cannot be overridden"):
        solve(selected, backend=stub, shots=4)
    assert tuple(StubBackend.calls) == before


def test_failed_acquisition_spends_original_shots_and_extension_never_runs(stub):
    selected = plan(ToyProblem(), method=ToyMethod(), shots=4, seed=7)
    prepared = prepare(selected, backend=stub, limits=ExecutionLimits(max_total_shots=4))
    StubBackend.fail = True
    with pytest.raises(OSError, match="supplied acquisition"):
        submit(prepared)
    run = prepared.run
    assert run.exposure["uncertain"]["shots"] == 4
    before, random = tuple(StubBackend.calls), run.rng.snapshot()
    run.extend_limits(max_total_shots=9)
    assert run.rng.snapshot() == random and tuple(StubBackend.calls) == before
    run.check_capacity(shots=5)
    with pytest.raises(ValueError, match=r"run\.extend_limits\(max_total_shots="):
        run.check_capacity(shots=6)
    run.cancel()
    with pytest.raises(ValueError, match="cancellation"):
        run.check_capacity(circuits=1)
    assert run.exposure["uncertain"]["shots"] == 4


def test_data_and_complete_workload_limits_stop_before_native_work(stub):
    selected = plan(ToyProblem(), method=ToyMethod(), shots=4, seed=7)
    with pytest.raises(ValueError, match=r"ExecutionLimits\(max_total_shots="):
        prepare(selected, backend=stub, limits=ExecutionLimits(max_total_shots=3))
    assert StubBackend.calls == []
    with pytest.raises(ValueError, match=r"ExecutionLimits\(max_data_bytes="):
        prepare(selected, backend=stub, limits=ExecutionLimits(max_data_bytes=100))
    assert StubBackend.calls == []


def test_random_stream_snapshot_preserves_both_draw_positions():
    original = RandomStreams(17)
    original.method.integers(100, size=3)
    original.next_seed()
    restored = RandomStreams.restore(original.snapshot())
    assert restored.next_seed() == original.next_seed()
    assert restored.method.integers(100) == original.method.integers(100)


def test_aer_width_precedes_lowering_and_memory_reaches_native_configuration(stub, monkeypatch):
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.backends.targets import AER_STATEVECTOR_TARGET
    from nwqlib.blocks import lowering

    class LocalStub(StubBackend):
        kind: Literal["qiskit_aer"] = "qiskit_aer"

    selected = plan(ToyProblem(width=2), method=ToyMethod(), shots=4, seed=7)
    original = lowering._lower_qiskit
    monkeypatch.setattr(lowering, "_lower_qiskit", lambda *a, **k: pytest.fail("width refusal followed lowering"))
    with pytest.raises(ValueError, match="max_simulation_qubits"):
        prepare(selected, backend=LocalStub(), limits=ExecutionLimits(max_simulation_qubits=1))
    assert StubBackend.calls == []
    monkeypatch.setattr(lowering, "_lower_qiskit", original)
    legal = plan(ToyProblem(), method=ToyMethod(), shots=4, seed=7)
    prepare(legal, backend=LocalStub(), limits=ExecutionLimits(max_simulation_qubits=1))
    assert len(StubBackend.calls) == 1

    received = []
    class StopAtConfiguration(Exception):
        pass
    def simulator(**options):
        received.append(options)
        raise StopAtConfiguration
    monkeypatch.setattr(aer, "AerSimulator", simulator)
    with pytest.raises(StopAtConfiguration):
        aer._prepare_aer_execution(AER_STATEVECTOR_TARGET, SimpleNamespace(), shots=None, seed=7,
                                   simulator_memory_mb=37)
    assert received[0]["max_memory_mb"] == 37
