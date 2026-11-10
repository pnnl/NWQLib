"""Direct installed-model comparison, count mapping and independent cycle relations."""

import json
import subprocess
import sys
from collections import Counter
from math import pi

import pytest

from nwqlib.backends.nwqec import compile_logical
from nwqlib.backends.qre import PhysicalProjection, estimate_physical
from test_nwqec_compilation import lowered_rotation, require_native


@pytest.fixture(scope="module")
def compilation():
    require_native()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QDK_PYTHON_TELEMETRY", "none")
        pytest.importorskip("qdk", reason="physical projection requires the optional qre extra")
        yield compile_logical(lowered_rotation(), epsilon=0.001)


def direct_qdk_configurations(counts):
    """Return (physical qubits, seconds, modeled error) of ``counts`` from a direct call to the installed QDK.

    The parameters are the default model of ``estimate_physical``.
    """
    from qdk.estimator import LogicalCounts
    from qdk.qre import estimate, ErrorComposition, PSSPC, LatticeSurgery
    from qdk.qre.application import QSharpApplication
    from qdk.qre.models.qubits import GateBased
    from qdk.qre.models.qec import SurfaceCode
    from qdk.qre.models.factories import Litinski19Factory

    arch = GateBased(error_rate=1e-4, gate_time=100, two_qubit_gate_time=100, measurement_time=500)
    direct = estimate(
        QSharpApplication(LogicalCounts(counts), use_cache=False, use_trace_backend=False),
        arch,
        SurfaceCode.q(distance=[3, 5, 7]) * Litinski19Factory.q(),
        PSSPC.q(num_ts_per_rotation=20, ccx_magic_states=False)
        * LatticeSurgery.q(slow_down_factor=1.0),
        max_error=0.01,
        composition=ErrorComposition.UnionBound,
        use_graph=False,
        post_process=False,
    )
    return sorted((row.qubits, row.runtime / 1_000_000_000, row.error) for row in direct)


def test_projection_matches_direct_qdk_and_fixed_time_scaling(compilation, monkeypatch, tmp_path):
    monkeypatch.setenv("QDK_PYTHON_TELEMETRY", "none")
    actual = estimate_physical(compilation, measurement_count=1)
    doubled = estimate_physical(
        compilation, measurement_count=1, gate_time_ns=200, measurement_time_ns=1000
    )
    expected_counts = dict(
        numQubits=1,
        tCount=sum(value for name, value in compilation.raw_counts if name in {"t", "tdg"}),
        measurementCount=1,
        rotationCount=0,
        rotationDepth=0,
        cczCount=0,
        ccixCount=0,
    )
    assert dict(actual.logical_counts) == expected_counts
    expected = direct_qdk_configurations(expected_counts)
    assert len(actual.entries) == len(expected) > 0
    for row, (qubits, seconds, error) in zip(
        sorted(actual.entries, key=lambda row: row.physical_qubits), expected, strict=True
    ):
        assert row.physical_qubits == qubits
        assert row.runtime_seconds == pytest.approx(seconds, rel=1e-15, abs=0)
        assert row.modeled_error == error
        configuration = json.loads(row.configuration_json)
        assert {
            node["parameters"]["distance"]
            for node in configuration["nodes"]
            if node["transform"] == "SurfaceCode"
        } <= {3, 5, 7}
        other = next(
            entry
            for entry in doubled.entries
            if entry.physical_qubits == qubits and entry.modeled_error == error
        )
        assert other.runtime_seconds == pytest.approx(2 * seconds, rel=1e-15, abs=0)
    from nwqlib.backends import _auxiliary_process

    monkeypatch.setattr(
        _auxiliary_process,
        "run_auxiliary",
        lambda *a, **k: pytest.fail("stored projection reran QDK"),
    )
    actual.save(tmp_path / "projection.json")
    assert PhysicalProjection.load(tmp_path / "projection.json") == actual
    assert "Clifford schedule" in str(actual) and "synthesis error" in str(actual)
    compilation.save(tmp_path / "compiled.json")
    # A fresh process catches even an unnecessary SDK import, including a
    # swallowed import attempt that would not be visible as an exception.
    code = """
import builtins, importlib.abc, importlib.util, sys
original = builtins.__import__
attempts = []
def check(name):
    if name.split('.')[0] in {'nwqec', 'qdk', 'qiskit', 'qiskit_aer'}:
        attempts.append(name)
        raise ImportError('forbidden SDK import: ' + name)
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    absolute = importlib.util.resolve_name('.'*level + name, globals['__package__']) if level else name
    check(absolute)
    return original(name, globals, locals, fromlist, level)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        check(fullname)
        return None
sys.meta_path.insert(0, Guard())
builtins.__import__ = guarded
from nwqlib.backends.nwqec import LogicalCompilation
from nwqlib.backends.qre import PhysicalProjection
print(LogicalCompilation.load(sys.argv[1]))
print(PhysicalProjection.load(sys.argv[2]))
assert not attempts, attempts
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path / "compiled.json"),
            str(tmp_path / "projection.json"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def test_a_compiled_query_with_sx_gates_projects_with_counts_read_from_its_own_output(monkeypatch):
    """Fixed sx and sxdg gates add nothing to the counts of QDK's model, and the rest of the record is unchanged.

    The first query of a two-qubit RWPE whose Hamiltonian holds YZ keeps both gates in its final output. The T count is
    read from that output_qasm and checked against raw_counts, the width and terminal measurements come from the
    prepared circuit, and the physical configurations come from a direct QDK call. Clifford+T counts can differ between
    compiler runs, so everything here reads one compilation.
    """
    require_native()
    monkeypatch.setenv("QDK_PYTHON_TELEMETRY", "none")
    pytest.importorskip("qdk", reason="physical projection requires the optional qre extra")
    import nwqlib
    from nwqlib.algorithms import RWPE
    from nwqlib.operators import ingest_pauli

    target = ingest_pauli((("XI", .7), ("ZX", .4), ("YZ", .3), ("ZZ", .2)), num_qubits=2)
    selected = nwqlib.plan(nwqlib.Eigenproblem(A=target), seed=7,
                           method=RWPE(initial_state=[1, 0, 0, 0], max_steps=6, ctrl_power="trotter_error_budgeted",
                                       ctrl_power_budget=.01))
    prepared = nwqlib.prepare(selected, settings="first", progress=False)
    with prepared.run:
        source = prepared.circuits[0]
        compiled = compile_logical(prepared, rz_err="total", epsilon=1e-3)
    written = Counter(line.split()[0] for line in compiled.output_qasm.splitlines() if line.strip())
    counted = {name: written[name] for name in ("t", "tdg", "sx", "sxdg")}
    assert counted["sx"] > 0 and counted["sxdg"] > 0
    assert counted == {name: dict(compiled.raw_counts).get(name, 0) for name in counted}
    expected_counts = dict(
        numQubits=source.num_qubits,
        tCount=counted["t"] + counted["tdg"],
        measurementCount=sum(item.operation.name == "measure" for item in source.data),
        rotationCount=0,
        rotationDepth=0,
        cczCount=0,
        ccixCount=0,
    )
    assert expected_counts["tCount"] > 0 and expected_counts["measurementCount"] > 0
    actual = estimate_physical(compiled)
    assert dict(actual.logical_counts) == expected_counts
    assert actual.count_population == "one static compiler body"
    expected = direct_qdk_configurations(expected_counts)
    assert len(actual.entries) == len(expected) > 0
    for row, (qubits, seconds, error) in zip(
        sorted(actual.entries, key=lambda row: row.physical_qubits), expected, strict=True
    ):
        assert (row.physical_qubits, row.modeled_error) == (qubits, error)
        assert row.runtime_seconds == pytest.approx(seconds, rel=1e-15, abs=0)


def test_a_compiled_record_outside_full_clifford_t_is_refused_before_qdk(monkeypatch):
    """Only a full clifford_t record of admitted gate labels reaches QDK's T-count model.

    Counting another gate as no T would understate the physical cost. A count-only record stops before fusion, and a
    clifford_reduction output writes non-Clifford gates as rotations. A controlled SX is H conjugate to a controlled S,
    which is not Clifford. A rotation label is refused whatever its angle, so rx(pi/2) is refused although that angle
    is Clifford. The two copies differ from an admitted full record only in raw_counts, since a real clifford_t output
    should not hold those labels.
    """
    require_native()
    from nwqlib.backends import _auxiliary_process

    full = compile_logical(lowered_rotation(), epsilon=1e-3)
    records = (compile_logical(lowered_rotation(), count_only=True),
               compile_logical(lowered_rotation(pi / 4), target="clifford_reduction"))
    monkeypatch.setattr(_auxiliary_process, "run_auxiliary", lambda *a, **k: pytest.fail("QDK ran"))
    for record in records:
        with pytest.raises(ValueError, match="full clifford_t compilation"):
            estimate_physical(record, measurement_count=1)
    for label in ("csx", "rx(pi/2)"):
        residual = full.revise(raw_counts=(*full.raw_counts, (label, 1)))
        with pytest.raises(ValueError, match="residual unsupported gates"):
            estimate_physical(residual, measurement_count=1)


def test_a_resource_estimate_of_one_execution_is_projected_with_a_t_count_independent_of_shots(monkeypatch):
    """A clifford_t estimate is projected only when it counts one circuit execution.

    A controlled arbitrary phase gets 100 T at eps = 1e-10 under the leading-term synthesis, both
    without a terminal batch and in a one-shot batch, and a branch between one and two such phases
    inside the execution gives the larger path's 200 T. Eight repetitions, a Plan with eight shots
    and an unknown T count are refused before QDK runs, with a remedy that fits the number of
    settings. Measurements follow the compilation rule.
    """
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms import ExpectationMethod, RWPE
    from nwqlib.backends import _auxiliary_process
    from nwqlib.operators import ingest_pauli
    from nwqlib.resources import TYPICAL_ROTATION_SYNTHESIS, ResourceContext
    from nwqlib.resources.fold import estimate
    from qiskit import QuantumCircuit
    from test_resource_fold import SYNTHESIS, branch_call, controlled_phase, single_call

    requests = []
    reply = dict(version="1.32.3", module_file="qdk", model={}, entries=[], statistics={}, diagnostics="")
    monkeypatch.setattr(_auxiliary_process, "run_auxiliary",
                        lambda kind, request, **kwargs: requests.append(request["counts"]) or (reply, 0.0))
    rule = ResourceContext(**SYNTHESIS, synthesis=TYPICAL_ROTATION_SYNTHESIS)
    block = controlled_phase("near", complex(-2**-52, 1))
    unbatched = estimate_physical(estimate(single_call(block), context=rule), measurement_count=1)
    one_shot = estimate_physical(estimate(single_call(block, repetitions=1), context=rule), measurement_count=1)
    assert [counts["tCount"] for counts in requests] == [100, 100]
    assert unbatched.count_population.startswith("one circuit execution (settings=0")
    assert one_shot.count_population.startswith("one circuit execution (settings=1, acquisitions=1")
    assert (one_shot.count_source, one_shot.t_interpretation, one_shot.rotation_count) == ("resource_estimate", "estimate", 1)
    assert (one_shot.synthesis_source, one_shot.requested_synthesis_epsilon) == (TYPICAL_ROTATION_SYNTHESIS, 1e-10)
    assert one_shot.synthesis_error_policy == "resource_context_per_rotation" and one_shot.rotation_error_sum >= 1e-10
    for repetitions in (None, 1):
        branched = estimate_physical(compilation=estimate(branch_call(block, repetitions=repetitions), context=rule),
                                     measurement_count=1)
        assert branched.t_interpretation == "estimate"
    assert [counts["tCount"] for counts in requests] == [100, 100, 200, 200]
    with pytest.raises(ValueError, match="explicit measurement_count"):
        estimate_physical(estimate(single_call(block), context=rule))
    for construction in (single_call(block, repetitions=8), branch_call(block, repetitions=8)):
        with pytest.raises(ValueError, match="one circuit execution"):
            estimate_physical(estimate(construction, context=rule), measurement_count=1)
    with pytest.raises(ValueError, match="T count is not a known integer"):
        estimate_physical(estimate(single_call(block), context=ResourceContext(basis="clifford_t")),
                          measurement_count=1)
    observable = ingest_pauli([("Z", 1)], num_qubits=1)
    eight = nwqlib.plan(Expectation(state=[1.0, 0.0], observable=observable), method=ExpectationMethod(), shots=8, seed=7)
    with pytest.raises(ValueError, match="one circuit execution") as one_setting:
        estimate_physical(nwqlib.estimate(eight, context=ResourceContext(basis="clifford_t")))
    assert "shots=1" in str(one_setting.value)
    # RWPE takes one shot per step and refuses any other, so only compiling its circuits one at a time can help.
    target = ingest_pauli((("XI", .7), ("ZX", .4), ("YZ", .3), ("ZZ", .2)), num_qubits=2)
    steps = nwqlib.plan(nwqlib.Eigenproblem(A=target), seed=7,
                        method=RWPE(initial_state=[1, 0, 0, 0], max_steps=6, ctrl_power="trotter_error_budgeted",
                                    ctrl_power_budget=.01))
    with pytest.raises(ValueError, match="one circuit execution") as six_settings:
        estimate_physical(nwqlib.estimate(steps, context=ResourceContext(basis="clifford_t")))
    assert "settings=6" in str(six_settings.value) and "compile_logical" in str(six_settings.value)
    assert "shots=1" not in str(six_settings.value)
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.rz(0.1234, 0)
    rotation = nwqlib.plan(Expectation(state=circuit, observable=observable), method=ExpectationMethod(),
                           shots=1, seed=7)
    with pytest.raises(ValueError, match="compile the circuit with compile_logical"):
        estimate_physical(nwqlib.estimate(rotation, context=rule))
    assert len(requests) == 4


def test_a_one_shot_plan_estimate_reaches_qdk_with_its_terminal_measurement():
    """A one-shot basis-state Plan has an exact T count of 0 and one terminal measurement, which cannot be overridden."""
    pytest.importorskip("qdk", reason="physical projection requires the optional qre extra")
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms import ExpectationMethod
    from nwqlib.operators import ingest_pauli
    from nwqlib.resources import ResourceContext

    selected = nwqlib.plan(Expectation(state=[1.0, 0.0], observable=ingest_pauli([("Z", 1)], num_qubits=1)),
                           method=ExpectationMethod(), shots=1, seed=7)
    workload = nwqlib.estimate(selected, context=ResourceContext(basis="clifford_t"))
    with pytest.raises(ValueError, match="cannot be overridden"):
        estimate_physical(workload, measurement_count=2)
    projection = estimate_physical(workload)
    counts = dict(projection.logical_counts)
    assert (counts["numQubits"], counts["tCount"], counts["measurementCount"]) == (1, 0, 1)
    assert (projection.count_source, projection.t_interpretation, projection.measurement_source) == (
        "resource_estimate", "exact", "original_terminal_readout")
    assert projection.requested_synthesis_epsilon is None and projection.entries
