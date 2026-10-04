"""Direct installed-model comparison, count mapping and independent cycle relations."""

import json
import math
import subprocess
import sys

import pytest

from nwqlib.backends.nwqec import compile_logical
from nwqlib.backends.qre import PhysicalProjection, estimate_physical
from test_nwqec_compilation import expectation_plan, lowered_rotation, require_native


@pytest.fixture(scope="module")
def compilation():
    require_native()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QDK_PYTHON_TELEMETRY", "none")
        pytest.importorskip("qdk", reason="physical projection requires the optional qre extra")
        yield compile_logical(lowered_rotation(), epsilon=0.001)


def test_projection_matches_direct_qdk_and_fixed_time_scaling(compilation, monkeypatch, tmp_path):
    monkeypatch.setenv("QDK_PYTHON_TELEMETRY", "none")
    from qdk.estimator import LogicalCounts
    from qdk.qre import estimate, ErrorComposition, PSSPC, LatticeSurgery
    from qdk.qre.application import QSharpApplication
    from qdk.qre.models.qubits import GateBased
    from qdk.qre.models.qec import SurfaceCode
    from qdk.qre.models.factories import Litinski19Factory

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
    arch = GateBased(error_rate=1e-4, gate_time=100, two_qubit_gate_time=100, measurement_time=500)
    direct = estimate(
        QSharpApplication(LogicalCounts(expected_counts), use_cache=False, use_trace_backend=False),
        arch,
        SurfaceCode.q(distance=[3, 5, 7]) * Litinski19Factory.q(),
        PSSPC.q(num_ts_per_rotation=20, ccx_magic_states=False)
        * LatticeSurgery.q(slow_down_factor=1.0),
        max_error=0.01,
        composition=ErrorComposition.UnionBound,
        use_graph=False,
        post_process=False,
    )
    expected = sorted((row.qubits, row.runtime / 1_000_000_000, row.error) for row in direct)
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


def test_missing_readout_and_residual_gates_refuse_before_model(compilation, monkeypatch):
    from nwqlib.backends import _auxiliary_process

    monkeypatch.setattr(
        _auxiliary_process,
        "run_auxiliary",
        lambda *a, **k: pytest.fail("invalid counts reached QDK"),
    )
    for options in (
        {},
        {"measurement_count": -1},
        {"measurement_count": 1, "max_error": 0},
        {"measurement_count": 100001},
    ):
        with pytest.raises(ValueError):
            estimate_physical(compilation, **options)
    residual = compilation.revise(raw_counts=(("rz", 1),))
    with pytest.raises(ValueError, match="residual"):
        estimate_physical(residual, measurement_count=1)


def test_empty_finite_model_set_is_preserved(compilation):
    tiny = compile_logical(lowered_rotation(math.pi / 4))
    assert sum(value for name, value in tiny.raw_counts if name in {"t", "tdg"}) == 1
    result = estimate_physical(tiny, measurement_count=1)
    assert result.entries == ()
    assert "No feasible result" in str(result)


def test_original_counts_mapping_cannot_be_overridden(monkeypatch):
    require_native()
    from nwqlib import prepare
    from nwqlib.backends import _auxiliary_process

    prepared = prepare(expectation_plan(shots=8))
    try:
        compiled = compile_logical(prepared)
    finally:
        prepared.run.close()
    captured = []

    def stop_at_model(operation, request, **kwargs):
        captured.append(request)
        raise RuntimeError("mapping reached model")

    monkeypatch.setattr(_auxiliary_process, "run_auxiliary", stop_at_model)
    with pytest.raises(ValueError, match="cannot be overridden"):
        estimate_physical(compiled, measurement_count=1)
    assert captured == []
    with pytest.raises(RuntimeError, match="mapping reached model"):
        estimate_physical(compiled)
    assert captured[0]["counts"]["measurementCount"] == len(compiled.terminal_measurements) == 1
