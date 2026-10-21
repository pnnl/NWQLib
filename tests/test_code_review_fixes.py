"""Tests for scientific maintenance command behavior."""

from __future__ import annotations

import json
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from conftest import load_docs_script as _load_script


def _write_probe_module(tmp_path, monkeypatch, probes, text: str) -> Path:
    target = tmp_path / "src" / "nwqlib" / "demo.py"
    target.parent.mkdir(parents=True)
    target.write_text(text, encoding="utf-8")
    monkeypatch.setattr(probes, "ROOT", tmp_path)
    return target


@pytest.mark.parametrize(
    ("source", "unresolved"),
    [("def mutate():\n    x = 1\n", "x = 9")],
)
def test_mutation_campaign_resolves_all_targets_before_running_baseline(
    tmp_path, monkeypatch, source, unresolved
) -> None:
    probes = _load_script("docs/scripts/mutation_probes.py")
    target = _write_probe_module(tmp_path, monkeypatch, probes, source)
    original = target.read_bytes()
    first = probes.Probe(
        name="first",
        replacement=probes.Replacement(module="nwqlib.demo", qualname="mutate", old="x = 1", new="x = 2"),
        pytest_args=("tests/test_demo.py::test_demo",),
    )
    stale = probes.Probe(
        name="stale",
        replacement=probes.Replacement(module="nwqlib.demo", qualname="mutate", old=unresolved, new="x = 2"),
        pytest_args=first.pytest_args,
    )
    monkeypatch.setattr(probes, "PROBES", (first, stale))
    monkeypatch.setattr(sys, "argv", ["mutation_probes.py"])

    def unexpected(*_args, **_kwargs):
        pytest.fail("an unresolved campaign started a baseline or mutation")

    monkeypatch.setattr(probes, "_check_baseline", unexpected)
    monkeypatch.setattr(probes, "_run_probe", unexpected)
    with pytest.raises(RuntimeError, match="stale.*zero occurrences"):
        probes.main()
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    ("returncode", "stdout", "requires", "junit_cases"),
    [
        (1, "1 failed in 0.01s", (), '<testcase><failure message="assert False"/></testcase>'),
        (0, "1 skipped in 0.01s", (), "<testcase><skipped/></testcase>"),
    ],
)
def test_mutation_campaign_rejects_unavailable_baseline_before_write(
    tmp_path, monkeypatch, returncode, stdout, requires, junit_cases
) -> None:
    probes = _load_script("docs/scripts/mutation_probes.py")
    target = _write_probe_module(
        tmp_path,
        monkeypatch,
        probes,
        "def mutate():\n    x = 1\n",
    )
    probe = probes.Probe(
        name="demo",
        replacement=probes.Replacement(
            module="nwqlib.demo",
            qualname="mutate",
            old="x = 1",
            new="x = 2",
        ),
        pytest_args=("tests/test_demo.py::test_demo",),
        requires=requires,
    )
    monkeypatch.setattr(probes, "PROBES", (probe,))
    monkeypatch.setattr(sys, "argv", ["mutation_probes.py"])
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        assert target.read_bytes() == b"def mutate():\n    x = 1\n"
        if command[-1].startswith("--junitxml="):
            # Simulate the subprocess report without permitting source writes.
            with open(command[-1].split("=", 1)[1], "w", encoding="utf-8") as report:
                report.write(f"<testsuites><testsuite>{junit_cases}</testsuite></testsuites>")
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    def forbid_write(*_args, **_kwargs):
        pytest.fail("source write before a passing baseline")

    monkeypatch.setattr(probes.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "write_text", forbid_write)
    monkeypatch.setattr(Path, "write_bytes", forbid_write)
    with pytest.raises(RuntimeError, match="baseline"):
        probes.main()
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("returncode", "stdout", "failure_message", "expected"),
    [
        (
            1,
            "captured text: 1 skipped, 1 error\nFAILED tests/test_demo.py::test_demo\n"
            "1 failed in 0.01s",
            "assert not True",
            "KILLED",
        ),
        (0, "1 passed in 0.01s", None, "SURVIVED"),
        (1, "1 error in 0.01s", None, "infrastructure"),
        (None, "", None, "interrupted"),
        ("timeout", "", None, "no result within"),
    ],
)
def test_mutation_campaign_classifies_and_restores(
    tmp_path, monkeypatch, capsys, returncode, stdout, failure_message, expected
) -> None:
    probes = _load_script("docs/scripts/mutation_probes.py")
    target = _write_probe_module(
        tmp_path,
        monkeypatch,
        probes,
        "def mutate():\r\n    x = 1\r\n    return x\r\n",
    )
    original = target.read_bytes()
    probe = probes.Probe(
        name="demo",
        replacement=probes.Replacement(
            module="nwqlib.demo",
            qualname="mutate",
            old="x = 1\r\n    return x",
            new="x = 2\r\n    return x",
        ),
        pytest_args=("tests/test_demo.py::test_demo",),
    )

    second = probes.Probe(
        name="second", replacement=probe.replacement, pytest_args=probe.pytest_args
    )
    unrelated = probes.Probe(
        name="optional",
        replacement=probe.replacement,
        pytest_args=("tests/test_optional.py::test_skipped",),
        requires=("missing_extra",),
    )
    monkeypatch.setattr(probes, "PROBES", (probe, second, unrelated))
    monkeypatch.setattr(sys, "argv", ["mutation_probes.py", "--probe", "demo", "--probe", "second"])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert command[:-1] == [sys.executable, "-m", "pytest", *probe.pytest_args, "--color=no"]
        assert command[-1].startswith("--junitxml=")
        report = ET.Element("testsuites")
        case = ET.SubElement(ET.SubElement(report, "testsuite"), "testcase")
        if len(calls) > 1 and failure_message is not None:
            ET.SubElement(case, "failure", message=failure_message)
        ET.SubElement(case, "system-out").text = stdout
        ET.ElementTree(report).write(command[-1].split("=", 1)[1], encoding="utf-8")
        if len(calls) == 1:
            # Both probes select the same node: baseline runs it only once,
            # before either mutation, against exactly the original bytes.
            assert target.read_bytes() == original
            return types.SimpleNamespace(returncode=0, stdout="1 passed in 0.01s", stderr="")
        assert target.read_bytes() == original.replace(b"x = 1", b"x = 2")
        if returncode is None:
            raise KeyboardInterrupt
        if returncode == "timeout":
            # What subprocess.run raises, after killing pytest, when a mutant
            # makes the selected test loop past the harness bound.
            raise probes.subprocess.TimeoutExpired(command, kwargs["timeout"])
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(probes.subprocess, "run", fake_run)
    if expected in ("KILLED", "SURVIVED"):
        assert probes.main() == (0 if expected == "KILLED" else 1)
        assert len(calls) == 3
        assert f"demo: {expected}" in capsys.readouterr().out
    elif expected == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            probes.main()
    else:
        with pytest.raises(RuntimeError, match=expected):
            probes.main()
    assert target.read_bytes() == original


def test_golden_snapshot_compare_rejects_missing_evidence_and_drift(tmp_path) -> None:
    golden = _load_script("docs/scripts/golden_snapshot.py")
    left = tmp_path / "left"
    right = tmp_path / "right"
    assert golden.compare(left, right) == 1
    for directory in (left, right):
        directory.write_text("{}", encoding="utf-8")
    assert golden.compare(left, right) == 1
    left.unlink()
    right.unlink()
    left.mkdir()
    right.mkdir()
    assert golden.compare(left, right) == 1
    for directory in (left, right):
        (directory / "notes.txt").write_text("No JSON evidence", encoding="utf-8")
    assert golden.compare(left, right) == 1
    payload = {"outer": {"value": 1}}
    for directory in (left, right):
        (directory / "artifact.json").write_text(json.dumps(payload), encoding="utf-8")

    # Matching nonempty subsets of the generated inventory are supported.
    assert golden.compare(left, right) == 0

    (right / "artifact.json").write_text(json.dumps({"outer": {"value": 2}}), encoding="utf-8")
    assert golden.compare(left, right) == 1

    (right / "artifact.json").write_text(json.dumps(payload), encoding="utf-8")
    (right / "extra.json").write_text("{}", encoding="utf-8")
    assert golden.compare(left, right) == 1


@pytest.mark.parametrize("left,right,equal", [
    (float("nan"), float("nan"), True),
    (1, 1.0, False),
])
def test_golden_snapshot_scalar_comparison_preserves_types(left, right, equal) -> None:
    golden = _load_script("docs/scripts/golden_snapshot.py")
    assert golden._values_equal(left, right) is equal
