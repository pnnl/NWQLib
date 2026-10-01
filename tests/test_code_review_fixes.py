"""Tests for scientific maintenance command behavior."""

from __future__ import annotations

import json
import re
import sys
import tomllib
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from conftest import load_docs_script as _load_script

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_probe_module(tmp_path, monkeypatch, probes, text: str) -> Path:
    target = tmp_path / "src" / "nwqlib" / "demo.py"
    target.parent.mkdir(parents=True)
    target.write_text(text, encoding="utf-8")
    monkeypatch.setattr(probes, "ROOT", tmp_path)
    return target


@pytest.mark.parametrize(
    ("source", "unresolved"),
    [("def mutate():\n    x = 1\n", "x = 9"),
     ("def mutate():\r\n    x = 1\r\n    return x\r\n", "x = 1\n    return x")],
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
        (4, "test not found", (), ""),
        (0, "1 skipped in 0.01s", (), "<testcase><skipped/></testcase>"),
        (0, "1 passed, 1 skipped in 0.01s", (), "<testcase/><testcase><skipped/></testcase>"),
        (1, "ModuleNotFoundError", ("missing_extra",), ""),
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


def _distribution_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_mutation_workflow_installs_every_probe_requirement() -> None:
    # The hosted campaign stops at its baseline when a probe's import is missing,
    # which a local environment with every extra cannot show.
    probes = _load_script("docs/scripts/mutation_probes.py")
    required = {module for probe in probes.PROBES for module in probe.requires}
    workflow = (REPO_ROOT / ".github" / "workflows" / "mutation-probes.yml").read_text(encoding="utf-8")
    (install,) = [line for line in workflow.splitlines() if "pip install" in line and '-e ".[' in line]
    extras = re.search(r'-e "\.\[([^\]]*)\]"', install).group(1).split(",")
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    specs = [*project["dependencies"], *(spec for extra in extras for spec in project["optional-dependencies"][extra])]
    installed = {_distribution_key(re.match(r"[A-Za-z0-9_.-]+", spec).group(0)) for spec in specs}
    # Direct-URL requirements are installed from their locked line, as in Full CI.
    installed |= {_distribution_key(name) for name in re.findall(r"/\^([A-Za-z0-9_.-]+) @ /p", install)}
    # Each required module is imported under its distribution's name.
    missing = sorted(module for module in required if _distribution_key(module) not in installed)
    assert not missing, f"mutation-probes.yml does not install {missing}"


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
        (
            1,
            "Captured stdout\n[XPASS(strict)] only application text\n"
            "FAILED tests/test_demo.py::test_demo\n1 failed in 0.01s",
            "assert False",
            "KILLED",
        ),
        (0, "1 passed in 0.01s", None, "SURVIVED"),
        (
            1,
            "FAILED tests/test_demo.py::test_demo\n1 failed in 0.01s\n"
            " ** On entry to DLASCL, parameter number  4 had an illegal value",
            "assert False",
            "KILLED",
        ),
        (0, "1 passed in 0.01s\n ** On entry to DLASCL, parameter number  4 had an illegal value", None, "SURVIVED"),
        (
            1,
            "[XPASS(strict)] mode limitation\nFAILED tests/test_demo.py::test_demo\n"
            "1 failed in 0.01s",
            "[XPASS(strict)] mode limitation",
            "unavailable",
        ),
        (1, "F [100%]\n1 failed in 0.01s", "[XPASS(strict)] mode limitation", "unavailable"),
        (1, "1 failed in 0.01s", None, "infrastructure"),
        (4, "test not found", None, "infrastructure"),
        (1, "1 error in 0.01s", None, "infrastructure"),
        (0, "1 skipped in 0.01s", None, "unavailable"),
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


def test_mutation_probe_keys_derive_segment_counts(tmp_path, monkeypatch) -> None:
    probes = _load_script("docs/scripts/mutation_probes.py")
    source = _write_probe_module(
        tmp_path,
        monkeypatch,
        probes,
        (
            "value = 1\n\n"
            "def outer():\n"
            "    value = 1\n"
            "    def target():\n"
            "        value = 1\n"
            "        value = 1\n"
            "    return value\n\n"
            "class Box:\n"
            "    def target(self):\n"
            "        value = 1\n"
        ),
    )
    replacement = probes.Replacement(
        module="nwqlib.demo",
        qualname="outer.target",
        old="value = 1",
        new="value = 2",
    )
    resolved = probes._resolve_replacement("synthetic", replacement)

    assert resolved.path == source
    assert resolved.count == 2
    mutated = probes._mutate(source.read_text(), resolved)
    assert mutated.count("value = 2") == 2
    assert mutated.splitlines()[0] == "value = 1"
    assert "class Box:\n    def target(self):\n        value = 1\n" in mutated

    monkeypatch.setattr(probes, "ROOT", REPO_ROOT)
    # Resolve the real inventory before a campaign can spend time executing
    # baselines. Resolution reads source only and never applies replacements.
    for real in probes.PROBES:
        probes._resolve_replacement(real.name, real.replacement)


def test_policy_lint_backticked_implementation_refs_report_ghosts(
    tmp_path,
) -> None:
    lint = _load_script("docs/scripts/policy_lint.py")
    docs = tmp_path / "docs"
    docs.mkdir()
    docs.joinpath("CODE_TOUR.md").write_text(
        # The leading fenced block pins span pairing: without fence
        # stripping, every inline backtick after a fence mis-pairs and
        # the ghost below silently escapes the scan.
        "```\nlayer map\n```\n\n`solve_lchs` / `_circuit`\n`solve_ghost`\n"
        "`value_selected/value_theta`\n",
        encoding="utf-8",
    )
    source = tmp_path / "src" / "nwqlib"
    source.mkdir(parents=True)
    source.joinpath("demo.py").write_text(
        "def solve_lchs():\n    return None\nCURRENT_LIMIT = 4\n# REMOVED_LIMIT was deleted.\n",
        encoding="utf-8",
    )
    docs.joinpath("ENGINEERING_CONSTANTS.md").write_text(
        "| Constant | Value |\n| --- | --- |\n"
        "| `CURRENT_LIMIT` | 4 |\n| `REMOVED_LIMIT` | 8 |\n", encoding="utf-8",
    )
    guides = docs / "algorithms"
    guides.mkdir()
    guides.joinpath("demo.md").write_text(
        "## Implementation map\n\n`demo.py` and `missing.py`\n", encoding="utf-8",
    )
    docs.joinpath("mathematics.md").write_text(
        "| `src/nwqlib/demo.py::Owner.solve_lchs` | `src/nwqlib/demo.py::solve_moved` |"
        " `src/nwqlib/gone.py::solve_lchs` |\n",
        encoding="utf-8",
    )
    lint.REPO_ROOT = tmp_path
    failures = []
    lint.check_backticked_implementation_references(failures)
    assert len(failures) == 5
    assert any("solve_ghost" in failure for failure in failures)
    assert any("missing.py" in failure for failure in failures)
    assert any("REMOVED_LIMIT" in failure for failure in failures)
    assert any("demo.py::solve_moved" in failure for failure in failures)
    assert any("gone.py::solve_lchs" in failure for failure in failures)


def test_policy_lint_internal_markdown_links_report_synthetic_failures(
    tmp_path,
    monkeypatch,
) -> None:
    """Nested broken links must fail while illustrative fenced links and nonlocal links remain
    valid.
    """
    lint = _load_script("docs/scripts/policy_lint.py")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (docs / "root.md").write_text("# Root\n", encoding="utf-8")
    (docs / "nested").mkdir()
    (docs / "nested" / "guide.md").write_text(
        "[existing](../guide.md)\n[missing](missing.md)\n"
        "```markdown\n[illustrative](not-a-real-file.md)\n```\n", encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        '[relative](docs/guide.md "Guide title")\n'
        "[root](/docs/root.md#section)\n"
        "[web](https://example.com/path)\n"
        "[mail](mailto:maintainer@example.com)\n"
        "[same page](#section)\n"
        "[archived paper](docs/missing.PdF)\n"
        "[missing](docs/missing.md)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(lint, "REPO_ROOT", tmp_path)

    failures = []
    lint.check_internal_markdown_links_resolve(failures)
    assert len(failures) == 2
    assert any("docs/missing.md" in failure for failure in failures)
    assert any("docs/nested/guide.md" in failure for failure in failures)

    (docs / "nested" / "math.md").write_text(
        "```math\n0<\\beta<1,\\quad x\\lt y\n```\n\n```math\n\\sum_{i<j}x_i\n```\n", encoding="utf-8",
    )
    failures = []
    lint.check_math_fences_open_no_html_tags(failures)
    assert len(failures) == 1
    assert failures[0].startswith("docs/nested/math.md:5:")


def test_policy_lint_rendered_api_and_links(tmp_path, monkeypatch) -> None:
    lint = _load_script("docs/scripts/policy_lint.py")
    monkeypatch.setattr(lint, "REPO_ROOT", tmp_path)
    api = tmp_path / "docs" / "api"
    api.mkdir(parents=True)
    api.joinpath("method.md").write_text("::: demo.method\n", encoding="utf-8")
    api.joinpath("operation.md").write_text("::: demo.solve\n", encoding="utf-8")
    site = tmp_path / "site"
    method = site / "api" / "method" / "index.html"
    method.parent.mkdir(parents=True)
    operation = site / "api" / "operation" / "index.html"
    operation.parent.mkdir(parents=True)
    source_only = tmp_path / "docs" / "excluded.md"
    source_only.write_text("# Not published\n", encoding="utf-8")
    operation.write_text(
        '<article><div id="demo.solve" class="doc doc-function">solve(x)</div></article>',
        encoding="utf-8",
    )
    good = (
        '<nav><a href="theme-only.html">Navigation is outside the article check</a></nav>'
        '<article><h1 id="demo.method">Method</h1><span class>Empty class attribute</span>'
        '<div id="demo.method.Method" class="doc doc-class">Method(x)</div>'
        '<a href="../operation/#demo.solve">Solve</a>'
        '<a href="#demo.method.Method">Local class</a>'
        '<a href="https://example.com/">External documentation</a></article>'
    )
    method.write_text(good, encoding="utf-8")
    failures = []
    lint.check_rendered_documentation(failures, site)
    assert failures == []

    # A source link can exist even though the deployed page is absent. A module
    # heading can render successfully while lazy exports hide every signature.
    method.write_text(
        '<article><h1 id="demo.method">Method</h1><p>Module list only</p>'
        '<a href="../../excluded/">Excluded source</a>'
        '<a href="../operation/#demo.missing">Missing anchor</a></article>', encoding="utf-8",
    )
    failures = []
    lint.check_rendered_documentation(failures, site)
    assert len(failures) == 3
    assert any("no class or function" in failure for failure in failures)
    assert any("no rendered target" in failure for failure in failures)
    assert any("no rendered anchor" in failure for failure in failures)

    # An unrelated documented function must not stand in for the requested one.
    method.write_text(good, encoding="utf-8")
    operation.write_text(
        '<article><div id="demo.other" class="doc doc-function">other(x)</div></article>',
        encoding="utf-8",
    )
    failures = []
    lint.check_rendered_documentation(failures, site)
    assert any("API entry 'demo.solve' was not rendered" in failure for failure in failures)


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


def test_golden_snapshot_generation_preserves_inventory_and_values(tmp_path, monkeypatch) -> None:
    golden = _load_script("docs/scripts/golden_snapshot.py")
    schemas = golden._method_schemas()

    # Exercise file generation without repeating the scientific builder
    # workloads. Those builders keep their separate baseline/after receipts.
    payloads = {name: {"artifact": name, "values": [1, None, 0.25]} for name in golden.ARTIFACT_BUILDERS}
    monkeypatch.setattr(golden, "ARTIFACT_BUILDERS", {
        name: (lambda payload=payload: payload) for name, payload in payloads.items()
    })
    assert golden.generate(tmp_path) == 0
    expected = {**payloads, **{f"methods/{name}.json": value for name, value in schemas.items()}}
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*.json")} == set(expected)
    for name, value in expected.items():
        assert json.loads((tmp_path / name).read_text(encoding="utf-8")) == value

    # Golden clock determinism needs no native experiment: use the same timer
    # producer with a two-tick fixture and no backend call.
    from nwqlib import _prepared_execution as execution
    from nwqlib.execution import ConsumptionEvent
    def timed_event():
        started = execution.perf_counter()
        return ConsumptionEvent(attempt=str(execution.uuid4()), prepared_id="sha256:" + "a" * 64,
            status="uncertain", submission=str(execution.uuid4()), shots=0, evaluations=1, started=execution._now(),
            timing=execution._native_timing(started, "synthetic no-work clock fixture")).model_dump(mode="json")
    monkeypatch.setattr(golden, "ARTIFACT_BUILDERS", {"telemetry.json": timed_event})
    monkeypatch.setattr(golden, "_method_schemas", lambda: {})
    first, second = tmp_path / "clock-a", tmp_path / "clock-b"
    assert golden.generate(first) == golden.generate(second) == 0
    assert (first / "telemetry.json").read_bytes() == (second / "telemetry.json").read_bytes()
    assert json.loads((first / "telemetry.json").read_text())["timing"]["seconds"] == 1.


@pytest.mark.parametrize("left,right,equal", [
    (float("nan"), float("nan"), True),
    (float("nan"), 0.0, False),
    (1, 1.0, False),
    (True, 1, False),
    (0.25, 0.25, True),
])
def test_golden_snapshot_scalar_comparison_preserves_types(left, right, equal) -> None:
    golden = _load_script("docs/scripts/golden_snapshot.py")
    assert golden._values_equal(left, right) is equal


def test_public_dataclass_docstrings_require_sections_and_preserve_qualified_owners(tmp_path):
    lint = _load_script('docs/scripts/policy_lint.py')
    root = tmp_path/'src'/'nwqlib'
    root.mkdir(parents=True)
    root.joinpath('first.py').write_text('''from dataclasses import dataclass
from typing import ClassVar
@dataclass
class Base:
    """Base.

    Attributes:
        inherited: Original inherited meaning.
    """
    inherited: int
@dataclass
class Shared(Base):
    """Child.

    Attributes:
        inherited: Inherited field may be described without inventing a local one.
        local: Actual local value.
    """
    local: int
    cache: ClassVar[int] = 0
    _private: int = 0
@dataclass
class NoSection:
    """Missing its field section."""
    absent: int
@dataclass
class ArgsSection:
    """An Args heading documents fields too.

    Args:
        kept: Documented field.
    """
    kept: int
    dropped: int
''')
    root.joinpath('second.py').write_text('''from dataclasses import dataclass
@dataclass
class Shared:
    """Distinct same-named owner.

    Attributes:
        value: Local meaning.
        value: Duplicate description must not be collapsed.
        inherited: Belongs only to another module's class.
    """
    value: int
''')
    lint.REPO_ROOT = tmp_path
    lint.PUBLIC_DATACLASS_OWNERS = {'nwqlib.first': ('Base', 'Shared', 'NoSection', 'ArgsSection'),
                                    'nwqlib.second': ('Shared',)}
    failures = []
    lint.check_record_docstring_fields(failures)
    assert len(failures) == 4
    assert any('NoSection' in x and "'absent'" in x and 'missing' in x for x in failures)
    assert any('ArgsSection' in x and "'dropped'" in x and 'missing' in x for x in failures)
    assert any('nwqlib.second.Shared' in x and "'value'" in x and 'duplicate' in x for x in failures)
    assert any('nwqlib.second.Shared' in x and "'inherited'" in x and 'unknown' in x for x in failures)


def test_inaccessible_reference_rule_preserves_actual_plan_job_and_principal_angles(tmp_path):
    lint = _load_script('docs/scripts/policy_lint.py')
    lint.REPO_ROOT = tmp_path
    docs = tmp_path/'docs'
    docs.mkdir()
    legal = "The plan's branch-diagonal phases; backend job 7; principal-radian branch."
    path = docs/'example.md'
    path.write_text(legal)
    failures = []
    lint.check_inaccessible_references(failures)
    assert failures == []
    path.write_text('Job' + ' 17 sets the constant.')
    lint.check_inaccessible_references(failures)
    assert len(failures) == 1 and 'example.md:1' in failures[0]
    site = tmp_path/'site'
    site.mkdir()
    page = site/'index.html'
    page.write_text('<article>'+legal+'</article>')
    failures = []
    lint.check_rendered_documentation(failures, site)
    assert failures == []
    page.write_text('<article>Job' + ' <strong>17</strong> sets the constant.</article>')
    lint.check_rendered_documentation(failures, site)
    assert len(failures) == 1 and 'inaccessible' in failures[0]
