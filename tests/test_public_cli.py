"""Actual discovery and explicitly selected author checks at the public CLI."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nwqlib.algorithms.authoring import check_method
from author_method_fixture import case

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, cwd=None):
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests")))
    )
    return subprocess.run(
        [sys.executable, "-m", "nwqlib", *args],
        cwd=cwd or ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_actual_external_check_method_cli_and_zero_acquisition_falsifier():
    completed = cli("check-method", "_hadamard_method:case")
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "CONFORMANT"
    assert report["method"] == "example.hadamard_pauli_expectation"


def test_author_checker_rejects_false_oracle_ineffective_falsifier_and_disabled_pair_check(
    monkeypatch,
):
    from nwqlib.algorithms.expectation import ExpectationAnalysis

    original = case()
    with pytest.raises(ValueError, match="expected relation failed"):
        check_method(replace(original, accepts=lambda result: result.value == 3.0))
    with pytest.raises(ValueError, match="alter a scientific field"):
        check_method(replace(original, invalid_result=lambda result: result))
    validate = ExpectationAnalysis.validate_plan
    monkeypatch.setattr(ExpectationAnalysis, "validate_plan", lambda *args: None)
    with pytest.raises(ValueError, match="invalid scientific pair was accepted"):
        check_method(original)
    monkeypatch.setattr(ExpectationAnalysis, "validate_plan", validate)
    assert check_method(original)["status"] == "CONFORMANT"


def test_console_script_and_module_list_every_registered_method():
    from nwqlib import methods

    script = Path(sys.executable).with_name("nwqlib")
    if not script.exists():
        pytest.skip("the nwqlib console script exists only in an installed environment")
    module = cli("algorithms")
    installed = subprocess.run(
        [str(script), "algorithms"],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests")))),
        text=True,
        capture_output=True,
        check=False,
    )
    assert module.returncode == installed.returncode == 0, module.stderr + installed.stderr
    assert installed.stdout == module.stdout
    # Each listed Method is printed as name@version before its outputs.
    names = (line.split(":", 1)[0] for line in module.stdout.splitlines())
    listed = {tuple(name.rsplit("@", 1)) for name in names if "@" in name}
    assert listed == {(row.source.name, row.source.version) for row in methods()}


def test_options_and_card_match_each_registered_method_constructor(capsys):
    import pydantic

    from nwqlib import methods
    from nwqlib.algorithms import AlgorithmRegistry
    from nwqlib.cli import main
    from nwqlib.operators import ingest_pauli

    registrations = methods()
    registry = AlgorithmRegistry(registrations)
    assert main(["algorithms", "--json"]) == 0
    cards = json.loads(capsys.readouterr().out)
    assert {(card["source"]["name"], card["source"]["version"]) for card in cards} == {
        (row.source.name, row.source.version) for row in registrations
    }
    # The required inputs of the documented examples of the three Methods that have any.
    supplied = {
        "adapt_gcim": dict(initial_state=[1, 1], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),)),
        "fixed_gcim": dict(basis=([1, 1], [1, 1j])),
        "spe": dict(initial_state=[1, 0], overlap_lower_bound=1.0),
    }
    for row in registrations:
        name, version = row.source.name, row.source.version
        assert main(["options", name, "--version", version]) == 0
        schema = json.loads(capsys.readouterr().out)
        # The required fields `options` reports are the ones a Method constructed without configuration lacks.
        try:
            registry.resolve(row.source)
            missing = set()
        except pydantic.ValidationError as error:
            missing = {item["loc"][0] for item in error.errors() if item["type"] == "missing"}
        assert set(schema.get("required", ())) == missing, name
        method = registry.resolve(row.source, **supplied.get(name, {}))
        assert main(["card", name, "--version", version]) == 0
        card = json.loads(capsys.readouterr().out)
        assert tuple(card["descriptor"]["output_families"]) == tuple(method.descriptor.output_families)


def test_third_party_listing_reads_entry_points_without_importing_the_extension(tmp_path):
    # An installed distribution declares one Method by entry point. Importing its module writes a marker file.
    marker = tmp_path / "imported"
    metadata = tmp_path / "toyext-0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: toyext\nVersion: 0.1\n")
    (metadata / "entry_points.txt").write_text("[nwqlib.algorithms]\ntoy_method@1 = toyext:factory\n")
    (tmp_path / "toyext.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n\n"
        "def factory(**configuration):\n    raise AssertionError('factory called')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(ROOT / "src"))))

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "nwqlib", *args],
            cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
        )

    built_in, listed = run("algorithms"), run("algorithms", "--third-party")
    cards, options = run("algorithms", "--json", "--third-party"), run("options", "toy_method", "--third-party")
    assert built_in.returncode == listed.returncode == cards.returncode == 0
    assert "toy_method@1" in {line.split(":", 1)[0] for line in listed.stdout.splitlines()}
    assert "toy_method@1" not in {line.split(":", 1)[0] for line in built_in.stdout.splitlines()}
    (card,) = (card for card in json.loads(cards.stdout) if card["source"]["name"] == "toy_method")
    assert card["descriptor"] is None and card["options_available"] is False
    # A configuration schema needs the extension's code, so the CLI reports it unavailable with exit status 1.
    assert options.returncode == 1
    assert not marker.exists()
