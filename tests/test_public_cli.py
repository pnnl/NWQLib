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


def test_options_uses_actual_required_scientific_configuration_without_instantiation(capsys):
    from nwqlib.cli import main

    assert main(["options", "adapt_gcim"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert {"initial_state", "pool"} <= set(schema["required"])
    assert "theta" in schema["properties"]
    assert main(["options", "fixed_gcim"]) == 0
    assert "basis" in json.loads(capsys.readouterr().out)["required"]


def test_cli_module_entry_agrees_with_package_inventory_and_help(monkeypatch, capsys):
    from nwqlib import cli as module

    package = cli("algorithms", "--json")
    direct = subprocess.run([sys.executable, "-m", "nwqlib.cli", "algorithms", "--json"],
                            cwd=ROOT, text=True, capture_output=True)
    assert package.returncode == direct.returncode == 0
    assert json.loads(package.stdout) == json.loads(direct.stdout)
    monkeypatch.setattr(module, "_registrations", lambda *args: pytest.fail("help discovered methods"))
    for command, description in (("card", "declared scope"), ("options", "configuration schema")):
        with pytest.raises(SystemExit) as stopped:
            module.main([command, "--help"])
        assert stopped.value.code == 0
        assert description in capsys.readouterr().out


def test_actual_external_check_method_cli_and_zero_acquisition_falsifier():
    completed = cli("check-method", "_hadamard_method:case")
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "CONFORMANT"
    assert report["method"] == "example.hadamard_pauli_expectation"
    assert "not a general" in report["qualification"]


def test_check_method_propagates_factory_attribute_errors(monkeypatch):
    import types
    from nwqlib.cli import main

    # An AttributeError raised inside an existing factory is not a lookup error.
    def broken_case():
        raise AttributeError("internal author attribute")

    monkeypatch.setitem(sys.modules, "broken_author_module", types.SimpleNamespace(case=broken_case))
    with pytest.raises(AttributeError, match="internal author attribute"):
        main(["check-method", "broken_author_module:case"])


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


def test_author_checker_preserves_array_result_and_same_plan_falsifier():
    import numpy as np
    import nwqlib
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.authoring import MethodCase
    from nwqlib.problems.inputs import ingest_product

    # t=0 and zero physical input give an exact, independent two-entry oracle.
    original = MethodCase(
        method=LCHS(),
        problem=nwqlib.LinearDynamics(A=np.eye(2), initial_state=ingest_product([[0., 0.]]), time=0),
        execution="classical", evaluate=nwqlib.solve,
        accepts=lambda result: len(result.data.artifacts) == 1 and bool(np.array_equal(result.solution, np.zeros(2))),
        invalid_result=lambda result: result.revise(construction_id="sha256:" + "0" * 64),
    )
    assert check_method(original)["status"] == "CONFORMANT"
    with pytest.raises(ValueError, match="expected relation failed"):
        check_method(replace(original, accepts=lambda result: bool(np.array_equal(result.solution, np.ones(2)))))
