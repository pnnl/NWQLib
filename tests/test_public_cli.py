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
    for arguments in (
        ["card", "not-a-registered-method"],
        ["card", "chebyshev_lanczos", "--version", "999"],
    ):
        with pytest.raises(SystemExit) as absent:
            main(arguments)
        assert absent.value.code == 1 and "one known method/version" in capsys.readouterr().err


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


AUDIT_SCRIPTS = (
    "check_core_records", "check_prepared_records", "check_qasm_records", "check_profile_records",
    "check_algorithm_protocol", "check_scientist_records", "check_adapt_records", "check_df_records",
    "check_lchs_records", "check_ordered_operator_records", "check_qhd_records", "check_qpe_records",
    "check_qls_records",
)


@pytest.mark.parametrize("name", AUDIT_SCRIPTS)
def test_audit_help_and_typo_stop_before_work(name, monkeypatch, capsys):
    import builtins
    from importlib import import_module

    module = import_module("docs.scripts." + name)
    original = builtins.__import__

    def guarded(module_name, *args, **kwargs):
        if module_name.split(".")[0] in {"nwqlib", "numpy", "scipy", "qiskit"}:
            pytest.fail("help or invalid flag entered scientific audit")
        return original(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("help launched an audit child"))
    if hasattr(module, "child"):
        monkeypatch.setattr(module, "child", lambda *a, **k: pytest.fail("help entered audit"))
    for argument, code in (("--help", 0), ("--pois", 2)):
        with pytest.raises(SystemExit) as stopped:
            module.main([argument])
        assert stopped.value.code == code
        output = capsys.readouterr()
        assert "usage:" in (output.out if code == 0 else output.err)


def test_actual_external_check_method_cli_and_zero_acquisition_falsifier():
    completed = cli("check-method", "_hadamard_method:case")
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "CONFORMANT"
    assert report["method"] == "example.hadamard_pauli_expectation"
    assert "not a general" in report["qualification"]


def test_check_method_factory_lookup_separates_missing_names_from_factory_errors(monkeypatch, capsys):
    import types
    import nwqlib
    import nwqlib.algorithms.authoring as authoring
    from nwqlib.cli import main

    def forbidden(*args, **kwargs):
        pytest.fail("a missing factory name reached scientific work")

    for owner, name in ((authoring, "check_method"), (nwqlib, "plan"), (nwqlib, "solve")):
        monkeypatch.setattr(owner, name, forbidden)
    module = "_hadamard_method"
    with pytest.raises(SystemExit) as stopped:
        main(["check-method", f"{module}:misspelled_case"])
    error = capsys.readouterr().err
    assert stopped.value.code == 1 and module in error and "misspelled_case" in error
    assert "Traceback" not in error
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
