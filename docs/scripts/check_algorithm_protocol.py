#!/usr/bin/env python3
"""Fresh-process discovery, schema and selected-plan import-attempt witness."""

import argparse

import builtins
import importlib.abc
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))


def child(poison=None):
    """Check inert discovery and explicit Method selection against forbidden import/construction
    attempts.

    The external Method is the test suite's reference Hadamard Method in tests/.
    """
    blocked = (
        "qiskit",
        "qiskit_aer",
        "qiskit_ibm_runtime",
        "qiskit_ionq",
        "pyscf",
        "openfermion",
        "scikit_tt",
        "qulacs",
        "qnexus",
        "pytket",
        "unrelated_fixture",
    )
    attempts, selected = [], False

    def record(name):
        if (
            any(name == p or name.startswith(p + ".") for p in blocked)
            or name == "_hadamard_method"
            and not selected
        ):
            attempts.append(name)
            raise ImportError("forbidden import: " + name)

    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        record(name)
        if not level:
            for member in fromlist or ():
                record(name + "." + member)
        return original(name, globals, locals, fromlist, level)

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            record(fullname)

    # Intercept both import entry points and record attempts before raising.
    # The final assertion must also catch forbidden imports swallowed by callers.
    builtins.__import__ = guarded
    sys.meta_path.insert(0, Blocker())
    if poison:
        try:
            __import__(poison)
        except ImportError:
            pass
    from importlib.metadata import EntryPoint
    from nwqlib import methods
    from nwqlib.algorithms import _METHOD_OWNERS
    from nwqlib.algorithms import (
        AlgorithmRegistry,
        direct_method,
        options_schema,
        third_party_registrations,
    )
    from _hadamard_method_metadata import REGISTRATION
    from nwqlib.cli import main as cli

    # Builtin configuration modules may load; constructors and factories do not.
    from nwqlib.algorithms.protocol import Method

    original_init = Method.__init__

    def no_instance(*args, **kwargs):
        raise AssertionError("discovery constructed a Method with guessed scientific inputs")

    Method.__init__ = no_instance
    try:
        cli(["algorithms"])
        # The documented public discovery entry returns one registration per
        # builtin Method owner.
        rows = methods()
        assert _METHOD_OWNERS and sorted(row.method_type.__name__ for row in rows) == sorted(
            _METHOD_OWNERS
        ), rows
        for row in rows:
            options_schema(row)
    finally:
        Method.__init__ = original_init
    unrelated = third_party_registrations(
        (
            EntryPoint(
                name="unrelated@1", value="unrelated_fixture:factory", group="nwqlib.algorithms"
            ),
        )
    )
    registry = AlgorithmRegistry((REGISTRATION,) + unrelated)
    registry.discover()
    assert "_hadamard_method" not in sys.modules
    selected = True
    method = registry.resolve(REGISTRATION.source)
    assert direct_method(method) is method
    from nwqlib import Expectation, plan, estimate
    from nwqlib.operators import ingest_pauli

    chosen = plan(
        Expectation(state=(1.0, 1j), observable=ingest_pauli((("Y", 1.0),), num_qubits=1)),
        method=method,
        seed=7,
    )
    assert estimate(chosen).construction_id == chosen.construction.content_id
    loaded = [
        name for name in sys.modules if any(name == p or name.startswith(p + ".") for p in blocked)
    ]
    if attempts or loaded:
        raise AssertionError(f"import isolation violation: attempts={attempts}, loaded={loaded}")
    print(
        "Builtin schemas, inert external discovery and explicit Hadamard planning: zero SDK/provider imports."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--child", action="store_true", help="run the isolated audit directly")
    parser.add_argument("--poison", metavar="MODULE", help="inject this forbidden import; requires --child")
    args = parser.parse_args(argv)
    if args.poison and not args.child:
        parser.error("--poison requires --child")
    if args.child:
        child(args.poison)
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    for poison in (None, "qiskit", "unrelated_fixture", "_hadamard_method"):
        result = subprocess.run(
            command + (["--poison", poison] if poison else []),
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        if poison is None:
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            print(result.stdout, end="")
        elif result.returncode == 0 or f"attempts=['{poison}']" not in result.stderr:
            raise AssertionError("caught-import negative control failed:\n" + result.stderr)
    print("Caught-import negative controls rejected as required.")


if __name__ == "__main__":
    main()
