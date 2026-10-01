"""Fresh-process direct writer audit, including a swallowed-import control."""

import argparse

import builtins
import importlib.abc
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def child(poison=False):
    """Exercise direct QASM writing while poisoning SDK imports and native lowering."""
    attempts = []
    prefixes = ("qiskit", "qiskit_aer", "qiskit_qasm3_import", "nwqlib.backends")

    def forbidden(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)

    def record(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("blocked direct-writer import: " + name)

    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        record(name)
        for item in fromlist or ():
            record(name + "." + item)
        return original(name, globals, locals, fromlist, level)

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            record(fullname)
            return None

    # Intercept both import entry points and record attempts before raising.
    # The final assertion must also catch forbidden imports swallowed by callers.
    builtins.__import__ = guarded
    sys.meta_path.insert(0, Blocker())
    if poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass
    from nwqlib.io import QasmWriteBudget, write_qasm3
    import nwqlib
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.blocks import lowering
    def forbidden_lowering(*args, **kwargs):
        raise AssertionError("direct writer called native lowering")
    lowering.lower_qiskit = lowering._lower_qiskit = forbidden_lowering
    # The physical vector (2,2) has normalized direction |+>, prepared by one H.
    plan = nwqlib.plan(
        Expectation(state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)),
        method=ExpectationMethod(), seed=7,
    )
    budget = QasmWriteBudget(max_metadata_bytes=1000000, max_walk_steps=100000,
                             max_qubits=1, max_clbits=0, max_bytes=1024, max_instructions=10, max_chunk_bytes=16)
    sink = BytesIO()
    receipt = write_qasm3(plan.construction, sink, budget=budget)
    assert receipt.expanded_operations == 1 and b"h a0;" in sink.getvalue()
    loaded = [name for name in sys.modules if forbidden(name)]
    if attempts or loaded:
        raise AssertionError(f"SDK isolation violation: attempts={attempts}, loaded={loaded}")
    print("SDK-free public IO import and actual M1 direct writing passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--child", action="store_true", help="run the isolated audit directly")
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; requires --child")
    args = parser.parse_args(argv)
    if args.poison and not args.child:
        parser.error("--poison requires --child")
    if args.child:
        child(args.poison)
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    clean = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if clean.returncode:
        raise RuntimeError(clean.stdout + clean.stderr)
    print(clean.stdout, end="")
    poisoned = subprocess.run(command + ["--poison"], cwd=ROOT, env=env, text=True, capture_output=True)
    if poisoned.returncode == 0 or "attempts=['qiskit']" not in poisoned.stderr:
        raise AssertionError("caught-import control failed:\n" + poisoned.stderr)
    print("Caught blocked import control rejected as required.")


if __name__ == "__main__":
    main()
