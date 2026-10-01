#!/usr/bin/env python3
"""Fresh explicit QLS classical-model and archive SDK-attempt audit."""

import argparse

import builtins
import importlib.abc
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))


def main(argv=None):
    """Exercise three bounded host QLS models and their saved Plans without SDK imports or
    repeated analysis actions.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; this audit must fail")
    args = parser.parse_args(argv)
    prefixes = (
        "qiskit",
        "qiskit_aer",
        "qiskit_ibm_runtime",
        "qiskit_ionq",
        "cirq",
        "nwqlib.backends.qiskit_aer",
    )
    attempts = []

    def forbidden(name):
        return any(name == p or name.startswith(p + ".") for p in prefixes)

    def record(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("blocked SDK import: " + name)

    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        record(name)
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
    if args.poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass
    import numpy as np
    from unittest.mock import patch
    import nwqlib
    from nwqlib import LinearSystem, Solution, StateVector
    from nwqlib.algorithms.qls import QLS
    from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan

    for solver in ("qsvt_inverse", "shortcut_native_svp", "shortcut_dilation"):
        problem = LinearSystem(A=np.diag([1.0, -2.0]), b=np.array([1.0, 1j]))
        inverse = solver == "qsvt_inverse"
        output = (
            Solution()
            if inverse
            else StateVector(normalization="unit", global_phase="modulo_global_phase")
        )
        method = QLS(
            solver=solver,
            block_encoding_implementation="dense_dilation",
            encoded_solution_norm_estimate=None if inverse else 1.5,
        )
        selected = nwqlib.plan(problem, method=method, output=output, execution="classical", seed=7)
        with TemporaryDirectory() as directory:
            saved = save_plan(selected, ArchiveFiles(directory, 4_000_000))
            with patch.object(QLS, "plan", side_effect=AssertionError("archive replanned")):
                restored = load_plan(saved, ArchiveFiles(directory, 4_000_000))
            assert restored.content_id == selected.content_id
        result = nwqlib.solve(restored)
        with (
            patch("numpy.linalg.eigh", side_effect=AssertionError("analysis reran action")),
            patch("numpy.linalg.svd", side_effect=AssertionError("analysis reran action")),
        ):
            assert result.report()["plan"]["execution"] == "classical"
            np.testing.assert_array_equal(result.analyze().value, result.value)
    loaded = [n for n in sys.modules if forbidden(n)]
    assert not (attempts or loaded), (
        f"SDK isolation violation: attempts={attempts}, loaded={loaded}"
    )
    print(
        "Three explicit dense-dilation QLS classical models: actual saved Plans/results; attempts=[], loaded=[]"
    )


if __name__ == "__main__":
    main()
