#!/usr/bin/env python3
"""Fresh actual QPE Plan/host/injected-count/report SDK-attempt audit."""

import argparse

import builtins
import importlib.abc
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))


def main(argv=None):
    """Exercise four concrete classical QPE archives with injected spectral data, then
    save and load one exact trajectory and one pooled sampled quantum selection, with
    no native acquisitions.
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
        return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)

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
    from nwqlib.algorithms.qpe import numerical
    from nwqlib import solve
    from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan
    from test_qpe_selected import make_plan

    calls = []

    def diagonal_fixture(matrix):
        calls.append(1)
        return np.diag(matrix).copy(), np.eye(2, dtype=complex)

    # Use the known diagonal spectrum to isolate archive/reanalysis behavior.
    # The call counter detects an unintended repeat of spectral setup.
    with patch.object(numerical, "diagonalize_hermitian", diagonal_fixture):
        for estimator in ("qcels", "spe", "rfe", "rwpe"):
            selected = make_plan(estimator=estimator)
            with TemporaryDirectory() as directory:
                saved = save_plan(selected, ArchiveFiles(directory, 4_000_000))
                with patch.object(type(selected.method), "plan", side_effect=AssertionError("archive replanned")):
                    restored = load_plan(saved, ArchiveFiles(directory, 4_000_000))
                assert restored == selected
            result = solve(restored)
            before = len(calls)
            assert result.report()["plan"]["execution"] == "classical"
            assert result.analyze().estimator_value == result.estimator_value
            assert len(calls) == before
    # An exact quantum archive keeps its one trajectory Experiment, whose
    # points carry every query.
    selected = make_plan(execution="quantum")
    assert [e.name for e in selected.experiments] == ["trajectory"]
    assert {q.acquisition for q in selected.reconstruction.queries} == {"trajectory"}
    with TemporaryDirectory() as directory:
        saved = save_plan(selected, ArchiveFiles(directory, 4_000_000))
        assert saved["selected"]["format"] == "qpe/qcels/7"
        restored = load_plan(saved, ArchiveFiles(directory, 4_000_000))
        assert restored == selected
    # A sampled SPE archive keeps pooled draws: one query per setting whose
    # counts batch requests shots times its draw multiplicity.
    selected = make_plan(estimator="spe", execution="quantum", shots=3)
    queries = selected.reconstruction.queries
    nodes = {d.id: d.node for d in selected.construction.program.definitions}
    assert sum(q.multiplicity for q in queries) == 2 * selected.method.num_samples
    assert any(q.multiplicity > 1 for q in queries)
    assert len({(q.power, q.phase_shift) for q in queries}) == len(queries)
    assert [nodes[e.batch].repetitions for e in selected.experiments] == [3 * q.multiplicity for q in queries]
    with TemporaryDirectory() as directory:
        saved = save_plan(selected, ArchiveFiles(directory, 4_000_000))
        assert saved["selected"]["format"] == "qpe/spe/7"
        restored = load_plan(saved, ArchiveFiles(directory, 4_000_000))
        assert restored == selected
    loaded = [name for name in sys.modules if forbidden(name)]
    assert not (attempts or loaded), (
        f"SDK isolation violation: attempts={attempts}, loaded={loaded}"
    )
    print(
        "Four classical QPE archives with injected host execution, one exact trajectory and one pooled sampled "
        "native selection: attempts=[], loaded=[]"
    )


if __name__ == "__main__":
    main()
