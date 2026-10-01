#!/usr/bin/env python3
"""Actual SDK-free ADAPT planning, including the default fermionic compiler.

No acquisition, projected solver or injected planning implementation is used.
The --poison witness must fail even if a blocked SDK import is caught.
"""

import argparse

import builtins
import importlib.abc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main(argv=None):
    """Exercise actual ADAPT planning while recording even caught forbidden SDK import attempts."""
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
            raise ImportError("blocked SDK/native import: " + name)

    # Intercept both import entry points and record attempts before raising.
    # The final assertion must also catch forbidden imports swallowed by callers.
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        record(name)
        for member in fromlist or ():
            record(name + "." + member)
        return original(name, globals, locals, fromlist, level)

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            record(fullname)
            return None

    builtins.__import__ = guarded
    sys.meta_path.insert(0, Blocker())
    if args.poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass
    import numpy as np
    import scipy.sparse as sps
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import ingest_occupation
    from nwqlib.algorithms.gcim.adapt_acquisition import _point

    generator = ingest_pauli((("Y", 1j),), num_qubits=1)
    for sparse in (False, True):
        matrix = np.array([[0.5, 0.25], [0.25, -0.5]], complex)
        problem = Eigenproblem(A=sps.csr_matrix(matrix) if sparse else matrix)
        for shots in (None, 32):
            selected = plan(
                problem, method=ADAPT(initial_state=[1, 0], pool=(generator,),
                                     input_conversion="dense_pauli" if sparse else "auto"),
                shots=shots, seed=7
            )
            assert tuple((t.label, t.coefficient) for t in selected.reconstruction.terms) == (
                ("X", 0.25),
                ("Z", 0.5),
            )
            observation = _point(selected, "screen").resolved_observation(selected)[1]
            if shots is None:
                # The exact shared full-chain query: its diagonal reduction and
                # its screening labels at one boundary.
                assert [p.id for p in observation.positions] == ["diagonal", "screen"]
                assert observation.positions[1].labels == ("X", "Z")
            else:
                assert observation.labels == ()
            selected.model_dump_json()
    reference = ingest_occupation((1, 1, 0, 0), num_qubits=4)
    selected = plan(
        Eigenproblem(A=ingest_pauli((("IIIZ", 1.0),), num_qubits=4)),
        method=ADAPT(initial_state=reference, pool="spin_adapted_sd", n_spatial_orbitals=2),
        seed=11,
    )
    assert len(selected.reconstruction.pool) == 4
    selected.model_dump_json()
    # Planning builds no compiler plan; build each one as a first selection would.
    compilers = selected._native["compiler_plans"]
    assert compilers.built() == ()
    for index in range(len(compilers)):
        assert compilers[index].generator is selected._native["inputs"].pool[index]
    assert [index for index, _ in compilers.built()] == [0, 1, 2, 3]
    loaded = [name for name in sys.modules if forbidden(name)]
    assert not (attempts or loaded), (
        f"SDK isolation violation: attempts={attempts}, loaded={loaded}"
    )
    print("ADAPT actual dense/sparse/default-pool planning: attempts=[], loaded=[]")


if __name__ == "__main__":
    main()
