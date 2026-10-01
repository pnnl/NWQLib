"""A small exact-identity author falsifier using the actual Expectation Method."""

from nwqlib import Expectation, prepare, submit
from nwqlib.algorithms import ExpectationMethod
from nwqlib.algorithms.authoring import MethodCase
from nwqlib.operators import ingest_pauli


def case():
    """<psi|2I|psi>/<psi|psi>=2; this selected Method needs no acquisitions."""

    def evaluate(plan):
        with submit(prepare(plan)) as run:
            result = run.wait(timeout=1)
            assert not run.trace.events
            return result

    return MethodCase(
        method=ExpectationMethod(),
        problem=Expectation(state=(1.0, 0.0), observable=ingest_pauli((("I", 2.0),), num_qubits=1)),
        evaluate=evaluate,
        accepts=lambda result: result.value == 2.0,
        invalid_result=lambda result: result.revise(value=3.0),
        seed=7,
    )
