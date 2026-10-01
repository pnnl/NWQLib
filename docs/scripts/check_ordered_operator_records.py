"""Fresh-process actual ordered mapping/product metadata SDK attempt audit."""

import argparse

import builtins
import importlib.abc
import sys


def main(argv=None):
    """Check ordered fermion/Pauli mappings and compact metadata without importing backend or SDK
    code.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; this audit must fail")
    args = parser.parse_args(argv)
    attempts = []
    blocked = ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "qiskit_ionq", "braket",
               "cirq", "pennylane", "nwqlib.backends", "nwqlib.subroutines")

    def forbidden(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)

    def record(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("blocked ordered-operator import: " + name)

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
    if args.poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass
    from nwqlib.operators import (
        ProductManifest, FactorizedOperatorProduct, fermion_table, ingest_fermion,
        OperatorFactReport, OperatorInput, PeriodicStencil, operator_input, ingest_pauli, refine_operator_facts,
    )
    from nwqlib.core import Scope, Unit
    from nwqlib.problems.inputs import StateInput, state_input
    raw = fermion_table([(1, ((1, 1),))], num_modes=2)
    mapped = raw.to_pauli(mapping="jw")
    assert dict(mapped.pauli_terms().labels()) == {"XZ": 0.5, "YZ": -0.5j}
    assert dict(raw.to_pauli(mapping="z_free").pauli_terms().labels()) == {"XI": 0.5, "YI": -0.5j}
    native = ingest_fermion([(1, ((1, 1),))], num_modes=2)
    product = FactorizedOperatorProduct((native, mapped))
    assert product.manifest.factors == (native.manifest.reference, mapped.manifest.reference)
    assert ProductManifest.model_validate_json(product.manifest.model_dump_json()) == product.manifest
    assert native.manifest.model_validate_json(native.manifest.model_dump_json()) == native.manifest
    structural = ingest_pauli((("Z", 1.),), num_qubits=1)
    facts = refine_operator_facts(structural, unit=Unit(symbol="Ha", dimension="energy"),
        scope=Scope(domain="one admitted Pauli"), refinements=("certify_pauli_l1",))
    assert facts.facts[0].fact.value.numerator == facts.facts[0].fact.value.denominator == 1
    assert OperatorFactReport.model_validate_json(facts.model_dump_json()) == facts
    operator = operator_input([[1, 0], [0, 2]])
    state = state_input([3, 4j])
    for handle, reader in ((operator, OperatorInput), (state, StateInput)):
        restored = reader.from_record(handle.to_record())
        assert restored.reference == handle.reference
    assert state.preparation.physical_scale.as_float() == 5
    assert operator_input(PeriodicStencil(100, .1, .1)).basis.dimension == 1 << 100
    loaded = [name for name in sys.modules if forbidden(name)]
    assert not (attempts or loaded), f"SDK isolation violation: attempts={attempts}, loaded={loaded}"
    print("SDK-free ordered mapping, product metadata and explicit structural refinement passed; attempts=[], loaded=[]")


if __name__ == "__main__":
    main()
