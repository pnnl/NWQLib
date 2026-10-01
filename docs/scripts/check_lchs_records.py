#!/usr/bin/env python3
"""Fresh SDK-attempt audit of actual classical LCHS and compact periodic planning.

Use the development environment described in docs/MAINTENANCE.md. No SDK, backend, phase fitting or native circuit executes. The classical path computes its selected tiny model.
"""

import argparse

import builtins
import importlib.abc
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main(argv=None):
    """Exercise actual small host dynamics and archive reuse with all quantum SDK imports
    blocked.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; this audit must fail")
    args = parser.parse_args(argv)
    prefixes=("qiskit","qiskit_aer","qiskit_ibm_runtime","qiskit_ionq","cirq")
    attempts=[]
    def record(name):
        if any(name==prefix or name.startswith(prefix+'.') for prefix in prefixes):
            attempts.append(name)
            raise ImportError('blocked SDK import: '+name)
    original=builtins.__import__
    def guarded(name,globals=None,locals=None,fromlist=(),level=0):
        record(name)
        for member in fromlist or ():
            record(name+'.'+member)
        return original(name,globals,locals,fromlist,level)
    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            record(fullname)
            return None
    # Intercept both import entry points and record attempts before raising.
    # The final assertion must also catch forbidden imports swallowed by callers.
    builtins.__import__=guarded
    sys.meta_path.insert(0,Blocker())
    if args.poison:
        try:
            __import__('qiskit')
        except ImportError:
            pass
    from nwqlib import LinearDynamics,solve,load_result
    from nwqlib.algorithms.lchs import LCHS,LCHSVerification
    result=solve(LinearDynamics(A=[[.4,.15],[.05,.25]],initial_state=[1j,0],time=.1),
        method=LCHS(),execution='classical')
    with TemporaryDirectory() as folder:
        loaded=load_result(result.save(Path(folder)/'result'))
        assert loaded.plan==result.plan
        assert loaded.report()['result']==result.report()['result']
        receipt,facts=loaded.verify(checks=LCHSVerification(reference='selected_grid'))
        assert len(receipt.applications)==1 and facts[0].fact.value.value<1e-14
        from nwqlib import QuadraticForm,plan
        from nwqlib.algorithms.lchs import ProviderConfig
        from nwqlib.operators.inputs import PeriodicStencil,ingest_pauli
        from nwqlib.problems.inputs import ingest_occupation
        # A compact two-qubit periodic generator, Strang realization and IZ readout.
        periodic=LinearDynamics(A=PeriodicStencil(num_qubits=2,mass=1.,diffusion=.25,potential=.3),
            initial_state=ingest_occupation((0,0),num_qubits=2),time=.5)
        chosen=plan(periodic,output=QuadraticForm(observable=ingest_pauli([('IZ',1.)],num_qubits=2)),
            method=LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=2,lchs_kernel='cauchy_density',
                k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                    parameters={'num_qubits':2,'lsb_position':0})))
        assert chosen.reconstruction.psd_premise=='analytic_periodic'
    loaded=[name for name in sys.modules if any(name==prefix or name.startswith(prefix+'.') for prefix in prefixes)]
    assert not (attempts or loaded),f'SDK isolation violation: attempts={attempts}, loaded={loaded}'
    print('LCHS selected classical result, explicit verification and compact periodic Plan: attempts=[], loaded=[]')


if __name__=='__main__':
    main()
