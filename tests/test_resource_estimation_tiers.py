"""The dense-dilation CX law and the selected LCHS QSP resource breakdown.

Selected LCHS product-formula and dense resource laws are tested in
test_lchs_resource_structural_law.
"""

from __future__ import annotations

import numpy as np
import pytest

from nwqlib.backends.resources import block_encoding_per_query_cx

TRANSPILE_OPTIONS = {
    "basis_gates": ["cx", "u"],
    "optimization_level": 0,
    "seed_transpiler": 7,
}


def _heat_matrix(num_qubits: int):
    dimension=1<<num_qubits
    shift=np.roll(np.eye(dimension),1,axis=1)
    return (2*np.eye(dimension)-shift-shift.T).astype(complex)


@pytest.mark.parametrize("system_qubits,expected", [(1, 3), (3, 100)])
def test_dense_dilation_query_uses_complete_qsd_bound(system_qubits, expected):
    # SBM's optimized QSD law at the full dilation width, including its
    # single-qubit base case, supplies an integer circuit bound.
    assert block_encoding_per_query_cx(
        {}, implementation="dense_dilation", system_qubits=system_qubits
    ) == expected


@pytest.mark.parametrize('system_qubits,diagonal,expected_encoding',[
    (1,True,'dense_dilation'),(2,False,'banded')])
def test_selected_qsp_resource_structure_and_classification_admission(system_qubits,diagonal,expected_encoding):
    """The formula breakdown reports the coherent body once with the selected CX law, which bounds the built body.

    `resource_breakdown` reads the Program without building a circuit and
    gives the coherent body one call with its record's CX law, the
    attempt-CX law of the selected row, the no-ancilla upper bound of the
    same constructions (coherent_selection module docstring). Transpiled at
    fixed options, the built body has at most that many CX. A dense-dilation
    child has no CX law, and the breakdown then reports none.
    """
    from types import SimpleNamespace
    from qiskit import transpile
    from nwqlib import LinearDynamics,NormSquared,plan,estimate
    from nwqlib.algorithms.lchs import LCHS,ProviderConfig
    from nwqlib.algorithms.lchs.native import construct_coherent_lcu
    from nwqlib.algorithms.lchs.quantum_resources import resource_breakdown
    matrix=np.diag([.1,.2]) if diagonal else _heat_matrix(system_qubits)
    initial=np.zeros(len(matrix))
    initial[0]=1.
    if diagonal:
        initial[1]=.25
    # Four explicitly declared grid nodes keep this resource-law check bounded.
    # Default finite-tail selection is tested independently in test_lchs_default.
    method=LCHS(hamiltonian_evolution_backend='qsp_block_encoding',
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
            parameters={'num_qubits':2,'lsb_position':0}))
    if not diagonal:
        method=method.revise(approximation_tolerance=.5,
            lchs_kernel=ProviderConfig(implementation='near_optimal_eq7',parameters={'beta':.8}),
            k_quadrature=method.k_quadrature)
    problem=LinearDynamics(A=matrix,initial_state=initial,time=.1 if diagonal else .25)
    chosen=plan(problem,method=method,output=NormSquared())
    data=chosen._native['native_data']
    prepared=data.qsp_plan
    children={child.implementation for name,child in prepared.generator.components.items()
              if name in ('l_child','h_child') and child is not None}
    assert children=={expected_encoding}
    width=sum(r.width for r in chosen.construction.program.registers)
    assert width<=13
    folded=estimate(chosen)
    assert folded.construction_id==chosen.construction.content_id
    (entry,)=resource_breakdown(chosen)
    (body,)=[row for row in entry['blocks'] if row['block']=='lchs_coherent_lcu']
    law=prepared.candidate.cost['attempt_cx']
    assert body['calls']==1 and (law is None)==(expected_encoding=='dense_dilation')==(body['laws']==())
    built=construct_coherent_lcu(SimpleNamespace(_payload=data),(),None)
    cx=transpile(built,**TRANSPILE_OPTIONS).count_ops().get('cx',0)
    assert cx>0
    if law is not None:
        assert body['laws'][0].value==law and cx<=law
