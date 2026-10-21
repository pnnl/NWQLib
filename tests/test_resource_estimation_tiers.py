"""The dense-dilation CX law and selected LCHS QSP resource samples.

Selected LCHS product-formula and dense samples live in
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
def test_selected_qsp_resource_structure_and_classification_admission(system_qubits,diagonal,expected_encoding,monkeypatch):
    """Actual bounded inputs and read-only wide classification admission."""
    from nwqlib import LinearDynamics,NormSquared,plan,estimate
    from nwqlib.algorithms.lchs import LCHS,ProviderConfig
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
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
    selected=chosen._native['native_data'].qsp_plan
    assert selected['part_plans'][0].implementation==expected_encoding
    expansion=selected['expansion']
    queries=3*(expansion.cos_degree+expansion.sin_degree)
    law=selected['structural_law']
    assert law['block_encoding_queries']==law['child_encoding_calls']==queries
    assert law['projector_phase_count']==queries+6
    width=sum(r.width for r in chosen.construction.program.registers)
    assert width<=13
    folded=estimate(chosen)
    assert folded.construction_id==chosen.construction.content_id
    samples=sample_resources(chosen,max_qubits=13,transpile_options=TRANSPILE_OPTIONS)
    assert all(row['inventory']['basis']=='transpiled:cx,u'
        and row['inventory']['compiler']['options']['seed_transpiler']==7
        for row in samples['representatives'])
    assert set(samples['weighted_totals'])=={s.name for s in chosen.reconstruction.settings}
    assert all(totals['operations'].get('cx',0)>0 for totals in samples['weighted_totals'].values())
