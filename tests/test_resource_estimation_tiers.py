"""Shared dense/block-encoding CX laws, sampled-block records and LCHS QSP samples.

Selected LCHS product-formula and dense samples live in
test_lchs_resource_structural_law.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from nwqlib.backends.resources import (
    SampledBlock,
    block_encoding_per_query_cx,
)

TRANSPILE_OPTIONS = {
    "basis_gates": ["cx", "u"],
    "optimization_level": 0,
    "seed_transpiler": 7,
}


def test_sampled_block_multiplicity_is_a_nonnegative_integer() -> None:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    with pytest.raises(ValueError, match="multiplicity"):
        SampledBlock(name="invalid", circuit=circuit, multiplicity=-1)
    # A represented block can be absent (zero) or repeated; NumPy integers are exact.
    for multiplicity in (0, 3, np.int64(3)):
        block = SampledBlock(name="cx", circuit=circuit, multiplicity=multiplicity)
        assert block.multiplicity == multiplicity and type(block.multiplicity) is int


def test_prebuilt_block_encoding_has_no_width_only_cx_law() -> None:
    # An arbitrary prebuilt circuit has no width-only CX law.
    assert block_encoding_per_query_cx(
        {}, implementation="prebuilt_native", system_qubits=3
    ) is None


def _heat_matrix(num_qubits: int):
    dimension=1<<num_qubits
    shift=np.roll(np.eye(dimension),1,axis=1)
    return (2*np.eye(dimension)-shift-shift.T).astype(complex)


@pytest.mark.parametrize("system_qubits,expected", [(0, 0), (1, 3), (2, 20), (3, 100)])
def test_dense_dilation_query_uses_complete_qsd_bound(system_qubits, expected):
    # SBM's optimized QSD law at the full dilation width, including its
    # single-qubit base case, supplies an integer circuit bound.
    assert block_encoding_per_query_cx(
        {}, implementation="dense_dilation", system_qubits=system_qubits
    ) == expected


def test_outer_qsp_classification_gate_admits_six_and_refuses_seven_full_support_qubits():
    """The outer QSP gate's charge functions give the documented full-support planning ledger.

    The integer-law evaluations stated at compiled_selection.
    _compiled_qsp_part_plans and in ENGINEERING_CONSTANTS for two
    full-support children (m = d**2 terms, b = d bands): planning work
    654,924, 5,777,014, 60,033,772 and 735,289,238 units and outer bytes
    42,168,320, 161,732,608, 637,289,984 and 2,528,890,880 at q = 4 to 7.
    The ledger is two decompositions, 32*d**2 pruning, per child the tables
    and classification that _admit_pauli_plan returns, one dense
    reconstruction and one singular value normalization. Only integer
    charge functions run here.
    """
    from types import SimpleNamespace
    from nwqlib.algorithms.lchs.compiled_selection import _qsp_outer_bytes
    from nwqlib.algorithms.lchs.time_independent_terms import _pauli_decomposition_requirements
    from nwqlib.subroutines.block_encoding.core import _admit_pauli_plan, _dense_norm_work
    works=(654_924,5_777_014,60_033_772,735_289_238)
    sizes=(42_168_320,161_732_608,637_289_984,2_528_890_880)
    for q,work,size in zip(range(4,8),works,sizes,strict=True):
        d=1<<q
        n=d*d
        child=_admit_pauli_plan(SimpleNamespace(num_qubits=q,terms=range(n)),held_bytes=0,
            max_bytes=10**15,max_work=10**12,classify=True)
        ledger=_pauli_decomposition_requirements(d)[0]+32*n+2*(child+n*d*d+_dense_norm_work(d,False))
        assert ledger==work and _qsp_outer_bytes(q,(n,n))==size
        assert (ledger<=100_000_000 and size<=10_000_000_000)==(q<=6)


def test_outer_qsp_gate_passes_one_remaining_work_budget_and_live_bytes_to_its_children(monkeypatch):
    """Each QSP child is admitted against what the decomposition, pruning and earlier children left.

    compiled_selection._compiled_qsp_part_plans subtracts the two
    decompositions and 32*d**2 pruning before the L child, passes the L
    child's remainder to the H child and keeps the completed L table in the
    H child's held bytes (a dense-dilation L child also spends its
    reconstruction and any norm); plan_block_encoding admits each child
    against the same allowance its outer check used.
    """
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.algorithms.lchs.compiled_selection import _compiled_qsp_part_plans
    from nwqlib.algorithms.lchs.time_independent_terms import _pauli_decomposition_requirements
    calls = []
    original = core._admit_pauli_plan
    def record(decomposition, *, held_bytes, max_bytes, max_work, classify):
        work = original(decomposition, held_bytes=held_bytes, max_bytes=max_bytes, max_work=max_work, classify=classify)
        calls.append((len(decomposition.terms), held_bytes, max_work, work))
        return work
    monkeypatch.setattr(core, "_admit_pauli_plan", record)
    q, d, limit = 2, 4, 10**8
    l_part = np.diag([.1, .2, .3, .4]).astype(complex) + .05*np.diag(np.ones(3), 1) + .05*np.diag(np.ones(3), -1)
    h_part = np.diag([.3, -.1, .2, .05]).astype(complex) + .02j*np.diag(np.ones(3), 1) - .02j*np.diag(np.ones(3), -1)
    l_plan, h_plan, _, remaining = _compiled_qsp_part_plans(l_part, h_part, l_norm=.6, max_select_work=limit)
    assert l_plan.implementation != "exact_zero" and h_plan.implementation != "exact_zero" and len(calls) == 4
    (m_l, held_l, work_l, spent_l), inner_l, (m_h, held_h, work_h, _), inner_h = calls
    assert work_l == limit - _pauli_decomposition_requirements(d)[0] - 32*d*d and inner_l[2] == work_l
    # A dense-dilation L child also spends its reconstruction and any norm.
    assert work_h <= work_l - spent_l and inner_h[2] == work_h
    assert held_h == held_l + 64*(1 << (m_l-1).bit_length())*(q+16)
    # The caller receives what the H child left of the same ledger; a Pauli
    # child spends exactly its admitted table and classification work.
    _, _, _, spent_h = calls[2]
    assert 0 <= remaining <= work_h - spent_h
    if h_plan.implementation != "dense_dilation":
        assert remaining == work_h - spent_h


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
