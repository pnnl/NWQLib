"""Direct scientific LCHS inputs, numerical selection and circuit consumers."""


import numpy as np
import pytest

from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs.providers import resolve_lchs_coefficient_plan


def test_source_phase_and_original_coordinates_reach_both_actual_consumers():
    from nwqlib import plan, solve
    from nwqlib.algorithms import LCHS
    # Complex scaling is physical; the unused padded coordinate is not part
    # of the solution. Equal initial/source handles still have distinct roles.
    diagonal = np.array([.1+.05j,.2-.04j,.3+.02j])
    initial = .5*np.exp(1j*np.pi/3)*np.array([1.,1j,-.25j])
    problem = LinearDynamics(A=np.diag(diagonal),initial_state=initial,source=initial,time=.1)
    # This witness tests source/phase/original-coordinate association, not
    # automatic quadrature accuracy. Keep its selected native width bounded.
    from nwqlib.algorithms.lchs import ProviderConfig
    method = LCHS(duhamel_nodes=2, k_quadrature=ProviderConfig(
        implementation="signed_binary_uniform", parameters={"num_qubits":2, "lsb_position":-1}))
    chosen = plan(problem,method=method)
    assert sum(w.width for w in chosen.construction.program.registers)<=8
    selected = chosen._native['native_data']
    quad = selected.quadrature
    # Independent diagonal exponentials and coherent Duhamel sum. No SELECT,
    # Pauli/PF builder or result field supplies the expected native action.
    def evolution(elapsed):
        return sum(c*np.exp(-1j*elapsed*(k*diagonal.real+diagonal.imag))
            for k,c in zip(quad.k_nodes,quad.coefficients,strict=True))
    expected = evolution(.1)*initial
    for node,weight in zip(chosen.reconstruction.source_nodes,chosen.reconstruction.source_weights,strict=True):
        expected += weight*evolution(.1-node)*initial
    quantum = solve(chosen)
    classical = solve(problem,method=method,execution='classical')
    assert quantum.solution.shape==classical.solution.shape==(3,)
    # <=9q native synthesis roundoff; no assertion about finite-grid accuracy.
    np.testing.assert_allclose(quantum.solution,expected,rtol=0,atol=2e-9)
    np.testing.assert_allclose(classical.solution,expected,rtol=0,atol=2e-13)
    arguments = [{item.parameter:item.value for item in app.arguments} for app in classical.applications]
    assert [row['source_application'] for row in arguments]==[0,1,1]
    for row,weight in zip(arguments,(1.,.05,.05),strict=True):
        assert row['weight'].value==pytest.approx(weight*np.sqrt(33)/8,rel=2e-15, abs=0)
        assert row['recovery_scale'].value==1.  # Positive L needs no PSD shift.


def test_exact_zero_l_removes_k_quadrature_and_keeps_physical_phase():
    from nwqlib import plan, solve
    from nwqlib.algorithms import LCHS
    problem = LinearDynamics(A=1j*np.diag([.2,.3]),initial_state=[1,1j],time=.1)
    selected = plan(problem,method=LCHS(ham_evo='qsp_block_encoding'))
    assert selected.reconstruction.mode=='unitary'
    assert selected.reconstruction.physical_branches==1
    assert selected.reconstruction.kernel_approximation_bound==selected.reconstruction.quadrature_bound==0
    assert sum(w.width for w in selected.construction.program.registers)<=13
    result = solve(selected)
    expected = np.exp(-.1j*np.array([.2,.3]))*np.array([1,1j])
    # The kernel and k-quadrature stages are exact zeros here, so the
    # deviation is the evolution's own approximation, bounded by the Plan's
    # qsp_synthesis component, plus binary64 simulation rounding.
    synthesis = next(item.fact.value.value for item in selected.facts if item.fact.quantity=="qsp_synthesis")
    assert np.linalg.norm(result.solution-expected) <= synthesis+1e-12


def test_periodic_selected_repeat_matches_independent_strang_matrix():
    from nwqlib import plan, solve, NormSquared
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs import ProviderConfig, Trotter
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.problems.inputs import ingest_occupation
    q,steps,elapsed,mass,diffusion,potential = 4,3,.1,.1,.1,.05
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=q,mass=mass,diffusion=diffusion,potential=potential),
        initial_state=ingest_occupation((0,)*q,num_qubits=q),time=elapsed)
    chosen = plan(problem,method=LCHS(ham_evo=Trotter(steps=steps),
        lcu_select='structured',
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
            parameters={'num_qubits':3,'lsb_position':-1})),output=NormSquared())
    coefficients = resolve_lchs_coefficient_plan(chosen)
    d = 1<<q
    identity = np.eye(d,dtype=complex)
    x = identity[np.arange(d)^1]
    shift = np.roll(identity,1,axis=0)  # S|j> = |j+1 mod d>.
    z = 1-2*(np.arange(d)&1)
    rz = np.diag(np.exp(-.5j*elapsed/steps*potential*z))
    expected = np.zeros(d,dtype=complex)
    for k,c in zip(coefficients.nodes,coefficients.coefficients,strict=True):
        angle = -elapsed/steps*k*diffusion
        half = np.cos(angle/2)*identity-1j*np.sin(angle/2)*x
        full = np.cos(angle)*identity-1j*np.sin(angle)*x
        step = rz@half@shift@full@shift.conj().T@half@rz
        expected += c*np.exp(-1j*elapsed*k*(mass+2*diffusion))*np.linalg.matrix_power(step,steps)[:,0]
    # This protects Strang action at each actual node; automatic quadrature
    # accuracy is independent. Eight explicit addresses keep native width seven.
    assert sum(w.width for w in chosen.construction.program.registers)==7
    result = solve(chosen)
    # Construction identity against analytic Pauli rotations/permutations,
    # independent of the native QFT/multiplexor emitters; not exact-PDE error.
    assert result.norm_squared==pytest.approx(float(np.vdot(expected,expected).real),rel=0,abs=2e-11)


def test_periodic_hundred_qubit_estimate_keeps_symbolic_system(monkeypatch):
    from nwqlib import plan, estimate, NormSquared
    from nwqlib.algorithms import LCHS
    from nwqlib.operators.inputs import OperatorInput, PeriodicStencil
    from nwqlib.problems.inputs import StateInput, ingest_occupation
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=100,mass=.1,diffusion=.1,potential=.05),
        initial_state=ingest_occupation((0,)*100,num_qubits=100),time=.1)
    def forbidden(*args,**kwargs):
        pytest.fail('symbolic periodic work materialized a system input')
    monkeypatch.setattr(OperatorInput,'dense_array',forbidden)
    monkeypatch.setattr(StateInput,'physical_vector',forbidden)
    chosen = plan(problem,method=LCHS(ham_evo='trotter',lcu_select='structured'),
        output=NormSquared())
    result = estimate(chosen)
    assert result.construction_id==chosen.construction.content_id
    assert len(chosen.construction.selections)==5


@pytest.mark.parametrize('num_qubits,time', [(2, 1e-6), (3, 1e-8), (2, 1.)])
def test_dense_select_branches_match_exact_controlled_exponentials(num_qubits, time):
    from scipy.linalg import expm
    from qiskit.quantum_info import Operator
    from nwqlib import plan
    from nwqlib.algorithms import LCHS
    from nwqlib.blocks.lowering import _local_method_context
    # Qiskit 2.6.0 synthesized a near-identity branch with entry errors of
    # about 1e-6 on two system qubits at time 1e-6 and 2e-10 to 1e-9 on three
    # at time 1e-8. Time 1 gives generic branches. The expected branch is
    # SciPy's expm of the saved Cartesian parts, independent of the branch
    # constructor. An eight-node grid keeps three address qubits. 1e-11 is
    # about 80 times the largest rounding error of these controlled
    # constructions, 1.2e-13.
    rng = np.random.default_rng(31 + num_qubits)
    d = 1 << num_qubits
    x = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    y = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    matrix = x @ x.conj().T / d + .5j * (y + y.conj().T)
    initial = np.zeros(d)
    initial[0] = 1
    grid = {'implementation': 'signed_binary_uniform', 'parameters': {'num_qubits': 3, 'lsb_position': -1}}
    chosen = plan(LinearDynamics(A=matrix, initial_state=initial, time=time), method=LCHS(k_quadrature=grid))
    select = {block.record.signature.name: block for block in chosen.blocks}['lchs_select']
    data, elapsed = select._payload
    assert data.backend == chosen.reconstruction.selected_backend == 'dense_exact'
    circuit = select._constructor(select, (), _local_method_context())
    nodes = np.asarray(data.quadrature.k_nodes)
    coefficients = np.asarray(data.coefficient_plan.coefficients)
    address = circuit.num_qubits - num_qubits
    assert address == 3
    for branch in range(len(nodes)):
        exact = expm(-1j * elapsed * (nodes[branch] * data.dense.l_part + data.dense.h_part))
        expected = np.eye(1 << circuit.num_qubits, dtype=complex)
        selected = [(row << address) | branch for row in range(d)]
        expected[np.ix_(selected, selected)] = coefficients[branch] / abs(coefficients[branch]) * exact
        actual = Operator(circuit.data[branch].operation).data
        assert np.abs(actual - expected).max() <= 1e-11, (branch, nodes[branch])


def test_source_pf_padding_and_original_readout_follow_recorded_steps():
    from nwqlib import plan, solve, StateVector, NormSquared, NormalizedExpectation, Samples
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs.provider_config import ProviderConfig
    from _lchs_suzuki_reference import sorted_pauli_suzuki_2_reference
    matrix = np.array([[.3,.04+.01j,0],[.04-.03j,.4,.02],[0,.02,.2]])
    initial,source = np.array([1j,-.2,.1]),np.array([.1,.2j,-.1j])
    problem = LinearDynamics(A=matrix,initial_state=initial,source=source,time=.2)
    # The budgeted backend selects its own steps.
    method = LCHS(ham_evo='trotter_error_budgeted',duhamel_nodes=3,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen = plan(problem,method=method)
    data = chosen._native['native_data']
    layout,pf = data.source_layout,data.select_data.plan
    assert layout.branch_count==16 and len(layout.phases)==32
    assert any(slot is None for slot in layout.branch_to_node)
    assert sum(register.width for register in chosen.construction.program.registers)==7
    expected = np.zeros(4,dtype=complex)
    for slot,node in enumerate(layout.branch_to_node):
        if node is None:
            continue
        direction = data.dense.initial_direction if layout.branch_kinds[node]=='initial' else data.dense.source_direction
        action = sorted_pauli_suzuki_2_reference(data.dense.l_part,data.dense.h_part,
            k_value=layout.k_values[slot],final_time=layout.elapsed_times[slot],reps=pf.branch_step_counts[slot])
        # Slot coefficient from the layout's coefficient state: magnitude
        # gamma*amplitude**2 with gamma = coefficient_l1_norm*2**scale_exponent.
        coefficient = (layout.coefficient_l1_norm*2.0**layout.scale_exponent*layout.amplitudes[slot]**2
                       *np.exp(1j*layout.phases[slot]))
        expected += coefficient*(action@direction)
    actual = solve(chosen)
    np.testing.assert_allclose(actual.solution,expected[:3],rtol=0,atol=3e-10)
    unit = solve(problem,method=method,output=StateVector(normalization='unit'))
    np.testing.assert_allclose(unit.state_vector,expected[:3]/np.linalg.norm(expected[:3]),rtol=0,atol=3e-10)
    mass = np.linalg.norm(expected[:3])**2
    observable = np.diag([1.,-2.,.5])
    exact = solve(problem,method=method,output=NormalizedExpectation(observable=observable))
    expected_value = float(np.vdot(expected[:3],observable@expected[:3]).real/mass)
    assert exact.value==pytest.approx(expected_value,abs=3e-10)
    counts = solve(problem,method=method,output=NormSquared(),shots=4096,seed=11)
    alpha = chosen.reconstruction.recovery.as_float()
    # Independent Bernoulli physical success mass, with its actual recovery;
    # six standard deviations checks readout wiring rather than promising accuracy.
    p = mass/alpha**2
    assert abs(counts.value-mass)<6*alpha**2*np.sqrt(p*(1-p)/4096)
    # The reported shot radius is Hoeffding's for the success indicator in
    # [0, alpha**2] (M7.14), at the default failure probability .05; a
    # QuadraticForm's one Z-type group has scores in [-alpha**2 A, alpha**2 A]
    # with A the sum of the padded observable's Pauli coefficient magnitudes.
    def sampling(result):
        return next(item for item in result.facts if item.fact.quantity == "sampling")
    radius = sampling(counts)
    assert radius.failure_probability == .05
    assert radius.fact.value.value == pytest.approx(alpha**2*np.sqrt(np.log(2/.05)/(2*4096)), rel=1e-12)
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib import QuadraticForm
    padded = np.diag([1., -2., .5, 0.])
    form = solve(problem,method=method,output=QuadraticForm(observable=observable),shots=2048,seed=13)
    A = np.sum(np.abs(SparsePauliOp.from_operator(padded).coeffs))
    assert sampling(form).fact.value.value == pytest.approx(alpha**2*A*np.sqrt(2*np.log(2/.05)/2048), rel=1e-12)
    samples = solve(problem,method=method,output=Samples(),shots=128,seed=12)
    assert all(index<3 for index in samples.samples.indices.array)
    # Sample arrays are Result data, stored in bytes objects whose memory cannot be made writable.
    for array in (samples.samples.indices.array,samples.samples.counts.array):
        while isinstance(array.base,np.ndarray):
            array = array.base
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_component_facts_cover_the_ideal_rule_and_native_floating_point_names_the_coefficient_table():
    """Plans and Results scope the kernel, k-quadrature and Duhamel bounds to the ideal rule.

    Those bounds are for exact nodes, weights and coefficients. The concrete component Facts of a default
    LCHS Plan, with and without a constant source, of its classical Result and of the refined Duhamel bound
    say so. The required native_floating_point term, which holds the binary64 construction of the
    coefficient table and the execution, stays unknown with that reason, and the Result's assessment reports it.
    """
    from nwqlib import plan, solve
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs import LCHSRefinement
    assumptions = ("ideal selected algorithm relation with exact nodes, weights and coefficients, evaluated numerically",
                   "binary64 construction of the coefficient table and execution rounding are under native_floating_point")
    reason = ("binary64 construction of the coefficient table (quadrature nodes and weights, kernel values, "
              "Duhamel nodes and weights) and floating-point execution are not bounded")
    A, initial = np.array([[.6, .1], [.1, .3]]), np.array([1., 0.])
    for problem, method in ((LinearDynamics(A=A, initial_state=initial, time=1.), LCHS()),
                            (LinearDynamics(A=A, initial_state=initial, source=[.2, -.1], time=1.), LCHS(duhamel_nodes=2))):
        chosen = plan(problem, method=method, execution="classical")
        planned = {item.fact.quantity: item.fact for item in chosen.facts}
        for name in ("kernel_approximation", "k_quadrature"):
            assert planned[name].availability == "concrete" and planned[name].assumptions == assumptions
        native = next(term.fact.fact for term in chosen.error_model.terms if term.name == "native_floating_point")
        assert native.availability == "unknown" and native.reason == reason
        result = solve(chosen)
        # sampling comes from the shared exact-readout model. Every concrete LCHS component carries the scope.
        concrete = {item.fact.quantity: item.fact for item in result.facts if item.fact.availability == "concrete"}
        assert {"kernel_approximation", "k_quadrature"} <= concrete.keys()
        assert all(concrete[name].assumptions == assumptions for name in concrete.keys() - {"sampling"})
        assert f"native_floating_point: {reason}" in result.assess(absolute_tolerance=.01).prerequisites
    _, refined = result.verify(checks=LCHSRefinement(components=("duhamel",)))
    duhamel = next(item.fact for item in refined if item.fact.quantity == "duhamel_quadrature")
    assert duhamel.availability == "concrete" and duhamel.assumptions == assumptions


def test_default_byte_limit_is_the_physical_memory_read_when_a_step_checks(monkeypatch):
    """LCHS.max_bytes defaults to None, and each step then checks against the computer's physical memory.

    For a 2x2 A, the first array check of planning, the encoded inputs and Cartesian parts, needs
    64*d**2 + 128*d = 512 bytes. A physical memory of 511 bytes refuses it and names the physical memory, while
    512 passes it and the next stage refuses. An explicit max_bytes has the same boundary and names itself, a
    large physical memory plans, and the former explicit 10 GB value is still an ordinary explicit cap.
    """
    import nwqlib._limits as limits
    from nwqlib import plan, solve
    from nwqlib.algorithms import LCHS
    problem = LinearDynamics(A=np.array([[.6, .1], [.1, .3]]), initial_state=[1., 0.], time=1.)
    assert LCHS().max_bytes is None and LCHS(max_bytes=10_000_000_000).max_bytes == 10_000_000_000
    first = "LCHS encoded inputs and Cartesian arrays needs about 512 bytes, exceeding "
    for memory, refused in ((511, first + "the physical memory (511 bytes)"), (512, "LCHS quadrature arrays")):
        monkeypatch.setattr(limits, "physical_memory_bytes", lambda memory=memory: memory)
        with pytest.raises(ValueError, match=refused.replace("(", r"\(").replace(")", r"\)")):
            plan(problem, method=LCHS(), execution="classical")
    for cap, refused in ((511, first + "max_bytes=511"), (512, "LCHS quadrature arrays")):
        with pytest.raises(ValueError, match=refused):
            plan(problem, method=LCHS(max_bytes=cap), execution="classical")
    monkeypatch.setattr(limits, "physical_memory_bytes", lambda: 2**40)
    np.testing.assert_array_equal(solve(plan(problem, method=LCHS(), execution="classical")).solution,
                                  solve(plan(problem, method=LCHS(max_bytes=10_000_000_000), execution="classical")).solution)
