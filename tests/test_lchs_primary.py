"""Natural LCHS input admission, acquired populations and numerical work ownership."""

import weakref
import numpy as np
import pytest
from nwqlib import LinearDynamics,QuadraticForm,plan,solve
from nwqlib.algorithms.lchs import LCHS,ProviderConfig


@pytest.mark.parametrize("execution", ("classical", "quantum"))
def test_lchs_archive_preserves_selected_numerical_data(tmp_path, execution):
    from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan
    from nwqlib.algorithms.lchs.parameters import selected_identity

    method = LCHS(k_quadrature=ProviderConfig(
        implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": -1}))
    chosen = plan(LinearDynamics(A=np.diag([.2, .4]), initial_state=[1., 0.], time=.1),
                  method=method, execution=execution)
    files = ArchiveFiles(tmp_path, 4_000_000)
    saved = save_plan(chosen, files)
    restored = load_plan(saved, ArchiveFiles(tmp_path, 4_000_000))
    assert restored.content_id == chosen.content_id
    assert selected_identity(restored._native["native_data"], restored.problem, method) == selected_identity(
        chosen._native["native_data"], chosen.problem, method)


def test_selected_identity_encoding_distinguishes_mapping_nesting():
    from hashlib import sha256
    from nwqlib.algorithms.lchs.parameters import _feed_selected_value

    def encoded(value):
        digest = sha256()
        _feed_selected_value(digest, value)
        return digest.hexdigest()

    # Moving the key "y" into the nested mapping changes the selected data. A
    # stream that lists mapping entries without their count feeds the same
    # bytes for both values.
    assert encoded({"x": {"IZ": 1.0}, "y": 2.0}) != encoded({"x": {"IZ": 1.0, "y": 2.0}})
    # Legal preservation: insertion order and NumPy scalar types are not
    # selected data, so equal values keep one identity.
    assert encoded({"a": 1.0, "b": 2.0}) == encoded({"b": 2.0, "a": 1.0})
    assert encoded((np.float64(1.5), np.int64(3))) == encoded((1.5, 3))


def test_dense_host_streams_only_requested_operator_and_result_reanalysis_is_free(monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    from scipy.linalg import expm
    matrix=np.array([[-.2,.03j],[.03j,.4]])
    problem=LinearDynamics(A=matrix,initial_state=[1,1j],time=.1)
    method=LCHS(k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen=plan(problem,method=method,execution='classical')
    quad=chosen._native['native_data'].quadrature
    expected=sum(c*expm(-.1j*(k*quad.l_part+quad.h_part)) for k,c in zip(quad.k_nodes,quad.coefficients,strict=True))
    expected*=np.exp(.1*quad.conversion['psd_shift'])
    # The host applies each node evolution to the vector from one Hermitian
    # eigensystem per node, streamed: at most one eigenvector matrix is alive.
    original=terms._node_eigensystem
    refs=[]
    peak=0
    def node(*args):
        nonlocal peak
        values,vectors=original(*args)
        refs.append(weakref.ref(vectors))
        peak=max(peak,sum(ref() is not None for ref in refs))
        return values,vectors
    monkeypatch.setattr(terms,'_node_eigensystem',node)
    result=solve(chosen)
    np.testing.assert_allclose(result.solution,expected@np.array([1,1j]),rtol=0,atol=2e-14)
    assert len(refs)==4 and peak<=2 and not any(ref() is not None for ref in refs)
    monkeypatch.setattr(terms,'_node_eigensystem',lambda *a: pytest.fail('reanalysis repeated evolution'))
    repeated=result.analyze()
    np.testing.assert_array_equal(repeated.solution,result.solution)


def _price_out_dense_route(monkeypatch, host_pf, when=lambda rotations: True):
    """Make the dense route unaffordable where ``when`` holds, so a test of the vector action runs it.

    On a two-dimensional system the single-label action law
    (operators._pauli.pauli_action_requirements) makes the dense route the
    cheaper one.
    """
    law = host_pf._route_requirements
    def priced(dimension, steps, rotations, route):
        work, size, products = law(dimension, steps, rotations, route)
        return (10**15 if route == "dense_power" and when(rotations) else work), size, products
    monkeypatch.setattr(host_pf, "_route_requirements", priced)


@pytest.mark.parametrize('order',[1,2])
@pytest.mark.parametrize('with_source',[False,True])
def test_selected_pauli_vector_action_preserves_order_phase_source_and_replay(order,with_source,tmp_path,monkeypatch):
    from nwqlib import load_result
    from nwqlib.algorithms.lchs import host_pf
    _price_out_dense_route(monkeypatch, host_pf)
    from qiskit.quantum_info import Operator
    x=np.array([[0.,1.],[1.,0.]])
    z=np.diag([1.,-1.])
    initial=np.array([1j,.3])
    source=np.array([.2,.1j]) if with_source else None
    matrix=np.diag([-.2,.3])+1j*(.4*x+.17*np.eye(2))
    method=LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=3,trotter_order=order,duhamel_nodes=2,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen=plan(LinearDynamics(A=matrix,initial_state=initial,source=source,time=.2),method=method,execution='classical')
    data=chosen._native['native_data']
    assert all(node['route']=='vector' for action in data.host_actions['applications'] for node in action['positions'])
    expected=np.zeros(2,dtype=complex)
    for action in data.host_actions['applications']:
        duration=.2-action['start']
        vector=source if action['source'] else initial
        for k,c in zip(data.quadrature.k_nodes,data.quadrature.coefficients,strict=True):
            # Analytic exponentials of X,Z, with original identity and the
            # selected PSD shift. This is the literal ordered r=3 product.
            a=.4*duration/3
            b=-.25*k*duration/3
            rx=np.cos(a)*np.eye(2)-1j*np.sin(a)*x
            rz=np.cos(b)*np.eye(2)-1j*np.sin(b)*z
            if order==2:
                half=np.cos(a/2)*np.eye(2)-1j*np.sin(a/2)*x
                step=half@rz@half
            else:
                step=rz@rx
            phase=np.exp(-1j*duration*(.17+k*(.05+chosen.reconstruction.psd_shift)))
            expected+=action['weight']*c*np.exp(chosen.reconstruction.psd_shift*duration)*phase*(step@step@step@vector)
    def forbidden(*args,**kwargs):
        raise AssertionError('selected vector action extracted a full circuit operator or replanned')
    monkeypatch.setattr(Operator,'__init__',forbidden)
    executed=[]
    rotate=host_pf.apply_terms
    monkeypatch.setattr(host_pf,'apply_terms',lambda *a,**k: executed.append(a[0]) or rotate(*a,**k))
    result=solve(chosen)
    # At most12 selected node actions,9 rotations each and tiny matrix sums;
    # 256eps bounds this well-conditioned two-coordinate roundoff witness.
    np.testing.assert_allclose(result.solution,expected,rtol=256*np.finfo(float).eps,atol=256*np.finfo(float).eps)
    # One initial and two Duhamel applications over k=(0,.5,-1,-.5). H+kL keeps
    # X and -k/4 Z, so k=0 has one rotation per step and the other nodes have
    # two (Lie) or three (Suzuki); three steps each, all on the vector route.
    applications=3 if with_source else 1
    rotations=applications*3*(1+3*(1+order))
    assert tuple(data.quadrature.k_nodes)==(0.,.5,-1.,-.5)
    work=dict(chosen.reconstruction.classical_work)
    assert (work['dimension'],work['operator_applications'],work['evolution_calls'],work['pauli_decompositions'],
        work['commutator_checks'],work['matrix_power_products'],work['dense_step_rotations'])==(2,applications,4*applications,2,0,0,0)
    assert len(executed)==work['vector_rotations']==rotations
    kernel,=chosen.construction.kernels
    law,=kernel.resource_laws
    assert law.metric=='classical_work' and kernel.construction_work==work['selection_work']
    # Each executed rotation updates both amplitudes of the host vector.
    assert kernel.invocation_work==law.value==work['size_units']-work['selection_work']>=2*rotations
    line=f"Classical work proxy: {work['size_units']}; vector rotations={rotations}; matrix-power products=0"
    assert line in str(result)
    result.save(tmp_path/'result')
    monkeypatch.setattr(host_pf,'select_actions',forbidden)
    monkeypatch.setattr(host_pf,'apply_node',forbidden)
    loaded=load_result(tmp_path/'result')
    assert loaded.plan.reconstruction.classical_work==chosen.reconstruction.classical_work
    assert loaded.report()['plan']==result.report()['plan'] and line in str(loaded)
    np.testing.assert_array_equal(loaded.analyze().solution,result.solution)
    assert len(executed)==rotations


# Hermitian part with eigenvalue -.2, so planning shifts L and every
# application carries the recovery exp(shift*(T-s)).
_SHIFTED = [[-.2,.1,0],[.1,.4,.05j],[0,-.05j,.3]]


def test_classical_plan_publishes_the_component_facts_its_result_reports():
    """Planning fixes every component input, so the Plan states what solve records.

    The kernel and k-quadrature stages weight the provider bounds by the
    initial application's physical input weight ||u_0|| and PSD recovery
    exp(shift*T), and trotter_synthesis sums |c_j| times each node's stored
    product-formula and pruning bound. None of them needs the host action,
    and no stage stays unknown.
    """
    from math import exp, fsum
    initial = [1]+[0]*(len(_SHIFTED)-1)
    problem = LinearDynamics(A=_SHIFTED,initial_state=initial,time=.2)
    chosen = plan(problem,method=LCHS(hamiltonian_evolution_backend='trotter_error_budgeted'),execution='classical')
    planned = {item.fact.quantity:item for item in chosen.facts}
    assert planned == {term.name:term.fact for term in chosen.error_model.terms if term.role=='listed_only'}
    result = solve(chosen)
    assert {item.fact.quantity:item for item in result.facts if item.fact.quantity in planned} == planned
    assert {name:item.fact.reason for name,item in planned.items() if item.fact.availability=='unknown'} == {}
    # A concrete stage is ||u_0||*exp(shift*T) times its stored bound.
    rec,data = chosen.reconstruction,chosen._native['native_data']
    assert rec.psd_shift > 0
    inputs = [(0.,rec.initial_scale.as_float())]
    factors = [scale*exp(rec.psd_shift*(problem.elapsed_time-start)) for start,scale in inputs]
    for name,bound in (('kernel_approximation',rec.kernel_approximation_bound),('k_quadrature',rec.quadrature_bound)):
        assert planned[name].fact.value.value == pytest.approx(fsum(f*bound for f in factors),rel=4*np.finfo(float).eps)
    assert [action['start'] for action in data.host_actions['applications']] == [start for start,_ in inputs]
    stored = fsum(f*fsum(abs(c)*node['record'].combined_bound_value
                         for c,node in zip(data.quadrature.coefficients,action['positions'],strict=True))
                  for f,action in zip(factors,data.host_actions['applications'],strict=True))
    assert planned['trotter_synthesis'].fact.value.value == pytest.approx(stored,rel=4*np.finfo(float).eps)
    assert stored > 0


@pytest.mark.parametrize('options',[dict(hamiltonian_evolution_backend='trotter_error_budgeted')])
def test_durable_classical_source_run_reopens_from_a_small_manifest(options,tmp_path):
    """Plan-size data lives in its own archive files, so run.json stays a small manifest.

    The default budgeted source case stores host actions for 204 k-nodes and
    nine applications. load_run
    reads run.json under a one MiB cap before the journal supplies the data
    allowance. run.json therefore only names the selection file, and the
    LCHS selection in turn only names the file of its native numerical data.
    What stays in the selection is the Plan record, the input manifests and
    the Method, which this test bounds at 128 KiB.
    """
    import json
    from nwqlib import load_run
    from nwqlib._prepared_execution import Run
    problem = LinearDynamics(A=[[.3,.1],[.1,.5]],initial_state=[1,0],time=.2,source=[.1,-.2j])
    chosen = plan(problem,method=LCHS(**options),execution='classical')
    path = tmp_path/'run'
    with Run(chosen,directory=path) as run:
        result = run.wait()
    assert (path/'run.json').stat().st_size < 64*1024
    manifest = json.loads((path/'run.json').read_text())
    selection = path/manifest['selection']
    assert selection.stat().st_size < 128*1024
    native = json.loads(selection.read_text())['selected']['native']
    assert type(native) is str and (path/native).is_file()
    with load_run(path,backend=None) as reopened:
        # A source Plan binds no homogeneous coefficient plan, at planning or on loading.
        assert reopened.plan._native['coefficient_plan'] is None
        again = reopened.wait()
    assert again.content_id == result.content_id
    np.testing.assert_array_equal(again.solution,result.solution)


def test_dense_host_admits_every_node_exponential_before_planning_succeeds(monkeypatch):
    # Each dense node evolution comes from one Hermitian eigensystem of
    # k L + H, 8 D**3 units each (time_independent_terms.
    # _spectral_host_requirements, which also prices the generator passes,
    # projections and phase actions). A max_select_work of 8 D**3 per node
    # therefore lies below the admitted law and must refuse the Plan. The
    # default limit admits it.
    l_part = np.array([[.4, .1, 0, 0], [.1, .3, .1, 0], [0, .1, .35, .05], [0, 0, .05, .3]])
    h_part = np.diag([.1, .1, .1], 1) + np.diag([.1, .1, .1], -1)
    problem = LinearDynamics(A=l_part + 1j*h_part, initial_state=[1., 0., 0., 0.], time=.1)
    grid = ProviderConfig(implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": 0})
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    admitted = plan(problem, method=LCHS(k_quadrature=grid), execution="classical")
    work = dict(admitted.reconstruction.classical_work)
    formed = []
    original = terms._node_eigensystem
    monkeypatch.setattr(terms, "_node_eigensystem", lambda *a: formed.append(a[0]) or original(*a))
    solve(admitted)
    # The two-qubit grid has four k nodes, one eigensystem each and no
    # exponential, and planning charges every eigensystem that the host forms.
    assert work["eigh_calls"] == len(formed) == 4 and work["dense_expm_calls"] == 0
    with pytest.raises(ValueError, match="max_select_work"):
        plan(problem, method=LCHS(k_quadrature=grid, max_select_work=8*4**3*work["eigh_calls"]), execution="classical")


def _aer_run_counter(monkeypatch):
    import qiskit_aer
    calls = []
    original = qiskit_aer.AerSimulator.run
    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(qiskit_aer.AerSimulator, "run", counted)
    return calls


def _trotter_method(bits=2):
    return LCHS(hamiltonian_evolution_backend="trotter", k_quadrature=ProviderConfig(
        implementation="signed_binary_uniform", parameters={"num_qubits": bits, "lsb_position": -1}))


def test_exact_lchs_scalar_acquires_one_simulation_and_reloads_without_replay(tmp_path, monkeypatch):
    """An exact L = 4 Pauli observable needs one Aer simulation, and reanalysis after load_run none."""
    import nwqlib
    from nwqlib import NormalizedExpectation
    from nwqlib.backends import AerBackend
    from nwqlib._prepared_execution import Run
    from nwqlib.operators import ingest_pauli
    terms = (("ZI", .5), ("IZ", .3), ("XX", .2), ("YY", -.1))
    output = NormalizedExpectation(observable=ingest_pauli(terms, num_qubits=2))
    problem, method = LinearDynamics(A=np.diag([.1, .2, .3, .4]), initial_state=[1., 2., 3., 4.], time=.1), _trotter_method()
    chosen = plan(problem, method=method, output=output, execution="quantum", seed=7)
    assert [experiment.name for experiment in chosen.experiments] == ["projected_moments"]
    calls = _aer_run_counter(monkeypatch)
    with Run(chosen, directory=tmp_path / "run", progress=False) as run:
        result = chosen.method.execute(chosen, run=run)
    assert len(calls) == 1 and result.reduction is not None and result.value is not None
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as reopened:
        again = reopened.plan.method.execute(reopened.plan, run=reopened)
    assert len(calls) == 1 and again.value == result.value and again.reduction == result.reduction
    solved = solve(chosen)
    solved.save(tmp_path / "result")
    assert len(calls) == 2 and solved.value == result.value
    assert nwqlib.load_result(tmp_path / "result").reduction == solved.reduction and len(calls) == 2
    # The Aer receipt lists only "amplitude-derived masses", and resolving
    # that label leaves a qualified saved-state budget.
    from nwqlib._quantum_readout import PROJECTED_MASS_EXCLUSIONS
    [receipt] = solved.data.receipts
    assert receipt.probability_window_exclusions == PROJECTED_MASS_EXCLUSIONS
    assert receipt.saved_state_error(PROJECTED_MASS_EXCLUSIONS)[0] is not None


def test_saved_exact_lchs_result_loads_in_a_fresh_process(tmp_path):
    """load_result in a new interpreter resolves the saved reduction's reducer before any LCHS module loads."""
    import json
    import os
    import subprocess
    import sys
    import nwqlib
    from nwqlib import NormalizedExpectation
    from nwqlib.operators import ingest_pauli
    output = NormalizedExpectation(observable=ingest_pauli((("ZI", .5), ("XX", .2)), num_qubits=2))
    problem = LinearDynamics(A=np.diag([.1, .2, .3, .4]), initial_state=[1., 2., 3., 4.], time=.1)
    result = solve(plan(problem, method=_trotter_method(), output=output, execution="quantum", seed=7))
    assert result.reduction is not None and result.value is not None
    result.save(tmp_path / "result")
    fields = ("value", "norm_squared", "numerator", "numerator_frame", "unavailable", "content_id")
    source = (f"import json, nwqlib\nloaded = nwqlib.load_result({str(tmp_path / 'result')!r})\n"
              f"print(json.dumps([nwqlib.__file__] + [getattr(loaded, name) for name in {fields!r}]))")
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=dict(os.environ),
                               timeout=300)
    assert completed.returncode == 0, completed.stderr
    origin, *loaded = json.loads(completed.stdout.splitlines()[-1])
    assert origin == nwqlib.__file__
    assert loaded == [getattr(result, name) for name in fields]


def _reference_slice(chosen):
    """The bound Plan circuit in Statevector, with the physical prefix by explicit bit enumeration."""
    from qiskit.quantum_info import Statevector
    from nwqlib.blocks.lowering import lower_qiskit
    _, construction = chosen.resolve("projected_moments")._selected_construction(chosen)
    selected = {record.content_id for record in construction.selections}
    circuit = lower_qiskit(construction, blocks=tuple(block for block in chosen.blocks
                                                      if block.record.content_id in selected)).circuit
    state = Statevector(circuit).data
    rec = chosen.reconstruction
    success, coordinates = rec.success_bits, rec.system_bits
    algorithm = physical = 0.
    vector = np.zeros(rec.encoded_dimension, dtype=complex)
    for index, amplitude in enumerate(state):
        if any((index >> bit) & 1 for bit in success):
            continue
        algorithm += abs(amplitude)**2
        coordinate = sum(((index >> bit) & 1) << place for place, bit in enumerate(coordinates))
        if coordinate < rec.dimension:
            vector[coordinate] = amplitude
            physical += abs(amplitude)**2
    return vector, algorithm, physical


def _pauli_matrix(label):
    one = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    matrix = np.eye(1)
    for axis in label:  # qubit zero is the rightmost letter, the last Kronecker factor
        matrix = np.kron(matrix, one[axis])
    return matrix


def test_exact_lchs_outputs_match_an_independent_physical_slice_reference():
    """Projected p, p_alg and q against the same bound circuit in Statevector.

    The receipts exclude supplied-state operations, so this uses the unit-scale
    regression contract atol=2e-12, rtol=0 for raw masses and q with C <= 1,
    propagated to the normalized value by the quotient rule and to physical
    outputs by the recovery. It is not a native error certificate.
    """
    from nwqlib import NormalizedExpectation, NormSquared
    from nwqlib._quantum_readout import pair_value
    matrix = np.diag([.3, -.2, .1]) + .05 * (np.eye(3, k=1) + np.eye(3, k=-1))
    generator = np.diag([.1, .25, .4]) + .6 * (np.eye(3, k=1) + np.eye(3, k=-1))
    problem = LinearDynamics(A=generator, initial_state=[1., 2., 3.], time=.8)
    method = _trotter_method()
    tolerance = 2e-12
    for output in (NormalizedExpectation(observable=matrix), QuadraticForm(observable=matrix), NormSquared()):
        chosen = plan(problem, method=method, output=output, execution="quantum", seed=7)
        result = solve(chosen)
        vector, algorithm, physical = _reference_slice(chosen)
        rec = chosen.reconstruction
        terms = rec.terms if not isinstance(output, NormSquared) else ()
        coefficient_norm = sum(abs(c) for _, c in terms)
        assert coefficient_norm <= 1
        observable = sum((c * _pauli_matrix(label) for label, c in terms), np.zeros((rec.encoded_dimension,) * 2))
        numerator = float(np.vdot(vector, observable @ vector).real)
        stats = result.reduction
        p, q = pair_value(stats.physical_mass), pair_value(stats.numerator)
        assert abs(float(p) - physical) <= tolerance and abs(float(pair_value(stats.success_mass)) - algorithm) <= tolerance
        assert abs(float(q) - numerator) <= tolerance and physical > .1
        gamma2 = rec.recovery.as_float()**2
        assert algorithm - physical > 1e-11  # a nonzero dummy population is excluded from p
        if isinstance(output, NormalizedExpectation):
            bound = (tolerance + abs(numerator) / physical * tolerance) / (physical - tolerance) + 2**-53 * abs(result.value)
            assert abs(result.value - numerator / physical) <= bound
        elif isinstance(output, QuadraticForm):
            assert abs(result.value - gamma2 * numerator) <= gamma2 * tolerance * (1 + 1e-12)
        else:
            assert abs(result.value - gamma2 * physical) <= gamma2 * tolerance * (1 + 1e-12)


def test_exact_reduction_and_sampled_grouping_obey_the_readout_work_limit():
    """Exact reduction and sampled grouping refuse work above max_readout_work."""
    import json
    from nwqlib import NormalizedExpectation
    from nwqlib._quantum_readout import projected_requirements
    from nwqlib.operators import ingest_pauli
    problem = LinearDynamics(A=np.diag([.1, .2, .3, .4]), initial_state=[1., 2., 3., 4.], time=.1)

    def chosen(limit, terms=(("ZI", .5), ("XX", .2)), shots=None):
        method = _trotter_method().revise(max_readout_work=limit)
        output = NormalizedExpectation(observable=ingest_pauli(terms, num_qubits=2))
        return plan(problem, method=method, output=output, execution="quantum", shots=shots, seed=7)

    probe = chosen(2_000_000_000)
    point = probe.experiments[0].observation.positions[0]
    width = len(probe.reconstruction.success_bits)+len(probe.reconstruction.system_bits)
    _, work = projected_requirements(json.loads(point.parameters), width)
    # Construction keeps its own ledger; the exact route records no grouping comparison.
    assert probe.reconstruction.grouping_comparisons == 0 and probe.reconstruction.select_work > 1554
    assert solve(chosen(work)).value is not None
    with pytest.raises(ValueError, match=r"LCHS\.max_readout_work"):
        solve(chosen(work - 1))
    # ZI, XX and YY pairwise fail qubit-wise commutation: three groups after 1 + 2 comparisons.
    labels = (("ZI", .5), ("XX", .2), ("YY", .1))
    sampled = chosen(3, labels, shots=100)
    assert sampled.reconstruction.grouping_comparisons == 3 and len(sampled.reconstruction.groups) == 3
    # The 15 two-qubit labels also exceed a single grouping comparison.
    labels = tuple((a + b, 1 / (1 + i)) for i, (a, b) in enumerate(
        (a, b) for a in "IXYZ" for b in "IXYZ" if a + b != "II"))
    with pytest.raises(ValueError, match=r"LCHS\.max_readout_work=1\b"):
        chosen(1, labels, shots=100)
