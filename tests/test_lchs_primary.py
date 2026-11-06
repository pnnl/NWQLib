"""Natural LCHS input admission, acquired populations and numerical work ownership."""

import weakref
import numpy as np
import pytest
from nwqlib import LinearDynamics,NormSquared,QuadraticForm,plan,solve
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


def test_sampled_pair_rows_are_one_range_axis_counted_without_enumeration(monkeypatch):
    """A sampled-pair Plan holds its B rows as one Setting with a ``row`` RangeAxis, whatever B is.

    B = 40,000 rows plan at the default LCHS limits. With one Setting and Experiment per row, about 32 kept
    field slots each, this B exceeded the default admission bound of 1e6 slots. The breakdown's settings
    count, the setting times the axis points, times its repetitions equals the estimate's shots, and neither
    enumerates the axis (`Plan._static_points`) nor builds a circuit.
    """
    import qiskit
    from nwqlib import NormSquared, estimate
    from nwqlib.algorithms.lchs.quantum_resources import resource_breakdown
    from nwqlib.core.planning import Plan
    from nwqlib.ir import RangeAxis
    from nwqlib.operators import ingest_pauli

    linear = LinearDynamics(A=ingest_pauli([("Z", .4j), ("X", .1)], num_qubits=1), initial_state=[.6, .8], time=1.)
    method = LCHS(observable_estimator="sampled_pairs", hamiltonian_evolution_backend="trotter", trotter_steps=1,
                  pair_repeats=2)
    chosen = plan(linear, method=method, output=NormSquared(), shots=80_000, seed=3)
    (batch,) = [item.node for item in chosen.construction.program.definitions if item.id == "pairs"]
    assert len(chosen.experiments) == len(batch.settings) == 1
    assert batch.axes == (RangeAxis(parameter="row", start=0, stop=40_000),) and batch.repetitions == 2
    assert not batch.settings[0].bindings and chosen.reconstruction.pair_rows.clusters == 40_000
    built = []
    original = qiskit.QuantumCircuit.__init__
    monkeypatch.setattr(qiskit.QuantumCircuit, "__init__",
                        lambda self, *a, **k: (built.append(1), original(self, *a, **k))[1])
    monkeypatch.setattr(Plan, "_static_points", lambda *a: pytest.fail("the row axis was enumerated"))
    (entry,) = resource_breakdown(chosen)
    folded = estimate(chosen)
    shots, settings = folded.quantity("shots").fact.value, folded.quantity("settings").fact.value
    assert (entry["settings"], entry["repetitions"]) == (40_000, 2)
    assert (settings.numerator, settings.denominator) == (40_000, 1)
    assert (shots.numerator, shots.denominator) == (entry["settings"] * entry["repetitions"], 1) == (80_000, 1)
    assert not built


def test_sampled_pair_row_points_resolve_and_forecast_one_by_one(monkeypatch):
    """Each row is one static point ``pairs[row=i]``, resolved with its row and forecast as a planned execution.

    `Plan._point_realization`, which analysis uses without building the point's Program, equals
    `Plan.resolve` of the same point. The profile forecast resolves every row point of this static Plan, and
    lists the axis as unpredicted instead when the Method declares its experiments query templates.
    """
    from nwqlib import NormSquared
    from nwqlib.backends.assessment import _profile_points
    from nwqlib.ir import Binding
    from nwqlib.operators import ingest_pauli

    linear = LinearDynamics(A=ingest_pauli([("Z", .4j), ("X", .1)], num_qubits=1), initial_state=[.6, .8], time=1.)
    chosen = plan(linear, method=LCHS(observable_estimator="sampled_pairs", hamiltonian_evolution_backend="trotter",
                                      trotter_steps=1, pair_repeats=2), output=NormSquared(), shots=8, seed=3)
    points = tuple(chosen._static_points())
    assert [point.key for point in points] == [f"pairs[row={row}]" for row in range(4)]
    assert [point.bindings for point in points] == [(Binding(parameter="row", value=row),) for row in range(4)]
    for point in points:
        resolved = chosen.resolve("pairs", bindings=point.bindings)
        assert chosen._point_realization("pairs", point.bindings) == resolved
    forecast, unpredicted = _profile_points(chosen)
    assert [item.bindings for item in forecast] == [point.bindings for point in points] and not unpredicted
    # The row axis takes every row, so the Program cannot fix one.
    with pytest.raises(ValueError, match="Program bindings conflict with range axes"):
        chosen.construction.program.bind(row=2)
    monkeypatch.setattr(LCHS, "query_catalog_reason", lambda self: "each row is a query")
    forecast, unpredicted = _profile_points(chosen)
    assert forecast == () and unpredicted == ("pairs: range-axis points of a query template are not expanded "
                                              "for forecasting",)


def test_sampled_input_preparations_are_shared_and_rebuilt_after_archive(tmp_path, monkeypatch):
    from qiskit.quantum_info import Statevector
    from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan
    from nwqlib.algorithms.lchs import native, sampled_native
    from nwqlib.blocks.lowering import lower_qiskit
    from nwqlib.blocks.records import SelectedConstruction
    from nwqlib.operators import ingest_pauli

    problem = LinearDynamics(
        A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
        initial_state=[.6, .8j], source=[.3, -.4j], time=.7)
    method = LCHS(observable_estimator="sampled_pairs", pair_repeats=4,
                  hamiltonian_evolution_backend="trotter", trotter_steps=2, pair_source_time="continuous")
    chosen = plan(problem, method=method, output=QuadraticForm(observable=ingest_pauli([("X", 1.)], num_qubits=1)),
                  shots=64, seed=3)
    preparations, controls = [], []
    prepare, control = native.state_preparations, sampled_native.controlled

    def prepare_once(*args, **kwargs):
        preparations.append(1)
        return prepare(*args, **kwargs)

    def control_once(gate, *args, **kwargs):
        if gate.name.startswith("prep_"):
            controls.append((gate.name, kwargs.get("ctrl_state")))
        return control(gate, *args, **kwargs)

    monkeypatch.setattr(native, "state_preparations", prepare_once)
    monkeypatch.setattr(sampled_native, "controlled", control_once)

    def states(selected):
        values = []
        for row in range(selected.reconstruction.pair_rows.clusters):
            program = selected.construction.program.select_experiment("pairs", 0, row=row)
            construction = SelectedConstruction(program=program, selections=selected.construction.selections)
            circuit = lower_qiskit(construction, blocks=selected.blocks).circuit
            values.append(Statevector(circuit.remove_final_measurements(inplace=False)).data)
        return values

    before = states(chosen)
    assert len(preparations) == 1
    assert controls and len(controls) == len(set(controls))
    files = ArchiveFiles(tmp_path, 4_000_000)
    saved = save_plan(chosen, files)
    restored = load_plan(saved, ArchiveFiles(tmp_path, 4_000_000))
    controls.clear()
    after = states(restored)
    assert len(preparations) == 2
    assert controls and len(controls) == len(set(controls))
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-13)


def test_dense_host_streams_only_requested_operator_and_result_reanalysis_is_free(monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    from scipy.linalg import expm
    matrix=np.array([[-.2,.03j],[.03j,.4]])
    problem=LinearDynamics(A=matrix,initial_state=[1,1j],time=.1)
    method=LCHS(k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen=plan(problem,method=method,execution='classical')
    native=chosen._native['native_data']
    quad,dense=native.quadrature,native.dense
    expected=sum(c*expm(-.1j*(k*dense.l_part+dense.h_part)) for k,c in zip(quad.k_nodes,quad.coefficients,strict=True))
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
    rotate=host_pf._word_action
    monkeypatch.setattr(host_pf,'_word_action',lambda *a,**k: executed.append(a[1:4]) or rotate(*a,**k))
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


def _pf_positions(chosen):
    """The saved (route, work, step count) of every classical PF position, in application order."""
    return [(node['route'], node['work'], node['record'].step_count)
            for action in chosen._native['native_data'].host_actions['applications'] for node in action['positions']]


def test_host_action_route_follows_registered_work_and_the_byte_limit_only_admits_it(tmp_path, monkeypatch):
    """The cheaper registered PF route is chosen on every computer, and a byte limit below its arrays refuses it.

    For D = 32 and r = 1024 steps, binary powering (ten 32x32 products) costs about 3.3e5 work units and the
    vector route at least 4.8e5, so every position saves dense_power. Its arrays, beside the tables already
    held, are the planning peak W that the kernel workspace publishes. With max_bytes=None, physical memories
    of W and 2**40 bytes give the same Plan. W - 1 bytes refuse it before any action and name the route and W,
    although the vector route, with 9792 bytes of its own instead of 102400, would fit. An explicit
    max_bytes=W gives the same positions, facts, work and bitwise solution, and its Plan identity differs
    because the Method records the cap. Each Plan reopens and runs its saved routes without selecting them
    again, including a Plan saved with the vector route that a capacity fallback once chose.
    """
    import nwqlib._limits as limits
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib.algorithms.lchs import host_pf
    matrix = SparsePauliOp.from_list([("ZIIII", .2), ("XXIII", .25j), ("IIIIY", -.15j), ("IIIII", .1)]).to_matrix()
    problem = LinearDynamics(A=matrix, initial_state=np.full(32, 32**-.5), time=.3)
    grid = ProviderConfig(implementation='signed_binary_uniform', parameters={'num_qubits': 2, 'lsb_position': -1})

    def planned(memory, **limit):
        monkeypatch.setattr(limits, "physical_memory_bytes", lambda: memory)
        return plan(problem, method=LCHS(hamiltonian_evolution_backend='trotter', trotter_steps=1024,
                                         k_quadrature=grid, **limit), execution='classical', seed=5)

    chosen = planned(2**40)
    positions = _pf_positions(chosen)
    assert {route for route, _, _ in positions} == {'dense_power'}
    kernel, = chosen.construction.kernels
    peak = kernel.workspace[0].bytes
    assert (host_pf._route_requirements(32, 1024, 1, 'vector')[1],
            host_pf._route_requirements(32, 1024, 1, 'dense_power')[1]) == (9792, 102400)
    edge = planned(peak)
    assert edge.content_id == chosen.content_id and _pf_positions(edge) == positions
    refusal = f"selected Pauli action route dense_power needs about {peak} bytes, exceeding "
    with pytest.raises(ValueError, match=refusal + rf"the physical memory \({peak - 1} bytes\)$"):
        planned(peak - 1)
    with pytest.raises(ValueError, match=refusal + f"max_bytes={peak - 1}$"):
        planned(2**40, max_bytes=peak - 1)
    explicit = planned(2**40, max_bytes=peak)
    assert _pf_positions(explicit) == positions
    assert [item.fact.value for item in explicit.facts] == [item.fact.value for item in chosen.facts]
    assert explicit.reconstruction.classical_work == chosen.reconstruction.classical_work
    fields = {"max_bytes", "content_id"}
    assert explicit.method.model_dump(exclude=fields) == chosen.method.model_dump(exclude=fields)
    assert explicit.content_id != chosen.content_id
    solution = solve(chosen).solution
    np.testing.assert_array_equal(solve(explicit).solution, solution)
    law = host_pf._route_requirements
    _price_out_dense_route(monkeypatch, host_pf)
    old = planned(2**40)
    assert {route for route, _, _ in _pf_positions(old)} == {'vector'}
    monkeypatch.setattr(host_pf, "_route_requirements", law)
    # The two routes evaluate the same product formula in different orders.
    assert not np.array_equal(solve(old).solution, solution)
    monkeypatch.setattr(host_pf, "select_actions", lambda *a, **k: pytest.fail("a reopened Plan selected routes"))
    for index, saved in enumerate((chosen, explicit, old)):
        (tmp_path / str(index)).mkdir()
        files = ArchiveFiles(tmp_path / str(index), 10**8)
        restored = load_plan(save_plan(saved, files), files)
        assert restored.content_id == saved.content_id and _pf_positions(restored) == _pf_positions(saved)
        np.testing.assert_array_equal(solve(restored).solution, solve(saved).solution)


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


# A padded three-dimensional problem on a two-bit grid. Each outcome key reads (system coordinate)(success bits): shots
# with success bits 00 and a coordinate below three are kept, coordinate 3 is the dummy of the padded register, and a
# nonzero success bit is a failed LCU block. Requested 8; returned 5; selected 3.
_PARTIAL_COUNTS = {"0000": 2, "1000": 1, "1100": 1, "0001": 1}


def _counts_problem():
    return LinearDynamics(A=np.diag([.1, .2, .3]), initial_state=[1., 2., 3.], time=.1)


def _returning(monkeypatch, counts):
    """Let the real Aer path run and hand the Run a backend's own counts, as a provider returning part of a request does."""
    from dataclasses import replace
    from nwqlib.backends import qiskit_aer
    original = qiskit_aer._submit_aer_execution
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution",
                        lambda prepared: replace(original(prepared), raw_output={"counts": counts}))


def test_coherent_norm_squared_publishes_requested_returned_and_selected_shots(tmp_path, monkeypatch):
    """A saved, reopened and reanalysed NormSquared keeps three distinct counts, each from the backend's own histogram.

    Expected values come from the histogram above, not from the Plan or from a mass. The estimate is
    Gamma**2*selected/returned and its Hoeffding radius uses the returned count (`analysis._counts_radius`), so neither may
    read the 8 requested shots. The same counts read as Samples give the same three numbers, because that output keeps the
    same physical slice.
    """
    from math import log, sqrt
    import nwqlib
    from nwqlib import Samples
    problem = _counts_problem()
    _returning(monkeypatch, _PARTIAL_COUNTS)
    chosen = plan(problem, method=_trotter_method(), output=NormSquared(), shots=8, seed=7)
    assert chosen.reconstruction.settings[0].physical_projection
    result = solve(chosen)
    reloaded = nwqlib.load_result(result.save(tmp_path/"partial"))
    gamma_squared = chosen.reconstruction.recovery.as_float()**2
    for item in (result, reloaded, reloaded.analyze()):
        assert (item.submitted_shots, item.returned_shots, item.selected_shots) == (8, 5, 3)
        assert item.value == pytest.approx(gamma_squared*3/5, rel=1e-14, abs=0)
        radius = next(entry.fact.value.value for entry in item.facts if entry.fact.quantity == "sampling")
        assert radius == pytest.approx(gamma_squared*sqrt(log(2/chosen.method.moment_failure_probability)/10), rel=1e-14, abs=0)
    assert reloaded.content_id == result.content_id
    samples = solve(plan(problem, method=_trotter_method(), output=Samples(), shots=8, seed=7))
    assert (samples.submitted_shots, samples.returned_shots, samples.selected_shots) == (8, 5, 3)


def test_coherent_norm_squared_with_no_returned_shot_is_unavailable_and_says_so(tmp_path, monkeypatch):
    """A backend that returns nothing leaves the mass undefined: the zero counts are facts and the estimate is not zero."""
    import nwqlib
    _returning(monkeypatch, {})
    result = solve(plan(_counts_problem(), method=_trotter_method(), output=NormSquared(), shots=8, seed=7))
    for item in (result, nwqlib.load_result(result.save(tmp_path/"none"))):
        assert (item.submitted_shots, item.returned_shots, item.selected_shots) == (8, 0, 0)
        assert item.value is None and item.unavailable is not None
        sampling = next(entry for entry in item.facts if entry.fact.quantity == "sampling")
        assert sampling.fact.availability == "unknown"


def test_shot_fields_stay_empty_where_no_single_count_describes_the_output():
    """Exact readout acquires no shots, and a coherent QuadraticForm has one count per setting, in its groups."""
    from nwqlib import QuadraticForm as Quadratic
    problem = _counts_problem()
    exact = solve(plan(problem, method=_trotter_method(), output=NormSquared(), seed=7))
    assert (exact.submitted_shots, exact.returned_shots, exact.selected_shots) == (None, None, None)
    grouped = solve(plan(problem, method=_trotter_method(), output=Quadratic(observable=np.diag([1., -1., 1.])), shots=64, seed=7))
    assert (grouped.submitted_shots, grouped.returned_shots, grouped.selected_shots) == (None, None, None)
    assert all(item.returned_shots == 64 for item in grouped.groups)


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


def test_exact_reduction_runs_without_a_work_limit_and_sampled_grouping_obeys_it():
    """The exact scalar reduction runs under any max_readout_work, and sampled grouping refuses comparisons above it."""
    from nwqlib import NormalizedExpectation
    from nwqlib.operators import ingest_pauli
    problem = LinearDynamics(A=np.diag([.1, .2, .3, .4]), initial_state=[1., 2., 3., 4.], time=.1)

    def chosen(limit, terms=(("ZI", .5), ("XX", .2)), shots=None):
        method = _trotter_method().revise(max_readout_work=limit)
        output = NormalizedExpectation(observable=ingest_pauli(terms, num_qubits=2))
        return plan(problem, method=method, output=output, execution="quantum", shots=shots, seed=7)

    probe = chosen(None)
    # Construction keeps its own ledger; the exact route records no grouping comparison.
    assert probe.reconstruction.grouping_comparisons == 0 and probe.reconstruction.select_work > 1554
    unbounded = solve(probe).value
    assert unbounded is not None and solve(chosen(1)).value == unbounded
    # ZI, XX and YY pairwise fail qubit-wise commutation: three groups after 1 + 2 comparisons.
    labels = (("ZI", .5), ("XX", .2), ("YY", .1))
    sampled = chosen(3, labels, shots=100)
    assert sampled.reconstruction.grouping_comparisons == 3 and len(sampled.reconstruction.groups) == 3
    # The 15 two-qubit labels also exceed a single grouping comparison.
    labels = tuple((a + b, 1 / (1 + i)) for i, (a, b) in enumerate(
        (a, b) for a in "IXYZ" for b in "IXYZ" if a + b != "II"))
    with pytest.raises(ValueError, match=r"LCHS\.max_readout_work=1\b"):
        chosen(1, labels, shots=100)


def test_compact_26_qubit_lchs_plans_estimates_and_reloads_without_system_arrays(tmp_path, monkeypatch):
    """A 26-qubit Cartesian Pauli generator plans, estimates and reloads with no 2**26-entry object.

    A = L + iH with L = I + 0.01 sum_k Z_k Z_k+1 + 0.01 sum_{even k} X_k X_k+1
    (25 + 13 words) and H = 0.02 sum_k Z_k on n = 26 qubits, from an
    occupation state, for the grouped observable Z_0 + 0.5 X_0 X_1 with 1000
    shots. The Pauli enclosure c_I -+ sum_{P != I} |c_P| of L is
    [1 - 0.38, 1 + 0.38], so the plan takes out the known decay 0.62 and
    selects its grid for the bound 0.76 = 1.38 - 0.62 on ||L + s I|| with
    s = -0.62, both from the Pauli coefficients alone. Planning, the resource estimate and an
    archive round trip pass no limit and the Method is the literal LCHS(), so the
    defaults apply, including the backend chosen from the compact input. No statevector,
    operator matrix or dense Pauli matrix is formed, the plan holds no dense
    arrays, the system register keeps all 26 qubits, and the peak traced by
    tracemalloc, which NumPy reports its buffers to, stays below 2**26 bytes,
    the size of the smallest array with one entry per basis state. Planning
    only; no circuit is prepared or run.
    """
    import tracemalloc
    from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
    import nwqlib
    from nwqlib import NormalizedExpectation
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib.operators import ingest_cartesian, ingest_pauli
    from nwqlib.problems import ingest_occupation

    n = 26

    def word(letters):
        positions = ["I"]*n
        for qubit, letter in letters:  # qubit zero is the rightmost letter
            positions[n-1-qubit] = letter
        return "".join(positions)

    def forbidden(*args, **kwargs):
        pytest.fail("compact 26-qubit LCHS formed a full-state or dense operator object")

    for owner, name in ((Statevector, "__init__"), (Operator, "__init__"), (SparsePauliOp, "to_matrix")):
        monkeypatch.setattr(owner, name, forbidden)
    dissipative = ingest_pauli([(word(()), 1.)] + [(word(((k, "Z"), (k+1, "Z"))), .01) for k in range(n-1)]
                               + [(word(((k, "X"), (k+1, "X"))), .01) for k in range(0, n-1, 2)], num_qubits=n)
    hamiltonian = ingest_pauli([(word(((k, "Z"),)), .02) for k in range(n)], num_qubits=n)
    problem = LinearDynamics(A=ingest_cartesian(dissipative, hamiltonian),
                             initial_state=ingest_occupation([int(k < 2) for k in range(n)], num_qubits=n), time=.5)
    output = NormalizedExpectation(observable=ingest_pauli([(word(((0, "Z"),)), 1.), (word(((0, "X"), (1, "X"))), .5)],
                                                           num_qubits=n))
    tracemalloc.start()
    try:
        chosen = nwqlib.plan(problem, method=LCHS(), output=output, execution="quantum", shots=1000, seed=5)
        nwqlib.estimate(chosen)
        files = ArchiveFiles(tmp_path, 4_000_000_000)
        restored = load_plan(save_plan(chosen, files), ArchiveFiles(tmp_path, 4_000_000_000))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2**26
    rec, data = chosen.reconstruction, chosen._native["native_data"]
    assert data.dense is None and len(rec.system_bits) == n and rec.encoded_dimension == 2**n
    # The default backend "auto" chose the compact coherent route once; the reloaded Plan keeps that choice.
    assert rec.selected_backend == data.backend == "qsp_block_encoding"
    # 1 - (25 + 13)*0.01 is the exact structural lower bound, up to the
    # outward rounding of one binary64 step.
    assert rec.psd_shift == 0 and rec.scalar_decay == pytest.approx(.62, rel=1e-15)
    assert rec.l_norm == pytest.approx(.76, rel=1e-15) and rec.psd_premise == "certified_bound"
    assert restored.content_id == chosen.content_id and restored.reconstruction == rec


@pytest.mark.parametrize("terms,tolerance", (
    ([("I", 1.+.3j)], 1e-13),
    ([("X", .2j)], 1e-4),
))
def test_compact_degenerate_inputs_reduce_to_closed_forms(terms, tolerance):
    """A scalar A and a zero L stay legal as general complex Pauli input.

    ingest_pauli splits A = sum_P c_P P into L = sum_P Re(c_P) P and
    H = sum_P Im(c_P) P, and the expected u(T) = exp(-T A) u0 is exact for
    each case. A = (1 + .3i) I gives exp(-(1 + .3i) T) u0, which the plan takes
    out as known decay and scalar phase and computes up to binary64 rounding;
    a swapped or sign-flipped split changes it by O(1). A = .2i X has L = 0,
    where the LCHS integral collapses to the kernel's unit mass and only the
    QSP approximation of exp(-i T H) remains (the tolerance of
    test_lchs_qsp_source for the same method).
    """
    from scipy.linalg import expm
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.operators import ingest_pauli

    initial, time = np.array([.6, .8j]), .5
    method = LCHS(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=2,
                  lchs_kernel=ProviderConfig(implementation='cauchy_density'),
                  k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                              parameters={'num_qubits': 2, 'lsb_position': -1}))
    chosen = plan(LinearDynamics(A=ingest_pauli(terms, num_qubits=1), initial_state=initial, time=time),
                  method=method)
    expected = expm(-time*SparsePauliOp.from_list(terms).to_matrix())@initial
    np.testing.assert_allclose(solve(chosen).solution, expected, rtol=0, atol=tolerance)


def test_compact_band_dissipative_part_matches_independent_finite_lchs_sum():
    """A periodic band L with a Pauli H plans, builds and solves on the compact route.

    L = 2 I + beta S + conj(beta) S^-1 with beta = -.5 + .1i is Hermitian, and
    the band enclosure 2 -+ 2|beta| gives the known decay c = 2 - 2|beta|.
    The expected value is the finite LCHS sum on the same 4-node Cauchy grid,
    written here from explicit matrices with the cyclic shift
    S|x> = |x+1 mod 4>: sum_j c_j exp(-c T) exp(-i T (k_j (L - c I) + H)) u0.
    The tolerance is that of test_lchs_qsp_source for the QSP approximation
    of each branch.
    """
    from scipy.linalg import expm
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.operators import ingest_band, ingest_cartesian, ingest_pauli

    beta, time = -.5+.1j, .4
    method = LCHS(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=2,
                  lchs_kernel=ProviderConfig(implementation='cauchy_density'),
                  k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                              parameters={'num_qubits': 2, 'lsb_position': -1}))
    A = ingest_cartesian(ingest_band([0, 1, -1], [2., beta, beta.conjugate()], num_qubits=2),
                         ingest_pauli([("ZI", .3)], num_qubits=2))
    initial = np.array([.6, 0, .8j, 0])
    chosen = plan(LinearDynamics(A=A, initial_state=initial, time=time), method=method)
    decay = 2-2*abs(beta)
    assert chosen.reconstruction.scalar_decay == pytest.approx(decay, rel=1e-15)
    shift = np.roll(np.eye(4), 1, axis=0)
    L = 2*np.eye(4) + beta*shift + beta.conjugate()*shift.T
    H = SparsePauliOp.from_list([("ZI", .3)]).to_matrix()
    nodes = np.array([0., .5, -1., -.5])
    expected = sum(.5/(np.pi*(1+k**2))*np.exp(-decay*time)*(expm(-1j*time*(k*(L-decay*np.eye(4))+H))@initial)
                   for k in nodes)
    np.testing.assert_allclose(solve(chosen).solution, expected, rtol=0, atol=1e-4)


_SAMPLED_PAULI = {label: np.array(matrix, complex) for label, matrix in (
    ("I", [[1, 0], [0, 1]]), ("X", [[0, 1], [1, 0]]), ("Y", [[0, -1j], [1j, 0]]), ("Z", [[1, 0], [0, -1]]))}


def _sampled_suzuki(time, node, steps):
    """Ordered Suzuki-2 product X, Y, Z, Y, X of time*(node*.25 X - .15 Y + .2 Z) with ``steps`` steps."""
    pauli = _SAMPLED_PAULI
    coefficients = dict(X=.25 * node, Y=-.15, Z=.2)
    step = pauli["I"]
    for label, mu in (("X", .5), ("Y", .5), ("Z", 1.), ("Y", .5), ("X", .5)):
        angle = mu * time / steps * coefficients[label]
        step = (np.cos(angle) * pauli["I"] - 1j * np.sin(angle) * pauli[label]) @ step
    return np.linalg.matrix_power(step, int(steps))


@pytest.mark.parametrize("source, options, norm", (
    (None, dict(hamiltonian_evolution_backend="trotter", trotter_steps=2), 1.),
    (None, dict(hamiltonian_evolution_backend="trotter_error_budgeted", pair_step_buckets=3), 1.),
    ("mixed", dict(hamiltonian_evolution_backend="trotter_error_budgeted", pair_step_buckets=3), 1.),
    ("only", dict(hamiltonian_evolution_backend="trotter", trotter_steps=2), 1.),
    ("weak", dict(hamiltonian_evolution_backend="trotter", trotter_steps=2), 1.),
    ("mixed", dict(hamiltonian_evolution_backend="trotter", trotter_steps=2, pair_source_time="continuous"), 1.),
    (None, dict(hamiltonian_evolution_backend="trotter", trotter_steps=2,
                lchs_kernel=ProviderConfig(implementation="cauchy_density"),
                k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
                                            parameters={"num_qubits": 2, "lsb_position": -1072})), 1e300),
    (None, dict(hamiltonian_evolution_backend="trotter", trotter_steps=2,
                lchs_kernel=ProviderConfig(implementation="apel_berntson"),
                k_quadrature=ProviderConfig(implementation="tanh_sinh")), 1.),
    (None, dict(hamiltonian_evolution_backend="trotter_error_budgeted", pair_product_formula="richardson",
                pair_pf_allowance=.16), 4.),
    (None, dict(hamiltonian_evolution_backend="trotter_error_budgeted", trotter_order=1, pair_product_formula="ptsc",
                pair_pf_allowance=.16, pair_mu_cap=1.5, pair_taylor_cutoff=8), 4.),
))
def test_sampled_pair_rows_and_bias_bounds_match_dense_branches(source, options, norm):
    """Sampled-pair rows weigh each branch by a/p and the published bias bounds hold for dense branches.

    For A = L + iH with noncommuting L = .3 I + .25 X and H = .1 I + .2 Z -
    .15 Y, the initial state u0 = norm*(.6, .8i) and, for a source,
    b = norm*(.3, -.4i) (u0 = 0 for a source alone), the selected vector is
    v = sum_{i,j} M_i c_j exp(i phi) U(t_i, k_j) psi_i with psi_i the unit
    initial or source direction, phi = -t (k (.3 + s) + .1), the initial mass
    M_0 = ||u0|| exp(s T) and the Gauss-Legendre source masses
    M_q = w_q ||b|| exp(s (T - r_q)), or for a continuous source time the
    integral ||b|| int_0^T exp(s t) ... dt, here a 96-point Gauss-Legendre
    sum. Every saved branch magnitude times g must be |a|/(pi_i p_j) for the
    actual integer laws, which makes the pair scores unbiased; a continuous
    source branch has the constant M_src/pi_src, M_src = ||b|| expm1(s T)/s,
    whatever its drawn time. The exact pair-row mean
    sum_{x,y} conj(a_x) a_y <U~_x psi_x | O | U~_y psi_y>, with U~ the
    ordered Suzuki product and r = max(r_b(x), r_b(y)), must lie within the
    product-formula bias of v^dagger O v and ||v||**2, and those within the
    model bias of the physical u(T) = exp(-A T) u0 + int_0^T exp(-A (T - r)) b dr,
    which an augmented matrix exponential gives (LCHSPairBias). For a
    mixed-kind row, the lowered pair circuit must give
    Re(exp(i theta) <chi_x|P|chi_y>) with the physical relative phase of the
    two directions. Planning and estimation build no circuit, and the
    estimate counts B settings and B*m shots.

    The weak source b = 1e-18*(.3, -.4i) gives each of the eight default
    source masses a relative weight below 2**-52, so all eight round to zero
    counts and are raised to one. The application law must still sum to
    2**52 and keep at least one count on every positive mass; such a law
    exists, for example 2**52 - 8 counts on the initial mass.

    The registered grid signed_binary_uniform(2, lsb -1072) with the Cauchy
    density stores four subnormal coefficients c_j = 2**-1074, which the
    kernel law reads as the normal magnitudes 2**k*|c_j| with k = 52. With
    ||u0|| = 1e300 the branch weights are representable, and a row frame g
    that kept that lift or removed it twice would miss |a|/(pi_i p_j) by the
    factor 2**52, which a normalized expectation would not show. This row
    checks the coefficient lift and the phase association, with abs=0 so
    that an absolute tolerance cannot accept the tiny branch magnitudes. Its
    grid is no LCHS rule for exp(-A T), and at this norm the quadratic
    moments of u(T), of order norm**2, exceed binary64 even though u(T)
    itself can be representable. The model-bias comparison with the original
    ODE therefore runs only at norm 1, and this row does not test accuracy
    for the original ODE.

    The Apel-Berntson tanh-sinh row uses a rule that supplies its kernel
    values as an array, which planning reads for its byte charge
    (`sampled_plan._held_bytes`). Its coefficients c_j come from that rule, so
    the branch weights and both bias comparisons above apply to it.

    The two source-free norm-4 rows select a Richardson or PTSC recipe for
    pair_pf_allowance = .16 and check its published bias at planning only;
    no row circuit is built and no branch is evolved at the selected levels.
    These rows have no plain bucket steps, so the exact pair-mean
    containment above runs for the plain rows only. With S the sum of the
    application masses and beta = sum |c_j| over the nodes of positive
    count, B = S*beta bounds E|w|, and P = inner_bias*g**2 is the
    N bias in the physical frame, which must be at most .16 and at least a
    lower value formed here from the problem's coefficients. For
    Richardson (XS9, `sampled_recipes.select_richardson`) that value is
    B**2 times 4*(5/3)*max(1, A)**2*(A/s)**4 at levels (s, 2s) with
    s > 2A, where A = 4(2L - 1)T_e(lambda_H + K_* lambda_L), L = 3 labels,
    T_e = .7, lambda_H = .2 + .15, lambda_L = .25 and K_* the largest
    |k_j| of positive count. It omits the nonnegative coefficient
    discrepancy E_eff, so it does not detect a lost E_eff. For PTSC (P6,
    `sampled_recipes.select_ptsc`) it is B**2*(2 delta + delta**2) with
    delta = branch_bound <= 1, and the selection premise
    3 B**2 delta <= .16 must hold. At norm 4, B**2 (about
    29, asserted above 2) and g**2 (64, asserted above 1) are well above 1,
    so a B**2 or row frame lost in `sampled_plan._recipe_bias` makes P too
    small and fails the lower comparison. The relative slack 1e-12 covers
    the rounding of the masses and coefficient magnitudes summed here
    against the upward-rounded bound.
    """
    from scipy.linalg import expm
    import qiskit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from nwqlib import estimate
    from nwqlib.blocks.lowering import lower_qiskit
    from nwqlib.blocks.records import SelectedConstruction
    from nwqlib.operators import ingest_pauli
    from nwqlib.algorithms.lchs.sampled_rows import integer_categorical
    from nwqlib.algorithms.lchs.selection import select_compact

    P = _SAMPLED_PAULI
    A = .3 * P["I"] + .1j * P["I"] + .25 * P["X"] + .2j * P["Z"] - .15j * P["Y"]
    O = .7 * P["X"] - .45 * P["Z"] + .2 * P["I"]
    time = .7
    u0 = np.zeros(2, complex) if source == "only" else norm * np.array([.6, .8j])
    strength = 1e-18 if source == "weak" else 1.
    b = None if source is None else strength * norm * np.array([.3, -.4j])
    problem = LinearDynamics(A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
                             initial_state=u0, source=b, time=time)
    method = LCHS(observable_estimator="sampled_pairs", pair_repeats=4, **options)
    output = QuadraticForm(observable=ingest_pauli([("X", .7), ("Z", -.45), ("I", .2)], num_qubits=1))
    built = []
    original = qiskit.QuantumCircuit.__init__
    qiskit.QuantumCircuit.__init__ = lambda self, *a, **k: (built.append(1), original(self, *a, **k))[1]
    try:
        chosen = plan(problem, method=method, output=output, shots=64, seed=3)
        workload = estimate(chosen)
    finally:
        qiskit.QuantumCircuit.__init__ = original
    assert not built
    assert (workload.quantity("settings").fact.value.numerator, workload.quantity("shots").fact.value.numerator) == (16, 64)
    rec = chosen.reconstruction
    rows, bias = rec.pair_rows, rec.pair_bias
    g = rows.scale.mantissa * 2.0**rows.scale.exponent
    shift = rec.psd_shift - rec.scalar_decay
    data = select_compact(method, problem)
    k, c = np.asarray(data.quadrature.k_nodes), np.asarray(data.quadrature.coefficients)
    def phase(t, node):
        return np.exp(-1j * t * (node * (.3 + shift) + .1))
    # Input norms without squaring norm-scaled entries, which overflows at 1e300.
    nu0, nub = (norm * np.linalg.norm(vector) for vector in ([.6, .8j], [.3, -.4j]))
    nub *= strength
    directions = {0: None if source == "only" else u0 / nu0,
                  1: None if b is None else b / nub}
    continuous = options.get("pair_source_time") == "continuous"
    applications = [] if source == "only" else [(nu0 * np.exp(shift * time), time, 0)]
    # The plan forms its Duhamel rule independently of NumPy's leggauss, so
    # the two tables agree only up to node and weight formation roundoff, a
    # few binary64 units at these 8 nodes, plus the maps r = T (x + 1)/2 and
    # T - r. The window 1e-14 T covers that and lies far below the smallest
    # node gap (about .08 T here), so it identifies each row's source node.
    window = 1e-14 * time
    if b is not None and not continuous:
        x, w = np.polynomial.legendre.leggauss(method.duhamel_nodes)
        starts, start_weights = time * (x + 1) / 2, time * w / 2
        np.testing.assert_allclose(rec.source_nodes, starts, rtol=0, atol=window)
        np.testing.assert_allclose(rec.source_weights, start_weights, rtol=0, atol=window)
        applications += [(weight * nub * np.exp(shift * (time - r)), time - r, 1)
                         for r, weight in zip(starts, start_weights)]
    branches = [(mass * cj * phase(t, node), t, node, kind) for mass, t, kind in applications for cj, node in zip(c, k)]
    def exact(t, node):
        return expm(-1j * t * (node * .25 * P["X"] + .2 * P["Z"] - .15 * P["Y"]))
    v = sum(a * exact(t, node) @ directions[kind] for a, t, node, kind in branches)
    # The kernel law is over 2**k*|c_j|. Here k = 0, or all coefficients are
    # equal, so the lift leaves the relative magnitudes and counts unchanged.
    with np.errstate(divide="ignore"):
        p = integer_categorical(np.log(np.abs(c))).counts / 2.0**52
    if continuous:
        x, w = np.polynomial.legendre.leggauss(96)
        nodes_t, weights_t = time * (x + 1) / 2, time * w / 2
        v = v + sum(weight * nub * np.exp(shift * t) * cj * phase(t, node) * (exact(t, node) @ directions[1])
                    for t, weight in zip(nodes_t, weights_t) for cj, node in zip(c, k))
        masses = np.array([applications[0][0], nub * np.expm1(shift * time) / shift])
    else:
        masses = np.array([mass for mass, _, _ in applications])
    counts = integer_categorical(np.log(masses / masses.max())).counts
    assert counts.sum() == 2**52 and np.all(counts[masses > 0] >= 1)
    pi = counts / 2.0**52
    for side in ("x", "y"):
        kinds, times, nodes = (getattr(rows, f"{name}_{side}").array for name in ("kind", "time", "node"))
        for kind, t, node, magnitude, angle in zip(kinds, times, nodes, getattr(rows, f"magnitude_{side}").array,
                                                   getattr(rows, f"phase_{side}").array):
            j = int(np.flatnonzero(k == node)[0])
            if continuous:
                category = int(kind)
            else:
                (category,) = [i for i, (_, elapsed, branch_kind) in enumerate(applications)
                               if branch_kind == kind and abs(elapsed - t) <= window]
            a = masses[category] * c[j] * phase(t, node)
            assert magnitude * g * pi[category] * p[j] == pytest.approx(abs(a), rel=1e-12, abs=0)
            assert abs(np.exp(1j * angle) - a / abs(a)) < 1e-12
    g2, c_o = g * g, bias.observable_bound
    recipe = options.get("pair_product_formula", "plain")
    if recipe == "plain":
        steps = np.asarray(bias.bucket_steps)
        if continuous:
            # One bucket: the pair-independent product formula of every branch.
            r = int(steps[0])
            vt = sum(a * _sampled_suzuki(t, node, r) @ directions[kind] for a, t, node, kind in branches)
            vt = vt + sum(weight * nub * np.exp(shift * t) * cj * phase(t, node)
                          * (_sampled_suzuki(t, node, r) @ directions[1])
                          for t, weight in zip(nodes_t, weights_t) for cj, node in zip(c, k))
            mean_n, mean_q = np.vdot(vt, vt).real, np.vdot(vt, O @ vt).real
        else:
            a = np.array([branch[0] for branch in branches])
            r_of = steps[bias.bucket_of([branch[2] for branch in branches])]
            pair = np.maximum.outer(r_of, r_of)
            mean_n = mean_q = 0.
            for r in set(r_of.tolist()):
                states = np.array([_sampled_suzuki(t, node, r) @ directions[kind] for _, t, node, kind in branches])
                outer = np.outer(a.conj(), a)
                mean_n += float((outer * (states.conj() @ states.T))[pair == r].sum().real)
                mean_q += float((outer * (states.conj() @ (O @ states.T)))[pair == r].sum().real)
        assert abs(mean_n - np.vdot(v, v).real) <= bias.inner_bias * g2
        assert abs(mean_q - np.vdot(v, O @ v).real) <= c_o * bias.inner_bias * g2
    else:
        from fractions import Fraction
        epsilon, choice = options["pair_pf_allowance"], rec.pair_recipe
        kept = p > 0
        mean_bound = masses.sum() * np.abs(c[kept]).sum()
        physical = bias.inner_bias * g2
        assert mean_bound**2 > 2 and g2 > 1
        if recipe == "richardson":
            # Three labels X, Y, Z; lambda_H = .2 + .15 and lambda_L = .25 as the stored binary64 values.
            node_bound = Fraction(float(np.abs(k[kept]).max()))
            A_xs9 = 4 * (2 * 3 - 1) * Fraction(time) * (Fraction(.2) + Fraction(.15) + node_bound * Fraction(.25))
            s = choice.member_steps[0]
            assert choice.member_steps == (s, 2 * s) and s > 2 * A_xs9
            remainder = float(4 * Fraction(5, 3) * max(Fraction(1), A_xs9) ** 2 * (A_xs9 / s) ** 4)
            assert mean_bound**2 * remainder <= physical * (1 + 1e-12)
        else:
            delta = choice.branch_bound
            assert 0 <= delta <= 1
            assert mean_bound**2 * (2 * delta + delta**2) <= physical * (1 + 1e-12)
            assert 3 * mean_bound**2 * delta <= epsilon * (1 + 1e-12)
        assert physical <= epsilon * (1 + 1e-12)
    if norm == 1:
        augmented = np.zeros((3, 3), complex)
        augmented[:2, :2] = -A
        if b is not None:
            augmented[:2, 2] = b
        u = expm(-A * time) @ u0 + expm(augmented * time)[:2, 2]
        assert abs(np.vdot(v, v).real - np.vdot(u, u).real) <= bias.model_bias * g2
        assert abs(np.vdot(v, O @ v).real - np.vdot(u, O @ u).real) <= c_o * bias.model_bias * g2
    mixed = np.flatnonzero(rows.kind_x.array != rows.kind_y.array)
    if source == "mixed":
        row = int(mixed[0])
        program = chosen.construction.program.select_experiment("pairs", 0, row=row)
        circuit = lower_qiskit(SelectedConstruction(program=program, selections=chosen.construction.selections),
                               blocks=chosen.blocks).circuit.remove_final_measurements(inplace=False)
        label = rec.terms[int(rows.term.array[row])][0]
        value = Statevector(circuit).expectation_value(SparsePauliOp(("Z" if label != "I" else "I") + "Z")).real
        r = int(steps[bias.bucket_of([rows.node_x.array[row], rows.node_y.array[row]])].max())
        chi_x = _sampled_suzuki(rows.time_x.array[row], rows.node_x.array[row], r) @ directions[int(rows.kind_x.array[row])]
        chi_y = _sampled_suzuki(rows.time_y.array[row], rows.node_y.array[row], r) @ directions[int(rows.kind_y.array[row])]
        theta = rows.phase_y.array[row] - rows.phase_x.array[row]
        assert value == pytest.approx((np.exp(1j * theta) * np.vdot(chi_x, P[label] @ chi_y)).real, abs=1e-12)


_UNIFORMS = (7 * 2.0**-53, .25, .5, .75, 1 - 2.0**-53)


@pytest.mark.parametrize("decay, start, stop, uniforms, local", (
    *((decay, start, stop, _UNIFORMS, False) for decay, start, stop in (
        (2.0**-1074, 0., 1.), (-2.0**-1074, 0., 1.), (0., .3, 1.6), (1e-3, .3, 1.6), (-.9, 0., 1.), (40., .3, 1.6),
        (-40., .3, 1.6))),
    (40., .3, 1.6, (7 * 2.0**-53,), True), (-40., .3, 1.6, (1 - 2.0**-53,), True),
    (float(np.nextafter(1 / np.finfo(float).max, np.inf)), 0., float(np.finfo(float).max), (2.0**-53,), True)))
def test_sampled_source_times_follow_the_tilted_quantile(decay, start, stop, uniforms, local):
    """Continuous source times follow the tilted quantile within a rounding scale, also for extreme tilts and intervals.

    The density exp(s t) on [a, b] has the quantile
    t(u) = a + log1p(u expm1(s (b - a)))/s (a + u (b - a) for s = 0),
    evaluated here at 100 digits from the binary64 inputs and compared at
    that precision. A tilt of 2**-1074 underflows u*E, where the evaluation
    must keep the uniform displacement u (b - a) rather than return an
    endpoint. For these interval-scale cases the comparison scale is 16
    units in the last place of max(|a|, |b|), a few expm1, log1p, product,
    quotient and sum roundings of about one unit each.

    The ``local`` cases are s = 40 with u = 7*2**-53 and s = -40 with
    u = 1 - 2**-53 on [.3, 1.6], and the smallest tilt above 1/DBL_MAX on
    [0, DBL_MAX] with u = 2**-53. Each time lies at the distance
    d = F(x)/|s|, F(x) = log1p(p expm1(x)), from its low-density end c (a
    for s > 0, b for s < 0), with x = |s| (b - a) and p = u for s > 0,
    1 - u for s < 0. Measured from the other end, the same time needs the
    logarithm of the small number p + (1 - p) exp(-x), where an absolute
    rounding error of order 2**-53 moves the time by order
    2**-53/(p |s|), and on [0, DBL_MAX] the interval scale would accept any
    small displacement, so these cases use a local scale. The source
    (`sampled_rows.source_times`) measures them from c. Under binary64
    round-to-nearest with gradual underflow and expm1 and log1p with
    relative error at most 2*2**-53, the rounded x lies within
    eta = gamma_2 x of x, F is convex, so F' is at most
    lambda = p exp(x + eta)/(1 + p expm1(x + eta)) on that interval, and
    the other roundings contribute at most gamma_16 (|c| + d), with
    gamma_j = j 2**-53/(1 - j 2**-53). The error must stay within
    4 (2 lambda eta/|s| + gamma_16 (|c| + d)), where the factor 4 is
    regression headroom. The premise concerns the test platform's libm; a
    failure calls for diagnosing it, not for a wider scale.
    """
    mpmath = pytest.importorskip("mpmath")
    from nwqlib.algorithms.lchs.sampled_rows import source_times

    times = source_times(decay, start, stop, np.array(uniforms))
    with mpmath.workdps(100):
        s, a, b = (mpmath.mpf(value) for value in (decay, start, stop))
        sigma, unit = abs(s), mpmath.mpf(2)**-53
        x = sigma * (b - a)
        gamma2, gamma16 = (j * unit / (1 - j * unit) for j in (2, 16))
        for u, t in zip(uniforms, times):
            u = mpmath.mpf(u)
            exact = a + u * (b - a) if decay == 0 else a + mpmath.log1p(u * mpmath.expm1(s * (b - a))) / s
            if local:
                p = u if decay > 0 else 1 - u
                eta = gamma2 * x
                slope = p * mpmath.exp(x + eta) / (1 + p * mpmath.expm1(x + eta))
                distance = mpmath.log1p(p * mpmath.expm1(x)) / sigma
                scale = 4 * (2 * slope * eta / sigma + gamma16 * (abs(a if decay > 0 else b) + distance))
            else:
                scale = 16 * np.spacing(max(abs(start), abs(stop)))
            assert start <= t <= stop
            assert abs(mpmath.mpf(t) - exact) <= scale


_RECIPE_OPTIONS = {
    "richardson": dict(hamiltonian_evolution_backend="trotter", trotter_steps=2, pair_product_formula="richardson"),
    "mpf": dict(hamiltonian_evolution_backend="trotter", trotter_steps=2, pair_product_formula="mpf"),
    "ptsc": dict(hamiltonian_evolution_backend="trotter_error_budgeted", trotter_order=1, pair_product_formula="ptsc",
                 pair_pf_allowance=1e-2, pair_mu_cap=1.5, pair_taylor_cutoff=8),
}


@pytest.mark.parametrize("options", (dict(hamiltonian_evolution_backend="trotter", trotter_steps=2),
                                     *_RECIPE_OPTIONS.values()), ids=("plain", *_RECIPE_OPTIONS))
def test_sampled_pair_run_reports_raw_partial_sums_and_continues_only_missing_rows(tmp_path, monkeypatch, options):
    """A sampled-pair Run stopped after 7 of its 16 rows gives raw sums without confidence and continues to the one-pass result.

    `sampled_pair_progress` reads the stopped Run's saved data, before and
    after reopening it with `load_run`: rows 0-6 completed and 7-15 missing,
    one row of five shot sums per completed row whose totals are the
    moments' sums, and no estimate, radius or Result. Continuing measures
    exactly the 9 missing rows, and the continued Result's moments and
    estimate equal those of an uninterrupted solve of the same Plan, whose
    rows were drawn once at planning. That solved Result, saved and reloaded
    with `load_result`, keeps its moments, estimate and value, and analyzing
    the reloaded Result again gives the same estimate. With a
    `pair_product_formula` recipe the reloaded Plan keeps the recipe
    choices drawn at planning (member per row or PTSC corrections), which
    the continued rows and the reanalysis use.
    """
    import nwqlib
    from nwqlib import _prepared_execution as acquisition
    from nwqlib._prepared_execution import Run
    from nwqlib.algorithms.lchs import sampled_pair_progress
    from nwqlib.operators import ingest_pauli

    A = ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1)
    method = LCHS(observable_estimator="sampled_pairs", pair_repeats=4, **options)
    chosen = plan(LinearDynamics(A=A, initial_state=[.6, .8j], source=[.3, -.4j], time=.7), method=method,
                  output=QuadraticForm(observable=ingest_pauli([("X", .7)], num_qubits=1)), shots=64, seed=13)
    single = solve(chosen)
    loaded = nwqlib.load_result(single.save(tmp_path / "single"))
    assert loaded.plan.reconstruction.pair_recipe == chosen.reconstruction.pair_recipe
    assert (loaded.pair_moments, loaded.pair_estimate, loaded.value) == (
        single.pair_moments, single.pair_estimate, single.value)
    # Analysis resolves the readout of the one row experiment once for its 16 rows, and forms each row's
    # Realization from its values (`Plan._point_realization`) without building the row's Program.
    from nwqlib.core.planning import Plan
    resolved, resolve_once = [], Plan._resolve_selection_once
    with monkeypatch.context() as patch:
        patch.setattr(Plan, "_resolve_selection_once",
                      lambda self, *a: (resolved.append(1), resolve_once(self, *a))[1])
        assert loaded.analyze().pair_estimate == single.pair_estimate
    assert len(resolved) == 1
    submit, calls = acquisition.submit_experiment, []

    def interrupted(*args, **kwargs):
        if len(calls) == 7:
            raise RuntimeError("stop before next acquisition")
        calls.append(1)
        return submit(*args, **kwargs)
    with Run(chosen, directory=tmp_path / "run") as run:
        with monkeypatch.context() as patch:
            patch.setattr(acquisition, "submit_experiment", interrupted)
            with pytest.raises(RuntimeError, match="before next acquisition"):
                run.resume()
        chunks = len(run.observations.chunks)
        # The partial analysis forms only the points of its seven data collections, not all 16.
        formed, static_point = [], Plan._static_point
        with monkeypatch.context() as patch:
            patch.setattr(Plan, "_static_points", lambda *a: pytest.fail("partial analysis generated every point"))
            patch.setattr(Plan, "_static_point", lambda self, *a: (formed.append(1), static_point(self, *a))[1])
            progress = sampled_pair_progress(run)
        assert len(formed) == 7
        assert len(run.observations.chunks) == chunks and run.result is None
    assert progress.status == "incomplete" and progress.completed_rows.array.tolist() == list(range(7))
    assert progress.missing_rows.array.tolist() == list(range(7, 16))
    assert progress.row_sums.array.shape == (7, 5)
    assert progress.row_sums.array[:, 0].sum() == pytest.approx(progress.moments.sum_q, rel=1e-14, abs=1e-14)
    calls.clear()
    monkeypatch.setattr(acquisition, "submit_experiment", lambda *a, **k: (calls.append(1), submit(*a, **k))[1])
    with nwqlib.load_run(tmp_path / "run") as run:
        assert sampled_pair_progress(run) == progress
        result = run.wait()
        assert sampled_pair_progress(run).status == "complete"
    assert len(calls) == 9
    def numbers(m):
        return m.rows.array.tolist(), m.sum_q, m.sum_n, m.sum_qq, m.sum_nn, m.sum_qn, m.cluster_q, m.cluster_n
    assert numbers(result.pair_moments) == numbers(single.pair_moments)
    assert result.pair_estimate.model_dump(exclude={"parent_id"}) == single.pair_estimate.model_dump(exclude={"parent_id"})


def _sampled_rotation(label, angle):
    return np.cos(angle) * _SAMPLED_PAULI["I"] - 1j * np.sin(angle) * _SAMPLED_PAULI[label]


def _sampled_mpf(time, node, steps, alpha):
    """P7 word S = F_a e^{-ihG_3} R_b F_b e^{-ihG_3} R_a over G_1..G_3 = X, Y, Z terms, h = 1/r, r/2 macrosteps.

    F_a = prod_{j=1}^{2} e^{-i a h G_j} acts with j = 2 first and
    R_a = prod_{j=2}^{1} e^{-i a h G_j} with j = 1 first, b = 1 - a, and
    G_j = time*(node*l_j + h_j) P_j in the time-one convention.
    """
    coefficients = dict(X=.25 * node, Y=-.15, Z=.2)
    beta, h = 1.0 - alpha, 1.0 / steps
    sequence = ([("X", alpha), ("Y", alpha), ("Z", 1.0), ("Y", beta), ("X", beta), ("X", beta), ("Y", beta),
                 ("Z", 1.0), ("Y", alpha), ("X", alpha)])
    macro = _SAMPLED_PAULI["I"]
    for label, a in sequence:
        macro = _sampled_rotation(label, a * h * time * coefficients[label]) @ macro
    return np.linalg.matrix_power(macro, steps // 2)


def _sampled_lie(time, node, steps, h_terms, l_terms):
    """One PTSC Lie step: every H-list entry, then every L-list entry with coefficient node*l, in list order."""
    step = _SAMPLED_PAULI["I"]
    for label, value in (*h_terms, *((label, node * value) for label, value in l_terms)):
        step = _sampled_rotation(label, time / steps * value) @ step
    return step


@pytest.mark.parametrize("recipe", tuple(_RECIPE_OPTIONS))
def test_sampled_pair_recipe_rows_act_as_independent_branch_evolutions(recipe):
    """Each recipe row circuit gives sign(gamma)*Re(exp(i theta) <chi_x|P|chi_y>) for independently evolved branches.

    For the sampled keeper problem with a source, every row of a 12-row
    Plan is lowered and its exact pair expectations E[s_a s_P] (Q score) and
    E[s_a] (N score) are compared with chi_b = W_b psi_b, psi_b the unit
    initial or source direction. W_b is the Suzuki-2 word at the row's
    Richardson level (s or 2s), the P7 word of the row's MPF alpha with r/2
    macrosteps, or for PTSC the Lie step of the saved H and L lists each
    followed by the row's saved correction (`sampled_recipes.correction_matrix`;
    its law is checked against the Taylor series separately). A negative
    member multiplier must reverse both scores, and gamma times its
    integer probability must return the exact Richardson or MPF weight.
    The stored PTSC log mu must equal r*log(A_0 + sum_{s>=4} eta_s) formed
    directly. Each oracle is a few dozen 2x2 products, so 1e-11 bounds
    their rounding.
    """
    from math import factorial
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from nwqlib.blocks.lowering import lower_qiskit
    from nwqlib.blocks.records import SelectedConstruction
    from nwqlib.operators import ingest_pauli
    from nwqlib.algorithms.lchs.sampled_recipes import correction_matrix

    P = _SAMPLED_PAULI
    u0, b = np.array([.6, .8j]), np.array([.3, -.4j])
    directions = {0: u0 / np.linalg.norm(u0), 1: b / np.linalg.norm(b)}
    problem = LinearDynamics(A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
                             initial_state=u0, source=b, time=.7)
    options = dict(_RECIPE_OPTIONS[recipe], **({"trotter_steps": 3} if recipe == "richardson" else
                                               {"trotter_steps": 4} if recipe == "mpf" else {}))
    chosen = plan(problem, method=LCHS(observable_estimator="sampled_pairs", **options),
                  output=QuadraticForm(observable=ingest_pauli([("X", .7), ("Z", -.45), ("I", .2)], num_qubits=1)),
                  shots=12, seed=7)
    rec, payload = chosen.reconstruction, chosen._native["native_data"]
    rows, choice = rec.pair_rows, rec.pair_recipe
    signs = np.where(choice.row_phases() == np.pi, -1.0, 1.0)
    for row in range(rows.clusters):
        branches = []
        for side in ("x", "y"):
            t, k = (float(getattr(rows, f"{name}_{side}").array[row]) for name in ("time", "node"))
            chi = directions[int(getattr(rows, f"kind_{side}").array[row])]
            if recipe == "ptsc":
                lie = _sampled_lie(t, k, choice.steps, payload.h_terms, payload.l_terms)
                for segment in choice.corrections(row):
                    chi = correction_matrix(segment[0 if side == "x" else 1], 1) @ (lie @ chi)
            else:
                member = int(choice.member.array[row])
                steps = choice.member_steps[member]
                chi = (_sampled_suzuki(t, k, steps) if recipe == "richardson"
                       else _sampled_mpf(t, k, steps, choice.member_alphas[member])) @ chi
            branches.append(chi)
        program = chosen.construction.program.select_experiment("pairs", 0, row=row)
        circuit = lower_qiskit(SelectedConstruction(program=program, selections=chosen.construction.selections),
                               blocks=chosen.blocks).circuit.remove_final_measurements(inplace=False)
        label = rec.terms[int(rows.term.array[row])][0]
        state = Statevector(circuit)
        theta = rows.phase_y.array[row] - rows.phase_x.array[row]
        expected_q = signs[row] * (np.exp(1j * theta) * np.vdot(branches[0], P[label] @ branches[1])).real
        expected_n = signs[row] * (np.exp(1j * theta) * np.vdot(branches[0], branches[1])).real
        assert state.expectation_value(SparsePauliOp(("Z" if label != "I" else "I") + "Z")).real == pytest.approx(
            expected_q, abs=1e-11)
        assert state.expectation_value(SparsePauliOp("IZ")).real == pytest.approx(expected_n, abs=1e-11)
    if recipe != "ptsc":
        assert np.any(signs < 0)
        products = np.asarray(choice.member_counts) / 2.0**52 * np.asarray(choice.member_multipliers)
        assert products == pytest.approx([n / d for n, d in choice.member_weights], rel=0, abs=1e-15)
        return
    for row, side in np.ndindex(rows.clusters, 2):
        t = float((rows.time_x, rows.time_y)[side].array[row])
        k = float((rows.node_x, rows.node_y)[side].array[row])
        a = 2 * (.35 + abs(k) * .25) * t / choice.steps
        eta = [a**s / factorial(s) for s in range(choice.cutoff + 1)]
        mu = np.sqrt(1 + (eta[2] + eta[3]) ** 2) + sum(eta[4:])
        assert choice.log_mu.array[row, side] == pytest.approx(choice.steps * np.log(mu), rel=1e-12)


def _taylor_parts(h_list, l_list, x, k, cutoff):
    """Return F_0..F_cutoff, the degree-s parts of V = exp(-ixH) S_1^dagger, from truncated series products."""
    from math import factorial
    terms = [*h_list, *((label, k * value) for label, value in l_list)]
    generator = sum(value * _SAMPLED_PAULI[label] for label, value in terms)

    def series(matrix):
        return [np.linalg.matrix_power(matrix, n) / factorial(n) for n in range(cutoff + 1)]

    def multiply(left, right):
        out = [np.zeros((2, 2), complex) for _ in range(cutoff + 1)]
        for i, a in enumerate(left):
            for j, c in enumerate(right[:cutoff + 1 - i]):
                out[i + j] += a @ c
        return out
    result = series(-1j * x * generator)
    # S_1^dagger = exp(i x c_1 P_1) ... exp(i x c_L P_L), left to right.
    for label, value in terms:
        result = multiply(result, series(1j * x * value * _SAMPLED_PAULI[label]))
    return result


@pytest.mark.parametrize("k", (-1.3, 0.0))
def test_sampled_ptsc_segment_law_is_the_taylor_truncation(k):
    """mu_step*E[C] over the actual integer laws equals I + F_2 + ... + F_sc within mu_step*d_seg (PTSC, XS28e and P4).

    Exhaustive enumeration over the orders 2..s_c, all 2**s fair-bit
    patterns and every list-index sequence, through the production order,
    list-choice and list laws, signed words, segment composition
    (`segment_correction`) and correction matrices, at a = 2 lambda x near
    0.5 for k = -1.3 (both lists, negative k) and k = 0 (H list only). The
    H and L lists share X and Z. The expected value comes from the
    independent truncated series of exp(-ixH) S_1^dagger. An unmerged-list,
    sign(k), sorting or i-phase error, or a dropped Hermitian sample,
    changes the result by O(a**2); the bound is the P4 count perturbation
    mu_step*d_seg plus 1e-13 for the enumeration's few thousand 2x2
    products.

    For k = -1.3 the circuit of every order >= 4 correction the enumeration
    reaches is checked as well: one path per distinct (phase, x, z), at most
    16 on one qubit. Its correction is Q = (-i)**b i**(s-b) times the signed
    Paulis sign(c)*P of the first b labels and then the sorted rest, formed
    here as a dense product with c = h for H entries and k*l for L entries.
    `sampled_native._append_correction` under branch value v must give
    I (x) |1-v><1-v| + Q (x) |v><v| (branch qubit 0 is the last Kronecker
    factor), compared without global-phase alignment because the phase of Q
    is relative between the branches, and must emit `correction_cx`, one
    CX, CY or CZ per support site. The enumeration must reach an odd phase
    and an identity word with a nonzero phase. The two-qubit H list
    .2 ZI - .15 XX + .05 XI with all four bits 1 and labels (0, 1, 2, 2)
    gives Q = ZI(-XX)XI XI = -iYX. Its (phase, x, z) = (3, 3, 2) has an odd
    phase, a Y site and unequal masks, so a wire or phase error changes the
    operator. These matrices have entries 0 and powers of i, so 1e-13
    covers the native phase gates only.

    In both k cases, `draw_compensation` maps a fixed transcript of the
    primitive integers it draws to the descriptors expected from the
    Pauli products by hand. One row at t = .8 with real `select_ptsc` fields
    (r = 4, s_c = 5 at pair_mu_cap = 1.2, pair_pf_allowance = 6e-4 and
    K_* = 1.3) has branch x draw the orders 2, 3, 4, 5 with prescribed bits
    and labels and branch y the same paths in reverse. Each categorical
    integer lies inside its count interval of the law `draw_compensation`
    builds, so the draw returns the intended index, and every integer of
    the transcript must be drawn exactly once. The fixed stream repeats its
    values if the code under test asks for more, so an order offset still
    reaches the descriptor comparison. An order offset, a dropped sign(k) in
    the drawn words or an unadvanced segment start changes an order or a
    phase. This is a check of the draw mapping, not of the law's
    frequencies, which the enumeration above owns.
    """
    from fractions import Fraction
    from itertools import product
    from math import fsum
    from types import SimpleNamespace
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from nwqlib.algorithms.lchs.sampled_native import _append_correction
    from nwqlib.algorithms.lchs.sampled_recipes import (Support, _branch_laws, _segment_law_bound, correction_cx,
                                                         correction_matrix, draw_compensation, mixture_law,
                                                         segment_correction, select_ptsc, signed_words)
    from nwqlib.algorithms.lchs.sampled_rows import integer_categorical
    from nwqlib.core.records import FrozenArray
    from nwqlib.problems.inputs import PhysicalScale
    h_list, l_list = (("Z", .2), ("Y", -.15), ("X", .05)), (("X", .25), ("Z", -.1))
    cutoff, x = 5, .4
    h_norm, l_norm = sum(abs(v) for _, v in h_list), sum(abs(v) for _, v in l_list)
    log_step, angle, order_law = _branch_laws(2 * (h_norm + abs(k) * l_norm) * x, cutoff)
    mu = np.exp(log_step)
    mixture = mixture_law(h_norm, l_norm, k).counts / 2.0**52
    lists = [integer_categorical(np.log([abs(v) for _, v in terms])).counts / 2.0**52 for terms in (h_list, l_list)]
    label_law = np.concatenate([mixture[0] * lists[0], mixture[1] * lists[1]])
    words = signed_words(h_list, l_list, k)
    expectation = np.zeros((2, 2), complex)
    high_order = {}
    for index, p_order in enumerate(order_law.counts / 2.0**52):
        s = index + 2
        for bits in product((0, 1), repeat=s):
            for labels in product(range(len(words)), repeat=s):
                p = np.prod(label_law[list(labels)])
                if p:
                    phase, xm, zm = segment_correction(bits, labels, words)
                    expectation += p_order * 2.0**-s * p * correction_matrix((s, phase, xm, zm, angle), 1)
                    if s >= 4:
                        high_order.setdefault((phase, xm, zm), (s, bits, labels))
    parts = _taylor_parts(h_list, l_list, x, k, cutoff)
    assert np.allclose(parts[1], 0, atol=1e-15)
    assert np.linalg.norm(mu * expectation - (parts[0] + sum(parts[2:])), 2) <= (
        mu * _segment_law_bound(cutoff, len(h_list), len(l_list), k != 0) + 1e-13)

    def signed_product(bits, labels, terms):
        b = sum(bits)
        q = (-1j) ** b * 1j ** (len(bits) - b) * np.eye(2 ** len(terms[0][0]))
        for g in (*labels[:b], *sorted(labels[b:])):
            q = q @ (np.sign(terms[g][1]) * _pauli_matrix(terms[g][0]))
        return q

    def emitted(descriptor, width):
        """Operators and CX-basis count of the correction under branch values 0 and 1, root 0, system 1..width."""
        out = []
        for value in (0, 1):
            circuit = QuantumCircuit(1 + width)
            _append_correction(circuit, 0, list(range(1, 1 + width)), descriptor, value)
            ops = circuit.count_ops()
            out.append((Operator(circuit).data, ops.get("cx", 0) + ops.get("cy", 0) + ops.get("cz", 0)))
        return out

    projector = (np.diag([1., 0.]), np.diag([0., 1.]))
    if k < 0:
        assert any(phase % 2 for phase, _, _ in high_order) and any(
            phase and not (xm | zm) for phase, xm, zm in high_order)
        terms = (*h_list, *((label, k * value) for label, value in l_list))
        for (phase, xm, zm), (s, bits, labels) in high_order.items():
            q = signed_product(bits, labels, terms)
            for value, (matrix, cx) in enumerate(emitted((s, phase, xm, zm, angle), 1)):
                expected = np.kron(np.eye(2), projector[1 - value]) + np.kron(q, projector[value])
                assert np.allclose(matrix, expected, rtol=0, atol=1e-13)
                assert cx == correction_cx([(s, phase, xm, zm, angle)])
        h2 = (("ZI", .2), ("XX", -.15), ("XI", .05))
        descriptor = (4, *segment_correction((1, 1, 1, 1), (0, 1, 2, 2), signed_words(h2, (), 0.0)), 0.0)
        assert descriptor[1:4] == (3, 3, 2)
        q = signed_product((1, 1, 1, 1), (0, 1, 2, 2), h2)
        for value, (matrix, cx) in enumerate(emitted(descriptor, 2)):
            expected = np.kron(np.eye(4), projector[1 - value]) + np.kron(q, projector[value])
            assert np.allclose(matrix, expected, rtol=0, atol=1e-13)
            assert cx == correction_cx([descriptor]) == 2

    method = LCHS(observable_estimator="sampled_pairs", hamiltonian_evolution_backend="trotter_error_budgeted",
                  trotter_order=1, pair_product_formula="ptsc", pair_pf_allowance=6e-4, pair_mu_cap=1.2,
                  pair_taylor_cutoff=5)
    support = Support(active_time=.8, node_bound=1.3, h_norm=sum((Fraction(abs(v)) for _, v in h_list), Fraction(0)),
                      l_norm=sum((Fraction(abs(v)) for _, v in l_list), Fraction(0)), labels=3,
                      mean_weight=PhysicalScale(mantissa=.5, exponent=1), num_qubits=1)
    fields = select_ptsc(method, support, h_count=3, l_count=2, clusters=1)[0]
    assert (fields["steps"], fields["cutoff"]) == (4, 5)
    # a = 2*lambda(k)*t/r as draw_compensation forms it, with t/r = .8/4 = .2 exactly.
    _, _, draw_order_law = _branch_laws(2 * (fsum(abs(v) for _, v in h_list) + abs(k) * fsum(
        abs(v) for _, v in l_list)) * .2, 5)
    draw_mixture = mixture_law(fsum(abs(v) for _, v in h_list), fsum(abs(v) for _, v in l_list), k)
    draw_lists = [integer_categorical(np.log([abs(v) for _, v in terms])) for terms in (h_list, l_list)]
    bit_paths = ((0, 0), (1, 0, 0), (0, 1, 1, 0), (1, 0, 0, 0, 0))
    # Indices 0-2 are the H entries Z, Y, X and 3-4 the L entries X, Z. The
    # signed words are (Z, -Y, X, -X, Z) at k < 0, so for example the order-2
    # path gives i**2 Z(-X) = iY; at k = 0 only H entries are drawn.
    if k < 0:
        label_paths = ((0, 3), (1, 0, 3), (3, 1, 4, 2), (4, 3, 1, 0, 2))
        expected_x = ((2, 1, 1, 1), (3, 2, 0, 0), (4, 3, 1, 0), (5, 3, 1, 1))
    else:
        label_paths = ((0, 2), (1, 0, 2), (2, 1, 0, 2), (0, 2, 1, 0, 2))
        expected_x = ((2, 3, 1, 1), (3, 0, 0, 0), (4, 3, 1, 0), (5, 1, 1, 1))

    def interior(counts, index):
        return int(counts[:index].sum()) + (int(counts[index]) - 1) // 2

    tape = []
    for paths in ((0, 1, 2, 3), (3, 2, 1, 0)):
        labels = [label for j in paths for label in label_paths[j]]
        tape.append([interior(draw_order_law.counts, j) for j in paths])
        tape.append([bit for j in paths for bit in bit_paths[j]])
        if k != 0:
            tape.append([interior(draw_mixture.counts, int(label >= 3)) for label in labels])
        tape.append([interior(draw_lists[int(label >= 3)].counts, label - 3 * (label >= 3)) for label in labels])

    class FixedIntegers:
        def __init__(self, values):
            self.values = iter(values)

        def integers(self, low, high, *, size, dtype):
            return np.resize(np.asarray(next(self.values), dtype=dtype), size)

    stream = FixedIntegers(tape)
    rows = SimpleNamespace(clusters=1, time_x=FrozenArray(np.array([.8])), time_y=FrozenArray(np.array([.8])),
                           node_x=FrozenArray(np.array([k])), node_y=FrozenArray(np.array([k])))
    choice = draw_compensation(fields, SimpleNamespace(method=stream), rows, l_list, h_list, 1)
    drawn = np.stack([getattr(choice, f"segment_{name}").array[0] for name in ("order", "phase", "x", "z")], axis=-1)
    assert np.array_equal(drawn, np.array((expected_x, expected_x[::-1])))
    assert next(stream.values, None) is None


def test_sampled_mpf_word_cancels_every_cubic_defect():
    """The two weighted P7 words used by MPF rows cancel every cubic word defect of exp(2h(A + B + C + D)) (P7).

    `sampled_recipes.mpf_template` gives the macrostep of four ordered
    letters with 4K - 3 = 13 factors and total exponent 2h per letter. Its
    degree-3 coefficient of each ordered word, a sum over nondecreasing
    factor triples of prod c/prod multiplicity!, minus 8/6, the coefficient
    in exp(2h(A + B + C + D)), is weighted by c = (3/2, -1/2) at the exact
    alpha = ((3 + sqrt 3)/9, (1 + sqrt 3)/3), evaluated in exact arithmetic
    in Q(sqrt 3), where expansion gives a canonical form, and must vanish
    for all 64 words. Four letters are the fewest at which the reversed
    middle run beta G_{K-1}..G_2 has two labels: written forward it turns
    C_beta B_beta into B_beta C_beta, which raises the DBC coefficient by
    sum_j c_j beta_j**2 = 1/3.
    """
    from itertools import product
    import sympy
    from nwqlib.algorithms.lchs.sampled_recipes import mpf_template
    root = sympy.sqrt(3)
    letters = ("A", "B", "C", "D")
    total = {word: 0 for word in product(letters, repeat=3)}
    for alpha_float, alpha, weight in (((3 + 3**.5) / 9, (3 + root) / 9, sympy.Rational(3, 2)),
                                       ((1 + 3**.5) / 3, (1 + root) / 3, sympy.Rational(-1, 2))):
        meaning = {2 * alpha_float: alpha, 2.0: sympy.Integer(1), 2 * (1 - alpha_float): 1 - alpha,
                   4 * (1 - alpha_float): 2 * (1 - alpha)}
        word = [(label, meaning[multiplier]) for label, multiplier in mpf_template(letters, alpha_float)]
        assert len(word) == 13
        assert all(sum(c for label, c in word if label == letter) == 2 for letter in letters)
        for i, j, k in ((i, j, k) for i in range(13) for j in range(i, 13) for k in range(j, 13)):
            multiplicity = sympy.prod(sympy.factorial((i, j, k).count(t)) for t in {i, j, k})
            key = (word[i][0], word[j][0], word[k][0])
            total[key] += weight * word[i][1] * word[j][1] * word[k][1] / multiplicity
    assert all(sympy.expand(value - sympy.Rational(8, 6)) == 0 for value in total.values())


@pytest.mark.parametrize("recipe, select", (*((recipe, {}) for recipe in _RECIPE_OPTIONS),
                                            ("ptsc", dict(lcu_select_implementation="structured",
                                                          pair_control_copies=2))),
                         ids=(*_RECIPE_OPTIONS, "ptsc-structured-copies"))
def test_sampled_pair_recipe_resources_count_the_emitted_rows(recipe, select, monkeypatch):
    """`sampled_pair_resources` equals the CX-basis gates of every constructed row circuit and builds or draws nothing.

    The expected counts are read from the actual row circuits
    (`sampled_plan.construct_pair`), expanded one level as the SELECT law
    check does: each cx, cy and cz is one CX and each CRZ two. Preparation
    gates are opaque and excluded, as in the resource record. For a member
    recipe the expected per-shot SELECT CX is sum_j pi_j C_j with C_j read
    from built rows of member j. PTSC rows are counted both with the
    multiplexed SELECT, whose one-step count repeats in each of the r
    segments, and with the structured SELECT and two control copies, which
    adds each row's structured count in every segment and
    2*(r_c - 1) = 2 fanout CX per row.

    A backend that binds parameterized circuits receives each row as values
    of a shared template (`LCHS.row_templates`). The row circuit is the
    oracle: every bound template must give the row's unitary, compared
    without global-phase alignment, so a recipe's word, members or PTSC
    corrections cannot reach the provider with another row's angles.
    """
    import qiskit
    from qiskit.quantum_info import Operator
    from nwqlib.operators import ingest_pauli
    from nwqlib.algorithms.lchs import sampled_recipes, sampled_rows
    from nwqlib.algorithms.lchs.quantum_resources import sampled_pair_resources
    from nwqlib.algorithms.lchs.sampled_plan import construct_pair, point_row
    problem = LinearDynamics(A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
                             initial_state=[.6, .8j], source=[.3, -.4j], time=.7)
    chosen = plan(problem, method=LCHS(observable_estimator="sampled_pairs", pair_repeats=2, **_RECIPE_OPTIONS[recipe],
                                       **select),
                  output=QuadraticForm(observable=ingest_pauli([("X", .7), ("Z", -.45)], num_qubits=1)), shots=24,
                  seed=11)
    rows, choice = chosen.reconstruction.pair_rows, chosen.reconstruction.pair_recipe
    built, original = [], qiskit.QuantumCircuit.__init__
    for module, names in ((sampled_recipes, ("draw_members", "draw_compensation", "select_ptsc", "select_richardson",
                                             "select_mpf")), (sampled_rows, ("draw_pair_rows",))):
        for name in names:
            monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("resource estimate drew or selected again"))
    monkeypatch.setattr(qiskit.QuantumCircuit, "__init__",
                        lambda self, *a, **k: (built.append(1), original(self, *a, **k))[1])
    resources = sampled_pair_resources(chosen)
    monkeypatch.undo()
    assert not built and resources["recipe"] == choice.version
    block = next(item for item in chosen.blocks if item.record.signature.name == "lchs_pair")
    points = tuple(chosen._static_points())
    templates = chosen.method.row_templates(chosen, points)
    rows_of = {point.key: point_row(point) for point in points}
    bound = {rows_of[key]: template.circuit.assign_parameters(dict(zip(template.parameters, values)))
             for template in templates for key, values in template.rows}
    per_row = []
    for row in range(rows.clusters):
        circuit = construct_pair(block, (("row", row),), None)
        assert np.max(np.abs(Operator(bound[row]).data - Operator(circuit).data)) <= 1e-12
        counts = circuit.decompose(gates_to_decompose=["ucrz", "crz", "nwqlib_phase_diagonal"], reps=2).count_ops()
        per_row.append(counts.get("cx", 0) + counts.get("cy", 0) + counts.get("cz", 0))
    assert resources["pair_select_cx_total"] == sum(per_row) * rows.repeats
    if recipe == "ptsc":
        corrections = [sampled_recipes.correction_cx([d for segment in choice.corrections(row) for d in segment])
                       for row in range(rows.clusters)]
        assert resources["pair_correction_cx_per_row"] == (min(corrections), max(corrections))
    else:
        member_cx = {int(choice.member.array[row]): per_row[row] for row in range(rows.clusters)}
        assert len(member_cx) == 2
        assert resources["expected_select_cx_per_shot"] == pytest.approx(
            sum(count / 2**52 * member_cx[j] for j, count in enumerate(choice.member_counts)), rel=1e-15)


def test_quantum_readout_reads_the_pauli_terms_of_a_32_dimensional_dense_observable():
    """A dense observable above dimension 16 is read from its nonzero Pauli terms.

    The random Hermitian 32 x 32 observable has all 1024 Pauli terms, and the
    planned readout terms sum to it up to the recorded positive scale.
    """
    d = 32
    rng = np.random.default_rng(31)
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    observable = (a + a.conj().T) / 2
    hopping = np.eye(d, k=1) + np.eye(d, k=-1)
    problem = LinearDynamics(A=np.diag(np.linspace(.1, .5, d)) + .2j * hopping, initial_state=np.eye(d)[0], time=.5)
    chosen = plan(problem, method=_trotter_method(), output=QuadraticForm(observable=observable),
                  execution="quantum", seed=7)
    terms = chosen.reconstruction.terms
    assert len(terms) == d * d
    planned = sum(c * _pauli_matrix(label) for label, c in terms)
    scale = np.vdot(observable.ravel(), planned.ravel()).real / np.vdot(observable.ravel(), observable.ravel()).real
    assert scale > 0
    np.testing.assert_allclose(planned, scale * observable, atol=1e-13 * scale * np.abs(observable).max(), rtol=0)


def test_dense_observable_stages_are_checked_before_they_allocate(monkeypatch):
    """The Hermitian test reads one row and column at a time, and each conversion stage is checked first.

    A full-support 31-dimensional observable is padded to 32. Under 1,024
    bytes the 65,536-byte transform stage, which depends on the dimension
    alone, is refused before any row is compared. Under 98,304 bytes, which
    cover that stage, the 98,304-byte packing stage and the 85,001-byte label
    stage with the 32,768 bytes of masks and coefficients alive, the
    Hermitian test makes 31 row comparisons of at most 31 entries each and
    the zero-time Plan reads all 1024 terms.
    """
    from nwqlib import NormalizedExpectation
    from nwqlib.algorithms.lchs import quantum
    n = 31
    rng = np.random.default_rng(41)
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    output = NormalizedExpectation(observable=(b + b.conj().T) / 2)
    problem = LinearDynamics(A=np.eye(n), initial_state=np.eye(n)[0], time=0.)
    sizes = []
    equal = np.array_equal

    def traced(x, y, *args, **kwargs):
        sizes.append(max(np.size(x), np.size(y)))
        return equal(x, y, *args, **kwargs)
    with monkeypatch.context() as patched:
        patched.setattr(np, "array_equal", traced)
        with pytest.raises(ValueError, match="dense observable Pauli transform needs about 65536 bytes"):
            quantum.observable_terms(problem, output, max_bytes=1024)
        assert sizes == []
        assert len(quantum.observable_terms(problem, output, max_bytes=98_304)) == 1024
    assert len(sizes) == n and max(sizes) == n
    chosen = plan(problem, method=LCHS(max_bytes=98_304), output=output, execution="quantum", seed=11)
    assert len(chosen.reconstruction.terms) == 1024


@pytest.mark.parametrize(("options", "limited"), (
    (dict(pair_product_formula="richardson", pair_pf_allowance=.16), lambda recipe: recipe.member_steps[1]),
    (dict(trotter_order=1, pair_product_formula="ptsc", pair_pf_allowance=.16, pair_mu_cap=1.5, pair_taylor_cutoff=8),
     lambda recipe: recipe.steps),
))
def test_sampled_recipes_find_their_least_step_count_without_a_step_limit(options, limited):
    """Without max_trotter_steps the Richardson and PTSC searches double to a sufficient step count.

    Both choose the least count that a limit of 10**9 bisects to. A limit one
    below the largest count the recipe runs, 2s for Richardson levels (s, 2s)
    and r for PTSC, refuses the Plan at that capacity limit and names the field.
    """
    from nwqlib._limits import ResourceLimitError
    from nwqlib.operators import ingest_pauli
    problem = LinearDynamics(A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
                             initial_state=[2.4, 3.2j], time=.7)
    output = QuadraticForm(observable=ingest_pauli([("X", .7), ("Z", -.45), ("I", .2)], num_qubits=1))

    def recipe(**limit):
        method = LCHS(observable_estimator="sampled_pairs", pair_repeats=4,
                      hamiltonian_evolution_backend="trotter_error_budgeted", **options, **limit)
        return plan(problem, method=method, output=output, shots=64, seed=3).reconstruction.pair_recipe

    unlimited = recipe()
    assert unlimited == recipe(max_trotter_steps=10**9)
    with pytest.raises(ResourceLimitError, match="max_trotter_steps"):
        recipe(max_trotter_steps=limited(unlimited) - 1)


def test_budgeted_product_formula_nodes_need_no_step_limit():
    """A dense trotter_error_budgeted Plan whose nodes need more than 100,000 steps is planned by default.

    With unit-norm L (positive semidefinite) and H on two qubits, tolerance
    1e-2 and T = 30, the largest node needs more steps than the former
    default limit of 100,000. Setting that limit refuses the Plan and states
    the count it needed.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    l_part = x @ x.conj().T
    y = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h_part = (y + y.conj().T) / 2
    a = l_part / np.linalg.norm(l_part, 2) + 1j * h_part / np.linalg.norm(h_part, 2)
    problem = LinearDynamics(A=a, initial_state=[1., 0., 0., 0.], time=30.)
    options = dict(hamiltonian_evolution_backend="trotter_error_budgeted", approximation_tolerance=1e-2)
    steps = max(plan(problem, method=LCHS(**options)).reconstruction.step_counts)
    assert steps > 100_000
    with pytest.raises(ValueError, match=f"needs {steps} steps, above the step limit 100000"):
        plan(problem, method=LCHS(**options, max_trotter_steps=100_000))


def _sampled_row_positions(chosen):
    """Return the mean Aer positions of the built pair circuits of all rows of a sampled Plan."""
    from qiskit_aer import AerSimulator
    from nwqlib.backends.qiskit_aer import _AerDecompose, _lower_aer_circuit
    target = AerSimulator(method="statevector").target
    block = next(item for item in chosen.blocks if item.record.signature.name == "lchs_pair")
    rows = block._payload[1].pair_rows.clusters
    return block, np.mean([len(_lower_aer_circuit(block._constructor(block, (("row", row),), None),
                                                  _AerDecompose(target))[0].data) for row in range(rows)])


@pytest.mark.parametrize("name,options,inputs", (
    ("multiplexor", {}, None),
    ("structured", dict(lcu_select_implementation="structured"), None),
    ("control copies", dict(pair_control_copies=2), None),
    ("Richardson members", dict(pair_product_formula="richardson"), None),
    ("MPF members", dict(pair_product_formula="mpf"), None),
    ("vectors inside", dict(duhamel_nodes=2), "vector"),
    ("occupations inside", dict(duhamel_nodes=2), "occupation"),
), ids=lambda value: value if isinstance(value, str) else "")
def test_sampled_pair_memory_law_is_the_mean_of_its_rows(name, options, inputs, tmp_path):
    """One value stands for every row, so a uniform experiment's point count times it gives the total.

    Relation: an occurrence of weight w with x X and y Y factors holds 2x + 4y
    basis gates, 2(w-1) parity CX and a two-CX two-RZ rotation, and a row adds
    its branch gates, readout basis and inputs (sampled_plan._pair_memory). The
    built circuits of all rows supply the mean, which the estimate never builds.
    The mean is rounded up, so one position above an exact multiple of the rows
    takes one more position per row and the point count times it never falls
    short of the summed model.
    """
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation
    from nwqlib.algorithms.lchs.sampled_plan import _ROW_HANDLE_BYTES, _pair_memory
    from nwqlib.blocks.selection import NATIVE_POSITION_BYTES

    A = ingest_pauli([("ZI", .4j), ("XX", .1), ("ZZ", .2j), ("YI", .3), ("II", .5)], num_qubits=2)
    state = {} if inputs is None else (
        dict(source=[.1, .2, .3, .4], initial_state=[.6, .8, 0., 0.]) if inputs == "vector" else
        dict(source=ingest_occupation("11", num_qubits=2), initial_state=ingest_occupation("00", num_qubits=2)))
    chosen = plan(LinearDynamics(A=A, time=.5, **({"initial_state": [.6, .8, 0., 0.]} | state)),
                  method=LCHS(observable_estimator="sampled_pairs", hamiltonian_evolution_backend="trotter", trotter_steps=4, **options),
                  output=NormSquared(), shots=16, seed=7)
    block, built = _sampled_row_positions(chosen)
    positions = (block._native_bytes-_ROW_HANDLE_BYTES)/NATIVE_POSITION_BYTES
    assert built/2 <= positions <= 2*built
    if inputs is None:
        payload, rec, problem = block._payload
        multiple = 1000*rec.pair_rows.clusters
        assert (_pair_memory(payload, rec, problem, multiple + 1)["native_bytes"]
                - _pair_memory(payload, rec, problem, multiple)["native_bytes"]) == NATIVE_POSITION_BYTES
    loaded = load_plan(save_plan(chosen, ArchiveFiles(tmp_path, 10**8)), ArchiveFiles(tmp_path, 10**8))
    reloaded = next(item for item in loaded.blocks if item.record.signature.name == "lchs_pair")
    assert (reloaded._native_bytes, reloaded._definition_bytes) == (block._native_bytes, block._definition_bytes)


def test_sampled_pair_preparation_compares_a_row_and_the_whole_request_before_building(monkeypatch):
    """A huge step count is refused before its template is walked, and all rows are compared together.

    The original witness walked a row toward 3.4 GB under max_bytes=2e9. A limit
    between one row and all rows admits the first and refuses the whole request.
    """
    from nwqlib import prepare
    from nwqlib.algorithms.lchs import sampled_native
    from nwqlib.operators import ingest_pauli
    A = ingest_pauli([("ZI", .4j), ("XX", .1), ("ZZ", .2j), ("YI", .3)], num_qubits=2)
    problem = LinearDynamics(A=A, initial_state=[.6, .8, 0., 0.], time=.5)
    method = dict(observable_estimator="sampled_pairs", hamiltonian_evolution_backend="trotter")
    huge = plan(problem, method=LCHS(**method, trotter_steps=10**11, max_trotter_steps=10**12, max_bytes=2*10**9),
                output=NormSquared(), shots=8, seed=7)
    with monkeypatch.context() as patched:
        patched.setattr(sampled_native, "_walk_select", lambda *a, **k: pytest.fail("the word was walked"))
        with pytest.raises(ValueError, match="preparing the rows of these templates needs about .* exceeding max_bytes"):
            prepare(huge, progress=False)
    rows = plan(problem, method=LCHS(**method, trotter_steps=4, max_bytes=10**7), output=NormSquared(), shots=1024, seed=7)
    assert rows.reconstruction.pair_rows.clusters*(32 << 10) > 10**7
    assert prepare(rows, progress=False).circuits
    with pytest.raises(ValueError, match="exceeding max_bytes"):
        prepare(rows, settings="all", progress=False)


def test_sampled_pair_templates_are_compared_together_before_any_is_built(tmp_path, monkeypatch):
    """Eight rows of five templates are compared with the limit before any template exists.

    With 64 steps, each template fits a limit of one byte below their sum and all
    of them do not, which is the early-construction witness of this route. The
    expected total comes from each built circuit's own N entries and C CRZ entries,
    not from the key: B = 16384 + 256*N + 1024*C (sampled_native.pair_template_size),
    and the Run's request (_prepared_execution._admit_template_rows) is

        8 row handles + 5*(native estimate + 2 measurement positions)
        + largest(3 logical instructions + definition + 3*B + 2*B_before)

    These are declared allowances, so this checks the comparison, not a footprint.
    One byte below the total, and at that limit, nothing is built or charged. At the
    total each template is built and lowered once and every row is prepared, and
    reuse and reopening build nothing more.
    """
    import qiskit
    from nwqlib import load_run
    from nwqlib._prepared_execution import Run, prepare_static
    from nwqlib.algorithms.lchs import sampled_native
    from nwqlib.blocks import lowering
    from test_ibm_backend import sampled_lchs_plan

    base = sampled_lchs_plan(lcu_select_implementation="structured", pair_control_copies=2)

    def limited(limit):
        return plan(base.problem, method=base.method.revise(trotter_steps=64, max_bytes=limit), output=base.output,
                    shots=base.shots, seed=11)

    chosen = limited(None)
    points = tuple(chosen._static_points())
    created, circuit_init = [], qiskit.QuantumCircuit.__init__
    with monkeypatch.context() as patched:
        patched.setattr(qiskit.QuantumCircuit, "__init__",
                        lambda self, *a, **k: (created.append(1), circuit_init(self, *a, **k))[1])
        templates = chosen.method.row_templates(chosen, points)
    assert not created and len(points) == 8 and len(templates) == 5
    circuits = [template.circuit for template in templates]
    sizes = [16384 + 256*len(circuit.data) + 1024*circuit.count_ops().get("crz", 0) for circuit in circuits]
    assert [template._retained_bytes for template in templates] == sizes
    assert max(sizes) < sum(sizes) - 1
    block = next(item for item in chosen.blocks if item.record.signature.name == "lchs_pair")
    # The Program calls the pair block once and measures the branch and system qubits,
    # so it has 3 logical instructions, and 2 native positions beside the block's estimate.
    logical, measurements = 3*256, 2*200
    total = (8*(32 << 10) + 5*(block._native_bytes + measurements)
             + max(logical + block._definition_bytes + 3*size + 2*before for size, before in zip(sizes, (0, *sizes))))

    built, lowered = [], []
    build, lower = sampled_native.build_pair_template, lowering._lower_qiskit
    monkeypatch.setattr(sampled_native, "build_pair_template", lambda *a, **k: built.append(1) or build(*a, **k))
    monkeypatch.setattr(lowering, "_lower_qiskit", lambda *a, **k: lowered.append(1) or lower(*a, **k))
    for limit in (total - 1, sum(sizes) - 1):
        below = limited(limit)
        with Run(below, progress=False) as run:
            with pytest.raises(ValueError, match=f"preparing the rows of these templates needs about {total} bytes"):
                prepare_static(below, run=run, settings="all")
            assert run.trace.preparations == 0 and run._state["native_held"] == 0
        assert not built and not lowered
    fits = limited(total)
    with Run(fits, directory=tmp_path / "run", progress=False) as run:
        prepare_static(fits, run=run, settings="all")
        assert run.trace.preparations == 8 and (len(built), len(lowered)) == (5, 5)
        prepare_static(fits, run=run, settings="all")
        assert run.trace.preparations == 8 and len(built) == 5
    with load_run(tmp_path / "run", progress=False) as run:
        run.resume()
        assert run.result is not None and (len(built), len(lowered)) == (5, 5)


def test_lchs_refusals_name_a_remedy_that_plans():
    """Each refusal's advice leads past it, and differs from what the input has already set.

    A scalar L reduces to one unitary, so an analytic kernel PREP has no kernel to
    prepare. A compact A has only the coherent QSP route, whose SELECT is automatic.
    Dense input keeps its own advice, which also plans.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.operators import ingest_pauli
    A = ingest_pauli([("I", .3), ("Z", .4j)], num_qubits=1)
    compact = LinearDynamics(A=A, initial_state=[.6, .8j], time=.5)
    dense = LinearDynamics(A=np.diag([.3+.4j, .3-.4j]), initial_state=[.6, .8j], time=.5)
    qsp = dict(hamiltonian_evolution_backend="qsp_block_encoding")

    with pytest.raises(ApplicabilityError, match="lcu_state_preparation='direct'") as refused:
        plan(compact, method=LCHS(**qsp, lcu_state_preparation="analytic", lchs_kernel="low_somma_f2",
                                  k_quadrature="symmetric_uniform_trapezoid"), execution="quantum")
    assert "use lchs_kernel" not in str(refused.value)
    plan(compact, method=LCHS(**qsp, lcu_state_preparation="direct"), execution="quantum")

    with pytest.raises(ApplicabilityError, match="observable_estimator='sampled_pairs'") as refused:
        plan(compact, method=LCHS(lcu_select_implementation="multiplexor"), execution="quantum")
    assert "name the backend" not in str(refused.value)
    plan(compact, method=LCHS(), execution="quantum")
    plan(compact, method=LCHS(observable_estimator="sampled_pairs", hamiltonian_evolution_backend="trotter",
                              lcu_select_implementation="multiplexor"), output=NormSquared(), shots=8, seed=3)
    with pytest.raises(ApplicabilityError, match="dense A") as refused:
        plan(compact, method=LCHS(lcu_select_implementation="branch_controlled"), execution="quantum")
    assert "sampled_pairs" not in str(refused.value)
    with pytest.raises(ApplicabilityError, match="name the backend"):
        plan(dense, method=LCHS(lcu_select_implementation="multiplexor"), execution="classical")
    plan(dense, method=LCHS(lcu_select_implementation="multiplexor", hamiltonian_evolution_backend="trotter"),
         execution="quantum", output=NormSquared())

    tolerance = dict(hamiltonian_evolution_backend="trotter", trotter_steps=None, trotter_synthesis_tolerance=.01)
    with pytest.raises(ApplicabilityError, match="'qsp_block_encoding'") as refused:
        plan(compact, method=LCHS(**tolerance), execution="quantum")
    assert "trotter_error_budgeted" not in str(refused.value)
    with pytest.raises(ApplicabilityError, match="trotter_error_budgeted"):
        plan(dense, method=LCHS(**tolerance), execution="classical")
    plan(dense, method=LCHS(hamiltonian_evolution_backend="trotter_error_budgeted"), execution="classical")
