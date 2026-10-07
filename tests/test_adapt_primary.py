"""Actual ADAPT controller and selected-query relations on the common Plan/Run."""

from math import cos, pi, sin
from types import SimpleNamespace
import cmath
import numpy as np
import pytest
from nwqlib.algorithms.gcim import ADAPT, adapt_acquisition as acquisition
from nwqlib.core.planning import RandomStreams
from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
from nwqlib.operators import ingest_pauli
from nwqlib.problems.records import Eigenproblem


def plan_for(
    *,
    execution="quantum",
    shots=None,
    A=None,
    initial_state=(1, 1),
    pool=None,
    pool_rows=((("Y", 1j),), (("Y", 1j), ("I", 1j))),
    seed=7,
    **settings,
):
    operator = ingest_pauli((("Z", 1.0),), num_qubits=1) if A is None else A
    problem = Eigenproblem(A=operator)
    pool = (
        tuple(ingest_pauli(rows, num_qubits=len(rows[0][0])) for rows in pool_rows)
        if pool is None
        else pool
    )
    config = dict(theta=pi / 8, max_iterations=2)
    config.update(settings)
    method = ADAPT(initial_state=initial_state, pool=pool, **config)
    return method.plan(
        problem,
        output=problem.default_output(),
        execution=execution,
        shots=shots,
        rng=RandomStreams(seed),
    )


def matrix(rows):
    return np.array([[complex(z.real, z.imag) for z in row] for row in rows])


@pytest.mark.parametrize("execution", ["quantum", "classical"])
def test_product_screen_and_tie_survive_exact_ritz_solution(execution):
    result = Run(plan_for(execution=execution)).wait(timeout=10, poll_interval=0)
    assert dict(result.history[0].gradients) == pytest.approx({0: 2.0, 1: 2.0}, abs=1e-12, rel=0)
    assert result.history[0].winner == 0
    assert result.history[1].energy == pytest.approx(-1.0, abs=2e-12, rel=0)
    assert dict(result.history[1].gradients)[1] == pytest.approx(np.sqrt(2), abs=1e-12, rel=0)
    assert result.selected == (0, 1)
    assert all(step.residual_norm is None for step in result.history)


def test_point_fields_bind_actual_chain_and_readout():
    plan = plan_for()
    point = acquisition._point(plan, "pair", right=((0, 0.2),), term=0)
    assert point.plan_id == plan.content_id
    bad = tuple(b.revise(value=0) if b.parameter == "control_width" else b for b in point.bindings)
    with pytest.raises(ValueError, match="query width"):
        plan.resolve("pair", bindings=bad)
    other = plan_for(pool_rows=((("X", 1j),), (("Y", 1j),)))
    assert plan.content_id != other.content_id
    assert plan.construction.content_id != other.construction.content_id


@pytest.mark.parametrize("sign", [None, -1, 1])
def test_classical_insertion_energy_matches_analytic_rotation(sign):
    # exp(theta iY)|0> = cos(theta)|0> - sin(theta)|1>, so <Z> = cos(2 theta).
    # The inserted factor (I + i s Y)/sqrt(2) = exp(i s pi Y/4) adds s pi/4 to
    # the rotation angle: <Z> = cos(2 theta + s pi/2) = -s sin(2 theta).
    theta = 0.2
    plan = plan_for(
        execution="classical", initial_state=(1, 0), pool_rows=((("Y", 1j),),), max_iterations=1
    )
    insertion = None if sign is None else (0, 0, sign)
    point = acquisition._point(plan, "energy", right=((0, theta),), insertion=insertion)
    with Run(plan) as run:
        handle = prepare_experiment(point, run=run)
        value = submit_experiment(handle, run=run).values[0].value
    assert handle.record.execution == "host_kernel"
    expected = cos(2 * theta) if sign is None else -sign * sin(2 * theta)
    # A few binary64 products on a unit two-component state: error of a few ulp.
    assert value == pytest.approx(expected, rel=0, abs=1e-14)


def test_complex_pair_orientation_keeps_executed_sources(monkeypatch):
    """Use analytic complex overlaps to expose conjugation errors when an adaptive basis changes
    ordering.
    """
    from nwqlib.execution import ReducedValues, ScalarValue

    rows = tuple((("Z", 1j * k), ("I", 1j * k)) for k in (1, 2, 3, 4))
    selected = tuple(enumerate((0.1, 0.2, 0.3, 0.4)))
    shift = 0.37
    monkeypatch.setattr(
        acquisition,
        "_solve_projected_pencil",
        lambda *a, **k: (None, None, "injected no eigensolve", None),
    )
    for execution in ("classical", "quantum"):
        plan = plan_for(
            execution=execution,
            A=ingest_pauli((("Z", 1.0), ("I", shift)), num_qubits=1),
            pool_rows=rows,
            max_iterations=4,
        )
        context = acquisition.AdaptContext()
        acquired = []
        pencils = []
        for current in (selected[:3], selected):
            for _, _, point in acquisition._matrix_points(plan, current, context.observations):
                if point.content_id in context.observations:
                    continue
                values = {b.parameter: b.value for b in point.bindings}
                left, right = acquisition.chain(values, "l"), acquisition.chain(values, "r")
                if point.experiment == "screen":
                    left = right
                delta = sum((i + 1) * theta for i, theta in right) - sum(
                    (i + 1) * theta for i, theta in left
                )
                phase = cmath.exp(2j * delta)
                s = (1 + phase) / 2
                h0 = (phase - 1) / 2
                if execution == "classical":
                    h = h0 + shift * s
                    data = dict(s_real=s.real, s_imag=s.imag, h_real=h.real, h_imag=h.imag)
                    statistics = tuple(
                        ScalarValue(label=name, value=value) for name, value in data.items()
                    )
                else:
                    # An exact pair reduction returns (H0, S) of the canonical
                    # chain pair, the orientation of its bindings. The full
                    # chain's diagonal is the shared query's diagonal point, whose
                    # right chain alone gives (H0, S) = (0, 1) here.
                    statistics = (
                        ReducedValues(component=0, real=(h0.real, s.real), imaginary=(h0.imag, s.imag)),
                    )
                context.observations[point.content_id] = SimpleNamespace(
                    realization_id=point.content_id,
                    bindings=point.bindings,
                    population="unconditional",
                    values=statistics,
                    prepared_id=None,
                    point="diagonal" if point.experiment == "screen" else None,
                )
                acquired.append(point.content_id)
            pencil, _ = acquisition._pencil_from_observations(plan, current, context)
            pencils.append(pencil)
            size = len(acquisition.basis_chains(current))
            assert len(acquired) == (
                size * (size + 1) // 2
            )
        old_h, new_h = (matrix(p.hamiltonian) for p in pencils)
        old_s, new_s = (matrix(p.overlap) for p in pencils)
        assert abs(old_s[4, 5].imag) > 0.1 and abs(old_h[4, 5].imag) > 0.1
        assert new_s[6, 7] == pytest.approx(old_s[4, 5].conjugate(), abs=1e-14, rel=0)
        assert new_h[6, 7] - shift * new_s[6, 7] == pytest.approx(
            (old_h[4, 5] - shift * old_s[4, 5]).conjugate(), abs=1e-14, rel=0
        )


def test_optimizer_global_query_limit_and_saved_incumbent():
    plan = plan_for(
        execution="classical",
        max_iterations=1,
        optimize_every_m=1,
        optimize_rounds_n=20,
        optimize_max_evaluations=3,
    )
    result = Run(plan).wait(timeout=10, poll_interval=0)
    assert result.energy_queries <= 3 and result.optimizer_attempts == 1
    assert result.stop_reason == "optimization_evaluation_budget_exhausted"
    assert result.eigenvalue is not None


def test_quantum_optimizer_reaches_analytic_minimum_through_insertion_queries():
    # exp(theta iY)|+> = cos(theta)|+> + sin(theta)|->, so E(theta) = sin(2 theta).
    # From theta = pi/8 the nearest minimum is theta = -pi/4. BFGS stops at
    # |E'| = |2 cos(2 theta)| < 1e-5; with E'' = 4 there, theta is within
    # 2.5e-6 of the minimum.
    plan = plan_for(
        pool_rows=((("Y", 1j),),),
        max_iterations=1,
        optimize_every_m=1,
        optimize_rounds_n=20,
        optimize_max_evaluations=60,
    )
    result = Run(plan).wait(timeout=10, poll_interval=0)
    assert result.optimizer_attempts == 1 and result.stop_reason == "pool_exhausted"
    assert result.theta[0] == pytest.approx(-pi / 4, abs=1e-5, rel=0)


def test_quantum_optimizer_cadence_runs_only_at_multiples_of_m():
    # H = Z, pool iY and i(Y + I). The identity term is a phase, so the product
    # state is exp(phi iY)|+> with phi = theta_0 + theta_1, E = sin 2phi, and
    # both screen gradients are 2<X> = 2 cos 2phi. With theta = pi/12 and m = 2:
    # count 1: the tie (2, 2) at phi = 0 selects 0; 1 % 2 != 0, no optimization.
    # count 2: 1 is selected; 2 % 2 == 0, so one BFGS round starts at phi = pi/6,
    #   where dE/dtheta_i = 2 cos(pi/3) = 1 > 0 moves both coordinates down.
    # The pool is then exhausted. Optimizing at count 1 records the attempt in
    # history[1]; never optimizing leaves both coordinates at pi/12.
    plan = plan_for(
        theta=pi / 12,
        optimize_every_m=2,
        optimize_rounds_n=1,
        optimize_max_evaluations=60,
    )
    result = Run(plan).wait(timeout=10, poll_interval=0)
    assert result.selected == (0, 1) and result.stop_reason == "pool_exhausted"
    assert [step.optimizer_attempts for step in result.history] == [0, 0, 1]
    assert result.theta[0] < pi / 12 and result.theta[1] < pi / 12


def test_residual_stop_at_first_projection_below_threshold():
    # H=Z from |+>. At theta=pi/4, 4iY rotates by pi, a sign change, so the
    # first projection still spans |+>: E=0 and ||Z|+>||=1. The next generator
    # iY adds |0>, which completes the qubit: E=-1 and residual 0. The unused
    # third generator keeps pool exhaustion from ending the loop there.
    plan = plan_for(
        execution="classical",
        pool_rows=((("Y", 4j),), (("Y", 1j),), (("Y", 0.5j),)),
        theta=pi / 4,
        max_iterations=3,
        stop_criterion="residual_norm",
        residual_norm_tolerance=0.5,
    )
    result = Run(plan).wait(timeout=10, poll_interval=0)
    # Exact values; the only error is binary64 roundoff in the unit-scale solve.
    residuals = [step.residual_norm for step in result.history]
    assert residuals == pytest.approx([1.0, 1.0, 0.0], abs=1e-12, rel=0)
    energies = [step.energy for step in result.history]
    assert energies == pytest.approx([0.0, 0.0, -1.0], abs=1e-12, rel=0)
    assert result.stop_reason == "residual_norm_threshold" and result.selected == (0, 1)
    # The contract is residual <= tolerance. The same first residual, computed
    # the same way, stops at equality and continues one ulp below it.
    from math import nextafter

    for tolerance, selected in ((residuals[0], (0,)), (nextafter(residuals[0], 0.0), (0, 1))):
        boundary = plan_for(
            execution="classical",
            pool_rows=((("Y", 4j),), (("Y", 1j),), (("Y", 0.5j),)),
            theta=pi / 4,
            max_iterations=3,
            stop_criterion="residual_norm",
            residual_norm_tolerance=tolerance,
        )
        stopped = Run(boundary).wait(timeout=10, poll_interval=0)
        assert stopped.stop_reason == "residual_norm_threshold" and stopped.selected == selected


def test_cancellation_stops_even_algebraic_controller_advancement():
    plan = plan_for(A=ingest_pauli((("I", 0.0),), num_qubits=1))
    run = Run(plan)
    before = run.rng.snapshot()
    run.cancel()
    result = plan.method.execute(plan, run=run)
    assert (
        result.stop_reason == "cancelled" and not run.trace.events and run.rng.snapshot() == before
    )


def test_matrix_stage_limit_refuses_before_its_first_query_and_extension_resumes_it():
    """A circuit limit below one quantum H2 round refuses its two-state matrix stage before any of its queries.

    The count law gives the stage need from the Plan alone, and a run under a
    sufficient limit acquires exactly the round it predicts. After the refusal,
    raising the limit and resuming the same Run gives the uninterrupted run's
    seeds, selection and energy.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    import nwqlib
    from nwqlib.algorithms.gcim import build_gcim_chemistry_problem
    from nwqlib.algorithms.gcim.pencil import pair_count
    from nwqlib.execution import ExecutionLimits

    molecule = build_gcim_chemistry_problem("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    method = molecule.adapt_method(pool="spin_adapted_sd", theta=pi / 4, max_iterations=1)
    selected = nwqlib.plan(molecule.eigenproblem(), method=method, execution="quantum", seed=7)
    terms = len(selected.reconstruction.nonidentity_terms)
    first = pair_count(1, terms)  # the one-state pencil, whose shared query also serves the first screen
    stage = pair_count(2, terms) - pair_count(1, terms)
    reference = nwqlib.solve(selected, limits=ExecutionLimits(max_total_circuits=first + stage), progress=False)
    assert len(reference.data.trace.events) == first + stage
    prepared = nwqlib.prepare(selected, limits=ExecutionLimits(max_total_circuits=first + stage - 1), progress=False)
    with prepared.run as run:
        with pytest.raises(ValueError, match=rf"\({first} counted, {stage} requested, at least {first + stage} needed\)"):
            nwqlib.submit(prepared)
        assert len(run.trace.events) == first
        run.extend_limits(max_total_circuits=first + stage)
        run.resume()
        result = run.wait(timeout=60, poll_interval=0)
    assert (result.selected, result.eigenvalue) == (reference.selected, reference.eigenvalue)
    seeds = [[receipt.runtime.seed for receipt in r.data.receipts] for r in (result, reference)]
    assert seeds[0] == seeds[1]


def test_current_sampled_diagnostic_does_not_inherit_history_warning(tmp_path, monkeypatch):
    from nwqlib import solve, load_result

    chosen = plan_for(shots=16, seed=1, pool_rows=((('Y',1j),),), theta=pi/4, max_iterations=1)
    result = solve(chosen)
    assert result.eigenvalue < -1. and result.pencil.failure_reason is None
    assert result.pencil.sampled_failure == 'ritz_outside_operator_enclosure'
    assert result.stop_reason == 'pool_exhausted' and 'outside numerical window of processed Pauli enclosure' in str(result)
    monkeypatch.setattr(acquisition, 'submit_experiment', lambda *a,**k: pytest.fail('reanalysis acquired data'))
    changed = result.analyze(overlap_cutoff=1.)
    assert changed.history == result.history and changed.stop_reason == result.stop_reason
    assert changed.pencil.sampled_enclosure_passed is True
    assert 'outside numerical window of processed Pauli enclosure' not in str(changed)
    assert 'within numerical window of processed Pauli enclosure' in str(changed)
    loaded = load_result(changed.save(tmp_path/'changed-cutoff'))
    assert loaded.pencil == changed.pencil and loaded.eigenvalue == changed.eigenvalue
    assert loaded.history == result.history


@pytest.mark.parametrize('fault', ['population','bindings','realization','labels'])
def test_matrix_frontier_rejects_foreign_scalar_meaning_before_assembly(fault):
    from nwqlib import solve

    result = solve(plan_for(execution='classical', max_iterations=1))
    selected = ()
    _, _, point = next(acquisition._matrix_points(result.plan, selected))
    original = next(c for c in result.data.observations.chunks if c.realization_id == point.content_id)
    context = acquisition.AdaptContext()
    with pytest.raises(ValueError, match='exact query|unconditional|missing|undeclared|populations|ordered names'):
        if fault == 'population':
            changed = original.revise(population='unknown')
        elif fault == 'bindings':
            changed = original.revise(bindings=())
        elif fault == 'realization':
            changed = original.revise(realization_id='sha256:'+'0'*64)
        else:
            changed = original.revise(values=(original.values[0].revise(label='foreign'), *original.values[1:]))
        context.observations[point.content_id] = changed
        acquisition._matrix_from_observations(result.plan, selected, context)
    assert context.matrix_data is None


@pytest.mark.parametrize("execution, qubits, seed, draws", [("classical", 16, 1, 1), ("quantum", 10, 0, 3)])
def test_rank_deficient_basis_is_admitted_at_its_measured_gram_error(execution, qubits, seed, draws):
    from nwqlib import solve
    from nwqlib._projected_eigensolver import gram_formation_allowance
    from nwqlib.execution import ExecutionLimits

    # Qubit 0 starts in cos(.4)|0> + sin(.4)|1> and the other qubits in a
    # fixed spread state |r>. The pool iY_0, 2iY_0 keeps all four basis states
    # in span{|0>, |1>} x |r>, so the exact Gram matrix has two zero
    # eigenvalues, reached along different arithmetic paths. For the classical
    # entries, inner products of length 2**16, the computed ones fall below
    # zero by more than the eigensolver roundoff of S, which an allowance for
    # that roundoff alone refuses. The quantum entries are exact pair
    # reductions of Aer states.
    n = qubits
    hamiltonian = ingest_pauli((("I" * (n - 1) + "Z", 1.), ("X" + "I" * (n - 1), .3)), num_qubits=n)
    pool = tuple(ingest_pauli((("I" * (n - 1) + "Y", scale),), num_qubits=n) for scale in (1j, 2j))
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        rest = rng.normal(size=2**(n - 1)) + 1j * rng.normal(size=2**(n - 1))
    rest /= np.linalg.norm(rest)
    state = np.kron(rest, [cos(.4), sin(.4)])
    result = solve(Eigenproblem(A=hamiltonian),
                   method=ADAPT(initial_state=state, pool=pool, max_iterations=2, gradient_norm_floor=0.),
                   execution=execution, seed=7, limits=ExecutionLimits(max_total_circuits=20000))
    pencil = result.pencil
    spectrum = sorted(pencil.overlap_eigenvalues)
    assert pencil.failure_reason is None and pencil.kept_rank == 2 and min(spectrum[2:]) > .1
    # The kept span is span{|0>, |1>} x |r>, where the lowest Ritz value of
    # Z_0 + .3 X_(n-1) is -1 + .3 <r|X|r>. To first order the kept pencil
    # moves it by at most (||dH|| + |E| ||dS||)/s_kept. ||dS|| is within the
    # solve's allowance, ||dH|| within that allowance times the Pauli L1 norm
    # 1.3 of H0, and s_kept > .1, so 26 allowances bound the change. The exact
    # quantum entries are pair reductions, complex contractions of length
    # 2**n. Their receipts supply state errors, and the solve admits their Gram
    # at the row-sum allowance of the pair overlap bounds
    # (fixed_basis._exact_overlap_allowance). For them the window below uses
    # the smaller dot-rounding allowance instead, so it is a regression window
    # of the fixture, tighter than that derived allowance, not a certificate.
    allowance = gram_formation_allowance(2**n, sum(spectrum))
    half = 2**(n - 2)
    expected = -1 + .3 * 2 * np.vdot(rest[:half], rest[half:]).real
    assert result.eigenvalue == pytest.approx(expected, rel=0, abs=26 * allowance)


def test_exact_pair_stage_work_is_refused_before_its_first_query():
    """An exact ADAPT matrix stage is admitted against max_products as a whole, before any of its pair queries.

    Each exact pair reduction registers ``pair_reducer.pair_work`` host work,
    and each query alone fits a cap of one pair's work. A cap one unit below
    the stage's summed work therefore refuses at the stage gate, and the Run
    has acquired none of that stage's queries. The stage need is the
    reference run's acquisition count of that stage times the registered work
    of one pair, independent of the gate's own count law.
    """
    import json
    import nwqlib
    from nwqlib.algorithms.gcim.fixed_basis import pair_reduction_point
    from nwqlib.algorithms.gcim.pair_reducer import pair_work

    # At ten qubits one pair reduction (about 1e4 products) dominates the
    # planning work of this two-term Hamiltonian and two-member pool.
    n = 10
    hamiltonian = ingest_pauli((("I" * (n - 1) + "Z", 1.), ("X" + "I" * (n - 1), .3)), num_qubits=n)
    pool = tuple(ingest_pauli((("I" * (n - 1) + "Y", scale),), num_qubits=n) for scale in (1j, 2j))
    state = np.kron(np.full(2**(n - 1), 2**(-(n - 1) / 2)), [cos(.4), sin(.4)])

    def planned(max_products):
        return nwqlib.plan(Eigenproblem(A=hamiltonian), method=ADAPT(
            initial_state=state, pool=pool, max_iterations=2, gradient_norm_floor=0., max_products=max_products),
            execution="quantum", seed=7)

    reference = nwqlib.solve(planned(10**6), progress=False)
    rec = reference.plan.reconstruction
    work = pair_work(json.loads(pair_reduction_point(
        rec.processed_hamiltonian.identity, [t.coefficient for t in rec.nonidentity_terms], rec.num_qubits,
        diagonal=False).parameters))
    # The largest stage of k queries is the first one refused by a cap of
    # k*work - 1 as k decreases from the reference run's acquisition total.
    total = len(reference.data.trace.events)
    for k in range(total, 1, -1):
        try:
            nwqlib.solve(planned(k * work - 1), progress=False)
        except ValueError:
            break
    need = k * work
    prepared = nwqlib.prepare(planned(need - 1), progress=False)
    with prepared.run as run:
        with pytest.raises(ValueError, match=rf"action needs {need} scalar products, exceeding max_products={need - 1}"):
            nwqlib.submit(prepared)
        before = len(run.trace.events)
    # The refused stage is the last one, and none of its k queries ran.
    assert total - before == k
    admitted = nwqlib.solve(planned(need), progress=False)
    assert admitted.eigenvalue == reference.eigenvalue


def test_exact_off_diagonal_pair_bounds_cover_the_dense_entry_error(monkeypatch):
    """Exact ADAPT's two-state pair receipts are qualified, and their beta = 2 bounds cover the entry error.

    For an off-diagonal pair the overlap and H0 entries are bounded by
    ``pair_reducer.entry_bounds`` with beta = 2 from the state error of the
    pair's own receipt, and the solve's Gram allowance sums those overlap
    bounds. The independent reference is the dense product of
    ``scipy.linalg.expm`` chain states. Every off-diagonal entry error lies
    within its own bound, so within the largest one checked here. The test
    fails when a receipt leaves the roundoff model and has no state error, as
    with a matrix-form ancilla flip, when the bounds the solve receives are
    not those of the receipts, or when a bound falls below the dense error.
    """
    import json
    from scipy.linalg import expm
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib import solve
    from nwqlib.algorithms.gcim.adapt_acquisition import basis_chains
    from nwqlib.algorithms.gcim.pair_reducer import LAYOUT_OFF_DIAGONAL, entry_bounds

    terms = (("ZII", .6), ("IXZ", -.35), ("YYI", .25), ("XIX", .2), ("IZZ", -.4))
    labels = ("IIY", "IYZ", "YXI", "XYX")
    rng = np.random.default_rng(4)
    reference = rng.normal(size=8) + 1j * rng.normal(size=8)
    reference /= np.linalg.norm(reference)
    received = []
    allowance = acquisition._exact_overlap_allowance

    def spy(basis_size, pair_bounds):
        received.append({pair: bound for pair, bound in pair_bounds.items() if pair[0] != pair[1]})
        return allowance(basis_size, pair_bounds)

    monkeypatch.setattr(acquisition, "_exact_overlap_allowance", spy)
    result = solve(Eigenproblem(A=ingest_pauli(terms, num_qubits=3)),
                   method=ADAPT(initial_state=reference, pool=tuple(ingest_pauli(((p, 1j),), num_qubits=3)
                                                                    for p in labels),
                                theta=.37, max_iterations=2, gradient_norm_floor=0.),
                   execution="quantum", seed=5)
    generators = [1j * SparsePauliOp(label).to_matrix() for label in labels]

    def state(chain):
        vector = reference
        for index, theta in chain:
            vector = expm(theta * generators[index]) @ vector
        return vector

    chains = basis_chains(tuple(zip(result.selected, result.theta)))
    assert len(chains) >= 2
    v = np.column_stack([state(chain) for chain in chains])
    h, s = (np.array([[complex(z.real, z.imag) for z in row] for row in m])
            for m in (result.pencil.hamiltonian, result.pencil.overlap))
    off = ~np.eye(len(chains), dtype=bool)
    receipts = {r.content_id: r for r in result.data.receipts}
    bounds = []
    for chunk in result.data.observations.chunks:
        for point in chunk.observation.positions:
            parameters = json.loads(point.parameters) if point.kind == "reduction" else {}
            if parameters.get("layout") == LAYOUT_OFF_DIAGONAL:
                delta = receipts[chunk.prepared_id].saved_state_error(("amplitude-derived masses",))[0]
                assert delta is not None
                bounds.append(entry_bounds(delta, float.fromhex(parameters["c1"]), 1 << parameters["qubits"],
                                           parameters["terms"], diagonal=False))
    assert bounds and received[-1]
    assert set(received[-1].values()) <= {overlap for overlap, _ in bounds}
    overlap_bound, hamiltonian_bound = map(max, zip(*bounds))
    hamiltonian = SparsePauliOp.from_list(terms).to_matrix()
    assert np.max(np.abs((s - v.conj().T @ v)[off])) <= overlap_bound
    assert np.max(np.abs((h - v.conj().T @ hamiltonian @ v)[off])) <= hamiltonian_bound


def test_reanalysis_resolves_each_query_block_once_and_builds_no_construction(monkeypatch):
    """The ADAPT matcher misses the Plan's one-slot resolution memo once per query block.

    The Plan keeps only its most recent resolution. A forward pass over Q
    consecutive query blocks, each holding one or more point chunks of the
    same (experiment, bindings), therefore resolves Q points, and the
    matcher builds no selected construction. The expected Q is counted from
    the chunk sequence alone; a matcher that resolved per chunk (K > Q
    here, since the shared full-chain query supplies two point chunks) or
    rebuilt a construction per point fails.
    """
    import nwqlib
    from collections import Counter
    from nwqlib.core.planning import Plan

    plan = plan_for()
    result = nwqlib.solve(plan, progress=False)
    chunks = result.data.observations.chunks
    keys = [(c.experiment, c.bindings) for c in chunks]
    blocks = 1 + sum(a != b for a, b in zip(keys, keys[1:]))
    assert blocks >= 2 and len(chunks) > blocks

    misses = Counter()
    for name in ("_resolve_selection_once", "_selected_construction_once"):
        original = getattr(Plan, name)

        def counted(self, *args, _original=original, _name=name, **kwargs):
            misses[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Plan, name, counted)
    matched = tuple(acquisition._matched_query_chunks(plan, result.data))
    assert len(matched) == len(chunks)
    assert misses == Counter({"_resolve_selection_once": blocks})

    # A warm slot: the last block again resolves nothing new.
    misses.clear()
    last = chunks[-1]
    plan.resolve(last.experiment, bindings=last.bindings)
    assert misses == Counter()
