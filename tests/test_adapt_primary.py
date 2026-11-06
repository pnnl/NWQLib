"""Actual ADAPT controller and selected-query relations on the common Plan/Run."""

from math import cos, pi, sin
import numpy as np
import pytest
from nwqlib.algorithms.gcim import ADAPT, PencilAcquisition, adapt_acquisition as acquisition
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


@pytest.mark.parametrize("execution", ["quantum", "classical"])
def test_product_screen_and_tie_survive_exact_ritz_solution(execution):
    # The termwise request applies only to shots, so exact quantum readout
    # still prepares the controlled joint state of each off-diagonal pair of
    # the four-state value basis ((), a, b, ab), and classical execution none.
    termwise = PencilAcquisition(matrix_measurement="termwise", pair_preparation="controlled")
    result = Run(plan_for(execution=execution, acquisition=termwise)).wait(timeout=10, poll_interval=0)
    assert dict(result.history[0].gradients) == pytest.approx({0: 2.0, 1: 2.0}, abs=1e-12, rel=0)
    assert result.history[0].winner == 0
    assert result.history[1].energy == pytest.approx(-1.0, abs=2e-12, rel=0)
    assert dict(result.history[1].gradients)[1] == pytest.approx(np.sqrt(2), abs=1e-12, rel=0)
    assert result.selected == (0, 1)
    assert all(step.residual_norm is None for step in result.history)
    pairs = [((i, j), "controlled") for i in range(4) for j in range(i + 1, 4)]
    assert sorted((p.pair, p.route) for p in result.pair_preparations) == (
        pairs if execution == "quantum" else [])


@pytest.mark.parametrize("sign", [-1, 1])
def test_classical_insertion_energy_matches_analytic_rotation(sign):
    # exp(theta iY)|0> = cos(theta)|0> - sin(theta)|1>, so <Z> = cos(2 theta).
    # The inserted factor (I + i s Y)/sqrt(2) = exp(i s pi Y/4) adds s pi/4 to
    # the rotation angle: <Z> = cos(2 theta + s pi/2) = -s sin(2 theta).
    theta = 0.2
    plan = plan_for(
        execution="classical", initial_state=(1, 0), pool_rows=((("Y", 1j),),), max_iterations=1
    )
    point = acquisition._point(plan, "energy", right=((0, theta),), insertion=(0, 0, sign))
    with Run(plan) as run:
        handle = prepare_experiment(point, run=run)
        value = submit_experiment(handle, run=run).values[0].value
    assert handle.record.execution == "host_kernel"
    expected = -sign * sin(2 * theta)
    # A few binary64 products on a unit two-component state: error of a few ulp.
    assert value == pytest.approx(expected, rel=0, abs=1e-14)


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


def test_residual_stop_needs_an_explicit_tolerance_in_the_energy_unit():
    """The residual has the energy unit of H, so no default value can suit every unit."""
    pool = (ingest_pauli((("Y", 1j),), num_qubits=1),)
    with pytest.raises(ValueError, match="residual_norm_tolerance"):
        ADAPT(initial_state=(1, 1), pool=pool, stop_criterion="residual_norm")


def _two_qubit_adapt(operator, execution="classical", **settings):
    """ADAPT from |00> on a two-qubit H with five one-word generators whose gradients have no ties."""
    from nwqlib.problems import ingest_occupation

    pool = tuple(ingest_pauli(((p, c),), num_qubits=2)
                 for p, c in (("YI", 1j), ("IY", .8j), ("XY", .6j), ("YX", .5j), ("YZ", .3j)))
    problem = Eigenproblem(A=operator)
    method = ADAPT(initial_state=ingest_occupation((0, 0), num_qubits=2), pool=pool,
                   **dict(dict(theta=.37, max_iterations=4), **settings))
    return Run(method.plan(problem, output=problem.default_output(), execution=execution, shots=None,
                           rng=RandomStreams(7))).wait(timeout=60, poll_interval=0)


_TWO_QUBIT_TERMS = (("ZI", 1.0), ("IZ", .7), ("XX", .4), ("ZZ", .25), ("XI", .15), ("II", -.3))


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_optimized_run_follows_the_operator_scale(execution):
    """BFGS sees the same dimensionless function for H and H' = aH + bI, so the angles and E' = aE + b follow.

    The allowances 1e-4 rad and 1e-8 are regression allowances for this run, which can stop at its iteration limit
    before the product is stationary. They separate the failure in view, BFGS returning at its first point, which
    moves the angles by order one. The energy compared is the Ritz value of a full-rank basis, not the product energy.
    The Hartree run must iterate, or the comparison would pass with both runs at their first point.
    """
    settings = dict(optimize_every_m=1, optimize_rounds_n=4, optimize_max_evaluations=60, max_iterations=2)
    a, b = 4.3597447222071e-18, 3e-18
    rows = tuple((p, a * c + (b if p == "II" else 0.0)) for p, c in _TWO_QUBIT_TERMS)
    base = _two_qubit_adapt(ingest_pauli(_TWO_QUBIT_TERMS, num_qubits=2), execution, **settings)
    moved = _two_qubit_adapt(ingest_pauli(rows, num_qubits=2), execution, **settings)
    assert base.optimizer_rounds > 0 and moved.optimizer_rounds > 0
    assert moved.selected == base.selected and moved.stop_reason == base.stop_reason
    assert moved.theta == pytest.approx(base.theta, abs=1e-4, rel=0)
    # Recorded energies stay in the unit of H'.
    assert (moved.eigenvalue - b) / a == pytest.approx(base.eigenvalue, abs=1e-8, rel=0)


@pytest.mark.parametrize("representation", ["pauli", "dense", "csr", "csc", "pauli_quantum"])
def test_default_thresholds_follow_the_operator_scale(representation):
    """H' = aH + bI with a > 0 gives E' = aE + b and the same full selected sequence in every representation.

    The scale is the direct evaluation of ``||H - tr(H) I/d||_F / sqrt(d)`` on the dense matrix. At the
    reference |00> the hand values are E = 1 + .7 + .25 - .3 = 1.65 and g_2 = -2(.4)(.6) = -.48, against
    g_3 = -.4 and g_0 = -.3, far from a tie. The 1e-10 window on energies converted back to the unit of H
    is 100 times the 1e-12 of the same unit-scale solves in this file, and the failures it separates
    (energy 0 or the reference energy) differ by order one. Each run compares with its own representation,
    because a Pauli sum can be pruned and a classical matrix cannot, and no coefficient here is near the cutoff.
    """
    from scipy import sparse

    terms = _TWO_QUBIT_TERMS
    letter = dict(I=np.eye(2), X=np.array([[0, 1], [1, 0]]), Z=np.diag([1, -1]))
    matrix = sum(c * np.kron(letter[p[0]], letter[p[1]]) for p, c in terms)
    scale = np.linalg.norm(matrix - np.trace(matrix) / 4 * np.eye(4), "fro") / 2
    a, b = 4.3597447222071e-18, 3e-18

    def operator(s, shift):
        scaled = s * matrix + shift * np.eye(4)
        if representation == "dense":
            return scaled
        if representation in ("csr", "csc"):
            return getattr(sparse, representation + "_matrix")(scaled)
        return ingest_pauli(tuple((p, s * c + (shift if p == "II" else 0.0)) for p, c in terms), num_qubits=2)

    execution = "quantum" if representation == "pauli_quantum" else "classical"
    base, moved = _two_qubit_adapt(operator(1.0, 0.0), execution), _two_qubit_adapt(operator(a, b), execution)
    assert base.history[0].energy == pytest.approx(1.65, abs=1e-12, rel=0)
    assert dict(base.history[0].gradients)[2] == pytest.approx(-.48, abs=1e-12, rel=0)
    assert base.selected[0] == 2 and len(base.selected) >= 2
    assert moved.selected == base.selected and moved.stop_reason == base.stop_reason
    assert (moved.eigenvalue - b) / a == pytest.approx(base.eigenvalue, abs=1e-10, rel=0)
    assert [(step.energy - b) / a for step in moved.history] == pytest.approx(
        [step.energy for step in base.history], abs=1e-10, rel=0)
    # The scale of one operator agrees across its representations up to a few rounding errors of u = 1.1e-16.
    # The difference between d and a padded D, or between RMS and max, is at least a factor 1.15.
    rec, moved_rec = base.plan.reconstruction, moved.plan.reconstruction
    assert rec.energy_scale == pytest.approx(scale, rel=1e-14, abs=0.0)
    assert moved_rec.energy_scale == pytest.approx(a * scale, rel=1e-14, abs=0.0)
    assert moved_rec.gradient_norm_floor == pytest.approx(a * rec.gradient_norm_floor, rel=1e-14, abs=0.0)
    assert moved_rec.energy_change_tolerance == pytest.approx(a * rec.energy_change_tolerance, rel=1e-14, abs=0.0)
    assert moved_rec.pauli_coefficient_cutoff == pytest.approx(a * rec.pauli_coefficient_cutoff, rel=1e-14, abs=0.0)
    # What the user gave stays in the Method.
    assert base.plan.method.gradient_norm_floor is None and base.plan.method.pauli_coefficient_cutoff is None


@pytest.mark.parametrize("representation", ["dense", "csr", "csc"])
def test_energy_scale_of_a_padded_dimension_uses_the_original_dimension(representation):
    """A three-dimensional H is padded to dimension 4, and its scale keeps d = 3.

    The sparse forms store one of the three diagonal entries, and the direct dense evaluation counts the
    other two as zeros, each differing from the mean c = 2/3 by c. Mean padding leaves ``||H - cI||_F``
    unchanged, so dividing by sqrt(4) instead of sqrt(3) would be low by the factor 1.15 that the 1e-14
    window separates.

    H = .9 I is a multiple of the identity, so its scale and its energy need no computed reference: 0 and .9
    exactly. The rounded sum of .9/3 three times is not .9, so these fail if the mean of the diagonal comes from
    that sum. On the circuit route the padded operator must then be exactly .9 I, one Pauli word and no circuit.
    """
    from scipy import sparse
    from nwqlib import solve

    def form(array):
        return array if representation == "dense" else getattr(sparse, representation + "_matrix")(array)

    matrix = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    scale = np.linalg.norm(matrix - np.trace(matrix) / 3 * np.eye(3), "fro") / np.sqrt(3)
    generator = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    problem = Eigenproblem(A=form(matrix))
    plan = ADAPT(initial_state=[1, 0, 0], pool=(generator,), theta=.37, max_iterations=1).plan(
        problem, output=problem.default_output(), execution="classical", shots=None, rng=RandomStreams(7))
    assert plan.reconstruction.energy_scale == pytest.approx(scale, rel=1e-14, abs=0.0)
    constant = solve(Eigenproblem(A=form(.9 * np.eye(3))),
                     method=ADAPT(initial_state=[1, 0, 0], pool=(generator,), theta=.37, max_iterations=1,
                                  input_conversion="dense_pauli"),
                     execution="quantum", seed=7, progress=False)
    assert constant.plan.reconstruction.energy_scale == 0.0
    assert (constant.eigenvalue, constant.stop_reason) == (.9, "constant_hamiltonian")
    assert len(constant.plan.reconstruction.terms) == 1 and not constant.data.trace.events


@pytest.mark.parametrize("representation", ["pauli", "dense", "csr"])
@pytest.mark.parametrize("c", [0.75, 0.0])
def test_multiple_of_the_identity_ends_at_its_constant_energy(representation, c):
    """H = cI has energy c on every state, so the run ends after the reference without screening a gradient."""
    from scipy import sparse

    operator = (ingest_pauli((("II", c),), num_qubits=2) if representation == "pauli"
                else c * np.eye(4) if representation == "dense" else sparse.csr_matrix(c * np.eye(4)))
    result = _two_qubit_adapt(operator)
    assert (result.eigenvalue, result.selected, result.stop_reason) == (c, (), "constant_hamiltonian")
    assert result.plan.reconstruction.energy_scale == 0.0
    assert result.history[0].gradient_norm is None


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


@pytest.mark.parametrize("execution, qubits, seed, draws", [("classical", 16, 1, 1)])
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
    # that roundoff alone refuses.
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
    # 1.3 of H0, and s_kept > .1, so 26 allowances bound the change.
    allowance = gram_formation_allowance(2**n, sum(spectrum))
    half = 2**(n - 2)
    expected = -1 + .3 * 2 * np.vdot(rest[:half], rest[half:]).real
    assert result.eigenvalue == pytest.approx(expected, rel=0, abs=26 * allowance)


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


def test_exact_pair_bounds_are_computed_once_per_found_receipt(monkeypatch):
    """Each exact pair's entry bound is computed once its receipt is found and reused by later assemblies.

    H = 0.6 ZII - 0.35 IXZ + 0.25 YYI + 0.2 XIX - 0.4 IZZ, pool iIIY, iIYZ,
    iYXI, iXYX, theta = 0.37 and two iterations from a random 3-qubit state,
    solved as exact circuits. Across its assemblies the run computes one
    bound per pair of its final basis. The run's last assembly is then
    repeated from its context. An assembly without receipts has allowance 0 and keeps no bound,
    so a later assembly with the receipts computes one bound per basis pair
    and returns the run's allowance. A repeated assembly computes none.
    """
    from nwqlib import solve
    from nwqlib.algorithms.gcim import pair_reducer

    terms = (("ZII", .6), ("IXZ", -.35), ("YYI", .25), ("XIX", .2), ("IZZ", -.4))
    rng = np.random.default_rng(4)
    reference = rng.normal(size=8) + 1j * rng.normal(size=8)
    calls, assemblies = [], []
    entry_bounds, assemble = pair_reducer.entry_bounds, acquisition._matrix_from_observations

    def counted(*args, **kwargs):
        calls.append(args)
        return entry_bounds(*args, **kwargs)

    def kept(plan, basis, context, **kwargs):
        assemblies.append((plan, basis, context, kwargs.get("receipt")))
        return assemble(plan, basis, context, **kwargs)

    monkeypatch.setattr(pair_reducer, "entry_bounds", counted)
    monkeypatch.setattr(acquisition, "_matrix_from_observations", kept)
    result = solve(Eigenproblem(A=ingest_pauli(terms, num_qubits=3)),
                   method=ADAPT(initial_state=reference / np.linalg.norm(reference),
                                pool=tuple(ingest_pauli(((p, 1j),), num_qubits=3) for p in ("IIY", "IYZ", "YXI", "XYX")),
                                theta=.37, max_iterations=2, gradient_norm_floor=0.),
                   execution="quantum", seed=5)
    plan, basis, context, receipt = assemblies[-1]
    b = len(basis)
    assert b >= 3 and len(assemblies) >= 2 and len(calls) == b * (b + 1) // 2
    finished = assemble(plan, basis, context, receipt=receipt)
    context.pair_values = {key: entry[:2] for key, entry in context.pair_values.items()}
    calls.clear()
    assert assemble(plan, basis, context)[3] == 0.0 and not calls
    receipts = {r.content_id: r for r in result.data.receipts}
    found = assemble(plan, basis, context, receipt=receipts.get)
    assert len(calls) == b * (b + 1) // 2 and found[3] == finished[3] > 0
    assert np.array_equal(found[0], finished[0]) and np.array_equal(found[1], finished[1])
    assert assemble(plan, basis, context, receipt=receipts.get)[3] == found[3] and len(calls) == b * (b + 1) // 2


def test_checkpoints_write_each_history_record_once_and_a_reopened_run_keeps_the_history(tmp_path, monkeypatch):
    """A checkpoint encodes only the history record that changed, and a reopened Run ends with the same history.

    Classical ADAPT on two qubits with four pool generators makes four
    projected analyses, and each history record holds the gradients of the
    pool members still unselected. After the first checkpoint, every
    checkpoint names at most the newest record, so no earlier record is
    encoded again. A Run stopped right after the checkpoint that adds its
    second record and reopened from disk ends with the history of an
    uninterrupted Run, gradients included, apart from the identities of its
    own observations. Its saved controller re-analyzes to that history.
    """
    import nwqlib

    plan = plan_for(execution="classical",
                    A=ingest_pauli((("ZI", 1.0), ("IZ", .7), ("XX", .4), ("YY", .25), ("XZ", .3)), num_qubits=2),
                    initial_state=(1, 0, 0, 0), max_iterations=4,
                    pool_rows=((("YI", 1j),), (("IY", 1j),), (("XY", 1j),), (("YX", 1j),)))
    calls, real, stop = [], Run.checkpoint, []

    def recorded(self, value, *, changed=None):
        records = sorted((name for name in value if name.startswith("history.")), key=lambda name: int(name[8:]))
        calls.append((records, changed))
        sequence = real(self, value, changed=changed)
        if stop and records == stop:
            raise RuntimeError("stopped after a checkpoint")
        return sequence

    monkeypatch.setattr(Run, "checkpoint", recorded)
    expected = Run(plan).wait(timeout=30, poll_interval=0)
    assert len(expected.history) == 4 and [len(r.gradients) for r in expected.history][:3] == [4, 3, 2]
    path, stop[:] = tmp_path / "run", ["history.000000", "history.000001"]
    with Run(plan, directory=path) as run:
        with pytest.raises(RuntimeError, match="stopped after a checkpoint"):
            run.resume()
    del stop[:]
    with nwqlib.load_run(path) as restored:
        result = restored.wait(timeout=30, poll_interval=0)
    def science(history):
        return [record.model_dump(exclude={"content_id", "contribution_ids"}) for record in history]

    assert science(result.history) == science(expected.history)
    observed = {chunk.content_id for chunk in result.data.observations.chunks}
    assert all(set(record.contribution_ids) <= observed for record in result.history)
    assert result.analyze(overlap_cutoff=plan.method.overlap_cutoff).history == result.history
    assert all(changed is None and not records
               or [name for name in changed if name.startswith("history")] in ([], records[-1:])
               for records, changed in calls)


def _three_qubit_classical_adapt():
    from nwqlib.problems import ingest_occupation

    rows = (("XII", .875), ("IXI", .5), ("IIX", .3125), ("ZZI", .15))
    return plan_for(execution="classical", A=ingest_pauli(rows, num_qubits=3),
                    initial_state=ingest_occupation("000", num_qubits=3), theta=.42, max_iterations=3, seed=42,
                    pool_rows=((("YII", 1j),), (("IYI", 1j),), (("IIY", 1j),)))


def _interrupt_third_analysis(run):
    """Stop ``run`` inside its third projected analysis, after two history records are saved."""
    original = acquisition._pencil_from_observations

    def injected(*args, **kwargs):
        saved = run.checkpoint_state
        if sum(name.startswith("history.") for name in saved) == 2 and saved.get("analysis_in_progress"):
            raise RuntimeError("stopped in the third analysis")
        return original(*args, **kwargs)

    from unittest.mock import patch
    with patch.object(acquisition, "_pencil_from_observations", injected):
        with pytest.raises(RuntimeError, match="stopped in the third analysis"):
            run.resume()


def test_reanalysis_of_an_interrupted_run_writes_no_earlier_history_record(tmp_path, monkeypatch):
    """Retrying an interrupted analysis checkpoints only the fields it changes.

    Classical ADAPT on three qubits with three pool generators, saved to
    disk, stops inside its third projected analysis. Reopened, its
    ``resume(reanalyze=True)`` makes two checkpoints, one naming the retry
    entry and one naming the analysis markers, and writes neither of the
    two saved history records. The Run then ends with the four records of
    an uninterrupted Run, gradients included, and the same 24 observations.
    """
    import nwqlib

    plan = _three_qubit_classical_adapt()
    expected = Run(plan).wait(timeout=30, poll_interval=0)
    path = tmp_path / "run"
    with Run(plan, directory=path) as run:
        _interrupt_third_analysis(run)
        saved = {name: value for name, value in run.checkpoint_state.items() if name.startswith("history.")}
    assert sorted(saved) == ["history.000000", "history.000001"]
    names, rows, checkpoint, write = [], [], Run.checkpoint, Run._write

    def named(self, value, *, changed=None):
        names.append(changed)
        return checkpoint(self, value, changed=changed)

    def written(self, records=(), *args, **kwargs):
        records = tuple(records)
        rows.extend(key for kind, key, _ in records if kind == "checkpoint_field")
        return write(self, records, *args, **kwargs)

    monkeypatch.setattr(Run, "checkpoint", named)
    monkeypatch.setattr(Run, "_write", written)
    with nwqlib.load_run(path) as restored:
        restored.resume(reanalyze=True)
        assert names[:2] == [["analysis_retries"], ["analysis_in_progress", "stop_reason", "pending_labels"]]
        result = restored.wait(timeout=30, poll_interval=0)
    assert not set(saved) & set(rows)

    def science(history):
        return [(r.iteration, r.selected, r.theta, r.energy, r.winner, r.gradients) for r in history]

    assert len(result.history) == 4 and science(result.history) == science(expected.history)
    assert len(result.data.observations.chunks) == len(expected.data.observations.chunks) == 24
