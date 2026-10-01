"""ADAPT configuration, exact query association and original scientific conventions."""

from math import pi, sqrt
from types import SimpleNamespace
import numpy as np
import pytest
from test_adapt_primary import plan_for
from nwqlib.algorithms.gcim import ADAPT, adapt_acquisition
from nwqlib._numerics import stable_vector_norm
from nwqlib.execution import PauliValue
from nwqlib.operators import ingest_pauli


def test_default_controls_and_finite_basis_order(monkeypatch):
    method = ADAPT(initial_state=[1, 0], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),))
    assert method.theta == pi / 4 and method.gradient_norm_floor == 1e-8
    assert (
        method.energy_change_tolerance == 1e-6
        and method.t_auto_fraction == 0.2
        and method.t_user == 10
    )
    assert method.max_iterations == 8 and method.overlap_cutoff == 1e-12
    assert (
        method.optimize_every_m
        is method.optimize_rounds_n
        is method.optimize_max_evaluations
        is None
    )
    plan = plan_for(max_iterations=100)
    assert plan.reconstruction.max_selections == 2
    selected = ((0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4))
    assert adapt_acquisition.basis_chains(selected) == (
        (),
        *((x,) for x in selected),
        selected,
        selected[:2],
        selected[:3],
    )
    point = adapt_acquisition._point(plan, "screen", right=((0, 0.0),))
    assert adapt_acquisition.chain({b.parameter: b.value for b in point.bindings}, "r") == (
        (0, 0.0),
    )
    with pytest.raises(ValueError, match="twice"):
        adapt_acquisition._point(plan, "pair", right=((0, 0.1), (0, 0.2)))
    # Initial, two individual selections and their product give four columns,
    # even when the two pool generators share the same nonidentity Pauli word.
    bounded = plan_for(max_basis_size=4)
    assert adapt_acquisition.basis_chains(((0, 0.1), (1, 0.2))) == (
        (), ((0, 0.1),), ((1, 0.2),), ((0, 0.1), (1, 0.2)),
    )
    assert bounded.reconstruction.max_selections == 2
    from nwqlib.algorithms.gcim import adapt
    monkeypatch.setattr(adapt, "eigen_operator", lambda *a, **k: pytest.fail("basis cap reached operator processing"))
    with pytest.raises(ValueError, match=r"basis holds 4 states .* above ADAPT\(max_basis_size=3\)"):
        plan_for(max_basis_size=3)


@pytest.mark.parametrize(
    "changes",
    (
        {"theta": 0.0},
        {"theta": float("inf")},
        {"gradient_norm_floor": -1.0},
        {"energy_change_tolerance": 0.0},
        {"t_user": False},
        {"max_iterations": 0},
        {"overlap_cutoff": 0.0},
        {"optimize_every_m": 1},
        {"optimize_rounds_n": 1},
        {"optimize_max_evaluations": 1},
        {"optimize_every_m": 1, "optimize_rounds_n": 1, "optimize_max_evaluations": 0},
    ),
)
def test_invalid_controls_fail_at_configuration(changes):
    with pytest.raises(ValueError):
        plan_for(**changes)


def test_processed_commutator_sign_at_subnormal_squared_scale():
    tiny = 2.0**-600
    for shift in (0.0, 1e100):
        plan = plan_for(
            A=ingest_pauli((("Z", tiny), ("I", shift)), num_qubits=1),
            pool_rows=((("Y", 1j),),),
            max_iterations=1,
            pauli_coefficient_cutoff=0.0,
        )
        assert plan.reconstruction.pool[0].commutator.rows() == (
            ("X", 2 * tiny),
        )
    gradient = np.array([2 * tiny, 2 * tiny])
    assert gradient @ gradient == 0.0
    assert stable_vector_norm(gradient) == pytest.approx(2 * tiny * sqrt(2), rel=2e-15, abs=0.0)


def test_query_completeness_and_exact_point_identity():
    plan = plan_for()
    point = adapt_acquisition._point(plan, "screen")
    chunk = SimpleNamespace(
        realization_id=point.content_id,
        bindings=point.bindings,
        population="unconditional",
        values=(),
    )
    with pytest.raises(adapt_acquisition._PartialData, match="missing X"):
        adapt_acquisition._read_values(plan, point, chunk)
    chunk.values = (PauliValue(label="X", value=0.5),)
    assert adapt_acquisition._read_values(plan, point, chunk) == {"X": 0.5}
    changed = adapt_acquisition._point(plan, "screen", right=((0, pi / 8),))
    with pytest.raises(ValueError, match="exact query"):
        adapt_acquisition._read_values(plan, changed, chunk)


@pytest.mark.parametrize("representation", ["dense", "csr"])
def test_classical_matrix_identity_offset_moves_the_ritz_value_by_its_coefficient(representation):
    """E(A0 + cI) - c equals E(A0) to rounding for a dense or sparse matrix Hamiltonian."""
    from scipy import sparse
    from nwqlib import Eigenproblem, solve

    pauli = {"I": np.eye(2), "X": np.array([[0., 1.], [1., 0.]]),
             "Y": np.array([[0., -1j], [1j, 0.]]), "Z": np.diag([1., -1.])}
    terms = (("ZI", .5), ("XX", .25), ("IZ", .375), ("YY", .125))
    a0 = sum(c * np.kron(pauli[label[0]], pauli[label[1]]) for label, c in terms)
    pool = tuple(ingest_pauli(((p, 1j),), num_qubits=2) for p in ("YI", "IY", "XY", "YX"))
    # A fixed angle of 1e-3 makes the basis states nearly parallel, cond(S) about 2e6.
    method = ADAPT(initial_state=[1, 0, 0, 0], pool=pool, theta=1e-3, max_iterations=3)
    values, selections = {}, set()
    for c in (.25, -1e4):
        matrix = a0 + c * np.eye(4)
        result = solve(Eigenproblem(A=matrix if representation == "dense" else sparse.csr_matrix(matrix)),
                       method=method, execution="classical", seed=7)
        assert result.plan.reconstruction.terms == ()
        assert result.plan.reconstruction.host_identity_shift == c  # A0 is traceless.
        assert result.pencil.overlap_condition_number > 1e6
        values[c], selections = result.eigenvalue, selections | {result.selected}
    assert len(selections) == 1
    # Both runs solve the same offset-free pencil, and only adding c back
    # rounds. Solving (A0 + cI, S) directly would err by about u|c|cond(S),
    # 1e-6 here.
    assert abs((values[-1e4] + 1e4) - (values[.25] - .25)) <= (
        2 * 2.**-53 * (abs(values[-1e4]) + abs(values[.25])))


@pytest.mark.parametrize("pool", (("Y", "X"), ("Y",)))
def test_shared_full_chain_query_serves_the_diagonal_and_the_screen(tmp_path, monkeypatch, pool):
    """One exact query of each full chain supplies its pencil diagonal and its screening values.

    H = Z, reference |+>, theta = pi/8, one iteration. Each full chain (the
    empty chain, then the selected one) is acquired once, during its matrix
    stage, with a weighted diagonal-reduction point and, when active
    screening labels remain, a screen point at the same boundary. The later
    screen reuses that observation, and the round's ``basis_realization``
    names it. No diagonal pair query is acquired. With a one-member pool the
    terminal chain has no active labels, so its query carries the reduction
    point alone and no empty query is submitted. A saved Result reanalyzes
    without acquisition.
    """
    from nwqlib import Eigenproblem, load_result, solve

    method = ADAPT(initial_state=[1, 1], pool=tuple(ingest_pauli(((p, 1j),), num_qubits=1) for p in pool),
                   theta=pi / 8, max_iterations=1)
    result = solve(Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1)), method=method, execution="quantum")
    chunks = result.data.observations.chunks
    chains = ((), tuple(zip(result.selected, result.theta)))
    shared = [adapt_acquisition._point(result.plan, "screen", right=chain) for chain in chains]
    points = {point.content_id: sorted(c.point for c in chunks if c.realization_id == point.content_id)
              for point in shared}
    assert points[shared[0].content_id] == ["diagonal", "screen"]
    assert points[shared[1].content_id] == (["diagonal", "screen"] if len(pool) > 1 else ["diagonal"])
    assert len({c.attempt for c in chunks if c.experiment == "screen"}) == 2
    pairs = [c for c in chunks if c.experiment == "pair"]
    assert len({c.attempt for c in pairs}) == 1 and all(c.point == "pair" for c in pairs)
    assert result.history[-1].basis_realization == shared[1].content_id
    loaded = load_result(result.save(tmp_path / "result"))
    from nwqlib.backends import qiskit_aer

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", lambda *a, **k: pytest.fail("reanalysis acquired"))
    assert loaded.analyze(overlap_cutoff=1e-10).eigenvalue == pytest.approx(result.eigenvalue, abs=1e-14, rel=0)
