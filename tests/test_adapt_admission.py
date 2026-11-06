"""Actual ADAPT input and point-domain admission, without compatibility records."""

import pytest
from test_adapt_primary import plan_for
from nwqlib.algorithms.gcim import ADAPT
from nwqlib.operators import ingest_pauli


def test_original_hermiticity_and_explicit_pool_correction_have_distinct_owners():
    from nwqlib.problems.records import Eigenproblem

    with pytest.raises(ValueError, match="exactly Hermitian"):
        Eigenproblem(A=[[1, 1e-15], [0, -1]])
    near = ingest_pauli((("Y", 1j + 1e-14),), num_qubits=1)
    plan = plan_for(pool=(near,), input_symmetry_tolerance=1e-12)
    assert plan.reconstruction.pool[0].symmetry.correction_applied
    with pytest.raises(ValueError, match="anti-Hermitian"):
        plan_for(pool=(near,), input_symmetry_tolerance=0)


@pytest.mark.parametrize(
    "pool, enumerator, candidates",
    (
        # Two spatial orbitals give P = 3 pairs and 3 + 3 * 4 = 15 candidates.
        ("spin_adapted_sd", "enumerate_spin_adapted_gsd_pool", 15),
    ),
)
def test_named_pool_enumeration_is_admitted_before_it_starts(
    monkeypatch, pool, enumerator, candidates
):
    import nwqlib.algorithms.gcim.adapt_inputs as owner
    from nwqlib.core.planning import RandomStreams
    from nwqlib.problems.inputs import ingest_occupation
    from nwqlib.problems.records import Eigenproblem

    class Enumerated(Exception):
        pass

    def enumerate_pool(argument):
        raise Enumerated

    monkeypatch.setattr(owner, enumerator, enumerate_pool)
    problem = Eigenproblem(A=ingest_pauli((("ZIII", 1.0),), num_qubits=4))
    _, work = owner._pool_enumeration_requirements(candidates, 4)

    def plan(max_products):
        method = ADAPT(
            initial_state=ingest_occupation((1, 1, 0, 0), num_qubits=4),
            pool=pool,
            n_spatial_orbitals=2,
            max_products=max_products,
        )
        return method.plan(
            problem, output=problem.default_output(), execution="classical", shots=None,
            rng=RandomStreams(7),
        )

    with pytest.raises(ValueError, match=f"{work} scalar products"):
        plan(work - 1)
    with pytest.raises(Enumerated):
        plan(work)


def test_commutator_table_holds_the_exact_rows_and_its_construction_is_checked_where_it_is_built(monkeypatch):
    """The [H, A] table holds the exact commutator rows, and building it is the byte check that binds.

    The commutator construction counts its pair tiles, surviving products,
    sort and conversion against ``max_bytes`` before each allocation. Half
    its largest count refuses planning at that check, and the largest count
    passes it.
    """
    import nwqlib.algorithms.gcim.adapt_inputs as owner
    from qiskit.quantum_info import SparsePauliOp

    rows = (("IIIZ", 0.5), ("IIZZ", 0.25), ("IZIZ", 0.125),
            ("ZIIZ", 0.75), ("IIZI", 0.375))
    settings = dict(
        A=ingest_pauli(rows, num_qubits=4),
        pool=(ingest_pauli((("IIIY", 1j),), num_qubits=4),),
        initial_state=(1,) + (0,) * 15,
    )
    checked = []
    original = owner._check_bytes

    def observe(size, limit, what, *args, **kwargs):
        if what == "ADAPT commutators":
            checked.append(size)
        return original(size, limit, what, *args, **kwargs)

    monkeypatch.setattr(owner, "_check_bytes", observe)
    rec = plan_for(**settings).reconstruction
    h = SparsePauliOp.from_list(rows)
    a = SparsePauliOp.from_list((("IIIY", 1j),))
    expected = tuple(sorted((label, c.real) for label, c in (h @ a - a @ h).simplify().to_list()))
    assert rec.pool[0].commutator.rows() == expected
    table = rec.pool[0].commutator
    assert table.labels.array.nbytes + table.coefficients.array.nbytes == 160
    needed = max(checked)
    with pytest.raises(ValueError, match="ADAPT commutators needs about"):
        plan_for(max_bytes=needed // 2, **settings)
    try:
        plan_for(max_bytes=needed, **settings)
    except ValueError as error:
        assert "ADAPT commutators" not in str(error)


@pytest.mark.parametrize("execution", ("quantum",))
def test_largest_projected_solve_work_is_admitted_at_planning(execution):
    # Two pool members and max_iterations=2 reach a basis of b = 4 states.
    # Its projected solve needs 32*b**2 + 16*b**3 = 1536 products, which
    # every later analysis checks against max_products. Planning refuses one
    # product less and admits the exact amount.
    b = 4
    work = 32 * b * b + 16 * b ** 3
    with pytest.raises(ValueError, match=f"action needs {work} scalar products"):
        plan_for(execution=execution, max_products=work - 1)
    assert plan_for(execution=execution, max_products=work).reconstruction.max_selections == 2


def test_explicit_pool_has_no_member_count_limit():
    # A pool of 1100 explicit one-qubit generators plans, since only the allocation checks of the members, their
    # tables and the selection bound the pool.
    pool = tuple(ingest_pauli((("Y", 1j * (1 + k / 1100)),), num_qubits=1) for k in range(1100))
    selected = plan_for(execution="classical", pool=pool, max_iterations=1)
    assert len(selected.reconstruction.pool) == 1100


def test_commutator_index_matches_the_decoded_rows_and_keeps_one_screening_round():
    """The label index gives each member's decoded rows, the sorted distinct labels and bit-identical gradients.

    Synthetic pools with K members of r rows on q qubits, with labels
    shared between members, cover the packed path up to 32 qubits and the
    code-row path at 33. Indexing the distinct labels with a member's
    indices returns the labels of its decoded rows in row order, the
    labels are the sorted distinct ones, and the coefficient-weighted sum
    of random measured values equals the row-by-row fsum. The active labels
    of a screening round are the union over the members not removed, and
    only the current round's union stays in the cache.
    """
    from math import fsum
    from types import SimpleNamespace
    import numpy as np

    from nwqlib.algorithms.gcim.adapt_records import active_screen_labels, commutator_index, screen_labels
    from nwqlib.algorithms.gcim.fixed_basis import PauliArrays

    rng = np.random.default_rng(5)
    for q, K, r in ((1, 3, 2), (6, 5, 9), (32, 4, 7), (33, 4, 7)):
        population = sorted({"".join(rng.choice(list("IXYZ"), q)) for _ in range(3 * r)})
        members = [sorted({label: float(rng.normal()) for label in rng.choice(population, r)}.items())
                   for _ in range(K)]
        pool = tuple(SimpleNamespace(commutator=PauliArrays.from_rows(rows, q)) for rows in members) + (
            SimpleNamespace(commutator=None),)
        cache = {}
        labels, indices = commutator_index(pool, cache)
        assert labels == screen_labels(pool, cache) == tuple(sorted({label for rows in members for label, _ in rows}))
        assert len(indices[-1]) == 0
        measured = {label: float(rng.normal()) for label in labels}
        values = np.array([measured[label] for label in labels])
        for member, rows, index in zip(pool, members, indices):
            assert tuple(labels[k] for k in index) == tuple(label for label, _ in rows) == tuple(
                label for label, _ in member.commutator.rows())
            assert fsum(member.commutator.coefficients.array * values[index]) == fsum(
                c * measured[label] for label, c in member.commutator.rows())
        for removed in ({0}, {0, 1}, set()):
            active, members_set = active_screen_labels(pool, cache, removed)
            union = sorted({label for i, rows in enumerate(members) if i not in removed for label, _ in rows})
            assert active == tuple(union) and members_set == frozenset(union)
            assert cache["active_screen_labels"][0] == frozenset(removed)


def test_classical_frontier_is_checked_once_at_planning_against_the_physical_memory_by_default(monkeypatch):
    """Classical ADAPT checks its largest query's dominant arrays once at planning, and no query checks again.

    For H = Z_9 + 0.5 X...X on ten qubits from |0...0> with pool i Y_9 and
    i Y_0, the default ``max_bytes`` is None, the physical memory. The Plan's check holds
    ``classical_frontier_bytes`` plus the reference preparation's bytes,
    the largest query's arrays. An explicit ``max_bytes`` of half that
    refuses the Plan before any query, and the reported amount passes this
    check. The run of the unrefused Plan
    makes no further frontier check.
    """
    from nwqlib import solve
    from nwqlib.algorithms.gcim import adapt
    from nwqlib.operators import access

    assert ADAPT(initial_state=(1, 1), pool="spin_adapted_sd", n_spatial_orbitals=1).max_bytes is None
    checks, real = [], access._check_bytes

    def counted(size, max_bytes, operation="input"):
        if operation == "ADAPT classical frontier":
            checks.append((size, max_bytes))
        return real(size, max_bytes, operation)

    for module in (adapt, access):
        monkeypatch.setattr(module, "_check_bytes", counted)
    from nwqlib.problems import ingest_occupation

    setting = dict(execution="classical", A=ingest_pauli((("Z" + "I" * 9, 1.0), ("X" * 10, .5)), num_qubits=10),
                   initial_state=ingest_occupation((0,) * 10, num_qubits=10),
                   pool_rows=((("Y" + "I" * 9, 1j),), (("I" * 9 + "Y", 1j),)))
    plan = plan_for(**setting)
    ((need, limit),) = checks
    assert limit is None and need == adapt.classical_frontier_bytes(
        plan.method, plan._native["inputs"], plan.reconstruction.max_selections, 10
    ) + plan.reconstruction.reference_action_bytes
    with pytest.raises(ValueError, match=f"ADAPT classical frontier needs about {need} bytes"):
        plan_for(**setting, max_bytes=need // 2)
    try:
        plan_for(**setting, max_bytes=need)
    except ValueError as error:
        assert "ADAPT classical frontier" not in str(error)
    del checks[:]
    solve(plan, progress=False)
    assert not checks


def test_every_classical_plan_charges_the_energy_query_of_its_longest_chain():
    """One frontier holds with or without optimization, and a direct energy point of L generators fits it.

    Any classical Plan binds an energy point through ``prepare_experiment``,
    so ``classical_frontier_bytes`` is 16(C + 12)d + 24L for both values of
    ``optimize_every_m`` and is the host workspace the Plan declares. On the
    Plan without optimization, for H = Z_9 + 0.5 X...X on ten qubits with
    pool i Y_9 and i Y_0, the energy and gradient point of L =
    ``max_selections`` generators runs, and the arrays its query allocates
    stay within that frontier. Its preparation, traced apart, allocates
    about 635 kB of records at 8, 10 and 12 qubits alike and no state
    vector, so it is outside the frontier's array count.
    """
    import tracemalloc
    from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
    from nwqlib.algorithms.gcim import adapt
    from nwqlib.algorithms.gcim.adapt_acquisition import _point
    from nwqlib.problems import ingest_occupation

    setting = dict(execution="classical", A=ingest_pauli((("Z" + "I" * 9, 1.0), ("X" * 10, .5)), num_qubits=10),
                   initial_state=ingest_occupation((0,) * 10, num_qubits=10),
                   pool_rows=((("Y" + "I" * 9, 1j),), (("I" * 9 + "Y", 1j),)))
    plans = (plan_for(**setting),
             plan_for(**setting, optimize_every_m=1, optimize_rounds_n=2, optimize_max_evaluations=3))
    length = plans[0].reconstruction.max_selections
    frontiers = [adapt.classical_frontier_bytes(p.method, p._native["inputs"], length, 10) for p in plans]
    assert frontiers[0] == frontiers[1] == plans[0].construction.kernels[0].workspace[0].bytes \
        == plans[1].construction.kernels[0].workspace[0].bytes
    right = tuple((i % 2, 0.1 * (i + 1)) for i in range(length))

    def energy(traced):
        with Run(plans[0]) as run:
            handle = prepare_experiment(_point(plans[0], "energy", right=right), run=run)
            if traced:
                tracemalloc.start()
            try:
                values = submit_experiment(handle, run=run).values
                return {v.label: v.value for v in values}, tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

    # A first point leaves the one-time allocations of the process outside the traced peak.
    energy(False)
    scalars, peak = energy(True)
    assert {"energy", *(f"gradient_{i}" for i in range(length))} <= set(scalars)
    assert peak <= frontiers[0]
