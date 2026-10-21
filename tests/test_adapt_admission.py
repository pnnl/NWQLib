"""Actual ADAPT input and point-domain admission, without compatibility records."""

import json

import pytest
from test_adapt_primary import plan_for
from nwqlib.algorithms.gcim import ADAPT
from nwqlib.operators import ingest_pauli


def _compact_json_size(value):
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False))


def _compact_reconstruction_law(rec):
    """Measure JSON independently of all production admission helpers."""
    saved = json.loads(rec.model_dump_json())
    own_json = _compact_json_size(saved)
    own_json -= sum(_compact_json_size(member) for member in saved["pool"])
    own_json -= sum(_compact_json_size(term) for term in saved["terms"])
    b = tables_json = table_count = 0
    for member, data in zip(rec.pool, saved["pool"], strict=True):
        if member.commutator is not None:
            table_count += 1
            b += member.commutator.labels.array.nbytes
            b += member.commutator.coefficients.array.nbytes
            tables_json += _compact_json_size({"commutator": data["commutator"]})
    records = len(rec.terms) + sum(len(member.terms) for member in rec.pool)
    other = records * (6 * rec.num_qubits + 360) + 3 * sum(
        2048 + 128 * len(member.fermion_terms)
        + 16 * sum(len(ops) for _, ops in member.fermion_terms)
        + 16 * (len(member.spatial_indices) + len(member.spin_orbital_indices or ()))
        for member in rec.pool
    )
    indices = sum(map(len, rec.groups)) + sum(map(len, rec.energy_groups))
    # own_json already includes both complete readout group lists.
    # Keep the table call's existing two-character empty index allowance.
    j = tables_json + own_json + (2 if table_count else 0)
    return other + 2 * b + 4 * j + 16 * indices


def _reconstruction_charges(monkeypatch):
    import nwqlib.algorithms.gcim.adapt_inputs as owner

    checked = []
    original = owner._check_bytes

    def observe(size, limit, what, *args, **kwargs):
        if what.startswith("ADAPT reconstruction records"):
            checked.append(size)
        return original(size, limit, what, *args, **kwargs)

    monkeypatch.setattr(owner, "_check_bytes", observe)
    return checked


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


def test_reconstruction_pauli_records_and_commutator_arrays_are_charged_their_json(monkeypatch):
    from qiskit.quantum_info import SparsePauliOp

    rows = (("IIIZ", 0.5), ("IIZZ", 0.25), ("IZIZ", 0.125),
            ("ZIIZ", 0.75), ("IIZI", 0.375))
    settings = dict(
        A=ingest_pauli(rows, num_qubits=4),
        pool=(ingest_pauli((("IIIY", 1j),), num_qubits=4),),
        initial_state=(1,) + (0,) * 15,
    )
    checked = _reconstruction_charges(monkeypatch)
    rec = plan_for(**settings).reconstruction
    h = SparsePauliOp.from_list(rows)
    a = SparsePauliOp.from_list((("IIIY", 1j),))
    expected = tuple(sorted((label, c.real) for label, c in (h @ a - a @ h).simplify().to_list()))
    assert rec.pool[0].commutator.rows() == expected
    assert len(rec.terms) + sum(len(m.terms) for m in rec.pool) == 6
    table = rec.pool[0].commutator
    assert table.labels.array.nbytes + table.coefficients.array.nbytes == 160
    needed = max(checked)
    assert needed >= _compact_reconstruction_law(rec)
    plan_for(max_bytes=needed, **settings)
    with pytest.raises(ValueError, match="ADAPT reconstruction records and decoded label cache needs"):
        plan_for(max_bytes=needed - 1, **settings)


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


def test_label_cache_charge_covers_the_decoded_cache_it_reserves():
    """The per-Plan decoded label cache fits its allowance on ten synthetic pools.

    Each pool has K members with pairwise distinct commutator labels,
    r rows per member on q qubits, and S selections. The label encoding
    requires K*r <= 4**q, so the distinct-label population is R = K*r.
    The cache decodes the rows, sorts the screen and energy labels, and
    fills active-label entries through their clearing boundary: limit K
    for sampled calls and S for exact calls. Its traced incremental peak
    must not exceed the default label_cache_bytes allowance.
    """
    import itertools
    import tracemalloc
    from types import SimpleNamespace

    from nwqlib.algorithms.gcim.adapt_inputs import label_cache_bytes
    from nwqlib.algorithms.gcim.adapt_records import (
        active_screen_labels, commutator_rows, energy_labels, screen_labels)
    from nwqlib.algorithms.gcim.fixed_basis import PauliArrays

    for q, K, r, S in ((2, 3, 0, 2), (2, 3, 1, 2), (8, 12, 19, 4), (10, 12, 257, 4), (20, 5, 1024, 3)):
        members = [[("".join("IXYZ"[(i >> (2 * j)) & 3] for j in reversed(range(q))), float(i + 1))
                    for i in range(k * r, (k + 1) * r)] for k in range(K)]
        pool = tuple(SimpleNamespace(commutator=PauliArrays.from_rows(rows, q)) for rows in members)
        energies = tuple(label for label, _ in members[0][:min(r, 5)])
        for sampled in (False, True):
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            cache = {}
            commutator_rows(pool, cache)
            labels = screen_labels(pool, cache)
            energy_labels(energies, cache)
            limit = K if sampled else S
            keys = itertools.chain.from_iterable(itertools.combinations(range(K), k) for k in range(S + 1))
            for key in itertools.islice(keys, limit + 2):
                active_screen_labels(pool, cache, key, limit)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            assert len(labels) == K * r
            charge = label_cache_bytes(q, (r,) * K, K, len(energies), S, sampled=sampled)
            assert charge >= peak - baseline
