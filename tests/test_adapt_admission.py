"""Actual ADAPT input and point-domain admission, without compatibility records."""

import json

import pytest
from test_adapt_primary import plan_for
from nwqlib.algorithms.gcim import ADAPT
from nwqlib.operators import ingest_pauli
from nwqlib.subroutines.fermionic_pool import FermionicGenerator, _snapshot_generator


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


def test_snapshot_keeps_only_immutable_named_generator_inputs():
    rows = [["Y", 1j]]
    generator = FermionicGenerator(
        family="custom",
        spatial_indices=[],
        pool_index=0,
        num_qubits=1,
        fermion_terms=[],
        pauli_terms=rows,
    )
    method = ADAPT(initial_state=[1, 1], pool=(generator,))
    identity = method.content_id
    rows[0][0] = "X"
    assert method.pool[0].pauli_terms == (("Y", 1j),) and method.content_id == identity


def test_invalid_pool_and_original_dimension_are_not_late_native_failures():
    with pytest.raises(ValueError, match="original Eigenproblem dimension"):
        plan_for(pool=(ingest_pauli((("XX", 1j),), num_qubits=2),))
    with pytest.raises(ValueError, match="anti-Hermitian"):
        plan_for(pool=(ingest_pauli((("X", 1.0),), num_qubits=1),))
    with pytest.raises(ValueError, match="max_pool_size"):
        plan_for(max_pool_size=1)


def test_original_hermiticity_and_explicit_pool_correction_have_distinct_owners():
    from nwqlib.problems.records import Eigenproblem

    with pytest.raises(ValueError, match="exactly Hermitian"):
        Eigenproblem(A=[[1, 1e-15], [0, -1]])
    near = ingest_pauli((("Y", 1j + 1e-14),), num_qubits=1)
    plan = plan_for(pool=(near,), input_symmetry_tolerance=1e-12)
    assert plan.reconstruction.pool[0].symmetry.correction_applied
    with pytest.raises(ValueError, match="anti-Hermitian"):
        plan_for(pool=(near,), input_symmetry_tolerance=0)


def test_known_generator_storage_limit_precedes_snapshot():
    generator = FermionicGenerator(
        family="custom",
        spatial_indices=(),
        pool_index=0,
        num_qubits=1,
        fermion_terms=(),
        pauli_terms=(("Y", 1j),),
    )
    with pytest.raises(ValueError, match="bytes"):
        _snapshot_generator(generator, max_bytes=1)


@pytest.mark.parametrize(
    "pool, enumerator, candidates",
    (
        # Two spatial orbitals give P = 3 pairs and 3 + 3 * 4 = 15 candidates.
        ("spin_adapted_sd", "enumerate_spin_adapted_gsd_pool", 15),
        # With one up and one down electron there are two singles and one
        # opposite-spin double. The CEO pool forms the plus and the minus
        # combination of its direct and crossed excitations.
        ("uccsd_sd", "enumerate_uccsd_sd_pool", 3),
        ("qeb_sd", "enumerate_qeb_sd_pool", 3),
        ("ceo_ovp", "enumerate_ceo_ovp_pool", 4),
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


def test_classical_matrix_route_charges_the_pool_records(monkeypatch):
    import numpy as np
    from nwqlib.operators import operator_input

    # The dense Hamiltonian has no Pauli terms. Its one-term pool member
    # and the enclosing metadata both enter the record charge.
    settings = dict(A=operator_input(np.diag(np.arange(16.0))), execution="classical",
                    pool=(ingest_pauli((("IIIY", 1j),), num_qubits=4),), initial_state=(1,) + (0,) * 15)
    checked = _reconstruction_charges(monkeypatch)
    reconstruction = plan_for(**settings).reconstruction
    assert reconstruction.terms == () and [len(m.terms) for m in reconstruction.pool] == [1]
    law = max(checked)
    assert law >= _compact_reconstruction_law(reconstruction)
    # The packed action tables reserve 64 bytes per row beside the records:
    # the dense Hamiltonian has no H0 table, and the member has one
    # generator row and one insertion row (adapt_actions.build_action_tables).
    tables = law + 64 * 2
    plan_for(max_bytes=tables, **settings)
    with pytest.raises(ValueError, match="ADAPT packed action tables needs"):
        plan_for(max_bytes=tables - 1, **settings)
    with pytest.raises(ValueError, match="ADAPT reconstruction records needs"):
        plan_for(max_bytes=law - 1, **settings)


def test_pool_member_orbital_indices_are_charged_with_the_member(monkeypatch):
    # A supplied generator may carry any number of orbital indices, and each
    # enters the member record, so each is charged 16 bytes per copy. Each
    # generator snapshot is checked on its own, so with ten members the
    # running total binds first.
    pool = tuple(FermionicGenerator(family="single", spatial_indices=(0, 1) * 200, pool_index=index, num_qubits=4,
                                    fermion_terms=(), pauli_terms=(("IIXY", 0.5j),)) for index in range(10))
    settings = dict(A=ingest_pauli((("IIIZ", 0.5), ("IIZZ", 0.25)), num_qubits=4), pool=pool,
                    initial_state=(1,) + (0,) * 15, execution="classical")
    checked = _reconstruction_charges(monkeypatch)
    reconstruction = plan_for(**settings).reconstruction
    # A classical Plan forms no commutator tables.
    assert all(m.commutator is None for m in reconstruction.pool)
    law = max(checked)
    assert law >= _compact_reconstruction_law(reconstruction)
    # Packed tables: two H0 rows, and one generator row and one insertion
    # row per member, 64 bytes each beside the records.
    tables = law + 64 * (2 + 2 * len(pool))
    plan_for(max_bytes=tables, **settings)
    with pytest.raises(ValueError, match="ADAPT packed action tables needs"):
        plan_for(max_bytes=tables - 1, **settings)
    with pytest.raises(ValueError, match="ADAPT reconstruction records needs"):
        plan_for(max_bytes=law - 1, **settings)


@pytest.mark.parametrize("execution", ("quantum", "classical"))
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


def test_only_sampled_quantum_plans_admit_readout_grouping():
    # ADAPT passes max_products as the max_comparisons running budget of
    # first-fit QWC grouping, so a plan that groups needs the actual
    # comparison count of its labels, not the 200 * 199 / 2 worst case. Only
    # a sampled quantum Plan reads QWC groups and only a quantum Plan reads
    # [H, A], so one product below the count of the 200 Hamiltonian labels, a
    # classical or exact Plan is admitted without forming either, while the
    # sampled Plan is refused by the grouping budget and forms its groups at
    # the count.
    from itertools import islice, product

    from nwqlib.operators import pauli_table

    # The generator acts on qubit 7, where only the last label is not I, so
    # [H, A] has one surviving pair and its construction costs far less than
    # the grouping comparisons.
    labels = tuple(islice(("".join(p) for p in product("IXYZ", repeat=8) if set(p) != {"I"}), 199))
    labels += ("XIIIIIII",)
    settings = dict(
        A=ingest_pauli(tuple((label, 1 / (i + 1)) for i, label in enumerate(labels)), num_qubits=8),
        pool=(ingest_pauli((("YIIIIIII", 1j),), num_qubits=8),),
        initial_state=(1,) + (0,) * 255,
        max_iterations=1,
        max_bytes=10**10,
    )
    needed = pauli_table(((p, 1.0) for p in sorted(labels)), num_qubits=8).group(
        strategy="qwc", max_bytes=settings["max_bytes"], max_comparisons=200 * 199 // 2
    ).comparison_count
    assert needed < 200 * 199 // 2
    for execution in ("classical", "quantum"):
        reconstruction = plan_for(execution=execution, max_products=needed - 1, **settings).reconstruction
        assert reconstruction.groups == reconstruction.energy_groups == ()
        assert (reconstruction.pool[0].commutator is None) == (execution == "classical")
    refusal = rf"Pauli grouping exceeds ADAPT\.max_products={needed - 1}:"
    with pytest.raises(ValueError, match=refusal):
        plan_for(shots=64, max_products=needed - 1, **settings)
    sampled = plan_for(shots=64, max_products=needed, **settings).reconstruction
    assert sampled.energy_groups and sampled.groups


def test_label_cache_charge_covers_the_decoded_cache_it_reserves():
    """The per-Plan decoded label cache stays within ``label_cache_bytes`` on ten synthetic pools.

    Each pool has K members sharing one commutator table of r rows on q
    qubits and S selections. The cache decodes the rows, sorts the screen and
    energy labels and fills active-label entries up to its clearing boundary
    (the sampled caller's limit K, the exact one S). Its traced incremental
    peak must not exceed the charge, which assumes every row label distinct.
    """
    import itertools
    import tracemalloc
    from types import SimpleNamespace

    from nwqlib.algorithms.gcim.adapt_inputs import label_cache_bytes
    from nwqlib.algorithms.gcim.adapt_records import (
        active_screen_labels, commutator_rows, energy_labels, screen_labels)
    from nwqlib.algorithms.gcim.fixed_basis import PauliArrays

    for q, K, r, S in ((2, 3, 0, 2), (2, 3, 1, 2), (8, 12, 19, 4), (10, 12, 257, 4), (20, 5, 1024, 3)):
        rows = [("".join("IXYZ"[(i >> (2 * j)) & 3] for j in reversed(range(q))), float(i + 1)) for i in range(r)]
        pool = tuple(SimpleNamespace(commutator=PauliArrays.from_rows(rows, q)) for _ in range(K))
        energies = tuple(label for label, _ in rows[:min(r, 5)])
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
            charge = label_cache_bytes(q, (r,) * K, K, len(energies), S, sampled=sampled, distinct_cap=len(labels))
            assert charge >= peak - baseline
