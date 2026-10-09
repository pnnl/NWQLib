"""Concrete built-in value conformance, not certification of extension code."""

import pytest
from pydantic import model_validator

import nwqlib
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.core.planning import Plan
from nwqlib.core import Record
from nwqlib.core.records import FrozenArray, Rational
from nwqlib.evidence import Evidence
from nwqlib.operators import ingest_pauli
from nwqlib.problems import Expectation
from nwqlib.resources import ResourceLaw, Workspace


def assert_conforming_value(value):
    """Compare actual populated fields/types to a detached roundtrip at this scope.

    This test helper never scans Record subclasses or interprets annotations.
    Authors must supply representative values and scientific counterexamples.
    """
    def compare(original, restored, serialized):
        assert type(original) is type(restored)
        if isinstance(original, Record):
            assert type(original).model_config["frozen"]
            assert type(original).model_config["extra"] == "forbid"
            assert set(serialized) == set(type(original).model_fields)
            for name in type(original).model_fields:
                compare(getattr(original, name), getattr(restored, name), serialized[name])
        elif isinstance(original, tuple):
            assert len(original) == len(restored) == len(serialized)
            for before, after, exported in zip(original, restored, serialized, strict=True):
                compare(before, after, exported)
        elif isinstance(original, FrozenArray):
            assert type(restored) is FrozenArray and original == restored
            assert serialized == original.to_json() and FrozenArray.from_json(serialized) == original
        else:
            assert original is None or type(original) in (bool, int, float, str)
            assert original == restored == serialized
    restored = type(value).model_validate_json(value.model_dump_json())
    compare(value, restored, value.model_dump(mode="json", exclude_computed_fields=True))
    assert value.content_id == restored.content_id


def test_populated_concrete_plan_conformance_and_revision_identity():
    # A populated built-in Plan: the physical vector (2,2) and observable I+Z.
    problem = Expectation(
        state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)
    )
    base = nwqlib.plan(problem, method=ExpectationMethod(), seed=7)
    source = base.method.descriptor.source
    law = ResourceLaw(metric="operations", basis="selected_logical", value=7, interpretation="exact",
                      evidence=Evidence(kind="user_assertion", source=source))
    workspace = Workspace(location="host", purpose="workspace", bytes=16, source=source)
    selection = base.construction.selections[0].revise(resource_laws=(law,), workspace=(workspace,))
    plan = base.revise(construction=base.construction.revise(selections=(selection,)))
    # Live Problem/Method inputs are described by the Plan's portable record;
    # loading them requires the archive owner, not a generic Plan JSON cast.
    for value in (plan.construction, plan.method, plan.construction.model_copy()):
        assert_conforming_value(value)
    assert type(plan.method) is ExpectationMethod
    assert plan.construction.selections[0].resource_laws[0].value == 7
    assert plan.construction.selections[0].workspace[0].bytes == 16
    changed = plan.revise(construction=plan.construction.revise(selections=(
        selection.revise(workspace=(workspace.revise(bytes=32),)),)))
    assert changed.content_id != plan.content_id and changed.parent_id == plan.content_id
    assert_conforming_value(changed.construction)
    with pytest.raises(ValueError):
        law.revise(value=-1)
    with pytest.raises(ValueError):
        workspace.revise(bytes=-1)
    for value in (plan.method, plan.construction):
        owner = type(value)
        fields = value.model_dump(mode="json")
        fields["schema_version"] = 0
        with pytest.raises(ValueError):
            owner.model_validate(fields)


def test_final_identity_after_ordinary_normalization_without_callback_replay():
    calls = []
    target = [2]

    class Normalize(Record):
        value: int

        @model_validator(mode="after")
        def scientific_normalization(self):
            calls.append(self.content_id)
            object.__setattr__(self, "value", target[0])
            return self

    expected = Normalize(value=2).content_id
    calls.clear()
    value = Normalize(value=1)
    assert len(calls) == 1 and value.content_id != calls[0]
    assert value.content_id == expected
    payload = value.model_dump_json()
    calls.clear()
    assert Normalize.model_validate_json(payload).content_id == value.content_id
    assert len(calls) == 1
    target[0] = 3
    class Pair(Record):
        first: Normalize
        second: Normalize
    calls.clear()
    pair = Pair(first={"value": 0}, second={"value": 0})
    assert len(calls) == 2 and pair.first.value == pair.second.value == 3


def test_equal_records_stay_equal_after_reading_one_identity():
    # 2/4 and -1/-2 reduce to the same rational. Reading the identity of one
    # side must not change their value equality or their hashes.
    first, second = Rational(numerator=2, denominator=4), Rational(numerator=-1, denominator=-2)
    first.content_id
    assert first == second and second == first and hash(first) == hash(second)
    assert len({first, second}) == 1
    changed = first.revise(numerator=3)
    assert changed != first and changed.content_id != first.content_id


def test_admitted_records_are_embedded_by_reference_without_repeating_admission(monkeypatch):
    import pickle

    import nwqlib.ir.validation as validation
    from nwqlib.ir import Definition, Program, Register, Sequence

    checks = []
    original = validation.check_program
    monkeypatch.setattr(validation, "check_program", lambda program: checks.append(program) or original(program))
    program = Program(root="root", definitions=(Definition(id="root", node=Sequence()),),
                      registers=(Register(name="q", width=1),))
    identity, readiness = program.content_id, program.check_readiness()
    assert len(checks) == 1

    class Holder(Record):
        program: Program
        label: str = "a"

    # Two parents and a parent revision share the admitted Program, its
    # stored identity and its Readiness; its admission does not run again.
    first, second = Holder(program=program), Holder(program=program, label="b")
    revised = first.revise(label="c")
    assert first.program is second.program is revised.program is program
    assert Program.model_validate(program) is program
    assert len(checks) == 1 and program.check_readiness() is readiness
    assert object.__getattribute__(program, "_identity_cache") == identity
    # Construction from field values or JSON still runs the full admission.
    copied = program.model_copy()
    assert copied is not program and copied == program and copied.content_id == identity and len(checks) == 2
    loaded = Holder.model_validate_json(first.model_dump_json())
    assert loaded.program is not program and loaded.program == program and len(checks) == 3

    # An instance of a subclass is validated from its field values, so a
    # subclass field outside the declared type rejects as it does in JSON.
    class Labelled(Holder):
        note: str = ""

    with pytest.raises(ValueError):
        Holder.model_validate(Labelled(program=program))

    class Parent(Record):
        child: Holder

    with pytest.raises(ValueError):
        Parent(child=Labelled(program=program))

    class Plain(Holder):
        pass

    assert type(Parent(child=Plain(program=program)).child) is Holder
    # An instance of an unrelated record type is the wrong type even when its
    # fields fit, so it rejects and is never converted.
    class Unrelated(Record):
        program: Program
        label: str = "a"

    for convert in (lambda: Holder.model_validate(Unrelated(program=program)),
                    lambda: Parent(child=Unrelated(program=program))):
        with pytest.raises(ValueError, match="instance of Holder"):
            convert()
    assert len(checks) == 3
    # Equality and hashing ignore the stored Readiness and identity.
    restored = pickle.loads(pickle.dumps(program))
    assert restored == program and hash(restored) == hash(program) and len(checks) == 3
    assert restored.check_readiness() == readiness and len(checks) == 4


def test_plan_memo_keeps_one_resolution_and_one_construction_over_a_point_sequence(monkeypatch):
    import gc
    import tracemalloc
    import weakref

    from nwqlib.core.records import Float64
    from nwqlib.core.planning import Realization
    from nwqlib.ir import Binding, Expression, Parameter, ParameterRef

    problem = Expectation(state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1))
    base = nwqlib.plan(problem, method=ExpectationMethod(), seed=7)
    program = base.construction.program
    program = program.revise(
        parameters=program.parameters + (Parameter(name="a", domain="real"), Parameter(name="k", domain="integer")),
        expressions=program.expressions + (Expression(id="a", value=ParameterRef(parameter="a")),))
    construction = base.construction.revise(program=program)
    plan = base.revise(construction=construction,
                       error_model=base.error_model.revise(construction_id=construction.content_id))
    name = plan.experiments[0].name
    # The miss counter and the last missing Plan are kept without a list
    # that would grow inside the traced interval.
    misses = [0, None]
    once = Plan._resolve_selection_once

    def counted(self, *args):
        misses[:] = [misses[0] + 1, self]
        return once(self, *args)

    monkeypatch.setattr(Plan, "_resolve_selection_once", counted)

    def bindings(i, k=1):
        return Binding(parameter="a", value=Float64(value=0.123 + 1e-3 * i)), Binding(parameter="k", value=k)

    def point(i, target=plan):
        realization = target.resolve(name, bindings=bindings(i))
        return realization, realization.selected_construction(target)

    first = point(0)
    # Memo-owned storage and the Plan's private cache stay constant over a
    # long sequence of distinct points: 300 and 3,000 points hold the same.
    gc.collect()
    tracemalloc.start()
    try:
        for i in range(1, 301):
            point(i)
        gc.collect()
        after_300, entries = tracemalloc.get_traced_memory()[0], len(plan._cache)
        for i in range(301, 3001):
            point(i)
        gc.collect()
        growth = tracemalloc.get_traced_memory()[0] - after_300
    finally:
        tracemalloc.stop()
    assert len(plan._cache) == entries and growth < 65536
    assert misses[0] == 3001
    # The latest point hits for its requested and canonical bindings, and a
    # reordered request is resolved again to the same point.
    last, selected = point(3000)
    assert misses[0] == 3001 and selected is point(3000)[1]
    reordered = plan.resolve(name, bindings=bindings(3000)[::-1])
    assert misses[0] == 3002 and reordered == last
    assert plan.resolve(name, bindings=last.bindings) == last and misses[0] == 3002
    # An evicted point is resolved again to the same selection.
    again = point(0)
    assert misses[0] == 3003 and again[0] == first[0]
    assert again[1] == first[1] and again[1].content_id == first[1].content_id
    # A changed Plan starts with its own memo while this Plan holds the point.
    changed = plan.revise(shots=100)
    realization, _ = point(0, changed)
    assert misses[0] == 3004 and misses[1] is changed
    assert realization.plan_id == changed.content_id != plan.content_id
    # A rejected point leaves both slots empty, so nothing keeps the last
    # selected construction alive and the last point is resolved again.
    kept = weakref.ref(again[1])
    del again, first, last, selected
    with pytest.raises(ValueError, match="exact integer"):
        Realization(plan_id=plan.content_id, experiment=name,
                    bindings=bindings(1, k=Float64(value=1.5))).selected_construction(plan)
    gc.collect()
    assert kept() is None
    plan.resolve(name, bindings=bindings(0))
    assert misses[0] == 3006 and misses[1] is plan


def test_frozen_array_field_equality_identity_and_json_roundtrip():
    import base64
    import struct

    import numpy as np

    class Table(Record):
        values: FrozenArray
        counts: FrozenArray

    table = Table(values=np.array([[1.5, -0.0], [2.0, 3.0]]), counts=np.array([1, 2**62], dtype=np.int64))
    assert_conforming_value(table)
    # The identity hashes little-endian element bytes with dtype and shape, so
    # a big-endian copy and a positive zero give the same value and identity.
    fields = table.model_dump(mode="json")
    assert fields["counts"] == {"dtype": "<i8", "shape": [2],
                                "data": base64.b64encode(struct.pack("<2q", 1, 2**62)).decode("ascii")}
    assert fields["values"]["data"] == base64.b64encode(struct.pack("<4d", 1.5, 0.0, 2.0, 3.0)).decode("ascii")
    same = Table(values=np.array([[1.5, 0.0], [2.0, 3.0]], dtype=">f8"), counts=table.counts)
    assert same == table and hash(same) == hash(table) and same.content_id == table.content_id
    for other in (table.revise(values=np.array([[1.5, 0.0], [2.0, 3.5]])),
                  table.revise(values=np.array([1.5, 0.0, 2.0, 3.0])),
                  table.revise(counts=np.array([1, 2**62 + 1], dtype=np.int64))):
        assert other != table and other.content_id != table.content_id
    with pytest.raises(ValueError):
        table.values.array[0, 0] = 7.0
    # Indexing, iteration and tolist read the stored rows, and the rows they
    # return are read-only views.
    values, before = table.values, table.model_dump(mode="json")
    assert values.shape == values.array.shape == (2, 2)
    row = values[0]
    assert row.tolist() == values.array[0].tolist() and not row.flags.writeable
    with pytest.raises(ValueError):
        row[1] = 7.0
    assert values[0, 1] == values.array[0, 1] and type(values[0, 1]) is np.float64
    assert [r.tolist() for r in values] == values.array.tolist() and len(list(values)) == values.shape[0]
    assert all(not r.flags.writeable for r in values)
    assert list(table.counts) == [1, 2**62] and table.counts[-1] == 2**62
    assert values.tolist() == values.array.tolist() == [[1.5, 0.0], [2.0, 3.0]]
    assert table.model_dump(mode="json") == before and values.to_json() == before["values"]
    assert same == table and hash(same) == hash(table) and same.content_id == table.content_id
    assert table.content_id == Table(values=np.array([[1.5, 0.0], [2.0, 3.0]]), counts=table.counts).content_id
    # Copies and pickles keep the value read-only.
    import copy
    import pickle

    restored = pickle.loads(pickle.dumps(table.values))
    assert restored == table.values and not restored.array.flags.writeable
    restored = copy.deepcopy(table)
    assert restored == table and not restored.values.array.flags.writeable
    for invalid in (np.array([1.0, np.nan]), np.array([1.0], dtype=np.float32), [1.0, 2.0],
                    {"dtype": "<f8", "shape": [2], "data": base64.b64encode(struct.pack("<d", 1.0)).decode("ascii")}):
        with pytest.raises(ValueError):
            Table(values=invalid, counts=table.counts)


def test_qhd_marginals_index_and_iterate_as_read_only_rows():
    import numpy as np
    import sympy as sp

    from nwqlib.algorithms.qhd import QHD, QuadraticSchedule
    from nwqlib.problems import Optimization
    from nwqlib.scientist import plan, solve

    x, y = sp.symbols("x y")
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2 + y ** 2, variables=(x, y),
                           bounds=((-1.0, 1.0), (-1.0, 1.0)))
    method = QHD(num_grid_points=3, num_steps=2, total_time=0.17, schedule=QuadraticSchedule(gamma=0.3))
    marginals = solve(plan(problem, method=method, execution="classical", seed=7)).marginals
    assert marginals.shape == (2, 3)
    rows = list(marginals)
    assert len(rows) == 2 and [row.tolist() for row in rows] == marginals.array.tolist()
    assert marginals[1].tolist() == marginals.array[1].tolist() and marginals[1, 2] == marginals.array[1, 2]
    assert not any(row.flags.writeable for row in (*rows, marginals[0]))
    assert np.shares_memory(marginals[0], marginals.array)


def _admission_family_plan(family, max_admission_steps):
    """Plan one small quantum case of the Method whose field sets its Programs' admission ceiling."""
    import numpy as np
    from nwqlib import Eigenproblem, LinearDynamics
    from nwqlib.algorithms import LCHS, FixedGCIM
    rows = (("II", .7), ("ZI", .5), ("XY", .1))
    basis = (np.array([1, 0, 0, 0], dtype=complex), np.array([0, 1, 1, 0], dtype=complex) / 2 ** .5)
    plans = (
        lambda: nwqlib.plan(Expectation(state=[1, 1], observable=[[1, .2], [.2, -1]]),
                    method=ExpectationMethod(max_admission_steps=max_admission_steps), shots=100, seed=7),
        lambda: nwqlib.plan(Eigenproblem(A=ingest_pauli(rows, num_qubits=2)),
                    method=FixedGCIM(basis=basis, max_admission_steps=max_admission_steps),
                    execution="quantum", shots=100, seed=7),
        lambda: nwqlib.plan(LinearDynamics(A=[[0, 1], [-1, 0]], initial_state=[1, 0], time=.2),
                            method=LCHS(max_admission_steps=max_admission_steps), seed=7),
    )
    return plans[family]()


@pytest.mark.parametrize("family", range(3))
def test_max_admission_steps_reaches_each_program_and_the_plan_identity(family, tmp_path):
    # The field sets AdmissionLimits.max_steps of every Program the Method
    # builds (ir.validation.admitted_program). A tiny ceiling is refused at
    # planning as a ValueError naming the field and the complete stored field
    # inventory, which an independent count of the selected Program's record
    # fields and tuple items reproduces. That value passes the inventory, so a
    # retry at it can only be refused by a later stage. The explicit 1_000_000
    # setting admits, and the field enters the saved Plan's identity.
    import re
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan

    def kept(value):
        if isinstance(value, Record):
            names = type(value).model_fields
            return len(names) + sum(kept(getattr(value, name)) for name in names)
        return len(value) + sum(map(kept, value)) if isinstance(value, tuple) else 0

    with pytest.raises(ValueError, match=r"\.max_admission_steps=1: its stored field inventory counts \d+ kept "
                                         r"field slots, the complete inventory") as caught:
        _admission_family_plan(family, 1)
    named = int(re.search(r"inventory counts (\d+)", str(caught.value)).group(1))
    selected = _admission_family_plan(family, 1_000_000)
    assert named == 1 + kept(selected.construction.program)
    try:
        _admission_family_plan(family, named)
    except ValueError as error:
        assert "stored field inventory" not in str(error)
    assert selected.construction.program.limits.max_steps == selected.method.max_admission_steps
    saved = save_plan(selected, ArchiveFiles(tmp_path, 4_000_000))
    assert load_plan(saved, ArchiveFiles(tmp_path, 4_000_000)).content_id == selected.content_id
