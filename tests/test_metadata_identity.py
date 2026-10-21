"""Plan resolution memory over long point sequences and the per-Method admission ceiling."""

import pytest

import nwqlib
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.core.planning import Plan
from nwqlib.core import Record
from nwqlib.operators import ingest_pauli
from nwqlib.problems import Expectation


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


def _admission_plan(max_admission_steps):
    """Plan one small sampled expectation whose Method field sets its Programs' admission ceiling."""
    return nwqlib.plan(Expectation(state=[1, 1], observable=[[1, .2], [.2, -1]]),
                       method=ExpectationMethod(max_admission_steps=max_admission_steps), shots=100, seed=7)


def test_max_admission_steps_reaches_each_program_and_the_plan_identity(tmp_path):
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
        _admission_plan(1)
    named = int(re.search(r"inventory counts (\d+)", str(caught.value)).group(1))
    selected = _admission_plan(1_000_000)
    assert named == 1 + kept(selected.construction.program)
    try:
        _admission_plan(named)
    except ValueError as error:
        assert "stored field inventory" not in str(error)
    assert selected.construction.program.limits.max_steps == selected.method.max_admission_steps
    saved = save_plan(selected, ArchiveFiles(tmp_path, 4_000_000))
    assert load_plan(saved, ArchiveFiles(tmp_path, 4_000_000)).content_id == selected.content_id
