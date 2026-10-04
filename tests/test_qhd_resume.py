"""Interruption and resume of the durable directories of the QHD augmented-Lagrangian layer and box refinement.

Each run is interrupted by ``KeyboardInterrupt``, the manual interruption that the directories support, raised
from the public progress callback of an inner Run or while a Run waits for a detached job. The resumed run is
compared with an uninterrupted durable run of the same seed. The records must agree field by field apart from
the fields that differ between any two runs, namely the Run and Result identities, the stored-data byte
counts, whose text of wall times and measured timings varies in length, and the content identities that
contain them (``resume_augmented_lagrangian`` states the rule).
"""

from decimal import Decimal, localcontext
import json
from math import log, sqrt
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar, Literal
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp

from nwqlib.algorithms.qhd import (
    QHD,
    AugmentedLagrangian,
    BoxRefinement,
    UniformState,
    load_augmented_lagrangian,
    refine_box,
    resume_augmented_lagrangian,
    resume_box_refinement,
    run_resources,
    solve_augmented_lagrangian,
)
from nwqlib.algorithms.qhd._durable import header_committed
from nwqlib.backends.connection import BackendRefresh, NativePreparation, circuit_layout
from nwqlib._prepared_execution import PreparationNotRebuilt
from nwqlib.blocks.kernels import BoundKernel
from nwqlib.core.records import Record, Source
from nwqlib.execution import CountsSampling, ExecutionLimits, JobLocator, RunFailed
from nwqlib.problems import ConstrainedOptimization, Optimization

x, y = sp.symbols("x y")
BOX = ((-1.0, 1.0), (-1.0, 1.0))
# No grid point of the 3-by-3 interior grid meets x + y = 3/10, so every run ends at its iteration limit.
OFFGRID = ConstrainedOptimization(objective=(x - 1) ** 2 + (y - 1) ** 2, variables=(x, y), bounds=BOX,
                                  equalities=(x + y - sp.Rational(3, 10),))
QHD_SMALL = QHD(num_grid_points=3, num_steps=2, total_time=0.5)
REFINED = BoxRefinement(scaling="search_model", point_rule="best_observed", max_levels=2, mass_threshold=0.6)
DIFFERS = {"run_id", "result_id", "data_bytes", "content_id"}


def _solve(problem=OFFGRID, *, max_iterations=5, qhd=QHD_SMALL, **kwargs):
    kwargs.setdefault("progress", False)
    kwargs.setdefault("execution", "classical")
    options = AugmentedLagrangian(feasibility_tolerance=1e-3, max_iterations=max_iterations,
                                  inner_point="best_observed" if "refinement" in kwargs else "most_probable")
    return solve_augmented_lagrangian(problem, qhd=qhd, options=options, seed=3, **kwargs)


def _stop_at(stage, number):
    """A progress callback that interrupts the process at the ``number``-th report of ``stage`` by any Run."""
    seen = []

    def progress(name, done, total):
        if name == stage:
            seen.append(name)
            if len(seen) == number:
                raise KeyboardInterrupt(f"interrupted at {stage} {number}")
    return progress


def _committed(path):
    return json.loads((path / "controller.json").read_text())


@pytest.mark.parametrize("kind", ["constrained", "refinement"])
def test_a_serialization_interrupt_keeps_completed_outer_work(tmp_path, monkeypatch, kind):
    """Pydantic's real serialization wrapper must not turn Ctrl-C into a finished outer run."""
    import nwqlib.scientist as scientist

    original = sp.srepr
    planning = scientist._plan_with_streams
    armed = False
    plans = 0
    interrupt = KeyboardInterrupt("interrupted symbolic serialization")

    def interrupted_srepr(value, **kwargs):
        nonlocal armed
        if armed and sys._getframe(1).f_code.co_name == "_symbolic_record":
            armed = False
            raise interrupt
        return original(value, **kwargs)

    def interrupted_planning(*args, **kwargs):
        nonlocal plans, armed
        plans += 1
        armed = plans == 2
        return planning(*args, **kwargs)

    monkeypatch.setattr(sp, "srepr", interrupted_srepr)
    monkeypatch.setattr(scientist, "_plan_with_streams", interrupted_planning)
    directory = tmp_path / kind
    with pytest.raises(KeyboardInterrupt) as caught:
        if kind == "constrained":
            _solve(directory=directory)
        else:
            refine_box(Optimization(objective=(x - sp.Rational(3, 10)) ** 2,
                                    variables=(x,), bounds=((-1.0, 1.0),)),
                       qhd=QHD_SMALL, options=REFINED, execution="classical", seed=3,
                       directory=directory, progress=False)
    assert caught.value is interrupt
    saved = _committed(directory)
    assert saved["record"] is None
    assert len(saved["iterations" if kind == "constrained" else "levels"]) == 1


def _without_run_identities(value):
    if isinstance(value, dict):
        return {key: _without_run_identities(item) for key, item in value.items() if key not in DIFFERS}
    if isinstance(value, list):
        return [_without_run_identities(item) for item in value]
    return value


def _assert_same_run(resumed, whole):
    assert _without_run_identities(resumed.model_dump(mode="json")) == _without_run_identities(
        whole.model_dump(mode="json"))


@pytest.fixture
def counted(monkeypatch):
    """Count new Runs (``prepare``) and classical evolutions (host kernel invocations) while a test resumes."""
    import nwqlib.scientist as scientist

    counts = dict(prepare=0, evolutions=0)
    prepare, invoke = scientist.prepare, BoundKernel._invoke

    def counted_prepare(*args, **kwargs):
        counts["prepare"] += 1
        return prepare(*args, **kwargs)

    def counted_invoke(self):
        counts["evolutions"] += 1
        return invoke(self)

    monkeypatch.setattr(scientist, "prepare", counted_prepare)
    monkeypatch.setattr(BoundKernel, "_invoke", counted_invoke)
    return counts


def test_a_run_interrupted_between_rounds_resumes_without_repeating_them(tmp_path, counted):
    """Rounds 0 and 1 are committed when the process stops as round 2 creates its Run, before it prepares.

    Resume reads rounds 0 and 1 from the record and their Results from their Runs, continues round 2's
    existing Run and creates new Runs for rounds 3 and 4, so three evolutions and two new Runs follow. The
    progress callback of the resume call receives the reports of all three Runs, the continued one included,
    and only the two new Runs report their input.
    """
    whole = _solve(directory=tmp_path / "whole")
    counted.update(prepare=0, evolutions=0)
    with pytest.raises(KeyboardInterrupt):
        _solve(progress=_stop_at("input", 3), directory=tmp_path / "cut")
    committed = _committed(tmp_path / "cut")
    assert len(committed["iterations"]) == 2 and committed["record"] is None
    counted.update(prepare=0, evolutions=0)
    reports = []
    # Resume reads the saved preprocessing and check counts instead of evaluating the setup's tables and scans.
    with patch("nwqlib.algorithms.qhd.constrained._support_tables", side_effect=AssertionError("tabulated")), \
            patch("nwqlib.algorithms.qhd.constrained._domain_check", side_effect=AssertionError("scanned")):
        resumed = resume_augmented_lagrangian(tmp_path / "cut", backend=None,
                                              progress=lambda *report: reports.append(report))
    assert counted == dict(prepare=2, evolutions=3)
    assert reports.count(("analyze", 1, 1)) == 3 and reports.count(("input", 1, 1)) == 2
    assert resumed.termination == "iteration_limit" and len(resumed.iterations) == 5
    assert [item.run_id for item in resumed.iterations[:2]] == [item["run_id"] for item in committed["iterations"]]
    _assert_same_run(resumed.record, whole.record)
    # The resumed run saves and loads like any other, and resuming its ended directory returns it without
    # planning, evaluating or acquiring anything.
    loaded = load_augmented_lagrangian(resumed.save(tmp_path / "saved"))
    assert loaded.record.content_id == resumed.record.content_id and loaded.report() == resumed.report()
    with patch("nwqlib.algorithms.qhd.method.QHD.plan", side_effect=AssertionError("planned")), \
            patch("sympy.lambdify", side_effect=AssertionError("evaluated")), \
            patch("nwqlib.scientist.prepare", side_effect=AssertionError("acquired")):
        again = resume_augmented_lagrangian(tmp_path / "cut", backend=None)
    assert again.record.content_id == resumed.record.content_id
    assert [inner.content_id for inner in again.results] == [inner.content_id for inner in resumed.results]
    with pytest.raises(FileExistsError, match="resume_augmented_lagrangian"):
        _solve(directory=tmp_path / "cut")


class PendingBackend(Record):
    """Detached test backend with fixed one-hot counts, whose launch number ``pending`` is still running when first read.

    Grid point i of variable j is qubit ``j*K + i`` and qubit 0 is the rightmost count-key character, so
    48 of 64 shots land on grid point (1, 1) and 16 on (2, 0).
    """

    kind: Literal["pending_fixture"] = "pending_fixture"
    supports_synchronous: ClassVar[bool] = False
    requires_prepared_payload: ClassVar[bool] = False
    launches: ClassVar[list] = []
    refreshes: ClassVar[list] = []
    pending: ClassVar[int | None] = None

    def target_for(self, observation):
        return SimpleNamespace(readouts=("counts",))

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot):
        source = Source(name="pending fixture", version="1", domain="fixed one-hot counts", reference="test:pending")
        return NativePreparation(
            native=SimpleNamespace(width=circuit.num_clbits), target=source, compiler=source,
            native_basis=("supplied",), environment=(), quantum_layout=circuit_layout(circuit, circuit.qregs),
            classical_layout=circuit_layout(circuit, circuit.cregs), logical_to_native=tuple(range(circuit.num_qubits)),
            operations=len(circuit.data), population="unconditional", counts_sampling=CountsSampling(kind="fresh"),
            transformation="fixed one-hot counts")

    def restore_native(self, receipt, payload=None, *, run):
        return SimpleNamespace(width=sum(len(register.bits) for register in receipt.classical_layout))

    def admit_batch(self, natives):
        pass

    def launch(self, natives, *, submission_id, run):
        self.launches.append(submission_id)
        return JobLocator(provider=self.kind, job_id=submission_id)

    def refresh(self, locator, natives, *, run):
        self.refreshes.append(locator.job_id)
        if self.launches.index(locator.job_id) + 1 == self.pending and self.refreshes.count(locator.job_id) == 1:
            return BackendRefresh("acknowledged")
        width = natives[0].width

        def key(i, j):
            return format(1 << i | 1 << (width // 2 + j), f"0{width}b")

        result = SimpleNamespace(raw_output={"counts": {key(1, 1): 48, key(2, 0): 16}},
                                 metadata={"native_job_id": locator.job_id})
        return BackendRefresh("completed", results=(("0", result),))


def test_refined_rounds_resume_between_levels_and_inside_a_pending_level_run(tmp_path, counted, monkeypatch):
    """Each round refines its box over two levels, and the run is interrupted inside a round in two ways.

    Classical execution stops as the Run of round 1, level 2 is created, with round 0 and level 1 of round 1
    committed. Resume continues that Run, creates the two Runs of round 2 and runs three evolutions. With the
    detached backend, the job of round 0, level 2 is launched and still running when the process stops while
    waiting for it. Resume reads it once more from its original locator, launches only the four jobs of
    rounds 1 and 2, and charges the level's Run once.
    """
    whole = _solve(refinement=REFINED, max_iterations=3, directory=tmp_path / "whole")
    assert [len(item.refinement.levels) for item in whole.iterations] == [2, 2, 2]
    with pytest.raises(KeyboardInterrupt):
        _solve(refinement=REFINED, max_iterations=3, progress=_stop_at("input", 4), directory=tmp_path / "cut")
    committed = _committed(tmp_path / "cut")
    assert (len(committed["iterations"]), len(committed["levels"])) == (1, 1)
    counted.update(prepare=0, evolutions=0)
    resumed = resume_augmented_lagrangian(tmp_path / "cut", backend=None, progress=False)
    assert counted == dict(prepare=2, evolutions=3)
    assert resumed.iterations[1].refinement.levels[0].run_id == committed["levels"][0]["run_id"]
    _assert_same_run(resumed.record, whole.record)
    # The ended directory reopens with the problem that it stores.
    again = resume_augmented_lagrangian(tmp_path / "cut", backend=None)
    assert again.record.content_id == resumed.record.content_id and again.problem.content_id == OFFGRID.content_id

    quantum = dict(refinement=REFINED, max_iterations=3, execution="quantum", shots=64)
    monkeypatch.setattr(PendingBackend, "launches", [])
    monkeypatch.setattr(PendingBackend, "refreshes", [])
    whole = _solve(**quantum, backend=PendingBackend(), directory=tmp_path / "whole-detached")
    PendingBackend.launches.clear()
    monkeypatch.setattr(PendingBackend, "pending", 2)
    with patch("nwqlib._prepared_execution.sleep", side_effect=KeyboardInterrupt("interrupted while waiting")):
        with pytest.raises(KeyboardInterrupt):
            _solve(**quantum, backend=PendingBackend(), directory=tmp_path / "cut-detached")
    committed = _committed(tmp_path / "cut-detached")
    assert (len(committed["iterations"]), len(committed["levels"]), len(PendingBackend.launches)) == (0, 1, 2)
    interrupted = PendingBackend.launches[1]
    resumed = resume_augmented_lagrangian(tmp_path / "cut-detached", backend=PendingBackend(), progress=False)
    assert len(PendingBackend.launches) == 6 and PendingBackend.refreshes.count(interrupted) == 2
    level = resumed.iterations[0].refinement.levels[1]
    assert level.resources.circuit_attempts == 1 and level.resources.shots == 64
    _assert_same_run(resumed.record, whole.record)
    # The records carry known rotation counts, and the totals that run_resources forms from them and from the
    # inner Plans agree as well. Every level Run prepared one circuit, so the record's rotation total is the
    # body total of run_resources.
    assert level.resources.arbitrary_rotations > 0 and resumed.record.resources.arbitrary_rotations > 0
    totals = run_resources(resumed, synthesis_epsilon=1e-4)
    assert totals.model_dump() == run_resources(whole, synthesis_epsilon=1e-4).model_dump()
    assert totals.arbitrary_rotations == resumed.record.resources.arbitrary_rotations
    # The same counts trajectory supplies all six selected-region events.
    # PendingBackend returns (1, 1):48 and (2, 0):16; eta=.6 selects {1}
    # on both axes, so C_x = C_y = M = 48, S = 64 independently of the
    # coverage helper. Its Proposition 49 budget covers the whole AL run.
    before = dict(counted), len(PendingBackend.launches), list(PendingBackend.refreshes)
    with patch("nwqlib.execution.ObservationChunk.histogram", side_effect=AssertionError("report read observations")):
        for run in (whole, resumed):
            report = run.report()
            confidence = report["confidence"]
            assert confidence["horizon"] == 6 and confidence["failure_probability"] == 0.05
            assert report["summary"] == str(run)
            assert "simultaneous confidence 95%" in report["summary"]
            expected = [(item.iteration, level) for item in run.iterations for level in item.refinement.levels]
            assert len(confidence["levels"]) == 6
            for row, (iteration, level) in zip(confidence["levels"], expected, strict=True):
                assert (row["round"], row["level"], row["result_id"]) == (iteration, level.level, level.result_id)
                assert (row["dimension"], row["axis_counts"], row["joint_count"], row["valid_count"]) == (2, [48, 48], 48, 64)
                # Eight binary64 roundings cover the different evaluation order.
                radius = sqrt(log(4 * 6 * 12 / 0.05) / 128)
                assert row["radius"] == pytest.approx(radius, rel=8 * 2.0**-53, abs=0)
                # Independent binomial-tail residual of qhd._coverage.cp_lower;
                # the advisor's 1e-9 check is not a production error bound.
                with localcontext() as context:
                    context.prec = 75
                    p = Decimal.from_float(row["cp_lower"])
                    term = tail = p**64
                    for failures in range(1, 17):
                        term *= Decimal(65 - failures) / failures * (1 - p) / p
                        tail += term
                    beta = Decimal.from_float(0.05) / (2 * 6 * 6**2)
                    assert abs(tail / beta - 1) <= Decimal("1e-9")
            custom = run.report(failure_probability=0.1)
            assert "horizon 6, failure probability 0.1" in custom["summary"]
            assert custom["confidence"]["levels"][0]["lower_bound"] > confidence["levels"][0]["lower_bound"]
            assert run.report(failure_probability=None)["confidence"] is None
    assert (dict(counted), len(PendingBackend.launches), list(PendingBackend.refreshes)) == before


def test_a_standalone_refinement_resumes_between_levels(tmp_path, counted):
    """Two refinements stopped as a level creates its Run, whose stops depend on counts carried across it.

    With a split budget of one, level 2 splits a stalled box and level 5 stalls again, so the refinement
    stops with split_limit only when the resumed level 3 starts from the split count of level 2. With
    ``max_no_improve=2``, level 4 is best, level 5 does not improve on it, and the refinement stops with
    no_improvement after level 6 only when the resumed level 6 starts from level 4 as the best level and
    from the count that level 5 left. These stop sequences were recorded from the uniform start, which
    both Methods name because the default is the kinetic ground state.
    """
    problem = Optimization(objective=(2 * x**2 - 1) ** 2 + 3 * x / 5 + 2 * (y - sp.Rational(3, 10)) ** 2
                           + 6 * x * y / 5, variables=(x, y), bounds=((-1.2, 1.2), (-1.2, 1.2)))
    cases = (
        ("split", QHD(num_grid_points=5, num_steps=10, total_time=4.0, keep_state=True, initial_state=UniformState()),
         BoxRefinement(scaling="search_model", potential_gain=8.0, max_levels=8, max_no_improve=8,
                       stall_split="best_region"), 3, dict(prepare=2, evolutions=3)),
        ("stale", QHD(num_grid_points=4, num_steps=20, total_time=6.0, initial_state=UniformState()),
         BoxRefinement(scaling="search_model", potential_gain=8.0, max_levels=8, max_no_improve=2), 6,
         dict(prepare=0, evolutions=1)),
    )
    for name, qhd, options, level, work in cases:
        arguments = dict(qhd=qhd, options=options, execution="classical", seed=7)
        whole = refine_box(problem, **arguments, progress=False, directory=tmp_path / f"{name}-whole")
        if name == "split":
            assert whole.termination == "split_limit" and len(whole.levels) == 5
            assert [item.level for item in whole.levels if item.split is not None] == [2]
        else:
            assert whole.termination == "no_improvement" and len(whole.levels) == 6 and whole.best_level == 4
            assert whole.levels[4].objective > whole.levels[3].objective
        with pytest.raises(KeyboardInterrupt):
            refine_box(problem, **arguments, progress=_stop_at("input", level), directory=tmp_path / name)
        assert len(_committed(tmp_path / name)["levels"]) == level - 1
        counted.update(prepare=0, evolutions=0)
        resumed = resume_box_refinement(tmp_path / name, backend=None, progress=False)
        assert counted == work
        _assert_same_run(resumed, whole)
        # The refinement and the continuing resume attach the original problem, not a level's problem.
        assert whole.problem is problem and resumed.problem.content_id == problem.content_id
    again = resume_box_refinement(tmp_path / "stale", backend=None)
    assert again.content_id == resumed.content_id
    assert [inner.content_id for inner in again.results] == [inner.content_id for inner in resumed.results]
    # The resume of an ended refinement rebuilds the problem from problem.pickle, so its result saves.
    assert (again.problem.objective, again.problem.variables) == (problem.objective, problem.variables)
    assert again.problem.content_id == again.problem_id
    counted.update(prepare=0, evolutions=0)
    from nwqlib.algorithms.qhd import load_box_refinement

    loaded = load_box_refinement(again.save(tmp_path / "saved"))
    assert counted == dict(prepare=0, evolutions=0) and loaded.content_id == again.content_id


def test_level_file_save_and_load_are_admitted_against_the_level_max_bytes(tmp_path):
    """Saving and loading a level file are admitted against the level's QHD.max_bytes before the file is
    written or read, and each refusal names the smallest admitting limit, at which the file round-trips
    (``refinement._LevelTables``). The file-byte bound is the version-3 law of
    ``refinement._level_json_bound``, and a file of an earlier development format is refused.
    """
    problem = Optimization(objective=(2 * x**2 - 1) ** 2 + 3 * x / 5 + 2 * (y - sp.Rational(3, 10)) ** 2
                           + 6 * x * y / 5, variables=(x, y), bounds=((-1.2, 1.2), (-1.2, 1.2)))
    qhd = QHD(num_grid_points=4, num_steps=20, total_time=6.0, initial_state=UniformState())
    options = BoxRefinement(scaling="search_model", potential_gain=8.0, max_levels=8, max_no_improve=2)
    with pytest.raises(KeyboardInterrupt):
        refine_box(problem, qhd=qhd, options=options, execution="classical", seed=7,
                   progress=_stop_at("input", 3), directory=tmp_path / "run")
    from nwqlib.algorithms.qhd import refinement

    folder = tmp_path / "run" / "levels" / "1" / "run"
    saved = json.loads((folder.parent / "tables.json").read_text())
    box = [list(bounds) for bounds in problem.bounds]  # The first level's box is the problem's.
    specs = tuple((tuple(t["support"]), t["values"]["shape"][0]) for t in saved["tables"])
    stored = refinement._LevelTables.load(folder, box, qhd=QHD(), specs=specs, held_bytes=0)
    target = tmp_path / "copy" / "run"
    arguments = (target, box, stored.tables, stored.evaluations, stored.offset)
    with pytest.raises(ValueError, match=r"level tables save requires \d+ bytes") as caught:
        refinement._LevelTables.save(*arguments, qhd=QHD(max_bytes=1), held_bytes=0)
    limit = int(re.search(r"max_bytes>=(\d+)", str(caught.value)).group(1))
    with pytest.raises(ValueError, match=f"requires {limit} bytes"):
        refinement._LevelTables.save(*arguments, qhd=QHD(max_bytes=limit - 1), held_bytes=0)
    assert not (target.parent / "tables.json").exists()
    refinement._LevelTables.save(*arguments, qhd=QHD(max_bytes=limit), held_bytes=0)
    # A version-3 file holds exactly these four fields, within its file-byte bound.
    written = target.parent / "tables.json"
    assert set(json.loads(written.read_text())) == {"format", "tables", "evaluations", "offset"}
    assert written.stat().st_size <= refinement._level_json_bound(specs, stored.evaluations, stored.offset)
    with pytest.raises(ValueError, match=r"level tables load requires \d+ bytes") as caught:
        refinement._LevelTables.load(target, box, qhd=QHD(max_bytes=1), specs=specs, held_bytes=0)
    limit = int(re.search(r"max_bytes>=(\d+)", str(caught.value)).group(1))
    with pytest.raises(ValueError, match=f"requires {limit} bytes"):
        refinement._LevelTables.load(target, box, qhd=QHD(max_bytes=limit - 1), specs=specs, held_bytes=0)
    again = refinement._LevelTables.load(target, box, qhd=QHD(max_bytes=limit), specs=specs, held_bytes=0)
    assert [t.content_id for t in again.tables] == [t.content_id for t in stored.tables]
    assert (again.offset, again.evaluations) == (stored.offset, stored.evaluations)
    # J_v3 = 77 + T(specs) + D(evaluations) + R(offset) on the d = 2 fixture of three tables and
    # 24 evaluations, with a null or -5/2 offset (refinement._level_json_bound).
    fixture = (((0,), 4), ((1,), 4), ((0, 1), 16))
    assert refinement._level_json_bound(fixture, 24, None) == 1505
    assert refinement._level_json_bound(fixture, 24, sp.Rational(-5, 2)) == 1511


def test_resume_refuses_another_backend_another_layer_and_a_second_controller(tmp_path, monkeypatch):
    """Resume checks the backend, controller layer and exclusive lock before new work."""
    with patch("nwqlib.scientist._plan_with_streams", side_effect=KeyboardInterrupt("stopped while round 0 plans")):
        with pytest.raises(KeyboardInterrupt):
            _solve(execution="quantum", shots=64, directory=tmp_path / "a")
    refine_box(Optimization(objective=x**2, variables=(x,), bounds=((-1.0, 1.0),)), qhd=QHD_SMALL,
               options=REFINED, execution="classical", seed=3, progress=False, directory=tmp_path / "refined")
    monkeypatch.setattr("nwqlib.scientist._plan_with_streams", lambda *a, **k: pytest.fail("planned"))
    monkeypatch.setattr("nwqlib.scientist.prepare", lambda *a, **k: pytest.fail("prepared"))
    record = (tmp_path / "a" / "controller.json").read_bytes()
    from nwqlib.backends.connection import AerBackend

    with pytest.raises(ValueError, match="use the backend configuration"):
        resume_augmented_lagrangian(tmp_path / "a", backend=AerBackend.from_noise_model(_noise_model()),
                                    progress=False)
    assert (tmp_path / "a" / "controller.json").read_bytes() == record and not (tmp_path / "a" / "iterations").exists()
    with pytest.raises(ValueError, match="continue it with resume_box_refinement"):
        resume_augmented_lagrangian(tmp_path / "refined", backend=None)
    from nwqlib.algorithms.qhd._durable import Directory

    with Directory.open(tmp_path / "a", "constrained"), pytest.raises(BlockingIOError):
        resume_augmented_lagrangian(tmp_path / "a", backend=None)


def _files(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("kind", ["standalone", "constrained"])
def test_resume_refuses_a_problem_that_problem_pickle_does_not_rebuild(tmp_path, monkeypatch, kind):
    """Resume compares the rebuilt problem's identity with the stored record before it reads or advances anything.

    The objective sum_{n=0}^{2} Piecewise((x, x > 0), (x**2, True)), built with evaluation disabled, stores the
    summand as Mul(1, Piecewise(...)). ``archive._SymbolicReader`` rebuilds neither form of that Mul, since
    the evaluated build drops it and the unevaluated one wraps it again, so the rebuilt objective is another
    tree. A completed standalone refinement would otherwise return with that tree attached, and an
    interrupted refined constrained run would finish its later rounds, and its final record, on it. Each
    directory is refused with the message that its layer's loader gives, before any Run is reopened or
    planned, and the directory does not change.
    """
    from nwqlib.algorithms.qhd import load_box_refinement
    from nwqlib.algorithms.qhd.archive import _SymbolicReader

    r = sp.Symbol("x", real=True)
    n = sp.Symbol("n", integer=True)
    piece = sp.Piecewise((r, r > 0), (r**2, True), evaluate=False)
    with sp.evaluate(False):
        objective = sp.Sum(piece, (n, 0, 2))
    qhd = QHD(num_grid_points=5, include_boundary_points=True, num_steps=2, total_time=0.1)
    levels = BoxRefinement(max_levels=1, scaling="search_model")
    directory = tmp_path / "run"
    if kind == "standalone":
        problem = Optimization(objective=objective, variables=(r,), bounds=((-1.0, 1.0),))
        refined = refine_box(problem, qhd=qhd, options=levels, execution="classical", seed=1, progress=False,
                             directory=directory)
        message = "the saved SymPy objective and variables differ from the stored problem record"
        with pytest.raises(ValueError, match=message):
            load_box_refinement(refined.save(tmp_path / "saved"))
        resume = resume_box_refinement
    else:
        problem = ConstrainedOptimization(objective=objective, variables=(r,), bounds=((-1.0, 1.0),),
                                          equalities=(r + sp.Rational(1, 4),))
        with pytest.raises(KeyboardInterrupt):
            solve_augmented_lagrangian(problem, qhd=qhd, refinement=levels, execution="classical", seed=1,
                                       options=AugmentedLagrangian(max_iterations=3, feasibility_tolerance=1e-9),
                                       progress=_stop_at("input", 3), directory=directory)
        assert (len(_committed(directory)["iterations"]), len(_committed(directory)["levels"])) == (2, 0)
        message = "the live problem differs from the record's problem"
        resume = resume_augmented_lagrangian
    # The premise: the reader rebuilds the stored objective as another tree.
    with (directory / "problem.pickle").open("rb") as stream:
        rebuilt, *_ = _SymbolicReader(stream).load()
    assert sp.srepr(rebuilt) != sp.srepr(objective)
    committed, stored = _committed(directory), _files(directory)
    monkeypatch.setattr("nwqlib.scientist.load_run", lambda *a, **k: pytest.fail("reopened a Run"))
    monkeypatch.setattr("nwqlib.scientist._plan_with_streams", lambda *a, **k: pytest.fail("planned"))
    with pytest.raises(ValueError, match=message):
        resume(directory, backend=None)
    assert _committed(directory) == committed and _files(directory) == stored


@pytest.mark.parametrize("kind", ["constrained", "refined", "standalone"])
def test_an_unevaluated_quotient_resumes_as_the_uninterrupted_run(tmp_path, kind):
    """An objective kept as supplied passes the identity check of resume and continues as the uninterrupted run.

    f = x + (x + 2)/(x + 2), built with evaluate=False, has its pole outside [-1, 1], and SymPy's evaluation
    would turn it into x + 1. The rebuilt problem has the stored identity, so an interrupted run resumes,
    with each committed round or level kept, to the records of the uninterrupted run, and the ended
    directory reopens with the same record.
    """
    quotient = sp.Mul(x + 2, sp.Pow(x + 2, -1, evaluate=False), evaluate=False)
    objective = sp.Add(x, quotient, evaluate=False)
    if kind == "standalone":
        problem = Optimization(objective=objective, variables=(x,), bounds=((-1.0, 1.0),))

        def run(directory, progress=False):
            return refine_box(problem, qhd=QHD_SMALL, options=REFINED.revise(max_levels=3), execution="classical",
                              seed=3, progress=progress, directory=directory)

        resume, record = resume_box_refinement, lambda result: result
    else:
        problem = ConstrainedOptimization(objective=objective, variables=(x,), bounds=((-1.0, 1.0),),
                                          equalities=(x + sp.Rational(1, 4),))
        refinement = dict(refinement=REFINED) if kind == "refined" else {}

        def run(directory, progress=False):
            return _solve(problem, max_iterations=3, progress=progress, directory=directory, **refinement)

        resume, record = resume_augmented_lagrangian, lambda result: result.record
    whole = run(tmp_path / "whole")
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "cut", progress=_stop_at("input", 3))
    committed = _committed(tmp_path / "cut")
    # The third Run is round 2, the first level of round 1 or level 3, so the rounds or levels before it are kept.
    assert (len(committed.get("iterations", ())), len(committed["levels"])) == {
        "constrained": (2, 0), "refined": (1, 0), "standalone": (0, 2)}[kind]
    resumed = resume(tmp_path / "cut", backend=None, progress=False)
    _assert_same_run(record(resumed), record(whole))
    assert sp.srepr(resumed.problem.objective) == sp.srepr(objective)
    assert resumed.problem.content_id == problem.content_id
    assert record(resume(tmp_path / "cut", backend=None)).content_id == record(resumed).content_id


class RefusingBackend(PendingBackend):
    """PendingBackend whose native preparation raises from its second call on, as a failing native compiler would."""

    kind: Literal["refusing_fixture"] = "refusing_fixture"
    preparations: ClassVar[list] = []

    def prepare(self, circuit, **kwargs):
        self.preparations.append(circuit.num_qubits)
        if len(self.preparations) > 1:
            raise ValueError("the fixture refuses its second native preparation")
        return super().prepare(circuit, **kwargs)


def test_a_round_whose_preparation_raised_records_the_work_of_its_durable_run(tmp_path, monkeypatch):
    """The work that a failed round spent is counted, and a raised preparation adds no circuit law.

    Round 1's native preparation raises, so ``prepare`` closes its Run before the layer holds it. The durable
    Run folder keeps the charged preparation attempt and the stored data, and the round records them. The
    attempt does not establish a built circuit, so the round's CX bound and rotation count are unknown, and
    the run's rotation total is unknown both in the record and in ``run_resources``.
    """
    for name in ("launches", "refreshes", "preparations"):
        monkeypatch.setattr(RefusingBackend, name, [])
    result = _solve(execution="quantum", shots=64, backend=RefusingBackend(), directory=tmp_path / "refused")
    assert result.termination == "inner_failed" and len(result.iterations) == 2
    failed = result.iterations[1]
    counts = failed.resources
    assert failed.run_id is not None and failed.result_id is None
    assert (counts.circuit_preparations, counts.circuit_attempts, counts.shots) == (1, 0, 0) and counts.data_bytes > 0
    assert (counts.cx, counts.arbitrary_rotations) == (None, None)
    assert result.resources.arbitrary_rotations is None and run_resources(result).arbitrary_rotations is None


def test_a_round_refused_before_its_journal_header_records_zero_counts_and_its_stored_bytes(tmp_path):
    """A durable round whose Run is refused before its journal header records known zeros and its stored bytes.

    With ``max_data_bytes=220000`` round 0 completes, and round 1's Run stores its selected inputs, its
    selection file and ``run.json`` and is then refused while it commits its journal header, the record that
    holds its counters. Measured on 2026-09-27 with Python 3.12.14, Qiskit 2.5.2 and Aer 0.17.2 on macOS
    arm64, every tested limit from 203750 to 241000 gave this refusal, so 220000 lies well inside that
    band. A change of the stored sizes that moves the band fails the premise assertion below. Nothing is
    prepared or acquired before the header (``_durable.header_committed``), so every count of the round is
    zero, its CX bound and rotation count included, and its data bytes are the sizes of the files it stored
    apart from the journal. The run totals are then known sums over both rounds.
    """
    directory = tmp_path / "refused"
    result = _solve(execution="quantum", shots=64, max_iterations=3, directory=directory,
                    limits=ExecutionLimits(max_data_bytes=220000))
    folder = directory / "iterations" / "1" / "run"
    assert result.termination == "inner_failed" and len(result.iterations) == 2
    assert not header_committed(folder), "max_data_bytes=220000 no longer refuses round 1 before its header"
    first, refused = (item.resources for item in result.iterations)
    stored = sum(path.stat().st_size for path in folder.iterdir()
                 if path.is_file() and not path.name.startswith("run.sqlite"))
    assert result.iterations[1].run_id is None
    zero = ("circuit_preparations", "circuit_attempts", "completed_circuit_attempts", "shots", "completed_shots",
            "evolution_work", "construction_work", "synthesis_work", "cx", "arbitrary_rotations")
    assert {name: getattr(refused, name) for name in zero} == dict.fromkeys(zero, 0)
    assert refused.data_bytes == stored > 0 and refused.unavailable == ()
    totals = result.resources
    assert totals.data_bytes == first.data_bytes + stored
    assert {name: getattr(totals, name) for name in zero} == {name: getattr(first, name) for name in zero}
    assert run_resources(result).arbitrary_rotations == first.arbitrary_rotations


def test_a_failed_round_whose_journal_cannot_be_read_records_unknown_counts(tmp_path, monkeypatch):
    """A committed journal that cannot be read gives unknown counts, never the known zeros of a header-less Run.

    Round 1's native preparation raises after its Run committed the journal header and charged the
    preparation, as in ``test_a_round_whose_preparation_raised_records_the_work_of_its_durable_run``. While
    another connection holds an exclusive SQLite lock on that journal, reading the counters of the closed
    folder (``_durable.closed_trace``) fails. A failed read is no evidence that nothing was prepared
    (``_durable.header_committed``), so every count is unknown with the error as its reason.
    """
    import sqlite3

    from nwqlib.algorithms.qhd._durable import closed_trace, reopen
    from nwqlib.algorithms.qhd._outer import RUN_COUNTS, run_counts

    for name in ("launches", "refreshes", "preparations"):
        monkeypatch.setattr(RefusingBackend, name, [])
    monkeypatch.chdir(tmp_path)
    result = _solve(execution="quantum", shots=64, backend=RefusingBackend(), directory=Path("refused"))
    folder = Path("refused") / "iterations" / "1" / "run"
    assert result.iterations[1].resources.circuit_preparations == 1
    holder = sqlite3.connect(folder / "run.sqlite", isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        trace, reason = closed_trace(folder, RefusingBackend())
    finally:
        holder.close()
    counts, unavailable = run_counts(trace, reason)
    assert trace is None and "could not be read" in reason and "locked" in reason
    assert counts == dict.fromkeys(RUN_COUNTS) and dict(unavailable) == dict.fromkeys(RUN_COUNTS, reason)
    # A journal without read permission fails at once, with no lock to wait for. Reopening the committed
    # folder must keep it (``_durable.reopen``), never suggest removing it as a header-less folder.
    journal = folder / "run.sqlite"
    mode = journal.stat().st_mode
    journal.chmod(0)
    try:
        with pytest.raises(Exception) as caught:
            reopen(folder, backend=RefusingBackend(), progress=False)
    finally:
        journal.chmod(mode)
    assert "could not be read" in caught.value.__notes__[-1] and "Keep it" in caught.value.__notes__[-1]

    # SQLite creates the file before the transaction creates any tables. This is the genuine
    # empty-schema state of interrupted creation, without modifying a committed journal.
    empty = Path("empty")
    empty.mkdir()
    sqlite3.connect(empty / "run.sqlite").close()
    assert not header_committed(empty)
    with pytest.raises(FileNotFoundError) as caught:
        reopen(empty, backend=None, progress=False)
    assert "no committed journal header" in caught.value.__notes__[-1]
    assert (empty / "run.sqlite").is_file()


def _noise_model():
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    # Basis gates other than Aer's default ones select another Aer target, which a resumed run must keep.
    model = NoiseModel(basis_gates=["u", "cx"])
    model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["u"])
    model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    return model


def _native_bases(folder, backend):
    """The sorted native gate names that each Run folder under ``folder`` recorded for its preparations."""
    from nwqlib.scientist import load_run

    bases = []
    for run_folder in sorted(folder.glob("**/run")):
        with load_run(run_folder, backend=backend, progress=False) as run:
            bases.append(sorted({name for receipt in run.prepared_artifacts for name in receipt.native_basis}))
    return bases


# The noisy durable runs of test_a_noisy_aer_run_resumes_in_a_new_process: four rounds of OFFGRID, and a
# standalone refinement of three levels, each on depolarizing Aer with 64 shots.
NOISY_OPTIONS = AugmentedLagrangian(feasibility_tolerance=1e-3, max_iterations=4)
NOISY_LEVELS = BoxRefinement(scaling="search_model", point_rule="best_observed", max_levels=3, mass_threshold=0.6)
NOISY_REFINED = Optimization(objective=(x - sp.Rational(3, 10)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))


def _noisy_run(kind, backend, directory, progress=False):
    if kind == "constrained":
        return solve_augmented_lagrangian(OFFGRID, qhd=QHD_SMALL, options=NOISY_OPTIONS, backend=backend,
                                          execution="quantum", shots=64, seed=3, progress=progress,
                                          directory=directory)
    return refine_box(NOISY_REFINED, qhd=QHD_SMALL, options=NOISY_LEVELS, backend=backend, execution="quantum",
                      shots=64, seed=3, progress=progress, directory=directory)


def _killed_child(kind, directory):
    """Start the noisy durable run, print the backend's binding identity, and kill this process in its second Run.

    The kill comes at the second Run's completed analysis report, after its acquisition, so round 0 or level
    1 is committed and the Run of round 1 or level 2 exists with its observations.
    """
    from nwqlib.backends.connection import AerBackend

    backend = AerBackend.from_noise_model(_noise_model())
    print(backend.noise_model_id, flush=True)
    reports = []

    def progress(stage, done, total):
        if stage == "analyze" and done == total:
            reports.append(stage)
            if len(reports) == 2:
                os.kill(os.getpid(), signal.SIGKILL)

    _noisy_run(kind, backend, directory, progress)


def test_a_noisy_aer_run_resumes_in_a_new_process(tmp_path):
    """A durable run on noisy Aer, killed in a child process, resumes in this process with its noise model.

    The child binds a noise model with ``AerBackend.from_noise_model`` and is killed with SIGKILL during its
    second Run, as a crash would stop it. No object of this process holds that model, and the configuration
    ``AerBackend(noise_model_id=...)`` that the saved Runs require is unbound. Resume binds the model that the
    directory saved, with its basis gates u and cx, continues the second Run and runs the new Runs on it. The
    records equal those of an uninterrupted run on the same model, and every Run is lowered to the same u and
    cx target, not to Aer's default basis. A new ``from_noise_model`` binding has another identity, and resume
    refuses it before committing or creating anything.
    """
    from nwqlib.backends.connection import AerBackend

    tests = Path(__file__).parent
    children = {}
    for kind in ("constrained", "refinement"):
        code = f"import sys; sys.path.insert(0, {str(tests)!r}); import test_qhd_resume as t; t._killed_child(*sys.argv[1:])"
        children[kind] = subprocess.Popen([sys.executable, "-c", code, kind, str(tmp_path / kind)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for kind, child in children.items():
        output, errors = child.communicate(timeout=300)
        assert child.returncode == -signal.SIGKILL, errors
        identity = output.split()[0]
        folder = tmp_path / kind
        committed = _committed(folder)
        assert len(committed["iterations" if kind == "constrained" else "levels"]) == 1
        resume = resume_augmented_lagrangian if kind == "constrained" else resume_box_refinement
        record = (folder / "controller.json").read_bytes()
        runs = sorted(folder.glob("**/run"))
        with pytest.raises(ValueError, match=f"AerBackend\\(noise_model_id='{identity}'\\)"):
            resume(folder, backend=AerBackend.from_noise_model(_noise_model()), progress=False)
        assert (folder / "controller.json").read_bytes() == record and sorted(folder.glob("**/run")) == runs
        resumed = resume(folder, backend=AerBackend(noise_model_id=identity), progress=False)
        original = AerBackend.from_noise_model(_noise_model())
        whole = _noisy_run(kind, original, tmp_path / f"{kind}-whole")
        if kind == "constrained":
            assert resumed.termination == "iteration_limit" and len(resumed.iterations) == 4
            _assert_same_run(resumed.record, whole.record)
        else:
            assert resumed.termination == "level_limit" and len(resumed.levels) == 3
            _assert_same_run(resumed, whole)
        bases = _native_bases(folder, AerBackend(noise_model_id=identity))
        assert bases == _native_bases(tmp_path / f"{kind}-whole", original)
        assert all({"u", "cx"} <= set(basis) and not {"rz", "sx"} & set(basis) for basis in bases)


def test_an_interrupted_preparation_or_evolution_stops_resume_and_keeps_the_outer_record(tmp_path, counted):
    """An interruption leaves round 1's Run with a charged preparation or a reserved evolution, and neither can finish.

    Round 1's Run is interrupted after its preparation was charged, or after its evolution was reserved, and
    in a refined run the evolution of round 1, level 1 is. The Run neither builds a charged preparation again
    nor submits an uncertain attempt again, and resume never gives a round or level a second Run. Every
    default resume therefore raises the Run's error with a note that names the directory, prepares and
    evolves nothing and creates no Run. The outer record stays byte for byte. A resume with
    ``end_at_unfinishable=True`` ends the run there with inner_failed, still without new work, keeps round 0
    and records the error as the failure, and the directory then holds that ended run.
    """
    cases = (("prepared", {}, "prepare", 3, PreparationNotRebuilt, 2),
             ("evolved", {}, "submit", 2, RunFailed, 2),
             ("refined", dict(refinement=REFINED, max_iterations=3), "submit", 3, RunFailed, 3))
    for name, arguments, stage, number, error, count in cases:
        with pytest.raises(KeyboardInterrupt):
            _solve(**arguments, progress=_stop_at(stage, number), directory=tmp_path / name)
        record = (tmp_path / name / "controller.json").read_bytes()
        runs = sorted(path.relative_to(tmp_path) for path in (tmp_path / name).glob("iterations/**/run"))
        assert (len(json.loads(record)["iterations"]), len(json.loads(record)["levels"]), len(runs)) == (1, 0, count)
        counted.update(prepare=0, evolutions=0)
        for _ in range(2):
            with pytest.raises(error) as caught:
                resume_augmented_lagrangian(tmp_path / name, backend=None, progress=False)
            if error is RunFailed:
                assert (caught.value.stage, caught.value.status) == ("recovery", "uncertain")
            assert f"The durable run in {tmp_path / name} keeps" in caught.value.__notes__[-1]
        assert counted == dict(prepare=0, evolutions=0)
        assert (tmp_path / name / "controller.json").read_bytes() == record
        assert sorted(path.relative_to(tmp_path) for path in (tmp_path / name).glob("iterations/**/run")) == runs
        ended = resume_augmented_lagrangian(tmp_path / name, backend=None, progress=False, end_at_unfinishable=True)
        assert ended.termination == "inner_failed" and len(ended.iterations) == 2 and ended.iterations[1].evaluation is None
        assert ended.best is ended.last is ended.iterations[0] and ended.record.failure.startswith(error.__name__)
        assert counted == dict(prepare=0, evolutions=0) and _committed(tmp_path / name)["record"] is not None
        assert resume_augmented_lagrangian(tmp_path / name, backend=None).record.content_id == ended.record.content_id


def test_a_slack_round_resumes_with_its_committed_representation(tmp_path, counted):
    """Round 1 commits its inequality representation before it creates its Run, and the process stops in that Run.

    Resume reuses that representation without selecting again, continues round 1's Run and rebinds the
    completed round 0, whose inner Plan solves the slack objective, and the records equal those of an
    uninterrupted durable run. The equality x + y = 3/10 misses the 3-by-3 grid, so both runs reach
    their iteration limit.
    """
    problem = ConstrainedOptimization(objective=(x - 1) ** 2 + (y - 1) ** 2, variables=(x, y), bounds=BOX,
                                      equalities=(x + y - sp.Rational(3, 10),), inequalities=(x - y,))
    options = AugmentedLagrangian(feasibility_tolerance=1e-3, max_iterations=2, inequality_form="slack")

    def run(directory, progress=False):
        return solve_augmented_lagrangian(problem, qhd=QHD_SMALL, options=options, seed=3, execution="classical",
                                          progress=progress, directory=directory)

    whole = run(tmp_path / "whole")
    assert whole.iterations[0].representation.slacks
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "cut", progress=_stop_at("input", 2))
    committed = _committed(tmp_path / "cut")
    assert len(committed["iterations"]) == 1 and committed["representation"]["forms"] == list(
        whole.iterations[1].representation.forms)
    counted.update(prepare=0, evolutions=0)
    with patch("nwqlib.algorithms.qhd.constrained._slack_axis", side_effect=AssertionError("selected again")), \
            patch("nwqlib.algorithms.qhd.constrained._branch", side_effect=AssertionError("selected again")):
        resumed = resume_augmented_lagrangian(tmp_path / "cut", backend=None)
    assert counted == dict(prepare=0, evolutions=1)
    _assert_same_run(resumed.record, whole.record)
    loaded = load_augmented_lagrangian(resumed.save(tmp_path / "saved"))
    assert loaded.record.content_id == resumed.record.content_id


def test_a_converted_round_resumes_after_a_committed_stall_split_with_its_check_charges(tmp_path, counted,
                                                                                        monkeypatch):
    """A slack round's search-model refinement splits at level 2, and the process stops as level 3 creates its Run.

    f = x + 1 with the constant inequality -1 <= 0 in slack form on [-1, 1], K = 5 endpoint grids,
    ``mode_or_mean`` and three levels. Injected x marginals times a uniform slack factor make level 1 keep
    [-1, 1/4], make level 2 stall and split at its valley, and put level 3's mass on its last x point. A
    search-model level solves at unit points u and reports their rounded images a + D u, and the layer checks
    f and g at those images (constrained._level_check), so the split's scored points are checked at x = -11/16
    and 1/4, not at their unit coordinates 1/4 and 1. Resume restores levels 1 and 2 from the record without
    checking them again, continues level 3's Run and charges each restored level's checks, two for its grid
    point and mean and two more for its split, so the records, layer charges included, equal those of the
    uninterrupted run. The round checks two points per level, two scored points and its own point, its
    reservation (2 L + 2 B + 1) C of constrained._setup with L = 3 and B = 1.
    """
    from nwqlib.algorithms.qhd import constrained
    from nwqlib.algorithms.qhd.objective import node_count

    problem = ConstrainedOptimization(objective=x + 1, variables=(x,), bounds=((-1.0, 1.0),),
                                      inequalities=(sp.Integer(-1),))
    qhd = QHD(num_grid_points=5, include_boundary_points=True, keep_state=True)
    options = AugmentedLagrangian(inequality_form="slack", inner_point="mode_or_mean", max_iterations=1)
    refinement = BoxRefinement(scaling="search_model", point_rule="mode_or_mean", max_levels=3, max_no_improve=3,
                               mass_threshold=0.95, stall_split="best_region")
    checked, evaluate = [], constrained._evaluate

    def recorded(setup, point):
        checked.append(point)
        return evaluate(setup, point)

    def inject():
        rows = iter(((0.3, 0.5, 0.19, 0.005, 0.005), (0.15, 0.3, 0.01, 0.19, 0.35), (0.0, 0.0, 0.0, 0.0, 1.0)))
        monkeypatch.setattr("nwqlib.algorithms.qhd.method._evolve_restricted",
                            lambda *args: np.kron(np.sqrt(next(rows)), np.full(5, 1 / np.sqrt(5))).astype(complex))

    def run(directory, progress=False):
        return solve_augmented_lagrangian(problem, qhd=qhd, options=options, refinement=refinement,
                                          execution="classical", seed=3, progress=progress, directory=directory)

    monkeypatch.setattr(constrained, "_evaluate", recorded)
    inject()
    whole = run(tmp_path / "whole")
    item = whole.iterations[0]
    levels = item.refinement.levels
    split = levels[1].split
    assert item.representation.slacks and [level.split is not None for level in levels] == [False, True, False]
    assert levels[1].box == ((-1.0, 0.25), (0.0, 1.0)) and split.points == ((-0.6875, 0.0), (0.25, 0.0))
    assert len(checked) == 9 and checked[4:6] == [(-0.6875,), (0.25,)]
    cost = sum(node_count(term, 100) + 1 for term in (problem.objective, *problem.inequalities))
    assert item.resources.layer_evaluations == 2 * 9 and item.resources.layer_work == (2 * 3 + 2 * 1 + 1) * cost
    inject()
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "cut", progress=_stop_at("input", 3))
    committed = _committed(tmp_path / "cut")
    assert len(committed["levels"]) == 2 and committed["levels"][1]["split"] is not None
    checked.clear()
    counted.update(prepare=0, evolutions=0)
    resumed = resume_augmented_lagrangian(tmp_path / "cut", backend=None, progress=False)
    assert counted == dict(prepare=0, evolutions=1) and len(checked) == 3
    _assert_same_run(resumed.record, whole.record)
