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
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar, Literal
from unittest.mock import patch

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
from nwqlib.backends.connection import BackendRefresh, NativePreparation, circuit_layout
from nwqlib._prepared_execution import PreparationNotRebuilt
from nwqlib.blocks.kernels import BoundKernel
from nwqlib.core.records import Record, Source
from nwqlib.execution import CountsSampling, JobLocator, RunFailed
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
    with pytest.raises(ValueError, match="holds format"):
        resume_augmented_lagrangian(tmp_path / "refined", backend=None)
    from nwqlib.algorithms.qhd._durable import Directory

    with Directory.open(tmp_path / "a", "constrained"), pytest.raises(BlockingIOError):
        resume_augmented_lagrangian(tmp_path / "a", backend=None)


def test_an_unevaluated_quotient_resumes_as_the_uninterrupted_run(tmp_path):
    """An objective kept as supplied passes the expression check and continues as the uninterrupted run.

    f = x + (x + 2)/(x + 2), built with evaluate=False, has its pole outside [-1, 1], and SymPy's evaluation
    would turn it into x + 1. The rebuilt problem has the stored identity, so an interrupted run resumes,
    with each committed round kept, to the records of the uninterrupted run, and the ended directory
    reopens with the same record.
    """
    quotient = sp.Mul(x + 2, sp.Pow(x + 2, -1, evaluate=False), evaluate=False)
    objective = sp.Add(x, quotient, evaluate=False)
    problem = ConstrainedOptimization(objective=objective, variables=(x,), bounds=((-1.0, 1.0),),
                                      equalities=(x + sp.Rational(1, 4),))

    def run(directory, progress=False):
        return _solve(problem, max_iterations=3, progress=progress, directory=directory)

    resume, record = resume_augmented_lagrangian, lambda result: result.record
    whole = run(tmp_path / "whole")
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "cut", progress=_stop_at("input", 3))
    committed = _committed(tmp_path / "cut")
    # The third Run is round 2, so the two rounds before it are kept.
    assert (len(committed.get("iterations", ())), len(committed["levels"])) == (2, 0)
    resumed = resume(tmp_path / "cut", backend=None, progress=False)
    _assert_same_run(record(resumed), record(whole))
    assert sp.srepr(resumed.problem.objective) == sp.srepr(objective)
    assert resumed.problem.content_id == problem.content_id
    assert record(resume(tmp_path / "cut", backend=None)).content_id == record(resumed).content_id


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
