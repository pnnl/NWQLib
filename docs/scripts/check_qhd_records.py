#!/usr/bin/env python3
"""Actual QHD public planning/injected analysis/report import-attempt audit, one-hot and binary.

It also runs one classical augmented-Lagrangian round, once with a single
QHD solve and once with box refinement, and reloads both archives, saves and
reloads a standalone box refinement, and it interrupts and resumes the
refined round in a durable directory. The
checked external launcher owns this process; no native or
reference simulation runs. Inert JSON cannot recover an executable SymPy
binding.
"""

import argparse

import builtins
import importlib.abc
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main(argv=None):
    """Check real planning and archive behavior using an injected grid state instead of native
    evolution.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; this audit must fail")
    args = parser.parse_args(argv)
    prefixes = (
        "qiskit",
        "qiskit_aer",
        "qiskit_ibm_runtime",
        "qiskit_ionq",
        "cirq",
        "nwqlib.backends.qiskit_aer",
    )
    attempts = []

    def forbidden(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)

    def record(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("blocked SDK/native import: " + name)

    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        record(name)
        for member in fromlist or ():
            record(name + "." + member)
        return original(name, globals, locals, fromlist, level)

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            record(fullname)
            return None

    # Intercept both import entry points and record attempts before raising.
    # The final assertion must also catch forbidden imports swallowed by callers.
    builtins.__import__ = guarded
    sys.meta_path.insert(0, Blocker())
    if args.poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass
    import numpy as np
    import sympy as sp
    from unittest.mock import patch
    from nwqlib.algorithms.qhd import QHD, GaussianState
    from nwqlib.problems import Optimization
    from nwqlib.scientist import plan, solve
    from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan

    x = sp.Symbol("x")
    problem = Optimization(objective=(x - 0.2) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    # Quantum planning evaluates the initial amplitudes for the CX law of the
    # structured preparation.
    initial = GaussianState(center=(0.2,), widths=(0.5,))
    selected = plan(problem, method=QHD(num_grid_points=3, initial_state=initial), seed=7)
    assert selected.reconstruction.width == 3
    # Inject only the evolved state to isolate the planning/analysis import
    # contract. This witness does not validate numerical QHD evolution.
    with patch(
        "nwqlib.algorithms.qhd.method._evolve_restricted",
        return_value=np.array([1.0, 0.0, 0.0], dtype=complex),
    ):
        result = solve(problem, method=QHD(num_grid_points=3), execution="classical", seed=7)
        assert result.candidate == (-0.5,) and abs(result.objective - 0.49) < 1e-14
        assert result.analyze().candidate == result.candidate
        assert result.report()["plan"]["execution"] == "classical"
        # The split-step flavor plans its transform law and state budget
        # without an SDK import too.
        split = solve(problem, method=QHD(num_grid_points=3, theory_flavor="split_step"), execution="classical",
                      seed=7)
        assert split.candidate == (-0.5,) and split.analyze().candidate == split.candidate
    with TemporaryDirectory() as directory:
        data = save_plan(selected, ArchiveFiles(directory, 4_000_000))
        with patch.object(QHD, "plan", side_effect=AssertionError("load replanned")):
            restored = load_plan(data, ArchiveFiles(directory, 4_000_000))
        assert restored == selected
    # Binary planning for native execution and the classical binary product
    # kernel, which applies the emitted QFT gates and diagonals with NumPy,
    # run without an SDK import, and the binary Plan reloads without replanning.
    y = sp.Symbol("y")
    binary_problem = Optimization(objective=(x - 0.2) ** 2 + x * y / 3, variables=(x, y),
                                  bounds=((-1.0, 1.0), (0.0, 2.0)))
    binary = QHD(encoding="binary", boundary="periodic", num_grid_points=4, num_steps=2,
                 theory_flavor="ir_product")
    native_binary = plan(binary_problem, method=binary, seed=7)
    assert native_binary.reconstruction.width == 4 and native_binary.reconstruction.steps
    host = solve(binary_problem, method=binary, execution="classical", seed=7)
    assert host.invalid_mass == 0 and abs(host.valid_mass - 1) < 1e-12
    with TemporaryDirectory() as directory:
        data = save_plan(native_binary, ArchiveFiles(directory, 4_000_000))
        with patch.object(QHD, "plan", side_effect=AssertionError("load replanned")):
            assert load_plan(data, ArchiveFiles(directory, 4_000_000)) == native_binary
    # A classical augmented-Lagrangian run plans, executes and analyzes one
    # QHD round per multiplier update. The injected state puts all mass on
    # x = -0.5, which satisfies x <= 0 with mu+ = 0, so the run stops after
    # one round. Saving and loading the run must not plan again.
    from nwqlib.algorithms.qhd import (
        AugmentedLagrangian,
        load_augmented_lagrangian,
        solve_augmented_lagrangian,
    )
    from nwqlib.problems import ConstrainedOptimization

    constrained = ConstrainedOptimization(
        objective=(x - 0.2) ** 2, variables=(x,), bounds=((-1.0, 1.0),), inequalities=(x,)
    )
    with patch(
        "nwqlib.algorithms.qhd.method._evolve_restricted",
        return_value=np.array([1.0, 0.0, 0.0], dtype=complex),
    ):
        run = solve_augmented_lagrangian(
            constrained,
            qhd=QHD(num_grid_points=3),
            options=AugmentedLagrangian(inner_point="best_observed"),
            execution="classical",
            seed=7,
            progress=False,
        )
    assert run.termination == "feasible_complementary" and run.candidate == (-0.5,)
    assert run.report()["record"]["termination"] == "feasible_complementary"
    with TemporaryDirectory() as directory:
        path = run.save(Path(directory) / "constrained")
        with patch.object(QHD, "plan", side_effect=AssertionError("load replanned")):
            reloaded = load_augmented_lagrangian(path)
        assert reloaded.record.content_id == run.record.content_id
        assert reloaded.results[0].candidate == (-0.5,)
    # The same problem with box refinement in the round: the search model
    # plans each level twice (unscaled tables, then the normalized
    # objective), and the injected state puts every level's mass on its first
    # grid point. Level 1 reports x = -0.5 and level 2, on the box
    # [-1, -0.25], x = -0.8125, whose L_0 is larger, so the round keeps level 1.
    from nwqlib.algorithms.qhd import BoxRefinement

    with patch(
        "nwqlib.algorithms.qhd.method._evolve_restricted",
        return_value=np.array([1.0, 0.0, 0.0], dtype=complex),
    ):
        refined = solve_augmented_lagrangian(
            constrained,
            qhd=QHD(num_grid_points=3),
            options=AugmentedLagrangian(inner_point="best_observed"),
            refinement=BoxRefinement(scaling="search_model", point_rule="best_observed", max_levels=2),
            execution="classical",
            seed=7,
            progress=False,
        )
    (item,) = refined.iterations
    assert refined.termination == "feasible_complementary" and refined.candidate == (-0.5,)
    assert len(item.refinement.levels) == 2 and item.refinement.best_level == 1
    assert refined.report()["refinement"] == [dict(levels=2, termination="level_limit")]
    with TemporaryDirectory() as directory:
        path = refined.save(Path(directory) / "refined")
        with patch.object(QHD, "plan", side_effect=AssertionError("load replanned")):
            reloaded = load_augmented_lagrangian(path)
        assert reloaded.record.content_id == refined.record.content_id
        assert [inner.content_id for inner in reloaded.results[0]] == [
            level.result_id for level in item.refinement.levels
        ]
    # A standalone refinement of the objective summarizes itself from its
    # record, and its result archive reloads through the public loader without
    # planning again.
    from nwqlib.algorithms.qhd import load_box_refinement, refine_box

    with patch(
        "nwqlib.algorithms.qhd.method._evolve_restricted",
        return_value=np.array([1.0, 0.0, 0.0], dtype=complex),
    ):
        standalone = refine_box(problem, qhd=QHD(num_grid_points=3),
                                options=BoxRefinement(point_rule="best_observed", max_levels=2),
                                execution="classical", seed=7, progress=False)
    assert len(standalone.levels) == 2 and "readout mode status" in str(standalone)
    with TemporaryDirectory() as directory:
        path = standalone.save(Path(directory) / "standalone")
        with patch.object(QHD, "plan", side_effect=AssertionError("load replanned")):
            reloaded = load_box_refinement(path)
        assert reloaded.content_id == standalone.content_id
        assert reloaded.problem.content_id == problem.content_id
        assert [inner.content_id for inner in reloaded.results] == [level.result_id for level in standalone.levels]
    # The refined run again in a durable directory, interrupted as its second
    # level creates its Run and resumed, which reopens that Run with load_run
    # and finishes it. The durable Runs and the outer record stay SDK-free.
    from nwqlib.algorithms.qhd import resume_augmented_lagrangian

    created = []

    def interrupt(stage, done, total):
        if stage == "input":
            created.append(stage)
            if len(created) == 2:
                raise KeyboardInterrupt("audit interruption")

    with TemporaryDirectory() as directory, patch(
        "nwqlib.algorithms.qhd.method._evolve_restricted",
        return_value=np.array([1.0, 0.0, 0.0], dtype=complex),
    ):
        arguments = dict(
            qhd=QHD(num_grid_points=3),
            options=AugmentedLagrangian(inner_point="best_observed"),
            refinement=BoxRefinement(scaling="search_model", point_rule="best_observed", max_levels=2),
            execution="classical",
            seed=7,
            directory=Path(directory) / "durable",
        )
        try:
            solve_augmented_lagrangian(constrained, **arguments, progress=interrupt)
        except KeyboardInterrupt:
            pass
        resumed = resume_augmented_lagrangian(Path(directory) / "durable", backend=None, progress=False)
        assert len(created) == 2 and len(resumed.iterations[0].refinement.levels) == 2
        assert resumed.candidate == refined.candidate and resumed.termination == refined.termination
    loaded = [name for name in sys.modules if forbidden(name)]
    assert not (attempts or loaded), (
        f"SDK isolation violation: attempts={attempts}, loaded={loaded}"
    )
    print(
        "QHD planning, injected host analysis, report, selected-data archive, binary planning, binary host "
        "product and archive, constrained augmented-Lagrangian runs with and without box refinement, "
        "their archives, a standalone refinement archive and a durable resumed run: attempts=[], loaded=[]"
    )


if __name__ == "__main__":
    main()
