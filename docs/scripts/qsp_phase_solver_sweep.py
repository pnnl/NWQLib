#!/usr/bin/env python
"""Seeded exploration sweep for the symmetric-QSP phase solver.

Maintainer-batched characterization tool — NOT run by CI or the default
suite. Sweeps polynomial degree x coefficient class x seed over
``solve_symmetric_qsp_phases`` and prints one row per instance with the
outcome, so the numerical boundary of the [DMWL] solver stays observable
as degrees grow. Everything is deterministic. Targets come from
``numpy.random.default_rng(seed)`` over the fixed ``--seeds`` list, and the
solver draws no random numbers.

Coefficient classes (all definite parity, built at a stated sup norm):

- ``single_term``: the lone Chebyshev term ``0.5 * T_d`` — the slowest
  residual growth observed and the cleanest degree-scaling probe.
- ``random_decay``: geometrically decaying random parity coefficients at
  sup 0.5 — the friendly production-like regime.
- ``random_flat``: non-decaying random parity coefficients at sup 0.5.
- ``near_sup``: decaying random coefficients rescaled to sup 0.98, the
  regime where L-BFGS from the fixed start can stall and the Newton start
  of ``solve_symmetric_qsp_phases`` converges. Some degree-96 and
  degree-128 instances take 10 to 21 seconds each, which is why this sweep
  is maintainer-batched.

With seeds 11-13 and degrees up to 128 (Python 3.12.14, NumPy 2.5.2 and
SciPy 1.18.1 on macOS arm64), every instance of the three sup-0.5 classes
converges. Their largest residual grows from 1.3e-15 at degree 8 to
8.7e-14 at degree 128 (``0.5 * T_128``), and the 1e-12 acceptance gate is
the nearest threshold. Every admitted ``near_sup`` instance converges with
a residual below 2e-14. The degree-8 instance of seed 12 is rejected at
admission. ``build_target`` scales its maximum on ``4 (d + 1)`` nodes to
0.98, and its norming bound exceeds 1.

With ``--evolution`` the script instead runs ``prepare_qsp_evolution``
with its default controls at every point of ``--taus`` x ``--epsilons``.
The defaults are ``tau = 0.05, 0.10, ..., 10`` and ``epsilon`` in 1e-2,
1e-3, 1e-4, 1e-6 and 1e-9, which gives 1000 preparations of both parity
targets. Each row reports the selected degree and margin, the larger
parity residual and the evaluations of both parity solves together.
``docs/ENGINEERING_CONSTANTS.md`` records the result next to the phase
solver constants.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from nwqlib.subroutines.qsp.evolution import prepare_qsp_evolution
from nwqlib.subroutines.qsp.phases import chebyshev_grid, solve_symmetric_qsp_phases

CLASSES = ("single_term", "random_decay", "random_flat", "near_sup")
DEFAULT_DEGREES = (8, 16, 24, 32, 48, 64, 96, 128)
DEFAULT_SEEDS = (11, 12, 13)
DEFAULT_TAUS = tuple(round(0.05 * step, 2) for step in range(1, 201))
DEFAULT_EPSILONS = (1e-2, 1e-3, 1e-4, 1e-6, 1e-9)


def build_target(kind: str, degree: int, seed: int) -> np.ndarray:
    """Return definite-parity Chebyshev coefficients for one sweep instance."""

    rng = np.random.default_rng(seed)
    coefficients = np.zeros(degree + 1)
    parity = degree % 2
    if kind == "single_term":
        coefficients[degree] = 0.5
        return coefficients
    values = rng.normal(size=(degree // 2) + 1)
    if kind in ("random_decay", "near_sup"):
        values *= 0.85 ** np.arange(values.size)
    sup_target = 0.98 if kind == "near_sup" else 0.5
    coefficients[parity::2] = values
    grid = chebyshev_grid(4 * (degree + 1))
    sup = float(np.max(np.abs(np.polynomial.chebyshev.chebval(grid, coefficients))))
    return coefficients * (sup_target / sup)


def evolution_outcome(tau: float, epsilon: float) -> tuple[bool, str]:
    """Prepare one Jacobi-Anger evolution and summarize both parity solves."""

    try:
        prepared = prepare_qsp_evolution(tau=tau, epsilon=epsilon)
    except ValueError as error:
        return False, f"raised: {error}"
    solutions = (prepared.cos_solution, prepared.sin_solution)
    return True, (
        f"ok degree={prepared.expansion.degree} margin={prepared.margin:g} "
        f"residual={max(solution.max_residual for solution in solutions):.3e} "
        f"evaluations={sum(solution.evaluations for solution in solutions)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--classes", nargs="+", default=list(CLASSES), choices=CLASSES)
    parser.add_argument("--evolution", action="store_true")
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    arguments = parser.parse_args()

    if arguments.evolution:
        print(f"{'epsilon':>8s} {'tau':>7s} {'seconds':>8s}  outcome")
        failures = 0
        for epsilon in arguments.epsilons:
            for tau in arguments.taus:
                started = time.perf_counter()
                succeeded, outcome = evolution_outcome(tau, epsilon)
                failures += not succeeded
                elapsed = time.perf_counter() - started
                print(f"{epsilon:8.0e} {tau:7.3f} {elapsed:8.2f}  {outcome}")
        print(f"instances raised: {failures}")
        return 0

    print(f"{'class':13s} {'degree':>6s} {'seed':>5s} {'seconds':>8s}  outcome")
    failures = 0
    for kind in arguments.classes:
        for degree in arguments.degrees:
            for seed in arguments.seeds:
                target = build_target(kind, degree, seed)
                started = time.perf_counter()
                try:
                    solution = solve_symmetric_qsp_phases(target)
                    outcome = (
                        f"ok residual={solution.max_residual:.3e} "
                        f"evaluations={solution.evaluations}"
                    )
                except ValueError as error:
                    failures += 1
                    outcome = f"raised: {error}"
                elapsed = time.perf_counter() - started
                print(f"{kind:13s} {degree:6d} {seed:5d} {elapsed:8.2f}  {outcome}")
    print(f"instances raised: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
