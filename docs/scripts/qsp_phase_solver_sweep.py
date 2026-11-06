#!/usr/bin/env python
"""Explicit seeded characterization of the canonical real-Wx phase solver.

This maintainer tool is separate from CI. It evaluates the requested
coefficient classes, degrees and seeds and reports numerical residuals and
the FFT lengths of every attempt. A failure preserves the fixed target and
reports the attempts and why the FFT doubling stopped. Results qualify
only the chosen instances and numerical arithmetic.

The classes are a single Chebyshev term, geometrically decaying parity
coefficients, flat random parity coefficients, and a near-unit target.
The first three have numerical norming cap 0.5, the last cap 0.98.
With --evolution, each tau/epsilon point prepares the two real parity
targets under the evolution function's preselected scale.
The default evolution grid contains 1000 preparations, so an explicitly
bounded --taus/--epsilons roster is appropriate for a limited campaign.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from nwqlib.subroutines.qsp.evolution import prepare_qsp_evolution
from nwqlib.subroutines.qsp.phases import chebyshev_polynomial_sup_bound, solve_wx_qsp_phases

CLASSES = ("single_term", "random_decay", "random_flat", "near_sup")
DEFAULT_DEGREES = (8, 16, 24, 32, 48, 64, 96, 128)
DEFAULT_SEEDS = (11, 12, 13)
DEFAULT_TAUS = tuple(round(0.05 * step, 2) for step in range(1, 201))
DEFAULT_EPSILONS = (1e-2, 1e-3, 1e-4, 1e-6, 1e-9)


def build_target(kind: str, degree: int, seed: int) -> np.ndarray:
    """Build one seeded target, normalized by its Chebyshev-zero norming estimate."""

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
    return coefficients * (sup_target / chebyshev_polynomial_sup_bound(coefficients))


def evolution_outcome(tau: float, epsilon: float) -> tuple[bool, str]:
    """Prepare one Jacobi-Anger evolution and summarize both parity solves."""

    try:
        prepared = prepare_qsp_evolution(tau=tau, epsilon=epsilon)
    except ValueError as error:
        return False, f"raised: {error}"
    solutions = (prepared.cos_solution, prepared.sin_solution)
    return True, (
        f"ok degree={prepared.expansion.degree} cap={prepared.target_cap:g} "
        f"residual={max(solution.residual_sup_bound for solution in solutions):.3e} "
        f"attempts={sum(len(solution.evidence.fft_lengths) for solution in solutions)}"
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
                    solution = solve_wx_qsp_phases(target)
                    outcome = (
                        f"ok residual={solution.residual_sup_bound:.3e} "
                        f"FFT={solution.evidence.fft_lengths}"
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
