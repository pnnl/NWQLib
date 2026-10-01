"""Bounded 3/5-qubit sampling and first-order refinement relations."""

import numpy as np
import sympy as sp
from nwqlib.algorithms.qhd import QHD, QHDVerification, QuadraticSchedule
from nwqlib.problems import Optimization
from nwqlib.scientist import solve


def test_qhd_shots_statistical():
    x = sp.Symbol("x")
    problem = Optimization(objective=(x - 0.3) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=5, num_steps=40)
    exact = solve(problem, method=method, seed=1234)
    sampled = solve(problem, method=method, shots=2000, seed=1234)
    exact_histogram = exact.data.observations.chunks[0].histogram()
    sampled_histogram = sampled.data.observations.chunks[0].histogram()
    probabilities = dict(zip(exact_histogram.index_list(), exact_histogram.weights.tolist()))
    counts = dict(zip(sampled_histogram.index_list(), sampled_histogram.weights.tolist()))
    assert sampled.valid_probability >= 0.99 and sampled.candidate_indices == (3,)
    for index in range(5):
        outcome = 1 << index
        probability = probabilities[outcome]
        sigma = np.sqrt(probability * (1 - probability) / 2000)
        assert abs(counts.get(outcome, 0) / 2000 - probability) <= 4 * sigma
    assert sampled.data.observations.chunks[0].returned_shots == 2000


def test_qhd_trotter_convergence():
    x = sp.Symbol("x")
    problem = Optimization(objective=(x - 0.2) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    errors = []
    for steps in (1, 2, 4, 8):
        result = solve(
            problem,
            method=QHD(
                num_grid_points=3,
                num_steps=steps,
                total_time=0.8,
                schedule=QuadraticSchedule(gamma=0.2),
                trotter_order=1,
                keep_state=True,
            ),
            seed=1234,
        )
        receipt, _ = result.verify(checks=QHDVerification(comparisons=("schrodinger_fidelity",)))
        errors.append(
            next(
                f.fact.value.value
                for f in receipt.applications[0].facts
                if f.fact.quantity == "schrodinger_infidelity"
            )
        )
    assert errors == sorted(errors, reverse=True) and errors[-1] <= 0.01
