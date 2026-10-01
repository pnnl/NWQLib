"""Existing four-qubit flat-counter and independent projected-space witnesses."""

from dataclasses import replace
import numpy as np
import pytest
from nwqlib.algorithms.gcim import ADAPT
from nwqlib.core.planning import RandomStreams
from nwqlib._prepared_execution import Run
from nwqlib.problems.records import Eigenproblem
from nwqlib.problems.inputs import ingest_occupation
from nwqlib.subroutines.fermionic_pool import (
    FermionicGenerator,
    enumerate_spin_adapted_gsd_pool,
    apply_generator_exponential,
)
from _fermionic_references import jw_single_excitation_generator


def solve(A, pool, **settings):
    problem = Eigenproblem(A=A)
    method = ADAPT(
        initial_state=ingest_occupation((1, 1, 0, 0), num_qubits=4), pool=pool, **settings
    )
    plan = method.plan(
        problem,
        output=problem.default_output(),
        execution="classical",
        shots=None,
        rng=RandomStreams(7),
    )
    return Run(plan).wait(timeout=60, poll_interval=0)


@pytest.mark.parametrize("pool_size,patience,expected", [(34, 1, 1), (34, 10, 6), (3, 10, 1)])
def test_flat_counter_follows_minimum_user_and_twenty_percent_limits(pool_size, patience, expected):
    base = enumerate_spin_adapted_gsd_pool(2)
    pool = tuple(replace(base[i % len(base)], pool_index=i) for i in range(pool_size))
    rng = np.random.default_rng(456)
    h = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    h = (h + h.conj().T) / 2
    result = solve(
        h,
        pool,
        energy_change_tolerance=1e9,
        gradient_norm_floor=0,
        t_user=patience,
        max_iterations=20,
        max_basis_size=40,
    )
    # All changes are flat, starting with the reference-to-first-basis change.
    # With 34 members and cap10, step5 has threshold5.8 and step6 has5.6.
    # Cap1 and the three-member pool stop at their first flat change.
    assert result.stop_reason == "flat_counter" and len(result.selected) == expected
    assert result.history[-1].flat_count == expected


def test_unavailable_projected_energy_resets_flat_history(monkeypatch):
    from nwqlib.algorithms.gcim import adapt_acquisition as acquisition

    original = acquisition._pencil_from_observations

    def unavailable_second_selection(plan, selected, context, **kwargs):
        pencil, diagnostics = original(plan, selected, context, **kwargs)
        if len(selected) == 2:
            pencil = pencil.revise(eigenvalues=(), coordinate_vectors=(), kept_rank=0,
                failure_reason="no_usable_overlap_subspace")
        return pencil, diagnostics

    monkeypatch.setattr(acquisition, "_pencil_from_observations", unavailable_second_selection)
    base = enumerate_spin_adapted_gsd_pool(2)
    pool = tuple(replace(base[i % len(base)], pool_index=i) for i in range(34))
    rng = np.random.default_rng(456)
    h = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    h = (h + h.conj().T) / 2
    result = solve(h, pool, energy_change_tolerance=1e9, gradient_norm_floor=0,
                   t_user=10, max_iterations=4, max_basis_size=10)
    assert result.history[-2].flat_count == 1
    assert result.history[-1].energy is None
    assert result.history[-1].flat_count == 0
    assert result.stop_reason == "no_usable_overlap_subspace"


def test_empty_overlap_subspace_publishes_unavailable_result():
    result = solve(np.diag(np.arange(16)), enumerate_spin_adapted_gsd_pool(2),
                   overlap_cutoff=2., max_iterations=1)
    assert result.eigenvalue is None and result.pencil.coefficients == ()
    assert result.pencil.kept_rank == 0
    assert len(result.pencil.overlap) == 1
    assert result.stop_reason == "no_usable_overlap_subspace"


def test_two_givens_subspace_reproduces_independent_qr_energy():
    pool = tuple(
        FermionicGenerator(
            family="custom",
            spatial_indices=(),
            pool_index=i,
            num_qubits=4,
            fermion_terms=(),
            pauli_terms=tuple(jw_single_excitation_generator(p, q, num_qubits=4).to_list()),
        )
        for i, (p, q) in enumerate(((2, 0), (3, 1)))
    )
    reference = np.zeros(16, complex)
    reference[3] = 1
    vectors = [reference] + [apply_generator_exponential(g, reference, np.pi / 4) for g in pool]
    product = reference
    for g in pool:
        product = apply_generator_exponential(g, product, np.pi / 4)
    vectors.append(product)
    q, _ = np.linalg.qr(np.column_stack(vectors))
    rng = np.random.default_rng(20260703)
    subspace = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    subspace = (subspace + subspace.conj().T) / 2
    h = q @ subspace @ q.conj().T
    # Explicitly choose the exactly Hermitian problem represented by the computed projection.
    h = (h + h.conj().T) / 2
    result = solve(h, pool, max_iterations=2, gradient_norm_floor=0)
    assert set(result.selected) == {0, 1}
    assert [max(1, 2 * len(step.selected)) for step in result.history] == [1, 2, 4]
    assert result.eigenvalue == pytest.approx(np.linalg.eigvalsh(subspace)[0], abs=1e-12, rel=0)
