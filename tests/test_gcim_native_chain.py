"""Native ADAPT generator-chain witness with exact and sampled acquisition."""

import numpy as np
import pytest

from nwqlib._prepared_execution import Run
from nwqlib.algorithms.gcim.adapt_records import screen_labels


def test_tiny_actual_adapt_consumer_keeps_native_generator_basis_and_sampled_provenance():
    from qiskit import QuantumCircuit
    from test_adapt_primary import plan_for as adapt_plan
    from test_gcim_native_preparations import reference_kwargs

    reference = QuantumCircuit(1)
    reference.h(0)
    for shots in (None, 32):
        plan = adapt_plan(
            shots=shots,
            initial_state=reference_kwargs(reference)["reference"],
            theta=np.pi / 4,
            max_iterations=1,
            gradient_norm_floor=0.0,
            pool_rows=((("Y", 1j),), (("Y", 1j), ("I", 1j)), (("X", 1j),)),
        )
        result = Run(plan).wait(timeout=5, poll_interval=0)
        assert result.selected == (0,) and result.theta == (np.pi / 4,)
        # [Z,iY]=[Z,iY+iI]=2X and [Z,iX]=-2Y. Shared X is measured once;
        # the disjoint Y setting remains, independent of the pool size three.
        # Only a sampled Plan forms readout groups, as indices into the
        # sorted commutator labels.
        assert screen_labels(plan.reconstruction.pool, {}) == ("X", "Y")
        assert plan.reconstruction.groups == (((0,), (1,)) if shots else ())
        gradients = dict(result.history[0].gradients)
        assert gradients[0] == gradients[1] == pytest.approx(2, rel=0, abs=1e-12)
        assert -2 <= gradients[2] <= 2
        assert len(result.pencil.hamiltonian) == 2
        pairs = [c for c in result.data.observations.chunks if c.experiment == "pair"]
        screens = [c for c in result.data.observations.chunks if c.experiment == "screen"]
        # Exact: the diagonals come from the two shared full-chain queries, each
        # with a diagonal-reduction point and a screen point, and one pair remains.
        assert len(pairs) == (1 if shots is None else 6) and len(screens) == (4 if shots is None else 2)
        if shots is None:
            assert result.eigenvalue == pytest.approx(-1.0, rel=0.0, abs=1e-12)
        else:
            assert all(c.returned_shots == 32 for c in screens)
            assert all(sum(v.count for v in c.values) == 32 for c in pairs)
            assert all(c.returned_shots == 32 for c in pairs)
            assert set(i for row in result.history for i in row.contribution_ids) <= set(
                result.contribution_ids
            )
