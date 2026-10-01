"""Original four-qubit H2 pool ordering, energy and sector witnesses."""

import pytest
from _h2_sto3g_reference import H2_STO3G_JW_TERMS
from nwqlib.algorithms.gcim import ADAPT, AdaptVerificationOptions
from nwqlib.operators import ingest_pauli
from nwqlib.problems.inputs import ingest_occupation
from nwqlib.problems.records import Eigenproblem
from nwqlib.core.planning import RandomStreams
from nwqlib._prepared_execution import Run


@pytest.mark.parametrize(
    "pool,indices,family",
    [
        ("spin_adapted_sd", None, "double_singlet"),
        ("uccsd_sd", (2,), "uccsd_double"),
        ("qeb_sd", (2,), "qeb_double"),
        ("ceo_ovp", (2, 3), "ceo_ovp_plus"),
    ],
)
def test_published_h2_pool_anchor(pool, indices, family):
    problem = Eigenproblem(A=ingest_pauli(H2_STO3G_JW_TERMS, num_qubits=4))
    method = ADAPT(
        initial_state=ingest_occupation((1, 1, 0, 0), num_qubits=4),
        pool=pool,
        n_spatial_orbitals=2,
        max_iterations=4,
        gradient_norm_floor=0 if pool in ("qeb_sd", "ceo_ovp") else 1e-8,
    )
    plan = method.plan(
        problem,
        output=problem.default_output(),
        execution="classical",
        shots=None,
        rng=RandomStreams(7),
    )
    result = Run(plan).wait(timeout=15, poll_interval=0)
    assert result.eigenvalue == pytest.approx(-1.8510241683485222, abs=1e-12, rel=0)
    if indices is not None:
        assert result.selected == indices
    assert plan.reconstruction.pool[result.selected[0]].family == family
    _, facts = result.verify(
        checks=AdaptVerificationOptions(name="sector", comparisons=("sector",))
    )
    values = {f.fact.quantity.rsplit(".", 1)[-1]: f.fact.value.value for f in facts}
    assert values["particle_number"] == pytest.approx(2.0, abs=1e-12, rel=0)
    assert all(
        values[name] == pytest.approx(0.0, abs=1e-12, rel=0)
        for name in ("spin_z", "spin_squared", "spin_contamination")
    )
    if pool == "uccsd_sd":
        assert dict(result.history[0].gradients)[2] == pytest.approx(
            -0.2563810912791524, abs=1e-12, rel=0
        )
