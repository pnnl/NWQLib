"""Original four-qubit H2 pool ordering, energy, energy-unit and sector witnesses."""

from math import fsum
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
    ],
)
def test_published_h2_pool_anchor(pool, indices, family):
    problem = Eigenproblem(A=ingest_pauli(H2_STO3G_JW_TERMS, num_qubits=4))
    method = ADAPT(
        initial_state=ingest_occupation((1, 1, 0, 0), num_qubits=4),
        pool=pool,
        n_spatial_orbitals=2,
        max_iterations=4,
        gradient_norm_floor=1e-8,
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


# The Hartree energy in joules (CODATA 2018). 1e-18 J is the ordinary SI size of
# a molecular energy, so every Pauli coefficient lies far below the 1e-8 and
# 1e-14 that the explicit absolute controls below use.
HARTREE_IN_JOULES = 4.3597447222071e-18
H2_FCI_ELECTRONIC = -1.8510241683485222


def _h2_in_unit(a=1.0, b=0.0, **settings):
    """ADAPT on a*H2 + b*I with the spin-adapted pool, through the classical Plan and Run."""
    terms = tuple((p, a * c + (b if p == "IIII" else 0.0)) for p, c in H2_STO3G_JW_TERMS)
    problem = Eigenproblem(A=ingest_pauli(terms, num_qubits=4))
    method = ADAPT(initial_state=ingest_occupation((1, 1, 0, 0), num_qubits=4), pool="spin_adapted_sd",
                   n_spatial_orbitals=2, max_iterations=4, **settings)
    plan = method.plan(problem, output=problem.default_output(), execution="classical", shots=None,
                       rng=RandomStreams(7))
    return Run(plan).wait(timeout=15, poll_interval=0)


def test_energy_unit_change_rescales_the_energy_and_keeps_the_selection():
    """H' = aH + bI with a > 0 gives E' = aE + b and the same selection under the default thresholds.

    The expected energy is the FCI anchor of the test above, and 1e-12 is its
    tolerance, in Hartree after the conversion back. The explicit controls
    keep their absolute meaning in joules, so a floor of 1e-8 J lies above
    every gradient and the run ends at the reference energy of the Hartree
    run, and a cutoff of 1e-14 J removes every nonidentity term and leaves the
    identity coefficient whole.
    """
    a, b = HARTREE_IN_JOULES, 3e-18
    hartree = _h2_in_unit()
    joule = _h2_in_unit(a, b)
    assert hartree.eigenvalue == pytest.approx(H2_FCI_ELECTRONIC, abs=1e-12, rel=0)
    assert (joule.eigenvalue - b) / a == pytest.approx(H2_FCI_ELECTRONIC, abs=1e-12, rel=0)
    assert joule.selected == hartree.selected != ()
    assert joule.stop_reason == hartree.stop_reason

    floored = _h2_in_unit(a, b, gradient_norm_floor=1e-8)
    assert floored.selected == () and floored.stop_reason == "gradient_norm_floor"
    assert (floored.eigenvalue - b) / a == pytest.approx(hartree.history[0].energy, abs=1e-12, rel=0)

    cut = _h2_in_unit(a, b, pauli_coefficient_cutoff=1e-14)
    rec = cut.plan.reconstruction
    assert [t.label for t in rec.terms] == ["IIII"]
    assert (cut.eigenvalue - b) / a == pytest.approx(dict(H2_STO3G_JW_TERMS)["IIII"], abs=1e-12, rel=0)
    removed = fsum(abs(c) for p, c in H2_STO3G_JW_TERMS if p != "IIII")
    # Each scaled coefficient has relative error u = 1.1e-16 and fsum adds one rounding, so 1e-14 is a wide
    # roundoff window that still separates the sum from a mass that includes the identity coefficient.
    assert rec.hamiltonian_dropped_l1 / a == pytest.approx(removed, rel=1e-14, abs=0.0)
    assert format(rec.hamiltonian_dropped_l1, ".8g") in str(cut)
