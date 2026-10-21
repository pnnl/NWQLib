"""Explicit completed-state and supplied-scalar checks keep their scientific scope."""

import pytest
from test_adapt_primary import plan_for
from nwqlib._prepared_execution import Run
from nwqlib.algorithms.gcim import AdaptVerificationOptions, adapt_verification
from nwqlib.execution import ExecutionMode


def values(facts):
    return {
        f.fact.quantity.rsplit(".", 1)[-1]: None if f.fact.value is None else f.fact.value.value
        for f in facts
    }


def test_zero_residual_of_excited_state_does_not_identify_ground():
    result = Run(plan_for(execution="classical", initial_state=(1, 0))).wait(
        timeout=5, poll_interval=0
    )
    assert result.eigenvalue == 1
    receipt, facts = result.verify(
        checks=AdaptVerificationOptions(name="residual", comparisons=("residual",))
    )
    assert values(facts)["residual_norm"] == 0
    assert receipt.result_id == result.content_id


def test_residual_square_and_heuristic_do_not_square_before_dividing():
    options = AdaptVerificationOptions(name="scale", comparisons=("residual",))
    for sigma, status in ((1e200, "binary64_overflow"), (1e-200, "binary64_underflow")):
        square, actual = adapt_verification._binary64_square(sigma)
        assert square is None and actual == status
    result = adapt_verification._residual_metrics(
        2e154, 0.0, (0.0, 1e155), options, ExecutionMode.THEORY, "max_iterations"
    )
    assert result["kato_heuristic_lower"] == pytest.approx(-4e153, abs=0, rel=1e-15)
    sampled = adapt_verification._residual_metrics(
        0.0, 1.0, (1.0, 2.0), options, ExecutionMode.SHOTS, "pool_exhausted"
    )
    assert sampled["spectral_nearness_lower"] is None and sampled["residual_threshold_met"] is None


def test_sector_facts_attach_only_to_the_reference_spin_that_produced_them():
    """spin_contamination is S^2 - s(s+1) for the selected reference spin s, and the sector frame does
    not carry s. A value computed for s=0 must not answer a check selected for s=1/2.
    """
    from nwqlib.evidence.error_model import Certificate
    from nwqlib.operators import ingest_pauli

    result = Run(plan_for(execution="classical", A=ingest_pauli((("ZI", 1.0), ("IZ", 0.5)), num_qubits=2),
                          initial_state=(1, 0, 0, 0), pool_rows=((("YI", 1j),),))).wait(timeout=5, poll_interval=0)
    options = AdaptVerificationOptions(name="sector", comparisons=("sector",))
    receipt, facts = result.verify(checks=options)
    base = Certificate(plan_id=result.plan_id, result_id=result.content_id,
                       assessment=result.assess(absolute_tolerance=1.0), checks=())
    with pytest.raises(ValueError, match="options identity"):
        base.with_verification(result, options=options.revise(reference_spin=0.5), evidence=facts)
    attached = base.with_verification(result, options=options, evidence=facts)
    assert tuple(check.fact for check in attached.checks) == facts
    assert {fact.fact.evidence.options_id for fact in facts} == {options.content_id} == {receipt.options_id}
    # |00> is the empty spatial orbital: S^2 = 0, so the contamination is 0 for s=0 and -3/4 for s=1/2.
    _, spin_half = result.verify(checks=options.revise(reference_spin=0.5))
    assert (values(facts)["spin_contamination"], values(spin_half)["spin_contamination"]) == (0.0, -0.75)


def test_native_verification_charges_the_evolved_circuits_before_building_states(monkeypatch):
    from qiskit.quantum_info import Statevector
    from nwqlib.algorithms.gcim.adapt_acquisition import basis_chains

    result = Run(plan_for()).wait(timeout=10, poll_interval=0)
    rec = result.plan.reconstruction
    d = 1 << rec.num_qubits
    selected = tuple(zip(result.value_selected, result.value_theta, strict=True))
    options = AdaptVerificationOptions(name="residual", comparisons=("residual",))
    # Record every circuit the verification simulates, starting from |0> or
    # evolving an existing state.
    simulated = []
    start, evolve = Statevector.from_instruction, Statevector.evolve

    def recorded_start(cls, circuit):
        simulated.append(circuit)
        return start(circuit)

    def recorded_evolve(self, circuit, qargs=None):
        simulated.append(circuit)
        return evolve(self, circuit, qargs)

    with monkeypatch.context() as guard:
        guard.setattr(Statevector, "from_instruction", classmethod(recorded_start))
        guard.setattr(Statevector, "evolve", recorded_evolve)
        result.verify(checks=options)
    # Each instruction of a simulated circuit is a k-qubit matrix gate. It
    # costs d * 2**k multiply-adds, two d-element state copies, 4**k to form
    # its matrix and one unit for its instruction copy. A nonzero global
    # phase costs d, and the reference starts from a d-element |0>.
    assert simulated and all(hasattr(item.operation, "__array__") for c in simulated for item in c.data)
    evolution = d + sum(
        (d if circuit.global_phase else 0)
        + sum(1 + 4**k + d * (2**k + 2) for k in (item.operation.num_qubits for item in circuit.data))
        for circuit in simulated
    )
    ritz = d * (2 * len(basis_chains(selected)) + 10)
    work = ritz + evolution + rec.hamiltonian_action_work + 3 * d
    with monkeypatch.context() as guard:
        for name in ("from_instruction", "evolve"):
            guard.setattr(
                Statevector, name, lambda *a, **k: pytest.fail("a state was built before admission")
            )
        with pytest.raises(ValueError, match=rf"needs {work} scalar operations .*max_products={work - 1}"):
            result.verify(checks=options.revise(max_products=work - 1))
    _, facts = result.verify(checks=options.revise(max_products=work))
    assert values(facts)["residual_norm"] < 1e-11
