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


def test_supplied_reference_comparison_requires_matching_units_and_scope(monkeypatch):
    """A same-frame scalar reference needs no Ritz-state materialization, while a wrong unit must
    fail.
    """
    from nwqlib.core.records import Float64, Source, Unit
    from nwqlib.evidence.records import Fact, Evidence
    from nwqlib.evidence.error_model import Certificate, FramedFact

    result = Run(plan_for(execution="classical")).wait(timeout=5, poll_interval=0)
    frame = result.plan.output.frame(result.plan.problem)
    source = Source(
        name="independent scalar",
        version="1",
        domain="exact small example",
        reference="Z has spectrum -1,+1",
    )
    reference = FramedFact(
        frame=frame,
        bindings=(),
        fact=Fact(
            quantity="reference",
            unit=frame.unit,
            scope=frame.scope,
            availability="concrete",
            value=Float64(value=-1.0),
            evidence=Evidence(kind="user_assertion", source=source),
        ),
    )
    options = AdaptVerificationOptions(
        name="supplied",
        comparisons=("supplied_reference_energy",),
        reference_energy=reference,
        reference_tolerance=1e-10,
    )
    monkeypatch.setattr(
        adapt_verification,
        "_ritz_state",
        lambda *a, **k: pytest.fail("scalar check materialized state"),
    )
    receipt, facts = result.verify(checks=options)
    value = facts[0].fact.value
    assert value.numerator / value.denominator < 1e-12
    base = Certificate(plan_id=result.plan_id, result_id=result.content_id,
                       assessment=result.assess(absolute_tolerance=1.0), checks=())
    attached, = base.with_verification(result, options=options, evidence=facts).checks
    # ADAPT facts carry their processed-state assumptions, so the check stays
    # INCONCLUSIVE while keeping the original fact.
    assert (attached.status, attached.fact) == ("INCONCLUSIVE", facts[0])
    # The supplied reference enters the frame conditioning through the options
    # identity, so the frame check rejects another reference.
    moved = reference.revise(fact=reference.fact.revise(value=Float64(value=-0.5)))
    with pytest.raises(ValueError, match="frame"):
        base.with_verification(result, options=options.revise(reference_energy=moved), evidence=facts)
    assert facts[0].fact.evidence.options_id == options.content_id == receipt.options_id
    badframe = frame.revise(unit=Unit(symbol="wrong", dimension="custom"))
    bad = reference.revise(frame=badframe, fact=reference.fact.revise(unit=badframe.unit))
    with pytest.raises(ValueError, match="unit"):
        result.verify(checks=options.revise(reference_energy=bad))
    assert receipt.result_id == result.content_id


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


def test_completed_classical_verification_reuses_immutable_saved_vectors(tmp_path, monkeypatch):
    from nwqlib.algorithms.gcim import adapt_acquisition, adapt_verification
    from nwqlib.saved_evidence import load_result

    run = Run(plan_for(execution="classical"))
    result = run.wait(timeout=5, poll_interval=0)
    cached = result.data.method_context["vectors"]
    assert cached and all(not value.flags.writeable for value in cached.values())
    with pytest.raises(TypeError):
        cached[()] = cached[()]
    run._state["method_context"].vectors.clear()
    assert cached
    location = result.save(tmp_path / "result")
    loaded = load_result(location)
    for module in (adapt_acquisition, adapt_verification):
        monkeypatch.setattr(
            module,
            "packed_exponential",
            lambda *a, **k: pytest.fail("completed verification replayed a saved prefix"),
        )
    for candidate in (result, loaded):
        _, facts = candidate.verify(
            checks=AdaptVerificationOptions(name="residual", comparisons=("residual",))
        )
        assert values(facts)["residual_norm"] < 1e-11


def test_quantum_result_excludes_mutable_native_caches_and_verifies_saved_selection(tmp_path):
    from nwqlib.saved_evidence import load_result

    run = Run(plan_for())  # One system plus one interferometer qubit.
    result = run.wait(timeout=10, poll_interval=0)
    assert run._state["method_context"].gates
    native_names = {"gates", "preparations", "reference_gates", "matrix_data"}
    assert native_names.isdisjoint(result.data.method_context)
    loaded = load_result(result.save(tmp_path / "quantum-result"))
    assert native_names.isdisjoint(loaded.data.method_context)
    for candidate in (result, loaded):
        receipt, facts = candidate.verify(checks=AdaptVerificationOptions(
            name="selected-residual", comparisons=("residual",)))
        assert values(facts)["residual_norm"] < 1e-11
        work = {binding.parameter: binding.value for binding in receipt.applications[0].arguments}
        assert work["native_generator_constructions"] > 0


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


def test_statevector_evolution_work_follows_definitions_of_matrix_free_gates():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate, Instruction

    inner = QuantumCircuit(2, global_phase=0.3)
    inner.h(0)
    inner.cx(0, 1)
    outer = QuantumCircuit(3)
    outer.append(inner.to_gate(), [0, 2])
    outer.barrier()
    outer.x(1)
    # The composite gate has no matrix, so Qiskit evolves its definition: its
    # phase, its H and its CX. The barrier costs nothing.
    d = 8
    expected = d + (1 + 4 + 4 * d) + (1 + 16 + 6 * d) + (1 + 4 + 4 * d)
    assert adapt_verification._statevector_evolution_work(outer, d) == expected
    opaque = QuantumCircuit(1)
    opaque.append(Instruction("opaque", 1, 0, []), [0])
    assert not isinstance(opaque.data[0].operation, Gate)
    with pytest.raises(ValueError, match="cannot simulate instruction opaque"):
        adapt_verification._statevector_evolution_work(opaque, d)


def test_projected_verification_reads_the_adapt_pencil_without_new_work(monkeypatch):
    """ADAPT routes projected checks to its stored pencil: no state replay, Ritz state or solve."""
    import numpy as np
    from nwqlib.algorithms.gcim import adapt_acquisition
    from nwqlib.evidence.verification import ProjectedVerificationOptions

    result = Run(plan_for(execution="classical")).wait(timeout=5, poll_interval=0)
    pencil = result.pencil

    def forbidden(*args, **kwargs):
        pytest.fail("projected verification repeated numerical work")

    monkeypatch.setattr(adapt_acquisition, "packed_exponential", forbidden)
    monkeypatch.setattr(adapt_verification, "_ritz_state", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    comparisons = ("overlap_normalization", "gram_psd_deficit", "gram_hermiticity", "projected_backward_error")
    receipt, facts = result.verify(
        checks=ProjectedVerificationOptions(name="projected", comparisons=comparisons, tolerance=1e-10)
    )
    measured = {fact.frame.quantity: fact.fact.value.numerator / fact.fact.value.denominator for fact in facts}
    overlap = np.array([[complex(value.real, value.imag) for value in row] for row in pencil.overlap])
    defect = overlap - overlap.conj().T
    # Stored S entries and their mirrors are nearly equal, so these float differences are exact.
    assert measured["gram_hermiticity"] == max(np.abs(defect.real).max(), np.abs(defect.imag).max())
    assert measured["gram_psd_deficit"] == max(0.0, -min(pencil.overlap_eigenvalues))
    assert measured["overlap_normalization"] == pencil.overlap_normalization_error
    assert measured["projected_backward_error"] == pencil.projected_backward_error
    assert receipt.result_id == result.content_id and len(facts) == len(comparisons)
