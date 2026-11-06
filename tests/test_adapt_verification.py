"""Explicit completed-state and supplied-scalar checks keep their scientific scope."""

import numpy as np
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


def test_residual_classification_exists_only_for_a_supplied_tolerance():
    """H = Z from |+> with the only direction iZ has E = 0, zero gradient and sigma = ||Z|+>|| = 1.

    The raw residual and the interval [E - sigma, E + sigma] = [-1, 1], which holds both eigenvalues, need no
    target. The three flags compare sigma with a target in the energy unit, so without one they are
    unavailable, never a pass or a false, and the check's threshold and the flags read the same value. The
    values are exact: the state and H have entries 1/sqrt(2) and 1, so 1e-12 covers their rounding.
    """
    result = Run(plan_for(execution="classical", initial_state=(1, 1), pool_rows=((("Z", 1j),),))).wait(
        timeout=5, poll_interval=0)
    assert result.stop_reason == "gradient_norm_floor" and result.eigenvalue == pytest.approx(0.0, abs=1e-12, rel=0)
    flags = ("residual_threshold_met", "pool_limited", "stalled")

    def facts_of(**settings):
        options = AdaptVerificationOptions(name="r", comparisons=("residual",), **settings)
        thresholds = {check.name: check.threshold for check in options.verification_checks(result)}
        assert thresholds["r.residual_norm"] == settings.get("residual_tolerance")
        return {f.fact.quantity.removeprefix("r."): f.fact for f in result.verify(checks=options)[1]}

    unset = facts_of()
    assert unset["residual_norm"].value.value == pytest.approx(1.0, abs=1e-12, rel=0)
    assert unset["spectral_nearness_lower"].value.value == pytest.approx(-1.0, abs=1e-12, rel=0)
    assert unset["spectral_nearness_upper"].value.value == pytest.approx(1.0, abs=1e-12, rel=0)
    for name in flags:
        assert unset[name].availability == "unknown" and unset[name].value is None
        assert "residual_tolerance" in unset[name].reason
    # sigma = 1 is above .5 and below 2, far from either target. A gradient-floor stop above the target is
    # pool_limited, by the definition of that label.
    above = facts_of(residual_tolerance=.5)
    assert tuple(above[name].value.value for name in flags) == (0.0, 1.0, 0.0)
    below = facts_of(residual_tolerance=2.0)
    assert tuple(below[name].value.value for name in flags) == (1.0, 0.0, 0.0)


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


@pytest.mark.parametrize("comparisons", [("sector",), ("residual", "sector")])
def test_sector_verification_charges_its_live_arrays_before_building_the_ritz_state(comparisons, monkeypatch):
    """The sector stage is charged with the arrays it holds beside the caller's, and a smaller limit refuses first.

    Classical ADAPT on ten qubits, H = Z_9 + 0.5 X...X from |0...0> with
    pool i Y_9 and i Y_0. The verification check counts the Ritz state, a
    residual when one was computed, the normalized copy and probabilities
    of ``sector_expectations`` and the arrays of ``apply_spin_squared``,
    with the arrays the saved Result already holds. That count covers the
    traced peak of a repeated verification plus the arrays held before it,
    and half that count refuses before any sector array exists.
    """
    import tracemalloc
    from nwqlib.algorithms.gcim import sector
    from nwqlib.operators import ingest_pauli
    from nwqlib.operators import access
    from nwqlib.problems import ingest_occupation

    result = Run(plan_for(execution="classical", A=ingest_pauli((("Z" + "I" * 9, 1.0), ("X" * 10, .5)), num_qubits=10),
                          initial_state=ingest_occupation((0,) * 10, num_qubits=10),
                          pool_rows=((("Y" + "I" * 9, 1j),), (("I" * 9 + "Y", 1j),)))).wait(timeout=30, poll_interval=0)
    options = AdaptVerificationOptions(name="sector", comparisons=comparisons)
    checks, real = [], access._check_bytes

    def counted(size, max_bytes, operation="input"):
        if operation == "ADAPT explicit verification":
            checks.append(size)
        return real(size, max_bytes, operation)

    monkeypatch.setattr(adapt_verification, "_check_bytes", counted)
    held = sum({id(a): a.nbytes for name in ("vectors", "h_columns")
                for a in result.data.method_context[name].values()}.values())
    # A first call leaves the one-time allocations of the process outside the traced peak.
    result.verify(checks=options)
    del checks[:]
    tracemalloc.start()
    try:
        result.verify(checks=options)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert checks and checks[0] >= peak + held
    need = checks[0]
    del checks[:]
    monkeypatch.setattr(sector, "_apply_spin_squared", lambda *a, **k: pytest.fail("sector arrays built"))
    with pytest.raises(ValueError, match=f"ADAPT explicit verification needs about {need} bytes"):
        result.verify(checks=options.revise(max_bytes=need // 2))
    assert checks == [need]


@pytest.mark.parametrize(("comparisons", "outer"), [(("sector",), 1_409_028), (("residual", "sector"), 1_474_564)])
def test_twelve_qubit_sector_verification_count_is_its_threshold(comparisons, outer, monkeypatch):
    """The verification count of a 12-qubit vacuum Result is its threshold.

    Classical ADAPT for H = -Z_0 from |0...0> with pool i Y_0 and one
    iteration. Sector verification needs 1,409,028 bytes, or 1,474,564
    with the residual still held, and passes at that limit with the spin
    action's own count of 809,380 bytes, also when a standalone 2-qubit
    spin action runs between the verification count and the Ritz state.
    Its sector facts are those of the vacuum. Half that limit refuses at
    the verification count, with no spin check.
    """
    from nwqlib import Eigenproblem, plan, solve
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation
    from nwqlib.subroutines import fermionic_pool

    q = 12
    method = ADAPT(initial_state=ingest_occupation("0" * q, num_qubits=q),
                   pool=(ingest_pauli((("I" * (q - 1) + "Y", 1j),), num_qubits=q),), max_iterations=1)
    result = solve(plan(Eigenproblem(A=ingest_pauli((("I" * (q - 1) + "Z", -1.0),), num_qubits=q)), method=method,
                        execution="classical", seed=9), progress=False)
    checks = []

    def spy(module, operation):
        real = module._check_bytes

        def counted(size, max_bytes, name="input"):
            if name == operation:
                checks.append((operation, size))
            return real(size, max_bytes, name)
        monkeypatch.setattr(module, "_check_bytes", counted)

    spy(adapt_verification, "ADAPT explicit verification")
    spy(fermionic_pool, "spin-squared action")
    small = fermionic_pool._spin_squared_requirements(2)[0]
    ritz = adapt_verification._ritz_state

    def interleaved(*args, **kwargs):
        assert np.array_equal(fermionic_pool.apply_spin_squared(np.array([0., 1., 0., 0.]), num_qubits=2),
                              [0., .75, 0., 0.])
        return ritz(*args, **kwargs)

    options = AdaptVerificationOptions(name="sector", comparisons=comparisons)
    with monkeypatch.context() as patch:
        patch.setattr(adapt_verification, "_ritz_state", interleaved)
        _, facts = result.verify(checks=options.revise(max_bytes=outer))
    assert checks == [("ADAPT explicit verification", outer), ("spin-squared action", small),
                      ("spin-squared action", 809_380)]
    assert {k: values(facts)[k] for k in ("particle_number", "spin_z", "spin_squared")} == dict.fromkeys(
        ("particle_number", "spin_z", "spin_squared"), 0.0)
    del checks[:]
    with pytest.raises(ValueError, match=f"ADAPT explicit verification needs about {outer} bytes"):
        result.verify(checks=options.revise(max_bytes=outer // 2))
    assert checks == [("ADAPT explicit verification", outer)]


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
