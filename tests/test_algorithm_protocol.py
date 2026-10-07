"""Actual external Method science, selected bindings and explicit registration."""

from importlib.metadata import EntryPoint
from pathlib import Path
import subprocess
import sys

import pytest

from _hadamard_method import HadamardPauliExpectation
from _hadamard_method_metadata import DESCRIPTOR, REGISTRATION
from nwqlib import Expectation, plan, prepare, submit
from nwqlib.algorithms import (
    AlgorithmRegistry,
    Registration,
    direct_method,
    third_party_registrations,
)
from nwqlib.operators import ingest_pauli


def selected(state, label, *, coefficient=1.0, method=None, shots=None):
    return plan(
        Expectation(
            state=state, observable=ingest_pauli(((label, coefficient),), num_qubits=len(label))
        ),
        method=method or HadamardPauliExpectation(),
        shots=shots,
        seed=7,
    )


def execute(chosen):
    assert sum(r.width for r in chosen.construction.program.registers) <= 2
    with submit(prepare(chosen)) as run:
        result = run.wait(timeout=10)
        assert run.trace.jobs == 1 and len(result.contribution_ids) == 1
        return result


def test_external_hadamard_three_independent_states_and_wrong_pair():
    # Z|0>=|0>, <+|Z|+>=0, and Y(1,i)^T=(1,i)^T. The last
    # equality changes sign if controlled Y loses its complex phase.
    results = [
        execute(selected(state, pauli))
        for state, pauli in (((1.0, 0.0), "Z"), ((1.0, 1.0), "Z"), ((1.0, 1j), "Y"))
    ]
    assert [r.value for r in results] == pytest.approx([1.0, 0.0, 1.0], rel=0, abs=2e-12)
    for result in results:
        assert result.ancilla_label == "IZ"
        with pytest.raises(ValueError, match="reconstruction relation"):
            result.revise(value=0.5).validate_plan(result.plan)
    with pytest.raises(ValueError, match="actual Plan acquisition"):
        results[0].plan.method.analyze(results[0].plan, results[1].data, settings={})
    # The original legal Result still supports a declared reanalysis.
    assert results[0].analyze().value == pytest.approx(1.0, rel=0, abs=2e-12)


def test_actual_alternative_prep_counts_and_controlled_negative_phase():
    from nwqlib.problems.inputs import ingest_occupation

    state = ingest_occupation("1", num_qubits=1)
    chosen = selected(
        state, "Z", method=HadamardPauliExpectation(preparation_choice="hzh"), shots=8
    )
    assert tuple(p.gate for p in chosen.construction.selections[0].decomposition) == ("h", "z", "h")
    result = execute(chosen)
    assert result.value == -1.0  # Every ancilla shot is one for the Z eigenstate |1>.
    assert result.data.observations.chunks[0].returned_shots == 8
    negative = execute(selected((1.0, 1j), "Y", coefficient=-2.0))
    assert negative.value == pytest.approx(-2.0, rel=0, abs=4e-12)


def test_admission_rejects_unsupported_science_and_wrong_actual_wires():
    """Swap equal-width ports and the measured Pauli label to catch semantic errors that shape
    checks miss.
    """
    from nwqlib.algorithms import ApplicabilityError
    from nwqlib.ir import PortMap

    method = HadamardPauliExpectation()
    for observable in (
        ingest_pauli((("X", 1.0), ("Z", 1.0)), num_qubits=1),
        [[1.0, 0.0], [0.0, -1.0]],
    ):
        with pytest.raises(ApplicabilityError, match="Pauli"):
            plan(Expectation(state=(1.0, 0.0), observable=observable), method=method)
    original = selected((1.0, 1j), "Y")
    program = original.construction.program
    definitions = tuple(
        d.revise(
            node=d.node.revise(
                ports=(
                    PortMap(port="control", wire="system"),
                    PortMap(port="system", wire="ancilla"),
                )
            )
        )
        if d.id == "controlled_pauli"
        else d
        for d in program.definitions
    )
    wrong = original.revise(
        construction=original.construction.revise(program=program.revise(definitions=definitions))
    )
    with pytest.raises(ValueError, match="actual wires"):
        wrong.resolve("hadamard")
    observation = original.experiments[0].observation.revise(labels=("ZI",))
    wrong = original.revise(experiments=(original.experiments[0].revise(observation=observation),))
    with pytest.raises(ValueError, match="actual ancilla"):
        wrong.resolve("hadamard")
    assert original.resolve("hadamard").resolved_observation(original)[1].labels == ("IZ",)


def test_registry_loads_only_exact_selected_factory_and_never_source_paths(monkeypatch):
    import _hadamard_method as implementation

    registry = AlgorithmRegistry((REGISTRATION,))
    calls = []

    def factory(**configuration):
        calls.append(configuration)
        return HadamardPauliExpectation(**configuration)

    monkeypatch.setattr(implementation, "HadamardPauliExpectation", factory)
    assert AlgorithmRegistry().discover() == () and registry.discover() == (REGISTRATION,)
    assert not calls
    actual = registry.resolve(
        REGISTRATION.source, expected_type=HadamardPauliExpectation, preparation_choice="hzh"
    )
    assert actual.preparation_choice == "hzh" and calls == [{"preparation_choice": "hzh"}]
    assert direct_method(actual) is actual
    for source, error in (
        (REGISTRATION.source.revise(version="missing"), LookupError),
        (REGISTRATION.source.revise(reference="os:system"), ValueError),
    ):
        with pytest.raises(error):
            registry.resolve(source)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="duplicate"):
        AlgorithmRegistry((REGISTRATION, REGISTRATION))
    mismatch = Registration(
        source=REGISTRATION.source,
        factory=REGISTRATION.factory,
        descriptor=DESCRIPTOR.revise(maintenance="different description"),
    )
    with pytest.raises(ValueError, match="descriptor differs"):
        AlgorithmRegistry((mismatch,)).resolve(REGISTRATION.source)


def test_external_metadata_has_no_implicit_discovery_or_schema_execution(monkeypatch):
    import nwqlib.algorithms.registry as owner

    installed = (EntryPoint(name="not_installed@2", value="absent:factory", group="nwqlib.algorithms"),)
    monkeypatch.setattr(owner, "entry_points", lambda *, group: installed if group == "nwqlib.algorithms" else ())
    rows = third_party_registrations()
    registry = AlgorithmRegistry(rows)
    monkeypatch.setattr(
        owner, "import_module", lambda *args: pytest.fail("inventory imported a Method module")
    )
    assert registry.discover() == rows
    assert owner.algorithm_card(rows[0])["descriptor"] is None
    with pytest.raises(ValueError, match="explicitly select"):
        owner.options_schema(rows[0])
    for entries in (
        (EntryPoint(name="unversioned", value="absent:factory", group="nwqlib.algorithms"),),
    ):
        monkeypatch.setattr(owner, "entry_points", lambda *, group, entries=entries: entries)
        with pytest.raises(ValueError, match="method@version"):
            third_party_registrations()


def test_fresh_protocol_import_attempt_audit():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "docs/scripts/check_algorithm_protocol.py"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
