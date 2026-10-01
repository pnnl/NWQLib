#!/usr/bin/env python3
"""Fresh-process public core-operation witness, including caught import attempts."""

from __future__ import annotations

import argparse

import builtins
import importlib.abc
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BLOCKED = (
    "qiskit",
    "qiskit_aer",
    "qiskit_ibm_runtime",
    "qiskit_ibm_provider",
    "qiskit_ionq",
    "qiskit_braket_provider",
    "braket",
    "boto3",
    "botocore",
    "azure",
    "pytket",
    "qnexus",
    "cirq",
    "pennylane",
    "pyquil",
    "qulacs",
    "cudaq",
    "nwqsim",
    "nwqec",
    "qdk",
    "nwqlib.algorithms",
    "nwqlib.backends",
    "nwqlib.execution",
)


# The construction helpers import inside their bodies, so the audit guard
# installed by child() sees every import they make.


def scientific_question():
    """Ask for <psi|O|psi>/<psi|psi> with the physical amplitudes (3, 4i), whose norm is 5.

    The explicit classical action of O = X - 2Z on |0> is (-2, 1). No quantum oracle is inferred.
    """
    import numpy as np
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import Expectation, state_input

    observable = ingest_pauli([("X", 1), ("Z", -2)], num_qubits=1)
    problem = Expectation(state=state_input([3, 4j]), observable=observable)
    return problem, observable.matvec(np.array([1.0, 0.0]))


def shared_program():
    """Two angle settings share one kept rotation body repeated a symbolic n times."""
    from nwqlib.core import Float64, InputRef, Source
    from nwqlib.ir import (
        Allocate, Argument, Binary, Binding, BlockCall, BlockSignature, ClassicalValue, Constant,
        Definition, ExprRef, Expression, Measure, MeasurementBatch, MetadataRef, Parameter,
        ParameterRef, PortMap, Program, QuantumPort, Register, Release, Repeat, Sequence, Setting,
    )

    source = Source(
        name="audit rotation", version="1", domain="declared unitary", reference="audit:rotation"
    )
    metadata = MetadataRef(
        format=Source(
            name="audit readout",
            version="1",
            domain="setting metadata",
            reference="audit:readout-schema",
        ),
        data=InputRef(identity="audit:readout-z/1", representation="audit readout v1", source=source),
    )
    return Program(
        root="batch",
        parameters=(
            Parameter(name="n", domain="integer", lower=0),
            Parameter(name="theta", domain="real"),
        ),
        # The constraint 0 <= n rejects negative counts without expanding the body.
        expressions=(
            Expression(id="n", value=ParameterRef(parameter="n")),
            Expression(id="theta", value=ParameterRef(parameter="theta")),
            Expression(id="zero", value=Constant(value=0)),
            Expression(id="valid", value=Binary(op="le", left="zero", right="n")),
        ),
        constraints=(ExprRef(expression="valid"),),
        registers=(Register(name="q", width=1),),
        classical=(ClassicalValue(name="bits", dtype="bits", width=1),),
        signatures=(
            BlockSignature(
                name="rotation",
                target=source,
                quantum=(QuantumPort(name="target", width=1, requires="coherent"),),
                parameters=(Parameter(name="theta", domain="real"),),
                obligations=("The selected implementation must implement its declared rotation.",),
            ),
        ),
        definitions=(
            Definition(id="allocate", node=Allocate(wire="q")),
            Definition(
                id="call",
                node=BlockCall(
                    signature="rotation",
                    ports=(PortMap(port="target", wire="q"),),
                    arguments=(Argument(parameter="theta", value=ExprRef(expression="theta")),),
                ),
            ),
            Definition(id="repeat", node=Repeat(body="call", count=ExprRef(expression="n"))),
            Definition(id="measure", node=Measure(wire="q", result="bits")),
            Definition(id="release", node=Release(wire="q")),
            Definition(
                id="experiment",
                node=Sequence(children=("allocate", "repeat", "call", "measure", "release")),
            ),
            Definition(
                id="batch",
                node=MeasurementBatch(
                    body="experiment",
                    settings=tuple(
                        Setting(
                            label="Z",
                            bindings=(Binding(parameter="theta", value=Float64(value=x)),),
                            metadata=metadata,
                        )
                        for x in (0.0, 0.5)
                    ),
                ),
            ),
        ),
    )


def preparation(choice="native"):
    """Prepare |1> on one allocated wire with the selected X or HZH recipe."""
    from nwqlib.blocks import SelectedConstruction, select_preparation
    from nwqlib.ir import Allocate, BlockCall, Definition, PortMap, Program, Register, Sequence
    from nwqlib.problems import ingest_occupation

    block = select_preparation("prep", ingest_occupation("1", num_qubits=1), choice=choice)
    program = Program(
        root="root",
        registers=(Register(name="system", width=1),),
        signatures=(block.record.signature,),
        definitions=(
            Definition(id="allocate", node=Allocate(wire="system")),
            Definition(
                id="call",
                node=BlockCall(signature="prep", ports=(PortMap(port="system", wire="system"),)),
            ),
            Definition(id="root", node=Sequence(children=("allocate", "call"))),
        ),
    )
    return SelectedConstruction(program=program, selections=(block.record,)), (block,)


def pauli_encoding():
    """PREP-SELECT-unPREP for A = -2X + Y + 3I, whose projected block is A/6."""
    from nwqlib.blocks import (
        PauliEncoding, SelectedConstruction, select_pauli_preparation, select_signed_pauli,
        transform_block,
    )
    from nwqlib.ir import Allocate, BlockCall, Definition, PortMap, Program, Register, Sequence
    from nwqlib.operators import ingest_pauli

    select = select_signed_pauli("select", ingest_pauli([("X", -2), ("Y", 1), ("I", 3)], num_qubits=1))
    prep = select_pauli_preparation("prep", select)
    blocks = (prep, select, transform_block("unprep", prep, adjoint=True))
    # Two index qubits address three terms plus padding, next to one system qubit.
    program = Program(
        root="root",
        registers=(Register(name="index", width=2), Register(name="system", width=1)),
        signatures=tuple(block.record.signature for block in blocks),
        definitions=(
            Definition(id="index", node=Allocate(wire="index")),
            Definition(id="system", node=Allocate(wire="system")),
            Definition(
                id="prep",
                node=BlockCall(signature="prep", ports=(PortMap(port="system", wire="index"),)),
            ),
            Definition(
                id="select",
                node=BlockCall(
                    signature="select",
                    ports=(
                        PortMap(port="index", wire="index"),
                        PortMap(port="system", wire="system"),
                    ),
                ),
            ),
            Definition(
                id="unprep",
                node=BlockCall(signature="unprep", ports=(PortMap(port="system", wire="index"),)),
            ),
            Definition(id="encoding", node=Sequence(children=("prep", "select", "unprep"))),
            Definition(id="root", node=Sequence(children=("index", "system", "encoding"))),
        ),
    )
    construction = SelectedConstruction(
        program=program,
        selections=tuple(block.record for block in blocks),
        encodings=(
            PauliEncoding(root="encoding", select="select", prepare="prep", unprepare="unprep"),
        ),
    )
    return construction, blocks


def child(*, poison: bool) -> None:
    """Block import attempts before importing any NWQLib module."""
    attempts = []
    # Selected-block construction must also stay outside the native subroutine
    # layer. The prepared-record audit shares BLOCKED and imports that layer.
    blocked = BLOCKED + ("nwqlib.subroutines",)

    def forbidden(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)

    def record_attempt(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError(f"blocked SDK/execution import attempt: {name}")

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        record_attempt(name)
        for member in fromlist or ():
            record_attempt(name + "." + member)
        return original_import(name, globals, locals, fromlist, level)

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            record_attempt(fullname)
            return None

    builtins.__import__ = guarded_import
    sys.meta_path.insert(0, Blocker())
    if poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass

    import json

    import jsonschema
    import nwqlib
    from nwqlib import core, evidence, problems
    from nwqlib.ir import Definition, MeasurementBatch, Program, Sequence
    from nwqlib.blocks import SelectedConstruction, select_pauli_readout
    from nwqlib.resources import WorkloadEstimate, estimate

    assert Path(nwqlib.__file__).resolve() == ROOT / "src/nwqlib/__init__.py"
    question, action = scientific_question()
    assert tuple(action) == (-2.0, 1.0)  # The actual Pauli input X-2Z acts on |0>.
    assert question.state.preparation.physical_scale.as_float() == 5.0
    import sympy as sp
    x = sp.Symbol("x", real=True)
    # Ge(x, 1/2) is stored as the inequality residual 1/2 - x <= 0.
    constrained = problems.ConstrainedOptimization(
        objective=x**2, variables=(x,), bounds=((-1.0, 1.0),), inequalities=(sp.Ge(x, sp.Rational(1, 2)),))
    assert constrained.inequalities == (sp.Rational(1, 2) - x,)
    objects = (
        question,
        problems.Eigenproblem(A=question.observable),
        problems.LinearDynamics(A=question.observable, initial_state=question.state, time=0.25),
        constrained,
        problems.Accuracy(absolute_tolerance=0.01, component="sampling"),
        problems.Solution(),
        problems.StateVector(global_phase="modulo_global_phase"),
    )
    for original in objects:
        restored = type(original).model_validate_json(original.model_dump_json())
        assert restored.content_id == original.content_id
        schema = json.loads(json.dumps(type(original).model_json_schema(mode="serialization")))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(original.model_dump(mode="json"), schema)
    restored = problems.Expectation.model_validate_json(question.model_dump_json())
    try:
        restored.state.entry(0)
    except ValueError as error:
        assert "unavailable" in str(error)
    else:
        raise AssertionError("descriptive JSON invented executable state data")
    assert core.Rational(numerator=1, denominator=3).denominator == 3
    assert evidence.Fact is not None
    symbolic = shared_program()
    assert not symbolic.check_readiness().ready
    program = symbolic.bind(n=10**12)
    restored_program = Program.model_validate_json(program.model_dump_json())
    assert restored_program == program
    restored_program.check_readiness().require_ready()
    assert len(tuple(restored_program.iter_definitions())) == 7
    program_schema = Program.model_json_schema(mode="serialization")
    jsonschema.Draft202012Validator.check_schema(program_schema)
    jsonschema.validate(program.model_dump(mode="json"), program_schema)
    workloads = tuple(estimate(preparation(choice)[0]) for choice in ("native", "hzh"))
    for workload in workloads:
        restored_workload = WorkloadEstimate.model_validate_json(workload.model_dump_json())
        assert restored_workload == workload
        memory = restored_workload.quantity("memory", location="logical_device").fact
        assert memory.availability == "unknown" and memory.value is None
    assert [item.quantity("operations").fact.value.numerator for item in workloads] == [1, 3]
    batch = MeasurementBatch(
        body="empty",
        settings=(program.definitions[-1].node.settings[0],),
        repetitions=4,
        observation_kind="probabilities",
    )
    acquisition = Program(
        root="batch",
        definitions=(Definition(id="empty", node=Sequence()), Definition(id="batch", node=batch)),
        parameters=program.parameters,
    )
    exact = estimate(SelectedConstruction(program=acquisition, selections=()))
    assert exact.quantity("shots").fact.value.numerator == 0
    assert exact.quantity("exact_evaluations").fact.value.numerator == 4
    assert WorkloadEstimate.model_validate_json(exact.model_dump_json()) == exact
    resource_schema = WorkloadEstimate.model_json_schema(mode="serialization")
    jsonschema.Draft202012Validator.check_schema(resource_schema)
    jsonschema.validate(workloads[0].model_dump(mode="json"), resource_schema)
    for construction, _ in (preparation(), preparation("hzh"), pauli_encoding()):
        restored = SelectedConstruction.model_validate_json(construction.model_dump_json())
        assert restored == construction
    _, pauli_blocks = pauli_encoding()
    select_pauli_readout("readout", pauli_blocks[1]).record.model_dump_json()
    loaded = [name for name in sys.modules if forbidden(name)]
    if attempts or loaded:
        raise AssertionError(f"SDK isolation violation: attempts={attempts}, loaded={loaded}")
    print(f"SDK-free scientific input/action/schema/JSON passed; identity={question.content_id}")
    print(
        f"SDK-free Program construction/binding/readiness/JSON passed; identity={program.content_id}"
    )
    print(
        "SDK-free selected resource operation/schema/JSON passed; X=1, HZH=3, unknown workspace preserved"
    )
    print("SDK-free selected-block construction JSON and Pauli readout selection passed")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--child", action="store_true", help="run the isolated audit directly")
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; requires --child")
    args = parser.parse_args(argv)
    if args.poison and not args.child:
        parser.error("--poison requires --child")
    if args.child:
        child(poison=args.poison)
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    clean = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if clean.returncode:
        raise RuntimeError(clean.stdout + clean.stderr)
    print(clean.stdout, end="")
    poisoned = subprocess.run(
        command + ["--poison"], cwd=ROOT, env=env, text=True, capture_output=True
    )
    if poisoned.returncode == 0 or "attempts=['qiskit']" not in poisoned.stderr:
        raise AssertionError(
            "caught-import control failed to falsify isolation:\n" + poisoned.stderr
        )
    print("Caught blocked import control rejected as required.")


if __name__ == "__main__":
    main()
