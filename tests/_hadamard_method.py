"""The test suite's reference external Method: a Hadamard test built from
NWQLib's existing selected blocks.

For A=cP, signed SELECT applies sign(c)P. H-controlled-SELECT-H on the
ancilla therefore has mean z=sign(c)<psi|P|psi>. Multiplying by |c| gives
the original normalized expectation. No reference state or eigensolve runs.
Its inert registration is declared in ``_hadamard_method_metadata``.
"""

from typing import ClassVar, Literal

from nwqlib.algorithms.protocol import Method, AlgorithmDescriptor, ApplicabilityError
from nwqlib.blocks import (
    SelectedConstruction,
    select_preparation,
    select_signed_pauli,
    transform_block,
)
from nwqlib.core.records import ContentID, PositiveInt, Real, Record, Text
from nwqlib.core.analysis import Result, capture_analysis_origin
from nwqlib.core.planning import Plan, Experiment, ObservationSpec
from nwqlib.evidence import ErrorModel, ErrorTerm, Fact, FramedFact
from nwqlib.execution import PauliValue
from nwqlib.ir import (
    Allocate,
    BlockCall,
    ClassicalValue,
    Definition,
    Measure,
    PortMap,
    Program,
    Register,
    Sequence,
)
from nwqlib.problems import Expectation, NormalizedExpectation
from nwqlib.problems.inputs import ingest_vector
from _hadamard_method_metadata import DESCRIPTOR, SOURCE


class Reconstruction(Record):
    state_id: ContentID
    observable_id: ContentID
    alpha: Real
    system_qubits: PositiveInt
    ancilla_label: Text


def _selected_relation(plan):
    """Check the selected operands and measured wire without rebuilding gates."""
    rec = plan.reconstruction
    if (
        not isinstance(plan.problem, Expectation)
        or not isinstance(rec, Reconstruction)
        or len(plan.construction.selections) != 4
    ):
        raise ValueError("Hadamard selection requires its four actual subroutines")
    # Check the selected operands and full controlled/adjoint relations first.
    # A matching result type alone cannot establish the Hadamard identity.
    system, h, action, inverse_h = plan.construction.selections
    if (
        system.semantics.input != plan.problem.state.manifest.reference
        or action.semantics.input != plan.problem.observable.manifest.reference
        or not action.controlled
        or action.semantics.base_semantics is None
        or action.semantics.base_semantics.alpha != rec.alpha
        or h.decomposition is None
        or tuple((p.gate, p.qubits) for p in h.decomposition) != (("h", (0,)),)
        or not inverse_h.adjoint
        or inverse_h.semantics.input != h.semantics.input
        or rec.ancilla_label != "I" * rec.system_qubits + "Z"
    ):
        raise ValueError("Hadamard selection changed its physical input, control or phase relation")
    # Check the actual ordered call body and ports, including the measured
    # ancilla. Metadata describing a different wire would change the observable.
    program = plan.construction.program
    if tuple((r.name, r.width) for r in program.registers) != (
        ("ancilla", 1),
        ("system", rec.system_qubits),
    ):
        raise ValueError("Hadamard register order must preserve the actual ancilla wire")
    nodes = {d.id: d.node for d in program.definitions}
    root = nodes[program.root]
    if not isinstance(root, Sequence):
        raise ValueError("Hadamard selection requires its ordered subroutine body")
    calls = tuple(
        (nodes[name].signature, tuple((p.port, p.wire) for p in nodes[name].ports))
        for name in root.children
        if isinstance(nodes[name], BlockCall)
    )
    expected = (
        ("system_prep", (("system", "system"),)),
        ("h", (("system", "ancilla"),)),
        ("controlled_pauli", (("control", "ancilla"), ("system", "system"))),
        ("inverse_h", (("system", "ancilla"),)),
    )
    if calls != expected:
        raise ValueError(
            "Hadamard calls must preserve PREP, H, controlled-P, inverse-H and their actual wires"
        )


def _ancilla_mean(data, *, plan_id, label, kind):
    """Read the one actually collected ancilla statistic, preserving its source."""
    if data.trace.plan_id != plan_id or len(data.observations.chunks) != 1:
        raise ValueError("Hadamard analysis requires its one actual Plan acquisition")
    (chunk,) = data.observations.chunks
    data.trace.validate_observation(chunk)
    if (
        chunk.plan_id != plan_id
        or chunk.run_id != data.trace.run_id
        or chunk.experiment != "hadamard"
        or chunk.population != "unconditional"
        or chunk.observation.kind != kind
    ):
        raise ValueError("Hadamard observation differs from its selected acquisition")
    # Join the observation to its completed event and original preparation.
    # Only that acquired population may contribute to this reconstruction.
    receipt = next((r for r in data.receipts if r.content_id == chunk.prepared_id), None)
    event = next((e for e in data.trace.events if e.attempt == chunk.attempt), None)
    if (
        receipt is None
        or event is None
        or event.status != "completed"
        or event.observation_id != chunk.content_id
        or receipt.realization_id != chunk.realization_id
        or receipt.observation != chunk.observation
    ):
        raise ValueError("Hadamard statistic has no matching completed preparation/readout")
    # Exact readout provides ancilla Z directly. Counts encode Z eigenvalues
    # as +1 for zero and -1 for one, averaged over returned shots.
    if kind == "pauli_expectation":
        if chunk.observation.labels != (label,) or len(chunk.values) != 1:
            raise ValueError("Hadamard exact statistic must measure the actual ancilla Z")
        (value,) = chunk.values
        if not isinstance(value, PauliValue) or value.label != label:
            raise ValueError("Hadamard Pauli value measures a different wire")
        mean = value.value
    else:
        histogram = chunk.histogram() if kind == "counts" else None
        if (
            not chunk.returned_shots
            or histogram is None
            or histogram.width != 1
            or sum(histogram.weights.tolist()) != chunk.returned_shots
        ):
            raise ValueError("Hadamard counts require a nonempty actual single-ancilla population")
        mean = (
            sum((1 if index == 0 else -1) * count
                for index, count in zip(histogram.index_list(), histogram.weights.tolist(), strict=True))
            / chunk.returned_shots
        )
    return mean, chunk


class HadamardExpectationResult(Result):
    """The original normalized expectation, reconstructed from ancilla Z."""

    value: Real
    ancilla_mean: Real
    alpha: Real
    ancilla_label: Text
    readout: Literal["pauli_expectation", "counts"]

    def _summary_lines(self):
        return (
            f"Normalized expectation: {self._scalar_text(self.value)}{self._unit_text()}",
            "Hadamard ancilla estimates Re<psi|P|psi>; one selected Pauli term",
        )

    def validate_plan(self, plan):
        self._validate_common_plan(plan, Plan)
        _selected_relation(plan)
        rec = plan.reconstruction
        if not isinstance(plan.method, HadamardPauliExpectation) or not isinstance(
            rec, Reconstruction
        ):
            raise ValueError("Hadamard result requires its actual Method reconstruction")
        if (
            self.construction_id != plan.construction.content_id
            or rec.state_id != plan.problem.state.manifest.content_id
            or rec.observable_id != plan.problem.observable.manifest.content_id
            or self.alpha != rec.alpha
            or self.ancilla_label != rec.ancilla_label
            or self.readout != plan.experiments[0].observation.kind
            or self.value != self.alpha * self.ancilla_mean
        ):
            raise ValueError(
                "Hadamard result violates its original input/ancilla reconstruction relation"
            )

    def validate_data(self, data):
        mean, chunk = _ancilla_mean(
            data, plan_id=self.plan_id, label=self.ancilla_label, kind=self.readout
        )
        if self.contribution_ids != (chunk.content_id,) or self.ancilla_mean != mean:
            raise ValueError("Hadamard result differs from the actual acquired ancilla statistic")


class HadamardPauliExpectation(Method):
    """One real nonzero Pauli term, using normalized preparation and one ancilla."""

    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR
    result_type: ClassVar[type[Result]] = HadamardExpectationResult
    preparation_choice: Literal["native", "hzh"] = "native"
    max_bytes: PositiveInt = 64 * 1024**2

    def validate_point(self, plan, experiment, values):
        _selected_relation(plan)
        observation = experiment.observation
        if observation is None or observation.kind not in {"pauli_expectation", "counts"}:
            raise ValueError("Hadamard selection requires its direct ancilla observation")
        if observation.kind == "pauli_expectation" and observation.labels != (
            plan.reconstruction.ancilla_label,
        ):
            raise ValueError("Hadamard readout must measure the actual ancilla Z")
        if observation.kind == "counts":
            measured = tuple(
                (d.node.wire, d.node.result)
                for d in plan.construction.program.definitions
                if isinstance(d.node, Measure)
            )
            if measured != (("ancilla", "ancilla_out"),):
                raise ValueError("Hadamard counts must measure the actual ancilla wire")

    def plan(self, problem, *, output, execution, shots, rng):
        if not isinstance(problem, Expectation) or not isinstance(output, NormalizedExpectation):
            raise ApplicabilityError(
                "HadamardPauliExpectation requires Expectation and NormalizedExpectation"
            )
        if execution != "quantum":
            raise ApplicabilityError("this external Hadamard Method executes its selected circuit")
        # Admit the mathematical domain before selecting any circuit blocks.
        # The example supports one real nonzero Pauli coefficient and matching state.
        observable = problem.observable
        if observable.structure != "hermitian" or "pauli_terms" not in observable.manifest.access:
            raise ApplicabilityError("Hadamard example requires explicit Hermitian Pauli input")
        rows = tuple(observable.pauli_terms().labels())
        if len(rows) != 1 or rows[0][1].imag != 0 or rows[0][1].real == 0:
            raise ApplicabilityError(
                "Hadamard example supports exactly one nonzero real Pauli term"
            )
        q = len(rows[0][0])
        if not q or problem.state.manifest.basis.dimension != 1 << q:
            raise ApplicabilityError("Hadamard system preparation and Pauli dimensions must agree")
        if (
            output.observable is not None
            and output.observable.manifest.content_id != observable.manifest.content_id
        ):
            raise ApplicabilityError("Hadamard example reads the Problem's actual Pauli observable")
        if shots is not None and (type(shots) is not int or shots < 1):
            raise ValueError("shots must be a positive integer or None")
        # Reuse native block selection: prepare psi, create |+> on the ancilla,
        # control sign(c)P, and undo H. Its ancilla mean contains the coefficient sign.
        system = select_preparation("system_prep", problem.state, choice=self.preparation_choice)
        h = select_preparation("h", ingest_vector((1.0, 1.0), max_bytes=self.max_bytes))
        action = transform_block(
            "controlled_pauli",
            select_signed_pauli("pauli", observable, max_bytes=self.max_bytes),
            control=True,
        )
        inverse_h = transform_block("inverse_h", h, adjoint=True)
        blocks = (system, h, action, inverse_h)
        # Describe the same physical sequence in the shared Program. Explicit
        # port maps attach the control to the ancilla and the Pauli to the system.
        nodes = {
            "allocate_ancilla": Allocate(wire="ancilla"),
            "allocate_system": Allocate(wire="system"),
            "system_prep": BlockCall(
                signature="system_prep", ports=(PortMap(port="system", wire="system"),)
            ),
            "h": BlockCall(signature="h", ports=(PortMap(port="system", wire="ancilla"),)),
            "controlled_pauli": BlockCall(
                signature="controlled_pauli",
                ports=(
                    PortMap(port="control", wire="ancilla"),
                    PortMap(port="system", wire="system"),
                ),
            ),
            "inverse_h": BlockCall(
                signature="inverse_h", ports=(PortMap(port="system", wire="ancilla"),)
            ),
        }
        # Measured counts need a classical output wire. Exact Pauli readout
        # queries the same ancilla observable without adding a measurement node.
        if shots is not None:
            nodes["measure"] = Measure(wire="ancilla", result="ancilla_out")
        nodes["root"] = Sequence(children=tuple(nodes))
        program = Program(
            root="root",
            registers=(
                Register(name="ancilla", width=1, role="clean_ancilla"),
                Register(name="system", width=q),
            ),
            signatures=tuple(b.record.signature for b in blocks),
            classical=()
            if shots is None
            else (ClassicalValue(name="ancilla_out", dtype="bits", width=1),),
            definitions=tuple(Definition(id=name, node=node) for name, node in nodes.items()),
        )
        construction = SelectedConstruction(
            program=program, selections=tuple(b.record for b in blocks)
        )
        label = "I" * q + "Z"  # q0 is the first declared ancilla, at the right of the Pauli label.
        spec = (
            ObservationSpec(kind="pauli_expectation", labels=(label,))
            if shots is None
            else ObservationSpec(kind="counts", shots=shots)
        )
        rec = Reconstruction(
            state_id=problem.state.manifest.content_id,
            observable_id=observable.manifest.content_id,
            alpha=abs(rows[0][1].real),
            system_qubits=q,
            ancilla_label=label,
        )
        # Declare unestablished error contributions in the requested output frame.
        # A correct algebraic Hadamard identity is not a backend error certificate.
        frame = output.frame(problem)
        names = ("native_preparation", "native_simulation", "sampling", "physical_model")
        model = ErrorModel(
            output_id=output.content_id,
            subject_id=problem.content_id,
            construction_id=construction.content_id,
            frame=frame,
            required_sources=names,
            source=SOURCE,
            terms=tuple(
                ErrorTerm(
                    name=name,
                    stage="execution",
                    source=SOURCE,
                    formula="unestablished absolute error in normalized expectation",
                    coverage=name,
                    fact=FramedFact(
                        frame=frame,
                        bindings=(),
                        fact=Fact(
                            quantity=name,
                            unit=frame.unit,
                            scope=frame.scope,
                            availability="unknown",
                            reason="no bound supplied for " + name,
                        ),
                    ),
                )
                for name in names
            ),
        )
        return Plan(
            problem=problem,
            method=self,
            output=output,
            execution=execution,
            shots=shots,
            randomness=rng.snapshot(),
            construction=construction,
            reconstruction=rec,
            error_model=model,
            experiments=(Experiment(name="hadamard", setting="ancilla_z", observation=spec),),
        )._bind(blocks=blocks)

    def analyze(self, plan, data, *, settings):
        if settings:
            raise ValueError("this Hadamard reconstruction has no post-hoc settings")
        rec = plan.reconstruction
        mean, chunk = _ancilla_mean(
            data,
            plan_id=plan.content_id,
            label=rec.ancilla_label,
            kind=plan.experiments[0].observation.kind,
        )
        # Multiply the signed ancilla mean by |c| exactly once to recover
        # the original normalized expectation, preserving its observation identity.
        result = HadamardExpectationResult(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=(chunk.content_id,),
            value=rec.alpha * mean,
            ancilla_mean=mean,
            alpha=rec.alpha,
            ancilla_label=rec.ancilla_label,
            readout=chunk.observation.kind,
            origin=capture_analysis_origin(analyzer=SOURCE, method_id=self.content_id),
        )
        result.validate_data(data)
        return result._attach(plan, data)

    def save_archive(self, plan, files):
        from nwqlib.blocks._archive import write_blocks

        return dict(
            plan=plan.to_record(),
            problem=files.write_problem(plan.problem),
            output=files.write_output(plan.output),
            blocks=write_blocks(plan.blocks, files),
        )

    @classmethod
    def load_archive(cls, saved, files):
        from nwqlib.blocks._archive import read_blocks

        fields = saved["plan"]
        method = cls.model_validate(fields["method"]["fields"])
        plan = files.read_plan(
            fields,
            problem=files.read_problem(saved["problem"]),
            method=method,
            output=files.read_output(saved["output"]),
            reconstruction=Reconstruction.model_validate(fields["reconstruction"]),
        )
        return plan._bind(blocks=read_blocks(saved["blocks"], plan.construction.selections, files))


def case():
    """A two-qubit |+i>,Y witness; sign errors cannot satisfy its exact value one."""
    from nwqlib import prepare, submit
    from nwqlib.algorithms.authoring import MethodCase
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli

    def evaluate(plan):
        with submit(prepare(plan, limits=ExecutionLimits(max_total_circuits=1))) as run:
            return run.wait(timeout=10)

    return MethodCase(
        method=HadamardPauliExpectation(),
        problem=Expectation(state=(1.0, 1j), observable=ingest_pauli((("Y", 1.0),), num_qubits=1)),
        evaluate=evaluate,
        accepts=lambda result: abs(result.value - 1.0) < 1e-12,
        invalid_result=lambda result: result.revise(value=-1.0),
        seed=7,
    )
