"""Actual acquisition association and host declarations for eigenvalue methods."""

from nwqlib.blocks import SelectedConstruction
from nwqlib.blocks.records import SelectedKernel
from nwqlib.core.planning import Experiment, ObservationSpec, Realization
from nwqlib.ir import ClassicalStage, Definition, Program
from nwqlib.ir.validation import admitted_program
from nwqlib.evidence.error_model import ErrorModel, ErrorTerm, FramedFact
from nwqlib.evidence.records import Fact


def host_construction(source, inputs, labels, *, work, description, frames=None, admission=None):
    """Declare one classical host kernel and the one Experiment that runs it.

    Classical execution of Lanczos, FixedGCIM and Expectation has no quantum
    Program. Its Plan instead holds a one-node Program whose ClassicalStage
    names a SelectedKernel. The Method binds the kernel's Python function
    after planning.

    Args:
        source: Source record of the Method that owns the kernel.
        inputs: InputRefs the kernel reads. Repeated identities are kept once.
        labels: Names of the scalars the kernel returns, in order.
        work: Declared work of one invocation, stored as the kernel's invocation_work.
        description: Short statement of what the kernel computes, stored in the observation's ``padding`` field.
        frames: Scalar frame per label, ``unit`` for every label when None.
        admission: ``(option, max_steps)`` of a Method that sets its Programs' admission
            ceiling, such as ``("FixedGCIM.max_admission_steps", 1_000_000)``; the
            shared ``AdmissionLimits`` default when None.

    Returns:
        (construction, experiments): the SelectedConstruction and a
        one-element tuple with the ``acquisition`` Experiment that observes
        host_scalars under labels.
    """
    kernel = SelectedKernel(
        name="acquisition",
        implementation=source,
        inputs=tuple(dict((item.identity, item) for item in inputs).values()),
        scalars=labels,
        scalar_frames=("unit",) * len(labels) if frames is None else frames,
        resource_laws=(),
        workspace=(),
        construction_work=0,
        invocation_work=work,
        dependencies=("numpy",),
    )
    fields = dict(
        root="acquisition",
        definitions=(
            Definition(
                id="acquisition",
                node=ClassicalStage(implementation=source, boundary="host", kernel="acquisition"),
            ),
        ),
    )
    program = Program(**fields) if admission is None else admitted_program(*admission, **fields)
    construction = SelectedConstruction(program=program, selections=(), kernels=(kernel,))
    return construction, (
        Experiment(
            name="acquisition",
            setting="acquisition",
            observation=ObservationSpec(kind="host_scalars", labels=labels, padding=description),
        ),
    )


def eigen_error_model(plan, source):
    """Declare unresolved error sources in the requested original-input eigenvalue frame."""
    frame = plan.output.frame(plan.problem)
    names = (
        "native_preparation",
        "native_simulation",
        "sampling",
        "projected_solve",
        "ground_identification",
        "physical_model",
    )
    return ErrorModel(
        output_id=plan.output.content_id,
        subject_id=plan.problem.content_id,
        construction_id=plan.construction.content_id,
        frame=frame,
        required_sources=names,
        terms=tuple(
            ErrorTerm(
                name=name,
                stage="analysis",
                source=source,
                formula="absolute error in the requested original-input eigenvalue frame",
                coverage=name,
                fact=FramedFact(
                    frame=frame,
                    bindings=(),
                    fact=Fact(
                        quantity=name,
                        unit=frame.unit,
                        scope=frame.scope,
                        availability="unknown",
                        reason=f"no established {name} bound",
                    ),
                ),
            )
            for name in names
        ),
        source=source,
    )


def matched_chunks(plan, data):
    """Validate real readout/point associations, not merely shapes or family tags.

    Each chunk's point is selected once, through the Plan's construction
    memo. The Experiment and SelectedConstruction it returns give the
    realization, the setting label and the declared readout, so the point is
    not resolved a second time. A single-endpoint chunk must carry exactly
    the declared readout. A trajectory point chunk carries the one-point
    readout of its point and the identity of the whole declaration. It must
    name a declared point, carry exactly that point's one-point readout and
    name the declaration's identity. Points of one trajectory can share a
    one-point readout, so the chunk's boundary must also equal the boundary
    its preparation receipt resolved for that point. The trace join checks
    its point chunk key.
    """
    if data.trace.plan_id != plan.content_id:
        raise ValueError("analysis trace belongs to another selected Plan")
    seen, receipts = set(), None
    for chunk in data.observations.chunks:
        if chunk.content_id in seen:
            raise ValueError("an acquisition cannot contribute twice")
        seen.add(chunk.content_id)
        data.trace.validate_observation(chunk)
        if chunk.plan_id != plan.content_id or chunk.run_id != data.trace.run_id:
            raise ValueError("analysis contribution belongs to another Plan/run")
        selected, construction = plan._selected_construction(chunk.experiment, chunk.bindings)
        point = Realization(plan_id=plan.content_id, experiment=chunk.experiment,
                           bindings=construction.program.bindings)
        label, observation = _declared_readout(selected, construction)
        if chunk.point is not None:
            if observation.kind != "trajectory" or chunk.trajectory_id != observation.content_id:
                raise ValueError("analysis point contribution differs from its selected trajectory")
            if receipts is None:
                receipts = {receipt.content_id: receipt for receipt in data.receipts}
            receipt = receipts.get(chunk.prepared_id)
            if (receipt is None or receipt.observation.content_id != chunk.trajectory_id
                    or receipt.boundaries[observation.point_index(chunk.point)] != chunk.boundary):
                raise ValueError("analysis point contribution differs from its receipt's point boundary")
            observation = observation.point_observation(chunk.point)
        if (
            point.content_id != chunk.realization_id
            or point.bindings != chunk.bindings
            or label != chunk.setting
            or observation != chunk.observation
            or chunk.population != "unconditional"
        ):
            raise ValueError("analysis contribution differs from its actual selected point/readout")
        yield chunk, construction


def _declared_readout(selected, construction):
    """Return (setting label, declared readout) of a selected Experiment and its construction.

    This is ``Realization.resolved_observation`` for a caller that already
    holds the point's Experiment and SelectedConstruction: a direct
    experiment keeps its declaration, and a batch experiment resolves its
    batch against an admission of the construction's Program, binding an
    amplitude declaration to that construction's identity.
    """
    from nwqlib.ir.validation import _Admission

    if selected.batch is None:
        return selected.setting, selected.observation
    admission = _Admission(construction.program)
    admission.admitted().require_ready()
    construction_id = (construction.content_id
                       if (admission.nodes[selected.batch].observation_kind == "probabilities"
                           and selected.readout.amplitudes is not None) else None)
    return selected._resolved_observation(admission, construction_id=construction_id)
