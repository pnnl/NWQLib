"""The pilot and main stages of ``PencilSampling`` on the common Run, for a fixed grouped GCiM pencil.

FixedGCIM and QFD measure every setting of the complete pencil in a pilot
stage with uniform shots, decide the main-stage shots of every setting once
from the pilot pencil (``pilot_decision``, with the arithmetic of
``pencil_sampling``), and measure every setting again in a fresh main stage
whose counts alone form the final H and S. Each stage point is the Plan
experiment of its setting with the Program parameters ``stage`` (0 pilot,
1 main) and ``shots`` bound through ``Plan.resolve``, so it keeps the batch,
counts readout and repetitions expression the Plan declares
(``fixed_basis._sampled_construction``). Lanczos's SensitivitySampling runs
a loop of the same kind in ``lanczos.workflow``, with its own split,
allocation rules and checkpoint.

The Run checkpoint records each decision before its side effect: a pending
point and its seed before its preparation, and the frozen ``ShotAllocation``
with the move to the main stage before the first main preparation. A
reopened Run therefore finds the original attempt or preparation, never
measures a pilot setting again and never chooses other main shots. Before
the first pilot preparation, the Run's limits must allow 2m preparations, 2m
circuits and the whole budget for m settings. An interrupted allocation or
final analysis is repeated only after ``run.resume(reanalyze=True)``, and
nothing is measured or submitted again for it.

``advance_stages`` drives the stages on a caller's state dictionary through
three functions: ``resolve(stage, index, shots)`` returns the point of one
setting, ``decide()`` the ``ShotAllocation`` from the collected pilot data,
and ``save(*fields)`` checkpoints the state, naming the fields that
changed. ``prepare_stages``, ``execute_stages`` and ``recover_stages`` are
the FixedGCIM and QFD hooks, whose iteration state is the whole checkpoint.
"""

import json
from math import isfinite

from nwqlib._prepared_execution import prepare_experiment, restore_prepared, submit_experiment
from nwqlib.algorithms._eigen_support import matched_chunks
from nwqlib.core.planning import Realization, RuntimeOptions
from nwqlib.core.records import Rational
from nwqlib.execution import ObservationChunk, PendingPreparation
from nwqlib.ir import Binding
from .fixed_basis import (
    ShotAllocation,
    pooled_statistics,
    processed_pauli_enclosure,
    read_groups,
    solve_pencil,
)
from .pencil import assemble_pair_pencil
from .pencil_sampling import (
    floored_allocation,
    influence_variances,
    local_variance,
    pilot_fallback_reason,
    pilot_split,
    sensitivity_weights,
    uniform_allocation,
)

KIND = "gcim.pencil_sampling/1"
# How to estimate one prepared point: the one sentence that ends the
# catalog reasons and prepare refusals of FixedGCIM, QFD and ADAPT, so that
# the three Methods name one estimate.
ONE_QUERY_ESTIMATE = ("nwqlib.resources.estimate(record.realization.selected_construction(plan)) "
                      "estimates one prepared query from its preparation record in run.data.receipts")
CATALOG_REASON = (
    "PencilSampling measures every setting in a pilot stage and again in a main stage whose shots the allocation "
    "chooses from the pilot observations, so the counts of the Plan's construction are not the totals of a Run. "
    "PencilSampling.total_shots is the shot budget of both stages, and " + ONE_QUERY_ESTIMATE)


def prepare_all_refusal(owner):
    """Return why a PencilSampling Plan of ``owner`` cannot prepare every setting in advance."""
    return (f"{owner} with PencilSampling cannot prepare every setting in advance, because the main stage takes "
            "the shots of each setting from the pilot-stage observations. prepare(plan) with the default "
            "settings='first' prepares the setting that the stages submit next, and " + ONE_QUERY_ESTIMATE)


def stage_chunks(chunks, stage):
    """Return the count chunks whose point binds ``stage`` (0 pilot, 1 main), in their order."""
    return tuple(chunk for chunk in chunks
                 if any(item.parameter == "stage" and item.value == stage for item in chunk.bindings))


def new_state(sampling, active):
    """Return the stage fields of a new pilot and main iteration over ``active`` settings.

    The pilot counts are the uniform split of the pilot budget of
    ``pilot_split``, which refuses a budget below four shots per setting.
    With no active setting the identity shortcut measures nothing: both
    stages are complete and the decision is the not-applicable
    ``ShotAllocation`` at once.
    """
    pilot, _ = pilot_split(sampling.total_shots, sampling.pilot_fraction, active)
    state = dict(stage=0, index=0, pilot_counts=list(uniform_allocation(pilot, active)), decision=None,
                 pending=None, allocation_in_progress=False, analysis_in_progress=False, analysis_attempts=0)
    if not active:
        decision = ShotAllocation(
            sampling=sampling, pilot_shots=0, main_shots=0, effective_fraction="not_applicable", pilot_counts=(),
            main_counts=(), variances=(), fallback_reason=None, local_variance=None, gap=None, pilot=None)
        state.update(stage=2, decision=decision.model_dump(mode="json", exclude_computed_fields=True))
    return state


def pilot_decision(sampling, settings, moments, *, basis_size, identity_shift, enclosure, cutoff):
    """Return the ``ShotAllocation`` that the pilot moments of the m settings decide.

    ``settings`` are the ``(name, i, j, quadrature, group)`` of the complete
    pencil in their order and ``moments`` the pilot-stage ``GroupMoment``
    records. The moments are pooled per setting and the pilot pencil is
    assembled by the shared ``pooled_statistics`` and
    ``assemble_pair_pencil``, and it is solved as the final pencil is, by
    ``solve_pencil`` with the sampled overlap policy, ``identity_shift`` and
    the operator ``enclosure`` diagnostic, at ``cutoff``. Without a solver
    failure, ``sensitivity_weights`` of its lowest coefficients and
    ``influence_variances`` with its lowest offset-free value e0 give v.
    ``pilot_fallback_reason`` reads the offset-free values, so the exact-tie
    test compares e0 and e1 and never the physical values. With a reason
    the main counts are the uniform split of ``B_main``, otherwise
    ``floored_allocation`` of v. ``gap`` is the rounded ``e1 - e0`` and
    ``local_variance`` the model's prediction for the main counts. A v_s
    that is not finite is recorded as None. No tolerance, floor or clipping
    enters.

    Raises:
        ValueError: If a pilot setting has no count data. A missing pilot
            measurement is an acquisition failure, never a reason for the
            uniform main stage.
    """
    pools = {}
    _, missing, pairs, _ = pooled_statistics(settings, moments, identity_shift, pools=pools)
    if missing:
        raise ValueError(f"the PencilSampling pilot stage has no count data for {', '.join(missing)}")
    pilot = solve_pencil(*assemble_pair_pencil(basis_size, pairs), cutoff=cutoff, sampled=True, enclosure=enclosure,
                         identity_shift=identity_shift)
    centered = pilot.offset_free_eigenvalues
    variances = () if pilot.failure_reason is not None else influence_variances(
        settings, pools, sensitivity_weights(settings, pilot.coefficients), centered[0])
    reason = pilot_fallback_reason(pilot.failure_reason, centered, pilot.coefficients, variances)
    pilot_shots, main_shots = pilot_split(sampling.total_shots, sampling.pilot_fraction, len(settings))
    counts = uniform_allocation(main_shots, len(settings)) if reason else floored_allocation(main_shots, variances)
    return ShotAllocation(
        sampling=sampling, pilot_shots=pilot_shots, main_shots=main_shots,
        effective_fraction=Rational(numerator=pilot_shots, denominator=sampling.total_shots),
        pilot_counts=uniform_allocation(pilot_shots, len(settings)), main_counts=counts,
        variances=tuple(v if v is None or isfinite(v) else None for v in variances),
        fallback_reason=reason, local_variance=local_variance(variances, counts) if variances else None,
        gap=centered[1] - centered[0] if len(centered) == 2 else None, pilot=pilot)


def _prepared(run, state, *, save, resolve, counts):
    """Prepare or reopen the pending point of the current stage and setting with its checkpointed seed.

    The pending point and seed are checkpointed before the preparation, so a
    restart finds the same attempt or preparation instead of drawing a new
    seed. An interrupted local preparation raises rather than repeating it.
    """
    if state["pending"] is None:
        point = resolve(state["stage"], state["index"], counts[state["index"]])
        state["pending"] = dict(point=point.model_dump(mode="json", exclude_computed_fields=True),
                                seed=run.rng.next_seed(), sequence=run.checkpoint_sequence + 1, prepared_id=None)
        save("pending")
    pending = state["pending"]
    event = run.attempt_for_checkpoint(pending["sequence"])
    if event is not None:
        return event
    prepared_id = pending["prepared_id"]
    if prepared_id is None:
        preparation_id = run.preparation_for_checkpoint(pending["sequence"])
        if preparation_id is not None:
            prepared_id = run._state["preparation_charges"][preparation_id].prepared_id
            if prepared_id is None:
                if preparation_id not in run._state["remote_preparations"]:
                    raise RuntimeError("the original PencilSampling preparation was interrupted; it was not repeated")
                from nwqlib._remote_preparation import refresh_preparation

                return refresh_preparation(preparation_id, run=run)
    if prepared_id is not None:
        return run._state["handles"].get(prepared_id) or restore_prepared(prepared_id, run=run, for_submission=True)
    prepared = prepare_experiment(Realization.model_validate(pending["point"]), run=run,
                                  runtime=RuntimeOptions(seed=pending["seed"]))
    if not isinstance(prepared, PendingPreparation):
        pending["prepared_id"] = prepared.record.content_id
        save("pending")
    return prepared


def advance_stages(run, state, *, save, resolve, decide):
    """Measure the remaining settings of both stages, with the main shots decided once in between.

    ``state`` holds the fields of ``new_state``. Each stage measures its
    settings one at a time in setting order. After the last pilot setting,
    ``decide()`` returns the ``ShotAllocation``, which is checkpointed with
    the move to the main stage before any main preparation. Returns True
    once both stages are complete, and False while a preparation or
    submission is pending, after an attempt that did not complete, or after
    a cancellation request.

    Raises:
        RuntimeError: If an earlier allocation was interrupted, until
            ``run.resume(reanalyze=True)`` clears its marker.
    """
    while state["stage"] < 2:
        if run.cancel_requested is not None:
            return False
        counts = state["pilot_counts"] if state["stage"] == 0 else state["decision"]["main_counts"]
        if state["index"] == len(counts):
            if state["stage"] == 0:
                if state["allocation_in_progress"]:
                    raise RuntimeError("the PencilSampling allocation was interrupted; explicit reanalysis is "
                                       "required")
                state["allocation_in_progress"] = True
                state["analysis_attempts"] += 1
                save("allocation_in_progress", "analysis_attempts")
                state["decision"] = decide().model_dump(mode="json", exclude_computed_fields=True)
                state.update(stage=1, index=0, allocation_in_progress=False)
                save("decision", "stage", "index", "allocation_in_progress")
            else:
                state.update(stage=2, index=0)
                save("stage", "index")
            continue
        prepared = _prepared(run, state, save=save, resolve=resolve, counts=counts)
        pending = state["pending"]
        event = run.attempt_for_checkpoint(pending["sequence"])
        if event is not None:
            if event.status != "completed":
                return False
            chunk = run.completed_observation(event.attempt)
        elif isinstance(prepared, PendingPreparation):
            return False
        else:
            chunk = submit_experiment(prepared, run=run, checkpoint_sequence=pending["sequence"])
            if not isinstance(chunk, ObservationChunk):
                return False
        if chunk.realization_id != Realization.model_validate(pending["point"]).content_id:
            raise ValueError("a PencilSampling completion differs from its pending point")
        run.collect(chunk)
        state.update(index=state["index"] + 1, pending=None)
        save("index", "pending")
    return True


def _state(plan, run):
    """Return the iteration state of a FixedGCIM or QFD PencilSampling Run, creating it on the first call."""
    if run.plan is not plan or plan.method.sampling is None:
        raise ValueError("PencilSampling requires the same Plan and Run")
    state = run.checkpoint_state
    if state is not None:
        if state.get("kind") != KIND or state.get("plan_id") != plan.content_id:
            raise ValueError("the PencilSampling checkpoint belongs to another iteration or Plan")
        return state
    m, sampling = len(plan.experiments), plan.method.sampling
    run.check_capacity(preparations=2 * m, circuits=2 * m, shots=sampling.total_shots if m else 0)
    state = dict(kind=KIND, plan_id=plan.content_id, **new_state(sampling, m))
    run.checkpoint(state)
    return state


def _drivers(plan, run, state, rec, label):
    """Return the ``save``, ``resolve`` and ``decide`` functions of a FixedGCIM or QFD iteration."""
    def save(*fields):
        return run.checkpoint(state, changed=fields)

    def resolve(stage, index, shots):
        return plan.resolve(plan.experiments[index].name, bindings=(
            Binding(parameter="stage", value=stage), Binding(parameter="shots", value=shots)))

    def decide():
        # read_groups checks the pilot pass against the Method's limits from its own chunks.
        settings = rec.settings()
        moments = read_groups(
            stage_chunks((chunk for chunk, _ in matched_chunks(plan, run.data)), 0), settings,
            members=[group.indices for group in rec.groups], rows=rec.nonidentity_terms,
            num_qubits=rec.num_qubits, basis_size=rec.basis_size, max_bytes=plan.method.max_bytes,
            max_analysis_work=plan.method.max_analysis_work, label=label)
        return pilot_decision(plan.method.sampling, settings, moments, basis_size=rec.basis_size,
                              identity_shift=rec.identity_shift, enclosure=processed_pauli_enclosure(rec),
                              cutoff=plan.method.overlap_cutoff)

    return save, resolve, decide


def prepare_stages(plan, *, run, rec, label):
    """Prepare the setting that the stages submit next, if one remains before the allocation or the end."""
    state = _state(plan, run)
    if state["stage"] < 2:
        counts = state["pilot_counts"] if state["stage"] == 0 else state["decision"]["main_counts"]
        if state["index"] < len(counts):
            save, resolve, _ = _drivers(plan, run, state, rec, label)
            _prepared(run, state, save=save, resolve=resolve, counts=counts)


def execute_stages(plan, *, run, rec, label):
    """Measure both stages of a FixedGCIM or QFD Plan and analyze the main stage once they are complete.

    ``rec`` is the Plan's ``FixedGCIMReconstruction`` and ``label`` starts
    the messages of the pilot pass. The final analysis is the Method's
    ``analyze`` on the Run's data, which reads the main-stage chunks and the
    frozen decision. It is marked in progress before it runs. Returns the
    Result, or None while work is pending.

    Raises:
        RuntimeError: If the allocation or the final analysis was
            interrupted, until ``run.resume(reanalyze=True)``.
    """
    state = _state(plan, run)
    save, resolve, decide = _drivers(plan, run, state, rec, label)
    if not advance_stages(run, state, save=save, resolve=resolve, decide=decide):
        return None
    if state["analysis_in_progress"]:
        raise RuntimeError("the PencilSampling final analysis was interrupted; explicit reanalysis is required")
    state["analysis_in_progress"] = True
    state["analysis_attempts"] += 1
    save("analysis_in_progress", "analysis_attempts")
    result = plan.method.analyze(plan, run.data, settings={})
    state["analysis_in_progress"] = False
    save("analysis_in_progress")
    # Attach a copy to the data after the last checkpoint, whose iteration state shows the analysis complete.
    return type(result).model_validate(result.model_dump(mode="json", exclude_computed_fields=True))._attach(
        plan, run.data)


def recover_stages(plan, *, run):
    """Clear an interrupted allocation or final-analysis marker on explicit request.

    It requires this Plan's iteration state with no pending measurement,
    only completed attempts and a count table of every setting in the
    interrupted stage. The retry is checkpointed before the marker is
    cleared, so the state keeps every analysis attempt. Nothing is measured
    or submitted again.
    """
    state = run.checkpoint_state
    if (run.plan is not plan or plan.method.sampling is None or not isinstance(state, dict)
            or state.get("kind") != KIND or state.get("plan_id") != plan.content_id
            or state.get("pending") is not None
            or not (state.get("allocation_in_progress") or state.get("analysis_in_progress"))):
        raise ValueError("reanalyze requires an interrupted analysis of this PencilSampling iteration")
    if any(event.status != "completed" for event in run.trace.events):
        raise ValueError("analysis recovery cannot resolve an uncertain acquisition")
    stage = 0 if state["allocation_in_progress"] else 1
    if {chunk.experiment for chunk in stage_chunks(run.observations.chunks, stage)} != {
            experiment.name for experiment in plan.experiments}:
        raise ValueError("the interrupted analysis lacks a count table of a setting of its stage")
    retry = dict(stage=stage, previous_attempts=state["analysis_attempts"], sequence=run.checkpoint_sequence + 1)
    state["analysis_retries"] = [*state.get("analysis_retries", ()), retry]
    run.checkpoint(state, changed=("analysis_retries",))  # Save the retry while the marker is still set.
    state.update(allocation_in_progress=False, analysis_in_progress=False)
    run.checkpoint(state, changed=("allocation_in_progress", "analysis_in_progress"))


def frozen_allocation(plan, data):
    """Return the ``ShotAllocation`` saved in the Run's iteration state of ``data``, or None before the decision.

    Analysis, reanalysis at another cutoff and loading read it as it was
    decided, so the pilot is never solved again and no shot is chosen again.

    Raises:
        ValueError: If the saved iteration state belongs to another
            iteration or Plan.
    """
    if data.controller is None:
        return None
    state = json.loads(data.controller)["state"]
    if state.get("kind") != KIND or state.get("plan_id") != plan.content_id:
        raise ValueError("PencilSampling analysis requires the iteration state of its own Plan")
    return None if state.get("decision") is None else ShotAllocation.model_validate(state["decision"])
