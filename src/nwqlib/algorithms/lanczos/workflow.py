"""Two-stage sensitivity decisions over the common Run and original acquisitions.

Oumarou et al., arXiv 2603.15552v1, Section 3.3.2, p. 29, allocate shots by
moment sensitivity and suggest spending part of the budget on a pilot
estimate in a hardware experiment. This controller realizes that suggestion
as NWQLib's own design: a uniform pilot stage, one allocation decision from
the pilot moments (numerical._sensitivity_weights) and a main stage whose
moments alone enter the final analysis. The pilot chose the main sample
sizes, so each main population is a fixed-size sample conditional on the
pilot, and fresh independent main shots support the usual moment variance.
Pooling the pilot with the main stage would need a treatment of those
data-dependent sizes.

The Run checkpoint records each decision before its side effect. A restart
therefore reopens the same pending point and seed and reuses the frozen
allocation. An interrupted allocation or final analysis is repeated only
after run.resume(reanalyze=True). Physical observations are never
resubmitted.
"""

import numpy as np
from nwqlib.core.planning import RuntimeOptions
from nwqlib.ir import Binding
from nwqlib.execution import ObservationChunk, PendingPreparation
from nwqlib._prepared_execution import prepare_experiment, submit_experiment, restore_prepared
from .numerical import _allocate, _apply_pilot_floor, _sensitivity_weights


def _state(plan, run):
    """Return the controller checkpoint dict, creating it on the first call.

    Before a new state is saved, the Run's caps must admit 2k preparations,
    2k circuits and the whole shot budget for k settings. The pilot shots
    are split uniformly. The fields are the stage (0 pilot, 1 main, 2 done),
    the next setting index, the pilot and main allocations, the allocation
    evidence, the pending point and seed, the in-progress markers and
    attempt count, and the stored result.
    """
    if run.plan is not plan or plan.method.sampling is None:
        raise ValueError("sensitivity requires the same selected Plan and Run")
    state = run.checkpoint_state
    if state is not None:
        if state.get("kind") != "lanczos.sensitivity/3" or state.get("plan_id") != plan.content_id:
            raise ValueError("sensitivity checkpoint belongs to another controller/Plan")
        return state
    k = len(plan.experiments)
    settings = plan.method.sampling
    if k and settings.total_shots < 4 * k:
        raise ValueError("sensitivity needs at least two shots per setting per stage")
    if any(
        i not in plan.reconstruction.known_moments and i not in plan.reconstruction.moment_indices
        for i in range(2 * plan.reconstruction.krylov_dimension)
    ):
        raise ValueError("sensitivity requires all unknown Chebyshev moments")
    run.check_capacity(preparations=2 * k, circuits=2 * k, shots=settings.total_shots if k else 0)
    # Pilot shots: the requested fraction, raised or lowered so that each
    # stage keeps at least two shots per setting.
    total = (
        min(
            max(int(settings.total_shots * settings.pilot_fraction), 2 * k),
            settings.total_shots - 2 * k,
        )
        if k
        else 0
    )
    state = dict(
        kind="lanczos.sensitivity/3",
        plan_id=plan.content_id,
        stage=0,
        index=0,
        pilot_allocations=list(map(int, _allocate(total, np.ones(k)))) if k else [],
        main_allocations=None,
        allocation_evidence=None,
        pending=None,
        allocation_in_progress=False,
        analysis_in_progress=False,
        analysis_attempts=0,
        result=None,
    )
    run.checkpoint(state)
    return state


def _prepare(plan, run, state):
    """Prepare or reopen the pending stage and setting with its checkpointed allocation and seed.

    The pending point and seed are checkpointed before preparation, so a
    restart finds the same attempt or preparation instead of drawing a new
    seed. An interrupted local preparation raises rather than repeating it.
    """
    if state["pending"] is None:
        allocations = (
            state["pilot_allocations"] if state["stage"] == 0 else state["main_allocations"]
        )
        point = plan.resolve(
            plan.experiments[state["index"]].name,
            bindings=(
                Binding(parameter="stage", value=state["stage"]),
                Binding(parameter="shots", value=allocations[state["index"]]),
            ),
        )
        state["pending"] = dict(
            point=point.model_dump(mode="json", exclude_computed_fields=True),
            seed=run.rng.next_seed(),
            sequence=run.checkpoint_sequence + 1,
            prepared_id=None,
        )
        run.checkpoint(state)
    pending = state["pending"]
    event = run.attempt_for_checkpoint(pending["sequence"])
    if event is not None:
        return event
    from nwqlib.core.planning import Realization

    point = Realization.model_validate(pending["point"])
    prepared_id = pending["prepared_id"]
    if prepared_id is None:
        preparation_id = run.preparation_for_checkpoint(pending["sequence"])
        if preparation_id is not None:
            charge = run._state["preparation_charges"][preparation_id]
            prepared_id = charge.prepared_id
            if prepared_id is None:
                if preparation_id not in run._state["remote_preparations"]:
                    raise RuntimeError(
                        "original sensitivity preparation was interrupted; it was not repeated"
                    )
                from nwqlib._remote_preparation import refresh_preparation

                return refresh_preparation(preparation_id, run=run)
    if prepared_id is not None:
        return run._state["handles"].get(prepared_id) or restore_prepared(
            prepared_id, run=run, for_submission=True
        )
    prepared = prepare_experiment(point, run=run, runtime=RuntimeOptions(seed=pending["seed"]))
    if not isinstance(prepared, PendingPreparation):
        pending["prepared_id"] = prepared.record.content_id
        run.checkpoint(state)
    return prepared


def prepare_sensitivity(plan, *, run):
    """Prepare the setting the controller will submit next, if any stage remains."""
    state = _state(plan, run)
    if plan.experiments and state["stage"] < 2:
        _prepare(plan, run, state)


def recover_sensitivity_analysis(plan, *, run):
    """Clear an interrupted allocation or final-analysis marker on explicit request.

    It requires this controller's checkpoint with no pending acquisition and
    no stored result, only completed attempts, and the complete moment
    population of the interrupted stage. The retry is checkpointed before the
    marker is cleared, so the history keeps every analysis attempt. Nothing
    is acquired or resubmitted.
    """
    state = run.checkpoint_state
    if (
        run.plan is not plan
        or plan.method.sampling is None
        or not isinstance(state, dict)
        or state.get("kind") != "lanczos.sensitivity/3"
        or state.get("plan_id") != plan.content_id
        or state.get("pending") is not None
        or state.get("result") is not None
        or not (state.get("allocation_in_progress") or state.get("analysis_in_progress"))
    ):
        raise ValueError("reanalyze requires this sensitivity controller's interrupted analysis")
    if any(event.status != "completed" for event in run.trace.events):
        raise ValueError("analysis recovery cannot resolve an uncertain acquisition")
    stage = 0 if state["allocation_in_progress"] else 1
    statistics, _, _ = plan.method.statistics(plan, run.data, stage=stage)
    if set(statistics) != set(plan.reconstruction.moment_indices):
        raise ValueError("interrupted analysis lacks its complete original moment population")
    retry = dict(
        stage=stage,
        previous_attempts=state["analysis_attempts"],
        sequence=run.checkpoint_sequence + 1,
    )
    state["analysis_retries"] = [*state.get("analysis_retries", ()), retry]
    run.checkpoint(state)  # Save explicit retry intent while the interrupted marker is still set.
    state.update(allocation_in_progress=False, analysis_in_progress=False)
    run.checkpoint(state)


def execute_sensitivity(plan, *, run):
    """Acquire pilot moments, freeze the remaining allocation and analyze the completed main stage.

    Each stage submits one setting at a time in experiment order. After the
    last pilot setting, _sensitivity_weights and _apply_pilot_floor fix the
    main allocations of the remaining shots once. After the last main
    setting, Lanczos.analyze runs on the main-stage data. Every decision is
    checkpointed before its side effect. Returns the LanczosResult, or None
    while a preparation or submission is pending, after an attempt that did
    not complete, or after a cancellation request. An interrupted
    allocation or analysis raises RuntimeError until
    run.resume(reanalyze=True).
    """
    state = _state(plan, run)
    k = len(plan.experiments)
    while state["stage"] < 2:
        if run.cancel_requested is not None:
            return None
        if state["index"] == k:
            if state["stage"] == 0:
                if state["allocation_in_progress"]:
                    raise RuntimeError(
                        "sensitivity allocation was interrupted; explicit reanalysis is required"
                    )
                state["allocation_in_progress"] = True
                state["analysis_attempts"] += 1
                run.checkpoint(state)
                if k:
                    # Lanczos.plan admitted the pilot's solve and derivative
                    # under this Plan's limits (numerical.admit_projected_analysis).
                    statistics, _, _ = plan.method.statistics(plan, run.data, stage=0)
                    rows = tuple(statistics[i] for i in plan.reconstruction.moment_indices)
                    weights, evidence = _sensitivity_weights(
                        plan.reconstruction,
                        rows,
                        plan.method,
                    )
                    # Spend only the remaining budget after the pilot, with the pilot-derived
                    # floor preventing a selected main-stage setting from becoming too small.
                    remaining = plan.method.sampling.total_shots - sum(state["pilot_allocations"])
                    allocation = _allocate(remaining, weights)
                    if any(row.shots is None for row in rows):
                        raise ValueError("pilot floor requires actual returned shot populations")
                    allocation, floor, binding = _apply_pilot_floor(
                        allocation, np.array([row.shots for row in rows], dtype=int)
                    )
                    evidence.update(pilot_floor=floor.tolist(), pilot_floor_binding_count=binding)
                    state.update(main_allocations=allocation.tolist(), allocation_evidence=evidence)
                else:
                    state.update(
                        main_allocations=[],
                        allocation_evidence={"policy": "scalar operator; no acquisitions"},
                    )
                state.update(stage=1, index=0, allocation_in_progress=False)
            else:
                state.update(stage=2, index=0)
            run.checkpoint(state)
            continue
        prepared = _prepare(plan, run, state)
        pending = state["pending"]
        event = run.attempt_for_checkpoint(pending["sequence"])
        if event is not None:
            if event.status != "completed":
                return None
            chunk = run.completed_observation(event.attempt)
        elif isinstance(prepared, PendingPreparation):
            return None
        else:
            chunk = submit_experiment(prepared, run=run, checkpoint_sequence=pending["sequence"])
            if not isinstance(chunk, ObservationChunk):
                return None
        from nwqlib.core.planning import Realization

        if chunk.realization_id != Realization.model_validate(pending["point"]).content_id:
            raise ValueError("sensitivity completion differs from its pending point")
        run.collect(chunk)
        state.update(index=state["index"] + 1, pending=None)
        run.checkpoint(state)
    from .records import LanczosResult

    # Save an in-progress analysis marker before reconstruction. An interrupted
    # solve requires explicit recovery rather than an invisible repeat.
    if state["result"] is None:
        if state["analysis_in_progress"]:
            raise RuntimeError(
                "sensitivity final analysis was interrupted; explicit reanalysis is required"
            )
        state["analysis_in_progress"] = True
        state["analysis_attempts"] += 1
        run.checkpoint(state)
        result = plan.method.analyze(plan, run.data, settings={})
        state.update(
            result=result.model_dump(mode="json", exclude_computed_fields=True),
            analysis_in_progress=False,
        )
        run.checkpoint(state)
    return LanczosResult.model_validate(state["result"])._attach(plan, run.data)
