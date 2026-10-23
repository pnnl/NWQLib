"""RWPE feedback loop over the shared Run, with checkpoints of its iteration state.

One RWPE step (Granade and Wiebe, arXiv:2208.04526v1, Algorithm 1):

1. _pending computes (time, feedback) from the current mean and width
   (numerical.rwpe_choose), draws a backend seed and saves all three in a
   Checkpoint before any preparation.
2. _prepare and submit_experiment obtain one observation. Quantum
   execution returns one shot, the Bernoulli datum. Classical execution
   returns the nominal ancilla mean, and run_rwpe samples the datum from it
   with the saved seed.
3. run_rwpe applies the Eq. (7a) mean update (numerical.rwpe_update) and
   saves the new mean with the processed-step count in one checkpoint.

The width needs no storage because Eq. (7b) makes it a function of the
processed-step count (numerical.rwpe_scale). A restart reads the checkpoint
and resumes at the saved step with the saved point and seed. It never
replays earlier observations or reconstructs the trajectory, and a
completed observation updates the mean exactly once. Creating the
iteration state adds 2 work units. Choosing a point and applying an update
each add 32 work units to Checkpoint.work, which is checked against
Method.max_work before the step runs. The iteration state counts as 256
bytes against Method.max_bytes.

The loop has no consistency checks or unwinding, which the paper adds in
Algorithm 2. numerical.rwpe_estimate states what that omission means for
the estimate.
"""

from types import SimpleNamespace

import numpy as np
from pydantic import model_validator

from nwqlib.core.records import ContentID, Float64, Real, Record, Text
from nwqlib.operators.access import Count
from nwqlib.execution import ObservationChunk, ObservationView, PendingPreparation
from nwqlib.core.planning import RuntimeOptions
from nwqlib.ir import Binding
from nwqlib._prepared_execution import prepare_experiment, restore_prepared, submit_experiment
from nwqlib._counts import CountsSources
from . import numerical
from .records import RWPEGaussian, query_at, validate_selection


class Pending(Record):
    """The saved next RWPE query, written before its preparation.

    Attributes:
        checkpoint_sequence: Run checkpoint of this pending step and its preparation.
        experiment: Name of the continuous-time query in the Plan.
        power: Predetermined relative evolution time for this step.
        feedback: Ancilla feedback angle in radians.
        seed: Backend seed drawn once from the Run RNG, in the uint32 range.
    """

    checkpoint_sequence: Count
    experiment: Text
    power: Real
    feedback: Real
    seed: Count


class Checkpoint(Record):
    """RWPE iteration state between steps, with the last observation it used.

    Attributes:
        plan_id: Content hash of the original Plan, which fixes the prior and time schedule.
        mean: Gaussian mean in unwrapped phase radians.
        processed: Number of observations already incorporated in the mean.
        work: Cumulative work units of the iteration loop counted against this Plan.
        pending: Saved next experiment, feedback and backend seed, if any.
        last_attempt: Identifier of the last observation incorporated, so it cannot update twice.
    """

    plan_id: ContentID
    mean: Real
    processed: Count
    work: Count = 0
    pending: Pending | None = None
    last_attempt: Text | None = None

    @model_validator(mode="after")
    def _domain(self):
        """Require a pending backend seed in the uint32 range."""
        if self.pending is not None and self.pending.seed >= 2**32:
            raise ValueError("RWPE pending seed must belong to the backend uint32 stream")
        return self


def _state(plan, run):
    """Load or initialize the iteration state and check it against the Plan.

    A new Run saves the Gaussian prior before any work. A Run that has events
    but no checkpoint is rejected, because reconstructing the adaptive path
    would replay sampling decisions. A saved pending point must match
    the power and feedback that the Plan resolves for it.
    """
    validate_selection(plan)
    if run.plan is not plan or plan.method.estimator != "rwpe":
        raise ValueError("RWPE requires its exact original selected Plan and Run")
    saved = run.checkpoint_state
    if saved is None:
        if run.trace.events:
            raise ValueError("RWPE cannot infer missing controller state from observations")
        plan.method._admit(2, 256)
        state = Checkpoint(
            plan_id=plan.content_id, mean=plan.method.prior_mean, processed=0, work=2
        )
        run.checkpoint(state)
    else:
        state = Checkpoint.model_validate(saved)
    if (
        state.plan_id != plan.content_id
        or state.processed > plan.method.max_steps
    ):
        raise ValueError("RWPE checkpoint differs from its original selected Plan and steps")
    plan.method._admit(state.work, 256)
    if state.pending is not None:
        power, feedback = query_at(
            plan,
            state.pending.experiment,
            (Binding(parameter="feedback", value=Float64(value=state.pending.feedback)),),
        )
        if power != state.pending.power or feedback != state.pending.feedback:
            raise ValueError("RWPE pending point differs from its actual query")
    return state


def _checkpoint(run, state):
    """Save ``state`` as the Run's current checkpoint and return the saved record.

    The record is rebuilt without its ``parent_id``, so each checkpoint
    replaces the previous one instead of growing a chain of ancestors.
    """
    fields = {name: getattr(state, name) for name in Checkpoint.model_fields if name != "parent_id"}
    current = Checkpoint(**fields)
    run.checkpoint(current)
    return current


def _pending(plan, run, state):
    """Choose and save the next query unless one is pending or all steps are done.

    The work count is saved before the choice is computed, and the chosen
    power, feedback and backend seed are saved before preparation, so a
    restart reuses the same point and seed. The time 1/sigma_k must equal the
    planned query power, which planned_power_schedule fixed from the same
    rwpe_scale.
    """
    if state.pending is not None or state.processed == plan.method.max_steps:
        return state
    work = state.work + 32
    plan.method._admit(work, 256)
    state = _checkpoint(run, state.revise(work=work))
    power, feedback = numerical.rwpe_choose(
        state.mean, numerical.rwpe_scale(plan.method.prior_std, state.processed)
    )
    query = plan.reconstruction.queries[state.processed]
    if query.power != power:
        raise ValueError("RWPE continuous time differs from its planned Gaussian scale")
    # Persist the actual backend draw and selected point before preparation.
    seed = run.rng.next_seed()
    return _checkpoint(
        run,
        state.revise(
            pending=Pending(
                checkpoint_sequence=run.checkpoint_sequence + 1,
                experiment=query.experiment,
                power=power,
                feedback=feedback,
                seed=seed,
            )
        ),
    )


def _prepare(plan, run, state):
    """Prepare the pending query, or return the preparation already recorded for it.

    A restored preparation must match the saved feedback and seed. A local
    preparation that was interrupted before its prepared circuit was saved
    raises an error instead of building a replacement, since a replacement
    would be a different observation for the same pending step. A remote
    preparation is refreshed through its original locator at the provider.
    """
    pending = state.pending
    point = plan.resolve(
        pending.experiment,
        bindings=(Binding(parameter="feedback", value=Float64(value=pending.feedback)),),
    )
    identity = run.preparation_for_checkpoint(pending.checkpoint_sequence)
    if identity is None:
        return prepare_experiment(point, run=run, runtime=RuntimeOptions(seed=pending.seed))
    charge = run._state["preparation_charges"][identity]
    if charge.prepared_id is None:
        if identity not in run._state["remote_preparations"]:
            raise RuntimeError(
                "original RWPE preparation was interrupted; no replacement was built"
            )
        from nwqlib._remote_preparation import refresh_preparation

        return refresh_preparation(identity, run=run)
    prepared = run._state["handles"].get(charge.prepared_id) or restore_prepared(
        charge.prepared_id, run=run, for_submission=True
    )
    if (
        prepared.record.realization_id != point.content_id
        or prepared.record.runtime.seed != pending.seed
    ):
        raise ValueError("restored RWPE preparation differs from saved feedback or seed")
    return prepared


def prepare_rwpe(plan, *, run):
    """Return the preparation of the next RWPE query, or None when all steps are done."""
    state = _pending(plan, run, _state(plan, run))
    if state.pending is not None:
        return _prepare(plan, run, state)
    return None


def run_rwpe(plan, *, run):
    """Apply each completed one-bit update exactly once, and leave an unresolved step pending.

    Planning/reporting never replays the update history. Per-step Gaussian
    work is counted before it runs and survives an interrupted update. The
    current mean, pending seed and point, and Run RNG states stay unchanged
    until the corresponding new checkpoint has been saved.
    """
    from .method import _samples, _analysis

    state = _state(plan, run)
    sources = CountsSources(
        run.prepared_artifact, run._acquisition, max_bytes=run.limits.max_data_bytes
    )
    for chunk in run.observations.chunks:
        if chunk.observation.kind == "counts":
            sources.require_independent(chunk)
    # Advance one saved query at a time, preserving its chosen power, feedback
    # and seed through preparation, submission and collection.
    while state.processed < plan.method.max_steps:
        if run.cancel_requested is not None:
            break
        state = _pending(plan, run, state)
        pending = state.pending
        event = run.attempt_for_checkpoint(pending.checkpoint_sequence)
        if event is None:
            prepared = _prepare(plan, run, state)
            if isinstance(prepared, PendingPreparation):
                return None
            if prepared.record.observation.kind == "counts":
                sources.admit_preparation(prepared.record.content_id)
            chunk = submit_experiment(
                prepared, run=run, checkpoint_sequence=pending.checkpoint_sequence
            )
            if not isinstance(chunk, ObservationChunk):
                return None
            event = run.attempt_for_checkpoint(pending.checkpoint_sequence)
        elif event.status != "completed":
            return None
        else:
            if run.prepared_artifact(event.prepared_id).runtime.seed != pending.seed:
                raise ValueError("completed RWPE acquisition differs from its pending seed")
            chunk = run.completed_observation(event.attempt)
        if event is None or event.status != "completed" or event.attempt == state.last_attempt:
            raise ValueError("RWPE update requires one distinct completed acquisition")
        if chunk.observation.kind == "counts":
            sources.require_independent(chunk)
        view = ObservationView(chunks=(chunk,))
        trace = SimpleNamespace(
            plan_id=plan.content_id,
            run_id=run.run_id,
            events=(event,),
            validate_observation=run.validate_observation,
            _acquisition=run._acquisition,
        )
        (sample,) = _samples(
            plan, view, trace, receipts=(run.prepared_artifact(chunk.prepared_id),)
        )
        if sample.power != pending.power or sample.phase_shift != pending.feedback:
            raise ValueError("completed RWPE sample differs from its pending controller choice")
        if plan.execution == "quantum" and sample.shots != 1:
            raise ValueError("RWPE requires one returned Bernoulli datum per update")
        run.collect(chunk)
        # Charge the next Gaussian update before computing it. Publish its mean
        # and processed count together so a completed sample is not applied twice.
        work = state.work + 32
        plan.method._admit(work, 256)
        state = _checkpoint(run, state.revise(work=work))
        if plan.execution == "quantum":
            datum = sample.ones
        else:
            # Classical mode samples one bit from the acquired nominal signal.
            # Its saved point seed makes this emulation reproducible across an
            # interrupted update, without claiming a quantum shot or redrawing
            # the Run's stream. The raw exact signal stays in the observation.
            # sample.mean is <Z> = P(0) - P(1), so P(0) = (1 + mean)/2 and a
            # uniform u gives d = 1 exactly when u >= P(0).
            datum = int(np.random.default_rng(pending.seed).random() >= .5*(1+sample.mean))
        # Eq. (7a) with the width before this update. The feedback sign that
        # makes d = 0 raise the mean is derived in numerical.rwpe_choose.
        mean = numerical.rwpe_update(state.mean,
            numerical.rwpe_scale(plan.method.prior_std, state.processed), datum)
        state = _checkpoint(
            run,
            state.revise(
                mean=mean,
                processed=state.processed + 1,
                pending=None,
                last_attempt=event.attempt,
            ),
        )
    stop = (
        "original RWPE step count completed"
        if state.processed == plan.method.max_steps
        else "cancelled by caller"
    )
    run.terminate(reason=stop)
    run.progress("analyze", 0, 1)
    data = run.data
    samples = _samples(plan, data.observations, data.trace, receipts=data.receipts)
    result = _analysis(
        plan,
        data.observations,
        samples,
        receipts=data.receipts,
        gaussian=RWPEGaussian(mean=state.mean,
            standard_deviation=numerical.rwpe_scale(plan.method.prior_std, state.processed)),
        processed=state.processed,
        controller_work=state.work,
        stop_reason=stop,
    )
    run.progress("analyze", 1, 1)
    return result
