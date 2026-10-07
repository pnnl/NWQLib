"""One selected-experiment lifecycle for local and durable backend execution.

A Run is the single owner of preparation, submission, collection, native and
numerical caches, cumulative limits and exposure for one execution (one
``run_id``) of a selected Plan. Method controllers, backend adapters and the
public scientist entry points all act through it. An acquisition is therefore
reserved in one place, its outcome is published in one place, and one journal
records both. A second executor inside the same execution would need its own
exposure ledger, and the two ledgers could disagree about what was spent or which
job produced which data. Repeating a Plan's work deliberately creates a second
Run with its own ledger.

Lifecycle rules and the failure each one prevents:

1. Reserve before work. A preparation commits its charge and the RNG position
   after its seed draw, and a submission commits its intent, reserved events and
   output-byte reservation, before any native or provider call starts. A crash
   inside the call then leaves a durable record that the draw was used and the
   exposure was spent. Without that record a restart could reuse a draw or submit
   the same work twice.
2. Publish outcomes atomically. A completed observation, its event, its submission
   status and every array it published commit in one journal transaction. The
   journal never holds an observation without its exposure record, or an array
   without the acquisition that produced it.
3. An unrecoverable outcome stays uncertain. An interrupted or unacknowledged
   acquisition keeps its reserved circuits and shots charged and is never relabelled
   failed, resubmitted or refunded, because the provider may already have executed
   or billed it. Only a host kernel that finished with an ordinary exception becomes
   a failed attempt. A provider-reported failure or cancellation is recorded on the
   submission, and its unfinished attempts stay uncertain, because the provider
   does not report how much of the work ran. When no retrieval route remains,
   ``Run.resume`` raises ``RunFailed`` with stage ``"recovery"`` and status
   ``"uncertain"``. A reopened synchronous Aer submission with no locator is
   marked failed with its results consumed because its in-process output is
   unavailable. Its attempt stays uncertain and charged, and continuation
   raises the recovery error with that attempt's identity when the Method
   cannot return a lawful Result and cancellation has not been requested.
   ``Run._restore_submissions`` makes this change and releases the
   submission's output reservation.
4. Pending work keeps its identity. Retrieval uses the original locator, attempt
   identities and submission item ordinals, and reopening restores the RNG
   position and the collection ordinals. A refresh error is recorded on the same
   submission and propagates. Nothing is launched in its place.
5. Live state follows the commit. In-memory counters, dictionaries and handles
   change only after the journal transaction succeeds, or are rolled back when it
   fails. A failed journal write sets ``storage_failed``, which blocks further work
   until the durable Run is reopened, so memory never runs ahead of disk.

Commits of one static acquisition, in the order they happen, and what reopening
finds when the process stops after each one. A static Plan has a fixed list of
experiments, each prepared and acquired once by ``prepare_static`` and
``execute_static``. ``prepare_static`` prepares the first experiment, or all of
them for ``prepare(plan, settings="all")``, and ``execute_static`` prepares each
remaining one when it reaches it. An adaptive controller chooses its next
experiment from data. It uses the same preparation, submission and outcome
commits, with its controller checkpoint (``Run.checkpoint``) in place of the
workflow row. The numbering is the same as in ``docs/development/execution.md``,
and each function below that makes one of these commits names it in its
docstring. The docstrings of the other writes (limit amendments, synthesis
reservations, cancellation requests, failure revisions and the Result row)
state what a stop before or after them leaves.

1. Workflow row, with the experiment's runtime seed and the RNG position
   (``_static_item``). A restart before commit 2 prepares the experiment with this
   same seed.
2. Preparation charge and RNG position (``prepare_experiment``). If the receipt
   has not committed, reopening counts a circuit preparation's charge against
   ``max_total_circuits`` and the static workflow does not rebuild the
   preparation. The Run never repeats a failed or interrupted preparation on its
   own, since that would be new charged work the caller did not request, so a
   Run whose local preparation has a charge and no receipt cannot complete
   (``_unbuilt``), and ``execute_static`` raises ``PreparationNotRebuilt``
   before it submits any further experiment. A saved remote preparation can
   still be refreshed instead. A revision of the charge reserves each exact
   dense synthesis before it starts, or records its refusal and leaves the
   experiment unprepared (``Run._charge_synthesis``).
3. Receipt and native payload (``_finish_native_preparation`` or
   ``_prepare_host``). A later resume submits the saved preparation without
   lowering it again.
4. Submission intent, reserved events and output-byte reservation
   (``Run._begin_submission``). The backend is contacted only after this
   commit. If no acknowledgement or outcome follows, reopening marks the intent
   and its events uncertain. Only a backend ``reconcile`` can then find the job,
   by the original submission identity. A synchronous backend has no
   acknowledgement step, so this window lasts until commit 6. Without a
   retrieval route, resume raises the recovery error. A reopened synchronous
   Aer submission with no locator follows lifecycle rule 3. A host kernel
   (``_submit_host``) commits only its reserved event, because nothing leaves
   the process. Reopening makes that event uncertain, and no route can
   retrieve its outcome.
5. Acknowledged locator (``Run._ack_submission``, detached backends only). The
   submission stays acknowledged and its events reserved. The next refresh
   retrieves the original job by its locator.
6. Outcome (``Run._finish_attempt``). The completed event, the observation chunk,
   the submission status and any published arrays commit in one transaction, so
   reopening never finds one without the others.
7. Collection ordinal (``Run.collect``). The Method's consumption of the
   observation is its own commit, so the order in which a controller consumed
   observations survives reopening even when completion order differed.

A refresh of a detached job (``refresh_submissions``) splits commit 6 in two. It
first commits the provider status and item associations, which must survive a
later decoding error. The observations commit in a second transaction, which also
sets the submission's ``results_consumed`` once the job is terminal. If the
process stops between the two, the flag is still unset, so the next refresh reads
the same job again and skips items already published. Each observation is
therefore published once.

These windows describe an interrupted process, for example a keyboard interrupt,
an unhandled exception or a killed process. Recovery after power loss is outside
this contract.
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from time import monotonic, perf_counter, sleep
from uuid import uuid4
import json
import sys

from nwqlib.artifacts import ARRAY_PAYLOAD_FORMATS
from nwqlib.backends.capabilities import unsupported_readout
from nwqlib.core.analysis import Result, RunData
from nwqlib.core.planning import ObservationSpec, Plan, RandomStreams, Realization, RuntimeOptions, readout_shape
from nwqlib.core.records import Record, Source
from nwqlib.execution import (
    ConsumptionEvent, CountBin, EstimateValue, ExecutionLimits, ExecutionTrace, JobLocator, LimitAmendment,
    LogicalPreparationReceipt, MAX_COUNT, ObservationChunk, ObservationView, PauliValue, PayloadRef,
    PendingPreparation, PreparedArtifact, RegisterMap, SubmissionItem, _probability_mapping, point_chunk_key,
    RunFailed, ScalarValue, SubmissionRecord, TimingObservation,
)
from nwqlib._run_journal import (
    RUN_FORMAT, JSONBytesExceeded, LocalJournal, PreparationCharge, WorkflowItem, encode_records,
    unsupported_run_format,
)
from nwqlib.ir.validation import AdmissionStepsExceeded, _Admission, admission_refusal

# Backend kinds that simulate on this machine: local Aer and a locally launched
# NWQ-Sim runner. The Run bounds their circuit width before native work
# (ExecutionLimits.max_simulation_qubits). They and host execution are also the
# backends on which prepare(plan, settings="all") prepares every setting,
# because their preparation is local lowering with no remote compilation or
# provider charge.
LOCAL_SIMULATORS = frozenset({"qiskit_aer", "nwqsim"})
# The recorded failure of a reopened synchronous Aer submission whose
# in-process result is gone (Run._restore_submissions). Its attempt has no
# route to an outcome, so resume raises the recovery error with this text
# (Run._raise_unrecoverable_outcome).
_AER_REOPEN_FAILURE = "interrupted synchronous Aer result is unavailable after reopen"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _limit_remedy(name):
    """The sentence of a refusal message that names both ways to raise the execution limit ``name``.

    ``solve``, ``prepare`` and ``Run`` take ``limits`` for the Run they create.
    ``solve`` closes that Run when an error leaves it, so its caller passes a
    larger limit to a new call. ``Run.extend_limits`` raises a limit of an
    open Run, and the work already counted stays counted.
    """
    return (f"Raise it with limits=ExecutionLimits({name}=...) in solve, prepare or Run, "
            f"or with run.extend_limits({name}=...) on an open Run.")


def _data_limit_clause(run, unit=""):
    """The clause that names a Run's ``max_data_bytes`` as the source of a limit passed to a shared check.

    ``readout_shape`` and ``admit_materialization`` append it to a refusal,
    because they name only their own ``max_items`` or ``max_bytes``. ``unit``
    states how the passed limit follows from ``max_data_bytes``.
    """
    return (f"which is this Run's max_data_bytes={run.limits.max_data_bytes}{unit}. "
            f"{_limit_remedy('max_data_bytes')}")


def _cumulative_limit_error(name, limit, noun, used, requested, *, owner="this Run's"):
    """Return the error that refuses ``requested`` more units of ``noun`` when ``used`` count against ``name``.

    The message gives the limit, both amounts and the remedy. ``owner`` names
    whose limit it is, ``"the proposed"`` for a value that ``extend_limits``
    has not yet published. For ``max_total_circuits`` and ``max_total_shots``
    the request is an exact count, and the message names the sum of both
    amounts as the smallest limit that admits it. An adaptive Method requests
    its work in steps, so its whole workload can need more than that sum.
    For ``max_data_bytes`` the cap reserves nothing and the complete
    requirement is not known at this point, so the message suggests
    ``max(used + requested, 2 * limit, 100_000_000)`` and names the default
    ``DEFAULT_MAX_BYTES``.
    """
    if name == "max_data_bytes":
        suggested = max(used + requested, 2 * limit, 100_000_000)
        return ValueError(f"{owner} {name}={limit} cannot admit the requested {noun} ({used} counted, "
                          f"{requested} requested). Raise max_data_bytes to at least {suggested}: the cap "
                          "reserves nothing, and the complete requirement is not known at this point "
                          f"(the default, DEFAULT_MAX_BYTES, is {DEFAULT_MAX_BYTES}, 10 GB). "
                          f"{_limit_remedy(name)}")
    return ValueError(f"{owner} {name}={limit} cannot admit the requested {noun} ({used} counted, "
                      f"{requested} requested, at least {used + requested} needed). {_limit_remedy(name)}")


def _notebook_progress():
    """Use one live-kernel display. Ordinary Python does not import IPython."""
    ipython = sys.modules.get("IPython")
    shell = None if ipython is None else ipython.get_ipython()
    if getattr(shell, "kernel", None) is None:
        return None
    handle = None

    def render(stage, done, total):
        """Show one line of text for ``stage`` in a single updatable notebook output.

        The first call creates the display and later calls replace its text, so a
        long Run shows its current stage instead of appending one line per event.
        ``done`` is a running count (prepared artifacts or collected
        observations), and ``done == total`` marks the end of analysis.
        """
        from IPython.display import display

        nonlocal handle
        text = {"input": "Input accepted", "prepare": "Preparing selected work",
                "submit": "Submitting acquisitions", "collect": "Collecting observations",
                "analyze": "Analyzing observations"}[stage]
        if stage == "prepare" and done is not None:
            text = f"Prepared artifacts: {done}"
        elif stage == "collect" and done is not None:
            text = f"Collected observations: {done}"
        elif stage == "analyze" and total is not None and done == total:
            text = "Analysis finished"
        value = {"text/plain": text}
        if handle is None:
            handle = display(value, raw=True, display_id=True)
        else:
            handle.update(value, raw=True)

    return render


@dataclass(frozen=True, init=False, eq=False)
class Run:
    """One execution of a Plan: its circuits, attempts, measured data, limits and counted work.

    [`prepare`][nwqlib.scientist.prepare] creates a Run (`prepared.run`),
    [`submit`][nwqlib.scientist.submit] starts it, and
    [`load_run`][nwqlib.scientist.load_run] reopens a saved one. `wait()`
    returns the Result, and `resume()` continues the Run once without
    waiting. Close a Run when done, for example with
    `with prepared.run as run:`. Closing cancels no job.

    The Run submits the work, saves its outcomes and counts the work against
    the limits. The limits apply to totals
    over the Run's whole life, failed and uncertain attempts included, so
    raising the limits or reopening the Run never gives back work already
    counted. A Run with a folder records every step in `run.sqlite` there
    and holds an exclusive lock on the folder until it is closed. A Run
    without a folder keeps the same counts in memory and can be saved later
    with `save`. [Run on a backend](../prepared_execution.md) and [Continue
    an interrupted run](../run_archives.md) describe both. The fields below
    are read-only.

    Attributes:
        plan: The Plan being executed. Continuing the Run never replaces it.
        backend: The backend, or `None` for a Run whose only computations are
            classical computations that the Method runs on this computer.
        forecast: The `PlanEstimate` given with the Plan, or `None`.
        allocation: The `Allocation` given with the Plan, or `None`. Continuing
            the Run does not recompute it.
        run_id: Identifier of this execution, distinct from another Run of the
            same Plan.
    """

    plan: Plan
    backend: object
    forecast: object
    allocation: object
    run_id: str
    _state: dict = field(repr=False)
    _lock: object = field(repr=False)

    def __init__(self, plan, *, backend=None, limits=None, directory=None, progress=None,
                 forecast=None, allocation=None):
        # Admit the selected execution and create its original identity, streams
        # and optional journal. ``prepare`` normally constructs the Run.
        #
        # A durable Run writes, in this order, its new directory, the journal
        # with its controller lock, the selected inputs with ``run.json``, and
        # finally the journal header with the initial RNG snapshot. The selected
        # inputs are therefore on disk before any backend can start work. A
        # crash before the header commits leaves a folder that ``load_run``
        # rejects, and no preparation or acquisition can have happened by then.
        #
        # plan: selected Plan. Its randomness snapshot seeds the Run's streams.
        # backend: backend connection. None selects local Aer for quantum
        # execution and no backend for classical execution. limits: cumulative
        # ExecutionLimits, the defaults when None. directory: new durable
        # folder. Remote-capable backends always get one, under
        # ~/.nwqlib/runs/<run_id> when none is given, so that an acknowledged
        # external job can be reopened after the process ends. progress:
        # progress callback, False to disable, or None for the notebook
        # display. forecast: already computed PlanEstimate for this Plan.
        # allocation: supplied Allocation matching the forecast.
        if not isinstance(plan, Plan):
            raise TypeError("Run requires a selected Plan")
        from nwqlib._run_archive import validate_provenance
        validate_provenance(plan, forecast, allocation)
        if limits is None:
            limits = ExecutionLimits()
        if type(limits) is not ExecutionLimits:
            raise TypeError("limits must be ExecutionLimits")
        if plan.execution == "classical" and backend is not None:
            raise ValueError("classical execution cannot select a quantum backend")
        if backend is None and plan.execution == "quantum":
            from nwqlib.backends.connection import AerBackend
            backend = AerBackend()
        # A synchronous backend returns results from submit. A detached backend admits and
        # launches a batch, then refreshes it from restored native handles. Reconcile and
        # cancel stay optional because their callers handle their absence.
        required = (("target_for", "submit") if getattr(backend, "supports_synchronous", True)
                    else ("target_for", "admit_batch", "launch", "refresh", "restore_native"))
        if backend is not None and not all(callable(getattr(backend, name, None)) for name in required):
            raise TypeError("backend must provide selected preparation and submission")
        identity = str(uuid4())
        self._initialize(plan, backend, limits, identity, progress, forecast, allocation)
        # Remote-capable backends need a durable frontier even without an explicit
        # directory, so an acknowledged external job can be reopened.
        durable = backend is not None and (callable(getattr(backend, "launch", None))
                                           or callable(getattr(backend, "prepare_local", None)))
        try:
            if directory is not None or durable:
                path = Path(directory) if directory is not None else Path.home() / ".nwqlib" / "runs" / identity
                path.mkdir(parents=True, exist_ok=False)
                self._state["directory"] = path
                self._state["journal"] = LocalJournal(path / "run.sqlite", limits.max_data_bytes, create=True)
                self._state["backend_context"]["persist_prepared"] = True
                from nwqlib._run_archive import initialize
                initialize(self)
            self._write((("header", "run", dict(format=RUN_FORMAT, run_id=identity,
                plan_id=plan.content_id, backend=None if backend is None else backend.model_dump(mode="json"),
                limits=limits, forecast=forecast, allocation=allocation)), ("rng", "current", self.rng.snapshot())))
        except BaseException as error:
            if self.directory is not None:
                error.add_note(f"Original run initialization data is saved at {self.directory}")
            self.close()
            raise

    def _initialize(self, plan, backend, limits, identity, progress, forecast, allocation):
        """Bind the identity fields and define every mutable state key once.

        A new Run and ``_restore`` both start here, so a reopened Run cannot miss a
        key that a new one has. The counters mean:

        - ``preparations`` counts every preparation attempt, including failed and
          interrupted ones. ``host_preparations`` is its host-kernel subset.
        - ``shots`` sums the reserved raw shots of every attempt, including uncertain
          ones, and ``events`` lists every attempt in reservation order.
        - ``pending_output`` maps each attempt whose outcome is not yet published to
          its reserved output bytes.
        - Stored data is split into ``data_bytes`` (journal JSON rows),
          ``array_bytes`` (published arrays), ``payload_bytes`` (other binary
          payloads) and ``external_bytes`` (archive files on disk). Together with
          ``pending_output`` they are charged against ``max_data_bytes``.
        - ``synthesis_work`` sums the exact dense synthesis work that preparations
          reserved against ``max_synthesis_work`` (``_charge_synthesis``), and
          ``synthesis_preparation`` names the preparation that is reserving it.
          ``synthesis_cache`` holds the circuits of the syntheses that basis
          lowering made in this Run object's preparations
          (``_exact_dense_unitaries``), each reserved against
          ``max_synthesis_work``.
          :meth:`close` empties it. It is not saved, so a reopened Run starts
          with an empty cache.
        """
        for name, value in (("plan", plan), ("backend", backend), ("run_id", identity), ("_lock", RLock()),
                            ("forecast", forecast), ("allocation", allocation)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_state", dict(
            limits=limits, initial_limits=limits, limit_amendments=[],
            rng=RandomStreams.restore(plan.randomness), closed=False, storage_failed=False,
            preparations=0, host_preparations=0, construction_work=0, host_preparation_work=0,
            shots=0, host_invocations=0, host_work=0, synthesis_work=0, synthesis_preparation=None,
            synthesis_cache={},
            events=[], attempt_indices={}, submissions={}, submission_items={}, pending_submissions={},
            chunks={}, receipts={}, by_attempt={}, prepared_artifacts={}, local_prepared_ids=[], handles={},
            refresh_cache={}, observation_view=None, trace_snapshot=None,
            counts_sources=None, counts_source_failure=None,
            preparation_charges={}, checkpoint_preparations={}, checkpoint_attempts={},
            checkpoint_fields=None, checkpoint_sequence=0,
            remote_preparations={}, definitions=({}, {}), specializations={}, backend_context={},
            method_context=None, workflow_items={}, workflow_current=None, pending_output={},
            data_sizes={}, data_bytes=0, array_bytes=0, payload_sizes={}, payload_bytes=0, external_bytes=0,
            artifacts=None, publication_batch=None,
            artifact_payloads={}, journal=None, directory=None, started=False, result=None,
            cancel_requested=None, termination_reason=None,
            progress=_notebook_progress() if progress is None else progress, warnings=[]))
        from nwqlib._run_archive import bind_caches
        bind_caches(self)

    @property
    def limits(self):
        """The Run's current [`ExecutionLimits`][nwqlib.execution.ExecutionLimits]."""
        return self._state["limits"]

    @property
    def limit_amendments(self):
        """Every raise of the Run's limits so far, in order.

        Each is a [`LimitAmendment`][nwqlib.execution.LimitAmendment] with
        the old and new limits and the Run's counts just before the raise.
        """
        return tuple(self._state["limit_amendments"])

    @property
    def rng(self):
        return self._state["rng"]

    @property
    def directory(self):
        """The Run's folder, as a `pathlib.Path`, or `None` for a Run without one.

        A backend that runs jobs outside this process always gets a folder,
        under `~/.nwqlib/runs/<run_id>` when `prepare` was given no
        `directory`, so that a submitted job can be reopened after the
        process ends.
        """
        return self._state["directory"]

    @property
    def result(self):
        """The Run's Result once the run is complete, and `None` before."""
        return self._state["result"]

    @property
    def cancel_requested(self):
        return self._state["cancel_requested"]

    def _ensure_open(self):
        """Refuse work on a closed Run, or after a failed journal write.

        After a failed write the live state may no longer match the journal, so
        only reopening the durable Run from disk (lifecycle rule 5) can continue.
        """
        if self._state["closed"]:
            raise ValueError("run is closed")
        if self._state["storage_failed"]:
            raise ValueError(f"run storage failed; close this Run (after a first submit, prepared.run) and "
                             f"load the durable run from {self.directory} with nwqlib.load_run before further work")

    def check_data(self, size):
        """Reject one requested or decoded object larger than the whole data cap.

        This is a cheap per-object check before the object is built or read.
        Cumulative stored data is admitted separately by ``check_capacity`` and at
        each journal write.
        """
        if type(size) is not int or size < 0:
            raise ValueError("data byte count must be a nonnegative integer")
        if size > self.limits.max_data_bytes:
            raise ValueError(f"native output or decoded data of {size} bytes exceeds this Run's "
                             f"max_data_bytes={self.limits.max_data_bytes}. {_limit_remedy('max_data_bytes')}")

    def _write(self, records=(), *, collected=(), payloads=(), release_output=None, reserve_output=0,
               max_data_bytes=None, deleted=(), metadata_credit=0):
        """Admit only the changed records before their durable or live publication.

        This is the single write path of a Run. The stored-data invariant is::

            data_bytes + array_bytes + payload_bytes + external_bytes
                + pending_output <= max_data_bytes

        The new rows are admitted and encoded against the space left by that sum
        before any JSON text is allocated, and the actual encoded size is checked
        again. Output reservations named by ``release_output`` leave the pending
        total in the same step, because their acquisition's outcome is being
        published. A durable Run commits the rows in one SQLite transaction. The
        live byte counters change only after that commit. A failed commit sets
        ``storage_failed`` so later work stops until the Run is reopened from disk.
        Only cache rows and removed checkpoint fields may be deleted. Receipts,
        events and observations are permanent evidence. metadata_credit is the
        part of a completion's metadata that its reservation already funded,
        the declared host application receipts, so only the rest is checked
        against max_completion_metadata_bytes.

        Returns:
            The encoded ``(kind, key, text)`` rows that were written.
        """
        state = self._state
        data_limit = self.limits.max_data_bytes if max_data_bytes is None else max_data_bytes
        released = set(release_output) if isinstance(release_output, tuple) else {release_output}
        pending = sum(value for key, value in state["pending_output"].items() if key not in released)
        # Published arrays are charged through array_bytes by their publication
        # batch. Counting their payload here as well would charge them twice.
        payload_sizes = {reference.content_id: reference.bytes for reference, _ in payloads
                         if reference.format not in ARRAY_PAYLOAD_FORMATS}
        payload_added = sum(size - state["payload_sizes"].get(key, 0) for key, size in payload_sizes.items())
        keys = {(kind, key) for kind, key, _ in records}
        removed = set(deleted)
        if len(removed) != len(deleted) or keys & removed:
            raise ValueError("duplicate row or upsert/delete conflict in an atomic transition")
        if any(kind not in {"cache", "checkpoint_field"} for kind, _ in removed):
            raise ValueError("only cache rows and checkpoint fields can be deleted from a run journal")
        # A chunk row stores its readout through its receipt
        # (_run_journal._row_value), so the two must declare the same readout.
        # A point chunk holds its point's one-point readout of the receipt's trajectory.
        for kind, _, value in records:
            if kind == "chunk" and isinstance(value, ObservationChunk):
                receipt = state["prepared_artifacts"].get(value.prepared_id)
                if receipt is None or not value.declares_readout_of(receipt):
                    raise ValueError("an observation chunk must declare its preparation receipt's readout")
        replaced_sizes = {key: state["data_sizes"].get(key, 0) for key in keys | removed}
        replaced = sum(replaced_sizes.values())
        counted = (state["data_bytes"] - replaced + state["array_bytes"] + state["payload_bytes"]
                   + state["external_bytes"] + pending)
        available = data_limit - (counted + payload_added + reserve_output)
        owner = "this Run's" if max_data_bytes is None else "the proposed"
        if release_output is not None:
            try:
                completed, metadata = _completion_metadata_bound(records, data_limit)
                metadata -= _probability_header_credit(records, data_limit, run=self)
            except JSONBytesExceeded as error:
                raise ValueError(f"one completion record needs more than {owner} max_data_bytes={data_limit} "
                                 f"by itself. {_limit_remedy('max_data_bytes')}") from error
            allowance = self.limits.max_completion_metadata_bytes
            if completed and metadata - replaced - metadata_credit > completed*allowance:
                # Raising the limit does not clear a synchronous acquisition's
                # stored terminal failure (_raise_terminal_failure re-raises it
                # and does not reacquire the discarded output), so the open-Run
                # remedy is not offered for it.
                raise ValueError(f"completion metadata exceeds max_completion_metadata_bytes={allowance} per "
                                 f"completed acquisition ({metadata - replaced - metadata_credit} bytes for "
                                 f"{completed} completed, {completed*allowance} allowed). A synchronous result "
                                 "refused here is discarded; acquiring it again needs a new Run with "
                                 "limits=ExecutionLimits(max_completion_metadata_bytes=...) in solve, prepare "
                                 "or Run. A detached job whose output is still retrievable can be refreshed "
                                 "again after run.extend_limits(max_completion_metadata_bytes=...).")
        # A reservation or payload beyond the limit leaves negative space. The
        # encoder takes only a nonnegative limit, so zero space refuses every
        # row with the message below, and a write without rows reaches the
        # final comparison.
        try:
            encoded = encode_records(records, max(0, available))
        except JSONBytesExceeded as error:
            minimum = payload_added + reserve_output + max(0, available) + 1
            raise _cumulative_limit_error(
                "max_data_bytes", data_limit, "serialized-data admission bytes (lower bound)", counted, minimum,
                owner=owner) from error
        sizes = {(kind, key): size for (kind, key, _), size in zip(encoded.rows, encoded.sizes, strict=True)}
        added = sum(sizes.values()) - replaced
        requested = sum(sizes.values()) + payload_added + reserve_output
        if counted + requested > data_limit:
            raise _cumulative_limit_error("max_data_bytes", data_limit, "stored data bytes", counted, requested,
                                          owner=owner)
        journal = state["journal"]
        if journal is not None:
            try:
                journal.commit(records, collected=collected, payloads=payloads,
                               deleted=deleted, encoded=encoded, replaced=replaced_sizes)
            except BaseException:
                state["storage_failed"] = True
                raise
        for key in removed:
            state["data_sizes"].pop(key, None)
        state["data_sizes"].update(sizes)
        state["data_bytes"] += added
        state["payload_sizes"].update(payload_sizes)
        state["payload_bytes"] += payload_added
        for key in released:
            state["pending_output"].pop(key, None)
        return encoded.rows

    def progress(self, stage, done=None, total=None):
        """Report a lifecycle stage to the observer callback, if any.

        An exception raised by the callback disables the callback and is recorded
        in ``Run.warnings``. A display error therefore cannot interrupt an
        acquisition whose reservation has already committed.
        """
        callback = self._state["progress"]
        if callable(callback):
            try:
                callback(stage, done, total)
            except Exception as error:
                self._state["warnings"].append(f"progress callback disabled: {type(error).__name__}: {error}")
                self._state["progress"] = False

    @property
    def warnings(self):
        """Notices recorded in this process, such as a disabled progress callback or a backend's notice.

        Reading them repeats no work. They are not saved with the Run.
        """
        return tuple(self._state["warnings"])

    def _method_context(self, factory, *, marks_changes=False):
        """The live method cache. Each access marks a non-dictionary context for saving at
        the next journal write, unless the caller marks its own changes with cache_method_dirty."""
        with self._lock:
            self._ensure_open()
            if self._state["method_context"] is None:
                context = factory()
                if isinstance(context, dict):
                    from nwqlib._run_archive import _CacheEntries
                    context = _CacheEntries(context, prefix=("method",), changes=self._state["cache_changes"])
                self._state["method_context"] = context
                self._state["cache_method_dirty"] = True
            if not marks_changes and not isinstance(self._state["method_context"], dict):
                self._state["cache_method_dirty"] = True
            return self._state["method_context"]

    def _backend_notice(self):
        """Warn once per Run about the backend's qualification scope.

        ``notice_shown`` lives in the backend context, which the frontier cache row
        saves at the next cache write, so a Run reopened after that write does not
        repeat the warning.
        """
        message = getattr(self.backend, "qualification_notice", None)
        if message and not self._state["backend_context"].get("notice_shown"):
            import warnings
            warnings.warn(message, UserWarning, stacklevel=3)
            self._state["backend_context"]["notice_shown"] = True

    def check_capacity(self, *, preparations=0, circuits=0, shots=0, data_bytes=0):
        """Admit new work against the cumulative caps before it starts.

        Circuit preparations and circuit acquisition attempts are two separate
        populations, each capped by ``max_total_circuits``. Host preparations and
        host invocations carry their own work counters and are excluded. Shots and
        attempts include every reserved attempt, completed, uncertain or failed, so
        spent exposure never frees capacity. A cancellation request blocks new
        preparation or acquisition.
        """
        values = (preparations, circuits, shots, data_bytes)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("new work requires nonnegative integer counts")
        self._ensure_open()
        state = self._state
        if self.cancel_requested is not None and any((preparations, circuits, shots)):
            raise ValueError("run cancellation prevents new preparation or acquisition")
        limits = self.limits
        circuit_preparations = state["preparations"] - state["host_preparations"]
        if circuit_preparations + preparations > limits.max_total_circuits:
            raise _cumulative_limit_error("max_total_circuits", limits.max_total_circuits, "circuit preparations",
                                          circuit_preparations, preparations)
        acquisitions = len(state["events"]) - state["host_invocations"]
        if acquisitions + circuits > limits.max_total_circuits:
            raise _cumulative_limit_error("max_total_circuits", limits.max_total_circuits,
                                          "circuit acquisition attempts", acquisitions, circuits)
        if state["shots"] + shots > limits.max_total_shots:
            raise _cumulative_limit_error("max_total_shots", limits.max_total_shots, "raw shots",
                                          state["shots"], shots)
        stored = (state["data_bytes"] + state["array_bytes"] + state["payload_bytes"] + state["external_bytes"]
                  + sum(state["pending_output"].values()))
        if stored + data_bytes > limits.max_data_bytes:
            raise _cumulative_limit_error("max_data_bytes", limits.max_data_bytes, "stored data bytes",
                                          stored, data_bytes)

    def _available_data_bytes(self):
        """Bytes of ``max_data_bytes`` not charged by stored rows, arrays, payloads, external data or pending output."""
        state = self._state
        stored = (state["data_bytes"] + state["array_bytes"] + state["payload_bytes"] + state["external_bytes"]
                  + sum(state["pending_output"].values()))
        return max(0, self.limits.max_data_bytes - stored)

    def _charge_synthesis(self, work, operation):
        """Reserve ``work`` units of exact dense synthesis for the preparation in progress, before the synthesis starts.

        Backends that lower a circuit to a gate basis, through
        :meth:`_exact_dense_unitaries`, and lowering that controls a
        transformed block call this once per batch of syntheses.
        ``ExecutionLimits.max_synthesis_work`` counts the total over the
        Run, so a synthesis that a reopened Run makes again, with its empty
        cache, is counted again. The reservation revises the preparation's
        charge in one journal commit, so a reopened Run keeps it charged
        even when the synthesis never finished.

        When the Run's total would exceed the limit, the charge instead
        records the refused work, and the same commit removes the
        preparation from its static workflow item or controller checkpoint.
        The refused preparation stays counted against ``max_total_circuits``,
        because its lowering had started, but its experiment is left
        unprepared, so the workflow or controller prepares it again at its
        next call. ``ValueError`` is raised before any synthesis starts.

        A backend called directly with this Run, outside ``prepare_experiment``,
        makes no preparation of the Run, so its work is compared with the
        remaining allowance and not recorded, like the rest of that call. Its
        syntheses stay out of the Run's synthesis cache
        (:meth:`_exact_dense_unitaries`).
        """
        state = self._state
        identity = state["synthesis_preparation"]
        limit = self.limits.max_synthesis_work
        if identity is None:
            if state["synthesis_work"] + work > limit:
                raise ValueError(f"{operation} needs {work} work units, beyond the rest of this Run's "
                                 f"max_synthesis_work={limit}, of which {state['synthesis_work']} are already "
                                 f"reserved. {_limit_remedy('max_synthesis_work')}")
            return
        charge = state["preparation_charges"][identity]
        if state["synthesis_work"] + work <= limit:
            revised = charge.revise(synthesis_work=charge.synthesis_work + work)
            self._write((("preparation", identity, revised),))
            state["preparation_charges"][identity] = revised
            state["synthesis_work"] += work
            return
        refused = charge.revise(synthesis_refused=work)
        records = [("preparation", identity, refused)]
        current = state["workflow_current"]
        item = None if current is None else state["workflow_items"][current]
        if item is not None and item.preparation_id == identity:
            item = item.revise(preparation_id=None)
            records.append(("workflow_item", current, item))
        else:
            item = None
        self._write(tuple(records))
        state["preparation_charges"][identity] = refused
        if item is not None:
            state["workflow_items"][current] = item
        if state["checkpoint_preparations"].get(charge.checkpoint_sequence) == identity:
            del state["checkpoint_preparations"][charge.checkpoint_sequence]
        raise ValueError(f"{operation} needs {work} work units, and this Run has already reserved "
                         f"{state['synthesis_work']} of its max_synthesis_work={limit}. "
                         f"{_limit_remedy('max_synthesis_work')} After run.extend_limits, the next "
                         "call on the open Run prepares the experiment again.")

    def _exact_dense_unitaries(self, circuit):
        """Return ``circuit`` with its dense unitaries synthesized exactly, each distinct matrix once per Run object.

        The backends that lower a circuit to a gate basis call this before they
        compile. During a preparation of this Run it passes the Run's synthesis
        cache and :meth:`_charge_synthesis` to
        ``qiskit_compat.exact_dense_unitaries``. The Run keeps the circuit of
        every synthesis that its preparations made, keyed by the content of its
        matrix, so a later preparation that holds the same matrix, even in new
        gate objects, reuses the circuit. Only the matrices missing from the
        cache are reserved through :meth:`_charge_synthesis` and synthesized.
        The cache lives until this Run object is closed. Reopening the Run
        starts an empty cache, and a synthesis made again after reopening is
        reserved again.

        A backend called directly with this Run, outside its preparations,
        reserves nothing, so its call neither reads nor adds to the cache and
        every cached circuit stays reserved against ``max_synthesis_work``.
        Its syntheses are still compared with the rest of that limit.
        """
        from nwqlib.subroutines.qiskit_compat import exact_dense_unitaries
        state = self._state
        cache = {} if state["synthesis_preparation"] is None else state["synthesis_cache"]
        return exact_dense_unitaries(circuit, charge=self._charge_synthesis, cache=cache)

    def extend_limits(self, **changes):
        """Raise one or more limits of this Run, keeping everything it has counted.

        `run.extend_limits(max_total_shots=128)` sets the Run's total shot
        limit to 128. It does not add 128 shots, reset any count or change a
        scientific setting, so raising a limit gives back no work already
        counted. Limits can only be kept or raised. A call that raises a limit
        records a [`LimitAmendment`][nwqlib.execution.LimitAmendment] in
        `limit_amendments`. The new limits and that record are saved
        together, so a process that stops before the save reopens with the
        old limits, and one that stops after it reopens with the new limits
        and their record. A call that raises nothing records nothing.

        Args:
            **changes (int): New values of
                [`ExecutionLimits`][nwqlib.execution.ExecutionLimits] fields,
                by name, as integers at least the current values.

        Returns:
            run (Run): This Run.

        Raises:
            ValueError: If no limit is named, a name is not a limit, or a
                value is not an integer at least the current one.
        """
        # The new ``limits`` row and one ``LimitAmendment`` row commit together.
        with self._lock:
            self._ensure_open()
            allowed = {name for name in ExecutionLimits.model_fields if name not in {"parent_id", "schema_version"}}
            if not changes:
                raise ValueError("extend_limits requires named execution limits")
            previous = self.limits
            fields = {name: getattr(previous, name) for name in allowed}
            fields.update(changes)
            limits = ExecutionLimits(**fields)
            if any(type(value) is not int or value < getattr(previous, name) for name, value in changes.items()):
                raise ValueError("extend_limits can only preserve or increase positive execution limits")
            # A no-op amendment must not consume history or change content identity.
            if all(value == getattr(previous, name) for name, value in changes.items()):
                return self
            state = self._state
            amendment = LimitAmendment(sequence=len(state["limit_amendments"]) + 1,
                recorded_at=datetime.now(timezone.utc), old=previous, new=limits,
                circuit_preparations=state["preparations"] - state["host_preparations"],
                circuit_attempts=len(state["events"]) - state["host_invocations"],
                raw_shots=state["shots"],
                stored_data_bytes=state["data_bytes"] + state["array_bytes"] + state["payload_bytes"] + state["external_bytes"],
                pending_output_bytes=sum(state["pending_output"].values()), synthesis_work=state["synthesis_work"])
            # Write the new cap and one history row atomically under the new byte cap.
            # If storage fails, restore the previous limit before exposing any change.
            journal = state["journal"]
            if journal is not None:
                journal.max_bytes = limits.max_data_bytes
            try:
                self._write((("limits", "current", limits),
                    ("limit_amendment", str(amendment.sequence), amendment)),
                    max_data_bytes=limits.max_data_bytes)
            except BaseException:
                if journal is not None:
                    journal.max_bytes = previous.max_data_bytes
                raise
            state["limits"] = limits
            state["limit_amendments"].append(amendment)
            store = state["artifacts"]
            if store is not None:
                store._set_limit(limits.max_data_bytes)
            return self

    @property
    def checkpoint_sequence(self):
        return self._state["checkpoint_sequence"]

    @property
    def checkpoint_state(self):
        """The last committed controller state as a dictionary, or None before any checkpoint."""
        fields = self._state["checkpoint_fields"]
        return None if fields is None else {name: json.loads(text) for name, text in fields.items()}

    def _controller_text(self):
        """The committed checkpoint as one JSON document, joined from its field rows."""
        fields = self._state["checkpoint_fields"]
        if fields is None:
            return None
        body = ",".join(json.dumps(name, ensure_ascii=False) + ":" + text for name, text in fields.items())
        return '{"sequence":' + str(self.checkpoint_sequence) + ',"state":{' + body + "}}"

    def checkpoint(self, value, *, changed=None):
        """Publish a controller state and the RNG position in one transition.

        Each top-level field is its own journal row. changed names the fields
        that differ from the previous checkpoint, and only those are encoded
        and written, so a checkpoint costs its changed data rather than the
        whole state. The caller guarantees that every other field is unchanged.
        None writes every field.

        The checkpoint header, the changed field rows, deletions of removed fields,
        the RNG snapshot and the changed native and method cache rows commit in one
        transaction. Controller decisions consume RNG draws, so saving the RNG
        position with the decision lets a reopened controller continue from the
        exact draw that followed it. If the write fails, the previous checkpoint
        stays current and the cache dictionaries are restored
        (``_run_archive.write_caches``).

        For an adaptive controller this is commit 1 of the module docstring. A
        preparation charge made while this checkpoint is current records its
        sequence number, and the controller passes the same number to its
        submission. A reopened controller therefore finds that work through
        ``preparation_for_checkpoint`` and ``attempt_for_checkpoint`` instead of
        starting it again.
        """
        with self._lock:
            self._ensure_open()
            from nwqlib._run_journal import _json_bound

            def admit(state):
                """Refuse a checkpoint state larger than the whole data limit before encoding it."""
                limit = self.limits.max_data_bytes
                try:
                    _json_bound(state, limit)
                except JSONBytesExceeded as error:
                    raise ValueError(f"the controller checkpoint needs more than this Run's max_data_bytes={limit} "
                                     f"by itself. {_limit_remedy('max_data_bytes')}") from error

            if isinstance(value, Record):
                admit(value)
                value = value.model_dump(mode="json", exclude_computed_fields=True)
            if type(value) is not dict or any(type(name) is not str for name in value):
                raise TypeError("a checkpoint state is a Record or a dictionary with string keys")
            previous = self._state["checkpoint_fields"] or {}
            names = tuple(value) if changed is None else tuple(changed)
            if changed is not None and (any(type(name) is not str for name in names)
                    or len(set(names)) != len(names) or set(names) - set(value) - set(previous)
                    or (set(value) ^ set(previous)) - set(names)):
                raise ValueError("changed must name distinct checkpoint fields, including every added or removed one")
            # Admit the changed fields before the RNG snapshot or any cache work.
            admit({name: value[name] for name in names if name in value})
            sequence = self.checkpoint_sequence + 1
            records = (("checkpoint", "current", dict(sequence=sequence, fields=list(value))),
                       *(("checkpoint_field", name, value[name]) for name in names if name in value),
                       ("rng", "current", self.rng.snapshot()))
            from nwqlib._run_archive import write_caches
            encoded = write_caches(self, records,
                deleted=tuple(("checkpoint_field", name) for name in previous if name not in value))
            written = {key: text for kind, key, text in encoded if kind == "checkpoint_field"}
            fields = {name: written[name] if name in written else previous[name] for name in value}
            self._state.update(checkpoint_fields=fields, checkpoint_sequence=sequence)
            return sequence

    def attempt_for_checkpoint(self, sequence):
        """The acquisition a controller started under checkpoint ``sequence``, or None.

        Each checkpoint can name at most one attempt. A reopened controller looks
        its attempt up here and retrieves it, rather than starting a second one.
        """
        index = self._state["checkpoint_attempts"].get(sequence)
        return None if index is None else self._state["events"][index]

    def preparation_for_checkpoint(self, sequence):
        """The latest preparation charged while checkpoint ``sequence`` was current, or None."""
        return self._state["checkpoint_preparations"].get(sequence)

    def attempt_record(self, attempt):
        index = self._state["attempt_indices"].get(attempt)
        if index is None:
            raise ValueError("unknown acquisition attempt")
        return self._state["events"][index]

    def prepared_artifact(self, identity):
        try:
            return self._state["prepared_artifacts"][identity]
        except KeyError:
            raise ValueError("unknown prepared receipt") from None

    @property
    def prepared_artifacts(self):
        return tuple(self._state["prepared_artifacts"].values())

    def completed_observation(self, attempt):
        """The published chunk of ``attempt``, whether or not the Method has collected it.

        For a trajectory acquisition it is the tuple of its point chunks in
        schedule order.
        """
        try:
            return self._state["by_attempt"][attempt]
        except KeyError:
            raise ValueError("attempt has no completed observation") from None

    @property
    def observations(self):
        """The observations that the Method has collected so far, in the order it collected them.

        An `ObservationView`. Its `chunks` are the
        [`ObservationChunk`][nwqlib.execution.ObservationChunk] records, each
        holding the statistics of one measurement or of one trajectory point.
        A completed measurement that the Method has not yet
        collected is absent. The same object is returned until the next
        collection.
        """
        state = self._state
        chunks, cached = state["chunks"], state["observation_view"]
        if cached is None or cached[0] is not chunks or cached[1] != len(chunks):
            cached = state["observation_view"] = (chunks, len(chunks), ObservationView(chunks=tuple(chunks.values())))
        return cached[2]

    def collect(self, chunk):
        """Mark one completed observation as consumed by the Method, once.

        Completion and collection are separate events. The journal gives each
        collected chunk the next positive ordinal, and ``observations`` lists
        chunks in that order. A reopened Run therefore shows a controller its
        observations in the order it consumed them, even when they completed in
        another order. This is commit 7 of the module docstring. Returns False,
        and writes nothing, for a repeated call.
        """
        with self._lock:
            self._ensure_open()
            actual = self._state["receipts"].get(chunk.acquisition_key)
            if actual is None or actual != chunk:
                raise ValueError("observation differs from this run's completed acquisition")
            fresh = chunk.acquisition_key not in self._state["chunks"]
            if fresh:
                self._write(collected=(chunk.content_id,))
                self._state["chunks"][chunk.acquisition_key] = actual
                self.progress("collect", len(self._state["chunks"]), None)
            return fresh

    def _acquisition(self, attempt):
        event = self.attempt_record(attempt)
        return self._state["submissions"][event.submission], self._state["submission_items"][attempt]

    def validate_observation(self, chunk):
        """Check that a quantum chunk names its own run, attempt, preparation, job and result key.

        The join goes through the chunk's submission item
        (``SubmissionItem.validate_observation``), so data returned for one item
        cannot be attached to another. Host chunks have no submission.
        """
        if chunk.execution != "host_kernel":
            submission, item = self._acquisition(chunk.attempt)
            item.validate_observation(chunk, run_id=self.run_id, locator=submission.locator)

    def _index_counts(self, *, receipt=None, chunk=None):
        """Update derived identities only after their canonical write succeeds.

        Invalid completed data remain durable raw evidence. Remember the error
        for the next admission, rather than failing after an atomic publication
        and rewriting its completed event as an uncertain acquisition.
        """
        state = self._state
        sources = state["counts_sources"]
        if sources is None or state["counts_source_failure"] is not None:
            return
        try:
            if receipt is not None and receipt.observation.kind == "counts":
                sources.reserve_preparation(receipt.content_id)
            if chunk is not None and chunk.observation.kind == "counts":
                sources.require_independent(chunk)
        except ValueError as error:
            state["counts_source_failure"] = str(error)

    def admit_count_preparations(self, prepared, *, require_all=False):
        """Admit new likelihood inputs using this Run's incremental source index.

        A producer with ``fixed_seed`` sampling resets its stream to the same seed
        on every call, so two count populations from one such stream are the same
        sample, not independent ones. The index marks every stream that a committed
        reservation has used, including uncertain and uncollected attempts, and
        rejects a new setting that would reuse one. Only source keys are kept,
        bounded by the already admitted receipts and acquisitions. The index is not
        serialized. A reopened Run rebuilds it once on its first count admission,
        and later commits update it. The current batch is checked without reserving
        a stream before its commit.
        """
        from nwqlib._counts import CountsSources

        state = self._state
        if state["counts_sources"] is None:
            state["counts_sources"] = CountsSources(
                self.prepared_artifact, self._acquisition, max_bytes=self.limits.max_data_bytes)
            for event in state["events"]:
                self._index_counts(receipt=self.prepared_artifact(event.prepared_id))
            for chunk in state["receipts"].values():
                self._index_counts(chunk=chunk)
        sources = state["counts_sources"]
        sources.max_bytes = self.limits.max_data_bytes
        if state["counts_source_failure"] is not None:
            raise ValueError(state["counts_source_failure"])
        streams = set()
        for handle in prepared:
            if handle.record.observation.kind != "counts":
                if require_all:
                    raise ValueError("sampling accuracy requires actual raw-count preparations")
                continue
            sources.admit_preparation(handle.record.content_id)
            source = sources._preparation(handle.record.content_id)
            if source.kind == "fixed_seed":
                if source.stream in streams:
                    raise ValueError("one fixed sampling stream cannot supply independent likelihood settings")
                streams.add(source.stream)

    def _install_attempt(self, event):
        """Add one already committed attempt to the live counters.

        Called after a reservation commits (``_begin_submission`` and
        ``_submit_host``) and while ``_restore`` reads the journal, so a reopened
        Run charges exactly the attempts that were durably reserved. The attempt's
        output reservation stays pending until its outcome is published or it fails
        terminally.
        """
        state = self._state
        if event.attempt in state["attempt_indices"]:
            raise ValueError("duplicate acquisition attempt")
        sequence = event.checkpoint_sequence
        if sequence is not None:
            if sequence in state["checkpoint_attempts"] or not 1 <= sequence <= self.checkpoint_sequence:
                raise ValueError("attempt requires its original unused controller checkpoint")
            state["checkpoint_attempts"][sequence] = len(state["events"])
        state["attempt_indices"][event.attempt] = len(state["events"])
        state["events"].append(event)
        state["shots"] += event.shots
        state["host_invocations"] += event.host_invocations
        state["host_work"] += event.host_work
        state["pending_output"][event.attempt] = event.data_bytes_reserved

    def _begin_submission(self, submission, events, prepared):
        """Atomically reserve a complete submission and its prepared acquisition inventory.

        The submission intent, one reserved event per item, any receipt not yet in
        this Run and the static workflow join commit together with the items'
        output-byte reservation. When a forecast is attached, each event is bound to
        its original point assessment here, before native work, so continuation can
        never attach a later assessment. The caller contacts the backend only after
        this method returns. This is commit 4 of the module docstring.
        """
        state = self._state
        if (submission.submission_id in state["submissions"] or submission.run_id != self.run_id
                or len(events) != len(submission.items) or len(events) != len(prepared)):
            raise ValueError("submission differs from its new complete acquisition inventory")
        self.check_capacity(circuits=len(events), shots=sum(event.shots for event in events),
                            data_bytes=sum(event.data_bytes_reserved for event in events))
        if self.forecast is not None:
            events = tuple(event.revise(assessment_id=self._assessment_id(receipt))
                           for event, receipt in zip(events, prepared, strict=True))
        elif any(event.assessment_id is not None for event in events):
            raise ValueError("an acquisition assessment requires its original forecast")
        records = [("submission", submission.submission_id, submission)]
        recorded = set()
        for event, receipt, item in zip(events, prepared, submission.items, strict=True):
            if (event.status != "reserved" or event.submission != submission.submission_id
                    or event.attempt != item.attempt or event.prepared_id != receipt.content_id
                    or item.prepared_id != receipt.content_id):
                raise ValueError("submission item differs from its original prepared acquisition")
            records.append(("event", event.attempt, event))
            if receipt.content_id not in state["prepared_artifacts"] and receipt.content_id not in recorded:
                records.append(("prepared", receipt.content_id, receipt))
                recorded.add(receipt.content_id)
        current = state["workflow_current"]
        changed = None
        if current is not None:
            if len(events) != 1:
                raise ValueError("one static item requires exactly one acquisition")
            changed = state["workflow_items"][current].revise(attempt=events[0].attempt)
            records.append(("workflow_item", current, changed))
        # Persist all reservations before installing counters in live memory.
        # The caller may contact the backend only after this transition succeeds.
        self._write(tuple(records), reserve_output=sum(event.data_bytes_reserved for event in events))
        if changed is not None:
            state["workflow_items"][current] = changed
        state["submissions"][submission.submission_id] = submission
        state["pending_submissions"][submission.submission_id] = None
        for event, receipt, item in zip(events, prepared, submission.items, strict=True):
            self._install_attempt(event)
            state["prepared_artifacts"][receipt.content_id] = receipt
            state["submission_items"][event.attempt] = item
            self._index_counts(receipt=receipt)
        return tuple((state["attempt_indices"][event.attempt], event) for event in events)

    def _assessment_id(self, receipt):
        """Identity of the forecast's point assessment for this receipt, or None without one."""
        if self.forecast is None:
            return None
        assessment = self.forecast.assessment_for(self.plan, receipt)
        return None if assessment is None else assessment.content_id

    def _ack_submission(self, identity, locator):
        """Record the provider's job locator once. A different second locator rejects.

        This is commit 5 of the module docstring. It closes the window in which a
        lost acknowledgement leaves only ``reconcile`` able to find the job.
        """
        previous = self._state["submissions"][identity]
        if locator.provider != self.backend.kind:
            raise ValueError("backend acknowledgement belongs to another provider")
        if previous.locator is not None and previous.locator != locator:
            raise ValueError("backend acknowledgement differs from its original locator")
        value = previous.revise(status="acknowledged", locator=locator)
        self._write((("submission", identity, value),))
        self._state["submissions"][identity] = value
        return value

    def _finish_attempt(self, index, event, chunk, *, native_simulations=None, metadata_credit=0):
        """Publish one synchronous acquisition's outcome in one transaction.

        This is commit 6 of the module docstring. The completed event, its
        observation chunk, the completed submission with its locator and every
        array the acquisition published commit together, and the attempt's output
        reservation is released in the same write. Live maps change after the
        commit. The counts index learns the chunk last, so an index error cannot
        undo a durable publication. Multi-item batches complete through
        ``refresh_submissions``, which publishes all newly returned items of one
        submission in one transaction. A trajectory acquisition passes the
        tuple of its point chunks, which commit together with its one event.
        """
        chunks = chunk if isinstance(chunk, tuple) else (chunk,)
        records = [("event", event.attempt, event)] + [("chunk", item.content_id, item) for item in chunks]
        submission = None
        if event.submission is not None:
            previous = self._state["submissions"][event.submission]
            if len(previous.items) != 1:
                raise ValueError("batch completion requires atomic per-item publication")
            job = chunks[0].job
            locator = previous.locator or JobLocator(provider=getattr(self.backend, "kind", "external"), job_id=job)
            if any(locator.job_id != item.job for item in chunks):
                raise ValueError("completed observation differs from its original job locator")
            submission = previous.revise(status="completed", locator=locator, finished=event.finished,
                native_simulations=native_simulations, timing=event.timing,
                native_invocations=1 if event.timing is not None else previous.native_invocations, results_consumed=True)
            records.append(("submission", submission.submission_id, submission))
        self._flush_publications(tuple(records), release_output=event.attempt, metadata_credit=metadata_credit)
        self._state["events"][index] = event
        for item in chunks:
            self._state["receipts"][item.acquisition_key] = item
        self._state["by_attempt"][event.attempt] = chunk
        if submission is not None:
            self._state["submissions"][submission.submission_id] = submission
            self._state["pending_submissions"].pop(submission.submission_id, None)
        for item in chunks:
            self._index_counts(chunk=item)

    def _fail_attempt(self, index, event, error, *, discarded=False):
        """Record an acquisition whose call raised, without claiming more than is known.

        Only a host kernel whose invocation ended with an ordinary ``Exception``
        becomes ``failed``. Local deterministic code then has a confirmed outcome
        and no hidden provider work, and its output reservation is released. Every
        other case, including every quantum call and a ``KeyboardInterrupt``,
        becomes ``uncertain``. The backend may have executed or billed the work
        before the error arrived, so the reserved shots stay charged and the output
        reservation stays pending in case a later refresh retrieves the result.
        Unpublished arrays of the outcome are discarded first.

        ``discarded`` names a local simulator whose synchronous call returned
        its result in process, after which decoding or publication refused
        it: the work ran and no later refresh can retrieve the output. Its
        submission becomes failed with its results consumed, which releases
        the output reservation durably (also on reopen). The event stays
        uncertain, the only unsuccessful status of a quantum attempt.

        If the process stops before this write commits, reopening turns the
        reserved event into ``uncertain`` (``_restore``), which is the status this
        method records for every case except a finished host exception.
        """
        self._abort_publications()
        # A failed durable write leaves only the original reservation known.
        # Do not mask that error while trying to publish another event revision.
        if self._state["storage_failed"]:
            return
        terminal = event.execution == "host_kernel" and event.timing is not None and isinstance(error, Exception)
        outcome = event.revise(status="failed" if terminal else "uncertain", finished=_now(),
            observation_id=None, failure=f"{type(error).__name__}: {error}")
        records = [("event", event.attempt, outcome)]
        submission = None
        if event.submission is not None:
            previous = self._state["submissions"][event.submission]
            submission = previous.revise(status="failed" if discarded else "uncertain", finished=outcome.finished,
                results_consumed=discarded or previous.results_consumed,
                failure=outcome.failure, timing=event.timing,
                native_invocations=1 if event.timing is not None else previous.native_invocations,
                native_simulations=getattr(error, "native_simulations", previous.native_simulations))
            records.append(("submission", submission.submission_id, submission))
        self._write(tuple(records), release_output=event.attempt if terminal or discarded else None)
        self._state["events"][index] = outcome
        if submission is not None:
            self._state["submissions"][submission.submission_id] = submission

    @property
    def trace(self):
        """The Run's attempts and counted work as they stand, as an [`ExecutionTrace`][nwqlib.execution.ExecutionTrace].

        It includes failed, uncertain and unused work, so a Result built from
        it accounts for all the work of the Run rather than only the
        measurements it uses. The same `ExecutionTrace` object is returned
        until an attempt, submission, count or limit changes.
        """
        # The comparison key holds the current event and submission records,
        # which a change replaces, and the counters and limits themselves.
        state = self._state
        key = (tuple(state["events"]), tuple(state["submissions"].values()), state["preparations"],
               state["construction_work"], state["host_preparations"], state["host_preparation_work"],
               state["host_invocations"], state["host_work"], state["data_bytes"], state["array_bytes"],
               state["payload_bytes"], state["external_bytes"], state["synthesis_work"],
               tuple(state["local_prepared_ids"]), state["cancel_requested"], state["termination_reason"],
               state["limits"], tuple(state["limit_amendments"]))
        cached = state["trace_snapshot"]
        if cached is not None and cached[0] == key:
            return cached[1]
        trace = self._trace()
        state["trace_snapshot"] = (key, trace)
        return trace

    def _trace(self):
        """Build the ExecutionTrace of the current state (``trace``)."""
        state = self._state
        return ExecutionTrace(run_id=self.run_id, plan_id=self.plan.content_id,
            preparations=state["preparations"], construction_work_reserved=state["construction_work"],
            events=tuple(state["events"]), submissions=tuple(state["submissions"].values()),
            host_preparations=state["host_preparations"], host_preparation_work_reserved=state["host_preparation_work"],
            host_invocations=state["host_invocations"], host_work_reserved=state["host_work"],
            data_bytes_reserved=sum(event.data_bytes_reserved for event in state["events"]),
            data_bytes=state["data_bytes"] + state["array_bytes"] + state["payload_bytes"] + state["external_bytes"],
            synthesis_work_reserved=state["synthesis_work"],
            local_prepared_ids=tuple(state["local_prepared_ids"]),
            cancel_requested=self.cancel_requested, termination_reason=state["termination_reason"],
            limits=self.limits, limit_amendments=self.limit_amendments)

    @property
    def data(self):
        """The Run's measured data as it stands, as a [`RunData`][nwqlib.core.analysis.RunData].

        It holds the collected observations, preparation records, trace and
        saved arrays, shared rather than copied. Its `method_context` is set
        only when the Method keeps analysis data with the Result. Repeated
        reads reuse the same `observations` and `trace` objects.
        """
        store = self._state["artifacts"]
        context = None
        snapshot = getattr(self.plan.method, "snapshot_result_context", None)
        if callable(snapshot) and self._state["method_context"] is not None:
            context = snapshot(self._state["method_context"])
        return RunData(self.observations, self.trace, self.prepared_artifacts,
                       () if store is None else store.snapshot(), self._controller_text(), context,
                       self.forecast, self.allocation)

    @property
    def exposure(self):
        """The work submitted so far, by outcome: completed, reserved, uncertain and failed.

        A dictionary with the keys `"completed"`, `"reserved"`, `"uncertain"`
        and `"failed"`, each mapping `jobs`, `circuits`, `shots` and
        `provider_managed_sampling` to a count. `provider_managed_sampling`
        counts attempts that return a provider estimate whose sampling the
        provider chooses. Their sampling is unknown and is not in `shots`.
        Jobs are counted per submission. Submissions not yet acknowledged by the backend, or
        acknowledged and not finished, are reserved. Failed, cancelled and
        uncertain submissions are all counted as uncertain jobs, because a
        provider's final status does not state how much work it consumed.
        Circuits, shots and provider-managed sampling are counted per attempt,
        by the attempt's status. Only classical computations that the Method
        runs on this computer can fail, so the failed category never contains
        circuits or shots.
        """
        result = {status: dict(jobs=0, circuits=0, shots=0, provider_managed_sampling=0)
                  for status in ("completed", "reserved", "uncertain", "failed")}
        for submission in self._state["submissions"].values():
            status = "completed" if submission.status == "completed" else "reserved" if submission.status in {"intent", "acknowledged"} else "uncertain"
            result[status]["jobs"] += 1
        for event in self._state["events"]:
            result[event.status]["circuits"] += int(event.execution == "quantum_circuit")
            result[event.status]["shots"] += event.shots
            result[event.status]["provider_managed_sampling"] += int(event.provider_managed_sampling)
        return result

    def cancel(self, reason="user requested cancellation"):
        """Stop new work in this Run, then ask the backend to cancel its pending jobs.

        The request is saved first, so a Run reopened after a crash while
        contacting the provider still refuses new preparations and
        measurements. Pending remote preparations and submissions are then
        refreshed, which sends each job at most one cancellation request. A
        request does not prove that a job was cancelled, and only a status
        returned by the provider does. All counted work stays counted. A
        backend without a cancellation operation, such as local NWQ-Sim,
        receives no request.

        Args:
            reason (str): Reason recorded with the request. Default
                `"user requested cancellation"`.

        Returns:
            run (Run): This Run.
        """
        with self._lock:
            self._ensure_open()
            if self.cancel_requested is None:
                self._write((("control", "current", dict(cancel_requested=reason,
                    termination_reason=self._state["termination_reason"])),))
                self._state["cancel_requested"] = reason
            from nwqlib._remote_preparation import refresh_preparation
            for identity, pending in tuple(self._state["remote_preparations"].items()):
                if pending.status not in {"completed", "failed", "cancelled"}:
                    refresh_preparation(identity, run=self)
            if callable(getattr(self.backend, "refresh", None)):
                refresh_submissions(run=self)
            return self

    def terminate(self, *, reason):
        """Journal a Method controller's reason for ending early. No job is cancelled.

        The reason and the current cancellation request share one control row.
        A stop before this commit leaves the previous reason in the journal.
        """
        self._write((("control", "current", dict(cancel_requested=self.cancel_requested, termination_reason=reason)),))
        self._state["termination_reason"] = reason

    def resume(self, *, reanalyze=False):
        """Continue the Run once: retrieve pending results, let the Method go on, and return.

        `resume` reads each pending submission once, then lets the Method
        continue from where the Run stands, without waiting. When the Method
        finishes, `run.result` holds the Result. Otherwise `run.result` is
        still `None`, and a later `resume` or `wait` continues. A complete
        Run returns at once. Work done before an error, such as a computed
        circuit or numerical intermediate, is kept for the next `resume`. A
        Result always accounts for every attempt of the Run, failed and
        uncertain ones included.

        Args:
            reanalyze (bool): Default `False`. `True` retries an interrupted
                analysis of a Method that supports it, from its saved data,
                before continuing. A completed Run is analyzed again with
                `result.analyze(...)` instead.

        Returns:
            run (Run): This Run.

        Raises:
            RunFailed: If no Result is available and an uncertain attempt
                cannot be retrieved by any route (stage `"recovery"`).
            ValueError: If `reanalyze=True` is given for a Run that already
                has its Result.

        Examples:
            A local Aer run completes in its first pass:

            >>> from nwqlib import Expectation, plan, prepare, submit
            >>> from nwqlib.algorithms import ExpectationMethod
            >>> problem = Expectation(state=[1.0, 0.0],
            ...                       observable=[[1.0, 0.0], [0.0, -1.0]])
            >>> selected = plan(problem, method=ExpectationMethod(), shots=64,
            ...                 seed=7)
            >>> prepared = prepare(selected)
            >>> with prepared.run as run:
            ...     started = submit(prepared)
            ...     print(started is run, run.resume().result.value)
            True 1.0
        """
        # Order: an explicit ``reanalyze`` recovery, then one refresh of
        # existing submissions, then the Method's ``execute`` on the same Plan
        # and Run. Changed caches are saved even when ``execute`` raises. A
        # returned Result that the Method already attached must belong to this
        # Plan, and its data must equal the Run's current observations,
        # receipts, forecast, allocation and trace. The stored byte counter is
        # excluded because it keeps growing after the Result was captured. An
        # unattached Result is attached to the Run's current data. Either way a
        # Method cannot publish a Result that omits failed or uncertain
        # exposure. The Result row commits after every observation it uses. If
        # the process stops before that commit, the reopened Run has no
        # Result, and the next ``resume`` calls ``execute`` again on the saved
        # frontier.
        if type(reanalyze) is not bool:
            raise TypeError("reanalyze must be a boolean")
        with self._lock:
            self._ensure_open()
            if self.result is not None:
                if reanalyze:
                    raise ValueError("this run already has its Result; use result.analyze for a new analysis")
                return self
            # Reanalysis recovery is explicit. Ordinary resumption first refreshes
            # existing jobs and then lets the Method continue its saved frontier.
            if reanalyze:
                self.plan.method.recover_analysis(self.plan, run=self)
            if callable(getattr(self.backend, "refresh", None)):
                refresh_submissions(run=self)
            if self.cancel_requested is not None:
                from nwqlib._remote_preparation import refresh_preparation
                for identity, pending in tuple(self._state["remote_preparations"].items()):
                    if pending.status not in {"completed", "failed", "cancelled"}:
                        refresh_preparation(identity, run=self)
            try:
                result = self.plan.method.execute(self.plan, run=self)
            finally:
                if not self._state["storage_failed"]:
                    from nwqlib._run_archive import write_caches
                    write_caches(self)
            # A completed Result must account for this run's observations, preparations
            # and exposure before it becomes the durable published result.
            if result is not None:
                if not isinstance(result, Result):
                    raise TypeError("Method.execute must return its Result or a pending None")
                if result._plan is not None:
                    trace = self.trace
                    if (result.plan is not self.plan or result.data.observations != self.observations
                            or result.data.receipts != self.prepared_artifacts
                            or result.data.forecast != self.forecast or result.data.allocation != self.allocation
                            or any(getattr(result.data.trace, name) != getattr(trace, name)
                                   for name in ExecutionTrace.model_fields if name != "data_bytes")):
                        raise ValueError("Method result omits or changes this run's actual acquisition exposure")
                if result._plan is None:
                    result = result._attach(self.plan, self.data)
                from nwqlib._run_archive import result_record
                self._write((("result", "current", result_record(result, self)),))
                self._state["result"] = result
            else:
                self._raise_unrecoverable_outcome()
            return self

    def _raise_unrecoverable_outcome(self):
        """Expose unavailable outcomes without changing their recorded uncertainty.

        An uncertain attempt without an observation is still retrievable when its
        backend can refresh and the submission has a locator, or the backend can
        reconcile a lost acknowledgement with currently observable progress.
        A capability refusal raises without closing the original retrieval path,
        so an explicit later resume can still collect a late outcome.
        A failed or cancelled submission is left
        to ``_raise_terminal_failure``, except a reopened synchronous Aer
        submission (failure ``_AER_REOPEN_FAILURE``): its failed status records
        only that the in-process result cannot be retrieved, so its uncertain
        attempt raises here like any other. Any other uncertain attempt, including every
        interrupted host kernel, has no route to its outcome, and this raises
        ``RunFailed`` naming that attempt. The event stays uncertain and charged,
        because the error reports missing information, not a failed execution.
        After an explicit cancellation nothing is raised here.
        """
        if self.cancel_requested is not None:
            return
        state = self._state
        refreshable = callable(getattr(self.backend, "refresh", None))
        reconcilable = callable(getattr(self.backend, "reconcile", None))
        for event in state["events"]:
            if event.status != "uncertain" or event.attempt in state["by_attempt"]:
                continue
            submission = state["submissions"].get(event.submission)
            if submission is not None:
                if submission.status in {"failed", "cancelled"} and submission.failure != _AER_REOPEN_FAILURE:
                    continue  # _raise_terminal_failure reports a confirmed failed or cancelled submission.
                if refreshable and submission.locator is not None:
                    continue
                if refreshable and reconcilable:
                    available = getattr(self.backend, "can_reconcile", None)
                    if not callable(available):
                        continue
                    possible = available(submission, run=self)
                    if type(possible) is not bool:
                        raise TypeError("backend can_reconcile must return a bool")
                    if possible:
                        continue
            raise RunFailed(stage="recovery", status="uncertain",
                locator=None if submission is None else submission.locator,
                failure=event.failure if submission is None else submission.failure or event.failure,
                directory=self.directory, trace=self.trace, exposure=self.exposure,
                attempt=event.attempt)

    def wait(self, *, timeout=None, poll_interval=1.):
        """Continue the Run until it has a Result, and return the Result.

        Each pass calls `resume`, which reads every pending provider job once,
        and then sleeps `poll_interval` seconds. A Method may return a valid
        partial Result from the data it has before a failed job is reported.
        A retrieval error propagates and keeps the job's locator (a
        `JobLocator` with the provider, the job identifier and the account
        context, without credentials), so a later `resume` or `wait` can read
        the same job, and an uncertain submission is never resubmitted.

        Args:
            timeout (float | None): Seconds to wait before raising
                `TimeoutError`. `None`, the default, waits without a limit.
            poll_interval (float): Default `1.0`. Seconds between passes. A
                longer interval sends fewer status requests to a provider.
                The default is a polling interval chosen without tuning
                ([Engineering
                constants](../ENGINEERING_CONSTANTS.md#explicit-workflow-and-reference-controls)).

        Returns:
            result (Result): The Run's Result.

        Raises:
            RunFailed: If a preparation, classical computation or submission
                ended in failure or cancellation, or an uncertain attempt cannot
                be recovered.
            TimeoutError: If the Run is still pending after `timeout`. The
                same Run can be continued later.
            RuntimeError: If `cancel` was called on this Run before it
                completed and no failed or cancelled work is confirmed yet. Its
                pending work is saved in its folder.

        Examples:
            See [`submit`][nwqlib.scientist.submit].
        """
        started = monotonic()
        while self.result is None:
            self.resume()
            if self.result is not None:
                break
            self._raise_terminal_failure()
            if self.cancel_requested is not None:
                raise RuntimeError(f"run cancelled; original pending work is saved at {self.directory}")
            if timeout is not None and monotonic() - started >= timeout:
                raise TimeoutError(f"run is still pending; continue the same run at {self.directory}")
            sleep(poll_interval)
        return self.result

    def _raise_terminal_failure(self):
        """Raise ``RunFailed`` for a confirmed terminal outcome that ``wait`` cannot pass.

        Confirmed outcomes are a failed host event, a failed or cancelled remote
        preparation, and a failed or cancelled submission with items that have no
        observation, other than a reopened synchronous Aer submission
        (``_AER_REOPEN_FAILURE``), whose failed status records only failed
        retrieval.
        ``wait`` calls this only after the Method has had its chance to return a
        partial Result from the data it has.
        """
        state = self._state
        for event in state["events"]:
            if event.status == "failed":
                raise RunFailed(stage="execute", status="failed", locator=None,
                    failure=event.failure, directory=self.directory, trace=self.trace,
                    exposure=self.exposure)
        for pending in state["remote_preparations"].values():
            if pending.status in {"failed", "cancelled"}:
                raise RunFailed(stage="prepare", status=pending.status,
                    locator=pending.locator, failure=pending.failure,
                    directory=self.directory, trace=self.trace, exposure=self.exposure)
        for submission in state["submissions"].values():
            # A reopened synchronous Aer submission records failed retrieval, not a
            # confirmed terminal execution (_raise_unrecoverable_outcome owns it).
            if submission.status in {"failed", "cancelled"} and submission.failure != _AER_REOPEN_FAILURE and any(
                    item.attempt not in state["by_attempt"] for item in submission.items):
                raise RunFailed(stage="execute", status=submission.status,
                    locator=submission.locator, failure=submission.failure,
                    directory=self.directory, trace=self.trace, exposure=self.exposure)

    @property
    def artifacts(self):
        """The store of the arrays this Run has saved, created on first use.

        Get the handle of one array with `run.artifacts.get(manifest)`, which
        needs the full `ArtifactManifest`, or take the handles from
        `run.data.artifacts`. Read the values with `handle.array`.
        """
        # The store shares the Run lock and routes each publication through the
        # Run, so an array commits in the same transaction as the observation
        # that produced it. A reopened Run's arrays are registered lazily: the
        # store reads a saved payload from the journal on its first use
        # (``_saved_payload``).
        if self._state["artifacts"] is None:
            from nwqlib.artifacts import ArtifactStore
            store = ArtifactStore(max_bytes=self.limits.max_data_bytes)
            store._lock = self._lock
            store._persistence = (self._reserve_publication, self._finish_publication)
            store._loader = self._saved_payload
            self._state["artifacts"] = store
        return self._state["artifacts"]

    def _saved_payload(self, manifest):
        """The journal reader of a published array's saved bytes, or None when the journal has none.

        Returns ``(read_chunks, chunk_bytes)`` for ``ArtifactStore._hydrate``.
        The block size must reproduce the blocks that ``LocalJournal.commit``
        stored (1 MiB, or one block under a smaller data cap), because
        ``_hydrate`` rejects a block of any other size. The payload is also
        checked against the per-object data cap before it is read.
        """
        reference = self._state["artifact_payloads"].get(manifest.content_id)
        journal = self._state["journal"]
        if reference is None or journal is None:
            return None
        self.check_data(manifest.data_bytes)
        return (lambda: journal.payload_chunks(reference)), min(1024**2, self.limits.max_data_bytes//16*16)

    def _reserve_publication(self, data_bytes, provenance):
        """Admit one array against its in-flight acquisition before copying or hashing it.

        The array must belong to this Plan and Run, arrive inside an open outcome
        batch and fit the output bytes reserved for its attempt. The reservation row
        commits now. The manifest and bytes replace it in the outcome's transaction.
        If the process stops before that transaction, the row has no manifest, and
        ``_restore`` skips it when it rebuilds the array inventory.
        """
        acquisition = provenance["acquisition"]
        if provenance["plan_id"] != self.plan.content_id or acquisition[0] != self.run_id:
            raise ValueError("array publication belongs to another Plan or run")
        if self._state["publication_batch"] is None:
            raise ValueError("array publication requires its acquisition outcome")
        if data_bytes > self._state["pending_output"].get(acquisition[1], 0):
            raise ValueError("array exceeds its selected acquisition output size")
        identity = str(uuid4())
        charge = dict(data_bytes=data_bytes, acquisition=acquisition)
        self._write((("publication", identity, charge),))
        return identity, charge

    def _finish_publication(self, reservation, manifest, owned, *, duplicate):
        """Queue a checked array for the outcome's transaction.

        The binary payload is keyed by the array digest. Two acquisitions that
        produce identical bytes therefore share one stored payload, while each
        keeps its own publication row and manifest with its own provenance.
        """
        identity, charge = reservation
        payload = PayloadRef(format=f"nwqlib.array/{manifest.output.storage_dtype}", bytes=manifest.data_bytes,
                             key=manifest.digest)
        batch = self._state["publication_batch"]
        batch["records"].extend((("publication", identity, dict(charge, manifest=manifest, duplicate=duplicate,
            payload=payload)), ("payload", payload.content_id, payload)))
        batch["payloads"].append((payload, memoryview(owned).cast("B")))
        batch["payload_refs"][manifest.content_id] = payload
        if not duplicate:
            batch["array_bytes"] += manifest.data_bytes

    def _start_publications(self):
        """Open an outcome batch, saving the store state that an abort restores."""
        if self._state["publication_batch"] is not None:
            raise ValueError("an outcome already has an unfinished array publication")
        store = self.artifacts
        self._state["publication_batch"] = dict(records=[], payloads=[], payload_refs={}, array_bytes=0,
            handles=dict(store._handles), store_bytes=store.data_bytes)

    def _abort_publications(self):
        """Discard an unfinished outcome batch and restore the store's handles and byte count.

        Arrays queued in the batch never become visible, because their outcome
        did not commit. The saved store state comes from ``_start_publications``.
        """
        batch = self._state["publication_batch"]
        if batch is not None:
            self.artifacts._handles = batch["handles"]
            self.artifacts._data_bytes = batch["store_bytes"]
            self._state["publication_batch"] = None

    def _flush_publications(self, rows, *, release_output=None, metadata_credit=0):
        """Commit the queued arrays and the outcome rows in one transaction.

        On failure the array byte counter and the store's handles return to their
        state before the batch, so no array becomes visible without its
        observation.
        """
        batch = self._state["publication_batch"]
        if batch is None:
            self._write(rows, release_output=release_output, metadata_credit=metadata_credit)
            return
        records = {(kind, key): (kind, key, value) for kind, key, value in batch["records"]}
        payloads = {reference.content_id: (reference, buffer) for reference, buffer in batch["payloads"]}
        old = self._state["array_bytes"]
        self._state["array_bytes"] += batch["array_bytes"]
        try:
            self._write((*records.values(), *rows), payloads=tuple(payloads.values()), release_output=release_output,
                        metadata_credit=metadata_credit)
        except BaseException:
            self._state["array_bytes"] = old
            self._abort_publications()
            raise
        self._state["artifact_payloads"].update(batch["payload_refs"])
        self._state["publication_batch"] = None

    @classmethod
    def _restore(cls, plan, *, backend, directory, archive_files, selection, saved_limits, progress=None):
        """Open an original frontier without new preparation or acquisition.

        The journal is read in dependency order by the steps called below, each
        named after what it restores. The header and backend come first, then
        the row sizes, the current limits and amendments, and the RNG. The
        receipts, preparation charges, remote preparations and payload sizes
        follow, then the checkpoint, submissions and events, observations,
        workflow and control rows, and array manifests. Current limits precede
        larger rows. Saved observations are joined to their attempts without
        repeating the checks that admitted them when they were created.

        Committed cache rows and the files they reference are then registered
        under the controller lock (``_run_archive.restore_caches``). Unpublished
        files in the reserved cache namespace can fill the cap after an
        interruption, so they are reclaimed before the stored file bytes are
        counted and before the recovery write below. Registration does not
        require an eager read of a lazy payload.

        An interrupted intent may have reached the backend, so its outcome
        becomes uncertain. Reserved events without a known outcome also stay
        charged. Recovery revisions commit together after the saved state is
        restored, in one final write. Every step before that write only reads the journal.
        Failure closes the journal and releases the controller lock.

        selection is the record that _run_archive.load built: the name of the
        selection file that run.json names, and the selection read from it.
        progress is the reopened Run's progress callback, with the meaning it
        has for a new Run (``__init__``).
        """
        path = Path(directory)
        journal = LocalJournal(path / "run.sqlite", saved_limits.max_data_bytes, create=False, restore_limits=True)
        try:
            result = cls._restored_header(plan, backend, path, journal, progress)
            result._restore_sizes(journal)
            result._restore_limits(journal)
            result._restore_rng(journal)
            result._restore_preparations(journal)
            result._restore_checkpoint(journal)
            uncertain = result._restore_submissions(journal)
            result._restore_observations(journal)
            result._restore_workflow(journal)
            result._restore_publications(journal)
            from nwqlib._run_archive import restore_caches
            restore_caches(result, archive_files, selection)
            if uncertain:
                result._write(tuple(uncertain))
            return result
        except BaseException:
            journal.close()
            raise

    @classmethod
    def _restored_header(cls, plan, backend, path, journal, progress=None):
        """Check the one journal header against the Plan and backend, then build the Run from it.

        The header supplies the run identity, the initial limits and the saved
        forecast and allocation. A different Plan or backend configuration
        raises before any other row is read, because the saved acquisitions
        belong to the original selection only. ``backend`` None selects local
        Aer for a quantum Run, as in ``__init__``.
        """
        headers = list(journal.rows("header"))
        if len(headers) != 1:
            raise ValueError("saved run requires its one original header")
        header = headers[0][1]
        if header.get("format") != RUN_FORMAT:
            raise unsupported_run_format(header.get("format"), "saved run journal")
        if header["plan_id"] != plan.content_id:
            raise ValueError("saved run differs from its original selected Plan")
        saved = header["backend"]
        if backend is None and plan.execution == "quantum":
            # A Run on any backend other than the default AerBackend() fails
            # the comparison below and is asked for its saved configuration.
            from nwqlib.backends.connection import AerBackend
            backend = AerBackend()
        configuration = None if backend is None else backend.model_dump(mode="json")
        if configuration != saved:
            if saved is None:
                raise ValueError("saved run was prepared with backend=None (classical execution); "
                                 "pass backend=None or omit backend")
            raise ValueError(f"saved run was prepared with the backend configuration {json.dumps(saved)}; "
                             "pass a backend with that configuration")
        from nwqlib._run_archive import load_provenance
        forecast, allocation = load_provenance(header)
        result = object.__new__(cls)
        initial_limits = ExecutionLimits.model_validate(header["limits"])
        result._initialize(plan, backend, initial_limits, header["run_id"], progress, forecast, allocation)
        result._state.update(directory=path, journal=journal, started=True)
        result._state["backend_context"]["persist_prepared"] = True
        return result

    def _restore_sizes(self, journal):
        """Record each journal row's stored size in ``data_sizes`` and their total in ``data_bytes``.

        The sizes come from the rows' ``size`` column, without reading their text.
        """
        state = self._state
        for kind, key, size in journal.connection.execute("SELECT kind,key,size FROM records"):
            state["data_sizes"][kind, key] = size
            state["data_bytes"] += size

    def _restore_limits(self, journal):
        """Restore the current limits and their amendment chain before larger rows are read.

        The current ``max_data_bytes`` becomes the journal's allowance for the
        rows read after this step. The saved amendments populate the Run's
        limit history.
        """
        state = self._state
        limit_rows = list(journal.rows("limits"))
        if limit_rows:
            state["limits"] = ExecutionLimits.model_validate(limit_rows[0][1])
            journal.max_bytes = self.limits.max_data_bytes
        for _, fields, _ in journal.rows("limit_amendment"):
            state["limit_amendments"].append(LimitAmendment.model_validate(fields))

    def _restore_rng(self, journal):
        """Restore the saved RNG position, so later draws continue the original streams."""
        from nwqlib.core.planning import RandomState
        for _, fields, _ in journal.rows("rng"):
            self._state["rng"] = RandomStreams.restore(RandomState.model_validate(fields))

    def _restore_preparations(self, journal):
        """Restore receipts, preparation charges, remote preparations and payload sizes.

        Identities and charges come from the saved records. No quantum circuit
        is rebuilt and no host preparation is repeated. A charge whose
        ``prepared_id`` is None is a failed or interrupted preparation (commit
        2 of the module docstring), which stays counted against the limits. Its
        reserved synthesis work stays charged too. A charge refused by
        ``max_synthesis_work`` names no checkpoint preparation, because its
        experiment was left unprepared (``Run._charge_synthesis``).
        """
        state = self._state
        for key, fields, _ in journal.rows("prepared"):
            record = PreparedArtifact.model_validate(fields)
            state["prepared_artifacts"][key] = record
        for key, fields, _ in journal.rows("preparation"):
            charge = PreparationCharge.model_validate(fields)
            state["preparations"] += 1
            state["host_preparations"] += int(charge.execution == "host_kernel")
            state["preparation_charges"][key] = charge
            state["synthesis_work"] += charge.synthesis_work
            if charge.checkpoint_sequence is not None and charge.synthesis_refused is None:
                state["checkpoint_preparations"][charge.checkpoint_sequence] = key
            if charge.prepared_id is not None:
                record = state["prepared_artifacts"][charge.prepared_id]
                state["local_prepared_ids"].append(charge.prepared_id)
                state["construction_work"] += record.construction_work_reserved
                state["host_preparation_work"] += record.host_preparation_work_reserved
        for key, fields, _ in journal.rows("remote_preparation"):
            record = PendingPreparation.model_validate(fields)
            state["remote_preparations"][key] = record
        for key, fields, _ in journal.rows("payload"):
            reference = PayloadRef.model_validate(fields)
            if reference.format not in ARRAY_PAYLOAD_FORMATS:
                state["payload_sizes"][key] = reference.bytes
                state["payload_bytes"] += reference.bytes

    def _restore_checkpoint(self, journal):
        """Restore the controller checkpoint, one header naming its fields and one row per field."""
        state = self._state
        headers = list(journal.rows("checkpoint"))
        saved = dict(journal.connection.execute("SELECT key,payload FROM records WHERE kind='checkpoint_field'"))
        if headers:
            _, header, _ = headers[0]
            state["checkpoint_sequence"] = header["sequence"]
            state["checkpoint_fields"] = {name: saved[name] for name in header["fields"]}

    def _restore_submissions(self, journal):
        """Restore submissions and attempts, returning the uncertain revisions not yet written.

        An intent without an acknowledgement (commit 4 of the module docstring)
        may already have reached the backend, so it becomes uncertain instead
        of an unsubmitted fresh request. A reserved event becomes uncertain
        when it is a host kernel, has no saved submission, or belongs to an
        uncertain, failed or cancelled submission, because its outcome is
        unknown when the Run reopens. A later ``reconcile`` and refresh can
        still complete the events of an uncertain submission. Each attempt
        re-enters the live counters through ``_install_attempt``. The returned
        rows are written only after every later step has validated.

        Reopening an interrupted synchronous Aer attempt keeps its execution
        status uncertain and its job, circuit and shot exposure charged. When
        its output existed only in the interrupted process and no durable
        retrieval path remains, the Run releases the output-byte reservation.
        Detached submissions keep that reservation while their original job can
        still supply output. NWQ-Sim can recover a lost launch acknowledgement
        from the original submission UUID even when no job locator was saved.

        The predicate is the concrete ``AerBackend`` route (``submit_experiment``
        takes the synchronous branch when ``supports_synchronous`` is true and
        keeps the result only in process), ``status == "uncertain"`` after the
        intent conversion and ``locator is None``. ``LOCAL_SIMULATORS`` is not
        this predicate: it includes NWQ-Sim, whose detached adapter recovers a
        result by submission UUID without a locator. Such a submission becomes
        failed with ``results_consumed=True`` in one revised row. The event
        restoration below then makes its reserved event uncertain, and
        ``_restore_observations`` removes its pending reservation. The failed
        submission records failed retrieval. It does not establish failed
        execution, refund exposure or retry the quantum call. The capacity
        invariant is::

            D_stored + D_arrays + D_payloads + D_external + sum_pending R_a <= max_data_bytes

        ``R_a`` reserves capacity for a possible future publication. Once the
        original synchronous Aer output is irrecoverable it has no publication
        to fund, so releasing it leaves stored data unchanged and keeps the
        event, circuit, job and shot counters, as the live
        ``_fail_attempt(..., discarded=True)`` path does.
        """
        from nwqlib.backends.connection import AerBackend

        state = self._state
        uncertain = []
        was_synchronous_aer = (type(self.backend) is AerBackend
                               and getattr(self.backend, "supports_synchronous", True) is True)
        for key, fields, _ in journal.rows("submission"):
            submission = SubmissionRecord.model_validate(fields)
            changed = False
            if submission.status == "intent":
                submission = submission.revise(status="uncertain", failure="interrupted before acknowledgement")
                changed = True
            if submission.status == "uncertain" and submission.locator is None and was_synchronous_aer:
                submission = submission.revise(
                    status="failed", results_consumed=True,
                    failure=_AER_REOPEN_FAILURE)
                changed = True
            if changed:
                uncertain.append(("submission", key, submission))
            state["submissions"][key] = submission
            for item in submission.items:
                state["submission_items"][item.attempt] = item
        for key, fields, _ in journal.rows("event"):
            event = ConsumptionEvent.model_validate(fields)
            submission = state["submissions"].get(event.submission)
            if event.status == "reserved" and (event.execution == "host_kernel" or submission is None
                    or submission.status in {"uncertain", "failed", "cancelled"}):
                event = event.revise(status="uncertain", failure="interrupted before durable completion")
                uncertain.append(("event", key, event))
            self._install_attempt(event)
            if event.status in {"completed", "failed"}:
                state["pending_output"].pop(event.attempt, None)
        return uncertain

    def _restore_observations(self, journal):
        """Index each completed observation by its attempt and restore the collection order.

        A quantum observation was checked against its receipt when it was
        decoded and against its submission's job when it was published, and
        every observation committed in one transaction with its completed
        attempt (commit 6 of the module docstring), so reopening indexes it
        without repeating those checks. Each row is still validated as an
        ``ObservationChunk`` (its layout and associations, and for counts the
        count domain), and a probability chunk's arrays are not read.
        Collection is a separate ordinal (commit 7), so the order in which the Method consumed
        observations survives independently of completion order, and
        uncollected receipts stay resumable. A submission stays pending while
        an item lacks an observation and its results are not consumed.
        """
        state = self._state
        collected_chunks, by_attempt_lists, readouts = {}, {}, {}
        for key, fields, collected in journal.rows("chunk"):
            # A chunk row stores its readout through its receipt
            # (_run_journal._row_value), whose rows were restored first. A
            # point chunk takes its point's one-point readout, built once per
            # receipt for all of its points.
            receipt = state["prepared_artifacts"][fields["prepared_id"]]
            observation = receipt.observation
            if fields.get("point") is not None:
                if receipt.content_id not in readouts:
                    readouts[receipt.content_id] = observation.point_observations()
                observation = readouts[receipt.content_id][fields["point"]]
            # A probability chunk reads its arrays through the Run's store on
            # first use. Nothing is hydrated here.
            chunk = ObservationChunk.model_validate(dict(fields, observation=observation))._attach_store(self.artifacts)
            state["receipts"][chunk.acquisition_key] = chunk
            if chunk.point is None:
                state["by_attempt"][chunk.attempt] = chunk
            else:
                by_attempt_lists.setdefault(chunk.attempt, []).append(chunk)
            if collected:
                collected_chunks[key] = chunk
        # A trajectory attempt's point chunks committed together (commit 6).
        # Build a list per attempt while restoring, then freeze it once in
        # schedule order, with each receipt's point order indexed once.
        point_order = {}
        for attempt, chunks in by_attempt_lists.items():
            receipt = state["prepared_artifacts"][chunks[0].prepared_id]
            if receipt.content_id not in point_order:
                point_order[receipt.content_id] = {point.id: slot for slot, point in
                                                   enumerate(receipt.observation.positions)}
            order = point_order[receipt.content_id]
            ordered = [None] * len(order)
            for chunk in chunks:
                slot = order[chunk.point]
                ordered[slot] = chunk
            ordered = tuple(ordered)
            state["by_attempt"][attempt] = ordered
        for key, in journal.connection.execute(
                "SELECT key FROM records WHERE kind='chunk' AND collected>0 ORDER BY collected"):
            chunk = collected_chunks[key]
            state["chunks"][chunk.acquisition_key] = chunk
        state["pending_submissions"] = {key: None for key, value in state["submissions"].items()
            if not value.results_consumed and any(item.attempt not in state["by_attempt"] for item in value.items)}
        for submission in state["submissions"].values():
            if submission.results_consumed:
                for item in submission.items:
                    state["pending_output"].pop(item.attempt, None)

    def _restore_workflow(self, journal):
        """Restore the static workflow rows (commit 1) and the cancellation or termination control row."""
        state = self._state
        for key, fields, _ in journal.rows("workflow_item"):
            item = WorkflowItem.model_validate(fields)
            state["workflow_items"][key] = item
        for _, control, _ in journal.rows("control"):
            state.update(cancel_requested=control["cancel_requested"], termination_reason=control["termination_reason"])

    def _restore_publications(self, journal):
        """Reopen the manifests of published arrays and charge their bytes."""
        state = self._state
        manifests = {}
        from nwqlib.artifacts import ArtifactManifest
        for _, row, _ in journal.rows("publication"):
            if row.get("manifest") is not None:
                manifest = ArtifactManifest.model_validate(row["manifest"])
                manifests[manifest.content_id] = manifest
                if row.get("payload") is not None:
                    state["artifact_payloads"][manifest.content_id] = PayloadRef.model_validate(row["payload"])
        if manifests:
            self.artifacts._restore(tuple(manifests.values()))
            state["array_bytes"] = sum(manifest.data_bytes for manifest in manifests.values())

    def save(self, path):
        """Copy the Run's current state to a new folder. The open Run continues unchanged.

        Call it while the Run is open, after a Run call has returned. The copy
        holds the Plan, the limits and counts, the random state, the
        preparations, the attempts and the locators of pending jobs, the
        observations, the arrays and any Result. A Run without a folder gets
        one this way. `load_run` reopens the copy. The copy and the original
        name the same provider jobs, so continue only one of them. The copied
        files count against `max_data_bytes`, and a failed copy leaves no new
        folder.

        Args:
            path (str | os.PathLike): New folder to create.

        Returns:
            path (pathlib.Path): The created folder.

        Raises:
            ValueError: If the Run is in the middle of a step, for example
                while an outcome is being saved.
        """
        from nwqlib._run_archive import save
        return save(self, path=path)

    def release_native(self):
        """Release the in-memory Qiskit circuits whose results the Method has all collected, that are not in use and that are saved as QPY.

        For a long Run with a folder, this frees the Qiskit circuits that are
        no longer needed. The preparation records and the order of the
        prepared circuits stay. Inspecting a released circuit reads its saved
        QPY once, and nothing is built, transpiled or submitted again.
        Circuits of other backends, unsaved circuits and circuits still in use
        are kept. Aer is the backend whose circuits are released. The counts
        describe released circuits, not measured memory.

        Returns:
            released (dict): `released`, the content hashes of the released
                preparations, and `kept`, pairs of a content hash and the
                reason it was kept.
        """
        from nwqlib._run_archive import release_native
        with self._lock:
            self._ensure_open()
            return release_native(self)

    def close(self):
        """Close the Run's folder, release its lock and free its working caches. Closing twice is harmless.

        Closing cancels no job. The saved files, the Result, the preparation
        records, the arrays and every count stay, so a later `load_run` or
        inspection sees the same work. `with prepared.run as run:` closes the
        Run at the end of the block.
        """
        with self._lock:
            if not self._state["closed"]:
                self._state["closed"] = True
                if self._state["journal"] is not None:
                    self._state["journal"].close()
                # Scientific snapshots own their immutable arrays separately.
                # Closing releases execution caches, the synthesis cache
                # included, without resetting exposure, the reserved synthesis
                # work, stored-data charges, receipts or the already attached
                # Result.
                self._state.update(method_context=None, handles={}, refresh_cache={}, definitions=({}, {}),
                    counts_sources=None, counts_source_failure=None, synthesis_cache={},
                    specializations={}, backend_context={}, archive_files=None, progress=None,
                    cache_changes={}, cache_rows={}, cache_keys={}, cache_file_refs={}, cache_row_refs={})

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *unused):
        self.close()


@dataclass(frozen=True, init=False, eq=False)
class PreparedHandle:
    """The prepared circuit of one experiment for one backend, with its preparation record.

    Submission uses exactly this circuit, so the circuit that executes is the
    one the preparation record describes. After `Run.release_native` the
    circuit is dropped from memory and read back from the Run's saved QPY
    when needed, without building or transpiling it again. Method authors
    receive handles from the Run's preparation calls. The fields below are
    read-only.

    Attributes:
        record: The preparation record
            ([`PreparedArtifact`][nwqlib.execution.PreparedArtifact]), with
            the readout and the backend.
        realization: The experiment's concrete parameter values and
            construction (`Realization`).
    """

    record: PreparedArtifact
    realization: Realization
    _native: object = field(repr=False)
    _items: int = field(repr=False)
    _setting: str = field(repr=False)
    _bindings: tuple = field(repr=False)
    _saved_native: object = field(repr=False)

    @classmethod
    def _make(cls, record, realization, native, items, setting, bindings):
        value = object.__new__(cls)
        for name, item in (("record", record), ("realization", realization), ("_native", native),
                           ("_items", items), ("_setting", setting), ("_bindings", bindings),
                           ("_saved_native", None)):
            object.__setattr__(value, name, item)
        return value

    def _restore_native(self):
        """Return the native object, reloading it from the saved QPY after ``Run.release_native``.

        The reload reads the circuit file named in the handle's cache row, and the
        backend rebuilds its native object from that circuit and the saved
        metadata (``restore_native_data``). Nothing is lowered or transpiled.
        """
        if self._native is None:
            if self._saved_native is None:
                raise ValueError("prepared handle has no stored native snapshot")
            from nwqlib._choice_archive import ArchiveFiles
            backend, path, name, metadata = self._saved_native
            circuit = ArchiveFiles(path, None).read_circuit(name)
            native = backend.restore_native_data(self.record, circuit, metadata, run=None)
            object.__setattr__(self, "_native", native)
        return self._native

    def inspect_circuit(self):
        """Return a copy of the prepared Qiskit circuit.

        `QuantumCircuit.copy` gives the copy its own instruction list and its
        own operation objects (checked against the installed Qiskit), so
        changing the copy leaves the circuit that executes unchanged.

        Returns:
            circuit (QuantumCircuit): The copy.

        Raises:
            ValueError: If this preparation is a classical computation
                (`execution="host_kernel"`) without a circuit.
        """
        self._restore_native()
        if self.record.execution != "quantum_circuit" or self._native.circuit is None:
            raise ValueError("this prepared handle has no live circuit to inspect")
        return self._native.circuit.copy()

    def inspect_resources(self, *, transpile_options=None, max_operations=100_000, max_bytes=DEFAULT_MAX_BYTES):
        """Count the operations of the prepared circuit, without changing or running it.

        Each top-level instruction of the circuit counts once under its
        operation name, measurements, barriers, simulator saves, `Clifford`
        objects and user-defined gates included, and names are reported as
        they are. Gate definitions and control-flow bodies are not expanded.
        With `transpile_options`, a transpiled copy is counted instead, with
        the simulator saves removed first. The circuit
        that executes is never changed or run.

        Args:
            transpile_options (dict | None): Options for Qiskit's `transpile`.
                `None`, the default, counts the prepared circuit as it is.
            max_operations (int): Default `100_000`. Largest number of
                operations of the circuit, and of its transpiled copy.
            max_bytes (int): Default 10 GB (decimal, `10_000_000_000`).
                Largest size of the known inspection data and options. It
                does not bound the compiler's own memory.

        Returns:
            inventory (dict): `circuit` (a text description of the counted
                circuit), `basis` (the gate basis label), `compiler` (Qiskit
                version and options, or `None`), `operations` (count by name),
                `total_operations`, `num_qubits`, `num_clbits` and `depth`
                (Qiskit's default depth, which skips directives such as
                barriers and simulator saves).

        Raises:
            ValueError: If this preparation has no circuit or a limit is
                exceeded.
        """
        self._restore_native()
        if self.record.execution != "quantum_circuit" or self._native.circuit is None:
            raise ValueError("this prepared handle has no live circuit to inspect")
        from nwqlib.backends.inspection import inspect_circuit_resources
        return inspect_circuit_resources(self._native.circuit, native_basis=self.record.native_basis,
            label=f"prepared circuit {self.record.content_id}",
            transpile_options=transpile_options, max_operations=max_operations, max_bytes=max_bytes)


@dataclass(frozen=True)
class Prepared:
    """The prepared circuits of a Plan and their open Run, before submission.

    [`prepare`][nwqlib.scientist.prepare] returns it. Inspect the circuits
    with `circuits`, `circuit(index)`, `setting_names` and
    `inspect_resources`, then pass it to [`submit`][nwqlib.scientist.submit].
    The Run holds the preparations, the limits, the measured data and
    everything that follows. Close it when done, for example with
    `with prepared.run as run:`. The field below is read-only.

    Attributes:
        run: The open [`Run`][nwqlib._prepared_execution.Run].
    """

    run: Run

    def _circuit_handles(self):
        """The Run's quantum preparation handles in the order they were prepared.

        This order is the circuit index of ``circuits``, ``setting_names`` and
        ``inspect_resources``. Host-kernel preparations have no circuit and are
        skipped.
        """
        self.run._ensure_open()
        return tuple(handle for handle in self.run._state["handles"].values()
                     if handle.record.execution == "quantum_circuit")

    @property
    def circuits(self):
        """Copies of every prepared Qiskit circuit, in index order.

        The index of a circuit is the one that `circuit`, `setting_names` and
        `inspect_resources` use. Each access copies every circuit, and
        `circuit(index)` copies one. Classical computations that the Method
        runs on this computer have no circuit and are skipped.
        """
        return tuple(handle.inspect_circuit() for handle in self._circuit_handles())

    def circuit(self, index=0):
        """Return a copy of one prepared Qiskit circuit.

        Args:
            index (int): Default `0`. Position of the circuit in `circuits`
                and `setting_names`.

        Returns:
            circuit (QuantumCircuit): The copy.

        Raises:
            ValueError: If no circuit has this index. After the default
                `prepare(plan, settings="first")` only the first setting is
                prepared, and `settings="all"` prepares every setting of
                fixed experiments.
        """
        return self._circuit_handle(index).inspect_circuit()

    def _circuit_handle(self, index):
        """The handle at a circuit index, refusing an index outside the prepared circuits."""
        if type(index) is not int or index < 0:
            raise ValueError("prepared circuit index must be a nonnegative integer")
        handles = self._circuit_handles()
        if index >= len(handles):
            raise ValueError(f"no prepared quantum circuit at index {index} (prepared circuits: {len(handles)}). "
                             "For a static Plan, prepare(plan, settings='all') prepares every setting")
        return handles[index]

    @property
    def setting_names(self):
        """The Plan experiment of each prepared circuit, in the index order of `circuits`.

        Entry i is the name (`Experiment.name` in `plan.experiments`) of the
        experiment that circuit i implements. After
        `prepare(plan, settings="all")` the names of fixed experiments are
        those of the Plan's circuit experiments, in Plan order. The default
        `settings="first"` prepares only the setting that the Method submits
        first, and an adaptive Method can prepare one experiment several
        times with different parameter values.
        """
        return tuple(handle.realization.experiment for handle in self._circuit_handles())

    def inspect_resources(self, *, index=0, transpile_options=None, max_operations=100_000, max_bytes=DEFAULT_MAX_BYTES):
        """Count the operations of one prepared circuit, without changing or running it.

        Each top-level instruction of the circuit counts once under its
        operation name, measurements, barriers, simulator saves, `Clifford`
        objects and user-defined gates included, and names are reported as
        they are. Gate definitions and control-flow bodies are not expanded.
        With `transpile_options`, a transpiled copy is counted instead, with
        the simulator saves removed first. The circuit
        that executes is never changed or run.

        Args:
            index (int): Default `0`. Position of the circuit in `circuits`.
            transpile_options (dict | None): Options for Qiskit's `transpile`.
                `None`, the default, counts the prepared circuit as it is.
            max_operations (int): Default `100_000`. Largest number of
                operations of the circuit, and of its transpiled copy.
            max_bytes (int): Default 10 GB (decimal, `10_000_000_000`).
                Largest size of the known inspection data and options. It
                does not bound the compiler's own memory.

        Returns:
            inventory (dict): `circuit` (a text description of the counted
                circuit), `basis` (the gate basis label), `compiler` (Qiskit
                version and options, or `None`), `operations` (count by name),
                `total_operations`, `num_qubits`, `num_clbits` and `depth`
                (Qiskit's default depth, which skips directives such as
                barriers and simulator saves).

        Raises:
            ValueError: If no circuit has this index or a limit is exceeded.
        """
        return self._circuit_handle(index).inspect_resources(transpile_options=transpile_options,
                                                             max_operations=max_operations, max_bytes=max_bytes)


def _check_scope(run, point):
    """Refuse a point from another Plan, and any new work after a cancellation request."""
    run._ensure_open()
    if point.plan_id != run.plan.content_id:
        raise ValueError("selected point belongs to another Plan")
    if run.cancel_requested is not None:
        raise ValueError("run cancellation prevents new work")


def _template_seed(randomness, construction_id):
    """Seed of a backend's native template for one construction, in the uint32 runtime-seed domain.

    It is the first uint32 word that NumPy's ``SeedSequence`` generates for
    the Plan's recorded root entropy with the child path
    ``(*randomness.spawn_key, 2, c)``, where ``c`` is the construction
    identity's SHA-256 digest read as an integer. ``RandomStreams`` spawns the
    method and backend streams as the children 0 and 1 of the same root, so
    this path draws nothing from them: row seeds and the method's draws stay
    as they were. The same Plan and construction give the same seed in every
    Run, before or after reopening.
    """
    import numpy as np
    digest = int(construction_id.removeprefix("sha256:"), 16)
    sequence = np.random.SeedSequence(randomness.entropy, spawn_key=(*randomness.spawn_key, 2, digest))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _refuse_unsupported_target(run, observation):
    """Refuse a readout the backend's target cannot execute, naming the capability."""
    run._backend_notice()
    target = run.backend.target_for(observation)
    # A trajectory also needs the multi-position feature, each point's
    # readout kind, views and each reducer (capabilities.unsupported_readout).
    unsupported = unsupported_readout(target, observation)
    if unsupported is not None:
        name = getattr(target, 'name', getattr(run.backend, 'kind', type(run.backend).__name__))
        if observation.kind == "trajectory" and unsupported == f"readout {observation.kind!r}":
            # A target without trajectories names the schedule, views and reductions.
            observation.reject_unsupported_schedule(f"backend target {name!r}")
        selected = f"selected {unsupported}" if unsupported.startswith("readout ") else unsupported
        raise ValueError(f"backend target {name!r} cannot execute {selected}; "
                         f"supported readouts: {target.readouts}")


def _named_admission_refusal(plan, error, action="preparation"):
    """Name the Method field that sets the refused Program's admission ceiling.

    A Method with a ``max_admission_steps`` field (FixedGCIM,
    ExpectationMethod, LCHS, QLS) sets its Programs' ``max_steps`` from it,
    so the refusal names that field, the refused stage and its counted need
    (``admission_refusal``). ``action`` names the step that checked it: preparation,
    restoration of a saved Run, or loading saved evidence. Another Method's
    refusal is returned unchanged.
    """
    name = type(plan.method).__name__
    if "max_admission_steps" not in type(plan.method).model_fields:
        return error
    return admission_refusal(f"{name}.max_admission_steps", error, subject=f"{action} of the selected {name} Program")


def _admitted_readout(admission, experiment, construction):
    """Read the widths and resolve the observation of an admitted selected Program.

    ``admission`` has run ``admitted()``. The root binding context, one
    width reading per quantum and classical bit register and the observation
    are charged to it, as preparation charges them. Returns
    ``(qwidths, cwidths, setting, observation)``. QLS planning runs the same
    calls to count the preparation's admission work before any charge.
    """
    program = construction.program
    context = admission.expressions(admission.binding_map(program.bindings))
    qwidths = [(reg.name, admission.integer(reg.width, context, "register width")) for reg in program.registers]
    cwidths = [(reg.name, admission.integer(reg.width, context, "classical width"))
               for reg in program.classical if reg.dtype == "bits"]
    setting, observation = experiment._resolved_observation(admission, construction_id=construction.content_id)
    return qwidths, cwidths, setting, observation


def _selected_readout(plan, point, experiment, construction):
    """Admit a point's Program and resolve its register widths, observation and prospective chunk header.

    Returns ``(admission, qwidths, cwidths, setting, observation, header)``.
    ``header`` holds the lowering-independent selected context of the point's
    chunks (experiment, setting, bindings, Plan and realization identities and
    the logical layouts of the resolved registers), which the metadata laws of
    ``_admit_readout`` and ``prepare_static`` price before native work.
    """
    admission = _Admission(construction.program)
    admission.admitted().require_ready()
    qwidths, cwidths, setting, observation = _admitted_readout(admission, experiment, construction)
    header = dict(experiment=point.experiment, setting=setting, bindings=construction.program.bindings,
                  plan_id=plan.content_id, realization_id=point.content_id,
                  quantum_layout=_register_layout(qwidths), classical_layout=_register_layout(cwidths))
    return admission, qwidths, cwidths, setting, observation, header


def prepare_experiment(point, *, run, runtime=None, held=None):
    """Prepare one actual point, saving its backend draw before native work.

    Admission runs from cheap to expensive, so an unsupported request fails before
    any circuit exists. The checks are Program readiness, simulator width,
    readout shape and cardinality, target readout support, the largest direct
    state preparation of the Plan, amplitude materialization size and the
    cumulative circuit cap, in that order, except that a trajectory's target
    support is checked before its readout shape and metadata allowance, so a
    backend that cannot execute it refuses naming the capability. A
    trajectory's selected IR must be one coherent evolution along its
    executed prefix (``admit_pure_trajectory`` with
    ``_selected_body_issues``), checked before its reductions. A trajectory
    also admits the registered host work of its reductions against the
    allowance of the Method's ``reduction_allowance`` hook, which first runs
    the Method's own gate on its private reduction workspace
    (``_method_reduction_allowance``), and the saved simulator states of its
    amplitude and reduction points (``_admit_trajectory_work``). Then,
    unless the caller supplies ``runtime``, the runtime seed is drawn, and the
    preparation charge commits with the RNG position. Only after that commit does
    lowering start, reusing the Run's definition cache, and the backend prepares
    the native circuit. A remote-compiling backend continues in
    ``_remote_preparation``. A local one publishes its receipt through
    ``_finish_native_preparation``. The charge is commit 2 of the module
    docstring.

    ``held`` is the point's ``(Experiment, SelectedConstruction)`` pair
    when the caller already holds it (``prepare_static`` holds its first
    experiment's), so the point is not resolved again. It must name the
    point's experiment and effective bindings.
    """
    _check_scope(run, point)
    plan = run.plan
    if held is None:
        experiment, construction = point._selected_construction(plan)
    else:
        experiment, construction = held
        if experiment.name != point.experiment or construction.program.bindings != point.bindings:
            raise ValueError("the held selection belongs to another point of the Plan")
    try:
        admission, qwidths, cwidths, setting, observation, header = _selected_readout(plan, point, experiment,
                                                                                     construction)
    except AdmissionStepsExceeded as error:
        raise _named_admission_refusal(plan, error) from error
    width = sum(item for _, item in qwidths)
    cwidth = sum(item for _, item in cwidths)
    host = observation.kind == "host_scalars"
    if not host and getattr(run.backend, "kind", None) in LOCAL_SIMULATORS and width > run.limits.max_simulation_qubits:
        remedy = _limit_remedy('max_simulation_qubits')
        if plan.method.descriptor.method == "qhd" and plan.method.encoding == "one_hot":
            k = plan.method.num_grid_points
            if plan.method.boundary == "periodic" and k & (k - 1) == 0:
                binary_width = len(plan.problem.variables) * (k.bit_length() - 1)
                remedy += f' A new QHD Plan with encoding="binary" uses {binary_width} qubits for this periodic grid.'
            else:
                remedy += (' QHD encoding="binary" requires a periodic grid with a power-of-two '
                           'num_grid_points; switching changes the grid or boundary of this computation.')
        raise ValueError(f"selected circuit has {width} qubits, exceeding the Run's "
                         f"max_simulation_qubits={run.limits.max_simulation_qubits} before native preparation. "
                         f"{remedy}")
    # A backend that cannot execute a trajectory refuses with its capability
    # message before the trajectory's metadata allowance is checked.
    if observation.kind == "trajectory":
        _refuse_unsupported_target(run, observation)
    # Resolve the readout items, their payload reservation and backend support
    # before any native preparation or large amplitude materialization.
    items = _admit_readout(observation, width=width, classical_width=cwidth, run=run, header=header)
    if not host:
        if observation.kind != "trajectory":
            _refuse_unsupported_target(run, observation)
        if observation.kind == "trajectory":
            # The selected IR must be one coherent evolution along the
            # executed prefix. The adapter checks its own facts later.
            try:
                selected_length, body_issue, view_issues = _selected_body_issues(construction, admission, observation,
                                                                                 dict(qwidths))
            except AdmissionStepsExceeded as error:
                raise _named_admission_refusal(plan, error) from error
            admit_pure_trajectory(observation, backend_name="selected IR", body_length=selected_length,
                classical_width=cwidth, body_issue=body_issue, view_issues=view_issues)
            _admit_trajectory_work(observation, width=width, run=run,
                                   allowance=_method_reduction_allowance(run, point, observation, width))
            # A view's tail and inverse are definitions of the selected Program.
            defined = {item.id for item in construction.program.definitions}
            for scheduled in observation.positions:
                if scheduled.view is not None and not {scheduled.view.tail, scheduled.view.inverse} <= defined:
                    raise ValueError(f"the readout view of point {scheduled.id!r} names a tail or inverse that is not a "
                                     "definition of the selected Program")
        # A direct state preparation is synthesized during lowering, after the
        # charge below. Checking every native block of the Plan here rejects an
        # over-limit Plan before its first preparation.
        from nwqlib.blocks.selection import SelectedBlock, direct_preparation_amplitudes
        amplitudes = max((direct_preparation_amplitudes(block) for block in plan.blocks
                          if isinstance(block, SelectedBlock)), default=0)
        if amplitudes > run.limits.max_direct_amplitudes:
            raise ValueError(f"selected direct preparation has {amplitudes} amplitudes, exceeding the Run's "
                             f"max_direct_amplitudes={run.limits.max_direct_amplitudes} before native preparation. "
                             f"{_limit_remedy('max_direct_amplitudes')}")
    if observation.amplitudes is not None:
        observation.amplitudes.admit_materialization(max_bytes=run.limits.max_data_bytes,
                                                     max_bytes_source=_data_limit_clause(run))
    run.check_capacity(preparations=int(not host))
    runtime = runtime or RuntimeOptions(seed=run.rng.next_seed())
    identity = str(uuid4())
    charge = PreparationCharge(execution="host_kernel" if host else "quantum_circuit", preparation_id=identity,
                               checkpoint_sequence=run.checkpoint_sequence or None)
    state = run._state
    records = [("preparation", identity, charge), ("rng", "current", run.rng.snapshot())]
    current = state["workflow_current"]
    item = None if current is None else state["workflow_items"][current].revise(preparation_id=identity)
    if item is not None:
        records.append(("workflow_item", current, item))
    # Save the preparation charge and consumed RNG position before execution.
    # An interruption therefore cannot silently reuse or skip a random draw.
    run._write(tuple(records))
    if item is not None:
        state["workflow_items"][current] = item
    state["preparations"] += 1
    state["host_preparations"] += int(host)
    state["preparation_charges"][identity] = charge
    if charge.checkpoint_sequence is not None:
        state["checkpoint_preparations"][charge.checkpoint_sequence] = identity
    run.progress("prepare")
    if host:
        return _prepare_host(point, construction, observation, runtime, charge, run, items, setting)
    from nwqlib.blocks.lowering import _lower_qiskit
    specializations = {}
    selected = {record.content_id for record in construction.selections}
    blocks = tuple(block for block in plan.blocks if block.record.content_id in selected)
    # Lower only the selected reachable blocks, reusing this run's definition
    # cache and giving parameter specializations their own bounded lifetime.
    # 4096 is the lowering owner's default width ceiling. Local simulators also
    # apply the Run's max_simulation_qubits before any register is allocated.
    # The Run's direct-amplitude limit reaches each preparation constructor.
    # Lowering and the backend reserve each exact dense synthesis that they
    # make on this charge (Run._charge_synthesis) before it starts.
    state["synthesis_preparation"] = identity
    try:
        logical = _lower_qiskit(construction, blocks=blocks, definition_cache=state["definitions"],
            specialization_cache=(state["specializations"], specializations), method_context=run._method_context,
            max_direct_amplitudes=run.limits.max_direct_amplitudes, synthesis_charge=run._charge_synthesis,
            max_qubits=min(4096, run.limits.max_simulation_qubits) if getattr(run.backend, "kind", None) in LOCAL_SIMULATORS else 4096)
        owned = logical.circuit
        # The bound logical body is the lowered logical circuit, one operation
        # per bound call, before native lowering expands any of them. A
        # trajectory resolves every point boundary, the end shorthand
        # included, against its length here, before native preparation, and
        # the adapter receives the tuple of resolved boundaries as position.
        body_length = len(owned.data)
        views, tails, checked = {}, [], set()
        if observation.kind == "trajectory":
            if body_length != selected_length:
                # The prefix check above counted the body as lowering emits it.
                raise ValueError(f"the lowered body has {body_length} operations, but its selected IR counts "
                                 f"{selected_length}; the executed prefix was not checked")
            position = observation.boundaries(body_length)
            # Each view's tail and exact inverse are lowered once, onto the
            # body's registers with the same bindings and caches, and the
            # adapter inserts them at the view's boundary. Their logical
            # visits and work join the receipt's.
            for scheduled in observation.positions:
                for definition in () if scheduled.view is None else (scheduled.view.tail, scheduled.view.inverse):
                    if definition not in views:
                        lowered = _lower_qiskit(construction, definition=definition, blocks=blocks,
                            definition_cache=state["definitions"],
                            specialization_cache=(state["specializations"], specializations),
                            method_context=run._method_context,
                            max_direct_amplitudes=run.limits.max_direct_amplitudes,
                            synthesis_charge=run._charge_synthesis, max_qubits=owned.num_qubits)
                        views[definition] = lowered.circuit
                        tails.append(lowered)
                    if (definition, scheduled.view.wires) not in checked:
                        # A single-call tail acts on its ports' wires in port
                        # order, which must be the declared order. A longer
                        # tail acts only on declared wires.
                        circuit = views[definition]
                        calls = [tuple(circuit.find_bit(qubit).index for qubit in item.qubits) for item in circuit.data]
                        placed = (calls[0] == scheduled.view.wires if len(calls) == 1 else
                                  {index for call in calls for index in call} <= set(scheduled.view.wires))
                        if circuit.global_phase != 0 or not placed:
                            raise ValueError(f"the view {definition!r} of point {scheduled.id!r} must act on its "
                                             "declared wires in their declared order, without a top-level phase")
                        checked.add((definition, scheduled.view.wires))
        else:
            position = body_length if observation.position is None else observation.position
            if position > body_length:
                raise ValueError("observation position exceeds the selected circuit")
        snapshot = str(uuid4())
        callback = getattr(run.backend, "prepare_local", None) or run.backend.prepare
        # The template hook (backends/connection.py module docstring) reaches
        # only an adapter that declares ``prepares_from_templates``, so other
        # adapters receive no template arguments.
        hook = {}
        if getattr(run.backend, "prepares_from_templates", False):
            hook = dict(construction_id=construction.content_id, backend_context=state["backend_context"],
                        template_seed=_template_seed(plan.randomness, construction.content_id))
        native = callback(owned, observation=observation, runtime=runtime, position=position,
            source_definitions=tuple(state["definitions"][0].values()) + tuple(state["definitions"][1].values())
                + tuple(state["specializations"].values()) + tuple(specializations.values()), run=run, snapshot=snapshot,
            **hook, **({"views": views} if views else {}))
    except AdmissionStepsExceeded as error:
        raise _named_admission_refusal(plan, error) from error
    finally:
        state["synthesis_preparation"] = None
    charge = state["preparation_charges"][identity]
    fields = dict(plan_id=plan.content_id, realization_id=point.content_id, construction_id=construction.content_id,
        realization=point, runtime=runtime, snapshot=snapshot, observation=observation,
        quantum_layout=tuple(RegisterMap(name=name, bits=bits) for name, bits in logical.quantum_layout),
        classical_layout=tuple(RegisterMap(name=reg.name, bits=tuple(owned.find_bit(bit).index for bit in reg)) for reg in owned.cregs),
        logical=LogicalPreparationReceipt(dynamic_visits=logical.dynamic_visits + sum(item.dynamic_visits for item in tails),
            reserved_work=logical.construction_work + sum(item.construction_work for item in tails),
            defined_selections=tuple(dict.fromkeys(logical.defined_selections
                                                   + tuple(s for item in tails for s in item.defined_selections)))))
    if observation.kind == "trajectory":
        fields.update(body_length=body_length, boundaries=position)
    # The new specializations replace the frontier row at the next cache commit.
    # The committed ones are kept until then, so a failed commit can restore them.
    state.setdefault("cache_previous_specializations", state["specializations"])
    state["specializations"] = specializations
    state["cache_frontier_dirty"] = True
    if callable(getattr(run.backend, "prepare_local", None)):
        from nwqlib._remote_preparation import begin_preparation
        return begin_preparation(native, run=run, reservation=charge, fields=fields, items=items,
                                 setting=setting, bindings=construction.program.bindings)
    return _finish_native_preparation(native, run=run, reservation=charge, fields=fields, items=items,
                                     setting=setting, bindings=construction.program.bindings)


def _finish_native_preparation(preparation, *, run, reservation, fields, items, setting, bindings, remote=None):
    """Publish one completed preparation, its payload and original reservation together.

    This is commit 3 of the module docstring. The charge row gains the receipt
    identity in the same transaction, so a reopened Run can tell a finished
    preparation from an interrupted one (a charge without ``prepared_id``). For a
    remote compilation, the pending record becomes ``completed`` in the same
    transaction. A preparation can record only the template seed that the
    preparation hook supplied to its adapter.
    """
    observation = fields["observation"]
    template_seed = getattr(preparation, "template_seed", None)
    if template_seed is not None and (not getattr(run.backend, "prepares_from_templates", False)
                                      or template_seed != _template_seed(run.plan.randomness, fields["construction_id"])):
        raise ValueError("a preparation records only the template seed that the preparation hook supplied")
    if observation.kind != "counts" and observation.population == "unconditional" and preparation.population != "unconditional":
        raise ValueError("selected unconditional readout has a conditioned or unknown population")
    payload = None if preparation.payload is None else PayloadRef(format=preparation.payload_format,
                                                               bytes=len(preparation.payload), key=str(uuid4()))
    record = PreparedArtifact(**fields, target=preparation.target, compiler=preparation.compiler,
        native_quantum_layout=preparation.quantum_layout, native_classical_layout=preparation.classical_layout,
        logical_to_native=preparation.logical_to_native, native_basis=preparation.native_basis,
        environment=preparation.environment, payload=payload, backend_configuration_id=run.backend.content_id,
        preparation_time=_now(), construction_work_reserved=fields["logical"].reserved_work,
        native_operations=preparation.operations, population=preparation.population, transformation=preparation.transformation,
        counts_sampling=preparation.counts_sampling, provider_options_json=preparation.provider_options_json,
        probability_window_exclusions=preparation.probability_window_exclusions,
        template_seed=template_seed, statevector_roundoff=preparation.statevector_roundoff)
    handle = PreparedHandle._make(record, fields["realization"], preparation.native, items, setting, bindings)
    records = [("preparation", reservation.preparation_id, reservation.revise(prepared_id=record.content_id)),
               ("prepared", record.content_id, record)]
    payloads = ()
    if payload is not None:
        records.append(("payload", payload.content_id, payload))
        payloads = ((payload, preparation.payload),)
    if remote is not None:
        remote = remote.revise(status="completed", prepared_id=record.content_id, failure=None)
        records.append(("remote_preparation", remote.preparation_id, remote))
    current = run._state["workflow_current"]
    item = None if current is None else run._state["workflow_items"][current].revise(prepared_id=record.content_id)
    if item is not None:
        records.append(("workflow_item", current, item))
    # The charge revision, receipt, native payload and, for Aer, the handle's QPY
    # cache row commit in one transaction. The handle enters the change-tracked
    # cache dictionary first only so that its cache row joins this transaction.
    # A failed commit removes it again (finish_caches), and the live receipt maps
    # change only after the commit. A receipt therefore never commits without
    # the native payload the backend supplied for it.
    from nwqlib._run_archive import write_caches
    run._state["handles"][record.content_id] = handle
    write_caches(run, tuple(records), payloads=payloads)
    if item is not None:
        run._state["workflow_items"][current] = item
    run._state["prepared_artifacts"][record.content_id] = record
    run._state["local_prepared_ids"].append(record.content_id)
    run._state["preparation_charges"][reservation.preparation_id] = reservation.revise(prepared_id=record.content_id)
    run._state["construction_work"] += record.construction_work_reserved
    if remote is not None:
        run._state["remote_preparations"][remote.preparation_id] = remote
    run.progress("prepare", len(run._state["local_prepared_ids"]), None)
    return handle


def _publishes_arrays(observation):
    """Whether an acquisition of ``observation`` publishes binary arrays with its outcome.

    Amplitude and probability readouts do, and so does a trajectory with an
    amplitude or probability point. Their arrays commit in the outcome's
    transaction, so the acquisition opens a publication batch first.
    """
    kinds = {"amplitudes", "probabilities"}
    return observation.kind in kinds or any(point.kind in kinds for point in observation.positions)


def submit_experiment(prepared, *, run, checkpoint_sequence=None):
    """Reserve the actual acquisition before the backend call. Never rebuild.

    ``Method.before_submit`` runs once, before the intent, so a Method can reject
    a population it cannot use (for example a reused fixed sampling stream) before
    any exposure is spent. The intent and reserved event then commit (commit 4 of
    the module docstring), the backend runs the saved native object, and the
    outcome is published by ``Run._finish_attempt`` (commit 6) or recorded by
    ``Run._fail_attempt``. A synchronous backend has no commit 5. Host kernels
    and detached backends take their own routes with the same ordering.

    Returns the observation chunk, or for a trajectory the tuple of its point
    chunks in schedule order, all from one acquisition.
    """
    _check_scope(run, prepared.realization)
    prepared._restore_native()
    if prepared.record.execution == "host_kernel":
        return _submit_host(prepared, run=run, checkpoint_sequence=checkpoint_sequence)
    if prepared.record.backend_configuration_id != run.backend.content_id:
        raise ValueError("prepared artifact belongs to another backend configuration")
    if not getattr(run.backend, "supports_synchronous", True):
        return submit_detached((prepared,), run=run, checkpoint_sequences=(checkpoint_sequence,))
    run.plan.method.before_submit(run.plan, (prepared,), run=run)
    observation = prepared.record.observation
    attempt = str(uuid4())
    event = ConsumptionEvent(attempt=attempt, prepared_id=prepared.record.content_id, status="reserved",
        submission=str(uuid4()), shots=observation.shots, evaluations=int(observation.kind != "counts"), started=_now(),
        checkpoint_sequence=checkpoint_sequence, provider_managed_sampling=observation.kind == "estimated_observable",
        data_bytes_reserved=readout_bytes(prepared, metadata_bytes=run.limits.max_completion_metadata_bytes,
            max_bytes=run.limits.max_data_bytes, run=run, attempt=attempt))
    submission = SubmissionRecord(submission_id=event.submission, run_id=run.run_id, backend=prepared.record.target,
        items=(SubmissionItem(attempt=attempt, prepared_id=prepared.record.content_id, item=0),), status="intent", started=event.started)
    index, event = run._begin_submission(submission, (event,), (prepared.record,))[0]
    started = perf_counter()
    result = None
    try:
        run.progress("submit")
        if _publishes_arrays(observation):
            run._start_publications()
        try:
            result = run.backend.submit(prepared._native, submission_id=event.submission, run=run)
        finally:
            event = event.revise(timing=_native_timing(started, f"{type(run.backend).__module__}.{type(run.backend).__name__}.submit"))
        chunk = _quantum_chunk(prepared, result, run=run, attempt=attempt)
        if isinstance(chunk, tuple):
            # A trajectory acquisition completes once with all its point
            # chunks. The event binds the ordered collection by the identity
            # of their ObservationView, computed once.
            complete = event.revise(status="completed", finished=_now(),
                                    observation_id=ObservationView(chunks=chunk).content_id)
        else:
            complete = event.revise(status="completed", finished=_now(), returned_shots=chunk.returned_shots,
                                    observation_id=chunk.content_id)
        run._finish_attempt(index, complete, chunk, native_simulations=result.metadata.get("native_simulations"))
        return chunk
    except BaseException as error:
        run._fail_attempt(index, event, error, discarded=(
            result is not None and isinstance(error, Exception)
            and getattr(run.backend, "kind", None) in LOCAL_SIMULATORS))
        raise


def _statistic_stride(kind, *, key_width, shots=0, max_bytes=DEFAULT_MAX_BYTES):
    """Bytes one saved statistic reserves.

    A JSON statistic reserves its prototype envelope and one separating
    comma. The prototypes are those of ``readout_bytes``. A Pauli value has
    the widest label, a count bin the widest key and the decimal length of
    the requested shot number, which no single count can exceed. A host
    scalar has its prototype without the label text, a lower bound per
    scalar that ``readout_bytes`` completes with the actual labels. A
    probability is stored in a binary array, and dense or adaptively sparse
    probabilities reserve eight bytes per possible outcome (``readout_bytes``).
    """
    from nwqlib._run_journal import _json_bound

    if kind == "probabilities":
        return 8
    if kind == "pauli_expectation":
        prototype = PauliValue(label="I" * max(1, key_width), value=0.)
    elif kind == "counts":
        prototype = CountBin(bits="0" * max(1, key_width), count=shots)
    else:
        prototype = ScalarValue(label="x", value=0., frame="encoded_branch", parent_id="sha256:" + "f" * 64)
    return _json_bound(prototype, max_bytes) + 1


def _admit_readout(observation, *, width, classical_width, run, header=None):
    """Admit a readout's items and payload bytes before native preparation, and return its item count.

    The NWQLib admission law: admission precedes native
    construction and submission. Subtract known declaration and metadata
    reserves before deriving any payload capacity. For positive homogeneous
    item count ``L``, check ``K <= remaining_items//L`` before forming
    ``K*L``. For a marginal and nonnegative remaining item capacity
    ``M_items``, reject when ``q_k >= M_items.bit_length()`` before forming
    ``1 << q_k``. Check heterogeneous additions against remaining capacity and
    apply the corresponding division check before multiplying by the selected
    byte stride. Unresolved executable dimensions reject rather than count as
    zero. Pauli and count values use the conservative JSON prototype envelope
    of ``_statistic_stride``, and probabilities the eight bytes per possible
    outcome of their binary array. Amplitude artifacts keep their own binary
    payload and scalar-metadata accounting. An unsupported reduction or a reduction with
    unresolved output shape fails before native preparation.

    For a single-endpoint readout the declaration reserve is the
    observation's own JSON, the part of the declaration envelope ``J`` known
    before lowering, and the metadata reserve ``M`` is the Run's
    ``max_completion_metadata_bytes``. At submission ``readout_bytes``
    charges the complete reservation, including the bindings and layouts,
    against the remaining data budget. The item capacity of each readout is
    the remaining payload bytes divided by its statistic stride
    (``_statistic_stride``), 16 bytes per complex128 amplitude, so
    ``items*stride`` never exceeds the remaining bytes.

    A trajectory subtracts the full known prospective storage for all points
    and one ``M`` before deriving a value capacity: the complete contribution
    of the acquisition, including the newly stored receipt declaration, is
    ``R_new selection = J_T + sum_k (D_k + H_k + V_k) + M``
    (``trajectory_reservation``, ``_trajectory_fixed_bounds``,
    ``_trajectory_value_bounds``). ``header`` holds the lowering-independent
    selected context (experiment, setting, bindings, Plan and realization
    identities) and the logical layouts derived from the resolved registers
    (``_point_chunk_headers``). The Run's identity, the known encoded
    lengths of the attempt UUID and of a SHA-256 content identity, the
    trajectory identity, each point's slot, explicit position and
    acquisition source join it. Only the end shorthand is resolved after
    lowering. It is priced at its shortest form, so this gate charges no more
    than the final one, and the completed receipt supplies the final
    submission-time bound (``readout_bytes``). Without ``header``
    (restoration), the point headers are not charged. The early condition ``M_min(K) <= M`` applies as well,
    except to a detached NWQ-Sim or NWQ-Sim Slurm restoration. A fetch's
    outcome transaction credits the actual stored event and status rows
    when published, and a new submission from a restored handle is admitted
    with the fresh-acquisition minimum by ``readout_bytes`` before launch.
    The per-point item count is ``point_items``.
    """
    from nwqlib._run_journal import _json_bound
    from nwqlib.core.planning import point_items

    limit = run.limits.max_data_bytes
    declaration = _json_bound(observation, limit)
    remaining = limit - declaration - run.limits.max_completion_metadata_bytes
    if remaining < 0:
        raise _cumulative_limit_error(
            "max_data_bytes", limit, "readout declaration and completion metadata bytes", declaration,
            run.limits.max_completion_metadata_bytes)

    def clause(stride):
        return _data_limit_clause(run, f" less the readout declaration and completion metadata, at {stride} bytes "
                                       "per entry")

    kind = observation.kind
    if kind == "counts" and observation.shots > MAX_COUNT:
        # The int64 count representation requires 1 <= shots <=
        # 2**63-1 before submission (ObservationSpec admits shots >= 1).
        raise ValueError(f"requested shots {observation.shots} exceed the int64 count limit 2**63 - 1 = {MAX_COUNT}")
    if kind != "trajectory":
        if kind == "amplitudes":
            stride = 16
        elif kind == "estimated_observable":
            stride = 1  # One provider estimate. Its size is completion metadata.
        else:
            key_width = {"pauli_expectation": width, "probabilities": len(observation.qubits),
                         "counts": classical_width}.get(kind, 0)
            stride = _statistic_stride(kind, key_width=key_width, shots=observation.shots, max_bytes=limit)
        items = readout_shape(kind=kind, details=observation, width=width, classical_width=classical_width,
                             repetitions=observation.shots, max_items=remaining // stride,
                             max_items_source=clause(stride))
        if kind == "probabilities" and header is not None:
            required = _probability_endpoint_minimum(observation, header, run=run, max_bytes=limit)
            _check_probability_endpoint_metadata(required, run.limits.max_completion_metadata_bytes)
        elif header is not None:
            required = _known_endpoint_metadata(observation, header, run=run, max_bytes=limit)
            if required is not None:
                _check_completion_metadata(required, run.limits.max_completion_metadata_bytes)
        return items
    # Validate the whole schedule and its dimensions first. Every item costs
    # at least one byte, so the remaining bytes bound the item count.
    readout_shape(kind=kind, details=observation, width=width, classical_width=classical_width,
                  max_items=remaining, max_items_source=clause(1))
    # Then the complete trajectory law R = J_T + sum_k (D_k + H_k + V_k) + M
    # (``trajectory_reservation``): the receipt's declaration J_T is new here,
    # the point headers use the fields known before lowering, and fields known
    # only after native preparation are charged by readout_bytes at submission.
    common = per_point = None
    if header is not None:
        backend = getattr(run, "backend", None)
        version = getattr(backend, "native_target_version", None)
        common, per_point = _point_chunk_headers(
            observation, header, run_id=run.run_id, prepared_id=_CONTENT_ID_PROTOTYPE, attempt=_UUID_PROTOTYPE,
            boundaries=tuple(0 if point.position is None else point.position for point in observation.positions),
            backend_kind=getattr(backend, "kind", "x"), target_version=version() if callable(version) else "x")
    # The receipt's declaration field: key, colon, separating comma and value.
    receipt = _json_bound("observation", limit) + 2 + declaration
    # Preparation checks against the Run's remaining stored-data capacity,
    # and restoration, whose receipt is already stored, against the cap.
    available = limit if header is None else run._available_data_bytes()
    declarations, headers, minimum = _trajectory_fixed_bounds(observation, common=common, per_point=per_point,
                                                              max_bytes=limit, job_length=_job_length(run),
                                                              completion_minimum=_completion_minimum(run, limit))
    if header is None and getattr(getattr(run, "backend", None), "kind", None) in {"nwqsim", "slurm"}:
        # Fetch restoration has no new acquisition to admit. Its outcome
        # transaction credits the actual stored event and status rows.
        minimum = 0
    capacity = available - receipt - run.limits.max_completion_metadata_bytes - sum(declarations) - sum(headers)
    if capacity < 0 or minimum > run.limits.max_completion_metadata_bytes:
        trajectory_reservation(declarations, headers, (0,) * len(headers),
                               metadata_bytes=run.limits.max_completion_metadata_bytes, available_bytes=available,
                               new_receipt_bytes=receipt, minimum_metadata_bytes=minimum,
                               points=observation.positions, limit=limit)
    values = _trajectory_value_bounds(observation, width=width, capacity=capacity, max_bytes=limit,
                                      source=_data_limit_clause(run, " less the trajectory declarations, point "
                                                                     "headers and completion metadata"))
    trajectory_reservation(declarations, headers, values, metadata_bytes=run.limits.max_completion_metadata_bytes,
                           available_bytes=available, new_receipt_bytes=receipt, minimum_metadata_bytes=minimum,
                           points=observation.positions, limit=limit)
    return sum(point_items(point, width=width) for point in observation.positions)


def admit_pure_trajectory(spec, *, backend_name, body_length, classical_width, body_issue, view_issues):
    """Admit a trajectory as one deterministic, noiseless coherent evolution of a pure state.

    The selected contract is a normalized pure initial state with fixed,
    bound, coherent evolution. At logical boundary ``b`` the observed state
    is ``|psi_b> = U_b ... U_1 |psi_0>``. For a general channel the
    unconditional state evolves instead as
    ``rho' = sum_a K_a rho K_a^dagger`` with ``sum_a K_a^dagger K_a = I``,
    and an individual stochastic trajectory has state proportional to
    ``K_a psi`` with outcome-dependent ``a``, so one saved trajectory
    statistic need not equal the unconditional statistic. Refusal
    directions of features before an observed boundary, or in an executed
    intermediate view:

    - Fixed unitary gates, coherent controlled gates, barriers, a selected
      unitary ``StatePreparation``, or resolved fixed repetitions of those
      operations: admit unchanged when the adapter supports the
      operations/readout.
    - Reset, including reset hidden in a composite or ``Initialize``: can
      produce mixed unconditional states or a randomly selected pure branch.
      Refuse on the selected pure-trajectory route.
    - Mid-body measurement, even when its outcome is discarded: refuse for
      the pure-trajectory schedule.
    - Outcome-dependent classical branch, adaptive evolution,
      postselection: refuse.
    - Non-unitary channel, Kraus instruction, quantum-error instruction,
      state replacement that is not the selected coherent preparation:
      refuse.
    - Unknown or unsupported simulator method or opaque instruction
      semantics: refuse unless the adapter establishes the pure-state,
      non-destructive semantics for this selected operation/readout.

    This is a sufficient structural admission rule for the selected
    contract, not a claim that every listed operation always mixes every
    input. Only the actually executed prefix through the last requested
    point, its preparation, and its executed views affect these
    quantities. ``body_issue`` is the first incompatible operation on the
    executed prefix, including hidden definitions. ``view_issues`` is
    computed during the same view traversal. Refusals name the first
    body/view feature that violates the contract.

    The selected-IR side (``_selected_body_issues``) calls it before
    lowering with the facts of the bound IR. The adapter-side facts (noise
    configuration, effective pure-state method, phase faithfulness and the
    reducer's state domain) belong to each adapter's preparation check,
    before submission.
    """
    if spec.kind != "trajectory":
        return
    if spec.shots != 0 or spec.population != "unconditional" or classical_width:
        raise ValueError("trajectory requires zero shots and an unconditional coherent body without measurement registers")
    boundaries = spec.boundaries(body_length)
    if body_issue is not None:
        raise ValueError(
            f"{backend_name} trajectory cannot observe the selected pure prefix: {body_issue}"
        )
    for point, boundary in zip(spec.positions, boundaries, strict=True):
        if point.id in view_issues:
            raise ValueError(f"trajectory point {point.id!r} has an unsupported coherent view: {view_issues[point.id]}")


def _selected_body_issues(construction, admission, observation, widths):
    """Return the bound body's length, its first incompatible operation on the executed prefix and each view's.

    This is the selected-IR side of ``admit_pure_trajectory``, run before
    lowering. It walks the bound Program from its root in the order in
    which logical lowering emits the body (``blocks.lowering._lower_qiskit``):
    a ``BlockCall`` is one operation, a ``Measure`` or ``Reset`` one
    operation per qubit of its register, ``Allocate`` and ``Release`` none,
    and a ``Repeat`` its body's operations times its bound count. The
    returned length is therefore the bound body length against which the
    point boundaries resolve. Each definition is summarized once, however
    many calls or repetitions share it, and a ``Repeat`` is not unrolled.

    An explicit ``Reset`` or ``Measure`` of a nonempty register, an
    outcome-dependent ``Branch`` and an ``AdaptiveLoop`` are incompatible.
    Such an operation is on the executed prefix when it starts before the
    last point's boundary (the end of the body when a point uses the end
    shorthand), since an operation strictly after the last save cannot
    alter an earlier statistic. The tail and the inverse of each readout
    view are checked whole. Other nodes that logical lowering does not
    emit are refused by lowering. Instructions inside a selected block's
    native definition are not visible in the IR: lowering converts each
    selected block to a Qiskit ``Gate`` (``to_gate``), which Qiskit refuses
    for a circuit holding a reset, measurement or ``Initialize``, and the
    adapter checks the top-level instructions of the lowered circuit.
    """
    from nwqlib.ir import AdaptiveLoop, BlockCall, Branch, CoherentRegion, Measure, MeasurementBatch, Repeat, Reset
    from nwqlib.ir import Sequence as Serial
    nodes = admission.nodes
    binding = admission.binding_map(construction.program.bindings)
    root = nodes[construction.program.root]
    if isinstance(root, MeasurementBatch) and len(root.settings) == 1:
        binding.update(admission.binding_map(root.settings[0].bindings))
    context = admission.expressions(binding)
    memo = {}

    def summary(name):
        """Operations of one definition and its first incompatible operation, as ``(offset, text)`` or None."""
        if name in memo:
            return memo[name]
        node, length, issue = nodes[name], 0, None
        if isinstance(node, Serial):
            for child in node.children:
                size, found = summary(child)
                if issue is None and found is not None:
                    issue = (length + found[0], found[1])
                length += size
        elif isinstance(node, (Repeat, CoherentRegion, MeasurementBatch)):
            times = admission.integer(node.count, context, "Repeat") if isinstance(node, Repeat) else 1
            size, found = summary(node.body)
            length, issue = times * size, found if times else None
        elif isinstance(node, BlockCall):
            length = 1
        elif isinstance(node, (Measure, Reset)):
            length = widths.get(node.wire, 0)
            if length:
                issue = (0, f"{node.kind} of register {node.wire!r} in definition {name!r}")
        elif isinstance(node, (Branch, AdaptiveLoop)):
            issue = (0, f"outcome-dependent {node.kind} in definition {name!r}")
        memo[name] = length, issue
        return memo[name]

    length, found = summary(construction.program.root)
    last = length if any(point.position is None for point in observation.positions) else max(
        (point.position for point in observation.positions), default=0)
    body_issue = found[1] if found is not None and found[0] < last else None
    view_issues = {}
    for point in observation.positions:
        for definition in () if point.view is None else (point.view.tail, point.view.inverse):
            if definition in nodes and point.id not in view_issues:
                found = summary(definition)[1]
                if found is not None:
                    view_issues[point.id] = f"selected IR {found[1]}"
    return length, body_issue, view_issues


def _method_reduction_allowance(run, point, observation, width):
    """Ask the Run's Method for the reduction-work allowance of one trajectory point, or return None.

    The shared lifecycle funds registered reductions from the Method's own
    work ledger. A Method whose Plans can declare reduction points defines
    the hook

        reduction_allowance(self, plan, point, *, observation, width, run) -> int

    which receives the Plan, the point's ``Realization``, its resolved
    trajectory ``ObservationSpec``, the logical width of its selected
    Program and the Run. The hook first runs the Method's gate on the
    private workspace its reducers allocate, raising ``ValueError`` to
    refuse the point, and then returns the remaining allowance in the unit
    of the registered ``work`` functions. ``admit_reductions`` checks the
    summed registered work against it. The call happens before native
    work: before the preparation charge in ``prepare_experiment``, and
    again when the static path hands an existing preparation, live or saved
    by an earlier process, to submission (``_prepare_static_item``), so a
    resumed static Run is admitted under its current ledger and limits. The
    hook can therefore be asked more than once for one point. It reads the
    Method's ledger and does not change it. Only a trajectory with a
    reduction point asks.
    A Method without the hook supplies no allowance, and
    ``admit_reductions`` then refuses its reductions.
    """
    if observation.kind != "trajectory" or not any(item.kind == "reduction" for item in observation.positions):
        return None
    hook = getattr(run.plan.method, "reduction_allowance", None)
    if hook is None:
        return None
    allowance = hook(run.plan, point, observation=observation, width=width, run=run)
    if type(allowance) is not int or allowance < 0:
        raise ValueError(f"{type(run.plan.method).__name__}.reduction_allowance must return a nonnegative integer "
                         f"work allowance, not {allowance!r}")
    return allowance


def _admit_trajectory_work(observation, *, width, run, allowance):
    """Admit a trajectory's reduction work and saved simulator states before native preparation.

    Reductions: the registered host work of every reduction point, summed
    against ``allowance``, the calling Method's remaining work allowance
    (``core.planning.admit_reductions``). A caller-supplied size is never the
    reservation: the registry's shape, work and execution functions are.

    Saved buffers: a backend with ``admit_trajectory_buffers`` (Aer) admits
    the live state, every kept save (probability marginals, saved
    expectations and states) and its transient marginal workspace against
    ``simulator_memory_mb``. Otherwise the saved states are checked here. An
    amplitude or reduction point saves the complex128 simulator state, and a
    saved state costs ``16*2**w`` bytes per position, in addition to the live
    simulator state. Standard Aer saves accumulate
    results until job completion, so reducing them afterwards does not reduce
    that peak. The simultaneous footprint of ``S`` saved states and the live
    state, ``(1+S)*16*2**w`` bytes for the full native width ``w``, must fit
    the Run's ``simulator_memory_mb`` before native preparation. ``S`` counts
    the distinct declared positions of those points (the end shorthand as
    one), at least the number of distinct saved states.

    Outputs: each amplitude point's materialization
    (``AmplitudeReadout.admit_materialization``). The stored JSON of the
    reduction outputs is a point value payload ``V_k`` of the stored-data
    reservation (``_trajectory_value_bounds``).
    """
    from nwqlib.core.planning import admit_reductions

    admit_reductions(observation, width=width, allowance=allowance)
    # A backend that executes trajectories admits its own simulation-phase
    # arrays (AerBackend.admit_trajectory_buffers: live state, kept
    # saves, marginal workspace), also when no state is saved.
    admit = getattr(run.backend, "admit_trajectory_buffers", None)
    if admit is not None:
        try:
            admit(observation, width=width, memory_mb=run.limits.simulator_memory_mb, run=run)
        except ValueError as error:
            raise ValueError(f"the trajectory's simulation-phase arrays exceed the Run's memory limit before native "
                             f"preparation: {error}. {_limit_remedy('simulator_memory_mb')}") from None
    saved = len({point.position for point in observation.positions if point.kind in ("amplitudes", "reduction")})
    if saved and admit is None:
        limit = run.limits.simulator_memory_mb << 20
        needed = (1 + saved) * 16 << width
        if needed > limit:
            raise ValueError(f"the trajectory saves {saved} simulator state(s) of {width} qubits; with the live "
                             f"state they need {needed} bytes, more than simulator_memory_mb="
                             f"{run.limits.simulator_memory_mb} ({limit} bytes) before native preparation. "
                             f"{_limit_remedy('simulator_memory_mb')}")
    # An amplitude point materializes its selected output as a single-endpoint
    # amplitude readout does, and is admitted by the same owner before work.
    for point in observation.positions:
        if point.kind == "amplitudes":
            point.amplitudes.admit_materialization(max_bytes=run.limits.max_data_bytes,
                                                   max_bytes_source=_data_limit_clause(run))


_READOUT_DECLARATION_FIELDS = (
    "observation", "bindings", "quantum_layout", "classical_layout", "experiment", "setting",
)

# Encoded-length prototypes of an attempt UUID (``str(uuid4())``, 36 ASCII
# characters) and of a SHA-256 content identity. Both have known encoded
# lengths before their values exist.
_UUID_PROTOTYPE = "0" * 36
_CONTENT_ID_PROTOTYPE = "sha256:" + "0" * 64

# The fields walked by the point completion-metadata check. A funded
# probability header receives the bounded credit described below. Every other
# field of a point chunk is its declaration (``observation``, D_k) or a fixed
# header field (H_k, ``_point_chunk_headers``). A Pauli or reduction point's
# JSON value elements are its value payload V_k, so only the empty
# ``values`` envelope stays variable. A probability point's binary arrays are
# its V_k. Its ``values`` are counted by the completion walker and credited
# up to their funded header bound by _probability_header_credit.
_POINT_VARIABLE_FIELDS = ("job", "values")
_AMPLITUDE_POINT_VARIABLE_FIELDS = ("job", "values", "physical_scale", "physical_scale_unavailable", "artifacts",
                                    "unavailable")
# The smallest legal value of each variable field: a nonempty job text, an
# empty value array, and an absent scale, explanation and array list.
_POINT_VARIABLE_MINIMUM = dict(job="x", values=(), physical_scale=None, physical_scale_unavailable=None,
                               artifacts=(), unavailable=())


def _point_variable_fields(kind):
    """The variable fields of a point chunk whose one-point readout has ``kind``."""
    return _AMPLITUDE_POINT_VARIABLE_FIELDS if kind == "amplitudes" else _POINT_VARIABLE_FIELDS


def trajectory_reservation(declarations, headers, values, *, metadata_bytes, available_bytes, new_receipt_bytes=0,
                           minimum_metadata_bytes=0, points=(), limit=None):
    """Return a trajectory acquisition's stored-data reservation, checked against the available bytes.

    For one acquisition with ``K >= 1`` points, with ``J`` the
    ``_run_journal._json_bound`` of the journal encoding (computed fields
    excluded):

    - ``J_T`` (``new_receipt_bytes``): bound of the full trajectory
      declaration in its receipt, including its ``observation`` field
      envelope, charged when that receipt is first stored.
    - ``D_k`` (``declarations``): ``J({"observation": S_k})`` for the point's
      one-point spec ``S_k``.
    - ``H_k`` (``headers``): bound of that chunk's known fields and its
      funded probability record and net publication JSON, once per point.
    - ``V_k`` (``values``): worst-case value payload of the actual
      representation, excluding the fields in ``D_k`` or ``H_k``.
    - ``M`` (``metadata_bytes``): ``max_completion_metadata_bytes``, once per
      acquisition. It funds the variable remainder ``J(U_k)`` of every point
      chunk and the net other completion rows of the acquisition.

    The complete contribution of the acquisition, including a newly stored
    receipt declaration, is ``R_new selection = J_T + sum_k (D_k + H_k + V_k)
    + M``. After the receipt has been stored the execution reservation is
    ``R_pending = sum_k (D_k + H_k + V_k) + M`` (``new_receipt_bytes=0``, the
    submission-time case). Adding ``J_T`` again would reserve a second copy of
    an already charged declaration.

    Split each actual row into the declaration, fixed and variable maps. For
    disjoint nonempty maps, the sum of their JSON bounds exceeds the bound of
    their union by one byte per additional map, and replacing a scalar
    ``values`` array by ``()`` removes the element bounds that the scalar
    prototype law restores. Binary values are charged by the binary
    publication owner. Therefore ``A(point row k) + B_binary,k <= D_k + H_k +
    V_k + J(U_k)``, and the completion condition is ``sum_k J(U_k) +
    sum_(other completion rows) J(r) - C_meta - G_meta <= M``
    (``_completion_metadata_bound``). The early condition
    ``M_min(K) <= M`` (``minimum_metadata_bytes``), where ``M_min`` includes
    the variable envelopes and the established completion-row allowance,
    is checked before native work. Probability record and publication costs
    are funded by H_k at their largest legal encoding. For probability and
    Pauli trajectories on Aer, and on the fresh success path of NWQ-Sim and
    NWQ-Sim Slurm, the variable envelope and completion-row growth have
    upper bounds, so this metadata check is sufficient for those fields.
    Another backend's completion-row allowance is a temporary zero
    (``_completion_minimum``). Each amount is compared with the remaining bytes
    before it is added, so no product is formed beyond the available budget.

    The derivation is given above. The refusal messages name the point, the
    counted, requested and remaining bytes and the limit.

    Args:
        declarations: ``D_k`` per point, in schedule order.
        headers: ``H_k`` per point.
        values: ``V_k`` per point.
        metadata_bytes: ``M``, charged once.
        available_bytes: The remaining ``max_data_bytes`` of the Run.
        new_receipt_bytes: ``J_T`` when the receipt is not yet stored, else zero.
        minimum_metadata_bytes: ``M_min(K)``.
        points: The trajectory's points, named in a refusal.
        limit: ``max_data_bytes``, named in a refusal.
    """
    if minimum_metadata_bytes > metadata_bytes:
        raise ValueError(f"trajectory metadata envelopes of {len(headers)} point chunk(s) and the completion rows "
                         f"need at least {minimum_metadata_bytes} bytes, more than "
                         f"max_completion_metadata_bytes={metadata_bytes}, "
                         f"before native work. {_limit_remedy('max_completion_metadata_bytes')}")
    remaining = available_bytes
    total = 0
    for name, amount in (("the receipt declaration", new_receipt_bytes),
                         ("the completion metadata allowance", metadata_bytes)):
        if amount > remaining:
            raise ValueError(f"trajectory fixed storage exceeds remaining max_data_bytes: {name} requests {amount} "
                             f"bytes, {remaining} remain after {total} counted (max_data_bytes={limit}). "
                             f"{_limit_remedy('max_data_bytes')}")
        total += amount
        remaining -= amount
    for index, (declaration, header, payload) in enumerate(zip(declarations, headers, values, strict=True)):
        for amount in (declaration, header, payload):
            if amount > remaining:
                point = points[index].id if index < len(points) else index
                raise ValueError(f"trajectory point {point!r} storage exceeds remaining max_data_bytes: it requests "
                                 f"{amount} bytes, {remaining} remain after {total} counted "
                                 f"(max_data_bytes={limit}). {_limit_remedy('max_data_bytes')}")
            total += amount
            remaining -= amount
    return total


def _register_layout(widths):
    """Logical register maps of consecutive bits for ``(name, width)`` pairs, zero-width registers omitted.

    This is the layout that lowering allocates (``blocks.lowering``): one
    Qiskit register per nonempty declared register, in declaration order.
    """
    layout, offset = [], 0
    for name, width in widths:
        if width:
            layout.append(RegisterMap(name=name, bits=tuple(range(offset, offset + width))))
            offset += width
    return tuple(layout)


def _point_chunk_headers(observation, header, *, run_id, prepared_id, attempt, boundaries, backend_kind,
                         target_version, result_key="0"):
    """Return the fixed point-chunk fields as ``(common, per_point)``.

    ``header`` supplies the selected identities, experiment, setting,
    bindings and logical layouts. Before lowering, callers supply the
    encoded-length identity prototypes, zero for an unresolved end boundary,
    and the adapter's target version or its shortest legal text. Submission
    and publication supply the receipt's identities, boundaries and version.

    ``readout_bytes`` prices these maps in ``H_k`` and ``_trajectory_chunks``
    uses them when publishing each point. The declaration and variable fields
    remain separate. Amplitude points keep their declared source and an empty
    classical layout.
    """
    fields = dict(header)
    classical = fields.pop("classical_layout")
    common = dict(fields, schema_version=3, parent_id=None, run_id=run_id, execution="quantum_circuit",
                  prepared_id=prepared_id, attempt=attempt, population="unconditional", returned_shots=None,
                  trajectories=1, selected_kernel_id=None, applications=(), trajectory_id=observation.content_id)
    per_point = []
    for point, boundary in zip(observation.positions, boundaries, strict=True):
        key = point_chunk_key(result_key, point.id)
        values = dict(chunk=key, point=point.id, boundary=boundary)
        if point.kind == "amplitudes":
            values.update(classical_layout=(), source=point.amplitudes.source)
        else:
            values.update(classical_layout=classical, physical_scale=None, physical_scale_unavailable=None,
                          artifacts=(), unavailable=(),
                          source=Source(name=f"{backend_kind} acquisition", version=target_version,
                                        domain="backend observation", reference=f"{run_id}/{attempt}/{key}"))
        per_point.append(values)
    return common, per_point


def _receipt_header(prepared):
    """The ``header`` of a prepared point read from its receipt, for its chunk-field and metadata laws.

    It holds the same fields that ``_selected_readout`` derives before
    lowering, so submission and publication price and store the receipt's
    values.
    """
    record = prepared.record
    return dict(plan_id=record.plan_id, realization_id=prepared.realization.content_id,
                experiment=prepared.realization.experiment, setting=prepared._setting,
                bindings=prepared._bindings, quantum_layout=record.quantum_layout,
                classical_layout=record.classical_layout)


def _job_length(run):
    """The adapter-established length of its native job identifier, or 1 (the shortest legal job text)."""
    length = getattr(getattr(run, "backend", None), "native_job_id_length", None)
    return length() if callable(length) else 1


def _trajectory_fixed_bounds(observation, *, common, per_point, max_bytes, job_length=1, completion_minimum=0):
    """Return point declarations D_k, funded headers H_k and the required metadata allowance.

    D_k is J of the point's declaration field map. H_k contains its known
    chunk fields and, for probabilities, the worst legal encoding's
    ProbabilityArrays record, publication-row growth and payload-reference
    rows (``_probability_point_metadata``). Binary arrays are V_k, charged
    separately by ``_trajectory_value_bounds``.

    The field-tree rule for a map is two braces, one comma per adjacent pair
    and J(key) + one colon + J(value) for every field. Adding a field to the
    nonempty common map therefore adds J(key) + 2 + J(value). The singleton
    probability record adds exactly its J to the empty values tuple.
    Splitting declaration, header and variable maps conservatively counts
    their delimiters, so their sum bounds the combined chunk.

    The variable allowance includes each point's job and empty values
    envelope, plus the amplitude-specific variable fields where present,
    and ``completion_minimum``, the required growth allowance for the other
    completion rows. Aer uses their established maximum, G_max. Its UUID
    gives J({"job": job, "values": ()}) = 58, hence probability and Pauli
    trajectories without a forecast require 58*K + 1113 bytes. A fresh
    detached NWQ-Sim or NWQ-Sim Slurm outcome uses 168 bytes, or 99 with a
    forecast (``detached_completion_growth``). NWQ-Sim's 36-character UUID
    gives 58 bytes per point, and a Slurm job ID of at most ten ASCII digits
    gives at most 32, hence 58*K + 168 and 32*K + 168 bytes without a
    forecast. Another backend's completion allowance is a temporary zero
    (``_completion_minimum``).

    Without common headers, as in restoration's cardinality check, H_k is
    zero and this is only a preliminary metadata check. Submission prices
    the complete stored-receipt headers before acquiring another result.
    The receipt owns the complete trajectory declaration. The journal omits
    the chunk's observation field, so D_k is conservative reservation space.
    """
    from nwqlib._run_journal import _json_bound

    points = observation.positions
    declarations = [_json_bound({"observation": ObservationSpec.point_readout_fields(point)}, max_bytes)
                    for point in points]
    if common is None:
        headers = [0] * len(points)
    else:
        shared = _json_bound(common, max_bytes)
        headers = [shared + sum(_json_bound(name, max_bytes) + 2 + _json_bound(value, max_bytes)
                                for name, value in fields.items()) for fields in per_point]
    smallest = dict(_POINT_VARIABLE_MINIMUM, job="x" * job_length)
    minimum = {kind: _json_bound({name: smallest[name] for name in _point_variable_fields(kind)}, max_bytes)
               for kind in ("amplitudes", "other")}
    total = completion_minimum
    for index, point in enumerate(points):
        if point.kind == "probabilities" and common is not None:
            fixed, variable = _probability_point_metadata(
                point, common, per_point[index], smallest["job"], max_bytes)
            headers[index] += fixed
            total += variable
        else:
            total += minimum["amplitudes" if point.kind == "amplitudes" else "other"]
    return declarations, headers, total


def _probability_point_metadata(point, common, fields, job, max_bytes):
    """Return ``(F_k, J(U_k))`` for one probability point's fixed JSON and variable envelope.

    ``J(U_k) = J({"job": job, "values": ()})``. For encoding E, ``F_k(E)``
    is ``J(ProbabilityArrays_E)`` plus, for every array, the publication
    row's net growth and its payload-reference row. A singleton values tuple
    adds exactly ``J(ProbabilityArrays_E)`` to the empty tuple. The complete
    probability metadata bound is ``F_k + J(U_k)``.

    ``F_k`` is the maximum over dense and sparse encodings. The sparse
    candidate uses ``s_max = (2**q - 1)//(W + 1)``, from the strict binary
    storage choice ``(W + 1)*s < 2**q``. Integer field sizes are monotone in
    their nonnegative maxima, and each scalar float has the 32-byte JSON
    envelope. Digests and record identities have fixed encoded lengths.
    The publication reservation contains only integers and strings, so its
    actual encoded size equals J and its replacement credit is exact.

    The trajectory reserves F_k in its point header. ``job`` supplies the
    adapter-established escaped-size prototype, an exact upper bound for
    Aer's ASCII UUID. With an unknown job-size bound, use the minimum legal
    text and leave any excess in completion metadata. Publication credits
    at most the funded F_k and only bytes included in that completion's net
    charge (``_probability_header_credit``).

    This constructs scalar metadata only. Callers admit the logical outcome
    count before this function forms powers of two.
    """
    from nwqlib._run_journal import _json_bound
    from nwqlib.artifacts import ArtifactManifest, ReadoutArray
    from nwqlib.execution import ProbabilityArrays

    width = len(point.qubits)
    words = -(-width // 64)
    provenance = dict(plan_id=common["plan_id"], realization_id=common["realization_id"],
                      construction_id=common["prepared_id"], producer_id=_CONTENT_ID_PROTOTYPE,
                      acquisition=(common["run_id"], common["attempt"], job, fields["chunk"]), source=fields["source"])

    def manifest(component, shape):
        output = ReadoutArray(component=component, dtype="uint64" if component == "indices" else "float64",
                              shape=shape)
        return ArtifactManifest(output=output, digest=_CONTENT_ID_PROTOTYPE, data_bytes=output.data_bytes,
                                encoding=output.encoding, **provenance)

    def publication(item):
        charge = dict(data_bytes=item.data_bytes, acquisition=item.acquisition)
        payload = PayloadRef(format=f"nwqlib.array/{item.output.storage_dtype}", bytes=item.data_bytes,
                             key=item.digest)
        return (_json_bound(dict(charge, manifest=item, duplicate=False, payload=payload), max_bytes)
                - _json_bound(charge, max_bytes) + _json_bound(payload, max_bytes))

    largest = ((1 << width) - 1) // (words + 1)
    sparse = (manifest("indices", (largest,) if words == 1 else (largest, words)), manifest("probabilities", (largest,)))
    candidates = [(ProbabilityArrays(encoding="sparse", width=width, entries=largest, nonzero=largest, mass=0.,
                                     maximum=0., indices=sparse[0], probabilities=sparse[1]), sparse)]
    if words == 1 and width < 63:
        dense = manifest("probabilities", (1 << width,))
        candidates.append((ProbabilityArrays(encoding="dense", width=width, entries=1 << width, nonzero=1 << width,
                                             mass=0., maximum=0., probabilities=dense), (dense,)))
    variable = _json_bound(dict(job=job, values=()), max_bytes)
    fixed = max(_json_bound(record, max_bytes) + sum(publication(item) for item in items)
                for record, items in candidates)
    return fixed, variable


def _probability_header_credit(records, max_bytes, *, run):
    """Credit funded probability-point JSON once, limited by each point's header reservation.

    The uncredited completion bound includes each ProbabilityArrays record,
    publication row and payload-reference row. Only a manifest named by a
    probability trajectory chunk can receive this credit. For changed rows,
    credit their positive net growth after the actual old row size, since
    Run._write already applies that replacement credit. A shared payload row
    is assigned to one point and counted once. Existing payload rows have
    zero net growth because their scalar bound equals their stored size.

    Per point, credit min(actual included growth, reserved F_k). An unknown
    job string or extra publication metadata therefore continues to spend M
    when it exceeds the funded prototype. No array values are read. Work is
    linear in the completion's records and metadata field trees.
    """
    from nwqlib._run_journal import _json_bound

    caps, costs, owners, payload_owners = {}, {}, {}, {}
    job = "x" * _job_length(run)
    for kind, _, chunk in records:
        if (kind != "chunk" or not isinstance(chunk, ObservationChunk)
                or chunk.point is None or chunk.observation.kind != "probabilities"):
            continue
        key = chunk.acquisition_key
        common = {name: getattr(chunk, name) for name in
                  ("plan_id", "realization_id", "prepared_id", "run_id", "attempt")}
        fields = dict(chunk=chunk.chunk, source=chunk.source)
        caps[key], _ = _probability_point_metadata(chunk.observation, common, fields, job, max_bytes)
        costs[key] = _json_bound(chunk.values, max_bytes) - 2
        record = chunk.values[0]
        for manifest in (record.probabilities, record.indices):
            if manifest is not None:
                owners[manifest.content_id] = key
    if not caps:
        return 0
    previous = run._state["data_sizes"]
    for kind, key, value in records:
        if kind != "publication" or "manifest" not in value:
            continue
        owner = owners.get(value["manifest"].content_id)
        if owner is None:
            continue
        costs[owner] += max(0, _json_bound(value, max_bytes) - previous.get((kind, key), 0))
        payload_owners.setdefault(value["payload"].content_id, owner)
    for kind, key, value in records:
        if kind == "payload" and key in payload_owners:
            costs[payload_owners[key]] += max(0, _json_bound(value, max_bytes) - previous.get((kind, key), 0))
    return sum(min(costs[key], cap) for key, cap in caps.items())


def _probability_endpoint_minimum(observation, header, *, run, max_bytes,
                                  prepared_id=_CONTENT_ID_PROTOTYPE, attempt=_UUID_PROTOTYPE,
                                  result_key="0", target_version=None, population=None):
    """Required metadata for a probability endpoint under its existing B + J_decl + M reservation.

    The endpoint has no trajectory header credit. Its metadata map contains
    every chunk field except _READOUT_DECLARATION_FIELDS. Price that map
    with empty values, then add the probability record and publication
    growth at the largest legal encoding, and the completion-row allowance.
    This is the full endpoint remainder, not only a trajectory point's job
    envelope. The caller supplies the selected header before lowering or
    the completed receipt's header at submission. Unknown adapter text
    lengths or completion growth prevent a sufficiency guarantee. The
    completion-row allowance comes from ``_completion_minimum``. For a
    backend other than Aer, NWQ-Sim and NWQ-Sim Slurm it is zero, a
    temporary unpriced allowance rather than a derived minimum, until a
    derivation of that backend's completion lifecycle establishes the
    growth of its completion rows.
    """
    from nwqlib._run_journal import _json_bound

    if target_version is None:
        version = getattr(run.backend, "native_target_version", None)
        target_version = version() if callable(version) else "x"
    job = "x" * _job_length(run)
    fields = {name: field.default for name, field in ObservationChunk.model_fields.items()
              if not field.is_required()}
    fields.update(header, run_id=run.run_id, prepared_id=prepared_id, attempt=attempt,
                  job=job, chunk=result_key, observation=observation, values=(),
                  population=observation.population if population is None else population,
                  returned_shots=None, trajectories=1,
                  source=Source(name=f"{run.backend.kind} acquisition", version=target_version,
                                domain="backend observation", reference=f"{run.run_id}/{attempt}/{result_key}"))
    fixed, _ = _probability_point_metadata(observation, fields, fields, job, max_bytes)
    rest = {name: value for name, value in fields.items() if name not in _READOUT_DECLARATION_FIELDS}
    return fixed + _json_bound(rest, max_bytes) + _completion_minimum(run, max_bytes)


def _check_probability_endpoint_metadata(required, allowance):
    """Refuse an endpoint whose known metadata cannot fit before native work."""
    if required > allowance:
        raise ValueError(f"probability endpoint metadata and completion rows need at least {required} bytes, "
                         f"more than max_completion_metadata_bytes={allowance}, before native work. "
                         f"{_limit_remedy('max_completion_metadata_bytes')}")


def _scalar_endpoint_metadata(observation, header, *, run, max_bytes,
                              prepared_id=_CONTENT_ID_PROTOTYPE,
                              attempt=_UUID_PROTOTYPE, result_key="0",
                              target_version=None, population=None):
    """Required completion metadata M of a synchronous Aer counts or exact Pauli scalar endpoint.

    This applies the accounting split of the probability endpoint
    (``_probability_endpoint_minimum``) before preparation. A single
    nontrajectory chunk already funds its readout declaration and numerical
    value elements in ``readout_bytes``. Its remaining field map, with
    ``values=()``, and the net growth of its event/submission rows consume M.

    Let J be ``_run_journal._json_bound``, U the chunk fields other than the
    declaration fields with empty values, and G the completion-row growth
    after crediting the actual stored initial event and intent. The
    sufficient requirement is::

        M_required = J(U_max) + G_max

    All fixed identities, selected layouts and source fields have their
    actual or bounded lengths, with the backend's established maximum job-ID
    and timestamp widths. For synchronous Aer exact scalars, G is 1,113 bytes
    without a forecast and 1,044 with one (``_completion_minimum``). Counts
    change one further event field, ``returned_shots: None -> returned_count``,
    so::

        G_counts = G_scalar + max(0, digits(shots) - 4)

    is sufficient even when the returned-shot field may remain null. On the
    successful counts path it is the tighter ``G_scalar + digits(shots) - 4``,
    which this function uses. The chunk also holds ``returned_shots``, priced
    at J(shots) for the successful counts-only domain. Counts have
    ``trajectories=None``. Exact Pauli scalars have ``trajectories=1`` and
    ``returned_shots=None``. For the RFE Plan of the ``rfe`` case of
    ``tests/test_observation_schedule.py::test_static_metadata_check_names_the_largest_setting_requirement``,
    the difference between a
    three-digit and a four-digit (pooled, 1,200-shot) setting is exactly these
    two returned-shot occurrences, one byte each, so its allowance grows from
    2,079 to 2,081.

    This is a sufficient successful-completion law for Aer with the qualified
    Source and job domains. A detached adapter needs its own event
    returned-shot delta added to its completion-growth owner, and unpriced
    backend text or provider estimate records are not bounded by it, so the
    callers apply it on Aer only (``_known_endpoint_metadata``). The
    derivation is given above. Its allowances were measured on an exact
    Expectation (2,078 bytes, completed at that value) and the 92-setting RFE
    Plan (2,079 or 2,081 by pooled shots, maximum actual charge 2,081).
    """
    from nwqlib._run_journal import _json_bound

    counts = observation.kind == "counts"
    if observation.kind not in {"counts", "pauli_expectation"}:
        raise ValueError("scalar metadata sizing requires counts or Pauli expectations")
    if target_version is None:
        version = getattr(run.backend, "native_target_version", None)
        target_version = version() if callable(version) else "x"
    job = "x" * _job_length(run)
    fields = {
        name: field.default for name, field in ObservationChunk.model_fields.items()
        if not field.is_required()
    }
    fields.update(
        header, run_id=run.run_id, prepared_id=prepared_id, attempt=attempt,
        job=job, chunk=result_key, observation=observation, values=(),
        population=observation.population if population is None else population,
        returned_shots=observation.shots if counts else None,
        trajectories=None if counts else 1,
        source=Source(
            name=f"{run.backend.kind} acquisition", version=target_version,
            domain="backend observation",
            reference=f"{run.run_id}/{attempt}/{result_key}",
        ),
    )
    rest = {name: value for name, value in fields.items()
            if name not in _READOUT_DECLARATION_FIELDS}
    growth = _completion_minimum(run, max_bytes)
    if counts:
        growth += _json_bound(observation.shots, max_bytes) - _json_bound(None, max_bytes)
    return _json_bound(rest, max_bytes) + growth


def _amplitude_endpoint_metadata(observation, header, *, run, max_bytes):
    """Required completion metadata M of a synchronous Aer amplitude endpoint (a fresh result key ``"0"``).

    For each legal amplitude outcome o, U_o is the full chunk remainder:
    scalar mass summaries, physical scale and either the array manifest or
    the declared unavailability record. The successful-array branch also adds
    ``J(publication_completed) - J(publication_reserved) + J(payload_reference)``.
    The reserved publication row holds only integers and strings, so its J
    equals its stored size. The sufficient allowance is the maximum of these
    branch totals plus G_max (``_completion_minimum``). The binary array bytes
    stay in the separate data reservation.

    The text domains are fixed by ``amplitudes.reduce_amplitudes`` and
    ``_publish_amplitudes``. Finite float values use J's 32-byte envelope. A
    produced finite native norm has a bounded binary exponent: for output
    dimension D and selected recovery exponent e_R,
    ``abs(e_R) + 1076 + bit_length(D)`` conservatively bounds the magnitude of
    the composed exponent, covering subnormals and normalization, and a
    negative prototype at that magnitude bounds its decimal length. The two
    unavailable-output reasons and the scalar-mass unavailability message come
    from those two functions. No provider text is bounded by an invented
    length.

    The prototype identities have the fixed lengths of the actual prepared
    and attempt identities. A trajectory amplitude point uses its own header
    split instead. The derivation is given above. Its allowance was measured
    on the exact default QLS (four qubits, admitted at 5,316 bytes covering
    its 5,264-byte charge) and LCHS (nine qubits, admitted at 5,132 covering
    5,128) Plans.
    """
    from nwqlib._run_journal import _json_bound
    from nwqlib.artifacts import ArtifactManifest, UnavailableOutput
    from nwqlib.problems.inputs import PhysicalScale
    from nwqlib.amplitudes import AMPLITUDE_MASS_LABELS, AMPLITUDE_MASS_UNAVAILABLE

    readout = observation.amplitudes
    output = readout.output
    attempt = _UUID_PROTOTYPE
    job = "x" * _job_length(run)
    acquisition = (run.run_id, attempt, job, "0")

    def bound(value):
        return _json_bound(value, max_bytes)

    manifest = ArtifactManifest(
        output=output, digest=_CONTENT_ID_PROTOTYPE, data_bytes=output.data_bytes,
        encoding=output.encoding, plan_id=header["plan_id"],
        realization_id=header["realization_id"], construction_id=readout.construction_id,
        producer_id=_CONTENT_ID_PROTOTYPE, acquisition=acquisition, source=readout.source,
    )
    payload = PayloadRef(format=f"nwqlib.array/{output.storage_dtype}",
                         bytes=output.data_bytes, key=manifest.digest)
    charge = dict(data_bytes=output.data_bytes, acquisition=acquisition)
    publication = (bound(dict(charge, manifest=manifest, duplicate=False, payload=payload))
                   - bound(charge) + bound(payload))
    masses = []
    for label in AMPLITUDE_MASS_LABELS if readout.keep_masses else ():
        numeric = ScalarValue(label=label, value=0., frame="encoded_branch")
        absent = ScalarValue(label=label, value=None, frame="encoded_branch",
                             unavailable=AMPLITUDE_MASS_UNAVAILABLE)
        masses.append(max((numeric, absent), key=bound))
    scale = None
    if readout.recovery is not None:
        exponent = -(abs(readout.recovery.exponent) + 1076 + output.basis.dimension.bit_length())
        scale = PhysicalScale(mantissa=.5, exponent=exponent)
    fields = {name: field.default for name, field in ObservationChunk.model_fields.items()
              if not field.is_required()}
    fields.update(
        header, run_id=run.run_id, prepared_id=_CONTENT_ID_PROTOTYPE,
        attempt=attempt, job=job, chunk="0", observation=observation,
        population="unconditional", classical_layout=(), returned_shots=None,
        trajectories=1, source=readout.source, values=tuple(masses), physical_scale=scale,
        physical_scale_unavailable=None if scale is not None else "physical recovery is unavailable",
    )
    candidates = []
    outcomes = (
        ((manifest,), (), publication),
        ((), (UnavailableOutput(output=output,
            reason="unit vector is undefined for a zero selected branch"),), 0),
        ((), (UnavailableOutput(output=output,
            reason="selected physical vector is not representable in complex128"),), 0),
    )
    for artifacts, unavailable, growth in outcomes:
        rest = {name: value for name, value in
                dict(fields, artifacts=artifacts, unavailable=unavailable).items()
                if name not in _READOUT_DECLARATION_FIELDS}
        candidates.append(bound(rest) + growth + _completion_minimum(run, max_bytes))
    return max(candidates)


def _known_endpoint_metadata(observation, header, *, run, max_bytes, **identities):
    """Known completion metadata of a counts, exact Pauli scalar or amplitude endpoint, or None.

    The laws of ``_scalar_endpoint_metadata`` and ``_amplitude_endpoint_metadata``
    are sufficient for the synchronous Aer lifecycle only, so another backend
    or readout kind returns None and is checked at publication. ``identities``
    are the actual prepared and attempt identities, result key, target version
    and population of a completed receipt. The amplitude law uses the
    fixed-length prototypes and the fresh result key ``"0"`` of a synchronous
    acquisition.
    """
    if getattr(run.backend, "kind", None) != "qiskit_aer":
        return None
    if observation.kind in {"counts", "pauli_expectation"}:
        return _scalar_endpoint_metadata(observation, header, run=run, max_bytes=max_bytes, **identities)
    if observation.kind == "amplitudes":
        return _amplitude_endpoint_metadata(observation, header, run=run, max_bytes=max_bytes)
    return None


def _check_completion_metadata(required, allowance):
    """Refuse selected acquisitions whose known completion metadata cannot fit, before native preparation.

    A capacity refusal before the attempt is launched can be cleared on the
    open Run, unlike a publication refusal that already made a synchronous
    acquisition terminal (``Run._write``).
    """
    if required > allowance:
        raise ValueError(
            f"The selected acquisitions need max_completion_metadata_bytes "
            f"of at least {required} per completion, exceeding {allowance}, "
            "before native preparation. Set "
            f"ExecutionLimits(max_completion_metadata_bytes={required}) "
            "or raise this limit on the open Run before resuming preparation."
        )


def _repeated_bound(element_bound, count):
    """``J`` of a tuple of ``count`` elements of bound ``element_bound``: ``count + 1 + count*element_bound``, or 2 when empty."""
    return 2 if count == 0 else count + 1 + count * element_bound


def _trajectory_value_bounds(observation, *, width, capacity, max_bytes, source=None):
    """Return each trajectory point's worst-case value payload ``V_k`` in stored bytes.

    The value laws of the actual representations, with disjoint
    representation categories:

    1. A Pauli value stored as a JSON record uses its actual prototype
       envelope. For ``L_k`` entries, ``L_k * _statistic_stride(...)``
       covers all element envelopes and commas.
    2. Dense or adaptively sparse probability arrays reserve ``8 * 2**q_k``
       bytes. With ``W_k = max(1, (q_k+63)//64)``, sparse storage costs
       ``8*(W_k+1)*s_k`` and is chosen only when smaller than dense storage.
       The point's array manifests, summaries and publication-row growth
       are funded by its fixed header at their largest legal encoding.
    3. An amplitude point reserves its declared complex128 output, normally
       ``16*L_k``. Its scalar masses and manifests are in the checked
       variable remainder, not complex-vector bytes.
    4. A registered reduction's outputs are stored as JSON ``ReducedValues``
       records, one per registered component, so they also use their actual
       prototype envelope: a finite float has the 32-byte envelope, an int64
       the decimal length of ``-2**63``, and a tuple of ``n`` elements of
       bound ``e`` has bound ``n + 1 + n*e`` (``_repeated_bound``), plus one
       separating comma per component.

    Only probability and amplitude points publish binary arrays: a
    probability point reserves eight bytes per outcome (law 2) and an
    amplitude point 16 bytes per complex128 value (law 3). Pauli and
    reduction values are JSON records (laws 1 and 4). These laws do not
    include declarations or headers, and they do not imply that a scalar
    stored as JSON occupies eight stored bytes. ``point_items`` supplies
    ``L_k``, which for a probability point is ``2**q_k``, independently of
    its eventual sparse stored count. With a ``capacity`` (the pre-lowering
    gate), each point's item capacity is ``capacity`` divided by its stride,
    checked before ``1 << q_k`` is formed, and a reduction output larger than
    ``capacity`` refuses before its prototype is counted. The laws are given
    above, and ``docs/development/execution.md`` (Readout reservation) gives
    the reservation of each readout kind.
    """
    from nwqlib._run_journal import _json_bound
    from nwqlib.core.planning import point_items, reducer_outputs
    from nwqlib.execution import ReducedValues

    def items(point, stride):
        if capacity is None:
            return point_items(point)
        return point_items(point, width=width, max_items=capacity // stride, max_items_source=source)

    values = []
    for point in observation.positions:
        if point.kind == "reduction":
            payload = 0
            for index, output in enumerate(reducer_outputs(point.reducer, point.parameters)):
                if capacity is not None and output.items > capacity:
                    raise ValueError(f"trajectory point {point.id!r} has {output.items} reduction output items, more "
                                     f"than the {capacity} bytes available{'' if source is None else ', ' + source}")
                element = _json_bound(-(1 << 63) if output.dtype == "int64" else 0., max_bytes)
                arrays = (1 if output.dtype != "complex128" else 2)
                payload += (_json_bound(ReducedValues(component=index), max_bytes)
                            + arrays * (_repeated_bound(element, output.items) - 2) + 1)
        elif point.kind == "amplitudes":
            items(point, 16)
            payload = point.amplitudes.output.data_bytes
        else:
            key_width = width if point.kind == "pauli_expectation" else len(point.qubits)
            stride = _statistic_stride(point.kind, key_width=key_width, max_bytes=max_bytes)
            payload = items(point, stride) * stride
        values.append(payload)
    return values


def readout_bytes(prepared, *, metadata_bytes, max_bytes=DEFAULT_MAX_BYTES, run=None, attempt=None, result_key="0"):
    """Reserve an acquisition's stored bytes before it runs, without enumerating outcomes.

    A probability endpoint keeps R = B + J_decl + M, where B = 8*2**q
    bounds the dense or adaptively sparse binary payload, J_decl funds
    _READOUT_DECLARATION_FIELDS, and M funds the full remaining chunk map
    and net publication/completion JSON. With a Run, check that required
    metadata allowance before submission. Preparation performs the same
    check with its prospective fields before lowering. Without a Run this
    function only sizes the endpoint reservation.

    A trajectory reserves sum_k(D_k + H_k + V_k) + M. Its receipt owns the
    complete trajectory declaration and is charged when stored. D_k bounds
    each point's declaration, H_k funds its known chunk fields and the
    largest legal probability record/publication encoding, and V_k funds
    the representation-specific values. M funds one acquisition's point
    job envelopes and other completion growth. Probability publication
    credits only included net metadata bytes up to each funded H_k amount.
    Shared payload rows receive one credit. Probability and Pauli
    trajectories without a forecast require 58*K + 1113 metadata bytes on
    Aer, 58*K + 168 on NWQ-Sim and 32*K + 168 on NWQ-Sim Slurm, whose
    fresh detached completion growth ``_completion_minimum`` prices.
    Another backend's completion allowance is a temporary zero there. A
    single probability endpoint on NWQ-Sim or NWQ-Sim Slurm includes the
    same 168-byte allowance, or 99 bytes with a forecast.

    Counts and Pauli statistics use their JSON prototype bounds. Counts
    have at most min(shots, 2**width) entries, each at most shots. Host
    scalars use their declared labels and prototype envelopes, alongside
    the kernel's declared array and application-receipt bytes. Amplitudes
    reserve the selected output's binary bytes.

    Run._write releases the acquisition's pending reservation and charges
    its actual JSON and binary bytes. The uncredited completion bound from
    _completion_metadata_bound, less actual replacement-row credits,
    declared host application credits and _probability_header_credit, must
    fit max_completion_metadata_bytes per completed acquisition. These are
    serialized-data bounds, not process-memory bounds.
    """
    from nwqlib._run_journal import _json_bound

    record = prepared.record
    spec = record.observation
    # Only the metadata checks against a Run read the receipt's header.
    header = None if run is None else _receipt_header(prepared)
    if spec.kind == "trajectory":
        if run is None or attempt is None:
            raise ValueError("a trajectory reservation needs its Run and attempt, whose values its point chunks hold")
        common, per_point = _point_chunk_headers(spec, header, run_id=run.run_id, prepared_id=record.content_id,
                                                 attempt=attempt, boundaries=record.boundaries,
                                                 backend_kind=run.backend.kind, target_version=record.target.version,
                                                 result_key=result_key)
        declarations, headers, minimum = _trajectory_fixed_bounds(spec, common=common, per_point=per_point,
                                                                  max_bytes=max_bytes, job_length=_job_length(run),
                                                                  completion_minimum=_completion_minimum(run, max_bytes))
        width = sum(len(reg.bits) for reg in prepared.record.quantum_layout)
        values = _trajectory_value_bounds(spec, width=width, capacity=None, max_bytes=max_bytes)
        return trajectory_reservation(declarations, headers, values, metadata_bytes=metadata_bytes,
                                      available_bytes=run._available_data_bytes(),
                                      minimum_metadata_bytes=minimum, points=spec.positions, limit=max_bytes)
    # These declaration fields have a fixed reservation, even for thousands
    # of Pauli labels. The endpoint's remaining chunk metadata and net
    # publication/completion JSON spend the metadata allowance.
    declaration = _json_bound(dict(observation=spec, bindings=prepared._bindings,
        quantum_layout=prepared.record.quantum_layout,
        classical_layout=prepared.record.classical_layout,
        experiment=prepared.realization.experiment, setting=prepared._setting), max_bytes)
    metadata_reservation = declaration + metadata_bytes
    if run is not None and spec.kind in {"counts", "pauli_expectation", "amplitudes"}:
        # The counts, exact Pauli scalar and amplitude laws repeat the
        # preparation-time check with the completed receipt's identities.
        identities = {} if spec.kind == "amplitudes" else dict(
            prepared_id=prepared.record.content_id, attempt=_UUID_PROTOTYPE if attempt is None else attempt,
            result_key=result_key, target_version=prepared.record.target.version,
            population=prepared.record.population)
        required = _known_endpoint_metadata(spec, header, run=run, max_bytes=max_bytes, **identities)
        if required is not None:
            _check_completion_metadata(required, metadata_bytes)
    if spec.kind == "amplitudes":
        return spec.amplitudes.output.data_bytes + metadata_reservation
    if spec.kind == "estimated_observable":
        return metadata_reservation  # The provider estimate itself has variable size.
    if spec.kind == "host_scalars":
        # Per scalar: the bound of a prototype ScalarValue minus its one-letter
        # label, plus the actual escaped label. The prototype has the 32-byte
        # float envelope, the longest frame name, a full-length parent identity
        # and no unavailability text, which _completion_metadata_bound charges
        # to the metadata allowance instead. Then add the declared bytes of the
        # kernel's output arrays and application receipts, one comma between
        # consecutive scalars, and the metadata reservation.
        kernel = prepared._native.record
        base = ScalarValue(label="x", value=0., frame="encoded_branch", parent_id="sha256:"+"f"*64)
        constant = _json_bound(base, max_bytes)-_json_bound(base.label, max_bytes)
        values = sum(constant+_json_bound(label, max_bytes) for label in kernel.scalars)
        return (kernel.data_bytes + kernel.application_bytes + values + max(0, len(kernel.scalars)-1)
                + metadata_reservation)
    if spec.kind == "probabilities":
        if run is not None:
            required = _probability_endpoint_minimum(
                spec, header, run=run, max_bytes=max_bytes, prepared_id=prepared.record.content_id,
                attempt=_UUID_PROTOTYPE if attempt is None else attempt, result_key=result_key,
                target_version=prepared.record.target.version, population=prepared.record.population)
            _check_probability_endpoint_metadata(required, metadata_bytes)
        # B = 8*D for the D = 2**k possible outcomes (prepared._items, from
        # readout_shape): the adaptive encoding stores at most 8*D bytes, and
        # before acquisition the sparse count is unknown.
        return 8 * prepared._items + metadata_reservation
    if spec.kind == "counts":
        width = sum(len(reg.bits) for reg in prepared.record.classical_layout)
        prototype = CountBin(bits="0"*max(1, width), count=spec.shots)
    else:
        prototype = PauliValue(label="I"*max(map(len, spec.labels)), value=0.)
    # Terms: prepared._items statistics, each bounded by the prototype, one comma
    # between consecutive items, and the metadata reservation. The prototype has the
    # widest key, and either the 32-byte envelope of a real value or, for a
    # count, the decimal length of the requested shot number, which no single
    # count can exceed.
    item = _json_bound(prototype, max_bytes)
    return prepared._items*item + max(0, prepared._items-1) + metadata_reservation


def _completion_metadata_bound(records, max_bytes):
    """Separate bounded numeric statistics from one completion's other JSON.

    Array bytes are charged by their publication owner. The caller credits
    rows being replaced, so updating a large batch header charges its growth.
    This visits metadata and any scalar-unavailability messages, without
    copying or examining amplitude or probability arrays or count values. A
    probability chunk's values are its array manifests and scalar summaries,
    which are counted here. Run._write subtracts the funded probability
    point header credit with _probability_header_credit before testing M.
    Single endpoints have no such credit. Binary payloads are charged by
    publication.

    A trajectory point chunk excludes its declaration and header only because
    that particular occurrence was reserved (``readout_bytes``: ``D_k`` and
    ``H_k``, built by ``_point_chunk_headers`` and checked at publication by
    ``_trajectory_chunks``), and its JSON value elements because the value
    law reserved them (``V_k``). Its remainder ``U_k`` is its variable fields
    (``_point_variable_fields``) with an empty ``values`` envelope, except that
    an amplitude or probability point keeps its ``values`` (masses, or array
    manifests and summaries). The
    total covers all ``U_k`` and the other completion rows, and the
    acquisition count is the cardinality of the set of ``(run_id,
    attempt)``, not the number of chunks. ``Run._write`` checks
    ``sum_k J(U_k) + sum_(other completion rows) J(r) - C_meta - G_meta <= M``
    per acquisition after subtracting the probability header credit as
    another G_meta term. It releases that attempt's one pending reservation
    and charges all its actual rows and binary bytes atomically.
    """
    from nwqlib._run_journal import _json_bound

    acquisitions, total = set(), 0
    for kind, _, value in records:
        if kind == "chunk" and isinstance(value, ObservationChunk):
            acquisitions.add((value.run_id, value.attempt))
            if value.point is not None:
                readout = value.observation.kind
                rest = {name: getattr(value, name) for name in _point_variable_fields(readout)}
                if readout not in ("amplitudes", "probabilities"):
                    rest["values"] = ()  # elements were charged by the value law
                total += _json_bound(rest, max_bytes)
                continue
            # These fields were funded at preparation, using the same JSON
            # owner. Splitting the two field maps counts one extra delimiter,
            # so their summed bound also covers the combined map.
            fields = {name: getattr(value, name) for name in type(value).model_fields
                      if name not in _READOUT_DECLARATION_FIELDS}
            readout = value.observation.kind
            if readout in {"counts", "pauli_expectation", "host_scalars"}:
                fields["values"] = ()
                if readout == "host_scalars":
                    total += sum(max(0, _json_bound(item.unavailable, max_bytes)-4)
                                 for item in value.values if item.unavailable is not None)
            total += _json_bound(fields, max_bytes)
            continue
        total += _json_bound(value, max_bytes)
    return len(acquisitions), total


def synchronous_completion_growth(*, provider, timing_reference, job_length, max_bytes,
                                  event_already_revised=False):
    """Return ``(minimum, maximum)`` of the net JSON growth of a synchronous acquisition's completion rows.

    Trajectory metadata admission includes the minimum point envelopes and
    the net growth of its synchronous completion rows. The row-growth
    calculation includes revision parents, completion timestamps, the
    observation identity, the job locator and timing evidence, with the
    original stored rows credited once. Fields copied unchanged, including
    the output reservation, cancel out. The minimum is a necessary admission
    check. Aer admission uses the maximum because its UUID, timestamp and
    completion-field domains establish an upper bound. Variable metadata is
    also checked against the acquisition's allowance when published.

    Let ``E_r, S_i`` be the event and submission stored before the
    synchronous call and ``E_c, S_c`` their completed forms, ``J`` the
    journal bound and ``A`` the actual encoded length. The net charge is
    ``G = J(E_c) + J(S_c) - A(E_r) - A(S_i) = [J(E_c) - J(E_r)] + [J(S_c) -
    J(S_i)] + [J(E_r) - A(E_r)] + [J(S_i) - A(S_i)]``. The last two terms are
    nonnegative (zero for these initial rows, whose field trees hold no
    floats or array encodings). The same keys occur before and after, so
    braces, keys, colons and commas cancel in each difference and only the
    changed field values remain. The reservation integer cancels for any
    number of digits. The changed fields are the event's parent, status,
    finish time, observation identity and timing, and the submission's
    parent, status, locator, finish time, invocation count, timing and
    ``results_consumed``. A quoted SHA-256 content identity costs 73 bytes,
    a shortest ``_now()`` UTC timestamp 27 and one with microseconds 34, so
    the maximum adds seven timestamp bytes in each row. ``J`` prices every
    finite float at 32 bytes, so durations change neither result. For the
    Aer timing source ``nwqlib.backends.connection.AerBackend.submit`` and a
    ``qiskit_aer`` locator with a 36-character UUID this gives ``G_min =
    2*J_t + J_l + 241 = 1099`` and ``G_max = G_min + 14 = 1113`` with
    ``J_t = 322`` and ``J_l = 214``. A forecast revises the event before it
    is stored (``Run._begin_submission``), so its parent contributes nothing
    (``event_already_revised``). ``job_length`` is an established
    escaped-size bound for the job identifier, a lower bound for the minimum
    and an upper bound for the maximum. Aer's ASCII UUID has the same bound in
    both directions. The derivation is given above.
    """
    from nwqlib._run_journal import _json_bound

    # All input text and the job-ID width have already passed their input cap.
    def bound(value):
        return _json_bound(value, max_bytes)

    cid = "sha256:" + "0" * 64
    time_min = "0001-01-01T00:00:00+00:00"
    timing = TimingObservation(
        seconds=0.0, scope="native_call_wall",
        source=Source(name="perf_counter", version="python", domain="one local backend call",
                      reference=timing_reference),
    )
    locator = JobLocator(provider=provider, job_id="x" * job_length)
    changes = [
        (cid if event_already_revised else None, cid),
        ("reserved", "completed"), (None, time_min), (None, cid), (None, timing),
        (None, cid), ("intent", "completed"), (None, locator), (None, time_min),
        (None, 1), (None, timing), (False, True),
    ]
    minimum = sum(bound(new) - bound(old) for old, new in changes)
    return minimum, minimum + 14


def detached_completion_growth(*, max_bytes, event_already_revised=False):
    """Return minimum and maximum JSON growth of a fresh detached outcome write.

    The successful provider-status row has already committed. The outcome
    transaction replaces that row and the reserved event, and changes the
    submission's results_consumed from False to True. Its row charge is
    J(E_completed) + J(S_consumed) - A(E_reserved) - A(S_status).

    J equals A for these string/integer rows. The event gains its revision
    parent (69 bytes unless already revised), completed status (1 byte),
    finish time (23 or 30 bytes), and observation identity (69 bytes).
    Consuming the submission saves one byte. Thus the pair is (161, 168),
    or (92, 99) when a forecast already revised the reserved event.

    The locator, provider status and simulation count cancel against the
    stored status row. The reservation integer also cancels. This law is
    independent of the number of trajectory points. It covers a fresh
    successful NWQ-Sim or NWQ-Sim Slurm acquisition with one event, no
    timing evidence and no prior event failure. Recovery must credit its
    actual stored rows. These are JSON bytes, not process memory. The
    derivation is given above.
    """
    from nwqlib._run_journal import _json_bound

    cid = "sha256:" + "0" * 64
    changes = (
        (cid if event_already_revised else None, cid),
        ("reserved", "completed"),
        (None, "0001-01-01T00:00:00+00:00"),
        (None, cid),
        (False, True),
    )
    minimum = sum(_json_bound(new, max_bytes) - _json_bound(old, max_bytes)
                  for old, new in changes)
    return minimum, minimum + 7


def _completion_minimum(run, max_bytes):
    """Return the required completion-row allowance G_max of a new acquisition, or zero for an unpriced backend.

    The synchronous Aer lifecycle establishes both the job's ASCII UUID
    size and every completion field's upper bound. Price the maximum of
    ``synchronous_completion_growth`` so the two timestamps' fractional
    seconds cannot exhaust an admitted metadata allowance. This is 1113
    bytes without a forecast and 1044 when the event already has a revision
    parent. A fresh detached NWQ-Sim or NWQ-Sim Slurm outcome uses the
    maximum of ``detached_completion_growth``: 168 bytes, or 99 bytes with
    a forecast. Another backend requires its own lifecycle and field bounds.
    Its zero is a temporary unpriced allowance, not a derived minimum, for
    trajectories and single-endpoint probability readouts alike. It ends
    when a derivation of that backend's completion lifecycle, as for
    ``detached_completion_growth``, establishes the growth of its
    completion rows. Despite its name, ``_completion_minimum`` returns the
    upper end of the row-growth range for Aer and NWQ-Sim, the allowance its
    callers require.
    """
    backend = getattr(run, "backend", None)
    kind = getattr(backend, "kind", None)
    already_revised = getattr(run, "forecast", None) is not None
    if kind in {"nwqsim", "slurm"}:
        return detached_completion_growth(
            max_bytes=max_bytes, event_already_revised=already_revised)[1]
    if kind == "qiskit_aer":
        return synchronous_completion_growth(
            provider=kind,
            timing_reference=f"{type(backend).__module__}.{type(backend).__name__}.submit",
            job_length=_job_length(run), max_bytes=max_bytes,
            event_already_revised=already_revised)[1]
    return 0


def _native_timing(started, reference):
    """Wall-clock seconds of one local backend call since ``started`` (``perf_counter``)."""
    return TimingObservation(seconds=perf_counter()-started, scope="native_call_wall",
        source=Source(name="perf_counter", version="python", domain="one local backend call", reference=reference))


def _quantum_chunk(prepared, result, *, run, attempt, result_key="0"):
    """Decode backend output and check it against its own preparation receipt.

    The exact-probability window depends on the executed circuit's operation
    count, which only the receipt knows. The unit bound is therefore checked here,
    where backend output first meets its receipt. A standalone Result checks it
    again before it is saved (``saved_evidence._validate_recorded_data``).
    """
    if prepared.record.observation.kind == "trajectory":
        chunks = _trajectory_chunks(prepared, result, run=run, attempt=attempt, result_key=result_key)
        for chunk in chunks:
            chunk.validate_unit_bound(prepared.record)
        return chunks
    chunk = _decoded_chunk(prepared, result, run=run, attempt=attempt, result_key=result_key)
    chunk.validate_unit_bound(prepared.record)
    return chunk


def _trajectory_chunks(prepared, result, *, run, attempt, result_key):
    """Translate one trajectory result into one chunk per point, in schedule order.

    The backend returns each point's values under its point ID, and a Pauli
    value under its (point, label) pair. A point chunk records its point ID
    and resolved boundary from the receipt, and its key
    ``point_chunk_key(result_key, point)``. Its values follow the point's
    one-point readout and are checked like that readout's. Every point
    chunk of the acquisition shares its run, attempt and job. Its fixed
    fields are the header maps that ``readout_bytes`` priced
    (``_point_chunk_headers``), and each published chunk is checked against
    them, so the completion-metadata check may exclude them.
    """
    from nwqlib.core.planning import ReductionContext, execute_reduction, point_items, registered_reducer
    record = prepared.record
    spec = record.observation
    returned = result.raw_output["trajectory"]
    readouts = spec.point_observations()
    if set(returned) != {point.id for point in spec.positions}:
        raise ValueError("trajectory output differs from its declared points")
    common, per_point = _point_chunk_headers(spec, _receipt_header(prepared), run_id=run.run_id,
                                             prepared_id=record.content_id, attempt=attempt,
                                             boundaries=record.boundaries, backend_kind=run.backend.kind,
                                             target_version=record.target.version, result_key=result_key)
    chunks = []
    for point, fields in zip(spec.positions, per_point, strict=True):
        raw = returned[point.id]
        key = fields["chunk"]
        if point.kind == "amplitudes":
            chunk = _publish_amplitudes(prepared, result, run=run, attempt=attempt, result_key=key,
                                        statevector=raw["statevector"], point=point.id, boundary=fields["boundary"],
                                        trajectory_id=common["trajectory_id"])
            if any(getattr(chunk, name) != value for name, value in (*common.items(), *fields.items())):
                raise ValueError(f"the chunk of point {point.id!r} differs from its reserved header")
            chunks.append(chunk)
            continue
        if point.kind == "reduction":
            # The registered reduction runs once, on the state saved at this
            # point, with the experiment's immutable bindings, which the chunk
            # keeps beside its outputs.
            # A reducer registered with receives_context also reads the
            # producing receipt's saved-state window, exclusions and
            # qualified state error, after resolving the labels the reducer
            # declares. The saved-state owners add the host phase product's
            # envelope, which every consumer of a corrected shared array
            # carries. The receipt's budget is modulo one common global phase:
            # a phase-invariant reducer receives it as its state error, and
            # any other reducer receives no phase-defined state error, with
            # the reason, and the modulo-phase budget separately
            # (ReductionContext). An unregistered reducer resolves nothing
            # here, and execute_reduction refuses it by name.
            reducer = registered_reducer(point.reducer)
            resolved = () if reducer is None else reducer.state_error_resolutions
            modulo_delta, modulo_reason = record.saved_state_error(resolved)
            invariant = reducer is not None and reducer.phase_invariant
            phase_reason = (
                "phase-defined state error is unavailable: native and prefix "
                "global-phase errors have no bound"
            )
            context = ReductionContext(
                prepared_id=record.content_id,
                probability_window=record.saved_state_probability_window,
                probability_window_exclusions=record.probability_window_exclusions,
                state_error=modulo_delta if invariant else None,
                state_error_reason=modulo_reason if invariant else phase_reason,
                modulo_phase_state_error=modulo_delta,
            )
            values = tuple(
                {"kind": "reduced", "component": index, "real": array.real.ravel().tolist(),
                 "imaginary": array.imag.ravel().tolist()} if array.dtype.kind == "c" else
                {"kind": "reduced", "component": index, "integers": array.ravel().tolist()} if array.dtype.kind == "i" else
                {"kind": "reduced", "component": index, "real": array.ravel().tolist()}
                for index, array in enumerate(execute_reduction(point, raw["statevector"], bindings=prepared._bindings,
                                                                context=context)))
        elif point.kind == "probabilities":
            # The marginal is published as binary arrays with this acquisition,
            # like a single-endpoint probability readout (_probability_output).
            values, indices = _probability_output(raw["probabilities"], width=len(point.qubits),
                                                  items=point_items(point))
            chunks.append(ObservationChunk._from_probability_arrays(
                values, indices, store=run.artifacts, **common, **fields, job=result.metadata["native_job_id"],
                observation=readouts[point.id]))
            del values, indices
            continue
        else:
            values = raw["pauli_expectations"]
            if len(values) > point_items(point):
                raise ValueError(f"backend output of point {point.id!r} exceeds its selected readout cardinality")
            values = tuple({"kind": "pauli", "label": label, "value": value} for label, value in values.items())
        chunks.append(ObservationChunk(**common, **fields, job=result.metadata["native_job_id"],
                                       observation=readouts[point.id], values=values))
    return tuple(chunks)


def _probability_output(raw, *, width, items):
    """The ``(values, indices)`` arrays of one backend probability output of ``width`` observed qubits.

    An adapter hands over one of three forms. A dense marginal is its
    float64 buffer, outcome j at position j, which publication keeps without
    a copy on the dense branch (``ObservationChunk._from_probability_arrays``)
    and whose indices are formed only on the sparse branch, so a dense
    marginal is never turned into an index array to be passed on. An indexed
    pair ``(indices, values)`` holds 1-D uint64 indices for a width of at
    most 64, or ``(entries, W)`` packed words, and float64 values. A mapping
    from bit-string key to value is the output of adapters that return
    dictionaries and of test backends. The number of entries of a pair or a
    mapping is checked against the selected cardinality ``items`` before
    any array is built. A dense buffer has exactly ``2**width`` entries,
    which ``artifacts.probability_readout`` checks, and which also sorts the
    indices of the other forms and chooses the stored encoding.
    """
    import numpy as np
    if isinstance(raw, np.ndarray):
        return raw, None
    if len(raw[1] if isinstance(raw, tuple) else raw) > items:
        raise ValueError("backend output exceeds the selected readout cardinality")
    if isinstance(raw, tuple):
        indices, values = raw
        return np.asarray(values), np.asarray(indices)
    return _probability_mapping(raw, width)


def _decoded_chunk(prepared, result, *, run, attempt, result_key):
    """Translate one backend result into an ObservationChunk with the receipt's layouts.

    The number of returned entries is checked against the selected readout
    cardinality before any record is built. Backend count keys follow Qiskit's
    convention, in which spaces separate classical registers and the rightmost
    character is classical bit 0. The spaces are removed here, and the receipt's
    classical layout names the registers. Counts also arrive as an indexed
    pair ``(indices, counts)`` in the ``Histogram`` layout and are stored as
    JSON count records. A probability marginal, a dense buffer, an indexed
    pair or a mapping, is published as binary arrays with the acquisition,
    inside its outcome batch (``_probability_output``). The chunk carries its original run, attempt,
    job and item, the identity that later deduplication and joins use.
    """
    spec = prepared.record.observation
    if spec.kind == "amplitudes":
        return _publish_amplitudes(prepared, result, run=run, attempt=attempt, result_key=result_key)
    fields = dict(run_id=run.run_id, plan_id=run.plan.content_id, realization_id=prepared.realization.content_id,
        prepared_id=prepared.record.content_id, experiment=prepared.realization.experiment,
        setting=prepared._setting, bindings=prepared._bindings, quantum_layout=prepared.record.quantum_layout,
        classical_layout=prepared.record.classical_layout, attempt=attempt, job=result.metadata["native_job_id"],
        chunk=result_key, observation=spec, population=prepared.record.population,
        trajectories=None if spec.kind in {"counts", "estimated_observable"} else 1,
        source=Source(name=f"{run.backend.kind} acquisition", version=prepared.record.target.version,
            domain="backend observation", reference=f"{run.run_id}/{attempt}/{result_key}"))
    if spec.kind == "probabilities":
        values, indices = _probability_output(result.raw_output["probabilities"], width=len(spec.qubits),
                                              items=prepared._items)
        return ObservationChunk._from_probability_arrays(values, indices, store=run.artifacts, returned_shots=None,
                                                         **fields)
    field = {"counts": "counts", "pauli_expectation": "pauli_expectations",
             "estimated_observable": "estimates"}[spec.kind]
    raw = result.raw_output[field]
    if len(raw[1] if spec.kind == "counts" and isinstance(raw, tuple) else raw) > prepared._items:
        raise ValueError("backend output exceeds the selected readout cardinality")
    if spec.kind == "counts" and isinstance(raw, tuple):
        # An indexed pair (indices, integer counts) in the Histogram layout.
        # Counts stay JSON count records, totalled with Python integers.
        return ObservationChunk.from_histogram(raw, returned_shots=sum(int(count) for count in raw[1]), **fields)
    # Counts and Pauli values enter as field mappings, so the chunk validates
    # each one once instead of validating a built record again.
    if spec.kind == "counts":
        values = tuple({"kind": "count", "bits": key.replace(" ", ""), "count": value} for key, value in raw.items())
        # A non-integer count is rejected by the chunk's own validation.
        returned = sum(value for value in raw.values() if type(value) is int)
    elif spec.kind == "estimated_observable":
        values = tuple(EstimateValue.model_validate(value) for value in raw)
        returned = None
    else:
        values = tuple({"kind": "pauli", "label": key, "value": value} for key, value in raw.items())
        returned = None
    return ObservationChunk(returned_shots=returned, values=values, **fields)


def _static_item(plan, run, experiment, *, program=None):
    """The durable workflow row of one static experiment, created once.

    Its runtime seed is drawn and committed with the RNG position before any
    preparation (commit 1 of the module docstring), so the experiment's seed never
    changes across restarts. The row later records the preparation, receipt and
    attempt that belong to it. A caller that already holds the experiment's
    selected Program passes it as ``program``, whose bindings are the point's
    effective bindings, so the point is not resolved again.
    """
    key = experiment.name
    item = run._state["workflow_items"].get(key)
    if item is None:
        point = plan.resolve(key) if program is None else Realization(
            plan_id=plan.content_id, experiment=key, bindings=program.bindings)
        item = WorkflowItem(realization=point, runtime=RuntimeOptions(seed=run.rng.next_seed()))
        run._write((("workflow_item", key, item), ("rng", "current", run.rng.snapshot())))
        run._state["workflow_items"][key] = item
    return item


class PreparationNotRebuilt(RuntimeError):
    """A static item's local preparation has a charge but no receipt, and the Run does not build it again.

    ``_unbuilt`` decides this and ``_not_rebuilt`` builds the error.
    ``execute_static`` raises it before it submits any new experiment, and
    ``_prepare_static_item`` raises it for its own item. A caller that
    continues a reopened Run can tell this outcome, which only new charged
    work could complete, from an error that an uninterrupted Run meets as well
    (``algorithms.qhd._durable.unrecoverable``).
    """


def _unbuilt(run, item):
    """Whether a static workflow item holds a local preparation charge without a receipt.

    Such a preparation failed or was interrupted after its charge (commit 2
    of the module docstring). The charge is already spent, and the Run never
    repeats a preparation attempt on its own, since that would be new charged
    work the caller did not request, so the Plan cannot complete in this Run.
    A saved remote preparation does not count, because its compilation can
    still be refreshed, and a refused synthesis removes the preparation from
    its item (``Run._charge_synthesis``), which can then be prepared again.
    """
    return (item.prepared_id is None and item.preparation_id is not None
            and item.preparation_id not in run._state["remote_preparations"])


def _not_rebuilt(setting):
    """The ``PreparationNotRebuilt`` error for ``setting``, with one message for both raisers."""
    return PreparationNotRebuilt(f"the original preparation of setting {setting!r} failed or was interrupted "
                                 "and no replacement was built, so this Run cannot complete and submits "
                                 "nothing more. A new Run prepares the Plan again")


def _prepare_static_item(plan, run, experiment, *, held=None):
    """Return the item's one preparation, live, restored from its receipt, or new.

    An item that ``_unbuilt`` finds is refused rather than rebuilt, with
    ``PreparationNotRebuilt``. The item's seed stays in its workflow row.
    The reductions of an existing preparation are admitted again against
    the allowance of the Method's ``reduction_allowance`` hook before it is
    returned (``_method_reduction_allowance``). A new preparation is
    admitted inside ``prepare_experiment``.
    ``held`` is the experiment's ``(Experiment, SelectedConstruction)``
    pair when the caller already holds it, which a new preparation reuses.
    """
    key = experiment.name
    item = _static_item(plan, run, experiment, program=None if held is None else held[1].program)
    if item.prepared_id is not None:
        # An existing preparation, live or saved by an earlier process, is
        # submitted under the Method's current ledger and limits, so its
        # reductions are admitted again before any native work.
        record = run.prepared_artifact(item.prepared_id)
        width = sum(len(reg.bits) for reg in record.quantum_layout)
        allowance = _method_reduction_allowance(run, record.realization, record.observation, width)
        if allowance is not None:
            from nwqlib.core.planning import admit_reductions
            admit_reductions(record.observation, width=width, allowance=allowance)
        return run._state["handles"].get(item.prepared_id) or restore_prepared(item.prepared_id, run=run, for_submission=True)
    if item.preparation_id is not None:
        if _unbuilt(run, item):
            raise _not_rebuilt(key)
        from nwqlib._remote_preparation import refresh_preparation
        return refresh_preparation(item.preparation_id, run=run)
    run._state["workflow_current"] = key
    try:
        return prepare_experiment(item.realization, run=run, runtime=item.runtime, held=held)
    finally:
        run._state["workflow_current"] = None


def prepare_static(plan, *, run, settings="first"):
    """Admit the complete static population, then prepare its first item or every item.

    The admission checks every experiment's readout against the backend target,
    and the circuits, preparations and shots of the whole Plan against the
    Run's cumulative limits, whichever items are prepared now. ``settings="all"``
    therefore needs no limit of its own. It prepares every experiment in Plan
    order through ``_prepare_static_item`` and submits nothing. That function
    returns an item's existing preparation when there is one, so each item is
    prepared and charged once, and a durable Run journals it. ``execute_static``
    later finds the receipt in the item's workflow row and submits the saved
    preparation. A preparation that fails or is interrupted stops the loop. If
    it stopped after its charge, the charge stays on the item's workflow row
    without a receipt (``_unbuilt``), and ``execute_static`` raises
    ``PreparationNotRebuilt`` before it submits anything. A
    refused synthesis clears the row instead, and a check that fails before
    the charge charges nothing, so in those cases the item can be prepared
    again.
    Each workflow row draws its runtime seed when it is created, which happens
    in Plan order in both modes, so both modes prepare with the same seeds.
    Through ``scientist.prepare``, ``"all"`` reaches this function only for
    ``LOCAL_SIMULATORS`` and host execution.
    A target refusal during the admission names the Experiment and, for a
    counts readout, the shots per program that it requests, the pooled
    product for a pooled query, or, for any other readout, its readout kind,
    before any preparation or submission. The first experiment's Experiment
    and construction, resolved during the admission, are passed to its
    preparation, so that point is not resolved again.
    On Aer the admission also checks, once, the largest known completion
    metadata requirement over all counts, exact Pauli scalar and amplitude
    settings (``_known_endpoint_metadata``) against
    ``max_completion_metadata_bytes``, so a later setting's larger requirement
    refuses before the first preparation rather than after earlier settings
    ran. An adaptive Method, whose later settings are not known, is checked
    per newly known population in ``prepare_experiment``. On Aer a setting
    whose observation comes from its Program's batch, or whose readout kind
    is priced, is admitted once in this scan, by the readout resolution
    (``_selected_readout``) that gives both its observation and the header
    its metadata law prices, until a cumulative cap is known to be exceeded. An admission refusal in the scan
    names the Method's ``max_admission_steps`` field when it has one. Once
    the circuits or shots counted so far, with the Run's existing ones,
    exceed ``max_total_circuits`` or ``max_total_shots``, the remaining
    settings are counted from their observations without pricing their
    metadata, and the cap refuses with the complete population before the
    metadata check.
    """
    if run.plan is not plan:
        raise ValueError("static preparation requires the same live Plan")
    priced = {"counts", "pauli_expectation", "amplitudes"}
    aer = getattr(run.backend, "kind", None) == "qiskit_aer"
    state, limits = run._state, run.limits
    # Circuit preparations and acquisition attempts are each capped by
    # max_total_circuits, so the larger of the two counts decides below.
    counted = max(state["preparations"] - state["host_preparations"],
                  len(state["events"]) - state["host_invocations"])
    circuits = shots = required = 0
    held = None
    over = False
    for index, experiment in enumerate(plan.experiments):
        header = None
        try:
            if index == 0:
                # The first experiment is prepared right after this loop, so its
                # Experiment and construction are kept for that preparation.
                held = plan._selected_construction(experiment.name, ())
            if aer and not over:
                # One readout resolution gives both the observation and the
                # header that the Aer completion-metadata law prices.
                point = plan.resolve(experiment.name)
                selected = held[0] if index == 0 else point._selected_program(plan)[0]
                if selected.batch is None and selected.observation.kind not in priced:
                    spec = selected.observation
                else:
                    selected, construction = held if index == 0 else point._selected_construction(plan)
                    *_, spec, header = _selected_readout(plan, point, selected, construction)
            elif index == 0:
                if held[0].batch is None:
                    spec = held[0].observation
                else:
                    admission = _Admission(held[1].program)
                    admission.admitted().require_ready()
                    _, spec = held[0]._resolved_observation(admission, construction_id=held[1].content_id)
            else:
                _, spec = plan.resolve(experiment.name).resolved_observation(plan)
        except AdmissionStepsExceeded as error:
            raise _named_admission_refusal(plan, error) from error
        if spec.kind == "host_scalars":
            continue
        try:
            run.backend.target_for(spec)
        except ValueError as error:
            # A counts readout names its shots per program. Any other readout
            # kind names the kind, which is what the target refuses.
            request = (f"{spec.shots} shots per program" if spec.kind == "counts"
                       else f"readout kind {spec.kind!r}")
            raise ValueError(f"Experiment {experiment.name!r} requests {request}: {error}") from error
        circuits += 1
        shots += spec.shots
        # Once the population counted so far exceeds a cumulative cap, later
        # settings are still counted, so the refusal names the complete
        # population, but their completion metadata is not priced.
        over = over or counted + circuits > limits.max_total_circuits or state["shots"] + shots > limits.max_total_shots
        if header is not None and not over and spec.kind in priced:
            required = max(required, _known_endpoint_metadata(spec, header, run=run,
                                                              max_bytes=limits.max_data_bytes))
    # The capacity checks and the first preparation follow the whole loop. A
    # population known to exceed a cap refuses before the metadata check.
    if over:
        run.check_capacity(preparations=circuits, circuits=circuits, shots=shots)
    _check_completion_metadata(required, run.limits.max_completion_metadata_bytes)
    run.check_capacity(preparations=circuits, circuits=circuits, shots=shots)
    for index, experiment in enumerate(plan.experiments if settings == "all" else plan.experiments[:1]):
        _prepare_static_item(plan, run, experiment, held=held if index == 0 else None)


def execute_static(plan, *, run):
    """Advance original finite settings. Completed attempts are never repeated.

    Experiments run in Plan order, one attempt each. The loop returns None at the
    first experiment that cannot contribute an observation yet, because its
    attempt has not completed, its preparation is still compiling remotely or its
    detached submission is still running. After cancellation it returns None
    before new work. An existing attempt is never submitted again. Analysis runs
    once every experiment has a collected observation.

    An item with a local preparation charge but no receipt (``_unbuilt``)
    means that the Plan cannot complete in this Run. The loop raises
    ``PreparationNotRebuilt`` before it submits any new experiment, whose
    exposure would otherwise be spent without a Result. After
    ``prepare(plan, settings="all")`` such an item can come before items that
    were never submitted. With the default ``settings="first"`` only the last
    item created can be one, so there the check raises the error that
    ``_prepare_static_item`` would raise for that item.
    """
    if run.plan is not plan:
        raise ValueError("static execution requires the same live Plan")
    unbuilt = next((key for key, item in run._state["workflow_items"].items() if _unbuilt(run, item)), None)
    for experiment in plan.experiments:
        key = experiment.name
        item = _static_item(plan, run, experiment)
        if item.attempt is not None:
            event = run.attempt_record(item.attempt)
            if event.status != "completed":
                return None
            completed = run.completed_observation(item.attempt)
            for chunk in completed if isinstance(completed, tuple) else (completed,):
                run.collect(chunk)
            continue
        if run.cancel_requested is not None:
            return None
        if unbuilt is not None:
            raise _not_rebuilt(unbuilt)
        prepared = _prepare_static_item(plan, run, experiment)
        if isinstance(prepared, PendingPreparation):
            return None
        run._state["workflow_current"] = key
        try:
            chunk = submit_experiment(prepared, run=run)
        finally:
            run._state["workflow_current"] = None
        chunks = chunk if isinstance(chunk, tuple) else (chunk,)
        if not all(isinstance(item, ObservationChunk) for item in chunks):
            return None
        for item in chunks:
            run.collect(item)
    run.progress("analyze", 0, 1)
    result = plan.method.analyze(plan, run.data, settings={})
    run.progress("analyze", 1, 1)
    return result


def selected_host_kernel(construction):
    """The kernel declaration named by the construction's root host stage.

    Host execution requires the Program root itself to be a host
    ``ClassicalStage`` with a kernel, so exactly one declared kernel runs.
    """
    from nwqlib.ir import ClassicalStage
    root = next(definition.node for definition in construction.program.definitions if definition.id == construction.program.root)
    if not isinstance(root, ClassicalStage) or root.boundary != "host" or root.kernel is None:
        raise ValueError("host execution requires one direct selected host stage")
    return next(item for item in construction.kernels if item.name == root.kernel)


def _host_binding(construction, observation, *, run):
    """Return the selected kernel declaration and the Plan's one bound implementation of it.

    The readout must name exactly the kernel's scalars, and the Plan must bind
    one ``BoundKernel`` whose record equals the Plan's own declaration. A kernel
    bound for another Plan or declaration therefore cannot run under this one.
    """
    from nwqlib.blocks.kernels import BoundKernel
    kernel = selected_host_kernel(construction)
    blocks = run.plan.blocks
    if observation.labels != kernel.scalars or len(blocks) != 1 or type(blocks[0]) is not BoundKernel:
        raise ValueError("host preparation requires its exact bound kernel and scalar labels")
    native = blocks[0]
    template = next((item for item in run.plan.construction.kernels if item.name == kernel.name), None)
    if native.record != template or native.plan_id != run.plan.content_id:
        raise ValueError("host binding belongs to another Plan or selected declaration")
    return kernel, native


def _prepare_host(point, construction, observation, runtime, charge, run, items, setting):
    """Record a host preparation receipt. Numerical work starts only at submission.

    The receipt binds the selected kernel declaration, the point and the installed
    versions of the kernel's declared dependencies. It carries no quantum layout
    or lowering, and its population is unconditional, as ``PreparedArtifact``
    requires for host work. The receipt and the revised charge are commit 3 of
    the module docstring.
    """
    from importlib.metadata import version
    from nwqlib.backends.host import HOST_TARGET
    kernel, native = _host_binding(construction, observation, run=run)
    run.check_capacity(data_bytes=kernel.data_bytes)
    native = native._at(point, kernel, run)
    record = PreparedArtifact(execution="host_kernel", plan_id=run.plan.content_id,
        realization_id=point.content_id, realization=point, construction_id=construction.content_id, snapshot=str(uuid4()),
        target=Source(name=HOST_TARGET.name, version="1", domain=HOST_TARGET.provider, reference="nwqlib.backends.host.HOST_TARGET"),
        compiler=Source(name="selected host binding", version="1", domain="original method inputs", reference="nwqlib.blocks.kernels.BoundKernel"),
        runtime=runtime, observation=observation, quantum_layout=(), classical_layout=(), native_basis=(),
        environment=tuple(Source(name=name, version=version(name), domain="host dependency", reference="importlib.metadata") for name in kernel.dependencies),
        preparation_time=_now(), construction_work_reserved=0, host_preparation_work_reserved=kernel.construction_work,
        selected_kernel_id=kernel.content_id, logical=None, native_operations=None, population="unconditional",
        transformation="selected immutable host inputs; numerical work starts at submit")
    changed = charge.revise(prepared_id=record.content_id)
    records = [("preparation", charge.preparation_id, changed), ("prepared", record.content_id, record)]
    key = run._state["workflow_current"]
    item = None if key is None else run._state["workflow_items"][key].revise(prepared_id=record.content_id)
    if item is not None:
        records.append(("workflow_item", key, item))
    run._write(tuple(records))
    handle = PreparedHandle._make(record, point, native, items, setting, construction.program.bindings)
    state = run._state
    state["prepared_artifacts"][record.content_id] = record
    state["local_prepared_ids"].append(record.content_id)
    state["handles"][record.content_id] = handle
    state["preparation_charges"][charge.preparation_id] = changed
    state["host_preparation_work"] += kernel.construction_work
    if item is not None:
        state["workflow_items"][key] = item
    run.progress("prepare", len(state["local_prepared_ids"]), None)
    return handle


def _submit_host(prepared, *, run, checkpoint_sequence=None):
    """Reserve one bound host invocation and atomically publish its declared scalar/array
    outputs.

    The reserved event commits before the kernel runs (commit 4 of the module
    docstring). There is no submission record, because nothing leaves the
    process. The chunk, the completed event and the kernel's arrays then commit
    together (commit 6). The kernel must return exactly its declared scalars in
    their declared frames, and each declared array or an explicit reason that it
    is unavailable. An ordinary exception raised by the kernel, or by these
    output checks after it returns, makes the attempt failed. An interruption
    such as ``KeyboardInterrupt`` leaves it uncertain (``Run._fail_attempt``).
    """
    from nwqlib._run_journal import _json_bound

    run.plan.method.before_submit(run.plan, (prepared,), run=run)
    kernel, native = prepared._native.record, prepared._native
    if native._realization_id is not None and native._realization_id != prepared.realization.content_id:
        raise ValueError("host closure differs from its prepared point")
    expected_bytes = readout_bytes(prepared, metadata_bytes=run.limits.max_completion_metadata_bytes,
            max_bytes=run.limits.max_data_bytes)
    run.check_capacity(data_bytes=expected_bytes)
    attempt = str(uuid4())
    event = ConsumptionEvent(execution="host_kernel", attempt=attempt, prepared_id=prepared.record.content_id,
        status="reserved", shots=0, evaluations=0, started=_now(), host_invocations=1,
        host_work=kernel.invocation_work, data_bytes_reserved=expected_bytes,
        checkpoint_sequence=checkpoint_sequence, artifacts_reserved=len(kernel.outputs),
        assessment_id=run._assessment_id(prepared.record))
    records = [("event", attempt, event)]
    current = run._state["workflow_current"]
    item = None if current is None else run._state["workflow_items"][current].revise(attempt=attempt)
    if item is not None:
        records.append(("workflow_item", current, item))
    run._write(tuple(records), reserve_output=expected_bytes)
    if item is not None:
        run._state["workflow_items"][current] = item
    run._install_attempt(event)
    index = run._state["attempt_indices"][attempt]
    if kernel.outputs:
        run._start_publications()
    try:
        run.progress("submit")
        started = perf_counter()
        try:
            output = native._invoke()
        finally:
            event = event.revise(timing=_native_timing(started, "nwqlib.blocks.kernels.BoundKernel._invoke"))
        if (tuple(item.label for item in output.scalars) != kernel.scalars
                or tuple(item.frame for item in output.scalars) != kernel.scalar_frames):
            raise ValueError("host scalar output differs from its selected declaration")
        # Require exactly the declared outputs or explicit unavailability. Publish
        # arrays with the same acquisition identity as the scalar observation.
        selected = {array.name: array for array in kernel.outputs}
        names = tuple(name for name, _ in output.arrays) + tuple(item.output.name for item in output.unavailable)
        if len(set(names)) != len(names) or set(names) != selected.keys():
            raise ValueError("host arrays differ from selected names or explicit missing output")
        manifests = []
        for name, array in output.arrays:
            selected[name].validate_array(array)
            handle = run.artifacts._publish(array, output=selected[name], provenance=dict(
                plan_id=run.plan.content_id, realization_id=prepared.realization.content_id,
                construction_id=prepared.record.construction_id, producer_id=kernel.content_id,
                acquisition=(run.run_id, attempt, attempt, "0"), source=kernel.implementation))
            manifests.append(handle.manifest)
        chunk = ObservationChunk(execution="host_kernel", run_id=run.run_id, plan_id=run.plan.content_id,
            realization_id=prepared.realization.content_id, prepared_id=prepared.record.content_id,
            experiment=prepared.realization.experiment, setting=prepared._setting, bindings=prepared._bindings,
            quantum_layout=(), classical_layout=(), attempt=attempt, job=attempt, chunk="0", observation=prepared.record.observation,
            population="unconditional", returned_shots=None, trajectories=None, values=output.scalars,
            artifacts=tuple(manifests), applications=output.applications, unavailable=output.unavailable,
            physical_scale=output.physical_scale, physical_scale_unavailable=output.physical_scale_unavailable,
            selected_kernel_id=kernel.content_id, source=kernel.implementation)
        # Declared receipts were reserved with the outputs. Only receipt bytes
        # beyond the declaration spend the variable metadata allowance.
        credit = (min(kernel.application_bytes, _json_bound(output.applications, run.limits.max_data_bytes))
                  if kernel.application_bytes else 0)
        event = event.revise(status="completed", finished=_now(), observation_id=chunk.content_id)
        run._finish_attempt(index, event, chunk, metadata_credit=credit)
        return chunk
    except BaseException as error:
        run._fail_attempt(index, event, error)
        raise


def _publish_amplitudes(prepared, result, *, run, attempt, result_key="0", point=None, statevector=None,
                        boundary=None, trajectory_id=None):
    """Reduce a native statevector to its selected amplitudes and publish them.

    Both the prepared native object and the result must describe the state
    before the final measurements. Any other trajectory would be a conditioned
    state. ``reduce_amplitudes`` selects the declared coordinates, physical scale
    and branch masses, and the array is published with this acquisition's
    identity, so the vector and its scalar masses commit together. For an
    amplitude point of a trajectory, ``point`` is its ID, ``statevector`` the
    state saved at its ``boundary`` and ``result_key`` its point chunk key.
    """
    from nwqlib.amplitudes import reduce_amplitudes, AMPLITUDE_MASS_LABELS, AMPLITUDE_MASS_UNAVAILABLE, validate_amplitude_masses
    from nwqlib.execution import ScalarValue
    observation = prepared.record.observation
    if point is not None:
        observation = observation.point_observation(point)
    declaration = observation.amplitudes
    if (prepared._native.metadata.get("statevector_semantics") != "pre_final_measurement"
            or result.metadata.get("statevector_semantics") != "pre_final_measurement"):
        raise ValueError("native amplitudes have incompatible trajectory semantics")
    array, scale, absent, masses = reduce_amplitudes(
        declaration, result.raw_output["statevector"] if point is None else statevector,
        max_bytes=run.limits.max_data_bytes, max_bytes_source=_data_limit_clause(run))
    values = tuple(ScalarValue(label=label, value=value, frame="encoded_branch",
        unavailable=AMPLITUDE_MASS_UNAVAILABLE if value is None else None)
        for label, value in zip(AMPLITUDE_MASS_LABELS if declaration.keep_masses else (), masses, strict=True))
    validate_amplitude_masses(declaration, values)
    job, manifests = result.metadata["native_job_id"], ()
    if array is not None:
        # reduce_amplitudes returns a new array that nothing else holds, so it
        # is handed over to publication instead of being copied again.
        handle = run.artifacts._publish(array, output=declaration.output, private=True, provenance=dict(
            plan_id=run.plan.content_id, realization_id=prepared.realization.content_id,
            construction_id=prepared.record.construction_id, producer_id=declaration.content_id,
            acquisition=(run.run_id, attempt, job, result_key), source=declaration.source))
        del array
        manifests = (handle.manifest,)
    return ObservationChunk(run_id=run.run_id, plan_id=run.plan.content_id, realization_id=prepared.realization.content_id,
        prepared_id=prepared.record.content_id, experiment=prepared.realization.experiment, setting=prepared._setting,
        bindings=prepared._bindings, quantum_layout=prepared.record.quantum_layout, classical_layout=(), attempt=attempt,
        job=job, chunk=result_key, observation=observation, population="unconditional",
        returned_shots=None, trajectories=1, values=values, source=declaration.source, artifacts=manifests,
        unavailable=() if absent is None else (absent,), physical_scale=scale,
        physical_scale_unavailable=None if scale is not None else "physical recovery is unavailable",
        point=point, boundary=boundary, trajectory_id=trajectory_id)


def restore_prepared(prepared_id, *, run, for_submission=False):
    """Rebuild the handle of a saved preparation from its receipt, without lowering.

    A fetch-only handle (``for_submission=False``) reads no saved native
    payload, so it never decodes the original native circuit. The receipt must
    still match the Plan's selected construction and readout and the Run's backend
    configuration. A fetch-only handle is enough to decode results of a known job.
    A handle for new execution also needs the saved native payload, which a
    durable Run stored with the receipt.
    """
    record = run.prepared_artifact(prepared_id)
    experiment, construction = record.realization._selected_construction(run.plan)
    admission = _Admission(construction.program)
    try:
        admission.admitted().require_ready()
        setting, observation = experiment._resolved_observation(admission, construction_id=construction.content_id)
    except AdmissionStepsExceeded as error:
        raise _named_admission_refusal(run.plan, error, "restoration") from error
    if record.plan_id != run.plan.content_id or record.observation != observation or record.construction_id != construction.content_id:
        raise ValueError("saved preparation differs from its original selected science/readout")
    items = _admit_readout(observation, width=sum(len(reg.bits) for reg in record.quantum_layout),
        classical_width=sum(len(reg.bits) for reg in record.classical_layout), run=run)
    if record.execution == "host_kernel":
        kernel, native = _host_binding(construction, observation, run=run)
        if record.selected_kernel_id != kernel.content_id:
            raise ValueError("saved host receipt differs from its actual selected kernel")
        native = native._at(record.realization, kernel, run)
    else:
        if record.backend_configuration_id != run.backend.content_id:
            raise ValueError("saved native handle belongs to another backend configuration")
        payload = None
        if for_submission and record.payload is not None:
            if run._state["journal"] is None:
                raise ValueError("new execution requires its original saved native bytes")
            payload = run._state["journal"].read_payload(record.payload)
        if for_submission and record.payload is None and getattr(run.backend, "requires_prepared_payload", True):
            raise ValueError("saved preparation has no native payload for new execution")
        native = run.backend.restore_native(record, payload, run=run)
    handle = PreparedHandle._make(record, record.realization, native, items, setting, construction.program.bindings)
    if for_submission:
        run._state["handles"][record.content_id] = handle
    return handle


def submit_detached(prepared, *, run, checkpoint_sequences=None):
    """Submit prepared circuits as one job to a backend that runs jobs outside this process, without waiting.

    [`submit`][nwqlib.scientist.submit] and `Run.resume` call it for such a
    backend, so most code never does. The [IonQ guide](../ionq.md) calls it
    directly to submit several circuits with equal shots as one job. The
    backend checks the whole batch, the submission is saved in the Run's
    folder, the backend then starts the job, and the job's locator is saved
    when the backend returns it. If the process ends between the start and
    the saved locator, the reopened Run has an uncertain submission without
    a locator, and only the backend's own lookup by the original submission
    can find the job. An error while starting the job marks the submission
    and its attempts uncertain, with their work still counted, because the
    provider may have accepted the request before the error reached this
    process. The job is never resubmitted or replaced. Read its results with
    `run.resume()`, `run.wait()` or `refresh_submissions`.

    Args:
        prepared (tuple[PreparedHandle, ...]): The prepared circuits of this
            Run and backend, at least one.
        run (Run): The Run that holds them. It needs a folder.
        checkpoint_sequences (tuple | None): For an adaptive Method, the
            iteration checkpoint of each circuit, in order, or `None`.

    Returns:
        submission (SubmissionRecord): The saved submission with its job
            locator.

    Raises:
        TypeError: If `prepared` is not a nonempty tuple of prepared circuits.
        ValueError: If the Run has no folder, the backend cannot start jobs,
            a circuit belongs to another backend, or `checkpoint_sequences`
            does not match `prepared`.
    """
    # The backend admits the whole batch and the Method's ``before_submit``
    # runs before the intent commits (commit 4 of the module docstring). Then
    # ``launch`` starts the provider job and the returned locator is
    # acknowledged (commit 5). Only the backend's ``reconcile`` (by the
    # original submission identity) can recover a job whose acknowledgement
    # was lost.
    if type(prepared) is not tuple or not prepared or any(type(handle) is not PreparedHandle for handle in prepared):
        raise TypeError("detached submission requires a nonempty tuple of native handles")
    # A trajectory item publishes one chunk per point from its refresh
    # (_refresh_outcomes). Its reservation is the trajectory's readout_bytes.
    for handle in prepared:
        handle._restore_native()
    if checkpoint_sequences is not None and (len(checkpoint_sequences) != len(prepared)
            or any(sequence is not None and type(sequence) is not int for sequence in checkpoint_sequences)):
        raise ValueError("controller checkpoint sequences must match the ordered batch")
    if run._state["journal"] is None or not callable(getattr(run.backend, "launch", None)):
        raise ValueError("detached submission requires a durable run and selected launcher")
    for handle in prepared:
        _check_scope(run, handle.realization)
        if handle.record.backend_configuration_id != run.backend.content_id:
            raise ValueError("batch contains another backend's prepared artifact")
    natives = tuple(handle._native for handle in prepared)
    run.backend.admit_batch(natives)
    run.plan.method.before_submit(run.plan, prepared, run=run)
    identity, started = str(uuid4()), _now()
    events, items = [], []
    for index, handle in enumerate(prepared):
        observation, attempt = handle.record.observation, str(uuid4())
        events.append(ConsumptionEvent(attempt=attempt, prepared_id=handle.record.content_id, submission=identity,
            status="reserved", started=started, shots=observation.shots, evaluations=int(observation.kind != "counts"),
            data_bytes_reserved=readout_bytes(handle, metadata_bytes=run.limits.max_completion_metadata_bytes,
            max_bytes=run.limits.max_data_bytes, run=run, attempt=attempt, result_key=str(index)),
            provider_managed_sampling=observation.kind == "estimated_observable",
            checkpoint_sequence=None if checkpoint_sequences is None else checkpoint_sequences[index]))
        items.append(SubmissionItem(attempt=attempt, prepared_id=handle.record.content_id, item=index, result_key=str(index)))
    submission = SubmissionRecord(submission_id=identity, run_id=run.run_id, backend=prepared[0].record.target,
                                  items=tuple(items), status="intent", started=started)
    installed = run._begin_submission(submission, tuple(events), tuple(handle.record for handle in prepared))
    try:
        run.progress("submit")
        locator = run.backend.launch(natives, submission_id=identity, run=run)
        return run._ack_submission(identity, locator)
    except BaseException as error:
        if not run._state["storage_failed"]:
            previous = run._state["submissions"][identity]
            value = previous.revise(status="uncertain", failure=f"{type(error).__name__}: {error}")
            failed = [(index, event.revise(status="uncertain", failure=str(error))) for index, event in installed]
            run._write((("submission", identity, value), *(("event", event.attempt, event) for _, event in failed)))
            run._state["submissions"][identity] = value
            for index, event in failed:
                run._state["events"][index] = event
        raise


def _merged_associations(submission, reported, *, run):
    """Return the submission's items, in their original order, with new provider identifiers filled in.

    A provider may learn a child job or a result identifier after launch, and
    ``reported`` carries them as ``SubmissionItem`` revisions keyed by
    ``result_key``. Each reported item must name a distinct original item. It
    may fill ``child_job`` or ``provider_result_id`` where the original is None
    and must repeat a known identifier unchanged. Changing any other field, or
    replacing a known identifier, raises, because that would rewrite the
    originally admitted acquisition.
    """
    # A cheap size check before the lookup tables below are built. The stored
    # items already fit the data cap at more than 32 bytes each (an attempt UUID
    # alone has 36 characters), so this check rejects only an oversized provider
    # report, before the loop walks it.
    count = len(submission.items) + len(reported)
    run.check_data(32*count)
    associations = {item.result_key: item for item in submission.items}
    seen = set()
    for item in reported:
        if type(item) is not SubmissionItem or item.result_key not in associations or item.result_key in seen:
            raise ValueError("provider association requires a unique original acquisition item")
        previous = associations[item.result_key]
        if (any(getattr(item, name) != getattr(previous, name)
                for name in SubmissionItem.model_fields
                if name not in {"parent_id", "child_job", "provider_result_id"})
                or any(getattr(previous, name) is not None
                       and getattr(item, name) != getattr(previous, name)
                       for name in ("child_job", "provider_result_id"))):
            raise ValueError("provider cannot replace an original acquisition or known child/result")
        fields = {name: getattr(item, name) for name in ("child_job", "provider_result_id")}
        if any(value != getattr(previous, name) for name, value in fields.items()):
            associations[item.result_key] = previous.revise(**fields)
        seen.add(item.result_key)
    return tuple(associations[item.result_key] for item in submission.items)


def refresh_submissions(*, run: Run):
    """Read each pending job of a Run once and save the new results, without submitting anything.

    `Run.resume` and `Run.wait` call it, so most code never does. Each new
    result is saved with its attempt in one step and can be collected once.
    A job read twice still saves each result once, because results that are
    already saved are skipped. A submission whose locator was lost is read
    only through the backend's own lookup by the original submission. Items
    of a job keep their original positions and result keys, and a provider
    may add a child job or result identifier to an item but cannot replace
    one. When a provider reports failure, cancellation or
    an uncertain state, the remaining attempts become uncertain and stay
    counted. If reading a job fails, the failure is recorded on its
    submission and the original exception propagates, with the job's
    locator, counted work and earlier results kept, so a later call can read
    the same job again.

    Args:
        run (Run): The Run whose pending jobs are read.

    Returns:
        observations (tuple[ObservationChunk, ...]): The newly saved
            observations.
    """
    # Each refresh splits commit 6 of the module docstring into two
    # transactions, provider status first (``_commit_refresh_status``) and
    # observations second (``_commit_refresh_outcomes``). A reconciled locator
    # is acknowledged (commit 5) before the refresh reads the job. A
    # cancellation request is written before it is sent
    # (``_request_cancellation``), and a failed refresh is recorded on its
    # submission (``_record_refresh_failure``).
    with run._lock:
        run._ensure_open()
        observed = []
        for submission_id in tuple(run._state["pending_submissions"]):
            submission = run._state["submissions"][submission_id]
            pending = tuple(item for item in submission.items if item.attempt not in run._state["by_attempt"])
            if not pending or submission.results_consumed:
                run._state["pending_submissions"].pop(submission_id, None)
                continue
            if submission.locator is None and not callable(getattr(run.backend, "reconcile", None)):
                continue  # Method may still publish a lawful partial Result before recovery fails.
            run._backend_notice()
            try:
                handles, natives = _refresh_natives(run, submission, pending)
                if submission.locator is None:
                    locator = run.backend.reconcile(submission, natives, run=run)
                    if locator is None:
                        continue
                    submission = run._ack_submission(submission_id, locator)
                submission = _request_cancellation(run, submission_id, submission)
                update = run.backend.refresh(submission.locator, natives, run=run)
                submission, revised = _commit_refresh_status(run, submission_id, submission, update)
                if any(_publishes_arrays(handle.record.observation) for handle in handles.values()):
                    run._start_publications()
                changes, chunks = _refresh_outcomes(run, submission, pending, handles, update)
                terminal = update.status in {"completed", "failed", "cancelled"} and update.error is None
                if terminal:
                    revised = revised.revise(results_consumed=True)
                observed.extend(_commit_refresh_outcomes(run, submission_id, revised, changes, chunks))
                if (revised.results_consumed
                        or all(item.attempt in run._state["by_attempt"] for item in submission.items)):
                    run._state["pending_submissions"].pop(submission_id, None)
                    for item in submission.items:
                        run._state["pending_output"].pop(item.attempt, None)
                if update.error is not None:
                    raise update.error
            except Exception as error:
                _record_refresh_failure(run, submission_id, error)
                raise
        return tuple(observed)


def _refresh_natives(run, submission, pending):
    """Restore what one refresh of ``submission`` needs from its saved receipts.

    Returns ``(handles, natives)``. ``handles`` maps the attempt of each
    pending item (one without an observation) to its restored
    ``PreparedHandle``, which decodes that item's result. ``natives`` holds the
    backend's native object for every item in original ordinal order, and an
    item that already has an observation is restored from its receipt alone.
    The adapter receives the complete association because a provider numbers
    batch items by their original ordinals, even after partial publication.

    A Run's live handle with its native object serves directly. Otherwise
    each restored handle and native object is kept in the Run's private
    ``refresh_cache`` (released at ``close``), so later polls of the same job
    restore nothing again.
    """
    live, cache = run._state["handles"], run._state["refresh_cache"]

    def handle(prepared_id):
        found = live.get(prepared_id)
        if found is not None and found._native is not None:
            return found
        if ("handle", prepared_id) not in cache:
            cache["handle", prepared_id] = restore_prepared(prepared_id, run=run)
        return cache["handle", prepared_id]

    def native(prepared_id):
        found = live.get(prepared_id)
        if found is not None and found._native is not None:
            return found._native
        if ("handle", prepared_id) in cache:
            return cache["handle", prepared_id]._native
        if ("native", prepared_id) not in cache:
            cache["native", prepared_id] = run.backend.restore_native(run.prepared_artifact(prepared_id), run=run)
        return cache["native", prepared_id]

    handles = {item.attempt: handle(item.prepared_id) for item in pending}
    natives = tuple(handles[item.attempt]._native if item.attempt in handles else native(item.prepared_id)
                    for item in submission.items)
    return handles, natives


def _request_cancellation(run, submission_id, submission):
    """Forward a Run cancellation to one unfinished job at most once, and return the submission.

    The request is written on the submission before the backend call, so a
    process that stops during the call reopens with the request recorded and
    never sends it a second time. Only a later refresh reports whether the job
    was cancelled, and the reserved exposure stays charged.
    """
    if (run.cancel_requested is not None and submission.cancel_requested is None
            and submission.status not in {"completed", "failed", "cancelled"}):
        submission = submission.revise(cancel_requested=run.cancel_requested)
        run._write((("submission", submission_id, submission),))
        run._state["submissions"][submission_id] = submission
        cancel = getattr(run.backend, "cancel", None)
        if callable(cancel):
            cancel(submission.locator, run=run)
        else:
            run._state["warnings"].append("backend has no cancellation operation; original job remains observable")
    return submission


def _commit_refresh_status(run, submission_id, submission, update):
    """Commit the provider's status and item associations, the first transaction of a refresh.

    Returns ``(submission, revised)``. ``submission`` has the child job and
    result identifiers reported in ``update.associations`` merged into its
    items (``_merged_associations``). ``revised`` is that submission with the
    reported status, which this function has just written.

    A poll that changes nothing is not written: when the status, provider
    status, failure, native simulation count and item associations equal the
    stored ones and no refresh failure is recorded, ``revised`` is the
    stored submission and no row is committed. A recorded refresh failure is
    cleared by the next successful refresh, which is then written. A
    terminal status keeps the ``finished`` time of its first report, so
    reading the same terminal status again changes nothing.
    """
    stored = submission
    if update.associations:
        submission = submission.revise(items=_merged_associations(submission, update.associations, run=run))
    terminal = update.status in {"completed", "failed", "cancelled"}
    finished = (stored.finished if terminal and stored.status == update.status and stored.finished is not None
                else _now() if terminal else None)
    revised = submission.revise(status=update.status, provider_status=update.provider_status,
        failure=update.failure, refresh_failure=None, native_simulations=update.native_simulations,
        finished=finished)
    # A revision records its parent. Every other field is compared.
    if all(getattr(revised, name) == getattr(stored, name) for name in type(stored).model_fields
           if name != "parent_id"):
        return submission, stored
    # The provider's status and associations commit first, on their own.
    # They are facts about the job that must survive even if decoding or
    # admitting its results fails later. The observations commit in the next
    # transaction, which also sets results_consumed once the job is terminal.
    # A crash between the two leaves results_consumed unset, so the next
    # refresh reads the same job again.
    run._write((("submission", submission_id, revised),))
    run._state["submissions"][submission_id] = revised
    if update.associations:
        for item in revised.items:
            run._state["submission_items"][item.attempt] = item
    return submission, revised


def _refresh_outcomes(run, submission, pending, handles, update):
    """Build, without writing, the event and observation rows of one refresh.

    Returns ``(changes, chunks)``. ``changes`` holds a completed event and its
    ``ObservationChunk`` for each returned result of a pending item (for a
    trajectory item, the tuple of its point chunks, whose ordered
    ``ObservationView`` the event binds), and an
    uncertain event for each pending item without a result when the provider
    reports failure, cancellation or an uncertain state. ``chunks`` pairs each
    completed event with its chunk. A result for an item that already has an
    observation is skipped, so a job read twice publishes each observation
    once. A completed job that reports no error but omits a pending item
    raises.
    """
    by_key = {item.result_key: item for item in submission.items}
    changes, chunks, seen = [], [], set()
    for key, result in update.results:
        item = by_key.get(key)
        if item is None or key in seen:
            raise ValueError("backend result has an unknown or repeated item coordinate")
        seen.add(key)
        if item.attempt not in handles:
            continue  # Already published. A second population is never created.
        chunk = _quantum_chunk(handles[item.attempt], result, run=run, attempt=item.attempt, result_key=key)
        if isinstance(chunk, tuple):
            # A trajectory item publishes the tuple of its point chunks with
            # its one event, which binds their ordered ObservationView.
            for point in chunk:
                item.validate_observation(point, run_id=run.run_id, locator=submission.locator)
            event = run.attempt_record(item.attempt).revise(status="completed", finished=_now(),
                observation_id=ObservationView(chunks=chunk).content_id, failure=None)
            changes.extend((("event", item.attempt, event), *(("chunk", point.content_id, point) for point in chunk)))
            chunks.append((event, chunk))
            continue
        item.validate_observation(chunk, run_id=run.run_id, locator=submission.locator)
        event = run.attempt_record(item.attempt).revise(status="completed", finished=_now(),
            returned_shots=chunk.returned_shots, observation_id=chunk.content_id, failure=None)
        changes.extend((("event", item.attempt, event), ("chunk", chunk.content_id, chunk)))
        chunks.append((event, chunk))
    if update.status in {"failed", "cancelled", "uncertain"}:
        for item in pending:
            if item.result_key not in seen:
                event = run.attempt_record(item.attempt).revise(status="uncertain", finished=_now(),
                    failure=update.failure or update.status)
                changes.append(("event", item.attempt, event))
    if (update.status == "completed" and update.error is None
            and any(item.result_key not in seen for item in pending)):
        raise ValueError("completed backend result omits an original acquisition item")
    return changes, chunks


def _commit_refresh_outcomes(run, submission_id, revised, changes, chunks):
    """Commit one refresh's events, observations and arrays with its submission row.

    This is the second transaction of a refresh. Live counters and indexes
    change only after it commits. Returns the newly published chunks. With no
    change to write, the unfinished array batch is discarded and nothing is
    written.
    """
    if not changes:
        run._abort_publications()
        return []
    changes.append(("submission", submission_id, revised))
    run._flush_publications(tuple(changes), release_output=tuple(event.attempt for event, _ in chunks))
    run._state["submissions"][submission_id] = revised
    for kind, key, event in changes:
        if kind == "event":
            run._state["events"][run._state["attempt_indices"][key]] = event
    published = []
    for event, chunk in chunks:
        # A trajectory attempt holds the tuple of its point chunks.
        points = chunk if isinstance(chunk, tuple) else (chunk,)
        for point in points:
            run._state["receipts"][point.acquisition_key] = point
        run._state["by_attempt"][event.attempt] = chunk
        for point in points:
            run._index_counts(chunk=point)
        published.extend(points)
    return published


def _record_refresh_failure(run, submission_id, error):
    """Record a failed refresh on its submission before the caller re-raises ``error``.

    An unfinished array batch is discarded first. After a journal failure
    nothing more is written, because the durable Run must be reopened. The
    original job locator stays stored. When the submission has a locator, a
    note added to ``error`` names that stored job and says that no
    replacement job was submitted.
    """
    run._abort_publications()
    if run._state["storage_failed"]:
        return
    previous = run._state["submissions"][submission_id]
    interrupted = previous.revise(refresh_failure=f"{type(error).__name__}: {error}")
    run._write((("submission", submission_id, interrupted),))
    run._state["submissions"][submission_id] = interrupted
    if previous.locator is not None:
        error.add_note(f"Original {previous.locator.provider} job {previous.locator.job_id} is stored; "
                       "no replacement job was submitted. Resolve the retrieval failure, then call "
                       "refresh_submissions on the same run.")
