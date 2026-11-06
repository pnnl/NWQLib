"""The single lifecycle of a Run, which prepares, submits, collects and saves the experiments of one Plan.

For one execution (one ``run_id``) of a Plan, the Run alone manages
preparation, submission, collection, the native and numerical caches, the
cumulative limits and the exposure (the counts of submitted jobs, circuits,
shots and provider-managed sampling, by outcome). The iteration code of an
adaptive Method (the code that chooses each next experiment), backend adapters
and the public scientist entry points all act through it. An acquisition, one
execution of a prepared circuit with its specified readout and shot count or
one invocation of a classical routine (a host kernel in the code), together
with the collection of the resulting data, is therefore reserved in one place,
its outcome is saved in one place, and one run log (``run.sqlite``, written by
``_run_journal.LocalJournal``) records both. A second executor inside the same
execution would need its own record of counted work, and the two records could
disagree about what was counted or which job produced which data. Repeating a
Plan's work deliberately creates a second Run with its own record.

Lifecycle rules and the failure each one prevents:

1. Reserve before work. A preparation commits its charge and the RNG position
   after its seed draw, and a submission commits its intent, reserved events and
   output-byte reservation, before any native or provider call starts. A crash
   inside the call then leaves a saved record that the draw was used and the
   work was counted. Without that record a restart could reuse a draw or submit
   the same work twice.
2. Save outcomes atomically. A completed observation, its event, its submission
   status and every array saved with it commit in one transaction of the run
   log. The run log never holds an observation without the record of its
   counted work, or an array without the acquisition that produced it.
3. An unrecoverable outcome stays uncertain. An interrupted or unacknowledged
   acquisition keeps its reserved circuits and shots counted and is never relabelled
   failed, resubmitted or refunded, because the provider may already have executed
   or billed it. Only a classical routine that finished with an ordinary exception becomes
   a failed attempt. A provider-reported failure or cancellation is recorded on the
   submission, and its unfinished attempts stay uncertain, because the provider
   does not report how much of the work ran. When no retrieval route remains,
   ``Run.resume`` raises ``RunFailed`` with stage ``"recovery"`` and status
   ``"uncertain"``. A reopened synchronous Aer submission with no locator is
   marked failed with its results consumed because its in-process output is
   unavailable. Its attempt stays uncertain and counted, and continuation
   raises the recovery error with that attempt's identity when the Method
   cannot return a valid Result and cancellation has not been requested.
   ``Run._restore_submissions`` makes this change and releases the
   submission's output reservation.
4. Pending work keeps its identity. Retrieval uses the original locator, attempt
   identities and submission item ordinals, and reopening restores the RNG
   position and the collection ordinals. A refresh error is recorded on the same
   submission and propagates. Nothing is launched in its place.
5. In-memory state follows the commit. In-memory counters, dictionaries and
   handles change only after the transaction of the run log succeeds, or are
   rolled back when it fails. A failed write to the run log sets
   ``storage_failed``, which blocks further work until the Run is reopened from
   its folder, so memory never runs ahead of disk.

Commits of one static acquisition, in the order they happen, and what reopening
finds when the process stops after each one. A static Plan has a fixed list of
points, its experiments and each range-axis point of an experiment with axes
(``Plan._static_points``), each prepared and acquired once by ``prepare_static`` and
``execute_static``. ``prepare_static`` prepares the first point, or all of
them for ``prepare(plan, settings="all")``, and ``execute_static`` prepares each
remaining one when it reaches it. The iteration code of an adaptive Method
chooses its next experiment from data. It uses the same preparation, submission
and outcome commits, with a checkpoint of its iteration state
(``Run.checkpoint``) in place of the workflow row. The numbering is the same as
in ``docs/development/execution.md``, and each function below that makes one of
these commits names it in its docstring. The docstrings of the other writes (limit amendments, synthesis
reservations, cancellation requests, failure revisions and the Result row)
state what a stop before or after them leaves.

1. Workflow rows of a template batch, with their runtime seeds and the RNG
   position, in one transaction (``_static_items``). A restart before commit 2
   prepares each point with its same seed. A refused or failed commit keeps
   none of the seeds, so a retry draws them again from the same position.
   A point prepared on its own has no
   commit 1: its row and seed commit with its preparation charge, so a restart
   before that charge draws the seed again from the same saved RNG position.
2. Preparation charge and RNG position (``prepare_experiment``). A seed
   drawn for the charge is kept only if the charge commits, so a retry of a
   refused charge draws it again from the same position. If the
   preparation record has not committed, reopening counts the circuit
   preparation against ``max_total_circuits``. The Run never repeats a failed
   or interrupted preparation on its own, since that would be new counted work
   the caller did not request, so a static row whose local preparation has a
   charge and no preparation record stops the Run (``_unbuilt``), and
   ``execute_static`` raises ``PreparationNotRebuilt`` before it submits any
   further experiment. ``Run.resume(rebuild_preparations=True)`` is the caller's
   request for that work. It prepares each such row once more under the row's
   saved realization and seed (``_rebuild_static_preparations``), and the old
   charge stays counted. The new charge and the row's pointer to it commit
   together, so a reopen finds ``charged/no-record``,
   ``new charged/no-record``, ``prepared record`` or ``submitted``, and never a
   mix. A saved remote preparation can still be refreshed instead. An adaptive
   Method has no workflow rows, so it keeps its own refusal.
3. Preparation record and native payload (``_finish_native_preparation`` or
   ``_prepare_host``). A later resume submits the saved preparation without
   lowering it again. The charge row of this transaction adds the work of the
   exact dense syntheses that the preparation made (``Run._record_synthesis``).
   A remote preparation adds it when ``begin_preparation`` saves its local
   data.
4. Submission intent, reserved events and output-byte reservation
   (``Run._begin_submission``). The backend is contacted only after this
   commit. If no acknowledgement or outcome follows, reopening marks the intent
   and its events uncertain. Only a backend ``reconcile`` can then find the job,
   by the original submission identity. A synchronous backend has no
   acknowledgement step, so this window lasts until commit 6. Without a
   retrieval route, resume raises the recovery error. A reopened synchronous
   Aer submission with no locator follows lifecycle rule 3. A classical routine
   (``_submit_host``) commits only its reserved event, because nothing leaves
   the process. Reopening makes that event uncertain, and no route can
   retrieve its outcome.
5. Acknowledged locator (``Run._ack_submission``, detached backends only). The
   submission stays acknowledged and its events reserved. The next refresh
   retrieves the original job by its locator.
6. Outcome (``Run._finish_attempt``). The completed event, the observation chunk,
   the submission status and any arrays saved with the observation commit in
   one transaction, so reopening never finds one without the others.
7. Collection ordinal (``Run.collect``). The Method's collection of the
   observation is its own commit, so reopening restores the order in which the
   Method collected observations even when they completed in a different order.
   The static workflow collects each synchronous outcome as it completes, so
   its ordinals commit with commit 6 (``submit_experiment(collect=True)``).

A refresh of a detached job (``refresh_submissions``) splits commit 6 in two. It
first commits the provider status and item associations, which must survive a
later decoding error. The observations commit in a second transaction, which also
sets the submission's ``results_consumed`` once the job is terminal. If the
process stops between the two, the flag is still unset, so the next refresh reads
the same job again and skips items already saved. Each observation is
therefore saved once.

These windows describe an interrupted process, for example a keyboard interrupt,
an unhandled exception or a killed process. Recovery after power loss is not
covered.
"""

from nwqlib._limits import ResourceLimitError, check_memory, exceeds, known_sum, memory_limit

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from threading import RLock
from time import monotonic, perf_counter, sleep
from typing import NamedTuple
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
    RunFailed, ScalarValue, SubmissionRecord, TimingObservation, _unbounded, LOCAL_SIMULATORS, with_paid_budgets,
)
from nwqlib._run_journal import (
    RUN_FORMAT, JSONBytesExceeded, LocalJournal, PreparationCharge, WorkflowItem, encode_records,
    unsupported_run_format,
)
from nwqlib.ir.validation import _Admission, children
# The recorded failure of a reopened synchronous Aer submission whose
# in-process result is gone (Run._restore_submissions). Its attempt has no
# route to an outcome, so resume raises the recovery error with this text
# (Run._raise_unrecoverable_outcome).
_AER_REOPEN_FAILURE = "interrupted synchronous Aer result is unavailable after reopen"
# Run._write checks a write against the Run's own max_data_bytes unless a limit
# amendment passes the proposed one, which may be None (no limit).
_RUN_DATA_LIMIT = object()
# Bytes that one prepared experiment keeps when it binds a shared template, whose
# circuit counts once: its handle, preparation record and parameter values. Sampled
# LCHS Runs grew about this much per row, linearly to 10**5 rows, with the handle
# dominating, on Qiskit 2.6.0 and Aer 0.18.0. A coarse estimate, not a byte bound.
_TEMPLATE_ROW_BYTES = 32 << 10
# Circuit containers of a template's size that exist together while it is lowered and
# compiled on Qiskit 2.6.0 and Aer 0.18.0, counted live on a small template. A new
# template has three: its built root, the gate definition that ``circuit_to_gate``
# fills, and either the copy in which that function assigns parameters or then the
# definition that Aer translates. The template before it has two: its gate definition
# and its translated definition. Containers can share operations, so these are coarse
# allowances, not byte counts.
_TEMPLATE_BUILD_COPIES = 3
_TEMPLATE_PREDECESSOR_COPIES = 2


def _now():
    return datetime.now(timezone.utc).isoformat()


def _limit_remedy(name):
    """The sentences of a refusal message that say how to go on past the execution limit ``name``.

    ``Run.extend_limits`` raises or removes a limit of the Run that refused,
    which keeps its saved circuits, jobs and results. A preparation of a Plan
    with fixed settings that the refusal left counted but not saved is prepared
    again by ``Run.resume(rebuild_preparations=True)``, so the same steps lead
    past both kinds of refusal there. An adaptive Method has no such rows, so the
    way past that case is a new Run, which the message names. A closed Run,
    including one that ``solve`` closed when the error left it, is reopened with ``load_run``
    first. ``solve``, ``prepare`` and ``Run`` take ``limits`` for a new Run.
    """
    return (f"Continue this Run with run.extend_limits({name}=...), a larger value or None for no limit, which keeps "
            "its saved circuits, jobs and results, and then run.resume(). "
            "If a Plan with fixed settings stopped while saving an already counted preparation, use "
            "run.resume(rebuild_preparations=True), which prepares it once more and sends no job again. "
            "A Method whose later settings depend on earlier outcomes cannot prepare it again: a new Run with a "
            "larger limit prepares the Plan again and can repeat earlier jobs, and the original saved observations "
            "stay readable. "
            "A closed Run with a folder, including one that solve closed "
            f"after this error, is reopened with load_run first. A new solve, prepare or Run takes "
            f"limits=ExecutionLimits({name}=...).")


def _data_limit_clause(run, unit=""):
    """The clause that names a Run's ``max_data_bytes`` as the source of a limit passed to a shared check.

    ``readout_shape`` and ``admit_materialization`` append it to a refusal,
    because they name only their own ``max_items`` or ``max_bytes``. ``unit``
    states how the passed limit follows from ``max_data_bytes``.
    """
    return (f"which is this Run's max_data_bytes={run.limits.max_data_bytes}{unit}. "
            f"{_limit_remedy('max_data_bytes')}")


def _cumulative_limit_error(name, limit, noun, used, requested, *, owner="this Run's", at_least=False):
    """Return the error that refuses ``requested`` more units of ``noun`` when ``used`` count against ``name``.

    The message gives the limit, both amounts, their sum as the smallest
    limit that accepts this request, and the remedy. ``owner`` names whose
    limit it is, ``"the proposed"`` for a value that ``extend_limits`` has
    not yet applied. The request is an exact amount, or a lower bound when
    ``at_least`` is set. Later work of the Run, such as the next step of an
    adaptive Method or the next saved result, can need more than that sum.
    """
    return ValueError(f"{owner} {name}={limit} cannot admit the requested {noun} ({used} counted, "
                      f"{'at least ' if at_least else ''}{requested} requested, at least {used + requested} needed). "
                      f"{_limit_remedy(name)}")


def _notebook_progress():
    """Return a progress display in one updatable notebook output, or None outside a notebook. Ordinary Python does not import IPython."""
    ipython = sys.modules.get("IPython")
    shell = None if ipython is None else ipython.get_ipython()
    if getattr(shell, "kernel", None) is None:
        return None
    handle = None

    def render(stage, done, total):
        """Show one line of text for ``stage`` in a single updatable notebook output.

        The first call creates the display and later calls replace its text, so a
        long Run shows its current stage instead of appending one line per event.
        ``done`` is a running count (preparations or collected
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
        # The header below saves the circuit and shot budgets of a backend
        # that may charge for them, so a reopened Run keeps the same budgets.
        limits = with_paid_budgets(limits, backend)
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
                # Resolved once here, so a later change of working directory cannot lose the folder.
                # Nothing saved holds this path, so a moved folder still opens.
                path = Path(directory).resolve() if directory is not None else Path.home() / ".nwqlib" / "runs" / identity
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
        - ``pending_output`` maps each attempt whose outcome is not yet saved to
          its reserved output bytes.
        - Saved data is split into ``data_bytes`` (JSON rows of the run log),
          ``array_bytes`` (saved observation arrays), ``payload_bytes`` (other
          binary payloads) and ``external_bytes`` (files of the Run folder).
          Together with ``pending_output`` they count against ``max_data_bytes``.
        - ``synthesis_work`` sums the exact dense synthesis work of the saved
          preparations. ``synthesis_preparation`` names the preparation in
          progress, and ``synthesis_pending`` holds the work of the syntheses it
          has made (``_record_synthesis``) until the transaction that first
          saves its lowering result. ``synthesis_cache`` holds the circuits of
          the syntheses that basis lowering made in this Run object's
          preparations (``_exact_dense_unitaries``), each made once, and
          ``synthesis_cache_kept`` the sum of their kept allowances
          (``dense_synthesis_size``). :meth:`close` empties the cache. It is not
          saved, so a reopened Run starts with an empty cache.
        - ``receipt_commits`` maps the charge of a remote preparation that a
          refresh may still complete to the ``(commit, withdraw)`` callbacks
          of the instructions its lowering copied
          (``_ConstructorContext.report_copies``) and to the specializations
          it built. These become the frontier
          that later preparations reuse only when a refresh saves its
          record, so no other preparation borrows gates whose copies no saved
          record counts. The commits run once its record is saved, and a
          failed or cancelled preparation runs the withdrawals and drops both
          (``_withdraw_preparation``). A local preparation publishes its
          specializations just before its record is saved and passes its
          callbacks to ``_finish_native_preparation`` directly. It is not
          saved, and :meth:`close` empties it.
        - ``native_held`` sums the memory estimates of the native circuits that
          live handles hold (``PreparedHandle._native_bytes``). Saving a
          preparation record adds to it, ``release_native`` subtracts only what
          it releases, ``PreparedHandle._restore_native`` adds back and
          :meth:`close` clears it. It is not saved, so a reopened Run starts it at
          zero. An Aer handle's saved row keeps its estimate, which is compared
          with the Method's limit before the circuit's file is read and added when
          the circuit returns. A handle of any other backend restored from its
          record adds nothing (``restore_prepared``, ``_held_bytes``).
          ``_held_bytes`` adds the definitions and frontiers, which it reads from
          the live caches.
        - ``static_workload_admitted`` records that this Run object has checked
          the known work of its remaining static points (``_admit_static_workload``).
        """
        for name, value in (("plan", plan), ("backend", backend), ("run_id", identity), ("_lock", RLock()),
                            ("forecast", forecast), ("allocation", allocation)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_state", dict(
            limits=limits, initial_limits=limits, limit_amendments=[],
            rng=RandomStreams.restore(plan.randomness), closed=False, storage_failed=False,
            preparations=0, host_preparations=0, construction_work=0, host_preparation_work=0,
            shots=0, host_invocations=0, host_work=0, synthesis_work=0, synthesis_preparation=None,
            synthesis_pending=0, synthesis_cache={}, synthesis_cache_kept=0,
            events=[], attempt_indices={}, submissions={}, submission_items={}, pending_submissions={},
            chunks={}, receipts={}, by_attempt={}, prepared_artifacts={}, local_prepared_ids=[], handles={},
            template_sources={}, template_classes={}, receipt_commits={},
            refresh_cache={}, observation_view=None, trace_snapshot=None,
            counts_sources=None, counts_source_failure=None,
            preparation_charges={}, checkpoint_preparations={}, checkpoint_attempts={},
            checkpoint_fields=None, checkpoint_sequence=0,
            remote_preparations={}, definitions=({}, {}), specializations={}, backend_context={},
            method_context=None, workflow_items={}, workflow_current=None, pending_output={},
            data_sizes={}, data_bytes=0, array_bytes=0, payload_sizes={}, payload_bytes=0, external_bytes=0,
            artifacts=None, publication_batch=None,
            artifact_payloads={}, journal=None, directory=None, started=False, result=None,
            cancel_requested=None, termination_reason=None, native_held=0, template_circuits={},
            static_workload_admitted=False,
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
        """The Run's folder, as an absolute `pathlib.Path`, or `None` for a Run without one.

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
        """Refuse work on a closed Run, or after a failed write to the run log.

        After a failed write the in-memory state may no longer match the run
        log, so only reopening the Run from its folder (lifecycle rule 5) can
        continue.
        """
        if self._state["closed"]:
            raise ValueError("run is closed")
        if self._state["storage_failed"]:
            raise ValueError(f"run storage failed; close this Run (after a first submit, prepared.run) and "
                             f"load the durable run from {self.directory} with nwqlib.load_run before further work")

    def check_data(self, size):
        """Reject one requested or decoded object larger than the whole data limit.

        This is a cheap per-object check before the object is built or read.
        Cumulative saved data is checked separately by ``check_capacity`` and at
        each write to the run log.
        """
        if exceeds(size, self.limits.max_data_bytes):
            raise ValueError(f"native output or decoded data of {size} bytes exceeds this Run's "
                             f"max_data_bytes={self.limits.max_data_bytes}. {_limit_remedy('max_data_bytes')}")

    def _write(self, records=(), *, collected=(), payloads=(), release_output=None, reserve_output=0,
               max_data_bytes=_RUN_DATA_LIMIT, deleted=()):
        """Check the changed records against the data limit, then commit them to the run log and the in-memory state.

        This is the single write path of a Run. ``max_data_bytes`` is an
        admission limit, the Run's own unless a limit amendment passes the
        proposed one, and None sets no limit. Work and derived records are
        admitted while::

            data_bytes + array_bytes + payload_bytes + external_bytes
                + pending_output <= max_data_bytes

        A write that settles admitted work (``_settles``) is never refused by
        this limit. Its rows are measured and counted, so the saved total can
        exceed the limit by lifecycle rows of admitted attempts that no
        reservation could bound, such as provider text. Every other write is
        encoded once and refused when it adds bytes and the total would
        exceed the limit, so after such an excess new work and new derived
        records wait for ``Run.extend_limits(max_data_bytes=...)``. A write
        that adds no bytes, such as one that only deletes cache rows, is not
        refused. Output reservations named by ``release_output`` leave the
        pending total in the same step, because their acquisition's outcome
        is being saved. A Run with a folder commits the rows in one SQLite
        transaction. The in-memory byte counters change only after that
        commit. A failed commit sets ``storage_failed`` so later work stops
        until the Run is reopened from disk. Only cache rows and removed
        checkpoint fields may be deleted. Preparation records, events and
        observations are permanent records.

        Returns:
            The encoded ``(kind, key, text)`` rows that were written.
        """
        state = self._state
        data_limit = self.limits.max_data_bytes if max_data_bytes is _RUN_DATA_LIMIT else max_data_bytes
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
        replaced = sum(state["data_sizes"].get(key, 0) for key in keys | removed)
        counted = (state["data_bytes"] - replaced + state["array_bytes"] + state["payload_bytes"]
                   + state["external_bytes"] + pending)
        available = None if data_limit is None else data_limit - (counted + payload_added + reserve_output)
        owner = "this Run's" if max_data_bytes is _RUN_DATA_LIMIT else "the proposed"
        settles = self._settles(records, deleted=deleted, reserve_output=reserve_output,
                                max_data_bytes=max_data_bytes)
        if settles or data_limit is None:
            # The rows of admitted work are bounded by their own decoding
            # checks, so their encoding takes no bound from the data limit.
            encoded = encode_records(records, None)
        else:
            # A row that replaces at least its own size adds nothing, so the
            # encoding bound covers the replaced bytes even beyond the limit.
            bound = max(0, available, replaced)
            try:
                encoded = encode_records(records, bound)
            except JSONBytesExceeded as error:
                minimum = payload_added + reserve_output + bound + 1
                raise _cumulative_limit_error("max_data_bytes", data_limit, "stored data bytes", counted, minimum,
                                              owner=owner, at_least=True) from error
        sizes = {(kind, key): size for (kind, key, _), size in zip(encoded.rows, encoded.sizes, strict=True)}
        added = sum(sizes.values()) - replaced
        requested = sum(sizes.values()) + payload_added + reserve_output
        if not settles and added + payload_added + reserve_output > 0 and exceeds(counted + requested, data_limit):
            raise _cumulative_limit_error("max_data_bytes", data_limit, "stored data bytes", counted, requested,
                                          owner=owner)
        journal = state["journal"]
        if journal is not None:
            try:
                journal.commit(records, collected=collected, payloads=payloads, deleted=deleted, encoded=encoded)
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

    def _settles(self, records, *, deleted, reserve_output, max_data_bytes):
        """Whether a write only settles admitted work, so that ``max_data_bytes`` does not refuse it.

        Such a write reserves no output, is not a limit amendment, deletes no
        row, and each of its rows is one of:

        - a revision of an ``event`` or ``submission`` row already saved, or of
          a saved ``remote_preparation`` row,
        - a ``chunk`` or ``publication`` row of an attempt whose output is
          still reserved or whose event is revised in the same write, or a
          ``payload`` row of such a publication,
        - the ``control`` row that only sets a cancellation request.

        A write without rows, such as a collection ordinal, settles too. The
        decision reads the rows, not ``release_output``. A ``remote_preparation``
        revision that starts an upload or a compilation settles too, but the
        stage is refused before it is written while the Run's counted data
        exceed the limit (``_remote_preparation._advance``), because it starts
        new remote work.
        """
        if reserve_output or max_data_bytes is not _RUN_DATA_LIMIT or deleted:
            return False
        state = self._state
        saved, pending = state["data_sizes"], state["pending_output"]
        revised = {value.attempt for kind, key, value in records if kind == "event" and (kind, key) in saved}

        def admitted(attempt):
            return attempt in pending or attempt in revised

        payloads = set()
        for kind, key, value in records:
            if kind == "publication" and admitted(value["acquisition"][1]) and "payload" in value:
                payloads.add(value["payload"].content_id)
        for kind, key, value in records:
            if kind in {"event", "submission", "remote_preparation"}:
                settled = (kind, key) in saved
            elif kind == "chunk":
                settled = admitted(value.attempt)
            elif kind == "publication":
                settled = admitted(value["acquisition"][1])
            elif kind == "payload":
                settled = key in payloads
            elif kind == "control":
                settled = (value["cancel_requested"] is not None and state["cancel_requested"] is None
                           and value["termination_reason"] == state["termination_reason"])
            else:
                settled = False
            if not settled:
                return False
        return True

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
        """The current method cache. Each access marks a non-dictionary context for saving at
        the next write to the run log, unless the caller marks its own changes with cache_method_dirty."""
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
        """Check new work against the cumulative limits before it starts.

        Circuit preparations and circuit acquisition attempts are counted
        separately, each against ``max_total_circuits``. Host preparations and
        host invocations carry their own work counters and are excluded. Shots and
        attempts include every reserved attempt, completed, uncertain or failed, so
        work already counted never frees capacity. A cancellation request blocks new
        preparation or acquisition.
        """
        self._ensure_open()
        state = self._state
        if self.cancel_requested is not None and any((preparations, circuits, shots)):
            raise ValueError("run cancellation prevents new preparation or acquisition")
        limits = self.limits
        circuit_preparations = state["preparations"] - state["host_preparations"]
        if exceeds(circuit_preparations + preparations, limits.max_total_circuits):
            raise _cumulative_limit_error("max_total_circuits", limits.max_total_circuits, "circuit preparations",
                                          circuit_preparations, preparations)
        acquisitions = len(state["events"]) - state["host_invocations"]
        if exceeds(acquisitions + circuits, limits.max_total_circuits):
            raise _cumulative_limit_error("max_total_circuits", limits.max_total_circuits,
                                          "circuit acquisition attempts", acquisitions, circuits)
        if exceeds(state["shots"] + shots, limits.max_total_shots):
            raise _cumulative_limit_error("max_total_shots", limits.max_total_shots, "raw shots",
                                          state["shots"], shots)
        stored = (state["data_bytes"] + state["array_bytes"] + state["payload_bytes"] + state["external_bytes"]
                  + sum(state["pending_output"].values()))
        if exceeds(stored + data_bytes, limits.max_data_bytes):
            raise _cumulative_limit_error("max_data_bytes", limits.max_data_bytes, "stored data bytes",
                                          stored, data_bytes)

    def _available_data_bytes(self):
        """Bytes of ``max_data_bytes`` not used by saved rows, arrays, payloads, external files or pending output, or None without a limit."""
        if self.limits.max_data_bytes is None:
            return None
        state = self._state
        stored = (state["data_bytes"] + state["array_bytes"] + state["payload_bytes"] + state["external_bytes"]
                  + sum(state["pending_output"].values()))
        return max(0, self.limits.max_data_bytes - stored)

    def _record_synthesis(self, work):
        """Add ``work`` units of exact dense synthesis to the preparation in progress.

        Basis lowering (:meth:`_exact_dense_unitaries`) and lowering that
        controls a transformed block call this for each batch of syntheses
        they make. The amount waits in memory and is saved with the
        transaction that first saves the preparation's lowering result
        (``_finish_native_preparation``, ``_bind_template_rows`` or
        ``begin_preparation``). Only then does it join
        ``ExecutionTrace.synthesis_work``, so a preparation that fails or is
        interrupted before that transaction counts no synthesis. A backend
        called directly with this Run, outside ``prepare_experiment``, makes
        no preparation of the Run, so its work is not counted.
        """
        if self._state["synthesis_preparation"] is not None:
            self._state["synthesis_pending"] += work

    def _exact_dense_unitaries(self, circuit):
        """Return ``circuit`` with its dense unitaries synthesized exactly, each distinct matrix once per Run object.

        The backends that lower a circuit to a gate basis call this before they
        compile. During a preparation of this Run it passes the Run's synthesis
        cache to ``qiskit_compat.exact_dense_unitaries``. The Run keeps the
        circuit of every synthesis that its preparations made, keyed by the
        content of its matrix, so a later preparation that holds the same
        matrix, even in new gate objects, reuses the circuit. Only the
        matrices missing from the cache are synthesized and counted
        (:meth:`_record_synthesis`). The cache lives until this Run object is
        closed. Reopening the Run starts an empty cache, and a synthesis made
        again after reopening is counted again.

        Before the syntheses start, the largest working allowance and every
        kept allowance of ``dense_synthesis_size``, with the kept allowances
        of the circuits already in the cache, are checked against the
        physical memory. When the cache alone takes them over it, the cache
        is emptied, which changes no circuit since the synthesis is
        deterministic, and the syntheses of every matrix of the circuit are
        checked by themselves. Syntheses that do not fit are refused before
        any of them starts. A backend called directly with this Run, outside
        its preparations, neither reads nor adds to the cache, and only its
        own syntheses are checked.
        """
        from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses, dense_synthesis_size
        from nwqlib.subroutines.qiskit_compat import dense_unitary_census, exact_dense_unitaries
        state = self._state
        preparing = state["synthesis_preparation"] is not None
        cache = state["synthesis_cache"] if preparing else {}
        needed = dense_unitary_census(circuit, cached=True)[0]
        widths = [width for key, width in needed.items() if key not in cache]
        if widths:
            sizes = [dense_synthesis_size(width) for width in widths]
            data = max(size[1] for size in sizes) + sum(size[2] for size in sizes)
            if cache and exceeds(data + state["synthesis_cache_kept"], memory_limit(None)):
                cache.clear()
                state["synthesis_cache_kept"] = 0
                widths = list(needed.values())
            _, work = admit_dense_syntheses(widths, max_work=None, max_bytes=None, operation=(
                f"exact synthesis of {len(widths)} dense unitaries on at most {max(widths)} qubits"))
        result = exact_dense_unitaries(circuit, cache=cache)
        if widths and preparing:
            state["synthesis_cache_kept"] += sum(dense_synthesis_size(width)[2] for width in widths)
            self._record_synthesis(work)
        return result

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
                by name, as integers at least the current values, or None
                to remove an optional limit such as `max_total_shots`.

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
            # ExecutionLimits takes exact integers, including NumPy integer
            # scalars, and refuses unknown names and other values.
            limits = ExecutionLimits(**fields)
            if any(_unbounded(getattr(limits, name)) < _unbounded(getattr(previous, name)) for name in changes):
                raise ValueError("extend_limits can only preserve or increase positive execution limits")
            # A no-op amendment must not consume history or change content identity.
            if all(getattr(limits, name) == getattr(previous, name) for name in changes):
                return self
            state = self._state
            stored = state["data_bytes"] + state["array_bytes"] + state["payload_bytes"] + state["external_bytes"]
            pending = sum(state["pending_output"].values())
            # Records of admitted work can take the counted data beyond the
            # current limit (Run._write), and every new limit must cover them.
            if exceeds(stored + pending, limits.max_data_bytes):
                raise _cumulative_limit_error("max_data_bytes", limits.max_data_bytes, "data bytes already counted",
                                              stored + pending, 0, owner="the proposed")
            amendment = LimitAmendment(sequence=len(state["limit_amendments"]) + 1,
                recorded_at=datetime.now(timezone.utc), old=previous, new=limits,
                circuit_preparations=state["preparations"] - state["host_preparations"],
                circuit_attempts=len(state["events"]) - state["host_invocations"],
                raw_shots=state["shots"], stored_data_bytes=stored, pending_output_bytes=pending)
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
        """The last committed iteration state as a dictionary, or None before any checkpoint."""
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
        """Save the iteration state of an adaptive Method and the RNG position in one transaction.

        Each top-level field is its own row of the run log. changed names the fields
        that differ from the previous checkpoint, and only those are encoded
        and written, so a checkpoint costs its changed data rather than the
        whole state. The caller guarantees that every other field is unchanged.
        None writes every field.

        The checkpoint header, the changed field rows, deletions of removed fields,
        the RNG snapshot and the changed native and method cache rows commit in one
        transaction. Decisions of the iteration code consume RNG draws, so saving
        the RNG position with the decision lets the iteration code of a reopened
        Run continue from the exact draw that followed it. If the write fails, the previous checkpoint
        stays current and the cache dictionaries are restored
        (``_run_archive.write_caches``).

        For the iteration code of an adaptive Method this is commit 1 of the
        module docstring. A preparation charge made while this checkpoint is
        current records its sequence number, and the iteration code passes the
        same number to its submission. After reopening, the iteration code
        therefore finds that work through
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
        """The acquisition that the iteration code started under checkpoint ``sequence``, or None.

        Each checkpoint can name at most one attempt. After reopening, the
        iteration code looks its attempt up here and retrieves it, rather than
        starting a second one.
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
        """The saved chunk of ``attempt``, whether or not the Method has collected it.

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
        holding one recorded batch of outputs or the outputs of one
        trajectory point. Outputs that are saved but that the Method has not
        yet collected are absent. The same object is returned until the next
        collection.
        """
        state = self._state
        chunks, cached = state["chunks"], state["observation_view"]
        if cached is None or cached[0] is not chunks or cached[1] != len(chunks):
            cached = state["observation_view"] = (chunks, len(chunks), ObservationView(chunks=tuple(chunks.values())))
        return cached[2]

    def collect(self, chunk):
        """Mark one completed observation as collected by the Method, once.

        Completion and collection are separate events. The run log gives each
        collected chunk the next positive ordinal, and ``observations`` lists
        chunks in that order. A reopened Run therefore shows the Method its
        observations in the order it collected them, even when they completed in
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
        """Update the index of sampling streams only after the write it derives from succeeds.

        Invalid completed data remain saved as raw data. The error is
        remembered for the next check, rather than raised after an atomic save,
        which would rewrite its completed event as an uncertain acquisition.
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
        """Check new likelihood inputs against this Run's incremental index of sampling streams.

        A backend with ``fixed_seed`` sampling resets its stream to the same seed
        on every call, so two sets of counts from one such stream are the same
        sample, not independent ones. The index marks every stream that a committed
        reservation has used, including uncertain and uncollected attempts, and
        rejects a new setting that would reuse one. Only stream keys are kept,
        bounded by the preparation records and acquisitions already checked. The
        index is not saved. A reopened Run rebuilds it once at its first check of
        counts, and later commits update it. The current batch is checked without
        reserving a stream before its commit.
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
        """Add one already committed attempt to the in-memory counters.

        Called after a reservation commits (``_begin_submission`` and
        ``_submit_host``) and while ``_restore`` reads the run log, so a reopened
        Run counts exactly the attempts whose reservation was saved. The attempt's
        output reservation stays pending until its outcome is saved or it fails
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
        """Atomically reserve a complete submission and all its prepared acquisitions.

        The submission intent, one reserved event per item, any preparation record
        not yet in this Run and the static workflow join commit together with the items'
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
        # The static workflow rows join their attempts in this transaction: one
        # row named by workflow_current, or for a template job the tuple of
        # row keys in the order of the events.
        current = state["workflow_current"]
        keys = () if current is None else current if isinstance(current, tuple) else (current,)
        if keys and len(keys) != len(events):
            raise ValueError("each static item requires exactly one acquisition")
        changed = tuple((key, state["workflow_items"][key].revise(attempt=event.attempt))
                        for key, event in zip(keys, events))
        records.extend(("workflow_item", key, item) for key, item in changed)
        # Persist all reservations before installing counters in live memory.
        # The caller may contact the backend only after this transition succeeds.
        self._write(tuple(records), reserve_output=sum(event.data_bytes_reserved for event in events))
        state["workflow_items"].update(changed)
        state["submissions"][submission.submission_id] = submission
        state["pending_submissions"][submission.submission_id] = None
        for event, receipt, item in zip(events, prepared, submission.items, strict=True):
            self._install_attempt(event)
            state["prepared_artifacts"][receipt.content_id] = receipt
            state["submission_items"][event.attempt] = item
            self._index_counts(receipt=receipt)
        return tuple((state["attempt_indices"][event.attempt], event) for event in events)

    def _assessment_id(self, receipt):
        """Content hash of the forecast's point assessment for this preparation record, or None without one."""
        if self.forecast is None:
            return None
        assessment = self.forecast.assessment_for(self.plan, receipt)
        return None if assessment is None else assessment.content_id

    def _ack_submission(self, identity, locator):
        """Record the provider's job locator once. A different second locator is rejected.

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

    def _finish_attempt(self, index, event, chunk, *, native_simulations=None, collect=False):
        """Save one synchronous acquisition's outcome in one transaction.

        This is commit 6 of the module docstring. The completed event, its
        observation chunk, the completed submission with its locator and every
        array the acquisition saved commit together, and the attempt's output
        reservation is released in the same write. ``collect=True`` also gives
        the chunks their collection ordinals in that transaction, in their
        order, as ``Run.collect`` would right after it (commit 7). The in-memory maps change
        after the commit. The counts index learns the chunk last, so an index
        error cannot undo a saved outcome. Multi-item batches complete through
        ``refresh_submissions``, which saves all newly returned items of one
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
        self._flush_publications(tuple(records), release_output=event.attempt,
                                 collected=tuple(item.content_id for item in chunks) if collect else ())
        self._state["events"][index] = event
        for item in chunks:
            self._state["receipts"][item.acquisition_key] = item
        self._state["by_attempt"][event.attempt] = chunk
        if submission is not None:
            self._state["submissions"][submission.submission_id] = submission
            self._state["pending_submissions"].pop(submission.submission_id, None)
        for item in chunks:
            self._index_counts(chunk=item)
        if collect:
            for item in chunks:
                self._state["chunks"][item.acquisition_key] = item

    def _fail_attempt(self, index, event, error, *, discarded=False):
        """Record an acquisition whose call raised, without claiming more than is known.

        Only a classical routine whose invocation ended with an ordinary ``Exception``
        becomes ``failed``. Local deterministic code then has a confirmed outcome
        and no hidden provider work, and its output reservation is released. Every
        other case, including every quantum call and a ``KeyboardInterrupt``,
        becomes ``uncertain``. The backend may have executed or billed the work
        before the error arrived, so the reserved shots stay counted and the output
        reservation stays pending in case a later refresh retrieves the result.
        Arrays of the outcome that are not yet saved are discarded first.

        ``discarded`` names a local simulator whose synchronous call returned
        its result in process, after which decoding or saving refused it. The
        work ran and no later refresh can retrieve the output. Its submission
        becomes failed with its results consumed, which releases the output
        reservation in the run log, so the release holds after reopening as
        well. The event stays uncertain, the only unsuccessful status of a
        quantum attempt.

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
            synthesis_work=state["synthesis_work"],
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
        contacting the provider still rejects new preparations and
        measurements. Pending remote preparations and submissions are then
        refreshed, which sends each job at most one cancellation request and
        then reads that job. A request does not prove that a job was cancelled,
        and only a status returned by the provider does. All counted work stays
        counted. A backend without a cancellation operation, such as local
        NWQ-Sim, receives no request, and neither does a job already recorded
        as ended, with or without a result. A request that fails is recorded as
        not sent, so calling `cancel` again sends it until the job is recorded
        as ended. It does not keep the other jobs from receiving theirs or keep
        its own job from being read, so the result of a job that had already
        ended is saved. The first such error is raised after all of them.

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
            self._forward_cancellation()
            return self

    def _forward_cancellation(self):
        """Refresh every pending remote preparation and job once, which forwards the cancellation, and then raise the first error.

        One failing request must not keep the others from the provider, because
        each job left running can keep charging. Only an ordinary ``Exception`` is
        held back. An interruption such as ``KeyboardInterrupt`` propagates at once.
        """
        from nwqlib._remote_preparation import refresh_preparation
        first = None
        for identity, pending in tuple(self._state["remote_preparations"].items()):
            if pending.status not in {"completed", "failed", "cancelled"}:
                try:
                    refresh_preparation(identity, run=self)
                except Exception as error:
                    first = first or error
        if callable(getattr(self.backend, "refresh", None)):
            try:
                refresh_submissions(run=self)
            except Exception as error:
                first = first or error
        if first is not None:
            raise first

    def terminate(self, *, reason):
        """Record in the run log the reason that the iteration code of a Method gives for ending early. No job is cancelled.

        The reason and the current cancellation request share one control row.
        A stop before this commit leaves the previous reason in the run log.
        """
        self._write((("control", "current", dict(cancel_requested=self.cancel_requested, termination_reason=reason)),))
        self._state["termination_reason"] = reason

    def resume(self, *, reanalyze=False, rebuild_preparations=False):
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
            rebuild_preparations (bool): Default `False`. `True` asks for new
                counted work: each setting of a `Plan` with fixed settings
                whose local preparation failed or was interrupted after it was
                counted, and which has no preparation record, is prepared once
                more in this call, before anything new is submitted. It keeps
                its realization, bindings and runtime seed. The earlier attempt
                stays counted and each new circuit preparation counts against
                `max_total_circuits`. A compiled template that is already saved
                is reused without compiling it again. No submitted job is sent
                again, and an attempt whose outcome is uncertain keeps its
                identity. The request covers this call only, so a preparation
                that fails again needs another call with `True`, and `wait`
                never rebuilds. With nothing to rebuild, including an adaptive
                Method, whose later settings depend on earlier outcomes, or a
                cancelled Run, it changes nothing and the call continues as
                usual. [Continue an interrupted
                run](../run_archives.md#rebuild-an-interrupted-preparation)
                describes it.

        Returns:
            run (Run): This Run.

        Raises:
            RunFailed: If no Result is available and an uncertain attempt
                cannot be retrieved by any route (stage `"recovery"`).
            RuntimeError: If a setting of a `Plan` with fixed settings has a
                counted preparation without a record and `rebuild_preparations`
                is `False` (`PreparationNotRebuilt`). Nothing further is
                submitted.
            ValueError: If `reanalyze=True` is given for a Run that already
                has its Result, or if a limit refuses a preparation that
                `rebuild_preparations=True` needs. `Run.extend_limits` raises
                the limit.
            Exception: A provider's retrieval or cancellation error, in the
                type the provider raises. A failed read propagates with the
                job's locator saved. After `cancel`, the error of a failed
                cancellation request propagates after that job has been read
                and what the read found has been saved.

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
        # existing submissions, then an explicit rebuild, then the Method's
        # ``execute`` on the same Plan and Run. The rebuild runs here rather
        # than inside ``execute``, because a detached backend's ``execute``
        # returns after its first new submission and a row later in the Plan
        # would outlast this call's request. Changed caches are saved even
        # when ``execute`` raises. A
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
        if type(rebuild_preparations) is not bool:
            raise TypeError("rebuild_preparations must be a boolean")
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
            if self.cancel_requested is not None:
                self._forward_cancellation()
            elif callable(getattr(self.backend, "refresh", None)):
                refresh_submissions(run=self)
            try:
                if rebuild_preparations and self.cancel_requested is None:
                    _rebuild_static_preparations(self)
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
        submission (failure ``_AER_REOPEN_FAILURE``), whose failed status records
        only that the in-process result cannot be retrieved, so its uncertain
        attempt raises here like any other. Any other uncertain attempt, including every
        interrupted classical routine, has no route to its outcome, and this raises
        ``RunFailed`` naming that attempt. The event stays uncertain and counted,
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
                # The attempt's own text is what its call raised. Reopening a
                # synchronous Aer Run replaces the submission's text with its own.
                failure=event.failure or (None if submission is None else submission.failure),
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
        the same job, and an uncertain submission is never resubmitted. After
        `cancel`, the error of a cancellation request that fails propagates
        the same way, after its job has been read.

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
                ended in failure or cancellation, an uncertain attempt cannot
                be recovered, or the provider or local system has confirmed that a
                job ended without a result. A result that arrives later can still
                be collected with `resume`.
            TimeoutError: If the Run is still pending after `timeout`. The
                same Run can be continued later.
            RuntimeError: If `cancel` was called on this Run before it
                completed and no failed or cancelled work is confirmed yet. Its
                pending work is saved in its folder. Also `PreparationNotRebuilt`,
                if a setting of a `Plan` with fixed settings has a counted
                preparation without a record. Nothing further is submitted, and
                `run.resume(rebuild_preparations=True)` prepares it again.

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
        preparation, a failed or cancelled submission with items that have no
        observation, other than a reopened synchronous Aer submission
        (``_AER_REOPEN_FAILURE``), whose failed status records only failed
        retrieval, and a job that ended without a result
        (``SubmissionRecord.terminal_without_result``), whose outcome stays
        uncertain because its output could still arrive.
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
            if submission.terminal_without_result:
                missing = next((item.attempt for item in submission.items if item.attempt not in state["by_attempt"]), None)
                if missing is not None:
                    raise RunFailed(stage="recovery", status="uncertain", locator=submission.locator,
                        failure=submission.failure, directory=self.directory, trace=self.trace,
                        exposure=self.exposure, attempt=missing)

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
        """The reader of a saved observation array's bytes in the run log, or None when the run log has none.

        Returns ``(read_chunks, chunk_bytes)`` for ``ArtifactStore._hydrate``.
        The block size is the one that ``LocalJournal.commit`` saved
        (``_run_journal.PAYLOAD_BLOCK_BYTES``), because ``_hydrate`` rejects a
        block of any other size. The payload is also checked against the data
        limit for one object before it is read.
        """
        from nwqlib._run_journal import PAYLOAD_BLOCK_BYTES

        reference = self._state["artifact_payloads"].get(manifest.content_id)
        journal = self._state["journal"]
        if reference is None or journal is None:
            return None
        self.check_data(manifest.data_bytes)
        return (lambda: journal.payload_chunks(reference)), PAYLOAD_BLOCK_BYTES

    def _reserve_publication(self, data_bytes, provenance):
        """Check one array against the output bytes reserved for its running acquisition before copying or hashing it.

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
        produce identical bytes therefore share one saved payload, while each
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

    def _flush_publications(self, rows, *, release_output=None, collected=()):
        """Commit the queued arrays and the outcome rows, with any collection ordinals, in one transaction.

        On failure the array byte counter and the store's handles return to their
        state before the batch, so no array becomes visible without its
        observation.
        """
        batch = self._state["publication_batch"]
        if batch is None:
            self._write(rows, release_output=release_output, collected=collected)
            return
        records = {(kind, key): (kind, key, value) for kind, key, value in batch["records"]}
        payloads = {reference.content_id: (reference, buffer) for reference, buffer in batch["payloads"]}
        old = self._state["array_bytes"]
        self._state["array_bytes"] += batch["array_bytes"]
        try:
            self._write((*records.values(), *rows), payloads=tuple(payloads.values()), release_output=release_output,
                        collected=collected)
        except BaseException:
            self._state["array_bytes"] = old
            self._abort_publications()
            raise
        self._state["artifact_payloads"].update(batch["payload_refs"])
        self._state["publication_batch"] = None

    @classmethod
    def _restore(cls, plan, *, backend, directory, archive_files, selection, saved_limits, progress=None):
        """Reopen a saved Run at the state from which it resumes, without new preparation or acquisition.

        The run log is read in dependency order by the steps called below, each
        named after what it restores. The header and backend come first, then
        the row sizes, the current limits and amendments, and the RNG. The
        preparation records, preparation charges, remote preparations and payload sizes
        follow, then the checkpoint, submissions and events, observations,
        workflow and control rows, and array manifests. Current limits precede
        larger rows. Saved observations are joined to their attempts without
        repeating the checks that accepted them when they were created.

        Committed cache rows and the files they reference are then registered
        while the Run's exclusive lock is held (``_run_archive.restore_caches``).
        Uncommitted files with the reserved cache prefix can fill the data limit
        after an interruption, so they are deleted before the saved file bytes
        are counted and before the recovery write below. Registration does not
        read a payload whose reading is deferred.

        An interrupted intent may have reached the backend, so its outcome
        becomes uncertain. Reserved events without a known outcome also stay
        counted. Recovery revisions commit together after the saved state is
        restored, in one final write. Every step before that write only reads the run log.
        Failure closes the run log and releases the Run's exclusive lock.

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
        """Check the single header of the run log against the Plan and backend, then build the Run from it.

        The header supplies the run identity, the initial limits and the saved
        forecast and allocation. A different Plan or backend configuration
        raises before any other row is read, because the saved acquisitions
        belong to the original Plan and backend only. ``backend`` None selects local
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
        """Record the saved size of each run-log row in ``data_sizes`` and their total in ``data_bytes``.

        The sizes come from the rows' ``size`` column, without reading their text.
        """
        state = self._state
        for kind, key, size in journal.connection.execute("SELECT kind,key,size FROM records"):
            state["data_sizes"][kind, key] = size
            state["data_bytes"] += size

    def _restore_limits(self, journal):
        """Restore the current limits and their amendment chain before larger rows are read.

        The current ``max_data_bytes`` becomes the run log's limit on each
        array or binary payload read after this step. The rows themselves are
        read whatever their total, which the rows of admitted work can take
        beyond that limit (``Run._write``). The saved amendments populate the
        Run's limit history.
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
        """Restore preparation records, preparation charges, remote preparations and payload sizes.

        Identities and charges come from the saved records. No quantum circuit
        is rebuilt and no host preparation is repeated. A charge whose
        ``prepared_id`` is None is a failed or interrupted preparation (commit
        2 of the module docstring), which stays counted against the limits.
        The synthesis work of each charge, saved with its lowering result,
        joins the trace's total, so a reopened Run reports the same total as
        the live one.
        """
        state = self._state
        for key, fields, _ in journal.rows("prepared"):
            record = PreparedArtifact.model_validate(fields)
            state["prepared_artifacts"][key] = record
            _index_template_record(state, record)
        for key, fields, _ in journal.rows("preparation"):
            charge = PreparationCharge.model_validate(fields)
            state["preparations"] += 1
            state["host_preparations"] += int(charge.execution == "host_kernel")
            state["preparation_charges"][key] = charge
            state["synthesis_work"] += charge.synthesis_work
            if charge.checkpoint_sequence is not None:
                state["checkpoint_preparations"][charge.checkpoint_sequence] = key
            if charge.prepared_id is not None:
                record = state["prepared_artifacts"][charge.prepared_id]
                state["local_prepared_ids"].append(charge.prepared_id)
                state["construction_work"] = known_sum(state["construction_work"], record.construction_work_reserved)
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
        """Restore the checkpoint of the iteration state, one header naming its fields and one row per field."""
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
        of a new request that was never sent. A reserved event becomes uncertain
        when it is a classical routine, has no saved submission, or belongs to an
        uncertain, failed or cancelled submission, because its outcome is
        unknown when the Run reopens. A later ``reconcile`` and refresh can
        still complete the events of an uncertain submission. Each attempt
        re-enters the in-memory counters through ``_install_attempt``. The returned
        rows are written only after every later step has validated.

        Reopening an interrupted synchronous Aer attempt keeps its execution
        status uncertain and its job, circuit and shot counts in the exposure.
        When its output existed only in the interrupted process and no saved
        route to retrieve it remains, the Run releases the output-byte reservation.
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
        submission records failed retrieval. It does not establish a failed
        execution, return counted work or retry the quantum call. New work is
        admitted while::

            D_stored + D_arrays + D_payloads + D_external + sum_pending R_a <= max_data_bytes

        and the rows of admitted work can take the left side beyond the limit
        (``Run._write``). ``R_a`` reserves capacity for an output that may still be saved. Once
        the original synchronous Aer output cannot be recovered, nothing is
        left to save, so releasing ``R_a`` leaves saved data unchanged and keeps
        the event, circuit, job and shot counters, as
        ``_fail_attempt(..., discarded=True)`` does in an open Run.
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

        A quantum observation was checked against its preparation record when it was
        decoded and against its submission's job when it was saved, and
        every observation committed in one transaction with its completed
        attempt (commit 6 of the module docstring), so reopening indexes it
        without repeating those checks. Each row is still validated as an
        ``ObservationChunk`` (its layout and associations, and for counts the
        count domain), and a probability chunk's arrays are not read.
        Collection is a separate ordinal (commit 7), so reopening restores the
        order in which the Method collected observations independently of the
        order of completion, and uncollected observations remain available to
        the resumed Run. A submission stays pending while
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
        """Restore the static workflow rows (commits 1 and 2) and the cancellation or termination control row."""
        state = self._state
        for key, fields, _ in journal.rows("workflow_item"):
            item = WorkflowItem.model_validate(fields)
            state["workflow_items"][key] = item
        for _, control, _ in journal.rows("control"):
            state.update(cancel_requested=control["cancel_requested"], termination_reason=control["termination_reason"])

    def _restore_publications(self, journal):
        """Reopen the manifests of saved observation arrays and count their bytes."""
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
                while an outcome is being saved, or if its counted data
                exceed `max_data_bytes`. Records of work that had already
                started can raise the total above the limit. Raise it with
                `run.extend_limits(max_data_bytes=...)` and save again.
        """
        from nwqlib._run_archive import save
        return save(self, path=path)

    def release_native(self):
        """Release the in-memory Qiskit circuits whose results the Method has all collected, that are not in use and that are saved as QPY.

        For a long Run with a folder, this frees the Qiskit circuits that are
        no longer needed. The preparation records and the order of the
        prepared circuits stay. Inspecting a released circuit reads its saved
        QPY file (Qiskit's binary circuit format) once, and nothing is built,
        transpiled or submitted again.
        Circuits of other backends, unsaved circuits and circuits still in use
        are kept. Aer is the backend whose circuits are released. A circuit that
        the experiments of one template share stays in memory until the last of
        them is released. The counts describe released circuits, not measured
        memory.

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
                # Closing releases execution caches, including the synthesis
                # cache and the template circuits, template QPY bytes and
                # mapped observable that IBM preparation keeps, without
                # resetting exposure, the recorded synthesis work, stored-data
                # charges, receipts or the already attached Result. Saved
                # payload files stay.
                self._state.update(method_context=None, handles={}, native_held=0, template_circuits={}, refresh_cache={},
                    definitions=({}, {}),
                    counts_sources=None, counts_source_failure=None, synthesis_cache={}, synthesis_cache_kept=0,
                    template_bases={}, template_payloads={}, ibm_templates={}, ibm_observable=None,
                    specializations={}, receipt_commits={}, backend_context={}, archive_files=None, progress=None,
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
    file (Qiskit's binary circuit format) when needed, without building or transpiling it again. Method authors
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
    _native_bytes: int = field(repr=False)

    @classmethod
    def _make(cls, record, realization, native, items, setting, bindings, *, native_bytes=0):
        value = object.__new__(cls)
        for name, item in (("record", record), ("realization", realization), ("_native", native),
                           ("_items", items), ("_setting", setting), ("_bindings", bindings),
                           ("_saved_native", None), ("_native_bytes", native_bytes)):
            object.__setattr__(value, name, item)
        return value

    def _restore_native(self):
        """Return the native object, reloading it from the saved QPY after ``Run.release_native``.

        The reload reads the circuit file named in the handle's cache row, and the
        backend rebuilds its native object from that circuit and the saved
        metadata (``restore_native_data``). The saved source vector of a
        directly set preparation, if any, is read only when a compiled
        inspection or a cache copy needs it. Nothing is lowered or transpiled,
        and nothing is charged. The circuit is compared with the Method's memory
        limit before its file is read, and it counts in the Run's ``native_held``
        again. A row bound to a template reuses the circuit that another of its
        rows holds in memory, and otherwise brings it back for the rest of them
        (``_LiveTemplate``). Sharing the object is safe because Aer binds counts
        readouts only, whose restoration leaves the circuit as it is. A copy
        returned for inspection is not counted.
        """
        if self._native is None:
            if self._saved_native is None:
                raise ValueError("prepared handle has no stored native snapshot")
            from functools import partial
            from nwqlib._choice_archive import ArchiveFiles
            backend, path, name, metadata, initial_state, run = self._saved_native
            state = run._state
            key = _template_key(self.record)
            live = state["template_circuits"].get(key)
            new = _restored_bytes(state, self)
            _admit_circuit_memory(run, _CircuitMemory(new, 0, {}, 0, None, 0), "restoring this prepared circuit")
            files = ArchiveFiles(path, None)
            circuit = files.read_circuit(name) if live is None else live.circuit
            native = backend.restore_native_data(self.record, circuit, metadata, run=None,
                                                 initial_state=None if initial_state is None
                                                 else partial(files.read_array, initial_state))
            object.__setattr__(self, "_native", native)
            if key is not None:
                if live is None:
                    live = state["template_circuits"][key] = _LiveTemplate(circuit, new - _TEMPLATE_ROW_BYTES)
                live.owners += 1
            state["native_held"] += new
        return self._native

    def inspect_circuit(self):
        """Return a copy of the prepared Qiskit circuit.

        `QuantumCircuit.copy` gives the copy its own instruction list and its
        own operation objects (checked against the installed Qiskit), so
        changing the copy leaves the circuit that executes unchanged.

        A preparation bound to a shared template (`record.template`) returns
        the template with this experiment's values assigned.

        Returns:
            circuit (QuantumCircuit): The copy.

        Raises:
            ValueError: If this preparation is a classical computation
                (`execution="host_kernel"`) without a circuit, or if its
                circuit was released and reading it back would exceed the
                Method's memory limit.
        """
        self._restore_native()
        if self.record.execution != "quantum_circuit" or self._native.circuit is None:
            raise ValueError("this prepared handle has no live circuit to inspect")
        if self.record.template is not None:
            return _bound_template(self._native.circuit, self.record.template)
        return self._native.circuit.copy()

    def inspect_resources(self, *, transpile_options=None, max_bytes=None):
        """Count the operations of the prepared circuit, without changing or running it.

        Each top-level instruction of the circuit counts once under its
        operation name, measurements, barriers, simulator saves, `Clifford`
        objects and user-defined gates included, and names are reported as
        they are. Gate definitions and control-flow bodies are not expanded.
        With nonempty `transpile_options`, a transpiled copy is counted
        instead, with the simulator saves removed first. A state that Aer
        sets directly in place of a direct state preparation
        (`PreparedArtifact.initial_state_error`) is compiled in that copy as
        the preparation's own circuit, built from the prepared vector, the
        same circuit as without direct setting. The circuit that executes is
        never changed or run.

        Args:
            transpile_options (dict | None): Options for Qiskit's `transpile`.
                `None` or `{}`, the default `None` included, counts the
                prepared circuit as it is.
            max_bytes (int | None): Default `None`, the physical memory of this computer.
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
            ValueError: If this preparation has no circuit, `max_bytes` is
                exceeded, or its circuit was released and reading it back
                would exceed the Method's memory limit.
        """
        self._restore_native()
        if self.record.execution != "quantum_circuit" or self._native.circuit is None:
            raise ValueError("this prepared handle has no live circuit to inspect")
        from nwqlib.backends.inspection import inspect_circuit_resources
        circuit = self._native.circuit
        if self.record.template is not None:
            circuit = _bound_template(circuit, self.record.template)
        state = getattr(self._native, "initial_state", None)
        return inspect_circuit_resources(circuit, native_basis=self.record.native_basis,
            label=f"prepared circuit {self.record.content_id}",
            transpile_options=transpile_options, max_bytes=max_bytes,
            initial_state=None if state is None else (state, self._native.metadata["initial_state_qubits"]))


def _bound_template(circuit, binding):
    """A copy of the template ``circuit`` with the values of ``binding`` (`TemplateBinding`) assigned by parameter name."""
    by_name = {parameter.name: parameter for parameter in circuit.parameters}
    return circuit.assign_parameters({by_name[name]: value for name, value in zip(binding.parameters, binding.values)},
                                     inplace=False)


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
            ValueError: If no circuit has this index, or if the circuit was
                released and reading it back would exceed the Method's memory
                limit. After the default `prepare(plan, settings="first")` only
                the first setting is prepared, and `settings="all"` prepares
                every setting of fixed experiments.
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
        those of the Plan's circuit experiments, in Plan order, with an
        experiment whose batch has range axes repeated once per axis point,
        in increasing axis order. The default
        `settings="first"` prepares only the setting that the Method submits
        first, and an adaptive Method can prepare one experiment several
        times with different parameter values.
        """
        return tuple(handle.realization.experiment for handle in self._circuit_handles())

    def inspect_resources(self, *, index=0, transpile_options=None, max_bytes=None):
        """Count the operations of one prepared circuit, without changing or running it.

        Each top-level instruction of the circuit counts once under its
        operation name, measurements, barriers, simulator saves, `Clifford`
        objects and user-defined gates included, and names are reported as
        they are. Gate definitions and control-flow bodies are not expanded.
        With `transpile_options`, a transpiled copy is counted instead, with
        the simulator saves removed first. A state that Aer sets directly in
        place of a direct state preparation
        (`PreparedArtifact.initial_state_error`) is compiled in that copy as
        the preparation's circuit. The circuit that executes is never changed
        or run.

        Args:
            index (int): Default `0`. Position of the circuit in `circuits`.
            transpile_options (dict | None): Options for Qiskit's `transpile`.
                `None`, the default, counts the prepared circuit as it is.
            max_bytes (int | None): Default `None`, the physical memory of this computer.
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
            ValueError: If no circuit has this index, `max_bytes` is exceeded,
                or the circuit was released and reading it back would exceed
                the Method's memory limit.
        """
        return self._circuit_handle(index).inspect_resources(transpile_options=transpile_options, max_bytes=max_bytes)


def _check_scope(run, point):
    """Refuse a point from another Plan, and any new work after a cancellation request."""
    run._ensure_open()
    if point.plan_id != run.plan.content_id:
        raise ValueError("selected point belongs to another Plan")
    if run.cancel_requested is not None:
        raise ValueError("run cancellation prevents new work")


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


def _admitted_readout(admission, experiment, construction):
    """Read the widths and resolve the observation of a checked Program.

    ``admission`` has run ``admitted()``. The root binding context, one
    width reading per quantum and classical bit register and the observation
    are counted in its work count, as preparation counts them.
    Returns ``(qwidths, cwidths, setting, observation)``.
    """
    program = construction.program
    context = admission.expressions(admission.binding_map(program.bindings))
    qwidths = [(reg.name, admission.integer(reg.width, context, "register width")) for reg in program.registers]
    cwidths = [(reg.name, admission.integer(reg.width, context, "classical width"))
               for reg in program.classical if reg.dtype == "bits"]
    setting, observation = experiment._resolved_observation(admission, construction_id=construction.content_id)
    return qwidths, cwidths, setting, observation


def _selected_readout(plan, point, experiment, construction):
    """Check a point's Program and resolve its register widths, observation and the header its chunks will have.

    Returns ``(admission, qwidths, cwidths, setting, observation, header)``.
    ``header`` holds the context of the point's chunks that does not depend on
    lowering (experiment, setting, bindings, Plan and realization identities and
    the logical layouts of the resolved registers), whose bytes the metadata
    formulas of ``_admit_readout`` and ``prepare_static`` count before native work.
    """
    admission = _Admission(construction.program)
    admission.admitted().require_ready()
    qwidths, cwidths, setting, observation = _admitted_readout(admission, experiment, construction)
    header = dict(experiment=point.experiment, setting=setting, bindings=construction.program.bindings,
                  plan_id=plan.content_id, realization_id=point.content_id,
                  quantum_layout=_register_layout(qwidths), classical_layout=_register_layout(cwidths))
    return admission, qwidths, cwidths, setting, observation, header


def prepare_experiment(point, *, run, runtime=None, held=None):
    """Prepare one point of the Plan, saving its random draw for the backend before native work.

    The checks run in order of increasing cost, so an unsupported request fails before
    any circuit exists. The checks are Program readiness, simulator memory,
    readout shape and cardinality, target readout support, the direct state
    preparations of the circuit, amplitude materialization size and the
    cumulative circuit limit, in that order, except that a trajectory's target
    support is checked before its readout shape and storage reservation, so a
    backend that cannot execute it refuses with the name of the missing
    capability. The chosen IR (the Program) of a trajectory must be one
    coherent evolution along its executed prefix (``admit_pure_trajectory`` with
    ``_selected_body_issues``), checked before its reductions. A trajectory
    also checks its reductions' registered shape and execution, their
    registered classical work against the finite allowance that the Method's
    optional ``reduction_allowance`` hook returns, after the hook's own check
    of its private reduction workspace (``_method_reduction_allowance``),
    and the saved simulator states of its amplitude and reduction points with
    the largest registered reduction workspace (``_admit_trajectory_work``). Then,
    unless the caller supplies ``runtime``, the runtime seed is drawn, and the
    preparation charge commits with the RNG position. Only after that commit does
    lowering start, reusing the Run's definition cache, and the backend prepares
    the native circuit. A remote-compiling backend continues in
    ``_remote_preparation``. A local one saves its preparation record through
    ``_finish_native_preparation``. The charge is commit 2 of the module
    docstring. When the Run's current workflow key names a static point
    without a workflow row, the row, with the seed just drawn, commits with
    the charge.

    ``held`` is the point's ``(Experiment, SelectedConstruction)`` pair
    when the caller already holds it (``prepare_static`` holds its first
    experiment's), so the point is not resolved again. It must name the
    point's experiment and effective bindings.
    """
    construction, setting, observation, items, selected_length, admission, counted, native_bytes = _admitted_point(
        point, run=run, held=held)
    plan, host = run.plan, observation.kind == "host_scalars"
    run.check_capacity(preparations=int(not host))
    # A seed drawn here is kept only with the charge that saves it.
    start = None if runtime is not None else run.rng.backend.bit_generator.state
    runtime = runtime or RuntimeOptions(seed=run.rng.next_seed())
    identity = str(uuid4())
    charge = PreparationCharge(execution="host_kernel" if host else "quantum_circuit", preparation_id=identity,
                               checkpoint_sequence=run.checkpoint_sequence or None)
    state = run._state
    records = [("preparation", identity, charge), ("rng", "current", run.rng.snapshot())]
    current = state["workflow_current"]
    if current is None:
        item = None
    elif current in state["workflow_items"]:
        item = state["workflow_items"][current].revise(preparation_id=identity)
    else:
        # A static point's first preparation creates its workflow row here,
        # with the seed just drawn, in the charge's transaction.
        item = WorkflowItem(realization=point, runtime=runtime, preparation_id=identity)
    if item is not None:
        records.append(("workflow_item", current, item))
    # Save the preparation charge and consumed RNG position before execution.
    # An interruption therefore cannot silently reuse or skip a random draw.
    try:
        run._write(tuple(records))
    except BaseException:
        # A refused or failed charge saves nothing, so the backend stream
        # returns to the position that the run log still holds.
        if start is not None:
            run.rng.backend.bit_generator.state = start
        raise
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
    # Lowering and the backend count the exact dense syntheses that they make
    # for this preparation (Run._record_synthesis), and the total is saved
    # with its charge once its lowering result is saved.
    state["synthesis_preparation"], state["synthesis_pending"] = identity, 0
    # Copies that constructors report join the receipt below, and their
    # commits run only once its record is saved (_finish_native_preparation).
    commits = []
    try:
        logical = _lower_qiskit(construction, blocks=blocks, definition_cache=state["definitions"],
            specialization_cache=(state["specializations"], specializations), method_context=run._method_context,
            receipt_commits=commits, record_synthesis=run._record_synthesis, admission=admission,
            counted=counted[None])
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
                            method_context=run._method_context, receipt_commits=commits,
                            record_synthesis=run._record_synthesis, admission=admission,
                            counted=counted[definition])
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
        native = callback(owned, observation=observation, runtime=runtime, position=position,
            source_definitions=tuple(state["definitions"][0].values()) + tuple(state["definitions"][1].values())
                + tuple(state["specializations"].values()) + tuple(specializations.values()), run=run, snapshot=snapshot,
            **({"views": views} if views else {}))
    except BaseException:
        _abandon_copies(commits)
        raise
    finally:
        state["synthesis_preparation"] = None
        synthesis, state["synthesis_pending"] = state["synthesis_pending"], 0
    charge = state["preparation_charges"][identity]
    fields = dict(plan_id=plan.content_id, realization_id=point.content_id, construction_id=construction.content_id,
        realization=point, runtime=runtime, snapshot=snapshot, observation=observation,
        quantum_layout=tuple(RegisterMap(name=name, bits=bits) for name, bits in logical.quantum_layout),
        classical_layout=tuple(RegisterMap(name=reg.name, bits=tuple(owned.find_bit(bit).index for bit in reg)) for reg in owned.cregs),
        logical=LogicalPreparationReceipt(dynamic_visits=logical.dynamic_visits + sum(item.dynamic_visits for item in tails),
            reserved_work=known_sum(logical.construction_work, *(item.construction_work for item in tails)),
            defined_selections=tuple(dict.fromkeys(logical.defined_selections
                                                   + tuple(s for item in tails for s in item.defined_selections)))))
    if observation.kind == "trajectory":
        fields.update(body_length=body_length, boundaries=position)
    try:
        if callable(getattr(run.backend, "prepare_local", None)):
            from nwqlib._remote_preparation import begin_preparation
            # A remote preparation saves its record when a later refresh
            # completes it. Until then its copy commits and its
            # specializations stay with its charge, so another preparation
            # cannot reuse gates whose copies no saved record counts.
            state["receipt_commits"][identity] = commits, specializations
            return begin_preparation(native, run=run, reservation=charge, fields=fields, items=items,
                                     setting=setting, bindings=construction.program.bindings, synthesis=synthesis)
        _publish_specializations(state, specializations)
        return _finish_native_preparation(native, run=run, reservation=charge, fields=fields, items=items,
                                         setting=setting, bindings=construction.program.bindings, commits=commits,
                                         synthesis=synthesis, native_bytes=native_bytes)
    except BaseException:
        if identity not in state["remote_preparations"]:
            _withdraw_preparation(state, identity, specializations, commits)
        raise


def _publish_specializations(state, specializations):
    """Make ``specializations`` the Run's frontier, which the next preparation can reuse.

    They replace the frontier row at the next cache commit. The committed
    ones are kept until then, so a failed commit can restore them.
    """
    state.setdefault("cache_previous_specializations", state["specializations"])
    state["specializations"] = specializations
    state["cache_frontier_dirty"] = True


def _withdraw_preparation(state, identity, specializations, copies=()):
    """Undo what the preparation of charge ``identity`` built or published before its record, once it ends without one.

    No record was saved and none can be refreshed. The withdrawals of its
    copies run, those waiting with its charge or else ``copies``, so each
    constructor retires what it built for it, and its commits are
    dropped. Its specializations, while still the published ones, give way
    to the committed ones, as after a failed cache commit
    (``_run_archive.finish_caches``). Their gates hold copies that no
    receipt counts, so a retry of the same point builds them again.
    """
    waiting = state["receipt_commits"].pop(identity, None)
    _abandon_copies(copies if waiting is None else waiting[0])
    if state["specializations"] is specializations and "cache_previous_specializations" in state:
        state["specializations"] = state.pop("cache_previous_specializations")


def _abandon_copies(copies):
    """Run the withdraw callback of each ``(commit, withdraw)`` pair of a preparation that saved no record."""
    for _, withdraw in copies:
        withdraw()


def _admitted_point(point, *, run, held=None, admit_memory=True):
    """Run the checks of ``prepare_experiment`` that come before its charge, and return what preparation needs.

    Returns ``(construction, setting, observation, items, selected_length,
    admission, counted, native_bytes)``, with ``selected_length`` the body
    length of a trajectory's selected IR and None otherwise, ``admission``
    the ready ``_Admission`` of the construction's Program, ``counted`` the
    counting passes of ``_admit_direct_preparations`` (empty for a
    classical routine), both of which lowering reuses, and ``native_bytes``
    the memory estimate of the native circuit that its handle will hold. The
    last check compares that circuit's memory estimate and what the Run
    already holds with the Method's memory limit (``_admit_circuit_memory``),
    unless ``admit_memory`` is False, as for a template's rows, which
    ``_admit_template_rows`` compares together. Nothing is charged, drawn or
    lowered.
    """
    _check_scope(run, point)
    plan = run.plan
    selected_length, counted, native_bytes = None, {}, 0
    if held is None:
        experiment, construction = point._selected_construction(plan)
    else:
        experiment, construction = held
        if experiment.name != point.experiment or construction.program.bindings != point.bindings:
            raise ValueError("the held selection belongs to another point of the Plan")
    admission, qwidths, cwidths, setting, observation, header = _selected_readout(plan, point, experiment,
                                                                                 construction)
    width = sum(item for _, item in qwidths)
    cwidth = sum(item for _, item in cwidths)
    host = observation.kind == "host_scalars"
    # The simulator memory of a readout other than a trajectory follows from
    # the width and the readout kind alone, so it is checked before the
    # readout is sized or anything is charged. A trajectory is checked with
    # its points (_admit_trajectory_work).
    admit = getattr(run.backend, "admit_simulation_memory", None)
    memory_mb = run.limits.simulator_memory()
    if not host and observation.kind != "trajectory" and admit is not None and memory_mb is not None:
        try:
            admit(observation, width=width, memory_mb=memory_mb, run=run)
        except ValueError as error:
            state = f"{16 << width} bytes" if width < 64 else f"16*2**{width} bytes"
            remedy = _limit_remedy('simulator_memory_mb')
            if plan.method.descriptor.method == "qhd" and plan.method.encoding == "one_hot":
                k = plan.method.num_grid_points
                if plan.method.boundary == "periodic" and k & (k - 1) == 0:
                    binary_width = len(plan.problem.variables) * (k.bit_length() - 1)
                    remedy += (f' A new QHD Plan with encoding="binary" uses {binary_width} qubits for this periodic '
                               'grid.')
                else:
                    remedy += (' QHD encoding="binary" requires a periodic grid with a power-of-two '
                               'num_grid_points. Switching changes the grid or boundary of this computation.')
            raise ValueError(f"the selected circuit has {width} qubits, whose complex128 state alone takes {state}, "
                             f"and its simulation exceeds simulator_memory_mb={memory_mb} ({memory_mb << 20} bytes) "
                             f"before native preparation ({error}). {remedy}") from None
    # A backend that cannot execute a trajectory refuses with its capability
    # message before the trajectory's storage reservation is checked.
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
            selected_length, body_issue, view_issues = _selected_body_issues(construction, admission, observation,
                                                                             dict(qwidths))
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
        counted, trees = _admit_direct_preparations(run, construction, admission, observation)
    if observation.amplitudes is not None:
        observation.amplitudes.admit_materialization(max_bytes=run.limits.max_data_bytes,
                                                     max_bytes_source=_data_limit_clause(run))
    if not host:
        # Limits that the adapter can decide from the readout and its own
        # configuration refuse here, before any charge or random draw, so
        # Run.extend_limits leads past them.
        admit_preparation = getattr(run.backend, "admit_preparation", None)
        if callable(admit_preparation):
            admit_preparation(observation, classical_widths=tuple(item for _, item in cwidths if item), run=run)
        memory = _circuit_memory(run, counted, trees)
        if admit_memory:
            _admit_circuit_memory(run, memory, "preparing this circuit")
        native_bytes = memory.kept
    return construction, setting, observation, items, selected_length, admission, counted, native_bytes


def _admit_direct_preparations(run, construction, admission, observation):
    """Check the direct state preparation trees of one point's circuit against the physical memory, before its charge.

    The circuit and, for a trajectory, the tail and inverse of each readout
    view are counted by ``lowering._count_definition``, which lowering then
    reuses. Each occurrence of a call of a block that declares a direct
    state preparation counts one tree (``direct.direct_tree_bytes``), with
    Repeat counts multiplied, and a Plan's blocks that this circuit does
    not call count nothing. The tree of a control or adjoint is built by
    lowering, so every such call in the circuit and in each lowered view
    counts. The tree of an untransformed preparation is built when the
    backend translates that occurrence. On the noiseless Aer statevector
    target and the local NWQ-Sim runner an exact Pauli readout or a
    trajectory translates only the instructions before its position or last
    boundary (``connection.executed_stop``), so only those occurrences
    count. A trajectory on Aer applies each view as often as
    ``qiskit_aer._view_applications`` says, and NWQ-Sim translates each
    view's tail once and derives its inverse from that translation. Every
    other readout and backend translates the whole circuit. When the circuit's first instruction is an untransformed
    preparation that Aer sets as the whole initial state for this readout,
    which ``qiskit_aer._direct_initial_state_bytes`` decides as
    ``AerBackend.prepare`` does, that one occurrence counts the
    ``48 * 2**n`` bytes of the set state on all n qubits of the circuit
    instead of its tree. Returns ``({definition: count}, need)``, with None for
    the circuit itself in the mapping and ``need`` the bytes of the trees.
    """
    from nwqlib.backends.connection import AerBackend, _run_state_threads, executed_stop
    from nwqlib.backends.nwqsim import NWQSimBackend
    from nwqlib.blocks.lowering import _count_definition
    selected = {record.content_id for record in construction.selections}
    blocks = tuple(block for block in run.plan.blocks if block.record.content_id in selected)
    counted = {}
    root = counted[None] = _count_definition(construction, blocks=blocks, admission=admission)
    if observation.kind == "trajectory":
        for scheduled in observation.positions:
            for definition in () if scheduled.view is None else (scheduled.view.tail, scheduled.view.inverse):
                if definition not in counted:
                    counted[definition] = _count_definition(construction, blocks=blocks, admission=admission,
                                                            definition=definition)
    # Lowering builds the trees of controls and adjoints, in the circuit and
    # in each view definition, once each.
    need = sum(count.eager_bytes for count in counted.values())
    aer_backend = isinstance(run.backend, AerBackend)
    truncates = (aer_backend and run.backend.noise_model_id is None) or isinstance(run.backend, NWQSimBackend)
    if truncates and observation.kind in ("pauli_expectation", "trajectory"):
        if observation.kind == "pauli_expectation":
            stop = executed_stop(root.instructions, position=(root.instructions if observation.position is None
                                                              else observation.position))
            applications = {}
        else:
            boundaries = observation.boundaries(root.instructions)
            stop = executed_stop(root.instructions, last=boundaries[-1])
            if aer_backend:
                from nwqlib.backends import qiskit_aer as aer
                applications = aer._view_applications(aer._trajectory_saves(observation, boundaries)[0])
            else:
                applications = {point.view.tail: 1 for point in observation.positions if point.view is not None}
        need += root.lazy_bytes_before(stop)
        need += sum(number * counted[definition].lazy_bytes for definition, number in applications.items())
    else:
        need += root.lazy_bytes
    opening = root.opening_direct
    if opening is not None and aer_backend:
        from nwqlib.backends import qiskit_aer as aer
        from nwqlib.subroutines.state_preparation.direct import direct_tree_bytes
        # An explicit position 0 observes before the first instruction, and
        # the end shorthand observes after it.
        points = observation.positions if observation.kind == "trajectory" else (observation,)
        injected = aer._direct_initial_state_bytes(
            observation, noiseless=run.backend.noise_model_id is None,
            observes=any(point.position is None or point.position > 0 for point in points),
            width=sum(root.widths.values()), memory_mb=run.limits.simulator_memory() or 0,
            state_threads=_run_state_threads(run))
        if injected is not None:
            # The opening call lies in every prefix that observes after it.
            need += injected - direct_tree_bytes(opening)
    try:
        check_memory(need, None, "building the direct state preparations of this circuit")
    except ValueError as error:
        raise ValueError(f"{error}. Each direct preparation of a vector builds a circuit of about 1 KiB per "
                         "amplitude, once for every time the circuit applies it, and this point was refused before "
                         "native preparation. Use a compact state preparation, such as a product state or a "
                         "supplied circuit, or a computer with more memory") from None
    return counted, need


class _CircuitMemory(NamedTuple):
    """The memory estimates of a circuit, or of several circuits together, for ``_admit_circuit_memory``.

    ``kept`` stays with the prepared handles until ``Run.release_native`` or
    ``Run.close``. ``temporary`` is the instruction list that lowering holds
    until the backend has translated it, with the new frontier of definitions
    with parameters, which replaces the previous frontier. ``static`` holds the
    new static definitions by content hash, each counted once. ``calls`` and
    ``definitions`` count what has no estimate, with ``first_call`` the block
    of the first such call. ``frontier`` is the size of the definitions with
    parameters that one circuit's calls use, summed over the calls because
    arguments equal at this point can differ at another. ``logical`` is the
    instruction list alone. Only the whole-Plan check reads these two
    (``_static_points_workload``).
    """

    kept: int
    temporary: int
    static: dict
    calls: int
    first_call: str | None
    definitions: int
    frontier: int = 0
    logical: int = 0


def _circuit_memory(run, counted, trees):
    """Return the ``_CircuitMemory`` of one circuit from its counting passes and direct preparation ``trees``.

    Definitions are found by the keys of lowering's own caches
    (``lowering._definition_closure``): a static one is new when the Run has not
    built it, and one with parameters when the current frontier does not hold
    it, the only frontier that lowering reuses.
    """
    from nwqlib.blocks.lowering import INSTRUCTION_BYTES, _definition_closure
    from nwqlib.blocks.selection import direct_preparation
    state = run._state
    gates, frontier = state["definitions"][0], state["specializations"]
    logical = temporary = INSTRUCTION_BYTES * sum(count.instructions for count in counted.values())
    static, definitions = {}, 0
    for key, block in _definition_closure(pair for count in counted.values() for pair in count.selected.values()).items():
        if key in frontier if key[1] else key[0] in gates:
            continue
        # The tree bytes of a direct preparation are its estimate (``trees``).
        definitions += block._definition_bytes is None and direct_preparation(block) is None
        if key[1]:
            temporary += block._definition_bytes or 0
        else:
            static[key[0]] = block
    calls = sum(count.native_unknown for count in counted.values())
    first = next((count.first_unknown for count in counted.values() if count.first_unknown is not None), None)
    frontier_bytes = sum(block._definition_bytes or 0 for count in counted.values()
                         for block, arguments in count.selected.values() if arguments)
    return _CircuitMemory(sum(count.native_bytes for count in counted.values()) + trees, temporary, static,
                          calls, first, definitions, frontier_bytes, logical)


def _held_bytes(run):
    """Return the memory estimate of the circuits and gate definitions that the Run holds now.

    Static definitions and frontiers are read from the live caches, whose
    owners already commit, roll back, close and restore them, so this cannot
    drift from them. Their sizes come from everything the bound blocks reach
    (``lowering._definition_closure``), because lowering keeps a transform's base
    whether or not the Plan also binds that base by name. A frontier counts while
    it is the current one, the committed one that a failed cache commit would
    restore, or the one of a remote preparation that waits for its record. A
    frontier held in two of those places counts twice, at most one frontier too
    many, and a replaced frontier counts no more. Native circuits are read from
    ``native_held``, because scanning the handles at every preparation would grow
    quadratically. Other Runs and objects of this process are not included.

    A reopened Run on a backend other than Aer starts ``native_held`` at zero, and
    the handles restored from their records add nothing to it, so this total and
    the number a refusal prints are low by what was prepared and not yet submitted
    when the Run stopped. Each of those circuits was compared with the same limit
    when it was prepared, unless ``max_bytes`` is None and the Run reopens on a
    computer with less memory. The whole-Plan comparison of the reopened Run
    (``_admit_static_workload``) covers the points not yet prepared. Charging the
    position law here would be wrong: such a handle holds a backend's request text
    or provider reference, or a decoded circuit for which the law is not calibrated,
    and a false refusal would strand a saved, possibly paid Run, which keeps the
    limit of its Method.

    The controlled gates that lowering keeps per base (group 1 of ``definitions``)
    are not read, because their key is the base's hash and the base's size does not
    describe them. A forward controlled gate is the object that group 0 holds under
    that block's own hash, so its law counts it once. One that only a controlled
    adjoint reaches has no size of its own, so only that adjoint's law can cover it.
    """
    from nwqlib.blocks.lowering import _definition_closure
    from nwqlib.blocks.selection import SelectedBlock
    state = run._state
    # ponytail: restored records of backends other than Aer are not counted; add them only once a
    # memory law for those backends is calibrated or an over-admission is measured.
    # A bound kernel runs on the host and enters no gate cache.
    closure = _definition_closure((block, ()) for block in run.plan.blocks if isinstance(block, SelectedBlock))
    sizes = {identity: block._definition_bytes or 0 for (identity, _), block in closure.items()}
    frontiers = (state["specializations"], state.get("cache_previous_specializations", ()),
                 *(frontier for _, frontier in state["receipt_commits"].values()))
    return (state["native_held"] + sum(sizes.get(key, 0) for key in state["definitions"][0])
            + sum(sizes.get(key[0], 0) for frontier in frontiers for key in frontier))


def _admit_circuit_memory(run, memory, stage, *, whole=False):
    """Refuse the circuits of ``memory`` when they and what the Run holds exceed the Method's memory limit.

    The limit is the Method's ``max_bytes``, or the physical memory when that is
    None. The peak is about what stays from earlier stages plus what this stage
    adds, so the compared size is

        held + kept + temporary + new static definitions

    It is a coarse estimate of the parts that have one, not a bound on process
    memory, and its target is the order of the measured increase, usually within
    a factor of two. A call without an estimate does not switch the comparison
    off, so a small uncovered gate leaves a large known block protected. The
    refusal says how much it left out. Nothing is built and no scientific choice
    changes. The check runs before the charge of a point, or before the first
    new work of a static Run, so a refusal counts nothing.

    The native positions behind the estimate are counted and calibrated on Aer.
    Other backends have no calibration of their own and use the same count, and
    the refusal says so. ``Run.release_native`` frees circuits on Aer only, so a
    refusal of one circuit offers it for Aer. On another backend it keeps every
    handle, as "no durable QPY snapshot", and frees nothing. A refusal of several
    circuits at once (``whole``) does not offer it, because it comes before they
    run, so no circuit of theirs has results to release, and a larger limit is what
    gets past it.
    """
    held, limit = _held_bytes(run), getattr(run.plan.method, "max_bytes", None)
    new = memory.kept + memory.temporary + sum(block._definition_bytes or 0 for block in memory.static.values())
    try:
        check_memory(held + new, limit, stage)
    except ValueError as error:
        left_out = ""
        if memory.calls or memory.definitions:
            left_out = (f"Not counted, for lack of an estimate: calls {memory.calls}"
                        + (f" (first of block {memory.first_call!r})" if memory.calls else "")
                        + f", gate definitions {memory.definitions}. ")
        computer = "use a computer with more memory, " if limit is None else ""
        raise ResourceLimitError(f"{error}. This is a coarse estimate of the parts that have one: {held} bytes "
                                 f"already held by this Run and {new} bytes new. The estimate counts each circuit's "
                                 "top-level instructions and is calibrated on Aer, and other backends, which have no "
                                 f"calibration of their own, use the same count. {left_out}Reduce the size or accuracy "
                                 f"of the problem, {computer}or plan again with a larger max_bytes in the Method, "
                                 "because a Run keeps the limit of its Method. "
                                 + ("" if whole else "On Aer, in a Run with a folder, Run.release_native frees "
                                                     "circuits whose results have been collected. ")
                                 + "Nothing was built or counted for this request") from None


def _finish_native_preparation(preparation, *, run, reservation, fields, items, setting, bindings, remote=None,
                               commits=None, synthesis=0, native_bytes=0):
    """Save one completed preparation, its payload and its original charge together.

    This is commit 3 of the module docstring. The charge row gains the content
    hash of the preparation record in the same transaction, so a reopened Run can tell a finished
    preparation from an interrupted one (a charge without ``prepared_id``), and
    adds ``synthesis``, the exact dense synthesis work of a local preparation
    (``Run._record_synthesis``). For a remote compilation, whose charge saved
    that work when it began (``begin_preparation``), the pending record
    becomes ``completed`` in the same transaction.
    """
    observation = fields["observation"]
    if observation.kind != "counts" and observation.population == "unconditional" and preparation.population != "unconditional":
        raise ValueError("selected unconditional readout has a conditioned or unknown population")
    payload = None if preparation.payload is None else PayloadRef(format=preparation.payload_format,
                                                               bytes=len(preparation.payload), key=str(uuid4()))
    record = PreparedArtifact(**fields, **_native_fields(preparation, payload),
        backend_configuration_id=run.backend.content_id, preparation_time=_now(),
        construction_work_reserved=fields["logical"].reserved_work)
    handle = PreparedHandle._make(record, fields["realization"], preparation.native, items, setting, bindings,
                                  native_bytes=native_bytes)
    charge = reservation.revise(prepared_id=record.content_id, synthesis_work=reservation.synthesis_work + synthesis)
    records = [("preparation", reservation.preparation_id, charge),
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
    if commits is None:
        # A completed remote preparation publishes its specializations with
        # its record, and a failed commit restores the earlier ones.
        commits, published = run._state["receipt_commits"].get(reservation.preparation_id, ((), None))
        if published is not None:
            _publish_specializations(run._state, published)
    write_caches(run, tuple(records), payloads=payloads)
    run._state["native_held"] += native_bytes
    run._state["receipt_commits"].pop(reservation.preparation_id, None)
    if item is not None:
        run._state["workflow_items"][current] = item
    run._state["prepared_artifacts"][record.content_id] = record
    run._state["local_prepared_ids"].append(record.content_id)
    run._state["preparation_charges"][reservation.preparation_id] = charge
    run._state["synthesis_work"] += synthesis
    run._state["construction_work"] = known_sum(run._state["construction_work"], record.construction_work_reserved)
    for commit, _ in commits:
        commit()
    if remote is not None:
        run._state["remote_preparations"][remote.preparation_id] = remote
    run.progress("prepare", len(run._state["local_prepared_ids"]), None)
    return handle


# The preparation-record fields that describe the native circuit rather than
# the experiment. The records of one compiled template share them.
_NATIVE_FIELDS = ("snapshot", "target", "compiler", "native_quantum_layout", "native_classical_layout",
                  "logical_to_native", "native_basis", "environment", "payload", "native_operations", "population",
                  "transformation", "counts_sampling", "provider_options_json", "probability_window_exclusions",
                  "statevector_roundoff", "initial_state_error", "native_depth")


def _native_fields(preparation, payload):
    """The preparation-record fields of a backend's ``NativePreparation`` other than ``snapshot``, with ``payload`` saved for it."""
    return dict(target=preparation.target, compiler=preparation.compiler,
        native_quantum_layout=preparation.quantum_layout, native_classical_layout=preparation.classical_layout,
        logical_to_native=preparation.logical_to_native, native_basis=preparation.native_basis,
        environment=preparation.environment, payload=payload, native_operations=preparation.operations,
        population=preparation.population, transformation=preparation.transformation,
        counts_sampling=preparation.counts_sampling, provider_options_json=preparation.provider_options_json,
        probability_window_exclusions=preparation.probability_window_exclusions,
        statevector_roundoff=preparation.statevector_roundoff, initial_state_error=preparation.initial_state_error,
        native_depth=preparation.native_depth)


def _groups_points(plan, run):
    """Whether the remaining points of a static Plan are collected for the Method to group (``_row_templates``).

    They are on a backend that binds parameterized circuits, for a Method
    that can group the Plan's points (``Method.has_row_templates``) and, on
    a backend that binds only some readout kinds (``template_readouts``),
    when an experiment's batch declares one of them. Otherwise each point is
    generated, prepared and run in turn.
    """
    if not getattr(run.backend, "supports_template_pubs", False) or not plan.method.has_row_templates(plan):
        return False
    kinds = getattr(run.backend, "template_readouts", None)
    return kinds is None or any(experiment.batch is not None
                                and plan._program_nodes[experiment.batch].observation_kind in kinds
                                for experiment in plan.experiments)


def _row_templates(plan, run, points):
    """The Method's templates with parameters of the static points ``points`` (`StaticPoint`), or None when there are none to bind.

    Only a backend that binds parameterized circuits (``supports_template_pubs``)
    asks the Method (``Method.row_templates``), so planning, estimates,
    archive reads and other backends do no template work. A backend that
    binds only some readout kinds (``template_readouts``) asks about the
    points whose batch declares one of them. A template without parameters
    has nothing to bind, so its points, like the points the Method is not asked
    about, are prepared on their own. Whether the backend can translate a template
    is asked when ``prepare_template_rows`` has its circuit, because reading the
    circuit of a ``_DeferredRowTemplate`` builds it.
    """
    if not points or not getattr(run.backend, "supports_template_pubs", False):
        return None
    kinds = getattr(run.backend, "template_readouts", None)
    if kinds is not None:
        points = tuple(point for point in points if point.experiment.batch is not None
                       and plan._program_nodes[point.experiment.batch].observation_kind in kinds)
        if not points:
            return None
    templates = run.plan.method.row_templates(plan, points)
    if templates is None:
        return None
    if sorted(key for template in templates for key, _ in template.rows) != sorted(point.key for point in points):
        raise ValueError("Method.row_templates must place each requested point in exactly one template")
    return tuple(template for template in templates if template.parameters) or None


def prepare_template_rows(plan, *, run, points, templates, rebuild=frozenset()):
    """Prepare the points of ``templates`` as bindings of compiled templates and return ``{key: handle}`` in Plan order.

    Every point of ``points``, the static points held together, gets its
    workflow row first. The points that no template holds keep that row and
    are prepared on their own later, with its seed.
    Each template is compiled at most once per Run: a point whose
    template a preparation record of this Run already holds binds that
    compiled circuit, or a circuit folded from it, restored from its saved
    QPY after reopening, without lowering or compiling (``_bind_template``).
    A point that is already prepared returns its handle, and one whose
    preparation was charged without a record raises ``PreparationNotRebuilt``,
    unless its key is in ``rebuild`` (``_rebuild_static_preparations``). Such a
    point is prepared like a point without a charge. It binds the template that a
    saved record holds, and compiles only when no record holds it.

    After the comparison with the memory limit (``_admit_template_rows``), a new
    template's circuit is read once, before that template's charge, and it is
    released before the next template's circuit is read. A compiled template's
    circuit is not read at all. The rows of a template that the backend cannot
    translate with its angles as parameters (``translates_template``) need each
    point's values, so they stay for their own preparation with their seeds.
    """
    state = run._state
    # The missing workflow rows, and their seeds, are created in the order of
    # the points (Plan order), in one transaction, so each point has the seed
    # that preparing the points one by one would draw.
    items = _static_items(plan, run, points)
    _admit_template_rows(plan, run, templates, items, rebuild)
    translates = getattr(run.backend, "translates_template", None)
    handles = {}
    for template in templates:
        circuit = None
        missing = []
        for name, values in template.rows:
            item = items[name]
            if item.prepared_id is not None:
                handles[name] = (state["handles"].get(item.prepared_id)
                                 or restore_prepared(item.prepared_id, run=run, for_submission=True))
            elif item.preparation_id is not None and name not in rebuild:
                raise _not_rebuilt(name)
            else:
                missing.append((name, item, values))
        if not missing:
            continue
        if (run.backend.content_id, template.key) not in state["template_sources"]:
            circuit = template.circuit
            if translates is not None and not translates(circuit):
                continue
        handles.update(_bind_template(plan, run, template, circuit, missing))
    return {name: handles[name] for name in items if name in handles}


@dataclass
class _LiveTemplate:
    """A compiled template circuit in memory, which the template's bound rows share (Aer's ``bind_template``).

    ``bytes`` is the circuit's estimate and ``owners`` counts the rows that hold
    the circuit now, so the estimate stays in ``native_held`` until the last of
    them is released, whichever row compiled it. Only ``Run._state`` holds these,
    and nothing is saved: after reopening, the estimate comes from the compiling
    row's saved ``native_bytes`` (``_restored_bytes``).
    """

    circuit: object
    bytes: int
    owners: int = 0


def _template_key(record):
    """Return the ``state["template_circuits"]`` key of the template circuit that ``record`` shares, or None.

    A record of IBM's folded correction classes has None. Its circuits are never
    released, because only Aer saves a handle's circuit, so its compiling row
    keeps their one estimate and no claim is made per class.
    """
    template = record.template
    return None if template is None or template.correction is not None else (record.backend_configuration_id, template.key)


def _restored_bytes(state, handle):
    """Return what restoring the released ``handle`` adds to ``native_held``.

    A row of a template adds its own ``_TEMPLATE_ROW_BYTES`` when another row holds
    the circuit in memory. Otherwise it brings the circuit back too, which together
    is what the compiling row's saved ``native_bytes`` states.
    """
    key = _template_key(handle.record)
    if key is None:
        return handle._native_bytes
    if key in state["template_circuits"]:
        return _TEMPLATE_ROW_BYTES
    source = state["template_sources"].get(key)
    compiling = None if source is None else state["handles"].get(source.content_id)
    return _TEMPLATE_ROW_BYTES if compiling is None else compiling._native_bytes


def _release_held(state, handle):
    """Take what releasing ``handle`` frees out of ``native_held``, and return whether its circuit leaves memory."""
    key = _template_key(handle.record)
    live = state["template_circuits"].get(key)
    if live is None:
        state["native_held"] -= handle._native_bytes
        return True
    state["native_held"] -= _TEMPLATE_ROW_BYTES
    live.owners -= 1
    if live.owners:
        return False
    state["native_held"] -= live.bytes
    del state["template_circuits"][key]
    return True


def _admit_template_rows(plan, run, templates, items, rebuild=frozenset()):
    """Compare the rows of ``templates`` not yet prepared, and their new circuits, with the memory limit before any charge.

    The actual population is known here. A row in ``rebuild`` counts as missing
    (``prepare_template_rows``). Each missing row keeps
    ``_TEMPLATE_ROW_BYTES``, and each template without a compiled record adds one
    circuit, sized from its first missing row, which is one more counting pass per
    new template. Their temporaries do not overlap, so the largest counts, and a
    static definition counts once. A compiled template whose circuit was released
    comes back for its new rows. A row that reuses a compiled template builds no
    circuit, so ``_bind_template_rows`` does not compare a circuit for it again.

    A template that states its size B (``_DeferredRowTemplate``) is not built yet,
    so its temporary also takes what building and lowering it hold together, for
    the one call of its block that the Program makes:

        usual temporary + _TEMPLATE_BUILD_COPIES * B + _TEMPLATE_PREDECESSOR_COPIES * B_before

    with ``B_before`` the size that the template before it in this request states.
    The last term is what Aer holds from that template:
    ``_AerPreparation._blocks`` keeps its source gate and translated definition until
    the next ``AerBackend.prepare`` calls ``keep_sources``, and
    ``_CacheEntries.previous`` keeps them until that preparation commits. The selected
    block's own definition stays in the total though the template replaces it, because
    it may already be cached and so contribute nothing here, and subtracting its size
    could leave too little. These are declared allowances for objects traced in the
    source, not an enclosure of what the process allocates, and calibration judges
    them. A template that a Program calls more than once has one gate definition per
    call, and a gate left by an earlier request is counted only when its template is
    the one before in this request. After the last template, ``_held_bytes`` does not
    read what Aer keeps of it.

    This covers the known parts only, as a coarse estimate. It does not cover:

    - what an eager ``Method.row_templates`` built before this call, as sampled QSP
      and a custom Method can, or what any Method allocates to group its rows;
    - the class circuits and kept base of IBM's folded templates, because folding
      needs the compilation that follows the first charge;
    - bodies of input preparations and what backend translation holds in passing.

    Rows of a template that the backend turns out not to bind count here as rows,
    and they pass the check of their own preparation afterwards.
    """
    state = run._state
    kept = temporary = calls = definitions = preceding = 0
    static, first_call = {}, None
    for template in templates:
        stated = template._retained_bytes or 0
        missing = [name for name, _ in template.rows
                   if items[name].prepared_id is None and (items[name].preparation_id is None or name in rebuild)]
        if missing:
            kept += len(missing) * _TEMPLATE_ROW_BYTES
            source = state["template_sources"].get((run.backend.content_id, template.key))
            if source is None:
                first = _admitted_point(items[missing[0]].realization, run=run, admit_memory=False)
                memory = _circuit_memory(run, first[6], 0)
                kept += first[7]
                temporary = max(temporary, (memory.temporary + _TEMPLATE_BUILD_COPIES * stated
                                            + _TEMPLATE_PREDECESSOR_COPIES * preceding))
                static.update(memory.static)
                calls += memory.calls
                first_call = first_call or memory.first_call
                definitions = max(definitions, memory.definitions)
            else:
                handle = state["handles"].get(source.content_id)
                if handle is not None and handle._native is None:
                    kept += _restored_bytes(state, handle)
        preceding = stated
    _admit_circuit_memory(run, _CircuitMemory(kept, temporary, static, calls, first_call, definitions),
                          "preparing the rows of these templates", whole=True)


def _index_template_record(state, record):
    """Index a record bound to a template by backend configuration and key, and by its correction class too.

    ``_bind_template_rows`` reads these indexes to find the record that
    holds a compiled template, and the record of a correction class already
    built. The first record of each index entry is kept, in the order the
    Run made its records or a reopened Run restores them from the run log.
    """
    if record.template is not None:
        key = (record.backend_configuration_id, record.template.key)
        state["template_sources"].setdefault(key, record)
        state["template_classes"].setdefault((*key, record.template.correction), record)


def _bind_template(plan, run, template, circuit, missing):
    """Prepare the experiments ``missing`` of one template (``_bind_template_rows``), withdrawing copies if no record is saved."""
    copies = []
    try:
        return _bind_template_rows(plan, run, template, circuit, missing, copies)
    except BaseException:
        _abandon_copies(copies)
        raise


def _bind_template_rows(plan, run, template, circuit, missing, copies):
    """Prepare the experiments ``missing`` of one template and return ``{name: handle}``.

    ``circuit`` is the template's circuit when no record of this Run holds the
    template, and None otherwise, when nothing here reads it.

    Every experiment passes the checks of ``prepare_experiment`` before its
    charge (``_admitted_point``), except the comparison of one bound circuit
    with the memory limit, because a row builds none and
    ``_admit_template_rows`` has compared what these rows do keep. The capacity
    check counts one preparation per experiment, so a template does not shrink
    the population that ``max_total_circuits`` protects. When no record of this Run holds
    the template, the first experiment's charge commits first (commit 2),
    its point is lowered with the template in place of the Method's block
    (``_lower_qiskit(templates=...)``) and the backend compiles that circuit
    once with the experiment's runtime seed. Binding a value row does no
    backend work, so the other experiments' charges commit with their
    records. Each record takes the template's native fields and runtime
    except the fields that ``bind_template`` returns for its experiment
    (Aer returns its runtime and counts sampling). All records, their charges, workflow rows and the template's
    one saved QPY commit in one transaction (commit 3). Construction work
    is counted on the compiling record only.

    When the backend folds the compiled template for each experiment
    (``fold_template``: IBM fractional gates with parameterized RZZ
    angles), the compiled template is the base from which the circuit of
    each correction class is built once (``template_class``), without
    another compilation, layout or seed. The experiments of a class bind
    its circuit with their transformed values, taken by name in that
    circuit's own parameter order. The compiled template's QPY is saved
    once, in the same transaction as the first class records, and every
    folded record refers to it (``TemplateBinding.base``). A later call
    loads it at most once per open Run (``restore_template``), reuses the
    record of a class already built and builds only new classes, all before
    its one commit, so an interruption leaves its experiments unprepared.
    Without saved compiled template bytes, new preparation refuses as for a
    record without a payload, and nothing is compiled again.
    """
    from nwqlib.execution import TemplateBinding
    state = run._state
    checked = [_admitted_point(item.realization, run=run, admit_memory=False) for _, item, _ in missing]
    admitted = [(*entry, *result[:4]) for entry, result in zip(missing, checked, strict=True)]
    source = state["template_sources"].get((run.backend.content_id, template.key))
    # The compiled circuit and every class circuit folded from it carry the
    # readout and shots of the experiment that compiled them, and the
    # unconditional-population check below runs only at compilation. So
    # every experiment, in this call or a later one, must have that readout.
    observation = admitted[0][5] if source is None else source.observation
    if any(entry[5] != observation for entry in admitted):
        raise ValueError("experiments of one template must share one readout (ObservationSpec, shots included), "
                         "because their circuits carry the readout and shots of the template's first prepared "
                         "experiment. Method.row_templates must give experiments with different readouts "
                         "different RowTemplate.key values. None of the template's new experiments was prepared.")
    run.check_capacity(preparations=len(admitted))
    names = tuple(parameter.name for parameter in template.parameters)
    rows = tuple(dict(zip(names, values, strict=True)) for _, _, values, *_ in admitted)
    # The compiled templates of this Run that are folded per experiment,
    # kept in memory after compilation or after their first load.
    bases = state.setdefault("template_bases", {})
    charges, payloads, compiled, synthesis = {}, [], source is None, 0
    if compiled:
        name, item, _, construction = admitted[0][:4]
        charge = PreparationCharge(execution="quantum_circuit", preparation_id=str(uuid4()),
                                   checkpoint_sequence=run.checkpoint_sequence or None)
        item = item.revise(preparation_id=charge.preparation_id)
        run._write((("preparation", charge.preparation_id, charge), ("workflow_item", name, item)))
        state["workflow_items"][name] = item
        state["preparations"] += 1
        state["preparation_charges"][charge.preparation_id] = charge
        run.progress("prepare")
        charges[name] = charge.preparation_id
        snapshot = str(uuid4())
        logical, owned, preparation, synthesis = _compile_template(plan, run, template, circuit, construction, observation,
                                                         item.runtime, charge.preparation_id, name, snapshot, copies)
        if observation.kind != "counts" and observation.population == "unconditional" and preparation.population != "unconditional":
            raise ValueError("selected unconditional readout has a conditioned or unknown population")
        reference = None if preparation.payload is None else PayloadRef(format=preparation.payload_format,
                                                                     bytes=len(preparation.payload), key=str(uuid4()))
        if reference is not None:
            payloads.append((reference, preparation.payload))
        shared = dict(snapshot=snapshot, **_native_fields(preparation, reference))
        parameters, native, runtime = preparation.parameters, preparation.native, item.runtime
        layouts = dict(quantum_layout=tuple(RegisterMap(name=name, bits=bits) for name, bits in logical.quantum_layout),
                       classical_layout=tuple(RegisterMap(name=reg.name, bits=tuple(owned.find_bit(bit).index for bit in reg))
                                              for reg in owned.cregs))
        receipt = LogicalPreparationReceipt(dynamic_visits=logical.dynamic_visits, reserved_work=logical.construction_work,
                                            defined_selections=logical.defined_selections)
        folded = run.backend.fold_template(native, rows)
    else:
        shared = {name: getattr(source, name) for name in _NATIVE_FIELDS}
        runtime, reference = source.runtime, source.template.base
        layouts = dict(quantum_layout=source.quantum_layout, classical_layout=source.classical_layout)
        receipt = source.logical
        if source.template.correction is None:
            # A handle with a saved native snapshot (Aer) reloads it; another
            # is restored from its preparation record and saved payload.
            handle = state["handles"].get(source.content_id)
            if handle is None or handle._native is None and handle._saved_native is None:
                handle = restore_prepared(source.content_id, run=run, for_submission=True)
            parameters, native, folded = source.template.parameters, handle._restore_native(), None
        else:
            native = bases.get(template.key)
            if native is None:
                if reference is None or state["journal"] is None:
                    raise ValueError("saved preparation has no native payload for new execution")
                native = bases[template.key] = run.backend.restore_template(
                    source, state["journal"].read_payload(reference), run=run)
            folded = run.backend.fold_template(native, rows)
    # A bound record repeats the template's lowering counts but reserves no work.
    bound = LogicalPreparationReceipt(dynamic_visits=receipt.dynamic_visits, reserved_work=0,
                                      defined_selections=receipt.defined_selections)
    # Each template circuit that runs, by correction class: its native
    # object, parameter names in binding-column order and record fields. A
    # backend that records native depths (IBM, template_depth) gives each new
    # class the depth of its circuit bound to the class's first row, and a
    # class read from its saved record keeps the saved depth.
    depth = getattr(run.backend, "template_depth", None)
    if folded is None:
        if sorted(names) != sorted(parameters):
            raise ValueError("the compiled template's parameters differ from the Method's template parameters")
        if compiled and depth is not None:
            shared = {**shared, "native_depth": depth(native, rows[0], run=run)}
        classes, folded = {None: (native, parameters, shared)}, tuple((None, row) for row in rows)
    else:
        classes = {}
        for correction in dict.fromkeys(correction for correction, _ in folded):
            record = state["template_classes"].get((run.backend.content_id, template.key, correction))
            if record is not None:
                handle = state["handles"].get(record.content_id)
                if handle is None or handle._native is None:
                    handle = restore_prepared(record.content_id, run=run, for_submission=True)
                classes[correction] = (handle._native, record.template.parameters,
                                       {name: getattr(record, name) for name in _NATIVE_FIELDS})
                continue
            # A new class is built from the compiled template, so its record
            # keeps the compilation's target, compiler, layouts and options,
            # with its own snapshot, saved circuit, operation count and the
            # environment of the process that built it.
            snapshot = str(uuid4())
            variant, variant_parameters, data, operations, environment = run.backend.template_class(
                native, correction, snapshot=snapshot, run=run)
            payload = None if data is None else PayloadRef(format=reference.format, bytes=len(data), key=str(uuid4()))
            if payload is not None:
                payloads.append((payload, data))
            fields = {**shared, "snapshot": snapshot, "payload": payload, "native_operations": operations,
                      "environment": environment}
            if depth is not None:
                first = next(columns for row_correction, columns in folded if row_correction == correction)
                fields["native_depth"] = depth(variant, first, run=run)
            classes[correction] = (variant, variant_parameters, fields)
    records, prepared, handles = [], [], {}
    for index, ((correction, columns), (name, item, _, construction, setting, _, items)) in enumerate(
            zip(folded, admitted, strict=True)):
        variant, variant_parameters, fields = classes[correction]
        # Each value is found by the name of the circuit's own parameter:
        # the Method's parameter for the compiled template, or the folded
        # value from fold_template for a class circuit.
        if not set(variant_parameters) <= set(columns):
            raise ValueError("the compiled template's parameters differ from the Method's template parameters")
        binding = TemplateBinding(key=template.key, parameters=variant_parameters, compiled=compiled and index == 0,
                                  values=tuple(columns[parameter] for parameter in variant_parameters),
                                  correction=correction, base=None if correction is None else reference)
        logical_receipt = receipt if binding.compiled else bound
        # The backend states the record fields that this experiment does not
        # share with the template, such as Aer's runtime seed (bind_template).
        bound_native, own = run.backend.bind_template(variant, binding.values, runtime=item.runtime)
        own = dict(own)
        record = PreparedArtifact(**{**fields, **own, "runtime": own.get("runtime", runtime)}, **layouts,
            plan_id=plan.content_id, realization_id=item.realization.content_id,
            construction_id=construction.content_id, realization=item.realization, observation=observation,
            logical=logical_receipt, backend_configuration_id=run.backend.content_id, preparation_time=_now(),
            construction_work_reserved=logical_receipt.reserved_work, template=binding)
        if name in charges:
            # The compiling experiment's charge saves the work of the
            # syntheses that its compilation made.
            charge = state["preparation_charges"][charges[name]]
            charge = charge.revise(prepared_id=record.content_id, synthesis_work=charge.synthesis_work + synthesis)
        else:
            charge = PreparationCharge(execution="quantum_circuit", preparation_id=str(uuid4()),
                                       checkpoint_sequence=run.checkpoint_sequence or None,
                                       prepared_id=record.content_id)
        item = (state["workflow_items"][name] if name in charges else item.revise(preparation_id=charge.preparation_id))
        item = item.revise(prepared_id=record.content_id)
        records += [("preparation", charge.preparation_id, charge), ("prepared", record.content_id, record),
                    ("workflow_item", name, item)]
        prepared.append((name, record, charge, item))
        # The compiling row's saved bytes state the circuit's estimate, which outlasts that
        # row while another row runs the circuit (_LiveTemplate).
        handles[name] = PreparedHandle._make(record, item.realization, bound_native,
                                             items, setting, construction.program.bindings,
                                             native_bytes=_TEMPLATE_ROW_BYTES + (checked[index][7] if binding.compiled
                                                                                 else 0))
    records.extend(("payload", payload.content_id, payload) for payload, _ in payloads)
    from nwqlib._run_archive import write_caches
    for name, record, _, _ in prepared:
        state["handles"][record.content_id] = handles[name]
    write_caches(run, tuple(records), payloads=tuple(payloads))
    state["synthesis_work"] += synthesis
    state["native_held"] += sum(handles[name]._native_bytes for name, _, _, _ in prepared)
    if folded[0][0] is None:
        state["template_circuits"].setdefault((run.backend.content_id, template.key), _LiveTemplate(
            native.circuit, checked[0][7] if compiled else 0)).owners += len(prepared)
    if compiled and folded[0][0] is not None:
        bases[template.key] = native
    for name, record, charge, item in prepared:
        state["preparations"] += name not in charges
        state["workflow_items"][name] = item
        state["prepared_artifacts"][record.content_id] = record
        _index_template_record(state, record)
        state["local_prepared_ids"].append(record.content_id)
        state["preparation_charges"][charge.preparation_id] = charge
        state["construction_work"] = known_sum(state["construction_work"], record.construction_work_reserved)
    for commit, _ in copies:
        commit()
    run.progress("prepare", len(state["local_prepared_ids"]), None)
    return handles


def _compile_template(plan, run, template, circuit, construction, observation, runtime, identity, name, snapshot, copies):
    """Lower one experiment's point with the template's ``circuit`` in place of its block, and let the backend compile it as ``snapshot``.

    Returns ``(logical, circuit, preparation, synthesis)``. The ``(commit,
    withdraw)`` callbacks of copies that constructors report join
    ``copies``, for the caller to run once the records are saved or after a
    failure. ``synthesis`` is the work of the exact dense syntheses that
    lowering and the backend made for the charge ``identity``
    (``Run._record_synthesis``), which the caller saves with that charge, as
    in ``prepare_experiment``.
    """
    from nwqlib.blocks.lowering import _lower_qiskit
    state = run._state
    selected = {record.content_id for record in construction.selections}
    blocks = tuple(block for block in plan.blocks if block.record.content_id in selected)
    state["synthesis_preparation"], state["workflow_current"], state["synthesis_pending"] = identity, name, 0
    try:
        logical = _lower_qiskit(construction, blocks=blocks, definition_cache=state["definitions"],
            method_context=run._method_context, receipt_commits=copies, record_synthesis=run._record_synthesis,
            templates={template.block_id: circuit})
        owned = logical.circuit
        position = len(owned.data) if observation.position is None else observation.position
        if position > len(owned.data):
            raise ValueError("observation position exceeds the selected circuit")
        preparation = run.backend.prepare(owned, observation=observation, runtime=runtime, position=position,
            source_definitions=tuple(state["definitions"][0].values()) + tuple(state["definitions"][1].values())
                + tuple(state["specializations"].values()), run=run, snapshot=snapshot)
    finally:
        state["synthesis_preparation"], state["workflow_current"] = None, None
        synthesis, state["synthesis_pending"] = state["synthesis_pending"], 0
    return logical, owned, preparation, synthesis


def _publishes_arrays(observation):
    """Whether an acquisition of ``observation`` saves binary arrays with its outcome.

    Amplitude and probability readouts do, and so does a trajectory with an
    amplitude or probability point. Their arrays commit in the outcome's
    transaction, so the acquisition opens a batch for them first
    (``Run._start_publications``).
    """
    kinds = {"amplitudes", "probabilities"}
    return observation.kind in kinds or any(point.kind in kinds for point in observation.positions)


def submit_experiment(prepared, *, run, checkpoint_sequence=None, collect=False):
    """Reserve the acquisition before the backend call, without rebuilding the preparation.

    ``Method.before_submit`` runs once, before the intent, so a Method can reject
    a sample set it cannot use (for example one from a reused fixed sampling
    stream) before any work is counted. The intent and reserved event then commit (commit 4 of
    the module docstring), the backend runs the saved native object, and the
    outcome is saved by ``Run._finish_attempt`` (commit 6) or recorded by
    ``Run._fail_attempt``. A synchronous backend has no commit 5. Classical routines
    and detached backends take their own routes with the same ordering.
    ``collect=True`` collects a synchronous outcome in its own transaction
    (commit 7 joins commit 6), as the static workflow does. A detached
    submission is collected after its refresh.

    Returns the observation chunk, or for a trajectory the tuple of its point
    chunks in schedule order, all from one acquisition.
    """
    _check_scope(run, prepared.realization)
    prepared._restore_native()
    if prepared.record.execution == "host_kernel":
        return _submit_host(prepared, run=run, checkpoint_sequence=checkpoint_sequence, collect=collect)
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
        data_bytes_reserved=readout_bytes(prepared, max_bytes=run.limits.max_data_bytes, run=run, attempt=attempt))
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
        run._finish_attempt(index, complete, chunk, native_simulations=result.metadata.get("native_simulations"),
                            collect=collect)
        return chunk
    except BaseException as error:
        run._fail_attempt(index, event, error, discarded=(
            result is not None and isinstance(error, Exception)
            and getattr(run.backend, "kind", None) in LOCAL_SIMULATORS))
        raise


def _statistic_stride(kind, *, key_width, shots=0, max_bytes=None):
    """Bytes one saved statistic reserves.

    A JSON statistic reserves the JSON size of its prototype and one
    separating comma. The prototypes are those of ``readout_bytes``. A Pauli value has
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
    """Check a readout's items and payload bytes before native preparation, and return its item count.

    These checks precede native construction and submission. Subtract the
    known declaration reservation before deriving any payload capacity. For
    positive homogeneous
    item count ``L``, check ``K <= remaining_items//L`` before forming
    ``K*L``. For a marginal and nonnegative remaining item capacity
    ``M_items``, reject when ``q_k >= M_items.bit_length()`` before forming
    ``1 << q_k``. Check heterogeneous additions against remaining capacity and
    apply the corresponding division check before multiplying by the chosen
    byte stride. Unresolved executable dimensions are rejected rather than
    counted as zero. Pauli and count values use the conservative JSON size of
    the prototype of ``_statistic_stride``, and probabilities the eight bytes
    per possible outcome of their binary array. Amplitude arrays keep their
    own binary payload. An unsupported reduction or a reduction with
    unresolved output shape fails before native preparation.

    For a single-endpoint readout the declaration reservation is the
    observation's own JSON, the part of the declaration's JSON ``J`` known
    before lowering. At submission ``readout_bytes`` counts the complete
    reservation, including the bindings and layouts, against the remaining
    data budget. The item capacity of each readout is the remaining payload
    bytes divided by its statistic stride (``_statistic_stride``), 16 bytes
    per complex128 amplitude, so ``items*stride`` never exceeds the
    remaining bytes.

    A trajectory subtracts the full known future storage for all points
    before deriving a value capacity. The complete contribution of the
    acquisition, including the newly saved declaration of its preparation
    record, is ``R_new selection = J_T + sum_k (D_k + H_k + V_k)``
    (``trajectory_reservation``, ``_trajectory_fixed_bounds``,
    ``_trajectory_value_bounds``). ``header`` holds the context that does not
    depend on lowering (experiment, setting, bindings, Plan and realization
    identities) and the logical layouts derived from the resolved registers
    (``_point_chunk_headers``). The Run's identifier, the known encoded
    lengths of the attempt UUID and of a SHA-256 content hash, the
    trajectory's content hash, each point's chunk key, explicit position and
    acquisition source are added to it. Only the end shorthand is resolved after
    lowering. It is counted at its shortest form, so this check counts no more
    than the final one, and the completed preparation record supplies the final
    bound at submission (``readout_bytes``). Without ``header``
    (when a Run is restored), the point headers are not counted. The
    per-point item count is ``point_items``.
    """
    from nwqlib._run_journal import _json_bound
    from nwqlib.core.planning import point_items

    limit = run.limits.max_data_bytes
    declaration = _json_bound(observation, limit)
    remaining = None if limit is None else limit - declaration
    if remaining is not None and remaining < 0:
        raise _cumulative_limit_error("max_data_bytes", limit, "readout declaration bytes", 0, declaration)

    def clause(stride):
        return _data_limit_clause(run, f" less the readout declaration, at {stride} bytes per entry")

    kind = observation.kind
    if kind == "counts" and observation.shots > MAX_COUNT:
        # The int64 count representation requires 1 <= shots <=
        # 2**63-1 before submission (ObservationSpec admits shots >= 1).
        raise ValueError(f"requested shots {observation.shots} exceed the int64 count limit 2**63 - 1 = {MAX_COUNT}")
    if kind != "trajectory":
        if kind == "amplitudes":
            stride = 16
        elif kind == "estimated_observable":
            stride = 1  # One provider estimate, counted at its actual size when saved.
        else:
            key_width = {"pauli_expectation": width, "probabilities": len(observation.qubits),
                         "counts": classical_width}.get(kind, 0)
            stride = _statistic_stride(kind, key_width=key_width, shots=observation.shots, max_bytes=limit)
        return readout_shape(kind=kind, details=observation, width=width, classical_width=classical_width,
                             repetitions=observation.shots, max_items=None if remaining is None else remaining // stride,
                             max_items_source=clause(stride))
    # Validate the whole schedule and its dimensions first. Every item costs
    # at least one byte, so the remaining bytes bound the item count.
    readout_shape(kind=kind, details=observation, width=width, classical_width=classical_width,
                  max_items=remaining, max_items_source=clause(1))
    # Then the complete trajectory law R = J_T + sum_k (D_k + H_k + V_k)
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
    declarations, headers = _trajectory_fixed_bounds(observation, common=common, per_point=per_point,
                                                     max_bytes=limit, job_length=_job_length(run))
    capacity = None if available is None else available - receipt - sum(declarations) - sum(headers)
    if capacity is not None and capacity < 0:
        trajectory_reservation(declarations, headers, (0,) * len(headers), available_bytes=available,
                               new_receipt_bytes=receipt, points=observation.positions, limit=limit)
    values = _trajectory_value_bounds(observation, width=width, capacity=capacity, max_bytes=limit,
                                      source=_data_limit_clause(run, " less the trajectory declarations and point "
                                                                     "headers"))
    trajectory_reservation(declarations, headers, values, available_bytes=available, new_receipt_bytes=receipt,
                           points=observation.positions, limit=limit)
    return sum(point_items(point, width=width) for point in observation.positions)


def admit_pure_trajectory(spec, *, backend_name, body_length, classical_width, body_issue, view_issues):
    """Check that a trajectory is one deterministic, noiseless coherent evolution of a pure state.

    The required model is a normalized pure initial state with fixed,
    bound, coherent evolution. At logical boundary ``b`` the observed state
    is ``|psi_b> = U_b ... U_1 |psi_0>``. For a general channel the
    unconditional state evolves instead as
    ``rho' = sum_a K_a rho K_a^dagger`` with ``sum_a K_a^dagger K_a = I``,
    and an individual stochastic trajectory has state proportional to
    ``K_a psi`` with outcome-dependent ``a``, so one saved trajectory
    statistic need not equal the unconditional statistic. The treatment of
    each feature that appears before an observed boundary, or in an executed
    intermediate view:

    - Fixed unitary gates, coherent controlled gates, barriers, a chosen
      unitary ``StatePreparation``, or resolved fixed repetitions of those
      operations: accept unchanged when the adapter supports the
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

    This is a sufficient structural rule for the required model, not a
    claim that every listed operation always mixes every input. Only the
    executed prefix through the last requested point, its preparation, and
    its executed views affect these quantities. ``body_issue`` is the first
    incompatible operation on the executed prefix, including hidden
    definitions. ``view_issues`` is computed during the same view traversal.
    Refusals name the first body/view feature that violates the model.

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
    emit are refused by lowering. Instructions inside a chosen block's
    native definition are not visible in the IR. Lowering converts each
    chosen block to a Qiskit ``Gate`` (``to_gate``), which Qiskit refuses
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
    """Ask the Run's Method for the reduction-work allowance of one trajectory point, or return None for no limit.

    A Method whose reductions count against an explicit work limit of its
    own defines the optional hook

        reduction_allowance(self, plan, point, *, observation, width, run) -> int | None

    which receives the Plan, the point's ``Realization``, its resolved
    trajectory ``ObservationSpec``, the logical width of its selected
    Program and the Run. The hook may first apply the Method's own check of
    the private workspace its reducers allocate, raising ``ValueError`` to
    refuse the point, and then returns the remaining allowance in the unit
    of the registered ``work`` functions, or None when the Method sets no
    work limit. ``admit_reductions`` checks the summed registered work
    against a finite allowance. The call happens before native work: before
    the preparation charge in ``prepare_experiment``, and again when the
    static path hands an existing preparation, live or saved by an earlier
    process, to submission (``_prepare_static_item``), so a resumed static
    Run is checked against its current work budget and limits. The hook can
    therefore be asked more than once for one point. It reads the Method's
    work count and does not change it. Only a trajectory with a reduction
    point asks. A Method without the hook sets no work limit on its
    reductions.
    """
    if observation.kind != "trajectory" or not any(item.kind == "reduction" for item in observation.positions):
        return None
    hook = getattr(run.plan.method, "reduction_allowance", None)
    if hook is None:
        return None
    allowance = hook(run.plan, point, observation=observation, width=width, run=run)
    if allowance is not None and (type(allowance) is not int or allowance < 0):
        raise ValueError(f"{type(run.plan.method).__name__}.reduction_allowance must return a nonnegative integer "
                         f"work allowance or None, not {allowance!r}")
    return allowance


def _admit_trajectory_work(observation, *, width, run, allowance):
    """Check a trajectory's reduction work and saved simulator states before native preparation.

    Reductions: every reduction point's registered shape and execution, and
    with a finite ``allowance``, the calling Method's remaining work
    allowance, their summed registered work (``core.planning.admit_reductions``).
    A size that the caller supplies is never the reservation, which comes
    from the registry's shape, work, workspace and execution functions.

    Saved buffers: a backend with ``admit_simulation_memory`` (Aer) checks
    the live state, every save it keeps (probability marginals, saved
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
    one), at least the number of distinct saved states. The reductions run
    one at a time after the simulation, while the kept saves are held and
    the live state is released, so the kept saves plus the largest
    registered reduction workspace (``Reducer.workspace``) must also fit
    ``simulator_memory_mb``. The kept saves are the bytes that
    ``admit_simulation_memory`` returns (Aer: saved states, probabilities
    and expectations), and otherwise the ``S*16*2**w`` bytes of the saved
    states.

    Outputs: each amplitude point's materialization
    (``AmplitudeReadout.admit_materialization``). The saved JSON of the
    reduction outputs is a point value payload ``V_k`` of the saved-data
    reservation (``_trajectory_value_bounds``).
    """
    from types import MappingProxyType
    from nwqlib.core.planning import admit_reductions, registered_reducer

    admit_reductions(observation, width=width, allowance=allowance)
    # A backend that executes trajectories admits its own simulation-phase
    # arrays (AerBackend.admit_simulation_memory: live state, kept
    # saves, marginal workspace), also when no state is saved.
    admit = getattr(run.backend, "admit_simulation_memory", None)
    memory_mb = run.limits.simulator_memory()
    kept = None
    if admit is not None and memory_mb is not None:
        try:
            kept = admit(observation, width=width, memory_mb=memory_mb, run=run)
        except ValueError as error:
            raise ValueError(f"the trajectory's simulation-phase arrays exceed the Run's memory limit before native "
                             f"preparation: {error}. {_limit_remedy('simulator_memory_mb')}") from None
    saved = len({point.position for point in observation.positions if point.kind in ("amplitudes", "reduction")})
    if saved and admit is None and memory_mb is not None:
        limit = memory_mb << 20
        needed = (1 + saved) * 16 << width
        if needed > limit:
            raise ValueError(f"the trajectory saves {saved} simulator state(s) of {width} qubits; with the live "
                             f"state they need {needed} bytes, more than simulator_memory_mb="
                             f"{memory_mb} ({limit} bytes) before native preparation. "
                             f"{_limit_remedy('simulator_memory_mb')}")
    # The reductions run one at a time while every kept save is held, so
    # the largest registered reduction workspace adds to the kept saves.
    workspace = max((reducer.workspace(MappingProxyType(json.loads(point.parameters)), width)
                     for point in observation.positions if point.kind == "reduction"
                     for reducer in (registered_reducer(point.reducer),) if reducer.workspace is not None),
                    default=0)
    if workspace and memory_mb is not None:
        limit = memory_mb << 20
        held = saved * 16 << width if kept is None else kept
        needed = held + workspace
        if needed > limit:
            raise ValueError(f"the trajectory's reductions run while its saves, {held} bytes with {saved} saved "
                             f"simulator state(s) of {width} qubits, are held, and their largest reduction needs "
                             f"{workspace} bytes of arrays besides them, {needed} bytes together, more than "
                             f"simulator_memory_mb={memory_mb} ({limit} bytes) before native preparation. "
                             f"{_limit_remedy('simulator_memory_mb')}")
    # An amplitude point materializes its selected output as a single-endpoint
    # amplitude readout does, and is admitted by the same owner before work.
    for point in observation.positions:
        if point.kind == "amplitudes":
            point.amplitudes.admit_materialization(max_bytes=run.limits.max_data_bytes,
                                                   max_bytes_source=_data_limit_clause(run))


# Encoded-length prototypes of an attempt UUID (``str(uuid4())``, 36 ASCII
# characters) and of a SHA-256 content identity. Both have known encoded
# lengths before their values exist.
_UUID_PROTOTYPE = "0" * 36
_CONTENT_ID_PROTOTYPE = "sha256:" + "0" * 64

def trajectory_reservation(declarations, headers, values, *, available_bytes, new_receipt_bytes=0, points=(),
                           limit=None):
    """Return a trajectory acquisition's saved-data reservation, checked against the available bytes.

    For one acquisition with ``K >= 1`` points, with ``J`` the
    ``_run_journal._json_bound`` of the run-log encoding (computed fields
    excluded):

    - ``J_T`` (``new_receipt_bytes``): bound of the full trajectory
      declaration in its preparation record, including the JSON of its
      ``observation`` field, counted when that record is first saved.
    - ``D_k`` (``declarations``): ``J({"observation": S_k})`` for the point's
      one-point spec ``S_k``.
    - ``H_k`` (``headers``): bound of that chunk's known fields and, for a
      probability point, of the JSON of its probability record and of the
      net rows written when its arrays are saved, once per point.
    - ``V_k`` (``values``): worst-case value payload of the actual
      representation, excluding the fields in ``D_k`` or ``H_k``.

    The complete contribution of the acquisition, including a newly saved
    declaration in the preparation record, is ``R_new selection = J_T +
    sum_k (D_k + H_k + V_k)``. After the preparation record has been saved
    the execution reservation is ``R_pending = sum_k (D_k + H_k + V_k)``
    (``new_receipt_bytes=0``, the submission-time case). Adding ``J_T``
    again would reserve a second copy of an already counted declaration.
    The variable fields of each point chunk, such as its job text, and the
    other completion rows of the acquisition are not reserved. They are
    counted at their actual size when the outcome is saved (``Run._write``).
    Each amount is compared with the remaining bytes before it is added, so
    no product is formed beyond the available budget.

    The refusal messages name the point, the counted, requested and
    remaining bytes and the limit.

    Args:
        declarations: ``D_k`` per point, in schedule order.
        headers: ``H_k`` per point.
        values: ``V_k`` per point.
        available_bytes: The remaining ``max_data_bytes`` of the Run, or None without a limit.
        new_receipt_bytes: ``J_T`` when the preparation record is not yet saved, else zero.
        points: The trajectory's points, named in a refusal.
        limit: ``max_data_bytes``, named in a refusal.
    """
    remaining = available_bytes
    total = 0
    if exceeds(new_receipt_bytes, remaining):
        raise ValueError(f"trajectory fixed storage exceeds remaining max_data_bytes: the receipt declaration "
                         f"requests {new_receipt_bytes} bytes, {remaining} remain (max_data_bytes={limit}). "
                         f"{_limit_remedy('max_data_bytes')}")
    total += new_receipt_bytes
    if remaining is not None:
        remaining -= new_receipt_bytes
    for index, (declaration, header, payload) in enumerate(zip(declarations, headers, values, strict=True)):
        for amount in (declaration, header, payload):
            if exceeds(amount, remaining):
                point = points[index].id if index < len(points) else index
                raise ValueError(f"trajectory point {point!r} storage exceeds remaining max_data_bytes: it requests "
                                 f"{amount} bytes, {remaining} remain after {total} counted "
                                 f"(max_data_bytes={limit}). {_limit_remedy('max_data_bytes')}")
            total += amount
            if remaining is not None:
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

    ``header`` supplies the identities of the Plan and realization, the
    experiment, setting, bindings and logical layouts. Before lowering,
    callers supply prototypes of the identifiers with their encoded lengths,
    zero for an unresolved end boundary, and the adapter's target version or
    its shortest valid text. Submission and saving supply the identities,
    boundaries and version of the preparation record.

    ``readout_bytes`` counts the bytes of these maps in ``H_k``, and
    ``_trajectory_chunks`` uses them when saving each point. The declaration and variable fields
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
    """The ``header`` of a prepared point read from its preparation record, for its chunk-field and metadata formulas.

    It holds the same fields that ``_selected_readout`` derives before
    lowering, so submission and saving count and save the values of the
    preparation record.
    """
    record = prepared.record
    return dict(plan_id=record.plan_id, realization_id=prepared.realization.content_id,
                experiment=prepared.realization.experiment, setting=prepared._setting,
                bindings=prepared._bindings, quantum_layout=record.quantum_layout,
                classical_layout=record.classical_layout)


def _job_length(run):
    """The length of the backend's job identifier that the adapter establishes, or 1 (the shortest valid job text)."""
    length = getattr(getattr(run, "backend", None), "native_job_id_length", None)
    return length() if callable(length) else 1


def _trajectory_fixed_bounds(observation, *, common, per_point, max_bytes, job_length=1):
    """Return the point declarations D_k and point headers H_k of a trajectory.

    D_k is J of the point's declaration field map. H_k contains its known
    chunk fields and, for probabilities, the ProbabilityArrays record of the
    largest valid encoding, the growth of the publication rows and the
    payload-reference rows (``_probability_point_metadata``). Binary arrays
    are V_k, counted separately by ``_trajectory_value_bounds``. The
    variable fields of a point chunk, such as its job text, are counted when
    the outcome is saved.

    The field-tree rule for a map is two braces, one comma per adjacent pair
    and J(key) + one colon + J(value) for every field. Adding a field to the
    nonempty common map therefore adds J(key) + 2 + J(value). The singleton
    probability record adds exactly its J to the empty values tuple.
    Splitting declaration and header maps conservatively counts their
    delimiters, so their sum bounds the reserved part of the chunk.

    Without common headers, as in the cardinality check when a Run is
    restored, H_k is zero. Submission counts the complete headers from the
    saved preparation record before it runs another acquisition. The
    preparation record holds the complete trajectory declaration. The run
    log omits the chunk's observation field, so D_k is a conservative
    reservation.
    """
    from nwqlib._run_journal import _json_bound

    points = observation.positions
    declarations = [_json_bound({"observation": ObservationSpec.point_readout_fields(point)}, max_bytes)
                    for point in points]
    if common is None:
        return declarations, [0] * len(points)
    shared = _json_bound(common, max_bytes)
    headers = [shared + sum(_json_bound(name, max_bytes) + 2 + _json_bound(value, max_bytes)
                            for name, value in fields.items()) for fields in per_point]
    for index, point in enumerate(points):
        if point.kind == "probabilities":
            headers[index] += _probability_point_metadata(point, common, per_point[index], "x" * job_length,
                                                          max_bytes)
    return declarations, headers


def _probability_point_metadata(point, common, fields, job, max_bytes):
    """Return ``F_k``, the reserved JSON of one probability point's record and array rows.

    For encoding E, ``F_k(E)`` is ``J(ProbabilityArrays_E)`` plus, for every
    array, the publication row's net growth and its payload-reference row. A
    singleton values tuple adds exactly ``J(ProbabilityArrays_E)`` to the
    empty tuple.

    ``F_k`` is the maximum over dense and sparse encodings. The sparse
    case uses ``s_max = (2**q - 1)//(W + 1)``, from the strict binary
    storage choice ``(W + 1)*s < 2**q``. Integer field sizes are monotone in
    their nonnegative maxima, and each scalar float has the JSON upper bound
    of 32 bytes. Digests and record identities have fixed encoded lengths.
    The publication reservation contains only integers and strings, so its
    actual encoded size equals J.

    The trajectory reserves F_k in its point header. ``job`` supplies a
    prototype of the job identifier with the escaped size that the adapter
    establishes, or the shortest valid text when that size is unknown. A
    longer job text is counted when the arrays are saved.

    This constructs scalar metadata only. Callers check the logical outcome
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
    return max(_json_bound(record, max_bytes) + sum(publication(item) for item in items)
               for record, items in candidates)


def _repeated_bound(element_bound, count):
    """``J`` of a tuple of ``count`` elements of bound ``element_bound``: ``count + 1 + count*element_bound``, or 2 when empty."""
    return 2 if count == 0 else count + 1 + count * element_bound


def _trajectory_value_bounds(observation, *, width, capacity, max_bytes, source=None):
    """Return each trajectory point's worst-case value payload ``V_k`` in saved bytes.

    The value formulas of the representations, which form disjoint
    categories:

    1. A Pauli value saved as a JSON record uses the JSON size of its
       prototype. For ``L_k`` entries, ``L_k * _statistic_stride(...)``
       covers the JSON of all elements and the commas.
    2. Dense or adaptively sparse probability arrays reserve ``8 * 2**q_k``
       bytes. With ``W_k = max(1, (q_k+63)//64)``, sparse storage costs
       ``8*(W_k+1)*s_k`` and is chosen only when smaller than dense storage.
       The point's array manifests, summaries and publication-row growth
       are covered by its fixed header at their largest valid encoding.
    3. An amplitude point reserves its declared complex128 output, normally
       ``16*L_k``. Its scalar masses and manifests are in the checked
       variable remainder, not complex-vector bytes.
    4. A registered reduction's outputs are saved as JSON ``ReducedValues``
       records, one per registered component, so they also use the JSON size
       of their prototype, in which a finite float has the upper bound of 32
       bytes, an int64 the decimal length of ``-2**63``, and a tuple of ``n``
       elements of bound ``e`` has bound ``n + 1 + n*e`` (``_repeated_bound``),
       plus one separating comma per component.

    Only probability and amplitude points save binary arrays. A
    probability point reserves eight bytes per outcome (formula 2) and an
    amplitude point 16 bytes per complex128 value (formula 3). Pauli and
    reduction values are JSON records (formulas 1 and 4). These formulas do not
    include declarations or headers, and they do not imply that a scalar
    saved as JSON occupies eight saved bytes. ``point_items`` supplies
    ``L_k``, which for a probability point is ``2**q_k``, independently of
    how many entries its sparse array eventually saves. With a ``capacity``
    (the check before lowering), each point's item capacity is ``capacity``
    divided by its stride, checked before ``1 << q_k`` is formed, and a
    reduction output larger than ``capacity`` is refused before its prototype
    is counted. The formulas are given above, and ``docs/development/execution.md``
    (Readout reservation) gives the reservation of each readout kind.
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


def readout_bytes(prepared, *, max_bytes=None, run=None, attempt=None, result_key="0"):
    """Reserve an acquisition's saved bytes before it runs, without enumerating outcomes.

    A single endpoint reserves its declared values and J_decl, the JSON
    bound of the chunk's observation, bindings, quantum and classical
    layouts, experiment and setting. A probability endpoint's values
    are B = 8*2**q, which bounds the dense or adaptively sparse binary
    payload.

    A trajectory reserves sum_k(D_k + H_k + V_k) and needs its Run and
    attempt. Its preparation record holds the complete trajectory
    declaration and is counted when it is saved. D_k bounds each point's
    declaration, H_k covers its known chunk fields and the largest valid
    encoding of the probability record and its publication rows, and V_k
    covers the values in their representation.

    Counts and Pauli statistics use the JSON bounds of their prototypes.
    Counts have at most min(shots, 2**width) entries, each at most shots.
    Host scalars use their declared labels and the JSON size of their
    prototypes, alongside the declared bytes of the kernel's arrays and
    application records. Amplitudes reserve the binary bytes of the declared
    output. A provider estimate reserves its declaration and the fields of its
    value that are known before submission, which the backend adapter bounds
    (``estimate_bytes``). Its provider metadata, known only after the paid
    job, is counted at its actual size when the outcome is saved.

    The other fields of the saved chunk and the completion rows of the
    acquisition, such as its job text, timing and provider status, are not
    reserved. Run._write releases the acquisition's pending reservation and
    counts its actual JSON and binary bytes. These are serialized-data
    bounds, not process-memory bounds.
    """
    from nwqlib._run_journal import _json_bound

    record = prepared.record
    spec = record.observation
    if spec.kind == "trajectory":
        if run is None or attempt is None:
            raise ValueError("a trajectory reservation needs its Run and attempt, whose values its point chunks hold")
        common, per_point = _point_chunk_headers(spec, _receipt_header(prepared), run_id=run.run_id,
                                                 prepared_id=record.content_id, attempt=attempt,
                                                 boundaries=record.boundaries, backend_kind=run.backend.kind,
                                                 target_version=record.target.version, result_key=result_key)
        declarations, headers = _trajectory_fixed_bounds(spec, common=common, per_point=per_point,
                                                         max_bytes=max_bytes, job_length=_job_length(run))
        width = sum(len(reg.bits) for reg in prepared.record.quantum_layout)
        values = _trajectory_value_bounds(spec, width=width, capacity=None, max_bytes=max_bytes)
        return trajectory_reservation(declarations, headers, values, available_bytes=run._available_data_bytes(),
                                      points=spec.positions, limit=max_bytes)
    # These declaration fields have a fixed reservation, even for thousands
    # of Pauli labels.
    declaration = _json_bound(dict(observation=spec, bindings=prepared._bindings,
        quantum_layout=prepared.record.quantum_layout,
        classical_layout=prepared.record.classical_layout,
        experiment=prepared.realization.experiment, setting=prepared._setting), max_bytes)
    if spec.kind == "amplitudes":
        return spec.amplitudes.output.data_bytes + declaration
    if spec.kind == "estimated_observable":
        return declaration + run.backend.estimate_bytes(prepared.record, max_bytes=max_bytes)
    if spec.kind == "host_scalars":
        # Per scalar: the bound of a prototype ScalarValue minus its one-letter
        # label, plus the actual escaped label. The prototype has the 32-byte
        # float envelope, the longest frame name, a full-length parent identity
        # and no unavailability text, which is counted when it is saved. Then
        # add the declared bytes of the kernel's output arrays and application
        # receipts, one comma between consecutive scalars, and the declaration.
        kernel = prepared._native.record
        base = ScalarValue(label="x", value=0., frame="encoded_branch", parent_id="sha256:"+"f"*64)
        constant = _json_bound(base, max_bytes)-_json_bound(base.label, max_bytes)
        values = sum(constant+_json_bound(label, max_bytes) for label in kernel.scalars)
        return (kernel.data_bytes + kernel.application_bytes + values + max(0, len(kernel.scalars)-1)
                + declaration)
    if spec.kind == "probabilities":
        # B = 8*D for the D = 2**k possible outcomes (prepared._items, from
        # readout_shape): the adaptive encoding stores at most 8*D bytes, and
        # before acquisition the sparse count is unknown.
        return 8 * prepared._items + declaration
    if spec.kind == "counts":
        width = sum(len(reg.bits) for reg in prepared.record.classical_layout)
        prototype = CountBin(bits="0"*max(1, width), count=spec.shots)
    else:
        prototype = PauliValue(label="I"*max(map(len, spec.labels)), value=0.)
    # Terms: prepared._items statistics, each bounded by the prototype, one comma
    # between consecutive items, and the declaration. The prototype has the
    # widest key, and either the 32-byte envelope of a real value or, for a
    # count, the decimal length of the requested shot number, which no single
    # count can exceed.
    item = _json_bound(prototype, max_bytes)
    return prepared._items*item + max(0, prepared._items-1) + declaration




def _native_timing(started, reference):
    """Wall-clock seconds of one local backend call since ``started`` (``perf_counter``)."""
    return TimingObservation(seconds=perf_counter()-started, scope="native_call_wall",
        source=Source(name="perf_counter", version="python", domain="one local backend call", reference=reference))


def _quantum_chunk(prepared, result, *, run, attempt, result_key="0"):
    """Decode backend output and check it against its own preparation record.

    The exact-probability window depends on the executed circuit's operation
    count, which only the preparation record knows. The unit bound is therefore
    checked here, where backend output first meets its preparation record. A standalone Result checks it
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
    and resolved boundary from the preparation record, and its key
    ``point_chunk_key(result_key, point)``. Its values follow the point's
    one-point readout and are checked like that readout's. Every point
    chunk of the acquisition shares its run, attempt and job. Its fixed
    fields are the header maps whose bytes ``readout_bytes`` counted
    (``_point_chunk_headers``), and each saved chunk is checked against
    them.
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
    float64 buffer, outcome j at position j, which saving keeps without
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
    indices of the other forms and chooses the saved encoding.
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
    """Translate one backend result into an ObservationChunk with the layouts of the preparation record.

    The number of returned entries is checked against the selected readout
    cardinality before any record is built. Backend count keys follow Qiskit's
    convention, in which spaces separate classical registers and the rightmost
    character is classical bit 0. The spaces are removed here, and the
    classical layout of the preparation record names the registers. Counts
    also arrive as an indexed pair ``(indices, counts)`` in the ``Histogram``
    layout and are saved as JSON count records. A probability marginal, a
    dense buffer, an indexed pair or a mapping, is saved as binary arrays with
    the acquisition, in the batch of its outcome (``_probability_output``). The chunk carries its original run, attempt,
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


def _point_realization(plan, point, *, program=None):
    """The Realization of one static point (`StaticPoint`).

    A caller that already holds the point's selected Program passes it as
    ``program``, whose bindings are the point's effective bindings, so the
    point is not resolved again.
    """
    name = point.experiment.name
    return plan.resolve(name, bindings=point.bindings) if program is None else Realization(
        plan_id=plan.content_id, experiment=name, bindings=program.bindings)


def _static_items(plan, run, points):
    """The saved workflow rows of the static points ``points`` (`StaticPoint`), creating the missing ones together.

    The rows are those of a template batch, which ``prepare_template_rows``
    holds before any of its points is prepared. Each missing row draws its
    runtime seed in the order of ``points`` (Plan order), and all of them
    commit with the RNG position in one transaction (commit 1 of the module
    docstring), so the seeds never change across restarts. The seeds are
    kept only with that transaction. If it is refused, for example by
    ``max_data_bytes``, or fails, the backend stream returns to its position
    before the first draw, so a retry after ``Run.extend_limits`` draws the
    same seeds as a reopened Run. A point prepared on its own creates its
    row with its preparation charge instead (``_prepare_static_item``).
    """
    state, created = run._state, {}
    start = run.rng.backend.bit_generator.state
    try:
        for point in points:
            if point.key not in state["workflow_items"] and point.key not in created:
                created[point.key] = WorkflowItem(realization=_point_realization(plan, point),
                                                  runtime=RuntimeOptions(seed=run.rng.next_seed()))
        if created:
            run._write((*(("workflow_item", key, item) for key, item in created.items()),
                        ("rng", "current", run.rng.snapshot())))
    except BaseException:
        run.rng.backend.bit_generator.state = start
        raise
    state["workflow_items"].update(created)
    return {point.key: state["workflow_items"][point.key] for point in points}


class PreparationNotRebuilt(RuntimeError):
    """A static item's local preparation has a charge but no preparation record, and the Run was not asked to build it again.

    ``_unbuilt`` decides this and ``_not_rebuilt`` builds the error.
    ``execute_static`` raises it before it submits any new experiment, and
    ``_prepare_static_item`` raises it for its own item.
    ``Run.resume(rebuild_preparations=True)`` is the explicit request that
    leads past it. A caller that
    continues a reopened Run can tell this outcome, which only new counted
    work could complete, from an error that an uninterrupted Run meets as well
    (``algorithms.qhd._durable.unrecoverable``, whose resume functions make no
    such request).
    """


def _unbuilt(run, item):
    """Whether a static workflow item holds a local preparation charge without a preparation record.

    Such a preparation failed or was interrupted after its charge (commit 2
    of the module docstring). The charge is already counted, and the Run never
    repeats a preparation attempt on its own, since that would be new counted
    work the caller did not request, so the Plan cannot complete in this Run
    without that request (``_rebuild_static_preparations``).
    A saved remote preparation does not count, because its compilation can
    still be refreshed.
    """
    return (item.prepared_id is None and item.preparation_id is not None
            and item.preparation_id not in run._state["remote_preparations"])


def _not_rebuilt(setting):
    """The ``PreparationNotRebuilt`` error for ``setting``, with one message for both raisers."""
    return PreparationNotRebuilt(f"the original preparation of setting {setting!r} failed or was interrupted "
                                 "and no replacement was built, so this Run submits nothing more. "
                                 "run.resume(rebuild_preparations=True) prepares each such setting once more in "
                                 "this Run. The failed attempt stays counted, each new circuit preparation counts "
                                 "against max_total_circuits, and no job already submitted is sent again. If a "
                                 "limit caused the failure, raise it first with run.extend_limits")


def _rebuild_static_preparations(run):
    """Prepare again, once each, the static rows that a counted local preparation left without a record.

    This is the caller's explicit request, ``Run.resume(rebuild_preparations=True)``.
    Preparation attempts P and execution attempts are counted separately and
    nothing is refunded. With U the eligible rows captured by this call, the bound
    applies to the charges of this one rebuild, and over the whole continuation

        P_after = P_before + k_rebuild + k_new,    0 <= k_rebuild <= |U|

    with k_new the first preparations of rows not prepared before. These count
    committed charges, including those whose preparation then fails, and they are
    lifecycle facts, not numerical claims. A rebuild creates no execution attempt.
    The request is not saved, so a later ``wait`` or ``resume`` makes no new attempt
    from it, and a second failure needs a new explicit request.

    A row is eligible only when all of these hold: it points at a counted local
    preparation (``_unbuilt``, and its charge has no record), there is no remote
    preparation for it, and no submitted or uncertain execution is attached to it,
    either on the row or as an event of a record of its realization. Known and
    uncertain submissions keep their identity and are never sent again.

    The row keeps its realization, bindings and runtime seed. The new charge and the
    row's new pointer commit in one transaction, which ``prepare_experiment`` or
    ``_bind_template_rows`` already does for a row that exists, and restore reads
    each charge on its own, so one row with several charges needs no format
    change. Each interruption therefore reopens at a committed prefix of
    ``charged/no-record -> new charged/no-record -> prepared record -> submitted``.

    Limits are checked before the first new charge (``_admit_static_workload``), and
    each new charge is refused on its own as any preparation is. The new preparation
    payload has no way around ``max_data_bytes``. A row of a template goes through
    ``prepare_template_rows``, so a template that a saved record holds is reused and
    never compiled again to get a new row. A row that no template takes is prepared
    on its own. The seed and the deterministic offline harness make the recovered
    result equal an uninterrupted one there. A live backend's compilation or
    randomness can differ, and only the associations and the absence of repeated
    submissions are claims for it.
    """
    plan, state = run.plan, run._state
    items = state["workflow_items"]
    unbuilt = {key: item for key, item in items.items() if _unbuilt(run, item)}
    if not unbuilt:
        return
    records = state["prepared_artifacts"]
    attached = {records[event.prepared_id].realization_id for event in state["events"]}
    eligible = frozenset(key for key, item in unbuilt.items()
                         if state["preparation_charges"][item.preparation_id].prepared_id is None
                         and item.attempt is None and item.realization.content_id not in attached)
    if not eligible:
        return
    _admit_static_workload(plan, run)
    points = tuple(point for point in plan._static_points() if point.key in eligible)
    templates = _row_templates(plan, run, points) if _groups_points(plan, run) else None
    if templates is not None:
        prepare_template_rows(plan, run=run, points=points, templates=templates, rebuild=eligible)
    for point in points:
        if items[point.key].prepared_id is None:
            _prepare_static_item(plan, run, point, rebuild=True)


def _prepare_static_item(plan, run, point, *, held=None, rebuild=False):
    """Return the item's one preparation, in memory, restored from its preparation record, or new.

    A point without a workflow row is prepared at once, and
    ``prepare_experiment`` draws its runtime seed and creates its row in the
    transaction of its preparation charge (commit 2 of the module
    docstring), so a refusal before that charge leaves neither. An item that
    ``_unbuilt`` finds is refused rather than rebuilt, with
    ``PreparationNotRebuilt``, unless ``rebuild`` carries the caller's request
    (``_rebuild_static_preparations``), and it then takes a new charge. The item's
    seed stays in its workflow row.
    The reductions of an existing preparation are checked again against
    the allowance of the Method's ``reduction_allowance`` hook before it is
    returned (``_method_reduction_allowance``). A new preparation is
    checked inside ``prepare_experiment``.
    ``held`` is the point's ``(Experiment, SelectedConstruction)``
    pair when the caller already holds it, which a new preparation reuses.
    """
    key = point.key
    item = run._state["workflow_items"].get(key)
    if item is None:
        realization = _point_realization(plan, point, program=None if held is None else held[1].program)
        run._state["workflow_current"] = key
        try:
            return prepare_experiment(realization, run=run, held=held)
        finally:
            run._state["workflow_current"] = None
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
    if item.preparation_id is not None and not rebuild:
        if _unbuilt(run, item):
            raise _not_rebuilt(key)
        from nwqlib._remote_preparation import refresh_preparation
        return refresh_preparation(item.preparation_id, run=run)
    run._state["workflow_current"] = key
    try:
        return prepare_experiment(item.realization, run=run, runtime=item.runtime, held=held)
    finally:
        run._state["workflow_current"] = None


def _fixes_footprint(plan, experiment):
    """Whether every point of the batch ``experiment`` has the memory of its first point.

    A point's Program comes from the Plan's own selections with the point's
    bindings (``Plan._selected_program``), and ``Method.specialize_experiment``
    can change only its readout. In the counting pass
    (``lowering._count_definition``), what a circuit keeps, its instructions and
    its direct preparation trees depend on the reachable Repeat counts, the widths
    of the registers it allocates, and each block's values, which do not vary with
    its arguments. So when no such count or width reads an axis, they are the same
    at every point. The expressions are evaluated without the axis values, as
    ``Plan._point_shots`` does, so that a value depending on an axis is None.
    """
    from nwqlib.ir import Allocate, ExprRef, Measure, Repeat, Reset
    batch = plan._program_nodes[experiment.batch]
    if not batch.axes:
        return True
    admission = plan._program_admission
    context = admission.binding_map(admission.p.bindings)
    context.update(admission.binding_map(batch.settings[experiment.setting_index].bindings))
    values = admission.expressions(context)
    seen, pending = set(), [batch.body]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        node = admission.nodes[name]
        count = (node.count if isinstance(node, Repeat) else
                 admission.registers[node.wire].width if isinstance(node, (Allocate, Measure, Reset)) else None)
        if isinstance(count, ExprRef) and values[count.expression] is None:
            return False
        pending.extend(children(node))
    return True


def _static_points_workload(plan, run, fixed, *, whole):
    """Check the readouts of a static Plan against the backend and return ``(circuits, shots, held, workload)``.

    ``fixed`` pairs each experiment with its ``Plan._point_shots``. A counts,
    Pauli or probability batch whose Program fixes the shots gives every point
    the same shots and a readout kind that a target treats alike, the only
    fields that a target reads for these kinds (a Method can turn probabilities
    into their amplitude form). Its first point stands for the others, which
    are counted from the axis bounds and resolved only when each is prepared.
    Any other experiment is resolved point by point, because its shots or the
    readout fields that a target reads, such as an estimate's coefficients or a
    trajectory's positions, can differ between points.

    ``circuits`` and ``shots`` count the whole Plan and ``held`` is the first
    point's ``(Experiment, SelectedConstruction)``, kept for its preparation.
    With ``whole``, ``workload`` is the ``_CircuitMemory`` of every point that is
    not yet prepared: the native circuits of all of them, which their handles keep,
    the new static definitions of all of them once each, and the largest temporary.
    One point's instruction list gives way to the next, but its frontier stays live
    while the next one is built, so a temporary includes the preceding frontier.

    With ``whole``, a first point stands for the others of its experiment only when
    they have its memory (``_fixes_footprint``), and it then multiplies what it keeps
    by their number. Arguments equal at the first point can differ at later ones, so
    its own keys do not bound a later frontier. Such a point takes
    ``_CircuitMemory.frontier``, the sizes summed over the calls, beside at most the
    same amount from the point before it, which the Run already holds when only one
    point remains. That overcounts by at most one frontier when the arguments of the
    two points coincide. Otherwise every point is visited, at one selection and
    counting pass each, until what is kept exceeds the limit. The caller's refusal
    then follows with the known total so far. Nothing is built.

    On a backend that binds templates (``_groups_points``), the experiments it binds
    count one circuit and ``_TEMPLATE_ROW_BYTES`` per point, as a first coarse check,
    because the number of templates is known only once the Method has grouped them.
    ``prepare_template_rows`` compares the actual templates before their charge.
    Rows of a template that the backend cannot bind stay under the check of each
    point's own preparation, though they count as rows here. Without ``whole``,
    ``workload`` is empty and no point is counted unless the Plan has direct
    preparations, whose trees are checked for every resolved point.
    """
    from nwqlib.blocks.selection import SelectedBlock, direct_preparation
    items = run._state["workflow_items"]
    prepared = Counter(item.realization.experiment for item in items.values() if item.prepared_id is not None)
    templated = whole and _groups_points(plan, run)
    template_kinds = getattr(run.backend, "template_readouts", None)
    circuits = shots = kept = temporary = calls = definitions = previous_frontier = 0
    held, first_call, static, bound_experiments = None, None, {}, set()
    direct = any(isinstance(block, SelectedBlock) and direct_preparation(block) is not None for block in plan.blocks)
    already = _held_bytes(run) if whole else 0
    limit = memory_limit(getattr(plan.method, "max_bytes", None)) if whole else None
    for experiment, point_shots in fixed:
        kind = None if experiment.batch is None else plan._program_nodes[experiment.batch].observation_kind
        uniform = (point_shots is not None and kind in {"counts", "pauli_expectation", "probabilities"}
                   and (not whole or _fixes_footprint(plan, experiment)))
        binds = templated and experiment.batch is not None and (template_kinds is None or kind in template_kinds)
        points = plan._static_points((experiment,))
        for point in islice(points, 1) if uniform else points:
            construction = None
            if held is None:
                # The first point is prepared right after this loop, so its
                # Experiment and construction are kept for that preparation.
                held = plan._selected_construction(experiment.name, point.bindings)
                construction = held[1]
                if held[0].batch is None:
                    spec = held[0].observation
                else:
                    admission = _Admission(held[1].program)
                    admission.admitted().require_ready()
                    _, spec = held[0]._resolved_observation(admission, construction_id=held[1].content_id)
            else:
                _, spec = plan.resolve(experiment.name, bindings=point.bindings).resolved_observation(plan)
            if spec.kind == "host_scalars":
                continue
            label = experiment.name if uniform else point.key
            try:
                run.backend.target_for(spec)
            except ValueError as error:
                # A counts readout names its shots per program. Any other readout
                # kind names the kind, which is what the target refuses.
                request = (f"{spec.shots} shots per program" if spec.kind == "counts"
                           else f"readout kind {spec.kind!r}")
                raise ValueError(f"Experiment {label!r} requests {request}: {error}") from error
            count = plan._point_count((experiment,)) if uniform else 1
            circuits += count
            shots += count * spec.shots
            remaining = (count - prepared[experiment.name] if uniform else
                         int(getattr(items.get(point.key), "prepared_id", None) is None))
            if direct or (whole and remaining):
                # The trees of the Plan's direct preparations are checked for
                # every resolved point, as its preparation will check them, so
                # a later setting is not refused after earlier ones were paid for.
                if construction is None:
                    construction = plan._selected_construction(experiment.name, point.bindings)[1]
                try:
                    admission = _Admission(construction.program)
                    admission.admitted().require_ready()
                    counted, trees = _admit_direct_preparations(run, construction, admission, spec)
                except ValueError as error:
                    raise ValueError(f"Experiment {label!r}: {error}") from None
                if whole and remaining:
                    memory = _circuit_memory(run, counted, trees)
                    if binds:
                        kept += remaining * _TEMPLATE_ROW_BYTES
                        if experiment.name not in bound_experiments:
                            kept += memory.kept
                            calls += memory.calls
                            bound_experiments.add(experiment.name)
                    else:
                        kept += remaining * memory.kept
                        calls += remaining * memory.calls
                    temporary = max(temporary, memory.temporary + previous_frontier)
                    if uniform and count > 1:
                        temporary = max(temporary, memory.logical + min(remaining, 2) * memory.frontier)
                    previous_frontier = memory.frontier
                    static.update(memory.static)
                    first_call = first_call or memory.first_call
                    definitions = max(definitions, memory.definitions)
                    if exceeds(already + kept, limit):
                        return circuits, shots, held, _CircuitMemory(kept, temporary, static, calls, first_call, definitions)
    return circuits, shots, held, _CircuitMemory(kept, temporary, static, calls, first_call, definitions)


def _admit_static_workload(plan, run):
    """Check the known work of a static Plan's remaining points before this Run object adds any.

    Paid work is counted when each attempt reserves it, so without this check a
    Run reached without ``prepare_static``, or a Plan whose later points do not
    fit, would spend its budget on the first points before a refusal. Each check
    uses the same readout walk as ``prepare_static``. Without any attempt yet,
    the circuits and shots of the whole Plan go against the cumulative limits.
    Each point acquires its circuit once, so attempts need no further count.
    Preparations do, whether or not attempts exist, because a charge without a
    record stays counted and its row needs another: the circuit preparations counted
    so far plus one for each circuit point without a record or remote preparation go
    against ``max_total_circuits``, which caps preparations and execution attempts
    independently, not their sum, and leaves host kernels out of both. Without a
    charge left over that total is the
    Plan's own count, so this refuses only a Run that an explicit rebuild would
    otherwise take through paid jobs to a refusal at its last point. Data limits
    stay incremental: what results will
    save is not known in advance, and adding their upper bounds would refuse Plans
    that fit. The memory of the points not yet prepared goes against the Method's
    limit whether or not attempts exist. The check runs once per open Run object
    unless it refuses, so a retry after ``Run.extend_limits`` checks again, and
    ``_rebuild_static_preparations`` runs it again for the limits and preparations
    of its call. The
    counts that the axis bounds give go first, because the walk of a point-dependent
    experiment (``_static_points_workload``) costs a pass per point.
    """
    fixed = [(experiment, plan._point_shots(experiment)) for experiment in plan.experiments]
    if not run._state["events"]:
        run.check_capacity(circuits=plan._point_count(tuple(experiment for experiment, _ in fixed
                                                            if experiment.batch is not None)),
                           shots=sum(plan._point_count((experiment,)) * shots for experiment, shots in fixed
                                     if shots is not None))
    circuits, shots, _, workload = _static_points_workload(plan, run, fixed, whole=True)
    if not run._state["events"]:
        run.check_capacity(circuits=circuits, shots=shots)
    _admit_circuit_memory(run, workload, "running the remaining circuits of this Plan", whole=True)
    # The walk above returned early only for the memory refusal, so circuits is complete here.
    state = run._state
    charges, remote = state["preparation_charges"], state["remote_preparations"]
    saved = sum(1 for item in state["workflow_items"].values()
                if (item.prepared_id is not None or item.preparation_id in remote)
                and charges[item.preparation_id].execution == "quantum_circuit")
    run.check_capacity(preparations=circuits - saved)
    state["static_workload_admitted"] = True


def prepare_static(plan, *, run, settings="first"):
    """Check all points of a static Plan, then prepare its first item or every item.

    The items are the Plan's points (`Plan._static_points`): each experiment,
    or for an experiment with range axes each point of its axes, under the
    point's key. Before any point is generated, the circuits of the batch
    experiments are counted from their axis bounds (`Plan._point_count`),
    one for each point, and a count beyond ``max_total_circuits`` refuses at
    once. The shots of each batch experiment whose Program fixes the shots
    of every point (`Plan._point_shots`), such as a counts batch with
    constant repetitions, are counted the same way against
    ``max_total_shots``. A direct experiment, whose readout the Method can
    specialize, and shots that depend on the point are counted once the
    point is resolved, so with either the early counts are lower bounds and
    a refusal says so. The readouts are then checked against the backend
    target. A counts, Pauli expectation or probability batch whose Program
    fixes the shots of every point gives each point the same shots and a
    readout kind that an adapter's ``target_for`` treats alike, and
    ``target_for`` reads only these fields for these kinds
    (docs/development/execution.md, Backend adapter contract).
    Only its first point is resolved and checked for the whole experiment,
    and its points are counted from the axis bounds. Every other point,
    whose shots, estimate or trajectory positions can depend on the point,
    is resolved and checked one at a time. Each point's own Program checks
    run when that point is prepared, before its charge. The check covers
    the circuits, preparations and shots of the whole Plan against the
    Run's cumulative limits, whichever items are prepared now. ``settings="all"``
    therefore needs no limit of its own. It prepares every point in Plan
    order through ``_prepare_static_item`` and submits nothing. That function
    returns an item's existing preparation when there is one, so each item is
    prepared and counted once, and a Run with a folder records it in its run
    log. ``execute_static`` later finds the preparation record in the item's
    workflow row and submits the saved preparation. A preparation that fails
    or is interrupted stops the loop. If it stopped after its charge, the
    charge stays on the item's workflow row without a preparation record
    (``_unbuilt``), and ``execute_static`` raises
    ``PreparationNotRebuilt`` before it submits anything, unless the caller asks
    for ``_rebuild_static_preparations``. A check that fails
    before the charge counts nothing, so the item can then be prepared again.
    Each workflow row draws its runtime seed when it is created, which happens
    in Plan order in both modes, so both modes prepare with the same seeds.
    Through ``scientist.prepare``, ``"all"`` reaches this function only for
    ``LOCAL_SIMULATORS`` and host execution. ``"all"`` also compares the memory
    of every point that is not yet prepared with the Method's limit before the
    first preparation (``_static_points_workload``), and ``"first"`` leaves that
    to the first static execution, so a first circuit can be inspected without
    the rest of the Plan fitting in memory.
    A target refusal during this check names the point's key, or the
    experiment when one point stands for it, and, for a counts readout, the shots per program that it requests, the pooled
    product for a pooled query, or, for any other readout, its readout kind,
    before any preparation or submission. The first point's Experiment
    and construction, resolved during the check, are passed to its
    preparation, so that point is not resolved again.
    When the Method can group the Plan's points on this backend
    (``_groups_points``), the selected points are held together, the
    points of the Method's templates that the backend
    binds are prepared as bindings (``_row_templates``,
    ``prepare_template_rows``), each template compiled once, and the other
    points are prepared on their own, in Plan order. Otherwise each point
    is prepared as it is generated. The axis-bound refusal of
    ``max_total_circuits`` bounds this scan whenever that limit is set.
    """
    if run.plan is not plan:
        raise ValueError("static preparation requires the same live Plan")
    state, limits = run._state, run.limits
    # Circuit preparations and acquisition attempts are each capped by
    # max_total_circuits, so the larger of the two counts decides below.
    counted = max(state["preparations"] - state["host_preparations"],
                  len(state["events"]) - state["host_invocations"])
    # Each point of a batch experiment is one circuit, so the axis bounds give
    # those circuits before any point is generated or resolved, and their
    # shots when the Program fixes the shots of every point
    # (Plan._point_shots). A direct experiment is one point whose readout the
    # Method can specialize, so it is counted only once resolved, below, and
    # so are point-dependent shots. Until then these counts are lower bounds.
    batches = tuple(experiment for experiment in plan.experiments if experiment.batch is not None)
    total = plan._point_count(batches)
    if exceeds(counted + total, limits.max_total_circuits):
        if len(batches) == len(plan.experiments):
            run.check_capacity(preparations=total, circuits=total)
        raise _cumulative_limit_error("max_total_circuits", limits.max_total_circuits, "circuits", counted, total,
                                      at_least=True)
    fixed = [(experiment, plan._point_shots(experiment)) for experiment in plan.experiments]
    known = sum(plan._point_count((experiment,)) * shots for experiment, shots in fixed if shots is not None)
    if exceeds(state["shots"] + known, limits.max_total_shots):
        raise _cumulative_limit_error("max_total_shots", limits.max_total_shots, "raw shots", state["shots"], known,
                                      at_least=any(shots is None for _, shots in fixed))
    circuits, shots, held, workload = _static_points_workload(plan, run, fixed, whole=settings == "all")
    # The capacity check and the first preparation follow the whole loop.
    run.check_capacity(preparations=circuits, circuits=circuits, shots=shots)
    if settings == "all":
        _admit_circuit_memory(run, workload, "preparing every circuit of this Plan", whole=True)
    selected = plan._static_points() if settings == "all" else islice(plan._static_points(), 1)
    if _groups_points(plan, run):
        # A backend that binds parameterized circuits prepares the selected
        # points, held together for the Method to group, as bindings of their
        # templates, each compiled once. The loop below returns those
        # preparations and prepares the other points on their own.
        selected = tuple(selected)
        templates = _row_templates(plan, run, selected)
        if templates is not None:
            prepare_template_rows(plan, run=run, points=selected, templates=templates)
    for index, point in enumerate(selected):
        _prepare_static_item(plan, run, point, held=held if index == 0 else None)


def execute_static(plan, *, run):
    """Run the fixed points of a static Plan in order. Completed attempts are never repeated.

    The points (`Plan._static_points`) are the experiments, and each range-axis
    point of an experiment with axes, and they run in Plan order, one attempt
    each, generated one at a time. The loop returns None at the
    first point that cannot contribute an observation yet, because its
    attempt has not completed, its preparation is still compiling remotely or its
    detached submission is still running. After cancellation it returns None
    before new work. An existing attempt is never submitted again. Analysis runs
    once every point has a collected observation.

    Before the first new work of this Run object, the known circuits, shots and
    memory of the remaining points are checked (``_admit_static_workload``).

    An item with a local preparation charge but no preparation record
    (``_unbuilt``) means that the Plan cannot complete in this Run unless the
    caller asks for a rebuild, which ``Run.resume`` does before this function
    (``_rebuild_static_preparations``). The loop
    raises ``PreparationNotRebuilt`` before it submits any new experiment,
    whose work would otherwise be counted without a Result. After
    ``prepare(plan, settings="all")`` such an item can come before items that
    were never submitted. With the default ``settings="first"`` only the last
    item created can be one, so there the check raises the error that
    ``_prepare_static_item`` would raise for that item.

    When the Method can group the Plan's points on a backend that binds
    parameterized circuits (``_groups_points``), the first point without
    an attempt and every later one without an attempt are
    held together, and the points of the templates that the backend binds
    (``_row_templates``) are prepared as template bindings, each keeping its
    own workflow row, attempt and shots. A detached backend sends them in
    one job (``submit_detached``), and the loop waits for that job. A
    synchronous backend (Aer) runs each bound point as its own attempt, in
    Plan order, as for any other point. The other held points are prepared
    on their own. Points with attempts are never part of it. The
    remaining points are held together once, at the first point
    without an attempt, and the Method is asked once per call.
    """
    if run.plan is not plan:
        raise ValueError("static execution requires the same live Plan")
    unbuilt = next((key for key, item in run._state["workflow_items"].items() if _unbuilt(run, item)), None)
    points = plan._static_points()
    group = _groups_points(plan, run)
    while (point := next(points, None)) is not None:
        key = point.key
        item = run._state["workflow_items"].get(key)
        if item is not None and item.attempt is not None:
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
        if not run._state["static_workload_admitted"]:
            _admit_static_workload(plan, run)
        if group:
            # When the Method can group the Plan's points here, this and
            # every later point without an attempt are held together, and the
            # points of the templates that it binds are prepared as bindings
            # (prepare_template_rows). A detached backend sends those in one
            # job, and the loop waits for that job. A synchronous one (Aer)
            # runs them below, one attempt each in Plan order. The Method is
            # asked once, and every other point is prepared on its own.
            group = False
            rest = tuple(points)
            items = run._state["workflow_items"]
            pending = tuple(later for later in (point, *rest)
                            if getattr(items.get(later.key), "attempt", None) is None)
            templates = _row_templates(plan, run, pending)
            if templates is not None:
                handles = prepare_template_rows(plan, run=run, points=pending, templates=templates)
                if handles and callable(getattr(run.backend, "launch", None)):
                    run._state["workflow_current"] = tuple(handles)
                    try:
                        submit_detached(tuple(handles.values()), run=run)
                    finally:
                        run._state["workflow_current"] = None
                    return None
            points = iter(rest)
        prepared = _prepare_static_item(plan, run, point)
        if isinstance(prepared, PendingPreparation):
            return None
        run._state["workflow_current"] = key
        try:
            # A synchronous outcome commits with its collection ordinals.
            chunk = submit_experiment(prepared, run=run, collect=True)
        finally:
            run._state["workflow_current"] = None
        chunks = chunk if isinstance(chunk, tuple) else (chunk,)
        if not all(isinstance(item, ObservationChunk) for item in chunks):
            return None
        # Reported once the submission returns, so an interruption in the
        # progress observer cannot reach its error handling.
        collected = len(run._state["chunks"])
        for done in range(collected - len(chunks) + 1, collected + 1):
            run.progress("collect", done, None)
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
    """Save the preparation record of a classical routine. Numerical work starts only at submission.

    The record binds the chosen kernel declaration, the point and the installed
    versions of the kernel's declared dependencies. It carries no quantum layout
    or lowering, and its population is unconditional, as ``PreparedArtifact``
    requires for host work. The record and the revised charge are commit 3 of
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


def _submit_host(prepared, *, run, checkpoint_sequence=None, collect=False):
    """Reserve one bound host invocation and atomically save its declared scalar/array
    outputs.

    The reserved event commits before the kernel runs (commit 4 of the module
    docstring). There is no submission record, because nothing leaves the
    process. The chunk, the completed event and the kernel's arrays then commit
    together (commit 6), with its collection ordinal when ``collect`` is
    True. The kernel must return exactly its declared scalars in
    their declared frames, and each declared array or an explicit reason that it
    is unavailable. An ordinary exception raised by the kernel, or by these
    output checks after it returns, makes the attempt failed. An interruption
    such as ``KeyboardInterrupt`` leaves it uncertain (``Run._fail_attempt``).
    """
    run.plan.method.before_submit(run.plan, (prepared,), run=run)
    kernel, native = prepared._native.record, prepared._native
    if native._realization_id is not None and native._realization_id != prepared.realization.content_id:
        raise ValueError("host closure differs from its prepared point")
    expected_bytes = readout_bytes(prepared, max_bytes=run.limits.max_data_bytes)
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
        event = event.revise(status="completed", finished=_now(), observation_id=chunk.content_id)
        run._finish_attempt(index, event, chunk, collect=collect)
        return chunk
    except BaseException as error:
        run._fail_attempt(index, event, error)
        raise


def _publish_amplitudes(prepared, result, *, run, attempt, result_key="0", point=None, statevector=None,
                        boundary=None, trajectory_id=None):
    """Reduce a native statevector to its declared amplitudes and save them.

    Both the prepared native object and the result must describe the state
    before the final measurements. Any other trajectory would be a conditioned
    state. ``reduce_amplitudes`` selects the declared coordinates, physical scale
    and branch masses, and the array is saved with this acquisition's
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
        # reduce_amplitudes returns a new array with finite entries that nothing
        # else holds, so publication neither copies nor scans it again.
        handle = run.artifacts._publish(array, output=declaration.output, private=True, finite=True, provenance=dict(
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
    """Rebuild the handle of a saved preparation from its preparation record, without lowering.

    A handle used only to retrieve results (``for_submission=False``) reads
    no saved native payload, so it never decodes the original native circuit.
    The preparation record must still match the Plan's chosen construction and
    readout and the Run's backend configuration. Such a handle is enough to
    decode results of a known job. A handle for new execution also needs the
    saved native payload, which a Run with a folder saved with the
    preparation record.

    A handle restored here adds nothing to ``native_held``, and its payload is not
    compared with ``max_bytes`` before it is read. Only Aer keeps a memory
    estimate with its saved circuit, which ``PreparedHandle._restore_native``
    compares before the file is read. ``_held_bytes`` says what this leaves out on
    other backends.
    """
    record = run.prepared_artifact(prepared_id)
    experiment, construction = record.realization._selected_construction(run.plan)
    admission = _Admission(construction.program)
    admission.admitted().require_ready()
    setting, observation = experiment._resolved_observation(admission, construction_id=construction.content_id)
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
            # The records of one template share its saved circuit, which is
            # read once per open Run.
            shared = run._state.setdefault("template_payloads", {}) if record.template is not None else {}
            payload = shared.get(record.payload.content_id) or run._state["journal"].read_payload(record.payload)
            if record.template is not None:
                shared[record.payload.content_id] = payload
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
    folder, the backend then starts the job, and the job's locator (a
    `JobLocator`, the provider's identifier of the job) is saved when the
    backend returns it. If the process ends between the start and
    the saved locator, the reopened Run has an uncertain submission without
    a locator, and only the backend's own lookup by the original submission
    can find the job. An error while starting the job marks the submission
    and its attempts uncertain, with their work still counted, because the
    provider may have accepted the request before the error reached this
    process. A request that the backend can refuse from its saved bytes and
    configuration is refused before the submission is saved, with nothing
    counted, so `Run.extend_limits` can lead past it. Each circuit is one
    attempt with its own shots, also when the
    backend sends several circuits as rows of one parameterized unit, and
    the submission records each circuit's position in the job before it
    starts. The job is never resubmitted or replaced. Read its results with
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
    # The Method's ``before_submit`` runs and the backend admits the whole
    # batch, by ``admit_batch`` and by ``admit_submission`` for what depends on
    # the Run's limits, before the intent commits (commit 4 of the module docstring). Then
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
    # The backend admits the batch before the intent, after the Method's
    # check, and launch sends these same native objects.
    run.plan.method.before_submit(run.plan, prepared, run=run)
    natives = tuple(handle._native for handle in prepared)
    run.backend.admit_batch(natives)
    admit_submission = getattr(run.backend, "admit_submission", None)
    if callable(admit_submission):
        admit_submission(natives, run=run)
    # An adapter that places several items in one provider unit names each
    # item's position (result_keys), which the intent saves before launch.
    result_keys = getattr(run.backend, "result_keys", None)
    keys = result_keys(natives) if callable(result_keys) else tuple(str(index) for index in range(len(natives)))
    identity, started = str(uuid4()), _now()
    events, items = [], []
    for index, handle in enumerate(prepared):
        observation, attempt = handle.record.observation, str(uuid4())
        events.append(ConsumptionEvent(attempt=attempt, prepared_id=handle.record.content_id, submission=identity,
            status="reserved", started=started, shots=observation.shots, evaluations=int(observation.kind != "counts"),
            data_bytes_reserved=readout_bytes(handle, max_bytes=run.limits.max_data_bytes, run=run, attempt=attempt,
                                              result_key=keys[index]),
            provider_managed_sampling=observation.kind == "estimated_observable",
            checkpoint_sequence=None if checkpoint_sequences is None else checkpoint_sequences[index]))
        items.append(SubmissionItem(attempt=attempt, prepared_id=handle.record.content_id, item=index,
                                    result_key=keys[index]))
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
            # The exception's type keeps the text nonempty for KeyboardInterrupt()
            # and TimeoutError(), which have no message of their own.
            value = previous.revise(status="uncertain", failure=f"{type(error).__name__}: {error}")
            failed = [(index, event.revise(status="uncertain", failure=value.failure)) for index, event in installed]
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
    originally accepted acquisition.
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
    the same job again. After `Run.cancel`, each job is sent its cancellation
    before it is read, and an error from either step does not keep the other
    jobs from theirs. A failed request does not stop the read of its own job, so
    what the read finds is saved first, including the result of a job that
    had already ended. The first error is raised after all the jobs. When the
    request and the read of one job both fail, the request's error is raised
    and names the read's failure, and the submission records both.

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
    # submission (``_record_refresh_failure``). The error of a failed request
    # waits until both transactions have run, so a provider that refuses to
    # cancel a finished job cannot keep that job's result from being saved.
    with run._lock:
        run._ensure_open()
        observed, first_error = [], None
        for submission_id in tuple(run._state["pending_submissions"]):
            submission = run._state["submissions"][submission_id]
            pending = tuple(item for item in submission.items if item.attempt not in run._state["by_attempt"])
            if not pending or submission.results_consumed:
                run._state["pending_submissions"].pop(submission_id, None)
                continue
            if submission.locator is None and not callable(getattr(run.backend, "reconcile", None)):
                continue  # Method may still publish a lawful partial Result before recovery fails.
            run._backend_notice()
            held = None
            try:
                handles, natives = _refresh_natives(run, submission, pending)
                if submission.locator is None:
                    locator = run.backend.reconcile(submission, natives, run=run)
                    if locator is None:
                        continue
                    submission = run._ack_submission(submission_id, locator)
                submission, held = _request_cancellation(run, submission_id, submission)
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
                if held is not None:
                    raise held
            except Exception as error:
                surfaced = _record_refresh_failure(run, submission_id, error, held)
                if run.cancel_requested is None:
                    raise
                first_error = first_error or surfaced
        if first_error is not None:
            raise first_error
        return tuple(observed)


def _refresh_natives(run, submission, pending):
    """Restore what one refresh of ``submission`` needs from its saved preparation records.

    Returns ``(handles, natives)``. ``handles`` maps the attempt of each
    pending item (one without an observation) to its restored
    ``PreparedHandle``, which decodes that item's result. ``natives`` holds the
    backend's native object for every item in original ordinal order, and an
    item that already has an observation is restored from its preparation
    record alone. The adapter receives the complete association because a
    provider numbers batch items by their original ordinals, even after some
    results have been saved.

    A handle of the open Run that still holds its native object serves directly. Otherwise
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
    """Forward a Run cancellation to one unfinished job at most once, and return the submission and the request's error.

    The request is written on the submission before the backend call, so a
    process that stops during the call reopens with the request recorded and
    never sends it a second time. A call that raises an ordinary ``Exception``
    did not deliver a request that this process can count on, so only this
    submission's flag returns to unset, the Run's own request stays, and a later
    refresh or ``Run.cancel`` sends it again. That error is returned, or None
    without one, and ``refresh_submissions`` raises it after reading the job,
    because the provider may be refusing a job that already ended. A failed
    write to the run log still raises here, and no read follows it. Only a later
    refresh reports whether the job was cancelled, and the reserved work stays
    counted. A job recorded as ended receives no request. That includes one
    that ended without a result (``terminal_without_result``), so a refused
    request cannot hide the recovery failure that ``Run.wait`` reports for it.
    """
    held = None
    if (run.cancel_requested is not None and submission.cancel_requested is None
            and submission.status not in {"completed", "failed", "cancelled"}
            and not submission.terminal_without_result):
        submission = submission.revise(cancel_requested=run.cancel_requested)
        run._write((("submission", submission_id, submission),))
        run._state["submissions"][submission_id] = submission
        cancel = getattr(run.backend, "cancel", None)
        if callable(cancel):
            try:
                cancel(submission.locator, run=run)
            except Exception as error:
                submission = submission.revise(cancel_requested=None)
                run._write((("submission", submission_id, submission),))
                run._state["submissions"][submission_id] = submission
                held = error
        else:
            run._state["warnings"].append("backend has no cancellation operation; original job remains observable")
    return submission, held


def _commit_refresh_status(run, submission_id, submission, update):
    """Commit the provider's status and item associations, the first transaction of a refresh.

    Returns ``(submission, revised)``. ``submission`` has the child job and
    result identifiers reported in ``update.associations`` merged into its
    items (``_merged_associations``). ``revised`` is that submission with the
    reported status, which this function has just written.

    A poll that changes nothing is not written. When the status, provider
    status, failure, native simulation count and item associations equal the
    saved ones and no refresh failure is recorded, ``revised`` is the
    saved submission and no row is committed. A recorded refresh failure is
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
        finished=finished, terminal_without_result=update.terminal_without_result)
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
    observation is skipped, so a job read twice saves each observation
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

    This is the second transaction of a refresh. In-memory counters and
    indexes change only after it commits. Returns the newly saved chunks. With no
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


def _record_refresh_failure(run, submission_id, error, held):
    """Record a failed refresh on its submission and return the error that the caller raises.

    ``held`` is the error of this refresh's failed cancellation request, or
    None (``_surfaced_failure``). The record is written after the refresh's own
    commits, because a good status clears the previous one
    (``_commit_refresh_status``) and this is the only saved trace of a failed
    request. An unfinished array batch is discarded first. After a failed write to
    the run log nothing more is written, because the Run must be reopened
    from its folder. The original job locator stays saved. When the
    submission has a locator, a note added to the returned error names that
    saved job, says that no replacement job was submitted and gives the next step.
    """
    from nwqlib._remote_preparation import _surfaced_failure
    run._abort_publications()
    surfaced, text = _surfaced_failure(error, held)
    if run._state["storage_failed"]:
        return surfaced
    previous = run._state["submissions"][submission_id]
    interrupted = previous.revise(refresh_failure=text)
    run._write((("submission", submission_id, interrupted),))
    run._state["submissions"][submission_id] = interrupted
    if previous.locator is not None:
        next_step = ("Its cancellation request failed. While the job has not ended, a later Run.cancel or "
                     "refresh_submissions on the same run sends the request again and reads the job. Once the job "
                     "is recorded as ended, no further request is sent. Run.wait returns the Result when the saved "
                     "outcomes complete it. Otherwise a cancelled Run starts no new work and Run.wait raises "
                     "RuntimeError naming its folder."
                     if held is not None else
                     "Resolve the retrieval failure, then call refresh_submissions on the same run.")
        surfaced.add_note(f"Original {previous.locator.provider} job {previous.locator.job_id} is stored; "
                          f"no replacement job was submitted. {next_step}")
    return surfaced
