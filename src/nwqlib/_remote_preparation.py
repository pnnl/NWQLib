"""The saved progress of one remote preparation, from local data through upload to compilation, kept by the Run.

Only a new stage can create a remote resource. An interrupted intent is
reconciled by its original preparation identity, which the backend records on
the remote resource it creates, and a known resource is never submitted again.

States of one ``PendingPreparation``::

    local -> upload_intent -> uploaded -> compile_intent -> compile_pending
          -> completed | failed | cancelled

Each ``*_intent`` state commits before the SDK call that may create the remote
resource. The backend adapter passes the new upload or compile identity to an
acknowledgement callback as soon as the SDK returns it, and the callback commits
it before the adapter does any further work that could fail. After a crash, an
intent without an acknowledged identity is resolved only by
``reconcile_preparation`` with the original preparation identity. When that finds
nothing, the state becomes ``upload_uncertain`` or ``compile_uncertain`` and is
reconciled again later. It is never uploaded or compiled a second time.
"""

from uuid import uuid4

from nwqlib.execution import PendingPreparation, PayloadRef


def _save(run, pending, **fields):
    """Commit one revision of the pending record, then put it into the Run's in-memory state.

    Every revision between the initial ``local`` record (``begin_preparation``)
    and completion (``_finish_native_preparation``) goes through this write.
    The initial record, each revision and the completion commit before the
    in-memory state changes, so that state never runs ahead of the run log
    (``run.sqlite``, the SQLite file in which the Run records each step).
    """
    revised = pending.revise(**fields)
    run._write((("remote_preparation", revised.preparation_id, revised),))
    run._state["remote_preparations"][revised.preparation_id] = revised
    if revised.status in ("failed", "cancelled") and revised.preparation_id in run._state["receipt_commits"]:
        from nwqlib._prepared_execution import _withdraw_preparation
        _withdraw_preparation(run._state, revised.preparation_id,
                              run._state["receipt_commits"][revised.preparation_id][1])
    return revised


def _surfaced_failure(error, held):
    """Return the error to raise for a failed refresh of one job and the text to save for it.

    ``held`` is the error of a cancellation request that failed earlier in the
    same refresh, or None. The request went first, so its error is the one
    raised. When the rest of the refresh then raised a different ``error``, that
    error stays on it as a note and as its ``__context__``, and the saved text
    names both.
    """
    text = f"{type(error).__name__}: {error}"
    if held is None or held is error:
        return error, text
    held.__context__ = error
    held.add_note(f"Refreshing the same job then failed: {text}")
    return held, f"{type(held).__name__}: {held}; refreshing the same job then failed: {text}"


def begin_preparation(data, *, run, reservation, fields, items, setting, bindings, synthesis=0):
    """Save the local preparation data and a ``local`` record, then start the upload.

    The payload, the pending record and the charge with ``synthesis``, the
    exact dense synthesis work of the local lowering
    (``Run._record_synthesis``), commit together before any provider call,
    so a restart can continue this exact preparation from its saved bytes
    and never needs to lower the circuit again. Completion keeps that work
    on the charge.
    """
    from nwqlib._run_journal import encode
    values = data.model_dump(mode="json")
    payload = encode(values, run.limits.max_data_bytes).encode("utf-8")
    reference = PayloadRef(format=f"{run.backend.kind}.preparation/1", key=str(uuid4()), bytes=len(payload))
    charge = reservation.revise(synthesis_work=reservation.synthesis_work + synthesis)
    saved = {name: fields[name] for name in ("realization", "runtime", "snapshot", "observation", "logical",
                                            "quantum_layout", "classical_layout", "construction_id")}
    pending = PendingPreparation(**saved, preparation_id=charge.preparation_id,
        run_id=run.run_id, local_data=reference, items=items, setting=setting, bindings=bindings,
        workflow_item=run._state["workflow_current"])
    records = (("remote_preparation", pending.preparation_id, pending), ("payload", reference.content_id, reference))
    if synthesis:
        records += (("preparation", charge.preparation_id, charge),)
    run._write(records, payloads=((reference, payload),))
    run._state["remote_preparations"][pending.preparation_id] = pending
    run._state["preparation_charges"][charge.preparation_id] = charge
    run._state["synthesis_work"] += synthesis
    return _advance(pending, run, data=data)


def refresh_preparation(preparation_id, *, run):
    """Refresh one original preparation once and return its prepared circuit or its saved pending state.

    No circuit is built or run here. A finished compilation reads its local
    data only when it builds its preparation record.
    """
    from nwqlib._prepared_execution import restore_prepared
    with run._lock:
        run._ensure_open()
        reservation = run._state["preparation_charges"].get(preparation_id)
        if reservation is None:
            raise ValueError("preparation ID does not belong to this run")
        if reservation.prepared_id is not None:
            return restore_prepared(reservation.prepared_id, run=run, for_submission=True)
        pending = run._state["remote_preparations"].get(preparation_id)
        if pending is None:
            raise ValueError("preparation interrupted before durable local data; automatic reconstruction is forbidden")
        run._backend_notice()
        return _advance(pending, run)


def _advance(pending, run, *, data=None):
    """Move one preparation forward without waiting and return its new state or handle.

    Local data are read from the run log only when a stage needs them. Initial
    submission never waits for compilation. Once compilation is pending, each
    call makes one ``refresh_compile`` and either saves the provider status or
    saves the finished preparation record through ``_finish_native_preparation``,
    joined to the original static workflow item. Cancellation marks a ``local``
    or ``uploaded`` preparation as cancelled without a provider call, and a
    compile already running receives one cancellation request before it is
    read. A request that fails is recorded as not sent, so a later call sends it
    again until the job is recorded as ended. The same call still reads the job
    and saves what it finds before raising the request's error, because the
    provider may be refusing a job that already ended, whose record or failure
    must not wait for a request that cannot succeed. An upload or
    a compilation is new remote work, so it starts only while the Run's
    counted data are within ``max_data_bytes`` (``Run.check_capacity``). A
    refused stage stays ``local`` or ``uploaded`` with its identities, and
    after ``Run.extend_limits`` the next call starts it. Reconciling,
    polling, acknowledging and cancelling a stage that has already started
    record facts about it and are never refused for the data limit. Unless a
    write to the run log has already failed, any error keeps every acknowledged
    identity and turns an unacknowledged intent into its uncertain state, and a
    preparation that has ended keeps its saved state and failure text. The
    error then propagates.
    """
    backend, identity = run.backend, pending.preparation_id
    held = None

    def load_data():
        """Read and decode the saved local preparation data once, on first need."""
        nonlocal data
        if data is None:
            payload = run._state["journal"].read_payload(pending.local_data)
            data = backend.restore_preparation(payload, run=run)
        return data

    try:
        if pending.status in {"completed", "failed", "cancelled"}:
            return pending
        if run.cancel_requested is not None and pending.status in {"local", "uploaded"}:
            return _save(run, pending, status="cancelled", cancel_requested=run.cancel_requested)
        if pending.status == "local":
            # An upload is new remote work, refused while the Run's counted
            # data exceed max_data_bytes (Run._write).
            run.check_capacity()
            # Save the possible upload before the public SDK can create it.
            pending = _save(run, pending, status="upload_intent")
            def upload_ack(program_id):
                """Commit the upload identity as soon as the SDK returns it.

                After this commit a crash leaves ``uploaded`` with a known
                identity, and the upload is never repeated.
                """
                nonlocal pending
                pending = _save(run, pending, status="uploaded", upload_id=program_id, failure=None)
            reference = backend.upload(load_data(), preparation_id=identity,
                run=run, acknowledge=upload_ack)
            pending = _save(run, pending, status="uploaded", input_ref_json=reference, failure=None)
        elif pending.status in {"upload_intent", "upload_uncertain"}:
            reference = backend.reconcile_preparation(identity, "upload", run=run)
            if reference is None:
                return _save(run, pending, status="upload_uncertain")
            pending = _save(run, pending, status="uploaded", input_ref_json=reference, failure=None)
        if pending.status == "uploaded":
            if run.cancel_requested is not None:
                return _save(run, pending, status="cancelled", cancel_requested=run.cancel_requested)
            if pending.input_ref_json is None:
                # An acknowledgement survived, but later SDK serialization failed.
                reference = backend.restore_upload(pending.upload_id, run=run)
                pending = _save(run, pending, input_ref_json=reference, failure=None)
            # A compilation is new remote work, refused like an upload.
            run.check_capacity()
            pending = _save(run, pending, status="compile_intent")
            def compile_ack(locator):
                """Commit the compile job locator as soon as the SDK returns it.

                After this commit a crash leaves ``compile_pending`` with the
                locator, so the next refresh reads that job instead of compiling again.
                """
                nonlocal pending
                pending = _save(run, pending, status="compile_pending", locator=locator, failure=None)
            locator = backend.start_compile(pending.input_ref_json, preparation_id=identity,
                run=run, acknowledge=compile_ack)
            if pending.locator is None:
                pending = _save(run, pending, status="compile_pending", locator=locator, failure=None)
            elif pending.locator != locator:
                raise ValueError("compile return differs from its acknowledged job")
            return pending  # Initial submission never waits for compilation.
        if pending.status in {"compile_intent", "compile_uncertain"}:
            locator = backend.reconcile_preparation(identity, "compile", run=run,
                input_ref_json=pending.input_ref_json)
            if locator is None:
                return _save(run, pending, status="compile_uncertain")
            pending = _save(run, pending, status="compile_pending", locator=locator, failure=None)
        if pending.status == "compile_pending":
            if run.cancel_requested is not None and pending.cancel_requested is None:
                pending = _save(run, pending, cancel_requested=run.cancel_requested)
                try:
                    backend.cancel_preparation(pending.locator, run=run)
                except Exception as error:
                    # Not delivered as far as this process can tell, so a later
                    # refresh sends it again. A stopped process stays recorded.
                    pending = _save(run, pending, cancel_requested=None)
                    held = error
            update = backend.refresh_compile(pending.locator, pending.input_ref_json, load_data,
                run=run)
            from nwqlib.backends.connection import NativePreparation
            if not isinstance(update, NativePreparation):
                status = update.status if update.status in {"failed", "cancelled"} else "compile_pending"
                pending = _save(run, pending, status=status, provider_status=update.provider_status, failure=update.failure)
                if held is not None:
                    raise held
                return pending
            from nwqlib._prepared_execution import _finish_native_preparation
            fields = {name: getattr(pending, name) for name in ("realization", "runtime", "snapshot", "observation",
                "logical", "quantum_layout", "classical_layout", "construction_id")}
            fields.update(plan_id=run.plan.content_id, realization_id=pending.realization.content_id)
            # Fixed-workflow joins use the original item, including after restart.
            current = run._state["workflow_current"]
            if pending.workflow_item is not None:
                item = run._state["workflow_items"][pending.workflow_item]
                if item.preparation_id != pending.preparation_id:
                    raise ValueError("remote preparation differs from its original workflow item")
                run._state["workflow_current"] = pending.workflow_item
            try:
                handle = _finish_native_preparation(update, run=run,
                    reservation=run._state["preparation_charges"][identity], fields=fields,
                    items=pending.items, setting=pending.setting, bindings=pending.bindings, remote=pending)
            finally:
                run._state["workflow_current"] = current
            if held is not None:
                raise held
            return handle
        return pending
    except Exception as error:
        surfaced, text = _surfaced_failure(error, held)
        if not run._state["storage_failed"]:
            # Never erase a minimal acknowledged ID because a subsequent public
            # SDK call, payload encoding or bounded result read failed.
            pending = run._state["remote_preparations"][identity]
            if pending.status not in {"completed", "failed", "cancelled"}:
                status = {"upload_intent": "upload_uncertain", "compile_intent": "compile_uncertain"}.get(
                    pending.status, pending.status)
                _save(run, pending, status=status, failure=text)
            if pending.locator is not None:
                next_step = ("its cancellation request failed. While the job has not ended, a later Run.cancel or "
                             "refresh of the same preparation sends the request again and reads the job. Once the "
                             "job is recorded as ended, no further request is sent. Run.wait returns the Result "
                             "when the saved outcomes complete it. Otherwise a cancelled Run starts no new work "
                             "and Run.wait raises RuntimeError naming its folder."
                             if held is not None else
                             "resolve the retrieval failure, then refresh the same preparation.")
                surfaced.add_note(f"Original {pending.locator.provider} preparation job {pending.locator.job_id} "
                                  f"is saved; {next_step}")
        raise surfaced
