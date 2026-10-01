"""One durable local-data -> upload -> compile frontier owned by Run.

Only a fresh stage can create a remote resource. An interrupted intent is
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
    """Commit one revision of the pending record, then install it in live state.

    Every revision between the initial ``local`` record (``begin_preparation``)
    and completion (``_finish_native_preparation``) goes through this write.
    The initial record, each revision and the completion commit before live
    state changes, so live state never runs ahead of the journal.
    """
    revised = pending.revise(**fields)
    run._write((("remote_preparation", revised.preparation_id, revised),))
    run._state["remote_preparations"][revised.preparation_id] = revised
    return revised


def begin_preparation(data, *, run, reservation, fields, items, setting, bindings):
    """Save the local preparation data and a ``local`` record, then start the upload.

    The payload and the pending record commit together, before any provider
    call, so a restart can continue this exact preparation from its saved bytes
    and never needs to lower the circuit again.
    """
    from nwqlib._run_journal import encode
    values = data.model_dump(mode="json")
    payload = encode(values, run.limits.max_data_bytes).encode("utf-8")
    reference = PayloadRef(format=f"{run.backend.kind}.preparation/1", key=str(uuid4()), bytes=len(payload))
    charge = reservation
    saved = {name: fields[name] for name in ("realization", "runtime", "snapshot", "observation", "logical",
                                            "quantum_layout", "classical_layout", "construction_id")}
    pending = PendingPreparation(**saved, preparation_id=charge.preparation_id,
        run_id=run.run_id, local_data=reference, items=items, setting=setting, bindings=bindings,
        workflow_item=run._state["workflow_current"])
    run._write((("remote_preparation", pending.preparation_id, pending),
                ("payload", reference.content_id, reference)), payloads=((reference, payload),))
    run._state["remote_preparations"][pending.preparation_id] = pending
    return _advance(pending, run, data=data)


def refresh_preparation(preparation_id, *, run):
    """Refresh one original preparation once; return a handle or saved pending state.

    No circuit construction or quantum acquisition occurs here. A terminal
    compilation consumes its selected local data only when building its receipt.
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

    Local data are read from the journal only when a stage needs them. Initial
    submission never waits for compilation. Once compilation is pending, each
    call makes one ``refresh_compile`` and either saves the provider status or
    publishes the finished receipt through ``_finish_native_preparation``,
    joined to the original static workflow item. Cancellation marks a ``local``
    or ``uploaded`` preparation as cancelled without a provider call, and a
    compile already running receives one cancellation request. Any error keeps
    every acknowledged identity, turns an unacknowledged intent into its
    uncertain state and propagates.
    """
    backend, identity = run.backend, pending.preparation_id

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
                backend.cancel_preparation(pending.locator, run=run)
            update = backend.refresh_compile(pending.locator, pending.input_ref_json, load_data,
                run=run)
            from nwqlib.backends.connection import NativePreparation
            if not isinstance(update, NativePreparation):
                status = update.status if update.status in {"failed", "cancelled"} else "compile_pending"
                return _save(run, pending, status=status, provider_status=update.provider_status, failure=update.failure)
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
                return _finish_native_preparation(update, run=run,
                    reservation=run._state["preparation_charges"][identity], fields=fields,
                    items=pending.items, setting=pending.setting, bindings=pending.bindings, remote=pending)
            finally:
                run._state["workflow_current"] = current
        return pending
    except Exception as error:
        if not run._state["storage_failed"]:
            # Never erase a minimal acknowledged ID because a subsequent public
            # SDK call, payload encoding or bounded result read failed.
            pending = run._state["remote_preparations"][identity]
            status = {"upload_intent": "upload_uncertain", "compile_intent": "compile_uncertain"}.get(
                pending.status, pending.status)
            _save(run, pending, status=status, failure=f"{type(error).__name__}: {error}")
            if pending.locator is not None:
                error.add_note(f"Original {pending.locator.provider} preparation job {pending.locator.job_id} is saved; "
                               "resolve the retrieval failure, then refresh the same preparation.")
        raise
