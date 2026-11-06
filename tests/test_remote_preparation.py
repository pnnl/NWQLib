"""Original remote preparation intents survive errors on either side of acknowledgement."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from nwqlib import load_run
from nwqlib.backends.connection import BackendRefresh
from nwqlib.execution import JobLocator, PendingPreparation, PreparedHandle
from nwqlib._prepared_execution import Run, prepare_experiment
from nwqlib._remote_preparation import refresh_preparation
from test_run_lifecycle import InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend


@pytest.mark.parametrize("interruption", [None, "upload_before_ack", "upload_after_ack",
                                         "compile_before_ack", "compile_after_ack"])
def test_remote_intents_and_acknowledgements_resume_original_preparation(
        tmp_path, monkeypatch, interruption):
    """Interrupt before or after upload/compile acknowledgement and require reconciliation of the
    original preparation.
    """
    from nwqlib._run_journal import LocalJournal

    chosen = selected(5)
    backend = InjectedBackend()
    calls, template = [], {}
    status = ["QUEUED"]
    original = InjectedBackend.prepare

    def fail(point):
        if interruption == point:
            raise ConnectionError(point)

    def local(self, circuit, **kwargs):
        calls.append("local")
        template["native"] = replace(original(self, circuit, **kwargs),
            payload=b"compiled reference", payload_format="fixture.reference/1", operations=None)
        template["data"] = dict(snapshot=kwargs["snapshot"], submitted_program="one measured qubit")
        return SimpleNamespace(model_dump=lambda **kw: template["data"])

    def upload(self, data, *, preparation_id, run, acknowledge):
        calls.append("upload")
        assert run._state["remote_preparations"][preparation_id].status == "upload_intent"
        fail("upload_before_ack")  # The service accepted, but its ID was not recorded.
        acknowledge("original-upload")
        fail("upload_after_ack")
        return '{"id":"original-upload"}'

    def compile(self, reference, *, preparation_id, run, acknowledge):
        calls.append("compile")
        assert json.loads(reference)["id"] == "original-upload"
        assert run._state["remote_preparations"][preparation_id].status == "compile_intent"
        fail("compile_before_ack")
        locator = JobLocator(provider="injected", job_id="original-compile")
        acknowledge(locator)
        fail("compile_after_ack")
        return locator

    def reconcile(self, identity, stage, **kwargs):
        calls.append("reconcile_" + stage)
        return ('{"id":"original-upload"}' if stage == "upload" else
                JobLocator(provider="injected", job_id="original-compile"))

    def restore_upload(self, program_id, **kwargs):
        calls.append("restore_upload")
        assert program_id == "original-upload"
        return '{"id":"original-upload"}'

    def restore_data(self, payload, **kwargs):
        calls.append("restore_data")
        assert json.loads(payload) == template["data"]
        return json.loads(payload)

    def refresh(self, locator, reference, load_data, **kwargs):
        calls.append("refresh")
        assert locator.job_id == "original-compile"
        if status[0] != "COMPLETED":
            return BackendRefresh(status="acknowledged", provider_status=status[0])
        load_data()
        return template["native"]

    for name, function in (("prepare_local", local), ("upload", upload), ("start_compile", compile),
                           ("reconcile_preparation", reconcile), ("restore_upload", restore_upload),
                           ("restore_preparation", restore_data), ("refresh_compile", refresh)):
        monkeypatch.setattr(InjectedBackend, name, function, raising=False)
    with Run(chosen, backend=backend, directory=tmp_path/"original") as run:
        run.checkpoint(dict(point="original acquisition"))
        if interruption is None:
            pending = prepare_experiment(chosen.resolve("counts"), run=run)
            assert isinstance(pending, PendingPreparation) and pending.status == "compile_pending"
        else:
            with pytest.raises(ConnectionError, match=interruption):
                prepare_experiment(chosen.resolve("counts"), run=run)
        saved, = run._state["remote_preparations"].values()
        identity, original_runtime = saved.preparation_id, saved.runtime
        if interruption == "upload_after_ack":
            assert saved.upload_id == "original-upload"
        if interruption == "compile_after_ack":
            assert saved.locator.job_id == "original-compile"
        assert run.trace.preparations == 1 and run.trace.jobs == 0 and run.trace.events == ()
        archive = run.save(tmp_path/"saved")

    def forbidden(*args, **kwargs):
        raise AssertionError("queued refresh reconstructed or loaded a circuit")

    monkeypatch.setattr(InjectedBackend, "prepare_local", forbidden)
    with load_run(archive, backend=backend, method=chosen.method) as run:
        assert tuple(run._state["remote_preparations"]) == (identity,)
        assert run.checkpoint_state == {"point": "original acquisition"}
        with monkeypatch.context() as guarded:
            guarded.setattr(LocalJournal, "read_payload", forbidden)
            pending = refresh_preparation(identity, run=run)
            assert isinstance(pending, PendingPreparation)
        status[0] = "COMPLETED"
        handle = refresh_preparation(identity, run=run)
        assert isinstance(handle, PreparedHandle)
        assert handle.record.runtime == original_runtime and handle.record.observation.shots == 5
        assert handle.record.snapshot == template["data"]["snapshot"]
        assert handle.record.native_operations is None
        assert run.trace.preparations == 1 and run.trace.jobs == 0
        assert calls.count("local") == calls.count("upload") == calls.count("compile") == 1
        assert calls.count("restore_data") == 1
        assert calls.count("reconcile_upload") == int(interruption == "upload_before_ack")
        assert calls.count("restore_upload") == int(interruption == "upload_after_ack")
        assert calls.count("reconcile_compile") == int(interruption == "compile_before_ack")


def test_a_run_beyond_its_data_limit_starts_no_upload_or_compilation(tmp_path, monkeypatch):
    """A refresh starts no new remote stage while the Run's counted data exceed max_data_bytes.

    One preparation stops at ``local`` and another at ``uploaded``, after its upload was acknowledged. A
    detached job's long provider status then takes the Run beyond its limit. A refresh of either
    preparation is refused with the ``extend_limits`` remedy before any provider call and keeps its stage
    and upload identity. After ``extend_limits`` each starts its remaining stages once.
    """
    from nwqlib._prepared_execution import refresh_submissions, submit_detached
    from nwqlib.core.planning import RuntimeOptions
    from nwqlib.execution import ExecutionLimits

    cap, calls, stop = 50_000, [], [False]
    original = InjectedBackend.prepare

    def local(self, circuit, **kwargs):
        native = replace(original(self, circuit, **kwargs), payload=b"compiled", payload_format="fixture.reference/1",
                         operations=None)
        data = dict(snapshot=kwargs["snapshot"], native=native.payload.decode())
        return SimpleNamespace(model_dump=lambda **kw: data)

    def upload(self, data, *, preparation_id, run, acknowledge):
        calls.append("upload")
        acknowledge("upload-" + preparation_id)
        if stop[0]:
            raise ConnectionError("upload response lost")
        return json.dumps({"id": "upload-" + preparation_id})

    def compile(self, reference, *, preparation_id, run, acknowledge):
        calls.append("compile")
        locator = JobLocator(provider="injected", job_id="compile-" + preparation_id)
        acknowledge(locator)
        return locator

    chosen, backend = selected(shots=4), InjectedBackend()
    with Run(chosen, backend=backend, directory=tmp_path / "run", limits=ExecutionLimits(max_data_bytes=cap)) as run:
        point = chosen.resolve("counts")
        submit_detached((prepare_experiment(point, run=run, runtime=RuntimeOptions(seed=31)),), run=run)
        for name, function in (("prepare_local", local), ("upload", upload), ("start_compile", compile),
                               ("restore_upload", lambda self, program_id, **kw: json.dumps({"id": program_id})),
                               ("restore_preparation", lambda self, payload, **kw: json.loads(payload))):
            monkeypatch.setattr(InjectedBackend, name, function, raising=False)
        with monkeypatch.context() as interrupted:
            interrupted.setattr("nwqlib._remote_preparation._advance", lambda *a, **k: (_ for _ in ()).throw(
                KeyboardInterrupt()))
            with pytest.raises(KeyboardInterrupt):
                prepare_experiment(point, run=run, runtime=RuntimeOptions(seed=32))
        stop[0] = True
        with pytest.raises(ConnectionError):
            prepare_experiment(point, run=run, runtime=RuntimeOptions(seed=33))
        stop[0] = False
        saved = {item.status: item for item in run._state["remote_preparations"].values()}
        assert set(saved) == {"local", "uploaded"} and calls == ["upload"]
        calls.clear()
        with monkeypatch.context() as provider:
            provider.setattr(InjectedBackend, "refresh", lambda self, locator, natives, **kw: BackendRefresh(
                status="completed", provider_status="offline provider " + "x" * cap,
                results=(("0", self._result(natives[0])),)))
            refresh_submissions(run=run)
        for item in saved.values():
            with pytest.raises(ValueError, match=r"run\.extend_limits\(max_data_bytes=\.\.\.\)"):
                refresh_preparation(item.preparation_id, run=run)
        assert calls == []
        run.extend_limits(max_data_bytes=1_000_000)
        assert {refresh_preparation(item.preparation_id, run=run).status for item in saved.values()} == {
            "compile_pending"}
        assert calls.count("upload") == 1 and calls.count("compile") == 2
