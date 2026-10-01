"""Actual saved metadata reports reject changed records without loading binary data."""

import hashlib
import json
import re

import pytest

from nwqlib import plan
from nwqlib._choice_archive import ArchiveFiles
from nwqlib.saved_evidence import read_report
from _hadamard_method import case
from test_public_cli import cli


def test_read_report_preserves_v7_identity_and_never_loads_method_or_arrays(tmp_path, monkeypatch):
    import nwqlib.saved_evidence as storage
    from nwqlib.core.planning import Plan

    witness = case()
    selected = plan(witness.problem, method=witness.method, seed=7)
    result = witness.evaluate(selected)  # Actual two-qubit H-controlled-Y-H.
    path = result.save(tmp_path / "result")
    fields = selected.to_record()
    # The current v7 encoding, independent of the extracted helper.
    current = dict(format="nwqlib.plan/7", type="nwqlib.core.planning.Plan", fields=fields)
    digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                current, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    )
    assert Plan.record_identity(fields) == digest == selected.content_id
    before = (path / "result.json").read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("metadata report loaded Method, array, circuit or analysis")

    for owner, operation in (
        (ArchiveFiles, "read_array"),
        (ArchiveFiles, "read_circuit"),
        (storage, "load_plan"),
        (type(witness.method), "analyze"),
    ):
        monkeypatch.setattr(owner, operation, forbidden)
    report = read_report(path)
    assert report["result"]["value"] == pytest.approx(1.0, rel=0, abs=2e-12)
    assert report["metadata_validation"]["canonical_plan_identity"] == "checked"
    assert report["metadata_validation"]["binary_payload_integrity"].startswith("not checked")
    # This metadata-only operation makes no promise about saved numerical bytes.
    for array in path.glob("*.npy"):
        array.write_bytes(b"unread binary fixture")
    assert read_report(path) == report
    assert (path / "result.json").read_bytes() == before
    completed = cli("report", str(path))
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == report
    damaged = json.loads(before)
    damaged["selection"]["selected"]["plan"]["method"]["fields"]["preparation_choice"] = "hzh"
    (path / "result.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="Plan content_id"):
        read_report(path)
    # A v5 Plan description (schema 5 with its retired checks field) is not read as v6.
    damaged = json.loads(before)
    damaged["selection"]["selected"]["plan"].update(schema_version=5, checks=[])
    (path / "result.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="invalid canonical saved Plan metadata"):
        read_report(path)
    damaged = json.loads(before)
    damaged["data"]["trace"]["events"][0]["shots"] = 13
    (path / "result.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="content_id"):
        read_report(path)
    (path / "result.json").write_bytes(before)
    assert read_report(path) == report


@pytest.mark.parametrize("damage", ["list", "missing", "extra", "previous_format", "selection", "receipts", "artifact"])
def test_saved_envelope_has_one_admission_before_method_or_native_reads(tmp_path, monkeypatch, damage):
    import nwqlib
    import nwqlib.saved_evidence as storage
    from nwqlib.algorithms import LCHS

    result = nwqlib.solve(nwqlib.LinearDynamics(A=[[0., 0.], [0., 0.]], initial_state=[0., 0.], time=0),
                          method=LCHS(), execution="classical")
    path = result.save(tmp_path / "result")
    saved = json.loads((path / "result.json").read_text())
    if damage == "list":
        saved = []
    elif damage == "missing":
        del saved["data"]
    elif damage == "extra":
        saved["ignored"] = True
    elif damage == "previous_format":
        saved["format"] = "nwqlib.result/3"
    elif damage == "selection":
        saved["selection"] = []
    elif damage == "receipts":
        saved["data"]["receipts"] = {}
    else:
        saved["data"]["artifacts"] = [dict(manifest={}, array=[])]
    (path / "result.json").write_text(json.dumps(saved))
    monkeypatch.setattr(storage, "load_plan", lambda *a, **k: pytest.fail("invalid envelope imported Method"))
    monkeypatch.setattr(ArchiveFiles, "read_array", lambda *a: pytest.fail("invalid envelope read array"))
    message = "saved"
    if damage in {"list", "missing", "extra"}:
        message = "invalid saved Result envelope"
    elif damage == "previous_format":
        message = re.escape(f"unsupported saved Result format 'nwqlib.result/3'. This NWQLib reads only "
                            f"{storage.RESULT_FORMAT!r}. Open it with the NWQLib release that wrote it, "
                            "or plan and run the problem again.")
    for operation in (storage.load_result, read_report):
        with pytest.raises(ValueError, match=message):
            operation(path)
