"""Actual saved metadata reports keep record identities without loading binary data."""

import json

import pytest

from nwqlib import plan
from nwqlib._choice_archive import ArchiveFiles
from nwqlib.saved_evidence import read_report
from _hadamard_method import case
from test_public_cli import cli


def test_read_report_preserves_current_identity_and_never_loads_method_or_arrays(tmp_path, monkeypatch):
    import nwqlib.saved_evidence as storage

    witness = case()
    selected = plan(witness.problem, method=witness.method, seed=7)
    result = witness.evaluate(selected)  # Actual two-qubit H-controlled-Y-H.
    path = result.save(tmp_path / "result")
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
    # This metadata-only operation makes no promise about saved numerical bytes.
    for array in path.glob("*.npy"):
        array.write_bytes(b"unread binary fixture")
    assert read_report(path) == report
    assert (path / "result.json").read_bytes() == before
    completed = cli("report", str(path))
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == report
