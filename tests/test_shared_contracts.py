"""Tests for stable shared public contracts."""

import json
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path

import pytest
import numpy as np

import nwqlib
from nwqlib.reporting import ReportSection


def test_dense_pauli_default_preserves_scale_and_relative_pruning_is_covariant() -> None:
    from nwqlib.subroutines.pauli_decomposition import decompose_matrix_to_pauli

    matrix = np.array([[1.0, 1.0e-6], [1.0e-6, -1.0]])
    absolute = decompose_matrix_to_pauli(matrix, atol=1.0e-12)
    np.testing.assert_array_equal(absolute.to_sparse_pauli_op().to_matrix(), matrix)
    relative = decompose_matrix_to_pauli(matrix, atol=1.0e-12, rtol=1.0e-5)
    assert [term.label for term in relative.terms] == ["Z"]
    assert np.linalg.norm(relative.to_sparse_pauli_op().to_matrix() - matrix, 2) == 1.0e-6
    tiny_matrix = np.array([[1.0, 1.0e-100], [1.0e-100, -1.0]])
    exact = decompose_matrix_to_pauli(tiny_matrix)
    np.testing.assert_array_equal(exact.to_sparse_pauli_op().to_matrix(), tiny_matrix)
    for scale in (1e-300, 1e-100, 1e100, 1e300):
        full = decompose_matrix_to_pauli(matrix*scale)
        assert {term.label for term in full.terms} == {"X", "Z"}
        pruned = decompose_matrix_to_pauli(matrix*scale, rtol=1e-5)
        assert [term.label for term in pruned.terms] == ["Z"]
        assert pruned.terms[0].coefficient == scale


def test_documentation_uses_one_source_tree() -> None:
    repo_root = Path(__file__).parents[1]
    config = (repo_root / "mkdocs.yml").read_text()
    docs_dir = repo_root / "docs"
    nav_targets = re.findall(r"(?m)^\s+- [^:]+: (\S+\.md)$", config)
    site_pages = {
        page.relative_to(docs_dir).as_posix()
        for page in docs_dir.rglob("*.md")
        if page.relative_to(docs_dir).parts[0] not in {"References", "scripts"}
    }

    assert re.search(r"(?m)^docs_dir: docs$", config)
    assert nav_targets and len(nav_targets) == len(set(nav_targets))
    assert set(nav_targets) == site_pages
    for target in nav_targets:
        page = docs_dir / target
        assert page.is_file(), target
        assert "--8<--" not in page.read_text(), target


def test_root_surface_rejects_unknown_names() -> None:
    with pytest.raises(AttributeError, match="not_a_nwqlib_name"):
        nwqlib.not_a_nwqlib_name  # noqa: B018


def test_root_import_stays_lazy() -> None:
    script = (
        "import sys\n"
        "import warnings\n"
        "import nwqlib\n"
        "qiskit_modules = [name for name in sys.modules if name.startswith('qiskit')]\n"
        "assert not qiskit_modules, qiskit_modules\n"
        "public = [name for name in nwqlib.__all__ if name != '__version__']\n"
        "loaded = [f'nwqlib.{name}' for name in public if f'nwqlib.{name}' in sys.modules]\n"
        "assert not loaded, loaded\n"
        "warnings.simplefilter('error')\n"
        "import nwqlib.backends as backends\n"
        "for name in ('AerBackend', 'NWQSimBackend', 'NWQSimSlurmBackend', 'IBMRuntimeBackend', 'IonQBackend', 'NexusBackend'):\n"
        "    assert getattr(backends, name).__name__ == name\n"
        "optional = ('qiskit', 'qiskit_aer', 'qiskit_ibm_runtime', 'qiskit_ionq', 'qnexus', 'pytket')\n"
        "assert not any(name == prefix or name.startswith(prefix+'.') for name in sys.modules for prefix in optional)\n"
        "for name in public:\n"
        "    module = getattr(nwqlib, name)\n"
        "    if name in nwqlib._SCIENTIST_OPERATIONS:\n"
        "        assert module.__module__ == 'nwqlib.scientist' and module.__name__ == name\n"
        "    elif name in nwqlib._SEARCH_OPERATIONS:\n"
        "        assert module.__module__ == 'nwqlib.search' and module.__name__ == name\n"
        "    elif name in nwqlib._SCIENTIFIC_INPUTS:\n"
        "        assert isinstance(module, type)\n"
        "        assert module.__module__ == 'nwqlib.problems.records' and module.__name__ == name\n"
        "    else:\n"
        "        assert module.__name__ == f'nwqlib.{name}', module.__name__\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_nested_records_serialize_recursively() -> None:
    status = Enum("Status", {"PASSED": "passed"})
    section = ReportSection(
        title="nested",
        data={
            "status": status.PASSED,
            "records": (ReportSection(title="value", data={"value": np.float32(0.5)}),),
            "lists": [(1, np.int64(2))],
            "flag": np.bool_(True),
            "real_scalars": [
                scalar(value)
                for scalar in (float, np.float32, np.float64, np.longdouble)
                for value in (0.0, 0.5)
            ],
            "large_integer": np.uint64(2**63 + 1),
        },
    )

    payload = section.to_dict()
    assert payload["data"]["status"] == "passed"
    assert payload["data"]["records"] == [{"title": "value", "lines": [], "data": {"value": 0.5}}]
    assert payload["data"]["lists"] == [[1, 2]]
    assert payload["data"]["flag"] is True
    # Exact binary fractions survive conversion, and integer precision stays exact.
    assert payload["data"]["real_scalars"] == [0.0, 0.5] * 4
    assert all(type(value) is float for value in payload["data"]["real_scalars"])
    assert payload["data"]["large_integer"] == 2**63 + 1
    assert type(payload["data"]["large_integer"]) is int
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_plan_display_reads_only_small_selected_metadata(monkeypatch):
    from nwqlib.algorithms import ExpectationMethod
    from nwqlib.core.planning import Plan
    from nwqlib.blocks.records import SelectedConstruction

    selected = nwqlib.plan(nwqlib.Expectation(state=[1,0], observable=[[1,0],[0,-1]]),
                           method=ExpectationMethod(), execution='classical')
    def forbidden(*args, **kwargs):
        pytest.fail('Plan display traversed or serialized scientific data')
    monkeypatch.setattr(Plan, 'to_record', forbidden)
    monkeypatch.setattr(Plan, 'model_dump', forbidden)
    monkeypatch.setattr(Plan, 'content_id', property(forbidden))
    monkeypatch.setattr(SelectedConstruction, '__getattribute__', forbidden)
    text = str(selected)
    assert 'ExpectationMethod' in text and 'execution=classical' in text
    assert 'shots=None' in text and 'experiments=' in text
    assert len(text) < 250  # Fixed scalar-field summary, not a serialized Program.
