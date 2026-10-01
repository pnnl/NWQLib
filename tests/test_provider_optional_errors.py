"""Selected provider boundaries name only genuinely missing root extras."""

import pytest

from nwqlib import _optional
from nwqlib.backends import IBMRuntimeBackend, IonQBackend, NexusBackend


@pytest.mark.parametrize("provider,root,extra,transitive", [
    ("ibm", "qiskit_ibm_runtime", "ibm", False),
    ("ionq", "qiskit_ionq", "ionq", False),
    ("nexus", "qnexus", "nexus", False),
    ("qre", "qdk", "qre", False),
    ("ionq_http", "requests", "ionq", False),
    ("ibm", "qiskit_ibm_runtime", "ibm", True),
])
def test_selected_missing_provider_preserves_original_failure(monkeypatch, provider, root, extra, transitive):
    calls = []
    error = ModuleNotFoundError("selected dependency missing", name="provider_internal_dependency" if transitive else root)

    def missing(name):
        calls.append(name)
        raise error

    monkeypatch.setattr(_optional, "import_module", missing)
    if provider == "ibm":
        monkeypatch.delenv("NWQLIB_IBM_RUNTIME_TOKEN", raising=False)
        backend = IBMRuntimeBackend(device="device", instance="instance", max_input_bytes=1024)
        operation = backend._new_service
    elif provider == "nexus":
        backend = NexusBackend(project="project", device="H2-1", max_input_bytes=1024)
        operation = backend._sdk
    elif provider in {"ionq", "ionq_http"}:
        backend = IonQBackend(device="qpu.test", max_input_bytes=1024, max_response_bytes=1024)
        def operation():
            return backend._convert(None) if provider == "ionq" else backend._request("GET", "jobs", run=None)
    else:
        from nwqlib.backends.qre import _estimate_qdk

        def operation():
            return _estimate_qdk({})
    assert not calls
    with pytest.raises(ModuleNotFoundError) as caught:
        operation()
    assert caught.value is error
    assert len(calls) == 1 and calls[0].split(".")[0] == root
    assert getattr(error, "__notes__", []) == ([] if transitive else [
        f"This selected operation requires the '{extra}' extra: pip install 'nwqlib[{extra}]'."
    ])
