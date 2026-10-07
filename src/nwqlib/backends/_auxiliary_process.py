"""Bounded local process transport for explicit compiler/estimator operations.

The file and capture limits bound transport/publication, not a native library's
internal allocations. Each operation owns its finite scientific workload.
"""

from __future__ import annotations

import os
from pathlib import Path
import selectors
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from nwqlib._choice_archive import ArchiveFiles
from nwqlib._validation import finite_real, integer


def run_auxiliary(operation, request, *, max_bytes, timeout_seconds, files=()):
    """Run one explicit operation, without retries or unbounded pipe capture.

    ``files`` holds ``(name, bytes)`` inputs that the parent writes into the
    child's directory beside ``request.json`` (NWQEC's ``body.qasm``), charged
    with the request against one ``max_bytes`` allowance, so a large text
    input is written once as it is and is not escaped into the JSON request.

    NWQEC and QDK are native libraries that can hang, abort or print without
    limit. Running each call in a child Python process lets the parent enforce
    a wall-clock deadline, cap the captured diagnostics and kill the child,
    none of which is possible for an in-process native call. The child writes
    its result file only after the operation succeeds, and the parent reads it
    only after a zero exit status, so a partial result is never published.
    QDK telemetry is switched off in the child environment.

    Returns:
        The decoded result mapping with the captured diagnostics, and the
        elapsed wall time in seconds.
    """
    max_bytes = integer(max_bytes, "max_bytes", 1)
    timeout_seconds = finite_real(timeout_seconds, "timeout_seconds")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if operation not in {"nwqec", "qre"}:
        raise ValueError("unsupported auxiliary operation")
    with TemporaryDirectory(prefix="nwqlib-auxiliary-") as directory:
        archive = ArchiveFiles(directory, max_bytes)
        archive.write_json("request.json", request)
        for name, data in files:
            with archive.writer(name) as stream:
                stream.write(data)
        started = time.monotonic()
        child = subprocess.Popen(
            [sys.executable, "-m", __name__, operation, directory, str(max_bytes)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "QDK_PYTHON_TELEMETRY": "none"},
        )
        captured = bytearray()
        try:
            # Drain both pipes while the child works. A noisy dependency cannot
            # fill a pipe indefinitely or turn diagnostics into an unbounded log.
            # The 0.1 s select tick and 8 KiB reads are untuned I/O granularity
            # (docs/ENGINEERING_CONSTANTS.md). They change neither the deadline
            # nor the capture cap.
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                selector.register(child.stderr, selectors.EVENT_READ)
                while selector.get_map() or child.poll() is None:
                    remaining = timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError(
                            f"{operation} exceeded timeout_seconds={timeout_seconds}"
                        )
                    for key, _ in selector.select(min(remaining, 0.1)):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            if len(captured) + len(chunk) > max_bytes:
                                raise ValueError(f"{operation} exceeded bounded diagnostic capture")
                            captured.extend(chunk)
            child.wait()
            if child.returncode:
                detail = captured.decode("utf-8", errors="replace")
                raise RuntimeError(f"{operation} child failed ({child.returncode}): {detail}")
            result = ArchiveFiles(directory, max_bytes).read_json("result.json")
            result["diagnostics"] = captured.decode("utf-8", errors="replace")
            return result, time.monotonic() - started
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()
            child.stderr.close()


def _compile_nwqec(request, directory):
    """Use the released compiler once at the requested output stage.

    The version is pinned because the stored record describes stage semantics
    (count-only synthesis counts before fusion, final artifact after fusion)
    and effective options that were checked for NWQEC 0.1.2 only.
    """
    from nwqlib._optional import optional_import

    nwqec = optional_import("nwqec", extra="nwqec")

    if nwqec.__version__ != "0.1.2":
        raise RuntimeError(f"requires nwqec==0.1.2; found {nwqec.__version__}")
    # The parent wrote the QASM2 body (run_auxiliary files).
    source = Path(directory) / "body.qasm"
    circuit = nwqec.load_qasm(str(source))
    options = {"rz_err": request["rz_err"], "epsilon": request["epsilon"]}
    target = request["target"]
    if request["count_only"]:
        counts = nwqec.get_clifford_t_counts(circuit, keep_ccx=False, **options)
        if sum(counts.values()) > request["max_operations"]:
            raise ValueError("compiled output exceeds max_operations")
        qasm, depth, width = None, None, circuit.num_qubits()
    else:
        if target == "clifford_t":
            compiled = nwqec.to_clifford_t(circuit, keep_ccx=False, **options)
        elif target in {"pbc", "pbc_tfuse"}:
            compiled = nwqec.to_pbc(
                circuit, keep_cx=False, optimize_t_count=target == "pbc_tfuse", **options
            )
        else:
            compiled = nwqec.to_clifford_reduction(circuit, **options)
        counts = compiled.count_ops()
        width = compiled.num_qubits()
        if sum(counts.values()) > request["max_operations"]:
            raise ValueError("compiled output exceeds max_operations")
        # PBC's width-only depth walker does not establish an execution depth.
        depth = None if target in {"pbc", "pbc_tfuse"} else compiled.depth()
        qasm = compiled.to_qasm()
    return {
        "version": nwqec.__version__,
        "counts": counts,
        "width": width,
        "depth": depth,
        "qasm": qasm,
        "module_file": nwqec.__file__,
    }


def _main():
    """Run one operation in the child process from its bounded request file."""
    operation, directory, limit = sys.argv[1:]
    max_bytes = int(limit)
    request = ArchiveFiles(directory, max_bytes).read_json("request.json")
    if operation == "nwqec":
        result = _compile_nwqec(request, directory)
    else:
        from .qre import _estimate_qdk

        result = _estimate_qdk(request)
    # Stream bounded JSON to a private temporary artifact. Only a successful
    # process exit lets the parent publish this complete result.
    ArchiveFiles(directory, max_bytes).write_json("result.json", result)


if __name__ == "__main__":
    _main()
