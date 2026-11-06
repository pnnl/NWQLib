"""Run the NWQEC compiler or the QDK resource estimator in a local child process with limits.

The byte limit bounds the buffers that the parent holds: the request and input
files, the captured diagnostics with their decoded text and the returned
result, not the internal allocations of a compiled library. The
parameters of each operation bound its finite scientific workload.
"""

from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import selectors
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from nwqlib._choice_archive import ArchiveFiles
from nwqlib._limits import byte_allowance, exceeds, memory_limit
from nwqlib._optional import optional_import
from nwqlib._validation import finite_real, integer

# The package that each operation's child imports, and the extra that installs it.
_PACKAGES = {"nwqec": ("nwqec", "nwqec"), "qre": ("qdk", "qre")}


def optional_timeout(value):
    """Return ``value`` as positive finite seconds, or None for no deadline."""
    if value is None:
        return None
    value = finite_real(value, "timeout_seconds")
    if value <= 0:
        raise ValueError("timeout_seconds must be positive")
    return value


def run_auxiliary(operation, request, *, max_bytes, timeout_seconds, files=()):
    """Run one explicit operation, without retries, capturing its output under ``max_bytes``.

    ``max_bytes`` bounds the buffers that the parent holds at the same time:
    the request and the input files it writes, the captured stdout and
    stderr with their decoded text, and the result it reads back. It does not
    bound the child's own memory (RSS). ``max_bytes=None`` uses the physical
    memory of this computer. Where the platform does not report its memory no
    byte limit applies, and only ``timeout_seconds`` stops a child that keeps
    printing.

    The n captured bytes of both pipes are counted, before each chunk is
    kept, together with a bound of k*n bytes on their decoded text: CPython
    stores every character of a string in 1, 2 or 4 bytes, by its widest
    character, and UTF-8 decoding with replacement gives at most one
    character per byte. k is 1 while no captured byte is 0x80 or larger
    (ASCII), 2 while none is 0xF0 or larger (no character beyond U+FFFF,
    U+FFFD included), and 4 otherwise. The diagnostics are decoded once,
    completely, after the child exits, and the raw bytes are released
    before ``result.json`` is read. Four copies of that file must fit in
    what the request, the input files and the decoded text leave. A refusal
    names ``max_bytes`` or the physical memory.

    ``files`` holds ``(name, bytes)`` inputs that the parent writes into the
    child's directory beside ``request.json`` (NWQEC's ``body.qasm``), counted
    with the request against one ``max_bytes`` allowance, so a large text
    input is written once as it is and is not escaped into the JSON request.

    NWQEC and QDK are native libraries that can hang, abort or print without
    limit. Running each call in a child Python process lets the parent enforce
    a wall-clock deadline when ``timeout_seconds`` is set (``None`` sets
    none), cap the captured diagnostics and kill the child, none of which is
    possible for an in-process native call. The child writes
    its result file only after the operation succeeds, and the parent reads it
    only after a zero exit status, so a partial result is never returned.
    QDK telemetry is switched off in the child environment.

    The parent first checks that the operation's package is installed, so a
    missing extra raises the usual ``ModuleNotFoundError`` with its install
    hint in the caller's process instead of a child failure.

    Returns:
        The decoded result mapping with the captured diagnostics, and the
        elapsed wall time in seconds.
    """
    declared = None if max_bytes is None else integer(max_bytes, "max_bytes", 1)
    limit = memory_limit(declared)
    timeout_seconds = optional_timeout(timeout_seconds)
    if operation not in _PACKAGES:
        raise ValueError("unsupported auxiliary operation")
    package, extra = _PACKAGES[operation]
    if find_spec(package) is None:
        optional_import(package, extra=extra)
    with TemporaryDirectory(prefix="nwqlib-auxiliary-") as directory:
        # The parent keeps the request and the input files for the whole
        # call, so the capture and the result share what their written bytes
        # leave of the limit (archive.remaining, None without a limit).
        archive = ArchiveFiles(directory, limit)
        archive.write_json("request.json", request)
        for name, data in files:
            with archive.writer(name) as stream:
                stream.write(data)
        started = time.monotonic()
        child = subprocess.Popen(
            [sys.executable, "-m", __name__, operation, directory, str(limit)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "QDK_PYTHON_TELEMETRY": "none"},
        )
        captured, width = bytearray(), 1
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
                    tick = 0.1
                    if timeout_seconds is not None:
                        remaining = timeout_seconds - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError(
                                f"{operation} exceeded timeout_seconds={timeout_seconds}"
                            )
                        tick = min(remaining, tick)
                    for key, _ in selector.select(tick):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            # width is k of the docstring. isascii() is a C-level
                            # test, and max() is one linear pass over a chunk that
                            # is not ASCII, until width reaches 4.
                            if width < 4 and not chunk.isascii():
                                width = 4 if max(chunk) >= 0xF0 else 2
                            size = len(captured) + len(chunk)
                            if exceeds((1 + width) * size, archive.remaining):
                                raise ValueError(
                                    f"{operation} diagnostics of {size} bytes need {(1 + width) * size} bytes "
                                    f"with their decoded text, beyond the {archive.remaining} bytes that "
                                    f"{byte_allowance(declared)} leaves after the request and input files"
                                )
                            captured.extend(chunk)
            child.wait()
            if child.returncode:
                # Decoding the prefix with the captured bytes gives the message
                # in one decoding, so the exception holds the only decoded copy.
                captured[:0] = f"{operation} child failed ({child.returncode}): ".encode()
                message = captured.decode("utf-8", errors="replace")
                del captured
                raise RuntimeError(message)
            diagnostics = captured.decode("utf-8", errors="replace")
            allowance = None if archive.remaining is None else archive.remaining - width * len(captured)
            del captured
            # Four copies of the result file: its text, the parsed result and, in
            # compile_logical, the record's JSON and its UTF-8 bytes.
            size = archive.file("result.json").stat().st_size
            if exceeds(4 * size, allowance):
                raise ValueError(
                    f"{operation} result of {size} bytes needs {4 * size} bytes for its copies, beyond the "
                    f"{allowance} bytes that {byte_allowance(declared)} leaves after the request, input "
                    "files and diagnostics"
                )
            result = archive.read_json("result.json")
            result["diagnostics"] = diagnostics
            return result, time.monotonic() - started
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()
            child.stderr.close()


def _compile_nwqec(request, directory, max_bytes):
    """Use the released compiler once at the requested output stage.

    The version is pinned because the saved record describes what each stage
    returns (count-only synthesis counts before fusion, the final circuit after
    fusion) and effective options that were checked for NWQEC 0.1.2 only.
    A count-only result is a short table of gate counts and is returned as
    it is. The QASM text of a full result is counted four times against
    ``max_bytes`` before it is written. The parent holds at most three
    copies at once (the decoded text that the result and the record share,
    and the record's JSON and its UTF-8 bytes in ``compile_logical``'s final
    size check), and the fourth covers the newlines that JSON escapes.
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
        # PBC's width-only depth walker does not establish an execution depth.
        depth = None if target in {"pbc", "pbc_tfuse"} else compiled.depth()
        qasm = compiled.to_qasm()
        if exceeds(4 * len(qasm), max_bytes):
            raise ValueError(
                f"compiled QASM output needs {4 * len(qasm)} bytes for its copies, "
                f"above max_bytes={max_bytes}"
            )
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
    max_bytes = None if limit == "None" else int(limit)
    request = ArchiveFiles(directory, max_bytes).read_json("request.json")
    if operation == "nwqec":
        result = _compile_nwqec(request, directory, max_bytes)
    else:
        from .qre import _estimate_qdk

        result = _estimate_qdk(request)
    # Stream bounded JSON to a private temporary artifact. Only a successful
    # process exit lets the parent publish this complete result.
    ArchiveFiles(directory, max_bytes).write_json("result.json", result)


if __name__ == "__main__":
    _main()
