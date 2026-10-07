"""IonQ QIS/native conversion and single-request v0.4 execution.

The adapter calls the documented v0.4 REST endpoints directly with one HTTP
request per operation, so no SDK retry can create a second job. No idempotency
key or exact name lookup is qualified for the v0.4 API, so the adapter exposes
no reconciliation capability. A lost creation acknowledgement stays uncertain,
and Run.wait reports RunFailed without creating a replacement or refunding work.
Raw histograms are the only accepted result artifact, because probabilities
are not a count population. docs/ionq.md records the API sources and the
offline qualification scope.
"""

from dataclasses import dataclass
from io import BytesIO
import json
import os
from typing import Annotated, ClassVar, Literal
from urllib.parse import quote

from pydantic import Field

from nwqlib.core.records import PositiveInt, Real, Record, Source, Text
from nwqlib.execution import CountsSampling
from nwqlib.operators.access import Count
from nwqlib.backends._qpy_buffer import QpyBuffer


@dataclass(frozen=True)
class _PreparedIonQ:
    """One converted IonQ circuit and its measurement map.

    input is the IonQ circuit JSON, or None after a result-only restore.
    measurement_map[b] is the qubit measured into classical bit b, or None
    when bit b is never written.
    """

    circuit: object
    input: dict | None
    shots: int
    measurement_map: tuple
    classical_layout: tuple
    num_qubits: int
    metadata: dict


def ionq_decode_bytes(outcomes, qubits, width):
    """Bound one nonempty IonQ histogram decode's live ndarray element bytes.

    Args:
        outcomes: Positive number m of provider histogram entries, including
            zero-count entries and distinct spellings of the same state.
        qubits: Declared native qubit count q, between zero and 64.
        width: Classical measurement-map length c, between zero and 64.

    At most K=min(m, 2**min(q, c)) distinct mapped indices reach unique.
    Allow a separate gather buffer even for the 64-bit identity map.
    NumPy 2.5.2 unique(return_inverse=True) simultaneously holds the native
    states, gathered input, flattening copy, argsort permutation, sorted
    values, cumulative mask and inverse indices, each using 8*m bytes,
    plus the m-byte Boolean mask and 8*K bytes of unique values. The
    cumulative-mask expression's two simultaneous intp buffers fit the
    same bound. Its peak is at most 57*m + 8*K bytes.

    Bit gathering needs at most 32*m bytes. Count scattering needs at most
    24*m + 16*K, and filtering needs at most 16*m + 33*K, including both
    returned arrays. Both are below the unique bound because K <= m.
    intp, uint64 and int64 must each occupy eight bytes. Views count once.

    This bounds one decode through its returned pair, not Python lists,
    dictionaries, integer/string objects, ndarray headers, native workspace,
    transport, public count records, other decodes or process RSS. Stored
    output has its own admission. Re-derive the law if gather_bits,
    remap_counts, NumPy's unique implementation, dtype widths or lifetimes
    change. The helper does constant work and allocates no arrays.
    """
    if outcomes <= 0:
        raise ValueError("outcomes must be a positive integer")
    if not 0 <= qubits <= 64:
        raise ValueError("qubits must be an integer between 0 and 64")
    if not 0 <= width <= 64:
        raise ValueError("width must be an integer between 0 and 64")
    distinct = min(outcomes, 1 << min(qubits, width))
    return 57 * outcomes + 8 * distinct


class IonQBackend(Record):
    """IonQ connection for raw sampled counts on an IonQ QPU, through the v0.4 REST API.

    Build it with keyword arguments, for example
    `IonQBackend(device="qpu.forte-1", max_input_bytes=1_000_000, max_response_bytes=1_000_000)`,
    and pass it as `backend=` to [`prepare`][nwqlib.scientist.prepare].
    `device`, `max_input_bytes` and `max_response_bytes` are required. It needs
    the `ionq` extra. Construction imports no SDK and contacts no service. The API
    key is read from the environment variable named by `token_env` only when a job
    is submitted or retrieved, and it is never stored.

    Each operation is one HTTP request, so no SDK retry can create a second job.
    IonQ's v0.4 API offers no job lookup by name that NWQLib has tested, so a lost
    submission acknowledgement leaves the outcome uncertain, and `Run.wait` raises
    `RunFailed` without submitting a replacement. Only raw histograms are read, because
    IonQ's probability results are not counts of shots. The Run bounds decoded
    data and stored observations. NWQLib has tested this backend only offline,
    with SDK and API checks, and has not tested live QPU behavior. The
    [IonQ guide](../ionq.md) gives the API sources, batch limits and histogram
    decoding.

    Attributes:
        kind: Fixed `"ionq"`, the backend type.
        device: Required. A v0.4 QPU name starting with `qpu.`. The ideal
            simulator is rejected because it ignores the requested shots.
        gateset: Default `"qis"`. `"qis"` translates circuits with Qiskit to Rx, Ry,
            Rz and CX, with angles in radians. `"native"` passes IonQ native gates
            unchanged, with angles in turns.
        token_env: Default `"NWQLIB_IONQ_API_KEY"`. Environment variable that holds
            the API key.
        max_input_bytes: Required. Positive. Limit in bytes on the serialized
            request and the optional QPY copy of each prepared circuit.
        max_response_bytes: Required. Positive. Limit in bytes on each HTTP
            response body that NWQLib reads. It does not bound network buffers,
            provider work, billing or process memory.
        request_timeout_seconds: Default `30.0`. Positive. Connect and read
            timeout of each request in seconds, not a deadline for the whole
            request.
        optimization_level: Default `0`. Qiskit transpiler level, 0 to 3, of the
            `"qis"` translation. The `"native"` gateset accepts only 0. Levels 2 and 3
            resynthesize two-qubit blocks with Qiskit's own synthesis. A level
            other than 0 adds the
            [roundoff exclusion](../glossary.md#roundoff-exclusion)
            `"optimization_level"` to the
            [preparation record](../glossary.md#preparation-record), and the
            operation count describes the compiled circuit. The IonQ target has
            no derived roundoff constant, so
            [`PreparedArtifact.state_error()`][nwqlib.execution.PreparedArtifact.state_error]
            gives no bound at any level.

    Raises:
        ValueError: At preparation, before any request, if `device` does not
            start with `qpu.` or is the ideal simulator, the readout is not
            counts, the shot count is outside 1 to 1,000,000, the `"native"`
            gateset has an `optimization_level` other than 0, or the circuit
            has unbound parameters, no measurements, a measurement followed by
            another operation on its qubit, or two measurements into one
            classical bit.
    """

    qualification_notice: ClassVar[str] = (
        "IonQ has offline SDK/API qualification only; live QPU behavior and raw artifact availability "
        "are unqualified. Check the selected account/device before relying on hardware results.")
    kind: Literal["ionq"] = "ionq"
    supports_synchronous: ClassVar[bool] = False
    device: Text
    gateset: Literal["qis", "native"] = "qis"
    token_env: Text = "NWQLIB_IONQ_API_KEY"
    max_input_bytes: PositiveInt
    max_response_bytes: PositiveInt
    # Untuned per-request connect/read timeout, not a job deadline
    # (docs/ENGINEERING_CONSTANTS.md, "Provider and compiler policies"). Revisit
    # for the selected service and network workload.
    request_timeout_seconds: Annotated[Real, Field(gt=0)] = 30.
    optimization_level: Annotated[Count, Field(le=3)] = 0

    def target_for(self, observation):
        """Admit raw counts on an explicit ``qpu.*`` device before any conversion or request.

        The ideal simulator ignores requested shots, so its histogram is not
        the requested count population.
        """
        if observation.kind == "trajectory":
            observation.reject_unsupported_schedule(f"IonQ {self.device}")
        from nwqlib.backends.capabilities import BackendCapability, BackendTarget
        if observation.kind != "counts":
            raise ValueError("IonQ supports raw sampled counts only")
        if self.device == "simulator":
            raise ValueError("IonQ ideal simulator ignores shots; raw-count execution is unavailable")
        if not self.device.startswith("qpu."):
            raise ValueError("IonQ requires an explicit v0.4 qpu.* device")
        # The shots range of the v0.4 create-job schema for the single-circuit
        # and multi-circuit job types (docs/ENGINEERING_CONSTANTS.md, "Provider
        # and compiler policies").
        # A selected device may accept fewer shots.
        if not 1 <= observation.shots <= 1_000_000:
            raise ValueError("IonQ shots must be between 1 and 1000000")
        return BackendTarget(name=self.device, provider=self.kind,
            capabilities=(BackendCapability.COUNTS, BackendCapability.HARDWARE_SUBMIT,
                          BackendCapability.NATIVE_GATE_TARGET), readouts=("counts",),
            description="IonQ raw histograms with original circuit/child association; offline qualification only")

    def _json_bytes(self, value, run):
        """Encode a request body within the configured input and Run data byte limits.

        The Run share is one third of ``max_data_bytes``, an untuned factor that
        leaves room for the JSON text and its UTF-8 bytes to coexist with journal
        metadata (docs/ENGINEERING_CONSTANTS.md, "Provider and compiler
        policies"). It does not bound the memory of requests or the SDK.
        """
        from nwqlib._run_journal import encode
        return encode(value, min(self.max_input_bytes, run.limits.max_data_bytes // 3)).encode("utf-8")

    def _request(self, method, path, *, run, body=None):
        """Send one v0.4 HTTP request and return its parsed JSON body.

        The body is read from the documented streamed ``response.raw`` and
        admitted read by read against max_response_bytes and the Run data
        allowance before it is parsed. The reader never holds more than
        max_response_bytes + 1 body bytes. A transport error during that read
        propagates as a ``urllib3.exceptions`` error, for example
        ``ProtocolError`` or ``ReadTimeoutError``, not as a
        ``requests.RequestException``. Redirects are refused and nothing is
        retried.
        """
        from nwqlib._optional import optional_import
        requests = optional_import("requests", extra="ionq")
        token = os.environ.get(self.token_env)
        if not token:
            raise ValueError(f"IonQ credential environment variable {self.token_env} is unavailable")
        data = None if body is None else self._json_bytes(body, run)

        # Public requests API: one call, no provider create retry or redirects.
        with requests.request(method, "https://api.ionq.co/v0.4/" + path,
                headers={"Authorization": "apiKey " + token, "Content-Type": "application/json"},
                data=data, timeout=self.request_timeout_seconds, stream=True, allow_redirects=False) as response:
            response.raise_for_status()
            if not 200 <= response.status_code < 300:
                raise ValueError("IonQ returned an unsupported HTTP redirect")
            # Each read asks for at most the bytes still allowed plus one, so the
            # total never exceeds max_response_bytes + 1, and that one extra byte
            # is what reveals an oversized body. urllib3 2, which the ionq extra
            # requires, returns at most the requested number of decoded bytes
            # from HTTPResponse.read. The 64 KiB read size is untuned. Five times
            # the bytes read so far is an untuned allowance for the parsed JSON,
            # checked against the Run's data cap before json.loads runs. Both are
            # registered in docs/ENGINEERING_CONSTANTS.md.
            parts, size = [], 0
            while part := response.raw.read(min(65536, self.max_response_bytes + 1 - size), decode_content=True):
                size += len(part)
                if size > self.max_response_bytes:
                    raise ValueError("IonQ response exceeds max_response_bytes")
                run.check_data(5 * size)
                parts.append(part)
            run.check_data(5 * size)
            return json.loads(b"".join(parts)) if size else None

    def _convert(self, circuit):
        """Convert with the public qiskit-ionq helper, returning IonQ JSON and the measurement map."""
        from nwqlib._optional import optional_import

        helpers = optional_import("qiskit_ionq.helpers", extra="ionq")
        gates, _, mapping = helpers.qiskit_circ_to_ionq_circ(circuit, gateset=self.gateset)
        return dict(qubits=circuit.num_qubits, gateset=self.gateset, circuit=gates), tuple(mapping)

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot):
        """Convert one bound circuit with final measurements into IonQ circuit JSON.

        The flat IonQ format returns one final-state histogram, so every
        measurement must be final and write a distinct classical bit. The QIS
        gateset is reached through Qiskit's Rx/Ry/Rz/CX basis at the selected
        optimization level, with angles in radians. A native-gateset circuit is passed
        unchanged, keeping its angles in turns. The measurement map returned by
        the converter is the translation the decoder later applies.
        """
        context = run._state["backend_context"]
        from qiskit import qpy, transpile
        from nwqlib.backends.connection import NativePreparation, circuit_layout, environment
        self.target_for(observation)
        if circuit.num_parameters or not circuit.num_clbits:
            raise ValueError("IonQ requires a bound circuit with final classical measurements")
        if self.gateset == "native" and self.optimization_level != 0:
            raise ValueError("IonQ native-gate circuits are converted without Qiskit lowering; optimization_level must be 0")
        # The flat format returns the final quantum state. No earlier measured
        # value may be overwritten or followed by another quantum operation.
        measured, classical = set(), set()
        for inst in circuit.data:
            if inst.operation.name == "measure":
                bit = circuit.find_bit(inst.clbits[0]).index
                if bit in classical:
                    raise ValueError("IonQ does not support overwritten classical measurements")
                classical.add(bit)
                measured.update(circuit.find_bit(q).index for q in inst.qubits)
            elif inst.operation.name != "barrier" and any(circuit.find_bit(q).index in measured for q in inst.qubits):
                raise ValueError("IonQ flat circuits require final measurements")
        # Qiskit would lower a dense UnitaryGate with its inexact synthesis, so
        # levels 0 and 1 receive the exact synthesis. The Run makes it for each
        # distinct matrix once per Run object and reserves each new one against
        # its max_synthesis_work first. Levels 2 and 3 resynthesize two-qubit
        # blocks with Qiskit's own synthesis, so they keep the logical circuit
        # as given (docs/dependency_issues.md).
        native = (transpile(run._exact_dense_unitaries(circuit) if self.optimization_level <= 1 else circuit,
                            basis_gates=["rx", "ry", "rz", "cx"], optimization_level=self.optimization_level,
                            seed_transpiler=runtime.seed) if self.gateset == "qis" else circuit)
        ionq_input, mapping = self._convert(native)
        if not any(bit is not None for bit in mapping):
            raise ValueError("IonQ requires final classical measurements")
        options = dict(gateset=self.gateset, measurement_map=mapping, num_qubits=native.num_qubits)
        options_json = self._json_bytes(options, run).decode("utf-8")
        payload = None
        if context.get("persist_prepared", False):
            # The QPY buffer and the bytes copy from getvalue() each hold at
            # most max_input_bytes.
            run.check_data(2 * self.max_input_bytes)
            with QpyBuffer(self.max_input_bytes, message="IonQ QPY exceeds max_input_bytes") as buffer:
                qpy.dump(native, buffer)
                payload = buffer.getvalue()
        layout = circuit_layout(native, native.cregs)
        prepared = _PreparedIonQ(native, ionq_input, observation.shots, mapping, layout, native.num_qubits,
                                 dict(readout_population="unconditional", observation=observation))
        return NativePreparation(native=prepared,
            target=Source(name=self.device, version="v0.4", domain="explicit IonQ raw-count target",
                          reference="https://docs.ionq.com/api-reference/v0.4/jobs/create-job"),
            compiler=Source(name="qiskit-ionq public circuit converter", version=environment(("qiskit-ionq",))[0].version,
                domain="QIS radians; native turns; no automatic mitigation",
                reference="qiskit_ionq.helpers.qiskit_circ_to_ionq_circ"),
            native_basis=tuple(sorted({inst.operation.name for inst in native.data})),
            environment=environment(("nwqlib", "qiskit", "qiskit-ionq", "requests")),
            quantum_layout=circuit_layout(native, native.qregs), classical_layout=layout,
            logical_to_native=(tuple(range(circuit.num_qubits)) if native.layout is None
                              else tuple(native.layout.final_index_layout())), operations=len(native.data), population="unconditional",
            transformation=(("exact dense-unitary synthesis, " if self.optimization_level <= 1 else "")
                            + "public QIS lowering and converter" if self.gateset == "qis"
                            else "public native-gate converter") + "; final measured-wire to classical-bit mapping",
            payload=payload, payload_format="nwqlib.ionq.qpy/1" if payload is not None else None,
            counts_sampling=CountsSampling(kind="fresh"), provider_options_json=options_json,
            probability_window_exclusions=("optimization_level",) if self.optimization_level != 0 else None)

    def restore_native(self, record, payload=None, *, run):
        """Restore the saved measurement map, converting QPY again only when a payload is supplied."""
        text = record.provider_options_json
        run.check_data(4 * len(text))
        options = json.loads(text)
        circuit = ionq_input = None
        mapping = tuple(options["measurement_map"])
        if payload is not None:
            from qiskit import qpy
            # The saved bytes and the stream that qpy.load reads, at most one
            # payload-sized copy each (an untuned allowance).
            run.check_data(2 * len(payload))
            circuit, = qpy.load(BytesIO(payload))
            ionq_input, mapping = self._convert(circuit)
        return _PreparedIonQ(circuit, ionq_input, record.observation.shots, mapping, record.classical_layout,
                            options["num_qubits"], dict(readout_population=record.population, observation=record.observation))

    def admit_batch(self, natives):
        """Admit one IonQ job with at most 5000 circuits, 150000 gates and one shared shot count.

        IonQ's v0.4 create-job reference limits a multi-circuit job to 5000
        circuits and 150000 gates in total across them. Gates are counted here
        as entries of each circuit's IonQ JSON gate list. The reference gives no
        gate limit for a single-circuit job, and this adapter applies the same
        gate total to one as its own cap (docs/ENGINEERING_CONSTANTS.md,
        "Provider and compiler policies", row "IonQ multi-circuit job"). A v0.4
        job has a single shots field, so its circuits must request equal
        shots.
        """

        if not 1 <= len(natives) <= 5000:
            raise ValueError("IonQ requires between 1 and 5000 circuits per submission")
        if len({native.shots for native in natives}) != 1:
            raise ValueError("IonQ batch circuits require equal requested shots")
        if sum(len(native.input["circuit"]) for native in natives if native.input is not None) > 150000:
            raise ValueError("IonQ submission exceeds 150000 gates in total")
        for native in natives:
            self.target_for(native.metadata["observation"])

    def launch(self, natives, *, submission_id, run):
        """Create one IonQ job with a single POST and return its locator.

        Batch circuits are named ``nwqlib:<submission_id>:<index>`` so that
        ``refresh`` can join each returned child job to its original item by
        name rather than by list position. Debiasing is sent as False.
        """
        from nwqlib.execution import JobLocator
        self.admit_batch(natives)
        if any(native.input is None for native in natives):
            raise ValueError("new IonQ submission requires the prepared circuit payload")
        # Untuned 128-byte allowance per circuit for the item names and request
        # entries built below. The request body itself is admitted by _json_bytes.
        run.check_data(128 * len(natives))
        if len(natives) == 1:
            kind, ionq_input = "ionq.circuit.v1", natives[0].input
        else:
            kind = "ionq.multi-circuit.v1"
            ionq_input = dict(gateset=self.gateset, qubits=max(native.num_qubits for native in natives),
                circuits=[dict(native.input, name=f"nwqlib:{submission_id}:{index}")
                          for index, native in enumerate(natives)])
        response = self._request("POST", "jobs", run=run, body=dict(type=kind, backend=self.device,
            shots=natives[0].shots, input=ionq_input, name="nwqlib:" + submission_id,
            metadata={"nwqlib_submission": submission_id}, settings={"error_mitigation": {"debiasing": False}}))
        return JobLocator(provider=self.kind, job_id=response["id"], account=self.token_env, instance=self.device)

    def _path(self, locator):
        """API path of the original job, after checking that the locator names this account and device."""
        if locator.provider != self.kind or locator.account != self.token_env or locator.instance != self.device:
            raise ValueError("IonQ locator differs from the selected account/device")
        return "jobs/" + quote(locator.job_id, safe="")

    def _decode(self, payload, *, artifact_format, native, shots, job_id, run):
        """Translate one raw IonQ histogram into classical outcome counts.

        A v1 histogram key is a decimal state whose bit q is qubit q, the
        convention of qiskit-ionq's map_output. A v2 output_all key is read
        most significant first to give the same integer. The v2 convention
        has offline fixture qualification only. Classical bit b takes qubit
        measurement_map[b], and an unwritten bit stays zero. The integer total
        must equal actual shots and must not exceed admitted shots.

        Up to 64 native qubits and 64 classical bits, return a uint64/int64
        pair after admitting ionq_decode_bytes. Wider inputs use Python
        integers and bit-string keys. Both routes validate every provider entry
        and omit outcomes whose summed count is zero. The m*(c+16) admission
        covers represented count-key JSON, not Python heap memory or the public
        CountBin envelope.
        The common publication owner separately reserves public count records.
        """
        from nwqlib.backends.results import BackendRunResult, remap_counts
        from nwqlib.execution import ExecutionMode

        if artifact_format == "ionq.result.histogram.json.v1":
            histogram = payload
            base = 10
        else:
            if (type(payload) is not dict or type(payload.get("histogram")) is not dict
                    or type(payload["histogram"].get("registers")) is not dict):
                raise ValueError("IonQ v2 result requires a histogram registers mapping")
            registers = payload["histogram"]["registers"]
            if set(registers) != {"output_all"}:
                raise ValueError("IonQ counts require a joint output_all histogram; separate marginals cannot be joined")
            histogram, base = registers["output_all"], 2
        if type(histogram) is not dict or not histogram:
            raise ValueError("IonQ raw histogram must contain shot counts")
        outcomes, width = len(histogram), len(native.measurement_map)
        # Represented bit-string count entries, with at most seven count digits.
        # Public CountBin JSON and cumulative storage have separate reservations.
        run.check_data(outcomes * (width + 16))
        paired = native.num_qubits <= 64 and width <= 64
        if paired:
            run.check_data(ionq_decode_bytes(outcomes, native.num_qubits, width))
            import numpy as np
            states = np.empty(outcomes, dtype=np.uint64)
        else:
            counts = {}
        total = 0
        for position, (key, count) in enumerate(histogram.items()):
            if type(key) is not str or not key or (base == 2 and (len(key) != native.num_qubits or set(key)-{"0", "1"})):
                raise ValueError("IonQ histogram has an invalid outcome")
            state = int(key, base)
            if not 0 <= state < 1 << native.num_qubits or type(count) is not int or count < 0:
                raise ValueError("IonQ raw counts require nonnegative integers in the declared qubit space")
            if paired:
                states[position] = state
            elif count > 0:
                mapped = sum(((state >> qubit) & 1) << bit
                             for bit, qubit in enumerate(native.measurement_map)
                             if qubit is not None)
                bits = format(mapped, f"0{width}b")
                counts[bits] = counts.get(bits, 0) + count
            total += count
        if total != shots or total > native.shots:
            raise ValueError("IonQ histogram population differs from actual/admitted shots")
        if paired:
            counts = remap_counts(states, list(histogram.values()), native.measurement_map)
        return BackendRunResult(
            execution_mode=ExecutionMode.SHOTS,
            backend_target=self.target_for(native.metadata["observation"]),
            raw_output={"counts": counts}, metadata={"native_job_id": job_id})

    def _raw_job(self, job):
        """Reject a job whose results could merge debiasing variants or nested children."""
        variants = job.get("output", {}).get("error_mitigation", {}).get("debiasing", {}).get("variants")
        if variants or job.get("settings", {}).get("error_mitigation", {}).get("debiasing") is not False:
            raise ValueError("IonQ raw counts require confirmed debiasing=False and no merged variants")
        if job.get("child_job_ids"):
            raise ValueError("IonQ circuit job has unexpected nested child jobs")

    @staticmethod
    def _status(job):
        """Map an IonQ job status to the common submission states. Unknown statuses stay uncertain."""
        from nwqlib.backends.connection import BackendRefresh
        status = job["status"]
        if status in {"submitted", "ready", "started"}:
            return BackendRefresh(status="acknowledged", provider_status=status)
        if status in {"failed", "canceled"}:
            return BackendRefresh(status="failed" if status == "failed" else "cancelled", provider_status=status,
                                  failure=str(job.get("failure") or status))
        if status != "completed":
            return BackendRefresh(status="uncertain", provider_status=str(status), failure="unknown IonQ job status")
        return BackendRefresh(status="completed", provider_status=status)

    @staticmethod
    def _histogram_artifact(job, native, known_result_id=None):
        """Select the raw histogram artifact and its ID, keeping a previously recorded ID fixed.

        Once an artifact ID is saved for an acquisition, a different ID is
        rejected, so a replaced provider artifact cannot silently change the
        population behind an existing observation.
        """
        shots = job["shots"]
        if type(shots) is not int or not 1 <= shots <= native.shots:
            raise ValueError("IonQ actual shots exceed or invalidate the admitted population")
        artifacts = job.get("results") or {}
        for artifact_format in ("ionq.result.histogram.json.v1", "ionq.result.histogram.json.v2"):
            if artifact_format in artifacts:
                descriptor = artifacts[artifact_format]
                if known_result_id is not None and descriptor["id"] != known_result_id:
                    continue
                if descriptor["format"] != artifact_format:
                    raise ValueError("IonQ artifact descriptor has a different format")
                if type(descriptor["id"]) is not str or not descriptor["id"].strip():
                    raise ValueError("IonQ histogram artifact requires a nonempty provider result ID")
                return artifact_format, descriptor["id"]
        if known_result_id is not None:
            raise ValueError("IonQ histogram artifact changed or is no longer available for the original acquisition")
        raise ValueError("IonQ job has no raw histogram artifact; probabilities cannot supply binomial counts")

    def _histogram(self, job, native, *, run, artifact=None):
        """Download one job's selected raw histogram artifact and decode it (``_decode``)."""
        artifact_format, result_id = self._histogram_artifact(job, native) if artifact is None else artifact
        path = "jobs/" + quote(job["id"], safe="") + "/artifacts/" + quote(result_id, safe="")
        payload = self._request("GET", path, run=run)
        return self._decode(payload, artifact_format=artifact_format, native=native, shots=job["shots"],
                            job_id=job["id"], run=run)

    def refresh(self, locator, natives, *, run):
        """Read the original job once and download completed raw histograms.

        A batch parent is read once. While it is pending no child is read.
        Otherwise each unconsumed child is read once. Every child
        must name its parent and carry an original item name. Children are
        joined to items by that name, and new child and artifact IDs are
        returned as associations for the common owner to persist. All
        associations are validated before any artifact download, and a later
        download or decode failure is returned with the results decoded before
        it.
        """
        from dataclasses import replace
        self.admit_batch(natives)
        job = self._request("GET", self._path(locator), run=run)
        kind = "ionq.circuit.v1" if len(natives) == 1 else "ionq.multi-circuit.v1"
        if (job["id"] != locator.job_id or job["backend"] != self.device or job["type"] != kind
                or job.get("parent_job_id")):
            raise ValueError("IonQ returned another job/device/circuit type")
        status = self._status(job)
        if len(natives) == 1:
            self._raw_job(job)
            return (replace(status, results=(("0", self._histogram(job, natives[0], run=run)),))
                    if status.status == "completed" else status)
        submission_id = (job.get("metadata") or {}).get("nwqlib_submission")
        submission = run._state["submissions"].get(submission_id)
        if submission is None or submission.locator != locator:
            raise ValueError("IonQ batch metadata differs from the original submission")
        children = job.get("child_job_ids") or []
        if (type(children) is not list or len(children) > len(natives) or len(set(children)) != len(children)
                or any(item.child_job is not None and item.child_job not in children for item in submission.items)):
            raise ValueError("IonQ batch changed or repeated its child job inventory")
        if status.status == "completed" and len(children) != len(natives):
            raise ValueError("completed IonQ batch omits an original circuit")
        if status.status == "acknowledged":
            # The children of a pending parent are read once it reaches a
            # terminal state, so a poll of a running batch costs one GET.
            return status
        # Untuned 256-byte allowance per item for the name table and the
        # associations built below.
        run.check_data(256 * len(natives))
        names = {f"nwqlib:{submission_id}:{item.item}": (index, item)
                 for index, item in enumerate(submission.items)}
        consumed = {item.child_job for item in submission.items if item.attempt in run._state["by_attempt"]}
        pending, associations, seen, results = [], {}, set(), []
        try:
            for child_id in children:
                if child_id in consumed:
                    continue
                child = self._request("GET", "jobs/" + quote(child_id, safe=""), run=run)
                if (child["id"] != child_id or child["backend"] != self.device or child["type"] != "ionq.circuit.v1"
                        or child.get("parent_job_id") != locator.job_id or child.get("name") not in names):
                    raise ValueError("IonQ child differs from its original parent/circuit name")
                index, item = names[child["name"]]
                if (index in seen or item.attempt in run._state["by_attempt"]
                        or item.child_job is not None and item.child_job != child_id):
                    raise ValueError("IonQ child repeats or replaces an original circuit acquisition")
                seen.add(index)
                if item.child_job is None:
                    run.check_data(len(child_id))
                    item = item.revise(child_job=child_id)
                    associations[item.result_key] = item
                self._raw_job(child)
                child_status = self._status(child)
                if status.status == "completed" and child_status.status != "completed":
                    raise ValueError("completed IonQ batch has an incomplete child")
                if child_status.status == "completed":
                    artifact = self._histogram_artifact(child, natives[index], item.provider_result_id)
                    if item.provider_result_id is None:
                        run.check_data(len(artifact[1]))
                        associations[item.result_key] = item.revise(provider_result_id=artifact[1])
                    pending.append((item.result_key, child, natives[index], artifact))
            # Validate every child association before downloading any result.
            # A later failure must not discard an earlier decoded population.
            for key, child, native, artifact in pending:
                results.append((key, self._histogram(child, native, run=run, artifact=artifact)))
        except Exception as error:
            if run._state.get("storage_failed", False) or not (results or associations):
                raise
            # The common owner persists progress, then raises this same error.
            return replace(status, results=tuple(results), associations=tuple(associations.values()), error=error)
        return replace(status, results=tuple(results), associations=tuple(associations.values()))

    def cancel(self, locator, *, run):
        """Send one cancellation request for the original job. A later refresh reads the outcome."""
        self._request("PUT", self._path(locator) + "/status/cancel", run=run)


__all__ = ["IonQBackend"]
