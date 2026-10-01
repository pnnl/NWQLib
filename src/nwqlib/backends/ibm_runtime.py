"""IBM Quantum Runtime detached adapter for SamplerV2 counts and EstimatorV2 estimates.

Preparation compiles each selected circuit to the device ISA with the Plan's
runtime seed and keeps the logical-to-physical layout. Every submitted job
carries the tag ``nwqlib:<submission UUID>``, which is how ``reconcile`` finds
a job whose creation acknowledgement was lost. Each PUB carries its original
coordinate and preparation snapshot in circuit metadata, so decoding does not
depend on the order of returned PUBs. The adapter follows the contract in
``nwqlib.backends.connection``. docs/ibm.md records the offline qualification
scope.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from io import BytesIO
import json
import os
from typing import Annotated, ClassVar, Literal

from pydantic import Field, model_validator

from nwqlib.core.records import PositiveInt, Record, Source, Text
from nwqlib.execution import CountsSampling
from nwqlib.operators.access import Count
from nwqlib.backends._qpy_buffer import QpyBuffer


def sampler_decode_bytes(widths, shots):
    """Bound one scalar Sampler PUB's join and decode ndarray element bytes.

    Args:
        widths: Nonnegative classical-register widths in join order, with
            at least one register. Their sum w can exceed 64.
        shots: Nonnegative common returned-shot population S.

    Let Q=sum(ceil(w_r/8)) and B=ceil(w/8). Include the selected packed
    input arrays, counting shared element buffers once. Qiskit 2.5.2
    concatenation holds the inputs, their padded unpacked bases and the
    concatenated bits, using S*(9*Q+w) bytes. Repacking holds the inputs,
    concatenated bits, padding when needed and the packed output, using
    S*(Q+w+(9 if w%8 else 1)*B). For a padded two-axis array, add 96 bytes
    for at most three four-entry intp pad-specification buffers. This
    allowance also covers the smaller constant-fill auxiliary array.

    Above 64 bits, get_counts reshapes the packed result and visits row
    views. Each row's tobytes, int conversion, mask, binary spelling and
    defaultdict(int) accumulation use Python objects, with no new ndarray
    element buffer. The final dict copy shares its keys and values. The
    adapter's bit-string permutation also uses Python objects. The only
    live array payload after joining is S*(Q+B), already below the join
    bound, so this branch returns the join bound.

    At w<=64, byte packing and gather hold at most four uint64 shot arrays,
    or 32*S bytes. NumPy 2.5.2 unique(return_counts=True) holds native
    outcomes, gathered input, a sorted flattening copy, a Boolean mask,
    unique values and run-start/count arrays, bounded by 25*S+24*M+16
    bytes with M=min(S, 2**w). Packed inputs and joined bytes add S*(Q+B).
    Disjoint phases use their maximum, including the returned pair.

    This logical payload bound assumes eight-byte intp, uint64 and int64
    and release of previous PUB temporary arrays before the next admission.
    It excludes Python headers, dictionaries, strings, integers, public
    records, transport, other PUBs and fixed native workspace. It is not
    a Python-heap or process-RSS bound. Re-derive it when Qiskit's join,
    padding or get_counts, the gather expressions, NumPy's sorted unique
    implementation, dtype widths, output representation or lifetimes change.
    """
    if type(shots) is not int or shots < 0:
        raise ValueError("shots must be a nonnegative integer")
    widths = tuple(widths)
    if not widths or any(type(width) is not int or width < 0 for width in widths):
        raise ValueError("register widths must be nonnegative integers")
    width = sum(widths)
    packed_registers = sum((width + 7) // 8 for width in widths)
    packed_joint = (width + 7) // 8
    join = shots * max(
        9 * packed_registers + width,
        packed_registers + width + (9 if width % 8 else 1) * packed_joint)
    if width % 8:
        join += 96
    if width > 64:
        return join
    distinct = min(shots, 1 << width)
    decode = shots * (packed_registers + packed_joint) + max(
        32 * shots, 25 * shots + 24 * distinct + 16)
    return max(join, decode)


@dataclass(frozen=True)
class _PreparedIBM:
    """One ISA circuit with its original readout association.

    circuit is None after a result-only restore, when retrieval needs only the
    saved association. classical_layout is the logical register layout that the
    decoder restores. observable is the layout-mapped estimator observable, or
    None for counts.
    """

    circuit: object
    payload: bytes | None
    snapshot: str
    shots: int
    classical_layout: tuple
    metadata: dict
    observable: object | None = None


class IBMRuntimeBackend(Record):
    """IBM Quantum Runtime connection for sampled counts and provider expectation estimates.

    Build it with keyword arguments, for example
    `IBMRuntimeBackend(device="ibm_example", instance="my-instance", max_input_bytes=65_536)`,
    and pass it as `backend=` to [`prepare`][nwqlib.scientist.prepare].
    `device`, `instance` and `max_input_bytes` are required. It needs the `ibm`
    extra. Counts use SamplerV2 and a provider estimate uses EstimatorV2.
    Construction reads no credential and contacts no service. Preparation and
    retrieval use the saved SDK account named by `account_name`, or otherwise the
    token in the environment variable named by `token_env`, and no credential is
    stored in the configuration or the Run's files.

    Preparation compiles each circuit for the device with the Plan's runtime seed
    and keeps the logical-to-physical layout. The Estimator's resilience level is
    zero unless `options_json` selects another, so no error mitigation runs
    implicitly, and the Plan sets the precision. The Run bounds serialized
    buffers and decoded arrays. SDK HTTP buffering, retries, compilation memory,
    wall time and billing are not bounded. A refresh keeps the results already
    decoded when a later result fails to decode. A resumed read fetches the
    whole job payload again, and results already saved skip only the local
    decoding. Only offline SDK and transport checks qualify this backend, and live
    accounts, queues and devices are unqualified. The [IBM Runtime guide](../ibm.md)
    describes jobs, retrieval and cancellation.

    Attributes:
        device: Required. IBM device name.
        instance: Required. IBM instance. `"auto"` is rejected, because retrieval
            and job recovery compare it with the job's own instance.
        account_name: Default `None`. Name of a saved IBM SDK account. With
            `None`, the token is read from the variable `token_env` names.
        token_env: Default `"NWQLIB_IBM_RUNTIME_TOKEN"`. Environment variable that
            holds the API token, read only during preparation or retrieval.
        mode: Default `"job"`. `"batch"` or `"session"` attaches every job to the
            existing container `container_id`.
        container_id: Default `None`. ID of an existing Batch or Session. Required
            in batch and session mode, and must be `None` in job mode.
        calibration_id: Default `None`. Device calibration ID passed to the IBM
            SDK when the device is opened.
        options_json: Default `"{}"`. JSON object of the requested primitive
            options. It states the requested options, not the server's defaults.
            Experimental option overrides are rejected, because they can bypass the
            requested readout and its uncertainty.
        optimization_level: Default `1`. Qiskit preset pass-manager level, 0 to 3.
            A level other than 1 is recorded as the exclusion
            `"optimization_level"` in the preparation record, and the operation
            count describes the compiled circuit. The IBM target has no derived
            roundoff constant, so `state_error()` gives no bound at any level.
        initial_layout: Default `None`. Physical qubit for each logical qubit,
            without repeats.
        max_input_bytes: Required. Positive. Limit in bytes on the QPY copy of
            each prepared circuit, which only a Run saved to a directory writes.

    Raises:
        ValueError: If `container_id` is given in job mode or missing in batch or
            session mode, if `instance` is `"auto"`, or if `initial_layout`
            repeats a qubit.
    """

    qualification_notice: ClassVar[str] = (
        "IBM Runtime has offline SDK/transport qualification only; live account, queue and hardware "
        "behavior are unqualified. Check the selected device before relying on hardware results.")
    kind: Literal["ibm"] = "ibm"
    supports_synchronous: ClassVar[bool] = False
    device: Text
    instance: Text
    account_name: Text | None = None
    token_env: Text = "NWQLIB_IBM_RUNTIME_TOKEN"
    mode: Literal["job", "batch", "session"] = "job"
    container_id: Text | None = None
    calibration_id: Text | None = None
    options_json: Text = "{}"
    optimization_level: Annotated[Count, Field(le=3)] = 1
    initial_layout: tuple[Count, ...] | None = None
    max_input_bytes: PositiveInt

    @model_validator(mode="after")
    def _mode(self):
        """Check the execution mode, the instance and the explicit layout.

        Job mode has no container, and batch or session mode names an existing
        one. The instance must be explicit (not ``auto``), because locators,
        retrieval and reconciliation compare it with the job's own instance.
        """
        if (self.mode == "job") != (self.container_id is None):
            raise ValueError("batch/session mode requires an explicit existing container ID")
        if self.instance == "auto":
            raise ValueError("IBM execution requires one explicit original instance")
        if self.initial_layout is not None and len(set(self.initial_layout)) != len(self.initial_layout):
            raise ValueError("initial physical layout cannot repeat a qubit")
        return self

    def target_for(self, observation):
        """Admit counts or a provider estimate before any service contact.

        A zero observable is rejected because its expectation is exactly zero,
        so the Method can resolve that scalar without a job.
        """
        if observation.kind == "trajectory":
            observation.reject_unsupported_schedule("IBM Runtime")
        from nwqlib.backends.capabilities import BackendCapability, BackendTarget
        if observation.kind not in {"counts", "estimated_observable"}:
            raise ValueError("IBM primitives require counts or an explicitly selected provider estimate")
        if observation.kind == "estimated_observable" and not any(observation.estimate.coefficients):
            raise ValueError("IBM Estimator does not accept a zero observable; the method can resolve that scalar without a job")
        return BackendTarget(name=self.device, provider=self.kind,
            capabilities=(BackendCapability.COUNTS, BackendCapability.ESTIMATED_OBSERVABLE,
                          BackendCapability.HARDWARE_SUBMIT, BackendCapability.NATIVE_GATE_TARGET),
            readouts=("counts", "estimated_observable"),
            description="Explicit IBM V2 primitives; native target is discovered during preparation")

    def _new_service(self):
        """Open an SDK service from the saved account or, otherwise, the token variable.

        The credential is read here, at the first explicit action, and is never
        saved with the configuration or in the Run's files. The live service
        stays in the Run's backend context only while the process runs.
        """
        from nwqlib._optional import optional_import
        IBMQuantumComputeService = optional_import("qiskit_ibm_runtime", extra="ibm").IBMQuantumComputeService
        if self.account_name is not None:
            return IBMQuantumComputeService(name=self.account_name, instance=self.instance)
        token = os.environ.get(self.token_env)
        if not token:
            raise ValueError(f"IBM credential environment variable {self.token_env} is unavailable")
        return IBMQuantumComputeService(channel="ibm_quantum_platform", token=token, instance=self.instance)

    def _service(self, context):
        """The Run's one live service, opened on first use and never saved."""
        if "service" not in context:
            context["service"] = self._new_service()
        return context["service"]

    def _backend(self, context):
        """The Run's live device handle, checked to be the configured device."""
        if "device" not in context:
            context["device"] = self._service(context).backend(self.device, instance=self.instance,
                                                         calibration_id=self.calibration_id)
        backend = context["device"]
        if backend.name != self.device:
            raise ValueError("IBM service returned a different device")
        return backend

    def _options(self, run, kind):
        """Build the requested primitive options and their canonical JSON identity.

        The same JSON is compared at launch, so a job is submitted with exactly
        the options recorded at preparation. Estimator precision belongs to the
        Plan and cannot be overridden here. Resilience level zero is the default
        so the wrapper does not start mitigation work unasked
        (ENGINEERING_CONSTANTS.md, "IBM primitive preparation").
        """
        from nwqlib._optional import optional_import
        options_module = optional_import("qiskit_ibm_runtime.options", extra="ibm")
        from nwqlib._run_journal import encode
        # Text length bounds the JSON node population before parsing it. A text
        # of n characters has at most n JSON nodes and at most 4n UTF-8 bytes.
        run.check_data(4 * len(self.options_json))
        values = json.loads(self.options_json)
        if type(values) is not dict:
            raise ValueError("IBM options require a JSON object")
        if values.get("experimental") not in (None, {}):
            raise ValueError("experimental IBM option overrides are unsupported; use the declared primitive options")
        encode(values, run.limits.max_data_bytes)
        if kind == "estimated_observable":
            if values.get("default_shots") is not None or "default_precision" in values:
                raise ValueError("the original estimate Plan owns precision; do not override it in primitive options")
            if "resilience_level" not in values:
                # The added key's escaped JSON (at most 6 bytes per character,
                # the length of a \uXXXX escape) plus 16 bytes for its value and
                # punctuation.
                run.check_data(6 * len('resilience_level') + 16)
                values["resilience_level"] = 0
            options = options_module.EstimatorOptions(**values)
        else:
            options = options_module.SamplerOptions(**values)
        # Classified bit strings are the actual current counts consumer.
        Unset = optional_import("qiskit_ibm_runtime.options.utils", extra="ibm").Unset
        if kind == "counts" and options.execution.meas_type not in (Unset, "classified"):
            raise ValueError("counts require classified IBM measurement results")
        # UTF-8 JSON, the serialization that encode() admitted above and that
        # the Run journal and record identities write, so the stored text never
        # exceeds the admitted size. ASCII escaping could take up to 12 bytes
        # for a character outside the Basic Multilingual Plane.
        return options, json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False,
                                   ensure_ascii=False)

    @staticmethod
    def _observable(estimate, layout, width, run):
        """Map the full weighted logical observable onto the ISA circuit's physical qubits.

        The byte admission and the exact-zero simplification are described in
        ENGINEERING_CONSTANTS.md, "IBM primitive preparation". The mapped
        observable of the most recent (estimate identity, layout, width) is
        kept in the open Run's live state and shared by consecutive points that
        repeat that key; a new key replaces it, and a reopened Run builds it
        again when needed.
        """
        key = (estimate.content_id, tuple(layout), width)
        cached = run._state.get("ibm_observable")
        if cached is None or cached[0] != key:
            # Only the most recent observable is kept, so a scan that repeats
            # one key builds it once and a Run holds at most one of them.
            cached = run._state["ibm_observable"] = (key, IBMRuntimeBackend._mapped_observable(
                estimate, layout, width, run))
        return cached[1]

    @staticmethod
    def _mapped_observable(estimate, layout, width, run):
        """Build and admit one layout-mapped ``ObservablesArray`` (``_observable``)."""
        from qiskit.primitives.containers import ObservablesArray
        from qiskit.quantum_info import SparseObservable, SparsePauliOp
        terms, logical_width = len(estimate.labels), len(estimate.labels[0])
        run.check_data(16 * terms * (logical_width + width) + 128 * terms)
        operator = SparsePauliOp.from_list(list(zip(estimate.labels, estimate.coefficients, strict=True)))
        native = operator.apply_layout(list(layout), num_qubits=width)
        # Qiskit 2.5's default ObservablesArray coercion simplifies at 1e-8,
        # silently deleting smaller nonzero coefficients. The common declaration
        # already validates finite real unique Pauli terms. Use the public
        # prevalidated-input API with exact-zero simplification; PUB validation
        # still checks the observable/circuit widths and parameter coordinates.
        sparse = SparseObservable.from_sparse_pauli_op(native).simplify(0.)
        return ObservablesArray(sparse, num_qubits=width, validate=False)

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot):
        """Compile the bound point to IBM ISA while preserving logical readout and observable
        layout.

        The Plan's runtime seed selects the transpiler seed, so compilation is
        repeatable for the same device target. The snapshot is written into circuit
        metadata and later checked on every returned PUB. The named classical
        registers must survive compilation unchanged, because the decoder joins
        them per shot and restores the original bit positions. The circuit is
        saved as QPY before submission only when the Run is durable.
        """
        if observation.kind == "trajectory":
            observation.reject_unsupported_schedule("IBM Runtime")
        context = run._state["backend_context"]
        from importlib.metadata import version
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from nwqlib.backends.connection import NativePreparation, circuit_layout, environment

        self.target_for(observation)
        _, options_json = self._options(run, observation.kind)
        backend = self._backend(context)
        if backend.name != self.device or circuit.num_qubits > backend.num_qubits:
            raise ValueError("logical circuit differs from the selected IBM device or exceeds its width")
        if observation.kind == "estimated_observable" and (circuit.num_clbits or any(
                instruction.operation.name in {"measure", "reset", "if_else", "for_loop", "while_loop", "switch_case"}
                for instruction in circuit.data)):
            raise ValueError("provider estimation requires the selected unmeasured final state")
        if self.initial_layout is not None and (len(self.initial_layout) != circuit.num_qubits
                or max(self.initial_layout, default=-1) >= backend.num_qubits):
            raise ValueError("explicit IBM layout differs from the original logical/physical widths")
        # Qiskit would lower a dense UnitaryGate with its inexact synthesis,
        # so levels 0 and 1 receive the exact synthesis. The Run makes it for
        # each distinct matrix once per Run object and reserves each new one
        # against its max_synthesis_work first. Levels 2 and 3 resynthesize two-qubit
        # blocks with Qiskit's own synthesis, so they keep the logical circuit
        # as given (docs/dependency_issues.md).
        lowered = (run._exact_dense_unitaries(circuit)
                   if self.optimization_level <= 1 else circuit)
        native = generate_preset_pass_manager(backend=backend, optimization_level=self.optimization_level,
            seed_transpiler=runtime.seed, initial_layout=None if self.initial_layout is None else list(self.initial_layout)
            ).run(lowered)
        if native.num_parameters:
            raise ValueError("a prepared IBM acquisition must bind one complete original parameter point")
        # Verify named classical registers after compilation, then map the
        # weighted observable with the actual logical-to-native layout.
        classical = circuit_layout(native, native.cregs)
        logical_classical = circuit_layout(circuit, circuit.cregs)
        bits = tuple(bit for register in classical for bit in register.bits)
        if sorted(bits) != list(range(native.num_clbits)):
            raise ValueError("IBM counts require each classical bit in exactly one named result register")
        if {reg.name: len(reg.bits) for reg in classical} != {reg.name: len(reg.bits) for reg in logical_classical}:
            raise ValueError("IBM ISA changed the named logical measurement registers")
        native.metadata = {**(native.metadata or {}), "nwqlib_snapshot": snapshot}
        payload = None
        if context.get("persist_prepared", False):
            from qiskit import qpy
            # The QPY buffer and the bytes copy from getvalue() each hold at
            # most max_input_bytes.
            run.check_data(2 * self.max_input_bytes)
            with QpyBuffer(self.max_input_bytes,
                           message="prepared QPY bytes exceed the selected per-input limit") as buffer:
                qpy.dump(native, buffer)
                payload = buffer.getvalue()
        layout = tuple(native.layout.final_index_layout(filter_ancillas=True))
        metadata = dict(readout_population="unconditional", requested_shots=observation.shots,
                        logical_to_native=layout, observation=observation, requested_options_json=options_json)
        observable = (None if observation.estimate is None else
                      self._observable(observation.estimate, layout, native.num_qubits, run))
        prepared = _PreparedIBM(native, payload, snapshot, observation.shots, logical_classical, metadata, observable)
        return NativePreparation(native=prepared,
            target=Source(name=self.device, version=backend.backend_version,
                domain=f"IBM target configuration; calibration_id={backend.calibration_id or 'unavailable'}",
                reference="IBMQuantumComputeService.backend"),
            compiler=Source(name="Qiskit IBM ISA pass manager", version=version("qiskit"),
                domain=f"optimization_level={self.optimization_level}; original runtime seed selects compilation",
                reference="qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager"),
            native_basis=tuple(sorted(backend.operation_names)),
            environment=environment(("nwqlib", "qiskit", "qiskit-ibm-runtime")),
            quantum_layout=circuit_layout(native, native.qregs), classical_layout=classical,
            logical_to_native=layout, operations=len(native.data), population="unconditional",
            transformation=("exact dense-unitary synthesis, then " if self.optimization_level <= 1 else "")
                + "IBM ISA circuit; original logical/classical mapping and requested primitive options"
                + ("; layout-mapped weighted observable" if observable is not None else ""),
            payload=payload, payload_format="nwqlib.ibm.qpy/1",
            counts_sampling=(CountsSampling(kind="unknown" if backend.configuration().simulator else "fresh")
                             if observation.kind == "counts" else None), provider_options_json=options_json,
            probability_window_exclusions=("optimization_level",) if self.optimization_level != 1 else None)

    def restore_native(self, record, payload=None, *, run):
        """Rebuild the prepared association from a saved receipt, loading QPY only when supplied.

        Retrieval of a known job needs no circuit, so ``payload`` is None there.
        A new submission from saved data needs the QPY, whose snapshot metadata
        must match the receipt. No recompilation happens here.
        """
        if payload is not None and (record.payload is None or record.payload.format != "nwqlib.ibm.qpy/1"):
            raise ValueError("prepared IBM artifact requires its original QPY payload reference")
        circuit = None
        observable = None
        if payload is not None:
            from qiskit import qpy
            # The saved bytes and the stream that qpy.load reads, at most one
            # payload-sized copy each (an untuned allowance).
            run.check_data(2 * len(payload))
            circuits = qpy.load(BytesIO(payload))
            if len(circuits) != 1 or circuits[0].metadata.get("nwqlib_snapshot") != record.snapshot:
                raise ValueError("restored QPY differs from the original IBM snapshot")
            circuit = circuits[0]
            if record.observation.estimate is not None:
                observable = self._observable(record.observation.estimate, record.logical_to_native,
                                               circuit.num_qubits, run)
        if record.provider_options_json is None:
            raise ValueError("IBM preparation requires its original requested primitive options")
        return _PreparedIBM(circuit, payload, record.snapshot, record.observation.shots,
            record.classical_layout, dict(readout_population=record.population, observation=record.observation,
                                         requested_options_json=record.provider_options_json), observable)

    def admit_batch(self, natives):
        """Admit PUBs that one primitive job can carry.

        They share one primitive and one option set. Estimator PUBs share one
        precision, and Sampler PUBs share one positive shot count.
        """

        if not natives or len({native.metadata["observation"].kind for native in natives}) != 1:
            raise ValueError("one IBM job requires one primitive kind")
        if len({native.metadata["requested_options_json"] for native in natives}) != 1:
            raise ValueError("one IBM job requires matching original primitive options")
        if natives[0].metadata["observation"].kind == "estimated_observable":
            if len({native.metadata["observation"].estimate.precision for native in natives}) != 1:
                raise ValueError("current IBM EstimatorV2 jobs require one common requested precision")
        elif len({native.shots for native in natives}) != 1 or any(native.shots < 1 for native in natives):
            raise ValueError("IBM sampled acquisitions require positive shots")

    def launch(self, natives, *, submission_id, run):
        """Submit one primitive job for the admitted PUBs and return its locator.

        The job is tagged ``nwqlib:<submission_id>`` for ``reconcile``. Options
        must equal those recorded at preparation. In job mode an ambient
        Session or Batch context is rejected, since the SDK would otherwise
        attach the job to that container. Batch and Session modes attach only
        to the configured existing container. ``primitive.run`` is called once.
        The SDK's own HTTP retries are described in docs/ibm.md.
        """
        from nwqlib._optional import optional_import
        sdk = optional_import("qiskit_ibm_runtime", extra="ibm")
        Batch, EstimatorV2, SamplerV2, Session = sdk.Batch, sdk.EstimatorV2, sdk.SamplerV2, sdk.Session
        from nwqlib.execution import JobLocator
        self.admit_batch(natives)
        if any(native.circuit is None for native in natives):
            raise ValueError("new IBM submission requires the original native payloads")
        kind = natives[0].metadata["observation"].kind
        options, requested = self._options(run, kind)
        if requested != natives[0].metadata["requested_options_json"]:
            raise ValueError("primitive options differ from the original prepared selection")
        tag = "nwqlib:" + submission_id
        options.environment.job_tags = [*(options.environment.job_tags or ()), tag]
        if self.mode == "job":
            mode = self._backend(run._state["backend_context"])
        else:
            cls = Batch if self.mode == "batch" else Session

            service = self._service(run._state["backend_context"])
            mode = cls.from_id(self.container_id,
                service=service, calibration_id=self.calibration_id)
            if (mode.backend() != self.device or service.active_instance() != self.instance
                    or mode.session_id != self.container_id):
                raise ValueError("existing IBM container belongs to another device/instance or has a different ID")
        primitive = (EstimatorV2 if kind == "estimated_observable" else SamplerV2)(mode=mode, options=options)
        if self.mode == "job" and primitive.mode is not None:
            raise ValueError("ambient IBM Session/Batch would change explicit job mode; exit its context first")

        pubs = []
        for index, native in enumerate(natives):
            # Each PUB needs its own correlation metadata even when the same
            # frozen circuit appears twice. A shallow container shares the
            # immutable circuit data; only its public metadata is replaced.
            from nwqlib._run_journal import encode
            encode(native.circuit.metadata, run.limits.max_data_bytes)
            circuit = copy(native.circuit)
            circuit.metadata = {**native.circuit.metadata, "nwqlib_pub": index}
            if kind == "estimated_observable":
                if native.observable is None:
                    raise ValueError("new Estimator submission requires the original layout-mapped observable")
                pubs.append((circuit, native.observable, None, native.metadata["observation"].estimate.precision))
            else:
                pubs.append((circuit, None, native.shots))
        job = primitive.run(pubs)
        locator = JobLocator(provider=self.kind, job_id=job.job_id(), instance=self.instance,
                             account=self.account_name or self.token_env)
        run._state["backend_context"].setdefault("jobs", {})[locator.job_id] = job
        return locator

    def _job(self, locator, run):
        """Return the original job, rejecting a locator or job from another account, instance or device."""
        if (locator.provider != self.kind or locator.instance != self.instance
                or locator.account != (self.account_name or self.token_env)):
            raise ValueError("IBM job locator differs from the original provider/account/instance")
        jobs = run._state["backend_context"].setdefault("jobs", {})
        if locator.job_id not in jobs:

            jobs[locator.job_id] = self._service(run._state["backend_context"]).job(locator.job_id)
        job = jobs[locator.job_id]
        if (job.job_id() != locator.job_id or job.instance != self.instance
                or job.primitive_id not in {"sampler", "estimator"} or job.backend().name != self.device):
            raise ValueError("retrieved IBM job has a different identity/instance/primitive")
        return job

    def reconcile(self, submission, natives, *, run):
        """Find a job whose creation acknowledgement was lost by its submission tag.

        At most two matches are requested. One verified match becomes the
        locator. No match leaves the intent unresolved, and two raise, because
        neither case identifies the original job and neither authorizes a new
        submission.
        """
        from nwqlib.execution import JobLocator

        primitive = "estimator" if natives[0].metadata["observation"].kind == "estimated_observable" else "sampler"
        matches = self._service(run._state["backend_context"]).jobs(limit=2, backend_name=self.device, instance=self.instance,
                                        program_id=primitive, job_tags=["nwqlib:" + submission.submission_id])
        if not matches:
            return None  # Absence is not proof that a lost submission was rejected.
        if len(matches) != 1:
            raise ValueError("IBM submission correlation is ambiguous; no new job is authorized")
        job, = matches
        if (job.instance != self.instance or job.primitive_id != primitive
                or job.backend().name != self.device or "nwqlib:" + submission.submission_id not in job.tags):
            raise ValueError("IBM reconciliation returned a foreign job")
        locator = JobLocator(provider=self.kind, job_id=job.job_id(), instance=self.instance,
                             account=self.account_name or self.token_env)
        run._state["backend_context"].setdefault("jobs", {})[locator.job_id] = job
        return locator

    def _decode(self, result, natives, *, job_id, run, completed=()):
        """Validate the full PUB association, then yield unconsumed observations."""
        from qiskit.primitives.containers import BitArray, PrimitiveResult, PubResult, SamplerPubResult
        from nwqlib.backends.results import BackendRunResult, gather_bits
        from nwqlib.execution import ExecutionMode

        if not isinstance(result, PrimitiveResult) or len(result) != len(natives):
            raise ValueError("IBM result must contain exactly the original PUB inventory")

        seen = set()
        for pub in result:
            if not isinstance(pub, PubResult):
                raise ValueError("IBM result has a different primitive data type")
            metadata = pub.metadata.get("circuit_metadata", {})
            index = metadata.get("nwqlib_pub")
            if type(index) is not int or not 0 <= index < len(natives) or index in seen:
                raise ValueError("IBM result has an unknown or repeated original PUB coordinate")
            native, key = natives[index], str(index)
            if metadata.get("nwqlib_snapshot") != native.snapshot:
                raise ValueError("IBM result snapshot differs from its original PUB coordinate")
            seen.add(index)
        # A swapped snapshot or repeated coordinate invalidates the result
        # inventory before any PUB is decoded, including on resumed reads.
        for pub in result:
            index = pub.metadata["circuit_metadata"]["nwqlib_pub"]
            native, key = natives[index], str(index)
            if key in completed:
                continue
            if native.metadata["observation"].kind == "estimated_observable":
                yield key, self._estimate_result(pub, result.metadata, native, job_id=job_id, run=run)
                continue
            if not isinstance(pub, SamplerPubResult):
                raise ValueError("IBM Sampler result has a different primitive data type")
            names = tuple(register.name for register in native.classical_layout)
            if set(pub.data) != set(names):
                raise ValueError("IBM result registers differ from the prepared classical layout")
            arrays = tuple(pub.data[name] for name in names)
            if any(not isinstance(array, BitArray) or array.shape != () or array.num_shots > native.shots
                   for array in arrays):
                raise ValueError("IBM PUB data differ from the original scalar parameter coordinate or shot cap")
            for register in native.classical_layout:
                if pub.data[register.name].num_bits != len(register.bits):
                    raise ValueError("IBM result register width differs from the prepared mapping")
            if len({array.num_shots for array in arrays}) != 1:
                raise ValueError("IBM classical registers do not share a joint shot population")
            shots = arrays[0].num_shots
            width = sum(array.num_bits for array in arrays)
            run.check_data(sampler_decode_bytes(
                tuple(len(register.bits) for register in native.classical_layout), shots))
            joined = pub.join_data(names)
            # BitArray concatenates registers along its least-significant-first
            # bit axis; get_counts renders most-significant-first strings.
            # Preserve shot alignment and map to original global classical bits.
            # Joined bit j, counted from the right, is global bit positions[j].
            # Qiskit's join_data docstring describes the opposite register
            # order, but its 2.5 implementation (bit_array._unpack and _pack)
            # puts the first register at the least significant end.
            positions = tuple(bit for register in native.classical_layout for bit in register.bits)
            if width <= 64:
                # Each shot's joined integer is formed from the BitArray's
                # big-endian bytes (last byte holds bits 0-7), global classical
                # bit positions[j] takes joined bit j (gather_bits; the identity
                # layout keeps the joined integers), and equal outcomes are
                # counted once with np.unique, so no per-shot string is built.
                import numpy as np
                rows = joined.array
                native_outcomes = np.zeros(rows.shape[0], dtype=np.uint64)
                for column in range(rows.shape[1]):
                    native_outcomes = (native_outcomes << np.uint64(8)) | rows[:, column].astype(np.uint64)
                sources = [None] * width
                for index, target in enumerate(positions):
                    sources[target] = index
                indices, counts = np.unique(gather_bits(native_outcomes, sources), return_counts=True)
                ordered = (indices, counts.astype(np.int64, copy=False))
                # _decode is a generator: release this PUB's transient arrays
                # before the next PUB's admission, as the bound assumes.
                del rows, native_outcomes, indices, counts
            else:
                ordered = {}
                for bits, count in joined.get_counts().items():
                    output = ["0"] * width
                    for index, target in enumerate(positions):
                        output[width - target - 1] = bits[width - index - 1]
                    label = "".join(output)
                    ordered[label] = ordered.get(label, 0) + count
            del joined
            yield key, BackendRunResult(execution_mode=ExecutionMode.SHOTS,
                backend_target=self.target_for(native.metadata["observation"]),
                raw_output={"counts": ordered}, metadata={"native_job_id": job_id})
            del ordered

    def _estimate_result(self, pub, job_metadata, native, *, job_id, run):
        """Decode one weighted scalar estimate without conflating sampling and extrapolation
        uncertainty.
        """
        import numpy as np
        from importlib.metadata import version
        from nwqlib._run_journal import encode
        from nwqlib.backends.results import BackendRunResult
        from nwqlib.execution import EstimateValue, ExecutionMode

        estimate = native.metadata["observation"].estimate
        value = pub.data["evs"] if "evs" in pub.data else None
        error = pub.data["stds"] if "stds" in pub.data else None
        ensemble = pub.data["ensemble_standard_error"] if "ensemble_standard_error" in pub.data else None
        if (not isinstance(value, np.ndarray) or value.shape != () or value.dtype.kind != "f"
                or error is not None and (not isinstance(error, np.ndarray) or error.shape != () or error.dtype.kind != "f")):
            raise ValueError("Estimator data must match the original scalar observable and parameter coordinate")
        if ensemble is not None and (not isinstance(ensemble, np.ndarray) or ensemble.shape != () or ensemble.dtype.kind != "f"):
            raise ValueError("Estimator ensemble uncertainty has a foreign observable/parameter shape")
        target_precision = pub.metadata.get("target_precision")
        if target_precision is not None and target_precision != estimate.precision:
            raise ValueError("Estimator result has a different original target precision")
        from math import isfinite
        raw_error, raw_ensemble = (None if error is None else float(error)), (None if ensemble is None else float(ensemble))
        if any(item is not None and (not isfinite(item) or item < 0) for item in (raw_error, raw_ensemble)):
            raise ValueError("provider-reported uncertainty must be finite and nonnegative")
        # The meaning of stds depends on whether ZNE was applied. Preserve raw
        # provider uncertainties even when a sampling standard error is unavailable.
        options_text = native.metadata["requested_options_json"]
        run.check_data(4 * len(options_text))
        requested = json.loads(options_text)
        resilience = job_metadata.get("resilience")
        zne = resilience.get("zne_mitigation") if isinstance(resilience, dict) else None
        if type(zne) is not bool:
            zne = requested.get("resilience", {}).get("zne_mitigation")
        if type(zne) is not bool and requested.get("resilience_level") == 0:
            zne = False
        reason = ("provider did not return stds" if error is None else
                  "provider stds reports extrapolation-fit uncertainty, stored separately from sampling standard error"
                  if zne is True else "provider did not establish the statistical meaning of stds" if zne is not False else None)
        metadata = dict(primitive=job_metadata, pub=pub.metadata,
                        reported_data=dict(stds=raw_error, ensemble_standard_error=raw_ensemble))
        stored = encode(metadata, run.limits.max_data_bytes)
        result = EstimateValue(estimate_id=estimate.content_id, value=float(value),
            standard_error=raw_error if reason is None else None, uncertainty_unavailable=reason,
            uncertainty_source=Source(name="IBM EstimatorV2 data.stds", version=version("qiskit-ibm-runtime"),
                domain="stds is SEM across twirls without ZNE, or extrapolation-fit uncertainty with ZNE; no physical bias bound",
                reference="https://quantum.cloud.ibm.com/docs/en/guides/estimator-input-output"),
            requested_options_json=native.metadata["requested_options_json"], provider_metadata_json=stored)
        return BackendRunResult(execution_mode=ExecutionMode.ESTIMATED,
            backend_target=self.target_for(native.metadata["observation"]),
            raw_output={"estimates": (result,)}, metadata={"native_job_id": job_id})

    def refresh(self, locator, natives, *, run):
        """Read the original job's status once and decode its PUBs after a terminal DONE.

        Pending states (INITIALIZING, QUEUED, VALIDATING and RUNNING) map to
        "acknowledged", ERROR and CANCELLED to terminal
        failure or cancellation, and any other state to "uncertain". PUBs that
        the Run has already published are skipped, and a decode error after
        some PUBs succeeded is returned with those results.
        """
        from nwqlib.backends.connection import BackendRefresh
        job = self._job(locator, run)
        primitive = "estimator" if natives[0].metadata["observation"].kind == "estimated_observable" else "sampler"
        if job.primitive_id != primitive:
            raise ValueError("retrieved IBM primitive differs from the original acquisition kind")

        status = job.status()
        if status in {"INITIALIZING", "QUEUED", "RUNNING", "VALIDATING"}:
            return BackendRefresh(status="acknowledged", provider_status=status)
        if status in {"ERROR", "CANCELLED"}:
            run._state["backend_context"]["jobs"].pop(locator.job_id, None)
            return BackendRefresh(status="failed" if status == "ERROR" else "cancelled", provider_status=status,
                                  failure="IBM job " + status)
        if status != "DONE":
            return BackendRefresh(status="uncertain", provider_status=str(status), failure="unrecognized IBM job state")
        # result() is called only after a terminal status; timeout=0 prevents a
        # status race from turning one refresh into a provider-queue wait.
        result = job.result(timeout=0)
        # launch tags the job nwqlib:<submission_id>, so the submission is
        # found by its id; the locator comparison rejects a foreign tag.
        submissions = run._state["submissions"]
        submission = next((submissions[tag[7:]] for tag in (job.tags or ())
                           if isinstance(tag, str) and tag.startswith("nwqlib:") and tag[7:] in submissions
                           and submissions[tag[7:]].locator == locator), None)
        completed = set() if submission is None else {
            item.result_key for item in submission.items if item.attempt in run._state["by_attempt"]}
        decoded = []
        try:
            for item in self._decode(result, natives, job_id=locator.job_id, run=run, completed=completed):
                decoded.append(item)
        except Exception as error:
            if run._state.get("storage_failed", False) or not decoded:
                raise
            # Keep the live job while local decoding is incomplete. The common
            # owner commits earlier PUBs, then raises this original exception.
            return BackendRefresh(status="completed", provider_status=status,
                                  results=tuple(decoded), error=error)
        # Terminal retrieval no longer needs the run-owned live job object;
        # keep the reduced statistics and durable locator for later consumers.
        run._state["backend_context"]["jobs"].pop(locator.job_id, None)
        return BackendRefresh(status="completed", provider_status=status,
                              results=tuple(decoded))

    def cancel(self, locator, *, run):
        """Request cancellation of the original job once. A later refresh reads the outcome."""

        try:
            self._job(locator, run).cancel()
        finally:
            # RuntimeJobV2.cancel marks its local status CANCELLED immediately.
            # Discard that optimistic status so refresh reads the actual server.
            run._state["backend_context"].get("jobs", {}).pop(locator.job_id, None)


__all__ = ["IBMRuntimeBackend"]
