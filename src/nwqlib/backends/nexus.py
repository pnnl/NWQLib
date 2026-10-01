"""Quantinuum Nexus H2 lifecycle using public qnexus operations.

Remote preparation steps are deliberately separate. The execution owner
(``nwqlib._remote_preparation``) commits an intent before upload or compile and
its acknowledgement before the next step. No method retries a create
operation or waits through the provider queue. Each remote resource is named
``nwqlib:<stage>:<identity>``, and each compile or execute job carries a
description of its programs, shots and target. Reconciliation finds the
original resource by exact name within the project and, for a job, checks
that description. docs/nexus.md gives the stage table and the qualification
scope.
"""

from dataclasses import dataclass, replace
from importlib.metadata import version
from itertools import islice
import json
from numbers import Integral
import re
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nwqlib.backends.connection import BackendRefresh, NativePreparation, circuit_layout, environment
from nwqlib.core.planning import ObservationSpec
from nwqlib.core.records import Nonnegative, PositiveInt, Record, Source, Text
from nwqlib.execution import CountsSampling, JobLocator, RegisterMap
from nwqlib.operators.access import Count


class NexusPreparationData(BaseModel):
    """Selected local circuit data before any Nexus upload.

    circuit_json is pytket's own Circuit.to_dict format, not a gate schema.
    bits lists (register, index) in increasing original global classical position.
    No circuit hash or mathematical validation is used when restoring this data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot: Text
    shots: PositiveInt
    circuit_json: Text
    bits: tuple[tuple[Text, Count], ...]
    quantum_layout: tuple[RegisterMap, ...]
    classical_layout: tuple[RegisterMap, ...]


@dataclass(frozen=True)
class _PreparedNexus:
    """One compiled Nexus program reference with the original named classical bits.

    program_json is the public CircuitRef JSON of the compiled program.
    bits lists (register, index) in increasing global classical position, the
    order the decoder passes to pytket.
    """

    program_json: str
    program_id: str
    snapshot: str
    shots: int
    bits: tuple
    classical_layout: tuple
    metadata: dict
    circuit: None = None  # Opaque provider reference; explicit inspection is unavailable.


class NexusBackend(Record):
    """Explicit Nexus project and H2 device, without import/login on construction.

    credential_name refers to a Nexus-linked credential, never a secret.
    Authentication uses the user's existing qnexus configuration. max_cost_hqc
    is per submitted program in Hardware Quantum Credits; None requests no cap.
    The Run bounds stored wrapper data, not HTTP buffering, billing or SDK RSS.
    qnexus 0.49 has no public per-request timeout; refresh avoids queue waits
    but one HTTP request can still block indefinitely.
    optimization_level is the Nexus compile level; a level other than the
    default 1 is recorded as the receipt exclusion "optimization_level", which
    marks a circuit compiled at a non-default level. The target has no derived
    roundoff constant, so state_error() gives no bound at any level.
    """

    qualification_notice: ClassVar[str] = (
        "Nexus H2 has offline SDK/converter qualification only; live compilation and execution are unqualified. "
        "qnexus 0.49 has no public per-request deadline, so one refresh can wait indefinitely. "
        "The qualified pandas 3.0.5 environment conflicts with qnexus' declared pandas<3 cap; "
        "bounded offline checks found no runtime failure. See docs/nexus.md before installing or running.")
    requires_prepared_payload: ClassVar[bool] = False
    kind: Literal["nexus"] = "nexus"
    supports_synchronous: ClassVar[bool] = False
    project: Text
    device: Text
    target_region: Literal["us", "sg"] = "us"
    credential_name: Text | None = None
    optimization_level: Annotated[Count, Field(le=3)] = 1
    max_cost_hqc: Nonnegative | None = None
    max_input_bytes: PositiveInt

    @model_validator(mode="after")
    def _h2(self):
        """Accept only H2 device names: ``H2-<n>``, with ``E`` or ``LE`` for the emulators.

        This adapter submits compiled pytket circuits (CircuitRef), which is the
        H2 route. Helios takes HUGR or QIR programs, a separate route.
        """
        if re.fullmatch(r"H2-[1-9][0-9]*(?:E|LE)?", self.device) is None:
            raise ValueError("Nexus CircuitRef adapter requires an H2 device; Helios HUGR/QIR is a separate route")
        return self

    def _sdk(self):
        """Import qnexus on first explicit use, with an install hint if it is missing."""
        from nwqlib._optional import optional_import
        return optional_import("qnexus", extra="nexus")

    def _remote_sdk(self):
        """Admit local auth presence without login, refresh, or token inspection."""
        sdk = self._sdk()
        if sdk.client.utils.is_managed_token_environment():
            return sdk
        client = sdk.client.get_nexus_client()  # Keep the selected singleton and in-memory login.
        cookies = getattr(client.auth, "cookies", ())
        if not any(name in {"myqos_oat", "myqos_id"} for name in cookies):
            raise ValueError("Nexus local authentication is unavailable; configure qnexus authentication before requesting remote work")
        return sdk

    def target_for(self, observation):
        """The H2 counts target. Every other readout is rejected before any SDK import."""
        if observation.kind == "trajectory":
            observation.reject_unsupported_schedule(f"Nexus {self.device}")
        from nwqlib.backends.capabilities import BackendCapability, BackendTarget
        if observation.kind != "counts":
            raise ValueError("Nexus H2 CircuitRef acquisition supports counts only")
        # H2's published per-job shot limit (docs/ENGINEERING_CONSTANTS.md,
        # "Provider and compiler policies"; revisit with a changed provider
        # contract or target). It applies to each program's pooled request
        # N_q = s*m_q, plan shots times the query's recorded multiplicity, and
        # every Experiment must satisfy it before the first preparation or
        # submission. prepare_static calls this method for every Experiment
        # before its first preparation, so an over-limit query is refused
        # before earlier queries spend provider credits. admit_batch still
        # validates the actual submitted inventory.
        if not 1 <= observation.shots <= 10_000:
            raise ValueError(
                f"Nexus H2 program requests {observation.shots} shots, "
                "outside the inclusive per-program range [1, 10000]"
            )
        return BackendTarget(name=self.device, provider=self.kind,
            capabilities=(BackendCapability.COUNTS, BackendCapability.HARDWARE_SUBMIT,
                          BackendCapability.NATIVE_GATE_TARGET), readouts=("counts",),
            description="Nexus H2 compiled CircuitRef; public SDK; no HTTP timeout guarantee")

    def _config(self):
        """Fixed H2 target configuration, also used to recognize original jobs.

        Leakage detection would add a qubit and a classical bit that the
        decoder does not expect, and post-processing would move end-of-circuit
        operations into classical processing of the returned results. All four
        optional flags are set explicitly, so ``_check_job`` can compare a
        retrieved job's configuration with this exact value.
        """
        return self._sdk().models.QuantinuumConfig(device_name=self.device,
            postprocess=False, leakage_detection=False, simplify_initial=False,
            attempt_batching=False)

    def _association(self, programs, shots, run):
        """Encode the job description that binds a remote job to its programs, shots and target.

        The same text is compared during reconciliation and refresh, so a job
        with the right name but another program or shot inventory is rejected.
        """
        return self._encode(dict(programs=programs, shots=shots, project=self.project,
            backend_config=self._config().model_dump(mode="json"),
            credential_name=self.credential_name, target_region=self.target_region,
            optimization_level=self.optimization_level, max_cost_hqc=self.max_cost_hqc), run)

    def _project(self):
        """The configured project, checked to be the one the service returned."""
        project = self._remote_sdk().projects.get(id=self.project)
        if str(project.id) != self.project:
            raise ValueError("Nexus returned a different project")
        return project

    def _encode(self, value, run, *, input_payload=False):
        """Encode JSON under the Run's data cap, and also under ``max_input_bytes`` for circuit input."""
        from nwqlib._run_journal import encode
        limit = min(self.max_input_bytes, run.limits.max_data_bytes) if input_payload else run.limits.max_data_bytes
        return encode(value, limit)

    def _program(self, text, run):
        """Parse a saved CircuitRef and require that it belongs to the configured project.

        ``4 * len(text)`` bounds the UTF-8 bytes of the text before it is parsed.
        """
        from qnexus.models.references import CircuitRef
        run.check_data(4 * len(text))
        program = CircuitRef.model_validate_json(text)
        if str(program.project.id) != self.project:
            raise ValueError("Nexus program belongs to a different project")
        return program

    def prepare_local(self, circuit, *, observation, runtime, position,
                      source_definitions, run, snapshot):
        """Convert one selected circuit to pytket JSON locally, before any remote request.

        Every classical bit must belong to exactly one named register, because
        pytket names bits by (register, index) and the decoder requests results
        in that order. The Plan's runtime seed selects the u/cx lowering.
        """
        if observation.kind == "trajectory":
            observation.reject_unsupported_schedule(f"Nexus {self.device}")
        from qiskit import transpile
        from pytket.extensions.qiskit import qiskit_to_tk
        # target_for also refuses a shot count outside the per-program range.
        self.target_for(observation)
        # Public universal-basis lowering supplies the converter's documented
        # UGate/CXGate inputs, including empty/nested selected gate definitions.
        # Bare or overlapping classical bits cannot be named unambiguously in tket.
        bits = []
        for bit in circuit.clbits:
            registers = circuit.find_bit(bit).registers
            if len(registers) != 1:
                raise ValueError("Nexus requires each classical bit in exactly one named register")
            register, index = registers[0]
            bits.append((register.name, index))

        # Qiskit would lower a dense UnitaryGate with its inexact synthesis.
        # The Run makes the exact synthesis of each distinct matrix once per Run
        # object and reserves each new one against its max_synthesis_work first.
        lowered = transpile(run._exact_dense_unitaries(circuit),
                            basis_gates=["u", "cx"], optimization_level=0, seed_transpiler=runtime.seed)
        native = qiskit_to_tk(lowered)
        if {(bit.reg_name, tuple(bit.index)) for bit in native.bits} != {(name, (index,)) for name, index in bits}:
            raise ValueError("Qiskit converter changed the selected classical bit inventory")
        text = self._encode(native.to_dict(), run, input_payload=True)
        return NexusPreparationData(snapshot=snapshot, shots=observation.shots,
            circuit_json=text, bits=tuple(bits), quantum_layout=circuit_layout(circuit, circuit.qregs),
            classical_layout=circuit_layout(circuit, circuit.cregs))

    def restore_preparation(self, payload, *, run):
        """Decode the saved local preparation data, as ``prepare_local`` produced it."""
        run.check_data(4 * len(payload))
        return NexusPreparationData.model_validate_json(payload)

    @staticmethod
    def _name(stage, identity):
        """The exact resource name ``nwqlib:<stage>:<identity>`` that reconciliation searches for."""
        return f"nwqlib:{stage}:{identity}"

    def upload(self, data, *, preparation_id, run, acknowledge):
        """Upload the local circuit once and acknowledge its program ID immediately.

        ``acknowledge`` runs before the reference is serialized, so a later
        serialization failure is recovered through ``restore_upload`` by that ID
        and never by a second upload.
        """
        from pytket import Circuit
        project = self._project()
        run.check_data(4 * len(data.circuit_json))
        circuit = Circuit.from_dict(json.loads(data.circuit_json))

        ref = self._remote_sdk().circuits.upload(circuit=circuit, project=project,
            name=self._name("upload", preparation_id))
        acknowledge(str(ref.id))
        return self._encode(ref.model_dump(mode="json"), run)

    def restore_upload(self, program_id, *, run):
        """Read an already acknowledged uploaded program by its ID, without uploading again."""

        ref = self._remote_sdk().circuits.get(id=program_id)
        self._check_program(ref)
        if str(ref.id) != program_id:
            raise ValueError("Nexus retrieved a different uploaded program")
        return self._encode(ref.model_dump(mode="json"), run)

    def _check_program(self, ref):
        """Reject a returned reference that is not a CircuitRef of the configured project."""
        if ref.type != "CircuitRef" or str(ref.project.id) != self.project:
            raise ValueError("Nexus returned a foreign project/program type")

    def _locator(self, job):
        """Locator that records the job with its project, credential name and region."""
        return JobLocator(provider=self.kind, job_id=str(job.id), project=self.project,
                          account=self.credential_name, region=self.target_region)

    def start_compile(self, input_ref_json, *, preparation_id, run, acknowledge):
        """Start one remote compile job for the acknowledged upload and acknowledge its locator."""
        ref = self._program(input_ref_json, run)
        association = self._association([str(ref.id)], [], run)

        job = self._remote_sdk().start_compile_job(programs=[ref], project=ref.project,
            backend_config=self._config(), name=self._name("compile", preparation_id),
            description=association,
            optimisation_level=self.optimization_level, credential_name=self.credential_name,
            skip_intermediate_circuits=True)
        locator = self._locator(job)
        acknowledge(locator)
        return locator

    def _check_job(self, job, kind):
        """Reject a job from another project, of another type, or with another target configuration."""
        if (str(job.project.id) != self.project or job.job_type != kind
                or job.backend_config != self._config()):
            raise ValueError("Nexus job differs from the original project/type/target configuration")

    def _job(self, locator, kind):
        """Fetch the original job after checking that the locator matches this configuration."""
        if (locator.provider != self.kind or locator.project != self.project
                or locator.account != self.credential_name or locator.region != self.target_region):
            raise ValueError("Nexus job locator differs from the original project/credential")

        job = self._remote_sdk().jobs.get(id=locator.job_id)
        if str(job.id) != locator.job_id:
            raise ValueError("Nexus retrieved a different job")
        self._check_job(job, kind)
        return job

    def _unique(self, stage, identity, association=None):
        """Find the one project resource with the original exact name, or None.

        Reading at most two matches is enough to tell unique from ambiguous.
        Ambiguity and a foreign match raise, and neither case authorizes a
        replacement upload, compile or execution.
        """
        sdk = self._remote_sdk()
        name = self._name(stage, identity)
        project = self._project()
        # At most two matches and at most two public iterator page reads.

        api = sdk.circuits if stage == "upload" else sdk.jobs
        kwargs = {} if stage == "upload" else {"job_type": [stage]}
        matches = tuple(islice(api.get_all(name_exact=[name], project=project,
                                         page_size=2, **kwargs), 2))
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("Nexus correlation is ambiguous; no replacement submission is authorized")
        ref, = matches
        if str(ref.project.id) != self.project or ref.annotations.name != name:
            raise ValueError("Nexus correlation returned a foreign resource")
        if stage == "upload":
            self._check_program(ref)
        else:
            # JobRef.backend_config may fetch once.
            self._check_job(ref, stage)
            if ref.annotations.description != association:
                raise ValueError("Nexus correlation differs from the original submitted program/shot association")
        return ref

    def reconcile_preparation(self, preparation_id, stage, *, run, input_ref_json=None):
        """Recover an upload or compile whose acknowledgement was lost, by exact name.

        A compile match must also carry the description built from the
        acknowledged input reference. None leaves the stage unresolved.
        """
        if stage not in {"upload", "compile"}:
            raise ValueError("unknown Nexus preparation stage")
        association = None
        if stage == "compile":
            if input_ref_json is None:
                raise ValueError("Nexus compile reconciliation requires its acknowledged input reference")
            source = self._program(input_ref_json, run)
            association = self._association([str(source.id)], [], run)
        ref = self._unique(stage, preparation_id, association)
        if ref is None:
            return None
        return self._encode(ref.model_dump(mode="json"), run) if stage == "upload" else self._locator(ref)

    def refresh_compile(self, locator, input_ref_json, load_data, *, run):
        """Refresh the original compile job and admit only its associated input/output program
        reference.

        ``load_data`` reads the saved local circuit data only after a completed
        compilation has been matched to its input, so queued refreshes read no
        payload. The compiled reference is opaque, so the receipt reports no
        native gate inventory or physical layout.
        """
        job = self._job(locator, "compile")

        status = self._sdk().jobs.status(job).status
        if status != "COMPLETED":
            return self._refresh_state(status)

        results = self._sdk().jobs.results(job, allow_incomplete=True)
        if len(results) != 1 or results[0].type != "CompilationResultRef":
            raise ValueError("Nexus completed compilation omits its original item")
        result = results[0]
        source = self._program(input_ref_json, run)
        if job.annotations.description != self._association([str(source.id)], [], run):
            raise ValueError("Nexus compile job differs from original program association")
        actual = result.get_input()
        self._check_program(actual)
        if str(actual.id) != str(source.id) or str(result.project.id) != self.project:
            raise ValueError("Nexus compilation output belongs to a different input")
        output = result.get_output()
        self._check_program(output)
        data = load_data()
        program_json = self._encode(output.model_dump(mode="json"), run)
        options = self._encode(dict(compiled_program=json.loads(program_json), bits=data.bits), run)
        native = _PreparedNexus(program_json, str(output.id), data.snapshot, data.shots, data.bits,
            data.classical_layout, dict(readout_population="unconditional"))
        return NativePreparation(native=native,
            target=Source(name=self.device, version=version("qnexus"), domain="Nexus H2 CircuitRef",
                          reference=f"Nexus project {self.project}; target region {self.target_region}"),
            compiler=Source(name="Nexus remote compile", version=version("qnexus"),
                domain=f"optimisation_level={self.optimization_level}; opaque provider artifact",
                reference=f"qnexus.start_compile_job; job {locator.job_id}"), native_basis=(),
            environment=environment(("nwqlib", "qiskit", "qnexus", "pytket", "pytket-qiskit")),
            quantum_layout=(), classical_layout=data.classical_layout, logical_to_native=(),
            operations=None, population="unconditional",
            transformation="exact dense-unitary synthesis and Qiskit u/cx lowering at optimization_level=0 followed by pytket conversion and Nexus H2 compilation; original named classical bits",
            counts_sampling=CountsSampling(kind="unknown" if self.device.endswith("E") else "fresh"),
            provider_options_json=options,
            probability_window_exclusions=("optimization_level",) if self.optimization_level != 1 else None)

    def restore_native(self, record, payload=None, *, run):
        """Restore the compiled program reference and bit order from the saved receipt alone."""
        if record.provider_options_json is None:
            raise ValueError("Nexus receipt needs the original compiled program reference")
        text = record.provider_options_json
        run.check_data(4 * len(text))
        options = json.loads(text)
        program_json = self._encode(options["compiled_program"], run)
        bits = tuple((name, index) for name, index in options["bits"])
        return _PreparedNexus(program_json, options["compiled_program"]["id"], record.snapshot, record.observation.shots, bits,
                             record.classical_layout, dict(readout_population=record.population))

    def admit_batch(self, natives):
        """Admit at most 300 distinct Nexus items with 1 to 10,000 shots each.

        Nexus defines an execute job as a list of program items, each with
        its own n_shots and max_cost. The 300-program boundary comes from its
        Jobs in Nexus guide. The H2 shot cap is applied to each item, rather
        than being presented as an aggregate limit for a Nexus batch.
        """

        if not natives or len(natives) > 300:
            raise ValueError("Nexus execute jobs require a nonempty inventory of at most 300 programs")
        if any(not 1 <= native.shots <= 10_000 for native in natives):
            raise ValueError("Nexus requires 1 to 10000 shots per program")
        # Public results identify input programs. Repeated references are ambiguous
        # unless a separately documented job-item ordinal contract is available.
        ids = [native.program_id for native in natives]
        if len(set(ids)) != len(ids):
            raise ValueError("Nexus batch requires distinct compiled program references")

    def launch(self, natives, *, submission_id, run):
        """Start one execute job with one HQC cap per program and return its locator.

        ``max_cost_hqc`` is repeated for each program because the Nexus API
        takes one cap per submitted program, not one total for the job.
        """
        self.admit_batch(natives)
        refs = [self._program(native.program_json, run) for native in natives]
        association = self._association([str(ref.id) for ref in refs], [n.shots for n in natives], run)

        job = self._remote_sdk().start_execute_job(programs=refs, n_shots=[native.shots for native in natives],
            backend_config=self._config(), project=refs[0].project, name=self._name("execute", submission_id),
            description=association,
            credential_name=self.credential_name, valid_check=True,
            target_region=self.target_region,
            max_cost=[self.max_cost_hqc]*len(refs))
        return self._locator(job)

    def reconcile(self, submission, natives, *, run):
        """Find a lost execute acknowledgement by exact job name and program/shot description."""
        self.admit_batch(natives)
        association = self._association([n.program_id for n in natives],
                                        [n.shots for n in natives], run)
        job = self._unique("execute", submission.submission_id, association)
        return None if job is None else self._locator(job)

    @staticmethod
    def _refresh_state(status, results=()):
        """Map a Nexus job status to the common submission states.

        RETRYING and CANCELLING are still pending on the provider side.
        DEPLETED and TERMINATED end the job as failures. Unrecognized statuses
        stay uncertain.
        """
        status = str(status.value if hasattr(status, "value") else status)
        if status in {"CREATED", "SUBMITTED", "QUEUED", "RUNNING", "RETRYING", "CANCELLING"}:
            return BackendRefresh(status="acknowledged", provider_status=status, results=results)
        if status == "COMPLETED":
            return BackendRefresh(status="completed", provider_status=status, results=results)
        if status in {"ERROR", "FAILED", "DEPLETED", "TERMINATED", "CANCELLED"}:
            return BackendRefresh(status="cancelled" if status == "CANCELLED" else "failed",
                                  provider_status=status, results=results, failure="Nexus job " + status)
        return BackendRefresh(status="uncertain", provider_status=status, results=results,
                              failure="unrecognized Nexus job state")

    def refresh(self, locator, natives, *, run):
        """Read the original execute job and decode its terminal items.

        Each result is joined to its item by its input program reference, not
        by list position. Items still running are skipped because their shot
        sets can still grow. A result ID already saved for a consumed item is
        skipped before any download, and a changed result ID for a known item
        is rejected.
        """
        self.admit_batch(natives)
        job = self._job(locator, "execute")
        association = self._association([n.program_id for n in natives],
                                        [n.shots for n in natives], run)
        if job.annotations.description != association:
            raise ValueError("Nexus job differs from original submitted program/shot association")

        status = self._sdk().jobs.status(job).status

        refs = self._sdk().jobs.results(job, allow_incomplete=True)
        if len(refs) != len(natives):
            raise ValueError("Nexus result inventory differs from submitted program inventory")
        by_program = {native.program_id: (str(index), native)
                      for index, native in enumerate(natives)}
        # Untuned allowance for one downloaded result: len(bits) + 8 bytes per
        # requested shot of the largest item (docs/ENGINEERING_CONSTANTS.md).
        download_bytes = max(n.shots*(len(n.bits)+8) for n in natives)
        # launch names the job nwqlib:execute:<submission_id>, so the submission
        # is found by its id; the locator comparison rejects a foreign name.
        prefix = self._name("execute", "")
        name = job.annotations.name
        submission = (run._state["submissions"].get(name[len(prefix):])
                      if isinstance(name, str) and name.startswith(prefix) else None)
        if submission is not None and submission.locator != locator:
            submission = None
        items = {} if submission is None else {str(item.item): item for item in submission.items}
        completed = {item.provider_result_id: natives[item.item].program_id
            for item in items.values()
            if item.provider_result_id is not None and item.attempt in run._state["by_attempt"]}
        decoded, seen, assigned = [], set(), []
        try:
            for ref in refs:
                if str(ref.project.id) != self.project:
                    raise ValueError("Nexus result belongs to a different project")
                if ref.type == "IncompleteJobItemRef":
                    identity = str(ref.program_id)
                    if ref.program_type != "circuit" or identity not in by_program or identity in seen:
                        raise ValueError("Nexus incomplete item has an unknown or repeated original program")
                    seen.add(identity)
                    continue
                if ref.type != "ExecutionResultRef" or ref.result_type != "PYTKET":
                    raise ValueError("Nexus H2 requires PYTKET execution results")
                result_id = str(ref.id)
                known_program = completed.get(result_id)
                if known_program is not None:
                    if known_program in seen:
                        raise ValueError("Nexus result repeats an original program")
                    seen.add(known_program)
                    continue
                # Running partial shot sets can grow. get_input() also fetches
                # their result data, so skip them before either public accessor.
                item_status = ref.last_status_detail.status if ref.last_status_detail is not None else status
                if item_status not in {"COMPLETED", "CANCELLED", "ERROR", "DEPLETED", "TERMINATED"}:
                    continue
                # The program identity becomes available only after this download;
                # admit the largest original item until that association is known.
                run.check_data(download_bytes)

                program = ref.get_input()
                self._check_program(program)
                identity = str(program.id)
                if identity not in by_program or identity in seen:
                    raise ValueError("Nexus result has an unknown or repeated original program")
                seen.add(identity)
                key, native = by_program[identity]
                item = items.get(key)
                if item is not None:
                    if item.provider_result_id is not None and item.provider_result_id != result_id:
                        raise ValueError("Nexus result identity changed for an original acquisition")
                    run.check_data(len(result_id))
                    assigned.append(item.revise(provider_result_id=result_id))
                    if item.attempt in run._state["by_attempt"]:
                        continue
                result = ref.download_result()
                decoded.append((key, self._decode(result, native, locator.job_id, run)))
        except Exception as error:
            if run._state.get("storage_failed", False) or not (decoded or assigned):
                raise
            # The common owner commits earlier results/associations first,
            # then raises this exact exception. No failed result is published.
            return replace(self._refresh_state(status, tuple(decoded)),
                           associations=tuple(assigned), error=error)
        return replace(self._refresh_state(status, tuple(decoded)), associations=tuple(assigned))

    def _decode(self, result, native, job_id, run):
        """Translate one pytket BackendResult into NWQLib count keys over the original bits.

        Counts are requested jointly over all original bits, so registers are
        never marginalized separately. The count total may be below the
        requested shots and must not exceed it.
        """
        from pytket.backends.backendresult import BackendResult
        from pytket.circuit import Bit
        from nwqlib.backends.results import BackendRunResult
        from nwqlib.execution import ExecutionMode
        if not isinstance(result, BackendResult):
            raise TypeError("Nexus PYTKET result must be a BackendResult")
        bits = [Bit(name, index) for name, index in native.bits]
        if set(result.get_bitlist()) != set(bits):
            raise ValueError("Nexus result classical registers differ from original named bits")
        width = len(bits)
        # Untuned allowance, per requested shot, of 4 bytes per bit plus 8 for
        # the outcome tuples and count labels built below
        # (docs/ENGINEERING_CONSTANTS.md).
        run.check_data(native.shots * (4 * width + 8))
        counts = result.get_counts(cbits=bits)
        ordered, total = {}, 0
        for outcome, count in counts.items():
            if (len(outcome) != width or any(bit not in (0, 1) for bit in outcome)
                    or isinstance(count, bool) or not isinstance(count, Integral) or count < 0):
                raise ValueError("Nexus result contains invalid raw counts")
            total += int(count)
            if total > native.shots:
                raise ValueError("Nexus result exceeds requested shot population")
            # pytket cbits is the explicit low-to-high global bit sequence;
            # NWQLib count labels render global high-to-low order.
            label = "".join(str(int(bit)) for bit in reversed(outcome))
            ordered[label] = int(count)
        return BackendRunResult(execution_mode=ExecutionMode.SHOTS,
            backend_target=self.target_for(ObservationSpec(kind="counts", shots=native.shots)),
            raw_output={"counts": ordered}, metadata={"native_job_id": job_id})

    def cancel(self, locator, *, run):
        """Request cancellation of the original execute job once."""
        job = self._job(locator, "execute")

        self._sdk().jobs.cancel(job)

    def cancel_preparation(self, locator, *, run):
        """Request cancellation of the original compile job once."""
        job = self._job(locator, "compile")

        self._sdk().jobs.cancel(job)


__all__ = ["NexusBackend", "NexusPreparationData"]
