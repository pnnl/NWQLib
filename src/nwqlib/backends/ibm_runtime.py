"""IBM Quantum Runtime adapter for SamplerV2 counts and EstimatorV2 estimates, with jobs that continue after Python exits.

Jobs run on the client-side ``SamplerV2`` and ``EstimatorV2`` that
qiskit-ibm-runtime exports at its top level from 0.52.0 on. Each builds the
program of IBM's ``executor`` from its PUBs on this computer and sends it in
one request (``launch``). The SDK turns a returned executor result into
Sampler counts or Estimator values when the job is read, and qiskit-mitigation
computes the Estimator's values from the returned bits (``_estimate_result``).

Preparation compiles each circuit to the device's instruction set (an ISA
circuit, in IBM's terms) with the Plan's runtime seed and keeps the
logical-to-physical layout. Every submitted job carries the tag
``nwqlib:<submission UUID>``, which is how ``reconcile`` finds a job whose
creation acknowledgement was lost. Each PUB (primitive unified bloc, a circuit
with its parameter values and settings, the unit of input of an IBM primitive)
carries its original coordinate and the UUID of its
prepared circuit (``snapshot``) in circuit metadata, so decoding does not
depend on the order of returned PUBs. The adapter follows the rules in
``nwqlib.backends.connection``. docs/ibm.md records the offline qualification
scope.

Experiments whose circuits differ only in rotation angles
(``Method.row_templates``) share one compiled parameterized template. One
job sends the experiments of a template as one Sampler PUB with a
two-dimensional binding array, one row of values per experiment in job
order, and the PUB's circuit metadata lists the job items of its rows
(``nwqlib_items``). Each row keeps its own job item, attempt and shots, so a
PUB is not one acquisition. Decoding checks the whole PUB inventory and
every row's item against the item keys saved before the job was sent
(``result_keys``), then decodes the rows one at a time.

With fractional gates, a compiled template whose RZZ angles are parameters
does not run as compiled, because IBM accepts RZZ angles only in [0, pi/2].
Each experiment's angles are folded into that range with the same unitary
and global phase (``IBMRuntimeBackend.fold_template``), and the experiments
that need the same correction gates share one circuit built locally from the
compiled template (``template_class``), with its own snapshot and PUB.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from io import BytesIO
import json
import os
from typing import Annotated, ClassVar, Literal

from pydantic import Field, StrictBool, model_validator

from nwqlib.core.records import Record, Source, Text
from nwqlib.execution import CountsSampling, NativeDepth
from nwqlib.operators.access import Count
from nwqlib.backends._qpy_buffer import check_qpy_read, run_qpy_buffer


def sampler_decode_bytes(widths, shots):
    """Estimate the array bytes of joining and decoding one Sampler PUB, ``S*(9*Q + w + 32)``.

    Args:
        widths: Classical-register widths in join order. Their sum w can
            exceed 64.
        shots: Number S of returned shots, common to all registers.

    Q = sum(ceil(w_r/8)) is the packed bytes of one shot over all registers.
    Qiskit's ``BitArray.concatenate_bits`` unpacks each register to one byte
    per bit of its padded bytes, 8*Q per shot, joins the w bits and packs
    them into at most Q bytes. Up to 64 bits, the adapter then forms the
    joined integers, their gathered classical order and ``np.unique`` of
    them, about four 8-byte arrays per shot, which the 32 covers. With Qiskit
    2.6.0 and NumPy 2.5.2, the measured array peak of one decode was 43 to 83
    percent of this estimate for one to eight registers of 1 to 64 bits in
    total and 1,000 or 100,000 shots. Above 64 bits, the count dictionary of
    bit strings is Python objects, which this estimate does not include.
    """
    widths = tuple(widths)
    return shots * (9 * sum((width + 7) // 8 for width in widths) + sum(widths) + 32)


# The reasons that _estimate_result gives for an unavailable standard error:
# no stds, ZNE applied, and ZNE neither shown nor excluded.
_ESTIMATE_REASONS = (
    "provider did not return stds",
    "ZNE was applied, so the provider's stds is kept only in provider_metadata_json",
    "NWQLib did not determine whether ZNE was applied, so the provider's stds is kept only in provider_metadata_json",
)


def _estimate_value(estimate_id, value, error, reason, options_text, metadata_text):
    """The `EstimateValue` of one Estimator PUB, with the versions of the installed packages that compute its values.

    The client-side Estimator's values and stds are computed on this
    computer when the job is read, by qiskit-ibm-runtime with
    qiskit-mitigation.
    """
    from importlib.metadata import version
    from nwqlib.execution import EstimateValue
    return EstimateValue(estimate_id=estimate_id, value=value, standard_error=error, uncertainty_unavailable=reason,
        uncertainty_source=Source(name="IBM EstimatorV2 data.stds",
            version=f"qiskit-ibm-runtime {version('qiskit-ibm-runtime')}, "
                    f"qiskit-mitigation {version('qiskit-mitigation')}",
            domain="Provider-reported stds. NWQLib does not verify its statistical interpretation. "
                   "NWQLib's error budget excludes provider processing and hardware error.",
            reference="https://quantum.cloud.ibm.com/docs/en/guides/estimator-input-output"),
        requested_options_json=options_text, provider_metadata_json=metadata_text)


def _portable_metadata(value):
    """The JSON form of the provider metadata ``value``, with every key kept, for the Run's journal.

    The client-side Estimator's metadata holds the SDK's ``Metadata``
    dataclass, whose timing spans have datetimes, NumPy arrays and scalars,
    and tuples in its options, which ``nwqlib._run_journal.encode`` refuses.
    Dataclasses become dictionaries of their fields, dictionary keys text,
    tuples and arrays lists, NumPy scalars Python numbers and datetimes ISO
    8601 text. Any other object becomes its ``repr``, so a paid estimate is
    saved whatever objects its metadata holds.
    """
    import dataclasses
    from datetime import datetime
    import numpy as np
    if isinstance(value, dict):
        return {str(key): _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_metadata(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _portable_metadata(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, (np.ndarray, np.generic)):
        return _portable_metadata(value.tolist())
    if isinstance(value, datetime):
        return value.isoformat()
    return value if value is None or type(value) in (bool, int, float, str) else repr(value)


def _rzz_angles(circuit):
    """The distinct parameterized RZZ angles of a compiled circuit, each with its number of gates.

    Returns ``((expression, count), ...)`` in the order of each expression's
    first RZZ gate. Gates whose angles are equal Qiskit expressions share one
    entry. Only top-level gates are read: IBM does not run fractional gates
    in a circuit with control flow (qiskit-ibm-runtime 0.52.0,
    ``utils.is_fractional_gate``), and the SDK's RZZ check before a job is
    recorded (``IBMRuntimeBackend.admit_batch``) still reads nested gates.
    """
    from qiskit.circuit import ParameterExpression
    counts = {}
    for instruction in circuit.data:
        angle = instruction.operation.params[0] if instruction.operation.name == "rzz" else None
        if isinstance(angle, ParameterExpression) and angle.parameters:
            counts[angle] = counts.get(angle, 0) + 1
    return tuple(counts.items())


def _folded_names(parameters, count):
    """Names of the ``count`` folded RZZ angles and of the global-phase parameter of a folded template.

    The prefix grows until no compiled parameter name in ``parameters``
    starts with it, so the new names differ from every compiled one, and a
    later process gives the same names for the same compiled template.
    """
    prefix = "nwqlib_rzz"
    while any(name.startswith(prefix) for name in parameters):
        prefix += "_"
    return tuple(f"{prefix}[{index}]" for index in range(count)), prefix + "_phase"


@dataclass(frozen=True)
class _PreparedIBM:
    """One ISA circuit with its original readout association.

    circuit is None after a result-only restore, when retrieval needs only the
    saved association. classical_layout is the logical register layout that the
    decoder restores. observable is the layout-mapped estimator observable, or
    None for counts. metadata["requested_options_json"] is the JSON of the
    effective primitive options (IBMRuntimeBackend._options), the request with
    the frozen counts defaults and the Estimator resilience level written in.
    It equals PreparedArtifact.provider_options_json, and admit_batch builds
    the options of a new job from it.

    For a template, circuit keeps its parameters, whose names parameters lists
    in the order of the binding columns, and values holds one experiment's
    row (None for the template circuit itself). The circuit is the compiled
    template or, with parameterized RZZ angles under fractional gates, a
    circuit folded from it for one correction class (``template_class``).
    The experiments of one such circuit share circuit and snapshot. A bound
    circuit has parameters () and values None.
    """

    circuit: object
    payload: bytes | None
    snapshot: str
    shots: int
    classical_layout: tuple
    metadata: dict
    observable: object | None = None
    parameters: tuple = ()
    values: tuple | None = None


class IBMRuntimeBackend(Record):
    """IBM Quantum Runtime connection for sampled counts and provider expectation estimates.

    Build it with keyword arguments, for example
    `IBMRuntimeBackend(device="ibm_example", instance="my-instance")`,
    and pass it as `backend=` to [`prepare`][nwqlib.scientist.prepare].
    `device` and `instance` are required. It needs the `ibm`
    extra. Counts use the client-side SamplerV2 of qiskit-ibm-runtime and a
    provider estimate its client-side EstimatorV2, which build each job on
    this computer and submit it to IBM's `executor` program. The Estimator's
    values and their reported uncertainty are computed on this computer by
    qiskit-mitigation when the job is read, and NWQLib keeps them as
    reported without verifying their statistical interpretation.
    Construction reads no credential and contacts no service. Preparation and
    retrieval use the saved SDK account named by `account_name`, or otherwise the
    token in the environment variable named by `token_env`, and no credential is
    saved in the configuration or the Run's files.

    Preparation compiles each circuit for the device with the Plan's runtime seed
    and keeps the logical-to-physical layout. The preparation record also keeps
    the native two-qubit depth of the compiled circuit
    ([`PreparedArtifact.native_depth`][nwqlib.execution.PreparedArtifact]). It
    is qualified only for counts whose options keep the provider's dynamical
    decoupling, gate twirling and measurement twirling off (`options_json`),
    and it describes the compiled circuit that NWQLib submits. It is never
    qualified for a provider estimate,
    because the Estimator measures the observable with circuits that the
    provider builds from the compiled one. A requested depth limit also
    refuses a circuit when the installed nwqlib, Qiskit or qiskit-ibm-runtime
    version differs from the one recorded at its preparation, as after a Run
    is reopened in another environment
    (`nwqlib.backends.inspection.check_native_depth_cap`).
    The Estimator's resilience level is
    zero unless `options_json` selects another, so no error mitigation runs
    implicitly, and the Plan sets the precision. A Run with a folder saves each
    prepared circuit as QPY (Qiskit's binary circuit format). Before it keeps the
    bytes, it compares their number with what is left of `max_data_bytes` and
    twice their number with the Method's `max_bytes` or the physical memory of
    the computer. The Run also bounds decoded arrays. SDK HTTP buffering, retries,
    compilation memory, the SDK's own QPY serialization, the SDK's building of
    each job and processing of its result on this computer, wall time and
    billing are not bounded. When a later result fails to
    decode, `Run.resume`, which `Run.wait` calls on each pass, keeps the
    results already decoded. A resumed read fetches the whole job payload
    again, and results already saved skip only the local decoding. NWQLib has
    tested this backend only offline, with SDK and
    transport checks, and has not tested live accounts, queues or devices. The
    [IBM Runtime guide](../ibm.md) describes jobs, retrieval and cancellation.

    When a Plan's experiments differ only in rotation angles, as the rows of
    `LCHS(estimator="sampled_pairs")` do, preparation compiles one
    circuit with those angles as parameters for each group of experiments
    with the same gates, and one job sends all unsubmitted experiments, each
    group as one Sampler PUB with one row of parameter values per experiment.
    Every experiment keeps its own preparation record (with its values in
    `PreparedArtifact.template`), attempt and requested shots, which count
    against the Run's limits as before. With `use_fractional_gates=True`, a
    compiled circuit whose RZZ angles are parameters is instead folded for
    each experiment (see `use_fractional_gates`), and the experiments that
    need the same correction gates share one circuit and one PUB, so the
    number of PUBs depends on the values. A reopened Run binds a later
    experiment to the saved compiled circuit, or folds that circuit for it,
    without compiling again.

    Attributes:
        kind: Fixed `"ibm"`, the backend type.
        device: Required. IBM device name.
        instance: Required. IBM instance. `"auto"` is rejected, because retrieval
            and job recovery compare it with the job's own instance.
        account_name: Default `None`. Name of a saved IBM SDK account. With
            `None`, the token is read from the variable `token_env` names.
        token_env: Default `"NWQLIB_IBM_RUNTIME_TOKEN"`. Environment variable that
            holds the API token, read only during preparation or retrieval.
        mode: Default `"job"`. `"batch"` or `"session"` attaches every job to the
            existing Batch or Session whose ID is `container_id`.
        container_id: Default `None`. ID of an existing Batch or Session. Required
            in batch and session mode, and must be `None` in job mode.
        calibration_id: Default `None`. Device calibration ID passed to the IBM
            SDK when the device is opened.
        options_json: Default `"{}"`. JSON object of the requested primitive
            options. It states the requested options, not the server's defaults,
            and the configuration keeps it as given. For counts, preparation
            sets each of `dynamical_decoupling.enable`, `twirling.enable_gates`
            and `twirling.enable_measure` that the request leaves unset to
            `false`, the client-side Sampler's default. With twirling on,
            `twirling.num_randomizations` and
            `twirling.shots_per_randomization` must both be integers whose
            product is the requested shots, so that the Sampler returns that
            number of shots. Preparation refuses `environment.private=true`,
            whose result IBM returns only once. A request that the SDK
            refuses while it builds a job raises only after the submission
            is recorded, and the Run keeps that submission as uncertain with
            its usage counted. The preparation record keeps these effective options
            ([`PreparedArtifact.provider_options_json`][nwqlib.execution.PreparedArtifact]),
            and every job of the preparation is sent with them, also from a
            reopened Run with another installed SDK, whose own validation can
            still reject them. An option set to `true` stays on, and the
            native depth of that preparation is then unqualified.
            Experimental option overrides are rejected, because they can
            bypass the requested readout and its uncertainty.
        optimization_level: Default `1`. Qiskit preset pass-manager level, 0 to 3.
            A level other than 1 adds the
            [roundoff exclusion](../glossary.md#roundoff-exclusion)
            `"optimization_level"` to the
            [preparation record](../glossary.md#preparation-record), and the
            operation count describes the compiled circuit. The IBM target has
            no derived roundoff constant, so
            [`PreparedArtifact.state_error()`][nwqlib.execution.PreparedArtifact.state_error]
            gives no bound at any level.
        initial_layout: Default `None`. Physical qubit for each logical qubit,
            without repeats.
        use_fractional_gates: Default `False`. `True` opens the device with
            IBM's fractional gates, RX(θ) at any angle and RZZ(θ) with
            0 ≤ θ ≤ π/2, and compiles for that device target, whose RZZ angle
            bounds make Qiskit fold a bound RZZ angle into that range. A device
            without them is refused when it is opened. Preparation refuses
            options that IBM documents as
            unsupported with fractional gates: gate twirling, probabilistic
            error cancellation and noise amplification by probabilistic error
            amplification (PEA), including Estimator resilience level 2, which
            turns gate twirling on. The compiler cannot fold an RZZ angle that
            is a parameter of a compiled template (`Method.row_templates`), so
            preparation folds each experiment's angle θ = kπ + r, with
            -π/2 ≤ r ≤ π/2 and integer k, without compiling again: RZZ(|r|)
            on the same two qubits, with an X gate on one of them before and
            after it when r < 0, then RZ(π) on each of them when k is odd,
            and kπ/2 added to the circuit's global phase. The folded circuit
            has the same unitary as the compiled one, global phase included.
            Preparation refuses a device target without these gates on the
            qubits of such an RZZ, or without RZZ angles 0 to π/2 there.
            Before a job is recorded, every row of parameter values is
            checked with the SDK's RZZ angle check (`validate_rzz_pubs`). The
            compiled and folded circuits are saved as they were made, so a
            reopened Run uses the same choice.

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
    supports_template_pubs: ClassVar[bool] = True
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
    use_fractional_gates: StrictBool = False

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
        """Accept counts or a provider estimate before any contact with the service.

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

    def estimate_bytes(self, record, *, max_bytes):
        """Bound the JSON of the `EstimateValue` that a prepared estimate saves, except its provider metadata.

        Every other field is known before submission: the estimate's identity,
        a value and a standard error, or instead the longest reason that
        `_estimate_result` can give, the uncertainty source with the installed
        qiskit-ibm-runtime and qiskit-mitigation versions and the options
        text saved with the preparation.
        ``readout_bytes`` reserves this bound with the declaration. The
        provider metadata is counted at its actual size when the outcome is
        saved (``Run._write``).
        """
        from nwqlib._run_journal import _json_bound
        identity, options = record.observation.estimate.content_id, record.provider_options_json
        # The metadata holds one placeholder character, since the field is nonempty text.
        return max(_json_bound(_estimate_value(identity, 0., error, reason, options, "x"), max_bytes)
                   for error, reason in ((0., None), (None, max(_ESTIMATE_REASONS, key=len))))

    def _new_service(self):
        """Open an SDK service from the saved account or, otherwise, the token variable.

        The credential is read here, at the first explicit action, and is never
        saved with the configuration or in the Run's files. The service object
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
        """The Run's one service object, opened on first use and never saved."""
        if "service" not in context:
            context["service"] = self._new_service()
        return context["service"]

    def _backend(self, context):
        """The Run's device object, opened with the configured fractional-gate choice and checked to be the configured device.

        qiskit-ibm-runtime 0.52.0 raises ``IBMInputValueError`` when
        ``use_fractional_gates`` is True for a device without ``rx`` and
        ``rzz``.
        """
        if "device" not in context:
            context["device"] = self._service(context).backend(self.device, instance=self.instance,
                calibration_id=self.calibration_id, use_fractional_gates=self.use_fractional_gates)
        backend = context["device"]
        if backend.name != self.device:
            raise ValueError("IBM service returned a different device")
        return backend

    def _options(self, run, kind):
        """Build the canonical JSON of the effective primitive options and the reason they leave
        the native depth unqualified.

        The effective options are the request of ``options_json`` with the
        defaults below written in, validated by the installed SDK. ``prepare``,
        the only caller, saves their JSON with the preparation, and
        ``admit_batch`` builds every job's options from that saved JSON
        without writing in any default again, so a job is sent with the
        options recorded at preparation, also after the Run is reopened with
        another SDK. Estimator precision belongs to the Plan and
        cannot be overridden here. Resilience level zero is the Estimator
        default so the wrapper does not start mitigation work unasked
        (ENGINEERING_CONSTANTS.md, "IBM primitive preparation"). The options
        of a private job (``environment.private``) are refused, because IBM
        returns its result only once.

        For counts, each of ``dynamical_decoupling.enable``,
        ``twirling.enable_gates`` and ``twirling.enable_measure`` holds its
        SDK-validated Boolean, with an unset one written as False. The reason
        is None only for counts with all three False. Otherwise it is the text
        that the preparation records in place of the native depth.
        """
        from nwqlib._optional import optional_import
        options_module = optional_import("qiskit_ibm_runtime.options_models", extra="ibm")
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
                values["resilience_level"] = 0
            # The installed SDK validates the request before it is saved.
            options = options_module.EstimatorOptions(**values)
        else:
            options = options_module.SamplerOptions(**values)
        if self.use_fractional_gates:
            self._refuse_fractional_options(options, kind)
        # IBM returns the result of a private job only once
        # (EnvironmentOptions.private), so a Run could not read a paid result
        # again after a failed or interrupted retrieval.
        if options.environment.private:
            raise ValueError("environment.private=True lets IBM return a job's result only once, so a Run could not "
                             "read it again after a failed or interrupted retrieval; leave it false")
        # The native depth of the ISA circuit describes the execution only when
        # the provider runs that circuit unchanged. The same unitary is not
        # enough: gate folding turns one CX, of two-qubit depth 1, into three
        # consecutive CX gates on the same qubits, whose product is that CX
        # but whose depth is 3. Dynamical decoupling and gate or measurement
        # twirling change the gates of the circuit, and even added one-qubit
        # gates change its total depth.
        if kind == "estimated_observable":
            # Even at resilience level 0 the Estimator measures the observable
            # with circuits that the provider builds from the submitted one.
            unqualified = ("an IBM Estimator preparation has no qualified native depth, because the provider "
                           "measures the observable with circuits that it builds from the submitted circuit, and "
                           "runs mitigation circuits when they are selected, and the Run neither saves nor checks "
                           "these circuits; counts (SamplerV2) with dynamical decoupling and twirling off can have "
                           "a qualified depth")
        else:
            # The counts decoder reads classified bit strings, so another
            # measurement type is refused.
            if options.execution.meas_type != "classified":
                raise ValueError("counts require classified IBM measurement results")
            # The client-side Sampler applies these three options on this
            # computer when it builds the job. An unset twirling option is None,
            # which the Sampler takes as False (finalize_sampler_options), and
            # dynamical decoupling defaults to False. Each unset option is
            # written as an explicit False, so the saved JSON states the
            # values with which every job of the preparation is built, and an
            # option set to True stays on.
            enabled = []
            for section, name in (("dynamical_decoupling", "enable"), ("twirling", "enable_gates"),
                                  ("twirling", "enable_measure")):
                value = getattr(getattr(options, section), name) is True
                values.setdefault(section, {})[name] = value
                if value:
                    enabled.append(f"{section}.{name}=True")
            unqualified = None if not enabled else (
                "a qualified native depth needs the IBM Sampler options dynamical_decoupling.enable, "
                "twirling.enable_gates and twirling.enable_measure all false, so that the provider runs "
                "the submitted circuit without changing its gates, but this request has " + ", ".join(enabled))
        # UTF-8 JSON, the serialization that encode() admitted above and that
        # the Run journal and record identities write, so the stored text never
        # exceeds the admitted size. ASCII escaping could take up to 12 bytes
        # for a character outside the Basic Multilingual Plane.
        return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False,
                          ensure_ascii=False), unqualified

    @staticmethod
    def _refuse_fractional_options(options, kind):
        """Refuse primitive options that IBM documents as unsupported with fractional gates.

        IBM's fractional-gate guide
        (https://quantum.cloud.ibm.com/docs/en/guides/fractional-gates) lists
        Pauli (gate) twirling, probabilistic error cancellation (PEC) and
        zero-noise extrapolation by probabilistic error amplification (PEA)
        as unsupported, and measurement twirling (TREX) as supported. The
        effective values follow the defaults that the client-side option
        docstrings of qiskit-ibm-runtime 0.52.0 state, where an unset option
        is None: Sampler gate twirling is off unless set; for the Estimator an
        unset gate twirling or ZNE is off at resilience levels 0 and 1 and on
        at level 2, and PEC is off unless set. ``_options`` writes resilience
        level 0 into an Estimator request that leaves it unset.
        """
        def effective(value, level_two):
            return value if type(value) is bool else level_two
        level_two = kind == "estimated_observable" and options.resilience_level == 2
        found = []
        if effective(options.twirling.enable_gates, level_two):
            found.append("gate twirling (twirling.enable_gates" + (", on at resilience level 2)"
                                                                   if options.twirling.enable_gates is None else ")"))
        if kind == "estimated_observable":
            if options.resilience.pec_mitigation is True:
                found.append("probabilistic error cancellation (resilience.pec_mitigation)")
            if effective(options.resilience.zne_mitigation, level_two) and options.resilience.zne.amplifier == "pea":
                found.append("noise amplification by PEA (resilience.zne.amplifier='pea')")
        if found:
            raise ValueError("IBM fractional gates are unsupported with " + ", ".join(found)
                             + "; turn these options off or open the device with use_fractional_gates=False")

    @staticmethod
    def _observable(estimate, layout, width, run):
        """Map the full weighted logical observable onto the ISA circuit's physical qubits.

        The byte check and the exact-zero simplification are described in
        ENGINEERING_CONSTANTS.md, "IBM primitive preparation". The mapped
        observable of the most recent (estimate content hash, layout, width) is
        kept in the open Run's in-memory state and shared by consecutive points that
        repeat that key. A new key replaces it, and a reopened Run builds it
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
        """Build and check one layout-mapped ``ObservablesArray`` (``_observable``)."""
        from qiskit.primitives.containers import ObservablesArray
        from qiskit.quantum_info import SparseObservable, SparsePauliOp
        terms, logical_width = len(estimate.labels), len(estimate.labels[0])
        run.check_data(16 * terms * (logical_width + width) + 128 * terms)
        operator = SparsePauliOp.from_list(list(zip(estimate.labels, estimate.coefficients, strict=True)))
        native = operator.apply_layout(list(layout), num_qubits=width)
        # Qiskit 2.6's default ObservablesArray coercion simplifies at 1e-8,
        # silently deleting smaller nonzero coefficients. The common declaration
        # already validates finite real unique Pauli terms. Use the public
        # prevalidated-input API with exact-zero simplification. PUB validation
        # still checks the observable/circuit widths and parameter coordinates.
        sparse = SparseObservable.from_sparse_pauli_op(native).simplify(0.)
        return ObservablesArray(sparse, num_qubits=width, validate=False)

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot):
        """Compile the circuit of the parameter-bound point to an IBM ISA circuit, keeping the
        logical readout and observable layout.

        The Plan's runtime seed selects the transpiler seed, so compilation is
        repeatable for the same device target. The UUID of the prepared circuit
        (``snapshot``) is written into circuit metadata and later checked on
        every returned PUB. The named classical
        registers must survive compilation unchanged, because the decoder joins
        them per shot and restores the original bit positions. The circuit is
        saved as QPY before submission only when the Run has a folder on disk.
        The preparation record keeps the effective primitive options
        (``_options``) and the native two-qubit depth of the ISA circuit on the
        device target (``inspection.inspect_native_depth``). When those options
        or the Estimator let the provider change the circuit, the record keeps
        an unqualified depth with the reason instead, and the circuit is not
        scanned.

        Counts with twirling are refused here unless
        ``twirling.num_randomizations`` times
        ``twirling.shots_per_randomization``, both integers, equals the
        requested shots, because otherwise the client-side Sampler chooses
        the split and can return another number of shots, which shows only
        after the job is paid.

        A counts template from the Run's template preparation
        (``_prepared_execution._bind_template``) keeps its parameters through
        compilation. ``NativePreparation.parameters`` lists their names in the
        order of the binding columns, and ``bind_template`` gives each
        experiment's native object. When the compiled template keeps
        parameterized RZZ angles, it is the compiled template from which
        ``template_class`` builds the folded circuits that run, and its QPY
        is saved once for the later preparation of other experiments.
        """
        context = run._state["backend_context"]
        from importlib.metadata import version
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from nwqlib.backends.connection import NativePreparation, circuit_layout, environment

        self.target_for(observation)
        options_json, unqualified = self._options(run, observation.kind)
        twirling = json.loads(options_json).get("twirling", {}) if observation.kind == "counts" else {}
        if twirling.get("enable_gates") or twirling.get("enable_measure"):
            # A twirled Sampler returns num_randomizations * shots_per_randomization
            # shots of each row when both are integers, and with an automatic
            # split possibly another number. Only the requested number is
            # accepted here, because the decoder would refuse more only after
            # the job is paid (_decode) and would keep fewer as the counts.
            split = (twirling.get("num_randomizations"), twirling.get("shots_per_randomization"))
            if any(type(value) is not int for value in split) or split[0] * split[1] != observation.shots:
                raise ValueError(
                    f"IBM twirling needs twirling.num_randomizations and twirling.shots_per_randomization in "
                    f"options_json as integers whose product is the {observation.shots} shots of each experiment, "
                    "for example 16 and 64 for 1024 shots, because the Sampler otherwise chooses the split and can "
                    "return another number of shots")
        backend = self._backend(context)
        if backend.name != self.device or circuit.num_qubits > backend.num_qubits:
            raise ValueError("logical circuit differs from the selected IBM device or exceeds its width")
        if observation.kind == "estimated_observable" and (circuit.num_clbits or any(
                instruction.operation.name in {"measure", "reset", "if_else", "for_loop", "while_loop", "switch_case"}
                for instruction in circuit.data)):
            raise ValueError("provider estimation requires the selected unmeasured final state")
        logical_classical = circuit_layout(circuit, circuit.cregs)
        if self.initial_layout is not None and (len(self.initial_layout) != circuit.num_qubits
                or max(self.initial_layout, default=-1) >= backend.num_qubits):
            raise ValueError("explicit IBM layout differs from the original logical/physical widths")
        # Qiskit would lower a dense UnitaryGate with its inexact synthesis,
        # so levels 0 and 1 receive the exact synthesis. The Run makes it for
        # each distinct matrix once per Run object and records the work of each
        # new one first. Levels 2 and 3 resynthesize two-qubit
        # blocks with Qiskit's own synthesis, so they keep the logical circuit
        # as given (docs/dependency_issues.md).
        lowered = (run._exact_dense_unitaries(circuit)
                   if self.optimization_level <= 1 else circuit)
        # A template (Method.row_templates) keeps its angles as parameters,
        # which each experiment's row of values binds in the job. Compilation
        # must keep exactly those parameters, or the rows saved for them would
        # bind another circuit. A bound circuit has none.
        if circuit.num_parameters and observation.kind != "counts":
            raise ValueError("IBM parameterized templates are prepared for counts (SamplerV2) only")
        # With fractional gates the device target declares RZZ angle bounds,
        # and Qiskit 2.6.0's preset translation stage (WrapAngles) folds each
        # bound RZZ angle into [0, pi/2]. A parameterized angle stays as it is,
        # and each experiment's value is folded later (fold_template).
        native = generate_preset_pass_manager(backend=backend, optimization_level=self.optimization_level,
            seed_transpiler=runtime.seed, initial_layout=None if self.initial_layout is None else list(self.initial_layout)
            ).run(lowered)
        parameters = tuple(parameter.name for parameter in native.parameters)
        if sorted(parameters) != sorted(parameter.name for parameter in circuit.parameters):
            raise ValueError("IBM compilation changed the template's parameters, so its saved rows of values "
                             "would bind another circuit")
        # Verify named classical registers after compilation, then map the
        # weighted observable with the actual logical-to-native layout.
        classical = circuit_layout(native, native.cregs)
        bits = tuple(bit for register in classical for bit in register.bits)
        if sorted(bits) != list(range(native.num_clbits)):
            raise ValueError("IBM counts require each classical bit in exactly one named result register")
        if {reg.name: len(reg.bits) for reg in classical} != {reg.name: len(reg.bits) for reg in logical_classical}:
            raise ValueError("IBM ISA changed the named logical measurement registers")
        native.metadata = {**(native.metadata or {}), "nwqlib_snapshot": snapshot}
        # The native depth of this ISA circuit on the device target that
        # compiled it describes the execution only when the effective options
        # let the provider run the circuit unchanged (_options). Otherwise the
        # record keeps the width and the reason. A
        # Method's requested depth limit reads it before submission
        # (inspection.check_native_depth_cap).
        # The scan uses the inspection's default, which sets no operation
        # limit, so a long ISA circuit is scanned too. A template with
        # unbound parameters has no qualified depth, and its experiments get
        # the depth of their bound class circuit (template_depth), so it is
        # not scanned.
        from nwqlib.backends.inspection import inspect_native_depth
        native_depth = (inspect_native_depth(native, backend.target) if unqualified is None and not parameters
                        else NativeDepth(qualified=False, reason=unqualified or "the circuit has unbound parameters",
                                         width=native.num_qubits))
        payload = self._qpy(native, run)
        layout = tuple(native.layout.final_index_layout(filter_ancillas=True))
        metadata = dict(readout_population="unconditional", requested_shots=observation.shots,
                        logical_to_native=layout, observation=observation, requested_options_json=options_json)
        observable = (None if observation.estimate is None else
                      self._observable(observation.estimate, layout, native.num_qubits, run))
        prepared = _PreparedIBM(native, payload, snapshot, observation.shots, logical_classical, metadata, observable,
                                parameters=parameters)
        fractional = "fractional gates enabled" if self.use_fractional_gates else "fractional gates disabled"
        # The records of every circuit folded from this template keep this
        # description, so it states the folding instead of the count of
        # binding columns, which differs for a folded circuit.
        folded = sum(count for _, count in _rzz_angles(native)) if parameters else 0
        template = ("" if not parameters else
                    f"; parameterized template bound per experiment ({len(parameters)} parameters)" if not folded else
                    f"; parameterized template ({len(parameters)} parameters) whose {folded} parameterized RZZ "
                    "gates are folded per experiment into [0, pi/2] by X and RZ(pi) gates and a global-phase "
                    "parameter, then bound per experiment")
        return NativePreparation(native=prepared,
            target=Source(name=self.device, version=backend.backend_version,
                domain=f"IBM target configuration; calibration_id={backend.calibration_id or 'unavailable'}; "
                       f"{fractional}",
                reference="IBMQuantumComputeService.backend"),
            compiler=Source(name="Qiskit IBM ISA pass manager", version=version("qiskit"),
                domain=f"optimization_level={self.optimization_level}; original runtime seed selects compilation",
                reference="qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager"),
            native_basis=tuple(sorted(backend.operation_names)),
            environment=environment(("nwqlib", "qiskit", "qiskit-ibm-runtime")),
            quantum_layout=circuit_layout(native, native.qregs), classical_layout=classical,
            logical_to_native=layout, operations=len(native.data), population="unconditional",
            transformation=("exact dense-unitary synthesis, then " if self.optimization_level <= 1 else "")
                + "IBM ISA circuit; original logical/classical mapping and effective primitive options"
                + ("; layout-mapped weighted observable" if observable is not None else "") + template,
            parameters=parameters,
            payload=payload, payload_format="nwqlib.ibm.qpy/1",
            counts_sampling=(CountsSampling(kind="unknown" if backend.configuration().simulator else "fresh")
                             if observation.kind == "counts" else None), provider_options_json=options_json,
            probability_window_exclusions=("optimization_level",) if self.optimization_level != 1 else None,
            native_depth=native_depth)

    def _qpy(self, circuit, run):
        """The QPY bytes of a prepared circuit when the Run saves to a folder, otherwise None.

        The writes are checked against the memory and data limits of ``run_qpy_buffer``.
        """
        if not run._state["backend_context"].get("persist_prepared", False):
            return None
        from qiskit import qpy
        with run_qpy_buffer(run) as buffer:
            qpy.dump(circuit, buffer)
            return buffer.getvalue()

    def bind_template(self, native, values, *, runtime):
        """Return one experiment's native object for the template circuit of ``native``, with its row of ``values``, and no record fields of its own.

        The template circuit is the compiled template or a circuit folded
        from it (``template_class``). ``values`` follow ``native.parameters``.
        The experiments of one template circuit share its circuit object,
        snapshot and preparation-record fields, the runtime included, because
        the runtime seed only compiles the template, so ``runtime`` is unused.
        """
        if not native.parameters or len(values) != len(native.parameters):
            raise ValueError("an IBM template binding needs one value for each compiled parameter")
        return replace(native, values=tuple(values)), {}

    def template_depth(self, native, values, *, run):
        """The native depth of the experiments of the template circuit of ``native``, from the circuit bound to ``values``.

        ``values`` maps each parameter name of the circuit to its value in
        the first row of its experiments. The experiments run the same gates
        on the same qubits in the same order with their own values, so one
        scan of the bound circuit (``inspection.inspect_native_depth``)
        gives the depth of each of them when the target accepts every value
        of each parameterized gate: a gate whose target operation takes free
        parameters without angle bounds, or a folded RZZ of a class circuit
        (``template_class``), whose angles lie in [0, pi/2] by construction.
        Otherwise, and when the effective options let the provider change
        the circuit (``_options``), the depth is unqualified with that reason.
        """
        from qiskit.circuit import ParameterExpression
        from nwqlib.backends.inspection import inspect_native_depth
        circuit = native.circuit
        _, unqualified = self._options(run, "counts")
        target = self._backend(run._state["backend_context"]).target
        if unqualified is None:
            for name in dict.fromkeys(item.operation.name for item in circuit.data if any(
                    isinstance(value, ParameterExpression) for value in item.operation.params)):
                operation = target.operation_from_name(name) if name in target.operation_names else None
                if name != "rzz" and (operation is None or isinstance(operation, type) or target.gate_has_angle_bounds(name)
                                      or not all(isinstance(value, ParameterExpression) for value in operation.params)):
                    unqualified = f"the target's support of {name} depends on each experiment's parameter values"
                    break
        if unqualified is not None:
            return NativeDepth(qualified=False, reason=unqualified, width=circuit.num_qubits)
        return inspect_native_depth(circuit.assign_parameters({p: values[p.name] for p in circuit.parameters}), target)

    def fold_template(self, native, rows):
        """Fold each experiment's parameterized RZZ angles of the compiled template ``native`` into [0, pi/2].

        ``rows`` maps, for each experiment, every parameter name of the
        compiled template to its value. Returns None when the template has
        no parameterized RZZ angle (``_rzz_angles``), and otherwise, for each
        row, ``(correction, columns)``: the correction class
        (``TemplateBinding.correction``) and the value, by name, of every
        parameter that a circuit folded from this template can use. These
        are the compiled parameters, one folded angle for each distinct RZZ
        angle (``_folded_names``) and the global-phase parameter. The
        compiled template itself is not changed or copied here.

        Relation (Qiskit's convention RZZ(theta) = exp(-i theta P/2) with
        P = Z_a Z_b, so P**2 = I and X_a P X_a = -P). Write
        theta = k*pi + r with integer k and -pi/2 <= r <= pi/2, alpha = |r|,
        b = 1 if r < 0 else 0 and c = k mod 2. The commuting exponents give
        RZZ(theta) = RZZ(k*pi) RZZ(r). Since exp(-i*pi*P/2) = -i*P,
        RZZ(k*pi) = (-i)**k P**c, also for negative k. Conjugation by X_a
        negates the exponent, so RZZ(r) = X_a**b RZZ(alpha) X_a**b. With
        RZ_a(pi) RZ_b(pi) = -P and (-1)**c = (-1)**k,

            RZZ(theta) = exp(i*k*pi/2) [RZ_a(pi) RZ_b(pi)]**c X_a**b RZZ(alpha) X_a**b,

        where the rightmost factor acts first. On the eigenspace P = z,
        exp(i*k*pi/2) (-z)**c exp(-i*r*z/2) = exp(-i*theta*z/2), an
        independent check of the phase. Replacing the m_j gates of the j-th
        distinct angle adds the global phase d*pi/2 with
        d = (sum_j m_j (k_j mod 4)) mod 4, summed in integers, so no large
        floating sum of k*pi/2 is formed. The identity is exact in real
        arithmetic; a 2*pi shift of theta changes the full operator's sign,
        so folding modulo 2*pi alone would be wrong.

        Floating point. Each angle theta_hat is the binary64 value of the
        compiled expression at the row's values, and nonfinite or complex
        angles are refused. ``math.remainder(theta_hat, math.pi)`` is the
        exact IEEE remainder r_hat, with |r_hat| <= pi_hat/2 and ties to an
        even quotient, and k is recovered from that same remainder, so the
        sign, the parity and the residual come from one reduction. For the
        exact angle theta_j at the stored values, an evaluation error e_j,
        rho_j = theta_hat_j - k_j*pi_hat - r_hat_j and a global-phase
        evaluation error e_phase, the folded circuit's operator differs from
        the original's by at most
        (1/2) sum_j m_j (e_j + |k_j| |pi_hat - pi| + |rho_j|)
        + sum_j m_j c_j |pi_hat - pi| + e_phase in the spectral norm, because
        ||RZZ(u) - RZZ(v)|| <= |u - v|/2 and unitary products add their
        local differences. None of these terms is computed. The folding is
        checked at ordinary angles, including boundary crossings and the
        2*pi sign (tests/test_ibm_backend.py), not for every binary64 angle.
        Compiler and hardware errors are separate.
        """
        import math
        angles = _rzz_angles(native.circuit)
        if not angles:
            return None
        names, phase = _folded_names(native.parameters, len(angles))
        folded = []
        for row in rows:
            if sorted(row) != sorted(native.parameters):
                raise ValueError("the compiled template's parameters differ from the Method's template parameters")
            columns, digits, residue = dict(row), [], 0
            for (expression, count), name in zip(angles, names):
                theta = complex(expression.bind_all({parameter: row[parameter.name]
                                                     for parameter in expression.parameters}))
                if theta.imag or not math.isfinite(theta.real):
                    raise ValueError(f"the compiled IBM template's RZZ angle {expression} is {theta} for an "
                                     "experiment's parameter values, not a finite real angle, so it cannot be "
                                     "folded into [0, pi/2]. Nothing was submitted; the Method's parameter values "
                                     "must give finite real angles")
                remainder = math.remainder(theta.real, math.pi)
                winding = round((theta.real - remainder) / math.pi)
                columns[name] = abs(remainder)
                digits.append(str(2 * (winding % 2) + (remainder < 0)))
                residue += count * (winding % 4)
            columns[phase] = (residue % 4) * math.pi / 2
            folded.append(("".join(digits), columns))
        return tuple(folded)

    def template_class(self, native, correction, *, snapshot, run):
        """Build the circuit that the experiments of one correction class run, from the compiled template ``native``.

        Each parameterized RZZ gate of the compiled template, on physical
        qubits (a, b), becomes, in this order: X on a when the class's digit
        for its angle (``fold_template``) has b = 1, RZZ of a new angle
        parameter shared by the gates of equal angles, X on a again when
        b = 1, then RZ(pi) on a and on b when c = 1. The RZZ gate stays even
        for a zero angle. One new parameter is added to the compiled global
        phase, whose own expression is kept. Every other gate, the layout,
        the registers and the measurements stay as compiled, so the class
        circuit runs on the compiled layout without another compilation. The
        circuit's metadata names ``snapshot``.

        The device target must support RZZ on (a, b) with angles 0 to pi/2,
        X on a when the class needs it and RZ on a and b when it needs them.
        A missing gate refuses, and the SDK's ISA check
        (``validate_isa_circuits``) of the whole circuit follows. The QPY
        is checked like a compiled circuit's (``_qpy``).

        Returns ``(native, parameters, payload, operations, environment)``:
        the class's native object, its parameter names in binding-column
        order, its QPY bytes or None, its operation count and the package
        versions of this process.
        """
        import math
        from qiskit.circuit import Parameter
        from nwqlib._optional import optional_import
        from nwqlib.backends.connection import environment
        validate_isa_circuits = optional_import("qiskit_ibm_runtime.utils", extra="ibm").validate_isa_circuits
        angles = _rzz_angles(native.circuit)
        names, phase = _folded_names(native.parameters, len(angles))
        if len(correction) != len(angles):
            raise ValueError("an IBM correction class needs one digit for each parameterized RZZ angle")
        slots = {expression: (Parameter(name), int(digit))
                 for (expression, _), name, digit in zip(angles, names, correction, strict=True)}
        target = self._backend(run._state["backend_context"]).target
        missing = []
        if target.gate_has_angle_bounds("rzz") and not all(
                target.supported_angle_bound("rzz", [angle]) for angle in (0.0, math.pi / 2)):
            missing.append("RZZ angles from 0 to pi/2")
        circuit = native.circuit.copy_empty_like()
        for instruction in native.circuit.data:
            angle = instruction.operation.params[0] if instruction.operation.name == "rzz" else None
            if angle not in slots:
                circuit.append(instruction, copy=False)
                continue
            parameter, digit = slots[angle]
            first, second = instruction.qubits
            qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
            needed = [("rzz", qubits)] + [("x", qubits[:1])] * (digit & 1) + [
                ("rz", qubits[:1]), ("rz", qubits[1:])] * (digit >> 1)
            missing.extend(f"{name} on qubits {qargs}" for name, qargs in needed
                           if not target.instruction_supported(name, qargs))
            if digit & 1:
                circuit.x(first)
            circuit.rzz(parameter, first, second)
            if digit & 1:
                circuit.x(first)
            if digit >> 1:
                circuit.rz(math.pi, first)
                circuit.rz(math.pi, second)
        if missing:
            raise ValueError(f"folding the parameterized RZZ angles of an IBM template needs "
                             f"{', '.join(dict.fromkeys(missing))} on device {self.device}, which its target "
                             "does not provide. Nothing was submitted. Prepare the Plan in a new Run with "
                             "use_fractional_gates=False")
        circuit.global_phase = native.circuit.global_phase + Parameter(phase)
        circuit.metadata = {**(native.circuit.metadata or {}), "nwqlib_snapshot": snapshot}
        try:
            validate_isa_circuits([circuit], target)
        except ValueError as error:
            raise ValueError(f"the folded IBM template is not an ISA circuit of device {self.device}: {error}. "
                             "Nothing was submitted. Prepare the Plan in a new Run with "
                             "use_fractional_gates=False") from error
        payload = self._qpy(circuit, run)
        parameters = tuple(parameter.name for parameter in circuit.parameters)
        return (replace(native, circuit=circuit, payload=payload, snapshot=snapshot, parameters=parameters,
                        values=None), parameters, payload, len(circuit.data),
                environment(("nwqlib", "qiskit", "qiskit-ibm-runtime")))

    def restore_template(self, record, payload, *, run):
        """Load the compiled template that the folded preparation ``record`` names (``TemplateBinding.base``).

        ``payload`` holds its saved QPY. A later experiment of the template
        is folded from it (``fold_template``, ``template_class``) without
        compiling again. The Run keeps the result in memory, so this is
        called at most once per template and open Run.
        """
        from qiskit import qpy
        if record.template.base.format != "nwqlib.ibm.qpy/1":
            raise ValueError("a folded IBM template requires its original QPY payload reference")
        check_qpy_read(len(payload), run)
        (circuit,) = qpy.load(BytesIO(payload))
        return _PreparedIBM(circuit, payload, circuit.metadata["nwqlib_snapshot"], record.observation.shots,
            record.classical_layout, dict(readout_population=record.population, observation=record.observation,
                                         requested_options_json=record.provider_options_json),
            parameters=tuple(parameter.name for parameter in circuit.parameters))

    def restore_native(self, record, payload=None, *, run):
        """Rebuild the prepared association from a saved preparation record, loading QPY only when supplied.

        Retrieval of a known job needs no circuit, so ``payload`` is None there.
        A new submission from saved data needs the QPY. No recompilation happens
        here. The experiments of one template circuit (the compiled template,
        or one folded correction class) share one saved circuit, which is
        loaded once per open Run, and each keeps its row of values from
        ``record.template``. A folded record needs only its own circuit, not
        the compiled template.
        """
        if payload is not None and (record.payload is None or record.payload.format != "nwqlib.ibm.qpy/1"):
            raise ValueError("prepared IBM artifact requires its original QPY payload reference")
        circuit = None
        observable = None
        binding = record.template
        if payload is not None:
            from qiskit import qpy
            # Kept in the open Run's memory, like ibm_observable, not in the
            # backend context, whose entries are saved with the Run.
            templates = {} if binding is None else run._state.setdefault("ibm_templates", {})
            circuit = templates.get(record.payload.content_id)
            if circuit is None:
                check_qpy_read(len(payload), run)
                (circuit,) = qpy.load(BytesIO(payload))
                if binding is not None:
                    if tuple(parameter.name for parameter in circuit.parameters) != binding.parameters:
                        raise ValueError("saved IBM template parameters differ from their preparation record")
                    templates[record.payload.content_id] = circuit
            if record.observation.estimate is not None:
                observable = self._observable(record.observation.estimate, record.logical_to_native,
                                               circuit.num_qubits, run)
        return _PreparedIBM(circuit, payload, record.snapshot, record.observation.shots,
            record.classical_layout, dict(readout_population=record.population, observation=record.observation,
                                         requested_options_json=record.provider_options_json), observable,
            parameters=() if binding is None else binding.parameters,
            values=None if binding is None else binding.values)

    def admit_preparation(self, observation, *, classical_widths, run):
        """Check what the readout and this configuration fix against the Run's data limit, before the preparation is charged.

        The requested options are read as JSON, of at most 4 bytes per character.
        The decode scratch of a returned PUB grows with its shots, which never
        exceed the requested ones (refresh), so refresh's check of each row
        cannot refuse. The size of the QPY payload is not known yet, so it is
        checked when the SDK writes it (``_qpy``), after the charge.
        """
        run.check_data(4 * len(self.options_json))
        if observation.kind == "counts" and classical_widths:
            run.check_data(sampler_decode_bytes(classical_widths, observation.shots))

    def admit_batch(self, natives):
        """Accept the PUBs of one new primitive job and return the SDK options of that job.

        The job's structure and options are checked by ``_job_options``, and
        the RZZ angles of its bound template rows by the SDK's own check, here
        before the shared execution code saves the submission intent. It is
        called only for a new submission. Retrieval of a known job does not
        call it.
        """
        options = self._job_options(natives)
        if any(native.values is not None for native in natives):
            # The client-side SamplerV2 does not check RZZ angles (launch), so
            # this SDK check is the only one before the job is sent. The
            # binding arrays are built for it once, as float64 copies of the
            # values in the preparation records of the job's items, repeated
            # items included, with the circuits' actual columns. They take
            # eight bytes per value, less than each item's own admitted record
            # and circuit, so these transient arrays are not checked on their
            # own.
            from qiskit.primitives.containers.sampler_pub import SamplerPub
            from nwqlib._optional import optional_import
            validate_rzz_pubs = optional_import("qiskit_ibm_runtime.utils", extra="ibm").validate_rzz_pubs
            layout, _ = self._pub_layout(natives)
            pubs = [SamplerPub.coerce((natives[items[0]].circuit, self._binding_rows(natives, items),
                                       natives[items[0]].shots))
                    for items in layout if natives[items[0]].values is not None]
            try:
                validate_rzz_pubs(pubs)
            except ValueError as error:
                raise ValueError(f"an experiment's prepared parameter values give an IBM RZZ angle outside "
                                 f"[0, pi/2] in its prepared circuit. Preparation folds every parameterized RZZ "
                                 f"angle of a compiled template into that range, so this preparation record does "
                                 f"not match its circuit. Nothing was submitted. Prepare the Plan again in a new "
                                 f"Run. SDK check: {error}") from error
        return options

    def _job_options(self, natives):
        """Check the structure of one primitive job and return its SDK options.

        The PUBs share one primitive and one set of saved effective options.
        Estimator PUBs share one precision, and Sampler PUBs share one positive
        shot count. Each PUB has its prepared ISA circuit, which a Run restores
        from the saved QPY for a new submission. The options are built from the
        effective options saved at preparation (``_options``) without writing
        in any default again, so the installed SDK's validation can reject a
        saved request it does not support, also without a depth limit.
        """
        if not natives or len({native.metadata["observation"].kind for native in natives}) != 1:
            raise ValueError("one IBM job requires one primitive kind")
        if len({native.metadata["requested_options_json"] for native in natives}) != 1:
            raise ValueError("one IBM job requires matching original primitive options")
        kind = natives[0].metadata["observation"].kind
        if kind == "estimated_observable":
            if len({native.metadata["observation"].estimate.precision for native in natives}) != 1:
                raise ValueError("current IBM EstimatorV2 jobs require one common requested precision")
        elif any(native.shots < 1 for native in natives):
            raise ValueError("IBM sampled acquisitions require positive shots")
        elif len({native.shots for native in natives}) != 1:
            raise ValueError("one IBM Sampler job requires one shot count for all its circuits; "
                             "submit circuits with different shots in separate jobs")
        if any(native.circuit is None for native in natives):
            raise ValueError("new IBM submission requires the original native payloads")
        from nwqlib._optional import optional_import
        options_module = optional_import("qiskit_ibm_runtime.options_models", extra="ibm")
        # The saved text belongs to the preparation record, which passed the
        # Run's data limit when it was written (Run._write), so it is parsed
        # here without another check.
        values = json.loads(natives[0].metadata["requested_options_json"])
        return (options_module.EstimatorOptions if kind == "estimated_observable"
                else options_module.SamplerOptions)(**values)

    @staticmethod
    def _pub_layout(natives):
        """Group the items of one job into PUBs and name each item's result key.

        Returns ``(layout, keys)``. ``layout[p]`` lists the item indices of
        PUB p in binding order: one item for a bound circuit, and for a
        template every item with its snapshot, in job order, the template
        PUB taking the position of its first item. ``keys[i]`` is ``"p"``
        for item i alone in PUB p and ``"p:j"`` for row j of template PUB p.
        Without templates, PUB p is item p and its key is ``"p"``.
        """
        layout, templates, keys = [], {}, []
        for index, native in enumerate(natives):
            if native.values is None:
                keys.append(str(len(layout)))
                layout.append([index])
                continue
            if native.snapshot not in templates:
                templates[native.snapshot] = len(layout)
                layout.append([])
            pub = templates[native.snapshot]
            keys.append(f"{pub}:{len(layout[pub])}")
            layout[pub].append(index)
        return tuple(tuple(items) for items in layout), tuple(keys)

    def result_keys(self, natives):
        """The result key of each item of one job (``_pub_layout``), saved with the submission before ``launch``."""
        return self._pub_layout(natives)[1]

    @staticmethod
    def _binding_rows(natives, items):
        """The binding array of one template PUB: ``{parameter names: values}`` with one row per item."""
        import numpy as np
        from qiskit.primitives.containers.bindings_array import BindingsArray
        native = natives[items[0]]
        rows = np.array([natives[item].values for item in items], dtype=np.float64).reshape(len(items), -1)
        return BindingsArray({native.parameters: rows})

    def launch(self, natives, *, submission_id, run):
        """Submit one primitive job for the accepted PUBs and return its job locator.

        The job's structure is checked here too, and it is sent with the
        options that ``_job_options`` builds from the effective options saved
        at preparation, tagged ``nwqlib:<submission_id>`` for ``reconcile``.
        In job mode an ambient Session or Batch context is rejected, since
        the SDK would otherwise attach the job to that container. Batch and
        Session modes attach only to the configured existing container.
        ``primitive.run`` is called once. The SDK's own HTTP retries are
        described in docs/ibm.md.

        The primitives are the client-side ``SamplerV2`` and ``EstimatorV2``
        of qiskit-ibm-runtime 0.52.0 and later. Their ``run`` builds the
        program of IBM's ``executor`` from the PUBs on this computer, applying
        the twirling, dynamical decoupling and mitigation options there, and
        then sends one request. A refusal raised by ``run`` comes after the
        shared execution code has saved the submission intent, so the Run
        keeps that submission as uncertain with its usage counted. NWQLib's
        own earlier refusals are those of preparation (``_options`` and
        ``prepare``) and of ``admit_batch``. These primitives do not check the
        RZZ angles or the instruction set of the PUBs, so the RZZ check of
        ``admit_batch`` is the only one before a job is sent.
        """
        from nwqlib._optional import optional_import
        sdk = optional_import("qiskit_ibm_runtime", extra="ibm")
        Batch, Session, EstimatorV2, SamplerV2 = sdk.Batch, sdk.Session, sdk.EstimatorV2, sdk.SamplerV2
        from nwqlib.execution import JobLocator
        options = self._job_options(natives)
        kind = natives[0].metadata["observation"].kind
        tag = "nwqlib:" + submission_id
        options.environment.job_tags = [*(options.environment.job_tags or ()), tag]
        if self.mode == "job":
            mode = self._backend(run._state["backend_context"])
        else:
            cls = Batch if self.mode == "batch" else Session

            service = self._service(run._state["backend_context"])
            # The container opens its device with the fractional-gate choice
            # of preparation, so the target that the SDK reads when it builds
            # the job, for dynamical decoupling, is the one the circuits were
            # compiled for.
            mode = cls.from_id(self.container_id, service=service, calibration_id=self.calibration_id,
                               use_fractional_gates=self.use_fractional_gates)
            if (mode.backend() != self.device or service.active_instance() != self.instance
                    or mode.session_id != self.container_id):
                raise ValueError("existing IBM container belongs to another device/instance or has a different ID")
        primitive = (EstimatorV2 if kind == "estimated_observable" else SamplerV2)(mode=mode, options=options)
        if self.mode == "job" and primitive.mode is not None:
            raise ValueError("ambient IBM Session/Batch would change explicit job mode; exit its context first")

        layout, _ = self._pub_layout(natives)
        pubs = []
        for index, items in enumerate(layout):
            native = natives[items[0]]
            # Each PUB needs its own correlation metadata even when the same
            # frozen circuit appears twice. A shallow container shares the
            # immutable circuit data, and only its public metadata is replaced.
            # A template PUB lists the job items of its rows in binding order.
            metadata = {**native.circuit.metadata, "nwqlib_pub": index}
            if native.values is not None:
                metadata["nwqlib_items"] = list(items)
            circuit = copy(native.circuit)
            circuit.metadata = metadata
            if kind == "estimated_observable":
                if native.observable is None:
                    raise ValueError("new Estimator submission requires the original layout-mapped observable")
                pubs.append((circuit, native.observable, None, native.metadata["observation"].estimate.precision))
            elif native.values is None:
                pubs.append((circuit, None, native.shots))
            else:
                pubs.append((circuit, self._binding_rows(natives, items), native.shots))
        job = primitive.run(pubs)
        locator = JobLocator(provider=self.kind, job_id=job.job_id(), instance=self.instance,
                             account=self.account_name or self.token_env)
        run._state["backend_context"].setdefault("jobs", {})[locator.job_id] = job
        return locator

    def _job(self, locator, run):
        """Return the original job, rejecting a locator or job from another account, instance or device.

        Both primitives submit to IBM's ``executor`` program, so the program
        does not tell counts from an estimate. ``_decode`` checks the kind of
        each returned PUB.
        """
        if (locator.provider != self.kind or locator.instance != self.instance
                or locator.account != (self.account_name or self.token_env)):
            raise ValueError("IBM job locator differs from the original provider/account/instance")
        jobs = run._state["backend_context"].setdefault("jobs", {})
        if locator.job_id not in jobs:

            jobs[locator.job_id] = self._service(run._state["backend_context"]).job(locator.job_id)
        job = jobs[locator.job_id]
        if (job.job_id() != locator.job_id or job.instance != self.instance
                or job.primitive_id != "executor" or job.backend().name != self.device):
            raise ValueError("retrieved IBM job has a different identity/instance/primitive")
        return job

    def reconcile(self, submission, natives, *, run):
        """Find a job whose creation acknowledgement was lost by its submission tag.

        At most two ``executor`` jobs of the configured device and instance
        with the submission's tag are requested. One verified match becomes
        the job locator. No match leaves the intent unresolved, and two raise,
        because neither case identifies the original job and neither
        authorizes a new submission.
        """
        from nwqlib.execution import JobLocator

        matches = self._service(run._state["backend_context"]).jobs(limit=2, backend_name=self.device, instance=self.instance,
                                        program_id="executor", job_tags=["nwqlib:" + submission.submission_id])
        if not matches:
            return None  # Absence is not proof that a lost submission was rejected.
        if len(matches) != 1:
            raise ValueError("IBM submission correlation is ambiguous; no new job is authorized")
        job, = matches
        if (job.instance != self.instance or job.primitive_id != "executor"
                or job.backend().name != self.device or "nwqlib:" + submission.submission_id not in job.tags):
            raise ValueError("IBM reconciliation returned a foreign job")
        locator = JobLocator(provider=self.kind, job_id=job.job_id(), instance=self.instance,
                             account=self.account_name or self.token_env)
        run._state["backend_context"].setdefault("jobs", {})[locator.job_id] = job
        return locator

    def _decode(self, result, natives, *, job_id, run, completed=(), keys=None):
        """Validate the full PUB inventory and row association, then yield the observations not yet saved.

        ``keys`` are the item result keys saved with the submission, in item
        order, and must equal the layout that ``launch`` used
        (``_pub_layout``). Every returned PUB must name a distinct layout PUB
        with its snapshot, and a template PUB the job items of its rows, in
        order, before any row is decoded, including on resumed reads.
        """
        from qiskit.primitives.containers import BitArray, PrimitiveResult, PubResult, SamplerPubResult
        from nwqlib.backends.results import BackendRunResult
        from nwqlib.execution import ExecutionMode

        layout, expected = self._pub_layout(natives)
        if keys is not None and tuple(keys) != expected:
            raise ValueError("saved IBM item keys differ from the PUB layout of the original job items")
        if not isinstance(result, PrimitiveResult) or len(result) != len(layout):
            raise ValueError("IBM result must contain exactly the original PUB inventory")

        seen = set()
        for pub in result:
            if not isinstance(pub, PubResult):
                raise ValueError("IBM result has a different primitive data type")
            metadata = pub.metadata.get("circuit_metadata", {})
            index = metadata.get("nwqlib_pub")
            if type(index) is not int or not 0 <= index < len(layout) or index in seen:
                raise ValueError("IBM result has an unknown or repeated original PUB coordinate")
            native = natives[layout[index][0]]
            if metadata.get("nwqlib_snapshot") != native.snapshot:
                raise ValueError("IBM result snapshot differs from its original PUB coordinate")
            if native.values is not None and metadata.get("nwqlib_items") != list(layout[index]):
                raise ValueError("IBM template PUB rows differ from their original job items")
            seen.add(index)
        # A swapped snapshot, repeated coordinate or reordered row invalidates
        # the result inventory before any PUB is decoded.
        for pub in result:
            items = layout[pub.metadata["circuit_metadata"]["nwqlib_pub"]]
            native = natives[items[0]]
            if native.metadata["observation"].kind == "estimated_observable":
                if expected[items[0]] not in completed:
                    yield expected[items[0]], self._estimate_result(pub, result.metadata, native, job_id=job_id,
                                                                    run=run)
                continue
            if not isinstance(pub, SamplerPubResult):
                raise ValueError("IBM Sampler result has a different primitive data type")
            names = tuple(register.name for register in native.classical_layout)
            if set(pub.data) != set(names):
                raise ValueError("IBM result registers differ from the prepared classical layout")
            arrays = tuple(pub.data[name] for name in names)
            shape = () if native.values is None else (len(items),)
            if any(not isinstance(array, BitArray) or array.shape != shape or array.num_shots > native.shots
                   for array in arrays):
                raise ValueError("IBM PUB data differ from the original " + (
                    "scalar parameter coordinate" if native.values is None else "template rows") + " or shot cap")
            for register in native.classical_layout:
                if pub.data[register.name].num_bits != len(register.bits):
                    raise ValueError("IBM result register width differs from the prepared mapping")
            if len({array.num_shots for array in arrays}) != 1:
                raise ValueError("IBM classical registers do not share a joint shot population")
            widths = tuple(len(register.bits) for register in native.classical_layout)
            positions = tuple(bit for register in native.classical_layout for bit in register.bits)
            for row, item in enumerate(items):
                if expected[item] in completed:
                    continue
                # Each row is joined and decoded alone, with the scalar join
                # law (sampler_decode_bytes) of its own S shots: the row's
                # register arrays are views into the PUB's arrays, joined by the
                # same concatenate_bits that join_data calls, and its temporary
                # arrays are released before the next row's check.
                run.check_data(sampler_decode_bytes(widths, arrays[0].num_shots))
                joined = (pub.join_data(names) if native.values is None else
                          BitArray.concatenate_bits([array[row] for array in arrays]))
                ordered = self._joint_counts(joined, positions)
                del joined
                yield expected[item], BackendRunResult(execution_mode=ExecutionMode.SHOTS,
                    backend_target=self.target_for(native.metadata["observation"]),
                    raw_output={"counts": ordered}, metadata={"native_job_id": job_id})
                del ordered

    @staticmethod
    def _joint_counts(joined, positions):
        """Count the joined shot outcomes of one scalar PUB or template row over the original global classical bits.

        BitArray concatenates registers along its least-significant-first
        bit axis, and get_counts renders most-significant-first strings.
        Shot alignment is preserved and joined bit j, counted from the
        right, is global classical bit positions[j]. Qiskit's join_data
        docstring describes the opposite register order, but its 2.5
        implementation (bit_array._unpack and _pack) puts the first register
        at the least significant end. At most 64 bits give an ``(indices,
        counts)`` pair, wider outcomes a dictionary of bit strings.
        """
        import numpy as np
        from nwqlib.backends.results import gather_bits
        width = len(positions)
        if width <= 64:
            # Each shot's joined integer is formed from the BitArray's
            # big-endian bytes (last byte holds bits 0-7), global classical
            # bit positions[j] takes joined bit j (gather_bits, and the identity
            # layout keeps the joined integers), and equal outcomes are
            # counted once with np.unique, so no per-shot string is built.
            # The transient arrays are released when this call returns, before
            # the next row's check, as the decode bound assumes.
            rows = joined.array
            native_outcomes = np.zeros(rows.shape[0], dtype=np.uint64)
            for column in range(rows.shape[1]):
                native_outcomes = (native_outcomes << np.uint64(8)) | rows[:, column].astype(np.uint64)
            sources = [None] * width
            for index, target in enumerate(positions):
                sources[target] = index
            indices, counts = np.unique(gather_bits(native_outcomes, sources), return_counts=True)
            return indices, counts.astype(np.int64, copy=False)
        ordered = {}
        for bits, count in joined.get_counts().items():
            output = ["0"] * width
            for index, target in enumerate(positions):
                output[width - target - 1] = bits[width - index - 1]
            label = "".join(output)
            ordered[label] = ordered.get(label, 0) + count
        return ordered

    def _estimate_result(self, pub, job_metadata, native, *, job_id, run):
        """Decode one weighted scalar estimate, with the provider's stds as its standard error only when ZNE was
        not applied.

        The raw stds and ensemble standard error stay in the provider metadata
        in every case. Their statistical interpretation is the provider's and
        is not verified here.
        """
        import numpy as np
        from nwqlib._run_journal import encode
        from nwqlib.backends.results import BackendRunResult
        from nwqlib.execution import ExecutionMode

        estimate = native.metadata["observation"].estimate
        value = pub.data["evs"] if "evs" in pub.data else None
        error = pub.data["stds"] if "stds" in pub.data else None
        ensemble = pub.data["ensemble_standard_error"] if "ensemble_standard_error" in pub.data else None
        # A scalar field is a 0-d float array, or with ZNE (qiskit-mitigation
        # 0.2.0) a NumPy float scalar, which has the same shape and dtype.
        scalar = (np.ndarray, np.floating)
        if (not isinstance(value, scalar) or value.shape != () or value.dtype.kind != "f"
                or error is not None and (not isinstance(error, scalar) or error.shape != () or error.dtype.kind != "f")):
            raise ValueError("Estimator data must match the original scalar observable and parameter coordinate")
        if ensemble is not None and (not isinstance(ensemble, scalar) or ensemble.shape != () or ensemble.dtype.kind != "f"):
            raise ValueError("Estimator ensemble uncertainty has a foreign observable/parameter shape")
        # The client-side EstimatorV2 of qiskit-ibm-runtime 0.52.0 puts target_precision in the job
        # metadata, not in the PUB metadata read here, so this check does not trigger on its results.
        target_precision = pub.metadata.get("target_precision")
        if target_precision is not None and target_precision != estimate.precision:
            raise ValueError("Estimator result has a different original target precision")
        from math import isfinite
        raw_error, raw_ensemble = (None if error is None else float(error)), (None if ensemble is None else float(ensemble))
        if any(item is not None and (not isfinite(item) or item < 0) for item in (raw_error, raw_ensemble)):
            raise ValueError("provider-reported uncertainty must be finite and nonnegative")
        # stds becomes the standard error only when ZNE is known to be off,
        # and the raw provider uncertainties are kept either way.
        # The options text was admitted at preparation and reserved with the
        # estimate (estimate_bytes).
        options_text = native.metadata["requested_options_json"]
        requested = json.loads(options_text)
        resilience = job_metadata.get("resilience")
        zne = resilience.get("zne_mitigation") if isinstance(resilience, dict) else None
        if type(zne) is not bool:
            zne = requested.get("resilience", {}).get("zne_mitigation")
        if type(zne) is not bool and requested.get("resilience_level") == 0:
            zne = False
        no_stds, fit, unknown = _ESTIMATE_REASONS
        reason = (no_stds if error is None else fit if zne is True else unknown if zne is not False else None)
        metadata = dict(primitive=_portable_metadata(job_metadata), pub=_portable_metadata(pub.metadata),
                        reported_data=dict(stds=raw_error, ensemble_standard_error=raw_ensemble))
        # The provider metadata of a paid job is encoded without a size limit and counted at its actual
        # size when the outcome is saved (Run._write), so a large one never needs a second fetch.
        stored = encode(metadata, None)
        result = _estimate_value(estimate.content_id, float(value), raw_error if reason is None else None, reason,
                                 options_text, stored)
        return BackendRunResult(execution_mode=ExecutionMode.ESTIMATED,
            backend_target=self.target_for(native.metadata["observation"]),
            raw_output={"estimates": (result,)}, metadata={"native_job_id": job_id})

    def refresh(self, locator, natives, *, run):
        """Read the original job's status once and decode its PUBs after a terminal DONE.

        Pending states (INITIALIZING, QUEUED, VALIDATING and RUNNING) map to
        "acknowledged", ERROR and CANCELLED to terminal
        failure or cancellation, and any other state to "uncertain". PUBs that
        the Run has already saved are skipped, and a decode error after
        some PUBs succeeded is returned with those results.
        """
        from nwqlib.backends.connection import BackendRefresh
        job = self._job(locator, run)
        status = job.status()
        if status in {"INITIALIZING", "QUEUED", "RUNNING", "VALIDATING"}:
            return BackendRefresh(status="acknowledged", provider_status=status)
        if status in {"ERROR", "CANCELLED"}:
            run._state["backend_context"]["jobs"].pop(locator.job_id, None)
            return BackendRefresh(status="failed" if status == "ERROR" else "cancelled", provider_status=status,
                                  failure="IBM job " + status)
        if status != "DONE":
            return BackendRefresh(status="uncertain", provider_status=str(status), failure="unrecognized IBM job state")
        # result() is called only after a terminal status. timeout=0 prevents a
        # status race from turning one refresh into a provider-queue wait.
        result = job.result(timeout=0)
        # launch tags the job nwqlib:<submission_id>, so the submission is
        # found by its id. The locator comparison rejects a foreign tag.
        submissions = run._state["submissions"]
        submission = next((submissions[tag[7:]] for tag in (job.tags or ())
                           if isinstance(tag, str) and tag.startswith("nwqlib:") and tag[7:] in submissions
                           and submissions[tag[7:]].locator == locator), None)
        completed = set() if submission is None else {
            item.result_key for item in submission.items if item.attempt in run._state["by_attempt"]}
        keys = None if submission is None else tuple(item.result_key for item in submission.items)
        decoded = []
        try:
            for item in self._decode(result, natives, job_id=locator.job_id, run=run, completed=completed,
                                     keys=keys):
                decoded.append(item)
        except Exception as error:
            if run._state.get("storage_failed", False) or not decoded:
                raise
            # Keep the live job while local decoding is incomplete. The common
            # owner commits earlier PUBs, then raises this original exception.
            return BackendRefresh(status="completed", provider_status=status,
                                  results=tuple(decoded), error=error)
        # Terminal retrieval no longer needs the run-owned live job object.
        # Keep the reduced statistics and durable locator for later consumers.
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
