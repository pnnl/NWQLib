"""Explicit backend configuration, native preparation metadata and the adapter contract.

Configuration construction and interchange import no circuit/provider SDK.
Native operations are called only by the common admitted execution owner,
``nwqlib._prepared_execution`` (docs/development/execution.md).

An adapter is an immutable configuration Record with one of two call shapes.
A synchronous adapter (Aer) implements ``target_for``, ``prepare`` and
``submit``, and ``submit`` returns the native result. A detached adapter
(NWQ-Sim, Slurm, IBM Runtime, IonQ and Nexus) sets ``supports_synchronous``
false and implements ``target_for``, ``prepare`` (Nexus uses staged remote
preparation instead), ``restore_native``, ``admit_batch``, ``launch`` and
``refresh``, with optional ``reconcile``, ``can_reconcile`` and ``cancel``. Every adapter keeps
these relations, which the common owner relies on:

- ``target_for`` and ``admit_batch`` reject an unsupported readout, shot count
  or batch before preparation, upload or submission, so a request that cannot
  run spends no provider work.
- The common owner persists the submission intent and its reserved exposure
  before calling ``launch``. ``launch`` sends one create request, labels it
  with a name, tag or spool path derived from the submission UUID and returns
  the provider locator. The adapter never retries a create request, because a
  request whose acknowledgement was lost may already have been accepted. The
  IBM SDK may still retry certain server errors under IBM's safe-retry
  contract (docs/ibm.md).
- ``reconcile`` runs only for an intent without a locator. It searches by that
  label and returns None when nothing matches. Absence does not prove
  rejection, so the intent stays uncertain with its exposure charged, and no
  replacement job is launched. Several matches raise for IBM and Nexus and
  return None for Slurm. IonQ exposes no reconciliation method, since no exact
  lookup is qualified for its API. Its lost acknowledgement therefore raises
  RunFailed with the outcome still uncertain.
  An optional ``can_reconcile`` hook may use existing submission metadata to
  determine whether automatic waiting has observable progress. A refusal raises
  with status uncertain and leaves later explicit retrieval possible. The hook
  must not submit or run a scientific computation.
- ``refresh`` reads the original job and never submits. When a later item
  fails, it returns the items already decoded together with that exception in
  ``BackendRefresh.error``, so the common owner publishes them before raising.
- ``cancel`` sends one request for the original job. Only a later provider
  status confirms cancellation, and reserved circuits and shots stay charged.
- Each adapter translates measurement maps and bit order in one decoder
  (``_decode`` for IBM, IonQ and Nexus, ``_submit_aer_execution`` for Aer and
  ``NWQSimBackend._result`` with the runner for NWQ-Sim and Slurm). IonQ's
  native gate angles, in turns rather than radians, are fixed at preparation.
  Count keys are bit strings over
  the global classical bits with bit 0 rightmost. Probability keys follow the
  requested qubit tuple with its first qubit rightmost. An adapter may instead
  return probabilities as a dense float64 array of length
  ``2**len(qubits)`` or an ``(indices, values)`` pair, and counts as an
  ``(indices, counts)`` pair. The index, uint64 or, for probabilities of more
  than 64 qubits, packed words, has the first requested qubit, or classical
  bit 0, as its least significant bit. Pauli labels use Qiskit order, with
  qubit 0 rightmost.
- Provider responses, result files and saved payloads that NWQLib reads or
  parses are admitted through ``run.check_data`` before the read or parse.
  Transfers inside a provider SDK are outside that admission.
- ``qualification_notice`` states the evidence scope once per Run. Offline SDK
  and transport checks do not qualify a live account, queue, device or site.

An adapter that sets ``prepares_from_templates`` true receives three more
keyword arguments at ``prepare`` (or ``prepare_local``), so that it can cache a
parameterized native template per construction and assign each point's
parameters to it (``_prepared_execution.prepare_experiment``):
``construction_id`` (the selected construction's content identity, the same for
every point of that construction), ``backend_context`` (the Run's per-Run
backend dictionary) and ``template_seed`` (a seed for building that template,
derived from the Plan's randomness and the construction identity, see
``_prepared_execution._template_seed``). An adapter that used a template
returns that seed in ``NativePreparation.template_seed`` and the receipt
records it. Other adapters receive none of the three and prepare as before.
"""

from dataclasses import dataclass
from importlib.metadata import version
import platform
from typing import Literal
from uuid import uuid4

from pydantic import PrivateAttr

from nwqlib.backends.capabilities import unsupported_readout
from nwqlib.core.records import Record, Source, Text
from nwqlib.execution import CountsSampling, RegisterMap, SubmissionItem


@dataclass(frozen=True)
class NativePreparation:
    """The native input returned by a backend's ``prepare``, with the facts its receipt records.

    ``_finish_native_preparation`` copies these fields into the
    ``PreparedArtifact`` receipt and stores ``payload`` beside it, so the
    receipt describes the object that will execute.

    Attributes:
        native: Actual backend-native prepared input, not an executable reconstructed from metadata.
        target: Source identifying the selected device or simulator target.
        compiler: Source identifying the actual preparation/compiler operation.
        native_basis: Actual native gate vocabulary after preparation.
        environment: Versions recorded from the preparation process.
        quantum_layout: Ordered native quantum registers and global bit indices.
        classical_layout: Ordered native classical registers and global bit indices.
        logical_to_native: Map from selected logical wires to actual native wire positions.
        operations: Actual prepared-operation count; None if unavailable.
        population: Readout population meaning, including conditioning or explicit unknown status.
        transformation: Scientific/compiler description of how the selected input became this native input.
        payload: Optional saved native bytes for later restoration; None if not supplied.
        payload_format: Native byte format identifier; None without a payload format.
        counts_sampling: Fresh/fixed-seed/unknown sampling-source description for counts; None for other readouts.
        provider_options_json: Optional serialized effective provider settings, excluding credentials.
        probability_window_exclusions: Labels of the native operations, readout path,
            simulator version or non-default compiler optimization level outside the
            exact-probability roundoff derivation, an empty
            tuple when nothing is, or None when not assessed.
        template_seed: The ``template_seed`` that the preparation hook supplied,
            when this preparation used a per-construction template; None otherwise.
        statevector_roundoff: The ``(t, e)`` envelope of the host phase
            correction of the saved statevectors, the componentwise maxima
            over the save keys (``PreparedArtifact.statevector_roundoff``).
            The pair ``(t, e)`` bounds the output phase product on saved
            statevectors. ``(0.0, 0.0)`` denotes an assessed zero-error
            operation, including no product or multiplication by the
            represented factor ``(1.0, ±0.0)`` under the host arithmetic
            model. ``None`` means the host correction was not assessed.
    """

    native: object
    target: Source
    compiler: Source
    native_basis: tuple
    environment: tuple
    quantum_layout: tuple
    classical_layout: tuple
    logical_to_native: tuple
    operations: int | None
    population: str
    transformation: str
    payload: bytes | None = None
    payload_format: str | None = None
    counts_sampling: CountsSampling | None = None
    provider_options_json: str | None = None
    probability_window_exclusions: tuple[str, ...] | None = None
    template_seed: int | None = None
    statevector_roundoff: tuple[float, float] | None = None


@dataclass(frozen=True)
class BackendRefresh:
    """What one ``refresh`` call learned about an original job.

    ``refresh_submissions`` commits ``status`` and ``associations`` first and
    then publishes ``results``. When ``error`` is set, the results decoded
    before it are still published, and the error is raised afterwards.

    Attributes:
        status: Common submission state reported by this refresh: "acknowledged"
            (still pending), "completed", "failed", "cancelled" or "uncertain"
            (the outcome cannot be established and nothing is resubmitted).
        results: Original item keys paired with available backend results; already published items are not new populations.
        provider_status: Raw provider state label, or None if unavailable.
        failure: Provider failure explanation without changing prior observations; None when absent.
        native_simulations: Observed native simulation count, or None when unavailable.
        associations: Newly learned child/result locators tied to original submission items.
        error: Original retrieval/decoding exception to propagate after preserving available state; None on a successful refresh.
    """

    status: str
    results: tuple = ()
    provider_status: str | None = None
    failure: str | None = None
    native_simulations: int | None = None
    associations: tuple[SubmissionItem, ...] = ()
    error: Exception | None = None


def environment(packages):
    """Record installed package and Python versions of the process that prepared a circuit.

    Receipts carry these Sources so a result can be traced to the SDK versions that
    produced its native input, not to whatever is installed when it is reopened.
    """
    return tuple(Source(name=name, version=version(name), domain="actual preparation process package",
                        reference="importlib.metadata") for name in packages) + (
        Source(name="python", version=platform.python_version(), domain="actual preparation process",
               reference="platform.python_version"),)


def _run_state_threads(run):
    """The positive state-update thread cap of an open Run's Aer statevector simulations.

    The value (``qiskit_aer._aer_state_threads``) is resolved once while the
    Run is open and kept in its live state, not in the saved backend
    context, so the saved-buffer admission and every preparation of that
    open Run use the same cap, which ``_prepare_aer_execution`` passes to Aer
    as ``max_parallel_threads``. A reopened Run resolves it on its own host.
    A restored handle keeps the cap recorded in its preparation metadata.
    """
    from nwqlib.backends import qiskit_aer as aer
    return run._state.setdefault("aer_state_threads", aer._aer_state_threads())


def coherent_body_issue(circuit, stop=None):
    """The first operation of a bound body that a pure-state trajectory cannot observe, or None.

    Only the executed prefix, the first ``stop`` instructions (all when
    None), is inspected: nothing after the last observation is executed, so
    an operation there cannot change an earlier statistic.

    A trajectory evaluates the declared points of one deterministic, noiseless
    coherent evolution, so the selected body and views must not introduce
    measurement-conditioned evolution, resets, postselection or non-unitary
    channels. A Qiskit ``Gate`` denotes a unitary operation and is admitted;
    a barrier or delay does not act on the state. A measurement, a reset or a
    control-flow operation is refused wherever it occurs, including inside the
    definition of a composite instruction, so a composite (``Initialize``
    among them, whose definition resets) cannot hide one. Any other
    instruction is admitted only through its definition, each shared
    definition being inspected once; an opaque instruction without one has no
    established pure-state semantics and is refused. No operator is built and
    nothing is simulated.
    """
    from qiskit.circuit import Barrier, ControlFlowOp, Delay, Gate, Measure, Reset
    seen = set()

    def issue(body, outer, stop=None):
        for item in body.data[:stop]:
            operation = item.operation
            if isinstance(operation, (Gate, Barrier, Delay)):
                continue
            label = repr(operation.name) if outer is None else f"{outer!r} contains {operation.name!r}, which"
            if isinstance(operation, (Measure, Reset, ControlFlowOp)):
                return f"operation {label} measures, resets or branches"
            definition = operation.definition
            if definition is None:
                return f"operation {label} is an opaque instruction without a definition"
            if id(definition) not in seen:
                seen.add(id(definition))
                found = issue(definition, outer or operation.name)
                if found is not None:
                    return found
        return None

    return issue(circuit, None, stop)


def circuit_layout(circuit, registers):
    """Map each named register to its circuit-global bit indices, least significant first."""
    return tuple(RegisterMap(name=reg.name, bits=tuple(circuit.find_bit(bit).index for bit in reg)) for reg in registers)


class AerBackend(Record):
    """Local Aer configuration; explicit noise applies to sampled counts only.

    from_noise_model binds the actual caller-owned SDK object without copying or
    serializing it. Do not mutate that model while this backend or its prepared
    handles are reused. noise_model_id identifies this binding, not model contents.
    JSON configuration alone is inert. Run.save stores the bound model as the
    SDK's native dictionary with NumPy array files. load_run binds the model it
    rebuilds from them to the reopened Run's own copy of this configuration
    (``run.backend``) and leaves the backend passed to it unchanged.
    """

    kind: Literal["qiskit_aer"] = "qiskit_aer"
    noise_model_id: Text | None = None
    _noise_model: object = PrivateAttr(default=None)

    @classmethod
    def from_noise_model(cls, model):
        """Bind one supplied NoiseModel; import its SDK only on this explicit call."""
        from qiskit_aer.noise import NoiseModel
        if not isinstance(model, NoiseModel):
            raise TypeError("from_noise_model requires a Qiskit Aer NoiseModel")
        backend = cls(noise_model_id=str(uuid4()))
        backend._noise_model = model
        return backend

    def _bound_noise_model(self):
        """The bound NoiseModel, or None for a noiseless backend.

        A configuration that names a binding but holds no model, for example one
        rebuilt from JSON, raises instead of running noiselessly.
        """
        if self.noise_model_id is not None and self._noise_model is None:
            raise ValueError("Aer noise model is unbound; load_run restores its saved SDK input, "
                             "or use from_noise_model to start a new explicit binding")
        return self._noise_model

    def revise(self, **changes):
        """Revise the configuration, keeping the bound model while ``noise_model_id`` is unchanged.

        ``Record.revise`` and ``Record.model_copy`` rebuild the record from its
        fields, which drops private attributes such as the bound model.
        """
        revised = super().revise(**changes)
        if revised.noise_model_id == self.noise_model_id:
            revised._noise_model = self._noise_model
        return revised

    def model_copy(self, *, update=None, deep=False):
        """Copy the configuration, keeping the bound model as ``revise`` does."""
        copied = super().model_copy(update=update, deep=deep)
        if copied.noise_model_id == self.noise_model_id:
            copied._noise_model = self._noise_model
        return copied

    def target_for(self, observation):
        """Select the counts or statevector target, rejecting a noisy exact readout first.

        Noise acts on sampled counts. An exact expectation, probability or
        amplitude with a bound noise model would describe neither the ideal nor
        the noisy population, so it is rejected before any lowering.
        """
        from nwqlib.backends.targets import AER_COUNTS_TARGET, AER_STATEVECTOR_TARGET
        if self.noise_model_id is not None and observation.kind != "counts":
            raise ValueError("explicit Aer noise supports counts; exact readouts require a noiseless backend")
        self._bound_noise_model()
        return AER_COUNTS_TARGET if observation.kind == "counts" else AER_STATEVECTOR_TARGET

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot, views=None):
        """Lower one selected logical circuit for Aer and describe the native input.

        ``views`` maps each trajectory readout view's tail and inverse
        definition to its logical circuit on the body's registers; each is
        checked like the body and inserted at its view's boundary.

        Lowered gate definitions are cached per Run and target, keyed by the
        source gate object, so repeated preparations of one construction lower
        each definition once. ``keep_sources`` drops cache entries whose source
        gates left the current selection, which bounds the cache during
        parameter scans. On a noise target without the unitary instruction,
        lowering takes the exact syntheses of dense unitaries from the Run's
        synthesis cache (``Run._exact_dense_unitaries``), so a definition
        that is lowered again after ``keep_sources`` dropped it needs no new
        synthesis. The noise model is bound once per Run context. Counts
        record a fixed-seed sampling source because the Plan's runtime seed
        selects Aer's stream. Amplitude readout requires the unconditional
        pre-final-measurement state, since a measurement-conditioned
        trajectory is not the selected state.
        """
        from nwqlib.backends import qiskit_aer as aer
        target = self.target_for(observation)
        unsupported = unsupported_readout(target, observation)
        if unsupported is not None:
            raise ValueError(f"Aer target {target.name!r} does not execute {unsupported}")
        # A trajectory's position is the tuple of its resolved point
        # boundaries (ReadoutDetails.boundaries), resolved before this call.
        trajectory, phase_keys = None, frozenset()
        if observation.kind == "trajectory":
            if (type(position) is not tuple
                    or position != observation.boundaries(len(circuit.data))):
                raise ValueError("a trajectory is prepared with the resolved boundaries of its bound body")
            # Only the executed prefix, through the last point, is checked.
            self._refuse_inadmissible_trajectory(circuit, stop=position[-1])
            for tail in (views or {}).values():
                self._refuse_inadmissible_trajectory(tail)
            trajectory = aer._trajectory_saves(observation, position)
            # A saved state that serves an amplitude point, or a reduction
            # whose reducer is not registered phase invariant, gets the phase
            # of its own prefix; one serving only phase-invariant reductions
            # may be read with the executed circuit's phase.
            from nwqlib.core.planning import registered_reducer
            phase_keys = frozenset(
                keys for point, (_, kind, keys) in zip(observation.positions, trajectory[1], strict=True)
                if kind == "amplitudes" or (kind == "reduction"
                                            and not registered_reducer(point.reducer).phase_invariant))
        if circuit.num_qubits > run.limits.max_simulation_qubits:
            raise ValueError("Aer circuit exceeds max_simulation_qubits")
        context = run._state["backend_context"]
        preparations = context.setdefault("aer_preparations", {})
        if target.name not in preparations:
            preparations[target.name] = aer._AerPreparation()
            from nwqlib._run_archive import _CacheEntries
            preparations[target.name]._blocks = _CacheEntries(prefix=("aer", target.name),
                changes=run._state["cache_changes"])
        # Keep only definitions owned by the selected live lowering population.
        sources = tuple(source_definitions)
        for owner in preparations.values():
            owner.keep_sources(iter(sources))
        noise = None
        if self.noise_model_id is not None:
            noise = context.setdefault("noise_model", self._bound_noise_model())
        options = {} if noise is None else {"noise_model": noise}
        native = aer._prepare_aer_execution(target, circuit, shots=observation.shots or None, seed=runtime.seed,
            pauli_expectation_readout=((position, observation.labels) if observation.kind == "pauli_expectation" else None),
            probability_qubits=observation.qubits if observation.kind == "probabilities" else None,
            trajectory=trajectory, phase_keys=phase_keys, views=views, preparation=preparations[target.name],
            simulator_memory_mb=run.limits.simulator_memory_mb, state_threads=_run_state_threads(run),
            exact_synthesis=run._exact_dense_unitaries, **options)
        if trajectory is not None and native.simulator.options.method != "statevector":
            # Pure-state execution is established by the effective method,
            # not by the absence of a noise-model option.
            raise ValueError(f"an Aer trajectory requires deterministic noiseless pure-state execution; "
                             f"the selected simulator method is {native.simulator.options.method!r}")
        native.metadata["simulator_memory_mb"] = run.limits.simulator_memory_mb
        native.metadata["simulator_method"] = native.simulator.options.method
        native.metadata["simulator_zero_threshold"] = native.simulator.options.zero_threshold
        population = ("unconditional" if observation.kind == "counts" else native.metadata["readout_population"])
        if observation.kind == "amplitudes":
            if native.metadata.get("statevector_semantics") != "pre_final_measurement":
                raise ValueError("amplitudes require an unconditional pre-final-measurement trajectory")
            population = "unconditional"
        elif trajectory is not None and any(point.kind == "amplitudes" for point in observation.positions):
            if native.metadata.get("statevector_semantics") != "pre_final_measurement":
                raise ValueError("amplitude points require an unconditional pre-final-measurement trajectory")
        return NativePreparation(native=native,
            target=Source(name=target.name, version=self.native_target_version(),
                domain=target.description + ("; explicit noise model" if noise is not None else ""),
                reference="nwqlib.backends.qiskit_aer actual target"),
            compiler=Source(name="Aer native decomposition"
                    + ("" if "unitary" in native.simulator.target.operation_names
                       else " with exact dense-unitary synthesis")
                    + (" and noise-target basis translation" if
                    native.metadata.get("execution_preprocessing", {}).get("native_basis_translation") else ""), version=version("qiskit"),
                domain="selected logical instruction positions; no inventory transpilation",
                reference="nwqlib.backends.qiskit_aer._prepare_aer_execution"),
            native_basis=tuple(sorted(native.simulator.target.operation_names)),
            environment=environment(("nwqlib", "numpy", "qiskit", "qiskit-aer")),
            quantum_layout=circuit_layout(native.circuit, native.circuit.qregs),
            classical_layout=circuit_layout(native.circuit, native.circuit.cregs),
            logical_to_native=tuple(range(circuit.num_qubits)), population=population,
            # G of a coherent statevector readout counts its native evolution
            # operations and no save instruction (qiskit_aer._aer_operation_counts).
            # A body with control flow has no fixed native count and keeps its
            # instruction inventory; shot sampling keeps its own count.
            operations=(aer._aer_operation_counts(native.circuit)[0] if observation.kind != "counts"
                        and not native.circuit.has_control_flow_op() else len(native.circuit.data)),
            counts_sampling=CountsSampling(kind="fixed_seed", seed=runtime.seed) if observation.kind == "counts" else None,
            probability_window_exclusions=native.metadata.get("probability_window_exclusions"),
            # The envelope of the stored phase factors; (0, 0) when no saved
            # state is multiplied, including every non-trajectory readout.
            statevector_roundoff=aer._statevector_roundoff(native),
            transformation="fresh logical container with run-owned definitions; shared Aer lowering/readout; "
                "owned in-process snapshot; selected simulator memory limit, no process RSS claim")

    def admit_trajectory_buffers(self, observation, *, width, memory_mb, run):
        """Admit a trajectory's live state, retained saves and marginal workspace against ``memory_mb``.

        This is the Aer trajectory preflight before native lowering
        (``qiskit_aer._admit_aer_saved_buffers``). The trajectory memory check
        uses the same positive thread cap passed to Aer's
        ``max_parallel_threads`` (``_run_state_threads``). The cap is part of the
        effective preparation configuration and is reapplied on restoration.
        Saved marginal workspace is charged for at most that many
        state-update workers under the CPU double-statevector execution
        configuration. Returns the counted bytes.
        """
        from nwqlib.backends import qiskit_aer as aer
        return aer._admit_aer_saved_buffers(observation, width=width, memory_mb=memory_mb,
                                            state_threads=_run_state_threads(run), additional_peak_bytes=0)

    def native_target_version(self):
        """Version of the executing package, ``qiskit-aer``, recorded in each receipt's target and chunk source."""
        return version("qiskit-aer")

    def native_job_id_length(self):
        """Encoded length of Aer's job identifier, ``str(uuid.uuid4())`` in ``AerBackend.run``: 36 characters."""
        return 36

    def _refuse_inadmissible_trajectory(self, circuit, stop=None):
        """Refuse a trajectory that is not one noiseless coherent evolution, before lowering.

        This is the trajectory admissibility check of the Aer adapter
        (``coherent_body_issue``). A trajectory has zero raw shots, which does
        not admit measurement-conditioned, reset, postselected,
        noisy-trajectory or outcome-dependent evolution. A bound noise model is
        refused earlier, by ``target_for``, for every readout other than
        counts, and ``prepare`` checks the effective simulator method after
        lowering. ``stop`` is the last point's boundary, so only the executed
        prefix of the body is checked.
        """
        issue = coherent_body_issue(circuit, stop)
        if issue is not None:
            raise ValueError(f"an Aer trajectory requires a coherent body; {issue}")

    def submit(self, native, *, submission_id, run):
        """Execute the already prepared native circuit synchronously and return its raw output."""
        from nwqlib.backends.qiskit_aer import _submit_aer_execution
        return _submit_aer_execution(native)

    def restore_native_data(self, record, circuit, metadata, *, run):
        """Restore the already lowered circuit and effective simulator options."""
        from qiskit_aer import AerSimulator
        from nwqlib.backends.qiskit_aer import _PreparedAerExecution, _restore_aer_readout, _trajectory_saves
        from nwqlib.backends.targets import AER_STATEVECTOR_TARGET
        observation = record.observation
        trajectory = observation.kind == "trajectory"
        _restore_aer_readout(circuit, observation, record.boundaries if trajectory else ())
        target = self.target_for(observation)
        options = dict(method=metadata["simulator_method"], seed_simulator=record.runtime.seed,
                       max_memory_mb=metadata["simulator_memory_mb"],
                       zero_threshold=metadata["simulator_zero_threshold"])
        if target.name == AER_STATEVECTOR_TARGET.name:
            # The thread cap that the saved-buffer admission used.
            if "simulator_max_parallel_threads" not in metadata:
                raise ValueError(
                    "a restored Aer statevector preparation lacks its native metadata key "
                    "simulator_max_parallel_threads; its Run folder was written by a revision "
                    "that did not record the thread cap and cannot be restored by this one")
            options["max_parallel_threads"] = metadata["simulator_max_parallel_threads"]
        if self.noise_model_id is not None:
            options["noise_model"] = self._bound_noise_model()
        return _PreparedAerExecution(AerSimulator(**options), circuit, target,
            observation.shots or None,
            (observation.position, observation.labels) if observation.kind == "pauli_expectation" else None,
            observation.qubits if observation.kind == "probabilities" else None, metadata,
            _trajectory_saves(observation, record.boundaries) if trajectory else None)
