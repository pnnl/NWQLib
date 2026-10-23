"""Backend configurations, the description of a prepared circuit and the rules of backend adapters.

Constructing a configuration, or converting it to and from JSON, imports no
circuit or provider SDK. The adapter methods that use an SDK are called only by
the shared execution code, ``nwqlib._prepared_execution``, after its checks
against the run's limits (docs/development/execution.md).

An adapter is an immutable configuration Record with one of two call shapes.
A synchronous adapter (Aer) implements ``target_for``, ``prepare``, ``submit``
and ``restore_native_data``, and ``submit`` returns the SDK's result.
``restore_native_data`` rebuilds the SDK object from a saved circuit when a
prepared circuit of a reopened Run, or one released by
``Run.release_native``, is used again. A detached adapter, whose jobs continue
after Python exits
(NWQ-Sim, Slurm, IBM Runtime, IonQ and Nexus), sets ``supports_synchronous``
false and implements ``target_for``, ``prepare`` (Nexus uses staged remote
preparation instead), ``restore_native``, ``admit_batch``, ``launch`` and
``refresh``, with optional ``reconcile``, ``can_reconcile`` and ``cancel``. Every adapter keeps
these relations, which the shared execution code relies on:

- ``target_for`` and ``admit_batch`` reject an unsupported readout, shot count
  or batch before preparation, upload or submission, so a request that cannot
  run spends no provider work.
- The shared execution code saves the submission intent and the circuits and
  shots set aside for it (its exposure) before calling ``launch``. ``launch``
  sends one create request, labels it with a name, tag or spool path derived
  from the submission UUID and returns the provider's job locator. The adapter
  never retries a create request, because a request whose acknowledgement was
  lost may already have been accepted. The IBM SDK may still retry certain
  server errors under IBM's safe-retry rules (docs/ibm.md).
- ``reconcile`` runs only for an intent without a locator. It searches by that
  label and returns None when nothing matches. Absence does not prove
  rejection, so the intent stays uncertain with its exposure counted against
  the limits, and no replacement job is launched. Several matches raise for
  IBM and Nexus and return None for Slurm. IonQ has no ``reconcile`` method,
  since no exact lookup is qualified for its API. Its lost acknowledgement
  therefore raises RunFailed with the outcome still uncertain.
  An optional ``can_reconcile`` hook may use existing submission metadata to
  determine whether automatic waiting can detect progress. A refusal raises
  with status uncertain and leaves later explicit retrieval possible. The hook
  must not submit or run a scientific computation.
- ``refresh`` reads the original job and never submits. When a later item
  fails, it returns the items already decoded together with that exception in
  ``BackendRefresh.error``, so the shared execution code saves them before
  raising.
- ``cancel`` sends one request for the original job. Only a later provider
  status confirms cancellation, and the circuits and shots set aside stay
  counted.
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
  parses are checked by ``run.check_data`` before the read or parse.
  Transfers inside a provider SDK are outside that check.
- ``qualification_notice`` states the scope of the evidence once per Run. Offline SDK
  and transport checks do not qualify a live account, queue, device or site.
"""

from dataclasses import dataclass
from importlib.metadata import version
import platform
from typing import Literal
from uuid import uuid4

from pydantic import PrivateAttr

from nwqlib.core.records import Record, Source, Text
from nwqlib.execution import CountsSampling, RegisterMap, SubmissionItem


@dataclass(frozen=True)
class NativePreparation:
    """The SDK object that a backend's ``prepare`` returns, with the facts its preparation record keeps.

    ``_finish_native_preparation`` copies these fields into the
    ``PreparedArtifact`` preparation record and saves ``payload`` beside it, so
    the record describes the object that will execute.

    Attributes:
        native: The prepared input in the backend SDK's own form, not an executable rebuilt from metadata.
        target: Source identifying the chosen device or simulator target.
        compiler: Source identifying the preparation or compiler operation that was performed.
        native_basis: The backend's gate names after preparation.
        environment: Versions recorded from the preparation process.
        quantum_layout: Ordered quantum registers of the prepared circuit and their global bit indices.
        classical_layout: Ordered classical registers of the prepared circuit and their global bit indices.
        logical_to_native: Map from the logical wires to the wire positions of the prepared circuit.
        operations: Operation count of the prepared circuit. None if unavailable.
        population: Whether a measurement or reset before the readout conditions the read state:
            "unconditional", "native_conditioned" or an explicit "unknown".
        transformation: Description of how the chosen circuit became this prepared input.
        payload: Optional bytes of the prepared input, saved for later restoration. None if not supplied.
        payload_format: Identifier of the payload's byte format. None without a payload format.
        counts_sampling: Fresh/fixed-seed/unknown sampling-source description for counts. None for other readouts.
        provider_options_json: Optional serialized effective provider settings, excluding credentials.
        probability_window_exclusions: Labels of the prepared operations, readout path,
            simulator version or non-default compiler optimization level that the
            roundoff derivation of exact probabilities does not cover, an empty
            tuple when it covers everything, or None when not assessed.
        statevector_roundoff: The ``(t, e)`` error bound of the classical phase
            correction of the saved statevectors, the componentwise maxima
            over the save keys (``PreparedArtifact.statevector_roundoff``).
            The pair ``(t, e)`` bounds the output phase product on saved
            statevectors. ``(0.0, 0.0)`` denotes an assessed zero-error
            operation, including no product or multiplication by the
            represented factor ``(1.0, ±0.0)`` under the arithmetic model of
            the classical computation. ``None`` means the classical correction
            was not assessed.
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
    statevector_roundoff: tuple[float, float] | None = None


@dataclass(frozen=True)
class BackendRefresh:
    """What one ``refresh`` call learned about an original job.

    ``refresh_submissions`` saves ``status`` and ``associations`` first and
    then saves ``results``. When ``error`` is set, the results decoded before
    it are still saved, and the error is raised afterwards.

    Attributes:
        status: Common submission state reported by this refresh: "acknowledged"
            (still pending), "completed", "failed", "cancelled" or "uncertain"
            (the outcome cannot be established and nothing is resubmitted).
        results: Original item keys paired with available backend results. Already saved items are not new measurements.
        provider_status: Raw provider state label, or None if unavailable.
        failure: Provider failure explanation without changing prior observations. None when absent.
        native_simulations: Number of simulations that the backend reports, or None when unavailable.
        associations: Newly learned references to child jobs or results, tied to original submission items.
        error: Original retrieval or decoding exception, raised after the available state is saved. None on a successful refresh.
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

    Preparation records carry these Sources so a result can be traced to the SDK
    versions that produced its prepared input, not to whatever is installed when it
    is reopened.
    """
    return tuple(Source(name=name, version=version(name), domain="actual preparation process package",
                        reference="importlib.metadata") for name in packages) + (
        Source(name="python", version=platform.python_version(), domain="actual preparation process",
               reference="platform.python_version"),)


def _run_state_threads(run):
    """The positive state-update thread cap of an open Run's Aer statevector simulations.

    The value (``qiskit_aer._aer_state_threads``) is resolved once while the
    Run is open and kept in its in-memory state, not in the saved
    backend context, so the memory check of saved buffers and every
    preparation of that open Run use the same cap, which
    ``_prepare_aer_execution`` passes to Aer as ``max_parallel_threads``. A
    reopened Run resolves it on its own host. A restored prepared circuit
    keeps the cap recorded in its preparation metadata.
    """
    from nwqlib.backends import qiskit_aer as aer
    return run._state.setdefault("aer_state_threads", aer._aer_state_threads())


def coherent_body_issue(circuit, stop=None):
    """The first operation of a parameter-bound circuit body that a pure-state trajectory cannot evaluate, or None.

    Only the executed prefix, the first ``stop`` instructions (all when
    None), is inspected: nothing after the last observation is executed, so
    an operation there cannot change an earlier statistic.

    A trajectory evaluates the declared points of one deterministic, noiseless
    coherent evolution, so the circuit body and its readout views must not
    introduce measurement-conditioned evolution, resets, postselection or
    non-unitary channels. A Qiskit ``Gate`` denotes a unitary operation and is
    accepted.
    A barrier or delay does not act on the state. A measurement, a reset or a
    control-flow operation is refused wherever it occurs, including inside the
    definition of a composite instruction, so a composite (``Initialize``
    among them, whose definition resets) cannot hide one. Any other
    instruction is accepted only through its definition, each shared
    definition being inspected once. An opaque instruction without one has no
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
    """Local Qiskit Aer simulator, the default backend of `prepare` and `solve`.

    Build it with `AerBackend()` for noiseless simulation, or with
    `AerBackend.from_noise_model(model)` for a Qiskit Aer noise model, and pass
    it as `backend=` to [`solve`][nwqlib.scientist.solve] or
    [`prepare`][nwqlib.scientist.prepare]. It needs the `aer` extra
    (`pip install "nwqlib[aer]"`) and no credentials. A noise model acts on
    sampled counts only. An exact expectation, probability or amplitude would
    describe neither the noiseless circuit nor the noisy samples, so those readouts
    reject a noisy backend before any circuit is built.

    The backend holds the caller's `NoiseModel` object without copying or
    serializing it, so keep that model unchanged while this backend or its
    prepared circuits are in use. `Run.save` saves the model as the SDK's
    dictionary with NumPy array files, and `load_run` attaches the rebuilt model
    to the reopened Run's own copy of this configuration (`run.backend`), leaving
    the backend passed to `load_run` unchanged. `revise` keeps the attached
    model while `noise_model_id` is unchanged. The [Aer guide](../aer.md) covers
    readouts, noise and circuit inspection.

    Attributes:
        kind: Fixed `"qiskit_aer"`, the backend type.
        noise_model_id: Default `None`, no noise model. `from_noise_model` sets
            it to a fresh UUID that identifies the model object this backend
            holds, not the model's contents. A configuration rebuilt from JSON with this field set
            holds no model and raises when used.
    """

    kind: Literal["qiskit_aer"] = "qiskit_aer"
    noise_model_id: Text | None = None
    _noise_model: object = PrivateAttr(default=None)

    @classmethod
    def from_noise_model(cls, model):
        """Return an Aer backend that applies a Qiskit Aer noise model to sampled counts.

        The model is held without copying, and `qiskit_aer` is imported only by
        this call.

        Args:
            model (qiskit_aer.noise.NoiseModel): The noise model to apply.

        Returns:
            backend (AerBackend): A new backend whose `noise_model_id` is a fresh UUID.

        Raises:
            TypeError: If `model` is not a Qiskit Aer `NoiseModel`.
        """
        from qiskit_aer.noise import NoiseModel
        if not isinstance(model, NoiseModel):
            raise TypeError("from_noise_model requires a Qiskit Aer NoiseModel")
        backend = cls(noise_model_id=str(uuid4()))
        backend._noise_model = model
        return backend

    def _bound_noise_model(self):
        """The attached NoiseModel, or None for a noiseless backend.

        A configuration that has a ``noise_model_id`` but holds no model, for
        example one rebuilt from JSON, raises instead of running noiselessly.
        """
        if self.noise_model_id is not None and self._noise_model is None:
            raise ValueError("Aer noise model is unbound; load_run restores its saved SDK input, "
                             "or use from_noise_model to start a new explicit binding")
        return self._noise_model

    def revise(self, **changes):
        """Revise the configuration, keeping the attached model while ``noise_model_id`` is unchanged.

        ``Record.revise`` and ``Record.model_copy`` rebuild the record from its
        fields, which drops private attributes such as the attached model.
        """
        revised = super().revise(**changes)
        if revised.noise_model_id == self.noise_model_id:
            revised._noise_model = self._noise_model
        return revised

    def model_copy(self, *, update=None, deep=False):
        """Copy the configuration, keeping the attached model as ``revise`` does."""
        copied = super().model_copy(update=update, deep=deep)
        if copied.noise_model_id == self.noise_model_id:
            copied._noise_model = self._noise_model
        return copied

    def target_for(self, observation):
        """Select the counts or statevector target, rejecting a noisy exact readout first.

        Noise acts on sampled counts. An exact expectation, probability or
        amplitude with an attached noise model would describe neither the ideal
        nor the noisy measurement statistics, so it is rejected before the
        circuit is built.
        """
        from nwqlib.backends.targets import AER_COUNTS_TARGET, AER_STATEVECTOR_TARGET
        if self.noise_model_id is not None and observation.kind != "counts":
            raise ValueError("explicit Aer noise supports counts; exact readouts require a noiseless backend")
        self._bound_noise_model()
        return AER_COUNTS_TARGET if observation.kind == "counts" else AER_STATEVECTOR_TARGET

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot, views=None):
        """Build the Aer circuit of one logical circuit and describe the prepared input.

        ``views`` maps the tail and inverse definition of each trajectory
        readout view to its logical circuit on the registers of the circuit
        body. Each is checked like the body and inserted at its view's
        position.

        Translated gate definitions are cached per Run and target, keyed by the
        source gate object, so repeated preparations of one construction
        translate each definition once. ``keep_sources`` drops cache entries
        whose source gates are no longer in the current construction, which
        bounds the cache during parameter scans. On a noise target without the
        unitary instruction, translation takes the exact syntheses of dense
        unitaries from the Run's synthesis cache
        (``Run._exact_dense_unitaries``), so a definition that is translated
        again after ``keep_sources`` dropped it needs no new synthesis. The
        noise model is attached once per Run context. Counts record a
        fixed-seed sampling source because the Plan's runtime seed selects
        Aer's random stream. Amplitude readout requires the unconditional
        pre-final-measurement state, since a measurement-conditioned
        trajectory is not the requested state.
        """
        from nwqlib.backends import qiskit_aer as aer
        target = self.target_for(observation)
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
            # of its own prefix. One serving only phase-invariant reductions
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
        # Release the cached definitions, in every target's cache, whose source
        # gates are not in the current selection (``_AerPreparation.keep_sources``).
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
            # G, the operation count of the exact-probability window
            # (``_validation.exact_probability_window``), counts the native
            # evolution operations of a coherent statevector readout and no save
            # instruction (qiskit_aer._aer_operation_counts).
            # A body with control flow has no fixed native count and keeps its
            # instruction inventory. Shot sampling keeps its own count.
            operations=(aer._aer_operation_counts(native.circuit)[0] if observation.kind != "counts"
                        and not native.circuit.has_control_flow_op() else len(native.circuit.data)),
            counts_sampling=CountsSampling(kind="fixed_seed", seed=runtime.seed) if observation.kind == "counts" else None,
            probability_window_exclusions=native.metadata.get("probability_window_exclusions"),
            # The envelope of the stored phase factors, (0, 0) when no saved
            # state is multiplied, including every non-trajectory readout.
            statevector_roundoff=aer._statevector_roundoff(native),
            transformation="fresh logical container with run-owned definitions; shared Aer lowering/readout; "
                "owned in-process snapshot; selected simulator memory limit, no process RSS claim")

    def admit_trajectory_buffers(self, observation, *, width, memory_mb, run):
        """Check the memory of a trajectory's simulated state, saved results and marginal workspace against ``memory_mb``.

        This is the Aer trajectory check before the Aer circuit is built
        (``qiskit_aer._admit_aer_saved_buffers``). The trajectory memory check
        uses the same positive thread cap passed to Aer's
        ``max_parallel_threads`` (``_run_state_threads``). The cap is part of the
        effective preparation configuration and is reapplied on restoration.
        Saved marginal workspace is counted for at most that many
        state-update threads under the CPU double-statevector execution
        configuration. Returns the counted bytes.
        """
        from nwqlib.backends import qiskit_aer as aer
        return aer._admit_aer_saved_buffers(observation, width=width, memory_mb=memory_mb,
                                            state_threads=_run_state_threads(run))

    def native_target_version(self):
        """Version of the executing package, ``qiskit-aer``, recorded in the target and chunk sources of each preparation record."""
        return version("qiskit-aer")

    def native_job_id_length(self):
        """Encoded length of Aer's job identifier, 36 characters.

        qiskit-aer 0.18.0 generates it with ``str(uuid.uuid4())`` in its own
        ``AerBackend.run``.
        """
        return 36

    def _refuse_inadmissible_trajectory(self, circuit, stop=None):
        """Refuse a trajectory that is not one noiseless coherent evolution, before the circuit is built.

        This is the trajectory admissibility check of the Aer adapter
        (``coherent_body_issue``). A trajectory has zero raw shots, which does
        not allow measurement-conditioned, reset, postselected,
        noisy-trajectory or outcome-dependent evolution. A bound noise model is
        refused earlier, by ``target_for``, for every readout other than
        counts, and ``prepare`` checks the effective simulator method after
        the circuit is built. ``stop`` is the last point's boundary, so only the executed
        prefix of the body is checked.
        """
        issue = coherent_body_issue(circuit, stop)
        if issue is not None:
            raise ValueError(f"an Aer trajectory requires a coherent body; {issue}")

    def submit(self, native, *, submission_id, run):
        """Execute the prepared Aer circuit synchronously and return its raw output."""
        from nwqlib.backends.qiskit_aer import _submit_aer_execution
        return _submit_aer_execution(native)

    def restore_native_data(self, record, circuit, metadata, *, run):
        """Restore the already built Aer circuit and the effective simulator options."""
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
            options["max_parallel_threads"] = metadata["simulator_max_parallel_threads"]
        if self.noise_model_id is not None:
            options["noise_model"] = self._bound_noise_model()
        return _PreparedAerExecution(AerSimulator(**options), circuit, target,
            observation.shots or None,
            (observation.position, observation.labels) if observation.kind == "pauli_expectation" else None,
            observation.qubits if observation.kind == "probabilities" else None, metadata,
            _trajectory_saves(observation, record.boundaries) if trajectory else None)
