"""NWQ-Sim runner build and local detached backend adapter.

NWQLib runs NWQ-Sim through its own small runner (``_native/nwqsim_runner.cpp``)
compiled for one public backend/method pair. Python lowers each circuit to
Qiskit-convention U and CX gates, writes one JSON request into a spool
directory named by the submission UUID, and launches the runner as a detached
process. The runner writes one terminal result file, beside binary files for
amplitudes, probability marginals and saved trajectory states, and a CPU/SV
trajectory request is evolved once and observed at every point. After a crash it leaves
a ``.claim`` directory, which marks an attempt whose outcome must be checked
before anyone reruns it. ``reconcile`` uses the same spool path to bind a lost
launch acknowledgement to its result without starting a second process.
docs/nwqsim.md records the qualified NWQ-Sim revision, the reason revision
``b35763d`` is rejected, and the byte formulas used by ``_buffer_bytes``.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from importlib.metadata import version
import json
from math import isfinite
import os
import platform
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Annotated, ClassVar, Literal
from uuid import UUID

from pydantic import Field, model_validator

from nwqlib.backends.capabilities import BackendCapability, BackendTarget
from nwqlib.core.records import PositiveInt, Record, Source, Text
from nwqlib.execution import CountsSampling
from nwqlib.operators.access import Count

# The runner protocol: request, result, failure packet, --describe and the
# --phase-factor reply. The runner checks the same string; no reader accepts
# another format.
NWQSIM_FORMAT = "nwqlib.nwqsim/3"

# CPU/SV executes trajectories: several positions, readout views, and every
# registered reduction on its saved state (``_trajectory_request``).
NWQSIM_CPU_TARGET = BackendTarget(name="nwqsim_cpu_sv", provider="nwqsim",
    description="NWQ-Sim CPU statevector execution with native sampling or exact readout",
    capabilities=(BackendCapability.STATEVECTOR, BackendCapability.COUNTS, BackendCapability.EXPECTATION),
    native_basis_gates=("u", "cx"),
    readouts=("counts", "pauli_expectation", "probabilities", "amplitudes", "trajectory"),
    readout_features=("multi_position", "views"), reducers="registered",
    artifacts=(NWQSIM_FORMAT,))


def _target(backend, method):
    """Return the declared target. Only CPU/SV reads host amplitudes, so only it has exact readouts."""
    if backend == "CPU" and method == "SV":
        return NWQSIM_CPU_TARGET
    return BackendTarget(name=f"nwqsim_{backend.lower()}_{method.lower()}", provider="nwqsim",
        description=f"NWQ-Sim {backend}/{method} native counts; platform qualification is separate",
        capabilities=(BackendCapability.COUNTS,), native_basis_gates=("u", "cx"),
        readouts=("counts",), artifacts=(NWQSIM_FORMAT,))


def _route(backend, method, ranks):
    """Reject backend/method/rank routes that the NWQ-Sim public factory cannot run as selected.

    NVGPU_MPI SV/DM constructors increment an uninitialized ``gpu_mem`` counter
    at both inspected revisions. The MPI factory silently substitutes SV for
    DM, which would change the selected method. The public SV_MPI constructor
    requires a power-of-two rank count. The table in docs/nwqsim.md lists the
    same routes.
    """
    if backend == "NVGPU_MPI":
        raise ValueError("NVGPU_MPI is blocked by uninitialized gpu_mem in the inspected b35763d and efd0226 sources; no qualified build route")
    if backend not in {"CPU", "MPI", "NVGPU", "AMDGPU"} or method not in {"SV", "DM"}:
        raise ValueError("unsupported NWQ-Sim backend/method")
    if backend == "MPI":
        if method != "SV":
            raise ValueError("NWQ-Sim MPI supports SV only; its factory silently substitutes SV for DM")
        if ranks & (ranks - 1):
            raise ValueError("NWQ-Sim MPI requires a power-of-two rank count")
    elif ranks != 1:
        raise ValueError("only the MPI backend supports multiple cooperating ranks")


def _buffer_bytes(backend, method, ranks, width, observation):
    """Known resident state/work/sample arrays per process, excluding SDK workspace.

    Returns (state bytes, readout bytes). The formulas follow the inspected
    NWQ-Sim constructor and sampler allocations listed in docs/nwqsim.md and
    ENGINEERING_CONSTANTS.md, "Numerical guards and tolerances". A density
    matrix stores 4**q entries. The runner applies the same formulas again
    before allocating state.

    With ``dim`` stored entries (2**q for SV, 4**q for DM) the terms are
    as follows. Array names are those of the inspected NWQ-Sim sources.

    - CPU/SV ``24*dim + 8``: ``sv_real``, ``sv_imag`` and ``m_real``, three
      float64 arrays of ``dim`` entries, with one extra entry in ``m_real``.
    - CPU/DM ``16*dim + 8*(2**q + 1)``: the float64 density arrays ``dm_real``
      and ``dm_imag`` plus ``m_real`` with 2**q + 1 entries over the diagonal.
    - CPU readout: ``16`` bytes per marginal entry (the runner's compensated
      sum) for probabilities, otherwise ``8`` bytes per shot (the sampled
      outcomes, which the runner projects, sorts and counts in place). A
      trajectory holds one marginal's sums at a time, so it charges its
      largest marginal; its saved states and marginals are written to files
      one point at a time, which the output limit admits.
      The marginal leaves the runner as a dense float64 sidecar of
      ``8 * 2**k`` bytes, which the output limit admits (``_result``).
    - MPI/SV ``32*dim // ranks`` per process: four local float64 arrays, and
      ``24*shots`` for three full-shot arrays (global and local outcomes and
      the random draws).
    - GPU SV/DM ``48*dim + 16``: two host and four device float64 arrays, two
      of them with one extra slot, and ``32*shots`` for four full-shot arrays
      (outcomes and random draws, on host and device).
    """
    shots = observation.shots
    dim = 1 << (width if method == "SV" else 2 * width)
    if backend == "CPU":
        state = 24 * dim + 8 if method == "SV" else 16 * dim + 8 * ((1 << width) + 1)
        if observation.kind == "trajectory":
            # The runner holds one marginal's compensated sums at a time.
            return state, max((16 << len(point.qubits) for point in observation.positions
                               if point.kind == "probabilities"), default=0)
        readout = 16 * (1 << len(observation.qubits)) if observation.kind == "probabilities" else 8 * shots
        return state, readout
    if backend == "MPI":
        return 32 * dim // ranks, 24 * shots
    # Two pinned host arrays and four device arrays (two have one extra slot).
    return 48 * dim + 16, 32 * shots


# The exact-probability roundoff constant NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION
# (_validation.py) was derived by reading these files at this revision: the
# binary64 value type (ValType = double, declared in both config.hpp and
# nwq_util.hpp), the fusion switch (ENABLE_FUSION in config.hpp), the U and CX
# gate records the circuit builds (circuit.hpp), the gate-matrix storage
# (private/sim_gate.hpp), the fusion pass, the U/CX gate matrices and the
# CPU/SV kernels. A runner is inside that derivation when its NWQ-Sim revision
# is or descends from the base, so it carries the base's CPU memory-counter and
# U-phase corrections, and when these files are byte-identical to the base. The
# second condition matters because a later upstream commit can change any of
# these premises while still descending, and the revision string alone would
# not show the user that change. build_runner decides both from the git
# checkout and compiles the outcome into the runner. Preparation reads it from
# --describe (_roundoff_exclusions).
_ROUNDOFF_BASE_REVISION = "efd02262ff9c5def4f2eac1df6416c1ee2023d5b"
_ROUNDOFF_SOURCES = (
    "include/config.hpp",
    "include/nwq_util.hpp",
    "include/circuit.hpp",
    "include/private/sim_gate.hpp",
    "include/circuit_pass/fusion.hpp",
    "include/private/gate_factory/sv_gate.hpp",
    "include/svsim/sv_cpu.hpp",
)


def _roundoff_exclusions(build):
    """Receipt exclusions for a runner's ``--describe`` output, from the rule above.

    ``roundoff_base_revision`` names the base when the runner's revision
    descends from it and is null otherwise; ``roundoff_sources_identical``
    says whether ``_ROUNDOFF_SOURCES`` match the base. A description without
    these fields, or checked against another base, is outside the derivation.
    """
    if build.get("roundoff_base_revision") != _ROUNDOFF_BASE_REVISION:
        return {"unchecked NWQ-Sim revision"}
    if build.get("roundoff_sources_identical") is not True:
        return {"changed NWQ-Sim roundoff sources"}
    return set()


def _describe(executable, backend, method):
    """Check the runner's ``--describe`` output against the configured target before use.

    The description must name the configured backend/method pair, its
    readouts, MPI support for the MPI route and enabled GPU error checking for
    GPU routes. Revision ``b35763d`` is rejected because its CPU constructors
    increment an uninitialized memory counter and its U factory can overflow
    the sum of finite phases (docs/nwqsim.md). Passing this check does not
    qualify another revision.
    """
    try:
        described = subprocess.run([executable, "--describe"], check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise FileNotFoundError("NWQ-Sim runner is missing; build it with python -m nwqlib.backends.nwqsim") from error
    build = json.loads(described.stdout)
    readouts = ["pauli" if kind == "pauli_expectation" else kind for kind in _target(backend, method).readouts]
    if build.get("format") != NWQSIM_FORMAT:
        raise ValueError(
            f"selected runner reports protocol format {build.get('format')!r}; NWQLib requires {NWQSIM_FORMAT!r}. "
            f"Rebuild it with python -m nwqlib.backends.nwqsim SOURCE OUTPUT --backend {backend} --method {method} "
            "(SOURCE: a clean NWQ-Sim source checkout; OUTPUT: a new executable path)"
        )
    if (build.get("backends") != [backend]
            or build.get("methods") != [method] or not isinstance(build.get("readouts"), list)
            or sorted(build["readouts"]) != sorted(readouts)
            or build.get("distributed") is not (backend == "MPI")
            or (backend in {"NVGPU", "AMDGPU"} and build.get("gpu_error_check") is not True)):
        raise ValueError("selected runner does not describe the configured native target/readouts")
    if build.get("nwqsim_revision") == "b35763d846e6512ed817d3f88ac8ce79a7e82a7e":
        raise ValueError("NWQ-Sim revision b35763d has undefined behavior in CPU constructors and finite U phase overflow; use qualified corrected source")
    return build


def _phase_factor(executable, phase, *, run):
    """Ask the selected runner for the finite binary64 factor used by its output product."""
    # The scalar reply is at most 128 ASCII bytes. Admit its read and decode.
    run.check_data(256)
    try:
        reply = subprocess.run(
            [executable, "--phase-factor", json.dumps(phase, allow_nan=False)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"NWQ-Sim phase-factor query failed: {error.stderr.strip()}") from error
    if len(reply.stdout) > 256:
        raise ValueError("NWQ-Sim phase-factor reply exceeds its admitted 256 bytes")
    data = json.loads(reply.stdout)
    if data.get("format") != NWQSIM_FORMAT:
        raise ValueError("NWQ-Sim phase-factor query requires the current runner format")
    a, b = data["phase_factor"]
    if not isfinite(a) or not isfinite(b):
        raise ValueError("NWQ-Sim output phase factor must be finite")
    return complex(a, b)


def _phase_envelope(factor, width):
    """Charge the actual output product, including exact identity as a zero-error case."""
    from nwqlib._phase_product import phase_product_envelope
    if factor is None or factor == complex(1.0, 0.0):
        return (0.0, 0.0)
    return phase_product_envelope(factor, 1 << width)


@dataclass(frozen=True)
class _PreparedNWQSim:
    """One runner request. payload is the exact JSON the runner reads, and input_id binds its result."""

    circuit: object
    payload: bytes
    input_id: str
    metadata: dict


class NWQSimExecutionError(RuntimeError):
    """A terminal native failure, carrying the evolution count the runner reported, if valid.

    The count is kept so that a failed attempt still records the evolutions it
    actually performed. An invalid count becomes None rather than a guess.
    """

    def __init__(self, message, *, native_simulations=None):
        super().__init__(message)
        self.native_simulations = native_simulations if type(native_simulations) is int and native_simulations >= 0 else None


class NWQSimBackend(Record):
    """Explicit native target, local launch and finite file/known-array allowances.

    This configuration is inert. Preparation compiles the selected circuit;
    the common Run owns every launch and the stored-data limit.
    backend selects CPU, MPI, NVGPU or AMDGPU; method selects SV or DM (MPI
    requires SV). ranks counts cooperating MPI processes; mpi_launcher names
    the installed launcher for local MPI execution. max_buffer_bytes covers
    known state/work/sample arrays per process, including host plus device
    storage for GPUs, excluding fused gates, SDK workspace and whole-process RSS.
    optimization_level is the Qiskit transpiler level of the U/CX lowering.
    A level other than the default 0 is recorded as the receipt exclusion
    "optimization_level", so the receipt has no certified state error and its
    error numbers are a reference; its native operation count describes the
    compiled circuit. Levels 2 and 3 resynthesize two-qubit blocks with
    Qiskit's own synthesis and receive the logical circuit as given; an exact
    readout or trajectory is refused when they relabel wires by eliding
    permutations, since the runner reads the output in logical wire order.
    At optimization levels 2 and 3, a measured circuit can supply
    computational-basis probabilities when lowering reports no layout.
    Amplitude readouts require level 0 or 1, or a circuit without
    measurements, because removing terminal diagonal gates can change
    relative phases.
    """

    kind: Literal["nwqsim"] = "nwqsim"
    supports_synchronous: ClassVar[bool] = False
    backend: Literal["CPU", "MPI", "NVGPU", "AMDGPU", "NVGPU_MPI"] = "CPU"
    method: Literal["SV", "DM"] = "SV"
    ranks: PositiveInt = 1
    mpi_launcher: Text = "mpirun"
    executable: Text
    spool: Text
    max_input_bytes: PositiveInt
    max_output_bytes: PositiveInt
    max_buffer_bytes: PositiveInt
    optimization_level: Annotated[Count, Field(le=3)] = 0

    @model_validator(mode="after")
    def _paths(self):
        """Reject an unsupported route, and relative runner or spool paths.

        The spool path identifies a detached job after the Python process
        exits, so it must not depend on the working directory.
        """
        _route(self.backend, self.method, self.ranks)
        if not Path(self.executable).is_absolute() or not Path(self.spool).is_absolute():
            raise ValueError("NWQ-Sim runner and spool require explicit absolute paths")
        return self

    def target_for(self, observation):
        """The declared target of the configured backend/method pair.

        A trajectory whose points this route cannot serve is refused here,
        before any preparation charge: an amplitude point (its point array is
        not published by the detached refresh), and a reduction declared at an
        explicit position before the last point's explicit position whose
        reducer is not registered phase invariant (no phase ledger).
        ``_trajectory_request`` repeats the reduction check on the resolved
        boundaries, which also covers the end shorthand.
        """
        target = _target(self.backend, self.method)
        if observation is not None and observation.kind == "trajectory" and "trajectory" in target.readouts:
            name = f"NWQ-Sim {self.backend}/{self.method}"
            last = observation.positions[-1].position
            for point in observation.positions:
                if point.kind == "amplitudes":
                    raise ValueError(f"{name} does not publish the amplitude point {point.id!r} of a detached "
                                     "trajectory; use AerBackend, or observe the state in a separate single-endpoint "
                                     "amplitudes experiment")
                if (point.kind == "reduction" and point.position is not None and last is not None
                        and point.position < last):
                    _refuse_phase_variant_reduction(name, point)
        return target

    @property
    def qualification_notice(self):
        """The once-per-Run warning that names the execution host and spool."""
        return (f"NWQ-Sim {self.backend}/{self.method} runs on execution host {platform.node()}; "
                f"that host or VM must remain running until completion. Results are stored under {self.spool}. "
                "An available native target does not establish GPU, MPI or site qualification.")

    def native_job_id_length(self):
        """The local job identifier is the canonical 36-character submission UUID."""
        return 36

    def _lower(self, logical, *, run, seed, exact, amplitudes=False):
        """Lower one logical circuit to U/CX at the selected optimization level.

        Levels 0 and 1 receive the exact synthesis of each dense unitary
        (``Run._exact_dense_unitaries``); levels 2 and 3 resynthesize two-qubit
        blocks with Qiskit's own synthesis, so they keep the logical circuit
        as given (docs/dependency_issues.md). Without a coupling map, levels 2
        and 3 may elide SWAP and permutation gates and relabel the wires that
        follow them, which Qiskit reports as a layout. Measurements follow the
        relabeling, so counts stay correct and their receipt records the final
        layout, but an exact readout (``exact``) reads the state in logical wire
        order and is refused.

        Levels 2 and 3 can remove diagonal gates immediately before terminal
        measurements. This preserves computational-basis probabilities and their
        marginals but can change the pre-measurement amplitudes. Amplitude readouts
        of measured circuits therefore require level 0 or 1, or a circuit without
        measurements. Exact probability readouts remain subject to the separate
        refusal whenever lowering reports a layout.

        For a computational-basis diagonal unitary D=diag(d_x) with |d_x|=1,
        |(D psi)_x|^2 = |psi_x|^2 for every x. Summing over unobserved
        coordinates gives the same equality for every computational-basis
        marginal, including a proper-prefix marginal and an arbitrary ordered
        subset. For a mixed state the diagonal of D rho D^dagger is also
        unchanged. This requires the removed gates to be terminal relative to
        the affected measured wires, with no later operation that can turn
        their phases into the requested population. NWQ-Sim's accepted
        single-endpoint circuits have terminal measurements only and reject
        reset and dynamic control. Qiskit's RemoveDiagonalGatesBeforeMeasure
        pass removes diagonal one- and two-qubit gates (checked with Qiskit
        2.5.2). Counts use the actual post-lowering measurement map, and Pauli
        readouts refuse a measured circuit at every level.
        """
        from qiskit import transpile
        from qiskit.circuit import Measure
        if (amplitudes and self.optimization_level >= 2
                and any(isinstance(item.operation, Measure) for item in logical.data)):
            raise ValueError(
                f"NWQ-Sim optimization_level={self.optimization_level} removes diagonal gates before "
                "measurements. Amplitude readouts at levels 2 and 3 need a circuit without measurements, "
                "or use optimization_level 0 or 1"
            )
        native = transpile(run._exact_dense_unitaries(logical) if self.optimization_level <= 1 else logical,
                           basis_gates=["u", "cx"], optimization_level=self.optimization_level, seed_transpiler=seed)
        if exact and native.layout is not None:
            raise ValueError(f"NWQ-Sim optimization_level={self.optimization_level} relabeled wires by eliding "
                             "permutations; exact readouts need optimization_level 0 or 1")
        return native

    def _exclusions(self, exclusions):
        """Add the marker of a non-default optimization level to a receipt's exclusions."""
        return exclusions | ({"optimization_level"} if self.optimization_level != 0 else set())

    def _build(self, context):
        """The runner's checked ``--describe`` output, run once per Run and kept in the backend context."""
        if "build" not in context:
            context["build"] = _describe(self.executable, self.backend, self.method)
        return context["build"]

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot, views=None):
        """Admit one observation, lower it to U/CX and encode the runner request.

        The readout, width, classical-bit, MPI rank and shot, and state-buffer
        admissions run before the runner is described or the circuit is
        lowered. The MPI global-gate transfer check runs during lowering, and
        the input-byte limit applies while the request is encoded
        (``_stream_request``). An amplitude endpoint, or a trajectory with a
        reduction at its last boundary, asks the runner once for its output
        phase factor after lowering (``_phase_factor``) and records that
        factor's envelope in ``statevector_roundoff``. A Pauli readout keeps only the
        instruction prefix before its original position. Measurements must be
        terminal, and their clbit-to-qubit map travels with the request so the
        runner builds count keys in NWQLib order. The runtime seed selects
        both the transpiler seed and the native sampler seed. A trajectory
        (CPU/SV) is lowered once and executed as one evolution
        (``_trajectory_request``); ``views`` maps each view's tail and inverse
        definition to its logical circuit on the body's registers.
        """
        context = run._state["backend_context"]
        from qiskit.circuit import Barrier, ControlFlowOp, Measure, Reset
        from qiskit.circuit.library import CXGate, UGate
        from nwqlib.backends.connection import NativePreparation, circuit_layout, environment
        from nwqlib.backends.capabilities import unsupported_readout
        target = self.target_for(observation)
        if observation.kind == "trajectory":
            if "trajectory" not in target.readouts:
                observation.reject_unsupported_schedule(f"NWQ-Sim {self.backend}/{self.method}")
            unsupported = unsupported_readout(target, observation)
            if unsupported is not None:
                raise ValueError(f"NWQ-Sim {self.backend}/{self.method} does not execute {unsupported}")
        if observation.kind not in target.readouts:
            raise ValueError(f"NWQ-Sim {self.backend}/{self.method} does not support {observation.kind}; only CPU/SV has exact readouts")
        # Representability guards for signed 64-bit state indices, the 64-bit
        # count key and MPI's int count (ENGINEERING_CONSTANTS.md, "Numerical
        # guards and tolerances"). They do not promise an allocatable state.
        # SV at 58 qubits and DM at 29 qubits both store 2**58 entries, so one
        # entry-count limit serves both methods. The runner applies the same
        # width guard.
        if circuit.num_qubits < 1 or circuit.num_qubits > (58 if self.method == "SV" else 29):
            raise ValueError("NWQ-Sim state index/byte products exceed their representable width")
        if circuit.num_clbits > 63:
            raise ValueError("NWQ-Sim native counts support at most 63 classical bits")
        if self.backend == "MPI" and (circuit.num_qubits < self.ranks.bit_length() + 1 or observation.shots > 2**31 - 1):
            raise ValueError("NWQ-Sim MPI requires at least two local qubits and an MPI int-sized shot count")
        state_bytes, readout_bytes = _buffer_bytes(self.backend, self.method, self.ranks, circuit.num_qubits, observation)
        if state_bytes + readout_bytes > self.max_buffer_bytes:
            raise ValueError("native state/readout arrays exceed the selected byte limit")
        build = self._build(context)
        if observation.kind == "trajectory":
            return self._trajectory_request(circuit, observation, runtime, position, views, run, snapshot, build, target)
        # A Pauli readout names an original logical instruction boundary. Only
        # its prefix determines the requested expectation.
        selected = circuit
        if observation.kind == "pauli_expectation" and position != len(circuit.data):
            selected = circuit.copy_empty_like()
            selected.data = circuit.data[:position]
        if any(isinstance(item.operation, (ControlFlowOp, Reset)) for item in selected.data):
            raise ValueError("NWQ-Sim does not support dynamic control or reset in the selected observation prefix")
        # Qiskit would lower a dense UnitaryGate with its inexact synthesis.
        # The Run makes the exact synthesis of each distinct matrix once per Run
        # object and reserves each new one against its max_synthesis_work first.
        native = self._lower(
            selected, run=run, seed=runtime.seed,
            exact=observation.kind != "counts",
            amplitudes=observation.kind == "amplitudes",
        )
        # The measurement map is part of the request's fixed fields, which are
        # admitted before any gate is written, so it is read first.
        measurements = {native.find_bit(item.clbits[0]).index: native.find_bit(item.qubits[0]).index
                        for item in native.data if isinstance(item.operation, Measure)}
        local_dim = (1 << native.num_qubits) // self.ranks

        def gates():
            """Yield each native U/CX gate as its request object, one at a time, in circuit order."""
            measured = False
            for item in native.data:
                operation = item.operation
                if isinstance(operation, Barrier):
                    continue
                if isinstance(operation, Measure):
                    measured = True
                    continue
                if measured or not isinstance(operation, (UGate, CXGate)):
                    raise ValueError("NWQ-Sim requires typed U/CX gates and terminal measurements only")
                params = [float(value) for value in operation.params]
                if any(not isfinite(value) for value in params):
                    raise ValueError("NWQ-Sim gate arguments must be finite")
                wires = [native.find_bit(bit).index for bit in item.qubits]
                if (self.backend == "MPI" and local_dim > 2**31 - 1
                        and any(1 << wire >= local_dim for wire in wires)):
                    raise ValueError("NWQ-Sim MPI global gates require an int-sized local state transfer count")
                yield dict(name=operation.name, qubits=wires, params=params)
        if observation.kind == "counts" and not measurements:
            raise ValueError("NWQ-Sim counts require actual terminal measurements")
        if observation.kind == "pauli_expectation" and measurements:
            raise ValueError("NWQ-Sim exact Pauli readout cannot follow a measurement")
        phase = float(native.global_phase)
        if not isfinite(phase):
            raise ValueError("NWQ-Sim global phase must be finite")
        input_id = snapshot
        kind = "pauli" if observation.kind == "pauli_expectation" else observation.kind
        readout = dict(kind=kind, labels=list(observation.labels),
            qubits=list(measurements.values()) if kind == "counts" else list(observation.qubits),
            clbits=list(measurements) if kind == "counts" else [])
        head = dict(format=NWQSIM_FORMAT, input_id=input_id, backend=self.backend, method=self.method, ranks=self.ranks,
            num_qubits=native.num_qubits, num_clbits=native.num_clbits)
        factor = _phase_factor(self.executable, phase, run=run) if kind == "amplitudes" else None
        tail = dict(phase_factor=None if factor is None else [factor.real, factor.imag],
            observation=readout, shots=observation.shots, seed=runtime.seed,
            max_buffer_bytes=self.max_buffer_bytes, max_output_bytes=self.max_output_bytes)
        payload, operations = _stream_request(head, gates(), tail, min(self.max_input_bytes, run.limits.max_data_bytes))
        metadata = dict(readout_population="unconditional", statevector_semantics="pre_final_measurement" if self.method == "SV" else None,
                        nwqsim_revision=build["nwqsim_revision"], native_simulations=None,
                        native_readout=kind, requested_shots=observation.shots, num_qubits=native.num_qubits,
                        marginal_width=len(observation.qubits) if kind == "probabilities" else None)
        prepared = _PreparedNWQSim(native, payload, input_id, metadata)
        return NativePreparation(native=prepared,
            target=Source(name=target.name, version=build["nwqsim_revision"],
                          domain=target.description, reference=self.executable),
            compiler=Source(name="Qiskit U/CX lowering", version=version("qiskit"),
                domain=f"optimization_level={self.optimization_level}; runtime seed selects compilation and sampling; native C++ {build['compiler']}; executable architecture {build.get('architecture', 'unknown')}",
                reference="qiskit.transpile and NWQLib native runner"),
            native_basis=("u", "cx"), environment=environment(("nwqlib", "numpy", "qiskit")),
            quantum_layout=circuit_layout(native, native.qregs), classical_layout=circuit_layout(native, native.cregs),
            logical_to_native=(tuple(range(circuit.num_qubits)) if native.layout is None
                              else tuple(native.layout.final_index_layout())), operations=operations, population="unconditional",
            transformation=("exact dense-unitary synthesis and " if self.optimization_level <= 1 else "")
                + f"Qiskit U/CX lowering; original observation boundary and logical bit order; {self.backend}/{self.method} native kernels",
            payload=payload, payload_format=NWQSIM_FORMAT,
            statevector_roundoff=_phase_envelope(factor, native.num_qubits),
            counts_sampling=CountsSampling(kind="fixed_seed", seed=runtime.seed) if kind == "counts" else None,
            probability_window_exclusions=None if kind == "counts" and self.optimization_level == 0 else tuple(sorted(
                self._exclusions(({"amplitude-derived masses"} if kind == "amplitudes" else set())
                                 | _roundoff_exclusions(build)))))

    def _trajectory_request(self, circuit, observation, runtime, boundaries, views, run, snapshot, build, target):
        """Lower a trajectory once and encode it as one runner evolution observed at every point.

        Evaluate all selected point readouts while advancing one coherent
        simulator state. A point's boundary precedes its reversible readout
        view. Pauli saves leave the state unchanged. A probability view applies
        its selected tail, saves the marginal, and applies the exact inverse
        before continuation. Count actual native operations through the final
        observation, including all executed view operations, and admit the sum
        of the point payloads before acquisition.

        Trajectory readout evaluates the declared points of one deterministic,
        noiseless coherent evolution. The body and every view must pass
        ``connection.coherent_body_issue`` (no measurement, reset, control flow
        or opaque instruction, also inside composite definitions); the CPU/SV
        runner has no noise and evolves a pure state. The body is cut after the
        last point, since nothing after the last observation is executed, and a
        full-width barrier labeled ``nwqlib_boundary_<b>`` marks each distinct
        boundary b of the bound body before Qiskit lowers it to U/CX once, at
        the adapter's optimization level: saves are placed at the original bound circuit
        boundaries before lowering can move independent gates across them, and
        each boundary's native gate position is read from its barrier, without
        constructing any prefix again. Each view's tail is lowered once, and
        its inverse is the reversed, adjointed tail of that same lowered
        sequence (U(t, p, l) becomes U(-t, -l, -p); CX is its own inverse), so
        the declared inverse definition is not lowered separately. The runner
        applies an inverse only when a later point follows.

        The CPU/SV roundoff model charges the U/CX gates actually executed
        through the observation, including earlier reversible readout
        excursions. Segment-local fusion uses no more than the charged fold and
        application operations. The model remains first order and subject to
        the receipt's exclusions. The receipt's G is therefore the body's U/CX
        gates through the last point plus every forward tail and every inverse
        that runs, and the runner reports the same count back. Save
        instructions add readout work but no coherent gate charge, and neither
        G nor the number of simulations grows with the number of points.

        A phase-sensitive intermediate output uses the phase ledger of its
        actual boundary-preserving native construction. Scalar reducers
        declared invariant under one common input phase can consume raw saved
        amplitudes. Relative branch phases remain part of the coherent
        preparation. Qiskit's lowering reports one global phase for the whole
        lowered prefix and no ledger per segment, so a reduction point before
        the last boundary is served only for a reducer registered
        ``phase_invariant`` and reads the raw saved amplitudes; a saved state at
        the last boundary is multiplied, as an output copy, by the phase of the
        lowered prefix through that boundary, which includes the logical
        circuit's own top-level phase. Amplitude points are refused: their
        detached point-array publication is not part of the refresh route.
        """
        from qiskit.circuit import Barrier
        from qiskit.circuit.library import CXGate, UGate
        from nwqlib.backends.connection import NativePreparation, circuit_layout, coherent_body_issue, environment
        name = f"NWQ-Sim {self.backend}/{self.method}"
        last = boundaries[-1]
        issue = coherent_body_issue(circuit, last)
        if issue is None:
            issue = next((f"view {key!r}: {found}" for key, tail in (views or {}).items()
                          if (found := coherent_body_issue(tail)) is not None), None)
        if issue is not None:
            raise ValueError(f"{name} trajectory requires a coherent body; {issue}")
        for point, boundary in zip(observation.positions, boundaries, strict=True):
            if point.kind == "reduction" and boundary != last:
                _refuse_phase_variant_reduction(name, point)
        marks = set(boundaries)
        marked = circuit.copy_empty_like()
        for index, item in enumerate(circuit.data[:last]):
            if index in marks:
                marked.append(Barrier(circuit.num_qubits, label=f"nwqlib_boundary_{index}"), marked.qubits)
            marked.append(item)
        if last in marks:
            marked.append(Barrier(circuit.num_qubits, label=f"nwqlib_boundary_{last}"), marked.qubits)

        def lowered(logical):
            """Lower one logical circuit to U/CX at the selected level (``_lower``)."""
            return self._lower(logical, run=run, seed=runtime.seed, exact=True)

        native = lowered(marked)
        # Native gate position of each boundary, read from its barrier.
        position, native_boundaries = 0, {}
        for item in native.data:
            operation = item.operation
            if isinstance(operation, Barrier):
                label = operation.label or ""
                if label.startswith("nwqlib_boundary_"):
                    native_boundaries[int(label.removeprefix("nwqlib_boundary_"))] = position
                continue
            if not isinstance(operation, (UGate, CXGate)):
                raise ValueError(f"{name} requires typed U/CX gates in a trajectory body")
            position += 1
        if set(native_boundaries) != marks:
            raise ValueError("Qiskit lowering changed the trajectory's boundary markers")

        def gate_list(logical):
            """U/CX request gates of one lowered view tail."""
            tail = lowered(logical)
            gates = []
            for item in tail.data:
                operation = item.operation
                if isinstance(operation, Barrier):
                    continue
                if not isinstance(operation, (UGate, CXGate)):
                    raise ValueError(f"{name} requires typed U/CX gates in a readout view")
                params = [float(value) for value in operation.params]
                if any(not isfinite(value) for value in params):
                    raise ValueError("NWQ-Sim gate arguments must be finite")
                gates.append(dict(name=operation.name, qubits=[tail.find_bit(bit).index for bit in item.qubits],
                                  params=params))
            return gates

        def inverse(gates):
            """The reversed, adjointed sequence: U(t, p, l) -> U(-t, -l, -p), CX -> CX."""
            return [dict(name=gate["name"], qubits=gate["qubits"],
                         params=[] if gate["name"] == "cx" else [-gate["params"][0], -gate["params"][2], -gate["params"][1]])
                    for gate in reversed(gates)]

        phase = float(native.global_phase)
        if not isfinite(phase):
            raise ValueError("NWQ-Sim global phase must be finite")
        corrected = any(point.kind == "reduction" and boundary == last
                        for point, boundary in zip(observation.positions, boundaries, strict=True))
        factor = _phase_factor(self.executable, phase, run=run) if corrected else None
        factor_pair = None if factor is None else [factor.real, factor.imag]
        tails, points, executed = {}, [], position
        kinds = {"pauli_expectation": "pauli", "probabilities": "probabilities", "reduction": "state"}
        for index, (point, boundary) in enumerate(zip(observation.positions, boundaries, strict=True)):
            view = None
            if point.view is not None:
                if point.view.tail not in tails:
                    tails[point.view.tail] = gate_list(views[point.view.tail])
                forward = tails[point.view.tail]
                view = dict(tail=forward, inverse=inverse(forward))
                executed += len(forward) * (2 if index + 1 < len(boundaries) else 1)
            points.append(dict(id=point.id, kind=kinds[point.kind], boundary=native_boundaries[boundary],
                labels=list(point.labels), qubits=list(point.qubits) if point.kind == "probabilities" else [],
                phase_factor=factor_pair if point.kind == "reduction" and boundary == last else None, view=view))

        def gates():
            """Yield each body U/CX gate through the last point as its request object."""
            for item in native.data:
                operation = item.operation
                if isinstance(operation, Barrier):
                    continue
                params = [float(value) for value in operation.params]
                if any(not isfinite(value) for value in params):
                    raise ValueError("NWQ-Sim gate arguments must be finite")
                yield dict(name=operation.name, qubits=[native.find_bit(bit).index for bit in item.qubits], params=params)

        head = dict(format=NWQSIM_FORMAT, input_id=snapshot, backend=self.backend, method=self.method, ranks=self.ranks,
            num_qubits=native.num_qubits, num_clbits=native.num_clbits)
        tail = dict(observation=dict(kind="trajectory", points=points), shots=0, seed=runtime.seed,
            max_buffer_bytes=self.max_buffer_bytes, max_output_bytes=self.max_output_bytes)
        payload, _ = _stream_request(head, gates(), tail, min(self.max_input_bytes, run.limits.max_data_bytes))
        metadata = dict(readout_population="unconditional", statevector_semantics="pre_final_measurement",
                        nwqsim_revision=build["nwqsim_revision"], native_simulations=None, native_readout="trajectory",
                        requested_shots=0, num_qubits=native.num_qubits, marginal_width=None,
                        operations=executed, points=_trajectory_points(observation))
        segments = len(marks)
        return NativePreparation(native=_PreparedNWQSim(native, payload, snapshot, metadata),
            target=Source(name=target.name, version=build["nwqsim_revision"],
                          domain=target.description, reference=self.executable),
            compiler=Source(name="Qiskit U/CX lowering", version=version("qiskit"),
                domain=f"optimization_level={self.optimization_level}; runtime seed selects compilation; native C++ {build['compiler']}; executable architecture {build.get('architecture', 'unknown')}",
                reference="qiskit.transpile and NWQLib native runner"),
            native_basis=("u", "cx"), environment=environment(("nwqlib", "numpy", "qiskit")),
            quantum_layout=circuit_layout(native, native.qregs), classical_layout=circuit_layout(native, native.cregs),
            logical_to_native=tuple(range(circuit.num_qubits)), operations=executed, population="unconditional",
            transformation=(("exact dense-unitary synthesis and " if self.optimization_level <= 1 else "")
                            + f"one Qiskit U/CX lowering of the body through its last "
                            f"point with {segments} boundary marker(s); one {self.backend}/{self.method} evolution of "
                            f"{position} body gates in {segments} segment(s); view tails lowered once with "
                            f"reversed adjoint inverses; G={executed} counts every executed forward tail and inverse; "
                            "no phase ledger: reductions before the last boundary read raw saved amplitudes of "
                            "phase-invariant reducers, and a saved state at the last boundary carries the lowered "
                            "prefix phase"),
            payload=payload, payload_format=NWQSIM_FORMAT,
            statevector_roundoff=_phase_envelope(factor, native.num_qubits),
            # A reduction save hands the saved amplitudes to a host reducer,
            # whose masses the native readout derivation does not cover; a
            # reducer whose own arithmetic bounds them resolves this label
            # (Reducer.state_error_resolutions). A trajectory of only
            # probability and Pauli points carries no such label.
            probability_window_exclusions=tuple(sorted(self._exclusions(
                _roundoff_exclusions(build)
                | ({"amplitude-derived masses"}
                   if any(point.kind == "reduction"
                          for point in observation.positions) else set())
            ))))

    def admit_batch(self, natives):
        """Admit exactly one prepared circuit, since one runner request evolves one circuit."""
        if len(natives) != 1:
            raise ValueError("one NWQ-Sim runner request executes exactly one prepared circuit")

    def launch(self, natives, *, submission_id, run):
        """Create the spool directory once, write the request and start the runner detached.

        ``mkdir(exist_ok=False)`` makes a repeated launch for the same UUID
        fail before a second process starts. The process runs in its own
        session, so it outlives the Python process, and the locator records
        its PID and host for later liveness checks.
        """
        from nwqlib.execution import JobLocator
        self.admit_batch(natives)
        native, = natives
        if str(UUID(submission_id)) != submission_id:
            raise ValueError("local native submission requires its canonical UUID")
        directory = Path(self.spool) / submission_id
        # Input is written once and read by every cooperating rank. Rank zero
        # writes the combined JSON/artifact output once; fetch admits its read.
        payload_bytes = (1 + self.ranks) * len(native.payload) + self.max_output_bytes
        run.check_data(payload_bytes)
        directory.mkdir(parents=True, exist_ok=False)
        source, output = directory / "input.json", directory / "result.json"
        with source.open("xb") as stream:
            stream.write(native.payload)
        command = [self.executable, str(source), str(output), str(self.max_input_bytes)]
        if self.backend == "MPI":
            command = [self.mpi_launcher, "-n", str(self.ranks), *command]
        process = subprocess.Popen(command,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        run._state["backend_context"].setdefault("processes", {})[submission_id] = process
        return JobLocator(provider=self.kind, job_id=submission_id, process_id=process.pid, host=platform.node())

    def restore_native(self, record, payload=None, *, run):
        """Restore the result association from a receipt, checking saved request bytes when supplied.

        A supplied payload must match the receipt's backend, method, ranks,
        seed, shots, snapshot, width and readout, so a saved request cannot be
        paired with another preparation.
        """
        if record.payload.format != NWQSIM_FORMAT:
            raise ValueError("prepared NWQ-Sim payload has an unsupported format")
        if record.target.name != self.target_for(record.observation).name:
            raise ValueError("prepared NWQ-Sim target differs from its configured backend/method")
        kind = "pauli" if record.observation.kind == "pauli_expectation" else record.observation.kind
        width = sum(len(register.bits) for register in record.native_quantum_layout)
        if payload is not None:
            # The saved request bytes and the text that json.loads decodes from
            # them, one payload-sized copy each (an untuned allowance).
            run.check_data(2 * len(payload))
            data = json.loads(payload)
            if (data.get("format") != NWQSIM_FORMAT or data.get("backend") != self.backend or data.get("method") != self.method
                    or type(data.get("ranks")) is not int or data["ranks"] != self.ranks
                    or data.get("seed") != record.runtime.seed or data.get("shots") != record.observation.shots
                    or data.get("input_id") != record.snapshot or data.get("num_qubits") != width
                    or data["observation"]["kind"] != kind):
                raise ValueError("prepared NWQ-Sim bytes differ from their original snapshot/runtime/readout")
        # Fetch needs only the saved association, never circuit deserialization.
        trajectory = kind == "trajectory"
        return _PreparedNWQSim(None, payload, record.snapshot,
            dict(readout_population=record.population, statevector_semantics="pre_final_measurement" if self.method == "SV" else None,
                 nwqsim_revision=record.target.version, native_simulations=None, native_readout=kind,
                 requested_shots=record.observation.shots, num_qubits=width,
                 marginal_width=len(record.observation.qubits) if kind == "probabilities" else None,
                 **(dict(operations=record.native_operations, points=_trajectory_points(record.observation))
                    if trajectory else {})))

    def refresh(self, locator, natives, *, run):
        """Read the original spool once and report pending, completed, failed or uncertain.

        Without a result file, a process known to have exited, or a PID absent
        on the same host, makes the outcome uncertain. The run is never
        relaunched. A result file is decoded against the original request.
        """
        from nwqlib.backends.connection import BackendRefresh
        self.admit_batch(natives)
        native, = natives
        if locator.provider != self.kind or str(UUID(locator.job_id)) != locator.job_id:
            raise ValueError("native locator does not identify this backend's request")
        output = Path(self.spool) / locator.job_id / "result.json"
        process = run._state["backend_context"].get("processes", {}).get(locator.job_id)
        code = None if process is None else process.poll()
        if not output.is_file():
            if code is not None:
                return BackendRefresh("uncertain", failure="native process exited without a terminal result")
            if locator.process_id is not None and locator.host == platform.node():
                try:
                    os.kill(locator.process_id, 0)
                except ProcessLookupError:
                    return BackendRefresh("uncertain", failure="native process is absent and no terminal result is available")
            return BackendRefresh("acknowledged", provider_status="terminal result unavailable")
        # Every actual result read has its own finite allowance; status-only
        # checks do not reserve a nonexistent output transfer.
        run.check_data(self.max_output_bytes)
        try:
            result = self._result(output, native=native, job_id=locator.job_id, returncode=code or 0)
        except NWQSimExecutionError as error:
            return BackendRefresh("failed", provider_status="failed", failure=str(error),
                                  native_simulations=error.native_simulations)
        return BackendRefresh("completed", results=(("0", result),), provider_status="completed", native_simulations=1)

    def can_reconcile(self, submission, *, run):
        """Whether automatic waiting has an original process or terminal output.

        A directory alone does not establish that Popen ran. Without an
        observable process or result, report the uncertain outcome instead of
        polling forever. A later explicit resume can still bind output that
        arrives at the original path, without replacing or refunding the job.
        This checks file metadata and an existing child handle only.
        """
        output = Path(self.spool)/submission.submission_id/"result.json"
        process = run._state["backend_context"].get("processes", {}).get(submission.submission_id)
        return output.is_file() or (process is not None and process.poll() is None)

    def reconcile(self, submission, natives, *, run):
        """Bind a lost launch acknowledgement to the result file at the original spool path, if any.

        The deterministic spool location of the submission UUID binds the
        terminal result to the saved intent; no replacement process is
        launched. The result is not read here: ``refresh`` parses it once and
        checks its ``input_id`` against the original request before any value
        is used, and a mismatch fails there.
        """
        from nwqlib.execution import JobLocator
        self.admit_batch(natives)
        if str(UUID(submission.submission_id)) != submission.submission_id:
            raise ValueError("native reconciliation requires its original UUID")
        output = Path(self.spool) / submission.submission_id / "result.json"
        if not output.is_file():
            return None
        return JobLocator(provider=self.kind, job_id=submission.submission_id, host=platform.node())

    def _packet(self, output, native):
        """Admit and parse a result file that names this request and, when completed, this build.

        A completed packet must report exactly one native evolution, since the
        runner evolves the circuit once regardless of shots or ranks.
        """
        if output.stat().st_size > self.max_output_bytes:
            raise ValueError("native result exceeds its byte limit")
        data = json.loads(output.read_bytes())
        if (data.get("format") != NWQSIM_FORMAT or data.get("input_id") != native.input_id
                or data.get("status") not in {"completed", "failed"}):
            raise ValueError("NWQ-Sim result differs from its actual prepared input")
        if data["status"] == "completed" and (
                data.get("backend") != self.backend or data.get("method") != self.method
                or type(data.get("ranks")) is not int or data["ranks"] != self.ranks
                or data.get("nwqsim_revision") != native.metadata["nwqsim_revision"]
                or type(data.get("native_simulations")) is not int or data["native_simulations"] != 1):
            raise ValueError("NWQ-Sim completed result has an unexpected native target/ranks/build or evolution population")
        return data

    def _result(self, output, *, native, job_id, returncode=0):
        """Decode a completed packet, or raise NWQSimExecutionError for a failed or mismatched one.

        A nonzero process or scheduler exit blocks success even when a
        completed packet exists. Amplitudes and probability marginals come
        from binary sidecars (``_sidecar``): interleaved complex128 amplitudes
        and a dense float64 marginal of ``2**k`` values whose index bit zero
        is the first observed qubit. A marginal, of a single endpoint or of a
        trajectory point (``_trajectory_values``), reaches the decoder as that
        dense buffer.
        """
        from nwqlib.backends.results import BackendRunResult
        from nwqlib.execution import ExecutionMode
        if not output.is_file():
            raise RuntimeError(f"NWQ-Sim exited {returncode} without a terminal result; reconcile {output.parent}")
        data = self._packet(output, native)
        if data.get("status") != "completed" or returncode:
            raise NWQSimExecutionError(f"NWQ-Sim execution failed: {data.get('error', 'unknown native error')}",
                                      native_simulations=data.get("native_simulations"))
        kind = data["kind"]
        if kind != native.metadata["native_readout"] or data["shots"] != native.metadata["requested_shots"]:
            raise NWQSimExecutionError("NWQ-Sim result differs from its requested readout/shot population", native_simulations=1)
        if kind == "amplitudes":
            values = {"statevector": self._sidecar(output, data[kind], kind, 1 << native.metadata["num_qubits"])}
        elif kind == "probabilities":
            values = {kind: self._sidecar(output, data[kind], kind, 1 << native.metadata["marginal_width"])}
        elif kind == "trajectory":
            values = {kind: self._trajectory_values(output, data, native)}
        else:
            values = {"pauli_expectations" if kind == "pauli" else kind: data[kind]}
        return BackendRunResult(execution_mode=ExecutionMode.SHOTS if kind == "counts" else ExecutionMode.STATEVECTOR,
            backend_target=self.target_for(None), raw_output=values,
            metadata=dict(native.metadata, native_job_id=job_id, native_simulations=1))


    def _sidecar(self, output, descriptor, name, count, *, earlier=0):
        """Read one binary sidecar ``<result>.<name>`` after checking its declared layout and bytes.

        ``name`` ends in ``amplitudes`` (``complex128-native``) or
        ``probabilities`` (``float64-native``). The descriptor must name the
        file beside the result, its native dtype, the admitted element count
        and bytes, and the file must have exactly those bytes. The result, the
        ``earlier`` bytes of sidecars already read for it and this sidecar must
        together fit ``max_output_bytes``, which ``refresh`` reserved before
        this read.
        """
        import numpy as np
        dtype = (("complex128-native", np.complex128) if name.endswith("amplitudes")
                 else ("float64-native", np.float64))
        path = output.with_name(output.name + "." + name)
        if (type(descriptor) is not dict or descriptor.get("file") != path.name or descriptor.get("dtype") != dtype[0]
                or descriptor.get("count") != count or descriptor.get("bytes") != np.dtype(dtype[1]).itemsize * count
                or path.stat().st_size != descriptor["bytes"]
                or output.stat().st_size + earlier + descriptor["bytes"] > self.max_output_bytes):
            raise ValueError(f"NWQ-Sim {name} artifact differs from its admitted layout/bytes")
        return np.fromfile(path, dtype=dtype[1])

    def _trajectory_values(self, output, data, native):
        """Each point's values under its point ID, from one completed trajectory packet.

        The packet must list the prepared points in schedule order and report
        the prepared executed gate count G. Pauli values are keyed by
        ``(point_id, label)``; a marginal is read from its dense float64 file
        and handed to the shared trajectory decoder as that buffer, ``2**k``
        values with outcome j at position j (bit zero the first observed
        qubit), and a saved state from its complex128 file as
        ``statevector``, which the shared decoder reduces with the point's
        registered reducer.
        """
        entries, points = data.get("trajectory"), native.metadata["points"]
        if (type(entries) is not list or len(entries) != len(points)
                or any(type(entry) is not dict or (entry.get("id"), entry.get("kind")) != (point_id, kind)
                       for entry, (point_id, kind, _) in zip(entries, points))
                or data.get("executed_gates") != native.metadata["operations"]):
            raise NWQSimExecutionError("NWQ-Sim trajectory result differs from its prepared points or gate count",
                                      native_simulations=1)
        values, read = {}, 0
        for index, (entry, (point_id, kind, width)) in enumerate(zip(entries, points, strict=True)):
            if kind == "pauli":
                values[point_id] = {"pauli_expectations": entry["pauli"]}
            elif kind == "probabilities":
                dense = self._sidecar(output, entry["probabilities"], f"p{index}.probabilities", 1 << width, earlier=read)
                read += dense.nbytes
                values[point_id] = {"probabilities": dense}
            else:
                state = self._sidecar(output, entry["amplitudes"], f"p{index}.amplitudes",
                                      1 << native.metadata["num_qubits"], earlier=read)
                read += state.nbytes
                values[point_id] = {"statevector": state}
        return values


def _refuse_phase_variant_reduction(name, point):
    """Refuse a reduction before the last boundary unless its reducer is registered phase invariant."""
    from nwqlib.core.planning import READOUT_REDUCERS
    reducer = READOUT_REDUCERS.get(point.reducer)
    # An unregistered reducer is left to the capability check, which names it.
    if reducer is not None and not reducer.phase_invariant:
        raise ValueError(f"{name} has no phase ledger for the reduction point {point.id!r} before the last "
                         f"boundary; its reducer {point.reducer!r} is not registered phase invariant; move the point to "
                         "the last boundary, or register the reducer with phase_invariant=True when it is invariant")


def _trajectory_points(observation):
    """The (point ID, runner kind, marginal width) of each trajectory point, in schedule order."""
    kinds = {"pauli_expectation": "pauli", "probabilities": "probabilities", "reduction": "state"}
    return tuple((point.id, kinds[point.kind], len(point.qubits) if point.kind == "probabilities" else None)
                 for point in observation.positions)


def _streamed_gate_bound(name, wires, params, *, preceded_by_comma):
    """Largest compact JSON size of one U or CX request gate, checked before it is encoded.

    The U/CX request writer admits each compact JSON gate before encoding or
    writing it. A U gate costs at most ``134+digits(q)`` bytes and a CX gate at
    most ``38+digits(q0)+digits(q1)``, plus its list separator. Fixed request
    fields and the closing suffix are reserved separately. The writer
    validates finite binary64 parameters and verifies each actual encoded
    length against the bound.

    Derivation, for the fields ``name``, ``qubits`` and ``params`` in compact
    ASCII JSON with separators ``(',', ':')``, finite binary64 parameters and
    nonnegative integer wires: the empty-list U skeleton
    ``{"name":"u","qubits":[],"params":[]}`` has 36 bytes; insert one wire,
    three float strings of at most 32 bytes each and two parameter commas, so
    ``B_U = 36 + d(q) + 3*32 + 2 = 134 + d(q)``. The CX skeleton is one byte
    longer, with two wires, one comma and an empty parameter list, so
    ``B_CX = 37 + d(q0) + d(q1) + 1 = 38 + d(q0) + d(q1)``. One byte is added
    for the preceding list comma except at the first gate. The 32-byte float
    envelope is the binary64 JSON envelope of ``_run_journal._json_bound``;
    signs, exponent and decimal point are inside it. The bound covers no other
    spelling (spaces, indentation, custom float formatting or extra fields).
    """
    if any(type(q) is not int or q < 0 for q in wires):
        raise ValueError("wire indices must be nonnegative native integers")
    if any(type(x) is not float or not isfinite(x) for x in params):
        raise ValueError("gate parameters must be finite native binary64")
    if name == "u" and len(wires) == 1 and len(params) == 3:
        size = 134 + len(str(wires[0]))
    elif name == "cx" and len(wires) == 2 and not params:
        size = 38 + sum(len(str(q)) for q in wires)
    else:
        raise ValueError("streaming requires the admitted typed U/CX gate")
    return size + int(preceded_by_comma)


def _write_gate(stream, gate, *, used, limit, suffix_bytes, first):
    """Admit one gate by ``_streamed_gate_bound``, then encode and write it; return the bytes used."""
    bound = _streamed_gate_bound(gate["name"], gate["qubits"], gate["params"],
                                preceded_by_comma=not first)
    if bound > limit-used-suffix_bytes:
        raise ValueError("streamed request exceeds remaining input-byte allowance")
    body = json.dumps(gate, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")
    data = (b"" if first else b",") + body
    if len(data) > bound:
        raise ValueError("gate encoder exceeds its established byte bound")
    stream.write(data)
    return used + len(data)


def _stream_request(head, gates, tail, limit):
    """Encode the runner request with its gates written one at a time; return (bytes, gate count).

    The request is the compact journal JSON of ``head``, then ``"gates"``,
    then ``tail``, in that key order, so it is the same bytes as encoding the
    whole request object at once. The fixed fields of ``head`` and ``tail``
    are bounded and encoded first, and the closing suffix stays reserved
    while each gate is admitted by ``_write_gate`` before it is written. No
    list of all gate objects and no second text copy of the request is made.
    ``limit`` is the smaller of the input-byte limit and the Run's data limit.
    """
    from io import BytesIO
    from nwqlib._run_journal import encode
    prefix = encode(head, limit)[:-1].encode("utf-8") + b',"gates":['
    suffix = b"]," + encode(tail, limit)[1:].encode("utf-8")
    if len(prefix) + len(suffix) > limit:
        raise ValueError("streamed request exceeds remaining input-byte allowance")
    stream = BytesIO()
    stream.write(prefix)
    used, count = len(prefix), 0
    for gate in gates:
        used = _write_gate(stream, gate, used=used, limit=limit, suffix_bytes=len(suffix), first=not count)
        count += 1
    stream.write(suffix)
    return stream.getvalue(), count


def build_runner(source: str | Path, output: str | Path, *, backend: str = "CPU", method: str = "SV", compiler: str | None = None) -> Path:
    """Build one selected native route against an explicit clean source checkout.

    This explicit operation invokes the existing C++17 compiler. It installs or
    downloads nothing and builds no separate frontend. The executable's
    ``--describe`` output keeps the NWQ-Sim revision, the actual compiler and
    whether the exact-probability roundoff derivation covers that revision
    (``_ROUNDOFF_BASE_REVISION``). Ordinary backend imports never call this
    operation.

    Args:
        source: NWQ-Sim source checkout containing its bundled headers.
        output: New executable path; an existing file is not overwritten.
        backend: CPU, MPI, NVGPU or AMDGPU public factory route.
        method: SV or DM; MPI requires SV.
        compiler: Existing compiler; defaults to c++, mpic++, nvcc or hipcc for the selected route.

    Returns:
        Absolute path of the newly built executable.
    """
    _route(backend, method, 1)
    compiler = compiler or {"CPU": "c++", "MPI": "mpic++", "NVGPU": "nvcc", "AMDGPU": "hipcc"}[backend]
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"native runner already exists: {output}")
    if not (source / "include" / "backendManager.hpp").is_file():
        raise ValueError("NWQ-Sim source must contain include/backendManager.hpp")
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(f"C++17 compiler not found: {compiler}")
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout
    # The build includes every header under include/, so an untracked or
    # ignored header there would enter a runner that reports this revision.
    dirty += subprocess.run(
        ["git", "-C", str(source), "ls-files", "--others", "--", "include"],
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise ValueError("record a clean NWQ-Sim revision before building its runner")
    # Roundoff derivation rule (_ROUNDOFF_BASE_REVISION). merge-base exits 0
    # when the base is HEAD or its ancestor, 1 when it is not, and 128 when the
    # base commit is absent from this checkout, which cannot show descent.
    roundoff = []
    if subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", _ROUNDOFF_BASE_REVISION, "HEAD"],
                      check=False, capture_output=True).returncode == 0:
        # ls-tree lists mode, blob id and path of each derivation source; a
        # changed or deleted file changes the listing.
        base, head = (subprocess.run(["git", "-C", str(source), "ls-tree", commit, "--", *_ROUNDOFF_SOURCES],
                                     check=True, capture_output=True, text=True).stdout
                      for commit in (_ROUNDOFF_BASE_REVISION, "HEAD"))
        roundoff = [f'-DNWQLIB_ROUNDOFF_BASE_REVISION="{_ROUNDOFF_BASE_REVISION}"',
                    f"-DNWQLIB_ROUNDOFF_SOURCES_IDENTICAL={int(base == head)}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    native = Path(__file__).with_name("_native") / "nwqsim_runner.cpp"
    with TemporaryDirectory(prefix=".nwqsim-build-", dir=output.parent) as directory:
        built = Path(directory) / output.name
        flags = {"CPU": [], "MPI": ["-DMPI_ENABLED"],
                 "NVGPU": ["-DCUDA_ENABLED", "-DGPU_ERROR_CHECK", "-x", "cu"],
                 "AMDGPU": ["-DHIP_ENABLED", "-DGPU_ERROR_CHECK", "-x", "hip"]}[backend]
        subprocess.run([
            executable, "-std=c++17", "-O3", *flags, "-I", str(source / "include"),
            f'-DNWQLIB_BACKEND="{backend}"', f'-DNWQLIB_METHOD="{method}"',
            f'-DNWQLIB_NWQSIM_REVISION="{revision}"', *roundoff, str(native), "-o", str(built),
        ], check=True)
        # Linking within the same directory/filesystem makes publication atomic
        # and refuses a concurrently created output instead of replacing it.
        os.link(built, output)
    return output


def _main():
    """Command-line entry for ``build_runner``: ``python -m nwqlib.backends.nwqsim SOURCE OUTPUT``."""
    import argparse

    parser = argparse.ArgumentParser(description="Build one NWQLib NWQ-Sim native target")
    parser.add_argument("source", type=Path, help="clean NWQ-Sim source checkout")
    parser.add_argument("output", type=Path, help="new native executable path")
    parser.add_argument("--backend", choices=("CPU", "MPI", "NVGPU", "AMDGPU"), default="CPU")
    parser.add_argument("--method", choices=("SV", "DM"), default="SV")
    parser.add_argument("--compiler", help="existing target compiler; defaults to c++, mpic++, nvcc or hipcc")
    arguments = parser.parse_args()
    print(build_runner(arguments.source, arguments.output, backend=arguments.backend, method=arguments.method, compiler=arguments.compiler))


if __name__ == "__main__":
    _main()
