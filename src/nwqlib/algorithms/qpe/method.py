"""QPE Methods: input checks, the controlled-evolution Program and estimate reconstruction.

Every estimator observes the signal z_p = <psi|U^p|psi> through one
ancilla, with U = exp(-i*tau*H) for a Hamiltonian H or the polar base of the
supplied unitary. An exact static Plan reads Re z_p and Im z_p as the ancilla
X and Y expectations at the power positions of one controlled trajectory. A
sampled Hadamard test turns the measured mean <Z> into Re(exp(i*s)*z_p) with
its ancilla phase s. Reading order for one ``solve`` call:

1. ``_QPEMethod.plan`` checks the input (``_inputs``), chooses tau
   (``_safe_time`` or the Pauli l1 bound), separates and prunes Pauli terms
   (``_split_pauli_terms``), draws the acquisition schedule
   (numerical.planned_power_schedule), selects the controlled powers once
   per Plan (powers.select_powers) and builds either the quantum Program
   (``_quantum_program``) or the classical host kernel
   (``_host_construction``).
2. The shared Run executes the Program. RWPE instead runs its one-bit
   feedback loop in controller.py.
3. ``_samples`` reduces saved observations to one ancilla mean per query,
   and ``_analysis`` applies the estimator kernel in numerical.py and
   derives the phase in turns and the output-frame interval from its value
   (records.public_estimate).

Size laws passed to ``_QPEMethod._admit`` are counted before the work they
describe. They are allowances in scalar-operation and byte units, not
measured CPU time or memory (docs/CODE_TOUR.md, "Byte and work budgets").
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

from dataclasses import dataclass
from math import fsum, isfinite, pi
from typing import Annotated, ClassVar, Literal
import numpy as np
from pydantic import Field, field_validator
from nwqlib._validation import NUMERICAL_RELATION_RTOL
from nwqlib.algorithms.protocol import Method, AlgorithmDescriptor, ApplicabilityError
from nwqlib.blocks import SelectedConstruction, select_preparation
from nwqlib.blocks.kernels import BoundKernel, KernelOutput
from nwqlib.blocks.records import SelectedKernel
from nwqlib.core.planning import Plan, Experiment, ObservationPoint, ObservationSpec, ReadoutDetails
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.core.records import Float64, FrozenArray, Source, PositiveInt, Real, Nonnegative
from nwqlib.evidence import Evidence, Fact
from nwqlib.evidence.error_model import ErrorModel, ErrorTerm, FramedFact, exact_readout_sampling
from nwqlib.execution import KernelApplication, ScalarValue
from nwqlib.ir import (
    Allocate,
    Argument,
    BlockCall,
    ClassicalStage,
    ClassicalValue,
    Constant,
    Definition,
    ExprRef,
    Expression,
    Measure,
    MeasurementBatch,
    MetadataRef,
    Parameter,
    ParameterRef,
    PortMap,
    Program,
    Register,
    Release,
    Repeat,
    Sequence,
    Setting,
    Binding,
)
from nwqlib.operators import OperatorInput
from nwqlib.operators.access import Count
from nwqlib.operators.inputs import ingest_dense
from nwqlib.operators._pauli import pauli_coefficients
from nwqlib.problems import Eigenphase, Eigenproblem, Eigenvalue, SpectralEstimation
from nwqlib.problems.inputs import StateInput, ingest_vector
from nwqlib.problems.records import StateData
from nwqlib.resources import ResourceLaw, Workspace
from . import numerical
from .powers import polar_base, select_auxiliary, select_powers
from .records import (
    METHOD,
    DESCRIPTOR,
    REFERENCES,
    QPEAnalysis,
    QPEExposure,
    QPEQuery,
    QPEReconstruction,
    QPESample,
    RWPEGaussian,
    bounded_mean,
    identification_note,
    public_estimate,
    query_at,
    validate_selection,
)

_ANALYSIS_SOURCE = Source(
    name="qpe.analysis",
    version="3",
    domain="actual scalar samples or saved RWPE Gaussian moments",
    reference="nwqlib.algorithms.qpe.method",
)
_QCELS_ANALYSIS_SOURCE = Source(
    name="qpe.qcels.analysis",
    version="3",
    domain="actual complex observations; finite search without an uncertainty interval",
    reference="Ding and Lin, arXiv:2211.11973v2 Eq.(2); dimensionless amplitude-eliminated fit",
)
HOST = Source(
    name="qpe.nominal_overlaps",
    version="1",
    domain="estimator signal from exact eigendecomposition",
    reference="one eigensystem and prepared spectral weights; no independent baseline or quantum execution",
)
# Error contributions that no QPE estimator composes into a bound in the
# requested output frame. _unknown_error_model records each one as unknown.
ERROR_SOURCES = (
    "spectral_identification",
    "aliasing",
    "sampling",
    "estimator_model",
    "finite_grid",
    "controlled_evolution",
    "native_roundoff",
)


@dataclass(frozen=True)
class _Input:
    """Native inputs bound to a Plan for execution, archives and verification.

    Attributes:
        target: Admitted operator on the power-of-two register, after the
            padding of ``_QPEMethod._inputs``.
        reference: Admitted initial state on the same register.
        tau: Time step of U = exp(-i*tau*H), or None for unitary input.
        base: For unitary input, the admitted polar factor V = polar(A) of
            the target, the base of every power and spectral consumer. None
            for Hamiltonian input.
    """

    target: OperatorInput
    reference: StateInput
    tau: float | None
    base: OperatorInput | None = None

    @property
    def spectral(self):
        """The operator whose spectrum defines the signal: V for unitary input, else the target."""
        return self.target if self.base is None else self.base


def _new_context():
    """Return an empty run-local spectral cache.

    Keys: ``target`` (operator identity and decomposition kind),
    ``eigensystem`` (read-only eigenvalues and eigenvectors),
    ``preparation`` (the preparation record the reference belongs to),
    ``reference`` ((vector, basis index) from _reference_data),
    ``weights`` (numerical.spectral_weights) and ``squares`` (the base's
    identity and its cached binary squares, powers.cached_unitary_power;
    None until the first unitary power). The host kernel and dense powers
    fill it lazily and share it within one Run. Verification starts from a
    copy of the context saved with the Result, which keeps no squares.
    """
    return {
        "target": None,
        "eigensystem": None,
        "preparation": None,
        "reference": None,
        "weights": None,
        "squares": None,
    }


def _check_context(context, target, kind, preparation=None):
    """Check actual immutable dependencies; numeric tau is not one of them."""
    if context["target"] is not None and context["target"] != (target, kind):
        raise ValueError("live QPE context belongs to another target or decomposition kind")
    if (
        preparation is not None
        and context["preparation"] is not None
        and context["preparation"] != preparation
    ):
        raise ValueError("live QPE reference belongs to another exact preparation")


def _eigensystem(target, matrix, *, kind, context):
    """Reuse target setup within the admitted live-run/standalone lifetime.

    Within one context, the host signal, the dense Hamiltonian powers and
    explicit verification share this one decomposition. The arrays are made
    read-only so that sharing cannot let one consumer alter another's
    spectrum. Returns the eigensystem and whether this call computed it.
    """
    _check_context(context, target, kind)
    created = context["eigensystem"] is None
    if created:
        eigensystem = numerical.nominal_eigensystem(matrix, kind=kind)
        for array in eigensystem:
            array.flags.writeable = False
        context.update(target=(target, kind), eigensystem=eigensystem)
    return context["eigensystem"], created


def _scalar_reference_available(state):
    """Return whether ``state`` is a nonzero one-dimensional vector with stored entries.

    A dimension-one problem has a scalar target and no system register, so
    the Hadamard test needs no preparation circuit and the reference
    direction is read directly from the admitted vector.
    """
    return (
        isinstance(state, StateInput)
        and state.manifest.basis.dimension == 1
        and state.manifest.reference.representation == "vector"
        and "entries" in state.manifest.access
        and state._direction is not None
        and state.preparation.physical_scale.mantissa != 0
    )


def _reference_data(state):
    """Return the prepared reference as (vector, None) or (None, basis index).

    The classical route and explicit verification need the reference
    direction without executing its preparation circuit. An occupation-number
    state returns its basis index, with qubit q contributing bit q, so that
    spectral_weights reads one eigenvector row. A product state is assembled
    with np.kron(pair, vector), which puts the first listed pair on qubit 0,
    the rightmost Kronecker factor in Qiskit's little-endian order. A supplied circuit has no classical
    direction here and is rejected rather than simulated.
    """
    if _scalar_reference_available(state):
        return state._direction, None
    implementation = state.preparation.implementation
    if implementation == "qiskit.occupation":
        index = sum(int(bit) << q for q, bit in enumerate(state._physical))
        return None, index
    if implementation == "qiskit.direct":
        return state._direction, None
    if implementation == "qiskit.product":
        vector = np.ones(1, dtype=complex)
        for pair in state._direction:
            vector = np.kron(pair, vector)
        vector.flags.writeable = False
        return vector, None
    raise ValueError(
        "QPE classical has no classical reference access; supplied circuits are not materialized"
    )


def _spectrum(data, context):
    """Return ((values, vectors, weights), setup, projected) from the run-local context.

    The eigendecomposition and the projection of the reference each run at
    most once per context. ``setup`` and ``projected`` report whether this
    call performed them, and the host kernel records both counts so that a
    report distinguishes reuse from new work.
    """
    kind = "unitary" if data.tau is None else "hamiltonian"
    preparation = data.reference.preparation
    spectral = data.spectral
    _check_context(context, spectral.manifest.reference, kind, preparation)
    (values, vectors), setup = _eigensystem(
        spectral.manifest.reference, spectral.dense_array(), kind=kind, context=context
    )
    if context["reference"] is None:
        reference = _reference_data(data.reference)
        context.update(preparation=preparation, reference=reference)
    projected = context["weights"] is None
    if projected:
        reference, occupation = context["reference"]
        context["weights"] = numerical.spectral_weights(vectors, reference, occupation=occupation)
        context["weights"].flags.writeable = False
    return (values, vectors, context["weights"]), setup, projected


def _safe_time(matrix, supplied, *, phase_radius=pi):
    """Return (tau, R, selection label) for a dense Hamiltonian.

    R = ||H||_inf, the largest absolute row sum, bounds every |E| because
    any induced norm bounds the spectral radius. The automatic time
    tau = 0.9*phase_radius/R keeps every scaled eigenvalue tau*E inside
    [-0.9*phase_radius, 0.9*phase_radius]. QCELS, RFE and RWPE use
    phase_radius = pi, so tau*E stays on one branch of the eigenphase.
    SPE passes numerical.spe_phase_radius. The 0.9 margin is an untuned
    default registered in docs/ENGINEERING_CONSTANTS.md (QPE automatic time),
    not a paper constant. A supplied time is returned unchanged, and R is
    still reported for the aliasing check. A zero matrix returns tau = 1 and
    R = 0. R is None when it overflows binary64. This reads the stored dense
    input and performs no eigensolve.
    """
    # Scaled row-sum calculation avoids complex scaling underflow and
    # overflowing an otherwise representable conservative time.
    scale = max(float(np.max(np.abs(matrix.real))), float(np.max(np.abs(matrix.imag))))
    if scale == 0:
        return (
            (1.0 if supplied is None else supplied),
            0.0,
            "zero Hamiltonian" if supplied is None else "supplied",
        )
    magnitudes = np.hypot(matrix.real / scale, matrix.imag / scale)
    scaled_bound = float(np.max(np.sum(magnitudes, axis=1)))
    with np.errstate(over="ignore"):
        bound = float(np.multiply(scale, scaled_bound))
        tau = float(min(np.divide(0.9 * phase_radius / scaled_bound, scale), np.finfo(float).max))
    return (
        (tau if supplied is None else supplied),
        (bound if isfinite(bound) else None),
        ("scaled absolute row-sum bound" if supplied is None else "supplied"),
    )


def _analysis_size(options, count, *, power_span, complex_points):
    """Return (work, bytes) declared for one estimator analysis, or for RWPE its controller run.

    ``count`` is the number of scalar samples (acquisition settings) and
    ``complex_points`` the number N of distinct complex points after the two
    quadratures of each power are combined. Planning admits this size before
    the schedule is drawn, and reanalysis admits it again before a changed
    grid is evaluated, so max_work and max_bytes bound the analysis before it
    runs. The values are operation-size units, not measured CPU time
    (docs/ENGINEERING_CONSTANTS.md, QCELS finite complex search).
    """
    if options.estimator == "qcels":
        grid = numerical.qcels_grid_size(options.grid_size, power_span)
        # Work. One objective evaluation costs 32 size units per complex
        # point (exponential, product, mean, centered squares and reduction).
        # The G grid points plus at most G local minima, each refined with
        # QCELS_BRACKET_EVALUATIONS evaluations, give at most
        # (1 + QCELS_BRACKET_EVALUATIONS)*G evaluations. 64 units per sample
        # cover the reduction of samples to complex points.
        # Bytes. An allowance of 32 per grid point (grid and scores), 256 per
        # sample and 128 per complex point (samples, grouping and row
        # temporaries).
        return (
            32 * (1 + numerical.QCELS_BRACKET_EVALUATIONS) * grid * complex_points
            + 64 * max(1, count),
            32 * grid + 256 * max(1, count) + 128 * complex_points,
        )
    if options.estimator == "rwpe":
        # RWPE analysis reads only the saved Gaussian moments, so the law
        # covers the controller instead. It charges 2 units when it starts
        # and 32 for each choice and each update (controller.py), and the last
        # update runs after the last shot. The floor of 64 covers the 16 units
        # of the final analysis (_analysis) when max_steps is 0. On resume an
        # interrupted choice or update is charged again, and the controller
        # admits that charge when it happens. Bytes are a constant allowance
        # per step.
        return max(64, 2 + 64*options.max_steps), 256+256*options.max_steps
    elif options.estimator == "rfe":
        grid, rows = options.num_frequencies, count
    else:
        grid = options.grid_size
        rows = count
    # SPE and RFE evaluate one complex row of length `grid` per drawn power
    # (numerical.spe_cdf, numerical.rfe), and each draw has two scalar
    # samples. The law charges 16 units per grid point and sample plus 64 per
    # sample for pooling. Bytes: an allowance of 160 per grid point for the
    # coordinates, accumulator and row temporaries, and 128 per sample.
    return 16 * grid * max(1, rows) + 64 * max(1, rows), 160 * grid + 128 * rows


def _native_blocks(data, blocks):
    """Return (prep, h, identity_phase, feedback, all blocks) for the quantum Program.

    ``prep`` is the reference preparation on the system register, or None
    for a scalar target. ``h``, ``identity_phase`` and ``feedback`` are the
    one-qubit ancilla blocks of powers.select_auxiliary. The last entry is
    prep (when present), h, identity_phase and feedback followed by the
    selected power blocks, the order in which the Program declares their
    signatures.
    """
    width = data.target.manifest.basis.dimension.bit_length() - 1
    prep = select_preparation("preparation", data.reference) if width else None
    h, identity, feedback = select_auxiliary(
        basis=data.target.manifest.basis, reference=data.target.manifest.reference
    )
    return (
        prep,
        h,
        identity,
        feedback,
        (*((prep,) if prep is not None else ()), h, identity, feedback, *blocks),
    )


def trajectory_labels(width):
    """Return the full-width ancilla X and Y labels, with ancilla qubit 0 rightmost."""
    return "I" * width + "X", "I" * width + "Y"


def _quantum_program(data, queries, selection, options, shots, route):
    """Build the quantum Program of the selected route from its selected power constructions.

    ``route`` is ``trajectory``, ``sampled`` or ``rwpe``.

    Exact static trajectory. For a fixed unitary V, normalized phi and
    nonnegative integer powers p_1 < ... < p_K, the body prepares once
    |chi_0> = (|0>|phi> + |1>|phi>)/sqrt2 (preparation and ancilla H).
    Between powers p_(i-1) and p_i it applies controlled V exactly
    p_i - p_(i-1) times, taking p_0 = 0. Since the two control projectors are
    orthogonal, controlled(A) controlled(B) = controlled(AB). At position p,
    |chi_p> = (|0>|phi> + |1>V^p|phi>)/sqrt2, z_p = <phi|V^p|phi>,
    <X_a> = Re z_p and <Y_a> = Im z_p. With ancilla qubit zero the full
    labels are ``"I"*n+"X"`` and ``"I"*n+"Y"`` (``trajectory_labels``), read
    at one ``pauli_expectation`` point per distinct power. A final phase s
    and H read Re(exp(i*s)*z_p), so phases 0 and -pi/2 give
    exactly these quadratures. Those phase/H tails are unnecessary with X/Y
    saves. Leaving them in the continuing circuit without undoing them would
    change subsequent signals. Nothing follows the last point. Power 0 is
    observed after the preparation and H as an actual acquisition. The
    base-oracle cost is P_max controlled applications, compared with
    sum_i p_i for separately prepared powers with combined X/Y readout, or
    2*sum_i p_i for two quadrature circuits; preparation occurs once and
    there are 2K scalar outputs. A dense gap is one controlled gap-power
    block ``power_p<k>`` of the selected base (powers.select_powers). A
    product-formula gap repeats the one shared step by the difference of the
    cumulative counts r(p_i) - r(p_(i-1)) and applies the identity phase
    increment P(-c_I*(p_i-p_(i-1))*val(tau)) on the control, the binary64
    value the common-step recheck certifies; an increment accumulated per
    gap does not double count earlier phase, as adding -c_I*t_i at every
    position would. Gap node ids are ``gap_<p>``.

    Sampled static Hadamard tests. Positive-shot X and Y acquisitions still
    require separate preparations. Each query's body is prepare, H,
    ``selected_<p>``, its quadrature RZ(s), H and the ancilla measurement
    (Ding and Lin, arXiv:2211.11973v2, Fig. 1, p. 3): with ancilla phase s
    the ancilla mean is <Z> = Re(exp(i*s) <psi|U^p|psi>), so s = 0 and
    s = -pi/2 (W = S^dagger) give the real and imaginary parts of their
    Eqs. (4)-(5) (p. 8). RZ(s) equals P(s) up to a global phase of the whole
    circuit. A sampled setting starts from its own preparation and does not
    share the propagated state at earlier powers, so ``selected_<p>``
    applies the power's own independently selected step repeated ``steps``
    times with the control phase P(-p*tau*c_I) for a product formula, and
    the dense power block of p for a dense power. A counts batch requests
    ``shots * query.multiplicity`` repetitions, the only place where the
    multiplicity of pooled SPE and RFE draws enters the requested
    population.

    RWPE. Each query is one Hadamard test at its continuous time with a
    symbolic ``feedback`` phase that its controller binds, its own
    independently selected step repeated ``steps`` times and the control
    phase P(-p*tau*c_I), because controlled(exp(i*a) V) equals P(a) on the
    control followed by controlled V.

    Program layout. Definition ids name the emitted nodes: ``allocate_*`` and
    ``release_*`` for the registers, ``prepare`` and ``h``, ``identity_<p>``
    phases, ``gap_<p>``, ``selected_<p>``, the block calls named by their
    signatures (``common_step`` with ``common_step_<p>`` repeats,
    ``power_p<k>``), ``feedback_<i>`` with its expression ``quadrature_<i>``
    or ``feedback_value``, ``measure_<i>`` and one ``batch_<i>`` per sampled
    or RWPE query, or ``body_trajectory`` and ``batch_trajectory``.

    Returns:
        (construction, experiments, native blocks): the SelectedConstruction
        holding the Program, its Experiments and the blocks bound to the
        Program's signatures.
    """
    width = data.target.manifest.basis.dimension.bit_length() - 1
    prep, h, identity, feedback, native = _native_blocks(data, selection.blocks)
    definitions, expressions = [], []

    def define(name, node):
        """Append a Program definition and return its id for later references."""
        definitions.append(Definition(id=name, node=node))
        return name

    def call(name, block, ports, angle=None):
        """Define a call of ``block`` with (port, wire) pairs and an optional angle expression id."""
        args = (
            () if angle is None else (Argument(parameter="angle", value=ExprRef(expression=angle)),)
        )
        return define(
            name,
            BlockCall(
                signature=block.record.signature.name,
                ports=tuple(PortMap(port=port, wire=wire) for port, wire in ports),
                arguments=args,
            ),
        )

    def phase(tag, value):
        """Define the control phase P(value) with its constant expression ``tag``."""
        expressions.append(Expression(id=tag, value=Constant(value=Float64(value=value))))
        return call(tag, identity, (("register", "ancilla"),), tag)

    allocate = tuple(
        define("allocate_" + name, Allocate(wire=name))
        for name in (("ancilla", "system") if width else ("ancilla",))
    )
    release = tuple(
        define("release_" + name, Release(wire=name))
        for name in (("ancilla", "system") if width else ("ancilla",))
    )
    prepare = (call("prepare", prep, (("system", "system"),)),) if prep is not None else ()
    hadamard = call("h", h, (("register", "ancilla"),))
    controls = (("control", "ancilla"),) + ((("system", "system"),) if width else ())
    calls = {}

    def block_call(block):
        """Define one call node per selected power block and reuse it."""
        name = block.record.signature.name
        if name not in calls:
            calls[name] = call(name, block, controls)
        return calls[name]

    powers = sorted({query.power for query in queries})
    records = {record.power: record for record in selection.powers}
    gap_nodes, gap_lengths = {}, {}
    if route == "trajectory":
        previous = 0
        for power in powers:
            children, length = [], 0
            if records[power].backend == "trotter_error_budgeted":
                angle = selection.increments[power]
                if angle:
                    children.append(phase(f"identity_{power}", angle))
                    length += 1
                delta = records[power].steps - (records[previous].steps if previous in records else 0)
                if selection.step is not None and delta:
                    children.append(define(f"common_step_{power}",
                                           Repeat(body=block_call(selection.step), count=delta)))
                    length += delta
            elif power != previous:
                children.append(block_call(selection.dense[power - previous]))
                length += 1
            if children:
                gap_nodes[power] = define(f"gap_{power}", Sequence(children=tuple(children)))
            gap_lengths[power] = length
            previous = power
    power_nodes = {}
    if route in ("rwpe", "sampled"):
        for power in powers:
            record = records[power]
            children = []
            if record.backend == "trotter_error_budgeted" and power and data.tau is not None:
                # The scalar identity contribution is controlled phase, never an
                # unconditional global phase and never duplicated in Pauli terms.
                value = selection.increments[power]
                if value:
                    children.append(phase(f"identity_{power}", value))
            if record.selected_definition is not None:
                block = next(item for item in selection.blocks
                             if item.record.content_id == record.selected_definition)
                node = call(block.record.signature.name, block, controls)
                if record.steps:
                    node = define(node + "_repeat", Repeat(body=node, count=record.steps))
                children.append(node)
            power_nodes[power] = define(f"selected_{power}", Sequence(children=tuple(children)))
    if route == "trajectory":
        # One point per distinct power, after the preparation, the H and the
        # gaps up to that power.
        position = len(prepare) + 1
        points = []
        for power in powers:
            position += gap_lengths[power]
            points.append(ObservationPoint(id=f"power_{power}", position=position,
                                           kind="pauli_expectation", labels=trajectory_labels(width)))
        body = define("body_trajectory", Sequence(
            children=(*allocate, *prepare, hadamard, *(gap_nodes[p] for p in powers if p in gap_nodes),
                      *release)))
        batch = define("batch_trajectory", MeasurementBatch(
            body=body,
            settings=(Setting(label="trajectory",
                              metadata=MetadataRef(format=METHOD, data=data.target.manifest.reference)),),
            repetitions=1,
            observation_kind="trajectory",
        ))
        experiments = (Experiment(name="trajectory", batch=batch, setting_index=0,
                                  readout=ReadoutDetails(positions=tuple(points))),)
        root = define("schedule", Sequence(children=(batch,)))
        parameters, classical = (), ()
    else:
        # Static quadratures have fixed phases. RWPE leaves feedback symbolic
        # until the controller binds its next experiment.
        parameters = ()
        if options.estimator == "rwpe":
            parameters = (Parameter(name="feedback", domain="real"),)
            expressions.append(
                Expression(id="feedback_value", value=ParameterRef(parameter="feedback"))
            )
        # H-controlled-U-phase-H maps interference to the ancilla Z counts.
        batches, experiments = [], []
        for index, query in enumerate(queries):
            angle = "feedback_value"
            if options.estimator != "rwpe":
                angle = f"quadrature_{index}"
                expressions.append(
                    Expression(id=angle, value=Constant(value=Float64(value=query.phase_shift)))
                )
            rotation = call(f"feedback_{index}", feedback, (("register", "ancilla"),), angle)
            measure = define(f"measure_{index}", Measure(wire="ancilla", result="readout"))
            children = (*allocate, *prepare, hadamard, power_nodes[query.power], rotation, hadamard,
                        measure)
            body = define(f"body_{index}", Sequence(children=children + release))
            batch = define(
                f"batch_{index}",
                MeasurementBatch(
                    body=body,
                    settings=(
                        Setting(
                            label=query.experiment,
                            metadata=MetadataRef(format=METHOD, data=data.target.manifest.reference),
                        ),
                    ),
                    repetitions=shots * query.multiplicity,
                    observation_kind="counts",
                ),
            )
            batches.append(batch)
            experiments.append(
                Experiment(name=query.experiment, batch=batch, setting_index=0,
                           readout=ReadoutDetails(qubits=()))
            )
        experiments = tuple(experiments)
        root = (
            batches[0]
            if options.estimator == "rwpe"
            else define("schedule", Sequence(children=tuple(batches)))
        )
        classical = (ClassicalValue(name="readout", dtype="bits", width=1),)
    program = Program(
        root=root,
        definitions=tuple(definitions),
        expressions=tuple(expressions),
        parameters=parameters,
        registers=(Register(name="ancilla", width=1, role="clean_ancilla"),)
        + ((Register(name="system", width=width),) if width else ()),
        classical=classical,
        signatures=tuple(block.record.signature for block in native),
    )
    return (
        SelectedConstruction(
            program=program,
            selections=tuple(block.record for block in native),
        ),
        experiments,
        native,
    )


def _bind_host(plan, data, declaration):
    """Return the bound host kernel that evaluates the ideal Hadamard means classically.

    For each query (power p, ancilla phase s) the kernel returns
    Re(exp(i*s)*z_p) with z_p = sum_j w_j*exp(-i*E_j*tau*p), or
    sum_j w_j*lambda_j**p for unitary input, from one dense
    eigendecomposition and the prepared spectral weights w_j
    (numerical.overlap). This is the noiseless value of the ancilla <Z> in
    method._quantum_program. Binding performs no eigensolve. The
    decomposition runs at the first invocation and is cached in the Run
    context.
    """

    def bind(realization, selected, run):
        """Return the invocation for one static batch or one RWPE query with its bound feedback."""
        context = run._method_context(_new_context)
        queries = (
            (query_at(plan, realization.experiment, realization.bindings),)
            if plan.method.estimator == "rwpe"
            else tuple((query.power, query.phase_shift) for query in plan.reconstruction.queries)
        )

        def invoke():
            """Evaluate Re(exp(i phase) <psi|U^power|psi>) and record spectral setup/reuse work."""
            (values, _, weights), setup, projected = _spectrum(data, context)
            # Reuse each distinct scalar power within this query batch, including
            # the real/imaginary quadratures that share the same overlap.
            powers = tuple(dict.fromkeys(power for power, _ in queries))
            overlaps = {
                power: numerical.overlap(values, weights, power, data.tau) for power in powers
            }
            scalars = tuple(
                ScalarValue(
                    label=label,
                    value=float(np.real(np.exp(1j * phase) * overlaps[power])),
                    frame="unit",
                )
                for label, (power, phase) in zip(selected.scalars, queries, strict=True)
            )
            dimension = data.target.manifest.basis.dimension
            # spectral_cache_payload_bound, in bytes: 16*D**2 for the complex128
            # eigenvector matrix and 40*D for the D-vectors kept with it
            # (eigenvalues up to 16*D, reference 16*D, weights 8*D).
            application = KernelApplication(
                name="nominal_overlaps",
                implementation=HOST,
                arguments=tuple(
                    Binding(parameter=key, value=value)
                    for key, value in (
                        ("eigensystems", int(setup)),
                        ("projections", int(projected)),
                        ("scalar_powers", len(powers)),
                        (
                            "spectral_cache_payload_bound",
                            16 * dimension * dimension + 40 * dimension,
                        ),
                    )
                ),
            )
            return KernelOutput(
                plan_id=plan.content_id,
                selected_kernel_id=selected.content_id,
                scalars=scalars,
                physical_scale=data.reference.preparation.physical_scale,
                applications=(application,),
            )

        return invoke

    return BoundKernel._bind_pointwise(plan, declaration, bind)


def _samples(plan, observations, trace, *, receipts=()):
    """Return one QPESample (ancilla mean with its power and phase) per acquired query value.

    Every observation chunk is first joined to one completed event, its
    selected Experiment and readout, and, for counts, the receipt of its
    selected construction and an independent seed population. A trajectory
    point chunk holds the one-point readout of its declared point, and its
    attempt's completed event names the ``ObservationView`` identity of the
    ordered point collection; when every declared point is present that
    identity is checked, and a missing point leaves its queries without
    samples, so analysis reports them missing instead of replaying any
    prefix. Each Experiment and binding is resolved once per call, however
    many point chunks it produced. _host_samples then reduces a classical
    host-kernel chunk, _trajectory_samples a trajectory point and
    _native_sample a one-bit count readout. Analysis, the RWPE controller and
    saved-data validation all call this one reduction, so a loaded Result is
    checked against the same samples its producer used.
    """
    from nwqlib._counts import CountsSources
    from nwqlib.execution import ObservationView

    receipt_map = {receipt.content_id: receipt for receipt in receipts}
    if len(receipt_map) != len(receipts):
        raise ValueError("QPE analysis requires distinct actual preparation receipts")
    sources = CountsSources(receipt_map.get, trace._acquisition, max_bytes=plan.method.max_bytes)
    completed, attempts = {}, {}
    for event in trace.events:
        if event.status == "completed":
            key = (event.attempt, event.prepared_id, event.observation_id, event.execution)
            if key in completed or event.attempt in attempts:
                raise ValueError("QPE acquisition requires a unique completed event")
            completed[key] = attempts[event.attempt] = event
    resolved, points, collections = {}, {}, {}
    samples = []
    for chunk in observations.chunks:
        trace.validate_observation(chunk)
        identity = (chunk.experiment, chunk.bindings)
        if identity not in resolved:
            realization = plan.resolve(chunk.experiment, bindings=chunk.bindings)
            resolved[identity] = (realization, *realization.resolved_observation(plan))
        realization, setting, observation = resolved[identity]
        if chunk.point is not None:
            if observation.kind != "trajectory":
                raise ValueError("QPE point chunk requires its trajectory Experiment")
            if id(observation) not in points:
                points[id(observation)] = observation.point_observations()
            declared = points[id(observation)].get(chunk.point)
            event = attempts.get(chunk.attempt)
            joined = (declared is not None and chunk.observation == declared
                      and event is not None and event.prepared_id == chunk.prepared_id
                      and event.execution == chunk.execution)
            if joined:
                collections.setdefault(chunk.attempt, (observation, []))[1].append(chunk)
        else:
            event = completed.get((chunk.attempt, chunk.prepared_id, chunk.content_id, chunk.execution))
            joined = chunk.observation == observation and event is not None
        if (
            not joined
            or chunk.plan_id != plan.content_id
            or chunk.run_id != trace.run_id
            or chunk.realization_id != realization.content_id
            or chunk.bindings != realization.bindings
            or chunk.setting != setting
            or chunk.population != "unconditional"
        ):
            raise ValueError("QPE acquisition differs from its exact selected point or completion")
        if observation.kind == "counts":
            sources.require_independent(chunk)
            _, construction = realization._selected_construction(plan)
            if receipt_map[chunk.prepared_id].construction_id != construction.content_id:
                raise ValueError("QPE counts receipt differs from its actual selected construction")
        if plan.execution == "classical":
            samples.extend(_host_samples(plan, chunk, event, realization))
        elif chunk.point is not None:
            samples.extend(_trajectory_samples(plan, chunk, event, receipt_map))
        else:
            samples.append(_native_sample(plan, chunk, event, observation))
    for attempt, (observation, chunks) in collections.items():
        if len({chunk.point for chunk in chunks}) != len(chunks):
            raise ValueError("QPE trajectory acquisition repeats a point")
        if len(chunks) == len(observation.positions):
            ordered = sorted(chunks, key=lambda chunk: observation.point_index(chunk.point))
            if attempts[attempt].observation_id != ObservationView(chunks=tuple(ordered)).content_id:
                raise ValueError("QPE trajectory points differ from their completed acquisition")
    return tuple(samples)


def _host_samples(plan, chunk, event, realization):
    """Return the samples of one classical host-kernel chunk.

    The chunk must come from the Plan's single host kernel with its declared
    scalar frames, application arguments and work. A static batch holds one
    scalar per query, and an RWPE chunk holds its one query. Each scalar is
    the ideal ancilla mean Re(exp(i*s)*z_p) of _bind_host. bounded_mean
    stores a value that lies within NUMERICAL_RELATION_RTOL outside [-1, 1]
    in raw_mean and uses the nearer endpoint as mean. A missing scalar
    (value None) produces no sample.
    """
    kernel = realization.selected_construction(plan).kernels[0]
    if (
        chunk.execution != "host_kernel"
        or chunk.selected_kernel_id != kernel.content_id
        or chunk.source != kernel.implementation
        or chunk.artifacts
        or chunk.unavailable
        or tuple(v.frame for v in chunk.values) != kernel.scalar_frames
        or len(chunk.applications) != 1
        or chunk.applications[0].name != kernel.name
        or chunk.applications[0].implementation != kernel.implementation
        or chunk.physical_scale != plan.reconstruction.preparation.physical_scale
        or event.host_work != kernel.invocation_work
    ):
        raise ValueError(
            "QPE classical observation differs from its selected scalar kernel"
        )
    queries = (
        (query_at(plan, chunk.experiment, chunk.bindings),)
        if plan.method.estimator == "rwpe"
        else tuple((q.power, q.phase_shift) for q in plan.reconstruction.queries)
    )
    names = (
        (chunk.experiment,)
        if plan.method.estimator == "rwpe"
        else tuple(q.experiment for q in plan.reconstruction.queries)
    )
    # The application arguments are the setup counts and the cache size
    # written by _bind_host. They must match the work this chunk declares.
    arguments = {item.parameter: item.value for item in chunk.applications[0].arguments}
    dimension = plan.reconstruction.preparation.basis.dimension
    if (
        arguments.keys()
        != {"eigensystems", "projections", "scalar_powers", "spectral_cache_payload_bound"}
        or type(arguments["eigensystems"]) is not int
        or arguments["eigensystems"] not in (0, 1)
        or type(arguments["projections"]) is not int
        or arguments["projections"] not in (0, 1)
        or arguments["scalar_powers"] != len({power for power, _ in queries})
        or arguments["spectral_cache_payload_bound"] != 16 * dimension**2 + 40 * dimension
    ):
        raise ValueError(
            "QPE application evidence differs from its actual bounded host workload"
        )
    samples = []
    for name, (power, feedback), value in zip(names, queries, chunk.values, strict=True):
        if value.value is None:
            continue
        mean, raw_mean = bounded_mean(value.value)
        fields = dict(
            experiment=name,
            power=power,
            phase_shift=feedback,
            mean=mean,
            raw_mean=raw_mean,
            contributions=(chunk.content_id,),
        )
        samples.append(QPESample(**fields))
    return samples


def _expected_layout(plan):
    """Return the native layout of a QPE readout: ancilla on qubit 0, system on qubits 1..n."""
    width = plan.reconstruction.preparation.basis.dimension.bit_length() - 1
    return (("ancilla", (0,)),) + ((("system", tuple(range(1, width + 1))),) if width else ())


def _trajectory_samples(plan, chunk, event, receipt_map):
    """Return the samples of one trajectory point: its ancilla X and Y expectations.

    With ancilla qubit zero the point saves ``"I"*n+"X"`` and ``"I"*n+"Y"``,
    whose expectations at power p are Re z_p and Im z_p. Each query of the
    point keeps its logical quadrature (0 for X, -pi/2 for Y) and names the
    point chunk as its one contribution, so the two quadratures of one power
    share one acquisition. Each saved expectation is admitted by
    ``bounded_mean`` within the receipt's ``probability_window``: one
    whole-trajectory G gives a conservative window for every point, and the
    window is not multiplied by the number of saved labels or positions.
    """
    width = plan.reconstruction.preparation.basis.dimension.bit_length() - 1
    if (
        chunk.execution != "quantum_circuit"
        or tuple((register.name, register.bits) for register in chunk.quantum_layout)
        != _expected_layout(plan)
        or event.evaluations != 1
    ):
        raise ValueError("QPE trajectory layout/evaluation differs from its selected acquisition")
    receipt = receipt_map.get(chunk.prepared_id)
    window = NUMERICAL_RELATION_RTOL if receipt is None else receipt.probability_window
    values = {item.label: item.value for item in chunk.values}
    real, imag = trajectory_labels(width)
    if values.keys() != {real, imag}:
        raise ValueError("QPE trajectory point requires its ancilla X and Y expectations")
    samples = []
    for query in plan.reconstruction.queries:
        if query.point != chunk.point:
            continue
        mean, raw_mean = bounded_mean(values[real if query.phase_shift == 0 else imag], window)
        samples.append(QPESample(experiment=query.experiment, power=query.power, phase_shift=None,
                                 mean=mean, raw_mean=raw_mean, contributions=(chunk.content_id,),
                                 point=chunk.point))
    return samples


def _native_sample(plan, chunk, event, observation):
    """Return the sample of one sampled Hadamard test: the ancilla mean <Z> = P(0) - P(1).

    The chunk must hold the ancilla on qubit 0 and the system register on
    qubits 1..n, and its shot counts must match the event. The mean is
    (zeros - ones)/(zeros + ones) from the returned shots, never from the
    requested shot count.
    """
    from nwqlib._quantum_readout import validate_readout_layout

    if (
        chunk.execution != "quantum_circuit"
        or observation.kind != "counts"
        or tuple((register.name, register.bits) for register in chunk.quantum_layout)
        != _expected_layout(plan)
        or event.shots != observation.shots
        or event.returned_shots != chunk.returned_shots
        or event.evaluations != 0
    ):
        raise ValueError(
            "QPE native layout/population exposure differs from its selected acquisition"
        )
    validate_readout_layout(
        chunk,
        observed=(0,),
        classical=plan.construction.program.classical,
    )
    power, feedback = query_at(plan, chunk.experiment, chunk.bindings)
    histogram = chunk.histogram()
    # The layout check above makes the readout one bit wide, so the outcome
    # index is the ancilla bit.
    values = dict(zip(histogram.index_list(), histogram.weights.tolist(), strict=True))
    zero, one = values.get(0, 0), values.get(1, 0)
    if zero + one <= 0:
        raise ValueError("QPE cannot infer a mean from an empty actual count population")
    return QPESample(
        experiment=chunk.experiment,
        power=power,
        phase_shift=feedback,
        mean=(zero - one) / (zero + one),
        zeros=zero,
        ones=one,
        contributions=(chunk.content_id,),
    )


def _aggregate(samples):
    """Return (mean, shots) of the samples acquired for one query.

    Count samples are pooled: mean = (sum zeros - sum ones)/(total returned
    shots), with that total as the second value. Exact samples, which carry
    no shot count, are averaged and return shots None. A query cannot mix the
    two populations.
    """
    sampled = tuple(sample.shots is not None for sample in samples)
    if any(sampled) and not all(sampled):
        raise ValueError("one QPE query cannot mix sampled and exact populations")
    if all(sampled):
        zero, one = sum(sample.zeros for sample in samples), sum(sample.ones for sample in samples)
        return (zero - one) / (zero + one), zero + one
    return fsum(sample.mean for sample in samples) / len(samples), None


def _mean_window(plan, samples, receipts):
    """Return the error window of one exact ancilla mean, or None for count data.

    QCELS and RFE use it to decide which powers carry a signal
    (numerical._signal_vanishes). Count samples take the exact-zero test, so
    any sample with shots gives None. Classical execution evaluates host
    scalars, whose means ``bounded_mean`` admits within
    NUMERICAL_RELATION_RTOL. Exact trajectory expectations are admitted
    within their receipt's ``probability_window`` (_trajectory_samples): one
    whole-trajectory G gives a conservative window for every point, so the
    window is the largest over the Run's receipts, or
    NUMERICAL_RELATION_RTOL without a receipt.
    """
    if any(sample.shots is not None for sample in samples):
        return None
    if plan.execution == "classical":
        return NUMERICAL_RELATION_RTOL
    return max((receipt.probability_window for receipt in receipts), default=NUMERICAL_RELATION_RTOL)


def _static_estimate(plan, samples, options, *, mean_window=None):
    """Return (NumericalEstimate or None, missing query names, exposure facts or None) for QCELS, SPE or RFE.

    Samples are grouped by query and pooled with _aggregate. The means at
    ancilla phases 0 and -pi/2 are the real and imaginary parts of z_p, so
    each power gives one complex point. SPE and RFE keep one point per
    distinct drawn power together with its draw multiplicity, while QCELS
    keeps one point per distinct power. ``mean_window`` is ``_mean_window``
    of these samples. QCELS and RFE use it to tell an exact zero signal from
    roundoff. When any planned query has no sample, no estimator runs and the
    missing names are returned.

    Each count-mode SPE or RFE query contributes its mean over the shots
    actually returned, weighted by its original random-power draw
    multiplicity (``received_fourier_exposure``); the third returned value
    holds the exposure facts of those count queries, None otherwise.
    """
    rec = plan.reconstruction
    powers = tuple(power.power for power in rec.powers)
    options._admit(
        *_analysis_size(
            options, len(samples), power_span=max(powers) - min(powers), complex_points=len(powers)
        )
    )
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.experiment, []).append(sample)
    missing = tuple(q.experiment for q in rec.queries if q.experiment not in grouped)
    if missing:
        return None, missing, None
    if options.estimator in {"spe", "rfe"}:
        drawn_powers, overlaps, multiplicities, received = [], [], [], []
        for real, imag in zip(rec.queries[::2], rec.queries[1::2], strict=True):
            real_mean, real_shots = _aggregate(grouped[real.experiment])
            imag_mean, imag_shots = _aggregate(grouped[imag.experiment])
            received.append((real, imag, real_shots, imag_shots))
            drawn_powers.append(real.power)
            overlaps.append(complex(real_mean, imag_mean))
            multiplicities.append(real.multiplicity)
        counted = {shots is not None for _, _, *pair in received for shots in pair}
        if len(counted) != 1:
            raise ValueError("one QPE Fourier analysis cannot mix sampled and exact populations")
        exposure = received_fourier_exposure(received, plan.shots) if counted == {True} else None
        if options.estimator == "spe":
            return numerical.spe(drawn_powers, overlaps, tau=rec.tau,
                grid_size=options.grid_size, fourier_degree=options.fourier_degree,
                filter_beta=options.filter_beta, overlap_lower_bound=options.overlap_lower_bound,
                multiplicities=multiplicities), (), exposure
        return numerical.rfe(drawn_powers, overlaps, tau=rec.tau, num_frequencies=options.num_frequencies,
                             multiplicities=multiplicities, mean_window=mean_window), (), exposure
    pairs = {}
    for query in rec.queries:
        mean, _ = _aggregate(grouped[query.experiment])
        pairs.setdefault(query.power, {})[query.phase_shift] = mean
    # The 0 and -pi/2 phase settings recover real and imaginary overlap parts.
    # Keep their sign convention when assembling the complex signal.
    powers = sorted(pairs)
    overlaps = tuple(complex(pairs[power][0.0], pairs[power][-pi / 2]) for power in powers)
    return numerical.qcels(powers, overlaps, tau=rec.tau, grid_size=options.grid_size,
                           mean_window=mean_window), (), None


def _exposure_fields(exposure):
    """Return the QPEAnalysis exposure fields of ``received_fourier_exposure``'s facts.

    The exact effective exposure keeps its numerator and denominator, and
    the exact RFE coordinate proxy is published upward in binary64.
    """
    from nwqlib.subroutines.trotterization.error_budget import _finite_upper

    if exposure is None:
        return {}
    effective = exposure["minimum_effective_shots_per_draw"]
    return dict(
        exposure=tuple(QPEExposure(experiment=name, multiplicity=m, requested=requested,
                                   received=received)
                       for name, m, requested, received in exposure["queries"]),
        requested_exposure_complete=exposure["requested_exposure_complete"],
        minimum_effective_shots_per_draw=(effective.numerator, effective.denominator),
        rfe_coordinate_variance_upper=_finite_upper(
            exposure["rfe_coordinate_variance_upper"], "RFE coordinate variance proxy"),
    )


def received_fourier_exposure(pairs, shots):
    """Return the received-population facts of count-mode SPE or RFE queries.

    ``pairs`` contains (real_query, imag_query, real_received, imag_received)
    per drawn power. Let M be the number of paired random-power draws and m
    the multiplicity of a distinct drawn power, with sum m = M. A query
    requests N = s*m shots and returns n with 1 <= n <= N, and its mean is
    (Z - O)/n. Conditioned on the drawn schedule and the received counts,
    and assuming every counted observation is an actual outcome of that
    query with the same conditional mean and was not selected by its
    outcome or an outcome-dependent stopping rule, linearity gives
    E[m*mean | schedule, counts] = m*mu for every positive integer n,
    including n < m; the two quadratures need not agree. Keeping the original
    draw multiplicities therefore keeps the same unconditional finite Fourier
    or CDF target. With conditionally independent +/-1 shots of variance
    v = 1 - mu**2, Var(m*mean | schedule, counts) = m**2*v/n. The RFE
    coordinate proxy is A = sum_u (m_u/M)**2 max(1/n_uR, 1/n_uI), an upper
    bound on each real and imaginary coordinate proxy of the sampled Fourier
    coefficients, kept exact. The uniform effective exposure is the rational
    min_q n_q/m_q, not a fractional physical shot count. These are
    statements about the finite CDF or Fourier arrays, not about the
    nonlinear crossing or peak, and exclude evolution bias and
    floating-point evaluation error. Source: NWQLib's received-population
    analysis of the pooled draws, with
    the uniform random draws of Kshirsagar, Katabarwa and Johnson,
    arXiv:2209.11322v3, Eqs. (5)-(7), and the sampled filter of Wan, Berta
    and Campbell, arXiv:2110.12071v2.

    Refuses an empty draw schedule, quadratures that do not represent the
    same draws and a received count outside 1..N (no shots, or an over-return
    beyond the admitted request).
    """
    from fractions import Fraction

    M = sum(real.multiplicity for real, _, _, _ in pairs)
    if M < 1:
        raise ValueError("Fourier analysis needs a nonempty draw schedule")
    proxy = Fraction(0)
    effective = None
    exposures = []
    for real, imag, nr, ni in pairs:
        m = real.multiplicity
        if imag.multiplicity != m or real.power != imag.power:
            raise ValueError("Fourier quadratures must represent the same draws")
        for query, n in ((real, nr), (imag, ni)):
            requested = shots * m
            if type(n) is not int or not 1 <= n <= requested:
                raise ValueError(
                    f"QPE query {query.experiment} returned {n!r} shots "
                    f"for an admitted request of {requested}"
                )
            exposures.append((query.experiment, m, requested, n))
            per_draw = Fraction(n, m)
            effective = per_draw if effective is None else min(effective, per_draw)
        proxy += Fraction(m*m, M*M) * max(Fraction(1, nr), Fraction(1, ni))
    return {
        "draws": M,
        "queries": tuple(exposures),
        "requested_exposure_complete": all(req == got for _, _, req, got in exposures),
        "minimum_effective_shots_per_draw": effective,
        "rfe_coordinate_variance_upper": proxy,
    }


def _schedule_size(config, tau):
    """Return (acquisition count, maximum power) and admit the schedule size.

    ``count`` is the number of logical quadrature settings before repeated
    SPE and RFE draws are grouped, two per power for the static estimators and
    one per RWPE step.
    ``maximum`` is the largest power the schedule can contain, which
    max_power caps: the largest QCELS power, 2*fourier_degree+1 for SPE,
    K-1 for RFE and the last relative time 1/sigma for RWPE. Planning
    calls this before the method RNG draws the schedule, so an oversized
    schedule or a power above max_power is rejected before any draw or
    construction. The admitted size is 8 work units and 64 bytes per unit of
    ``frontier``, which counts the schedule entries plus 2*fourier_degree+1
    for SPE (the filter's maximum frequency) or K for RFE (its number of
    frequencies).
    """
    if config.estimator == "qcels":
        positive, maximum = numerical.qcels_schedule_size(config, tau)
        # A single positive linspace point is power 1, not the far endpoint.
        maximum = 1 if positive == 1 else maximum
        count = 2 * (1 + positive)
        frontier = count
    elif config.estimator == "spe":
        maximum, count = 2*config.fourier_degree+1, 2*config.num_samples
        frontier = maximum + count
    elif config.estimator == "rfe":
        maximum = config.num_frequencies - 1
        count = 2 * config.num_samples
        frontier = count + config.num_frequencies
    else:
        maximum = 1/numerical.rwpe_scale(config.prior_std, max(0, config.max_steps-1))
        count = frontier = max(1, config.max_steps)
    if maximum > config.max_power:
        raise ValueError(f"selected QPE power {maximum} exceeds max_power={config.max_power}")
    config._admit(8 * frontier, 64 * frontier)
    return count, maximum


def _host_declaration(data, labels, *, work, payload, pointwise, query_count):
    """Declare the classical kernel that evaluates the nominal signal.

    The kernel computes the estimator signal from one dense eigendecomposition
    and the prepared spectral weights. It is a nominal model, not quantum
    execution or an independent reference. ``work`` and ``payload`` are the
    size laws the caller admitted. LAPACK internals and SDK workspace stay unknown.
    """
    law = ResourceLaw(
        metric="classical_work",
        basis="selected_logical",
        value=work,
        interpretation="upper_bound",
        evidence=Evidence(kind="numerical_estimate", source=HOST),
        assumptions=("finite dense/scalar size units; LAPACK work and SDK workspace unknown",),
    )
    return SelectedKernel(
        name="nominal_overlaps",
        implementation=HOST,
        inputs=(data.target.reference, data.reference.reference),
        scalars=labels,
        scalar_frames=("unit",) * len(labels),
        resource_laws=(law,),
        workspace=(Workspace(location="host", purpose="workspace", bytes=payload, source=HOST),),
        construction_work=len(labels) + 1 + (query_count if pointwise else 0),
        invocation_work=work,
    )


def _analysis_configuration(method, settings):
    """Return the Method with reanalysis settings applied.

    Only QCELS and SPE accept ``grid_size``, which changes the analysis grid
    over the same observations. RFE's K fixes both its power distribution and
    its frequency grid, and every RWPE setting fixes its controller, so both
    accept no setting. A setting that would need new acquisitions raises
    ValueError.
    """
    allowed = set() if method.estimator in {"rfe", "rwpe"} else {"grid_size"}
    if set(settings) - allowed:
        raise ValueError("these settings change acquisition/controller choices; create a new Plan")
    return method.revise(**settings) if settings else method


def _analysis(
    plan,
    observations,
    samples,
    *,
    receipts=(),
    settings=None,
    gaussian=None,
    processed=0,
    controller_work=0,
    stop_reason=None,
):
    """Return the QPEAnalysis Result for existing samples, without acquisition.

    RWPE reports the saved Gaussian moments through numerical.rwpe_estimate
    and is complete once all planned steps are processed. The static
    estimators run _static_estimate and are complete when no query is
    missing and the kernel identified an estimate. records.public_estimate
    derives the phase in turns and the output-frame interval from the kernel
    value, and identification_note prefixes any unverified branch or filter
    premise to the interval text. ``settings`` holds reanalysis choices
    accepted by _analysis_configuration. ``receipts`` are the Run's
    preparation receipts, whose windows ``_mean_window`` reads.
    """
    settings = settings or {}
    config = _analysis_configuration(plan.method, settings)
    exposure = None
    if config.estimator == "rwpe":
        estimate = None
        missing = (
            ("RWPE Gaussian moments require their saved controller state",) if gaussian is None else ()
        )
        if gaussian is not None:
            config._admit(16, 256)
            estimate = numerical.rwpe_estimate(
                gaussian.mean, gaussian.standard_deviation, tau=plan.reconstruction.tau
            )
        complete = gaussian is not None and processed == config.max_steps
    else:
        estimate, missing, exposure = _static_estimate(
            plan, samples, config, mean_window=_mean_window(plan, samples, receipts))
        complete = not missing and estimate is not None and estimate.value is not None
        if estimate is not None and estimate.stop_reason is not None:
            stop_reason = estimate.stop_reason
    value, phase, interval = public_estimate(
        estimate, phase_output=isinstance(plan.output, Eigenphase), tau=plan.reconstruction.tau
    )
    note = identification_note(plan)
    if interval is not None and note is not None:
        interval = interval.revise(interpretation=note + " " + interval.interpretation)
    # RFE and SPE average over randomly drawn powers or frequencies
    # (numerical.planned_power_schedule). Their papers bound this Monte Carlo
    # error through the sample count, M in Kshirsagar, Katabarwa and Johnson,
    # arXiv:2209.11322v3, Theorem 2.1, and C_sample in Wan, Berta and
    # Campbell, arXiv:2110.12071v2, Eq. (11). Every RWPE update consumes one
    # random bit, a shot in quantum execution and a bit emulated from the
    # exact signal in classical execution (controller.run_rwpe). Exact
    # readout leaves these draws in the estimate, so only QCELS and an RWPE
    # prior without updates can have zero sampling error.
    random_draws = (config.estimator in {"rfe", "spe"}
                    or config.estimator == "rwpe" and processed > 0)
    result = QPEAnalysis(
        origin=capture_analysis_origin(
            analyzer=_QCELS_ANALYSIS_SOURCE if config.estimator == "qcels" else _ANALYSIS_SOURCE,
            method_id=plan.method.content_id,
            dependencies=("numpy", "scipy"),
        ),
        plan_id=plan.content_id,
        construction_id=plan.construction.content_id,
        observation_id=observations.content_id,
        contribution_ids=tuple(c.content_id for c in observations.chunks),
        estimator=config.estimator,
        estimator_value=value,
        phase_turns=phase,
        interval=interval,
        fit=None if estimate is None else estimate.fit,
        samples=samples,
        missing=missing,
        complete=complete,
        gaussian=gaussian,
        processed_steps=processed,
        controller_work=controller_work,
        stop_reason=stop_reason,
        analysis_settings=tuple(
            Binding(parameter=name, value=value if type(value) is int else Float64(value=value))
            for name, value in sorted((name, getattr(config, name)) for name in settings)
        ),
        facts=exact_readout_sampling(plan, observations, random_draws=random_draws),
        **_exposure_fields(exposure),
    )
    result.validate_plan(plan)
    return result


def _split_pauli_terms(raw, rtol):
    """Return (identity, pruned_mass, labels, coefficients) from (label, coefficient) Pauli pairs.

    - identity: the real coefficient c_I of the all-identity label. It is not
      a Pauli rotation. _quantum_program applies it on the control through
      phase increments, because controlled(exp(-i*t*c_I)*V) equals
      P(-t*c_I) on the control followed by controlled V.
    - pruned_mass: an upward bound d_up on d = sum |c_j| over nonidentity
      terms with |c_j| <= rtol*max_k |c_k|, the maximum including the
      identity (``error_budget.upper_dropped_mass``). powers.select_powers
      charges abs(t)*d_up at the exact target time of each power. For H',
      ||exp(-i*t*H) - exp(-i*t*H')|| <= abs(t)*||H - H'|| <= abs(t)*d, the second step
      by the triangle inequality because every Pauli string has norm one.
      The emitted-parameter rechecks require the exact mass or an outward upper
      bound on it.
    - labels, coefficients: the kept nonidentity terms in input order, which
      fixes the product-formula term order.

    Empty input returns (0.0, 0.0, (), ()). A nonzero imaginary coefficient
    raises ValueError instead of being dropped, since H must be Hermitian.
    The cutoff pauli_pruning_rtol is registered in
    docs/ENGINEERING_CONSTANTS.md.
    """
    from nwqlib.subroutines.trotterization.error_budget import upper_dropped_mass

    identity, mass, labels, coefficients = 0.0, 0.0, (), ()
    if raw:
        if any(complex(c).imag != 0 for _, c in raw):
            raise ValueError("imaginary Pauli coefficients are not discarded")
        cutoff = rtol * max(abs(c) for _, c in raw)
        identity = fsum(float(complex(c).real) for label, c in raw if not set(label) - {"I"})
        dropped = [c for label, c in raw if set(label) - {"I"} and abs(c) <= cutoff]
        mass = upper_dropped_mass(complex(c).real for c in dropped)
        kept = tuple((label, float(complex(c).real)) for label, c in raw
                     if set(label) - {"I"} and abs(c) > cutoff)
        labels = tuple(label for label, _ in kept)
        coefficients = tuple(coefficient for _, coefficient in kept)
    return identity, mass, labels, coefficients


def _host_construction(method, data, d, queries, powers):
    """Return (construction, experiments, declaration, work, payload) for classical execution.

    The Program has one ClassicalStage that calls the host kernel of
    _bind_host. Static estimators evaluate every query in one ``overlaps``
    experiment with scalars ``z_0``, ``z_1``, and so on. RWPE has one
    experiment per query with the single scalar ``z`` and a symbolic
    ``feedback`` parameter. ``work`` and ``payload`` are the admitted size
    law of the kernel, stored in QPEReconstruction.
    """
    # Size law of the host kernel for dimension D:
    # - 8*D**3 work: the dense eigendecomposition (numerical.nominal_eigensystem).
    # - 16*D**2 work: projecting the reference on the eigenvectors.
    # - 16*D per distinct power in one invocation: the D-term spectral sum of
    #   numerical.overlap. A static batch evaluates all powers, RWPE one.
    # - 64 work units and 64 bytes per query for the scalar outputs.
    # - 160*D**2 + 128*D bytes: an allowance of ten D x D and eight
    #   length-D complex128 arrays for the eigensystem, reference and weights.
    per_batch = 1 if method.estimator == "rwpe" else len(powers)
    work = 8 * d**3 + 16 * d * d + 16 * d * per_batch + 64 * len(queries)
    payload = 160 * d * d + 128 * d + 64 * len(queries)
    method._admit(work, payload)
    labels = (
        ("z",) if method.estimator == "rwpe" else tuple(f"z_{i}" for i in range(len(queries)))
    )
    declaration = _host_declaration(
        data,
        labels,
        work=work,
        payload=payload,
        pointwise=method.estimator == "rwpe",
        query_count=len(queries),
    )
    parameters = (
        (Parameter(name="feedback", domain="real"),) if method.estimator == "rwpe" else ()
    )
    program = Program(
        root="overlaps",
        parameters=parameters,
        definitions=(
            Definition(
                id="overlaps",
                node=ClassicalStage(
                    implementation=HOST, boundary="host", kernel=declaration.name
                ),
            ),
        ),
    )
    construction = SelectedConstruction(
        program=program, selections=(), kernels=(declaration,)
    )
    observation = ObservationSpec(kind="host_scalars", labels=labels)
    experiments = (
        tuple(
            Experiment(name=q.experiment, setting=q.experiment, observation=observation)
            for q in queries
        )
        if method.estimator == "rwpe"
        else (Experiment(name="overlaps", setting="overlaps", observation=observation),)
    )
    return construction, experiments, declaration, work, payload


def _unknown_error_model(problem, output, construction):
    """Return the ErrorModel that records every QPE error source as unknown.

    The estimator kernels give at most a local model interval. None of them
    composes the ERROR_SOURCES contributions into a bound in the requested
    output frame, so each one is listed with availability "unknown" instead
    of being omitted.
    """
    frame = output.frame(problem)
    return ErrorModel(
        output_id=output.content_id,
        subject_id=problem.content_id,
        construction_id=construction.content_id,
        frame=frame,
        required_sources=ERROR_SOURCES,
        source=METHOD,
        terms=tuple(
            ErrorTerm(
                name=name,
                stage="analysis",
                source=METHOD,
                formula="no composed bound in the requested output frame",
                coverage=name,
                fact=FramedFact(
                    frame=frame,
                    bindings=(),
                    fact=Fact(
                        quantity=name,
                        unit=frame.unit,
                        scope=frame.scope,
                        availability="unknown",
                        reason="local interval does not establish " + name,
                    ),
                ),
            )
            for name in ERROR_SOURCES
        ),
    )


class _QPEMethod(Method):
    """Settings that `QCELS`, `SPE`, `RFE` and `RWPE` all accept, with the same defaults.

    Each estimator observes the signal `z_p = <psi|U^p|psi>` through one
    ancilla qubit, with `U = exp(-i*tau*H)` for a Hamiltonian H, or the
    polar factor of a supplied unitary. An `Eigenproblem` takes its prepared
    state from `initial_state`, and a `SpectralEstimation` supplies its own.
    A peak, fit or credible interval does not establish the requested ground
    state or accuracy on a mixed spectrum.

    Attributes:
        initial_state: Default `None`. Reference state `|psi>`, required for
            an `Eigenproblem`. A `SpectralEstimation` supplies its own state,
            and this must then be `None`.
        tau: Default `None`, which chooses `0.9*phase_radius/R` from a
            spectral-radius bound R, the dense row-sum bound or the L1 norm
            of the Pauli coefficients with the identity included. The phase
            radius is pi, and smaller for SPE, as the
            [QPE guide](../../algorithms/qpe.md) explains. Positive time step
            of `U = exp(-i*tau*H)`, in inverse operator units. Unitary input
            takes `None`.
        controlled_power_backend: Default `"auto"`, which picks
            `"trotter_error_budgeted"` for Pauli input and `"dense_exact"`
            otherwise. Construction of the controlled powers `U^p`:
            `"dense_exact"`, the exact power of explicit dense input, or
            `"trotter_error_budgeted"`, a second-order product formula, which
            needs a Hamiltonian.
        controlled_power_error_budget: Default `1e-3`, positive. Operator-norm
            allowance for each power, covering Pauli pruning plus
            product-formula error. It does not bound the estimator error.
        pauli_pruning_rtol: Default `1e-12`, nonnegative. Non-identity Pauli
            terms with `abs(c) <= pauli_pruning_rtol*max abs(c)` are dropped,
            and their L1 mass is counted against each power's error
            allowance.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Limit on
            the known bytes of input and analysis arrays.
        max_work: Default `1_000_000_000`. Limit on the counted work of the
            likelihood, the grid and the evolution choice.
        max_power: Default `100_000`. Largest absolute evolution time in
            units of tau, including the non-integer times of RWPE.
        max_trotter_steps: Default `1_000_000`. Largest number of
            product-formula steps per controlled power.
    """

    result_type: ClassVar[type] = QPEAnalysis
    initial_state: StateData | None = None
    tau: Annotated[Real, Field(gt=0)] | None = None
    estimator: ClassVar[str]
    controlled_power_backend: Literal["auto", "dense_exact", "trotter_error_budgeted"] = "auto"
    controlled_power_error_budget: Annotated[Real, Field(gt=0)] = 1e-3
    pauli_pruning_rtol: Nonnegative = 1e-12

    # Selected scalar/array limits, independent of Run exposure and SDK internals.
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_work: PositiveInt = 1_000_000_000
    max_power: PositiveInt = 100_000
    max_trotter_steps: PositiveInt = 1_000_000

    def _admit(self, work, payload_bytes):
        """Raise ValueError if work > max_work or bytes > max_bytes, naming the Method class, the exceeded limits and both sizes.

        Callers pass the size law of the next step before running it, so a
        refused step never starts.
        """
        if work > self.max_work or payload_bytes > self.max_bytes:
            name = type(self).__name__
            exceeded = [(field, need) for field, need, cap in (
                ("max_work", work, self.max_work), ("max_bytes", payload_bytes, self.max_bytes))
                if need > cap]
            raise ValueError(
                "selected QPE numerical/schedule work exceeds "
                + " and ".join(f"{name}.{field}" for field, _ in exceeded)
                + f": requires work={work}, bytes={payload_bytes}; "
                f"max_work={self.max_work}, max_bytes={self.max_bytes}. Raise "
                + " and ".join(f"{name}.{field} to at least {need}" for field, need in exceeded) + ".")

    def _inputs(self, problem, output):
        """Admit Hamiltonian/unitary inputs and pad only explicit finite dense coordinates.

        A dense input of dimension d that is not a power of two is embedded in
        the next power of two. The added block is c*I with c = Tr(H)/d for a
        Hamiltonian, or I for a unitary, and the preparation is zero-padded.
        The padded operator is block diagonal and the prepared state has no
        weight in the added block, so every ideal signal <psi|U^p|psi> is
        unchanged. Because c is the mean eigenvalue and |c| <= max_i |H_ii|,
        the padding adds no eigenvalue outside the original spectral interval
        and does not raise the row-sum bound used to choose tau.
        """
        if isinstance(problem, Eigenproblem):
            if not isinstance(output, (Eigenvalue, Eigenphase)):
                raise ApplicabilityError("QPE requires an eigenvalue or phase output")
            if self.initial_state is None:
                raise ApplicabilityError(
                    "QPE requires an explicit initial_state on its Method for Eigenproblem"
                )
            operator, state = problem.A, self.initial_state
        elif isinstance(problem, SpectralEstimation):
            if self.initial_state is not None:
                raise ValueError(
                    "SpectralEstimation already owns initial_state; omit the Method initialization"
                )
            if not isinstance(output, (Eigenvalue, Eigenphase)):
                raise ApplicabilityError("QPE requires an eigenvalue or phase output")
            operator, state = problem.operator, problem.initial_state
        else:
            raise ApplicabilityError("QPE requires Eigenproblem or SpectralEstimation")
        unitary = isinstance(problem, SpectralEstimation) and problem.unitary is not None
        if unitary and (
            isinstance(output, Eigenvalue) or self.tau is not None
        ):
            raise ApplicabilityError(
                "unitary-only QPE has no Hamiltonian eigenvalue or tau"
            )
        d = problem.dimension
        if state.basis.dimension != d or state.preparation.physical_scale.mantissa == 0:
            raise ValueError("QPE initial_state must be nonzero in the original coordinates")
        if d & (d - 1):
            if operator.reference.representation != "dense":
                raise ApplicabilityError("non-power-of-two QPE requires explicit small dense input")
            padded = 1 << (d - 1).bit_length()
            # P = padded dimension. Work: 16 units per padded entry for
            # filling, copying and checking. Bytes: an allowance of four
            # P x P complex128 arrays, the padded matrix and the copies that
            # ingest_dense makes for its snapshot and Hermiticity comparison.
            self._admit(16 * padded * padded, 64 * padded * padded)
            matrix = np.zeros((padded, padded), dtype=complex)
            matrix[:d, :d] = operator.dense_array()
            dummy = 1.0 if unitary else fsum(float(matrix[i, i].real) / d for i in range(d))
            matrix[d:, d:] = dummy * np.eye(padded - d)
            operator = ingest_dense(matrix, max_bytes=self.max_bytes)
            vector = np.zeros(padded, dtype=complex)
            vector[:d] = state.physical_vector()
            state = ingest_vector(vector, max_bytes=self.max_bytes)
        if state.preparation.blocker and not _scalar_reference_available(state):
            raise ApplicabilityError(state.preparation.blocker)
        return operator, state, "unitary" if unitary else "hamiltonian"

    def plan(self, problem, *, output, execution, shots, rng):
        """Return the Plan: time step, schedule, controlled powers and circuit description.

        Steps in order, each checked against the limits before it runs:

        1. Check the problem, the output and the requested shots. RWPE needs
           a Hamiltonian and exactly one shot per quantum update.
        2. Choose the controlled-power construction from the input access. A
           Pauli target never triggers a dense spectral construction.
        3. Choose tau: the supplied value, or `0.9*phase_radius/R` with R the
           dense row-sum bound or the Pauli L1 norm. A unitary has no tau.
        4. Obtain Pauli coefficients for product-formula powers and separate
           the identity term and the pruned terms.
        5. Draw the schedule. Repeated SPE and RFE draws of one
           (power, phase) become one query with a multiplicity. Exact
           readout evaluates it once, and counts measure `shots` times its
           multiplicity in fresh shots.
        6. Choose the powers once: one polar factor for a unitary, one common
           step for exact-trajectory product-formula powers, one
           independently chosen step per sampled or RWPE product-formula
           power, and one dense block per trajectory gap or sampled power.
        7. Build the circuit description: one trajectory for exact static
           estimators, one Hadamard test per sampled or RWPE query, or the
           classical kernel, with the record that analysis reads and the
           error model.
        """
        if self.estimator == "rwpe" and isinstance(problem, SpectralEstimation) and problem.unitary is not None:
            raise ApplicabilityError("RWPE requires continuous Hamiltonian evolution, not only integer powers of a unitary")
        target, state, kind = self._inputs(problem, output)
        if execution == "classical" and shots is not None:
            raise ValueError("classical QPE has no requested circuit shots")
        if self.estimator == "rwpe":
            if kind != "hamiltonian":
                raise ApplicabilityError("RWPE requires continuous Hamiltonian evolution, not only integer powers of a unitary")
            if execution == "quantum":
                if shots not in (None, 1):
                    raise ValueError("RWPE uses exactly one shot per Gaussian update")
                shots = 1
        d = target.basis.dimension
        width = d.bit_length() - 1
        representation = target.reference.representation
        if (
            representation not in ("dense", "pauli")
            or kind == "unitary"
            and representation != "dense"
        ):
            raise ApplicabilityError("QPE requires explicit dense or Pauli Hamiltonian access")
        if execution == "classical" and representation != "dense":
            raise ApplicabilityError(
                "classical QPE needs explicit dense spectral input; no implicit Pauli expansion"
            )
        if execution == "classical":
            # The query-independent part of the host-kernel law that
            # _host_construction admits in full once the schedule is known.
            self._admit(8 * d**3 + 16 * d * d, 160 * d * d + 128 * d)
            if not _scalar_reference_available(state) and state.preparation.implementation not in (
                "qiskit.direct",
                "qiskit.product",
                "qiskit.occupation",
            ):
                raise ApplicabilityError(
                    "classical QPE has no supplied-circuit reference expansion"
                )
        # Choose a realization from available input access. A Pauli target does
        # not silently trigger dense spectral construction.
        backend = self.controlled_power_backend
        if backend == "auto":
            backend = "trotter_error_budgeted" if representation == "pauli" else "dense_exact"
        if kind == "unitary" and backend != "dense_exact":
            raise ApplicabilityError("product powers need a Hamiltonian")
        if backend == "dense_exact" and representation != "dense":
            raise ApplicabilityError("dense_exact requires explicit dense input")
        if representation == "dense":
            from nwqlib.operators.access import refuse_known_need

            # The dense phases below are checked one at a time against the
            # same limits; their known needs combine by maximum here, before
            # the range scan: the range scan (8D**2 work) or unitary
            # validation (4D**3 work), both 96D**2 bytes, and a requested
            # dense Pauli conversion (32(q+1)D**2 work, (192(q+1)+q)D**2
            # bytes). The classical spectral core is admitted above, and
            # explicit padding in _inputs.
            phases = [(4 * d**3 if kind == "unitary" else 8 * d * d, 96 * d * d)]
            if execution == "quantum" and kind == "hamiltonian" and backend == "trotter_error_budgeted":
                phases.append((32 * (width + 1) * d * d, (192 * (width + 1) + width) * d * d))
            refuse_known_need(type(self).__name__, "max_work", self.max_work, max(work for work, _ in phases),
                              "selected power construction and coefficient-dependent step selection")
            refuse_known_need(type(self).__name__, "max_bytes", self.max_bytes, max(size for _, size in phases),
                              "selected census structure and power-construction workspace")
        # Select a positive Hamiltonian time and record its spectral-radius basis.
        # Unitary-only input already defines phase and has no Hamiltonian time scale.
        tau, radius, tau_selection, raw = None, None, "unitary input", ()
        phase_radius = (numerical.spe_phase_radius(self.fourier_degree, self.filter_beta,
                                                 self.overlap_lower_bound)
                        if self.estimator == "spe" else pi)
        if representation == "dense":
            # Work: 4*D**3 for the product U^dagger U of the unitarity check,
            # or 8*D**2 for the elementwise row sums of _safe_time. Bytes: an
            # allowance of six D x D complex128 temporaries (96*D**2).
            self._admit(4 * d**3 if kind == "unitary" else 8 * d * d, 96 * d * d)
            matrix = target.dense_array()
            if kind == "unitary":
                from nwqlib.subroutines._matrix_checks import is_unitary

                # Entrywise atol 1e-8 on U^dagger U - I, registered in
                # docs/ENGINEERING_CONSTANTS.md (_matrix_checks.is_unitary).
                if not is_unitary(matrix):
                    raise ValueError("QPE unitary input violates numerical unitarity")
            else:
                tau, radius, tau_selection = _safe_time(
                    matrix, self.tau, phase_radius=phase_radius)
                if execution == "quantum" and backend == "trotter_error_budgeted":
                    # pauli_coefficients runs width block levels of O(D**2)
                    # arithmetic and then one O(D**2) pass that checks the
                    # coefficients and assembles their labels. The law charges
                    # width+1 such passes at 32*D**2 work units and 192*D**2
                    # bytes of numerical traffic each, plus width*D**2 bytes
                    # of label text for at most D**2 Pauli words.
                    self._admit(32 * (width + 1) * d * d, (192 * (width + 1) + width) * d * d)
                    raw = pauli_coefficients(matrix)
        else:
            raw = target.pauli_terms().labels()
            if any(c.imag != 0 for _, c in raw):
                raise ValueError("Hamiltonian Pauli coefficients must be real")
            # sum_j |c_j|, identity included, bounds ||H||, so the automatic
            # time uses the same estimator-specific phase radius as _safe_time.
            radius = fsum(abs(c.real) for _, c in raw)
            tau = (
                self.tau
                if self.tau is not None
                else (1.0 if radius == 0 else 0.9 * phase_radius / radius)
            )
            tau_selection = "supplied" if self.tau is not None else "Pauli coefficient l1 bound"
        if tau is not None and (not isfinite(tau) or tau <= 0):
            raise ValueError("Hamiltonian tau must remain positive and finite")
        identity, mass, labels, coefficients = _split_pauli_terms(
            raw, self.pauli_pruning_rtol)
        # Admit estimator work before drawing the schedule, then select each
        # distinct power once for reuse across its interference quadratures.
        count, maximum = _schedule_size(self, tau)
        self._admit(
            *_analysis_size(
                self,
                count,
                power_span=maximum,
                complex_points=count // 2,
            )
        )
        schedule = numerical.planned_power_schedule(self, tau, rng.method)
        if self.estimator in {"spe", "rfe"}:
            # One query per distinct drawn (power, quadrature), weighted by its
            # draw multiplicity. A counts batch acquires shots_per_draw times
            # multiplicity fresh shots (_quantum_program), and analysis uses
            # the shots each query actually returned (_static_estimate,
            # received_fourier_exposure). Counter keeps first-occurrence
            # order, so the real and imaginary queries of one power stay
            # adjacent (_static_estimate pairs queries[::2] with
            # queries[1::2]). An exact query is evaluated once.
            from collections import Counter
            entries = Counter(schedule).items()
        else:
            entries = ((entry, 1) for entry in schedule)
        # The exact static quantum route observes every query on one
        # controlled trajectory; sampled static queries and RWPE keep one
        # Hadamard test each, and the classical static route one host kernel.
        if execution == "classical":
            route = "classical"
        elif self.estimator == "rwpe":
            route = "rwpe"
        else:
            route = "trajectory" if shots is None else "sampled"
        queries = tuple(
            QPEQuery(
                experiment=f"query_{i}", power=power, phase_shift=phase, multiplicity=count,
                acquisition=(f"query_{i}" if route in ("sampled", "rwpe") or self.estimator == "rwpe"
                             else "trajectory" if route == "trajectory" else "overlaps"),
                point=f"power_{power}" if route == "trajectory" else None,
            )
            for i, ((power, phase), count) in enumerate(entries)
        )
        if route == "trajectory" and any(type(q.power) is not int or q.power < 0 for q in queries):
            raise ValueError("an exact QPE trajectory observes only nonnegative integer powers")
        base = None
        if kind == "unitary":
            # One polar base per Plan: every power, gap, spectral consumer
            # and verification uses V = polar(A) (powers.polar_base).
            base = ingest_dense(polar_base(matrix, self), max_bytes=self.max_bytes)
        data = _Input(target, state, tau, base)
        selection = select_powers(
            target=target,
            base=base,
            tau=tau,
            labels=labels,
            coefficients=coefficients,
            identity=identity,
            pruned_mass=mass,
            powers=tuple(sorted({q.power for q in queries})),
            backend="classical" if execution == "classical" else backend,
            config=self,
            route=route,
        )
        # Declare native experiments or a bounded host spectral model explicitly.
        # Neither construction performs the eventual acquisition at this point.
        work = payload = 0
        declaration = None
        if execution == "quantum":
            construction, experiments, blocks = _quantum_program(
                data, queries, selection, self, shots, route
            )
        else:
            construction, experiments, declaration, work, payload = _host_construction(
                self, data, d, queries, selection.powers
            )
            blocks = ()
        evaluation = selection.evaluation
        census = selection.census_block is not None
        reconstruction = QPEReconstruction(
            target=target.reference,
            original_target=(
                problem.A if isinstance(problem, Eigenproblem) else problem.operator
            ).reference,
            original_preparation=(
                self.initial_state if isinstance(problem, Eigenproblem) else problem.initial_state
            ).reference,
            method_id=self.content_id,
            preparation=state.preparation,
            target_kind=kind,
            tau=tau,
            tau_selection=tau_selection,
            spectral_radius_bound=radius,
            aliasing_bound_sufficient=kind == "unitary"
            or radius is not None
            and tau * radius < pi,
            queries=queries,
            powers=selection.powers,
            pauli_labels=labels,
            pauli_coefficients=FrozenArray(np.array(coefficients, dtype=float)) if labels else None,
            identity_coefficient=identity,
            pruned_mass=mass,
            pair_commutation_checks=evaluation.pair_commutation_checks if census else 0,
            nested_commutation_checks=evaluation.nested_commutation_checks if census else 0,
            bound_variant=evaluation.bound_variant if census else None,
            coefficient_arithmetic=evaluation.coefficient_arithmetic if census else None,
            bound_coefficient=(evaluation.coefficient.numerator, evaluation.coefficient.denominator)
            if census else None,
            census_block=selection.census_block,
            common_step=selection.common,
            selected_base=None if base is None else base.reference,
            numerical_work=work,
            numerical_payload=payload,
        )
        model = _unknown_error_model(problem, output, construction)
        selected = Plan(
            problem=problem,
            method=self,
            output=output,
            execution=execution,
            shots=shots,
            randomness=rng.snapshot(),
            construction=construction,
            experiments=experiments,
            reconstruction=reconstruction,
            error_model=model,
            assumptions=(
                "estimate concerns the explicitly prepared spectral population; no ground identity guarantee",
            ),
        )
        if declaration is not None:
            blocks = (_bind_host(selected, data, declaration),)
        validate_selection(selected)
        return selected._bind(blocks=blocks, input=data)

    def validate_point(self, plan, experiment, values):
        """Require a bound ``feedback`` value for every RWPE query, raising KeyError otherwise."""
        if self.estimator == "rwpe" and "feedback" not in values:
            raise KeyError("feedback")

    def before_submit(self, plan, prepared, *, run):
        """Reject known unusable likelihood populations before new submission."""
        if any(handle.record.observation.kind == "counts" for handle in prepared):
            run.admit_count_preparations(prepared)

    def prepare_all_refusal(self):
        """RWPE refuses ``settings="all"``, because each feedback angle depends on the outcomes before it."""
        if self.estimator == "rwpe":
            return (
                "RWPE cannot prepare every setting in advance, because the feedback angle of each "
                "query depends on the outcomes of the queries before it. prepare(plan) with the default "
                "settings='first' prepares the next query, and estimate(plan) folds the Plan's "
                "resource laws without building circuits"
            )
        return super().prepare_all_refusal()

    def prepare(self, plan, *, run, settings="first"):
        """Prepare the first static setting, every one for ``settings="all"``, or only RWPE's next feedback query.

        RWPE's query comes from ``controller.prepare_rwpe``. For RWPE,
        ``settings`` is always ``"first"``, because ``prepare_all_refusal``
        refuses ``"all"`` before a Run exists, so ``prepare_rwpe`` does not
        take it.

        Quantum RWPE plans ``max_steps`` circuit preparations, acquisitions
        and one-shot observations. It checks all three totals before preparing
        the first feedback query. A caller cancellation can end the run
        earlier. The feedback angle remains dependent on preceding outcomes,
        so only the next circuit is prepared at a time. ``controller.run_rwpe``
        processes one datum at each of ``max_steps`` scheduled steps, and
        quantum preparation enforces one shot per step, so without
        cancellation or execution failure::

            preparations = acquisitions = shots = max_steps

        The Gaussian width update introduces no early convergence stop, so
        ``max_steps`` is the exact completed-work total of the normal route
        and a sufficient envelope when the caller cancels. The check is
        ``Run.check_capacity``, not a reservation, so the per-step charges are
        unchanged and nothing is counted twice. It runs on a fresh Run only: a
        restored Run already holds exposure and may have a prepared or
        completed pending step, and its existing per-step checks protect
        resumed and exceptional work. The classical route uses host
        invocations and a local Bernoulli draw and is not charged as quantum
        shots.
        """
        if self.estimator == "rwpe":
            from .controller import prepare_rwpe

            if plan.execution == "quantum" and run.checkpoint_state is None:
                run.check_capacity(preparations=self.max_steps, circuits=self.max_steps, shots=self.max_steps)
            return prepare_rwpe(plan, run=run)
        return super().prepare(plan, run=run, settings=settings)

    def execute(self, plan, *, run):
        """Run the static schedule through the shared Run, or RWPE's feedback loop (controller.run_rwpe)."""
        if self.estimator == "rwpe":
            from .controller import run_rwpe

            return run_rwpe(plan, run=run)
        return super().execute(plan, run=run)

    def analyze(self, plan, data, *, settings):
        """Reduce saved observations to ancilla means and evaluate the estimator.

        `result.analyze(...)` calls it, and it measures nothing new. RWPE
        reads the Gaussian mean and the number of processed steps from the
        saved run state, and its width follows Granade and Wiebe,
        arXiv:2208.04526v1, Eq. (7b), without replaying observations.
        Without saved run state, the RWPE estimate is reported as missing.
        Its stop reason is the one the run recorded when it ended.
        """
        validate_selection(plan)
        samples = _samples(plan, data.observations, data.trace, receipts=data.receipts)
        state = None
        if self.estimator == "rwpe" and data.controller is not None:
            import json
            from .controller import Checkpoint

            saved = (
                json.loads(data.controller) if isinstance(data.controller, str) else data.controller
            )
            state = Checkpoint.model_validate(saved.get("state", saved))
        return _analysis(
            plan,
            data.observations,
            samples,
            receipts=data.receipts,
            settings=settings,
            gaussian=None if state is None else RWPEGaussian(mean=state.mean,
                standard_deviation=numerical.rwpe_scale(self.prior_std, state.processed)),
            processed=0 if state is None else state.processed,
            controller_work=0 if state is None else state.work,
            stop_reason=data.trace.termination_reason if self.estimator == "rwpe" else None,
        )

    def save_archive(self, plan, files):
        from .archive import save

        return save(self, plan, files)

    @classmethod
    def load_archive(cls, data, files):
        from .archive import load

        return load(cls, data, files)

    @staticmethod
    def snapshot_result_context(context):
        """Keep already-produced spectral arrays by immutable reference."""
        from types import MappingProxyType

        if context is None:
            return None
        if set(context) != {"target", "eigensystem", "preparation", "reference", "weights",
                            "squares"}:
            raise ValueError("QPE result context contains unknown mutable or native data")
        # The binary squares are a recomputable construction cache, not a result.
        snapshot = {name: value for name, value in context.items() if name != "squares"}
        for name in ("eigensystem", "reference"):
            if snapshot[name] is not None:
                snapshot[name] = tuple(snapshot[name])
        arrays = () if snapshot["eigensystem"] is None else snapshot["eigensystem"]
        if snapshot["reference"] is not None and snapshot["reference"][0] is not None:
            arrays += (snapshot["reference"][0],)
        if snapshot["weights"] is not None:
            arrays += (snapshot["weights"],)
        if any(not isinstance(array, np.ndarray) or array.flags.writeable for array in arrays):
            raise ValueError("QPE result context requires existing read-only numerical arrays")
        return MappingProxyType(snapshot)

    def save_run_context(self, context, files):
        from .archive import save_context

        return save_context(context, files)

    @staticmethod
    def load_run_context(data, files):
        from .archive import load_context

        return load_context(data, files)

    def verify(self, plan, result, *, checks):
        from .verification import verify

        return verify(plan, result, checks=checks)


def _estimator_integer(value, info):
    """Validate an integer count field: at least 0 for ``max_steps``, at least 1 otherwise."""
    from nwqlib._validation import integer

    return integer(value, info.field_name, 0 if info.field_name == "max_steps" else 1)


class QCELS(_QPEMethod):
    """QCELS method for an eigenvalue or eigenphase, by a single-mode complex least-squares fit.

    Build it with keyword arguments and pass it as `method=`, for example
    `solve(Eigenproblem(A=H), method=QCELS(initial_state=psi))`. An
    `Eigenproblem` needs `initial_state`, and a `SpectralEstimation`
    supplies its own state. The other arguments are optional, and the
    [shared settings][nwqlib.algorithms.qpe.method._QPEMethod] apply too. The
    result is a [`QPEAnalysis`][nwqlib.algorithms.qpe.records.QPEAnalysis]
    whose `value` comes from the single mode `a*exp(-i*p*theta)`, with
    `theta = E*tau`, that best fits the signals `z_p = <psi|U^p|psi>` of a
    schedule of integer powers p.

    The fit minimizes the QCELS objective of Ding and Lin,
    arXiv:2211.11973v2, Eq. (2). Their accuracy theorems, Theorems 1 and 2
    (p. 14), assume `p0 > 0.71` for the squared overlap
    `p0 = |<psi|psi_0>|**2` of the prepared state psi with the target
    eigenvector psi_0, and NWQLib does not check p0. Smaller overlaps need
    the Fourier-filtered variant of their Sec. IV (p. 15). Neither that
    variant nor the multi-level schedule of their Algorithm 1 (p. 13) is
    implemented. The finite grid search and bracket refinement have no
    global-optimum or uncertainty guarantee, and `interval` is None.
    `result.analyze(grid_size=...)` fits the same signals on another grid.
    The [QPE guide](../../algorithms/qpe.md) describes the search and its
    aliasing limits.

    Attributes:
        num_times: Default `16`. Upper bound on the number of positive
            sampling times. The zero-power quadratures are added separately.
            For maximum power P (set by `max_time`), `num_times >= P` gives
            the consecutive powers 0 through P, the form of the data set of
            Ding and Lin, arXiv:2211.11973v2, Eq. (6). Otherwise power 0 is
            kept and `num_times` integer powers are spread over 1 through P,
            a schedule their Theorems 1 and 2 do not describe and whose fit
            can land on a replica of the energy
            ([QPE guide](../../algorithms/qpe.md#qcels-schedule)).
        max_time: Default `None`, which selects maximum integer power 10,
            independent of units. Otherwise the positive maximum physical
            sampling time, which selects maximum power `floor(max_time/tau)`,
            or `floor(max_time)` for a unitary.
        grid_size: Default `4096`, at least 8. Requested minimum number of
            phase-grid points. The effective grid is
            `max(grid_size, 8*(max(p) - min(p)) + 1)`, so it also resolves
            the span of powers.

    Examples:
        `H = diag(-0.5, 0.5)` has eigenvalue -0.5 for the eigenvector `|0>`:

        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import QCELS
        >>> problem = Eigenproblem(A=[[-0.5, 0], [0, 0.5]])
        >>> result = solve(problem, method=QCELS(initial_state=[1, 0]), seed=7)
        >>> print(round(result.eigenvalue, 10))
        -0.5
    """

    estimator: ClassVar[str] = "qcels"
    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR.revise(
        method="qcels", references=(REFERENCES[0],)
    )
    num_times: PositiveInt = 16
    max_time: Annotated[Real, Field(gt=0)] | None = None
    grid_size: Annotated[PositiveInt, Field(ge=8)] = 4096
    _integers = field_validator("num_times", "grid_size", mode="before")(_estimator_integer)


class SPE(_QPEMethod):
    """SPE method for the lowest eigenvalue or eigenphase in the spectrum of the prepared state.

    Build it with keyword arguments and pass it as `method=`, for example
    `solve(Eigenproblem(A=H), method=SPE(initial_state=psi, overlap_lower_bound=0.5))`.
    `overlap_lower_bound` is required, and an `Eigenproblem` also needs
    `initial_state`. The other arguments are optional, and the
    [shared settings][nwqlib.algorithms.qpe.method._QPEMethod] apply too. The
    result is a [`QPEAnalysis`][nwqlib.algorithms.qpe.records.QPEAnalysis]
    whose `value` is the lowest value in the prepared spectral support. For
    a Hamiltonian that is the lowest energy. For a unitary it is the lowest
    principal eigenphase angle theta, with `U v = exp(i*theta) v` and theta
    scanned on `[-pi/2, pi/2)`. When `tau*|E| < pi/2` for every prepared
    energy E of `U = exp(-i*tau*H)`, `theta = -tau*E`, so on such a unitary
    SPE returns the phase of the highest energy of the prepared support.
    Supplying H, or `U^dagger = exp(i*tau*H)`, targets the lowest energy
    instead.

    Statistical phase estimation locates the first crossing of `eta/2` by a
    Fourier-filtered approximate cumulative distribution function (CDF) of
    the prepared state's spectrum, scanned on a finite grid. The filter and
    the `eta/2` threshold come from Wan, Berta and Campbell,
    arXiv:2110.12071v2, PDF Eqs. (A1)-(A2) (HTML Eqs. (16)-(17)) and
    Algorithm 1. Positive odd frequencies are drawn by their PDF Eq. (A2)
    Fourier weights, numbered Eq. (17) in the HTML version. Both
    quadratures of the controlled evolution are measured for each draw, and
    conjugation supplies the negative-frequency contribution. Repeated
    draws of one frequency share one setting per quadrature, which with
    counts receives `shots` times the number of draws in fresh shots. The
    finite filter and sample defaults do not enforce the paper's precision
    theorem. A scan over `grid_size` thresholds replaces
    its O(log(1/delta)) binary-search decisions, and the sample count is not
    derived from its Eq. (11), so its Theorem 1 does not apply. The chosen
    controlled-evolution construction sets its own approximation,
    independently of the paper's randomized compiler. `interval` is None,
    and `result.analyze(grid_size=...)` scans the same signals on another
    grid. The [QPE guide](../../algorithms/qpe.md) lists each departure from
    the paper.

    Attributes:
        overlap_lower_bound: Required, in (0, 1]. Caller-declared lower bound
            eta on the probability of the target component in the prepared
            state. NWQLib does not verify it.
        num_samples: Default `32`. Independent positive-frequency draws, each
            with two quadratures.
        fourier_degree: Default `11`. Filter parameter d, with maximum
            frequency `2*d + 1`.
        filter_beta: Default `6.0`, positive. Error-function smoothing
            parameter beta.
        grid_size: Default `4096`, at least 8. Number of threshold points of
            the approximate CDF in `[-pi/2, pi/2)`.

    Examples:
        `H = diag(-0.5, 0.5)` has eigenvalue -0.5 for the eigenvector `|0>`.
        SPE returns the first threshold crossing on its finite grid, from 32
        random frequency draws:

        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import SPE
        >>> problem = Eigenproblem(A=[[-0.5, 0], [0, 0.5]])
        >>> method = SPE(initial_state=[1, 0], overlap_lower_bound=1.0)
        >>> result = solve(problem, method=method, seed=7)
        >>> print(round(result.eigenvalue, 6))
        -0.499674
    """

    schema_version: Literal[2] = 2
    estimator: ClassVar[str] = "spe"
    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR.revise(
        method="spe", version="2", references=(REFERENCES[2],)
    )
    overlap_lower_bound: Annotated[Real, Field(gt=0, le=1)]
    num_samples: PositiveInt = 32
    fourier_degree: PositiveInt = 11
    filter_beta: Annotated[Real, Field(gt=0)] = 6.
    grid_size: Annotated[PositiveInt, Field(ge=8)] = 4096
    _integers = field_validator("num_samples", "fourier_degree", "grid_size", mode="before")(
        _estimator_integer
    )


class RFE(_QPEMethod):
    """RFE method for an eigenvalue or eigenphase, from the largest sampled Fourier coefficient.

    Build it with keyword arguments and pass it as `method=`, for example
    `solve(Eigenproblem(A=H), method=RFE(initial_state=psi))`. An
    `Eigenproblem` needs `initial_state`. The other arguments are optional,
    and the [shared settings][nwqlib.algorithms.qpe.method._QPEMethod] apply
    too. The result is a
    [`QPEAnalysis`][nwqlib.algorithms.qpe.records.QPEAnalysis] whose phase
    is `2*pi*j/K` for the frequency j of the largest of the K sampled Fourier
    coefficients, on the principal branch, and whose energy is `-phase/tau`.

    In randomized Fourier estimation, each of M independent draws picks a
    uniform integer power in `[0, K)` and measures the real and imaginary
    Hadamard tests of Kshirsagar, Katabarwa and Johnson, arXiv:2209.11322v3
    (Eqs. (1)-(6), p. 4).
    Algorithm 1 and Eq. (7), p. 5, define the sampled Fourier coefficients.
    The default exact readout reads the same quadratures as the ancilla X
    and Y expectations at each power of one controlled trajectory, and
    sampled readout keeps the paired tests. Repeated draws of one power
    share one setting per quadrature, which with counts receives `shots`
    times the number of draws in fresh shots. The precision theorem,
    Theorem 2.1 (p. 6), assumes an eigenstate and enough draws, and the
    default M is far below the count it requires. `interval` is None.
    Changing K needs a new Plan. The [QPE guide](../../algorithms/qpe.md)
    gives the sign convention and the sample bound.

    Attributes:
        num_samples: Default `97`. Number M of independent power draws, each
            with two quadratures.
        num_frequencies: Default `49`, at least 2. Fourier size K, one greater
            than the largest sampled power.

    Examples:
        `H = diag(-0.5, 0.5)` has eigenvalue -0.5 for the eigenvector `|0>`.
        RFE returns a phase on the grid `2*pi*j/49`:

        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import RFE
        >>> problem = Eigenproblem(A=[[-0.5, 0], [0, 0.5]])
        >>> result = solve(problem, method=RFE(initial_state=[1, 0]), seed=7)
        >>> print(round(result.eigenvalue, 6))
        -0.498866
    """

    estimator: ClassVar[str] = "rfe"
    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR.revise(
        method="rfe", version="2", references=(REFERENCES[3],)
    )
    num_samples: PositiveInt = 97
    num_frequencies: Annotated[PositiveInt, Field(ge=2)] = 49
    _integers = field_validator("num_samples", "num_frequencies", mode="before")(_estimator_integer)


class RWPE(_QPEMethod):
    """RWPE method for an eigenvalue, by a Gaussian random walk with one measured bit per step.

    Build it with keyword arguments and pass it as `method=`, for example
    `solve(Eigenproblem(A=H), method=RWPE(initial_state=psi))`. An
    `Eigenproblem` needs `initial_state`. The other arguments are optional,
    and the [shared settings][nwqlib.algorithms.qpe.method._QPEMethod] apply
    too. The result is a
    [`QPEAnalysis`][nwqlib.algorithms.qpe.records.QPEAnalysis] whose energy
    is `-mean/tau` for the posterior mean of the phase `phi = -tau*E`, with
    the nominal 95-percent interval of the Gaussian model in `interval` and
    the posterior moments in `gaussian`. The input must be a Hamiltonian,
    because the method needs continuous evolution times, which the integer
    powers of a bare unitary do not supply. It uses exactly one shot per
    step. Because each feedback angle depends on the outcomes before it,
    `prepare(plan, settings="all")` is refused, and `prepare(plan)` prepares
    the next query.

    The random-walk phase estimation update is the basic walk of Granade and
    Wiebe, arXiv:2208.04526v1, Eq. (7) and Algorithm 1: one bit at time
    `1/sigma`, a mean shift of `+/-sigma/sqrt(e)` and a width shrink by
    `sqrt((e-1)/e)`. The likelihood is Eq. (2) of Granade and Wiebe,
    arXiv:2208.04526v1 (p. 2). The feedback sign follows that likelihood
    and the positive outcome-zero update in Eq. (7a), correcting the
    opposite sign printed in Algorithm 1. The
    consistency checks and unwinding of their Algorithm 2 are not
    implemented. The Gaussian interval is a model output, not a coverage
    guarantee. The [QPE guide](../../algorithms/qpe.md) describes the finite
    range that the walk can explore.

    Attributes:
        max_steps: Default `14`, nonnegative. Largest number of feedback
            steps, which a run uses exactly unless it is cancelled or fails.
            Zero returns the prior without measurement.
        prior_mean: Default `0.0`. Prior mean of the unwrapped phase
            `-tau*E`, in radians.
        prior_std: Default `pi`, positive. Prior standard deviation of the
            phase, in radians.

    Examples:
        `H = diag(-0.5, 0.5)` has eigenvalue -0.5 for the eigenvector `|0>`.
        After 14 one-bit updates the estimate and its nominal 95-percent
        interval are:

        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import RWPE
        >>> problem = Eigenproblem(A=[[-0.5, 0], [0, 0.5]])
        >>> result = solve(problem, method=RWPE(initial_state=[1, 0]), seed=7)
        >>> low, high = result.interval.low, result.interval.high
        >>> print(round(result.eigenvalue, 4), round(low, 4), round(high, 4))
        -0.4784 -0.5223 -0.4345
    """

    schema_version: Literal[2] = 2
    estimator: ClassVar[str] = "rwpe"
    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR.revise(
        method="rwpe", version="2", references=(REFERENCES[1],)
    )
    max_steps: Count = 14
    prior_mean: Real = 0.
    prior_std: Annotated[Real, Field(gt=0)] = pi
    _integers = field_validator("max_steps", mode="before")(_estimator_integer)
