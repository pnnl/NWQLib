"""ADAPT-GCiM's actual finite pool, selected query arguments and controller."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import pi
from types import SimpleNamespace
from typing import Annotated, Any, ClassVar, Literal
from pydantic import Field, field_validator, field_serializer, model_validator
from nwqlib.algorithms.protocol import Method, AlgorithmDescriptor, ApplicabilityError
from nwqlib.algorithms._eigen_inputs import (
    eigen_operator,
    eigen_state,
    preparation_requirements,
    DEFAULT_CONVERSION_WORK,
)
from nwqlib.algorithms._eigen_support import eigen_error_model
from nwqlib.blocks import SelectedBlock, SelectedConstruction
from nwqlib.blocks.records import BlockSemantics, SelectedDefinition, SelectedKernel
from nwqlib.core.planning import Plan, Experiment, ObservationSpec, ReadoutDetails
from nwqlib.core.records import InputRef, Source, PositiveInt, Real
from nwqlib.ir import (
    Allocate,
    Argument,
    BlockCall,
    BlockSignature,
    ClassicalStage,
    ClassicalValue,
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
    QuantumPort,
    Register,
    Release,
    Sequence,
    Setting,
)
from nwqlib.ir.expressions import number
from nwqlib.problems.records import StateData
from nwqlib.operators.inputs import OperatorInput, operator_input
from nwqlib.operators.access import _check_bytes, _check_products
from nwqlib.resources import Workspace
from nwqlib.subroutines.fermionic_circuits import (
    _GeneratorCircuitPlan,
    _plan_generator_circuit,
    _require_default_double,
    _UnsupportedDefaultGenerator,
)
from nwqlib.subroutines.fermionic_pool import (
    FermionicGenerator,
    _check_generator_work,
    _snapshot_generator,
)
from .adapt_actions import action_sizes
from .adapt_inputs import prepare_adapt_inputs
from .adapt_records import (
    AdaptReconstruction,
    ADAPTResult,
    active_screen_labels,
    chain,
    energy_labels,
    label_cache,
    screen_labels,
)
from .pencil import _projected_solve_bytes, _projected_solve_work

METHOD = Source(
    name="adapt_gcim",
    version="4",
    domain="finite adaptive generator-coordinate eigenvalue",
    reference="Zheng et al., npj Quantum Information 10, 127 (2024), arXiv:2312.07691v3",
)
DESCRIPTOR = AlgorithmDescriptor(
    method=METHOD.name,
    version=METHOD.version,
    problem_families=("eigenproblem",),
    output_families=("eigenvalue",),
    access_families=("pauli", "dense", "csr", "csc"),
    references=(METHOD,),
    limitations=("projected estimate does not establish ground identity or adaptive coverage",),
)
NATIVE = Source(
    name="adapt.query",
    version="2",
    domain="actual ordered generator, Hadamard and joint-state queries",
    reference="nwqlib.algorithms.gcim.adapt_acquisition.construct_adapt",
)
HOST = Source(
    name="adapt.classical_query",
    version="2",
    domain="explicit classical selected state query",
    reference="nwqlib.algorithms.gcim.adapt_acquisition.bind_classical",
)


def _parameters(length, pool_size, term_count, group_count, *, residual=False, qubits=1):
    """Declare the finite parameter vector shared by every ADAPT query.

    One Program serves all adaptive queries, so each query is a point in this
    space rather than a new construction. ``l_*`` and ``r_*`` hold the left and
    right generator chains with at most ``length`` slots. Inactive slots are
    canonical (index -1, angle 0), which lets equal queries share one identity.
    ``term`` and ``quadrature`` select a Hadamard-test Pauli term and real or
    imaginary part. ``group`` selects a sampled QWC readout group. The
    ``insert_*`` fields place one Pauli-insertion shift for measured
    derivatives. Residual queries also carry the Ritz energy and coefficients.
    """
    parameters = []
    for side in ("l", "r"):
        parameters.append(Parameter(name=f"{side}_count", domain="integer", lower=0, upper=length))
        for i in range(length):
            parameters.extend(
                (
                    Parameter(
                        name=f"{side}_pool_{i}", domain="integer", lower=-1, upper=pool_size - 1
                    ),
                    Parameter(name=f"{side}_theta_{i}", domain="real"),
                )
            )
    parameters.extend(
        (
            Parameter(name="term", domain="integer", lower=-1, upper=max(-1, term_count - 1)),
            Parameter(name="control_width", domain="integer", lower=0, upper=1),
            Parameter(name="readout_width", domain="integer", lower=1, upper=qubits),
            Parameter(name="quadrature", domain="integer", lower=0, upper=1),
            Parameter(name="group", domain="integer", lower=0, upper=max(0, group_count - 1)),
            Parameter(name="insert_position", domain="integer", lower=-1, upper=length - 1),
            Parameter(name="insert_term", domain="integer", lower=-1),
            Parameter(name="insert_sign", domain="integer", lower=-1, upper=1),
        )
    )
    if residual:
        parameters.append(Parameter(name="ritz_energy", domain="real"))
        parameters.extend(
            Parameter(name=f"coefficient_{i}_{part}", domain="real")
            for i in range(2 * length)
            for part in ("real", "imag")
        )
    return tuple(parameters)


def generator_pool_slot_bound(num_qubits, term_counts):
    """Bound generator construction from the system width and stored Pauli term counts.

    The commuting route uses at most 6q slots per term. Pair/split,
    four-distinct and shared-index default routes fit 24q+6192 slots, using
    at most six active modes and occupation blocks of dimension at most five
    for the last route. The bound counts logical appended instructions, not
    native synthesis gates. It does not establish support for an arbitrary
    custom noncommuting generator.

    For q system qubits and T stored Pauli terms, the circuit routes in
    ``fermionic_circuits.build_generator_circuit`` have bounds ``6q max(1,T)``
    (commuting: at most ``4w`` basis changes, ``2(w-1)`` parity CX gates and
    one central rotation per term of support ``w <= q``), ``5(2q+11)``
    (pair/split: five factors, each with two parity/control sides and one
    controlled rotation), ``6(2q+516)`` for singlet and ``12(2q+516)`` for
    triplet four-distinct doubles (the worst multiplexor has eight controls
    and at most 512 appended RY/CX instructions), and
    ``4 sum_i(active_i-i) + (4m-3) sum_blocks k_b(k_b-1)/2`` for the shared
    index, with ``m <= min(q,6)``, each ``k_b <= 5`` and disjoint blocks among
    ``2**m`` active occupations. The sorted active indices satisfy
    ``sum(active_i-i) <= m(q-m)``, ``k(k-1)/2 <= 2k`` for ``k <= 5`` and
    ``sum k <= 2**m``, so ``B_shared(q,m) <= 4m(q-m)+(4m-3)2**(m+1)``. For
    ``m <= 6`` the first term is at most ``24q-4m**2`` and the remaining
    constant is at most 2544 at ``m = 6``, so ``B_shared <= 24q+2544``,
    below ``24q+6192``, the triplet four-distinct envelope. Pair/split is
    also below that value for ``q >= 1``. A route-independent bound for any
    supported generator is thus ``B(q,T) = max{6q max(1,T), 24q+6192}``. It
    uses the O(P) existing tuple-length metadata of a P-member pool, with no
    state expansion, commutation census or occupation-block planning.
    """
    counts = tuple(term_counts)
    if not counts:
        return 0
    largest = max(counts)
    return max(6 * num_qubits * max(1, largest),
               24 * num_qubits + 6192)


def _admit_generator_routes(pool, *, max_bytes, max_products):
    """Refuse at planning a generator that no compiler route supports, and return whether all commute.

    This is the eligibility part of ``fermionic_circuits._plan_generator_circuit``
    with its charge: two Pauli words with bit masks ``(x_i, z_i)`` and
    ``(x_j, z_j)`` commute exactly when ``popcount(x_i & z_j) +
    popcount(z_i & x_j)`` is even, and a generator whose terms do not all
    commute must be a normalized default GSD double
    (``_require_default_double``). Occupation blocks and compiler plans are
    built only when a generator is first selected (``CompilerPlans``).
    """
    from sys import int_info
    from nwqlib.operators._pauli import _pauli_masks

    commuting = True
    for generator in pool:
        rows = generator.pauli_terms
        m, q = len(rows), generator.num_qubits
        digits = max(1, (q + int_info.bits_per_digit - 1) // int_info.bits_per_digit)
        pairs = m * (m - 1) // 2
        _check_generator_work(max_bytes, max_products, payload_bytes=m * (32 + 8 * digits),
                              work=m * (q * digits + 1) + pairs * (digits + 1))
        masks = tuple(_pauli_masks(label)[:2] for label, _ in rows)
        if all(((x & masks[j][1]).bit_count() + (z & masks[j][0]).bit_count()) % 2 == 0
               for i, (x, z) in enumerate(masks) for j in range(i)):
            continue
        commuting = False
        try:
            _require_default_double(generator, max_bytes=max_bytes, max_products=max_products)
        except _UnsupportedDefaultGenerator as error:
            raise ValueError(
                f"pool generator {generator.pool_index} has no exact compiler route: {error}"
            ) from error
    return commuting


class CompilerPlans:
    """Generator compiler plans built when a generator is first selected, keyed by pool index.

    ``construct_adapt``, ``_native_preparation``, the optimization shift-rule
    selection and ``adapt_archive`` read this one map. An entry is built by
    ``fermionic_circuits._plan_generator_circuit`` for the admitted
    generator at its pool index and reused for every later angle, and each
    entry compiles that same generator object. ``built()`` returns only the
    plans constructed so far, which are the ones a Plan archive and a Run
    checkpoint save (``adapt_archive``). ``saved`` and ``restore`` take
    ``(pool index, kind, active modes, occupation blocks)``
    entries saved from the same Plan, restoring its compiler blocks by pool index.
    """

    def __init__(self, pool, *, max_bytes, max_products, saved=()):
        self._pool = tuple(pool)
        self._limits = dict(max_bytes=max_bytes, max_products=max_products)
        self._built = {}
        self.restore(saved)

    def restore(self, saved):
        """Restore saved compiler plans at their pool indices, keeping entries already built."""
        for index, kind, modes, blocks in saved:
            if index not in self._built:
                self._built[index] = _GeneratorCircuitPlan(kind, self._pool[index], tuple(modes), tuple(blocks))

    def __len__(self):
        return len(self._pool)

    def __getitem__(self, index):
        if index not in self._built:
            plan = _plan_generator_circuit(self._pool[index], **self._limits)
            # Saved maps and Result snapshots share the blocks, so they are read-only.
            for _, block in plan.occupation_blocks:
                block.flags.writeable = False
            self._built[index] = plan
        return self._built[index]

    def built(self):
        """Return ``((pool index, plan), ...)`` of the plans constructed so far, by pool index."""
        return tuple(sorted(tuple(self._built.items())))


def _construction(inputs, reference, method, length, compiler_plans, slot_bound, execution, shots):
    """Select the one parameterized construction that every ADAPT query reuses.

    The adaptive trajectory is unknown at planning time, so the Plan fixes a
    finite query space instead of concrete circuits. Each later query is a
    point in the ``_parameters`` space, admitted by ``ADAPT.validate_point``
    and bound through ``Plan.resolve``. The same construction therefore serves
    live execution, resume and archive reload without replanning.

    Classical execution selects one host kernel with experiments ``screen``,
    ``pair``, ``energy`` and, for residual stopping, ``residual``. An
    ``energy`` query returns the product energy and its gradient in the
    chain's angles from one evaluation. Its known workspace is the bounded
    frontier of cached state vectors and Hamiltonian actions, and with
    optimization the energy-and-gradient reservation. A supplied Qiskit
    reference adds a materialization whose cost is declared unknown.

    Quantum execution selects one native block per query family. ``screen`` and
    ``energy`` prepare a generator chain on the system register and read Pauli
    labels. ``pair`` adds an ancilla. With shots it is the Hadamard test for one
    transition amplitude. Exact execution prepares the pair's phase-faithful
    joint state, or a diagonal pair's system state, reduced at acquisition to
    ``(H0_ij, S_ij)`` (``pair_reducer``), and the exact ``screen`` query of the
    full chain also reduces its diagonal.
    ``construct_adapt`` builds the circuit for each bound point.
    ``construction_work`` is a declared envelope of logical construction slots
    for the longest chain, not a native synthesis, CPU or memory bound.
    ADAPT admits a family-level logical construction envelope before
    acquisition. A generator's angle-independent compiler data is
    constructed when that generator is selected and reused across its later
    angles. The envelope uses its system width, Pauli term count and
    supported compiler family. Logical construction slots and later native
    synthesis costs are reported separately.

    Returns:
        ``(construction, experiments, blocks)`` for the Plan.
    """
    q = reference.manifest.basis.dimension.bit_length() - 1
    parameters = _parameters(
        length,
        len(inputs.pool),
        len(inputs.nonidentity),
        max(len(inputs.groups), len(inputs.energy_groups)),
        residual=method.stop_criterion == "residual_norm",
        qubits=q,
    )
    experiments, definitions, blocks, kernels = [], [], [], []
    if execution == "classical":
        d = 1 << q
        # Declared host workspace: the classical query frontier of
        # adapt_acquisition.bind_classical (8L + 12 complex128 state vectors),
        # the Plan's packed action tables (adapt_actions.build_action_tables),
        # the largest shared action remainder max_A(B_A - 32d) beside the
        # input and output vectors the frontier already holds, and the
        # processed Hamiltonian's payload.
        # With optimization, an energy-and-gradient query on a chain of at
        # most L generators reserves 16 (C_cache + 2k + 12) d + 24k on top of
        # the named non-vector terms; with the old caches C_cache <= 8L and
        # k <= L that is 16 (10L + 12) d + 24L
        # (adapt_acquisition._classical_energy_gradient).
        tables = inputs.cache["action_tables"]
        vectors = (
            16 * (10 * length + 12) * d + 24 * length
            if method.optimize_every_m is not None
            else 16 * (8 * length + 12) * d
        )
        known_bytes = (
            vectors
            + tables["resident_bytes"]
            + max((size - 32 * d for size in action_sizes(inputs, d)), default=0)
            + inputs.hamiltonian.manifest.payload_bytes
            + inputs.offset_free_requirements[0]
        )
        supplied = reference.preparation.implementation == "qiskit.supplied"
        workspace = (
            Workspace(location="host", purpose="workspace", bytes=known_bytes, source=HOST),
        )
        if supplied:
            workspace += (
                Workspace(
                    location="host",
                    purpose="materialization",
                    bytes=None,
                    source=Source(
                        name="supplied reference SDK expansion",
                        version="1",
                        domain="unknown decomposition/HLS algorithmic expansion work and workspace",
                        reference="Qiskit decompose in Aer lowering and adapt_verification._lowered_native_circuits; no census or dense surrogate",
                    ),
                ),
            )
        labels = tuple(f"g_{i}" for i in range(len(inputs.pool)))
        kernel = SelectedKernel(
            name="adapt_query",
            implementation=HOST,
            inputs=tuple(
                dict(
                    (item.identity, item)
                    for item in (
                        inputs.hamiltonian.manifest.reference,
                        reference.manifest.reference,
                        *(m.processed_input for m in inputs.members),
                    )
                ).values()
            ),
            scalars=labels,
            scalar_frames=("physical",) * len(labels),
            resource_laws=(),
            workspace=workspace,
            construction_work=len(parameters) + 1 + tables["packing_work"],
            invocation_work=0,
            dependencies=("numpy", "scipy"),
        )
        kernels.append(kernel)
        definitions.append(
            Definition(
                id="root",
                node=ClassicalStage(implementation=HOST, boundary="host", kernel=kernel.name),
            )
        )
        for name, labels in (
            ("screen", labels),
            ("pair", ("s_real", "s_imag", "h_real", "h_imag")),
            ("energy", ("energy", *(f"gradient_{j}" for j in range(length)))),
        ):
            experiments.append(
                Experiment(
                    name=name,
                    setting=name,
                    observation=ObservationSpec(kind="host_scalars", labels=labels),
                )
            )
        if method.stop_criterion == "residual_norm":
            experiments.append(
                Experiment(
                    name="residual",
                    setting="residual",
                    observation=ObservationSpec(kind="host_scalars", labels=("residual_norm",)),
                )
            )
        registers, classical = (), ()
    else:
        native_identity = ":".join(member.content_id for member in inputs.members)
        registers = (
            Register(
                name="ancilla", width=ExprRef(expression="control_width"), role="clean_ancilla"
            ),
            Register(name="system", width=q),
        )
        classical = (
            (
                ClassicalValue(
                    name="outcome", dtype="bits", width=ExprRef(expression="readout_width")
                ),
            )
            if shots
            else ()
        )
        for wire in ("ancilla", "system"):
            definitions.extend(
                (
                    Definition(id=f"allocate_{wire}", node=Allocate(wire=wire)),
                    Definition(id=f"release_{wire}", node=Release(wire=wire)),
                )
            )
        if shots:
            for wire in ("ancilla", "system"):
                definitions.append(
                    Definition(id=f"measure_{wire}", node=Measure(wire=wire, result="outcome"))
                )
        # Logical construction slots of one query, one per instruction that
        # construct_adapt and the circuits it uses append. This is not a
        # native synthesis, CPU or RSS bound. A query has a left and a right
        # chain of at most `length` generators and two reference
        # preparations. Each generator costs its circuit, bounded by the
        # pool's family-level envelope B (`slot_bound`), and one instruction
        # that appends it: W = 2 W_reference + 2 length (1 + B) + 8q + 7. The
        # final 8q + 7 bounds the gates outside the generator circuits: q + 7
        # for the Hadamard test (two H, at most q controlled Pauli gates, one
        # S^dagger, two preparation instructions and their two reference
        # instructions), or 8q + 1 for a screen or energy query (one
        # preparation and its reference instruction, an inserted Pauli
        # rotation of at most 6q - 1 gates and at most 2q readout rotations).
        # Opaque reference-preparation work stays unknown when its owner has
        # no law.
        construction_work = 2 * reference.preparation.work + 2 * length * (1 + slot_bound) + 8 * q + 7
        for name in ("screen", "energy", "pair"):
            ports = ((QuantumPort(name="control", width=1),) if name == "pair" else ()) + (
                QuantumPort(name="system", width=q),
            )
            signature = BlockSignature(
                name=name, target=NATIVE, quantum=ports, parameters=parameters
            )
            semantic = BlockSemantics(
                kind="unknown",
                input=reference.manifest.reference,
                basis=reference.manifest.basis,
                relation="exact ordered generator product; pair is a phase-sensitive Hadamard or joint-state query",
                input_projector="all-zero allocated input",
                output_projector="declared selected readout",
                success="deterministic unitary preparation before requested measurement",
                workspace=0,
                restoration="system transformed; no unreported state expansion",
                epsilon=None,
                approximation_metric="phase-sensitive native scalar action",
                approximation_evidence="floating native synthesis error unknown",
                inverse_legal=False,
                control_legal=False,
                phase="reference and generator physical phase kept under internal control/inverse",
                # construct_adapt synthesizes this reference preparation, so
                # lowering and the Run admit its size like any preparation block.
                preparation=reference.preparation,
            )
            selected = SelectedDefinition(
                signature=signature,
                semantics=semantic,
                implementation=NATIVE,
                choice=f"{name}:{'shots' if shots else 'exact'}:{inputs.hamiltonian.manifest.reference.identity}:{reference.manifest.reference.identity}:{native_identity}",
                decomposition=None,
                cost_law=None,
                construction_work=construction_work,
                cost_context="selected actual prefix only; declared maximum logical construction slots, native synthesis/CPU/RSS unknown",
            )
            payload = (name, inputs, reference, compiler_plans, method)
            from .adapt_acquisition import construct_adapt

            blocks.append(
                SelectedBlock.bind(selected, payload=payload, constructor=construct_adapt)
            )
            definitions.append(
                Definition(
                    id=f"call_{name}",
                    node=BlockCall(
                        signature=name,
                        ports=tuple(
                            PortMap(
                                port=p.name, wire="ancilla" if p.name == "control" else "system"
                            )
                            for p in ports
                        ),
                        arguments=tuple(
                            Argument(parameter=p.name, value=ExprRef(expression=p.name))
                            for p in parameters
                        ),
                    ),
                )
            )
            children = (["allocate_ancilla"] if name == "pair" else []) + [
                "allocate_system",
                f"call_{name}",
            ]
            if shots:
                children.append("measure_ancilla" if name == "pair" else "measure_system")
            children.append("release_system")
            if name == "pair":
                children.append("release_ancilla")
            definitions.append(
                Definition(id=f"body_{name}", node=Sequence(children=tuple(children)))
            )
            metadata = MetadataRef(
                format=METHOD,
                data=InputRef(identity=f"adapt:{name}", representation="query law", source=METHOD),
            )
            definitions.append(
                Definition(
                    id=f"batch_{name}",
                    node=MeasurementBatch(
                        body=f"body_{name}",
                        settings=(Setting(label=name, metadata=metadata),),
                        repetitions=shots or 1,
                        observation_kind="counts" if shots else (
                            "trajectory" if name in ("pair", "screen") else "pauli_expectation"),
                    ),
                )
            )
            experiments.append(
                Experiment(
                    name=name,
                    batch=f"batch_{name}",
                    setting_index=0,
                    readout=_pair_reduction_readout(
                        inputs.hamiltonian.reference.identity, [c for _, c in inputs.nonidentity], q, diagonal=False)
                    if name == "pair" and not shots
                    else _shared_readout(inputs.hamiltonian.reference.identity,
                                         [c for _, c in inputs.nonidentity], q,
                                         screen_labels(inputs.members, inputs.cache))
                    if name == "screen" and not shots
                    else ReadoutDetails(
                        labels=() if shots else ("I" * q + "Z" if name == "pair" else "I" * q,)
                    ),
                )
            )
        definitions.append(
            Definition(id="root", node=Sequence(children=tuple(e.batch for e in experiments)))
        )
    program = Program(
        root="root",
        definitions=tuple(definitions),
        registers=registers,
        classical=classical,
        parameters=parameters,
        expressions=tuple(
            Expression(id=p.name, value=ParameterRef(parameter=p.name)) for p in parameters
        )
        if blocks
        else (),
        signatures=tuple(b.record.signature for b in blocks),
    )
    construction = SelectedConstruction(
        program=program,
        selections=tuple(b.record for b in blocks),
        kernels=tuple(kernels),
    )
    return construction, tuple(experiments), tuple(blocks)


def _shared_readout(identity, coefficients, q, labels):
    """Return the readout of the exact shared full-chain query at the end of its chain.

    Its ``diagonal`` point is the weighted-pencil reduction of the system
    state, ``H0_ii`` and the raw norm, when the Hamiltonian has a nonidentity
    term, and its ``screen`` point reads the active screening labels, when
    there are any. Both points share one preparation and receipt, and no view
    sits between them.
    """
    from nwqlib.core.planning import ObservationPoint
    from .fixed_basis import pair_reduction_point

    points = []
    if coefficients:
        points.append(pair_reduction_point(identity, coefficients, q, diagonal=True, ancilla=False,
                                           point="diagonal"))
    if labels or not points:
        points.append(ObservationPoint(id="screen", kind="pauli_expectation", labels=tuple(labels) or ("I" * q,)))
    return ReadoutDetails(positions=tuple(points))


def _pair_reduction_readout(identity, coefficients, q, *, diagonal):
    """Return the readout of an exact quantum pair query: one weighted-pencil reduction at the end."""
    from .fixed_basis import pair_reduction_point

    return ReadoutDetails(positions=(pair_reduction_point(identity, coefficients, q, diagonal=diagonal),))


class ADAPT(Method):
    """Adaptive GCiM (ADAPT-GCIM) method for the smallest eigenvalue of an `Eigenproblem`.

    It grows its trial basis from a pool of generators. Build it with
    keyword arguments and pass it as `method=`, for example
    `solve(Eigenproblem(A=H), method=ADAPT(initial_state=psi, pool=generators))`.
    `initial_state` and `pool` are required. The result is an
    [`ADAPTResult`][nwqlib.algorithms.gcim.adapt_records.ADAPTResult] whose
    `eigenvalue` is the lowest projected Ritz value in the trial basis the run
    built.

    The method follows ADAPT-GCIM of Zheng et al., npj Quantum Inf. 10, 127
    (2024), arXiv:2312.07691v3. Each round screens the current product state
    with the ADAPT gradient `<[H, A_i]>` of every unselected generator (main
    text, "Integration of GCIM with ADAPT approach", p. 4, and Grimsley et
    al. (2019), arXiv:1812.11173v2, Sec. II.B, steps 5-7). The selected
    generator enters at the fixed angle `theta`, and the basis grows by two
    states per selection (METHODS, p. 9). A selected generator is not
    screened again, following the paper's QuGCM `adapt_gcim.py`
    implementation (commit c5efcb0, lines 303-398), whereas Grimsley et al.
    step 7 permits reuse in its ADAPT-VQE ansatz. Optimization every
    `optimize_every_m` selections is the intermittent, truncated optimization
    of Appendix H, and its extra energy evaluations run through the same Run.
    Because later queries act on generators selected from earlier outcomes,
    `prepare(plan, settings="all")` is refused.

    The default flat-counter stop follows METHODS, p. 9: stop after
    `T = min(T_auto, T_usr)` consecutive energy changes below
    `energy_change_tolerance`, with `T_auto` equal to `t_auto_fraction`
    (20%) of the unselected generators and `T_usr = t_user`. Counting starts
    at the first change from the reference energy, and the integer counter
    reaches a fractional T at its ceiling. QuGCM's code instead uses a max
    rule with 10% and a warmup. With `max_iterations=8` and `t_user=10`, the
    flat counter can fire only for pools of at most 42 generators. After k
    selections a pool of P generators has `P - k` unselected members and a
    flat count of at most k, so the stop needs `k >= 0.2*(P - k)`. The
    iteration cap is checked first and ends the run at the eighth selection,
    so `k <= 7` and `P <= 42`. Larger pools reach the iteration cap first
    unless another stop applies. Every stop, including the gradient-norm
    floor, is an operational rule, not proof of ground-state accuracy.
    `result.analyze(overlap_cutoff=...)` solves the saved basis again at
    another cutoff. The [GCiM guide](../../algorithms/gcim.md) describes the
    screening rule, the pools and the stopping rules.

    Attributes:
        initial_state: Required. Reference state in the operator's original
            basis, in a form that `StateData` accepts.
        pool: Required. A nonempty tuple of anti-Hermitian generators,
            `FermionicGenerator` records or operators that `operator_input`
            accepts, or the name of a built-in fermionic pool:
            `"spin_adapted_sd"`, `"uccsd_sd"`, `"qeb_sd"` or `"ceo_ovp"`.
            Pools other than `"spin_adapted_sd"` need an occupation-number
            reference state.
        n_spatial_orbitals: Default `None`. Number of spatial orbitals,
            required by a named pool, with twice as many qubits. `None` for
            an explicit pool.
        theta: Default `pi/4`, nonzero. Angle in radians of each selected
            generator's exponential.
        gradient_norm_floor: Default `1e-8`, nonnegative. The run stops when
            the norm of the pool gradients falls below it.
        energy_change_tolerance: Default `1e-6`, positive. Absolute change
            between consecutive energies below which the flat counter counts
            a round.
        t_auto_fraction: Default `0.2`, in (0, 1]. Fraction of the unselected
            pool size that sets `T_auto`.
        t_user: Default `10`. Cap `T_usr` on consecutive flat rounds in the
            rule `min(T_auto, T_usr)`. A fractional round count is rounded up.
        max_iterations: Default `8`. Largest number of selection rounds. The
            basis holds at most `2*min(max_iterations, pool size)` states, so
            `max_basis_size=64` allows at most 32 selections for a pool of
            more than 32 members. The default is a cost cap. With exact
            evaluation and the spin-adapted pool, the H4 chain of the eigenvalue example reaches
            FCI at the eighth selection, and 10-qubit LiH is then 0.46 mHa
            above its FCI energy. The flat counter would stop them only at
            selections 18 and 19
            ([Engineering constants](../../ENGINEERING_CONSTANTS.md)).
        overlap_cutoff: Default `1e-12`, positive. Overlap eigenvalue cutoff
            that defines the kept Gram subspace.
        symmetrize_matrices: Default `True`. Request Hermitian averaging in
            the projected solve. The raw measured values are still checked.
        stop_criterion: Default `"flat_counter"`, which counts repeated small
            changes. `"residual_norm"` uses the full-state residual
            `||(H - E) psi||` of the normalized Ritz state psi of the current
            basis, which only classical execution can evaluate. It differs
            from the pencil's `projected_residual` `||H c - E S c||` in basis
            coordinates.
        residual_norm_tolerance: Default `1.6e-3`, positive. Threshold on that
            full-state residual for the residual stop, in the Hamiltonian's
            energy unit.
        optimize_every_m: Default `None`, no optimization. Run coefficient
            optimization every m selected generators.
        optimize_rounds_n: Default `None`. Largest number of BFGS iterations
            per optimization, required together with the other two
            optimization settings.
        optimize_max_evaluations: Default `None`. Largest number of energy
            queries for enabled optimization, counted as distinct measured
            parameter points. A classical point is one energy-and-gradient
            evaluation, whose work record includes its forward, derivative
            and reverse Taylor work.
        max_basis_size: Default `64`. Limit on the size of the growing trial
            basis, checked before matrix construction.
        max_pool_size: Default `1024`. Limit on the number of generators,
            checked before a pool is expanded.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Limit on
            the known bytes of the pool, matrices, states and readout
            workspace.
        max_products: Default `1_000_000_000`. Limit on the counted work of
            conversion, matrices and operator products. It also limits,
            separately, the exact synthesis of the dense unitaries in a
            supplied reference circuit and Qiskit's control of them. Before a
            classical query starts, its reference, generator, Hamiltonian and
            scalar work are summed and checked, and each later query is
            checked afresh. Packed Pauli tables are checked once per Plan.
            The recorded allowance covers a whole query, while the counters
            record the actions performed after cache reuse.
        input_conversion: Default `"auto"`, which keeps the accepted input
            access. `"dense_pauli"` permits explicit dense-to-Pauli
            conversion.
        max_conversion_work: Default `100_000_000`. Separate limit on the
            work of the chosen input conversion.
        pauli_coefficient_cutoff: Default `1e-14`, nonnegative. Operator
            coefficients at or below it in magnitude are removed, and the
            removed mass is recorded.
        input_symmetry_tolerance: Default `1e-12`, nonnegative. Relative
            Frobenius window for the anti-Hermitian projection of each pool
            generator. The Hamiltonian gets no window and must be exactly
            Hermitian.

    Raises:
        ValueError: If `theta` is zero, if only some of the three
            optimization settings are given, or if an explicit pool is empty
            or has more than `max_pool_size` members.

    Examples:
        `H = Z` has lowest eigenvalue -1. From `|+>`, ADAPT with the
        one-generator pool `{iY}` returns it after one selection:

        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import ADAPT
        >>> from nwqlib.operators import ingest_pauli
        >>> pool = (ingest_pauli((("Y", 1j),), num_qubits=1),)
        >>> method = ADAPT(initial_state=[1, 1], pool=pool)
        >>> result = solve(Eigenproblem(A=[[1, 0], [0, -1]]), method=method, seed=7)
        >>> print(result.selected, result.stop_reason, round(result.eigenvalue, 10))
        (0,) pool_exhausted -1.0
    """

    # max_pool_size, max_products and max_basis_size are untuned workload
    # ceilings, and the numerical defaults below are registered with their
    # revisit conditions in docs/ENGINEERING_CONSTANTS.md.
    max_pool_size: PositiveInt = 1024
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_products: PositiveInt = 1_000_000_000
    input_conversion: Literal["auto", "dense_pauli"] = "auto"
    max_conversion_work: PositiveInt = DEFAULT_CONVERSION_WORK
    initial_state: StateData
    pool: Any
    n_spatial_orbitals: PositiveInt | None = None
    theta: Real = pi / 4
    gradient_norm_floor: Annotated[Real, Field(ge=0)] = 1e-8
    energy_change_tolerance: Annotated[Real, Field(gt=0)] = 1e-6
    t_auto_fraction: Annotated[Real, Field(gt=0, le=1)] = 0.20
    t_user: PositiveInt = 10
    max_iterations: PositiveInt = 8
    overlap_cutoff: Annotated[Real, Field(gt=0)] = 1e-12
    symmetrize_matrices: bool = True
    stop_criterion: Literal["flat_counter", "residual_norm"] = "flat_counter"
    residual_norm_tolerance: Annotated[Real, Field(gt=0)] = 1.6e-3
    optimize_every_m: PositiveInt | None = None
    optimize_rounds_n: PositiveInt | None = None
    optimize_max_evaluations: PositiveInt | None = None
    max_basis_size: PositiveInt = 64
    pauli_coefficient_cutoff: Annotated[Real, Field(ge=0)] = 1e-14
    input_symmetry_tolerance: Annotated[Real, Field(ge=0)] = 1e-12
    result_type: ClassVar[type] = ADAPTResult

    @field_validator("pool", mode="before")
    @classmethod
    def _pool(cls, value, info):
        """Snapshot an explicit pool, or pass a built-in pool name through.

        The member count is checked against ``max_pool_size`` before any
        member is copied. Each ``FermionicGenerator`` is admitted by
        ``_snapshot_generator`` and every other entry by ``operator_input``.
        """
        if isinstance(value, str):
            return value
        if type(value) not in (tuple, list) or not value:
            raise ValueError("pool must be a named family or finite nonempty sequence")
        cap = info.data.get("max_pool_size", 1024)
        if len(value) > cap:
            raise ValueError(f"pool has {len(value)} members, more than ADAPT(max_pool_size={cap})")
        max_bytes = info.data.get("max_bytes", DEFAULT_MAX_BYTES)
        max_products = info.data.get("max_products", 1_000_000_000)
        return tuple(
            _snapshot_generator(g, max_bytes=max_bytes, max_products=max_products)
            if isinstance(g, FermionicGenerator)
            else operator_input(g, max_bytes=max_bytes)
            for g in value
        )

    @field_serializer("pool")
    def _pool_record(self, value):
        """Serialize a supplied operator as its input record and a generator as its fields."""
        from .adapt_archive import _write_generator

        if isinstance(value, str):
            return value
        return tuple(
            dict(kind="operator", data=g.to_record())
            if isinstance(g, OperatorInput)
            else dict(kind="generator", data=_write_generator(g))
            for g in value
        )

    @model_validator(mode="after")
    def _controls(self):
        """Reject a zero angle and a partial set of optimization controls.

        At ``theta = 0`` every generator exponential ``exp(theta A)`` is the
        identity, so each new basis state would repeat the reference.
        Optimization needs its period, BFGS iteration cap and total
        evaluation allowance together.
        """
        if self.theta == 0:
            raise ValueError("new-generator theta must be nonzero")
        enabled = self.optimize_every_m is not None
        if enabled != (self.optimize_rounds_n is not None) or enabled != (
            self.optimize_max_evaluations is not None
        ):
            raise ValueError(
                "optimization requires period, rounds and an explicit total evaluation limit"
            )
        return self

    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR

    def plan(self, problem, *, output, execution, shots, rng):
        """Check the generator pool and choose the gradient, matrix-element and energy queries.

        After k selections the basis holds 2k states, so
        `2*min(pool size, max_iterations)` bounds the projected dimension,
        which must not exceed `max_basis_size`. For an explicit pool this
        bound is checked before conversion and commutator formation. A named
        pool is checked after it is enumerated, under that enumeration's own
        byte and product limits.

        Raises:
            ValueError: If `2*min(pool size, max_iterations)` exceeds
                `max_basis_size`, if classical execution is given shots, or if
                `stop_criterion="residual_norm"` is used without classical
                execution.
            ApplicabilityError: If the Problem names a fixed subspace, which
                the changing basis of ADAPT does not implement.
        """
        if (
            isinstance(self.pool, tuple)
            and 2 * min(len(self.pool), self.max_iterations) > self.max_basis_size
        ):
            raise ValueError(
                f"maximum selected ADAPT basis holds {2 * min(len(self.pool), self.max_iterations)} states "
                f"(2 * min(pool size {len(self.pool)}, max_iterations={self.max_iterations})), above "
                f"ADAPT(max_basis_size={self.max_basis_size})")
        if problem.subspace is not None:
            raise ApplicabilityError(
                "ADAPT's changing trial basis does not implement a supplied fixed scientific subspace"
            )
        if execution == "classical" and shots is not None:
            raise ValueError("classical ADAPT queries do not use shots")
        if self.stop_criterion == "residual_norm" and execution != "classical":
            raise ValueError("residual_norm stopping requires explicit classical acquisition")
        from nwqlib.algorithms._eigen_inputs import conversion_requirements
        from nwqlib.operators.inputs import OperatorInput

        matrices = ([problem.A] if execution == "quantum" else []) + (
            [g for g in self.pool if isinstance(g, OperatorInput)]
            if isinstance(self.pool, tuple)
            else []
        )
        conversion_work = 0
        for matrix in matrices:
            if "pauli_terms" not in matrix.manifest.access:
                conversion_work += conversion_requirements(
                    matrix,
                    input_conversion=self.input_conversion,
                    max_conversion_work=self.max_conversion_work,
                    max_bytes=self.max_bytes,
                )[2]
        if conversion_work > self.max_conversion_work:
            raise ValueError("ADAPT complete input conversion exceeds max_conversion_work")
        operator = eigen_operator(
            problem,
            output,
            max_bytes=self.max_bytes,
            as_pauli=execution == "quantum",
            input_conversion=self.input_conversion,
            max_conversion_work=self.max_conversion_work,
            pad_classical=True,
        )
        reference = eigen_state(self.initial_state, problem, max_bytes=self.max_bytes)
        if execution == "quantum" and reference.preparation.implementation == "qiskit.supplied":
            # Each Hadamard-test and joint-state query controls the reference preparation
            # (adapt_acquisition._native_preparation), and the Run's method
            # context builds that controlled gate once. qiskit_compat.controlled
            # synthesizes each distinct dense unitary of a supplied circuit
            # exactly, and Qiskit then controls every occurrence of the
            # synthesized gates.
            from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses
            from nwqlib.subroutines.qiskit_compat import dense_control_counts, dense_synthesis_widths
            admit_dense_syntheses(dense_synthesis_widths(reference._native),
                                  controls=(dense_control_counts(reference._native, 1),),
                                  max_work=self.max_products, max_bytes=self.max_bytes,
                                  operation="ADAPT controlled reference synthesis")
        reference_requirements = preparation_requirements(reference)
        inputs = prepare_adapt_inputs(
            operator,
            reference,
            self,
            execution=execution,
            shots=shots,
            original_dimension=problem.dimension,
            original_hamiltonian=problem.A.reference,
            reference_requirements=reference_requirements,
        )
        length = min(self.max_iterations, len(inputs.pool))
        if 2 * length > self.max_basis_size:
            raise ValueError(
                f"maximum selected ADAPT basis holds {2 * length} states "
                f"(2 * min(pool size {len(inputs.pool)}, max_iterations={self.max_iterations})), above "
                f"ADAPT(max_basis_size={self.max_basis_size})")
        # After `length` selections the basis has 2 * length states
        # (adapt_acquisition.basis_chains), the largest the controller can
        # reach. Its projected solve must fit both limits that
        # adapt_acquisition._matrix_from_observations checks before each
        # analysis. A run that might stop before reaching that basis is still
        # refused here, so no admitted run fails these two checks at a late
        # analysis.
        _check_bytes(_projected_solve_bytes(2 * length), self.max_bytes, "ADAPT projected solve")
        _check_products(_projected_solve_work(2 * length), self.max_products)
        compiler_plans, slot_bound = (), 0
        if execution == "quantum":
            # Only the route eligibility is checked here. Each generator's
            # compiler plan is built when it is first selected. A pool whose
            # generators all commute uses the commuting route's 6q max(1, T).
            commuting = _admit_generator_routes(
                inputs.pool, max_bytes=self.max_bytes, max_products=self.max_products
            )
            q = reference.basis.dimension.bit_length() - 1
            counts = tuple(len(g.pauli_terms) for g in inputs.pool)
            slot_bound = (
                6 * q * max(1, max(counts)) if commuting else generator_pool_slot_bound(q, counts)
            )
            compiler_plans = CompilerPlans(
                inputs.pool, max_bytes=self.max_bytes, max_products=self.max_products
            )
        construction, experiments, blocks = _construction(
            inputs, reference, self, length, compiler_plans, slot_bound, execution, shots
        )
        rec = AdaptReconstruction(
            reference=reference.preparation,
            original_hamiltonian=problem.A.reference,
            processed_hamiltonian=inputs.hamiltonian.reference,
            terms=inputs.terms,
            pool=inputs.members,
            pool_name=self.pool if isinstance(self.pool, str) else "explicit",
            max_selections=length,
            num_qubits=reference.basis.dimension.bit_length() - 1,
            hamiltonian_action_work=inputs.hamiltonian_work,
            hamiltonian_dropped_l1=inputs.dropped_l1,
            groups=inputs.groups,
            energy_groups=inputs.energy_groups,
            processing=inputs.processing,
            hamiltonian_symmetry=inputs.hamiltonian_symmetry,
            processing_source=inputs.processing_source,
            reference_action_work=reference_requirements[1],
            reference_action_bytes=reference_requirements[0],
            host_identity_shift=inputs.host_identity_shift,
            offset_free_action_bytes=inputs.offset_free_requirements[0],
            offset_free_action_work=inputs.offset_free_requirements[1],
        )
        # eigen_error_model reads only the output, problem and construction,
        # so the Plan is validated once with its error model.
        components = SimpleNamespace(output=output, problem=problem, construction=construction)
        plan = Plan(
            problem=problem,
            method=self,
            output=output,
            execution=execution,
            shots=shots,
            randomness=rng.snapshot(),
            construction=construction,
            experiments=experiments,
            reconstruction=rec,
            assumptions=(
                "processed-input adaptive Ritz estimate; ground identity and simultaneous adaptive coverage unresolved",
            ),
            error_model=eigen_error_model(components, METHOD),
        )
        if execution == "classical":
            from .adapt_acquisition import bind_classical

            blocks = (bind_classical(plan, inputs, reference),)
        extra = {}
        if execution == "quantum" and shots is None:
            from .fixed_basis import _pair_table

            extra["pair_table"] = _pair_table(inputs.hamiltonian)
        return plan._bind(
            blocks=blocks, inputs=inputs, reference=reference, compiler_plans=compiler_plans, **extra
        )

    def reduction_allowance(self, plan, point, *, observation, width, run):
        """Admit an exact pair reduction's private workspace and return the host-work allowance of its query.

        The workspace gate refuses a point whose ``pair_reducer.pair_bytes``
        exceeds ``max_bytes``. The controller admits the registered work of
        every new pair reduction of a matrix stage against ``max_products``
        before the stage's first query, so one query's allowance is the whole
        ``max_products``. The ledger is read, never changed.
        """
        import json
        from .pair_reducer import pair_bytes

        for item in observation.positions:
            if item.kind == "reduction":
                _check_bytes(pair_bytes(json.loads(item.parameters)), self.max_bytes,
                             "ADAPT pair reduction workspace")
        return int(self.max_products)


    def prepare_all_refusal(self):
        """ADAPT refuses ``settings="all"``, because its later queries act on generators selected from earlier outcomes."""
        return (
            "ADAPT cannot prepare every setting in advance, because each iteration screens and "
            "projects with the generators selected from the outcomes of earlier queries. prepare(plan) "
            "with the default settings='first' prepares the controller's first query, and "
            "estimate(plan) folds the Plan's resource laws without building circuits"
        )

    def prepare(self, plan, *, run):
        """Prepare the controller's first new query (adapt_acquisition.prepare_adapt)."""
        from .adapt_acquisition import prepare_adapt

        return prepare_adapt(plan, run=run)

    def execute(self, plan, *, run):
        """Advance the ADAPT controller on the Run's original work (adapt_acquisition.drive_adapt)."""
        from .adapt_acquisition import drive_adapt

        return drive_adapt(plan, run=run)

    def recover_analysis(self, plan, *, run):
        from .adapt_acquisition import recover_adapt_analysis

        return recover_adapt_analysis(plan, run=run)

    def analyze(self, plan, data, *, settings):
        from .adapt_acquisition import analyze_adapt

        return analyze_adapt(plan, data, settings=settings)

    def verify(self, plan, result, *, checks):
        from nwqlib.evidence.verification import ProjectedVerificationOptions, verify_projected

        if isinstance(checks, ProjectedVerificationOptions):
            return verify_projected(result, options=checks)
        from .adapt_verification import verify

        return verify(plan, result, options=checks)

    def save_archive(self, plan, files):
        from .adapt_archive import save

        return save(plan, files)

    @classmethod
    def load_archive(cls, data, files):
        from .adapt_archive import load

        return load(data, files)

    def save_run_context(self, context, files):
        """Save the state and action caches; the Run or Result owns the collected observations.

        A live Run's journal and a saved Result's RunData hold every collected
        chunk and its order, so the context marks ``observations_in_run`` and
        refers to them instead of storing a second copy of each chunk. Array
        and gate files are written once per object and shared by later
        checkpoints (``ArchiveFiles.write_array``).
        """
        from .adapt_archive import save_context

        return save_context(context, files)

    @staticmethod
    def _load_live_run_context(data, files, observations):
        """Restore indices once from actual collected data, without analysis."""
        from .adapt_archive import load_context
        from .adapt_acquisition import _observation_index

        context = load_context(data, files)
        for key, observation in _observation_index(observations.chunks).items():
            context.observations.setdefault(key, observation)
        for chunk in observations.chunks:
            if chunk.content_id not in context.contribution_seen:
                context.contribution_seen.add(chunk.content_id)
                context.contribution_order.append(chunk.content_id)
        return context

    @staticmethod
    def snapshot_result_context(context):
        """Share immutable scientific data; keep mutable native caches private.

        The observation index is left out: the Result's RunData owns the
        collected chunks and their order, and a saved Result refers to them
        (``save_run_context``), so a live and a reloaded Result carry the same
        context. ``compiler_plans`` becomes the read-only ``(pool index,
        kind, active modes, occupation blocks)`` entries built so far, so a
        Result from a reopened Run and its saved copy carry the compiler
        plans of the selected generators, which native verification adopts.
        """
        from collections.abc import Mapping
        from dataclasses import fields
        from types import MappingProxyType
        from .adapt_acquisition import AdaptContext
        from .adapt_archive import _compiler_entries

        def freeze(value):
            if isinstance(value, Mapping):
                return MappingProxyType({key: freeze(item) for key, item in value.items()})
            if isinstance(value, (tuple, list)):
                return tuple(freeze(item) for item in value)
            if isinstance(value, (set, frozenset)):
                return frozenset(value)
            return value

        return MappingProxyType(
            {
                field.name: None if field.name == "result"
                else freeze(_compiler_entries(context.compiler_plans)) if field.name == "compiler_plans"
                else freeze(getattr(context, field.name))
                for field in fields(AdaptContext)
                if field.name not in {"gates", "preparations", "reference_gates", "matrix_data", "pair_values",
                                      "observations", "contribution_order", "contribution_seen"}
            }
        )

    @staticmethod
    def load_run_context(data, files):
        from .adapt_archive import load_context

        return load_context(data, files)

    def validate_point(self, plan, experiment, values):
        """Require canonical active/inactive chain slots and an actual supported query insertion."""
        experiment = experiment.name
        r = plan.reconstruction
        if number(values["control_width"]) != int(experiment == "pair") or number(
            values["readout_width"]
        ) != (1 if experiment == "pair" else r.num_qubits):
            raise ValueError("query width must match its actual system/ancilla acquisition")
        for side in ("l", "r"):
            count = int(number(values[f"{side}_count"]))
            indices = []
            for i in range(r.max_selections):
                index, theta = (
                    number(values[f"{side}_pool_{i}"]),
                    number(values[f"{side}_theta_{i}"]),
                )
                if i < count:
                    if not 0 <= index < len(r.pool):
                        raise ValueError("active generator index outside the admitted pool")
                    indices.append(index)
                elif index != -1 or theta != 0:
                    raise ValueError("inactive slots require index=-1 and theta=0")
            if len(indices) != len(set(indices)):
                raise ValueError("a selected generator cannot occur twice in one chain")
        pos, term, sign = (
            number(values[k]) for k in ("insert_position", "insert_term", "insert_sign")
        )
        right = chain(values, "r")
        if pos == -1:
            if term != -1 or sign != 0:
                raise ValueError("inactive insertion must be canonical")
        elif (
            experiment != "energy"
            or not 0 <= pos < len(right)
            or not 0 <= term < len(r.pool[right[pos][0]].terms)
            or sign not in (-1, 1)
        ):
            raise ValueError("insertion must select an actual energy derivative term")
        if experiment != "pair" and chain(values, "l"):
            raise ValueError("only a pair query accepts a left chain")
        if experiment != "pair" and (
            number(values["term"]) != -1 or number(values["quadrature"]) != 0
        ):
            raise ValueError("inactive pair term/quadrature must be canonical")
        if experiment == "pair" and number(values["group"]) != 0:
            raise ValueError("pair queries have one ancilla setting, not a QWC group")
        if plan.execution == "classical" and (
            number(values["term"]) != -1
            or number(values["quadrature"]) != 0
            or number(values["group"]) != 0
        ):
            raise ValueError("THEORY returns all pair scalars in one canonical host query")
        return None

    def labels_at(self, plan, experiment, values):
        """Return the scalar labels one bound query must return.

        For quantum plans, screening reads only the commutator labels of
        generators not yet in the chain, energy queries read the nonidentity
        Hamiltonian labels, and a pair query reads the ancilla Z label. A
        sampled query reads the subset that falls in its QWC group. Classical
        plans return host scalars instead: ``g_i`` per unselected generator,
        the four real and imaginary H/S parts of a pair, ``energy`` with
        ``gradient_j`` for each angle of the chain (only ``energy`` for a
        Pauli-insertion query) or ``residual_norm``. Both planning and
        ``_read_values`` use this one definition, so an observation carrying
        any other label is rejected.
        """
        r = plan.reconstruction
        if plan.execution == "classical":
            if experiment == "screen":
                removed = {i for i, _ in chain(values, "r")}
                return tuple(f"g_{i}" for i in range(len(r.pool)) if i not in removed)
            if experiment == "residual":
                return ("residual_norm",)
            if experiment == "pair":
                return ("s_real", "s_imag", "h_real", "h_imag")
            if int(number(values["insert_position"])) >= 0:
                return ("energy",)
            return ("energy", *(f"gradient_{j}" for j in range(len(chain(values, "r")))))
        if experiment == "pair":
            return ("I" * r.num_qubits + "Z",)
        if experiment == "screen":
            removed = {i for i, _ in chain(values, "r")}
            labels, members = active_screen_labels(r.pool, label_cache(plan), removed, r.max_selections)
            ordered, groups = screen_labels(r.pool, label_cache(plan)), r.groups
        else:
            labels = tuple(t.label for t in r.nonidentity_terms)
            members = frozenset(labels)
            ordered = energy_labels((t.label for t in r.nonidentity_terms), label_cache(plan))
            groups = r.energy_groups
        if plan.shots is not None:
            group = int(number(values["group"]))
            if not 0 <= group < len(groups):
                raise ValueError("query group is outside the admitted table")
            labels = tuple(ordered[i] for i in groups[group] if ordered[i] in members)
        return labels

    def specialize_experiment(self, plan, experiment, values):
        """Narrow the planned readout to the labels of this point.

        An empty label set would be a zero-work query, so the controller must
        skip it before preparation rather than submit it. An exact quantum
        pair query reads one weighted-pencil reduction instead of labels; its
        layout is the diagonal system state when both chains are equal and
        the ancilla-bit-0 joint state otherwise.
        """
        r = plan.reconstruction
        if experiment.name == "pair" and plan.execution == "quantum" and plan.shots is None:
            return experiment.revise(readout=_pair_reduction_readout(
                r.processed_hamiltonian.identity, [t.coefficient for t in r.nonidentity_terms], r.num_qubits,
                diagonal=chain(values, "l") == chain(values, "r")))
        if experiment.name == "screen" and plan.execution == "quantum" and plan.shots is None:
            # The exact full-chain query: the chain's weighted diagonal
            # reduction and its active screening labels at the same boundary.
            labels = self.labels_at(plan, experiment.name, values)
            if not labels and not r.nonidentity_terms:
                raise ValueError("mathematically empty query is zero work; skip it before prepare")
            return experiment.revise(readout=_shared_readout(
                r.processed_hamiltonian.identity, [t.coefficient for t in r.nonidentity_terms], r.num_qubits,
                labels))
        labels = self.labels_at(plan, experiment.name, values)
        if not labels:
            raise ValueError("mathematically empty query is zero work; skip it before prepare")
        if experiment.observation is not None:
            observation = ObservationSpec(kind="host_scalars", labels=labels)
            return experiment.revise(observation=observation)
        readout = ReadoutDetails(
            labels=() if plan.shots is not None else labels,
        )
        return experiment.revise(readout=readout)

    def selected_kernels(self, plan, experiment, program):
        """Admit all actions and other work of one classical query together.

        ``max_products`` bounds this complete invocation. The sum
        (``adapt_actions.query_work``) covers the reference, generator,
        Hamiltonian and scalar work of the invocation before it starts, so
        every actual query term is bounded by its nonnegative outer charge
        and no constituent can exceed the cap; the inner matvec gate may
        still receive the whole cap. Every later invocation is checked
        afresh. Resolution, analysis and repeated preparation checks do not
        consume an execution budget, so this method keeps no Run-wide
        ledger. Returns the host-kernel declaration revised with this
        point's labels and its full ``invocation_work``, or an empty tuple
        for quantum plans. Application counters record the actions actually
        performed after cache reuse.
        """
        from .adapt_actions import query_work

        if not plan.construction.kernels:
            return ()
        work = query_work(plan, experiment, program)
        if work > self.max_products:
            raise ValueError(
                f"ADAPT classical query needs {work} scalar products, exceeding "
                f"max_products={self.max_products}. Later queries with longer "
                "generator chains can need more."
            )
        labels = experiment.observation.labels
        return (
            plan.construction.kernels[0].revise(
                scalars=labels,
                scalar_frames=("physical",) * len(labels),
                invocation_work=work,
                resource_laws=(),
            ),
        )
