"""The ADAPT Method: pool checks, the parameterized construction that every query reuses, and the generator compiler plans.

The ADAPT iteration itself is ``adapt_acquisition.drive_adapt``.
"""

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
from nwqlib.core.records import InputRef, Record, Source, PositiveInt, Real
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
from nwqlib.operators.access import Count, _check_bytes, _check_products
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
from .fixed_basis import (
    PencilAcquisition,
    PencilSampling,
    grouped_pass_requirements,
    real_generator_reason,
    real_hamiltonian_reason,
    real_state_reason,
    resolve_acquisition,
    sampled_setting_count,
)
from .pencil import _projected_solve_bytes, _projected_solve_work
from .pencil_workflow import ONE_QUERY_ESTIMATE

METHOD = Source(
    name="adapt_gcim",
    version="5",
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
    version="3",
    domain="actual ordered generator, Hadamard and joint-state queries",
    reference="nwqlib.algorithms.gcim.adapt_acquisition.construct_adapt",
)
HOST = Source(
    name="adapt.classical_query",
    version="2",
    domain="explicit classical selected state query",
    reference="nwqlib.algorithms.gcim.adapt_acquisition.bind_classical",
)


def _parameters(length, pool_size, term_count, group_count, *, residual=False, qubits=1, stage_shots=None):
    """Declare the finite parameter vector shared by every ADAPT query.

    One Program serves all adaptive queries, so each query is a point in this
    space rather than a new construction. ``l_*`` and ``r_*`` hold the left and
    right generator chains with at most ``length`` positions. Unused positions
    take canonical values (index -1, angle 0), so equal queries have one
    content hash. ``term`` and ``quadrature`` select a Hadamard-test Pauli
    term and real or imaginary part. ``group`` selects a sampled QWC
    measurement group, for a grouped pencil setting a group of the
    nonidentity Hamiltonian terms (``energy_groups``). The
    ``insert_*`` fields place one Pauli-insertion shift for measured
    derivatives. Residual queries also carry the Ritz energy and coefficients.
    A ``PencilSampling`` Plan that measures gives ``stage_shots``, the
    largest number of shots that one query can bind, and every query then
    also binds ``stage`` (-1 for a selection query, 0 for the pilot and 1
    for the main stage of the final pencil) and ``shots``, the repetitions
    of its batch. A selection query binds the Plan's ``shots``, and the
    bound ``stage`` makes the selection, pilot and main queries of one
    setting distinct points.
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
    if stage_shots is not None:
        parameters.extend((Parameter(name="stage", domain="integer", lower=-1, upper=1),
                           Parameter(name="shots", domain="integer", lower=1, upper=stage_shots)))
    return tuple(parameters)


def generator_pool_slot_bound(num_qubits, term_counts):
    """Bound generator construction from the system width and stored Pauli term counts.

    The commuting route uses at most 6q instructions per term. Pair/split,
    four-distinct and shared-index default routes fit 24q+6192 instructions, using
    at most six active modes and occupation blocks of dimension at most five
    for the last route. The bound counts logical appended instructions, not
    backend synthesis gates. It does not establish support for an arbitrary
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
    below ``24q+6192``, the triplet four-distinct bound. Pair/split is
    also below that value for ``q >= 1``. A route-independent bound for any
    supported generator is thus ``B(q,T) = max{6q max(1,T), 24q+6192}``. It
    uses the O(P) existing tuple-length metadata of a P-member pool, with no
    state expansion, commutation count or occupation-block planning.
    """
    counts = tuple(term_counts)
    if not counts:
        return 0
    largest = max(counts)
    return max(6 * num_qubits * max(1, largest),
               24 * num_qubits + 6192)


def _admit_generator_routes(pool, *, max_bytes, max_products, vacuum=False):
    """Reject during planning a generator that no circuit route supports, and return whether all commute and the vacuum reason.

    This is the eligibility part of ``fermionic_circuits._plan_generator_circuit``.
    Its numeric charge for a generator of ``m`` rows on ``q`` qubits,
    ``m*(32 + 8*digits)`` bytes and ``m*(q*digits + 1) + m(m-1)/2*(digits + 1)``
    work, is the planning estimate for the mask construction and the pair
    scan of the commutation test. Two Pauli words with bit masks ``(x_i, z_i)`` and
    ``(x_j, z_j)`` commute exactly when ``popcount(x_i & z_j) +
    popcount(z_i & x_j)`` is even, and a generator whose terms do not all
    commute must be a normalized default GSD double
    (``_require_default_double``). Occupation blocks and compiler plans are
    built only when a generator is first selected (``CompilerPlans``).

    It returns ``(commuting, reason)``. With ``vacuum`` the same loop over the
    generators also decides the structural vacuum rule of the vacuum pair
    constructions (``PencilAcquisition.pair_preparation``), and ``reason``
    names the first generator that fails it. ``reason`` is None when every
    generator passes or ``vacuum`` is False. The rule is about the literal
    chain U after the reference preparation B, not about U B, and it is
    sufficient, not necessary. It reads the stored rows and never a family
    name or a number-conservation label.

    - **Commuting rows** ``i c_P P``, with ``c_P`` the stored imaginary part
      (the coefficient ``build_generator_circuit`` reads). With ``x(P)`` the
      X/Y flip mask and ``y(P)`` the number of Y letters,
      ``P|vac> = i**y(P) |x(P)>``. For each exact nonzero magnitude
      ``d = |c_P|`` and each nonzero ``x``, the Gaussian integer
      ``b[d, x] = sum_(|c_P| = d, x(P) = x) sign(c_P) i**y(P)`` must be
      exactly zero. Exact-zero coefficients are skipped, as the constructor
      skips them, and no magnitude cutoff applies. The constructor emits
      the angle ``phi_P = RN(RN(-2 theta) c_P)``, and round-to-nearest is
      symmetric in sign, so the finite emitted angles of one bucket are
      ``sign(c_P) phi_d`` for one ``phi_d``. With
      ``C_d = sum sign(c_P) P`` over the bucket's nonidentity words the
      integer condition gives ``C_d|vac> = a_d|vac>``, where ``a_d`` sums
      the signs of the bucket's I/Z words, and since the words commute the
      bucket's rotations multiply to ``exp(-i phi_d C_d / 2)``, which maps
      the vacuum to ``exp(-i phi_d a_d / 2)|vac>``. A cancellation only
      in the weighted sum of unequal magnitudes does not qualify. The
      phase of one realized instance is read from its circuit
      (``adapt_acquisition._vacuum_phase``).
    - **Default doubles**, the noncommuting generators admitted above. Their
      complete ``pair_split``, ``shared_index`` and ``four_distinct`` circuits
      map the vacuum to itself with phase 0. The ``pair_split`` factors
      rotate only between two occupations whose excitation controls are all
      1, and the vacuum has a control at 0. In ``shared_index`` the vacuum
      is a singleton occupation block, which the construction skips, and the
      routing SWAP and CZ act on it with +1. In ``four_distinct`` the
      rounded Walsh angles of the multiplexor have the form
      ``(-1)**popcount(t) v_h`` over the three excitation-control bits t,
      so at every control address whose excitation bits are not all 1,
      where the vacuum is, the effective rotation angle is zero.

    The vacuum rule reuses the masks and Y counts that ``_pauli_masks``
    returns for the commutation test. It makes one pass over the stored rows,
    which adds integer sign sums to the buckets, each row adding at most one
    key, and then one scan of the bucket values, without a state vector, a
    matrix or a Pauli product. The Y counts added to the masks, the
    magnitude-keyed Gaussian-integer buckets and the temporaries kept from
    the previous generator while the next one is admitted are outside the
    numeric charge above, which is no qualified combined bound for them. The
    count of bucket keys is no CPU, heap or complete scalar-work guarantee.
    """
    from sys import int_info
    from nwqlib.operators._pauli import _pauli_masks

    commuting, reason = True, None
    for generator in pool:
        rows = generator.pauli_terms
        m, q = len(rows), generator.num_qubits
        digits = max(1, (q + int_info.bits_per_digit - 1) // int_info.bits_per_digit)
        pairs = m * (m - 1) // 2
        _check_generator_work(max_bytes, max_products, payload_bytes=m * (32 + 8 * digits),
                              work=m * (q * digits + 1) + pairs * (digits + 1))
        masks = tuple(_pauli_masks(label) for label, _ in rows)
        if all(((x & masks[j][1]).bit_count() + (z & masks[j][0]).bit_count()) % 2 == 0
               for i, (x, z, _) in enumerate(masks) for j in range(i)):
            if vacuum and reason is None:
                buckets = {}
                for (_, value), (x, _, y) in zip(rows, masks, strict=True):
                    c = complex(value).imag
                    if c != 0.0 and x:
                        real, imag = buckets.get((abs(c), x), (0, 0))
                        sign = 1 if c > 0 else -1
                        # sign * i**y as (real, imag) parts.
                        buckets[abs(c), x] = (real + sign * (1, 0, -1, 0)[y % 4],
                                              imag + sign * (0, 1, 0, -1)[y % 4])
                if any(value != (0, 0) for value in buckets.values()):
                    reason = (f"pool generator {generator.pool_index} has commuting Pauli terms of one "
                              "coefficient magnitude whose action on the all-zero state does not cancel")
            continue
        commuting = False
        try:
            _require_default_double(generator, max_bytes=max_bytes, max_products=max_products)
        except _UnsupportedDefaultGenerator as error:
            raise ValueError(
                f"pool generator {generator.pool_index} has no exact compiler route: {error}"
            ) from error
    return commuting, reason


class CompilerPlans:
    """Generator compiler plans built when a generator is first selected, keyed by pool index.

    ``construct_adapt``, ``_native_preparation``, the optimization shift-rule
    selection and ``adapt_archive`` read this one map. An entry is built by
    ``fermionic_circuits._plan_generator_circuit`` for the checked
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


def _construction(inputs, reference, method, length, compiler_plans, slot_bound, execution, shots, acquisition):
    """Choose the one parameterized construction that every ADAPT query reuses.

    The adaptive trajectory is unknown at planning time, so the Plan fixes a
    finite query space instead of concrete circuits. Each later query is a
    point in the ``_parameters`` space, checked by ``ADAPT.validate_point``
    and bound through ``Plan.resolve``. The same construction therefore serves
    live execution, resume and archive reload without replanning.

    Classical execution uses one classical routine with experiments ``screen``,
    ``pair``, ``energy`` and, for residual stopping, ``residual``. An
    ``energy`` query returns the product energy and its gradient in the
    chain's angles from one evaluation. Its known workspace is the bounded
    set of cached state vectors and Hamiltonian actions held at one time,
    and, with optimization, the memory set aside for energy-and-gradient
    evaluations. A supplied Qiskit reference circuit adds a statevector
    construction whose cost is declared unknown.

    Quantum execution uses one circuit block per query family. ``screen`` and
    ``energy`` prepare a generator chain on the system register and read Pauli
    labels. ``pair`` adds an ancilla. With grouped shots (the default
    ``PencilAcquisition``) it prepares the pair's phase-faithful joint state
    and measures the ancilla into ``phase`` and the system into ``outcome``
    after one group's readout rotations, the classical layout of
    ``fixed_basis.reduce_group``, and the ``diagonal`` experiment measures a
    chain's system state in the same layout without measuring the idle
    ancilla. With termwise shots ``pair`` is the Hadamard test for one
    transition amplitude. Exact execution prepares the pair's phase-faithful
    joint state, or a diagonal pair's system state, reduced when it is
    computed to ``(H0_ij, S_ij)`` (``pair_reducer``), and the exact ``screen``
    query of the full product state also reduces its diagonal. The layout
    follows ``acquisition``, the Plan's ``ResolvedAcquisition``, which the
    runtime readers (``_point``, the ``_read_values`` bit offset,
    ``_matrix_entries``, ``_new_pencil_queries`` and
    ``ADAPT.validate_point``) also read. A Plan that measures nothing
    resolves no measurement and so has no grouped layout.
    ``construct_adapt`` builds the circuit for each bound point from the
    block choice, which names the resolved measurement, and the resolved
    ``pair_preparation`` that each block's payload carries.
    ``construction_work`` is a declared upper bound on the circuit
    instructions appended for the longest chain, not a bound on backend
    synthesis, CPU time or memory. ADAPT checks this bound for the generator
    family before any circuit runs. A generator's angle-independent compiler
    data is constructed when that generator is selected and reused across
    its later angles. The bound uses the system width, Pauli term count and
    supported circuit family. The appended instructions and the later cost
    of synthesizing them into backend gates are reported separately.

    A ``PencilSampling`` Plan that measures repeats every batch ``shots``
    times, the Program parameter that each query binds with its ``stage``
    (``_parameters``), so every selection, pilot and main query is a point
    of one of these experiments with its own repetitions. Every other Plan
    repeats each batch its fixed ``shots`` times, or once for exact readout.

    Returns:
        ``(construction, experiments, blocks)`` for the Plan.
    """
    q = reference.manifest.basis.dimension.bit_length() - 1
    staged = method.sampling is not None and acquisition.matrix_measurement == "grouped"
    parameters = _parameters(
        length,
        len(inputs.pool),
        len(inputs.nonidentity),
        max(len(inputs.groups), len(inputs.energy_groups)),
        residual=method.stop_criterion == "residual_norm",
        qubits=q,
        stage_shots=max(shots, method.sampling.total_shots) if staged else None,
    )
    experiments, definitions, blocks, kernels = [], [], [], []
    if execution == "classical":
        d = 1 << q
        # Declared host workspace: the classical query frontier of
        # adapt_acquisition.bind_classical (cached_state_vectors + 12
        # complex128 state vectors, 8L + 12 for the default growth),
        # the Plan's packed action tables (adapt_actions.build_action_tables),
        # the largest shared action remainder max_A(B_A - 32d) beside the
        # input and output vectors the frontier already holds,
        # and the processed Hamiltonian's payload plus the additional bytes for
        # removing its mean diagonal (inputs.offset_free_requirements[0]). This
        # additional byte charge is zero for dense or Pauli input and for a zero shift.
        # With optimization, an energy-and-gradient query with k <= L generators
        # has a vector and scalar-array allowance of 16 (C_cache + 2k + 12) d + 24k,
        # where C_cache counts the vectors already held by the Run's caches.
        # Using C_cache <= 8L and k <= L, the Plan reserves
        # 16 (10L + 12) d + 24L on top of the named non-vector terms
        # (adapt_acquisition._classical_energy_gradient).
        tables = inputs.cache["action_tables"]
        vectors = (
            16 * (10 * length + 12) * d + 24 * length
            if method.optimize_every_m is not None
            else 16 * (cached_state_vectors(method, length) + 12) * d
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
        grouped = acquisition.matrix_measurement == "grouped"
        registers = (
            Register(
                name="ancilla", width=ExprRef(expression="control_width"), role="clean_ancilla"
            ),
            Register(name="system", width=q),
        )
        # Grouped pencil settings use the classical layout of
        # fixed_basis.reduce_group: the ancilla at bit 0 (phase) and system
        # qubit k at bit k + 1 (outcome). A classical value needs a positive
        # width, so every query of a grouped Plan has the phase bit. Screen,
        # energy and diagonal queries leave it unmeasured at zero, and
        # _read_values reads their system bits from bit 1.
        classical = (
            ((ClassicalValue(name="phase", dtype="bits", width=1),) if grouped else ())
            + ((ClassicalValue(name="outcome", dtype="bits", width=ExprRef(expression="readout_width")),)
               if shots else ())
        )
        for wire in ("ancilla", "system"):
            definitions.extend(
                (
                    Definition(id=f"allocate_{wire}", node=Allocate(wire=wire)),
                    Definition(id=f"release_{wire}", node=Release(wire=wire)),
                )
            )
        if shots:
            definitions.append(Definition(id="measure_system", node=Measure(wire="system", result="outcome")))
            definitions.append(Definition(id="measure_phase" if grouped else "measure_ancilla",
                                          node=Measure(wire="ancilla", result="phase" if grouped else "outcome")))
        # Logical construction slots of one query, one per instruction that
        # construct_adapt and the circuits it uses append. This is not a
        # native synthesis, CPU or RSS bound, and the Program's terminal
        # measurements are not counted. A query has a left and a right chain
        # of at most `length` generators and two reference preparations. Each
        # generator occurrence costs its circuit, bounded by the pool's
        # family-level envelope B (`slot_bound`), and one instruction that
        # appends it. A common prefix of r generators runs once, so chains of
        # l_left and l_right generators give l_left + l_right - r <= 2 length
        # occurrences: W = 2 W_reference + 2 length (1 + B) + 8q + 7. The
        # final 8q + 7 bounds the instructions outside the generator circuits
        # for q >= 1: q + 7 for the termwise Hadamard test (two H, at most q
        # controlled Pauli gates, one S^dagger, two preparation instructions
        # and their two reference instructions), 2q + 9 for a grouped joint
        # state (at most one H and two X on the ancilla, two whole-preparation
        # instructions and their two reference instructions, at most two
        # ancilla and 2q system readout rotations, fewer for a diagonal
        # setting), and 8q + 1 for a screen or energy query (one preparation
        # and its reference instruction, an inserted Pauli rotation of at
        # most 6q - 1 gates and at most 2q readout rotations). Opaque
        # reference-preparation work stays unknown when its owner has no law,
        # and B holds only for the generator families it is derived for.
        #
        # The vacuum constructions (adapt_acquisition._append_vacuum_pair)
        # run the reference twice under control, each chain's generators
        # without control and one phase gate per vacuum phase component.
        # A generator occurrence has at most T + 1 components (its actual
        # global phase and at most T diagonal words) for T stored Pauli
        # terms, so a chain has at most length (T_max + 1), with T_max the
        # largest term count of the admitted pool. The construction of the
        # reference row has at most `length` occurrences and one chain's
        # components: W_reference_pair = 2 W_reference + length (1 + B)
        # + length (T_max + 1) + 8q + 7. The restoration has
        # l_right + 2 l_left <= 3 length occurrences, at most `length` of
        # them adjoints, and both chains' components. An adjoint
        # (qiskit_compat.inverse_realized_gate) has as many top-level
        # instructions as its forward gate and copies the definitions of
        # its Gray-code RY multiplexors, both of which B already counts, and
        # it also copies the definition of each multi-controlled RY with
        # k >= 2 controls, which B counts as one instruction. That
        # definition has at most 96 instructions, the Qiskit 2.6.0 count for
        # k = 5 that _realized_adjoint checks before copying, and a
        # generator has at most 128 such gates (at most five pair/split
        # factors, and at most sum k_b (k_b - 1)/2 <= 2 * 2**6 Givens
        # rotations of the shared-index blocks, none in the commuting and
        # four-distinct routes), so an adjoint adds at most 96 * 128 = 12288.
        # W_restore = 2 W_reference + 3 length (1 + B) + 12288 length
        # + 2 length (T_max + 1) + 8q + 7. Outside the generators and phase
        # gates both have at most one H and two X, two controlled-reference
        # instructions and 2q + 2 readout rotations, 2q + 7 <= 8q + 7.
        # "vacuum_reference" uses the reference-row construction only for
        # the pairs of the reference and the prefix construction for every
        # other pair, so one query of it is bounded by max(W_reference_pair, W).
        # Diagonal queries prepare one chain and are covered by W.
        reference_work, rule = 2 * reference.preparation.work, acquisition.pair_preparation
        construction_work = reference_work + 2 * length * (1 + slot_bound) + 8 * q + 7
        if rule in ("vacuum_reference", "vacuum_restore"):
            phases = length * (max(len(g.pauli_terms) for g in inputs.pool) + 1)
            if rule == "vacuum_reference":
                construction_work = max(construction_work,
                                        reference_work + length * (1 + slot_bound) + phases + 8 * q + 7)
            else:
                construction_work = (reference_work + 3 * length * (1 + slot_bound) + 12288 * length
                                     + 2 * phases + 8 * q + 7)
        mode = f"shots:{acquisition.matrix_measurement}" if shots else "exact"
        phase = "reference and generator physical phase kept under internal control/inverse"
        cost_context = ("selected actual prefix only; declared maximum logical construction slots, native "
                        "synthesis/CPU/RSS unknown")
        if rule in ("vacuum_reference", "vacuum_restore"):
            # The selected definitions state the scope of the vacuum rule
            # (_admit_generator_routes) beside the slot law they declare.
            phase = (f"{rule} pairs control only the reference and correct the known vacuum phase of the "
                     "uncontrolled generator chains after it by ancilla phase gates; the vacuum rule is a "
                     "structural sufficient condition on the stored terms of the generators after the "
                     "reference, without a claim about native depth or numerical error")
            cost_context = (f"{rule}: declared maximum logical construction slots "
                            + ("max(W_reference_pair, W_prefix)" if rule == "vacuum_reference"
                               else "W_restore")
                            + ", with adjoint copies and phase gates; not native depth, synthesis, CPU or RSS")
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
                phase=phase,
                # construct_adapt synthesizes this reference preparation, so
                # lowering and the Run admit its size like any preparation block.
                preparation=reference.preparation,
            )
            selected = SelectedDefinition(
                signature=signature,
                semantics=semantic,
                implementation=NATIVE,
                choice=f"{name}:{mode}:{acquisition.pair_preparation}:{inputs.hamiltonian.manifest.reference.identity}:{reference.manifest.reference.identity}:{native_identity}",
                decomposition=None,
                cost_law=None,
                construction_work=construction_work,
                cost_context=cost_context,
            )
            payload = (name, inputs, reference, compiler_plans, method, acquisition.pair_preparation)
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
                children += (["measure_phase", "measure_system"] if grouped and name == "pair" else
                             ["measure_ancilla" if name == "pair" else "measure_system"])
            children.append("release_system")
            if name == "pair":
                children.append("release_ancilla")
            bodies = [(name, children)]
            if grouped and name == "pair":
                # A diagonal setting prepares one chain on the system and
                # leaves the idle ancilla unmeasured, so its phase bit stays
                # at the initial zero that reduce_group requires, whatever the
                # device's readout error.
                bodies.append(("diagonal", ["allocate_ancilla", "allocate_system", "call_pair", "measure_system",
                                            "release_system", "release_ancilla"]))
            for experiment, children in bodies:
                definitions.append(
                    Definition(id=f"body_{experiment}", node=Sequence(children=tuple(children)))
                )
                metadata = MetadataRef(
                    format=METHOD,
                    data=InputRef(identity=f"adapt:{experiment}", representation="query law", source=METHOD),
                )
                definitions.append(
                    Definition(
                        id=f"batch_{experiment}",
                        node=MeasurementBatch(
                            body=f"body_{experiment}",
                            settings=(Setting(label=experiment, metadata=metadata),),
                            repetitions=ExprRef(expression="shots") if staged else shots or 1,
                            observation_kind="counts" if shots else (
                                "trajectory" if name in ("pair", "screen") else "pauli_expectation"),
                        ),
                    )
                )
                experiments.append(
                    Experiment(
                        name=experiment,
                        batch=f"batch_{experiment}",
                        setting_index=0,
                        readout=_pair_reduction_readout(
                            inputs.hamiltonian.reference.identity, [c for _, c in inputs.nonidentity], q,
                            diagonal=False)
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
    """Return the readout specification of the exact shared query on the full product state.

    Its ``diagonal`` point is the weighted-pencil reduction of the system
    state, ``H0_ii`` and the raw norm, when the Hamiltonian has a nonidentity
    term, and its ``screen`` point reads the active screening labels, when
    there are any. Both points share one preparation and preparation record,
    and no measurement basis change sits between them.
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
    """Return the readout specification of an exact quantum pair query, one weighted-pencil reduction at the end."""
    from .fixed_basis import pair_reduction_point

    return ReadoutDetails(positions=(pair_reduction_point(identity, coefficients, q, diagonal=diagonal),))


class BoundedRitzGrowth(Record):
    """Grow the ADAPT basis one short generator chain at a time, adding the best of a few scored extensions each round.

    Pass it as `ADAPT(basis_growth=BoundedRitzGrowth(max_chain_length=...))`.
    The basis is then an explicit list of chains instead of the reference,
    single-generator states and products of the default ADAPT-GCiM basis. A
    chain of generators `A_{g_1}, ..., A_{g_k}` of the pool is the state
    `exp(theta A_{g_k}) ... exp(theta A_{g_1})|ref>` at ADAPT's fixed angle
    `theta`, and the basis starts with the reference alone. Each round
    extends a parent chain of the basis by one generator. The parents are
    the reference and the `parent_window - 1` chains added last. A round
    visits them in turn, repeatedly, in order of basis position, starting
    after the parent whose extension the previous round examined last, and
    each visit extends a parent by its next pool generator, so each parent
    continues through the pool in pool order. A generator may occur more than
    once in a chain, also twice in a row, and every occurrence counts
    toward `max_chain_length` and the cost. A chain equal to one already in
    the basis is not considered again.

    A round examines extensions until `candidates_per_round` of them are
    new to the basis and within `basis_depth_limit`, `max_candidate_scans`
    have been examined or no parent has a generator left. For each such
    extension v it obtains the matrix elements of v with every basis state
    and with itself and computes its score when v passes
    `independence_cutoff`, or with shots the sampled eligibility rule
    below, then adds the extension with the largest score,
    the earliest examined one among equal scores. The enlarged basis is
    solved with ADAPT's `overlap_cutoff` before the next round. With b
    basis states a round therefore obtains the matrix
    elements of at most `candidates_per_round * (b + 1)` new pairs of
    states. The extensions that are not added stay in the Run's counted
    work and saved observations. No coherent Ritz state is prepared, no
    full-space residual is computed and no angle is optimized.

    The score uses the current lowest Ritz value E and Ritz state psi of
    the solved basis. Let q be the normalized part of v outside the kept
    overlap subspace, the span of the overlap directions that the solve
    keeps above `overlap_cutoff`, and `eta` its squared norm before
    normalization. With `delta = <q|A|q> - E` and `rho = |<q|A - E|psi>|`,
    the score is `(sqrt(delta**2 + 4 rho**2) - delta)/2`, the amount by
    which the lowest eigenvalue of A projected on psi and q lies below E.
    Projecting on the kept subspace together with q gives a lowest value at
    or below that eigenvalue, so the score is a lower bound on the energy
    decrease that an exact solve of the kept subspace together with q would
    give. It is not a prediction of the next round's
    energy, because the solve of the enlarged basis keeps its own subspace,
    and it does not rank extensions by the energy of that solve. A score of
    zero does not show that an extension cannot lower the energy: an
    extension coupled only to an excited Ritz state has `rho = 0`, while
    adding it can lower the lowest Ritz value. The score uses the
    Hamiltonian without its identity coefficient, so a large constant term
    cannot round it away. `ADAPTResult.history` records the examined
    extensions of each round with their reasons, scores, `eta`, `rho` and
    `delta`, and the energy decrease and kept overlap rank that each solve
    actually gave, without clipping a rise. The
    [GCiM guide](../../algorithms/gcim.md#bounded-ritz-growth) works
    through an example.

    With shots, the score divides sampled matrix elements by `sqrt(eta)`
    and `eta`, which can enlarge their sampling noise without bound, so an
    evaluated extension is scored only when it passes a sampled
    eligibility rule. The rule reads the sample-mean variances of the
    measured overlaps between the current basis states and between them
    and the extension, and takes for each measured real or imaginary part
    the larger of its variance and `1/(N + 1)**2` for its N shots, so that
    a part whose outcomes were all equal still counts as noisy. With the
    fixed multiplier k = 3 it forms a first-order spread `sigma` of the
    estimated `eta`, the ratio `epsilon` of the noise scale of the current
    overlap matrix to its smallest kept eigenvalue, and a second-order
    term. An extension passes the rule when `epsilon < 1/2` and the margin,
    the estimated `eta` minus `k sigma` and that term, exceeds
    `independence_cutoff`. The rule is empirical and conditional on
    independently measured settings. It is neither a confidence interval
    nor a proof that the extension is independent, and an extension that
    fails it has not been shown to depend on the basis. It uses only the
    measured data and adds no shot. It does not control the noise of the
    Hamiltonian elements or of the current Ritz pair, so the score of an
    extension that passes is a raw estimate, not a validated energy
    decrease, and the round ranks these extensions by it as with exact
    elements. More basis states and a smaller kept overlap eigenvalue
    enlarge `epsilon`. As a rough planning estimate, not a recommendation,
    overlap variances near 1/N per part need N above
    `36(b - 1)/lambda**2` shots per setting for `epsilon < 1/2` with b
    basis states, real parts only and smallest kept eigenvalue lambda, and
    `72(b - 1)/lambda**2` with both parts, for example 540 for b = 16,
    lambda = 1 and real parts only. Each
    extension's margin still decides whether it is scored. Sampled bounded
    growth reads these variances from the overlap settings of grouped
    measurement and is implemented for that measurement only, so planning
    raises `ValueError` for shots with a requested
    `matrix_measurement="termwise"`.

    These operational limits end the growth, given in
    `ADAPTResult.stop_reason`:

    - `max_basis_size`, or `max_iterations` once that many chains have been
      added.
    - `flat_counter`: consecutive energy changes below
      `energy_change_tolerance`, as in the default growth, with the pool
      size P in place of the number of unselected generators.
    - `basis_depth_limit`: the round examined at least one extension that
      is not already in the basis, and every such extension exceeds
      `basis_depth_limit` or has no CX count.
    - `parent_window_exhausted`: no parent has a generator left, and every
      extension the round examined was already in the basis.
    - `candidate_scans_exhausted`: the round examined `max_candidate_scans`
      extensions, all already in the basis.
    - `no_admissible_candidate`: without shots, no evaluated extension
      passed `independence_cutoff` with a finite score. With shots, at least
      one evaluated extension passed the sampled eligibility rule and none
      of these has a finite score.
    - `noise_inconclusive`: with shots, no evaluated extension passed the
      sampled eligibility rule. The basis stays as it is, the result
      reports its last solve, and no shot is added.

    When a round finds no extension to evaluate, `basis_depth_limit` takes
    precedence over `parent_window_exhausted`, and that over
    `candidate_scans_exhausted`. Each reason describes only the extensions
    that were examined. None of them shows that the energy has converged or
    reached a requested accuracy. Without shots, a solve that keeps no
    overlap direction also stops the run with `no_usable_overlap_subspace`.
    With shots, a solve that the eigensolver rejects, including one that
    keeps no overlap direction, stops it with `invalid_sampled_evidence`.
    Execution states of `ADAPTResult.stop_reason` such as `pending_acquisition`,
    `cancelled` and `native_depth_refused` remain possible.

    Attributes:
        parent_window: Default `4`, positive. Number of parent chains of a
            round: the reference and at most `parent_window - 1` chains
            added last. An engineering choice that bounds the work of a
            round, not a convergence parameter.
        candidates_per_round: Default `4`, positive. Largest number of
            extensions whose matrix elements one round obtains and scores.
            With 1 and without shots, a round whose only evaluated extension
            does not pass `independence_cutoff` with a finite score ends the
            growth with `no_admissible_candidate`. With shots, a round whose
            only evaluated extension fails the sampled eligibility rule ends
            the growth with `noise_inconclusive`, and a round whose only
            evaluated extension passes the rule without a finite score ends
            the growth with `no_admissible_candidate`. The GCiM guide describes an adjacent
            repeat of a generator with `A**2 = -I`, for which this can
            happen.
        max_candidate_scans: Default `256`, positive. Largest number of
            extensions one round examines, including repeated chains and
            extensions excluded by `basis_depth_limit`, which obtain no
            matrix element. It limits this inexpensive examination, not the
            number of scored extensions.
        max_chain_length: Required, positive. Largest number of generator
            occurrences in one chain, repeats included. It is not a circuit
            depth.
        basis_depth_limit: Default `None`, which sets no limit and adds no
            work. A nonnegative integer D admits only an extension whose CX
            count bound is at most D, with D itself admitted. The bound is
            the sum of the CX counts of the reference preparation and of
            every generator occurrence of the chain, each lowered to
            one-qubit gates and CX before routing, with no cancellation, so
            it bounds the CX depth of the chain's preparation. An occupation
            or product reference counts 0 and every direct vector on q
            qubits `2(2**q - 2)`, a computational basis state given as a
            vector included, without reading its amplitudes, so a direct
            reference can be refused although its actual circuit is
            shallower. Planning raises `ValueError` for a supplied or
            matrix-product-state reference, whose count is unknown, for a
            reference above D, and for classical execution. The counts are
            read from the generator circuits that quantum execution builds,
            so the limit is supported for quantum execution only, and
            classical bounded growth without a limit is unaffected. A
            generator circuit is built and its instructions are read when
            one of its extensions is first examined. An extension whose
            generator circuit contains an instruction without a known CX
            count is excluded, as is one above D. The limit changes the
            trial space. It is not the native two-qubit depth of the
            executed circuits, whose pair constructions, readout and routing
            add depth, and which `ADAPT.max_native_two_qubit_depth` checks.
        independence_cutoff: Default `None`, which uses ADAPT's
            `overlap_cutoff` and then requires it to be below 1. Otherwise a
            value in (0, 1). An extension is scored only when `eta`, the
            squared norm of its part outside the kept overlap subspace,
            exceeds it. On its own this floor is no guard against sampling
            noise, so with shots the margin of the sampled eligibility rule
            must exceed it instead.
    """

    parent_window: PositiveInt = 4
    candidates_per_round: PositiveInt = 4
    max_candidate_scans: PositiveInt = 256
    max_chain_length: PositiveInt
    basis_depth_limit: Count | None = None
    independence_cutoff: Annotated[Real, Field(gt=0, lt=1)] | None = None


def bounded_basis_size(method):
    """Return the largest basis of a bounded-growth ADAPT run: the reference and one chain per round, at most ``max_basis_size``."""
    return min(method.max_basis_size, method.max_iterations + 1)


def _admit_final_stages(method, b, num_qubits, groups, terms, quadratures):
    """Check the final pilot and main stages of a ``PencilSampling`` Plan on its largest basis before any query.

    ``b`` is the largest basis that the Plan can reach, ``2*min(max_iterations,
    pool size)`` states of the default growth or ``bounded_basis_size``,
    ``groups`` the resolved Hamiltonian groups, ``terms`` the number L of
    nonidentity terms and ``quadratures`` the resolved quadratures. The
    m = ``sampled_setting_count(b, G, quadratures)`` settings of that basis
    bound those of every basis a run can end with, because
    m(b + 1) - m(b) = G + Q b max(1, G) >= 0 with Q measured quadratures, so
    B = ``total_shots`` >= 4m gives every final setting two pilot and two
    main shots. A run that stops earlier may need less. The grouped pass of
    one stage of at most B shots on that basis is charged with the
    stage-budget form of ``grouped_pass_requirements``: one ``(L_g, H)``
    descriptor per group with the K = ``sampled_setting_count(b, 1, Q)``
    settings of each group as copies and P = b(b+1)/2 pairs, or for a
    pencil without nonidentity terms one ``(0, H)`` descriptor with the m
    overlap settings as copies and P = b(b-1)/2, H = min(B, 2**(n+1)) and the
    oriented view. H does not shrink with the settings of an early stop, so
    every table of a smaller final basis stays inside the bound. The bytes
    are compared with ``max_bytes`` and the work with ``max_products``.
    """
    budget = method.sampling.total_shots
    m = sampled_setting_count(b, len(groups), quadratures)
    reduce = ("max_iterations or the pool size, which set it as 2*min(max_iterations, pool size)"
              if method.basis_growth is None
              else "max_iterations or max_basis_size, which set it as min(max_basis_size, max_iterations + 1)")
    if budget < 4 * m:
        raise ValueError(
            f"ADAPT.sampling needs PencilSampling.total_shots of at least 4m = {4 * m} shots, two pilot and two "
            f"main shots for each of the m = {m} grouped settings of the largest basis this Plan can reach, {b} "
            f"states, but it is {budget}. A run that stops with a smaller basis may need fewer shots. Raise "
            f"PencilSampling.total_shots, or reduce the largest basis through {reduce}.")
    bins = min(budget, 1 << (num_qubits + 1))
    chunks, copies, pairs = (
        ([(len(group), bins) for group in groups], sampled_setting_count(b, 1, quadratures), b * (b + 1) // 2)
        if groups else ([(0, bins)], m, b * (b - 1) // 2))
    size, work = grouped_pass_requirements(num_qubits, chunks, terms, b, settings=m, pairs=pairs, oriented=True,
                                           copies=copies, shot_total=budget)
    for field, need, limit in (("max_bytes", size, method.max_bytes), ("max_products", work, method.max_products)):
        if need > limit:
            unit = "data bytes" if field == "max_bytes" else "work units"
            raise ValueError(
                f"The grouped analysis of one PencilSampling stage of at most {budget} shots on the largest basis "
                f"this ADAPT Plan can reach, {b} states, has a planning bound of {need} {unit}, above "
                f"{field}={limit}. This bound is conservative and can refuse a run whose final basis and "
                f"allocation would fit. Raise ADAPT.{field} to at least {need}, or reduce "
                f"PencilSampling.total_shots, the number of grouped Pauli terms or the largest basis through "
                f"{reduce}.")


def independence_floor(method):
    """Return the exact independence floor of bounded growth: ``independence_cutoff``, or ADAPT's ``overlap_cutoff`` when it is None."""
    cutoff = method.basis_growth.independence_cutoff
    return method.overlap_cutoff if cutoff is None else cutoff


def cached_state_vectors(method, length):
    """Return the most state vectors that the state cache and the ``(H - cI)`` column cache of a classical ADAPT Run hold together.

    ``adapt_acquisition._keep_current`` keeps the prefixes of the current
    basis chains and of the two chains of the active query, and both caches
    are keyed by these prefixes. A chain of at most L = ``length``
    generators has at most L nonempty prefixes, so the two query chains add
    at most 2L to each cache. The default basis of k <= L selections has 2k
    prefixes, including the reference, so each cache holds at most 4L
    vectors. Every prefix of a bounded-growth chain is a chain of the
    basis, because each added chain extends a basis chain by one generator,
    so a bounded basis of at most B states (``bounded_basis_size``) has B
    prefixes, and each cache holds at most B + 2L vectors. The count is of
    live cache entries, not a bound on the Run's whole memory, its results
    or its archives.
    """
    if method.basis_growth is None:
        return 8 * length
    return 2 * (bounded_basis_size(method) + 2 * length)


class ADAPT(Method):
    """Adaptive GCiM (ADAPT-GCiM) method for the smallest eigenvalue of an `Eigenproblem`.

    It grows its trial basis from a pool of generators. Build it with
    keyword arguments and pass it as `method=`, for example
    `solve(Eigenproblem(A=H), method=ADAPT(initial_state=psi, pool=generators))`.
    `initial_state` and `pool` are required. The result is an
    [`ADAPTResult`][nwqlib.algorithms.gcim.adapt_records.ADAPTResult] whose
    `eigenvalue` is the lowest projected Ritz value in the trial basis the run
    built.

    The method follows ADAPT-GCiM of Zheng et al., npj Quantum Inf. 10, 127
    (2024), arXiv:2312.07691v3. Each round screens the current product state
    with the ADAPT gradient `<[H, A_i]>` of every unselected generator (main
    text, "Integration of GCiM with ADAPT approach", p. 4, and Grimsley et
    al. (2019), arXiv:1812.11173v2, Sec. II.B, steps 5-7). The selected
    generator enters at the fixed angle `theta`, and the basis grows by two
    states per selection (METHODS, p. 9). A selected generator is not
    screened again, following the paper's QuGCM `adapt_gcim.py`
    implementation (commit c5efcb0, lines 303-398), whereas Grimsley et al.
    step 7 permits reuse in its ADAPT-VQE ansatz. Optimization every
    `optimize_every_m` selections is the intermittent, truncated optimization
    of Appendix H, and its extra energy evaluations run through the same Run,
    the saved execution of the Plan.
    Because the circuits of each round depend on generators selected from
    earlier results, `prepare(plan, settings="all")` is rejected.

    The default flat-counter stop follows METHODS, p. 9, and stops after
    `T = min(T_auto, T_usr)` consecutive energy changes below
    `energy_change_tolerance`, with `T_auto` equal to `t_auto_fraction`
    (20%) of the unselected generators and `T_usr = t_user`. Counting starts
    at the first change from the reference energy. T is recomputed after each
    selection from the current number of unselected generators, and the run
    stops as soon as the integer count is at least `ceil(T)`. QuGCM's code
    instead uses a max rule with 10% and a warmup. With `max_iterations=8` and `t_user=10`, the
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

    For a basis of b states, ADAPT by default measures the projected
    matrices of the current basis with the same joint-state construction
    and count reduction as `FixedGCIM` for the same ordered states, and
    each round measures only the pairs of generator chains that no earlier
    round measured. With shots the default `acquisition` measures
    grouped joint states, `b**2 G` settings for G qubit-wise commuting
    groups of Hamiltonian terms, or `b(b+1)G/2` when the processed
    Hamiltonian, the reference and every pool generator are proved real.
    One Hadamard test per Pauli term, the comparison construction, is
    `PencilAcquisition(matrix_measurement="termwise", pair_preparation="controlled")`.
    `PencilAcquisition(pair_preparation="vacuum_reference")` and
    `"vacuum_restore"` prepare the same joint states as the default by
    controlling only the reference preparation, for pools whose generators
    planning can show, from their Pauli terms, to map the all-zero state to
    itself up to a phase (`acquisition` below).

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
            basis holds at most ``2*min(max_iterations, pool size)`` states, so
            `max_basis_size=64` allows at most 32 selections for a pool of
            more than 32 members. The default limits the cost. With exact
            evaluation and the spin-adapted pool, the H4 chain of the eigenvalue example reaches
            FCI at the eighth selection, and 10-qubit LiH is then 0.46 mHa
            above its FCI energy. With `max_iterations=32`, the largest that
            `max_basis_size=64` allows for these pools, the flat counter stops
            them at selections 18 and 19
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
        optimize_every_m: Default `None`, no optimization. Optimize the
            generator angles of the product state every m selected generators.
        optimize_rounds_n: Default `None`. Largest number of BFGS iterations
            per optimization, required together with the other two
            optimization settings.
        optimize_max_evaluations: Default `None`. Largest number of energy
            evaluations when optimization is enabled, counted as distinct
            evaluated parameter points. A classical point is one energy-and-gradient
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
            classical query (one evaluation of gradients, matrix elements or
            an energy) starts, its reference, generator, Hamiltonian and
            scalar work are summed and checked, and each later query is
            checked afresh. Packed Pauli tables are checked once per Plan.
            The recorded allowance covers a whole query, while the counters
            record the actions performed after cache reuse.
        input_conversion: Default `"auto"`, which keeps Pauli input as it
            is and converts an explicit dense matrix of original dimension at
            most 16 to Pauli form. This applies to the Hamiltonian of a
            quantum run and to custom pool generators that need conversion. A
            classical run keeps a dense or sparse Hamiltonian in its form,
            padded when needed. `"dense_pauli"` also permits the conversion of
            sparse matrices and of larger dense matrices, within
            `max_conversion_work` and `max_bytes`.
        max_conversion_work: Default `100_000_000`. Separate limit on the
            work of the chosen input conversion.
        pauli_coefficient_cutoff: Default `1e-14`, nonnegative. Operator
            coefficients at or below it in magnitude are removed, and the
            sum of the removed magnitudes is recorded.
        input_symmetry_tolerance: Default `1e-12`, nonnegative. Largest
            relative Frobenius-norm correction that the anti-Hermitian
            projection of each pool generator may make. The Hamiltonian has
            no tolerance and must be exactly Hermitian.
        acquisition: Default `PencilAcquisition()`, grouped joint-state
            measurement with common-prefix preparation and automatic
            quadratures
            ([`PencilAcquisition`][nwqlib.algorithms.gcim.fixed_basis.PencilAcquisition]).
            Planning decides real quadratures from a structural proof over
            the processed Hamiltonian, the reference and every admitted pool
            generator, not only the generators a run happens to select, and
            records the resolved choices in `plan.reconstruction.acquisition`.
            The processed Hamiltonian is the one measured, the operator after
            `pauli_coefficient_cutoff`, whose removed coefficient magnitudes
            sum to `plan.reconstruction.hamiltonian_dropped_l1`. The proof
            says nothing about the imaginary parts of matrix elements of the
            original operator. With a scalar processed Hamiltonian the
            default growth measures nothing, and `quadratures` then has no
            effect. Bounded Ritz growth on such a Hamiltonian still measures
            overlaps whenever its basis can hold more than the reference, so
            its request is resolved as for any measured Plan.
            `pair_preparation="vacuum_reference"` and `"vacuum_restore"`
            control only the reference preparation B and run the generator
            chains without control. They need the exponential of every pool
            generator, the whole pool and not only the generators a run
            selects, to map the all-zero state `|vac>` to itself up to a
            phase, which quantum planning decides from the generators' Pauli
            terms by the following vacuum rule. A generator whose terms
            commute qualifies when, for each coefficient magnitude and each
            nonempty set of X/Y positions, the terms with that magnitude and
            those positions cancel on `|vac>`, as in `i c (XX + YY)`, while
            `i c X` and `i (XX + 0.5 YY)` do not. Its phase comes from its
            identity term and its terms of only I and Z letters. A double
            excitation whose terms do not commute, which planning accepts
            only in the form the built-in pools build, qualifies with
            phase 0. The rule concerns the generator chain after B, not B
            itself, and is sufficient rather than necessary. With a failing
            generator, planning raises `ApplicabilityError` naming the first
            one by its pool index. Both values apply when joint
            states are prepared, exactly or with grouped shots, and
            `ADAPTResult.pair_preparations` records the construction of each
            pair. For classical execution and for a Plan that measures
            nothing, both values are inactive, so the vacuum rule is not
            required and the request is kept. Before a query runs, the
            vacuum phase is read from its generator circuits. They are
            checked for finite angles and global phases, for rotation angles
            of I/Z terms whose halves are binary64 numbers and, under
            `"vacuum_restore"`, for definitions from which the adjoints can
            be formed, and a failing circuit raises `ApplicabilityError`
            before that query is measured. The
            [GCiM guide](../../algorithms/gcim.md#how-matrix-elements-are-obtained)
            describes both constructions and their costs.
        basis_growth: Default `None`, the gradient screening and basis of
            ADAPT-GCiM described above. A
            [`BoundedRitzGrowth`][nwqlib.algorithms.gcim.adapt.BoundedRitzGrowth]
            instead grows an explicit list of short generator chains, one
            scored extension per round, at the fixed angle `theta`. Each of
            its added chains counts toward `max_iterations`, and its basis stops
            growing at `max_basis_size` states. It screens no gradients, so
            `gradient_norm_floor` has no effect and the commutators
            `[H, A_j]` are not formed. `ADAPT` raises `ValueError` for it
            together with `optimize_every_m` or
            `stop_criterion="residual_norm"`. With shots it supports the
            default grouped measurement, and planning raises `ValueError`
            when `acquisition` requests `matrix_measurement="termwise"`.
            With a scalar processed Hamiltonian and `max_basis_size>1`,
            quantum planning still resolves the measurement choices because
            evaluating extensions can need overlaps. With `max_basis_size=1`
            the Plan measures nothing and keeps the request unused.
        max_native_two_qubit_depth: Default `None`, which requests no depth
            limit. `None` is neither a depth of 0 nor evidence that the
            circuits suit a device. A nonnegative integer D requires each
            submitted quantum circuit, the pencil, screening and energy
            circuits alike, to have a native two-qubit depth of at most D,
            with D itself allowed. This depth is that of the circuit the
            backend prepared for execution, with the reference, the
            generator exponentials, their controlled forms, the measurement
            basis changes and the backend's qubit routing, computed once when
            the circuit is prepared
            ([`PreparedArtifact.native_depth`][nwqlib.execution.PreparedArtifact]).
            Each circuit is checked just before its submission, and one whose
            depth exceeds D or is unavailable is not submitted. The run then
            ends with `stop_reason` `"native_depth_refused"`, and `pending`
            holds the refusal, which names the circuit's experiment and
            preparation record, the limit and the depth or why it is
            unavailable. The Result reports the last pencil solved before the
            refusal, and the circuits submitted earlier keep their counts and
            shots in the Run. Of the current backends only IBM Runtime
            computes this depth, for its compiled circuits, and only when its
            options keep the provider's dynamical decoupling and twirling off
            and qiskit-ibm-runtime 0.49.0 is installed
            (`nwqlib.backends.IBMRuntimeBackend`). Otherwise the depth is
            unavailable and an integer limit refuses the first circuit, as it
            does on every other backend, Aer included. The limit also refuses
            a circuit whose preparation recorded a version of nwqlib, Qiskit
            or qiskit-ibm-runtime other than the installed one, as in a Run
            reopened in another environment. That refusal depends on the
            installed packages, not on the circuit, so it does not end the
            run: resuming the run raises
            `nwqlib.backends.inspection.NativeDepthCapRefusal` with its
            `provenance` set, and the Run keeps its iteration state and the
            prepared circuit. Resumed where the recorded versions are
            installed, the Run checks that circuit's depth against D again
            and, when the depth meets the limit, submits it and continues. In
            the installed environment, a new Run prepares its circuits again,
            and their preparation records then decide the limit. Execution
            without the limit is another option.
            The limit changes no circuit, generator or Hamiltonian term. No
            value suits every device, so choose D for the device and the
            circuits, for example from the depth recorded for the first query
            prepared by `nwqlib.prepare` (`prepared.run.prepared_artifacts`).
        sampling: Default `None`, with which a run with shots takes
            `eigenvalue` and `pencil` from the same measurements that chose
            the basis. A
            [`PencilSampling`][nwqlib.algorithms.gcim.fixed_basis.PencilSampling]
            measures the final basis again once its growth has stopped: a
            pilot stage with uniform shots, one decision of the shots of
            every setting from the pilot pencil, and a main stage whose
            measurements alone form `eigenvalue`, `pencil` and the group
            moments of the Result. The stages follow the rules of
            `PencilSampling` for the grouped settings of the final basis,
            and `ADAPTResult.shot_allocation` records the decision. `shots=`
            still sets the shots of every query that grows the basis, and
            `total_shots` is a separate budget of the two final stages. When
            the final basis has settings to measure and both stages
            complete, the Run spends `total_shots` in addition to the shots
            of the growth. A Run that ends during the stages counts the
            shots already submitted, up to `total_shots` stage shots.
            A final basis without settings adds no shots. The
            main measurements are new, but the basis was chosen from the
            earlier ones, so the estimate is conditional on that basis.
            `sampling` supports quantum execution with a positive `shots=`
            and the default grouped measurement. Planning raises
            `ValueError` for classical execution in every case. When the
            Plan measures anything, it also raises it for `shots=None` and
            for `matrix_measurement="termwise"`. A quantum Plan that
            measures nothing, such as the default growth for a multiple of
            the identity, accepts `sampling` with `shots=None` or termwise
            measurement, runs no stage and records zero pilot and main
            shots. Before the first query, planning requires at least four
            of the `total_shots` for every setting of the largest basis the
            Plan can reach, `2*min(max_iterations, pool size)` states, or
            `min(max_basis_size, max_iterations + 1)` with `basis_growth`,
            so a run that stops with a smaller basis may have needed fewer
            shots. It also checks the analysis of one stage of `total_shots`
            shots on that basis against `max_bytes` and `max_products`.
            `nwqlib.estimate(plan)` still reports the totals of the whole
            Plan as unavailable, and one prepared query is estimated
            normally. The stages follow every scientific stop of the
            growth, `invalid_sampled_evidence` included, and the main stage
            can still end without an eigenvalue, as when its solve fails. A
            circuit that `max_native_two_qubit_depth` refuses, or a
            cancellation, ends the Run without the remaining stages. The
            Result then reports the last pencil that the growth solved, if
            there is one, together with the decision if it was made. The Run
            counts every shot already submitted, those of an unfinished
            stage included. If no projected analysis has started, as when a
            circuit needed for the first pencil is refused, `eigenvalue` is None and
            `analysis_attempts` is 0. A cancellation during the stages
            returns at once a Result built from the measurements that the
            Run had collected, without waiting for a submitted stage job.
            For such a job, data that the Run has retrieved and saved can be
            read through `Run.completed_observation`, including data
            obtained by a later explicit refresh. These data are not added
            to that Result. Once the Result
            exists, `run.resume()` returns without asking the backend
            about its jobs again, so a job that completes later enters
            neither the Run nor the Result by itself.

    Raises:
        ValueError: If `theta` is zero, if only some of the three
            optimization settings are given, if an explicit pool is empty
            or has more than `max_pool_size` members, or if `basis_growth`
            is combined with optimization or `stop_criterion="residual_norm"`,
            or has no `independence_cutoff` while `overlap_cutoff` is at
            least 1.

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
    acquisition: PencilAcquisition = PencilAcquisition()
    basis_growth: BoundedRitzGrowth | None = None
    max_native_two_qubit_depth: Count | None = None
    sampling: PencilSampling | None = None
    result_type: ClassVar[type] = ADAPTResult

    @field_validator("pool", mode="before")
    @classmethod
    def _pool(cls, value, info):
        """Copy an explicit pool, or pass a built-in pool name through.

        The member count is checked against ``max_pool_size`` before any
        member is copied. Each ``FermionicGenerator`` is checked and copied by
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
        """Reject a zero angle, a partial set of optimization controls and bounded growth with an optimizer or a residual stop.

        At ``theta = 0`` every generator exponential ``exp(theta A)`` is the
        identity, so each new basis state would repeat the reference.
        Optimization needs its period, BFGS iteration limit and total
        evaluation limit together. ``BoundedRitzGrowth`` keeps every angle
        at ``theta``, so that the effect of the fixed-angle basis can be read
        on its own, and it computes no full-space residual. Its exact
        independence floor must lie in (0, 1).
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
        growth = self.basis_growth
        if growth is not None:
            if enabled:
                raise ValueError("BoundedRitzGrowth keeps every angle at theta and cannot be combined with "
                                 "optimize_every_m, optimize_rounds_n and optimize_max_evaluations")
            if self.stop_criterion == "residual_norm":
                raise ValueError("BoundedRitzGrowth computes no full-space residual; "
                                 "use stop_criterion='flat_counter'")
            if growth.independence_cutoff is None and not self.overlap_cutoff < 1:
                raise ValueError(f"BoundedRitzGrowth needs an independence_cutoff in (0, 1), because "
                                 f"overlap_cutoff={self.overlap_cutoff} is not below 1")
        return self

    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR

    def plan(self, problem, *, output, execution, shots, rng):
        """Check the generator pool and choose the gradient, matrix-element and energy queries.

        After k selections the basis holds 2k states, so
        `2*min(pool size, max_iterations)` bounds the projected dimension,
        which must not exceed `max_basis_size`. For an explicit pool this
        bound is checked before conversion and commutator formation. A named
        pool is checked after it is enumerated, under that enumeration's own
        byte and product limits. A `BoundedRitzGrowth` basis holds at most
        `min(max_basis_size, max_iterations + 1)` states, because each round
        adds one chain or ends the run with one of its stop reasons, and its
        run stops when it reaches `max_basis_size`.

        Raises:
            ValueError: If `2*min(pool size, max_iterations)` exceeds
                `max_basis_size` without `basis_growth`, if classical
                execution is given shots, if `stop_criterion="residual_norm"`
                is used without classical execution, or if `basis_growth`
                is used with shots and a requested
                `matrix_measurement="termwise"`, or sets a
                `basis_depth_limit` while the reference's CX count bound
                exceeds it, the reference has no CX count or execution is
                classical. Also if `sampling` is used with classical
                execution, or, when the Plan measures anything, with
                `shots=None` or a requested `matrix_measurement="termwise"`,
                with fewer than four of its `total_shots` for each setting
                of the largest reachable basis, or with an analysis of one
                stage on that basis above `max_bytes` or `max_products`.
            ApplicabilityError: If the Problem names a fixed subspace, which
                the changing basis of ADAPT does not implement.
        """
        growth = self.basis_growth
        if growth is not None and shots is not None and self.acquisition.matrix_measurement == "termwise":
            # The supported sampled route of bounded growth reads its
            # eligibility evidence from grouped overlap moments.
            raise ValueError("BoundedRitzGrowth with shots supports grouped measurement, whose overlap moments "
                             "decide which extensions are scored; use the default "
                             "PencilAcquisition(matrix_measurement='grouped'), or basis_growth=None for a "
                             "termwise comparison")
        if growth is not None and growth.basis_depth_limit is not None and execution == "classical":
            raise ValueError("BoundedRitzGrowth(basis_depth_limit=...) reads the CX counts of the generator "
                             "circuits that quantum execution builds; classical execution builds none")
        if (
            growth is None
            and isinstance(self.pool, tuple)
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
        if execution == "classical" and self.sampling is not None:
            raise ValueError("ADAPT.sampling measures the final pencil with shots on a quantum register; "
                             "use execution='quantum'")
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
        if growth is not None and growth.basis_depth_limit is not None:
            # The CX count bound of every chain starts from the reference's
            # law, read from its stored preparation record without a vector scan.
            from .bounded_growth import reference_cx_bound

            bound, reason = reference_cx_bound(reference.preparation, reference.basis.dimension.bit_length() - 1)
            if bound is None:
                raise ValueError(f"BoundedRitzGrowth(basis_depth_limit=...) needs a CX count bound of the "
                                 f"reference: {reason}")
            if bound > growth.basis_depth_limit:
                raise ValueError(f"the reference preparation has a CX count bound of {bound}, above "
                                 f"BoundedRitzGrowth(basis_depth_limit={growth.basis_depth_limit})")
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
        if growth is None:
            length = min(self.max_iterations, len(inputs.pool))
            size = 2 * length
            if size > self.max_basis_size:
                raise ValueError(
                    f"maximum selected ADAPT basis holds {size} states "
                    f"(2 * min(pool size {len(inputs.pool)}, max_iterations={self.max_iterations})), above "
                    f"ADAPT(max_basis_size={self.max_basis_size})")
        else:
            length, size = growth.max_chain_length, bounded_basis_size(self)
        # After `length` selections the legacy basis has 2 * length states
        # (adapt_acquisition.basis_chains). A bounded basis of chains of at
        # most `length` generators grows by one chain per round up to
        # bounded_basis_size. That is the largest basis the controller can
        # reach, and a bounded candidate's assembled matrices enlarge a
        # smaller basis by one chain. The projected solve must fit both
        # limits that adapt_acquisition._matrix_from_observations checks
        # before each analysis. A run that might stop before reaching that
        # basis is still refused here, so no admitted run fails these two
        # checks at a late analysis.
        _check_bytes(_projected_solve_bytes(size), self.max_bytes, "ADAPT projected solve")
        _check_products(_projected_solve_work(size), self.max_products)
        commuting, vacuum_failure = True, None
        if execution == "quantum":
            # Only the route eligibility is checked here. Each generator's
            # compiler plan is built when it is first selected. A pool whose
            # generators all commute uses the commuting route's 6q max(1, T).
            # The same pass decides the structural vacuum rule, which the
            # resolution below reads when a vacuum pair construction is
            # requested and used.
            commuting, vacuum_failure = _admit_generator_routes(
                inputs.pool, max_bytes=self.max_bytes, max_products=self.max_products,
                vacuum=self.acquisition.pair_preparation in ("vacuum_reference", "vacuum_restore"))
        # One proof covers every basis this Plan can build: the processed
        # Hamiltonian, the reference and the whole admitted pool, scanned
        # once from their stored records. The processed Hamiltonian is the
        # measured target. The recorded pauli_coefficient_cutoff truncation
        # (hamiltonian_dropped_l1) defines it, so the proof says nothing
        # about imaginary entries of the original operator.
        # A scalar processed Hamiltonian makes the legacy growth of
        # adapt_acquisition.basis_chains measure nothing: it has no
        # nonidentity energy settings, its commutators give no screening
        # labels, so the controller stops before selecting a generator, and
        # the reference diagonal has no settings. The quadratures choice is
        # then unused, as in FixedGCIM. The bounded controller obtains the
        # overlap row of each candidate against the reference in its first
        # round whenever its basis can grow beyond the reference, also for a
        # scalar Hamiltonian, so that Plan acquires.
        acquires = bool(inputs.nonidentity) or growth is not None and bounded_basis_size(self) > 1
        if self.sampling is not None and acquires:
            # The supported route of PencilSampling, whose final stages
            # choose the shots of grouped counts settings, with a selection
            # that measures counts too. A Plan that measures nothing accepts
            # the request without these conditions.
            if shots is None:
                raise ValueError("ADAPT.sampling measures the final pencil of a basis chosen with shots; give a "
                                 "positive shots= for the queries that grow the basis")
            if self.acquisition.matrix_measurement == "termwise":
                raise ValueError("ADAPT.sampling chooses the shots of grouped measurement settings; use the default "
                                 "PencilAcquisition(matrix_measurement='grouped')")
        acquisition = resolve_acquisition(
            self.acquisition, execution=execution, shots=shots, acquires=acquires,
            real_reason=lambda: real_hamiltonian_reason((t.label, t.coefficient) for t in inputs.terms)
            or real_state_reason(reference) or next(filter(None, map(real_generator_reason, inputs.pool)), None),
            scope=f"processed Hamiltonian terms after pauli_coefficient_cutoff, the reference and all "
                  f"{len(inputs.pool)} admitted pool generators",
            vacuum_reason=lambda: vacuum_failure)
        if self.sampling is not None and acquisition.matrix_measurement is not None:
            # `size` is the largest basis this Plan can reach, the bound of
            # both growth rules above.
            _admit_final_stages(self, size, reference.basis.dimension.bit_length() - 1, inputs.energy_groups,
                                len(inputs.nonidentity), acquisition.quadratures)
        if (acquisition.pair_preparation in ("controlled", "vacuum_reference", "vacuum_restore")
                and reference.preparation.implementation == "qiskit.supplied"):
            # Under the resolved "controlled" rule each Hadamard-test and
            # joint-state query controls the reference preparation
            # (adapt_acquisition._native_preparation), and both vacuum
            # constructions control it twice (adapt_acquisition._append_vacuum_pair).
            # The Run's method context builds that controlled gate once.
            # qiskit_compat.controlled synthesizes each distinct dense
            # unitary of a supplied circuit exactly, and Qiskit then
            # controls every occurrence of the synthesized gates. Every
            # chain starts from the reference, so under "prefix" it is the
            # common prefix of every pair and is never controlled.
            # Classical execution and a Plan that measures nothing resolve
            # no rule and control nothing. The vacuum rule concerns the
            # generators after the reference and says nothing about the
            # reference's own synthesis, so this charge stays as it is.
            from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses
            from nwqlib.subroutines.qiskit_compat import dense_control_counts, dense_synthesis_widths
            admit_dense_syntheses(dense_synthesis_widths(reference._native),
                                  controls=(dense_control_counts(reference._native, 1),),
                                  max_work=self.max_products, max_bytes=self.max_bytes,
                                  operation="ADAPT controlled reference synthesis")
        compiler_plans, slot_bound = (), 0
        if execution == "quantum":
            q = reference.basis.dimension.bit_length() - 1
            counts = tuple(len(g.pauli_terms) for g in inputs.pool)
            slot_bound = (
                6 * q * max(1, max(counts)) if commuting else generator_pool_slot_bound(q, counts)
            )
            compiler_plans = CompilerPlans(
                inputs.pool, max_bytes=self.max_bytes, max_products=self.max_products
            )
        construction, experiments, blocks = _construction(
            inputs, reference, self, length, compiler_plans, slot_bound, execution, shots, acquisition
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
            acquisition=acquisition,
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
        """Check an exact pair reduction's private workspace and return the classical-work allowance of its query.

        The workspace check rejects a point whose ``pair_reducer.pair_bytes``
        exceeds ``max_bytes``. Before the first query of a newly entered exact
        quantum matrix stage, the ADAPT iteration checks a lower bound on the
        registered work of the stage's new pair reductions against
        ``max_products`` (``adapt_acquisition._new_pencil_queries``). A resumed
        matrix stage skips this stage check. Each query receives the whole
        ``max_products`` as its allowance. The work record is read, never
        changed.
        """
        import json
        from .pair_reducer import pair_bytes

        for item in observation.positions:
            if item.kind == "reduction":
                _check_bytes(pair_bytes(json.loads(item.parameters)), self.max_bytes,
                             "ADAPT pair reduction workspace")
        return int(self.max_products)

    def before_submit(self, plan, prepared, *, run):
        """Refuse a circuit whose native two-qubit depth exceeds `max_native_two_qubit_depth` or is unavailable, before the Run records its submission.

        ``drive_adapt`` turns a refusal of the circuit's depth into the stop
        ``native_depth_refused`` and raises a refusal for the installed package
        versions (``provenance``) without ending the iteration.
        """
        from nwqlib.backends.inspection import check_native_depth_cap

        check_native_depth_cap(prepared, self.max_native_two_qubit_depth)

    def prepare_all_refusal(self):
        """ADAPT rejects ``settings="all"``, because its later queries act on generators selected from earlier results."""
        return (
            "ADAPT cannot prepare every setting in advance, because each iteration screens and "
            "projects with the generators selected from the outcomes of earlier queries. prepare(plan) "
            "with the default settings='first' prepares the controller's first query, and "
            + ONE_QUERY_ESTIMATE
        )

    def query_catalog_reason(self):
        """Return the reason that the experiments of every ADAPT Plan are query templates (``Method.query_catalog_reason``).

        The controller binds the chains, angles, term, group and register
        widths of each query from earlier outcomes. On the exact, sampled
        and classical routes every query it requests is a ``Plan.resolve``
        point of one experiment of ``_construction``. A quantum point's
        Program is a copy of that experiment's batch with its one setting
        (``ir.selection.select_experiment``), and a classical point keeps
        the one host stage. ``specialize_experiment`` and
        ``selected_kernels`` revise only the readout labels and the
        classical-routine declaration, so each point keeps the batch,
        observation kind and repetitions of its experiment. Under
        ``sampling`` every selection, pilot and main query also binds its
        ``stage`` and ``shots``, and its batch repeats ``shots`` times, so its
        estimated shots are its own. Every declared batch of a Plan with
        shots reads counts, so the whole-Plan exact evaluations are zero.
        """
        return (
            "ADAPT binds the generator chains, angles and other parameters of each query from the "
            "outcomes of earlier queries, so the experiments of its Plan are query templates and the "
            "number of queries is not known before a Run. " + ONE_QUERY_ESTIMATE
        )

    def prepare(self, plan, *, run, settings="first"):
        """Prepare the first new query of the ADAPT iteration (adapt_acquisition.prepare_adapt).

        ``settings`` is always ``"first"``, because ``prepare_all_refusal``
        rejects ``"all"`` before a Run exists, so ``prepare_adapt`` does not
        take it.
        """
        from .adapt_acquisition import prepare_adapt

        return prepare_adapt(plan, run=run)

    def execute(self, plan, *, run):
        """Advance the ADAPT iteration on the Run's original work (adapt_acquisition.drive_adapt)."""
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
        """Save the state and action caches, without the collected observations, which the Run or Result holds.

        A live Run's run log and a saved Result's RunData hold every collected
        chunk and its order, so the context refers to them instead of storing
        a second copy of each chunk. Array and gate files are written once per
        object and shared by later checkpoints (``ArchiveFiles.write_array``).
        """
        from .adapt_archive import save_context

        return save_context(context, files)

    @staticmethod
    def _load_live_run_context(data, files, observations):
        """Restore indices once from the collected data, without analysis."""
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
        """Share immutable scientific data and keep the mutable Qiskit caches private.

        The observation index is left out, because the Result's RunData holds
        the collected chunks and their order, and a saved Result refers to them
        (``save_run_context``), so a live and a reloaded Result carry the same
        context. ``compiler_plans`` becomes the read-only ``(pool index,
        kind, active modes, occupation blocks)`` entries built so far, so a
        Result from a reopened Run and its saved copy carry the compiler
        plans of the selected generators, which circuit-level verification uses.
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
                field.name: freeze(_compiler_entries(context.compiler_plans)) if field.name == "compiler_plans"
                else freeze(getattr(context, field.name))
                for field in fields(AdaptContext)
                if field.name not in {"gates", "adjoints", "preparations", "reference_gates", "matrix_data", "pair_values",
                                      "observations", "contribution_order", "contribution_seen",
                                      "pending_items"}
            }
        )

    @staticmethod
    def load_run_context(data, files):
        from .adapt_archive import load_context

        return load_context(data, files)

    def validate_point(self, plan, experiment, values):
        """Require canonical used and unused chain positions and a supported Pauli insertion, if any.

        A grouped pencil setting measures the ancilla and the full system
        register, an off-diagonal ``pair`` of two different chains with one
        quadrature and Hamiltonian group, or a ``diagonal`` of one chain on
        both sides. Real-only quadratures admit no imaginary pair query.
        Under ``sampling`` a selection query binds the Plan's ``shots``, and
        only a pencil setting has a pilot or main stage.
        """
        experiment = experiment.name
        r = plan.reconstruction
        grouped = r.acquisition.matrix_measurement == "grouped"
        if "stage" in values and (number(values["stage"]) == -1 and number(values["shots"]) != plan.shots
                                  or number(values["stage"]) != -1 and experiment not in ("pair", "diagonal")):
            raise ValueError("a selection query binds the Plan's shots, and only a pencil setting has a pilot "
                             "or main stage")
        if number(values["control_width"]) != int(experiment in ("pair", "diagonal")) or number(
            values["readout_width"]
        ) != (1 if experiment == "pair" and not grouped else r.num_qubits):
            raise ValueError("query width must match its actual system/ancilla acquisition")
        if experiment in ("pair", "diagonal") and r.acquisition.quadratures == "real" and number(
                values["quadrature"]):
            raise ValueError("a real-only pencil has no imaginary pair query")
        if grouped and experiment in ("pair", "diagonal"):
            if (number(values["term"]) != -1
                    or (chain(values, "l") == chain(values, "r")) != (experiment == "diagonal")
                    or experiment == "diagonal" and number(values["quadrature"])
                    or not 0 <= number(values["group"]) < (len(r.energy_groups) if experiment == "diagonal"
                                                            else max(1, len(r.energy_groups)))):
                raise ValueError("a grouped pencil setting needs its chains, quadrature and Hamiltonian group")
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
            # The default growth selects each generator once. A bounded chain
            # may repeat one, since chains are compared as whole tuples.
            if plan.method.basis_growth is None and len(indices) != len(set(indices)):
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
        if experiment not in ("pair", "diagonal") and chain(values, "l"):
            raise ValueError("only a pair query accepts a left chain")
        if experiment not in ("pair", "diagonal") and (
            number(values["term"]) != -1 or number(values["quadrature"]) != 0
        ):
            raise ValueError("inactive pair term/quadrature must be canonical")
        if experiment == "pair" and not grouped and number(values["group"]) != 0:
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
        Hamiltonian labels, and a termwise pair query reads the ancilla Z
        label. A sampled query reads the subset that falls in its QWC group.
        A grouped pencil setting estimates its Hamiltonian group's labels,
        and an off-diagonal one also the overlap, named by the identity word;
        ``fixed_basis.reduce_group`` reads its counts. Classical
        plans return classical scalars instead: ``g_i`` per unselected generator,
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
        if experiment in ("pair", "diagonal") and r.acquisition.matrix_measurement == "grouped":
            ordered = energy_labels((t.label for t in r.nonidentity_terms), label_cache(plan))
            group = int(number(values["group"]))
            members = tuple(ordered[i] for i in r.energy_groups[group]) if r.energy_groups else ()
            return (("I" * r.num_qubits,) if experiment == "pair" else ()) + members
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

        An empty label set would be a zero-work query, so the ADAPT iteration
        must skip it before preparation instead of submitting it. An exact quantum
        pair query reads one weighted-pencil reduction instead of labels. Its
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
        """Check all actions and other work of one classical query together.

        ``max_products`` bounds this complete call. The sum
        (``adapt_actions.query_work``) covers the reference, generator,
        Hamiltonian and scalar work of the call before it starts, so every
        term of the query is bounded by the nonnegative total counted for it,
        and no part can exceed the limit. The inner matrix-vector check may
        still receive the whole limit. Every later call is checked afresh.
        Resolution, analysis and repeated preparation checks use no
        execution budget, so this method keeps no Run-wide record. Returns
        the classical-routine declaration revised with this point's labels
        and its full ``invocation_work``, or an empty tuple for quantum
        plans. Application counters record the actions performed after
        cache reuse.
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
