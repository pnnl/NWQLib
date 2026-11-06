"""Controlled powers of U for the Hadamard test: step selection and circuit constructors.

select_powers chooses, once per Plan, how controlled powers of U are
built: exact dense powers of one base (the polar factor of a supplied
unitary, or the spectral exponential of a Hamiltonian), or repetitions of
controlled second-order Suzuki steps whose counts are chosen from the
commutator bound of Childs et al. doi:10.1103/PhysRevX.11.011020
(subroutines/trotterization/error_budget.py). An exact static trajectory
repeats one shared step on a common integer grid and applies gap blocks
between its power positions. Sampled static settings and RWPE choose a
step count for each power independently. The ``construct_*``
functions build the Qiskit circuits only when a Qiskit circuit is requested.
For product-formula powers the Hamiltonian identity coefficient is split off
and acts as a separate phase gate on the control (method._quantum_program).
A dense power contains it in its matrix.
"""

from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import isfinite

from nwqlib._limits import ResourceLimitError, exceeds
from nwqlib._linalg_laws import hermitian_eigensystem_work
from nwqlib.blocks.records import BlockSemantics, Primitive, SelectedDefinition
from nwqlib.blocks.selection import NATIVE_POSITION_BYTES, SelectedBlock
from nwqlib.core.records import Float64, Source
from nwqlib.evidence import Evidence
from nwqlib.ir import Binding, BlockSignature, Parameter, QuantumPort
from nwqlib.resources import ResourceLaw
from nwqlib.subroutines._dense_synthesis import controlled_synthesis_size
from nwqlib.subroutines.trotterization.error_budget import (
    PruningBudgetExhausted,
    TrotterStepSelection,
    _BoundCoefficientEvaluation,
    _emitted_parts,
    _finite_upper,
    _is_normal,
    _pauli_bound_coefficient_from_terms,
    _remaining_budget_after_pruning,
    _select_common_step,
    _selection_from_evaluation,
    _smallest_step_count,
    _upward_float,
    census_bytes,
    census_work,
)
from .records import QPECommonStep, QPEPower


def source(name):
    """Return the implementation Source ``qpe.<name>`` of a QPE block kind."""
    return Source(
        name="qpe." + name,
        version="1",
        domain="selected physical controlled QPE construction",
        reference="nwqlib.algorithms.qpe.powers; identity phase stored under control",
    )


# Sources and qualifications of the per-block CX laws. The step law is the
# decomposition count derived in suzuki_step_cx. A one-qubit P or RZ gate has
# no CX in any lowering to CX and one-qubit gates.
_CX_SOURCES = {
    "suzuki_step": Source(
        name="qpe.suzuki_step_cx",
        version="1",
        domain="controlled second-order Suzuki step",
        reference="4*sum_j w_j CX with 2*w per controlled Pauli leaf and two leaves per term, "
        "for Qiskit 2.6.0 PauliEvolutionGate.control at optimization level 0 without routing",
    ),
    "phase": Source(
        name="qpe.phase_cx",
        version="1",
        domain="one-qubit P or RZ",
        reference="a one-qubit gate has no CX",
    ),
}
_CX_ASSUMPTIONS = {
    # Stored text, part of Plan content identity.
    "suzuki_step": (
        "Qiskit 2.6.0 lowering to cx and one-qubit gates at optimization level 0, "
        "before cancellation, rotation merging and routing, and another Qiskit version can differ",
    ),
    "phase": (),
}


def _make(
    name,
    kind,
    *,
    basis,
    reference,
    payload,
    width,
    work,
    epsilon=None,
    evidence="selected second-order/pruning relation where given; native numerical error unknown",
    constructor,
    parameters=(),
    cost_parameters=(),
    operations=None,
    cx=None,
    decomposition=None,
    context="",
    memory=None,
):
    """Return a SelectedBlock for one QPE block kind with its payload and constructor.

    ``kind`` is ``suzuki_step`` or ``dense_power`` (ports ``control`` and,
    when ``width > 1``, ``system``) or ``hadamard`` or ``phase`` (one
    ``register`` port). ``payload`` is the data the constructor needs,
    ``work`` the construction size checked against the limits and
    ``epsilon`` the per-block operator-norm error bound, when one exists,
    with ``evidence`` stating why. ``operations`` gives the exact number of
    instructions per block in the circuit description and ``cx`` its exact
    CX count in the qualified decomposition into CX and one-qubit gates
    (``suzuki_step_cx`` for a step, zero for a one-qubit phase). Each is
    recorded as a cost rule (``ResourceLaw``) only for blocks without a
    primitive ``decomposition``.

    The resource total applies a cost rule only when its fixed bindings
    equal the call's cost parameters (``resources/fold.py``, ``_Fold.leaf``),
    so both cost rules bind exactly the ``cost_parameters`` of their block,
    for example the ``power``, ``steps`` and ``step_time`` of one Suzuki
    step. The formal ``parameters`` (the ``angle`` of a phase leaf) stay
    unbound and the cost rules cover every allowed angle, since one P or RZ
    instruction with no CX is recorded whatever the angle. The cost rules keep
    ``controlled=False``, because a Suzuki step already contains its QPE
    control and the flag describes an additional control added to the
    chosen block. ``memory`` is the block's coarse memory estimate
    (``step_memory_bytes``, ``dense_memory_bytes`` or
    ``NATIVE_POSITION_BYTES`` for a one-qubit phase). It stays on the live
    binding, for one call in a native circuit and for one gate definition, and
    a block with a primitive ``decomposition`` takes the rule of
    ``SelectedBlock.bind`` instead.
    """
    implementation = source(kind)
    costs = tuple(Binding(parameter=key, value=value) for key, value in cost_parameters)
    ports = (
        (QuantumPort(name="control", width=1),)
        + ((QuantumPort(name="system", width=width - 1),) if width > 1 else ())
        if kind in ("suzuki_step", "dense_power")
        else (QuantumPort(name="register", width=width),)
    )
    signature = BlockSignature(
        name=name, target=implementation, quantum=ports, parameters=parameters
    )
    semantics = BlockSemantics(
        kind="unknown",
        input=reference,
        basis=basis,
        relation=context,
        input_projector="identity on the selected register",
        output_projector="identity on the selected register",
        success="deterministic selected unitary action",
        workspace=0,
        restoration="no additional workspace",
        epsilon=epsilon,
        approximation_metric="unitary operator norm before native floating synthesis",
        approximation_evidence=evidence,
        inverse_legal=False,
        control_legal=False,
        phase="physical phase stored; identity acts on the QPE control as P(-power*tau*identity)",
    )
    # Logical constructor slots are exact at this basis only. They do not
    # predict SDK decomposition or target-native gate counts.
    laws = ()
    if operations is not None and decomposition is None:
        law_source = Source(
            name="qpe.selected_slots",
            version="1",
            domain=kind,
            reference="one stored instruction per selected phase or controlled Pauli rotation; no native basis prediction",
        )
        laws = (
            ResourceLaw(
                metric="operations",
                basis="selected_logical",
                value=operations,
                interpretation="exact",
                bindings=costs,
                unbound_parameters=tuple(parameter.name for parameter in parameters),
                evidence=Evidence(kind="proved_relation", source=law_source),
                assumptions=("logical constructor slots; HLS/native basis inventory is separate",),
            ),
        )
    if cx is not None and decomposition is None:
        laws += (
            ResourceLaw(
                metric="cx",
                basis="cx",
                value=cx,
                interpretation="exact",
                bindings=costs,
                unbound_parameters=tuple(parameter.name for parameter in parameters),
                evidence=Evidence(kind="proved_relation", source=_CX_SOURCES[kind]),
                assumptions=_CX_ASSUMPTIONS[kind],
            ),
        )
    record = SelectedDefinition(
        signature=signature,
        semantics=semantics,
        implementation=implementation,
        choice=name,
        decomposition=decomposition,
        cost_law=None,
        resource_laws=laws,
        cost_parameters=costs,
        cost_context=context
        + "; logical construction size excludes unknown SDK synthesis/allocator overhead",
        construction_work=work,
    )
    return SelectedBlock.bind(record, payload=payload, constructor=constructor,
                              native_bytes=memory, definition_bytes=memory)


def step_memory_bytes(terms):
    """Return the coarse bytes of one controlled second-order step in a native circuit or a gate definition.

    The estimate is ``NATIVE_POSITION_BYTES`` for each Aer native position.
    A term of Pauli weight w with x letters X or Y lowers in one sweep to its
    parity chain and inverse, 2(w - 1) CX, the central controlled RZ and two
    basis changes for each of those x letters (``suzuki_step_cx`` derives
    this form), and a step has two sweeps. The count matched Aer's lowered
    circuit on every label tried, with Qiskit 2.6.0 and Aer 0.18.0.
    ``Repeat`` multiplies the native part by the step count and leaves the
    single definition alone. Footprint growth of 159 to 217 bytes per
    instruction on periodic Ising chains supports the constant. It is an
    estimate for comparing with ``max_bytes``, not a bound on process memory.
    """
    return NATIVE_POSITION_BYTES * 2 * sum(
        2 * (len(label) - label.count("I") - 1) + 1 + 2 * (label.count("X") + label.count("Y"))
        for label, _ in terms
    )


def dense_memory_bytes(dimension):
    """Return the coarse bytes of one controlled dense power of a ``dimension``-square matrix.

    It is the kept-circuit estimate of ``_dense_synthesis.controlled_synthesis_size``
    on the control and system qubits. Aer keeps every position of that
    circuit, measured as ``0.44 M**2`` to ``0.68 M**2`` positions for
    ``M = 2 dimension``, 259 or more estimated bytes per position, so the same value
    serves for a native call and for the gate definition.
    """
    return controlled_synthesis_size(dimension.bit_length())[2]


def select_auxiliary(*, basis, reference):
    """Select the Hadamard and the two one-qubit phase blocks of the QPE control, with exact angles.

    The phase blocks take their angle as a call argument and apply it without
    truncation. Returns the Hadamard, the identity-phase block P(angle) and
    the feedback block RZ(angle). P(angle) = diag(1, exp(i*angle)) is exactly the controlled
    scalar phase of the Hamiltonian identity term. RZ(angle) equals P(angle)
    up to a global phase of the whole circuit, which is all a quadrature or
    feedback rotation on the uncontrolled ancilla needs.
    """
    hadamard = _make(
        "hadamard",
        "hadamard",
        basis=basis,
        reference=reference,
        payload=None,
        constructor=construct_hadamard,
        width=1,
        work=1,
        decomposition=(Primitive(gate="h", qubits=(0,)),),
        context="Hadamard on the QPE control",
    )
    parameter = Parameter(name="angle", domain="real")
    phase = _make(
        "identity_phase",
        "phase",
        basis=basis,
        reference=reference,
        payload="p",
        constructor=construct_phase,
        width=1,
        work=1,
        parameters=(parameter,),
        operations=1,
        cx=0,
        memory=NATIVE_POSITION_BYTES,
        context="P(angle) on the control; original Hamiltonian identity phase",
    )
    feedback = _make(
        "feedback",
        "phase",
        basis=basis,
        reference=reference,
        payload="rz",
        constructor=construct_phase,
        width=1,
        work=1,
        parameters=(parameter,),
        operations=1,
        cx=0,
        memory=NATIVE_POSITION_BYTES,
        context="RZ(angle) on the control; real/imaginary quadrature or RWPE feedback",
    )
    return hadamard, phase, feedback


def _census(labels, coefficients, width, config):
    """Evaluate the one second-order commutator count after checking its work and bytes.

    The expression is chosen from
    structural counts (``error_budget.census_structure``): the full
    Pauli-triangle expression ("exact_census") when its w*N(A) nested word
    tests are at most ``EXACT_CENSUS_WORD_TESTS``, and the degree-capped
    suffix relaxation ("relaxed_prefix") above. The limits never choose it.

    Work is ``2*p*q`` label reads, for the packed masks and the support sum
    of ``suzuki_step_cx``, plus ``error_budget.census_work`` of the chosen
    expression, counted once for all powers. The pair scan, ``2*p*q + w*P``
    with w=ceil(q/64) and P=p(p-1)/2, is checked against ``max_work``
    before the masks exist. After the scan, the work of the chosen
    expression is checked against ``max_work`` and its
    ``error_budget.census_bytes`` against ``max_bytes``, before any nested
    test. A refusal states the work and bytes needed. Step-count and error
    checks remain.
    """
    count = len(labels)
    linear = 2 * count * width
    config._admit(linear + (width + 63) // 64 * (count * (count - 1) // 2), 0)

    def admit(census):
        exact = census.variant == "exact_census"
        work = linear + census_work(count, width, order=2, variant=census.variant, pairs=census.pairs,
                                    nested=census.nested_tests if exact else 0)
        size = census_bytes(count, width, census.pairs, order=2, variant=census.variant, tile=census.nested_tile)
        config._admit(work, size)

    return _pauli_bound_coefficient_from_terms(
        tuple(zip(labels, coefficients, strict=True)), 2, max_bytes=config.max_bytes, admit=admit)


def polar_base(matrix, config):
    """Return V = polar(A) = U @ Vh from one NumPy SVD of the dense input A, refusing an A that is not unitary.

    For dense unitary input, planning forms one polar factor and constructs
    controlled gap powers of it. For a nonsingular A the ideal target is
    V = polar(A), a unitary. If eta_A = ||A^dagger A - I||_2 < 1, singular
    values give ||A - V||_2 = max_j |sigma_j(A) - 1|
    <= eta_A/[1 + sqrt(1 - eta_A)], from |sigma-1| = |sigma**2-1|/(sigma+1).
    Numerical polar computation contributes its own error relative to V.

    The same singular values decide the unitarity check of A, the entrywise
    window max_ij |(A^dagger A - I)_ij| <= atol of _matrix_checks.is_unitary.
    With E = A^dagger A - I, eta_A = ||E||_2 = max_j |sigma_j**2 - 1| and
    max_ij |E_ij| <= ||E||_2 <= D*max_ij |E_ij|. So eta_A <= atol accepts
    A, eta_A > D*atol refuses it, and only in between does is_unitary form
    A^dagger A, after the singular vectors are released. Each decision
    agrees with the entrywise test up to the rounding of the two
    computations: near atol and D*atol the decision follows the computed
    singular values, or the computed A^dagger A in between, and can differ
    from the entrywise test by their rounding. atol is a screening window
    for matrices that are not unitary. The accuracy statement is the
    distance bound above, which concerns the polar target V.

    Limit check. The SVD counts 9*D**3 + 8*D**2 work units, and an
    undecided unitarity check another 4*D**3. The bytes are 128*D**2 for the
    dominant arrays: the input, NumPy's copy of it, the two singular-vector
    frames, and ZGESDD's real workspace of 5*D**2 + 7*D values and complex
    workspace of D**2 + 3*D entries. The polar factor and the entrywise
    check fit once the frames are released. The polar factor keeps 16*D**2
    bytes in memory. This runs once per Plan for unitary dense input.
    Hamiltonian input keeps its Hermitian spectral route.
    """
    import numpy as np

    from nwqlib.subroutines._matrix_checks import UNITARY_ATOL, is_unitary

    d = len(matrix)
    config._admit(9 * d**3 + 8 * d * d, 128 * d * d)
    left, values, right = np.linalg.svd(matrix)
    base = left @ right
    del left, right
    defect = float(np.max(np.abs(values * values - 1)))
    if defect > UNITARY_ATOL:
        unitary = False
        if defect <= d * UNITARY_ATOL:
            config._admit(4 * d**3, 128 * d * d)
            unitary = is_unitary(matrix)
        if not unitary:
            raise ValueError("QPE unitary input violates numerical unitarity")
    base.flags.writeable = False
    return base


def gap_powers(powers):
    """Return the positive gaps p_i - p_(i-1) of sorted distinct nonnegative powers, with p_0 = 0.

    Between powers p_(i-1) and p_i the trajectory applies controlled V
    exactly p_i - p_(i-1) times. Since the two control projectors are
    orthogonal, controlled(A) controlled(B) = controlled(AB), so the
    boundary after the gap to p_i holds (|0>|phi> + |1> V**p_i |phi>)/sqrt2.
    Repeated powers have zero gaps and reuse their point, and power zero is
    algebraic.
    """
    gaps, previous = {}, 0
    for power in sorted(powers):
        gaps[power] = power - previous
        previous = power
    return gaps


@dataclass(frozen=True)
class PowerSelection:
    """The constructions chosen for one Plan's powers.

    Attributes:
        powers: One QPEPower per distinct power, sorted.
        blocks: Every power block, each once.
        evaluation: The one evaluation of the Pauli commutator count, or None.
        census: Whether the commutator count ran, so that ``evaluation``
            describes it.
        common: The shared step record of exact-trajectory product-formula powers.
        step: The shared controlled step block, or None.
        increments: Certified binary64 identity phases. The exact
            trajectory records the increment before each power position,
            and sampled and RWPE Plans record the whole phase for each
            independent power.
        dense: Dense power blocks by their nonnegative exponent.
    """

    powers: tuple
    blocks: tuple
    evaluation: object = None
    census: bool = False
    common: object = None
    step: object = None
    increments: dict = field(default_factory=dict)
    dense: dict = field(default_factory=dict)


def select_powers(*, target, base, tau, labels, coefficients, identity, pruned_mass, powers,
                  backend, config, route):
    """Count the commutators of the kept terms once and choose every power construction once.

    ``route`` is ``trajectory`` (the exact static quantum trajectory),
    ``sampled`` (separate static Hadamard tests with counts), ``rwpe`` or
    ``classical``.

    Product formula, trajectory, sampled and RWPE routes. One count of Pauli
    commutators gives an upper bound W_up on the second-order coefficient of
    Childs, Su, Tran, Wiebe and Zhu, Phys. Rev. X 11, 011020 (2021),
    doi:10.1103/PhysRevX.11.011020, Prop. 10, Eq. (121) (arXiv:1912.08854v3
    Prop. 16, Eq. (152)), from the Pauli-triangle expansion in
    trotterization.error_budget. The count uses the full Pauli-triangle
    expression when its nested word tests are at most
    ``EXACT_CENSUS_WORD_TESTS``, and otherwise the second-order
    degree-capped suffix relaxation. Work and memory limits refuse the
    chosen expression and never change it (``_census``).

    Product formula, exact static trajectory. The exact static trajectory
    uses one propagated state and keeps the common grid.
    ``error_budget.select_common_step`` chooses one ordered second-order
    step for the nonnegative integer power positions. The ideal rational
    grid uses ``t_p=p*val(tau)`` and integer prefix counts. The emitted
    binary64 step and the cumulative count at every point are saved. Each
    prefix is accepted using its pruning bound, ``W*r_p*abs(val(h_hat))**3``,
    kept-generator time-displacement bound and applicable identity-phase and
    rotation-angle formation bounds. The counts are sufficient for the
    shared trajectory and need not minimize each prefix independently. Every
    power's allowance is ``controlled_power_error_budget``. A failed
    subtotal gets a bounded correction of the shared subdivision m, never of
    one power's count (``error_budget._common_envelope_count``), because a
    separate count would leave the common grid. If that confirms no count,
    planning stops with an error that names the power, the allowance and the
    contributions. Target increments are ``(p_k-p_(k-1))*val(tau)``. Each
    recheck takes O(L+K) arithmetic reductions for L kept terms and K powers,
    with at most two added by the correction, and is not compared with
    ``max_work``.

    Product formula, sampled static settings and RWPE. A sampled setting
    starts from its own preparation and runs its own Hadamard circuit. It
    does not share the propagated state at earlier powers. Therefore it can
    use an independent Suzuki step count, and so does each continuous RWPE
    power. Each power has exact signed target t=val(p)*val(tau). Its trial
    step count is chosen independently at the displayed positive time, then
    its emitted parameters are checked against the complete
    subtotal in _independent_powers. The step block carries its intrinsic
    formula-plus-angle bound at the represented step time. Pruning, time
    displacement and identity phase enter only the complete power bound.
    These routes record no common step. A count whose complete subtotal fails
    gets one bounded correction (``_corrected_steps``), and this recheck is
    not compared with ``max_work`` either.

    Dense powers. A Hamiltonian power is the exact spectral power. For a
    dense unitary input A, the base is its unitary polar factor V
    (``base``, ``polar_base``). Every requested power and trajectory gap uses
    this same base. A gap of k powers is constructed by repeated squaring of
    V and built as a controlled dense block. The target operator at
    power p is V^p. The polar decomposition, the floating-point matrix
    powers and the synthesis of the controlled gates contribute numerical
    error. Each gap's limit check covers its matrix work, its synthesis and
    the cached matrices held in memory. The trajectory constructs one block
    per distinct positive gap and a sampled or RWPE Plan one block per
    distinct nonzero power.

    Returns:
        PowerSelection.
    """

    dimension = target.manifest.basis.dimension
    width = dimension.bit_length() - 1
    common_grid = route == "trajectory"
    records, blocks, dense = [], [], {}
    if backend == "dense_exact":
        gaps = gap_powers(powers) if route == "trajectory" else {}
        exponents = sorted({k for k in gaps.values() if k} if route == "trajectory"
                           else {abs(p) for p in powers if p})
        # J from every exponent the Plan constructs, admitted before the cache exists.
        squares = max((k.bit_length() - 1 for k in exponents if type(k) is int), default=0)
        matrix = (target if base is None else base).dense_array()
        reference = (target if base is None else base).manifest.reference
        for exponent in exponents:
            dense[exponent] = _dense_block(target, reference, matrix, tau, exponent, config,
                                           squares=squares)
            blocks.append(dense[exponent])
        for power in powers:
            key = gaps.get(power) if route == "trajectory" else abs(power)
            block = dense.get(key) if power else None
            records.append(QPEPower(
                power=power, backend=backend, evolution_time=_display_time(power, tau),
                steps=0, pruning_error=None, evolution_error=None, total_error=None,
                selected_definition=None if block is None else block.record.content_id))
        return PowerSelection(powers=tuple(records), blocks=tuple(blocks), dense=dense)
    if backend != "trotter_error_budgeted":
        for power in powers:
            records.append(QPEPower(
                power=power, backend=backend, evolution_time=_display_time(power, tau),
                steps=0, pruning_error=None, evolution_error=None, total_error=None))
        return PowerSelection(powers=tuple(records), blocks=())
    terms = tuple(zip(labels, coefficients, strict=True))
    evaluation = None
    allowances = ({power: config.controlled_power_error_budget for power in powers}
                  if common_grid else {})
    if terms:
        evaluation = _census(labels, coefficients, width, config)
    census = evaluation is not None
    if not common_grid:
        return _independent_powers(target, tau, terms, identity, pruned_mass, powers, config,
                                   evaluation)
    if evaluation is None:
        # An empty kept generator: W = 0 and no census work.
        evaluation = _BoundCoefficientEvaluation(
            coefficient=Fraction(0), pauli_term_count=0, pair_commutation_checks=0,
            nested_commutation_checks=0, bound_variant="exact_census")
    selection, prefixes, certified = _select_common_step(
        evaluation, coefficients=coefficients, identity=identity, tau=tau,
        allowances=allowances, dropped_mass=pruned_mass, max_steps=config.max_trotter_steps)
    common = step = None
    # The recheck certified these emitted increments. Without a positive
    # power the only position is power 0, with no phase.
    increments = certified or {power: 0.0 for power in powers}
    if selection is not None:
        common = QPECommonStep(
            tau=selection.common_tau, g=selection.common_g, m=selection.common_m,
            step_time=selection.common_step_time, step_count=selection.step_count,
            evolution_time=selection.evolution_time, error_budget=selection.error_budget,
            bound_value=selection.bound_value)
        if terms:
            angle_per_step = prefixes[max(prefixes)][1]["angles"] / selection.step_count
            step = _step_block(
                target, terms, selection.common_step_time, evaluation, config,
                angle_per_step=angle_per_step,
            )
            blocks.append(step)
    counts = {} if selection is None else dict(selection.common_prefix_steps)
    published = {} if selection is None else dict(selection.common_prefix_bounds)
    for power in powers:
        if selection is None:
            pruning = evolution = total = 0.0
        else:
            exact, parts = prefixes[power]
            pruning = _finite_upper(parts["pruning"], "pruning error")
            evolution = _finite_upper(exact - parts["pruning"], "evolution error")
            total = published[power]
        records.append(QPEPower(
            power=power, backend=backend, evolution_time=_display_time(power, tau),
            steps=counts.get(power, 0), pruning_error=pruning, evolution_error=evolution,
            total_error=total,
            selected_definition=None if step is None or not power else step.record.content_id))
    return PowerSelection(powers=tuple(records), blocks=tuple(blocks), evaluation=evaluation,
                          census=census, common=common, step=step,
                          increments=increments)


def _display_time(power, tau):
    """Return the binary64 display of the target time |p|*val(tau), or None for unitary input.

    For an integer power the display is the correctly rounded exact product.
    An RWPE relative time keeps its binary64 product.
    """
    if tau is None:
        return None
    time = float(abs(power) * Fraction(tau)) if type(power) is int else abs(power) * tau
    if not isfinite(time):
        raise ValueError("selected power evolution time must be finite")
    return time


def _admit_step_count(power, steps, config):
    """Refuse a step count above the explicit ``max_trotter_steps``, naming the field and the count it needs."""
    if exceeds(steps, config.max_trotter_steps):
        raise ResourceLimitError(
            f"QPE power {power!r} needs {steps} product-formula steps, above "
            f"max_trotter_steps={config.max_trotter_steps!r}. Raise max_trotter_steps to at least {steps}, or raise "
            f"controlled_power_error_budget={config.controlled_power_error_budget!r} to need fewer steps")


def _parts_text(parts):
    """Display the five parts of B(r), each rounded upward."""
    return ", ".join(f"{name}={_upward_float(value)!r}" for name, value in parts.items())


def _corrected_steps(power, time, target_time, steps, step_time, parts, *, W, mass, config, emit):
    """Return ``(count, emit(count))`` for a power whose first count failed its complete subtotal, or refuse.

    The first count inverts only the ideal formula term, and the time and
    angle terms come from rounding, so it fails once its margin below epsilon
    falls below those terms. The count where that happens depends on epsilon,
    and B(r) need not decrease with r. Acceptance is
    still the exact B(r) <= epsilon of ``_emitted_parts``, so an accepted
    count is safe whichever way it was found. This rule only proposes counts.

    Let T = abs(time), the binary64 time, with h, T and the half step 0.5*h
    all normal, u = 2**-53 and M the kept mass. Then
    abs(abs(h) - T/r) <= u T/r, so W r abs(h)**3 <= C/r**2 with
    C = W (1 + u)**3 T**3, and M abs(r h - t) <= M (abs(T - abs(t)) + u T). The
    half step is exactly h/2, so r A_step <= u (1 + u) T M + r F, with F the
    subnormal floor of ``angle_per_step``. With the pruning and identity parts
    fixed by the Plan,

        B(r) <= C/r**2 + D + r F,
        D = abs(t) d_drop + abs(phi + t c_I) + M (abs(T - abs(t)) + u T) + u (1 + u) T M.

    D is an upper envelope. For Q = epsilon - D > 0 the candidates are
    r1 = max(1, ceil(sqrt(C/Q))) and r1 + 1, which each recheck the normal
    premise, the explicit ``max_trotter_steps`` and the exact B(r), so one
    failed power costs at most two scalar evaluations and no commutator
    expansion. Since C <= Q r1**2, ``F (r1 + 1)**3 <= Q (2 r1 + 1)`` makes the
    envelope pass at r1 + 1. When that inequality fails, the rule can find
    nothing. The three refusals therefore state what they know: ``Q <= 0`` and
    two failed candidates show only that this rule confirmed no count, not
    that the budget is too small or that no count exists.
    """
    epsilon = config.controlled_power_error_budget
    failed = f"QPE power {power!r}: emitted subtotal exceeds controlled_power_error_budget={epsilon!r}"
    raised = "Raise controlled_power_error_budget"
    if not steps:
        raise ValueError(f"{failed} with no product-formula step to change: {_parts_text(parts)}. {raised}")

    def premise(count, value):
        if not all(_is_normal(item) for item in (time, value, 0.5 * value)):
            raise ValueError(f"{failed}: at {count} steps the evolution time {time!r} or the step {value!r} is below "
                             f"the normal binary64 range, so the rounding envelope that proposes a step count does "
                             f"not apply: {_parts_text(parts)}. Rescale the units of the Hamiltonian and tau")

    premise(steps, step_time)
    u, span, budget = Fraction(1, 2**53), Fraction(time), Fraction(epsilon)
    constant = W * (1 + u)**3 * span**3
    fixed = (parts["pruning"] + parts["identity"] + mass * (abs(span - abs(target_time)) + u * span)
             + u * (1 + u) * span * mass)
    if budget <= fixed:
        raise ValueError(f"{failed}: the conservative rounding envelope {_upward_float(fixed)!r} of its "
                         f"step-independent terms leaves no room for a step count, which does not show that the "
                         f"true error exhausts the budget. {raised} above that envelope. At {steps} steps: "
                         f"{_parts_text(parts)}")
    first = _smallest_step_count(constant, budget - fixed, 2)
    tried = []
    for count in (first, first + 1):
        _admit_step_count(power, count, config)
        emission = emit(count)
        premise(count, emission[0])
        if emission[1] <= budget:
            return count, emission
        tried.append(f"{count} steps: {_parts_text(emission[2])}")
    raise ValueError(f"{failed}: the bounded step rule confirmed neither candidate, which does not show that no count "
                     f"exists. {raised}. " + "; ".join(tried))


def _independent_powers(target, tau, terms, identity, pruned_mass, powers, config,
                        evaluation):
    """Choose and certify the step counts of independent QPE powers against t=val(p)*val(tau).

    The powers are the integer powers of sampled QCELS, SPE and RFE
    settings or the continuous powers of RWPE. Each is chosen on its own,
    because its circuit starts from its own preparation and does not share
    a propagated state with other powers.

    Choose one trial count at the displayed positive time using the
    outward pruning subtotal. With t_p=p*val(tau), d_up the dropped
    coefficient mass and epsilon_p=controlled_power_error_budget, the ideal
    remaining allowance is epsilon_p-|t_p|*d_up. At order two the formula
    term is W_up*|t_p|**3/r_p**2, so for a positive remaining allowance the
    smallest sufficient integer count is
    r_p = max(1, ceil(sqrt(W_up*|t_p|**3/(epsilon_p-|t_p|*d_up)))),
    evaluated by the exact rational inversion in _selection_from_evaluation,
    not by floating ceil(sqrt(...)). Recheck the emitted signed step h and
    both leaf sweeps. With M=sum|c_j|, the angle formation error
    A=2*sum|val(a_j)-val(h)*c_j/2| of the half angles a_j = fl(s*c_j),
    s = fl(0.5*h), that rotation_schedule emits is bounded a priori. With
    gradual underflow every rounding obeys |fl(x)-x| <= u|x| + 2**-1075,
    u = 2**-53, so |a_j - h*c_j/2| <= u|s||c_j| + 2**-1075*(1 + |c_j|) and
    A <= A_up = 2u|s|M + 2**-1074*(L + M) for L kept terms. The intrinsic
    error of one step is W*|val(h)|**3 + A_up. The complete power subtotal is
    B_p = |t|*d_up + r*(W*|val(h)|**3+A_up) + M*|r*val(h)-t| + |val(phi)+t*c_I|,
    with every other quantity an exact rational of the binary64 parameters.
    B_p <= epsilon_p is required and its upward rounding is recorded.
    phi=-p*tau*c_I is emitted separately on the control and is recorded for
    the Program to use. A count that passes is kept. A count whose complete
    subtotal fails gets one bounded correction (``_corrected_steps``). The
    initial smallest-r claim concerns only the ideal formula bound at the
    displayed time, not the complete emitted-parameter subtotal. Source: the
    independent minimum-count rule of Childs et al.
    doi:10.1103/PhysRevX.11.011020, Sec. V B, with the second-order
    coefficient of Prop. 10, Eq. (121). docs/algorithms/qpe.md states the
    per-power choice.
    """
    coefficients = tuple(value for _, value in terms)
    step_cx = suzuki_step_cx(terms) if terms else None
    step_bytes = step_memory_bytes(terms)
    current = evaluation
    records, blocks, increments = [], [], {}
    mass = sum((abs(Fraction(value)) for value in coefficients), Fraction(0))
    # Every half angle is finite when the one of the largest coefficient is,
    # since rounding is monotone.
    largest = max(map(abs, coefficients), default=0.0)
    unit, tiny = Fraction(1, 2**53), Fraction(1, 2**1074)
    floor = tiny * (len(coefficients) + mass)
    W = Fraction(0) if evaluation is None else evaluation.coefficient
    dropped, budget = Fraction(pruned_mass), Fraction(config.controlled_power_error_budget)

    def emitted(power, time, phase, target_time, steps):
        """Return the signed step, the exact subtotal and parts of B(r), and the angle allowance of ``steps`` steps."""
        step_time = (1 if power > 0 else -1) * time / steps if steps else 0.0
        half = 0.5 * step_time
        if not all(isfinite(value) for value in (step_time, phase, half * largest)):
            raise ValueError("emitted phases and angles must be finite")
        angle_per_step = 2 * unit * abs(Fraction(half)) * mass + floor if coefficients else Fraction(0)
        return step_time, *_emitted_parts(
            W, dropped, mass, angle_per_step, Fraction(phase),
            target_time, Fraction(step_time), Fraction(identity), steps,
        ), angle_per_step

    for power in powers:
        time = abs(power) * tau
        if not isfinite(time):
            raise ValueError("selected signed-power evolution time must be finite")
        target_time = Fraction(power) * Fraction(tau)
        pruning_exact = abs(target_time) * Fraction(pruned_mass)
        pruning = _finite_upper(pruning_exact, "pruning error")
        try:
            remaining = _remaining_budget_after_pruning(
                total_budget=config.controlled_power_error_budget, pruning_error=pruning
            )
            selection = None
            if power and current is not None:
                if remaining == 0:
                    # Only an exactly zero bound fits a zero remaining allowance.
                    if current.coefficient != 0:
                        raise PruningBudgetExhausted
                    selection = TrotterStepSelection(
                        formula_order=2, evolution_time=time, error_budget=0.0,
                        bound_value=0.0, step_count=1,
                        pauli_term_count=current.pauli_term_count,
                        pair_commutation_checks=current.pair_commutation_checks,
                        nested_commutation_checks=current.nested_commutation_checks,
                        bound_variant=current.bound_variant,
                    )
                else:
                    selection = _selection_from_evaluation(
                        current, time=time, error_budget=remaining, order=2
                    )
                current = replace(current, pair_commutation_checks=0, nested_commutation_checks=0)
        except PruningBudgetExhausted as error:
            raise ValueError(
                "controlled-power allowance is exhausted by the actual pruning perturbation"
            ) from error
        steps = 0 if selection is None else selection.step_count
        _admit_step_count(power, steps, config)
        phase = -power * tau * identity
        step_time, exact, parts, angle_per_step = emitted(power, time, phase, target_time, steps)
        if exact > budget:
            steps, (step_time, exact, parts, angle_per_step) = _corrected_steps(
                power, time, target_time, steps, step_time, parts, W=W, mass=mass, config=config,
                emit=lambda count: emitted(power, time, phase, target_time, count),
            )
        evolution = _finite_upper(exact - parts["pruning"], "evolution error")
        total = _finite_upper(exact, "power error")
        step_error = _finite_upper(W * abs(Fraction(step_time))**3 + angle_per_step, "step error")
        block = bind_power(
            target=target, terms=terms, power=power, steps=steps, step_time=step_time,
            step_error=step_error, config=config, step_cx=step_cx, step_bytes=step_bytes,
        )
        if block is not None:
            blocks.append(block)
        increments[power] = phase
        records.append(QPEPower(
            power=power, backend="trotter_error_budgeted", evolution_time=time, steps=steps,
            pruning_error=pruning, evolution_error=evolution, total_error=total,
            selected_definition=None if block is None else block.record.content_id))
    return PowerSelection(powers=tuple(records), blocks=tuple(blocks), evaluation=evaluation,
                          census=evaluation is not None, increments=increments)


def _step_block(target, terms, step_time, evaluation, config, *, angle_per_step):
    """Return the one controlled second-order step shared by every exact-trajectory power position.

    The step applies every kept term for step_time/2 in the given order and
    again in reverse order (``rotation_schedule``), with leaf times
    ``0.5*step_time*c_j``, the values the common-step recheck certifies.
    Two controlled Pauli rotations per term, each on at most width+1 qubits,
    give 2*L*(width+1) work units for L terms, recorded as the block's
    construction work. Bytes are an allowance of 32 per term and qubit,
    checked against the memory limit. The work is of the same order, so
    max_work does not apply. The block's structural operator-norm error is the
    upward rounding of W_up*abs(val(step_time))**3 + angle_per_step, where
    angle_per_step is twice the exact sum of represented half-angle
    discrepancies. It bounds the emitted forward/reverse step against
    evolution of the kept nonidentity generator at the represented step
    time. Pruning, prefix time displacement and the separately emitted
    identity phase enter the complete prefix bound.
    The Program repeats the block by the
    difference of the cumulative counts between adjacent power positions.
    """
    width = target.manifest.basis.dimension.bit_length() - 1
    size = 2 * len(terms) * (width + 1)
    config._admit(None, 32 * len(terms) * (width + 1))
    return _make(
        "common_step",
        "suzuki_step",
        basis=target.manifest.basis,
        constructor=construct_suzuki_step,
        reference=target.manifest.reference,
        payload=(terms, step_time),
        width=width + 1,
        cost_parameters=(("step_time", Float64(value=step_time)),),
        work=size,
        epsilon=_finite_upper(
            evaluation.coefficient * abs(Fraction(step_time))**3 + angle_per_step,
            "step error",
        ),
        operations=2 * len(terms),
        cx=suzuki_step_cx(terms),
        context=f"shared controlled second-order step at time {step_time!r}; repeated between "
                "power positions; identity handled separately",
        memory=step_memory_bytes(terms),
    )


def _dense_block(target, reference, matrix, tau, exponent, config, *, squares):
    """Return the controlled dense block of one nonnegative exponent k of the base.

    Work counts D**3 units per product of two D-square matrices, the unit of
    _linalg_laws. For a unitary the per-exponent upper bound is
    W_k = [bitlength(k) + popcount(k) - 2]*D**3 + 8*D**2 + W_synthesis. It
    remains valid with shared squares, since a cache miss cannot require
    more squares than an independent construction. For a Hamiltonian,
    V*exp(-i*E*tau*k)*V^dagger is one product, and the eigendecomposition it
    reads is computed by whichever block is constructed first, so every
    block counts _linalg_laws.hermitian_eigensystem_work as well.
    Bytes: for a Hamiltonian an allowance of six D x D complex128 arrays
    (96*D**2). For a unitary the 96*D**2 allowance does not cover the
    persistent square cache, and the sufficient replacement is
    (96+16J)*D**2 + B_synthesis,working + B_synthesis,kept + H_cache with
    H_cache = 256(J+1) + H0, H0 = 65536, and J the largest square index of
    any exponent the Plan constructs, checked against the limits before the
    cache is filled (``construct_dense_power``). The synthesis of the
    controlled matrix on width + 1 qubits adds the work, working bytes and
    kept circuit of _dense_synthesis.controlled_synthesis_size. Each block is
    checked on its own. Comparing the reported Plan total with a limit would
    be a separate policy (docs/algorithms/qpe.md).
    """
    dimension = target.manifest.basis.dimension
    width = dimension.bit_length() - 1
    synthesis_work, working_bytes, kept_bytes = controlled_synthesis_size(width + 1)
    if tau is None:
        matrix_work = (exponent.bit_length() + exponent.bit_count() - 2) * dimension**3
        size = ((96 + 16 * squares) * dimension**2 + working_bytes + kept_bytes
                + 256 * (squares + 1) + 65536)
    else:
        matrix_work = hermitian_eigensystem_work(dimension) + dimension**3
        size = 96 * dimension**2 + working_bytes + kept_bytes
    work = matrix_work + 8 * dimension**2 + synthesis_work
    config._admit(work, size)
    return _make(
        _power_name(exponent),
        "dense_power",
        basis=target.manifest.basis,
        constructor=construct_dense_power,
        reference=reference,
        payload=(matrix, tau, exponent),
        width=width + 1,
        cost_parameters=(("power", exponent if type(exponent) is int else Float64(value=exponent)),)
        + (() if tau is None else (("tau", Float64(value=tau)),)),
        work=work,
        epsilon=0.0,
        evidence=("exact spectral power" if tau is not None else
                  "Exact-real-arithmetic power of the selected polar base. The computed gap matrix "
                  "is realized through its unitary polar factor, with numerical polar-selection, "
                  "power-formation and synthesis errors separate."),
        memory=dense_memory_bytes(dimension),
        context=f"explicit small dense controlled nominal power {exponent}, tau={tau!r}; cubic numerical size and exact controlled synthesis",
    )


def bind_power(*, target, terms, power, steps, step_time, step_error, config, step_cx, step_bytes):
    """Return the controlled step of one independently chosen sampled or RWPE power.

    step_time is the checked signed binary64 step time of that power.

    The Program repeats this block steps times. step_error bounds one
    emitted forward/reverse sweep against the kept nonidentity generator
    at val(step_time). Pruning, whole-power time displacement and the
    separately emitted identity phase belong to QPEPower.total_error.
    step_cx and step_bytes are computed once from the shared terms. Zero power and an
    empty kept generator need no block.
    """
    dimension = target.manifest.basis.dimension
    width = dimension.bit_length() - 1
    if steps:
        # Two controlled Pauli rotations per term (rotation_schedule), each on
        # at most width+1 qubits, give 2*L*(width+1) work units for L terms,
        # recorded as the block's construction work. Only the bytes, an
        # allowance of 32 per term and qubit, are checked, since the work is
        # of the same order.
        size = 2 * len(terms) * (width + 1)
        config._admit(None, 32 * len(terms) * (width + 1))
        return _make(
            _power_name(power), "suzuki_step", basis=target.manifest.basis,
            constructor=construct_suzuki_step, reference=target.manifest.reference,
            payload=(terms, step_time), width=width + 1,
            cost_parameters=(
                ("power", power if type(power) is int else Float64(value=power)),
                ("steps", steps), ("step_time", Float64(value=step_time)),
            ),
            work=size, epsilon=step_error, operations=2 * len(terms), cx=step_cx, memory=step_bytes,
            context=f"controlled second-order step at signed time {step_time!r}; repeat {steps} "
                    f"times for power {power}; identity handled separately",
        )
    return None


def _power_name(power):
    return f"power_{'m' if power < 0 else 'p'}{abs(power)}"


def construct_hadamard(block, arguments, method_context):
    """Build the one-qubit Hadamard circuit of the ancilla."""
    if arguments:
        raise ValueError("Hadamard has no scalar arguments")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1)
    circuit.h(0)
    return circuit


def construct_phase(block, arguments, method_context):
    """Build P(angle) or RZ(angle) on one qubit, as named by the block payload ``p`` or ``rz``."""
    if len(arguments) != 1 or arguments[0][0] != "angle":
        raise ValueError("QPE phase requires its exact admitted angle")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1)
    getattr(circuit, block._payload)(float(arguments[0][1]), 0)
    return circuit


def rotation_schedule(terms, step_time):
    """Ordered second-order rotations that the Qiskit circuit construction applies directly.

    One step of the second-order Suzuki formula S2(dt) applies every term for
    dt/2 in the given order and then again in reverse order. Term 1 acts
    first, which is the ordering for which Childs et al., Phys. Rev. X 11,
    011020 (2021), doi:10.1103/PhysRevX.11.011020, state Prop. 10
    (arXiv:1912.08854v3 Prop. 16). Their gamma = 1 factor is rightmost.
    PauliEvolutionGate(P, time=a) applies exp(-i*a*P), so the entry
    (label, dt*c/2) applies exp(-i*dt*c*P/2).

    Returns:
        Tuple of 2*L (label, angle) pairs for L terms, in application order.
    """
    half_step = tuple((label, 0.5 * step_time * coefficient) for label, coefficient in terms)
    return (*half_step, *reversed(half_step))


def suzuki_step_cx(terms):
    """Count CX gates in one controlled Suzuki step of QPE.

    Let w_j be the number of nonidentity symbols in nonidentity Pauli term
    j, and W_support = sum_j w_j over the terms after splitting and pruning.
    rotation_schedule emits each term twice. Write R_P(theta) =
    exp(-i*theta*P/2). A single-qubit basis change B maps each active factor
    of P to Z, and a parity chain A of w - 1 CX gates maps the product of
    those Z factors to one target Z, so with V = A*B,
    R_P(theta) = V^dagger R_Z(theta) V. For one control,
    C(R_P(theta)) = (I x V^dagger) CR_Z(theta) (I x V): on the inactive
    control branch V^dagger V = I, so the basis changes and parity chains
    stay uncontrolled. The central CR_Z(theta) takes two CX, as
    R_Z(theta/2), CX, R_Z(-theta/2), CX, since X R_Z(-theta/2) X =
    R_Z(theta/2). Thus the basis changes cost zero CX, the parity chain and
    its inverse 2*(w_j - 1), the central CRZ two, the controlled leaf
    2*w_j, and the controlled second-order step 4*W_support.

    construct_suzuki_step uses qiskit_compat.controlled on a one-term
    PauliEvolutionGate. Qiskit 2.6.0 controls its central rotation through
    the projector representation (``PauliEvolutionGate.control`` in
    qiskit/circuit/library/pauli_evolution.py, decomposed by
    crates/synthesis/src/pauli_evolution.rs), leaving the parity and basis
    networks uncontrolled. The corresponding uncontrolled leaf costs
    2*(w_j - 1), and the uncontrolled schedule 4*(W_support - L).

    These are exact decomposition counts for CX plus arbitrary single-qubit
    gates, qualified with Qiskit 2.6.0 and optimization level 0 without a
    coupling map or backend target. They exclude cancellation, rotation
    merging, alternative synthesis and routing. Angles and power signs do
    not change the emitted count. Identity evolution is a separate ancilla
    P(-power*tau*identity_coefficient), which costs zero CX, as do the
    one-qubit feedback RZ and Hadamard gates.

    select_powers computes W_support once in O(L*q) label reads and uses
    constant scalar work to instantiate each power's CX cost rule. The value
    is per step. Repeat supplies the steps multiplier, and the workload
    supplies repetitions. The cost rule binds the exact power, steps and
    step_time cost parameters (``_make``). It does not apply to the
    separately chosen dense-power synthesis.
    """
    return 4 * sum(len(label) - label.count("I") for label, _ in terms)


def construct_suzuki_step(block, arguments, method_context):
    """Build one controlled S2 step as a sequence of controlled Pauli exponentials.

    Controlling every factor equals controlling their product, so the step
    needs no controlled global phase. The identity phase is applied
    separately on the control. The reverse sweep of ``rotation_schedule``
    repeats the forward (label, angle) entries, so each of the L controlled
    leaves is constructed once and appended in both sweeps.
    """
    if arguments:
        raise ValueError("selected QPE step has no unresolved scalar arguments")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.subroutines.qiskit_compat import controlled

    terms, step_time = block._payload
    circuit = QuantumCircuit(sum(port.width for port in block.record.signature.quantum))
    schedule = rotation_schedule(terms, step_time)
    leaves = [
        controlled(PauliEvolutionGate(SparsePauliOp.from_list([(label, 1.0)]), time=time), 1)
        for label, time in schedule[: len(terms)]
    ]
    for leaf in (*leaves, *reversed(leaves)):
        circuit.append(leaf, circuit.qubits, copy=False)
    return circuit


def cached_unitary_power(matrix, power, squares):
    """Return matrix**power from the binary squares in ``squares``, appending missing squares.

    Reuse binary squares of the same checked unitary within the Plan
    context, preserving NumPy's multiplication order including its
    power-three case. The limit check counts every cached square that can
    remain in memory while the Qiskit circuit is built.

    The cache holds Z_0 = U (the block's own matrix, not a copy) and
    Z_j = Z_(j-1) @ Z_(j-1). For n = abs(power) > 0 with j = bit_length(n)-1
    and j_old the highest square cached before the call, it needs
    [max(0, j-j_old) + popcount(n) - 1]*D**3 of matrix work, and across calls
    [J + sum_calls(popcount(abs(p)) - 1)]*D**3 with J the largest square
    index. For n = 3 NumPy's special product Z_1 @ Z_0 is kept, since the
    general least-significant-bit accumulation Z_0 @ Z_1 need not round
    identically. The multiplication order is that of NumPy 2.5.2's
    ``matrix_power`` (numpy/linalg/_linalg.py), so every result keeps its
    bits. Negative powers follow the conjugate-transpose convention, not
    NumPy's inverse path, and power zero is algebraic.
    """
    import numpy as np

    n = abs(power)
    if n == 0:
        return np.eye(len(matrix), dtype=matrix.dtype)
    while len(squares) < n.bit_length():
        squares.append(squares[-1] @ squares[-1])
    if n == 3:
        result = squares[1] @ squares[0]
    else:
        result = None
        for j in range(n.bit_length()):
            if (n >> j) & 1:
                result = squares[j] if result is None else result @ squares[j]
    return result if power > 0 else result.conj().T


def construct_dense_power(block, arguments, method_context):
    """Build the controlled dense power of one exponent of the base.

    Construct a controlled gap power of the Plan's polar base. For a
    computed gap matrix B with operator-norm unitarity defect eta below
    one, its polar factor differs from B by at most eta/(1+sqrt(1-eta)).
    Error against the ideal power additionally includes B's formation
    error and circuit-synthesis error.

    For a Hamiltonian, U^p = V diag(exp(-i*E*tau*p)) V^dagger from the
    eigensystem kept in the method context, so every power and the
    classical signal reuse one eigendecomposition. For unitary input the
    payload matrix is the polar base, and its power comes from the binary
    squares kept in the method context under the base's input reference
    (``cached_unitary_power``). The squares are a recomputable construction
    cache and are not saved with a Result. ``qiskit_compat.controlled``
    synthesizes the unitary polar factor of the controlled matrix to
    binary64 rounding (``_dense_synthesis.controlled_unitary_circuit``). On
    m = n + 1 >= 3 qubits for n system qubits that takes at most
    ``(25/96) 4**m - 2**m + 4/3`` CX.
    """
    if arguments:
        raise ValueError("selected dense QPE power has no unresolved scalar arguments")
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from nwqlib.subroutines.qiskit_compat import controlled
    from .method import _new_context, _eigensystem

    matrix, tau, power = block._payload
    if tau is None:
        context = method_context(_new_context)
        key = block.record.semantics.input
        if context["squares"] is None or context["squares"][0] != key:
            context["squares"] = (key, [matrix])
        unitary = cached_unitary_power(matrix, power, context["squares"][1])
    else:
        (eigenvalues, eigenvectors), _ = _eigensystem(
            block.record.semantics.input,
            matrix,
            kind="hamiltonian",
            context=method_context(_new_context),
        )
        unitary = (eigenvectors * np.exp(-1j * eigenvalues * tau * power)) @ eigenvectors.conj().T
    # A computed gap matrix B need not be exactly unitary, and check_input=False
    # does not change it into one: it only skips Qiskit's constructor check of
    # B^dagger B (numpy.allclose with the identity, atol 1e-8 and rtol 1e-5),
    # and the controlled synthesis realizes the polar factor of B.
    gate = controlled(UnitaryGate(unitary, label=f"U^{power}", check_input=False), 1)
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, circuit.qubits, copy=False)
    return circuit
