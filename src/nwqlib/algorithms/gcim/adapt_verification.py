"""Explicit ADAPT state/sector/scalar checks with scoped verification receipts."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import isfinite
from typing import Annotated, Literal
import numpy as np
from pydantic import Field, model_validator
from nwqlib._numerics import normalize_state_vector, stable_vector_norm
from nwqlib.core.records import Float64, Rational, Record, Source, Text, Real, PositiveInt, Unit
from nwqlib.evidence._work import ExactArithmetic
from nwqlib.evidence.error_model import FramedFact, CheckSpec, CheckDomain, ErrorFrame, _same_scope
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.execution import (
    ExecutionMode,
    KernelApplication,
    VerificationReceipt,
    verification_invocation,
)
from nwqlib.ir import Binding
from nwqlib.operators.access import _check_bytes
from nwqlib.problems.inputs import prepare_qiskit
from nwqlib.operators.inputs import _matvec_requirements
from .adapt_actions import action_sizes, packed_exponential
from .adapt_acquisition import AdaptContext, _native_generator, _ritz_state, basis_chains
from .adapt_records import _exponential_action_work
from .sector import _sector_work, sector_expectations


class AdaptVerificationOptions(Record):
    """Options for checking the Ritz state of an `ADAPT` result.

    Build it with keyword arguments, for example
    `AdaptVerificationOptions(name="adapt", comparisons=("residual", "sector"))`,
    and pass it to `result.verify(checks=...)`. `name` and `comparisons` are
    required. `"residual"` and `"sector"` rebuild the Ritz state of the
    processed Hamiltonian, and `"supplied_reference_energy"` compares two
    recorded numbers. No reference eigensolve runs. A zero residual can
    belong to an excited eigenstate, sector expectations do not prove sector
    membership, and the signed spin contamination keeps its sign. The
    [GCiM guide](../../algorithms/gcim.md) gives the work limits of these
    checks.

    Attributes:
        name: Required. Prefix for the check names of these options.
        comparisons: Required. Distinct nonempty tuple of `"residual"`,
            `"sector"` and `"supplied_reference_energy"`.
        residual_tolerance: Default `1.6e-3`, positive. Residual threshold in
            the problem's energy unit for the non-sampled
            `residual_threshold_met` classification.
        reference_spin: Default `0.0`. Spin quantum number s, a nonnegative
            integer or half-integer, whose `s(s+1)` defines the spin
            contamination.
        reference_energy: Default `None`. Supplied reference energy, with the
            quantity and unit it refers to and its source, required by
            `"supplied_reference_energy"`. It must be a concrete real scalar
            with empty `bindings`, that is, not restricted to particular
            parameter values.
        reference_tolerance: Default `None`. Positive absolute comparison
            threshold for that energy, required by
            `"supplied_reference_energy"`.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Limit on
            the known bytes of the verification workspace.
        max_products: Default `1_000_000_000`. Limit on the counted state and
            operator work of the verification, including the exact synthesis
            of the dense unitaries in a supplied reference circuit.

    Raises:
        ValueError: If `comparisons` is empty or repeats a check, if
            `reference_spin` is not an integer or half-integer, or if
            `"supplied_reference_energy"` lacks `reference_energy` or
            `reference_tolerance`, or has them without that comparison. Also
            if `reference_energy` is not a concrete real scalar with a source
            and empty `bindings`.
    """

    name: Text
    comparisons: tuple[Literal["residual", "sector", "supplied_reference_energy"], ...]
    residual_tolerance: Annotated[Real, Field(gt=0)] = 1.6e-3
    reference_spin: Annotated[Real, Field(ge=0)] = 0.0
    reference_energy: FramedFact | None = None
    reference_tolerance: Annotated[Real, Field(gt=0)] | None = None
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    # Untuned verification workload ceiling (docs/ENGINEERING_CONSTANTS.md).
    max_products: PositiveInt = 1_000_000_000

    def verification_checks(self, result):
        return tuple(CheckSpec(**fields) for fields in self._check_fields(result.plan))

    @model_validator(mode="after")
    def _selected(self):
        """Require a consistent selection of checks before any state is rebuilt.

        The comparisons must be distinct and nonempty. ``reference_spin`` must
        be an integer or half-integer s, and ``s(s+1)`` must be finite when
        the sector check is selected. The supplied-energy comparison needs
        both ``reference_energy`` and ``reference_tolerance``, and that
        energy must be a concrete real scalar with provenance and no point
        bindings.
        """
        if not self.comparisons or len(set(self.comparisons)) != len(self.comparisons):
            raise ValueError("verification requires distinct selected comparisons")
        if self.reference_spin * 2 % 1 != 0:
            raise ValueError("reference_spin must be a nonnegative integer or half-integer")
        if "sector" in self.comparisons and not isfinite(
            self.reference_spin * (self.reference_spin + 1)
        ):
            raise ValueError(
                "reference spin offset must be finite in the selected diagnostic representation"
            )
        supplied = "supplied_reference_energy" in self.comparisons
        if supplied != (self.reference_energy is not None) or supplied != (
            self.reference_tolerance is not None
        ):
            raise ValueError(
                "supplied_reference_energy requires reference_energy and reference_tolerance together"
            )
        if supplied:
            reference = self.reference_energy
            if (
                reference.bindings
                or reference.fact.availability != "concrete"
                or type(reference.fact.value) not in (Rational, Float64)
                or reference.fact.evidence is None
            ):
                raise ValueError(
                    "supplied reference must be a concrete real scalar with provenance and no point restrictions"
                )
        return self

    @property
    def source(self):
        return Source(
            name="adapt.verification." + self.name,
            version="1",
            domain="explicit completed processed Ritz-state diagnostics; ground identity and floating/joint coverage unknown; "
            "native composite decomposition/HLS expansion work and internal workspace unknown; "
            "supplied_reference_energy compares only two recorded numerical scalars exactly; "
            "the complete supplied frame, conditioning and provenance remain in selected options, not a ground-energy claim",
            reference="nwqlib.algorithms.gcim.adapt_verification.verify",
        )

    def _check_fields(self, plan):
        """Yield the ``CheckSpec`` fields for every scalar the selected checks report.

        Each check carries this options' ``content_id`` and the output claim, so
        a fact answers only the options and output claim that produced it. A changed
        threshold then needs a new verification (docs/verification.md, "Facts
        belong to the options that produced them"). Domains encode what each
        scalar can mathematically be. Sector bounds follow from ``q/2`` spatial
        orbitals, each with spin at most 1/2, so ``S_max = q/4``. Only the
        non-sampled residual norm has a threshold, because sampled results
        claim no deterministic spectral interval.
        """
        energy = plan.output.unit or plan.problem.evidence_unit
        dimensionless = Unit(symbol="1", dimension="dimensionless")
        names = []
        if "supplied_reference_energy" in self.comparisons:
            reference = self.reference_energy
            if (
                not reference.frame.unit.same_unit(energy)
                or not _same_scope(reference.frame.scope, plan.problem.evidence_scope)
            ):
                raise ValueError(
                    "supplied reference energy must use the output energy unit and the Problem's evidence scope"
                )
            names.append(
                ("supplied_reference_energy", "absolute_error", energy, self.reference_tolerance)
            )
        if "residual" in self.comparisons:
            names += [
                (
                    "residual_norm",
                    "full_state_residual_norm",
                    energy,
                    None if plan.shots is not None else self.residual_tolerance,
                ),
                (
                    "residual_squared",
                    "squared_residual_norm",
                    Unit(symbol=energy.symbol + "^2", dimension="custom"),
                    None,
                ),
                ("spectral_nearness_lower", "conditional_spectral_nearness_endpoint", energy, None),
                ("spectral_nearness_upper", "conditional_spectral_nearness_endpoint", energy, None),
                ("second_ritz", "second_ritz_gap_proxy", energy, None),
                ("kato_heuristic_lower", "second_ritz_heuristic_endpoint", energy, None),
            ]
            names += [
                (name, "diagnostic_status_flag", dimensionless, None)
                for name in (
                    "square_overflow",
                    "square_underflow",
                    "residual_threshold_met",
                    "pool_limited",
                    "stalled",
                )
            ]
        if "sector" in self.comparisons:
            names += [
                (name, "sector_expectation_diagnostic", dimensionless, None)
                for name in ("particle_number", "spin_z", "spin_squared", "spin_contamination")
            ]
        for name, metric, unit, threshold in names:
            scalar_only = name == "supplied_reference_energy"
            domain = CheckDomain()
            if scalar_only:
                domain = CheckDomain(lower=0.0)
            elif name in {"residual_norm", "residual_squared"}:
                domain = CheckDomain(lower=0.0)
            elif metric == "diagnostic_status_flag":
                domain = CheckDomain(lower=0.0, upper=1.0, integer=True)
            elif name in {"particle_number", "spin_z", "spin_squared", "spin_contamination"}:
                width = plan.problem.dimension.bit_length() - 1
                # Two interleaved spin orbitals form one spatial orbital:
                # empty/double occupation has spin 0, single occupation 1/2.
                # Thus S_max=q/4, not q/2 as for q independent spins.
                maximum = width * (width + 4) / 16
                offset = self.reference_spin * (self.reference_spin + 1)
                bounds = {
                    "particle_number": (0.0, float(width)),
                    "spin_z": (-width / 4, width / 4),
                    "spin_squared": (0.0, maximum),
                    "spin_contamination": (-offset, maximum - offset),
                }
                lower, upper = bounds[name]
                domain = CheckDomain(lower=lower, upper=upper)
            yield dict(
                name=self.name + "." + name,
                domain=domain,
                options_id=self.content_id,
                claim_id=plan.output.content_id,
                frame=ErrorFrame(
                    quantity="recorded_processed_ritz_scalar_vs_supplied_scalar"
                    if scalar_only
                    else "processed_completed_ritz_state",
                    metric=metric,
                    unit=unit,
                    scope=plan.problem.evidence_scope,
                    conditioning=(
                        "exact difference of recorded scalars only; supplied frame and assumptions kept in options "
                        + self.content_id
                    )
                    if scalar_only
                    else plan.output.frame(plan.problem).conditioning,
                ),
                source=self.source,
                threshold=threshold,
                prerequisites=(),
                experiments=0,
                access=(
                    "exact completed basis parameters and projected coordinates; bound processed inputs",
                ),
                classical_work="admitted exact scalar subtraction and absolute value"
                if scalar_only
                else "one normalized state combination; one H action only for residual; sector actions only if selected",
                reference_work="supplied scalar only; no state, operator action or eigensolve"
                if scalar_only
                else "explicit numerical or actual preparation-suffix state action; SDK expansion/internal workspace unknown",
                data_description="framed scalar facts and actual application counts; no state/basis array",
            )


def _binary64_square(value):
    """Return ``(value**2, status)``, with None and the status when binary64 loses it.

    The status is ``binary64_overflow`` for an infinite square and
    ``binary64_underflow`` for a nonzero value whose square rounds to zero,
    so a lost square is never reported as a number.
    """
    square = value * value
    if not isfinite(square):
        return None, "binary64_overflow"
    if value != 0.0 and square == 0.0:
        return None, "binary64_underflow"
    return square, "nonnegative"


def _residual_metrics(sigma, energy, eigenvalues, options, mode, stop_reason):
    """Return the residual metrics of a Ritz pair: the norm, its square with overflow and underflow flags and, for non-sampled results, the nearness interval, the threshold classification and the heuristic Kato-Temple endpoint.

    For a unit vector ``psi`` and real ``E``, ``||(H - E) psi|| >= min_i |lambda_i - E|``,
    so some eigenvalue of H lies in ``[E - sigma, E + sigma]``. This gives the
    ``spectral_nearness`` endpoints, which say nothing about which eigenvalue
    it is. The Kato-Temple bound ``lambda_1 >= E - sigma**2 / (eta - E)``
    needs ``E`` to be the Rayleigh quotient of ``psi`` and a lower bound
    ``eta`` on the next eigenvalue ``lambda_2``. The second Ritz value
    is an upper bound on ``lambda_2``, so ``kato_heuristic_lower`` is a heuristic
    endpoint and not a certified bound. Sampled results report only the norm
    and its square.
    """
    square, status = _binary64_square(sigma)
    metrics = dict(
        residual_norm=sigma,
        residual_squared=square,
        square_overflow=float(status == "binary64_overflow"),
        square_underflow=float(status == "binary64_underflow"),
        spectral_nearness_lower=None,
        spectral_nearness_upper=None,
        second_ritz=None,
        kato_heuristic_lower=None,
        residual_threshold_met=None,
        pool_limited=None,
        stalled=None,
    )
    if mode is ExecutionMode.SHOTS:
        return metrics
    met = sigma <= options.residual_tolerance
    pool_limited = not met and stop_reason in ("gradient_norm_floor", "pool_exhausted")
    metrics.update(
        spectral_nearness_lower=energy - sigma,
        spectral_nearness_upper=energy + sigma,
        residual_threshold_met=float(met),
        pool_limited=float(pool_limited),
        stalled=float(not met and not pool_limited),
    )
    if len(eigenvalues) > 1:
        second = eigenvalues[1]
        gap = second - energy
        metrics["second_ritz"] = second
        if gap > sigma and isfinite(gap):
            # Divide first: the equivalent sigma**2 can overflow spuriously.
            metrics["kato_heuristic_lower"] = energy - sigma * (sigma / gap)
    return metrics


def _lowered_native_circuits(plan, context, selected, counts, *, max_work, max_bytes):
    """Build and lower, once, every circuit that the native Ritz evolution in ``verify`` simulates.

    Returns ``({(): reference, (index, theta): suffix}, synthesis_work)``.
    The reference is the actual preparation circuit of the Plan's reference
    state. Each suffix applies the uncontrolled native gate of one selected
    generator, built by the same factory and cache key as the query gates
    (``_native_generator``). Every circuit is decomposed as for Aer execution
    (``AER_EXECUTION_DECOMPOSE_REPS``). A supplied reference circuit can hold
    a dense ``UnitaryGate``, which Aer applies as its matrix but ``decompose``
    would expand through Qiskit's inexact synthesis, so it is first replaced
    by its exact synthesis. That synthesis is admitted against ``max_work``
    and ``max_bytes`` before it starts, and ``synthesis_work`` is its work
    law (``_dense_synthesis.dense_synthesis_size``), which the caller adds to
    its total. No state is simulated here, so the verification work law can
    charge the instructions these circuits actually contain before any
    statevector exists.
    """
    from qiskit import QuantumCircuit
    from nwqlib.backends.qiskit_aer import AER_EXECUTION_DECOMPOSE_REPS
    from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses, dense_synthesis_size
    from nwqlib.subroutines.qiskit_compat import dense_synthesis_widths, exact_dense_unitaries

    reference = prepare_qiskit(plan._native["reference"], max_bytes=plan.method.max_bytes).circuit
    widths = dense_synthesis_widths(reference)
    synthesis_work = sum(dense_synthesis_size(width)[0] for width in widths)
    if synthesis_work > max_work:
        raise ValueError(f"ADAPT explicit verification needs {synthesis_work} work units for the exact synthesis "
                         "of its supplied reference circuit, beyond the rest of "
                         "AdaptVerificationOptions.max_products. Raise max_products to run this check.")
    admit_dense_syntheses(widths, max_work=synthesis_work, max_bytes=max_bytes,
                          operation="ADAPT explicit verification reference synthesis")
    reference = exact_dense_unitaries(reference)
    lowered = {(): reference.decompose(reps=AER_EXECUTION_DECOMPOSE_REPS)}
    for index, theta in selected:
        key = (index, theta, False, False)
        counts["native_gate_reuses"] += int(key in context.gates)
        counts["native_generator_constructions"] += int(key not in context.gates)
        gate = _native_generator(
            index,
            theta,
            controlled=False,
            inverse=False,
            inputs=plan._native["inputs"],
            compiler_plans=plan._native["compiler_plans"],
            context=context,
        )
        suffix = QuantumCircuit(plan.reconstruction.num_qubits)
        suffix.append(gate, suffix.qubits, copy=False)
        lowered[(index, theta)] = suffix.decompose(reps=AER_EXECUTION_DECOMPOSE_REPS)
    return lowered, synthesis_work


def _statevector_evolution_work(circuit, dimension):
    """Return the scalar operations of evolving a ``dimension``-amplitude state through ``circuit``.

    This follows Qiskit's ``Statevector._evolve_instruction``. A nonzero
    global phase multiplies every amplitude once. A barrier does nothing. A
    gate with a matrix on ``k`` qubits costs one unit for its copy in the
    instruction that ``evolve`` or ``from_instruction`` converts the circuit
    to (a gate reached through a definition is not copied, so this
    overcharges it by one), ``4**k`` to form its
    ``2**k``-by-``2**k`` matrix, ``dimension * 2**k`` multiply-adds to apply
    it, and ``2 * dimension`` for the at most two copies of the state that
    Qiskit makes when it transposes the gate's qubits to the front of the
    state and back (``Statevector._evolve_operator``). Any other instruction is
    evolved through its definition, and one without a definition cannot be
    simulated.
    """
    from qiskit.circuit import Gate

    work = dimension if circuit.global_phase else 0
    for item in circuit.data:
        operation = item.operation
        if operation.name == "barrier":
            continue
        if isinstance(operation, Gate) and hasattr(operation, "__array__"):
            k = operation.num_qubits
            work += 1 + 4**k + dimension * (2**k + 2)
        elif operation.definition is not None:
            work += _statevector_evolution_work(operation.definition, dimension)
        else:
            raise ValueError(f"native verification cannot simulate instruction {operation.name}")
    return work


def _largest_evolved_gate(circuit):
    """Return the largest qubit count ``k`` of a gate matrix that evolving ``circuit`` forms.

    The census follows ``_statevector_evolution_work``: a gate with a matrix
    counts its own width, a barrier nothing, and any other instruction the
    gates of its definition.
    """
    from qiskit.circuit import Gate

    largest = 0
    for item in circuit.data:
        operation = item.operation
        if operation.name == "barrier":
            continue
        if isinstance(operation, Gate) and hasattr(operation, "__array__"):
            largest = max(largest, operation.num_qubits)
        elif operation.definition is not None:
            largest = max(largest, _largest_evolved_gate(operation.definition))
    return largest


def _state_frontier(length):
    """Return ``F(l)``, the full-vector frontier of streaming a Ritz state of ``l`` selections.

    A multi-gate native suffix can hold the current state, a transposed or
    reshaped copy and a contraction result together, three additional full
    vectors during an evolution (Qiskit's ``Statevector._evolve_operator``).
    The scalar multiply-add into the Ritz state needs one temporary. For
    ``l >= 2`` six vectors are live while a later single is evolved with the
    reference, single 1 and the accumulator, and during the last product
    evolution at most ``l - 2`` later-needed product prefixes, the
    accumulator and three evolution vectors, ``l + 2``. So ``F(0) = 3``,
    ``F(1) = 5`` and ``F(l) = max(6, l + 2)``. These are upper frontiers, not
    a promise that every gate layout allocates that many arrays. The final
    normalizer's two complex arrays and Boolean scan scratch lie below it
    apart from its one-byte mask, and the classical two-buffer Taylor action
    fits within the same frontier, with its shared action remainder and the
    packed tables charged separately.
    """
    if length == 0:
        return 3
    if length == 1:
        return 5
    return max(6, length + 2)


def verify(plan, result, *, options):
    """Compare the completed Ritz state, or a supplied scalar reference, under bounded verification work.

    The state checks rebuild the normalized Ritz state
    ``psi = sum_i c_i |b_i>`` of the value basis (``basis_chains``) from the
    stored pencil coefficients. ``residual`` reports ``||(H - E) psi||`` and
    the derived metrics of ``_residual_metrics``. ``sector`` reports
    ``<N>``, ``<S_z>``, ``<S^2>`` and the spin contamination
    (``sector_expectations``). ``supplied_reference_energy`` reports the
    exact absolute difference of two recorded scalars and builds no state.
    Classical plans reuse the prefix vectors saved with the Result when they
    are present and build the missing ones by generator exponentials.
    Quantum plans build them from the native gates. Both stream the Ritz
    combination (``adapt_acquisition.stream_ritz``): a basis or prefix array
    is released after its final contribution and successor evolution.

    Args:
        plan: The ADAPT Plan that produced ``result``.
        result: A completed ``ADAPTResult`` with a projected value.
        options: The selected ``AdaptVerificationOptions``.

    Returns:
        ``(receipt, facts)``: the ``VerificationReceipt`` with the actual
        application counts, and one ``FramedFact`` per reported scalar, in the
        order of ``options.verification_checks``. A concrete value is
        witnessed by the receipt. A value that is missing or not finite has
        availability ``unknown``.
    """
    if type(options) is not AdaptVerificationOptions:
        raise TypeError("ADAPT checks require AdaptVerificationOptions")
    if result.eigenvalue is None:
        raise ValueError("verification requires a completed projected value")
    checks = options.verification_checks(result)
    selected = tuple(zip(result.value_selected, result.value_theta, strict=True))
    state_checks = bool(set(options.comparisons) & {"residual", "sector"})
    if state_checks and (
        result.pencil is None or len(result.pencil.coefficients) != len(basis_chains(selected))
    ):
        raise ValueError(
            "state verification needs the actual completed pencil and basis coordinates"
        )
    if "sector" in options.comparisons and plan.reconstruction.num_qubits % 2:
        raise ValueError("sector diagnostics require an even interleaved spin-orbital width")
    counts = dict(
        reference_materializations=0,
        generator_exponentials=0,
        hamiltonian_actions=0,
        generator_actions=0,
        scalar_products=0,
        theory_vector_reuses=0,
        ritz_combinations=0,
        native_reference_materializations=0,
        native_suffix_evolutions=0,
        native_prefix_reuses=0,
        native_gate_reuses=0,
        native_generator_constructions=0,
        sector_bundles=0,
        supplied_scalar_comparisons=int("supplied_reference_energy" in options.comparisons),
    )
    metrics = {}
    if "supplied_reference_energy" in options.comparisons:
        arithmetic = ExactArithmetic()
        difference = abs(
            arithmetic.fraction(result.eigenvalue)
            - arithmetic.fraction(options.reference_energy.fact.value)
        )
        metrics["supplied_reference_energy"] = Rational(
            numerator=difference.numerator, denominator=difference.denominator
        )
    # Residual/sector checks need an explicitly reconstructed completed Ritz
    # state. A supplied scalar comparison alone does not incur that state work.
    if state_checks:
        rec = plan.reconstruction
        d = 1 << rec.num_qubits
        l = len(selected)
        context = AdaptContext()
        saved = result.data.method_context
        if saved is not None:
            context.vectors = dict(saved["vectors"])
        # Known-buffer admission of the streamed state checks, before any
        # circuit is lowered or state built: B_states +
        # reference_action_bytes + B_gate + B_other_action_metadata, with
        # B_states = 16 d max{F(l), 8 [residual], 32 [sector]} + d. The
        # residual allowance holds the Ritz state, H@state, the scalar
        # product/subtraction arrays and the input and output of the H action.
        # The 32-vector sector allowance covers sector_expectations' normalized copy and probabilities,
        # apply_spin_squared's diagonals and index temporaries, both
        # sequential ladder actions and the final products. Each applicable
        # action adds its shared allowance less the 32d of its input and
        # output, max_A(B_A - 32d) (adapt_actions.action_sizes): on a
        # classical Plan over its packed generator and H actions, beside
        # the packed tables themselves, and on a quantum Plan over the H
        # action when residual verification is requested. This is incremental
        # verification workspace: max_bytes also includes the arrays the
        # saved Result already owns, whose distinct buffers are added once
        # (conservatively, whether or not the Ritz state uses them). The native gate term B_gate joins
        # after the lowered-instruction census below, and the reference
        # synthesis keeps its own admission.
        allowance = max(_state_frontier(l), 8 * ("residual" in options.comparisons),
                        32 * ("sector" in options.comparisons))
        inputs = plan._native["inputs"]
        if plan.execution == "classical":
            other_bytes = inputs.cache["action_tables"]["resident_bytes"] + max(
                (size - 32 * d for size in action_sizes(inputs, d)), default=0)
        elif "residual" in options.comparisons:
            other_bytes = _matvec_requirements(inputs.hamiltonian)[1] - 32 * d
        else:
            other_bytes = 0
        owned = {}
        if saved is not None:
            for name in ("vectors", "h_columns"):
                for array in saved[name].values():
                    owned.setdefault(id(array), array.nbytes)
        known_bytes = 16 * d * allowance + d + rec.reference_action_bytes + other_bytes + sum(owned.values())
        _check_bytes(known_bytes, options.max_bytes, "ADAPT explicit verification")
        # Work, in the units of _exponential_action_work: for _ritz_state, one
        # zero fill, then a scaling and an addition for each basis state (2l
        # of them, or the reference alone when l = 0) and nine
        # renormalization passes.
        work = d * (2 * len(basis_chains(selected)) + 10)
        lowered = None
        if plan.execution == "classical":
            # The reference preparation and one generator exponential per
            # built state. Each selected generator enters one single-generator
            # state and at most one step of the product prefixes, so two
            # exponentials per generator bound the 2l - 1 that are built.
            work += rec.reference_action_work if rec.reference_action_work is not None else 0
            for index, theta in selected:
                work += 2 * _exponential_action_work(rec.pool[index], theta, d)
        else:
            # The native evolution starts the reference from |0> (d) and evolves it
            # through its lowered preparation circuit, then builds every
            # distinct nonempty prefix of the basis chains by evolving its own
            # prefix through the lowered suffix of its last generator. The
            # circuits are built and lowered here, before any state exists,
            # so the law charges the instructions they actually contain.
            # Building them is checked only in part. prepare_qiskit checks the
            # reference payload against the Method's max_bytes and a direct
            # vector preparation against the default max_direct_amplitudes of
            # 2**16 amplitudes. The at most l generator circuits come from
            # compiler plans that were checked against the Method's max_bytes
            # and max_products when their generators were first selected, and
            # they stay within the family-level slot envelope that the Plan's
            # construction_work declares (adapt.generator_pool_slot_bound).
            # That declaration is a reported envelope, not an enforced limit,
            # and Qiskit's decomposition has no NWQLib law at all. The exact
            # synthesis of a dense unitary in a supplied reference circuit is
            # charged to max_products before it starts and joins the total.
            if saved is not None:
                # The Result's snapshot holds the compiler plans its Run built,
                # which a Plan reopened from a Run archive written before the
                # selections lacks. Adopting them compiles no generator again.
                plan._native["compiler_plans"].restore(saved["compiler_plans"])
            lowered, synthesis = _lowered_native_circuits(
                plan, context, selected, counts, max_work=options.max_products - work, max_bytes=options.max_bytes)
            work += synthesis
            evolution = {key: _statevector_evolution_work(c, d) for key, c in lowered.items()}
            largest_gate = max(_largest_evolved_gate(c) for c in lowered.values())
            prefixes = {
                chain[:end] for chain in basis_chains(selected) for end in range(1, len(chain) + 1)
            }
            work += d + evolution[()] + sum(evolution[prefix[-1]] for prefix in prefixes)
        # A residual adds one H action and 3d for the scaling by E, the
        # subtraction and the norm. The sector bundle adds _sector_work.
        if "residual" in options.comparisons:
            work += rec.hamiltonian_action_work + 3 * d
        if "sector" in options.comparisons:
            work += _sector_work(rec.num_qubits)
        if work > options.max_products:
            raise ValueError(
                f"ADAPT explicit verification needs {work} scalar operations on "
                f"{rec.num_qubits}-qubit states, exceeding "
                f"AdaptVerificationOptions.max_products={options.max_products}. "
                "Raise max_products to run this full-state check."
            )
        if lowered is not None:
            # B_gate = 16*4**k_max covers the largest lowered native gate
            # matrix, with k_max from the census of the circuits whose
            # evolution work is charged above.
            _check_bytes(known_bytes + 16 * 4**largest_gate, options.max_bytes, "ADAPT explicit verification")
        coefficients = tuple(complex(z.real, z.imag) for z in result.pencil.coefficients)
        if plan.execution == "classical":
            generators = plan._native["inputs"].cache["action_tables"]["generators"]

            def reference_at():
                if () in context.vectors:
                    counts["theory_vector_reuses"] += 1
                    return context.vectors[()]
                from nwqlib.algorithms._eigen_inputs import state_direction

                counts["reference_materializations"] += 1
                return state_direction(plan._native["reference"], max_bytes=plan.method.max_bytes)

            def evolve(previous, items):
                # The saved prefix, or one exponential normalized where _vector
                # normalizes, without keeping it past its last use.
                if items in context.vectors:
                    counts["theory_vector_reuses"] += 1
                    return context.vectors[items]
                index, theta = items[-1]
                counts["generator_exponentials"] += 1
                return normalize_state_vector(packed_exponential(generators[index], previous, theta))[0]
        else:
            from qiskit.quantum_info import Statevector

            def reference_at():
                counts["native_reference_materializations"] += 1
                return np.asarray(Statevector.from_instruction(lowered[()]).data, dtype=complex)

            def evolve(previous, items):
                # Evolve the prefix state by the actual native gate of the last
                # generator, decomposed as for Aer execution. This avoids a
                # nominal generator action and a second full-prefix simulation.
                counts["native_suffix_evolutions"] += 1
                return np.asarray(Statevector(previous).evolve(lowered[items[-1]]).data)
        reuses = {}
        state = _ritz_state(selected, coefficients, reference_at, evolve, reuses)
        counts["ritz_combinations"] += 1
        if lowered is not None:
            counts["native_prefix_reuses"] += reuses.get("reuses", 0)
            # Each selected gate was built once above. Every later evolution
            # through the same suffix reuses it.
            counts["native_gate_reuses"] += counts["native_suffix_evolutions"] - (len(lowered) - 1)
        if "residual" in options.comparisons:
            residual = (
                plan._native["inputs"].hamiltonian.matvec(
                    state, max_bytes=options.max_bytes, max_products=options.max_products
                )
                - result.eigenvalue * state
            )
            counts["hamiltonian_actions"] += 1
            mode = (
                ExecutionMode.SHOTS
                if plan.shots
                else ExecutionMode.THEORY
                if plan.execution == "classical"
                else ExecutionMode.STATEVECTOR
            )
            metrics.update(
                _residual_metrics(
                    stable_vector_norm(residual),
                    result.eigenvalue,
                    result.pencil.eigenvalues,
                    options,
                    mode,
                    result.stop_reason,
                )
            )
        if "sector" in options.comparisons:
            counts["sector_bundles"] += 1
            sector = sector_expectations(
                state, num_qubits=rec.num_qubits, reference_spin=options.reference_spin
            )
            metrics.update(
                (name, sector[name])
                for name in ("particle_number", "spin_z", "spin_squared", "spin_contamination")
            )
    # Frame each observed diagnostic separately and keep unavailable values
    # explicit. A small Ritz residual does not identify the ground eigenspace.
    raw = []
    for check in checks:
        value = metrics.get(check.name.removeprefix(options.name + "."))
        fields = dict(
            quantity=check.name,
            unit=check.frame.unit,
            scope=check.frame.scope,
            assumptions=(
                "processed completed Ritz state; residual does not identify ground energy",
                "nominal floating state action; the work law excludes Qiskit decomposition and SDK workspace",
            ),
        )
        if value is None or not isinstance(value, Rational) and not isfinite(value):
            fields.update(
                availability="unknown",
                reason="selected diagnostic lacks joint evidence or finite representation",
            )
        else:
            fields.update(
                availability="concrete",
                value=value if isinstance(value, Rational) else Float64(value=float(value)),
                evidence=Evidence(kind="numerical_estimate", source=options.source),
            )
        raw.append(FramedFact(frame=check.frame, bindings=(), fact=Fact(**fields)))
    application = KernelApplication(
        name=options.name,
        implementation=options.source,
        arguments=tuple(Binding(parameter=name, value=value) for name, value in counts.items()),
        facts=tuple(raw),
    )
    receipt = VerificationReceipt(
        plan_id=plan.content_id,
        result_id=result.content_id,
        invocation_id=verification_invocation(),
        construction_id=plan.construction.content_id,
        artifact_ids=(),
        reference=options.source,
        options_id=options.content_id,
        applications=(application,),
    )
    witnessed = []
    for check, fact in zip(checks, raw, strict=True):
        if fact.fact.evidence is None:
            witnessed.append(fact)
        else:
            evidence = Evidence(
                kind="numerical_estimate",
                source=options.source,
                status="witnessed",
                artifact_kind="verification_receipt",
                artifact=receipt.content_id,
                subject_id=result.content_id,
                witnessed_scope=fact.frame.scope,
                options_id=receipt.options_id,
                check_id=check.content_id,
            )
            witnessed.append(fact.revise(fact=fact.fact.revise(evidence=evidence)))
    return receipt, tuple(witnessed)
