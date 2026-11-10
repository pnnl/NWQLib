"""Planning, product-formula steps, bias bounds and the row-bound circuit block of the sampled-pair LCHS observable.

``LCHS(estimator="sampled_pairs")`` estimates the quadratic form,
norm squared or normalized expectation of the selected LCHS vector from
pair rows (`sampled_rows`, `sampled_moments`, `sampled_native`). For
product-formula branches the generator A must be a Pauli sum or a Cartesian
input with Pauli or zero parts, since other parts have no Pauli exponential
here; the QSP bank below takes more forms. A constant source
b may act with the initial state or alone. The applications come from
``source_selection.application_facts``. With ``SampledPairs.source_time="discrete"``
they are the initial state and the ``duhamel_nodes`` Gauss-Legendre nodes
r_q of the source integral on [0, T] (An, Childs and Lin,
arXiv:2312.03916v2, Eq. (2)), each acting for the elapsed time T - r_q with
mass w_q*||b||*exp(s*(T - r_q)). With ``"continuous"`` the source is one
category of mass ||b||*integral_0^T exp(s*t) dt whose elapsed time is drawn
from the tilted density (`sampled_rows.source_times`).

Program. One shared body allocates the branch qubit, any control copies,
the system and, for a sampled QSP Plan, its success flags, calls the pair
block with the integer parameter ``row``, measures the branch qubit into bit
0, system qubit j into bit 1 + j and flag f into bit 1 + q + f, and releases
the wires. The flag width is one Plan-wide layout, the largest QSP workspace
of the support; a row with a smaller workspace leaves the remaining flags at
zero, and every flag must be zero for a shot to score (`sampled_qsp`).
Without a source, the state's own preparation block first prepares the
initial direction on the system; it commutes with H on
the branch qubit, so the pair block holds no input preparation. With a
source, the pair block prepares each row's inputs: one uncontrolled
preparation for equal kinds and, for different kinds, each direction under
the branch value of its slot, which keeps their physical relative phase
(`sampled_native.build_pair_circuit`). One `MeasurementBatch` has one
`Setting`, which binds nothing, one range axis over ``row`` in [0, B) and
``SampledPairs.repeats`` repetitions, and the Plan has one `Experiment`, so its
records do not grow with B. Each point of the axis is one row and one outer
cluster of `sampled_moments`, and a static Run prepares and runs it once
under the key ``pairs[row=i]``. Every number of a row
lives once in the row table of the Plan's reconstruction, and the block
constructor reads it there. Planning draws the rows once from the Method's
random stream, and loading, estimation and analysis never draw them again.

Product-formula steps. Branch x of a row evolves under t_x*(k_x*L_c + H_c)
by r order-p steps of the sorted Pauli labels that the pair SELECT emits,
with operator error delta_x <= Wbar(k_x)*t_x**(p+1)/r**p, where Wbar is the
Pauli-triangle census of the coefficients k*l_P + h_P (`_census`; Childs et
al., doi:10.1103/PhysRevX.11.011020, Propositions 9-10). Nodes are split
into predeclared |k| buckets with upper ends kappa_b and step counts r_b,
and a row runs r = max(r_b(x), r_b(y)) steps, so
``delta_x <= Wbar_b(x)*t_x**(p+1)/r_b(x)**p`` whatever the partner.
`LCHSPairBias` gives the two resulting bias relations: ``"finite_sum"`` for
finite applications and discrete nodes, where the integer probabilities
cancel in E[|w|*delta], and ``"bucket_maximum"`` for a continuous source
time or kernel node, which uses the largest bucket value
E_sup = T_e**(p+1)*max_b Wbar_b/r_b**p and so needs no bucket probability.
The probabilities of the floating-point midpoint law of a continuous draw
are not those of the analytic distribution function, which is why the
second relation is used there. For a continuous kernel Wbar_b is the census
of the majorant coefficients kappa_b*|l_P| + |h_P|, formed with the same
two roundings as the circuit's coefficients, so it bounds the census of
every rounded node coefficient vector of the bucket: each census term is a
product of coefficient magnitudes with a nonnegative weight, or for the
relaxed order-2 census such a product with the minimum of two
nondecreasing linear forms of the magnitudes, so the census only grows
with every magnitude.

Fixed steps use one bucket and the Method's ``Trotter.steps`` and publish
their bound. Weighted steps meet the per-unit allowance epsilon of
``trotter_error_budgeted`` on E. ``"finite_sum"`` requires
``T_e**(p+1)*sum_b M_b/r_b**p <= epsilon*beta`` with
M_b = sum_{j in b} |c_j|*Wbar_j and beta = sum_j |c_j|: one bucket takes the
smallest common r, and several start from the continuous relaxation that
minimizes sum_b P_b*r_b under that constraint for the exact bucket
probabilities P_b, r_b proportional to (M_b/P_b)**(1/(p+1)), round up and
add a step to every bucket until the exact rational check holds.
``"bucket_maximum"`` gives each bucket the smallest r_b with
``T_e**(p+1)*Wbar_b/r_b**p <= epsilon``, which meets E_sup <= epsilon with
the fewest steps in every bucket. A schedule above a set ``TrotterErrorBudgeted.max_steps``
is refused with the bound it reaches at the cap.

Recipes. ``SampledPairs.product_formula`` other than ``"plain"`` selects the
Richardson, PTSC or two-member MPF recipe of `sampled_recipes` from
support-wide facts (B, K_*, T_e and the Pauli one-norms) before any row is
drawn, admits its saved arrays and generation work, draws the plain row
table and then the recipe choices, and replaces the census buckets by the
recipe's own bias relation. The model, source and finite-bit bounds of the
plain route are shared (`_model_bias`). Every row's circuit is the word
`_row_words` gives from the saved choices.

QSP bank. With ``ham_evo="qsp_block_encoding"`` each
branch runs a polynomial of a finite phase bank instead (`sampled_qsp`), on
encodings of the parts of A: native Pauli or band parts, supplied encodings
of Hermitian parts, or a general A given as one supplied encoding; the bank
is planned with the laws, prepared once before the rows are drawn, and its
bias replaces the product-formula stage.

Every evolving sampled route prepares each input direction with the state's
own preparation, so it refuses ``init_prep="mps_circuit"``
before the common stage rather than ignore it.

Accuracy. An ``accuracy`` request fixes B before any draw from the
support-wide score range (`_sufficient_rows`): one deterministic pass
(`_forecast`) in ``LCHS.sampling_shots`` and the same pass again in
planning, which checks the count and then prepares and draws once.
"""

from nwqlib._limits import ResourceLimitError, exceeds, memory_limit
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, exp, isfinite, ldexp, log, nextafter, pi, remainder, sqrt, tau

import numpy as np

from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.blocks.records import BlockSemantics, SelectedConstruction, SelectedDefinition
from nwqlib.blocks.selection import NATIVE_POSITION_BYTES, SelectedBlock, _source, select_preparation
from nwqlib.core.planning import Experiment, Plan, ReadoutDetails
from nwqlib.core.records import InputRef
from nwqlib.evidence.records import Evidence
from nwqlib.ir import (Allocate, Argument, BlockCall, BlockSignature, ClassicalValue, Definition, ExprRef,
                       Expression, Measure, MeasurementBatch, MetadataRef, Parameter, ParameterRef, PortMap,
                       Program, QuantumPort, RangeAxis, Register, Release, Sequence, Setting)
from nwqlib.operators.access import _check_bytes
from nwqlib.problems.inputs import PhysicalScale, compose_recovery
from nwqlib.problems.records import NormalizedExpectation, NormSquared, QuadraticForm
from nwqlib.resources.records import ResourceLaw, Workspace
from .primary_records import ASSERTED_L_BOUNDS, METHOD, LCHSPairBias, LCHSReconstruction
from .route_options import route_name
from .sampled_rows import DiscreteKernel, draw_pair_rows, kernel_law

PAIR_TARGET = "lchs.sampled_pair"
BATCH = "pairs"
# Coarse bytes of a prepared row's handle, preparation record and values
# (_pair_memory), which dominate a short row. Sampled LCHS Runs grew 31-34 KB per
# row linearly to 1e5 rows. Registered in ENGINEERING_CONSTANTS.
_ROW_HANDLE_BYTES = 32 << 10
_TOTAL = 1 << 52
_HALF_BIT = 2.0 ** -53


@dataclass(frozen=True)
class LCHSSampledPayload:
    """Circuit data of the sampled-pair block that every row shares.

    The rows and their step buckets are in the Plan's reconstruction and
    the input states in its Problem, which the block receives beside this
    payload.

    Attributes:
        num_system_qubits: System qubits q.
        l_terms: Pauli table ``(label, coefficient)`` of L_c = L - c_L*I,
            real coefficients in Qiskit label order, without the identity
            label, whose coefficient is part of the scalar phase.
        h_terms: Pauli table of H_c = H - c_H*I in the same form.
        order: Product-formula order, 1 (Lie) or 2 (symmetric Suzuki).
        implementation: SELECT construction of `sampled_native`,
            ``"multiplexor"`` or ``"structured"``.
        control_copies: Copies of the branch value that control the SELECT.
        term_labels: Labels of the Plan's observable terms, in the order of
            ``LCHSReconstruction.terms``.
        prepares_inputs: Whether the pair block prepares each row's input
            directions, as it does for a Problem with a constant source.
        max_bytes: Byte limit of those input preparations.
        qsp_bank: The `sampled_qsp_bank.PreparedSampledQSPBank` of a
            sampled QSP Plan (``implementation="qsp"``), prepared once at
            planning, or None. A sampled QSP Plan whose generator vanishes
            on the whole support keeps None and runs no evolution.
        l_child, h_child: The branch children of a sampled QSP Plan
            (`sampled_qsp`, Children): a centered native plan, the operator
            input of a supplied Hermitian part, or a single-A plan of a
            general supplied A; None for an exact zero or a product-formula
            Plan.
    """

    num_system_qubits: int
    l_terms: tuple
    h_terms: tuple
    order: int
    implementation: str
    control_copies: int
    term_labels: tuple
    prepares_inputs: bool = False
    max_bytes: int = 0
    qsp_bank: object = None
    l_child: object = None
    h_child: object = None


def _centered_terms(operator, identity, max_bytes):
    """Return the nonzero real non-identity Pauli terms of one Hermitian part, or () for an exactly zero part."""
    if operator is None:
        return ()
    terms = []
    for label, value in operator.pauli_terms().labels(max_bytes=max_bytes):
        value = complex(value)
        if value.imag != 0.0:
            raise ValueError("a Hermitian part of A has a Pauli coefficient with a nonzero imaginary part")
        if label != identity and value.real != 0.0:
            terms.append((label, float(value.real)))
    return tuple(terms)


def _census(labels, rows, num_qubits, method, held):
    """Return a rational outward bound on the Pauli-triangle census W of each key of ``rows`` and the census work.

    ``rows`` maps a key to its ``(label, coefficient)`` rows in the order of
    ``labels``. A key without rows, or a generator without labels, has W = 0.
    """
    from .time_independent_terms import _node_bound_coefficients
    present = {key: value for key, value in rows.items() if value}
    if not labels or not present:
        return {key: Fraction(0) for key in rows}, 0
    evaluations, ledger = _node_bound_coefficients(labels, num_qubits, present, tuple(present),
        order=method.ham_evo.order, max_work=method.max_select_work, max_bytes=method.max_bytes,
        census_held=held)
    return {key: Fraction(evaluations[key].coefficient) if key in present else Fraction(0) for key in rows}, \
        ledger["work"]


# ponytail: equal-width |k| buckets; a quantile or geometric split would suit
# heavy-tailed node laws better if bucket step savings matter.
def _bucket_edges(top, count):
    """Return ``count`` equal-width upper ends in |k| on [0, top], the last exactly ``top``, or one end for top = 0."""
    if top == 0.0:
        return (0.0,)
    return tuple(top * (index + 1) / count for index in range(count - 1)) + (top,)


def _application_facts(method, problem, generator, shift):
    """Return the shared application facts of the sampled route (module docstring).

    The children's extracted phase uses (c_L + s, c_H), and
    ``application_facts`` takes the identity coefficients (c_L, c_H)
    themselves. The Gauss-Legendre rule of a discrete source is checked
    against the limits of the coherent constant-source route: 128*n bytes
    and n**2 + 8*n work units for its n nodes.
    """
    from .inhomogeneous_theory import duhamel_quadrature
    from .source_selection import application_facts
    final = float(problem.elapsed_time)
    common = dict(final_time=final, signed_shift=shift, initial_norm=problem.initial_state.preparation.physical_scale,
                  scalars=(generator.dissipative_extracted - shift, generator.hamiltonian_extracted))
    if problem.source is None:
        return application_facts(kind="homogeneous", **common)
    source = problem.source.preparation.physical_scale
    if method.estimator.source_time == "continuous":
        return application_facts(kind="continuous_source", source_norm=source, source_interval=(0.0, final), **common)
    nodes = weights = ()
    if source.mantissa > 0:
        count = method.duhamel_nodes
        _check_bytes(128 * count, method.max_bytes, "sampled LCHS source quadrature")
        work, cap = count**2 + 8 * count, method.max_quadrature_work
        if exceeds(work, cap):
            raise ResourceLimitError(f"the sampled source quadrature needs {work} work units (n**2 + 8*n for n={count} "
                                     f"duhamel_nodes), above LCHS.max_quadrature_work={cap}")
        nodes, weights = duhamel_quadrature(final, count)
    return application_facts(kind="discrete_source", source_norm=source, source_nodes=nodes, source_weights=weights,
                             **common)


def _held_bytes(problem, output, common, data):
    """Return H, the known numerical payload held from selection until the Plan is assembled.

    H adds, in logical value bytes:

    - the declared ``manifest.payload_bytes`` of A, of its selected parts L
      and H, of the initial and source states and of the observable. Every
      input this route accepts declares its size. A part shared with A is
      counted twice, which only widens H.
    - the raw grid arrays ``data.quadrature.k_nodes`` and ``coefficients``.
    - the raw coefficient plan ``data.quadrature.coefficient_plan``: 8 bytes
      per node and base weight, which its quadrature rule shares, and 16 per
      raw coefficient and per kernel value when the rule supplies them.
    - the separate compensated coefficient tuple of the homogeneous plan
      ``data.coefficient_plan``, 16 bytes per value. Its nodes, weights and
      rule are those of the raw plan.

    A continuous kernel builds no grid (``data`` None) and holds only the
    inputs; a discrete kernel uses the grid's own lengths, not the sampled
    population. These widths are value bytes, not the size of boxed Python
    numbers. Python objects such as the
    centered Pauli tables, the observable term tuple and the application
    facts are outside this count.
    """
    inputs = [problem.A, common.generator.dissipative, common.generator.hamiltonian, problem.initial_state,
              problem.source, getattr(output, "observable", None)]
    held = sum(item.manifest.payload_bytes for item in inputs if item is not None)
    if data is None:
        return held
    raw, homogeneous = data.quadrature.coefficient_plan, data.coefficient_plan
    kernel_values = raw.quadrature.kernel_values
    kernel_count = 0 if kernel_values is None else len(kernel_values)
    return (held
            + data.quadrature.k_nodes.nbytes + data.quadrature.coefficients.nbytes
            + 8 * (len(raw.nodes) + len(raw.base_weights))
            + 16 * (len(raw.coefficients) + kernel_count + len(homogeneous.coefficients)))


def _row_bytes(clusters, facts, kernel, terms, buckets, qsp=False):
    """Return R = 1024*B + 256*(A + M + O + K), plus 128*B for sampled QSP rows, the known-array allowance of the laws, rows and buckets.

    B is the row count, A the application categories (two for a continuous
    source, its initial atom and source, even when a mass is zero), M the
    discrete kernel slots including zero coefficients (0 for a continuous
    kernel), O the observable terms and K the requested buckets, an upper
    bound on the bucket count. Allocated lengths are used throughout.

    - 256 bytes per category, kernel slot or term: 128 for one
      ``integer_categorical`` call, whose input logarithms, relative values,
      targets, counts, sort order, gathered copies and masks need fewer than
      twelve eight-byte vectors and eight masks at once, and 96 for the
      caller's tables of that population (category kinds, times, ratios and
      counts; kernel copy, nodes, magnitudes, ratios, phases and counts;
      observable values, ranges and counts). ``kernel_law``'s lift holds a
      16M complex copy with an 18M scan, then 24M, keeping 8M; it ends
      before the law starts and fits the same 256M. The second kernel law,
      in `_pair_bias`, runs after the first one's workspace is released.
    - 256 bytes per bucket for its edges, steps and index arrays.
    - 1024 bytes per row: 480 for the twelve saved eight-byte fields, their
      base64 Record JSON (4/3 of the bytes) and that text's encoded copy
      beside the content hash, 96*(1 + 3*4/3); 256 + 16 for 32 eight-byte
      and 16 one-byte vectors of the drawing tables, draws, gathers and
      frame scratch; and 160 + 8 for 20 eight-byte and 8 one-byte vectors of
      ``source_times``, whose two calls are sequential. These add to 920.

    A sampled QSP row adds 128 bytes: its saved bucket and two member
    indices, 24 bytes with their base64 Record JSON and encoded copy,
    24*(1 + 3*4/3) = 120, rounded up; K is then the bank bucket count.

    This is a visible-array allowance, not a heap bound: NumPy's sort
    workspace, Python objects, allocator overhead and the nested Plan JSON
    beyond the row and record models are not covered.
    """
    categories = 2 if facts.kind == "continuous_source" else len(facts.applications)
    slots = len(kernel.coefficients) if isinstance(kernel, DiscreteKernel) else 0
    return (1152 if qsp else 1024) * clusters + 256 * (categories + slots + len(terms) + buckets)


def _application_masses(facts):
    """Return ``(mass, elapsed time)`` of every nonzero application category, time None for a continuous source."""
    items = [(item.mass, float(item.elapsed_time)) for item in facts.applications]
    if facts.kind == "continuous_source" and facts.source_mass is not None and facts.source_mass.mantissa > 0:
        items.append((facts.source_mass, None))
    return items


def _select_steps(method, relation, values, probabilities, norm, time_power):
    """Return the step count of every bucket and the allowance they meet (module docstring).

    ``values`` are M_b for ``"finite_sum"`` and Wbar_b for
    ``"bucket_maximum"``, ``norm`` is beta and ``time_power`` T_e**(p+1),
    all exact rationals. ``probabilities`` P_b only shape the finite-sum
    relaxation.

    Raises:
        ResourceLimitError: When the schedule needs more than a set
            ``max_steps`` steps of the product-formula backend.
    """
    from nwqlib.subroutines.trotterization.error_budget import _smallest_step_count, _upward_float
    from .time_independent_terms import _lchs_trotter_error_budget_per_node
    formula = method.ham_evo
    order = formula.order
    if formula.kind == "trotter":
        return (formula.steps,) * len(values), None
    allowance = _lchs_trotter_error_budget_per_node(method)
    if relation == "bucket_maximum":
        steps = tuple(_smallest_step_count(time_power * value, allowance, order) if value else 1 for value in values)
    elif len(values) == 1:
        steps = (_smallest_step_count(time_power * values[0] / norm, allowance, order) if values[0] else 1,)
    else:
        shapes = [float(value / Fraction(probability)) ** (1.0 / (order + 1)) if value and probability else 0.0
                  for value, probability in zip(values, probabilities)]
        total = sum(float(value) * shape ** -order for value, shape in zip(values, shapes) if shape)
        factor = (float(time_power) * total / (allowance * float(norm))) ** (1.0 / order) if total else 0.0
        steps = [max(1, ceil(factor * shape)) for shape in shapes]
        target = Fraction(allowance) * norm
        while time_power * sum(value / Fraction(step) ** order for value, step in zip(values, steps)) > target:
            steps = [step + 1 if value else step for value, step in zip(values, steps)]
        steps = tuple(steps)
    cap = formula.max_steps
    if cap is not None and max(steps) > cap:
        capped = [min(step, cap) for step in steps]
        if relation == "bucket_maximum":
            reached = time_power * max(value / Fraction(step) ** order for value, step in zip(values, capped))
            name, remedies = "largest bucket bound T_e**(p+1)*Wbar_b/r_b**p", ""
        else:
            reached = time_power * sum(value / Fraction(step) ** order for value, step in zip(values, capped)) / norm
            name = "weighted bound T_e**(p+1)*sum_b M_b/r_b**p/beta"
            remedies = ", use SampledPairs(step_buckets=1)" if len(values) > 1 else ""
        raise ResourceLimitError(
            f"sampled-pair product-formula steps {max(steps)} exceed {type(formula).__name__}.max_steps={cap}: at "
            f"the cap the {name} is {_upward_float(reached)!r}, above the allowance {allowance!r} (0.1 times "
            f"approx_tol). Raise approx_tol or max_steps{remedies}, or choose "
            "ham_evo=Trotter(steps=...) with a fixed step count, which publishes its bound")
    return steps, allowance


def _kernel_lipschitz(kernel, *, end, active_time, l_norm):
    """Return L_k = D_rho + W_k*(L_theta + T_e*B_Ls), the k-Lipschitz bound of a continuous branch vector on |k| <= end.

    The branch vector r(k)*exp(-i*t*(k*L_s + H))*psi has magnitude rho(k)
    and phase theta(k). Duhamel's formula for two exponentials of Hermitian
    generators bounds the change in k of the evolution by t*||L_s||*|dk|.
    Apel-Berntson has constant magnitude, D_rho = 0, and phase derivative
    2*Delta*R/(pi*(R**2 - k**2)) <= L_theta = 2*|Delta|*R/(pi*(R**2 - end**2)).
    Low-Somma has theta = -c*k, L_theta = |c|, and rho = W_k*exp(-k**2/(4*gamma**2)),
    whose derivative k*rho/(2*gamma**2) peaks at z = min(end, sqrt(2)*gamma),
    giving D_rho = W_k*z*exp(-z**2/(4*gamma**2))/(2*gamma**2).
    """
    weight = float(kernel.importance_weight_bound)
    derived = kernel.derived_parameters
    if kernel.resolved_lchs_kernel.implementation == "apel_berntson":
        radius = float(kernel.radius)
        magnitude, phase = 0.0, 2.0 * abs(float(derived["Delta"])) * radius / (pi * (radius ** 2 - end ** 2))
    else:
        gamma = float(derived["gamma"])
        z = min(end, sqrt(2.0) * gamma)
        magnitude = weight * z * exp(-z * z / (4.0 * gamma * gamma)) / (2.0 * gamma * gamma)
        phase = abs(float(kernel.resolved_lchs_kernel.parameters["c"]))
    return magnitude + weight * (phase + active_time * l_norm)


def _model_vector_bound(method, problem, common, data, kernel, facts, total, mean, cap, end):
    """Return ``(Delta_v, reasons)`` with Delta_v >= ||u - v|| as a PhysicalScale, or None when a part is unknown.

    v is the vector that the sampling law targets before product-formula
    error and u the physical solution. Three parts add, each counting every
    application once with its mass:

    - Kernel: the total mass S = ``total`` times the kernel and k-quadrature
      bounds of a discrete kernel, or the continuous kernel and raw
      endpoint-guard bounds of a continuous one. The k grid or support is
      chosen for the final time T and its bounds hold for every elapsed
      time up to T, as the coherent constant-source route uses them.
    - Source rule: the Gauss-Legendre remainder of a discrete source rule
      with the original ||A|| (``inhomogeneous_theory.duhamel_log_bound``).
    - Finite bits of a continuous draw: a midpoint u_b replaces the uniform
      u within h = 2**-53, and for an increasing quantile f the mean of
      |f(u) - f(u_b)| is at most h*(f(1) - f(0)), because on each cell the
      two half-cell integrals are at most h times the cell's increment and
      the increments telescope. The branch vector is L_k-Lipschitz in k
      (`_kernel_lipschitz`) and has |r|*D_t-Lipschitz evolution in t with
      D_t = K*B_Ls + B_H. Changing k first, then t, and cancelling pi_i
      against M_i/pi_i gives min(2B, 2*K*h*S*L_k + h*(b - a)*M_src*W*D_t), with
      the k term for a continuous kernel, the t term for a continuous
      source, W the kernel cap (W_k, or beta for discrete nodes) and K the
      node bound ``end``.
    """
    from nwqlib.problems.inputs import PhysicalScale
    from .inhomogeneous_theory import duhamel_log_bound
    from .solution_error_budget import scaled_exp, scaled_sum
    reasons, parts = [], []
    discrete = isinstance(kernel, DiscreteKernel)
    if discrete:
        bounds = data.quadrature.coefficient_plan.quadrature
        unit = None if None in (bounds.approximate_lchs_error_bound, bounds.quadrature_error_bound) else \
            float(bounds.approximate_lchs_error_bound) + float(bounds.quadrature_error_bound)
        if unit is None:
            reasons.append("the kernel-quadrature pair has no kernel or k-quadrature component bound")
    else:
        unit = float(kernel.kernel_error_bound) + float(kernel.guard_omission_bound)
    if unit:
        parts.append(compose_recovery(total, unit))
    generator = common.generator
    source = None if problem.source is None else problem.source.preparation.physical_scale
    if source is not None and source.mantissa > 0 and facts.kind == "discrete_source" and not generator.exact_zero:
        lower = generator.dissipative_facts.lower
        if generator.generator_norm is None or lower is None:
            reasons.append("the source-quadrature bound needs bounds on ||A|| and on the lower spectrum of L")
        else:
            carried = source.exponent if abs(source.exponent) > 4096 else 0
            bound = scaled_exp(duhamel_log_bound(final_time=problem.elapsed_time, node_count=method.duhamel_nodes,
                generator_norm=generator.generator_norm, lower_bound=lower,
                log_source_norm=log(source.mantissa) + (source.exponent - carried) * log(2.0)))
            if bound is None:
                reasons.append("the source-quadrature bound exceeds the binary64 exponent range")
            else:
                parts.append(bound if not carried else
                             compose_recovery(bound, PhysicalScale(mantissa=0.5, exponent=carried + 1)))
    bits = []
    l_norm = float(common.l_norm)
    if not discrete:
        lipschitz = _kernel_lipschitz(kernel, end=end, active_time=float(facts.active_time), l_norm=l_norm)
        if lipschitz > 0:
            bits.append(compose_recovery(total, 2.0 * end * _HALF_BIT * lipschitz))
    source_mass = facts.source_mass if facts.kind == "continuous_source" else None
    if source_mass is not None and source_mass.mantissa > 0:
        h_norm = generator.hamiltonian_facts.norm
        if h_norm is None:
            reasons.append("the source-time coupling needs a bound on ||H||")
        else:
            low, high = facts.source_interval
            rate = end * l_norm + float(h_norm)
            if rate > 0:
                bits.append(compose_recovery(source_mass, (high - low) * _HALF_BIT * cap * rate))
    if bits:
        coupling, ceiling = scaled_sum(bits), compose_recovery(mean, 2.0)
        parts.append(ceiling if (coupling.exponent, coupling.mantissa) > (ceiling.exponent, ceiling.mantissa)
                     else coupling)
    if reasons:
        return None, reasons
    return scaled_sum(parts), reasons


def _row_frame(scale, frame, power):
    """Return scale/g**power as a binary64 upper value for g = 2**frame, 5e-324 for a positive underflow, or None on overflow."""
    from .source_selection import omission_upper_float
    if scale.mantissa == 0:
        return 0.0
    return omission_upper_float(compose_recovery(scale, PhysicalScale(mantissa=0.5, exponent=1 - power * frame)))


def _kernel_support(kernel):
    """Return ``(end, cap, discrete)``: the node bound K_*, the kernel cap and the discrete kernel's tables.

    For a discrete kernel, ``end`` is the largest |k_j| of a nonzero
    coefficient (one with a positive integer count), ``cap`` the upward
    binary64 value of beta = sum_j |c_j| and ``discrete`` the tuple
    ``(nodes, coefficients, counts, kept, magnitudes, beta)``. For a
    continuous kernel, ``end`` is the next binary64 number above the larger
    endpoint node, ``cap`` the importance-weight bound W_k and ``discrete``
    None. Both `_pair_bias` and the recipe bounds read B = S*cap and K_*
    here.
    """
    from nwqlib._numerics import complex_magnitude_upper
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    if not isinstance(kernel, DiscreteKernel):
        return nextafter(max(abs(kernel.node(0.0)), abs(kernel.node(1.0))), np.inf), \
            float(kernel.importance_weight_bound), None
    nodes = np.asarray(kernel.nodes, dtype=float)
    coefficients = np.asarray(kernel.coefficients, dtype=complex)
    counts = kernel_law(coefficients)[0].counts
    kept = np.flatnonzero(counts)
    magnitudes = {int(j): complex_magnitude_upper(complex(coefficients[j])) for j in kept}
    norm = sum(magnitudes.values(), Fraction(0))
    return float(np.abs(nodes[kept]).max()), _upward_float(norm), (nodes, coefficients, counts, kept, magnitudes,
                                                                     norm)


def _model_bias(method, problem, common, data, kernel, facts, total, mean, cap, end, frame):
    """Return ``(Delta_v, model_bias, reasons)`` of the model stage in the row frame g = 2**frame (`LCHSPairBias`)."""
    from .solution_error_budget import scaled_sum
    vector, reasons = _model_vector_bound(method, problem, common, data, kernel, facts, total, mean, cap, end)
    reasons = list(reasons)
    model = None
    if vector is not None:
        if vector.mantissa == 0:
            model = 0.0
        else:
            model = _row_frame(compose_recovery(vector, scaled_sum([compose_recovery(mean, 2.0), vector])), frame, 2)
            if model is None:
                reasons.append("the model bias exceeds the binary64 range of the row frame")
    return vector, model, reasons


def _observable_bound(terms):
    """Return C_O = sum_l |o_l| rounded upward."""
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    return _upward_float(sum((Fraction(abs(value)) for _, value in terms), Fraction(0)))


def _pair_bias(method, problem, common, data, kernel, facts, frame, terms, l_terms, h_terms, held):
    """Select the bucket steps and assemble the `LCHSPairBias` of one row table, before any row is drawn (module docstring).

    ``frame`` is the exponent of the weight frame g = 2**frame of the
    deterministic laws (`sampled_rows.pair_laws`); every step and bound is
    support-wide, so nothing depends on a drawn row.

    ``held`` is the admitted H + E + R of `plan_sampled`. Before the second
    kernel law, the magnitude table and the node-term map of `_census` are
    built, the allowance
    C_map = ``_node_cache_bytes``(N, P + 1, q) = 128*N*(P + 2)*(q + 1) + 8192*N
    is admitted on top of it, with N the M discrete kernel slots or the K
    buckets of a continuous kernel and P the nonidentity labels; P + 1 is
    the label count with the identity that ``_node_cache_bytes`` takes,
    whether or not the identity is present. This existing engineering
    allowance for per-node label/value rows, keys and a scalar census record
    covers the map, the census's view of it, the magnitude dictionary and
    the census records. The total stays in ``census_held`` while the census
    runs, whose own workspace its owner adds. Not covered: the bucket member
    lists (at most M indices, but K list headers), later Fraction
    intermediates, sorting and SDK workspace and process memory.
    """
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    from .solution_error_budget import scaled_sum
    from .time_independent_terms import _node_cache_bytes
    order, active, q = method.ham_evo.order, float(facts.active_time), common.generator.num_qubits
    labels = tuple(sorted({label for label, _ in l_terms} | {label for label, _ in h_terms}))
    count = method.estimator.step_buckets
    discrete = isinstance(kernel, DiscreteKernel)
    held += _node_cache_bytes(len(kernel.coefficients) if discrete else count, len(labels) + 1, q)
    _check_bytes(held, method.max_bytes, "sampled LCHS held data, pair rows and product-formula census map")
    time_power = Fraction(active) ** (order + 1)
    l_table, h_table = dict(l_terms), dict(h_terms)
    reasons = []
    applications = _application_masses(facts)
    total = scaled_sum([mass for mass, _ in applications])
    continuous_time = any(elapsed is None for _, elapsed in applications)
    end, cap, tables = _kernel_support(kernel)
    edges = _bucket_edges(end, count)
    if discrete:
        nodes, _, counts, kept, magnitudes, norm = tables
        census, work = _census(labels, {int(j): tuple(
            (label, value) for label in labels
            if (value := float(nodes[j]) * l_table.get(label, 0.0) + h_table.get(label, 0.0)) != 0.0)
            for j in kept}, q, method, held)
        bucket = np.searchsorted(np.asarray(edges), np.abs(nodes[kept]), side="left")
        members = [[int(j) for j, b in zip(kept, bucket) if b == index] for index in range(len(edges))]
    else:
        census, work = _census(labels, {index: tuple(
            (label, edge * abs(l_table.get(label, 0.0)) + abs(h_table.get(label, 0.0))) for label in labels)
            for index, edge in enumerate(edges)}, q, method, held)
    relation = "finite_sum" if discrete and not continuous_time else "bucket_maximum"
    if relation == "finite_sum":
        values = [sum((magnitudes[j] * census[j] for j in group), Fraction(0)) for group in members]
        probabilities = [sum(int(counts[j]) for j in group) / _TOTAL for group in members]
        steps, allowance = _select_steps(method, relation, values, probabilities, norm, time_power)
        # E = D/B = (sum_i M_i*t_i**(p+1)/S) * (sum_b M_b/r_b**p)/beta.
        weighted = scaled_sum([compose_recovery(mass, elapsed ** (order + 1)) for mass, elapsed in applications
                               if elapsed > 0])
        kernel_part = _upward_float(sum((value / Fraction(step) ** order for value, step in zip(values, steps)),
                                        Fraction(0)) / norm)
        operator = 0.0 if weighted.mantissa == 0 or kernel_part == 0 else \
            ldexp(weighted.mantissa / total.mantissa, weighted.exponent - total.exponent) * kernel_part
    else:
        values = [max((census[j] for j in group), default=Fraction(0)) for group in members] if discrete else \
            [census[index] for index in range(len(edges))]
        steps, allowance = _select_steps(method, relation, values, None, None, time_power)
        operator = _upward_float(time_power * max(value / Fraction(step) ** order
                                                  for value, step in zip(values, steps)))
    if not isfinite(operator):
        operator = None
        reasons.append("the product-formula bound exceeds the binary64 range")
    mean = compose_recovery(total, cap)
    pf_bias = None if operator is None else 0.0 if operator == 0 else \
        _row_frame(compose_recovery(mean, mean, 2.0 * operator), frame, 2)
    if operator is not None and pf_bias is None:
        reasons.append("the product-formula bias exceeds the binary64 range of the row frame")
    vector, model, model_reasons = _model_bias(method, problem, common, data, kernel, facts, total, mean, cap, end,
                                               frame)
    reasons += model_reasons
    bias = LCHSPairBias(
        step_rule="fixed" if allowance is None else "weighted", inner_relation=relation, bucket_edges=edges,
        bucket_steps=steps, step_allowance=allowance, product_formula_operator=operator, mean_weight_bound=mean,
        model_vector_bound=vector, inner_bias=pf_bias, model_bias=model,
        observable_bound=_observable_bound(terms),
        kernel_allowances=None if discrete else (0.5 * float(method.approx_tol),) * 2,
        reasons=tuple(reasons))
    return bias, work


def _recipe_support(method, kernel, facts, l_terms, h_terms, num_qubits):
    """Return the `sampled_recipes.Support` of the whole support and ``(total, cap, end)``, before any row is drawn."""
    from .sampled_recipes import Support
    from .solution_error_budget import scaled_sum
    total = scaled_sum([mass for mass, _ in _application_masses(facts)])
    end, cap, _ = _kernel_support(kernel)
    labels = {label for label, _ in l_terms} | {label for label, _ in h_terms}
    support = Support(active_time=float(facts.active_time), node_bound=end,
                      h_norm=sum((Fraction(abs(value)) for _, value in h_terms), Fraction(0)),
                      l_norm=sum((Fraction(abs(value)) for _, value in l_terms), Fraction(0)), labels=len(labels),
                      mean_weight=compose_recovery(total, cap), num_qubits=num_qubits)
    return support, (total, cap, end)


def _recipe_bias(method, problem, common, data, kernel, facts, frame, terms, support, scales, operator, rule, reasons):
    """Return the `LCHSPairBias` of a recipe Plan: its product-formula bias (B/g)**2*operator and the shared model bias.

    ``operator`` is the recipe's factor of B**2 in its N bias
    (`sampled_recipes`): B_ext + E_eff for Richardson, 2*delta + delta**2
    for PTSC, None when unknown. The complete product B**2*operator is
    formed as a PhysicalScale before the row frame divides it, so no
    intermediate power underflows to zero.
    """
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    total, cap, end = scales
    mean = support.mean_weight
    reasons = list(reasons)
    pf_bias = None
    if operator is not None:
        value = _upward_float(Fraction(operator))
        pf_bias = 0.0 if value == 0 else _row_frame(compose_recovery(mean, mean, value), frame, 2)
        if pf_bias is None:
            reasons.append("the product-formula bias exceeds the binary64 range of the row frame")
    vector, model, model_reasons = _model_bias(method, problem, common, data, kernel, facts, total, mean, cap, end,
                                               frame)
    return LCHSPairBias(
        step_rule=rule, inner_relation=route_name(method.estimator.product_formula),
        product_formula_operator=None,
        mean_weight_bound=mean, model_vector_bound=vector, inner_bias=pf_bias, model_bias=model,
        observable_bound=_observable_bound(terms),
        kernel_allowances=None if isinstance(kernel, DiscreteKernel) else
        (0.5 * float(method.approx_tol),) * 2, reasons=tuple(reasons + model_reasons))


def _row_words(payload, rec, corrections=True):
    """Return ``word(row)``, the occurrence word that evolves both branches of a row, from the saved choices.

    ``word(row)`` is ``(template, steps, repetitions, coefficients,
    corrections, phase)`` for `sampled_native.pair_select_plan` and
    `sampled_native.build_pair_circuit`: the (label, multiplier) word, the
    divisor of its angles, its repetitions per shot, explicit (l, h)
    coefficients of each occurrence or None, the PTSC correction segments
    or None, and the recipe's readout phase (pi for a negative member
    multiplier). Plain rows run the order-p template of the sorted labels
    at their bucket steps. Richardson rows run the Strang template at their
    member's level s or 2s. MPF rows run their member's P7 macrostep with
    divisor r, r/2 times. PTSC rows run one Lie step of the unmerged H then
    L lists with divisor r once per segment, each followed by the segment's
    corrections (`sampled_recipes`). With ``corrections=False`` a PTSC word
    holds None in place of its r segment descriptors; a resource count reads
    the segment count from ``recipe.steps`` and the correction CX from the
    saved arrays (`row_correction_cx`) without forming the descriptors.
    Nothing is drawn or selected here.
    """
    from .sampled_recipes import mpf_template
    from .select_synthesis import product_formula_occurrence_template
    labels = tuple(sorted({label for label, _ in payload.l_terms} | {label for label, _ in payload.h_terms}))
    recipe = rec.pair_recipe
    if recipe is None:
        template = product_formula_occurrence_template(labels, order=payload.order)
        rows, bias = rec.pair_rows, rec.pair_bias
        buckets = np.asarray(bias.bucket_steps)

        def plain(row):
            steps = int(buckets[bias.bucket_of([rows.node_x.array[row], rows.node_y.array[row]])].max())
            return template, steps, steps, None, None, 0.0
        return plain
    if recipe.recipe == "ptsc":
        lie = tuple((label, 2.0) for label, _ in payload.h_terms + payload.l_terms)
        coefficients = (tuple((0.0, value) for _, value in payload.h_terms)
                        + tuple((value, 0.0) for _, value in payload.l_terms))
        return lambda row: (lie, recipe.steps, 1, coefficients, recipe.corrections(row) if corrections else None, 0.0)
    if recipe.recipe == "richardson":
        templates = (product_formula_occurrence_template(labels, order=2),) * 2
    else:
        templates = tuple(mpf_template(labels, alpha) for alpha in recipe.member_alphas)
    member, phases = recipe.member.array, recipe.row_phases()

    def word(row):
        index = int(member[row])
        return (templates[index], recipe.member_steps[index], recipe.member_repetitions[index], None, None,
                float(phases[row]))
    return word


def _pair_plan(payload, rows, row, template, steps, repetitions, coefficients):
    """Return the two-slot SELECT plan of ``row`` for one occurrence word (`sampled_native.pair_select_plan`).

    The payload's L_c and H_c tables are checked once (`sampled_native._pair_tables`)
    and kept in a private attribute that is not a dataclass field, so the
    numerical Plan archive excludes it, as for ``_input_preparation_cache``.
    A multiplexor SELECT gets no structure certificate, which only the
    structured SELECT reads.
    """
    from .sampled_native import _pair_select_plan, _pair_tables
    checked = getattr(payload, "_pair_table_cache", None)
    if checked is None:
        checked = _pair_tables(dict(payload.l_terms), dict(payload.h_terms))
        object.__setattr__(payload, "_pair_table_cache", checked)
    return _pair_select_plan(checked, times=(float(rows.time_x.array[row]), float(rows.time_y.array[row])),
                             nodes=(float(rows.node_x.array[row]), float(rows.node_y.array[row])), steps=steps,
                             order=payload.order, template=template, repetitions=repetitions,
                             occurrence_coefficients=coefficients,
                             structured=payload.implementation == "structured")


def row_select_cx(payload, rec):
    """Return the pair SELECT CX count of every row before routing, without building a circuit.

    The multiplexed SELECT keeps its parity and multiplexor CX whatever its
    angles, so a row whose word repeats n times (r steps, r/2 MPF
    macrosteps or r PTSC Lie steps) emits n times the count of one word.
    The structured SELECT drops the two CX of every occurrence whose two
    angles are exactly equal in the row, so each row's tables are formed
    (`sampled_native.pair_select_plan`) and their counts read from
    `product_formula_select_resource_law`. Fanout CX of control copies and
    PTSC corrections (`row_correction_cx`) are not included.

    Raises:
        ApplicabilityError: For the structured SELECT, when a row's two-entry
            table differs from its one-bit affine form by more than its
            binary64 window.
    """
    from nwqlib.subroutines._multiplexors import product_formula_select_resource_law
    rows, recipe = rec.pair_rows, rec.pair_recipe
    word = _row_words(payload, rec, corrections=False)
    segments = recipe.steps if recipe is not None and recipe.recipe == "ptsc" else 1
    counts = np.empty(rows.clusters, dtype=np.int64)
    one = {}
    for row in range(rows.clusters):
        template, steps, repetitions, coefficients, _, _ = word(row)
        if payload.implementation == "multiplexor":
            if id(template) not in one:
                one[id(template)] = int(product_formula_select_resource_law(
                    _pair_plan(payload, rows, row, template, steps, 1, coefficients),
                    implementation="multiplexor")["select_basis_cx_count"])
            counts[row] = segments * repetitions * one[id(template)]
            continue
        selected = _pair_plan(payload, rows, row, template, steps, repetitions, coefficients)
        if not selected.structure_certificate["eligible"]:
            raise ApplicabilityError(f"row {row} has a structured pair SELECT table outside its binary64 affine "
                                     "window; use lcu_select 'auto' or 'multiplexor'")
        counts[row] = segments * product_formula_select_resource_law(
            selected, implementation="structured")["select_basis_cx_count"]
    return counts


def row_correction_cx(rec):
    """Return the CX-basis count of every row's PTSC corrections, zero otherwise.

    The rule of `sampled_recipes.correction_cx`, applied to the saved
    (B, 2, r) arrays at once: a segment of order s and Pauli support
    w = popcount(x | z) costs w CX for s >= 4, 2w for a low-order rotation
    and nothing for an identity correction (s = 0) or an empty word.
    """
    recipe, clusters = rec.pair_recipe, rec.pair_rows.clusters
    if recipe is None or recipe.recipe != "ptsc":
        return np.zeros(clusters, dtype=np.int64)
    order = recipe.segment_order.array
    support = np.bitwise_count(recipe.segment_x.array | recipe.segment_z.array).astype(np.int64)
    cost = np.where(order >= 4, support, 2 * support) * (order != 0)
    return cost.reshape(clusters, -1).sum(axis=1)


def _pair_block(method, problem, rows, payload, rec, row_cx):
    """Return the row-parameterized pair block with its CX law, exact, an upper bound, or absent with inputs inside.

    ``row_cx`` is each row's SELECT plus PTSC correction CX count.
    """
    q, copies, order = payload.num_system_qubits, payload.control_copies, payload.order
    recipe = rec.pair_recipe
    system = "zero" if payload.prepares_inputs else "live"
    ports = (("branch", 1, "zero", "unitary"), ("branch_copy", copies - 1, "zero", "zero"),
             ("system", q, system, "unitary"))
    signature = BlockSignature(name="lchs_pair", target=_source(PAIR_TARGET),
        quantum=tuple(QuantumPort(name=name, width=width, requires=requires, ensures=ensures)
                      for name, width, requires, ensures in ports if width),
        parameters=(Parameter(name="row", domain="integer", lower=0, upper=rows.clusters - 1),), coupling="joint")
    # Each emitted occurrence of support w, and each PTSC correction, is at
    # most 4w basis gates, 2(w-1) parity CX and a controlled rotation of
    # 2 CX and 2 RZ (or w controlled Paulis and a phase).
    emitted, word_positions = _pair_word_sizes(payload, rec)
    work = emitted * (6 * q + 2) + 2 * q + 2 * copies + 4
    count = "SELECT and PTSC correction CX" if recipe is not None and recipe.recipe == "ptsc" else "SELECT CX"
    laws, context = (), ("pair SELECT, input preparations of the row's kinds, branch phase and readout basis of "
                         "one bound row; the preparations' elementary cost is not derived, so no CX law is given "
                         f"(sampled_pair_resources counts {count} and preparations per row)")
    if not payload.prepares_inputs:
        largest = int(row_cx.max())
        laws = (ResourceLaw(metric="cx", basis="cx", value=largest + 2 * (copies - 1),
            interpretation="exact" if int(row_cx.min()) == largest else "upper_bound",
            evidence=Evidence(kind="proved_relation", source=METHOD), unbound_parameters=("row",),
            assumptions=(f"{payload.implementation} pair {count} before routing (sampled_plan.row_select_cx, "
                         "row_correction_cx) plus 2*(r_c - 1) fanout CX",
                         "every row emits the same count" if int(row_cx.min()) == largest else
                         "bound at the largest row count; other rows emit fewer CX")),)
        context = "pair SELECT, branch phase and readout basis of one bound row; no input preparation"
    if recipe is None:
        formula = "a branch-selected product formula"
        evidence = "CSTWZ bucket bound in the Plan reconstruction (pair_bias)"
        choice = f"{payload.implementation}:order={order}:copies={copies}"
    else:
        formula = f"the row's {recipe.recipe} choice of a branch-selected product formula"
        evidence = f"{recipe.recipe} bias relation in the Plan reconstruction (pair_bias, pair_recipe)"
        choice = f"{payload.implementation}:order={order}:copies={copies}:recipe={recipe.version}"
    semantics = BlockSemantics(kind="unknown",
        input=InputRef(identity=rows.content_id, representation="lchs_pair_rows", source=METHOD),
        basis=problem.basis,
        relation=f"row r: (|0>U_x|psi_x> + exp(i*theta_r)|1>U_y|psi_y>)/sqrt(2) by {formula}, then H on the branch "
                 "qubit and the basis change of the row's Pauli term",
        input_projector=("branch qubit, control copies and system zero" if payload.prepares_inputs else
                         "branch qubit and control copies zero; system holds the prepared initial direction"),
        output_projector="control copies zero", success="deterministic unitary", workspace=copies - 1,
        restoration="control copies uncomputed to zero", epsilon=None,
        approximation_metric="product-formula operator norm of each branch",
        approximation_evidence=evidence,
        inverse_legal=False, control_legal=False,
        phase="row phase theta applied once at the branch qubit; the scalar centering phase is in the branch "
              "phases; a controlled input preparation keeps its global phase as the relative input phase")
    record = SelectedDefinition(signature=signature, semantics=semantics, implementation=signature.target,
        choice=choice, decomposition=None, cost_law=None,
        construction_work=work, resource_laws=laws, coefficient_selection_id=rows.content_id,
        cost_context=context, workspace=(Workspace(location="host", purpose="workspace", bytes=None, source=METHOD),))
    return SelectedBlock.bind(record, payload=(payload, rec, problem), constructor=construct_pair,
                              **_pair_memory(payload, rec, problem, word_positions))


def _pair_word_sizes(payload, rec):
    """Return the largest occurrence count of a row and the total word positions of all rows, from one pass over them.

    An occurrence of weight w with x X and y Y factors keeps 2x + 4y basis
    gates, 2(w-1) parity CX and a two-CX, two-RZ controlled rotation at top level
    on Aer. A PTSC segment adds its two corrections of at most 7q + 3 positions
    each. The rotation count is an upper count, because a multiplexor omits a
    zero angle, and the structured SELECT keeps fewer CX.
    """
    recipe = rec.pair_recipe
    ptsc = recipe is not None and recipe.recipe == "ptsc"
    segments = recipe.steps if ptsc else 1
    correction = 2*(7*payload.num_system_qubits + 3) if ptsc else 0
    word = _row_words(payload, rec, corrections=False)
    per_template = {}
    emitted = total = 0
    for row in range(rec.pair_rows.clusters):
        template, _, repetitions, _, _, _ = word(row)
        emitted = max(emitted, segments * (repetitions * len(template) + (2 if ptsc else 0)))
        positions = per_template.get(id(template))
        if positions is None:
            positions = per_template[id(template)] = sum(
                2*label.count("X") + 4*label.count("Y") + 2*(len(label) - label.count("I") - 1) + 4
                for label, _ in template)
        total += segments*(repetitions*positions + correction)
    return emitted, total


def _pair_memory(payload, rec, problem, word_positions):
    """Return the ``SelectedBlock.bind`` estimates of the pair block, from the mean over all rows, or {} without a law.

    A uniform experiment multiplies its first point by the point count
    (``SelectedBlock.bind``). The mean positions of the words and inputs are
    rounded up, so that product covers their total, by less than one position
    per row more. This is a mean over the rows, not the largest row or template,
    whose exact entries `sampled_native.pair_template_size` counts separately. A
    row keeps its word (``_pair_word_sizes``), two branch H and the branch
    phase, 2(r_c - 1) fanout CX, at most 2q readout basis gates and any inputs
    prepared inside the pair. Measurements are counted by the Program.

    Inside the pair, a row whose branches share an input keeps one uncontrolled
    preparation, and a row whose branches differ keeps both controlled. By
    circuit counts at q = 2, 3, 4, a direct preparation is at most 3*2**q
    positions alone and 8*2**q as a controlled pair, and occupations q. Other
    inputs have no measured count, so the block then has no law, and neither
    does a sampled QSP row.

    A backend that binds templates, such as local Aer, keeps this estimate for the
    one row that compiles the template, and the Run adds ``_TEMPLATE_ROW_BYTES``
    to every row, so that one row counts a handle twice. That 32 KiB per template
    is padding, kept until a footprint calibration can judge it. Elsewhere each
    row keeps this estimate. Positions take ``NATIVE_POSITION_BYTES``. The per-row
    retention itself is unchanged.
    """
    if rec.pair_qsp is not None:
        return {}
    rows = rec.pair_rows
    q = payload.num_system_qubits
    inputs = 0
    if payload.prepares_inputs:
        states = (problem.initial_state, problem.source)
        counts = {"qiskit.occupation": (q, q), "qiskit.direct": (3 << q, 8 << q)}
        kinds = np.unique(np.concatenate((rows.kind_x.array, rows.kind_y.array)))
        used = [counts.get(states[int(kind)].preparation.implementation) for kind in kinds]
        if None in used:
            return {}
        mixed = int(np.count_nonzero(rows.kind_x.array != rows.kind_y.array))
        inputs = (rows.clusters - mixed)*max(alone for alone, _ in used) + mixed*max(pair for _, pair in used)
    positions = (word_positions + inputs + rows.clusters - 1)//rows.clusters + 3 + 2*(payload.control_copies - 1) + 2*q
    return dict(native_bytes=_ROW_HANDLE_BYTES + positions*NATIVE_POSITION_BYTES,
                definition_bytes=positions*NATIVE_POSITION_BYTES)


def _row_inputs(payload, rows, word, row):
    """Return the arguments of `sampled_native.build_pair_circuit` for ``row`` other than its input preparations.

    ``word`` is `_row_words` of the Plan. The recipe's readout phase (pi
    for a negative member multiplier) is added to the row's physical phase
    theta only here, at the circuit; the row table and the recipe record
    keep them apart.
    """
    template, steps, repetitions, coefficients, corrections, recipe_phase = word(row)
    term = int(rows.term.array[row])
    return dict(plan=_pair_plan(payload, rows, row, template, steps, repetitions, coefficients),
                kinds=(int(rows.kind_x.array[row]), int(rows.kind_y.array[row])),
                theta=remainder(float(rows.phase_y.array[row]) - float(rows.phase_x.array[row]) + recipe_phase, tau),
                term_label=payload.term_labels[term] if term >= 0 else "I" * payload.num_system_qubits,
                implementation=payload.implementation, control_copies=payload.control_copies,
                corrections=corrections)


def _input_preparations(payload, problem):
    """Reuse this payload's input circuits and controlled gates, or return None without inputs.

    The private cache holds its actual immutable input objects. It is not a
    dataclass field, so the numerical Plan archive excludes it and a loaded
    Plan builds its own circuits on first use.
    """
    if not payload.prepares_inputs:
        return None
    from .native import state_preparations
    from .sampled_native import _InputPreparations

    cached = getattr(payload, "_input_preparation_cache", None)
    if cached is None or cached[0] is not problem.initial_state or cached[1] is not problem.source:
        circuits = state_preparations(problem.initial_state, problem.source, max_bytes=payload.max_bytes)
        preparations = _InputPreparations({index: circuits[name]
                                          for index, name in enumerate(("initial", "source")) if name in circuits})
        cached = (problem.initial_state, problem.source, preparations)
        object.__setattr__(payload, "_input_preparation_cache", cached)
    return cached[2]


def construct_pair(block, arguments, method_context):
    """Build the unitary pair circuit of the bound row from the saved rows, recipe or bank choices and input states.

    A sampled QSP row is built by `sampled_qsp.build_qsp_pair` from its saved
    bucket, members and prepared phases; nothing is fitted or drawn.
    """
    from .sampled_native import build_pair_circuit
    payload, rec, problem = block._payload
    if len(arguments) != 1 or arguments[0][0] != "row":
        raise ValueError("the sampled pair block takes exactly the row argument")
    if rec.pair_qsp is not None:
        from .sampled_qsp import build_qsp_pair
        return build_qsp_pair(payload, rec, problem, int(arguments[0][1]))
    inputs = _row_inputs(payload, rec.pair_rows, _row_words(payload, rec), int(arguments[0][1]))
    return build_pair_circuit(preparations=_input_preparations(payload, problem), measure=False,
                              max_bytes=payload.max_bytes, **inputs).circuit


def row_templates(plan, points):
    """Group the row points ``points`` of a sampled-pair Plan by the gates of their pair circuits (`Method.row_templates`).

    Each row's gate key and slot values come from
    `sampled_native.pair_key`, which builds no circuit and does not walk
    the repeated SELECT occurrences. Rows with equal keys share one
    template, whose parameters ``lchs_pair[i]`` take slot i. The template key is
    the SHA-256 of the gate key's text, which is deterministic for one Plan.
    Every row of a Plan has one readout.

    Grouping builds no circuit, so the Run can compare all templates with what it
    holds before any exists. Each template states the allowance of
    `sampled_native.pair_template_size` and, when the Run reads its circuit
    (`protocol._DeferredRowTemplate`), builds
    `sampled_native.build_pair_template` of the group's first row, with the input
    preparations that the Plan's first build caches. The allowance prices each
    preparation as its wrapper entry only.

    A sampled QSP Plan groups its rows by their bank, member, kind and branch-query
    choices instead (`sampled_qsp.qsp_row_templates`), and it builds each template
    here, with no stated size.
    """
    from hashlib import sha256
    from qiskit.circuit import ParameterVector
    from nwqlib.algorithms.protocol import _DeferredRowTemplate
    from .sampled_native import build_pair_template, pair_key, pair_template_size
    block = next(item for item in plan.blocks if item.record.signature.name == "lchs_pair")
    payload, rec, problem = block._payload
    if rec.pair_qsp is not None:
        from .sampled_qsp import qsp_row_templates
        return qsp_row_templates(plan, points, block, payload, rec, problem)
    rows, word = rec.pair_rows, _row_words(payload, rec)
    groups = {}
    for point in points:
        inputs = _row_inputs(payload, rows, word, point_row(point))
        key, values = pair_key(prepares=payload.prepares_inputs, **inputs)
        # The template is built from the first row of its group, whose inputs are kept.
        groups.setdefault(key, (inputs, []))[1].append((point.key, values))

    def deferred(key, inputs, members):
        parameters = tuple(ParameterVector("lchs_pair", len(members[0][1])))

        def build():
            return build_pair_template(parameters=parameters, preparations=_input_preparations(payload, problem),
                                       max_bytes=payload.max_bytes, **inputs)

        return _DeferredRowTemplate(key="sha256:" + sha256(repr(key).encode()).hexdigest(),
                                    block_id=block.record.content_id, parameters=parameters, rows=tuple(members),
                                    build=build, retained_bytes=pair_template_size(key)[2])

    return tuple(deferred(key, inputs, members) for key, (inputs, members) in groups.items())


def _pair_batch(clusters, repeats, rows_id):
    """Return the MeasurementBatch of the B = ``clusters`` rows and its one Experiment."""
    metadata = MetadataRef(format=METHOD, data=InputRef(identity=rows_id, representation="lchs_pair_rows",
                                                         source=METHOD))
    batch = MeasurementBatch(body="body", settings=(Setting(label="pairs", metadata=metadata),),
                             axes=(RangeAxis(parameter="row", start=0, stop=clusters),), repetitions=repeats,
                             observation_kind="counts")
    return batch, Experiment(name=BATCH, batch=BATCH, setting_index=0, readout=ReadoutDetails())


def point_row(point):
    """Return the row index of a sampled-pair `StaticPoint`, the value of its one ``row`` axis."""
    (binding,) = point.bindings
    return int(binding.value)


def _exact_readout(problem):
    """Whether the Problem is the exact no-evolution readout: zero time, zero inputs or an input without its scale."""
    states = (problem.initial_state,) + (() if problem.source is None else (problem.source,))
    return (problem.elapsed_time == 0 or any(state.preparation.physical_scale is None for state in states)
            or all(state.preparation.physical_scale.mantissa == 0 for state in states))


def _sampled_applicability(method, output, execution):
    """Refuse an output, execution or coherent-only choice of a sampled-pair Method, before any common-stage work.

    Provider enumeration compares the rows of the coherent route, and an MPS
    or analytic coefficient PREP prepares the coherent LCU's coefficient
    state; sampled pairs draw their applications and kernel nodes
    classically and prepare no coefficient state, so neither choice has a
    sampled construction.
    """
    if not isinstance(output, (NormSquared, QuadraticForm, NormalizedExpectation)):
        raise ApplicabilityError("sampled pairs estimate NormSquared, QuadraticForm or NormalizedExpectation. "
                                 "Their pair draws and signed scores are not a Solution, StateVector or Born "
                                 "Samples; use estimator='coherent' for those outputs")
    if execution != "quantum":
        raise ApplicabilityError("sampled pairs need quantum execution, because their scores are measured")
    if method.lchs_kernel.implementation == "auto":
        raise ApplicabilityError("lchs_kernel='auto' with k_quadrature='auto' compares the providers by the rows of "
                                 "the coherent route; sampled pairs draw from one concrete kernel-quadrature pair. "
                                 "Give a concrete pair, or use estimator='coherent'")
    if method.lcu_prep != "direct":
        raise ApplicabilityError(f"lcu_prep={route_name(method.lcu_prep)!r} prepares the coefficient "
                                 "state of the coherent LCU; sampled pairs draw their applications and kernel nodes "
                                 "classically and prepare no coefficient state. Use 'direct', or "
                                 "estimator='coherent'")


@dataclass
class _Pass:
    """One deterministic planning pass of a sampled-pair Plan up to its draws (`_forecast`)."""

    common: object
    data: object
    kernel: object
    facts: object
    terms: tuple
    l_terms: tuple
    h_terms: tuple
    laws: object
    clusters: int
    held: int
    recipe: tuple | None
    bias: object
    census_work: int
    qsp: object


def _sufficient_rows(method, laws, factor, output, accuracy):
    r"""Return B, the independent rows whose Hoeffding radius meets an absolute sampling tolerance (SC1, SC2).

    The scaled range of one row mean is ``r_N = (W/g)**2 F`` for the norm
    squared and ``r_Q = H_O r_N`` for the quadratic form, with ``W/g`` the
    all-support weight cap, ``H_O = max_l |o_l|/p_l`` over the actual term law
    (identity included) and F the support-wide bound of the inner factor:
    1 for a plain product formula, the Richardson multiplier bound, 2 for
    MPF, ``mu_cap**2`` for PTSC and ``max(1, max_b zeta_b**2)`` for a QSP
    bank. The physical range is ``H = g**2 r`` with ``g = 2**frame``. The B
    row means are independent and lie in ``[-H, H]`` whatever the repeats,
    so `sampled_moments.hoeffding_radius`, ``H sqrt(2 log(2/delta)/B)``, is
    at most epsilon exactly when

        \[B\ge 2\,(H/\epsilon)^2\log(2/\delta),\]

    with ``delta = 1 - confidence``: the whole failure probability of the one
    requested scalar (`analysis._pair_analysis` uses one quantity for a norm
    or a quadratic form). B is that ceiling, at least 1; H = 0 (a zero
    observable) gives one row. The repeats m do not enter. As in
    ``analysis.sufficient_counts``, ``g**4`` is the exact ``2**(4 frame)``,
    the inputs are exact binary64 values and the quotient and logarithm are
    80-digit Decimal values rounded upward. A count whose row data alone
    (1024 bytes per row, `_row_bytes`) exceed ``max_bytes`` is refused
    before the integer is formed.
    """
    from decimal import ROUND_CEILING, Decimal, localcontext
    range_q = Fraction(laws.range_bound) if isinstance(output, QuadraticForm) else Fraction(1)
    scaled = Fraction(laws.weight_bound) ** 2 * Fraction(factor) * range_q
    if scaled == 0:
        return 1
    ratio = scaled / Fraction(accuracy.absolute_tolerance)
    delta = 1 - Fraction(accuracy.confidence)
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_CEILING
        square = (Decimal(ratio.numerator) / Decimal(ratio.denominator)) ** 2
        argument = Decimal(2) / (Decimal(delta.numerator) / Decimal(delta.denominator))
        required = 2 * square * Decimal(2) ** (4 * laws.frame) * argument.ln().next_plus()
        byte_limit = memory_limit(method.max_bytes)
        if byte_limit is not None and required > byte_limit // 1024:
            budget = (f"LCHS.max_bytes={method.max_bytes}" if method.max_bytes is not None
                      else f"the physical memory of {byte_limit} bytes (LCHS.max_bytes=None)")
            raise ResourceLimitError(f"the sampling accuracy needs B = {required:.6E} independent rows "
                                     f"({method.estimator.repeats} "
                                     f"shots each), whose row data alone exceed {budget} "
                                     f"(1024 bytes per row); raise the tolerance or max_bytes")
        return max(1, int(required.to_integral_value(rounding=ROUND_CEILING)))


def _forecast(method, problem, output, *, shots=None, accuracy=None):
    """Run one deterministic planning pass of a sampled-pair Plan up to its draws: no RNG, no phase fit.

    First refuses an initial-state preparation other than ``"direct"`` and an
    A outside the route's input forms. Then builds the common-stage facts
    (`selection.compact_common`) and, for a
    discrete kernel or the exact unitary reduction, the Method's one grid;
    a continuous kernel builds no grid. Then the outer laws and frame
    (`sampled_rows.pair_laws`), the B-independent recipe facts or the
    phase-free QSP bank (`sampled_qsp.forecast_qsp`), the row count B
    (`_sufficient_rows` for an accuracy request, else ``shots // m``), and
    every admission that B decides before a row or member array exists:
    held inputs and grid, row arrays and laws, recipe arrays and
    generation work, the QSP bank's known arrays, the plain census and
    steps (`_pair_bias`), the structured SELECT tables and the input PREPs.
    For an accuracy request, `sampled_accuracy_shots` and `plan_sampled`
    share one run within a planning call (`_accuracy_forecast`).
    """
    from .providers import resolve_lchs_continuous_kernel
    from .quantum import observable_terms
    from .sampled_rows import pair_laws
    from .selection import compact_common, with_providers
    from .solution_error_budget import signed_shift
    qsp = route_name(method.ham_evo) == "qsp_block_encoding"
    estimator = method.estimator
    if method.init_prep != "direct":
        raise ApplicabilityError(
            f"init_prep={route_name(method.init_prep)!r} compresses a dense initial vector, "
            "and sampled pairs evolve the compact parts of A and prepare each input with the state's own "
            "preparation, so the request would be ignored. Use init_prep='direct'")
    representation = problem.A.reference.representation
    allowed = ("pauli", "cartesian", "band", "block_encoding") if qsp else ("pauli", "cartesian")
    if representation not in allowed:
        parts, forms = (("Pauli, band or supplied block-encoding parts", "a Pauli sum, a Hermitian band operator, "
                         "Cartesian parts or a supplied block encoding") if qsp
                        else ("Pauli terms", "a Pauli sum or Pauli parts"))
        raise ApplicabilityError(
            f"sampled {'QSP ' if qsp else ''}pairs evolve each branch from the {parts} of A, and a "
            f"{representation!r} A has no such {'encoding' if qsp else 'exponential'} in this route; give A as "
            f"{forms}, or use estimator='coherent'")
    common = compact_common(method, problem)
    generator = common.generator
    if not qsp:
        for name, facts in (("L", generator.dissipative_facts), ("H", generator.hamiltonian_facts)):
            if facts.access not in ("zero", "pauli"):
                raise ApplicabilityError(f"sampled pairs exponentiate Pauli terms, and the {facts.access!r} part "
                                         f"{name} has no Pauli exponential in this route")
    q = generator.num_qubits
    identity = "I" * q
    l_terms = h_terms = ()
    if not qsp:
        l_terms = _centered_terms(generator.dissipative, identity, method.max_bytes)
        h_terms = _centered_terms(generator.hamiltonian, identity, method.max_bytes)
    terms = observable_terms(problem, output, max_bytes=method.max_bytes)
    data = None
    if estimator.kernel == "continuous" and common.l_norm != 0:
        half = 0.5 * float(method.approx_tol)
        kernel = resolve_lchs_continuous_kernel(lchs_kernel=method.lchs_kernel, kernel_allowance=half,
                                                guard_allowance=half)
    else:
        # A zero bound on ||L + s*I|| selects the exact one-node unitary
        # reduction, which replaces a continuous kernel as well.
        data = with_providers(common, problem)
        kernel = DiscreteKernel(nodes=data.quadrature.k_nodes, coefficients=data.quadrature.coefficients,
                                formula_id=data.quadrature.coefficient_plan.lchs_kernel_formula_id)
    facts = _application_facts(method, problem, generator, signed_shift(common.conversion))
    laws = pair_laws(kernel=kernel, applications=facts, terms=terms)
    held = _held_bytes(problem, output, common, data)
    name, factor, forecast, recipe, selection = route_name(estimator.product_formula), Fraction(1), None, None, None
    if qsp:
        from .sampled_qsp import forecast_qsp
        forecast = forecast_qsp(method, common, kernel, facts)
        factor = forecast.factor
    elif name != "plain":
        from . import sampled_recipes
        support, scales = _recipe_support(method, kernel, facts, l_terms, h_terms, q)
        if name == "ptsc":
            factor = Fraction(estimator.product_formula.mu_cap) ** 2
        else:
            select = sampled_recipes.select_richardson if name == "richardson" else sampled_recipes.select_mpf
            selection = select(method, support)
            factor = Fraction(selection[0]["multiplier_bound"])
    clusters = (_sufficient_rows(method, laws, factor, output, accuracy) if accuracy is not None
                else shots // estimator.repeats)
    # Known bytes held from the first law to the assembled Plan, admitted
    # before any row array or draw: H (_held_bytes) and R (_row_bytes). The
    # Plan's one Setting and range axis over the rows (`_pair_batch`) do not
    # grow with B.
    buckets = (len(forecast.bank.buckets) if forecast.bank is not None else 1) if qsp else estimator.step_buckets
    held += _row_bytes(clusters, facts, kernel, terms, buckets, qsp=qsp)
    # A recipe's parameters, bounds, generation work and saved arrays are
    # selected and admitted from support-wide facts and B before any row is
    # drawn.
    if not qsp and name != "plain":
        from . import sampled_recipes
        if name == "ptsc":
            fields, operator, recipe_work, size = sampled_recipes.select_ptsc(
                method, support, h_count=len(h_terms), l_count=len(l_terms), clusters=clusters)
            rule, recipe_reasons = "weighted", ()
        else:
            fields, operator, rule, recipe_reasons = selection
            recipe_work, size = 0, sampled_recipes.member_bytes(clusters)
        held += size
        recipe = (fields, operator, rule, recipe_reasons, support, scales, recipe_work)
    bank = forecast.bank if qsp else None
    _check_bytes(held + (0 if bank is None else bank.retained_array_bytes), method.max_bytes,
                 "sampled LCHS held data, pair rows, their laws, settings and experiments"
                 + (" and the QSP bank's known arrays" if bank is not None else ""))
    bias, census_work = None, 0
    if not qsp and name == "plain":
        bias, census_work = _pair_bias(method, problem, common, data, kernel, facts, laws.frame, terms, l_terms,
                                       h_terms, held=held)
    if not qsp and method.lcu_select == "structured":
        # Two-entry tables per occurrence of the longest word: 2L - 1 for the
        # plain and Richardson Strang word, 4L - 3 for the MPF macrostep and
        # the unmerged H and L lists for a PTSC Lie step.
        labels = len(set(dict(l_terms)) | set(dict(h_terms)))
        width = {"mpf": 4 * labels, "ptsc": len(l_terms) + len(h_terms)}.get(name, 2 * labels)
        if exceeds(clusters * max(1, width), method.max_select_work):
            raise ResourceLimitError(f"structured pair SELECT tables of {clusters} rows need {clusters * max(1, width)} "
                                     f"work units, above LCHS.max_select_work={method.max_select_work}")
    for state in (problem.initial_state,) + (() if problem.source is None else (problem.source,)):
        spec = state.preparation
        _check_bytes(spec.payload_bytes, method.max_bytes, "LCHS input PREP")
        if exceeds(spec.work, method.max_select_work):
            raise ResourceLimitError(f"LCHS input PREP exceeds max_select_work: it needs {spec.work} work units, "
                                     f"max_select_work={method.max_select_work}")
    return _Pass(common=common, data=data, kernel=kernel, facts=facts, terms=terms, l_terms=l_terms, h_terms=h_terms,
                 laws=laws, clusters=clusters, held=held, recipe=recipe, bias=bias, census_work=census_work,
                 qsp=forecast)


def sampled_accuracy_shots(method, problem, *, output, accuracy):
    """Return the shots B*m that `plan_sampled` selects for a sampling-accuracy request, without drawing or fitting.

    `LCHS.sampling_shots` reaches it through ``selection.accuracy_shots``.
    The exact no-evolution readout keeps its own count
    (``host.initial_accuracy_shots``). Otherwise one deterministic pass
    (`_forecast`) gives B, and the count is ``B * SampledPairs.repeats``: the Program
    runs m shots of each of the B rows, while the radius counts B.
    `plan_sampled` in the same planning call reads this pass
    (`_accuracy_forecast`) and checks the count. A ValueError from the pass
    means `plan_sampled` would refuse the same B before any draw, for
    example when the row arrays exceed ``max_bytes``.
    """
    _sampled_applicability(method, output, "quantum")
    if _exact_readout(problem):
        from .host import initial_accuracy_shots
        from .selection import _route
        _route(method, problem, output=output, execution="quantum", accuracy=accuracy)
        return initial_accuracy_shots(method, problem, output=output, accuracy=accuracy)
    return _accuracy_forecast(method, problem, output, accuracy).clusters * method.estimator.repeats


def _accuracy_forecast(method, problem, output, accuracy):
    """Return `_forecast` for an accuracy request, once per planning call (`core.planning.planning_memo`)."""
    from nwqlib.core.planning import planning_memo
    return planning_memo("lchs_sampled_forecast", (method, problem, output, accuracy),
                         lambda: _forecast(method, problem, output, accuracy=accuracy))


def _qsp_pair_block(method, problem, rows, payload, rec, forecast):
    """Return the row-parameterized sampled QSP pair block: inputs inside, ports branch, system and success flags.

    No CX law is given: the controlled children queries have no law with two
    controls, and `sampled_qsp.qsp_pair_resources` counts the queries. The
    construction size rule is a coarse upper value per row,
    ``sum_branches (c d + 1) u + 8 q + 8``, with d the member's signals, c = 1
    (direct walk) or 2 (lift) scaled-query calls per signal and
    ``u = 16 (1 + q + F)(1 + P)`` gate units per scaled-query call for P
    child units (`sampled_qsp.child_units`), F flags and q system qubits; the
    largest row is used. The work of the branch queries that one pair circuit
    builds (`sampled_qsp.forecast_qsp`) is added, and a supplied gate whose
    definition is not built leaves the construction work unresolved.
    """
    from .sampled_qsp import child_units, row_evolutions
    record = rec.pair_qsp
    q, flags = payload.num_system_qubits, record.flag_qubits
    system = "zero" if payload.prepares_inputs else "live"
    ports = (("branch", 1, "zero", "unitary"), ("system", q, system, "unitary"), ("flags", flags, "zero", "unitary"))
    signature = BlockSignature(name="lchs_pair", target=_source(PAIR_TARGET),
        quantum=tuple(QuantumPort(name=name, width=width, requires=requires, ensures=ensures)
                      for name, width, requires, ensures in ports if width),
        parameters=(Parameter(name="row", domain="integer", lower=0, upper=rows.clusters - 1),), coupling="joint")
    stored = sum(child_units(child) for child in (payload.l_child, payload.h_child) if child is not None)
    unit = 16 * (1 + q + flags) * (1 + stored)
    largest = 0
    for branches in row_evolutions(payload, rows, record):
        work = 0
        for _, evolution in branches:
            calls = (2 if evolution.signal_access == "generic_lift" else 1) * evolution.degree
            work += (calls + 1) * unit
        largest = max(largest, work)
    semantics = BlockSemantics(kind="unknown",
        input=InputRef(identity=rows.content_id, representation="lchs_pair_rows", source=METHOD),
        basis=problem.basis,
        relation="row r: (|0>P_x|psi_x> + exp(i*theta_r)|1>P_y|psi_y>)/sqrt(2) with P_b the selected block of the "
                 "row's bank member on the scaled branch query, approximately U_b/zeta_b, then H on the branch "
                 "qubit and the basis change of the row's Pauli term",
        input_projector=("branch qubit, system and flags zero" if payload.prepares_inputs else
                         "branch qubit and flags zero; system holds the prepared initial direction"),
        output_projector="every flag zero (the success projector of both QSP sequences)",
        success="all flags zero", workspace=flags, restoration="flags measured with the branch and system",
        epsilon=None, approximation_metric="recovered mean amplitude of each branch polynomial",
        approximation_evidence="QSP bank bias relation in the Plan reconstruction (pair_bias, pair_qsp)",
        inverse_legal=False, control_legal=False,
        phase="row phase theta applied once at the branch qubit; the scalar centering phase is in the branch phases; "
              "a controlled input preparation keeps its global phase as the relative input phase")
    record_block = SelectedDefinition(signature=signature, semantics=semantics, implementation=signature.target,
        choice=f"qsp_bank:{record.policy}:buckets={len(record.bucket_times)}", decomposition=None, cost_law=None,
        construction_work=None if forecast.query_work is None else largest + 8 * q + 8 + forecast.query_work,
        resource_laws=(), coefficient_selection_id=rows.content_id,
        cost_context="sampled QSP pair of one bound row; sampled_pair_resources counts its queries, reflections, "
                     "rotations and input preparations, and no CX law is derived for the controlled queries"
                     + ("" if forecast.query_unresolved is None else
                        f"; construction work unresolved: {forecast.query_unresolved}"),
        workspace=(Workspace(location="host", purpose="workspace", bytes=None, source=METHOD),))
    return SelectedBlock.bind(record_block, payload=(payload, rec, problem), constructor=construct_pair)


def plan_sampled(method, problem, *, output, execution, shots, rng, accuracy=None):
    """Plan a sampled-pair LCHS observable, or the exact no-evolution readout at zero time or zero input.

    With ``accuracy`` the shots are the count `sampled_accuracy_shots`
    returned; this pass repeats its deterministic choice (`_forecast`),
    checks that count and records the request in ``Plan.selection_accuracy``.
    A sampled QSP Plan then prepares its bank once, before the rows are
    drawn; the rows and the members are drawn once from the Method's stream.

    Raises:
        ApplicabilityError: For an output other than NormSquared,
            QuadraticForm or NormalizedExpectation, classical execution,
            shots that are absent or not divisible by ``SampledPairs.repeats``, a
            coherent-only choice (`_sampled_applicability`), and, when the
            state evolves, ``init_prep`` other than
            ``"direct"``, an A that the route cannot evolve (product formulas
            need Pauli parts; QSP also takes band parts, supplied encodings
            of Hermitian parts and one supplied encoding of a general A), a
            supplied L or general A whose missing error bound leaves no PSD
            shift or norm bound (`selection.compact_common`), a range or
            step limit of the route, or a circuit that measures more than the
            64 bits of one count index, which the pair analysis decodes.
        ValueError: For shots that differ from an accuracy request's count,
            or an exceeded work or byte limit. A row count whose row arrays
            exceed ``max_bytes`` is refused before the bank is prepared or
            any row is drawn.
    """
    from .sampled_qsp_bank import prepare_sampled_qsp_bank
    _sampled_applicability(method, output, execution)
    estimator = method.estimator
    repeats = estimator.repeats
    if accuracy is None and (shots is None or shots % repeats):
        raise ApplicabilityError(f"sampled pairs need finite shots divisible by SampledPairs.repeats={repeats}, "
                                 f"which gives shots/repeats rows; got shots={shots}")
    if _exact_readout(problem):
        from .selection import plan_lchs
        return plan_lchs(method, problem, output=output, execution=execution, shots=shots, rng=rng,
                         accuracy=accuracy)
    p = (_forecast(method, problem, output, shots=shots) if accuracy is None
         else _accuracy_forecast(method, problem, output, accuracy))
    clusters, common, data, kernel, facts = p.clusters, p.common, p.data, p.kernel, p.facts
    if accuracy is not None and shots != clusters * repeats:
        raise ValueError(f"shots={shots} differ from the {clusters}*{repeats} that this sampling accuracy selects; "
                         "pass the count of LCHS.sampling_shots")
    generator, q = common.generator, common.generator.num_qubits
    measured = 1 + q + (0 if p.qsp is None else p.qsp.flag_qubits)
    if measured > 64:
        raise ApplicabilityError(f"sampled-pair LCHS readout needs {measured} classical bits, more than its 64-bit "
                                 "count decoder supports")
    prepared = None if p.qsp is None or p.qsp.bank is None else prepare_sampled_qsp_bank(p.qsp.bank)
    rows = draw_pair_rows(rng=rng, clusters=clusters, repeats=repeats, laws=p.laws,
                          scalar_phase=(generator.dissipative_extracted, generator.hamiltonian_extracted))
    has_source = problem.source is not None
    term_labels = tuple(label for label, _ in p.terms)
    pair_recipe = pair_qsp = None
    bias, census_work, step_counts = p.bias, p.census_work, ()
    if p.qsp is not None:
        from .sampled_qsp import bind_rows, expected_pair_signals, generator_error, qsp_pair_bias, qsp_record
        binding = bind_rows(p.qsp, rows, prepared, rng)
        # The QSP bank applies no product formula; its payload keeps the order field at 2.
        payload = LCHSSampledPayload(num_system_qubits=q, l_terms=(), h_terms=(), order=2,
                                     implementation="qsp", control_copies=1, term_labels=term_labels,
                                     prepares_inputs=has_source, max_bytes=method.max_bytes, qsp_bank=prepared,
                                     l_child=p.qsp.l_child, h_child=p.qsp.h_child)
        term = generator_error(p.qsp, prepared)
        bias = qsp_pair_bias(method, problem, common, data, kernel, facts, p.laws.frame, p.terms, p.qsp, prepared,
                             term)
        pair_qsp = qsp_record(method, p.qsp, prepared, binding, term, expected_pair_signals(p.laws, p.qsp))
        # The SELECT construction of a sampled QSP Plan is one pair circuit, whose branch queries forecast_qsp
        # counted (None when a supplied definition is not built).
        census_work = p.qsp.query_work
    else:
        implementation = "structured" if method.lcu_select == "structured" else "multiplexor"
        payload = LCHSSampledPayload(num_system_qubits=q, l_terms=p.l_terms, h_terms=p.h_terms,
                                     order=method.ham_evo.order, implementation=implementation,
                                     control_copies=estimator.control_copies, term_labels=term_labels,
                                     prepares_inputs=has_source, max_bytes=method.max_bytes)
        if p.recipe is None:
            step_counts = bias.bucket_steps
        else:
            from . import sampled_recipes
            fields, operator, rule, recipe_reasons, support, scales, census_work = p.recipe
            if route_name(estimator.product_formula) == "ptsc":
                pair_recipe = sampled_recipes.draw_compensation(fields, rng, rows, p.l_terms, p.h_terms, q)
                step_counts = (pair_recipe.steps,)
            else:
                pair_recipe = sampled_recipes.draw_members(fields, rng, clusters)
                step_counts = tuple(sorted(set(pair_recipe.member_steps)))
            bias = _recipe_bias(method, problem, common, data, kernel, facts, p.laws.frame, p.terms, support, scales,
                                operator, rule, recipe_reasons)
    discrete = isinstance(kernel, DiscreteKernel)
    quadrature = None if data is None else data.quadrature
    sources = [item for item in facts.applications if item.kind == "source"]
    conversion = common.conversion
    rec = LCHSReconstruction(elapsed_time=problem.elapsed_time, dimension=problem.dimension,
        encoded_dimension=problem.dimension, mode="unitary" if common.l_norm == 0 else "quadrature",
        has_source=has_source, initial_scale=problem.initial_state.preparation.physical_scale,
        source_scale=problem.source.preparation.physical_scale if has_source else None,
        source_nodes=tuple(item.start_time for item in sources), source_weights=tuple(item.weight for item in sources),
        terms=p.terms, coefficient_l1_norm=quadrature.coefficient_l1_norm if discrete else 0.0,
        physical_branches=len(quadrature.k_nodes) if discrete else 0,
        selected_backend=route_name(method.ham_evo), selected_select="sampled_pairs",
        psd_premise=conversion["psd_premise_evidence"] if common.psd_premise_satisfied else "unavailable",
        psd_shift=conversion["psd_shift"], scalar_decay=conversion["scalar_decay"], l_norm=common.l_norm,
        kernel_approximation_bound=(quadrature.coefficient_plan.quadrature.approximate_lchs_error_bound if discrete
                                    else kernel.kernel_error_bound),
        quadrature_bound=quadrature.coefficient_plan.quadrature.quadrature_error_bound if discrete else None,
        step_counts=step_counts, select_work=census_work, pair_rows=rows, pair_bias=bias, pair_recipe=pair_recipe,
        pair_qsp=pair_qsp)
    if pair_qsp is not None:
        pair = _qsp_pair_block(method, problem, rows, payload, rec, p.qsp)
    else:
        pair = _pair_block(method, problem, rows, payload, rec, row_select_cx(payload, rec) + row_correction_cx(rec))
    blocks = (pair,) if has_source else (select_preparation("initial_prep", problem.initial_state), pair)
    copies = payload.control_copies
    flags = 0 if pair_qsp is None else pair_qsp.flag_qubits
    registers = [Register(name="branch", width=1, role="clean_ancilla")]
    if copies > 1:
        registers.append(Register(name="branch_copy", width=copies - 1, role="clean_ancilla"))
    registers.append(Register(name="system", width=q, role="system"))
    if flags:
        registers.append(Register(name="flags", width=flags, role="clean_ancilla"))
    definitions = [Definition(id=f"allocate_{item.name}", node=Allocate(wire=item.name)) for item in registers]
    definitions += [Definition(id=f"release_{item.name}", node=Release(wire=item.name)) for item in registers]
    prepare = ()
    if not has_source:
        definitions.append(Definition(id="initial_prep", node=BlockCall(signature="initial_prep",
                                                                        ports=(PortMap(port="system", wire="system"),))))
        prepare = ("initial_prep",)
    measures = ("measure_branch", "measure_system") + (("measure_flags",) if flags else ())
    definitions += [
        Definition(id="pair", node=BlockCall(signature="lchs_pair",
            ports=tuple(PortMap(port=port.name, wire=port.name) for port in pair.record.signature.quantum),
            arguments=(Argument(parameter="row", value=ExprRef(expression="row")),))),
        Definition(id="measure_branch", node=Measure(wire="branch", result="readout_branch")),
        Definition(id="measure_system", node=Measure(wire="system", result="readout_system")),
        *((Definition(id="measure_flags", node=Measure(wire="flags", result="readout_flags")),) if flags else ()),
        Definition(id="body", node=Sequence(children=(
            *(f"allocate_{item.name}" for item in registers), *prepare, "pair", *measures,
            *(f"release_{item.name}" for item in registers)))),
    ]
    batch, experiment = _pair_batch(clusters, repeats, rows.content_id)
    definitions.append(Definition(id=BATCH, node=batch))
    classical = (ClassicalValue(name="readout_branch", dtype="bits", width=1),
                 ClassicalValue(name="readout_system", dtype="bits", width=q),
                 *((ClassicalValue(name="readout_flags", dtype="bits", width=flags),) if flags else ()))
    program = Program(root=BATCH,
        definitions=tuple(definitions), registers=tuple(registers), classical=classical,
        parameters=(Parameter(name="row", domain="integer", lower=0, upper=clusters - 1),),
        expressions=(Expression(id="row", value=ParameterRef(parameter="row")),),
        signatures=tuple(block.record.signature for block in blocks))
    construction = SelectedConstruction(program=program, selections=tuple(block.record for block in blocks))
    from .analysis import error_model
    plan = Plan(problem=problem, method=method, output=output, execution="quantum", shots=clusters * repeats,
        randomness=rng.snapshot(), selection_accuracy=accuracy, construction=construction,
        experiments=(experiment,), reconstruction=rec,
        error_model=error_model(problem, output, construction, shots=clusters * repeats),
        assumptions=("construction tolerance is not a total physical output error bound",
                     "sampled-pair confidence holds for the complete predeclared row table only",
                     *((ASSERTED_L_BOUNDS,) if rec.psd_premise == "user_assertion" else ())))
    return plan._bind(blocks=blocks, native_data=payload)
