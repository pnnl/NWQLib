"""LCHS reconstruction records and the results in the original coordinates."""

from typing import Annotated, Literal

from pydantic import Field

from nwqlib._quantum_readout import ReadoutSetting
from nwqlib.algorithms.protocol import AlgorithmDescriptor
from nwqlib.artifacts import ArtifactManifest
from nwqlib.core.analysis import Result
from nwqlib.core.records import ContentID, FrozenArray, Nonnegative, PositiveInt, Real, Record, Source, Text
from nwqlib.execution import KernelApplication
from nwqlib.operators.access import Count
from nwqlib.problems.inputs import PhysicalScale, compose_recovery
from .sampled_moments import LCHSPairMoments
from .sampled_qsp import LCHSPairQSP
from .sampled_recipes import LCHSPairCompensation, LCHSPairMembers
from .sampled_rows import LCHSSampledRows

# The qualification that a user_assertion PSD premise adds to every consumer of the shift. It never becomes verified.
ASSERTED_L_BOUNDS = ("the bounds of L are the caller's assertion, not verified by NWQLib; every shift, kernel grid and "
                     "error bound that uses them is conditional on it")
# What chose the route under fixed shots, by LCHSFixedShotPrecision.ranking, for the sentence that print(plan) adds.
FIXED_SHOT_RULES = {
    "cx_per_circuit": "by CX per circuit",
    "original_u_calls": "by fewest original-U calls",
    "original_u_calls_and_order": ("without a known CX count, by fewest original-U calls within its alternative and grid "
                                   "and by the order of the alternatives and grids between them"),
}

METHOD = Source(name="lchs", version="4", domain="constant du/dt=-A u+b",
    reference="An, Childs and Lin, arXiv:2312.03916v2 Eqs. (4), (6)-(7), (61), (186); finite tail-budget inversion; Trefethen ATAP ISBN 978-1-61197-239-9, Theorem19.3, n=Q-1")
DESCRIPTOR = AlgorithmDescriptor(method="lchs", version="4",
    problem_families=("linear_dynamics",),
    output_families=("solution", "state_vector", "norm_squared", "quadratic_form", "normalized_expectation", "samples"),
    access_families=("dense", "periodic_stencil"),
    resource_coverage=("selected native recipe and declared host numerical work",),
    evidence_coverage=("physical reconstruction and applicable conditional component bounds",),
    limitations=("dense SELECT uses classically exponentiated branches",
                 "construction tolerance does not bound total physical output error",
                 "floating-point and provider/model errors remain separate"), references=(METHOD,), maintenance="NWQLib")


class LCHSPairBias(Record):
    """Inner-evolution steps and deterministic bias bounds of a sampled-pair LCHS Plan.

    `LCHSReconstruction.pair_bias` holds it, chosen once at planning. Write
    M_i for the physical mass of application i (the initial atom, a
    Gauss-Legendre source node or the continuous source), S = sum_i M_i,
    and B = S*beta with beta = sum_j |c_j| for discrete kernel coefficients
    or the importance-weight cap W_k of a continuous kernel. B bounds the
    mean branch weight E|w| and the norm of the selected vector v for the
    exact weights a/q. The rounding of the stored weights is not enclosed,
    so the bounds below hold for the actual weights only up to that
    rounding. The bias values are in the frame of the row table, divided by
    g**2 for its weight frame g, and are binary64 evaluations of the
    exact-real bounds, not outward-rounded enclosures. A positive value
    below the binary64 range is given as 2**-1074, which still bounds it.

    Product-formula stage. A branch at elapsed time t whose node k lies in
    bucket b runs at least r_b order-p product-formula steps, so its
    operator error is ``delta <= Wbar*t**(p+1)/r_b**p``, with Wbar the
    Pauli-triangle census of its coefficients k*l_P + h_P in the emitted
    label order (Childs et al., doi:10.1103/PhysRevX.11.011020,
    Propositions 9-10). A row runs max(r_b(x), r_b(y)) steps on both
    branches, so each branch keeps its own bound whatever its partner. For
    independent branch draws,

        |E Z_N - ||v||**2| <= E[|w_x w_y| (delta_x + delta_y)] <= 2*B*D,

    with D >= E[|w| delta]. ``inner_relation`` names how D is formed:

    - ``"finite_sum"`` (finite applications and discrete nodes): the integer
      probabilities cancel in E[|w| delta], so
      D = (sum_i M_i*t_i**(p+1)) * (sum_j |c_j|*Wbar_j/r_b(j)**p).
    - ``"bucket_maximum"`` (a continuous source time or kernel node):
      D = B*E_sup with E_sup = T_e**(p+1)*max_b Wbar_b/r_b**p, where
      Wbar_b bounds the census of every node of bucket b (the census of the
      majorant coefficients kappa_b*|l_P| + |h_P| for a continuous kernel,
      the largest node census of a discrete bucket). This bound holds for
      any law supported on the buckets, so no bucket probability enters.

    In both, ``product_formula_operator`` is E = D/B and
    ``inner_bias`` is 2*(B/g)**2*E. The census covers the rounded
    coefficients that the circuit forms. Their difference from the exact
    k*l_P + h_P adds at most t*sum_P |rounded - exact| to a branch's error
    and is not bounded here.

    Recipe stage. With ``SampledPairs.product_formula`` other than ``"plain"``
    the rows run the recipe of `LCHSReconstruction.pair_recipe`
    (`sampled_recipes`), and ``inner_relation`` names its relation. Those
    Plans have no buckets and no operator bound E, and ``inner_bias`` is the
    recipe's bias bound of N in the row frame:

    - ``"richardson"``: (B/g)**2*(B_ext + E_eff), the XS9 remainder of the
      m = 2 levels plus the exact coefficient discrepancy of the stored
      multipliers, or None when the XS9 premise fails for fixed steps.
    - ``"ptsc"``: (B/g)**2*(2*delta + delta**2) with the uniform branch bound
      delta of the Taylor tail and the finite-law count perturbation.
    - ``"mpf"``: None for two or more distinct labels, because no uniform
      finite-step remainder is admitted, and 0 for exact evolution.

    QSP stage. With ``ham_evo="qsp_block_encoding"``
    the rows run the finite phase bank of `LCHSReconstruction.pair_qsp`
    (`sampled_qsp`), ``inner_relation`` is ``"qsp_bank"`` and
    ``inner_bias`` is (B/g)**2*(2*beta + beta**2), the independent-member
    pair factor beta_x + beta_y + beta_x*beta_y (B13 of the finite-bank relations in
    `sampled_qsp`; a deterministic bank is its one-member case, B15) with one
    support-wide recovered amplitude bound beta (`sampled_qsp.qsp_pair_bias`).
    Those Plans have no buckets, step rule or operator bound E.

    Model stage. ``model_vector_bound`` Delta_v bounds ||u - v|| for the
    physical solution u: S times the kernel and k-quadrature bounds of a
    discrete kernel, or the kernel and endpoint-guard bounds of a continuous
    one, plus the Gauss-Legendre bound of a discrete source rule, plus the
    coupling of the finite random bits of a continuous draw. Since
    ||v|| <= B and ||u|| <= B + Delta_v,
    ``| ||v||**2 - ||u||**2 | <= 2*B*Delta_v + Delta_v**2``, which is
    ``model_bias`` in the row frame. QuadraticForm bounds are
    ``observable_bound`` times the NormSquared bounds, with
    observable_bound = sum_l |o_l| >= ||O||.

    Attributes:
        step_rule: ``"fixed"`` (the Method's ``Trotter.steps``) or
            ``"weighted"`` (steps selected to meet an allowance:
            ``step_allowance`` for the plain relations, the recipe's
            ``pf_allowance`` otherwise), None for a QSP bank.
        inner_relation: ``"finite_sum"`` or ``"bucket_maximum"`` for the
            plain product formula, the recipe ``"richardson"``, ``"ptsc"``
            or ``"mpf"``, or ``"qsp_bank"``, as above.
        bucket_edges: Upper ends kappa_b of the |k| buckets, increasing. A
            branch belongs to the first bucket with |k| <= kappa_b. The last
            end covers the whole node support. Empty for a recipe.
        bucket_steps: Step count r_b of each bucket. Empty for a recipe.
        step_allowance: The allowance on E that ``"weighted"`` plain steps
            meet, 0.1 times ``approx_tol``, or None for fixed
            steps and recipes. ``"finite_sum"`` steps meet it with
            T_e**(p+1) in place of the mass-weighted time power, a larger
            value.
        product_formula_operator: E = D/B of the plain relations, or None
            when unknown or for a recipe.
        mean_weight_bound: B as a PhysicalScale.
        model_vector_bound: Delta_v as a PhysicalScale, or None when a
            component is unknown.
        inner_bias: The bias bound of N in the row frame from the branch
            evolutions by ``inner_relation``, 2*(B/g)**2*E for the plain
            relations, or None when unknown.
        model_bias: (2*B*Delta_v + Delta_v**2)/g**2, or None when unknown.
        observable_bound: sum_l |o_l| of the observable terms, 0 for a
            NormSquared output.
        kernel_allowances: ``(kernel, guard)`` allowances of a continuous
            kernel, each half of ``approx_tol``, else None.
        reasons: Why a component is unknown, empty when every component is
            known.
    """

    step_rule: Literal["fixed", "weighted"] | None
    inner_relation: Literal["finite_sum", "bucket_maximum", "richardson", "ptsc", "mpf", "qsp_bank"]
    bucket_edges: tuple[Nonnegative, ...] = ()
    bucket_steps: tuple[PositiveInt, ...] = ()
    step_allowance: Nonnegative | None = None
    product_formula_operator: Nonnegative | None
    mean_weight_bound: PhysicalScale
    model_vector_bound: PhysicalScale | None
    inner_bias: Nonnegative | None
    model_bias: Nonnegative | None
    observable_bound: Nonnegative = 0.0
    kernel_allowances: tuple[Nonnegative, Nonnegative] | None = None
    reasons: tuple[Text, ...] = ()

    def norm_bias(self):
        """Return the NormSquared bias bound in the row frame, ``inner_bias + model_bias``, or None when unknown."""
        if self.inner_bias is None or self.model_bias is None:
            return None
        return self.inner_bias + self.model_bias

    def bucket_of(self, nodes):
        """Return the bucket index of each node, the first b with |k| <= kappa_b."""
        import numpy as np
        index = np.searchsorted(np.asarray(self.bucket_edges), np.abs(np.asarray(nodes, dtype=float)), side="left")
        if np.any(index >= len(self.bucket_edges)):
            raise ValueError("a drawn node lies beyond the last product-formula bucket")
        return index


def _rounded_scale(scale):
    """Four significant digits of a PhysicalScale for a printed line, in binary form when binary64 cannot hold it."""
    value = scale.as_float()
    return f"{scale.mantissa:.4g}*2**{scale.exponent}" if value is None else format(value, ".4g")


class LCHSFixedShotPrecision(Record):
    """What the coherent route chosen under fixed `shots` gives up in precision, beside the count that ranked it.

    Under fixed shots, planning ranks the routes by CX per circuit. When their
    CX counts are unknown, it takes the fewest original-U calls within one
    alternative on one kernel and quadrature grid, and between alternatives
    and grids it takes the first in their fixed order, so the chosen route can
    have more calls than another compared route. These rankings leave out
    the recovery factor `Gamma`, the factor that multiplies the measured
    success amplitudes, and routes can differ in it by large factors. For a
    moment measured with every shot, the
    Hoeffding radius scales exactly as `Gamma**2` at fixed shots `N` and
    statistical model, with `N` independent stationary shots. For the same
    absolute radius, failure probability, observable range and setting
    allocation, the unrounded sufficient count scales as `Gamma**4`. For a
    success-conditional output the accepted probability is `q/Gamma**2`, with
    `q` the squared norm of the circuit's approximate vector, so it scales as
    `1/Gamma**2` only if `q` is the same, and different kernels or filters can
    change `q`. These relations control sampling about the moment of the finite
    circuit. They bound no error against the exact solution.

    `LCHSReconstruction.fixed_shot_precision` holds it, when a compared route
    has a smaller recovery. `print(plan)` and `print(result)` show a
    rounded sentence built from these fields. When the selected route has a
    known CX count, an `accuracy` request instead ranks the routes by shots
    times CX, which contains `Gamma**4`.

    Attributes:
        readout: `"unconditional_counts"`, a moment measured with every shot,
            or `"conditional_counts"`, a success-conditional output.
        ranking: What chose the route: `"cx_per_circuit"`; `"original_u_calls"`
            when the CX counts are unknown and one alternative on one grid was
            compared; or `"original_u_calls_and_order"` when they are unknown
            and several alternatives or grids were compared, so that calls
            decide only within one alternative on one grid and the fixed order
            decides between them.
        selected_row: The selected route, as `alternative:generator:route[level]`,
            where the alternative is `lchs` (the kernel sum), `filter` or
            `product`. Under `lchs_kernel="auto"` it ends with `on grid` and
            the label of its kernel and quadrature pair, from the
            [list of pairs](../../algorithms/lchs.md#automatic-kernel-and-grid).
        smallest_row: The route with the smallest recovery among the admitted
            routes this planning pass compared, in the same form. Under
            `lchs_kernel="auto"` only the pairs that a Method can name enter, so
            a Method that names this route in `QSPBlockEncoding.route` and its pair in
            `lchs_kernel` and `k_quadrature` gives this recovery. This does not
            hold for a commuting product, because that route has no parameter
            for its H factor, so no Method names that route.
        smallest_recovery: `Gamma` of `smallest_row`. `Gamma` of `selected_row`
            is `LCHSReconstruction.recovery`.
        factor: `(Gamma_selected/Gamma_smallest)**2`, unrounded, the widening
            of the radius at equal shots. For a success-conditional output at
            equal `q`, its inverse is the factor on the accepted probability.
    """

    readout: Literal["unconditional_counts", "conditional_counts"]
    ranking: Literal["cx_per_circuit", "original_u_calls", "original_u_calls_and_order"]
    selected_row: Text
    smallest_row: Text
    smallest_recovery: PhysicalScale
    factor: PhysicalScale


class LCHSReconstruction(Record):
    """Recovery scale and coordinate mapping of an LCHS Plan, never a second scientific input.

    Attributes:
        elapsed_time: Final time minus initial time.
        dimension: Original physical coordinate dimension.
        encoded_dimension: Dimension including any coherent dummy coordinates.
        mode: "quadrature" (finite LCHS sum), "unitary" (a quantum Plan
            whose L is exactly zero), "initial" (zero elapsed time, so the
            initial vector is the solution) or "zero" (zero initial state and
            no nonzero source acting over a positive time, so the solution is
            zero).
        has_source: Whether the original problem includes a constant source.
        initial_scale: Original initial-vector norm in binary physical-scale form.
        source_scale: Original source-vector scale, when the problem has a source.
        recovery: Physical scale that multiplies the success-projected
            amplitudes. For LCHS evolution it composes the coefficient 1-norm,
            any QSP amplitude factor, without a constant source the
            initial-state norm, and the power of two 2**e that the
            coefficients omit (see coefficient_l1_norm). A no-evolution
            readout uses the initial-state norm alone. None for classical
            Plans and for no-evolution Plans answered directly from the
            input. A unit-normalized output does not depend on it.
        success_bits: Qubit indices of the success projector.
        system_bits: Ordered encoded coordinate qubits.
        terms: Pauli labels and coefficients of the observable.
        settings: Readout settings run: the one exact ``projected_moments``
            reduction, the amplitude or Samples readout, or one counts
            setting per qubit-wise commuting group (its label is the group's
            accumulated basis), the physical-mass counts setting of a
            NormSquared output and, for a padded normalized output, the
            unrotated ``physical_mass`` setting.
        groups: Member labels of each group setting, in first-fit order,
            aligned with the settings named ``group_<g>``.
        grouping_comparisons: Comparisons evaluated by the qubit-wise commuting
            (QWC) grouping of sampled terms, counted against max_readout_work.
        select_work: Sum of the SELECT construction work counted at planning,
            including the decomposition, selection and classification work that
            is checked separately against the max_select_work limit of each
            construction stage. `None` when part of that construction cannot
            be counted at planning, as for the controlled query of a supplied
            gate whose definition is not built. The SELECT block's
            `cost_context` gives the reason.
        coefficient_l1_norm: Default 0, which the modes "initial" and "zero"
            keep. Numerical one-norm of the coefficient table that this Plan
            uses, without the power of two 2**E that its recovery carries.
            For a quantum Plan without a constant source, and for the
            coefficient summary of a classical Plan, the table is the kernel
            quadrature rule multiplied by the signed compensation exp(s*T),
            with s = psd_shift - scalar_decay, and by any extracted scalar
            phase, then divided by 2**E. Here E = e - 1 - k, where
            exp(s*T) = mu*2**e with 0.5 <= mu < 1, and k is the binary lift
            exponent: multiplication by 2**k brings every nonzero real and
            imaginary coefficient part into the normal binary64 range.
            This one-norm is dimensionless.
            A classical Plan evolves the raw rule and does not use this
            normalization for its physical vector. For a quantum Plan with a
            constant source, the table holds the rule once per application,
            each copy multiplied by that application's input norm, its time
            weight, exp(s*t_a) for its elapsed time t_a and any extracted
            scalar phase. Then E = max_a(e_a) - 1 - k, where application a
            has the mass mu_a*2**e_a, and this one-norm has the solution's
            unit. A PeriodicStencil Plan instead shows the one-norm of the
            raw kernel coefficients. Its recovery computes their magnitudes
            without underflow, so this binary64 sum need not keep the
            relative precision of a complex subnormal magnitude. The
            ``recovery`` field carries the complete physical scale of a
            quantum output. This field is neither the solution norm nor an
            error bound. With `ham_evo` equal to
            `"qsp_block_encoding"` the coefficient state is the product of a
            kernel factor and an application factor, and this field is the
            one-norm `sum_j |c_j|` of the kernel's k-quadrature coefficients
            alone, because `recovery` carries the application masses; it is
            1 when the imaginary-time filter or the commuting product is
            selected, since those routes use no kernel.
        physical_branches: Nonpadding entries of that table, the k nodes of
            one application, or for a quantum constant-source Plan the k
            nodes of the initial application and of every Duhamel node. With
            `"qsp_block_encoding"` it is the number of kept applications
            times the number of k nodes, or times 1 for the imaginary-time
            filter and the commuting product; an application left out under
            `source_drop` is not counted.
        padded_branches: Number of address slots of that table, a power of
            two that includes the padding slots. A classical Plan runs no
            SELECT, and the value is the number of address slots its kernel
            table would need. With `"qsp_block_encoding"` it is
            `2**(kernel qubits + application qubits)` of the coefficient
            state.
        selected_backend: Hamiltonian-evolution implementation chosen once
            at planning: the Method's `ham_evo`, or
            for `"auto"` the backend chosen from how A is given
            (`"dense_exact"` or `"qsp_block_encoding"`). Loading and running
            the Plan use this value.
        selected_select: Concrete SELECT implementation or identity route.
        psd_premise: "numerical" (the scale-aware eigenvalue check of
            _numerical_psd_decision passed, after any shift),
            "analytic_periodic" (PSD by construction for a PeriodicStencil),
            "certified_bound" (exact structural bounds of a compact
            generator), "user_assertion" (the asserted bounds of a supplied
            encoding, or a `dissipative_lower_bound` given to
            `ingest_cartesian` that improves on the structural bound, which
            NWQLib does not verify, so the shift and the bounds that use it
            are conditional on it) or "unavailable".
        psd_shift: Applied nonnegative operator shift s_+, with physical
            growth restored elsewhere. The imaginary-time filter and the
            commuting product apply no such shift, because their recovery
            comes from the normalization of their encoding, and for them
            this field and `scalar_decay` report the compensation of the
            lower bound of L.
        scalar_decay: Nonnegative known decay rate c = max(0, ell) taken out
            of a compact generator whose L has the lower bound ell, restored
            as exp(-c*t) in the coefficients. The signed shift is psd_shift - scalar_decay.
        l_norm: Selected norm bound of the processed Cartesian L + s*I.
        kernel_approximation_bound: Kernel-integral approximation bound. This is
            the cutoff tail for the ACL arXiv:2312.03916v2 Eq.(7) kernel and
            also includes the approximate-identity error for Low-Somma
            arXiv:2508.19238v2. It bounds the operator
            norm of the propagator error for a unit input vector, before the
            PSD growth factor. The fact ``kernel_approximation`` in
            ``Plan.facts``, which a classical Result also carries in
            ``Result.facts``, is the physical L2 bound instead, this field
            times the input norm, the Duhamel weight and
            ``exp(psd_shift * t)`` of each operator application, summed over
            the applications. It is not total error. It is exactly 0 when the
            imaginary-time filter or the commuting product of
            `"qsp_block_encoding"` is selected, whose finite sum contains no
            kernel integral.
        quadrature_bound: Available k-quadrature component bound, in the same
            unit-input operator norm. The fact ``k_quadrature``, reported
            the same way, is its physical L2 bound, weighted and summed as
            above. Not total error. It is exactly 0 on the imaginary-time
            filter and the commuting product, which use no k quadrature.
        source_nodes: Duhamel time nodes.
        source_weights: Matching Duhamel time-quadrature weights.
        step_counts: Product-formula step counts, where applicable.
        classical_work: Counts of classical work, not CPU timing.
        pair_rows: The drawn pair rows of a sampled-pair Plan
            (`LCHS.estimator="sampled_pairs"`), saved once at
            planning and never drawn again, or None.
        pair_bias: Steps and deterministic bias bounds of a sampled-pair
            Plan, or None.
        pair_recipe: The product-formula recipe of a sampled-pair Plan with
            `SampledPairs.product_formula` other than ``"plain"``: its selected
            parameters, member law or PTSC laws, bound qualification and
            the per-row choices drawn once at planning
            (`sampled_recipes.LCHSPairMembers` or
            `sampled_recipes.LCHSPairCompensation`), or None.
        pair_qsp: The finite QSP phase bank summary and the per-row bucket
            and member choices of a sampled-pair Plan with
            `ham_evo="qsp_block_encoding"`, drawn once
            at planning (`sampled_qsp.LCHSPairQSP`), or None.
        fixed_shot_precision: For a coherent route chosen under fixed shots,
            its `LCHSFixedShotPrecision` against the route with the smallest
            recovery among those compared, or None when none of them has a
            smaller recovery. Planning compares only what the Method leaves
            open: a named `QSPBlockEncoding.route` is that route alone, and a route under a
            named kernel and quadrature is compared within that one grid, so
            None says nothing about a route or grid that was not compared.
    """

    schema_version: Literal[6] = 6
    elapsed_time: Nonnegative
    dimension: PositiveInt
    encoded_dimension: PositiveInt
    mode: Literal["quadrature", "initial", "zero", "unitary"] = "quadrature"
    has_source: bool = False
    initial_scale: PhysicalScale
    source_scale: PhysicalScale | None = None
    recovery: PhysicalScale | None = None
    success_bits: tuple[Count, ...] = ()
    system_bits: tuple[Count, ...] = ()
    terms: tuple[tuple[Text, Real], ...] = ()
    settings: tuple[ReadoutSetting, ...] = ()
    groups: tuple[tuple[Text, ...], ...] = ()
    grouping_comparisons: Count = 0
    select_work: Count | None = 0
    coefficient_l1_norm: Nonnegative = 0.
    physical_branches: Count = 0
    padded_branches: PositiveInt = 1
    selected_backend: Text
    selected_select: Text
    psd_premise: Literal["numerical", "analytic_periodic", "certified_bound", "user_assertion",
                         "unavailable"] = "unavailable"
    psd_shift: Real = 0.
    scalar_decay: Nonnegative = 0.
    l_norm: Nonnegative = 0.
    kernel_approximation_bound: Nonnegative | None = None
    quadrature_bound: Nonnegative | None = None
    source_nodes: tuple[Real, ...] = ()
    source_weights: tuple[Real, ...] = ()
    step_counts: tuple[Count, ...] = ()
    classical_work: tuple[tuple[Text, Count], ...] = ()
    pair_rows: LCHSSampledRows | None = None
    pair_bias: LCHSPairBias | None = None
    pair_recipe: Annotated[LCHSPairMembers | LCHSPairCompensation, Field(discriminator="recipe")] | None = None
    pair_qsp: LCHSPairQSP | None = None
    fixed_shot_precision: LCHSFixedShotPrecision | None = None

    def _premise_line(self):
        """Return the summary line of the PSD premise, which names an asserted premise as conditional."""
        qualification = f"; {ASSERTED_L_BOUNDS}" if self.psd_premise == "user_assertion" else ""
        return f"PSD premise: {self.psd_premise}; shift={self.psd_shift:.8g}{qualification}"

    def _fixed_shot_line(self):
        """Return the rounded sentence that reports `fixed_shot_precision`, built from the record's fields."""
        precision = self.fixed_shot_precision
        known_cx = precision.ranking == "cx_per_circuit"
        # An accuracy request takes no success-conditional output, so only a moment measured with every shot can offer it.
        can_request = precision.readout == "unconditional_counts"
        if "commuting_product" in precision.smallest_row:
            # QSPBlockEncoding.route has no parameter for the H factor, so no Method names this row. Without known
            # CX counts an accuracy request ranks the routes of one alternative by recovery first
            # (coherent_selection._choose).
            restricted = "Naming commuting_product in QSPBlockEncoding.route restricts the choice to such routes"
            remedy = ("To rank routes by shots times CX, make an accuracy request, which need not choose that route."
                      if known_cx and can_request else
                      f"{restricted}, and an accuracy request then ranks them by recovery factor first."
                      if can_request else f"{restricted}.")
            action = (f"The H factor of that route has no parameter in QSPBlockEncoding.route, so the route cannot be "
                      f"named. {remedy}")
        else:
            naming = "name a route in QSPBlockEncoding.route" + (" and its pair in lchs_kernel and k_quadrature"
                                                                 if " on grid " in precision.smallest_row else "")
            action = (f"To rank routes by shots times CX, make an accuracy request, or {naming}."
                      if known_cx and can_request else f"To change the route, {naming}.")
        comparison = (f"recovery factor {_rounded_scale(self.recovery)} against "
                      f"{_rounded_scale(precision.smallest_recovery)} of route {precision.smallest_row}")
        if precision.readout == "unconditional_counts":
            consequence = f"Its {comparison} widens the sampling radius by {_rounded_scale(precision.factor)} at equal shots."
        else:
            kept = compose_recovery((1.0, precision.factor.mantissa),
                                    PhysicalScale(mantissa=0.5, exponent=1 - precision.factor.exponent))
            consequence = (f"At equal squared norm of the circuit's approximate vector, its {comparison} keeps "
                           f"{_rounded_scale(kept)} of the accepted shots, and different kernels or filters can change "
                           "that norm.")
        return (f"Fixed shots: route {precision.selected_row} was chosen {FIXED_SHOT_RULES[precision.ranking]}, which leaves "
                f"out the recovery factor. {consequence} This is sampling about the finite circuit's moment, not its error against the exact "
                f"solution. {action}")

    def _plan_summary_lines(self):
        """Return the lines that `print(plan)` adds: a conditional PSD premise and the fixed-shot consequence."""
        lines = [self._premise_line()] if self.psd_premise == "user_assertion" else []
        return (*lines, *(() if self.fixed_shot_precision is None else (self._fixed_shot_line(),)))


class LCHSSamples(Record):
    """Sampled output of an LCHS run: the observed original coordinates and their counts, as two int64 arrays.

    `LCHSAnalysis.samples` holds it for a `Samples` output. For m indices,
    analysis copies the two int64 arrays that the sample reduction returns
    into immutable storage one at a time, holding at most 24m bytes of data
    while it does. The two stored arrays occupy 16m bytes of data. These
    counts exclude input data held by the caller, earlier reduction
    workspace and Python object overhead. Building the record does not
    check these bytes against a memory limit.

    Attributes:
        indices: Distinct original-coordinate indices below the original
            dimension, increasing. Padding coordinates are excluded.
        counts: Positive returned counts of those indices, each at most
            ``MAX_COUNT``.
    """

    indices: FrozenArray
    counts: FrozenArray

    @property
    def total(self):
        """Sum of the counts as a Python integer."""
        return int(self.counts.array.sum())


class LCHSProjectedMoments(Record):
    """Saved statistics of the exact scalar readout of an LCHS run (`shots=None`): the three masses and the projected moment.

    `LCHSAnalysis.reduction` holds it.
    Each scalar is a canonical ``(mantissa, exponent)`` pair: zero is
    ``(0.0, 0)``, a nonzero mantissa has absolute value in [1/2, 1).

    Attributes:
        kernel: Versioned name of the computation of the masses and the
            moment, such as `"pow2-mass-pauli-moment/1"`.
        complete_mass: Computed sum of squared amplitude magnitudes over the
            full saved state, including all ``populations[0]`` amplitudes.
        success_mass: Success-only mass ``p_alg``, computed as the sum of squared
            magnitudes of the ``populations[1]`` amplitudes whose success
            qubits take their success values. This selection can include
            branches that fail the additional conditions and coordinates added
            by padding.
        physical_mass: Physical-slice mass p over the ``populations[2]``
            success-and-physical coordinates.
        numerator: Projected moment ``q = v^dagger O v`` of the saved Pauli
            observable on the physical slice v, not a moment of the whole
            success-projected block. Zero for a NormSquared output.
        numerator_radius: Upper bound, rounded upward, on the absolute rounding
            error in ``numerator`` from its classical evaluation using the saved
            observable and the computed physical slice. It assumes binary64
            round-to-nearest arithmetic with gradual underflow and finite
            intermediate results. Errors in the prepared state and in converting
            the original observable to the saved one are separate.
        lost_state_components: Number of initially nonzero real or imaginary
            parts of physical-slice amplitudes that round to zero when rescaled
            by a power of two for the moment calculation. Real and imaginary
            parts are counted separately. Default ``0``.
        lost_coefficients: Number of initially nonzero coefficients of
            non-identity Pauli terms that round to zero when rescaled by a power
            of two for the moment calculation. Default ``0``.
        populations: Numbers of amplitudes in the full saved state, the
            success-only selection and the physical slice, in that order. The
            last is the original dimension. These counts include zero amplitudes.
        contribution_id: Content hash of the `ObservationChunk` that holds
            the saved reduction.
    """

    kernel: Text
    complete_mass: tuple[Real, int]
    success_mass: tuple[Real, int]
    physical_mass: tuple[Real, int]
    numerator: tuple[Real, int]
    numerator_radius: tuple[Real, int]
    lost_state_components: Count = 0
    lost_coefficients: Count = 0
    populations: tuple[Count, Count, Count]
    contribution_id: ContentID


class LCHSGroupMoments(Record):
    """Shot counts and weighted moments of one sampled measurement setting of an LCHS run.

    `LCHSAnalysis.groups` holds one per setting.

    Attributes:
        name: Setting name.
        labels: Member Pauli labels of a group, in first-fit order. Empty for
            a mass setting.
        basis: Accumulated measurement basis, qubit zero rightmost.
        population: ``success_conditional`` (weighted values over the
            shots postselected on success), ``unconditional`` (success indicator
            times weighted values over all returned shots, the identity term
            assigned to the first group) or ``physical_prefix`` (the
            unrotated mass setting).
        returned_shots: Shots returned for the setting.
        selected_shots: Returned shots whose success qubits take their success
            values (and, for the mass setting, whose coordinate is physical).
        mean: Weighted outcome mean, or the postselected fraction of the mass
            setting. None when there are no shots to average, that is, no
            postselected shots for ``success_conditional`` and no returned
            shots otherwise.
        second_moment: Weighted second moment, for a group.
        variance: Unbiased variance of the mean, for a group of at least two shots.
        contribution_id: Content hash of the `ObservationChunk` that holds
            the setting's counts.
    """

    name: Text
    labels: tuple[Text, ...] = ()
    basis: Text
    population: Literal["success_conditional", "unconditional", "physical_prefix"]
    returned_shots: Count
    selected_shots: Count
    mean: Real | None = None
    second_moment: Real | None = None
    variance: Real | None = None
    contribution_id: ContentID


class LCHSPairEstimate(Record):
    """Raw estimates and covering intervals of a sampled-pair LCHS run, in the physical frame.

    `LCHSAnalysis.pair_estimate` holds it. Q is the physical quadratic form
    u(T)^dagger O u(T) and N the physical norm squared ||u(T)||**2. The raw
    estimates are the equal-weight means of the row means of the signed
    pair scores, so N_hat can be negative. With probability at least
    1 - (sum of the failure shares), every requested quantity lies within
    its radius of its raw estimate, the radius being the Hoeffding radius of
    the B row means plus the deterministic bias bound
    ([derivation](../../algorithms/lchs_derivations.md#sampled-branch-pairs)).
    These radii are
    binary64 evaluations of exact-real bounds. No enclosure of the rounding
    of the scores, their sums or the radii themselves is established, and
    the reported endpoints are not outward rounded.

    Attributes:
        quadratic_form: Raw Q_hat, or None without an observable or when
            not representable in binary64.
        norm_squared: Raw signed N_hat, or None when not representable.
        sampling_radius_q, sampling_radius_n: Hoeffding radii of the
            requested quantities, or None when not requested.
        bias_q, bias_n: Deterministic bias bounds, or None when unknown or
            not requested.
        radius_q, radius_n: Sampling radius plus bias, or None when either
            is unavailable.
        failure_q, failure_n: Failure probability assigned to each requested
            quantity, an equal share of the accuracy request's `1 - confidence`
            or, without that request, of `SampledPairs.fail_prob`.
        ratio_interval: Interval of Q/N over the jointly covered rectangle,
            given only when N_hat - radius_n > 0, else None.
        norm_interval: The covering interval of N intersected with N >= 0,
            which leaves the raw N_hat unchanged, or None.
        qualification: Scope and numerical status of these values.
    """

    quadratic_form: Real | None = None
    norm_squared: Real | None = None
    sampling_radius_q: Nonnegative | None = None
    sampling_radius_n: Nonnegative | None = None
    bias_q: Nonnegative | None = None
    bias_n: Nonnegative | None = None
    radius_q: Nonnegative | None = None
    radius_n: Nonnegative | None = None
    failure_q: Nonnegative | None = None
    failure_n: Nonnegative | None = None
    ratio_interval: tuple[Real, Real] | None = None
    norm_interval: tuple[Nonnegative, Nonnegative] | None = None
    qualification: Text


class LCHSPairProgress(Record):
    """Raw sums of the measured rows of a sampled-pair LCHS Run, which may still lack rows.

    `sampled_pair_progress` returns it from a Run's saved data and changes
    nothing. It is not a Result and holds no estimate, confidence radius,
    ratio or normalized value: those need the complete predeclared rows,
    and rows lost to failed or outcome-dependent acquisitions are not a
    random sample of the rows. Scores are in the scaled frame of the row
    table, so a physical sum of scores is the scaled sum times g**2 and a
    sum of squares or products the scaled one times g**4, for the weight
    frame g = ``moments.scale``. Once a continued Run has measured every
    row, ``moments`` equals the moments of the complete analysis.

    Attributes:
        plan_id: Content hash of the Run's Plan.
        run_id: Identity of the Run.
        clusters: Rows B of the Plan.
        repeats: Shots m of every row.
        completed_rows: Sorted indices of the rows whose m shots all
            returned.
        missing_rows: Sorted indices of the other rows, which a
            continuation acquires.
        row_sums: One row per completed row, in the order of
            ``completed_rows``: the sums over its m shots of Z_Q, Z_N,
            Z_Q**2, Z_N**2 and Z_Q*Z_N.
        moments: Signed sums and joint moments of the completed rows,
            `sampled_moments.LCHSPairMoments`.
        returned_shots: Shots returned in the Run, counted over every
            measured row.
        status: ``"complete"`` when no row is missing, else ``"incomplete"``.
    """

    plan_id: ContentID
    run_id: Text
    clusters: PositiveInt
    repeats: PositiveInt
    completed_rows: FrozenArray
    missing_rows: FrozenArray
    row_sums: FrozenArray
    moments: LCHSPairMoments
    returned_shots: Count
    status: Literal["complete", "incomplete"]


class LCHSAnalysis(Result):
    """Solution or requested output of `du/dt = -A u + b` from an `LCHS` run.

    [`solve`][nwqlib.scientist.solve] returns it for an `LCHS` method, and
    `load_result` reopens a saved one. For the default `Solution` output the
    answer is `solution`, the physical solution u(T) in the original
    coordinates, with its norm and phase, in the problem's `unit`. A
    `StateVector` output is in `state_vector`. `NormSquared`,
    `QuadraticForm` and `NormalizedExpectation` give the scalar `value`, and
    `Samples` gives `samples`. Counts do not recover a complex vector. When
    the output cannot be recovered, `unavailable` gives the reason, for
    example a normalized output at zero physical mass. `print(result)`
    shows the value with the construction tolerance and the kernel bounds
    and, where they apply, a nonzero Pauli pruning bound and the phase
    residuals and target of a shared QSP route, each labeled as not a total
    error bound. `result.analyze()` takes no
    settings. The fields below are read-only. The fields of
    [`Result`][nwqlib.core.analysis.Result] are present too.

    The kernel-integral and k-quadrature bounds are facts of the Plan
    (`result.plan.facts`), and a classical Result also carries them in
    `result.facts`. They are physical L2 bounds of error components, not a
    bound on the total error of the output. No reference solution is
    computed unless `result.verify` is called.

    Attributes:
        value: Requested scalar of a `NormSquared`, `QuadraticForm` or
            `NormalizedExpectation` output, physical or unit-normalized as
            that output defines it. `None` for vector and `Samples` outputs
            and when unavailable. With
            `LCHS(estimator="sampled_pairs")` it is the raw
            sampled estimate: a NormSquared value can be negative, and a
            NormalizedExpectation value is given only when the covering
            interval of the norm squared is strictly positive
            (`pair_estimate`).
        unavailable: Reason the requested output cannot be recovered, or
            `None`.
        norm_squared: Recovered physical norm squared `||u(T)||**2`, when
            available. `None` for sampled pairs, whose signed raw estimate
            is in `pair_estimate.norm_squared`.
        physical_scale: Physical scale of the obtained vector, when
            representable.
        numerator: Observable value, the normalized ratio for a
            `NormalizedExpectation` and the recovered physical quadratic form
            for a `QuadraticForm`.
        numerator_frame: Normalization of `numerator`, `"unit"` for the
            normalized ratio or `"physical"` for the quadratic form.
        artifact: Description of the saved solution or state array.
            Printing the result does not load the array, and `solution` and
            `state_vector` read it.
        applications: Numerical operator applications of the run, each with
            its error-component facts.
        submitted_shots: Shots requested from the backend for a finite-shot
            `Samples` or coherent `NormSquared` output, and over all rows of
            sampled pairs. `None` for exact readout, when nothing is measured
            (a `NormSquared` with no evolution takes its value from the input)
            and for other outputs. For a coherent `QuadraticForm` or
            `NormalizedExpectation`, `groups` gives the returned and selected
            shots of each setting, and `result.plan.shots` is the request for
            each setting.
        returned_shots: Shots the backend returned for the same outputs,
            before postselection on the physical coordinates. It is at most
            `submitted_shots`, and smaller when the backend returned only part
            of the request. `None` for the same outputs as `submitted_shots`.
        selected_shots: Returned shots kept for the physical output: for
            `Samples` the sum of the sample counts, for a coherent
            `NormSquared` the shots whose success qubits read zero and whose
            coordinate lies inside the problem's dimension, and for sampled
            pairs the shots of the completed rows. `None` for the same outputs
            as `submitted_shots`.
        samples: For a `Samples` output, the distinct original-coordinate
            indices (`samples.indices`, increasing, padding coordinates
            excluded) and their positive counts (`samples.counts`), both
            int64 arrays. They do not cover all submitted shots.
        reduction: Saved statistics of the exact scalar readout
            (`shots=None`): the complete, success and physical-slice masses
            and the projected moment, as mantissa-exponent pairs.
        groups: Weighted moments of each sampled measurement setting, with
            its Pauli labels, basis, returned shots and postselected shots.
        pair_moments: For sampled pairs, the signed sums and joint moments
            of the pair scores over the completed rows, in the binary frame
            of the row table.
        pair_estimate: For sampled pairs, the raw physical estimates, their
            radii, failure shares and the normalized interval.
    """

    value: Real | None = None
    unavailable: Text | None = None
    norm_squared: Nonnegative | None = None
    physical_scale: PhysicalScale | None = None
    numerator: Real | None = None
    numerator_frame: Literal["physical", "unit"] | None = None
    artifact: ArtifactManifest | None = None
    applications: tuple[KernelApplication, ...] = ()
    submitted_shots: Count | None = None
    returned_shots: Count | None = None
    selected_shots: Count | None = None
    samples: LCHSSamples | None = None
    reduction: LCHSProjectedMoments | None = None
    groups: tuple[LCHSGroupMoments, ...] = ()
    pair_moments: LCHSPairMoments | None = None
    pair_estimate: LCHSPairEstimate | None = None

    def _summary_lines(self):
        """Describe the Result from saved records only, without loading arrays.

        Every error-related line states its scope. The construction tolerance,
        the kernel bounds, the pruning bound and the phase residuals of a
        shared route are labeled as not total physical-output error, and the
        layered MPS circuit error is reported as not evaluated.
        """
        plan = self._plan
        kind = None if plan is None else plan.output.kind
        if self.artifact is not None:
            label = "Array output" if kind is None else "Physical solution" if kind == "solution" else "State vector"
            lines = [label + ": " + self._array_text(self.artifact)]
            if acquisition := self._array_acquisition_text(self.artifact):
                lines.append(acquisition)
        elif kind in {"solution", "state_vector"}:
            label = "Physical solution" if kind == "solution" else "State vector"
            frame = "physical" if kind == "solution" else plan.output.normalization
            phase = "physical" if kind == "solution" else plan.output.global_phase
            description = f"shape=({plan.problem.dimension},), {frame}, phase={phase}"
            if unit := self._unit_text():
                description += "; unit=" + unit.strip()
            state = plan.problem.initial_state
            array = None
            if (self.unavailable is None and plan.reconstruction.mode in {"initial", "zero"}
                    and self._data is not None and not self._data.observations.chunks
                    and state.reference.representation == "vector"):
                # Ingested vectors alone own resident physical/unit arrays.
                # Positive-norm normalization preserves their physical phase.
                array = state._direction if frame == "unit" else state._physical
            text = (description + "; no acquired array" if array is None else
                    self._resident_array_text(array, description) + "; supplied initial vector")
            lines = [label + ": " + text]
        elif kind == "samples":
            stored = 0 if self.samples is None else len(self.samples.indices.array)
            lines = [f"Samples: {stored} distinct indices; selected shots={self.selected_shots}; "
                     f"returned shots={self.returned_shots}"]
        else:
            lines = [f"{kind or 'Scalar value'}: {self._scalar_text(self.value)}{self._unit_text()}"]
        if self.unavailable:
            lines.append(self.unavailable)
        if plan is not None:
            rec = plan.reconstruction
            lines.append(f"LCHS mode: {rec.mode}; elapsed time={self._scalar_text(rec.elapsed_time)}")
            if rec.mode in {"quadrature", "unitary"}:
                lines.append(f"Realization: {rec.selected_backend}; SELECT={rec.selected_select}")
                if rec.selected_backend == "dense_exact":
                    lines.append("Dense realization uses classically exponentiated branches")
                lines.append(f"Construction tolerance: {self._scalar_text(plan.method.approx_tol)}; not a total error bound")
                lines.append(f"Selected kernel bounds per unit input before PSD growth: "
                             f"approximation={self._scalar_text(rec.kernel_approximation_bound)}; "
                             f"quadrature={self._scalar_text(rec.quadrature_bound)}; not total physical-output error")
                lines.append(rec._premise_line())
                if rec.fixed_shot_precision is not None:
                    lines.append(rec._fixed_shot_line())
                # The Plan holds this physical-solution value in both executions. A quantum Result's facts are
                # in the frame of its output, so they do not repeat it. Zero when nothing was dropped, the usual
                # case, which needs no line.
                pruning = next((item.fact.value.value for item in plan.facts
                                if item.fact.quantity == "trotter_pruning" and item.fact.availability == "concrete"), 0.)
                if pruning:
                    lines.append(f"Pauli pruning bound: {self._scalar_text(pruning)} in the physical solution; "
                                 "a part of the unknown product-formula error, not total physical-output error")
                # Only a shared route has one target per parity and these two solutions. Both are saved with the Plan.
                prepared = plan._native["native_data"].qsp_plan if rec.selected_select == "coherent_lcu" else None
                if prepared is not None and prepared.candidate.route.startswith("shared"):
                    lines.append(f"Phase synthesis ({prepared.candidate.route}): residuals "
                                 f"cos={prepared.phases.cos_solution.residual_sup_bound:.3g}, "
                                 f"sin={prepared.phases.sin_solution.residual_sup_bound:.3g} against target "
                                 f"{prepared.candidate.allowance['synthesis_per_parity']:.3g} per parity for the "
                                 "normalized polynomials; nominal numerical estimates, not total physical-output error")
                if rec.has_source:
                    lines.append(f"Source quadrature: {len(rec.source_nodes)} nodes; {rec.physical_branches} physical branches")
                if rec.pair_rows is not None and self.pair_estimate is not None:
                    estimate = self.pair_estimate
                    lines.append(f"Sampled pairs: {rec.pair_rows.clusters} rows x {rec.pair_rows.repeats} shots; "
                                 f"raw N_hat={self._scalar_text(estimate.norm_squared)} "
                                 f"(radius {self._scalar_text(estimate.radius_n)}); "
                                 f"raw Q_hat={self._scalar_text(estimate.quadratic_form)} "
                                 f"(radius {self._scalar_text(estimate.radius_q)}); radii are Hoeffding plus bias "
                                 "for the complete row table")
                if rec.classical_work:
                    work = dict(rec.classical_work)
                    lines.append(f"Classical work proxy: {work['size_units']}; vector rotations={work.get('vector_rotations',0)}; matrix-power products={work.get('matrix_power_products',0)}")
                for block in plan.construction.selections:
                    if block.choice == 'mps_circuit':
                        parameters = {item.parameter:item.value for item in block.cost_parameters}
                        lines.append(f"{block.signature.name}: MPS layers={parameters['mps_layers']}, max stored bond={parameters['mps_max_stored_bond']}; core bytes={parameters['mps_core_bytes']}; construction work={block.construction_work}; SDK workspace unknown")
                        lines.append("Layered MPS circuit error: not evaluated; TT-SVD discarded weight is a separate compression quantity")
        return lines

    @property
    def solution(self):
        """The physical solution u(T) of a `Solution` output, a complex array.

        It is in the original coordinates and keeps the physical norm and
        phase. With no evolution (`time` equal to `initial_time`, or a zero
        input) it is the supplied initial vector.

        Raises:
            AttributeError: If the requested output was not `Solution`.
            ValueError: If the array is unavailable, with the reason.
        """
        if self.plan.output.kind != "solution":
            raise AttributeError("solution was not the selected output")
        if self.artifact is None:
            if self.plan.reconstruction.mode in {"initial","zero"} and not self.data.observations.chunks:
                return self.plan.problem.initial_state.physical_vector()
            raise ValueError(self.unavailable or "solution array is unavailable")
        return self.data.artifact(self.artifact).array

    @property
    def state_vector(self):
        """The state vector of a `StateVector` output, a complex array.

        It is in the original coordinates, normalized and phased as the
        `StateVector` output requests.

        Raises:
            AttributeError: If the requested output was not `StateVector`.
            ValueError: If the array is unavailable, with the reason.
        """
        if self.plan.output.kind != "state_vector":
            raise AttributeError("state_vector was not the selected output")
        if self.artifact is None:
            if self.unavailable is None and self.plan.reconstruction.mode in {"initial","zero"} and not self.data.observations.chunks:
                state = self.plan.problem.initial_state
                return state._direction if self.plan.output.normalization=="unit" else state.physical_vector()
            raise ValueError(self.unavailable or "state array is unavailable")
        return self.data.artifact(self.artifact).array

    def validate_plan(self, plan):
        """Reject a Result that does not match its Plan's construction, output kind or coordinates."""
        super().validate_plan(plan)
        if plan.method.descriptor.method != "lchs" or self.construction_id != plan.construction.content_id:
            raise ValueError("LCHS result requires its actual selected construction")
        output = plan.output.kind
        if output in {"solution", "state_vector", "samples"} and self.value is not None:
            raise ValueError("array/sample output cannot carry an LCHS scalar value")
        sampled = plan.reconstruction.pair_rows is not None
        if sampled and (self.pair_moments is None or self.pair_moments.table_id != plan.reconstruction.pair_rows.content_id
                        or self.norm_squared is not None):
            raise ValueError("sampled-pair LCHS result requires the moments of its Plan's row table")
        if output == "norm_squared" and not sampled and self.value != self.norm_squared:
            raise ValueError("LCHS norm output differs from its observed norm squared")
        if output == "normalized_expectation" and self.norm_squared == 0 and self.value is not None:
            raise ValueError("normalized expectation is undefined for the zero physical vector")
        if self.artifact is not None and (
            self.artifact.plan_id != self.plan_id or self.artifact.construction_id != self.construction_id
            or self.artifact.output.basis != plan.problem.basis
        ):
            raise ValueError("LCHS array must preserve its actual acquisition and original coordinates")
