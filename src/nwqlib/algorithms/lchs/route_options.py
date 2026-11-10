"""Options of the LCHS routes, one record per route that has options.

`LCHS` chooses a route with each of four fields: `ham_evo`,
`init_prep`, `lcu_prep` and
`estimator`. Each field takes the route's name or the record below
that carries the route's options, for example
`ham_evo=Trotter(steps=4)`. A name stands for the record
with its default options, and building the Method replaces it by that record,
so both spellings give the same Method and the same content hash. A route
without options (`"auto"`, `"direct"`, `"analytic"`, and the `"plain"` and
`"mpf"` recipes of `SampledPairs.product_formula`) stays a name.
"""

from typing import Annotated, Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator

from nwqlib.core.records import PositiveInt, Real, Record, _ConstructorDisplay
from .provider_config import ProviderConfig


def provider_config(value):
    """Return a registry name, a {"implementation", "parameters"} dict or a ProviderConfig as a ProviderConfig."""
    if isinstance(value, str):
        return ProviderConfig(implementation=value)
    if isinstance(value, dict):
        return ProviderConfig(**value)
    if not isinstance(value, ProviderConfig):
        raise TypeError("LCHS provider choice must be a name or ProviderConfig")
    return value


def _integer_order(value):
    """Require an integer order before the Literal check, so a float 2.0 or a bool fails."""
    from nwqlib._validation import integer
    return integer(value, "order", 1)


class Trotter(_ConstructorDisplay, Record):
    """Options of the `"trotter"` evolution backend: a sorted-Pauli product formula with one step count for all nodes.

    Pass it as `LCHS(ham_evo=Trotter(steps=4))`. The
    name `"trotter"` stands for `Trotter()`. Every argument is optional. With
    the default `Coherent` estimator the backend needs a dense A or a
    `PeriodicStencil` A, and a `PeriodicStencil` A needs `order=2` and
    quantum execution. `SampledPairs` evolves an A given as a Pauli sum or
    by Pauli parts.

    Attributes:
        kind: `"trotter"`, the route's name. It is fixed.
        steps: Default `1`, or `None` when only `synthesis_tol` is
            given. Fixed step count of every node, a positive integer at
            most `max_steps`, or `None` when `synthesis_tol` chooses
            it. Exactly one of the two is set: `Trotter()` runs one step,
            `Trotter(synthesis_tol=eps)` is
            `Trotter(steps=None, synthesis_tol=eps)`, and a step
            count given together with a tolerance is refused. A fixed count leaves the `trotter_synthesis` error of a general
            A unknown, so the total error is unknown. `plan.facts` then lists
            `trotter_pruning`, the physical-solution bound on the Pauli terms
            removed from each node generator, for any requested output: a term
            is removed when its size is at most 1e-12 times the largest
            coefficient of H or k*L at that node, the identity included. This
            is one part of the unknown error, not a bound on the total.
            Printing the Result shows a nonzero value. A classical Result also
            lists it in `result.facts`. A quantum Result does not, because
            its facts are in the frame of the requested output.
        synthesis_tol: Default `None`. Positive tolerance on the
            weighted Strang synthesis bound of a `PeriodicStencil` A, in the
            same unit-input norm as `LCHS.approx_tol`. Planning
            chooses the smallest step count r with ``B/r**2`` at most this
            value, where T is the elapsed time, q is the system-qubit count,
            ``a_j = |k_j| diffusion``, ``p = |potential|`` and
            ``B = T**3 sum_j |c_j| W_q(a_j, p)`` over the quadrature nodes
            and coefficients. The matching-operator commutators give
            ``W_q(a, p) = g a**3/2 + 4 a**2 p/3 + a p**2/3``, with g = 0
            for q = 1, 2 and g = 1 for q >= 3, when the circuit applies the
            potential, even-matching and odd-matching factors in that order
            (Childs et al.,
            doi:10.1103/PhysRevX.11.011020, Proposition 10, Eq. (121),
            or arXiv:1912.08854v3, Proposition 16, Eq. (152)).
            The bound covers the planned finite weighted action on a unit input.
            Input scaling and other error components are treated separately. With
            zero potential the ideal synthesis error is zero at q = 1, 2.
            Given without `steps`, it sets `steps` to `None`. Planning
            refuses it for an A that is not a `PeriodicStencil`.
        order: Default `2`. Order of the product formula, 1 for the Lie
            formula or 2 for the symmetric Suzuki (Strang) formula.
        max_steps: Default `None`, no limit. Optional upper limit on the
            product-formula step count of any node, fixed or chosen by
            `synthesis_tol`. A SELECT stores one angle row per distinct
            repetition block, so the step count does not by itself set the
            size of the planned tables.

    Raises:
        ValueError: If both `steps` and `synthesis_tol` are set,
            if `steps=None` is given without a tolerance, or if `steps`
            exceeds a set `max_steps`.
    """

    kind: Literal["trotter"] = "trotter"
    steps: PositiveInt | None = 1
    synthesis_tol: Annotated[Real, Field(gt=0)] | None = None
    order: Literal[1, 2] = 2
    max_steps: PositiveInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _tolerance_alone(cls, data):
        """A tolerance given without steps chooses the step count, so steps is None rather than its default 1."""
        if isinstance(data, dict) and data.get("synthesis_tol") is not None and "steps" not in data:
            return {**data, "steps": None}
        return data

    @field_validator("order", mode="before")
    @classmethod
    def _order(cls, value):
        return _integer_order(value)

    @model_validator(mode="after")
    def _steps(self):
        """Exactly one of steps and synthesis_tol, so a fixed count and a selected count cannot disagree."""
        from nwqlib._limits import exceeds
        if (self.steps is None) == (self.synthesis_tol is None):
            raise ValueError("set exactly one of steps and synthesis_tol; synthesis_tol alone chooses "
                             "the periodic Strang step count")
        if self.steps is not None and exceeds(self.steps, self.max_steps):
            raise ValueError(f"steps={self.steps} exceeds max_steps={self.max_steps}")
        return self


class TrotterErrorBudgeted(_ConstructorDisplay, Record):
    """Options of the `"trotter_error_budgeted"` evolution backend: a product formula with a step count chosen per node.

    Pass it as `LCHS(ham_evo=TrotterErrorBudgeted(order=1))`.
    The name `"trotter_error_budgeted"` stands for `TrotterErrorBudgeted()`.
    Each node gets the smallest step count whose product-formula bound plus
    the Pauli pruning bound fits 0.1 times `LCHS.approx_tol`
    (Childs et al., doi:10.1103/PhysRevX.11.011020, Propositions 9-10 and
    Sec. V B): Proposition 9, Eq. (120) for order 1 and Proposition 10,
    Eq. (121) for order 2. With the default `Coherent` estimator the backend
    needs a dense A. `SampledPairs` evolves an A given as a Pauli sum or by
    Pauli parts.

    Attributes:
        kind: `"trotter_error_budgeted"`, the route's name. It is fixed.
        order: Default `2`. Order of the product formula, 1 for the Lie
            formula or 2 for the symmetric Suzuki (Strang) formula.
        max_steps: Default `None`, no limit. Optional upper limit on the
            step count that planning chooses for any node, for a sampled-pair
            bucket or for a sampled-pair recipe, which planning refuses when
            it needs more. A SELECT stores one angle row per distinct
            repetition block, so the step count does not by itself set the
            size of the planned tables.
    """

    kind: Literal["trotter_error_budgeted"] = "trotter_error_budgeted"
    order: Literal[1, 2] = 2
    max_steps: PositiveInt | None = None

    @field_validator("order", mode="before")
    @classmethod
    def _order(cls, value):
        return _integer_order(value)


class DenseExact(_ConstructorDisplay, Record):
    """Options of the `"dense_exact"` evolution backend: classically computed branch matrices in one dense SELECT.

    Pass it as `LCHS(ham_evo=DenseExact(control_route="gatewise"))`.
    The name `"dense_exact"` stands for `DenseExact()`. The backend computes
    each branch matrix `exp(-i t (k_j L + H))` from the node's Hermitian
    eigensystem and controls its circuit on the branch's address. It needs a
    dense A. `ham_evo="auto"` uses `DenseExact()` for a
    dense A, so changing these options needs the backend written out.

    Attributes:
        kind: `"dense_exact"`, the route's name. It is fixed.
        control_route: Default `"auto"`. How the SELECT controls the dense
            unitary of each branch, on two or more qubits, on its address
            bits. `"gatewise"` synthesizes the unitary and lets Qiskit
            control each synthesized gate, `"whole_matrix"` synthesizes the
            controlled matrix, and `"auto"` takes the whole-matrix route for
            one control and the gate-wise route for more. On the
            whole-matrix route a constant-source branch is one unitary, its
            evolution times the matrix of its input preparation.
        max_select_slots: Default `4_096`. Upper limit on the number of
            address slots of the dense SELECT circuit, the branch count
            rounded up to a power of two. Planning refuses a larger
            construction and never changes the grid, Duhamel nodes or
            tolerance to fit. A composite Gauss grid that alone exceeds the
            limit is refused before its nodes and weights are computed. The
            limit does not apply to classical execution.
    """

    kind: Literal["dense_exact"] = "dense_exact"
    control_route: Literal["gatewise", "whole_matrix", "auto"] = "auto"
    max_select_slots: PositiveInt = 4_096


class QSPBlockEncoding(_ConstructorDisplay, Record):
    """Options of the `"qsp_block_encoding"` evolution backend: quantum signal processing on block encodings of L and H.

    Pass it as `LCHS(ham_evo=QSPBlockEncoding(route="laurent"))`.
    The name `"qsp_block_encoding"` stands for `QSPBlockEncoding()`. The
    backend evolves under the joint generator with quantum signal processing
    on block encodings of L and H (Pocrnic et al., arXiv:2506.20760v2,
    Section IV), never forms a matrix of the system dimension and needs
    quantum execution. It controls each dense-dilation child gate-wise.
    `ham_evo="auto"` uses `QSPBlockEncoding()` for an
    A given as a Pauli sum, Cartesian parts, a Hermitian band operator or a
    supplied block encoding.

    Attributes:
        kind: `"qsp_block_encoding"`, the route's name. It is fixed.
        route: Default `"auto"`. Polynomial transformation, given as a name,
            a dict `{"implementation": name, "parameters": {...}}` or a
            [`ProviderConfig`][nwqlib.algorithms.lchs.provider_config.ProviderConfig].
            `"auto"` compares the routes below by their CX
            count before any phase is synthesized, then synthesizes phases for the chosen
            route only: shared real QSVT without (`"shared_raw"`) or with
            three-use oblivious amplitude amplification (`"shared_oaa"`) at
            common cap 0.999, 0.99, 0.9375 or 0.5; the two-sided (`"laurent"`)
            and one-sided (`"gqet"`) Jacobi–Anger polynomials at margin 1e-3,
            1e-2 or 1/16; and, when H is a multiple of the identity, the
            imaginary-time filter `"imaginary_gqet"`, a modified-Bessel
            polynomial of L that needs no k quadrature, at the same margins;
            and, when L and H are both periodic band operators on one grid,
            which commute exactly, `"commuting_product"`, that filter of L
            times one of the real-time routes above for H, each with half of
            the allowance in the sense `2 b + b**2 = B`, also without a k
            quadrature.
            For `Samples` the CX count of each choice is weighted by the
            square of its recovery factor zeta gamma, the factor that
            multiplies the measured success amplitudes, since accepted shots
            occur with probability proportional to its inverse square; other
            outputs compare the CX count of one circuit. Between the kernel
            sum and the filter, which approximate the solution differently,
            this comparison is nominal: it does not prove which needs less
            work for the same total error. Only known costs are ranked.
            When every eligible cost of the kernel sum, the filter or the
            commuting product is unknown, the choice among its routes uses
            query-call count, then the total degree of the phase polynomials,
            then recovery. `Samples` and `accuracy` requests prioritize recovery,
            then query-call count, then phase degree. Between them, with no
            known cost, the choice takes the kernel sum, so the chosen route
            can have more query calls and a larger recovery factor than
            another compared route. Exactly tied costs go
            to the lower phase degree. Unknown CX counts remain unknown, and
            this choice establishes no CX minimum. A named route restricts
            the choice to it. `"commuting_product"` without that structural
            commutation, `"berry"`, `"outer_amplitude_amplification"`,
            `"martyn_one_shot"` and `"coherent_mpf"` are refused at planning
            with the reason. The
            parameters are `level`, the cap or margin of a named route;
            `node`, the encoding of the kernel nodes (`"rotation_table"`,
            `"comparator"`, `"twos_complement_pauli"` or
            `"signed_magnitude_pauli"` for a uniform grid, `"table"`
            otherwise); `joint`, the generator form (`"separate_parts"`,
            `"single_a"` or the filter's `"convex_contraction"`). A phase synthesis
            doubles its transform length until its actual residual meets the
            tolerance that the route's allowance leaves after the truncation
            tail, and a failed synthesis raises an error without trying
            another route. A request that
            no route meets raises ApplicabilityError before any phase is
            synthesized. With `estimator=SampledPairs(...)` the
            branches run the phase bank of `SampledPairs.qsp_bank` instead,
            and `route` must stay `"auto"`.

    Raises:
        ValueError: If `route` names an unknown route or parameter, or a
            level that its route does not have.
        TypeError: If `route` is not a name, a dict or a `ProviderConfig`.
    """

    kind: Literal["qsp_block_encoding"] = "qsp_block_encoding"
    route: Any = Field(default_factory=lambda: ProviderConfig(implementation="auto"))

    @field_validator("route", mode="before")
    @classmethod
    def _route(cls, value):
        return provider_config(value)

    @field_serializer("route")
    def _route_record(self, value):
        """Serialize the route request as its explicit implementation and parameters."""
        return value.to_dict()

    @model_validator(mode="after")
    def _route_choice(self):
        """Reject a route that names an unknown route or parameter, or a level that its route does not have.

        The route names and their finite levels come from
        ``coherent_selection`` (`SHARED_TARGET_CAPS`, `GENERAL_MARGINS` and
        the refused routes of `NOT_ADMITTED_ROUTES`). A refused or
        inapplicable route is accepted here and refused at planning with its
        reason, before any phase is synthesized.
        """
        route, parameters = self.route.implementation, dict(self.route.parameters)
        if route == "auto" and not parameters:
            return self
        from .coherent_selection import GENERAL_MARGINS, NOT_ADMITTED_ROUTES, SHARED_TARGET_CAPS
        levels = dict(shared_oaa=SHARED_TARGET_CAPS, shared_raw=SHARED_TARGET_CAPS, laurent=GENERAL_MARGINS,
                      gqet=GENERAL_MARGINS, imaginary_gqet=GENERAL_MARGINS, commuting_product=(),
                      **dict.fromkeys(NOT_ADMITTED_ROUTES, ()))
        if route != "auto" and route not in levels:
            raise ValueError(f"route implementation {route!r} is not 'auto' or one of {sorted(levels)}")
        unknown = sorted(set(parameters) - {"level", "node", "joint"})
        if unknown:
            raise ValueError(f"route takes the parameters level, node and joint, not {unknown}")
        if "level" in parameters and (route == "auto" or parameters["level"] not in levels[route]):
            raise ValueError(f"route level {parameters['level']!r} needs a named route with that level; "
                             f"{route!r} has {levels.get(route, ())}")
        if parameters.get("joint", "separate_parts") not in ("separate_parts", "single_a", "convex_contraction"):
            raise ValueError("route joint must be 'separate_parts', 'single_a' or 'convex_contraction'")
        if parameters.get("node", "table") not in ("rotation_table", "comparator", "twos_complement_pauli",
                                                    "signed_magnitude_pauli", "table"):
            raise ValueError("route node must be 'rotation_table', 'comparator', 'twos_complement_pauli', "
                             "'signed_magnitude_pauli' or 'table'")
        return self


class MPSCircuit(_ConstructorDisplay, Record):
    """Options of the `"mps_circuit"` state preparation: a layered circuit from a matrix-product-state compression.

    Pass it as `LCHS(init_prep=MPSCircuit(num_layers=3))` or
    `LCHS(lcu_prep=MPSCircuit(...))`. The name `"mps_circuit"`
    stands for `MPSCircuit()`. The options apply to the compression of the
    state that the field prepares: the dense initial direction for
    `init_prep`, the coefficient state for
    `lcu_prep`. Planning for quantum execution compresses it once, and the
    TT-SVD and the construction of its layered circuit count against
    `LCHS.max_svd_work`. Classical execution evaluates the finite sum of
    branches directly and prepares no state, so these options do not change a
    classical result, and that result carries no compression error. The same
    Method serves both executions.

    Attributes:
        kind: `"mps_circuit"`, the route's name. It is fixed.
        max_bond_dim: Default `None`, no limit. Upper limit on the kept bond
            dimension of the compression.
        threshold: Default `1e-14`, positive. Singular-value truncation
            threshold of the compression.
        num_layers: Default `2`, positive. Number of layers of the MPS
            circuit.
    """

    kind: Literal["mps_circuit"] = "mps_circuit"
    max_bond_dim: PositiveInt | None = None
    threshold: Annotated[Real, Field(gt=0)] = 1e-14
    num_layers: PositiveInt = 2


class Coherent(_ConstructorDisplay, Record):
    """Options of the default `"coherent"` observable estimator, which prepares the solution coherently and reads the output.

    `LCHS()` uses `Coherent()`, and the name `"coherent"` stands for it. Pass
    `LCHS(estimator=Coherent(fail_prob=.01))` to change
    the failure probability of the reported sampling radius.

    Attributes:
        kind: `"coherent"`, the route's name. It is fixed.
        fail_prob: Default `0.05`, strictly between 0 and 1. Upper bound on
            the total probability that the reported sampling radius of a
            `NormSquared`, `QuadraticForm` or `NormalizedExpectation` output
            measured with `shots` fails, under independent, stationary shots
            and the numerical assumptions stated in the sampling fact. It is
            split equally over the measurement settings that acquire counts.
            The radius is Hoeffding's bound for the independent shots of each
            setting, from the range of that setting's per-shot score, the
            square of the recovery factor times the sum of the group's Pauli
            coefficient magnitudes, summed over the settings. The counts of a
            `NormSquared` output use the bound of an indicator in [0, 1]. A
            normalized expectation without padded coordinates uses the
            conditional means of the shots that pass the success test; with
            padding it is the ratio of the observable and the physical-mass
            setting, which is one of the settings that share the
            probability, and its radius exists only when the mass interval
            is positive. The radius covers the shot noise of the finite
            circuit's moment, not its approximation error, which
            `algorithmic_approximation` reports separately. Exact readout
            (`shots=None`) has no sampling error. A `NormSquared` output at
            zero elapsed time, and a `NormSquared` or `QuadraticForm` output
            of zero inputs, is known from the input without counts, and its
            sampling fact is 0. The exact no-evolution
            readout of a `SampledPairs` Method (zero elapsed time or zero
            inputs), measured with `shots`, reports this radius at
            `SampledPairs.fail_prob`. When an `accuracy` request
            chose the shots (`LCHS.sampling_shots`), the request's
            `1 - confidence` is this probability instead, and a non-default
            value that differs from it is refused.
    """

    kind: Literal["coherent"] = "coherent"
    fail_prob: Annotated[Real, Field(gt=0, lt=1)] = .05


class Richardson(_ConstructorDisplay, Record):
    """The Richardson recipe of `SampledPairs.product_formula`, which cancels the h**2 bias term of the Strang formula.

    Pass it as `SampledPairs(product_formula=Richardson(pf_allowance=.01))`.
    The name `"richardson"` stands for `Richardson()`. Each row draws one of
    the two Strang step counts s and 2s with probabilities near 1/5 and 4/5
    and weights -1/3 and 4/3, which cancels the h**2 bias term (Watson and
    Watkins, arXiv:2408.14385v2, Sec. 5). It needs a backend of `order=2`.
    With `Trotter`, s is `Trotter.steps` and 2s must not exceed a set
    `Trotter.max_steps`. For two or more distinct nonidentity Pauli labels,
    the published bias bound needs s > 2A, with
    A = 4(2L - 1)T_e(lambda_H + K lambda_L). Here H_c and L_c are H and L
    with their identity terms removed. L in the factor 2L - 1 counts the
    distinct labels in their union, lambda_H and lambda_L are the
    coefficient one-norms of H_c and L_c, T_e is the largest elapsed time,
    and K bounds |k| over the entire kernel support. The remainder is
    unknown when s <= 2A. When T_e(lambda_H + K lambda_L) is 0 or at most
    one distinct nonidentity Pauli label occurs, the evolution remainder
    is 0 and s need not exceed 2A. The discrepancy of the finite member
    law still enters the bias bound. With `TrotterErrorBudgeted`, planning
    chooses a sufficient s from `pf_allowance`.

    Attributes:
        kind: `"richardson"`, the recipe's name. It is fixed.
        pf_allowance: Default `None`. Positive allowance on the
            product-formula bias of the norm squared ||u(T)||**2, in the
            solution's unit squared; the quadratic form's allowance is the
            sum of the observable's |Pauli coefficients| times it. Required
            with `TrotterErrorBudgeted`, and refused with `Trotter`, whose
            step count is fixed.
    """

    kind: Literal["richardson"] = "richardson"
    pf_allowance: Annotated[Real, Field(gt=0)] | None = None


class PTSC(_ConstructorDisplay, Record):
    """The paired Taylor-series compensation recipe of `SampledPairs.product_formula`.

    Pass it as `SampledPairs(product_formula=PTSC(pf_allowance=.01,
    mu_cap=1.5, taylor_cutoff=8))`. All three arguments are required, so the
    name `"ptsc"` alone is refused. The recipe adds to every first-order
    (Lie) step a random Pauli correction of the paired Taylor-series
    compensation (Zeng, Sun, Jiang and Zhao, arXiv:2212.04566v2, Sec. IV.2),
    independently for the two branches of a row, and multiplies the row by
    the two branches' normalizations. It needs
    `TrotterErrorBudgeted(order=1)`; planning chooses the common step count
    from `mu_cap` and the cutoff from `pf_allowance`.

    Attributes:
        kind: `"ptsc"`, the recipe's name. It is fixed.
        pf_allowance: Required, positive. Allowance on the product-formula
            bias of the norm squared ||u(T)||**2, in the solution's unit
            squared; the quadratic form's allowance is the sum of the
            observable's |Pauli coefficients| times it.
        mu_cap: Required. Bound mu_cap > 1 on the normalization of each
            branch. The common step count r is the smallest with r > 2 L and
            r**3 log(mu_cap) >= (e + 2/9)(2 L)**4, L =
            T_e(lambda_H + K lambda_L), so a larger mu_cap allows a smaller
            r. The score range, and with it the sampling radius for a fixed
            number of independent rows, grows as mu_cap**2. For a fixed
            failure probability delta of the quantity, a bound W on the
            physical importance weight of a single branch and
            `SampledPairs.repeats` R, the number of independent rows
            sufficient for a sampling radius eps grows as mu_cap**4: an
            integer B of at least 1 with B >= 2 (H/eps)**2 log(2/delta),
            where H = W**2 mu_cap**2 for the norm squared and
            H = H_O W**2 mu_cap**2 for the quadratic form, H_O the largest
            |o_l|/p_l of the observable terms. The shot count B R grows in
            the same way. This sufficient count follows from the
            score range, not from an observed variance, and because mu_cap
            also changes r and the corrections, it gives no ratio of total
            run times.
        taylor_cutoff: Required. Largest Taylor order, at least 3, that the
            corrections may use. Planning takes the smallest order from 3 up
            whose truncation and finite probability-table bound meets
            `pf_allowance`, and refuses when none up to this value does.
    """

    kind: Literal["ptsc"] = "ptsc"
    pf_allowance: Annotated[Real, Field(gt=0)]
    mu_cap: Annotated[Real, Field(gt=1)]
    taylor_cutoff: Annotated[PositiveInt, Field(ge=3)]


class SampledPairs(_ConstructorDisplay, Record):
    """Options of the `"sampled_pairs"` observable estimator, which estimates a scalar output from random branch pairs.

    Pass it as `LCHS(estimator=SampledPairs(repeats=4))`. The name
    `"sampled_pairs"` stands for `SampledPairs()`. The estimator, its inputs
    and its backends are described under `LCHS.estimator`.

    Attributes:
        kind: `"sampled_pairs"`, the route's name. It is fixed.
        repeats: Default `1`. Shots m of each sampled row. The Plan draws
            `shots/m` independent rows, so `shots` must be divisible by m.
            Repeating a row does not narrow the confidence radius, which
            counts rows only.
        fail_prob: Default `0.05`, strictly between 0 and 1. For an evolving
            sampled-pair Plan, total probability that a reported covering
            radius fails, split equally
            over the quantities the output needs (the quadratic form and the
            norm squared of a `NormalizedExpectation`). For the complete
            planned set of rows, the covering radius targets the physical
            quantity and adds both evolution and model bias to the
            Hoeffding radius. Model bias includes the kernel, k quadrature,
            source quadrature and applicable finite random-bit effects. An
            unknown bias leaves the covering radius unavailable while the
            sampling radius remains available. The bounds are numerical
            estimates without a complete enclosure of arithmetic errors.
            At zero elapsed time or zero inputs, no pair rows are drawn. Any
            counts sampling radius then uses this probability split equally
            over its measurement settings, including the physical-mass
            setting for a padded normalized expectation, as described by
            `Coherent.fail_prob`; an output known from the input acquires no
            counts, and its sampling fact is 0.
            An `accuracy` request supplies `1 - confidence` instead, and a
            non-default value that differs from it is refused.
        kernel: Default `"discrete"`, which draws the nodes of the Method's
            kernel-quadrature pair with probability proportional to their
            coefficient magnitudes. `"continuous"` draws k from a truncated
            Cauchy distribution of the kernel itself, without a k
            quadrature. It needs `LCHS.lchs_kernel` `"apel_berntson"` or
            `"low_somma_f2"` (given with one of their allowed quadrature
            rules, which it does not use), and gives half of
            `LCHS.approx_tol` to the kernel integral and half to
            the endpoint guard of the distribution.
        source_time: Default `"discrete"`. How the estimator treats the time
            integral of a constant source. `"discrete"` draws one of the
            `LCHS.duhamel_nodes` Gauss–Legendre nodes, with the same
            source-quadrature error bound as the coherent route.
            `"continuous"` draws the source's elapsed time t in [0, T] with
            density proportional to exp(s*t), for the signed shift s of the
            solution, so no source-quadrature error arises. It has no effect
            on a problem without a source.
        control_copies: Default `1`. Number r of copies of the sampled
            branch value that control the product-formula rotations, using
            r - 1 extra clean qubits and 2(r - 1) CX gates. A
            `QSPBlockEncoding` backend accepts only 1.
        step_buckets: Default `1`. Number of |k| buckets of
            `TrotterErrorBudgeted` step selection. With one bucket every row
            runs the smallest common step count whose product-formula bound
            meets 0.1 times `LCHS.approx_tol`. With more buckets, each
            bucket receives its selected step count under the same bound,
            and a row uses the larger of its two branches' counts. The bound
            is a weighted sum over the nodes when the source and the kernel
            are both discrete, and otherwise the largest bucket bound, because a
            continuous draw gives no exact bucket probabilities. A schedule
            above a set `TrotterErrorBudgeted.max_steps` is refused. More
            than one bucket needs `TrotterErrorBudgeted` and
            `product_formula="plain"`.
        product_formula: Default `"plain"`. How the estimator evolves each
            branch and combines the rows. `"plain"` runs the backend's
            product formula of its `order` with the steps above. The other
            choices reduce the product-formula bias and multiply each row's
            scores by a factor that the sampling radius includes.
            [`Richardson`][nwqlib.algorithms.lchs.route_options.Richardson]
            and [`PTSC`][nwqlib.algorithms.lchs.route_options.PTSC] bound
            the remaining product-formula bias under the premises given for
            each. `"mpf"` draws one of two symmetric product formulas of the
            same depth with weights 3/2 and -1/2 (probabilities 3/4 and 1/4),
            which cancels the leading h**2 term (constant-depth
            multi-product formula, arXiv:2607.12225v1, Sec. IV). It needs
            `Trotter` of `order=2` with an even `steps` r, run as r/2 double
            steps. With at least two distinct nonidentity Pauli labels of H
            and L together (a label that occurs in both counts once), its
            finite-step bias has no bound, so it reports raw estimates and
            sampling radii without a covering radius or a normalized value.
            With at most one distinct nonidentity Pauli label the evolution
            remainder is 0, and when the other
            biases are known the covering radius is reported, with the
            normalized value when the norm covering interval is also
            positive. Every choice other than `"plain"` needs
            `step_buckets=1`, and a `QSPBlockEncoding` backend accepts only
            `"plain"`.
        qsp_bank: Default `"deterministic"`. Phase bank of the estimator
            with a `QSPBlockEncoding` backend, given as a name, a dict or a
            `ProviderConfig`. Planning covers every branch of the support:
            with alpha_L and alpha_H the normalizations of the encodings that
            a branch calls for L and H, a branch with node k and elapsed time
            t has `a = |k| alpha_L + alpha_H`, and the largest product t*a
            over all nodes and elapsed times, b_max, is known before any row
            is drawn. A Pauli or band part is encoded after removing its
            identity term, and a supplied encoding of a Hermitian part is
            used as given. A general A given as one supplied encoding with
            normalization alpha_A is called through two encodings of its
            Hermitian parts L and H, each with normalization alpha_A, so
            `a = (|k| + 1) alpha_A`; every call of either encoding calls A
            once and its adjoint once.
            `buckets=K` (default 1) gives the K bank times
            `b_max * 2**(b - K + 1)`, b = 0..K-1. A row uses the smallest bank
            time t_b at or above both of its branches' t*a. Each branch runs
            a Jacobi–Anger Laurent polynomial of time t_b, fitted once at
            planning (the two-sided target of `QSPBlockEncoding(route="laurent")`),
            on its encoding scaled by `t*a/t_b`, so the polynomial
            approximates that branch's own `exp(-i t (k L_c + H_c))`, with
            L_c and H_c the operators those encodings encode. The
            polynomial's recovery factor zeta_b multiplies the row's scores
            by zeta_b**2. Shots whose QSP qubits do not all return to zero
            score zero, so the mean still divides by every shot.
            `"deterministic"` fits one polynomial per bank time, with
            recovered error at most `LCHS_QSP_EPSILON_FRACTION` (0.1) times
            `LCHS.approx_tol`. `"stochastic"` keeps the Laurent terms
            up to degree `cutoff` (default half the Jacobi–Anger degree of
            the bank time, rounded down) and replaces the
            rest by one randomly drawn higher term with a fixed weight, drawn
            separately for the two branches of every row; the same allowance
            then bounds the error of the average polynomial, which is what the
            measured cross term carries, while each drawn polynomial keeps a
            larger error of its own, recorded in the Plan. It fits one
            polynomial for every supported higher term before drawing rows.
            A larger zeta_b enlarges the score range by zeta_b**2 and the rows
            needed for a fixed sampling radius by zeta_b**4. A lower average
            degree does not by itself make it cheaper. Other parameters:
            `margin` (default 1/16), the polynomial's distance below 1 on the
            unit circle. The member arrays of the whole bank and the first
            transform arrays of every member's phase fit are checked against
            `LCHS.max_bytes` before any row is drawn or phase synthesized. The
            polynomial error estimates are numerical, without a complete
            enclosure of coefficient or phase arithmetic. Refused with other
            backends.

    Raises:
        ValueError: If `product_formula` is the name `"ptsc"`, whose options
            have no defaults, or a recipe other than `"plain"` with
            `step_buckets` other than 1, or if `qsp_bank` names an unknown
            bank, parameter or value.
        TypeError: If `qsp_bank` is not a name, a dict or a `ProviderConfig`.
    """

    kind: Literal["sampled_pairs"] = "sampled_pairs"
    repeats: PositiveInt = 1
    fail_prob: Annotated[Real, Field(gt=0, lt=1)] = .05
    kernel: Literal["discrete", "continuous"] = "discrete"
    source_time: Literal["discrete", "continuous"] = "discrete"
    control_copies: PositiveInt = 1
    step_buckets: PositiveInt = 1
    product_formula: Literal["plain", "richardson", "mpf"] | Annotated[
        Richardson | PTSC, Field(discriminator="kind")] = "plain"
    qsp_bank: Any = Field(default_factory=lambda: ProviderConfig(implementation="deterministic"))

    @field_validator("product_formula", mode="before")
    @classmethod
    def _recipe(cls, value):
        """Replace the name of a recipe with options by its record with default options."""
        if value == "richardson":
            return Richardson()
        if value == "ptsc":
            raise ValueError("product_formula 'ptsc' has no default options; pass PTSC(pf_allowance=..., "
                             "mu_cap=..., taylor_cutoff=...)")
        return value

    @field_validator("qsp_bank", mode="before")
    @classmethod
    def _bank(cls, value):
        return provider_config(value)

    @field_serializer("qsp_bank")
    def _bank_record(self, value):
        """Serialize the bank request as its explicit implementation and parameters."""
        return value.to_dict()

    @model_validator(mode="after")
    def _options(self):
        """Refuse a recipe with step buckets and an invalid bank, which no backend could read."""
        if self.product_formula != "plain" and self.step_buckets != 1:
            raise ValueError(f"product_formula={route_name(self.product_formula)!r} selects its own step counts; "
                             "leave step_buckets=1")
        if self.qsp_bank != ProviderConfig(implementation="deterministic"):
            from .sampled_qsp import bank_options
            bank_options(self.qsp_bank)
        return self


# The record of each route name that has options, by LCHS field; the other names stay strings.
ROUTE_RECORDS = {
    "ham_evo": {record.model_fields["kind"].default: record
                                      for record in (DenseExact, Trotter, TrotterErrorBudgeted, QSPBlockEncoding)},
    "init_prep": {"mps_circuit": MPSCircuit},
    "lcu_prep": {"mps_circuit": MPSCircuit},
    "estimator": {"coherent": Coherent, "sampled_pairs": SampledPairs},
}


def route_name(value):
    """Return the name of a route field's value: the name itself, or the `kind` of its options record."""
    return value if isinstance(value, str) else value.kind


def dense_options(method):
    """Return the DenseExact options of a Method whose backend is `"dense_exact"` or `"auto"` resolved to it.

    `"auto"` carries no route options, so it resolves to `DenseExact()`.
    """
    backend = method.ham_evo
    return DenseExact() if backend == "auto" else backend


def requested_qsp_route(method):
    """Return the QSP route request of a Method whose backend is `"qsp_block_encoding"` or `"auto"` resolved to it.

    `"auto"` carries no route options, so it requests the default `"auto"` route.
    """
    backend = method.ham_evo
    return ProviderConfig(implementation="auto") if backend == "auto" else backend.route
