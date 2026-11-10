"""LCHS choices for the original equation du/dt=-A u+b.

LinearDynamics defines the input and its coordinates. This Method chooses the
numerical approximation, a Run executes the circuits or classical evaluation,
and the chosen readout reconstructs the output.
"""

from nwqlib._limits import exceeds

from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, StrictBool, ValidationInfo, field_serializer, field_validator, model_validator

from nwqlib.algorithms.protocol import Method, ApplicabilityError
from nwqlib.core.records import PositiveInt, Real
from nwqlib.problems.records import LinearDynamics, Solution, StateVector, Samples
from .primary_records import DESCRIPTOR, LCHSAnalysis
from .provider_config import ProviderConfig
from .providers import resolve_lchs_provider_requests
from .route_options import (ROUTE_RECORDS, Coherent, DenseExact, MPSCircuit, QSPBlockEncoding, SampledPairs,
                            Trotter, TrotterErrorBudgeted, provider_config, route_name)
from .solution_error_budget import DEFAULT_PSD_TOLERANCE


class LCHS(Method):
    """LCHS method for the solution of a linear ODE given by `LinearDynamics`.

    Build it with keyword arguments and pass it as `method=`, for example
    `solve(LinearDynamics(A=A, initial_state=u0, time=T), method=LCHS())`.
    Every argument is optional. The result is an
    [`LCHSAnalysis`][nwqlib.algorithms.lchs.primary_records.LCHSAnalysis]
    whose `solution`, for the default `Solution` output, approximates
    `u(T) = exp(-A T) u0 + (integral_0^T exp(-A s) ds) b`, the solution of
    `du/dt = -A u + b` at elapsed time T, with u0 the `initial_state` and b
    the `source` (An, Childs and Lin, arXiv:2312.03916v2, here ACL,
    Eq. (2)).

    Linear combination of Hamiltonian simulation (LCHS) writes `exp(-A T)`
    as an integral over real k of `g(k) exp(-i T (k L + H))`, with
    `A = L + iH`, `L = (A + A^dagger)/2` and `H = (A - A^dagger)/(2i)`
    (ACL Eqs. (3)-(4), (6) and (60)), and replaces the integral by a finite
    sum over quadrature nodes k_j. L must be positive semidefinite,
    or the default `make_l_psd=True` shifts it. The default kernel is ACL's
    near-optimal kernel, Eq. (7), with `beta = 0.75`, on composite Gauss
    quadrature. `lchs_kernel` also offers the Cauchy kernel (An, Liu and
    Lin, doi:10.1103/PhysRevLett.131.150603, restated as ACL Eqs. (5) and
    (181)) and the Low–Somma kernel (Low and Somma, arXiv:2508.19238v2,
    Eq. (6)), whose integral representation is approximate (their Eq. (8)).

    `approx_tol` bounds only the kernel-integral and
    k-quadrature components of the error. It does not bound the total error
    of the physical output, which also depends on input scaling, PSD
    recovery, state preparation, evolution and floating-point error. For a
    dense A the default backend is `dense_exact`, which computes each
    branch matrix `exp(-i t (k_j L + H))` classically and builds one dense
    SELECT circuit that applies branch j when its address register holds
    j. For an A given as a Pauli sum, Cartesian parts, a Hermitian band
    operator or a supplied block encoding it is `qsp_block_encoding`, which
    never forms a matrix of the system dimension. With the default coherent
    estimator, that backend evolves a supplied encoding of a general A from
    its one circuit but builds no evolution under an encoding asserted
    Hermitian, given as A or as a Cartesian part, so planning raises
    ApplicabilityError for it when the elapsed time is positive and the
    initial state or source is nonzero. `estimator="sampled_pairs"`
    evolves both forms (see `SampledPairs.qsp_bank`). The product-formula
    backends build other structured circuits, each with its own limit checks.

    Four fields choose a route: `ham_evo`,
    `init_prep`, `lcu_prep` and
    `estimator`. Each takes the route's name or the record of
    its options from `nwqlib.algorithms.lchs`, for example
    `LCHS(ham_evo=Trotter(steps=4))`. A name stands for
    the record with default options, so `"trotter"` and `Trotter()` give the
    same Method.

    Vector outputs (`Solution`, `StateVector`) need exact amplitude readout,
    so they take no `shots`. `Samples` needs `execution="quantum"` and
    positive `shots`. `execution="classical"` takes no `shots` and does not
    run the `qsp_block_encoding` backend. `result.analyze()` takes no
    settings, because the Plan fixes the recovery scale and coordinates.
    `result.verify(checks=...)` takes one
    [`LCHSVerification`][nwqlib.algorithms.lchs.verification.LCHSVerification]
    or [`LCHSRefinement`][nwqlib.algorithms.lchs.refinement.LCHSRefinement],
    extra computation that `plan`, `solve` and analysis never run. The
    [LCHS guide](../../algorithms/lchs.md) explains kernel and quadrature
    selection, the evolution backends and each error component.

    Examples:
        The exact solution of this two-coordinate problem at T = 0.1 is
        `exp(-0.1 A) [1, 0]`, approximately `[0.96082565, -0.00484022]`.
        The default run on local Aer differs from it by about 0.000821 in
        the L2 norm.

        >>> import numpy as np
        >>> from nwqlib import LinearDynamics, solve
        >>> from nwqlib.algorithms.lchs import LCHS
        >>> problem = LinearDynamics(A=[[.4, .15], [.05, .25]],
        ...                          initial_state=[1, 0], time=.1)
        >>> result = solve(problem, method=LCHS())
        >>> print(np.round(result.solution, 4))
        [ 0.9601-0.j -0.0051+0.j]

    Attributes:
        approx_tol: Default `0.01`, strictly between 0 and 1.
            Construction tolerance of the kernel-quadrature pair, in the
            operator norm for a unit input before PSD growth. The default
            pair gives half to the cutoff tail and half to the k quadrature.
            QSP synthesis, the polynomial approximation and aliasing of an
            `"analytic"` kernel PREP, and each node of
            `trotter_error_budgeted` separately receive 0.1 times this value;
            the analytic PREP's total state-preparation error stays unknown.
            Planning refuses a
            coefficient table whose binary64 rounding `u*alpha` exceeds 0.1
            times this value, with `u = 2**-53` and alpha the coefficient
            1-norm. It is not a tolerance on the total physical-output error.
        lchs_kernel: Default `"near_optimal_eq7"` with `beta=0.75`. Kernel
            g(k) of the integral, given as a name, as a dict
            `{"implementation": name, "parameters": {...}}` or as a
            [`ProviderConfig`][nwqlib.algorithms.lchs.provider_config.ProviderConfig].
            `"near_optimal_eq7"` is ACL Eq. (7) with `beta` strictly between
            0 and 1. `"cauchy_density"` is the Cauchy kernel and takes no
            parameters. `"low_somma_f2"` is the Low–Somma kernel with shift
            `c > 0`, default 1, whose coefficient 1-norm grows with c. It
            requires `approx_tol` at most 4/5 and gives the
            kernel-integral and quadrature components a third of it each
            (Low and Somma, Theorems 2-4). `"apel_berntson"` is the compact
            kernel on (-R, R) of Apel and Berntson (arXiv:2610.03669v1, a
            preprint whose peer review is unconfirmed), with `radius` R,
            default `None` for the radius of their Theorem 6.
        k_quadrature: Default `"composite_gauss"` with
            `truncation_multiplier=1`. Quadrature rule in k, given like
            `lchs_kernel`. `"composite_gauss"` chooses the cutoff and the
            Gauss–Legendre panels from the tolerance, and
            `truncation_multiplier`, at least 1, enlarges the cutoff.
            `"signed_binary_uniform"` is a uniform grid of `2**num_qubits`
            nodes with spacing `2**lsb_position`, both required, and carries
            no error bound. `"symmetric_uniform_trapezoid"` is the Low–Somma
            trapezoid rule and takes no parameters. The allowed pairs are
            `near_optimal_eq7` with `composite_gauss` (both component bounds
            hold) or with `signed_binary_uniform`, `cauchy_density` with
            `signed_binary_uniform`, `low_somma_f2` with
            `symmetric_uniform_trapezoid` (both component bounds hold, and
            the synthesis stages are not budgeted with them), and
            `apel_berntson` with `"tanh_sinh"` (its continuous bound, from
            their Theorem 4 and Proposition 10, and its discretization and
            truncation bounds, from Theorem 8 or, at a
            supplied radius, the adaptation in
            [LCHS derivations](../../algorithms/lchs_derivations.md#apel-berntson-sources),
            each meet their own share). Building the Method refuses any
            other pair. `lchs_kernel="auto"` together
            with `k_quadrature="auto"`, without parameters, lets planning
            choose the pair on the `"qsp_block_encoding"` route from a fixed
            list of at most 12 candidates: the default pair with
            `beta` 0.75, 0.7 and 0.8, and with `beta=0.75` on the narrower
            strip `|Im k| <= 0.5` of its Gauss panel bound; `low_somma_f2`
            with `symmetric_uniform_trapezoid` for `c` 1, 0.5 and 2, each
            also on the uniform grid of J = 2**p - 1 nodes per side when
            that differs; `apel_berntson` with `tanh_sinh` at the paper's
            radius; and, for `Samples` and `accuracy` requests, one more
            `apel_berntson` radius from the stationary point of a
            leading-cost model of the circuit's normalization. For each it
            computes the coefficient grid and chooses the circuit route
            without fitting phases, then keeps the grid whose route costs
            least for the requested output: the CX count per circuit for
            exact or fixed-shot readout, that count times the squared
            recovery factor (the factor that multiplies the measured success
            amplitudes) for `Samples`, and the CX count times the shots of an
            `accuracy` request. Ties keep the earlier candidate, so the
            default pair wins a tie. The list is an engineering choice, not
            an optimum over kernels, parameters or quadrature rules. Each
            candidate is checked against `max_quadrature_work`, `max_bytes`
            and the route limits separately, so a planning pass can build up
            to 12 grids and route choices. Planning keeps only the best candidate so far and the one
            being built, computes only the chosen grid's phases, and records
            every candidate's parameters, outcome, route, cost and work in
            the Plan, with the size, finite kernel mass and component error
            bounds of each grid it built. Building the Method refuses the
            request with another explicit backend, and planning refuses it
            when `"auto"` chooses another backend, the execution is
            classical or the observable estimator is `"sampled_pairs"`.
        ham_evo: Default `"auto"`. How each
            branch `exp(-i t (k_j L + H))` is built, given as a backend name
            or as the record of its options:
            [`DenseExact`][nwqlib.algorithms.lchs.route_options.DenseExact]
            (`"dense_exact"`),
            [`Trotter`][nwqlib.algorithms.lchs.route_options.Trotter]
            (`"trotter"`),
            [`TrotterErrorBudgeted`][nwqlib.algorithms.lchs.route_options.TrotterErrorBudgeted]
            (`"trotter_error_budgeted"`) or
            [`QSPBlockEncoding`][nwqlib.algorithms.lchs.route_options.QSPBlockEncoding]
            (`"qsp_block_encoding"`). A name stands for its record with
            default options. `"auto"` chooses once, at planning, from how A
            is given: `DenseExact()` for a dense A and `QSPBlockEncoding()`
            for a Pauli sum, Cartesian parts, a Hermitian band operator or a
            supplied block encoding. It carries no options, so other options
            of either backend need the backend's record. The Plan reports
            the choice as `reconstruction.selected_backend`, and loading or
            running the Plan uses that choice. `"dense_exact"` computes the
            branch matrix from the node's Hermitian eigensystem and controls
            its circuit. `"trotter"` uses a sorted-Pauli product formula of
            order `Trotter.order` with `Trotter.steps` steps.
            `"trotter_error_budgeted"` gives each node the smallest step
            count whose product-formula bound plus the Pauli pruning bound
            fits 0.1 times `approx_tol` (Childs et al.,
            doi:10.1103/PhysRevX.11.011020, Propositions 9-10 and Sec. V B).
            `"qsp_block_encoding"` evolves under the joint generator with
            quantum signal processing on block encodings of L and H (Pocrnic
            et al., arXiv:2506.20760v2, Section IV) and needs quantum
            execution. A `PeriodicStencil` A needs an explicit `Trotter`
            with `order=2` and quantum execution.
        lcu_select: Default `"auto"`. How the SELECT circuit
            applies the branches. `"dense_exact"` accepts `"auto"` or
            `"branch_controlled"`, and `"qsp_block_encoding"` accepts only
            `"auto"`. With the `"auto"` backend, a value that the backend
            chosen for A cannot use raises ApplicabilityError at planning.
            With `"trotter_error_budgeted"`, `"auto"` gives
            `"multiplexor"`, and `"structured"` is refused, because per-node
            step counts break the affine address structure. With
            `"trotter"`, `"auto"` gives `"structured"` only when an affine
            address structure is eligible, as checked from the k-node
            addresses and the real Pauli coefficients of the generator, and
            its CX count is lower than the multiplexor's, and gives
            `"multiplexor"` otherwise. An explicit `"structured"` without
            that eligibility is refused. A `PeriodicStencil` A accepts
            `"auto"` or `"structured"`. With
            `estimator="sampled_pairs"`, `"auto"` gives the
            multiplexed pair SELECT, and `"structured"` writes each row's two
            rotation angles as an offset and a difference, which saves the two
            CX of every rotation whose angles are exactly equal in that row (for
            example H terms of two branches with the same elapsed time). It is
            refused when a row's angles leave the binary64 window of that form.
        init_prep: Default `"direct"`, which prepares the
            initial state exactly. `"mps_circuit"`, or an
            [`MPSCircuit`][nwqlib.algorithms.lchs.route_options.MPSCircuit]
            with its options, prepares it approximately with a layered
            circuit from a matrix-product-state (MPS) compression of the
            dense initial vector. Quantum execution that evolves the state
            requires `"direct"` with a constant source, a `PeriodicStencil`
            A, or an A given in compact form (a Pauli sum, Cartesian parts, a
            Hermitian band operator or a supplied block encoding), whose
            constructions prepare the state with its own preparation; every
            `estimator="sampled_pairs"` route evolves such an A.
            Zero elapsed time or zero inputs keep the exact readout with
            either choice. Classical execution evaluates the finite sum of
            branches directly and prepares no state, so neither choice changes
            a classical result.
        lcu_prep: Default `"direct"`. Preparation of the
            coefficient state `sqrt(|c_j|/alpha)` that weights the branches,
            with the same choices and restrictions as
            `init_prep`, except that `"qsp_block_encoding"`
            also accepts `"mps_circuit"` with a constant source: it prepares
            the kernel factor and the application factor (initial state and
            source times) of its coefficient state as two separate MPS
            circuits, and its route comparison counts their CX as such. The
            circuit error of `"mps_circuit"` stays unevaluated until an
            explicit validation, even when the compression discards no
            weight. `"analytic"` prepares the kernel factor of the
            `"qsp_block_encoding"` coefficient state from the closed form of
            the `low_somma_f2` kernel on its `symmetric_uniform_trapezoid`
            grid, by quantum signal processing on the node register with
            amplitude amplification, instead of loading its table. It adds
            work qubits that must read zero for success. It applies to the
            routes that encode the k nodes with arithmetic on their indices,
            not to the identity, imaginary-time filter or commuting-product
            routes. Its polynomial approximation and aliasing errors share
            0.1 times `approx_tol`
            ([derivation](../../algorithms/lchs_derivations.md#analytic-preparation)).
            Its numerical state error
            is reported, but the `lcu_coefficient_preparation` error stays
            unknown, because the rounding of its gate angles and the
            synthesis of its gates have no bound.
            Building the Method refuses `"analytic"` with another backend or
            another kernel and quadrature pair. Classical execution prepares
            no state. With `ham_evo="auto"`, zero elapsed time or zero inputs
            take the exact no-evolution path, which does not use `lcu_prep`.
            Other classical requests with `lcu_prep="analytic"` are refused.
            Other choices of `lcu_prep` do not change a classical result.
        estimator: Default `"coherent"`, given as an estimator
            name or as the record of its options:
            [`Coherent`][nwqlib.algorithms.lchs.route_options.Coherent]
            (`"coherent"`) or
            [`SampledPairs`][nwqlib.algorithms.lchs.route_options.SampledPairs]
            (`"sampled_pairs"`). `"coherent"` prepares the solution
            coherently and reads the requested output as above.
            `"sampled_pairs"` estimates a `NormSquared`, `QuadraticForm` or
            `NormalizedExpectation` output from random pairs of LCHS
            branches instead: each row draws two branches x and y and one
            observable term, prepares
            `(|0> chi_x + exp(i theta)|1> chi_y)/sqrt(2)` on one extra qubit
            and the system, and measures that qubit in X and the system in
            the term's basis (An, Liu and Lin, arXiv:2303.01029v2,
            Supplement IV; Chakraborty, arXiv:2302.13555v4, Sec. 3.2). A
            branch starts from the initial state or, with a constant
            source, from the source state at a source time
            (`SampledPairs.source_time`). The mean signed scores estimate the
            physical quadratic form and norm squared, with separate bounds on
            evolution and model bias. Each branch evolves by a product formula
            (`ham_evo` `"trotter"` or
            `"trotter_error_budgeted"`, `lcu_select`
            `"auto"`, `"multiplexor"` or `"structured"`, an A given as a
            Pauli sum or by Pauli parts) or, with `"qsp_block_encoding"`, by
            a polynomial from the finite phase bank of
            `SampledPairs.qsp_bank` on encodings of the parts of A: Pauli or
            band parts, supplied encodings of Hermitian parts, or a general A
            given as one supplied encoding. It needs quantum execution and
            `shots` divisible by `SampledPairs.repeats`, or an `accuracy`
            request (`sampling_shots`). The result's `pair_estimate` gives
            the raw estimates and their covering radii, which only the
            complete set of rows provides. For a Run that has measured only
            some rows, `nwqlib.algorithms.lchs.sampled_pair_progress(run)`
            gives the raw sums of the measured rows. Zero elapsed time or
            zero inputs keep the exact no-evolution readout and draw no
            rows; an `accuracy` request then selects shots per measurement
            setting, which need not be a multiple of `SampledPairs.repeats`
            (`sampling_shots`). When that readout acquires counts, its
            sampling radius uses the counts-setting rule described by
            `Coherent.fail_prob`, with `SampledPairs.fail_prob` as the total
            failure budget, or `1 - accuracy.confidence` when an accuracy
            request selected the shots.
        duhamel_nodes: Default `8`. Number of Gauss–Legendre nodes of the
            time integral of a constant source, one panel on [0, T] (ACL
            Eq. (72)). It is never reduced automatically to fit a limit. It
            applies only when the Problem has a source: a Problem without a
            source has no such integral, so the same Method serves both.
            `SampledPairs(source_time="continuous")` draws the source time
            instead of using these nodes.
        source_drop: Default `None`. Absolute error allowance,
            in the units of the solution, for leaving out constant-source
            applications whose magnitude `w*||input||*exp(s*t)` is below
            `2**-1022` times the largest one, so that binary64 cannot
            represent them beside it. This happens only for extreme input
            norms or decay. Planning omits such applications only when the
            allowance covers the bound `sum_j |c_j| * 2**(E - 1022)` per
            omitted application, with `2**E` the common scale, and reports
            that bound as the planning fact `source_range_omission`.
            `None` refuses such a source instead. It applies only when the
            Problem has a source.
        make_l_psd: Default `True`. Rule for an eigenvalue of L below
            `-psd_tol*||L||_2`. `True` shifts L by
            `s = -lambda_min + psd_tol*||L||_2` and multiplies the
            result by the growth factor `exp(s*T)`. `False` refuses the
            input. A compact A has a lower bound `ell` of L instead of an
            eigenvalue, from its coefficients or from what you state to
            `ingest_cartesian`. `True` shifts by `-ell` when `ell` is negative,
            and `False` refuses a negative `ell`, which is an enclosure and
            not a negative eigenvalue. A positive `ell` is known decay, which
            planning extracts under either value.
        psd_tol: Default `1e-12`, positive. Relative window of the PSD
            check. An eigenvalue of L at or above `-psd_tol*||L||_2`
            is accepted without a shift. The window is the scale of the
            eigensolver's rounding (LAPACK Users' Guide, 3rd ed., Sec. 4.7),
            so the decision does not depend on the time unit. It does not
            permit a negative eigenvalue below that window.
        max_bytes: Default `None`, the physical memory of this computer. Upper
            limit on the known bytes of input and numerical workspace. Each
            step checks the estimated bytes of the large arrays it is about
            to form before forming them. Planning a compact generator forms
            no array of the system dimension, and a provider comparison
            keeps at most two candidates at once. These are coarse estimates
            of the dominant arrays; Python objects and SDK, compiler and
            process memory are not counted, so the limit is not a bound on
            the process's peak memory. `prepare` also compares this limit
            with a coarse estimate of the circuit's memory before building
            the circuit, and a periodic stencil is compared only there, so
            its `plan` and `estimate` stay available at any size. The
            refusal names any calls the estimate could not count.
        max_svd_work: Default `None`, no work limit. Upper limit on the
            counted work of state and coefficient compression, including the
            construction of an `"mps_circuit"` layered circuit.
        max_spectral_work: Default `None`, no work limit. Upper limit on the counted work of
            the eigenvalue check of L for dense A, `d**3` for physical
            dimension d. An explicit cap of 100,000,000 accepts a dense A of
            dimension at most 464 when L is neither zero nor diagonal. A zero
            or diagonal L takes its smallest and largest eigenvalues from its
            diagonal, with no eigenvalue check and no work counted. A
            [periodic stencil](../../algorithms/lchs.md#periodic-stencils)
            has an analytically PSD L and no eigenvalue check.
        max_quadrature_work: Default `None`, no work limit. Upper limit on nominal, size-based
            work for building the k nodes and weights and for constant-source
            quadrature and branch-layout construction. For `"composite_gauss"` it
            covers the completed panel search plus the construction of the
            chosen rule, not every scalar probe. The limit applies to each
            construction separately: each candidate grid of
            `lchs_kernel="auto"`, and the radius search of its stationary
            `apel_berntson` candidate, is checked against it on its own.
        max_select_work: Default `None`, no work limit. Upper limit on the counted work of
            building the SELECT circuit, including the branch matrices and
            their synthesis for `"dense_exact"` and the Pauli decomposition
            of the product-formula backends. The
            [LCHS guide](../../algorithms/lchs.md#native-methods-and-physical-coordinates)
            gives sizes under explicit work caps.
        max_readout_work: Default `None`, no work limit. Optional positive integer giving an
            inclusive limit on the comparisons made when grouping the Pauli
            terms of sampled observables into qubit-wise commuting measurement
            settings. Planning checks and records the grouping comparisons. A
            work-limit refusal names `LCHS.max_readout_work`.

    Raises:
        ValueError: If the kernel-quadrature pair is not allowed or a kernel
            or quadrature parameter is invalid; if `lchs_kernel="auto"` is
            combined with a backend other than `"auto"` or
            `"qsp_block_encoding"`; if `lcu_prep="analytic"` is
            combined with another backend or another kernel-quadrature pair;
            if `"sampled_pairs"` is combined with a backend other than
            `"trotter"`, `"trotter_error_budgeted"` or
            `"qsp_block_encoding"`, with `"qsp_block_encoding"` and a
            product-formula option (`SampledPairs.product_formula`,
            `step_buckets` or `control_copies`), a SELECT choice or a
            `QSPBlockEncoding.route`, with a product-formula backend and a
            non-default `SampledPairs.qsp_bank` or a `"branch_controlled"`
            SELECT, with a continuous `SampledPairs.kernel` and a kernel other
            than `"apel_berntson"` or `"low_somma_f2"`, or with several
            `SampledPairs.step_buckets` without `"trotter_error_budgeted"`;
            or if a `SampledPairs.product_formula` recipe gets a backend,
            `order` or step count it is not built on, or an option it does
            not use, or lacks one it needs. Each route record raises its own
            errors for its options.
        TypeError: If `lchs_kernel` or `k_quadrature` is not a name, a dict
            or a `ProviderConfig`.
    """

    schema_version: Literal[7] = 7
    approx_tol: Annotated[Real, Field(gt=0, lt=1)] = .01
    lchs_kernel: Any = Field(default_factory=lambda: ProviderConfig(implementation="near_optimal_eq7"))
    k_quadrature: Any = Field(default_factory=lambda: ProviderConfig(implementation="composite_gauss"))
    ham_evo: Literal[
        "auto", "dense_exact", "trotter", "trotter_error_budgeted", "qsp_block_encoding"
    ] | Annotated[DenseExact | Trotter | TrotterErrorBudgeted | QSPBlockEncoding, Field(discriminator="kind")] = "auto"
    lcu_select: Literal["auto", "multiplexor", "structured", "branch_controlled"] = "auto"
    init_prep: Literal["direct", "mps_circuit"] | MPSCircuit = "direct"
    lcu_prep: Literal["direct", "mps_circuit", "analytic"] | MPSCircuit = "direct"
    estimator: Literal["coherent", "sampled_pairs"] | Annotated[
        Coherent | SampledPairs, Field(discriminator="kind")] = "coherent"
    duhamel_nodes: PositiveInt = 8
    source_drop: Annotated[Real, Field(ge=0)] | None = None
    make_l_psd: StrictBool = True
    psd_tol: Annotated[Real, Field(gt=0)] = DEFAULT_PSD_TOLERANCE
    max_bytes: PositiveInt | None = None
    max_svd_work: PositiveInt | None = None
    max_spectral_work: PositiveInt | None = None
    max_quadrature_work: PositiveInt | None = None
    max_select_work: PositiveInt | None = None
    max_readout_work: PositiveInt | None = None

    descriptor: ClassVar = DESCRIPTOR
    result_type: ClassVar = LCHSAnalysis

    @field_validator("lchs_kernel", "k_quadrature", mode="before")
    @classmethod
    def _provider(cls, value):
        """Accept a registry name, a {"implementation", "parameters"} dict or a ProviderConfig."""
        return provider_config(value)

    @field_serializer("lchs_kernel", "k_quadrature")
    def _provider_record(self, value):
        """Serialize a provider request as its explicit implementation and parameters."""
        return value.to_dict()

    @field_validator("ham_evo", "init_prep", "lcu_prep",
                     "estimator", mode="before")
    @classmethod
    def _route(cls, value, info: ValidationInfo):
        """Replace the name of a route that has options by its record with default options.

        This runs before the content hash exists, so a name and its default
        record are one Method with one content hash and one saved form.
        """
        record = ROUTE_RECORDS[info.field_name].get(value) if isinstance(value, str) else None
        return value if record is None else record()

    @model_validator(mode="after")
    def _choices(self):
        """Reject configurations whose choices conflict, before any problem is seen.

        The kernel-quadrature pair must be registered and allowed. Each route
        record checks its own options; the checks here relate the options of
        different fields. Provider enumeration compares the rows of the
        coherent QSP route, and an analytic coefficient PREP prepares that
        route's kernel factor. Sampled pairs evolve each branch by a product
        formula or a finite QSP bank (`_sampled_qsp_choices`), with the
        recipe premises of `_recipe_choices`. An option that the chosen
        routes do not read is refused rather than ignored.
        """
        backend = route_name(self.ham_evo)
        enumeration = [config.implementation == "auto" for config in (self.lchs_kernel, self.k_quadrature)]
        if any(enumeration):
            if not all(enumeration) or self.lchs_kernel.parameters or self.k_quadrature.parameters:
                raise ValueError("provider enumeration needs lchs_kernel='auto' and k_quadrature='auto' together, "
                                 "without parameters; a concrete kernel or quadrature keeps its pair fixed")
            if backend not in ("auto", "qsp_block_encoding"):
                raise ValueError("lchs_kernel='auto' compares the providers by the coherent route's rows; use "
                                 "ham_evo 'auto' or 'qsp_block_encoding', or a concrete pair")
        else:
            resolve_lchs_provider_requests(self.lchs_kernel, self.k_quadrature)
        if self.lcu_prep == "analytic":
            pair = (self.lchs_kernel.implementation, self.k_quadrature.implementation)
            if backend not in ("auto", "qsp_block_encoding"):
                raise ValueError("lcu_prep='analytic' prepares the kernel factor of the coherent LCU; "
                                 "use ham_evo 'auto' or 'qsp_block_encoding'")
            if pair != ("low_somma_f2", "symmetric_uniform_trapezoid"):
                raise ValueError("lcu_prep='analytic' prepares the low_somma_f2 kernel on its "
                                 f"symmetric_uniform_trapezoid grid, not {pair[0]!r} with {pair[1]!r}")
        estimator = self.estimator
        if type(estimator) is Coherent:
            return self
        if backend == "qsp_block_encoding":
            return self._sampled_qsp_choices()
        if backend not in ("trotter", "trotter_error_budgeted"):
            raise ValueError("estimator='sampled_pairs' evolves each branch by a Pauli product formula or a "
                             "finite QSP bank; use ham_evo 'trotter', "
                             "'trotter_error_budgeted' or 'qsp_block_encoding'")
        if estimator.qsp_bank != ProviderConfig(implementation="deterministic"):
            raise ValueError("SampledPairs.qsp_bank chooses the phase bank of sampled pairs with "
                             "ham_evo='qsp_block_encoding'")
        if self.lcu_select not in ("auto", "multiplexor", "structured"):
            raise ValueError("sampled pairs build the multiplexed or structured pair SELECT; use "
                             "lcu_select 'auto', 'multiplexor' or 'structured'")
        self._continuous_kernel()
        if estimator.step_buckets > 1 and backend != "trotter_error_budgeted":
            raise ValueError("SampledPairs.step_buckets > 1 selects bucket step counts, which needs "
                             "ham_evo='trotter_error_budgeted'")
        self._recipe_choices()
        return self

    def _continuous_kernel(self):
        """Refuse a continuous kernel draw without a kernel whose importance weights are bounded on its distribution."""
        if self.estimator.kernel == "continuous" and self.lchs_kernel.implementation not in (
                "apel_berntson", "low_somma_f2"):
            raise ValueError("SampledPairs(kernel='continuous') needs lchs_kernel 'apel_berntson' or 'low_somma_f2', "
                             "whose importance weights have a bound on the truncated Cauchy distribution")

    def _sampled_qsp_choices(self):
        """Reject a sampled QSP Method with a product-formula or coherent option that its route does not read.

        Each branch runs a finite Laurent bank on the centered children
        (`sampled_qsp`), so the product-formula recipe, step buckets, pair
        SELECT and its control copies, and the coherent transformation of
        ``QSPBlockEncoding.route`` are not used; they keep their defaults
        rather than being ignored. `SampledPairs` checks its bank. Provider
        enumeration and a non-direct coefficient PREP are refused at
        planning, with ApplicabilityError, so that ``compare`` keeps the row.
        """
        estimator = self.estimator
        changed = [name for name, unread in (
            ("SampledPairs.product_formula", estimator.product_formula != "plain"),
            ("SampledPairs.step_buckets", estimator.step_buckets != 1),
            ("SampledPairs.control_copies", estimator.control_copies != 1),
            ("lcu_select", self.lcu_select != "auto"),
            ("QSPBlockEncoding.route", self.ham_evo.route != ProviderConfig(
                implementation="auto"))) if unread]
        if changed:
            raise ValueError(f"{', '.join(changed)} {'does' if len(changed) == 1 else 'do'} not apply to sampled "
                             "pairs with ham_evo='qsp_block_encoding', whose branches run the "
                             "finite QSP bank of SampledPairs.qsp_bank")
        self._continuous_kernel()
        return self

    def _recipe_choices(self):
        """Reject a product-formula recipe whose backend, order, steps or options do not match it.

        Each recipe fixes its base formula: Richardson extrapolates one
        Strang (order 2) word, PTSC compensates the Lie (order 1) step and
        the MPF members are order-2 words. A recipe's step count comes from
        the fixed ``Trotter.steps`` or from its own allowance, never from the
        plain buckets (`SampledPairs` checks the buckets), and an option a
        recipe does not read is refused rather than ignored. PTSC requires
        its three options as fields of its record.
        """
        recipe = self.estimator.product_formula
        name = route_name(recipe)
        if name == "plain":
            return
        backend = self.ham_evo
        budgeted = type(backend) is TrotterErrorBudgeted
        if name == "richardson" and budgeted and recipe.pf_allowance is None:
            raise ValueError("product_formula='richardson' with ham_evo="
                             "'trotter_error_budgeted' needs Richardson(pf_allowance=...)")
        if name == "richardson" and not budgeted and recipe.pf_allowance is not None:
            raise ValueError("Richardson.pf_allowance does not apply to product_formula='richardson' with fixed "
                             "Trotter.steps")
        order = 1 if name == "ptsc" else 2
        if backend.order != order:
            raise ValueError(f"product_formula={name!r} is built on the order-{order} product formula; set "
                             f"order={order} in the backend's options")
        if name == "ptsc" and not budgeted:
            raise ValueError("product_formula='ptsc' chooses its step count from PTSC.mu_cap and its cutoff from "
                             "PTSC.pf_allowance; use ham_evo='trotter_error_budgeted'")
        if name in ("richardson", "mpf") and not budgeted and backend.steps is None:
            raise ValueError(f"product_formula={name!r} needs Trotter.steps; Trotter.synthesis_tol chooses "
                             "periodic-stencil steps only")
        if name == "mpf" and (budgeted or backend.steps % 2):
            raise ValueError("product_formula='mpf' has no finite-step bias bound to choose steps from; use "
                             "ham_evo=Trotter(steps=r) with an even r, run as r/2 double steps")
        if name == "richardson" and not budgeted and exceeds(2 * backend.steps, backend.max_steps):
            raise ValueError(f"product_formula='richardson' also runs 2*Trotter.steps = {2 * backend.steps} "
                             f"steps, which exceeds Trotter.max_steps={backend.max_steps}")

    def _accuracy_request(self, problem, *, output, accuracy, execution):
        """Refuse a sampling-accuracy request outside the domain of `sampling_shots`, before any planning work."""
        from nwqlib.problems.records import Accuracy, NormSquared, QuadraticForm
        if not isinstance(problem, LinearDynamics):
            raise ApplicabilityError("LCHS requires LinearDynamics(A, initial_state, time, source)")
        if (not isinstance(accuracy, Accuracy) or accuracy.component != "sampling" or accuracy.absolute_tolerance is None
                or accuracy.relative_tolerance is not None):
            raise ApplicabilityError("LCHS sampling accuracy takes Accuracy(absolute_tolerance=..., component="
                                     "'sampling'); a total or relative accuracy is not a sampling request")
        if execution != "quantum":
            raise ApplicabilityError("LCHS sampling accuracy chooses the shots of quantum counts")
        if not isinstance(output, (NormSquared, QuadraticForm)):
            raise ApplicabilityError("LCHS sampling accuracy supports the physical NormSquared and QuadraticForm "
                                     "outputs: a normalized expectation needs a positive denominator interval that is "
                                     "unknown before the counts, a vector has no shots and Samples are Born draws")
        estimator = self.estimator
        failure = estimator.fail_prob
        if failure != .05 and 1.0-failure != accuracy.confidence:
            name = type(estimator).__name__
            raise ApplicabilityError(f"{name}.fail_prob={failure!r} disagrees with the accuracy "
                                     f"confidence {accuracy.confidence!r}; the request sets the confidence of its "
                                     "radius, so leave fail_prob at its default or set it to "
                                     "1 - confidence")
        output.frame(problem)

    def sampling_shots(self, problem, *, output, accuracy, execution, shots):
        """Return the shots per measurement setting that meet an absolute sampling tolerance.

        `plan`, `solve` and `compare` call it when they receive
        `accuracy=Accuracy(absolute_tolerance=epsilon, component="sampling",
        confidence=c)` in place of `shots`. It chooses the route as planning
        does and gives each candidate route the count that makes its
        sampling radius at most epsilon with probability at least c, for the
        moment of the finite circuit that the route builds, provided each
        setting returns at least the selected number of independent shots:
        `N = max(1, ceil(2 Gamma**4 A**2 log(2 G/delta)/epsilon**2))` shots
        on each of the G measurement settings of a `QuadraticForm`, with A
        the sum over the settings of their Pauli coefficient magnitudes (the
        identity coefficient counted once) and `delta = 1 - c`, or
        `N = max(1, ceil(Gamma**4 log(2/delta)/(2 epsilon**2)))` for
        `NormSquared`.
        Gamma is the route's physical recovery factor, the factor that
        multiplies the measured success amplitudes. These counts invert the
        radius that the result reports (see `Coherent.fail_prob`).
        Among the admitted routes it then takes the one with the least
        `N G C`, where C is the route's CX count per circuit, an upper bound
        before routing, so the choice minimizes this forecast and not a
        measured cost. When no admitted route has a known CX count, the one
        with the smallest recovery factor is taken. The radius covers the
        shot noise of the circuit's
        moment, not its approximation error, which stays in the result's
        `algorithmic_approximation`. No phase is synthesized while choosing.
        Within the same call of `plan`, `solve` or `compare`, planning reads
        the choice made here instead of making it again, checks the count
        and fits phases for the chosen route only.

        With `estimator="sampled_pairs"`, when evolution draws pair rows, the
        count is `B * m` shots, m = `SampledPairs.repeats`, for
        `B = max(1, ceil(2 (H/epsilon)**2 log(2/delta)))` independent rows,
        where H bounds the physical score of one row over every row the
        sampling law can draw: the square of the largest branch importance
        weight, times the largest |o_l|/p_l of the observable terms for a
        `QuadraticForm`, times the largest factor of the branch evolution (1
        for a plain product formula, the largest absolute stored importance
        multiplier `abs(b_j/pi_j)` over the Richardson or MPF member law,
        `PTSC.mu_cap**2` for PTSC, the largest zeta_b**2 of the QSP bank).
        At zero elapsed time or zero inputs no pair rows are drawn, and the
        no-evolution readout takes its own count: the `QuadraticForm` count
        `N` above per measurement setting, which need not be a multiple of
        m, or no shots when the output is known from the input.
        This count constrains the Hoeffding sampling radius only. The
        covering radius also includes evolution and model bias. The sampling
        radius counts rows, so m does not reduce B. Each pass plans the
        sampling laws and the phase bank without drawing a row or fitting a
        phase; planning then fits the bank and draws the rows once. A B
        whose row data exceed `max_bytes` is refused with that count.

        The B rows are the points of one range axis of the Plan's single
        measurement setting, so the Plan's `Program` (NWQLib's description
        of a circuit as named steps) has the same size for every B.
        `compare` turns an ApplicabilityError into a row with a reason but
        passes a ValueError on to its caller.

        Args:
            problem (LinearDynamics): The problem.
            output (NormSquared | QuadraticForm): The requested physical
                output.
            accuracy (Accuracy): `component="sampling"` with an
                `absolute_tolerance` in the output's unit.
            execution (str): `"quantum"`.
            shots (int | None): Must be None.

        Returns:
            shots (int | None): The shots per setting, or None when the
                Plan measures nothing (zero elapsed time or zero inputs with
                an output known from the input).

        Raises:
            ApplicabilityError: For a total or relative request, classical
                execution, a normalized expectation, a vector or Samples
                output, a backend other than `"qsp_block_encoding"` (after
                `"auto"` is resolved from A) for coherent estimation, a
                `PeriodicStencil` A, a non-default
                `Coherent.fail_prob` or
                `SampledPairs.fail_prob` other than
                `1 - confidence`, a choice that sampled pairs
                cannot plan, or when no coherent route has a count of
                shots that a saved Plan can hold.
            ValueError: If `shots` is also given, or if the sampled rows
                would exceed `max_bytes`.
        """
        self._accuracy_request(problem, output=output, accuracy=accuracy, execution=execution)
        if shots is not None:
            raise ValueError("explicit shots conflict with an accuracy-selected sampling allocation")
        from .selection import accuracy_shots
        return accuracy_shots(self, problem, output=output, accuracy=accuracy)

    def before_submit(self, plan, prepared, *, run):
        """Refuse reused fixed sampling streams before submitting a Plan whose shots an accuracy request chose."""
        if plan.selection_accuracy is None:
            return
        run.admit_count_preparations(prepared, require_all=True)

    def plan(self, problem, *, output, execution, shots, rng, accuracy=None):
        """Check output, execution and shots against LCHS, then build the Plan once.

        A complex vector needs amplitude readout, so vector outputs reject
        shots. Samples need quantum execution with shots. QSP evolution has no
        classical execution. All numerical choices are made in plan_lchs, or
        in sampled_plan.plan_sampled for estimator="sampled_pairs".
        With ``accuracy``, ``shots`` is the count `sampling_shots` returned.
        Planning reads the choice that `sampling_shots` made in the same
        planning call, checks that count and records the request in
        ``Plan.selection_accuracy``. ``dense_exact`` and
        ``qsp_block_encoding`` apply no product formula and record 0 steps.
        """
        if not isinstance(problem, LinearDynamics):
            raise ApplicabilityError("LCHS requires LinearDynamics(A, initial_state, time, source)")
        if output.kind not in self.descriptor.output_families:
            raise ApplicabilityError("LCHS supports a physical solution, state, norm, observable or samples")
        if accuracy is not None:
            self._accuracy_request(problem, output=output, accuracy=accuracy, execution=execution)
        if type(self.estimator) is SampledPairs:
            from .sampled_plan import plan_sampled
            return plan_sampled(self, problem, output=output, execution=execution, shots=shots, rng=rng,
                                accuracy=accuracy)
        if isinstance(output, (Solution, StateVector)) and shots is not None:
            raise ApplicabilityError("counts do not recover a complex vector; omit shots or select a scalar/Samples output")
        if isinstance(output, Samples) and (shots is None or execution != "quantum"):
            raise ApplicabilityError("Samples requires quantum execution and finite shots")
        if execution == "classical" and shots is not None:
            raise ApplicabilityError("classical LCHS acquires deterministic numerical data; shots belong to quantum readout")
        if execution == "classical" and type(self.ham_evo) is QSPBlockEncoding:
            raise ApplicabilityError("QSP LCHS has no selected classical acquisition; use quantum execution")
        from .selection import plan_lchs
        return plan_lchs(self, problem, output=output, execution=execution, shots=shots, rng=rng, accuracy=accuracy)

    def analyze(self, plan, data, *, settings):
        """Reconstruct the requested output from the execution data (analysis.analyze).

        LCHS fixes its recovery scale and coordinates in the Plan, so there
        are no reconstruction settings to choose after execution.
        """
        from nwqlib._validation import reanalysis_settings
        from .analysis import analyze

        reanalysis_settings(settings, (), "LCHS")
        return analyze(plan, data)

    def has_row_templates(self, plan):
        """Return whether `plan` is a sampled-pair Plan, the only LCHS Plan whose rows `row_templates` groups."""
        return getattr(plan.reconstruction, "pair_rows", None) is not None

    def row_templates(self, plan, points):
        """Group the rows of a sampled-pair Plan by the gates of their pair circuits (`sampled_plan.row_templates`).

        Every sampled-pair row belongs to one group, whatever its product
        formula recipe. Other LCHS Plans return None.
        """
        if getattr(plan.reconstruction, "pair_rows", None) is None:
            return None
        from .sampled_plan import row_templates
        return row_templates(plan, points)

    def save_archive(self, plan, files):
        """Save the Plan's numerical data and the data assigned to each circuit block (archive.save)."""
        from .archive import save
        return save(self, plan, files)

    @classmethod
    def load_archive(cls, data, files):
        """Restore a saved Plan with its numerical data exactly as saved, after checking the archive format (archive.load)."""
        from .archive import load
        return load(data, files)

    def verify(self, plan, result, *, checks):
        """Run one explicit LCHSVerification reference or LCHSRefinement request.

        Both are extra work requested by the caller. Neither is part of plan,
        solve or analysis.
        """
        if self != plan.method:
            raise ValueError("LCHS verification requires the Plan's original configured Method")
        from .verification import LCHSVerification, verify
        from .refinement import LCHSRefinement, refine
        if type(checks) is LCHSVerification:
            return verify(plan, result, checks=checks)
        if type(checks) is LCHSRefinement:
            return refine(plan, result, checks=checks)
        raise TypeError("checks requires one explicit LCHSVerification or LCHSRefinement")
