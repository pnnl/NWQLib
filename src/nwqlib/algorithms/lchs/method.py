"""LCHS choices for the original equation du/dt=-A u+b.

LinearDynamics defines the input and its coordinates. This Method chooses the
numerical approximation, a Run executes the circuits or classical evaluation,
and the chosen readout reconstructs the output.
"""

from nwqlib._limits import exceeds

from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, PrivateAttr, StrictBool, field_serializer, field_validator, model_validator

from nwqlib.algorithms.protocol import Method, ApplicabilityError
from nwqlib.core.records import PositiveInt, Real
from nwqlib.problems.records import LinearDynamics, Solution, StateVector, Samples
from .primary_records import DESCRIPTOR, LCHSAnalysis
from .provider_config import ProviderConfig
from .providers import resolve_lchs_provider_requests
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

    `approximation_tolerance` bounds only the kernel-integral and
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
    initial state or source is nonzero. `observable_estimator="sampled_pairs"`
    evolves both forms (see `pair_qsp_bank`). The product-formula backends
    build other structured circuits, each with its own limit checks.

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
        approximation_tolerance: Default `0.01`, strictly between 0 and 1.
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
            requires `approximation_tolerance` at most 4/5 and gives the
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
        hamiltonian_evolution_backend: Default `"auto"`. How each
            branch `exp(-i t (k_j L + H))` is built. `"auto"` chooses once,
            at planning, from how A is given: `"dense_exact"` for a dense A
            and `"qsp_block_encoding"` for a Pauli sum, Cartesian parts, a
            Hermitian band operator or a supplied block encoding. The Plan
            reports the choice as `reconstruction.selected_backend`, and
            loading or running the Plan uses that choice. A `PeriodicStencil`
            A needs an explicit `"trotter"` (see below). `"dense_exact"` computes
            the branch matrix from the node's Hermitian eigensystem and
            controls its circuit. `"trotter"` uses a sorted-Pauli product
            formula of order `trotter_order` with `trotter_steps` steps.
            `"trotter_error_budgeted"` gives each node the smallest step
            count whose product-formula bound plus the Pauli pruning bound
            fits 0.1 times `approximation_tolerance` (Childs et al.,
            doi:10.1103/PhysRevX.11.011020, Propositions 9-10 and Sec. V B).
            `"qsp_block_encoding"` evolves under the joint generator with
            quantum signal processing on block encodings of L and H (Pocrnic
            et al., arXiv:2506.20760v2, Section IV) and needs quantum
            execution. A `PeriodicStencil` A needs `"trotter"` with
            `trotter_order=2` and quantum execution.
        lcu_select_implementation: Default `"auto"`. How the SELECT circuit
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
            `observable_estimator="sampled_pairs"`, `"auto"` gives the
            multiplexed pair SELECT, and `"structured"` writes each row's two
            rotation angles as an offset and a difference, which saves the two
            CX of every rotation whose angles are exactly equal in that row (for
            example H terms of two branches with the same elapsed time). It is
            refused when a row's angles leave the binary64 window of that form.
        dense_control_route: Default `"auto"`. How a construction controls
            a dense unitary on two or more qubits. It chooses the controlled
            synthesis of each `dense_exact` branch on its address bits.
            `"gatewise"` synthesizes the unitary and lets Qiskit control
            each synthesized gate, `"whole_matrix"` synthesizes the
            controlled matrix, and `"auto"` takes the whole-matrix route for
            one control and the gate-wise route for more. On the
            whole-matrix route a constant-source branch is one unitary, its
            evolution times the matrix of its input preparation. The
            `qsp_block_encoding` backend controls each dense-dilation child
            gate-wise on every route.
        trotter_steps: Default `1`. Fixed step count of the `"trotter"`
            backend, a positive integer at most `max_trotter_steps`, or
            `None` when `trotter_synthesis_tolerance` chooses it. Exactly one
            of the two is set. `"trotter_error_budgeted"` chooses its own
            counts and requires the default 1. `"auto"`, `"dense_exact"` and
            `"qsp_block_encoding"` apply no product formula, so they treat
            any value as 1, and planning warns about a value other than 1.
            A fixed count leaves the `trotter_synthesis` error of a general
            A unknown, so the total error is unknown. `plan.facts` then lists
            `trotter_pruning`, the physical-solution bound on the Pauli terms
            removed from each node generator, for any requested output: a term
            is removed when its size is at most 1e-12 times the largest
            coefficient of H or k*L at that node, the identity included. This
            is one part of the unknown error, not a bound on the total.
            Printing the Result shows a nonzero value. A classical Result also
            lists it in `result.facts`. A quantum Result does not, because
            its facts are in the frame of the requested output.
        trotter_synthesis_tolerance: Default `None`. Positive tolerance on
            the weighted Strang synthesis bound of a `PeriodicStencil` A, in the
            same unit-input norm as `approximation_tolerance`. Planning
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
            It requires
            `hamiltonian_evolution_backend="trotter"` and
            `trotter_steps=None`, and other inputs refuse it.
        trotter_order: Default `2`. Order of the product formula of the
            `"trotter"` and `"trotter_error_budgeted"` backends, 1 for the
            Lie formula or 2 for the symmetric Suzuki (Strang) formula.
            `"auto"`, `"dense_exact"` and `"qsp_block_encoding"` accept only 2.
        max_trotter_steps: Default `None`, no limit. Optional upper limit
            on the product-formula step count of any node, fixed or chosen
            automatically. A SELECT stores one angle row per distinct
            repetition block, so the step count does not by itself set the
            size of the planned tables.
        duhamel_nodes: Default `8`. Number of Gauss–Legendre nodes of the
            time integral of a constant source, one panel on [0, T] (ACL
            Eq. (72)). It is never reduced automatically to fit a limit.
        source_drop_allowance: Default `None`. Absolute error allowance,
            in the units of the solution, for leaving out constant-source
            applications whose magnitude `w*||input||*exp(s*t)` is below
            `2**-1022` times the largest one, so that binary64 cannot
            represent them beside it. This happens only for extreme input
            norms or decay. Planning omits such applications only when the
            allowance covers the bound `sum_j |c_j| * 2**(E - 1022)` per
            omitted application, with `2**E` the common scale, and reports
            that bound as the planning fact `source_range_omission`.
            `None` refuses such a source instead.
        make_l_psd: Default `True`. Rule for an eigenvalue of L below
            `-psd_tolerance*||L||_2`. `True` shifts L by
            `s = -lambda_min + psd_tolerance*||L||_2` and multiplies the
            result by the growth factor `exp(s*T)`. `False` refuses the
            input. A compact A has a lower bound `ell` of L instead of an
            eigenvalue, from its coefficients or from what you state to
            `ingest_cartesian`. `True` shifts by `-ell` when `ell` is negative,
            and `False` refuses a negative `ell`, which is an enclosure and
            not a negative eigenvalue. A positive `ell` is known decay, which
            planning extracts under either value.
        psd_tolerance: Default `1e-12`, positive. Relative window of the PSD
            check. An eigenvalue of L at or above `-psd_tolerance*||L||_2`
            is accepted without a shift. The window is the scale of the
            eigensolver's rounding (LAPACK Users' Guide, 3rd ed., Sec. 4.7),
            so the decision does not depend on the time unit. It does not
            permit a negative eigenvalue below that window.
        initial_state_preparation: Default `"direct"`, which prepares the
            initial state exactly. `"mps_circuit"` prepares it approximately
            with a layered circuit from a matrix-product-state (MPS)
            compression of the dense initial vector. Quantum execution that
            evolves the state requires `"direct"` with a constant source, a
            `PeriodicStencil` A, or an A given in compact form (a Pauli sum,
            Cartesian parts, a Hermitian band operator or a supplied block
            encoding), whose constructions prepare the state with its own
            preparation; every `observable_estimator="sampled_pairs"` route
            evolves such an A. Zero elapsed time or zero inputs keep the
            exact readout with either choice.
        initial_state_mps_max_bond_dim: Default `None`, no limit. Upper
            limit on the kept bond dimension of the initial-state
            compression.
        initial_state_mps_threshold: Default `1e-14`, positive.
            Singular-value truncation threshold of the initial-state
            compression.
        initial_state_mps_num_layers: Default `2`, positive. Number of
            layers of the initial-state MPS circuit.
        lcu_state_preparation: Default `"direct"`. Preparation of the
            coefficient state `sqrt(|c_j|/alpha)` that weights the branches,
            with the same choices and restrictions as
            `initial_state_preparation`, except that `"qsp_block_encoding"`
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
            0.1 times `approximation_tolerance`
            ([derivation](../../algorithms/lchs_derivations.md#analytic-preparation)).
            Its numerical state error
            is reported, but the `lcu_coefficient_preparation` error stays
            unknown, because the rounding of its gate angles and the
            synthesis of its gates have no bound.
            Building the Method refuses `"analytic"` with another backend or
            another kernel and quadrature pair.
        lcu_mps_max_bond_dim: Default `None`, no limit. Upper limit on the
            kept bond dimension of the coefficient-state compression.
        lcu_mps_threshold: Default `1e-14`, positive. Singular-value
            truncation threshold of the coefficient-state compression.
        mps_num_layers: Default `2`, positive. Number of layers of the
            coefficient-state MPS circuit.
        max_dense_select_slots: Default `4_096`. Upper limit on the number
            of address slots of the dense SELECT circuit of `"dense_exact"`,
            the branch count rounded up to a power of two. Planning refuses a
            larger construction and never changes the grid, Duhamel nodes or
            tolerance to fit. A composite Gauss grid that alone exceeds the
            limit is refused before its nodes and weights are computed. The
            limit does not apply to classical execution or to other backends.
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
        qsp_route: Default `"auto"`. Polynomial transformation of
            `"qsp_block_encoding"`, given like `lchs_kernel` as a name, a dict
            `{"implementation": name, "parameters": {...}}` or a
            `ProviderConfig`. `"auto"` compares the routes below by their CX
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
            synthesized. Other backends accept only `"auto"`.
        moment_failure_probability: Default `0.05`, strictly between 0 and
            1. Total probability that the reported sampling radius of a
            `NormSquared`, `QuadraticForm` or `NormalizedExpectation` output
            measured with `shots` fails, split equally over its measurement
            settings. The radius is Hoeffding's bound for the independent
            shots of each setting, from the range of that setting's per-shot
            score, the square of the recovery factor times the sum of the
            group's Pauli coefficient magnitudes, summed over the settings.
            A normalized expectation without padded coordinates uses the
            shots that pass the success test; with padding it is the ratio
            of the observable and the physical-mass setting, whose radius
            exists only when the mass interval is positive. The radius
            covers the shot noise of the finite circuit's moment, not its
            approximation error, which `algorithmic_approximation` reports
            separately. Exact readout (`shots=None`) has no sampling error.
            When an `accuracy` request chose the shots (`sampling_shots`),
            the request's `1 - confidence` is this probability instead, and
            a non-default value that differs from it is refused.
            Sampled pairs use `pair_failure_probability` instead.
        observable_estimator: Default `"coherent"`, which prepares the
            solution coherently and reads the requested output as above.
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
            (`pair_source_time`). The mean signed scores estimate the physical
            quadratic form and norm squared, with separate bounds on evolution
            and model bias. Each branch evolves by a product formula
            (`hamiltonian_evolution_backend` `"trotter"` or
            `"trotter_error_budgeted"`, `lcu_select_implementation`
            `"auto"`, `"multiplexor"` or `"structured"`, an A given as a
            Pauli sum or by Pauli parts) or, with `"qsp_block_encoding"`, by
            a polynomial from the finite phase bank of `pair_qsp_bank` on
            encodings of the parts of A: Pauli or band parts, supplied
            encodings of Hermitian parts, or a general A given as one
            supplied encoding. It needs quantum execution and `shots`
            divisible by `pair_repeats`, or an `accuracy` request
            (`sampling_shots`). The result's `pair_estimate` gives the raw estimates and
            their covering radii, which only the complete set of rows
            provides. For a Run that has measured only some rows,
            `nwqlib.algorithms.lchs.sampled_pair_progress(run)`
            gives the raw sums of the measured rows. Zero elapsed time or
            zero inputs keep the exact no-evolution readout.
        pair_repeats: Default `1`. Shots m of each sampled row. The Plan
            draws `shots/m` independent rows. Repeating a row does not
            narrow the confidence radius, which counts rows only.
        pair_failure_probability: Default `0.05`, strictly between 0 and 1.
            Total probability that a reported covering radius fails, split
            equally over the quantities the output needs (the quadratic form
            and the norm squared of a `NormalizedExpectation`). For the
            complete planned set of rows, the covering radius targets the
            physical quantity and adds both evolution and model bias to the
            Hoeffding radius. Model bias includes the kernel, k quadrature,
            source quadrature and applicable finite random-bit effects. An
            unknown bias leaves the covering radius unavailable while the
            sampling radius remains available. The bounds are numerical
            estimates without a complete enclosure of arithmetic errors.
            An `accuracy` request supplies `1 - confidence` instead, and a
            non-default value that differs from it is refused.
        pair_kernel: Default `"discrete"`, which draws the nodes of the
            Method's kernel-quadrature pair with probability proportional to
            their coefficient magnitudes. `"continuous"` draws k from a
            truncated Cauchy distribution of the kernel itself, without a k
            quadrature. It needs `lchs_kernel` `"apel_berntson"` or
            `"low_somma_f2"` (given with one of their allowed quadrature
            rules, which it does not use), and gives half of
            `approximation_tolerance` to the kernel integral and half to the
            endpoint guard of the distribution.
        pair_source_time: Default `"discrete"`. How sampled pairs treat the
            time integral of a constant source. `"discrete"` draws one of
            the `duhamel_nodes` Gauss–Legendre nodes, with the same
            source-quadrature error bound as the coherent route.
            `"continuous"` draws the source's elapsed time t in [0, T] with
            density proportional to exp(s*t), for the signed shift s of the
            solution, so no source-quadrature error arises. It has no effect
            on a problem without a source.
        pair_control_copies: Default `1`. Number r of copies of the
            sampled branch value that control the product-formula rotations,
            using r - 1 extra clean qubits and 2(r - 1) CX gates.
        pair_step_buckets: Default `1`. Number of |k| buckets of
            `"trotter_error_budgeted"` step selection for sampled pairs.
            With one bucket every row runs the smallest common step count
            whose product-formula bound meets 0.1 times
            `approximation_tolerance`. With more buckets, larger |k| runs
            more steps under the same bound, and a row runs the larger count
            of its two branches. The bound is a weighted sum over the nodes
            when the source and the kernel are both discrete, and otherwise
            the largest bucket bound, because a continuous draw gives no
            exact bucket probabilities. A schedule above a set
            `max_trotter_steps` is refused.
        pair_product_formula: Default `"plain"`. How sampled pairs evolve
            each branch and combine the rows. `"plain"` runs the product
            formula of `trotter_order` with the steps above. The other
            choices reduce the product-formula bias and multiply each row's
            scores by a factor that the sampling radius includes.
            `"richardson"` and `"ptsc"` bound the remaining product-formula
            bias under the premises given for each below. `"mpf"` cancels
            the leading h**2 term but has no finite-step bias bound, so it
            reports raw estimates and sampling radii only.
            `"richardson"` draws, for each row, one of the two Strang step
            counts s and 2s with probabilities near 1/5 and 4/5 and weights
            -1/3 and 4/3, which cancels the h**2 bias term (Watson and
            Watkins, arXiv:2408.14385v2, Sec. 5). It needs
            `trotter_order=2`. With `"trotter"`, s is `trotter_steps` and
            2s must not exceed a set `max_trotter_steps`; the published bias bound
            needs s above 2A, A = 4(2L - 1)T_e(lambda_H + K lambda_L) for L
            distinct Pauli labels, the last elapsed time T_e, the
            coefficient one-norms lambda_H, lambda_L of H and L and the
            largest kernel node K, and is unknown otherwise. With
            `"trotter_error_budgeted"`, planning chooses a sufficient s from
            `pair_pf_allowance`. `"ptsc"` adds to every first-order (Lie)
            step a random Pauli correction of the paired Taylor-series
            compensation (Zeng, Sun, Jiang and Zhao, arXiv:2212.04566v2,
            Sec. IV.2), independently for the two branches of a row, and
            multiplies the row by the two branches' normalizations. It
            needs `trotter_order=1`, `"trotter_error_budgeted"` and the
            three options below; planning chooses the common step count
            from `pair_mu_cap` and the cutoff from `pair_pf_allowance`.
            `"mpf"` draws one of two symmetric product formulas of the same
            depth with weights 3/2 and -1/2 (probabilities 3/4 and 1/4),
            which cancels the leading h**2 term (constant-depth
            multi-product formula, arXiv:2607.12225v1, Sec. IV). It needs
            `trotter_order=2` and `"trotter"` with an even `trotter_steps`
            r, run as r/2 double steps. Its finite-step bias has no bound,
            so it reports raw estimates and sampling radii without a
            covering radius or a normalized value. Every choice other than
            `"plain"` needs `pair_step_buckets=1`.
        pair_pf_allowance: Default `None`. Positive allowance on the
            product-formula bias of the norm squared ||u(T)||**2, in the
            solution's unit squared; the quadratic form's allowance is the
            sum of the observable's |Pauli coefficients| times it. Required
            by `"richardson"` with `"trotter_error_budgeted"` and by
            `"ptsc"`, and refused with other choices.
        pair_mu_cap: Default `None`. Bound mu_cap > 1 on the normalization of
            each `"ptsc"` branch. The common step count r is the smallest
            with r > 2 L and r**3 log(mu_cap) >= (e + 2/9)(2 L)**4, L =
            T_e(lambda_H + K lambda_L), so a larger mu_cap allows a smaller
            r. The score range, and with it the sampling radius for a fixed
            number of independent rows, grows as mu_cap**2. For a fixed
            failure probability delta of the quantity, bound W on the pair
            importance weights and `pair_repeats` R, the number of
            independent rows sufficient for a sampling radius eps grows as
            mu_cap**4, B >= 2 W**4 mu_cap**4 log(2/delta)/eps**2, and so
            does the shot count B R. This sufficient count follows from the
            score range, not from an observed variance, and because mu_cap
            also changes r and the corrections, it gives no ratio of total
            run times. Required by `"ptsc"` and refused otherwise.
        pair_taylor_cutoff: Default `None`. Largest Taylor order, at least
            3, that `"ptsc"` may use for its corrections. Planning takes
            the smallest order from 3 up whose truncation and finite
            probability-table bound meets `pair_pf_allowance`, and refuses
            when none up to this value does. Required by `"ptsc"` and
            refused otherwise.
        pair_qsp_bank: Default `"deterministic"`. Phase bank of sampled
            pairs with `hamiltonian_evolution_backend="qsp_block_encoding"`,
            given like `lchs_kernel` as a name, a dict or a `ProviderConfig`.
            Planning covers every branch of the support: with alpha_L and
            alpha_H the normalizations of the encodings that a branch calls
            for L and H, a branch with node k and elapsed time t has
            `a = |k| alpha_L + alpha_H`, and the largest product t*a over all
            nodes and elapsed times, b_max, is known before any row is drawn.
            A Pauli or band part is encoded after removing its identity
            term, and a supplied encoding of a Hermitian part is used as
            given. A general A given as one supplied encoding with
            normalization alpha_A is called through two encodings of its
            Hermitian parts L and H, each with normalization alpha_A, so
            `a = (|k| + 1) alpha_A`; every call of either encoding calls A
            once and its adjoint once.
            `buckets=K` (default 1) gives the K bank times
            `b_max * 2**(b - K + 1)`, b = 0..K-1. A row uses the smallest bank
            time t_b at or above both of its branches' t*a. Each branch runs
            a Jacobi–Anger Laurent polynomial of time t_b, fitted once at
            planning (the two-sided target of `qsp_route="laurent"`), on its
            encoding scaled by `t*a/t_b`, so the polynomial approximates that
            branch's own `exp(-i t (k L_c + H_c))`, with L_c and H_c the
            operators those encodings encode. The polynomial's recovery
            factor zeta_b
            multiplies the row's scores by zeta_b**2. Shots whose QSP qubits do not all
            return to zero score zero, so the mean still divides by every
            shot. `"deterministic"` fits one polynomial per bank time, with
            recovered error at most `LCHS_QSP_EPSILON_FRACTION` (0.1) times
            `approximation_tolerance`. `"stochastic"` keeps the Laurent terms
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
            degree does not by itself make it cheaper. Other parameters: `margin`
            (default 1/16), the polynomial's distance below 1 on the unit
            circle. The member arrays of the whole bank and the first
            transform arrays of every member's phase fit are checked against
            `max_bytes` before any row is drawn or phase synthesized. The polynomial error estimates
            are numerical, without a complete enclosure of coefficient or
            phase arithmetic. Refused with other backends.

    Raises:
        ValueError: If the kernel-quadrature pair is not allowed or a kernel
            or quadrature parameter is invalid, if both or neither of
            `trotter_steps` and `trotter_synthesis_tolerance` are set, if
            `trotter_synthesis_tolerance` is set without the `"trotter"`
            backend, if `"trotter_error_budgeted"` gets `trotter_steps`
            other than 1, if `trotter_steps` exceeds a set `max_trotter_steps`, if
            `trotter_order=1` is set without a product-formula backend, if
            `"sampled_pairs"` is combined with a backend other than
            `"trotter"`, `"trotter_error_budgeted"` or
            `"qsp_block_encoding"`, with `"qsp_block_encoding"` and a
            product-formula option, a SELECT choice or a `qsp_route`, a
            `"branch_controlled"` SELECT, a continuous
            kernel other than `"apel_berntson"` or `"low_somma_f2"`, or
            several step buckets without `"trotter_error_budgeted"`, if a
            `pair_*` option differs from its default with
            `observable_estimator="coherent"`, or if a `pair_product_formula`
            recipe gets a `trotter_order`, backend, step count or
            `pair_step_buckets` it is not built on, lacks a recipe option it
            needs or gets one it does not use, or if `qsp_route` names an
            unknown route or parameter or a level its route does not have, or
            is set with a backend other than `"qsp_block_encoding"`, or if
            `pair_qsp_bank` names an unknown bank, parameter or value or is
            set off the sampled QSP route.
        TypeError: If `lchs_kernel`, `k_quadrature`, `qsp_route` or
            `pair_qsp_bank` is not a name, a dict or a `ProviderConfig`.
    """

    schema_version: Literal[6] = 6
    approximation_tolerance: Annotated[Real, Field(gt=0, lt=1)] = .01
    lchs_kernel: Any = Field(default_factory=lambda: ProviderConfig(implementation="near_optimal_eq7"))
    k_quadrature: Any = Field(default_factory=lambda: ProviderConfig(implementation="composite_gauss"))
    hamiltonian_evolution_backend: Literal[
        "auto", "dense_exact", "trotter", "trotter_error_budgeted", "qsp_block_encoding"
    ] = "auto"
    lcu_select_implementation: Literal["auto", "multiplexor", "structured", "branch_controlled"] = "auto"
    dense_control_route: Literal["gatewise", "whole_matrix", "auto"] = "auto"
    trotter_steps: PositiveInt | None = 1
    trotter_synthesis_tolerance: Annotated[Real, Field(gt=0)] | None = None
    trotter_order: Literal[1, 2] = 2
    max_trotter_steps: PositiveInt | None = None
    duhamel_nodes: PositiveInt = 8
    source_drop_allowance: Annotated[Real, Field(ge=0)] | None = None
    make_l_psd: StrictBool = True
    psd_tolerance: Annotated[Real, Field(gt=0)] = DEFAULT_PSD_TOLERANCE
    initial_state_preparation: Literal["direct", "mps_circuit"] = "direct"
    initial_state_mps_max_bond_dim: PositiveInt | None = None
    initial_state_mps_threshold: Annotated[Real, Field(gt=0)] = 1e-14
    initial_state_mps_num_layers: PositiveInt = 2
    lcu_state_preparation: Literal["direct", "mps_circuit", "analytic"] = "direct"
    lcu_mps_max_bond_dim: PositiveInt | None = None
    lcu_mps_threshold: Annotated[Real, Field(gt=0)] = 1e-14
    mps_num_layers: PositiveInt = 2
    max_dense_select_slots: PositiveInt = 4_096
    max_bytes: PositiveInt | None = None
    max_svd_work: PositiveInt | None = None
    max_spectral_work: PositiveInt | None = None
    max_quadrature_work: PositiveInt | None = None
    max_select_work: PositiveInt | None = None
    max_readout_work: PositiveInt | None = None
    qsp_route: Any = Field(default_factory=lambda: ProviderConfig(implementation="auto"))
    moment_failure_probability: Annotated[Real, Field(gt=0, lt=1)] = .05
    observable_estimator: Literal["coherent", "sampled_pairs"] = "coherent"
    pair_repeats: PositiveInt = 1
    pair_failure_probability: Annotated[Real, Field(gt=0, lt=1)] = .05
    pair_kernel: Literal["discrete", "continuous"] = "discrete"
    pair_source_time: Literal["discrete", "continuous"] = "discrete"
    pair_control_copies: PositiveInt = 1
    pair_step_buckets: PositiveInt = 1
    pair_product_formula: Literal["plain", "richardson", "ptsc", "mpf"] = "plain"
    pair_pf_allowance: Annotated[Real, Field(gt=0)] | None = None
    pair_mu_cap: Annotated[Real, Field(gt=1)] | None = None
    pair_taylor_cutoff: Annotated[PositiveInt, Field(ge=3)] | None = None
    pair_qsp_bank: Any = Field(default_factory=lambda: ProviderConfig(implementation="deterministic"))
    _ignored_trotter_steps: int | None = PrivateAttr(default=None)

    descriptor: ClassVar = DESCRIPTOR
    result_type: ClassVar = LCHSAnalysis

    @field_validator("lchs_kernel", "k_quadrature", "qsp_route", "pair_qsp_bank", mode="before")
    @classmethod
    def _provider(cls, value):
        """Accept a registry name, a {"implementation", "parameters"} dict or a ProviderConfig."""
        if isinstance(value, str):
            return ProviderConfig(implementation=value)
        if isinstance(value, dict):
            return ProviderConfig(**value)
        if not isinstance(value, ProviderConfig):
            raise TypeError("LCHS provider choice must be a name or ProviderConfig")
        return value

    @field_serializer("lchs_kernel", "k_quadrature", "qsp_route", "pair_qsp_bank")
    def _provider_record(self, value):
        """Serialize a provider request as its explicit implementation and parameters."""
        return value.to_dict()

    @field_validator("trotter_order", mode="before")
    @classmethod
    def _order(cls, value):
        """Require an integer order before the Literal check, so a float 2.0 or a bool fails."""
        from nwqlib._validation import integer
        return integer(value, "trotter_order", 1)

    @model_validator(mode="after")
    def _choices(self):
        """Reject configurations whose choices conflict, before any problem is seen.

        The kernel-quadrature pair must be registered and allowed, and
        an active trotter_steps may not exceed a set max_trotter_steps. Both product-formula
        backends accept Lie order 1 and Suzuki order 2. ``trotter_error_budgeted``
        selects each node's step count from the bound of the chosen order,
        CSTWZ (Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020)
        Prop. 9, Eq. (120) for order 1 or Prop. 10, Eq. (121) for order 2.
        auto, dense_exact and qsp_block_encoding apply no product formula,
        so they accept only the default order 2 (``auto`` resolves to one of
        the other two at planning, never to a product formula).

        Exactly one of trotter_steps and trotter_synthesis_tolerance is set,
        so a fixed step count and the count that a synthesis tolerance
        selects cannot disagree. The tolerance selects the step count of the
        fixed ``trotter`` backend only. ``trotter_error_budgeted`` selects
        every node's step count from its allowance and never reads
        trotter_steps, so any value other than the default 1 is refused
        rather than ignored. An explicit 1 equals the default, which the
        saved record cannot tell apart, so it stays accepted. ``auto``,
        ``dense_exact`` and ``qsp_block_encoding`` never read trotter_steps. They accept
        a value other than 1, normalize it to 1 before the content hash is
        computed, and keep the supplied value only for ``plan``'s warning.
        """
        enumeration = [config.implementation == "auto" for config in (self.lchs_kernel, self.k_quadrature)]
        if any(enumeration):
            if not all(enumeration) or self.lchs_kernel.parameters or self.k_quadrature.parameters:
                raise ValueError("provider enumeration needs lchs_kernel='auto' and k_quadrature='auto' together, "
                                 "without parameters; a concrete kernel or quadrature keeps its pair fixed")
            if self.hamiltonian_evolution_backend not in ("auto", "qsp_block_encoding"):
                raise ValueError("lchs_kernel='auto' compares the providers by the coherent route's rows; use "
                                 "hamiltonian_evolution_backend 'auto' or 'qsp_block_encoding', or a concrete pair")
        else:
            resolve_lchs_provider_requests(self.lchs_kernel, self.k_quadrature)
        self._qsp_route_choices()
        if self.lcu_state_preparation == "analytic":
            pair = (self.lchs_kernel.implementation, self.k_quadrature.implementation)
            if self.hamiltonian_evolution_backend not in ("auto", "qsp_block_encoding"):
                raise ValueError("lcu_state_preparation='analytic' prepares the kernel factor of the coherent LCU; "
                                 "use hamiltonian_evolution_backend 'auto' or 'qsp_block_encoding'")
            if pair != ("low_somma_f2", "symmetric_uniform_trapezoid"):
                raise ValueError("lcu_state_preparation='analytic' prepares the low_somma_f2 kernel on its "
                                 f"symmetric_uniform_trapezoid grid, not {pair[0]!r} with {pair[1]!r}")
        if (self.trotter_steps is None) == (self.trotter_synthesis_tolerance is None):
            raise ValueError(
                "set exactly one of trotter_steps and trotter_synthesis_tolerance. "
                "The tolerance selects the periodic Strang step count, so pass trotter_steps=None with it"
            )
        if self.trotter_synthesis_tolerance is not None and self.hamiltonian_evolution_backend != "trotter":
            raise ValueError("trotter_synthesis_tolerance selects the step count of hamiltonian_evolution_backend='trotter'")
        if self.hamiltonian_evolution_backend == "trotter_error_budgeted" and self.trotter_steps != 1:
            raise ValueError(
                "hamiltonian_evolution_backend='trotter_error_budgeted' selects its own step count for each node "
                "from its allowance and does not use trotter_steps. Leave trotter_steps at its default, "
                "or use hamiltonian_evolution_backend='trotter' for a fixed step count"
            )
        if self.hamiltonian_evolution_backend in ("auto", "dense_exact", "qsp_block_encoding") and self.trotter_steps != 1:
            self._ignored_trotter_steps = self.trotter_steps
            object.__setattr__(self, "trotter_steps", 1)
        if self.trotter_steps is not None and exceeds(self.trotter_steps, self.max_trotter_steps):
            raise ValueError(f"trotter_steps={self.trotter_steps} exceeds max_trotter_steps={self.max_trotter_steps}")
        if self.trotter_order != 2 and self.hamiltonian_evolution_backend not in ("trotter", "trotter_error_budgeted"):
            raise ValueError(
                "trotter_order=1 requires hamiltonian_evolution_backend 'trotter' or "
                "'trotter_error_budgeted', because auto, dense_exact and qsp_block_encoding use no product formula"
            )
        pairs = dict(pair_repeats=1, pair_failure_probability=.05, pair_kernel="discrete", pair_source_time="discrete",
                     pair_control_copies=1, pair_step_buckets=1, pair_product_formula="plain", pair_pf_allowance=None,
                     pair_mu_cap=None, pair_taylor_cutoff=None,
                     pair_qsp_bank=ProviderConfig(implementation="deterministic"))
        if self.observable_estimator == "coherent":
            changed = [name for name, default in pairs.items() if getattr(self, name) != default]
            if changed:
                raise ValueError(f"{', '.join(changed)} apply to observable_estimator='sampled_pairs' only")
            return self
        if self.moment_failure_probability != .05:
            raise ValueError("moment_failure_probability applies to observable_estimator='coherent'; sampled pairs "
                             "use pair_failure_probability")
        if self.hamiltonian_evolution_backend == "qsp_block_encoding":
            return self._sampled_qsp_choices()
        if self.hamiltonian_evolution_backend not in ("trotter", "trotter_error_budgeted"):
            raise ValueError("observable_estimator='sampled_pairs' evolves each branch by a Pauli product formula or a "
                             "finite QSP bank; use hamiltonian_evolution_backend 'trotter', "
                             "'trotter_error_budgeted' or 'qsp_block_encoding'")
        if self.pair_qsp_bank != ProviderConfig(implementation="deterministic"):
            raise ValueError("pair_qsp_bank chooses the phase bank of sampled pairs with "
                             "hamiltonian_evolution_backend='qsp_block_encoding'")
        if self.lcu_select_implementation not in ("auto", "multiplexor", "structured"):
            raise ValueError("sampled pairs build the multiplexed or structured pair SELECT; use "
                             "lcu_select_implementation 'auto', 'multiplexor' or 'structured'")
        if self.pair_kernel == "continuous" and self.lchs_kernel.implementation not in ("apel_berntson",
                                                                                       "low_somma_f2"):
            raise ValueError("pair_kernel='continuous' needs lchs_kernel 'apel_berntson' or 'low_somma_f2', whose "
                             "importance weights have a bound on the truncated Cauchy distribution")
        if self.pair_step_buckets > 1 and self.hamiltonian_evolution_backend != "trotter_error_budgeted":
            raise ValueError("pair_step_buckets > 1 selects bucket step counts, which needs "
                             "hamiltonian_evolution_backend='trotter_error_budgeted'")
        self._recipe_choices()
        return self

    def _sampled_qsp_choices(self):
        """Reject a sampled QSP Method with a product-formula or coherent option that its route does not read.

        Each branch runs a finite Laurent bank on the centered children
        (`sampled_qsp`), so the product-formula recipe, step buckets, pair
        SELECT and its control copies, and the coherent transformation of
        ``qsp_route`` are not used; they keep their defaults rather than
        being ignored. ``pair_qsp_bank`` must name a valid bank. Provider
        enumeration and a non-direct coefficient PREP are refused at
        planning, with ApplicabilityError, so that ``compare`` keeps the row.
        """
        from .sampled_qsp import bank_options
        unread = dict(pair_product_formula="plain", pair_step_buckets=1, pair_control_copies=1, pair_pf_allowance=None,
                      pair_mu_cap=None, pair_taylor_cutoff=None, lcu_select_implementation="auto")
        changed = [name for name, default in unread.items() if getattr(self, name) != default]
        if self.qsp_route != ProviderConfig(implementation="auto"):
            changed.append("qsp_route")
        if changed:
            raise ValueError(f"{', '.join(changed)} {'does' if len(changed) == 1 else 'do'} not apply to sampled "
                             "pairs with hamiltonian_evolution_backend='qsp_block_encoding', whose branches run the "
                             "finite QSP bank of pair_qsp_bank")
        if self.pair_kernel == "continuous" and self.lchs_kernel.implementation not in ("apel_berntson",
                                                                                       "low_somma_f2"):
            raise ValueError("pair_kernel='continuous' needs lchs_kernel 'apel_berntson' or 'low_somma_f2', whose "
                             "importance weights have a bound on the truncated Cauchy distribution")
        bank_options(self)
        return self

    def _qsp_route_choices(self):
        """Reject a qsp_route that names an unknown route or parameter, or that another backend would ignore.

        The route names and their finite levels come from
        ``coherent_selection`` (`SHARED_TARGET_CAPS`, `GENERAL_MARGINS` and
        the refused routes of `NOT_ADMITTED_ROUTES`). A refused or
        inapplicable route is accepted here and refused at planning with its
        reason, before any phase is synthesized.
        """
        route, parameters = self.qsp_route.implementation, dict(self.qsp_route.parameters)
        if route == "auto" and not parameters:
            return
        if self.hamiltonian_evolution_backend != "qsp_block_encoding":
            raise ValueError("qsp_route chooses the coherent transformation of "
                             "hamiltonian_evolution_backend='qsp_block_encoding'")
        from .coherent_selection import GENERAL_MARGINS, NOT_ADMITTED_ROUTES, SHARED_TARGET_CAPS
        levels = dict(shared_oaa=SHARED_TARGET_CAPS, shared_raw=SHARED_TARGET_CAPS, laurent=GENERAL_MARGINS,
                      gqet=GENERAL_MARGINS, imaginary_gqet=GENERAL_MARGINS, commuting_product=(),
                      **dict.fromkeys(NOT_ADMITTED_ROUTES, ()))
        if route != "auto" and route not in levels:
            raise ValueError(f"qsp_route implementation {route!r} is not 'auto' or one of {sorted(levels)}")
        unknown = sorted(set(parameters) - {"level", "node", "joint"})
        if unknown:
            raise ValueError(f"qsp_route takes the parameters level, node and joint, not {unknown}")
        if "level" in parameters and (route == "auto" or parameters["level"] not in levels[route]):
            raise ValueError(f"qsp_route level {parameters['level']!r} needs a named route with that level; "
                             f"{route!r} has {levels.get(route, ())}")
        if parameters.get("joint", "separate_parts") not in ("separate_parts", "single_a", "convex_contraction"):
            raise ValueError("qsp_route joint must be 'separate_parts', 'single_a' or 'convex_contraction'")
        if parameters.get("node", "table") not in ("rotation_table", "comparator", "twos_complement_pauli",
                                                    "signed_magnitude_pauli", "table"):
            raise ValueError("qsp_route node must be 'rotation_table', 'comparator', 'twos_complement_pauli', "
                             "'signed_magnitude_pauli' or 'table'")

    def _recipe_choices(self):
        """Reject a pair_product_formula recipe whose order, backend, steps or recipe options do not match it.

        Each recipe fixes its base formula: Richardson extrapolates one
        Strang (order 2) word, PTSC compensates the Lie (order 1) step and
        the MPF members are order-2 words. A recipe's step count comes from
        fixed ``trotter_steps`` or from its own allowance, never from the
        plain buckets, and an option a recipe does not read is refused
        rather than ignored.
        """
        recipe, budgeted = self.pair_product_formula, self.hamiltonian_evolution_backend == "trotter_error_budgeted"
        options = dict(pair_pf_allowance=self.pair_pf_allowance is not None, pair_mu_cap=self.pair_mu_cap is not None,
                       pair_taylor_cutoff=self.pair_taylor_cutoff is not None)
        needed = {"plain": (), "richardson": ("pair_pf_allowance",) if budgeted else (),
                  "ptsc": ("pair_pf_allowance", "pair_mu_cap", "pair_taylor_cutoff"), "mpf": ()}[recipe]
        extra = [name for name, given in options.items() if given and name not in needed]
        missing = [name for name in needed if not options[name]]
        if extra:
            raise ValueError(f"{', '.join(extra)} {'does' if len(extra) == 1 else 'do'} not apply to "
                             f"pair_product_formula={recipe!r}"
                             + (" with fixed trotter_steps" if recipe == "richardson" and not budgeted else ""))
        if missing:
            raise ValueError(f"pair_product_formula={recipe!r} with this backend needs {', '.join(missing)}")
        if recipe == "plain":
            return
        if self.pair_step_buckets != 1:
            raise ValueError(f"pair_product_formula={recipe!r} selects its own step counts; leave pair_step_buckets=1")
        order = 1 if recipe == "ptsc" else 2
        if self.trotter_order != order:
            raise ValueError(f"pair_product_formula={recipe!r} is built on the order-{order} product formula; set "
                             f"trotter_order={order}")
        if recipe == "ptsc" and not budgeted:
            raise ValueError("pair_product_formula='ptsc' chooses its step count from pair_mu_cap and its cutoff from "
                             "pair_pf_allowance; use hamiltonian_evolution_backend='trotter_error_budgeted'")
        if recipe in ("richardson", "mpf") and self.trotter_steps is None:
            raise ValueError(f"pair_product_formula={recipe!r} needs trotter_steps; trotter_synthesis_tolerance "
                             "chooses periodic-stencil steps only")
        if recipe == "mpf" and (budgeted or self.trotter_steps % 2):
            raise ValueError("pair_product_formula='mpf' has no finite-step bias bound to choose steps from; use "
                             "hamiltonian_evolution_backend='trotter' with an even trotter_steps r, run as r/2 "
                             "double steps")
        if recipe == "richardson" and not budgeted and exceeds(2 * self.trotter_steps, self.max_trotter_steps):
            raise ValueError(f"pair_product_formula='richardson' also runs 2*trotter_steps = {2 * self.trotter_steps} "
                             f"steps, which exceeds max_trotter_steps={self.max_trotter_steps}")

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
        if (self.observable_estimator == "sampled_pairs" and self.pair_failure_probability != .05
                and 1.0-self.pair_failure_probability != accuracy.confidence):
            raise ApplicabilityError(f"pair_failure_probability={self.pair_failure_probability!r} disagrees with the "
                                     f"accuracy confidence {accuracy.confidence!r}; the request sets the confidence "
                                     "of its radius, so leave pair_failure_probability at its default or set it to "
                                     "1 - confidence")
        if self.moment_failure_probability != .05 and 1.0-self.moment_failure_probability != accuracy.confidence:
            raise ApplicabilityError(f"moment_failure_probability={self.moment_failure_probability!r} disagrees with "
                                     f"the accuracy confidence {accuracy.confidence!r}; the request sets the "
                                     "confidence of its radius, so leave moment_failure_probability at its default "
                                     "or set it to 1 - confidence")
        output.frame(problem)

    def sampling_shots(self, problem, *, output, accuracy, execution, shots):
        """Return the shots per measurement setting that meet an absolute sampling tolerance.

        `plan`, `solve` and `compare` call it when they receive
        `accuracy=Accuracy(absolute_tolerance=epsilon, component="sampling",
        confidence=c)` in place of `shots`. It chooses the route as planning
        does and gives each candidate route the count that makes its
        sampling radius at most epsilon with probability at least c, for the
        moment of the finite circuit that the route builds:
        `N = ceil(2 Gamma**4 A**2 log(2 G/delta)/epsilon**2)` shots on each of
        the G measurement settings of a `QuadraticForm`, with A the sum over
        the settings of their Pauli coefficient magnitudes (the identity
        coefficient counted once) and `delta = 1 - c`, or
        `N = ceil(Gamma**4 log(2/delta)/(2 epsilon**2))` for `NormSquared`.
        Gamma is the route's physical recovery factor, the factor that
        multiplies the measured success amplitudes. These counts invert the
        radius that the result reports (see `moment_failure_probability`).
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

        With `observable_estimator="sampled_pairs"` the count is `B * m`
        shots, m = `pair_repeats`, for
        `B = ceil(2 (H/epsilon)**2 log(2/delta))` independent rows, where H
        bounds the physical score of one row over every row the sampling law
        can draw: the square of the largest branch importance weight, times
        the largest |o_l|/p_l of the observable terms for a `QuadraticForm`,
        times the largest factor of the branch evolution (1 for a plain
        product formula, the largest Richardson or MPF member weight,
        `pair_mu_cap**2` for PTSC, the largest zeta_b**2 of the QSP bank).
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
                `moment_failure_probability` or `pair_failure_probability`
                other than `1 - confidence`, a choice that sampled pairs
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
        in sampled_plan.plan_sampled for observable_estimator="sampled_pairs".
        With ``accuracy``, ``shots`` is the count `sampling_shots` returned.
        Planning reads the choice that `sampling_shots` made in the same
        planning call, checks that count and records the request in
        ``Plan.selection_accuracy``.

        ``dense_exact`` and ``qsp_block_encoding`` apply no product formula
        and record 0 steps. An ignored trotter_steps is canonicalized to 1
        during Method validation, leaving the Method's content hash
        unchanged. Planning warns with the supplied value, which the Method keeps privately.
        The warning skips frames inside the nwqlib package,
        so it names the caller's line whether the call came through
        ``plan``, ``solve`` or ``compare``. Constructing, copying or loading
        a Method or Plan does not warn.
        """
        if not isinstance(problem, LinearDynamics):
            raise ApplicabilityError("LCHS requires LinearDynamics(A, initial_state, time, source)")
        if output.kind not in self.descriptor.output_families:
            raise ApplicabilityError("LCHS supports a physical solution, state, norm, observable or samples")
        if accuracy is not None:
            self._accuracy_request(problem, output=output, accuracy=accuracy, execution=execution)
        if self.observable_estimator == "sampled_pairs":
            from .sampled_plan import plan_sampled
            return plan_sampled(self, problem, output=output, execution=execution, shots=shots, rng=rng,
                                accuracy=accuracy)
        if isinstance(output, (Solution, StateVector)) and shots is not None:
            raise ApplicabilityError("counts do not recover a complex vector; omit shots or select a scalar/Samples output")
        if isinstance(output, Samples) and (shots is None or execution != "quantum"):
            raise ApplicabilityError("Samples requires quantum execution and finite shots")
        if execution == "classical" and shots is not None:
            raise ApplicabilityError("classical LCHS acquires deterministic numerical data; shots belong to quantum readout")
        if execution == "classical" and self.hamiltonian_evolution_backend == "qsp_block_encoding":
            raise ApplicabilityError("QSP LCHS has no selected classical acquisition; use quantum execution")
        if self._ignored_trotter_steps is not None:
            import os
            import warnings
            import nwqlib
            warnings.warn(
                f"trotter_steps={self._ignored_trotter_steps} has no effect with hamiltonian_evolution_backend="
                f"'{self.hamiltonian_evolution_backend}', which applies no product formula. trotter_steps "
                "applies only to the 'trotter' backend, so this Plan selects the same approximation and "
                "circuit as the default trotter_steps=1. To use "
                f"{self._ignored_trotter_steps} steps, set hamiltonian_evolution_backend='trotter' and "
                f"trotter_steps={self._ignored_trotter_steps} together. Revising only the backend keeps "
                "trotter_steps=1",
                UserWarning, skip_file_prefixes=(os.path.dirname(nwqlib.__file__) + os.sep,),
            )
        from .selection import plan_lchs
        return plan_lchs(self, problem, output=output, execution=execution, shots=shots, rng=rng, accuracy=accuracy)

    def analyze(self, plan, data, *, settings):
        """Reconstruct the requested output from the execution data (analysis.analyze).

        LCHS fixes its recovery scale and coordinates in the Plan, so there
        are no reconstruction settings to choose after execution.
        """
        if settings:
            raise ValueError("LCHS has no post-hoc reconstruction settings")
        from .analysis import analyze
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
