"""LCHS choices for the original equation du/dt=-A u+b.

Input and coordinates belong to LinearDynamics. Numerical selection belongs to
this Method, acquisition to Run, and reconstruction to the selected readout.
"""

import json

from nwqlib._limits import DEFAULT_MAX_BYTES

from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, PrivateAttr, StrictBool, field_serializer, field_validator, model_validator

from nwqlib._quantum_readout import PROJECTED_MOMENTS, projected_requirements
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
    recovery, state preparation, evolution and floating-point error. The
    default `dense_exact` backend computes each branch matrix
    `exp(-i t (k_j L + H))` classically and builds one dense SELECT circuit,
    which applies branch j when its address register holds j. The
    product-formula and `qsp_block_encoding` backends build more structured
    circuits, each with its own limit checks.

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
            QSP synthesis, and each node of `trotter_error_budgeted`,
            separately receives 0.1 times this value. Planning refuses a
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
            (Low and Somma, Theorems 2-4).
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
            `signed_binary_uniform`, and `low_somma_f2` with
            `symmetric_uniform_trapezoid` (both component bounds hold, and
            the synthesis stages are not budgeted with them). Building the
            Method refuses any other pair.
        hamiltonian_evolution_backend: Default `"dense_exact"`. How each
            branch `exp(-i t (k_j L + H))` is built. `"dense_exact"` computes
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
            `"auto"`. With `"trotter_error_budgeted"`, `"auto"` gives
            `"multiplexor"`, and `"structured"` is refused, because per-node
            step counts break the affine address structure. With
            `"trotter"`, `"auto"` gives `"structured"` only when an affine
            address structure is eligible, as checked from the k-node
            addresses and the real Pauli coefficients of the generator, and
            its CX count is lower than the multiplexor's, and gives
            `"multiplexor"` otherwise. An explicit `"structured"` without
            that eligibility is refused. A `PeriodicStencil` A accepts
            `"auto"` or `"structured"`.
        dense_control_route: Default `"auto"`. How a construction controls
            a dense unitary on two or more qubits. It applies to a
            `dense_exact` branch on its address bits and to a dense-dilation
            QSP child on the combine qubit of the joint generator.
            `"gatewise"` synthesizes the unitary and lets Qiskit control
            each synthesized gate, `"whole_matrix"` synthesizes the
            controlled matrix, and `"auto"` takes the whole-matrix route for
            one control and the gate-wise route for more. On the
            whole-matrix route a constant-source branch is one unitary, its
            evolution times the matrix of its input preparation. The QSP
            pass that holds the generator is controlled gate-wise on every
            route.
        trotter_steps: Default `1`. Fixed step count of the `"trotter"`
            backend, a positive integer at most `max_trotter_steps`, or
            `None` when `trotter_synthesis_tolerance` chooses it. Exactly one
            of the two is set. `"trotter_error_budgeted"` chooses its own
            counts and requires the default 1. `"dense_exact"` and
            `"qsp_block_encoding"` apply no product formula, so they treat
            any value as 1, and planning warns about a value other than 1.
        trotter_synthesis_tolerance: Default `None`. Positive tolerance on
            the weighted Strang synthesis bound of a `PeriodicStencil` A, in the
            same unit-input norm as `approximation_tolerance`. Planning
            chooses the smallest step count r with ``B/r**2`` at most this
            value, where T is the elapsed time, q is the system-qubit count,
            ``a_j = |k_j| diffusion``, ``p = |potential|`` and
            ``B = T**3 sum_j |c_j| W_q(a_j, p)`` over the quadrature nodes
            and coefficients. The matching-operator commutators give
            ``W_q(a, p) = g a**3/2 + 4 a**2 p/3 + a p**2/3``, with g = 0
            for q = 1, 2 and g = 1 for q >= 3, in the emitted potential,
            even-matching, odd-matching order (Childs et al.,
            doi:10.1103/PhysRevX.11.011020, Proposition 10, Eq. (121),
            or arXiv:1912.08854v3, Proposition 16, Eq. (152)).
            The bound covers the selected finite weighted action on a unit input;
            input scaling and other error components are handled separately. With
            zero potential the ideal synthesis error is zero at q = 1, 2.
            It requires
            `hamiltonian_evolution_backend="trotter"` and
            `trotter_steps=None`, and other inputs refuse it.
        trotter_order: Default `2`. Order of the product formula of the
            `"trotter"` and `"trotter_error_budgeted"` backends, 1 for the
            Lie formula or 2 for the symmetric Suzuki (Strang) formula.
            `"dense_exact"` and `"qsp_block_encoding"` accept only 2.
        max_trotter_steps: Default `100_000`. Upper limit on the
            product-formula step count, fixed or chosen automatically.
        duhamel_nodes: Default `8`. Number of Gauss–Legendre nodes of the
            time integral of a constant source, one panel on [0, T] (ACL
            Eq. (72)). It is never reduced automatically to fit a limit.
        make_l_psd: Default `True`. Rule for an eigenvalue of L below
            `-psd_tolerance*||L||_2`. `True` shifts L by
            `s = -lambda_min + psd_tolerance*||L||_2` and multiplies the
            result by the growth factor `exp(s*T)`. `False` refuses the
            input.
        psd_tolerance: Default `1e-12`, positive. Relative window of the PSD
            check. An eigenvalue of L at or above `-psd_tolerance*||L||_2`
            is accepted without a shift. The window is the scale of the
            eigensolver's rounding (LAPACK Users' Guide, 3rd ed., Sec. 4.7),
            so the decision does not depend on the time unit. It does not
            permit a negative eigenvalue below that window.
        initial_state_preparation: Default `"direct"`, which prepares the
            initial state exactly. `"mps_circuit"` prepares it approximately
            with a layered circuit from a matrix-product-state (MPS)
            compression. Quantum execution with a constant source, or with a
            `PeriodicStencil` A, requires `"direct"`.
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
            `initial_state_preparation`. The circuit error of
            `"mps_circuit"` stays unevaluated until an explicit validation,
            even when the compression discards no weight.
        lcu_mps_max_bond_dim: Default `None`, no limit. Upper limit on the
            kept bond dimension of the coefficient-state compression.
        lcu_mps_threshold: Default `1e-14`, positive. Singular-value
            truncation threshold of the coefficient-state compression.
        mps_num_layers: Default `2`, positive. Number of layers of the
            coefficient-state MPS circuit.
        max_dense_select_slots: Default `4_096`. Upper limit on the padded
            address count, a power of two, of the dense SELECT circuit of
            `"dense_exact"`. Planning refuses a larger construction and never
            changes the grid, Duhamel nodes or tolerance to fit.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Upper
            limit on the known bytes of input and numerical workspace.
        max_svd_work: Default `1e8` (`100_000_000`). Upper limit on the
            counted work of state and coefficient compression, including the
            construction of an `"mps_circuit"` layered circuit.
        max_spectral_work: Default `1e8`. Upper limit on the counted work of
            the eigenvalue check of L for dense A, `d**3` for physical
            dimension d, so the default accepts a dense A of dimension at
            most 464 when L is nonzero. A
            [periodic stencil](../../algorithms/lchs.md#periodic-stencils)
            has an analytically PSD L and no eigenvalue check.
        max_quadrature_work: Default `1e8`. Upper limit on the counted work
            of building the k nodes and weights. For `"composite_gauss"` it
            covers the completed panel search plus the construction of the
            chosen rule, not every scalar probe.
        max_select_work: Default `1e9`. Upper limit on the counted work of
            building the SELECT circuit, including the branch matrices and
            their synthesis for `"dense_exact"` and the Pauli decomposition
            of the product-formula backends. The
            [LCHS guide](../../algorithms/lchs.md#native-methods-and-physical-coordinates)
            gives the sizes that fit this default.
        max_readout_work: Default `2_000_000_000`. Positive integer giving an
            inclusive limit on candidate comparisons when grouping the Pauli
            terms of sampled observables into qubit-wise commuting measurement
            settings, and on character or kernel-input visits for exact scalar
            readout (`shots=None`). Planning checks and records the grouping
            comparisons. Exact scalar readout is checked against the full limit
            before acquisition. Its declared live data and workspace must also
            fit `max_bytes`. A work-limit refusal names `LCHS.max_readout_work`.
        max_qsp_degree: Default `256`. Upper limit on the QSP polynomial
            degree of `"qsp_block_encoding"`.
        max_qsp_evaluations: Default `20_000`. Upper limit on the residual
            evaluations of the QSP phase solver.
        max_admission_steps: Default `1_000_000`. Upper limit on the
            planning work of checking each `Program` that the method builds
            (NWQLib's description of a circuit as named steps). It caps the
            number of stored fields of a Program and the work units of one
            check of it, and summing a Program's resource counts may use up
            to 24 times this value. A larger Program is refused with a
            ValueError that names this field, the refused stage and its
            count. Raising the limit permits a larger check and changes no
            quantum operation
            ([planning work limit](../../development/program_checks.md#planning-work-limit)).

    Raises:
        ValueError: If the kernel-quadrature pair is not allowed or a kernel
            or quadrature parameter is invalid, if both or neither of
            `trotter_steps` and `trotter_synthesis_tolerance` are set, if
            `trotter_synthesis_tolerance` is set without the `"trotter"`
            backend, if `"trotter_error_budgeted"` gets `trotter_steps`
            other than 1, if `trotter_steps` exceeds `max_trotter_steps`, or
            if `trotter_order=1` is set without a product-formula backend.
        TypeError: If `lchs_kernel` or `k_quadrature` is not a name, a dict
            or a `ProviderConfig`.
    """

    schema_version: Literal[6] = 6
    approximation_tolerance: Annotated[Real, Field(gt=0, lt=1)] = .01
    lchs_kernel: Any = Field(default_factory=lambda: ProviderConfig(implementation="near_optimal_eq7"))
    k_quadrature: Any = Field(default_factory=lambda: ProviderConfig(implementation="composite_gauss"))
    hamiltonian_evolution_backend: Literal[
        "dense_exact", "trotter", "trotter_error_budgeted", "qsp_block_encoding"
    ] = "dense_exact"
    lcu_select_implementation: Literal["auto", "multiplexor", "structured", "branch_controlled"] = "auto"
    dense_control_route: Literal["gatewise", "whole_matrix", "auto"] = "auto"
    trotter_steps: PositiveInt | None = 1
    trotter_synthesis_tolerance: Annotated[Real, Field(gt=0)] | None = None
    trotter_order: Literal[1, 2] = 2
    max_trotter_steps: PositiveInt = 100_000
    duhamel_nodes: PositiveInt = 8
    make_l_psd: StrictBool = True
    psd_tolerance: Annotated[Real, Field(gt=0)] = DEFAULT_PSD_TOLERANCE
    initial_state_preparation: Literal["direct", "mps_circuit"] = "direct"
    initial_state_mps_max_bond_dim: PositiveInt | None = None
    initial_state_mps_threshold: Annotated[Real, Field(gt=0)] = 1e-14
    initial_state_mps_num_layers: PositiveInt = 2
    lcu_state_preparation: Literal["direct", "mps_circuit"] = "direct"
    lcu_mps_max_bond_dim: PositiveInt | None = None
    lcu_mps_threshold: Annotated[Real, Field(gt=0)] = 1e-14
    mps_num_layers: PositiveInt = 2
    max_dense_select_slots: PositiveInt = 4_096
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_svd_work: PositiveInt = 100_000_000
    max_spectral_work: PositiveInt = 100_000_000
    max_quadrature_work: PositiveInt = 100_000_000
    max_select_work: PositiveInt = 1_000_000_000
    max_readout_work: PositiveInt = 2_000_000_000
    max_qsp_degree: PositiveInt = 256
    max_qsp_evaluations: PositiveInt = 20_000
    max_admission_steps: PositiveInt = 1_000_000
    _ignored_trotter_steps: int | None = PrivateAttr(default=None)

    descriptor: ClassVar = DESCRIPTOR
    result_type: ClassVar = LCHSAnalysis

    @field_validator("lchs_kernel", "k_quadrature", mode="before")
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

    @field_serializer("lchs_kernel", "k_quadrature")
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
        an active trotter_steps may not exceed max_trotter_steps. Both product-formula
        backends accept Lie order 1 and Suzuki order 2. ``trotter_error_budgeted``
        selects each node's step count from the bound of the chosen order,
        CSTWZ (Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020)
        Prop. 9, Eq. (120) for order 1 or Prop. 10, Eq. (121) for order 2.
        dense_exact and qsp_block_encoding apply no product formula, so they
        accept only the default order 2.

        Exactly one of trotter_steps and trotter_synthesis_tolerance is set,
        so a fixed step count and the count that a synthesis tolerance
        selects cannot disagree. The tolerance selects the step count of the
        fixed ``trotter`` backend only. ``trotter_error_budgeted`` selects
        every node's step count from its allowance and never reads
        trotter_steps, so any value other than the default 1 is refused
        rather than ignored. An explicit 1 equals the default, which the
        stored record cannot tell apart, so it stays admitted. ``dense_exact``
        and ``qsp_block_encoding`` also never read trotter_steps. They admit
        a value other than 1, normalize it to 1 before content identity is
        computed, and keep the supplied value only for ``plan``'s warning.
        """
        resolve_lchs_provider_requests(self.lchs_kernel, self.k_quadrature)
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
        if self.hamiltonian_evolution_backend in ("dense_exact", "qsp_block_encoding") and self.trotter_steps != 1:
            self._ignored_trotter_steps = self.trotter_steps
            object.__setattr__(self, "trotter_steps", 1)
        if self.trotter_steps is not None and self.trotter_steps > self.max_trotter_steps:
            raise ValueError("trotter_steps exceeds max_trotter_steps")
        if self.trotter_order != 2 and self.hamiltonian_evolution_backend not in ("trotter", "trotter_error_budgeted"):
            raise ValueError(
                "trotter_order=1 requires hamiltonian_evolution_backend 'trotter' or "
                "'trotter_error_budgeted', because dense_exact and qsp_block_encoding use no product formula"
            )
        return self

    def plan(self, problem, *, output, execution, shots, rng):
        """Check output, execution and shots against LCHS, then select the Plan once.

        A complex vector needs amplitude readout, so vector outputs reject
        shots. Samples need quantum execution with shots. QSP evolution has no
        classical acquisition. All numerical selection happens in plan_lchs.

        ``dense_exact`` and ``qsp_block_encoding`` apply no product formula
        and record 0 steps. An ignored trotter_steps is canonicalized to 1
        during Method validation, leaving the selected content identity
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
        return plan_lchs(self, problem, output=output, execution=execution, shots=shots, rng=rng)

    def analyze(self, plan, data, *, settings):
        """Reconstruct the requested output from the acquired data (analysis.analyze).

        LCHS fixes its recovery scale and coordinates in the Plan, so there
        are no reconstruction settings to choose after acquisition.
        """
        if settings:
            raise ValueError("LCHS has no post-hoc reconstruction settings")
        from .analysis import analyze
        return analyze(plan, data)

    def reduction_allowance(self, plan, point, *, observation, width, run):
        """Check the Plan's single exact scalar readout against ``max_bytes`` for its declared live data and workspace, and ``max_readout_work`` for character or kernel-input visits, before acquisition.

        Return the full ``max_readout_work`` allowance for that reduction.
        """
        requested = 0
        for item in observation.positions:
            if item.kind != "reduction" or item.reducer != PROJECTED_MOMENTS:
                continue
            size, work = projected_requirements(json.loads(item.parameters), width)
            if size > self.max_bytes:
                raise ValueError(
                    f"LCHS projected reduction needs {size} bytes, max_bytes={self.max_bytes}. "
                    f"Raise LCHS(max_bytes=...) to at least {size} "
                    f"(increase {size - self.max_bytes}). Refused before acquisition.")
            requested += work
        limit = self.max_readout_work
        if requested > limit:
            raise ValueError(
                f"LCHS projected reduction needs {requested} work units. "
                f"LCHS.max_readout_work={limit}. "
                f"Raise LCHS.max_readout_work to at least {requested} "
                f"(increase {requested - limit}). Refused before acquisition.")
        return limit

    def save_archive(self, plan, files):
        """Save the Plan's selected numerical data and block bindings (archive.save)."""
        from .archive import save
        return save(self, plan, files)

    @classmethod
    def load_archive(cls, data, files):
        """Restore a saved Plan and check its selected-data identities (archive.load)."""
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
