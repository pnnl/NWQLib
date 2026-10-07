"""QHD one-hot compiler from SymPy objectives to Pauli evolution blocks."""

from __future__ import annotations

import math
from fractions import Fraction
from math import isfinite
from typing import Any, Mapping

import numpy as np

from nwqlib.algorithms.qhd.grid import OneHotGrid
from nwqlib.algorithms.qhd.kinetic import KineticCompiler
from nwqlib.algorithms.qhd.objective import ObjectiveDecomposer, uncentered_objective
from nwqlib.algorithms.qhd.potential import (
    PotentialCompiler,
    coerce_real_scalar,
    _evaluate_objective,
)
from nwqlib.core.records import FrozenArray
from nwqlib.problems.records import Optimization
from nwqlib.algorithms.qhd.schedules import step_weights
from nwqlib.algorithms.qhd.validation import (
    _kinetic_range_error,
    _lambdify_objective,
    _normal_product,
    _normal_quotient,
    _normal_range,
    _underflows,
)
from nwqlib.subroutines.hamiltonian_evolution import (
    PauliEvolutionBlock,
)

# Engineering allowances of the support-table evaluation, registered in
# docs/ENGINEERING_CONSTANTS.md ("Budgets and mechanical bounds"). H0 is an
# explicit allowance for scalar bookkeeping, ndarray headers, iterators and
# fixed sort stacks on the checked 64-bit CPython/NumPy stack, not a universal
# interpreter or process-RSS theorem (the wrapper_fixed_bytes part of each support
# table's fixed headers H_S, passed to support_chunk_size, and the table metadata
# and serialization headers of method._support_table_bytes).
# OBJECT_HEADER_BYTES is the allowance of 256 bytes per inventoried array or
# wrapper object of the evaluator (support_workspace). Requalify them when the
# NumPy printer, NumPy or the Python runtime changes.
H0 = 65536
OBJECT_HEADER_BYTES = 256


def support_workspace(nodes):
    """Return ``(bytes per entry, fixed bytes)`` of the evaluator workspace of one support table.

    For a slab of b entries of a support S whose NumPy-printed callable has
    ``N_S = nodes`` expression nodes (``objective.node_count``), the
    evaluator's live numerical intermediates are charged
    ``W_S(b) = 8*b*(N_S + 2)`` bytes. This is an engineering allowance, not a
    derived bound. Its premise is that the printer's evaluation holds at most
    one slab-sized float64 temporary per expression node, plus the output and
    one broadcast input. A primitive with complex intermediates, input-dependent
    or hidden storage lies outside the premise. The fixed part counts
    ``OBJECT_HEADER_BYTES`` for each of those ``N_S + 2`` array objects,
    which enters ``H_S`` of the slab charge (``support_chunk_size``) with H0.
    The measurement that qualifies the allowance is registered in
    docs/ENGINEERING_CONSTANTS.md ("Budgets and mechanical bounds").
    """
    return 8 * (nodes + 2), OBJECT_HEADER_BYTES * (nodes + 2)


def support_chunk_size(entries, support_size, remaining, evaluator_bytes_per_entry,
                       evaluator_fixed_bytes, wrapper_fixed_bytes):
    """Return the largest flattened chunk of a support table whose evaluation fits ``remaining`` bytes.

    For support size s and a chunk of b entries the slab charge is
    ``B_slab,S(b) = W_S(b) + (8s+64)b + H_S``: the evaluator's intermediates
    ``W_S(b)`` (``support_workspace``), ``8(s+4)b`` comprising ``8(s+3)b``
    for s coordinate arrays, an index vector and quotient/remainder scratch
    and an additional 8b allowance, ``32b`` for
    complex conversion, Boolean validation and float64 conversion, and the
    fixed headers ``H_S``, here ``evaluator_fixed_bytes + wrapper_fixed_bytes``.
    With the linear evaluator law ``W_S(b) = A_S b`` of ``support_workspace``
    the largest sufficient chunk is ``min(entries, floor((R - H_S)/(A_S + 8s + 64)))``
    for the remaining allowance ``R = max_bytes - B_held - 8E - 8dK - H_tables``
    after the final payloads, the centered coordinates and the metadata are
    reserved.

    Raises:
        ValueError: ``remaining`` cannot admit one entry.
    """
    fixed = evaluator_fixed_bytes + wrapper_fixed_bytes
    rate = evaluator_bytes_per_entry + 8*support_size + 64
    if remaining < fixed + rate:
        raise ValueError("support evaluation cannot admit one entry")
    return min(entries, (remaining-fixed)//rate)


def chunk_coordinates(centered_axes, start, stop):
    """Return the coordinate arguments of the flattened C-order entries ``start`` to ``stop - 1``.

    Entry e of a support table with axis lengths ``K_1, ..., K_s`` has the
    mixed-radix digits ``i_j = (e // prod_(l > j) K_l) % K_j``, so the last
    support variable varies fastest and the first is most significant, the
    order in which the table is stored. The entry count is admitted against
    intp before this runs.
    """
    index = np.arange(start, stop, dtype=np.intp)
    arguments = []
    stride = 1
    for axis in reversed(centered_axes):
        arguments.append(axis[(index // stride) % len(axis)])
        stride *= len(axis)
    return arguments[::-1]


def user_point(grid, centers, support, expression):
    """Return ``locate(indices)`` for a centered support table: its expression and grid point in the problem's coordinates.

    The expression is ``objective.uncentered_objective`` of the centered
    support expression and the point is ``grid.grid_value`` at the entry's
    support indices, in the coordinates of the problem being tabulated (a
    refinement level's search model has its own). Both are formed only when
    an entry is refused.
    """
    def locate(indices):
        point = tuple(grid.grid_value(j, i) for j, i in zip(support, indices, strict=True))
        # The grid may name its variables by string. Shift the expression's own symbols of those names.
        symbols = {str(s): s for s in expression.free_symbols}
        pairs = [(symbols[str(grid.variables[j])], centers[j]) for j in support if str(grid.variables[j]) in symbols]
        return uncentered_objective(expression, [v for v, _ in pairs], [c for _, c in pairs]), point

    return locate


def evaluate_support(evaluator, centered_axes, chunk, *, expression, support, name="QHD objective", locate=None):
    """Return the flattened C-order float64 table of one lambdified support expression.

    ``centered_axes[j]`` holds the centered coordinates ``x - m`` of the
    j-th support variable. Each chunk of at most ``chunk`` entries
    (``support_chunk_size``) evaluates the NumPy callable once on its
    coordinate arguments (``chunk_coordinates``), and a scalar output is a
    constant broadcast over the chunk. Each output must convert to finite
    complex128 with imaginary part exactly zero, then to float64. A warning
    from an unused ``Piecewise`` branch alone does not invalidate a finite
    selected result. An exception, nonfinite selected output or nonzero
    imaginary component does. The first bad entry is mapped back to its
    support indices and reported under ``name``, the function whose table
    this is. ``locate(indices)``, when given, returns that entry's
    expression and point in the problem's coordinates, formed only for the
    message (``evaluate_chunk``).

    Array evaluations may differ from scalar evaluations by an ulp or more,
    because SIMD elementary functions and expression broadcasting can follow
    different numerical paths. No universal evaluation-error bound exists for
    arbitrary lambdified expressions. The stored array is the selected data,
    and every table-consuming grid evaluation must index it. Repeating the
    vectorized function on a one-element array does not recover the original
    slab's bits.

    An expression whose callable cannot be evaluated or converted on arrays
    (an ArithmeticError, ValueError, TypeError or NameError from the chunk
    call or its complex128 conversion) keeps its admitted scalar
    functionality: that chunk is evaluated entry by entry by
    ``potential._evaluate_objective``, which raises ValueError for an entry
    that is not finite and real. Any other exception propagates. A nonfinite
    or complex array result is never re-evaluated as a scalar.
    """
    shape = tuple(len(axis) for axis in centered_axes)
    entries = math.prod(shape)
    output = np.empty(entries, dtype=np.float64)
    for start in range(0, entries, chunk):
        stop = min(entries, start + chunk)
        evaluate_chunk(evaluator, centered_axes, start, stop, output[start:stop], expression=expression,
                       support=support, name=name, locate=locate)
    return output


def evaluate_chunk(evaluator, axes, start, stop, out, *, expression, support, name="QHD objective", locate=None):
    """Write into ``out`` the values of one lambdified expression at the flattened C-order entries ``start`` to ``stop - 1``.

    ``axes[j]`` holds the coordinate arguments of the j-th variable of the
    expression, and entry e of the Cartesian product has the mixed-radix
    digits of ``chunk_coordinates``. The callable is evaluated once on the
    chunk's coordinate arguments, and a scalar output is a constant broadcast
    over the chunk. Each output must convert to finite complex128 with
    imaginary part exactly zero, then to float64, which ``out`` receives. An
    ArithmeticError, ValueError, TypeError or NameError from the chunk call or
    its conversion evaluates the chunk entry by entry
    (``potential._evaluate_objective``), which raises for an entry that is not
    finite and real. A nonfinite or complex array result is
    never re-evaluated as a scalar: its first bad entry is mapped back to its
    indices and coordinates and reported under ``name``, the function the
    expression belongs to. When ``axes`` are centered coordinates,
    ``locate(indices)`` returns the refused entry's expression and point in
    the problem's coordinates, and it is called only when a value is refused. ``evaluate_support`` evaluates a
    support table with it, chunk by chunk.
    """
    shape = tuple(len(axis) for axis in axes)

    def refusal_context(position):
        # The expression and point of a refused entry, in the problem's coordinates when locate is given.
        indices = tuple(int(i) for i in np.unravel_index(position, shape))
        if locate is None:
            shown, point = expression, tuple(float(axis[i]) for axis, i in zip(axes, indices, strict=True))
        else:
            shown, point = locate(indices)
        return f"{name} {shown} on support {support} at grid point {point}"

    arguments = chunk_coordinates(axes, start, stop)
    try:
        with np.errstate(all="ignore"):
            raw = evaluator(*arguments)
        values = np.broadcast_to(np.asarray(raw, dtype=np.complex128), (stop - start,))
    except (ArithmeticError, ValueError, TypeError, NameError):
        for position in range(start, stop):
            point = tuple(float(argument[position - start]) for argument in arguments)
            try:
                out[position - start] = _evaluate_objective(evaluator, point, expression=expression,
                                                            support=support, name=name)
            except ValueError as error:
                if locate is None:
                    raise
                raise ValueError(f"{refusal_context(position)} failed finite-real evaluation: "
                                 f"{error.__cause__ or error}") from error
        return out
    good = np.isfinite(values.real) & np.isfinite(values.imag) & (values.imag == 0)
    if not good.all():
        local = int(np.argmin(good))
        value = np.broadcast_to(np.asarray(raw), (stop - start,))[local]
        context = refusal_context(start + local)
        try:
            coerce_real_scalar(value, context=name)
        except ValueError as error:
            raise ValueError(f"{context} failed finite-real evaluation: {error}") from error
        raise ValueError(f"{context} failed finite-real evaluation: {values[local]!r} is not finite and real")
    out[...] = values.real
    return out


class QHDCompiler:
    """Compile a QHD-admitted problem into Pauli-evolution blocks.

    Implements the one-hot finite-difference/product-formula realization of
    Quantum Hamiltonian Descent: Hamiltonian Eq. (1) and finite differences
    Eqs. (F.7)/(F.9) of Leng et al., arXiv:2303.01471v1. This compiler uses
    one-hot registers, unlike that paper's radix-2/Fourier realization in
    Algorithm 1.

    The one-hot potential follows the occupation-operator encoding of Wu et
    al., arXiv:2605.12066v1, Sec. IV.A, Eqs. (9)-(10). Each multivariate
    support term becomes a sum of number-operator products weighted by its
    grid values, which extends their product rule for products of
    single-variable functions to a general support term. Step k uses the
    weights that the Method's coefficient rule selects
    (``schedules.step_weights``), either the point values at its midpoint
    ``(k + 1/2)*dt`` or the step averages of the interval integrals over
    ``[k dt, (k + 1) dt]``. Eq. (E.3) and Algorithm 1 of Leng et al. use the
    left endpoint ``t_j = j*dt``. With a time-dependent schedule the left endpoint
    limits the step to first order in dt, while the midpoint keeps the
    symmetric second-order step second order when the schedule is smooth. A
    time-independent schedule makes all three choices coincide.

    Identity components of the Hamiltonian (the kinetic stencil diagonal, the
    constant objective and the identity part of each number-operator
    product) are not emitted as gates. Their phases are summed in the
    ``dropped_global_phase`` metadata, which the native builder and the
    ``ir_product`` reference restore once as the physical phase.
    """

    def __init__(
        self,
        problem: Optimization,
        options,
        decomposer: ObjectiveDecomposer,
        chunks: Mapping[tuple[int, ...], int],
    ) -> None:
        """Fix the step weights and evaluate every support table once.

        ``decomposer`` is the expansion of ``problem.objective`` about
        ``expansion_centers(problem.bounds)`` whose support expressions
        ``QHD._admit_symbolic_work`` counted, so the tables evaluate the
        expressions that were admitted, and ``chunks`` maps each support to
        the entries per evaluation chunk that the same admission chose
        (``support_chunk_size``). Every table is evaluated here, so a
        nonfinite or complex objective value rejects before any block is
        compiled or pruned.
        """
        self.problem = problem
        self.options = options
        self.grid = self._build_grid()
        self.decomposer = decomposer
        self.chunks = dict(chunks)
        self.step_weights = step_weights(
            options.schedule, options.coefficient_rule, options.total_time, options.num_steps
        )
        self._grid_value_cache: dict[tuple[int, ...], FrozenArray] = {}
        self._validate_objective_values_are_real()
        self.kinetic_compiler = KineticCompiler(rotation_threshold=self.options.rotation_threshold)
        self.potential_compiler = PotentialCompiler(
            rotation_threshold=self.options.rotation_threshold
        )
        self._metadata: dict[str, Any] = {}
        self._reset_metadata()

    def build_step_pauli_ir(self) -> tuple[tuple[PauliEvolutionBlock, ...], ...]:
        """Return the compiled Pauli-evolution blocks of every schedule step.

        Blocks alternate potential and kinetic terms with the selected step
        weights, in forward splitting or symmetric half steps. Scalar
        identity contributions are recorded as dropped global phase.

        Blocks apply in list order. First order applies ``V(dt)`` and then
        ``K(dt)``, the operator order of Eq. (E.3) in Leng et al.,
        arXiv:2303.01471v1. Second order applies ``V(dt/2) K(dt) V(dt/2)``,
        where ``K(dt)`` is itself the
        symmetric parity split built by ``KineticCompiler``. Under the
        integrated rule a step's weights are its whole-step averages
        ``A/dt`` and ``B/dt`` (``schedules.step_weights``), so each potential
        half applies ``B/2`` and the odd, even and odd link layers apply
        ``A/2``, ``A`` and ``A/2``. The step is then exactly the symmetric
        (Strang) split of the first Magnus exponent ``-i (A K + B V)``, whose
        splitting error starts at third order in the exponents. The
        time-ordering error of the step (the Magnus terms from ``Omega_2`` on,
        ``schedules.step_weights``) and the product-formula error therefore
        stay separate, as ``method._error_model`` declares them. Potential
        halves ``B_1`` (applied first) and ``B_2`` with ``B_1 + B_2 = B``, such
        as separate first-half and second-half integrals, would add the
        second-order BCH term ``(A (B_1 - B_2)/2) [V, K]`` of
        ``exp(-i B_2 V) exp(-i A K) exp(-i B_1 V)``. That term is of the same
        order ``dt**3`` as ``Omega_2`` and partly cancels it, so such a step is
        a different method that mixes the two error sources.
        """

        dt = self.options.total_time / self.options.num_steps
        step_groups: list[tuple[PauliEvolutionBlock, ...]] = []
        self._reset_metadata()

        # The constant is normalized once to the binary64 value that the
        # reconstruction stores (QHDReconstruction.constant), so each step's
        # coefficient is the rounded product fl(b_k c) of two stored binary64
        # inputs, the model under which method._compiled_phase_allowance
        # bounds the phase ledger.
        constant_objective = coerce_real_scalar(self.decomposer.constant, context="constant objective")
        for time, kinetic_weight, potential_weight in self.step_weights:
            blocks: list[PauliEvolutionBlock] = []

            self._record_kinetic_global_phase(dt, kinetic_weight)

            if constant_objective != 0:
                self._record_constant_phase(dt, potential_weight, constant_objective)

            if self.options.trotter_order == 1:
                blocks.extend(self._compile_potential_with_phase_record(dt, potential_weight, time))
                blocks.extend(
                    self.kinetic_compiler.compile(
                        self.grid,
                        dt=dt,
                        kinetic_weight=kinetic_weight,
                        time=time,
                    )
                )
            else:
                blocks.extend(
                    self._compile_potential_with_phase_record(dt / 2.0, potential_weight, time)
                )
                blocks.extend(
                    self.kinetic_compiler.compile(
                        self.grid,
                        dt=dt,
                        kinetic_weight=kinetic_weight,
                        time=time,
                        trotter_order=2,
                    )
                )
                blocks.extend(
                    self._compile_potential_with_phase_record(dt / 2.0, potential_weight, time)
                )

            step_groups.append(tuple(blocks))
        return tuple(step_groups)

    def metadata(self) -> dict[str, Any]:
        """Return the phase and angle-threshold ledgers of the last compiled schedule.

        ``dropped_global_phase`` holds ``phase_angle``, the total identity
        phase in radians that the native circuit and the ``ir_product``
        reference restore, and the per-source totals
        of angles (``per_source_total_angles``) and of Hamiltonian
        coefficients (``per_source_coefficient_totals``) for the sources
        ``kinetic_diagonal``, ``objective_constant`` and
        ``projector_identity``. ``rotation_threshold_truncation`` holds the
        number of nonzero rotations removed by the threshold, the sum of
        their absolute angles in radians and that sum exactly, as an integer
        count of ``2**-1074`` (``dropped_angle_units``). ``range_omissions``
        holds the projector occurrences and identity events that the
        lower-range selection omitted, with the exact sums of their traceless
        and identity charges (``_compile_potential_with_phase_record``,
        ``_record_constant_phase``).
        """

        return {
            **self._metadata,
            "rotation_threshold_truncation": {
                "dropped_nonzero_contribution_count": (
                    self.kinetic_compiler._dropped_nonzero_contribution_count
                    + self.potential_compiler._dropped_nonzero_contribution_count
                ),
                "dropped_absolute_angle_sum": (
                    self.kinetic_compiler._dropped_absolute_angle_sum
                    + self.potential_compiler._dropped_absolute_angle_sum
                ),
                "dropped_angle_units": (
                    self.kinetic_compiler._dropped_angle_units
                    + self.potential_compiler._dropped_angle_units
                ),
            },
            "range_omissions": {
                "projectors": self.potential_compiler._omitted_count,
                "projector_charge": self.potential_compiler._omitted_traceless,
                "identity_events": self.potential_compiler._omitted_count + self._omitted_constant_events,
                "identity_charge": self.potential_compiler._omitted_identity + self._omitted_constant_charge,
            },
        }

    def _build_grid(self) -> OneHotGrid:
        return OneHotGrid(
            variables=self.problem.variables,
            bounds=self.problem.bounds,
            num_grid_points=self.options.num_grid_points,
            include_boundary_points=self.options.include_boundary_points,
            boundary=self.options.boundary,
        )

    def _reset_metadata(self) -> None:
        """Empty both ledgers, so a repeated compilation does not add to an earlier one."""
        self.kinetic_compiler._reset_dropped_rotations()
        self.potential_compiler._reset_dropped_rotations()
        # Neumaier accumulators (running sum, correction) of the angle totals.
        self._phase_sums: dict[str, tuple[float, float]] = {}
        # Constant events omitted below the normal range and the exact sum of |dt b c| over them.
        self._omitted_constant_events = 0
        self._omitted_constant_charge = Fraction(0)
        self._metadata = {
            "dropped_global_phase": {
                "phase_angle": 0.0,
                "per_source_total_angles": {},
                "per_source_coefficient_totals": {},
            }
        }

    def _compile_potential_with_phase_record(
        self,
        dt: float,
        potential_weight: float,
        time: float,
    ) -> tuple[PauliEvolutionBlock, ...]:
        """Compile one potential factor ``exp(-i dt V)`` and record its identity phase.

        ``dt`` is the full step for first order and half the step for each
        of the two second-order potential factors. One selection
        (``potential.PotentialCompiler.select_occurrences``) keeps or omits
        each entry's projector and its identity event together. A product of
        s number operators ``prod (I - Z)/2`` has identity component
        ``I/2**s``, and the native projector evolves only the traceless
        remainder, so a kept entry records ``-fl(t q)`` with
        ``q = fl(b v)/2**s``. An omitted entry records a zero event, which
        keeps the contribution count of ``method._phase_contributions``, and
        its identity charge ``2**-s |t b v|`` enters the phase ledger.
        """
        occurrences = self.potential_compiler.select_occurrences(
            self._potential_grid_values(), num_grid_points=self.grid.num_grid_points, dt=dt,
            potential_weight=potential_weight)
        for occurrence in occurrences:
            self._record_global_phase(
                coefficient=occurrence.identity,
                evolution_time=dt,
                source="projector_identity",
            )
        return self.potential_compiler.compile_selected(occurrences, self.grid, dt=dt, time=time)

    def _record_constant_phase(self, dt: float, potential_weight: float, constant: float) -> None:
        """Record the objective constant's phase ``-fl(dt fl(b c))`` of one step, or omit a tiny one.

        When ``fl(b c)`` or its product with dt would be nonzero and below
        ``2**-1022`` (``validation._underflows``), the step records a zero
        event, which keeps the contribution count, and the exact
        ``|dt b c|`` joins the omitted identity charge of the phase ledger,
        since ``|exp(-i a) - 1| <= |a|``, as for the identity of an omitted
        projector (``potential.PotentialCompiler.select_occurrences``). No
        ``rotation_pruning`` term accompanies a scalar identity.
        """
        coefficient = potential_weight * constant
        omitted = _underflows(coefficient, "the objective constant's coefficient b c",
                              lambda: Fraction(potential_weight) * Fraction(constant), zero=potential_weight == 0)
        if not omitted:
            omitted = _underflows(-dt * coefficient, "the objective constant's phase contribution -dt b c",
                                  lambda: Fraction(dt) * Fraction(coefficient), zero=coefficient == 0)
        if omitted:
            self._omitted_constant_events += 1
            self._omitted_constant_charge += abs(Fraction(dt) * Fraction(potential_weight) * Fraction(constant))
            coefficient = 0.0
        self._record_global_phase(
            coefficient=coefficient,
            evolution_time=dt,
            source="objective_constant",
        )

    def _record_kinetic_global_phase(
        self,
        dt: float,
        kinetic_weight: float,
    ) -> None:
        """Record the kinetic stencil diagonal of one step as phase.

        The diagonal ``kinetic_weight/h**2`` of each variable is the same on
        every one-hot state, hence a phase. This holds on both boundaries,
        because the chain and the periodic cycle of ``OneHotGrid.links`` give
        every point the same diagonal ``1/h**2`` with that grid's spacing h.
        Leng et al. drop the corresponding constant as a global phase in the
        sentence that introduces Eq. (F.14) of arXiv:2303.01471v1, and Liu et al.,
        arXiv:2607.16996v1, write their periodic one-hot kinetic term,
        Eq. (12), without it and omit it from the circuit as an overall phase
        of the one-hot sector (Sec. V A 1). Here it enters the physical phase
        ledger instead, so that the circuit's global phase matches the
        evolution of the finite model, which the error ledger compares with
        its phase included (``resources.circuit_resources``). Each
        ``a/h**2`` is a normal binary64 number (``validation._normal_range``),
        and their compensated sum of positive terms is finite. The built-in
        ``sum`` below is compensated for floats from Python 3.12, the minimum
        that ``pyproject.toml`` requires. A kinetic term outside that range is refused (``validation._kinetic_range_error``).
        """
        try:
            coefficient = sum(
                _normal_quotient(kinetic_weight, self.grid.spacing(var_index) ** 2,
                                 f"the kinetic diagonal a/h**2 of variable {var_index}")
                for var_index in range(self.grid.num_variables)
            )
            _normal_range(coefficient, "the kinetic diagonal sum_j a/h_j**2")
            _normal_product(-dt, coefficient, "the kinetic_diagonal phase contribution -t q")
        except ValueError as error:
            raise _kinetic_range_error(error) from error
        self._record_global_phase(
            coefficient=coefficient,
            evolution_time=dt,
            source="kinetic_diagonal",
        )

    def _record_global_phase(
        self,
        *,
        coefficient: float,
        evolution_time: float,
        source: str,
    ) -> None:
        """Add the phase of ``exp(-i * evolution_time * coefficient * I)``.

        Totals are kept overall and per source so a report can show where the
        restored physical phase comes from. The recorded contribution is
        ``x_j = -fl(t_j q_j)`` for the incoming binary64 coefficient q_j and
        duration t_j. The angle totals accumulate the x_j with Neumaier's
        compensation (``_compensated_add``) and store the rounded corrected
        sum as an ordinary float. The coefficient totals are a plain sum that
        no phase bound uses.

        Rounding of a compensated total. Assume round-to-nearest binary64
        arithmetic, no overflow and no inexact underflow. Planning supplies
        these premises, since every coefficient and ``x_j`` passes the range
        rule (``validation._normal_range``) and ``_compensated_add`` raises
        when an accumulator stops being finite. The residual formed
        with the larger-magnitude operand first is then exactly the rounding
        error of each addition. For m >= 1 contributions with ``(m - 1) u < 1``,
        u = 2**-53, let ``X = sum_j x_j``, ``A = sum_j |x_j|`` and
        ``g_m = gamma_(m-1)``, ``gamma_r = r u/(1 - r u)``. The absolute
        residuals total at most ``g_m A``, summing them with ordinary
        additions adds at most ``g_m**2 A``, and the final addition adds at
        most ``u |X|`` plus u times the correction's error, so the stored
        total differs from X by at most
        ``u |X| + (1 + u) g_m**2 A <= beta_m A``, with
        ``beta_m = u + (1 + u) gamma_(m-1)**2``. The algorithm is Neumaier's
        (Z. Angew. Math. Mech. 54 (1974) 39-51, doi:10.1002/zamm.19740540106).
        Its result equals that of the cascaded summation of Ogita, Rump and
        Oishi (SIAM J. Sci. Comput. 26 (2005) 1955-1988,
        doi:10.1137/030601818, Algorithm 4.1), and their Proposition 4.5
        proves the smaller bound ``u |X| + gamma_(m-1)**2 A`` for the
        equivalent Algorithm 4.4 (``method._compiled_phase_allowance``).
        An ordinary running
        sum has the allowance ``gamma_(m-1) A``, first order in ``m u``,
        while beta_m is u plus a term of second order in ``m u``. For a long
        schedule the compensated allowance is therefore smaller by about a
        factor m while m stays well below ``u**(-1/2)``, about 9.5e7, which
        matters because ``method._admit_compiled_phase`` requires the
        allowance of a kept state to stay below pi. The total uses all M events
        (``method._phase_contributions``), including both
        second-order potential halves, and a source total uses its own event
        count, N (``num_steps``) for the objective constant's one event per step. Cancellation
        does not reduce the required sum of absolute contributions. Comparing with
        the intended Hamiltonian phase adds each contribution's formation
        allowance. For the constant, whose coefficient ``fl(b_k c)`` and
        angle are two rounded products, this gives
        ``[eta_2 + (1 + eta_2) beta_N] S_C`` with ``eta_2 = 2u + u**2`` and
        ``S_C = |c| sum_k |dt b_k|``, which is ``3u S_C`` to first order.
        ``method._compiled_phase_allowance`` evaluates the bound of the total
        for the Plan. The empty ledger has zero phase and zero allowance.
        """
        phase_angle = _normal_product(-evolution_time, coefficient, f"the {source} phase contribution -t q")
        dropped = self._metadata["dropped_global_phase"]
        # The source first, so an overflow names the source that caused it.
        angles = dropped["per_source_total_angles"]
        angles[source] = self._compensated_add(source, phase_angle)
        dropped["phase_angle"] = self._compensated_add("total", phase_angle)
        coefficients = dropped["per_source_coefficient_totals"]
        coefficients[source] = coefficients.get(source, 0.0) + coefficient

    def _compensated_add(self, name: str, value: float) -> float:
        """Add ``value`` to the Neumaier accumulator ``name`` and return its rounded corrected total.

        The accumulator keeps a running sum s and a correction c, both zero at
        first. For ``t = fl(s + value)`` the residual is
        ``fl(fl(s - t) + value)`` when ``|s| >= |value|`` and
        ``fl(fl(value - t) + s)`` otherwise, which is exactly ``s + value - t``
        under the assumptions of ``_record_global_phase``. The correction adds
        the residual, and the total is ``fl(t + c)``. The running sum, the
        residual, the correction and the corrected total must stay finite, and
        an overflow raises here, since cancellation later could not make an
        overflowed partial sum valid.
        """
        high, correction = self._phase_sums.get(name, (0.0, 0.0))
        total = high + value
        if abs(high) >= abs(value):
            residual = (high - total) + value
        else:
            residual = (value - total) + high
        correction += residual
        result = total + correction
        if not all(isfinite(x) for x in (total, residual, correction, result)):
            raise ValueError(
                "the compiled physical phase of the QHD circuit exceeds the binary64 range"
                + ("" if name == "total" else f" in its {name} part")
                + ". The native circuit and the ir_product reference apply it before any readout"
                + (". The objective constant does not affect the optimization, so removing it from the "
                   "objective removes its part" if name == "objective_constant" else "")
                + ". The classical schrodinger or split_step flavor without keep_state reads the probabilities "
                "without this phase")
        self._phase_sums[name] = (total, correction)
        return result

    def _validate_objective_values_are_real(self) -> None:
        """Evaluate the constant and every support table, raising on a nonfinite or complex value.

        The stored binary64 constant and table values are the exact data of
        the phase and angle ledgers and need only be finite. A contribution
        formed from them that would fall below the normal binary64 range is
        omitted and charged where it is formed (``_record_constant_phase``,
        ``potential.PotentialCompiler.select_occurrences``).
        """
        coerce_real_scalar(self.decomposer.constant, context="constant objective")
        self._potential_grid_values()

    def _potential_grid_values(self) -> dict[tuple[int, ...], FrozenArray]:
        """Return the cached grid table of each support group, keyed by sorted variable indices."""
        return {
            support: self._objective_grid_values(support)
            for support in self.decomposer.support_expressions
        }

    def _objective_grid_values(self, support: tuple[int, ...]) -> FrozenArray:
        """Evaluate one support expression on its ``K**len(support)`` grid tuples as one float64 array.

        This is the one producer of the support tables. The entries are in C
        order, first support variable most significant, the order that
        ``method.objective_at`` reads. The support expression, which the
        decomposer expanded in the centered coordinates, is evaluated at
        ``x - m``: the one-dimensional centered coordinate vectors are formed
        once with that binary64 subtraction, and the table is evaluated in
        flattened C-order chunks of the admitted size (``evaluate_support``,
        ``support_chunk_size``). A nonfinite or complex entry raises with the
        support indices and coordinates of the first one. The finished table
        is frozen (``core.records.FrozenArray``) before the next support is
        evaluated, and ``SupportValues`` shares that array. The expression of
        a support is fixed for the compiler, so the cache key is the support
        and each table is evaluated once per compiler however many steps
        reuse it.
        """
        cached = self._grid_value_cache.get(support)
        if cached is not None:
            return cached

        expression = self.decomposer.support_expressions[support]
        variables = [self.grid.variables[index] for index in support]
        evaluator = _lambdify_objective(variables, expression)
        k = self.grid.num_grid_points
        centered_axes = [
            np.array([self.grid.grid_value(var_index, grid_index) - self.decomposer.centers[var_index]
                      for grid_index in range(k)], dtype=np.float64)
            for var_index in support
        ]
        table = evaluate_support(evaluator, centered_axes, self.chunks[support], expression=expression,
                                 support=tuple(support), locate=user_point(self.grid, self.decomposer.centers, support, expression))
        self._grid_value_cache[support] = FrozenArray(table)
        return self._grid_value_cache[support]
