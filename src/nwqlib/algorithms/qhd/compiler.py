"""QHD support tables and step rows of a problem, evaluated once for planning."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from nwqlib.algorithms.qhd.grid import OneHotGrid
from nwqlib.algorithms.qhd.objective import ObjectiveDecomposer, uncentered_objective
from nwqlib.algorithms.qhd.potential import (
    coerce_real_scalar,
    _evaluate_objective,
)
from nwqlib.core.records import FrozenArray
from nwqlib.problems.records import Optimization
from nwqlib.algorithms.qhd.schedules import step_weights
from nwqlib.algorithms.qhd.validation import _lambdify_objective

# Engineering allowances of the support-table evaluation, registered in
# docs/ENGINEERING_CONSTANTS.md ("Budgets and mechanical bounds"). H0 is an
# explicit allowance for scalar bookkeeping, ndarray headers, iterators and
# fixed sort stacks on the checked 64-bit CPython/NumPy stack, not a universal
# interpreter or process-RSS theorem (part of each support table's fixed headers
# H_S in slab_bytes, and the table metadata and serialization headers of
# method._support_table_bytes).
# OBJECT_HEADER_BYTES is the allowance of 256 bytes per inventoried array or
# wrapper object of the evaluator (support_workspace). Requalify them when the
# NumPy printer, NumPy or the Python runtime changes.
H0 = 65536
OBJECT_HEADER_BYTES = 256


def support_workspace(nodes):
    """Return ``(bytes per entry, fixed bytes)`` of the evaluator workspace of one support table.

    For a slab of b entries of a support S whose NumPy-printed callable has
    ``N_S = nodes`` expression nodes (``objective.node_count``), the
    evaluator's numerical intermediates held in memory are counted as
    ``W_S(b) = 8*b*(N_S + 2)`` bytes. This is an engineering allowance, not a
    derived bound. It assumes that the printer's evaluation holds at most
    one slab-sized float64 temporary per expression node, plus the output and
    one broadcast input. A primitive with complex intermediates, input-dependent
    or hidden storage lies outside this assumption. The fixed part counts
    ``OBJECT_HEADER_BYTES`` for each of those ``N_S + 2`` array objects,
    which enters ``H_S`` of the slab byte count (``slab_bytes``) with H0.
    The measurement that qualifies the allowance is registered in
    docs/ENGINEERING_CONSTANTS.md ("Budgets and mechanical bounds").
    """
    return 8 * (nodes + 2), OBJECT_HEADER_BYTES * (nodes + 2)


# Entries of a support table that one array evaluation covers. NumPy's array
# and scalar evaluations of an elementary function can differ in their last
# bits (evaluate_support), so the block boundaries belong to the stored table
# and do not follow max_bytes. Registered in docs/ENGINEERING_CONSTANTS.md.
SUPPORT_BLOCK_ENTRIES = 2**16


def support_chunk_size(entries):
    """Return the entries of a support table evaluated together, ``min(entries, SUPPORT_BLOCK_ENTRIES)``.

    The block is fixed, so the stored table of a support does not depend on
    ``max_bytes``. ``slab_bytes`` counts the bytes of evaluating one block.
    """
    return min(entries, SUPPORT_BLOCK_ENTRIES)


def slab_bytes(chunk, support_size, evaluator_bytes_per_entry, evaluator_fixed_bytes):
    """Return ``B_slab,S(b)``, the bytes of evaluating a block of ``b = chunk`` entries of one support table.

    For support size s, ``B_slab,S(b) = W_S(b) + (8s+64)b + H_S``: the
    evaluator's intermediates ``W_S(b)`` (``support_workspace``), ``8(s+4)b``
    comprising ``8(s+3)b`` for s coordinate arrays, an index vector and
    quotient/remainder scratch and an additional 8b allowance, ``32b`` for
    complex conversion, Boolean validation and float64 conversion, and the
    fixed headers ``H_S``, here ``evaluator_fixed_bytes + H0``. Callers add
    the bytes they hold and refuse a Plan whose total exceeds ``max_bytes``.
    """
    return (evaluator_bytes_per_entry + 8 * support_size + 64) * chunk + evaluator_fixed_bytes + H0


def chunk_coordinates(centered_axes, start, stop):
    """Return the coordinate arguments of the flattened C-order entries ``start`` to ``stop - 1``.

    Entry e of a support table with axis lengths ``K_1, ..., K_s`` has the
    mixed-radix digits ``i_j = (e // prod_(l > j) K_l) % K_j``, so the last
    support variable varies fastest and the first is most significant, the
    order in which the table is stored. The entry count is checked against
    the intp range before this runs.
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
    an entry is rejected.
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
    arbitrary lambdified expressions. The stored array is the data that
    planning fixes, and every grid evaluation that uses the table must index
    it. Repeating the vectorized function on a one-element array does not
    recover the original slab's bits.

    An expression whose callable cannot be evaluated or converted on arrays
    (an ArithmeticError, ValueError, TypeError or NameError from the chunk
    call or its complex128 conversion) is still evaluated as scalars, as
    planning accepted it. That chunk is evaluated entry by entry by
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
    finite and real. A nonfinite or complex array result is never
    re-evaluated as a scalar. Its first bad entry is mapped back to its
    indices and coordinates and reported under ``name``, the function the
    expression belongs to. When ``axes`` are centered coordinates,
    ``locate(indices)`` returns the rejected entry's expression and point in
    the problem's coordinates, and it is called only when a value is
    rejected. ``evaluate_support`` evaluates a support table with it, chunk
    by chunk.
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
    """Evaluate the support tables and step rows of a problem that QHD accepts.

    The tables tabulate the support expressions of the objective's expansion
    on the grid of the Hamiltonian of Eq. (1) of Leng et al.,
    arXiv:2303.01471v1, with the finite-grid potential of their Eqs.
    (F.7)/(F.9), in the occupation-operator encoding of Wu et al.,
    arXiv:2605.12066v1, Sec. IV.A (``onehot_schedule``, ``binary``). Step k
    uses the weights of the Method's coefficient rule
    (``schedules.step_weights``), either the point values at its midpoint
    ``(k + 1/2)*dt`` or the step averages of the interval integrals over
    ``[k dt, (k + 1) dt]``. Eq. (E.3) and Algorithm 1 of Leng et al. use the
    left endpoint ``t_j = j*dt``. With a time-dependent schedule the left
    endpoint limits the step to first order in dt, while the midpoint keeps
    the symmetric second-order step second order when the schedule is
    smooth. A time-independent schedule makes all three choices coincide.
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
        expressions whose work was checked, and ``chunks`` maps each support
        to the entries per evaluation chunk whose bytes the same check
        counted (``support_chunk_size``). Every table is evaluated here, so a
        nonfinite or complex objective value raises before any block is
        derived.
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

    def _build_grid(self) -> OneHotGrid:
        return OneHotGrid(
            variables=self.problem.variables,
            bounds=self.problem.bounds,
            num_grid_points=self.options.num_grid_points,
            boundary_points=self.options.boundary_points,
            boundary=self.options.boundary,
        )

    def _validate_objective_values_are_real(self) -> None:
        """Evaluate the constant and every support table, raising on a nonfinite or complex value.

        The stored binary64 constant and table values are the exact data of
        the phase and angle error bounds and need only be finite. A
        contribution formed from them that would fall below the normal
        binary64 range is omitted, and its error is counted where it is
        formed (``onehot_schedule.OneHotSchedule``, ``binary.compile_binary_steps``).
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

        This is the only function that forms the support tables. The entries
        are in C order, first support variable most significant, the order
        that ``method.objective_at`` reads. The support expression, which the
        decomposer expanded in the centered coordinates, is evaluated at
        ``x - m``. The one-dimensional centered coordinate vectors are formed
        once with that binary64 subtraction, and the table is evaluated in
        flattened C-order chunks of the size checked at planning
        (``evaluate_support``, ``support_chunk_size``). A nonfinite or complex
        entry raises with the support indices and coordinates of the first
        one. The finished table is made read-only
        (``core.records.FrozenArray``) before the next support is evaluated,
        and ``SupportValues`` shares that array. The expression of
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
        # Changing -0.0 to +0.0 in this private table keeps FrozenArray to one copy (_support_table_bytes).
        table += 0.0
        self._grid_value_cache[support] = FrozenArray(table)
        return self._grid_value_cache[support]
