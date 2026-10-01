"""Potential-term compiler for the one-hot QHD Hamiltonian."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

import numpy as np

from nwqlib.algorithms.qhd.grid import OneHotGrid
from .validation import DEFAULT_ROTATION_THRESHOLD, _underflows, angle_units
from nwqlib.subroutines.hamiltonian_evolution import (
    PauliEvolutionBlock,
)
from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import (
    resolve_number_projector_lowering,
    wrapped_projector_phase,
)

# The support tables of one objective: each sorted variable support of
# objective.ObjectiveDecomposer maps to the float64 array of its K**|S| finite
# real values in C order, first support variable most significant, evaluated
# at the centered coordinates x - m, with m the box point nearest the origin
# (objective.expansion_centers, compiler.QHDCompiler._objective_grid_values).
SupportTables = Mapping[tuple[int, ...], Any]


def coerce_real_scalar(value: object, *, context: str) -> float:
    """Return ``value`` as a finite real scalar.

    Any nonzero imaginary part rejects, however small. A complex objective
    value would make the diagonal potential non-Hermitian, and dropping the
    imaginary part would change the objective.
    """

    scalar = complex(value)
    if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
        raise ValueError(f"{context} evaluated to a non-finite value: {value!r}")
    if scalar.imag != 0.0:
        raise ValueError(f"{context} evaluated to a complex value: {value!r}")
    return float(scalar.real)


@dataclass(frozen=True)
class ProjectorOccurrence:
    """One table entry's projector in one potential factor, as the range selection kept or omitted it.

    Attributes:
        support: The variable support of the table.
        grid_indices: The entry's grid-index tuple.
        coefficient: ``p = fl(b v)`` of a kept entry, zero for a zero value or weight.
        angle: ``zeta = fl(t p)`` of a kept entry, zero for a zero value or weight.
        identity: ``q = p/2**s``, the identity coefficient of a kept entry.
        omitted: The exact ``|Z| = |t b v|`` of an omitted entry, None when kept.
    """

    support: tuple[int, ...]
    grid_indices: tuple[int, ...]
    coefficient: float
    angle: float
    identity: float
    omitted: Fraction | None


class PotentialCompiler:
    """Compile diagonal objective terms into Pauli-Z projector products.

    This realizes the diagonal potential ``f(x)`` in the QHD Hamiltonian of
    Eq. (1) in Leng et al., arXiv:2303.01471v1, with the diagonal finite-grid
    potential of Eq. (F.9) and, for d variables, Eq. (F.15).
    The one-hot register representation is distinct from that paper's
    Hamming-weight symmetric-state encoding in Eq. (F.36).

    With ``n = (I - Z)/2`` on each qubit, Wu et al., arXiv:2605.12066v1,
    Sec. IV.A, Eqs. (9)-(10), encode a single-variable function as
    ``sum_k f(g_k) n_k``. The text after their Eq. (10) builds multivariate
    functions from products of single-variable operators. NWQLib extends
    this to a general support term: ``f_S`` becomes
    ``sum_g f_S(g) prod_{j in S} n_{j, g_j}``, which equals ``f_S(x_S)`` on
    every state with one excitation per register, because exactly one
    product is nonzero there. Each table entry gives one number-projector
    block with phase angle ``dt * potential_weight * f_S(g)``, the duration
    times the diagonal matrix element. A zero table value gives a zero angle
    and emits nothing. Every other entry is kept or omitted by
    ``select_occurrences`` before pruning, and kept angles below the
    threshold are counted in the dropped-rotation ledger.
    """

    def __init__(self, rotation_threshold: float = DEFAULT_ROTATION_THRESHOLD) -> None:
        self.rotation_threshold = float(rotation_threshold)
        # Wrapped projector phases of this compiler's range selection, by (angle.hex(), s).
        self._wrapped = {}
        self._reset_dropped_rotations()

    def _reset_dropped_rotations(self) -> None:
        """Reset the ledgers of the rotations that the threshold dropped and of the range omissions."""

        self._dropped_nonzero_contribution_count = 0
        self._dropped_absolute_angle_sum = 0.0
        # Exact sum of the dropped absolute angles in units of 2**-1074
        # (validation.angle_units), the source of the pruning bound.
        self._dropped_angle_units = 0
        # Range omissions (select_occurrences): the count and the exact sums of
        # (1 - 2**-s) |Z| and 2**-s |Z| over the omitted occurrences.
        self._omitted_count = 0
        self._omitted_traceless = Fraction(0)
        self._omitted_identity = Fraction(0)

    def select_occurrences(
        self,
        grid_values: SupportTables,
        *,
        num_grid_points: int,
        dt: float,
        potential_weight: float,
    ) -> tuple[ProjectorOccurrence, ...]:
        """Keep or omit each table entry's projector and its identity event together, for one potential factor.

        ``grid_values`` holds the stored support tables (``SupportTables``),
        whose entry at C-order position e has the grid indices of the
        mixed-radix digits of e in base ``num_grid_points``. For the duration t (dt, or dt/2 for a second-order half), the step
        weight b, a table value v and support size s, the intended exponent
        is ``Z = t b v``. A zero value or weight is kept with zero angle and
        emits nothing. Otherwise the checked construction forms ``p = fl(b v)``,
        ``zeta = fl(t p)``, the identity coefficient ``q = p/2**s`` and its
        phase ``fl(-t q)``, and needs ``|zeta| 2**-s >= 2**-1022`` for the
        projector's power-of-two parameter scalings and, with the
        phase-diagonal provider, the same for its nonzero wrapped phase
        (``pauli_evolution.wrapped_projector_phase``). A value need not be
        normal itself: a subnormal v with ``b = 16`` can give normal results.
        When any of these results would be nonzero and below ``2**-1022``
        (``validation._underflows``), the whole occurrence is omitted, the
        projector and its identity event together, and no later product of
        its coefficient is formed. Overflow and nonfinite results still
        raise. One owner decides both halves, since independent decisions
        could keep a projector without its identity or the reverse.

        Charges. The projector splits as ``P = (P - 2**-s I) + 2**-s I`` with
        ``||P - 2**-s I|| = 1 - 2**-s``, and ``||exp(-i a G) - I|| <= |a| ||G||``
        for a Hermitian G, so an omitted occurrence costs ``(1 - 2**-s) |Z|``
        in its traceless part and ``2**-s |Z|`` in its identity phase, which
        add to ``|Z|``, the direct bound for dropping a projector of norm one. The traceless charge is already in
        the omitted weight of ``circuit_errors.block_errors`` (the
        ``rotation_pruning`` entry), which subtracts the kept blocks from the
        intended weight of the original table, and the identity charge enters
        the phase ledger (``method._phase_ledger``). Both sums are kept exact.
        Occurrences whose products are all normal pass every range check of
        the construction, the provider's included.
        """
        from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import (
            structured_number_projector_provider,
        )

        occurrences = []
        for support, table in grid_values.items():
            s = len(support)
            scale = 1.0 / (2**s)
            diagonal = structured_number_projector_provider(s)["provider"] == "diagonal_synthesis"
            # itertools.product enumerates the grid-index tuples in the C order of the table.
            entries = zip(itertools.product(range(num_grid_points), repeat=s), np.asarray(table).tolist(),
                          strict=True)
            for grid_indices, value in entries:
                key = (tuple(support), tuple(grid_indices))
                if value == 0.0 or potential_weight == 0.0:
                    occurrences.append(ProjectorOccurrence(*key, 0.0, 0.0, 0.0, None))
                    continue
                where = f"support {key[0]} at grid indices {key[1]}"
                kept = self._kept_products(value, dt, potential_weight, scale, s, diagonal, where, self._wrapped)
                if kept is None:
                    # |Z| = |t b v| exactly, split into the charges (1 - 2**-s)|Z| and 2**-s|Z|.
                    exact = abs(Fraction(dt) * Fraction(potential_weight) * Fraction(value))
                    self._omitted_count += 1
                    self._omitted_traceless += exact * (1 - Fraction(1, 2**s))
                    self._omitted_identity += exact / 2**s
                    occurrences.append(ProjectorOccurrence(*key, 0.0, 0.0, 0.0, exact))
                else:
                    occurrences.append(ProjectorOccurrence(*key, *kept, None))
        return tuple(occurrences)

    @staticmethod
    def _kept_products(value, duration, weight, scale, s, diagonal, where, wraps=None):
        """Return ``(p, zeta, q)`` of one occurrence, or None when a product would fall below the normal range."""
        coefficient = weight * value
        if _underflows(coefficient, f"the potential coefficient b v of {where}",
                       lambda: Fraction(weight) * Fraction(value)):
            return None
        angle = duration * coefficient
        if _underflows(angle, f"the projector angle t b v of {where}",
                       lambda: Fraction(duration) * Fraction(coefficient)):
            return None
        identity = coefficient * scale
        if _underflows(identity, f"the projector identity coefficient b v/2**s of {where}",
                       lambda: Fraction(coefficient) * Fraction(scale)):
            return None
        if _underflows(-duration * identity, f"the projector identity phase of {where}",
                       lambda: Fraction(duration) * Fraction(identity)):
            return None
        # |zeta| 2**-s >= 2**-1022, compared through the exponent of zeta.
        if math.frexp(abs(angle))[1] - 1 < s - 1022:
            return None
        if diagonal:
            wrapped = wrapped_projector_phase(angle, s, wraps)
            if not math.isfinite(wrapped):
                raise ValueError(f"the wrapped projector phase of {where} is {wrapped!r}, not finite")
            if wrapped != 0.0 and math.frexp(wrapped)[1] - 1 < s - 1022:
                return None
        return coefficient, angle, identity

    def compile(
        self,
        grid_values: SupportTables,
        grid: OneHotGrid,
        *,
        dt: float,
        potential_weight: float,
        time: float | None = None,
    ) -> tuple[PauliEvolutionBlock, ...]:
        """Return Pauli-evolution blocks for the potential operator (``select_occurrences``, ``compile_selected``).

        ``grid_values`` maps each variable support to the finite real objective
        table that ``QHDCompiler`` evaluated once on every grid point of that
        support. This method never evaluates the objective.
        """
        occurrences = self.select_occurrences(grid_values, num_grid_points=grid.num_grid_points, dt=dt,
                                              potential_weight=potential_weight)
        return self.compile_selected(occurrences, grid, dt=dt, time=time)

    def compile_selected(
        self,
        occurrences,
        grid: OneHotGrid,
        *,
        dt: float,
        time: float | None = None,
    ) -> tuple[PauliEvolutionBlock, ...]:
        """Return the blocks of the kept nonzero occurrences, pruning angles below the threshold."""

        blocks: list[PauliEvolutionBlock] = []
        for occurrence in occurrences:
            support, grid_indices = occurrence.support, occurrence.grid_indices
            hamiltonian_coefficient, angle = occurrence.coefficient, occurrence.angle
            if occurrence.omitted is not None or angle == 0.0:
                continue
            qubits = [
                grid.qubit(var_index, grid_index)
                for var_index, grid_index in zip(support, grid_indices, strict=True)
            ]
            if abs(angle) >= self.rotation_threshold:
                resolution = resolve_number_projector_lowering(len(qubits))
                blocks.append(
                    PauliEvolutionBlock(
                        terms=(),
                        time_step=float(dt),
                        time=time,
                        kind="number_projector",
                        support=tuple(qubits),
                        angle=float(angle),
                        metadata={
                            "coefficient": float(hamiltonian_coefficient),
                            "variable_support": tuple(support),
                            "lowering_resolution": resolution,
                        },
                    )
                )
            else:
                self._dropped_nonzero_contribution_count += 1
                self._dropped_absolute_angle_sum += abs(angle)
                self._dropped_angle_units += angle_units(angle)
        return tuple(blocks)


def _evaluate_objective(evaluator, point, *, expression, support, name="QHD objective"):
    """Return the function ``name`` at one grid point as a finite real float, or raise ValueError naming it."""
    try:
        return coerce_real_scalar(evaluator(*point), context=name)
    except (ArithmeticError, ValueError, TypeError, NameError) as error:
        context = f"{name} {expression} on support {support} at grid point {point}"
        raise ValueError(f"{context} failed finite-real evaluation: {error}") from error
