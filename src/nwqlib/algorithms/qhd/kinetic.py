"""Kinetic-term compiler for the one-hot QHD Hamiltonian."""

from __future__ import annotations

from nwqlib.algorithms.qhd.grid import OneHotGrid
from .validation import (
    DEFAULT_ROTATION_THRESHOLD,
    _kinetic_range_error,
    _normal_product,
    _normal_quotient,
    angle_units,
)
from nwqlib.subroutines.hamiltonian_evolution import PauliEvolutionBlock, PauliEvolutionTerm


class KineticCompiler:
    """Compile nearest-neighbor one-hot kinetic terms.

    The restricted matrix is the finite-difference kinetic operator for
    ``-Delta / 2`` in Eq. (1) of Leng et al., arXiv:2303.01471v1, with the
    grid's boundary condition. It acts on one excitation per variable
    register, not the Hamming-weight symmetric-state encoding of that paper's
    Eq. (F.36).

    Per variable the operator is ``-(1/2) L`` with ``L = (A - 2I)/h**2`` as in
    Remark 6 of the same paper, where A is the adjacency matrix of the grid's
    links (``OneHotGrid.links``). On the Dirichlet grids the links form a
    chain, the stencil of that paper's Eqs. (F.7)/(F.9). On the periodic grid they form a
    cycle, so ``-(1/2) L`` is the circulant matrix with diagonal ``1/h**2``
    and ``-1/(2 h**2)`` at ``(i, i +- 1 mod K)``, and the hopping sum
    including the wrap link is Eq. (12) of Liu et al., arXiv:2607.16996v1,
    Sec. III C. On the one-hot subspace
    ``(X_i X_j + Y_i Y_j)/2`` acts as ``|e_i><e_j| + h.c.``, a direct
    two-qubit calculation with no paper source, so each link carries the
    coefficient ``-kinetic_weight/(4 h**2)`` on XX and on YY. The hopping
    conserves the excitation number of each register, so the evolution never
    leaves the valid subspace. The stored block angle is
    ``2*duration*coefficient``, which equals the duration times the stencil
    off-diagonal ``-kinetic_weight/(2 h**2)``. The stencil diagonal
    ``kinetic_weight/h**2`` of each variable is the same on every one-hot
    state, so it is a global phase, which the compiler adds to its physical
    phase ledger instead of emitting a gate
    (``compiler.QHDCompiler._record_kinetic_global_phase``, which cites the
    papers that drop it as well).

    Links that share a point do not commute. First order applies the links in
    increasing link order for the full dt (the wrap link last), a further
    product approximation of ``exp(-i dt K)``, whose operator-norm error
    ``evolution_bounds.evolution_bound`` bounds. Second order applies the
    odd-indexed links for dt/2, the even-indexed links for dt and the
    odd-indexed links for dt/2. Links of one parity must be disjoint, so that
    they commute and each layer is exact, which makes the split symmetric. On
    the chain link l joins points l and l+1, so two links of equal parity
    are at least two apart and disjoint. On the cycle the wrap link
    ``(0, K-1)`` has index K-1. For even K it is odd, and the other odd links
    ``(1, 2), ..., (K-3, K-2)`` use points 1..K-2, so both layers stay
    disjoint. For odd K it is even and shares point 0 with link ``(0, 1)``,
    and an exact split would need a third layer. Liu et al.,
    arXiv:2607.16996v1, likewise group the strings of their Eq. (12) into
    two commuting layers of bonds starting on even and odd sites (Sec. III C),
    with K a power of two (Sec. III). The QHD Method therefore
    requires an even K on the periodic grid, a limit of this two-layer
    compiler, not of the periodic Laplacian, and it also excludes the
    degenerate cycle K = 2 (``method.QHD._periodic_grid``).
    """

    def __init__(self, rotation_threshold: float = DEFAULT_ROTATION_THRESHOLD) -> None:
        self.rotation_threshold = float(rotation_threshold)
        self._reset_dropped_rotations()

    def _reset_dropped_rotations(self) -> None:
        """Reset metadata for nonzero rotations omitted by the threshold."""

        self._dropped_nonzero_contribution_count = 0
        self._dropped_absolute_angle_sum = 0.0
        # Exact sum of the dropped absolute angles in units of 2**-1074
        # (validation.angle_units), the source of the pruning bound.
        self._dropped_angle_units = 0

    def compile(
        self,
        grid: OneHotGrid,
        *,
        dt: float,
        kinetic_weight: float,
        time: float | None = None,
        trotter_order: int = 1,
    ) -> tuple[PauliEvolutionBlock, ...]:
        """Return Pauli-evolution blocks for the kinetic operator.

        Range admission (``validation._normal_range``) happens before
        pruning. The coefficient ``c_h = fl(-a/fl(fl(4 h) h))``, the doubled
        duration ``fl(2 t)``, the stored angle ``ell = fl(fl(2 t) c_h)`` and
        the native parameter ``2 ell`` must be finite, and the coefficient and
        angle normal. The duration t is positive, so the angle is zero only for a
        zero weight a, which emits nothing, and any other zero would be range
        loss, which is refused with its remedy
        (``validation._kinetic_range_error``), since it depends only on the
        grid and the schedule. With ``dt = 1e308``, the full-duration hopping
        layer overflows at ``fl(2 t)`` even when the exact product ``2 t c_h``
        with a small normal coefficient is moderate, so the refusal requests a
        shorter ``total_time``. This evaluation order is the one the native
        parameter repeats (``pauli_evolution._hopping_parameter``), whose
        equality with the stored angle makes the hopping-parameter entry of
        the error ledger zero.
        """

        blocks: list[PauliEvolutionBlock] = []
        links = grid.links()
        # Odd-indexed links, then even-indexed links, then the odd ones again
        # (the class docstring shows that each layer is disjoint).
        segments = (
            ((links[1::2], dt / 2), (links[0::2], dt), (links[1::2], dt / 2))
            if trotter_order == 2
            else ((links, dt),)
        )
        for var_index in range(grid.num_variables):
            spacing = grid.spacing(var_index)
            # (XX + YY) maps |01> <-> |10> with amplitude 2, so this
            # coefficient realizes the stencil off-diagonal
            # -kinetic_weight / (2 h^2) between the linked one-hot states. The
            # uniform stencil diagonal (+kinetic_weight / h^2 per variable,
            # on both boundaries) is an identity in the one-hot subspace and
            # is recorded by the compiler as dropped global phase (source
            # "kinetic_diagonal"), not emitted as a Pauli term here.
            try:
                # c_h = -a/D with D = fl(fl(4 h) h), which the grid keeps finite and normal (OneHotGrid).
                coefficient = _normal_quotient(-kinetic_weight, 4.0 * spacing * spacing,
                                               f"the hopping coefficient -a/(4 h**2) of variable {var_index}")
            except ValueError as error:
                raise _kinetic_range_error(error) from error
            for layer, duration in segments:
                # The doubled duration depends on total_time and num_steps alone, so its refusal names
                # total_time instead of the kinetic-scale remedy. The duration is normal (schedules.step_weights),
                # so only its upper end can fail.
                try:
                    doubled = _normal_product(2.0, duration, "twice the hopping duration 2 t")
                except ValueError as error:
                    raise ValueError(f"{error}. Choose a shorter total_time") from error
                try:
                    angle = _normal_product(doubled, coefficient,
                                            f"the hopping angle 2 t c of variable {var_index}")
                    _normal_product(2.0, angle, f"the hopping gate parameter 4 t c of variable {var_index}")
                except ValueError as error:
                    raise _kinetic_range_error(error) from error
                if angle == 0.0:
                    continue
                if abs(angle) < self.rotation_threshold:
                    self._dropped_nonzero_contribution_count += len(layer)
                    self._dropped_absolute_angle_sum += len(layer) * abs(angle)
                    self._dropped_angle_units += len(layer) * angle_units(angle)
                    continue
                for left, right in layer:
                    q0 = grid.qubit(var_index, left)
                    q1 = grid.qubit(var_index, right)
                    blocks.append(
                        PauliEvolutionBlock(
                            terms=(
                                PauliEvolutionTerm(pauli=f"x{q0}x{q1}", coefficient=coefficient),
                                PauliEvolutionTerm(pauli=f"y{q0}y{q1}", coefficient=coefficient),
                            ),
                            time_step=float(duration),
                            time=time,
                            kind="kinetic",
                            angle=angle,
                        )
                    )
        return tuple(blocks)
