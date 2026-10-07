"""One-hot finite-difference grid records for QHD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class OneHotGrid:
    """Tensor-product grid encoded with one qubit per variable grid point.

    The Optimization problem admits the ordered variables and finite
    ``lower < upper`` bounds. The QHD Method admits ``num_grid_points >= 2``,
    and with ``boundary="periodic"`` no boundary points, an even
    ``num_grid_points >= 4`` for the one-hot encoding and ``2**b`` for the
    binary encoding (``method.QHD._periodic_grid`` gives the reasons).
    This record is the one definition of the coordinates, the spacing and the
    kinetic links that the compiler, the classical references, the native
    builder, the analysis and the verification read. With K points per
    variable the local index i = 0..K-1 increases with the coordinate, and the
    one-hot wire for variable a and local point i is a*K+i, with adjacent
    variable blocks in the same register. The binary encoding reads the same
    coordinates and spacing, its classical stencil
    (``theory.restricted_kinetic_sparse``) also reads the links, and it lays
    out its own register of b qubits per variable (``binary`` module), so
    ``qubit`` and ``num_qubits`` describe the one-hot register only.

    - Dirichlet interior grid, the default: ``x_i = lower + (i+1) h`` with
      ``h = (upper-lower)/(K+1)``. The unknown amplitudes sit at interior
      points. Homogeneous Dirichlet boundary values are zero at ``lower`` and
      ``upper``, so they need no register coordinates. The three-point
      stencil approximates -1/2 times the second derivative with O(h**2)
      local error for smooth data.
    - Dirichlet endpoint grid, ``include_boundary_points=True``:
      ``x_i = lower + i h`` with ``h = (upper-lower)/(K-1)``, so
      ``x_0 = lower`` and ``x_(K-1) = upper``. The endpoint amplitudes are
      dynamical, with missing neighbors treated as zero outside that grid.
      This reproduces the mesh j/r, j=0..r, in Leng et al., arXiv:2303.01471v1
      Eqs. (F.5) and (F.9) (pp. 46-47), with K=r+1. Eq. (F.7) gives the
      stencil. The paper states vanishing boundary values with Eq. (F.4), yet
      its (r+1)-square matrix in Eq. (F.9) keeps the endpoint amplitudes,
      which is what the endpoint grid reproduces. The interior grid is the one
      that fixes the amplitude to zero at the box endpoints.
    - Periodic grid, ``boundary="periodic"``: ``x_i = lower + i h`` with
      ``h = (upper-lower)/K``, endpoint exclusive. The period is
      ``L = upper - lower``, so ``upper`` is the same point as ``lower`` and
      the K points are distinct. The point after ``x_(K-1)`` is
      ``x_(K-1) + h = lower + L``, which is ``x_0``, so the stencil also joins
      those two points (the wrap link of ``links``). The periodic second
      difference ``(psi_(i+1) - 2 psi_i + psi_(i-1))/h**2``, indices modulo K,
      approximates the second derivative of a smooth L-periodic function with
      O(h**2) local error at every point, including i = 0 and K-1. Liu et
      al., arXiv:2607.16996v1, Sec. III C, Eq. (12), give this one-hot
      kinetic term with the wrap term ``X_(N-1) X_0 + Y_(N-1) Y_0``. These
      are the points ``{lower, lower + dx, ..., upper - dx}``,
      ``dx = (upper - lower)/2**q``, of the mesh in Leng et al.,
      arXiv:2303.01471v1, Algorithm 1 step 1 (p. 44), which writes the box
      as ``[a, b]``, with ``K = 2**q``. That algorithm's Fourier kinetic
      step is periodic on this mesh as well. The spectral
      kinetic model (``QHD.kinetic_model``) acts on the same points with
      period ``L = K h`` (``split_step.kinetic_eigenvalues``).

    The record admits a box only where these numbers exist in binary64
    (``__post_init__``).

    Attributes:
        variables: The Problem's variables, in coordinate order.
        bounds: ``(lower, upper)`` of each variable, in the same order.
        num_grid_points: K, the grid points of each variable.
        include_boundary_points: Place the Dirichlet grid on both box
            endpoints instead of the interior.
        boundary: ``"dirichlet"`` or ``"periodic"``, the boundary condition
            of the finite-difference kinetic term.
    """

    variables: tuple[Any, ...]
    bounds: tuple[tuple[float, float], ...]
    num_grid_points: int
    include_boundary_points: bool = False
    boundary: Literal["dirichlet", "periodic"] = "dirichlet"

    def __post_init__(self) -> None:
        """Reject a box whose spacing, kinetic coefficient or coordinates binary64 cannot hold.

        The three-point stencil weights the kinetic term by ``1/h**2`` on the
        diagonal and ``-1/(2 h**2)`` between neighbors (Leng et al.,
        arXiv:2303.01471v1, Eq. (F.7)), and every consumer computes its
        coefficients from ``spacing`` in binary64: ``1/(h*h)`` and
        ``-0.5/(h*h)`` (``theory.restricted_kinetic_sparse``,
        ``method._schrodinger_kinetic_bounds``) and the hopping coefficient
        ``-w/(4.0*h*h)`` with the schedule weight w
        (``kinetic.KineticCompiler``), and the phase ledger and the binary
        kinetic table divide by ``h**2``. Each variable's spacing h must
        therefore be finite and positive, ``h*h`` and ``h**2`` normal binary64
        numbers, ``(4 h) h`` finite and ``0.25/(h*h)`` normal, that is at least
        ``sys.float_info.min``. Then every stencil coefficient ``1/h**2``,
        ``1/(2 h**2)`` and ``1/(4 h**2)`` is a normal binary64 number and
        every denominator is finite. Otherwise a denominator overflows and the
        coefficient becomes zero, ``h*h`` underflows to zero and the
        coefficient divides by zero, or a value is subnormal and keeps fewer
        than 53 significant bits. These hold for h from about 1.5e-154 to
        3.3e153. The range of w belongs to the schedule. The box
        ``[-1e308, 1e308]`` fails because its width overflows to infinity,
        and ``[0, 1e-200]`` because ``h*h`` underflows to zero.

        The coordinates are checked as ``grid_value`` produces them in
        binary64. They must be finite and strictly increasing,
        ``x_(i+1) > x_i``. The periodic grid starts at ``lower`` and ends
        below ``upper``, the interior Dirichlet grid excludes both endpoints,
        and the endpoint Dirichlet grid starts at ``lower``. Its last
        coordinate ``lower + (K - 1) h`` is only checked to be finite, because
        rounding moves it off ``upper`` for many ordinary boxes, for example
        to 0.9999999999999998 on ``[-1, 1]`` with K = 50. Strict increase rejects duplicate encoded points even when
        the spacing is positive. On ``[1e16, 1e16 + 4)`` with K = 4 the
        periodic offsets would be ``0, 0, 2, 4``, a duplicate and the
        excluded endpoint. Checking only h, only the last coordinate or only
        ``lower + h != lower`` misses such grids. The check forms each
        variable's K coordinates as one float64 array (``coordinate_array``),
        one variable at a time, and planning admits it with its ``d K``
        initial-state work before building the grid (``method.QHD.plan``). It does not form the Cartesian product or
        require each rounded gap to equal h. A coordinate need not be
        normal, since an exact zero or subnormal coordinate is harmless when
        the spacing, distinctness, table and coefficient conditions hold.
        The check's array payload peaks at 16K bytes for one variable and 25K-1
        bytes for two or more variables, because the previous coordinates and
        failure mask remain live while the next axis's indices and coordinates are
        built. Planning covers this scratch with the 320dK term of
        method._initial_state_bytes and its fixed-object allowance. Grid validation
        and initial-state evaluation share that scratch allowance as successive
        phases.

        Raises:
            ValueError: A variable's grid fails one of these conditions. The
                message names the variable, its box and spacing or
                coordinates, and for coordinates that round onto a bound
                the remedy, a wider box or one nearer zero.
        """
        from math import isfinite
        from sys import float_info

        import numpy as np

        interior = self.boundary == "dirichlet" and not self.include_boundary_points
        for j, (lower, upper) in enumerate(self.bounds):
            name = self.variables[j]
            h = self.spacing(j)
            if not (isfinite(h) and h > 0):
                raise ValueError(f"the grid of {name} on [{lower}, {upper}] has spacing h = {h!r}, which is "
                                 "not a finite positive binary64 number")
            # The evaluated forms h*h, h**2 and (4 h) h that consumers form. Python's float power raises
            # OverflowError where the products give inf, and inf fails the check below with this box's message.
            try:
                power = h**2
            except OverflowError:
                power = float("inf")
            square, quadruple = h * h, (4.0 * h) * h
            quarter = 0.25 / square if square else float("inf")
            if not (float_info.min <= min(square, power) and quadruple <= float_info.max
                    and quarter >= float_info.min):
                raise ValueError(f"the grid of {name} on [{lower}, {upper}] has spacing h = {h!r}, so the kinetic "
                                 f"coefficients 1/h**2 to 1/(4 h**2) are not all normal binary64 numbers with "
                                 f"finite denominators (h**2 = {square!r})")
            # A first or last coordinate that rounds onto a bound belongs to a box too narrow for the
            # magnitude of its coordinates.
            remedy = "Choose a wider box or one nearer zero"
            x = self.coordinate_array(j)
            first = float(x[0])
            if not isfinite(first) or (first <= lower if interior else first != lower):
                raise ValueError(f"the first grid coordinate {first!r} of {name} on [{lower}, {upper}] is not "
                                 f"{'inside the box' if interior else 'the lower bound'} in binary64. {remedy}")
            # Strict increase compares neighbors without subtracting them, which could overflow.
            failed = ~(np.isfinite(x[1:]) & (x[1:] > x[:-1]))
            if failed.any():
                i = int(np.argmax(failed)) + 1
                previous, value = float(x[i - 1]), float(x[i])
                raise ValueError(f"the grid of {name} on [{lower}, {upper}] with spacing h = {h!r} has the "
                                 f"coordinates {previous!r} and {value!r} at indices {i - 1} and {i}, which are "
                                 "not finite and strictly increasing in binary64. The box is too narrow for its "
                                 "magnitude or too wide")
            last = float(x[-1])
            if (self.boundary == "periodic" or interior) and last >= upper:
                raise ValueError(f"the last grid coordinate {last!r} of {name} on [{lower}, {upper}] is not "
                                 f"below the upper bound in binary64. {remedy}")

    @property
    def num_variables(self) -> int:
        """Return the number of encoded variables."""

        return len(self.variables)

    @property
    def num_qubits(self) -> int:
        """Return the number of one-hot qubits, ``num_variables`` times ``num_grid_points``."""

        return self.num_variables * self.num_grid_points

    def qubit(self, var_index: int, grid_index: int) -> int:
        """Return the physical qubit index for a variable/grid-point pair."""

        if not 0 <= var_index < self.num_variables:
            raise IndexError("var_index out of range")
        if not 0 <= grid_index < self.num_grid_points:
            raise IndexError("grid_index out of range")
        return var_index * self.num_grid_points + grid_index

    @property
    def num_links(self) -> int:
        """Return the links per variable, K - 1 of a chain or K of a periodic cycle.

        Planning counts the hopping blocks from this number before any link
        is formed (``method.QHD._admit_symbolic_work``).
        """

        k = self.num_grid_points
        return k if self.boundary == "periodic" else k - 1

    def links(self) -> tuple[tuple[int, int], ...]:
        """Return the local point pairs ``(i, j)``, ``i < j``, that the kinetic stencil joins.

        Link l joins the points l and (l + 1) mod K for l below
        ``num_links``. On the Dirichlet grids these are the K-1 neighbors
        ``(l, l+1)`` of a chain. The periodic grid adds link K-1, the wrap
        link ``(0, K-1)`` between ``x_(K-1)`` and ``x_0``, which closes the
        chain into a cycle. Every variable has the same links. The pair is
        written with the smaller index first because compact Pauli labels list
        their qubits in increasing order. The hopping ``XX + YY`` is symmetric
        in its two qubits, so the order does not change the operator. With
        K = 2 the wrap link joins the same two points as link 0. The one-hot
        QHD excludes that grid, and for the binary encoding at K = 2 the two
        coinciding links of the classical stencil add to the off-diagonal
        ``-1/h**2`` of ``(I - X)/h**2`` (``theory.restricted_kinetic_sparse``).
        """

        k = self.num_grid_points
        return tuple((min(l, (l + 1) % k), max(l, (l + 1) % k)) for l in range(self.num_links))

    def spacing(self, var_index: int) -> float:
        """Return the grid spacing ``h = (upper - lower)/n`` of one variable.

        n counts the intervals of width h across the box: K + 1 on the
        Dirichlet interior grid, K - 1 on the Dirichlet endpoint grid and K on
        the periodic grid, whose last interval ends at the identified point
        ``upper = lower + L``.
        """

        lower, upper = self.bounds[var_index]
        k = self.num_grid_points
        if self.boundary == "periodic":
            intervals = k
        elif self.include_boundary_points:
            intervals = k - 1
        else:
            intervals = k + 1
        return (upper - lower) / intervals

    def coordinate_array(self, var_index: int):
        """Return the K coordinates of one variable as a float64 array, bit for bit ``grid_value``.

        Form each coordinate by converting its integer grid index, multiplying by the stored spacing and adding
        the lower bound, in that order. ``__post_init__`` checks finiteness, strict increase and the boundary
        conditions of the selected grid on this array. It uses the same binary64 coordinates as ``grid_value``.

        ``grid_value(j, i)`` is ``lower + (i + interior) h``, where interior is one only for a Dirichlet grid
        without endpoints. The multiplication converts the integer to binary64 and rounds once, then the
        addition rounds once. The integer index is generated first, converted to float64, and multiplied and
        added by separate ufuncs, which gives the same numbers on every admitted materializable grid. A floating
        ``arange`` with a large start, a fused expression or ``linspace`` does not have this property. For K
        below ``2**53`` every index is exactly representable.
        """
        import numpy as np

        interior = int(self.boundary == "dirichlet" and not self.include_boundary_points)
        numbers = np.arange(self.num_grid_points, dtype=np.int64)
        numbers += interior
        x = numbers.astype(np.float64)
        x *= self.spacing(var_index)
        x += self.bounds[var_index][0]
        return x

    def grid_value(self, var_index: int, grid_index: int) -> float:
        """Return ``x_i = lower + (i + 1) h`` on the interior grid and ``lower + i h`` on the others."""

        lower, _ = self.bounds[var_index]
        interior = self.boundary == "dirichlet" and not self.include_boundary_points
        return lower + (grid_index + int(interior)) * self.spacing(var_index)
