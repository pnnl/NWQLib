"""Restricted-subspace numerical references for QHD.

For one-hot encoding, ``_run_ir_product`` applies the analytic restricted
action of each stored block, a projector phase on a grid slice or a
two-slice hopping rotation, at its stored angle on the K**d valid states
(``apply_projector``, ``apply_hopping``). Each ideal block preserves that
subspace. ``onehot_product_state_error`` prices these actions and
``onehot_product_sizes`` gives their work and byte law.
``restricted_kinetic_sparse`` builds the finite-difference kinetic
operator there. For binary encoding, ``run_binary_product`` evaluates
the selected QFT operations and computed diagonal phase arrays in place
on the whole register in lexicographic order. ``binary_product_state_error``
prices state application relative to exact unitaries at the kernel's
phase parameters, excluding formation of the diagonal arrays.
``binary_product_sizes`` gives the work and byte law.

These are numerical references with their own arithmetic. In particular,
binary Walsh reconstruction can change relative phases compared with
the native product of separate rotations. They share the Plan's grid,
objective tables and schedule, so comparisons with them do not supply
independent discretization evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import scipy.sparse

from nwqlib.algorithms.qhd.grid import OneHotGrid
from nwqlib.subroutines.hamiltonian_evolution import PauliEvolutionBlock

# First-order local state-operation charges, in units of u = 2**-53, of one
# direct one-hot hopping and one projector block:
# rho_H <= (1 + 3 sqrt(2)) u < 6u and rho_P <= 10u (onehot_product_state_error).
# Registered in docs/ENGINEERING_CONSTANTS.md ("QHD one-hot product budget"). Revisit when the kernel
# changes its arithmetic.
ONEHOT_HOPPING_STATE_ROUNDOFF = 6
ONEHOT_PROJECTOR_STATE_ROUNDOFF = 10


def restricted_kinetic_work(dimension, variables, raw_slots, entries):
    """Return the logical construction-work bound of restricted_kinetic_sparse.

    For D=dimension, d=variables, raw COO slots S=raw_slots and final CSR
    entries M=entries, W_stencil=28*S+4*M+(37*d+12)*D+40*(d+1).
    One unit is one element of a fill, arithmetic, comparison, reduction,
    cast, gather or scatter pass. An indexed update counts its update once,
    and a sparse reduction counts each input record once. This is a logical
    visit model, not a FLOP count or a time bound.

    Raw row, column and value writes cost 3*S, and their separate integer
    index checks cost another 3*S. Base, lengths and the prefix sum cost
    at most 3*D. Length calculation costs at most 7*d*D. The remaining
    fill calculations cost at most 30*d*D, including two mask scans per
    Boolean extraction and assignment, plus their gathered or written
    values. Scalar shape, stride, coefficient and loop bookkeeping costs
    at most 32*(d+1) logical operations. The 40*(d+1) bundle covers them
    and the five conversion endpoints because 32*(d+1)+5 <= 40*(d+1)
    for every admitted d >= 1.

    SciPy 1.18.1 conversion uses at most 22*S+4*M+9*D+5 visits for
    index selection/casts, coordinate checks, COO-to-CSR counting and
    scatter, compressed-index checks, sorted/canonical scans, duplicate
    reduction and possible final compaction. The column-major raw fill
    gives sorted CSR rows, so no comparison sort is reached. The relevant
    dependency owners are scipy/sparse/_coo.py, _compressed.py, _base.py,
    _sputils.py and sparsetools/coo.h and csr.h at v1.18.1. Requalify after
    changing that path, the fill ordering or the logical-unit convention.
    NumPy 2.5.2 boolean indexing in numpy/_core/src/multiarray/mapping.c
    counts the mask before extraction or assignment. Its
    mapiter_trivial_set in lowlevel_strided_loops.c.src checks indices
    before the write pass.

    Counts are Python integers before allocation. S and M may be upper
    bounds, since all coefficients are positive. Opaque library internals,
    allocation management and integer bit complexity are outside this law.
    """
    return (28 * raw_slots + 4 * entries + (37 * variables + 12) * dimension
            + 40 * (variables + 1))


def restricted_kinetic_sparse(grid: OneHotGrid) -> scipy.sparse.csr_matrix:
    """Build the finite-difference kinetic operator without dense allocation.

    Each variable contributes diagonal ``1/h**2`` and the entry
    ``-1/(2 h**2)`` between the two local points of each of its links
    (``OneHotGrid.links``), which is ``-(1/2) (A - 2I)/h**2`` of Remark 6
    and Eq. (F.14) of Leng et al., arXiv:2303.01471v1, with A the adjacency
    matrix of the links. On the
    Dirichlet grids the links form a chain and a boundary point has one
    neighbor. On the periodic grid they form a cycle, the wrap link gives
    points 0 and K-1 their second neighbor, and each variable's matrix is
    circulant (``kinetic.KineticCompiler``). At K = 2, which only the binary
    encoding admits, the two links join the same pair of points and their
    entries add to ``-1/h**2``, the periodic stencil ``(I - X)/h**2``. The
    diagonal stays in this matrix, whereas the compiled product moves it into
    the physical phase.

    Construction. For axis j, the stored diagonal
    contribution is ``1.0/(h_j*h_j)`` and each directed link contributes
    ``-0.5/(h_j*h_j)``. The code forms ``h*h``, the product whose normality
    grid admission checks (``OneHotGrid.__post_init__``), and the stored bits
    follow that form. The exact mathematical diagonal is
    ``sum_j h_j**-2``, and each distinct adjacent coordinate differs by
    ``-1/(2*h_j**2)``. At periodic K = 2, two links join the same points, so
    that off-diagonal entry is ``-1/h_j**2``. The diagonal does not double.
    Broadcasting indices constructs precisely these directed pairs. To fix
    the order in which conversion sums duplicate entries, the raw COO entries
    are emitted in column, variable, local-neighbor order. The periodic neighbor
    order is ``(1,K-1)`` at local 0, ``(i-1,i+1)`` in the interior and
    ``(K-2,0)`` at K-1. A change from point-major to axis-major COO can
    change the order in which duplicate diagonals are summed by conversion,
    so equality of mathematical matrices alone does not prove identical
    diagonal bits. This fill fixes the COO conversion input and therefore its
    stored values on a given SciPy version.

    Buffer cost. With ``D = K**d``, the raw slots S
    and final stored entries M are ``S = 3dD`` (periodic) or
    ``S = dD + 2d(K-1)D/K`` (Dirichlet), and ``M = (1+2d)D`` (periodic,
    K > 2), ``(1+d)D`` (periodic, K = 2) or ``D + 2d(K-1)D/K`` (Dirichlet).
    Three arrays with int64 row, int64 column and complex128 value cost
    exactly ``32*S`` payload bytes. A final CSR with index width w in {4,8}
    costs ``C(M,w) = (16+w)M + w(D+1)``. Conversion initially allocates data
    and indices for S entries, not M. SciPy can also cast both COO index
    arrays, and duplicate elimination can allocate a compact final
    data/index copy. A conservative explicit-array construction envelope for
    this preallocated algorithm is
    ``B_build = 32S + 96D + 2wS + (16+w)S + w(D+1) + (16+w)M``.
    The ``96D`` covers the bounded index/mask scratch in this fill,
    including the ``rows`` index expression and Boolean-gather temporaries.
    It is an array allowance, not Python RSS or undocumented sorting
    workspace. The ``2wS`` allows a simultaneous COO-index cast while
    caller-owned int64 arrays remain alive. The final term covers compaction
    even when SciPy can otherwise avoid it. S is computed from its formula
    with Python integers before any allocation, and the last ``slots`` and
    Boolean scratch are released before conversion.

    Construction work is W_stencil=28S+4M+(37d+12)D+40(d+1)
    logical visits, derived in restricted_kinetic_work. It counts the
    raw fill, index and mask passes, dtype checks/casts, sorted CSR
    conversion, duplicate reduction and possible compaction once.
    Callers admit this work before constructing the stencil.
    """
    k, d = grid.num_grid_points, grid.num_variables
    dimension = k**d
    periodic = grid.boundary == "periodic"
    # Raw COO slots from the exact formula (Python integers), before allocation.
    slots_total = 3 * d * dimension if periodic else d * dimension + 2 * d * (k - 1) * (dimension // k)
    base = np.arange(dimension, dtype=np.int64)
    lengths = np.full(dimension, d, dtype=np.int64)
    for j in range(d):
        local = (base // k ** (d - 1 - j)) % k
        lengths += 2 if periodic else (local > 0).astype(np.int64) + (local < k - 1)
    offsets = np.empty(dimension, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths[:-1], out=offsets[1:])
    del lengths
    rows = np.empty(slots_total, dtype=np.int64)
    cols = np.empty(slots_total, dtype=np.int64)
    data = np.empty(slots_total, dtype=np.complex128)
    for j in range(d):
        stride = k ** (d - 1 - j)
        local = (base // stride) % k
        h = grid.spacing(j)
        rows[offsets] = base
        cols[offsets] = base
        data[offsets] = 1.0 / (h * h)
        offsets += 1
        # Every local point has at least one neighbor for k >= 2.
        neighbor = np.where(local == 0, 1, local - 1)
        rows[offsets] = base + (neighbor - local) * stride
        cols[offsets] = base
        data[offsets] = -0.5 / (h * h)
        offsets += 1
        if periodic:
            neighbor = np.where(local == 0, k - 1, np.where(local == k - 1, 0, local + 1))
            rows[offsets] = base + (neighbor - local) * stride
            cols[offsets] = base
            data[offsets] = -0.5 / (h * h)
            offsets += 1
        else:
            interior = (local > 0) & (local < k - 1)
            slots = offsets[interior]
            rows[slots] = base[interior] + stride
            cols[slots] = base[interior]
            data[slots] = -0.5 / (h * h)
            offsets[interior] += 1
            del interior, slots
    del base, offsets, local, neighbor
    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(dimension, dimension)).tocsr()


def apply_projector(view: np.ndarray, qubits, angle) -> None:
    """Apply one stored number-projector block to the ``(K,)*d`` restricted view in place.

    Direct one-hot kernel. Each stored projector
    block has ``U_P = exp[-i theta (Pi_S - 2**-s I)]
    = exp(i theta 2**-s) [I + (exp(-i theta) - 1) Pi_S]`` with ``s = |S|``.
    A support qubit q maps to variable ``j = q // K`` and local point
    ``i = q % K``. On the ``(K,)*d`` restricted view, the projector-one set
    fixes each such axis j to i and leaves other axes free. Compiled
    objective projectors have at most one qubit per variable. Conflicting
    excitations of one variable would give an empty projector on the valid
    subspace, but are not emitted by this compiler. Empty support represents
    the identity and the displayed traceless generator is zero, also not an
    emitted objective block.

    The phase multiplying an inactive entry is ``z0 = exp(i*theta*2**-s)``.
    The active factor is ``z1 = z0*exp(-i*theta)``. Planning's
    kept-projector normal-range checks make ``theta*2**-s`` an exact
    power-of-two scaling. Forming one accumulated global angle and
    evaluating its exponential only once would introduce an additional
    angle-summation and argument-formation problem, which this action
    avoids. An inactive entry has one complex multiplication. Each active
    final entry is one multiplication of its original value by the scalar
    z1, after the temporary preserves that value. The intermediate
    whole-array product on those entries is overwritten. This costs a D pass
    plus a slice of size ``D/K**s``, with bounded scratch of that slice
    size, and two scalar exponentials.
    """
    k = view.shape[0]
    index = [slice(None)] * view.ndim
    seen = set()
    for q in qubits:
        j, i = divmod(q, k)
        if j in seen:
            raise ValueError("compiled projector must name distinct variables")
        seen.add(j)
        index[j] = i
    if not qubits or angle == 0:
        return
    index = tuple(index)
    z0 = np.exp(1j * math.ldexp(float(angle), -len(qubits)))
    z1 = z0 * np.exp(-1j * float(angle))
    active = np.array(view[index], copy=True)
    view *= z0
    view[index] = active * z1


def apply_hopping(view: np.ndarray, qubits, angle) -> None:
    """Apply one stored XX+YY hopping block to the ``(K,)*d`` restricted view in place.

    Direct one-hot kernel. On ``|10>, |01>``, both XX
    and YY interchange the two states with coefficient +1. Their images out
    of the one-excitation sector cancel on ``|00>`` and ``|11>``. A block
    ``exp(-i*t*c*(XX+YY))`` therefore has the two-state matrix
    ``[[cos v, -i sin v], [-i sin v, cos v]]`` with ``v = 2tc``.
    ``KineticCompiler.compile`` forms ``c = RN(-a/RN(RN(4*h)*h))`` and stores
    ``angle = RN(RN(2*t)*c)``, and the doubled duration is admitted and
    exact, so the stored angle is the kernel's reference. For positive
    kinetic weight c is negative, so the off-diagonal update has positive
    imaginary sign. Replacing it by ``+i*sin(angle)`` would reverse the
    kinetic evolution.

    For a link between p and q on axis j, all other indices define
    independent two-state fibers. Each entry outside those two slices is
    unchanged, including boundary and wrap links. Compiled block order stays
    unchanged, because adjacent links generally do not commute. A zero
    angle is the identity and is skipped exactly, as its budget
    (``onehot_product_state_error``) and work charge
    (``onehot_product_sizes``) assume.
    """
    k = view.shape[0]
    (j, p), (other, q) = (divmod(x, k) for x in qubits)
    if j != other or p == q:
        raise ValueError("hopping needs two distinct points of one variable")
    if angle == 0:
        return
    lo = [slice(None)] * view.ndim
    hi = list(lo)
    lo[j], hi[j] = p, q
    lo, hi = tuple(lo), tuple(hi)
    x0 = np.array(view[lo], copy=True)
    x1 = np.array(view[hi], copy=True)
    c, s = math.cos(float(angle)), math.sin(float(angle))
    view[lo] = c * x0 - 1j * (s * x1)
    view[hi] = c * x1 - 1j * (s * x0)


def _run_ir_product(
    grid: OneHotGrid, pauli_ir: Iterable[PauliEvolutionBlock], state: np.ndarray
) -> np.ndarray:
    """Apply the stored one-hot blocks' analytic restricted actions in order on the K**d grid.

    Apply the analytic restricted action of each selected block at its
    stored angle, in stored order. A projector multiplies its selected grid
    slice by the active phase and the remaining entries by its
    identity-compensation phase. A hopping link rotates the two
    corresponding slices with ``cos(angle)`` and ``-i*sin(angle)``. These
    actions are numerical evaluations of the analytic blocks. The caller
    applies the stored physical phase.

    ``pauli_ir`` is read once, block by block (``native.raw_blocks``), and
    each block names its support: the two linked qubits of a hopping or the
    qubits of a projector. ``state`` is the selected initial vector in
    lexicographic grid order. The caller transfers ownership of this
    writable complex128 buffer, the start vector it has just built, and
    the kernel updates it in place and returns it, so one state is live
    (``onehot_product_sizes``). ``onehot_product_state_error`` bounds the
    roundoff.
    """
    k, d = grid.num_grid_points, grid.num_variables
    tensor = np.asarray(state, dtype=np.complex128)
    view = tensor.reshape((k,) * d)
    for block in pauli_ir:
        if block.kind == "number_projector":
            apply_projector(view, block.support, block.angle)
        else:
            apply_hopping(view, block.support, block.angle)
    return view.reshape(-1)


def onehot_product_state_error(steps, *, start) -> float:
    """First-order 2-norm error budget delta of ``_run_ir_product`` from the Plan's start vector.

    The one-hot product reference is the ordered product of analytic
    projector and hopping blocks at their stored binary64 angles, applied to
    the exact initial state. Under normal round-to-nearest arithmetic and
    the stated one-ulp elementary-function premise, its first-order
    state-operation budget is ``u*(start+10*B_P+6*B_H+5)``. The counts
    include every executed occurrence and the final term covers the
    physical-phase application. Parameter formation, native lowering, time
    splitting and grid errors are outside this budget. The mass and mode
    windows use the same state-to-probability conversions as the other
    classical kernels.

    Derivation. Take the exact initial unit state, the
    ordered analytic blocks at their stored angles, and the final scalar
    phase at its stored binary64 angle as the reference. This excludes
    formation of these parameters from analytic grid/schedule data and
    excludes native lowering errors, as the parameter-defined binary host
    reference does. It differs from exact exponentials of rounded sparse
    generators, so the reference states of the two are not bit identical.

    - Unit-phase application: one-ulp sine/cosine or the qualified complex
      exponential gives factor error at most 2u and the complex product at
      most ``sqrt(2)*gamma(2)``, first-order coefficient
      ``2+2*sqrt(2) < 5``, ``GLOBAL_PHASE_STATE_ROUNDOFF``.
    - Projector (``apply_projector``): z1 is formed with two phase-entry
      errors and one complex product, then the state is multiplied once. Its
      active-entry coefficient is ``4+4*sqrt(2) < 10``, and the inactive
      entries use at most 5. Squared errors on disjoint slices add, so the
      whole block has relative defect at most 10u to first order,
      independently of s and D (``ONEHOT_PROJECTOR_STATE_ROUNDOFF``).
    - Hopping (``apply_hopping``): with returned c and s of absolute error at
      most 2u each, the error matrix ``dc*I - i*ds*X`` has norm
      ``sqrt(dc**2+ds**2) <= 2*sqrt(2)*u``. Real scaling of the two complex
      vectors contributes at most ``u*(|c|+|s|) <= sqrt(2)*u`` to first
      order, and their componentwise addition at most u times the norm of
      the formed rotation, so ``rho_H <= (1+3*sqrt(2))*u < 6u``
      (``ONEHOT_HOPPING_STATE_ROUNDOFF``). This uses real scaling followed by
      the exact multiplication by -i. Applying the same fiber bound over
      orthogonal pairs preserves the relative 2-norm factor for the whole
      state.

    For B_H hopping and B_P projector occurrences, including both potential
    halves, ``delta = u (start + 6 B_H + 10 B_P + 5)``. The last 5 prices the
    caller's physical-phase application. Empty block products still carry
    the initial-state and final-phase charges. A zero block skipped exactly
    has no local charge. Occurrence counts come from the selected steps, not
    unique constructions, and a count that is not exactly representable is
    converted upward. As in Proposition 41 of docs/mathematics.md (anchor
    r41), local relative defects rho give the finite propagation relation
    ``delta_next <= (1+rho)*delta + rho``, hence
    ``(1+delta0)*prod(1+rho)-1``, and this budget keeps only its first-order
    part. Underflowing state components need absolute local
    terms for an all-orders extension, which is not established here.

    ``steps`` are the stored ``QHDBlock`` groups and ``start`` the bound of
    the selected ``initial_state.restricted_state_error``, in units of u.
    """
    from nwqlib._validation import GLOBAL_PHASE_STATE_ROUNDOFF, UNIT_ROUNDOFF

    charge = 0
    for group in steps:
        for block in group:
            if block.angle == 0:
                continue
            charge += (ONEHOT_HOPPING_STATE_ROUNDOFF if block.kind == "kinetic"
                       else ONEHOT_PROJECTOR_STATE_ROUNDOFF)
    count = charge + GLOBAL_PHASE_STATE_ROUNDOFF
    upward = float(count)
    if upward < count:
        upward = math.nextafter(upward, math.inf)
    return UNIT_ROUNDOFF * (start + upward)


def onehot_product_sizes(dimension: int, num_grid_points: int, steps, *, start_work: int, start_bytes: int,
                         held_bytes: int) -> tuple[int, int]:
    """Return ``(work_units, workspace_bytes)`` of one ``_run_ir_product`` evolution and its final phase.

    The one-hot product reference applies each stored projector phase or
    two-slice hopping directly. Admission counts every block occurrence and
    the largest slice-copy/expression workspace together with the live
    state.

    Law. Use the actual ordered block occurrences, not unique supports. For
    a projector with s distinct fixed variable axes put ``S_P = D/K**s``.
    Copying the active slice, multiplying it and writing it consume
    ``3 S_P`` visits, the full-state phase consumes D, and phase/index setup
    is charged ``2s+5`` scalar visits, including the two scalar
    exponentials: ``W_P = D + 3 S_P + 2s + 5``, ``B_P,scratch = 32 S_P``
    (the original active slice and its product). For a hopping,
    ``S_H = D/K``. The two saved complex slices occupy ``32 S_H``, and during
    either expression ``c*x0-1j*(s*x1)`` up to three complex slice
    temporaries coexist: ``W_H = 12 S_H + 8`` (two copies, both
    scale/phase/subtract expressions and writes, and the scalar
    trigonometry/support checks), ``B_H,scratch = 80 S_H``. For B_P projector
    and B_H hopping occurrences,
    ``W_onehot = W_start + sum_P W_P + sum_H W_H + D`` and
    ``B_onehot = B_held + H0 + max{B_start, 16D + max(max_P 32 S_P, max_H 80 S_H)}``.
    The last D is the final scalar phase pass. Empty maxima are zero. There
    is one state, and all per-block scratch is released before the next
    block. Empty support and zero angle return before these arrays, so such
    a block is charged only its scalar checks, the ``2s+5`` or 8 of its
    law. This route has no cached Pauli matrices, shifted-CSR workspace or
    expm call terms. ``method.restricted_sizes`` and
    ``verification._reference_sizes`` use this same law on the one-hot
    ``ir_product`` route.

    ``start_work`` and ``start_bytes`` are the start vector's work
    (``initial_state.start_vector_work``) and explicit payload
    (``initial_state.restricted_state``), and ``held_bytes`` the caller's
    already owned buffers that stay live (``B_held``). ``steps`` are the
    stored ``QHDBlock`` groups. ``H0`` is ``compiler.H0``.
    """
    from .compiler import H0

    k = num_grid_points
    hopping_slice = dimension // k
    work = start_work + dimension
    scratch = 0
    for group in steps:
        for block in group:
            if block.kind == "kinetic":
                if block.angle == 0:
                    work += 8
                    continue
                work += 12 * hopping_slice + 8
                scratch = max(scratch, 80 * hopping_slice)
            else:
                s = len(block.support)
                if s == 0 or block.angle == 0:
                    work += 2 * s + 5
                    continue
                active = dimension // k**s
                work += dimension + 3 * active + 2 * s + 5
                scratch = max(scratch, 32 * active)
    return work, held_bytes + H0 + max(start_bytes, 16 * dimension + scratch)

def run_binary_product(grid: OneHotGrid, method, reconstruction, state: np.ndarray) -> np.ndarray:
    """Apply the Plan's stored binary blocks to a restricted state in lexicographic order.

    The binary register holds every grid point, so the restricted state of
    length ``K**d`` is the whole register in the lexicographic order
    ``i = sum_j n_j K**(d-1-j)``. Reshaped to ``(2,)*(d b)`` in C order, axis
    ``j b + b - 1 - l`` carries bit l of ``n_j``, which the circuit holds on
    qubit ``j b + l``. This is the explicit variable-axis permutation between
    Qiskit's index ``z = sum_j n_j K**j`` and the lexicographic one
    (``binary`` module docstring). The kernel follows the selected block
    order (``native.append_binary_steps``), rebuilding each construction
    with ``binary.BinaryModel``. It applies the QFT's H, controlled-phase
    and swap operations on the variable's axes, and multiplies a diagonal
    by ``exp(i hat_phi)`` at the binary64 values returned by
    ``DiagonalConstruction.emitted_phases``. For Walsh synthesis those
    values are rounded reconstructions of the separate rotation phases,
    so this host calculation is a numerical reference for the selected
    blocks. A kinetic diagonal acts on variable j's Fourier-register
    axis of the ``(K,)*d`` view, and a potential diagonal on its support's
    axes. The caller restores the physical phase.

    ``binary_product_state_error`` bounds the roundoff and
    ``binary_product_sizes`` the work and bytes.

    In-place operations. The diagonal update
    ``new = old * table`` and ``np.multiply(old, table, out=old)`` have the
    same per-entry product when ``table`` does not alias the updated state.
    The shaped table broadcasts along the same axes, and the state stays
    complex128. The same phase array serves the equal potential halves of a
    step when its construction key is identical, and both applications still
    count. For an H, both original operands are kept until their sum and
    difference are formed: one explicit half-copy ``x0`` of the low slice,
    ``low = x0 + high`` and ``high = x0 - high``, then both are scaled by the
    same ``math.sqrt(0.5)``. The original high slice is untouched when both
    additions read it, so each output is the same ``RN(RN(x0 +/- x1)*scale)``
    as a copying update. QFT swap copies preserve values exactly.

    The caller transfers ownership of ``state``, a writable complex128
    buffer that it has just built (``method._evolve_restricted`` passes its
    new start vector), and the kernel updates it in place without copying
    it. The caller's reference to that buffer stays live through the call,
    so after a swap rebinds the working tensor the original buffer, the
    current tensor and the reordered copy can coexist
    (``binary_product_sizes``, ``i_start = 1``). Each diagonal's reshaped view
    is released right after its in-place multiplication, so no fourth state
    owner survives into a later swap, and the reservation is 48D.
    """
    from math import sqrt

    from .binary import BinaryModel, lexicographic_table
    from .method import _binary_source_reservation

    # The state buffers and the complex phase tables that stay live coexist with the model and its
    # construction cache. The swap path can hold three D-entry state owners (the caller's start
    # buffer, the current tensor and a new swap result), 48D bytes, because each diagonal's local
    # view is released right after its in-place multiplication. For the generated first/second-order
    # schedule the current step's potential exponentials, the previous kinetic factor and a preceding
    # block's table reference fit 16 (sum_potential E_t + K + E_max) bytes. The Plan's source records,
    # including its source tables once, stay live beside the model's address-ordered copies
    # (_binary_source_reservation).
    k = grid.num_grid_points
    entries = [t.values.array.size for t in reconstruction.support_values]
    specs = tuple((int(t.values.array.size), len(t.support)) for t in reconstruction.support_values)
    source, _ = _binary_source_reservation(
        grid.num_variables, k, specs, method.num_steps, method.trotter_order)
    held = (source + 48 * k ** grid.num_variables
            + 16 * (sum(entries) + k + max([k, *entries])))
    model = BinaryModel(grid, method, reconstruction.support_values, held_bytes=held)
    d, bits = grid.num_variables, model.bits
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * (d * bits))
    # fl(sqrt(1/2)) is correctly rounded, the H scale (binary_product_state_error).
    scale = sqrt(0.5)

    def axis(offset, level):
        return offset + bits - 1 - level

    def gate(item, offset):
        nonlocal tensor
        if item[0] == "h":
            a = axis(offset, item[1])
            low = [slice(None)] * tensor.ndim
            high = list(low)
            low[a], high[a] = 0, 1
            low, high = tuple(low), tuple(high)
            x0 = tensor[low].copy()
            np.add(x0, tensor[high], out=tensor[low])
            np.subtract(x0, tensor[high], out=tensor[high])
            tensor[low] *= scale
            tensor[high] *= scale
        elif item[0] == "cp":
            index = [slice(None)] * tensor.ndim
            index[axis(offset, item[2])] = index[axis(offset, item[3])] = 1
            tensor[tuple(index)] *= np.exp(1j * item[1])
        else:
            tensor = np.ascontiguousarray(np.swapaxes(tensor, axis(offset, item[1]), axis(offset, item[2])))

    for step in reconstruction.steps:
        # Phase arrays of this step's potential constructions by construction key, so
        # that the two equal halves of a second-order step share one array.
        potential_phases = {}
        for block in step:
            if block.kind == "binary_kinetic":
                built = model.construction(block.kind, block.variables, block.exponent, block.synthesis)
                factor = np.exp(1j * built.emitted_phases())
                (j,) = block.variables
                gate_offset = j * bits
                for item in model.forward:
                    gate(item, gate_offset)
                shape = [1] * d
                shape[j] = k
                view = tensor.reshape((k,) * d)
                np.multiply(view, factor.reshape(shape), out=view)
                del view
                for item in model.inverse:
                    gate(item, gate_offset)
            else:
                key = (block.variables, float(block.exponent).hex(), block.synthesis)
                table = potential_phases.get(key)
                if table is None:
                    built = model.construction(block.kind, block.variables, block.exponent, block.synthesis)
                    shape = [k if j in block.variables else 1 for j in range(d)]
                    table = lexicographic_table(np.exp(1j * built.emitted_phases()), len(block.variables),
                                                k).reshape(shape)
                    potential_phases[key] = table
                view = tensor.reshape((k,) * d)
                np.multiply(view, table, out=view)
                del view
    return tensor.reshape(-1)


def binary_product_state_error(bits: int, steps, *, cutoff, start) -> float:
    """First-order 2-norm error budget delta of ``run_binary_product`` from the Plan's start vector.

    The reference is the exact unitary product at the kernel's phase
    parameters: exact H gates, controlled phases at its floating-point
    angles, diagonals at the binary64 entries returned by
    ``DiagonalConstruction.emitted_phases``, and the caller's physical
    phase at its computed angle, applied to the exact initial state.
    The one-hot host budget similarly takes the analytic blocks at their
    stored angles as its reference (``onehot_product_state_error``).
    In-place execution changes the byte law, not these charges or their
    target.
    Assume finite intermediates, normal values wherever a relative-error
    model is used, and the scalar-function accuracy stated below. Each
    local state-application error is propagated by unitaries without
    changing its norm, so its first-order charges add, in units of u:

    - start vector: ``start``, the bound of the selected
      ``initial_state.restricted_state_error`` (2 for the uniform state),
      which planning stores as ``QHDReconstruction.initial_state_error``.
    - H on one axis: 3. ``fl((x0 +- x1) s)`` with ``s = fl(sqrt(1/2))`` has one
      rounding in the complex sum, one in each real product and one in s, so
      the pair moves by at most ``3u`` times its norm.
    - controlled phase and every diagonal of unit phases:
      ``GLOBAL_PHASE_STATE_ROUNDOFF`` = 5, for an entry ``exp(i phi)``
      accurate to 2u (one-ulp cos and sin) and a complex product of at most
      ``sqrt(2) gamma_2``.
    - swap: 0, a copy.
    - the caller's final physical-phase product: 5.

    A kinetic block has two QFTs of b H gates and ``C_p`` controlled phases
    (``binary.qft_controlled_phases``) and one diagonal, ``2 (3 b + 5 C_p) + 5``,
    and a potential block one diagonal, 5:

    ``delta = u (start + sum_blocks charge + 5)``.

    The phase parameters are the binary64 entries that
    ``DiagonalConstruction.emitted_phases`` returns. Taking them as exact
    inputs excludes the error of forming the Walsh ``identity_phase``, the
    coefficients and rotation angles, the ``walsh_sum`` reconstruction and
    the dense entries ``-x v[z]``, and every time, splitting, AQFT, pruning
    and grid error. The per-diagonal charge covers ``exp(i phi)`` and its
    multiplication into the state under the scalar-function premises above.
    It does not cover angle formation against an exact table or establish
    equality with the native product of the separately stored rotation
    angles. Kept-state phase admission adds the Walsh identity formation
    ``F_W`` and the reconstruction ``R_W`` to the constant-phase ledger
    (``binary.WalshPhaseTerms``, ``method._admit_compiled_phase``).
    The budget serves the kernel's mass window and most-probable tie
    window relative to this parameter-defined reference
    (``method._host_state_error``). Those windows do not add ``F_W`` or
    ``R_W`` and do not establish accuracy against native gate execution.
    """
    from nwqlib._validation import GLOBAL_PHASE_STATE_ROUNDOFF, UNIT_ROUNDOFF

    from .binary import qft_controlled_phases

    diagonal = GLOBAL_PHASE_STATE_ROUNDOFF
    # Two QFTs of b H (3u each) and C_p controlled phases (5u each), and one diagonal.
    kinetic = 2 * (3 * bits + diagonal * qft_controlled_phases(bits, cutoff)) + diagonal
    total = sum(kinetic if block.kind == "binary_kinetic" else diagonal for step in steps for block in step)
    # The start vector, the blocks, and the final physical-phase product.
    return (start + total + GLOBAL_PHASE_STATE_ROUNDOFF) * UNIT_ROUNDOFF


def binary_product_sizes(dimension: int, bits: int, variables: int, table_sizes, steps, *, cutoff,
                         swaps, walsh, start_work) -> tuple[int, int]:
    """Return ``(size_units, workspace_bytes)`` of one ``run_binary_product`` evolution.

    D is ``dimension = K**d``, ``table_sizes`` the ``2**n_S`` entries of each
    support table and ``steps`` the stored blocks. ``walsh`` is whether the
    model computes the support tables' Walsh coefficients, false under
    ``BinarySynthesis(potential="dense_diagonal")``, whose tables are built
    without them (``binary.PhaseTable.dense``), as the native charge
    (``method.QHD._select_binary_native``) also counts. The units count array
    element operations: ``start_work`` for forming the start vector once
    (``initial_state.start_vector_work``, D for the direct uniform fill),
    ``D`` for the final phase, ``d C(b)`` for the d kinetic tables that the
    model builds once (``binary.kinetic_setup_work``), ``(n + 1) 2**n`` per
    support table for the Walsh coefficients that the model computes once
    when ``walsh`` holds (the n transform levels and the normalization
    divisions of the unit rule stated at ``method.QHD._select_binary_native``), per
    kinetic block ``(2 (b + C_p + s) + 1) D`` for
    the two QFTs, whose H gates and swaps touch every entry and whose
    controlled phases a quarter of them (charged D each), and the diagonal,
    plus ``(b + 3) K`` for its phase table (the synthesis, a Walsh sum and
    the exponential), and per potential block ``D + (n + 3) 2**n`` for the
    multiplication and its phase table. ``C_p`` is
    ``binary.qft_controlled_phases`` and s the ``floor(b/2)`` swaps when
    selected. The Walsh term excludes the normalization-loss arithmetic,
    at most ``3 * 2**n`` operations per table for coefficients omitted by
    normalization, that the native charge adds to give ``(n + 4) 2**n``.

    Bytes. Binary state operations update the selected complex state
    in place. The state-array envelope is 32D bytes when all separate
    references to the initial buffer are released before swap rebinding, or
    48D when that original buffer remains live. Phase-table construction and
    caches are admitted separately. One live complex state is 16D. An H
    needs one half-state copy, 8D, giving 24D. A QFT swap can overlap the
    current state with a contiguous reordered copy, giving 32D. The
    start-vector construction's existing maximum, its explicit payload of at
    most ``24 D + 8 d K <= 32 D`` bytes (``initial_state.restricted_state``),
    is also at most 32D. These phases are sequential. The final phase and
    diagonal multiply use ``out=`` and have no D-entry product array. The
    law is ``B_binary = (32 + 16 i_start) D + 64 (sum_S K**|S| + d K)``, where
    the 64 bytes per table entry, for the d kinetic tables and every support
    table, cover the stored values, coefficients and weights and a
    construction's phases, exponentials and Walsh sum under the existing
    table/construction-lifetime convention. ``i_start = 1``: the caller
    ``method._evolve_restricted`` keeps its reference to the start vector
    through the call, so the original buffer can remain live after a swap
    rebinds the working tensor. A verified release of that reference would
    allow ``i_start = 0``.
    """
    from .binary import kinetic_setup_work, qft_controlled_phases

    k = 1 << bits
    controlled = qft_controlled_phases(bits, cutoff)
    swap_count = bits // 2 if swaps else 0
    per_kinetic = (2 * (bits + controlled + swap_count) + 1) * dimension + (bits + 3) * k
    # Start vector and final phase, and (n + 1) 2**n for the Walsh transform of each table that has one,
    # without the normalization-loss arithmetic that _select_binary_native charges.
    size = start_work + dimension + variables * kinetic_setup_work(bits)
    size += sum(entries.bit_length() * entries for entries in table_sizes) if walsh else 0
    for step in steps:
        for block in step:
            if block.kind == "binary_kinetic":
                size += per_kinetic
            else:
                entries = k ** len(block.variables)
                size += dimension + (entries.bit_length() + 2) * entries
    # B_binary with i_start = 1: the caller keeps its start vector live (docstring).
    start_live = 1
    workspace = (32 + 16 * start_live) * dimension + 64 * (sum(table_sizes) + variables * k)
    return size, workspace
