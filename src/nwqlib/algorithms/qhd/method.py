"""QHD Method: one-hot or binary grid planning, native or classical execution, and result analysis.

The Method evolves the time-dependent Hamiltonian of Leng et al., arXiv:2303.01471v1, Eq. (1), p. 3,
``H(t) = exp(phi_t) (-Delta/2) + exp(chi_t) f(x)``, written here as ``H(t) = a(t) K + b(t) V`` with the
schedule's kinetic and potential weights a and b (``schedules``), K the kinetic operator on a finite grid
of the box and V the diagonal of objective values. Leng et al.'s Algorithm 1, p. 44, is the reference
workflow. It meshes the box, prepares an initial state, applies one product-formula step per time step
and measures the position. The QHD guide, ``docs/algorithms/qhd.md``, section "Source map", names the
paper, version and equation behind each step, or marks the step as standard or NWQLib's own.

NWQLib departs from Algorithm 1 in these choices, each stated with its reason at its owner.

- Register. Algorithm 1 gives each variable a q-qubit register whose basis states enumerate its
  ``2**q`` mesh points. The default one-hot register uses K qubits per variable, one per grid point
  (Wu et al., arXiv:2605.12066v1, Sec. IV.A, Eqs. (9)-(10), p. 5). ``encoding="binary"`` is
  Algorithm 1's register on its periodic mesh (Eq. (E.1), p. 43), with the kinetic factor applied by a
  QFT conjugation, as Eq. (E.4), p. 44, applies it with the shifted Fourier transform
  (``QHD.encoding``, ``binary``).
- Kinetic operator and boundary. Algorithm 1 assumes a periodic box and the Fourier eigenvalues of
  ``-Delta/2``, which ``kinetic_model="spectral"`` selects. The default kinetic operator is the
  finite-difference stencil of Eqs. (F.7) and (F.9), pp. 46-47, because every route applies it
  (``QHD._kinetic_route``). The default boundary is Dirichlet, because it admits the default K = 2
  with the one-hot encoding (``QHD._periodic_grid``), and its default interior grid has the vanishing
  boundary values of Eq. (F.4).
- Time sampling and order. Algorithm 1 applies the potential and then the whole kinetic factor with
  weights at the left endpoint ``t_j = j s``, a first-order product. Every route takes the weights at
  the step midpoint or averages them over the step (``QHD.coefficient_rule``), because left-endpoint
  weights leave a product first order in the step even with symmetric placement (guide, "Split-step
  classical flavor"). The block order shared by circuit selection and the numerical
  ``ir_product`` reference is symmetric and second order by default
  (``QHD.trotter_order``). The compiled one-hot products further
  split the kinetic factor into link exponentials, which give local hopping gates. Adjacent links do not
  commute. First order applies them in increasing link order, and second order uses odd half, even
  full, odd half layers (``kinetic.KineticCompiler``). This is an additional product approximation to
  the whole kinetic exponential in Algorithm 1. Compiled binary kinetic factors use a Fourier
  conjugation of a phase diagonal, with synthesis approximations charged separately. The ``split_step``
  flavor always applies a symmetric potential/whole-kinetic product, and the default ``schrodinger``
  flavor applies the exponential of each step's full Hamiltonian through ``expm_multiply``.
- Initial state. Algorithm 1, step 3, suggests the uniform or a Gaussian state. The default is the
  kinetic ground state, for the reasons in the comment above ``QHD.initial_state``.
- Readout. Algorithm 1, step 6, measures one grid point. Analysis reports the best observed candidate
  and the most probable point of the observed population, with invalid one-hot outcomes kept as
  invalid mass (``_summarize``).

Reading order of the module. ``QHD.plan`` evaluates each objective support table once, compiles the
step blocks when a circuit or the ``ir_product`` flavor needs them, stores both in
``records.QHDReconstruction`` and binds the native block (``native.construct_qhd``) or the classical
kernel (``_execute_theory``). ``_constant_phase`` to ``_admit_compiled_phase`` bound the rounding of
the global phase that a kept state (``QHD.keep_state``) carries. ``_host_state_error`` to
``_readout_window`` bound the roundoff of the probabilities that the readout compares. ``_summarize``
and ``QHD.analyze`` turn observed probabilities or counts into the reported points.
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

from contextvars import ContextVar
from fractions import Fraction
from math import fsum, inf, isfinite, nextafter, pi
from typing import Annotated, Literal, ClassVar
import sympy as sp
from pydantic import Field, StrictBool, field_validator, model_validator
from nwqlib._validation import (
    ABSOLUTE_SQUARE_ROUNDOFF,
    COMPONENT_SQUARES_ROUNDOFF,
    NUMERICAL_RELATION_RTOL,
    UNIT_ROUNDOFF,
    expm_multiply_state_error,
    integer,
    probability_difference_window,
    state_mass_window,
    validate_normalized_mass,
)
from nwqlib.algorithms.protocol import Method, AlgorithmDescriptor, ApplicabilityError
from nwqlib.artifacts import ArrayOutput
from nwqlib.blocks import SelectedBlock, SelectedConstruction
from nwqlib.blocks.kernels import BoundKernel, KernelOutput
from nwqlib.blocks.records import BlockSemantics, SelectedDefinition, SelectedKernel
from nwqlib.core.planning import Plan, Experiment, ObservationSpec, ReadoutDetails
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.core.records import Basis, Float64, InputRef, Nonnegative, PositiveInt, Real, Source
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.evidence.error_model import ErrorModel, ErrorTerm, FramedFact
from nwqlib.execution import (
    KernelApplication,
    ObservationView,
    RegisterMap,
    ScalarValue,
)
from nwqlib.ir import (
    Allocate,
    Binding,
    BlockCall,
    BlockSignature,
    ClassicalStage,
    ClassicalValue,
    Definition,
    Measure,
    MeasurementBatch,
    MetadataRef,
    PortMap,
    Program,
    QuantumPort,
    Register,
    Release,
    Sequence,
    Setting,
)
from nwqlib.ir.validation import _Admission
from nwqlib.problems.records import Optimization, OptimizationCandidate
from nwqlib.resources.records import ResourceLaw, Workspace
from .binary import BinarySynthesis
from .compiler import QHDCompiler
from .decoding import decode_onehot_basis_index
from .evolution_bounds import _upward
from .grid import OneHotGrid
from .initial_state import (
    InitialState, KineticGroundState, UniformState, chain_links, chain_selection, evaluate as evaluate_initial_state,
    start_vector_work, stored_amplitudes,
)
from .objective import ObjectiveDecomposer, centered_objective, expansion_centers, monomial_bound, node_count
from .potential import coerce_real_scalar
from .schedules import QuadraticSchedule, Schedule
from .validation import DEFAULT_ROTATION_THRESHOLD
from .records import (
    DESCRIPTOR,
    METHOD,
    QHDAnalysis,
    QHDBlock,
    QHDRangeOmissions,
    QHDReconstruction,
    QHDTableOmission,
    QHDWalshPhase,
    SupportValues,
    objective_reference,
    validate_selection,
)

_ANALYSIS_SOURCE = Source(
    name="qhd.analysis",
    version="4",
    domain="observed grid-point population",
    reference="nwqlib.algorithms.qhd.method",
)


# Names of the QHD.boundary values in the Plan's assumptions and the native
# block's relation text. Dirichlet is a proper name.
_BOUNDARY_NAMES = {"dirichlet": "Dirichlet", "periodic": "periodic"}

# Planning allowance of the initial state, measured and implementation
# specific, not derived: Python 3.12.14, NumPy 2.5.2, 64-bit CPython on macOS
# arm64 (2026-09-26). The per-point peaks were traced at NWQLib commit
# 781239621a49b8f682f3c36eab546edde25d510b and the K = 2 trace at
# 651c725ba2e3557ca25466653bd1daa14c189ff4, branch commits that develop holds
# as a478113d15ff235832a5896280baa14732763ac2 and
# f0da79ccc545d11664fd0938845d56c340a97ed7. The component per grid point of
# each variable, _INITIAL_STATE_BYTES, covers the evaluator's Python float
# lists and float64 vectors. GaussianState's evaluation, which also forms its
# entry-error bounds, had a traced peak of about 277 bytes per point for one
# variable with K from 4096 to 1e6, and the kinetic ground state's 73. That component
# alone does not cover the fixed costs of small grids. At d = 1, K = 2 the
# Gaussian planning stage (evaluate and stored_amplitudes, after one warm-up
# call) traced 1,505 bytes against 640. _initial_state_bytes therefore adds
# the normalized arrays, the stored payload and _INITIAL_STATE_OBJECT_BYTES of
# untuned Python and NumPy bookkeeping per variable plus one. Registered in
# docs/ENGINEERING_CONSTANTS.md ("QHD selected operation sizes"). Requalify
# when the evaluator, the stored representation, the object lifetimes or the
# Python runtime changes.
_INITIAL_STATE_BYTES = 320
_INITIAL_STATE_OBJECT_BYTES = 4096

# The wrapper of one stored initial vector (core.records.FrozenArray), the
# engineering allowance per vector beyond its owning array's data and header:
# the object with its two slots, 48 bytes, and the read-only view of the
# owning array that it keeps, 112 bytes, measured with sys.getsizeof on
# 64-bit CPython 3.12.14 with NumPy 2.5.2, and its cached hash, once
# computed, a Python integer of at most 63 magnitude bits charged
# L(63) = 32 + 4 ceil(63/30) = 44 bytes. The owning-array formula of
# _initial_state_bytes counts one ndarray header per vector, so the view's
# 112 bytes are charged here. The allowance holds for each distinct stored
# rank-one vector, and with the owning arrays it gives the stored-population
# allowance 8dK + 324d + 64. Registered in docs/ENGINEERING_CONSTANTS.md
# ("Budgets and mechanical bounds"). Requalify when FrozenArray's storage or
# the runtime changes.
_FROZEN_ARRAY_BYTES = 48 + 112 + 44

# Planning allowances per schedule step and per compiled block occurrence,
# measured and implementation specific, not derived: Python 3.12.14,
# Pydantic 2.13.5, NumPy 2.5.2, 64-bit CPython on macOS arm64 (2026-09-26),
# traced at NWQLib commit 432a54719e47a36878045d17fcbe3f38ea84e7f3, a branch
# commit that develop holds as 2c0c54f88f1e0f8d2b0ff8415adc462190957ce3.
# A Plan keeps per step its (t, a, b) row in the compiler and in
# QHDReconstruction.step_weights, and the content identities of the
# reconstruction and the Plan write each row into two JSON copies
# (core.records.Record). At the traced commit, Schrodinger planning also held
# one (N, r) generator norm per step, which _generator_norms now yields one at
# a time, so the schrodinger slope below includes about 80 bytes per step
# that current planning does not hold.
# The slope of the traced peak of plan() between 2,000 and 4,000 steps, after
# one warm-up Plan, was 404 to 409 bytes per step for split_step and 492 to
# 497 for schrodinger, over d = 1 and 2, K = 4, the three schedules and both
# coefficient rules. _STEP_BYTES charges 768. A compiled block occurrence
# (compiler block, QHDBlock or QHDBinaryBlock record and its identity JSON)
# traced 3,257 to 3,506 bytes on the one-hot encoding and 2,446 to 2,468 on
# the binary encoding, the ir_product slope between 500 and 1,000 steps less
# the split-step rows, per block of a step, over d = 1 to 4, K = 2 to 16 and
# projector supports of up to 4 qubits. _BLOCK_BYTES charges 4096. Both are
# admitted before the compiler forms any row or block
# (QHD._admit_symbolic_work), so that a step or block count whose records
# would exceed max_bytes is refused before any of them exists. Registered in
# docs/ENGINEERING_CONSTANTS.md ("Budgets and mechanical bounds"). Requalify when the rows, the block
# records, their identity encoding or the Python runtime changes.
# The admitted symbolic work of one (Method, problem, compact) that the augmented-Lagrangian layer has
# already admitted for a candidate (constrained._planned_costs): the planning of that candidate reuses the
# admission, its decomposer and chunks instead of expanding the objective again (QHD._admit_symbolic_work).
_ADMITTED_SYMBOLIC = ContextVar("qhd_admitted_symbolic", default=None)

_STEP_BYTES = 768
_BLOCK_BYTES = 4096

# Readout allowance per published host-kernel scalar, measured and
# implementation specific, not derived, in the runtime and at the commit
# above. _execute_theory
# returns d K + 3 d + 13 ScalarValue records, one per marginal entry, index
# and summary value (_scalar_names). After one warm-up call, the traced
# memory still held by its output grew by 594 bytes per scalar between K =
# 1,024 and 8,192 at d = 1, and averaged 648 bytes per scalar at K = 65,536.
# _SCALAR_BYTES charges 1024 in the kernel's admission
# (QHD._host_construction). Registered in docs/ENGINEERING_CONSTANTS.md
# ("Budgets and mechanical bounds"). Requalify when the kernel's outputs or
# the record implementation changes.
_SCALAR_BYTES = 1024

# Fixed allowance per host-kernel call, measured and implementation specific,
# not derived: Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1, Pydantic 2.13.5,
# 64-bit CPython on macOS arm64 (2026-09-27), traced with the streaming
# change of commit 6dcc5af2 on an earlier develop base, and rechecked at
# 6af8d3b4764fcf0247d88ff84cb99fcf6b15b03a on develop
# 67e361216aa8950217799231caa4a0adc441510d on the same cases except one-hot
# ir_product at K >= 32. Besides the arrays and records
# that the kernel laws count, one _execute_theory call holds Python objects
# of bounded total size: the fixed objects of the readout and of the returned
# records and, on the schrodinger flavor's expm_multiply calls, the 2-element tuples of SciPy's
# sparse shapes that CPython keeps on its free list, at most 2,000, which
# tracemalloc counts as about 112 KB. The kernel reads its generator norms
# and compiled blocks one at a time (_generator_norms, native.raw_blocks), so
# none of this grows with the step or block count. Measured as for
# _SCALAR_BYTES, after one warm-up call, the traced peak of _execute_theory
# exceeded the workspace that the kernel's size laws admit, for split_step
# less its whole transform-scratch allowance
# (split_step.SCRATCH_BYTES_PER_POINT and SCRATCH_BYTES_PER_AXIS), by at
# most 156,775 bytes over the schrodinger, split_step, one-hot ir_product and
# binary ir_product kernels. The cases were d = 1 and 2 with K = 2 to 64 and
# 2 to 2,048 steps, with the grid size times the step count at most 2**16
# for one-hot ir_product up to 2**22 for split_step, the initial states,
# keep_state and both coefficient rules at 32 and 512 steps, d = 3 with
# K = 2 and 4, d = 4 with K = 2, and up to 32,768 steps for schrodinger and
# binary ir_product at d = 1. The one-hot ir_product cases were traced with an
# earlier kernel that evaluated each block with expm_multiply; the direct block
# actions of theory._run_ir_product have not been traced against this
# allowance. _KERNEL_CALL_BYTES charges 262,144 once in the kernel's admission
# (QHD._host_construction). Registered
# in docs/ENGINEERING_CONSTANTS.md ("Budgets and mechanical bounds").
# Requalify when the kernel, its outputs, SciPy or the Python runtime changes.
_KERNEL_CALL_BYTES = 262144


def _initial_state_bytes(d, k):
    """Return the bytes that planning admits for evaluating and storing the initial state of d variables on K points.

    Store the initial amplitudes as one immutable float64 vector per
    variable, preserving the selected entries and axis order. Charge each
    distinct data buffer and its array and container headers once.

    ``320 d K + B_A + B_cache + d F + 4096 (d + 1)``, with
    ``B_A = 8 d K + 120 d + 40`` for the normalized float64 vectors and their
    tuple, and ``B_cache = 8 d K + 120 d + 64`` for the stored payload
    (``QHDReconstruction.initial_amplitudes`` and ``initial_state_error``): d
    separately owning one-dimensional float64 arrays of length K, which
    counts data 8dK, ndarray headers 112d, the outer tuple 8d + 40 and the
    existing 24-byte error float. The record copies the evaluated vectors
    into its own arrays (``core.records.FrozenArray``), so both populations
    exist and B_A and B_cache are both charged. ``F = _FROZEN_ARRAY_BYTES``
    is the measured engineering allowance for each vector's FrozenArray
    wrapper. The stored initial vectors cost ``8dK + 324d + 64`` bytes on
    the qualified runtime, including each owning array, its separate
    read-only view, the FrozenArray wrapper, a cached-hash allowance, the
    outer tuple and the error scalar. Revalidated wrappers share the
    existing array storage. The first and last terms are the measured evaluator
    coefficient and the untuned bookkeeping allowance
    (``_INITIAL_STATE_BYTES``, ``_INITIAL_STATE_OBJECT_BYTES``), so the
    allowance is measured and specific to the implementation and runtime
    named there, not a derived bound. The early check of ``QHD.plan`` and
    the final admission of ``QHD._admit_symbolic_work`` include the same
    value.
    """
    arrays = 8 * d * k + 120 * d + 40
    stored = 8 * d * k + 120 * d + 64 + _FROZEN_ARRAY_BYTES * d
    return _INITIAL_STATE_BYTES * d * k + arrays + stored + _INITIAL_STATE_OBJECT_BYTES * (d + 1)


def _binary_source_reservation(d, k, table_specs, num_steps, trotter_order, *, grouped_terms=0):
    """Return the source-record and planning byte allowances of a binary Plan.

    For T potential tables, M support positions and f potential passes per
    step, B=N*(d+f*T) block occurrences contain J=N*(d+f*M) positions.
    H_tables=65536+6144*T+(24+L(bit_length(max(0,d-1))))*M+256*d is the
    qualified table/metadata inventory of _support_table_bytes. Add the
    source payload 8*sum(E), stored initial vectors 8*d*k+324*d+64,
    _STEP_BYTES*N, _BLOCK_BYTES*B and (32+L+12*digits(max(0,d-1)))*J
    for variable-sized block support slots and their identity encoding.
    The source allowance includes each source table payload once.

    Planning also holds its normalized initial arrays, 8*d*k+120*d+40,
    and 16 bytes per position in the decomposer's grouped-term lists.
    It reserves the future stored vectors as well. Symbolic expression
    payloads have the existing SymPy exclusion. Identity array encoding
    has its separate successive-phase workspace I. These are qualified
    object allowances, not a process-RSS bound.
    """
    from .binary import _integer_storage

    table_specs = tuple((int(e), int(s)) for e, s in table_specs)
    t = len(table_specs)
    indices = sum(s for _, s in table_specs)
    l_d = _integer_storage(max(0, d - 1).bit_length())
    f = 2 if trotter_order == 2 else 1
    blocks = num_steps * (d + f * t)
    block_indices = num_steps * (d + f * indices)
    table_metadata = 65536 + 6144 * t + (24 + l_d) * indices + 256 * d
    table_payload = 8 * sum(e for e, _ in table_specs)
    stored_initial = 8 * d * k + 324 * d + 64
    support_extension = (32 + l_d + 12 * len(str(max(0, d - 1)))) * block_indices
    records = _STEP_BYTES * num_steps + _BLOCK_BYTES * blocks + support_extension
    source = table_payload + table_metadata + stored_initial + records
    raw_initial = 8 * d * k + 120 * d + 40
    return source, source + raw_initial + 16 * int(grouped_terms)


def _support_table_bytes(max_bytes, d, k, held, nodes):
    """Return ``(F_tables, W_eval, I, chunks)``, the byte phases of the support tables of ``QHD._admit_symbolic_work``.

    Admit the final float64 tables, centered coordinates, support metadata
    and the largest live construction workspace, including the source array
    and finite-value mask during freezing. The metadata allowance covers the
    compiler's shared arrays and both SupportValues populations created
    while the reconstruction is validated.

    For T tables with M total support indices in d variables, the qualified
    metadata allowance is ``65536 + 6144T + (24 + L(b_d))M + 256d`` bytes,
    where ``b_d = bit_length(max(0, d - 1))`` and
    ``L(b) = 32 + 4 ceil(max(1,b)/30)``. Array data are counted separately
    and shared buffers count once. Its inventory: ``2 * 2048 T`` for the two
    SupportValues populations (per record the object, field dictionary and
    field-name set, the wrapper, a 44-byte hash allowance, a 112-byte cached
    identity string, three 24-byte extrema and a 40-byte support-tuple
    header), ``2048 T`` for one shared owning-array/view header pair, the
    compiler's wrapper and hash, the source support-tuple header and
    table-related mapping and sequence slots, ``24 M`` for the slots of at
    most three support tuples per table, ``L(b_d) M`` for their integer
    referents, ``256 d`` for the coordinate-array headers and their sequence
    slots, and H0 for fixed bookkeeping. These object sizes are qualified for
    64-bit CPython 3.12.14, NumPy 2.5.2 and Pydantic 2.13.5 and are
    registered in docs/ENGINEERING_CONSTANTS.md ("Budgets and mechanical
    bounds").

    Evaluation phase. With ``n_S = K**|S|`` and ``E = sum_S n_S``,
    ``F_tables = 8E + 8dK + H_tables`` and
    ``W_eval = max(40K if T else 0, max_S 9 n_S, max_S B_slab,S(b_S))``,
    with empty maxima zero. Freezing a float64 table keeps its source
    ``8 n_S`` bytes, its new owner and an ``n_S``-byte ``isfinite`` mask
    simultaneously, so after the final 8E are reserved the snapshot term is
    ``9 n_S``. Each centered axis is built from a Python list of K floats,
    ``40K`` bytes (24 per float and 16 per list slot), one axis at a time.
    With ``R = max_bytes - B_held - F_tables``, the chunk b_S of each support
    is the largest with ``B_slab,S(b_S) <= R`` (``compiler.support_chunk_size``),
    and reducing the chunk cannot repair an unaffordable snapshot or staging
    term. When R cannot hold one entry, the chunk is one entry and the
    admission refuses the Plan.

    Identity phase. For each serialized array occurrence i with raw length
    ``b_i``, ``q_i = 4*((b_i+2)//3)``, ``Q = sum(q_i)`` and ``q_max = max(q_i)``.
    The first identity read of the reconstruction serializes its tables and
    its d initial vectors of length K, with

    ``J_S = 227 + digits(n_S) + sum_(j in S) digits(j) + max(0, s_S - 1) + 4*((8*n_S + 2)//3)``,
    ``J_tables = 2 + max(0, T - 1) + sum_S J_S``,
    ``q_K = 4*((8*K + 2)//3)`` and
    ``J_initial = 2 + max(0, d - 1) + d*(36 + digits(K) + q_K)``,

    plus 40 bytes for their field keys and two separators, and
    ``Q = sum_S 4*((8*n_S + 2)//3) + d*q_K``. The serialization workspace is
    bounded by ``H_json + Q + q_max + 12J + U_other`` on the qualified
    runtime, where J bounds the complete UTF-8 JSON length, Q is the sum of
    padded-base64 lengths, ``q_max`` is their maximum, ``H_json`` covers
    portable containers and headers, and ``U_other`` covers a temporary
    escaped non-array token. The coefficient 12 permits four-byte Unicode
    storage of the complete JSON text. For the arrays,
    ``H_json,arrays = H0 + 2048 (T + d) + (8 + L(b_d)) M`` and J is the
    arrays' part ``J_tables + J_initial + 40``. The other fields of the
    reconstruction and their non-array tokens keep their existing identity
    allowances (``_STEP_BYTES``, ``_BLOCK_BYTES``). The returned I is the
    arrays' serialization workspace.

    The caller admits ``B_held + F_tables + max(W_eval, I)``: the evaluation
    and identity workspaces are successive, and B_held covers the other data
    of both phases.

    Raises:
        ValueError: A table has more entries than a NumPy index can address.
    """
    import numpy as np

    from .compiler import H0, support_chunk_size, support_workspace

    def digits(x):
        return len(str(x))

    def base64_length(n):
        # q = 4*((b + 2)//3) padded base64 characters for b = 8 n raw bytes
        return 4 * ((8 * n + 2) // 3)

    sizes = {support: k ** len(support) for support in nodes}
    tables, indices = len(sizes), sum(len(support) for support in sizes)
    entries = sum(sizes.values())
    bits = max(0, d - 1).bit_length()
    # L(b) = 32 + 4 ceil(max(1, b)/30)
    referent = 32 + 4 * -(-max(1, bits) // 30)
    metadata = H0 + 6144 * tables + (24 + referent) * indices + 256 * d
    # F_tables = 8E + 8dK + H_tables, reserved first.
    reserved = 8 * entries + 8 * d * k + metadata
    remaining = max_bytes - held - reserved
    workspace, chunks = (40 * k if tables else 0), {}
    for support, count in nodes.items():
        size = sizes[support]
        if size > np.iinfo(np.intp).max:
            raise ValueError(f"the support table of {support} has {size} entries, more than a NumPy index "
                             "addresses")
        rate, fixed = support_workspace(count)
        try:
            chunk = support_chunk_size(size, len(support), remaining, rate, fixed, H0)
        except ValueError:
            chunk = 1
        chunks[support] = chunk
        # max(40K, max_S 9 n_S, max_S B_slab,S(b_S))
        workspace = max(workspace, 9 * size, (rate + 8 * len(support) + 64) * chunk + fixed + H0)
    j_tables = 2 + max(0, tables - 1) + sum(
        227 + digits(size) + sum(digits(j) for j in support) + max(0, len(support) - 1) + base64_length(size)
        for support, size in sizes.items())
    j_initial = 2 + max(0, d - 1) + d * (36 + digits(k) + base64_length(k))
    lengths = [base64_length(size) for size in sizes.values()] + [base64_length(k)] * d
    # H_json,arrays + Q + q_max + 12 J for the arrays' part J of the identity JSON
    identity = (H0 + 2048 * (tables + d) + (8 + referent) * indices + sum(lengths) + max(lengths)
                + 12 * (j_tables + j_initial + 40))
    return reserved, workspace, identity, chunks


def _grid(plan):
    """Return the Plan's grid, built once from its Problem bounds and Method grid options.

    ``OneHotGrid`` owns the coordinates and spacing of both encodings. The
    binary register places variable j on qubits ``j b`` to ``j b + b - 1``
    instead of ``j K`` to ``j K + K - 1`` (``binary`` module docstring).
    The validated immutable grid is kept in the Plan's private cache with the
    options it was built from, and reused while they are equal, so caching
    changes no coordinate.
    """
    key = (plan.problem.variable_names, plan.problem.bounds, plan.method.num_grid_points,
           plan.method.include_boundary_points, plan.method.boundary)
    cached = plan._cache.get("qhd_grid")
    if cached is not None and cached[0] == key:
        return cached[1]
    grid = OneHotGrid(*key)
    plan._cache["qhd_grid"] = (key, grid)
    return grid


def _bits(method):
    """Return b with ``K = 2**b`` for the binary encoding, or None for the one-hot encoding."""
    return method.num_grid_points.bit_length() - 1 if method.encoding == "binary" else None


def _admit_register_basis(width):
    """Raise at planning when the native register's basis dimension ``2**width`` cannot enter a content identity.

    Native execution records the register basis, ``2**width`` states, as the
    integer ``Basis.dimension`` of its selected block and of a kept state, and
    a record's content identity writes every integer as decimal text. Python
    refuses to convert an integer with more decimal digits than
    ``sys.get_int_max_str_digits()``, 4300 by default and no limit when it is
    0. ``2**width`` has at most L digits exactly when ``2**width < 10**L``.
    Every width below 3L passes without forming ``10**L``, because
    ``2**(3L) = 8**L < 10**L``. At the default limit the first refused
    register has 14,285 qubits, whose 2**14285 states have 4301 digits.
    Classical execution evolves the
    ``K**d`` grid amplitudes and records no register basis, so it is not
    checked here.
    """
    import sys

    limit = sys.get_int_max_str_digits()
    if limit and width >= 3 * limit and 1 << width >= 10**limit:
        raise ValueError(
            f"native execution records its register of {width} qubits by its 2**{width} basis states, "
            f"which have more than {limit} decimal digits, the integer-to-text limit of this Python process "
            "(sys.get_int_max_str_digits()), so the Plan cannot receive its content identity. The binary "
            "encoding on the periodic grid uses d log2(K) qubits, and classical execution evolves the K**d "
            "grid amplitudes without a register basis"
        )


def _decoder(plan):
    """Return the map from a full-register basis index to its grid tuple, None for an invalid one-hot outcome."""
    grid, bits = _grid(plan), _bits(plan.method)
    if bits is None:
        return lambda index: decode_onehot_basis_index(index, grid)
    from .binary import decode_register_index

    return lambda index: decode_register_index(index, grid.num_variables, bits)


def _register_indices(plan):
    """Return the full-register basis index of each grid point in lexicographic order.

    ``state[_register_indices(plan)]`` restricts a native state to the grid
    points in the order of the classical kernels, the one-hot states
    (``theory.onehot_basis_indices``) or the binary permutation
    ``P|i> = |z>`` (``binary.lexicographic_register_indices``).
    """
    grid, bits = _grid(plan), _bits(plan.method)
    if bits is None:
        from .theory import onehot_basis_indices

        return list(onehot_basis_indices(grid))
    from .binary import lexicographic_register_indices

    return lexicographic_register_indices(grid.num_variables, bits)


def _table_values(reconstruction, indices, k):
    """Return the value of each stored support table at grid tuple ``indices``.

    No SymPy expression or native code is evaluated. A table for support
    ``(j_1, ..., j_s)`` stores values in C order with ``j_1`` most
    significant, the order in which the compiler evaluated them, and each
    value is read by its index as a Python float.
    """
    values = []
    for table in reconstruction.support_values:
        position = 0
        for variable in table.support:
            position = position * k + indices[variable]
        values.append(table.values.array.item(position))
    return values


def objective_at(reconstruction, indices, k):
    """Return the objective at grid tuple ``indices``, the ``fsum`` of the constant and the support tables.

    Readout and records use this value, constant included. The classical
    evolution uses the support tables alone (``split_step.potential_diagonal``).
    """
    values = [reconstruction.constant, *_table_values(reconstruction, indices, k)]
    return coerce_real_scalar(fsum(values), context="reconstructed objective")


def _constant_phase(reconstruction, method):
    """Return the angle phi of the global phase ``exp(i phi)`` that the objective constant c contributes.

    ``H(t) = a(t) K + b(t) (V_0 + c I)``, with ``V_0`` the support tables, and
    ``c I`` commutes with every operator, so each step's exponential factors
    exactly. The ``schrodinger`` step ``exp(-i dt (a_k K + b_k V))`` becomes
    ``exp(-i c dt b_k) exp(-i dt (a_k K + b_k V_0))``, and the ``split_step``
    step, whose two potential halves each carry ``dt b_k/2``, likewise gains
    ``exp(-i c dt b_k)``. Over all steps, with the step weights of the
    coefficient rule, ``phi = -c sum_k dt b_k``. Both flavors evolve ``V_0``
    (``split_step.potential_diagonal``), and
    ``_execute_theory`` multiplies the kept state by ``exp(i phi)`` once,
    after the probabilities are read, so they do not depend on c. Each term
    of the computed phi, ``-c fsum(fl(dt b_k))``, passes through its product
    rounding, the rounding of the accurate sum and the final multiplication
    by c. With ``S_C = |c| sum_k |dt b_k|``, and assuming that ``math.fsum``
    is correctly rounded, normal relative-error arithmetic and finite
    intermediates, expanding those three rounding factors puts phi within
    ``(3u + 3u**2 + u**3) S_C`` of the exact angle, which is ``3u S_C`` to
    first order (``_admit_constant_phase``). It is a global phase, and the
    multiplication adds at most ``GLOBAL_PHASE_STATE_ROUNDOFF`` u to the
    state. The ``ir_product`` flavor already restores the compiler's
    physical phase, which includes the constant. Each product ``fl(dt b_k)``
    passes the range rule (``validation._normal_range``), which supplies the
    normal-arithmetic premise. A finite final product that would be nonzero
    and below ``2**-1022`` is omitted as phase zero, and its exact value
    ``|c fsum(fl(dt b_k))|`` is charged by ``_admit_constant_phase``, since
    ``|exp(i a) - 1| <= |a|``. An overflowing sum or product is left to
    ``_admit_constant_phase``, which names the constant as its cause.
    """
    return _constant_phase_and_omission(reconstruction.constant, reconstruction.step_weights, method)[0]


def _constant_phase_and_omission(c, step_weights, method):
    """Return ``(phi, omitted)`` of ``_constant_phase``, with ``omitted`` the exact omitted ``|c fsum(fl(dt b_k))|``."""
    from .validation import _normal_product, _underflows

    dt = method.total_time / method.num_steps
    total = fsum(_normal_product(dt, b, "the potential exponent dt b_k of a step") for _t, _a, b in step_weights)
    # phi = -c sum_k dt b_k
    phi = -c * total
    if isfinite(phi) and _underflows(phi, "the objective constant's phase -c sum_k dt b_k",
                                     lambda: -Fraction(c) * Fraction(total), zero=c == 0 or total == 0):
        return 0.0, abs(Fraction(c) * Fraction(total))
    return phi, Fraction(0)


def _admit_constant_phase(reconstruction, method):
    """Raise at planning when a kept classical state cannot carry the objective constant's phase.

    A ``schrodinger`` or ``split_step`` Plan with ``keep_state`` multiplies
    its final state by ``exp(i phi)`` (``_constant_phase``). A finite c with
    finite step weights can still give a phi beyond the binary64 range, for
    example ``c = 1e308`` over two unit steps, and the kept state would then
    be NaN after the whole evolution. Otherwise phi lies within the finite
    allowance ``(3u + 3u**2 + u**3) S_C`` of the exact angle, with
    ``S_C = |c| sum_k |dt b_k|``, under the premises of ``_constant_phase``,
    among them a correctly rounded ``math.fsum``. When the final product was
    omitted below the normal range, its exact magnitude is added to the
    allowance, since it replaces a product whose rounding the allowance
    already covers. Once that allowance
    reaches pi it gives no nontrivial guarantee on ``exp(i phi)``, which
    does not show that the computed phase is wrong. Both cases are rejected
    before any evolution. The allowance is evaluated upward at every
    operation (``_add_up``, ``_mul_up``, ``_sum_up``), so rounding cannot
    move an allowance above pi below the threshold. The probabilities are
    read before the phase is applied, so the same objective is admitted
    without ``keep_state``.
    """
    c = reconstruction.constant
    u = UNIT_ROUNDOFF
    dt = method.total_time / method.num_steps
    try:
        phi, omitted = _constant_phase_and_omission(reconstruction.constant, reconstruction.step_weights, method)
    except OverflowError:
        phi, omitted = float("inf"), 0
    if not isfinite(phi):
        raise ValueError(
            f"keep_state needs the global phase -c sum_k dt b_k of the objective constant c = {c!r}, "
            "which exceeds the binary64 range. The constant does not affect the optimization, so "
            "remove it from the objective, or plan without keep_state to read the probabilities, "
            "which do not depend on c"
        )
    # S_C = |c| sum_k |dt b_k| and (3u + 3u**2 + u**3) S_C, upward
    s_c = _mul_up(abs(c), _sum_up(_mul_up(abs(dt), abs(b)) for _t, _a, b in reconstruction.step_weights))
    coefficient = _add_up(_add_up(3 * u, _mul_up(3 * u, u)), _mul_up(_mul_up(u, u), u))
    # (3u + 3u**2 + u**3) S_C, plus the omitted product
    allowance = _add_up(_mul_up(coefficient, s_c), _upward(omitted))
    if not allowance < pi:
        raise ValueError(
            f"keep_state needs the global phase -c sum_k dt b_k of the objective constant c = {c!r}, "
            f"whose rounding allowance (3u + 3u**2 + u**3) |c| sum_k |dt b_k| = {allowance!r} rad "
            "reaches pi, so the allowance cannot establish the requested physical phase. The constant "
            "does not affect the optimization, so remove it from the objective, or plan without "
            "keep_state to read the probabilities, which do not depend on c"
        )


# Upward evaluation of the nonnegative phase allowances. Round-to-nearest puts
# a computed sum, product or quotient within half a unit in the last place of
# its exact value, so the next binary64 number above the computed result is at
# least the exact value, and an inexact positive result that rounded to zero is
# lifted to 2**-1074. An exact zero stays zero, because it follows from zero
# operands. Monotone combinations of upper bounds stay upper bounds, and a
# positive quotient takes an upper numerator and a lower denominator. One
# fixed inflation factor at the end would not do, because the number of
# operations grows with the Plan. For example, 1.0 followed by 2**15 terms of
# 2**-54 sums to 1.0 in plain binary64, and the exact sum 1 + 2**-39 exceeds
# even the inflated 1 + 2**-40. The phase allowances of this module,
# _admit_constant_phase and _phase_ledger to _admit_compiled_phase, and the
# binary encoding's Walsh phase terms and reconstruction check
# (binary.compile_binary_steps, binary._admit_reconstruction) evaluate every
# operation this way.
def _add_up(a, b):
    """Return an upper bound on ``a + b`` for nonnegative upper bounds a and b (``_compiled_phase_allowance``)."""
    return 0.0 if a == 0 and b == 0 else nextafter(a + b, inf)


def _mul_up(a, b):
    """Return an upper bound on ``a b`` for nonnegative upper bounds a and b, lifting an underflow to 2**-1074."""
    return 0.0 if a == 0 or b == 0 else nextafter(a * b, inf)


def _div_up(a, b):
    """Return an upper bound on ``a/b`` for a nonnegative upper bound a and a positive lower bound b."""
    return 0.0 if a == 0 else nextafter(a / b, inf)


def _sum_up(values):
    """Return an upper bound on the sum of nonnegative upper bounds, one upward addition at a time."""
    total = 0.0
    for value in values:
        total = _add_up(total, value)
    return total


def _gamma_up(r):
    """Return an upper bound on ``gamma_r = r u/(1 - r u)``, infinite when ``1 - r u`` has no positive lower bound.

    ``gamma_r`` bounds ``|prod_(i=1)^r (1 + delta_i)**(+-1) - 1|`` for ``|delta_i| <= u`` when ``r u < 1``
    (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., doi:10.1137/1.9780898718027,
    Lemma 3.1). The count r is an exact integer, so ``r >= 2**53``, where ``r u >= 1``, is tested before
    any rounding, and the denominator ``1 - r u`` takes its lower value.
    """
    if r <= 0:
        return 0.0
    if r >= 2**53:
        return inf
    ru = _mul_up(float(r), UNIT_ROUNDOFF)
    lower = nextafter(1.0 - ru, -inf)
    return _div_up(ru, lower) if lower > 0 else inf


def _schrodinger_kinetic_bounds(grid):
    """Return J >= ||abs(Khat)||_1 and H >= ||Khat-tr(Khat)I/D||_1.

    The column-major COO fill in restricted_kinetic_sparse and SciPy 1.18.1
    produce sorted CSR rows before duplicate reduction. Each diagonal is
    therefore the same left-to-right binary64 sum of 1/(h*h), in variable
    order. The exact range of the stored diagonal is zero. The off-diagonal
    column sum has one neighbor per axis for Dirichlet K=2 and two otherwise.
    Periodic K=2 adds two equal link values exactly. This is an O(d) scalar
    bound for the stored matrix, with no Cartesian-grid allocation.

    The raw COO entries are ordered by column and then by variable. SciPy 1.18.1
    coo_tocsr preserves their order within each CSR row, whose column indices
    are therefore already nondecreasing. CSR sum_duplicates skips index sorting
    and adds each equal-column run from left to right. Every stored diagonal is
    the same finite binary64 sum of 1.0/(h_j*h_j), in variable order, so its
    exact range is zero. This statement depends on that fill and conversion
    path. The ordered scalar loop reproduces the stored diagonal without
    constructing the stencil.

    With ``q_j = |RN(-0.5/RN(h_j h_j))|`` and the column multiplicity nu (1 for
    Dirichlet K = 2, 2 otherwise), upward arithmetic gives
    ``O >= sum_j nu q_j``, ``H = O`` and ``J >= c + O`` for the stored diagonal
    c. The loop uses ``h*h`` as the stencil does, and its diagonal accumulator
    is the ordered ``+=``: Python's ``sum``, ``math.fsum`` or a NumPy reduction
    can round differently (with d = 4, K = 2 and axis contributions
    ``(1, 2**-54, 2**-54, 2**-54)`` every stored diagonal is
    ``0x1.0000000000000p+0``, whereas ``fsum`` gives ``0x1.0000000000001p+0``).
    The SciPy sources are ``scipy/sparse/_coo.py`` (``tocsr``),
    ``sparsetools/coo.h`` (``coo_tocsr``), ``_compressed.py``
    (``sum_duplicates``) and ``sparsetools/csr.h`` (``csr_sum_duplicates``)
    at tag v1.18.1. K >= 2, the same axis coefficients in every column and
    ordinary ordered binary64 addition are further premises. An axis-major
    fill, another conversion path, explicit sorting before conversion, a
    changed dependency or reassociating arithmetic needs requalification.

    Raises:
        ValueError: The stored kinetic diagonal overflows binary64.
    """
    diagonal = off_diagonal = 0.0
    neighbors = 1 if grid.boundary == "dirichlet" and grid.num_grid_points == 2 else 2
    for j in range(grid.num_variables):
        h = grid.spacing(j)
        square = h * h
        diagonal += 1.0 / square
        off_diagonal = _add_up(off_diagonal, neighbors * abs(-0.5 / square))
    if not isfinite(diagonal):
        raise ValueError("the stored Schrodinger kinetic diagonal overflows binary64. "
                         "Use fewer grid points or wider boxes")
    return _add_up(diagonal, off_diagonal), off_diagonal


def _schrodinger_step_bounds(method, grid, tables, step_weights):
    """Yield (N, r, Q) for each stored Schrodinger step, including gradual underflow.

    The reference is dt*(a*Khat+b*diag(Vstar)), with stored Khat and the
    exact sum Vstar of stored support entries. With u=2**-53, eta=2**-1074,
    m=max(T-1,0), W=sum_t max(abs(t_min),abs(t_max)) and
    R=sum_t(t_max-t_min), upward arithmetic gives E_V=gamma_m*W and
    W_vec=W+E_V. Finite binary64 additions satisfy the relative model even
    when their result is subnormal, since a subnormal exact sum is an
    exactly representable integer multiple of eta.

    Each assembly leaf has at most three roundings. Multiplication obeys
    abs(RN(x*y)-x*y) <= u*abs(x*y)+eta/2 with gradual underflow. A column
    has at most r=1+2*d kinetic products and one potential product. Their
    absolute errors pass through at most an addition and the dt product,
    giving eta/2*((r+1)*abs(dt)*(1+u)**2+r). Since (1+u)**2<2, use
    U=eta*((r+1)*abs(dt)+r), evaluated with eta scaled before dt.
    A=gamma_3*abs(dt)*(abs(a)*J+abs(b)*W_vec)+U bounds assembly error,
    Q=abs(dt*b)*E_V+A bounds table and assembly error, and
    N=abs(dt)*(abs(a)*H+abs(b)*R)+2*Q bounds the exact trace-shifted
    generator. J and H are _schrodinger_kinetic_bounds' stored-matrix
    bounds. Symmetry bounds the spectral norm by the column 1-norm and
    exact centering increases a perturbation bound by at most two.

    The arithmetic premise is round-to-nearest binary64 with gradual
    underflow and finite assembly intermediates. The SciPy state model
    remains conditional and first order, including its computed-trace-shift
    premise. This assembly allowance does not establish that model.

    For overflow of the assembly intermediate before gamma_3, lower
    total_time or increase num_steps at fixed weights, scale the objective
    down or widen the box at fixed num_grid_points, rechecking the table
    magnitude and every recomputed weight until abs(total_time/num_steps) <
    binary64_max/(abs(a)*J+abs(b)*W_vec) has outward-rounding headroom
    and the unscaled products and sum are finite.
    The fallback diagnostic reports abs(dt), a, b, J and W_vec from
    the stored step and bounds.

    Raises:
        ValueError: The kinetic diagonal overflows, or a step's N or Q is
            nonfinite. The diagnostic identifies the failing base-norm
            operation or the table, assembly or centering bound.
    """
    j_bound, h_bound = _schrodinger_kinetic_bounds(grid)
    magnitude = _sum_up(max(abs(t.minimum), abs(t.maximum)) for t in tables)
    spread = _sum_up(
        0.0 if t.minimum == t.maximum else nextafter(t.maximum - t.minimum, inf)
        for t in tables
    )
    table_error = _mul_up(_gamma_up(max(len(tables) - 1, 0)), magnitude)
    vector_magnitude = _add_up(magnitude, table_error)
    gamma3 = _gamma_up(3)
    dt = abs(method.total_time / method.num_steps)
    rows = 1 + 2 * grid.num_variables
    eta = nextafter(0.0, inf)
    underflow = _add_up(
        _mul_up(dt, _mul_up(float(rows + 1), eta)),
        _mul_up(float(rows), eta),
    )
    for index, (_, a, b) in enumerate(step_weights):
        relative = _mul_up(gamma3, _mul_up(dt, _add_up(
            _mul_up(abs(a), j_bound), _mul_up(abs(b), vector_magnitude))))
        assembly = _add_up(relative, underflow)
        potential = _mul_up(_mul_up(dt, abs(b)), table_error)
        perturbation = _add_up(potential, assembly)
        norm = _add_up(_mul_up(dt, _add_up(
            _mul_up(abs(a), h_bound), _mul_up(abs(b), spread))),
            _mul_up(2.0, perturbation))
        if not (isfinite(norm) and isfinite(perturbation)):
            base_norm = dt * (abs(a) * h_bound + abs(b) * spread)
            detail = (_step_norm_refusal(index, base_norm, dt, a, b, h_bound, spread)
                      if not isfinite(base_norm) else
                      "The table/assembly or outward centering bound is nonfinite. "
                      "The assembly bound forms abs(dt)*(abs(a)*J + abs(b)*W_vec) before gamma_3, "
                      "with dt = total_time/num_steps. "
                      "Lower total_time or increase num_steps to lower abs(dt) when a and b are fixed. "
                      "Scale the objective down to lower W_vec, or widen the box at fixed num_grid_points "
                      "to lower J and recheck W_vec on the new box. "
                      "Require abs(dt) < binary64_max/(abs(a)*J + abs(b)*W_vec) with headroom for "
                      "outward rounding and finite unscaled products and sum. "
                      "Time-dependent weights are recomputed when total_time or num_steps changes, "
                      "so recheck this relation at every new step. Other Plan range and resource limits still apply. "
                      f"Here abs(dt) = {dt!r}, a = {a!r}, b = {b!r}, "
                      f"J = {j_bound!r}, W_vec = {vector_magnitude!r}.")
            raise ValueError(f"Schrodinger step {index} has nonfinite bounds "
                             f"N={norm!r}, Q={perturbation!r}. {detail}")
        yield norm, rows, perturbation


def _schrodinger_state_error(method, grid, tables, step_weights, start):
    """Return the conditional first-order error against the table-sum product.

    Unitary propagation adds the per-step table and assembly perturbations
    to expm_multiply_state_error. start is in units of u. The perturbations
    are already in state 2-norm units and are added without another factor u.

    The reference is the ordered product of exact exponentials of
    ``G_k* = dt (a_k Khat + b_k diag(V*))``. For Hermitian A and B,
    Duhamel's identity and unitarity give
    ``||exp(-iA) - exp(-iB)||_2 <= ||A - B||_2``, so telescoping adds the
    ``Q_k`` of ``_schrodinger_step_bounds`` without kinetic exponential
    amplification:
    ``delta_sch = u (start + sum_k [c(N_k, r) + N_k]) + sum_k Q_k``, with
    ``c(N, r)`` the numerical call charge of ``expm_multiply_state_error``.
    Analytic spacing, kinetic-entry formation, table evaluation, schedule
    approximation and time discretization are outside this stored-matrix and
    table reference, and the objective constant is outside its phase
    convention.
    """
    perturbation = 0.0

    def calls():
        nonlocal perturbation
        for norm, rows, local in _schrodinger_step_bounds(method, grid, tables, step_weights):
            perturbation = _add_up(perturbation, local)
            yield norm, rows

    numerical = expm_multiply_state_error(calls(), start=start)
    return _add_up(numerical, perturbation)


def _phase_contributions(reconstruction, method):
    """Return M, the number of contributions that the compiled phase ledger sums, from the reconstruction.

    Without compiled steps nothing is recorded and M = 0, in either
    encoding.

    One-hot. Each of the N steps (``QHDReconstruction.step_weights``)
    records the kinetic diagonal once
    (``compiler.QHDCompiler._record_kinetic_global_phase``), the objective
    constant once when the stored constant c is nonzero, and the identity of
    every row of every stored support table once per potential factor
    (``compiler.QHDCompiler._compile_potential_with_phase_record``), F = 1
    factor for first order and 2 for second order. A contribution omitted
    below the normal range records a zero event and is still counted
    (``records.QHDRangeOmissions``). With R the total number of table rows,

    ``M = N (1 + [c != 0] + F R)``.

    Binary. The ledger holds only the constant's contributions
    (``binary.compile_binary_steps``), ``M = N [c != 0]``. Planning's admission
    (``_admit_compiled_phase``) and the error ledger
    (``circuit_errors.phase_allowance``, ``circuit_errors.binary_phase_allowance``)
    take M from here.
    """
    r = reconstruction
    if not r.compact_schedule_selected:
        return 0
    steps, constant = len(r.step_weights), int(r.constant != 0)
    if method.encoding == "binary":
        return steps * constant
    factors = 1 if method.trotter_order == 1 else 2
    rows = sum(len(table.values) for table in r.support_values)
    # M = N (1 + [c != 0] + F R)
    return steps * (1 + constant + factors * rows)


def _compiled_phase_allowance(reconstruction, method, grid, contributions):
    """Return an upper allowance, in radians, on the compiled identity phase ``physical_phase``.

    The reference is the exact identity angle of the discrete model with the
    stored binary64 dt, step weights ``(a_k, b_k)``, spacings h_l, table
    values v_(S,r) and constant c read as exact inputs,
    ``Phi = -sum_k dt (a_k sum_l h_l**-2 + b_k c + b_k sum_(S,r) 2**-|S| v_(S,r))``,
    where the two second-order potential halves together contribute the
    full step. It excludes the rounding of those inputs, time
    discretization, rotation pruning and synthesis.

    Formation. With ``S_C = |c| sum_k |dt b_k|``,
    ``S_P = sum_k |dt b_k| sum_(S,r) 2**-|S| |v_(S,r)|`` and
    ``S_K = sum_k |dt a_k| sum_l h_l**-2``, each admitted constant and
    projector contribution is two rounded products of stored binary64 values
    (``compiler.QHDCompiler`` normalizes c once), so its relative formation
    error is at most ``eta_2 = (1 + u)**2 - 1 = 2u + u**2``, the power-of-two
    scaling and the halving of dt being exact under the range rule of
    planning (``validation._normal_range``). The kinetic coefficient
    ``sum(a_k/h_l**2)`` over the d variables is summed by Python's float
    ``sum``, which since Python 3.12 uses Neumaier's compensated summation
    (Neumaier, Z. Angew. Math. Mech. 54 (1974) 39-51,
    doi:10.1002/zamm.19740540106). Its high part and correction follow the
    recurrences of the cascaded summation of Ogita, Rump and Oishi (SIAM J.
    Sci. Comput. 26 (2005) 1955-1988, doi:10.1137/030601818, Algorithm 4.1),
    and their Proposition 4.5 bounds the error of the equivalent Algorithm
    4.4 on d terms with absolute sum A_d and exact sum X by
    ``u |X| + gamma_(d-1)**2 A_d``. The residual argument written out in
    ``compiler.QHDCompiler._record_global_phase`` gives the slightly larger
    ``u |X| + (1 + u) gamma_(d-1)**2 A_d <= beta_d A_d`` with
    ``beta_d = u + (1 + u) gamma_(d-1)**2``, which this allowance uses. For
    comparison, Higham, *Accuracy and Stability of Numerical Algorithms*,
    2nd ed., doi:10.1137/1.9780898718027, Sec. 4.3, Eq. (4.10), gives a
    backward error bound for compensated summation with separately
    accumulated corrections, the variant of Neumaier's paper, in which each
    term is perturbed relatively by at most 2.1u when ``d**2 u <= 0.1``. Assuming each
    evaluated ``h**2`` has relative error at most u, the rounded square and
    division contribute at most ``(1 + u)/(1 - u)`` and the duration product
    ``1 + u``, so the kinetic formation is within
    ``eta_K = (1 + u)**2 (1 + beta_d)/(1 - u) - 1``, evaluated in the
    expanded form ``(3u + u**2 + (1 + 2u + u**2) beta_d)/(1 - u)``, which
    subtracts no nearly equal numbers, of S_K. The formation charge is
    ``D = eta_2 (S_C + S_P) + eta_K S_K``.

    Accumulation. The recorder adds the M = ``contributions`` formed angles
    (``_phase_contributions``) with Neumaier's compensation
    (``compiler.QHDCompiler._record_global_phase``), whose error is at most
    ``beta_M A`` with ``beta_M = u + (1 + u) gamma_(M-1)**2`` for the sum A
    of their absolute values, and ``A <= S_C + S_P + S_K + D``. The allowance
    is

    ``E_arith = D + beta_M (S_C + S_P + S_K + D)``,

    to first order ``3u (S_C + S_P) + 5u S_K``, without a factor of the
    contribution count. For the constant's own total over N steps the same
    argument gives ``[eta_2 + (1 + eta_2) beta_N] S_C``. Cancellation between
    contributions does not reduce it. The bound assumes round-to-nearest
    binary64 arithmetic and no overflow. Planning establishes the stated
    relative-error premises for admitted contributions. Lower-range
    omissions are charged by their exact omitted identity action, separately
    from that arithmetic allowance. The empty ledger has allowance zero.

    The relative-error derivation applies to the admitted contribution
    arithmetic. Contributions omitted below the normal range are zero events
    in the accumulated ledger. Their exact identity-action charge O is added
    once, so the returned allowance is ``up(E_arith + O)``. The source
    components for the objective constant, projector identities and kinetic
    diagonal add to E_arith before outward evaluation. The separate
    ``omitted_identity`` component supplies O.

    Evaluation. Every elementary operation of the allowance, its input sums
    included, is rounded upward with ``math.nextafter`` (``_add_up``,
    ``_mul_up``, ``_div_up``), a positive quotient takes a lower
    denominator, ``h**-2`` is two upward divisions by the exact ``|h|``, and
    ``gamma_r`` uses the lower denominator of ``1 - r u`` (``_gamma_up``).
    One final inflation factor would not do, because the number of
    operations grows with the Plan. The result is an upper value of the
    expression, infinite when a denominator has no positive lower bound or
    the evaluation overflows.

    Binary encoding. The ledger contains only the objective constant. Its
    reference is ``Phi_C = -sum_k dt b_k c``, with the stored binary64
    duration, weights and normalized constant read as exact inputs, so
    ``S_P = S_K = 0`` and M counts only the actual constant contributions.
    Each admitted contribution ``-fl(dt fl(b_k c))`` uses two rounded products, so
    ``D = (2u + u**2) S_C`` and ``E_arith = D + beta_M (S_C + D)``, zero for
    an empty ledger. ``binary.compile_binary_steps`` sums the formed angles
    with ``math.fsum``, whose correctly rounded result errs by at most
    ``u |X|``, below ``beta_M`` times their absolute sum. This use of fsum
    rests on that rounding premise. Planning establishes the stated
    relative-error premises for admitted contributions. Lower-range
    omissions are charged by their exact omitted identity action, separately
    from that arithmetic allowance. The diagonals' identity phases stay
    inside the binary blocks and are not ledger contributions. Their
    separate formation allowance ``F_W`` (``binary.WalshPhaseTerms``) is
    required on both circuit routes (``_admit_compiled_phase``) and is not
    zero merely because ``S_P = S_K = 0``.
    """
    return _phase_ledger(reconstruction, method, grid, contributions)[0]


def _phase_ledger(reconstruction, method, grid, contributions):
    """Return ``E_ledger`` (``_compiled_phase_allowance``) and its additive component for each phase source.

    Expanding ``E_arith = D + beta_M (S_C + S_P + S_K + D)`` with
    ``D = eta_2 (S_C + S_P) + eta_K S_K`` splits it exactly into
    ``L_s = [eta_s + beta_M (1 + eta_s)] S_s`` for the objective constant,
    the projector identities and the kinetic diagonal, with
    ``eta_C = eta_P = eta_2``. The component formulas add up to
    ``E_arith`` exactly before rounding. Each returned component is an
    upper value evaluated upward on its own, separately from the total, so
    the binary64 components need not sum to the returned total. They rank
    the sources for the rejection message of ``_admit_compiled_phase``,
    which uses the total for the decision.

    The relative-error derivation applies to the admitted contribution
    arithmetic. Contributions omitted below the normal range are zero events
    in the accumulated ledger. Their exact identity-action charge O is added
    once, so the returned allowance is ``up(E_arith + O)``. The source
    components for the objective constant, projector identities and kinetic
    diagonal add to E_arith before outward evaluation. The separate
    ``omitted_identity`` component supplies O.

    Omitted identities. A projector identity or constant contribution
    omitted below the normal range enters the ledger as a zero event, so the
    relative-error terms above, evaluated with the original absolute sums,
    still bound the admitted contributions' arithmetic. The omitted identity
    actions ``2**-s |t b v|`` and ``|dt b c|`` are then added once as the
    component ``omitted_identity`` (``QHDRangeOmissions.identity_charge``),
    before the native assignment allowances and kept-state admission. An
    omitted identity ``exp(-i a I)`` differs from I by at most ``|a|``.
    """
    if contributions == 0:
        return 0.0, {}
    u = UNIT_ROUNDOFF
    dt = method.total_time / method.num_steps
    d = grid.num_variables
    potential_time = _sum_up(_mul_up(abs(dt), abs(b)) for _t, _a, b in reconstruction.step_weights)
    kinetic_time = _sum_up(_mul_up(abs(dt), abs(a)) for _t, a, _b in reconstruction.step_weights)
    # S_C, S_P and S_K
    constant = _mul_up(abs(reconstruction.constant), potential_time)
    if method.encoding == "binary":
        # The binary ledger holds only the constant's contributions
        # (binary.compile_binary_steps), so S_P = S_K = 0.
        projector = kinetic = 0.0
    else:
        tables = _sum_up(_div_up(_sum_up(abs(v) for v in t.values.array), 2.0 ** len(t.support))
                         for t in reconstruction.support_values)
        projector = _mul_up(potential_time, tables)
        spacings = (abs(grid.spacing(j)) for j in range(d))
        kinetic = _mul_up(kinetic_time, _sum_up(_div_up(_div_up(1.0, h), h) for h in spacings))
    eta_2 = _add_up(2 * u, _mul_up(u, u))
    one_plus_u = _add_up(1.0, u)
    gamma_d = _gamma_up(d - 1)
    beta_d = _add_up(u, _mul_up(one_plus_u, _mul_up(gamma_d, gamma_d)))
    numerator = _add_up(_add_up(3 * u, _mul_up(u, u)),
                        _mul_up(_add_up(_add_up(1.0, 2 * u), _mul_up(u, u)), beta_d))
    eta_k = _div_up(numerator, nextafter(1.0 - u, -inf))
    # D = eta_2 (S_C + S_P) + eta_K S_K
    formation = _add_up(_mul_up(eta_2, _add_up(constant, projector)), _mul_up(eta_k, kinetic))
    # E_arith = D + beta_M (S_C + S_P + S_K + D), beta_M = u + (1 + u) gamma_(M-1)**2
    gamma_m = _gamma_up(contributions - 1)
    beta_m = _add_up(u, _mul_up(one_plus_u, _mul_up(gamma_m, gamma_m)))
    absolute = _sum_up((constant, projector, kinetic, formation))
    # L_s = [eta_s + beta_M (1 + eta_s)] S_s
    components = {
        "objective_constant": _mul_up(_add_up(eta_2, _mul_up(beta_m, _add_up(1.0, eta_2))), constant),
        "projector_identity": _mul_up(_add_up(eta_2, _mul_up(beta_m, _add_up(1.0, eta_2))), projector),
        "kinetic_diagonal": _mul_up(_add_up(eta_k, _mul_up(beta_m, _add_up(1.0, eta_k))), kinetic),
    }
    total = _add_up(formation, _mul_up(beta_m, absolute))
    omitted = reconstruction.range_omissions.identity_charge
    if omitted:
        components["omitted_identity"] = omitted
        total = _add_up(total, omitted)
    return total, components


def _count_up(count):
    """Return an upper binary64 value of a nonnegative integer count."""
    value = float(count)
    return value if int(value) >= count else nextafter(value, inf)


def _native_phase_allowance(reconstruction, ledger, walsh=None):
    """Return an upper allowance, in radians, on the native circuit's global phase for a kept state.

    Let u = 2**-53, lambda = 2**-1074 and tau the binary64 2 pi,
    0x1.921fb54442d18p+2, with ``Delta = 2**-51 >= |tau - 2 pi|``. The
    bound assumes round-to-nearest binary64 arithmetic, gradual underflow,
    finite native operations and finite exact divisors ``2**s``, and
    ``ledger`` (``_compiled_phase_allowance``) must already cover the
    formation and accumulation errors of ``physical_phase``.

    The native builder assigns the circuit's global phase once per stored
    ``number_projector`` block b, adding ``theta_b/2**s_b`` for its angle and
    support size (``subroutines.hamiltonian_evolution.append_number_projector_phase``),
    and once more for ``physical_phase`` (``native.construct_qhd``). B counts
    those blocks, both potential halves included, and
    ``Y = sum_b |theta_b| 2**-s_b`` with the stored angles taken as exact. The
    intended scalar compensation is ``sum_b theta_b 2**-s_b``, and scaling an
    angle can incur at most lambda/2 of underflow error.

    Qiskit 2.5.2 stores each assigned phase with Rust's
    ``f64::rem_euclid(tau)`` (``crates/circuit/src/circuit_data.rs``,
    ``set_global_phase_f64``, at tag 2.5.2), whose rounded exact remainder
    can equal tau, so a stored phase g satisfies ``0 <= g <= tau``. For an
    intended increment y with ``v = |y|`` and increment error e, put
    ``H = tau + v + e``, ``A = u H + lambda`` and ``K = (H + A)/tau + 1``.
    The addition error is at most A, the integer reduction quotient has
    absolute value at most K, and rounding the remainder, which lies in
    [0, tau) below 8, costs at most Delta. Compared modulo the mathematical
    2 pi, one assignment therefore errs by at most
    ``C(v, e) = e + A + Delta K + Delta``. With e = lambda for projector
    scaling and e = 0 for ``physical_phase``, and since circular distance is
    translation invariant, the errors of successive assignments add, so the
    projector assignments contribute ``E_g <= sum_b C(|theta_b| 2**-s_b, lambda)``.
    For the final assignment ``E_add = u (tau + |physical_phase|) + lambda``,
    ``E_rem = Delta`` and its quotient bound is
    ``(tau + |physical_phase| + E_add)/tau + 1``.

    The quotient term ``Delta K`` is the reason this bookkeeping is charged
    at every assignment. Reducing by the binary64 tau instead of 2 pi errs
    by the quotient times ``|tau - 2 pi|``, which grows with the unreduced
    angle, so a large intermediate phase costs accuracy even when the final
    phase is small. With Qiskit 2.5.2 (Python 3.12.14, macOS arm64,
    2026-09-26, NWQLib commit 418f9907ec4d629011486c14eb23cf4fdbdd1d3e),
    assigning -2e17 to an empty circuit stored a phase 1.51 rad from the
    exact reduction, and two cancelling increments of 1e16 and -1e16
    followed by a zero final phase left 0.64 rad, which a bound on the final
    assignment alone would miss.

    Taking ``e = lambda`` bounds every assignment, the final one included,
    because C increases with e. Substituting H, A and K and ``Delta = 4u``
    gives

    ``C(v, lambda) = [u + 4u (1 + u)/tau] v + u tau + 4u (3 + u) + (2 + u) lambda (1 + 4u/tau)``.

    Since 6 < tau < 7, the coefficient of v is below 2u, and the constant is
    below 20u because ``u tau + 12u < 19u`` and
    ``4u**2 + (2 + u) lambda (1 + 4u/tau) < u``, hence

    ``E_native = E_ledger + 2u (Y + |physical_phase|) + 20u (B + 1)``,

    which includes the final assignment and the scaling underflow. It is
    evaluated upward at every operation, like the ledger. This bounds the
    scalar phase bookkeeping for the stored block gates. Gate execution,
    state preparation, exponentials and the multiplication of state
    amplitudes need their own error qualifications.

    Binary encoding (``walsh``, ``binary.WalshPhaseTerms``). A binary Plan
    stores no projector block, and B and Y count the Walsh diagonals instead
    (``_circuit_assignments``). B counts the
    Walsh diagonal blocks that ``native.append_diagonal`` applies, kinetic
    blocks, both potential halves and blocks with a zero identity increment
    or no kept rotation included, and ``Y = sum_b |identity_phase_b|`` uses
    the computed binary64 identity phases of the selected constructions.
    ``F_W`` charges their formation against the intended identity
    increments, exponent formation included, and the stored identity phases
    are exact inputs to the consumer part only. Each Walsh block executes
    ``circuit.global_phase += identity_phase_b``, so the ``C(v, e)``
    derivation above applies with ``v = |identity_phase_b|`` and ``e = 0``,
    at most ``2u v + 20u`` each, and circular errors add. With formation and
    the final ``physical_phase`` assignment,

    ``E_native = E_ledger + F_W + 2u (Y + |physical_phase|) + 20u (B + 1)``,

    evaluated upward, the count conversion included. Dense diagonals on
    nonempty registers keep their identity phase inside the gate definition
    and assign no outer phase, and that internal phase has gate-synthesis
    and execution errors of its own. This allowance compares the scalar
    phase with ``Phi_C`` plus the intended Walsh identity phases, modulo
    mathematical 2 pi. It does not bound nonidentity angle formation or the
    complete state error.
    """
    u = UNIT_ROUNDOFF
    scaled, blocks = _circuit_assignments(reconstruction, walsh)
    if walsh is not None:
        ledger = _add_up(ledger, walsh.formation)
    phases = _add_up(scaled, abs(reconstruction.physical_phase))
    # E_native = E_ledger (+ F_W) + 2u (Y + |physical_phase|) + 20u (B + 1)
    return _add_up(_add_up(ledger, _mul_up(2 * u, phases)), _mul_up(20 * u, _count_up(blocks + 1)))


def _projector_assignments(reconstruction):
    """Return an upper bound on ``Y = sum_b |theta_b| 2**-s_b`` and the count B of ``_native_phase_allowance``."""
    projectors = [b for step in reconstruction.steps for b in step if b.kind == "number_projector"]
    # Y = sum_b |theta_b| 2**-s_b
    return _sum_up(_div_up(abs(b.angle), 2.0 ** len(b.support)) for b in projectors), len(projectors)


def _circuit_assignments(reconstruction, walsh):
    """Return ``(Y, B)`` of the per-block circuit phase assignments that ``_native_phase_allowance`` charges.

    A one-hot circuit assigns the scaled angle of each number-projector block
    (``_projector_assignments``) and a binary circuit the identity phase of
    each Walsh diagonal (``walsh``, ``binary.WalshPhaseTerms``). A binary
    Plan stores no projector block and a one-hot Plan has no Walsh terms, so
    each encoding takes its own set and no assignment is counted twice.
    """
    return _projector_assignments(reconstruction) if walsh is None else (walsh.identity_sum, walsh.count)


def _native_components(reconstruction, components, walsh=None):
    """Return the additive components of ``_native_phase_allowance`` for the rejection message.

    ``E_native = E_ledger + 2u (Y + |physical_phase|) + 20u (B + 1)`` splits
    into the ledger components ``L_C`` and ``L_K`` (``_phase_ledger``), the
    projector component ``L_P + 2u Y + 20u B``, which adds the projector
    blocks' own circuit assignments, and the final assignment of
    ``physical_phase``, ``2u |physical_phase| + 20u``. The final phase sums
    all sources, which can cancel, so its assignment is not attributed to
    one Hamiltonian source. Each component is evaluated upward with the
    same Y and B as the allowance.

    For a binary Plan (``walsh``) the ledger holds only ``L_C``, and the
    allowance adds ``F_W`` and the Walsh diagonals' own assignments
    ``2u Y_W + 20u B_W``, which are the components
    ``walsh_identity_formation`` and ``walsh_phase_assignments``. With the
    final assignment the component formulas again add up to the allowance
    exactly before rounding. The returned upper values are evaluated
    separately from the total and need not sum to its binary64 value.
    """
    u = UNIT_ROUNDOFF
    scaled, blocks = _circuit_assignments(reconstruction, walsh)
    # 2u Y + 20u B of the per-block assignments
    consumer = _add_up(_mul_up(2 * u, scaled), _mul_up(20 * u, _count_up(blocks)))
    final = _add_up(_mul_up(2 * u, abs(reconstruction.physical_phase)), 20 * u)
    if walsh is None:
        return {
            **components,
            "projector_identity": _add_up(components.get("projector_identity", 0.0), consumer),
            "final_assignment": final,
        }
    return {
        **components,
        "walsh_identity_formation": walsh.formation,
        "walsh_phase_assignments": consumer,
        "final_assignment": final,
    }


def _admit_compiled_phase(reconstruction, method, grid, contributions, execution, walsh=None):
    """Raise at planning when a kept native or ``ir_product`` state cannot carry its compiled phase.

    A normalized state whose global phase is known within an angle
    allowance E is known within ``2 sin(min(E, pi)/2)`` in the 2-norm, which
    is the maximum value 2 once E reaches pi, so the allowance then gives no
    guarantee on the physical phase. It does not show that the computed
    phase is wrong. The ``ir_product`` reference multiplies its state by
    ``exp(i physical_phase)``, and its allowance is the ledger's,
    ``_compiled_phase_allowance``. The native circuit assigns its global
    phase for every projector block and for ``physical_phase``, and its
    allowance is ``_native_phase_allowance``. A Plan with a kept state is
    admitted only when the upper allowance is finite and below ``math.pi``,
    which lies below the mathematical pi. Passing the check does not certify
    the other errors of the state, and finite phase uncertainty does not
    affect the probabilities, so only a kept state is checked. A rejection
    names the largest additive component of the allowance, the ledger
    components of ``_phase_ledger`` and, natively, those of
    ``_native_components``, and advises removing the objective constant only
    when its component is the largest.

    For a kept binary state (``walsh``, ``binary.WalshPhaseTerms``) both
    routes add the Walsh identity formation allowance ``F_W`` to the
    constant-only ledger. Native execution also charges every Walsh
    global-phase assignment (``_native_phase_allowance``). ``ir_product``
    instead adds ``R_W``, the reconstruction allowance of
    ``DiagonalConstruction.emitted_phases``, so its allowance is
    ``E_ledger + F_W + R_W``, whose components are the ledger's,
    ``walsh_identity_formation`` and ``walsh_phase_reconstruction``. These
    terms are absent from ``theory.binary_product_state_error``, whose
    reference takes the reconstructed phase entries as floating-point
    inputs. ``R_W`` bounds per-entry angle errors, which can be relative
    phases. For an entrywise perturbation ``|delta_r| <= R < pi`` a
    normalized state keeps ``Re <psi|diag(exp(i delta_r))|psi> >= cos R``,
    so its angle on the real unit sphere is at most R, unitary propagation
    keeps that angle, and with the common scalar error the state distance is
    at most ``2 sin(E/2)`` for ``E < pi``. A full state claim still needs
    the other formation and execution errors.
    """
    allowance, components = _phase_ledger(reconstruction, method, grid, contributions)
    if execution == "quantum":
        allowance = _native_phase_allowance(reconstruction, allowance, walsh)
        components = _native_components(reconstruction, components, walsh)
    elif walsh is not None:
        # E_ir = E_ledger + F_W + R_W
        allowance = _sum_up((allowance, walsh.formation, walsh.reconstruction))
        components = {**components, "walsh_identity_formation": walsh.formation,
                      "walsh_phase_reconstruction": walsh.reconstruction}
    if not allowance < pi:
        route = "native circuit" if execution == "quantum" else "ir_product reference"
        largest = max(components, key=components.get) if components else "total"
        names = {
            "objective_constant": "the objective constant's contributions",
            "projector_identity": "the projector-identity contributions"
            + (" with the projector blocks' own phase assignments" if execution == "quantum" else ""),
            "kinetic_diagonal": "the kinetic-diagonal contributions",
            "omitted_identity": "the identity phases of contributions omitted below the normal binary64 range",
            "final_assignment": "the final assignment of the physical phase to the circuit",
            "walsh_identity_formation": "the formation of the Walsh diagonals' identity phases",
            "walsh_phase_assignments": "the Walsh diagonals' own phase assignments to the circuit",
            "walsh_phase_reconstruction": "the reconstruction of the Walsh diagonals' phases from their kept "
            "rotation angles",
        }
        raise ValueError(
            f"keep_state needs the physical phase {reconstruction.physical_phase!r} rad of the {route}, "
            f"whose rounding allowance {allowance!r} rad reaches pi, so the allowance cannot establish "
            f"the requested physical phase. Its largest component, {components.get(largest)!r} rad, "
            f"comes from {names.get(largest, largest)}"
            + (". The objective constant does not affect the optimization, so removing it from the "
               "objective removes that component" if largest == "objective_constant" else "")
            + ". A Plan without keep_state needs no phase allowance"
        )


def _range_omissions(tables, omitted, *, chains, amplitudes):
    """Return the ``QHDRangeOmissions`` record of a Plan from its compilation's omissions and its stored data.

    ``omitted`` holds the counts and exact charges of the compiled route
    (``compiler.QHDCompiler.metadata``, ``binary.compile_binary_steps``) or
    of the classical constant phase (``_constant_phase``), with
    ``walsh_tables`` empty unless binary tables were compiled. The subnormal
    entries are counted in the stored tables, and ``chains`` selects the
    structured one-hot preparation, whose prefixes come from
    ``initial_state.chain_selection`` of the stored ``amplitudes``. Every
    charge is rounded upward once.
    """
    from sys import float_info

    import numpy as np

    from .binary import NO_WALSH_OMISSION

    walsh = omitted.get("walsh_tables") or (NO_WALSH_OMISSION,) * len(tables)
    records = tuple(
        QHDTableOmission(
            subnormal_entries=int(np.count_nonzero((table.values.array != 0)
                                                   & (np.abs(table.values.array) < float_info.min))),
            walsh_coefficients=w[0], walsh_identity=w[1],
            walsh_identity_charge=_upward(w[2]), walsh_charge=_upward(w[3]))
        for table, w in zip(tables, walsh, strict=True))
    selections = [chain_selection(alpha) for alpha in amplitudes] if chains else []
    counts = {name: omitted.get(name, 0) for name in
              ("projectors", "rotations", "dense_entries", "identity_phases", "identity_events")}
    charges = {name: _upward(omitted.get(name, 0)) for name in
               ("projector_charge", "rotation_charge", "dense_charge", "identity_phase_charge", "identity_charge")}
    return QHDRangeOmissions(tables=records, **counts, **charges,
                             chains=tuple((s.links, len(s.angles), _upward(s.charge)) for s in selections))


def _array_mean(values, probabilities, valid):
    """Return the conditional objective mean of the positive points from their values and probabilities.

    The same binary64 values, probability conversions, multiplication order
    and ``fsum`` calls give the same mean in any iteration order, except
    that an intermediate overflow of the direct ``fsum``, whose
    ``OverflowError`` depends on the order of the terms, can select the
    endpoint route in one order and not in another. The finite
    span branches subtract the endpoint before summation, avoiding
    cancellation for a narrow range near a large endpoint. If the span
    overflows, division by the selected endpoint's magnitude bounds each
    scaled difference. The overflow-sign pass chooses the endpoint only. It
    is not the final mean. ``valid > 0`` and finite positive weights are
    required. An ``fsum`` ``ValueError``, for example from both signed
    infinities in an invalid product stream, is not swallowed by the
    ``OverflowError`` handler.
    """
    import numpy as np

    values = np.asarray(values, dtype=np.float64).ravel(order="C")
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel(order="C")
    if values.size == 0:
        return None
    minimum, maximum = float(values.min()), float(values.max())

    def entries():
        return zip(map(float, values), map(float, probabilities), strict=True)

    if minimum == maximum:
        return minimum
    try:
        mean = fsum(p * v for v, p in entries()) / valid
    except OverflowError:
        mean = inf
    if not isfinite(mean) or not minimum <= mean <= maximum:
        upper = mean > maximum
        if not isfinite(mean):
            objective_scale = max(abs(minimum), abs(maximum))
            upper = fsum((p / valid) * (v / objective_scale)
                         for v, p in entries()) >= 0
        if upper:
            if isfinite(maximum - minimum):
                mean = maximum - fsum((p / valid) * (maximum - v)
                                      for v, p in entries())
            else:
                objective_scale = abs(maximum)
                mean = objective_scale * (1 - fsum(
                    (p / valid) * (1 - v / objective_scale) for v, p in entries()))
        elif isfinite(maximum - minimum):
            mean = minimum + fsum((p / valid) * (v - minimum)
                                  for v, p in entries())
        else:
            objective_scale = abs(minimum)
            mean = objective_scale * (-1 + fsum(
                (p / valid) * (v / objective_scale + 1) for v, p in entries()))
    if not isfinite(mean) or not minimum <= mean <= maximum:
        raise ValueError("QHD weighted objective mean is outside its finite observed domain")
    return mean


def _summarize(plan, weights, invalid, total, window, unavailable=None, shots=None, points=None):
    """Summarize an observed grid population into the fields of ``QHDAnalysis``.

    Summarize the observed valid population in lexicographic grid order.
    Selection considers positive weights only. The candidate minimizes the
    binary64 objective obtained by ``objective_at``, with the first
    lexicographic tuple winning a tie. The probability maximizer and the
    window-selected mode are distinct outputs. Counts are compared and
    summed as integers before division. Probability masses and the
    conditional objective mean use ``fsum``. The mean uses endpoint
    differences if its direct evaluation overflows or leaves the observed
    value interval.

    ``weights`` is the ``(K,)*d`` array of the observed population, float64
    unconditional probabilities or, when ``shots``, the total returned
    shots, is given, integer counts (int64 while the pooled total fits it,
    Python integers in an object array otherwise). With ``points`` the
    population is sparse: ``points`` holds the observed points' grid-index
    tuples as rows, each once and in lexicographic order, and ``weights``
    their 1-D weights, and a point outside ``points`` is unobserved, not a
    measured zero. No flat index of the ``K**d`` grid is formed for it, so
    the grid may exceed the index range. ``invalid``
    is the probability of outcomes that encode no grid point (none for
    binary) and ``total`` the whole observed probability, or None for the
    valid mass plus ``invalid``. The returned dict holds ``value``, the
    ``candidate_*``, ``most_probable_*`` and ``probability_maximizer_*``
    fields, ``mode_status``, ``expected_objective``, ``marginals``,
    ``valid_mass``, ``invalid_mass``, ``observed_mass`` and ``missing``.

    Selection. For two different tuples, the first differing digit
    contributes at least ``K**r`` to the C-order flat index, while all later
    digits contribute at most ``K**r - 1``, so increasing C-order position is
    lexicographic tuple order and NumPy's first-index ``argmin`` and
    ``argmax`` implement the tuple tie rule. With v the values returned by
    ``objective_at`` and the positive mask ``w > 0``, the candidate is
    ``argmin(where(positive, v, inf))``, the computed probability maximizer
    ``argmax(where(positive, w, 0))`` with peak M, and the mode the first
    positive point with ``M - w <= window``; a second such point leaves the
    mode unresolved. The unrestricted predicate ``flatnonzero(M - p <= tie)``
    is wrong when a zero-weight cell precedes the positive cells and
    ``tie >= M``, so the positive mask is required. Counts use integer
    ``M - count`` with window zero. The empty positive case returns no
    candidate, no mean, no mode and ``mode_status="unavailable"``. NaNs are
    not a supported tie case: ``objective_at`` rejects NaN, infinity and
    every nonzero imaginary part.

    The rounded subtraction ``M - weight`` is exact when ``weight >= M/2``
    (Sterbenz's theorem, Higham, *Accuracy and Stability of Numerical
    Algorithms*, 2nd ed., doi:10.1137/1.9780898718027, Theorem 2.5); for
    smaller weights its rounding is at most ``u (M - weight)``. Rounding is
    monotone and fixes the stored binary64 window W, so an exact stored
    deficit at most W always rounds to at most W, and a deficit just above W
    can round down to W and be admitted. The predicate and recorded deficit
    use that rounded subtraction. Comparing every point with M, not
    neighbors with each other, keeps the tie relation well defined. ``window``
    bounds the error of a computed probability difference on the calling
    path (``_readout_window``); when it is None, ``unavailable`` says why and
    the computed probabilities are compared exactly. Under a pairwise error
    bound W, ``abs((q_i - q_j) - (p_i - p_j)) <= W``, a true maximizer t has
    ``p_t - p_s <= q_t - q_s + W <= M - q_s + W``; an exact accepted
    subtraction gives ``p_t - p_s <= 2 W`` and otherwise
    ``p_t - p_s <= W + W/(1 - u)``. ``mode_status`` is ``unavailable``
    without a positive point or a window, ``unresolved`` when a second point
    passes and ``resolved`` otherwise (``QHDAnalysis``, "Mode status").

    Masses and marginals. Probability valid mass is ``fsum`` over the
    observed entries. Counts sum as exact integers and are divided once by
    ``shots``. Each marginal entry is the accurately summed unconditional
    weight of its grid slice: ``fsum`` per slice for probabilities, which is
    not bit parity with sequential additions (the weights
    ``(1, 2**-53, 2**-53)`` give 1 by sequential addition and
    ``1 + 2**-52`` by ``fsum``), and exact integer sums divided once for
    counts. Position moments and refinement rules consume those entries. A
    change in reduction order can change the final bits and can change a
    decision at a numerical tie or threshold. Candidate objective
    comparisons continue to use the stored support entries combined by
    ``objective_at``. The mean keeps the scalar ``fsum`` arithmetic of every
    positive point's ``objective_at`` value (``_array_mean``).
    """
    import numpy as np

    grid = _grid(plan)
    d, k = grid.num_variables, grid.num_grid_points
    shape = (k,) * d
    scale = 1 if shots is None else shots or 1
    w = np.asarray(weights).reshape(-1)
    counts = shots is not None
    positive = w > 0
    if points is not None:
        points = np.asarray(points, dtype=np.int64).reshape(-1, d)

    def decode(position):
        if points is None:
            return tuple(int(i) for i in np.unravel_index(position, shape))
        return tuple(int(i) for i in points[position])

    # Release each chunk's exhausted rows iterator before building the next
    # chunk, and the position array before selector and mean gathers.
    # summary_sizes prices the decoding and mean phases separately.
    values = np.zeros(w.shape, dtype=np.float64)
    positions = np.flatnonzero(positive)
    found = positions.size
    for start in range(0, found, 4096):
        run = positions[start:start + 4096]
        rows = (zip(*(axis.tolist() for axis in np.unravel_index(run, shape))) if points is None
                else map(tuple, points[run].tolist()))
        for position, indices in zip(run.tolist(), rows, strict=True):
            values[position] = objective_at(plan.reconstruction, indices, k)
        rows = None
    positions = run = None
    if counts:
        valid = sum(map(int, w.flat)) / scale
    else:
        valid = fsum(map(float, w.flat))
    total = valid + invalid if total is None else total
    # Marginals: accurate probability sums per grid slice, exact integer sums for counts.
    marginals = np.empty((d, k), dtype=np.float64)
    if points is None:
        grid_weights = w.reshape(shape)
        for axis in range(d):
            for i in range(k):
                cut = [slice(None)] * d
                cut[axis] = i
                block = grid_weights[tuple(cut)]
                marginals[axis, i] = (sum(map(int, block.flat)) / scale if counts
                                      else fsum(map(float, block.flat)))
    else:
        for axis in range(d):
            order = np.argsort(points[:, axis], kind="stable")
            ends = np.searchsorted(points[order, axis], np.arange(k + 1))
            ordered = w[order]
            for i in range(k):
                block = ordered[ends[i]:ends[i + 1]]
                marginals[axis, i] = (sum(map(int, block.flat)) / scale if counts
                                      else fsum(map(float, block.flat)))
    empty = dict.fromkeys(("indices", "coordinates", "probability", "objective", "deficit", "tie_window",
                           "tie_window_unavailable"))
    if not found:
        return dict(
            value=None, candidate_indices=None, candidate_coordinates=None, candidate_probability=None,
            **{"most_probable_" + name: item for name, item in empty.items()},
            **{"probability_maximizer_" + name: None
               for name in ("indices", "coordinates", "probability", "objective")},
            mode_status="unavailable", expected_objective=None, marginals=marginals, valid_mass=valid,
            invalid_mass=invalid, observed_mass=total, missing=("no valid observed candidate",))

    def probability(position):
        return int(w[position]) / scale if counts else float(w[position])

    def coordinates_of(indices):
        return tuple(grid.grid_value(j, i) for j, i in enumerate(indices))

    candidate = int(np.argmin(np.where(positive, values, np.inf)))
    # Nonnegative weights and a nonempty positive set make the maximum positive.
    top = int(np.argmax(np.where(positive, w, 0)))
    peak = int(w[top]) if counts else float(w[top])
    tie = 0 if window is None else window
    tied = positive & ((peak - w) <= tie)
    first = int(np.argmax(tied))
    shared = bool(np.count_nonzero(tied) > 1)
    del tied
    status = "unavailable" if window is None else "unresolved" if shared else "resolved"
    # The compact positive probabilities of the mean, each count divided as a Python integer, gathered
    # with the positive mask after the selector temporaries are released.
    probabilities = (np.fromiter((int(c) / scale for c in w[positive].flat), dtype=np.float64, count=found)
                     if counts else np.asarray(w[positive], dtype=np.float64))
    mean = _array_mean(values[positive], probabilities, valid)
    del probabilities
    mode_indices, top_indices, candidate_indices = decode(first), decode(top), decode(candidate)
    weight = int(w[first]) if counts else float(w[first])
    return dict(
        value=float(values[candidate]),
        candidate_indices=candidate_indices,
        candidate_coordinates=coordinates_of(candidate_indices),
        candidate_probability=probability(candidate),
        most_probable_indices=mode_indices,
        most_probable_coordinates=coordinates_of(mode_indices),
        most_probable_probability=probability(first),
        most_probable_objective=float(values[first]),
        most_probable_deficit=(peak - weight) / scale if counts else peak - weight,
        most_probable_tie_window=window,
        most_probable_tie_window_unavailable=None if window is not None else unavailable,
        probability_maximizer_indices=top_indices,
        probability_maximizer_coordinates=coordinates_of(top_indices),
        probability_maximizer_probability=peak / scale if counts else peak,
        probability_maximizer_objective=float(values[top]),
        mode_status=status,
        expected_objective=mean,
        marginals=marginals,
        valid_mass=valid,
        invalid_mass=invalid,
        observed_mass=total,
        missing=(),
    )


def _pool_counts(histograms, grid, bits, *, exact):
    """Decode measured histograms and pool their counts per grid point as exact integers.

    Returns ``(unique, pooled, invalid, counted)``: the distinct decoded grid
    points as rows in lexicographic order, the C order of the grid, their
    pooled counts, the pooled count of the outcomes that encode no grid point
    and the pooled count of all outcomes. Only the observed outcomes are
    decoded (``decoding.decode_histogram``), one histogram at a time, and no
    flat index of the grid is formed, so the readout may be wider than 64
    bits. The pooled counts are int64, or with ``exact``, which the caller
    sets when the returned shots exceed ``execution.MAX_COUNT``, Python
    integers in an object array. Counts pool in any chunk order; pooling
    ``count/shots`` floats instead can split equal totals, for example
    ``3/10 + 0/10 = 0.3`` against ``1/10 + 2/10 = 0.30000000000000004``.
    """
    import numpy as np

    from .decoding import decode_histogram

    rows, weights = [], []
    invalid = counted = 0
    for histogram in histograms:
        valid, points = decode_histogram(histogram, grid, bits)
        counts = histogram.weights
        counted += sum(map(int, counts.tolist()))
        invalid += sum(map(int, counts[~valid].tolist()))
        rows.append(points)
        weights.append(counts[valid])
    points = np.concatenate(rows) if rows else np.zeros((0, grid.num_variables), dtype=np.int64)
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    if exact:
        pooled = [0] * len(unique)
        for position, count in zip(inverse.tolist(), (c for w in weights for c in w.tolist()), strict=True):
            pooled[position] += count
        pooled = np.array(pooled, dtype=object)
    else:
        pooled = np.zeros(len(unique), dtype=np.int64)
        if weights:
            np.add.at(pooled, inverse, np.concatenate(weights))
    return unique, pooled, invalid, counted


def summary_sizes(plan_or_reconstruction, variables, k, points):
    """Return ``(work, bytes)`` of one ``_summarize`` call over a population of D grid points, ``points`` of them positive.

    Summary admission counts every positive point's stored-table objective
    evaluation, the array selectors and mean passes, and all entries
    consumed by the marginal reductions. Workspace includes the objective
    array, positive/tie masks, selection or gather temporaries and one
    marginal result.

    Let D be the full restricted grid population, P<=D the positive observed
    population, T the support-table count and ``a=sum_t(|S_t|+2)``. The extra
    one in each table's former ``|S_t|+1`` law explicitly counts its input to
    ``objective_at``'s ``fsum``. The same stored table entries and scalar
    objective evaluation serve every positive point. Array selection is over
    a D-entry value array with unobserved/nonpositive entries excluded by
    masks. A sufficient logical work law is
    ``W_summary = P(a+2d) + D(d+50) + 2dK``. The P term covers support
    indexing, lookup, accurate objective summation and decoding original
    coordinates. The dD term counts the dK marginal reductions, each
    consuming ``K**(d-1)`` entries. The 50D reserve covers positive/observed
    masks, extrema, ``where`` writes and arg reductions, peak-minus-weight/tie
    comparisons, candidate gathers, mass reductions and up to the three mean
    passes. It is a declared conservative visit bundle, not an exact FLOP
    count; one reduction input counts as one visit, including ``fsum``.

    For dense float64 probabilities, the incremental array frontier after
    decoding is 25D + 8dK. The positive mask and objective array occupy 9D,
    and two compact float64 mean inputs occupy at most 16D. During decoding,
    the mask, objective and positive-position array occupy at most 17D.
    For C=min(P,4096), reserve H_decode=64C(d+1)+512(d+1) for the chunk's
    intp coordinates, Python coordinate and position lists, integer objects
    and descriptors, or zero when P=0. The integer allowance assumes the
    64-bit intp domain and the qualified CPython object convention. Each
    chunk's iterator is released before the next chunk is constructed. The
    position array is released before selectors and mean gathers.
    Incremental bytes are H0+max(17D+H_decode,25D+8dK), and the
    returned total adds the borrowed 8D probability array. Other held inputs
    are additional. H0 covers fixed bookkeeping, not a D-dependent reserve.
    These are known-buffer and qualified-object bounds, not process RSS.
    Sparse-point and object-count routes require their own storage laws.
    Marginals contribute one 8dK result. P=0 skips objective and mean work.
    H0 is ``compiler.H0``. A reference evolution does not use this allowance.
    """
    from .compiler import H0

    r = getattr(plan_or_reconstruction, "reconstruction", plan_or_reconstruction)
    dimension = k**variables
    a = sum(len(t.support) + 2 for t in r.support_values)
    work = points * (a + 2 * variables) + dimension * (variables + 50) + 2 * variables * k
    chunk = min(points, 4096)
    decode_objects = (64 * chunk * (variables + 1) + 512 * (variables + 1)
                      if points else 0)
    incremental = H0 + max(
        17 * dimension + decode_objects,
        25 * dimension + 8 * variables * k,
    )
    return work, 8 * dimension + incremental


def _admit_observations(plan, observations, *, result=None):
    """Check that supplied observation chunks belong to this Plan, then return their identities.

    Every chunk must name this Plan, its realization, the selected
    observation and bindings, the unconditional population, and register
    layouts equal to the widths declared by the Plan's Program. A classical
    Plan also requires one host-kernel chunk whose source, kernel identity,
    scalar frames, application record and optional stored state match the
    selected kernel. Duplicate chunks or acquisitions raise. With ``result``,
    the chunks must also be the population that result analyzed. Nothing is
    decoded, evaluated or evolved here.

    Returns:
        The tuple of chunk content identities, in order.
    """
    if type(observations) is not ObservationView:
        raise TypeError("QHD requires an ObservationView")
    count = len(observations.chunks)
    if not plan.experiments:
        raise ValueError("resource-only QHD has no observed candidate")
    realization = plan.resolve("qhd")
    selected = realization.resolved_observation(plan)[1]
    kernel = None
    if plan.execution == "classical" and count:
        if count != 1:
            raise ValueError("QHD THEORY analysis requires one selected evolution")
        kernel = plan.construction.kernels[0]
        applications = (_application(plan, kernel.implementation),)
    program = plan.construction.program
    # The two selected maps are new tuples even for an empty supplied view.
    admission = _Admission(program)
    admission.check().require_ready()
    context = admission.expressions(admission.binding_map(program.bindings))
    layouts = []
    for registers in (program.registers, program.classical):
        offset = 0
        layout = []
        for register in registers:
            size = admission.integer(register.width, context, "QHD layout width", positive=True)
            layout.append(RegisterMap(name=register.name, bits=tuple(range(offset, offset + size))))
            offset += size
        layouts.append(tuple(layout))
    ids = []
    for chunk in observations.chunks:
        if (
            chunk.plan_id != plan.content_id
            or chunk.realization_id != realization.content_id
            or chunk.observation != selected
            or chunk.setting != "qhd"
            or chunk.experiment != "qhd"
            or chunk.bindings != realization.bindings
            or chunk.population != "unconditional"
        ):
            raise ValueError("QHD observation differs from its Plan/realization/population")
        if chunk.quantum_layout != layouts[0] or chunk.classical_layout != layouts[1]:
            raise ValueError("QHD observation layout differs from its exact selected register maps")
        if kernel is not None:
            if (
                chunk.execution != "host_kernel"
                or chunk.source != kernel.implementation
                or chunk.selected_kernel_id != kernel.content_id
                or tuple(v.frame for v in chunk.values) != kernel.scalar_frames
                or chunk.applications != applications
            ):
                raise ValueError("QHD THEORY evidence differs from its selected kernel")
            if plan.method.keep_state and (
                len(chunk.artifacts) != 1 or chunk.artifacts[0].output != kernel.outputs[0]
            ):
                raise ValueError("QHD host stored state differs from its selection")
        ids.append(chunk.content_id)
    if len(set(ids)) != len(ids) or len({c.acquisition_key for c in observations.chunks}) != count:
        raise ValueError("duplicate QHD acquisition")
    if result is not None and (
        tuple(ids) != result.contribution_ids or observations.content_id != result.observation_id
    ):
        raise ValueError("QHD verification observations differ from the analyzed population")
    return tuple(ids)


def _scalar_names(d, k):
    """Labels of the host-kernel scalars, in the order the kernel publishes them.

    ``readout_state_error`` is the first-order state budget delta from which
    the kernel's most-probable tie window was derived (``_execute_theory``).
    ``mode_unresolved`` is 1 when a point other than the tie representative
    passes the tie test and 0 otherwise (``_summarize``), since a kernel
    always has a derived window and a positive valid point. There are
    ``d K + 3 d + 13`` labels.
    """
    return (
        "candidate_objective",
        "candidate_probability",
        "most_probable_objective",
        "most_probable_probability",
        "most_probable_deficit",
        "probability_maximizer_objective",
        "probability_maximizer_probability",
        "mode_unresolved",
        "readout_state_error",
        "expected_objective",
        "valid_mass",
        "invalid_mass",
        "observed_mass",
        *(f"candidate_index_{j}" for j in range(d)),
        *(f"most_probable_index_{j}" for j in range(d)),
        *(f"probability_maximizer_index_{j}" for j in range(d)),
        *(f"marginal_{j}_{i}" for j in range(d) for i in range(k)),
    )


def restricted_sizes(dimension, variables, table_count, steps, flavor, generator_norms, start_work):
    """Return ``(size_units, workspace_bytes)`` of one restricted host evolution of the ``schrodinger`` or one-hot ``ir_product`` flavor.

    The ``split_step`` flavor has its own law, ``split_step.sizes``, and the
    binary ``ir_product`` flavor ``theory.binary_product_sizes``; neither
    depends on the generator norms.
    D is the restricted dimension ``K**d`` and ``steps`` is the pair
    ``(step_weights, block_steps)``. ``start_work`` is the work
    ``W_start`` of forming the start vector once per evolution
    (``initial_state.start_vector_work``), D for the direct uniform fill and
    ``d K + sum_(j=2)^d K**j + D`` for the stored real factors.

    One-hot ``ir_product``. The one-hot product reference applies each
    stored projector phase or two-slice hopping directly. Admission counts
    every block occurrence and the largest slice-copy/expression workspace
    together with the live state. Its one implementation is
    ``theory.onehot_product_sizes``: for B_P projector occurrences with
    ``S_P = D/K**s`` and B_H hopping occurrences with ``S_H = D/K``,
    ``W_onehot = W_start + sum_P (D + 3 S_P + 2s + 5) + sum_H (12 S_H + 8) + D``
    and ``B_onehot = H0 + max{B_start, 16D + max(max_P 32 S_P, max_H 80 S_H)}``
    with the start vector's explicit payload ``B_start = 24 D + 8 d K``
    (``initial_state.restricted_state``). This route has no cached Pauli
    matrices, shifted-CSR workspace or expm call terms, and
    ``generator_norms`` is empty (``_generator_norms``).

    ``schrodinger`` makes one ``expm_multiply`` call per step.
    ``generator_norms`` gives the pair ``(N, r)`` of each call in call
    order and is read once (``_generator_norms``). A call's work and bytes
    ``W_exp,k`` and ``B_exp,k`` are ``_linalg_laws.expm_multiply_requirements``
    for a generator with at most ``r D`` stored entries and shifted 1-norm
    at most N; ``B_exp,k`` funds SciPy's identity, shifted CSR copy and
    Taylor/norm-estimation work.

    The Schrodinger law separates construction from evolution. Construction
    counts the raw COO slots, their row and column arrays, index conversion
    and the CSR conversion peak. Evolution holds one kinetic CSR pattern, an
    updated complex data array and the diagonal buffers, and adds the
    largest SciPy-call workspace. Duplicate diagonal contributions and the
    doubled K = 2 periodic link enter the raw-slot count. The byte law
    bounds the named numerical buffers and does not claim to bound process
    RSS.

    Schrodinger evolution stores one kinetic CSR pattern and updates a
    private generator value array in the selected multiply/add order.
    Admission includes the complex potential, diagonal slots/workspace and
    SciPy's shifted-matrix and exponential-action workspace.

    The stencil's raw slots S and stored entries M are ``S = 3dD``
    (periodic) or ``dD + 2d(K-1)D/K`` (Dirichlet) and ``M = (1+2d)D``
    (periodic, K > 2), ``(1+d)D`` (periodic, K = 2) or ``D + 2d(K-1)D/K``
    (Dirichlet) (``theory.restricted_kinetic_sparse``). This function does
    not see the boundary, so it uses ``S = 3dD`` and ``z = M = (1+2d)D``,
    which are at least every case's counts, and the index width
    ``I = w = 8``; every term below increases with S, z and I. The
    construction peak of the preallocated fill is
    ``B_stencil = B_build = 32S + 96D + 2wS + (16+w)S + w(D+1) + (16+w)M``,
    and the shared kinetic payload is ``B_K = (16+I)z + I(D+1)``. With the
    start vector's explicit payload ``B_start = 24 D + 8 d K``
    (``initial_state.restricted_state``) and H0 (``compiler.H0``) for
    scalar and short-lived headers,
    ``workspace_bytes = H0 + max{B_start, B_stencil + 16D, B_K + 40D,
    B_K + 16D + (17+I)z + (2I+24)D, B_K + 16z + 56D + max_k B_exp,k}``.
    The 16D beside the stencil is the live state. Setup forms diagonal
    slots, private generator data and the diagonal buffer before the
    potential exists. Its row-index array is released on return. The
    subsequent real-to-complex conversion has peak B_K+16z+64D, including
    the setup outputs, state and simultaneous real and complex potentials.
    Each fill has peak B_K+16z+72D because the diagonal gather adds 16D
    to the persistent 56D vectors. Execution has B_K+16z+56D+B_exp.
    Every priced exponential call reserves B_exp=64(e+D)+512D>=512D,
    so execution covers conversion by at least 504D and fill by at least
    496D. A positive-step QHD schedule supplies at least one such call,
    even for zero norm. No additional maximum term is required for these
    phases. The B_K+40D term is also dominated. Other Plan and summary
    payloads are charged at their owners.

    ``size_units = W_start + W_stencil + (T+2)D + Td + (5z+6D+1)
    + sum_k [2z + 5D + W_exp,k] + D`` for T support tables. The potential is
    one zero fill, T broadcast additions and one complex conversion, and Td
    prices its support-shape bookkeeping. Setup charges D each for arange,
    diff, a possible repeat-count cast, count validation/summation and group
    traversal, z repeated outputs and z equality tests. flatnonzero counts
    and scans z Boolean entries and can write z indices in its branchless
    path. The empty CSR initializes D+1 pointers, giving 5z+6D+1.
    Each fill charges two z multiplications and D each for potential
    multiplication, diagonal gather, addition, scatter-index validation
    and scatter writes. NumPy 2.5.2's get checks indices within its gather,
    while its set checks them in a separate pass. The last D is the final
    phase application when executed.
    W_stencil=28S+4M+(37d+12)D+40(d+1) is the logical construction
    charge owned by theory.restricted_kinetic_work. It includes the raw
    row, column and value writes, index/mask passes and the qualified
    sorted COO-to-CSR conversion. This boundary-free owner supplies the
    upper counts S=3dD and M=(1+2d)D to that monotone law.
    """
    from nwqlib._linalg_laws import expm_multiply_requirements

    if flavor == "ir_product":
        from .theory import onehot_product_sizes

        _require_binary64_grid(dimension, variables)
        k = _grid_points(dimension, variables)
        return onehot_product_sizes(dimension, k, steps[1], start_work=start_work,
                                    start_bytes=24 * dimension + 8 * variables * k, held_bytes=0)
    from .compiler import H0
    from .theory import restricted_kinetic_work

    k = _grid_points(dimension, variables)
    tables = table_count
    raw = 3 * variables * dimension
    entries = (1 + 2 * variables) * dimension
    index = 8
    build = (32 * raw + 96 * dimension + 2 * index * raw + (16 + index) * raw + index * (dimension + 1)
             + (16 + index) * entries)
    kinetic = (16 + index) * entries + index * (dimension + 1)
    stencil_work = restricted_kinetic_work(dimension, variables, raw, entries)
    size = (start_work + stencil_work + (tables + 2) * dimension + tables * variables
            + 5 * entries + 6 * dimension + 1 + dimension)
    call_bytes = 0
    for norm, rows in generator_norms:
        work, data_bytes = expm_multiply_requirements(dimension, rows * dimension, norm)
        size += 2 * entries + 5 * dimension + work
        call_bytes = max(call_bytes, data_bytes)
    workspace = H0 + max(24 * dimension + 8 * variables * k, build + 16 * dimension, kinetic + 40 * dimension,
                         kinetic + 16 * dimension + (17 + index) * entries + (2 * index + 24) * dimension,
                         kinetic + 16 * entries + 56 * dimension + call_bytes)
    return size, workspace


def _grid_points(dimension, variables):
    """Return the integer K with ``K**variables == dimension``, found without a float of the dimension."""
    from math import log2

    k = max(2, round(2 ** (log2(dimension) / variables)))
    while k**variables > dimension:
        k -= 1
    while (k + 1) ** variables <= dimension:
        k += 1
    return k


def _generator_norms(flavor, method, grid, support_values, steps, step_weights):
    """Yield the shifted norm and row-population bound of each restricted expm call.

    The Schrodinger pairs come from _schrodinger_step_bounds, including the
    stored kinetic matrix, exact table ranges, broadcast summation error and
    generator-formation error. N bounds the assembled generator after an exact
    trace shift, and r=1+2*d bounds its row population. The SciPy roundoff model
    also assumes N bounds its computed shifted matrix. The ordered scalar
    kinetic calculation uses the constant-diagonal premise of
    _schrodinger_kinetic_bounds. No kinetic CSR is constructed here.
    The direct one-hot ir_product kernel makes no expm call and yields no pair.
    Consumers read the pairs once for work admission or numerical error.

    Raises:
        ValueError: The stored kinetic diagonal overflows, or a step's norm or
            perturbation bound is not finite.
    """
    if flavor == "ir_product":
        return iter(())
    return ((norm, rows) for norm, rows, _ in
            _schrodinger_step_bounds(method, grid, support_values, step_weights))


def _step_norm_refusal(step, norm, dt, a, b, kinetic, spread):
    """Describe the first nonfinite operation in dt*(abs(a)*H + abs(b)*R).

    The unscaled products and sum precede dt. A change of num_steps or
    total_time also changes the weights, so reducing dt is a remedy only
    with those weights fixed. Midpoint potential contributions need not
    decrease when the step count increases. Integrated potential
    contributions decrease on the last nested interval, but their
    unscaled averages can still overflow.
    """
    pa, pb = abs(a) * kinetic, abs(b) * spread
    total = pa + pb
    kinetic_text = (
        f"|a| * H = {pa!r}, with a = {a!r} and stored kinetic column bound H = {kinetic!r}. "
        "Use fewer grid points or wider boxes to lower H. For CubicSchedule and "
        "ShiftedCubicSchedule, a larger s lowers the kinetic weight"
    )
    potential_text = (
        f"|b| * R = {pb!r}, with b = {b!r} and table-range upper bound R = {spread!r}. "
        "Scale the objective down. At fixed num_steps, a shorter total_time lowers a "
        "time-dependent potential weight"
    )
    if not isfinite(pa) or not isfinite(pb):
        details = []
        if not isfinite(pa):
            details.append("The unscaled kinetic product overflows: " + kinetic_text)
        if not isfinite(pb):
            details.append("The unscaled potential product overflows: " + potential_text)
    elif not isfinite(total):
        details = [
            "The sum |a| * H + |b| * R overflows before multiplication by dt. "
            "Reduce an unscaled contribution. Reducing dt alone does not repair this sum",
            kinetic_text, potential_text,
        ]
    else:
        details = [
            f"The unscaled products {pa!r}, {pb!r} and their sum {total!r} are finite, "
            "but the final multiplication by dt overflows. With a, b, H and R fixed, "
            "a smaller dt lowers this product. A shorter total_time at fixed num_steps "
            "lowers the scaled potential contribution. Changing num_steps recomputes "
            "the weights, so increasing it can raise midpoint contributions or cause "
            "an unscaled product to overflow",
            kinetic_text, potential_text,
        ]
    return (f"the schrodinger step generator norm of step {step} is {norm!r}, "
            f"not a finite binary64 number, for dt = {dt!r}. "
            + ". ".join(details) + ". Other Plan range and resource limits still apply")


def _host_start_error(plan):
    """Return the construction error, in units of u, of the classical kernel's initial vector.

    It is ``initial_state.restricted_state_error`` of the Method's initial
    state on the Plan's grid, the start term of every flavor's state budget
    (``_host_state_error``), which planning evaluated once and stored
    (``QHDReconstruction.initial_state_error``). Nothing is evaluated here.
    """
    return plan.reconstruction.initial_state_error


def _host_state_error(method, grid, tables, steps, step_weights, start):
    """Return the first-order 2-norm error budget delta of the classical kernel's final state.

    ``tables``, ``steps`` and ``step_weights`` are the Plan's stored support
    tables, compiled steps and step rows (``QHDReconstruction``), and
    ``start`` the construction error of the start vector
    (``QHDReconstruction.initial_state_error``). Every flavor starts from the
    Method's exact initial state, with ``start`` pricing its numerical
    construction. The ``schrodinger`` kernel uses ``_schrodinger_state_error`` against the
    exact stored-table sum and stored kinetic matrix. It adds broadcast
    summation and generator-formation perturbations to the conditional
    first-order ``expm_multiply_state_error`` charge. ``_generator_norms`` uses
    the same assembled-generator bounds for work admission. One-hot ``ir_product`` uses
    ``theory.onehot_product_state_error``: the one-hot product reference is
    the ordered product of analytic projector and hopping blocks at their
    stored binary64 angles, applied to the exact initial state. Under normal
    round-to-nearest arithmetic and the stated one-ulp elementary-function
    premise, its first-order state-operation budget is
    ``u*(start+10*B_P+6*B_H+5)``. The counts include every executed
    occurrence and the final term covers the physical-phase application.
    Parameter formation, native lowering, time splitting and grid errors are
    outside this budget. The mass and mode windows use the same
    state-to-probability conversions as the other classical kernels. Binary
    ``ir_product`` uses ``theory.binary_product_state_error`` relative to the
    exact product at its QFT angles and computed diagonal phase arrays. That
    budget does not price formation of those arrays or their difference from
    separate native rotations. ``split_step`` uses
    ``split_step.state_error``, which includes transform roundoff and
    phase-angle formation. The Schrodinger and split-step references
    omit the objective constant, whose global phase is applied after
    their probability readout (``_constant_phase``).

    Planning evaluates it once for a classical Plan (``_admit_host_windows``)
    and stores it with the mass window and the ceiling of the tie window
    for that reference (``QHDReconstruction.host_state_error``,
    ``host_mass_window`` and ``host_tie_window``), which later stages read.
    The split-step kernel selects with its smaller observed trajectory
    budget, capped by this stored value (``split_step.evolve``). No state is
    evolved here.
    """
    flavor = method.theory_flavor
    if flavor == "ir_product" and method.encoding == "binary":
        from .theory import binary_product_state_error

        return binary_product_state_error(_bits(method), steps, cutoff=method.binary_synthesis.aqft_cutoff,
                                          start=start)
    if flavor == "ir_product":
        from .theory import onehot_product_state_error

        return onehot_product_state_error(steps, start=start)
    if flavor == "split_step":
        from .split_step import state_error

        dt = method.total_time / method.num_steps
        return state_error(grid, method.kinetic_model, tables, step_weights, dt, start=start)
    return _schrodinger_state_error(method, grid, tables, step_weights, start)


def _host_probability_window(plan):
    """Return the admitted |mass - 1| of the classical kernel's final state.

    It is ``state_mass_window`` of the kernel's state budget
    (``_host_state_error``) and the restricted dimension ``K**d``, which adds
    the evaluation and summation of the ``K**d`` probabilities. The same
    budget gives the tie window of ``_host_tie_window``. Planning computed
    it once (``_admit_host_windows``), and this reads the stored value
    ``QHDReconstruction.host_mass_window``.
    """
    return plan.reconstruction.host_mass_window


def _host_window(delta):
    """Return the tie window of a classical kernel's probabilities whose state lies within delta of its model.

    ``_execute_theory`` evaluates each probability as ``np.abs(state)**2``
    (``ABSOLUTE_SQUARE_ROUNDOFF``). For distinct grid points i and j the
    diagonal operator ``|i><i| - |j><j|`` has norm one, so a computed state
    within delta of the model's unit state up to a global phase moves the
    difference of their probabilities by at most ``2 delta + delta**2``, and
    evaluating the two probabilities with relative error ``e = 5u`` adds
    ``e (1 + delta)**2``. The difference of two computed probabilities
    therefore lies within ``probability_difference_window(delta, 5u)`` of the
    exact difference, and so does each single probability.
    """
    return probability_difference_window(delta, ABSOLUTE_SQUARE_ROUNDOFF * UNIT_ROUNDOFF)


def _host_tie_window(plan):
    """Return the Plan's ceiling on the most-probable tie window of the classical kernel.

    It is ``_host_window`` of ``_host_state_error``, the Plan budget of
    ``_host_probability_window``. The ``schrodinger`` and ``ir_product``
    kernels select with it. The ``split_step`` kernel selects with the
    smaller window of the budget it observed on its trajectory
    (``split_step.evolve``), and this Plan window is the ceiling that the
    recorded window must not exceed (``_executed_host_window``). Planning
    computed it once (``_admit_host_windows``), and this reads the stored
    value ``QHDReconstruction.host_tie_window``. No state is evolved here.
    """
    return plan.reconstruction.host_tie_window


def _require_binary64_grid(dimension, variables):
    """Refuse a classical grid of ``dimension = K**d`` points that exceeds the binary64 range, naming its size.

    The classical kernels' size laws and probability windows convert the
    grid size to binary64.
    """
    try:
        float(dimension)
    except OverflowError:
        raise ValueError(
            f"the classical kernel's grid of {dimension} points, num_grid_points**{variables}, lies beyond the "
            f"binary64 range; use a smaller QHD(num_grid_points) or fewer variables") from None


def _admit_host_windows(method, grid, tables, steps, step_weights, start, dimension):
    """Return the stored host state budget and windows of a classical Plan, refusing ones beyond the binary64 range.

    The arguments are the Plan's data that ``_host_state_error`` reads and
    the restricted dimension ``K**d``. The returned mapping holds the state
    budget delta (``_host_state_error``), the mass window
    ``state_mass_window(delta, K**d)`` (``_host_probability_window``) and
    the tie-window ceiling ``_host_window(delta)`` (``_host_tie_window``),
    the fields ``host_state_error``, ``host_mass_window`` and
    ``host_tie_window`` of ``QHDReconstruction``. Both windows of every
    flavor are ``_validation.probability_difference_window`` of delta,
    whose ``delta**2`` overflows once delta exceeds about 1.3e154 although
    the evolution itself can finish.
    ``2 delta + delta**2 + e (1 + delta)**2`` increases with delta for
    ``e >= 0``. The producer uses the Plan budget or an explicitly capped
    observed budget (the ``split_step`` kernel returns the smaller of its
    observed budget and ``split_step.state_error``), and readback checks
    the resulting window against the finite Plan ceiling
    (``_executed_host_window``), so a finite ceiling cannot admit an
    infinite window. The check concerns window formation and valid recorded
    evidence. It does not certify every intermediate of an evolution.
    Planning calls this once before the reconstruction is formed. Nothing
    is evolved here.

    Raises:
        ValueError: The grid size ``K**d`` or the budget or one of the two
            windows is not a finite binary64 number.
    """
    _require_binary64_grid(dimension, grid.num_variables)
    delta = _host_state_error(method, grid, tables, steps, step_weights, start)
    values = {"state error budget": delta}
    if isfinite(delta):
        # The mass window of _host_probability_window and the tie-window ceiling of _host_tie_window.
        values.update({"mass window": state_mass_window(delta, dimension), "tie window": _host_window(delta)})
    for name, value in values.items():
        if not isfinite(value):
            raise ValueError(
                f"the {name} of the classical QHD {method.theory_flavor} kernel is {value!r}"
                + ("" if name == "state error budget" else f" for the Plan's state error budget delta = {delta!r}")
                + ", not a finite binary64 number, so no Result could record its probability windows. The "
                "budget adds the roundoff charges of every step (method._host_state_error), which on the "
                "split_step and schrodinger flavors grow with the step exponents, the kinetic energies and "
                "the objective values"
            )
    return dict(host_state_error=delta, host_mass_window=values["mass window"],
                host_tie_window=values["tie window"])


def _executed_host_window(delta, ceiling):
    """Return the tie window of the kernel's recorded state budget after checking it against the Plan ceiling.

    ``delta`` is the kernel's ``readout_state_error`` scalar and ``ceiling``
    the Plan's window (``_host_tie_window``), which the kernel's application
    records as ``tie_window_ceiling``. The budget must be a nonnegative
    number and its window ``_host_window(delta)`` at most the ceiling. The
    producer uses the Plan budget or an explicitly capped observed budget,
    and readback checks the resulting window against the finite Plan
    ceiling. The comparison is of windows, and rounded windows can be equal
    for distinct budgets, so it does not by itself prove that a recorded
    budget is at most the Plan's. The check does not certify the budget's
    computation, and it reads stored evidence only, with no evolution.
    """
    if delta is None or not delta >= 0:
        raise ValueError("QHD host readout_state_error must be a nonnegative number")
    window = _host_window(delta)
    if not window <= ceiling:
        raise ValueError("QHD host tie window exceeds the ceiling derived from its Plan")
    return window


def _readout_window(chunks, receipts):
    """Return ``(window, reason)`` for the most-probable selection over ``chunks``.

    ``receipts`` maps preparation identities to receipts. The window bounds
    how far the computed difference of two valid-point probabilities can lie
    from the difference of the model that the path evaluates
    (``probability_difference_window``), so points within it of the computed
    maximum are ties:

    - host kernel (one chunk): ``_host_window`` of the kernel's recorded
      ``readout_state_error``, at most the ``tie_window_ceiling`` that its
      application records (``_executed_host_window``);
    - counts: 0, because pooled integer counts are compared exactly;
    - exact probabilities from C chunks: with each chunk's state budget
      ``delta_c``, ``E_c = probability_difference_window(delta_c, 2u)``
      (``COMPONENT_SQUARES_ROUNDOFF``) and ``R_c = (1 + delta_c)**2``, the
      window is ``mean_c E_c + rho_C mean_c (1 + 2u) R_c``. ``analyze``
      divides each probability by C and adds it to its grid point, and one-hot or
      binary decoding maps a basis index to at most one point, so a point has at
      most C contributions and each passes at most C roundings, ``rho_C =
      gamma_C = C u/(1 - C u)``. With one chunk the division by one and the
      addition to zero are exact, ``rho_1 = 0``. The reference population of
      several chunks is the mean of their exact native populations. The
      state term is averaged, not divided by ``sqrt(C)``, because the
      chunks' numerical errors can be correlated. This
      pooling term widens the tie window of the pooled point probabilities.
      The masses have their own summation allowance,
      ``QHDAnalysis.pooled_summation_roundoff``;
    - stored amplitudes (one chunk): ``probability_difference_window(delta,
      5u)`` for the decoder's ``np.abs(z)**2`` (``ABSOLUTE_SQUARE_ROUNDOFF``).
      That decoder bounds its own point probabilities, so it resolves the
      receipt's ``"amplitude-derived masses"`` exclusion for this use only.

    ``delta_c`` is the receipt's ``state_error``, and for stored amplitudes
    its ``saved_state_error``, which adds the charge of a host phase
    correction of the saved state. A chunk without a receipt or
    a receipt outside that derivation returns ``(None, reason)``. The
    selection then compares computed probabilities exactly, which decides a
    tie by roundoff without any guarantee that the selected point is the
    model's most probable one. An excluded instruction, such as the
    ``unitary`` that Qiskit's state preparation lowers to, is such a case.
    The instruction count does not establish the per-instruction roundoff
    charge of the receipt's derivation for it. Applying a dense unitary on b
    amplitudes has the normwise rounding bound
    ``sqrt(2) gamma_(2b) sqrt(b) ||x||``, about ``2 sqrt(2) b**1.5 u ||x||``.
    On a register of at most 6 qubits (b <= 64) its coefficient, 181, 512
    and 1448 at b = 16, 32 and 64, stays below Aer's charge of about 3479 u
    per instruction, half of
    ``nwqlib._validation.AER_STATEVECTOR_ROUNDOFF_PER_OPERATION``. There
    the obstacle is the error of the matrix that Qiskit supplies, which the
    receipt cannot bound. From 7 qubits on, the coefficient, 4096 at
    b = 128, exceeds that charge before any error of the supplied matrix,
    which shows that the derivation does not cover the instruction, not
    that its actual error is that large. The reason names the exclusion, and the
    Result shows it in ``most_probable_tie_window_unavailable`` and in its
    summary line. Without chunks there is nothing to select and
    ``(None, None)`` is returned.
    """
    if not chunks:
        return None, None
    if chunks[0].execution == "host_kernel":
        (chunk,) = chunks
        (application,) = chunk.applications
        arguments = {b.parameter: b.value for b in application.arguments}
        delta = next((v.value for v in chunk.values if v.label == "readout_state_error"), None)
        return _executed_host_window(delta, arguments["tie_window_ceiling"].value), None
    kind = chunks[0].observation.kind
    if kind == "counts":
        return 0.0, None
    resolved = ("amplitude-derived masses",) if kind == "amplitudes" else ()
    deltas = []
    for chunk in chunks:
        receipt = receipts.get(chunk.prepared_id)
        if receipt is None:
            return None, "exact chunk without its preparation receipt"
        # The amplitude branch analyzes a saved statevector, so it uses the
        # saved-state budget; probability and Pauli saves do not consume a
        # multiplied array and keep the native readout budget.
        delta, reason = (receipt.saved_state_error(resolved) if kind == "amplitudes"
                         else receipt.state_error(resolved))
        if delta is None:
            return None, reason
        deltas.append(delta)
    if kind == "amplitudes":
        (delta,) = deltas
        return probability_difference_window(delta, ABSOLUTE_SQUARE_ROUNDOFF * UNIT_ROUNDOFF), None
    u = UNIT_ROUNDOFF
    count = len(deltas)
    # gamma_C bounds the C roundings of each pooled contribution, none for C = 1.
    rho = 0.0 if count == 1 else count * u / (1 - count * u)
    evaluation = COMPONENT_SQUARES_ROUNDOFF * u
    # The mean of E_c, the state and evaluation part of each chunk's pair bound.
    window = fsum(probability_difference_window(d, evaluation) for d in deltas) / count
    # rho_C times the evaluated pair mass (1 + 2u)(1 + delta_c)**2, averaged.
    return window + rho * fsum((1 + evaluation) * (1 + d) ** 2 for d in deltas) / count, None


def _application(plan, source):
    """Describe one host evolution by its dimension, numerical calls and roundoff windows.

    The ``schrodinger`` flavor records its ``expm_multiply`` call count, one
    per step, and ``ir_product``, which applies its blocks directly on both
    encodings, records zero such calls. The
    ``split_step`` flavor records its steps and its ``2 d`` transforms per
    step, and its Source names the kinetic model and the coefficient rule.
    The zero objective-evaluation and companion counts state that a classical
    QHD run reads the stored tables and performs no second evolution. The
    recorded ``probability_window`` is the kernel's counterpart of a circuit
    receipt's ``probability_window`` and bounds its reported masses. The
    recorded ``tie_window_ceiling`` (``_host_tie_window``) is the Plan's
    ceiling on the window of the kernel's most-probable selection, which the
    kernel derives from its ``readout_state_error`` scalar
    (``_readout_window``). Both depend on the Plan alone, so the record is
    the same for every execution of the Plan.
    """
    r, options = plan.reconstruction, plan.method
    if options.theory_flavor == "split_step":
        calls = (
            Binding(parameter="split_steps", value=options.num_steps),
            Binding(parameter="transforms", value=2 * len(plan.problem.variables) * options.num_steps),
        )
    else:
        # One expm_multiply call per step (schrodinger). The one-hot and binary products apply their
        # blocks without expm_multiply.
        calls = (
            Binding(
                parameter="expm_multiply_calls",
                value=options.num_steps if options.theory_flavor == "schrodinger" else 0,
            ),
        )
    return KernelApplication(
        name="qhd",
        implementation=source,
        arguments=(
            Binding(parameter="restricted_dimension", value=r.restricted_dimension),
            *calls,
            Binding(parameter="objective_evaluations", value=0),
            Binding(parameter="companion_evolutions", value=0),
            Binding(parameter="probability_window", value=Float64(value=_host_probability_window(plan))),
            Binding(parameter="tie_window_ceiling", value=Float64(value=_host_tie_window(plan))),
        ),
    )


def _host_summary(plan, values):
    """Rebuild analysis fields from the host kernel's labeled scalars.

    The objectives of the candidate, the most probable point and the
    probability maximizer are compared with the Plan's
    stored support tables, so a host value cannot disagree with the selected
    objective. Masses use the kernel's derived roundoff window,
    ``_host_probability_window``. The most-probable tie window is
    ``_host_window`` of the kernel's ``readout_state_error``, the budget it
    selected with, checked against the Plan's ceiling ``_host_tie_window``
    (``_executed_host_window``), and the selected probability and deficit
    are the kernel's own scalars. The kernel always has a window, so its
    ``mode_unresolved`` scalar, which must be 0 or 1, maps to the
    ``resolved`` or ``unresolved`` status. The kernel's d K
    ``marginal_{j}_{i}`` scalars fill the (d, K) float64 array
    ``QHDAnalysis.marginals`` in variable and grid order. Nothing is evolved.
    """
    import numpy as np

    stats = {v.label: v.value for v in values}
    d, k = len(plan.problem.variables), plan.method.num_grid_points
    required = _scalar_names(d, k)
    if tuple(v.label for v in values) != required:
        raise ValueError("QHD host statistics differ from selected labels")
    points = {}
    for name in ("candidate", "most_probable", "probability_maximizer"):
        indices = tuple(stats[f"{name}_index_{j}"] for j in range(d))
        if any(i is None or i != int(i) or not 0 <= i < k for i in indices):
            raise ValueError("QHD THEORY requires a valid observed candidate, most probable point and "
                             "probability maximizer")
        indices = tuple(int(i) for i in indices)
        expected = objective_at(plan.reconstruction, indices, k)
        if stats[f"{name}_objective"] != expected:
            raise ValueError(f"QHD {name.replace('_', '-')} objective differs from its admitted support table")
        points[name] = (indices, expected)
    if stats["mode_unresolved"] not in (0.0, 1.0):
        raise ValueError("QHD host mode_unresolved must be 0 or 1")
    window = _host_probability_window(plan)
    for name in ("valid_mass", "invalid_mass", "observed_mass"):
        validate_normalized_mass(stats[name], window)
    grid = _grid(plan)
    (indices, expected), (mode, mode_objective) = points["candidate"], points["most_probable"]
    top, top_objective = points["probability_maximizer"]
    return dict(
        value=expected,
        candidate_indices=indices,
        candidate_coordinates=tuple(grid.grid_value(j, i) for j, i in enumerate(indices)),
        candidate_probability=stats["candidate_probability"],
        most_probable_indices=mode,
        most_probable_coordinates=tuple(grid.grid_value(j, i) for j, i in enumerate(mode)),
        most_probable_probability=stats["most_probable_probability"],
        most_probable_objective=mode_objective,
        most_probable_deficit=stats["most_probable_deficit"],
        most_probable_tie_window=_executed_host_window(stats["readout_state_error"], _host_tie_window(plan)),
        most_probable_tie_window_unavailable=None,
        probability_maximizer_indices=top,
        probability_maximizer_coordinates=tuple(grid.grid_value(j, i) for j, i in enumerate(top)),
        probability_maximizer_probability=stats["probability_maximizer_probability"],
        probability_maximizer_objective=top_objective,
        mode_status="unresolved" if stats["mode_unresolved"] else "resolved",
        expected_objective=stats["expected_objective"],
        valid_mass=stats["valid_mass"],
        invalid_mass=stats["invalid_mass"],
        observed_mass=stats["observed_mass"],
        marginals=np.array([[stats[f"marginal_{j}_{i}"] for i in range(k)] for j in range(d)], dtype=np.float64),
        missing=(),
    )


def _evolve_restricted(plan, flavor, observed=None):
    """Return the final state of one host evolution on the K**d valid grid points.

    ``observed``, when it is a list, receives the first-order state budget
    that the ``split_step`` flavor observes on its own trajectory
    (``split_step.evolve``). The other flavors append nothing, and their
    budget is the Plan's (``_host_state_error``).

    The state has length K**d in lexicographic grid-index order. The
    objective enters through the stored tables or compiled blocks and is
    not evaluated again. Every flavor starts from the Method's initial state
    on the valid grid points (``initial_state.restricted_state``). A general
    product state is the tensor product of the per-variable vectors that
    planning stored (``QHDReconstruction.initial_amplitudes``). The uniform
    state and the kinetic ground state on the periodic grid are filled
    directly with ``1/sqrt(K**d)``, whose own construction term is 2u. The
    ``schrodinger`` and ``split_step`` states omit the global phase
    ``exp(i phi)`` of the objective constant (``_constant_phase``), because V
    below is the sum of the support tables alone (``split_step.potential_diagonal``), and
    ``_execute_theory`` restores that phase on the kept state. The
    ``ir_product`` state carries the compiler's physical phase, which
    includes the constant.

    ``schrodinger`` applies ``exp(-i dt (a_k K + b_k V))`` with the stored
    step weights, so it has no splitting error. Under the ``"midpoint"``
    rule these are the point values at the step midpoint, the standard
    exponential midpoint rule. Under the ``"integrated"`` rule they are the
    step averages, so the exponent is the first Magnus term
    ``-i (A_k K + B_k V)`` (``schedules.step_weights``). In ideal arithmetic,
    integrated coefficients give the exponential of the first Magnus term.
    It equals the time-ordered step when the Hamiltonians at different times
    commute. Midpoint coefficients also have coefficient-quadrature error,
    which need not vanish when the Hamiltonians commute. A constant schedule
    removes both time-ordering and coefficient-quadrature errors. The
    numerical exponential action still has its own floating-point and solver
    error. ``ir_product`` evaluates a numerical reference from the Plan's selected
    blocks and restores the physical phase. One-hot blocks apply their
    analytic restricted actions at the stored angles (``theory._run_ir_product``). Binary blocks use
    QFT operations and computed diagonal phase arrays on the whole binary
    register (``theory.run_binary_product``), whose Walsh reconstruction
    can differ from the separate native rotations. The binary encoding changes the circuit,
    not the finite model, so the other flavors evolve the same model for both
    encodings. ``split_step`` applies
    ``exp(-i dt b_k V/2) exp(-i dt a_k K) exp(-i dt b_k V/2)`` with the
    kinetic factor diagonal in the kinetic eigenbasis and evaluated
    numerically by the selected transforms
    (``split_step.evolve``), so it adds the Strang splitting error to the
    time error of the coefficient rule.
    """
    import numpy as np
    import scipy.sparse.linalg
    from nwqlib._linalg_laws import seeded_norm_estimates
    from .initial_state import restricted_state
    from .theory import _run_ir_product, restricted_kinetic_sparse, run_binary_product
    from .native import raw_blocks

    grid = _grid(plan)
    if flavor == "ir_product" and not plan.reconstruction.compact_schedule_selected:
        raise ValueError("IR reference was not selected by this Plan")
    dt = plan.method.total_time / plan.method.num_steps
    state = restricted_state(plan.method.initial_state, grid, plan.reconstruction.initial_amplitudes)
    if flavor == "ir_product":
        if plan.method.encoding == "binary":
            state = run_binary_product(grid, plan.method, plan.reconstruction, state)
        else:
            state = _run_ir_product(grid, raw_blocks(plan.reconstruction), state)
        state *= np.exp(1j * plan.reconstruction.physical_phase)
    elif flavor == "split_step":
        from .split_step import evolve

        # A split-step Plan stored this flavor's state budget at planning, which caps the observed one;
        # a reference evolution of a Plan of another flavor recomputes it (split_step.evolve).
        own = plan.method.theory_flavor == "split_step"
        state, delta = evolve(grid, plan.method.kinetic_model, plan.reconstruction, dt, state,
                              start=_host_start_error(plan),
                              ceiling=plan.reconstruction.host_state_error if own else None)
        if observed is not None:
            observed.append(delta)
    else:
        from .split_step import potential_diagonal

        kinetic = restricted_kinetic_sparse(grid)
        # Diagonal slots and the private generator data are formed before the potential exists.
        generator, slots, diagonal = _generator_workspace(kinetic)
        # The broadcast table sum, without the objective constant, converted once to complex128.
        real = potential_diagonal(plan.reconstruction, grid)
        potential = real.astype(np.complex128).reshape(-1)
        del real
        # expm_multiply estimates norms with onenormest, which draws from
        # NumPy's global generator (_linalg_laws.seeded_norm_estimates).
        with seeded_norm_estimates():
            for _time, kinetic_weight, potential_weight in plan.reconstruction.step_weights:
                _fill_generator(generator, kinetic, slots, diagonal, potential, kinetic_weight,
                                potential_weight, dt)
                state = scipy.sparse.linalg.expm_multiply(generator, state)
    return state


def _generator_workspace(kinetic):
    """Return a CSR generator sharing the kinetic pattern, its diagonal slots and a diagonal buffer.

    ``kinetic`` is the canonical CSR of ``theory.restricted_kinetic_sparse``,
    with complex128 data and one structural diagonal per row. The generator
    shares its indices and index pointers and owns a separate ``16 z``-byte
    data array for z stored entries, which ``_fill_generator`` overwrites at
    every step. The row-index array and Boolean test of the slot search are
    released on return, before the first exponential action.

    Raises:
        ValueError: The pattern lacks a diagonal slot in some row.
    """
    import numpy as np
    import scipy.sparse

    dimension = kinetic.shape[0]
    rows = np.repeat(np.arange(dimension, dtype=kinetic.indices.dtype), np.diff(kinetic.indptr))
    slots = np.flatnonzero(kinetic.indices == rows)
    del rows
    if slots.size != dimension:
        raise ValueError("kinetic pattern needs one diagonal slot per row")
    data = np.empty_like(kinetic.data)
    diagonal = np.empty(dimension, dtype=np.complex128)
    # Assign the arrays directly: the (data, indices, indptr) constructor copies an index array whose
    # dtype it narrows, which SciPy selects from the raw COO slot count rather than the final pattern.
    generator = scipy.sparse.csr_matrix(kinetic.shape, dtype=np.complex128)
    generator.data, generator.indices, generator.indptr = data, kinetic.indices, kinetic.indptr
    return generator, slots, diagonal


def _fill_generator(generator, kinetic, slots, diagonal, potential, a, b, dt):
    """Overwrite the generator data with ``-i dt (a K + b diag(V))`` in the fixed operation order.

    Every entry follows ``ka = RN(a*kin)``, ``pv = RN(b*V)``,
    ``h = RN(ka + pv)`` and ``g = RN((-1j*dt)*h)``, the order of the former
    sparse expression ``-1j * dt * (a * K + b * diag(V))``; dt is not
    distributed into the coefficients, whose rounding errors per entry,
    including subnormal products, ``_schrodinger_step_bounds`` prices.
    ``potential`` is the complex128 potential diagonal, converted once, so
    no real-to-complex boundary arithmetic enters. Nonzero entries are the same binary64 numbers as the
    sparse addition; a cancelled entry stays a stored zero, which leaves the
    matrix and its nonzero arithmetic unchanged, while its storage and
    signed-zero bits need not match.
    """
    import numpy as np

    np.multiply(kinetic.data, a, out=generator.data)
    np.multiply(potential, b, out=diagonal)
    np.add(generator.data[slots], diagonal, out=diagonal)
    generator.data[slots] = diagonal
    np.multiply(generator.data, -1j * dt, out=generator.data)
    return generator


def _execute_theory(plan, kernel):
    """Run the classical kernel and return its labeled scalars and optional state.

    The restricted state gives the probability of each grid point, which
    ``_summarize`` turns into the candidate, the most probable point, the
    marginals and the masses. The tie window of the most probable point is
    ``_host_window`` of the state budget delta, the budget that the
    ``split_step`` evolution observed on its trajectory
    (``split_step.evolve``), or for the other flavors the Plan's budget
    (``_host_state_error``) that planning stored as
    ``QHDReconstruction.host_state_error``, and the kernel publishes delta as
    ``readout_state_error``. The probability maximizer is published with
    its indices, probability and objective, and the mode status as the
    Boolean scalar ``mode_unresolved``, since the kernel has no state from
    which a later analysis could evaluate it. The scalars follow the order of
    ``_scalar_names``. The invalid mass is zero because the restricted basis
    holds only grid points. The ``schrodinger`` and ``split_step``
    probabilities are read before the objective constant's global phase is
    restored (``_constant_phase``), so they do not depend on the constant,
    and the kept state then carries the physical phase. The observed mass
    is the ``fsum`` valid mass of the probability array.
    """
    import numpy as np

    flavor = plan.method.theory_flavor
    observed = []
    state = _evolve_restricted(plan, flavor, observed)
    delta = observed[0] if observed else plan.reconstruction.host_state_error
    # One float64 buffer: |z| then its square in place, the same numbers as np.abs(state)**2.
    probabilities = np.abs(state)
    np.square(probabilities, out=probabilities)
    if plan.method.keep_state and flavor != "ir_product":
        # exp(i phi) of the objective constant, once, in place on the kept state only
        state *= np.exp(1j * _constant_phase(plan.reconstruction, plan.method))
    # The observed mass is the fsum valid mass, the invalid mass being zero on the restricted basis.
    result = _summarize(plan, probabilities, 0.0, None, _host_window(delta))
    data = {
        "candidate_objective": result["value"],
        "candidate_probability": result["candidate_probability"],
        "most_probable_objective": result["most_probable_objective"],
        "most_probable_probability": result["most_probable_probability"],
        "most_probable_deficit": result["most_probable_deficit"],
        "probability_maximizer_objective": result["probability_maximizer_objective"],
        "probability_maximizer_probability": result["probability_maximizer_probability"],
        "mode_unresolved": float(result["mode_status"] == "unresolved"),
        "readout_state_error": delta,
        "expected_objective": result["expected_objective"],
        "valid_mass": result["valid_mass"],
        "invalid_mass": 0.0,
        "observed_mass": result["observed_mass"],
        **{f"candidate_index_{j}": float(i) for j, i in enumerate(result["candidate_indices"])},
        **{f"most_probable_index_{j}": float(i) for j, i in enumerate(result["most_probable_indices"])},
        **{f"probability_maximizer_index_{j}": float(i)
           for j, i in enumerate(result["probability_maximizer_indices"])},
        **{
            f"marginal_{j}_{i}": v
            for j, row in enumerate(result["marginals"].tolist())
            for i, v in enumerate(row)
        },
    }
    return KernelOutput(
        plan_id=plan.content_id,
        selected_kernel_id=kernel.content_id,
        scalars=tuple(
            ScalarValue(label=n, value=data[n], frame="physical") for n in kernel.scalars
        ),
        physical_scale=None,
        physical_scale_unavailable="optimization population has no vector scale",
        arrays=(("state", state),) if plan.method.keep_state else (),
        applications=(_application(plan, kernel.implementation),),
    )


def _block_records(raw_steps):
    """Convert compiled evolution blocks into the ``QHDBlock`` records a Plan stores.

    A kinetic block keeps its two linked qubits, the common XX and YY
    coefficient and the two CX of one ``XXPlusYYGate`` (Qiskit 2.5.2's
    definition). A projector block
    keeps its qubits, its weighted objective value and the CX count of the
    provider that ``resolve_number_projector_lowering`` chose for its support
    size. Both keep the duration, the step midpoint time and the angle.
    """
    steps = []
    for group in raw_steps:
        items = []
        for block in group:
            if block.kind == "kinetic":
                from nwqlib.subroutines.hamiltonian_evolution.pauli_labels import (
                    parse_pauli_label,
                )

                support = tuple(parse_pauli_label(block.terms[0].pauli))
                coefficient = coerce_real_scalar(
                    block.terms[0].coefficient, context="kinetic coefficient"
                )
                provider, cx = "XXPlusYYGate", 2
            else:
                support, coefficient = block.support, block.metadata["coefficient"]
                resolved = block.metadata["lowering_resolution"]
                provider = resolved.get("structured_provider", {}).get("provider", "structured")
                cx = resolved["costs"][resolved["selected"]]
            items.append(
                QHDBlock(
                    kind=block.kind,
                    time_step=block.time_step,
                    time=block.time,
                    support=support,
                    coefficient=coefficient,
                    angle=coerce_real_scalar(block.angle, context="selected rotation"),
                    provider=provider,
                    cx=cx,
                )
            )
        steps.append(tuple(items))
    return tuple(steps)


def _error_model(problem, output, construction, *, compact, split, kinetic_model, execution, shots, rule,
                 aqft=False):
    """Declare every error source of the selected finite model, each with unknown size.

    The finite grid, the time discretization of the schedule, floating-point
    arithmetic and the gap between the best observed candidate and the
    optimum apply to every route. The time discretization is ``midpoint_time``
    under the ``"midpoint"`` coefficient rule. The ``"integrated"`` rule's
    exact coefficient integrals remove that coefficient quadrature error, but
    each step still freezes the time order, dropping the Magnus terms from
    ``Omega_2`` on (``schedules.step_weights``), an error that is nonzero
    when ``[K, V] != 0`` and ``a/b`` varies in time, so it declares
    ``time_ordering`` instead. The compiled product adds splitting error and
    any rotations removed by the angle threshold, and a truncated binary QFT
    (``aqft``) adds ``aqft_truncation``. The classical
    ``split_step`` flavor (``split``) adds the Strang splitting error of its
    potential and kinetic factors, also named ``product_formula``. The
    ``"spectral"`` kinetic model replaces the finite-difference stencil by
    the Fourier symbol of ``-Delta/2``, a model error named
    ``kinetic_model``. Native execution adds simulator or device error, and
    counts add sampling error. NWQLib propagates none of these to the
    objective, so every term is unknown. The dropped-angle total stored in
    the reconstruction is not an objective-gap bound either.
    """
    frame = output.frame(problem)
    time = "midpoint_time" if rule == "midpoint" else "time_ordering"
    errors = ("discretization", time, "floating_point", "optimization_gap")
    if kinetic_model != "finite_difference":
        errors += ("kinetic_model",)
    if compact:
        errors += ("product_formula", "rotation_pruning") + (("aqft_truncation",) if aqft else ())
    elif split:
        errors += ("product_formula",)
    if execution == "quantum":
        errors += ("native_execution",) + (("sampling",) if shots is not None else ())
    return ErrorModel(
        output_id=output.content_id,
        subject_id=problem.content_id,
        construction_id=construction.content_id,
        frame=frame,
        required_sources=errors,
        source=METHOD,
        terms=tuple(
            ErrorTerm(
                name=name,
                stage="analysis",
                source=METHOD,
                formula="no propagated objective-gap bound",
                coverage=name,
                fact=FramedFact(
                    frame=frame,
                    bindings=(),
                    fact=Fact(
                        quantity=name,
                        unit=frame.unit,
                        scope=frame.scope,
                        availability="unknown",
                        reason="observed candidate does not establish " + name,
                    ),
                ),
            )
            for name in errors
        ),
    )


class QHD(Method):
    """Quantum Hamiltonian Descent (QHD) method for the minimum of an objective over a box.

    Build it with keyword arguments and pass it as `method=` with an `Optimization`, for
    example `solve(problem, method=QHD(num_grid_points=4))`. Every argument is optional.
    The result is a [`QHDAnalysis`][nwqlib.algorithms.qhd.records.QHDAnalysis]. Its
    `candidate` is the best observed grid point and `objective` the objective there, in
    the Problem's coordinates and unit, and `most_probable_coordinates` is the grid
    point where the evolution concentrated probability. QHD is not a global optimizer.
    Both points belong to a finite grid of the box, and no setting implies a continuous
    global-optimization guarantee.

    The Method evolves `H(t) = a(t) K + b(t) V`, the Hamiltonian of Leng et al.,
    arXiv:2303.01471v1, Eq. (1), with `a = exp(phi_t)` and `b = exp(chi_t)`, K the
    kinetic operator `-Delta/2` on the grid and V the objective values, one step per
    time step, and reads out grid points. Each step is a product formula on the circuit
    routes and in the `ir_product` and `split_step` flavors, and a numerical evaluation
    of the exponential of the step's Hamiltonian in the default `schrodinger` flavor.
    The default quadratic schedule is the example of QHDOPT (Kushnir et al.,
    arXiv:2409.03121v1, Sec. 2.1), and the one-hot encoding and the shifted-cubic
    schedule come from Wu et al., arXiv:2605.12066v1. The rows and notes below say where
    a setting departs from Leng et al.'s Algorithm 1. Its step 6 measures one grid
    point, while QHD reports the best observed and the most probable point of the
    observed population. The [QHD guide](../../algorithms/qhd.md) explains the grids,
    schedules, initial states and readout, and its
    [source map](../../algorithms/qhd.md#source-map) gives the paper location of each
    step or marks it as standard or NWQLib's own.

    `max_work` and `max_bytes` limit the work and bytes that planning counts. They do
    not control undocumented SymPy, SciPy or Qiskit costs.

    Attributes:
        num_grid_points: Default `2`. Grid points K per variable, at least 2. The
            one-hot periodic grid needs an even K of at least 4, and the binary encoding
            needs `K = 2**b`. K is also the one-hot register width per variable, and the
            binary encoding uses b qubits per variable.
        encoding: Default `"one_hot"`, the established circuit. How each variable's grid
            index is stored in qubits: `"one_hot"`, K qubits with one excitation, or
            `"binary"`, b qubits for `K = 2**b`, which requires `boundary="periodic"`.
            The Encodings note below describes both registers.
        binary_synthesis: Default `BinarySynthesis()`. Circuit choices of the binary
            encoding: the potential and kinetic phase diagonals, the QFT bit reversal
            and the AQFT cutoff
            ([`BinarySynthesis`][nwqlib.algorithms.qhd.binary.BinarySynthesis]). The
            one-hot encoding requires the default record.
        num_steps: Default `1`. Positive number of evolution steps.
        total_time: Default `1.0`. Positive total evolution time. Each step lasts
            `total_time/num_steps`.
        schedule: Default `QuadraticSchedule()`, with gamma 0.3. Kinetic weight a(t) and
            potential weight b(t) of `H(t) = a(t) K + b(t) V`, one of
            [`QuadraticSchedule`][nwqlib.algorithms.qhd.schedules.QuadraticSchedule],
            [`CubicSchedule`][nwqlib.algorithms.qhd.schedules.CubicSchedule] or
            [`ShiftedCubicSchedule`][nwqlib.algorithms.qhd.schedules.ShiftedCubicSchedule].
            Its `kind` names the formula. A schedule parameter stays fixed when
            `num_steps` changes, and no schedule carries a convergence guarantee.
        coefficient_rule: Default `"midpoint"`. The weights each step uses:
            `"midpoint"`, the point values `a(t)` and `b(t)` at the step midpoint, or
            `"integrated"`, the step averages of their interval integrals. The Product
            formula note below explains both.
        trotter_order: Default `2`. Product-formula order, 1 or 2, shared by circuit
            selection and the numerical `ir_product` reference. The `split_step` flavor
            is always symmetric. The Product formula note below gives the factor order.
        boundary: Default `"dirichlet"`, which accepts the default K = 2 with the
            one-hot encoding. Boundary condition of the finite-difference kinetic term,
            `"dirichlet"` or `"periodic"`. The periodic grid takes no boundary points,
            and with the one-hot encoding it needs an even K of at least 4. The Grids
            note below gives the grid points.
        include_boundary_points: Default `False`. Place the Dirichlet grid on both box
            endpoints instead of the interior. Not available with the periodic boundary.
        kinetic_model: Default `"finite_difference"`, which every route applies. Kinetic
            operator K of each variable: `"finite_difference"`, the three-point stencil
            of `-Delta/2`, or `"spectral"`, the Fourier operator of `-Delta/2`. The
            spectral model needs the periodic boundary and a route that applies Fourier
            energies. The Kinetic models note below gives their eigenvalues and routes.
        rotation_threshold: Default `0.0`, which keeps every computed nonzero angle.
            Angle in radians, nonnegative, below which compiled rotations are omitted.
            The initial-state preparation is never pruned. The Rotation threshold note
            below says which rotations each encoding omits and how the Plan records
            them.
        initial_state: Default `KineticGroundState()`. State that every route starts
            from, a product of nonnegative per-variable amplitude vectors:
            [`KineticGroundState()`][nwqlib.algorithms.qhd.initial_state.KineticGroundState],
            [`UniformState()`][nwqlib.algorithms.qhd.initial_state.UniformState] or
            [`GaussianState(center=..., widths=...)`][nwqlib.algorithms.qhd.initial_state.GaussianState].
            Leng et al.'s Algorithm 1, step 3, names the uniform and a Gaussian state.
            The row "QHD Method defaults" of the
            [engineering constants](../../ENGINEERING_CONSTANTS.md#safety-factors-and-workflow-defaults)
            gives the evidence for the default and its limits.
        initial_state_preparation: Default `"structured"`, the library's circuit for the
            encoding and state, whose CX count the Plan records. How quantum execution
            prepares the initial state. `"qiskit_state_preparation"` appends a Qiskit
            `StatePreparation`, and `"none"` selects a resource-only construction, which
            classical execution rejects. The State preparation note below describes
            each.
        theory_flavor: Default `"schrodinger"`. Classical evolution of the `K**d` grid
            amplitudes under `execution="classical"`: `"schrodinger"`, the exponential
            of each step's Hamiltonian, `"ir_product"`, a numerical reference of the
            compiled product, or `"split_step"`, a symmetric split-step product. The
            Classical flavors note below compares them.
        keep_state: Default `False`. Keep the final state array when the output and
            execution support it. It requires exact readout, without `shots`.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Positive limit on
            the known bytes of state, grid and numerical workspace that planning counts.
            It does not bound the process memory.
        max_work: Default `1_000_000_000`. Positive limit on the counted work of
            objective tables, products and classical evolution, in planning work units,
            the scalar and array operations that planning declares. A work unit is not a
            measured CPU operation or a time. A refusal names the stage and the amount
            to set ([cost](../../algorithms/qhd.md#cost-and-existing-evidence)).

    Encodings:
        `"one_hot"` places variable j on the K qubits `j K` to `j K + K - 1` with one
        excitation, the one-hot encoding of Wu et al., arXiv:2605.12066v1, Sec. IV.A,
        with the occupation operators of their Eqs. (9)-(10). The one-hot encoding is
        the default as the established circuit.

        `"binary"` places it on the b qubits `j b` to `j b + b - 1`, grid index
        `n_j = sum_l 2**l n_(j,l)`, so every outcome is a grid point. This is the
        register of Leng et al.'s Algorithm 1, and it applies the kinetic term as
        `F^dagger exp(-i alpha diag(E)) F` with a QFT on each register, the form of
        their kinetic step, Eq. (E.4). It requires `boundary="periodic"`, whose kinetic
        operator the QFT diagonalizes, and `K = 2**b`, the register's state count.

    Grids:
        The Dirichlet interior grid, the default, has the vanishing boundary values of
        Leng et al.'s Eq. (F.4), and the endpoint grid of `include_boundary_points=True`
        keeps the endpoint amplitudes, as their Eq. (F.9) does. The periodic grid, the
        mesh of their Eq. (E.1), is `x_i = lower + i (upper - lower)/K`, i = 0..K-1,
        with `upper` identified with `lower`, and adds the wrap link between the points
        K-1 and 0. It takes no boundary points, and with the one-hot encoding it needs
        an even `num_grid_points` of at least 4. Dirichlet is the default because it
        accepts the default K = 2 with the one-hot encoding.

    Product formula:
        `coefficient_rule="midpoint"` uses the point values `a(t)` and `b(t)` at each
        step midpoint. `"integrated"` uses the step averages of the interval integrals,
        so each step's kinetic and potential exponents equal `A` and `B` over the step,
        the first Magnus term. Leng et al.'s Eq. (E.3) and Algorithm 1 take the left
        endpoint, which leaves the product first order in the step even with symmetric
        placement, so no left-endpoint rule is offered
        ([Split-step classical evaluation](../../algorithms/qhd.md#split-step-classical-flavor)).

        `trotter_order=1` uses the potential-then-kinetic ordering of Leng et al.'s
        Algorithm 1. Order 2 uses the symmetric composition of Strang, SIAM J. Numer.
        Anal. 5 (1968) 506-517, doi:10.1137/0705041, with a potential half on either
        side of the kinetic factor. The one-hot kinetic factor is itself approximated by
        a product over links, in increasing link order for order 1 and by an odd-half,
        even-full, odd-half split for order 2. Binary kinetic factors use Fourier
        conjugation of their phase diagonal.

    Kinetic models:
        `"finite_difference"` is the three-point stencil of `-Delta/2` from the central
        difference of Leng et al.'s Eqs. (F.7) and (F.9), with diagonal `1/h**2` and
        `-1/(2 h**2)` on each link. Its eigenvalues are
        `(2/h**2) sin(pi r/(2 (K + 1)))**2`, r = 1..K, on both Dirichlet grids (each
        with its own h), and `(2/h**2) sin(pi k/K)**2`, k = 0..K-1, on the periodic
        grid.

        `"spectral"` is the Fourier operator of `-Delta/2` that Leng et al.'s Eq. (E.4)
        and Algorithm 1 apply, on the periodic grid of period `L = K h`. Its eigenvalue
        is `2 pi**2 q**2/L**2` for the Fourier mode k whose signed index q is k below
        `ceil(K/2)` and `k - K` from there on. The two models agree at low momentum,
        `0 <= E_sp - E_FD <= 2 pi**4 q**4 h**2/(3 L**4)`, and differ by the factor
        `pi**2/4` at the Nyquist mode.

        The spectral model requires the periodic boundary and a route that applies the
        Fourier energies: the classical `split_step` flavor, or the binary encoding's
        quantum circuit and `ir_product` flavor. The one-hot circuit, the one-hot
        `ir_product` flavor and the `schrodinger` flavor apply the finite-difference
        stencil, and planning refuses the spectral model there. The finite-difference
        model is the default because every route applies it.

    Rotation threshold:
        On the one-hot encoding every compiled hopping or number-projector block whose
        nonzero angle magnitude is below `rotation_threshold` is omitted. On the binary
        encoding it removes only Walsh-string `Rz` rotations. The QFT controlled phases
        and the angles of a dense phase diagonal stay, and
        `binary_synthesis.aqft_cutoff` truncates the QFT instead. The initial-state
        preparation is never pruned. Each omission is an approximation, which the Plan
        records with the dropped count and an operator-norm bound on its effect,
        `pruning_error_bound`.

    State preparation:
        `initial_state_preparation="structured"` emits the library's structured circuit
        for the encoding and initial state, and the Plan records its CX count as a
        `ResourceLaw`. On the one-hot encoding it is the nonnegative amplitude chain of
        O(dK) gates, which for the uniform state is the linear W-state chain. The chain
        uses computed rotation parameters and can stop at its lower-range cutoff. The
        error bound of the omitted links is recorded per register in
        `range_omissions.chains` and included in the `state_preparation` error source.
        On the binary encoding it is one H per qubit, with no CX, which prepares the
        uniform state and the periodic kinetic ground state, the same state. Other
        states need `"qiskit_state_preparation"` there.

        `"qiskit_state_preparation"` appends one Qiskit `StatePreparation` per register,
        of `2**K` one-hot or K binary amplitudes. `"none"` selects a quantum
        resource-only construction without preparation. Classical execution prepares no
        circuit and rejects `"none"`.

    Classical flavors:
        `"schrodinger"` numerically evaluates each step's Hamiltonian exponential with
        SciPy's `expm_multiply`, whose work grows with the step's kinetic and potential
        weights and the objective's range. `"ir_product"` evaluates a numerical
        reference of the compiled product. It applies each one-hot block's analytic
        restricted action at its stored angle, and for the binary encoding the QFT
        operations with computed diagonal phase arrays. `"split_step"` applies the
        symmetric (Strang) product of half potential phases and the kinetic factor,
        diagonal in the kinetic eigenbasis and evaluated numerically by orthonormal
        DST-I on the Dirichlet grids and the FFT on the periodic grid. Its work per step
        depends on neither the schedule weights nor the objective's range, and it adds a
        splitting error that `"schrodinger"` does not have.

    Raises:
        ValueError: If `boundary="periodic"` is combined with
            `include_boundary_points=True`, or with the one-hot encoding and an odd
            `num_grid_points` or one below 4. If `kinetic_model="spectral"` is used with
            a Dirichlet boundary. If `encoding="binary"` is used without the periodic
            boundary or with a `num_grid_points` that is not a power of two. If
            `binary_synthesis` differs from its default with the one-hot encoding.

    Examples:
        The minimum of `(x - 0.2)**2 + (y + 0.2)**2` on `[-1, 1]**2` is 0 at (0.2,
        -0.2), which is a point of the 4-point Dirichlet interior grid (-0.6, -0.2, 0.2,
        0.6) of each axis. With exact readout on the default Aer simulator the candidate
        is the point of least evaluated objective among the grid points with positive
        probability, here that grid minimum. The evolution also puts the most
        probability there.

        >>> import sympy as sp
        >>> from nwqlib import solve
        >>> from nwqlib.problems import Optimization
        >>> from nwqlib.algorithms.qhd import QHD
        >>> x, y = sp.symbols("x y", real=True)
        >>> problem = Optimization(objective=(x - 0.2)**2 + (y + 0.2)**2,
        ...                        variables=(x, y),
        ...                        bounds=((-1.0, 1.0), (-1.0, 1.0)))
        >>> method = QHD(num_grid_points=4, num_steps=8, total_time=4.0)
        >>> result = solve(problem, method=method, seed=7)
        >>> print([round(v, 6) for v in result.candidate],
        ...       round(result.objective, 12))
        [0.2, -0.2] 0.0
        >>> print([round(v, 6) for v in result.most_probable_coordinates],
        ...       result.mode_status)
        [0.2, -0.2] resolved
    """

    result_type: ClassVar[type] = QHDAnalysis

    schema_version: Literal[3] = 3
    # num_grid_points=2, num_steps=1 and total_time=1 are the smallest legal
    # grid and step count with unit time, so a default solve uses two qubits
    # per variable. The quadratic schedule with gamma=0.3 is the illustrative
    # schedule of the QHD guide. The midpoint rule needs no interval
    # integrals, and a convergence check of a fixed model found it and the
    # integrated rule both second order, with errors within 1% of each other
    # (2026-09-26, NWQLib commit 06e1f2ab7b33c6eb54cb60c1a3373dc9aade668f,
    # Python 3.12.14, NumPy 2.5.2 and SciPy 1.18.1 on macOS arm64). The check
    # evolved the one-dimensional two-mode cosine of Liu et al.,
    # arXiv:2607.16996v1, on an eight-point periodic grid from the kinetic
    # ground state to T = 1, under the quadratic schedule and both cubic
    # schedules at s = 1, with the split-step flavor at 64, 128 and 256 steps,
    # and compared it with an adaptive DOP853 solution.
    # Revisit the coefficient rule after a check at small s, where the two
    # rules differ most. The Dirichlet boundary is the one that admits the
    # default K = 2 with the default one-hot encoding (_periodic_grid). The
    # finite-difference kinetic model is the one that every route applies
    # (_kinetic_route). None of these defaults is an accuracy choice.
    # Registered in docs/ENGINEERING_CONSTANTS.md. Revisit when a default
    # workload is chosen for a scientific purpose.
    num_grid_points: Annotated[int, Field(strict=True, ge=2)] = 2
    # The one-hot encoding is the established circuit. The binary synthesis
    # defaults are exact: min_cx takes the cheaper of two exact diagonal
    # syntheses per table, relabel removes the swap layers without changing
    # the operator, and the QFT keeps every controlled phase. Registered in
    # docs/ENGINEERING_CONSTANTS.md ("QHD Method defaults").
    encoding: Literal["one_hot", "binary"] = "one_hot"
    binary_synthesis: BinarySynthesis = BinarySynthesis()
    num_steps: PositiveInt = 1
    total_time: Annotated[Real, Field(gt=0)] = 1.0
    schedule: Schedule = QuadraticSchedule()
    coefficient_rule: Literal["midpoint", "integrated"] = "midpoint"
    trotter_order: Literal[1, 2] = 2
    boundary: Literal["dirichlet", "periodic"] = "dirichlet"
    include_boundary_points: StrictBool = False
    kinetic_model: Literal["finite_difference", "spectral"] = "finite_difference"
    rotation_threshold: Nonnegative = DEFAULT_ROTATION_THRESHOLD
    # The kinetic ground state is the ground state of the kinetic term,
    # whose weight relative to the potential, a(t)/b(t), is largest at t = 0
    # and decays afterwards. On a Dirichlet grid it has no excited kinetic
    # component, whose phase would depend on the step count
    # (docs/algorithms/qhd.md, "Initial states and preparation"). Under the
    # quadratic schedule both terms are present at t = 0, so it is not an
    # eigenstate of H(0) and gives no adiabatic guarantee. On the periodic
    # grid it equals the uniform state (initial_state.KineticGroundState).
    # On the Dirichlet grids its structured one-hot preparation has the same
    # 3(K - 1) CX per variable as the uniform state's, since both have full
    # positive support (initial_state.chain_links). In a comparison with
    # UniformState() on two-variable Dirichlet interior grids, with the
    # classical Schrodinger model, the quadratic schedule with gamma 0.3,
    # midpoint weights, T = 10, 200 steps and exact readout, it gave the
    # smallest best-point distance in all six standalone refinement runs with
    # the most probable point (double well, anisotropic quadratic and Ackley,
    # each with gains 1 and 8, at most ten levels), and a much better refined
    # constrained Rastrigin point (distance 0.0013 against 0.034 under
    # most_probable, three levels per augmented-Lagrangian round). It is not
    # better everywhere. The refined most_probable run on a mixed-scale
    # problem took 5 rounds instead of 3, and the unrefined mode_or_mean run
    # on the unit disk ended farther from the optimum (distance 0.023 against
    # 0.00079). The split-step flavor with 3,200 steps gave the same refined
    # constrained Rastrigin distances, so the Schrodinger and split-step
    # flavors share the default. These runs were made on 2026-09-26 with
    # Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1 and SymPy 1.14.0 on macOS
    # arm64, at NWQLib commits 461d7dce706eed4724a4e850915746f8683831b2 (the
    # standalone runs), 659fa80702e79192a27568ea4935531004d14e51 (the
    # augmented-Lagrangian runs) and 960d9d1dd49c805c05fe6d4a821aea99dee4a914
    # (the split-step runs). Revisit if a comparison at resolved step counts
    # on another problem class favors a different start. The choice
    # carries no global-optimization guarantee. UniformState(), the start of
    # Leng et al., arXiv:2303.01471v1, Algorithm 1 step 3, stays an explicit
    # option. Registered in docs/ENGINEERING_CONSTANTS.md ("QHD Method
    # defaults").
    initial_state: InitialState = KineticGroundState()
    initial_state_preparation: Literal["structured", "qiskit_state_preparation", "none"] = "structured"
    theory_flavor: Literal["schrodinger", "ir_product", "split_step"] = "schrodinger"
    keep_state: StrictBool = False
    # Default limits on known table and array work and bytes, not process RSS
    # or CPU. Registered in docs/ENGINEERING_CONSTANTS.md ("QHD selected
    # operation sizes"). Revisit for a concretely selected larger workload.
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_work: PositiveInt = 1_000_000_000

    @field_validator("num_grid_points", "num_steps", "trotter_order", mode="before")
    @classmethod
    def _integer_controls(cls, value, info):
        return integer(value, info.field_name, 1)

    @model_validator(mode="after")
    def _periodic_grid(self):
        """Reject a periodic grid with boundary points, and a one-hot periodic grid with K odd or below 4.

        The periodic grid identifies ``upper`` with ``lower``, so it has no
        boundary points to include. The second-order product splits each
        variable's links into two layers of disjoint links
        (``kinetic.KineticCompiler``). On the cycle of K links the wrap link
        ``(0, K-1)`` has index K-1, which for even K falls in the odd layer
        without sharing a point with another odd link. For odd K it shares
        point 0 with link ``(0, 1)`` in the even layer, and an exact split
        would need a third layer, which this compiler does not build. That is
        a limit of the current link compiler, not of the periodic Laplacian,
        and it ends when the compiler builds a third layer for odd cycles.
        K = 2 is a degenerate cycle, in which both neighbors of a point are the
        same point and the wrap link joins the same two points as link
        ``(0, 1)``. First order and the classical ``schrodinger`` and
        ``split_step`` flavors need no disjoint layers, but the rule holds
        for every Method, so that the periodic grids a Method admits do not
        depend on ``trotter_order`` or on the execution, and one Method can be
        planned for native execution and for its classical reference alike.

        The binary encoding uses no links. Its kinetic factor applies the
        Fourier energies ``2 sin**2(pi k/K)/h**2`` by a QFT conjugation, and
        ``_encoding_fields`` requires ``K = 2**b``. At K = 2 both neighbors of
        a point are the other point, so the periodic stencil
        ``(2I - P - P^dagger)/(2 h**2)`` with ``P = X`` is ``(I - X)/h**2``,
        with energies 0 and ``2/h**2``, which one H conjugation applies
        exactly. The classical ``schrodinger`` flavor adds the two coinciding
        links of ``OneHotGrid.links``, ``-1/(2 h**2)`` each, to the same
        off-diagonal ``-1/h**2``. The binary encoding therefore admits K = 2.
        """
        if self.boundary == "periodic":
            if self.include_boundary_points:
                raise ValueError(
                    "boundary='periodic' has no boundary points to include. Its grid "
                    "lower + i*(upper - lower)/K, i = 0..K-1, identifies upper with lower, so "
                    "include_boundary_points=True does not apply"
                )
            if self.encoding == "one_hot" and (self.num_grid_points < 4 or self.num_grid_points % 2):
                raise ValueError(
                    "boundary='periodic' with the one-hot encoding requires an even num_grid_points of "
                    "at least 4, got "
                    f"{self.num_grid_points}. The one-hot link compiler splits each variable's links "
                    "into two layers of disjoint links, which holds for the wrap link between points "
                    "K-1 and 0 only when K is even. This is a limit of the compiler, not of the "
                    "periodic Laplacian. K = 2 is a degenerate cycle whose wrap link repeats the link "
                    "between points 0 and 1"
                )
        return self

    @model_validator(mode="after")
    def _spectral_kinetic_grid(self):
        """Reject the spectral kinetic model on the Dirichlet grids.

        The spectral eigenvalue ``2 pi**2 q**2/L**2`` belongs to the Fourier
        mode ``exp(2 pi i q x/L)``, which is L-periodic and does not vanish at
        the box ends, so the operator is ``-Delta/2`` with the periodic
        boundary. The Dirichlet counterpart would be a sine-series operator
        with other eigenvalues, which this model does not define.
        """
        if self.kinetic_model == "spectral" and self.boundary != "periodic":
            raise ValueError(
                "kinetic_model='spectral' requires boundary='periodic'. Its Fourier modes "
                "exp(2 pi i q x/L) are periodic on the box and do not satisfy the Dirichlet "
                "boundary condition"
            )
        return self

    @model_validator(mode="after")
    def _encoding_fields(self):
        """Reject field combinations that no encoding implements.

        - The binary encoding needs ``K = 2**b``, because its b-qubit register
          has ``2**b`` states and every one of them must be a grid point, and
          the periodic grid, whose finite-difference and spectral kinetic
          operators are diagonal in the discrete Fourier basis that the QFT
          applies. The Dirichlet binary kinetic needs a sine transform circuit
          and is not implemented (``docs/ROADMAP.md`` lists it). Any b >= 1
          is admitted, K = 2 included
          (``_periodic_grid``).
        - ``binary_synthesis`` has no effect on a one-hot circuit, so only its
          default record is accepted there, which keeps equal one-hot Methods
          equal.
        """
        if self.encoding == "binary":
            k = self.num_grid_points
            if self.boundary != "periodic" or k & (k - 1):
                raise ValueError(
                    f"encoding='binary' requires boundary='periodic' and num_grid_points = 2**b, got "
                    f"boundary={self.boundary!r} and {k} points. The QFT diagonalizes the periodic kinetic "
                    "operator on a register of 2**b grid points, and a Dirichlet binary kinetic is not "
                    "implemented")
        elif self.binary_synthesis != BinarySynthesis():
            raise ValueError("binary_synthesis applies to encoding='binary' only")
        return self

    def _kinetic_route(self, execution):
        """Reject a kinetic model that the selected route does not apply.

        A route that fell back to another kinetic operator would evolve a
        different finite model from the one the Method names, so the
        refusal comes at planning. The one-hot circuit and the one-hot
        ``ir_product`` flavor apply the finite-difference links
        (``kinetic.KineticCompiler``), and the ``schrodinger`` flavor the
        finite-difference stencil
        (``theory.restricted_kinetic_sparse``). The classical ``split_step``
        flavor applies the kinetic operator in its eigenbasis
        (``split_step.kinetic_eigenvalues``), and the binary circuit and the
        binary ``ir_product`` flavor apply its Fourier energies
        (``binary.kinetic_table``), so these routes admit the spectral
        model.
        """
        if self.kinetic_model == "finite_difference":
            return
        route = "native" if execution == "quantum" else self.theory_flavor
        if route == "split_step" or (self.encoding == "binary" and route in ("native", "ir_product")):
            return
        raise ValueError(
            f"kinetic_model={self.kinetic_model!r} is not applied by the {self.encoding} {route} route, "
            "which applies the finite-difference stencil. Use classical execution with "
            "theory_flavor='split_step', or encoding='binary' with native execution or "
            "theory_flavor='ir_product'"
        )

    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR

    def _admit(self, work, data_bytes, *, stage="QHD selection", lower_bound=False, later=None):
        """Refuse the named selected operation before it exceeds either declared limit.

        ``later`` maps an exceeded field to the later phases whose populations
        are not fixed by the dimensions and can need more; the first admission
        of a Plan passes it with the combined known needs of its first phases.
        """
        short, beyond = [], []
        if work > self.max_work:
            short.append(f"set QHD(max_work={work}) or larger for this work charge")
            beyond.append((later or {}).get("max_work"))
        if data_bytes > self.max_bytes:
            short.append(f"set QHD(max_bytes={data_bytes}) or larger for this byte charge")
            beyond.append((later or {}).get("max_bytes"))
        if not short:
            return
        remedy = ". ".join(short)
        if any(beyond):
            remedy += ". Later phases can need more: " + "; ".join(text for text in beyond if text)
        if lower_bound:
            remedy += (
                ". Counting may have stopped early, so the displayed amounts are lower bounds "
                "on this stage's complete admission charges. Setting an exceeded limit to its "
                "displayed amount can refuse again at this stage. If resources permit, raise "
                "each exceeded limit by a larger factor, for example to twice its displayed amount"
            )
        if stage == "classical host evolution and readout" and self.theory_flavor == "schrodinger":
            remedy += (
                ". Try theory_flavor='split_step': at fixed grid, table shapes, initial-state recipe "
                "and step count, its kernel charge does not depend on the schedule weights or objective "
                "range. It evaluates a split product, so assess its step-count accuracy. Scaling the "
                "objective down, or increasing objective_scale in the augmented-Lagrangian options, "
                "can reduce the Schrodinger potential contribution but changes the selected dynamics"
            )
        elif stage == "symbolic tables and binary compilation":
            remedy += ". Fewer steps reduce the compilation term at fixed table structure"
        elif stage == "symbolic expansion":
            remedy += ". Reduce the symbolic expansion or use a less expanded objective representation"
        remedy += ". A smaller grid reduces explicit table and state dimensions. Replan to check all charges"
        raise ValueError(
            f"{stage} requires {'at least ' if lower_bound else ''}{work} work units and {data_bytes} bytes, "
            f"with max_work={self.max_work} and max_bytes={self.max_bytes}. {remedy}"
        )

    def _admit_symbolic_work(self, problem, compact):
        """Raise before SymPy expansion, table evaluation or block compilation exceeds the limits.

        One running total of work covers the initial state, the step rows,
        the tables and the compiled blocks of a Plan. It starts with the
        initial state's evaluation, ``d K`` units, and the step rows, which
        ``plan`` admits together before that evaluation runs and which this
        method admits again before the expansion. The first and the final admission also
        include the initial state's bytes (``_initial_state_bytes``), its
        evaluation and stored payload, and the step rows' bytes, so they
        belong to this admission like its work. A
        layer that bounds a run by the admissions per Plan
        (``constrained_records.AugmentedLagrangianRecord.admitted_work_bound``,
        ``refinement_records.BoxRefinementResult.max_plannings``) therefore
        counts the initial state inside this admission.

        The monomial bound (``objective.monomial_bound``) runs on the
        unexpanded tree of the objective centered as the compiler expands it
        (``objective.centered_objective``), so the number of monomials that
        the expansion forms is admitted before it runs, with an untuned 16
        bytes per monomial. After the expansion,
        each support group S has
        ``K**|S|`` grid tuples. At each of them the compiler forms |S|
        coordinates and evaluates the lambdified support expression
        (``ObjectiveDecomposer.support_expressions``), whose tree has ``N_S``
        nodes (``objective.node_count``). The tables therefore need
        ``sum_S K**|S| (N_S + |S|)`` work, one unit per node and per
        coordinate, admitted before the first evaluation. This remains a
        conservative logical node/coordinate-visit law for pointwise
        expression trees. A length-b array operation represents b visits.
        This is not a primitive CPU-operation count. Flattened indexing uses
        several integer operations per coordinate and validation adds passes.

        Table bytes. Admit the eight-byte support entries, their immutable
        snapshot peak, centered coordinate vectors and the workspace of the
        selected evaluator. Choose each slab from the remaining byte
        allowance, including expression intermediates and finite-real
        validation. Planning admits table construction and identity
        serialization as separate phases against ``max_bytes``, adding each
        phase's live arrays and metadata to its workspace. The serialization
        allowance includes the portable base64 strings, base64 conversion and
        the complete JSON and UTF-8 encoding buffers. With
        ``E = sum_S K**|S|`` entries the stored payload is 8E.
        ``_support_table_bytes`` returns ``F_tables = 8E + 8dK + H_tables``,
        the final tables, the centered coordinate vectors and the qualified
        metadata allowance, the construction workspace W_eval, the largest of
        the coordinate staging, one table's freezing snapshot and its slab
        charge ``B_slab,S(b) = W_S(b) + (8 |S| + 64) b + H_S`` for a
        flattened chunk of b entries (``compiler.support_chunk_size``), and
        the serialization workspace I of the tables and the initial vectors.
        The two workspaces are successive, so this admission charges
        ``B_held + F_tables + max(W_eval, I)``. The evaluator term
        ``W_S(b) = 8 b (N_S + 2)`` and its headers ``H_S = 256 (N_S + 2) + H0``
        (``compiler.support_workspace``) are an engineering
        allowance, not a derived bound. Its premise is that the NumPy
        printer's evaluation holds at most one slab-sized temporary per
        expression node, plus the output and one broadcast input. Every other
        population of the admission (the initial state and step rows, the
        compiled blocks, and for the binary encoding its compilation arrays)
        is reserved first as ``B_held``, and b_S is the largest chunk with
        ``B_slab,S(b_S) <= R = max_bytes - B_held - F_tables``.
        A compiled product adds one work unit and the measured
        allowance ``_BLOCK_BYTES`` per block occurrence. With L links per variable
        (``OneHotGrid.num_links``, K - 1 on the Dirichlet chain and K on the
        periodic cycle), a second-order step emits ``L + floor(L/2)``
        hopping blocks per variable, because the ``floor(L/2)`` odd-indexed
        links appear in both half layers, and a first-order step emits L.

        Use ``_select_binary_native``'s unit, extended to count each ``nextafter`` once in numerical
        bounds. Exact Fraction arithmetic counts as arithmetic operations, without pricing integer
        bit complexity. Float/Fraction representation conversions, comparisons and count bookkeeping
        are excluded. Logical construction keeps the existing one structural unit per block. The
        bound prices ``BinaryModel`` and ``compile_binary_steps``, after objective evaluation and
        before either runs. It does not price a native dense wrap, since planning only selects a
        dense construction and does not lower it.

        For a potential table of N entries, ``_scaled_absolute_sum_up`` costs at most ``6N+2``. Its
        ordinary path has N absolute values, at most N-1 sum additions, one upward step and one
        upward division (two operations). Its fallback has the attempted reduction, at most one
        upward step to infinity, N upward divisions and N upward additions. This is at most 6N, so
        ``6N+2`` covers both paths and empty/all-zero shortcuts. The potential identity radius adds
        six operations for ``_gamma_up``, two for multiplication, at most one upward conversion, and
        two for addition. Thus charge ``6N+13`` per potential table, dense tables included. Add
        ``(n+4)N`` for its Walsh transform only when the requested potential synthesis can use Walsh
        coefficients.

        For each kinetic table, keep C(b) and add 33: at most 27 for the spectral identity
        enclosure, four for its two outward distances, and two for ``BinaryModel.spacings``. The
        finite-difference enclosure is cheaper. These are setup costs even when the eventual kinetic
        synthesis is dense.

        For a dense-requested block, the exponent costs one, synthesis costs ``N + r_d + 3``, and
        its two numerical tally additions cost two. With ``r_d<=N``, this is at most ``2N+6`` value
        operations. Including the structural block unit gives

        ``L_dense(N) = 2N+7``.

        For a Walsh-capable request (``walsh_rotations`` or ``min_cx``), the exponent, exponent
        absolute value, doubling, N-1 trial angle products and two tally additions cost ``N+4``. If
        dense wins, the additional dense products and omissions cost at most ``2N+2``, so this
        branch is at most ``3N+6``, before its structural unit.

        If Walsh wins, partition its N-1 angles into kept r, threshold-dropped P, below-range o, and
        exact zero entries. The kept half-sum costs at most ``6r+2``, threshold drops ``4P``, and
        below-range charges ``2o``, plus one final range-charge product. Therefore the variable work
        is at most ``6(N-1)+2``. The two identity-product operations add two. The fixed ``charge``
        work is at most 64 operations: identity absolute value 1, exponent-error formation 8, exact
        phase residual 3, optional omitted-identity sum 1, outward conversion allowance 2,
        identity-error expression 14, reconstruction admission 18, and reconstruction allowance 17.
        This deliberately counts the reconstruction admission's scalar bound arithmetic even though
        it is a range predicate. Finally, the three ``_sum_up`` reductions at the end of compilation
        cost at most six operations per selected Walsh block. Including the structural unit, these
        terms are at most ``7N+74``. Use the integer ceiling

        ``L_Walsh-capable(N) = 7N+80``.

        This charge is based on the requested synthesis, before its choice is known. It covers both
        the selected Walsh branch and a ``min_cx`` trial followed by dense synthesis. No trial is
        skipped in the accounting.

        Each step costs at most nine further operations: the second-order half duration, constant
        coefficient and phase formation, the more expensive below-range constant-event charge, and
        its share of the final phase sum. The mutually exclusive normal and omitted branches must
        not both be multiplied by the step count. Once per compilation, the QFT angle list and
        inverse cost at most ``b(b-1)``, the AQFT allowance at most ``5b+2``, and dt costs one. Add
        eight for the two final dropped-angle scalings, the pruning sum and outward conversion, and
        fixed scalar formation. The convenient ceiling is ``b**2+4b+16``.

        Writing H for the potential support set and f=1 or 2 for first or second order, the added
        binary compilation charge is

        W_compile = d [C(b)+33]
         + sum_(S in H) [6N_S+13 + 1_(potential can use Walsh) (n_S+4)N_S]
         + N_steps [d L_kinetic(K) + f sum_(S in H) L_potential(N_S)]
         + 9N_steps + b**2+4b+16.


        The symbolic/table/initial-state total is added once. Table arrays and compiled records use
        an untuned 64 bytes per table entry and ``_BLOCK_BYTES`` per block, in addition to the
        initial state and evaluated tables. A binary classical Schrodinger or split-step Plan has
        ``compact=False``, so it pays none of this compilation charge. Its support-table and
        initial-state work still apply.

        Step rows. Every coefficient rule forms one weight row per step
        (``schedules.step_weights``): the midpoint rule's two point values,
        2 units, or the integrated rule's interval integrals,
        ``interval_work`` units, the integrand evaluations and closed forms
        of one step (129 for the cubic schedule's quadrature, 2 otherwise).
        The Plan keeps each row and the rows' share of the records and their
        identity JSON, charged the measured ``_STEP_BYTES`` per step. Its
        measurement also held one Schrodinger generator norm per step, which
        planning does not keep, because ``_generator_norms`` yields the norms
        one at a time. The rows are admitted
        with the initial state before the expansion, so that a large
        ``num_steps`` is refused before any row exists. Planning's later
        charges (the evolution laws, ``_host_construction``) come after the
        rows exist and do not replace this one.

        Returns:
            ``(E, decomposer, work, chunks)``, the number of grid tuples
            evaluated across all support tables, the expanded
            ObjectiveDecomposer whose support expressions were counted, for
            the compiler to evaluate, the admitted work: the monomial bound
            plus the final running total of the initial state, the step rows,
            the tables and the compilation, and the entries per evaluation
            chunk of each support (``compiler.QHDCompiler``). Automatic slack
            selection of the augmented-Lagrangian layer compares this work
            (``constrained._planned_costs``), and the planning of such a
            candidate reuses that admission (``_ADMITTED_SYMBOLIC``) instead
            of expanding its objective again.
        """
        admitted = _ADMITTED_SYMBOLIC.get()
        if admitted is not None and admitted[0] == (self.content_id, problem.content_id, bool(compact)):
            return admitted[1]
        d, k = len(problem.variables), self.num_grid_points

        # The initial state's evaluation (plan) and the step rows open the
        # running total: 2 point values or interval_work units per row, and
        # _STEP_BYTES per step that the Plan keeps.
        rows = self.num_steps * (type(self.schedule).interval_work if self.coefficient_rule == "integrated" else 2)
        state_bytes = _initial_state_bytes(d, k) + _STEP_BYTES * self.num_steps
        work, evaluations = d * k + rows, 0
        self._admit(work, state_bytes, stage="initial state and schedule rows")
        # The compiler expands the objective about expansion_centers, so the
        # bounds and the table groups follow that centered expression.
        centers = expansion_centers(problem.bounds)
        monomials = monomial_bound(
            centered_objective(problem.objective, problem.variables, centers),
            lambda count: self._admit(count, 16 * count, stage="symbolic expansion", lower_bound=True))
        decomposer = ObjectiveDecomposer(problem.objective, problem.variables, centers)
        nodes = {}
        for support, expression in decomposer.support_expressions.items():
            tuples = k ** len(support)
            evaluations += tuples
            # A count above the remaining work budget refuses without
            # traversing the rest of the tree.
            limit = max(0, self.max_work - work) // tuples
            nodes[support] = node_count(expression, limit)
            work += tuples * (nodes[support] + len(support))
            self._admit(work, 0, stage="objective support tables", lower_bound=True)
        # Blocks per step: the second-order hopping count L + floor(L/2) per
        # variable for L links (odd-indexed links appear twice), an upper
        # bound for first order, plus one or two projector passes over the
        # support tables.
        links = OneHotGrid(problem.variable_names, problem.bounds, k,
                           self.include_boundary_points, self.boundary).num_links
        occurrences = (
            self.num_steps
            * (d * (links + links // 2) + (2 if self.trotter_order == 2 else 1) * evaluations)
            if compact
            else 0
        )
        if self.encoding == "binary":
            from .binary import kinetic_setup_work

            factors = 2 if self.trotter_order == 2 else 1
            supports = tuple(decomposer.support_expressions)
            blocks = self.num_steps * (d + factors * len(supports)) if compact else 0
            compilation = 0
            if compact:
                bits = k.bit_length() - 1
                potential_choice = self.binary_synthesis.potential
                kinetic_choice = self.binary_synthesis.kinetic_phase

                # Requested synthesis determines the all-branch ceiling derived above.
                def block_work(entries, choice):
                    return 2 * entries + 7 if choice == "dense_diagonal" else 7 * entries + 80

                compilation = d * (kinetic_setup_work(bits) + 33)
                for support in supports:
                    entries = k ** len(support)
                    n = bits * len(support)
                    compilation += 6 * entries + 13
                    if potential_choice != "dense_diagonal":
                        compilation += (n + 4) * entries
                compilation += self.num_steps * (
                    d * block_work(k, kinetic_choice)
                    + factors * sum(block_work(k ** len(s), potential_choice) for s in supports)
                    + 9
                )
                compilation += bits * bits + 4 * bits + 16
            held = state_bytes + 64 * (evaluations + d * k) + _BLOCK_BYTES * blocks
            chunks = self._admit_table_phases(work + compilation, d, k, held, nodes,
                                              "symbolic tables and binary compilation")
            return evaluations, decomposer, monomials + work + compilation, chunks
        from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import (
            structured_number_projector_provider,
        )

        wrap_entries = max(
            (1 << len(support) for support in nodes
             if structured_number_projector_provider(len(support))["provider"] == "diagonal_synthesis"),
            default=0,
        ) if compact else 0
        held = state_bytes + _BLOCK_BYTES * occurrences
        chunks = self._admit_table_phases(
            work + occurrences, d, k, held, nodes,
            "symbolic tables and one-hot compilation", wrap_entries=wrap_entries)
        return evaluations, decomposer, monomials + work + occurrences, chunks

    def _admit_table_phases(self, work, d, k, held, nodes, stage, *, wrap_entries=0):
        """Admit the successive support-table, wrapped-phase and identity phases.

        Each phase shares held plus F_tables. A wrapped-phase owner reserves one
        scalar cache and its largest miss, 65536+40E bytes for E=2**s, and releases
        the cache when potential compilation finishes. The admitted scratch is
        max(W_eval, I, B_wrap), with zero B_wrap when this route performs no wrap.

        The serialization allowance I includes the portable base64 strings,
        base64 conversion and the complete JSON and UTF-8 encoding buffers
        (``_support_table_bytes``), and the admission also establishes
        ``I <= max_bytes - B_held - F_tables`` before any table is formed.
        For a support of size s and K >= 2, ``K**s >= 2**s``; a table's base64
        length ``q = 4 ceil(8 K**s/3)`` enters the identity length J, so
        ``12J >= 12q >= 128 K**s >= 128 * 2**s``, and I also contains
        H0 = 65536, hence ``I >= 65536 + 128 K**s >= 65536 + 40 * 2**s``
        (``wrapped_phase_workspace_bytes``). The wrap phase therefore does not
        raise the admitted maximum under the present table law, while the
        cache release after step compilation keeps the phases successive.
        """
        from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import wrapped_phase_workspace_bytes

        reserved, workspace, identity, chunks = _support_table_bytes(self.max_bytes, d, k, held, nodes)
        wrapped = wrapped_phase_workspace_bytes(wrap_entries)
        self._admit(work, held + reserved + max(workspace, identity, wrapped), stage=stage)
        return chunks

    def plan(self, problem, *, output, execution, shots, rng):
        """Choose the finite grid, the ordered evolution steps and the quantum or classical readout.

        `nwqlib.plan` and `solve` call it. Planning checks the objective and the operation
        sizes, evaluates each objective support table once, compiles the ordered step blocks
        when quantum execution or the `ir_product` flavor needs them, and binds either the
        compact circuit block or the classical kernel. It builds no circuit and evolves no
        state.

        Binary byte check. Compilation reserves its source records, initial vectors,
        compiler term lists and model baseline. After compilation releases the model,
        classical `ir_product` planning also checks the run's model phase. For `D = K**d`,
        potential-table sizes E_t and `E_max = max(K, E_1, ..., E_T)`, it requires
        `B_source + 48*D + 16*(sum(E_t) + K + E_max) + B_local` bytes. B_source is the byte
        allowance of the Plan's source records. B_local is the binary model's storage, plus
        one current construction per table, plus the larger of its synthesis workspace and
        its phase-application workspace. Both models use the same cutoff, swaps and
        potential-Walsh setting. This checks the named run arrays under the object
        allowances of the
        [engineering constants](../../ENGINEERING_CONSTANTS.md#qhd-selected-operation-sizes).
        Content hashing of the arrays is a separate phase.

        Range check. Planning completes the range check of its arithmetic, with or without a
        kept state. Before pruning it checks that the nonzero durations, weights,
        coefficients, angles and identity products are normal binary64 numbers, that the
        grid coordinates are distinct and increasing, and that the power-of-two scalings of
        the one-hot rotation count are in range. Stored table values need only be finite. A
        data-dependent contribution whose product would be nonzero and below `2**-1022`,
        such as the far tail of a narrow Gaussian well, is omitted and counted in the
        circuit's error sources. The structured one-hot chain likewise stops before a link
        whose parameter would fall below that range, and its omitted links are counted there
        too. Kinetic terms outside the range, and every overflow, are refused. Used half
        durations and projector scalings are then exact and normal, and phase sums and
        synthesized phases have finite intermediates. These conditions supply the range
        assumptions of the phase and angle error bounds. They do not establish the accuracy
        assumptions of scalar libraries, transforms or the circuit construction of dense
        diagonals, and a kept state has its own phase-accuracy check. Positive error bounds
        may be subnormal and are rounded upward.
        """
        if not isinstance(problem, Optimization) or not isinstance(output, OptimizationCandidate):
            raise ApplicabilityError("QHD requires an Optimization and OptimizationCandidate")
        from .validation import _validate_qhd_problem_fields

        _validate_qhd_problem_fields(problem)
        if execution == "classical" and shots is not None:
            raise ValueError("classical QHD has no sampled circuit acquisition")
        if self.initial_state_preparation == "none" and execution != "quantum":
            raise ValueError(
                "initial_state_preparation='none' describes a quantum resource-only construction"
            )
        if self.keep_state and shots is not None:
            raise ValueError("keep_state requires exact amplitude acquisition")
        self._kinetic_route(execution)
        # The binary structured preparation is one H per qubit, which prepares
        # the uniform state, equal to the kinetic ground state of the periodic
        # grid (initial_state.append_binary_initial_state). Classical
        # execution prepares no circuit.
        if (execution == "quantum" and self.encoding == "binary" and self.initial_state_preparation == "structured"
                and not isinstance(self.initial_state, (UniformState, KineticGroundState))):
            raise ValueError(
                "the structured preparation of the binary encoding is one H per qubit, which prepares the "
                f"uniform state only, not {type(self.initial_state).__name__}. Use "
                "initial_state_preparation='qiskit_state_preparation'"
            )
        # The physical one-hot register has d*K qubits, while its admissible
        # grid sector has K**d states. These are distinct resource dimensions.
        # The binary register has d*b qubits and holds exactly the K**d grid
        # states.
        d, k, bits = len(problem.variables), self.num_grid_points, _bits(self)
        width, dimension = d * (k if bits is None else bits), k**d
        # Evaluating the initial state takes a bounded number of scalar
        # operations per grid point of each variable, one work unit each, and
        # the measured allowance _initial_state_bytes for its evaluation and
        # stored payload. The grid's coordinate check (OneHotGrid.__post_init__)
        # is one more such pass, covered by the same unit, so this admission
        # runs before the grid is built. The initial state is then evaluated
        # once and checks its domain (a Gaussian's dimension and representable
        # amplitudes) before any table work. This is the one evaluation of the
        # amplitudes and of the start vector's error bound. The reconstruction
        # stores both, and no later stage evaluates them. The same units open
        # the running total that _admit_symbolic_work admits with the tables,
        # and its final admission includes these bytes, so this check is a
        # first part of that admission, not a separate one. The schedule rows
        # that _admit_symbolic_work adds next are fixed by the step count and
        # the schedule rule, so the known work d*K + rows and bytes
        # _initial_state_bytes(d, K) + _STEP_BYTES*S of both phases are
        # checked here, before the initial state is evaluated.
        rows = self.num_steps * (type(self.schedule).interval_work if self.coefficient_rule == "integrated" else 2)
        self._admit(d * k + rows, _initial_state_bytes(d, k) + _STEP_BYTES * self.num_steps,
                    stage="initial state and schedule rows", later={
                        "max_work": "symbolic supports, table evaluation and selected evolution",
                        "max_bytes": "support tables, their identities and selected evolution workspace"})
        grid = OneHotGrid(problem.variable_names, problem.bounds, k, self.include_boundary_points,
                          self.boundary)
        if self.kinetic_model == "spectral":
            # The spectral kinetic energies reach pi**2/(2 h**2) at the Nyquist
            # mode (split_step.kinetic_eigenvalues, binary.kinetic_walsh). The
            # grid guard keeps 1/h**2 to 1/(4 h**2) normal, which leaves this
            # energy free to overflow, so it is checked before any table work.
            for j, name in enumerate(problem.variable_names):
                h = grid.spacing(j)
                # E_Nyq = 2 (pi/(2 h))**2. The products overflow to inf where a power would raise.
                wave = pi / (2.0 * h)
                if not isfinite(2.0 * (wave * wave)):
                    raise ValueError(
                        f"kinetic_model='spectral' on the grid of {name} with spacing h = {h!r} has the Nyquist "
                        "energy pi**2/(2 h**2), which is not a finite binary64 number. Use a wider box, fewer "
                        "grid points or kinetic_model='finite_difference'"
                    )
        amplitudes, start_error = evaluate_initial_state(self.initial_state, grid)
        # Native execution and the ir_product flavor apply the compiled step
        # blocks. The schrodinger and split_step flavors need only the support tables.
        compact = execution == "quantum" or self.theory_flavor == "ir_product"
        split = execution == "classical" and self.theory_flavor == "split_step"
        evaluations, decomposer, _, chunks = self._admit_symbolic_work(problem, compact)
        # Evaluate objective terms only over their own variable support and
        # freeze the selected ordered time-step blocks for both execution and costs.
        # Each SupportValues shares the compiler's frozen table and selects its extrema.
        compiler = QHDCompiler(problem, self, decomposer, chunks)
        tables = tuple(
            SupportValues.tabulate(support, values)
            for support, values in compiler._potential_grid_values().items()
        )
        constant = coerce_real_scalar(compiler.decomposer.constant, context="constant objective")
        if bits is None:
            raw_steps = compiler.build_step_pauli_ir() if compact else ()
            # The wrapped-phase cache ends with step compilation; metadata reads the ledgers, not the cache.
            compiler.potential_compiler._wrapped.clear()
            metadata = compiler.metadata()
            steps = _block_records(raw_steps)
            phases = metadata["dropped_global_phase"]
            dropped = metadata["rotation_threshold_truncation"]
            omitted = metadata["range_omissions"]
            ledger = dict(
                physical_phase=phases["phase_angle"],
                phase_sources=tuple(phases["per_source_total_angles"].items()),
                dropped_angle_sum=dropped["dropped_absolute_angle_sum"],
                dropped_count=dropped["dropped_nonzero_contribution_count"],
                # Each dropped one-hot block differs from I by at most its angle, and an omitted projector by
                # (1 - 2**-s)|t b v| in its traceless part (records.QHDReconstruction.pruning_error_bound). The
                # exact sum is rounded upward once.
                pruning_error_bound=_upward(Fraction(dropped["dropped_angle_units"], 1 << 1074)
                                            + omitted["projector_charge"]),
                aqft_error_bound=0.0,
                range_omissions=dict(omitted, walsh_tables=()),
            )
        elif compact:
            # Binary compilation reserves the source tables and records, compiler term-list slots, initial
            # vectors, model storage and construction workspace before building the model. After compilation
            # releases that model, quantum planning admits the resource-inspection census under the same work
            # and byte limits, with source tables counted once. Array identity encoding is admitted as a
            # successive phase.
            from .binary import BinaryModel, _require_bytes, compile_binary_steps

            specs = tuple((int(t.values.array.size), len(t.support)) for t in tables)
            groups, constants = compiler.decomposer._groups
            grouped_terms = len(constants) + sum(len(terms) for terms in groups.values())
            source_bytes, planning_bytes = _binary_source_reservation(
                d, k, specs, self.num_steps, self.trotter_order, grouped_terms=grouped_terms)
            payload = 8 * sum(e for e, _ in specs)
            _, _, identity_bytes, _ = _support_table_bytes(
                self.max_bytes, d, k, 0, dict.fromkeys((t.support for t in tables), 0))
            _require_bytes(self.max_bytes, held=planning_bytes, local=identity_bytes,
                           stage="binary planning identity")
            model = BinaryModel(compiler.grid, self, tables, held_bytes=planning_bytes)
            model_local = self.max_bytes - planning_bytes - model._cache_capacity
            steps, ledger = compile_binary_steps(model, self, compiler.step_weights, constant)
            del model
            if execution == "quantum":
                from .resources import _admit_census

                _, _, held = _admit_census(
                    self, steps, tables, held_bytes=planning_bytes - payload)
                _require_bytes(self.max_bytes, held=held, local=model_local,
                               stage="binary planning and resource inspection")
            else:
                # The classical ir_product kernel builds the same model beside its source records, state
                # buffers and phase buffers (theory.run_binary_product), so planning admits that phase too.
                entries = [e for e, _ in specs]
                run_held = (source_bytes + 48 * k**d
                            + 16 * (sum(entries) + k + max([k, *entries])))
                _require_bytes(self.max_bytes, held=run_held, local=model_local,
                               stage="classical binary product and model")
        else:
            steps, ledger = (), dict(physical_phase=0.0, phase_sources=(),
                                     dropped_angle_sum=0.0, dropped_count=0, pruning_error_bound=0.0,
                                     aqft_error_bound=0.0, range_omissions=dict(walsh_tables=()))
        walsh = ledger.pop("walsh_phase", None)
        stored = stored_amplitudes(amplitudes)
        if execution == "classical" and self.keep_state and self.theory_flavor != "ir_product":
            # The one constant phase that these flavors restore for a kept state (_constant_phase). An
            # overflow is left to _admit_constant_phase, which names the constant.
            try:
                lost = _constant_phase_and_omission(constant, compiler.step_weights, self)[1]
            except OverflowError:
                lost = 0
            if lost:
                ledger["range_omissions"].update(identity_events=1, identity_charge=lost)
        ledger["range_omissions"] = _range_omissions(
            tables, ledger["range_omissions"],
            chains=bits is None and execution == "quantum" and self.initial_state_preparation == "structured",
            amplitudes=stored)
        # The native circuit adds the compiled identity phase to its global
        # phase and the ir_product reference multiplies its state by it before
        # the probabilities are read, so every readout of these routes needs it
        # finite. A finite constant can still overflow the per-step sum.
        overflowed = sorted(s for s, a in ledger["phase_sources"] if not isfinite(a))
        if compact and (overflowed or not isfinite(ledger["physical_phase"])):
            raise ValueError(
                "the compiled physical phase of the QHD circuit exceeds the binary64 range"
                + (f" in its {', '.join(overflowed)} part" if overflowed else "")
                + ". The native circuit and the ir_product reference apply it before any readout"
                + (". The objective constant does not affect the optimization, so removing it from the "
                   "objective removes its part" if "objective_constant" in overflowed else "")
                + ". The classical schrodinger or split_step flavor without keep_state reads the "
                "probabilities without this phase"
            )
        # Host evolution acts on the restricted grid sector. Native one-hot
        # execution uses the full one-hot register, including invalid encoded
        # states.
        if execution == "classical" and bits is not None and self.theory_flavor == "ir_product":
            from .theory import binary_product_sizes

            synthesis = self.binary_synthesis
            size, workspace = binary_product_sizes(
                dimension, bits, d, [k ** len(t.support) for t in tables], steps,
                cutoff=synthesis.aqft_cutoff, swaps=synthesis.qft_bit_reversal == "swap",
                walsh=synthesis.potential != "dense_diagonal",
                start_work=start_vector_work(self.initial_state, compiler.grid))
        elif split:
            from .split_step import sizes

            size, workspace = sizes(compiler.grid, tables, self.num_steps,
                                    start_vector_work(self.initial_state, compiler.grid))
        elif execution == "classical":
            norms = _generator_norms(self.theory_flavor, self, compiler.grid, tables, tuple(steps),
                                     compiler.step_weights)
            size, workspace = restricted_sizes(
                dimension, d, len(tables), (compiler.step_weights, tuple(steps)), self.theory_flavor, norms,
                start_vector_work(self.initial_state, compiler.grid),
            )
        else:
            size, workspace = 0, 0
        # The Plan's state budget, which bounds the split-step phase angles
        # (split_step.state_error), and the mass and tie windows formed from
        # it, computed once and stored. They must be finite before the kernel
        # is bound. A nonfinite one is refused after the kernel's own work
        # admission (_host_construction), which names the larger cost first.
        host, refused = {}, None
        if execution == "classical":
            try:
                host = _admit_host_windows(self, compiler.grid, tables, tuple(steps), compiler.step_weights,
                                           start_error, dimension)
            except ValueError as error:
                refused = error
        r = QHDReconstruction(
            method_id=self.content_id,
            expression=sp.srepr(problem.objective),
            symbolic_variables=tuple(sp.srepr(v) for v in problem.variables),
            support_values=tables,
            constant=constant,
            initial_amplitudes=stored,
            initial_state_error=start_error,
            steps=tuple(steps),
            step_weights=compiler.step_weights,
            compact_schedule_selected=compact,
            **ledger,
            walsh_phase=None if walsh is None else QHDWalshPhase(**walsh._asdict()),
            **host,
            width=width,
            restricted_dimension=dimension,
            support_evaluations=evaluations,
            size_units=size,
            workspace_bytes=workspace,
        )
        if execution == "classical" and self.keep_state and self.theory_flavor != "ir_product":
            _admit_constant_phase(r, self)
        elif compact and self.keep_state:
            _admit_compiled_phase(r, self, compiler.grid, _phase_contributions(r, self), execution, r.walsh_phase)
        domain = "selected finite QHD model"
        if split:
            domain += f"; {self.kinetic_model} kinetic model; {self.coefficient_rule} coefficient rule"
        source = Source(
            name="qhd." + (self.theory_flavor if execution == "classical" else "compact"),
            version="2",
            domain=domain,
            reference=METHOD.reference + "; selection " + r.content_id,
        )
        reference = objective_reference(r.expression, r.symbolic_variables, problem.bounds)
        # A full state is an explicitly requested output with its own size check.
        # Ordinary candidate readout does not acquire this array for reporting.
        outputs = ()
        if self.keep_state:
            state_dimension = dimension if execution == "classical" else 1 << width
            self._admit(state_dimension, 16 * state_dimension, stage="kept state")
            outputs = (
                ArrayOutput(
                    name="state",
                    kind="vector",
                    basis=Basis(
                        identity="qhd-" + execution,
                        dimension=state_dimension,
                        ordering="lexicographic grid"
                        if execution == "classical"
                        else "little-endian one-hot qubits"
                        if bits is None
                        else "little-endian binary registers",
                    ),
                    frame="physical",
                    global_phase="physical",
                ),
            )
        if execution == "classical":
            construction, observation, kernel = self._host_construction(
                r, d, source, reference, outputs, size, workspace
            )
            if refused is not None:
                raise refused
            blocks = ()
        else:
            census_held = 0
            if bits is None:
                groups, constant_terms = compiler.decomposer._groups
                grouped_terms = len(constant_terms) + sum(len(terms) for terms in groups.values())
                census_held = 8 * d * k + 16 * grouped_terms
            construction, observation, blocks = self._native_construction(
                problem, r, source, outputs, shots, amplitudes,
                census_held_bytes=census_held)
        from .binary import qft_controlled_phases

        # A cutoff below b - 1 omits controlled phases of every QFT.
        truncated = compact and bits is not None and qft_controlled_phases(
            bits, self.binary_synthesis.aqft_cutoff) < bits * (bits - 1) // 2
        model = _error_model(
            problem, output, construction, compact=compact, split=split, kinetic_model=self.kinetic_model,
            execution=execution, shots=shots, rule=self.coefficient_rule, aqft=truncated,
        )
        plan = Plan(
            problem=problem,
            method=self,
            output=output,
            execution=execution,
            shots=shots,
            randomness=rng.snapshot(),
            construction=construction,
            reconstruction=r,
            error_model=model,
            experiments=()
            if self.initial_state_preparation == "none"
            else ((Experiment(name="qhd", setting="qhd", observation=observation),)
                  if execution == "classical" else
                  (Experiment(name="qhd", batch="root", setting_index=0,
                              readout=ReadoutDetails(**observation.model_dump(
                                  exclude_computed_fields=True,
                                  exclude={"kind", "shots", "parent_id", "schema_version"}))),)),
            assumptions=(
                "observed candidate is not a certified optimum",
                f"finite {self.encoding.replace('_', '-')} {_BOUNDARY_NAMES[self.boundary]} box discretization",
            ),
        )
        if execution == "classical":
            blocks = (BoundKernel._bind(plan, kernel, lambda: _execute_theory(plan, kernel)),)
        validate_selection(plan)
        return plan._bind(blocks=blocks)

    def _host_construction(self, r, d, source, reference, outputs, size, workspace):
        """Select the classical kernel that evolves the restricted grid model on the host.

        ``size`` and ``workspace`` come from ``restricted_sizes``, or from
        ``split_step.sizes`` for the ``split_step`` flavor. Admission adds the
        summary of all D = K**d grid points, every one of them possibly
        positive (``summary_sizes``), whose byte law includes the float64
        probability array of ``_execute_theory``, 8 D bytes. The ``d K + 3 d + 13`` returned scalar records
        (``_scalar_names``) add the measured ``_SCALAR_BYTES`` each, and the
        call adds the measured ``_KERNEL_CALL_BYTES`` once for Python objects
        of bounded total size that no size law counts.

        Returns:
            The selected construction, its ``host_scalars`` observation and
            the selected kernel, which ``plan`` binds once the Plan exists.
        """
        k, dimension = self.num_grid_points, r.restricted_dimension
        summary_work, summary_bytes = summary_sizes(r, d, k, dimension)
        labels = _scalar_names(d, k)
        size += summary_work
        # The returned ScalarValue records take the measured _SCALAR_BYTES each,
        # and the call's bounded Python objects _KERNEL_CALL_BYTES.
        workspace += summary_bytes + _SCALAR_BYTES * len(labels) + _KERNEL_CALL_BYTES
        self._admit(size, workspace, stage="classical host evolution and readout")
        kernel = SelectedKernel(
            name="qhd",
            implementation=source,
            inputs=(reference,),
            scalars=labels,
            scalar_frames=("physical",) * len(labels),
            outputs=outputs,
            construction_work=1,
            invocation_work=size,
            dependencies=("numpy", "scipy"),
            resource_laws=(
                ResourceLaw(
                    metric="classical_work",
                    basis="selected_logical",
                    value=size,
                    interpretation="estimate",
                    evidence=Evidence(kind="numerical_estimate", source=source),
                    assumptions=(
                        "split-step nominal transform proxy L D and phase units; "
                        "ducc0.fft internal work unknown"
                        if self.theory_flavor == "split_step"
                        else "selected sparse model shape/action units; adaptive SciPy work unknown",
                    ),
                ),
            ),
            workspace=(
                Workspace(location="host", purpose="workspace", bytes=workspace, source=source),
                Workspace(location="host", purpose="workspace", bytes=None, source=source),
            ),
        )
        program = Program(
            root="qhd",
            definitions=(
                Definition(
                    id="qhd",
                    node=ClassicalStage(implementation=source, boundary="host", kernel="qhd"),
                ),
            ),
        )
        construction = SelectedConstruction(program=program, selections=(), kernels=(kernel,))
        observation = ObservationSpec(kind="host_scalars", labels=labels)
        return construction, observation, kernel

    def _native_construction(self, problem, r, source, outputs, shots, amplitudes, *, census_held_bytes=0):
        """Select the compact native block and the Program that allocates, evolves and reads out.

        Without ``shots`` the Program returns the exact probabilities of all
        ``d*K`` register qubits. With ``shots`` it measures the register into
        one classical value. ``keep_state`` replaces the probability readout
        by the exact amplitudes of the whole register with unit recovery
        scale. ``amplitudes`` are the initial state's per-variable vectors,
        which set the chain's emitted links (``_select_native``).
        ``census_held_bytes`` are the caller's buffers live during the
        one-hot rotation census (``_select_native``).

        Returns:
            The selected construction, its observation and a one-element
            tuple holding the bound native block.
        """
        width = r.width
        _admit_register_basis(width)
        if self.encoding == "binary":
            block = self._select_binary_native(problem, r, source, amplitudes)
        else:
            block = self._select_native(
                problem, r, source, amplitudes, census_held_bytes=census_held_bytes)
        signature = block.record.signature
        defs = [
            Definition(id="allocate", node=Allocate(wire="qhd")),
            Definition(
                id="evolve",
                node=BlockCall(
                    signature=signature.name, ports=(PortMap(port="system", wire="qhd"),)
                ),
            ),
        ]
        children = ["allocate", "evolve"]
        classical = ()
        if shots is not None:
            classical = (ClassicalValue(name="readout", dtype="bits", width=width),)
            defs.append(Definition(id="measure", node=Measure(wire="qhd", result="readout")))
            children.append("measure")
            observation = ObservationSpec(kind="counts", shots=shots)
        else:
            observation = ObservationSpec(kind="probabilities", qubits=tuple(range(width)))
        defs.append(Definition(id="release", node=Release(wire="qhd")))
        children.append("release")
        defs.append(Definition(id="body", node=Sequence(children=tuple(children))))
        defs.append(Definition(
            id="root",
            node=MeasurementBatch(
                body="body",
                settings=(Setting(label="qhd", metadata=MetadataRef(
                    format=METHOD,
                    data=InputRef(identity=problem.content_id, representation="scalar", source=METHOD),
                )),),
                repetitions=shots if shots is not None else 1,
                observation_kind="counts" if shots is not None else "probabilities",
            ),
        ))
        program = Program(
            root="root",
            definitions=tuple(defs),
            registers=(Register(name="qhd", width=width),),
            classical=classical,
            signatures=(signature,),
        )
        construction = SelectedConstruction(program=program, selections=(block.record,))
        if self.keep_state:
            from nwqlib.amplitudes import AmplitudeReadout
            from nwqlib.problems.inputs import compose_recovery

            observation = ObservationSpec(
                kind="amplitudes",
                amplitudes=AmplitudeReadout(
                    construction_id=construction.content_id,
                    source=source,
                    width=width,
                    coordinates=tuple(range(width)),
                    output=outputs[0],
                    recovery=compose_recovery(1.0),
                ),
            )
        return construction, observation, (block,)

    def _select_native(self, problem, r, source, amplitudes, *, census_held_bytes=0):
        """Bind the compact one-hot construction, its construction work, its CX bound and its rotation law.

        Construction work counts gates to emit. The structured preparation of
        the one-hot encoding, the amplitude chain
        (``initial_state.append_amplitude_chain``), has d X gates and one CRY
        and one CX per emitted link, ``d + 2 L`` for ``L = sum_j L_j``, where
        ``L_j`` is the number of links that the chain of variable j emits
        (``initial_state.chain_links``), at most ``K - 1``. Qiskit's
        ``StatePreparation`` receives one ``2**K`` amplitude vector per
        variable, charged ``d K 2**K`` work and ``16 * 2**K`` bytes for the
        complex128 vector. A block on s qubits adds ``1 + s`` units plus
        ``2**s`` for a diagonal-synthesis phase table or ``s**2`` for any
        other provider (a multi-controlled phase or the XX+YY gate), with an
        untuned 128-byte allowance per block.
        The CX law and the arbitrary-rotation law
        (``resources.rotation_law``) apply only to the structured preparation
        or no preparation, because the record has no law for Qiskit's generic
        state preparation.
        The rotation census admits its wrapped-phase cache and one miss beside
        its source records (``resources._admitted_onehot_population``), with
        ``census_held_bytes`` from the planning caller (the compiler's
        centered coordinates and grouped-term slots) and ``8dK + 120d + 40``
        for the normalized initial arrays.
        """
        d, k = len(problem.variables), self.num_grid_points
        blocks = tuple(b for group in r.steps for b in group)
        prep_work = prep_bytes = prep_cx = 0
        if self.initial_state_preparation == "structured":
            links = sum(chain_links(vector) for vector in amplitudes)
            # d X gates, one CRY and one CX per emitted link. A CRY lowers to two CX.
            prep_work, prep_cx = d + 2 * links, 3 * links
        elif self.initial_state_preparation == "qiskit_state_preparation":
            prep_work, prep_bytes = d * k * (1 << k), 16 * (1 << k)
        work = prep_work + sum(
            1
            + len(b.support)
            + (1 << len(b.support) if b.provider == "diagonal_synthesis" else len(b.support) ** 2)
            for b in blocks
        )
        self._admit(work, prep_bytes + 128 * len(blocks), stage="native one-hot construction")
        # Use the known chain/XX+YY/projector construction for a CX bound.
        # Generic SDK state preparation has no such bound in this record.
        # Terms: three CX per emitted chain link, two per fused XX+YY hopping
        # block and each projector block's selected lowering cost. The stored
        # blocks include every link of OneHotGrid.links, the periodic wrap
        # link too, so a second-order step has 2 d (L + floor(L/2)) hopping CX
        # for L links per variable.
        # The arbitrary-rotation law counts the same emitted blocks and chain
        # (resources.rotation_population), so it has the same exception.
        laws = ()
        if self.initial_state_preparation != "qiskit_state_preparation":
            from .resources import _admitted_onehot_population, rotation_law

            cx = prep_cx + sum(b.cx for b in blocks)
            laws = (
                ResourceLaw(
                    metric="cx",
                    basis="cx",
                    value=cx,
                    interpretation="upper_bound",
                    evidence=Evidence(kind="proved_relation", source=source),
                    assumptions=(
                        "selected amplitude chain, XX+YY and projector construction without routing",
                    ),
                ),
                rotation_law(_admitted_onehot_population(
                    self, r, held_bytes=census_held_bytes + 8 * d * k + 120 * d + 40)),
            )
        signature = BlockSignature(
            name="qhd",
            target=source,
            quantum=(
                QuantumPort(name="system", width=r.width, requires="zero", ensures="coherent"),
            ),
            coupling="joint",
        )
        semantics = BlockSemantics(
            kind="unknown",
            input=objective_reference(r.expression, r.symbolic_variables, problem.bounds),
            basis=Basis(
                identity="qhd-one-hot",
                dimension=1 << r.width,
                ordering="variable j occupies qubits j*K through (j+1)*K-1",
            ),
            relation=f"{self.initial_state.kind} one-hot initial state then selected ordered "
            f"compact {_BOUNDARY_NAMES[self.boundary]} product",
            input_projector="all-zero state",
            output_projector="one excitation per variable",
            success="unconditional",
            workspace=0,
            restoration="system transformed",
            epsilon=None,
            approximation_metric="objective gap unknown",
            approximation_evidence="dropped-angle ledger is not an objective-gap bound",
            inverse_legal=False,
            control_legal=False,
            phase="physical identity phase restored",
        )
        record = SelectedDefinition(
            signature=signature,
            semantics=semantics,
            implementation=source,
            choice=self.initial_state_preparation,
            decomposition=None,
            cost_law=None,
            coefficient_selection_id=r.content_id,
            cost_context="actual compact schedule; SDK synthesis and routing unknown",
            construction_work=work,
            resource_laws=laws,
            blocker="initial_state_preparation='none' is resource-only"
            if self.initial_state_preparation == "none"
            else None,
            workspace=(
                Workspace(
                    location="host",
                    purpose="preparation",
                    bytes=prep_bytes + 128 * len(blocks),
                    source=source,
                ),
            ),
        )
        from .native import construct_qhd

        return SelectedBlock.bind(
            record,
            payload=(r, self, problem.bounds, problem.variable_names),
            constructor=construct_qhd,
        )

    def _select_binary_native(self, problem, r, source, amplitudes):
        """Bind the compact binary construction, its construction work and its CX and rotation laws.

        Let ``b = log2(K) >= 1``, ``d`` be the variable count, ``S`` a potential support, and ``n_S
        = b |S|``, ``N_S = 2**n_S``. A block occurrence B has ``n_B = b |B.variables|`` and ``N_B =
        2**n_B``. Counts are per occurrence, including both second-order potential halves, not per
        distinct table.

        One value-operation unit is one scalar or array-element addition, subtraction,
        multiplication, division, absolute value, sign change, square, or elementary-function call.
        A complex exponential and ``angle`` each count as one call. Numerical omission charges are
        included. Comparisons, range predicates and the arithmetic used only to decide those
        predicates, index/count bookkeeping, copies, permutations, record construction, and
        dependency-internal arithmetic are excluded. In particular, ``_dense_nonzero`` is a range
        predicate.

        The complete count follows the actual native call chain ``construct_qhd ->
        append_binary_steps -> BinaryModel -> PhaseTable.synthesize -> append_diagonal``.

        1. A dense synthesis forms N phase products. A Walsh synthesis forms N-1 nonidentity angle
           products. Price N in either case.
        2. For a nonzero dense table, ``_multiplexors.append_control_diagonal_phases`` forms
           ``angle(exp(1j * phases))``. Its product, exponential, and argument cost ``3N``.
        3. At the successive dense recursion levels the number of pairs is ``N/2, N/4, ..., 1``.
           Each pair uses a difference, two halvings, and a sum, hence ``4(N-1)`` operations.
        4. A k-control Gray-code butterfly has k levels, each with ``2**k`` outputs. Each output
           halves two operands and adds or subtracts them. It costs ``3k2**k``. Summing over
           ``k=0,...,n-1`` gives ``3G(n)``, where

           ``G(n) = sum(k 2**k) = (n-2)2**n + 2``.

           At n=1 both sides are zero. Increasing n to n+1 adds ``n2**n`` to each side, proving the identity.
        5. Thus the dense work beyond the phase products is

           ``D(n) = 3N + 4(N-1) + 3G(n) = (3n+1)N + 2``.

           An all-zero dense table returns before the wrap and recursion. Charging D for it is conservative.
        6. A dense block adds three scalar value operations: ``abs(Fraction(exponent))``,
           ``-exponent``, and the product with the largest omitted magnitude. A Walsh block adds
           six: exponent absolute value, doubling, identity negation and multiplication,
           range-charge multiplication, and the circuit global-phase addition. Its N-entry allowance
           has one spare entry, so use ``c_B=3`` for dense and ``c_B=5`` for Walsh.
        7. A below-range dense entry adds one exact absolute value. A below-range Walsh angle adds
           an absolute value and one term of an exact sum. The extra work is ``R_d + 2 R_w``, using
           ``r.range_omissions.dense_entries`` and ``.rotations``. Identity range predicates remain
           excluded under the stated unit.
        8. A threshold-dropped Walsh rotation uses absolute value, integer division, multiplication,
           and addition to the dropped sum. Add ``4P``, where ``P=r.dropped_count`` includes all
           occurrences.
        9. A support-table Walsh transform has n levels of N additions/subtractions and N
           normalization divisions, ``(n+1)N``. For each coefficient omitted by normalization,
           ``walsh_admission`` additionally forms ``abs(Fraction(s))/N`` and, for a nonidentity
           coefficient, adds its loss. At most ``3N`` extra operations suffice. Charge ``(n+4)N``
           per transformed potential table. Dense-requested potential tables have no transform.
        10. The kinetic setup has the existing bound ``C(b)=10K+2bK+b**2+6b+16``. The source's
           model-specific counts are at most ``16+8K+(4b-3)K/2`` for finite differences and
           ``9K+b**2+5b+12`` for the spectral model. Their slack below C is respectively
           ``7K/2+b**2+6b`` and ``(2b+1)K+b+4``. For b>=1 and d>=1 this also covers the model's
           extra two spacing operations per variable, at most ``b(b-1)`` forward/inverse QFT-angle
           operations once per model, and the final physical-phase addition. There is no unpriced
           per-step QFT-angle construction.

        The resulting bound is

        W_native = W_prep
         + sum_B [c_B + N_B + cx_B + rotations_B + 2b 1_(B kinetic)]
         + d C(b) + sum_(S with Walsh coefficients) (n_S+4) N_S
         + sum_(B dense) D(n_B) + R_d + 2 R_w + 4P.


        ``W_prep`` is ``db`` for the structured H layer, ``dbK`` for the selected Qiskit preparation
        allowance, and zero for resource-only preparation. These remain declared construction
        allowances rather than SDK runtime bounds.

        Planning admits the native constructor's complete known resident and workspace phase,
        including source records, the binary model, the growing circuit's stored definitions and
        parameters, and one block's temporary construction storage (``native._binary_native_bytes``,
        ``_binary_source_reservation``, ``binary._model_reservation``). A resource-only selection
        has no circuit-construction byte phase.

        The CX law is the sum of the recorded block counts
        (``binary.compile_binary_steps``), which each block's construction
        reproduces (``native.append_binary_steps``), plus zero for the
        H layer. It is the count of the emitted circuit transpiled to
        CX and single-qubit gates at optimization level 0, before routing,
        and an upper bound for any further optimization. The rotation law
        sums the recorded rotations, an upper bound on the arbitrary-angle
        rotations because angle-specific Clifford or T replacement and
        omitted zero multiplexor angles only lower it. Its basis is
        ``selected_logical``, the basis of the one-hot rotation law
        (``resources.rotation_law``), so the default resource estimate reports
        both encodings' rotation counts, the binary one as an upper bound
        (``resources.circuit_resources`` classifies its angles exactly).
        Neither law is recorded with Qiskit's generic state preparation, whose
        synthesis count this record does not bound.
        """
        from .binary import kinetic_setup_work

        d, k, bits = len(problem.variables), self.num_grid_points, _bits(self)
        blocks = tuple(b for group in r.steps for b in group)
        prep_work = 0
        if self.initial_state_preparation == "structured":
            prep_work = d * bits
        elif self.initial_state_preparation == "qiskit_state_preparation":
            prep_work = d * bits * k
        walsh_tables = () if self.binary_synthesis.potential == "dense_diagonal" else r.support_values
        work = prep_work + d * kinetic_setup_work(bits)
        work += sum(len(t.values) * (len(t.values).bit_length() + 3)
                    for t in walsh_tables)
        for block in blocks:
            n = bits * len(block.variables)
            entries = 1 << n
            dense = block.synthesis == "dense_diagonal"
            work += (3 if dense else 5) + entries + block.cx + block.rotations
            if block.kind == "binary_kinetic":
                work += 2 * bits
            if dense:
                # D(n) = 3N + 4(N-1) + 3G(n), including each phase wrap.
                work += (3 * n + 1) * entries + 2
        work += (r.range_omissions.dense_entries
                 + 2 * r.range_omissions.rotations + 4 * r.dropped_count)
        if self.initial_state_preparation == "none":
            native_bytes = 0
        else:
            from .binary import _model_reservation
            from .native import _binary_native_bytes

            specs = tuple((int(t.values.array.size), len(t.support)) for t in r.support_values)
            source_bytes, _ = _binary_source_reservation(
                d, k, specs, self.num_steps, self.trotter_order)
            parts = _model_reservation(
                d, bits, specs, cutoff=self.binary_synthesis.aqft_cutoff,
                swaps=self.binary_synthesis.qft_bit_reversal == "swap",
                potential_walsh=self.binary_synthesis.potential != "dense_diagonal")
            native_bytes = (source_bytes + _binary_native_bytes(r, self, d)
                            + parts[0] + parts[1] + max(parts[2:]))
        self._admit(work, native_bytes, stage="native binary construction")
        laws = ()
        if self.initial_state_preparation != "qiskit_state_preparation":
            evidence = Evidence(kind="proved_relation", source=source)
            laws = (
                ResourceLaw(
                    metric="cx", basis="cx", value=sum(b.cx for b in blocks), interpretation="upper_bound",
                    evidence=evidence,
                    assumptions=("selected H-layer preparation, QFT, phase-diagonal and Walsh-rotation "
                                 "construction without routing",),
                ),
                ResourceLaw(
                    metric="arbitrary_rotations", basis="selected_logical", value=sum(b.rotations for b in blocks),
                    interpretation="upper_bound", evidence=evidence,
                    assumptions=("rotation gates before angle-specific Clifford or T replacement",),
                ),
            )
        signature = BlockSignature(
            name="qhd",
            target=source,
            quantum=(
                QuantumPort(name="system", width=r.width, requires="zero", ensures="coherent"),
            ),
            coupling="joint",
        )
        semantics = BlockSemantics(
            kind="unknown",
            input=objective_reference(r.expression, r.symbolic_variables, problem.bounds),
            basis=Basis(
                identity="qhd-binary",
                dimension=1 << r.width,
                ordering="variable j occupies qubits j*b through (j+1)*b-1, little-endian",
            ),
            relation=f"{self.initial_state.kind} binary initial state then selected ordered compact periodic "
            "binary product",
            input_projector="all-zero state",
            output_projector="every register state encodes a grid point",
            success="unconditional",
            workspace=0,
            restoration="system transformed",
            epsilon=None,
            approximation_metric="objective gap unknown",
            approximation_evidence="pruning and AQFT operator-norm bounds are not objective-gap bounds",
            inverse_legal=False,
            control_legal=False,
            phase="physical identity phase restored",
        )
        record = SelectedDefinition(
            signature=signature,
            semantics=semantics,
            implementation=source,
            choice=self.initial_state_preparation,
            decomposition=None,
            cost_law=None,
            coefficient_selection_id=r.content_id,
            cost_context="actual compact binary schedule; SDK synthesis and routing unknown",
            construction_work=work,
            resource_laws=laws,
            blocker="initial_state_preparation='none' is resource-only"
            if self.initial_state_preparation == "none" else None,
            workspace=(
                Workspace(location="host", purpose="preparation", bytes=native_bytes,
                          source=source),
            ),
        )
        from .native import construct_qhd

        return SelectedBlock.bind(
            record,
            payload=(r, self, problem.bounds, problem.variable_names),
            constructor=construct_qhd,
        )

    def save_archive(self, plan, files):
        """Write the Plan in the QHD archive format and return its manifest (``archive.save``)."""
        from .archive import save

        return save(self, plan, files)

    @classmethod
    def load_archive(cls, data, files):
        """Rebind a saved QHD Plan without replanning, or reject another format (``archive.load``)."""
        from .archive import load

        return load(data, files)

    def verify(self, plan, result, *, checks):
        """Run the explicitly requested checks on an analyzed result.

        ``NumberSectorOptions`` runs the shared one-excitation-per-register
        check on complete-register counts. ``QHDVerification`` runs the grid
        and fidelity comparisons of ``verification.verify``.
        """
        from nwqlib.evidence.sector import NumberSectorOptions, verify_number_sector
        from .verification import verify

        if type(checks) is NumberSectorOptions:
            result.validate_plan(plan)
            return verify_number_sector(result, options=checks)
        return verify(plan, result, checks=checks)

    def analyze(self, plan, data, *, settings):
        """Decode the observations into grid-point populations, the best observed candidate and the most probable point.

        Host-kernel scalars are read back by ``_host_summary``. A stored
        native state or measured bins are decoded and summarized by
        ``_summarize``, with invalid one-hot outcomes kept in the
        unconditional population as invalid mass. Counts are pooled as
        integers over all chunks, per decoded grid point and for the invalid
        outcomes, and divided by the total returned shots only after the
        most probable point is chosen from the integers (a count tie is
        equality of observed counts, not of the underlying probabilities).
        The Result keeps the pooled integer sum of the valid counts and the
        total returned shots as ``valid_count`` and ``returned_shots``.
        Exact probabilities are averaged over chunks. ``_readout_window``
        gives each path's tie window from its receipts or host application.
        """
        validate_selection(plan)
        observations, trace = data.observations, data.trace
        artifacts = data
        if type(plan) is not Plan or plan.method != self or trace.plan_id != plan.content_id:
            raise ValueError("analysis requires this exact QHD Plan and trace")
        ids = _admit_observations(plan, observations)
        events = {}
        for event in trace.events:
            if event.attempt in events:
                raise ValueError("QHD trace attempt IDs must be unique")
            events[event.attempt] = event
        observation = plan.resolve("qhd").resolved_observation(plan)[1]
        for chunk in observations.chunks:
            trace.validate_observation(chunk)
            event = events.get(chunk.attempt)
            if (
                chunk.run_id != trace.run_id
                or event is None
                or event.status != "completed"
                or event.prepared_id != chunk.prepared_id
                or event.observation_id != chunk.content_id
                or event.execution != chunk.execution
            ):
                raise ValueError("QHD contribution requires its unique completed execution attempt")
        # Host scalars, stored amplitudes and measured bins carry different
        # acquisition evidence but feed the same physical-grid summary.
        applications = ()
        artifact = None
        # Integer totals of a counts readout. Exact readout has none.
        totals = dict(valid_count=None, returned_shots=None)
        receipts = {receipt.content_id: receipt for receipt in artifacts.receipts}
        if plan.execution == "classical" and observations.chunks:
            chunk = observations.chunks[0]
            data = _host_summary(plan, chunk.values)
            applications = chunk.applications
            if plan.method.keep_state:
                artifact = chunk.artifacts[0]
        elif observation.kind == "amplitudes" and observations.chunks:
            from .decoding import decode_statevector_arrays

            if len(observations.chunks) != 1 or len(observations.chunks[0].artifacts) != 1:
                raise ValueError(
                    "stored QHD amplitude analysis requires one live selected state artifact"
                )
            chunk = observations.chunks[0]
            artifact = chunk.artifacts[0]
            declaration = observation.amplitudes
            if (
                chunk.execution != "quantum_circuit"
                or artifact.output != declaration.output
                or artifact.construction_id != plan.resolve("qhd").selected_construction(plan).content_id
            ):
                raise ValueError("QHD amplitude artifact differs from selected construction")
            state = artifacts.artifact(artifact).array
            grid, bits = _grid(plan), _bits(plan.method)
            # The valid amplitudes are gathered or permuted once into the (K,)*d grid, and the total and
            # invalid masses are direct fsum sums of their own register populations, streamed over the
            # register in bounded chunks rather than formed as one 2**width probability array.
            valid, invalid, total = decode_statevector_arrays(state, grid, bits)
            tie_window, unavailable = _readout_window(observations.chunks, receipts)
            data = _summarize(plan, valid, invalid, total, tie_window, unavailable)
        elif observation.kind == "counts":
            from nwqlib.execution import MAX_COUNT

            grid, bits = _grid(plan), _bits(plan.method)
            # Only the observed indices are decoded, and counts pool as exact integers (_pool_counts),
            # int64 only while the returned shots fit MAX_COUNT.
            shots = sum(c.returned_shots for c in observations.chunks)

            def histograms():
                for chunk in observations.chunks:
                    if chunk.execution != "quantum_circuit":
                        raise ValueError("native QHD requires quantum observation chunks")
                    # _admit_observations requires the selected counts observation.
                    yield chunk.histogram()

            unique, pooled, invalid, counted = _pool_counts(histograms(), grid, bits, exact=shots > MAX_COUNT)
            tie_window, unavailable = _readout_window(observations.chunks, receipts)
            data = _summarize(plan, pooled, invalid / (shots or 1), counted / (shots or 1),
                              tie_window, unavailable, shots=shots, points=unique)
            totals = dict(valid_count=sum(map(int, pooled.tolist())), returned_shots=shots)
            if any(c.returned_shots != observation.shots for c in observations.chunks):
                data["missing"] += ("fewer returned shots than requested",)
        else:
            import numpy as np
            from .decoding import decode_histogram

            grid, bits = _grid(plan), _bits(plan.method)
            # Average exact probabilities over chunks: each valid point adds its values v/C in
            # observation order, chunk by chunk, as sequential additions from zero. The total and the
            # invalid one-hot outcomes, kept in the unconditional population, are direct fsum sums of
            # their own values v/C.
            chunks = len(observations.chunks)
            decoded, terms, invalid_terms = [], [], []
            for chunk in observations.chunks:
                if chunk.execution != "quantum_circuit":
                    raise ValueError("native QHD requires quantum observation chunks")
                # _admit_observations requires the selected probabilities observation.
                histogram = chunk.histogram()
                valid, points = decode_histogram(histogram, grid, bits)
                values = histogram.weights / chunks
                terms.append(values)
                invalid_terms.append(values[~valid])
                decoded.append((points, values[valid]))
            total = fsum(v for values in terms for v in map(float, values.flat))
            invalid = fsum(v for values in invalid_terms for v in map(float, values.flat))
            points = (np.concatenate([p for p, _ in decoded]) if decoded
                      else np.zeros((0, grid.num_variables), dtype=np.int64))
            unique, inverse = np.unique(points, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            pooled = np.zeros(len(unique), dtype=np.float64)
            start = 0
            for rows, values in decoded:
                np.add.at(pooled, inverse[start:start + len(rows)], values)
                start += len(rows)
            # Exact simulator probabilities carry binary64 roundoff that grows
            # with the executed circuit. The chunk average stays within the
            # largest window of its chunks. The pooled bin values add
            # QHDAnalysis.pooled_summation_roundoff.
            window = max(
                (
                    NUMERICAL_RELATION_RTOL if receipt is None else receipt.probability_window
                    for receipt in (receipts.get(chunk.prepared_id) for chunk in observations.chunks)
                ),
                default=NUMERICAL_RELATION_RTOL,
            ) + QHDAnalysis.pooled_summation_roundoff(observations.chunks)
            tie_window, unavailable = _readout_window(observations.chunks, receipts)
            data = _summarize(plan, pooled, invalid, total, tie_window, unavailable, points=unique)
            if observation.kind == "probabilities" and abs(total - 1.0) > window:
                data["missing"] += ("incomplete probability mass",)
        if not observations.chunks:
            data["missing"] += ("no completed observation",)
        fields = dict(
            origin=capture_analysis_origin(
                analyzer=_ANALYSIS_SOURCE, method_id=plan.method.content_id, dependencies=("numpy",)
            ),
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=observations.content_id,
            contribution_ids=tuple(ids),
            applications=applications,
            artifact=artifact,
            **totals,
            **data,
        )
        result = QHDAnalysis(**fields)
        result.validate_plan(plan)
        # Masses decoded from a stored state carry no scalar for the chunk-receipt
        # check, so apply the same receipt window that the decoding of backend
        # output and save_result apply to chunks.
        result.validate_analysis_masses(artifacts)
        return result
