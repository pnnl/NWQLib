"""Admitted constant-source branch grids, without state PREP or evolution."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from dataclasses import dataclass

import numpy as np

from nwqlib.operators.inputs import _freeze_array
from .solution_error_budget import psd_recovery_exponent, psd_recovery_part


@dataclass(frozen=True, kw_only=True)
class SourceBranchLayout:
    """Frozen coefficient/time grid and the physical applications it represents.

    Attributes:
        coefficients: Complex slot coefficients, weight*exp(shift*elapsed)*c_j,
            zero on padding slots. When exp(shift*T) exceeds binary64 every
            slot is divided by 2**psd_recovery_exponent(shift, T), which the
            recovery then carries.
        k_values: k-node of each slot, zero on padding.
        elapsed_times: Evolution time of each slot, T for the initial state and
            T - s_q for Duhamel node s_q.
        branch_to_node: Physical branch index of each slot, or None for padding.
        branch_kinds: "initial" or "source" for each physical branch, which
            selects the input direction that branch prepares.
        applications: (start_time, elapsed_time, |weight|, slots) per physical
            application, where weight is the initial norm or w_q*||b||.
        source_nodes: Gauss-Legendre Duhamel times s_q on [0, T].
        source_weights: Matching Gauss-Legendre weights w_q.
        initial_norm: Physical norm of the initial vector.
        source_norm: Physical norm of the constant source.
        kind_control: Address bit that selects initial or source preparation
            when both are present in a padded layout, otherwise None.
        compact: Whether slots list physical branches consecutively without
            power-of-two field padding (branch-controlled SELECT).
    """

    coefficients: object
    k_values: object
    elapsed_times: object
    branch_to_node: tuple[int | None, ...]
    branch_kinds: tuple[str, ...]
    applications: tuple[tuple[float, float, float, tuple[int, ...]], ...]
    source_nodes: tuple[float, ...]
    source_weights: tuple[float, ...]
    initial_norm: float
    source_norm: float
    kind_control: int | None
    compact: bool

    @property
    def branch_count(self):
        return len(self.branch_kinds)


def select_source_branches(*, coefficients, k_nodes, initial_norm, source_norm, final_time,
                           psd_shift, duhamel_nodes, compact=False,
                           max_bytes=DEFAULT_MAX_BYTES, max_quadrature_work=100_000_000):
    """Reuse one Duhamel quadrature and distinguish physical branches from padding.

    The constant-source solution is exp(-A*T)u0 + integral over s of
    exp(-A*(T-s))b (ACL arXiv:2312.03916v2, Eq. (2)). Discretizing k as in
    Eq. (61) and s with one Gauss-Legendre panel of duhamel_nodes points
    gives Eq. (72) with
    T/h2 = 1 and coefficients c'_q = w_q*||b||. Every k-node is repeated for
    the initial application and for each Duhamel node. The node count is the
    caller's choice rather than the Lemma 13 prescription, and its remainder
    is bounded separately (inhomogeneous_theory).

    In the padded layout the low address bits index the k-node, the next bits
    the Duhamel node, and one top bit (kind_control) chooses the initial or
    source input when both are present. Each field is padded to a power of
    two, so the input choice is one address bit and one controlled
    preparation per input kind serves every branch. The compact layout, used
    with branch-controlled SELECT, instead prepares the input inside each
    branch.
    """
    from .inhomogeneous_theory import duhamel_quadrature

    has_initial, has_source = initial_norm > 0, source_norm > 0
    if not has_initial and not has_source:
        raise ValueError("constant-source circuit requires a nonzero initial or source input")
    m, s = len(k_nodes), duhamel_nodes if has_source else 0
    node_bits, source_bits = max(m-1, 0).bit_length(), max(s-1, 0).bit_length()
    physical = m*(int(has_initial)+s)
    size = physical if compact else 1 << (node_bits+source_bits+int(has_initial and has_source))
    from nwqlib._validation import integer
    from nwqlib.operators.access import _check_bytes
    # Bytes: the s-node Legendre rule (32*s**2 for its Jacobi eigensolve and
    # 32*s for nodes and weights), and 128 per slot for the complex
    # coefficient, k, time and branch-map entries. Work: s**3 for the
    # eigensolve, 8*s for mapping the nodes, and 12 per slot.
    _check_bytes(32*s**2+32*s+128*size, max_bytes, "constant-source quadrature and branch grid")
    work = s**3+8*s+12*size
    cap = integer(max_quadrature_work, "max_quadrature_work", 1)
    if work > cap:
        raise ValueError(
            f"constant-source branch selection exceeds max_quadrature_work: it needs {work} work units "
            f"(s**3+8*s+12*slots with s={s}, slots={size}), LCHS.max_quadrature_work={cap}. "
            f"Raise LCHS.max_quadrature_work to at least {work}.")
    nodes, weights = duhamel_quadrature(final_time, s) if has_source else ((), ())
    selected_coefficients = np.zeros(size, dtype=np.complex128)
    selected_k, times = np.zeros(size), np.zeros(size)
    branch_to_node = [None]*size
    kinds, applications = [], []
    specifications = ([("initial", 0., final_time, initial_norm)] if has_initial else [])
    if has_source:
        specifications += [("source", float(node), final_time-float(node), float(weight)*source_norm)
                           for node, weight in zip(nodes, weights, strict=True)]
    kind_control = node_bits+source_bits if has_initial and has_source and not compact else None
    exponent = psd_recovery_exponent(psd_shift, final_time)
    next_slot = 0
    for application, (kind, start, elapsed, weight) in enumerate(specifications):
        if compact:
            base = next_slot
        elif kind == "initial":
            base = 0
        else:
            source_index = application-int(has_initial)
            base = (0 if kind_control is None else 1 << kind_control) | (source_index << node_bits)
        scale = weight*psd_recovery_part(psd_shift, elapsed, exponent)
        slots = tuple(range(base, base+m))
        for slot, coefficient, k in zip(slots, coefficients, k_nodes, strict=True):
            selected_coefficients[slot] = scale*coefficient
            selected_k[slot], times[slot] = k, elapsed
            branch_to_node[slot] = len(kinds)
            kinds.append(kind)
        applications.append((start, elapsed, abs(weight), slots))
        next_slot += m
    return SourceBranchLayout(coefficients=_freeze_array(selected_coefficients), k_values=_freeze_array(selected_k),
        elapsed_times=_freeze_array(times), branch_to_node=tuple(branch_to_node), branch_kinds=tuple(kinds),
        applications=tuple(applications), source_nodes=tuple(map(float, nodes)), source_weights=tuple(map(float, weights)),
        initial_norm=initial_norm, source_norm=source_norm, kind_control=kind_control, compact=compact)
