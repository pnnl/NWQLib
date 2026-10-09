"""Circuit-free compiled LCHS parameter selection, with explicit SDK/numerical work."""

from __future__ import annotations

from nwqlib._limits import DEFAULT_MAX_BYTES

from dataclasses import replace
from math import fsum
from typing import Any, Mapping
from nwqlib._preparation_laws import _uniform_superposition_gate_slots

import numpy as np

from nwqlib.subroutines.block_encoding import BlockEncodingPlan, plan_block_encoding
from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, decompose_matrix_to_pauli, prune_pauli_terms_relative
from nwqlib.subroutines.qsp.evolution import _control_diagonal_generator_layout, compiled_select_resource_law, jacobi_anger_expansion


def _pad_diagonal(values: np.ndarray, padded_length: int) -> np.ndarray:
    """Zero-pad a physical branch diagonal onto the padded control register.

    Zero entries make padded control states evolve as the identity under
    ``exp(-i T' S')``, matching the identity padding of the dense SELECT.
    """

    padded = np.zeros(padded_length, dtype=float)
    padded[: values.size] = np.asarray(values, dtype=float)
    return padded


def _qsp_outer_bytes(q: int, children) -> int:
    """Return the outer QSP planning bytes for children with the given kept term counts.

    With q system qubits and ``d = 2**q``, each nonempty child of m terms has
    ``P = 2**ceil(log2 m)`` table addresses and ``b = min(m, d)`` candidate
    bands. The integer-width allowance is
    ``J = 4208 + 2q + 2 ceil(log2(b+1)) + ceil(log2(m+b+1))``, with
    ``4208 = 2*(1074+1024) + 12`` from the binary64 squared-mass width and
    twelve extra bits of allowance explained in
    ``nwqlib.subroutines.block_encoding.core._admit_pauli_plan``.
    With ``V = m+b+q+1``, the table allowance is ``T = 64P(q+16)`` and
    the active classifier allowance is ``X = [128 L(2J) + 512]V + H0``,
    with L = ``integer_object_bytes`` and H0 = ``BOOKKEEPING_BYTES``.
    The outer QSP gate adds both live child tables, the shared decomposition
    payloads and matrix buffers to the larger sequential classifier
    workspace:

        B_outer = max{B_dec(q, d), 96 d**2 + sum_c m_c (q+16) + sum_c T_c + max_c X_c}.

    The 96 d**2 are the L/H matrices (32 d**2) and a known-array reserve of
    64 d**2 for two completed dense child payloads and the current
    conversion/norm copies. The labels the classifiers parse are borrowed
    from the two decompositions, so no further m(q+16) is added. An empty
    child adds no classifier or table. For two full-support children
    (m = d**2, b = d) this is the closed sufficient law
    B_dec-or-96d**2 + 2m(q+16) + 2T + X: 42,168,320, 161,732,608,
    637,289,984 and 2,528,890,880 bytes at q = 4 to 7. It excludes norm
    workspace of the vendor library, Python headers and later native
    synthesis.
    """
    from nwqlib.algorithms.lchs.time_independent_terms import _pauli_decomposition_requirements
    from nwqlib.evidence._work import BOOKKEEPING_BYTES, integer_object_bytes

    d = 1 << q
    _, decomposition_bytes = _pauli_decomposition_requirements(d)
    payload = tables = classifier = 0
    for terms in children:
        payload += terms*(q+16)
        if terms:
            table = 1 << (terms-1).bit_length()
            bands = min(terms, d)
            width = 4208 + 2*q + 2*bands.bit_length() + (terms+bands).bit_length()
            tables += 64*table*(q+16)
            classifier = max(classifier, (128*integer_object_bytes(2*width)+512)*(terms+bands+q+1)
                             + BOOKKEEPING_BYTES)
    return max(decomposition_bytes, 96*d*d + payload + tables + classifier)


def _kept_pauli_part(
    decomposition: PauliDecomposition,
    kept_term_ids: set[int],
    *,
    pruned_mass_bound: float,
    dense_normalization: float | None = None,
    max_bytes=DEFAULT_MAX_BYTES,
    max_work=1_000_000_000,
    held_bytes: int = 0,
) -> tuple[BlockEncodingPlan, int]:
    """Select an encoding of the kept Pauli operator and attach the explicit pruning error.

    ``held_bytes`` are the caller's other live payloads, each counted once,
    including the decomposition this child borrows. The child's tables and
    classifier are admitted by ``_admit_pauli_plan`` against ``max_work``
    before ``plan_block_encoding`` runs with the same work allowance and the
    byte allowance left beside the other held payloads. A Pauli input
    selected for dense dilation adds its reconstruction, m*d**2, and a
    computed spectral norm, ``_dense_norm_work(d, False)``.

    Returns:
        The plan and the work it consumed from ``max_work``.
    """
    from nwqlib.subroutines.block_encoding.core import _admit_pauli_plan, _dense_norm_work

    kept = tuple(term for term in decomposition.terms if id(term) in kept_term_ids)
    if not kept:
        return BlockEncodingPlan(
            requested_implementation="auto",
            implementation="exact_zero",
            alpha=0.0,
            num_ancillas=0,
            system_qubits=decomposition.num_qubits,
            error_bound=0.0,
            source=None,
            decomposition=None,
            detail={"state": "exact_zero", "realization_status": "not_required"},
        ), 0
    scale = max(abs(complex(term.coefficient)) for term in kept)
    kept_decomposition = replace(decomposition, terms=kept)
    normalization = dense_normalization if pruned_mass_bound == 0.0 else None
    work = _admit_pauli_plan(kept_decomposition, held_bytes=held_bytes, max_bytes=max_bytes,
                             max_work=max_work, classify=True)
    # plan_block_encoding charges the borrowed kept terms again as its own
    # held payload, so that population is returned to its byte allowance.
    borrowed = len(kept)*(decomposition.num_qubits+16)
    plan = plan_block_encoding(
        kept_decomposition,
        normalization=normalization,
        max_bytes=max_bytes-held_bytes+borrowed, max_work=max_work,
    )
    if plan.implementation == "dense_dilation":
        d = decomposition.operator_dimension
        work += len(kept)*d*d + (_dense_norm_work(d, False) if normalization is None else 0)
    detail = dict(plan.detail)
    if "normalization_records" in detail:
        detail["normalization_records"] = {
            name: float(value) / float(scale)
            for name, value in detail["normalization_records"].items()
        }
        detail["error_evaluation"] = "algebraic Pauli-coefficient pruning bound; no dense residual"
        detail.pop("structured_input", None)
    return replace(
        plan,
        error_bound=float(plan.error_bound + pruned_mass_bound),
        decomposition=kept_decomposition,
        detail=detail,
    ), work


def _compiled_qsp_part_plans(
    l_part: np.ndarray,
    h_part: np.ndarray,
    *,
    l_norm: float | None = None,
    max_bytes=DEFAULT_MAX_BYTES,
    max_select_work=1_000_000_000,
) -> tuple[BlockEncodingPlan, BlockEncodingPlan, float, int]:
    """Classify both QSP children with one shared relative Pauli cutoff.

    Returns the L and H child plans, the pruned mass of an H child that
    pruning removed entirely (zero otherwise) and the max_select_work that
    this gate's ledger leaves for the later SELECT stages.

    L and H are pruned against one cutoff relative to the largest coefficient
    magnitude among their combined Pauli terms, so both children see the same
    threshold. The discarded coefficient mass of
    each child is added to its operator-error bound. If every H term is
    pruned, that mass is returned separately, because the joint generator then
    has no H child to carry it.

    The outer QSP gate adds both live child tables, the shared decomposition
    payloads and matrix buffers to the larger sequential classifier
    workspace (_qsp_outer_bytes, checked at the full envelope of d**2 terms
    per child before decomposition). Each selected stage consumes one
    remaining work budget, including any dense reconstruction and
    singular-value normalization: the two decompositions
    (time_independent_terms._pauli_decomposition_requirements), 32*d**2 for
    the shared pruning and part selection, and for each nonempty child the
    table and circulant-classification work that
    block_encoding.core._admit_pauli_plan returns (32*P*q**2 plus
    C(q, m, b) with b = min(m, d) before detection), m*d**2 for a Pauli
    child reconstructed as a dense matrix and _dense_norm_work(d, False)
    for a norm it computes. For two full-support children these add to
    654,924, 5,777,014, 60,033,772 and 735,289,238 units at q = 4 to 7, so
    these preprocessing stages admit all four under the default
    max_select_work=1_000_000_000 and max_bytes=10_000_000_000. Later
    SELECT stages have separate remaining-work checks. An empty H child is
    not classified. These units are admission proxies, not timings or
    equal-cost CPU operations.
    """

    from nwqlib._validation import integer
    from nwqlib.operators.access import _check_bytes
    from nwqlib.algorithms.lchs.time_independent_terms import _pauli_decomposition_requirements
    d = l_part.shape[0]
    q = d.bit_length()-1
    # One remaining-work ledger across decomposition, pruning, both
    # classifications, table planning, conversions and norms. The byte check
    # is the closed outer law at the full envelope of d**2 terms per child
    # (_qsp_outer_bytes), before the actual counts are known. These bound
    # selected arrays, not native RSS, and these units are admission
    # proxies, not timings or equal-cost CPU operations.
    _check_bytes(_qsp_outer_bytes(q, (d*d, d*d)), max_bytes, "QSP generator classification")
    remaining = integer(max_select_work, "max_select_work", 1)
    decomposition_work, _ = _pauli_decomposition_requirements(d)
    # Shared pruning and part selection over up to d**2 terms per child, 32
    # visits per term pair of L and H.
    pruning_work = 32*d*d
    if decomposition_work+pruning_work > remaining:
        raise ValueError("QSP generator classification exceeds max_select_work")
    remaining -= decomposition_work+pruning_work
    l_decomposition = decompose_matrix_to_pauli(l_part, atol=0.0, rtol=0.0)
    h_decomposition = decompose_matrix_to_pauli(h_part, atol=0.0, rtol=0.0)
    kept, _pruned_mass_bound = prune_pauli_terms_relative(
        (*l_decomposition.terms, *h_decomposition.terms)
    )
    kept_ids = {id(term) for term in kept}
    l_pruned_mass_bound = float(
        fsum(
            abs(complex(term.coefficient))
            for term in l_decomposition.terms
            if id(term) not in kept_ids
        )
    )
    h_pruned_mass_bound = float(
        fsum(
            abs(complex(term.coefficient))
            for term in h_decomposition.terms
            if id(term) not in kept_ids
        )
    )
    # Live beside each classification: L and H, the 64 d**2 reserve for
    # completed dense child payloads and conversion copies, and both
    # decompositions, which the kept children borrow.
    held = 96*d*d + (len(l_decomposition.terms)+len(h_decomposition.terms))*(q+16)
    l_plan, l_work = _kept_pauli_part(
        l_decomposition,
        kept_ids,
        pruned_mass_bound=l_pruned_mass_bound,
        dense_normalization=l_norm,
        max_bytes=max_bytes, max_work=remaining, held_bytes=held,
    )
    if l_plan.implementation == "exact_zero":
        raise ValueError(
            "hamiltonian_evolution_backend='qsp_block_encoding' requires a nonzero "
            "Hermitian part L after relative Pauli pruning; the input is effectively "
            "anti-Hermitian at the shared relative threshold"
        )
    if l_work > remaining:
        raise ValueError("QSP generator classification exceeds max_select_work")
    remaining -= l_work
    # The completed L child's table stays live while H is classified.
    held += 64*(1 << (len(l_plan.decomposition.terms)-1).bit_length())*(q+16)
    h_plan, h_work = _kept_pauli_part(
        h_decomposition,
        kept_ids,
        pruned_mass_bound=h_pruned_mass_bound,
        max_bytes=max_bytes, max_work=remaining, held_bytes=held,
    )
    if h_work > remaining:
        raise ValueError("QSP generator classification exceeds max_select_work")
    remaining -= h_work
    fully_pruned_h_mass_bound = (
        h_pruned_mass_bound if h_plan.implementation == "exact_zero" else 0.0
    )
    return l_plan, h_plan, fully_pruned_h_mass_bound, remaining


def _compiled_select_qsp_plan(
    *,
    part_plans: tuple[BlockEncodingPlan, BlockEncodingPlan, float],
    l_diagonal: np.ndarray,
    h_diagonal: np.ndarray,
    padded_length: int,
    evolution_time: float,
    epsilon_he: float,
    max_bytes=DEFAULT_MAX_BYTES,
    max_degree=256,
) -> dict[str, Any]:
    """Plan the selected joint-generator QSP construction without realization.

    With D_L = diag(k_j) and D_H = I, the joint generator is the effective
    Hamiltonian sum_j |j><j| (x) (k_j*L + H) of Pocrnic et al.,
    arXiv:2506.20760v2, Sec. IV, Eqs. (61)-(62), built here from separate L
    and H encodings rather than one encoding of A. Evolving it gives the
    whole SELECT at once (their Eqs. (63)-(65)).

    The joint generator D_L (x) L + D_H (x) H has subnormalization
    alpha = max|D_L|*alpha_L + max|D_H|*alpha_H, so QSP simulates
    exp(-i*tau*x) at tau = alpha*T. The Jacobi-Anger degrees are chosen once
    for epsilon_he and stored with the structural gate law.
    """

    from nwqlib.operators.access import _check_bytes
    # Envelope of eight float64 values per padded slot for the two padded
    # diagonals, their input arrays and the layout's per-slot tables.
    _check_bytes(64*padded_length, max_bytes, "QSP generator branch diagonals")
    l_plan, h_plan, _ = part_plans
    h_is_zero = h_plan.implementation == "exact_zero"
    d_l = _pad_diagonal(np.asarray(l_diagonal, dtype=float), padded_length)
    d_h = None if h_is_zero else _pad_diagonal(np.asarray(h_diagonal, dtype=float), padded_length)
    layout = _control_diagonal_generator_layout(
        l_plan, None if h_is_zero else h_plan, l_diagonal=d_l, h_diagonal=d_h
    )
    # min_degree=2 keeps the even (cos) polynomial nonconstant, which the
    # phase solver requires (see jacobi_anger_expansion).
    expansion = jacobi_anger_expansion(layout["alpha"] * evolution_time, epsilon_he,
        min_degree=2,max_degree=max_degree,max_bytes=max_bytes)
    return {
        "part_plans": part_plans,
        "l_diagonal": d_l,
        "h_diagonal": d_h,
        "layout": layout,
        "expansion": expansion,
        "evolution_time": evolution_time,
        "epsilon_he": epsilon_he,
        "structural_law": compiled_select_resource_law(
            control_qubits=layout["control_qubits"],
            cos_degree=expansion.cos_degree,
            sin_degree=expansion.sin_degree,
            child_count=layout["child_count"],
        ),
    }



# Census kinds group the gates of Qiskit's add_control basis (the names in
# EFFICIENTLY_CONTROLLED_GATES of qiskit/circuit/_add_control.py) by their
# price once controls are added. "cx" is CX. "h_or_x" is X, Y, Z or H.
# "rotation" is RY, RZ or RX. "phase" is P, which S and T become when a
# circuit is unrolled. "u" is the generic U gate. "global_phase" counts the
# scalar phases of the controlled circuits.
_CENSUS_KINDS = ("cx", "h_or_x", "rotation", "phase", "u", "global_phase")

# Each row counts by census kind the gates that add_control emits for the
# costliest gate of a kind under one control, after Qiskit unrolls them to
# its basis. A CX becomes a Toffoli (6 CX, 7 P from T and Tdg, 2 H), an H
# becomes S, H, T, CX, Tdg, H and Sdg, an RY or a U becomes a CU (2 CX, 3 P,
# 2 U), a P becomes a CP (2 CX, 3 P), and a global phase becomes a P on the
# control. An RZ becomes a CRZ (2 CX, 2 RZ), which costs less than an RY
# once controlled again. test_twice_controlled_kind_costs_match_qiskit
# recomputes these rows.
_ONE_CONTROL_UNROLLED_CENSUS = {
    "cx": {"cx": 6, "phase": 7, "h_or_x": 2},
    "h_or_x": {"cx": 1, "phase": 4, "h_or_x": 2},
    "rotation": {"cx": 2, "phase": 3, "u": 2},
    "phase": {"cx": 2, "phase": 3},
    "u": {"cx": 2, "phase": 3, "u": 2},
    "global_phase": {"phase": 1},
}

# CX count of the projector flip of a QSP pass, an X gate with k open
# controls on the block ancillas (evolution._append_projector_phase), after
# Qiskit adds the parity control of the pass, stored at index k - 1. The
# parity control makes add_control unroll the flip through its definition,
# not through the synthesis that MCX_CX_BY_CONTROLS records, and
# control each gate once more. Each CX of the definition then costs a
# Toffoli of 6 CX, each X or H 1 CX and each other one-qubit gate 2 CX.
# From k = 5 the definition is Qiskit 2.5.2's synth_mcx_noaux_v24, with
# _mcphase_cx(k + 1) CX and P, H and RZ counts without a published closed
# form, so the table stores the values. It covers 32 block ancillas. A
# Pauli child on q system qubits has at most 2q address qubits, so a
# generator exceeds the table only from 16 system qubits, where the outer
# classification gate of _compiled_qsp_part_plans already refuses at the
# default max_bytes: its 96*d**2 matrix reserve alone is 412,316,860,416
# bytes (_qsp_outer_bytes).
# test_qsp_projector_flip_table_matches_installed_qiskit recomputes every
# entry. Revisit when that test fails after a Qiskit upgrade or when the
# projector phase changes its construction.
LCHS_QSP_PROJECTOR_FLIP_CX = (
    8, 56, 122, 336, 746, 1220, 1918, 2840,
    3898, 5092, 6422, 7888, 9490, 11228, 13102, 15112,
    17258, 19540, 21958, 24512, 27202, 30028, 32990, 36088,
    39322, 42692, 46198, 49840, 53618, 57532, 61582, 65768,
)


def _controlled_kind_cx(controls: int) -> dict[str, int]:
    """CX of one gate of each census kind after Qiskit adds ``controls`` controls to it at once.

    Qiskit's add_control unrolls a gate to its basis and replaces every
    basis gate by a multi-controlled gate (``apply_basic_controlled_gate`` in
    ``qiskit/circuit/_add_control.py``). The prices are Qiskit's counts
    without ancillas. Transpiled builds with idle qubits, clean or already
    used, never exceeded them
    (``test_controlled_census_prices_bound_qiskit_for_every_kind``). With
    c = ``controls``:

    - cx: an X with c + 1 controls (``MCX_CX_BY_CONTROLS``).
    - h_or_x: an X with c controls between uncontrolled one-qubit gates.
    - rotation: an RZ becomes the MCRZ of ``native._controlled_gate_cx``.
      RY and RX use QuantumCircuit.mcry and mcrx in "noancilla" mode. For
      c <= 3 that is a Gray-code circuit of ``2**c - 1`` CU and
      ``2**c - 2`` CX gates, ``3*2**c - 4`` CX, and from c = 4 on it has
      the MCRZ count. The price is the larger of the two.
    - phase: MCPhase on c + 1 qubits (``_mcphase_cx``). An SX or SXdg
      becomes the same MCPhase between two uncontrolled H gates.
    - u: the generic U of ``native._controlled_gate_cx``.
    - global_phase: a phase gate on c - 1 controls, free for c = 1.

    The key rz adds the MCRZ price alone, for a caller that knows a
    rotation is an RZ. With no controls a CX costs one CX and the other
    kinds none.
    """
    if controls == 0:
        return dict.fromkeys((*_CENSUS_KINDS, "rz"), 0) | {"cx": 1}
    from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import _mcphase_cx
    from .native import _controlled_gate_cx
    cost = _controlled_gate_cx(controls)
    gray_code_ry = 3 * (1 << controls) - 4 if controls <= 3 else cost["rz"]
    return {"cx": cost["cx"], "h_or_x": cost["h"], "rotation": max(gray_code_ry, cost["rz"]),
            "phase": _mcphase_cx(controls + 1), "u": cost["u"], "global_phase": cost["global_phase"],
            "rz": cost["rz"]}


def _twice_controlled_kind_cx() -> dict[str, int]:
    """CX of one gate of each census kind inside a gate that Qiskit controls once, then once more.

    The compiled QSP SELECT controls each generator branch on the combine
    qubit, and the QSP pass that contains the generator on the parity qubit.
    The second add_control unrolls the first one's output and controls each
    of its gates, so a gate costs the one-control price of its one-control
    census. For example a CX costs 6*6 + 7*2 + 2*1 = 52 CX, where two
    controls at once cost 14.
    """
    once = _controlled_kind_cx(1)
    return {kind: _census_cx(census, once) for kind, census in _ONE_CONTROL_UNROLLED_CENSUS.items()}


def _direct_preparation_paths(num_qubits: int, *, complex_phases: bool) -> tuple[dict[str, int], ...]:
    """Gate censuses of the constructions of ``_build_normalized_state_preparation`` on n qubits.

    The builder takes one of them. X gates prepare a basis state and H gates
    equal amplitudes, n gates at most either way. Qiskit's
    UniformSuperpositionGate prepares a uniform prefix, with the elementary
    slots of ``_uniform_superposition_gate_slots``. The general tree has
    ``2**n - 1`` RY and ``2**n - 2`` CX, and a complex state adds a phase
    diagonal with as many RZ and CX (``direct_preparation_cx_bound``).
    """
    from nwqlib._preparation_laws import direct_preparation_cx_bound
    uniform = _uniform_superposition_gate_slots(num_qubits)
    return (
        {"h_or_x": num_qubits},
        {"cx": uniform.get("cx", 0), "rotation": uniform.get("ry", 0), "phase": uniform.get("p", 0),
         "h_or_x": uniform.get("h", 0) + uniform.get("x", 0)},
        {"cx": direct_preparation_cx_bound(num_qubits, complex_phases=complex_phases),
         "rotation": (1 + complex_phases) * ((1 << num_qubits) - 1)},
    )


def _census_cx(census: Mapping[str, int], kind_cx: Mapping[str, int]) -> int:
    """Price a gate census with the CX of one gate of each kind.

    ``preparation_qubits`` stands for a positive-amplitude direct PREP on
    that many qubits and its inverse, priced at the costliest of the
    builder's constructions.
    """
    total = sum(census.get(kind, 0) * kind_cx[kind] for kind in _CENSUS_KINDS)
    if "preparation_qubits" in census:
        paths = _direct_preparation_paths(census["preparation_qubits"], complex_phases=False)
        total += 2 * max(_census_cx(path, kind_cx) for path in paths)
    return total


def _controlled_direct_preparation_cx(num_qubits: int, controls: int) -> int:
    """CX of the direct preparation of a complex n-qubit state after Qiskit adds ``controls`` controls.

    It prices the costliest construction of the builder and one controlled
    global phase.
    """
    kind_cx = _controlled_kind_cx(controls)
    paths = _direct_preparation_paths(num_qubits, complex_phases=True)
    return kind_cx["global_phase"] + max(_census_cx(path, kind_cx) for path in paths)


def _qsp_child_gate_census(child: Any) -> dict[str, int]:
    """Upper bounds on the gates of one QSP child encoding by census kind, without realizing it.

    The counts are those of the child circuit after Qiskit unrolls it to its
    add_control basis, which is how the SELECT controls it.
    """

    n, a = child.system_qubits, child.num_ancillas
    detail = child.to_dict()
    table = 1 << a
    # Positive PREP and its inverse on the a address qubits.
    census = {"cx": 0, "preparation_qubits": a}
    if child.implementation == "banded":
        counts = detail.get("periodic_gate_counts", detail)
        cp_count = n * (n - 1)
        diagonal = int(counts["control_diagonal_gates"])
        census["cx"] += 2 * cp_count + 6 * (n // 2) + n * (table if a else 0)
        census["cx"] += diagonal * max(0, table - 2)
        census.update(
            rotation=n * table + diagonal * (table - 1),
            phase=3 * cp_count,
            h_or_x=2 * n,
            global_phase=1,
        )
        return census
    if child.implementation in ("pauli_lcu", "multiplexed_pauli"):
        # A system qubit whose Pauli table depends on r address bits gets
        # Qiskit's UCGate with 2**r one-qubit unitaries, each one U gate, and
        # 2**r - 1 CX. Its completion diagonal on r + 1 qubits adds
        # 2**(r+1) - 1 RZ and 2**(r+1) - 2 CX. For r = 0 the factor is one
        # U gate without a diagonal.
        rotation = u = 0
        for r in detail["effective_control_counts"]:
            local_table = 1 << r
            census["cx"] += 3 * (local_table - 1)
            u += local_table
            rotation += 2 * local_table - 1 if r else 0
        diagonal = int(detail["coefficient_diagonal_cx"] > 0)
        # At one address qubit a nonzero diagonal has zero CX but still has an RZ.
        if child.decomposition is not None:
            diagonal = int(any(np.angle(term.coefficient) != 0.0 for term in child.decomposition.terms))
        census["cx"] += diagonal * max(0, table - 2)
        census.update(rotation=rotation + diagonal * (table - 1), u=u, global_phase=1)
        return census
    if child.implementation == "dense_dilation":
        # The SELECT controls the exact synthesis of the dilation unitary on
        # m = n + 1 qubits (qiskit_compat.controlled), whose gate bounds are
        # the census of _dense_synthesis.dense_synthesis_gate_census.
        from nwqlib.subroutines._dense_synthesis import dense_synthesis_gate_census
        gates = dense_synthesis_gate_census(n + a)
        return {"cx": gates["cx"], "u": gates["u"], "rotation": gates["rz"], "h_or_x": gates["h"],
                "global_phase": 1}
    raise ValueError(f"no controlled construction census for {child.implementation!r}")


def _whole_matrix_children(qsp_plan, route: str) -> bool:
    """Whether the combine qubit controls the dense children of the compiled QSP SELECT on the whole-matrix route.

    Only a generator with two active children has a combine qubit, which
    adds one control (``_dense_synthesis.select_dense_control_route``).
    """
    from nwqlib.subroutines._dense_synthesis import select_dense_control_route
    layout = qsp_plan["layout"]
    active = sum(1 for key in ("l_diagonal_max_abs", "h_diagonal_max_abs") if layout[key] > 0.0)
    return active == 2 and select_dense_control_route(route, 1) == "whole_matrix"


def _controlled_dense_child_census(num_qubits: int) -> dict[str, int]:
    """Census kinds of the whole-matrix control of a dense child on ``num_qubits`` qubits by the combine qubit.

    The gates are those of ``_dense_synthesis.controlled_synthesis_gate_census``
    on the child's qubits and the control, with the global phase of that
    circuit and one phase gate on the control for a global phase of the
    branch (``qiskit_compat._controlled_whole_matrix``).
    """
    from nwqlib.subroutines._dense_synthesis import controlled_synthesis_gate_census
    gates = controlled_synthesis_gate_census(num_qubits + 1)
    return {"cx": gates["cx"], "u": gates["u"], "rotation": gates["rz"], "h_or_x": gates["h"], "global_phase": 1,
            "phase": 1}


def compiled_select_controlled_syntheses(qsp_plan, *, route: str = "gatewise") -> tuple[int, ...]:
    """Return the qubit count, control included, of each whole-matrix synthesis that the compiled QSP SELECT makes.

    On the whole-matrix route the combine qubit of a two-child generator
    controls each ``dense_dilation`` child through
    ``_dense_synthesis.controlled_unitary_circuit`` on its n system qubits,
    its ancilla and the control (``qiskit_compat.controlled``).
    """
    if not _whole_matrix_children(qsp_plan, route):
        return ()
    l_part, h_part, _ = qsp_plan["part_plans"]
    return tuple(child.system_qubits + child.num_ancillas + 1 for child in (l_part, h_part)
                 if child.implementation == "dense_dilation")


def compiled_select_dense_syntheses(qsp_plan, *, route: str = "gatewise") -> tuple[int, ...]:
    """Return the qubit count of each exact dense synthesis that the compiled QSP SELECT makes with ``dense_unitary_circuit``.

    Only a ``dense_dilation`` child holds a dense unitary, one
    ``UnitaryGate`` on its n system qubits and one ancilla.
    ``qiskit_compat.controlled`` replaces it by
    ``_dense_synthesis.dense_unitary_circuit`` wherever the construction
    first controls it. With two active children the combine qubit
    controls each branch (``evolution._combine_generator_children``), so
    each dense child is synthesized once, and the parity control of the
    QSP passes then finds no dense unitary left. With one active child the
    generator is not controlled, and each pass's parity control
    synthesizes the forward and the adjoint dilation it contains. A pass
    of degree d holds the forward query when d >= 1 and the adjoint when
    d >= 2, and the cosine and sine passes are controlled separately.
    Each entry of the result is one synthesis. When ``route`` selects the
    whole-matrix route for the combine qubit's one control, a two-child
    generator makes the syntheses of :func:`compiled_select_controlled_syntheses`
    instead, and this function returns an empty tuple.
    """
    if _whole_matrix_children(qsp_plan, route):
        return ()
    l_part, h_part, _ = qsp_plan["part_plans"]
    layout = qsp_plan["layout"]
    active = [child for child, scale in ((l_part, layout["l_diagonal_max_abs"]),
                                         (h_part, layout["h_diagonal_max_abs"])) if scale > 0.0]
    expansion = qsp_plan["expansion"]
    per_child = 1 if len(active) == 2 else min(2, expansion.cos_degree) + min(2, expansion.sin_degree)
    return tuple(child.system_qubits + child.num_ancillas
                 for child in active if child.implementation == "dense_dilation"
                 for _ in range(per_child))


def compiled_select_dense_controls(qsp_plan, *, route: str = "gatewise", queries=None) -> tuple[tuple[int, int, int], ...]:
    """Return (gates, instructions, heavy) of each Qiskit control call that unrolls a dense child of the compiled QSP SELECT.

    The triples are those of ``_dense_synthesis.gatewise_control_counts``
    for the dilation unitary on n + 1 qubits, and ``admit_dense_syntheses``
    takes them as its ``controls``. The expansion's cosine and sine passes
    have ``cos_degree`` and ``sin_degree`` generator queries, and the parity
    control of a pass (``evolution.build_qsp_evolution_encoding``) unrolls
    the child at every query.

    - With one active child the generator is not controlled, so each pass is
      one call that controls the synthesized child once per query.
    - With two active children the combine qubit first controls each dense
      child once. The parity control of each pass then unrolls the output of
      that control at every query and controls it again
      (``_twice_controlled_dense_counts``). On the whole-matrix route
      (:func:`compiled_select_controlled_syntheses`) the combine qubit
      makes no gate-wise control call, and the parity control unrolls the
      whole-matrix circuit of each dense child at every query
      (``_dense_synthesis.controlled_synthesis_gate_census`` on n + 2
      qubits, ``census_control_counts``).

    Other gates of the passes are controlled too and lie outside these
    counts. ``queries`` replaces the two pass degrees by other query counts,
    one control call each, as representative sampling does with ``(1,)``
    for its single controlled query.
    """
    from nwqlib.subroutines._dense_synthesis import (census_control_counts, controlled_synthesis_gate_census,
                                                     gatewise_control_counts)
    l_part, h_part, _ = qsp_plan["part_plans"]
    layout = qsp_plan["layout"]
    active = [child for child, scale in ((l_part, layout["l_diagonal_max_abs"]),
                                         (h_part, layout["h_diagonal_max_abs"])) if scale > 0.0]
    widths = [child.system_qubits + child.num_ancillas for child in active if child.implementation == "dense_dilation"]
    if not widths:
        return ()
    expansion = qsp_plan["expansion"]
    degrees = (expansion.cos_degree, expansion.sin_degree) if queries is None else tuple(queries)
    if _whole_matrix_children(qsp_plan, route):
        unrolled = [sum(counts) for counts in zip(*(census_control_counts(controlled_synthesis_gate_census(width + 1), 1)
                                                    for width in widths))]
        return tuple(tuple(degree * count for count in unrolled) for degree in degrees)
    if len(active) == 1:
        once = gatewise_control_counts(widths[0], 1)
        return tuple(tuple(degree * count for count in once) for degree in degrees)
    twice = [sum(counts) for counts in zip(*(_twice_controlled_dense_counts(width) for width in widths))]
    return (tuple(gatewise_control_counts(width, 1) for width in widths)
            + tuple(tuple(degree * count for count in twice) for degree in degrees))


def _twice_controlled_dense_counts(num_qubits: int) -> tuple[int, int, int]:
    """Return (gates, instructions, heavy) of a second control of one dense synthesis that a first control unrolled with one control.

    The first control turns each gate of ``dense_synthesis_gate_census`` into
    the one-control gates that ``_ONE_CONTROL_UNROLLED_CENSUS`` lists after
    Qiskit unrolls them to its basis, with an RZ priced as the costliest
    rotation. The second control, with one control, unrolls those gates and
    emits one instruction for each, except seven for an H (S, H, T, CX,
    Tdg, H and Sdg). Each P, U and rotation among them becomes a CP, CU or
    controlled rotation that holds its angles, the heavy instructions of
    ``gatewise_control_size``. The global phase adds one heavy gate and
    instruction, and the two X gates around an open control of the first
    step add two light ones.
    """
    from nwqlib.subroutines._dense_synthesis import dense_synthesis_gate_census
    census = dense_synthesis_gate_census(num_qubits)
    gates = instructions = 3
    heavy = 1
    for name, kind in (("cx", "cx"), ("u", "u"), ("rz", "rotation"), ("h", "h_or_x")):
        row = _ONE_CONTROL_UNROLLED_CENSUS[kind]
        gates += census[name] * sum(row.values())
        instructions += census[name] * (sum(row.values()) + 6 * row.get("h_or_x", 0))
        heavy += census[name] * (row.get("phase", 0) + row.get("u", 0) + row.get("rotation", 0))
    return gates, instructions, heavy


def compiled_select_cx_projection(qsp_plan, *, route: str = "gatewise"):
    """Upper bound on the CX of the compiled QSP SELECT leaf, as Qiskit 2.5.2 builds it.

    The leaf (native.construct_select) is the coefficient phase diagonal on
    the k address qubits, ``2**k - 2`` CX, followed by the QSP evolution of
    evolution.build_qsp_evolution_encoding. That evolution is three half
    steps joined by two reflections about the zero state of its N = b + 3
    ancillas, where b is the joint generator's block ancilla count. A half
    step controls its cosine pass and its sine pass on the parity qubit.
    Between them the two passes hold ``block_encoding_queries / 3``
    generator queries. A pass of degree d has d queries, d + 1 projector
    phases, one before the first query and one after each query, and two H
    gates on the pair qubit, and the odd pass also has a Z there. The
    parity control prices every gate of a pass at its one-control cost,
    except inside a two-child generator.

    - A query of a one-child generator costs the child and the RY
      multiplexor of its diagonal under the parity control alone.
    - In a two-child generator the combine qubit controls each branch, the
      child and its multiplexor, and the parity qubit then controls the
      result again, priced by ``_twice_controlled_kind_cx``. Two RY on the
      combine qubit and the X pair of the L branch's open control add their
      one-control cost. When ``route`` selects the whole-matrix route for
      the combine qubit's one control, a dense child is instead the
      ``controlled_unitary_circuit`` of its dilation on n + 2 qubits
      (``_controlled_dense_child_census``), which the parity control prices
      at one control, and its multiplexor stays twice controlled.
    - A projector phase costs two flips (``LCHS_QSP_PROJECTOR_FLIP_CX``),
      two CX from the pair qubit and one RZ.
    - A reflection, outside the passes, costs an X with N - 1 controls
      (``MCX_CX_BY_CONTROLS``).

    Coefficient PREP and a source input preparation are priced elsewhere.
    The value is recorded as an estimate, because another Qiskit version
    can control or lower these gates differently.

    Returns:
        A dict with select_cx, the per-query generator_cx, projector_cx,
        reflection_cx, fixed_cx for the pass boundaries, reflections and
        coefficient diagonal, the query count, the ancilla count N and the
        child censuses.
    """
    from nwqlib.subroutines._mcx_counts import MCX_CX_BY_CONTROLS
    l_part, h_part, _ = qsp_plan["part_plans"]
    layout = qsp_plan["layout"]
    child_plans = [
        child
        for child, scale in (
            (l_part, layout["l_diagonal_max_abs"]),
            (h_part, layout["h_diagonal_max_abs"]),
        )
        if scale > 0.0
    ]
    k = layout["control_qubits"]
    child_censuses = [_qsp_child_gate_census(child) for child in child_plans]
    # The RY multiplexor of one branch's rescaled diagonal on k address qubits.
    diagonal_census = {"cx": (1 << k) if k else 0, "rotation": 1 << k}
    once = _controlled_kind_cx(1)
    if layout["child_count"] == 1:
        generator_cx = _census_cx(child_censuses[0], once) + _census_cx(diagonal_census, once)
    else:
        twice = _twice_controlled_kind_cx()
        whole = _whole_matrix_children(qsp_plan, route)
        generator_cx = 0
        for child, census in zip(child_plans, child_censuses, strict=True):
            if whole and child.implementation == "dense_dilation":
                generator_cx += (_census_cx(_controlled_dense_child_census(child.system_qubits + child.num_ancillas),
                                            once) + _census_cx(diagonal_census, twice))
            else:
                generator_cx += _census_cx({kind: census.get(kind, 0) + diagonal_census.get(kind, 0)
                                            for kind in {*census, *diagonal_census}}, twice)
        generator_cx += _census_cx({"rotation": 2, "h_or_x": 2}, once)
    law = qsp_plan["structural_law"]
    queries = law["block_encoding_queries"]
    block_ancillas = layout["num_ancillas"]
    ancillas = block_ancillas + 3
    if block_ancillas > len(LCHS_QSP_PROJECTOR_FLIP_CX) or ancillas - 1 > len(MCX_CX_BY_CONTROLS):
        raise ValueError("the compiled QSP SELECT CX law covers at most "
                         f"{len(LCHS_QSP_PROJECTOR_FLIP_CX)} block ancillas")
    projector_cx = 2 * LCHS_QSP_PROJECTOR_FLIP_CX[block_ancillas - 1] + _census_cx({"cx": 2, "rotation": 1}, once)
    reflection_cx = MCX_CX_BY_CONTROLS[ancillas - 2]
    fixed_cx = ((law["pair_boundary_h_count"] + law["odd_pair_sign_count"]) * once["h_or_x"]
                + law["oaa_reflection_count"] * reflection_cx
                + max(0, (1 << k) - 2))
    return dict(select_cx=queries*generator_cx+law["projector_phase_count"]*projector_cx+fixed_cx,
        generator_cx=generator_cx, projector_cx=projector_cx, reflection_cx=reflection_cx,
        fixed_cx=fixed_cx, queries=queries, ancillas=ancillas, child_censuses=child_censuses)
