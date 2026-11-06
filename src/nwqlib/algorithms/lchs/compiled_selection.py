"""Circuit-free compiled LCHS parameter selection, with explicit SDK/numerical work."""

from __future__ import annotations

from nwqlib._limits import ResourceLimitError, byte_allowance, exceeds, memory_limit, optional_limit

from contextlib import contextmanager
from dataclasses import replace
from math import fsum
from typing import Any, Mapping
from nwqlib._preparation_laws import _uniform_superposition_gate_slots

import numpy as np

from nwqlib.subroutines.block_encoding import BlockEncodingPlan, plan_block_encoding
from nwqlib.subroutines.pauli_decomposition import PAULI_COEFFICIENT_RTOL, PauliDecomposition, PauliTerm


def _kept_pauli_part(
    part: np.ndarray,
    coefficients: np.ndarray,
    kept: np.ndarray,
    *,
    pruned_mass_bound: float,
    dense_normalization: float | None = None,
    max_bytes=None,
    max_select_work,
    charged: int,
    held_bytes: int = 0,
) -> tuple[BlockEncodingPlan, int]:
    """Select an encoding of the kept Pauli operator of one dense part and attach the explicit pruning error.

    ``coefficients`` holds the part's Pauli coefficients in transform-index
    order (operators._pauli._block_transform) and ``kept`` marks the words
    above the shared cutoff. The kept operator is the part itself when
    pruning removed no nonzero word. Otherwise its matrix is the inverse
    transform of the kept coefficients, and its dense normalization is its
    own spectral norm. With m kept words, the PREP pair of a Pauli encoding
    alone needs 2*(2**a - 2) CX for a = bit_length(m - 1), which exceeds
    the dense dilation when m > d**2/2. The dense choice is then settled
    unless the kept operator is circulant, so core._plan_block_encoding
    receives its matrix and m (``term_count``). It tests circulant
    structure on the matrix, records the ``auto`` choice from m and neither
    transforms the matrix again nor forms a Pauli term. Otherwise the kept
    words alone become the decomposition that core._plan_block_encoding
    classifies and compares. A dense choice then uses the part itself
    (``source_matrix``) when nothing was pruned, and reconstructs the kept
    operator from its terms when a word was pruned. The term route needs no
    dense-input allowance of 192*d**2 bytes, so a part that ends multiplexed
    or banded is planned beside L and H. ``held_bytes`` are the caller's
    other data in memory, each counted once, and ``charged`` is the work
    already drawn from ``max_select_work``.

    Work, as core._plan_block_encoding counts its own stages: on the matrix
    route, 32*q*d**2 for an inverse transform, 16*d**2 for the intake scan
    and 8*q*B for a circulant of B bands. On the term route, (2*q + 8)*m for
    the masks, labels and records of the m kept words, the classification
    and planning work of core._admit_pauli_plan and m*d**2 for a
    reconstruction. A dense choice that computes its spectral norm adds
    ``_dense_norm_work(d, False)``.

    The child's ``error_evaluation`` names the encoding's own bound and the
    added pruned mass.

    Returns:
        The plan and the work it adds to ``charged``.
    """
    from nwqlib._preparation_laws import direct_preparation_cx_bound
    from nwqlib.operators._pauli import PauliTerms, _index_masks, _inverse_block_transform
    from nwqlib.operators.access import _check_bytes
    from nwqlib.subroutines.block_encoding.core import (
        _admit_pauli_plan, _dense_dilation_predicted_cx, _dense_norm_work, _plan_block_encoding)

    d = part.shape[0]
    q = d.bit_length() - 1
    index = np.flatnonzero(kept)
    if not len(index):
        return _exact_zero_child(q), 0
    pruned = int(np.count_nonzero(coefficients)) != len(index)
    normalization = None if pruned else dense_normalization
    prep_pair_cx = 2 * direct_preparation_cx_bound(max(len(index) - 1, 0).bit_length(), complex_phases=False)
    if prep_pair_cx > _dense_dilation_predicted_cx(q):
        source, work = part, 0
        if pruned:
            work = 32*max(1, q)*d*d
            if exceeds(charged + work, max_select_work):
                raise ResourceLimitError(f"QSP generator classification exceeds max_select_work: it needs "
                                         f"{charged + work} work units, max_select_work={max_select_work}")
            # The inverse transform holds the masked coefficients and two
            # complex levels of d**2 entries.
            _check_bytes(held_bytes + 48*d*d, max_bytes, "kept Pauli operator reconstruction")
            source = _inverse_block_transform(np.where(kept, coefficients, 0), q)
        bytes_left, work_left = _bytes_left(max_bytes, held_bytes), _work_left(max_select_work, charged + work)
        with _in_caller_limits(max_select_work=max_select_work, charged=charged + work, max_bytes=max_bytes,
                               held_bytes=held_bytes):
            plan = _plan_block_encoding(source, implementation="auto", normalization=normalization,
                                        max_bytes=bytes_left, max_work=work_left, term_count=len(index))
        work += 16*d*d + (8*q*len(plan.source.offsets) if plan.implementation == "banded" else 0)
    else:
        packing = (2*q + 8)*len(index)
        if exceeds(charged + packing, max_select_work):
            raise ResourceLimitError(f"QSP generator classification exceeds max_select_work: it needs "
                                     f"{charged + packing} work units, max_select_work={max_select_work}")
        # The masks and their digit temporaries, (48 + 16*W)*m bytes for W
        # mask words, are at most 32*d**2 for m <= d**2/2 and fit in the
        # caller's dense reserve.
        x, z = _index_masks(index, q)
        bytes_left = _bytes_left(max_bytes, held_bytes)
        with _in_caller_limits(max_bytes=max_bytes, held_bytes=held_bytes):
            terms = tuple(PauliTerm(label=label, coefficient=value) for label, value in
                          PauliTerms._snapshot(q, x, z, coefficients[index]).labels(max_bytes=bytes_left))
        decomposition = PauliDecomposition(terms=terms, input_dimension=d, operator_dimension=d,
                                           num_qubits=q, atol=0.0)
        labels = len(terms)*(q+16)
        # The same classification and planning work that _plan_block_encoding
        # then checks itself, here with the label bytes held beside it.
        left = _work_left(max_select_work, charged + packing)
        with _in_caller_limits(max_select_work=max_select_work, charged=charged + packing):
            work = packing + _admit_pauli_plan(decomposition, held_bytes=held_bytes + labels, max_bytes=max_bytes,
                                               max_work=left, classify=True)
        with _in_caller_limits(max_select_work=max_select_work, charged=charged + packing, max_bytes=max_bytes,
                               held_bytes=held_bytes):
            plan = _plan_block_encoding(decomposition, implementation="auto", normalization=normalization,
                                        max_bytes=bytes_left, max_work=left, source_matrix=None if pruned else part)
        if plan.implementation == "dense_dilation" and pruned:
            work += len(terms)*d*d
    if plan.implementation == "dense_dilation" and normalization is None:
        work += _dense_norm_work(d, False)
    detail = dict(plan.detail)
    detail.pop("structured_input", None)
    bound = {"multiplexed_pauli": "exact block identity of the kept Pauli terms",
             "banded": "circulant detection residual of the kept operator"}.get(
                 plan.implementation, detail.get("error_evaluation"))
    detail["error_evaluation"] = f"{bound}, plus the pruned Pauli-coefficient mass"
    return replace(plan, error_bound=float(plan.error_bound + pruned_mass_bound), detail=detail), work


def _compiled_qsp_part_plans(
    l_part: np.ndarray,
    h_part: np.ndarray,
    *,
    l_norm: float | None = None,
    max_bytes=None,
    max_select_work,
) -> tuple[BlockEncodingPlan, BlockEncodingPlan, float, int]:
    """Classify both QSP children with one shared relative Pauli cutoff.

    Returns the L and H child plans, the pruned mass of an H child that
    pruning removed entirely (zero otherwise) and the work this stage drew
    from ``max_select_work`` (None: no limit).

    L and H are pruned against one cutoff relative to the largest coefficient
    magnitude among their combined Pauli terms, so both children see the same
    threshold. The discarded coefficient mass of
    each child is added to its operator-error bound. If every H term is
    pruned, that mass is returned separately, because the joint generator then
    has no H child to carry it.

    L and H are transformed once each into their Pauli coefficients
    (operators._pauli._block_transform), and the shared cutoff is applied to
    those arrays, so no Pauli term object exists for a pruned or a dense
    child (_kept_pauli_part). Planning first checks 128*d**2 bytes, held
    while each child is planned: L and H, both coefficient vectors and a
    64*d**2 reserve for completed dense child payloads and conversion
    copies. Before any child exists, the transform of H beside L, H and the
    coefficients of L uses 96*d**2 of them. Planning forms no SELECT
    table, and the block-encoding builders check the tables when a circuit
    is constructed. Each stage draws on one remaining work budget: 32*q*d**2
    for each transform, 32*d**2 for the shared pruning and part selection,
    and for each nonempty child the work that _kept_pauli_part returns.
    Later SELECT stages have separate remaining-work checks. An empty H child is
    not classified. These units are proxies for the limit checks, not timings or
    equal-cost CPU operations.
    """

    from nwqlib.operators._pauli import _block_transform
    from nwqlib.operators.access import _check_bytes
    d = l_part.shape[0]
    q = d.bit_length()-1
    # One work ledger across the transforms, pruning, both
    # classifications, conversions and norms. These bound selected arrays,
    # not native RSS, and these units are admission proxies, not timings or
    # equal-cost CPU operations.
    held = 128*d*d
    _check_bytes(held, max_bytes, "QSP generator Pauli transforms")
    cap = optional_limit(max_select_work, "max_select_work")
    used = 2*32*max(1, q)*d*d + 32*d*d
    if exceeds(used, cap):
        raise ResourceLimitError(f"QSP generator classification exceeds max_select_work: it needs {used} work "
                                 f"units, max_select_work={cap}")
    l_coefficients, h_coefficients = (np.zeros(d*d, dtype=complex) if coefficients is None else coefficients
                                      for coefficients in (_block_transform(l_part), _block_transform(h_part)))
    # The shared relative cutoff of prune_pauli_terms_relative over the words
    # of L and H together: a word is kept when its magnitude exceeds it.
    l_magnitudes, h_magnitudes = np.abs(l_coefficients), np.abs(h_coefficients)
    cutoff = PAULI_COEFFICIENT_RTOL*max(float(l_magnitudes.max()), float(h_magnitudes.max()))
    l_kept, h_kept = l_magnitudes > cutoff, h_magnitudes > cutoff
    # fsum reads only the pruned nonzero words, not every absent one.
    l_pruned_mass_bound = float(fsum(l_magnitudes[~l_kept & (l_magnitudes > 0)]))
    h_pruned_mass_bound = float(fsum(h_magnitudes[~h_kept & (h_magnitudes > 0)]))
    del l_magnitudes, h_magnitudes
    l_plan, l_work = _kept_pauli_part(
        l_part,
        l_coefficients,
        l_kept,
        pruned_mass_bound=l_pruned_mass_bound,
        dense_normalization=l_norm,
        max_bytes=max_bytes, max_select_work=cap, charged=used, held_bytes=held,
    )
    if l_plan.implementation == "exact_zero":
        raise ValueError(
            "hamiltonian_evolution_backend='qsp_block_encoding' requires a nonzero "
            "Hermitian part L after relative Pauli pruning; the input is effectively "
            "anti-Hermitian at the shared relative threshold"
        )
    used += l_work
    if exceeds(used, cap):
        raise ResourceLimitError(f"QSP generator classification exceeds max_select_work: it needs {used} work "
                                 f"units, max_select_work={cap}")
    h_plan, h_work = _kept_pauli_part(
        h_part,
        h_coefficients,
        h_kept,
        pruned_mass_bound=h_pruned_mass_bound,
        max_bytes=max_bytes, max_select_work=cap, charged=used, held_bytes=held,
    )
    used += h_work
    if exceeds(used, cap):
        raise ResourceLimitError(f"QSP generator classification exceeds max_select_work: it needs {used} work "
                                 f"units, max_select_work={cap}")
    fully_pruned_h_mass_bound = (
        h_pruned_mass_bound if h_plan.implementation == "exact_zero" else 0.0
    )
    return l_plan, h_plan, fully_pruned_h_mass_bound, used


def _exact_zero_child(num_qubits: int) -> BlockEncodingPlan:
    """Return the plan of an exactly zero child, which the joint generator omits."""
    return BlockEncodingPlan(requested_implementation="auto", implementation="exact_zero", alpha=0.0,
        num_ancillas=0, system_qubits=num_qubits, error_bound=0.0, source=None, decomposition=None,
        detail={"state": "exact_zero", "realization_status": "not_required"})


def _bytes_left(max_bytes, held_bytes):
    """Return the byte limit left beside ``held_bytes`` for a nested builder that counts only its own arrays.

    ``max_bytes=None`` is first resolved to the physical memory, so the
    builder receives the same remainder as under an explicit cap of that
    value. The result is None only when the platform reports no memory. A
    builder refuses a limit below 1 as an invalid option, and it holds at
    least one byte, so nothing left is a refusal at the caller's limit.
    """
    limit = memory_limit(max_bytes)
    if limit is None:
        return None
    if held_bytes >= limit:
        raise ResourceLimitError(f"this stage needs at least {held_bytes + 1} bytes with the {held_bytes} bytes held "
                                 f"beside it, exceeding {byte_allowance(max_bytes)}")
    return limit - held_bytes


def _work_left(max_select_work, charged):
    """Return what ``charged`` leaves of ``max_select_work`` for a nested builder that counts only its own work.

    ``None`` sets no limit. A builder refuses a limit below 1 as an invalid
    option, and it charges at least one unit, so nothing left is a refusal at
    the caller's limit.
    """
    if max_select_work is None:
        return None
    if charged >= max_select_work:
        raise ResourceLimitError(f"QSP generator classification exceeds max_select_work: it needs at least "
                                 f"{charged + 1} work units, max_select_work={max_select_work}")
    return max_select_work - charged


@contextmanager
def _in_caller_limits(*, max_select_work=None, charged=0, max_bytes=None, held_bytes=None):
    """Add to a nested builder's limit refusal what its limits are in the terms that the caller set.

    The builder knows only its own requirement and what is left, which it
    calls ``max_work`` and ``max_bytes``. The caller's limit must reach that
    requirement plus the amount already drawn, so the refusal adds that amount
    when it names such a remainder. ``held_bytes=None`` means that the builder
    received the caller's byte limit and counts the bytes held itself.
    """
    try:
        yield
    except ResourceLimitError as error:
        text = str(error).rstrip(".") + "."
        if max_select_work is not None and "max_work" in text:
            text += (f" Here max_work is what {charged} work units leave of max_select_work={max_select_work}, "
                     f"so add {charged} to a stated work requirement." if charged else
                     " Here max_work is max_select_work.")
        if held_bytes and "max_bytes" in text:
            text += (f" Here max_bytes is what {held_bytes} bytes held leave of {byte_allowance(max_bytes)}, "
                     f"so add {held_bytes} to a stated byte requirement.")
        raise ResourceLimitError(text) from error


def _compact_centered_child(part, num_qubits: int, *, max_bytes, max_select_work, charged, held_bytes):
    """Plan the encoding of the centered child X - c_X I of one compact Hermitian part X.

    A Pauli part drops its identity word and is encoded as the multiplexed
    Pauli SELECT of its other words, with normalization
    sum_{P != I} |c_P|. A band part drops its offset-0 band and uses the
    periodic shift LCU with normalization sum_{b != 0} |beta_b|. Both are
    exact encodings of the stored centered operator (error bound zero), and
    c_X enters the coefficient phases instead (``selection.LCHSGenerator``
    derives the factorization). The explicit
    Pauli request skips the dense-dilation comparison of ``auto``, so no
    matrix of the system dimension is formed. A supplied encoding has no
    joint-generator construction yet.

    Returns:
        The plan and the work it adds to ``charged``, the work already drawn
        from ``max_select_work``.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.operators.access import _check_bytes
    from nwqlib.subroutines.block_encoding.banded import BandSpecification
    from nwqlib.subroutines.block_encoding.core import _admit_pauli_plan
    from nwqlib.subroutines.pauli_decomposition import PauliTerm
    if part is None:
        return _exact_zero_child(num_qubits), 0
    representation = part.manifest.reference.representation
    if representation == "pauli":
        identity = "I"*num_qubits
        bytes_left = _bytes_left(max_bytes, held_bytes)
        with _in_caller_limits(max_bytes=max_bytes, held_bytes=held_bytes):
            terms = tuple(PauliTerm(label=label, coefficient=complex(value))
                          for label, value in part.pauli_terms().labels(max_bytes=bytes_left) if label != identity)
        if not terms:
            return _exact_zero_child(num_qubits), 0
        dimension = 1 << num_qubits
        decomposition = PauliDecomposition(terms=terms, input_dimension=dimension, operator_dimension=dimension,
                                           num_qubits=num_qubits, atol=0.0)
        # The m(q+16) bytes of the labels and coefficients are held here.
        # plan_block_encoding charges them itself, so it receives the limit
        # left after the other bytes the caller holds.
        labels = len(terms)*(num_qubits+16)
        left = _work_left(max_select_work, charged)
        with _in_caller_limits(max_select_work=max_select_work, charged=charged):
            work = _admit_pauli_plan(decomposition, held_bytes=held_bytes+labels, max_bytes=max_bytes,
                                     max_work=left, classify=False)
        with _in_caller_limits(max_select_work=max_select_work, charged=charged, max_bytes=max_bytes,
                               held_bytes=held_bytes):
            return plan_block_encoding(decomposition, implementation="multiplexed_pauli",
                                       max_bytes=bytes_left, max_work=left), work
    if representation == "band":
        bands = part.band()
        kept = [(offset, value) for offset, value in zip(bands.offsets, bands.coefficients, strict=True) if offset]
        if not kept:
            return _exact_zero_child(num_qubits), 0
        specification = BandSpecification(offsets=tuple(offset for offset, _ in kept),
                                          coefficients=tuple(value for _, value in kept), num_qubits=num_qubits)
        # The band planner forms no array (core._banded_plan), so the bytes
        # held beside it are charged here and it receives the limit itself.
        _check_bytes(held_bytes, max_bytes, "compact LCHS generator children")
        left = _work_left(max_select_work, charged)
        with _in_caller_limits(max_select_work=max_select_work, charged=charged):
            return plan_block_encoding(specification, max_bytes=max_bytes, max_work=left), 0
    raise ApplicabilityError("LCHS has no joint-generator construction yet for a supplied block encoding of L or H")


def _compiled_qsp_compact_part_plans(generator, *, unitary: bool, max_bytes=None,
                                     max_select_work):
    """Plan both joint-generator children of a compact generator from its operator inputs.

    The children are the centered parts L_c and H_c of
    ``selection.LCHSGenerator``. When L + s*I is exactly zero (``unitary``),
    H_c is the first and only child, as in ``_compiled_qsp_part_plans``.
    The work of both child plans draws on one ``max_select_work`` ledger,
    and their bytes are counted against ``max_bytes`` together: the
    completed first child keeps its m(q+16) bytes of terms or bands live
    while the second is planned. Nothing here grows with the system
    dimension, since the Pauli labels hold q characters per term. Planning
    forms no SELECT table, whose 64*P*(q+16) bytes for P padded addresses
    the block-encoding builders check when they form it.

    Returns:
        The two child plans, a zero pruned H mass (no coefficient is pruned)
        and the work they drew from ``max_select_work`` (None: no limit).
    """
    q = generator.num_qubits
    first, second = ((generator.hamiltonian, generator.dissipative) if unitary
                     else (generator.dissipative, generator.hamiltonian))
    cap = optional_limit(max_select_work, "max_select_work")
    used = held = 0
    plans = []
    for part in (first, second):
        plan, work = _compact_centered_child(part, q, max_bytes=max_bytes, max_select_work=cap, charged=used,
                                             held_bytes=held)
        used += work
        if exceeds(used, cap):
            raise ResourceLimitError(f"compact QSP generator classification exceeds max_select_work: it needs {used} "
                                     f"work units, max_select_work={cap}")
        if plan.implementation != "exact_zero":
            count = len(plan.decomposition.terms) if plan.decomposition is not None else len(plan.source.offsets)
            held += count*(q+16)
        plans.append(plan)
    return plans[0], plans[1], 0.0, used


# Census kinds group the gates of Qiskit's add_control basis (the names in
# EFFICIENTLY_CONTROLLED_GATES of qiskit/circuit/_add_control.py) by their
# price once controls are added. "cx" is CX. "h_or_x" is X, Y, Z or H.
# "rotation" is RY, RZ or RX. "phase" is P, which S and T become when a
# circuit is unrolled. "u" is the generic U gate. "global_phase" counts the
# scalar phases of the controlled circuits.
_CENSUS_KINDS = ("cx", "h_or_x", "rotation", "phase", "u", "global_phase")


def _controlled_kind_cx(controls: int) -> dict[str, int]:
    """CX of one gate of each gate kind after Qiskit adds ``controls`` controls to it at once.

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


def _direct_preparation_paths(num_qubits: int, *, complex_phases: bool) -> tuple[dict[str, int], ...]:
    """Gate counts by kind of the constructions of ``_build_normalized_state_preparation`` on n qubits.

    The builder takes one of them. X gates prepare a basis state and H gates
    equal amplitudes, n gates at most either way. Qiskit's
    UniformSuperpositionGate prepares a uniform prefix, with the elementary
    gate positions of ``_uniform_superposition_gate_slots``. The general tree has
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
    """Return the CX of gate counts by kind, given the CX of one gate of each kind.

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


def _controlled_state_preparation_cx(state: Any, controls: int) -> int | None:
    """CX of a state input's own preparation after Qiskit adds ``controls`` controls, or None when unknown.

    The constructions are those of ``problems.inputs.prepare_qiskit``. A
    vector uses the direct tree of ``_controlled_direct_preparation_cx``. An
    occupation applies one X per occupied qubit, priced at the controlled
    cost of the ``h_or_x`` kind, plus one controlled global phase. A product
    state applies one one-qubit direct preparation per qubit, each priced
    as a one-qubit direct tree with its controlled global phase, so the
    count is an upper bound. A supplied circuit has unknown gate content, so
    its controlled cost is unknown.
    """
    spec = state.preparation
    q = spec.basis.dimension.bit_length() - 1
    if spec.implementation == "qiskit.direct":
        return _controlled_direct_preparation_cx(q, controls)
    if spec.implementation == "qiskit.occupation":
        kind_cx = _controlled_kind_cx(controls)
        return int(np.count_nonzero(state._physical)) * kind_cx["h_or_x"] + kind_cx["global_phase"]
    if spec.implementation == "qiskit.product":
        return q * _controlled_direct_preparation_cx(1, controls)
    return None
