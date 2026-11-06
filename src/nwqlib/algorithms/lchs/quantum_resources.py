"""Formula resource accounting of an existing LCHS Plan, without building a circuit.

`resource_breakdown` lists each measurement batch's selected blocks with
their calls per shot and the per-call cost rules that their records
declare, and `nwqlib.estimate` totals the same rules over the workload.
`dense_representative_envelope` gives the structural gate bounds of one
dense SELECT branch, and `sampled_pair_resources` counts the rows of a
sampled-pair Plan.
"""

from __future__ import annotations

from math import prod

import numpy as np


def dense_representative_envelope(q, a, route="gatewise"):
    """Return value-independent counts in explicitly named gate bases.

    This computes a resource model. It builds no representative circuit and
    does not report the operations of an arbitrary transpiled circuit.

    Dense representative resource reporting uses the value-independent upper
    counts of the library synthesis and the chosen control route. Each record
    identifies its basis, route, construction assumptions and
    ``count_kind="structural_upper_bound"``. It is not a count of an observed
    circuit. Counts for any explicitly requested compilation still require
    inspection of that compiled circuit.

    For system width q >= 1 and address width a, the gate counts of the
    uncontrolled dense synthesis (C_q, U_q, Z_q, H_q) in the basis CX, U, RZ, H is
    (0, 1, 0, 0) at one qubit and, for q >= 2,
    C_q = (25*4**q - 72*2**q + 32)/48, U_q = 7*4**(q-2),
    Z_q = (3*4**q - 12*2**q)/8, H_q = (4**q - 16)/24
    (_dense_synthesis.dense_synthesis_gate_census). Whole-matrix control on
    m = q + a >= 3 qubits is two (m-1)-qubit syntheses and one multiplexed
    RZ (_dense_synthesis.controlled_synthesis_gate_census). Gatewise control
    has the CX bound C_branch = C_q*x(a+1) + H_q*x(a) + Z_q*z(a) + U_q*u(a)
    + p(a) of native.dense_branch_select_cx, with x the ancilla-free
    multi-controlled X count, z, y and u the MCRZ, MCRY and U costs and p
    the branch global phase, and the gate counts of the control constructor
    in _dense_synthesis.gatewise_control_counts. A gatewise U bound is not
    available and is reported as None, not zero. Open controls conjugate by
    at most 2*a X gates, which change no CX count. With no address bits the
    route is ``uncontrolled``. A one-system-qubit branch follows the
    gatewise count, as the builder does, and ``auto`` resolves through
    select_dense_control_route. The basis U bound of the whole-matrix and
    uncontrolled routes is the direct translation of U, RZ, H and the open
    control X gates to U, without routing or gate expansion.
    """
    from nwqlib.algorithms.lchs.native import dense_branch_select_cx
    from nwqlib.subroutines._dense_synthesis import (
        DENSE_CONTROL_ROUTES,
        dense_synthesis_gate_census,
        controlled_synthesis_gate_census,
        gatewise_control_counts,
        select_dense_control_route,
    )
    if type(q) is not int or q < 1 or type(a) is not int or a < 0:
        raise ValueError("dense representative widths must be q >= 1 and a >= 0")
    if route not in DENSE_CONTROL_ROUTES:
        raise ValueError("unknown dense control route")
    resolved = "uncontrolled" if a == 0 else select_dense_control_route(route, a)
    if q == 1 and a:
        resolved = "gatewise"
    cx = dense_branch_select_cx(q, a, route)
    result = dict(
        count_kind="structural_upper_bound",
        system_qubits=q,
        address_qubits=a,
        resolved_route=resolved,
        basis_cx_upper_bound=cx,
        open_control_x_upper_bound=2 * a,
        synthesis_gate_upper_bounds=None,
        control_constructor_upper_bounds=None,
        basis_u_upper_bound=None,
    )
    if resolved == "uncontrolled":
        census = dense_synthesis_gate_census(q)
    elif resolved == "whole_matrix":
        census = controlled_synthesis_gate_census(q + a)
    else:
        gates, instructions = gatewise_control_counts(q, a)
        result["synthesis_gate_upper_bounds"] = dense_synthesis_gate_census(q)
        result["control_constructor_upper_bounds"] = dict(gates=gates, instructions=instructions)
        return result
    result["synthesis_gate_upper_bounds"] = census
    # Direct translation of U/RZ/H/X to U, without routing or gate expansion.
    result["basis_u_upper_bound"] = (
        census["u"] + census["rz"] + census["h"] + 2 * a
    )
    return result


def resource_breakdown(plan, *, basis="cx"):
    """Return each measurement batch of a quantum LCHS Plan with its blocks, calls per shot and per-call cost rules.

    The Program is read once and no block is constructed. A body may hold
    sequences, block calls, repetitions with an integer count, and
    allocations, releases, measurements and resets, which call no block.
    Each batch gives its name, its ``settings`` count (its settings times
    the points of its range axes, so a sampled-pair Plan counts its B rows),
    its ``repetitions`` (the shots of each setting) and ``blocks``: for every selected block it
    calls, the signature name, ``calls`` per execution of the body,
    ``laws``, the record's `ResourceLaw` values in ``basis`` with the
    record's control and adjoint flags, empty when it declares none there,
    and ``decomposition``, the record's exact gate sequence or None, which
    `estimate` counts where no law applies.
    A vector or exact-reduction Plan has one entry for its root body, with
    ``repetitions`` None. `nwqlib.estimate(plan,
    context=ResourceContext(basis=basis))` gives the totals.

    Raises:
        ValueError: For a Plan that is not a quantum LCHS Plan, or a body
            node other than those above.
    """
    from nwqlib.core.planning import Plan
    from nwqlib.ir import Allocate, BlockCall, Measure, MeasurementBatch, Release, Repeat, Reset, Sequence
    from .method import LCHS
    if type(plan) is not Plan or type(plan.method) is not LCHS or plan.execution != "quantum":
        raise ValueError("resource_breakdown requires a quantum LCHS Plan")
    program = plan.construction.program
    nodes = {item.id: item.node for item in program.definitions}
    records = {record.signature.name: record for record in plan.construction.selections}

    def count(name, counts, factor):
        node = nodes[name]
        if isinstance(node, BlockCall):
            counts[node.signature] = counts.get(node.signature, 0) + factor
        elif isinstance(node, Sequence):
            for child in node.children:
                count(child, counts, factor)
        elif isinstance(node, Repeat) and type(node.count) is int:
            count(node.body, counts, factor * node.count)
        elif not isinstance(node, (Allocate, Release, Measure, Reset)):
            raise ValueError(f"resource_breakdown does not read the {type(node).__name__} node {name!r}")
        return counts

    def blocks(body):
        return tuple(dict(block=name, calls=calls, laws=tuple(
            law for law in records[name].resource_laws if law.basis == basis
            and law.controlled == records[name].controlled and law.adjoint == records[name].adjoint),
            decomposition=records[name].decomposition) for name, calls in count(body, {}, 1).items())
    batches = [(item.id, item.node) for item in program.definitions if isinstance(item.node, MeasurementBatch)]
    if not batches:
        return (dict(name=program.root, settings=1, repetitions=None, blocks=blocks(program.root)),)
    # A setting runs once per point of the batch's range axes, counted from
    # the axis bounds without enumerating the points.
    return tuple(dict(name=name, settings=len(node.settings) * prod(len(range(axis.start, axis.stop, axis.step))
                                                                    for axis in node.axes),
                      repetitions=node.repetitions, blocks=blocks(node.body))
                 for name, node in batches)


def sampled_pair_resources(plan):
    """Return the population, preparation and pair-SELECT CX counts of a sampled-pair Plan without building a circuit.

    ``rows`` B, ``bindings`` (one ``row`` value per point of the row axis), circuit
    ``preparations`` (one per row) and ``shots`` B*m are separate
    counts, and ``templates`` is the number of selected block definitions.
    ``pair_select_cx_per_row`` is the (smallest, largest) pair SELECT CX
    count of one row before routing (`sampled_plan.row_select_cx`): the
    multiplexed SELECT keeps its parity and multiplexor CX whatever its
    angles, and the structured SELECT omits the two CX of every rotation
    whose two angles are exactly equal in that row. ``fanout_cx`` adds
    2*(r_c - 1) per row, and ``pair_select_cx_total`` sums SELECT and
    fanout CX over every executed shot. For a Problem with a source, the
    pair block prepares each row's inputs: ``input_preparations`` counts
    the uncontrolled preparations of equal-kind rows and the controlled
    pairs of mixed-kind rows, per circuit, and their elementary gate counts
    are not derived (``input_preparation_cx`` is None). Without a source
    the initial preparation is its own block, counted by `estimate` from
    its cost law. Measurement, rotation counts and routing are not
    included, and no circuit depth is given.

    A product-formula recipe (``recipe`` names its version) is counted from
    its saved choices: ``row_steps`` is each row's step divisor (the
    Richardson level s or 2s, the MPF r run as r/2 double steps, the PTSC
    segment count r), ``pair_select_cx_total`` also adds the PTSC
    correction CX (``pair_correction_cx_per_row``, one CX per controlled
    Pauli and 2w per low-order rotation of support w), and ``member_rows``
    counts the rows of each Richardson or MPF member. For the multiplexed
    SELECT of a member recipe, ``expected_select_cx_per_shot`` is
    sum_j pi_j C_j over the member law, with C_j the member's SELECT CX; it
    is None for the structured SELECT, whose counts depend on each row's
    angles, and for PTSC, whose expected correction count is not derived.
    A sampled QSP Plan returns `sampled_qsp.qsp_pair_resources` instead.
    Nothing is drawn, selected, fitted or built.
    """
    from .sampled_plan import _row_words, row_correction_cx, row_select_cx
    rec, payload = plan.reconstruction, plan._native.get("native_data")
    rows, recipe = rec.pair_rows, rec.pair_recipe
    if rows is None:
        raise ValueError("pair resources need a sampled-pair LCHS Plan")
    if rec.pair_qsp is not None:
        from .sampled_qsp import qsp_pair_resources
        return qsp_pair_resources(plan)
    q, copies = payload.num_system_qubits, payload.control_copies
    select = row_select_cx(payload, rec)
    corrections = row_correction_cx(rec)
    row_cx = select + corrections + 2 * (copies - 1)
    preparations = None
    if payload.prepares_inputs:
        mixed = int(np.count_nonzero(rows.kind_x.array != rows.kind_y.array))
        preparations = dict(uncontrolled=rows.clusters - mixed, controlled_pairs=mixed)
    word = _row_words(payload, rec, corrections=False)
    row_steps = np.array([word(row)[1] for row in range(rows.clusters)], dtype=np.int64)
    member_rows = expected = None
    if recipe is not None and recipe.recipe != "ptsc":
        member_rows = tuple(int(np.count_nonzero(recipe.member.array == j)) for j in range(len(recipe.member_steps)))
        if payload.implementation == "multiplexor":
            # The multiplexed CX of one word is angle independent and equal
            # for both members' words, so any row gives it.
            one = int(select[0]) // word(0)[2]
            expected = sum(count / 2**52 * repetitions * one
                           for count, repetitions in zip(recipe.member_counts, recipe.member_repetitions))
    return dict(rows=rows.clusters, repeats=rows.repeats, bindings=rows.clusters, preparations=rows.clusters,
                shots=rows.clusters * rows.repeats, templates=len(plan.construction.selections),
                qubits=q + copies, clbits=q + 1, implementation=payload.implementation, control_copies=copies,
                fanout_cx=2 * (copies - 1), fanout_layers=2 * (copies - 1).bit_length(),
                recipe=None if recipe is None else recipe.version,
                row_steps=(int(row_steps.min()), int(row_steps.max())),
                pair_select_cx_per_row=(int(select.min()), int(select.max())),
                pair_correction_cx_per_row=None if recipe is None or recipe.recipe != "ptsc" else
                (int(corrections.min()), int(corrections.max())),
                member_rows=member_rows, expected_select_cx_per_shot=expected,
                pair_select_cx_total=int(row_cx.sum()) * rows.repeats,
                input_preparations=preparations, input_preparation_cx=None)
