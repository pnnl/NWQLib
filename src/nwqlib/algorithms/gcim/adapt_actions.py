"""Packed classical ADAPT Pauli actions and their shared admission law.

A classical ADAPT Plan packs the nonidentity Hamiltonian rows, each processed
generator's ordered rows and the unit-coefficient insertion rows once, and
every classical Pauli action of its queries and of explicit verification uses
those tables. Every action is admitted by the shared single-column law of
``operators._pauli.pauli_action_requirements``.

Shared action law. Let ``d = 2**q``, L the stored row count of the acting
table including zeros and duplicates, ``g = min(L, d)`` and
``t = min(d, 1024)``. For an already-complex128 single state and a packed
table the installed owner gives

    B_A = 33d + 24L + 16g + 8 + max(9L, 96t),
    W_A = 2dL + dg + 2d + 4L + g.

The 33d term includes input, output and the finite-mask reserve. Group
metadata and rotated coefficients remain live during action. The 9L
preprocessing frontier and 96t retry tile share a maximum. The work includes
group metadata, the three L-row preprocessing passes, the attempted grouped
action and a full possible ordered retry. Supplying ``out`` does not remove
the output slot. The empty-table allowance is conservatively 2d work. Every
Pauli action owner calls ``pauli_action_requirements(d, L, min(L, d),
complex_input=True)``; no ``d*L`` substitution is made, no successful grouped
path is inferred from a finite answer and the retry allowance is not
subtracted.

Packing. Let L0 be the nonidentity Hamiltonian rows and T_j the row count of
generator j. The new packed-row population is

    R = L0 + 2 sum_j T_j,  B_permanent = 32R,  B_construction <= 64R,
    W_packing = R(q + 4).

There are eight-byte X and Z masks and one complex128 coefficient per row.
Source plus immutable snapshot gives 64R. Packing visits q label letters and
one coefficient, then copies the three arrays into the snapshot, giving
``q + 4`` pass units per row. The extra ``sum T_j`` population is the
insertion table population. Labels already held by the input and Python object
headers are separate from these logical payloads.

The tables live in ``AdaptInputs.cache["action_tables"]``, alive with that
Plan's bound inputs. They contain no Run, receipt, vector, sampled value or
controller decision, and they are derived again when an archive reconstructs
the Plan. A different processed coefficient table, row order, width or
generator input needs a different Plan/input cache. Quantum execution builds
no classical generator-action tables.
"""

from __future__ import annotations

import numpy as np

from nwqlib.operators._pauli import PauliTerms, apply_terms, pauli_action_requirements
from nwqlib.operators.access import _check_bytes, _check_products
from nwqlib.subroutines.fermionic_pool import GENERATOR_TAYLOR_DEGREE, _taylor_steps


def pack_action_rows(rows, q):
    """Pack admitted ordered rows once, without coalescing or dropping any row.

    This single-word full-state owner requires ``0 <= q < 63``. The permanent
    table has 32L bytes and the source-plus-snapshot frontier has 64L bytes.
    ``build_action_tables`` charges ``L(q + 1) + 3L`` pass units per table
    before calling, including the snapshot copy. Row order, duplicate labels,
    zero coefficients and complex phases stay, so an action on this table is
    the action on the same rows given as labels.
    """
    if not 0 <= q < 63:
        raise ValueError("ADAPT full-state action requires 0 <= q < 63")
    x = np.zeros((len(rows), 1), np.uint64)
    z = np.zeros_like(x)
    c = np.empty(len(rows), np.complex128)
    for j, (label, value) in enumerate(rows):
        if len(label) != q or any(p not in "IXYZ" for p in label):
            raise ValueError("Pauli label differs from action width")
        xx = zz = 0
        # Qubit zero is the rightmost letter.
        for bit, p in enumerate(reversed(label)):
            if p in "XY":
                xx |= 1 << bit
            if p in "YZ":
                zz |= 1 << bit
        x[j, 0] = xx
        z[j, 0] = zz
        c[j] = value
    return PauliTerms._snapshot(q, x, z, c)


def action_requirements(d, terms):
    """Return the shared packed ``(B_A, W_A)`` of one action on a complex128 state.

    ``pauli_action_requirements(d, L, min(L, d), complex_input=True)``: see
    the module docstring for ``B_A`` and ``W_A``.
    """
    return pauli_action_requirements(d, terms, min(terms, d), complex_input=True)


def packed_exponential(table, state, theta):
    """Apply ``exp(theta * A)`` with the Plan's packed table of A.

    This is ``fermionic_pool.apply_generator_exponential`` on the packed
    table: the same ``_taylor_steps`` count, the same degree-18 Horner order,
    coefficient order, scaling and addition, so it introduces no numerical
    approximation beyond that kernel's. Each step starts from ``r = state``
    and evaluates ``r <- state + (step / k) * A r`` for ``k = 18, ..., 1``.
    """
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    steps = _taylor_steps(table.coefficients, theta)
    if not steps:
        return vector.copy()
    step = float(theta) / steps
    for _ in range(steps):
        result = np.empty_like(vector)
        work = np.empty_like(vector)
        result[:] = vector
        for degree in range(GENERATOR_TAYLOR_DEGREE, 0, -1):
            apply_terms(table, result, num_qubits=table.num_qubits, out=work)
            work *= step / degree
            work += vector
            result, work = work, result
        vector = result
    return vector


def build_action_tables(inputs, q, *, max_bytes, max_products, other_live=0):
    """Build the Plan-owned packed tables before classical queries, with one packing charge.

    R counts the H0 rows, the generator rows and their unit-coefficient
    insertion rows. Packing charges ``R(q + 4)`` pass units against
    ``max_products``. The permanent logical payload is 32R. Reserving 64R
    beside ``other_live`` covers all snapshots and one working population.
    The immutable processed full Hamiltonian already owns its packed table,
    and a dense or sparse Hamiltonian has no H0 table. This private cache
    contains no state vector, Run, receipt or query result. Returns the dict
    ``h0``, ``generators``, ``insertions``, ``resident_bytes`` (32R) and
    ``packing_work`` (``R(q + 4)``).
    """
    if "action_tables" in inputs.cache:
        return inputs.cache["action_tables"]
    pauli = inputs.hamiltonian.reference.representation == "pauli"
    h0 = inputs.nonidentity if pauli else ()
    rows = len(h0) + 2 * sum(len(g.pauli_terms) for g in inputs.pool)
    work = rows * (q + 4)
    _check_products(work, max_products)
    _check_bytes(other_live + 64 * rows, max_bytes, "ADAPT packed action tables")
    answer = dict(
        h0=pack_action_rows(h0, q) if pauli else None,
        generators=tuple(pack_action_rows(g.pauli_terms, q) for g in inputs.pool),
        insertions=tuple(
            tuple(pack_action_rows(((label, 1.0),), q) for label, _ in g.pauli_terms)
            for g in inputs.pool
        ),
        resident_bytes=32 * rows,
        packing_work=work,
    )
    inputs.cache["action_tables"] = answer
    return answer


def action_sizes(inputs, d):
    """Return the fresh action byte allowances of a classical Plan's actions.

    These are the full Hamiltonian's ``OperatorInput.matvec`` allowance, the
    packed H0 action for Pauli input, each generator's packed action and the
    one-row insertion action. The classical frontier adds the largest
    ``B_A - 32d``, because the input and output vectors of one action are
    already in the frontier's vector slots.
    """
    from nwqlib.operators.inputs import _matvec_requirements

    sizes = [_matvec_requirements(inputs.hamiltonian)[1]]
    if inputs.hamiltonian.reference.representation == "pauli":
        sizes.append(action_requirements(d, len(inputs.nonidentity))[0])
    sizes.extend(action_requirements(d, len(g.pauli_terms))[0] for g in inputs.pool)
    sizes.append(action_requirements(d, 1)[0])
    return tuple(sizes)


def restored_record_allowance(inputs, rec):
    """Reapply the classical reconstruction's logical record law on load.

    This is the ``other_live`` that ``adapt_inputs.prepare_adapt_inputs``
    passes to ``build_action_tables``: the generator records, the Pauli
    identity bytes of every stored row and four copies of the
    reconstruction's fixed JSON shell.
    """
    from nwqlib.operators.inputs import _pauli_identity_bytes
    from .adapt_inputs import _reconstruction_fixed_json

    fixed = 3 * sum(
        2048
        + 128 * len(g.fermion_terms)
        + 16 * sum(len(ops) for _, ops in g.fermion_terms)
        + 16 * (len(g.spatial_indices) + len(g.spin_orbital_indices or ()))
        for g in inputs.pool
    )
    rows = sum(len(g.pauli_terms) for g in inputs.pool) + len(rec.terms)
    omitted = {"schema_version", "parent_id", "pool", "terms", "groups", "energy_groups"}
    fields = {key: getattr(rec, key) for key in type(rec).model_fields if key not in omitted}
    shell = _reconstruction_fixed_json(fields, pool_size=len(inputs.pool), term_count=len(rec.terms))
    return fixed + rows * _pauli_identity_bytes(rec.num_qubits) + 4 * shell


def query_work(plan, experiment, program):
    """Price one complete classical query before invocation.

    The sum covers the reference preparation, the selected prefix/reference
    chain exponentials (``adapt_records._exponential_action_work``) and the
    query's own actions. ``W_H0`` is the packed nonidentity action for Pauli
    input, and ``W_H + offset_free_action_work`` for a dense or sparse
    matrix, whose conservative sparse setup is charged on every query. With
    ``W_Aj`` the shared action of generator j and ``W_A(1)`` the one-row
    insertion action, the additions to the chain exponentials are

        pair: W_H0 + 2d,
        screen: W_H0 + sum_(unselected j) (W_Aj + d),
        energy with insertion: W_A(1) + 3d + W_H + d,
        energy without insertion: the complete W_EG
            (adapt_records._energy_gradient_work) instead of the chain
            exponentials and the ordinary energy action,
        residual: W_H + (4 max_selections + 13)d, plus one more exponential
            per entry of the right chain for the single-generator states.

    The insertion's ``W_A(1) + 3d`` is its one-row action, the scaling, the
    sum and the division by sqrt(2). Packing is absent: it has its
    once-per-Plan charge.
    """
    from nwqlib.ir.expressions import number
    from .adapt_records import _energy_gradient_work, _exponential_action_work, chain

    r = plan.reconstruction
    d = 1 << r.num_qubits
    inputs = plan._native["inputs"]
    values = {b.parameter: b.value for b in program.bindings}
    right = chain(values, "r")
    work = r.reference_action_work or 0
    full_h = r.hamiltonian_action_work
    offset_h = (
        action_requirements(d, len(inputs.nonidentity))[1]
        if inputs.hamiltonian.reference.representation == "pauli"
        else full_h + r.offset_free_action_work
    )
    insertion = int(number(values["insert_position"])) >= 0
    if experiment.name == "energy" and not insertion:
        return work + _energy_gradient_work(
            tuple(r.pool[i] for i, _ in right), tuple(theta for _, theta in right), d, full_h
        )[0]
    for side in ("l", "r"):
        work += sum(_exponential_action_work(r.pool[i], theta, d) for i, theta in chain(values, side))
    if insertion:
        work += action_requirements(d, 1)[1] + 3 * d
    if experiment.name == "screen":
        removed = {i for i, _ in right}
        work += offset_h + sum(
            action_requirements(d, len(m.terms))[1] + d for m in r.pool if m.pool_index not in removed
        )
    elif experiment.name == "pair":
        work += offset_h + 2 * d
    elif experiment.name == "energy":
        work += full_h + d
    else:
        work += full_h + d * (4 * r.max_selections + 13)
        work += sum(_exponential_action_work(r.pool[i], theta, d) for i, theta in right)
    return work
