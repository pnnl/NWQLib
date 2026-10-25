"""Candidate frontier, canonical CX count and exact candidate score of ADAPT's bounded Ritz growth.

``ADAPT(basis_growth=BoundedRitzGrowth(...))`` keeps an ordered basis of
generator chains. A chain ``((i_1, t_1), ..., (i_k, t_k))`` denotes
``exp(t_k A_{i_k}) ... exp(t_1 A_{i_1})|ref>``, applying its first item
first, and the reference is the empty chain. Each round extends one basis
chain, the parent, by one generator at ADAPT's fixed angle ``theta``, scores
at most ``candidates_per_round`` such extensions from their projected
matrix columns and appends the best one. The functions here are pure: they
read chains, integers and small projected matrices and never touch a Run.
``frontier`` chooses the extensions of one round, ``reference_cx_bound``
and ``generator_cx_weight`` give the CX-count bound of a chain under an
explicit ``basis_depth_limit``, and ``current_projection``,
``candidate_score`` and ``ritz_gain`` evaluate the score of one extension.
With shots, ``sampled_metric`` and ``sampled_eligibility`` decide from the
overlap moments of the measured settings whether an extension's sampled
matrix entries resolve its independence well enough to be scored.
"""

from math import fsum, hypot, isfinite, sqrt

import numpy as np

# The multiplier k of the sampled eligibility rule (sampled_metric,
# sampled_eligibility), fixed before any data are seen. It is an empirical
# operational choice, not a coverage probability (docs/ENGINEERING_CONSTANTS.md).
ELIGIBILITY_MULTIPLIER = 3


def frontier(basis, cursor, *, pool_size, theta, growth, weight=None, reference_bound=0):
    """Return the extensions that one round scans, the next cursor and whether the parent window has no extension left.

    ``basis`` holds the chains of the current basis in basis order, the
    reference ``()`` first. Every basis chain is the reference or an
    earlier basis chain extended by one item, so a basis position
    identifies a parent for the rest of the run. The parents of a round,
    its window, are position 0 and the last ``growth.parent_window - 1``
    positions. A chain of ``growth.max_chain_length`` items has no
    extension and is left out. ``cursor`` holds ``(parent position, next
    pool index)`` pairs of the parents of an earlier round, and its last
    pair names the parent of the last examined extension, the anchor. A new
    parent starts at pool index 0, and an empty cursor has no anchor.

    The round visits the window in increasing cyclic order of position,
    starting at the first window position after the anchor, wrapping to
    the smallest one, or at the reference without an anchor, and goes
    round the window again and again. The anchor is read before the
    cursor is restricted to the window, because the anchor's parent may
    have left it. Each visit takes the parent's next pool index g, forms
    the extension ``parent + ((g, theta),)`` and advances that parent's
    index. A parent whose index reached ``pool_size`` is skipped. The round
    stops after ``growth.candidates_per_round`` admitted extensions, after
    ``growth.max_candidate_scans`` visited extensions or when no parent has
    a pool index left, so no extension tree is formed and the pool is never
    enumerated as a whole. Pool order, parent order, the per-parent
    indices and the anchor fix the result, so a later round continues the
    rotation where this one stopped. Two rounds that stop at scan limits m
    and n on an unchanged basis examine, in order, the extensions that one
    round stopping at scan limit m + n examines.

    Chains are compared as whole ordered tuples. An extension equal to a
    basis chain has reason ``"duplicate"``. A generator may occur more than
    once in a chain, also in adjacent positions, because
    ``exp(t A) exp(t A) = exp(2t A)`` is in general another state than
    ``exp(t A)``. The two literal factors stay two items, each counted in
    the chain length and its cost.

    With ``growth.basis_depth_limit`` set, ``weight(g)`` returns ``(w, None)``
    with the CX count bound of generator g at ``theta``
    (``generator_cx_weight``) or ``(None, reason)`` when it is unknown, and
    an extension's bound is ``reference_bound`` plus the weight of every
    literal item, repeats included. An extension with an unknown bound has
    reason ``"depth_unknown"``, one above the limit ``"above_depth_limit"``,
    and a bound equal to the limit is admitted. Without a limit ``weight``
    is never called.

    Returns:
        ``(labels, cursor, exhausted)``. ``labels`` is a tuple of dicts with
        keys ``chain``, ``reason`` (``"candidate"`` for an admitted
        extension), ``depth_bound`` and ``depth_reason``, in visiting order.
        ``cursor`` holds ``(parent position, next pool index)`` of this
        round's parents in cyclic position order ending with the parent of
        the last examined extension, also when that parent has no pool
        index left. A round that examines nothing returns the given cursor
        unchanged, so its anchor stays. ``exhausted`` is True when no parent
        of this round has a pool index left.
    """
    limit = growth.basis_depth_limit
    size = len(basis)
    window = [k for k in (0, *range(max(1, size - growth.parent_window + 1), size))
              if len(basis[k]) < growth.max_chain_length]
    previous = dict(cursor)
    following = {k: previous.get(k, 0) for k in window}
    anchor = cursor[-1][0] if cursor else -1
    start = next((i for i, k in enumerate(window) if k > anchor), 0)
    order = window[start:] + window[:start]
    present = set(basis)
    parent_bounds = {}

    def chain_bound(chain):
        """Return ``(bound, reason)`` of a chain: the reference bound plus each item's weight."""
        total = reference_bound
        for index, _ in chain:
            value, reason = weight(index)
            if value is None:
                return None, reason
            total += value
        return total, None

    labels, admitted, scans, last = [], 0, 0, None
    while admitted < growth.candidates_per_round and scans < growth.max_candidate_scans:
        visited = False
        for k in order:
            if admitted == growth.candidates_per_round or scans == growth.max_candidate_scans:
                break
            index = following[k]
            if index == pool_size:
                continue
            following[k], scans, visited, last = index + 1, scans + 1, True, k
            chain = basis[k] + ((index, theta),)
            label = dict(chain=chain, reason="candidate", depth_bound=None, depth_reason=None)
            if chain in present:
                label["reason"] = "duplicate"
            elif limit is not None:
                if k not in parent_bounds:
                    parent_bounds[k] = chain_bound(basis[k])
                bound, reason = parent_bounds[k]
                if bound is not None:
                    value, reason = weight(index)
                    bound = None if value is None else bound + value
                label.update(depth_bound=bound, depth_reason=reason)
                if bound is None:
                    label["reason"] = "depth_unknown"
                elif bound > limit:
                    label["reason"] = "above_depth_limit"
            admitted += label["reason"] == "candidate"
            labels.append(label)
        if not visited:
            break
    exhausted = all(following[k] == pool_size for k in window)
    if last is None:
        return tuple(labels), tuple(tuple(item) for item in cursor), exhausted
    end = window.index(last) + 1
    return tuple(labels), tuple((k, following[k]) for k in window[end:] + window[:end]), exhausted


def reference_cx_bound(preparation, num_qubits):
    """Return ``(bound, None)`` with the canonical CX count bound of ADAPT's reference preparation, or ``(None, reason)``.

    The count is that of a lowering to one-qubit U and CX gates before
    routing. An occupation reference uses X gates only and a product
    reference one single-qubit preparation per qubit, so both have 0. Every
    direct vector gets the upper law of the generic complex construction,
    ``2 max(0, 2**q - 2)`` (``_preparation_laws.direct_preparation_cx_bound``),
    read from the stored implementation and qubit count without scanning the
    vector. It covers the basis, uniform and prefix-uniform fast paths, so a
    shallow direct reference can exceed a limit that its actual circuit
    meets. A supplied circuit, a matrix-product-state circuit and a missing
    preparation have no count.
    """
    from nwqlib._preparation_laws import direct_preparation_cx_bound

    implementation = preparation.implementation
    if implementation in ("qiskit.occupation", "qiskit.product"):
        return 0, None
    if implementation == "qiskit.direct":
        return direct_preparation_cx_bound(num_qubits), None
    return None, f"the reference preparation {implementation!r} has no CX count"


def generator_cx_weight(definition):
    """Return ``(w, None)`` with the canonical CX count bound of one generator circuit, or ``(None, reason)``.

    ``definition`` is the top-level circuit of the uncontrolled forward
    generator gate that ``fermionic_circuits.build_generator_circuit``
    built. Each top-level instruction is read once, with no wrapper
    definition opened, no synthesis, transpilation or simulation, and its
    count under a lowering to one-qubit U and CX gates is added, assuming
    no cancellation or parallel execution, so ``w`` bounds the CX depth of
    the generator:

    - a standard one-qubit gate and the circuit's global phase: 0;
    - CX: 1, CZ: 1 (H-CX-H on the target), SWAP: 3 (three alternating CX),
      RXX, RYY, RZZ: 2 (one-qubit basis changes around CX-RZ-CX);
    - an RY with k >= 1 controls: ``2 M_k`` with ``M_k`` the CX count of a
      k-controlled X (``_mcx_counts.MCX_CX_BY_CONTROLS[k - 1]``), from
      ``C^k RY(phi) = RY(phi/2) X_k RY(-phi/2) X_k`` (rightmost factor
      first). With the controls inactive both ``X_k`` act as the identity
      and the rotations cancel, and with them active ``X RY(t) X = RY(-t)``
      gives ``RY(phi)``. Open controls add one-qubit X gates only;
    - the Gray-code multiplexor wrapper of ``_multiplexors._rotation_multiplexor``
      (a plain ``Gate`` named ``ucry`` on k + 1 qubits with its table of
      ``2**k`` angles): ``2**k``, one CX per Gray-code position, 0 for
      k = 0.

    Any other instruction, a controlled RY with more controls than the
    table holds, and an unbound or nonfinite parameter make the count
    unknown. The identities are those of ideal real angles. They add no
    synthesis and claim nothing about the numerical error of the actual
    generator circuit. The count is a pre-routing bound, not the native
    two-qubit depth of an executable circuit, which
    ``ADAPT.max_native_two_qubit_depth`` checks separately.
    """
    from qiskit.circuit import ControlledGate, Gate
    from qiskit.circuit.library import (
        CXGate, CZGate, HGate, IGate, PhaseGate, RXGate, RXXGate, RYGate, RYYGate, RZGate, RZZGate,
        SdgGate, SGate, SwapGate, SXdgGate, SXGate, TdgGate, TGate, UGate, XGate, YGate, ZGate)
    from nwqlib.subroutines._mcx_counts import MCX_CX_BY_CONTROLS

    single = (HGate, IGate, PhaseGate, RXGate, RYGate, RZGate, SdgGate, SGate, SXdgGate, SXGate,
              TdgGate, TGate, UGate, XGate, YGate, ZGate)
    fixed = {CXGate: 1, CZGate: 1, SwapGate: 3, RXXGate: 2, RYYGate: 2, RZZGate: 2}
    total = 0
    for instruction in definition.data:
        operation = instruction.operation
        name = operation.name
        # base_class names the public gate class also for Qiskit's singleton
        # instances of parameterless gates, such as H, whose own type is private.
        kind = operation.base_class
        if kind in single:
            count = 0
        elif kind in fixed:
            count = fixed[kind]
        elif isinstance(operation, ControlledGate) and type(operation.base_gate) is RYGate:
            controls = operation.num_ctrl_qubits
            if not 1 <= controls <= len(MCX_CX_BY_CONTROLS) or operation.num_qubits != controls + 1:
                return None, f"RY with {controls} controls has no tabulated CX count"
            count = 2 * MCX_CX_BY_CONTROLS[controls - 1]
        elif (kind is Gate and name == "ucry" and operation.num_qubits >= 1
              and len(operation.params) == 1 << (operation.num_qubits - 1)):
            controls = operation.num_qubits - 1
            count = (1 << controls) if controls else 0
        else:
            return None, f"instruction {name!r} on {operation.num_qubits} qubits has no CX count"
        if not all(isinstance(value, (int, float)) and isfinite(value) for value in operation.params):
            return None, f"instruction {name!r} has an unbound or nonfinite parameter"
        total += count
    return total, None


def current_projection(h0, coordinates, offset_free_energy):
    """Return the fixed inputs of every candidate score of one round, from the current solve.

    ``h0`` is the current b-by-b offset-free projected Hamiltonian ``H0``
    (the identity coefficient c removed), ``coordinates`` the solve's
    S-normalized coefficient vectors ``C`` (b by r, one column per kept
    direction, ``ProjectedPencil.coordinate_vectors``) and
    ``offset_free_energy`` the solve's lowest Ritz value before c is added,
    ``E0 = ProjectedPencil.offset_free_eigenvalues[0]``, never the physical
    value minus c.

    The score of the kept overlap subspace uses a retained map T with
    ``T^dagger S T = I_r`` (``T = V_r Lambda_r^(-1/2)`` from the kept overlap
    eigenpairs) and ``K0 = T^dagger H0 T``. The solve's ``C = T Z`` with Z
    the unitary eigenvectors of K0, so ``C^dagger S C = I_r`` and C spans
    the same kept subspace. Every score quantity depends on T only through
    ``t^dagger t``, ``t^dagger K0 t``, ``t^dagger h_r`` and ``T T^dagger``,
    which ``T -> T Z`` leaves unchanged, so ``C`` serves as T and no second
    overlap eigensolve is needed. ``c = C[:, 0]`` is the lowest Ritz vector.

    Returns:
        ``(T, K0, c, E0)`` with ``K0 = C^dagger H0 C``.
    """
    retained = np.asarray(coordinates, dtype=complex)
    k0 = retained.conj().T @ np.asarray(h0, dtype=complex) @ retained
    return retained, k0, retained[:, 0], float(offset_free_energy)


def candidate_score(current, h0, s, d0, independence_cutoff):
    """Return the exact score of one candidate state v against the current Ritz pair, with its reason.

    ``current`` is ``current_projection`` of the round. ``h0`` and ``s`` are
    the candidate's columns ``<phi_i|A0|v>`` and ``<phi_i|v>`` against the b
    current basis states, in that orientation (old state on the left), and
    ``d0 = <v|A0|v>`` its offset-free diagonal for the normalized v, with
    ``A0 = A - cI``. With ``t = T^dagger s`` and ``h_r = T^dagger h0``:

    - ``eta = 1 - sum_k (Re(t_k)**2 + Im(t_k)**2)`` is the squared norm of
      the part of v outside the kept span. The exact floor admits the
      candidate only when ``eta > independence_cutoff``.
    - ``r = (h0^dagger c - E0 s^dagger c)/sqrt(eta)`` is the coupling of
      the normalized residualized candidate q to the Ritz state
      ``Psi = Phi c`` through ``A - E``, and
      ``a0 = (d0 - 2 Re(t^dagger h_r) + Re(t^dagger K0 t))/eta`` is
      ``<q|A0|q>``. Writing ``h = h0 + c s``, ``d = d0 + c``,
      ``K = K0 + c I`` and ``E = E0 + c`` shows that c cancels from r and
      that ``<q|A|q> = a0 + c``, so ``delta = a0 - E0`` and ``rho = |r|``
      are formed before any identity coefficient is added.
    - The score is ``ritz_gain(delta, rho)``, the decrease
      ``E - lambda_-`` of the lowest eigenvalue of the 2-by-2 projection on
      ``{Psi, q}``.

    By Rayleigh-Ritz inclusion the lowest Ritz value of the kept span plus
    q lies at or below ``lambda_-``, so the score is a lower bound on the
    energy decrease of that ideal nested expansion, not an upper bound. It
    is not the full-space ground-energy error, it does not order the
    candidates by the decrease of a full solve, and a new solve of the
    enlarged basis with the overlap cutoff can keep a different subspace. A
    zero score does not show that a candidate cannot lower the energy: a
    candidate coupled only to an excited Ritz vector has ``rho = 0`` while
    the enlarged solve can be lower.

    Returns:
        A dict with ``eta``, ``coupling`` (rho), ``gap`` (delta), ``score``
        and ``reason``: ``"ranked"``, ``"below_independence_cutoff"`` or
        ``"nonfinite_score"``. A quantity that is not finite, or not
        computed because the floor rejected the candidate, is None.
    """
    retained, k0, ritz, energy = current
    s = np.asarray(s, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    t, eta = _kept_part(retained, s)
    record = dict(eta=eta if isfinite(eta) else None, coupling=None, gap=None, score=None)
    if not isfinite(eta):
        return dict(record, reason="nonfinite_score")
    if not eta > independence_cutoff:
        return dict(record, reason="below_independence_cutoff")
    coupling = abs((np.vdot(h0, ritz) - energy * np.vdot(s, ritz)) / sqrt(eta))
    diagonal = (float(d0) - 2.0 * np.vdot(t, retained.conj().T @ h0).real + np.vdot(t, k0 @ t).real) / eta
    gap = float(diagonal) - energy
    score = ritz_gain(gap, float(coupling)) if isfinite(gap) and isfinite(coupling) else float("nan")
    record.update(coupling=float(coupling) if isfinite(coupling) else None,
                  gap=gap if isfinite(gap) else None, score=score if isfinite(score) else None)
    return dict(record, reason="ranked" if isfinite(score) else "nonfinite_score")


def ritz_gain(delta, rho):
    """Return ``(sqrt(delta**2 + 4 rho**2) - delta)/2 >= 0`` without cancellation, overflow or avoidable underflow.

    This is ``E - lambda_-`` for the 2-by-2 matrix ``[[E, rho], [rho, E + delta]]``
    with ``lambda_- = (2E + delta - sqrt(delta**2 + 4 rho**2))/2``. For
    ``rho = 0`` it is ``max(0, -delta)``. Otherwise, with
    ``m = max(|delta|, rho)``, ``u = delta/m``, ``w = rho/m`` and
    ``t = hypot(u, 2w)``, it is ``rho (2w/(t + u))`` for ``delta >= 0``, the
    rationalized form ``2 rho**2/(sqrt(delta**2 + 4 rho**2) + delta)``, and
    ``m ((t - u)/2)`` for ``delta < 0``, each evaluated in that bracket
    order. The scaling keeps ``u`` and ``w`` at most 1, and ``hypot`` forms
    ``t`` without squaring ``w``.
    """
    if rho == 0:
        return max(0.0, -delta)
    m = max(abs(delta), rho)
    u, w = delta / m, rho / m
    t = hypot(u, 2 * w)
    if delta >= 0:
        return rho * ((2 * w) / (t + u))
    return m * ((t - u) / 2)


def _kept_part(retained, s):
    """Return ``t = T^dagger s`` and ``eta = 1 - sum_k (Re(t_k)**2 + Im(t_k)**2)`` of an overlap column ``s``."""
    t = retained.conj().T @ s
    return t, 1.0 - fsum(float(x.real) ** 2 + float(x.imag) ** 2 for x in t)


def overlap_proxy(shots, variance):
    """Return ``max(V, 1/(N + 1)**2)`` for one measured overlap part, or None when its evidence is unavailable.

    ``shots`` is the number N of returned shots of that part's own setting,
    not of the round, and ``variance`` the raw sample-mean variance V of its
    ancilla outcome (``GroupMoment.overlap_variance``). The evidence is
    available for N >= 2 and a finite nonnegative V. An outcome lies in
    [-1, 1], and 1/(N + 1) is a chosen scale for the resolution of one
    count, whose square keeps a setting whose N outcomes were all equal,
    with V = 0, from counting as noise-free. It is neither a lower bound on the true variance nor a
    confidence radius, and V itself is left as measured. ``(N + 1)**2`` is a
    Python integer, so a large N cannot overflow before the one division,
    and a floor that rounds to 0 makes the evidence unavailable.
    """
    if variance is None or not isfinite(variance) or variance < 0 or int(shots) < 2:
        return None
    floor = 1 / (int(shots) + 1) ** 2
    return max(float(variance), floor) if floor > 0 else None


def _entry_proxy(settings):
    """Return ``(Vtilde_real, Vtilde_imag)`` of one overlap entry from its ``(population, N, V)`` settings, or None.

    A one-setting entry is real-only: its imaginary part is a structural
    zero of the proved real pencil, not a measurement, and gets no floor.
    """
    proxies = [overlap_proxy(shots, variance) for _, shots, variance in settings]
    if None in proxies:
        return None
    return proxies[0], proxies[1] if len(proxies) == 2 else 0.0


def eta_gradients(a):
    """Return the gradients of eta with respect to the real and imaginary parts of the measured overlap entries.

    With ``a = T t`` and the current pencil's ``D = T^dagger (dS) T`` and
    ``dt = T^dagger ds``, the first-order change of
    ``eta = 1 - (t + dt)^dagger (I + D)^(-1) (t + dt)`` at ``D = 0`` is
    ``t^dagger D t - 2 Re(t^dagger dt) = a^dagger (dS) a - 2 Re(a^dagger ds)``,
    from ``d(M^(-1)) = -M^(-1) (dM) M^(-1)``. The diagonal ``S_ii = 1`` is
    exact and has no error, and the Hermitian completion ties ``S_ji`` to
    ``S_ij``. So an upper-triangle current entry ``S_ij`` (i < j) has the
    gradient ``(2 Re w, -2 Im w)`` with ``w = conj(a_i) a_j``, the minus
    sign coming from that orientation, and the candidate overlap
    ``s_i = <phi_i|v>`` has ``(-2 Re a_i, -2 Im a_i)``.

    Returns:
        ``(current, column)``: a dict ``{(i, j): (g_real, g_imag)}`` over
        i < j and a list of ``(u_real, u_imag)`` in basis order.
    """
    current = {}
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            w = a[i].conjugate() * a[j]
            current[i, j] = (2 * float(w.real), -2 * float(w.imag))
    return current, [(-2 * float(x.real), -2 * float(x.imag)) for x in a]


def sampled_metric(current, entries, smallest):
    """Return the part of one round's sampled eligibility that does not depend on the extension.

    ``current`` is ``current_projection`` of the round's pencil and
    ``entries`` maps every upper-triangle pair ``(i, j)``, i < j, of the b
    current states to the ``(population, N, V)`` settings of its overlap,
    real part first and the imaginary part second unless the pencil is
    real-only, from the overlap-supplying group zero. ``population`` names
    the raw counts of a setting (``GroupMoment.contribution_id``).
    ``smallest`` is the smallest overlap eigenvalue that the solve kept.

    Each part gets ``overlap_proxy``, and an entry its covariance proxy
    ``C_ij = diag(Vtilde_real, Vtilde_imag)``. A pair whose canonical chain
    order is reversed in the basis maps its raw parts by ``J = diag(1, -1)``,
    which changes only the sign of an imaginary gradient component and so
    no squared term below. The metric scale uses the row variances of the
    error of S, ``nu = max_i sum_(j != i) tr C_ij``: for independent
    Hermitian pair errors ``X_ij = z_ij |i><j| + conj(z_ij) |j><i|`` the
    cross terms of ``E[(dS)**2]`` vanish and ``X_ij**2`` puts ``|z_ij|**2``
    on the diagonal entries i and j, so ``E[(dS)**2]`` is diagonal with
    these row sums and ``nu`` is its operator norm. With the proxies in
    place of the true moments, ``beta = k sqrt(nu)`` and
    ``epsilon = beta / smallest`` for ``k = ELIGIBILITY_MULTIPLIER``. Neither
    is a bound on ``||dS||`` or a confidence radius. ``epsilon < 1/2`` keeps
    the amplification of ``(I + D)^(-1)`` at most 2 in the model of
    ``sampled_eligibility``. ``current[0]`` is the solve's coordinate matrix
    ``C = T Z`` with Z unitary, which has the row norms of T and gives the
    same ``a = T T^dagger s`` (``current_projection``). The work is O(b**2)
    once per round.

    Returns:
        A dict with ``reason`` (None, ``"screening_noise_unknown"`` or
        ``"screening_metric_unresolved"``), ``metric_beta`` and
        ``metric_epsilon`` (None when not computed), ``proxies`` of every
        entry, the ``populations`` that name each entry's settings and the
        squared norms ``rows`` of the rows of T, which every extension of the
        round reuses. Evidence is unknown when a part has N < 2 or no finite
        nonnegative V, or when one raw population serves two distinct
        entries, because the sum of independent terms below would then
        miss their correlation.
    """
    retained = current[0]
    rows = [fsum(float(x.real) ** 2 + float(x.imag) ** 2 for x in row) for row in retained]
    metric = dict(reason=None, metric_beta=None, metric_epsilon=None, proxies={}, populations={}, rows=rows)
    for pair, settings in entries.items():
        for population, _, _ in settings:
            if metric["populations"].setdefault(population, pair) != pair:
                metric["reason"] = "screening_noise_unknown"
        metric["proxies"][pair] = _entry_proxy(settings)
        if metric["proxies"][pair] is None:
            metric["reason"] = "screening_noise_unknown"
    if metric["reason"] is not None:
        return metric
    size = len(rows)
    nu = max(fsum(sum(metric["proxies"][min(i, j), max(i, j)]) for j in range(size) if j != i)
             for i in range(size))
    beta = ELIGIBILITY_MULTIPLIER * sqrt(nu)
    epsilon = beta / smallest
    metric.update(metric_beta=beta if isfinite(beta) else None, metric_epsilon=epsilon if isfinite(epsilon) else None)
    if not (isfinite(epsilon) and epsilon < 0.5):
        metric["reason"] = "screening_metric_unresolved"
    return metric


def sampled_eligibility(current, metric, s, column, independence_cutoff):
    """Return whether one extension's sampled overlaps resolve its part outside the kept span, with the quantities used.

    ``current`` and ``metric`` are ``current_projection`` and
    ``sampled_metric`` of the round, ``s`` the sampled overlap column
    ``<phi_i|v>`` and ``column`` the ``(population, N, V)`` settings of its
    b entries in basis order, as in ``sampled_metric``. With
    ``t = T^dagger s``, ``a = T t`` and ``etahat = 1 - t^dagger t``:

    - ``sigma**2 = sum_(i<j) g_ij^T C_ij g_ij + sum_i u_i^T C'_i u_i`` with
      the gradients of ``eta_gradients`` and the extension's covariance
      proxies ``C'_i``. The proxies are already variances of means and
      count each independent setting once.
    - ``b_s = sum_i ||T_(i,:)||**2 tr C'_i``, ``d = k sqrt(b_s)`` and
      ``x = ||t||``.
    - ``R = k sigma + (d + epsilon x)**2/(1 - epsilon)`` and the margin
      ``m = etahat - R``.

    The second term of R comes from
    ``(I + D)^(-1) = I - D + D**2 (I + D)^(-1)``: if ``||D|| <= epsilon < 1``
    and ``||dt|| <= d``, the change of eta beyond its first-order part is at
    most ``d**2 + 2 epsilon x d + epsilon d**2 + epsilon**2 (x + d)**2/(1 - epsilon)``,
    which equals ``(d + epsilon x)**2/(1 - epsilon)``, with equality at
    ``D = -epsilon I``, ``t = (x, 0)``, ``dt = (d, 0)``. It accounts for the
    quadratic noise, which remains at ``t = 0``, where the first-order term
    vanishes. epsilon and d come from variance proxies, not from
    norm bounds, so R is a conditional empirical resolution, not a
    confidence bound. It does not cover the rotation of the kept subspace by
    the noise of S, the selection of the extension by earlier data, bias
    or drift.

    The extension is eligible only when all its evidence is available,
    ``epsilon < 1/2`` and ``m > independence_cutoff``. etahat is not
    clipped at 0, and a margin at or below the cutoff gives
    ``"screening_independence_unresolved"``, which does not show that the
    extension depends on the current states. The work is O(b**2) for each
    extension and uses no extra shot.

    Returns:
        A dict with ``reason`` (None when eligible, otherwise
        ``"screening_noise_unknown"``, ``"screening_metric_unresolved"`` or
        ``"screening_independence_unresolved"``), ``eta`` (etahat),
        ``independence_cutoff``, ``multiplier`` (k), ``metric_beta``,
        ``metric_epsilon``, ``eta_sigma``, ``extension_noise`` (b_s),
        ``eta_resolution`` (R) and ``eta_margin`` (m), each None when it was
        not computed or is not finite.
    """
    retained = current[0]
    s = np.asarray(s, dtype=complex)
    t, eta = _kept_part(retained, s)
    record = dict(reason=metric["reason"], eta=eta if isfinite(eta) else None,
                  independence_cutoff=float(independence_cutoff), multiplier=ELIGIBILITY_MULTIPLIER,
                  metric_beta=metric["metric_beta"], metric_epsilon=metric["metric_epsilon"],
                  eta_sigma=None, extension_noise=None, eta_resolution=None, eta_margin=None)
    if record["reason"] is not None:
        return record
    # A raw population may serve only one entry, also across the current
    # pencil and this extension's column (sampled_metric).
    size, seen, proxies = len(retained), dict(metric["populations"]), []
    for i, settings in enumerate(column):
        for population, _, _ in settings:
            if seen.setdefault(population, (i, size)) != (i, size):
                record["reason"] = "screening_noise_unknown"
        proxies.append(_entry_proxy(settings))
    if record["reason"] is not None or None in proxies:
        return dict(record, reason="screening_noise_unknown")
    gradients, column_gradients = eta_gradients(retained @ t)
    sigma = sqrt(fsum([v * g[0] ** 2 + w * g[1] ** 2 for (v, w), g in
                       ((metric["proxies"][pair], gradients[pair]) for pair in gradients)]
                      + [v * u[0] ** 2 + w * u[1] ** 2 for (v, w), u in zip(proxies, column_gradients)]))
    noise = fsum(row * (v + w) for row, (v, w) in zip(metric["rows"], proxies))
    d = ELIGIBILITY_MULTIPLIER * sqrt(noise)
    x = sqrt(fsum(float(z.real) ** 2 + float(z.imag) ** 2 for z in t))
    epsilon = metric["metric_epsilon"]
    resolution = ELIGIBILITY_MULTIPLIER * sigma + (d + epsilon * x) ** 2 / (1 - epsilon)
    margin = eta - resolution
    values = dict(eta_sigma=sigma, extension_noise=noise, eta_resolution=resolution, eta_margin=margin)
    record.update({key: value if isfinite(value) else None for key, value in values.items()})
    if not all(isfinite(value) for value in (eta, *values.values())):
        return dict(record, reason="screening_metric_unresolved")
    return dict(record, reason=None if margin > independence_cutoff else "screening_independence_unresolved")
