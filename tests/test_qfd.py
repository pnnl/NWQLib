"""QFD workflows: the fixed-V basis, its joint-state readout, the reference energy and the Run lifecycle."""

from fractions import Fraction as F

import numpy as np
import pytest

import nwqlib
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms import QFD, PencilAcquisition
from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.operators import ingest_pauli
from nwqlib.problems import ingest_occupation, ingest_product

U = F(1, 2**53)
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1, -1])}
ROTATIONS = ("rx", "ry", "rz", "rxx", "ryy", "rzz", "crz")


def _pauli(label):
    """The matrix of a Pauli label with qubit 0 rightmost, as a Kronecker product from the left."""
    matrix = np.eye(1)
    for letter in label:
        matrix = np.kron(matrix, PAULI[letter])
    return matrix


def _rational(value):
    return F(float(value.real)), F(float(value.imag))


def _exact_entry(left, matrix_terms, right):
    """``<left| sum_k c_k P_k |right>`` in exact Gaussian rationals from binary64 vectors."""
    real = imag = F(0)
    for coefficient, label in matrix_terms:
        p = _pauli(label)
        for row in range(len(left)):
            lr, li = _rational(left[row])
            for column in range(len(right)):
                if p[row, column] == 0:
                    continue
                pr, pi = int(p[row, column].real), int(p[row, column].imag)
                rr, ri = _rational(right[column])
                # conj(l) * p * r with p in {+-1, +-i}
                ar, ai = lr * pr + li * pi, lr * pi - li * pr
                real += coefficient * (ar * rr - ai * ri)
                imag += coefficient * (ar * ri + ai * rr)
    return real, imag


def _oracle(plan, reference):
    """The basis states from the stored rotations, multiplied independently, with their T4 error bounds.

    Each factor is ``exp(-i theta P/2) = cos(theta/2) I - i sin(theta/2) P``
    with the stored angle. A negative power applies the factors in reverse
    order with negated angles. Every factor application adds the gate
    error ``4Du`` and the product error ``2D gamma_2D`` of a D-entry state.
    """
    rec = plan.reconstruction
    rows = rec.evolution_rows
    factors = [(rows[k][0], angle) for k, angle in zip(rec.step_rows.array.tolist(),
                                                       rec.step_angles.array.tolist(), strict=True)]
    d = len(reference)
    gate, product = 4 * d * U, 2 * d * (2 * d * U / (1 - 2 * d * U))
    states, errors = [], []
    for power in plan.method.powers:
        sequence = factors if power >= 0 else [(label, -angle) for label, angle in reversed(factors)]
        vector, error = reference.copy(), F(0)
        for _ in range(abs(power) * plan.method.steps_per_time_step):
            for label, angle in sequence:
                vector = (np.cos(angle / 2) * np.eye(d) - 1j * np.sin(angle / 2) * _pauli(label)) @ vector
                error = (1 + error) * (1 + gate) * (1 + product) - 1
        states.append(vector)
        errors.append(error)
    return states, errors


def _annihilators(modes):
    """The matrices of a_j from their action on occupation states, mode j on bit j and |1> occupied.

    ``a_j |n> = (-1)**(n_0 + ... + n_{j-1}) |n - e_j>`` when ``n_j = 1``, the
    sign of the lower occupied modes, formed without any Pauli mapping.
    """
    d = 1 << modes
    ladders = []
    for j in range(modes):
        a = np.zeros((d, d), dtype=int)
        for n in range(d):
            if n >> j & 1:
                a[n ^ 1 << j, n] = (-1) ** bin(n & ((1 << j) - 1)).count("1")
        ladders.append(a)
    return ladders


def _ladder_word(ladders, ops):
    """The integer matrix of a ladder word, its left factor first, with action 1 a creation."""
    matrix = np.eye(len(ladders[0]), dtype=int)
    for mode, action in ops:
        matrix = matrix @ (ladders[mode].T if action else ladders[mode])
    return matrix


def _pauli_decomposition(matrix):
    """The exact coefficients ``tr(P M)/d`` of every Pauli label of a matrix with Gaussian-integer entries."""
    from itertools import product

    n = len(matrix).bit_length() - 1
    out = {}
    for letters in product("IXYZ", repeat=n):
        label = "".join(letters)
        trace = complex(np.trace(_pauli(label) @ matrix))
        if trace:
            out[label] = (F(trace.real) / len(matrix), F(trace.imag) / len(matrix))
    return out


def _quadratic(left, matrix, right):
    """``left^dagger matrix right`` in exact Gaussian rationals of binary64 entries, as (real, imaginary)."""
    real = imag = F(0)
    for i, x in enumerate(left):
        xr, xi = _rational(x)
        for j, y in enumerate(right):
            mr, mi = _rational(matrix[i][j])
            yr, yi = _rational(y)
            ar, ai = xr * mr + xi * mi, xr * mi - xi * mr  # conj(x) * m
            real += ar * yr - ai * yi
            imag += ar * yi + ai * yr
    return real, imag


def _pencil_inputs(result):
    """The centered ``H0`` and unit-diagonal ``S`` that the solve used, from the exact pair records, and their entry bounds.

    The assembly is that of ``pencil.assemble_pair_pencil``: the upper
    triangle as recorded, the lower triangle by conjugation, ``S_ii = 1``
    and the real part of a diagonal ``H0_ii``. The bounds are the qualified
    entry bounds of the pair records, zero for the conventional ``S_ii``.
    """
    b = len(result.pencil.overlap)
    h, s = [[0j] * b for _ in range(b)], [[1 + 0j] * b for _ in range(b)]
    eh, es = [[0.0] * b for _ in range(b)], [[0.0] * b for _ in range(b)]
    for pair in result.pairs:
        i, j = pair.pair
        hv, sv = complex(pair.hamiltonian.real, pair.hamiltonian.imag), complex(pair.overlap.real, pair.overlap.imag)
        h[i][j], h[j][i] = (hv.real + 0j, hv.real + 0j) if i == j else (hv, hv.conjugate())
        eh[i][j] = eh[j][i] = pair.hamiltonian_bound
        if i != j:
            s[i][j], s[j][i] = sv, sv.conjugate()
            es[i][j] = es[j][i] = pair.overlap_bound
    return h, s, eh, es


def _ritz_allowance(h, s, eh, es, columns, energies, sigma, magnitude, norm, scale=None):
    """The allowance ``B_input + B_solve`` on each returned centered Ritz value, in exact rationals.

    ``eh`` and ``es`` bound the entries of ``h - H*`` and ``s - S*`` for the
    ideal pencil ``(H*, S*)``, so their largest row sums ``e_H`` and ``e_S``
    bound the spectral norms. ``sigma <= lambda_min(S*)`` and ``magnitude``
    bounds the ideal Rayleigh quotients. Comparing the Rayleigh quotients
    of the two pencils gives ``B_input = (e_H + magnitude e_S)/(sigma - e_S)``
    when ``e_S < sigma``. ``columns`` are the returned S-normalized
    coefficient columns of the full kept spectrum and ``energies`` their
    centered Ritz values. With ``F = C^dagger h C - D`` and
    ``G = C^dagger s C - I``, bounded by their largest row sums ``phi`` and
    ``g`` of ``|Re| + |Im|`` entries, congruence gives
    ``B_solve = (phi + max|E| g)/(1 - g)`` when ``g < 1``. The solve must
    also stay within the arithmetic resolution of its inputs:
    ``B_solve <= (e_H + magnitude e_S + norm gram(b, b))/(sigma - e_S)``,
    with ``norm`` the operator-norm bound of the centered Hamiltonian and
    ``gram`` the Gram formation allowance of b states. For the rank-one
    pencil ``S* = J = 11^dagger``, ``H* = k J`` every column c with
    ``sigma_c = |sum_i c_i|**2 > 0`` has the Rayleigh quotient k, so the
    one returned column is a fixed one-dimensional map: there ``scale`` is
    ``a = ||c||**2``, which multiplies ``e_H`` and ``e_S``, and ``sigma`` is
    ``sigma_c``. A failed premise raises instead of widening the allowance.
    """
    from nwqlib._projected_eigensolver import entrywise_gram_allowance, gram_formation_allowance

    e_h, e_s = F(entrywise_gram_allowance(eh)), F(entrywise_gram_allowance(es))
    if scale is not None:
        e_h, e_s = scale * e_h, scale * e_s
    assert e_s < sigma, "the overlap error reaches the ideal overlap's smallest eigenvalue"
    b = len(columns)
    rows_f, rows_g = [], []
    for j in range(b):
        row_f, row_g = F(0), F(0)
        for k in range(b):
            fr, fi = _quadratic(columns[j], h, columns[k])
            gr, gi = _quadratic(columns[j], s, columns[k])
            if j == k:
                fr, gr = fr - F(energies[j]), gr - 1
            row_f, row_g = row_f + abs(fr) + abs(fi), row_g + abs(gr) + abs(gi)
        rows_f.append(row_f)
        rows_g.append(row_g)
    phi, g = max(rows_f), max(rows_g)
    assert g < 1, "the returned coefficients are not S-normalized to within 1"
    solve = (phi + max(abs(F(e)) for e in energies) * g) / (1 - g)
    numerical = (e_h + magnitude * e_s + norm * F(gram_formation_allowance(b, b))) / (sigma - e_s)
    assert solve <= numerical, "the small solve exceeds its arithmetic-resolution acceptance (T7)"
    return (e_h + magnitude * e_s) / (sigma - e_s) + solve


_PRIMITIVES = {"x", "h", "s", "sdg", "cx", "rx", "ry", "rz", "rxx", "ryy", "rzz", "crz"}


def _primitive_gates(circuit, wires=None):
    """The primitive gates of a lowered circuit in execution order, as ``(name, params, qubits)``."""
    wires = list(range(circuit.num_qubits)) if wires is None else wires
    for item in circuit.data:
        operation = item.operation
        qubits = [wires[circuit.find_bit(qubit).index] for qubit in item.qubits]
        if operation.name in _PRIMITIVES:
            yield operation.name, tuple(float(p) for p in operation.params), qubits
        else:
            assert operation.definition is not None, operation.name
            yield from _primitive_gates(operation.definition, qubits)


def _apply_pauli(state, letters):
    """``P state`` for ``letters`` mapping qubits to X, Y or Z, by exact permutation and phases."""
    index = np.arange(state.size)
    mask, phase = 0, np.ones(state.size, dtype=complex)
    for qubit, letter in letters.items():
        bit = index >> qubit & 1
        mask |= (letter in "XY") << qubit
        if letter == "Y":
            phase = phase * np.where(bit, -1j, 1j)
        elif letter == "Z":
            phase = phase * np.where(bit, -1, 1)
    out = np.empty_like(state)
    out[index ^ mask] = phase * state
    return out


def _simulate(circuit):
    """The state of a lowered circuit from |0...0> by independent primitive gates, with its T4 norm-error bound.

    A rotation ``exp(-i theta P/2)`` is ``cos(theta/2) I - i sin(theta/2) P``,
    a controlled RZ applies RZ on the control's one branch, and H, S, S^dagger,
    X and CX act by their matrices. Each gate adds the gate error ``4Du``
    and the product error ``2D gamma_2D`` of a D-entry state,
    ``1 + delta' = (1 + delta)(1 + 4Du)(1 + 2D gamma_2D)``, D = 2**width, a
    bound that assumes one-ulp sine and cosine and is used for D <= 32 with
    the ancilla and at most 64 gates.
    """
    width = circuit.num_qubits
    state = np.zeros(1 << width, dtype=complex)
    state[0] = 1
    index = np.arange(state.size)
    gates = list(_primitive_gates(circuit))
    d = state.size
    assert d <= 32 and len(gates) <= 64, (d, len(gates))
    gate, product = 4 * d * U, 2 * d * (2 * d * U / (1 - 2 * d * U))
    error = F(0)
    for name, params, qubits in gates:
        if name in ("rx", "ry", "rz", "rxx", "ryy", "rzz"):
            letter = name[-1].upper()
            rotated = _apply_pauli(state, {qubit: letter for qubit in qubits})
            state = np.cos(params[0] / 2) * state - 1j * np.sin(params[0] / 2) * rotated
        elif name == "crz":
            control, target = qubits
            rotated = np.cos(params[0] / 2) * state - 1j * np.sin(params[0] / 2) * _apply_pauli(state, {target: "Z"})
            state = np.where(index >> control & 1, rotated, state)
        elif name == "h":
            (qubit,) = qubits
            state = (_apply_pauli(state, {qubit: "X"}) + _apply_pauli(state, {qubit: "Z"})) * 2**-0.5
        elif name in ("s", "sdg"):
            (qubit,) = qubits
            state = np.where(index >> qubit & 1, (1j if name == "s" else -1j) * state, state)
        elif name == "x":
            state = _apply_pauli(state, {qubits[0]: "X"})
        else:
            control, target = qubits
            state = np.where(index >> control & 1, _apply_pauli(state, {target: "X"}), state)
        error = (1 + error) * (1 + gate) * (1 + product) - 1
    return state, error


def _lowered(planned, name):
    """The Qiskit circuit of one experiment of a Plan, lowered from its selected construction."""
    from nwqlib.blocks.lowering import lower_qiskit

    _, construction = planned.resolve(name)._selected_construction(planned)
    ids = {record.content_id for record in construction.selections}
    return lower_qiskit(construction, blocks=tuple(b for b in planned.blocks if b.record.content_id in ids)).circuit


def _outside_weight(state, inside):
    """The exact squared norm, in rationals of the binary64 amplitudes, of the amplitudes whose index fails ``inside``."""
    return sum((F(float(z.real)) ** 2 + F(float(z.imag)) ** 2 for i, z in enumerate(state) if not inside(i)), F(0))


def test_three_powers_follow_the_base_power_rule_with_a_phased_reference_and_the_exact_adjoint():
    """The pair entries of powers (0, 2, -1) equal those of independently multiplied basis states.

    A = 0.25 II + 0.5 ZI + 0.3 XX - 0.4 IY on two qubits has noncommuting
    terms, so V, its order-1 step for tau = 0.7, commutes with neither the
    reference nor its own terms. The reference is the product of the exact
    dyadic states ((1+i)/2, (1+i)/2) on qubit 0 and ((1+i)/2, (-1+i)/2) on
    qubit 1, with a nontrivial global phase, and no term leaves it
    unchanged up to a phase. The pair (0, 1) controls V
    twice on the one branch, the pair (1, 2) runs V^dagger once without
    control and controls V three times on the zero branch, and the pair
    (0, 2) controls V^dagger. Every exact pair entry H0_ij and S_ij, with
    the oracle's contraction in exact rationals, lies within its qualified
    entry bound plus the oracle bound C(d_i + d_j + d_i d_j), and C omitted
    for S, of the T4 recurrence. The power -1 is the adjoint of the same V:
    the order-1 step at time -tau with the original term order gives an
    overlap S_02 farther from the measured one than a thousand times that
    allowance. Each stored angle is the correctly rounded 2 tau c.
    """
    rows = (("II", .25), ("ZI", .5), ("XX", .3), ("IY", -.4))
    qubit0, qubit1 = np.array([1 + 1j, 1 + 1j]) / 2, np.array([1 + 1j, -1 + 1j]) / 2
    reference = ingest_product(np.array([qubit0, qubit1]))
    method = QFD(initial_state=reference, time_step=.7, powers=(0, 2, -1), formula_order=1)
    result = solve(Eigenproblem(A=ingest_pauli(rows, num_qubits=2)), method=method, execution="quantum", seed=3)
    plan = result.plan
    rec = plan.reconstruction
    assert [p.route for p in rec.pencil.acquisition.pairs] == ["prefix"] * 3
    assert [p.shared_blocks for p in rec.pencil.acquisition.pairs] == [1, 1, 2]
    for (label, coefficient), angle in zip(rec.pencil.nonidentity_terms, rec.step_angles.array.tolist(),
                                           strict=True):
        assert angle == float(2 * F(.7) * F(coefficient))

    psi = np.kron(qubit1, qubit0)
    states, errors = _oracle(plan, psi)
    terms = [(F(c), label) for label, c in rows if label != "II"]
    mass = sum(abs(c) for c, _ in terms)
    for pair in result.pairs:
        i, j = pair.pair
        reference_error = errors[i] + errors[j] + errors[i] * errors[j]
        for measured, bound, matrix_terms, scale in (
                (pair.hamiltonian, pair.hamiltonian_bound, terms, mass),
                (pair.overlap, pair.overlap_bound, [(F(1), "II")], 1)):
            exact = _exact_entry(states[i], matrix_terms, states[j])
            difference = (F(measured.real) - exact[0], F(measured.imag) - exact[1])
            allowance = F(bound) + scale * reference_error
            assert difference[0] ** 2 + difference[1] ** 2 <= allowance ** 2, (pair.pair, measured)

    forward = [(rec.pencil.nonidentity_terms[k][0], -angle)
               for k, angle in zip(rec.step_rows.array.tolist(), rec.step_angles.array.tolist(), strict=True)]
    naive = psi.copy()
    for label, angle in forward:
        naive = (np.cos(angle / 2) * np.eye(4) - 1j * np.sin(angle / 2) * _pauli(label)) @ naive
    measured = next(p for p in result.pairs if p.pair == (0, 2))
    allowance = float(F(measured.overlap_bound) + errors[0] + errors[2] + errors[0] * errors[2])
    assert abs(np.vdot(psi, naive) - complex(measured.overlap.real, measured.overlap.imag)) > 1000 * allowance


def test_requested_power_order_keeps_the_reference_energy_through_save_load_and_reanalysis(tmp_path):
    """Powers (1, 0, 3) keep their order, the reference energy of power 0 and the fixed step after a reload.

    The reference |0> of A = 0.25 + 0.5 X + 0.75 Z has the exact energy
    1, and V is two order-2 substeps of time step 2. Each substep of
    length 1 runs the X and Z rotations for half the substep and then
    again in reverse order, so the rotations R_P(theta) = exp(-i theta P/2)
    of X, Z, Z and X have the angles theta = c = 0.5, 0.75, 0.75 and 0.5.
    The power 0 is basis state 1, so the reference energy is H[1, 1]. The
    large step does not conserve the energy, so H[0, 0] and H[2, 2] of the
    evolved states lie more than 0.05 from it, and the lowering is the
    reference energy less the Ritz value. The diagonal circuits of the
    powers 1, 0 and 3 contain 8, 0 and 24 Pauli rotations, |p| r 2L for
    L = 2 terms and r = 2. Saved and loaded, the Result keeps its values,
    the Plan its powers, step rows, angles and construction, the blocks
    rebuilt from the stored factors have the recorded selected
    definitions, and analysis at a new cutoff reuses the same H and S
    without new measurement. The pair (0, 2) runs V once without control
    and controls V twice, so the gates of its circuit lowered after the
    reload, by name and parameters in order, equal the planned ones only if
    both rebuilt blocks keep the sign and order of every rotation.
    """
    from nwqlib.blocks.lowering import lower_qiskit

    rows = (("I", .25), ("X", .5), ("Z", .75))
    method = QFD(initial_state=ingest_occupation((0,), num_qubits=1), time_step=2.0, powers=(1, 0, 3),
                 steps_per_time_step=2)
    result = solve(Eigenproblem(A=ingest_pauli(rows, num_qubits=1)), method=method, execution="quantum", seed=5)
    plan = result.plan
    assert plan.reconstruction.reference_index == 1 and plan.method.powers == (1, 0, 3)
    stored = plan.reconstruction
    terms = stored.pencil.nonidentity_terms
    assert [terms[k][0] for k in stored.step_rows.array.tolist()] == ["X", "Z", "Z", "X"]
    assert stored.substep == 1.0 and stored.step_angles.array.tolist() == [.5, .75, .75, .5]
    h = result.pencil.hamiltonian
    assert result.reference_energy == h[1][1].real == 1.0
    assert min(abs(h[0][0].real - 1.0), abs(h[2][2].real - 1.0)) > .05
    assert result.lowering == result.reference_energy - result.eigenvalue

    def lowered(planned, name):
        _, construction = planned.resolve(name)._selected_construction(planned)
        ids = {record.content_id for record in construction.selections}
        blocks = tuple(b for b in planned.blocks if b.record.content_id in ids)
        return lower_qiskit(construction, blocks=blocks).circuit

    def gates(body):
        # (name, parameters) of every gate in order. A gate other than a
        # rotation that has a body, such as a block's gate, whose generated
        # name differs between Plans, is replaced by that body.
        return [gate for item in body.data for gate in (
            gates(item.operation.definition) if item.operation.name not in ROTATIONS
            and item.operation.definition is not None else ((item.operation.name, tuple(item.operation.params)),))]

    assert [sum(name in ROTATIONS for name, _ in gates(lowered(plan, f"pair_{k}_{k}"))) for k in range(3)] == [
        8, 0, 24]

    loaded = nwqlib.load_result(result.save(tmp_path / "qfd"))
    assert (loaded.eigenvalue, loaded.reference_energy, loaded.lowering) == (
        result.eigenvalue, result.reference_energy, result.lowering)
    again = loaded.plan
    assert again.method.powers == (1, 0, 3) and again.construction == plan.construction
    assert tuple(block.record for block in again.blocks) == again.construction.selections
    assert gates(lowered(again, "pair_0_2")) == gates(lowered(plan, "pair_0_2"))
    rec, saved = again.reconstruction, plan.reconstruction
    assert rec.step_rows.array.tolist() == saved.step_rows.array.tolist()
    assert rec.step_angles.array.tobytes() == saved.step_angles.array.tobytes()
    reanalyzed = loaded.analyze(overlap_cutoff=1e-6)
    assert reanalyzed.pencil.hamiltonian == result.pencil.hamiltonian
    assert reanalyzed.pencil.overlap == result.pencil.overlap
    assert reanalyzed.reference_energy == result.reference_energy and reanalyzed.analysis_cutoff == 1e-6


def test_complete_sampled_pencil_has_its_settings_and_a_resumed_run_repeats_none(tmp_path, monkeypatch):
    """M = 3 powers and G = 2 groups give 18 settings and 18 N shots, and a reopened Run submits only what is left.

    A = 0.5 ZI - 0.3 IZ + 0.2 XX has the qubit-wise commuting groups
    {IZ, ZI} and {XX}. The complete complex pencil has M G = 6 diagonal and
    2 M(M-1)/2 G = 12 off-diagonal settings. A Run stopped before its eighth
    submission keeps its seven measurements. Reopened from disk it measures
    the eleven others, keeps the first seven count tables first, and holds
    18 jobs of N shots. Reopened again it submits nothing.
    """
    from nwqlib._prepared_execution import Run
    from nwqlib.backends import AerBackend, qiskit_aer

    shots = 16
    problem = Eigenproblem(A=ingest_pauli((("ZI", .5), ("IZ", -.3), ("XX", .2)), num_qubits=2))
    method = QFD(initial_state=ingest_occupation((1, 0), num_qubits=2), time_step=.4, powers=(0, 1, 2))
    plan = nwqlib.plan(problem, method=method, execution="quantum", shots=shots, seed=11)
    assert len(plan.reconstruction.pencil.groups) == 2 and len(plan.experiments) == 18
    workload = nwqlib.estimate(plan)
    assert workload.quantity("shots").fact.value.numerator == 18 * shots
    assert plan.reconstruction.pencil.acquisition.quadratures == "complex"

    original, submitted = QFD.before_submit, []

    def interrupted(self, *args, **kwargs):
        if len(submitted) == 7:
            raise RuntimeError("stop before the eighth submission")
        submitted.append(1)
        return original(self, *args, **kwargs)

    path = tmp_path / "qfd-run"
    with Run(plan, directory=path) as run:
        with monkeypatch.context() as patch:
            patch.setattr(QFD, "before_submit", interrupted)
            with pytest.raises(RuntimeError, match="eighth submission"):
                run.resume()
        prefix = tuple(chunk.content_id for chunk in run.observations.chunks)
    assert len(prefix) == 7
    with nwqlib.load_run(path, backend=AerBackend()) as restored:
        result = restored.wait()
        assert tuple(chunk.content_id for chunk in restored.observations.chunks[:7]) == prefix
        assert len(restored.trace.events) == 18 and sum(e.shots for e in restored.trace.events) == 18 * shots
    assert result.missing == () and len(result.group_moments) == 18
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", lambda *a, **k: pytest.fail("resubmitted"))
    with nwqlib.load_run(path, backend=AerBackend()) as reopened:
        assert reopened.wait().eigenvalue == result.eigenvalue and len(reopened.trace.events) == 18


def test_pencil_sampling_takes_the_estimate_and_reference_energy_from_the_main_stage(tmp_path, monkeypatch):
    """PencilSampling on the basis (psi, V psi) of A = -Z: integer stage split, main-only estimate, frozen decision.

    One group and two powers give M**2 G = 4 complex settings. B = 200 and
    f = 1/4 give B_p = min(B - 2m, max(2m, floor(B/4))) = 50, pilot counts
    (13, 13, 12, 12) by the quotient-and-remainder rule, and 150 main shots.
    At this seed the pilot gives no fallback reason, so the main counts
    follow the pilot allocation. The final pencil, its reference energy
    H[0, 0] and the contributions come from the main-stage tables only. Saved,
    loaded and reanalyzed at another cutoff, the Result keeps the decision
    without solving the pilot or allocating again.
    """
    from nwqlib.algorithms import PencilSampling
    from nwqlib.algorithms.gcim import pencil_workflow

    method = QFD(initial_state=ingest_product([[2**-.5, 2**-.5]]), time_step=.5, powers=(0, 1),
                 sampling=PencilSampling(total_shots=200, pilot_fraction=.25))
    result = solve(Eigenproblem(A=ingest_pauli((("Z", -1.0),), num_qubits=1)), method=method,
                   execution="quantum", seed=5)
    names = [experiment.name for experiment in result.plan.experiments]
    allocation = result.shot_allocation
    m, total = len(names), 200
    pilot = min(total - 2 * m, max(2 * m, total // 4))
    assert (m, pilot) == (4, 50)
    assert allocation.pilot_counts == tuple(pilot // m + (k < pilot % m) for k in range(m)) == (13, 13, 12, 12)
    main = total - pilot
    assert allocation.main_shots == main and sum(allocation.main_counts) == main
    assert min(allocation.main_counts) >= 2 and allocation.fallback_reason is None

    def stage(value):
        return [c for c in result.data.observations.chunks if {b.parameter: b.value for b in c.bindings}["stage"] == value]

    assert {c.experiment: c.returned_shots for c in stage(0)} == dict(zip(names, allocation.pilot_counts, strict=True))
    assert {c.experiment: c.returned_shots for c in stage(1)} == dict(zip(names, allocation.main_counts, strict=True))
    assert set(result.contribution_ids) == {c.content_id for c in stage(1)}
    (reference,) = [e for e in result.estimates if e.experiment == "group_0_0_real_0"]
    assert set(reference.contribution_ids) <= set(result.contribution_ids)
    assert result.reference_energy == result.pencil.hamiltonian[0][0].real == reference.value

    def fail(*args, **kwargs):
        pytest.fail("loading or reanalysis measured or allocated again")

    path = result.save(tmp_path / "qfd-sampling")
    for name in ("pilot_decision", "solve_pencil", "floored_allocation", "uniform_allocation", "pilot_split",
                 "prepare_experiment", "submit_experiment"):
        monkeypatch.setattr(pencil_workflow, name, fail)
    loaded = nwqlib.load_result(path)
    assert loaded.shot_allocation == allocation and loaded.reference_energy == result.reference_energy
    for source in (result, loaded):
        again = source.analyze(overlap_cutoff=1e-3)
        assert again.shot_allocation == allocation and again.analysis_cutoff == 1e-3


def test_phase_of_the_evolved_state_reaches_the_positive_imaginary_overlap():
    """V|0> = i|0> for A = -Z and tau = pi/2, so the Y quadrature of the joint state reads S_01 = i on every shot.

    S_01 = <0|i0> = i has real part 0 and imaginary part +1, so every shot
    of the overlap group's Y setting returns ancilla sign +1 and its H0
    group, the Z term with coefficient -1, the value -1: both means are
    exact. A wrong Y sign, swapped branches or a dropped controlled phase
    would give -1 or a random sign instead.
    """
    method = QFD(initial_state=ingest_occupation((0,), num_qubits=1), time_step=np.pi / 2, powers=(0, 1))
    result = solve(Eigenproblem(A=ingest_pauli((("Z", -1.0),), num_qubits=1)), method=method,
                   execution="quantum", shots=64, seed=13)
    (imaginary,) = [m for m in result.group_moments if m.pair == (0, 1) and m.quadrature == "imag"]
    assert (imaginary.overlap_mean, imaginary.mean) == (1.0, -1.0)


def test_real_hamiltonian_has_a_complex_pencil_and_real_quadratures_are_refused():
    """A = X is real, but <0|X V|0> = -i sin(tau), so QFD measures both quadratures and refuses real-only.

    With tau = 1 the exact pair entry H0_01 has imaginary part -sin(1) =
    -0.84 and no real part, far outside any rounding. A sampled Plan
    records complex quadratures with the reason that no structural proof
    covers the evolved basis, and explicitly requested real quadratures
    are refused at planning.
    """
    from nwqlib.algorithms.qfd.method import REAL_REASON

    problem = Eigenproblem(A=ingest_pauli((("X", 1.0),), num_qubits=1))
    method = QFD(initial_state=ingest_occupation((0,), num_qubits=1), time_step=1.0, powers=(0, 1))
    result = solve(problem, method=method, execution="quantum", seed=1)
    (entry,) = [p.hamiltonian for p in result.pairs if p.pair == (0, 1)]
    assert entry.imag < -.8 and abs(entry.real) < .01
    sampled = nwqlib.plan(problem, method=method, execution="quantum", shots=8, seed=1)
    acquisition = sampled.reconstruction.pencil.acquisition
    assert (acquisition.quadratures, acquisition.real_proof, acquisition.reason) == ("complex", "unknown", REAL_REASON)
    real = method.revise(acquisition=PencilAcquisition(quadratures="real"))
    with pytest.raises(ApplicabilityError, match="structural real proof"):
        nwqlib.plan(problem, method=real, execution="quantum", shots=8, seed=1)


@pytest.mark.parametrize("rule", ("vacuum_reference", "vacuum_restore"))
def test_fixed_gcim_and_qfd_refuse_the_vacuum_pair_constructions_whenever_they_prepare_joint_states(rule):
    """FixedGCIM and QFD refuse both vacuum constructions at planning, with exact readout and with shots.

    The constructions need each basis state to be a generator chain whose
    exponentials map the all-zero state to itself with a known phase. A
    FixedGCIM basis state is one supplied preparation and the QFD step V
    has no known vacuum phase, so both raise ApplicabilityError whenever
    they prepare joint states. Classical FixedGCIM prepares no joint state
    and keeps the request with no resolved rule.
    """
    from nwqlib.algorithms import FixedGCIM

    request = PencilAcquisition(pair_preparation=rule)
    problem = Eigenproblem(A=ingest_pauli((("Z", 1.0), ("X", .5)), num_qubits=1))
    fixed = FixedGCIM(basis=([1, 0], [0, 1]), acquisition=request)
    qfd = QFD(initial_state=ingest_occupation((0,), num_qubits=1), time_step=.5, powers=(0, 1), acquisition=request)
    for method in (fixed, qfd):
        for shots in (None, 8):
            with pytest.raises(ApplicabilityError):
                nwqlib.plan(problem, method=method, execution="quantum", shots=shots, seed=1)
    acquisition = nwqlib.plan(problem, method=fixed, execution="classical", seed=1).reconstruction.acquisition
    assert (acquisition.requested, acquisition.pair_preparation) == (request, None)


@pytest.mark.parametrize("shots", (None, 64))
def test_identity_offset_moves_every_energy_and_leaves_the_matrix_elements(shots):
    """A and A + 2**20 I give the same circuits, H0 and S, and energies that differ by the offset.

    The offset is removed before V is built and before the solve, so both
    Plans have the same step angles. With exact readout the pair entries
    agree bit for bit, and with 64 shots per setting the same seed gives
    the same counts, so the group moments agree. The offset-free Ritz
    values agree, and each eigenvalue is its offset-free value plus the
    Plan's identity coefficient, rounded once. With exact readout each
    reference energy is its offset-free entry plus that coefficient.
    """
    rows = (("ZI", .5), ("XX", .3), ("IY", -.4))
    method = QFD(initial_state=ingest_occupation((1, 0), num_qubits=2), time_step=.6, powers=(0, 1, 2))
    results = []
    for shift in (.25, .25 + 2.0**20):
        problem = Eigenproblem(A=ingest_pauli((("II", shift), *rows), num_qubits=2))
        results.append(solve(problem, method=method, execution="quantum", shots=shots, seed=2))
    small, large = results
    assert small.plan.reconstruction.step_angles.array.tobytes() == large.plan.reconstruction.step_angles.array.tobytes()
    assert small.pairs if shots is None else small.group_moments
    assert [(p.pair, p.hamiltonian, p.overlap) for p in small.pairs] == [
        (p.pair, p.hamiltonian, p.overlap) for p in large.pairs]
    assert [(m.mean, m.overlap_mean) for m in small.group_moments] == [
        (m.mean, m.overlap_mean) for m in large.group_moments]
    assert small.pencil.offset_free_eigenvalues == large.pencil.offset_free_eigenvalues
    for result, shift in zip(results, (.25, .25 + 2.0**20), strict=True):
        assert result.eigenvalue == result.pencil.offset_free_eigenvalues[0] + shift
        if shots is None:
            (zero,) = [p.hamiltonian for p in result.pairs if p.pair == (0, 0)]
            assert result.reference_energy == zero.real + shift


def test_projected_energy_without_ground_support_is_not_the_ground_energy():
    """For A = diag(-2, 1) and psi = |1>, every power of V keeps |1>, so the Ritz value is 1, not -2.

    The basis spans one direction, so the solve keeps rank 1 and its value
    lies far above the ground energy -2, and the Result states that a
    projected value does not establish the smallest eigenvalue.
    """
    problem = Eigenproblem(A=ingest_pauli((("I", -.5), ("Z", -1.5)), num_qubits=1))
    method = QFD(initial_state=ingest_occupation((1,), num_qubits=1), time_step=.8, powers=(0, 1, 2))
    result = solve(problem, method=method, execution="quantum", seed=4)
    assert result.pencil.kept_rank == 1 and result.eigenvalue > .5
    assert "smallest full-space eigenvalue not established" in str(result)


@pytest.mark.parametrize("fermionic", (False, True), ids=("pauli_terms", "fermionic_blocks"))
def test_identity_operator_returns_its_constant_without_matrices(fermionic):
    """A = 2.5 I needs no measurement: the energy, the reference energy and a zero lowering, with no H or S.

    With the fermionic generator ``B = a_0^dagger a_1 + a_1^dagger a_0``,
    whose Jordan-Wigner image is ``(X_0 X_1 + Y_0 Y_1)/2``, the target still
    needs no measurement and the Result is the same, while the Plan records
    B's table, the words XX and YY with coefficient 1/2 and c_B = 0, and its
    one order-2 hopping block, applied forward and then reversed.
    """
    from nwqlib.operators import ingest_fermion

    problem = Eigenproblem(A=ingest_pauli((("II", 2.5),), num_qubits=2))
    generator = dict(evolution="fermionic_blocks", evolution_terms=ingest_fermion(
        [(1.0, [(0, 1), (1, 0)]), (1.0, [(1, 1), (0, 0)])], num_modes=2)) if fermionic else {}
    method = QFD(initial_state=ingest_occupation((1, 0), num_qubits=2), time_step=.3, powers=(0, 1), **generator)
    plan = nwqlib.plan(problem, method=method, execution="quantum", shots=10, seed=1)
    assert plan.experiments == () and plan.blocks == ()
    if fermionic:
        rec = plan.reconstruction
        assert rec.evolution.rows() == (("II", 0.0), ("XX", .5), ("YY", .5))
        assert [rec.evolution_rows[k][0] for k in rec.step_rows.array.tolist()] == ["XX", "YY", "YY", "XX"]
        assert rec.step_blocks.array.tolist() == [0, 2]
    result = solve(problem, method=method, execution="quantum", shots=10, seed=1)
    assert (result.eigenvalue, result.reference_energy, result.lowering, result.pencil) == (2.5, 2.5, 0.0, None)
    assert "No H or S formed" in str(result)


def test_invalid_power_schedules_are_refused_at_planning():
    """Repeated powers, a schedule without 0 and classical execution are refused before any construction."""
    problem = Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1))
    reference = ingest_occupation((0,), num_qubits=1)
    with pytest.raises(ValueError, match="repeat a power"):
        nwqlib.plan(problem, method=QFD(initial_state=reference, time_step=.1, powers=(0, 1, 1)),
                    execution="quantum", seed=1)
    with pytest.raises(ValueError, match="must include 0"):
        nwqlib.plan(problem, method=QFD(initial_state=reference, time_step=.1, powers=(1, 2)),
                    execution="quantum", seed=1)
    with pytest.raises(ApplicabilityError, match="execution='quantum'"):
        nwqlib.plan(problem, method=QFD(initial_state=reference, time_step=.1, powers=(0, 1)),
                    execution="classical", seed=1)


@pytest.mark.parametrize("shots, calls", ((None, 2000), (10, 3000)))
def test_uses_of_v_in_all_circuits_are_limited_before_the_circuits_are_described(shots, calls, monkeypatch):
    """Powers (0, 1000) call V 1000 times in the diagonal circuit of power 1000 and 1000 times in the pair circuit.

    With exact readout that is 2000 uses. Sampled with the one group of
    A = Z, the off-diagonal pair is measured in two ancilla bases, so
    3000 uses. One use more than `max_admission_steps` refuses the Plan
    before any pair recipe is built and names the count. At a limit equal
    to the count, the uses pass, and the Program check that follows
    refuses the larger Program.

    The same limit bounds the circuit instructions of the distinct
    evolution blocks. For A = X + Z, the powers (0, 1) and r = 1000
    order-2 substeps, one substep rotates about X, Z, Z and X. V has one
    instruction per rotation, 4r in all. Its controlled form wraps the
    controlled RZ of each X rotation in two H, 3 instructions, and controls
    each Z rotation directly, 1 instruction, so 8r. The Plan uses these two
    blocks, 12r = 12000 instructions, with exact readout and with 10 shots
    alike, although the sampled Plan has 8 settings in two groups and two
    quadratures. A limit of 11999 refuses the Plan and names the count,
    12000 admits it, and the reference alone, powers (0,), plans with
    r = 10**12 because it uses no evolution block.
    """
    problem = Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1))
    reference = ingest_occupation((0,), num_qubits=1)

    def forbidden(*args, **kwargs):
        pytest.fail("QFD expanded its pair recipes before refusing the uses of V")

    with monkeypatch.context() as patch:
        patch.setattr("nwqlib.algorithms.qfd.method.power_pair_recipes", forbidden)
        with pytest.raises(ValueError, match=f"call V or its adjoint {calls} times in all"):
            nwqlib.plan(problem, method=QFD(initial_state=reference, time_step=.1, powers=(0, 1000),
                                            max_admission_steps=calls - 1),
                        execution="quantum", shots=shots, seed=1)
    with pytest.raises(ValueError, match="the Program exceeds QFD.max_admission_steps"):
        nwqlib.plan(problem, method=QFD(initial_state=reference, time_step=.1, powers=(0, 1000),
                                        max_admission_steps=calls), execution="quantum", shots=shots, seed=1)

    evolution = Eigenproblem(A=ingest_pauli((("X", 1.0), ("Z", 1.0)), num_qubits=1))
    with pytest.raises(ValueError, match=r"take 12000 circuit instructions in all, exceeding "
                                         r"QFD\.max_admission_steps=11999\."):
        nwqlib.plan(evolution, method=QFD(initial_state=reference, time_step=.1, powers=(0, 1),
                                          steps_per_time_step=1000, max_admission_steps=11999),
                    execution="quantum", shots=shots, seed=1)
    admitted = nwqlib.plan(evolution, method=QFD(initial_state=reference, time_step=.1, powers=(0, 1),
                                                 steps_per_time_step=1000, max_admission_steps=12000),
                           execution="quantum", shots=shots, seed=1)
    assert (len(admitted.reconstruction.pencil.groups), len(admitted.experiments)) == ((2, 8) if shots else (0, 3))
    nwqlib.plan(evolution, method=QFD(initial_state=reference, time_step=.1, powers=(0,), steps_per_time_step=10**12),
                execution="quantum", shots=shots, seed=1)


def test_native_depth_limit_refuses_a_controlled_pair_deeper_than_every_basis_state(
    tmp_path, monkeypatch, ibm_runtime_spy
):
    """A depth limit between the deepest basis-state setting and a complete pair circuit refuses that pair before its job.

    On the offline FakeManilaV2 device the product reference, the
    controlled step and the terminal readout of the pair (0, 1) compile to
    a deeper native two-qubit circuit than any diagonal setting, which
    prepares one basis state and its readout. With the deepest diagonal
    depth as the limit, the run submits the diagonal (0, 0) settings and
    refuses the first deeper pair setting after its preparation and before
    any job, keeping the earlier counts. The depths are those of an
    unlimited run that the same seed compiles identically.
    """
    import importlib.metadata
    from nwqlib.backends.inspection import NativeDepthCapRefusal

    if importlib.metadata.version("qiskit-ibm-runtime") != "0.49.0":
        pytest.skip("IBM native depths are qualified only with qiskit-ibm-runtime 0.49.0")
    spy, shots = ibm_runtime_spy, 8
    problem = Eigenproblem(A=ingest_pauli((("ZZ", 1.0), ("XI", .5)), num_qubits=2))
    reference = ingest_product(np.array([[.6, .8], [1.0, 0.0]]))

    def start(limit, directory):
        selected = nwqlib.plan(problem, method=QFD(initial_state=reference, time_step=.5, powers=(0, 1),
                                                   max_native_two_qubit_depth=limit),
                               execution="quantum", shots=shots, seed=7)
        return selected, nwqlib.prepare(selected, backend=spy.connection, directory=directory, progress=False)

    def finish(prepared):
        run = nwqlib.submit(prepared)
        while run.result is None:
            run.resume()
        return run.result

    selected, prepared = start(None, tmp_path / "unlimited")
    with prepared.run as run:
        assert finish(prepared).missing == ()
        depths = {a.realization.experiment: a.native_depth.native_two_qubit_depth for a in run.prepared_artifacts}
    names = [experiment.name for experiment in selected.experiments]
    diagonal = max(depth for name, depth in depths.items() if name.split("_")[1] == name.split("_")[2])
    first = next(index for index, name in enumerate(names) if depths[name] > diagonal)
    assert names[first].startswith("group_0_1_")
    requests = len(spy.api.requests)
    selected, prepared = start(diagonal, tmp_path / "limited")
    with prepared.run as run:
        with pytest.raises(NativeDepthCapRefusal) as refused:
            finish(prepared)
        assert (refused.value.experiment, refused.value.cap, refused.value.depth) == (
            names[first], diagonal, depths[names[first]])
        assert len(spy.api.requests) == requests + first
        assert run.exposure["completed"]["shots"] == first * shots
        partial = selected.method.analyze(selected, run.data, settings={})
        assert partial.missing == tuple(names[first:]) and partial.pencil is None


def test_exact_pencil_stays_above_the_ground_energy_while_a_perturbed_one_need_not():
    """The Ritz value of exact, consistent H and S of the approximate fixed-V basis bounds the ground energy from above.

    A = 0.5 ZI + 0.3 XX - 0.4 IY with the reference |00> and the powers
    (0, 1) of an order-2 step: the exact pair entries come from one
    normalized trial space, so the lowest Ritz value lies above the
    smallest eigenvalue of A, here by more than 0.01, although V only
    approximates exp(-i tau A). Lowering H_00 by twice that gap over
    |c_0|**2, for the lowest S-normalized Ritz vector c, is a signed
    perturbation of the kind sampling noise can produce. With c as trial
    vector of the same kept subspace the perturbed Rayleigh quotient is the
    ground energy less the gap, so the solve returns a raw value below the
    ground energy without refusing the pencil.
    """
    from nwqlib.algorithms.gcim.fixed_basis import solve_pencil

    rows = (("ZI", .5), ("XX", .3), ("IY", -.4))
    ground = float(np.linalg.eigvalsh(sum(c * _pauli(label) for label, c in rows))[0])
    method = QFD(initial_state=ingest_occupation((0, 0), num_qubits=2), time_step=.6, powers=(0, 1))
    result = solve(Eigenproblem(A=ingest_pauli(rows, num_qubits=2)), method=method, execution="quantum", seed=6)
    gap = result.eigenvalue - ground
    assert gap > .01
    pencil = result.pencil
    h = np.array([[complex(z.real, z.imag) for z in row] for row in pencil.hamiltonian])
    s = np.array([[complex(z.real, z.imag) for z in row] for row in pencil.overlap])
    lead = pencil.coefficients[0]
    h[0, 0] -= 2 * gap / abs(complex(lead.real, lead.imag)) ** 2
    perturbed = solve_pencil(h, s, cutoff=1e-12, sampled=True)
    assert perturbed.failure_reason is None and perturbed.eigenvalues[0] < ground


def _rotation_gates(circuit):
    """``(name, params)`` of every rotation of a lowered circuit, in order."""
    return [(name, params) for name, params, _ in _primitive_gates(circuit) if name in ROTATIONS]


def test_fermionic_generator_shapes_v_while_h_s_and_the_energies_belong_to_the_target(tmp_path):
    """V comes from B = 5 + n_0, while H, S and every energy come from A = 3 I + X, also after a reload.

    ``n_0 = a_0^dagger a_0 = (I - Z)/2``, so ``B0 = -Z/2`` and c_B = 5.5,
    and V is a Z rotation whatever the time step. For psi = |+> and the
    powers (0, 1) the two basis states span the qubit when the total stored
    angle alpha of V is not a multiple of 2 pi, so the centered Ritz values
    are the eigenvalues -1 and 1 of X and the physical ones 2 and 4, with
    the reference energy <+|A|+> = 4. tau = 2.5 gives two order-2 factors
    of angle -1.25 each, alpha = -2.5, and the off-diagonal overlap
    cos(alpha/2) has magnitude below 1/2, since pi/3 < 1.25 < 2 pi/3, so
    the ideal overlap's smallest eigenvalue is at least 1/2. Each centered
    Ritz value lies within its allowance from the qualified entry bounds of
    the pair records and the returned coefficients, and each physical value
    is the centered one plus c_I = 3, rounded once. The reference energy
    is the pair's H0_00, within its bound of 1, plus 3. The controlled V of
    the pair circuit is two CRZ gates. Saved and reloaded, the rebuilt
    blocks are the planned definitions and the pair circuit has the same
    gates: rebuilding from A's X row would give X rotations, and adding
    c_B would move every energy by 5.5. The Plan states the exact upper
    bounds ``delta_gen_0 = |1 - 0| + |0 - (-0.5)| = 1.5`` over X and Z and
    ``delta_gen_full = |3 - 5.5| + 1.5 = 4.0``, and no particle-number
    sector, since |+> is a product of superpositions.
    """
    from nwqlib.operators import ingest_fermion

    a = ingest_pauli((("I", 3.0), ("X", 1.0)), num_qubits=1)
    b = ingest_fermion([(5.0, []), (1.0, [(0, 1), (0, 0)])], num_modes=1)
    method = QFD(initial_state=ingest_product([[2**-0.5, 2**-0.5]]), time_step=2.5, powers=(0, 1),
                 evolution="fermionic_blocks", evolution_terms=b)
    result = solve(Eigenproblem(A=a), method=method, execution="quantum", seed=7)
    generator, sector = result.plan.assumptions[1:3]
    assert "delta_gen_0 = sum_{P != I} |a_P - b_P| = 1.5 >= ||A0 - B0||" in generator
    assert "delta_gen_full = |c_I - c_B| + delta_gen_0 = 4.0 >= ||A - B||" in generator
    assert sector == ("no particle-number sector is stated for the basis, because a product reference has an "
                      "established particle number only as an occupation or a product of computational basis states")
    rec = result.plan.reconstruction
    assert rec.evolution.rows() == (("I", 5.5), ("Z", -.5)) and rec.step_angles.array.tolist() == [-1.25, -1.25]
    assert rec.step_blocks.array.tolist() == [0, 1]
    assert _rotation_gates(_lowered(result.plan, "pair_0_1")) == [("crz", (-1.25,)), ("crz", (-1.25,))]

    pencil = result.pencil
    h, s, eh, es = _pencil_inputs(result)
    columns = [[row[k] for row in pencil.coordinate_vectors] for k in range(2)]
    centered = pencil.offset_free_eigenvalues
    allowance = _ritz_allowance(h, s, eh, es, columns, centered, sigma=F(1, 2), magnitude=1, norm=1)
    assert pencil.kept_rank == 2
    for value, ideal, physical in zip(centered, (-1, 1), pencil.eigenvalues, strict=True):
        assert abs(F(value) - ideal) <= allowance
        assert physical == float(F(value) + 3)
    (zero,) = [p for p in result.pairs if p.pair == (0, 0)]
    assert abs(F(zero.hamiltonian.real) - 1) <= F(zero.hamiltonian_bound)
    assert result.reference_energy == float(F(zero.hamiltonian.real) + 3) and result.eigenvalue == pencil.eigenvalues[0]

    loaded = nwqlib.load_result(result.save(tmp_path / "fermionic"))
    again = loaded.plan
    assert again.method.evolution_terms.reference.identity == b.reference.identity
    assert again.method.content_id == method.content_id and again.reconstruction.content_id == rec.content_id
    assert tuple(block.record for block in again.blocks) == again.construction.selections
    assert _rotation_gates(_lowered(again, "pair_0_1")) == _rotation_gates(_lowered(result.plan, "pair_0_1"))
    reanalyzed = loaded.analyze(overlap_cutoff=1e-6)
    assert (reanalyzed.pencil.hamiltonian, reanalyzed.pencil.overlap) == (pencil.hamiltonian, pencil.overlap)
    assert reanalyzed.eigenvalue == result.eigenvalue and reanalyzed.reference_energy == result.reference_energy


def test_scalar_generator_gives_copies_of_the_reference_and_the_target_energy():
    """B = 2 has no block, so V = I, every basis state is psi = |0> and the energy is <0|A|0> = 1.5 for A = 0.5 + Z.

    With the powers (0, 1, 2) the Plan has no factor, its evolution blocks
    build no gate and count no instruction, and every circuit still
    prepares the reference and measures A. The ideal pencil is
    ``S* = J = 11^dagger`` and ``H0* = k J`` with k = <0|Z|0> = 1. The
    returned kept rank is compared with an independent count of the overlap
    eigenvalues above the actual cutoff tau. The overlap s is exactly
    Hermitian, by the conjugate completion of its assembly, and its exact
    largest row sum ``distance`` of ``|Re(s_ij) - 1| + |Im(s_ij)|`` bounds
    ``||s - J||_2``. By Weyl two eigenvalues of s lie in
    ``[-distance, distance]`` and one in ``[3 - distance, 3 + distance]``,
    so ``distance < tau < 3 - distance`` leaves exactly one eigenvalue above
    tau. This counts eigenvalues above the cutoff. It does not show that s
    has rank one, and it does not bound the error of the computed
    eigenvalues. Every column c with ``sum_i c_i != 0`` has the
    Rayleigh quotient k in the ideal pencil, so the returned column gives
    the allowance of the centered value about k, and the physical value is
    it plus c_I = 0.5, rounded once. The constant 2 of B enters no energy.
    """
    from nwqlib._projected_eigensolver import entrywise_gram_allowance
    from nwqlib.operators import ingest_fermion

    method = QFD(initial_state=ingest_occupation((0,), num_qubits=1), time_step=.3, powers=(0, 1, 2),
                 evolution="fermionic_blocks", evolution_terms=ingest_fermion([(2.0, [])], num_modes=1))
    result = solve(Eigenproblem(A=ingest_pauli((("I", .5), ("Z", 1.0)), num_qubits=1)), method=method,
                   execution="quantum", seed=7)
    rec = result.plan.reconstruction
    assert rec.evolution.rows() == (("I", 2.0),) and rec.step_rows.array.size == rec.step_blocks.array.size == 0
    evolution = [block.record for block in result.plan.blocks if block.record.implementation.name == "qfd.pauli_evolution"]
    assert evolution and sum(record.construction_work for record in {r.content_id: r for r in evolution}.values()) == 0
    assert _rotation_gates(_lowered(result.plan, "pair_0_2")) == []

    pencil = result.pencil
    h, s, eh, es = _pencil_inputs(result)
    assert pencil.kept_rank == 1
    # The exact rational largest row sum of s - J bounds ||s - J||_2 because
    # s is Hermitian, so by Weyl the eigenvalues of s lie within distance of
    # {3, 0, 0}. distance < tau < 3 - distance leaves exactly one of them
    # above the actual cutoff tau, the count that kept_rank must match.
    distance = max(sum(abs(F(z.real) - 1) + abs(F(z.imag)) for z in row) for row in s)
    tau = F(pencil.overlap_cutoff)
    assert distance <= F(entrywise_gram_allowance(es))
    assert distance < tau < 3 - distance
    (column,) = [[row[0] for row in pencil.coordinate_vectors]]
    a = sum(F(z.real) ** 2 + F(z.imag) ** 2 for z in column)
    total = (sum(F(z.real) for z in column), sum(F(z.imag) for z in column))
    sigma = total[0] ** 2 + total[1] ** 2
    assert sigma >= F(1, 2)
    (centered,) = pencil.offset_free_eigenvalues
    allowance = _ritz_allowance(h, s, eh, es, [column], [centered], sigma=sigma, magnitude=1, norm=1, scale=a)
    assert abs(F(centered) - 1) <= allowance
    assert result.eigenvalue == float(F(centered) + F(.5))


def test_fermionic_blocks_reproduce_their_ladder_operators_and_v_reproduces_the_pair_entries():
    """Every stored block of a four-mode B is a complete Hermitian component of B with the exact angle of its amplitude.

    B = 0.7 + 0.5 n_1 + 1e-13 n_3 + g a_0^dagger a_2 + conj(g) a_2^dagger a_0
    - 0.25 a_3^dagger a_1^dagger a_3 a_1 + 0.15 (a_2^dagger a_1^dagger a_1 a_0
    + h.c.) + 0.1 (a_3^dagger a_2^dagger a_1 a_0 + h.c.) with g = 0.3 + 0.2i
    has two number terms, the second far below 1e-12 so that only a
    normal ordering at cutoff 0 keeps it, a hopping term whose parity
    string crosses the occupied mode 1 of the reference |0011>, a density
    term ``-0.25 a_3^dagger a_1^dagger a_3 a_1 = 0.25 n_3 n_1``, a hopping
    term times the spectator number n_1, a double excitation, and a complex
    pair with the two components ``0.3 (T + T^dagger)`` and
    ``0.2 i (T - T^dagger)``, seven blocks. Each component's matrix M
    follows from the action of the ladder operators on occupation states,
    without a Pauli mapping, and its Pauli coefficients ``tr(P M)/16`` are
    exact. Every nonidentity one has the magnitude 2**-kappa of its family
    (1, 1, 1, 1, 2, 2, 3). Each forward block of the order-2 substep holds
    exactly the words of one component with the angles
    ``RN(2 s val(u) c_P)``, s half the substep, so that
    ``sum_k (theta_k/theta_b) 2**-kappa P_k + c_0 I = M``. The reverse half
    repeats the blocks in reverse order, each whole and with its words
    reversed, at the positions of `step_blocks`. B's table holds every word
    with the summed coefficient of B and the identity row c_B, each rounded
    once from exact sums. The pair entry H0_01 and S_01 of A on the basis
    of V lie within their qualified bounds plus the error of multiplying
    the stored rotations independently.
    """
    from nwqlib.operators import ingest_fermion

    g = .3 + .2j
    words = {"number": [(1, 1), (1, 0)], "tiny": [(3, 1), (3, 0)], "hop": [(0, 1), (2, 0)],
             "density": [(3, 1), (1, 1), (3, 0), (1, 0)], "spectator": [(2, 1), (1, 1), (1, 0), (0, 0)],
             "double": [(3, 1), (2, 1), (1, 0), (0, 0)]}

    def adjoint(ops):
        return [(mode, 1 - action) for mode, action in reversed(ops)]

    rows = [(.7, []), (.5, words["number"]), (1e-13, words["tiny"]), (g, words["hop"]),
            (g.conjugate(), adjoint(words["hop"])), (-.25, words["density"]), (.15, words["spectator"]),
            (.15, adjoint(words["spectator"])), (.1, words["double"]), (.1, adjoint(words["double"]))]
    ladders = _annihilators(4)

    def matrix(name):
        return _ladder_word(ladders, words[name])

    components = [(.5, 1, matrix("number")), (1e-13, 1, matrix("tiny")), (.3, 1, matrix("hop") + matrix("hop").T),
                  (.2, 1, 1j * (matrix("hop") - matrix("hop").T)), (-.25, 2, matrix("density")),
                  (.15, 2, matrix("spectator") + matrix("spectator").T), (.1, 3, matrix("double") + matrix("double").T)]
    a = ingest_pauli((("IIII", .1), ("IIZZ", .4), ("XXII", .3), ("YYII", .3), ("IXZX", .2), ("IYZY", .2),
                      ("ZIII", -.6)), num_qubits=4)
    reference = ingest_occupation((1, 1, 0, 0), num_qubits=4)
    method = QFD(initial_state=reference, time_step=.6, powers=(0, 1), evolution="fermionic_blocks",
                 evolution_terms=ingest_fermion(rows, num_modes=4))
    result = solve(Eigenproblem(A=a), method=method, execution="quantum", seed=5)
    rec = result.plan.reconstruction
    s = F(rec.substep) / 2
    expected, shift, table = [], F(.7), {}
    for amplitude, kappa, m in components:
        decomposition = _pauli_decomposition(m)
        assert all(im == 0 for _, im in decomposition.values())
        assert {abs(re) for label, (re, _) in decomposition.items() if set(label) != {"I"}} == {F(1, 2**kappa)}
        identity = decomposition.get("IIII", (F(0), F(0)))[0]
        shift += F(amplitude) * identity
        angles = {label: float(2 * s * F(amplitude) * re) for label, (re, _) in decomposition.items() if label != "IIII"}
        expected.append((amplitude, kappa, m, identity, angles))
        for label, (re, _) in decomposition.items():
            table[label] = table.get(label, F(0)) + F(amplitude) * re
    table["IIII"] = shift
    assert rec.evolution.rows() == tuple((label, float(table[label])) for label in sorted(table))

    labels = [rec.evolution_rows[k][0] for k in rec.step_rows.array.tolist()]
    angles = rec.step_angles.array.tolist()
    starts = [*rec.step_blocks.array.tolist(), len(labels)]
    blocks = [list(zip(labels[i:j], angles[i:j], strict=True)) for i, j in zip(starts, starts[1:])]
    assert len(blocks) == 2 * len(expected)
    forward, reverse = blocks[:len(expected)], blocks[len(expected):]
    assert reverse == [block[::-1] for block in reversed(forward)]
    unmatched = list(expected)
    for block in forward:
        matches = [item for item in unmatched if item[4] == dict(block)]
        assert len(matches) == 1, block
        match = matches[0]
        unmatched.remove(match)
        amplitude, kappa, m, identity, _ = match
        theta = float(2 * s * F(amplitude) / 2**kappa)
        leaves = sum((angle / theta) * _pauli(label) for label, angle in block)
        assert np.array_equal(leaves + 2**kappa * identity * np.eye(16), 2**kappa * m)
    assert unmatched == []

    psi = np.zeros(16)
    psi[3] = 1
    states, errors = _oracle(result.plan, psi)
    terms = [(F(c.real), label) for label, c in a.pauli_terms().labels() if label != "IIII"]
    mass = sum(abs(c) for c, _ in terms)
    (pair,) = [p for p in result.pairs if p.pair == (0, 1)]
    for measured, bound, matrix_terms, scale in ((pair.hamiltonian, pair.hamiltonian_bound, terms, mass),
                                                 (pair.overlap, pair.overlap_bound, [(F(1), "IIII")], 1)):
        exact = _exact_entry(states[0], matrix_terms, states[1])
        difference = (F(measured.real) - exact[0], F(measured.imag) - exact[1])
        allowance = F(bound) + scale * (errors[0] + errors[1] + errors[0] * errors[1])
        assert difference[0] ** 2 + difference[1] ** 2 <= allowance ** 2


def test_complete_fermionic_blocks_keep_the_particle_number_that_the_pauli_label_order_breaks():
    """V|psi> from complete blocks stays in the one-particle sector, while the same terms in label order leave it.

    B = 0.5 (a_1^dagger a_2 + a_2^dagger a_1) + 0.5 n_1 (a_2^dagger a_0 + a_0^dagger a_2),
    the second term written as ``a_2^dagger a_1^dagger a_1 a_0`` and its
    adjoint, conserves the particle number. Its Jordan-Wigner image is
    A = 0.25 (XXI + YYI) + 0.125 (XZX + YZY - XIX - YIY), since
    ``n_1 = (I - Z_1)/2`` and Z_1 squares to I. The Pauli route orders the
    words by label, XIX, XXI, XZX, YIY, YYI, YZY, so in its order-1 step the
    rotations of the hopping block {XXI, YYI} alternate with those of the
    spectator block {XIX, XZX, YIY, YZY}, and XXI anticommutes with XZX, so
    the order is not that of whole blocks. The fermionic route applies the
    hopping block, then the spectator block. From the occupation reference
    |100> (mode 2 occupied) and the time step 3, the basis state of power 1
    is built by multiplying the primitive gates of the lowered diagonal
    circuit independently, with its norm-error bound delta. The weight
    outside N = 1 is at most delta**2 for the fermionic route, the weight of
    the exact state being 0. For the Pauli route its square root exceeds
    0.1 + delta, so the exact state of that route has weight at least 0.01
    outside the sector.
    """
    from nwqlib.operators import ingest_fermion

    a = ingest_pauli((("XXI", .25), ("YYI", .25), ("XZX", .125), ("YZY", .125), ("XIX", -.125), ("YIY", -.125)),
                     num_qubits=3)
    spectator = [(2, 1), (1, 1), (1, 0), (0, 0)]
    b = ingest_fermion([(.5, [(1, 1), (2, 0)]), (.5, [(2, 1), (1, 0)]), (.5, spectator),
                        (.5, [(0, 1), (1, 1), (1, 0), (2, 0)])], num_modes=3)
    reference = ingest_occupation((0, 0, 1), num_qubits=3)
    weights = {}
    for route, settings in (("pauli_terms", {}), ("fermionic_blocks", dict(evolution_terms=b))):
        method = QFD(initial_state=reference, time_step=3.0, powers=(0, 1), formula_order=1, evolution=route,
                     **settings)
        plan = nwqlib.plan(Eigenproblem(A=a), method=method, execution="quantum", seed=2)
        rec = plan.reconstruction
        order = [rec.evolution_rows[k][0] for k in rec.step_rows.array.tolist()]
        state, delta = _simulate(_lowered(plan, "pair_1_1"))
        weights[route] = _outside_weight(state, lambda i: bin(i >> 1).count("1") == 1), delta
        if route == "pauli_terms":
            assert order == ["XIX", "XXI", "XZX", "YIY", "YYI", "YZY"]
        else:
            assert order == ["XXI", "YYI", "XIX", "XZX", "YIY", "YZY"] and rec.step_blocks.array.tolist() == [0, 2]
    leaked, delta = weights["pauli_terms"]
    assert leaked >= (F(1, 10) + delta) ** 2
    kept, delta = weights["fermionic_blocks"]
    assert kept <= delta**2


def _chemistry_data(one_body, two_body, hamiltonian):
    """Two-orbital chemistry data with occupations (1, 1, 0, 0), built directly without an SCF calculation."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.algorithms.gcim.chemistry import GCIMChemistryProblemData

    return GCIMChemistryProblemData(
        hamiltonian=SparsePauliOp.from_list(hamiltonian), reference_preparation=QuantumCircuit(4),
        reference_occupations=(1, 1, 0, 0), n_spatial_orbitals=2, num_qubits=4, num_electrons=2,
        nuclear_repulsion_energy=.7, constant_energy=.7, rhf_energy=-1.0, one_body_integrals=one_body,
        two_body_integrals=two_body,
        metadata={"name": "two-orbital model", "integral_convention": "MO two-body tensor h[p,q,r,s] = (ps|qr)"})


def _upward(value):
    """The smallest binary64 value at or above an exact rational."""
    result = float(value)
    return result if F(result) >= value else float(np.nextafter(result, np.inf))


def test_chemistry_generator_projects_the_stored_one_body_pair_and_qfd_keeps_the_target(tmp_path):
    """The helper gives the stored one-body pair (0.05, -0.03) its exact midpoint, and QFD still measures the target.

    With mode 2p + sigma, the hopping pair between modes sigma and
    2 + sigma takes ``m = RN((0.05 - 0.03)/2)`` on both sides for each spin
    sigma, while the diagonal entries, the two-body terms from v[0,0,0,0] = 0.25 and
    v[1,1,1,1] = 0.125 and E_c = 0.7 stay. Two pairs change, none is
    removed, and with ``x = 0.05``, ``y = -0.03``, ``q = (x + y)/2`` the
    corrections are ``Delta_sym = 2|x - y|``, ``Delta_round = 4|m - q|``,
    ``sum |b - g| = 2(|m - x| + |m - y|)``, the asymmetry ``|x - y|`` and the
    one-body correction ``|m - x| + |m - y|`` with its bound twice that,
    each rounded up once. c_B is E_c plus each kept coefficient times the
    identity part ``tr(M)/16`` of its ladder matrix. The factor 1/2 is
    applied once: v[0,0,0,0] gives the two spin terms
    ``0.125 a_0^dagger a_1^dagger a_1 a_0`` and
    ``0.125 a_1^dagger a_0^dagger a_0 a_1``, each equal to
    ``-0.125 a_1^dagger a_0^dagger a_1 a_0`` in normal order, so that
    density monomial has -0.25, and v[1,1,1,1] gives -0.125 on modes 3 and 2.
    The cutoff 0.0125 removes both spins' pairs, ``|m|`` being about 0.01,
    and keeps the rest, the four number and two density terms, which are
    the six blocks that the spin statement counts. With the Coulomb entry
    v[0,1,1,0] = 1e-12 added, the four densities
    ``a_{2+tau}^dagger a_sigma^dagger a_{2+tau} a_sigma`` take
    ``-0.5e-12`` each, far below 1e-12. Cutoff 0 keeps them, and 0.0125
    removes them with the two pairs, six in all. That entry stays out of
    the generator that QFD uses below, because its four blocks would raise
    the simulated circuit above the 64 gates of the T4 bound. No call
    changes the integrals or the target. QFD with A, whose identity
    coefficient -0.3 differs from c_B, adds -0.3 once to the centered Ritz
    value and to the reference entry H0_00, which lies within its bound of
    <0011|A0|0011> = -0.9. The Plan states that the basis lies in the
    sector N = 2 of the occupation reference, and the order-1 basis state
    of power 1, built by independent primitive gates from the lowered
    circuit, has weight at most delta**2 outside N = 2 and Sz = 0. Saved and
    reloaded, the Method keeps the diagnostics and B's content hash, and
    diagnostics with another B are refused.
    """
    from nwqlib.algorithms.qfd import chemistry_evolution_terms

    x, y = .05, -.03
    h = np.array([[-1.25, x], [y, -.5]])
    v = np.zeros((2, 2, 2, 2))
    v[0, 0, 0, 0], v[1, 1, 1, 1] = .25, .125
    hamiltonian = [("IIII", -.3), ("IIIZ", .4), ("IIZI", .4), ("ZIZI", .1), ("IXZX", .05), ("IYZY", .05)]
    data = _chemistry_data(h, v, hamiltonian)
    before = (h.copy(), v.copy(), data.hamiltonian.copy())
    terms, diagnostics = chemistry_evolution_terms(data)
    fermions = terms.fermion_terms()

    def rows(generator):
        stored = generator.fermion_terms()
        offsets = stored.offsets.tolist()
        return {tuple(zip(stored.modes[i:j].tolist(), stored.actions[i:j].tolist(), strict=True)): c
                for i, j, c in zip(offsets[:-1], offsets[1:], stored.coefficients.tolist(), strict=True)}

    table = rows(terms)
    q = (F(x) + F(y)) / 2
    m = float(q)
    assert m not in (x, y) and table[()] == .7
    for sigma in (0, 1):
        assert table[(sigma, 1), (2 + sigma, 0)] == table[(2 + sigma, 1), (sigma, 0)] == m
        assert table[(sigma, 1), (sigma, 0)] == -1.25 and table[(2 + sigma, 1), (2 + sigma, 0)] == -.5
    assert table[(1, 1), (0, 1), (1, 0), (0, 0)] == -.25 and table[(3, 1), (2, 1), (3, 0), (2, 0)] == -.125
    ladders = _annihilators(4)
    shift = sum((F(c.real) * F(int(np.trace(_ladder_word(ladders, ops)))) / 16 for ops, c in table.items()), F(0))
    gap = abs(F(m) - F(x)) + abs(F(m) - F(y))
    assert (diagnostics.changed_pairs, diagnostics.dropped_pairs, diagnostics.dropped_mass) == (2, 0, 0.0)
    assert diagnostics.symmetrization_correction == _upward(2 * abs(F(x) - F(y)))
    assert diagnostics.rounding_correction == _upward(4 * abs(F(m) - q))
    assert diagnostics.raw_difference == _upward(2 * gap)
    assert diagnostics.max_one_body_asymmetry == _upward(abs(F(x) - F(y)))
    assert (diagnostics.one_body_correction, diagnostics.one_body_correction_bound) == (_upward(gap), _upward(2 * gap))
    assert diagnostics.generator_shift == float(shift) and diagnostics.evolution_input == terms.reference.identity
    assert "N = 2 and Sz = 0.0" in diagnostics.spin_statement
    cut, cut_diagnostics = chemistry_evolution_terms(data, evolution_coefficient_cutoff=.0125)
    cut_fermions = cut.fermion_terms()
    assert len(cut_fermions) == len(fermions) - 4 and cut_fermions.coefficients[0] == .7
    assert (cut_diagnostics.dropped_pairs, cut_diagnostics.dropped_mass) == (2, _upward(4 * abs(F(m))))
    # A monomial that annihilates exactly the modes it creates is self-adjoint
    # and one block of B. The other nonscalar monomials form adjoint pairs,
    # one block per pair.
    cut_table = rows(cut)
    own = sum({mode for mode, action in key if action} == {mode for mode, action in key if not action}
              for key in cut_table if key)
    count = own + (len(cut_table) - 1 - own) // 2
    assert count == 6 and f"{count} of {count} pairs of adjoint monomials" in cut_diagnostics.spin_statement
    coulomb = v.copy()
    coulomb[0, 1, 1, 0] = 1e-12
    densities = [((2 + tau, 1), (sigma, 1), (2 + tau, 0), (sigma, 0)) for sigma in (0, 1) for tau in (0, 1)]
    kept = rows(chemistry_evolution_terms(_chemistry_data(h, coulomb, hamiltonian))[0])
    removed, removed_diagnostics = chemistry_evolution_terms(_chemistry_data(h, coulomb, hamiltonian),
                                                             evolution_coefficient_cutoff=.0125)
    assert all(kept.get(key) == -.5 * 1e-12 for key in densities)
    assert not set(densities) & set(rows(removed)) and removed_diagnostics.dropped_pairs == 6
    assert np.array_equal(h, before[0]) and np.array_equal(v, before[1]) and data.hamiltonian == before[2]

    method = QFD(initial_state=ingest_occupation((1, 1, 0, 0), num_qubits=4), time_step=.5, powers=(0, 1),
                 formula_order=1, evolution="fermionic_blocks", evolution_terms=terms,
                 evolution_diagnostics=diagnostics)
    result = solve(data.eigenproblem(), method=method, execution="quantum", seed=3)
    assert result.plan.assumptions[2].startswith("the basis lies in the particle-number sector N = 2 of the reference")
    assert result.eigenvalue == float(F(result.pencil.offset_free_eigenvalues[0]) + F(-.3))
    (zero,) = [p for p in result.pairs if p.pair == (0, 0)]
    assert abs(F(zero.hamiltonian.real) - F(-.9)) <= F(zero.hamiltonian_bound)
    assert result.reference_energy == float(F(zero.hamiltonian.real) + F(-.3))

    def sector(index):
        system = index >> 1
        alpha, beta = bin(system & 0b0101).count("1"), bin(system & 0b1010).count("1")
        return alpha + beta == 2 and alpha == beta

    state, delta = _simulate(_lowered(result.plan, "pair_1_1"))
    assert _outside_weight(state, sector) <= delta**2

    loaded = nwqlib.load_result(result.save(tmp_path / "chemistry"))
    assert loaded.plan.method.evolution_diagnostics == diagnostics
    assert loaded.plan.method.evolution_terms.reference.identity == terms.reference.identity
    with pytest.raises(ValueError, match="evolution_diagnostics belong"):
        method.revise(evolution_terms=cut)


def test_fermionic_generators_outside_the_block_domain_are_refused_at_planning():
    """A pair whose coefficients differ by one ulp, a word with two creations and no annihilation, a mode count other than A's qubit count and a requested Problem sector are refused before any construction.

    The admission compares exact values, so 0.1 and the next binary64 above
    it are not conjugate. The sector refusal precedes the choice of route,
    so checking it on the fermionic route covers both routes.
    """
    from nwqlib.operators import ingest_fermion

    problem = Eigenproblem(A=ingest_pauli((("ZI", 1.0), ("XX", .5)), num_qubits=2))
    reference = ingest_occupation((1, 0), num_qubits=2)
    for rows, modes, message in (
            ([(.1, [(0, 1), (1, 0)]), (float(np.nextafter(.1, 1)), [(1, 1), (0, 0)])], 2, "not exactly Hermitian"),
            ([(1.0, [(1, 1), (0, 1)])], 2, "accepts scalars"),
            ([(1.0, [(0, 1), (0, 0)])], 3, "3 modes and A has 2 qubits")):
        method = QFD(initial_state=reference, time_step=.2, powers=(0, 1), evolution="fermionic_blocks",
                     evolution_terms=ingest_fermion(rows, num_modes=modes))
        with pytest.raises(ValueError, match=message):
            nwqlib.plan(problem, method=method, execution="quantum", seed=1)
    sector = Eigenproblem(A=ingest_pauli((("ZI", 1.0), ("XX", .5)), num_qubits=2), sector="N = 1")
    method = QFD(initial_state=reference, time_step=.2, powers=(0, 1), evolution="fermionic_blocks",
                 evolution_terms=ingest_fermion([(1.0, [(0, 1), (0, 0)])], num_modes=2))
    with pytest.raises(ApplicabilityError, match="accepts no requested Problem sector"):
        nwqlib.plan(sector, method=method, execution="quantum", seed=1)
