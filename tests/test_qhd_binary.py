"""Binary QHD encoding: the circuit against independent product oracles, the signed square, CX laws and the AQFT bound."""

import itertools
import math
from fractions import Fraction

import mpmath
import numpy as np
import pytest
import scipy.linalg
import sympy as sp

from nwqlib._validation import GATE_ENTRY_ERROR, UNIT_ROUNDOFF
from nwqlib.algorithms.qhd import QHD, BinarySynthesis, QHDVerification, QuadraticSchedule
from nwqlib.algorithms.qhd.initial_state import GaussianState, KineticGroundState, UniformState
from nwqlib.algorithms.qhd.binary import PhaseTable, signed_square_walsh, walsh_sum
from nwqlib.algorithms.qhd.method import _grid
from nwqlib.algorithms.qhd.native import append_binary_steps, append_diagonal, construct_qhd
from nwqlib.problems import Optimization
from nwqlib.scientist import plan, solve

GAMMA = 0.2


def _problem(k=4):
    """Two variables on unequal periodic boxes with an asymmetric coupled objective and a constant.

    Both boxes contain the origin, so planning expands about zero and forms
    the tables {x}: x**2/3, {y}: y/7 and {x, y}: -x y/5 and the constant 1/9
    (``objective.expansion_centers``). The coupling is not symmetric under
    x <-> y and the box lengths 1.7 and 2.2 differ, so a transposed table or
    a swapped variable axis changes the operator.
    """
    x, y = sp.symbols("x y", real=True)
    return Optimization(objective=x**2 / 3 - x * y / 5 + y / 7 + sp.Rational(1, 9), variables=(x, y),
                        bounds=((0.0, 1.7), (-1.0, 1.2)))


def _method(k, **fields):
    # The oracles start from the uniform state, so the Method names it unless a test sets its own start.
    fields.setdefault("initial_state", UniformState())
    return QHD(encoding="binary", boundary="periodic", num_grid_points=k, num_steps=2,
               total_time=0.45, schedule=QuadraticSchedule(gamma=GAMMA), **fields)


def _register_permutation(d, k):
    """Return Qiskit's index ``z = sum_j n_j K**j`` of each lexicographic position ``i = sum_j n_j K**(d-1-j)``."""
    return [sum(n << (j * (k.bit_length() - 1)) for j, n in enumerate(indices))
            for indices in itertools.product(range(k), repeat=d)]


def _oracle(problem, method):
    """Return the product operator of the finite periodic model in lexicographic order, written independently.

    Coordinates ``x_i = lower + i L/K``. The potential diagonal evaluates the
    objective, constant included, at every grid tuple. The kinetic factor of
    each variable is ``expm(-i alpha K_j)`` of the stencil
    ``(2I - P - P^T)/(2 h**2)`` for finite difference, and
    ``F^dagger diag(exp(-i alpha 2 pi**2 q**2/L**2)) F`` with the analytic
    Fourier matrix ``F[k, n] = exp(-2 pi i k n/K)/sqrt(K)`` and the signed
    indices ``q = k`` below K/2 and ``k - K`` above for the spectral model.
    Variable 0 is the most significant Kronecker factor. Step k uses the
    midpoint weights ``a = 1/(1 + gamma t**2)`` and ``b = 1 + gamma t**2`` at
    ``t = (k + 1/2) dt``, first order ``U_K(dt a) exp(-i dt b V)`` and second
    order ``exp(-i dt b V/2) U_K(dt a) exp(-i dt b V/2)``.
    """
    k, d = method.num_grid_points, len(problem.variables)
    grids = [[lower + i * (upper - lower) / k for i in range(k)] for lower, upper in problem.bounds]
    evaluate = sp.lambdify(problem.variables, problem.objective)
    potential = np.array([evaluate(*(grids[j][n] for j, n in enumerate(t)))
                          for t in itertools.product(range(k), repeat=d)])

    def kinetic(alpha, lower, upper):
        length = upper - lower
        if method.kinetic_model == "finite_difference":
            h = length / k
            shift = np.roll(np.eye(k), 1, axis=0)
            return scipy.linalg.expm(-1j * alpha * (2 * np.eye(k) - shift - shift.T) / (2 * h * h))
        fourier = np.exp(-2j * np.pi * np.outer(range(k), range(k)) / k) / np.sqrt(k)
        q = np.array([r if r < k / 2 else r - k for r in range(k)])
        return fourier.conj().T @ np.diag(np.exp(-1j * alpha * 2 * np.pi**2 * q**2 / length**2)) @ fourier

    dt = method.total_time / method.num_steps
    total = np.eye(k**d, dtype=complex)
    for step in range(method.num_steps):
        t = (step + 0.5) * dt
        b = 1 + GAMMA * t * t
        kinetic_step = np.eye(1)
        for lower, upper in problem.bounds:
            kinetic_step = np.kron(kinetic_step, kinetic(dt / b, lower, upper))
        if method.trotter_order == 1:
            step_operator = kinetic_step @ np.diag(np.exp(-1j * dt * b * potential))
        else:
            half = np.diag(np.exp(-1j * dt * b * potential / 2))
            step_operator = half @ kinetic_step @ half
        total = step_operator @ total
    return total


def _evolution_operator(selected):
    """Return the operator of the Plan's binary steps and physical phase, without its preparation, in lexicographic order.

    Also returns the number of gates of the fully decomposed circuit, which
    sets the machine-precision allowance of the comparison.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    r = selected.reconstruction
    circuit = QuantumCircuit(r.width)
    append_binary_steps(circuit, r, selected.method, _grid(selected), held_bytes=0)
    circuit.global_phase += r.physical_phase
    order = _register_permutation(len(selected.problem.variables), selected.method.num_grid_points)
    gates = len(circuit.decompose(reps=6))
    return Operator(circuit).data[np.ix_(order, order)], gates


def _allowance(gates, dimension):
    """Machine-precision allowance of a circuit operator against an oracle, in the operator 2-norm.

    Qiskit's Operator multiplies the gate matrices. A gate's entries are
    within ``GATE_ENTRY_ERROR`` u of exact and each output entry is a complex
    inner product of at most four terms, ``sqrt(2) gamma_8``, so one gate
    moves a unit column by at most ``(2 GATE_ENTRY_ERROR + 12) u`` and the
    operator by ``sqrt(N)`` times that in 2-norm. G gates add up, and the
    oracle's own ``expm`` and matrix products are covered by doubling:
    ``2 G sqrt(N) (2 GATE_ENTRY_ERROR + 12) u``.
    """
    return 2 * gates * math.sqrt(dimension) * (2 * GATE_ENTRY_ERROR + 12) * UNIT_ROUNDOFF


SYNTHESES = {
    "min_cx": BinarySynthesis(),
    "dense": BinarySynthesis(potential="dense_diagonal", kinetic_phase="dense_diagonal"),
    "walsh": BinarySynthesis(potential="walsh_rotations", kinetic_phase="walsh_rotations"),
}


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("reversal", ["relabel", "swap"])
@pytest.mark.parametrize("model", ["finite_difference", "spectral"])
@pytest.mark.parametrize("synthesis", list(SYNTHESES))
def test_binary_circuit_equals_independent_product(synthesis, model, reversal, order):
    """The emitted binary steps equal an independently written product of the same finite model.

    Four qubits (two variables, K = 4). The comparison is of full operators,
    global phase included, after the explicit register permutation
    ``P^dagger U P``, so it holds for every input state, including complex
    ones, and it checks the QFT sign and bit reversal, the kinetic energies
    of both models, the table address order of the coupled term, the
    identity phases and the step ordering. ``min_cx`` gives the coupled
    table a dense diagonal and the one-variable tables Walsh rotations here,
    and the other two choices force one construction everywhere. Without the
    variable-axis permutation, the bit-reversal relabel or the signed index,
    the distance is of order 0.1 to 1.
    """
    problem = _problem()
    method = _method(4, kinetic_model=model, trotter_order=order,
                     binary_synthesis=SYNTHESES[synthesis].revise(qft_bit_reversal=reversal))
    selected = plan(problem, method=method, execution="quantum", seed=7)
    operator, gates = _evolution_operator(selected)
    expected = _oracle(problem, method)
    assert np.linalg.norm(operator - expected, 2) <= _allowance(gates, 16)


@pytest.mark.parametrize("model", ["finite_difference", "spectral"])
def test_binary_circuit_equals_independent_product_at_three_bits(model):
    """Six qubits (two variables, K = 8), where the bit reversal of three bits and the Nyquist index 4 act."""
    problem = _problem()
    method = _method(8, kinetic_model=model)
    selected = plan(problem, method=method, execution="quantum", seed=7)
    operator, gates = _evolution_operator(selected)
    assert np.linalg.norm(operator - _oracle(problem, method), 2) <= _allowance(gates, 64)


@pytest.mark.parametrize("reversal", ["relabel", "swap"])
def test_binary_public_routes_match_the_oracle_state(reversal):
    """Native amplitudes, native exact probabilities and the classical ir_product state follow the oracle.

    The oracle state is the oracle operator applied to the uniform state,
    which the structured H layer prepares. The stored native state, read in
    lexicographic order through the independent register permutation, and
    the ``ir_product`` kernel's state must equal it with the global phase.
    The analysis marginals of exact probabilities, which decode every
    outcome ``z`` into ``n_j = (z >> j b) mod K``, must equal the oracle
    marginals, and the whole register is valid. Both bit-reversal choices
    run, so the kernel's swap gates are applied too. Tolerance: the
    machine-precision allowance of the circuit comparison, which also covers
    the kernel's own first-order budget (``theory.binary_product_state_error``,
    below 1e-12 here).
    """
    problem = _problem()
    method = _method(4, kinetic_model="finite_difference", keep_state=True,
                     binary_synthesis=BinarySynthesis(qft_bit_reversal=reversal))
    expected = _oracle(problem, method) @ np.full(16, 0.25, dtype=complex)
    native = solve(problem, method=method, seed=7)
    state = native.data.artifact(native.artifact).array[_register_permutation(2, 4)]
    product = solve(problem, method=method.revise(theory_flavor="ir_product"), execution="classical", seed=7)
    probabilities = solve(problem, method=method.revise(keep_state=False), seed=7)
    _, gates = _evolution_operator(native.plan)
    tolerance = _allowance(gates, 16)
    assert np.linalg.norm(state - expected) <= tolerance
    assert np.linalg.norm(product.data.artifact(product.artifact).array - expected) <= tolerance
    grid_probabilities = (np.abs(expected) ** 2).reshape(4, 4)
    for result in (native, probabilities):
        assert result.invalid_mass == 0 and result.valid_mass == pytest.approx(1, abs=tolerance)
        assert np.max(np.abs(np.array(result.marginals) - [grid_probabilities.sum(1), grid_probabilities.sum(0)])
                      ) <= 2 * tolerance
    # The explicit IR comparison restricts the stored native state by the same
    # permutation, so the circuit agrees with the emitted product. Both states
    # lie within the tolerance of the oracle, so they differ by at most twice
    # it, and 1 - |<a, b>|**2 <= 2 ||a - b|| for unit vectors.
    receipt, _ = native.verify(checks=QHDVerification(comparisons=("ir_product_fidelity",)))
    facts = {f.fact.quantity: f.fact.value.value for f in receipt.applications[0].facts}
    assert facts["ir_product_infidelity"] <= 4 * tolerance and facts["ir_product_fidelity.within_tolerance"] == 1.


@pytest.mark.parametrize("bits", range(1, 7))
def test_signed_square_expansion_equals_q_squared(bits):
    """The Pauli expansion of the signed square equals ``q**2`` on every basis state.

    q is the two's-complement value of the b bits, n below K/2 and n - K
    above, written here independently of the expansion. The coefficients are
    multiples of 1/2 whose absolute sum is ``2**(2b-2)``, so every sum is exact
    in binary64 for these b (``binary.signed_square_walsh`` derives the range
    ``b <= 27``), and equality is exact.
    Only the identity, single-Z and two-Z masks are nonzero, which gives the
    ``b (b - 1)`` CX of Walsh rotations. At K = 8 index 7 is momentum -1 and
    gives 1, where the unsigned square of Liu et al., arXiv:2607.16996v1,
    Eq. (90), gives 49.
    """
    k = 1 << bits
    coefficients = signed_square_walsh(bits)
    values = walsh_sum(coefficients)
    assert values.tolist() == [float((n if n < k / 2 else n - k) ** 2) for n in range(k)]
    assert all(bin(mask).count("1") <= 2 for mask in np.flatnonzero(coefficients))
    if bits == 3:
        assert values[7] == 1.0


def _single_block(construction, qubits):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(qubits)
    append_diagonal(circuit, construction, list(range(qubits)))
    return circuit


def _cx(circuit):
    from qiskit import transpile

    return transpile(circuit, basis_gates=["cx", "u"], optimization_level=0).count_ops().get("cx", 0)


@pytest.mark.parametrize("qubits", [2, 3, 4])
def test_diagonal_cx_laws_equal_transpiled_counts(qubits):
    """Each diagonal synthesis emits the CX count its law states, at optimization level 0.

    Dense: ``2**n - 2`` for a nonzero table, whatever its values, and no gate
    for an all-zero table or a zero exponent. Walsh: ``2 (w - 1)`` per
    emitted string of weight w, for a table with every mask nonzero, for a
    table whose only strings are single Z (no CX), and after a threshold
    drops strings. ``min_cx`` takes the smaller and Walsh on a tie.
    """
    rng = np.random.default_rng(11)
    generic = PhaseTable.from_values(rng.normal(size=1 << qubits))
    # Dyadic values, so the Walsh sums are exact and every multi-bit coefficient is exactly zero.
    linear = PhaseTable.from_values([0.25 * bin(z).count("1") + 0.125 * (z & 1) for z in range(1 << qubits)])
    zero = PhaseTable.from_values(np.zeros(1 << qubits))
    full_walsh = sum(2 * (bin(m).count("1") - 1) for m in range(1, 1 << qubits))
    cases = [
        (generic, 0.7, "dense_diagonal", 0.0, (1 << qubits) - 2),
        (generic, 0.7, "walsh_rotations", 0.0, full_walsh),
        (generic, 0.0, "dense_diagonal", 0.0, 0),
        (zero, 0.7, "dense_diagonal", 0.0, 0),
        (linear, 0.7, "walsh_rotations", 0.0, 0),
        (linear, 0.7, "min_cx", 0.0, 0),
    ]
    for table, exponent, choice, threshold, law in cases:
        built = table.synthesize(exponent, choice, threshold)
        assert built.cx == law == _cx(_single_block(built, qubits))
    angles = 1.4 * generic.coefficients
    threshold = float(np.median(np.abs(angles[1:])))
    pruned = generic.synthesize(0.7, "walsh_rotations", threshold)
    kept = [m for m in range(1, 1 << qubits) if abs(angles[m]) >= threshold]
    assert pruned.dropped_count == (1 << qubits) - 1 - len(kept) > 0
    assert pruned.cx == sum(2 * (bin(m).count("1") - 1) for m in kept) == _cx(_single_block(pruned, qubits))
    chosen = generic.synthesize(0.7, "min_cx", 0.0)
    assert chosen.synthesis == ("walsh_rotations" if full_walsh <= (1 << qubits) - 2 else "dense_diagonal")
    assert _cx(_single_block(zero.synthesize(0.7, "dense_diagonal", 0.0), qubits)) == 0
    assert len(_single_block(zero.synthesize(0.7, "dense_diagonal", 0.0), qubits)) == 0


@pytest.mark.parametrize("fields", [
    dict(),
    dict(binary_synthesis=BinarySynthesis(qft_bit_reversal="swap", potential="walsh_rotations")),
    dict(binary_synthesis=BinarySynthesis(kinetic_phase="dense_diagonal", potential="dense_diagonal",
                                          aqft_cutoff=1), trotter_order=1),
    dict(kinetic_model="spectral", binary_synthesis=BinarySynthesis(aqft_cutoff=0, qft_bit_reversal="swap")),
])
def test_plan_cx_law_equals_the_transpiled_circuit(fields):
    """The Plan's CX law equals the CX of its whole native circuit at optimization level 0.

    The law adds the recorded block counts: two QFTs of ``b (b - 1)`` CX
    without swaps, ``3 floor(b/2)`` more per QFT with them, two per kept
    controlled phase under an AQFT cutoff, and each phase diagonal's count.
    The structured H-layer preparation adds none. Six qubits (K = 8, two variables).
    The rotation law is at least the number of Rz of the circuit transpiled
    to CX, Rz, H, SX and X, with equality when no dense diagonal omits a
    zero multiplexor angle.
    """
    from qiskit import transpile

    selected = plan(_problem(), method=_method(8, **fields), execution="quantum", seed=7)
    circuit = construct_qhd(selected.blocks[0], (), None)
    laws = {law.metric: law.value for law in selected.construction.selections[0].resource_laws}
    assert laws["cx"] == _cx(circuit)
    rotations = transpile(circuit, basis_gates=["cx", "rz", "h", "sx", "x"], optimization_level=0)
    dense = any(b.synthesis == "dense_diagonal" for step in selected.reconstruction.steps for b in step)
    counted = rotations.count_ops().get("rz", 0)
    assert counted <= laws["arbitrary_rotations"] and (dense or counted == laws["arbitrary_rotations"])


@pytest.mark.parametrize("potential", ["dense_diagonal", "min_cx"])
def test_native_work_charges_the_gray_code_butterflies_of_every_dense_diagonal(potential, monkeypatch):
    """The native work law pays for every executed dense wrap, recursion and butterfly.

    A nonzero N-entry phase wrap runs a product, exponential and argument,
    costing 3N under QHD._select_binary_native's value-operation unit.
    Each recursive pair costs four operations. A k-level Gray-code
    butterfly costs three operations per output per level. Spies observe
    the tables passed to those operations during actual construction.
    After subtracting the law's other allowances, the dense allocation
    must equal that observed subtotal. This does not assert an exact
    count for the complete construction law.

    At K=4 the coupled support has 34 Walsh CX against 14 dense CX, so
    min_cx selects it densely while its linear supports use Walsh.
    Both parameter choices repeat nonzero dense blocks in two
    second-order steps. The structured H-layer start uses no phase wrap.
    """
    import nwqlib.subroutines._multiplexors as multiplexors
    from nwqlib.algorithms.qhd.binary import kinetic_setup_work

    operations = []
    original, multiplexor = multiplexors.gray_code_rotation_angles, multiplexors.append_uniformly_controlled_rz
    wrap_operations = []
    angle = multiplexors.np.angle

    def wrap(z, *args, **kwargs):
        wrap_operations.append(3 * np.asarray(z).size)
        return angle(z, *args, **kwargs)

    def spy(angles):
        values = np.asarray(angles, dtype=float)
        size = values.shape[-1]
        # tables of 2**k angles, k butterfly levels of 2**k outputs, three operations per output
        operations.append(3 * (values.size // size) * size * (size.bit_length() - 1))
        return original(angles)

    def pairs(circuit, target, controls, angles):
        # one pair difference per angle, and the pair's mean: a subtraction, two halvings and an addition
        operations.append(4 * len(angles))
        return multiplexor(circuit, target, controls, angles)

    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=(x - sp.Rational(3, 10)) * (y - sp.Rational(3, 5)) + sp.sin(3 * x * y),
                           variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    selected = plan(problem, method=_method(4, binary_synthesis=BinarySynthesis(potential=potential),
                                            trotter_order=2), execution="quantum", seed=7)
    monkeypatch.setattr(multiplexors, "gray_code_rotation_angles", spy)
    monkeypatch.setattr(multiplexors, "append_uniformly_controlled_rz", pairs)
    monkeypatch.setattr(multiplexors.np, "angle", wrap)
    construct_qhd(selected.blocks[0], (), None)
    r, bits, k, d = selected.reconstruction, 2, 4, 2
    blocks = [b for step in r.steps for b in step]
    dense = [b for b in blocks if b.synthesis == "dense_diagonal"]
    assert len(dense) >= 2 and len({(b.kind, b.variables) for b in dense}) < len(dense)
    widths = [bits * len(b.variables) for b in dense]
    assert sum(operations) == sum(3 * ((n - 2) * 2**n + 2) + 4 * (2**n - 1) for n in widths) > 0
    assert sum(wrap_operations) == sum(3 * (1 << n) for n in widths) > 0
    walsh = (0 if potential == "dense_diagonal" else
             sum(len(t.values) * (len(t.values).bit_length() + 3)
                 for t in r.support_values))
    other = (d * bits
             + sum((3 if b.synthesis == "dense_diagonal" else 5)
                   + b.cx + b.rotations
                   + (2 * bits if b.kind == "binary_kinetic" else 0)
                   + k ** len(b.variables) for b in blocks)
             + walsh + d * kinetic_setup_work(bits)
             + r.range_omissions.dense_entries
             + 2 * r.range_omissions.rotations + 4 * r.dropped_count)
    charged = selected.construction.selections[0].construction_work - other
    assert charged == sum(operations) + sum(wrap_operations)


def test_aqft_bound_covers_the_truncated_conjugation():
    """The recorded AQFT bound is at least the operator error of every truncated conjugation at b = 6.

    One variable with K = 64, zero objective, one first-order step and
    ``gamma = 0``, so the circuit is one kinetic conjugation with exponent
    ``alpha = T``. For each cutoff m = 0..5 (Qiskit approximation degree
    5..0) the emitted conjugation is compared with the exact
    ``F^dagger diag(exp(-i T E)) F`` built from the analytic Fourier matrix.
    ``aqft_error_bound`` is ``min(2, 2 e_F)`` (``binary.qft_error_bound``),
    zero for the exact QFT. The machine-precision allowance of the circuit
    comparison covers the roundoff of the measurement, which is all that
    remains at m = 5.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 6.4),))
    k, total_time = 64, 0.37
    h = 6.4 / k
    energies = np.array([2 * math.sin(math.pi * r / k) ** 2 / h**2 for r in range(k)])
    fourier = np.exp(-2j * np.pi * np.outer(range(k), range(k)) / k) / np.sqrt(k)
    exact = fourier.conj().T @ np.diag(np.exp(-1j * total_time * energies)) @ fourier
    bounds = []
    for cutoff in range(6):
        method = QHD(encoding="binary", boundary="periodic", num_grid_points=k,
                     num_steps=1, trotter_order=1, total_time=total_time, schedule=QuadraticSchedule(gamma=0.0),
                     binary_synthesis=BinarySynthesis(aqft_cutoff=cutoff))
        selected = plan(problem, method=method, execution="quantum", seed=7)
        circuit = QuantumCircuit(6)
        append_binary_steps(circuit, selected.reconstruction, method, _grid(selected), held_bytes=0)
        measured = np.linalg.norm(Operator(circuit).data - exact, 2)
        bound = selected.reconstruction.aqft_error_bound
        bounds.append(bound)
        assert measured <= bound + _allowance(len(circuit.decompose(reps=6)), k)
    assert bounds[-1] == 0.0 and all(a >= b for a, b in itertools.pairwise(bounds))


@pytest.mark.parametrize("variables,steps", [(1, 1), (1, 2), (2, 1)])
def test_aqft_ledger_counts_every_conjugation(variables, steps):
    """The recorded AQFT bound is ``2 N_s d e_F`` for a QFT that omits one controlled phase.

    With b = 4 and cutoff m = 2 each QFT omits exactly one gate, the phase
    ``pi/8`` at distance 3, so ``e_F = ||CP(pi/8) - I|| = 2 sin(pi/16)``,
    written here from that gate alone. Each of the ``N_s`` steps has one
    conjugation per variable and each conjugation two QFTs, so the ledger is
    ``2 N_s d 2 sin(pi/16)``, below the cap 2 in all three cases. The
    one-step, one-variable case detects a missing pair factor, and the other
    two a missing step or variable multiplier (``binary.compile_binary_steps``
    derives the ledger). Planning builds no circuit. The upper tolerance
    covers the ledger's outward factors ``1 + 8u`` and ``1 + 4u`` and its few
    roundings.
    """
    symbols = sp.symbols("x y", real=True)[:variables]
    problem = Optimization(objective=sp.Integer(0), variables=symbols, bounds=((0.0, 1.0),) * variables)
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=16,
                 num_steps=steps, binary_synthesis=BinarySynthesis(aqft_cutoff=2))
    recorded = plan(problem, method=method, execution="quantum", seed=7).reconstruction.aqft_error_bound
    with mpmath.workdps(30):
        expected = 2 * steps * variables * 2 * mpmath.sin(mpmath.pi / 16)
        assert expected <= recorded <= expected * (1 + 64 * UNIT_ROUNDOFF)


def test_pruning_bound_is_an_outward_upper_bound():
    """The recorded pruning bound is never below half the exact sum of the dropped Rz angles.

    ``V = x/2**38 + y/2**93`` on the periodic box ``[0, 1)**2`` with K = 4 has
    the tables ``i 2**-40`` and ``j 2**-95``, whose nonzero Walsh
    coefficients give, at exponent 1, the Rz angles ``2**-40``, ``2**-39``,
    ``2**-95`` and ``2**-94``. The threshold 1e-10 drops exactly these four,
    and their exact sum ``S = 3 (2**-40 + 2**-95)`` is not a binary64 number,
    so a sum rounded to nearest and halved can land below ``S/2``, which it did
    before the ledger summed the angles exactly
    (``binary.compile_binary_steps``). The
    comparison uses exact rationals. The same plan without a threshold drops
    nothing and records exactly zero.
    """
    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=x / sp.Integer(2) ** 38 + y / sp.Integer(2) ** 93, variables=(x, y),
                           bounds=((0.0, 1.0), (0.0, 1.0)))
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=4,
                 total_time=1.0, num_steps=1, trotter_order=1, schedule=QuadraticSchedule(gamma=0.0),
                 rotation_threshold=1e-10,
                 binary_synthesis=BinarySynthesis(potential="walsh_rotations", kinetic_phase="walsh_rotations"))
    r = plan(problem, method=method, execution="quantum", seed=7).reconstruction
    half_sum = Fraction(3, 2**41) + Fraction(3, 2**96)
    assert r.dropped_count == 4
    assert half_sum <= Fraction(r.pruning_error_bound) <= half_sum * (1 + Fraction(1, 2**52))
    unpruned = plan(problem, method=method.revise(rotation_threshold=0.0), execution="quantum", seed=7)
    assert unpruned.reconstruction.dropped_count == 0 and unpruned.reconstruction.pruning_error_bound == 0.0


@pytest.mark.parametrize("model", ["finite_difference", "spectral"])
def test_one_bit_binary_grid_matches_the_oracle(model):
    """K = 2, one qubit per variable: the circuit, the native state and the classical routes follow the oracle.

    On the periodic grid of two points each point's two neighbors are the
    other point, so the finite-difference stencil is ``(I - X)/h**2`` with
    energies 0 and ``2/h**2``, and the spectral model has the signed indices
    0 and -1, so ``2 pi**2/L**2`` at index 1. One H conjugates either
    diagonal. Two variables give two qubits. The emitted steps, the stored
    native state and the ``ir_product`` state are compared with the
    independent product oracle, and for finite difference the ``schrodinger``
    state with the exact exponential of each step's Hamiltonian
    ``a (Kx x I + I x Ky) + b diag(V)``, written here from ``(I - X)/h**2``.
    Tolerances: the machine-precision allowance of the circuit comparison,
    and for the ``schrodinger`` state twice its own first-order budget
    (``method._host_state_error``, start vector and constant phase
    included), which also covers the oracle's small exponentials.
    """

    problem = _problem()
    method = _method(2, kinetic_model=model, keep_state=True)
    selected = plan(problem, method=method, execution="quantum", seed=7)
    operator, gates = _evolution_operator(selected)
    tolerance = _allowance(gates, 4)
    oracle = _oracle(problem, method)
    assert np.linalg.norm(operator - oracle, 2) <= tolerance
    expected = oracle @ np.full(4, 0.5, dtype=complex)
    native = solve(problem, method=method, seed=7)
    assert np.linalg.norm(native.data.artifact(native.artifact).array[_register_permutation(2, 2)] - expected
                          ) <= tolerance
    product = solve(problem, method=method.revise(theory_flavor="ir_product"), execution="classical", seed=7)
    assert np.linalg.norm(product.data.artifact(product.artifact).array - expected) <= tolerance
    if model == "finite_difference":
        schrodinger = solve(problem, method=method, execution="classical", seed=7)
        evaluate = sp.lambdify(problem.variables, problem.objective)
        grids = [(lower, (upper - lower) / 2) for lower, upper in problem.bounds]
        potential = np.diag([evaluate(grids[0][0] + i * grids[0][1], grids[1][0] + j * grids[1][1])
                             for i in range(2) for j in range(2)])
        stencils = [(np.eye(2) - np.array([[0.0, 1.0], [1.0, 0.0]])) / h**2 for _, h in grids]
        kinetic = np.kron(stencils[0], np.eye(2)) + np.kron(np.eye(2), stencils[1])
        state = np.full(4, 0.5, dtype=complex)
        dt = method.total_time / method.num_steps
        for step in range(method.num_steps):
            t = (step + 0.5) * dt
            b = 1 + GAMMA * t * t
            state = scipy.linalg.expm(-1j * dt * (kinetic / b + b * potential)) @ state
        delta = schrodinger.plan.reconstruction.host_state_error
        assert np.linalg.norm(schrodinger.data.artifact(schrodinger.artifact).array - state) <= 2 * delta


@pytest.mark.parametrize("fields,message", [
    (dict(encoding="binary", num_grid_points=8), "requires boundary='periodic'"),
    (dict(encoding="binary", boundary="periodic", num_grid_points=6), "2\\*\\*b"),
    (dict(binary_synthesis=BinarySynthesis(qft_bit_reversal="swap")), "encoding='binary' only"),
])
def test_binary_field_combinations_are_checked(fields, message):
    """The binary encoding needs the periodic grid with K = 2**b, and its synthesis choices need the binary encoding."""
    with pytest.raises(ValueError, match=message):
        QHD(**fields)


def test_binary_structured_preparation_is_the_uniform_h_layer():
    """The structured binary preparation prepares the uniform state only, and only a native Plan prepares it.

    One H per qubit prepares the uniform state, which on the periodic grid is
    also the kinetic ground state, so both plan natively with no preparation
    CX. A native Gaussian start needs Qiskit's preparation, while the
    classical routes, which prepare no circuit, accept it with the default
    recipe.
    """
    problem = _problem()
    gaussian = GaussianState(center=(0.85, 0.1), widths=(0.5, 0.6))
    for state in (UniformState(), KineticGroundState()):
        native = plan(problem, method=_method(4, initial_state=state), execution="quantum", seed=7)
        law = next(law for law in native.construction.selections[0].resource_laws if law.metric == "cx")
        assert law.value == sum(b.cx for step in native.reconstruction.steps for b in step)
    with pytest.raises(ValueError, match="uniform state only"):
        plan(problem, method=_method(4, initial_state=gaussian), execution="quantum", seed=7)
    plan(problem, method=_method(4, initial_state=gaussian, initial_state_preparation="qiskit_state_preparation"),
         execution="quantum", seed=7)
    plan(problem, method=_method(4, initial_state=gaussian, theory_flavor="ir_product"), execution="classical",
         seed=7)


def test_binary_routes_start_from_the_stored_gaussian_amplitudes():
    """Both binary routes start from the per-variable vectors that planning stored, here a Gaussian.

    The native circuit appends one ``StatePreparation`` per register whose
    parameters are exactly the stored vector of that variable, and the
    ``ir_product`` state equals the oracle operator applied to the Kronecker
    product of the stored vectors, variable 0 most significant. The stored
    vectors' own accuracy is the owner's (``initial_state``). Tolerance: the
    machine-precision allowance of the circuit comparison, as in
    ``test_binary_public_routes_match_the_oracle_state``, which exceeds the
    kernel's own budget with its Gaussian start term, checked here.
    """

    problem = _problem()
    gaussian = GaussianState(center=(0.85, 0.1), widths=(0.5, 0.6))
    method = _method(4, initial_state=gaussian, initial_state_preparation="qiskit_state_preparation",
                     keep_state=True)
    native = plan(problem, method=method, execution="quantum", seed=7)
    stored = native.reconstruction.initial_amplitudes
    circuit = construct_qhd(native.blocks[0], (), None)
    preparations = [item for item in circuit.data if item.operation.name == "state_preparation"]
    assert [[circuit.find_bit(q).index for q in item.qubits] for item in preparations] == [[0, 1], [2, 3]]
    assert [tuple(item.operation.params) for item in preparations] == [tuple(complex(a) for a in v.array.tolist()) for v in stored]
    product = solve(problem, method=method.revise(theory_flavor="ir_product"), execution="classical", seed=7)
    assert product.plan.reconstruction.initial_amplitudes == stored
    _, gates = _evolution_operator(native)
    tolerance = _allowance(gates, 16)
    assert product.plan.reconstruction.host_state_error < tolerance
    expected = _oracle(problem, method) @ np.kron(np.array(stored[0]), np.array(stored[1]))
    assert np.linalg.norm(product.data.artifact(product.artifact).array - expected) <= tolerance


@pytest.mark.parametrize("execution,flavor", [("classical", "ir_product"), ("quantum", "schrodinger")])
def test_binary_kept_state_follows_the_pi_rule(execution, flavor):
    """A kept binary state is rejected when the constant's phase allowance reaches pi, and admitted otherwise.

    The binary ledger holds only the objective constant (``method._phase_ledger``).
    For c = 1e17 over two unit steps with unit potential weight its sum
    ``S_C = |c| sum_k |dt b_k|`` is 2e17 rad, and the allowance, about
    ``3u S_C``, exceeds pi on both circuit routes, so a kept state is
    rejected while the probabilities are admitted. A constant of 1 is
    admitted with the kept state.
    """
    x = sp.Symbol("x", real=True)
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=4, num_steps=2, total_time=2.0,
                 schedule=QuadraticSchedule(gamma=0.0), theory_flavor=flavor, keep_state=True)
    huge = Optimization(objective=x**2 + sp.Float(1e17), variables=(x,), bounds=((0.0, 1.0),))
    with pytest.raises(ValueError, match="reaches pi"):
        plan(huge, method=method, execution=execution, seed=7)
    plan(huge, method=method.revise(keep_state=False), execution=execution, seed=7)
    plan(huge.revise(objective=x**2 + 1), method=method, execution=execution, seed=7)


def _walsh_terms(selected):
    """Return the Walsh identity-phase terms of a binary Plan (``binary.WalshPhaseTerms``).

    The Plan stores the terms that planning compiled (``QHDReconstruction.walsh_phase``),
    and they must equal those of a fresh compilation of its stored tables and step rows.
    """
    from nwqlib.algorithms.qhd.binary import BinaryModel, compile_binary_steps
    from nwqlib.algorithms.qhd.records import QHDWalshPhase

    r = selected.reconstruction
    model = BinaryModel(_grid(selected), selected.method, r.support_values)
    terms = compile_binary_steps(model, selected.method, r.step_weights, r.constant)[1]["walsh_phase"]
    assert r.walsh_phase == QHDWalshPhase(**terms._asdict())
    return terms


def _kept_walsh_method(**fields):
    """One static first-order unit step on K = 4 points with Walsh potentials and a kept state."""
    return QHD(encoding="binary", boundary="periodic", num_grid_points=4, num_steps=1, trotter_order=1,
               total_time=1.0, schedule=QuadraticSchedule(gamma=0.0), keep_state=True,
               binary_synthesis=BinarySynthesis(potential="walsh_rotations"), **fields)


@pytest.mark.parametrize("execution,flavor", [("classical", "ir_product"), ("quantum", "schrodinger")])
def test_kept_binary_state_charges_walsh_identity_formation(execution, flavor):
    """A kept binary state is refused when only the Walsh identity formation F_W lifts its allowance past pi.

    On the four points ``i/4`` of the periodic box ``[0, 1)`` the objective
    ``1e16 cos(8 pi x)`` is exactly 1e16 at every point, so its Walsh table
    has the identity coefficient 1e16 and no other string, and there is no
    objective constant, so the ledger allowance is zero. At exponent 1 the
    identity phase is -1e16 rad. Its formation allowance is about
    ``4u 1e16 = 4.4`` rad (``binary.compile_binary_steps``), while the
    ``ir_product`` reconstruction term, about ``u 1e16``, and the native
    assignment term, about ``2u 1e16``, stay below pi, as checked here. The
    kept state is refused with the Walsh identity formation named as the
    largest component, and the probabilities are admitted.
    """
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=sp.Float(1e16) * sp.cos(8 * sp.pi * x), variables=(x,), bounds=((0.0, 1.0),))
    method = _kept_walsh_method(theory_flavor=flavor)
    probabilities = plan(problem, method=method.revise(keep_state=False), execution=execution, seed=7)
    assert probabilities.reconstruction.constant == 0.0
    terms = _walsh_terms(probabilities)
    assert terms.formation >= math.pi
    other = (terms.reconstruction if execution == "classical"
             else 2 * UNIT_ROUNDOFF * terms.identity_sum + 20 * UNIT_ROUNDOFF * (terms.count + 1))
    assert other < 3.0
    with pytest.raises(ValueError, match="reaches pi.*formation of the Walsh diagonals' identity phases"):
        plan(problem, method=method, execution=execution, seed=7)


@pytest.mark.parametrize("execution,flavor", [("classical", "ir_product"), ("quantum", "schrodinger")])
def test_walsh_terms_do_not_need_a_representable_absolute_sum(execution, flavor):
    """The Walsh phase terms of a finite table plan even when its unscaled absolute sum overflows.

    The objective ``2**1022`` for ``x < 3/4`` and ``-2**1022`` otherwise
    gives the table ``(a0, a0, a0, -a0)`` with ``a0 = 2**1022`` on the four
    points of ``[0, 1)``. Its Walsh coefficients ``(a0/2, a0/2, a0/2,
    -a0/2)`` and its mean absolute value a0 are finite, while its absolute
    sum ``2**1024`` is not. One step of duration ``2**-1021``, whose half is
    the smallest normal number, gives the identity phase -1 and the angles
    ``(2, 2, -2)``, and both routes plan with and without a kept state. One
    step of duration 2 instead gives three kept angles of magnitude
    ``2**1023``, whose inverse butterfly in the ``ir_product`` reconstruction
    would overflow, so planning refuses it on both routes
    (``binary._admit_reconstruction``).
    """
    x = sp.Symbol("x", real=True)
    big = math.ldexp(1.0, 1022)
    problem = Optimization(objective=sp.Piecewise((sp.Float(big), x < sp.Rational(3, 4)), (-sp.Float(big), True)),
                           variables=(x,), bounds=((0.0, 1.0),))
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=4, num_steps=1, trotter_order=1,
                 total_time=math.ldexp(1.0, -1021), schedule=QuadraticSchedule(gamma=0.0), theory_flavor=flavor,
                 binary_synthesis=BinarySynthesis(potential="walsh_rotations", kinetic_phase="walsh_rotations"))
    for keep_state in (False, True):
        selected = plan(problem, method=method.revise(keep_state=keep_state), execution=execution, seed=7)
        assert selected.reconstruction.support_values[0].values.array.tolist() == [big, big, big, -big]
    with pytest.raises(ValueError, match="reconstruction of its phases could overflow"):
        plan(problem, method=method.revise(total_time=2.0), execution=execution, seed=7)


def test_kept_binary_ir_product_charges_phase_reconstruction():
    """``ir_product`` refuses a kept state whose allowance crosses pi only through the reconstruction R_W.

    ``1.6e16 exp(-1e6 x**2)`` is 1.6e16 at x = 0 and exactly zero at the
    other three points ``i/4``, so all four Walsh coefficients are 4e15. At
    exponent 1 the three kept angles are 8e15 rad each, their half-sum T is
    1.2e16, and the reconstruction allowance, about ``3u T + u 4e15 = 4.4``
    rad, exceeds pi, while the formation allowance, about ``4u 4e15 = 1.8``
    rad, stays below it. The native route charges no reconstruction, and its
    allowance, F_W plus ``2u Y + 20u (B + 1)``, about 2.7 rad, admits the
    same kept state. B counts every executed Walsh block, the kinetic one
    included.
    """
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=sp.Float(1.6e16) * sp.exp(-1000000 * x**2), variables=(x,),
                           bounds=((0.0, 1.0),))
    method = _kept_walsh_method(theory_flavor="ir_product")
    native = plan(problem, method=method, execution="quantum", seed=7)
    terms = _walsh_terms(native)
    assert terms.count == sum(b.synthesis == "walsh_rotations" for step in native.reconstruction.steps for b in step)
    assert terms.formation < 2.0 and terms.reconstruction >= math.pi
    with pytest.raises(ValueError, match="reaches pi.*reconstruction of the Walsh diagonals' phases"):
        plan(problem, method=method, execution="classical", seed=7)
    plan(problem, method=method.revise(keep_state=False), execution="classical", seed=7)


@pytest.mark.parametrize("encoding,execution,flavor", [
    ("one_hot", "quantum", "schrodinger"),
    ("one_hot", "classical", "ir_product"),
    ("binary", "classical", "schrodinger"),
])
def test_spectral_kinetic_needs_a_route_that_applies_it(encoding, execution, flavor):
    """The binary product and the split-step flavor apply the spectral energies, the others the stencil."""
    method = QHD(encoding=encoding, boundary="periodic", num_grid_points=4, kinetic_model="spectral",
                 theory_flavor=flavor)
    with pytest.raises(ValueError, match="finite-difference stencil"):
        plan(_problem(), method=method, execution=execution, seed=7)


@pytest.mark.parametrize("encoding,execution,flavor", [
    ("binary", "quantum", "schrodinger"),
    ("one_hot", "classical", "split_step"),
])
def test_spectral_kinetic_refuses_an_overflowing_nyquist_energy(encoding, execution, flavor):
    """Planning refuses a spectral grid whose Nyquist energy ``pi**2/(2 h**2)`` is not a finite binary64 number.

    On (0, 6e-154) with K = 4 the spacing is 1.5e-154, which the grid guard admits because ``1/h**2`` to
    ``1/(4 h**2)`` are normal numbers, while ``2 (pi/(2 h))**2`` overflows. The spectral kinetic energies and
    coefficients would be infinite, so planning stops before any table work.
    """
    x = sp.Symbol("x", real=True)
    method = QHD(encoding=encoding, boundary="periodic", num_grid_points=4, num_steps=1, kinetic_model="spectral",
                 theory_flavor=flavor, initial_state=UniformState())
    with pytest.raises(ValueError, match=r"Nyquist energy pi\*\*2/\(2 h\*\*2\), which is not a finite"):
        plan(Optimization(objective=x**2, variables=(x,), bounds=((0.0, 6e-154),)), method=method,
             execution=execution, seed=7)


def test_binary_planning_refuses_known_work_before_compilation(monkeypatch):
    """Eight first-order K=2 Walsh steps need at least 128 compilation value operations.

    Each of the 16 blocks performs six synthesis operations plus its
    exponent product and the ledger's identity-phase absolute value.
    A limit of 127 must refuse before BinaryModel, independently of the
    larger upper bound derived in QHD._admit_symbolic_work. The default
    limit must still admit this small public native Plan.
    """
    import nwqlib.algorithms.qhd.binary as binary

    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(
        encoding="binary", boundary="periodic", num_grid_points=2,
        num_steps=8, total_time=1.0, trotter_order=1,
        schedule=QuadraticSchedule(gamma=0.0), initial_state=UniformState(),
        binary_synthesis=BinarySynthesis(
            potential="walsh_rotations", kinetic_phase="walsh_rotations"),
    )
    selected = plan(problem, method=method, execution="quantum", seed=7)
    assert len(selected.reconstruction.steps) == 8
    assert all(len(step) == 2 and all(b.synthesis == "walsh_rotations" and b.rotations == 1
                                    for b in step)
               for step in selected.reconstruction.steps)

    def forbidden(*args, **kwargs):
        pytest.fail("insufficient-work Plan reached BinaryModel")

    monkeypatch.setattr(binary, "BinaryModel", forbidden)
    with pytest.raises(ValueError, match="symbolic tables and binary compilation"):
        plan(problem, method=method.revise(max_work=127), execution="quantum", seed=7)


@pytest.mark.parametrize("model", ["finite_difference", "spectral"])
def test_kinetic_identity_enclosure_contains_the_exact_mean(model):
    """The identity-phase allowance requires enclosure of the exact kinetic mean.

    For K=2 the cosine sum vanishes, giving c0=h**-2. The signed
    spectral indices give sum q**2=K*(K**2+2)/12, hence
    c0=2*pi**2*(K**2+2)/(12*(K*h)**2). The stored spacing 0.185
    comes from the ordinary periodic box [0, 0.37). Exact fractions
    and 100-digit pi independently check containment, without a width
    tolerance. See binary.kinetic_identity_enclosure for the bound.
    """
    from nwqlib.algorithms.qhd.binary import kinetic_identity_enclosure

    h = 0.185
    low, high = kinetic_identity_enclosure(model, 1, h)
    if model == "finite_difference":
        exact = 1 / Fraction(h)**2
        assert Fraction(low) <= exact <= Fraction(high)
    else:
        with mpmath.workdps(100):
            exact = 2 * mpmath.pi**2 * 6 / (12 * (2 * mpmath.mpf(h))**2)
            assert mpmath.mpf(low) <= exact <= mpmath.mpf(high)


def test_spectral_wide_period_plans_finite_normal_coefficients():
    """Spectral planning succeeds for period 5*2**510 without an overflow refusal."""
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 5 * 2.0**510),))
    selected = plan(problem, method=QHD(
        encoding="binary", boundary="periodic", kinetic_model="spectral",
        theory_flavor="ir_product", num_grid_points=8, num_steps=1,
        total_time=1.0, schedule=QuadraticSchedule(gamma=0.0), initial_state=UniformState(),
    ), execution="quantum", seed=7)
    assert selected.reconstruction.steps
