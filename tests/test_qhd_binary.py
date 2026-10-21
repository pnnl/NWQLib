"""Binary QHD encoding: the circuit against independent product oracles, the signed square, CX laws and the AQFT bound."""

import itertools
import math

import mpmath
import numpy as np
import pytest
import scipy.linalg
import sympy as sp

from nwqlib._validation import GATE_ENTRY_ERROR, UNIT_ROUNDOFF
from nwqlib.algorithms.qhd import QHD, BinarySynthesis, QHDVerification, QuadraticSchedule
from nwqlib.algorithms.qhd.initial_state import GaussianState, UniformState
from nwqlib.algorithms.qhd.binary import signed_square_walsh, walsh_sum
from nwqlib.algorithms.qhd.method import _grid
from nwqlib.algorithms.qhd.native import append_binary_steps, construct_qhd
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


@pytest.mark.parametrize("reversal", ["relabel"])
def test_binary_public_routes_match_the_oracle_state(reversal):
    """Native amplitudes, native exact probabilities and the classical ir_product state follow the oracle.

    The oracle state is the oracle operator applied to the uniform state,
    which the structured H layer prepares. The stored native state, read in
    lexicographic order through the independent register permutation, and
    the ``ir_product`` kernel's state must equal it with the global phase.
    The analysis marginals of exact probabilities, which decode every
    outcome ``z`` into ``n_j = (z >> j b) mod K``, must equal the oracle
    marginals, and the whole register is valid. Tolerance: the
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


def _cx(circuit):
    from qiskit import transpile

    return transpile(circuit, basis_gates=["cx", "u"], optimization_level=0).count_ops().get("cx", 0)


@pytest.mark.parametrize("fields", [
    dict(),
    dict(binary_synthesis=BinarySynthesis(kinetic_phase="dense_diagonal", potential="dense_diagonal",
                                          aqft_cutoff=1), trotter_order=1),
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


@pytest.mark.parametrize("variables,steps", [(1, 2), (2, 1)])
def test_aqft_ledger_counts_every_conjugation(variables, steps):
    """The recorded AQFT bound is ``2 N_s d e_F`` for a QFT that omits one controlled phase.

    With b = 4 and cutoff m = 2 each QFT omits exactly one gate, the phase
    ``pi/8`` at distance 3, so ``e_F = ||CP(pi/8) - I|| = 2 sin(pi/16)``,
    written here from that gate alone. Each of the ``N_s`` steps has one
    conjugation per variable and each conjugation two QFTs, so the ledger is
    ``2 N_s d 2 sin(pi/16)``, below the cap 2 in both cases. The two cases
    detect a missing step or variable multiplier (``binary.compile_binary_steps``
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
