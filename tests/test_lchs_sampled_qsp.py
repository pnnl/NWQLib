"""Sampled LCHS pairs on a finite QSP phase bank, through the public workflow.

The Methods here plan banks at bucket time tau = 0.4 with approx_tol 0.2.
Their phases are not fitted again: ``data/lchs_sampled_qsp_targets.json`` holds the three prepared targets
(the deterministic member and the two members of the stochastic bank with cutoff 0) in the LCHS archive
format. The ``saved_phases`` fixture serves the phases of a requested target only when its polynomial and recovery
(the fields in ``DESCRIPTOR``) equal a saved one, its phase allowance and error bound agree within rounding and the
saved residual meets that allowance, and any phase solve fails the test. The error bounds of a served target come
from the requested plan. The targets were
fitted once by the library's own solver; every assertion below compares the circuits, rows and records built
from them with independent matrix exponentials, closed forms or counted events.
"""

import json
import math
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm
from test_ibm_backend import logical_row_operator, runtime_api, template_row_outcome  # noqa: F401  (fixture)

import nwqlib
from nwqlib import LinearDynamics, plan, solve
from nwqlib._limits import ResourceLimitError
from nwqlib.algorithms.lchs import LCHS, PTSC, ProviderConfig, SampledPairs, Trotter, TrotterErrorBudgeted
from nwqlib.operators import ingest_cartesian, ingest_pauli
from nwqlib.problems.records import Accuracy, NormalizedExpectation, NormSquared, QuadraticForm

# 0.2 rounded so that 0.1 * X02 == 0.02 in binary64, the tolerance at which the saved banks were planned.
X02 = next(float(v) for v in (.2, np.nextafter(.2, 0), np.nextafter(.2, 1)) if .1 * v == .02)
LIMITS = dict(max_bytes=32 * 2**20)
DESCRIPTOR = ("kind", "polynomial_time", "coefficients", "minimum_power", "signals", "negative_signals", "scale",
              "norm_bound")
# Relative tolerance for epsilon_phase and approximation_error_bound in this fixture's plans, with u = 2**-53.
# Assume normal binary64 round-to-nearest arithmetic and normalized-component division error at most 2u relative.
# Assume summation error at most 2u relative, special functions within 4 ulp, and unchanged degree, terminal
# and remainder branch. These plans have tau=0.4 and unnormalized coefficient one-norm below 1.4.
# The deterministic approximation_error_bound has a relative tail and normalization budget below 2600u.
# For the stochastic members, the coefficient-difference sum changes by at most 64u * tail_mass across platforms.
# With tail_mass<0.432, their relative approximation_error_bound budgets are below 512u and 64u.
# At allowance=0.02 and mean_error<0.00281, the relative epsilon_phase budget is below 2304u.
# Use 4096u for both comparisons, which still refuses a relative change of 1e-12.
SAVED_PLAN_RTOL = 2.0**-41
SIGNED = ProviderConfig(implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": -1})
STOCHASTIC0 = {"implementation": "stochastic", "parameters": {"cutoff": 0}}
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
OBSERVABLE = ingest_pauli([("X", .7), ("Z", -.3)], num_qubits=1)
# A = 0.4i Z: alpha_H = 0.4, so the bank time is b_max = alpha_H T = 0.4 for T = 1.
UNITARY = LinearDynamics(A=ingest_pauli([("Z", .4j)], num_qubits=1), initial_state=[.6, .8], time=1.)
# Noncommuting L = 0.2 I + 0.2 X (centered 0.2 X) and H = 0.2 Z with a source; on the signed grid
# (nodes {-1, -.5, 0, .5}) b_max = 1 * 0.2 + 0.2.
MIXED = LinearDynamics(A=ingest_pauli([("I", .2), ("X", .2), ("Z", .2j)], num_qubits=1), initial_state=[1, 0],
                       source=[0, 1j], time=1.)
# H = 0 with centered L = 0.4 X: the node k = 0 gives a zero branch, b_max = 0.4.
ZERO = LinearDynamics(A=ingest_pauli([("I", .5), ("X", .4)], num_qubits=1), initial_state=[.6, .8], source=[0, 1j],
                      time=1.)


def _psd_sqrt(matrix):
    w, v = np.linalg.eigh(matrix)
    return v @ np.diag(np.sqrt(np.clip(w, 0, None))) @ v.conj().T


def _supplied(block, alpha, name, *, error_bound, hermitian=True, lower_bound=None, rotation=False):
    """A public supplied encoding: the Halmos unitary dilation of C = block/alpha, ancilla low.

    ``ingest_block_encoding`` records no full-involution fact, so a supplied Hermitian part takes the generic
    lift. The Halmos dilation of a Hermitian C is [[C, S], [S, -C]] with S = sqrt(I - C**2), a Hermitian
    involution. ``rotation`` (Hermitian C only) negates its second block column, giving the rotation
    completion [[C, -S], [S, C]]: unitary because C and S commute, with the same selected block C, but its square
    [[C**2 - S**2, -2 C S], [2 C S, C**2 - S**2]] is not I unless S = 0, so a walk that assumed an involution
    would apply the wrong operator.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from nwqlib.core.records import InputRef, Source
    from nwqlib.operators import ingest_block_encoding
    from nwqlib.subroutines.block_encoding.core import BlockEncoding
    c = np.asarray(block, dtype=complex) / alpha
    eye = np.eye(len(c))
    u = np.block([[c, _psd_sqrt(eye - c @ c.conj().T)], [_psd_sqrt(eye - c.conj().T @ c), -c.conj().T]])
    if rotation:
        u[:, len(c):] *= -1
    order = np.arange(2 * len(c)).reshape(2, len(c)).T.reshape(-1)
    circuit = QuantumCircuit(int(np.log2(len(c))) + 1, name=name)
    circuit.append(UnitaryGate(u[np.ix_(order, order)]), circuit.qubits)
    encoding = BlockEncoding(circuit=circuit, alpha=alpha, num_ancillas=1, system_qubits=circuit.num_qubits - 1,
                             error_bound=error_bound, implementation=name, metadata={})
    return ingest_block_encoding(encoding, hermitian=hermitian, lower_bound=lower_bound, reference=InputRef(
        identity=f"test:{name}", representation="block_encoding",
        source=Source(name="test", version="1", domain="test", reference="declared")))


# A supplied Hermitian A = L with alpha_L = 0.4 whose block carries the anti-Hermitian defect 0.01i X:
# b_max = T K_* alpha_L = 0.4, and the generic lift keeps only the Hermitian part L.
L_SUPPLIED = np.array([[.1, .25], [.25, -.15]])
SUPPLIED = LinearDynamics(A=_supplied(L_SUPPLIED + .01j * PAULI["X"], .4, "supplied_a", error_bound=.01),
                          initial_state=[1, 0], source=[0, 1j], time=1.)
# A general A = 0.1 X + 0.08i Z given only as one supplied encoding with alpha_A = 0.2. The route builds two
# radius-one children of A with coefficients 1 and -i, L = Herm(A) = 0.1 X and H = Herm(-iA) = 0.08 Z, each a
# full-unitary involution normalized by alpha_A, so b_max = T alpha_A (K_* + 1) = 0.4.
A_GENERAL = .1 * PAULI["X"] + .08j * PAULI["Z"]
GENERAL_A = LinearDynamics(A=_supplied(A_GENERAL, .2, "general_a", error_bound=0., hermitian=False),
                           initial_state=[1, 0], source=[0, 1j], time=1.)
# Native L = 0.2 I + 0.2 X beside a supplied H = 0.18 Z with alpha_H = 0.2 in the rotation completion
# (C = 0.9 Z, S = sqrt(0.19) I), which is not an involution: b_max = 0.2 K_* + 0.2 = 0.4, and every branch with
# k != 0 holds both parts, whose SELECT is then no involution either.
SUPPLIED_H = LinearDynamics(A=ingest_cartesian(ingest_pauli([("I", .2), ("X", .2)], num_qubits=1),
                                               _supplied(.18 * PAULI["Z"], .2, "supplied_h", error_bound=0.,
                                                         rotation=True)),
                            initial_state=[1, 0], source=[0, 1j], time=1.)
# name: (problem, centered generator of a branch at node k, unit directions of the initial (0) and source (1)
# inputs, extracted identity coefficients (c_L, c_H); a supplied part extracts none)
CASES = {
    "mixed": (MIXED, lambda k: k * .2 * PAULI["X"] + .2 * PAULI["Z"], ([1, 0], [0, 1j]), (.2, 0.)),
    "zero": (ZERO, lambda k: k * .4 * PAULI["X"], ([.6, .8], [0, 1j]), (.5, 0.)),
    "supplied": (SUPPLIED, lambda k: k * L_SUPPLIED, ([1, 0], [0, 1j]), (0., 0.)),
    "general_a": (GENERAL_A, lambda k: k * .1 * PAULI["X"] + .08 * PAULI["Z"], ([1, 0], [0, 1j]), (0., 0.)),
    "supplied_h": (SUPPLIED_H, lambda k: k * .2 * PAULI["X"] + .18 * PAULI["Z"], ([1, 0], [0, 1j]), (.2, 0.)),
}


def _sampled(pairs=SampledPairs(), **options):
    return LCHS(estimator=pairs, ham_evo="qsp_block_encoding",
                **{"approx_tol": X02, **LIMITS, **options})


def _signed(pairs=None, **options):
    """The SIGNED one-node sampled QSP Method; `pairs` holds further SampledPairs options."""
    return _sampled(SampledPairs(source_time="discrete", **(pairs or {})), k_quadrature=SIGNED, duhamel_nodes=1,
                    **options)


def _within_rounding(saved, requested):
    """Return whether two values agree within ``SAVED_PLAN_RTOL``, or are both unavailable."""
    if saved is None or requested is None:
        return saved is requested
    return math.isclose(saved, requested, rel_tol=SAVED_PLAN_RTOL, abs_tol=0.)


def _saved_phases_apply(saved, target, requested):
    """Return whether ``target``, the phases solved for the plan ``saved``, can serve the plan ``requested``.

    The polynomial and recovery must match exactly, because the saved residual belongs to that polynomial, and that
    residual must meet the requested allowance. ``epsilon_phase`` and ``approximation_error_bound`` come from tail and
    normalization calculations that round differently across platforms (the deterministic bound by 6 ulp between macOS
    arm64 and Linux x86-64), so they agree within ``SAVED_PLAN_RTOL``, and the served target takes its bounds from
    ``requested``. A defect that moves a bound by less than that tolerance escapes this comparison, such as omitting
    the normalization term of the deterministic bound (relative size 4e-14). In the solver ``full_ja[1]`` enters the
    recorded evidence of the ideal rule and the decision to stop trying longer FFTs after a failed residual check, not
    the phases or their residual, so ``full_ja`` only has to keep its relation to the requested plan.
    """
    allowance, bound, full_ja = requested.epsilon_phase, requested.approximation_error_bound, requested.full_ja
    return (all(getattr(saved, name) == getattr(requested, name) for name in DESCRIPTOR)
            and _within_rounding(saved.epsilon_phase, allowance)
            and target.synthesis_residual is not None and 0 <= target.synthesis_residual <= allowance
            and _within_rounding(saved.approximation_error_bound, bound)
            and (saved.full_ja is None) == (full_ja is None)
            and full_ja in (None, (requested.scale, bound)))


@pytest.fixture
def saved_phases(monkeypatch):
    """Serve bank targets from the saved fixture when `_saved_phases_apply` allows it, and return the served targets."""
    from nwqlib.algorithms.lchs import archive, sampled_qsp_bank
    from nwqlib.subroutines.qsp import phases
    from nwqlib.subroutines.qsp.evolution import QubitizedTarget
    # As in prepare_qubitized_target, every field of a target but its phases and their residual comes from its plan.
    from_plan = {field.name for field in fields(QubitizedTarget)} - {"solution", "synthesis_residual"}
    codec = archive._SelectedData(None, None)
    saved = [(codec.read(item["plan"]), codec.read(item["target"])) for item in json.loads(
        (Path(__file__).parent / "data" / "lchs_sampled_qsp_targets.json").read_text())]
    served = []

    def serve(target_plan):
        match = next((t for p, t in saved if _saved_phases_apply(p, t, target_plan)), None)
        if match is None:
            differences = [{name: getattr(target_plan, name)
                            for name in (*DESCRIPTOR, "epsilon_phase", "approximation_error_bound")
                            if getattr(p, name) != getattr(target_plan, name)} for p, _ in saved]
            pytest.fail(f"no saved phases for {target_plan.kind} at tau={target_plan.polynomial_time}: the requested "
                        f"values that differ from each saved plan are {differences}")
        target = replace(match, **{name: getattr(target_plan, name) for name in from_plan})
        served.append(target)
        return target

    def forbidden(*args, **kwargs):
        pytest.fail("a phase solve in a test that uses saved phases")
    monkeypatch.setattr(sampled_qsp_bank, "prepare_qubitized_target", serve)
    monkeypatch.setattr(phases, "solve_laurent_qsp_phases", forbidden)
    monkeypatch.setattr(phases, "_solve", forbidden)
    return served


def _pauli(label):
    """The matrix of a Pauli label, its first letter on the highest qubit as in Qiskit."""
    matrix = np.eye(1)
    for letter in label:
        matrix = np.kron(matrix, PAULI[letter])
    return matrix


def _row_means(selected, row, members=None):
    """Return the row means (Z_Q, Z_N) of ``row`` from the exact output distribution of its pair circuit.

    ``members`` replaces the row's drawn member pair. The scores go through the production consumer
    `sampled_moments.row_scores` with the Plan's flag mask; probabilities enter as integer counts of 2**50 shots.
    """
    from qiskit.quantum_info import Statevector
    from nwqlib.core.records import FrozenArray
    from nwqlib.algorithms.lchs.sampled_moments import row_scores
    from nwqlib.algorithms.lchs.sampled_qsp import build_qsp_pair
    rec, payload = selected.reconstruction, selected._native["native_data"]
    rows, bank = rec.pair_rows, rec.pair_qsp
    choice = SimpleNamespace(**{name: getattr(bank, name) for name in ("bucket", "alpha_l", "alpha_h", "flag_qubits",
                                                                        "member_x", "member_y")})
    if members is not None:
        mx, my = bank.member_x.array.copy(), bank.member_y.array.copy()
        mx[row], my[row] = members
        choice.member_x, choice.member_y = FrozenArray(mx), FrozenArray(my)
    circuit = build_qsp_pair(payload, SimpleNamespace(pair_rows=rows, pair_qsp=choice), selected.problem, row)
    probabilities = Statevector.from_label("0" * circuit.num_qubits).evolve(circuit).probabilities()
    counts = np.round(probabilities * 2**50).astype(np.int64)
    keep = counts > 0
    term = int(rows.term.array[row])
    label = rec.terms[term][0] if term >= 0 else "I"
    parity = sum(1 << (1 + j) for j, axis in enumerate(reversed(label)) if axis != "I")
    sums = row_scores(np.flatnonzero(keep), counts[keep], branch_mask=1, parity_mask=parity,
                      magnitude=float(rows.score_magnitudes()[row] * bank.row_factors()[row]),
                      signed_range=float(rows.signed_range.array[row]),
                      good_mask=((1 << bank.flag_qubits) - 1) << (1 + int(math.log2(rec.dimension))))
    return sums[0] / counts.sum(), sums[1] / counts.sum(), label


def _check_rows(selected, generator, psi):
    """Return the worst ratios of the member-pair and law-mean deviations to their bounds, and the cross terms.

    The exact row means are Z_Q = h a Re(e^{i theta} <chi_x|P|chi_y>) and Z_N = a Re(e^{i theta} <chi_x|chi_y>),
    with chi = exp(-i t G(k)) psi from 2x2 matrix exponentials for each branch's recorded elapsed time t, node k
    and input direction psi, and theta the recorded phase difference. Every member pair of the stored integer
    law is compared with them within (d_x + d_y + d_x d_y) |h| a for Z_Q and a for Z_N, d = recovered
    individual error plus the generator error (B15); the law mean within 2 beta + beta**2 with beta the
    bucket's recovered mean error plus the generator error (B12, B13). A both-zero row (bucket -1) is exact
    up to rounding.
    """
    rec, payload = selected.reconstruction, selected._native["native_data"]
    rows, bank = rec.pair_rows, rec.pair_qsp
    worst, crosses = dict(member=0., mean=0., zero=0.), []

    def exact(row, label):
        theta = math.remainder(rows.phase_y.array[row] - rows.phase_x.array[row], math.tau)
        chi = [expm(-1j * t * generator(k)) @ np.asarray(psi[kind], dtype=complex) for t, k, kind in (
            (rows.time_x.array[row], rows.node_x.array[row], rows.kind_x.array[row]),
            (rows.time_y.array[row], rows.node_y.array[row], rows.kind_y.array[row]))]
        a, h = float(rows.score_magnitudes()[row]), float(rows.signed_range.array[row])
        cross = np.exp(1j * theta) * np.vdot(chi[0], chi[1])
        crosses.append(cross)
        q = h * a * np.real(np.exp(1j * theta) * np.vdot(chi[0], _pauli(label) @ chi[1]))
        return q, a * cross.real, h, a

    for row in range(rows.clusters):
        b = int(bank.bucket.array[row])
        if b < 0:
            q, n, label = _row_means(selected, row)
            exact_q, exact_n, _, _ = exact(row, label)
            worst["zero"] = max(worst["zero"], abs(q - exact_q), abs(n - exact_n))
            continue
        bucket = payload.qsp_bank.buckets[b]
        law = [count / 2**52 for count in bucket.plan.counts]
        individual = [t.approximation_error_bound + t.scale * t.synthesis_residual + bank.generator_error
                      for t in bucket.members]
        mean_q = mean_n = 0.
        for mx in range(len(law)):
            for my in range(len(law)):
                q, n, label = _row_means(selected, row, (mx, my))
                exact_q, exact_n, h, a = exact(row, label)
                bound = individual[mx] + individual[my] + individual[mx] * individual[my]
                worst["member"] = max(worst["member"], abs(q - exact_q) / (abs(h) * a * bound),
                                      abs(n - exact_n) / (a * bound))
                mean_q, mean_n = mean_q + law[mx] * law[my] * q, mean_n + law[mx] * law[my] * n
        beta = bucket.recovered_mean_error + bank.generator_error
        bound = 2 * beta + beta * beta
        worst["mean"] = max(worst["mean"], abs(mean_q - exact_q) / (abs(h) * a * bound),
                            abs(mean_n - exact_n) / (a * bound))
    return worst, crosses


def _assert_row_weights(selected, problem, c_l, c_h):
    """Each saved branch weight and phase equals the LCHS coefficient |a|/(pi p) and arg a, computed here.

    a = M_i c_j exp(-i t (k (c_L + s) + c_H)) with the extracted identity coefficients (c_L, c_H), the signed
    shift s = psd_shift - scalar_decay, the initial mass ||u0|| exp(s T) and the one Gauss-Legendre source
    node r = T/2 with mass T ||b|| exp(s (T - r)) (``duhamel_nodes=1``).
    """
    from nwqlib.algorithms.lchs.sampled_rows import integer_categorical
    from nwqlib.algorithms.lchs.selection import select_compact
    rec = selected.reconstruction
    rows, time = rec.pair_rows, float(problem.elapsed_time)
    g = rows.scale.mantissa * 2.0 ** rows.scale.exponent
    shift = rec.psd_shift - rec.scalar_decay
    data = select_compact(selected.method, problem)
    k, c = np.asarray(data.quadrature.k_nodes), np.asarray(data.quadrature.coefficients)
    u0, b = (state.preparation.physical_scale.mantissa * 2.0 ** state.preparation.physical_scale.exponent
             for state in (problem.initial_state, problem.source))
    applications = [(u0 * np.exp(shift * time), time, 0), (time * b * np.exp(shift * time / 2), time / 2, 1)]
    with np.errstate(divide="ignore"):
        p = integer_categorical(np.log(np.abs(c))).counts / 2.0**52
    masses = np.array([mass for mass, _, _ in applications])
    pi = integer_categorical(np.log(masses / masses.max())).counts / 2.0**52
    for side in ("x", "y"):
        for kind, t, node, magnitude, angle in zip(*(getattr(rows, f"{name}_{side}").array for name in (
                "kind", "time", "node", "magnitude", "phase"))):
            j = int(np.flatnonzero(k == node)[0])
            category = next(i for i, (_, elapsed, branch) in enumerate(applications) if branch == kind and elapsed == t)
            a = masses[category] * c[j] * np.exp(-1j * t * (node * (c_l + shift) + c_h))
            assert magnitude * g * pi[category] * p[j] == pytest.approx(abs(a), rel=1e-12, abs=0)
            assert abs(np.exp(1j * angle) - a / abs(a)) < 1e-12


def _count_gates(circuit, name):
    """Count instructions named ``name`` at any depth of custom definitions, without expanding standard gates."""
    from qiskit.circuit import ControlledGate
    from qiskit.circuit.library import get_standard_gate_name_mapping
    standard = set(get_standard_gate_name_mapping()) | {"unitary", "initialize", "state_preparation", "measure",
                                                         "barrier", "reset"}
    total = 0
    for item in circuit.data:
        operation = item.operation
        if operation.name == name:
            total += 1
        elif operation.name in standard or isinstance(operation, ControlledGate) and \
                operation.base_gate.name in standard:
            continue
        elif operation.definition is not None:
            total += _count_gates(operation.definition, name)
    return total


def _supplied_calls(circuit, name):
    """Count the controlled calls of the supplied circuit ``name`` and of its adjoint (``name_dg``) at any depth.

    The opaque query controls the ingested circuit gate by gate and reverses it for the adjoint, so each call is
    one ControlledGate whose base gate carries that name; a counted call is not entered again.
    """
    from qiskit.circuit import ControlledGate
    counts = np.zeros(2, dtype=np.int64)
    for item in circuit.data:
        operation = item.operation
        base = operation.base_gate.name if isinstance(operation, ControlledGate) else None
        if base in (name, name + "_dg"):
            counts[int(base != name)] += 1
        elif getattr(operation, "_standard_gate", None) is None and operation.name != "unitary" \
                and operation.definition is not None:
            counts += _supplied_calls(operation.definition, name)
    return counts


@pytest.mark.parametrize("case, bank", [("mixed", "deterministic"), ("mixed", STOCHASTIC0), ("zero", "deterministic"),
                                        ("supplied", "deterministic"), ("general_a", "deterministic"),
                                        ("supplied_h", "deterministic")],
                         ids=["mixed-deterministic", "mixed-stochastic", "zero-branch", "supplied-lift", "general-a",
                              "supplied-h"])
def test_sampled_qsp_rows_match_exact_cross_terms_and_count_their_queries(saved_phases, case, bank, monkeypatch):
    """Each built row circuit realizes its recorded branch pair, weights and calls, against independent oracles.

    Rows mix initial and source inputs at unequal times with signed nodes. Every member pair of the stored
    integer law stays within B15 of the exact cross term, and the finite law mean within B13, for Z_Q and Z_N
    alike; the stochastic bank's individual errors exceed its mean error tenfold, so only the law mean, not each
    member, meets the B13 bound. The cross terms have nonzero imaginary parts, so dropping the branch phase or a
    conjugation changes Z_N. In the zero-branch case H = 0, so a node k = 0 gives a zero branch D(0) (x) I; its
    rows mix zero and nonzero branches under the common bucket recovery, and a both-zero row (bucket -1) is
    exact. The supplied case encodes L with a defect and no involution fact, so its branches take the generic
    lift with forward and adjoint calls of the supplied circuit; the coherent route refuses that input. The
    general-A case gives one supplied encoding of a non-Hermitian A, whose two children (coefficients 1 and -i)
    are involutions and take the direct walk; a sign or normalization error in a child changes the action. The
    supplied-H case puts a native L and a supplied H that is not an involution in one branch, which must take
    the generic lift; a branch that took the direct walk would apply another operator. The row oracles run
    before the route and census assertions, so a wrong route fails at its action first.

    Each row's saved weight and phase is the LCHS coefficient (`_assert_row_weights`). Row templates bind to each
    row's own circuit within 1e-12, including rows of one template with different nodes. The resource census
    equals the repeats m = 2 times the scaled-query gates that the row circuits emit or, with a supplied
    encoding, times their forward and adjoint calls of the supplied circuit. Per branch of query degree d
    these are d of each for a supplied Hermitian part, at k != 0 when it is L and at every node when it is H,
    and for a general A 2d of each at k != 0, where the branch applies both children, and d at k = 0. The
    expected pair signals are 2 sum_b pi_b d_b over the stored integer member law, computed here.

    Missing encoding errors are checked by planning once. Without an error bound, alpha is no norm bound: a
    supplied general A then has no lower bound for L and a supplied L only its asserted lower bound, so both are
    refused at the PSD premise of L, while a supplied H beside a bounded native L plans with an unknown inner
    bias, not zero.
    """
    from qiskit.quantum_info import Operator
    from nwqlib.algorithms.lchs.quantum_resources import sampled_pair_resources
    from nwqlib.algorithms.lchs.sampled_plan import construct_pair, point_row
    from nwqlib.algorithms.lchs.sampled_qsp import row_branches
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.subroutines.qsp.evolution import plan_qubitized_evolution
    problem, generator, psi, (c_l, c_h) = CASES[case]
    selected = plan(problem, method=_signed(dict(qsp_bank=bank, repeats=2)),
                    output=QuadraticForm(observable=OBSERVABLE), shots=48, seed=5)
    rec, payload = selected.reconstruction, selected._native["native_data"]
    rows, record = rec.pair_rows, rec.pair_qsp
    assert len(set(zip(rows.kind_x.array, rows.kind_y.array))) > 2 and len(set(rows.time_x.array)) == 2
    assert np.any(rows.node_x.array < 0)
    _assert_row_weights(selected, problem, c_l, c_h)
    worst, crosses = _check_rows(selected, generator, psi)
    assert worst["member"] <= 1 and worst["mean"] <= 1 and worst["zero"] <= 1e-12
    assert max(abs(cross.imag) for cross in crosses) > .05
    if case == "zero":
        zero_x, zero_y = rows.node_x.array == 0, rows.node_y.array == 0
        assert np.any(zero_x != zero_y) and np.any(zero_x & zero_y)
        assert np.all(record.bucket.array[zero_x & zero_y] == -1)
    if record.policy == "stochastic":
        assert len(record.counts[0]) == 2 and record.individual_errors[0] > 10 * record.mean_errors[0]

    block = next(item for item in selected.blocks if item.record.signature.name == "lchs_pair")
    points = tuple(selected._static_points())
    rows_of = {point.key: point_row(point) for point in points}
    templates = selected.method.row_templates(selected, points)
    for template in templates:
        for key, values in template.rows:
            bound = template.circuit.assign_parameters(dict(zip(template.parameters, values)))
            direct = construct_pair(block, (("row", rows_of[key]),), None)
            assert np.max(np.abs(Operator(bound).data - Operator(direct).data)) <= 1e-12
    assert any(len({(rows.node_x.array[rows_of[k]], rows.node_y.array[rows_of[k]]) for k, _ in t.rows}) > 1
               for t in templates)

    import nwqlib.subroutines.qsp.evolution as evolution_module
    planned = []

    def counted(*args, **kwargs):
        planned.append(1)
        return plan_qubitized_evolution(*args, **kwargs)

    # Branches with one query choice, member and bucket time share one evolution plan.
    monkeypatch.setattr(evolution_module, "plan_qubitized_evolution", counted)
    resources = sampled_pair_resources(selected)
    monkeypatch.undo()
    branches = sum(len(row_branches(payload, rows, record, row) or ()) for row in range(rows.clusters))
    assert 0 < len(planned) < branches
    circuits = [construct_pair(block, (("row", row),), None) for row in range(rows.clusters)]
    if case in ("supplied", "general_a", "supplied_h"):
        supplied = problem.A.cartesian_parts().hamiltonian if case == "supplied_h" else problem.A
        name = supplied.supplied_block_encoding().encoding.circuit.name
        emitted, table = sum(_supplied_calls(circuit, name) for circuit in circuits), np.zeros(2, dtype=np.int64)
        for row in range(rows.clusters):
            for access, target, time in row_branches(payload, rows, record, row) or ():
                evolution = plan_qubitized_evolution(access, target, evolution_time=time)
                if case == "general_a":
                    assert evolution.signal_access == "direct_involution"
                    table += (2 if access.node else 1) * evolution.degree
                else:
                    lift = access.scale != 0 and (access.node != 0 or case == "supplied_h")
                    assert (evolution.signal_access == "generic_lift") == lift
                    table += evolution.degree if lift else 0
        census = np.array([resources["supplied_forward_calls_total"], resources["supplied_adjoint_calls_total"]])
        assert np.all(census == 2 * emitted) and np.all(census == 2 * table) and census[0] > 0
    else:
        emitted = sum(_count_gates(circuit, "scaled_branch") for circuit in circuits)
        assert emitted * 2 == resources["scaled_query_calls_total"] > 0
    if case == "supplied":
        with pytest.raises(ApplicabilityError, match="supplied block encoding of L or H"):
            plan(problem, method=LCHS(ham_evo="qsp_block_encoding"),
                 output=QuadraticForm(observable=OBSERVABLE), shots=8)
        unknown = dict(initial_state=[1, 0], source=[0, 1j], time=1.)
        for refused in (_supplied(A_GENERAL, .2, "general_a_unknown", error_bound=None, hermitian=False),
                        ingest_cartesian(_supplied(.2 * PAULI["X"], .2, "supplied_l_unknown", error_bound=None,
                                                   lower_bound=-.2), ingest_pauli([("Z", .2)], num_qubits=1))):
            with pytest.raises(ApplicabilityError, match="PSD premise of L"):
                plan(LinearDynamics(A=refused, **unknown), method=_signed(),
                     output=QuadraticForm(observable=OBSERVABLE), shots=8)
        unknown_h = ingest_cartesian(ingest_pauli([("I", .2), ("X", .2)], num_qubits=1),
                                     _supplied(.2 * PAULI["Z"], .2, "supplied_h_unknown", error_bound=None))
        assert plan(LinearDynamics(A=unknown_h, **unknown), method=_signed(),
                    output=QuadraticForm(observable=OBSERVABLE), shots=8).reconstruction.pair_bias.inner_bias is None
    if case == "mixed":
        members = payload.qsp_bank.plan.buckets[0]
        law = np.asarray(members.counts) / 2**52
        degrees = np.asarray([member.target.signals for member in members.members])
        assert record.expected_pair_signals == pytest.approx(2 * float(law @ degrees), rel=1e-14)
        assert resources["maximum_pair_signals"] == 2 * degrees.max()


def test_sampled_qsp_public_outputs_follow_the_dense_solution(saved_phases):
    """The QSP Plan's rows weigh each branch by |a|/(pi p) and its biases bound the public physical outputs.

    A = (0.3 + 0.1i) I + 0.4i Z with u0 = (.6, .8i), b = (.3, -.4i), T = 1 and the default kernel grid, whose
    model bias is known; the centered generator is H = 0.4 Z, so b_max = 0.4. Independently of the bank, the
    LCHS vector v = sum_{i,j} M_i c_j exp(i phi) U(t_i) psi_i with phi = -t (k (0.3 + s) + 0.1), s = psd_shift -
    scalar_decay, the initial mass ||u0|| exp(s T) and the Gauss-Legendre source masses w_q ||b|| exp(s (T - r_q))
    gives ||v||**2 and v^dagger O v within the published model bias of the dense ODE solution
    u(T) = exp(-A T) u0 + int_0^T exp(-A (T - r)) b dr, and each saved row magnitude times g equals
    |a|/(pi_i p_j), so the scores are unbiased. The bank enters through the published inner bias, which must
    equal E|w|**2 (2 beta + beta**2) up to rounding, E|w| = (sum_i M_i)(sum_j |c_j|), beta = recovered mean error
    plus generator error (B13). With the row containment of the test above, the law mean of the estimator
    lies within inner plus model bias of the physical value.
    """
    from nwqlib.algorithms.lchs.sampled_rows import integer_categorical
    from nwqlib.algorithms.lchs.selection import select_compact
    u0, b, time = np.array([.6, .8j]), np.array([.3, -.4j]), 1.
    A = (.3 + .1j) * PAULI["I"] + .4j * PAULI["Z"]
    problem = LinearDynamics(A=ingest_pauli([("I", .3 + .1j), ("Z", .4j)], num_qubits=1), initial_state=u0,
                             source=b, time=time)
    method = _sampled(SampledPairs(source_time="discrete"), duhamel_nodes=2)
    selected = plan(problem, method=method, output=QuadraticForm(observable=OBSERVABLE), shots=24, seed=5)
    rec, payload = selected.reconstruction, selected._native["native_data"]
    rows, bias, bank = rec.pair_rows, rec.pair_bias, rec.pair_qsp
    assert bias.inner_relation == "qsp_bank" and bias.step_rule is None and bias.model_bias is not None
    g = rows.scale.mantissa * 2.0**rows.scale.exponent
    shift = rec.psd_shift - rec.scalar_decay
    data = select_compact(method, problem)
    k, c = np.asarray(data.quadrature.k_nodes), np.asarray(data.quadrature.coefficients)

    def phase(t, node):
        return np.exp(-1j * t * (node * (.3 + shift) + .1))
    nu0, nub = np.linalg.norm(u0), np.linalg.norm(b)
    directions = {0: u0 / nu0, 1: b / nub}
    x, w = np.polynomial.legendre.leggauss(2)
    applications = [(nu0 * np.exp(shift * time), time, 0)] + [
        (weight * nub * np.exp(shift * (time - r)), time - r, 1) for r, weight in zip(time * (x + 1) / 2, time * w / 2)]
    v = sum(mass * cj * phase(t, node) * (expm(-1j * t * .4 * PAULI["Z"]) @ directions[kind])
            for mass, t, kind in applications for cj, node in zip(c, k))
    with np.errstate(divide="ignore"):
        p = integer_categorical(np.log(np.abs(c))).counts / 2.0**52
    masses = np.array([mass for mass, _, _ in applications])
    pi = integer_categorical(np.log(masses / masses.max())).counts / 2.0**52
    for side in ("x", "y"):
        for kind, t, node, magnitude, angle in zip(*(getattr(rows, f"{name}_{side}").array for name in (
                "kind", "time", "node", "magnitude", "phase"))):
            j = int(np.flatnonzero(k == node)[0])
            category = next(i for i, (_, elapsed, branch) in enumerate(applications) if branch == kind and elapsed == t)
            a = masses[category] * c[j] * phase(t, node)
            assert magnitude * g * pi[category] * p[j] == pytest.approx(abs(a), rel=1e-12, abs=0)
            assert abs(np.exp(1j * angle) - a / abs(a)) < 1e-12
    augmented = np.zeros((3, 3), complex)
    augmented[:2, :2], augmented[:2, 2] = -A, b
    u = expm(-A * time) @ u0 + expm(augmented * time)[:2, 2]
    O = .7 * PAULI["X"] - .3 * PAULI["Z"]
    g2 = g * g
    assert abs(np.vdot(v, v).real - np.vdot(u, u).real) <= bias.model_bias * g2
    assert abs(np.vdot(v, O @ v).real - np.vdot(u, O @ u).real) <= bias.observable_bound * bias.model_bias * g2
    beta = payload.qsp_bank.buckets[0].recovered_mean_error + bank.generator_error
    mean_bound = masses.sum() * np.abs(c[p > 0]).sum()
    assert bias.inner_bias * g2 == pytest.approx(mean_bound**2 * (2 * beta + beta**2), rel=1e-9)


def test_good_flag_scores_and_all_bad_rows():
    """Row scores count only good-flag shots, and a row whose shots are all bad completes at zero (B16-B18).

    Projection: two completions with the same selected block give the same good-flag scores but different
    unprojected ones. U_plus = [[s, q], [-q, s]] and U_minus = [[s, -q], [q, s]], s = q = 1/sqrt(2): branch x
    applies U_plus and branch y U_plus or U_minus to one flag qubit. Good-flag means are both s*s = 1/2; ignoring
    the flag gives 1 for equal completions and s*s - q*q = 0 for opposite ones.

    Five sums: bits 0 branch, 1 parity, 2 flag. Good outcomes 0, 1, 2, 3 with counts 3, 2, 5, 7 give M_g = 17,
    S_a = 3 - 2 + 5 - 7 = -1, S_aP = 3 - 2 - 5 + 7 = 3, S_P = 3 + 2 - 5 - 7 = -7; outcomes 4 and 7 are bad.
    The all-bad row has 41 shots too, so the mean over both rows divides -r by 2 * 41.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector
    from nwqlib.algorithms.lchs.sampled_moments import LCHSPairMoments, add_rows, pair_confidence, row_scores
    from nwqlib.core.records import FrozenArray
    from nwqlib.problems.inputs import PhysicalScale
    s = q = 1 / math.sqrt(2)
    plus, minus = np.array([[s, q], [-q, s]]), np.array([[s, -q], [q, s]])
    means = {}
    for name, other in (("same", plus), ("opposite", minus)):
        circuit = QuantumCircuit(2)  # qubit 0 is the branch (bit 0), qubit 1 the flag (bit 1)
        circuit.h(0)
        circuit.append(UnitaryGate(plus).control(1, ctrl_state=0), [0, 1])
        circuit.append(UnitaryGate(other).control(1, ctrl_state=1), [0, 1])
        circuit.h(0)
        counts = np.round(Statevector(circuit).probabilities() * 2**50).astype(np.int64)
        bits = np.arange(4)
        good = row_scores(bits, counts, branch_mask=1, parity_mask=0, magnitude=1., signed_range=1., good_mask=2)
        free = row_scores(bits, counts, branch_mask=1, parity_mask=0, magnitude=1., signed_range=1.)
        means[name] = good[1] / counts.sum(), free[1] / counts.sum()
    assert means["same"][0] == pytest.approx(.5, abs=1e-12) and means["opposite"][0] == pytest.approx(.5, abs=1e-12)
    assert means["same"][1] == pytest.approx(1, abs=1e-12) and means["opposite"][1] == pytest.approx(0, abs=1e-12)

    bits, counts = np.array([0, 1, 2, 3, 4, 7]), np.array([3, 2, 5, 7, 11, 13])
    h, r = .7, 1.3
    sums = row_scores(bits, counts, branch_mask=1, parity_mask=2, magnitude=r, signed_range=h, good_mask=4)
    assert sums == pytest.approx((3 * h * r, -r, 17 * h * h * r * r, 17 * r * r, -7 * h * r * r), rel=1e-15)
    bad = row_scores(np.array([4, 5, 6]), np.array([10, 15, 16]), branch_mask=1, parity_mask=2, magnitude=r,
                     signed_range=h, good_mask=4)
    assert bad == (0., 0., 0., 0., 0.)
    moments = LCHSPairMoments(table_id="sha256:" + "0" * 64, clusters=2, repeats=41,
                              rows=FrozenArray(np.zeros(0, dtype=np.int64)), score_bound_q=2., score_bound_n=2.,
                              scale=PhysicalScale(mantissa=.5, exponent=1))
    complete = add_rows(moments, [0, 1], [sums, bad])
    assert complete.completed == 2 and complete.cluster_n == pytest.approx(-r / 41, rel=1e-15)
    assert pair_confidence(complete, failure=.1, quantities=("n",), bias_n=0.)["n"][0] == pytest.approx(-r / 82)


def test_saved_bank_replay_and_partial_continuation(saved_phases, tmp_path, monkeypatch):
    """A stochastic Plan's Result and a partial Run replay its saved targets, counts, members and rows.

    Solving, loading and continuing neither draw rows, bind members nor prepare phases; the continued moments
    equal the uninterrupted ones exactly.
    """
    from nwqlib import _prepared_execution as acquisition
    from nwqlib._prepared_execution import Run
    from nwqlib.algorithms.lchs import sampled_pair_progress, sampled_plan, sampled_qsp, sampled_qsp_bank, sampled_rows
    chosen = plan(MIXED, method=_signed(dict(qsp_bank=STOCHASTIC0, repeats=3)),
                  output=QuadraticForm(observable=OBSERVABLE), shots=30, seed=7)
    for owner, name in ((sampled_rows, "draw_pair_rows"), (sampled_plan, "draw_pair_rows"),
                        (sampled_qsp, "bind_rows"), (sampled_qsp_bank, "prepare_qubitized_target")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("replay drew rows or members or fitted"))
    single = solve(chosen)
    loaded = nwqlib.load_result(single.save(tmp_path / "single"))
    assert loaded.plan.reconstruction == chosen.reconstruction
    saved, original = loaded.plan._native["native_data"].qsp_bank, chosen._native["native_data"].qsp_bank
    assert saved.plan == original.plan
    assert [[(t.solution, t.scale) for t in b.members] for b in saved.buckets] == \
        [[(t.solution, t.scale) for t in b.members] for b in original.buckets]
    assert (loaded.pair_moments, loaded.pair_estimate, loaded.value) == (
        single.pair_moments, single.pair_estimate, single.value)
    assert loaded.analyze().pair_estimate == single.pair_estimate and single.pair_moments.clusters == 10
    submit, calls = acquisition.submit_experiment, []

    def interrupted(*args, **kwargs):
        if len(calls) == 4:
            raise RuntimeError("stop before next acquisition")
        calls.append(1)
        return submit(*args, **kwargs)
    with Run(chosen, directory=tmp_path / "run") as run:
        with monkeypatch.context() as patch:
            patch.setattr(acquisition, "submit_experiment", interrupted)
            with pytest.raises(RuntimeError, match="before next acquisition"):
                run.resume()
        progress = sampled_pair_progress(run)
    assert progress.completed_rows.array.tolist() == [0, 1, 2, 3] and progress.status == "incomplete"
    calls.clear()
    monkeypatch.setattr(acquisition, "submit_experiment", lambda *a, **k: (calls.append(1), submit(*a, **k))[1])
    with nwqlib.load_run(tmp_path / "run") as run:
        assert sampled_pair_progress(run) == progress
        result = run.wait()
    assert len(calls) == 6

    def numbers(m):
        return m.rows.array.tolist(), m.sum_q, m.sum_n, m.sum_qq, m.sum_nn, m.sum_qn, m.cluster_q, m.cluster_n
    assert numbers(result.pair_moments) == numbers(single.pair_moments)


@pytest.mark.parametrize("options, output", [
    (dict(ham_evo="qsp_block_encoding", estimator=SampledPairs(repeats=3),
          approx_tol=X02, **LIMITS),
     QuadraticForm(observable=ingest_pauli([("X", 1.), ("I", .5)], num_qubits=1))),
    (dict(ham_evo=Trotter(steps=2), estimator=SampledPairs(repeats=3)),
     NormSquared()),
    (dict(ham_evo=TrotterErrorBudgeted(order=1), estimator=SampledPairs(
        repeats=3, product_formula=PTSC(pf_allowance=1e-2, mu_cap=1.5, taylor_cutoff=8))),
     QuadraticForm(observable=ingest_pauli([("X", 1.), ("I", .5)], num_qubits=1)))],
    ids=["qsp-quadratic", "plain-norm", "ptsc-quadratic"])
def test_sampled_accuracy_selects_the_least_row_count_in_one_pass(saved_phases, monkeypatch, options, output):
    """B is the least row count whose Hoeffding radius of the support-wide range meets epsilon (SC1-SC2).

    H = g**2 (W/g)**2 F H_O from the saved row table and the whole-law factor F (max(1, max zeta_b**2) for a
    QSP bank, the recipe's score factor, 1 for plain rows) and H_O including the identity term; radius(B) <=
    epsilon < radius(B - 1), shots B*m with m = 3. The accuracy hook and the Plan share one deterministic
    pass, the Plan draws rows once and prepares a QSP bank once, and the Plan keeps the request. A Method whose
    SampledPairs.fail_prob differs from 1 - confidence is refused, since its radius would state another
    confidence. On the QSP bank, an accuracy whose rows cannot fit max_bytes is refused by a byte gate before
    the bank is prepared or any row drawn.
    """
    from fractions import Fraction
    from nwqlib.algorithms.lchs import sampled_plan, sampled_qsp_bank
    from nwqlib.algorithms.protocol import ApplicabilityError
    counter = dict(forecast=0, draw=0, prepare=0)
    for owner, name, key in ((sampled_plan, "_forecast", "forecast"), (sampled_plan, "draw_pair_rows", "draw"),
                             (sampled_qsp_bank, "prepare_sampled_qsp_bank", "prepare")):
        original = getattr(owner, name)
        monkeypatch.setattr(owner, name, lambda *a, _o=original, _k=key, **k: (
            counter.__setitem__(_k, counter[_k] + 1), _o(*a, **k))[1])
    method = LCHS(**options)
    accuracy = Accuracy(absolute_tolerance=.5, component="sampling", confidence=.9)
    chosen = plan(UNITARY, method=method, output=output, accuracy=accuracy, seed=11)
    rec = chosen.reconstruction
    rows = rec.pair_rows
    factor = Fraction(1)
    if rec.pair_qsp is not None:
        factor = max(Fraction(1), max(Fraction(value) ** 2 for value in rec.pair_qsp.recoveries))
    elif rec.pair_recipe is not None:
        factor = Fraction(rec.pair_recipe.score_factor)
    range_o = Fraction(rows.score_bound_q) / Fraction(rows.weight_bound) ** 2 if isinstance(output, QuadraticForm) \
        else Fraction(1)
    h = float(Fraction(2) ** (2 * (rows.scale.exponent - 1)) * Fraction(rows.weight_bound) ** 2 * factor * range_o)
    clusters = rows.clusters

    def radius(b):
        return h * math.sqrt(2 * math.log(2 / .1) / b)
    assert radius(clusters) <= .5 * (1 + 1e-12) and (clusters == 1 or radius(clusters - 1) > .5)
    assert chosen.shots == 3 * clusters and chosen.selection_accuracy == accuracy
    assert counter["forecast"] == 1 and counter["draw"] == 1
    assert counter["prepare"] == (1 if rec.pair_qsp is not None else 0)
    with pytest.raises(ApplicabilityError, match="SampledPairs.fail_prob"):
        plan(UNITARY, method=method.revise(estimator=method.estimator.revise(
            fail_prob=.2)), output=output, accuracy=accuracy)
    if rec.pair_qsp is None:
        return

    # NormSquared at epsilon = 1e-4: the reached law has frame g = 2, W/g = 0.5 and the deterministic member's
    # recovery zeta = 1.0697, so H = g**2 (W/g)**2 zeta**2 = 1.144 and SC2 needs B = 2 (H/epsilon)**2 log(2/0.1),
    # about 7.8e8 rows. Their coarse row data, 1024 bytes each, is about 8e11 bytes, far above the 32 MiB cap.
    # The Plan's one row axis does not grow with B, so a byte gate (`_sufficient_rows` or the held-data check)
    # refuses it, naming max_bytes.
    def forbidden(*args, **kwargs):
        pytest.fail("an accuracy beyond max_bytes prepared a bank or drew rows")
    monkeypatch.setattr(sampled_plan, "draw_pair_rows", forbidden)
    monkeypatch.setattr(sampled_qsp_bank, "prepare_sampled_qsp_bank", forbidden)
    with pytest.raises(ValueError, match="max_bytes="):
        plan(UNITARY, method=method, output=NormSquared(),
             accuracy=Accuracy(absolute_tolerance=1e-4, component="sampling", confidence=.9))


def test_26_qubit_sampled_qsp_plan_and_estimate_build_nothing(saved_phases, monkeypatch):
    """A 26-system-qubit sampled QSP Plan and its estimate use the saved bank and form no system-sized object.

    A = 0.4i Z_0 on 26 qubits from the zero occupation state: b_max = 0.4, so the saved deterministic member
    serves the bank. No circuit, statevector, operator or dense Pauli matrix is formed, and the traced peak stays
    below 2**26 bytes.
    """
    import tracemalloc
    import qiskit
    from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
    from nwqlib.problems import ingest_occupation

    def forbidden(*args, **kwargs):
        pytest.fail("26-qubit sampled QSP planning formed a circuit or system-sized object")
    for owner, name in ((Statevector, "__init__"), (Operator, "__init__"), (SparsePauliOp, "to_matrix"),
                        (qiskit.QuantumCircuit, "__init__")):
        monkeypatch.setattr(owner, name, forbidden)
    n = 26
    big = LinearDynamics(A=ingest_pauli([("I" * (n - 1) + "Z", .4j)], num_qubits=n),
                         initial_state=ingest_occupation([0] * n, num_qubits=n), time=1.)
    tracemalloc.start()
    try:
        selected = plan(big, method=_sampled(), output=NormSquared(), shots=4, seed=1)
        workload = nwqlib.estimate(selected)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2**26
    assert workload.quantity("shots").fact.value.numerator == 4
    assert len(saved_phases) == 1 and saved_phases[0].polynomial_time == .4
    assert selected.reconstruction.dimension == 2**n


def test_normalized_ratio_unresolved_under_cancellation(saved_phases):
    """A source that cancels the initial evolution at T leaves N = 0: the ratio is unresolved, raw values remain.

    For A = 0.4i Z, u0 = (1, 0) and b = (-a exp(-a)/(1 - exp(-a)), 0) with a = 0.4i, u(T) = exp(-a T) +
    (1 - exp(-a T)) b_0 / a = 0 at T = 1, so ||u(T)||**2 = 0 and no sample size resolves Q/N. The Result has
    no value, says why, and keeps the raw signed N and Q and the covering interval of N clipped at zero.
    """
    a = .4j
    b0 = -a * np.exp(-a) / (1 - np.exp(-a))
    problem = LinearDynamics(A=ingest_pauli([("Z", .4j)], num_qubits=1), initial_state=[1, 0], source=[b0, 0],
                             time=1.)
    exact = expm(-np.diag([a, -a])) @ np.array([1, 0]) + np.array([(1 - np.exp(-a)) * b0 / a, 0])
    assert np.linalg.norm(exact) < 1e-15
    selected = plan(problem, method=_sampled(SampledPairs(source_time="discrete"), duhamel_nodes=8),
                    output=NormalizedExpectation(observable=ingest_pauli([("X", 1.)], num_qubits=1)),
                    shots=20, seed=2)
    result = solve(selected)
    assert result.value is None and "not strictly positive" in result.unavailable
    estimate = result.pair_estimate
    assert estimate.norm_squared is not None and estimate.quadratic_form is not None
    assert estimate.ratio_interval is None
    low, high = estimate.norm_interval
    assert low == 0. < high and estimate.norm_squared - estimate.radius_n <= 0


def test_ibm_rows_keep_pub_identity_and_good_flag_scores(saved_phases, runtime_api, tmp_path,  # noqa: F811
                                                         monkeypatch):
    """Prepare first, then resume: each QSP row's saved template binding returns its own outcome and scores.

    The synthetic service returns, in every shot of a binding row, an outcome hashed from that row's values,
    so each chunk's histogram must be m copies of its own row's outcome. The good-flag scores are recomputed
    here from those outcomes and the logical layout (bit 0 branch, bits 1..q system, q+1..q+F flags):
    Z_N = r I_g s_a and Z_Q = h r I_g s_a s_P per shot, with r the row's magnitude times zeta_b**2.
    Each bound ISA circuit equals its row's own lowered circuit without phase alignment, and each logical template
    is compiled once over prepare and resume: the reopened Run reuses the template saved by prepare.
    """
    from qiskit.quantum_info import Operator
    from nwqlib._run_archive import load as load_run
    from nwqlib.backends import IBMRuntimeBackend
    compiled, compile_isa = [], IBMRuntimeBackend.prepare
    monkeypatch.setattr(IBMRuntimeBackend, "prepare", lambda self, circuit, **options: (
        compiled.append(circuit.num_parameters), compile_isa(self, circuit, **options))[1])
    selected = plan(MIXED, method=_signed(dict(repeats=2)), output=QuadraticForm(observable=OBSERVABLE), shots=32,
                    seed=4)
    rec = selected.reconstruction
    rows, bank = rec.pair_rows, rec.pair_qsp
    points = tuple(selected._static_points())
    names = tuple(point.key for point in points)
    templates = selected.method.row_templates(selected, points)
    assert sum(len(template.rows) for template in templates) == len(names) == rows.clusters
    connection = runtime_api.connection.revise(initial_layout=None)
    api = runtime_api.api
    directory = tmp_path / "sampled-qsp"
    prepared = nwqlib.prepare(selected, backend=connection, directory=directory)
    prepared.run.close()
    assert len(compiled) == 1 and compiled[0] > 0
    with load_run(directory, backend=connection) as run:
        run.resume()
        assert run.result is None and run.trace.jobs == 1 and len(runtime_api.requests) == 1
        assert len(compiled) == len(templates) > 1 and all(compiled)
        records = run.prepared_artifacts
        assert len(records) == len(names)
        for record in records:
            bound = run._state["handles"][record.content_id].inspect_circuit()
            bound.remove_final_measurements()
            expected = logical_row_operator(selected, record.realization, bound.num_qubits)
            assert np.max(np.abs(Operator.from_circuit(bound).data - expected)) <= 1e-10
        api.status = "COMPLETED"
        run.resume()
        result = run.result
        q, flags = 1, bank.flag_qubits
        magnitude = rows.score_magnitudes() * bank.row_factors()
        sums, good, outcomes = np.zeros(2), [], []
        for chunk in run.observations.chunks:
            record = run.prepared_artifact(chunk.prepared_id)
            positions = {register.name: register.bits for register in record.classical_layout}
            outcome = 0
            for index, register in enumerate(record.native_classical_layout):
                value = template_row_outcome(record.template.values, len(register.bits), index)
                outcome |= sum(((value >> bit) & 1) << position
                               for bit, position in enumerate(positions[register.name]))
            histogram = chunk.histogram()
            assert dict(zip(histogram.index_list(), histogram.weights.tolist())) == {outcome: 2}
            row = next(int(item.value) for item in record.realization.bindings if item.parameter == "row")
            label = rec.terms[int(rows.term.array[row])][0]
            ok = (outcome >> (q + 1)) & ((1 << flags) - 1) == 0
            good.append(ok)
            outcomes.append(outcome)
            s_a = 1 - 2 * (outcome & 1)
            s_p = 1 - 2 * (bin(outcome & sum(1 << (1 + j) for j, axis in enumerate(reversed(label))
                                             if axis != "I")).count("1") & 1)
            sums += 2 * ok * magnitude[row] * s_a * np.array([float(rows.signed_range.array[row]) * s_p, 1.])
        assert any(good) and not all(good), (flags, [bin(value) for value in outcomes])
        assert result.pair_moments.sum_q == pytest.approx(sums[0], rel=1e-12, abs=1e-15)
        assert result.pair_moments.sum_n == pytest.approx(sums[1], rel=1e-12, abs=1e-15)


def test_bank_keeps_failure_causes_and_refuses_quadratic_member_arrays_before_allocation(monkeypatch):
    """A failed member preparation or construction keeps its stage and cause, and the member arrays are checked first.

    A stochastic bank whose retained member arrays grow quadratically with the degree is refused against
    max_bytes before any member is allocated or given its phase allowance.
    """
    from nwqlib.algorithms.lchs import sampled_qsp_bank as bank
    from nwqlib.subroutines.qsp import evolution
    from nwqlib.subroutines.qsp.phases import _QSPPhaseLimitError
    limits = dict(max_bytes=32 * 2**20, max_fft=4096)
    kwargs = dict(bucket_times=(.4,), mean_amplitude_allowances=.02, policy="stochastic", cutoffs=(0,))
    planned = bank.plan_sampled_qsp_bank(**kwargs, **limits)
    cause = ValueError("no spent-work evidence")

    def fail(*args, **kwargs):
        raise cause
    monkeypatch.setattr(bank, "prepare_qubitized_target", fail)
    with pytest.raises(bank.SampledQSPBankError) as caught:
        bank.prepare_sampled_qsp_bank(planned)
    assert caught.value.cause is cause and caught.value.stage == "phase preparation"
    assert not isinstance(caught.value, ResourceLimitError)
    # A limit refusal keeps the same stage, member and evidence, and is a limit refusal.
    stop = _QSPPhaseLimitError("the next FFT does not fit", evidence="the attempts")

    def limited(*args, **kwargs):
        raise stop
    with monkeypatch.context() as patch:
        patch.setattr(bank, "prepare_qubitized_target", limited)
        with pytest.raises(ResourceLimitError) as stopped:
            bank.prepare_sampled_qsp_bank(planned)
    assert isinstance(stopped.value, bank.SampledQSPBankError) and stopped.value.cause is stop
    assert stopped.value.stage == caught.value.stage and stopped.value.member_index == caught.value.member_index
    assert stopped.value.evidence == "the attempts"
    with monkeypatch.context() as patch:
        patch.setattr(bank, "plan_laurent_target", fail)
        with pytest.raises(bank.SampledQSPBankError) as construction:
            bank.plan_sampled_qsp_bank(**kwargs, **limits)
    assert construction.value.stage == "member construction"
    assert construction.value.cause is cause
    # A supplied degree-10 expansion with ten nonzero shells; its tail is not evidence about any exponential.
    expansion = replace(evolution.jacobi_anger_expansion(0, .1), tau=.4, degree=10,
                        cos_coefficients=(.4,) + (.01,) * 10, sin_coefficients=(0.,),
                        tail_bound=.001, tail_log_bound=math.log(.001))
    monkeypatch.setattr(bank, "jacobi_anger_expansion", lambda *a, **k: expansion)
    monkeypatch.setattr(bank, "plan_laurent_target", lambda *a, **k: pytest.fail("allocated a member"))
    monkeypatch.setattr(bank, "with_phase_allowance", lambda *a, **k: pytest.fail("scheduled a member"))
    with pytest.raises(bank.SampledQSPBankError) as retained:
        bank.plan_sampled_qsp_bank((.4,), mean_amplitude_allowances=.1, policy="stochastic", cutoffs=(0,),
                                   max_bytes=3000, max_fft=4096)
    assert retained.value.stage == "retained member arrays" and isinstance(retained.value, ResourceLimitError)


def test_bank_checks_the_first_transform_of_its_highest_member_before_any_fit(monkeypatch):
    """The first FFT length of each bucket's highest-degree member is checked when the bank is planned (B9).

    That length, the least power of two above twice the degree, is admitted exactly at max_fft equal to it and
    refused one below, at stage "phase start" with the member, its shell, the cause and the direction, before
    any phase fit.
    """
    from nwqlib.algorithms.lchs import sampled_qsp_bank as bank
    monkeypatch.setattr(bank, "prepare_qubitized_target", lambda *a, **k: pytest.fail("fitted a phase"))
    kwargs = dict(bucket_times=(.4,), mean_amplitude_allowances=.02, policy="stochastic", cutoffs=(0,),
                  max_bytes=32 * 2**20)
    members = bank.plan_sampled_qsp_bank(**kwargs).buckets[0].members
    highest = max(range(len(members)), key=lambda index: members[index].target.signals)
    start = 1 << (2 * members[highest].target.signals).bit_length()
    bank.plan_sampled_qsp_bank(**kwargs, max_fft=start)
    with pytest.raises(bank.SampledQSPBankError) as refused:
        bank.plan_sampled_qsp_bank(**kwargs, max_fft=start - 1)
    error = refused.value
    assert (error.stage, error.bucket_index, error.member_index, error.shell) == (
        "phase start", 0, highest, members[highest].shell)
    assert f"above max_fft={start - 1}" in str(error) and "approx_tol" in str(error)
    assert isinstance(error, ResourceLimitError)


def test_planning_refuses_a_native_branch_query_above_max_bytes_at_its_built_controls():
    """A both-part branch queries each native child under three controls, so planning checks that query's bytes.

    The controls are the branch qubit, the signal qubit of the direct full-involution walk and the combine qubit.
    Both centered parts of this A have m = 6 Pauli terms on n = 3 qubits, padded to a = 3 address qubits, so the
    query holds m (n + 16) + 4 * 64 * 2**(a + 3) (n + 16) bytes (core._admit_native_query). Planning refuses one
    byte less with the builder's message and plans at that size.
    """
    labels = ["XII", "IZI", "IIY", "ZIX", "YIZ", "IXX", "ZYY", "YXI", "XXZ", "ZYI", "XZZ", "IYX"]
    terms = [("III", .5)] + [(label, .0125 if index % 2 == 0 else .025j) for index, label in enumerate(labels)]
    problem = LinearDynamics(A=ingest_pauli(terms, num_qubits=3), initial_state=[1.] + [0.] * 7,
                             source=[0., 1j] + [0.] * 6, time=.5)
    output = QuadraticForm(observable=ingest_pauli([("XII", .7), ("IZI", -.3)], num_qubits=3))
    need = 6 * 19 + 4 * 64 * 2**6 * 19

    def method(cap):
        return LCHS(estimator=SampledPairs(repeats=2, qsp_bank=ProviderConfig(**STOCHASTIC0)),
                    ham_evo="qsp_block_encoding", duhamel_nodes=1, k_quadrature=SIGNED,
                    max_bytes=cap)
    with pytest.raises(ValueError, match=r"block-encoding query needs 114 \+ 4864 \* 2\*\*6 bytes"):
        plan(problem, method=method(need - 1), output=output, shots=16, seed=7)
    plan(problem, method=method(need), output=output, shots=16, seed=7)


@pytest.mark.parametrize("case", ["supplied", "general_a", "supplied_h"])
def test_planning_refuses_the_largest_supplied_branch_query_that_preparation_admits(saved_phases, case, monkeypatch,
                                                                                    tmp_path):
    """A supplied child is queried through the opaque fallback, whose controlled synthesis planning also checks.

    In these one-qubit Plans the synthesis of a controlled supplied query is the largest byte requirement. The
    largest synthesis that preparation admits is refused in planning one byte below and prepares at that size.
    """
    from nwqlib.subroutines import _dense_synthesis
    law, sizes = _dense_synthesis.admit_dense_syntheses, []

    def traced(widths, **kwargs):
        data, work = law(widths, **kwargs)
        if kwargs["operation"] == "opaque block-encoding query synthesis":
            sizes.append(data)
        return data, work
    monkeypatch.setattr(_dense_synthesis, "admit_dense_syntheses", traced)
    problem, output = CASES[case][0], QuadraticForm(observable=OBSERVABLE)
    selected = plan(problem, method=_signed(), output=output, shots=16, seed=7)
    sizes.clear()
    with nwqlib.prepare(selected, directory=tmp_path / "default", progress=False).run:
        pass
    largest = max(sizes)
    with pytest.raises(ValueError, match="opaque block-encoding query synthesis"):
        plan(problem, method=_signed(max_bytes=largest - 1), output=output, shots=16, seed=7)
    selected = plan(problem, method=_signed(max_bytes=largest), output=output, shots=16, seed=7)
    with nwqlib.prepare(selected, directory=tmp_path / "edge", progress=False).run:
        pass


def test_a_supplied_gate_without_a_built_definition_is_planned_as_unknown_and_built_at_preparation(saved_phases,
                                                                                                    tmp_path,
                                                                                                    monkeypatch):
    """Planning reads only built definitions, so it builds no definition of a supplied gate.

    A two-qubit StatePreparation, whose definition Qiskit builds only when asked, is the supplied encoding of a
    general A with normalization 0.2, as in GENERAL_A. The default Plan records the controlled query synthesis
    of that gate as unknown: select_work and the pair block's construction work are None, with the reason, and
    the Plan's estimate reports the construction work as unknown. An explicit max_select_work cannot be checked
    and is refused in planning. The same circuit with its definitions built before intake is priced: half
    the recorded select_work is refused and the recorded value plans. Preparation builds the definition,
    prepares the row and records its construction work as None, which a reopened Run reads back.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry, StatePreparation
    from nwqlib.core.records import InputRef, Source
    from nwqlib.operators import ingest_block_encoding
    from nwqlib._run_archive import load as load_run
    from nwqlib.subroutines.block_encoding.core import BlockEncoding
    defined = []
    for cls in (StatePreparation, Isometry):
        original = cls._define
        monkeypatch.setattr(cls, "_define", lambda self, original=original: defined.append(self.name) or original(self))

    def problem(built):
        circuit = QuantumCircuit(2)
        circuit.append(StatePreparation([.6, 0., .8j, 0.]), circuit.qubits)
        if built:
            pending = [item.operation for item in circuit.data]
            while pending:
                pending += [item.operation for item in pending.pop().definition.data if not item.is_standard_gate()]
        encoding = BlockEncoding(circuit=circuit, alpha=.2, num_ancillas=1, system_qubits=1, error_bound=0.,
                                 implementation="state_preparation", metadata={})
        source = ingest_block_encoding(encoding, hermitian=False, reference=InputRef(
            identity=f"test:state-preparation-{built}", representation="block_encoding",
            source=Source(name="test", version="1", domain="test", reference="declared")))
        return LinearDynamics(A=source, initial_state=[1, 0], source=[0, 1j], time=1.)

    def planned(chosen, **options):
        return plan(chosen, method=_signed(**options), output=QuadraticForm(observable=OBSERVABLE), shots=16, seed=7)
    lazy = problem(False)
    selected = planned(lazy)
    pair = next(item for item in selected.construction.selections if item.signature.name == "lchs_pair")
    assert not defined and selected.reconstruction.select_work is None and pair.construction_work is None
    assert ("controlled query synthesis of supplied operation state_preparation (2 qubits): definition not built "
            "at planning") in pair.cost_context
    assert nwqlib.estimate(selected).quantity("construction_work").fact.availability == "unknown"
    with pytest.raises(ValueError, match=r"so the SELECT construction cannot be counted against "
                                         r"LCHS\.max_select_work=1000\. Expand the supplied circuit"):
        planned(lazy, max_select_work=1000)
    assert not defined
    eager = problem(True)
    defined.clear()
    need = planned(eager).reconstruction.select_work
    with pytest.raises(ValueError, match=f"need {need} work units, above LCHS.max_select_work={need // 2}"):
        planned(eager, max_select_work=need // 2)
    assert planned(eager, max_select_work=need).reconstruction.select_work == need and not defined
    with nwqlib.prepare(selected, directory=tmp_path / "run", progress=False).run as run:
        assert run.trace.preparations == 1 and run.trace.construction_work_reserved is None
        backend = run.backend
    assert defined
    with load_run(tmp_path / "run", backend=backend) as run:
        assert run.trace.preparations == 1 and run.trace.construction_work_reserved is None


def test_a_known_dense_matrix_beside_an_unbuilt_definition_is_priced_at_planning(saved_phases, monkeypatch):
    """Planning skips only the definition that is not built, so the known dense matrix of the same circuit is priced.

    The supplied encoding holds a two-qubit diagonal UnitaryGate and a StatePreparation whose definition is not
    built, in either order. The controlled queries need the bytes of the matrix alone, read from the encoding
    without the StatePreparation, and planning still finds that matrix: max_bytes one byte below the need is
    refused in planning and the need plans, with select_work None because the StatePreparation's synthesis is
    unknown. No definition is built.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, UnitaryGate
    from nwqlib.core.records import InputRef, Source
    from nwqlib.operators import ingest_block_encoding
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.subroutines.block_encoding.core import BlockEncoding
    defined, sizes = [], []
    original_define, original_query = StatePreparation._define, core._opaque_query_requirements
    monkeypatch.setattr(StatePreparation, "_define", lambda self: defined.append(self) or original_define(self))

    def recorded(*args, **kwargs):
        result = original_query(*args, **kwargs)
        sizes.append(result[0])
        return result
    monkeypatch.setattr(core, "_opaque_query_requirements", recorded)

    def planned(order, **options):
        circuit = QuantumCircuit(2)
        for item in order:
            circuit.append(UnitaryGate(np.diag(np.exp(1j * np.array([.1, .2, .3, .4]))))
                           if item == "matrix" else StatePreparation([.6, 0., .8j, 0.]), circuit.qubits)
        encoding = BlockEncoding(circuit=circuit, alpha=.2, num_ancillas=1, system_qubits=1, error_bound=0.,
                                 implementation="state_preparation", metadata={})
        source = ingest_block_encoding(encoding, hermitian=False, reference=InputRef(
            identity="test:" + "/".join(order), representation="block_encoding",
            source=Source(name="test", version="1", domain="test", reference="declared")))
        problem = LinearDynamics(A=source, initial_state=[1, 0], source=[0, 1j], time=1.)
        return plan(problem, method=_signed(**options), output=QuadraticForm(observable=OBSERVABLE), shots=16, seed=7)
    planned(("matrix",))
    need = max(sizes)
    for order in (("lazy", "matrix"), ("matrix", "lazy")):
        with pytest.raises(ValueError, match=f"opaque block-encoding query synthesis needs about {need} bytes"):
            planned(order, max_bytes=need - 1)
        assert planned(order, max_bytes=need).reconstruction.select_work is None
    assert not defined


def test_a_failing_member_refuses_every_seed_whether_or_not_it_is_drawn(saved_phases, monkeypatch):
    """Every member is fitted before the draws, so a failing member refuses the Plan whatever the seed draws.

    The shell-2 member of the two-member stochastic bank is drawn under one seed and not under another. Unpatched,
    the seed that does not draw it still fits it (solver_calls equals the bank's target count). Forced to fail,
    both seeds are refused at stage "phase preparation" with its bucket, member, shell and cause.
    """
    from nwqlib.algorithms.lchs import sampled_qsp_bank
    method, output = _signed(dict(qsp_bank=STOCHASTIC0)), QuadraticForm(observable=OBSERVABLE)
    found = {}
    for seed in range(32):
        chosen = plan(MIXED, method=method, output=output, shots=4, seed=seed)
        record = chosen.reconstruction.pair_qsp
        drawn = set(record.member_x.array.tolist()) | set(record.member_y.array.tolist())
        found.setdefault(1 in drawn, (seed, chosen))
        if len(found) == 2:
            break
    assert set(found) == {True, False}
    unused = found[False][1]
    assert unused.reconstruction.pair_qsp.solver_calls == unused._native["native_data"].qsp_bank.plan.target_count
    serve, forced = sampled_qsp_bank.prepare_qubitized_target, ValueError("forced phase failure of shell 2")

    def fail_shell_two(target_plan):
        if target_plan.minimum_power == -2:
            raise forced
        return serve(target_plan)
    monkeypatch.setattr(sampled_qsp_bank, "prepare_qubitized_target", fail_shell_two)
    for seed, _ in found.values():
        with pytest.raises(sampled_qsp_bank.SampledQSPBankError) as refused:
            plan(MIXED, method=method, output=output, shots=4, seed=seed)
        error = refused.value
        assert (error.stage, error.bucket_index, error.member_index, error.shell) == ("phase preparation", 0, 1, 2)
        assert error.cause is forced and "seed" not in str(error)


def _branch_rounding(chosen):
    """Return, in 60-digit arithmetic, the largest weight and angle discrepancy of the Plan's saved branches (B23).

    A branch of node k and time t has the target weights t|k| alpha_L and t alpha_H. Its circuit realizes
    t_b r_hat c**2 and t_b r_hat (1 - c**2), with r_hat and c the cosines of half the stored scale and combine
    angles that the row uses (c = 1 for one part), and a zero branch realizes t_b r_hat on its whole block.
    Also return the support-wide rounding term b_max rho + eta (3 b_max + 2 T_e + 1).
    """
    mpmath = pytest.importorskip("mpmath")
    from nwqlib.algorithms.lchs.sampled_qsp import (BRANCH_ROUNDING, SMALLEST_SUBNORMAL, _continuous_angles,
                                                    row_branches)
    rec, payload = chosen.reconstruction, chosen._native["native_data"]
    rows, record = rec.pair_rows, rec.pair_qsp
    largest = mpmath.mpf(0)
    with mpmath.workdps(60):
        for row in np.flatnonzero(record.bucket.array >= 0):
            branches = row_branches(payload, rows, record, int(row))
            for (access, _, time), elapsed in zip(branches, (rows.time_x.array[row], rows.time_y.array[row])):
                scale_angle, combine_angle = _continuous_angles(access)
                t, t_b, k = (mpmath.mpf(float(value)) for value in (elapsed, time, access.node))
                w_l = abs(k) * mpmath.mpf(record.alpha_l) if payload.l_child is not None and k != 0 else 0
                w_h = mpmath.mpf(record.alpha_h)
                r_hat = mpmath.cos(mpmath.mpf(scale_angle) / 2)
                if access.scale == 0.0:
                    error = t_b * abs(r_hat) + t * (w_l + w_h)
                elif combine_angle is None:
                    error = abs(t_b * r_hat - t * (w_l + w_h))
                else:
                    c2 = mpmath.cos(mpmath.mpf(combine_angle) / 2) ** 2
                    error = abs(t_b * r_hat * c2 - t * w_l) + abs(t_b * r_hat * (1 - c2) - t * w_h)
                largest = max(largest, error)
        coverage, active = mpmath.mpf(record.coverage), mpmath.mpf(record.active_time)
        support = (coverage * BRANCH_ROUNDING.numerator / BRANCH_ROUNDING.denominator
                   + (3 * coverage + 2 * active + 1) / mpmath.mpf(SMALLEST_SUBNORMAL.denominator))
    return largest, support


def test_generator_term_holds_the_child_bound_once_and_does_not_depend_on_the_draw(saved_phases):
    """beta's generator term is the support-wide child term plus the B23 rounding term, the same for every seed.

    For the supplied L with declared error 0.01 (T = K_* = 1, H = 0) the child term T K_* delta_L = 0.01 enters
    once. For it and for the two-part native generator, two seeds that draw different rows record the same
    generator term and inner bias, and every saved branch's weight and angle discrepancy, evaluated in 60 digits
    from its stored angles, lies below the support-wide rounding term.
    """
    from fractions import Fraction
    from nwqlib.algorithms.lchs.sampled_qsp import BRANCH_ROUNDING, SMALLEST_SUBNORMAL
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    output = QuadraticForm(observable=OBSERVABLE)
    for problem, child in ((SUPPLIED, Fraction(.01)), (MIXED, Fraction(0))):
        chosen = [plan(problem, method=_signed(), output=output, shots=4, seed=seed) for seed in (2, 5)]
        records = [item.reconstruction.pair_qsp for item in chosen]
        assert not all(np.array_equal(getattr(chosen[0].reconstruction.pair_rows, name).array,
                                      getattr(chosen[1].reconstruction.pair_rows, name).array)
                       for name in ("node_x", "node_y", "time_x", "time_y"))
        assert records[0].generator_error == records[1].generator_error
        assert chosen[0].reconstruction.pair_bias.inner_bias == chosen[1].reconstruction.pair_bias.inner_bias
        coverage, active = Fraction(records[0].coverage), Fraction(records[0].active_time)
        expected = child + coverage * BRANCH_ROUNDING + SMALLEST_SUBNORMAL * (3 * coverage + 2 * active + 1)
        assert records[0].generator_error == _upward_float(expected) < float(child) + 1e-13
        for item in chosen:
            largest, support = _branch_rounding(item)
            assert largest <= support


def test_generator_term_covers_a_branch_weight_that_rounds_to_zero():
    """A subnormal normalization rounds a branch's weight and time to zero, and the eta term of B23 covers it.

    A supplied identity L with alpha_L = eta = 2**-1074 and zero error, T = 2**1023 and the signed grid give
    b_max = 2**-51. At k = +-1/2 the weight eta/2 rounds to zero, so the saved branch is the zero block while its
    generator time is T eta/2 = 2**-52 = b_max/2. The recorded generator term is at least that, and every saved
    branch's discrepancy lies below the support-wide rounding term.
    """
    from qiskit import QuantumCircuit
    from nwqlib.core.records import InputRef, Source
    from nwqlib.operators import ingest_block_encoding
    from nwqlib.subroutines.block_encoding.core import BlockEncoding
    eta, time = math.ulp(0.), 2.**1023
    identity = ingest_block_encoding(
        BlockEncoding(circuit=QuantumCircuit(1), alpha=eta, num_ancillas=0, system_qubits=1, error_bound=0.,
                      implementation="identity", metadata={}),
        hermitian=True, lower_bound=0., reference=InputRef(identity="test:subnormal_identity",
            representation="block_encoding", source=Source(name="test", version="1", domain="test",
                                                         reference="declared")))
    chosen = plan(LinearDynamics(A=identity, initial_state=[1., 0.], time=time),
                  method=_sampled(k_quadrature=SIGNED), output=NormSquared(), shots=4, seed=2)
    rec = chosen.reconstruction
    nodes = np.concatenate((rec.pair_rows.node_x.array, rec.pair_rows.node_y.array))
    assert rec.pair_qsp.coverage == 2.**-51 and np.any(np.abs(nodes) == .5)
    assert rec.pair_qsp.generator_error >= 2.**-52
    largest, support = _branch_rounding(chosen)
    assert largest >= 2.**-52 and largest <= support
