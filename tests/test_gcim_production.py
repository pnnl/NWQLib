"""Fixed GCiM transition amplitudes, acquisition association and finite admission."""

import numpy as np
import pytest
from nwqlib.algorithms.gcim import FixedGCIM
from nwqlib.core.planning import RandomStreams
from nwqlib.core.analysis import RunData
from nwqlib._prepared_execution import Run
from nwqlib.execution import ObservationView
from nwqlib.problems.records import Eigenproblem


def plan_for(
    *, A=((1, 0), (0, -1)), basis=((1, 1), (1, 1j)), shots=None, execution="quantum", **settings
):
    problem = Eigenproblem(A=A)
    method = FixedGCIM(basis=basis, **settings)
    return method.plan(
        problem,
        output=problem.default_output(),
        execution=execution,
        shots=shots,
        rng=RandomStreams(19),
    )


def matrix(rows):
    return np.array([[complex(z.real, z.imag) for z in row] for row in rows])


def test_complex_phase_and_identity_share_actual_gram(monkeypatch):
    from nwqlib.operators.inputs import OperatorInput

    monkeypatch.setattr(
        OperatorInput,
        "matvec",
        lambda *a, **k: pytest.fail("quantum GCIM invoked a classical operator"),
    )
    expected_s = np.array([[1, 0.5 + 0.5j], [0.5 - 0.5j, 1]])
    expected_h = np.array([[0, 0.5 - 0.5j], [0.5 + 0.5j, 0]])
    for shift in (0.0, 1.0):
        result = Run(plan_for(A=np.diag([1 + shift, -1 + shift]))).wait(timeout=5, poll_interval=0)
        np.testing.assert_allclose(matrix(result.pencil.overlap), expected_s, rtol=0, atol=3e-13)
        np.testing.assert_allclose(
            matrix(result.pencil.hamiltonian), expected_h + shift * expected_s, rtol=0, atol=3e-13
        )
        assert result.eigenvalue == pytest.approx(-1 + shift, abs=1e-12, rel=0)


@pytest.mark.parametrize("execution", ["quantum", "classical"])
def test_requested_real_parts_without_shots_keep_the_complex_pencil(execution):
    """Exact readout and classical execution leave a requested real-only pencil without effect.

    `quadratures="real"` is used only by measurement with shots, so neither
    execution refuses it, although no input below has a real proof, and
    both keep it as requested with no resolved quadratures or proof.
    A = Y with Phi = (|0>, |1>) has H_01 = <0|Y|1> = -i and S_01 = 0, and
    A = Z with phi_0 = |0> and phi_1 = (i|0> + |1>)/sqrt(2) has
    S_01 = H_01 = i/sqrt(2). Under exact readout each computed entry lies
    within the entry bound of its pair (`PairEstimate.hamiltonian_bound` and
    `overlap_bound`) of the exact value, with 1/sqrt(2) compared through
    exact squares.
    Classical execution of the first case multiplies and adds only 0, 1 and
    -i, so its entries are exact, and in the second case it keeps nonzero
    imaginary parts.
    """
    from fractions import Fraction as F
    from nwqlib.algorithms import PencilAcquisition
    from nwqlib.operators import ingest_pauli

    def near(value, exact, bound):
        """Whether binary64 ``value`` lies within ``bound`` of exact ``exact``, or of 1/sqrt(2) when it is None."""
        if exact is not None:
            return abs(F(value) - exact) <= F(bound)
        low, high = F(value) - F(bound), F(value) + F(bound)
        return low >= 0 and low * low <= F(1, 2) <= high * high

    request = PencilAcquisition(quadratures="real")
    cases = ((("Y", 1.0), ((1, 0), (0, 1)), (0, -1), (0, 0)), (("Z", 1.0), ((1, 0), (1j, 1)), (0, None), (0, None)))
    for term, basis, h01, s01 in cases:
        plan = plan_for(A=ingest_pauli((term,), num_qubits=1), basis=basis, execution=execution,
                        acquisition=request)
        acquisition = plan.reconstruction.acquisition
        assert (acquisition.requested, acquisition.quadratures, acquisition.real_proof) == (request, None, None)
        result = Run(plan).wait(timeout=5, poll_interval=0)
        h, s = matrix(result.pencil.hamiltonian), matrix(result.pencil.overlap)
        if execution == "quantum":
            (pair,) = (p for p in result.pairs if p.pair == (0, 1))
            for z, (re, im), bound in ((pair.hamiltonian, h01, pair.hamiltonian_bound),
                                       (pair.overlap, s01, pair.overlap_bound)):
                assert near(z.real, re, bound) and near(z.imag, im, bound)
        elif term[0] == "Y":
            assert (h[0, 1], s[0, 1], h[0, 0], h[1, 1], s[0, 0], s[1, 1]) == (-1j, 0, 0, 0, 1, 1)
        else:
            assert s[0, 1].imag != 0 and h[0, 1].imag != 0


def test_missing_pair_is_partial_without_smaller_solve(monkeypatch):
    import nwqlib.algorithms.gcim.fixed_basis as owner

    plan = plan_for()
    run = Run(plan)
    result = run.wait(timeout=5, poll_interval=0)
    missing = next(
        chunk for chunk in result.data.observations.chunks if chunk.experiment == "pair_0_1"
    )
    observations = ObservationView(
        chunks=tuple(c for c in result.data.observations.chunks if c is not missing)
    )
    data = RunData(observations, result.data.trace, result.data.receipts)
    monkeypatch.setattr(
        owner,
        "_solve_projected_pencil",
        lambda *a, **k: pytest.fail("partial pair data reached solve"),
    )
    partial = plan.method.analyze(plan, data, settings={})
    assert partial.eigenvalue is None and missing.experiment in partial.missing
    assert partial.pencil is None


def test_classical_pencil_uses_original_access_without_pauli_conversion(monkeypatch):
    # Patch the name the eigen-input owner calls; it binds it at import.
    import nwqlib.algorithms._eigen_inputs as eigen_inputs

    monkeypatch.setattr(
        eigen_inputs,
        "pauli_coefficients",
        lambda *a, **k: pytest.fail("classical projection decomposed A"),
    )
    result = Run(plan_for(execution="classical")).wait(timeout=5, poll_interval=0)
    assert result.eigenvalue == pytest.approx(-1, abs=1e-12, rel=0)
    assert (
        len(result.data.trace.events) == 1
        and result.data.trace.events[0].execution == "host_kernel"
    )


def test_classical_projection_keeps_rayleigh_cancellation_under_column_scaling():
    from decimal import Decimal, localcontext

    # In v^H h v, terms of magnitude 0.9 cancel to 0.01.
    h = np.array(
        [
            [-0.933524768543805, -0.27227062330429774 - 0.9577388861395033j],
            [-0.27227062330429774 + 0.9577388861395033j, -0.8961856333347091],
        ]
    ) + 0.01 * np.eye(2)
    v = np.array(
        [-0.30541079989086395 - 0.48655571915760076j, 0.8179065595564124 - 0.03188471890491738j]
    )
    w = np.array([-v[1].conjugate(), v[0].conjugate()])  # w^H v = 0, so (v, w) spans C^2.
    with localcontext() as context:
        context.prec = 80
        (xr, xi), (yr, yi) = [(Decimal.from_float(z.real), Decimal.from_float(z.imag)) for z in v]
        a, d = (Decimal.from_float(h[i, i].real) for i in range(2))
        br, bi = Decimal.from_float(h[0, 1].real), Decimal.from_float(h[0, 1].imag)
        x2, y2 = xr**2 + xi**2, yr**2 + yi**2
        cross = xr * (br * yr - bi * yi) + xi * (br * yi + bi * yr)  # Re(conj(v0) h01 v1)
        rayleigh = float((a * x2 + d * y2 + 2 * cross) / (x2 + y2))
        mean, radius = (a + d) / 2, ((a - d) ** 2 / 4 + br**2 + bi**2).sqrt()
        spectrum = [float(mean - radius), float(mean + radius)]
    # Column normalization, length-2 complex products and a well-conditioned
    # (single or orthonormal column) Lowdin solve each add at most a few
    # eps*||h||_F by the standard inner-product and backward-error bounds.
    tolerance = 64 * np.finfo(float).eps * np.linalg.norm(h)
    # FixedGCIM normalizes each column, so power-of-two factors and 3 - 4j
    # (modulus 5 with a complex phase) leave the projected energies unchanged.
    # Without normalization, 2**-30 would put a Gram mode below the 1e-12 cutoff.
    for scale in (1.0, 2.0**-30, 2.0**30, 3 - 4j):
        single = Run(plan_for(A=h, basis=(scale * v,), execution="classical"))
        assert single.wait(timeout=5, poll_interval=0).eigenvalue == pytest.approx(
            rayleigh, rel=0, abs=tolerance
        )
        pair = Run(plan_for(A=h, basis=(v, scale * w), execution="classical"))
        pencil = pair.wait(timeout=5, poll_interval=0).pencil
        assert pencil.kept_rank == 2
        np.testing.assert_allclose(pencil.eigenvalues, spectrum, rtol=0, atol=tolerance)


@pytest.mark.parametrize("kind, q, b", (("csr", 14, 2), ("dense", 10, 2)))
def test_classical_projection_peak_fits_its_byte_law(kind, q, b):
    import tracemalloc
    from scipy import sparse
    from nwqlib.algorithms._eigen_inputs import preparation_requirements
    from nwqlib.algorithms.gcim.fixed_basis import _classical_projection_requirements
    from nwqlib.problems.inputs import ingest_product

    # Product states make the kernel allocate every trial vector itself, so
    # V and (A - cI) V are new complex128 d-by-b arrays and each column's
    # preparation is a temporary of the preparation stage.
    # NumPy reports its buffers to tracemalloc.
    d = 1 << q
    rng = np.random.default_rng(5)
    diagonal = np.linspace(-1, 1, d)
    basis = tuple(
        ingest_product([(np.cos(t), np.sin(t)) for t in rng.uniform(0, np.pi, q)]) for _ in range(b)
    )
    a = sparse.diags(diagonal).tocsr() if kind == "csr" else np.diag(diagonal)
    plan = plan_for(A=a, basis=basis, execution="classical")
    (kernel,) = plan.blocks
    # Its mean diagonal is a roundoff-sized nonzero c, so the kernel takes the
    # blocked route that removes c, the route with the most workspace.
    shift = plan.reconstruction.identity_shift
    assert 0 < abs(shift) < 1e-15
    selected = plan._native["basis"]
    preparation = max(preparation_requirements(s)[0] for s in selected)
    law, _ = _classical_projection_requirements(plan._native["operator"], b, shift, preparation, b)
    tracemalloc.start()
    try:
        start, _ = tracemalloc.get_traced_memory()
        kernel._call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The law leaves out Python objects such as the 2b(b + 1) scalar
    # records, which the 64 KiB slack covers. The check is one-sided and
    # the law's margin over the measured peak exceeds one complex128
    # d-by-b block in every case here, so it catches an undercharge larger
    # than that margin plus the slack, such as a full copy of A (8 MiB at
    # q = 10), and not one missed block. The dense b = 2 case has the
    # smallest margin.
    assert peak - start <= law + 65536


def test_sampled_pencil_enclosure_keeps_raw_estimate_and_solver_scope(tmp_path, monkeypatch):
    from nwqlib import solve, load_result
    from nwqlib.evidence.verification import ProjectedVerificationOptions
    from nwqlib.algorithms.gcim.fixed_basis import processed_pauli_enclosure
    from nwqlib.algorithms.gcim import fixed_basis

    result = solve(Eigenproblem(A=np.diag([1.,-1.])), method=FixedGCIM(basis=([1,1],[1,1j])),
                   execution='quantum', shots=256, seed=0)
    p = result.pencil
    assert result.eigenvalue < -1.  # This fixed observed population violates Z's exact [-1,1] enclosure.
    assert p.failure_reason is None and p.sampled_failure == 'ritz_outside_operator_enclosure'
    assert p.sampled_enclosure_passed is False
    # Outward binary64 rounding changes the exact Z bound by only a few ulps.
    np.testing.assert_allclose(p.sampled_enclosure, [-1.,1.], rtol=0., atol=1e-14)
    assert p.sampled_enclosure_violation == pytest.approx(-1.-result.eigenvalue, rel=0., abs=1e-14)
    assert 'outside numerical window of processed Pauli enclosure' in str(result)
    assert p.sampled_enclosure == processed_pauli_enclosure(result.plan.reconstruction)
    checks = ProjectedVerificationOptions(name='all', comparisons=(
        'overlap_normalization','gram_psd_deficit','gram_hermiticity','projected_backward_error'), tolerance=1e-9)
    _, facts = result.verify(checks=checks)
    assert all(f.fact.availability == 'concrete' for f in facts)
    path = result.save(tmp_path/'sampled')
    monkeypatch.setattr(fixed_basis, '_solve_projected_pencil', lambda *a,**kw: pytest.fail('load solved another pencil'))
    loaded = load_result(path)
    assert loaded.eigenvalue == result.eigenvalue and loaded.pencil == p
    with pytest.raises(ValueError, match='actual Ritz'):
        result.revise(pencil=p.revise(sampled_enclosure_violation=0.)).validate_plan(result.plan)


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_identity_offset_moves_the_ritz_value_by_its_coefficient(execution):
    from nwqlib import solve
    from nwqlib.operators import ingest_pauli

    pauli = {"I": np.eye(2), "X": np.array([[0., 1.], [1., 0.]]),
             "Y": np.array([[0., -1j], [1j, 0.]]), "Z": np.diag([1., -1.])}
    terms = (("ZI", .5), ("XX", .3), ("IY", .2))
    a0 = sum(c * np.kron(pauli[label[0]], pauli[label[1]]) for label, c in terms)
    rng = np.random.default_rng(5)
    phi = [rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2)]
    chi = rng.normal(size=4) + 1j * rng.normal(size=4)
    # The third column is within 1e-5 of the first, so cond(S) is about 1e11.
    basis = (phi[0], phi[1], phi[0] + 1e-5 * chi)
    values = {}
    for c in (.25, -1e4):
        a = (a0 + c * np.eye(4) if execution == "classical"
             else ingest_pauli(terms + (("II", c),), num_qubits=2))
        result = solve(Eigenproblem(A=a), method=FixedGCIM(basis=basis), execution=execution, seed=7)
        assert result.pencil.overlap_condition_number > 1e10
        values[c] = result.eigenvalue
    # Both runs solve bitwise the same offset-free pencil. The quantum H0
    # never measures the identity term. The classical kernel removes the mean
    # diagonal, which is c because A0 is traceless, and the diagonal entries
    # +-0.5 + c minus c are exact for these dyadic values. Only adding c back
    # rounds, once in each run, so the two values of E - c agree within
    # u(|E(.25)| + |E(-1e4)|), here with a factor 2 to spare. Solving
    # (H0 + cS, S) directly would carry an error of order u|c|/s_min, about
    # 1e-2 here.
    assert abs((values[-1e4] + 1e4) - (values[.25] - .25)) <= (
        2 * 2.**-53 * (abs(values[-1e4]) + abs(values[.25])))


def test_rank_deficient_basis_is_admitted_at_its_gram_formation_error():
    from scipy import sparse
    from qiskit import QuantumCircuit
    from nwqlib import solve
    from nwqlib._projected_eigensolver import gram_formation_allowance
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import state_input

    # In a classical Run the basis (a, b, a + b) is legal, and its exact Gram
    # matrix has one zero eigenvalue. At 14 qubits the length-d inner
    # products put the computed one below zero by more than the eigensolver
    # roundoff of S for these two seeds, so an allowance for that roundoff
    # alone refuses them as invalid Gram matrices.
    d = 1 << 14
    rng = np.random.default_rng(3)
    diagonal, off = rng.normal(size=d), rng.normal(size=d - 1)
    a_matrix = sparse.diags([off, diagonal, off], [-1, 0, 1], format="csr")
    for seed in (100, 101):
        draw = np.random.default_rng(seed)
        a, b = (draw.normal(size=d) + 1j * draw.normal(size=d) for _ in range(2))
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        result = solve(Eigenproblem(A=a_matrix), method=FixedGCIM(basis=(a, b, a + b)),
                       execution="classical", seed=7)
        pencil = result.pencil
        assert pencil.failure_reason is None and pencil.kept_rank == 2
        # The kept subspace is span{a, b}, so the value is the lowest Ritz
        # value of A there, from an independent orthonormal basis. To first
        # order the kept pencil moves it by at most (||dH|| + |E| ||dS||)/s_kept,
        # with ||dS|| within the Gram allowance and ||dH|| within that
        # allowance times ||A - cI|| <= 2 ||A||. The largest absolute row sum
        # bounds ||A|| and |E|, so three of them times the allowance over
        # s_kept bound the change. The QR reference adds only u-sized error.
        allowance = gram_formation_allowance(d, sum(pencil.overlap_eigenvalues))
        s_kept = sorted(pencil.overlap_eigenvalues)[1]
        norm = float(abs(a_matrix).sum(axis=1).max())
        q, _ = np.linalg.qr(np.column_stack([a, b]))
        reference = np.linalg.eigvalsh(q.conj().T @ (a_matrix @ q))[0]
        assert result.eigenvalue == pytest.approx(reference, rel=0, abs=3 * norm * allowance / s_kept)

    # The normalized columns (a, b, exp(0.7i) a) span two dimensions.
    # Native pair preparations have qualified state budgets, whose overlap
    # bounds admit the Gram's formation error before the rank cutoff acts.
    def layered(seed, phase=0.0):
        draw = np.random.default_rng(seed)
        circuit = QuantumCircuit(2, global_phase=phase)
        for _ in range(300):
            for qubit in range(2):
                circuit.u(*draw.uniform(-np.pi, np.pi, 3), qubit)
            circuit.cx(0, 1)
        return circuit

    terms = (("ZI", .5), ("XX", .3), ("IY", .2))
    a, b = layered(4), layered(104)
    result = solve(Eigenproblem(A=ingest_pauli(terms, num_qubits=2)),
                   method=FixedGCIM(basis=tuple(state_input(c) for c in (a, b, layered(4, .7)))),
                   execution="quantum", seed=7)
    pencil = result.pencil
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from nwqlib.algorithms.gcim.fixed_basis import _exact_overlap_allowance

    assert all(p.overlap_bound is not None for p in result.pairs)
    allowance = _exact_overlap_allowance(
        3, {p.pair: p.overlap_bound for p in result.pairs}
    )
    assert allowance is not None
    assert pencil.failure_reason is None and pencil.kept_rank == 2

    columns = np.column_stack([Statevector(a).data, Statevector(b).data])
    q, _ = np.linalg.qr(columns)
    hamiltonian = SparsePauliOp.from_list(terms).to_matrix()
    reference = np.linalg.eigvalsh(q.conj().T @ hamiltonian @ q)[0]
    s_kept = sorted(pencil.overlap_eigenvalues)[1]
    norm_bound = sum(abs(coefficient) for _, coefficient in terms)
    # The first-order projected-space perturbation is bounded by
    # (||dH|| + |E| ||dS||) / s_kept. The Pauli L1 norm bounds ||H||,
    # and this fixture's receipt terms dominate its short reference dots.
    # The factor three covers those Hamiltonian and Gram contributions.
    assert result.eigenvalue == pytest.approx(
        reference, rel=0, abs=3 * norm_bound * allowance / s_kept
    )



def test_classical_offset_removal_agrees_across_storage_formats():
    from scipy import linalg, sparse
    from nwqlib import solve
    from nwqlib._projected_eigensolver import gram_formation_allowance

    # A banded Hermitian matrix with a large diagonal offset has nnz = 7d, so
    # the CSR and CSC routes cut it into several blocks of at most d entries
    # and patch each block's diagonal. Every storage must give the Ritz value
    # of the same pencil V^dagger A V, V^dagger V.
    d = 64
    rng = np.random.default_rng(8)
    bands = {k: rng.normal(size=d - k) + 1j * rng.normal(size=d - k) * (k > 0) for k in range(4)}
    dense = sum(np.diag(bands[k], k) + (np.diag(bands[k].conj(), -k) if k else 0) for k in range(4))
    dense = dense + 1e3 * np.eye(d)
    basis = tuple(rng.normal(size=d) + 1j * rng.normal(size=d) for _ in range(3))
    v = np.column_stack([b / np.linalg.norm(b) for b in basis])
    s = v.conj().T @ v
    reference = linalg.eigh(v.conj().T @ dense @ v, s, eigvals_only=True)[0]
    # To first order the Ritz value moves by at most (||dH|| + |E| ||dS||)/s_min,
    # with ||dS|| within the Gram allowance and ||dH|| within that allowance
    # times ||A - cI|| <= 2 ||A||. The largest absolute row sum bounds ||A||
    # and |E|, and the reference eigensolve adds only u-sized error.
    norm = float(np.abs(dense).sum(axis=1).max())
    tolerance = 3 * norm * gram_formation_allowance(d, np.trace(s).real) / np.linalg.eigvalsh(s)[0]
    for a in (dense, sparse.csr_matrix(dense), sparse.csc_matrix(dense)):
        result = solve(Eigenproblem(A=a), method=FixedGCIM(basis=basis), execution="classical", seed=7)
        assert result.plan.reconstruction.identity_shift == pytest.approx(1e3, rel=0, abs=1.)
        assert result.eigenvalue == pytest.approx(reference, rel=0, abs=tolerance)


def test_exact_pair_reductions_match_dense_reference_entries():
    """Exact FixedGCIM reduces one phase-faithful state per pair to the dense ``H0_ij`` and ``S_ij``.

    With b = 3 trial states and L = 4 nonidentity terms (a Y word and a
    Z-only word among them) the route acquires ``(M - D) + D = 6``
    reductions, one per upper-triangle pair, where the per-element Hadamard
    route took ``2(M - D)(L + 1) + DL = 42`` circuits. The trial states are a
    product circuit, an entangled circuit and a product vector, two of them
    with nontrivial global phases. Each recorded entry is compared with
    ``<phi_i|H0|phi_j>`` and ``<phi_i|phi_j>`` from the preparation
    statevectors, global phases included, within the entry bounds
    ``E_H``, ``E_S`` from the state error of its own receipt, which every
    pair of this fixture has.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from nwqlib import solve
    from nwqlib.operators import ingest_pauli
    from nwqlib.algorithms.gcim.pencil import pair_count
    from nwqlib.problems.inputs import ingest_product, state_input

    first = QuantumCircuit(2, global_phase=0.4)
    first.ry(0.7, 0)
    first.rx(-0.3, 1)
    second = QuantumCircuit(2, global_phase=-1.1)
    second.h(0)
    second.cx(0, 1)
    second.rz(0.5, 1)
    third = [(np.cos(0.2), np.sin(0.2)), (0.6, 0.8j)]
    terms = (("II", 0.25), ("ZI", 0.5), ("XY", 0.3), ("IZ", -0.2), ("YY", 0.15))
    result = solve(Eigenproblem(A=ingest_pauli(terms, num_qubits=2)),
                   method=FixedGCIM(basis=(state_input(first), state_input(second), ingest_product(third))),
                   execution="quantum", seed=3)
    assert len(result.data.observations.chunks) == len(result.pairs) == pair_count(3, 4) == 6
    vectors = [Statevector(first).data, Statevector(second).data,
               np.kron(np.asarray(third[1]), np.asarray(third[0]))]
    h0 = SparsePauliOp.from_list(terms[1:]).to_matrix()
    for estimate in result.pairs:
        left, right = estimate.pair
        expected = (np.vdot(vectors[left], h0 @ vectors[right]), np.vdot(vectors[left], vectors[right]))
        acquired = (complex(estimate.hamiltonian.real, estimate.hamiltonian.imag),
                    complex(estimate.overlap.real, estimate.overlap.imag))
        for value, target, bound in zip(acquired, expected, (estimate.hamiltonian_bound, estimate.overlap_bound)):
            assert bound is not None and abs(value - target) <= bound


def test_pair_reductions_are_admitted_against_their_work_and_workspace(monkeypatch):
    """Preparation admits the summed pair-reduction work, and each point's workspace, before native work.

    Planning records the summed ``pair_work`` of the pair reductions without
    refusing it. With ``max_classical_products`` one unit below that sum,
    preparation refuses with the cap and the registered work before any
    native lowering; at exactly the sum it prepares. The Method hook refuses
    a point whose ``pair_bytes`` exceeds ``max_bytes`` and otherwise returns
    the cap less the other points' work.
    """
    import json
    import nwqlib.blocks.lowering as lowering
    from nwqlib.algorithms.gcim.pair_reducer import pair_bytes, pair_work
    from nwqlib.operators import ingest_pauli

    settings = dict(A=ingest_pauli((("ZI", 0.5), ("XY", 0.3), ("IZ", -0.2)), num_qubits=2),
                    basis=([1, 0, 0, 0], [0.6, 0.8, 0, 0]))
    plan = plan_for(**settings)
    points = [p for e in plan.experiments for p in e.readout.positions]
    total = sum(pair_work(json.loads(p.parameters)) for p in points)
    assert len(points) == 3
    lowered = [0]
    original = lowering._lower_qiskit

    def counted(*args, **kwargs):
        lowered[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(lowering, "_lower_qiskit", counted)
    short = Run(plan_for(max_classical_products=total - 1, **settings))
    with pytest.raises(ValueError, match=rf"registered work {total}, max_classical_products={total - 1}\. "
                                         rf"Raise FixedGCIM\.max_classical_products to at least {total}\."):
        short.plan.method.prepare(short.plan, run=short)
    assert lowered[0] == 0 and short._state["preparations"] == 0
    exact = Run(plan_for(max_classical_products=total, **settings))
    exact.plan.method.prepare(exact.plan, run=exact)
    assert lowered[0] > 0
    point = points[1]
    observation = plan.experiments[1].readout
    own = pair_work(json.loads(point.parameters))
    allowance = plan.method.reduction_allowance(plan, None, observation=observation, width=3, run=None)
    assert type(allowance) is int and allowance == plan.method.max_classical_products - (total - own)
    needed = pair_bytes(json.loads(point.parameters))
    small = plan.method.revise(max_bytes=needed - 1)
    with pytest.raises(ValueError, match="GCIM pair reduction workspace"):
        small.reduction_allowance(plan, None, observation=observation, width=3, run=None)


def _smallest_exact_adapt():
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.operators import ingest_pauli

    pool = tuple(ingest_pauli(((p, 1j),), num_qubits=1) for p in ("Y", "X"))
    return (Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1)),
            ADAPT(initial_state=[1, 1], pool=pool, theta=np.pi / 8, max_iterations=1))


def _smallest_exact_qfd():
    from nwqlib.algorithms import QFD

    return Eigenproblem(A=np.diag([1., -1.])), QFD(initial_state=[1, 1], time_step=.5, powers=(0, 1))


def _smallest_exact_fermionic_qfd():
    from nwqlib.algorithms import QFD
    from nwqlib.operators import ingest_fermion, ingest_pauli

    generator = ingest_fermion([(5.0, []), (1.0, [(0, 1), (0, 0)])], num_modes=1)
    return (Eigenproblem(A=ingest_pauli((("I", 3.0), ("X", 1.0)), num_qubits=1)),
            QFD(initial_state=[1, 1], time_step=2.5, powers=(0, 1), evolution="fermionic_blocks",
                evolution_terms=generator))


@pytest.mark.parametrize("build", (
    lambda: (Eigenproblem(A=np.diag([1., -1.])), FixedGCIM(basis=([1, 1], [1, 1j]))),
    _smallest_exact_qfd,
    _smallest_exact_fermionic_qfd,
), ids=("FixedGCIM", "QFD", "QFD-fermionic"))
def test_saved_exact_gcim_result_loads_in_a_fresh_process(tmp_path, build):
    """load_result in a new interpreter resolves the saved pair reduction before any GCiM module loads.

    For QFD it also builds the evolution blocks of V again from the saved
    reconstruction, and with a fermionic generator it restores that input
    from its saved arrays, with the same content hash and Method content
    hash.
    """
    import json
    import os
    import subprocess
    import sys
    import nwqlib

    problem, method = build()
    result = nwqlib.solve(problem, method=method, execution="quantum", seed=7)
    assert result.pencil is not None and result.eigenvalue is not None
    result.save(tmp_path / "result")
    fields = ("repr(loaded.eigenvalue), repr(loaded.pencil.hamiltonian), repr(loaded.pencil.overlap), "
              "repr([(p.pair, p.hamiltonian, p.overlap) for p in getattr(loaded, 'pairs', ())]), "
              "loaded.plan.method.content_id, "
              "getattr(getattr(loaded.plan.method, 'evolution_terms', None), 'reference', None) and "
              "loaded.plan.method.evolution_terms.reference.identity")
    source = (f"import json, nwqlib\nloaded = nwqlib.load_result({str(tmp_path / 'result')!r})\n"
              f"print(json.dumps([nwqlib.__file__, {fields}]))")
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=dict(os.environ),
                               timeout=300)
    assert completed.returncode == 0, completed.stderr
    origin, *loaded = json.loads(completed.stdout.splitlines()[-1])
    assert origin == nwqlib.__file__
    generator = getattr(method, "evolution_terms", None)
    assert loaded == [repr(result.eigenvalue), repr(result.pencil.hamiltonian), repr(result.pencil.overlap),
                      repr([(p.pair, p.hamiltonian, p.overlap) for p in getattr(result, "pairs", ())]),
                      method.content_id, None if generator is None else generator.reference.identity]


def test_exact_gcim_pair_reductions_run_on_nwqsim_with_aer_entries(tmp_path):
    """Exact FixedGCIM and ADAPT prepare and reduce their pairs on NWQ-Sim's CPU/SV target.

    That target runs trajectories with registered reducers, so each exact
    pair is one native evolution whose saved state the host reducer
    contracts. Its entries match Aer's for the same Plan within the
    predeclared small-fixture regression threshold 2e-12 used by the other
    NWQ-Sim trajectory checks, a regression threshold at two qubits, not a
    receipt bound.
    """
    import os
    from pathlib import Path
    import nwqlib
    from nwqlib import solve
    from nwqlib.backends import NWQSimBackend
    from nwqlib.operators import ingest_pauli

    executable = os.environ.get("NWQLIB_TEST_NWQSIM_EXECUTABLE")
    if executable is None:
        pytest.skip("set NWQLIB_TEST_NWQSIM_EXECUTABLE to a qualified native build")
    backend = NWQSimBackend(executable=str(Path(executable).resolve(strict=True)), spool=str(tmp_path / "spool"),
                            max_input_bytes=1 << 20, max_output_bytes=1 << 20, max_buffer_bytes=1 << 20)

    def entries(result):
        return np.array([[complex(z.real, z.imag) for z in row]
                         for matrix in (result.pencil.hamiltonian, result.pencil.overlap) for row in matrix])

    problem = Eigenproblem(A=ingest_pauli((("ZI", .5), ("XY", .3), ("IX", .2)), num_qubits=2))
    for problem, method in ((problem, FixedGCIM(basis=([1, 0, 0, 0], [0.6, 0.8j, 0, 0]))), _smallest_exact_adapt()):
        plan = nwqlib.plan(problem, method=method, execution="quantum", seed=7)
        aer = solve(plan, progress=False)
        # The detached native Run keeps its folder under tmp_path, not the
        # default run directory.
        with pytest.warns(UserWarning, match="NWQ-Sim CPU/SV runs on execution host"):
            prepared = nwqlib.prepare(plan, backend=backend, directory=tmp_path / type(method).__name__,
                                      progress=False)
            with nwqlib.submit(prepared) as run:
                native = run.wait(timeout=60, poll_interval=0.1)
        assert native.pencil.failure_reason is None
        np.testing.assert_allclose(entries(native), entries(aer), rtol=0, atol=2e-12)


def test_grouped_sampled_pencil_matches_the_dense_reference_within_its_standard_errors(tmp_path):
    """Grouped sampling: b**2 G settings, entries within six estimated standard errors of V†HV and V†V.

    Three normalized complex columns on three qubits and the six nonidentity
    terms ZII, IZI, ZZI, XII, IYI, XYI with identity coefficient 0.7 form
    two QWC groups, XYI and ZZI, so the pencil takes 3**2 * 2 = 18 settings
    instead of the 60 elementary Hadamard settings of per-term acquisition.
    The words IYI and XYI have one Y each, so the default quadratures resolve
    to both parts, with the first odd-Y word as the recorded reason.
    The independent reference is the dense Pauli matrix and the normalized
    columns. Each off-diagonal overlap quadrature and each physical-H
    quadrature must lie within six estimated standard errors, with a 1e-12
    arithmetic margin: a fixed-seed regression criterion, not a coverage
    theorem. The physical-H variance must include the overlap-Hamiltonian
    covariance of the overlap-supplying group,
    V_H = sum_g V_(Y_g) + c**2 V_X + 2 c C_(X,Y_0) (fixed_basis.SampledPairVariance),
    recomputed here from the recorded group moments. The saved Result
    reloads with its group moments and pair variances.
    """
    import nwqlib
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.operators import ingest_pauli

    rng = np.random.default_rng(31)
    basis = tuple(rng.normal(size=8) + 1j * rng.normal(size=8) for _ in range(3))
    rows = (("III", .7), ("ZII", .5), ("IZI", -.3), ("ZZI", .2), ("XII", .4), ("IYI", -.25), ("XYI", .1))
    problem = Eigenproblem(A=ingest_pauli(rows, num_qubits=3))
    plan = nwqlib.plan(problem, method=FixedGCIM(basis=basis), execution="quantum", shots=8192, seed=104)
    groups = plan.reconstruction.groups
    assert [group.basis for group in groups] == ["XYI", "ZZI"]
    acquisition = plan.reconstruction.acquisition
    assert (acquisition.quadratures, acquisition.real_proof, acquisition.reason) == (
        "complex", "unknown", "Hamiltonian word IYI has an odd number of Y")
    assert len(plan.experiments) == 18
    result = nwqlib.solve(plan, progress=False)
    v = np.column_stack([a / np.linalg.norm(a) for a in basis])
    reference = {"overlap": v.conj().T @ v,
                 "hamiltonian": v.conj().T @ SparsePauliOp.from_list(rows).to_matrix() @ v}
    variances = {row.pair: row for row in result.sampled_pair_variances}
    moments = {(m.pair, m.quadrature, m.group): m for m in result.group_moments}
    covariance_seen = False
    for i in range(3):
        for j in range(i, 3):
            for k, quad in enumerate(("real", "imag")):
                if i == j and quad == "imag":
                    continue
                count = len(groups) if i == j else max(1, len(groups))
                expected = sum(moments[(i, j), quad, g].mean_variance for g in range(count))
                if i != j:
                    zero = moments[(i, j), quad, 0]
                    expected += .7 ** 2 * zero.overlap_variance + 2 * .7 * zero.mean_covariance
                    covariance_seen |= zero.mean_covariance != 0
                assert variances[i, j].physical_h[k] == pytest.approx(expected, rel=1e-12, abs=0)
                for kind, variance in (("overlap", variances[i, j].overlap[k]),
                                       ("hamiltonian", variances[i, j].physical_h[k])):
                    if kind == "overlap" and i == j:
                        continue
                    actual = getattr(matrix(getattr(result.pencil, kind))[i, j], quad)
                    target = getattr(reference[kind][i, j], quad)
                    assert abs(actual - target) <= 6 * np.sqrt(variance) + 1e-12
    assert covariance_seen
    loaded = nwqlib.load_result(result.save(tmp_path / "grouped"))
    assert loaded.group_moments == result.group_moments
    assert loaded.sampled_pair_variances == result.sampled_pair_variances
    assert loaded.plan.reconstruction.groups == groups
    assert loaded.plan.reconstruction.acquisition == acquisition


def _hoeffding_radius(ranges, assertions, shots):
    """Reference radius of one asserted sum of setting means under the ideal sampling model.

    Setting k of the sum has per-shot values in [-R_k, R_k]. If the
    ``shots`` shots of each setting were independent draws from the ideal
    outcome probabilities of the intended states, giving each of the K
    settings of one assertion 2**-20/(MK), with M = ``assertions``, would
    make Hoeffding's inequality for a mean of N values in an interval of
    width 2R_k give t_k = R_k sqrt(2 log(2MK 2**20)/N), and all M sums would
    lie within their sum_k t_k except with probability 2**-20. The Aer
    counts are not qualified draws from that law, because no count receipt
    bounds the distance of the simulator's outcome law from it. The tests
    therefore use the radius as a fixed empirical threshold for one seeded
    Aer run, without a coverage claim for the actual sampler, and a failure
    is investigated rather than rerun with another seed.
    """
    import math

    level = math.log(2 * assertions * len(ranges) * 2**20)
    return sum(r * math.sqrt(2 * level / shots) for r in ranges)


def _grouped_tail(handle):
    """Return the exact ancilla and ancilla-system Z parities of a prepared grouped setting, with their error bound.

    The handle's native circuit, with its final measurements removed, is
    evaluated as a binary64 statevector. Classical bit 0 holds the ancilla
    and bit 1 the first system qubit, and each parity is the Z expectation
    on the qubits measured into those bits. Every gate must be a fixed
    Clifford or T gate, a rotation by an exact angle of the fixtures (0,
    +-1/2, +-1), a phase k fl(pi)/4 with integer k, or a one-qubit unitary
    whose entry parts are 0, +-1 or within 2**-53 of +-2**-0.5. Each gate's
    entries then lie within 2 sqrt(2) u of its exact matrix, u = 2**-53, so
    its norm error is at most 4Du for D = 2**qubits, and k fl(pi)/4 differs
    from k pi/4 by at most |k| 2**-54, which adds that much to a phase
    gate. One dense product per gate errs by at most 2D gamma_2D, gamma_n
    = nu/(1 - nu), so after the gates 1 + delta = prod (1 + e)(1 + 2D gamma_2D)
    bounds ||psi_hat - psi|| (one-ulp cos and sin assumed). A parity of the
    computed state then differs from the exact one by at most
    2 delta + delta**2 plus the dot error sqrt(2) gamma_2D (1 + delta)**2,
    bounded here with 3/2 for sqrt(2).
    """
    import math
    from fractions import Fraction as F
    from qiskit.quantum_info import SparsePauliOp, Statevector

    circuit = handle._restore_native().circuit
    measured = {circuit.find_bit(item.clbits[0]).index: circuit.find_bit(item.qubits[0]).index
                for item in circuit.data if item.operation.name == "measure"}
    body = circuit.remove_final_measurements(inplace=False)
    dimension = 1 << body.num_qubits
    u = F(1, 2**53)
    product = 2 * dimension * (2 * dimension * u / (1 - 2 * dimension * u))
    growth = F(1)
    for item in body.data:
        angle = F(0)
        if item.operation.name == "unitary":
            parts = [abs(x) for z in item.operation.params[0].flat for x in (z.real, z.imag)]
            assert all(x in (0.0, 1.0) or abs(x - 2**-.5) <= 2**-53 for x in parts), item.operation
        else:
            assert item.operation.name in {"h", "x", "s", "sdg", "t", "tdg", "cx", "mcphase", "cu", "crz"}, item
            for value in map(float, item.operation.params):
                k = round(value / (math.pi / 4))
                if value not in (0.0, .5, -.5, 1.0, -1.0):
                    assert value == k * (math.pi / 4), item.operation
                    angle += abs(k) * F(1, 2**54)
        growth *= (1 + 4 * dimension * u + angle) * (1 + product)
    delta = growth - 1
    state = Statevector(body)

    def parity(bits):
        label = ["I"] * body.num_qubits
        for bit in bits:
            label[body.num_qubits - 1 - measured[bit]] = "Z"
        return state.expectation_value(SparsePauliOp("".join(label))).real

    gamma = 2 * dimension * u / (1 - 2 * dimension * u)
    return (parity((0,)), parity((0, 1))), 2 * delta + delta**2 + F(3, 2) * gamma * (1 + delta)**2


def _entry_ranges(rec, i, j):
    """Per-shot value ranges of the settings whose means form the physical H_ij of a grouped pencil.

    Group g contributes ``Y_g = sum_(k in g) c_k Z_k`` with ``|Z_k| = 1``.
    Off the diagonal the identity coefficient times the ancilla sign is
    added in the overlap-supplying group 0, the same setting.
    """
    coefficients = [abs(c) for _, c in rec.nonidentity_terms]
    ranges = [sum(coefficients[k] for k in group.indices) for group in rec.groups]
    if i != j:
        ranges[0] += abs(rec.identity_shift)
    return ranges


def test_structurally_real_basis_measures_only_real_parts():
    """Real inputs measure b(b+1)G/2 settings, whose real entries are compared with exact rationals.

    A = 0.5 I + Z + 0.5 X has real coefficients on words without Y, and the
    basis |0> (occupation), (3/4, 1) (product) and (1, -3/4) (vector) has
    real stored amplitudes, so H and S are real. The default resolves real
    parts, and the run measures 3 * 4 * 2 / 2 = 12 settings of the G = 2
    groups X and Z, one count chunk each, instead of 3**2 * 2 = 18, with
    every imaginary entry and its variances exact zeros. The normalized
    states are |0>, (3/5, 4/5) and (4/5, -3/5), so the reference H and S
    are exact rationals. In one empirical Aer run (seed 7, 4096 shots per
    setting) each of the nine asserted real entries (six of H, three
    off-diagonal of S) lies within the ideal-model reference radius
    ``_hoeffding_radius`` of its settings' value ranges (``_entry_ranges``).
    Replacing the X term by Y keeps both parts and 18 settings, an explicit
    real request is then refused, and an explicit complex request on the
    real inputs measures 18 settings.
    """
    from fractions import Fraction as F
    import nwqlib
    from nwqlib.algorithms import ApplicabilityError, PencilAcquisition
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation, ingest_product, ingest_vector

    basis = (ingest_occupation((0,), num_qubits=1), ingest_product([[.75, 1.0]]), ingest_vector([1.0, -.75]))
    problem = Eigenproblem(A=ingest_pauli((("I", .5), ("Z", 1.0), ("X", .5)), num_qubits=1))

    def select(problem, **request):
        return nwqlib.plan(problem, method=FixedGCIM(basis=basis, acquisition=PencilAcquisition(**request)),
                           execution="quantum", shots=4096, seed=7)

    plan = select(problem)
    rec = plan.reconstruction
    assert [group.basis for group in rec.groups] == ["X", "Z"]
    assert (rec.acquisition.quadratures, rec.acquisition.real_proof) == ("real", "proved")
    names = sorted(e.name for e in plan.experiments)
    assert len(names) == 3 * 4 * 2 // 2 and not any("imag" in name for name in names)
    result = nwqlib.solve(plan, progress=False)
    chunks = result.data.observations.chunks
    assert sorted(c.experiment for c in chunks) == names
    assert all(c.returned_shots == 4096 for c in chunks)
    assert sum(event.shots for event in result.data.trace.events) == 12 * 4096
    h, s = matrix(result.pencil.hamiltonian), matrix(result.pencil.overlap)
    assert not h.imag.any() and not s.imag.any()
    assert all(row.h0[1] == row.overlap[1] == row.covariance[1] == row.physical_h[1] == 0.0
               for row in result.sampled_pair_variances)

    states = ((F(1), F(0)), (F(3, 5), F(4, 5)), (F(4, 5), F(-3, 5)))
    a = ((F(1, 2) + 1, F(1, 2)), (F(1, 2), F(1, 2) - 1))  # 0.5 I + Z + 0.5 X
    for i in range(3):
        for j in range(i, 3):
            exact_s = sum(x * y for x, y in zip(states[i], states[j], strict=True))
            exact_h = sum(states[i][r] * a[r][c] * states[j][c] for r in range(2) for c in range(2))
            assert abs(F(h[i, j].real) - exact_h) <= F(_hoeffding_radius(_entry_ranges(rec, i, j), 9, 4096))
            if i != j:
                assert abs(F(s[i, j].real) - exact_s) <= F(_hoeffding_radius([1.0], 9, 4096))

    with_y = Eigenproblem(A=ingest_pauli((("I", .5), ("Z", 1.0), ("Y", .5)), num_qubits=1))
    both = select(with_y)
    assert (both.reconstruction.acquisition.quadratures, both.reconstruction.acquisition.reason) == (
        "complex", "Hamiltonian word Y has an odd number of Y")
    assert len(both.experiments) == 18
    with pytest.raises(ApplicabilityError, match="odd number of Y"):
        select(with_y, quadratures="real")
    explicit = select(problem, quadratures="complex")
    assert (explicit.reconstruction.acquisition.quadratures, explicit.reconstruction.acquisition.real_proof) == (
        "complex", None)
    assert len(explicit.experiments) == 18


@pytest.mark.parametrize("kind", ["complex128 real vector", "vector lost imaginary part",
                                  "product lost imaginary part", "supplied circuit", "metadata-only vector",
                                  "metadata-only occupation"])
def test_real_proof_reads_the_stored_amplitudes(kind):
    """The real proof scans each state's stored amplitudes, never its rounded direction.

    The stored amplitudes (2**1022 + 2**-1022 i, 1) normalize to a direction
    whose imaginary parts are all zero, because 2**-2044 underflows, yet the
    state is complex, so this vector and the product state with this factor
    have no proof. A real vector stored as complex128 has one. A supplied
    circuit holds no amplitudes, and a state rebuilt from its metadata
    record holds no data in any representation, an occupation state
    included. With |0> and the real A = 0.5 I + Z + 0.5 X, the default
    resolves real parts exactly for a proved state, and an explicit real
    request is refused otherwise. A metadata-only state is refused at its
    preparation with the error of the state input. Planning only.
    """
    import nwqlib
    from qiskit import QuantumCircuit
    from nwqlib.algorithms import ApplicabilityError, PencilAcquisition
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import StateInput, ingest_occupation, ingest_product, ingest_vector

    lost = np.array([2.0**1022 + 2.0**-1022 * 1j, 1.0])
    circuit = QuantumCircuit(1)
    circuit.h(0)
    state = {
        "complex128 real vector": lambda: ingest_vector(np.array([1.0 + 0j, -.75 + 0j])),
        "vector lost imaginary part": lambda: ingest_vector(lost),
        "product lost imaginary part": lambda: ingest_product(lost[None, :]),
        "supplied circuit": lambda: circuit,
        "metadata-only vector": lambda: StateInput.from_record(ingest_vector(np.array([1.0, 0.0])).to_record()),
        "metadata-only occupation": lambda: StateInput.from_record(ingest_occupation((1,), num_qubits=1).to_record()),
    }[kind]()
    if "lost" in kind:
        assert not state._direction.imag.any() and state._physical.imag.any()
    proved = kind == "complex128 real vector"
    problem = Eigenproblem(A=ingest_pauli((("I", .5), ("Z", 1.0), ("X", .5)), num_qubits=1))

    def select(**request):
        method = FixedGCIM(basis=(ingest_occupation((0,), num_qubits=1), state),
                           acquisition=PencilAcquisition(**request))
        return nwqlib.plan(problem, method=method, execution="quantum", shots=16, seed=7)

    if proved:
        assert select(quadratures="real").reconstruction.acquisition.quadratures == "real"
    else:
        with pytest.raises(ApplicabilityError, match="needs a structural real proof"):
            select(quadratures="real")
    if kind.startswith("metadata-only"):
        with pytest.raises(ValueError, match="native data is unavailable"):
            select()
    else:
        acquisition = select().reconstruction.acquisition
        assert (acquisition.quadratures, acquisition.real_proof) == (
            ("real", "proved") if proved else ("complex", "unknown"))


def test_joint_readout_imaginary_part_has_the_positive_sign():
    """<Y_a P> = +Im<phi_0|P|phi_1> in the actual grouped circuits, exactly and in one Aer run.

    With phi_0 = |-i> (H, then S^dagger), phi_1 = exp(-iY/2) phi_0 =
    exp(i/2) phi_0 (then RY(1)) and A = Y, S_01 = exp(i/2) and
    H_01 = -exp(i/2). The real and imaginary settings that FixedGCIM
    prepares, with their measurements removed, then have the ancilla Z
    parity cos(1/2) and sin(1/2), and after the system Y rotation the
    ancilla-system parity -cos(1/2) and -sin(1/2). Each value of the exact
    statevector lies within the primitive-gate bound of ``_grouped_tail``
    plus one ulp of the binary64 cos or sin, which separates the ancilla Y
    sign, the system Y rotation and the bit order.

    With phi_0 = |0>, phi_1 = i|0> and A = Z, z = <0|P|i 0> = i for P = I
    and P = Z. The joint state (|0>|0> + |1> i|0>)/sqrt(2) is |+i>|0>, whose
    ideal outcome law gives the ancilla sign +1 and the system parity +1 on
    every shot of the imaginary setting, while the real setting's ideal
    mean is 0. In one empirical Aer run (seed 7, 4096 shots) the sampled S
    and H have imaginary parts exactly +1 above and -1 below the diagonal,
    the imaginary-setting variances are exactly 0 and H_00 = H_11 = 1
    exactly, and the real parts of S_01 and H_01 lie within the ideal-model
    reference radius ``_hoeffding_radius`` for two assertions. A reversed
    sign convention gives -1.
    """
    import math
    from fractions import Fraction as F
    from qiskit import QuantumCircuit
    import nwqlib
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_vector

    minus_i = QuantumCircuit(1)
    minus_i.h(0)
    minus_i.sdg(0)
    rotated = minus_i.copy()
    rotated.ry(1.0, 0)
    tail = nwqlib.plan(Eigenproblem(A=ingest_pauli((("Y", 1.0),), num_qubits=1)),
                       method=FixedGCIM(basis=(minus_i, rotated)), execution="quantum", shots=16, seed=7)
    with Run(tail) as run:
        from nwqlib._prepared_execution import prepare_experiment

        for name, value in (("group_0_1_real_0", math.cos(.5)), ("group_0_1_imag_0", math.sin(.5))):
            (ancilla, joint), bound = _grouped_tail(prepare_experiment(tail.resolve(name), run=run))
            allowance = bound + F(math.ulp(value))
            assert abs(F(ancilla) - F(value)) <= allowance and abs(F(joint) + F(value)) <= allowance

    problem = Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1))
    plan = nwqlib.plan(problem, method=FixedGCIM(basis=(ingest_vector([1.0, 0.0]), ingest_vector([1j, 0.0]))),
                       execution="quantum", shots=4096, seed=7)
    assert plan.reconstruction.acquisition.quadratures == "complex"
    result = nwqlib.solve(plan, progress=False)
    moment = next(m for m in result.group_moments if m.pair == (0, 1) and m.quadrature == "imag")
    assert (moment.overlap_mean, moment.mean, moment.overlap_variance, moment.mean_variance) == (1.0, 1.0, 0.0, 0.0)
    h, s = matrix(result.pencil.hamiltonian), matrix(result.pencil.overlap)
    assert (s[0, 1].imag, s[1, 0].imag, h[0, 1].imag, h[1, 0].imag) == (1.0, -1.0, 1.0, -1.0)
    assert (h[0, 0], h[1, 1]) == (1.0, 1.0)
    radius = _hoeffding_radius([1.0], 2, 4096)
    assert abs(s[0, 1].real) <= radius and abs(h[0, 1].real) <= radius


def _count_chunk(width, counts, returned=None):
    """A duck-typed count chunk over packed keys ``{index: count}`` for the group reducer."""
    from types import SimpleNamespace
    from nwqlib.execution import Histogram

    words = (width + 63) // 64
    keys = sorted(counts)
    packed = np.array([[(key >> (64 * w)) & (2**64 - 1) for w in range(words)] for key in keys], dtype=np.uint64)
    weights = np.array([counts[key] for key in keys], dtype=np.int64)
    histogram = Histogram._from_arrays(width, packed, weights)
    total = sum(counts.values()) if returned is None else returned
    return SimpleNamespace(histogram=lambda: histogram, returned_shots=total, values=keys,
                           content_id="sha256:" + "0" * 64)


def test_group_moments_equal_exact_rational_moments_of_the_same_count_table():
    """reduce_group against exact rational moments of one count table, including its exact marginal.

    With a = (-1)**phase at classical bit 0 and system qubit k at bit k + 1,
    an off-diagonal shot contributes Z_k = a p_k and Y = sum_k c_k Z_k. The
    table has 2**12 shots over all 512 keys of 9 bits (explicit zero bins
    included) and dyadic coefficients, so the reducer's raw first, second
    and cross moments are exact binary64 values equal to the rational
    moments. Its centered products and their sums are exact as well, so the
    variances and the covariance equal the correctly rounded rational values
    after the division by 4095. The rational
    moments of the full table equal those of its exact marginal on the bits
    the statistics read. A diagonal setting must keep bit 0 zero, the count
    total must equal returned_shots, a one-shot table has unavailable
    variances, a constant outcome with a non-dyadic coefficient keeps a
    nonnegative variance, and a label on system qubit 63 reads the second
    key word.
    """
    from fractions import Fraction as F
    from nwqlib.algorithms.gcim.fixed_basis import reduce_group

    n = 8
    member_rows = (("IIIZIIZI", .5), ("ZIIZIIII", -.75))  # system qubits 1, 4 and 4, 7

    def reduce(chunk, setting, rows=member_rows, num_qubits=n):
        return reduce_group(chunk, setting, rows=rows, members=range(len(rows)), num_qubits=num_qubits)

    rng = np.random.default_rng(3)
    counts = dict(enumerate(rng.multinomial(4096, np.full(512, 1 / 512)).tolist()))
    moment = reduce(_count_chunk(n + 1, counts), ("group_0_1_real_0", 0, 1, "real", 0))

    def exact(table, bit_of):
        total = sum(table.values())
        sums = [F(0)] * 6
        for key, count in table.items():
            a = 1 - 2 * ((key >> bit_of(0)) & 1)
            z = []
            for label, _ in member_rows:
                parity = sum((key >> bit_of(q + 1)) & 1 for q, letter in enumerate(reversed(label)) if letter != "I")
                z.append(a * (1 - 2 * (parity & 1)))
            y = sum(F(c) * zk for (_, c), zk in zip(member_rows, z, strict=True))
            for index, value in enumerate((a, z[0], z[1], y, y * y, a * y)):
                sums[index] += count * value
        return tuple(s / total for s in sums)

    full = exact(counts, lambda bit: bit)
    read = (0, 2, 5, 8)
    marginal = {}
    for key, count in counts.items():
        small = sum(((key >> bit) & 1) << position for position, bit in enumerate(read))
        marginal[small] = marginal.get(small, 0) + count
    assert full == exact(marginal, lambda bit: read.index(bit))
    x, z0, z1, y, y2, xy = full
    assert (moment.overlap_mean, moment.label_means, moment.mean, moment.second, moment.overlap_cross) == (
        float(x), (float(z0), float(z1)), float(y), float(y2), float(xy))
    assert moment.mean_variance == float((y2 - y * y) / 4095)
    assert moment.overlap_variance == float((1 - x * x) / 4095)
    assert moment.mean_covariance == float((xy - x * y) / 4095)

    diagonal = ("group_0_0_real_0", 0, 0, "real", 0)
    with pytest.raises(ValueError, match="unused phase bit"):
        reduce(_count_chunk(n + 1, {0b10: 3, 0b11: 1}), diagonal)
    with pytest.raises(ValueError, match="returned_shots"):
        reduce(_count_chunk(n + 1, {0: 3}, returned=4), diagonal)
    single = reduce(_count_chunk(n + 1, {0b101: 1}), ("group_0_1_imag_0", 0, 1, "imag", 0))
    assert single.mean_variance is single.overlap_variance is single.mean_covariance is None
    # Keys 0b010 and 0b100 both give ZZ parity 1, so Y = -0.1 on all five
    # shots and the exact variance is zero.
    constant = reduce(_count_chunk(3, {2: 1, 4: 4}), diagonal, rows=(("ZZ", .1),), num_qubits=2)
    assert constant.mean_variance >= 0

    # System qubit 63 is classical bit 64, the first bit of the second word.
    table = {1 << 64: 3, 0: 1}
    assert reduce(_count_chunk(65, table), diagonal, rows=(("Z" + "I" * 63, 1.0),), num_qubits=64).mean == -0.5


def test_repeated_group_acquisitions_pool_with_squared_population_weights():
    """pooled_statistics: weights N_a/sum N, variances and covariances with squared weights, raw negatives flagged.

    Two acquisitions of 100 and 300 shots of one off-diagonal setting pool
    with weights 1/4 and 3/4, variance 1/16 V_1 + 9/16 V_2 and covariance
    1/16 C_1 + 9/16 C_2 (fixed_basis.pooled_statistics). A negative raw
    variance is published unclipped and flagged. The same moments pool
    identically for an explicit subset of pairs. Real-only settings record
    each unmeasured imaginary part as an exact zero with zero variance,
    while a planned setting without a moment is missing. All values are
    dyadic, so every expected value is exact.
    """
    from nwqlib.algorithms.gcim.fixed_basis import GroupMoment, pencil_pairs, pooled_statistics, sampled_settings

    def moment(name, pair, quad, shots, mean, variance, overlap=None, covariance=None, key="0"):
        supplies = pair[0] != pair[1]
        return GroupMoment(
            experiment=name, contribution_id="sha256:" + key * 64, pair=pair, quadrature=quad, group=0,
            shots=shots, label_means=(mean,), mean=mean, second=1.0, mean_variance=variance,
            overlap_mean=overlap if supplies else None, overlap_cross=0.0 if supplies else None,
            overlap_variance=.5 ** 4 if supplies else None, mean_covariance=covariance if supplies else None)

    moments = (
        moment("group_0_0_real_0", (0, 0), "real", 10, .5, -.25, key="1"),
        moment("group_0_1_real_0", (0, 1), "real", 100, .25, .5, .5, .125, key="2"),
        moment("group_0_1_real_0", (0, 1), "real", 300, .75, .25, .25, -.5, key="3"),
        moment("group_0_1_imag_0", (0, 1), "imag", 20, .5, .125, .25, .0625, key="4"),
        moment("group_1_1_real_0", (1, 1), "real", 10, -.5, .125, key="5"),
    )
    # One qubit, A = 0.5 I + Z: two states, one group, identity coefficient 0.5.
    settings = tuple(sampled_settings(pencil_pairs(2), 1))
    estimates, missing, pairs, variances = pooled_statistics(settings, moments, .5)
    assert missing == ()
    real = next(e for e in estimates if e.experiment == "group_0_1_real_0")
    assert real.weights == (.25, .75) and real.value == .25 * .25 + .75 * .75
    assert pairs[0, 1] == (complex(.25 * .25 + .75 * .75, .5), complex(.25 * .5 + .75 * .25, .25))
    row = next(v for v in variances if v.pair == (0, 1))
    assert row.h0 == (.5 / 16 + 9 * .25 / 16, .125)
    assert row.overlap == (10 * .5 ** 4 / 16, .5 ** 4)
    assert row.covariance == (.125 / 16 - 9 * .5 / 16, .0625)
    diagonal = next(v for v in variances if v.pair == (0, 0))
    assert diagonal.h0 == (-.25, 0.0) and diagonal.variance_negative
    assert estimates[0].chunk_mean_variances == (-.25,)

    subset = tuple(s for s in settings if s[1:3] == (0, 1))
    _, missing, sub_pairs, sub_variances = pooled_statistics(subset, moments[1:4], .5)
    assert missing == () and sub_pairs == {(0, 1): pairs[0, 1]} and sub_variances == (row,)

    real_only = tuple(sampled_settings(pencil_pairs(2), 1, "real"))
    assert [s[0] for s in real_only] == ["group_0_0_real_0", "group_0_1_real_0", "group_1_1_real_0"]
    measured = tuple(m for m in moments if m.quadrature == "real")
    estimates, missing, real_pairs, real_variances = pooled_statistics(real_only, measured, .5)
    assert missing == () and len(estimates) == 3
    assert real_pairs[0, 1] == (complex(pairs[0, 1][0].real, 0.0), complex(pairs[0, 1][1].real, 0.0))
    real_row = next(v for v in real_variances if v.pair == (0, 1))
    assert (real_row.h0, real_row.overlap, real_row.covariance) == (
        (row.h0[0], 0.0), (row.overlap[0], 0.0), (row.covariance[0], 0.0))
    assert real_row.physical_h == (row.physical_h[0], 0.0)
    _, missing, partial, partial_variances = pooled_statistics(real_only, measured[:-1], .5)
    assert missing == ("group_1_1_real_0",) and partial == {} and partial_variances == ()


def test_influence_variance_cancels_the_shared_shot_covariance_of_y_equal_x():
    """The pilot allocation keeps the H0/S covariance of one shot: y = x with E0 = omega = 1 has zero influence.

    One shot of group zero of an off-diagonal setting changes the lowest Ritz
    value by kappa (y - E0 x), so its per-shot variance is
    N kappa**2 (V_Y + E0**2 V_X - 2 E0 C_XY) from the sample-mean moments. On
    the table with ancilla signs x = (+1, +1, +1, -1) and system parity +1,
    y = x, every sample-mean (co)variance is (1 - 1/4)/3 = 1/4 exactly, and
    E0 = 1 cancels the bracket to exactly zero. The same moments without the
    covariance give 4 (9/16)(1/2) = 9/8 for kappa = 3/4, so an allocation that
    dropped it would move shots to a setting whose shots cannot change the
    energy. The check sits at the kernel because a workflow cannot fix the
    pilot's offset-free energy at exactly 1.

    The main-stage rule that reads these variances is checked on the
    unit-cost example v = (1, 4) with a main budget of 12 and the two-shot
    floor. The continuous minimizer of sum_s v_s/N_s, N_s = 12 sqrt(v_s)/3 =
    (4, 8), is already an integer and above the floor, so the allocation is
    (4, 8) with local variance 1/4 + 4/8 = 3/4.
    """
    from fractions import Fraction as F
    from nwqlib.algorithms.gcim.fixed_basis import pooled_statistics, reduce_group
    from nwqlib.algorithms.gcim.pencil_sampling import floored_allocation, influence_variances, local_variance

    setting = ("group_0_1_real_0", 0, 1, "real", 0)
    # Bit 0 is the ancilla phase and bit 1 the system qubit, read in Z.
    moment = reduce_group(_count_chunk(2, {0b00: 3, 0b01: 1}), setting, rows=(("Z", 1.0),), members=(0,),
                          num_qubits=1)
    assert (moment.mean_variance, moment.overlap_variance, moment.mean_covariance) == (.25, .25, .25)

    def variance(row):
        pools = {}
        pooled_statistics((setting,), (row,), 0.0, pools=pools)
        return influence_variances((setting,), pools, (.75,), 1.0)

    assert variance(moment) == (0.0,)
    assert variance(moment.revise(mean_covariance=0.0)) == (float(4 * F(9, 16) * F(1, 2)),)
    assert floored_allocation(12, (1.0, 4.0)) == (4, 8)
    assert local_variance((1.0, 4.0), (4, 8)) == float(F(1, 4) + F(4, 8))


def _pencil_sampling_plan(identity=0.0, overlap_cutoff=1e-12, total_shots=120):
    """FixedGCIM with PencilSampling(total_shots, 1/4) for A = identity I + Z in the real basis (|0>, |+>)."""
    import nwqlib
    from nwqlib.algorithms.gcim import PencilSampling
    from nwqlib.operators import ingest_pauli

    rows = (("I", identity),) * (identity != 0) + (("Z", 1.0),)
    method = FixedGCIM(basis=([1, 0], [2**-.5, 2**-.5]), overlap_cutoff=overlap_cutoff,
                       sampling=PencilSampling(total_shots=total_shots, pilot_fraction=.25))
    return nwqlib.plan(Eigenproblem(A=ingest_pauli(rows, num_qubits=1)), method=method, execution="quantum",
                       seed=7)


def _stage(result, stage):
    """The count chunks of one PencilSampling stage, by their bound ``stage`` parameter."""
    return [chunk for chunk in result.data.observations.chunks
            if {b.parameter: b.value for b in chunk.bindings}["stage"] == stage]


def _split(total, fraction, m):
    """Pilot and main shots and the pilot counts: B_p = min(B - 2m, max(2m, floor(val(f) B))), q/r of B_p."""
    p, q = fraction.as_integer_ratio()
    pilot = min(total - 2 * m, max(2 * m, p * total // q))
    return pilot, total - pilot, tuple(pilot // m + (k < pilot % m) for k in range(m))


def test_pencil_sampling_floors_the_deterministic_setting_and_keeps_its_frozen_decision(tmp_path, monkeypatch):
    """PencilSampling on A = Z in the basis (|0>, |+>): stage split, floored main counts, main-only estimate, offset, reload.

    The real basis has m = 3 settings in one group: (0, 0), (0, 1) real and
    (1, 1). B = 120 and f = 1/4 give B_p = min(B - 2m, max(2m, floor(B/4))) =
    30, ten pilot shots each, and 90 main shots. |0> gives Z = +1 on every
    shot, so the (0, 0) table has one bin and V_Y = 0 exactly under the
    centered reduction, and the floored allocation puts its v = 0 on the
    two-shot floor. The overlap setting has nonzero sensitivity at the
    pilot's lowest value near -1 and gets more shots, as does (1, 1). The Run
    has 2m completed submissions and B shots, and the estimate reads the
    main-stage tables only.

    A + 2**53 I under the same seed has the same circuits and counts, so the
    pilot moments are equal, while its physical pilot Ritz value, rounded
    once after the identity coefficient is added, has lost the offset-free
    e0. The influence variances, the offset-free pair, the reason, the main
    counts and the local variance are then bitwise equal. Saved, loaded and
    reanalyzed at another cutoff, the Result keeps the same decision without
    solving the pilot or allocating again.
    """
    from fractions import Fraction as F
    import nwqlib
    from nwqlib.algorithms.gcim import fixed_basis, pencil_workflow

    plan = _pencil_sampling_plan()
    result = nwqlib.solve(plan, progress=False)
    names = [experiment.name for experiment in plan.experiments]
    assert names == ["group_0_0_real_0", "group_0_1_real_0", "group_1_1_real_0"]
    allocation = result.shot_allocation
    pilot_shots, main_shots, pilot_counts = _split(120, .25, 3)
    assert (allocation.pilot_shots, allocation.main_shots, allocation.pilot_counts) == (30, 90, (10, 10, 10))
    assert (pilot_shots, main_shots, pilot_counts) == (30, 90, (10, 10, 10))
    fraction = allocation.effective_fraction
    assert F(fraction.numerator, fraction.denominator) == F(30, 120)
    assert allocation.fallback_reason is None and allocation.variances[0] == 0.0
    counts = allocation.main_counts
    assert counts[0] == 2 and min(counts[1:]) > 2 and sum(counts) == 90
    pilot, main = _stage(result, 0), _stage(result, 1)
    assert {c.experiment: c.returned_shots for c in pilot} == dict(zip(names, pilot_counts, strict=True))
    assert {c.experiment: c.returned_shots for c in main} == dict(zip(names, counts, strict=True))
    events = result.data.trace.events
    assert len(events) == 6 and all(e.status == "completed" for e in events) and sum(e.shots for e in events) == 120
    assert set(result.contribution_ids) == {c.content_id for c in main}
    assert not set(result.contribution_ids) & {c.content_id for c in pilot}
    assert "Shot allocation: pilot allocation; 30 pilot and 90 main shots" in str(result)

    alpha = 2.0**53
    shifted = nwqlib.solve(_pencil_sampling_plan(identity=alpha), progress=False)
    rec = plan.reconstruction

    def pilot_moments(run_result):
        return [row.model_dump(exclude={"contribution_id"}, exclude_computed_fields=True) for row in fixed_basis.read_groups(
            _stage(run_result, 0), rec.settings(), members=[g.indices for g in rec.groups],
            rows=rec.nonidentity_terms, num_qubits=1, basis_size=2, max_bytes=plan.method.max_bytes,
            max_analysis_work=plan.method.max_analysis_work)]

    assert pilot_moments(shifted) == pilot_moments(result)  # premise: matched pilot inputs
    e0 = allocation.pilot.offset_free_eigenvalues[0]
    assert F((e0 + alpha) - alpha) != F(e0)
    moved = shifted.shot_allocation
    assert (moved.variances, moved.pilot.offset_free_eigenvalues, moved.fallback_reason, moved.main_counts,
            moved.local_variance) == (allocation.variances, allocation.pilot.offset_free_eigenvalues,
                                      allocation.fallback_reason, counts, allocation.local_variance)
    assert moved.pilot.eigenvalues != allocation.pilot.eigenvalues

    path = result.save(tmp_path / "pencil_sampling")

    def fail(*args, **kwargs):
        pytest.fail("loading or reanalysis measured or allocated again")

    for name in ("pilot_decision", "solve_pencil", "floored_allocation", "uniform_allocation", "pilot_split",
                 "prepare_experiment", "submit_experiment"):
        monkeypatch.setattr(pencil_workflow, name, fail)
    loaded = nwqlib.load_result(path)
    assert loaded.shot_allocation == allocation and loaded.eigenvalue == result.eigenvalue
    for source in (result, loaded):
        again = source.analyze(overlap_cutoff=1e-3)
        assert again.shot_allocation == allocation and again.pencil.overlap_cutoff == 1e-3
    assert allocation.pilot.overlap_cutoff == 1e-12


def test_failed_pilot_solve_gives_a_fresh_uniform_main_stage():
    """A pilot that keeps no overlap direction sends the main stage to the uniform split of its budget.

    Each sampled quadrature of the overlap s of two states lies in [-1, 1],
    so the overlap eigenvalues 1 +- |s| stay below 3, and
    overlap_cutoff = 3 leaves the pilot solve without a usable subspace.
    B = 101 and f = 1/4 give B_p = 25, pilot counts (9, 8, 8), and B_main = 76,
    whose uniform split is (26, 25, 25), measured afresh. Reanalysis at a
    small cutoff solves the main pencil and keeps the decision.
    """
    import nwqlib

    result = nwqlib.solve(_pencil_sampling_plan(overlap_cutoff=3.0, total_shots=101), progress=False)
    allocation = result.shot_allocation
    assert allocation.pilot.failure_reason == "no_usable_overlap_subspace"
    assert (allocation.fallback_reason, allocation.variances, allocation.gap) == ("pilot_solve_failed", (), None)
    pilot_shots, main_shots, pilot_counts = _split(101, .25, 3)
    q, r = divmod(main_shots, 3)
    uniform = tuple(q + (k < r) for k in range(3))
    assert (pilot_shots, main_shots, pilot_counts, uniform) == (25, 76, (9, 8, 8), (26, 25, 25))
    assert (allocation.pilot_counts, allocation.main_counts) == (pilot_counts, uniform)
    names = [experiment.name for experiment in result.plan.experiments]
    assert {c.experiment: c.returned_shots for c in _stage(result, 1)} == dict(zip(names, uniform, strict=True))
    assert result.eigenvalue is None
    again = result.analyze(overlap_cutoff=1e-6)
    assert again.pencil.failure_reason is None and again.eigenvalue is not None
    assert again.shot_allocation == allocation


def test_reopened_pencil_sampling_run_measures_each_remaining_main_setting_once(tmp_path, monkeypatch):
    """A Run stopped inside the main stage resumes with the saved decision, the saved seed and no new pilot.

    The stop comes before the fifth submission, the second main setting, after
    its preparation with the seed checkpointed for it. Reopened from disk with
    the allocation functions disabled, the Run submits that prepared circuit
    and the last main setting once each and decides nothing again: six
    submissions, three per stage, and B = 120 shots in all.
    """
    import nwqlib
    from nwqlib.algorithms.gcim import pencil_workflow
    from nwqlib.algorithms.gcim.fixed_basis import ShotAllocation
    from nwqlib.backends import AerBackend
    from nwqlib.core.planning import Realization

    plan = _pencil_sampling_plan()
    original, submitted = FixedGCIM.before_submit, []

    def interrupted(self, *args, **kwargs):
        if len(submitted) == 4:
            raise RuntimeError("stop before the fifth submission")
        submitted.append(1)
        return original(self, *args, **kwargs)

    path = tmp_path / "pencil-run"
    with Run(plan, directory=path) as run:
        with monkeypatch.context() as patch:
            patch.setattr(FixedGCIM, "before_submit", interrupted)
            with pytest.raises(RuntimeError, match="fifth submission"):
                run.resume()
        state = run.checkpoint_state
        first = tuple(chunk.content_id for chunk in run.observations.chunks)
    assert (state["stage"], state["index"], len(first)) == (1, 1, 4)
    pending = state["pending"]

    def fail(*args, **kwargs):
        pytest.fail("the reopened Run decided its shots again")

    for name in ("pilot_decision", "floored_allocation", "uniform_allocation", "pilot_split"):
        monkeypatch.setattr(pencil_workflow, name, fail)
    with nwqlib.load_run(path, backend=AerBackend()) as restored:
        result = restored.wait()
        events = restored.trace.events
        records = {record.content_id: record for record in restored.prepared_artifacts}
    assert tuple(chunk.content_id for chunk in result.data.observations.chunks[:4]) == first
    assert len(events) == 6 and sum(e.shots for e in events) == 120 and len(records) == 6
    assert len(_stage(result, 0)) == 3
    main = _stage(result, 1)
    assert sorted(c.experiment for c in main) == sorted(e.name for e in plan.experiments)
    point = Realization.model_validate(pending["point"]).content_id
    (resumed,) = [chunk for chunk in main if chunk.realization_id == point]
    assert resumed.prepared_id == pending["prepared_id"]
    assert records[resumed.prepared_id].runtime.seed == pending["seed"]
    assert result.shot_allocation == ShotAllocation.model_validate(state["decision"])


def test_grouped_analysis_is_admitted_before_setting_construction(monkeypatch):
    import numpy as np
    import pytest
    import nwqlib
    from nwqlib.algorithms.gcim import FixedGCIM, fixed_basis
    from nwqlib.operators import ingest_pauli

    rng = np.random.default_rng(31)
    basis = tuple(rng.normal(size=8) + 1j * rng.normal(size=8) for _ in range(3))
    rows = (("III", .7), ("ZII", .5), ("IZI", -.3), ("ZZI", .2),
            ("XII", .4), ("IYI", -.25), ("XYI", .1))
    problem = nwqlib.Eigenproblem(A=ingest_pauli(rows, num_qubits=3))

    def select(cap):
        return nwqlib.plan(problem, method=FixedGCIM(basis=basis, max_analysis_work=cap),
                           execution="quantum", shots=64, seed=104)

    def unexpected_construction(*args, **kwargs):
        raise AssertionError("grouped settings constructed before analysis admission")

    # Two groups of three terms, 18 settings on 6 pairs, at most 16 bins on
    # four bits. The last two terms are the pooling and pair assembly.
    work = 18 * (16 * (4 + 44 + 8 * 3) + 3 * 4) + 6 * 4 + 32 * 9 + 16 * 27 + 23 * 18 + 22 * 6
    assert work == 22242
    # The floor that FixedGCIM.plan checks before grouping charges one group
    # of all six terms to each of the K = 9 settings. It is the planned work
    # less K(G - 1)[m(c + 44) + 23] = 9 * (16 * 48 + 23) for G = 2, m = 16 and
    # c = 4.
    floor = 9 * (16 * (4 + 44 + 8 * 6) + 6 * 4) + 6 * 4 + 32 * 9 + 16 * 27 + 23 * 9 + 22 * 6
    assert floor == work - 9 * (16 * 48 + 23) == 15123
    with monkeypatch.context() as guarded:
        guarded.setattr(fixed_basis, "_sampled_construction", unexpected_construction)
        # The one-group floor rejects 5000 and the actual grouped work
        # rejects work - 1, both before any sampled setting is constructed.
        with pytest.raises(ValueError, match=rf"known analysis requirements need at least {floor} work units, "
                                             r"exceeding max_analysis_work=5000\b"):
            select(5000)
        with pytest.raises(ValueError, match=rf"max_analysis_work={work - 1}\b"):
            select(work - 1)
    selected = select(work)
    assert len(selected.experiments) == 18
    assert selected.shots == 64
    # The same planned population in compact form: one (L_g, 16) descriptor
    # per group, repeated for the K = 9 settings of each group.
    assert fixed_basis.grouped_pass_requirements(
        3, [(len(group.indices), 16) for group in selected.reconstruction.groups], 6, 3, settings=18, pairs=6,
        oriented=False, copies=9) == (13882 + 256 * 18 + 296 * 6, work)

    # The byte bound refuses first on a wider register: -Z^10 - 0.7 X^10 has
    # G = 2 groups of one term, b = 2, n = 10, c = 11, w = 1 and
    # m = min(4096, 2**11) = 2048 bins, so the planned bytes B_plan are
    # 8*2*4*2*2048 + 112*2048 + 8*1 + 8*4*(2+16*2) + 26*2 + 512*4, plus
    # 256*8 + 296*3 for pooling the K*G = 8 settings on P = 3 pairs.
    n = 10
    wide = nwqlib.Eigenproblem(A=ingest_pauli((("Z" * n, -1.0), ("X" * n, -.7)), num_qubits=n))
    rng = np.random.default_rng(5)
    states = tuple(rng.normal(size=2**n) + 1j * rng.normal(size=2**n) for _ in range(2))
    size = 8 * 2 * 4 * 2 * 2048 + 112 * 2048 + 8 * 1 + 8 * 4 * (2 + 16 * 2) + 26 * 2 + 512 * 4 + 256 * 8 + 296 * 3
    assert size == 497652

    def select_wide(cap):
        return nwqlib.plan(wide, method=FixedGCIM(basis=states, max_bytes=cap),
                           execution="quantum", shots=4096, seed=1)

    with monkeypatch.context() as guarded:
        guarded.setattr(fixed_basis, "_sampled_construction", unexpected_construction)
        with pytest.raises(ValueError, match=rf"GCIM planned grouped analysis needs {size} data bytes"):
            select_wide(size - 1)
    assert len(select_wide(size).experiments) == 8


def test_default_analysis_work_admits_a_fourteen_qubit_grouped_plan():
    """The default max_analysis_work admits a 14-qubit grouped plan at 8,192 shots.

    Its planning bound, the grouped-pass law with one chunk per planned
    setting, is 308,208,312 work units, above 10**8
    (docs/ENGINEERING_CONSTANTS.md, row "Sampled FixedGCIM analysis
    admission"). Planning only; no circuit is run.
    """
    import nwqlib
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_product

    rng = np.random.default_rng(11)
    n = 14
    labels = set()
    while len(labels) < 80:
        labels.add("".join(rng.choice(list("IIXYZ"), size=n)))
    rows = tuple((label, float(rng.normal())) for label in sorted(labels) if label != "I" * n)
    basis = tuple(ingest_product([[np.cos(t), np.sin(t)]] * n) for t in (.3, .6, .9))
    problem = nwqlib.Eigenproblem(A=ingest_pauli(rows, num_qubits=n))

    def select(**limits):
        return nwqlib.plan(problem, method=FixedGCIM(basis=basis, **limits), execution="quantum",
                           shots=8192, seed=1)

    assert len(select().experiments) == 540
    with pytest.raises(ValueError, match=r"needs at most 308208312 work units.*max_analysis_work=100000000\."):
        select(max_analysis_work=10**8)


def test_native_depth_limit_refuses_the_first_deeper_setting_and_keeps_the_earlier_data(
    tmp_path, monkeypatch, ibm_runtime_spy
):
    """A native two-qubit depth limit admits each IBM ISA setting at or below it and refuses the first deeper one before its job.

    Three occupation states on two qubits under H = ZZ + 0.5 ZI give six
    real grouped settings on the offline FakeManilaV2 device. Their ISA
    circuits differ in native two-qubit depth, because the controlled
    occupation preparations add CX gates and routing adds SWAP gates. The
    limits come from the depths recorded by a run without a limit, whose
    compilation the same seed repeats. The largest recorded depth D admits
    all six settings, D itself included, and the limit D - 1 admits the
    settings before the first one deeper than D - 1 and refuses that one
    after its preparation and before any job for it. Its earlier counts and
    shots stay in the Run, the analysis of the collected data lists every
    setting without data as missing and has no pencil, and neither a second
    resume nor a reopened Run prepares or submits again. `solve`, which
    returns no Run, names the folder that keeps the same partial data. A
    detached batch of the depth-0 setting and a deeper one under the limit 0
    is refused as a whole before any submission is recorded, and the
    shallow setting alone is then submitted. Aer
    records no native depth, so a limit refuses its first setting before any
    job; without a limit the existing Aer workflows above run unchanged.
    Only qiskit-ibm-runtime 0.49.0 qualifies an IBM depth, so under another
    installed version every IBM depth is unqualified and a limit refuses the
    first setting as on Aer. The synthetic counts decide no relation here.
    """
    import importlib.metadata
    from pathlib import Path

    import nwqlib
    from nwqlib.backends import IBMRuntimeBackend
    from nwqlib.backends.inspection import NativeDepthCapRefusal
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation

    spy, shots = ibm_runtime_spy, 8
    problem = Eigenproblem(A=ingest_pauli((("ZZ", 1.0), ("ZI", 0.5)), num_qubits=2))
    basis = tuple(ingest_occupation(bits, num_qubits=2) for bits in ("00", "01", "11"))

    def start(limit, backend=spy.connection, directory=None):
        selected = nwqlib.plan(problem, method=FixedGCIM(basis=basis, max_native_two_qubit_depth=limit),
                               execution="quantum", shots=shots, seed=7)
        return selected, nwqlib.prepare(selected, backend=backend, directory=directory, progress=False)

    def finish(prepared):
        run = nwqlib.submit(prepared)
        while run.result is None:
            run.resume()
        return run.result

    def recorded(run):
        return {a.realization.experiment: a.native_depth for a in run.prepared_artifacts}

    selected, prepared = start(None, directory=tmp_path / "unlimited")
    names = [experiment.name for experiment in selected.experiments]
    with prepared.run as run:
        assert finish(prepared).missing == ()
        depths = recorded(run)
    assert list(depths) == names and len(spy.api.requests) == 6

    _, prepared = start(0, backend=None)
    with prepared.run as run:
        with pytest.raises(NativeDepthCapRefusal) as refused:
            finish(prepared)
        assert (refused.value.experiment, refused.value.native_depth) == (names[0], None)
        assert run.trace.events == () and run.exposure["completed"]["shots"] == 0

    installed = importlib.metadata.version("qiskit-ibm-runtime")
    if installed != "0.49.0":
        assert all(not depth.qualified and f"qiskit-ibm-runtime {installed} is installed" in depth.reason
                   for depth in depths.values())
        _, prepared = start(0, directory=tmp_path / "unchecked")
        with prepared.run as run:
            with pytest.raises(NativeDepthCapRefusal) as refused:
                finish(prepared)
            assert (refused.value.experiment, refused.value.depth) == (names[0], None)
            assert run.trace.events == () and len(spy.api.requests) == 6
        return
    assert all(depth.qualified for depth in depths.values())
    depths = {name: depth.native_two_qubit_depth for name, depth in depths.items()}
    deepest = max(depths.values())
    assert depths[names[0]] == 0 < deepest  # |00> needs no two-qubit gate

    _, prepared = start(deepest, directory=tmp_path / "at_largest")
    with prepared.run as run:
        assert finish(prepared).missing == ()
        assert {name: depth.native_two_qubit_depth for name, depth in recorded(run).items()} == depths
    assert len(spy.api.requests) == 12

    limit = deepest - 1
    first = next(index for index, name in enumerate(names) if depths[name] > limit)
    path = tmp_path / "refused"
    selected, prepared = start(limit, directory=path)
    run = prepared.run
    # prepare compiles only the first setting, whose recorded depth can be read before any job.
    (prepared_first,) = run.prepared_artifacts
    assert prepared_first.native_depth.native_two_qubit_depth == depths[names[0]]
    with run:
        with pytest.raises(NativeDepthCapRefusal) as refused:
            finish(prepared)
        artifacts = {a.realization.experiment: a for a in run.prepared_artifacts}
        assert list(artifacts) == names[:first + 1]
        assert (refused.value.experiment, refused.value.prepared_id, refused.value.cap, refused.value.depth) == (
            names[first], artifacts[names[first]].content_id, limit, depths[names[first]])
        assert all(event.status == "completed" for event in run.trace.events)
        assert {event.prepared_id for event in run.trace.events} == {
            artifacts[name].content_id for name in names[:first]}
        exposure = run.exposure
        assert (exposure["completed"]["jobs"], exposure["completed"]["shots"]) == (first, first * shots)
        assert exposure["reserved"]["circuits"] == exposure["uncertain"]["circuits"] == 0
        assert len(spy.api.requests) == 12 + first
        partial = selected.method.analyze(selected, run.data, settings={})
        assert partial.missing == tuple(names[first:])
        assert partial.pencil is None and partial.eigenvalue is None
        with pytest.raises(NativeDepthCapRefusal):
            run.resume()
        assert len(run.prepared_artifacts) == first + 1 and len(spy.api.requests) == 12 + first

    # solve keeps its Run, which IBM Runtime saves under ~/.nwqlib/runs, so
    # its refusal names that folder.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("nwqlib._prepared_execution.sleep", lambda seconds: None)
    with pytest.raises(NativeDepthCapRefusal) as solved:
        nwqlib.solve(selected, backend=spy.connection, progress=False)
    folder = Path(solved.value.__notes__[-1].removeprefix("Original execution data is saved at "))
    assert folder.parent == tmp_path / ".nwqlib" / "runs" and len(spy.api.requests) == 12 + 2 * first

    compile_for_ibm = IBMRuntimeBackend.prepare
    monkeypatch.setattr(IBMRuntimeBackend, "prepare", lambda *a, **k: pytest.fail("reopening compiled again"))
    with nwqlib.load_run(path, backend=spy.connection, progress=False) as run:
        with pytest.raises(NativeDepthCapRefusal) as again:
            run.resume()
        assert again.value.prepared_id == refused.value.prepared_id
        assert len(run.trace.events) == first and len(spy.api.requests) == 12 + 2 * first
    with nwqlib.load_run(folder, backend=spy.connection, progress=False) as run:
        assert len(run.trace.events) == first
        assert selected.method.analyze(selected, run.data, settings={}).missing == tuple(names[first:])

    # A detached batch is checked as a whole before its submission intent: one member above the limit refuses
    # the tuple, so neither member is submitted, and the shallow member alone is then submitted.
    from nwqlib.execution import Run, prepare_experiment, submit_detached

    shallow, deep = names[0], next(name for name in names if depths[name] > 0)
    selected = nwqlib.plan(problem, method=FixedGCIM(basis=basis, max_native_two_qubit_depth=0),
                           execution="quantum", shots=shots, seed=7)
    sent = len(spy.api.requests)
    with Run(selected, backend=spy.connection, directory=tmp_path / "batch") as run, monkeypatch.context() as patch:
        patch.setattr(IBMRuntimeBackend, "prepare", compile_for_ibm)
        handles = tuple(prepare_experiment(selected.resolve(name), run=run) for name in (shallow, deep))
        depth = [handle.record.native_depth.native_two_qubit_depth for handle in handles]
        assert depth[0] == 0 < depth[1]
        with pytest.raises(NativeDepthCapRefusal) as refused:
            submit_detached(handles, run=run)
        assert refused.value.prepared_id == handles[1].record.content_id
        assert run.trace.events == () and len(spy.api.requests) == sent
        submit_detached(handles[:1], run=run)
        assert len(run.trace.events) == 1 and len(spy.api.requests) == sent + 1


def test_default_limits_plan_compact_26_qubit_gcim_without_full_state_work(monkeypatch):
    """FixedGCIM, ADAPT and QFD plan compact 26-qubit inputs under their default limits and form no 2**26-entry object.

    H on n = 26 qubits has the 26 terms -0.5 Z_k, the 25 terms
    0.25 Z_k Z_k+1 and the 13 terms 0.1 X_k X_k+1 on even k, so its
    qubit-wise commuting groups are the 51 Z words and the 13 X words,
    G = 2. Every state is an occupation, prepared by X gates, and H has no Y,
    so FixedGCIM and ADAPT prove the input real. FixedGCIM with three occupation
    states and 8192 shots selects b(b+1)G/2 = 12 settings, 12 * 8192 shots,
    on 27 logical qubits, the system and one ancilla. In each group its three
    diagonal pairs measure the 26 system bits and its three off-diagonal
    pairs also the ancilla, so 8192 * 2 * 3 * (26 + 27) measurements. ADAPT
    with four real pool words i X_a Y_b and two selections keeps the same
    two energy groups. QFD with the powers (0, 1, 2) of one order-2 step
    measures both quadratures, M**2 G = 18 settings, and its CX count per
    shot is the sum over settings of r|b| C_U + r|q - b| C_C, the base
    power b and the other power q of each pair, with C_U = 4(sum_k w_k - L)
    and C_C = 4 sum_k w_k for the L words of weight w_k, since the
    occupation reference and the readout need no CX. QFD with the
    fermionic generator ``B = sum_k 0.3 n_k + sum_{even k} 0.1 (a_k^dagger
    a_{k+1} + h.c.)`` on 26 modes measures the same 18 settings. B has 39
    blocks, a number block ``n_k = (I - Z_k)/2`` with the one word Z_k and a
    hopping block with the words ``X_k X_{k+1}`` and ``Y_k Y_{k+1}``, so
    one order-2 substep has 2 * 52 factors in 78 blocks, and the same CX
    sum holds over these 52 words. FixedGCIM with
    PencilSampling(4 * 12 * 8192, 1/4), B = 393216, measures the same 12
    settings in a pilot stage of 98304 shots and a main stage of 294912.
    Planning bounds the grouped analysis of one stage with the larger stage
    budget S = 294912 at 141276072 work units and 37765896 bytes, within
    the default limits, and the pre-group work floor is 20082. A planned
    call without the stage budget would count S stored outcomes for each of
    the 12 settings, 1157247912 work units, and exceed the default
    max_analysis_work = 10**9,
    so this plan also checks that planning passes the stage budget. Its
    whole-Plan estimate leaves the settings and shots unavailable with the
    PencilSampling reason while the exact evaluations of its counts-only
    measurements are zero.
    All plans and
    resource estimates pass no limit, so the defaults apply. None forms a
    statevector, an operator matrix or a dense Pauli matrix, and the peak
    traced by tracemalloc, which NumPy reports its buffers to, stays below
    2**26 bytes, the size of the smallest array with one entry per basis
    state. Planning only; no circuit is prepared or run.
    """
    import tracemalloc
    from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
    import nwqlib
    from nwqlib.algorithms import ADAPT, QFD, PencilSampling
    from nwqlib.operators import ingest_fermion, ingest_pauli
    from nwqlib.problems import ingest_occupation
    from nwqlib.resources import ResourceContext

    n, shots = 26, 8192
    generator = ingest_fermion([(.3, [(k, 1), (k, 0)]) for k in range(n)] + [
        (.1, [(k + d, 1), (k + 1 - d, 0)]) for k in range(0, n - 1, 2) for d in (0, 1)], num_modes=n)

    def word(letters):
        positions = ["I"] * n
        for qubit, letter in letters:
            positions[qubit] = letter
        return "".join(positions)

    def occupied(*qubits):
        return ingest_occupation([int(k in qubits) for k in range(n)], num_qubits=n)

    rows = ([(word([(k, "Z")]), -0.5) for k in range(n)]
            + [(word([(k, "Z"), (k + 1, "Z")]), 0.25) for k in range(n - 1)]
            + [(word([(k, "X"), (k + 1, "X")]), 0.1) for k in range(0, n - 1, 2)])
    problem = Eigenproblem(A=ingest_pauli(tuple(rows), num_qubits=n))
    pool = tuple(ingest_pauli(((word([(a, "X"), (b, "Y")]), 1j),), num_qubits=n)
                 for a, b in ((0, 2), (1, 3), (0, 3), (1, 2)))

    def forbidden(*args, **kwargs):
        pytest.fail("compact 26-qubit planning formed a full-state or dense operator object")

    for owner, name in ((Statevector, "__init__"), (Operator, "__init__"), (SparsePauliOp, "to_matrix")):
        monkeypatch.setattr(owner, name, forbidden)
    tracemalloc.start()
    try:
        fixed = nwqlib.plan(problem, method=FixedGCIM(basis=(occupied(0, 1), occupied(0, 2), occupied(1, 3))),
                            execution="quantum", shots=shots, seed=3)
        fixed_workload = nwqlib.estimate(fixed)
        staged = nwqlib.plan(problem, method=fixed.method.revise(
            sampling=PencilSampling(total_shots=4 * 12 * shots, pilot_fraction=.25)), execution="quantum", seed=3)
        staged_workload = nwqlib.estimate(staged)
        adapt = nwqlib.plan(problem, method=ADAPT(initial_state=occupied(0, 1), pool=pool, max_iterations=2),
                            execution="quantum", shots=shots, seed=3)
        adapt_workload = nwqlib.estimate(adapt)
        # ADAPT with pair_preparation="vacuum_restore" and the hopping pool
        # 0.5 i(X_a X_b + Y_a Y_b), whose flips of the all-zero state cancel
        # (1 - 1 = 0), declares W_restore = 2 W_ref + 3l(1 + B) + 12288 l
        # + 2l(T_max + 1) + 8n + 7 for the occupation reference's W_ref = n,
        # l = 2 selections, T_max = 2 terms and the commuting bound
        # B = 6 n T_max, so 2*26 + 6*313 + 24576 + 12 + 215 = 26733.
        from nwqlib.algorithms import PencilAcquisition

        hopping = tuple(ingest_pauli(((word([(a, "X"), (b, "X")]), .5j), (word([(a, "Y"), (b, "Y")]), .5j)),
                                     num_qubits=n) for a, b in ((0, 2), (1, 3)))
        restore = nwqlib.plan(problem, method=ADAPT(initial_state=occupied(0, 1), pool=hopping, max_iterations=2,
                                                    acquisition=PencilAcquisition(pair_preparation="vacuum_restore")),
                              execution="quantum", shots=shots, seed=3)
        assert restore.reconstruction.acquisition.pair_preparation == "vacuum_restore"
        assert {s.construction_work for s in restore.construction.selections} == {
            2 * n + 3 * 2 * (1 + 6 * n * 2) + 12288 * 2 + 2 * 2 * (2 + 1) + 8 * n + 7} == {26733}
        qfd = nwqlib.plan(problem, method=QFD(initial_state=occupied(0, 1), time_step=.1, powers=(0, 1, 2)),
                          execution="quantum", shots=shots, seed=3)
        qfd_workload = nwqlib.estimate(qfd)
        qfd_cx = nwqlib.estimate(qfd, context=ResourceContext(basis="cx"))
        fermionic = nwqlib.plan(problem, method=QFD(initial_state=occupied(0, 1), time_step=.1, powers=(0, 1, 2),
                                                    evolution="fermionic_blocks", evolution_terms=generator),
                                execution="quantum", shots=shots, seed=3)
        fermionic_workload = nwqlib.estimate(fermionic)
        fermionic_cx = nwqlib.estimate(fermionic, context=ResourceContext(basis="cx"))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2**n

    def count(workload, metric, **location):
        return workload.quantity(metric, **location).fact.value.numerator

    groups = sorted(len(group.indices) for group in fixed.reconstruction.groups)
    assert groups == [13, 51] and fixed.reconstruction.acquisition.quadratures == "real"
    assert len(fixed.experiments) == count(fixed_workload, "settings") == 12
    assert count(fixed_workload, "shots") == 12 * shots
    assert count(fixed_workload, "logical_width", location="logical_device") == n + 1
    assert count(fixed_workload, "measurements") == shots * 2 * 3 * (n + n + 1)
    assert [e.name for e in staged.experiments] == [e.name for e in fixed.experiments]
    # The staged plan separates the two work bounds only because 141276072 <= 10**9 < 1157247912, so a
    # larger default would silently stop it detecting a planned call that drops shot_total.
    assert FixedGCIM.model_fields["max_analysis_work"].default == 10**9
    reason = staged.method.query_catalog_reason()
    assert reason.startswith("PencilSampling measures every setting in a pilot stage")
    assert all(staged_workload.quantity(metric).fact.reason == reason for metric in ("settings", "shots"))
    assert count(staged_workload, "exact_evaluations") == 0
    rec = adapt.reconstruction
    assert sorted(map(len, rec.energy_groups)) == [13, 51] and rec.acquisition.quadratures == "real"
    assert (len(rec.pool), rec.max_selections, rec.num_qubits) == (4, 2, n)
    assert count(adapt_workload, "system", location="logical_device") == n
    pencil = qfd.reconstruction.pencil
    assert sorted(len(group.indices) for group in pencil.groups) == [13, 51]
    assert pencil.acquisition.quadratures == "complex"
    assert len(qfd.experiments) == count(qfd_workload, "settings") == 18
    assert count(qfd_workload, "shots") == 18 * shots
    assert count(qfd_workload, "logical_width", location="logical_device") == n + 1
    def cx_per_shot(weights):
        c_u, c_c = 4 * (sum(weights) - len(weights)), 4 * sum(weights)
        total = 0
        for i, p in enumerate((0, 1, 2)):
            total += 2 * p * c_u  # the diagonal settings of the two groups
            for q in (0, 1, 2)[i + 1:]:
                base = min(p, q, key=abs)
                total += 2 * 2 * (abs(base) * c_u + abs(q - p) * c_c)  # two quadratures, two groups
        return total

    assert count(qfd_cx, "cx") == shots * cx_per_shot([len(label) - label.count("I") for label, _ in rows])
    rec = fermionic.reconstruction
    assert (rec.step_rows.array.size, rec.step_blocks.array.size) == (2 * 52, 2 * 39)
    assert len(fermionic.experiments) == count(fermionic_workload, "settings") == 18
    assert count(fermionic_workload, "shots") == 18 * shots
    assert count(fermionic_cx, "cx") == shots * cx_per_shot([1] * n + [2] * n)
