"""Fixed GCiM transition amplitudes, acquisition association and finite admission."""

import numpy as np
import pytest
from nwqlib.algorithms.gcim import FixedGCIM
from nwqlib.core.planning import RandomStreams
from nwqlib.execution import ExecutionLimits
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


def test_foreign_observations_reject_before_projected_work(monkeypatch):
    import nwqlib.algorithms.gcim.fixed_basis as owner

    plan = plan_for()
    data = Run(plan_for(A=np.diag([2.0, -1.0]))).wait(timeout=5, poll_interval=0).data
    monkeypatch.setattr(
        owner, "_solve_projected_pencil", lambda *a, **k: pytest.fail("foreign data reached solve")
    )
    with pytest.raises(ValueError, match="another selected Plan"):
        plan.method.analyze(plan, data, settings={})


def test_caps_and_scalar_identity_precede_native_work(monkeypatch):
    import nwqlib.blocks.lowering as lowering

    monkeypatch.setattr(
        lowering, "_lower_qiskit", lambda *a, **k: pytest.fail("inadmissible native preparation")
    )
    run = Run(plan_for(), limits=ExecutionLimits(max_total_circuits=1))
    with pytest.raises(ValueError, match=r"ExecutionLimits\(max_total_circuits="):
        run.plan.method.prepare(run.plan, run=run)
    scalar = Run(plan_for(A=np.eye(2) * 3)).wait(timeout=5, poll_interval=0)
    assert scalar.eigenvalue == 3 and not scalar.data.trace.events
    # Two basis states and one nonidentity term need b(b+1)/2 = 3 exact pair
    # acquisitions (pencil.pair_count); the refusal names that count and the cap.
    with pytest.raises(ValueError, match="population 3 exceeds max_experiments=1 before expansion"):
        plan_for(max_experiments=1)


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


@pytest.mark.parametrize("kind, q, b", (("csr", 14, 1), ("csr", 14, 2), ("csr", 14, 6), ("csr", 12, 8), ("dense", 10, 2), ("dense", 10, 16)))
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



def test_unbounded_off_diagonal_gram_refusal_names_its_remedies(monkeypatch):
    """A negative Gram mode refused without off-diagonal bounds shows the remedies that apply to it.

    The rank-deficient basis (a, b, exp(0.7i) a) solves with qualified pair
    receipts. With every receipt's state error made unavailable, the solve
    admits no input error beyond its roundoff and refuses the computed
    negative mode; the summary then names sampled acquisition, ADAPT and a
    smaller basis, and says that overlap_cutoff does not change the check.
    """
    from qiskit import QuantumCircuit
    from nwqlib import solve
    from nwqlib.execution import PreparedArtifact
    from nwqlib.algorithms.gcim.fixed_basis import UNQUALIFIED_GRAM_REFUSAL
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import state_input

    def layered(seed, phase=0.0):
        draw = np.random.default_rng(seed)
        circuit = QuantumCircuit(2, global_phase=phase)
        for _ in range(300):
            for qubit in range(2):
                circuit.u(*draw.uniform(-np.pi, np.pi, 3), qubit)
            circuit.cx(0, 1)
        return circuit

    def run():
        return solve(Eigenproblem(A=ingest_pauli((("ZI", .5), ("XX", .3), ("IY", .2)), num_qubits=2)),
                     method=FixedGCIM(basis=tuple(state_input(c) for c in (layered(4), layered(104), layered(4, .7)))),
                     execution="quantum", seed=7)

    assert UNQUALIFIED_GRAM_REFUSAL not in str(run())
    monkeypatch.setattr(PreparedArtifact, "state_error", lambda self, resolved=(): (None, "unqualified"))
    refused = run()
    assert refused.pencil.failure_reason == "negative_deterministic_overlap_eigenvalue"
    assert UNQUALIFIED_GRAM_REFUSAL in str(refused)

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


def test_projected_energy_chunks_record_no_recovery_scale():
    """FixedGCIM host scalars are energies, so no basis-order-dependent scale is recorded."""
    import nwqlib
    from nwqlib.algorithms.lanczos import Lanczos

    H = np.array([[1.0, 0.3, 0, 0.1], [0.3, -0.5, 0.2, 0], [0, 0.2, 0.4, 0.1j], [0.1, 0, -0.1j, -1.0]])
    c1, c2 = 3.0 * np.array([1, 1, 0, 0]), (0.5 - 0.5j) * np.array([0, 1, 1j, 1])
    energies = []
    for basis in ((c1, c2), (c2, c1)):
        result = nwqlib.solve(Eigenproblem(A=H), method=FixedGCIM(basis=basis), execution="classical", seed=1)
        (chunk,) = result.data.observations.chunks
        assert chunk.physical_scale is None and "no recovery applies" in chunk.physical_scale_unavailable
        energies.append(result.eigenvalue)
    assert energies[0] == pytest.approx(energies[1], rel=0, abs=1e-12)
    # Lanczos keeps the norm of its supplied input state.
    u = 5j * np.array([1.0, 0.2, -0.3, 0.1j])
    lanczos = nwqlib.solve(Eigenproblem(A=H), method=Lanczos(initial_state=u, krylov_dimension=2),
                           execution="classical", seed=1)
    (chunk,) = lanczos.data.observations.chunks
    assert chunk.physical_scale.as_float() == pytest.approx(np.linalg.norm(u), rel=1e-15, abs=0)


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


def test_exact_plan_at_one_hundred_qubits_plans_and_estimates_with_default_caps():
    """Exact FixedGCIM planning and estimation do not admit the execution-time pair-reduction work.

    Two product trial states of the parity Hamiltonian ``-Z^q - 0.7 X^q`` at
    q = 100 have three pair reductions (``pencil.pair_count``) whose summed
    registered work exceeds the default ``max_classical_products``. That sum
    is a host cost of acquisition, admitted when a pair is prepared, so
    planning and resource estimation proceed with the default caps.
    """
    from nwqlib import estimate
    from nwqlib.algorithms.gcim.pencil import pair_count
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import ingest_product

    q = 100
    parity = ingest_pauli((("Z" * q, -1.0), ("X" * q, -0.7)), num_qubits=q)
    basis = tuple(ingest_product([[np.cos(t), np.sin(t)]] * q) for t in (np.pi / 8, 3 * np.pi / 8))
    plan = plan_for(A=parity, basis=basis)
    assert len(plan.experiments) == pair_count(2, 2)
    exact = estimate(plan).quantity("exact_evaluations").fact.value
    assert (exact.numerator, exact.denominator) == (pair_count(2, 2), 1)


def test_pair_reduction_hook_decodes_each_pair_point_once_per_plan(monkeypatch):
    """One hook call per point decodes the Plan's M pair points a bounded number of times.

    The Plan-level total is fixed by its experiments. Re-decoding it on each
    hook call costs M parameter decodes per call, M**2 over one pass of the
    Plan's points, which at b = 48 (M = 1176) took seconds of host time.
    Summed once, a pass decodes each point for the total and once as its
    own point, at most 2M.
    """
    from nwqlib.algorithms.gcim import pair_reducer
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import ingest_product

    rng = np.random.default_rng(1)
    basis = tuple(ingest_product([(np.cos(t), np.sin(t)), (np.cos(s), np.exp(1j * s) * np.sin(s))])
                  for t, s in rng.uniform(0, 3, size=(48, 2)))
    plan = plan_for(A=ingest_pauli((("ZI", .5), ("XY", .3), ("IX", .2)), num_qubits=2), basis=basis)
    readouts = [e.readout for e in plan.experiments]
    assert len(readouts) == 48 * 49 // 2
    calls = [0]
    original = pair_reducer.pair_work

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pair_reducer, "pair_work", counted)
    for readout in readouts:
        plan.method.reduction_allowance(plan, None, observation=readout, width=5, run=None)
    assert calls[0] <= 2 * len(readouts)


def _smallest_exact_adapt():
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.operators import ingest_pauli

    pool = tuple(ingest_pauli(((p, 1j),), num_qubits=1) for p in ("Y", "X"))
    return (Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1)),
            ADAPT(initial_state=[1, 1], pool=pool, theta=np.pi / 8, max_iterations=1))


@pytest.mark.parametrize("build", (
    lambda: (Eigenproblem(A=np.diag([1., -1.])), FixedGCIM(basis=([1, 1], [1, 1j]))),
    _smallest_exact_adapt,
), ids=("FixedGCIM", "ADAPT"))
def test_saved_exact_gcim_result_loads_in_a_fresh_process(tmp_path, build):
    """load_result in a new interpreter resolves the saved pair reduction before any GCiM module loads."""
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
              "repr([(p.pair, p.hamiltonian, p.overlap) for p in getattr(loaded, 'pairs', ())])")
    source = (f"import json, nwqlib\nloaded = nwqlib.load_result({str(tmp_path / 'result')!r})\n"
              f"print(json.dumps([nwqlib.__file__, {fields}]))")
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=dict(os.environ),
                               timeout=300)
    assert completed.returncode == 0, completed.stderr
    origin, *loaded = json.loads(completed.stdout.splitlines()[-1])
    assert origin == nwqlib.__file__
    assert loaded == [repr(result.eigenvalue), repr(result.pencil.hamiltonian), repr(result.pencil.overlap),
                      repr([(p.pair, p.hamiltonian, p.overlap) for p in getattr(result, "pairs", ())])]


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


def test_repeated_pair_acquisitions_pool_with_population_weights_and_their_bound():
    """Two acquisitions of one exact pair pool into their population-weighted mean with a pooled bound.

    Each reduction contributes population one, so the pooled entry is the
    equal-weight mean of the two raw reductions. ``pair_reducer.pooled_bound``
    adds rounding terms to the weighted mean of the contribution bounds, so
    the pooled bound exceeds that mean for nonzero entries. The pooled entries lie within
    their pooled bounds of the dense ``V^dagger H0 V`` and ``V^dagger V``.
    """
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.core.planning import RuntimeOptions
    from nwqlib._prepared_execution import prepare_experiment, submit_experiment
    from nwqlib.operators import ingest_pauli

    terms = (("ZI", .5), ("XY", .3), ("IX", .2))
    basis = ([1, 0, 0, 0], [.6, .8j, 0, 0])
    plan = plan_for(A=ingest_pauli(terms, num_qubits=2), basis=basis)
    raw = []
    with Run(plan) as run:
        for experiment in plan.experiments:
            handle = prepare_experiment(plan.resolve(experiment.name), run=run, runtime=RuntimeOptions(seed=7))
            for _ in range(2 if experiment.name == "pair_0_1" else 1):
                for chunk in submit_experiment(handle, run=run):
                    run.collect(chunk)
                    if experiment.name == "pair_0_1":
                        (value,) = chunk.values
                        raw.append((complex(value.real[0], value.imaginary[0]),
                                    complex(value.real[1], value.imaginary[1])))
        data = run.data
    pooled = plan.method.analyze(plan, data, settings={})
    # Both contributions come from one receipt, so each has the bound of a
    # single acquisition of the pair.
    with Run(plan) as fresh:
        single = {p.pair: p for p in fresh.wait(timeout=5, poll_interval=0).pairs}
    (pair,) = [p for p in pooled.pairs if p.pair == (0, 1)]
    assert pair.weights == (.5, .5) and len(pair.contribution_ids) == 2 and len(raw) == 2
    assert complex(pair.hamiltonian.real, pair.hamiltonian.imag) == pytest.approx(
        (raw[0][0] + raw[1][0]) / 2, rel=0, abs=2.0**-52)
    assert complex(pair.overlap.real, pair.overlap.imag) == pytest.approx(
        (raw[0][1] + raw[1][1]) / 2, rel=0, abs=2.0**-52)
    # The pooling rounding term is positive for nonzero entries, so the
    # pooled bound exceeds the equal contributions' common bound.
    assert pair.overlap_bound > single[(0, 1)].overlap_bound
    assert pair.hamiltonian_bound > single[(0, 1)].hamiltonian_bound
    assert all(p.weights == (1.0,) for p in pooled.pairs if p.pair != (0, 1))
    v = np.column_stack(basis).astype(complex)
    v /= np.linalg.norm(v, axis=0)
    h0 = SparsePauliOp.from_list(terms).to_matrix()
    assert abs(complex(pair.overlap.real, pair.overlap.imag) - (v[:, 0].conj() @ v[:, 1])) <= pair.overlap_bound
    assert abs(complex(pair.hamiltonian.real, pair.hamiltonian.imag)
               - (v[:, 0].conj() @ h0 @ v[:, 1])) <= pair.hamiltonian_bound


def test_grouped_sampled_pencil_matches_the_dense_reference_within_its_standard_errors(tmp_path):
    """Grouped sampling: b**2 G settings, entries within six estimated standard errors of V†HV and V†V.

    Three normalized complex columns on three qubits and the six nonidentity
    terms ZII, IZI, ZZI, XII, IYI, XYI with identity coefficient 0.7 form
    two QWC groups, XYI and ZZI, so the pencil takes 3**2 * 2 = 18 settings
    instead of the 60 elementary Hadamard settings of per-term acquisition.
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
    included) and dyadic coefficients, so every binary64 moment of the
    reducer is exact and must equal the rational moment. The rational
    moments of the full table equal those of its exact marginal on the bits
    the statistics read. A diagonal setting must keep bit 0 zero, the count
    total must equal returned_shots, a one-shot table has unavailable
    variances, and a label on system qubit 63 reads the second key word.
    """
    from fractions import Fraction as F
    from nwqlib.algorithms.gcim.fixed_basis import (
        FixedGCIMReconstruction, PauliArrays, SampledGroup, reduce_group)

    n = 8
    rows = (("IIIZIIZI", .5), ("ZIIZIIII", -.75))  # system qubits 1, 4 and 4, 7
    rec = FixedGCIMReconstruction(
        terms=PauliArrays.from_rows(sorted(rows), n), basis_size=2, num_qubits=n,
        groups=(SampledGroup(basis="ZIIZIIZI", indices=(0, 1)),))
    member_rows = rec.nonidentity_terms
    rng = np.random.default_rng(3)
    counts = dict(enumerate(rng.multinomial(4096, np.full(512, 1 / 512)).tolist()))
    moment = reduce_group(_count_chunk(n + 1, counts), rec, ("group_0_1_real_0", 0, 1, "real", 0))

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
        reduce_group(_count_chunk(n + 1, {0b10: 3, 0b11: 1}), rec, diagonal)
    with pytest.raises(ValueError, match="returned_shots"):
        reduce_group(_count_chunk(n + 1, {0: 3}, returned=4), rec, diagonal)
    single = reduce_group(_count_chunk(n + 1, {0b101: 1}), rec, ("group_0_1_imag_0", 0, 1, "imag", 0))
    assert single.mean_variance is single.overlap_variance is single.mean_covariance is None

    wide = FixedGCIMReconstruction(
        terms=PauliArrays.from_rows((("Z" + "I" * 63, 1.0),), 64), basis_size=1, num_qubits=64,
        groups=(SampledGroup(basis="Z" + "I" * 63, indices=(0,)),))
    # System qubit 63 is classical bit 64, the first bit of the second word.
    table = {1 << 64: 3, 0: 1}
    assert reduce_group(_count_chunk(65, table), wide, diagonal).mean == -0.5


def test_repeated_group_acquisitions_pool_with_squared_population_weights():
    """pooled_statistics: weights N_a/sum N, variances and covariances with squared weights, raw negatives flagged.

    Two acquisitions of 100 and 300 shots of one off-diagonal setting pool
    with weights 1/4 and 3/4, variance 1/16 V_1 + 9/16 V_2 and covariance
    1/16 C_1 + 9/16 C_2 (fixed_basis.pooled_statistics). A negative raw
    variance is published unclipped and flagged.
    """
    from nwqlib.algorithms.gcim.fixed_basis import (
        FixedGCIMReconstruction, GroupMoment, PauliArrays, SampledGroup, pooled_statistics)

    rec = FixedGCIMReconstruction(
        terms=PauliArrays.from_rows((("I", .5), ("Z", 1.0)), 1), basis_size=2, num_qubits=1,
        groups=(SampledGroup(basis="Z", indices=(0,)),))

    def moment(name, pair, quad, shots, mean, variance, overlap=None, covariance=None, key="0"):
        supplies = pair[0] != pair[1]
        return GroupMoment(
            experiment=name, contribution_id="sha256:" + key * 64, pair=pair, quadrature=quad, group=0,
            shots=shots, label_means=(mean,), mean=mean, second=1.0, mean_variance=variance,
            overlap_mean=overlap if supplies else None, overlap_cross=0.0 if supplies else None,
            overlap_variance=.5 ** 4 if supplies else None, mean_covariance=covariance if supplies else None,
            variance_negative=variance < 0)

    moments = (
        moment("group_0_0_real_0", (0, 0), "real", 10, .5, -.25, key="1"),
        moment("group_0_1_real_0", (0, 1), "real", 100, .25, .5, .5, .125, key="2"),
        moment("group_0_1_real_0", (0, 1), "real", 300, .75, .25, .25, -.5, key="3"),
        moment("group_0_1_imag_0", (0, 1), "imag", 20, 0.0, .0, 0.0, 0.0, key="4"),
        moment("group_1_1_real_0", (1, 1), "real", 10, -.5, .125, key="5"),
    )
    estimates, missing, pairs, variances = pooled_statistics(rec, moments)
    assert missing == ()
    real = next(e for e in estimates if e.experiment == "group_0_1_real_0")
    assert real.weights == (.25, .75) and real.value == .25 * .25 + .75 * .75
    assert pairs[0, 1] == (complex(.25 * .25 + .75 * .75, 0.0), complex(.25 * .5 + .75 * .25, 0.0))
    row = next(v for v in variances if v.pair == (0, 1))
    assert row.h0[0] == .5 / 16 + 9 * .25 / 16
    assert row.overlap[0] == 10 * .5 ** 4 / 16
    assert row.covariance[0] == .125 / 16 - 9 * .5 / 16
    diagonal = next(v for v in variances if v.pair == (0, 0))
    assert diagonal.h0 == (-.25, 0.0) and diagonal.variance_negative
    assert estimates[0].chunk_mean_variances == (-.25,)


@pytest.mark.parametrize("groups, message", (
    ((("ZI", (0,)),), "exactly once"),
    ((("ZI", (0, 0)), ("IX", (1,))), "exactly once"),
    ((("ZX", (0, 1)), ("II", ())), "empty membership"),
    ((("ZXI", (0, 1)),), "wrong basis width"),
    ((("ZZ", (0, 1)),), "differs from its Pauli rows"),
    ((("ZX", (0, 1, 2)),), "not qubit-wise commuting"),
))
def test_group_partition_must_cover_each_row_once_in_one_qwc_basis(groups, message):
    from nwqlib.algorithms.gcim.fixed_basis import FixedGCIMReconstruction, PauliArrays, SampledGroup

    rows = ((("IX", .5), ("YI", .25), ("ZI", -1.0)) if "commuting" in message else (("IX", .5), ("ZI", -1.0)))
    with pytest.raises(ValueError, match=message):
        FixedGCIMReconstruction(
            terms=PauliArrays.from_rows(rows, 2), basis_size=2, num_qubits=2,
            groups=tuple(SampledGroup(basis=basis, indices=indices) for basis, indices in groups))


def test_group_statistics_are_refused_outside_the_grouped_route_and_admitted_before_decoding():
    """Exact Results cannot carry group statistics, and a grouped pass is admitted at its stated law.

    The pass law of fixed_basis.read_groups over the stored bin counts m_a
    and group sizes L_a of each chunk, with c = n + 1 classical bits and one
    key word, is
    B = sum_a 16 m_a + max_a(80 m_a + 8 L_a) + 8 sum_a(L_a + 16) + (n + 16)L + 512b**2
    bytes and W = sum_a[m_a(c + 28 + 8 L_a) + L_a c] + L(n + 1) + 32b**2 + 16b**3
    work. At B and W the pass runs; one byte or one unit less refuses it
    before any chunk is decoded.
    """
    from types import SimpleNamespace
    import nwqlib
    from nwqlib.algorithms.gcim import fixed_basis
    from nwqlib.operators import ingest_pauli

    exact = nwqlib.solve(plan_for(), progress=False)
    with pytest.raises(ValueError, match="belong only to a sampled Plan"):
        exact.revise(sampled_pair_variances=(fixed_basis.SampledPairVariance(
            pair=(0, 1), h0=(0., 0.), overlap=(0., 0.), covariance=(0., 0.), physical_h=(0., 0.),
            variance_negative=False),))._attach(exact.plan, exact.data)

    rng = np.random.default_rng(2)
    n, b = 3, 3
    basis = tuple(rng.normal(size=2**n) + 1j * rng.normal(size=2**n) for _ in range(b))
    rows = (("III", .7), ("ZII", .5), ("IZI", -.3), ("ZZI", .2), ("XII", .4), ("IYI", -.25), ("XYI", .1))
    plan = nwqlib.plan(Eigenproblem(A=ingest_pauli(rows, num_qubits=n)), method=FixedGCIM(basis=basis),
                       execution="quantum", shots=37, seed=3)
    result = nwqlib.solve(plan, progress=False)
    rec, c = plan.reconstruction, n + 1
    L = len(rec.nonidentity_coefficients)
    settings = {s[0]: s for s in fixed_basis.sampled_settings(b, rec.groups)}
    bins = [len(chunk.values) for chunk in result.data.observations.chunks]
    sizes = [len(rec.groups[settings[chunk.experiment][-1]].indices) for chunk in result.data.observations.chunks]
    need = (sum(16 * m for m in bins) + max(80 * m + 8 * g for m, g in zip(bins, sizes))
            + 8 * sum(g + 16 for g in sizes) + (n + 16) * L + 512 * b * b)
    work = (sum(m * (c + 28 + 8 * g) + g * c for m, g in zip(bins, sizes))
            + L * (n + 1) + 32 * b * b + 16 * b**3)

    class Limited:
        def __init__(self, max_bytes, max_analysis_work):
            self.method = SimpleNamespace(max_bytes=max_bytes, max_analysis_work=max_analysis_work)

        def __getattr__(self, name):
            return getattr(plan, name)

    assert fixed_basis.read_groups(Limited(need, work), result.data) == result.group_moments
    decoded = []
    original = fixed_basis.reduce_group
    fixed_basis.reduce_group = lambda *a, **k: decoded.append(1) or original(*a, **k)
    try:
        for limits, message in (((need - 1, work), "max_bytes"), ((need, work - 1), "max_analysis_work")):
            with pytest.raises(ValueError, match=message):
                fixed_basis.read_groups(Limited(*limits), result.data)
    finally:
        fixed_basis.reduce_group = original
    assert decoded == []


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

    # Two groups of three terms, 18 settings, at most 16 bins on four bits.
    work = 18 * (16 * (4 + 28 + 8 * 3) + 3 * 4) + 6 * 4 + 32 * 9 + 16 * 27
    assert work == 17088
    with monkeypatch.context() as guarded:
        guarded.setattr(fixed_basis, "_sampled_construction", unexpected_construction)
        # Both the one-group floor and the actual grouped work reject
        # before any sampled setting is constructed.
        for cap in (5000, work - 1):
            with pytest.raises(ValueError, match=rf"max_analysis_work={cap}\b"):
                select(cap)
    selected = select(work)
    assert len(selected.experiments) == 18
    assert selected.shots == 64
    assert fixed_basis.sampled_analysis_requirements(
        3, 3, selected.reconstruction.groups, 64
    ) == (13370, 17088)

    # The byte bound refuses first on a wider register: -Z^10 - 0.7 X^10 has
    # G = 2 groups of one term, b = 2, n = 10, c = 11, w = 1 and
    # m = min(4096, 2**11) = 2048 bins, so B_plan (sampled_analysis_requirements)
    # is 8*2*4*2*2048 + 80*2048 + 8*1 + 8*4*(2+16*2) + 26*2 + 512*4.
    n = 10
    wide = nwqlib.Eigenproblem(A=ingest_pauli((("Z" * n, -1.0), ("X" * n, -.7)), num_qubits=n))
    rng = np.random.default_rng(5)
    states = tuple(rng.normal(size=2**n) + 1j * rng.normal(size=2**n) for _ in range(2))
    size = 8 * 2 * 4 * 2 * 2048 + 80 * 2048 + 8 * 1 + 8 * 4 * (2 + 16 * 2) + 26 * 2 + 512 * 4
    assert size == 429180

    def select_wide(cap):
        return nwqlib.plan(wide, method=FixedGCIM(basis=states, max_bytes=cap),
                           execution="quantum", shots=4096, seed=1)

    with monkeypatch.context() as guarded:
        guarded.setattr(fixed_basis, "_sampled_construction", unexpected_construction)
        with pytest.raises(ValueError, match=rf"GCIM planned grouped analysis needs {size} data bytes"):
            select_wide(size - 1)
    assert len(select_wide(size).experiments) == 8


def test_analysis_work_limits_cover_sampled_and_classical_plans(monkeypatch):
    """The known solve/analysis floor refuses before grouping, with a legal larger plan."""
    import nwqlib
    from nwqlib.algorithms.gcim import fixed_basis
    from nwqlib.operators import ingest_pauli

    A = [[1, .2, 0, .1], [.2, -1, .3, 0], [0, .3, .5, .2], [.1, 0, .2, -.4]]
    basis = ([1, 0, 0, 0], [0, 1, 1j, 0], [1, 1, 1, 1])
    problem = nwqlib.Eigenproblem(A=A)
    with monkeypatch.context() as guarded:
        guarded.setattr(fixed_basis, "sampled_groups", lambda *a, **k: pytest.fail("grouped before work admission"))
        with pytest.raises(ValueError, match=r"max_analysis_work=10\b"):
            nwqlib.plan(problem, method=FixedGCIM(basis=basis, max_analysis_work=10), shots=16, seed=1)
    selected = nwqlib.plan(problem, method=FixedGCIM(basis=basis), shots=16, seed=1)
    assert selected.experiments
    # Without shots nothing is grouped and the solve is the known requirement.
    problem = nwqlib.Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1))
    method = FixedGCIM(basis=([1, 0], [0, 1], [1, 1]), max_analysis_work=100)
    with pytest.raises(ValueError, match=r"max_analysis_work=100\b"):
        nwqlib.plan(problem, method=method, execution="classical", seed=1)


def test_default_analysis_work_admits_a_fourteen_qubit_grouped_plan():
    """The default max_analysis_work admits a 14-qubit grouped plan at 8,192 shots.

    Its planning bound (sampled_analysis_requirements) is 237,416,880 work units,
    above 10**8 (docs/ENGINEERING_CONSTANTS.md, row "Sampled FixedGCIM analysis
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
    with pytest.raises(ValueError, match=r"needs at most 237416880 work units.*max_analysis_work=100000000\."):
        select(max_analysis_work=10**8)
