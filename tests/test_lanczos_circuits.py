"""Signed readout/selected walk witnesses; native execution has one tiny test."""

from collections import Counter
import numpy as np
import pytest

from test_lanczos import selected
from nwqlib.algorithms.lanczos.readout import decode_histogram
from nwqlib.execution import Histogram
from nwqlib._prepared_execution import Run
from nwqlib.ir import BlockCall, MeasurementBatch, Repeat, Sequence
from nwqlib.ir.validation import _Admission


def test_shared_walk_metadata_counts_and_padding_readout():
    plan = selected(terms=(("IX", -1.0), ("IY", 1.0), ("IZ", 1.0)), m=3)
    reconstruction = plan.reconstruction
    assert reconstruction.num_index_qubits == 2
    assert reconstruction.moment_indices == (1, 2, 3, 4, 5)
    assert len(plan.construction.program.definitions) < 25
    occurrences = Counter()
    (experiment,) = plan.experiments
    construction = plan.resolve(experiment.name).selected_construction(plan)
    admission = _Admission(construction.program)
    admission.check().require_ready()
    values = admission.expressions(admission.binding_map(construction.program.bindings))
    signatures = {item.id: item.node.signature for item in construction.program.definitions
                  if isinstance(item.node, BlockCall)}

    def visit(name):
        node = admission.nodes[name]
        if isinstance(node, BlockCall):
            occurrences[node.signature] += 1
        elif isinstance(node, Sequence):
            for child in node.children:
                visit(child)
        elif isinstance(node, Repeat):
            for _ in range(admission.integer(node.count, values, "walks")):
                visit(node.body)
        elif isinstance(node, MeasurementBatch):
            visit(node.body)

    visit(construction.program.root)
    points = experiment.observation.positions
    for index, point in enumerate(points):
        occurrences[signatures[point.view.tail]] += 1
        if index + 1 < len(points):
            occurrences[signatures[point.view.inverse]] += 1
    # One reference and one coefficient PREP, then floor(5/2) = 2 = m-1 walks
    # (SELECT, PREP^dagger, zero reflection, PREP each). Points: degree 1
    # after 0 walks, degrees 2 and 3 after 1, degrees 4 and 5 after 2. One
    # forward tail per point (PREP^dagger for 2 and 4, the odd readout for
    # 1, 3 and 5) and one inverse per point except the last (odd readout
    # adjoint for 1 and 3, PREP for 2 and 4).
    assert occurrences == {
        "reference": 1,
        "select": 2,
        "coefficient_prep": 1 + 2 + 2,
        "coefficient_inverse": 2 + 2,
        "positive_zero": 2,
        "signed_readout": 3,
        "signed_readout_inverse": 2,
    }
    odd, even = reconstruction.settings[:2]
    # Index label 3 is padding. Its frequency stays in the denominator and
    # contributes zero first AND second moment; no success postselection.
    readout = plan._native["readout"]
    statistics = decode_histogram(readout, odd, {0: 2, 4: 1, 1: 3, 5: 2, 3: 2}, counts=True)
    assert statistics.mean == 0.0
    assert statistics.second_moment == 0.8 and statistics.shots == 10
    assert statistics.variance == pytest.approx(0.8 / 9, rel=0, abs=1e-16)
    reflection = decode_histogram(
        readout, even, Histogram(even.histogram_width, [0, 1], np.array([1, 4])), counts=True
    )
    assert reflection.mean == -0.6 and reflection.second_moment == 1.0
    # Exact probabilities keep denominator one: a mapping of total mass 0.5
    # decodes to its signed sum, -0.125 (address 0, IX, sign -1, system bit 0
    # reads 0) + 0.25 (address 0, system bit 0 reads 1) + 0 (padding address
    # 3), not to that sum divided by the mass.
    exact = decode_histogram(readout, odd, {0: 0.125, 4: 0.25, 3: 0.125}, counts=False)
    assert (exact.mean, exact.second_moment, exact.shots) == (0.125, 0.375, None)


def test_readout_wider_than_64_bits_decodes_label_parity_and_padding():
    """Settings wider than 64 bits decode Python-integer outcomes with two-word masks.

    The expected value of each outcome is recomputed from the Pauli labels:
    sign(c_i) times the parity of the system bits where label i acts, zero
    for the padding address and, for reflection, +1 only at index zero.
    """
    from nwqlib.algorithms.lanczos import Lanczos
    from nwqlib.core.planning import RandomStreams
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.records import Eigenproblem

    n = 66

    def label(ops):
        chars = ["I"] * n
        for qubit, pauli in ops.items():
            chars[n - 1 - qubit] = pauli
        return "".join(chars)

    terms = (
        (label({65: "Z", 0: "X"}), -0.5),
        (label({64: "Y"}), 0.25),
        (label({1: "Z", 65: "X"}), 1.0),
        (label({}), 2.0),
    )
    problem = Eigenproblem(A=ingest_pauli(terms, num_qubits=n))
    plan = Lanczos(krylov_dimension=2).plan(
        problem, output=problem.default_output(), execution="quantum", shots=10,
        rng=RandomStreams(1),
    )
    readout = plan._native["readout"]
    odd, even = plan.reconstruction.settings[:2]
    assert (odd.readout, even.readout) == ("select", "reflection")
    assert odd.histogram_width == even.histogram_width == 68
    assert readout.masks.shape == (3, 2)

    def expected(bits):
        index, system = bits & 3, bits >> 2
        if index >= 3:
            return 0
        text, coefficient = terms[index]
        support = sum(1 << qubit for qubit in range(n) if text[n - 1 - qubit] != "I")
        return (1 if coefficient > 0 else -1) * (-1) ** bin(system & support).count("1")

    outcomes = {1 << 67: 3, 1 | 1 << 66: 2, 2 | 3 << 66 | 1 << 3: 4, 3 | 1 << 67: 1, 2 | 1 << 3: 6}
    statistics = decode_histogram(readout, odd, outcomes, counts=True)
    assert statistics.shots == 16
    assert statistics.mean == sum(expected(b) * w for b, w in outcomes.items()) / 16 == -1 / 16
    assert statistics.second_moment == sum(expected(b) ** 2 * w for b, w in outcomes.items()) / 16
    reflection = decode_histogram(readout, even, {1 << 67: 2, 1: 3}, counts=True)
    assert reflection.mean == -0.2 and reflection.second_moment == 1.0


@pytest.mark.parametrize("terms", (
    (("I", 1.0), ("X", -1.0), ("Z", 0.5)),
    (("I", 1.0), ("Y", -1.0), ("Z", 0.5)),
))
def test_native_signed_centered_two_qubit_public_chain(monkeypatch, terms):
    """One preparation and one simulation read all three moments; no reference operator/eigensolve.

    The exact trajectory reads mu_1 on the initial walk state and mu_2, mu_3
    after one walk step, through reversible views that restore the
    continuation state, and the odd view's inverse is the native table of
    the forward tail's adjoint matrices. The analytic moments
    [1, 1/3, 1/9, -7/27] are the independent reference for both H = I - X + Z/2
    and H = I - Y + Z/2 = S (I - X + Z/2) S^dagger, since S|0> = |0>; 3e-14
    is this test's regression threshold, not a derived bound. The Y case's
    readout table holds the complex Y-basis change Had S^dagger (Hadamard
    times S^dagger) and the identity, so the bitwise table
    check rejects a transposed or a conjugated table in place of the
    adjoint. Shared native definitions must be reused by the actual
    preparation chain; separate metadata/kernel checks do not exercise that
    native population.
    """
    from nwqlib.backends import inspection, qiskit_aer as aer
    from nwqlib.algorithms.lanczos import numerical

    plan = selected(terms=terms)
    run = Run(plan)
    calls = Counter()
    for name in ("_prepare_aer_execution", "_submit_aer_execution"):
        original = getattr(aer, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(aer, name, counted)
    original_solve = numerical._solve_projected_pencil

    def solve(*args, **kwargs):
        calls["projected_solve"] += 1
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(numerical, "_solve_projected_pencil", solve)

    def forbidden(*args, **kwargs):
        pytest.fail("native scalar chain performed classical recurrence or inventory replay")

    monkeypatch.setattr(numerical, "chebyshev_moments", forbidden)
    monkeypatch.setattr(inspection, "inspect_circuit_resources", forbidden)
    result = run.wait(timeout=5, poll_interval=0)
    np.testing.assert_allclose(result.moments, [1, 1 / 3, 1 / 9, -7 / 27], rtol=0, atol=3e-14)
    assert result.eigenvalue == pytest.approx(1 - np.sqrt(5) / 2, rel=0, abs=2e-13)
    assert len(run.data.trace.events) == run.data.trace.preparations == 1
    assert all(
        c.trajectories == 1 and c.returned_shots is None for c in run.data.observations.chunks
    )
    assert calls == {"_prepare_aer_execution": 1, "_submit_aer_execution": 1, "projected_solve": 1}
    # The odd view's inverse is the multiplexer of the forward table's adjoint
    # matrices (qiskit_compat.inverse_realized_gate): as many native
    # instructions as its forward tail, not an inverted UCG decomposition.
    from qiskit_aer import AerSimulator
    from nwqlib.blocks.lowering import lower_definition

    experiment, = plan.experiments
    _, construction = plan.resolve(experiment.name)._selected_construction(plan)
    identities = {record.content_id for record in construction.selections}
    blocks = tuple(b for b in plan.blocks if b.record.content_id in identities)
    decompose = aer._AerDecompose(AerSimulator(method="statevector").target)
    forward, inverse = (
        aer._prepare_execution_circuit(lower_definition(construction, name, blocks=blocks).circuit,
                                       decompose=decompose, add_save_statevector=False)[0]
        for name in ("signed_readout", "signed_readout_inverse")
    )
    assert forward.count_ops() == inverse.count_ops() == {"multiplexer": 1}
    for ahead, behind in zip(forward.data, reversed(inverse.data), strict=True):
        assert [forward.find_bit(q).index for q in ahead.qubits] == [inverse.find_bit(q).index for q in behind.qubits]
        for matrix, adjoint in zip(ahead.operation.params, behind.operation.params, strict=True):
            assert np.array_equal(adjoint, matrix.conj().T)
    # The chain's readout target has an index control, so the executed
    # trajectory, inverse included, contains no matrix unitary.
    receipt, = run.data.receipts
    assert "unitary" not in receipt.probability_window_exclusions
