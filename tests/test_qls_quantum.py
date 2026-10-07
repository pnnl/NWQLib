"""Actual QLS selected bodies, native original-coordinate readout and storage."""

import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import nwqlib
from nwqlib import (
    LinearSystem,
    Solution,
    StateVector,
    NormSquared,
    QuadraticForm,
    NormalizedExpectation,
    Samples,
)
from nwqlib.algorithms.qls import QLS, QLSVerification
from nwqlib.algorithms.qls import quantum, method as owner
from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan
from nwqlib.blocks import select_block_encoding
from nwqlib.blocks.selection import SelectedBlock
from nwqlib.execution import ObservationView
from nwqlib.operators import ingest_dense
from nwqlib.subroutines.block_encoding import BlockEncoding
from test_qls_primary import MATRIX, RHS, forbidden, aligned, facts


def choose(*, A=MATRIX, b=RHS, solver="qsvt_inverse", output=None, shots=None, **fields):
    if output is None:
        output = (
            Solution()
            if solver == "qsvt_inverse"
            else StateVector(normalization="unit", global_phase="modulo_global_phase")
        )
    selected = nwqlib.plan(
        LinearSystem(A=A, b=b),
        method=QLS(
            solver=solver,
            encoded_solution_norm_estimate=None if solver == "qsvt_inverse" else 1.5,
            **fields,
        ),
        output=output,
        shots=shots,
        seed=7,
    )
    assert selected.reconstruction.width <= 13
    return selected


def compact_key(rec, setting, ones):
    """The count key of a setting's measured register with the physical bits ``ones`` set.

    The setting measures its selectors and coordinates into the prefix
    ``quantum._counts_bits`` names, in the shared layout of
    ``quantum._counts_width`` bits.
    """
    observed = quantum._counts_bits(rec, setting)
    return format(sum(1 << observed.index(bit) for bit in ones), f"0{quantum._counts_width(rec)}b")


def selector_ones(rec):
    return [bit for bit, value in rec.success + rec.conditions if value]


def changed_data(result, chunk):
    original = next(c for c in result.data.observations.chunks if c.experiment == chunk.experiment)
    chunks = tuple(
        chunk if c.content_id == original.content_id else c for c in result.data.observations.chunks
    )
    trace = result.data.trace.revise(
        events=tuple(
            e.revise(observation_id=chunk.content_id, returned_shots=chunk.returned_shots)
            if e.observation_id == original.content_id
            else e
            for e in result.data.trace.events
        )
    )
    return replace(result.data, observations=ObservationView(chunks=chunks), trace=trace)


def test_complex_three_coordinate_native_uses_original_svd_once(monkeypatch):
    original = np.linalg.svd
    seen = []

    def svd(matrix, *args, **kwargs):
        if matrix.shape == (3, 3) and np.array_equal(matrix, MATRIX/2):
            seen.append("original")
        elif matrix.shape == (4, 4) and any(np.array_equal(matrix[:3, :3], value)
                for value in (MATRIX, MATRIX/2)):
            raise AssertionError("padded SVD manufactured scale")
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", svd)
    selected = choose()
    assert selected.reconstruction.width == 6
    result = nwqlib.solve(selected)
    expected = np.linalg.solve(MATRIX, RHS)
    assert np.linalg.norm(result.x - expected) / np.linalg.norm(expected) < 0.01
    assert seen == ["original"]
    base = selected._native["base"]
    assert base.svd is selected._native["svd"]
    base.validate()
    wrong = replace(base, original=ingest_dense(MATRIX + np.eye(3)))
    with pytest.raises(ValueError, match="original input binding"):
        wrong.validate()
    wrong_source = replace(base.selected._payload, source=np.eye(4))
    wrong = replace(
        base,
        selected=SelectedBlock.bind(
            base.selected.record, payload=wrong_source, constructor=base.selected._constructor
        ),
    )
    with pytest.raises(ValueError, match="exact selected input"):
        wrong.validate()


@pytest.mark.parametrize("solver", ["shortcut_native_svp", "shortcut_dilation"])
def test_original_three_coordinate_shortcut_matches_padded_native_model(solver):
    selected = choose(solver=solver)
    assert selected.reconstruction.width in (8, 9)
    result = nwqlib.solve(selected)
    classical = nwqlib.solve(
        selected.problem,
        method=selected.method,
        output=selected.output,
        execution="classical",
        seed=7,
    )
    np.testing.assert_allclose(
        aligned(result.value, classical.value), classical.value, rtol=1e-11, atol=1e-11
    )
    assert result.algorithm_success_mass == pytest.approx(
        classical.algorithm_success_mass, abs=2e-11
    )
    assert result.value.shape == (3,) and result.physical_scale is None


@pytest.mark.parametrize("family", ["banded", "multiplexed_pauli", "dense_dilation", "compact_pauli"])
def test_planned_encoding_archive_keeps_constructor_phases_and_frames(
    tmp_path, monkeypatch, family
):
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    if family == "compact_pauli":
        # 2 I - X keeps Pauli access; its decomposition is the saved operator's
        # own term table, so the archive names problem.A instead of copying it.
        from nwqlib.operators import ingest_pauli

        selected = choose(A=ingest_pauli((("I", 2.0), ("X", -1.0)), num_qubits=1), b=[1.0, 1j], kappa=3.0)
    else:
        if family == "multiplexed_pauli":
            # I, X, Y and Z terms, so the saved packed table exercises every label letter.
            A = np.array([[2.0, -1.0 + 0.5j], [-1.0 - 0.5j, 1.5]])
        selected = choose(A=A, b=[1.0, 1j], block_encoding_implementation=family)
    saved = save_plan(selected, ArchiveFiles(tmp_path, 4_000_000))
    decomposition = saved["selected"]["encoding"].get("decomposition")
    assert (decomposition is not None and decomposition.get("reference") == "problem.A") == (
        family == "compact_pauli")
    monkeypatch.setattr(QLS, "plan", forbidden)
    monkeypatch.setattr(owner, "select_polynomial", forbidden)
    monkeypatch.setattr("nwqlib.subroutines.block_encoding.core.plan_block_encoding", forbidden)
    monkeypatch.setattr("nwqlib.subroutines.qsp.phases.solve_symmetric_qsp_phases", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    loaded = load_plan(saved, ArchiveFiles(tmp_path, 4_000_000))
    assert loaded.content_id == selected.content_id
    assert loaded._native["encoding"].record == selected._native["encoding"].record
    # The saved Pauli table decodes to the selected labels and coefficients.
    assert loaded._native["encoding"]._payload.decomposition == selected._native["encoding"]._payload.decomposition
    assert loaded.reconstruction.phase_solution == selected.reconstruction.phase_solution
    result = nwqlib.solve(loaded)
    expected = np.linalg.solve(A, [1.0, 1j])
    np.testing.assert_allclose(result.x, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("spectral_rounding", ["native", "upper_up", "both_up"])
def test_supplied_complex_global_phase_survives_control_inverse_and_archive(
    tmp_path, monkeypatch, spectral_rounding
):
    """A supplied unitary has inverse A-dagger, exposing lost global phase through control,
    adjoint and archive paths.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    circuit = QuantumCircuit(1)
    circuit.ry(0.4, 0)
    circuit.global_phase = 0.37
    matrix = Operator(circuit).data
    operator = ingest_dense(matrix)
    if spectral_rounding != "native":
        from nwqlib.algorithms.qls import numerical
        # A unitary has both singular values exactly one. LAPACK platforms can
        # return an endpoint one ulp above one; force that independent failure
        # mode at the numerical owner without changing the circuit or alpha.
        original = numerical._extreme_singular_values

        def rounded(*args, **kwargs):
            _, _, method = original(*args, **kwargs)
            high = float(np.nextafter(1., np.inf))
            return (high if spectral_rounding == "both_up" else 1.), high, method

        monkeypatch.setattr(numerical, "_extreme_singular_values", rounded)
    encoding = select_block_encoding(
        "supplied",
        BlockEncoding(
            circuit=circuit,
            alpha=1.0,
            num_ancillas=0,
            system_qubits=1,
            error_bound=0.0,
            implementation="supplied",
            metadata={},
        ),
        operator=operator,
    )
    selected = choose(A=operator, b=[1.0, 1j], encoding=encoding)
    assert selected.reconstruction.width == 4
    assert selected.reconstruction.alpha == 1.
    if spectral_rounding != "native":
        assert selected.reconstruction.sigma_max == np.nextafter(1., np.inf)
        assert selected.reconstruction.kappa_be == 1.
    result = nwqlib.solve(selected)
    expected = matrix.conj().T @ np.array([1.0, 1j])
    np.testing.assert_allclose(result.x, expected, rtol=0.01, atol=0.01)
    result.save(tmp_path / "result")
    monkeypatch.setattr(QLS, "plan", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr("nwqlib.subroutines.qsp.phases.solve_symmetric_qsp_phases", forbidden)
    loaded = nwqlib.load_result(tmp_path / "result")
    np.testing.assert_array_equal(loaded.x, result.x)
    assert loaded.plan.method.encoding is loaded.plan._native["encoding"]
    assert loaded.plan.method.encoding._payload.circuit == encoding._payload.circuit


def test_supplied_encodings_stay_distinct_and_the_default_quantum_plan_id_is_reproducible():
    """Two supplied circuits with equal metadata get distinct selected identities, while the
    default quantum QLS Plan, whose encoding the Method selects, gets one identity per input.
    """
    from qiskit import QuantumCircuit

    operator = ingest_dense(np.eye(2))

    def supplied(angle):
        circuit = QuantumCircuit(1)
        circuit.ry(angle, 0)
        return select_block_encoding("supplied", BlockEncoding(
            circuit=circuit, alpha=1.0, num_ancillas=0, system_qubits=1, error_bound=0.0,
            implementation="supplied", metadata={}), operator=operator)

    def default_plan():
        return nwqlib.plan(LinearSystem(A=MATRIX, b=RHS), method=QLS(), output=Solution(), seed=7)

    assert supplied(0.4).record.content_id != supplied(0.7).record.content_id
    assert default_plan().content_id == default_plan().content_id


@pytest.mark.parametrize(
    "output",
    [
        NormSquared(),
        QuadraticForm(observable=np.diag([2.0, -1.0, 0.5])),
        NormalizedExpectation(observable=np.diag([2.0, -1.0, 0.5])),
    ],
)
def test_padded_scalar_readout_preserves_original_projection(output, monkeypatch):
    """An exact scalar is reduced from one simulation of the selected body, bound to its projectors.

    The padded d = 3 observable has three non-identity P O P terms, which a
    per-setting readout would simulate once each, plus a mass setting.
    """
    import json
    from nwqlib.backends import qiskit_aer
    from nwqlib.ir import Measure, MeasurementBatch

    selected = choose(output=output)
    simulations = []
    submit = qiskit_aer._submit_aer_execution
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution",
                        lambda prepared, **kw: simulations.append(prepared) or submit(prepared, **kw))
    result = nwqlib.solve(selected)
    classical = nwqlib.solve(
        selected.problem, method=selected.method, output=output, execution="classical", seed=7
    )
    assert result.value == pytest.approx(classical.value, rel=2e-11, abs=2e-11)
    rec = selected.reconstruction
    assert len(simulations) == 1 and [e.name for e in selected.experiments] == ["projected_moments"]
    (point,) = selected.experiments[0].observation.positions
    parameters = json.loads(point.parameters)
    assert (point.kind, point.reducer, point.position) == ("reduction", "projected_moments", None)
    assert parameters["coordinates"] == list(rec.coordinates) and parameters["dimension"] == 3
    assert parameters["success"] == [list(item) for item in rec.success]
    assert parameters["conditions"] == [list(item) for item in rec.conditions]
    if not isinstance(output, NormSquared):
        assert sum(set(label) != {"I"} for label, _ in parameters["terms"]) == 3
    assert not any(isinstance(item.node, (Measure, MeasurementBatch))
                   for item in selected.construction.program.definitions)
    assert result.mass_contribution_id == result.reduction.contribution_id == result.contribution_ids[0]


@pytest.mark.parametrize(("observable", "mass_setting", "expected"), [
    (np.diag([2.0, -1.0, 0.5]), "group_0", 2.0),
    (np.diag([1.0, 0.0, 1.0]), "group_0", 1.0),
    (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "physical_mass", 1.0),
])
def test_padded_counts_measure_every_coordinate_and_use_actual_mass_population(
    monkeypatch, observable, mass_setting, expected
):
    """A padded normalized output divides by the physical-prefix mass of an unrotated acquisition.

    A group whose basis has no X or Y measures every coordinate unrotated and
    supplies that mass, also when its Pauli support is one coordinate, as for
    ``diag(1, 0, 1)``, whose zero extension is 0.5 II + 0.5 IZ. Otherwise the unrotated ``physical_mass`` setting does.
    The mass setting's register holds every valued selector and coordinate,
    and the shared classical layout is that register's width.
    """
    from nwqlib.backends import qiskit_aer

    selected = choose(output=NormalizedExpectation(observable=observable), shots=12)
    names = [experiment.name for experiment in selected.experiments]
    assert names == (["group_0"] if mass_setting == "group_0" else ["group_0", "physical_mass"])
    rec = selected.reconstruction
    mass = next(setting for setting in rec.settings if setting.name == mass_setting)
    assert quantum._counts_bits(rec, mass) == tuple(bit for bit, _ in rec.success + rec.conditions) + rec.coordinates
    assert quantum._counts_width(rec) == len(quantum._counts_bits(rec, mass))
    calls = []

    def acquire(prepared, **kwargs):
        assert prepared.circuit.num_clbits == quantum._counts_width(rec)
        returned = 5 + len(calls)
        calls.append(returned)
        setting = rec.settings[len(calls) - 1]
        return SimpleNamespace(
            raw_output={"counts": {compact_key(rec, setting, selector_ones(rec)): returned}},
            metadata={"native_job_id": str(returned)},
        )

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", acquire)
    result = nwqlib.solve(selected)
    mass = next(c for c in result.data.observations.chunks if c.experiment == mass_setting)
    assert result.returned_shots == mass.returned_shots == calls[-1]
    assert result.mass_contribution_id == mass.content_id
    assert result.algorithm_success_mass == result.physical_slice_mass == 1.0
    assert result.value == pytest.approx(expected, rel=2e-14, abs=0)
    receipt = result.verify(checks=QLSVerification(comparisons=("inverse_success",)))[0]
    predicted = facts(receipt)["reference.inverse_success.predicted_mass"]
    assert facts(receipt)["reference.inverse_success.shot_allowance"] == pytest.approx(
        4 * np.sqrt(predicted * (1 - predicted) / calls[-1]),
        rel=2e-14, abs=0,
    )


def test_padded_counts_publish_no_physical_mass_from_a_rotated_group(monkeypatch):
    """A padded quadratic form measured only in a rotated basis has no physical-prefix mass or norm."""
    from nwqlib.backends import qiskit_aer

    observable = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    selected = choose(output=QuadraticForm(observable=observable), shots=12)
    rec = selected.reconstruction
    assert [s.label for s in rec.settings] == ["ZX"] and rec.groups == (("IX", "ZX"),)
    key = compact_key(rec, rec.settings[0], selector_ones(rec))
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", lambda prepared, **kw: SimpleNamespace(
        raw_output={"counts": {key: 12}}, metadata={"native_job_id": "rotated"}))
    result = nwqlib.solve(selected)
    assert result.physical_slice_mass is None and result.norm_squared is None
    assert result.algorithm_success_mass == 1.0 and result.physical_selected_shots is None
    # Both parities are +1 on the all-zero outcome, so the numerator is 0.5 + 0.5 before recovery.
    assert result.value == pytest.approx(rec.recovery.as_float() ** 2, rel=1e-14, abs=0)


def test_padded_normalized_counts_take_the_prefix_mass_from_an_unrotated_later_group(monkeypatch):
    """With a rotated first group, the unrotated second group supplies the physical-prefix mass.

    The zero-extended observable is 0.5 (IZ + ZZ + IX + ZX). Each group sees
    r selected shots at original coordinate 0, three at original coordinate
    1 and two at the dummy coordinate 3. Every label of either group has
    parity +1 at coordinate 0 and -1 at coordinate 1, and the two labels of
    each group have opposite parities at coordinate 3, so each unconditional
    group mean is (r - 3)/(r + 5), and the physical-prefix fraction of the
    unrotated group is (r1 + 3)/(r1 + 5). Coordinate 1 is decoded from the
    measured positions of the compact register, not from the physical wires.
    """
    from nwqlib.backends import qiskit_aer

    observable = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    selected = choose(output=NormalizedExpectation(observable=observable), shots=12)
    rec = selected.reconstruction
    assert [s.label for s in rec.settings] == ["ZX", "ZZ"] and rec.groups == (("IX", "ZX"), ("IZ", "ZZ"))
    calls = []

    def acquire(prepared, **kwargs):
        returned = 5 + len(calls)
        setting = rec.settings[len(calls)]
        calls.append(returned)
        counts = {compact_key(rec, setting, selector_ones(rec)): returned,
                  compact_key(rec, setting, selector_ones(rec) + [rec.coordinates[0]]): 3,
                  compact_key(rec, setting, selector_ones(rec) + list(rec.coordinates)): 2}
        return SimpleNamespace(raw_output={"counts": counts}, metadata={"native_job_id": str(returned)})

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", acquire)
    result = nwqlib.solve(selected)
    r0, r1 = calls
    mass = next(c for c in result.data.observations.chunks if c.experiment == "group_1")
    assert result.mass_contribution_id == mass.content_id and result.returned_shots == r1 + 5
    assert result.algorithm_success_mass == 1.0 and result.physical_slice_mass == (r1 + 3) / (r1 + 5)
    assert result.value == pytest.approx(((r0 - 3) / (r0 + 5) + (r1 - 3) / (r1 + 5)) / ((r1 + 3) / (r1 + 5)),
                                         rel=1e-14, abs=0)


def test_sampled_plan_with_a_measured_register_above_64_bits_is_refused_before_the_phase_solve(monkeypatch):
    """Counts decode 64-bit outcome indices of the shared layout, sized by the largest measured register.

    A selected setting that measures 65 bits is refused before QSP work,
    whatever the circuit width.
    """
    counts_bits = quantum._counts_bits
    monkeypatch.setattr(quantum, "_counts_bits", lambda rec, setting: counts_bits(rec, setting) + tuple(
        range(rec.width, rec.width + 65 - len(counts_bits(rec, setting)))))
    monkeypatch.setattr("nwqlib.subroutines.qsp.phases.solve_symmetric_qsp_phases", forbidden)
    with pytest.raises(ValueError, match="needs 65 classical bits, more than its 64-bit count decoder"):
        choose(output=Samples(), shots=12)


def test_sampled_group_measures_its_support_and_valued_selectors_into_a_compact_register():
    """A d = 64 IIIIIZ group measures one coordinate and its six selectors, not all 12 wires.

    The count table then has at most 2**7 outcomes. On the qualified seeded
    stack (Python 3.12.14, Qiskit 2.5.2, Aer 0.17.2, seed 5, 100,000 shots)
    it stores 31 entries, where measuring every wire stored 449.
    """
    from nwqlib.operators import ingest_pauli

    selected = nwqlib.plan(
        LinearSystem(A=np.diag(np.linspace(1.0, 2.0, 64)), b=np.random.default_rng(3).normal(size=64)),
        method=QLS(), output=NormalizedExpectation(observable=ingest_pauli((("IIIIIZ", 1.0),), num_qubits=6)),
        shots=100_000, seed=5)
    assert selected.reconstruction.width == 12
    result = nwqlib.solve(selected)
    (chunk,) = result.data.observations.chunks
    assert chunk.histogram().width == 7 and chunk.histogram().entries <= 2**7
    assert chunk.histogram().entries == 31  # Qualified seeded fixture, 449 when every wire is measured.


def test_sampled_register_position_reads_the_wire_its_layout_names():
    """Each position of a compact count register holds the physical wire ``quantum._counts_bits`` names.

    With diagonal A and b = e_2 the solution is a multiple of e_2, so every
    selected shot has coordinate bit 1 set and the ``ZI`` parity is -1 on each
    of them. The ``ZI`` group measures its selectors and coordinate bit 1
    only, a register that is not a prefix of the physical wires, so a
    position that measured another wire would publish a different value.
    """
    from nwqlib.operators import ingest_pauli

    selected = choose(A=np.diag([1.0, 1.5, 2.0, 2.5]), b=[0.0, 0.0, 1.0, 0.0], shots=200,
                      output=NormalizedExpectation(observable=ingest_pauli((("ZI", 1.0),), num_qubits=2)))
    rec = selected.reconstruction
    observed = quantum._counts_bits(rec, rec.settings[0])
    assert observed != tuple(range(len(observed))) and observed[-1] == rec.coordinates[1]
    result = nwqlib.solve(selected)
    assert result.groups[0].selected_shots > 0
    assert result.value == -1.0


def test_sampled_program_with_many_distinct_measured_registers_plans_at_the_default_admission_ceiling():
    """Admission work grows with the number of distinct measured registers, and the default admits such a Program.

    160 distinct random Pauli labels on 8 qubits form 108 groups with 27
    distinct measured registers. Their Program needs 130,560 admission steps,
    above the shared ``AdmissionLimits`` default of 100,000, and the QLS
    default admits it.
    """
    from nwqlib.operators import ingest_pauli
    from nwqlib.ir import AdmissionLimits

    rng = np.random.default_rng(11)
    labels = set()
    while len(labels) < 160:
        label = "".join(rng.choice(list("IXYZ"), size=8))
        if set(label) != {"I"}:
            labels.add(label)
    observable = ingest_pauli(tuple((label, float(rng.normal())) for label in sorted(labels)), num_qubits=8)
    selected = nwqlib.plan(
        LinearSystem(A=np.diag(np.linspace(1.0, 2.0, 256)), b=np.random.default_rng(3).normal(size=256)),
        method=QLS(), output=NormalizedExpectation(observable=observable), shots=100, seed=5)
    rec = selected.reconstruction
    assert len({quantum._counts_bits(rec, setting) for setting in rec.settings}) > 1
    readiness = selected.construction.program.check_readiness()
    assert readiness.expression_evaluations + readiness.lifecycle_steps > AdmissionLimits().max_steps


def test_compact_count_table_keeps_the_full_tables_selected_weighted_sum():
    """Marginalizing a nine-wire count table onto a group's measured register keeps its sufficient statistics.

    Every parity, success and condition predicate depends only on the
    measured bits, so the full table of all 512 outcomes and its marginal
    give the same rational weighted sum and selected count. The decoder
    remaps the physical selectors into the measured positions, with
    noncontiguous bits and nonzero selector values. With the parity supports
    mapped into the same positions, its binary64 mean is the rational
    quotient converted once, within 8u for this bounded weighted mean with
    dyadic coefficients.
    """
    from fractions import Fraction
    from nwqlib._quantum_readout import ReadoutSetting, weighted_group_moments
    from nwqlib.core import Source
    from nwqlib.core.planning import ObservationSpec
    from nwqlib.execution import ObservationChunk, RegisterMap

    rec = SimpleNamespace(width=9, coordinates=(2, 5, 7, 8), success=((0, 1), (3, 0), (4, 0), (6, 1)),
                          conditions=((1, 1),), original_dimension=16,
                          settings=(ReadoutSetting(name="group_0", label="IXZY"),))
    observed = quantum._counts_bits(rec, rec.settings[0])
    full = {i: 1 + i % 7 for i in range(512)}
    marginal = {}
    for x, n in full.items():
        y = sum(((x >> bit) & 1) << p for p, bit in enumerate(observed))
        marginal[y] = marginal.get(y, 0) + n
    identity = "sha256:" + "1" * 64
    chunk = ObservationChunk.from_histogram(
        {format(x, f"0{len(observed)}b"): n for x, n in marginal.items()},
        run_id="r", plan_id=identity, realization_id=identity, prepared_id=identity,
        experiment="group_0", setting="group_0", bindings=(), quantum_layout=(),
        classical_layout=(RegisterMap(name="c", bits=tuple(range(len(observed)))),),
        attempt="a", job="j", chunk="c", observation=ObservationSpec(kind="counts", shots=sum(full.values())),
        population="unconditional", returned_shots=sum(full.values()), trajectories=None,
        source=Source(name="supplied", version="1", domain="nine-bit count identity",
                      reference="integer enumeration"))
    bits, counts, success, selected, physical = quantum._selection(
        rec, chunk, rec.settings[0], classical_width=quantum._counts_width(rec))
    positions = {bit: p for p, bit in enumerate(observed)}
    supports = ((2, 7), (5,))
    coefficients = (Fraction(1, 2), Fraction(-1, 4))

    def total(table, mapping):
        value = Fraction(0)
        population = 0
        for x, n in table.items():
            if any(((x >> mapping[b]) & 1) != v for b, v in rec.success + rec.conditions):
                continue
            population += n
            value += n * sum(c * (-1) ** sum((x >> mapping[b]) & 1 for b in support)
                             for c, support in zip(coefficients, supports, strict=True))
        return value, population

    full_sum, full_selected = total(full, {i: i for i in range(9)})
    marginal_sum, marginal_selected = total(marginal, positions)
    assert (full_sum, full_selected) == (marginal_sum, marginal_selected) == (Fraction(-17, 4), 59)
    masks = [sum(1 << positions[b] for b in support) for support in supports]
    mean, _, _ = weighted_group_moments(bits[selected], counts[selected], 1.0, masks, list(map(float, coefficients)))
    assert abs(mean - float(full_sum / full_selected)) <= 8 * 2**-53
    assert np.array_equal(physical, selected)


def test_sampled_samples_adopt_the_reduced_arrays_without_copying(monkeypatch):
    """The Samples record adopts the two fresh int64 arrays of ``reduce_sample_arrays``.

    Its stored arrays are views of the reducer's own buffers, so the record
    holds no copy of them.
    """
    returned = []
    reduce = quantum.reduce_sample_arrays
    monkeypatch.setattr(quantum, "reduce_sample_arrays",
                        lambda *args, **kwargs: returned.append(reduce(*args, **kwargs)) or returned[-1])
    result = nwqlib.solve(choose(output=Samples(), shots=200))
    ((indices, counts, _, _),) = returned
    assert indices.size > 0
    assert np.shares_memory(result.samples.indices.array, indices)
    assert np.shares_memory(result.samples.counts.array, counts)


def test_reduction_allowance_admits_its_workspace_and_reads_the_completed_reduction_ledger():
    """The exact reduction is funded by max_work less the reductions other experiments completed.

    The hook returns a Python int, reads the Run's completed chunks without
    changing them, and refuses the reduction's workspace above max_bytes
    before acquisition.
    """
    from nwqlib._quantum_readout import projected_requirements

    selected = choose(output=NormalizedExpectation(observable=np.diag([0.4, -0.2, 0.1])))
    result = nwqlib.solve(selected)
    observation = selected.experiments[0].observation
    (point,) = observation.positions
    width = selected.reconstruction.width
    reserved, work = projected_requirements(json.loads(point.parameters), width)
    empty = SimpleNamespace(observations=SimpleNamespace(chunks=()))
    completed = SimpleNamespace(observations=result.data.observations)
    same, other = SimpleNamespace(experiment="projected_moments"), SimpleNamespace(experiment="other")
    method = selected.method
    allowance = method.reduction_allowance(selected, same, observation=observation, width=width, run=empty)
    assert type(allowance) is int and allowance == method.max_work
    assert method.reduction_allowance(selected, same, observation=observation, width=width,
                                      run=completed) == method.max_work
    assert method.reduction_allowance(selected, other, observation=observation, width=width,
                                      run=completed) == method.max_work - work
    with pytest.raises(ValueError, match=f"needs {reserved} bytes"):
        method.revise(max_bytes=reserved - 1).reduction_allowance(
            selected, same, observation=observation, width=width, run=empty)


def test_counts_with_no_physical_mass_have_no_samples(monkeypatch):
    from nwqlib.backends import qiskit_aer

    selected = choose(output=Samples(), shots=12)
    rec = selected.reconstruction
    # Successful selectors and coordinate3: outside the original dimension3.
    key = compact_key(rec, rec.settings[0], selector_ones(rec) + list(rec.coordinates))
    monkeypatch.setattr(
        qiskit_aer,
        "_submit_aer_execution",
        lambda prepared, **kw: SimpleNamespace(
            raw_output={"counts": {key: 12}},
            metadata={"native_job_id": "dummy-only"},
        ),
    )
    result = nwqlib.solve(selected)
    assert result.samples.total == 0 and result.samples.indices.array.shape == (0,)
    assert result.physical_slice_mass == 0.0
    assert result.algorithm_success_mass == 1.0 and "no observed physical" in result.unavailable
    before = result.model_dump(mode="json")
    def no_summary_work(*args, **kwargs):
        pytest.fail("sample summary recomputed science or acquired data")
    with monkeypatch.context() as guard:
        guard.setattr(QLS, "analyze", no_summary_work)
        guard.setattr(QLS, "plan", no_summary_work)
        guard.setattr(qiskit_aer, "_submit_aer_execution", no_summary_work)
        guard.setattr(type(result), "validate_plan", no_summary_work)
        text = str(result)
        assert "Samples: 0 stored original-coordinate indices; returned shots=12" in text
        assert "Selected counts: algorithm branch=12; physical coordinates=0" in text
        assert "Observed selection masses: algorithm branch=1; physical slice=0" in text
        assert result.model_dump(mode="json") == before


def _reduced(chunk, pairs):
    """The point chunk with its saved scalar pairs replaced, other components kept."""
    from nwqlib.execution import ReducedValues

    mantissas, exponents, losses = (next(v for v in chunk.values if v.component == k) for k in range(3))
    values = (ReducedValues(component=0, real=tuple(m for m, _ in pairs)),
              ReducedValues(component=1, integers=tuple(e for _, e in pairs)), losses)
    return chunk.revise(values=values)


def test_exact_mass_junction_checks_the_complete_native_norm_and_keeps_subpopulations():
    """The complete native norm is checked against the producing receipt before any scalar is published.

    Drifts inside the receipt's window pass with their raw values; a lost
    one hundredth fails. Success and physical masses are subpopulations: a
    total of one with p_alg = 0.9 and p = 0.7 passes, and a physical mass
    above its success mass beyond the host relation fails.
    """
    from nwqlib._quantum_readout import fraction_pair as pair, pair_value

    result = nwqlib.solve(choose(A=np.diag([1.0, -2.0, 1.5]), b=[1.0, 1j, 0.5],
                                 output=NormalizedExpectation(observable=np.diag([0.4, -0.2, 0.1]))))
    chunk = result.data.observations.chunks[0]
    saved = result.reduction
    receipt = next(item for item in result.data.receipts if item.content_id == chunk.prepared_id)
    window = receipt.probability_window
    assert 0 < window < 1e-6
    tail = [saved.numerator, saved.numerator_radius]
    for drift in (-window / 2, window / 2):
        complete = pair(pair_value(saved.complete_mass) * (1 + drift))
        changed = _reduced(chunk, [complete, saved.success_mass, saved.physical_mass, *tail])
        stats = quantum.projected_moments(result.data, changed)
        assert stats.complete_mass == complete and stats.physical_mass == saved.physical_mass
    lost = _reduced(chunk, [pair(pair_value(saved.complete_mass) * 0.99), saved.success_mass,
                            saved.physical_mass, *tail])
    with pytest.raises(ValueError, match="outside the producing window"):
        quantum.projected_moments(result.data, lost)
    subset = _reduced(chunk, [pair(1), pair(0.9), pair(0.7), *tail])
    stats = quantum.projected_moments(result.data, subset)
    assert pair_value(stats.success_mass) == pair_value(pair(0.9)) and pair_value(stats.physical_mass) != 1
    swapped = _reduced(chunk, [pair(1), pair(0.7), pair(0.9), *tail])
    with pytest.raises(ValueError, match="population relation"):
        quantum.projected_moments(result.data, swapped)


def test_exact_mass_checks_take_the_projected_kernels_resolved_budget_on_aer(monkeypatch):
    """Reduction and publication receive the Aer receipt's resolved budget.

    The Aer receipt of the exact reduction lists "amplitude-derived masses",
    so its unresolved ``state_error()`` is unavailable. The projected kernel
    resolves that label (``PROJECTED_MASS_EXCLUSIONS``), and each mass check
    must receive the same qualified budget
    ``saved_state_error(PROJECTED_MASS_EXCLUSIONS)[0]`` from the receipt's own
    method: the native budget with the charge of the host phase correction
    that the reducer's saved state receives. The branch this choice decides is checked on a synthetic receipt
    in ``test_observation_schedule.py``
    (``test_projected_reducer_resolves_only_the_amplitude_mass_label_at_every_mass_checkpoint``).
    """
    import nwqlib._quantum_readout as readout
    from nwqlib._quantum_readout import PROJECTED_MASS_EXCLUSIONS

    output = NormalizedExpectation(observable=np.diag([0.4, -0.2, 0.1]))
    validate = readout.validate_saved_masses
    received = []

    def recording(*args, **kwargs):
        received.append(kwargs.get("delta"))
        return validate(*args, **kwargs)

    monkeypatch.setattr(readout, "validate_saved_masses", recording)
    result = nwqlib.solve(choose(output=output))
    [receipt] = result.data.receipts
    delta = receipt.saved_state_error(PROJECTED_MASS_EXCLUSIONS)[0]
    assert receipt.probability_window_exclusions == PROJECTED_MASS_EXCLUSIONS
    assert receipt.state_error()[0] is None and delta is not None
    # One call at acquisition and one at publication.
    assert len(received) >= 2 and set(received) == {delta}


DENSE = np.diag([0.4, -0.2, 0.1])
SAVED_CASES = [
    pytest.param(output, shots, "qsvt_inverse", {}, id=f"{name}-{route}")
    for name, output in (("normalized", NormalizedExpectation(observable=DENSE)),
                         ("quadratic", QuadraticForm(observable=DENSE)), ("norm", NormSquared()))
    for route, shots in (("exact", None), ("sampled", 12))
] + [
    pytest.param(Solution(), None, "qsvt_inverse", {}, id="solution-exact"),
    pytest.param(StateVector(), None, "qsvt_inverse", {}, id="state_vector-exact"),
    pytest.param(StateVector(normalization="unit", global_phase="modulo_global_phase"), None, "shortcut_dilation",
                 {}, id="state_vector-shortcut"),
    pytest.param(NormalizedExpectation(observable=DENSE), None, "shortcut_dilation", {}, id="normalized-shortcut"),
    pytest.param(Solution(), None, "qsvt_inverse", {"A": MATRIX * 1e-300, "b": RHS * 1e10},
                 id="solution-unrepresentable"),
]


@pytest.mark.parametrize(("output", "shots", "solver", "inputs"), SAVED_CASES)
def test_saved_quantum_result_loads_with_its_identity(tmp_path, output, shots, solver, inputs):
    """A saved quantum QLS Result of every route loads with its identity.

    A Solution whose physical vector is not representable in complex128
    (``A`` scaled by 1e-300, ``b`` by 1e10) has no artifact and no value,
    and publishes the acquisition's reason.
    """
    result = nwqlib.solve(choose(output=output, shots=shots, solver=solver, **inputs))
    if inputs:
        assert result.artifact is None and result.value is None and result.unavailable is not None
    result.save(tmp_path / "result")
    assert nwqlib.load_result(tmp_path / "result").content_id == result.content_id


def test_sampled_readout_grouping_obeys_qls_max_work():
    """A sampled QLS readout refuses grouping above its Method work limit."""
    import itertools
    from nwqlib.operators import ingest_pauli

    n = 6
    labels = tuple("".join(p) for p in itertools.product("IXYZ", repeat=n) if set(p) != {"I"})
    observable = ingest_pauli(tuple((label, 1.0 / (1 + i)) for i, label in enumerate(labels)), num_qubits=n)

    def selected(cap):
        return choose(A=ingest_pauli((("I" * n, 2.0), ("Z" + "I" * (n - 1), 0.5)), num_qubits=n),
                      b=np.linspace(1.0, 0.1, 1 << n), kappa=2.0, max_work=cap, shots=10,
                      output=NormalizedExpectation(observable=observable))

    with pytest.raises(ValueError, match=r"Pauli grouping exceeds QLS\.max_work=146000\b"):
        selected(146_000)


def _reference_slice(selected):
    """The bound Plan circuit in Statevector, with the physical slice by explicit bit enumeration."""
    from qiskit.quantum_info import Statevector
    from nwqlib.blocks.lowering import lower_qiskit

    _, construction = selected.resolve("projected_moments")._selected_construction(selected)
    chosen = {record.content_id for record in construction.selections}
    circuit = lower_qiskit(construction, blocks=tuple(
        block for block in selected.blocks if block.record.content_id in chosen)).circuit
    state = Statevector(circuit).data
    rec = selected.reconstruction
    algorithm = physical = 0.0
    vector = np.zeros(1 << len(rec.coordinates), dtype=complex)
    for index, amplitude in enumerate(state):
        if any((index >> bit) & 1 != value for bit, value in rec.success):
            continue
        algorithm += abs(amplitude) ** 2
        if any((index >> bit) & 1 != value for bit, value in rec.conditions):
            continue
        coordinate = sum(((index >> bit) & 1) << place for place, bit in enumerate(rec.coordinates))
        if coordinate < rec.original_dimension:
            vector[coordinate] = amplitude
            physical += abs(amplitude) ** 2
    return vector, algorithm, physical


def _pauli_matrix(label):
    one = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
           "Z": np.diag([1, -1])}
    matrix = np.eye(1)
    for axis in label:  # qubit zero is the rightmost letter, the last Kronecker factor
        matrix = np.kron(matrix, one[axis])
    return matrix


@pytest.mark.parametrize("case", ["padded_inverse", "unpadded_pauli", "non_hermitian", "shortcut_dilation",
                                  "small_even_exponent", "small_odd_exponent"])
def test_exact_qls_outputs_match_an_independent_selected_circuit_reference(case):
    """Projected p, p_alg and q against the same bound circuit in Statevector.

    The slice uses the actual success, condition and coordinate maps, and
    each Pauli matrix is built independently from its label. The receipts
    exclude supplied-state operations, so this uses the unit-scale regression
    contract atol=2e-12, rtol=0 for raw masses and q with C <= 1, propagated
    to the normalized value by the quotient rule
    ``[Q + (|q|/p) P]/(p - P) + u |q_hat/p_hat|`` and to physical outputs by
    the recovery. It is not a native error certificate. The unpadded case
    gives a Pauli observable as input, and its terms are read back by label.
    The two small-mass NormSquared cases have physical masses with binary
    exponents -4 and -5, where the published scale halves the exponent.
    """
    from nwqlib._quantum_readout import pair_value
    from nwqlib.operators import ingest_pauli

    dense = np.diag([0.4, -0.2, 0.1]) + 0.05 * (np.eye(3, k=1) + np.eye(3, k=-1))
    A, b, solver, observable = MATRIX, RHS, "qsvt_inverse", dense
    if case == "unpadded_pauli":
        A, b = np.diag([1.0, -2.0, 1.5, 0.8]), [1.0, 0.5, 0.25, -0.5]
        observable = ingest_pauli((("II", 0.1), ("ZX", -0.3), ("XI", 0.25), ("YY", 0.2), ("IZ", 0.0)),
                                  num_qubits=2)
    elif case == "non_hermitian":
        A, b = np.array([[1.5, 0.4], [-0.2, 1.1]]), [1.0, 0.5]
        observable = np.array([[0.3, 0.1], [0.1, -0.2]])
    elif case == "shortcut_dilation":
        solver = "shortcut_dilation"
    elif case == "small_even_exponent":
        A, b = np.diag([1.0, 0.25]), [1.0, 0.01]
    elif case == "small_odd_exponent":
        A, b = np.diag([1.0, 0.5, 1 / 6]), [1.0, 0.1, 0.0]
    outputs = [NormalizedExpectation(observable=observable)]
    if solver == "qsvt_inverse":
        outputs += [QuadraticForm(observable=observable), NormSquared()]
    if case.startswith("small"):
        outputs = [NormSquared()]
    tolerance = 2e-12
    for output in outputs:
        selected = choose(A=A, b=b, solver=solver, output=output)
        result = nwqlib.solve(selected)
        rec = selected.reconstruction
        if case == "non_hermitian":
            assert rec.conditions == ((rec.coordinates[-1] + 1, 1),)
        if case == "shortcut_dilation":
            assert (rec.width - 1, 1) in rec.success
        vector, algorithm, physical = _reference_slice(selected)
        (point,) = selected.experiments[0].observation.positions
        import json
        terms = json.loads(point.parameters)["terms"]
        assert sum(abs(c) for _, c in terms) <= 1
        matrix = sum((c * _pauli_matrix(label) for label, c in terms), np.zeros((len(vector),) * 2))
        numerator = float(np.vdot(vector, matrix @ vector).real)
        if isinstance(observable, np.ndarray) and not isinstance(output, NormSquared):
            padded = np.zeros((len(vector),) * 2)
            padded[:len(observable), :len(observable)] = observable
            assert np.abs(matrix - padded).max() < 1e-15
        elif not isinstance(output, NormSquared):
            rows = (("II", 0.1), ("ZX", -0.3), ("XI", 0.25), ("YY", 0.2))
            assert np.abs(matrix - sum(c * _pauli_matrix(label) for label, c in rows)).max() < 1e-15
        stats = result.reduction
        p, q = float(pair_value(stats.physical_mass)), float(pair_value(stats.numerator))
        assert abs(p - physical) <= tolerance and abs(float(pair_value(stats.success_mass)) - algorithm) <= tolerance
        assert abs(q - numerator) <= tolerance
        assert physical > 0.1 or case.startswith("small")
        assert result.physical_slice_mass == p and result.algorithm_success_mass == float(
            pair_value(stats.success_mass))
        if rec.recovery is not None:
            # The published scale is Gamma*sqrt(p), with the root split by the
            # parity of p's binary exponent (even on the padded and small even
            # fixtures, odd on the unpadded, non-Hermitian and small odd ones).
            # Its square is Gamma**2 times the reference physical mass.
            gamma2 = rec.recovery.as_float() ** 2
            assert abs(result.physical_scale.squared_as_float() - gamma2 * physical) <= (
                gamma2 * tolerance * (1 + 1e-12))
        if isinstance(output, NormalizedExpectation):
            bound = (tolerance + abs(numerator) / physical * tolerance) / (physical - tolerance) + 2**-53 * abs(result.value)
            assert abs(result.value - numerator / physical) <= bound
            continue
        gamma2 = rec.recovery.as_float() ** 2
        target = numerator if isinstance(output, QuadraticForm) else physical
        assert abs(result.value - gamma2 * target) <= gamma2 * tolerance * (1 + 1e-12)


def test_zero_physical_mass_leaves_normalized_output_unavailable_and_physical_outputs_zero():
    """Saved statistics with a zero physical slice give no normalized value and zero physical outputs."""
    from dataclasses import replace
    from nwqlib._quantum_readout import fraction_pair as pair
    from nwqlib.execution import ObservationView

    for output, expected in ((NormalizedExpectation(observable=np.diag([0.4, -0.2, 0.1])), None),
                             (QuadraticForm(observable=np.diag([0.4, -0.2, 0.1])), 0.0), (NormSquared(), 0.0)):
        selected = choose(output=output)
        result = nwqlib.solve(selected)
        chunk = result.data.observations.chunks[0]
        changed = _reduced(chunk, [pair(1), pair(0.5), (0.0, 0), (0.0, 0), (0.0, 0)])
        event_id = ObservationView(chunks=(chunk,)).content_id
        trace = result.data.trace.revise(events=tuple(
            event.revise(observation_id=ObservationView(chunks=(changed,)).content_id)
            if event.observation_id == event_id else event for event in result.data.trace.events))
        data = replace(result.data, observations=ObservationView(chunks=(changed,)), trace=trace)
        analyzed = quantum.analyze_quantum(selected, data)
        assert analyzed.scalar_value == expected and analyzed.physical_slice_mass == 0.0
        assert analyzed.norm_squared == 0.0 and analyzed.physical_scale.as_float() == 0.0
        if expected is None:
            assert "zero physical mass" in analyzed.unavailable


def test_grouped_sampled_setting_shares_one_count_table_within_its_sampling_radius():
    """Two qubit-wise commuting terms share one counts setting with ``shots`` returned shots.

    With C_g = sum |c_j| over the group and n_g its selected shots, the
    estimate lies within ``r = sqrt(2 log(2/delta) sum_g C_g**2/n_g)`` of the
    ideal value with probability at least 1 - delta; the fixed seed makes
    this a deterministic regression at delta = 1e-6. The classical model
    evaluates the same selected polynomial.
    """
    from math import log, sqrt as root
    from nwqlib.operators import ingest_pauli

    rows = (("ZI", 0.3), ("ZZ", -0.2), ("II", 0.1))
    observable = ingest_pauli(rows, num_qubits=2)
    output = NormalizedExpectation(observable=observable)
    A, b = np.diag([1.0, -2.0, 1.5, 0.8]), [1.0, 0.5, 0.25, -0.5]
    selected = choose(A=A, b=b, output=output, shots=4000)
    assert [e.name for e in selected.experiments] == ["group_0"]
    assert selected.reconstruction.groups == (("ZI", "ZZ"),)
    assert selected.reconstruction.grouping_comparisons == 1
    result = nwqlib.solve(selected)
    (group,) = result.groups
    assert group.returned_shots == 4000 and group.population == "success_conditional"
    assert result.data.observations.chunks[0].returned_shots == 4000
    classical = nwqlib.solve(selected.problem, method=selected.method, output=NormalizedExpectation(
        observable=sum(c * _pauli_matrix(label) for label, c in rows).real), execution="classical", seed=7)
    radius = root(2 * log(2 / 1e-6) * 0.5**2 / group.selected_shots)
    assert abs(result.value - classical.value) <= radius


def test_uncontrolled_adjoint_query_keeps_its_native_multiplexers_in_either_query_order():
    """The adjoint base gate a Run shares serves each consumer its own representation.

    An uncontrolled adjoint query keeps the native adjoint multiplexer
    table, while a controlled one is built from the realized forward
    definition. Both share one method context, so the order in which the
    two queries are first built must not decide which representation the
    uncontrolled query receives: in both orders it lowers on Aer to its two
    native multiplexers and 17 operations.
    """
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.subroutines.block_encoding.core import build_block_encoding

    decompose = aer._AerDecompose(AerSimulator(method="statevector").target)
    matrix = SparsePauliOp.from_list([("XZ", .5), ("YY", .3), ("ZI", -.2), ("IX", .4)]).to_matrix()
    for order in ((0, 1, 0, 1), (1, 0, 1, 0)):
        block = select_block_encoding("child", build_block_encoding(matrix, implementation="pauli_lcu"),
                                      operator=ingest_dense(matrix))
        shared = {}
        queries = {controls: quantum._query_leaf(block, kind="test", controls=controls,
                                                 basis=block.record.semantics.basis)
                   for controls in (0, 1)}
        for controls in order:
            circuit = quantum._query_circuit(queries[controls], dict(adjoint=1, control_state=0),
                                             lambda factory: shared)
            if controls == 0:
                ops = aer._prepare_execution_circuit(circuit, decompose=decompose,
                                                     add_save_statevector=False)[0].count_ops()
                assert (ops.get("multiplexer", 0), sum(ops.values())) == (2, 17), order


@pytest.mark.parametrize("solver", ["shortcut_native_svp", "shortcut_dilation"])
def test_quantum_norm_search_is_rejected_before_polynomial(monkeypatch, solver):
    monkeypatch.setattr(owner, "select_inputs", forbidden)
    with pytest.raises(ValueError, match="numeric t"):
        nwqlib.plan(
            LinearSystem(A=np.eye(2), b=[1.0, 1.0]),
            method=QLS(solver=solver, encoded_solution_norm_estimate="grid"),
            output=StateVector(normalization="unit", global_phase="modulo_global_phase"),
            seed=1,
        )


def test_dimension_one_gets_analytic_positive_dummy_and_original_output():
    selected = choose(A=np.array([[2.0]]), b=np.array([3j]))
    result = nwqlib.solve(selected)
    assert selected.problem.dimension == 1 and selected.reconstruction.padded_dimension == 2
    assert result.x.shape == (1,)
    np.testing.assert_allclose(result.x, [1.5j], rtol=0.01, atol=0.01)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_eq17_phase_widens_both_endpoints_without_changing_raw_criterion(side):
    """Perturb mass between raw and phase-widened endpoints to test both sides without changing
    the raw discrepancy.
    """
    from nwqlib.algorithms.qls.verification import _comparisons

    rec = SimpleNamespace(
        sigma_min=1.0,
        sigma_max=2.0,
        alpha=2.0,
        kappa_be=2.0,
        polynomial_kappa=2.0,
        phase_error=0.05,
        polynomial=SimpleNamespace(rescale=2.0, eta=0.1),
        t=1.5,
    )
    plan = SimpleNamespace(reconstruction=rec, method=QLS(), execution="quantum")
    center = 4 * (9 / 4) * (5 / 2) / ((9 / 4) + (5 / 2)) ** 2
    lower = center * (0.9 / 1.1) ** 2 / 4
    upper = (center + 4 * 0.1**2 / 1.1**2) / 4
    widened_lower = max(0, np.sqrt(lower) - 0.05) ** 2
    widened_upper = (np.sqrt(upper) + 0.05) ** 2
    mass = (lower + widened_lower) / 2 if side == "lower" else (upper + widened_upper) / 2
    observed = SimpleNamespace(algorithm_success_mass=mass, returned_shots=None)
    automatic = QLSVerification(name="phase", comparisons=("eq17",))
    values = _comparisons(
        plan,
        observed,
        automatic,
        bounds=(1.0, 2.0),
        norm=np.sqrt(5 / 2),
        discrepancy=None,
        parameters={},
    )
    assert values["phase.eq17.phase_lower"] == pytest.approx(widened_lower, abs=1e-15)
    assert values["phase.eq17.phase_upper"] == pytest.approx(widened_upper, abs=1e-15)
    expected = lower - mass if side == "lower" else mass - upper
    assert values["phase.eq17.raw_discrepancy"] == pytest.approx(expected, abs=1e-15)
    assert values["phase.eq17"] < 1
    raw = _comparisons(
        plan,
        observed,
        automatic.revise(probability_tolerance=0.0),
        bounds=(1.0, 2.0),
        norm=np.sqrt(5 / 2),
        discrepancy=None,
        parameters={},
    )
    assert raw["phase.eq17"] == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("observed", [0.0, 1.0])
def test_binomial_allowance_uses_reference_rate_with_legal_observed_endpoints(observed):
    from nwqlib.algorithms.qls.verification import _comparisons

    rec = SimpleNamespace(
        alpha=2.0,
        kappa_be=2.0,
        polynomial_kappa=2.0,
        phase_error=0.0,
        polynomial=SimpleNamespace(rescale=2.0),
    )
    plan = SimpleNamespace(reconstruction=rec, method=QLS(), execution="quantum")
    options = QLSVerification(name="shots", comparisons=("inverse_success",))
    result = SimpleNamespace(algorithm_success_mass=observed, returned_shots=12)
    values = _comparisons(
        plan, result, options, bounds=(1.0, 2.0), norm=np.sqrt(2.5), discrepancy=None, parameters={}
    )
    assert values["shots.inverse_success.shot_allowance"] == pytest.approx(
        4 * np.sqrt((5 / 32) * (27 / 32) / 12), abs=1e-15
    )
    assert values["shots.inverse_success.raw_discrepancy"] == pytest.approx(
        abs(observed - 5 / 32), abs=1e-15
    )
    rec.phase_error = None
    values = _comparisons(
        plan, result, options, bounds=(1.0, 2.0), norm=np.sqrt(2.5), discrepancy=None, parameters={}
    )
    assert (
        values["shots.inverse_success"]
        is values["shots.inverse_success.deterministic_allowance"]
        is None
    )
    assert values["shots.inverse_success.raw_discrepancy"] == pytest.approx(
        abs(observed - 5 / 32), abs=1e-15
    )


def test_outside_bernoulli_reference_keeps_raw_prediction_without_heuristic():
    from nwqlib.algorithms.qls.verification import _comparisons

    rec = SimpleNamespace(
        alpha=2.0,
        kappa_be=2.0,
        polynomial_kappa=2.0,
        phase_error=0.0,
        polynomial=SimpleNamespace(rescale=0.2),
    )
    plan = SimpleNamespace(reconstruction=rec, method=QLS(), execution="quantum")
    choice = QLSVerification(comparisons=("inverse_success",))
    result = SimpleNamespace(algorithm_success_mass=0.4, returned_shots=12)
    values = _comparisons(
        plan, result, choice, bounds=(1.0, 2.0), norm=np.sqrt(2.5), discrepancy=None, parameters={}
    )
    assert values["reference.inverse_success.predicted_mass"] > 1
    assert (
        values["reference.inverse_success.shot_allowance"]
        is values["reference.inverse_success"]
        is None
    )
    assert values["reference.inverse_success.raw_discrepancy"] > 1


def test_run_archive_keeps_already_constructed_queries(tmp_path, monkeypatch):
    selected = choose(A=np.array([[1.0, 0.1j], [0.2, 2.0]]), b=[1.0, 1j])
    prepared = nwqlib.prepare(selected)
    result = nwqlib.submit(prepared).wait()
    run = prepared.run
    keys = tuple(run._state["method_context"]["qls_base_gates"])
    assert keys
    run.save(tmp_path / "run")
    monkeypatch.setattr(quantum, "_native_encoding", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    loaded = nwqlib.load_run(tmp_path / "run", backend=run.backend)
    assert tuple(loaded._state["method_context"]["qls_base_gates"]) == keys
    again = loaded.resume().wait()
    np.testing.assert_array_equal(again.x, result.x)


@pytest.mark.parametrize(
    "solver,general,a_queries,a_controls,rhs_queries",
    [
        ("qsvt_inverse", False, 1, 0, 0),
        ("qsvt_inverse", True, 2, 1, 0),
        ("shortcut_native_svp", False, 1, 1, 2),
        ("shortcut_dilation", False, 2, 2, 4),
    ],
)
def test_actual_query_projector_and_phase_populations_match_independent_counts(
    solver, general, a_queries, a_controls, rhs_queries
):
    """Walk actual query calls and compare solver-specific multiplicities, controls and phase
    counts with independent expectations.
    """
    from nwqlib.ir import BlockCall, Sequence

    selected = choose(
        A=np.array([[1.0, 0.1], [0.0, 0.5]]) if general else np.diag([1.0, 0.5]),
        b=[1.0, 1j],
        solver=solver,
        epsilon_inv=0.1,
        block_encoding_implementation="dense_dilation",
    )
    r = selected.reconstruction
    program = selected.construction.program
    definitions = {d.id: d.node for d in program.definitions}
    expressions = {e.id: e.value.value for e in program.expressions}
    calls = []

    def walk(name):
        node = definitions[name]
        if isinstance(node, Sequence):
            for child in node.children:
                walk(child)
        elif isinstance(node, BlockCall):
            calls.append(
                (
                    node.signature,
                    {
                        a.parameter: getattr(
                            expressions[a.value.expression],
                            "value",
                            expressions[a.value.expression],
                        )
                        for a in node.arguments
                    },
                )
            )

    walk(program.root)
    query = next(s for s in selected.construction.selections if s.signature.name.startswith("original_A"))
    assert query.semantics.base_semantics == selected._native["encoding"].record.semantics
    aq = [row for row in calls if row[0].startswith("original_A_query")]
    rhs = [row for row in calls if row[0].startswith("rhs_query")]
    assert len(aq) == a_queries * r.degree and all(row[0].endswith(f"_c{a_controls}") for row in aq)
    assert len(rhs) == rhs_queries * r.degree
    assert sum(name.startswith("rz_") for name, _ in calls) == r.degree + 1
    assert sum(name.startswith("global_phase") for name, _ in calls) == 1
    if general or solver == "shortcut_dilation":
        assert sum(args["adjoint"] for _, args in aq) == r.degree
        assert {args["control_state"] for _, args in aq} == {
            0,
            1,
        }  # outer dilation bit precedes fixed-zero augmentation bit
    else:
        assert sum(args["adjoint"] for _, args in aq) == r.degree // 2
    report = nwqlib.estimate(selected)
    count = report.quantity("calls").fact.value
    assert count.numerator / count.denominator == len(calls)


def _supplied_child(circuit, *, ancillas, alpha, error):
    """Select a supplied child encoding together with its declared original operator."""
    from qiskit.quantum_info import Operator

    step = 1 << ancillas
    operator = ingest_dense(alpha * Operator(circuit).data[::step, ::step])
    encoding = select_block_encoding(
        "child",
        BlockEncoding(
            circuit=circuit,
            alpha=alpha,
            num_ancillas=ancillas,
            system_qubits=circuit.num_qubits - ancillas,
            error_bound=error,
            implementation="supplied",
            metadata={},
        ),
        operator=operator,
    )
    return operator, encoding


def _lowered_forward_query_block(selected):
    """Lower only the selected forward query of the actual Program and return its encoded block."""
    from qiskit.quantum_info import Operator
    from nwqlib.blocks.lowering import lower_qiskit
    from nwqlib.blocks.records import SelectedConstruction
    from nwqlib.ir import Definition, Sequence as IRSequence
    from nwqlib.subroutines.block_encoding import block_encoding_top_left

    program = selected.construction.program
    allocations = tuple(d.id for d in program.definitions if d.id.startswith("allocate_"))
    root = Definition(
        id="forward_query_only", node=IRSequence(children=(*allocations, "encoding_forward"))
    )
    query = SelectedConstruction(
        program=program.revise(root=root.id, definitions=program.definitions + (root,)),
        selections=selected.construction.selections,
    )
    unitary = Operator(lower_qiskit(query, blocks=selected.blocks).circuit).data
    # Pair, signal, child and shortcut ancillas are the low-order wires below the system.
    return block_encoding_top_left(unitary, num_ancillas=selected.reconstruction.coordinates[0])


@pytest.mark.parametrize(("child_ancillas", "system_qubits"), ((0, 1), (1, 1), (0, 2)))
def test_shortcut_queries_keep_child_ancillas_complex_block_and_error_premise(
    child_ancillas, system_qubits
):
    """The selected G_t query and its dilation act on a supplied complex child as in Dalzell
    arXiv:2406.12086v2, App. A; the expected blocks come from the child's own unitary.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    a, n = child_ancillas, system_qubits
    child = QuantumCircuit(a + n)
    child.global_phase = 0.19
    child.rx(0.51, a)
    child.rz(0.2, a)
    if a:
        child.ry(0.6, 0)
        child.cx(0, 1)
        child.ry(0.41, 0)
    encoded = Operator(child).data[:: 1 << a, :: 1 << a]
    operator, encoding = _supplied_child(child, ancillas=a, alpha=2.0, error=0.02)
    rhs = np.zeros(1 << n, dtype=complex)
    rhs[:2] = [np.sqrt(0.3), 1j * np.sqrt(0.7)]
    # A_t = (A/alpha) (+) |0><0|/t, b' = (b, 1)/sqrt(2) and G_t = (I - |b'><b'|) A_t,
    # with t = 1.5 from choose().
    augmented = np.zeros((2 << n, 2 << n), dtype=complex)
    augmented[: 1 << n, : 1 << n] = encoded
    augmented[1 << n, 1 << n] = 1 / 1.5
    b_prime = np.zeros(2 << n, dtype=complex)
    b_prime[: 1 << n] = rhs / np.sqrt(2)
    b_prime[1 << n] = 1 / np.sqrt(2)
    g_t = (np.eye(2 << n) - np.outer(b_prime, b_prime.conj())) @ augmented
    # G_t is not normal, so a projector applied on the wrong side changes the block.
    assert np.max(np.abs(g_t.conj().T @ g_t - g_t @ g_t.conj().T)) >= 1.0e-2
    zero = np.zeros_like(g_t)
    targets = {
        "shortcut_native_svp": g_t,
        "shortcut_dilation": np.block([[zero, g_t], [g_t.conj().T, zero]]),
    }
    for solver, target in targets.items():
        # kappa=4 bounds the child's condition number, which is at most 1.15 here.
        selected = choose(
            A=operator, b=rhs, solver=solver, encoding=encoding, kappa=4.0, epsilon_inv=0.1
        )
        rec = selected.reconstruction
        # Wire order: pair and signal, the child ancillas, two shortcut ancillas, the
        # system, the augmentation bit and, for the dilation, one more bit.
        assert rec.encoding_ancillas == a
        assert rec.coordinates == tuple(range(a + 4, a + 4 + n))
        assert rec.width == a + 5 + n + (solver == "shortcut_dilation")
        # The child's alpha and original-A error premise reach every query unchanged.
        assert rec.alpha == 2.0 and rec.encoding_error == 0.02
        premises = {
            (s.semantics.base_semantics.alpha, s.semantics.base_semantics.epsilon)
            for s in selected.construction.selections
            if s.signature.name.startswith("original_A_query")
        }
        assert premises == {(2.0, 0.02)}
        # Binary64 products of these few dozen unit-norm gates stay near 1e-14. A wrong
        # projector, control value, phase or wire moves entries by O(0.1).
        np.testing.assert_allclose(
            _lowered_forward_query_block(selected), target, rtol=0, atol=2.0e-12
        )


@pytest.mark.parametrize(
    ("solver", "child_states", "rhs_states"),
    (("shortcut_native_svp", (0, 0), (0,)), ("shortcut_dilation", (1, 0), (0, 1))),
)
def test_shortcut_lowering_synthesizes_each_controlled_query_once(
    monkeypatch, solver, child_states, rhs_states
):
    """Degree-many query calls reuse one controlled synthesis per adjoint flag and control value.

    The forward G_t query controls A on the fixed augmentation value and prepares b' with
    controlled U_b and U_b^dagger. The dilation applies G_t^dagger on dilation value 0 and
    G_t on value 1, which doubles the controlled RHS syntheses but not the child's.
    """
    # Aer's import-time name mapping calls Gate.control; import it before counting.
    import qiskit_aer  # noqa: F401
    from collections import Counter
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    from nwqlib.blocks.lowering import lower_qiskit

    child = QuantumCircuit(1)
    child.ry(0.31, 0)
    child.rz(0.2, 0)
    child.global_phase = 0.19
    operator, encoding = _supplied_child(child, ancillas=0, alpha=1.0, error=0.0)
    selected = choose(
        A=operator,
        b=[np.sqrt(0.3), 1j * np.sqrt(0.7)],
        solver=solver,
        encoding=encoding,
        kappa=2.0,
        epsilon_inv=0.1,
    )
    assert selected.reconstruction.degree > 2
    base = selected._native["encoding"]._payload.circuit.name
    events = Counter()
    original = Gate.control

    def counted(gate, *args, **kwargs):
        # Circuit-defined composite gates carry the generic controlled synthesis.
        if type(gate) is Gate:
            events[gate.name, kwargs.get("ctrl_state")] += 1
        return original(gate, *args, **kwargs)

    monkeypatch.setattr(Gate, "control", counted)
    lower_qiskit(selected.construction, blocks=selected.blocks)
    observed = sorted(
        (name.removesuffix("_dg") == base, name.endswith("_dg"), state, count)
        for (name, state), count in events.items()
    )
    forward, adjoint = child_states
    expected = sorted(
        [(True, False, forward, 1), (True, True, adjoint, 1)]
        + [(False, inverse, state, 1) for inverse in (False, True) for state in rhs_states]
    )
    assert observed == expected


def test_explicit_prepared_inventory_uses_actual_selected_circuit_without_submission(monkeypatch):
    selected = choose(A=np.diag([1.0, -2.0]), b=[1.0, 1j], output=NormSquared(), epsilon_inv=0.1)
    original = quantum._native_encoding
    monkeypatch.setattr(quantum, "_native_encoding", forbidden)
    report = nwqlib.estimate(selected)
    assert report.construction_id == selected.construction.content_id
    monkeypatch.setattr(quantum, "_native_encoding", original)
    prepared = nwqlib.prepare(selected)
    assert len(prepared.circuits) == 1 and not prepared.run.trace.events
    circuit = prepared.circuits[0]
    assert circuit.num_qubits == selected.reconstruction.width <= 13
    snapshot = prepared.circuits[0]
    assert snapshot.count_ops() == circuit.count_ops() and snapshot is not circuit
    assert not prepared.run.trace.events


def test_multi_controlled_x_cx_law_reads_the_stored_synthesis_counts():
    from nwqlib.subroutines._mcx_counts import MCX_CX_BY_CONTROLS

    # tests/test_mcx_counts.py checks the table against the installed Qiskit.
    # A multi-controlled X takes its entry as an estimate, and a gate beyond
    # the table has no CX law.
    assert quantum._primitive_cx_law("x", 2) == (MCX_CX_BY_CONTROLS[1], False)
    assert quantum._primitive_cx_law("x", len(MCX_CX_BY_CONTROLS) + 1) is None


@pytest.mark.parametrize(
    ("solver", "hermitian", "route"),
    [("qsvt_inverse", False, "gatewise"), ("qsvt_inverse", False, "auto"), ("shortcut_native_svp", False, "auto"),
     ("shortcut_dilation", False, "auto"), ("shortcut_dilation", False, "whole_matrix"), ("qsvt_inverse", True, "auto")],
)
def test_controlled_dense_query_synthesis_is_admitted_at_planning(solver, hermitian, route):
    """Refuse the exact synthesis of a controlled dense query, and Qiskit's control of it, before the original SVD.

    A controlled query of the dense dilation of a 16-dimensional system
    synthesizes its unitary on log2(16) + 1 = 5 qubits, once for the forward
    and once for the adjoint specialization, and Qiskit then controls the
    synthesized gates with one control, or two for ``shortcut_dilation``.
    On the whole-matrix route, which "auto" takes for one control, each
    specialization instead synthesizes the controlled matrix on five qubits
    plus the controls. Planning charges twice the laws of the route. The
    Hermitian inverse queries the dilation without controls, and its
    construction synthesizes nothing.
    """
    from test_dense_synthesis import _dense_synthesis_work
    from test_synthesis_admission import _controlled_synthesis_work, _gatewise_control_work

    rng = np.random.default_rng(4)
    A = np.eye(16) + 0.2 * rng.normal(size=(16, 16))
    if hermitian:
        A = (A + A.T) / 2
    controls = 2 if solver == "shortcut_dilation" else 1
    if route == "whole_matrix" or (route == "auto" and controls == 1):
        need = 2 * _controlled_synthesis_work(5 + controls)
    else:
        need = 2 * (_dense_synthesis_work(5) + _gatewise_control_work(5, controls))

    def planned(max_work):
        return choose(A=A, b=np.ones(16), solver=solver, epsilon_inv=0.1,
                      block_encoding_implementation="dense_dilation", max_work=max_work,
                      dense_control_route=route)

    if hermitian:
        planned(need - 1)
        return
    with pytest.raises(ValueError, match="QLS controlled dense-query synthesis exceeds max_work"):
        planned(need - 1)
    planned(need)


@pytest.mark.parametrize(("general", "syntheses", "route"), [(True, 2, "gatewise"), (True, 2, "auto"),
                                                            (False, 0, "auto")])
def test_dense_query_syntheses_of_a_run_match_the_planning_charge(monkeypatch, general, syntheses, route):
    """Count the exact syntheses and control steps of a whole inverse Run against the two of each that planning charges.

    For the general A the controlled query of the dense dilation acts on
    two qubits with one control, and "auto" synthesizes the controlled
    matrix on three qubits instead. The Hermitian A is queried without a
    control and synthesizes nothing. The shortcut solvers, whose two readout
    settings reuse the specializations of the first preparation, are
    counted in ``test_synthesis_admission.py``.
    """
    from test_dense_synthesis import _counted_syntheses
    from test_synthesis_admission import _counted_dense_controls

    selected = choose(A=np.array([[1.0, 0.1], [0.0, 0.5]]) if general else np.diag([1.0, 0.5]), b=[1.0, 1j],
                      epsilon_inv=0.1, block_encoding_implementation="dense_dilation", dense_control_route=route)
    calls = _counted_syntheses(monkeypatch)
    controls = _counted_dense_controls(monkeypatch)
    nwqlib.solve(selected, progress=False)
    assert calls == [3 if route == "auto" else 2] * syntheses
    assert controls == [(1, (2,))] * syntheses
