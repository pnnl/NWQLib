"""Tests for MPS state-preparation helpers."""

from __future__ import annotations

import sys
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from nwqlib.subroutines.state_preparation import mps_circuit as mps_circuit_module
from nwqlib.subroutines.state_preparation import mps as mps_module
from nwqlib.subroutines.lcu import build_lcu_prepare, prepare_lcu_data
from nwqlib.subroutines.state_preparation import (
    analyze_mps_state_compression,
    build_mps_circuit_state_preparation,
    build_qiskit_state_preparation,
    decompose_state_to_mps,
    validate_mps_circuit_state_preparation,
)


# Exact circuit classes differ only by ideal-gate floating evaluation at the
# scale of a short statevector simulation (FRAMEWORK machine-precision class).
EXACT_PREPARATION_ATOL = 1.0e-14


def test_mps_circuit_backend_requires_tensor_dependency_for_entangled_state(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "scikit_tt", None)
    monkeypatch.setitem(sys.modules, "scikit_tt.tensor_train", None)
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)
    state[3] = 1.0 / np.sqrt(2)

    with pytest.raises(ImportError, match="optional tensor"):
        build_mps_circuit_state_preparation(state, num_layers=2)


@pytest.mark.parametrize("num_layers", (1, 2))
def test_mps_circuit_backend_prepares_small_entangled_state_when_tensor_is_available(
    monkeypatch, num_layers
) -> None:
    pytest.importorskip("scikit_tt")
    _, tt_class = mps_circuit_module._require_scikit_tt()
    original_dot, original_ortho = tt_class.dot, tt_class.ortho
    residual_calls = {"dot": 0, "ortho": 0}

    def recording_dot(self, *args, **kwargs):
        residual_calls["dot"] += 1
        return original_dot(self, *args, **kwargs)

    def recording_ortho(self, *args, **kwargs):
        if kwargs.get("threshold") == 1.0e-12:
            residual_calls["ortho"] += 1
        return original_ortho(self, *args, **kwargs)

    monkeypatch.setattr(tt_class, "dot", recording_dot)
    monkeypatch.setattr(tt_class, "ortho", recording_ortho)
    state = np.zeros(4, dtype=complex)
    state[0] = np.exp(0.37j) * np.sqrt(0.6)
    state[3] = 1j * np.exp(0.37j) * np.sqrt(0.4)

    preparation = build_mps_circuit_state_preparation(state, num_layers=num_layers)
    validated = validate_mps_circuit_state_preparation(preparation)

    # Two extracted local unitaries need a residual only when another layer follows.
    expected_updates = 2 * (num_layers - 1)
    assert residual_calls == {"dot": expected_updates, "ortho": expected_updates}
    assert preparation.method == "mps_disentangling_circuit"
    assert preparation.fidelity_to_target is None
    assert validated.circuit_fidelity_status == "evaluated"
    assert validated.fidelity_to_target == pytest.approx(1.0, abs=1.0e-10)
    np.testing.assert_allclose(validated.prepared_state, state, atol=1e-12, rtol=0)
    assert validated.preparation_l2_error == pytest.approx(
        np.linalg.norm(validated.prepared_state - validated.target_state),
        rel=1.0e-12,
        abs=1.0e-15,
    )


def test_mps_validator_simulates_completed_circuit_exactly_once(monkeypatch) -> None:
    preparation = build_mps_circuit_state_preparation(
        np.array([1.0, 1.0, 0.0, 0.0], dtype=complex),
        max_bond_dim=1,
    )
    simulation = Mock(wraps=mps_circuit_module.Statevector.from_instruction)
    monkeypatch.setattr(
        mps_circuit_module.Statevector,
        "from_instruction",
        simulation,
    )
    validated = validate_mps_circuit_state_preparation(preparation)

    assert simulation.call_count == 1
    assert validated.prepared_state is not None
    assert validated.circuit_fidelity_status == "evaluated"


def test_mps_builders_and_resource_tiers_never_simulate(monkeypatch) -> None:
    """Poison simulation and count MPS builds to distinguish metadata folding from explicit
    representative/full preparation.
    """
    def forbidden_simulation(*_args, **_kwargs):
        raise AssertionError("ordinary MPS construction must not simulate")

    monkeypatch.setattr(
        mps_circuit_module.Statevector,
        "from_instruction",
        forbidden_simulation,
    )
    build_mps_circuit_state_preparation(
        np.array([1.0, 1.0, 0.0, 0.0], dtype=complex),
        max_bond_dim=1,
    )

    identities = [np.eye(2, dtype=complex) for _ in range(4)]
    lcu_data = prepare_lcu_data(np.ones(4, dtype=complex), identities)
    build_lcu_prepare(
        lcu_data,
        preparation_backend="mps_circuit",
        mps_max_bond_dim=1,
    )

    from nwqlib import LinearDynamics,NormSquared,plan,prepare,estimate
    from nwqlib.algorithms.lchs import LCHS, ProviderConfig
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    builder=Mock(wraps=mps_circuit_module.build_mps_circuit_state_preparation)
    monkeypatch.setattr(mps_circuit_module,'build_mps_circuit_state_preparation',builder)
    chosen=plan(LinearDynamics(A=[[.25,.04j],[.04j,.4]],initial_state=[1,.2],time=.1),
        method=LCHS(approximation_tolerance=.8,hamiltonian_evolution_backend='trotter',
            k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                parameters={'num_qubits':2,'lsb_position':0}),
            initial_state_preparation='mps_circuit',initial_state_mps_max_bond_dim=1,
            lcu_state_preparation='mps_circuit',lcu_mps_max_bond_dim=1),output=NormSquared())
    estimate(chosen)
    assert builder.call_count==0
    sample_resources(chosen,max_qubits=5)
    assert builder.call_count==2  # Actual initial/coefficient PREP, inverse reuses the latter.
    builder.reset_mock()
    preview=prepare(chosen)
    assert all(c.num_qubits<=5 for c in preview.circuits)
    assert builder.call_count==2


def test_mps_circuit_reuses_the_compression_cores_once(monkeypatch) -> None:
    pytest.importorskip("scikit_tt")
    state = np.array([0.5, 0.25j, -0.5, 0.5j], dtype=complex)
    consumed_cores = []
    original = mps_circuit_module._scikit_tt_mps_from_cores

    def recording_conversion(cores, *, tt_class):
        consumed_cores.append(cores)
        return original(cores, tt_class=tt_class)

    monkeypatch.setattr(
        mps_circuit_module,
        "_scikit_tt_mps_from_cores",
        recording_conversion,
    )
    preparation = build_mps_circuit_state_preparation(state, num_layers=2)

    assert len(consumed_cores) == 1
    assert all(
        consumed is recorded
        for consumed, recorded in zip(
            consumed_cores[0],
            preparation.decomposition.cores,
            strict=True,
        )
    )


@pytest.mark.parametrize("truncated", (False, True))
def test_mps_builder_uses_cores_without_full_reconstruction(monkeypatch, truncated) -> None:
    decomposition_calls = 0
    reconstruction_calls = 0
    original_decomposition = mps_module._decompose_normalized_state
    original_reconstruction = mps_module._reconstruct_cores

    def counting_decomposition(*args, **kwargs):
        nonlocal decomposition_calls
        decomposition_calls += 1
        return original_decomposition(*args, **kwargs)

    def counting_reconstruction(cores, **kwargs):
        nonlocal reconstruction_calls
        reconstruction_calls += 1
        return original_reconstruction(cores, **kwargs)

    monkeypatch.setattr(
        mps_module,
        "_decompose_normalized_state",
        counting_decomposition,
    )
    monkeypatch.setattr(mps_module, "_reconstruct_cores", counting_reconstruction)
    state = np.exp(0.37j) * (
        np.array([np.sqrt(0.8), 0, 0, 1j * np.sqrt(0.2)])
        if truncated
        else np.array([1.0, 1.0j, 0.0, 0.0]) / np.sqrt(2)
    )
    preparation = build_mps_circuit_state_preparation(state, max_bond_dim=1)
    metadata = preparation.to_dict()

    assert decomposition_calls == 1
    assert reconstruction_calls == 0
    assert metadata["compression_fidelity_to_target"] is None
    assert "reconstruction_error" not in metadata["decomposition"]
    assert "compression" not in vars(preparation)
    assert preparation.prepared_state is None
    expected = np.exp(0.37j) * np.array([1, 0, 0, 0]) if truncated else state
    validated = validate_mps_circuit_state_preparation(preparation)
    np.testing.assert_allclose(validated.prepared_state, expected, atol=1e-14, rtol=0)
    # The normalized four-amplitude comparison above bounds overlap roundoff at this scale.
    assert validated.fidelity_to_target == pytest.approx(
        0.8 if truncated else 1.0, rel=0.0, abs=1.0e-13
    )
    assert reconstruction_calls == 0

    analysis = analyze_mps_state_compression(preparation.decomposition, reference=state)
    assert decomposition_calls == 1
    assert reconstruction_calls == 1
    assert analysis.measured_normalized_fidelity == pytest.approx(0.8 if truncated else 1.0, abs=1e-13, rel=0)
    assert analysis.measured_raw_l2 == pytest.approx(np.sqrt(0.2) if truncated else 0, abs=1e-13, rel=0)


def test_mps_product_preparation_preserves_controlled_phase_and_endian_order():
    state = np.exp(0.37j) * np.kron(
        np.array([1, 1j]) / np.sqrt(2),
        np.kron(np.array([0, 1]), np.array([np.sqrt(0.3), np.sqrt(0.7)])),
    )
    preparation = build_mps_circuit_state_preparation(state, max_bond_dim=1)
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.append(preparation.circuit.to_gate().control(1), range(4))
    actual = Statevector.from_instruction(circuit).data
    expected = np.zeros(16, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[1::2] = state / np.sqrt(2)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    assert preparation.circuit.count_ops() == {"u": 3}


def test_mps_scalar_state_uses_only_its_phase():
    state = np.array([np.exp(0.37j)])
    preparation = build_mps_circuit_state_preparation(state)
    assert preparation.circuit.num_qubits == 0
    assert preparation.circuit.global_phase == pytest.approx(0.37, rel=0.0, abs=1.0e-14)
    np.testing.assert_allclose(decompose_state_to_mps(state).to_statevector(), state)


@pytest.mark.parametrize(
    ("state", "expected_operation"),
    [
        (np.array([0.0, 0.0, 1.0j, 0.0]), "x"),
        (np.full(4, np.exp(0.37j) / 2.0), "h"),
        (
            np.array(
                [
                    np.exp(-0.23j) / np.sqrt(3.0),
                    np.exp(-0.23j) / np.sqrt(3.0),
                    np.exp(-0.23j) / np.sqrt(3.0),
                    0.0,
                ]
            ),
            "USup",
        ),
    ],
)
def test_exact_direct_fast_paths_preserve_independent_reference(
    state: np.ndarray,
    expected_operation: str,
) -> None:
    preparation = build_qiskit_state_preparation(state)
    actual = np.asarray(Statevector.from_instruction(preparation.circuit).data)
    expected = np.asarray(state, dtype=complex) / np.linalg.norm(state)

    assert expected_operation in preparation.circuit.count_ops()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )
    assert abs(np.vdot(expected, actual)) ** 2 == pytest.approx(
        1.0,
        rel=0.0,
        abs=EXACT_PREPARATION_ATOL,
    )


def test_direct_preparation_generic_native_action() -> None:
    state = np.array([0.5, 0.25j, -0.5, 0.5j], dtype=complex)
    preparation = build_qiskit_state_preparation(state)

    np.testing.assert_allclose(
        Statevector.from_instruction(preparation.circuit).data,
        state / np.linalg.norm(state),
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )
    assert preparation.to_dict()["fidelity_to_target"] is None
    assert preparation.to_dict()["preparation_error_model"] == "ideal_gate_construction"


@pytest.mark.parametrize(
    "state",
    [
        [1.0, 1.0e-12],
        [1.0, 1.0e-200, 0.0, 0.0],
        [1.0, 1.0e-12j, -1.0e-12, -1.0e-12j],
        [1.0e-12j, 1.0j, 0.0, -1.0e-12],
        [1.0, 1.0e-12j, 0.0, 0.0, -1.0, -1.0e-12j, 0.0, 0.0],
    ],
)
def test_direct_preparation_preserves_small_complex_components(state) -> None:
    expected = np.asarray(state, dtype=complex) / np.hypot.reduce(np.abs(state))
    preparation = build_qiskit_state_preparation(state)
    actual = Statevector.from_instruction(preparation.circuit.decompose(reps=10)).data
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=EXACT_PREPARATION_ATOL)
    small = (np.abs(expected) < 1.0e-10) & (expected != 0.0)
    # Scaling separates a missing amplitude from binary64 roundoff in the circuit.
    np.testing.assert_allclose(actual[small] / expected[small], 1.0, rtol=0.0, atol=0.01)


def test_direct_preparation_controlled_relative_phase_and_inverse() -> None:
    state = np.asarray([1.0j, 1.0e-8, -0.3, 0.2j])
    target = state / np.hypot.reduce(np.abs(state))
    preparation = build_qiskit_state_preparation(state)
    gate = preparation.circuit.to_gate()
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.append(gate.control(1, annotated=False), circuit.qubits)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[1::2] = target / np.sqrt(2)
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit.decompose(reps=10)).data,
        expected,
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )
    circuit.append(gate.inverse().control(1, annotated=False), circuit.qubits)
    expected[:] = 0.0
    expected[:2] = 1 / np.sqrt(2)
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit.decompose(reps=10)).data,
        expected,
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )


@pytest.mark.parametrize(
    "state",
    [
        np.array(
            [0.5, 0.5, 0.5, np.nextafter(0.5, 1.0)],
            dtype=complex,
        ),
        np.array([0.5, 0.5j, -0.5, -0.5j], dtype=complex),
    ],
    ids=("nearly_equal_full_support", "relative_phases"),
)
def test_direct_preparation_exact_classifier_rejects_approximate_uniform_states(
    state: np.ndarray,
) -> None:
    preparation = build_qiskit_state_preparation(state)
    actual = np.asarray(Statevector.from_instruction(preparation.circuit).data)
    expected = np.asarray(state, dtype=complex) / np.linalg.norm(state)

    assert "h" not in preparation.circuit.count_ops()
    assert "USup" not in preparation.circuit.count_ops()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )


def test_direct_and_mps_state_preparation_implementations_are_equivalent() -> None:
    pytest.importorskip("scikit_tt")
    state = np.array([0.5, 0.25j, -0.5, 0.5j], dtype=complex)

    direct = build_qiskit_state_preparation(state)
    mps = build_mps_circuit_state_preparation(state, num_layers=2)

    direct_state = np.asarray(Statevector.from_instruction(direct.circuit).data)
    mps_state = np.asarray(Statevector.from_instruction(mps.circuit).data)
    # Machine-precision identity: both implementations prepare the same normalized state.
    assert abs(np.vdot(direct_state, mps_state)) ** 2 >= 1.0 - 1.0e-10


def test_mps_state_preparation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="power of two"):
        decompose_state_to_mps(np.ones(3))

    with pytest.raises(ValueError, match="max_bond_dim"):
        decompose_state_to_mps(np.ones(4), max_bond_dim=0)

    with pytest.raises(ValueError, match="threshold"):
        decompose_state_to_mps(np.ones(4), threshold=-1.0)


def test_low_somma_mps_fixture_calibrates_from_actual_ordered_coefficients():
    from nwqlib import LinearDynamics,plan
    from nwqlib.algorithms.lchs import LCHS,resolve_lchs_coefficient_plan
    pytest.importorskip('scikit_tt')
    chosen=plan(LinearDynamics(A=[[.25,.04j],[.04j,.4]],initial_state=[1,.2],time=.1),
        method=LCHS(approximation_tolerance=.8,lchs_kernel='low_somma_f2',k_quadrature='symmetric_uniform_trapezoid',
            hamiltonian_evolution_backend='trotter',lcu_state_preparation='mps_circuit',mps_num_layers=2))
    coefficients=resolve_lchs_coefficient_plan(chosen)
    target=coefficients.prep_amplitudes()
    cores=coefficients.mps_decomposition
    assert len(coefficients.coefficients)==15 and len(target)==16
    assert chosen.reconstruction.selected_select=='multiplexor'
    assert cores.bond_dimensions==(1,2,4,2,1)
    validated=validate_mps_circuit_state_preparation(build_mps_circuit_state_preparation(
        target,num_layers=2,_selected_decomposition=cores))
    assert validated.fidelity_to_target>=.999
    # Compression describes the selected tensor; a separately acquired
    # circuit fidelity is not silently copied into the Plan's PREP error.
    preparation=next(b for b in chosen.blocks if b.record.signature.name=='coefficient_prep')
    assert preparation.record.semantics.epsilon is None


def test_existing_mps_analysis_metrics_reuse_order_and_before_svd_cost(monkeypatch):
    state = np.array([np.sqrt(3)/2, 0, 0, .5], dtype=complex)
    truncated = decompose_state_to_mps(state, max_bond_dim=1)
    product = decompose_state_to_mps(np.array([1, 2, 2, 4])/5, max_bond_dim=1)
    reordered = decompose_state_to_mps(np.array([1, 2, 4, 2])/5)
    assert product.bond_dimensions == (1, 1, 1)
    assert reordered.bond_dimensions == (1, 2, 1)
    assert product.input_id != reordered.input_id
    assert all(not core.flags.writeable for core in truncated.cores)
    with pytest.raises(ValueError):
        truncated.cores[0].setflags(write=True)
    original = mps_module._reconstruct_cores
    calls = []
    monkeypatch.setattr(np.linalg, "svd", Mock(side_effect=AssertionError("unexpected SVD")))
    monkeypatch.setattr(mps_module, "_reconstruct_cores", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    scalar = analyze_mps_state_compression(truncated)
    scalar.to_dict()
    assert calls == []
    assert scalar.decomposition is truncated
    assert scalar.estimated_raw_l2 == pytest.approx(.5, rel=0, abs=1e-14)
    assert scalar.estimated_normalized_fidelity == pytest.approx(.75, rel=0, abs=1e-14)
    comparison = analyze_mps_state_compression(truncated, reference=state)
    assert calls == [1]
    assert comparison.reference_matches_input
    assert comparison.measured_raw_l2 == pytest.approx(.5, rel=0, abs=1e-14)
    assert comparison.measured_normalized_fidelity == pytest.approx(.75, rel=0, abs=1e-14)
    assert analyze_mps_state_compression(product).estimated_raw_l2 == pytest.approx(0, rel=0, abs=1e-14)
    with pytest.raises(ValueError, match="max_svd_work"):
        decompose_state_to_mps(np.ones(8), max_svd_work=1)
    with pytest.raises(ValueError, match="max_bytes"):
        truncated.to_statevector(max_bytes=1)


def test_selected_mps_reuse_binds_actual_normalized_input(monkeypatch):
    target = np.array([1., 0., 0., 0.], dtype=complex)
    mps = decompose_state_to_mps(target, max_bond_dim=1)
    monkeypatch.setattr(np.linalg, "svd", Mock(side_effect=AssertionError("selected cores must not repeat SVD")))
    built = build_mps_circuit_state_preparation(target, max_bond_dim=1, _selected_decomposition=mps)
    assert built.decomposition is mps
    foreign = np.array([0., 0., 0., 1.], dtype=complex)
    with pytest.raises(ValueError, match="selected core"):
        build_mps_circuit_state_preparation(foreign, max_bond_dim=1, _selected_decomposition=mps)


def test_measured_mps_fidelity_depends_only_on_directions(monkeypatch):
    """The measured normalized fidelity is invariant to how the vectors were normalized.

    Its roundoff window covers only the evaluation of the fidelity, so the
    value must not inherit the error of a norm routine. The norm of the raw
    tensor is replaced by one that is 0.1 percent too large. A fidelity
    computed as ``|<target, raw / norm>|**2`` would then read 0.998 for an
    exact decomposition, while the fidelity of the two directions is one.
    """
    state = np.array([1.0, 2.0, 2.0, 4.0], dtype=complex) / 5.0
    exact = decompose_state_to_mps(state)
    true_norm = mps_module.stable_vector_norm
    monkeypatch.setattr(mps_module, "stable_vector_norm", lambda value: 1.001 * true_norm(value))
    analysis = analyze_mps_state_compression(exact, reference=state)
    assert analysis.fidelity_roundoff_window > 0.0
    assert abs(analysis.raw_measured_fidelity - 1.0) <= analysis.fidelity_roundoff_window
    assert analysis.measured_normalized_fidelity == pytest.approx(1.0, rel=0, abs=1e-15)


def test_fidelity_kernel_accepts_extreme_but_finite_entry_magnitudes():
    """Vectors whose largest modulus is subnormal or overflows keep their fidelity.

    Dividing by a subnormal scale overflows NumPy's reciprocal, and the
    modulus of ``1.5e308 + 1.5e308j`` overflows although both parts are
    finite. Exact power-of-two rescaling avoids both, so the values below
    are the exact fidelities 1/2 and 1 up to the returned roundoff window.
    """
    from nwqlib._numerics import normalized_fidelity_with_window

    tiny = np.array([1e-310, 0.0], dtype=complex)
    huge = np.array([1.5e308 + 1.5e308j, 1.5e308 + 1.5e308j])
    for vector, other, expected in (
        (tiny, np.array([1.0, 1.0], dtype=complex), 0.5),
        (huge, np.array([1.0 + 1.0j, 1.0 + 1.0j]), 1.0),
        (huge, np.array([1.0, 0.0], dtype=complex), 0.5),
    ):
        fidelity, window = normalized_fidelity_with_window(vector, other)
        assert abs(fidelity - expected) <= window


@pytest.mark.parametrize("distance", [1e-9, 1e-7, 1e-5])
def test_mps_two_qubit_gates_are_exact_near_a_product_state(distance) -> None:
    # A two-qubit state has bond dimension two, so one layer prepares it
    # exactly apart from the synthesis of its local unitaries. Qiskit 2.5.2's
    # TwoQubitBasisDecomposer replaced the nearly local unitary of this state
    # by a local one and prepared |00> alone, an error equal to the distance.
    pytest.importorskip("scikit_tt")
    state = np.array([1.0, 0.0, 0.0, distance], dtype=complex)
    state /= np.linalg.norm(state)
    preparation = build_mps_circuit_state_preparation(state, num_layers=1)
    prepared = Statevector.from_instruction(preparation.circuit).data
    np.testing.assert_allclose(prepared, state, atol=EXACT_PREPARATION_ATOL, rtol=0)


def test_mps_generic_state_keeps_three_cx_per_two_qubit_gate() -> None:
    # Away from a product state every extracted two-qubit unitary is generic
    # and takes three CX in either synthesis. One layer on four qubits emits
    # three of them, the slot bound 3 * layers * (n - 1) of
    # docs/ENGINEERING_CONSTANTS.md.
    pytest.importorskip("scikit_tt")
    rng = np.random.default_rng(8)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    preparation = build_mps_circuit_state_preparation(state, num_layers=1)
    assert preparation.circuit.count_ops()["cx"] == 9


def test_mps_builder_admits_its_syntheses_and_layered_construction_first(monkeypatch) -> None:
    # Each layer synthesizes at most n - 1 two-qubit unitaries exactly. The
    # builder charges layers * (n - 1) syntheses against max_svd_work before
    # the first one, apart from its TT-SVD, whose work for 16 amplitudes is
    # far smaller. One unit below the law refuses before any synthesis, and
    # the law admits.
    pytest.importorskip("scikit_tt")
    from test_dense_synthesis import _dense_synthesis_work

    rng = np.random.default_rng(9)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    calls = []
    original = mps_circuit_module.dense_unitary_circuit
    monkeypatch.setattr(mps_circuit_module, "dense_unitary_circuit",
                        lambda matrix: calls.append(np.shape(matrix)) or original(matrix))
    need = 2 * 3 * _dense_synthesis_work(2)
    with pytest.raises(ValueError, match="max_svd_work"):
        build_mps_circuit_state_preparation(state, num_layers=2, max_svd_work=need - 1)
    assert calls == []
    build_mps_circuit_state_preparation(state, num_layers=2, max_svd_work=need)
    assert calls and set(calls) == {(4, 4)} and len(calls) <= 6

    # After the TT-SVD, which runs NumPy's SVD, scikit_tt orthonormalizes the
    # MPS again after every extracted gate with SciPy's SVD. For ten random
    # qubits those SVDs alone, m*c*min(m, c) each, exceed the syntheses and
    # the TT-SVD, so a limit one below their traced total must refuse the
    # construction before scikit_tt starts. The default limit builds it.
    import scipy.linalg

    state = rng.normal(size=1024) + 1j * rng.normal(size=1024)
    traced = []
    original_svd = scipy.linalg.svd

    def svd(matrix, *args, **kwargs):
        traced.append(matrix.shape[0] * matrix.shape[1] * min(matrix.shape))
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "svd", svd)
    build_mps_circuit_state_preparation(state, num_layers=2)
    started = Mock(wraps=mps_circuit_module._scikit_tt_mps_from_cores)
    monkeypatch.setattr(mps_circuit_module, "_scikit_tt_mps_from_cores", started)
    with pytest.raises(ValueError, match="layered MPS construction needs .* max_svd_work"):
        build_mps_circuit_state_preparation(state, num_layers=2, max_svd_work=sum(traced) - 1)
    assert started.call_count == 0
