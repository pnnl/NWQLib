"""Selected compact QHD scientific/resource witnesses.

All native selectors here use at most eight total qubits.
"""

import numpy as np
import pytest
import scipy.linalg
import sympy as sp
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from nwqlib.algorithms.qhd.grid import OneHotGrid
from nwqlib.algorithms.qhd.kinetic import KineticCompiler
from nwqlib.algorithms.qhd.potential import PotentialCompiler
from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import (
    append_number_projector_phase,
    append_pauli_evolution_block,
    structured_number_projector_provider,
)


def test_paired_hopping_matches_independent_dense_reference_with_two_cx() -> None:
    grid = OneHotGrid(variables=("x",), bounds=((-1.0, 1.0),), num_grid_points=2)
    block = KineticCompiler().compile(grid, dt=0.37, kinetic_weight=0.6)[0]
    circuit = QuantumCircuit(2)
    append_pauli_evolution_block(circuit, block)

    x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
    coefficient = complex(block.terms[0].coefficient).real
    reference = scipy.linalg.expm(
        -1j
        * block.time_step
        * coefficient
        * (np.kron(x_matrix, x_matrix) + np.kron(y_matrix, y_matrix))
    )
    assert np.max(np.abs(Operator(circuit).data - reference)) <= 1.0e-12

    # exp(-i t c (XX+YY)) with 0 < |t c| < pi/4 has Weyl coordinates
    # (|t c|, |t c|, 0): two CX are necessary and sufficient.
    paired_cx = int(
        transpile(circuit, basis_gates=["cx", "u"], optimization_level=0).count_ops().get("cx", 0)
    )
    assert paired_cx == 2


@pytest.mark.parametrize("support_size", [1, 2, 3, 4])
def test_projector_lowering_matches_literal_dense_reference(support_size: int) -> None:
    if support_size == 4:
        assert structured_number_projector_provider(4)["provider"] == "diagonal_synthesis"
    angle = -0.37
    dimension = 2**support_size
    projector = np.zeros((dimension, dimension), dtype=complex)
    projector[-1, -1] = 1.0
    reference = scipy.linalg.expm(-1j * angle * (projector - np.eye(dimension) / dimension))
    circuit = QuantumCircuit(support_size)
    append_number_projector_phase(circuit, range(support_size), angle)
    assert np.max(np.abs(Operator(circuit).data - reference)) <= 1.0e-12


@pytest.mark.parametrize("support_size", (3, 4))
def test_number_projector_small_phase_survives_execution_decomposition(support_size) -> None:
    circuit = QuantumCircuit(support_size)
    append_number_projector_phase(circuit, range(support_size), 1.0e-11)
    diagonal = np.diag(Operator(circuit.decompose(reps=10)).data)
    relative = diagonal[-1] / diagonal[0]
    assert relative.imag / -1.0e-11 == pytest.approx(1.0, rel=1.0e-10, abs=0.0)


@pytest.mark.parametrize("angle", (0.0, 0.37))
def test_projector_routing_prices_the_realized_provider(angle) -> None:
    expected_costs = [0, 2, 6, 14, 30, 62, 126, 220]
    for support_size, expected_cost in enumerate(expected_costs, start=1):
        provider = structured_number_projector_provider(support_size)
        support = tuple(range(support_size))
        grid = OneHotGrid(
            variables=sp.symbols(f"x:{support_size}"),
            bounds=((-1.0, 1.0),) * support_size,
            num_grid_points=2,
        )
        compiler = PotentialCompiler(rotation_threshold=0.0)
        # The table entry at grid indices (0, ..., 0) is the angle; the other entries are zero.
        table = np.zeros(2**support_size)
        table[0] = angle
        blocks = compiler.compile_selected(
            compiler.select_occurrences(
                {support: table},
                num_grid_points=grid.num_grid_points,
                dt=1.0,
                potential_weight=1.0,
            ),
            grid,
            dt=1.0,
        )
        # A zero projector is the identity: no emitted block may reserve CX work,
        # even when the user disables pruning. Nonzero blocks keep the routing.
        assert len(blocks) == int(angle != 0.0)
        assert compiler._dropped_nonzero_contribution_count == 0
        if angle == 0.0:
            assert (
                sum(
                    block.metadata["projector_provider"]["cx"] for block in blocks
                )
                == 0
            )
            continue
        assert blocks[0].metadata["projector_provider"] == provider
        circuit = QuantumCircuit(support_size)
        append_number_projector_phase(circuit, range(support_size), angle)
        realized = transpile(
            circuit,
            basis_gates=["cx", "u"],
            optimization_level=0,
            seed_transpiler=7,
        )

        # Exact for the pinned provider and cx/u basis at optimization level zero.
        assert provider["cx"] == expected_cost
        assert int(realized.count_ops().get("cx", 0)) == expected_cost
