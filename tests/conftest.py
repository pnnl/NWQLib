"""Shared test helpers."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import Operator, SparseObservable, SparsePauliOp

from nwqlib.subroutines.block_encoding import BlockEncoding, block_encoding_top_left


def lchs_quadrature_options() -> dict[str, object]:
    from nwqlib.algorithms.lchs.provider_config import ProviderConfig

    return {
        "lchs_kernel": ProviderConfig(
            implementation="near_optimal_eq7",
            parameters={"beta": 0.8},
        ),
        "epsilon": 0.5,
        "k_quadrature": ProviderConfig(
            implementation="composite_gauss",
            parameters={"truncation_multiplier": 1.0},
        ),
    }


def dense_evolution_copy(circuit: QuantumCircuit) -> QuantumCircuit:
    """Copy with each raw PauliEvolutionGate replaced by its exact unitary.

    Use SciPy's sparse exponential on CSC input as an independent reference
    without the sparse-format warnings from Qiskit's raw-gate conversion.
    """

    replaced = circuit.copy_empty_like()
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, PauliEvolutionGate):
            operator = operation.operator
            if isinstance(operator, SparseObservable):
                operator = SparsePauliOp.from_sparse_observable(operator)
            generator = operator.to_matrix(sparse=True).tocsc()
            exact = scipy.sparse.linalg.expm(-1j * float(operation.time) * generator)
            operation = UnitaryGate(np.asarray(exact.todense()))
        replaced.append(operation, instruction.qubits, instruction.clbits)
    return replaced


def _encoded_block(circuit: QuantumCircuit, num_ancillas: int) -> np.ndarray:
    """Return the all-zero-ancilla block encoded by ``circuit``."""

    return block_encoding_top_left(Operator(circuit).data, num_ancillas=num_ancillas)


_BLOCK_ENCODING_METADATA_MIRRORS = frozenset(
    {
        f"block_encoding_{field.name}"
        for field in fields(BlockEncoding)
        if field.name not in {"circuit", "metadata"}
    }
    | {
        # Historical spellings that do not follow the field-name prefix.
        "block_encoding_ancillas",
        "block_encoding_error",
        "resolved_block_encoding_implementation",
    }
)


def assert_no_block_encoding_metadata_mirrors(record) -> None:
    """Pin top-level BlockEncoding fields as the only structural owners."""

    assert _BLOCK_ENCODING_METADATA_MIRRORS.isdisjoint(record.metadata)


REPO_ROOT = Path(__file__).resolve().parents[1]

def load_docs_script(relative_path: str):
    """Import a docs/scripts module by path (shared _load_script pattern)."""

    import importlib.util
    import sys

    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclass decorators look the module up
    spec.loader.exec_module(module)
    return module


# Used only by explicit metadata/lifecycle tests. Real native witnesses keep
# their own fixtures and do not acquire this replacement implicitly.
@pytest.fixture
def prepared_stubs(monkeypatch):
    """Instrument both native boundaries with scalar stand-ins; no circuit SDK work."""
    from types import SimpleNamespace
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.blocks import lowering

    state = SimpleNamespace(lowered=[], prepared=[], population="unconditional")

    class Register(tuple):
        def __new__(cls, name, bits):
            value = super().__new__(cls, bits)
            value.name = name
            return value

        def __getnewargs__(self):
            return self.name, tuple(self)

    def logical(construction, **kwargs):
        state.lowered.append(construction)
        quantum, width = [], 0
        for reg in construction.program.registers:
            quantum.append(Register(reg.name, range(width, width + reg.width)))
            width += reg.width
        offset = 0
        registers = []
        for reg in construction.program.classical:
            if reg.dtype == "bits":
                registers.append(Register(reg.name, range(offset, offset + reg.width)))
                offset += reg.width
        circuit = SimpleNamespace(data=[SimpleNamespace(operation="logical operation")], cregs=registers, qregs=quantum,
                                  num_qubits=width, find_bit=lambda bit: SimpleNamespace(index=bit),
                                  has_control_flow_op=lambda: False)
        return lowering.LogicalCircuit(
            construction_id=construction.content_id, circuit=circuit,
            quantum_layout=tuple((reg.name, tuple(reg)) for reg in quantum),
            measurement_layout=(), dynamic_visits=7, construction_work=11,
            defined_selections=tuple(item.content_id for item in construction.selections))

    def native(target, circuit, **kwargs):
        state.prepared.append((target, kwargs))
        return SimpleNamespace(circuit=circuit, metadata={"readout_population": state.population},
                               simulator=SimpleNamespace(target=SimpleNamespace(operation_names=("stub",)),
                                   options=SimpleNamespace(method="statevector", zero_threshold=0.0)))

    def forbidden(*args, **kwargs):
        pytest.fail("metadata/stub test crossed an uninstrumented native boundary")

    monkeypatch.setattr(lowering, "_lower_qiskit", logical)
    monkeypatch.setattr(aer, "_prepare_aer_execution", native)
    monkeypatch.setattr(aer, "_lower_aer_circuit", forbidden)
    monkeypatch.setattr(aer, "_submit_aer_execution", forbidden)
    state.logical = logical
    return state
