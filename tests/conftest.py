"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import Operator, SparseObservable, SparsePauliOp

from nwqlib.subroutines.block_encoding import block_encoding_top_left


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


@pytest.fixture
def ibm_runtime_spy(monkeypatch):
    """An IBMRuntimeBackend on the real SDK and the offline FakeManilaV2 target, with only the network replaced.

    Compilation, ISA preparation, the native-depth record, PUB encoding and
    result decoding are the SDK's and NWQLib's own. The replaced API client
    gives each job its own identifier, reports every job completed and
    returns, for each PUB, its requested shots with every classical bit 0.
    These counts are synthetic, so a test reads acquisition and exposure
    facts from them, not energies. ``spy.api.requests`` holds every job
    request.
    """
    import json
    from types import SimpleNamespace
    from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult
    from qiskit_ibm_runtime import IBMBackend, IBMQuantumComputeService, RuntimeEncoder
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from nwqlib.backends import IBMRuntimeBackend

    instance = "crn:supplied-instance"
    fake = FakeManilaV2()
    configuration = fake.configuration()
    configuration.backend_name = "ibm_supplied"

    class API:
        _instance = instance

        def __init__(self):
            self.jobs, self.requests = {}, []

        def backend_status(self, name):
            return dict(backend_name=name, backend_version="1", operational=True, pending_jobs=0,
                        status_msg="active")

        def program_run(self, **kwargs):
            self.requests.append(kwargs)
            job = f"job-{len(self.requests)}"
            self.jobs[job] = (kwargs["program_id"], kwargs["params"]["pubs"], kwargs["job_tags"])
            return {"id": job, "backend": "ibm_supplied"}

        def job_get(self, job_id, **kwargs):
            primitive, _, tags = self.jobs[job_id]
            return dict(id=job_id, backend="ibm_supplied", program={"id": primitive},
                        state={"status": "COMPLETED"}, tags=tags)

        def job_results(self, job_id):
            _, pubs, _ = self.jobs[job_id]
            results = [SamplerPubResult(
                DataBin(**{register.name: BitArray.from_samples(["0" * register.size] * pub.shots,
                                                                num_bits=register.size)
                           for register in pub.circuit.cregs}),
                metadata={"circuit_metadata": pub.circuit.metadata, "shots": pub.shots}) for pub in pubs]
            return json.dumps(PrimitiveResult(results, metadata={}), cls=RuntimeEncoder)

    api = API()
    service = object.__new__(IBMQuantumComputeService)
    service._channel = "ibm_quantum_platform"
    service._api_clients = {instance: api}
    service._active_api_client = api
    # IBMQuantumComputeService.__init__ starts with this value. From qiskit-ibm-runtime 0.51
    # on, decoding a job looks its backend up here, and an empty list gives is_mock=False.
    service._backend_instance_groups = []
    service._check_instance_usage = lambda: None
    backend = IBMBackend(configuration, service=service, api_client=api, instance=instance)
    backend._target, backend._properties = (fake.target, fake.properties())
    service.backend = lambda name, **kwargs: backend
    service._create_backend_obj = lambda name, **kwargs: backend
    monkeypatch.setattr(IBMRuntimeBackend, "_new_service", lambda self: service)
    connection = IBMRuntimeBackend(device=backend.name, instance=instance, max_input_bytes=65536)
    return SimpleNamespace(connection=connection, api=api)
