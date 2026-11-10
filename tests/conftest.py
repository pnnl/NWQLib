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


def lchs_dense_quadrature(matrix, final_time, method):
    """Return the selected quadrature and padded (L, H) of a dense A, as LCHS planning selects them."""
    from nwqlib import LinearDynamics
    from nwqlib.algorithms.lchs.route_options import route_name
    from nwqlib.algorithms.lchs.selection import select_dense

    problem = LinearDynamics(A=matrix, initial_state=np.eye(len(matrix))[0], time=final_time)
    data = select_dense(method, problem, backend=route_name(method.ham_evo))
    return data.quadrature, data.dense.l_part, data.dense.h_part


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
                                  has_control_flow_op=lambda: False, parameters=())
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


def offline_ibm_service(api, instance):
    """A real IBM SDK service and device whose API client is ``api``, on the offline FakeManilaV2 target.

    ``api`` stands in for the SDK's HTTP client, so only the network is
    replaced. Returns ``(service, backend)``. The device is named
    ``ibm_supplied``.
    """
    from qiskit_ibm_runtime import IBMBackend, IBMQuantumComputeService
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    fake = FakeManilaV2()
    configuration = fake.configuration()
    configuration.backend_name = "ibm_supplied"
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
    return service, backend


def executor_program(params):
    """The program of one request to IBM's ``executor``, decoded from its JSON ``params`` by the SDK's own converter."""
    from ibm_quantum_schemas.executor.version_2_0 import ParamsModel
    from qiskit_ibm_runtime.quantum_program.converters.converters_2_0 import quantum_program_from_2_0

    return quantum_program_from_2_0(ParamsModel.model_validate(params))[0]


def executor_items(program, outcomes, seed=0):
    """Synthetic result arrays of each item of ``program``, in the form that IBM's ``executor`` returns them.

    ``outcomes(index, item, register_index, register, shape, shots)`` gives the
    integer outcome of every shot of one classical register of item ``index``,
    an array that broadcasts to ``shape + (shots,)``, where ``shots`` is the
    program's shots of each row. ``shape`` is the item's shape: () for
    a circuit without parameter values, one entry per row of values otherwise,
    and for a twirled item (a samplex item) the randomizations first. Bit j of
    an outcome is bit j of the register, as in the executor's little-endian
    boolean arrays. A samplex item also returns the outputs of its samplex
    other than the parameter values, sampled with ``seed``, and its bits are
    flipped by its measurement flips, so the SDK's decoder, which flips them
    back, recovers the outcomes.
    """
    from qiskit_ibm_runtime.fake_provider.executor.broadcast_sample import broadcast_sample
    from samplomatic.quantum_program import SamplexItem

    rng, data = np.random.default_rng(seed), []
    for index, item in enumerate(program.items):
        if isinstance(item, SamplexItem):
            shape = tuple(item.shape)
            item.samplex.finalize()
            arrays = broadcast_sample(item.samplex, item.samplex_arguments, shape, rng)
            arrays.pop("parameter_values", None)
        else:
            arrays, values = {}, item.circuit_arguments
            shape = () if values is None or values.ndim < 2 else values.shape[:-1]
        for register_index, register in enumerate(item.circuit.cregs):
            outcome = np.broadcast_to(np.asarray(outcomes(index, item, register_index, register, shape, program.shots),
                                                 dtype=np.uint64), (*shape, program.shots))
            bits = ((outcome[..., None] >> np.arange(register.size, dtype=np.uint64)) & np.uint64(1)).astype(bool)
            flips = arrays.get("measurement_flips." + register.name)
            arrays[register.name] = bits if flips is None else bits ^ flips
        data.append(arrays)
    return data


def executor_result_json(program, data, *, passthrough=None, item_metadata=None, chunk_timing=()):
    """The JSON of an ``executor`` result (schema v2.0) with the item arrays ``data`` of ``program``.

    ``passthrough`` replaces the program's passthrough data, which carries the
    circuit metadata of each PUB. ``item_metadata`` gives the metadata fields
    of each item, and ``chunk_timing`` the job's timing spans.
    """
    from ibm_quantum_schemas.common import CompressedTensorModel
    from ibm_quantum_schemas.executor.version_2_0.models import (
        ItemMetadataModel, MetadataModel, QuantumProgramResultItemModel, QuantumProgramResultModel)
    from qiskit_ibm_runtime.quantum_program.converters.converters_2_0 import passthrough_data_to_2_0

    item_metadata = item_metadata or [{}] * len(data)
    return QuantumProgramResultModel(
        data=[QuantumProgramResultItemModel(
            results={name: CompressedTensorModel.from_numpy(np.asarray(array)) for name, array in arrays.items()},
            metadata=ItemMetadataModel(**fields)) for arrays, fields in zip(data, item_metadata, strict=True)],
        metadata=MetadataModel(chunk_timing=list(chunk_timing)),
        passthrough_data=passthrough_data_to_2_0(program.passthrough_data if passthrough is None else passthrough),
        semantic_role=program._semantic_role).model_dump_json()


@pytest.fixture
def ibm_runtime_spy(monkeypatch):
    """An IBMRuntimeBackend on the real SDK and the offline FakeManilaV2 target, with only the network replaced.

    Compilation, ISA preparation, the native-depth record, the client-side
    primitives' job programs and result decoding are the SDK's and NWQLib's
    own. The replaced API client gives each job its own identifier, reports
    every job completed and returns, for each item, the program's shots with
    every classical bit 0. These counts are synthetic, so a test reads
    acquisition and exposure facts from them, not energies.
    ``spy.api.requests`` holds every job request.
    """
    from types import SimpleNamespace
    from nwqlib.backends import IBMRuntimeBackend

    instance = "crn:supplied-instance"

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
            self.jobs[job] = (kwargs["program_id"], kwargs["params"], kwargs["job_tags"])
            return {"id": job, "backend": "ibm_supplied"}

        def job_get(self, job_id, **kwargs):
            program, _, tags = self.jobs[job_id]
            return dict(id=job_id, backend="ibm_supplied", program={"id": program},
                        state={"status": "COMPLETED"}, tags=tags)

        def job_results(self, job_id):
            program = executor_program(self.jobs[job_id][1])
            return executor_result_json(program, executor_items(program, lambda *args: 0))

    api = API()
    service, backend = offline_ibm_service(api, instance)
    monkeypatch.setattr(IBMRuntimeBackend, "_new_service", lambda self: service)
    connection = IBMRuntimeBackend(device=backend.name, instance=instance)
    return SimpleNamespace(connection=connection, api=api)
