"""Native preparation and generator-gate cache relations for ADAPT.

The controller's bounded native trajectory witness is in test_adapt_primary.
"""

from types import SimpleNamespace
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.linalg import expm
from nwqlib.algorithms.gcim import adapt_acquisition as acquisition
from nwqlib.subroutines.fermionic_pool import enumerate_spin_adapted_gsd_pool
from nwqlib.subroutines.fermionic_circuits import _plan_generator_circuit
from _fermionic_references import generator_matrix
from test_gcim_native_preparations import reference_kwargs


def provider_fixture(monkeypatch, generator, theta, basis_index):
    reference = QuantumCircuit(generator.num_qubits, global_phase=0.23)
    for qubit in range(generator.num_qubits):
        if (basis_index >> qubit) & 1:
            reference.x(qubit)
    kwargs = reference_kwargs(reference)
    kwargs.update(
        inputs=SimpleNamespace(pool={generator.pool_index: generator}),
        compiler_plans={generator.pool_index: _plan_generator_circuit(generator)},
    )
    items = ((generator.pool_index, theta),)

    def forbidden(*args, **kwargs):
        pytest.fail("compact preparation expanded scientific state/operator")

    with monkeypatch.context() as guard:
        guard.setattr(Statevector, "from_instruction", forbidden)
        guard.setattr(SparsePauliOp, "to_matrix", forbidden)
        forward = acquisition._native_preparation(items, controlled=False, inverse=False, **kwargs)
        controlled = acquisition._native_preparation(
            items, controlled=True, inverse=False, **kwargs
        )
    return forward, controlled, kwargs


@pytest.mark.parametrize(
    "n_spatial,pool_index,basis_index", ((2, 1, 3), (3, 8, 5), (4, 25, 5), (3, 0, 3))
)
def test_compact_provider_matches_reference_and_controlled_phase(
    monkeypatch, n_spatial, pool_index, basis_index
):
    generator = enumerate_spin_adapted_gsd_pool(n_spatial)[pool_index]
    theta = 0.17
    reference = np.zeros(1 << generator.num_qubits, dtype=complex)
    reference[basis_index] = np.exp(0.23j)
    expected = expm(theta * generator_matrix(generator)) @ reference
    forward, controlled, _ = provider_fixture(monkeypatch, generator, theta, basis_index)
    np.testing.assert_allclose(
        Statevector.from_instruction(forward).data, expected, atol=1e-12, rtol=0.0
    )
    superposition = QuantumCircuit(generator.num_qubits + 1)
    superposition.h(0)
    superposition.append(controlled, superposition.qubits)
    target = np.zeros(2 * len(expected), complex)
    target[0] = 1 / np.sqrt(2)
    target[1::2] = expected / np.sqrt(2)
    np.testing.assert_allclose(
        Statevector.from_instruction(superposition).data, target, atol=1e-12, rtol=0.0
    )


def test_compact_provider_reuses_current_preparations_and_generator_gates(monkeypatch):
    generator = enumerate_spin_adapted_gsd_pool(3)[0]
    theta = 0.17
    forward, controlled, kwargs = provider_fixture(monkeypatch, generator, theta, 3)
    context = kwargs["context"]
    ids = {key: id(value) for key, value in context.gates.items()}
    from nwqlib.subroutines import fermionic_circuits

    monkeypatch.setattr(
        fermionic_circuits,
        "build_generator_circuit",
        lambda *a, **k: pytest.fail("unchanged native generator rebuilt"),
    )
    items = ((generator.pool_index, theta),)
    assert (
        acquisition._native_preparation(items, controlled=False, inverse=False, **kwargs) is forward
    )
    assert (
        acquisition._native_preparation(items, controlled=True, inverse=False, **kwargs)
        is controlled
    )
    assert {key: id(value) for key, value in context.gates.items()} == ids


def test_compiler_plans_are_built_for_selected_generators_only_and_saved_with_the_run(
    tmp_path, monkeypatch
):
    """Planning builds no generator compiler plan; a run builds one per selected generator.

    A durable Run writes its Plan archive before any selection. Stopped by
    an interruption inside the matrix stage that built the first selected
    generator's plan and reopened, it resumes with that plan from its
    checkpoint. Native verification and saving and loading the Result of the
    completed Run, or of the Run reopened again, build no further plan and
    acquire nothing again. This H2 Run selects one generator, whose plan is
    on the commuting route; the next test covers occupation blocks.
    """
    import nwqlib
    from _h2_sto3g_reference import H2_STO3G_JW_TERMS
    from nwqlib.algorithms.gcim import ADAPT, AdaptVerificationOptions, adapt
    from nwqlib.core.planning import RandomStreams
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import ingest_occupation
    from nwqlib.problems.records import Eigenproblem
    from nwqlib._prepared_execution import Run

    built = []

    def counted(generator, **limits):
        built.append(generator.pool_index)
        return _plan_generator_circuit(generator, **limits)

    monkeypatch.setattr(adapt, "_plan_generator_circuit", counted)
    problem = Eigenproblem(A=ingest_pauli(H2_STO3G_JW_TERMS, num_qubits=4))
    method = ADAPT(initial_state=ingest_occupation((1, 1, 0, 0), num_qubits=4),
                   pool="spin_adapted_sd", n_spatial_orbitals=2, max_iterations=2)

    def plan():
        return method.plan(problem, output=problem.default_output(), execution="quantum",
                           shots=None, rng=RandomStreams(7))

    with Run(plan()) as run:
        reference = run.wait(timeout=60, poll_interval=0)
    selected = sorted(set(reference.selected))
    assert len(run.plan.reconstruction.pool) == 4 and selected and sorted(built) == selected
    built.clear()
    # The reference pencil's shared query also serves the first screen. The
    # Run is stopped at the second query of the two-state matrix stage, after
    # its first pair query built the first selected generator's plan.
    submit, submitted = acquisition.submit_experiment, []

    def interrupted(*args, **kwargs):
        if len(submitted) == 2:
            raise RuntimeError("stop inside the matrix stage")
        submitted.append(1)
        return submit(*args, **kwargs)

    with Run(plan(), directory=tmp_path / "run") as run:
        with monkeypatch.context() as patch:
            patch.setattr(acquisition, "submit_experiment", interrupted)
            with pytest.raises(RuntimeError, match="inside the matrix stage"):
                run.wait(timeout=60, poll_interval=0)
        backend = run.backend
    assert built == selected[:1]
    built.clear()
    with nwqlib.load_run(tmp_path / "run", backend=backend) as restored:
        result = restored.wait(timeout=60, poll_interval=0)
    assert (result.selected, result.eigenvalue) == (reference.selected, reference.eigenvalue)
    assert sorted(built) == selected[1:]
    built.clear()
    loaded = nwqlib.load_result(result.save(tmp_path / "result"))
    assert [index for index, _ in loaded.plan._native["compiler_plans"].built()] == selected
    with nwqlib.load_run(tmp_path / "run", backend=backend) as restored:
        chunks = len(restored.observations.chunks)
        again = restored.wait(timeout=60, poll_interval=0)
        assert len(restored.observations.chunks) == chunks
    checks = AdaptVerificationOptions(name="residual", comparisons=("residual",))
    for candidate in (loaded, again, nwqlib.load_result(again.save(tmp_path / "reopened-result"))):
        candidate.verify(checks=checks)
    assert again.eigenvalue == result.eigenvalue and built == []


def test_shared_index_compiler_blocks_round_trip_through_a_run_checkpoint(tmp_path, monkeypatch):
    """A Run checkpoint saves a shared-index compiler plan and a reopened Plan's map reuses it.

    The restored plan has the same route, active modes and read-only
    occupation blocks, compiles the same generator circuit and shift rule, and
    no compiler plan is built again.
    """
    import json
    from qiskit.quantum_info import Operator
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.algorithms.gcim import adapt, adapt_archive
    from nwqlib.algorithms.gcim.optimization import _generator_shift_rule
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    pool = enumerate_spin_adapted_gsd_pool(3)
    limits = dict(max_bytes=10**9, max_products=10**9)
    live = adapt.CompilerPlans(pool, **limits)
    index = next(i for i in range(len(pool)) if live[i].kind == "shared_index")
    (tmp_path / "run").mkdir()
    saved = adapt_archive.save_context(
        acquisition.AdaptContext(compiler_plans=live), ArchiveFiles(tmp_path / "run", None),
    )
    monkeypatch.setattr(adapt, "_plan_generator_circuit",
                        lambda *a, **k: pytest.fail("a restored compiler plan was built again"))
    reopened = adapt.CompilerPlans(pool, **limits)
    context = adapt_archive.load_context(json.loads(json.dumps(saved)), ArchiveFiles(tmp_path / "run", None))
    reopened.restore(context.compiler_plans)
    plan, restored = live[index], reopened[index]
    assert (restored.kind, restored.active_modes) == (plan.kind, plan.active_modes)
    assert len(restored.occupation_blocks) == len(plan.occupation_blocks) > 0
    for (indices, block), (same, copy) in zip(plan.occupation_blocks, restored.occupation_blocks, strict=True):
        assert indices == same and np.array_equal(block, copy)
        assert not block.flags.writeable and not copy.flags.writeable
    theta = 0.37
    assert np.array_equal(Operator(build_generator_circuit(pool[index], theta, _plan=plan)).data,
                          Operator(build_generator_circuit(pool[index], theta, _plan=restored)).data)
    assert (_generator_shift_rule(pool[index], compilation_plan=plan)
            == _generator_shift_rule(pool[index], compilation_plan=restored))
