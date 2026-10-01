"""Supplied synthetic device profiles for actual one-qubit Plans.

All machine declarations and coefficients are synthetic test data, not
measured calibration. No SDK, pilot, fit or device query runs here.
"""

from datetime import datetime, timezone

from nwqlib import Expectation, plan as select_plan
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.backends import AER_STATEVECTOR_TARGET, InstructionSupport
from nwqlib.backends.assessment import estimate_plan
from nwqlib.backends.capabilities import BackendTarget
from nwqlib.backends.profiles import (
    Allocation,
    CalibrationReference,
    DeviceConfiguration,
    DeviceProfile,
    ModelDomain,
    ModelUncertainty,
    TimeCoefficient,
    TimeModel,
)
from nwqlib.core import Limit, Unit
from nwqlib.core.planning import RuntimeOptions
from nwqlib.core.records import Source
from nwqlib.evidence import Evidence
from nwqlib.operators.inputs import ingest_pauli
from nwqlib.problems.inputs import ingest_occupation
from nwqlib.resources import ResourceContext

SYNTHETIC_SOURCE = Source(
    name="synthetic stored-profile fixture",
    version="1",
    domain="arithmetic fixtures only; not observed hardware calibration",
    reference="tests/_profile_fixtures.py",
)
RECORDED = datetime(2026, 1, 1, tzinfo=timezone.utc)
AT = datetime(2026, 1, 2, tzinfo=timezone.utc)
EXPIRES = datetime(2026, 2, 1, tzinfo=timezone.utc)


def stored_inputs(*, calibrated=False):
    """Create one actual selected HZH preparation and its supplied profile.

    Returns the Plan, its selected realization, profile, allocation, resource
    context, runtime options and assessment time.
    """
    problem = Expectation(
        state=ingest_occupation("1", num_qubits=1),
        observable=ingest_pauli((("Z", 1.0),), num_qubits=1),
    )
    plan = select_plan(problem, method=ExpectationMethod(preparation_choice="hzh"), seed=7)
    realization = plan.resolve("expectation")
    # Declare only the H/Z operations and readout this supplied profile covers.
    # Assessment compares the actual selected HZH workload with that domain.
    target = AER_STATEVECTOR_TARGET.revise(
        description="supplied one-qubit HZH/Pauli subset",
        artifacts=("selected_construction",),
        readouts=("pauli_expectation",),
        max_qubits=1,
        instructions=tuple(InstructionSupport(primitive=gate, max_qubits=1) for gate in ("h", "z")),
        program_nodes=("sequence", "allocate", "block_call"),
    )
    configuration = DeviceConfiguration(
        name="synthetic single-device fixture",
        version="1",
        target=target,
        hardware=SYNTHETIC_SOURCE,
        runtime=Source(
            name=target.name,
            version="fixture-runtime-1",
            domain=target.description,
            reference="synthetic runtime declaration; no installed package queried",
        ),
        build=SYNTHETIC_SOURCE,
        compiler=SYNTHETIC_SOURCE,
        precision="complex128",
        representation="statevector",
    )
    specification = Evidence(kind="external_specification", source=SYNTHETIC_SOURCE)
    # A one-qubit complex128 body needs 32 bytes. This intentionally insufficient
    # 16-byte grant gives a proved metadata failure without allocating a state.
    allocation = Allocation(
        name="fixture grant",
        configuration_id=configuration.content_id,
        locations=("logical_device",),
        topology="single_device",
        limits=(
            Limit(
                stage="execution",
                metric="memory",
                unit=Unit(symbol="byte", dimension="bytes"),
                kind="capacity_stock",
                value=1024 if calibrated else 16,
                scope="logical_device",
            ),
        ),
        recorded_at=RECORDED,
        valid_until=EXPIRES,
        evidence=specification,
    )
    # Tie model validity to configuration, allocation, basis and workload
    # features. Counts outside this domain cannot use its timing coefficients.
    context = ResourceContext()
    domain = ModelDomain(
        configuration_id=configuration.content_id,
        allocation_id=allocation.content_id,
        basis=context.basis,
        batch_schedule=context.batch_schedule,
        runtime="seed_independent",
        acquisition="direct_observation",
        population="unconditional",
        readouts=("pauli_expectation",),
        selection_ids=(),
        primitive_gates=("h", "z"),
        min_qubits=1,
        max_qubits=1,
        min_operations=3,
        max_operations=3,
        max_shots=0,
        max_exact_evaluations=1,
        max_readout_items=1,
        max_resets=0,
        max_measurements=0,
        max_classical_work=0,
        max_adaptive_rounds=0,
    )
    if calibrated:
        # Hand arithmetic: one invocation * 1 s + one exact group * 2 s
        # + three logical operations * 0.5 s = 4.5 s, interval [4.25, 5.0].
        coefficients = (1.0, 2.0, 0.5)
        uncertainty = ModelUncertainty(
            kind="future_run_prediction",
            lower_residual_seconds=-0.25,
            upper_residual_seconds=0.5,
            coverage=0.9,
            source=SYNTHETIC_SOURCE,
        )
        calibration = CalibrationReference(
            source=SYNTHETIC_SOURCE,
            training=SYNTHETIC_SOURCE,
            validation=SYNTHETIC_SOURCE,
            validation_errors=SYNTHETIC_SOURCE,
        )
    else:
        # Engineering assumptions: 0.25 + 0.5 + 3 * 0.125 = 1.125 seconds.
        coefficients = (0.25, 0.5, 0.125)
        uncertainty = calibration = None
    # Assemble the supplied additive time relation. The calibrated branch has
    # synthetic prediction residuals, while the engineering branch has none.
    model = TimeModel(
        name="fixture supplied calibration" if calibrated else "fixture engineering relation",
        kind="calibrated" if calibrated else "engineering",
        scope="selected_acquisition",
        coefficients=tuple(
            TimeCoefficient(feature=feature, seconds_per_unit=value, unit=unit)
            for feature, value, unit in zip(
                ("invocations", "exact_evaluations", "logical_operations"),
                coefficients,
                ("s/invocation", "s/exact_evaluation", "s/logical_operation"),
                strict=True,
            )
        ),
        domain=domain,
        recorded_at=RECORDED,
        valid_until=EXPIRES,
        evidence=Evidence(
            kind="empirical_prediction" if calibrated else "numerical_estimate",
            source=SYNTHETIC_SOURCE,
        ),
        assumptions=(
            "synthetic one-qubit HZH reduced-Pauli acquisition coefficients; no hardware certification",
            "fixed runtime/build/precision and single declared acquisition; seed-independent relation",
        ),
        calibration=calibration,
        uncertainty=uncertainty,
    )
    # Bundle the model with its exact configuration and validity interval.
    # No device observation or coefficient fitting occurs in this function.
    profile = DeviceProfile(
        configuration=configuration,
        recorded_at=RECORDED,
        valid_until=EXPIRES,
        evidence=specification,
        models=(model,),
    )
    return plan, realization, profile, allocation, context, RuntimeOptions(seed=7), AT


def original_forecast(plan, *, model=True):
    """Supply a synthetic 2 + 0.5*shots time relation bound to this Plan and allocation."""
    source = Source(name="injected", version="1", domain="offline test", reference="test:injected")
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    configuration = DeviceConfiguration(
        name="supplied local fixture",
        version="1",
        target=BackendTarget(name="injected", provider="injected"),
        hardware=source,
        runtime=source,
        compiler=source,
        build=source,
        precision="complex128",
        representation="statevector",
    )
    evidence = Evidence(kind="external_specification", source=source)
    allocation = Allocation(
        name="original grant",
        configuration_id=configuration.content_id,
        locations=("logical_device",),
        topology="single_device",
        limits=(),
        recorded_at=at,
        valid_until=at,
        evidence=evidence,
    )
    models = ()
    if model:
        context = ResourceContext()
        domain = ModelDomain(
            configuration_id=configuration.content_id,
            allocation_id=allocation.content_id,
            basis=context.basis,
            batch_schedule=context.batch_schedule,
            runtime="seed_independent",
            acquisition="direct_observation",
            population="unconditional",
            readouts=("counts",),
            selection_ids=(),
            primitive_gates=(),
            min_qubits=1,
            max_qubits=1,
            min_operations=0,
            max_operations=0,
            max_shots=7,
            max_exact_evaluations=0,
            max_readout_items=2,
            max_resets=0,
            max_measurements=1,
            max_classical_work=0,
            max_adaptive_rounds=0,
        )
        models = (
            TimeModel(
                name="supplied arithmetic relation",
                kind="engineering",
                scope="selected_acquisition",
                coefficients=(
                    TimeCoefficient(
                        feature="invocations", seconds_per_unit=2.0, unit="s/invocation"
                    ),
                    TimeCoefficient(feature="sampled_shots", seconds_per_unit=0.5, unit="s/shot"),
                ),
                domain=domain,
                recorded_at=at,
                valid_until=at,
                evidence=Evidence(kind="numerical_estimate", source=source),
                assumptions=("synthetic relation only: 2 + 0.5 times the selected shots",),
            ),
        )
    profile = DeviceProfile(
        configuration=configuration,
        recorded_at=at,
        valid_until=at,
        evidence=evidence,
        models=models,
    )
    forecast = estimate_plan(plan, profile=profile, allocation=allocation, assessed_at=at)
    return forecast, allocation
