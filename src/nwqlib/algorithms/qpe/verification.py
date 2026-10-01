"""Explicit comparison of a QPE Result with the dense spectrum, run only on request.

``result.verify(checks=QPEVerification(...))`` diagonalizes the dense target
once (or reuses the Run's eigensystem), groups numerically close eigenvalues
into clusters, and computes the prepared weight of each cluster with a QR
projector. It then compares the estimate with the cluster the estimator
targets, the lowest cluster for SPE and the cluster of largest prepared
weight otherwise. The comparison is nominal. It neither proves degeneracy
nor identifies the ground state, and it never runs automatically.
"""

from math import fsum, isfinite, pi
import numpy as np
from nwqlib.core.records import Float64, Unit
from nwqlib.evidence import Evidence, Fact
from nwqlib.evidence.error_model import CheckDomain, CheckSpec, ErrorFrame, FramedFact
from nwqlib.evidence.verification import witness_check_facts
from nwqlib.execution import KernelApplication, VerificationReceipt, verification_invocation
from nwqlib.ir import Binding
from nwqlib.problems import Eigenphase
from .records import QPEVerification
from . import numerical


def _groups(values, *, tau, atol):
    """Group numerically degenerate eigenvalues into clusters of indices.

    Hamiltonian eigenvalues are sorted, and a cluster collects values within
    ``atol`` of its first member. Unitary eigenvalues are compared by the
    principal difference of their angles, so a cluster that straddles the
    branch cut at +-pi stays one cluster.
    """
    if tau is not None:
        groups, current, reference = [], [], None
        for index in np.argsort(np.asarray(values, dtype=float)):
            value = float(values[index])
            if reference is None or abs(value - reference) <= atol:
                current.append(int(index))
                if reference is None:
                    reference = value
            else:
                groups.append(current)
                current, reference = [int(index)], value
        if current:
            groups.append(current)
        return groups
    phases = np.angle(values)
    ordered = sorted(range(len(phases)), key=lambda index: float(phases[index]))
    assigned, groups = set(), []
    for index in ordered:
        if index in assigned:
            continue
        group = [
            candidate
            for candidate in ordered
            if candidate not in assigned
            and abs(numerical.angle_principal(float(phases[candidate] - phases[index]))) <= atol
        ]
        assigned.update(group)
        groups.append(group)
    return groups


def _projector_weight(vectors, indices, reference, occupation):
    """Return (raw, weight, tolerance) for the prepared weight of one eigenvalue cluster.

    ``raw`` = ||Q^dagger psi||**2, where Q is the orthonormal basis that QR
    gives for the cluster's eigenvector columns. A raw value within
    ``tolerance`` outside [0, 1] is clipped to ``weight`` in [0, 1], and a
    value beyond it raises ValueError. The tolerance is
    n*eps/(1 - n*eps)*max(1, |psi|**2) with n = 16*D*k for D rows and k
    cluster vectors and eps = 2**-52. Because eps is twice the unit roundoff
    u = 2**-53, the first factor is Higham's gamma_(2n) = 2n*u/(1 - 2n*u)
    (Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM 2002,
    doi:10.1137/1.9780898718027, Lemma 3.1),
    with an untuned factor 16 in n.
    """
    basis, _ = np.linalg.qr(vectors[:, indices])
    projections = (
        basis[occupation, :].conj() if occupation is not None else basis.conj().T @ reference
    )
    raw = float(np.sum(np.abs(projections) ** 2))
    # The D-term inner products, the squared magnitudes, the k-term sum and
    # the orthogonality of Q all add roundoff. The factor 16 is an untuned
    # allowance for them, registered in docs/ENGINEERING_CONSTANTS.md (QPE
    # projector roundoff window). This window admits a probability endpoint
    # and is not an eigensystem-accuracy theorem.
    n = 16 * vectors.shape[0] * max(1, len(indices))
    n_eps = n * np.finfo(float).eps
    if n_eps >= 1:
        raise ValueError("projector roundoff envelope is not informative")
    scale = 1.0 if occupation is not None else fsum(float(abs(value) ** 2) for value in reference)
    tolerance = n_eps / (1 - n_eps) * max(1.0, scale)
    if not isfinite(raw) or raw < -tolerance or raw > 1 + tolerance:
        raise ValueError("nominal projector overlap is outside its probability domain")
    value = 0.0 if raw < 0 else 1.0 if raw > 1 else raw
    return raw, value, tolerance


def _resolution(result):
    """Grid spacing of the Result's actual analysis grid in the output frame.

    SPE scans a span of pi, while QCELS and RFE cover 2*pi. RWPE reports
    its Gaussian phase width. Phase output divides by 2*pi and energy output
    by tau. This resolution is not a total error bound.
    """
    plan = result.plan
    options = plan.method
    if options.estimator == "rwpe":
        scale = 2*pi if isinstance(plan.output, Eigenphase) else plan.reconstruction.tau
        return result.gaussian.standard_deviation/scale
    size = (options.num_frequencies if options.estimator == "rfe" else
            next((item.value for item in result.analysis_settings if item.parameter == "grid_size"),
                 options.grid_size))
    if options.estimator == "qcels" and result.fit is not None:
        size = result.fit.effective_grid
    span = pi if options.estimator == "spe" else 2 * pi
    scale = 2 * pi if isinstance(plan.output, Eigenphase) else plan.reconstruction.tau
    return span / (scale * size)


def verify(plan, result, *, checks):
    """Run one QPEVerification and return ``(receipt, facts)``.

    The Result must belong to ``plan`` and hold an estimate. ``facts`` holds
    the receipt-witnessed ``component_error`` and ``overlap_deficit`` facts,
    in the order of ``checks.verification_checks(result)``, ready for
    ``Certificate.with_verification`` with the same options. The receipt
    keeps every fact that ``_verify`` lists.
    """
    if type(checks) is not QPEVerification:
        raise TypeError("checks requires one explicit QPEVerification")
    result.validate_plan(plan)
    if result.estimator_value is None:
        raise ValueError("verification requires an existing QPE estimate")
    return _verify(plan, result, checks)


# Facts in the Plan's output unit. The other QPE verification facts are
# dimensionless.
_OUTPUT_UNIT_FACTS = frozenset({"component_error", "reference_value", "method_resolution", "component_threshold"})


def _frame(plan, name):
    """Return the ErrorFrame of one verification fact, conditioned on the compared cluster."""
    output_frame = plan.output.frame(plan.problem)
    unit = output_frame.unit if name in _OUTPUT_UNIT_FACTS else Unit(symbol="1", dimension="dimensionless")
    return ErrorFrame(
        quantity=name,
        metric="nominal_" + name,
        unit=unit,
        scope=output_frame.scope,
        conditioning=("lowest spectral cluster under explicit numerical grouping" if plan.method.estimator == "spe"
                      else "largest prepared overlap under explicit numerical clustering; not ground identity"),
    )


def verification_checks(options, result):
    """Return the two CheckSpecs of one QPEVerification.

    ``component_error`` compares the estimate's distance to the compared
    cluster, in the output unit, with ``tolerance``. ``overlap_deficit`` is
    ``max(0, minimum_overlap - weight)`` for the cluster's prepared weight,
    so its threshold 0 passes exactly when the weight reaches the minimum
    overlap.
    """
    plan = result.plan
    result.validate_plan(plan)
    fields = dict(claim_id=plan.output.content_id, source=options.source, options_id=options.content_id,
        prerequisites=(), experiments=0, access=("existing QPE estimate and dense original target",),
        classical_work="one dense eigendecomposition, unless reused, and one QR projector per cluster",
        reference_work=options.source.reference,
        data_description="component error, overlap deficit and their companion facts")
    return (
        CheckSpec(name="component_error", frame=_frame(plan, "component_error"), threshold=options.tolerance,
                  domain=CheckDomain(lower=0.), **fields),
        CheckSpec(name="overlap_deficit", frame=_frame(plan, "overlap_deficit"), threshold=0.,
                  domain=CheckDomain(lower=0., upper=1.), **fields),
    )


def _verify(plan, result, choice):
    """Return a VerificationReceipt comparing the estimate with one cluster of the dense spectrum.

    The reported facts are the component error (circular distance in turns
    for phase output, absolute energy difference otherwise), the overlap
    deficit max(0, threshold - weight), the cluster's reference value, raw
    and clipped weight with its roundoff window, the cluster size and
    count, the method resolution (_resolution) and the two pass flags. Only
    dense input is compared. A supplied-circuit preparation is simulated
    only with ``materialize_preparation=True``.
    """
    from .method import (
        _new_context,
        _check_context,
        _eigensystem,
        _reference_data,
        _scalar_reference_available,
    )

    checks = choice.verification_checks(result)
    data = plan._native["input"]
    if data.target.reference.representation != "dense":
        raise ValueError("nominal QPE verification has no hidden Pauli densification")
    existing = getattr(result.data, "method_context", None)
    context = _new_context() if existing is None else dict(existing)
    kind = plan.reconstruction.target_kind
    # The Plan's selected spectral target: V = polar(A) for unitary input
    # (method._Input.spectral), never the original near-unitary A.
    spectral = data.spectral
    _check_context(context, spectral.reference, kind, data.reference.preparation)
    reused = context["eigensystem"] is not None
    reference_reused = context["reference"] is not None
    if choice.reuse_only and not (reused and reference_reused):
        raise ValueError("reuse_only needs existing matching eigensystem and preparation data")
    state = data.reference
    supplied = state.preparation.implementation == "qiskit.supplied"
    if supplied and not reference_reused and not choice.materialize_preparation:
        raise ValueError("supplied preparation needs explicit materialize_preparation")
    if (
        not supplied
        and not _scalar_reference_available(state)
        and state.preparation.implementation
        not in {"qiskit.direct", "qiskit.occupation", "qiskit.product"}
    ):
        raise ValueError("QPE reference direction is unavailable")
    # Admit reference work before obtaining an eigensystem or materializing
    # a supplied preparation, reusing matching stored data where available.
    # Work: 32*D**3 for the eigendecomposition and the per-cluster QR, and
    # 32*D**2 for the projections. Bytes: an allowance of sixteen D x D
    # complex128 arrays (256*D**2) and eight complex128 D-vectors (128*D).
    dimension = data.target.basis.dimension
    work = 32 * dimension**3 + 32 * dimension**2
    payload = 256 * dimension**2 + 128 * dimension
    if work > choice.max_work or payload > choice.max_bytes:
        raise ValueError("selected QPE reference exceeds max_work or max_bytes")
    (values, vectors), _ = _eigensystem(
        spectral.reference, spectral.dense_array(), kind=kind, context=context
    )
    if not reference_reused:
        if supplied:
            from nwqlib.problems.inputs import prepare_qiskit
            from qiskit.quantum_info import Statevector

            circuit = prepare_qiskit(state, max_bytes=choice.max_bytes).circuit
            reference, occupation = Statevector.from_instruction(circuit).data, None
        else:
            reference, occupation = _reference_data(state)
        context.update(preparation=state.preparation, reference=(reference, occupation))
    reference, occupation = context["reference"]
    # The CDF threshold targets the lowest spectral component. A dominant
    # component reference would assess a different quantity in a mixed state.
    groups = _groups(values, tau=data.tau, atol=choice.group_atol)
    candidates = []
    for group in groups:
        raw, weight, window = _projector_weight(vectors, group, reference, occupation)
        estimate = (
            float(np.mean(values[group]))
            if data.tau is not None
            else numerical.angle_principal(
                float(np.angle(np.mean(values[group] / np.abs(values[group]))))
            )
        )
        candidates.append((weight, raw, window, estimate, len(group)))
    lowest = plan.method.estimator == "spe"
    weight, raw, window, value, cluster_size = (
        min(candidates, key=lambda item: item[3]) if lowest
        else max(candidates, key=lambda item: item[0]))
    threshold = choice.minimum_overlap
    if threshold is None:
        threshold = plan.method.overlap_lower_bound if lowest else .9
    # Use circular distance for phase turns and ordinary absolute distance
    # for energy, so a phase crossing zero is not a spurious large error.
    if isinstance(plan.output, Eigenphase):
        reference_value = (
            (-data.tau * value / (2 * pi)) % 1.0
            if data.tau is not None
            else (value / (2 * pi)) % 1.0
        )
        difference = abs(result.phase - reference_value)
        error = min(difference, 1 - difference)
    else:
        reference_value = value
        error = abs(result.eigenvalue - value)
    deficit = max(0.0, threshold - weight)
    items = dict(
        component_error=error,
        overlap_deficit=deficit,
        reference_value=reference_value,
        raw_overlap=raw,
        valid_overlap=weight,
        overlap_adjustment=weight - raw,
        roundoff_window=window,
        cluster_dimension=cluster_size,
        cluster_count=len(groups),
        method_resolution=_resolution(result),
        component_threshold=choice.tolerance,
        component_within_tolerance=float(error <= choice.tolerance),
        overlap_threshold=threshold,
        overlap_sufficient=float(deficit == 0),
    )
    facts = []
    for name, value in items.items():
        frame = _frame(plan, name)
        facts.append(
            FramedFact(
                frame=frame,
                bindings=(),
                fact=Fact(
                    quantity=name,
                    unit=frame.unit,
                    scope=frame.scope,
                    availability="concrete",
                    value=Float64(value=float(value)),
                    evidence=Evidence(kind="numerical_estimate", source=choice.source),
                ),
            )
        )
    arguments = dict(
        eigensystems=int(not reused),
        spectral_reuses=int(reused),
        reference_reuses=int(reference_reused),
        reference_setups=int(not reference_reused),
        qr_clusters=len(groups),
        preparation_materializations=int(supplied and not reference_reused),
        dimension=dimension,
        known_work=work,
        known_bytes=payload,
    )
    application = KernelApplication(
        name="nominal_cluster",
        implementation=choice.source,
        arguments=tuple(Binding(parameter=name, value=value) for name, value in arguments.items()),
        facts=tuple(facts),
    )
    receipt = VerificationReceipt(
        invocation_id=verification_invocation(),
        plan_id=plan.content_id,
        result_id=result.content_id,
        construction_id=result.construction_id,
        artifact_ids=(),
        reference=choice.source,
        options_id=choice.content_id,
        applications=(application,),
    )
    return receipt, witness_check_facts(receipt, result, checks)
