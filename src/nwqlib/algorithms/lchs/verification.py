"""Explicit physical-vector reference comparisons for the actual LCHS Result."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import isfinite
from sys import float_info
from typing import Annotated, Literal

from pydantic import Field, model_validator

from nwqlib.core.planning import Plan
from nwqlib.core.records import Float64, Nonnegative, PositiveInt, Real, Record, Source, Text, Unit
from nwqlib.evidence.error_model import CheckDomain, CheckSpec, FramedFact
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.execution import KernelApplication, VerificationReceipt, verification_invocation
from nwqlib.ir import Binding
from nwqlib.operators.access import _check_bytes
from nwqlib.problems import Solution, StateVector
from .primary_records import LCHSAnalysis
from .references import dense_work


class LCHSVerification(Record):
    """Reference comparison for the physical solution of an `LCHS` Result.

    Build it with keyword arguments and pass it to
    `result.verify(checks=...)`, for example
    `result.verify(checks=LCHSVerification(reference="expm"))`. `reference`
    is the only required argument. The call returns `(receipt, facts)`.
    `facts[0].fact.value.value` is the discrepancy between the Result's
    physical solution u and the reference `u_ref`, and the receipt records
    every computed value and call count. The check is named `name`, and
    `"ivp_closed_form"` adds `name + ".reference_consistency"`.
    `Certificate.with_verification` turns `facts` into PASS when the value
    is at most `threshold`, FAIL when it is larger, and INCONCLUSIVE when
    the value or the threshold is missing.
    Planning, solving and ordinary analysis never compute a reference.

    The Result must hold a physical solution vector with its phase, from a
    `Solution` output or a physical `StateVector`, and A must be a dense
    matrix, never converted from a compact input. The work limits cover the
    whole call, including both references of `"ivp_closed_form"` and every
    node of `"selected_grid"`. Dense work is a size-based count, not a
    floating-point operation count. Reference arrays are temporary, and only
    discrepancies and call counts are kept. A discrepancy is numerical
    evidence for this input, not a proven error bound. The
    [LCHS guide](../../algorithms/lchs.md#explicit-checks-and-saved-results)
    shows a complete check.

    Attributes:
        reference: Required. `"expm"` compares with `exp(-A T) u0` for
            dynamics without a source (An, Childs and Lin,
            arXiv:2312.03916v2, here ACL, Eq. (2) with b = 0). `"ivp"` solves `du/dt = -A u + b` with SciPy's RK45
            under explicit `rtol`, `atol` and `max_rhs_calls`.
            `"closed_form"` evaluates ACL Eq. (2) for a constant source with
            one exponential of `[[-A*T, b*T], [0, 0]]`. `"ivp_closed_form"`
            runs both and also reports their signed consistency. These three
            need a source. A matrix with 1-norm above `2**37`, or with an
            exponential that is not finite in binary64, gives an unknown
            discrepancy. `"selected_grid"` reproduces the saved finite sum,
            with its nodes and product-formula steps, so its discrepancy
            excludes the kernel-integral, k- and Duhamel-quadrature and
            product-formula errors.
        name: Default `"reference_error"`. Name of the reported discrepancy
            and prefix of its companion values.
        metric: Default `"absolute_l2"`, the norm `||u - u_ref||_2` without a
            global-phase fit. `"relative_l2"` divides it by `||u_ref||_2`.
        threshold: Default `None`. Optional nonnegative pass threshold of the
            discrepancy check.
        consistency_threshold: Default `None`. Optional nonnegative threshold
            of the consistency check between the IVP and closed-form
            references, only for `"ivp_closed_form"`.
        rtol: Default `None`. Positive RK45 relative tolerance, at least
            SciPy's floor of 100 times the binary64 machine epsilon. Required
            for `"ivp"` and `"ivp_closed_form"` and refused otherwise.
        atol: Default `None`. Positive RK45 absolute tolerance. Required for
            `"ivp"` and `"ivp_closed_form"` and refused otherwise.
        max_rhs_calls: Default `None`. Upper limit on RK45 right-hand-side
            evaluations. Required for `"ivp"` and `"ivp_closed_form"` and
            refused otherwise.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Upper
            limit on the known arrays of this call.
        max_dense_work: Default `1e8` (`100_000_000`). Upper limit on the
            counted dense work of this call. For `"selected_grid"` it also
            covers reading the stored product-formula node table and the
            node actions.
        max_node_evaluations: Default `4096`. Upper limit on the number of
            k nodes times operator applications evaluated.
        max_pf_operations: Default `100_000`. Upper limit on the
            product-formula rotations or elementary gates that
            `"selected_grid"` evaluates.

    Raises:
        ValueError: If an IVP reference lacks `rtol`, `atol` or
            `max_rhs_calls`, if `rtol` is below SciPy's floor, if these are
            given for another reference, or if `consistency_threshold` is
            given without `"ivp_closed_form"`. `result.verify` also raises
            for `"expm"` with a source, for the source references without
            one, and for a non-dense A.
    """

    reference: Literal["expm", "ivp", "closed_form", "ivp_closed_form", "selected_grid"]
    name: Text = "reference_error"
    metric: Literal["absolute_l2", "relative_l2"] = "absolute_l2"
    threshold: Nonnegative | None = None
    consistency_threshold: Nonnegative | None = None
    rtol: Annotated[Real, Field(gt=0)] | None = None
    atol: Annotated[Real, Field(gt=0)] | None = None
    max_rhs_calls: PositiveInt | None = None
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_dense_work: PositiveInt = 100_000_000
    max_node_evaluations: PositiveInt = 4096
    max_pf_operations: PositiveInt = 100_000

    @model_validator(mode="after")
    def _reference_parameters(self):
        ivp = self.reference in {"ivp", "ivp_closed_form"}
        if ivp and any(value is None for value in (self.rtol, self.atol, self.max_rhs_calls)):
            raise ValueError("IVP verification requires explicit rtol, atol and max_rhs_calls")
        # SciPy RK45 clamps smaller rtol to 100*eps. Reject beforehand so the
        # receipt always records the tolerance actually used by that solver.
        if ivp and self.rtol < 100*float_info.epsilon:
            raise ValueError("RK45 rtol is below SciPy's float64 floor")
        if not ivp and any(value is not None for value in (self.rtol, self.atol, self.max_rhs_calls)):
            raise ValueError("RHS parameters belong only to an explicit IVP reference")
        if self.reference != "ivp_closed_form" and self.consistency_threshold is not None:
            raise ValueError("consistency_threshold requires ivp_closed_form")
        return self

    @property
    def source(self):
        references = dict(expm="ACL arXiv:2312.03916v2 Eq.(2), b=0: expm(-A*T) @ u0",
            ivp="ACL arXiv:2312.03916v2 Eq.(1); scipy.integrate.solve_ivp RK45 with explicit "
                "rtol/atol/RHS cap",
            closed_form="ACL arXiv:2312.03916v2 Eq.(2), constant A,b: expm(-A*T)u0 "
                        "+ integral_0^T expm(-A*s) ds b from one expm([[-A*T,b*T],[0,0]])",
            ivp_closed_form="explicit RK45 and constant-source closed form; signed consistency",
            selected_grid="ACL arXiv:2312.03916v2 Eqs.(2),(60)-(61); actual selected quadrature/PF schedule, dense exact nodes for QSP")
        return Source(name="lchs.reference."+self.reference, version="2",
            domain="explicit binary64 physical reference; vendor workspace and rounding error unbounded",
            reference=references[self.reference])

    def validate_domain(self, problem):
        if self.reference == "expm" and problem.source is not None:
            raise ValueError("expm verification requires homogeneous dynamics")
        if self.reference in {"ivp", "closed_form", "ivp_closed_form"} and problem.source is None:
            raise ValueError("the selected reference requires an explicit constant source")
        if problem.A.reference.representation != "dense":
            raise ValueError("LCHS vector references require existing dense physical inputs; no implicit densification")

    def verification_checks(self, result):
        """Declare the discrepancy check, and for ivp_closed_form also the consistency check.

        Each CheckSpec carries these options' identity, so its facts answer
        only this reference, metric and threshold.
        """
        _validate_result(result.plan, result)
        return self._check_specs(result.plan)

    def _check_specs(self, plan):
        """Return the CheckSpecs of verification_checks for a Result already validated against plan."""
        self.validate_domain(plan.problem)
        frame = _physical_frame(plan, self.metric)
        fields = dict(claim_id=plan.output.content_id, source=self.source, options_id=self.content_id,
            prerequisites=(), access=("existing phase-faithful physical solution vector",), experiments=0,
            classical_work="signed discrepancy and reference norm on original coordinates",
            reference_work=self.source.reference, data_description="scalar discrepancies and numerical call counts",
            domain=CheckDomain(lower=0.))
        checks = (CheckSpec(name=self.name, frame=frame, threshold=self.threshold, **fields),)
        if self.reference == "ivp_closed_form":
            checks += (CheckSpec(name=self.name+".reference_consistency", frame=_consistency_frame(plan),
                threshold=self.consistency_threshold, **fields),)
        return checks


def _validate_result(plan, result):
    from .method import LCHS
    if type(plan) is not Plan or type(plan.method) is not LCHS or type(result) is not LCHSAnalysis:
        raise TypeError("verification requires the actual LCHS Plan and Result")
    if result.plan is not plan:
        raise ValueError("verification requires the Result's original attached Plan")
    result.validate_plan(plan)
    from nwqlib.saved_evidence import _validate_data
    _validate_data(plan, result.data)
    if tuple(application for chunk in result.data.observations.chunks for application in chunk.applications) != result.applications:
        raise ValueError("LCHS Result applications differ from its actual acquisition")


def _target(plan, result, checks):
    """Resolve phase-faithful original coordinates before any reference call."""
    output = plan.output
    physical = isinstance(output, Solution) or (isinstance(output, StateVector)
        and output.normalization == "physical" and output.global_phase == "physical")
    if not physical:
        raise ValueError("verification requires an existing phase-faithful physical solution vector")
    d = plan.problem.dimension
    _check_bytes(64*d, checks.max_bytes, "LCHS verification physical target")
    manifest = result.artifact
    if manifest is None:
        if plan.reconstruction.mode not in {"initial", "zero"} or result.data.observations.chunks:
            raise ValueError("verification requires an existing phase-faithful physical solution vector")
        return (result.solution if isinstance(output, Solution) else result.state_vector), None
    if (manifest.plan_id != plan.content_id or manifest.construction_id != plan.construction.content_id
            or manifest.output.kind != "vector" or manifest.output.basis != plan.problem.basis
            or manifest.output.frame != "physical" or manifest.output.global_phase != "physical"):
        raise ValueError("verification target differs from the selected physical solution artifact")
    matching = tuple(chunk for chunk in result.data.observations.chunks if manifest in chunk.artifacts)
    if len(matching) != 1:
        raise ValueError("verification artifact needs its one actual producing acquisition")
    chunk = matching[0]
    if manifest.realization_id != chunk.realization_id:
        raise ValueError("verification artifact differs from its selected realization")
    point = plan.resolve(chunk.experiment)
    if plan.execution == "classical":
        kernel, = plan.construction.kernels
        if (manifest.producer_id != kernel.content_id or manifest.output not in kernel.outputs
                or manifest.source != kernel.implementation):
            raise ValueError("verification artifact differs from its selected host producer")
    else:
        declaration = point.resolved_observation(plan)[1].amplitudes
        if (declaration is None or manifest.producer_id != declaration.content_id
                or manifest.output != declaration.output or manifest.source != declaration.source):
            raise ValueError("verification artifact differs from its selected amplitude producer")
    array = result.data.artifact(manifest).array
    if array.shape != (d,):
        raise ValueError("verification artifact differs from the original physical dimension")
    return array, manifest


def _physical_frame(plan, metric="absolute_l2"):
    frame = Solution().frame(plan.problem)
    return frame.revise(metric=metric, unit=Unit(symbol="1", dimension="dimensionless")
        if metric == "relative_l2" else frame.unit)


def _consistency_frame(plan):
    return _physical_frame(plan, "relative_l2").revise(quantity="physical_reference_consistency",
        conditioning="signed physical IVP and closed-form references for the same original inputs")


def _admit_publication(checks, frames, arguments):
    """Admit the bytes of the facts and receipt this check will publish.

    The admission runs before any reference work, so an oversized publication
    is refused without spending that work.
    """
    from nwqlib._run_journal import _json_bound
    # Two fact copies, one receipt and their nested source/frame identities
    # are below sixteen portable projections of these actual scalar inputs.
    # The fixed 8192 bytes cover the record envelope that does not scale with
    # the inputs (registered in ENGINEERING_CONSTANTS).
    size = 16*_json_bound((checks, frames, arguments), checks.max_bytes) + 8192
    _check_bytes(size, checks.max_bytes, "LCHS verification scalar output")


def _publish(plan, result, checks, manifest, metrics, arguments, *, invocation_id, specs,
             witnessed_names=None, ideal_bounds=()):
    """Publish actual scalars, counts and source identity without numerical work.

    ``specs`` are the CheckSpecs the caller declared for the Result it
    validated at entry, so publication does not validate it again.
    """
    source = checks.source
    raw = []
    for name, frame, value, reason in metrics:
        fields = dict(quantity=name, unit=frame.unit, scope=frame.scope)
        if value is None or not isfinite(value):
            fields.update(availability="unknown", reason=reason)
        else:
            if value < 0:
                raise ValueError("LCHS norm, discrepancy and bound facts must be nonnegative")
            fields.update(availability="concrete", value=Float64(value=float(value)),
                evidence=Evidence(kind="numerical_estimate", source=source))
        raw.append(FramedFact(frame=frame, bindings=plan.construction.program.bindings,
            failure_probability=0. if name in ideal_bounds and value is not None and isfinite(value) else None,
            fact=Fact(**fields)))
    bindings = tuple(Binding(parameter=name, value=Float64(value=value) if type(value) is float else value)
                     for name, value in arguments)
    application = KernelApplication(name=checks.name, implementation=source, arguments=bindings, facts=tuple(raw))
    receipt = VerificationReceipt(invocation_id=invocation_id, plan_id=plan.content_id, result_id=result.content_id,
        construction_id=plan.construction.content_id, artifact_ids=() if manifest is None else (manifest.content_id,),
        reference=source, options_id=checks.content_id, applications=(application,))
    # A fact that answers a selected check names that exact CheckSpec. Output-error
    # components answer no check and carry only the options identity.
    specs = {spec.name: spec.content_id for spec in specs}
    witnessed = []
    for selected in raw:
        if witnessed_names is not None and selected.fact.quantity not in witnessed_names:
            continue
        if selected.fact.availability != "concrete":
            witnessed.append(selected)
            continue
        evidence = Evidence(kind="numerical_estimate", source=source, status="witnessed", artifact_kind="verification_receipt",
            artifact=receipt.content_id, subject_id=result.content_id, witnessed_scope=selected.frame.scope,
            options_id=receipt.options_id, check_id=specs.get(selected.fact.quantity))
        witnessed.append(selected.revise(fact=selected.fact.revise(evidence=evidence)))
    return receipt, tuple(witnessed)


def _norm(vector, checks, counts):
    """Return the L2 norm, or None when it overflows or underflows to zero.

    A zero result for a nonzero vector would report a representable
    discrepancy as exact zero, so it is unavailable instead.
    """
    import numpy as np
    from nwqlib._numerics import stable_vector_norm

    dense_work(checks, counts, work=2*len(vector), peak_bytes=64*len(vector))
    counts["norm_attempts"] += 1
    value = stable_vector_norm(vector)
    counts["norm_completed"] += 1
    return value if isfinite(value) and (value != 0 or not np.any(vector)) else None


def _compare(target, reference, checks, counts):
    """Return (||target - reference||_2, ||reference||_2), each None if unavailable."""
    import numpy as np
    if reference is None:
        return None, None
    if not isinstance(reference, np.ndarray) or reference.shape != target.shape:
        raise ValueError("reference returned a different physical solution shape")
    norm = _norm(reference, checks, counts)
    dense_work(checks, counts, work=len(target), peak_bytes=64*len(target))
    # Definition: absolute discrepancy is ||target-reference||_2, without a
    # global-phase fit. Relative discrepancy divides by this reference's norm.
    with np.errstate(over="ignore", invalid="ignore"):
        difference = target-reference
    counts["vector_differences"] += 1
    return _norm(difference, checks, counts), norm


def _relative(absolute, norm):
    """Return absolute/norm, or None when undefined, nonfinite or underflowed."""
    value = None if absolute is None or norm is None or norm == 0 else absolute/norm
    return value if value is None or (isfinite(value) and (value != 0 or absolute == 0)) else None


def _arguments(checks, elapsed, dimension, counts, *, reference_dimension):
    """Return the recorded (name, value) arguments of this reference call."""
    arguments = [("elapsed_time", elapsed), ("dimension", dimension), ("reference_dimension", reference_dimension)]
    if checks.reference in {"ivp", "ivp_closed_form"}:
        arguments += [("rtol", checks.rtol), ("atol", checks.atol), ("max_rhs_calls", checks.max_rhs_calls)]
    return arguments + list(counts.items())


def verify(plan, result, *, checks):
    """Run one explicitly selected bounded reference comparison in the original physical solution
    frame.

    The target is the Result's own phase-faithful physical vector, traced to
    its producing acquisition, so the comparison can never use a vector from
    another Plan or realization. Domain checks and the publication size
    admission run before the reference is computed. The reference is never
    part of planning or ordinary analysis. A reference whose matrix
    exponential is unavailable (references._exponential) or an
    unrepresentable selected-grid reference is reported as unknown, and a
    failed IVP solve raises. None of them is replaced by another reference.
    """
    from .references import closed_form_reference, expm_reference, ivp_reference
    if type(checks) is not LCHSVerification:
        raise TypeError("checks requires one explicit LCHSVerification")
    _validate_result(plan, result)
    target, manifest = _target(plan, result, checks)
    checks.validate_domain(plan.problem)
    selected_checks = checks._check_specs(plan)
    grid = None
    if checks.reference == "selected_grid" and plan.reconstruction.mode not in {"initial", "zero"}:
        from .selected_grid import selected_grid
        grid = selected_grid(plan, result, checks=checks)
    d, elapsed = plan.problem.dimension, plan.problem.elapsed_time
    _check_bytes(16*(d*d+6*d), checks.max_bytes, "LCHS original reference inputs")
    matrix = plan.problem.A.dense_array()
    initial = plan.problem.initial_state.physical_vector()
    source = None if plan.problem.source is None else plan.problem.source.physical_vector()
    counts = dict(expm_attempts=0, expm_completed=0, eigh_calls=0, matvec_attempts=0, matvec_completed=0,
        norm_attempts=0, norm_completed=0, vector_differences=0, ivp_attempts=0, ivp_completed=0,
        rhs_attempts=0, rhs_completed=0, dense_work=0, known_peak_bytes=0)
    if grid is not None:
        counts.update(fixed_pf_attempts=0, fixed_pf_completed=0, pauli_decomposition_attempts=0,
                      pauli_decomposition_completed=0, pf_operations=0, stored_node_read_work=0)
    absolute_frame = _physical_frame(plan)
    metrics = [(checks.name, selected_checks[0].frame, None, "reference discrepancy is unavailable")]
    if checks.metric == "relative_l2":
        metrics.append((checks.name+".absolute_l2", absolute_frame, None, "absolute discrepancy is unavailable"))
    metrics.append((checks.name+".reference_norm", absolute_frame.revise(metric="l2"), None, "reference norm is unavailable"))
    if checks.reference == "ivp_closed_form":
        name = checks.name+".reference_consistency"
        metrics += [(name, selected_checks[1].frame, None, "reference consistency is unavailable"),
            (name+".absolute_l2", absolute_frame, None, "reference consistency is unavailable"),
            (name+".reference_norm", absolute_frame.revise(metric="l2"), None, "closed-form norm is unavailable")]
    reference_dimension = d if grid is None else plan.reconstruction.encoded_dimension
    arguments = _arguments(checks, elapsed, d, {name:checks.max_dense_work for name in counts},
                           reference_dimension=reference_dimension)
    if grid is not None:
        from .selected_grid import grid_arguments
        arguments += grid_arguments(grid)
    _admit_publication(checks, tuple((name, frame, reason) for name, frame, _, reason in metrics), arguments)
    # Select the requested reference only after admitting its output/work
    # record. The reference choice determines which extra numerical work runs.
    invocation_id = verification_invocation()
    closed = None
    if grid is not None:
        from .selected_grid import grid_reference
        reference = grid_reference(
            plan, grid, checks=checks, counts=counts,
            held_inputs=(matrix, initial, source, target),
        )
    elif checks.reference == "selected_grid":
        reference = initial
    elif checks.reference == "expm":
        reference = expm_reference(matrix, initial, elapsed, checks=checks, counts=counts)
    elif checks.reference == "closed_form":
        reference = closed_form_reference(matrix, initial, source, elapsed, checks=checks, counts=counts)
    else:
        reference = ivp_reference(matrix, initial, source, elapsed, checks=checks, counts=counts)
        if checks.reference == "ivp_closed_form":
            closed = closed_form_reference(matrix, initial, source, elapsed, checks=checks, counts=counts)
    absolute, norm = _compare(target, reference, checks, counts)
    value = absolute if checks.metric == "absolute_l2" else _relative(absolute, norm)
    reason = ("selected-grid reference is not representable in binary64" if reference is None and grid is not None else
        "reference matrix exponential is unavailable: its 1-norm exceeds 2**37 or it is not finite in binary64"
        if reference is None else
        "relative discrepancy requires a nonzero representable reference norm and ratio" if checks.metric == "relative_l2" else
        "reference discrepancy is nonfinite or unrepresentable")
    values = {checks.name:value, checks.name+".absolute_l2":absolute, checks.name+".reference_norm":norm}
    if checks.reference == "ivp_closed_form":
        cross_absolute, cross_norm = _compare(reference, closed, checks, counts)
        name = checks.name+".reference_consistency"
        values.update({name:_relative(cross_absolute, cross_norm), name+".absolute_l2":cross_absolute,
                       name+".reference_norm":cross_norm})
    metrics = [(name, frame, values[name], reason if name == checks.name else missing)
               for name, frame, _, missing in metrics]
    arguments = _arguments(checks, elapsed, d, counts, reference_dimension=reference_dimension)
    if grid is not None:
        arguments += grid_arguments(grid)
    return _publish(plan, result, checks, manifest, metrics, arguments, invocation_id=invocation_id,
                    specs=selected_checks,
                    witnessed_names=tuple(check.name for check in selected_checks))
