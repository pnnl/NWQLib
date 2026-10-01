"""Explicit selected component bounds, without acquiring a solution vector."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import fsum, isfinite
from typing import Literal

import numpy as np
from pydantic import StrictBool, model_validator

from nwqlib.core.records import Float64, PositiveInt, Record, Source, Text, Unit
from nwqlib.evidence.error_model import ErrorFrame
from nwqlib.execution import verification_invocation
from nwqlib.operators.access import _check_bytes
from nwqlib.problems import Solution, QuadraticForm, NormalizedExpectation
from .references import dense_work
from .verification import _admit_publication, _publish, _validate_result


class LCHSRefinement(Record):
    """Acquire named bounds at the already selected quadrature/PF schedule.

    Dense validation evaluates dense commutator norms only when requested.
    It checks the structural relation and does not replace that relation.

    Attributes:
        components: Distinct requested analyses. ``spectral_norms`` evaluates
            ||A||, ||A + shift*I|| and ||H||. ``duhamel`` evaluates the
            source-time Gauss-Legendre remainder. ``fixed_pf`` evaluates the
            product-formula commutator bound at the saved step counts.
        name: Prefix of the published quantities that answer no error term.
        dense_validation: Also evaluate dense commutator norms for fixed_pf,
            as a check of the Pauli-triangle bound.
        max_bytes: Cap on known arrays of this call.
        max_dense_work: Cap on counted dense work units of this call.
        max_structural_work: Cap on counted stored-table read, census preparation and Pauli
            commutator work of this call.
        max_node_evaluations: Cap on selected k-node times application evaluations.
        max_steps: Largest saved product-formula step count this call accepts.
    """

    components: tuple[Literal["spectral_norms", "duhamel", "fixed_pf"], ...]
    name: Text = "refinement"
    dense_validation: StrictBool = False
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_dense_work: PositiveInt = 100_000_000
    max_structural_work: PositiveInt = 100_000_000
    max_node_evaluations: PositiveInt = 4096
    max_steps: PositiveInt = 100_000

    @model_validator(mode="after")
    def _components(self):
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("refinement requires nonempty distinct components")
        if self.dense_validation and "fixed_pf" not in self.components:
            raise ValueError("dense_validation requires explicit fixed_pf refinement")
        return self

    @property
    def source(self):
        return Source(name="lchs.refinement", version="2",
            domain="selected ideal physical component relations evaluated numerically; rounding remains separate",
            reference="ACL arXiv:2312.03916v2 Eq.(2); DLMF3.5.19/21; CSTWZ PRX11,011020 doi:10.1103/PhysRevX.11.011020 Props9-10, Eqs120-121")

    def verification_checks(self, result):
        """Declare no checks, since refinement publishes component bounds rather than a pass test."""
        _validate_result(result.plan, result)
        return ()


def _spectral_norm(matrix, checks, counts):
    """Return ||matrix||_2, or None when a nonzero matrix gives a nonfinite or zero norm.

    The 4*d*d entrywise scan and scaled copy are admitted first, and the
    singular values (_linalg_laws.singular_values_work) only for a nonzero
    matrix. An exactly zero matrix has norm zero.
    """
    from nwqlib._linalg_laws import singular_values_work
    from .inhomogeneous_theory import _scaling_safe_spectral_norm
    d = len(matrix)
    dense_work(checks, counts, work=4*d*d, peak_bytes=64*d*d)
    nonzero = bool(np.any(matrix))
    if nonzero:
        dense_work(checks, counts, work=singular_values_work(d), peak_bytes=64*d*d)
    counts["matrix_norm_attempts"] += 1
    counts["spectral_svd_attempts"] += int(nonzero)
    value = _scaling_safe_spectral_norm(matrix)
    counts["matrix_norm_completed"] += 1
    counts["spectral_svd_completed"] += int(nonzero)
    return value if isfinite(value) and (value != 0 or not nonzero) else None


def _component_values(plan, result, payload):
    """Collect the existing physical component facts that refinement may replace.

    Only listed_only error terms are read, and each must keep its planned
    physical frame, so a refined value cannot enter the wrong output frame.
    """
    # Result facts contain actual host application sums. Quantum construction
    # facts already belong to its selected numerical recipe. Never recompute
    # either owner merely to obtain a component for this later propagation.
    if plan.execution == "classical" and payload is not None:
        from .selected_grid import host_applications
        host_applications(plan, result, payload)
    supplied = {fact.fact.quantity:fact for fact in (*plan.facts, *result.facts)}
    values = {}
    for term in plan.error_model.terms:
        if term.role != "listed_only":
            continue
        fact = supplied.get(term.name, term.fact)
        if fact.frame != term.fact.frame:
            raise ValueError("refinement component differs from its physical frame")
        values[term.name] = fact.fact.value.value if isinstance(fact.fact.value, Float64) else None
    return values


def _fixed_pf(plan, grid, checks, counts):
    """Bound product-formula synthesis at the saved per-node steps of each application.

    Each application gets sum_j |c_j|*(B_p(t)/r_j**p + t*pruned_j) at its own
    elapsed time t, where B_p(t) is the order-p commutator bound of Childs et
    al., Phys. Rev. X 11, 011020, doi:10.1103/PhysRevX.11.011020,
    Propositions 9-10 (see
    time_independent_terms._fixed_trotter_certificate_records). It is then
    weighted by its physical input and PSD recovery. Step counts are read
    from the Plan's host actions (classical) or the selected SELECT
    (quantum), never reselected. Returns the bound, an unavailability
    reason, and the optional dense validation values per application.
    """
    from .time_independent_terms import (stored_pf_nodes,
        _fixed_trotter_certificate_records, _trotter_budget_quadrature_terms)
    from .solution_error_budget import _build_lchs_application_record, application_stage_bounds

    d = plan.reconstruction.encoded_dimension
    if any(step > checks.max_steps for app in grid.applications for step in app.steps):
        raise ValueError("fixed-PF refinement exceeds max_steps before work")
    active = tuple(app.weight != 0 and app.input_norm != 0 for app in grid.applications)
    if any(app.input_norm is None for app, on in zip(grid.applications, active, strict=True) if on):
        return None, "physical input scale is not representable", [None]*len(grid.applications)
    applications, dense_values = [], []
    raw = {"psd_shift":grid.shift}
    # The saved node table is read once per distinct node, before the
    # application loop, and evaluated at each application's saved
    # (elapsed_time, step_count); no Pauli decomposition runs. The grid's
    # dense L and H (32*d**2 bytes) stay live beside it.
    selected = [app for app, on in zip(grid.applications, active, strict=True) if on]
    certified = iter(_fixed_trotter_certificate_records(nodes=stored_pf_nodes(grid.payload),
        applications=tuple((app.elapsed, app.steps) for app in selected), method=grid.payload.method,
        dense_validation=checks.dense_validation, max_bytes=checks.max_bytes,
        max_dense_work=checks.max_dense_work, max_structural_work=checks.max_structural_work,
        max_steps=checks.max_steps, counts=counts, held_bytes=32*d*d) if selected else ())
    for app, on in zip(grid.applications, active, strict=True):
        if on:
            records = next(certified)
            synthesis = _trotter_budget_quadrature_terms(records, coefficients=grid.coefficients,
                method=grid.payload.method)["trotter_synthesis_error_bound"]
            dense = None if not checks.dense_validation else fsum(abs(coefficient)*record.dense_bound_value
                for coefficient, record in zip(grid.coefficients, records, strict=True))
        else:
            synthesis = 0.
            dense = 0. if checks.dense_validation else None
        # Triangle inequality on ACL arXiv:2312.03916v2 Eq.(2)/(60):
        # sum_j |c_j| epsilon_j, then this application's
        # |weight|*||input||*exp(shift*elapsed).
        applications.append(_build_lchs_application_record(raw, final_time=plan.problem.elapsed_time,
            start_time=app.start, weight=abs(app.weight), input_norm=app.input_norm or 0.,
            backend="trotter", synthesis_error_bound=synthesis))
        dense_values.append(dense)
    values = application_stage_bounds(raw, applications=tuple(applications), backend="trotter",
        psd_premise_satisfied=grid.payload.quadrature.numerical_psd_premise_satisfied)
    bound, reason = values["trotter_synthesis"]
    return bound, reason, dense_values


def refine(plan, result, *, checks):
    """Evaluate requested component bounds for an existing LCHS Result.

    Refinement answers what the selected construction's error components
    are, at the saved quadrature, Duhamel nodes and step counts. It never
    acquires a solution vector, reselects a grid or steps, or densifies a
    compact input. All published quantities, their frames and the argument
    record are admitted against max_bytes before any numerical work, and each
    numerical kernel is admitted against its own work cap before it runs.

    Refined duhamel_quadrature and trotter_synthesis values replace the
    planned components of the same name. The complete set is then propagated
    once into the requested output frame as algorithmic_approximation
    (solution_error_budget.propagate_physical_error). That value stays None
    while any component is unknown, because a missing term is not zero.
    Spectral norms and dense diagnostics are published under the checks name
    and answer no error term.

    Returns:
        A VerificationReceipt holding every observation, and a tuple with the
        refined physical-frame components and the output-frame
        algorithmic_approximation fact, which are the facts suitable for
        assessment. A request for spectral norms alone leaves that tuple empty.
    """
    from .selected_grid import selected_grid, selected_payload, grid_arguments
    from .inhomogeneous_theory import _duhamel_quadrature_error_bound
    from .solution_error_budget import propagate_physical_error
    from .periodic import _PeriodicPayload

    if type(checks) is not LCHSRefinement:
        raise TypeError("checks requires one explicit LCHSRefinement")
    _validate_result(plan, result)
    rec, problem = plan.reconstruction, plan.problem
    fixed = "fixed_pf" in checks.components
    if fixed and plan.method.hamiltonian_evolution_backend not in {"trotter", "trotter_error_budgeted"}:
        raise ValueError("fixed_pf refinement requires the selected product-formula realization")
    if "spectral_norms" in checks.components and problem.A.reference.representation != "dense":
        raise ValueError("spectral_norms requires existing dense physical input; no implicit densification")
    initial = rec.mode in {"initial", "zero"}
    payload = None if initial else selected_payload(plan)
    periodic = type(payload) is _PeriodicPayload
    if periodic and checks.dense_validation:
        raise ValueError("periodic refinement keeps its scalar Strang bound; dense commutator validation is unsupported")
    grid = selected_grid(plan, result, checks=checks) if fixed and not periodic and not initial else None
    physical = Solution().frame(problem)
    values = _component_values(plan, result, payload)
    names = tuple(name for name, requested in (("duhamel_quadrature", "duhamel" in checks.components),
                                             ("trotter_synthesis", fixed)) if requested and name in values)
    exported = names + (("algorithmic_approximation",) if names else ())
    metrics = [(name, physical if name != "algorithmic_approximation" else plan.error_model.frame,
                None, "selected physical bound or complete output propagation is unavailable") for name in exported]
    from nwqlib.problems.records import UNSPECIFIED_UNIT
    # In du/dt=-A*u+b, A*T is dimensionless. A supplied time unit therefore
    # fixes the reciprocal generator unit; absent time units stay unspecified.
    generator_unit = UNSPECIFIED_UNIT if problem.time_unit is None else (
        problem.time_unit if problem.time_unit.dimension == "dimensionless" else
        Unit(symbol=f"1/({problem.time_unit.symbol})", dimension="custom"))
    norm_frame = ErrorFrame(quantity="physical_generator", metric="spectral_norm", unit=generator_unit,
        scope=problem.evidence_scope, conditioning="original A, actual padded A+shift*I, or padded Hermitian H")
    if "spectral_norms" in checks.components:
        metrics += [(checks.name+"."+name, norm_frame, None, "spectral norm is nonfinite or unrepresentable")
                    for name in ("matrix_norm", "prepared_matrix_norm", "h_norm")]
    if "duhamel" in checks.components and "duhamel_quadrature" not in names:
        metrics.append((checks.name+".duhamel_quadrature", physical, None, "Duhamel bound is unavailable"))
    if fixed and "trotter_synthesis" not in names:
        metrics.append((checks.name+".trotter_synthesis", physical, None, "fixed-PF bound is unavailable"))
    if checks.dense_validation:
        # This dense diagnostic is before physical input/PSD weighting.
        unit_frame = physical.revise(quantity="selected_evolution", metric="operator_norm",
            unit=Unit(symbol="1", dimension="dimensionless"),
            conditioning="weighted fixed-node operator synthesis before physical input and PSD recovery")
        metrics += [(f"{checks.name}.application_{index}.dense_synthesis", unit_frame, None,
                     "dense commutator validation is unavailable")
                    for index in range(0 if grid is None else len(grid.applications))]
    counts = dict(matrix_norm_attempts=0, matrix_norm_completed=0, spectral_svd_attempts=0, spectral_svd_completed=0,
        duhamel_bound_attempts=0, duhamel_bound_completed=0, duhamel_source_norms=0,
        pauli_decomposition_attempts=0, pauli_decomposition_completed=0, structural_bound_attempts=0,
        structural_bound_completed=0, pair_commutation_checks=0, nested_commutation_checks=0,
        dense_bound_attempts=0, dense_bound_completed=0, dense_term_matrices=0, dense_matrix_products=0,
        dense_spectral_norms=0, dense_work=0, known_peak_bytes=0, structural_work=0, stored_node_read_work=0)
    args = [("dimension", problem.dimension), ("encoded_dimension", rec.encoded_dimension),
        ("elapsed_time", problem.elapsed_time), ("psd_shift", rec.psd_shift),
        ("dense_validation", int(checks.dense_validation)), ("duhamel_nodes", len(rec.source_nodes))]
    if grid is not None:
        args += [(name, value) for name, value in grid_arguments(grid) if name != "psd_shift"]
    _admit_publication(checks, tuple((name, frame, reason) for name, frame, _, reason in metrics),
        args+[(name, max(checks.max_dense_work, checks.max_structural_work)) for name in counts])
    invocation_id = verification_invocation()
    measured, reasons = {}, {}
    matrix = source = None
    matrix_norm = None
    if problem.A.reference.representation == "dense":
        d = problem.dimension
        _check_bytes(16*(d*d+2*d), checks.max_bytes, "LCHS refinement physical inputs")
        matrix = problem.A.dense_array()
    source_zero = problem.source is None or problem.source.preparation.physical_scale.mantissa == 0
    duhamel_nonzero = ("duhamel" in checks.components and not source_zero
        and problem.elapsed_time != 0 and matrix is not None and bool(np.any(matrix)))
    if "spectral_norms" in checks.components or duhamel_nonzero:
        matrix_norm = _spectral_norm(matrix, checks, counts)
    if "spectral_norms" in checks.components:
        prepared_norm = matrix_norm
        if rec.psd_shift != 0:
            d = rec.encoded_dimension
            dense_work(checks, counts, work=3*d*d, peak_bytes=64*d*d)
            # A + shift*I on the encoding is the shifted L plus i*H.
            prepared_norm = _spectral_norm(payload.quadrature.l_part+1j*payload.quadrature.h_part, checks, counts)
        if payload is None:
            d = problem.dimension
            dense_work(checks, counts, work=3*d*d, peak_bytes=64*d*d)
            h_part = (matrix-matrix.conj().T)/(2j)
        else:
            h_part = payload.quadrature.h_part
        measured.update({checks.name+".matrix_norm":matrix_norm,
            checks.name+".prepared_matrix_norm":prepared_norm,
            checks.name+".h_norm":_spectral_norm(h_part, checks, counts)})
    if "duhamel" in checks.components:
        bound = 0.
        if duhamel_nonzero and matrix_norm is None:
            bound = None
            reasons["duhamel_quadrature"] = "physical generator norm is nonfinite or unrepresentable"
        elif duhamel_nonzero:
            source = problem.source.physical_vector()
            minimum = payload.quadrature.conversion["min_l_eigenvalue_before_psd_conversion"]
            counts["duhamel_bound_attempts"] += 1
            counts["duhamel_source_norms"] += 1
            evaluated = _duhamel_quadrature_error_bound(matrix=matrix, source_term=source,
                final_time=problem.elapsed_time, node_count=len(rec.source_nodes),
                lambda_min_before_psd_conversion=minimum, matrix_norm=matrix_norm,
                max_bytes=checks.max_bytes, max_dense_work=checks.max_dense_work-counts["dense_work"])
            counts["dense_work"] += 8*problem.dimension+32
            counts["duhamel_bound_completed"] += 1
            bound = None if evaluated.unusable_nonzero or not isfinite(evaluated.value) else evaluated.value
            if bound is None:
                reasons["duhamel_quadrature"] = "nonzero Duhamel remainder is nonfinite or underflowed; no zero bound established"
        key = "duhamel_quadrature" if "duhamel_quadrature" in names else checks.name+".duhamel_quadrature"
        measured[key] = bound
        if key in values:
            values[key] = bound
    if fixed:
        if periodic:
            bound, reason, dense = values.get("trotter_synthesis"), None, []
        elif initial:
            bound, reason, dense = 0., None, []
        else:
            bound, reason, dense = _fixed_pf(plan, grid, checks, counts)
        key = "trotter_synthesis" if "trotter_synthesis" in names else checks.name+".trotter_synthesis"
        measured[key] = bound
        if key in values:
            values[key] = bound
        if reason:
            reasons[key] = reason
        if checks.dense_validation:
            measured.update((f"{checks.name}.application_{index}.dense_synthesis", value)
                            for index, value in enumerate(dense))
    if names:
        # Polarization/triangle bounds at the existing physical propagation
        # owner: for NormSquared, delta*(2*observed_radius+delta), once only.
        observable_norm = None
        if isinstance(plan.output, (QuadraticForm, NormalizedExpectation)) and rec.terms:
            try:
                observable_norm = fsum(abs(coefficient) for _, coefficient in rec.terms)
            except OverflowError:
                pass
        measured["algorithmic_approximation"] = propagate_physical_error(plan.output, values.values(),
            radius=None if result.physical_scale is None else result.physical_scale.as_float(), observable_norm=observable_norm)
    metrics = [(name, frame, measured[name], reasons.get(name, reason)) for name, frame, _, reason in metrics]
    return _publish(plan, result, checks, None, metrics, args+list(counts.items()), invocation_id=invocation_id,
        specs=(),
        witnessed_names=exported, ideal_bounds=exported)
