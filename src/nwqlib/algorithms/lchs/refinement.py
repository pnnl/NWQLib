"""Explicit component bounds of a planned LCHS construction, without computing a solution vector."""

from nwqlib._limits import exceeds

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
from .route_options import route_name
from .verification import _admit_publication, _publish, _validate_result


class LCHSRefinement(Record):
    """Error-component bounds of an `LCHS` Result at its saved grid and step counts.

    Build it with keyword arguments and pass it to
    `result.verify(checks=...)`, for example
    `result.verify(checks=LCHSRefinement(components=("duhamel",)))`.
    `components` is the only required argument. The call returns
    `(receipt, facts)`, where `receipt` is a `VerificationReceipt` that
    records the check, its raw values and the numerical calls it made, and
    `facts` is a tuple of `FramedFact` records that cite the receipt.
    `facts` holds the refined physical L2 components and their sum carried
    into the requested output, `algorithmic_approximation`, which
    `result.assess` can use. A request for `"spectral_norms"` alone
    leaves `facts` empty.
    Refinement defines no pass or fail check.

    Planning leaves two components unknown for a dense A, the Duhamel
    remainder of a constant source and the synthesis error of fixed-step
    `"trotter"`. Refinement evaluates them at the saved quadrature, Duhamel
    nodes and step counts and replaces the planned values of the same name.
    It never computes a solution vector, chooses a new grid or step count,
    or converts a compact input to a dense matrix. `algorithmic_approximation`
    stays `None` while any component is unknown, because a missing term is
    not zero. It covers the algorithmic components only. Floating-point
    error of the backend run and sampling error remain separate terms.

    Attributes:
        components: Required. Distinct analyses, at least one.
            `"spectral_norms"` evaluates `||A||`, `||A + shift*I||` and
            `||H||` and needs a dense A. `"duhamel"` evaluates the
            Gauss–Legendre remainder of the source time integral (DLMF
            Eqs. 3.5.19 and 3.5.21). The remainder grows as
            `||A||**(2*duhamel_nodes)`, so it is evaluated with `||A||`
            itself. For a Hermitian A both components, alone or together,
            take `||A||` and `||A + shift*I||` from the eigenvalues of L
            that planning computed, with `||H|| = 0`, and decompose no
            matrix. For another dense A, `||A||` and `||A + shift*I||` come
            from singular values and `||H||` from its eigenvalues, and
            `"duhamel"` reuses `||A||`. For a compact A, which planning
            bounds structurally, the planned remainder is reported
            unchanged. `"fixed_pf"` evaluates the
            product-formula commutator bound at the saved step counts
            (Childs et al., doi:10.1103/PhysRevX.11.011020, Propositions
            9-10) and needs the `"trotter"` or `"trotter_error_budgeted"`
            backend.
        name: Default `"refinement"`. Prefix of the reported values that are
            not error components, such as the spectral norms.
        dense_validation: Default `False`. Also evaluate dense commutator
            norms for `"fixed_pf"`, as a check of the Pauli-triangle bound
            (the commutator bound summed term by term over the Pauli
            commutation table), which it does not replace. Requires `"fixed_pf"`, and a periodic
            stencil refuses it.
        max_bytes: Default `None`, the physical memory of this computer.
            Upper limit on the known arrays of this call.
        max_dense_work: Default `None`, no work limit. Upper limit on the
            counted dense work of this call.
        max_structural_work: Default `None`, no work limit. Upper limit on the counted work
            of reading the saved node table, preparing the Pauli commutation
            census (the anticommuting Pauli pairs and nested commutation
            indicators of the commutator bound) and evaluating its terms.
        max_node_evaluations: Default `None`, no limit. Upper limit on the
            number of k nodes times operator applications evaluated.
        max_steps: Default `None`, no limit. Largest saved product-formula
            step count this call accepts.

    Raises:
        ValueError: If `components` is empty or repeats a name, or if
            `dense_validation` is set without `"fixed_pf"`.
    """

    components: tuple[Literal["spectral_norms", "duhamel", "fixed_pf"], ...]
    name: Text = "refinement"
    dense_validation: StrictBool = False
    max_bytes: PositiveInt | None = None
    max_dense_work: PositiveInt | None = None
    max_structural_work: PositiveInt | None = None
    max_node_evaluations: PositiveInt | None = None
    max_steps: PositiveInt | None = None

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
        """Declare no checks, since refinement reports component bounds rather than a pass test."""
        _validate_result(result.plan, result)
        return ()


def _spectral_norm(matrix, checks, counts, *, hermitian=False):
    """Return ||matrix||_2, or None when a nonzero matrix gives a nonfinite or zero norm.

    A general matrix uses its singular values without singular vectors
    (numpy.linalg.norm(ord=2), LAPACK gesdd with JOBZ='N'). A Hermitian one
    uses the largest magnitude of its eigenvalues without eigenvectors
    (numpy.linalg.eigvalsh, LAPACK heevd with JOBZ='N'). The d*d scan for a
    nonzero entry is counted first, then, for a nonzero matrix only,
    _linalg_laws.singular_values_work. The Hermitian route borrows that
    count as a proxy: its cubic term is twice the (2/3) d**3 multiply-adds
    of the tridiagonal reduction, and the eigenvalue stage after that
    reduction, not measured in this project, is assumed to fit within the
    extra (2/3) d**3 and the quadratic term. The count is up to about twice
    the actual cubic work, so an explicit max_dense_work can refuse this
    step earlier than needed. The default None is unaffected. An exactly
    zero matrix has norm zero. Bytes: the matrix and the copy that NumPy
    passes to LAPACK, 32*d*d.
    """
    from nwqlib._linalg_laws import singular_values_work
    d = len(matrix)
    dense_work(checks, counts, work=d*d, peak_bytes=32*d*d)
    nonzero = bool(np.any(matrix))
    if nonzero:
        dense_work(checks, counts, work=singular_values_work(d), peak_bytes=32*d*d)
    kind = "spectral_eigvalsh" if hermitian else "spectral_svd"
    counts["matrix_norm_attempts"] += 1
    counts[kind+"_attempts"] += int(nonzero)
    value = 0.0
    if nonzero:
        value = float(np.max(np.abs(np.linalg.eigvalsh(matrix)))) if hermitian else float(np.linalg.norm(matrix, 2))
    counts["matrix_norm_completed"] += 1
    counts[kind+"_completed"] += int(nonzero)
    return value if isfinite(value) and (value != 0 or not nonzero) else None


def _planned_hermitian_norms(payload, checks, counts):
    """Return (||A||, ||A_pad + s*I||) from the planned eigenvalues of L when the planned H is exactly zero, else None.

    When H = 0, A = L, and planning's one Hermitian eigensolve of the
    original L (time_independent_terms._prepare_decomposition), whose two
    endpoints it then extends by the zero eigenvalue of the padding, gives
    both norms: lambda_min before the PSD shift s >= 0, and l_norm =
    ||L_pad + s*I|| after it. For s > 0 the shifted spectrum is
    nonnegative, so lambda_max = l_norm - s, and for s = 0, l_norm =
    ||L_pad||. The padding zero does not raise the largest magnitude, so
    ||A|| = ||L|| = max(|lambda_min|, |l_norm - s|), and ||A_pad + s*I|| =
    l_norm, which equals ||A|| for s = 0. Detecting H = 0 scans the planned
    padded H once, D*D work. A refinement has no requested tolerance at
    which a looser bound on ||A|| would suffice for the Duhamel remainder,
    which grows as ||A||**(2m), so otherwise ||A|| comes from singular values.
    """
    h_part = payload.dense_inputs().h_part
    dense_work(checks, counts, work=h_part.size, peak_bytes=16*h_part.size)
    if np.any(h_part):
        return None
    conversion, l_norm = payload.quadrature.conversion, payload.quadrature.l_norm
    norm = max(abs(conversion["min_l_eigenvalue_before_psd_conversion"]), abs(l_norm-conversion["psd_shift"]))
    counts["planned_l_endpoint_norms"] += 1
    return norm, (l_norm if conversion["psd_shift"] else norm)


def _component_values(plan, result, payload):
    """Collect the existing physical component facts that refinement may replace.

    Only listed_only error terms that add to the total are read, and each
    must keep its planned physical frame, so a refined value cannot enter the
    wrong output frame.
    """
    from .analysis import adds_to_total
    # Classical Result facts are the host application sums, and
    # selected_grid.host_applications checks their receipts against the Plan
    # first. Quantum facts come from the selected numerical recipe. Neither
    # is recomputed to obtain a component for this propagation.
    if plan.execution == "classical" and payload is not None:
        from .selected_grid import host_applications
        host_applications(plan, result, payload)
    supplied = {fact.fact.quantity:fact for fact in (*plan.facts, *result.facts)}
    values = {}
    for term in plan.error_model.terms:
        if term.role != "listed_only" or not adds_to_total(term.name):
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
    from the Plan's classical actions (classical execution) or the planned
    SELECT (quantum execution), never chosen again. Returns the bound, an unavailability
    reason, and the optional dense validation values per application.
    """
    from .time_independent_terms import (stored_pf_nodes,
        _fixed_trotter_certificate_records, _trotter_budget_quadrature_terms)
    from .solution_error_budget import _build_lchs_application_record, application_stage_bounds

    d = plan.reconstruction.encoded_dimension
    if any(exceeds(step, checks.max_steps) for app in grid.applications for step in app.steps):
        raise ValueError("fixed-PF refinement exceeds max_steps before work")
    active = tuple(app.weight != 0 and app.input_norm != 0 for app in grid.applications)
    if any(app.input_norm is None for app, on in zip(grid.applications, active, strict=True) if on):
        return None, "physical input scale is not representable", [None]*len(grid.applications)
    applications, dense_values = [], []
    raw = {"psd_shift":grid.shift}
    # The saved node table is read once per distinct node, before the
    # application loop, and evaluated at each application's saved
    # (elapsed_time, step_count). No Pauli decomposition runs. The grid's
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
            # Nodes that planning omitted for this application are never applied.
            coefficients = grid.coefficients
            if app.omitted:
                coefficients = np.array(coefficients)
                coefficients[list(app.omitted)] = 0
            synthesis = _trotter_budget_quadrature_terms(records, coefficients=coefficients,
                method=grid.payload.method)["trotter_synthesis_error_bound"]
            dense = None if not checks.dense_validation else fsum(abs(coefficient)*record.dense_bound_value
                for coefficient, record in zip(coefficients, records, strict=True))
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
        psd_premise_satisfied=grid.payload.quadrature.psd_premise_satisfied)
    bound, reason = values["trotter_synthesis"]
    return bound, reason, dense_values


def refine(plan, result, *, checks):
    """Evaluate requested component bounds for an existing LCHS Result.

    Refinement evaluates the error components of the planned construction
    at the saved quadrature, Duhamel nodes and step counts. It never computes
    a solution vector, chooses a new grid or step count, or converts a
    compact input to a dense matrix. All reported quantities, their frames
    and the argument record are checked against max_bytes before any
    numerical work, and each numerical routine is checked against its own
    work limit before it runs.

    Refined duhamel_quadrature and trotter_synthesis values replace the
    planned components of the same name. The complete set is then propagated
    once into the requested output frame as algorithmic_approximation
    (solution_error_budget.propagate_physical_error). That value stays None
    while any component is unknown, because a missing term is not zero.
    Spectral norms and dense diagnostics are reported under the checks name
    and belong to no error term.

    Returns:
        A VerificationReceipt holding every observation, and a tuple with the
        refined physical-frame components and the output-frame
        algorithmic_approximation fact, which are the facts suitable for
        assessment. A request for spectral norms alone leaves that tuple empty.
    """
    from .selected_grid import selected_grid, selected_payload, grid_arguments
    from .inhomogeneous_theory import _duhamel_quadrature_error_bound
    from .solution_error_budget import _StageFailure, _stage_factor, _stage_value, propagate_physical_error
    from .periodic import _PeriodicPayload

    if type(checks) is not LCHSRefinement:
        raise TypeError("checks requires one explicit LCHSRefinement")
    _validate_result(plan, result)
    rec, problem = plan.reconstruction, plan.problem
    fixed = "fixed_pf" in checks.components
    if fixed and route_name(plan.method.ham_evo) not in {"trotter", "trotter_error_budgeted"}:
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
    # fixes the reciprocal generator unit. Without a time unit the generator
    # unit stays unspecified.
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
        spectral_eigvalsh_attempts=0, spectral_eigvalsh_completed=0, planned_l_endpoint_norms=0,
        duhamel_bound_attempts=0, duhamel_bound_completed=0,
        pauli_decomposition_attempts=0, pauli_decomposition_completed=0, structural_bound_attempts=0,
        structural_bound_completed=0, pair_commutation_checks=0, nested_commutation_checks=0,
        dense_bound_attempts=0, dense_bound_completed=0, dense_term_matrices=0, dense_matrix_products=0,
        dense_spectral_norms=0, dense_work=0, known_peak_bytes=0, structural_work=0, stored_node_read_work=0)
    args = [("dimension", problem.dimension), ("encoded_dimension", rec.encoded_dimension),
        ("elapsed_time", problem.elapsed_time), ("psd_shift", rec.psd_shift),
        ("dense_validation", int(checks.dense_validation)), ("duhamel_nodes", len(rec.source_nodes))]
    if grid is not None:
        args += [(name, value) for name, value in grid_arguments(grid) if name != "psd_shift"]
    _admit_publication(checks, tuple((name, frame, reason) for name, frame, _, reason in metrics), args, counts)
    invocation_id = verification_invocation()
    measured, reasons = {}, {}
    matrix = matrix_norm = None
    if problem.A.reference.representation == "dense":
        d = problem.dimension
        _check_bytes(16*(d*d+2*d), checks.max_bytes, "LCHS refinement physical inputs")
        matrix = problem.A.dense_array()
    source_zero = problem.source is None or problem.source.preparation.physical_scale.mantissa == 0
    duhamel_dense = ("duhamel" in checks.components and matrix is not None and not source_zero
                     and problem.elapsed_time != 0 and bool(np.any(matrix)))
    # One scan of the planned H serves both components: when it is exactly
    # zero, the norms come from the planned eigenvalues of L.
    planned = None
    if payload is not None and matrix is not None and ("spectral_norms" in checks.components or duhamel_dense):
        planned = _planned_hermitian_norms(payload, checks, counts)
    if "spectral_norms" in checks.components:
        if planned is not None:
            (matrix_norm, prepared_norm), h_norm = planned, 0.0
        else:
            d, shift = problem.dimension, rec.psd_shift
            matrix_norm = prepared_norm = _spectral_norm(matrix, checks, counts)
            if shift != 0:
                # A_pad + s*I = (A + s*I) ⊕ s*I, so its norm is
                # max(||A + s*I||, s) when there is padding, all on the
                # original dimension.
                dense_work(checks, counts, work=2*d*d, peak_bytes=32*d*d)
                shifted = matrix.astype(complex)
                shifted[np.diag_indices(d)] += shift
                prepared_norm = _spectral_norm(shifted, checks, counts)
                del shifted
                if prepared_norm is not None and rec.encoded_dimension > d:
                    prepared_norm = max(prepared_norm, shift)
            # ||H_pad|| = ||H||.
            dense_work(checks, counts, work=3*d*d, peak_bytes=64*d*d)
            h_norm = _spectral_norm((matrix-matrix.conj().T)/(2j), checks, counts, hermitian=True)
        measured.update({checks.name+".matrix_norm":matrix_norm,
            checks.name+".prepared_matrix_norm":prepared_norm, checks.name+".h_norm":h_norm})
    elif duhamel_dense:
        matrix_norm = planned[0] if planned is not None else _spectral_norm(matrix, checks, counts)
    if "duhamel" in checks.components:
        key = "duhamel_quadrature" if "duhamel_quadrature" in names else checks.name+".duhamel_quadrature"
        if matrix is None:
            # Planning evaluated the remainder of a compact A with its
            # structural bound on ||A|| (parameters._duhamel_bound). Without a
            # dense A nothing tightens it, so the planned value stays, unknown
            # when planning had none.
            bound = values.get("duhamel_quadrature", 0.)
        elif not duhamel_dense:
            bound = 0.
        else:
            bound = None
            if not matrix_norm or not isfinite(matrix_norm):
                reasons[key] = "physical generator norm is nonfinite or unrepresentable"
            else:
                counts["duhamel_bound_attempts"] += 1
                remainder = _duhamel_quadrature_error_bound(generator_norm=matrix_norm,
                    lower_bound=payload.quadrature.conversion["min_l_eigenvalue_before_psd_conversion"],
                    source_scale=problem.source.preparation.physical_scale, final_time=problem.elapsed_time,
                    node_count=len(rec.source_nodes))
                counts["duhamel_bound_completed"] += 1
                # The planned stage's single binary64 conversion: a positive
                # bound stays positive, and one above binary64 is unknown.
                try:
                    bound = _stage_value("duhamel_quadrature", _stage_factor("duhamel_quadrature", remainder))
                except _StageFailure as failure:
                    reasons[key] = str(failure)
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
