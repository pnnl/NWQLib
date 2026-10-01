"""Scientific questions and output quantities, with one owner for actual inputs.

Input handles own their immutable numerical data. JSON is a description of those
handles; it cannot manufacture native access. Algorithm choices and execution
limits belong to the selected method and run, respectively.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BeforeValidator, Field, PlainSerializer, field_validator, model_validator

from nwqlib.core.records import InputRef, Real, Record, Scope, Text, Unit
from nwqlib.evidence.records import Fact


UNSPECIFIED_UNIT = Unit(symbol="input units (unspecified)", dimension="custom")
ONE = Unit(symbol="1", dimension="dimensionless")
TURNS = Unit(symbol="turn", dimension="dimensionless")


def _unit(value):
    if value is None or isinstance(value, Unit):
        return value
    if isinstance(value, str):
        return Unit(symbol=value, dimension="custom")
    if type(value) is dict:
        return Unit.model_validate(value)
    raise TypeError("unit must be a label, Unit, or None; no unit conversion is performed")


OptionalUnit = Annotated[Unit | None, BeforeValidator(_unit)]


def _operator(value):
    """Admit live operator data, or read a saved description as a metadata-only handle.

    A saved description carries no native data, so the handle it yields
    refuses numerical access. An archive restores data access separately.
    """
    from nwqlib.operators.inputs import OperatorInput, operator_input
    if type(value) is dict and "manifest" in value:
        return OperatorInput.from_record(value)
    return operator_input(value)


def _hermitian_refusal(name, operator, matrix_remedy=""):
    """Return the refusal for an operator whose coalesced input is not exactly Hermitian.

    For a canonical Pauli sum, exact Hermiticity means that every coalesced
    coefficient is real, and its Hermitian part keeps every Pauli row with
    the real part of its coefficient. ``matrix.conj().T`` is not the adjoint
    of a Qiskit ``SparsePauliOp``, and a default-tolerance ``simplify()`` can
    remove small nonzero real terms, so neither belongs in the Pauli remedy.
    """
    if operator.manifest.reference.representation == "pauli":
        return (f"{name} must be exactly Hermitian. If its Hermitian part is the intended problem, for a Qiskit "
                "`SparsePauliOp` named `op` use `SparsePauliOp(op.paulis, coeffs=op.coeffs.real)`. This keeps "
                "every Pauli row without tolerance-based simplification.")
    return f"{name} must be exactly Hermitian{matrix_remedy}"


def _state(value):
    """Admit live state data, or read a saved description as a metadata-only handle."""
    from nwqlib.problems.inputs import StateInput, state_input
    if type(value) is dict and "manifest" in value:
        return StateInput.from_record(value)
    return state_input(value)


def _input_record(value):
    return value.to_record()


OperatorData = Annotated[Any, BeforeValidator(_operator), PlainSerializer(_input_record)]
StateData = Annotated[Any, BeforeValidator(_state), PlainSerializer(_input_record)]


class SymbolicDescription(Record):
    """A portable expression description, with no executable expression binding."""

    expression: Text
    display: Text


def _symbolic(value):
    if isinstance(value, SymbolicDescription):
        return value
    if type(value) is dict and "expression" in value:
        return SymbolicDescription.model_validate(value)
    import sympy as sp
    if not isinstance(value, sp.Basic):
        raise TypeError("objective and variables must be supplied SymPy expressions, not source strings")
    return value


def _symbolic_record(value):
    if isinstance(value, SymbolicDescription):
        return value.model_dump(mode="json", exclude_computed_fields=True)
    import sympy as sp
    return SymbolicDescription(expression=sp.srepr(value), display=str(value)).model_dump(
        mode="json", exclude_computed_fields=True)


SymbolicData = Annotated[Any, BeforeValidator(_symbolic), PlainSerializer(_symbolic_record)]


def squared_unit(unit):
    """Describe a squared quantity without converting its numerical value."""
    if unit is None:
        return UNSPECIFIED_UNIT
    if unit.same_unit(ONE):
        return ONE
    return Unit(symbol=f"({unit.symbol})^2", dimension="custom")


class _ScientificModel(Record):
    """Immutable user input without interchange version or ancestry arguments.

    The enclosing archive owns its format version. A changed scientific input
    has its own content identity; it does not need a user-managed revision chain.
    """

    schema_version: ClassVar[int] = 1
    parent_id: ClassVar[None] = None

    def revise(self, **changes):
        """Return a validated copy with ``changes``, without a parent link.

        Unlike ``Record.revise``, the result records no ``parent_id``. A
        changed scientific input is a new input with its own identity.
        """
        values = {name: getattr(self, name) for name in type(self).model_fields}
        return type(self).model_validate({**values, **changes})


class ProblemRecord(_ScientificModel):
    """Original scientific inputs plus optional interpretation, never a method.

    Attributes:
        unit: Unit of the primary result: eigenvalue, solution amplitude,
            observable expectation, or objective. None means unspecified.
        scope: Optional physical/model interpretation; no default physical claim.
        coordinates: Optional labels in the original input index order. Labels
            do not transform the basis or allocate missing state amplitudes.
        assumptions: Explicit scientific premises supplied with the question.
        facts: Existing input evidence; no automatic reference computation.
    """

    unit: OptionalUnit = None
    scope: Scope | None = None
    coordinates: tuple[Text, ...] | None = None
    assumptions: tuple[Text, ...] = ()
    facts: tuple[Fact, ...] = ()

    @property
    def evidence_unit(self):
        """Unit in which error evidence about this Problem is stated, explicitly unspecified when ``unit`` is None."""
        return self.unit if self.unit is not None else UNSPECIFIED_UNIT

    @property
    def evidence_scope(self):
        """Scope of error evidence, ``scope`` when given and otherwise the Problem kind in its original coordinates."""
        return self.scope or Scope(domain=f"{self.kind} in the original input coordinates")

    @model_validator(mode="after")
    def _coordinate_labels(self):
        if self.coordinates is not None and len(self.coordinates) != self.dimension:
            raise ValueError("coordinates must label every original input coordinate exactly once")
        return self

    def to_record(self):
        """Project descriptions on demand; actual input data stays with its owner."""
        return self.model_dump(mode="json", exclude_computed_fields=True)

    def default_output(self):
        raise TypeError("select output explicitly for this scientific problem")


class Eigenproblem(ProblemRecord):
    """Request the smallest eigenvalue of a Hermitian A.

    Attributes:
        A: Finite Hermitian matrix or supported structured operator.
        target: Currently "smallest"; a method returns its actual estimate.
        subspace: Optional explicitly defined scientific subspace. A method's
            trial basis or initialization does not change the full-space target.
        sector: Optional scientific sector requiring a compatible preparation.
    """

    kind: Literal["eigenproblem"] = "eigenproblem"
    A: OperatorData
    target: Literal["smallest"] = "smallest"
    subspace: InputRef | None = None
    sector: Text | None = None

    def default_output(self):
        return Eigenvalue()

    @property
    def dimension(self):
        return self.A.manifest.basis.dimension

    @property
    def basis(self):
        return self.A.manifest.basis

    @model_validator(mode="after")
    def _hermitian(self):
        if self.A.structure != "hermitian":
            raise ValueError(_hermitian_refusal(
                "A", self.A,
                "; for a matrix A, choose (A + A.conj().T) / 2 explicitly if that is your intended problem"))
        return self


class LinearDynamics(ProblemRecord):
    """Time-independent du/dt = -A u + source in the original coordinates.

    Attributes:
        A: Finite square matrix or a supported structured generator.
        initial_state: Physical u(initial_time), including magnitude and phase.
        time: Requested final time, at least initial_time.
        source: Optional constant physical source; None means zero.
        initial_time: Initial time, default 0.
        time_unit: Optional label; numerical A*time must be dimensionless.
    """

    kind: Literal["linear_dynamics"] = "linear_dynamics"
    A: OperatorData
    initial_state: StateData
    time: Real
    source: StateData | None = None
    initial_time: Real = 0.0
    time_unit: OptionalUnit = None

    def default_output(self):
        return Solution()

    @property
    def dimension(self):
        return self.A.manifest.basis.dimension

    @property
    def basis(self):
        return self.A.manifest.basis

    @property
    def elapsed_time(self):
        return self.time - self.initial_time

    @model_validator(mode="after")
    def _dynamics(self):
        if self.time < self.initial_time:
            raise ValueError("time must be at least initial_time for du/dt = -A u + source")
        for name, state in (("initial_state", self.initial_state), ("source", self.source)):
            if state is not None and state.manifest.basis != self.basis:
                raise ValueError(f"{name} must use A's original dimension and coordinate order")
        if self.time_unit is not None and self.time_unit.dimension not in {"time", "custom"}:
            raise ValueError("time_unit must describe time; no automatic conversion of A or time is performed")
        return self


class LinearSystem(ProblemRecord):
    """Request physical x in A x = b; the method owns any encoding or dilation.

    Attributes:
        A: Finite square matrix or supported structured operator.
        b: Physical right-hand side, with its original magnitude and phase.
    """

    kind: Literal["linear_system"] = "linear_system"
    A: OperatorData
    b: StateData

    def default_output(self):
        return Solution()

    @property
    def dimension(self):
        return self.A.manifest.basis.dimension

    @property
    def basis(self):
        return self.A.manifest.basis

    @model_validator(mode="after")
    def _system(self):
        if self.b.manifest.basis != self.basis:
            raise ValueError("b must use A's original dimension and coordinate order")
        return self


class Expectation(ProblemRecord):
    """A physical state and Hermitian observable in the same coordinates.

    The default quantity is state† observable state / (state† state).
    An explicit QuadraticForm requests the unnormalized quadratic form.

    Attributes:
        state: Physical state u, including its magnitude and phase.
        observable: Exactly Hermitian observable O in the same dimension and coordinate order.
    """

    kind: Literal["expectation"] = "expectation"
    state: StateData
    observable: OperatorData

    def default_output(self):
        return NormalizedExpectation(observable=self.observable)

    @property
    def dimension(self):
        return self.observable.manifest.basis.dimension

    @property
    def basis(self):
        return self.observable.manifest.basis

    @model_validator(mode="after")
    def _expectation(self):
        if self.observable.structure != "hermitian":
            raise ValueError(_hermitian_refusal("observable", self.observable))
        if self.state.manifest.basis != self.basis:
            raise ValueError("state and observable must use the same dimension and coordinate order")
        return self


class SpectralEstimation(ProblemRecord):
    """Estimate a component of an explicitly prepared spectral population.

    Supply exactly one of hamiltonian or unitary. The method owns its sampling
    schedule and branch convention. A unitary alone does not define an energy.
    The component each Method targets follows from its estimator. SPE, for
    example, targets the lowest value in the prepared support, which is the
    lowest energy of a Hamiltonian but the lowest principal eigenphase of a
    unitary. For ``U = exp(-i tau H)`` with ``tau |E| < pi/2`` for every
    prepared energy E, that eigenphase belongs to the highest energy of the
    support.

    Attributes:
        hamiltonian: Exactly Hermitian H, or None when a unitary is supplied.
        unitary: Unitary U, or None when a Hamiltonian is supplied. Its unitarity is checked by the selected Method.
        initial_state: State whose overlaps with the eigenvectors define the sampled spectral population.
    """

    kind: Literal["spectral_estimation"] = "spectral_estimation"
    hamiltonian: OperatorData | None = None
    unitary: OperatorData | None = None
    initial_state: StateData

    def default_output(self):
        return Eigenphase()

    @property
    def dimension(self):
        return self.operator.manifest.basis.dimension

    @property
    def basis(self):
        return self.operator.manifest.basis

    @property
    def operator(self):
        return self.hamiltonian if self.hamiltonian is not None else self.unitary

    @model_validator(mode="after")
    def _spectral(self):
        if (self.hamiltonian is None) == (self.unitary is None):
            raise ValueError("supply exactly one of hamiltonian or unitary")
        if self.hamiltonian is not None and self.hamiltonian.structure != "hermitian":
            raise ValueError(_hermitian_refusal("hamiltonian", self.hamiltonian))
        if self.initial_state.manifest.basis != self.basis:
            raise ValueError("initial_state must use the operator's dimension and coordinate order")
        # Unitarity of an explicitly supplied dense U is checked by the selected
        # QPE numerical owner under its finite dense-work controls. A portable
        # description alone never supplies executable unitary access.
        return self


def _variable_names(variables):
    return tuple(v.display if isinstance(v, SymbolicDescription) else str(v) for v in variables)


def _admit_box(variables, bounds, expressions):
    """Admit one ``lower < upper`` interval per uniquely named variable.

    ``expressions`` pairs each symbolic expression of the problem with the
    name used in messages, starting with the objective. Live SymPy input also
    requires Symbol variables and expressions whose free symbols are all
    variables. Saved descriptions skip those checks because they carry no
    executable expression. Mixing live and saved forms rejects.
    """
    if not variables or len(variables) != len(bounds):
        raise ValueError("bounds must contain one finite interval for every ordered variable")
    if len(set(_variable_names(variables))) != len(variables):
        raise ValueError("variables must have distinct names in the supplied order")
    if any(lower >= upper for lower, upper in bounds):
        raise ValueError("each box interval must have lower < upper")
    forms = {isinstance(item, SymbolicDescription) for item in (*variables, *(e for _, e in expressions))}
    if len(forms) > 1:
        raise ValueError("the objective, variables and any constraints must be consistently live "
                         "or descriptive symbolic inputs")
    if forms == {False}:
        import sympy as sp
        if any(not isinstance(v, sp.Symbol) for v in variables):
            raise ValueError("variables must be SymPy symbols")
        by_name = {str(v): v for v in variables}
        for name, expression in expressions:
            missing = expression.free_symbols - set(variables)
            if missing:
                # SymPy symbols with one name and different assumptions are different symbols.
                same = [f"{sp.srepr(s)} in {name} and {sp.srepr(by_name[str(s)])} in variables share a name but "
                        "not their assumptions" for s in sorted(missing, key=str) if str(s) in by_name]
                raise ValueError(f"{name} contains symbols missing from variables: "
                                 + ", ".join(sorted(map(str, missing)))
                                 + "".join(f". {text}, so SymPy treats them as different symbols" for text in same))


class Optimization(ProblemRecord):
    """Minimize a supplied SymPy objective over an ordered finite box.

    Attributes:
        objective: A SymPy scalar expression, not an executable source string.
        variables: Ordered SymPy symbols defining coordinate order.
        bounds: One finite (lower, upper) pair per variable.
    """

    kind: Literal["optimization"] = "optimization"
    objective: SymbolicData
    variables: tuple[SymbolicData, ...]
    bounds: tuple[tuple[Real, Real], ...]

    def default_output(self):
        return OptimizationCandidate()

    @property
    def dimension(self):
        return len(self.variables)

    @property
    def variable_names(self):
        return _variable_names(self.variables)

    @model_validator(mode="after")
    def _box(self):
        _admit_box(self.variables, self.bounds, (("objective", self.objective),))
        return self


def _residual(value, field, position):
    """Return the stored residual of one live constraint, h in ``h = 0`` or g in ``g <= 0``.

    ``field`` is ``"equalities"`` or ``"inequalities"``. An equality is an
    expression h or ``Eq(a, b)``, stored as ``a - b``. An inequality is an
    expression g, ``Le(a, b)``, stored as ``a - b``, or ``Ge(a, b)``, stored
    as ``b - a``. The expression, or each side of the relation, must be a
    scalar SymPy expression. A saved description passes through unchanged.
    """
    import sympy as sp
    where = f"{field}[{position}]"
    if isinstance(value, SymbolicDescription) or (type(value) is dict and "expression" in value):
        return value
    if not isinstance(value, sp.Basic):
        raise TypeError(f"{where} must be a SymPy expression or relation, not {type(value).__name__}")
    if isinstance(value, sp.logic.boolalg.BooleanAtom):
        raise ValueError(f"{where} is {value}, which SymPy evaluated from the supplied relation, so it has no "
                         "residual. Build the relation with evaluate=False to keep its constant residual")
    sides = None  # (a, b) with the stored residual a - b
    if field == "equalities":
        if isinstance(value, sp.Eq):
            sides = value.lhs, value.rhs
        elif isinstance(value, (sp.Le, sp.Ge, sp.Lt, sp.Gt)):
            raise ValueError(f"{where} is an inequality relation; supply it in inequalities")
    else:
        if isinstance(value, sp.Le):
            sides = value.lhs, value.rhs
        elif isinstance(value, sp.Ge):
            sides = value.rhs, value.lhs
        elif isinstance(value, (sp.Lt, sp.Gt)):
            raise ValueError(f"{where} is a strict inequality, which the stored residual g(x) <= 0 cannot "
                             "express. Supply a non-strict form with an explicit margin, for example "
                             "Le(x, 1 - margin) for x < 1")
        elif isinstance(value, sp.Eq):
            raise ValueError(f"{where} is an equality relation; supply it in equalities")
    if isinstance(value, sp.Ne):
        raise ValueError(f"{where} is an Ne relation, which neither h = 0 nor g <= 0 expresses")
    for term in sides or (value,):
        # A matrix is an Expr in SymPy, but a constraint residual is one scalar.
        if not isinstance(term, sp.Expr) or isinstance(term, sp.MatrixExpr):
            part = "a side of type" if sides else "type"
            raise ValueError(f"{where} has {part} {type(term).__name__}, which is not a scalar SymPy expression. "
                             "A constraint is a scalar expression or an admitted relation between two of them")
    return sides[0] - sides[1] if sides else value


class ConstrainedOptimization(ProblemRecord):
    """Minimize a SymPy objective over an ordered finite box subject to constraints.

    The stored constraints are residual expressions with the conventions
    ``h_i(x) = 0`` for equalities and ``g_j(x) <= 0`` for inequalities, the
    form of problem (4.1) in Birgin and Martinez, *Practical Augmented
    Lagrangian Methods for Constrained Optimization*, SIAM 2014,
    doi:10.1137/1.9781611973365, with the box as its set Omega. The QHD
    augmented-Lagrangian layer (``algorithms.qhd.constrained``) follows that
    book, so a residual here enters its multiplier updates and stopping tests
    with the book's signs.
    Live input may also give ``Eq(a, b)`` as an equality, stored as ``a - b``,
    and ``Le(a, b)`` or ``Ge(a, b)`` as an inequality, stored as ``a - b`` or
    ``b - a``. Strict relations, ``Ne``, relations that SymPy has already
    evaluated to True or False, other Boolean objects, matrices and
    constraints with symbols outside ``variables`` are rejected. Both
    constraint fields take a tuple or list, and at least one constraint is
    required.

    This record is not an ``Optimization``. A box-only Method such as QHD
    rejects it at planning with ``ApplicabilityError``, so no constraint is
    silently dropped.

    Attributes:
        objective: A SymPy scalar expression, not an executable source string.
        variables: Ordered SymPy symbols defining coordinate order.
        bounds: One finite (lower, upper) pair per variable.
        equalities: Residual expressions h_i with the constraint ``h_i(x) = 0``.
        inequalities: Residual expressions g_j with the constraint ``g_j(x) <= 0``.
    """

    kind: Literal["constrained_optimization"] = "constrained_optimization"
    objective: SymbolicData
    variables: tuple[SymbolicData, ...]
    bounds: tuple[tuple[Real, Real], ...]
    equalities: tuple[SymbolicData, ...] = ()
    inequalities: tuple[SymbolicData, ...] = ()

    def default_output(self):
        return OptimizationCandidate()

    @property
    def dimension(self):
        return len(self.variables)

    @property
    def variable_names(self):
        return _variable_names(self.variables)

    @field_validator("equalities", "inequalities", mode="before")
    @classmethod
    def _residuals(cls, value, info):
        # Pydantic would coerce a set or generator into the tuple without this
        # validator seeing its items, so only ordered sequences are admitted.
        if not isinstance(value, (tuple, list)):
            raise TypeError(f"{info.field_name} must be a tuple of constraints, not {type(value).__name__}")
        return tuple(_residual(item, info.field_name, i) for i, item in enumerate(value))

    @model_validator(mode="after")
    def _constraints(self):
        if not self.equalities and not self.inequalities:
            raise ValueError("supply at least one equality or inequality; use Optimization for a box-only problem")
        _admit_box(self.variables, self.bounds, (
            ("objective", self.objective),
            *((f"equalities[{i}]", h) for i, h in enumerate(self.equalities)),
            *((f"inequalities[{j}]", g) for j, g in enumerate(self.inequalities)),
        ))
        return self


class OutputRecord(_ScientificModel):
    """A requested quantity, with no acquired value or duplicated scope.

    unit labels a quantity that neither the Problem nor the output kind
    labels, for example an observable of a LinearDynamics Problem. It never
    converts numerical data. A unit whose symbol differs from the one the
    Problem defines or the kind fixes raises ValueError at planning, as
    ``frame`` describes.
    """

    unit: OptionalUnit = None

    @property
    def metric(self):
        """Name of the error metric of this output, ``absolute_error`` unless a subclass states another."""
        return "absolute_error"

    def frame(self, problem):
        """Return the ErrorFrame in which evidence about this output is stated.

        The frame fixes quantity, metric, unit, scope and conditioning, so an
        error fact about one output cannot be applied to another. The unit is
        the one the Problem defines, for example its own unit for an
        eigenvalue or the squared solution unit for ``norm_squared``, or the
        one the kind fixes, turns for ``eigenphase`` and 1 for ``samples`` and
        a unit ``StateVector``. The output's ``unit`` labels only a quantity
        without such a unit, which otherwise stays unspecified rather than
        guessed.

        Raises:
            ValueError: If ``unit`` has another symbol than the defined or
                fixed unit. No unit conversion exists, so accepting it would
                attach the wrong label to the unconverted value.
        """
        from nwqlib.evidence.error_model import ErrorFrame
        defined, conditioning = self._defined_unit(problem)
        if self.unit is not None and defined is not None and self.unit.symbol != defined.symbol:
            raise ValueError(
                f"output unit {self.unit.symbol!r} conflicts with the unit {defined.symbol!r} that the "
                f"Problem or the {self.kind} output defines. NWQLib converts no units, so omit the output "
                "unit or label the Problem instead")
        unit = defined if defined is not None else self.unit or UNSPECIFIED_UNIT
        return ErrorFrame(quantity=self.kind, metric=self.metric, unit=unit,
                          scope=problem.evidence_scope, conditioning=conditioning)

    def _defined_unit(self, problem):
        """Return ``(unit, conditioning)``, where unit is the Problem-defined or fixed unit, or None."""
        defined = problem.unit
        conditioning = "original scientific problem"
        if self.kind in {"normalized_expectation", "quadratic_form"}:
            # Expectation.unit labels the normalized observable quantity. A
            # physical quadratic form additionally needs amplitude units, which
            # StateInput does not declare; do not square/reuse observable units.
            defined = (problem.unit if self.kind == "normalized_expectation" and isinstance(problem, Expectation)
                       else None)
            conditioning = ("nonzero state norm" if self.kind == "normalized_expectation" else "physical state magnitude")
        elif self.kind == "norm_squared":
            defined = (squared_unit(problem.unit)
                       if isinstance(problem, (LinearDynamics, LinearSystem)) and problem.unit is not None else None)
        elif self.kind == "samples":
            # The bins count outcomes of the runs in which the Method's
            # success event occurred, for example its post-selection flag.
            defined, conditioning = ONE, "original-coordinate outcomes conditional on the method's success event"
        elif self.kind == "eigenphase":
            defined = TURNS
        elif self.kind in {"solution", "state_vector"}:
            defined = problem.unit if isinstance(problem, (LinearDynamics, LinearSystem)) else None
            if self.kind == "state_vector" and self.normalization == "unit":
                defined, conditioning = ONE, "nonzero state norm"
        elif self.kind == "eigenvalue" and isinstance(problem, Eigenproblem):
            conditioning = (f"explicit scientific subspace {problem.subspace.identity}" if problem.subspace is not None
                            else "smallest eigenvalue in the requested full space or explicit sector")
        return defined, conditioning


class Eigenvalue(OutputRecord):
    """An eigenvalue in the Problem's unit, with absolute error."""

    kind: Literal["eigenvalue"] = "eigenvalue"


class Eigenphase(OutputRecord):
    """A phase phi in turns, with ``U v = exp(2 pi i phi) v`` and phi in [0, 1).

    For a Hamiltonian input, U is ``exp(-i tau H)`` at the time tau that
    the Method selects, so an energy E has phase ``(-tau E / (2 pi)) mod 1``
    and the phase order need not follow the energy order. A Method that
    targets the lowest principal eigenphase of a supplied unitary
    ``exp(-i tau H)``, as SPE does, therefore reports the phase of the
    highest prepared energy of H when ``tau |E| < pi/2`` for every prepared
    E. The error metric of this
    output is the circular distance ``min(|d|, 1 - |d|)`` of the difference d
    between two phases in [0, 1), so estimates on either side of phase zero
    are close.
    """

    kind: Literal["eigenphase"] = "eigenphase"

    @property
    def metric(self):
        return "circular_distance_turns"


class _ObservableOutput(OutputRecord):
    """An output defined by an exactly Hermitian observable O."""

    observable: OperatorData

    @model_validator(mode="after")
    def _observable(self):
        if self.observable.structure != "hermitian":
            raise ValueError(_hermitian_refusal("observable", self.observable))
        return self


class NormalizedExpectation(_ObservableOutput):
    """u† O u / (u† u), undefined when u is zero."""

    kind: Literal["normalized_expectation"] = "normalized_expectation"


class QuadraticForm(_ObservableOutput):
    """u† O u using the physical magnitude of u."""

    kind: Literal["quadratic_form"] = "quadratic_form"


class NormSquared(OutputRecord):
    """The physical squared norm u† u, in the square of the solution unit."""

    kind: Literal["norm_squared"] = "norm_squared"


class Samples(OutputRecord):
    """Original-coordinate outcomes, conditional on the Method's success event.

    The success event is, for example, the post-selection flag of QLS or
    LCHS. Plan.shots selects the number of shots, and the stored bins count
    only the shots in which that event occurred.
    """

    kind: Literal["samples"] = "samples"

    @property
    def metric(self):
        return "total_variation"


class Solution(OutputRecord):
    """Physical solution amplitudes in original coordinates, including phase."""

    kind: Literal["solution"] = "solution"

    @property
    def metric(self):
        return "l2"


class StateVector(OutputRecord):
    """Explicit simulator amplitudes with a declared normalization/phase meaning."""

    kind: Literal["state_vector"] = "state_vector"
    normalization: Literal["physical", "unit"] = "unit"
    global_phase: Literal["physical", "modulo_global_phase"] = "physical"

    @property
    def metric(self):
        return "phase_aligned_l2" if self.global_phase == "modulo_global_phase" else "l2"


class OptimizationCandidate(OutputRecord):
    """A candidate point of the box and its objective value, with the objective gap as error.

    The objective gap is the candidate's objective value minus a reference
    minimum in objective units, for example the grid minimum that an
    explicit QHD check evaluates. A candidate is not an optimality
    certificate.
    """

    kind: Literal["optimization_candidate"] = "optimization_candidate"

    @property
    def metric(self):
        return "objective_gap"


class Accuracy(_ScientificModel):
    """Requested accuracy of one component, without a second quantity definition.

    Exactly one positive tolerance is required. At planning, only a Method
    that defines ``sampling_shots`` consumes an Accuracy, and
    ``ExpectationMethod`` accepts only an absolute sampling tolerance for
    finite Pauli raw counts. ``ErrorModel.assess`` also uses an Accuracy as a
    criterion for the total or sampling component. A total or relative
    target is not silently interpreted as that narrower request.

    Attributes:
        absolute_tolerance: Positive bound on the error in the output's error-frame unit, or None.
        relative_tolerance: Positive bound on the error divided by the target magnitude, or None.
        confidence: Required probability, strictly between 0 and 1, that the bound holds.
        component: ``total`` for the whole output error, or ``sampling`` for the sampling contribution only.
    """

    absolute_tolerance: Annotated[Real, Field(gt=0)] | None = None
    relative_tolerance: Annotated[Real, Field(gt=0)] | None = None
    confidence: Annotated[Real, Field(gt=0, lt=1)] = 0.95
    component: Literal["total", "sampling"] = "total"

    @model_validator(mode="after")
    def _one_tolerance(self):
        if (self.absolute_tolerance is None) == (self.relative_tolerance is None):
            raise ValueError("supply exactly one positive absolute_tolerance or relative_tolerance")
        return self
