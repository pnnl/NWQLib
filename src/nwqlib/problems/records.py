"""Problems, outputs and accuracy requests.

A Problem states a mathematical question with its inputs and units, an output
names the quantity to compute, and an `Accuracy` states a requested tolerance.
A Problem holds no Method setting and no run limit. The Method is given to
`plan` or `solve`, and the limits to `prepare` or `solve`. A Problem keeps
the numerical data of its inputs. Its JSON form describes them, and a Problem
rebuilt from that description alone cannot access the numbers.
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
"""Matrix or operator accepted by a Problem or output field, such as `Eigenproblem.A`.

Pass one of these forms:

- a square NumPy array, or a nested list or tuple of numbers
- a SciPy CSR or CSC matrix or array in canonical form
- a Qiskit `SparsePauliOp`
- a `PeriodicStencil`
- an `OperatorInput` from `nwqlib.operators.operator_input` or an
  `ingest_*` function of `nwqlib.operators`

The field keeps the representation it receives and does not convert sparse
or Pauli input to a dense matrix. Accepting the input is limited to 10 GB
(decimal) of known bytes. Pass `operator_input(value, max_bytes=...)` to set
another limit. A saved description, a dict with a `manifest` entry, gives a
handle with metadata only, which refuses numerical access. Loading a saved
archive restores the data. See [Supply operators and
states](../inputs.md).
"""
StateData = Annotated[Any, BeforeValidator(_state), PlainSerializer(_input_record)]
"""State or vector accepted by a Problem field, such as `Expectation.state` or `LinearSystem.b`.

Pass one of these forms:

- a NumPy array, or a list or tuple of numbers, with its magnitude and phase
- a Qiskit `Statevector`
- a Qiskit `QuantumCircuit` without classical bits or free parameters,
  which means its unitary applied to the all-zero state
- a `StateInput` from `nwqlib.problems.state_input` or an `ingest_*`
  function of `nwqlib.problems`

The field keeps the vector's magnitude and phase. A circuit is copied without
simulating or synthesizing it. Accepting the input is limited to 10 GB
(decimal) of known bytes. Pass `state_input(value, max_bytes=...)` to set
another limit. A saved description, a dict with a `manifest` entry, gives a
handle with metadata only, which refuses numerical access. Loading a saved
archive restores the data. See [Supply operators and
states](../inputs.md).
"""


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
    """The optional fields that every Problem accepts besides its own inputs.

    Build a Problem class, such as
    [`Eigenproblem`][nwqlib.problems.records.Eigenproblem], with keyword
    arguments. Every Problem also accepts the optional fields below. A
    Problem is immutable. `problem.revise(**changes)` returns a validated
    copy with the changes, and a Problem with changed inputs has its own
    content hash. A Problem holds no Method setting.

    Attributes:
        unit: Default `None`, which leaves the unit unspecified. Unit of the
            primary answer: the eigenvalue, the solution amplitudes, the
            observable's expectation or the objective. A string such as
            `"Hartree"` becomes a `Unit` of dimension `"custom"`. NWQLib
            converts no units. [Units](../problems.md#units) lists the unit
            of each output. Without a unit, error evidence about the Problem
            is stated in an explicitly unspecified unit.
        scope: Default `None`, which makes no physical claim. A `Scope` that
            states the physical or model meaning of the Problem. Without it,
            error evidence about the Problem is scoped to the Problem kind in
            its original input coordinates.
        coordinates: Default `None`. One label per input coordinate, in the
            input's index order. Labels do not change the basis or add
            missing state amplitudes.
        assumptions: Default `()`. Assumptions stated with the Problem, as
            text.
        facts: Default `()`. Evidence about the inputs (`Fact` records). No
            reference computation is run to produce it.

    Raises:
        TypeError: If `unit` is neither a string, a `Unit`, a dict of `Unit`
            fields nor `None`.
        ValueError: If `coordinates` does not label every input coordinate
            exactly once.
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
        """Return the output that `plan` and `compare` use when no `output` is given.

        Each built-in Problem returns its default output, such as
        `Eigenvalue()` for an `Eigenproblem`. A Problem type of a Method
        author defines its own, as [Run your own circuit](../own_circuit.md)
        shows. This base raises, so a Problem without its own default needs
        an explicit `output`.

        Returns:
            output (OutputRecord): The default output.

        Raises:
            TypeError: Always, on this base.
        """
        raise TypeError("select output explicitly for this scientific problem")


class Eigenproblem(ProblemRecord):
    """The smallest eigenvalue of a Hermitian matrix or operator `A`.

    Build it with keyword arguments, for example `Eigenproblem(A=matrix)`,
    and pass it to [`solve`][nwqlib.scientist.solve] or `plan` with a Method
    such as `Lanczos`. `A` is the only required argument. The other fields
    below have defaults, and the optional `unit`, `scope`, `coordinates`,
    `assumptions` and `facts` of
    [`ProblemRecord`][nwqlib.problems.records.ProblemRecord] are accepted
    too. Without `output=`, the requested output is
    [`Eigenvalue`][nwqlib.problems.records.Eigenvalue], in the Problem's
    `unit`.

    Attributes:
        A: Required. Finite Hermitian matrix or supported structured operator,
            in a form that [`OperatorData`][nwqlib.problems.records.OperatorData]
            accepts.
        target: Default `"smallest"`, the only accepted value. A Method
            returns its own estimate of this target.
        subspace: Default `None`. An explicitly defined scientific subspace.
            A Method's trial basis or initial state does not change the
            full-space target.
        sector: Default `None`. A scientific sector, which requires a
            compatible state preparation.

    Raises:
        ValueError: If `A` is not exactly Hermitian. If the Hermitian part
            of a matrix `A` is the intended problem, pass
            `(A + A.conj().T) / 2`. For a Qiskit `SparsePauliOp` named `op`,
            pass `SparsePauliOp(op.paulis, coeffs=op.coeffs.real)`, which
            keeps every Pauli term without tolerance-based simplification.

    Examples:
        The eigenvalues of `[[2, 1], [1, 2]]` are 1 and 3. Lanczos with a
        two-dimensional Krylov space recovers the smallest:

        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import Lanczos
        >>> problem = Eigenproblem(A=[[2.0, 1.0], [1.0, 2.0]], unit="Hartree")
        >>> method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
        >>> result = solve(problem, method=method, seed=7)
        >>> print(round(result.eigenvalue, 10))
        1.0
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
        """Number of coordinates of `A`."""
        return self.A.manifest.basis.dimension

    @property
    def basis(self):
        """The coordinate basis of `A` (a `Basis` record with its dimension and ordering)."""
        return self.A.manifest.basis

    @model_validator(mode="after")
    def _hermitian(self):
        if self.A.structure != "hermitian":
            raise ValueError(_hermitian_refusal(
                "A", self.A,
                "; for a matrix A, choose (A + A.conj().T) / 2 explicitly if that is your intended problem"))
        return self


class LinearDynamics(ProblemRecord):
    """The solution `u(time)` of the linear equation `du/dt = -A u + source` with constant `A` and `source`.

    Build it with keyword arguments, for example
    `LinearDynamics(A=matrix, initial_state=u0, time=1.0)`, and pass it to
    [`solve`][nwqlib.scientist.solve] or [`plan`][nwqlib.scientist.plan]
    with a Method such as `LCHS`. `A`, `initial_state` and `time` are
    required. The optional `unit`, `scope`, `coordinates`, `assumptions` and
    `facts` of [`ProblemRecord`][nwqlib.problems.records.ProblemRecord] are
    accepted too, and `unit` labels the solution amplitudes. Without
    `output=`, the requested output is
    [`Solution`][nwqlib.problems.records.Solution]: `u(time)` in the original
    coordinates, with its magnitude and phase.

    Attributes:
        A: Required. Finite square matrix or supported structured generator,
            in a form that [`OperatorData`][nwqlib.problems.records.OperatorData]
            accepts.
        initial_state: Required. The vector `u(initial_time)`, with its
            magnitude and phase, in the coordinates of `A` and in a form that
            [`StateData`](inputs.md#nwqlib.problems.records.StateData) accepts.
        time: Required. Final time, at least `initial_time`.
        source: Default `None`, which means zero. Constant source vector, with
            its magnitude and phase, in the coordinates of `A`.
        initial_time: Default `0.0`. Initial time.
        time_unit: Default `None`. Label of the time unit, of dimension
            `"time"` or `"custom"`. NWQLib converts no units, so the
            numerical `A*time` must be dimensionless.

    Raises:
        ValueError: If `time` is less than `initial_time`, if `initial_state`
            or `source` has another dimension or coordinate order than `A`,
            or if `time_unit` has another dimension.

    Examples:
        For `A = diag(1, 2)` the solution at `time=0.5` from `u = (1, 1)` is
        `(exp(-0.5), exp(-1)) = (0.6065..., 0.3679...)`. The default LCHS
        approximation, evaluated classically, gives:

        >>> import numpy as np
        >>> from nwqlib import LinearDynamics, solve
        >>> from nwqlib.algorithms import LCHS
        >>> problem = LinearDynamics(A=[[1.0, 0.0], [0.0, 2.0]],
        ...                          initial_state=[1.0, 1.0], time=0.5)
        >>> result = solve(problem, method=LCHS(), execution="classical")
        >>> print(np.round(result.solution.real, 2))
        [0.61 0.37]
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
        """Number of coordinates of `A`."""
        return self.A.manifest.basis.dimension

    @property
    def basis(self):
        """The coordinate basis of `A` (a `Basis` record with its dimension and ordering)."""
        return self.A.manifest.basis

    @property
    def elapsed_time(self):
        """The evolution time, `time - initial_time`."""
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
    """The solution `x` of the linear system `A x = b`.

    Build it with keyword arguments, `LinearSystem(A=matrix, b=vector)`, and
    pass it to [`solve`][nwqlib.scientist.solve] or
    [`plan`][nwqlib.scientist.plan] with a Method such as `QLS`. `A` and `b`
    are required. The optional `unit`, `scope`, `coordinates`, `assumptions`
    and `facts` of [`ProblemRecord`][nwqlib.problems.records.ProblemRecord]
    are accepted too, and `unit` labels the solution. Without `output=`, the
    requested output is [`Solution`][nwqlib.problems.records.Solution]: `x`
    in the original coordinates, with its magnitude and phase. The Method
    chooses any encoding or dilation of `A`.

    Attributes:
        A: Required. Finite square matrix or supported structured operator,
            in a form that [`OperatorData`][nwqlib.problems.records.OperatorData]
            accepts.
        b: Required. Right-hand side, with its magnitude and phase, in the
            coordinates of `A` and in a form that
            [`StateData`](inputs.md#nwqlib.problems.records.StateData) accepts.

    Raises:
        ValueError: If `b` has another dimension or coordinate order than
            `A`.

    Examples:
        The solution of this system is `(25/28, 5/28) = (0.8929..., 0.1786...)`.
        QLS selects its inverse polynomial for its default
        `epsilon_inv=0.01`, a construction target that does not bound the
        total error, and returns an approximation:

        >>> import numpy as np
        >>> from nwqlib import LinearSystem, solve
        >>> from nwqlib.algorithms.qls import QLS
        >>> problem = LinearSystem(A=[[1.1, 0.1], [0.1, 0.9]], b=[1.0, 0.25])
        >>> result = solve(problem, method=QLS(), seed=7)
        >>> print(np.round(result.x.real, 3))
        [0.897 0.181]
    """

    kind: Literal["linear_system"] = "linear_system"
    A: OperatorData
    b: StateData

    def default_output(self):
        return Solution()

    @property
    def dimension(self):
        """Number of coordinates of `A`."""
        return self.A.manifest.basis.dimension

    @property
    def basis(self):
        """The coordinate basis of `A` (a `Basis` record with its dimension and ordering)."""
        return self.A.manifest.basis

    @model_validator(mode="after")
    def _system(self):
        if self.b.manifest.basis != self.basis:
            raise ValueError("b must use A's original dimension and coordinate order")
        return self


class Expectation(ProblemRecord):
    """The expectation of a Hermitian observable `O` in a state `u` given in the same coordinates.

    Build it with keyword arguments, `Expectation(state=u, observable=O)`,
    and pass it to [`solve`][nwqlib.scientist.solve] or
    [`plan`][nwqlib.scientist.plan] with `ExpectationMethod`. Both arguments
    are required. The optional `unit`, `scope`, `coordinates`, `assumptions`
    and `facts` of [`ProblemRecord`][nwqlib.problems.records.ProblemRecord]
    are accepted too, and `unit` labels the normalized expectation. Without
    `output=`, the requested output is
    [`NormalizedExpectation`][nwqlib.problems.records.NormalizedExpectation]
    with this observable, the value `u† O u / (u† u)`. Request
    [`QuadraticForm`][nwqlib.problems.records.QuadraticForm] for the
    unnormalized `u† O u`.

    Attributes:
        state: Required. The state `u`, with its magnitude and phase, in a
            form that [`StateData`](inputs.md#nwqlib.problems.records.StateData)
            accepts.
        observable: Required. Exactly Hermitian observable `O`, with the
            dimension and coordinate order of `state`, in a form that
            [`OperatorData`][nwqlib.problems.records.OperatorData] accepts.

    Raises:
        ValueError: If `observable` is not exactly Hermitian, or `state` has
            another dimension or coordinate order. For a Qiskit
            `SparsePauliOp` named `op` whose Hermitian part is the intended
            observable, pass `SparsePauliOp(op.paulis, coeffs=op.coeffs.real)`.

    Examples:
        For `u = (2, 2)` and `O = diag(2, 0)`, the normalized expectation is
        `8 / 8 = 1` and the quadratic form is `8`:

        >>> from nwqlib import Expectation, QuadraticForm, solve
        >>> from nwqlib.algorithms import ExpectationMethod
        >>> problem = Expectation(state=[2.0, 2.0],
        ...                       observable=[[2.0, 0.0], [0.0, 0.0]])
        >>> result = solve(problem, method=ExpectationMethod())
        >>> print(round(result.value, 10))
        1.0
        >>> output = QuadraticForm(observable=problem.observable)
        >>> physical = solve(problem, method=ExpectationMethod(),
        ...                  output=output, execution="classical")
        >>> print(round(physical.value, 10))
        8.0
    """

    kind: Literal["expectation"] = "expectation"
    state: StateData
    observable: OperatorData

    def default_output(self):
        return NormalizedExpectation(observable=self.observable)

    @property
    def dimension(self):
        """Number of coordinates of `observable`."""
        return self.observable.manifest.basis.dimension

    @property
    def basis(self):
        """The coordinate basis of `observable` (a `Basis` record with its dimension and ordering)."""
        return self.observable.manifest.basis

    @model_validator(mode="after")
    def _expectation(self):
        if self.observable.structure != "hermitian":
            raise ValueError(_hermitian_refusal("observable", self.observable))
        if self.state.manifest.basis != self.basis:
            raise ValueError("state and observable must use the same dimension and coordinate order")
        return self


class SpectralEstimation(ProblemRecord):
    """An eigenphase or energy of a Hamiltonian or unitary, estimated from a prepared initial state.

    Build it with keyword arguments, for example
    `SpectralEstimation(hamiltonian=H, initial_state=psi)`, and pass it to
    [`solve`][nwqlib.scientist.solve] or [`plan`][nwqlib.scientist.plan] with
    a phase-estimation Method: `QCELS`, `SPE`, `RFE` or `RWPE`.
    `initial_state` is required, and exactly one of `hamiltonian` and
    `unitary`. The optional `unit`, `scope`, `coordinates`, `assumptions` and
    `facts` of [`ProblemRecord`][nwqlib.problems.records.ProblemRecord] are
    accepted too. Without `output=`, the requested output is
    [`Eigenphase`][nwqlib.problems.records.Eigenphase], a phase in turns. The
    overlaps of `initial_state` with the eigenvectors define the spectral
    population that the measurements sample. The Method chooses its sampling
    schedule and branch convention. A unitary alone does not define an
    energy.

    The component each Method targets follows from its estimator. SPE, for
    example, targets the lowest value in the prepared support, which is the
    lowest energy of a Hamiltonian but the lowest principal eigenphase of a
    unitary. For `U = exp(-i tau H)` with `tau |E| < pi/2` for every
    prepared energy E, that eigenphase belongs to the highest energy of the
    support.

    Attributes:
        hamiltonian: Default `None`. Exactly Hermitian `H`, in a form that
            [`OperatorData`][nwqlib.problems.records.OperatorData] accepts.
            Give it or `unitary`.
        unitary: Default `None`. Unitary `U`, in a form that
            [`OperatorData`][nwqlib.problems.records.OperatorData] accepts.
            Give it or `hamiltonian`. The Method checks that it is unitary.
        initial_state: Required. State whose overlaps with the eigenvectors
            define the sampled spectral population, in the coordinates of the
            operator and in a form that
            [`StateData`](inputs.md#nwqlib.problems.records.StateData) accepts.

    Raises:
        ValueError: If both or neither of `hamiltonian` and `unitary` are
            given, if `hamiltonian` is not exactly Hermitian, or if
            `initial_state` has another dimension or coordinate order than the
            operator.

    Examples:
        The initial state `(1, 0)` is the eigenvector of energy 0.2 of
        `H = diag(0.2, 0.7)`:

        >>> from nwqlib import SpectralEstimation, solve
        >>> from nwqlib.algorithms.qpe import QCELS
        >>> problem = SpectralEstimation(
        ...     hamiltonian=[[0.2, 0.0], [0.0, 0.7]], initial_state=[1, 0])
        >>> result = solve(problem, method=QCELS(), seed=7)
        >>> print(round(result.eigenvalue, 10))
        0.2
    """

    kind: Literal["spectral_estimation"] = "spectral_estimation"
    hamiltonian: OperatorData | None = None
    unitary: OperatorData | None = None
    initial_state: StateData

    def default_output(self):
        return Eigenphase()

    @property
    def dimension(self):
        """Number of coordinates of the operator."""
        return self.operator.manifest.basis.dimension

    @property
    def basis(self):
        """The coordinate basis of the operator (a `Basis` record with its dimension and ordering)."""
        return self.operator.manifest.basis

    @property
    def operator(self):
        """The Hamiltonian when one is given, otherwise the unitary."""
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
    """The minimum of a SymPy objective over a finite box of real variables.

    Build it with keyword arguments, for example
    `Optimization(objective=(x - 0.2)**2, variables=(x,), bounds=((-1.0, 1.0),))`
    with `x = sympy.Symbol("x", real=True)`, and pass it to
    [`solve`][nwqlib.scientist.solve] or [`plan`][nwqlib.scientist.plan] with
    `QHD`. The three arguments are required. The optional `unit`, `scope`,
    `coordinates`, `assumptions` and `facts` of
    [`ProblemRecord`][nwqlib.problems.records.ProblemRecord] are accepted
    too, and `unit` labels the objective. Without `output=`, the requested
    output is
    [`OptimizationCandidate`][nwqlib.problems.records.OptimizationCandidate],
    a candidate point and its objective value, which is not a proof of
    optimality.

    Attributes:
        objective: Required. Scalar SymPy expression in the variables, not a
            string of source code.
        variables: Required. SymPy symbols with distinct names. Their order
            is the coordinate order.
        bounds: Required. One finite `(lower, upper)` pair with
            `lower < upper` per variable, in the order of `variables`.

    Raises:
        TypeError: If `objective` or a variable is not a SymPy expression.
        ValueError: If `bounds` does not give one interval per variable, an
            interval has `lower >= upper`, two variables share a name, a
            variable is not a SymPy symbol, or `objective` contains a symbol
            that is not in `variables`.

    Examples:
        QHD's default grid has two interior points per variable, `-1/3` and
        `1/3` on `[-1, 1]`, and the objective is smaller at `1/3`:

        >>> import sympy as sp
        >>> from nwqlib import Optimization, solve
        >>> from nwqlib.algorithms.qhd import QHD
        >>> x = sp.Symbol("x", real=True)
        >>> problem = Optimization(objective=(x - 0.2)**2, variables=(x,),
        ...                        bounds=((-1.0, 1.0),))
        >>> result = solve(problem, method=QHD(), seed=7)
        >>> print([round(value, 10) for value in result.candidate])
        [0.3333333333]
    """

    kind: Literal["optimization"] = "optimization"
    objective: SymbolicData
    variables: tuple[SymbolicData, ...]
    bounds: tuple[tuple[Real, Real], ...]

    def default_output(self):
        return OptimizationCandidate()

    @property
    def dimension(self):
        """Number of variables."""
        return len(self.variables)

    @property
    def variable_names(self):
        """Names of the variables, in their order."""
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
    """The minimum of a SymPy objective over a finite box of real variables, subject to equality and inequality constraints.

    Build it with keyword arguments, for example
    `ConstrainedOptimization(objective=..., variables=(x,), bounds=((-1.0, 1.0),), inequalities=(sympy.Le(x, 0.5),))`.
    `objective`, `variables`, `bounds` and at least one constraint are
    required. The optional `unit`, `scope`, `coordinates`, `assumptions` and
    `facts` of [`ProblemRecord`][nwqlib.problems.records.ProblemRecord] are
    accepted too. Without `output=`, the requested output is
    [`OptimizationCandidate`][nwqlib.problems.records.OptimizationCandidate].
    The QHD augmented-Lagrangian layer, `solve_augmented_lagrangian`, solves
    it as a sequence of QHD box problems ([Constrained
    problems](../algorithms/qhd.md#constrained-problems)). This record is not
    an `Optimization`, so a Method for box problems alone, such as `QHD`,
    rejects it at planning with `ApplicabilityError`, and no constraint is
    silently dropped.

    The stored constraints are residual expressions with the conventions
    `h_i(x) = 0` for equalities and `g_j(x) <= 0` for inequalities, the
    form of problem (4.1) in Birgin and Martinez, *Practical Augmented
    Lagrangian Methods for Constrained Optimization*, SIAM 2014,
    doi:10.1137/1.9781611973365, with the box as its set Omega. The QHD
    augmented-Lagrangian layer follows that book, so a residual here enters
    its multiplier updates and stopping tests with the book's signs. A
    constraint given as a scalar SymPy expression is the residual itself.
    Live input may also give `Eq(a, b)` as an equality, stored as `a - b`,
    and `Le(a, b)` or `Ge(a, b)` as an inequality, stored as `a - b` or
    `b - a`.

    Attributes:
        objective: Required. Scalar SymPy expression in the variables, not a
            string of source code.
        variables: Required. SymPy symbols with distinct names. Their order
            is the coordinate order.
        bounds: Required. One finite `(lower, upper)` pair with
            `lower < upper` per variable, in the order of `variables`.
        equalities: Default `()`. Residuals `h_i` with the constraint
            `h_i(x) = 0`, given as a tuple or list.
        inequalities: Default `()`. Residuals `g_j` with the constraint
            `g_j(x) <= 0`, given as a tuple or list.

    Raises:
        TypeError: If a constraint field is not a tuple or list, or a
            constraint is not a SymPy expression or relation.
        ValueError: If no constraint is given, if a constraint is a strict
            relation (`Lt`, `Gt`), an `Ne` relation, a relation that SymPy
            has already evaluated to True or False, another Boolean object, a
            matrix or a relation in the wrong field, if a constraint contains
            a symbol outside `variables`, or if the box is invalid as for
            `Optimization`. For a strict inequality, give a non-strict form
            with an explicit margin, for example `Le(x, 1 - margin)` for
            `x < 1`.

    Examples:
        The inequality `x >= 0.5` is stored as the residual `0.5 - x <= 0`:

        >>> import sympy as sp
        >>> from nwqlib import ConstrainedOptimization
        >>> x = sp.Symbol("x", real=True)
        >>> problem = ConstrainedOptimization(
        ...     objective=(x - 0.2)**2, variables=(x,), bounds=((-1.0, 1.0),),
        ...     inequalities=(sp.Ge(x, 0.5),))
        >>> print(problem.inequalities)
        (0.5 - x,)
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
        """Number of variables."""
        return len(self.variables)

    @property
    def variable_names(self):
        """Names of the variables, in their order."""
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
    """The optional field that every output accepts.

    An output names the quantity to compute, for example `Eigenvalue()`,
    and is passed as `output=` to [`solve`][nwqlib.scientist.solve],
    [`plan`][nwqlib.scientist.plan] or [`compare`][nwqlib.scientist.compare].
    It holds no measured value and takes its scope from the Problem. The
    Method's Result holds the value. Build an output class, not this base.
    [Units](../problems.md#units) gives the unit of each output.

    Attributes:
        unit: Default `None`. Label of a quantity whose unit neither the
            Problem nor the output kind defines, for example an observable of
            a `LinearDynamics` Problem. A string becomes a `Unit` of
            dimension `"custom"`. It never converts numerical data.

    Raises:
        ValueError: At planning, if `unit` has another symbol than the unit
            that the Problem defines or the output kind fixes. NWQLib
            converts no units, so omit the output unit or label the Problem
            instead.
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
    """An eigenvalue, in the Problem's unit.

    It is the default output of `Eigenproblem`, whose `target` is the
    smallest eigenvalue. Pass `Eigenvalue()` as `output=` to request it
    explicitly. Its error metric is the absolute error.
    """

    kind: Literal["eigenvalue"] = "eigenvalue"


class Eigenphase(OutputRecord):
    """A phase `phi` in turns, with `U v = exp(2 pi i phi) v` and `phi` in [0, 1).

    It is the default output of `SpectralEstimation`. Pass `Eigenphase()` as
    `output=` to request it explicitly. Its unit is the turn, fixed by the
    output kind. For a Hamiltonian input, U is `exp(-i tau H)` at the time
    tau that the Method selects, so an energy E has phase
    `(-tau E / (2 pi)) mod 1` and the phase order need not follow the
    energy order. A Method that targets the lowest principal eigenphase of a
    supplied unitary `exp(-i tau H)`, as SPE does, therefore reports the
    phase of the highest prepared energy of H when `tau |E| < pi/2` for
    every prepared E. The error metric of this output is the circular
    distance `min(|d|, 1 - |d|)` of the difference d between two phases in
    [0, 1), so estimates on either side of phase zero are close.
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
    """The normalized expectation `u† O u / (u† u)` of a Hermitian observable `O`.

    u is the state of an `Expectation` Problem, or the solution of a
    `LinearDynamics` or `LinearSystem` Problem. The value is undefined when u
    is zero. Build it with keyword arguments, for example
    `NormalizedExpectation(observable=O)`, and pass it as `output=` to
    [`solve`][nwqlib.scientist.solve] or `plan`. An `Expectation` Problem
    requests it by default with its own observable. `observable` is the only
    required argument. For an `Expectation` Problem with a `unit`, the value
    is in that unit. Otherwise the optional `unit` of
    [`OutputRecord`][nwqlib.problems.records.OutputRecord] labels it, and
    without one the unit stays unspecified.

    Attributes:
        observable: Required. Exactly Hermitian observable `O`, in a form
            that [`OperatorData`][nwqlib.problems.records.OperatorData]
            accepts.

    Raises:
        ValueError: If `observable` is not exactly Hermitian. For a Qiskit
            `SparsePauliOp` named `op` whose Hermitian part is the intended
            observable, pass `SparsePauliOp(op.paulis, coeffs=op.coeffs.real)`.
    """

    kind: Literal["normalized_expectation"] = "normalized_expectation"


class QuadraticForm(_ObservableOutput):
    """The quadratic form `u† O u` of a Hermitian observable `O`, using the magnitude of `u`.

    u is the state of an `Expectation` Problem, or the solution of a
    `LinearDynamics` or `LinearSystem` Problem. Build it with keyword
    arguments, for example `QuadraticForm(observable=O)`, and pass it as
    `output=` to [`solve`][nwqlib.scientist.solve] or
    [`plan`][nwqlib.scientist.plan]. `observable` is the only required
    argument. The value's unit stays unspecified unless the optional `unit`
    of [`OutputRecord`][nwqlib.problems.records.OutputRecord] labels it,
    because NWQLib infers neither the observable's unit nor the unit of the
    product `u† O u`. This holds even when the Problem's `unit` labels u,
    as it does for `LinearDynamics` and `LinearSystem`.

    Attributes:
        observable: Required. Exactly Hermitian observable `O`, in a form
            that [`OperatorData`][nwqlib.problems.records.OperatorData]
            accepts.

    Raises:
        ValueError: If `observable` is not exactly Hermitian. For a Qiskit
            `SparsePauliOp` named `op` whose Hermitian part is the intended
            observable, pass `SparsePauliOp(op.paulis, coeffs=op.coeffs.real)`.
    """

    kind: Literal["quadratic_form"] = "quadratic_form"


class NormSquared(OutputRecord):
    """The squared norm `u† u` of a solution `u`, using its magnitude.

    Pass `NormSquared()` as `output=` with a `LinearDynamics` or
    `LinearSystem` Problem. Its unit is the square of the Problem's `unit`
    when the Problem has one.
    """

    kind: Literal["norm_squared"] = "norm_squared"


class Samples(OutputRecord):
    """Measured outcomes in the original coordinates, counted over the shots in which the Method succeeded.

    Pass `Samples()` as `output=`, together with a number of shots. The
    success event is, for example, the post-selection flag of QLS or LCHS.
    `shots` sets the number of shots, and the stored counts include only the
    shots in which that event occurred. Its unit is 1, and its error metric
    is the total variation distance between distributions.
    """

    kind: Literal["samples"] = "samples"

    @property
    def metric(self):
        return "total_variation"


class Solution(OutputRecord):
    """The solution vector in the original coordinates, with its magnitude and phase.

    It is the default output of `LinearDynamics` and `LinearSystem`. Pass
    `Solution()` as `output=` to request it explicitly. Its unit is the
    Problem's `unit`, and its error metric is the l2 norm of the difference.
    """

    kind: Literal["solution"] = "solution"

    @property
    def metric(self):
        return "l2"


class StateVector(OutputRecord):
    """The amplitudes of a simulated state, with a declared normalization and phase convention.

    Build it with keyword arguments, for example
    `StateVector(normalization="physical")`, and pass it as `output=`. Every
    argument is optional. With `normalization="physical"` the vector keeps
    the magnitude of the solution, in the Problem's `unit`. With `"unit"` it
    has length 1 and is dimensionless. Its error metric is the l2 norm of
    the difference, or with `global_phase="modulo_global_phase"` the l2 norm
    after the global phases are aligned. Full amplitudes are a simulator
    readout, not a hardware measurement.

    Attributes:
        normalization: Default `"unit"`. `"physical"` or `"unit"`.
        global_phase: Default `"physical"`. `"physical"`, which compares
            phases as they are, or `"modulo_global_phase"`, which treats
            vectors that differ by one global phase factor as equal.
    """

    kind: Literal["state_vector"] = "state_vector"
    normalization: Literal["physical", "unit"] = "unit"
    global_phase: Literal["physical", "modulo_global_phase"] = "physical"

    @property
    def metric(self):
        return "phase_aligned_l2" if self.global_phase == "modulo_global_phase" else "l2"


class OptimizationCandidate(OutputRecord):
    """A candidate point of the box and its objective value, with the objective gap as error.

    It is the default output of `Optimization` and `ConstrainedOptimization`.
    Pass `OptimizationCandidate()` as `output=` to request it explicitly. The
    objective gap is the candidate's objective value minus a reference
    minimum in objective units, for example the grid minimum that an
    explicit QHD check evaluates. A candidate is not an optimality
    certificate.
    """

    kind: Literal["optimization_candidate"] = "optimization_candidate"

    @property
    def metric(self):
        return "objective_gap"


class Accuracy(_ScientificModel):
    """A requested accuracy: one tolerance, a confidence and the error component it applies to.

    Build it with keyword arguments, for example
    `Accuracy(absolute_tolerance=0.01, component="sampling")`. Exactly one of
    `absolute_tolerance` and `relative_tolerance` is required. The quantity
    and unit it refers to come from the output. Pass it as `accuracy=` to
    [`plan`][nwqlib.scientist.plan], [`solve`][nwqlib.scientist.solve] or
    [`compare`][nwqlib.scientist.compare], where only a Method that defines
    `sampling_shots` uses it, to choose the shots. `ExpectationMethod`
    accepts only an absolute tolerance on the sampling component for
    finite Pauli counts. Pass it to `Result.assess(accuracy=...)` to check a
    Result against it, for the total or the sampling component. An Accuracy
    states a request and records no achieved accuracy. A total or relative
    target is not silently interpreted as the narrower sampling request.

    Attributes:
        absolute_tolerance: Default `None`. Positive bound on the error, in
            the unit of the output.
        relative_tolerance: Default `None`. Positive bound on the error
            divided by the magnitude of the target.
        confidence: Default `0.95`. Probability, strictly between 0 and 1,
            with which the bound must hold.
        component: Default `"total"`. `"total"` for the whole error of the
            output, or `"sampling"` for the sampling error only.

    Raises:
        ValueError: If both or neither tolerance is given, or a value is out
            of its range.
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
