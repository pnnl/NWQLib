"""Configured QLS methods on natural A,b, with actual selected model execution."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import isfinite
from typing import Annotated, Any, ClassVar, Literal
import numpy as np
from pydantic import Field, field_serializer, field_validator, model_validator
from nwqlib.algorithms.protocol import Method, ApplicabilityError, AlgorithmDescriptor
from nwqlib.artifacts import ArrayOutput, UnavailableOutput
from nwqlib.blocks.kernels import BoundKernel, KernelOutput
from nwqlib.blocks.records import SelectedConstruction, SelectedKernel
from nwqlib.blocks.selection import SelectedBlock
from nwqlib._validation import NUMERICAL_RELATION_RTOL, validate_normalized_mass
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.core.planning import Plan, Experiment, ObservationSpec
from nwqlib.core.records import Float64, PositiveInt, Real, Source
from nwqlib.evidence import Evidence, Fact
from nwqlib.evidence.error_model import ErrorModel, ErrorTerm, FramedFact
from nwqlib.execution import KernelApplication, ScalarValue
from nwqlib.ir import ClassicalStage, ClassicalValue, Definition, Program
from nwqlib.problems.inputs import compose_recovery
from nwqlib.problems.records import (
    LinearSystem,
    Solution,
    StateVector,
    NormSquared,
    QuadraticForm,
    NormalizedExpectation,
    Samples,
)
from nwqlib.problems.scalars import physical_scalar, physical_vector_statistics, _observable_moment
from nwqlib.resources.records import ResourceLaw, Workspace
from .host_planning import (
    application_arguments, input_access_refusal, original_factor_laws, select_inputs, select_polynomial,
    selected_work,
)
from .primary_records import METHOD, DESCRIPTOR, QLSAnalysis, QLSReconstruction, validate_selection

HOST = Source(
    name="qls.classical_model",
    version="2",
    domain="selected original-input polynomial model",
    reference="inverse polynomial eigenvalue calculus or Dalzell arXiv:2406.12086v2 singular-value/dilation model; no quantum execution",
)
_ANALYSIS_SOURCE = Source(
    name="qls.analysis",
    version="2",
    domain="actual original-coordinate acquisitions",
    reference="nwqlib.algorithms.qls.method",
)
UNKNOWN_SCALE = (
    "shortcut unit direction has no physical magnitude or physical global-phase reconstruction"
)


def _unknown(name, frame, reason):
    return FramedFact(
        frame=frame,
        bindings=(),
        fact=Fact(
            quantity=name, unit=frame.unit, scope=frame.scope, availability="unknown", reason=reason
        ),
    )


def _error_model(problem, method, output, execution, shots, construction, rec):
    """Declare every error source of the selected route as unknown.

    QLS publishes no propagated total-error certificate. The polynomial
    certificate, spectral premises, native execution and sampling remain
    separate component facts, and explicit ``QLSVerification`` supplies
    scoped comparisons. Listing the required sources as unknown keeps a
    consumer from reading a missing term as zero.
    """
    frame = output.frame(problem)
    required = ("algorithmic_approximation", "floating_point")
    if execution == "quantum":
        required += ("native_execution",) + (("sampling",) if shots is not None else ())
    terms = tuple(
        ErrorTerm(
            name=name,
            stage="analysis",
            source=METHOD,
            formula="complete propagated physical-output error is not established",
            coverage=name,
            fact=_unknown(name, frame, "selected component evidence does not establish " + name),
        )
        for name in required
    )
    return ErrorModel(
        output_id=output.content_id,
        subject_id=problem.content_id,
        construction_id=construction.content_id,
        frame=frame,
        required_sources=required,
        terms=terms,
        source=METHOD,
    )


def _array_output(problem, output):
    """Return the ``ArrayOutput`` of a vector output, or ``None`` for scalar and sample outputs.

    ``Solution`` publishes the physical ``x`` with its physical global phase.
    ``StateVector`` publishes in its requested normalization and phase
    convention. Classical and quantum planning and
    ``QLSAnalysis.validate_plan`` call this one function, so the declared
    and the published frames agree.
    """
    if isinstance(output, Solution):
        return ArrayOutput(
            name="solution",
            kind="vector",
            basis=problem.basis,
            frame="physical",
            global_phase="physical",
        )
    if isinstance(output, StateVector):
        return ArrayOutput(
            name="solution",
            kind="vector",
            basis=problem.basis,
            frame=output.normalization,
            global_phase=output.global_phase,
        )
    return None


def _admit(problem, method, output, execution, shots):
    """Reject output/access combinations whose requested physical quantity the selected QLS route
    cannot recover.
    """
    if not isinstance(problem, LinearSystem) or not isinstance(
        output, (Solution, StateVector, NormSquared, QuadraticForm, NormalizedExpectation, Samples)
    ):
        raise ApplicabilityError(
            "QLS requires LinearSystem and a solution/state/scalar/sample output"
        )
    observable = isinstance(output, (QuadraticForm, NormalizedExpectation))
    if observable and output.observable.basis != problem.basis:
        raise ValueError("QLS observable must use the original A,b coordinates")
    if (
        execution == "quantum"
        and observable
        and output.observable.reference.representation != "dense"
        and "pauli_terms" not in output.observable.manifest.access
    ):
        raise ApplicabilityError("quantum QLS observable requires dense or finite Pauli access")
    if isinstance(output, (Solution, StateVector)) and shots is not None:
        raise ValueError("QLS simulator vector requires exact amplitude acquisition")
    if isinstance(output, Samples) and shots is None:
        raise ValueError("QLS Samples requires an explicit positive shot count")
    if execution == "classical":
        if shots is not None or isinstance(output, Samples):
            raise ApplicabilityError(
                "classical QLS evaluates its numerical polynomial model without sampled circuit acquisition"
            )
        if (
            problem.A.reference.representation != "dense"
            or problem.b.reference.representation not in {"vector", "product", "occupation"}
        ):
            raise ApplicabilityError(
                input_access_refusal(problem.A, "classical")
                or "classical QLS requires original dense A and explicit numerical RHS access"
            )
        if observable and output.observable.reference.representation != "dense":
            raise ApplicabilityError(
                "classical QLS observable acquisition requires explicit dense observable access"
            )
    # Shortcut routes provide a direction without recovering the physical
    # solution norm, so unnormalized outputs cannot be admitted.
    if method.solver != "qsvt_inverse":
        if method.encoded_solution_norm_estimate is None:
            raise ApplicabilityError(
                "shortcut requires an explicit numeric t or named classical norm model"
            )
        if (
            isinstance(output, (Solution, NormSquared, QuadraticForm))
            or isinstance(output, StateVector)
            and (output.normalization != "unit" or output.global_phase != "modulo_global_phase")
        ):
            raise ApplicabilityError(
                "shortcut has no physical x scale; select unit StateVector modulo global phase, NormalizedExpectation or Samples"
            )
        if execution == "quantum" and isinstance(method.encoded_solution_norm_estimate, str):
            raise ApplicabilityError(
                "named norm-search probability models require explicit classical execution; quantum shortcut needs numeric t"
            )


class QLS(Method):
    """Quantum linear-system (QLS) method for `A x = b` given by `LinearSystem`.

    Build it with keyword arguments and pass it as `method=`, for example
    `solve(LinearSystem(A=A, b=b), method=QLS())`. Every argument is
    optional. The result is a
    [`QLSAnalysis`][nwqlib.algorithms.qls.primary_records.QLSAnalysis]
    whose `x`, for the default `Solution` output, is the physical solution
    in the original coordinates, including scale and phase, in the
    problem's `unit`.

    The default `solver="qsvt_inverse"` applies an odd Chebyshev polynomial
    P to `A/alpha`, through a Hermitian dilation when A is not Hermitian, by
    quantum singular value transformation (QSVT, Gilyén et al.,
    arXiv:1806.01838v1, Theorem 17, in the conventions of Martyn et al.,
    arXiv:2105.02859v5). Here alpha is the normalization of the block
    encoding of A, and the polynomial domain parameter `polynomial_kappa` is
    at least 1.01 and at least the encoded gap parameter `kappa_be`, which
    is `max(1, alpha/sigma_min(A))` or the supplied `kappa`. All three are
    recorded in `plan.reconstruction`. P approximates
    `1/(polynomial_kappa x)` with
    `|polynomial_kappa x P(x) - 1| <= epsilon_inv` for
    `1/polynomial_kappa <= |x| <= 1`. NWQLib fits P and evaluates this bound
    in binary64 on an affine Chebyshev grid (Ehlich and Zeller,
    doi:10.1007/BF01111276, Satz 2), and checks its degree against the
    degree bound of Childs, Kothari and Somma (arXiv:1511.02306v2, Lemmas
    17-19). Physical recovery multiplies the success branch of the circuit,
    or of the classical model, by `||b|| polynomial_kappa s / alpha`, with s
    the polynomial's rescale.
    The two shortcut solvers implement Dalzell's kernel-reflection method
    (arXiv:2406.12086v2, Algorithm 1) and return only the unit direction of
    x, modulo global phase.

    `epsilon_inv` sets the polynomial approximation. It does not bound the
    total error of x, which also depends on the spectral assumptions, the
    phase fit, the circuit execution and sampling. Vector outputs
    (`Solution`, `StateVector`) need exact amplitude readout, so they take
    no `shots`, and `Samples` needs positive `shots` with quantum execution.
    `execution="classical"` evaluates the chosen polynomial model on the
    host. It needs a dense A, a vector, product or occupation b and, for an
    observable output, a dense observable, and it takes no `shots`.
    `result.analyze()` takes no settings, so another polynomial needs a new
    Plan. `result.verify(checks=...)` takes a
    [`QLSVerification`][nwqlib.algorithms.qls.verification.QLSVerification].
    The [QLS guide](../../algorithms/qls.md) explains physical outputs,
    normalization, spectral assumptions and the shortcut restrictions, and
    its source map gives every equation and its code.

    Examples:
        The exact solution of this system is `(25/28, 5/28)`, about
        `(0.8929, 0.1786)`. `epsilon_inv=0.01` sets the inverse polynomial,
        not a bound on the difference.

        >>> import numpy as np
        >>> from nwqlib import LinearSystem, solve
        >>> from nwqlib.algorithms.qls import QLS
        >>> problem = LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1., .25])
        >>> result = solve(problem, method=QLS(), seed=7)
        >>> print(np.round(result.x, 4))
        [0.8966+0.j 0.1813+0.j]

    Attributes:
        solver: Default `"qsvt_inverse"`, which recovers the physical
            solution. `"shortcut_native_svp"` (Dalzell's kernel reflection
            on the right singular vectors of the augmented matrix `G_t`,
            Dalzell Eqs. (8)-(11)) and `"shortcut_dilation"` (an even
            polynomial on the Hermitian dilation of `G_t`) give only a unit
            `StateVector` modulo global phase, a `NormalizedExpectation` or
            `Samples`, and need `encoded_solution_norm_estimate`.
        epsilon_inv: Default `0.01`, strictly between 0 and 1. Polynomial
            construction target. For `"qsvt_inverse"` it bounds
            `|polynomial_kappa x P(x) - 1|` on the polynomial domain
            `1/polynomial_kappa <= |x| <= 1`. When every eigenvalue of a
            Hermitian `A/alpha`, or every singular value through the
            dilation, lies in that domain, the vector y after the polynomial
            step satisfies
            `||y - (polynomial_kappa A/alpha)^-1 b|| <= epsilon_inv ||(polynomial_kappa A/alpha)^-1 b||`.
            For the shortcuts it is `sqrt(2)` times the kernel-reflection
            parameter `eta`. It does not bound the total error of the
            physical output.
        alpha: Default `"auto"`, which takes the normalization of the chosen
            block encoding, for Pauli input the coefficient 1-norm. A
            positive number is the caller's assumption. Supplied `alpha` and
            `kappa` do not force an extra spectral computation. When planning
            computes the original singular endpoints anyway, an assumption
            that misses them by more than a relative window of `1e-12` is
            refused.
        kappa: Default `"auto"`, which uses `kappa_be = alpha/sigma_min(A)`
            from the original singular endpoints, at least 1. A number at
            least 1 is the caller's bound on `kappa_be`, the encoded gap
            parameter, which is not the condition number
            `sigma_max/sigma_min`. Pauli A, and a non-dense A with a
            supplied `encoding`, need a number at least
            `alpha/sigma_min(A)`, because QLS computes no singular values
            for them. The polynomial domain parameter `polynomial_kappa` is
            at least `kappa_be` and at least 1.01.
        block_encoding_implementation: Default `"auto"`, which chooses among
            the banded, Pauli and dense encodings that apply to the input.
            `"pauli_lcu"`, `"multiplexed_pauli"`, `"banded"` and
            `"dense_dilation"` request one family, which the input must
            support.
        encoding: Default `None`. A supplied block encoding of the original
            A, with its declared operator. Its projected equation and error
            are the caller's assumptions, not checked by a dense test.
        dense_control_route: Default `"auto"`. How a controlled query
            controls a dense-dilation encoding. `"gatewise"` synthesizes the
            dilation and lets Qiskit control each synthesized gate,
            `"whole_matrix"` synthesizes the controlled dilation, and
            `"auto"` takes the whole-matrix route for one control and the
            gate-wise route for more. With the default limits, `"auto"`
            accepts a controlled dense dilation of at most 64 padded
            coordinates. A supplied encoding and a supplied RHS circuit are
            controlled gate-wise on every route.
        encoded_solution_norm_estimate: Default `None`. Dalzell's norm
            parameter t of a shortcut, a number with
            `1 <= t <= polynomial_kappa`, which quantum shortcuts require.
            Classical shortcuts also accept `"grid"`,
            `"noisy_binary_search"` or `"linear_kappa_sequence"` (Dalzell
            Secs. 5.1-5.3), which evaluate the classical probability model
            of that search. t is not the recovered physical norm.
            `"qsvt_inverse"` refuses any value.
        max_degree: Default `256`. Upper limit on the polynomial degree,
            checked before coefficient and phase work.
        max_qsp_evaluations: Default `20_000`. Upper limit on the residual
            evaluations of the QSP phase solver.
        max_work: Default `1e9` (`1_000_000_000`). Upper limit on the
            counted classical work, applied separately to each planning
            phase, for example spectral selection, dense completion,
            encoding construction, the polynomial fit and the grouping of
            sampled Pauli terms. It includes the exact synthesis of a dense
            dilation, or of the dense unitaries in a supplied encoding or
            RHS circuit, that a controlled query needs, and Qiskit's control
            of the synthesized gates. The exact scalar readouts of one Run
            count together against it. Work units are operation counts, not
            timings.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Upper
            limit on the known bytes of numerical arrays, including those
            syntheses. It excludes undocumented vendor workspace. Each exact
            scalar readout must fit it before the circuits run.
        max_admission_steps: Default `1_000_000`, ten times the shared
            default, because the planning work of a sampled `Program` grows
            with the number of its distinct measured registers. Upper limit
            on the planning work of checking the quantum `Program` (NWQLib's
            description of a circuit as named steps), counting its stored
            fields and the work of validation, preparation and circuit
            building. A refusal with a complete count names a value that
            passes. Summing the Program's resource counts may use up to 24
            times this value. Raising it changes neither the polynomial nor
            any quantum operation. See the
            [planning work limit](../../development/program_checks.md#planning-work-limit).

    Raises:
        ValueError: If `encoded_solution_norm_estimate` is set with
            `"qsvt_inverse"`.
        TypeError: If `encoding` is not a selected block encoding.
    """

    result_type: ClassVar[type] = QLSAnalysis
    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR
    solver: Literal["qsvt_inverse", "shortcut_native_svp", "shortcut_dilation"] = "qsvt_inverse"
    epsilon_inv: Annotated[Real, Field(gt=0, lt=1)] = 0.01
    alpha: Annotated[Real, Field(gt=0)] | Literal["auto"] = "auto"
    kappa: Annotated[Real, Field(ge=1)] | Literal["auto"] = "auto"
    block_encoding_implementation: Literal[
        "auto", "pauli_lcu", "multiplexed_pauli", "banded", "dense_dilation"
    ] = "auto"
    encoding: Any = None
    dense_control_route: Literal["gatewise", "whole_matrix", "auto"] = "auto"
    encoded_solution_norm_estimate: (
        Annotated[Real, Field(ge=1)]
        | Literal["grid", "noisy_binary_search", "linear_kappa_sequence"]
        | None
    ) = None
    max_degree: PositiveInt = 256
    max_qsp_evaluations: PositiveInt = 20000
    max_work: PositiveInt = 1_000_000_000
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    # Ten times the shared Program default (ENGINEERING_CONSTANTS.md, "Shared
    # Program admission limits").
    max_admission_steps: PositiveInt = 1_000_000

    @field_validator("encoding")
    @classmethod
    def _encoding(cls, value):
        if value is not None and not isinstance(value, SelectedBlock):
            raise TypeError("encoding requires an actual selected block encoding")
        return value

    @field_serializer("encoding")
    def _describe_encoding(self, value):
        return (
            None
            if value is None
            else value.record.model_dump(mode="json", exclude_computed_fields=True)
        )

    @model_validator(mode="after")
    def _choices(self):
        if self.solver == "qsvt_inverse" and self.encoded_solution_norm_estimate is not None:
            raise ValueError("numeric t and norm models belong to shortcut solvers")
        return self

    def plan(self, problem, *, output, execution, shots, rng):
        """Select the encoding, inverse/reflection polynomial and physical recovery before
        declaring execution.
        """
        _admit(problem, self, output, execution, shots)
        encoding, encoded_operator, rhs, svd, spectrum = select_inputs(
            problem, self, execution=execution
        )
        polynomial = select_polynomial(self, spectrum["polynomial_kappa"])
        # A non-Hermitian inverse uses Hermitian dilation. Its auxiliary coordinate
        # is distinct from power-of-two padding of the original input.
        embedding = (
            "hermitian_dilation"
            if self.solver == "qsvt_inverse" and problem.A.structure != "hermitian"
            else "none"
        )
        t = self.encoded_solution_norm_estimate
        if isinstance(t, float) and t > spectrum["polynomial_kappa"]:
            raise ValueError("numeric shortcut t exceeds the selected polynomial domain")
        source = (
            "not_applicable"
            if self.solver == "qsvt_inverse"
            else t
            if isinstance(t, str)
            else "user"
        )
        array = _array_output(problem, output)
        held = 0
        if execution == "classical":
            hermitian = problem.A.structure == "hermitian"
            work = selected_work(
                problem.A.dense_array(),
                alpha=spectrum["alpha"],
                embedded=embedding != "none",
                method=self.solver,
                polynomial=polynomial,
                kappa=spectrum["polynomial_kappa"],
                t_source=source,
                output=output,
                outputs=() if array is None else (array,),
                hermitian=hermitian,
                factors_held=svd is not None,
            )
            # The admission limit counts all known resident arrays: original
            # factors already owned by the Plan add their persistent bytes to
            # the evaluation's workspace, and are not charged again as work.
            if svd is not None:
                held = original_factor_laws(problem.dimension, hermitian)[1]
        else:
            from .primary_records import QLSWork

            work = QLSWork(
                original_dimension=problem.dimension,
                system_dimension=problem.dimension * (2 if embedding != "none" else 1),
            )
        if work.size_units > self.max_work or work.workspace_bytes + held > self.max_bytes:
            raise ValueError("selected QLS classical action exceeds max_work or max_bytes")
        # Compose the RHS norm, encoding normalization and polynomial rescaling
        # once, so readout can recover the physical x rather than only its direction.
        recovery = (
            compose_recovery(
                problem.b.preparation.physical_scale,
                (spectrum["polynomial_kappa"], spectrum["alpha"]),
                polynomial.rescale,
            )
            if self.solver == "qsvt_inverse"
            else None
        )
        rec = QLSReconstruction(
            original_operator=problem.A.reference,
            original_rhs=problem.b.reference,
            encoding_operator=encoded_operator.reference,
            original_dimension=problem.dimension,
            padded_dimension=encoded_operator.basis.dimension,
            system_dimension=encoded_operator.basis.dimension * (2 if embedding != "none" else 1),
            embedding=embedding,
            rhs_scale=problem.b.preparation.physical_scale,
            polynomial=polynomial,
            t=t if isinstance(t, float) else None,
            t_source=source,
            encoding_id=encoding.record.content_id,
            encoding_family=encoding.record.implementation.domain,
            encoding_ancillas=encoding.record.semantics.index_qubits,
            encoding_error=encoding.record.semantics.epsilon,
            recovery=recovery,
            work=work,
            **spectrum,
        )
        if execution == "quantum":
            from .quantum import plan_quantum

            return plan_quantum(
                self,
                problem,
                output=output,
                shots=shots,
                rng=rng,
                reconstruction=rec,
                encoding=encoding,
                encoded_operator=encoded_operator,
                rhs=rhs,
                svd=svd,
            )
        return _classical_plan(self, problem, output=output, rng=rng, rec=rec, svd=svd)

    def analyze(self, plan, data, *, settings):
        if settings:
            raise ValueError("QLS has no posthoc algorithm changes; select a new Plan")
        validate_selection(plan)
        if plan.execution == "quantum":
            from .quantum import analyze_quantum

            return analyze_quantum(plan, data)
        return _analyze_classical(plan, data)

    def reduction_allowance(self, plan, point, *, observation, width, run):
        """Admit the exact reduction's workspace and return the remaining ``max_work``.

        Before native acquisition, each ``projected_moments`` point's
        ``readout_requirements`` bytes must fit ``max_bytes``. The returned
        allowance is ``max_work`` minus the registered work of the reductions
        this Run already completed for other experiments, against which the
        registry admits this point's reduction. QLS caps each planning phase,
        such as grouping, by ``max_work`` separately, so the reductions of a
        Run are charged only against each other. The hook reads that ledger
        without changing it, so it can be asked again on submission and
        after a reopen.
        """
        import json
        from nwqlib._quantum_readout import PROJECTED_MOMENTS, projected_requirements

        def reductions(readout):
            return (item for item in readout.positions
                    if item.kind == "reduction" and item.reducer == PROJECTED_MOMENTS)

        for item in reductions(observation):
            reserved, _ = projected_requirements(json.loads(item.parameters), width)
            if reserved > self.max_bytes:
                raise ValueError(f"QLS projected reduction needs {reserved} bytes, more than "
                                 f"max_bytes={self.max_bytes}; refused before acquisition")
        spent = sum(projected_requirements(json.loads(item.parameters), width)[1]
                    for chunk in run.observations.chunks
                    if chunk.point is not None and chunk.experiment != point.experiment
                    for item in reductions(chunk.readout()))
        return max(0, self.max_work - spent)

    def validate_point(self, plan, experiment, values):
        if not plan._bound:
            raise ValueError("QLS requires its actual selected native input bindings")
        validate_selection(plan)

    def save_archive(self, plan, files):
        from .archive import save

        return save(self, plan, files)

    @classmethod
    def load_archive(cls, data, files):
        from .archive import load

        return load(data, files)

    def verify(self, plan, result, *, checks):
        from .verification import verify

        return verify(plan, result, checks=checks)

    def save_run_context(self, context, files):
        """Save the Run's realized base-query gates as QPY instructions.

        ``_query_circuit`` in ``quantum.py`` caches one realized gate per
        selected encoding or RHS preparation, and its derived inverse, under
        ``qls_base_gates``. Saving them lets a loaded Run reuse the same gates
        instead of synthesizing the encodings again.
        """
        return dict(
            gates=tuple(
                (key, files.write_instruction(f"qls-query-{i}.qpy", gate))
                for i, (key, gate) in enumerate(context.get("qls_base_gates", {}).items())
            )
        )

    def load_run_context(self, data, files):
        """Restore the ``qls_base_gates`` cache written by ``save_run_context``.

        JSON stores an adjoint key ``(base_id, "adjoint")`` as a list, so it
        is turned back into a tuple.
        """
        return dict(
            qls_base_gates={
                tuple(key) if isinstance(key, list) else key: files.read_instruction(name)
                for key, name in data["gates"]
            }
        )


def _classical_plan(method, problem, *, output, rng, rec, svd):
    """Declare the selected polynomial action and its physical or encoded-branch scalar outputs."""
    array = _array_output(problem, output)
    observable = (
        output.observable if isinstance(output, (NormalizedExpectation, QuadraticForm)) else None
    )
    scalars = ("norm_squared", "algorithm_success_mass", "physical_slice_mass") + (
        ("numerator",) if observable is not None else ()
    )
    frames = ("physical", "encoded_branch", "encoded_branch") + (
        ("unit" if isinstance(output, NormalizedExpectation) else "physical",)
        if observable is not None
        else ()
    )
    inputs = (problem.A.reference, problem.b.reference) + (
        () if observable is None else (observable.reference,)
    )
    inputs = tuple(dict.fromkeys(inputs))
    # Declare bounded work and workspace without invoking the numerical model.
    # Scalar frames distinguish physical norm from encoded success populations.
    kernel = SelectedKernel(
        name="polynomial_action",
        implementation=HOST,
        inputs=inputs,
        scalars=scalars,
        scalar_frames=frames,
        outputs=() if array is None else (array,),
        resource_laws=(
            ResourceLaw(
                metric="classical_work",
                basis="selected_logical",
                value=rec.work.size_units,
                interpretation="upper_bound",
                evidence=Evidence(kind="proved_relation", source=HOST),
                assumptions=(
                    "known numerical size units; vendor workspace and elapsed time unknown",
                ),
            ),
        ),
        workspace=(
            Workspace(
                location="host", purpose="workspace", bytes=rec.work.workspace_bytes, source=HOST
            ),
        ),
        construction_work=len(inputs) + len(scalars),
        invocation_work=rec.work.size_units,
        dependencies=("numpy", "scipy"),
    )
    stage = ClassicalStage(
        implementation=HOST, boundary="host", outputs=scalars, kernel=kernel.name
    )
    program = Program(
        root="solution",
        definitions=(Definition(id="solution", node=stage),),
        classical=tuple(ClassicalValue(name=name, dtype="real") for name in scalars),
    )
    construction = SelectedConstruction(program=program, selections=(), kernels=(kernel,))
    plan = Plan(
        problem=problem,
        method=method,
        output=output,
        execution="classical",
        randomness=rng.snapshot(),
        construction=construction,
        reconstruction=rec,
        experiments=(
            Experiment(
                name="solution",
                setting="solution",
                observation=ObservationSpec(kind="host_scalars", labels=scalars),
            ),
        ),
        error_model=_error_model(problem, method, output, "classical", None, construction, rec),
        assumptions=(
            "explicit classical selected polynomial model; native phase/synthesis error is not simulated",
        ),
    )
    # The classical kernel reads the original A and b only. The Plan keeps
    # the selected encoding's alpha, family, ancilla count and error bound in
    # its reconstruction, not the encoding payload, and binds the original
    # factors when planning acquired them for the selected consumers.
    plan._bind(blocks=(_bind_host(plan),), svd=svd)
    validate_selection(plan)
    return plan


def _bind_host(plan):
    """Bind the Plan's single host kernel to a deferred call of ``_execute``.

    The callable reads the original dense ``A``, the unit RHS direction
    and the observable of a scalar output only when the kernel runs.
    Planning and archive loading both bind through this function, so a
    loaded classical Plan runs the same model as a fresh one.
    """
    from .numerical import _rhs_direction

    kernel = plan.construction.kernels[0]
    observable = (
        plan.output.observable
        if isinstance(plan.output, (NormalizedExpectation, QuadraticForm))
        else None
    )
    return BoundKernel._bind(
        plan,
        kernel,
        lambda: _execute(
            plan,
            kernel,
            matrix=plan.problem.A.dense_array(),
            rhs_direction=_rhs_direction(plan.problem.b),
            observable=observable,
        ),
    )


def _classical_publication(plan, chunk):
    """Return the published fields of a classical Result, formed from its host acquisition chunk.

    The keys are the ``QLSAnalysis`` fields ``scalar_value``,
    ``unavailable``, ``norm_squared``, ``physical_scale``,
    ``physical_scale_unavailable``, ``numerator``, ``numerator_frame`` and
    ``artifact``. A scalar output publishes ``physical_scalar`` of the
    chunk's norm squared, physical scale and numerator. The Result publishes
    the chunk's first artifact, or without an artifact the chunk's first
    unavailability reason, if any. ``_analyze_classical`` calls this.
    """
    values = {value.label: value for value in chunk.values}
    scale = chunk.physical_scale
    norm = values["norm_squared"].value
    numerator = values.get("numerator")
    scalar, unavailable = None, None
    if isinstance(plan.output, (NormSquared, QuadraticForm, NormalizedExpectation)):
        scalar, unavailable = physical_scalar(
            plan.output.kind,
            norm_squared=norm,
            scale=scale,
            numerator=None if numerator is None else numerator.value,
            numerator_frame="physical" if numerator is None else numerator.frame,
            numerator_unavailable=None if numerator is None else numerator.unavailable,
            norm_unavailable=values["norm_squared"].unavailable,
        )
    artifact = chunk.artifacts[0] if chunk.artifacts else None
    if artifact is None and chunk.unavailable:
        unavailable = chunk.unavailable[0].reason
    return dict(
        scalar_value=scalar,
        unavailable=unavailable,
        norm_squared=norm,
        physical_scale=scale,
        physical_scale_unavailable=chunk.physical_scale_unavailable,
        numerator=None if numerator is None else numerator.value,
        numerator_frame=None if numerator is None else numerator.frame,
        artifact=artifact,
    )


def _analyze_classical(plan, data):
    """Read one associated host acquisition and reconstruct its requested physical scalar or
    stored vector.
    """
    if len(data.observations.chunks) != 1:
        raise ValueError("classical QLS requires one completed model acquisition")
    chunk = data.observations.chunks[0]
    data.trace.validate_observation(chunk)
    realization = plan.resolve("solution")
    kernel = plan.construction.kernels[0]
    if (
        chunk.plan_id != plan.content_id
        or chunk.run_id != data.trace.run_id
        or chunk.realization_id != realization.content_id
        or chunk.execution != "host_kernel"
        or chunk.selected_kernel_id != kernel.content_id
        or chunk.source != HOST
        or tuple(value.label for value in chunk.values) != kernel.scalars
        or tuple(value.frame for value in chunk.values) != kernel.scalar_frames
    ):
        raise ValueError("QLS classical observation differs from the selected original-input model")
    _validate_application(plan, chunk.applications)
    values = {value.label: value for value in chunk.values}
    published = _classical_publication(plan, chunk)
    if published["artifact"] is not None:
        data.artifact(published["artifact"])
    result = QLSAnalysis(
        plan_id=plan.content_id,
        construction_id=plan._construction_id,
        observation_id=data.observations.content_id,
        contribution_ids=(chunk.content_id,),
        execution="classical",
        origin=capture_analysis_origin(
            analyzer=_ANALYSIS_SOURCE,
            method_id=plan.method.content_id,
            dependencies=("numpy", "scipy"),
        ),
        **published,
        algorithm_success_mass=values["algorithm_success_mass"].value,
        physical_slice_mass=values["physical_slice_mass"].value,
        applications=chunk.applications,
        mass_contribution_id=chunk.content_id,
    )
    # A host kernel's roundoff window is the fixed floor.
    for name in ("algorithm_success_mass", "physical_slice_mass"):
        validate_normalized_mass(values[name].value, NUMERICAL_RELATION_RTOL)
    result.validate_plan(plan)
    return result


def _validate_application(plan, applications):
    """Check that a host application reports the selected workload.

    Every recorded argument must equal its selected value, except that a
    named norm-search model's realized rows, trials and queries may fall
    below the selected envelope, its ``t`` may be any value in the
    polynomial domain, and its recorded encoded reference norm may be any
    finite positive value. This keeps a stored observation from being analyzed
    under a Plan whose workload it did not execute.
    """
    if (
        len(applications) != 1
        or applications[0].name != plan.construction.kernels[0].name
        or applications[0].implementation != plan.construction.kernels[0].implementation
    ):
        raise ValueError("QLS requires one actual selected application")
    expected = {b.parameter: b.value for b in application_arguments(plan.reconstruction)}
    actual = {b.parameter: b.value for b in applications[0].arguments}
    if actual.keys() != expected.keys():
        raise ValueError("QLS application parameters differ from selected workload")
    for name, value in actual.items():
        raw = value.value if isinstance(value, Float64) else value
        bound = expected[name].value if isinstance(expected[name], Float64) else expected[name]
        if name == "t_value":
            source = plan.reconstruction.t_source
            if source == "not_applicable":
                if raw != 1.0:
                    raise ValueError("inverse QLS requires its non-applicable t=1 placeholder")
            elif source == "user":
                if raw != plan.method.encoded_solution_norm_estimate:
                    raise ValueError("QLS observed t differs from the selected fixed numeric value")
            elif not 1 <= raw <= plan.reconstruction.polynomial_kappa:
                raise ValueError("QLS observed model t lies outside selected domain")
        elif name == "encoded_reference_norm":
            if not (isfinite(raw) and raw > 0):
                raise ValueError("QLS recorded encoded reference norm must be finite and positive")
        elif name in {"search_rows", "planned_trials", "planned_queries"}:
            if not 0 <= raw <= bound:
                raise ValueError("QLS realized model accounting exceeds selected envelope")
        elif value != expected[name]:
            raise ValueError("QLS observed work differs from selected reconstruction")


def _realize(plan, *, matrix, rhs_direction):
    """Return the selected physical slice and separate positive binary recovery.

    The inverse and the linear norm model reuse the original factors that
    planning acquired (``plan._native["svd"]``), or acquire them once here
    when supplied ``alpha`` and ``kappa`` deferred the factorization
    (``host_planning._original_factors``, charged in ``QLSWork``). The
    inverse applies its odd polynomial on the d original coordinates and
    returns the d-entry branch ``y = V P(Sigma/alpha) U† b_hat`` directly,
    so its algorithm-branch mass and physical-slice mass both equal
    ``||y||**2`` (``numerical.inverse_polynomial_action``). For the inverse,
    ``x = ||b|| (kappa_poly s / alpha) (P/s)(A/alpha) b_hat``, and the
    returned recovery composes that scale in binary form. A shortcut
    normalizes original A once; the grid and noisy-search norm models make
    one encoded reference solve to evaluate their success-probability model,
    that solve never supplies the output, and its norm is recorded as the
    ``encoded_reference_norm`` argument of the application. Physical array
    materialization belongs only to an explicit output request.
    """
    from nwqlib._numerics import stable_vector_norm, normalized_matrix
    from .host_planning import _original_factors, classical_factor_consumer
    from .numerical import (
        inverse_polynomial_action,
        shortcut_matrix,
        shortcut_polynomial_action,
    )
    from .norm_search import _resolve_shortcut_norm

    r = plan.reconstruction
    factors = plan._native.get("svd")
    if factors is not None:
        factors.require_source(plan.problem.A)
    elif classical_factor_consumer(plan.method):
        factors = _original_factors(plan.problem.A, plan.method)
    coefficients = np.asarray(r.polynomial.coefficients) / r.polynomial.rescale
    t_value, source, metadata, reference_norm = 1.0, "not_applicable", {}, None
    if plan.method.solver == "qsvt_inverse":
        left, values, right_h = factors.frames()
        selected, mass = inverse_polynomial_action(
            left, values, rhs_direction, alpha=r.alpha, coefficients=coefficients, right_h=right_h)
        slice_mass = mass
        recovery = compose_recovery(
            r.rhs_scale, (r.polynomial_kappa, r.alpha), r.polynomial.rescale
        )
    else:
        encoded = normalized_matrix(matrix, r.alpha)
        if r.work.target_solves:
            # y=(A/alpha)^-1*b_hat is already in the t model's encoded frame.
            # This is the selected classical norm-search model's only solve.
            reference_norm = stable_vector_norm(np.linalg.solve(encoded, rhs_direction))
        t_value, source, metadata = _resolve_shortcut_norm(
            plan.method.encoded_solution_norm_estimate,
            kappa_be=r.polynomial_kappa,
            encoded_norm=reference_norm,
            eta=r.polynomial.eta,
            factors=factors,
            alpha=r.alpha,
            rhs=rhs_direction,
        )
        g_t = shortcut_matrix(encoded, rhs_direction, t_value=t_value)
        del encoded
        selected, mass = shortcut_polynomial_action(
            g_t, coefficients, system_dimension=r.work.system_dimension, method=plan.method.solver
        )
        slice_mass, recovery = mass, None
    rows = (
        len(metadata.get("rows", ()))
        + len(metadata.get("rounds", ()))
        + sum(len(s["rows"]) for s in metadata.get("steps", ()))
    )
    trials = metadata.get("planned_trials_total", metadata.get("total_trials", 0))
    queries = metadata.get("planned_queries_total", 0)
    if source != r.t_source:
        raise ValueError("realized t source differs from selected route")
    arguments = application_arguments(r, t_value=t_value, rows=rows, trials=trials, queries=queries,
                                      reference_norm=reference_norm)
    return selected, recovery, mass, slice_mass, arguments


def _execute(plan, kernel, *, matrix, rhs_direction, observable):
    """Evaluate the selected polynomial model and publish its recoverable physical statistics and
    arrays.
    """
    selected, recovery, mass, slice_mass, arguments = _realize(
        plan, matrix=matrix, rhs_direction=rhs_direction
    )
    if not isfinite(mass) or not isfinite(slice_mass) or mass < 0 or slice_mass < 0:
        raise ValueError("QLS branch masses must be finite nonnegative numerical values")
    # Inverse reconstruction has a physical recovery factor. The shortcut
    # branch instead reports a unit direction and explicit unknown physical norm.
    if recovery is not None:
        scale, direction, statistics = physical_vector_statistics(
            selected,
            recovery=recovery,
            observable=observable,
            numerator_frame=kernel.scalar_frames[-1] if observable is not None else "physical",
            need_direction=any(o.frame == "unit" for o in kernel.outputs),
        )
        unavailable_scale = None
    else:
        from nwqlib._numerics import stable_vector_norm, normalized_vector

        norm = stable_vector_norm(selected)
        direction = normalized_vector(selected, norm) if norm > 0 else None
        scale, unavailable_scale = None, UNKNOWN_SCALE
        statistics = (ScalarValue(label="norm_squared", value=None, unavailable=UNKNOWN_SCALE),)
        if observable is not None:
            moment, reason = (None, "unit observable moment is unavailable for zero direction")
            if direction is not None:
                moment, reason = _observable_moment(direction, observable)
            statistics += (
                ScalarValue(
                    label="numerator",
                    frame="unit",
                    value=moment,
                    unavailable=reason,
                ),
            )
    by_name = {v.label: v for v in statistics}
    by_name["algorithm_success_mass"] = ScalarValue(
        label="algorithm_success_mass", frame="encoded_branch", value=mass
    )
    by_name["physical_slice_mass"] = ScalarValue(
        label="physical_slice_mass", frame="encoded_branch", value=slice_mass
    )
    # Materialize only declared outputs. A zero direction or unrepresentable
    # physical vector becomes explicit unavailability, not a substituted state.
    arrays, unavailable = [], []
    for output in kernel.outputs:
        array = recovery.apply_vector(selected) if output.frame == "physical" else direction
        if array is None:
            unavailable.append(
                UnavailableOutput(
                    output=output,
                    reason="selected physical vector is not representable in complex128"
                    if output.frame == "physical"
                    else "unit vector is undefined for zero selected physical slice",
                )
            )
        else:
            arrays.append((output.name, np.asarray(array, dtype=np.complex128)))
    return KernelOutput(
        plan_id=plan.content_id,
        selected_kernel_id=kernel.content_id,
        scalars=tuple(by_name[n] for n in kernel.scalars),
        physical_scale=scale,
        physical_scale_unavailable=unavailable_scale,
        arrays=tuple(arrays),
        unavailable=tuple(unavailable),
        applications=(
            KernelApplication(
                name="polynomial_action", implementation=kernel.implementation, arguments=arguments
            ),
        ),
    )
