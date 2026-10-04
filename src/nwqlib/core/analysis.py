"""Scientific Result values with immutable acquisition data shared by reference."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import platform
from uuid import uuid4
from typing import Literal

from pydantic import PrivateAttr, TypeAdapter, model_validator

from nwqlib.core.records import ContentID, Record, Source, Text
from nwqlib.evidence.error_model import FramedFact


class AnalysisOrigin(Record):
    """The analysis call that computed a Result, and its software versions.

    A Result records it in `origin`. Loading a saved Result keeps the
    original record and does not create a new one. The fields below are
    read-only.

    Attributes:
        invocation_id: Unique identifier of this analysis call. It does not
            identify a measurement.
        analyzer: Name, version and reference of the code that computed the
            Result (`Source`).
        method_id: Content hash of the configured Method, or `None` when no
            Method is involved.
        environment: The Python and package versions (`Source` records)
            found when the analysis ran.
        unavailable_versions: Names of requested packages whose version
            could not be read. A missing version is not version zero.
    """

    invocation_id: Text
    analyzer: Source
    method_id: ContentID | None = None
    environment: tuple[Source, ...] = ()
    unavailable_versions: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def _environment(self):
        names = tuple(source.name for source in self.environment)
        if (len(set(names)) != len(names) or len(set(self.unavailable_versions)) != len(self.unavailable_versions)
                or set(names) & set(self.unavailable_versions)):
            raise ValueError("analysis versions require distinct known or unavailable names")
        return self


def capture_analysis_origin(*, analyzer, method_id=None, dependencies=(), max_dependencies=64):
    """Read metadata at the producer after admission of all requested packages.

    max_dependencies bounds the complete supplied name population, including
    duplicates, before deduplication or any package/version metadata lookup.
    The fixed nwqlib/pydantic entries are additional to that caller population.
    """
    if not isinstance(analyzer, Source):
        raise TypeError("analysis analyzer must be a Source")
    method_id = TypeAdapter(ContentID | None).validate_python(method_id)
    if type(max_dependencies) is not int or max_dependencies < 0:
        raise ValueError("analysis max_dependencies must be a nonnegative integer")
    if type(dependencies) not in (tuple, list):
        raise TypeError("analysis dependencies must be a finite list or tuple of package names")
    if len(dependencies) > max_dependencies:
        raise ValueError("analysis dependency population exceeds max_dependencies")
    dependencies = TypeAdapter(tuple[Text, ...]).validate_python(dependencies)
    packages = tuple(dict.fromkeys(("nwqlib", "pydantic", *dependencies)))
    sources = [Source(name="python", version=platform.python_version(),
        domain="analysis process software", reference="platform.python_version")]
    unavailable = []
    for name in packages:
        try:
            selected = version(name)
        except PackageNotFoundError:
            unavailable.append(name)
        else:
            sources.append(Source(name=name, version=selected, domain="analysis process software",
                                  reference="importlib.metadata.version"))
    return AnalysisOrigin(invocation_id=str(uuid4()), analyzer=analyzer, method_id=method_id,
                          environment=tuple(sources), unavailable_versions=tuple(unavailable))


@dataclass(frozen=True)
class RunData:
    """The measured data of a Run, as attached to its Result.

    `result.data` and `run.data` return it. It shares the Run's immutable
    observations, preparation records and arrays, so taking it copies no
    array and builds no circuit, and it reevaluates no model. The fields
    below are read-only.

    Attributes:
        observations: Every observation the Run collected, as an
            `ObservationView`. Its `chunks` hold the statistics of each
            measurement.
        trace: The [`ExecutionTrace`][nwqlib.execution.ExecutionTrace]: every
            attempt, its status and the work counted against the limits,
            failed and uncertain attempts included.
        receipts: The preparation records (`PreparedArtifact`) of the
            circuits and of the host computations (classical computations
            that the Method runs on this machine) that ran. Taking the snapshot
            builds none of them again.
        artifacts: Handles of the saved arrays
            ([`ArtifactHandle`][nwqlib.artifacts.ArtifactHandle]). Reading a
            handle copies no data.
        controller: The saved iteration state and decision history of an
            adaptive Method, as JSON text, or `None`.
        method_context: Analysis data that the Method keeps with the Result,
            without its private working caches, or `None`.
        forecast: The `PlanEstimate` given to the Run, or `None`.
        allocation: The `Allocation` given to the Run, or `None`.
    """

    observations: object
    trace: object
    receipts: tuple = ()
    artifacts: tuple = ()
    controller: object = None
    method_context: object = None
    forecast: object = None
    allocation: object = None

    def artifact(self, manifest):
        """Return the handle of one saved array, without copying its data.

        Args:
            manifest (ArtifactManifest | str): The array's manifest, or its
                content hash.

        Returns:
            handle (ArtifactHandle): The handle. Read the values with
                `handle.array`.

        Raises:
            ValueError: If this Run's data holds no such array.
        """
        identity = manifest if isinstance(manifest, str) else manifest.content_id
        for handle in self.artifacts:
            if handle.manifest.content_id == identity:
                return handle
        raise ValueError("the requested array was not stored with this result")


class Result(Record):
    """The answer of a Method, with the Plan and measured data it came from.

    [`solve`][nwqlib.scientist.solve] and `Run.wait` return a Result, and
    [`load_result`][nwqlib.scientist.load_result] reopens a saved one. Each
    Method returns its own Result type, and its fields hold the answer:

    | Method | Result type | Answer | Unit |
    | --- | --- | --- | --- |
    | [`ExpectationMethod`][nwqlib.algorithms.expectation.ExpectationMethod] | [`ExpectationAnalysis`][nwqlib.algorithms.expectation.ExpectationAnalysis] | `value` | The Problem's `unit` for the default `NormalizedExpectation` |
    | [`Lanczos`][nwqlib.algorithms.lanczos.method.Lanczos] | [`LanczosResult`][nwqlib.algorithms.lanczos.records.LanczosResult] | `eigenvalue` | The Problem's `unit` |
    | [`QCELS`][nwqlib.algorithms.qpe.method.QCELS], [`SPE`][nwqlib.algorithms.qpe.method.SPE], [`RFE`][nwqlib.algorithms.qpe.method.RFE], [`RWPE`][nwqlib.algorithms.qpe.method.RWPE] | [`QPEAnalysis`][nwqlib.algorithms.qpe.records.QPEAnalysis] | `eigenvalue` for a Hamiltonian input, `phase` | The Problem's `unit` for `eigenvalue`, turns in [0, 1) for `phase` |
    | [`FixedGCIM`][nwqlib.algorithms.gcim.fixed_basis.FixedGCIM] | [`FixedGCIMResult`][nwqlib.algorithms.gcim.fixed_basis.FixedGCIMResult] | `eigenvalue` | The Problem's `unit` |
    | [`ADAPT`][nwqlib.algorithms.gcim.adapt.ADAPT] | [`ADAPTResult`][nwqlib.algorithms.gcim.adapt_records.ADAPTResult] | `eigenvalue` | The Problem's `unit` |
    | [`LCHS`][nwqlib.algorithms.lchs.method.LCHS] | [`LCHSAnalysis`][nwqlib.algorithms.lchs.primary_records.LCHSAnalysis] | `solution` for the default `Solution`, `value` for a scalar output | The Problem's `unit` for `solution` |
    | [`QLS`][nwqlib.algorithms.qls.method.QLS] | [`QLSAnalysis`][nwqlib.algorithms.qls.primary_records.QLSAnalysis] | `x` for the default `Solution`, `value` for any output | The Problem's `unit` for `x` |
    | [`QHD`][nwqlib.algorithms.qhd.method.QHD] | [`QHDAnalysis`][nwqlib.algorithms.qhd.records.QHDAnalysis] | `candidate`, the observed grid point with the smallest objective, and `value`, the objective there | Coordinates of the variables for `candidate`, the Problem's `unit` for `value` |

    For another requested output, the Result type names the field that holds
    it, and [Choose a problem and output](../problems.md) gives its unit.
    Each Result type states when its answer is unavailable. `print(result)` shows
    the answer with its main conditions. `analyze` recomputes the Result
    from the same data with other settings, `assess` compares it with an
    accuracy criterion, `verify` runs a named check, `report` returns its
    stored values as a dictionary and `save` writes it to a folder. The
    Plan (`result.plan`) and run data (`result.data`) are attached to the
    Result rather than stored in its fields, and `save` writes them with it.

    The fields below identify where the answer came from, and they are
    read-only. A Method author's Result type sets them, as the [Run your own
    circuit](../own_circuit.md) guide shows.

    Attributes:
        plan_id: Content hash of the Plan the Result was computed from.
        construction_id: Content hash of the construction of that Plan.
        observation_id: Content hash of the observations attached to the
            Result.
        contribution_ids: Content hashes of the parts of the observations
            that this Result uses.
        facts: Error evidence that the Method attached (`FramedFact`
            records), each stated for its quantity, unit and scope. `assess`
            reads it.
        origin: The [`AnalysisOrigin`][nwqlib.core.analysis.AnalysisOrigin]
            of the analysis, or `None` when it was not recorded.
    """

    schema_version: Literal[2] = 2
    plan_id: ContentID
    construction_id: ContentID
    observation_id: ContentID
    contribution_ids: tuple[ContentID, ...] = ()
    facts: tuple[FramedFact, ...] = ()
    origin: AnalysisOrigin | None = None
    _plan: object = PrivateAttr(default=None)
    _data: object = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _claim_lineage(self):
        if len({fact.fact.quantity for fact in self.facts}) != len(self.facts):
            raise ValueError("result error facts require distinct source names")
        return self

    @property
    def plan(self):
        """The [`Plan`][nwqlib.core.planning.Plan] this Result was computed from.

        Raises:
            ValueError: If no Plan is attached, as for a Result built from
                its fields alone. `load_result` attaches the saved Plan.
        """
        if self._plan is None:
            raise ValueError("result has no live Plan; load_result restores the saved selection")
        return self._plan

    @property
    def data(self):
        """The [`RunData`][nwqlib.core.analysis.RunData] this Result was computed from.

        It holds the observations, preparation records, attempt history and
        saved arrays of the Run.

        Raises:
            ValueError: If no run data is attached.
        """
        if self._data is None:
            raise ValueError("result has no acquisition data attached")
        return self._data

    def _attach(self, plan, data):
        """Attach the Plan and run data to a Result once, and return the Result.

        A supported hook for Method authors ([Add a
        Method](../algorithm_protocol.md#supported-protected-extension-hooks)).
        It checks that the Result names this Plan and these observations,
        that every part of the observations it uses is in the data, and calls
        `validate_plan` before attaching. It measures nothing and creates no
        missing observation.

        Args:
            plan (Plan): The Plan the Result was computed from.
            data (RunData): The run data the Result was computed from.

        Returns:
            result (Result): This Result, with `plan` and `data` attached.

        Raises:
            ValueError: If the Result names another Plan or other
                observations, uses an observation absent from the data, or
                is already attached to another Plan or data.
        """
        if not isinstance(data, RunData) or self.plan_id != plan.content_id:
            raise ValueError("result requires its original Plan and RunData")
        if data.trace.plan_id != plan.content_id or self.observation_id != data.observations.content_id:
            raise ValueError("result differs from its actual run observations")
        available = {chunk.content_id for chunk in data.observations.chunks}
        if not set(self.contribution_ids) <= available:
            raise ValueError("result names an acquisition absent from its run data")
        if self._plan is not None and (self._plan is not plan or self._data is not data):
            raise ValueError("a result cannot be rebound to different data")
        self.validate_plan(plan)
        self._plan, self._data = plan, data
        return self

    def validate_plan(self, plan):
        """Check that this Result belongs to `plan`, before the Plan is attached.

        The base check compares `plan_id` with the Plan's content hash. A
        Method's Result type overrides it to call `_validate_common_plan`
        and then check its own relation between the Plan and its fields
        ([Add a Method](../algorithm_protocol.md#supported-protected-extension-hooks)).

        Args:
            plan (Plan): The Plan to check against.

        Raises:
            ValueError: If the Result names another Plan.
        """
        if self.plan_id != plan.content_id:
            raise ValueError("result differs from its exact selected Plan")

    def _validate_common_plan(self, plan, plan_type):
        """Check the Plan type, the Plan's content hash and the Method that analyzed the Result.

        A supported hook for Method authors. Call it from `validate_plan`
        before checking the Method's own relation between the Plan and the
        Result ([Add a
        Method](../algorithm_protocol.md#supported-protected-extension-hooks)).

        Args:
            plan (Plan): The Plan to check against.
            plan_type (type): The exact Plan class that the Method produces.

        Raises:
            TypeError: If `plan` is not exactly of type `plan_type`.
            ValueError: If the Result names another Plan, or its `origin`
                names another Method than the Plan's.
        """
        if type(plan) is not plan_type:
            raise TypeError(f"{type(self).__name__} requires {plan_type.__name__}")
        if self.plan_id != plan.content_id:
            raise ValueError("result differs from its exact selected Plan")
        if self.origin is not None and self.origin.method_id is not None and self.origin.method_id != plan.method.content_id:
            raise ValueError("analysis origin differs from its configured Method")

    def analyze(self, **settings):
        """Recompute the Result from the same measured data with other analysis settings.

        The Method analyzes the attached run data again and returns a new
        Result, attached to the same Plan and data. Nothing is measured, and
        this Result is unchanged. A loaded Result can be analyzed too. The
        settings a Method accepts are listed in its guide, for example the
        `overlap_*` settings of Lanczos.

        Args:
            **settings (object): Analysis settings of the Method. Omitted, the
                analysis uses the original settings.

        Returns:
            result (Result): The new Result.

        Raises:
            ValueError: If no Plan or run data is attached.
        """
        result = self.plan.method.analyze(self.plan, self.data, settings=settings)
        if not isinstance(result, Result):
            raise TypeError("Method.analyze must return its scientific Result")
        return result._attach(self.plan, self.data)

    def assess(self, **criterion):
        """Check whether the Result's error evidence meets an accuracy criterion.

        `assess` combines the error bounds that the Method attached in
        `facts`, and any bounds supplied here, and compares them with the
        criterion. It measures nothing, reruns nothing and leaves the Result
        unchanged, so a stricter criterion later is a new assessment of the
        same data. A known bound on one error component is never counted as a
        bound on the total error. The [Check accuracy and verify a
        result](../verification.md) guide explains the criteria.

        Args:
            **criterion (object): The criterion, given either as
                `accuracy=Accuracy(...)` or by the fields of
                [`Accuracy`][nwqlib.problems.records.Accuracy]: exactly one
                of `absolute_tolerance` and `relative_tolerance`,
                `confidence` (default `0.95`) and `component` (`"total"`,
                the default, or `"sampling"`). The optional keyword `facts`
                gives error evidence (`FramedFact` records) that replaces the
                Result's evidence of the same name, `reference` gives a
                `TargetReference` that supplies the target's scale for a
                relative tolerance, `absolute_fallback` gives a positive
                absolute tolerance used when that scale is unavailable or
                zero, and `max_integer_bits`, default `4096`, limits the
                integers of the exact arithmetic.

        Returns:
            assessment (ClaimAssessment): The
                [`ClaimAssessment`][nwqlib.evidence.ClaimAssessment]. Its
                `status` is `"PASS"` when every required error source has a
                supported bound and together they meet the tolerance at the
                requested confidence, `"INCONCLUSIVE"` when they do not show
                it, for example because a bound exceeds the tolerance or a
                source is unknown, and `"NOT_APPLICABLE"` for
                `component="sampling"` when the Method's error model has no
                sampling source.

        Raises:
            ValueError: If `accuracy` is given together with the individual
                fields, or the Method has no error model.

        Examples:
            An exact classical solve has no sampling error, while its total
            error stays unproven:

            >>> from nwqlib import Eigenproblem, solve
            >>> from nwqlib.algorithms import FixedGCIM
            >>> problem = Eigenproblem(A=[[1.0, 0.0], [0.0, -1.0]])
            >>> method = FixedGCIM(basis=([1.0, 0.0], [0.0, 1.0]))
            >>> result = solve(problem, method=method, execution="classical")
            >>> print(result.eigenvalue)
            -1.0
            >>> print(result.assess(absolute_tolerance=0.01,
            ...                     component="sampling").status)
            PASS
            >>> print(result.assess(absolute_tolerance=0.01).status)
            INCONCLUSIVE
        """
        from nwqlib.evidence import assess_result
        return assess_result(self, **criterion)

    def verify(self, *, checks):
        """Run a named check of the Result, such as a residual or a comparison with a reference.

        The check is described by one options record of a type that the
        Method supports. It computes what the record names, at the cost the
        record states, and only when called. Every built-in options type
        returns `(receipt, facts)`. `receipt` records the check, its raw
        values and the numerical calls it made, and `facts` is a tuple of
        `FramedFact` records that cite `receipt`. The [Check accuracy and
        verify a result](../verification.md) guide describes each check.

        | Method | Options types |
        | --- | --- |
        | LCHS | [`LCHSVerification`][nwqlib.algorithms.lchs.verification.LCHSVerification], [`LCHSRefinement`][nwqlib.algorithms.lchs.refinement.LCHSRefinement] |
        | Lanczos, FixedGCIM | [`ProjectedVerificationOptions`][nwqlib.evidence.verification.ProjectedVerificationOptions], [`EnergyShiftOptions`][nwqlib.evidence.energy_shift.EnergyShiftOptions] |
        | ADAPT | [`ProjectedVerificationOptions`][nwqlib.evidence.verification.ProjectedVerificationOptions], [`AdaptVerificationOptions`][nwqlib.algorithms.gcim.adapt_verification.AdaptVerificationOptions] |
        | QHD | [`NumberSectorOptions`][nwqlib.evidence.sector.NumberSectorOptions], [`QHDVerification`][nwqlib.algorithms.qhd.records.QHDVerification] |
        | QLS | [`QLSVerification`][nwqlib.algorithms.qls.verification.QLSVerification] |
        | QCELS, SPE, RFE, RWPE | [`QPEVerification`][nwqlib.algorithms.qpe.records.QPEVerification] |

        Except for `LCHSRefinement`, `facts` answers the checks that the
        options list in `verification_checks`, and
        [`Certificate.with_verification`][nwqlib.evidence.Certificate.with_verification]
        attaches it with the same options. `LCHSRefinement` returns error
        components of the output for `assess` instead.

        Args:
            checks (object): One options record from the table.

        Returns:
            verification (tuple): `(receipt, facts)` as described above.

        Raises:
            ValueError: If no Plan or run data is attached. The Method raises
                its own error for an options type it does not support.

        Examples:
            The Gram matrix of an orthonormal trial basis is the identity, so
            its deficit from positive semidefiniteness is zero:

            >>> from nwqlib import Eigenproblem, solve
            >>> from nwqlib.algorithms import FixedGCIM
            >>> from nwqlib.evidence.verification import (
            ...     ProjectedVerificationOptions)
            >>> problem = Eigenproblem(A=[[1.0, 0.0], [0.0, -1.0]])
            >>> method = FixedGCIM(basis=([1.0, 0.0], [0.0, 1.0]))
            >>> result = solve(problem, method=method, execution="classical")
            >>> options = ProjectedVerificationOptions(
            ...     name="projected", comparisons=("gram_psd_deficit",),
            ...     tolerance=1e-10)
            >>> receipt, facts = result.verify(checks=options)
            >>> print(facts[0].fact.quantity, facts[0].fact.value.numerator)
            projected.gram_psd_deficit 0
        """
        return self.plan.method.verify(self.plan, self, checks=checks)

    @staticmethod
    def _scalar_text(value):
        return "unavailable" if value is None else format(value, ".8g")

    @classmethod
    def _values_text(cls, values):
        if values is None:
            return "unavailable"
        if len(values) > 8:
            return f"{len(values)} values (see report)"
        return "[" + ", ".join(cls._scalar_text(value) for value in values) + "]"

    def _array_text(self, manifest):
        """At most eight resident elements; shape alone for larger, not yet loaded or missing data."""
        shape = manifest.output.shape
        text = f"shape={shape}, {manifest.output.frame}, phase={manifest.output.global_phase}"
        if unit := self._unit_text():
            text += "; unit=" + unit.strip()
        if self._data is None:
            return text + "; data not attached"
        handle = next((item for item in self._data.artifacts if item.manifest == manifest), None)
        if handle is None or not handle.available:
            return text + "; array unavailable"
        if handle._array is None:
            return text + "; stored array, not loaded"
        return self._resident_array_text(handle._array, text)

    def _resident_array_text(self, array, description):
        """Format an existing array without copying, scanning or creating one."""
        if array.size <= 8:
            return "[" + ", ".join(self._scalar_text(value) for value in array.flat) + "]; " + description
        return description + "; stored array"

    def _array_acquisition_text(self, manifest):
        if self._data is not None:
            for chunk in self._data.observations.chunks:
                if manifest.acquisition == (chunk.run_id, chunk.attempt, chunk.job, chunk.chunk):
                    return "Array acquisition: " + chunk.execution.replace("_", " ")
        return None

    def _unit_text(self):
        if self._plan is None:
            return ""
        from nwqlib.problems.records import UNSPECIFIED_UNIT

        model = self._plan.error_model
        unit = self._plan.output.unit if model is None else model.frame.unit
        return "" if unit is None or unit.same_unit(UNSPECIFIED_UNIT) else " " + unit.symbol

    def _summary_lines(self):
        """Return the first lines of `print(result)`, built from the Result's stored values.

        A supported hook for Method authors ([Add a
        Method](../algorithm_protocol.md#supported-protected-extension-hooks)).
        An override must not measure, reanalyze, load arrays or recompute
        error evidence.

        Returns:
            lines (tuple[str, ...]): The lines.
        """
        return (f"{type(self).__name__}: scientific data in report",)

    def __str__(self):
        """Summarize the Result from records it already holds, without computing anything.

        The lines are the Method's ``_summary_lines``, then the Method and
        execution route, then the first prepared target, how many observation
        chunks this reduction used out of those acquired, and the attempt
        count with uncertain attempts and failed host invocations. Without an
        attached Plan or run data, the corresponding line says so. The last
        line always says ``accuracy not assessed``, because a Result stores no
        assessment and only an explicit ``assess`` call compares it with an
        accuracy criterion.
        """
        lines = list(self._summary_lines())
        if self._plan is None:
            lines.append("Method and execution data not attached")
        else:
            lines.append(f"Method: {self._plan.method.descriptor.method}; execution: {self._plan.execution}")
        if self._data is None:
            lines.append("Acquisition data unavailable")
        else:
            data = self._data
            target = (f"first target: {data.receipts[0].target.name}" if data.receipts
                      else "no prepared target recorded")
            uncertain = sum(event.status == "uncertain" for event in data.trace.events)
            failed = sum(event.status == "failed" for event in data.trace.events)
            lines.append(f"{target}; data: {len(self.contribution_ids)}/{len(data.observations.chunks)} chunks used; "
                         f"{len(data.trace.events)} attempts" + (f", {uncertain} uncertain" if uncertain else "")
                         + (f", {failed} failed host invocations" if failed else ""))
        lines.append("accuracy not assessed")
        return "\n".join(lines)

    def __repr__(self):
        return str(self)

    def _repr_pretty_(self, printer, cycle):
        printer.text(f"{type(self).__name__}(...)" if cycle else str(self))

    def report(self):
        """Return the Result's stored values, Plan and run records as a dictionary.

        The dictionary holds the printed summary and the JSON forms of the
        Plan, the Result, the attempt history, the observations, the
        preparation records, the manifests of the saved arrays, the forecast,
        the Allocation and the iteration state of an adaptive Method. It
        reads stored records only. It loads no array, runs no check,
        reassesses no accuracy, reanalyzes nothing and does not refresh a
        Run. Each array manifest
        comes with `available`, which says whether the array can be read,
        without reading it. A probability observation shows its array
        manifests and summary values. Each observation is listed with its
        `content_id`, the content hash that `contribution_ids` name. The
        records nested in an observation, such as its statistics, are listed
        without their own content hashes, so the report computes none per
        stored entry. The dictionary describes the Result, and it cannot be
        loaded back. Save with `save`, and read a saved folder without
        loading Method code with `nwqlib.saved_evidence.read_report`.

        Returns:
            report (dict): Keys `summary`, `plan`, `result`, `trace`,
                `observations`, `receipts`, `artifacts`, `forecast`,
                `allocation` and `controller`. Values that need run data are
                `None` when none is attached.

        Examples:
            >>> from nwqlib import Eigenproblem, solve
            >>> from nwqlib.algorithms import FixedGCIM
            >>> problem = Eigenproblem(A=[[1.0, 0.0], [0.0, -1.0]])
            >>> method = FixedGCIM(basis=([1.0, 0.0], [0.0, 1.0]))
            >>> result = solve(problem, method=method, execution="classical")
            >>> report = result.report()
            >>> print(report["summary"].splitlines()[0])
            Ritz eigenvalue: -1
            >>> print(report["result"]["eigenvalue"])
            -1.0
        """
        data = self._data

        def dump(record):
            return None if record is None else record.model_dump(mode="json")

        def observations(view):
            return dict(chunks=[dict(chunk.model_dump(mode="json", exclude_computed_fields=True),
                                     content_id=chunk.content_id) for chunk in view.chunks])

        return dict(summary=str(self), plan=None if self._plan is None else self._plan.to_record(),
            result=self.model_dump(mode="json"), trace=None if data is None else dump(data.trace),
            observations=None if data is None else observations(data.observations),
            receipts=None if data is None else [dump(receipt) for receipt in data.receipts],
            artifacts=None if data is None else [dict(manifest=dump(handle.manifest), available=handle.available)
                for handle in data.artifacts],
            forecast=None if data is None else dump(data.forecast),
            allocation=None if data is None else dump(data.allocation),
            controller=None if data is None else data.controller)

    def save(self, path):
        """Save this Result with its Plan and run data to a new directory.

        `load_result(path)` reopens the saved Result, and `Result.analyze`
        can then recompute it with other settings without new measurements.
        Saving checks that the Result, Plan and data belong together before
        it writes any file. If saving fails, the new directory is removed.
        The saved files may total at most 10 GB (decimal).

        Args:
            path (str | os.PathLike): Directory to create. Its parent must
                exist and the directory itself must not.

        Returns:
            path (pathlib.Path): The created directory.

        Raises:
            FileExistsError: If `path` already exists.
        """
        from nwqlib.saved_evidence import save_result
        return save_result(self, path)
