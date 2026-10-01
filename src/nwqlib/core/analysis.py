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
    """One producing invocation; loading never creates a new invocation.

    Attributes:
        invocation_id: Identity of this analysis invocation, not an acquisition ID.
        analyzer: Source of the actual reduction implementation.
        method_id: Configured Method identity, or None when no Method is associated.
        environment: Actual available package/interpreter versions at analysis time.
        unavailable_versions: Requested dependency names without available metadata; absence is not version zero.
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
    """One run snapshot; immutable observations, receipts and arrays are shared.

    artifacts is an immutable tuple of existing handles. Taking the snapshot
    never copies an array or expands a native circuit.
    forecast and allocation are the supplied immutable resource provenance;
    snapshotting never reevaluates models or changes the selected workload.

    Attributes:
        observations: Actual immutable observation population from this Run.
        trace: Recorded attempts, status and exposure associated with those observations.
        receipts: Actual prepared artifacts; they are not newly synthesized during snapshotting.
        artifacts: Shared immutable array handles; accessing a handle does not copy its payload.
        controller: Optional saved adaptive frontier and decision history.
        method_context: Method-owned analysis context excluding private mutable computational caches.
        forecast: Optional original PlanEstimate used for this Run.
        allocation: Optional original resource allocation used for this Run.
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
        """Return the stored array handle for a manifest or its identity, without copying data."""
        identity = manifest if isinstance(manifest, str) else manifest.content_id
        for handle in self.artifacts:
            if handle.manifest.content_id == identity:
                return handle
        raise ValueError("the requested array was not stored with this result")


class Result(Record):
    """Method scientific fields and exact acquisition lineage.

    The attached Plan and RunData are shared live inputs, excluded from this
    scalar record. save writes their actual selected data through method hooks.

    Attributes:
        plan_id: Exact selected Plan identity.
        construction_id: Selected construction interpreted by this Result.
        observation_id: Identity of the attached observation population.
        contribution_ids: Actual observation chunks used in this reduction.
        facts: Existing method-produced error or scientific facts with explicit frames.
        origin: Actual analysis provenance, or None when it was not recorded.
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
        if self._plan is None:
            raise ValueError("result has no live Plan; load_result restores the saved selection")
        return self._plan

    @property
    def data(self):
        if self._data is None:
            raise ValueError("result has no acquisition data attached")
        return self._data

    def _attach(self, plan, data):
        """Supported Result hook: bind the original Plan and RunData exactly once.

        Validate acquisition membership and call validate_plan before attaching.
        This does not perform an acquisition or synthesize missing observations.
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
        if self.plan_id != plan.content_id:
            raise ValueError("result differs from its exact selected Plan")

    def _validate_common_plan(self, plan, plan_type):
        """Supported Result hook for exact Plan type, identity and Method lineage.

        Call from validate_plan before checking the Method's own scientific pair.
        """
        if type(plan) is not plan_type:
            raise TypeError(f"{type(self).__name__} requires {plan_type.__name__}")
        if self.plan_id != plan.content_id:
            raise ValueError("result differs from its exact selected Plan")
        if self.origin is not None and self.origin.method_id is not None and self.origin.method_id != plan.method.content_id:
            raise ValueError("analysis origin differs from its configured Method")

    def analyze(self, **settings):
        """Reinterpret the attached RunData with new analysis settings, acquiring nothing.

        The Method's analyze builds a new Result, which is attached to the same
        Plan and RunData after the usual identity and membership checks. The
        original Result is unchanged.
        """
        result = self.plan.method.analyze(self.plan, self.data, settings=settings)
        if not isinstance(result, Result):
            raise TypeError("Method.analyze must return its scientific Result")
        return result._attach(self.plan, self.data)

    def assess(self, **criterion):
        """Assess an explicit accuracy criterion against existing error facts, acquiring nothing.

        A known component bound is not promoted to a total-error certificate.
        """
        from nwqlib.evidence import assess_result
        return assess_result(self, **criterion)

    def verify(self, *, checks):
        """Run the Method's explicitly selected checks, at their own stated cost.

        ``checks`` is one options record, and every built-in options type
        returns ``(receipt, facts)``. ``receipt`` is the VerificationReceipt
        with the raw facts and the numerical calls, and ``facts`` is a tuple
        of FramedFacts that cite that receipt. The types are
        LCHSVerification and LCHSRefinement for LCHS,
        ProjectedVerificationOptions for Lanczos, FixedGCIM and ADAPT,
        EnergyShiftOptions for Lanczos and FixedGCIM,
        AdaptVerificationOptions for ADAPT, NumberSectorOptions and
        QHDVerification for QHD, QLSVerification for QLS and QPEVerification
        for QPE. Except for LCHSRefinement, ``facts`` answers the options'
        ``verification_checks``, and ``Certificate.with_verification``
        attaches it with the same options. LCHSRefinement returns
        output-error components for ``assess`` instead.
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
        """Supported Result display hook returning lines from existing scalar records.

        Do not acquire, reanalyze, materialize arrays, or recompute scientific facts.
        """
        return (f"{type(self).__name__}: scientific data in report",)

    def __str__(self):
        """Summarize the Result from records it already holds, without computing anything.

        The lines are the Method's ``_summary_lines``, then the Method and
        execution route, the first prepared target, how many observation
        chunks this reduction used out of those acquired, the attempt count
        with uncertain attempts and failed host invocations, and the byte-check
        state of a standalone load. The last line always says that accuracy
        is not assessed, because only an explicit ``assess`` call compares
        the Result with an accuracy criterion.
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
        """Project existing scientific metadata and acquisition inventory only.

        Arrays/native caches stay with their owners. This does not validate a
        new scientific pair, reassess accuracy, hydrate data or refresh a Run.
        Observations are projected with their stored fields and each chunk's
        ``content_id``, the identity that ``contribution_ids`` name. The
        identities of the records nested in a chunk, such as its statistics,
        are left out, so the report computes no identity per stored entry. A
        probability chunk shows its array manifests and scalar summaries, and
        no array is loaded: each artifact's ``available`` says whether its
        payload is accessible, held or readable by its store, without reading
        it.
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
        from nwqlib.saved_evidence import save_result
        return save_result(self, path)
