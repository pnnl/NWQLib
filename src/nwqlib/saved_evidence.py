"""Current selected Result storage. Opening never plans or analyzes again.

A saved Result folder holds ``result.json`` (format, selection, data and result)
and its payload files. Saving checks the associations between Plan, receipts,
attempts, observations and arrays before any file is written. Reporting checks
the format, the schemas and identities of the saved records, the canonical Plan
identity and the associations of the Result with its observations and of the
attempts with their forecast. Loading adds the Method's archive hook with the
Plan identity, the receipts against the selected construction, the Result's
own ``validate_plan`` and the ``validate_data`` hook that an extension Result
may define. Neither repeats the save-time joins
of observations to their receipts and attempts, because saved folders are
treated as read-only (docs/saved_evidence.md, "Saved folders are read-only").
They never acquire, lower, decompose, solve or run a reference, so reading a
Result decodes what was saved
and computes nothing new.

Published arrays load as read-only memory maps. Their headers (shape, dtype,
byte order, size, contiguity) are always checked and a mismatch rejects. That
check reads no payload bytes, and it keeps a mismatched file from being read as
data of the wrong shape.
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

from pathlib import Path
import shutil

from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
from nwqlib.core.analysis import Result, RunData
from nwqlib.execution import ExecutionTrace, ObservationView, PreparedArtifact


# Format label of a saved Result folder, written to result.json. Raise it for
# changes to the envelope or shared stored records. A Method-owned Result
# field change may instead bump that Method's archive format when load_result
# checks it before validating the concrete Result. read_report checks shared
# metadata, leaving Method-owned fields and archive formats uninterpreted.
RESULT_FORMAT = "nwqlib.result/11"


def _validate_recorded_data(plan_id, observations, trace, receipts, manifests):
    """Validate portable acquisition joins without loading a Method or array.

    Every chunk must join to a completed attempt and to its own receipt, with the
    same realization, readout, bindings, layouts and population. This is also
    where a chunk meets its receipt before the Result is saved, so the
    exact-probability unit bound is checked here. Every array a chunk names must
    have its manifest in the saved inventory.

    The chunks are grouped by attempt, in one pass, and each group is frozen
    once. An ordinary readout's attempt holds its one chunk, whose identity
    the completed event names. A trajectory's attempt holds one chunk per
    declared point, placed by the receipt's point index; a duplicate or
    missing point is refused, and the event names the identity of the
    ``ObservationView`` of the ordered collection. Each point chunk declares
    the one-point readout of its point in the receipt's trajectory
    (``ObservationChunk.declares_readout_of``).
    """
    if trace.plan_id != plan_id:
        raise ValueError("saved execution belongs to another Plan")
    events = {event.attempt: event for event in trace.events}
    prepared = {receipt.content_id: receipt for receipt in receipts}
    if len(prepared) != len(receipts):
        raise ValueError("saved preparation receipts must be distinct")
    if any(receipt.plan_id != plan_id for receipt in receipts):
        raise ValueError("saved preparation belongs to another Plan")
    if any(event.prepared_id not in prepared for event in trace.events):
        raise ValueError("saved attempt has no original preparation receipt")
    from nwqlib.execution import ObservationView

    attempts = {}
    for chunk in observations.chunks:
        attempts.setdefault(chunk.attempt, []).append(chunk)
    differs = "saved observation differs from its actual preparation and completed attempt"
    for attempt, chunks in attempts.items():
        event, receipt = events.get(attempt), prepared.get(chunks[0].prepared_id)
        if event is None or receipt is None or event.status != "completed":
            raise ValueError(differs)
        declaration = receipt.observation
        if declaration.kind == "trajectory":
            ordered = [None] * len(declaration.positions)
            for chunk in chunks:
                try:
                    slot = declaration.point_index(chunk.point)
                except ValueError:
                    raise ValueError(differs) from None
                if ordered[slot] is not None:
                    raise ValueError("duplicate point in one saved trajectory acquisition")
                ordered[slot] = chunk
            if any(chunk is None for chunk in ordered):
                raise ValueError("saved trajectory acquisition is missing a declared point")
            identity = ObservationView(chunks=tuple(ordered)).content_id
        elif len(chunks) != 1:
            raise ValueError(differs)
        else:
            identity = chunks[0].content_id
        if event.observation_id != identity:
            raise ValueError(differs)
        for chunk in chunks:
            if (event.prepared_id != chunk.prepared_id
                    or chunk.plan_id != plan_id or chunk.run_id != trace.run_id
                    or event.execution != chunk.execution or event.returned_shots != chunk.returned_shots
                    or chunk.realization_id != receipt.realization_id or not chunk.declares_readout_of(receipt)
                    or chunk.bindings != receipt.realization.bindings or chunk.quantum_layout != receipt.quantum_layout
                    or chunk.classical_layout != receipt.classical_layout
                    or chunk.population != receipt.population):
                raise ValueError(differs)
            trace.validate_observation(chunk)
            chunk.validate_unit_bound(receipt)
    inventory = {manifest.content_id: manifest for manifest in manifests}
    if len(inventory) != len(manifests):
        raise ValueError("saved arrays require distinct manifests")
    if any(manifest.plan_id != plan_id for manifest in manifests):
        raise ValueError("saved array belongs to another Plan")
    for chunk in observations.chunks:
        if any(inventory.get(manifest.content_id) != manifest for manifest in chunk._manifests()):
            raise ValueError("saved observation is missing its requested array")


def _saved_arrays(data):
    """The ``(manifest, read)`` pairs of the arrays a saved Result holds, without reading any.

    They are the RunData's artifact handles, as given, then the arrays of
    probability chunks that the handles do not hold, for example a chunk
    built with ``ObservationChunk.from_histogram``, which keeps its arrays in
    memory. ``read()`` returns the stored array; it is called only when the
    array is written.
    """
    arrays = [(handle.manifest, lambda handle=handle: handle.array) for handle in data.artifacts]
    known = {manifest.content_id for manifest, _ in arrays}
    for chunk in data.observations.chunks:
        if chunk._holds_payload():
            summary = chunk.values[0]
            for manifest in (summary.indices, summary.probabilities):
                if manifest is not None and manifest.content_id not in known:
                    known.add(manifest.content_id)
                    arrays.append((manifest, lambda chunk=chunk, manifest=manifest: dict(
                        (item.content_id, array) for item, array in chunk._payload())[manifest.content_id]))
    return tuple(arrays)


def _validate_data(plan, data):
    """Check RunData against its Plan before a Result is saved.

    Every observation is joined to its receipt, attempt and arrays
    (``_validate_recorded_data``), and every receipt and attempt to the
    selected construction and forecast (``_validate_selection``). Some Methods
    also call this when they analyze or verify.
    """
    _validate_recorded_data(plan.content_id, data.observations, data.trace, data.receipts,
        tuple(manifest for manifest, _ in _saved_arrays(data)))
    _validate_selection(plan, data)


def _validate_selection(plan, data):
    """Join the receipts and attempts of RunData to the selected Plan and its forecast.

    Loading a Result runs this part of ``_validate_data`` only.
    """
    from nwqlib._run_archive import validate_provenance
    validate_provenance(plan, data.forecast, data.allocation)
    receipts = {receipt.content_id: receipt for receipt in data.receipts}
    for event in data.trace.events:
        receipt = receipts[event.prepared_id]
        assessment = None if data.forecast is None else data.forecast.assessment_for(plan, receipt)
        if event.assessment_id != (None if assessment is None else assessment.content_id):
            raise ValueError("saved attempt differs from its original forecast association")
    for receipt in data.receipts:
        receipt.realization.validate_plan(plan)
        if receipt.execution == "quantum_circuit":
            # The held Experiment and construction resolve the readout, so the
            # point is selected once.
            experiment, construction = receipt.realization._selected_construction(plan)
            if experiment.batch is None:
                observation = experiment.observation
            else:
                from nwqlib._prepared_execution import _named_admission_refusal
                from nwqlib.ir.validation import AdmissionStepsExceeded, _Admission
                admission = _Admission(construction.program)
                try:
                    admission.admitted().require_ready()
                    _, observation = experiment._resolved_observation(admission, construction_id=construction.content_id)
                except AdmissionStepsExceeded as error:
                    raise _named_admission_refusal(plan, error, "loading") from error
            if receipt.construction_id != construction.content_id or receipt.observation != observation:
                raise ValueError("saved preparation differs from its selected construction/readout")


def read_report(path):
    """Read current saved metadata without loading Method code or binary data.

    The returned ``metadata_validation`` entry lists what this read checked.
    Common schemas and record identities are checked, and so are the
    associations of the Result with its observations and of the attempts with
    their forecast. A canonical saved Plan description is also checked against
    its content identity (Plan schema version 7). The joins of observations to
    their receipts and attempts were checked when the Result was saved and are
    not repeated. Method-specific scientific fields and binary payload
    integrity require their explicit loader. A metadata report does not claim
    those checks or rerun analysis.
    """
    files = ArchiveFiles(Path(path), None)
    return _validated_metadata(files.read_json("result.json"))


def _validated_metadata(saved, admitted=None):
    """One metadata admission path before Method import or native payload reads.

    It checks the envelope, the record schemas and identities, the canonical
    Plan identity, and the associations of the Result with its observations and
    of the attempts with their forecast. The observation joins of
    ``_validate_recorded_data`` run when a Result is saved, not here. A
    dictionary passed as ``admitted`` receives the validated manifests,
    observations, trace, receipts, forecast and allocation, which
    ``load_data`` then uses without validating them again.
    """
    from pydantic import TypeAdapter
    from nwqlib.artifacts import ArtifactManifest
    from nwqlib.core.planning import Plan
    from nwqlib.core.records import ContentID
    from nwqlib._run_archive import load_provenance

    if type(saved) is not dict:
        raise ValueError("invalid saved Result envelope")
    found = saved.get("format")
    if found != RESULT_FORMAT:
        raise ValueError(f"unsupported saved Result format {found!r}. This NWQLib reads only "
                         f"{RESULT_FORMAT!r}. Open it with the NWQLib release that wrote it, or "
                         "plan and run the problem again.")
    if set(saved) != {"format", "selection", "result", "data"}:
        raise ValueError("invalid saved Result envelope")
    selection, result, data = saved["selection"], saved["result"], saved["data"]
    if (type(selection) is not dict or set(selection) != {"method", "plan_id", "selected"}
            or type(selection["method"]) is not str or not selection["method"]
            or type(selection["selected"]) is not dict or type(result) is not dict or type(data) is not dict):
        raise ValueError("invalid saved Result envelope")
    identity = TypeAdapter(ContentID)
    plan_id = identity.validate_python(selection["plan_id"])
    identity.validate_python(result.get("content_id"))
    # The concrete Result class is deliberately not imported from archive text.
    common = Result.model_validate({name: result[name] for name in Result.model_fields
        if name != "schema_version" and name in result})
    if common.plan_id != plan_id:
        raise ValueError("saved Result belongs to another Plan")
    descriptions = selection["selected"].get("plan")
    verified_plan = False
    if descriptions is not None:
        if type(descriptions) is not dict or set(descriptions) != set(Plan.model_fields):
            raise ValueError("invalid canonical saved Plan metadata")
        for name, field in Plan.model_fields.items():
            if name not in {"problem", "method", "output", "reconstruction"}:
                TypeAdapter(field.rebuild_annotation()).validate_python(descriptions[name])
        for name in ("problem", "method", "output"):
            declaration = descriptions[name]
            if (type(declaration) is not dict or set(declaration) != {"type", "fields"}
                    or type(declaration["type"]) is not str or type(declaration["fields"]) is not dict):
                raise ValueError("invalid saved scientific description")
        if descriptions["method"]["type"] != selection["method"]:
            raise ValueError("saved Method differs from its canonical Plan description")
        if Plan.record_identity(descriptions) != plan_id:
            raise ValueError("saved Plan content_id does not match its metadata")
        verified_plan = True
    if (set(data) != {"observations", "trace", "receipts", "controller", "forecast", "allocation", "method_context", "artifacts"}
            or type(data["receipts"]) is not list or type(data["artifacts"]) is not list):
        raise ValueError("invalid saved RunData metadata")
    manifests = []
    for item in data["artifacts"]:
        if (type(item) is not dict or set(item) != {"manifest", "array"}
                or type(item["manifest"]) is not dict or type(item["array"]) is not str
                or Path(item["array"]).name != item["array"] or item["array"] in ("", ".", "..")):
            raise ValueError("invalid saved array metadata")
        manifests.append(ArtifactManifest.model_validate(item["manifest"]))
    observations = ObservationView.model_validate(data["observations"])
    trace = ExecutionTrace.model_validate(data["trace"])
    receipts = tuple(PreparedArtifact.model_validate(row) for row in data["receipts"])
    if (common.observation_id != observations.content_id
            or not set(common.contribution_ids) <= {c.content_id for c in observations.chunks}):
        raise ValueError("saved Result differs from its actual collected observations")
    forecast, allocation = load_provenance(data)
    if forecast is not None:
        if forecast.plan_id != plan_id or forecast.allocation != allocation:
            raise ValueError("saved forecast belongs to another Plan or allocation")
    by_point = {} if forecast is None else {a.point.realization_id: a for a in forecast.assessments}
    prepared = {r.content_id: r for r in receipts}
    for event in trace.events:
        receipt = prepared[event.prepared_id]
        assessment = by_point.get(receipt.realization_id)
        if event.assessment_id != (None if assessment is None else assessment.content_id):
            raise ValueError("saved attempt differs from its original forecast association")
    if admitted is not None:
        admitted.update(manifests=tuple(manifests), observations=observations, trace=trace, receipts=receipts,
                        forecast=forecast, allocation=allocation)
    return {**saved, "metadata_validation": {
        "common_schemas_and_identities": "checked",
        "canonical_plan_identity": "checked" if verified_plan else "no canonical Plan description supplied",
        "result_observation_and_forecast_associations": "checked",
        "observation_receipt_and_attempt_joins": "not checked here because save_result checks them before writing",
        "method_specific_result_schema_identity_and_scientific_relation": "not checked; explicit Method loading is required",
        "binary_payload_integrity": "not checked; no arrays or native circuits were read",
    }}


def save_data(data, files, method):
    """Write RunData as JSON records plus one NPY file per distinct published array.

    The arrays are the RunData's artifacts and the arrays of its probability
    chunks (``_saved_arrays``). Method context is written only through the
    Method's own hook. No reference, analysis or missing cache is computed for
    storage.
    """
    return dict(observations=data.observations.model_dump(mode="json"),
        trace=data.trace.model_dump(mode="json"),
        receipts=[receipt.model_dump(mode="json") for receipt in data.receipts],
        controller=data.controller,
        forecast=None if data.forecast is None else data.forecast.model_dump(mode="json"),
        allocation=None if data.allocation is None else data.allocation.model_dump(mode="json"),
        method_context=None if data.method_context is None else method.save_run_context(data.method_context, files),
        artifacts=[dict(manifest=manifest.model_dump(mode="json"),
                        array=files.write_array(f"result-array-{index}.npy", array()))
                   for index, (manifest, array) in enumerate(_saved_arrays(data))])


def load_data(data, files, method, *, admitted=None):
    """Restore RunData with memory-mapped arrays.

    Each array's shape, byte count, contiguity and little-endian dtype
    (``<c16``, ``<f8`` or ``<u8``, the one its declared output
    selects) must match its manifest. That check reads only the NPY header, and a mismatched file would
    otherwise be read as data of the wrong shape. The array values are not read.
    A probability chunk reads its arrays through this store when its
    histogram is first requested. ``admitted`` holds the records that
    ``_validated_metadata`` already validated from this ``data`` (manifests,
    observations, trace, receipts, forecast and allocation); they are used
    as they are instead of being validated and hashed again.
    """
    from nwqlib.artifacts import ArtifactManifest, ArtifactStore
    from nwqlib._run_archive import load_provenance
    if admitted is None:
        forecast, allocation = load_provenance(data)
        manifests = tuple(ArtifactManifest.model_validate(item["manifest"]) for item in data["artifacts"])
        observations = ObservationView.model_validate(data["observations"])
        trace = ExecutionTrace.model_validate(data["trace"])
        receipts = tuple(PreparedArtifact.model_validate(row) for row in data["receipts"])
    else:
        forecast, allocation = admitted["forecast"], admitted["allocation"]
        manifests, observations = admitted["manifests"], admitted["observations"]
        trace, receipts = admitted["trace"], admitted["receipts"]
    store = ArtifactStore(max_bytes=DEFAULT_MAX_BYTES)
    store._restore(manifests)
    for manifest, item in zip(manifests, data["artifacts"], strict=True):
        array = files.read_array(item["array"])
        if (array.shape != manifest.output.shape or array.dtype.str != manifest.output.storage_dtype
                or array.nbytes != manifest.data_bytes or not array.flags.c_contiguous):
            raise ValueError("saved array differs from its declared shape and encoding")
        object.__setattr__(store._handles[manifest.content_id], "_array", array)
    context = data["method_context"]
    if context is not None:
        context = method.snapshot_result_context(method.load_run_context(context, files))
    for chunk in observations.chunks:
        chunk._attach_store(store)
    return RunData(observations, trace, receipts, store.snapshot(), data["controller"], context, forecast,
                   allocation)


def save_result(result, path):
    """Save the selected science and sufficient data to one new directory.

    The associations are validated before any file is written. The directory
    must not exist, and it is removed if saving fails, so a partial folder is
    never left that could look like a complete archive.
    """
    if not isinstance(result, Result):
        raise TypeError("save_result requires a scientific Result")
    _validate_data(result.plan, result.data)
    validate = getattr(result, "validate_data", None)
    if callable(validate):
        validate(result.data)
    result_type = getattr(result.plan.method, "result_type", None)
    if result_type is not type(result):
        raise TypeError("Method.result_type must name this concrete Result owner")
    files = ArchiveFiles(path, DEFAULT_MAX_BYTES)
    files.path.mkdir()
    try:
        files.write_json("result.json", dict(format=RESULT_FORMAT,
            selection=save_plan(result.plan, files), result=result.model_dump(mode="json"),
            data=save_data(result.data, files, result.plan.method)))
    except BaseException:
        shutil.rmtree(files.path)
        raise
    return files.path


def load_result(path, *, method=None):
    """Restore a saved Result with its Plan and RunData, keeping arrays as lazy mmaps.

    The metadata admission (``_validated_metadata``), the Plan identity, the
    receipts against the selected construction (``_validate_selection``), the
    attached observations and the ``validate_data`` hook that an extension
    Result may define are checked.
    The save-time observation joins are not repeated, and array values are not
    read (see the module docstring).
    """
    files = ArchiveFiles(Path(path), None)
    saved = files.read_json("result.json")
    admitted = {}
    _validated_metadata(saved, admitted)
    plan = load_plan(saved["selection"], files, method=method)
    result_type = getattr(plan.method, "result_type", None)
    if not isinstance(result_type, type) or not issubclass(result_type, Result):
        raise TypeError("Method.result_type must name its concrete Result owner")
    result = result_type.model_validate(saved["result"])
    data = load_data(saved["data"], files, plan.method, admitted=admitted)
    _validate_selection(plan, data)
    result = result._attach(plan, data)
    validate = getattr(result, "validate_data", None)
    if callable(validate):
        validate(data)
    return result
