"""An explicitly selected Method case through actual science and saved data."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from nwqlib.algorithms.protocol import Method
from nwqlib.algorithms.registry import direct_method
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Plan
from nwqlib.core.records import Record
from nwqlib.scientist import plan


@dataclass(frozen=True)
class MethodCase:
    """One trusted, bounded scientific case and an independent wrong-pair witness.

    evaluate receives the actual selected Plan and returns its attached Result.
    It explicitly chooses preparation/submission or already supplied RunData.
    No generic reference is performed by this checker. accepts owns the
    independent expected relation. invalid_result must change a scientific field
    while keeping the same concrete Result type and Plan identity; it must be
    intrinsically valid yet violate its method's Plan/result relation.

    The checker writes/reopens one temporary Result archive and tests altered
    result metadata against that same saved selection/data. Method archive hooks
    are explicitly trusted code, with the ordinary archive owner's byte bounds.

    Attributes:
        method: Actual configured Method under examination.
        problem: Concrete admitted scientific input for this bounded case.
        evaluate: Callback accepting the selected Plan and returning its actual attached Result.
        accepts: Independent expected-relation predicate on that Result.
        invalid_result: Callback producing an intrinsically valid wrong scientific pair with unchanged Result type and Plan identity.
        output: Requested output, or None for the problem's default.
        execution: Selected quantum or classical evaluation route.
        shots: Explicit sampled population request, or None for exact/default selection.
        seed: Root seed for this case, or None for the normal random initialization.
    """

    method: Method
    problem: Record
    evaluate: Callable[[Plan], Result]
    accepts: Callable[[Result], bool]
    invalid_result: Callable[[Result], Result]
    output: Record | None = None
    execution: str = "quantum"
    shots: int | None = None
    seed: int | None = None


def _saved_content(result):
    """Compare persistent science, not the identity of reopened native handles.

    Payload meaning remains the explicit case's independent expected relation;
    this comparison does not add a full-array scan to the generic checker.
    """
    data = result.data
    return (result.model_dump(mode="json"), result.plan.to_record(),
            data.observations, data.trace, data.receipts,
            tuple(handle.manifest for handle in data.artifacts),
            data.controller, data.forecast, data.allocation)


def check_method(case: MethodCase) -> dict:
    """Run this explicit case, saved-pair falsifier and legal preservation check.

    The check selects a Plan through the public ``plan``, evaluates it with
    the author's callback, and requires the author's independent relation to
    accept the Result. The Method's error model must equal the Plan's, and
    ``result_type`` must name the concrete Result. The falsifier's Result
    must be a valid record that ``validate_plan`` rejects, both in memory
    and after being written into a saved archive in place of the legal
    Result. The legal Result must then reload unchanged and still pass the
    author's relation.

    The wrong-pair steps test that the Method's own Plan/result validation
    detects a scientifically wrong Result under the same Plan identity,
    which record-level validation alone cannot do. The check adds no
    reference solve or acquisition beyond the author's callback.

    Args:
        case: The explicit MethodCase.

    Returns:
        A mapping with keys ``status`` (``CONFORMANT``), ``method`` (the
        descriptor's method name), ``plan_id``, ``result_id``, ``scope`` and
        ``qualification``. It covers this one case only.

    Raises:
        TypeError: The case, Method or Result type does not meet the protocol.
        ValueError: A relation, falsifier or archive round-trip step fails.
    """
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.saved_evidence import DEFAULT_MAX_BYTES, load_result

    if not isinstance(case, MethodCase):
        raise TypeError("author factory must return an actual MethodCase")
    method = direct_method(case.method)
    selected = plan(
        case.problem,
        method=method,
        output=case.output,
        execution=case.execution,
        shots=case.shots,
        seed=case.seed,
    )
    result = case.evaluate(selected)
    if not isinstance(result, Result) or result.plan is not selected:
        raise ValueError("author case must return the actual selected Plan's attached Result")
    result.validate_plan(selected)
    if case.accepts(result) is not True:
        raise ValueError("author's independent expected relation failed")
    if method.error_model(selected) != selected.error_model:
        raise ValueError("Method error model differs from its selected Plan")
    if getattr(method, "result_type", None) is not type(result):
        raise TypeError("Method.result_type must name its actual concrete Result")
    original = result.content_id, selected.content_id, method.content_id
    invalid = case.invalid_result(result)
    if (
        type(invalid) is not type(result)
        or invalid.plan_id != result.plan_id
        or invalid.content_id == result.content_id
    ):
        raise ValueError(
            "falsifier must alter a scientific field in the same concrete Plan/result pair"
        )
    # Intrinsic Record validity is not the method's scientific pair relation.
    type(invalid).model_validate(invalid.model_dump(mode="json"))
    try:
        invalid.validate_plan(selected)
    except (ValueError, TypeError):
        pass
    else:
        raise ValueError("invalid scientific pair was accepted by validate_plan")
    with TemporaryDirectory(prefix="nwqlib-method-check-") as temporary:
        path = result.save(Path(temporary) / "result")
        loaded = load_result(path, method=type(method))
        if _saved_content(loaded) != _saved_content(result) or case.accepts(loaded) is not True:
            raise ValueError(
                "saved Result lost its original scientific selection or expected relation"
            )
        files = ArchiveFiles(path, DEFAULT_MAX_BYTES)
        saved = files.read_json("result.json")
        changed = {**saved, "result": invalid.model_dump(mode="json")}
        files.write_json("invalid-result.json", changed)
        files.file("invalid-result.json").replace(files.file("result.json"))
        try:
            try:
                load_result(path, method=type(method))
            except (ValueError, TypeError):
                pass
            else:
                raise ValueError("invalid scientific pair was accepted by the Result archive")
        finally:
            files.write_json("original-result.json", saved)
            files.file("original-result.json").replace(files.file("result.json"))
        restored = load_result(path, method=type(method))
        if _saved_content(restored) != _saved_content(result) or case.accepts(restored) is not True:
            raise ValueError("wrong-pair check changed the legal saved result")
    result.validate_plan(selected)
    if (result.content_id, selected.content_id, method.content_id) != original or case.accepts(
        result
    ) is not True:
        raise ValueError("wrong-pair check changed the original legal result")
    return {
        "status": "CONFORMANT",
        "method": method.descriptor.method,
        "plan_id": selected.content_id,
        "result_id": result.content_id,
        "scope": "one explicit author evaluation, independent oracle and same-Plan scientific-pair falsifier",
        "qualification": "not a general scientific or backend qualification",
    }
