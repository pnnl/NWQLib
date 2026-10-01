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
    """One small test case for a new Method: a problem, how to run it, an independent check of the answer and a deliberately wrong answer.

    Build it with keyword arguments, usually in a `case()` function next to the
    Method, and pass it to [`check_method`][nwqlib.algorithms.authoring.check_method]
    or run `python -m nwqlib check-method my_methods:case`. `method`, `problem`,
    `evaluate`, `accepts` and `invalid_result` are required. The checker computes
    no reference answer of its own. `accepts` holds the independent expected
    relation, and `evaluate` decides whether the case prepares and submits
    circuits or analyzes data supplied with it. The Method's archive hooks are
    trusted code and run under the ordinary archive byte limits. The reference
    Hadamard Method, `tests/_hadamard_method.py`, defines such a `case()`.

    Args:
        method: The configured Method under test.
        problem: The problem of this case.
        evaluate: Called with the Plan, returns that Plan's attached Result, for
            example by `prepare`, `submit` and `run.wait()`.
        accepts: Called with the Result, returns `True` when the independent
            expected relation holds, for example `abs(result.value - 1.0) < 1e-12`.
        invalid_result: Called with the Result, returns a Result of the same class
            and Plan with a scientific field changed so that it is valid as a
            record but wrong for this Plan, for example
            `result.revise(value=-1.0)`.
        output: Requested output, or `None` for the problem's default.
        execution: `"quantum"` or `"classical"`.
        shots: Requested shots, or `None` for exact readout or the Method's
            default.
        seed: Seed of the Plan's random streams, or `None` for fresh entropy.
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
    """Check a new Method against its test case: the expected answer, the Plan checks and a saved-result round trip.

    The check plans the case through the public `plan`, evaluates the Plan with
    the case's callback, and requires the case's independent relation to accept
    the Result. The Method's `error_model` must equal the Plan's, and its
    `result_type` must name the Result class. The wrong Result from
    `invalid_result` must be a valid record that the Method's `validate_plan`
    rejects, both in memory and after it replaces the legal Result in a saved
    archive. The legal Result must then reload unchanged and still pass the
    relation. These wrong-answer steps show that the Method's own Plan and Result
    check catches a scientifically wrong answer under the same Plan, which a
    record's own field checks cannot do. Nothing beyond the case's callback is
    computed, neither a reference solve nor extra circuits. `"CONFORMANT"` means this
    one case passed. It does not certify other inputs, physical accuracy,
    backend support or cost. The example at the top of
    [Extending NWQLib](extending.md) runs it on the reference Hadamard Method.

    Args:
        case (MethodCase): The test case.

    Returns:
        report (dict): A dict with `status` (`"CONFORMANT"`), `method` (the descriptor's method
            name), `plan_id`, `result_id`, `scope` and `qualification`.

    Raises:
        TypeError: If the case, the Method or its Result class does not meet the
            Method protocol.
        ValueError: If the expected relation, the wrong-answer check or the saved
            round trip fails.
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
