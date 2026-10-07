"""Host kernels bound by a Method's factory.

A portable Source never resolves into a callable.
"""

from dataclasses import dataclass, field

from nwqlib.blocks.records import SelectedKernel
from nwqlib.artifacts import UnavailableOutput
from nwqlib.execution import KernelApplication, ScalarValue
from nwqlib.problems.inputs import PhysicalScale


@dataclass(frozen=True)
class KernelOutput:
    """One invocation's selected identity, statistics, named arrays and applications.

    Attributes:
        plan_id: Exact selected Plan identity for the invocation.
        selected_kernel_id: Selected host-kernel declaration that produced this output.
        scalars: Ordered named scalar values in the declaration's frames.
        physical_scale: Positive factor that restores physical magnitude to
            unit-normalized scalars or arrays, or None. Each Method defines it,
            as ``ObservationChunk.physical_scale`` lists.
        physical_scale_unavailable: Why ``physical_scale`` is None, because the
            scale is unavailable or no recovery applies. Exactly one of the two
            fields is set.
        arrays: Named actual complex128 arrays matching selected output declarations.
        applications: Actual selected operator/application receipts for this invocation.
        unavailable: Declared named outputs that could not be produced, with reasons.
    """

    plan_id: str
    selected_kernel_id: str
    scalars: tuple[ScalarValue, ...]
    physical_scale: PhysicalScale | None
    physical_scale_unavailable: str | None = None
    arrays: tuple[tuple[str, object], ...] = ()
    applications: tuple[KernelApplication, ...] = ()
    unavailable: tuple[UnavailableOutput, ...] = ()


    def __post_init__(self):
        if (self.physical_scale is None) != (self.physical_scale_unavailable is not None):
            raise ValueError("host physical scale requires exactly one value or unavailable reason")
        if self.physical_scale_unavailable is not None and not self.physical_scale_unavailable.strip():
            raise ValueError("host physical scale unavailable reason must be nonempty")


@dataclass(frozen=True, init=False, eq=False, slots=True)
class BoundKernel:
    """Method-owned immutable inputs and an exact static or point-bound closure.

    A point binder only captures already admitted inputs and arguments.
    Numerical work starts at invocation. Persisted declarations never load
    either callable.
    A point closure may capture its context, but must not keep the live Run.

    Attributes:
        record: Exact SelectedKernel declaration associated with the bound closure.
        plan_id: Identity of the original Plan whose immutable inputs the closure uses.
    """

    record: SelectedKernel
    plan_id: str
    _call: object = field(repr=False)
    _binder: object = field(repr=False)
    _realization_id: str | None

    def __init__(self, *args, **kwargs):
        raise TypeError("host kernels require the method's live selection factory")

    @classmethod
    def _bind(cls, plan, record, call):
        return cls._make(plan, record, call=call, binder=None)

    @classmethod
    def _bind_pointwise(cls, plan, record, binder):
        """Return a pointwise binding whose ``binder(realization, declaration, run)`` returns the no-argument closure at each admitted point (see ``_at``)."""
        return cls._make(plan, record, call=None, binder=binder)

    @classmethod
    def _make(cls, plan, record, *, call, binder):
        if (type(record) is not SelectedKernel or record not in plan.construction.kernels
                or (call is None) == (binder is None) or not callable(call if binder is None else binder)):
            raise ValueError("kernel binding requires its exact selected declaration and live method factory")
        result = object.__new__(cls)
        for name, value in (("record", record), ("plan_id", plan.content_id), ("_call", call),
                            ("_binder", binder), ("_realization_id", None)):
            object.__setattr__(result, name, value)
        return result

    def _at(self, realization, declaration, run):
        """Return the binding to invoke at one admitted point of the same Plan.

        The realization and the Run must both belong to this binding's Plan.
        A static binding returns itself, and only for its unchanged
        declaration. A pointwise binding accepts the template declaration or
        a revision of it (parent identity equal to the template, with the same
        name, implementation, inputs and dependencies), so a point can change
        its declared scalars, outputs and work but never its implementation or
        inputs. The binder receives the realization, declaration and Run and
        returns one no-argument closure, which must not keep the Run. ``_at``
        wraps that closure as a new static binding tied to the realization.
        """
        if realization.plan_id != self.plan_id:
            raise ValueError("host point belongs to another exact Plan")
        if run.plan.content_id != self.plan_id:
            raise ValueError("host point run belongs to another exact Plan")
        if self._binder is None:
            if declaration != self.record:
                raise ValueError("static host binding cannot supply a changed point declaration")
            return self
        if declaration != self.record and (declaration.parent_id != self.record.content_id or any(
                getattr(declaration, field) != getattr(self.record, field)
                for field in ("name", "implementation", "inputs", "dependencies"))):
            raise ValueError("host point declaration differs from its bound template/input")
        call = self._binder(realization, declaration, run)
        if not callable(call):
            raise TypeError("host point binder must return its immutable execution closure")
        result = object.__new__(type(self))
        for name, value in (("record", declaration), ("plan_id", self.plan_id), ("_call", call),
                            ("_binder", None), ("_realization_id", realization.content_id)):
            object.__setattr__(result, name, value)
        return result

    def _invoke(self):
        """Run the bound closure once and require output tied to this Plan and declaration."""
        if self._binder is not None:
            raise ValueError("pointwise host binding requires preparation at an admitted point")
        result = self._call()
        if (type(result) is not KernelOutput or result.plan_id != self.plan_id
                or result.selected_kernel_id != self.record.content_id):
            raise ValueError("native kernel output belongs to another Plan or selected kernel")
        return result
