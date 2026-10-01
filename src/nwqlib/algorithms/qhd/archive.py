"""Store the original symbolic input and already-selected QHD tables."""

import pickle
import sympy as sp
from nwqlib.blocks.kernels import BoundKernel
from nwqlib.blocks.selection import SelectedBlock
from nwqlib.problems.records import Optimization
from .method import QHD, _execute_theory
from .records import QHDReconstruction, validate_selection

# The one QHD archive format this version writes and reads. It changes with
# the stored Method or reconstruction fields, with the admission contract
# that a saved Plan was planned under, and with the fields of QHDAnalysis.
# A saved Result or Run folder keeps its QHDAnalysis beside the Plan and
# loads the Plan through this hook before it reads that record, so this
# format is what refuses a folder whose QHDAnalysis has other fields.
# Plans of qhd/5 and later passed the binary64 range admission of planning
# (validation._normal_range) and record their lower-range omissions under a
# named policy (records.QHDRangeOmissions). The phase and angle entries of
# the error ledger rest on both (circuit_errors), and loading does not
# replan, so a qhd/4 Plan could carry out-of-range arithmetic that those
# entries would not charge. Earlier formats are rejected rather than
# converted, because NWQLib owes no compatibility with the development
# schemas that preceded release 1.0 (docs/FRAMEWORK.md, "API stability").
FORMAT = "qhd/7"


class _SymbolicReader(pickle.Unpickler):
    """Unpickler that resolves only SymPy expression classes and rebuilds each expression node as it was saved.

    SymPy pickles an expression node as its class and ``args``
    (``Basic.__getnewargs__``), and plain unpickling calls the class with
    those args under SymPy's global ``evaluate`` parameter. A node built with
    ``evaluate=False``, such as the quotient in ``Add(x, Mul(x + 1, Pow(x +
    1, -1, evaluate=False), evaluate=False), evaluate=False)``, would then
    come back evaluated, here as ``x + 1``. Disabling evaluation for the
    whole load is no remedy either, because some constructors form their
    args with arithmetic of their own: ``Sum`` and ``Integral`` store
    ``1*f`` for the summand f, which becomes ``Mul(1, f)`` without
    evaluation.

    ``find_class`` therefore returns a stand-in that records each node's
    class and args, and ``load`` builds the nodes from the leaves up
    (``_rebuild``). Each node is first built with evaluation, as plain
    unpickling builds it (``_build``). When that changes its class or args,
    or raises, it is built again with evaluation disabled, which reproduces
    a node that was built with ``evaluate=False``. An expression that plain
    unpickling reproduces thus comes back as it does, and a reopened
    objective, variable or constraint keeps the ``srepr`` and content
    identity that the archive records. A node that neither build reproduces
    keeps the evaluated build, or raises its error, and the checks that
    compare the reopened expressions with the records
    (``records.validate_selection`` and the problem identities of the
    augmented-Lagrangian and refinement archives) refuse such a build.
    """

    def find_class(self, module, name):
        # Only SymPy expression classes belong to this format. In particular,
        # no callable, native closure or arbitrary Python global is loaded.
        if not module.startswith("sympy."):
            raise pickle.UnpicklingError("QHD archive requires SymPy expression data")
        cls = super().find_class(module, name)
        if not isinstance(cls, type) or not issubclass(cls, sp.Basic):
            raise pickle.UnpicklingError("QHD archive requires a SymPy expression class")
        return type(cls.__name__, (_StandIn,), {"saved": cls})

    def load(self):
        return _rebuild(super().load())


class _StandIn:
    """What ``_SymbolicReader.find_class`` returns for the SymPy class ``saved``.

    Unpickling calls it where it would call ``saved``, so each node is read
    as a ``_Node`` and nothing is constructed before ``_rebuild``.
    """

    saved = None

    def __new__(cls, *args, **kwargs):
        return _Node(cls.saved, args, kwargs)


class _Node:
    """A saved SymPy node read by ``_SymbolicReader``: its class, args, keyword args and pickled state."""

    __slots__ = ("cls", "args", "kwargs", "state")

    def __init__(self, cls, args, kwargs):
        self.cls, self.args, self.kwargs, self.state = cls, args, kwargs, None

    def __setstate__(self, state):
        self.state = state


_CONTAINERS = (_Node, tuple, list, dict, set, frozenset)


def _rebuild(root):
    """Return ``root`` with every ``_Node`` in it built, children before parents.

    The walk keeps its own stack, so the depth of an expression is not bound
    by Python's recursion limit, as it is not for plain unpickling. ``built``
    maps each node or container already rebuilt, by identity, to its value,
    so a node that the pickle shares is built once and stays shared. A class
    that the pickle holds as a value comes back as that class. A container
    that holds itself, which no SymPy expression does, is refused, since
    its rebuild would not end.
    """
    built, pending = {}, set()

    def value(item):
        if isinstance(item, type) and issubclass(item, _StandIn):
            return item.saved
        return built[id(item)] if isinstance(item, _CONTAINERS) else item

    stack = [(root, False)]
    while stack:
        item, ready = stack.pop()
        if not isinstance(item, _CONTAINERS) or id(item) in built:
            continue
        if isinstance(item, _Node):
            parts = (item.args, item.kwargs, item.state)
        else:
            parts = tuple(item) + (tuple(item.values()) if isinstance(item, dict) else ())
        if not ready:
            # A container popped while its own parts are being rebuilt is one of its own parts.
            if id(item) in pending:
                raise pickle.UnpicklingError("QHD archive holds a container that contains itself")
            pending.add(id(item))
            stack.append((item, True))
            stack.extend((part, False) for part in parts)
            continue
        if isinstance(item, _Node):
            built[id(item)] = _build(item.cls, value(item.args), value(item.kwargs), value(item.state))
        elif isinstance(item, dict):
            built[id(item)] = {value(key): value(part) for key, part in item.items()}
        else:
            built[id(item)] = type(item)(value(part) for part in item)
        pending.discard(id(item))
    return value(root)


def _build(cls, args, kwargs, state):
    """Build one node as plain unpickling does, or without evaluation when only that reproduces it.

    Plain unpickling calls ``cls.__new__`` with the saved args and then
    ``__setstate__`` with any saved state. The first build turns evaluation
    on, so a load inside the caller's ``sympy.evaluate(False)`` builds the
    same nodes. An atom, such as a symbol or number, is saved with its name
    or value rather than SymPy args, so it is kept as built. A compound node
    is reproduced when the built object has class ``cls`` and args equal to
    the saved args.
    """

    def construct(evaluate):
        # sympy.evaluate sets a thread-local flag, and each change of the flag
        # clears SymPy's construction cache, which all threads share. While a
        # node is built without evaluation, another thread that calls the same
        # constructor with the same args can receive the unevaluated node, and
        # a node that another thread cached meanwhile can keep this build from
        # reproducing the saved one (sympy.core.parameters).
        with sp.evaluate(evaluate):
            node = cls.__new__(cls, *args, **kwargs)
        if state is not None:
            node.__setstate__(state)
        return node

    def reproduced(node):
        return type(node) is cls and node.args == args

    if issubclass(cls, sp.Atom) or not all(isinstance(arg, sp.Basic) for arg in args):
        return construct(True)
    try:
        node = construct(True)
    except Exception:
        # Evaluation can fail where the saved node needs none, for example a
        # comparison of Max arguments that SymPy cannot decide.
        node = construct(False)
        if reproduced(node):
            return node
        raise
    if reproduced(node):
        return node
    unevaluated = construct(False)
    return unevaluated if reproduced(unevaluated) else node


def save(method, plan, files):
    """Write the SymPy objective and variables as a pickle, then return the ``FORMAT`` manifest.

    ``load`` reads the pickle back with ``_SymbolicReader``, which resolves
    SymPy classes only and evaluates no text. The Plan, Problem and Output
    records go through the shared archive writers, and the Method is stored
    as its JSON dump.
    """
    with files.writer("objective.pickle") as stream:
        pickle.dump((plan.problem.objective, plan.problem.variables), stream, protocol=5)
    return dict(
        format=FORMAT,
        plan=files.write_plan(plan),
        problem=files.write_problem(plan.problem),
        output=files.write_output(plan.output),
        method=method.model_dump(mode="json", exclude_computed_fields=True),
        objective="objective.pickle",
    )


def load(saved, files):
    """Rebind a saved QHD Plan from its stored objective and selected tables.

    The Plan is restored from the stored Method, reconstruction and
    SymPy objective without recompiling the schedule or evaluating the
    objective. ``validate_selection`` compares the stored objective text and
    variable order with the Problem, checks table supports and shapes, and
    binds bounds and reconstruction to the selection by content identity.
    It does not compare table values with the objective. The classical
    kernel or native block is then bound to the stored reconstruction.
    Any other format, including an earlier QHD format, is rejected before a
    file is read.
    """
    if saved.get("format") != FORMAT:
        raise ValueError(
            f"unsupported QHD archive format {saved.get('format')!r}. This NWQLib reads only "
            f"{FORMAT!r} and does not convert QHD data saved by an earlier release"
        )
    with files.read_path(saved["objective"]).open("rb") as stream:
        expression, variables = _SymbolicReader(stream).load()
    fields = dict(saved["problem"]["fields"])
    fields.update(objective=expression, variables=variables)
    problem = Optimization.model_validate(fields)
    method = QHD.model_validate(saved["method"])
    reconstruction = QHDReconstruction.model_validate(saved["plan"]["reconstruction"])
    plan = files.read_plan(
        saved["plan"],
        problem=problem,
        method=method,
        output=files.read_output(saved["output"]),
        reconstruction=reconstruction,
    )
    validate_selection(plan)
    if plan.execution == "classical":
        (kernel,) = plan.construction.kernels
        blocks = (BoundKernel._bind(plan, kernel, lambda: _execute_theory(plan, kernel)),)
    else:
        from .native import construct_qhd

        blocks = (
            SelectedBlock.bind(
                plan.construction.selections[0],
                payload=(reconstruction, method, problem.bounds, problem.variable_names),
                constructor=construct_qhd,
            ),
        )
    return plan._bind(blocks=blocks)
