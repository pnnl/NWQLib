"""Bounded exact expression and declared lifecycle admission, without execution.

Every Program is admitted here before any consumer does work with it: Program
construction, Plan points, experiment selection, logical lowering, the
resource fold and acquisition resolution. The public contract is
docs/development/program_checks.md, which states the declared quantum and
classical lifecycle.
"""

import math

from .expressions import Binary, Constant, ExprRef, ParameterRef, number, topological
from .records import (
    AdaptiveLoop, Allocate, BlockCall, Branch, ClassicalStage, CoherentRegion,
    Measure, MeasurementBatch, Parallel, Program, Readiness, Release, Repeat, Reset, Sequence,
)


def unique(items, key, what):
    """Index items by a name field, rejecting duplicates instead of letting one shadow another."""
    result = {}
    for item in items:
        name = getattr(item, key)
        if name in result:
            raise ValueError(f"duplicate {what}: {name}")
        result[name] = item
    return result


def children(node):
    """Return the definition IDs a node refers to: its ordered children, body or two branch arms."""
    if isinstance(node, (Sequence, Parallel)):
        return node.children
    if isinstance(node, (Repeat, CoherentRegion, AdaptiveLoop, MeasurementBatch)):
        return (node.body,)
    if isinstance(node, Branch):
        return (node.when_true, node.when_false)
    return ()


class AdmissionStepsExceeded(ValueError):
    """A Program's kept field slots or admission work exceed ``AdmissionLimits.max_steps``.

    A Method whose ``max_admission_steps`` field sets the ceiling catches this
    type to name that field (``steps_exceeded``, ``admission_refusal``,
    ``admitted_program``). Raised inside Program construction, it reaches the
    caller wrapped in Pydantic's ValidationError, whose ``errors()`` entry
    keeps it as ``ctx["error"]``.

    Attributes:
        stage: ``"stored field inventory"`` or ``"admission work"``.
        need: For the stored field inventory, the complete count of kept
            field slots: a ``max_steps`` of at least this value passes the
            inventory, and later admission, preparation and lowering can
            still need more. For admission work, the count when the check
            stopped; work is charged before the item it pays for, so the
            complete count is at least this value, and it is not a value
            that admits.
        ceiling: The ``max_steps`` that refused.
        complete: Whether ``need`` is the complete count of its stage
            (true for the stored field inventory only).
    """

    def __init__(self, stage, need, ceiling):
        self.stage, self.need, self.ceiling = stage, need, ceiling
        self.complete = stage == "stored field inventory"
        if self.complete:
            super().__init__(f"{stage} exceeds max_steps={ceiling}: {need} counted, the complete inventory")
        else:
            super().__init__(f"{stage} exceeds max_steps={ceiling}: at least {need} counted when the check stopped")


class AdmissionDefinitionsExceeded(ValueError):
    """A Program keeps more definitions and declarations than ``AdmissionLimits.max_definitions``.

    The count is the complete number of kept definitions, expressions,
    parameters, registers, classical values and signatures, taken before any
    table is built. No Method option sets this shared limit.

    Attributes:
        need: The complete count.
        ceiling: The ``max_definitions`` that refused.
    """

    def __init__(self, need, ceiling):
        super().__init__(f"kept definitions exceed max_definitions={ceiling}: {need} counted")
        self.need, self.ceiling = need, ceiling


def limit_exceeded(error, kinds=(AdmissionStepsExceeded, AdmissionDefinitionsExceeded)):
    """Return the admission-limit refusal of ``kinds`` that a Program construction error carries, or None.

    Pydantic wraps an error raised in the Program validator and keeps the
    original as ``ctx["error"]`` of its ``errors()`` entry.
    """
    if isinstance(error, kinds):
        return error
    from pydantic import ValidationError
    if isinstance(error, ValidationError):
        for item in error.errors():
            found = item.get("ctx", {}).get("error")
            if isinstance(found, kinds):
                return found
    return None


def steps_exceeded(error):
    """Return the ``AdmissionStepsExceeded`` that a Program construction error carries, or None."""
    return limit_exceeded(error, (AdmissionStepsExceeded,))


def admission_refusal(option, exceeded, *, subject="the Program"):
    """Word a refusal that names the limit that refused.

    For ``AdmissionStepsExceeded``, ``option`` is the public spelling of the
    Method field owning ``max_steps``, such as
    ``"FixedGCIM.max_admission_steps"``. The message names the refused stage,
    its counted need and the ceiling. For the stored field inventory the
    need is the complete count, and that value passes the inventory; for
    admission work it is the lower bound reached when the check stopped.
    Neither is a promised admitting value: later admission, preparation and
    lowering can need more. For
    ``AdmissionDefinitionsExceeded`` the message names the shared
    ``AdmissionLimits.max_definitions``, the complete count and the ceiling.
    """
    if isinstance(exceeded, AdmissionDefinitionsExceeded):
        return ValueError(
            f"{subject} keeps {exceeded.need} definitions, expressions, parameters, registers, classical "
            f"values and signatures, exceeding AdmissionLimits.max_definitions={exceeded.ceiling}. This "
            "shared IR limit has no Method option; select a smaller construction")
    if exceeded.complete:
        return ValueError(
            f"{subject} exceeds {option}={exceeded.ceiling}: its {exceeded.stage} counts {exceeded.need} kept "
            f"field slots, the complete inventory. Raise {option} to at least {exceeded.need} to pass this "
            "inventory; later admission, preparation and lowering can need more. This changes the "
            "metadata-work allowance, not the selected quantum operations")
    return ValueError(
        f"{subject} exceeds {option}={exceeded.ceiling}: its {exceeded.stage} counted at least "
        f"{exceeded.need} units when the check stopped. Raise {option} to at least {exceeded.need}; "
        "the complete count was not measured, and later admission stages can need more. This changes the metadata-work allowance, "
        "not the selected quantum operations")


def admitted_program(option, max_steps, **fields):
    """Construct a Program with ``AdmissionLimits(max_steps=max_steps)`` or refuse naming ``option``.

    A Method whose field ``max_admission_steps`` sets its Programs' ceiling
    builds them here, so an oversized Program is refused as a ValueError
    naming that field (``admission_refusal``), and a Program above the
    shared ``max_definitions`` is refused naming that limit, not as a
    Pydantic validation error.
    """
    from pydantic import ValidationError
    from .expressions import AdmissionLimits
    try:
        return Program(**fields, limits=AdmissionLimits(max_steps=max_steps))
    except ValidationError as error:
        exceeded = limit_exceeded(error)
        if exceeded is None:
            raise
        raise admission_refusal(option, exceeded) from error


def check_program(program):
    """Admit a whole Program at its own bindings and return its Readiness."""
    return _Admission(program).check()


class _Admission:
    """Abstract interpretation of a Program's declared effects before any work.

    The check follows declarations only. It never simulates a state and never
    proves that a selected implementation honors its signature. Its purpose
    is to reject an illegal lifecycle before a circuit, a resource estimate or
    a native preparation exists: use after release, a live quantum state
    crossing a host boundary, a coherence promise that a measurement already
    broke, branch arms that disagree on live registers, a loop whose state
    changes between iterations, or an independent experiment that leaks a wire.

    Each register is tracked as a ``(state, epoch, group)`` triple:

    - ``state`` is ``zero``, ``coherent``, ``measured``, ``mixed``, ``unknown``
      or ``released``. It records which promise a later port requirement or
      CoherentRegion claim may rely on. A measured register holds the
      observed computational basis state, so a local unitary turns ``zero``
      or ``measured`` into ``coherent``. It cannot repair ``mixed``.
    - ``epoch`` counts the register's coherence generations. Measurement,
      reset, release, a new preparation and reallocation advance it, so a
      claim made before a measurement cannot match after it. ``None`` means
      unresolved: it blocks readiness instead of matching a claim.
    - ``group`` is the interned set of registers that a joint call may have
      correlated with this one. A nonunitary effect on one member makes every
      partner ``mixed``, because measuring part of an entangled state leaves
      the rest without a coherent-state guarantee.

    Known violations raise ``ValueError``. Unresolved requirements, such as an
    unbound width or an unknown block effect, become readiness blockers, so a
    symbolic Program stays representable until the consumer that needs a
    value binds it.

    Work is charged by ``tick`` before the collection or context it pays for
    is built, against ``AdmissionLimits.max_steps`` (ENGINEERING_CONSTANTS.md,
    "Shared Program admission limits"). Visits are memoized per distinct
    (node, register states, available classical values, binding, protected
    registers) context, so cost follows the kept graph and its distinct
    contexts, never Repeat counts or setting multiplicity. The resource fold
    reuses one instance and raises ``step_limit`` in proportion to the work
    measured here.

    Attributes:
        p: The Program under admission.
        limits: Its AdmissionLimits.
        steps: Admission work charged so far, including expression evaluations.
        evaluations: Expression evaluations, reported separately in Readiness.
        step_limit: Work ceiling, ``limits.max_steps`` unless a consumer raises it.
        blockers: Unresolved requirements collected so far.
        nodes: Definition ID to node.
        expr: Expression ID to expression value.
        registers: Register name to Register.
        classical: Classical value name to ClassicalValue.
        signatures: Block signature name to BlockSignature.
        params: Parameter name to Parameter.
        definitions: Definition ID to Definition record, used by experiment selection.
        expression_definitions: Expression ID to Expression record, used by experiment selection.
        contexts: Evaluated expression values per distinct binding, filled by ``expressions``.
        memo: Result of ``visit`` per distinct context, as described above.
        groups: Interned correlation groups, so equal groups are one object.
        contains_batch: Whether each node is or contains a MeasurementBatch, set by setup.
        node_order: Definition IDs in dependency order, set by setup.
        order: Expression IDs in dependency order, set by setup.
        types: Expression ID to ``integer``, ``real`` or ``bool``, set by setup.
    """

    def __init__(self, program):
        """Index the Program's tables, rejecting an oversized or ambiguous Program first.

        The slot walk counts every kept record and tuple field of the Program
        and rejects when the complete count exceeds ``max_steps``, before any
        table is built. Duplicate names in any table reject. Definitions, expressions,
        parameters, registers, classical values and signatures together must
        not exceed ``max_definitions``. Nothing here depends on bindings.
        ``setup`` and ``check`` do the structural and lifecycle work.
        """
        self.p = program
        self.limits = program.limits
        self.steps = 0
        self.evaluations = 0
        self.blockers = set()
        # Count all kept record/tuple slots, including edges, settings and ports.
        # This does not expand references or serialize nested identities. An
        # inventory above max_steps is counted to its end, so the refusal
        # reports the complete count; no table is built for it.
        stored_slots = 0
        def reserve(amount):
            nonlocal stored_slots
            stored_slots += amount
        from .expressions import walk_kept
        for _ in walk_kept(program, reserve):
            pass
        if stored_slots > self.limits.max_steps:
            raise AdmissionStepsExceeded("stored field inventory", stored_slots, self.limits.max_steps)
        self.definitions = unique(program.definitions, "id", "definition")
        self.expression_definitions = unique(program.expressions, "id", "expression")
        self.nodes = {key: value.node for key, value in self.definitions.items()}
        self.expr = {key: value.value for key, value in self.expression_definitions.items()}
        self.params = unique(program.parameters, "name", "parameter")
        self.registers = unique(program.registers, "name", "register")
        self.classical = unique(program.classical, "name", "classical value")
        self.signatures = unique(program.signatures, "name", "signature")
        inventory = (program.definitions, program.expressions, program.parameters,
                     program.registers, program.classical, program.signatures)
        if sum(map(len, inventory)) > self.limits.max_definitions:
            raise AdmissionDefinitionsExceeded(sum(map(len, inventory)), self.limits.max_definitions)
        self.contexts = {}
        self.memo = {}
        self.groups = {}
        # Program admission work is bounded by max_steps. A consumer that repeats
        # the admitted traversal with more work per step, such as the resource
        # fold, sets its own limit from the work this check measured.
        self.step_limit = self.limits.max_steps

    def tick(self, amount=1):
        """Charge work before doing it, so an oversized Program rejects before the work exists."""
        self.steps += amount
        if self.steps > self.step_limit:
            if self.step_limit == self.limits.max_steps:
                raise AdmissionStepsExceeded("admission work", self.steps, self.step_limit)
            raise ValueError("resource fold work exceeds its limit proportional to Program admission work")

    def bounded(self, value):
        """Admit an exact integer within max_integer_bits or a finite real."""
        if type(value) is int and value.bit_length() > self.limits.max_integer_bits:
            raise ValueError("integer growth exceeds max_integer_bits")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("expression result is nonfinite")
        return value

    def binding_map(self, bindings):
        """Admit distinct declared parameters with in-domain values and return plain numbers."""
        self.tick(3 * len(bindings))
        table = unique(bindings, "parameter", "binding")
        for name, binding in table.items():
            if name not in self.params:
                raise ValueError(f"unknown bound parameter: {name}")
            self.bounded(self.params[name].admit(binding.value))
        return {key: number(binding.value) for key, binding in table.items()}

    def reference(self, ref, expected=None):
        if ref not in self.types:
            raise ValueError(f"unresolved expression: {ref}")
        if expected and self.types[ref] != expected:
            raise ValueError(f"expression {ref} must have type {expected}")

    def integer_ref(self, value):
        if isinstance(value, ExprRef):
            self.reference(value.expression, "integer")
        else:
            self.bounded(value)

    def arguments(self, arguments, context, formal=None):
        """Check call or stage arguments against their formal domains in one context.

        A known value outside the formal domain rejects. An unknown value
        becomes a blocker, because the consumer needs a concrete argument.
        """
        self.tick(2 * len(arguments))
        table = unique(arguments, "parameter", "argument")
        for name, item in table.items():
            self.reference(item.value.expression)
            if self.types[item.value.expression] == "bool":
                raise ValueError("boolean expression cannot be a scalar argument")
            value = context[item.value.expression]
            if formal is not None:
                parameter = formal[name]
                if self.types[item.value.expression] != parameter.domain:
                    raise ValueError("argument domain does not match formal parameter")
                if value is not None:
                    # Domain admission uses the same exact public scalar representation.
                    from nwqlib.core.records import Float64
                    parameter.admit(Float64(value=value) if parameter.domain == "real" else value)
            if value is None:
                self.blockers.add(f"unresolved argument: {item.value.expression}")

    def setup(self):
        """Check the binding-independent tables once, before any context is visited.

        This resolves every local reference, including unused definitions, so
        a dangling reference cannot hide behind an unreached branch. It orders
        the node and expression DAGs within the depth limit, types every
        expression, and checks signatures, port order and aliasing, Parallel
        disjointness, range-axis endpoints and setting bindings. It also
        records which nodes contain a MeasurementBatch, which fixes the
        terminal/outer distinction for acquisition kinds. Nothing here depends
        on parameter values, so a Plan runs it once and reuses the index for
        every selected point.
        """
        if self.p.root not in self.nodes:
            raise ValueError("unresolved root reference")
        self.node_order = topological(self.nodes, children, self.limits)
        # One bounded, shared-DAG inventory owns terminal acquisition ancestry.
        # Resource accounting reuses it; neither path expands repetitions/axes.
        self.contains_batch = {}
        for name in self.node_order:
            node = self.nodes[name]
            refs = children(node)
            self.tick(1 + len(refs))
            descendant_batch = any(self.contains_batch[child] for child in refs)
            if isinstance(node, MeasurementBatch) and descendant_batch and node.observation_kind is not None:
                raise ValueError("nonterminal MeasurementBatch must have observation_kind=None")
            self.contains_batch[name] = isinstance(node, MeasurementBatch) or descendant_batch
        self.order = topological(
            self.expr, lambda item: (item.left, item.right) if isinstance(item, Binary) else (),
            self.limits,
        )
        self.types = {}
        for parameter in self.params.values():
            for bound in (parameter.lower, parameter.upper):
                if bound is not None:
                    self.bounded(number(bound))
        for key in self.order:
            item = self.expr[key]
            if isinstance(item, Constant):
                self.bounded(number(item.value))
                dtype = "integer" if type(item.value) is int else "real"
            elif isinstance(item, ParameterRef):
                if item.parameter not in self.params:
                    raise ValueError(f"undeclared parameter: {item.parameter}")
                dtype = self.params[item.parameter].domain
            else:
                left, right = self.types[item.left], self.types[item.right]
                if "bool" in (left, right) or left != right:
                    raise ValueError("binary operands must have the same numeric domain")
                if item.op == "ceildiv" and left != "integer":
                    raise ValueError("ceildiv requires integer operands")
                dtype = "bool" if item.op in {"eq", "lt", "le"} else left
            self.types[key] = dtype
        for ref in self.p.constraints:
            self.reference(ref.expression, "bool")
        for reg in self.registers.values():
            self.integer_ref(reg.width)
        for value in self.classical.values():
            if value.width is not None:
                self.integer_ref(value.width)
        for signature in self.signatures.values():
            unique(signature.quantum, "name", "formal port")
            unique(signature.parameters, "name", "formal parameter")
            for port in signature.quantum:
                self.integer_ref(port.width)
        # Check all references, including unused definitions. Lifecycle is contextual.
        for node in self.nodes.values():
            if isinstance(node, Parallel):
                wires = set()
                for child in node.children:
                    self.tick()
                    call = self.nodes[child]
                    if not isinstance(call, BlockCall) or call.signature not in self.signatures:
                        raise ValueError("Parallel requires direct declared BlockCalls")
                    signature = self.signatures[call.signature]
                    self.tick(len(call.ports) + len(signature.quantum))
                    if (signature.interface != "declared" or any(
                        port.ensures not in {"unitary", "preserve"} for port in signature.quantum
                    )):
                        raise ValueError("Parallel requires declared unitary/preserve effects")
                    actual = {port.wire for port in call.ports}
                    if wires & actual:
                        raise ValueError("Parallel calls share a wire dependency")
                    wires.update(actual)
            if isinstance(node, (Allocate, Release, Reset, Measure)):
                if node.wire not in self.registers:
                    raise ValueError(f"undeclared register: {node.wire}")
            if isinstance(node, Measure) and node.result not in self.classical:
                raise ValueError("undeclared measurement result")
            if isinstance(node, Measure) and self.classical[node.result].dtype != "bits":
                raise ValueError("measurement result must have bits type")
            if isinstance(node, Branch):
                if node.condition not in self.classical or self.classical[node.condition].dtype != "bool":
                    raise ValueError("branch condition must be a declared bool")
            if isinstance(node, (Repeat, AdaptiveLoop)):
                self.integer_ref(node.count if isinstance(node, Repeat) else node.max_rounds)
            if isinstance(node, CoherentRegion):
                unique(node.claims, "wire", "coherent claim")
                if any(claim.wire not in self.registers for claim in node.claims):
                    raise ValueError("undeclared coherent register")
                for claim in node.claims:
                    self.bounded(claim.epoch)
            if isinstance(node, BlockCall):
                if node.signature not in self.signatures:
                    raise ValueError("undeclared block signature")
                signature = self.signatures[node.signature]
                mapping = unique(node.ports, "port", "port mapping")
                if tuple(mapping) != tuple(port.name for port in signature.quantum):
                    raise ValueError("port mappings must match formal port order")
                actual = [port.wire for port in node.ports]
                if len(set(actual)) != len(actual):
                    raise ValueError("aliased exclusive quantum ports")
                if any(name not in self.registers for name in actual):
                    raise ValueError("undeclared mapped register")
                args = unique(node.arguments, "parameter", "argument")
                if args.keys() != {p.name for p in signature.parameters}:
                    raise ValueError("block arguments must match formal parameters")
                for argument in node.arguments:
                    self.reference(argument.value.expression)
            stages = ((node,) if isinstance(node, ClassicalStage) else
                      (node.policy,) if isinstance(node, AdaptiveLoop) else ())
            for stage in stages:
                unique(stage.arguments, "parameter", "argument")
                for argument in stage.arguments:
                    self.reference(argument.value.expression)
                if len(set(stage.outputs)) != len(stage.outputs):
                    raise ValueError("duplicate classical output")
                if any(name not in self.classical for name in stage.inputs + stage.outputs):
                    raise ValueError("undeclared classical stage input/output")
            if isinstance(node, MeasurementBatch):
                if node.repetitions is not None:
                    self.integer_ref(node.repetitions)
                axes = unique(node.axes, "parameter", "range axis")
                for name, axis in axes.items():
                    if name not in self.params or self.params[name].domain != "integer":
                        raise ValueError("range axis requires an integer parameter")
                    for endpoint in (axis.start, axis.stop - 1, axis.step):
                        self.bounded(endpoint)
                    self.params[name].admit(axis.start)
                    last = axis.start + ((axis.stop - axis.start - 1) // axis.step) * axis.step
                    self.params[name].admit(last)
                for setting in node.settings:
                    binding = self.binding_map(setting.bindings)
                    if axes.keys() & binding.keys():
                        raise ValueError("setting bindings conflict with range axes")

    def expressions(self, binding):
        """Evaluate every kept expression once for one binding context.

        A missing parameter evaluates to None, meaning unknown, and never to
        zero or false. A false constraint rejects, while an unknown constraint
        or width becomes a blocker. Multiplication checks the result's bit
        length before forming it, so an oversized integer is never allocated.
        Results are cached per binding, which keeps width scans at one per
        distinct context however many settings share it.
        """
        self.tick(len(binding))
        key = frozenset(binding.items())
        if key in self.contexts:
            return self.contexts[key]
        values = {}
        for name in self.order:
            self.tick()
            self.evaluations += 1
            item = self.expr[name]
            if isinstance(item, Constant):
                value = number(item.value)
            elif isinstance(item, ParameterRef):
                value = binding.get(item.parameter)
            else:
                left, right = values[item.left], values[item.right]
                if item.op == "ceildiv" and right is not None and right <= 0:
                    raise ValueError("ceildiv denominator must be positive")
                if left is None or right is None:
                    value = None
                elif item.op == "add":
                    value = left + right
                elif item.op == "multiply":
                    # Bound before multiplication allocates its potentially huge integer.
                    # For nonzero integers bit_length(a*b) >= bit_length(a) +
                    # bit_length(b) - 1, so this rejects every product that must
                    # exceed the limit. A product one bit over is formed and
                    # rejected by bounded() below.
                    if (type(left) is int and left and right
                            and left.bit_length() + right.bit_length() - 1 > self.limits.max_integer_bits):
                        raise ValueError("integer growth exceeds max_integer_bits")
                    value = left * right
                elif item.op == "ceildiv":
                    # ceil(a/b) = -floor(-a/b) for b > 0, exact on Python integers.
                    value = -(-left // right)
                elif item.op == "min":
                    value = min(left, right)
                elif item.op == "max":
                    value = max(left, right)
                elif item.op == "eq":
                    value = left == right
                elif item.op == "lt":
                    value = left < right
                else:
                    value = left <= right
            values[name] = self.bounded(value)
        self.tick(len(self.p.constraints))
        for ref in self.p.constraints:
            value = values[ref.expression]
            if value is None:
                self.blockers.add(f"unresolved constraint: {ref.expression}")
            elif not value:
                raise ValueError(f"constraint is false: {ref.expression}")
        # Width requirements belong to this same context, including unknowns.
        self.validate_widths(values)
        self.contexts[key] = values
        return values

    def integer(self, value, context, role, positive=False):
        """Read a count or width. Unknown adds a blocker, and a known value below the minimum rejects."""
        self.tick()
        result = context[value.expression] if isinstance(value, ExprRef) else value
        if result is None:
            self.blockers.add(f"unresolved {role}: {value.expression}")
        elif result < int(positive):
            raise ValueError(f"{role} must be {'positive' if positive else 'nonnegative'}")
        return result

    def width(self, left, right, context):
        a = self.integer(left, context, "width")
        b = self.integer(right, context, "width")
        if a is not None and b is not None and a != b:
            raise ValueError("incompatible register/port widths")

    @staticmethod
    def live(q, name):
        if name not in q or q[name][0] == "released":
            raise ValueError(f"register is not live (unallocated or released): {name}")
        return q[name]

    def classical_inputs(self, inputs, available):
        self.tick(len(inputs))
        if not set(inputs) <= available:
            raise ValueError("classical value read before definition")

    def stage(self, stage, q, available, context):
        """Admit one ClassicalStage and return the classical values available after it.

        Its inputs must already be available and its arguments admitted. A
        host stage runs outside the quantum job, so no register may be live
        when it starts. Its outputs join the available values.
        """
        self.tick(1 + len(available) + len(stage.outputs))
        self.classical_inputs(stage.inputs, available)
        self.arguments(stage.arguments, context)
        if stage.boundary == "host" and self.live_names(q):
            raise ValueError("host boundary cannot transport live quantum state; release wires first")
        return available | set(stage.outputs)

    def visit(self, name, q, available, binding, protected=frozenset()):
        """Apply one node's declared effect and return the resulting context.

        The per-node rules are the admission table of
        docs/development/program_checks.md. The rules
        that shape this traversal are:

        - A BlockCall connects its registers' correlation groups unless its
          declared interface promises independent coupling. If any input is mixed
          or unknown, an unrestricted joint unitary may move that impurity to
          any output (SWAP is one such action), so every unitary output and
          its partners lose their coherent state.
        - Branch checks both arms. Afterwards only classical values defined on
          both arms are available, and register states are joined.
        - Repeat and AdaptiveLoop check the body at most twice. The second
          pass must reproduce the first pass's live registers, states and
          correlation partition. For a Repeat with a known count, epochs then
          compose arithmetically as ``epoch + (count - 1) * increment``
          without unrolling. For an AdaptiveLoop or an unresolved count, an
          epoch that the second pass leaves unchanged is kept and any other
          becomes unresolved. A body that may run zero times is joined with
          the path that skips it.
        - MeasurementBatch requires no live outer register. Each setting is
          checked as a fresh experiment that starts empty and must release
          every register it allocates.

        Args:
            name: Definition ID to visit.
            q: Register name to ``(state, epoch, group)`` before the node.
            available: Classical values defined on every path reaching the node.
            binding: Parameter values of this context. Missing names are unknown.
            protected: Registers whose epoch an enclosing CoherentRegion must keep.

        Returns:
            ``(q, available)`` after the node. The inputs are not mutated.
        """
        # Charge context size before copying/hashing it, including cache hits.
        self.tick(1 + len(q) + len(available) + len(binding) + len(protected))
        key = (name, frozenset(q.items()), frozenset(available),
               frozenset(binding.items()), protected)
        if key in self.memo:
            saved_q, saved_c = self.memo[key]
            self.tick(len(saved_q) + len(saved_c))
            return dict(saved_q), set(saved_c)
        self.tick(len(q) + len(available))
        q, available = dict(q), set(available)
        node, context = self.nodes[name], self.expressions(binding)
        if isinstance(node, (Sequence, Parallel)):
            for child in node.children:
                q, available = self.visit(child, q, available, binding, protected)
        elif isinstance(node, Allocate):
            if node.wire in q and q[node.wire][0] != "released":
                raise ValueError("register already allocated")
            # A first allocation starts at epoch 0. Reallocation continues from
            # the released wire's epoch, so an old claim cannot match again.
            epoch = q.get(node.wire, ("released", -1))[1]
            q[node.wire] = ("zero", None if epoch is None else epoch + 1, self.canonical_group({node.wire}))
        elif isinstance(node, (Measure, Reset, Release)):
            self.live(q, node.wire)
            status = "released" if isinstance(node, Release) else "zero" if isinstance(node, Reset) else "measured"
            self.break_epoch(q, node.wire, status, protected)
            if isinstance(node, Measure):
                self.width(self.registers[node.wire].width, self.classical[node.result].width, context)
                available.add(node.result)
        elif isinstance(node, BlockCall):
            signature = self.signatures[node.signature]
            self.tick(len(signature.parameters) + 3 * len(node.ports))
            if signature.interface == "unknown":
                self.blockers.add(f"unknown block interface: {signature.name}")
            self.arguments(node.arguments, context, {p.name: p for p in signature.parameters})
            joint_state = None
            for port, mapping in zip(signature.quantum, node.ports, strict=True):
                state, epoch, group = self.live(q, mapping.wire)
                if state == "mixed" or (state == "unknown" and joint_state != "mixed"):
                    joint_state = state
                self.width(port.width, self.registers[mapping.wire].width, context)
                if state == "unknown":
                    self.blockers.add(f"unknown state effect: {mapping.wire}")
                elif (port.requires == "zero" and state != "zero") or (
                    port.requires == "coherent" and state not in {"zero", "coherent"}
                ):
                    raise ValueError("block input state requirement is not satisfied")
            unitary_wires, unknown_wires = [], []
            for port, mapping in zip(signature.quantum, node.ports, strict=True):
                state, epoch, group = q[mapping.wire]
                effect = port.ensures if signature.interface == "declared" else "unknown"
                if effect == "unitary":
                    unitary_wires.append(mapping.wire)
                    # A local unitary cannot remove uncertainty in an entangled partner.
                    q[mapping.wire] = ("coherent" if state in {"zero", "coherent", "measured"} else state,
                                       epoch, group)
                elif effect == "unknown":
                    unknown_wires.append(mapping.wire)
                elif effect != "preserve":
                    self.break_epoch(q, mapping.wire, effect, protected)
                if effect == "unknown":
                    self.blockers.add(f"unknown block effect: {signature.name}.{port.name}")
            if signature.coupling == "joint" and joint_state is not None and unitary_wires:
                # SWAP is allowed by an unrestricted joint unitary declaration:
                # purity cannot be inferred independently for each output port.
                self.invalidate_coherence(q, unitary_wires, joint_state, protected)
            if unknown_wires:
                self.invalidate_coherence(q, unknown_wires, "unknown", protected)
            if signature.coupling == "joint" or signature.interface == "unknown":
                self.connect(q, (mapping.wire for mapping in node.ports))
        elif isinstance(node, CoherentRegion):
            self.tick(len(node.claims) + len(protected))
            for claim in node.claims:
                state, epoch, _ = self.live(q, claim.wire)
                if state not in {"zero", "coherent", "unknown"}:
                    raise ValueError("stale premeasurement coherent-state claim")
                if epoch is None or state == "unknown":
                    self.blockers.add(f"unresolved coherent epoch: {claim.wire}")
                elif epoch != claim.epoch:
                    raise ValueError("stale coherent epoch claim")
            q, available = self.visit(node.body, q, available, binding,
                                      protected | frozenset(c.wire for c in node.claims))
        elif isinstance(node, ClassicalStage):
            available = self.stage(node, q, available, context)
        elif isinstance(node, Branch):
            self.classical_inputs((node.condition,), available)
            qa, ca = self.visit(node.when_true, q, available, binding, protected)
            qb, cb = self.visit(node.when_false, q, available, binding, protected)
            q = self.join(qa, qb)
            self.tick(len(ca) + len(cb))
            available = ca & cb
        elif isinstance(node, (Repeat, AdaptiveLoop)):
            count = self.integer(node.count if isinstance(node, Repeat) else node.max_rounds,
                                 context, "repeat count")
            # Even a zero body is checked once for invalid declared use; it exports no effects.
            qa, ca = self.visit(node.body, q, available, binding, protected)
            if isinstance(node, AdaptiveLoop):
                ca = self.stage(node.policy, qa, ca, context)
            if count == 1 and isinstance(node, Repeat):
                q, available = qa, ca
            elif count != 0:
                qb, cb = self.visit(node.body, qa, ca, binding, protected)
                if isinstance(node, AdaptiveLoop):
                    cb = self.stage(node.policy, qb, cb, context)
                if self.live_names(q) != self.live_names(qa) or self.live_names(qa) != self.live_names(qb):
                    raise ValueError("unsupported loop lifetime change; use an independent MeasurementBatch")
                self.tick(2 * (len(qa) + len(qb)))
                if ({k: v[0] for k, v in qa.items()} != {k: v[0] for k, v in qb.items()}
                        or self.partition(qa) != self.partition(qb)):
                    raise ValueError("unsupported nonstationary loop state effects")
                result = dict(qa)
                for reg, (state, epoch, group) in qa.items():
                    second = qb[reg][1]
                    if epoch is None or second is None or count is None or isinstance(node, AdaptiveLoop):
                        result[reg] = (state, epoch if epoch == second else None, group)
                    else:
                        result[reg] = (state, self.bounded(epoch + (count - 1) * (second - epoch)), group)
                self.tick(len(available) + len(ca) + len(cb))
                may_skip = isinstance(node, AdaptiveLoop) or (
                    count is None and not self.positive_parameter_count(node.count)
                )
                if may_skip:
                    q = self.join(q, result)
                    available &= ca & cb
                else:
                    q, available = result, ca & cb
        elif isinstance(node, MeasurementBatch):
            if self.live_names(q):
                raise ValueError("independent batch with outer live wires is unsupported; release them first")
            for setting in node.settings:
                self.tick(1 + len(binding) + len(setting.bindings) + len(node.axes))
                point = dict(binding)
                point.update(self.binding_map(setting.bindings))
                for axis in node.axes:
                    point.pop(axis.parameter, None)
                if node.repetitions is not None:
                    self.integer(node.repetitions, self.expressions(point), "repetitions")
                end_q, _ = self.visit(node.body, {}, set(), point)
                if self.live_names(end_q):
                    raise ValueError("independent experiment must explicitly release all wires")
        # Every kept cache key and output collection is charged when created.
        self.tick(len(q) + len(available))
        self.memo[key] = (dict(q), frozenset(available))
        return q, available

    def live_names(self, q):
        self.tick(len(q))
        return {name for name, (state, _, _) in q.items() if state != "released"}

    def positive_parameter_count(self, count):
        """Only the admitted direct integer ParameterRef domain proves positivity."""
        if isinstance(count, ExprRef):
            expression = self.expr[count.expression]
            if isinstance(expression, ParameterRef):
                lower = self.params[expression.parameter].lower
                return lower is not None and lower >= 1
        return False

    def break_epoch(self, q, wire, status, protected):
        """Apply a nonunitary effect to one wire.

        The effect is a measurement, reset, release or a block port that
        promises a newly prepared zero or coherent output. The wire takes
        ``status`` and leaves its correlation group as a
        singleton. Every former partner becomes ``mixed``, because a
        nonunitary effect on part of a possibly entangled state leaves the
        rest without a coherent-state guarantee. All of them advance one
        epoch, and an unresolved epoch stays unresolved. Breaking a group that
        holds a protected wire rejects.
        """
        group = q[wire][2]
        self.tick(3 * len(group) + len(protected))
        if group & protected:
            raise ValueError("coherent region epoch is broken")
        remaining = self.canonical_group(group - {wire})
        singleton = self.canonical_group({wire})
        for partner in group:
            _, epoch, _ = q[partner]
            q[partner] = (status if partner == wire else "mixed",
                          None if epoch is None else self.bounded(epoch + 1),
                          singleton if partner == wire else remaining)

    def invalidate_coherence(self, q, wires, status, protected):
        """Connect ``wires`` into one group and give every member ``status``.

        ``mixed`` records a known loss of coherence. Each member advances one
        epoch, and a protected member rejects. ``unknown`` records an unknown
        input state or an undeclared effect. Its epochs become unresolved
        (None) and nothing rejects, even for a protected member, because the
        effect may still turn out legal. The caller has already added the
        readiness blocker for that unknown.
        """
        group = self.connect(q, wires)
        self.tick(len(group) + len(protected))
        if status == "mixed" and group & protected:
            raise ValueError("joint mixed input cannot preserve the protected coherent state")
        for wire in group:
            _, epoch, _ = q[wire]
            q[wire] = (status, None if status == "unknown" or epoch is None else self.bounded(epoch + 1), group)

    def connect(self, q, wires):
        """Conservatively keep correlation unless independence is explicitly promised."""
        groups = {}
        for wire in wires:
            self.tick()
            group = q[wire][2]
            groups[id(group)] = group
        if len(groups) == 1:
            return next(iter(groups.values()))
        # q contains partitions sharing one frozenset per component. Process each
        # component once, even when many supplied wires already belong to it.
        self.tick(2 * sum(map(len, groups.values())))
        group = self.canonical_group(frozenset().union(*groups.values()))
        for wire in group:
            state, epoch, _ = q[wire]
            q[wire] = (state, epoch, group)
        return group

    def canonical_group(self, members):
        """Intern once per content so memo-key equality never rescans a group per wire."""
        if type(members) is frozenset:
            # The immutable group already exists: charge its first hash, not
            # a copy that neither this branch nor frozenset would perform.
            self.tick(len(members))
            group = members
        else:
            self.tick(2 * len(members))  # Construct and hash the new group.
            group = frozenset(members)
        return self.groups.setdefault(group, group)

    def partition(self, q):
        """One kept set per live component; never compare it once per wire."""
        self.tick(len(q))
        groups = {id(group): group for state, _, group in q.values() if state != "released"}
        self.tick(sum(map(len, groups.values())))
        return frozenset(groups.values())

    def join(self, qa, qb):
        """Merge two exits of a Branch, or a loop's executed and skipped paths.

        Both paths must agree on live registers. A state that differs becomes
        the weaker common state (coherent when both are zero or coherent,
        unknown when either is unknown, otherwise mixed), a differing epoch
        becomes unresolved, and correlation groups from both paths merge
        transitively.
        """
        if self.live_names(qa) != self.live_names(qb):
            raise ValueError("branch/loop paths have incompatible register lifetimes")
        pa, pb = self.partition(qa), self.partition(qb)
        self.tick(3 * (len(qa) + len(qb)))
        result = {}
        for name in qa.keys() | qb.keys():
            a, b = qa.get(name), qb.get(name)
            if a is None:
                a = ("released", -1, self.canonical_group({name}))
            if b is None:
                b = ("released", -1, self.canonical_group({name}))
            state = a[0] if a[0] == b[0] else "coherent" if {a[0], b[0]} <= {"zero", "coherent"} else "mixed"
            if "unknown" in (a[0], b[0]):
                state = "unknown"
            result[name] = (state, a[1] if a[1] == b[1] else None, a[2])
        if pa == pb:
            return result
        # Different partitions form a bipartite incidence graph: each live wire
        # belongs to at most one component per arm. Traverse every incidence once.
        self.tick(sum(map(len, pa)) + sum(map(len, pb)))
        groups = pa | pb
        by_wire = {}
        for group in groups:
            self.tick(len(group))
            for wire in group:
                by_wire.setdefault(wire, []).append(group)
        seen = set()
        for start in groups:
            if id(start) in seen:
                continue
            pending, component = [start], set()
            while pending:
                group = pending.pop()
                if id(group) in seen:
                    continue
                seen.add(id(group))
                self.tick(3 * len(group))
                for wire in group:
                    if wire not in component:
                        component.add(wire)
                        pending.extend(by_wire[wire])
            self.tick(2 * len(component))
            merged = self.canonical_group(component)
            for wire in component:
                state, epoch, _ = result[wire]
                result[wire] = (state, epoch, merged)
        return result

    def validate_widths(self, context):
        """Read every declared register, classical and port width in one binding context.

        An unknown width becomes a blocker. A negative width, or a classical
        bit width below one, rejects.
        """
        for reg in self.registers.values():
            self.integer(reg.width, context, "register width")
        for value in self.classical.values():
            if value.width is not None:
                self.integer(value.width, context, "classical width", positive=True)
        for signature in self.signatures.values():
            for port in signature.quantum:
                self.integer(port.width, context, "port width")

    def check(self):
        """Admit the whole Program at its own bindings, starting from an empty root context."""
        self.setup()
        binding = self.binding_map(self.p.bindings)
        self.expressions(binding)
        self.visit(self.p.root, {}, set(), binding)
        return Readiness(blockers=tuple(sorted(self.blockers)),
                         expression_evaluations=self.evaluations,
                         lifecycle_steps=self.steps - self.evaluations)

    def admitted(self):
        """Index an admitted Program for a consumer and return its stored Readiness.

        A Program runs ``check`` once, at construction, and stores the
        Readiness (``Program.check_readiness``). A consumer that needs the
        tables, the root binding context and the work counter runs
        ``setup``, the root binding and its expressions, exactly as ``check``
        does, skips the lifecycle visit whose result is stored, and continues
        from the work that the full check measured: ``steps`` and
        ``evaluations`` are set to that check's totals and ``blockers`` to its
        blockers. The root binding context is evaluated here as ``check``
        evaluates it. Other binding contexts that the skipped visit would
        have evaluated, such as a setting's bindings, are evaluated on first
        use and charged one expression pass each, so a consumer that reads
        such a context charges that pass in addition to the stored total.
        """
        readiness = self.p.check_readiness()
        self.setup()
        self.expressions(self.binding_map(self.p.bindings))
        self.evaluations = readiness.expression_evaluations
        self.steps = readiness.expression_evaluations + readiness.lifecycle_steps
        self.blockers = set(readiness.blockers)
        return readiness
