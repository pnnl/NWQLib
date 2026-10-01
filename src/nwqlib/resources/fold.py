"""Bounded selected-construction fold. No payload access, SDK or recipe selection.

Planning must report what a selected construction costs without building,
transpiling or simulating it (docs/FRAMEWORK.md, "Resource Estimation Before
Execution"). This module walks the same Program that logical lowering
consumes and combines the laws declared on its selected definitions into
located quantities, each with its interpretation and original evidence. The
contract for each quantity is docs/resources.md, and docs/development/execution.md
("Resource estimate bookkeeping") describes how this fold combines the laws.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from nwqlib.core.records import Float64, Rational, Scope, Source, Symbol, Unit
from nwqlib.evidence import Evidence, Fact
from nwqlib.blocks.records import SelectedConstruction
from nwqlib.ir import (
    AdaptiveLoop, Allocate, BlockCall, Branch, ClassicalStage, CoherentRegion,
    Measure, MeasurementBatch, Parallel, Release, Repeat, Reset, Sequence,
)
from nwqlib.ir.expressions import Binary, Constant, ExprRef, Expression, ParameterRef, number
from nwqlib.ir.validation import _Admission
from .records import (CLIFFORD_ANGLES, COUNTS, DEPTHS, FOLD_WORK_PER_ADMISSION_STEP, GATES, T_ANGLES,
                      ResourceContext, ResourceQuantity, WorkloadEstimate)

FOLD_SOURCE = Source(name="selected_resource_fold", version="2", domain="selected logical construction",
                     reference="nwqlib.resources.fold; serial sum, explicit parallel max, lifetime peaks")


@dataclass(frozen=True)
class _Value:
    """One partial cost with its meaning and provenance.

    ``value`` is a concrete number, a string naming an entry of the fold's
    local expression table (an unresolved symbolic cost), or None for an
    unknown cost. Unknown is kept distinct from zero because a missing law,
    workspace or acquisition kind says nothing about the cost. Treating it as
    zero would make an incomplete estimate look cheaper and complete.

    ``interpretation`` orders as exact < upper_bound < conditional < estimate.
    A combination takes the weakest meaning among its operands. ``sources``
    are all evidence identities attached to the value. ``derivation_sources``
    are the ones that determined it, which excludes evidence used only to
    check a known recipe against a declared law. ``requires_serial`` marks a
    peak that holds only when independent acquisitions run one at a time.
    """

    value: int | float | str | None = 0
    interpretation: str = "exact"
    sources: frozenset = frozenset()
    assumptions: frozenset = frozenset()
    reasons: frozenset = frozenset()
    derivation_sources: frozenset | None = None
    requires_serial: bool = False

    @property
    def derived_from(self):
        return self.sources if self.derivation_sources is None else self.derivation_sources


def _unknown(reason):
    return _Value(None, reasons=frozenset((reason,)))


# The exact, source-free zero. It is the only additive identity. An unknown,
# estimated or evidence-bearing zero still participates in combination.
_EMPTY = _Value()


class _Fold:
    """Fold one SelectedConstruction into a WorkloadEstimate without building circuits.

    The fold is an interpretation of the admitted Program in which every node
    returns a cost per count metric and a peak per located footprint. Its
    rules are:

    - Composition. Sequence and ordered primitive recipes add work and depth.
      Parallel adds work and takes the maximum depth. Repeat multiplies the
      body cost by its count without unrolling. A Branch takes the per-metric
      maximum of its arms as an upper bound, with no probability model. An
      adaptive body without a declared per-round envelope has unknown costs.
    - Laws. A BlockCall costs what its SelectedDefinition declares: an exact
      primitive recipe, applicable ResourceLaw summaries, or one of the
      registered native CX envelopes. A call with none of these has unknown
      gate metrics, never zero. When a recipe and a law describe the same
      metric, the recipe must conform to the law.
    - Footprints. Allocate and Release move a running quantum width per
      (role, location). Workspace declarations add bytes per location. Peaks
      are maxima over the traversal, and a batch peak that assumes one job at
      a time is marked as requiring a serial schedule.
    - Acquisition. Terminal MeasurementBatch settings contribute shots or
      exact evaluations according to their declared observation kind.
      Unknown kinds or repetitions stay unknown.

    The fold first runs the full Program admission in its own admission
    instance and then continues on that instance, reusing its node tables,
    exact integer and depth limits and work counter. Its work ceiling is
    ``max(max_steps, FOLD_WORK_PER_ADMISSION_STEP * a)``, where ``a`` is the
    admission work just measured (ENGINEERING_CONSTANTS.md, "Shared Program
    admission limits"). A fold proportional to its Program is admitted, and
    one that grows faster is rejected. Node visits are memoized per
    (node, binding, live footprint, classical availability) and law lookups
    per (selected identity, actual cost parameters).

    Consumers are ``nwqlib.estimate`` and ``compare``, ``search.scan`` and
    the device assessment in ``backends.assessment``.

    Attributes:
        construction: The SelectedConstruction being folded.
        context: Its ResourceContext.
        admission: The Program's admission instance, reused for tables, limits and the work counter.
        readiness: Readiness returned by that admission. Its blockers become assumptions of the estimate.
        program: The folded Program.
        selected: Signature name to SelectedDefinition.
        kernels: Host kernel name to SelectedKernel.
        identities: Signature name to selected content identity.
        evidence: Output source table, evidence identity to Evidence.
        expr: Output expression table, local ID to Expression.
        expr_keys: Structural key to local ID, so equal expressions are stored once.
        contexts: Program expression values per distinct binding.
        memo: ``visit`` result per distinct node context.
        leaves: ``leaf`` result per (selected identity, actual cost parameters).
        quantum_footprints: Cache from a running width aggregate of ``update_live`` to its width quantities per (role, location) and total logical width per location, filled by ``quantum_snapshot``.
        zero_costs: Shared exact-zero cost map, the additive identity element of ``merge_cost``.
        defined: Potentially reached selected and kernel identities.
        terminal_points: Distinct terminal batch points and their axis cardinalities.
        unique_unavailable: Reasons that make the distinct-point count unknown.
        unique_conditional: Whether unresolved control flow makes that count an upper bound.
        root_populations: Root batch declaration and repetition totals.
        fold_evidence: The fold's own proved-relation Evidence record.
        fold_id: Identity of ``fold_evidence``, attached to recipe counts.
    """

    def __init__(self, construction, context):
        """Admit the Program, set the fold's work ceiling and create empty tables.

        The full Program admission runs first, and its measured work sets
        ``step_limit``. The selected definitions, kernels and context are
        then walked with the same charge-before-copy rule, and every integer
        in them is checked against ``max_integer_bits``, before their
        identities are hashed or the fold traverses them.
        """
        self.construction = construction
        self.context = context
        self.admission = _Admission(construction.program)
        self.readiness = self.admission.admitted()
        # The fold work limit is proportional to the admission work just measured,
        # and never below the Program's own step limit.
        self.admission.step_limit = max(construction.program.limits.max_steps,
                                        FOLD_WORK_PER_ADMISSION_STEP * self.admission.steps)
        self.program = construction.program
        self.selected = {item.signature.name: item for item in construction.selections}
        self.kernels = {item.name: item for item in construction.kernels}
        self.identities = {}
        self.evidence = {}
        self.expr = {}
        self.expr_keys = {}
        self.contexts = {}
        self.memo = {}
        self.leaves = {}
        self.quantum_footprints = {}
        self.zero_costs = MappingProxyType(dict.fromkeys(COUNTS, _EMPTY))
        self.defined = set()
        self.terminal_points = {}
        self.unique_unavailable = set()
        self.unique_conditional = False
        self.root_populations = {}
        # Charge selected declarations before hashing or traversing them.
        from nwqlib.ir.expressions import walk_kept
        for item in walk_kept((construction.selections, construction.kernels, context), self.admission.tick):
            if type(item) is int:
                self.admission.bounded(item)
        for item in construction.selections:
            self.identities[item.signature.name] = item.content_id
        self.fold_evidence = Evidence(kind="proved_relation", source=FOLD_SOURCE)
        self.fold_id = self.source(self.fold_evidence)

    def source(self, evidence):
        """Store one evidence record in the shared source table and return its identity."""
        self.admission.tick(1 + len(evidence.work))
        identity = evidence.content_id
        self.evidence[identity] = evidence
        return identity

    def expression(self, node):
        """Intern a symbolic cost node in the output expression table and return its local ID.

        Structurally equal nodes share one entry, so a symbolic cost reused
        across the graph is stored once and JSON never expands it.
        """
        self.admission.tick()
        if isinstance(node, Constant):
            key = ("constant", number(node.value))
        elif isinstance(node, ParameterRef):
            key = ("parameter", node.parameter)
        else:
            key = (node.op, node.left, node.right)
        if key not in self.expr_keys:
            self.admission.tick(3)
            name = f"r{len(self.expr)}"
            self.expr_keys[key] = name
            self.expr[name] = Expression(id=name, value=node)
        return self.expr_keys[key]

    def operand(self, value):
        """Return the expression ID of a symbolic value, interning a number as a Constant."""
        return value if isinstance(value, str) else self.expression(Constant(value=Float64(value=value) if type(value) is float else value))

    def arithmetic(self, op, left, right):
        """Apply one exact operation to numbers, or build a symbolic node if either is symbolic.

        Identities with 0 and 1 are applied before a node is built, so
        combining a symbolic cost with 0 or 1 creates no node. Integer
        products check their bit length first, as in Program admission.
        """
        self.admission.tick()
        if op == "multiply" and (left == 0 or right == 0):
            return 0
        if op == "multiply" and (left == 1 or right == 1):
            return right if left == 1 else left
        if op == "add" and (left == 0 or right == 0):
            return right if left == 0 else left
        if op in {"min", "max"} and left == right:
            return left
        if isinstance(left, str) or isinstance(right, str):
            return self.expression(Binary(op=op, left=self.operand(left), right=self.operand(right)))
        if op == "multiply":
            if (type(left) is int and type(right) is int and left and right
                    and left.bit_length() + right.bit_length() - 1 > self.admission.limits.max_integer_bits):
                raise ValueError("resource integer growth exceeds max_integer_bits")
            value = left * right
        elif op == "add":
            value = left + right
        elif op == "ceildiv":
            if right <= 0:
                raise ValueError("resource ceildiv denominator must be positive")
            value = -(-left // right)
        elif op == "min":
            value = min(left, right)
        elif op == "max":
            value = max(left, right)
        elif op == "eq":
            value = int(left == right)
        elif op == "lt":
            value = int(left < right)
        else:
            value = int(left <= right)
        return self.admission.bounded(value)

    def combine(self, left, right, op="add", *, bound=False):
        """Combine two partial costs, keeping unknowns, weakest meaning and all evidence.

        An unknown operand makes the result unknown, with the union of reasons.
        The one exception is multiplication by an exact zero. A Repeat or
        repetition count that is exactly zero removes the body's work even when
        that work is unknown. ``bound=True`` marks a maximum taken over
        exclusive alternatives, so the result is at best an upper bound.
        """
        if op == "add" and not bound:
            # Only the exact source-free identity can disappear. In particular
            # an unknown, estimated or source-bearing zero is kept below.
            if right == _EMPTY:
                self.admission.tick()
                return left
            if left == _EMPTY:
                self.admission.tick()
                return right
        self.admission.tick(1 + len(left.sources) + len(right.sources) + len(left.assumptions)
                    + len(right.assumptions) + len(left.reasons) + len(right.reasons)
                    + len(left.derived_from) + len(right.derived_from))
        if op == "max" and not bound and left == right:
            return left
        sources = left.sources | right.sources
        assumptions = left.assumptions | right.assumptions
        derivation = left.derived_from | right.derived_from
        serial = left.requires_serial or right.requires_serial
        if op == "multiply":
            zero = next((operand for operand in (left, right)
                         if operand.value == 0 and operand.interpretation == "exact"), None)
            if zero is not None:
                return _Value(0, sources=sources, assumptions=zero.assumptions,
                              derivation_sources=zero.derived_from, requires_serial=zero.requires_serial)
        reasons = left.reasons | right.reasons
        if left.value is None or right.value is None:
            return _Value(None, sources=sources, assumptions=assumptions, reasons=reasons,
                          derivation_sources=derivation, requires_serial=serial)
        kinds = {left.interpretation, right.interpretation}
        kind = ("estimate" if "estimate" in kinds else "conditional" if "conditional" in kinds
                else "upper_bound" if bound or "upper_bound" in kinds else "exact")
        return _Value(self.arithmetic(op, left.value, right.value), kind, sources=sources, assumptions=assumptions,
                      derivation_sources=derivation, requires_serial=serial)

    def values(self, bindings):
        """Evaluate Program expressions for one binding, keeping unbound parameters symbolic.

        This differs from admission, where an unbound parameter is unknown.
        Here it becomes a local ParameterRef so the published cost can still
        be evaluated when the caller later supplies a value.
        """
        key = tuple(sorted(bindings.items()))
        self.admission.tick(1 + len(bindings))
        if key not in self.contexts:
            result = {}
            # Reuse admitted expression order; preserve unresolved refs without expansion.
            for name in self.admission.order:
                self.admission.tick()
                node = self.admission.expr[name]
                if isinstance(node, Constant):
                    value = number(node.value)
                elif isinstance(node, ParameterRef):
                    value = bindings.get(node.parameter)
                    if value is None:
                        value = self.expression(node)
                else:
                    value = self.arithmetic(node.op, result[node.left], result[node.right])
                result[name] = value
            self.contexts[key] = result
        return self.contexts[key]

    def scalar(self, value, bindings):
        return self.values(bindings)[value.expression] if isinstance(value, ExprRef) else value

    def primitive(self, selected):
        """Count an exact ordered primitive recipe in the context's metric basis.

        A selected outer control adds one wire to every primitive, including
        the global phase, which becomes an observable one-qubit phase. A phase
        is Clifford only at a stored multiple of pi/2 in [-2pi, 2pi] and a T
        gate only at exactly +-pi/4, with no tolerance, so a nearby angle is
        never renamed. Any other non-Clifford operation leaves the T count
        unknown until a precision and synthesis law are selected. In the
        ``cx`` and ``clifford_t`` bases a recipe element outside the basis
        makes the gate metrics unknown (the arbitrary-rotation count is
        kept), because no synthesis is invented here. Without a recipe every
        gate metric is unknown.
        """
        if selected.decomposition is None:
            return {metric: _unknown(f"{selected.signature.name}: no selected {metric} law/decomposition") for metric in GATES}
        result = dict.fromkeys(GATES, _Value())
        counts = dict.fromkeys(GATES, 0)
        unknown_t = False
        basis_supported = True
        for operation in selected.decomposition:
            self.admission.tick()
            gate, width = operation.gate, len(operation.qubits)
            phase = gate == "phase"
            if phase and not selected.controlled:
                counts["global_phases"] += 1
                continue
            width += int(selected.controlled)
            counts["operations"] += 1
            counts["logical_depth"] += 1
            counts["single_qubit"] += width == 1
            counts["two_qubit"] += width == 2
            counts["controlled"] += selected.controlled or gate in {"mc_z", "cx"} and width > 1
            # Exact stored conventional values only (records.CLIFFORD_ANGLES):
            # no modulo rounding or tolerance snapping can turn a nearby
            # rotation into a named gate.
            clifford_phase = phase and operation.angle in CLIFFORD_ANGLES
            t_phase = phase and operation.angle in T_ANGLES
            clifford = (gate in {"x", "z", "mc_z"} and width <= 2 or gate == "h" and width == 1
                        or gate == "sdg" and width == 1 or gate == "cx" and width == 2
                        or clifford_phase)
            counts["clifford"] += clifford
            counts["cx"] += gate in {"x", "cx"} and width == 2
            counts["toffoli"] += gate == "cx" and width == 3
            counts["ccz"] += gate == "mc_z" and width == 3
            counts["arbitrary_rotations"] += phase and not (clifford or t_phase)
            counts["t"] += t_phase
            counts["t_depth"] += t_phase
            if not clifford:
                counts["non_clifford_depth"] += 1
                unknown_t |= not t_phase
            if self.context.basis == "clifford_t" and not (clifford or t_phase):
                basis_supported = False
            if self.context.basis == "cx" and not (width == 1 or gate in {"x", "cx"} and width == 2):
                basis_supported = False
        if self.context.basis in {"clifford_t", "cx"}:
            # No implicit synthesis of controlled H, CCZ, or other nonbasis gates.
            if not basis_supported:
                result = {metric: _unknown(f"{selected.signature.name}: selected recipe needs a {self.context.basis} synthesis law") for metric in GATES}
                if counts["arbitrary_rotations"]:
                    result["arbitrary_rotations"] = _Value(counts["arbitrary_rotations"], sources=frozenset((self.fold_id,)))
                return result
        for metric, value in counts.items():
            result[metric] = _Value(value, sources=frozenset((self.fold_id,)))
        if unknown_t:
            for metric in ("t", "t_depth"):
                result[metric] = _unknown("rotation precision/selected synthesis law unavailable")
        return result

    def native_cx(self, selected, bindings):
        """Evaluate a registered native CX envelope for one call, or return None.

        Only the four laws owned by ``_preparation_laws`` and
        ``blocks.selection`` are recognized, and only when both the cost-law
        Source and the implementation Source equal their registered
        identities. A record that merely reuses a law name therefore cannot
        borrow a bound derived for another construction. The result is an
        upper bound on CX slots before cancellation and routing.
        """
        from nwqlib.blocks.selection import _source, pauli_readout_cx_bound, signed_pauli_cx_bound
        from nwqlib._preparation_laws import direct_preparation_cx_bound, direct_preparation_controlled_cx_bound
        if selected.cost_law is None or self.context.basis != "cx":
            return None
        names = {"direct_preparation_cx_bound", "direct_preparation_controlled_cx_bound",
                 "signed_pauli_cx_bound", "pauli_readout_cx_bound"}
        name = selected.cost_law.name
        if name not in names or selected.cost_law != _source(name):
            return None
        implementation = ("preparation.native" if name.startswith("direct") else
                          "pauli.signed_select" if name == "signed_pauli_cx_bound" else "pauli.readout_basis")
        if selected.implementation != _source(implementation):
            return None
        expected = {"num_qubits", "multiplicity"} if name.startswith("direct") else {"index_qubits", "system_qubits"}
        if set(bindings) != expected or any(type(value) is not int or value < 0 for value in bindings.values()):
            return None
        # Reject huge shifts before calling the established law owner.
        exponent = bindings.get("num_qubits", bindings.get("index_qubits", 0))
        factor = bindings.get("multiplicity", bindings.get("system_qubits", 0))
        # All four registered envelopes are <= 32*max(1,factor)*2**exponent:
        # controlled SELECT has leading coefficient 24*q+8 <= 32*max(1,q),
        # controlled direct PREP <= 16*2**q (including its uniform fast path).
        # Reserve the upper bound's bit length before any native power or product.
        if exponent + max(1, factor).bit_length() + 5 > self.admission.limits.max_integer_bits:
            raise ValueError("CX law integer growth exceeds max_integer_bits")
        if name.startswith("direct"):
            if selected.controlled != (name == "direct_preparation_controlled_cx_bound"):
                return None
            law = direct_preparation_controlled_cx_bound if selected.controlled else direct_preparation_cx_bound
            value = self.arithmetic("multiply", bindings["multiplicity"], law(bindings["num_qubits"]))
        else:
            law = signed_pauli_cx_bound if name == "signed_pauli_cx_bound" else pauli_readout_cx_bound
            value = self.admission.bounded(law(bindings["index_qubits"], bindings["system_qubits"], controlled=selected.controlled))
        evidence = Evidence(kind="certified_bound", source=selected.cost_law)
        return _Value(value, "upper_bound", sources=frozenset((self.source(evidence),)),
                      assumptions=frozenset((selected.cost_context, "native CX slot envelope before cancellation/routing")))

    def leaf(self, call, bindings):
        """Cost one BlockCall from its selected laws, recipe and native envelope.

        A ResourceLaw applies only when its basis, control, adjoint,
        precision and synthesis equal this context and selection, and its
        fixed bindings equal the call's actual cost parameters (names listed
        in ``unbound_parameters`` may take any admitted value). Two applicable
        laws for one metric are ambiguous and reject. An applicable law owns
        its metric. When an exact recipe also fixes that metric, the recipe
        value is kept and must equal an exact law or stay within an upper
        bound, so a declared summary cannot silently contradict the recipe it
        describes. Laws backed by observed, empirically predicted or
        numerically estimated evidence are reported as estimates whatever
        interpretation they declare.
        """
        selected = self.selected[call.signature]
        actual = {item.parameter: number(item.value) for item in selected.cost_parameters}
        if len(actual) != len(selected.cost_parameters):
            raise ValueError("duplicate selected cost parameter")
        for argument in call.arguments:
            value = self.scalar(argument.value, bindings)
            if argument.parameter in actual and actual[argument.parameter] != value:
                raise ValueError("call argument conflicts with selected cost parameter")
            actual[argument.parameter] = value
        # Operation-local context fixes basis, precision and synthesis. Selected identity
        # fixes control, adjoint, base choice, laws and workspace declarations.
        key = (self.identities[call.signature], tuple(sorted(actual.items())))
        self.admission.tick(1 + len(actual))
        if key in self.leaves:
            return self.leaves[key]
        summaries, conformance = {}, {}
        for law in selected.resource_laws:
            self.admission.tick(1 + len(law.bindings) + len(law.unbound_parameters) + len(actual))
            fixed = {b.parameter: number(b.value) for b in law.bindings}
            covered = set(law.unbound_parameters)
            if (law.basis != self.context.basis or law.controlled != selected.controlled
                    or law.adjoint != selected.adjoint or law.precision != self.context.precision
                    or law.synthesis != self.context.synthesis
                    or not covered <= actual.keys()
                    or fixed != {key: value for key, value in actual.items() if key not in covered}):
                continue
            if law.metric not in GATES and law.metric != "classical_work":
                raise ValueError("selected per-call summary law must describe gates/depth or classical_work")
            if law.metric in summaries:
                raise ValueError("ambiguous applicable selected resource laws")
            conformance[law.metric] = law.interpretation
            interpretation = ("estimate" if law.evidence.kind in {"observed", "empirical_prediction", "numerical_estimate"}
                              else law.interpretation)
            summaries[law.metric] = _Value(number(law.value), interpretation,
                sources=frozenset((self.source(law.evidence),)), assumptions=frozenset(law.assumptions + (selected.cost_context,)))
        # Applicable summaries own their metrics. Kept literal facts are
        # checked once in this bounded context even when every metric has a law:
        # adding an unrelated law cannot disable conformance for a known recipe.
        result = self.primitive(selected)
        native = self.native_cx(selected, actual)
        if native is not None:
            summaries.setdefault("cx", native)
        for metric, value in summaries.items():
            recipe = result.get(metric)
            if recipe is not None and recipe.value is not None and recipe.interpretation == "exact":
                relation = conformance.get(metric, value.interpretation)
                if relation == "exact" and recipe.value != value.value:
                    raise ValueError("selected summary conflicts with exact primitive inventory")
                if relation == "upper_bound" and recipe.value > value.value:
                    raise ValueError("selected summary bound is below exact primitive inventory")
                result[metric] = _Value(recipe.value, "exact", sources=recipe.sources | value.sources,
                                       assumptions=recipe.assumptions | value.assumptions, derivation_sources=recipe.derived_from)
            else:
                result[metric] = value
        result["calls"] = _Value(1)
        from nwqlib.blocks.selection import _source
        if (selected.cost_law in (_source("direct_preparation_cx_bound"), _source("direct_preparation_controlled_cx_bound"))
                and set(actual) == {"num_qubits", "multiplicity"}
                and all(type(value) is int and value >= 0 for value in actual.values())):
            result["preparation_components"] = _Value(actual["multiplicity"], sources=frozenset((
                self.source(Evidence(kind="external_specification", source=selected.cost_law)),)),
                assumptions=frozenset((selected.cost_context, "selected native preparation component invocations, not synthesized gate count")))
        for metric in COUNTS:
            result.setdefault(metric, _Value())
        rotations = result["arbitrary_rotations"].value
        if (self.context.basis == "clifford_t" and type(rotations) in {int, float} and rotations > 0
                and (self.context.precision is None or self.context.synthesis is None)):
            for metric in ("t", "t_depth"):
                result[metric] = _unknown("arbitrary rotations require explicit precision and selected synthesis law")
        self.leaves[key] = result
        self.admission.tick(len(result))
        return result

    def merge_cost(self, left, right, *, parallel=False, branch=False):
        """Compose two per-metric cost maps: serial sum, parallel depth maximum or branch maximum."""
        # Exact empty work is the additive identity. Allocation/release changes
        # lifetimes, so its zero gate work need not rebuild every cost/evidence
        # union. Unknown costs, source-bearing zeros and branch bounds do not
        # enter this identity path.
        if not branch:
            if right is self.zero_costs:
                return left
            if left is self.zero_costs:
                return right
        result = {}
        for metric in COUNTS:
            op = "max" if branch or parallel and metric in DEPTHS else "add"
            result[metric] = self.combine(left[metric], right[metric], op, bound=branch)
        return result

    def scale_cost(self, costs, multiplicity, *, bound=False, assumption=None):
        """Multiply every metric by a count without unrolling the body it repeats."""
        factor = _Value(multiplicity, "upper_bound" if bound else "exact",
                        assumptions=frozenset((assumption,)) if assumption else frozenset())
        return {metric: self.combine(value, factor, "multiply") for metric, value in costs.items()}

    def update_live(self, live, wire, bindings, direction):
        """Update a width aggregate, not a second copy of admitted wire state.

        Admission already verifies exact wire membership and branch lifetimes.
        Numeric widths add/remove directly. An unresolved width is a formal
        expression with a multiplicity, so release needs no symbolic subtraction
        or cancellation of evidence. Register widths carry no external sources.
        """
        register = self.admission.registers[wire]
        width = self.scalar(register.width, bindings)
        groups = dict(live)
        key = (register.role, register.location)
        numeric, terms = groups.get(key, (0, ()))
        symbols = dict(terms)
        self.admission.tick(1 + len(groups) + len(symbols))
        if isinstance(width, str):
            multiplicity = self.admission.bounded(symbols.get(width, 0) + direction)
            if multiplicity < 0:
                raise ValueError("admitted release has no matching symbolic width")
            if multiplicity:
                symbols[width] = multiplicity
            else:
                symbols.pop(width)
        else:
            numeric = self.admission.bounded(numeric + direction * width)
            if numeric < 0:
                raise ValueError("admitted release exceeds the live width")
        if numeric or symbols:
            groups[key] = (numeric, tuple(sorted(symbols.items())))
        else:
            groups.pop(key, None)
        return tuple(sorted(groups.items()))

    def quantum_snapshot(self, live):
        """Expand a live-width aggregate into located width quantities, cached per aggregate."""
        if live not in self.quantum_footprints:
            result = {}
            for (role, location), (numeric, terms) in live:
                self.admission.tick(1 + len(terms))
                width = _Value(numeric)
                for expression, multiplicity in terms:
                    width = self.combine(width, self.combine(_Value(expression), _Value(multiplicity), "multiply"))
                key = ("logical_width", location)
                result[key] = self.combine(result.get(key, _Value()), width)
                result[(role, location)] = width
                for quantum_role in ("system", "clean_ancilla", "dirty_ancilla"):
                    result.setdefault((quantum_role, location), _Value())
            self.quantum_footprints[live] = result
        return self.quantum_footprints[live]

    def snapshot(self, live, classical, bindings, workspace=()):
        """Current live footprint; no byte conversion from logical quantum width."""
        result = dict(self.quantum_snapshot(live))
        def add(key, value):
            result[key] = self.combine(result.get(key, _Value()), value)
        for name, certain in classical:
            self.admission.tick()
            value = self.admission.classical[name]
            kind = "exact" if certain else "upper_bound"
            add(("classical_registers", "host"), _Value(1, kind))
            if value.width is not None:
                add(("classical_bits", "host"), _Value(self.scalar(value.width, bindings), kind))
        for item in self.context.resident + tuple(workspace):
            self.admission.tick()
            value = _unknown(f"{item.location}: unknown {item.purpose} workspace "
                f"({item.source.name}: {item.source.domain})") if item.bytes is None else _Value(
                item.bytes, sources=frozenset((self.source(Evidence(kind="external_specification", source=item.source)),)))
            add(("memory", item.location), value)
            if item.bytes is not None:
                add(("known_memory", item.location), value)
                metric = {"input": "input_bytes", "analysis": "analysis_bytes", "io": "io_bytes",
                          "materialization": "materialization_bytes", "stored": "stored_bytes"}.get(item.purpose)
                if metric:
                    add((metric, item.location), value)
        return result

    def peaks(self, left, right, *, bound=False, assumption=None):
        """Take the per-location maximum of two footprints.

        With ``bound=True`` (exclusive branch paths) a differing entry becomes
        an upper bound. An entry equal and exact on both sides stays exact.
        """
        self.admission.tick(len(left) + len(right))
        result = {}
        for key in left.keys() | right.keys():
            a, b = left.get(key, _Value()), right.get(key, _Value())
            invariant = a.value == b.value and a.interpretation == b.interpretation == "exact"
            value = self.combine(a, b, "max", bound=bound and not invariant)
            if assumption and not invariant:
                value = replace(value, assumptions=value.assumptions | {assumption})
            result[key] = value
        return result

    def batch_peaks(self, peaks, bindings):
        """Per-job reuse needs a serial schedule; invariant residents do not."""
        resident = self.snapshot((), frozenset(), bindings)
        result = {}
        for key, value in peaks.items():
            self.admission.tick()
            base = resident.get(key, _Value())
            invariant = value.value == base.value and value.interpretation == base.interpretation == "exact"
            result[key] = value if invariant else replace(value, requires_serial=True)
        return result

    def call_workspace(self, call):
        """Return a call's declared per-invocation workspace, or one unknown entry if none is declared."""
        selected = self.selected[call.signature]
        from .records import Workspace
        workspace = selected.workspace
        # Absence of a memory declaration cannot certify zero native workspace.
        if not workspace:
            location = self.admission.registers[call.ports[0].wire].location if call.ports else "host"
            workspace = (Workspace(location=location, purpose="workspace", bytes=None, source=selected.implementation),)
        return workspace

    def define_classical(self, classical, outputs):
        """Return the (name, certain) pair set with ``outputs`` added as certainly defined."""
        self.admission.tick(len(classical) + len(outputs))
        values = dict(classical)
        values.update(dict.fromkeys(outputs, True))
        return frozenset(values.items())

    def join_classical(self, left, right):
        """Keep may-live storage as well as guaranteed classical availability.

        A value defined on either path stays in the set, so its storage
        counts in the peak. It is certain only when both paths define it
        with certainty.
        """
        self.admission.tick(len(left) + len(right))
        a, b = dict(left), dict(right)
        return frozenset((name, a.get(name, False) and b.get(name, False)) for name in a.keys() | b.keys())

    def visit(self, name, bindings, live=(), classical=frozenset()):
        """Fold one node in one context.

        Args:
            name: Definition ID of the node.
            bindings: Parameter values in effect. Unbound parameters stay symbolic.
            live: Running quantum width per (role, location), as kept by update_live.
            classical: Pairs (name, certain) of defined classical values. ``certain``
                is False for a value that exists on only some paths, which
                still occupies storage in the peak.

        Returns:
            ``(costs, peaks, live, classical)``: per-metric work of the node,
            per-(metric, location) maximum footprint while it runs, and the
            footprint and classical values it leaves behind.

        MeasurementBatch settings are independent experiments. Their summed
        depth is a serial envelope over jobs, not a per-circuit depth, and a
        peak that reuses space across settings requires a serial schedule. The
        batch's observation kind decides which acquisition population it
        contributes: ``counts`` adds shots, ``pauli_expectation`` and
        ``probabilities`` add exact evaluations, and ``estimated_observable``
        leaves shots unknown because the provider chooses them. A known kind
        proves the other population zero even when repetitions are unknown.
        """
        self.admission.tick(1 + len(bindings) + len(live) + len(classical))
        key = (name, tuple(sorted(bindings.items())), live, classical)
        if key in self.memo:
            return self.memo[key]
        node = self.admission.nodes[name]
        costs = self.zero_costs
        peaks = self.snapshot(live, classical, bindings)
        if isinstance(node, Sequence):
            for child in node.children:
                part, child_peaks, live, classical = self.visit(child, bindings, live, classical)
                costs = self.merge_cost(costs, part)
                peaks = self.peaks(peaks, child_peaks)
        elif isinstance(node, Parallel):
            workspace = []
            extra_qubits = {}
            for child in node.children:
                part, child_peaks, _, _ = self.visit(child, bindings, live, classical)
                costs = self.merge_cost(costs, part, parallel=True)
                peaks = self.peaks(peaks, child_peaks)
                workspace.extend(self.call_workspace(self.admission.nodes[child]))
                call = self.admission.nodes[child]
                extra = self.selected[call.signature].semantics.workspace
                if extra != 0:
                    location = self.admission.registers[call.ports[0].wire].location if call.ports else "logical_device"
                    value = _Value(extra) if extra is not None else _unknown("unknown concurrent quantum workspace")
                    extra_qubits[location] = self.combine(extra_qubits.get(location, _Value()), value)
            peaks = self.peaks(peaks, self.snapshot(live, classical, bindings, workspace))
            baseline = self.snapshot(live, classical, bindings)
            for location, value in extra_qubits.items():
                peaks[("logical_width", location)] = self.combine(baseline.get(("logical_width", location), _Value()), value)
        elif isinstance(node, Allocate):
            live = self.update_live(live, node.wire, bindings, 1)
        elif isinstance(node, Release):
            live = self.update_live(live, node.wire, bindings, -1)
        elif isinstance(node, BlockCall):
            costs = self.leaf(node, bindings)
            self.defined.add(self.identities[node.signature])
            peaks = self.peaks(peaks, self.snapshot(live, classical, bindings, self.call_workspace(node)))
            extra = self.selected[node.signature].semantics.workspace
            if extra != 0:
                location = self.admission.registers[node.ports[0].wire].location if node.ports else "logical_device"
                value = _Value(extra) if extra is not None else _unknown("unknown extra quantum workspace")
                for metric in ("logical_width", "clean_ancilla", "dirty_ancilla"):
                    key_peak = (metric, location)
                    # An unclassified known workspace cannot be called clean or dirty.
                    if metric != "logical_width":
                        value = _unknown("extra workspace clean/dirty role not declared")
                    peaks[key_peak] = self.combine(peaks.get(key_peak, _Value()), value)
        elif isinstance(node, (Measure, Reset)):
            costs = dict(self.zero_costs)
            width = self.scalar(self.admission.registers[node.wire].width, bindings)
            costs["measurements" if isinstance(node, Measure) else "resets"] = _Value(width)
            depth = 1
            if isinstance(node, Reset) and not self.admission.positive_parameter_count(self.admission.registers[node.wire].width):
                depth = self.arithmetic("min", 1, width)
            costs["logical_depth"] = _Value(depth)
            if isinstance(node, Measure):
                classical = self.define_classical(classical, (node.result,))
        elif isinstance(node, ClassicalStage):
            costs = dict(self.zero_costs)
            classical = self.define_classical(classical, node.outputs)
            kernel = self.kernels.get(node.kernel)
            if kernel is None:
                costs["classical_work"] = _unknown(f"{node.implementation.name}: classical work law unavailable")
                peaks[("memory", "host")] = _unknown("classical stage extra workspace is not declared")
            else:
                self.defined.add(kernel.content_id)
                applicable = tuple(law for law in kernel.resource_laws
                    if law.basis == self.context.basis and law.precision == self.context.precision
                    and law.synthesis == self.context.synthesis)
                if len(applicable) > 1:
                    raise ValueError("ambiguous applicable selected host work laws")
                if applicable:
                    law = applicable[0]
                    interpretation = ("estimate" if law.evidence.kind in {"observed", "empirical_prediction", "numerical_estimate"}
                                      else law.interpretation)
                    costs["classical_work"] = _Value(number(law.value), interpretation,
                        sources=frozenset((self.source(law.evidence),)), assumptions=frozenset(law.assumptions))
                else:
                    costs["classical_work"] = _unknown("selected host work law is unavailable in this resource context "
                        f"({kernel.implementation.name}: {kernel.implementation.domain})")
                if kernel.workspace:
                    peaks = self.peaks(peaks, self.snapshot(live, classical, bindings, kernel.workspace))
                else:
                    peaks[("memory", "host")] = _unknown("selected host native workspace is unavailable")
        elif isinstance(node, CoherentRegion):
            costs, peaks, live, classical = self.visit(node.body, bindings, live, classical)
        elif isinstance(node, Branch):
            self.unique_conditional = True
            ca, pa, la, va = self.visit(node.when_true, bindings, live, classical)
            cb, pb, lb, vb = self.visit(node.when_false, bindings, live, classical)
            costs = self.merge_cost(ca, cb, branch=True)
            # Scaling by an exact one keeps every value and attaches the branch
            # assumption to each metric.
            costs = self.scale_cost(costs, 1, assumption=f"branch {node.condition}: maximum over both paths; no probability model")
            peaks = self.peaks(pa, pb, bound=True,
                               assumption=f"branch {node.condition}: maximum over exclusive paths; no probability model")
            if la != lb:
                raise ValueError("admitted branch paths disagree on live footprint")
            live, classical = la, self.join_classical(va, vb)
        elif isinstance(node, (Repeat, AdaptiveLoop)):
            count = self.scalar(node.count if isinstance(node, Repeat) else node.max_rounds, bindings)
            if isinstance(node, AdaptiveLoop) or isinstance(count, str):
                self.unique_conditional = True
            if count != 0:
                part, child_peaks, end_live, end_classical = self.visit(node.body, bindings, live, classical)
                if isinstance(node, AdaptiveLoop):
                    end_classical = self.define_classical(end_classical, node.policy.outputs)
                    classical = self.join_classical(classical, end_classical)
                    if node.resource_envelope is None:
                        costs = {metric: _unknown("history-dependent adaptive costs require a per-round envelope") for metric in COUNTS}
                    else:
                        costs = self.scale_cost(part, count, bound=True, assumption=node.resource_envelope)
                    costs["adaptive_rounds"] = _Value(count, "upper_bound")
                    costs["classical_work"] = _unknown("adaptive policy work is not covered by the body envelope")
                    # Multiplying by one with bound=True keeps each peak value
                    # and marks it an upper bound, because a round may not run.
                    child_peaks = {key: self.combine(value, _Value(1), "multiply", bound=True)
                                   if node.resource_envelope is not None else _unknown("adaptive peak requires a history envelope")
                                   for key, value in child_peaks.items()}
                else:
                    costs = self.scale_cost(part, count)
                    positive = type(count) is int and count > 0 or self.admission.positive_parameter_count(node.count)
                    classical = end_classical if positive else self.join_classical(classical, end_classical)
                    live = end_live
                    if not positive:
                        child_peaks = {key: self.combine(value, _Value(1), "multiply", bound=True)
                                       for key, value in child_peaks.items()}
                peaks = self.peaks(peaks, child_peaks)
                if isinstance(node, AdaptiveLoop):
                    peaks[("memory", "host")] = _unknown("adaptive policy extra workspace is not declared")
        elif isinstance(node, MeasurementBatch):
            costs = dict(self.zero_costs)
            # costs/peaks are local, memoized workload facts; independent bodies
            # export no lifetime. The terminal-point union, reachability flags and
            # root populations below are operation-wide inventories, recorded on
            # the first visit to this admitted context, not copied into each value.
            terminal = not self.admission.contains_batch[node.body]
            if not terminal and node.axes:
                self.unique_unavailable.add("distinct terminal-point union across a nonterminal compact axis is unsupported")
            # Grid points per setting: the product over axes of
            # ceil((stop - start) / step), the size of range(start, stop, step).
            axis_count = 1
            for axis in node.axes:
                axis_count = self.arithmetic("multiply", axis_count, -(-(axis.stop - axis.start) // axis.step))
            for setting in node.settings:
                point = dict(bindings)
                point.update({item.parameter: number(item.value) for item in setting.bindings})
                for axis in node.axes:
                    point.pop(axis.parameter, None)
                repetitions = None if node.repetitions is None else self.scalar(node.repetitions, point)
                factor = (_unknown("axis-dependent repetition sum is outside the supported scalar-law subset")
                          if node.axes and isinstance(repetitions, str) else
                          _unknown("repetitions per setting unavailable") if repetitions is None else _Value(repetitions))
                if name == self.program.root:
                    self.root_populations["root_setting_declarations"] = self.combine(
                        self.root_populations.get("root_setting_declarations", _Value()), _Value(axis_count))
                    self.root_populations["root_repetitions"] = self.combine(
                        self.root_populations.get("root_repetitions", _Value()), self.combine(factor, _Value(axis_count), "multiply"))
                if terminal:
                    self.admission.tick(1 + len(point))
                    point_key = (name, setting.content_id, tuple(sorted(point.items())))
                    self.terminal_points[point_key] = axis_count
                elif repetitions is None or isinstance(repetitions, str):
                    self.unique_conditional = True
                if repetitions == 0:
                    if terminal:
                        costs["settings"] = self.combine(costs["settings"], _Value(axis_count))
                    continue
                part, child_peaks, _, _ = self.visit(node.body, point)
                if node.axes:
                    # Parameter-dependent range sums are not n*f(one selected point).
                    part = {metric: (_unknown("axis-dependent aggregate is outside the supported scalar-law subset")
                                     if isinstance(value.value, str) else value) for metric, value in part.items()}
                    child_peaks = {key: (_unknown("axis-dependent peak is outside the supported scalar-law subset")
                                        if isinstance(value.value, str) else value) for key, value in child_peaks.items()}
                positive_repetitions = (type(repetitions) is int and repetitions > 0
                                  or node.repetitions is not None and self.admission.positive_parameter_count(node.repetitions))
                if not positive_repetitions:
                    child_peaks = {key: self.combine(value, _Value(1), "multiply", bound=True)
                                       for key, value in child_peaks.items()}
                if not (len(node.settings) == 1 and axis_count == 1 and repetitions == 1):
                    child_peaks = self.batch_peaks(child_peaks, point)
                scaled = {metric: self.combine(value, factor, "multiply") for metric, value in part.items()}
                scaled = self.scale_cost(scaled, axis_count)
                for metric in DEPTHS:
                    scaled[metric] = self.combine(scaled[metric], _Value(1, "upper_bound",
                        assumptions=frozenset(("serial envelope across independent repetitions/settings; cross-job concurrency unspecified",))), "multiply")
                if terminal:
                    scaled["settings"] = _Value(axis_count)
                    acquisitions = self.combine(factor, _Value(axis_count), "multiply")
                    if node.observation_kind is None:
                        allocation = _unknown("terminal observation_kind unavailable")
                        acquisitions = self.combine(acquisitions, allocation, "multiply")
                        scaled["shots"] = scaled["exact_evaluations"] = acquisitions
                    else:
                        scaled["shots"] = scaled["exact_evaluations"] = _Value()
                        if node.observation_kind == "estimated_observable":
                            scaled["shots"] = self.combine(acquisitions,
                                _unknown("provider-managed sampling cost is unavailable"), "multiply")
                        else:
                            # A nonempty trajectory folds one exact evaluation of its
                            # selected body and reports its readout items separately
                            # (assessment's readout_items). An empty observation
                            # request has zero acquisitions: it declares no
                            # experiment.
                            metric = "shots" if node.observation_kind == "counts" else "exact_evaluations"
                            scaled[metric] = acquisitions
                        # The kind excludes the opposite population independently
                        # of body-cost uncertainty and outer repetitions.
                costs = self.merge_cost(costs, scaled)
                peaks = self.peaks(peaks, child_peaks)
        else:
            raise ValueError(f"unsupported resource node: {node.kind}")
        peaks = self.peaks(peaks, self.snapshot(live, classical, bindings))
        # The memo stores four existing immutable/persistent values. Arithmetic,
        # footprint updates and constructed dictionaries pay at their owners;
        # storing their references does not copy every child field again.
        self.admission.tick(4)
        result = (costs, peaks, live, frozenset(classical))
        self.memo[key] = result
        return result

    def quantity(self, metric, value, *, location=None, population="dynamic workload"):
        """Publish one partial cost as a ResourceQuantity with its derived evidence kind.

        The evidence kind is the weakest among the sources that determined the
        value: any observed or empirically predicted input makes the result an
        empirical prediction, then numerical estimate, user assertion, external
        specification and certified bound follow in that order, and a value
        derived only from exact recipes is a proved relation. An unknown value
        becomes an unavailable Fact carrying its reasons. A peak that needs a
        serial schedule is conditional unless the context declares one.
        """
        self.admission.tick(1 + len(value.sources) + len(value.derived_from) + len(value.assumptions) + len(value.reasons))
        unit = Unit(symbol="byte", dimension="bytes") if metric.endswith("bytes") or metric in {"memory", "known_memory"} else Unit(symbol="count", dimension="count")
        required_schedule = "serial_acquisitions" if value.requires_serial else None
        assumptions = value.assumptions
        if required_schedule is not None:
            population = "whole-workload peak under serial independent acquisitions"
            assumptions = assumptions | {"independent acquisitions share workspace only under the required serial schedule"}
        args = dict(quantity=metric, unit=unit, scope=Scope(domain=population),
                    assumptions=tuple(sorted(assumptions)))
        if value.value is None:
            fact = Fact(**args, availability="unknown",
                               reason="; ".join(sorted(value.reasons)) or "unavailable cost law")
            interpretation = "unavailable"
        else:
            interpretation = value.interpretation
            if required_schedule is not None and self.context.batch_schedule == "unspecified":
                interpretation = "conditional"
            # Only authoritative inputs determine the derived evidence kind.
            # Observed data used in the value keep it a prediction even when
            # another summand is asserted. Ancillary conformance is kept in
            # sources but cannot downgrade an independently known literal.
            kinds = {self.evidence[identity].kind for identity in value.derived_from}
            if kinds & {"observed", "empirical_prediction"}:
                kind = "empirical_prediction"
            elif "numerical_estimate" in kinds:
                kind = "numerical_estimate"
            else:
                kind = next((kind for kind in ("user_assertion", "external_specification", "certified_bound")
                             if kind in kinds), "proved_relation")
            evidence = Evidence(kind=kind, source=FOLD_SOURCE)
            if isinstance(value.value, str):
                fact = Fact(**args, availability="symbolic", symbol=Symbol(name=value.value, reference=FOLD_SOURCE), evidence=evidence)
            else:
                scalar = Rational(numerator=value.value, denominator=1) if type(value.value) is int else Float64(value=value.value)
                fact = Fact(**args, availability="concrete", value=scalar, evidence=evidence)
        return ResourceQuantity(metric=metric, fact=fact, population=population,
            basis=self.context.basis, interpretation=interpretation, location=location,
            sources=tuple(sorted(value.sources)),
            derivation_sources=tuple(sorted(value.derived_from)), required_schedule=required_schedule)

    def run(self):
        """Publish selected workload envelopes, live-memory peaks and unique construction work
        from one fold.
        """
        bindings = {item.parameter: number(item.value) for item in self.program.bindings}
        costs, peaks, _, _ = self.visit(self.program.root, bindings)
        costs = dict(costs)
        costs["expected_operations"] = _unknown("expected cost requires a probability/history model; no model was supplied")
        # Distinct terminal points and dynamic visits are different populations.
        # Unresolved control flow yields an envelope rather than an expected cost.
        unique = _Value()
        for cardinality in self.terminal_points.values():
            unique = self.combine(unique, _Value(cardinality))
        if self.unique_unavailable and self.terminal_points:
            unique = _Value(None, reasons=frozenset(self.unique_unavailable))
        elif self.unique_conditional and unique.value != 0:
            unique = replace(unique, interpretation="upper_bound",
                             assumptions=frozenset(("potential terminal point union over unresolved control/repetition",)))
        costs["unique_settings"] = unique
        costs.update(self.root_populations)
        populations = {
            "shots": "terminal MeasurementBatch sampled shots",
            "exact_evaluations": "terminal MeasurementBatch exact-statistic evaluations",
            "settings": "dynamic terminal MeasurementBatch setting visits",
            "unique_settings": "distinct potentially reached terminal setting points",
            "root_setting_declarations": "Program-root MeasurementBatch setting declarations",
            "root_repetitions": "Program-root MeasurementBatch body repetitions",
        }
        quantities = [self.quantity(metric, value,
                       population=populations.get(metric, "workload schedule envelope" if metric in DEPTHS and value.interpretation == "upper_bound" else "dynamic workload"))
                      for metric, value in costs.items()]
        quantities += [self.quantity(metric, value, location=location, population="simultaneous live footprint")
                       for (metric, location), value in sorted(peaks.items())]
        work = _Value()
        for selected in self.construction.selections:
            if self.identities[selected.signature.name] in self.defined:
                work = self.combine(work, _Value(selected.construction_work, "upper_bound") if selected.construction_work is not None
                                    else _unknown("selected native definition construction work unavailable"))
        for kernel in self.construction.kernels:
            if kernel.content_id in self.defined:
                work = self.combine(work, _Value(kernel.construction_work, "upper_bound"))
        quantities.append(self.quantity("construction_work", work, population="potential unique selected definitions and host bindings; excludes dynamic replay and numerical host execution"))
        self.admission.tick(len(self.expr) + len(self.evidence) + len(quantities) + len(self.defined))
        return WorkloadEstimate(construction_id=self.construction.content_id, context=self.context,
            quantities=quantities, expressions=tuple(self.expr.values()), parameters=self.program.parameters,
            sources=tuple(self.evidence.values()), defined_selections=tuple(sorted(self.defined)),
            work_units=self.admission.steps, evaluated_contexts=len(self.memo) + len(self.leaves), limits=self.program.limits,
            assumptions=self.program.premises + self.readiness.blockers + ("logical dependency schedule; no physical QEC/routing/factory projection",))


def estimate(construction: SelectedConstruction, *, context: ResourceContext | None = None,
             memo: dict | None = None) -> WorkloadEstimate:
    """Estimate the resources of a planned circuit description without building or running it.

    Most callers use [`nwqlib.estimate(plan)`][nwqlib.scientist.estimate],
    which calls this function on `plan.construction`. The estimate walks the
    circuit description once per distinct context and keeps symbolic costs
    in a compact expression table. It reads no numerical input data,
    synthesizes no gates and does not unroll repeated parts. Its work is
    limited in proportion to the size checks of the circuit description
    ([engineering
    constants](../ENGINEERING_CONSTANTS.md#shared-program-admission-limits)).

    The result depends only on the construction and the context. A caller
    that estimates several points of one Plan can pass one `memo`
    dictionary for the duration of its operation, and an equal pair of
    construction and context then returns the estimate already computed.
    The `memo` dictionary belongs to that caller, and there is no
    process-wide cache.

    Args:
        construction (SelectedConstruction): The planned circuit description,
            `plan.construction`.
        context (ResourceContext | None): Gate basis, rotation precision,
            synthesis, resident data and batch schedule. `None`, the
            default, means `ResourceContext()`: the `"selected_logical"`
            basis, no resident data and an unspecified schedule.
        memo (dict | None): Optional dictionary owned by the caller, mapping
            `(construction.content_id, context.content_id)` to the estimate
            already computed for that pair.

    Returns:
        workload (WorkloadEstimate): One quantity per metric and location.
            Every quantity is `"planned"` and keeps its label, location and
            original evidence. A metric without an applicable cost rule is
            present as unavailable, not as zero.
    """
    context = ResourceContext() if context is None else context
    if memo is None:
        return _Fold(construction, context).run()
    key = (construction.content_id, context.content_id)
    if key not in memo:
        memo[key] = _Fold(construction, context).run()
    return memo[key]
