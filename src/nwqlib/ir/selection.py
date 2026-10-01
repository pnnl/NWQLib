"""Select a validated experiment closure using an already admitted graph index."""

from .expressions import Binary, Binding, ExprRef, ParameterRef, walk_kept
from .records import BlockCall, Definition, MeasurementBatch, RangeAxis
from .validation import children


def select_experiment(admission, batch_id, setting_index, axis_values, *, bindings=None, parent_id=None):
    """Keep selected dependencies and all global constraints and register layouts.

    The caller owns the admitted immutable source index. Plan reuses that index
    across experiments; direct Program selection creates one for its operation.
    New composite nodes have enclosing Program ancestry rather than hashing the
    whole original batch's setting table for every selected point.

    Args:
        admission: An ``_Admission`` of the source Program after ``setup``.
        batch_id: Definition ID of the MeasurementBatch to select from.
        setting_index: Index of the kept setting.
        axis_values: Exactly one integer per range axis of that batch, each on the axis grid.
        bindings: Complete effective bindings of the point, or None to merge the
            Program's bindings with the setting's bindings and the axis values,
            which take precedence over the Program's.
        parent_id: Identity recorded as the new Program's parent, or None for
            the source Program's identity.

    Returns:
        A validated Program rooted at a copy of the batch with only the
        selected setting and no axes. It keeps the reachable definitions and
        the signatures they call, every register, classical value and global
        constraint, the expressions all of these reference, and the
        parameters those expressions and bindings name.
    """
    program = admission.p
    batch = admission.nodes.get(batch_id)
    if not isinstance(batch, MeasurementBatch):
        raise ValueError("select_experiment requires a MeasurementBatch definition")
    if type(setting_index) is not int or not 0 <= setting_index < len(batch.settings):
        raise ValueError("setting_index must select a kept setting")
    if axis_values.keys() != {axis.parameter for axis in batch.axes}:
        raise ValueError("provide exactly one value for each range axis")
    for axis in batch.axes:
        value = axis_values[axis.parameter]
        if (type(value) is not int or not axis.start <= value < axis.stop
                or (value - axis.start) % axis.step):
            raise ValueError("selected value is outside the range axis")
    setting = batch.settings[setting_index]
    point = {item.parameter: item for item in setting.bindings}
    point.update({key: Binding(parameter=key, value=value) for key, value in axis_values.items()})
    if axis_values:
        setting = setting.revise(bindings=tuple(point[key] for key in sorted(point)))
    fields = {name: getattr(batch, name) for name in type(batch).model_fields}
    fields.update(settings=(setting,), axes=(), parent_id=None)
    selected = Definition(id=batch_id, node=MeasurementBatch(**fields))
    if bindings is None:
        bound = {item.parameter: item for item in program.bindings}
        bound.update(point)  # Raw IR keeps setting-local precedence.
        bindings = tuple(bound[key] for key in sorted(bound))

    # Follow edges, not an O(all definitions) filter per experiment. Declaration
    # order is immaterial to execution; deterministic root-first traversal keeps
    # direct and loaded selection identical without another global position map.
    definitions = {}
    stack = [batch_id]
    while stack:
        name = stack.pop()
        if name in definitions:
            continue
        definition = selected if name == batch_id else admission.definitions[name]
        definitions[name] = definition
        stack.extend(reversed(children(definition.node)))
    names = dict.fromkeys(definition.node.signature for definition in definitions.values()
                          if isinstance(definition.node, BlockCall))
    signatures = tuple(admission.signatures[name] for name in names)

    # A global constraint, quantum/classical width, nested batch binding, formal
    # port width or classical argument can keep dependencies outside the body's
    # direct expression references. Preserve them before normal validation.
    parameters = set(item.parameter for item in bindings)
    pending = []
    slots = 0
    def reserve(amount):
        nonlocal slots
        slots += amount
        if slots > program.limits.max_steps:
            raise ValueError("selected dependency inventory exceeds max_steps")
    for source in (tuple(definitions.values()), signatures, program.registers, program.classical, program.constraints):
        for value in walk_kept(source, reserve):
            if isinstance(value, ExprRef):
                pending.append(value.expression)
            elif isinstance(value, (Binding, RangeAxis)):
                parameters.add(value.parameter)
    expressions = {}
    while pending:
        name = pending.pop()
        if name in expressions:
            continue
        definition = admission.expression_definitions[name]
        expressions[name] = definition
        value = definition.value
        if isinstance(value, Binary):
            pending.extend((value.right, value.left))
        elif isinstance(value, ParameterRef):
            parameters.add(value.parameter)
    return type(program)(
        parent_id=program.content_id if parent_id is None else parent_id,
        root=batch_id, definitions=tuple(definitions.values()), expressions=tuple(expressions.values()),
        parameters=tuple(admission.params[name] for name in sorted(parameters)), bindings=bindings,
        signatures=signatures, registers=program.registers, classical=program.classical,
        constraints=program.constraints, limits=program.limits, premises=program.premises,
    )
