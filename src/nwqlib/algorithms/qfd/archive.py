"""Save and reload a bound QFD Plan without planning again or rerunning its circuits."""

from nwqlib._choice_archive import unsupported_archive_format
from nwqlib.blocks._archive import read_blocks, write_blocks
from .construction import EVOLUTION, evolution_block
from .records import QFDReconstruction

FORMAT = "qfd/1"


def save(method, plan, files):
    """Write the Plan and the inputs bound to it, and return the archive index.

    Besides the Plan, Problem, output and Method fields, the index names the
    supplied initial state, the fermionic generator `evolution_terms` when
    the Method has one, the operator and the reference state the Plan uses
    and the saved form of every block except the evolution blocks. Those
    are V, its adjoint and their controlled forms, whose factors and angles
    the reconstruction stores, so loading builds them again from the
    reconstruction instead of saving a second copy.
    """
    blocks = [block for block in plan.blocks if block.record.implementation != EVOLUTION]
    terms = method.evolution_terms
    return dict(
        format=FORMAT,
        plan=plan.to_record(),
        problem=files.write_problem(plan.problem),
        output=files.write_output(plan.output),
        method=method.model_dump(mode="json", exclude_computed_fields=True),
        initial=files.write_state("initial", method.initial_state),
        evolution_terms=None if terms is None else files.write_operator("evolution_terms", terms),
        operator=files.write_operator("selected_operator", plan._native["operator"]),
        reference=files.write_state("selected_reference", plan._native["reference"]),
        blocks=write_blocks(blocks, files),
    )


def load(method_type, saved, files):
    """Rebuild the bound Plan from an index written by ``save``, without planning again.

    The Method is rebuilt from its saved fields, initial state and
    fermionic generator, and the Plan from its saved record. The saved
    blocks are restored in their positions, and each evolution block is
    built from the reconstruction's `evolution_rows`, stored angles and
    substep count, with the saved generator as its input: `evolution_terms`
    on the fermionic route and the saved operator otherwise. That gives the
    selected definition the Plan records, without mapping the generator or
    computing an angle again. An exact-readout Plan binds the pair
    reductions' Pauli table of the saved operator again. Raises ValueError
    for another archive format.
    """
    from nwqlib.algorithms.gcim.fixed_basis import _pair_table

    if saved.get("format") != FORMAT:
        raise unsupported_archive_format("QFD archive", saved.get("format"), FORMAT)
    fields = dict(saved["method"])
    fields["initial_state"] = files.read_state(saved["initial"])
    if saved["evolution_terms"] is not None:
        fields["evolution_terms"] = files.read_operator(saved["evolution_terms"])
    method = method_type(**fields)
    plan = files.read_plan(
        saved["plan"],
        problem=files.read_problem(saved["problem"]),
        method=method,
        output=files.read_output(saved["output"]),
        reconstruction=QFDReconstruction.model_validate(saved["plan"]["reconstruction"]),
    )
    operator, reference = files.read_operator(saved["operator"]), files.read_state(saved["reference"])
    rec, selections = plan.reconstruction, plan.construction.selections
    library = iter(read_blocks(saved["blocks"], [r for r in selections if r.implementation != EVOLUTION], files))
    rows, source = rec.evolution_rows, operator if method.evolution_terms is None else method.evolution_terms
    blocks = tuple(
        next(library) if record.implementation != EVOLUTION else evolution_block(
            record.choice, rows, rec.step_rows.array, rec.step_angles.array, method.steps_per_time_step, source,
            rec.pencil.num_qubits)
        for record in selections)
    if plan.shots is None and plan.experiments:
        return plan._bind(blocks=blocks, operator=operator, reference=reference, pair_table=_pair_table(operator))
    return plan._bind(blocks=blocks, operator=operator, reference=reference)
