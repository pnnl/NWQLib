"""Save the actual selected walk, never replan or regenerate acquisitions."""

from nwqlib._choice_archive import unsupported_archive_format
from nwqlib.blocks._archive import read_blocks, write_blocks
from .records import LanczosReconstruction


def save(plan, files):
    """Write the Plan and the live inputs bound to it, and return the archive index.

    Besides the Plan, Problem, output and Method fields, the index names the
    supplied initial state, the selected operator access (Pauli access on the
    quantum path, converted from matrix input when needed), the selected
    reference (zero-padded on the quantum path) and, for quantum execution,
    the selected native blocks.
    These are the inputs that Plan._bind held, so load can rebind them
    without replanning. The Pauli readout table is not written. Load
    derives it again from the saved operator.
    """
    return dict(
        format="lanczos/6",
        plan=plan.to_record(),
        problem=files.write_problem(plan.problem),
        output=files.write_output(plan.output),
        method=plan.method.model_dump(mode="json", exclude_computed_fields=True),
        initial=None
        if plan.method.initial_state is None
        else files.write_state("initial", plan.method.initial_state),
        operator=files.write_operator("selected_operator", plan._native["operator"]),
        reference=files.write_state("selected_reference", plan._native["reference"]),
        blocks=write_blocks(plan.blocks, files) if plan.execution == "quantum" else None,
    )


def load(saved, files):
    """Rebuild the bound Plan from an index written by save, without replanning.

    The Method is rebuilt from its saved fields and the stored initial state,
    and the Plan from its saved record with that Method. The saved operator
    and reference are bound again. For Pauli input the readout table is
    derived again from the saved operator, whose identity and census must
    reproduce the stored frame. Quantum execution restores the saved
    blocks. Classical execution rebinds the host recurrence kernel when the
    Plan has an experiment, and binds nothing when every moment is known,
    as for a scalar operator. Raises ValueError for another archive format.
    """
    from .method import Lanczos, _plan_readout

    if saved.get("format") != "lanczos/6":
        raise unsupported_archive_format("Lanczos archive", saved.get("format"), "lanczos/6")
    fields = dict(saved["method"])
    fields["initial_state"] = (
        None if saved["initial"] is None else files.read_state(saved["initial"])
    )
    method = Lanczos(**fields)
    plan = files.read_plan(
        saved["plan"],
        problem=files.read_problem(saved["problem"]),
        method=method,
        output=files.read_output(saved["output"]),
        reconstruction=LanczosReconstruction.model_validate(saved["plan"]["reconstruction"]),
    )
    operator, reference = (
        files.read_operator(saved["operator"]),
        files.read_state(saved["reference"]),
    )
    readout = _plan_readout(operator, plan.reconstruction, max_bytes=method.max_bytes)
    blocks = (
        read_blocks(saved["blocks"], plan.construction.selections, files)
        if plan.execution == "quantum"
        else method._host_blocks(plan, operator, reference, readout)
        if plan.experiments
        else ()
    )
    return plan._bind(blocks=blocks, operator=operator, reference=reference, readout=readout)
