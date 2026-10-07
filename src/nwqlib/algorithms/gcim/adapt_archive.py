"""ADAPT input, compiler and cache snapshots.

Loading never replans.
"""

from nwqlib._choice_archive import unsupported_archive_format
from nwqlib.blocks.selection import SelectedBlock
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import OperatorInput
from nwqlib.subroutines.fermionic_pool import FermionicGenerator
from .adapt_inputs import AdaptInputs
from .adapt_records import AdaptReconstruction


def _write_generator(generator):
    """Return a JSON-ready dict of a ``FermionicGenerator``.

    Each complex coefficient is split into its real and imaginary parts
    within its row, ``(real, imag, operators)`` for a fermionic term and
    ``(label, real, imag)`` for a Pauli term.
    """
    return dict(
        family=generator.family,
        spatial_indices=generator.spatial_indices,
        pool_index=generator.pool_index,
        num_qubits=generator.num_qubits,
        spin_orbital_indices=generator.spin_orbital_indices,
        fermion_terms=[
            (value.real, value.imag, operators) for value, operators in generator.fermion_terms
        ],
        pauli_terms=[(label, value.real, value.imag) for label, value in generator.pauli_terms],
    )


def _read_generator(saved):
    """Rebuild a ``FermionicGenerator`` from ``_write_generator`` output, lists back to tuples."""
    fields = dict(saved)
    fields["spatial_indices"] = tuple(fields["spatial_indices"])
    fields["spin_orbital_indices"] = (
        None if fields["spin_orbital_indices"] is None else tuple(fields["spin_orbital_indices"])
    )
    fields["fermion_terms"] = tuple(
        (complex(real, imag), tuple(tuple(pair) for pair in operators))
        for real, imag, operators in fields["fermion_terms"]
    )
    fields["pauli_terms"] = tuple(
        (label, complex(real, imag)) for label, real, imag in fields["pauli_terms"]
    )
    return FermionicGenerator(**fields)


def _compiler_entries(compilers):
    """Return ``((pool index, kind, active modes, occupation blocks), ...)`` of built compiler plans.

    ``compilers`` is a Plan's ``adapt.CompilerPlans`` map, or such entries
    saved from one, which a reopened Run's context or a Result snapshot
    holds. A classical Plan's empty tuple and None give no entries.
    """
    from .adapt import CompilerPlans

    if isinstance(compilers, CompilerPlans):
        return tuple(
            (index, p.kind, p.active_modes, p.occupation_blocks)
            for index, p in compilers.built()
        )
    return tuple(compilers or ())


def _write_compilers(compilers, files):
    """Return the JSON rows of built compiler plans (``_compiler_entries``), one per pool index.

    Each occupation block is one NPY file, written once per array object
    (``ArchiveFiles.write_array``).
    """
    entries = _compiler_entries(compilers)
    return [
        dict(
            kind=kind,
            generator=index,
            active_modes=modes,
            blocks=[
                (indices, files.write_array(f"compiler_{index}_{i}.npy", array))
                for i, (indices, array) in enumerate(blocks)
            ],
        )
        for index, kind, modes, blocks in entries
    ]


def _read_compilers(rows, files):
    """Return the ``(pool index, kind, active modes, occupation blocks)`` entries of saved compiler rows.

    ``adapt.CompilerPlans.restore`` binds each entry to the generator at its
    pool index when a Plan, Run or Result restores the saved context.
    """
    return tuple(
        (
            item["generator"],
            item["kind"],
            tuple(item["active_modes"]),
            tuple((tuple(indices), files.read_array(array)) for indices, array in item["blocks"]),
        )
        for item in rows
    )


def save(plan, files):
    """Save original and processed inputs together with the generator compiler payloads built so far.

    A generator's compiler plan is built when that generator is first
    selected (``adapt.CompilerPlans``), so only those plans are saved, by pool
    index. Shared-index compiler blocks are the generator's matrices on its
    active occupation states. Storing them lets a reopened Plan bind the same
    compiler data without repeating the block construction. A generator
    first selected after reopening builds its plan then. A durable Run
    writes this archive before any selection, so the plans it builds later
    are saved with its checkpoints (``save_context``).

    The archive format is ``adapt/7``. Its saved reconstruction carries the
    Hamiltonian action allowance (``hamiltonian_action_work``) computed at
    planning under this format's law, and loading reuses it without
    recomputing it, so an archive of an earlier format, whose query and
    action laws differ, is refused. A classical Plan's packed action tables
    are not saved. Loading packs them again under their own admission
    (``adapt_actions.build_action_tables``).
    """
    method = plan.method
    inputs = plan._native["inputs"]
    pool = (
        method.pool
        if isinstance(method.pool, str)
        else [
            dict(kind="operator", data=files.write_operator(f"original_pool_{i}", g))
            if isinstance(g, OperatorInput)
            else dict(kind="generator", data=_write_generator(g))
            for i, g in enumerate(method.pool)
        ]
    )
    return dict(
        format="adapt/7",
        plan=plan.to_record(),
        problem=files.write_problem(plan.problem),
        output=files.write_output(plan.output),
        method=method.model_dump(mode="json", exclude_computed_fields=True),
        initial=files.write_state("initial", method.initial_state),
        reference=files.write_state("selected_reference", plan._native["reference"]),
        original_pool=pool,
        # A processed Hamiltonian whose admitted content identity equals the
        # problem's is written once, by reference to the problem's arrays.
        processed=files.write_operator(
            "processed",
            plan.problem.A if inputs.hamiltonian.reference == plan.problem.A.reference
            else inputs.hamiltonian,
        ),
        generators=[_write_generator(g) for g in inputs.pool],
        compilers=_write_compilers(plan._native["compiler_plans"], files),
    )


def load(saved, files):
    """Reopen saved generator data and bind the saved query construction without rerunning selection."""
    from .adapt import ADAPT, CompilerPlans

    if saved.get("format") != "adapt/7":
        raise unsupported_archive_format("ADAPT archive", saved.get("format"), "adapt/7")
    fields = dict(saved["method"])
    original = saved["original_pool"]
    fields["pool"] = (
        original
        if isinstance(original, str)
        else tuple(
            files.read_operator(item["data"])
            if item["kind"] == "operator"
            else _read_generator(item["data"])
            for item in original
        )
    )
    fields["initial_state"] = files.read_state(saved["initial"])
    method = ADAPT(**fields)
    rec = AdaptReconstruction.model_validate(saved["plan"]["reconstruction"])
    plan = files.read_plan(
        saved["plan"],
        problem=files.read_problem(saved["problem"]),
        method=method,
        output=files.read_output(saved["output"]),
        reconstruction=rec,
    )
    if plan.execution == "quantum":
        # A restored quantum Plan passes the decoded label cache's
        # count-based reservation before its first cache decode, from the
        # stored row shapes, members and group sizes.
        from .adapt_inputs import restored_cache_reservation

        _check_bytes(restored_cache_reservation(rec, sampled=plan.shots is not None),
                     method.max_bytes, "restored ADAPT records and decoded label cache")
    reference = files.read_state(saved["reference"])
    generators = tuple(_read_generator(g) for g in saved["generators"])
    inputs = AdaptInputs(
        hamiltonian=files.read_operator(saved["processed"]),
        pool=generators,
        members=rec.pool,
        terms=rec.terms,
        groups=rec.groups,
        energy_groups=rec.energy_groups,
        dropped_l1=rec.hamiltonian_dropped_l1,
        processing=rec.processing,
        hamiltonian_work=rec.hamiltonian_action_work,
        hamiltonian_symmetry=rec.hamiltonian_symmetry,
        processing_source=rec.processing_source,
        host_identity_shift=rec.host_identity_shift,
        offset_free_requirements=(rec.offset_free_action_bytes, rec.offset_free_action_work),
    )
    if plan.execution == "classical":
        from .adapt_actions import build_action_tables, restored_record_allowance

        build_action_tables(inputs, rec.num_qubits, max_bytes=method.max_bytes,
                            max_products=method.max_products,
                            other_live=restored_record_allowance(inputs, rec))
    compilers = (
        CompilerPlans(
            generators,
            max_bytes=method.max_bytes,
            max_products=method.max_products,
            saved=_read_compilers(saved["compilers"], files),
        )
        if plan.execution == "quantum"
        else ()
    )
    from .adapt_acquisition import construct_adapt, bind_classical

    blocks = (
        tuple(
            SelectedBlock.bind(
                record,
                payload=(record.signature.name, inputs, reference, compilers, method),
                constructor=construct_adapt,
            )
            for record in plan.construction.selections
        )
        if plan.execution == "quantum"
        else (bind_classical(plan, inputs, reference),)
    )
    extra = {}
    if plan.execution == "quantum" and plan.shots is None:
        from .fixed_basis import _pair_table

        extra["pair_table"] = _pair_table(inputs.hamiltonian)
    return plan._bind(blocks=blocks, inputs=inputs, reference=reference, compiler_plans=compilers, **extra)


def save_context(context, files):
    """Persist current state/action caches, compiled gates and the compiler plans built so far.

    The Run's Plan archive is written before any selection, so the compiler
    plans of generators selected later travel with the context
    (``_write_compilers``). ``ArchiveFiles`` writes each array or gate object once and later
    checkpoints of the same Run refer to its file, so a checkpoint writes
    only the entries added since the previous one. The Run's journal or the
    Result's RunData owns the collected chunks and their order.
    """
    from collections.abc import Mapping
    from .adapt_acquisition import AdaptContext

    if isinstance(context, Mapping):
        context = AdaptContext(**context)
    arrays = {
        name: [
            (key, files.write_array(f"adapt-{name}-{index}.npy", array))
            for index, (key, array) in enumerate(getattr(context, name).items())
        ]
        for name in ("vectors", "h_columns")
    }
    gates = {
        name: [
            (key, files.write_instruction(f"adapt-{name}-{index}.qpy", gate))
            for index, (key, gate) in enumerate(getattr(context, name).items())
        ]
        for name in ("preparations", "gates", "reference_gates")
    }
    rules = []
    for index, rule in context.rules.items():
        value = dict(rule)
        if "terms" in value:
            value["terms"] = [dict(term) for term in value["terms"]]
        rules.append((index, value))
    return dict(
        arrays=arrays,
        gates=gates,
        rules=rules,
        compilers=_write_compilers(context.compiler_plans, files),
        accepted=context.accepted,
        pencil=None
        if context.pencil is None
        else context.pencil.model_dump(mode="json", exclude_computed_fields=True),
        pencil_parameters=context.pencil_parameters,
        pencil_attempt=context.pencil_attempt,
    )


def load_context(data, files):
    """Restore a saved ``AdaptContext`` without acquisition, analysis or circuit rebuilding.

    Chain keys return as tuples so cache lookups match live keys. The Run or
    the Result owns the observations, and the context stores none: a reopened
    Run rebuilds the observation index from its own chunks
    (``ADAPT._load_live_run_context``), and a Result keeps them in its
    RunData.
    Saved compiler plans return as ``(pool index, kind, active modes,
    occupation blocks)`` entries, which the reopened Run's
    controller adopts into its Plan's map (``adapt_acquisition.drive_adapt``)
    through ``adapt.CompilerPlans.restore`` at the saved pool indices.
    """
    from .adapt_acquisition import AdaptContext
    from .fixed_basis import ProjectedPencil

    def chain(value):
        return tuple(tuple(item) for item in value)

    context = AdaptContext()
    for name, rows in data["arrays"].items():
        setattr(context, name, {chain(key): files.read_array(value) for key, value in rows})
    for name, rows in data["gates"].items():

        def key(value):
            return (chain(value[0]), *value[1:]) if name == "preparations" else tuple(value)

        setattr(context, name, {key(k): files.read_instruction(value) for k, value in rows})
    context.rules = dict(data["rules"])
    context.compiler_plans = _read_compilers(data["compilers"], files)
    context.accepted = chain(data["accepted"])
    context.pencil = (
        None if data["pencil"] is None else ProjectedPencil.model_validate(data["pencil"])
    )
    context.pencil_parameters = (
        None if data["pencil_parameters"] is None else chain(data["pencil_parameters"])
    )
    context.pencil_attempt = data["pencil_attempt"]
    return context
