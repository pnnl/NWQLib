"""Save actual selected QPE inputs, powers and existing numerical caches."""

from nwqlib.blocks.selection import SelectedBlock, _preparation_circuit
from . import powers
from .method import _Input, _bind_host
from .records import QPEReconstruction, validate_selection


def save(method, plan, files):
    """Persist selected power payloads by constructor kind, sharing the original target data.

    Only constructor kinds and their scalar parameters are written. A dense
    power must reference the bound base matrix (the target for a Hamiltonian,
    the selected polar base for a unitary, which the archive stores once) and a
    product-formula step must use the reconstruction's Pauli terms, so the
    archive holds one copy of each operator. The format label names this payload layout
    (``qpe/<estimator>/7``), and load rejects any
    other label instead of converting it.
    """
    data = plan._native["input"]
    payloads = []
    if plan.execution == "quantum":
        for block in plan.blocks:
            constructor, payload = block._constructor, block._payload
            if constructor is _preparation_circuit:
                item = dict(kind="preparation")
            elif constructor is powers.construct_hadamard:
                item = dict(kind="hadamard")
            elif constructor is powers.construct_phase:
                item = dict(kind="phase", gate=payload)
            elif constructor is powers.construct_dense_power:
                matrix, tau, power = payload
                if matrix is not data.spectral._data:
                    raise ValueError("QPE power differs from its bound base data")
                item = dict(kind="dense_power", tau=tau, power=power)
            elif constructor is powers.construct_suzuki_step:
                terms, step_time = payload
                if terms != plan.reconstruction.pauli_terms:
                    raise ValueError("QPE step differs from selected Pauli terms")
                item = dict(kind="suzuki_step", step_time=step_time)
            else:
                raise ValueError("QPE archive needs its actual built-in constructors")
            payloads.append(item)
    return dict(
        format=f"qpe/{method.estimator}/7",
        estimator=method.estimator,
        plan=files.write_plan(plan),
        problem=files.write_problem(plan.problem),
        output=files.write_output(plan.output),
        method=method.model_dump(mode="json", exclude_computed_fields=True),
        initial_state=files.write_state("method-initial", method.initial_state)
        if method.initial_state is not None
        else None,
        target=files.write_operator("selected-target", data.target),
        reference=files.write_state("selected-reference", data.reference),
        base=None if data.base is None else files.write_operator("selected-base", data.base),
        tau=data.tau,
        payloads=payloads,
    )


def load(method_type, saved, files):
    """Bind the saved QPE inputs and known constructors without rebuilding powers."""
    if saved["format"] != f"qpe/{method_type.estimator}/7":
        raise ValueError("unsupported QPE archive format")
    fields = dict(saved["method"])
    fields["initial_state"] = (
        None if saved["initial_state"] is None else files.read_state(saved["initial_state"])
    )
    method = method_type.model_validate(fields)
    problem = files.read_problem(saved["problem"])
    r = QPEReconstruction.model_validate(saved["plan"]["reconstruction"])
    data = _Input(
        files.read_operator(saved["target"]), files.read_state(saved["reference"]), saved["tau"],
        None if saved["base"] is None else files.read_operator(saved["base"]),
    )
    plan = files.read_plan(
        saved["plan"],
        problem=problem,
        method=method,
        output=files.read_output(saved["output"]),
        reconstruction=r,
    )
    validate_selection(plan)
    if (
        data.target.reference != r.target
        or data.reference.preparation != r.preparation
        or data.tau != r.tau
        or (None if data.base is None else data.base.reference) != r.selected_base
    ):
        raise ValueError("QPE saved native inputs differ from actual selected data")
    if plan.execution == "classical":
        blocks = (_bind_host(plan, data, plan.construction.kernels[0]),)
    else:
        blocks = []
        constructors = dict(
            preparation=_preparation_circuit,
            hadamard=powers.construct_hadamard,
            phase=powers.construct_phase,
            dense_power=powers.construct_dense_power,
            suzuki_step=powers.construct_suzuki_step,
        )
        for record, item in zip(plan.construction.selections, saved["payloads"], strict=True):
            kind = item["kind"]
            payload = (
                data.reference
                if kind == "preparation"
                else item["gate"]
                if kind == "phase"
                else (data.spectral._data, item["tau"], item["power"])
                if kind == "dense_power"
                else (r.pauli_terms, item["step_time"])
                if kind == "suzuki_step"
                else None
            )
            blocks.append(
                SelectedBlock.bind(record, payload=payload, constructor=constructors[kind])
            )
        blocks = tuple(blocks)
    return plan._bind(blocks=blocks, input=data)


def save_context(context, files):
    """Write the Run's existing spectral context: target identity, eigensystem, reference and weights.

    Nothing is computed, so a context without an eigensystem is saved without
    one and a reopened Run continues from exactly what it had.
    """
    target, eigensystem, preparation, reference = (
        context[name] for name in ("target", "eigensystem", "preparation", "reference")
    )
    return dict(
        target=None
        if target is None
        else (target[0].model_dump(mode="json", exclude_computed_fields=True), target[1]),
        eigensystem=None
        if eigensystem is None
        else [
            files.write_array(f"qpe-spectrum-{index}.npy", array)
            for index, array in enumerate(eigensystem)
        ],
        preparation=None
        if preparation is None
        else preparation.model_dump(mode="json", exclude_computed_fields=True),
        reference=None
        if reference is None
        else (files.write_array("qpe-state.npy", reference[0]), reference[1]),
        weights=files.write_array("qpe-weights.npy", context["weights"]),
    )


def load_context(data, files):
    """Restore a saved spectral context without recomputing any array."""
    from nwqlib.operators.access import InputRef
    from nwqlib.problems.inputs import StatePreparationSpec

    return dict(
        target=None
        if data["target"] is None
        else (InputRef.model_validate(data["target"][0]), data["target"][1]),
        eigensystem=None
        if data["eigensystem"] is None
        else tuple(files.read_array(name) for name in data["eigensystem"]),
        preparation=None
        if data["preparation"] is None
        else StatePreparationSpec.model_validate(data["preparation"]),
        reference=None
        if data["reference"] is None
        else (files.read_array(data["reference"][0]), data["reference"][1]),
        weights=files.read_array(data["weights"]),
        squares=None,
    )
