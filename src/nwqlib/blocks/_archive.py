"""Saved data for the existing preparation, Pauli and transformed blocks.

The saved graph carries its layout format ``BLOCKS_FORMAT``; ``read_blocks``
rejects any other layout.
"""

from . import selection as native
from .records import SelectedDefinition


BLOCKS_FORMAT = "nwqlib.blocks/2"

_CONSTRUCTORS = dict(preparation=native._preparation_circuit, primitives=native._primitive_circuit,
    select=native._select_circuit, readout=native._readout_circuit, reflection=native._reflection_circuit)


def validate_preparation_binding(record, state, constructor):
    """Join a saved known PREP to its actual input without inspecting amplitudes."""
    semantics = record.semantics
    expected = (native._primitive_circuit if record.implementation.name == "preparation.hzh"
                else native._preparation_circuit)
    if (constructor is not expected or semantics.kind != "preparation" or semantics.preparation != state.preparation
            or semantics.input != state.reference or semantics.basis != state.basis
            or sum(port.width for port in record.signature.quantum) != (state.basis.dimension - 1).bit_length()):
        raise ValueError("saved selected preparation differs from its actual input/specification/ports")


def write_blocks(blocks, files):
    """Return the saved form of a Method's selected blocks, for its `save_archive` hook.

    Call it in `save_archive(plan, files)` with the archive object `files` that
    the hook receives, and store the returned dict with the saved Plan. Each
    block is written once, base blocks before the blocks that transform them.
    Only the built-in preparation, Pauli SELECT and readout, reflection and
    transformed blocks can be saved. A signed Pauli SELECT or readout saves its
    operator and amplitude array only, because loading derives the coefficients
    from the operator, and blocks that share an operator or amplitude array
    write it once.

    The selection limit of
    [`select_signed_pauli`][nwqlib.blocks.selection.select_signed_pauli] does not
    bound the whole archive or the live packed operator. For m terms on q qubits,
    `P = 2**a >= m` padded amplitudes and `w = ceil(q/64)`, the packed operator
    holds `B_op = 16mw + 16m` bytes of numerical data and the one shared
    amplitude file adds `8P`, so

    ```text
    B_archive,data = 16mw + 16m + 8P,
    ```

    plus four NPY headers, at most 256 bytes each for these fixed-dtype
    rank-one and rank-two arrays, and the graph and manifest JSON. The NumPy NPY
    writer
    ([v2.5.2](https://github.com/numpy/numpy/blob/v2.5.2/numpy/lib/_format_impl.py))
    writes a contiguous array with at most one data chunk of
    `min(array.nbytes, 2**24)` bytes, and an array that must be copied because
    it is not contiguous needs a second such buffer. With contiguous packed and
    amplitude arrays, the numerical peak while writing is

    ```text
    B_held,other + B_op + 8m + 8P + min(max(8mw, 16m, 8P), 2**24) + H_archive,
    ```

    where `H_archive`, the graph serialization and writer bookkeeping, is not
    fixed for an arbitrary block graph. The peak of a save is the larger of this
    and the selection peak, counting shared operator bytes once. A transport that
    buffers a whole archive needs its own allowance. This function takes no byte
    limit, and the archive's byte limits apply to each file it writes.

    Args:
        blocks (Iterable[SelectedBlock]): The blocks to save, usually the Plan's
            bound blocks.
        files (ArchiveFiles): The archive object of the hook.

    Returns:
        saved (dict): The saved block graph, with its format, nodes and roots, for
            [`read_blocks`][nwqlib.blocks._archive.read_blocks].
    """
    from nwqlib.problems.inputs import StateInput
    roots = {id(block): index for index, block in enumerate(blocks)}
    nodes, saved = [], {}
    constructors = {value: key for key, value in _CONSTRUCTORS.items()}

    def write(block):
        """Append one block's node after its base and return its node index, once per block.

        A block that is one of the supplied roots stores its root index in
        place of its record, and ``read_blocks`` resolves that index in the
        caller's record table. Any other block, such as a base reached only
        through a transform, stores its record JSON. A transformed node
        stores its base's index. A leaf stores its constructor kind and its
        input, written through the archive's state, operator or array writers,
        or stored as is, such as a reflection's width.
        """
        if id(block) in saved:
            return saved[id(block)]
        base = None if block._base is None else write(block._base)
        index = len(nodes)
        name = f"block-{index}"
        record = roots.get(id(block))
        if record is None:
            record = block.record.model_dump(mode="json", exclude_computed_fields=True)
        if base is not None:
            node = dict(record=record, kind="transformed", base=base)
        else:
            kind = constructors[block._constructor]
            if kind == "primitives" and isinstance(block._payload, StateInput):
                kind = "preparation_primitives"
            if kind in ("preparation", "preparation_primitives"):
                payload = files.write_state(name, block._payload)
            elif kind in ("select", "readout"):
                operator, _, amplitudes = block._payload
                payload = dict(operator=files.write_operator(name, operator),
                               amplitudes=files.write_array(name+".amplitudes.npy", amplitudes))
            else:
                payload = block._payload
            node = dict(record=record, kind=kind, payload=payload)
        nodes.append(node)
        saved[id(block)] = index
        return index

    indices = [write(block) for block in blocks]
    return dict(format=BLOCKS_FORMAT, nodes=nodes, roots=indices)


def read_blocks(data, records, files):
    """Restore the blocks saved by `write_blocks`, for a Method's `load_archive` hook.

    A saved kind selects one constructor from a fixed table of built-in block
    kinds, so loading never imports or calls code named in the archive. A
    preparation is checked against its reloaded state input before it is bound,
    and a transformed block is bound again to its restored base. A signed Pauli
    SELECT or readout derives its coefficients from the restored operator with
    the expression selection uses, so alpha, the coefficients and the PREP
    amplitudes equal those before saving, and a SELECT and its readout share one
    restored operator.

    Args:
        data (dict): The saved block graph that `write_blocks` returned.
        records (Sequence[SelectedDefinition]): The selected definitions that the
            saved root blocks refer to by position. The reference Method passes
            `plan.construction.selections`.
        files (ArchiveFiles): The archive object of the hook.

    Returns:
        blocks (tuple[SelectedBlock, ...]): The restored blocks, in the saved order.

    Raises:
        ValueError: If the saved format is not the current block format, or a
            saved preparation does not match its input.
    """
    import json

    if data.get("format") != BLOCKS_FORMAT:
        raise ValueError(f"unsupported saved block graph format; expected {BLOCKS_FORMAT}")
    nodes, pauli = [], {}
    for item in data["nodes"]:
        saved_record = item["record"]
        record = records[saved_record] if type(saved_record) is int else SelectedDefinition.model_validate(saved_record)
        kind = item["kind"]
        if kind == "transformed":
            block = native.SelectedBlock.bind(record, base=nodes[item["base"]])
        else:
            payload = item["payload"]
            constructor = _CONSTRUCTORS["primitives" if kind == "preparation_primitives" else kind]
            if kind in ("preparation", "preparation_primitives"):
                payload = files.read_state(payload)
                validate_preparation_binding(record, payload, constructor)
            elif kind in ("select", "readout"):
                key = json.dumps(payload, sort_keys=True)
                if key not in pauli:
                    operator = files.read_operator(payload["operator"])
                    pauli[key] = (operator, native.signed_pauli_coefficients(operator),
                                  files.read_array(payload["amplitudes"]))
                payload = pauli[key]
            block = native.SelectedBlock.bind(record, payload=payload, constructor=constructor)
        nodes.append(block)
    return tuple(nodes[index] for index in data["roots"])
