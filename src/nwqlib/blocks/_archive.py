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
    """Serialize the shared selected-block graph once, placing transformed bases before their
    users.

    Selection admits the read-only coefficient array, padded amplitudes and
    their construction temporaries. The archive stores the shared operator
    and amplitude array once and derives coefficients from the restored
    operator. A signed Pauli SELECT or readout block therefore saves its
    operator and amplitude array only, for m terms on q qubits and
    P = 2**a >= m padded amplitudes.

    The selection law alone does not bound the whole archive or the live
    packed operator. With w=ceil(q/64), that operator has numerical data
    B_op=16mw+16m. A single shared amplitude file adds 8P. No coefficient
    file is needed when load derives coefficients from the operator:

        B_archive,data = 16mw + 16m + 8P.

    Add four NPY headers and the actual graph/manifest JSON. For these
    fixed-dtype rank-one/rank-two arrays with intp-sized shapes, 256 bytes
    per NPY header suffices. Shared SELECT/readout payloads and an
    already-written operator are counted once by object identity.

    For contiguous arrays, the inspected NumPy NPY writer
    (https://github.com/numpy/numpy/blob/v2.5.2/numpy/lib/_format_impl.py)
    uses at most one data bytes chunk of min(array.nbytes,2^24). If an
    iterator must copy noncontiguous data as well, charge a second such
    buffer. With contiguous packed and amplitude arrays, a numerical
    write-phase peak is

        B_held,other + B_op + 8m + 8P
         + min(max(8mw,16m,8P),2^24) + H_archive.

    H_archive includes actual graph serialization and writer bookkeeping and
    is not fixed for an arbitrary block graph. Take the maximum of this
    phase and selection, counting shared operator bytes once. A transport
    that buffers a whole archive needs its own additional charge. Keep the
    existing law (``_signed_pauli_requirements``) as the selection owner
    without relabeling it as an archive bound. This function takes no byte
    limit.
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
    """Restore saved blocks, binding only the known builtin constructors.

    A saved kind selects one constructor from this module's fixed table, so
    loading never imports or calls code named in the archive. A preparation
    is joined to its reloaded input before binding, and a transformed block
    is rebound to its already restored base. Selection admits the read-only
    coefficient array, padded amplitudes and their construction temporaries.
    The archive stores the shared operator and amplitude array once and
    derives coefficients from the restored operator. A signed Pauli SELECT
    or readout derives them with ``signed_pauli_coefficients``, the
    expression selection uses, and a SELECT and its readout share one
    restored payload.
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
