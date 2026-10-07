"""Save original inputs, actual selected bodies and already-produced QLS data."""

from nwqlib.blocks.records import SelectedDefinition
from nwqlib.blocks.selection import SelectedBlock, _preparation_circuit
from nwqlib.blocks.encoding import _encoding_circuit
from .host_planning import OriginalEigensystem, OriginalSVD
from .method import QLS, _bind_host
from .primary_records import QLSReconstruction, validate_selection


def _label_masks(labels, num_qubits):
    """Return packed ``(x, z)`` uint64 words of I/X/Y/Z labels.

    Bit j describes qubit j, the rightmost printed letter, and word j // 64
    holds bit j % 64, the convention of ``operators._pauli.PauliTerms``: X
    and Y set x, Z and Y set z.
    """
    import numpy as np

    words = max(1, (num_qubits + 63) // 64)
    x = np.zeros((len(labels), words), dtype=np.uint64)
    z = np.zeros((len(labels), words), dtype=np.uint64)
    for row, label in enumerate(labels):
        for qubit, axis in enumerate(reversed(label)):
            bit = np.uint64(1) << np.uint64(qubit % 64)
            if axis in "XY":
                x[row, qubit // 64] |= bit
            if axis in "ZY":
                z[row, qubit // 64] |= bit
    return x, z


def _mask_labels(x, z, num_qubits):
    """Return the I/X/Y/Z labels of packed ``(x, z)`` words (inverse of ``_label_masks``)."""
    return tuple("".join("IXZY"[((int(x[row, j // 64]) >> (j % 64)) & 1) + 2 * ((int(z[row, j // 64]) >> (j % 64)) & 1)]
                         for j in reversed(range(num_qubits))) for row in range(len(x)))


def _save_decomposition(source, files, original):
    """Store a Pauli decomposition by reference to the saved problem operator, or as arrays.

    A compact Pauli ``A`` is decomposed from ``original.pauli_terms()`` with
    ``atol = 0.0``, so its decomposition is that operator's term table. An
    exact comparison of the packed labels and complex coefficients at save
    time shows this, and the archive then names the saved ``problem.A``
    instead of writing the table again. Any other decomposition, such as
    the Pauli transform of a dense ``A``, is written as packed x/z words and
    complex coefficients through ``files.write_array``.
    """
    import numpy as np

    fields = dict(input_dimension=source.input_dimension, operator_dimension=source.operator_dimension,
                  num_qubits=source.num_qubits, atol=source.atol)
    x, z = _label_masks(tuple(term.label for term in source.terms), source.num_qubits)
    coefficients = np.array([term.coefficient for term in source.terms], dtype=np.complex128)
    if "pauli_terms" in original.manifest.access:
        table = original.pauli_terms()
        if (source.atol == 0.0 and table.num_qubits == source.num_qubits and x.shape == table.x.shape
                and np.array_equal(x, table.x) and np.array_equal(z, table.z)
                and np.array_equal(coefficients, table.coefficients)):
            return dict(fields, reference="problem.A")
    return dict(fields, x=files.write_array("encoding-decomposition-x.npy", x),
                z=files.write_array("encoding-decomposition-z.npy", z),
                coefficients=files.write_array("encoding-decomposition-coefficients.npy", coefficients))


def _load_decomposition(data, files, original):
    """Rebuild the ``PauliDecomposition`` record that ``_save_decomposition`` stored."""
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    values = dict(data)
    if values.pop("reference", None) == "problem.A":
        rows = original.pauli_terms().labels()
    else:
        x, z, coefficients = (files.read_array(values.pop(name)) for name in ("x", "z", "coefficients"))
        rows = zip(_mask_labels(x, z, values["num_qubits"]), coefficients.tolist(), strict=True)
    values["terms"] = tuple(PauliTerm(label=label, coefficient=complex(c)) for label, c in rows)
    return PauliDecomposition(**values)


def _save_encoding(block, files, original):
    """Store the selected encoding in its own native, banded, Pauli or dense-plan representation.

    A Pauli decomposition is stored by ``_save_decomposition``, by reference
    to ``original`` when it is that operator's term table.
    """
    payload = block._payload
    data = dict(
        record=block.record.model_dump(mode="json", exclude_computed_fields=True),
        alpha=payload.alpha,
        num_ancillas=payload.num_ancillas,
        system_qubits=payload.system_qubits,
        error_bound=payload.error_bound,
        implementation=payload.implementation,
    )
    if block.record.implementation.name == "encoding.native":
        data.update(
            circuit=files.write_circuit("encoding.qpy", payload.circuit),
            metadata=files.write_numpy_json("encoding-metadata.json", dict(payload.metadata)),
        )
    else:
        data.update(
            requested_implementation=payload.requested_implementation,
            detail=files.write_numpy_json("encoding-detail.json", dict(payload.detail)),
        )
        if payload.implementation == "banded":
            source = payload.source
            data["bands"] = tuple(
                (offset, value.real, value.imag)
                for offset, value in zip(source.offsets, source.coefficients, strict=True)
            )
        if payload.decomposition is not None:
            data["decomposition"] = _save_decomposition(payload.decomposition, files, original)
    return data


def _load_encoding(data, files, encoded, original):
    """Rebind the stored encoding payload without performing new normalization or decomposition."""
    from nwqlib.subroutines.block_encoding.core import BlockEncoding, BlockEncodingPlan

    fields = {
        key: data[key]
        for key in ("alpha", "num_ancillas", "system_qubits", "error_bound", "implementation")
    }
    record = SelectedDefinition.model_validate(data["record"])
    if record.implementation.name == "encoding.native":
        payload = BlockEncoding(
            **fields,
            circuit=files.read_circuit(data["circuit"]),
            metadata=files.read_numpy_json(data["metadata"]),
        )
    else:
        source = encoded._data if fields["implementation"] == "dense_dilation" else None
        if "bands" in data:
            from nwqlib.subroutines.block_encoding.banded import BandSpecification

            source = BandSpecification(
                offsets=tuple(row[0] for row in data["bands"]),
                coefficients=tuple(complex(row[1], row[2]) for row in data["bands"]),
                num_qubits=fields["system_qubits"],
            )
        decomposition = None
        if "decomposition" in data:
            decomposition = _load_decomposition(data["decomposition"], files, original)
        payload = BlockEncodingPlan(
            **fields,
            requested_implementation=data["requested_implementation"],
            source=source,
            decomposition=decomposition,
            detail=files.read_numpy_json(data["detail"]),
        )
    return SelectedBlock.bind(record, payload=payload, constructor=_encoding_circuit)


_FACTOR_ARRAYS = {"original_svd": (OriginalSVD, ("left", "singular", "right_h")),
                  "original_eigh": (OriginalEigensystem, ("values", "vectors"))}


def _save_factors(factors, files):
    """Store the original factors a Plan owns, with their input binding, or None."""
    if factors is None:
        return None
    prefix = "original-svd-" if factors.method == "original_svd" else "original-eigh-"
    return dict(
        kind=factors.method,
        source=factors.source.model_dump(mode="json", exclude_computed_fields=True),
        arrays=tuple(files.write_array(prefix + name + ".npy", getattr(factors, name))
                     for name in _FACTOR_ARRAYS[factors.method][1]),
    )


def _load_factors(data, files):
    """Rebind saved original factors without a new decomposition."""
    from nwqlib.core.records import InputRef

    if data is None:
        return None
    owner = _FACTOR_ARRAYS[data["kind"]][0]
    return owner(InputRef.model_validate(data["source"]), *(files.read_array(name) for name in data["arrays"]))


def save(method, plan, files):
    """Store the selected Plan with the inputs and data it already produced.

    A quantum Plan writes its encoding payload, padded operator and RHS,
    original SVD frames and RHS preparation record once. A classical Plan
    reads the original ``A`` and ``b`` only, so it writes the original
    factors that its inverse or linear norm model reuses, and the encoding
    payload only when the Method itself carries a supplied encoding. The
    selected encoding's alpha, family, ancilla count and error bound travel
    in the reconstruction. The phase table travels inside the Plan's
    reconstruction. Nothing is recomputed to fill the archive.
    """
    validate_selection(plan)
    native = plan._native
    data = dict(
        format="qls/7",
        plan=plan.to_record(),
        problem=files.write_problem(plan.problem),
        output=files.write_output(plan.output),
        method=method.model_dump(mode="json", exclude_computed_fields=True),
        factors=_save_factors(native["svd"], files),
    )
    if plan.execution == "quantum":
        data.update(
            encoding=_save_encoding(native["encoding"], files, plan.problem.A),
            encoded_operator=files.write_operator("selected-operator", native["encoded_operator"]),
            rhs=files.write_state("selected-rhs", native["rhs"]),
            preparation=native["preparation"].record.model_dump(mode="json", exclude_computed_fields=True),
        )
    elif method.encoding is not None:
        data["encoding"] = _save_encoding(method.encoding, files, plan.problem.A)
    return data


def load(saved, files):
    """Restore selected QLS inputs and bind the saved native payloads.

    Quantum block bindings are rebuilt from the stored phases and queries.
    The stored Program and experiments remain the execution description.
    Loading performs no phase search, decomposition, polynomial fit or reference solve.
    """
    from .quantum import BaseEncoding, _program

    if saved["format"] != "qls/7":
        raise ValueError("unsupported QLS archive")
    problem = files.read_problem(saved["problem"])
    quantum = "encoded_operator" in saved
    encoded = files.read_operator(saved["encoded_operator"]) if quantum else problem.A
    encoding = _load_encoding(saved["encoding"], files, encoded, problem.A) if "encoding" in saved else None
    fields = dict(saved["method"])
    fields["encoding"] = None if fields["encoding"] is None else encoding
    method = QLS.model_validate(fields)
    factors = _load_factors(saved["factors"], files)
    rec = QLSReconstruction.model_validate(saved["plan"]["reconstruction"])
    plan = files.read_plan(
        saved["plan"],
        problem=problem,
        method=method,
        output=files.read_output(saved["output"]),
        reconstruction=rec,
    )
    if plan.execution == "classical":
        plan._bind(blocks=(_bind_host(plan),), svd=factors)
    else:
        rhs = files.read_state(saved["rhs"]) if quantum else None
        preparation = SelectedBlock.bind(
            SelectedDefinition.model_validate(saved["preparation"]),
            payload=rhs,
            constructor=_preparation_circuit,
        )
        base = BaseEncoding(encoding, problem.A, encoded, factors, method.max_bytes, method.max_work)
        _, _, blocks = _program(
            method, problem, plan.output, plan.shots, rec, base, preparation
        )
        plan._bind(
            blocks=blocks,
            base=base,
            preparation=preparation,
            encoding=encoding,
            encoded_operator=encoded,
            rhs=rhs,
            svd=factors,
        )
    return plan
