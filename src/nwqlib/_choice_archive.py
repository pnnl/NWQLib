"""The files of a saved Result or Run folder, written as JSON, NumPy arrays and QPY circuits.

Loading rebuilds records from these files. It does not restore a Python
session and does not certify the saved evidence. Method code chooses its data
and binds its own constructors.

Loading checks layout, not contents. The array headers of native state and
operator inputs (``read_state``, ``read_operator``) and of published Result arrays
(``saved_evidence.load_data``) are checked against their declared shape, dtype,
byte order, size and contiguity, and a mismatch always rejects. That check costs
only the header, and a wrong header would make every later read wrong. Arrays that
a Method writes through its own archive hook are mapped as stored, and that Method
owns their checks. Array contents are not scanned, hashed or normalized. They load
as read-only memory maps, so opening a large archive does not read every payload.

Saving a Result or a Run copy charges every file to the folder's file-byte
allowance. Files written through ``ArchiveFiles``, and the pages that the backup
of a durable Run's journal copies, are charged before they are written. SQLite
allocates the pages of a commit during the commit, so the pages that a commit
adds to the journal of a Run copy are charged right after that commit
(``_run_archive.save``). A durable Run charges the writes to its own folder to
its stored-data total. The Result and Run loaders open a folder without an
allowance (``max_bytes=None``) and charge nothing for reading, because each file
was charged when it was written and saved folders are treated as read-only
(docs/saved_evidence.md, "Saved folders are read-only").

Executable code is never chosen by saved text. ``method_class`` resolves only
built-in Methods or an explicitly supplied class.
"""

from contextlib import contextmanager
import json
from pathlib import Path


CACHE_FILE_PREFIX = ".nwqlib-cache-"


def unsupported_archive_format(name, found, expected):
    """Return the refusal for a saved folder written in another archive format.

    No loader converts another format, so the message names the folder's
    format and the only one this NWQLib reads, and says how to proceed.
    """
    return ValueError(
        f"unsupported {name} format {found!r}. This NWQLib reads only {expected!r}. "
        "Open it with the NWQLib release that wrote it, or plan and run the problem again."
    )


def _json_scalar(value):
    """Write a NumPy scalar as its Python value, and refuse any other non-JSON object."""
    import numpy as np
    if isinstance(value, np.generic):
        scalar = value.item()
        if type(scalar) in (bool, int, float, str):
            return scalar
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _array_layout(array, shape, dtype, byte_order):
    """Check a stored header without scanning, casting or normalizing its data."""
    import numpy as np
    expected = np.dtype(dtype)
    if byte_order in ("little", "big"):
        expected = expected.newbyteorder("<" if byte_order == "little" else ">")
    if (array is None or array.shape != tuple(shape) or array.dtype != expected
            or not array.flags.c_contiguous):
        raise ValueError("saved native array layout differs from its input declaration")


class _LimitedWriter:
    """Charge each write to the archive's byte allowance before it reaches the file.

    QPY's writer needs ``tell`` and ``seek``, so those pass through unchanged. A
    write that would exceed the allowance raises before any byte of it is written.
    """

    def __init__(self, stream, archive):
        self.stream, self.archive = stream, archive

    def write(self, data):
        self.archive.reserve(memoryview(data).nbytes)
        return self.stream.write(data)

    def tell(self):
        return self.stream.tell()

    def seek(self, offset, whence=0):
        return self.stream.seek(offset, whence)

    def seekable(self):
        return self.stream.seekable()

    def flush(self):
        return self.stream.flush()


class ArchiveFiles:
    """The folder of a saved Result or Run, as passed to a Method's `save_archive` and `load_archive` hooks.

    A Method author writes and reads the Method's data through the `files`
    argument of those hooks, as [Add a Method](../algorithm_protocol.md) and
    [Run your own circuit](../own_circuit.md) show, rather than through a
    serializer of its own. Its operations keep each input in its own
    representation and count the written bytes against the folder's limit.
    Files have readable names, are created new and are never overwritten.
    An array or circuit used by several parts is written once per save and
    read once per load. Arrays load as read-only NumPy memory maps and are
    neither normalized nor content-hashed. The headers of input arrays are
    checked against their representation, dimensions and declared encoding.
    These checks do not validate every value or restore the digest that the
    input had when it was first accepted. Keep the folder available while
    using the restored Plan. Circuits use Qiskit's public QPY format. The
    memory and CPU use of the SDKs is unknown.
    """

    def __init__(self, path, max_bytes):
        """Open the folder `path`, with a limit of `max_bytes` on the bytes written.

        NWQLib opens the folder and passes it to the hooks, so Method code
        does not build one.

        Args:
            path (str | os.PathLike): The folder.
            max_bytes (int | None): Positive limit on the bytes written, or
                `None` for a folder without a limit of its own.

        Raises:
            ValueError: If `max_bytes` is neither a positive integer nor
                `None`.
        """
        # ``remaining`` is the allowance still available. With
        # ``max_bytes=None`` the folder has no allowance of its own and
        # ``remaining`` stays None. The Result and Run loaders open a folder
        # this way, and so does a durable Run for its own folder, because the
        # Run charges each write to its stored-data total through
        # ``_on_reserve``. The ``_written_*`` maps (by object identity) and
        # ``_read_*`` maps (by file name) let a shared object be written or
        # read once. ``_written_files`` lists files this session wrote or
        # registered, which orphan cleanup keeps. ``_dependencies`` records the
        # NPY files that a JSON file points to. ``_pending_files`` is not None
        # while a cache delta is being written, so a failed commit can delete
        # exactly those files and refund their bytes.
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("archive max_bytes must be a positive integer or None")
        self.path, self.remaining = Path(path), max_bytes
        self._written_arrays, self._read_arrays = {}, {}
        self._written_circuits, self._read_circuits = {}, {}
        self._written_instructions, self._read_instructions = {}, {}
        self._on_reserve = None
        self._written_files, self._dependencies = set(), {}
        self._file_cache_keys = {}
        self._pending_files = None
        self._pending_bytes = 0

    def reserve(self, size):
        """Charge ``size`` bytes before they are written, or read under an allowance.

        The charge goes to this folder's own allowance when it has one, and to
        the Run's stored-data total when a Run bound ``_on_reserve``.
        """
        if type(size) is not int or size < 0:
            raise ValueError("archive byte count must be a nonnegative integer")
        if self.remaining is not None and size > self.remaining:
            raise ValueError("archive exceeds its explicit file-byte allowance")
        if self._on_reserve is not None:
            self._on_reserve(size)
        if self.remaining is not None:
            self.remaining -= size
        if self._pending_files is not None:
            self._pending_bytes += size

    def file(self, name):
        """Path of ``name`` inside the archive folder.

        Saved JSON names its files, so only a plain basename is accepted. A name
        with a directory part could otherwise read or write outside the archive.
        """
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("archive file name must be a local basename")
        return self.path / name

    @contextmanager
    def writer(self, name):
        """Open a new file in exclusive-create mode behind a byte-charging writer.

        Mode ``xb`` fails if the file exists, so a file that a committed row
        names is never overwritten.
        """
        with self.file(name).open("xb") as stream:
            if self._pending_files is not None:
                self._pending_files.append(name)
            yield _LimitedWriter(stream, self)
        self._written_files.add(name)

    def _remember(self, kind, value, name):
        """Record that ``value`` was written to or read from the file ``name``.

        The record lets ``_forget_objects`` drop the value's cache entries when
        that file is deleted or its native handle is released
        (``_run_archive.release_native``).
        """
        key = id(value)
        getattr(self, "_written_" + kind)[key] = (value, name)
        self._file_cache_keys.setdefault(name, set()).add((kind, key))

    def _forget_objects(self, name):
        """Drop only objects attached to this file, without scanning history."""
        for kind, key in self._file_cache_keys.pop(name, ()):
            getattr(self, "_written_" + kind).pop(key, None)
        for kind in ("arrays", "circuits", "instructions"):
            getattr(self, "_read_" + kind).pop(name, None)
        # One bound noise model is a fixed-size owner, not growing history.
        for key, (_, path) in tuple(getattr(self, "_noise_models", {}).items()):
            if path == name:
                del self._noise_models[key]

    def _begin_cache_files(self):
        """Start recording the files and bytes of one cache delta (see ``_finish_cache_files``)."""
        if self._pending_files is not None:
            raise ValueError("an archive already has an unfinished cache delta")
        self._pending_files, self._pending_bytes = [], 0

    def _finish_cache_files(self, *, committed):
        """End one cache delta and return its new file names and charged bytes.

        After a failed commit the delta's new files are deleted. Only a Run
        writes cache deltas that can fail, and it refunds the returned bytes
        to its stored-data total.
        """
        names, size = self._pending_files, self._pending_bytes
        self._pending_files, self._pending_bytes = None, 0
        if not committed:
            for name in names:
                self.file(name).unlink(missing_ok=True)
                self._written_files.discard(name)
                self._dependencies.pop(name, None)
                self._forget_objects(name)
        return tuple(names), size

    def read_path(self, name):
        """Register a file that the Method's saved data depends on, and return its path.

        A `load_archive` hook that restores cached data must call it for
        every file it refers to, including files whose contents it reads only
        later. Registration reads the file's metadata only, so a missing file
        raises `FileNotFoundError` here, and it keeps the file from being
        removed as unused when the Run is reopened. A folder opened with a
        limit counts the file's size against it, and a folder opened without
        one (`max_bytes=None`) counts nothing.

        Args:
            name (str): The file name, without a directory part.

        Returns:
            path (pathlib.Path): The file's path.
        """
        path = self.file(name)
        size = path.stat().st_size
        if self.remaining is not None:
            self.reserve(size)
        self._written_files.add(name)
        return path

    def write_json(self, name, data):
        """Stream ``data`` as JSON, charging each encoded part before it is written.

        NaN and infinity are refused, so every saved number reads back finite.
        """
        with self.writer(name) as stream:
            for part in json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False,
                                         default=_json_scalar).iterencode(data):
                stream.write(part.encode("utf-8"))
            stream.write(b"\n")

    def read_json(self, name):
        """Read one JSON file, charging its size against the folder allowance first."""
        with self.read_path(name).open(encoding="utf-8") as stream:
            return json.load(stream)

    def write_array(self, name, array):
        """Save ``array`` as NPY without pickles, once per array object, and return its file name."""
        if array is None:
            return None
        previous = self._written_arrays.get(id(array))
        if previous is not None:
            return previous[1]
        name = self._unused_name(name)
        import numpy as np
        with self.writer(name) as stream:
            np.save(stream, array, allow_pickle=False)
        # Keep the object reference so temporary-object id reuse cannot alias
        # unrelated arrays. This does not copy or compare numerical contents.
        self._remember("arrays", array, name)
        return name

    def read_array(self, name):
        """Map a saved NPY file read-only without reading or checking its values.

        A folder opened with an allowance charges the file size before mapping
        (``read_path``). A truncated file or a bad NPY header fails here with the
        entry path. A caller that owns a declaration checks the header against it.
        """
        if name is None:
            return None
        if name in self._read_arrays:
            return self._read_arrays[name]
        import numpy as np
        # Consumers accept ordinary ndarray semantics. This view shares the
        # read-only mapping. It neither copies values nor exposes a subclass
        # with overridden numerical operations at a scientific input boundary.
        path = self.read_path(name)
        try:
            array = np.asarray(np.load(path, allow_pickle=False, mmap_mode="r"))
        except (ValueError, OSError, EOFError) as error:
            error.add_note(f"Cannot read saved array {path}; keep the complete archive available.")
            raise
        self._read_arrays[name] = array
        self._remember("arrays", array, name)
        return array

    def write_numpy_json(self, name, data):
        """Save SDK-owned JSON containers with ndarray leaves in exact NPY files.

        Array paths are separate from user dictionaries, so no reserved SDK key
        or interpretation of gate names, units or complex entries is required.
        """
        import numpy as np
        name = self._unused_name(name)
        arrays = []
        def convert(value, path):
            """Copy the container tree, replacing each ndarray by None and recording its key path."""
            if isinstance(value, np.ndarray):
                arrays.append(dict(path=path, file=self.write_array(f"{name}.{len(arrays)}.npy", value)))
                return None
            if isinstance(value, dict):
                return {key: convert(item, [*path, key]) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item, [*path, index]) for index, item in enumerate(value)]
            return value
        converted = convert(data, [])
        self.write_json(name, dict(data=converted, arrays=arrays))
        self._dependencies[name] = tuple(row["file"] for row in arrays)
        return name

    def read_numpy_json(self, name):
        """Restore ndarray leaves without changing their dtype, shape or values."""
        saved = self.read_json(name)
        self._dependencies[name] = tuple(row["file"] for row in saved["arrays"])
        data = saved["data"]
        for row in saved["arrays"]:
            array = self.read_array(row["file"])
            if not row["path"]:
                data = array
                continue
            parent = data
            for key in row["path"][:-1]:
                parent = parent[key]
            parent[row["path"][-1]] = array
        return data

    def write_circuit(self, name, circuit):
        """Save one circuit as raw QPY, or as the prefixed UCGate container when needed.

        Qiskit's QPY writes a UCGate that its own loader cannot rebuild.
        ``_qpy_archive.encode_circuit`` replaces each UCGate by a storage
        instruction and returns None when the circuit has none, so ordinary
        circuits stay plain QPY files.
        """
        if circuit is None:
            return None
        previous = self._written_circuits.get(id(circuit))
        if previous is not None:
            return previous[1]
        name = self._unused_name(name)
        from qiskit import qpy
        from nwqlib._qpy_archive import PREFIX, QpyStream, encode_circuit
        storage = encode_circuit(circuit)
        with self.writer(name) as stream:
            if storage is not None:
                stream.write(PREFIX)
            qpy.dump(circuit if storage is None else storage, stream if storage is None else QpyStream(stream))
        self._remember("circuits", circuit, name)
        return name

    def read_circuit(self, name):
        """Load one saved circuit, choosing the decoder by the file's leading prefix.

        Only a file that starts with ``PREFIX`` is decoded as the UCGate container,
        so raw QPY is never interpreted as that format. A decoding error carries
        the entry path, and no circuit is synthesized or simplified.
        """
        if name is None:
            return None
        if name in self._read_circuits:
            return self._read_circuits[name]
        from qiskit import qpy
        from nwqlib._qpy_archive import PREFIX, QpyStream, decode_circuit
        path = self.read_path(name)
        try:
            with path.open("rb") as stream:
                encoded = stream.read(len(PREFIX)) == PREFIX
                if not encoded:
                    stream.seek(0)
                circuit, = qpy.load(QpyStream(stream) if encoded else stream)
            if encoded:
                circuit = decode_circuit(circuit)
        except Exception as error:
            error.add_note(f"while reading native circuit archive entry {path}")
            raise
        self._read_circuits[name] = circuit
        self._remember("circuits", circuit, name)
        return circuit

    def write_instruction(self, name, instruction):
        """Use a one-operation QPY circuit for a cached native instruction."""
        previous = self._written_instructions.get(id(instruction))
        if previous is not None:
            return previous[1]
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(instruction.num_qubits, instruction.num_clbits)
        circuit.append(instruction, circuit.qubits, circuit.clbits, copy=False)
        name = self.write_circuit(name, circuit)
        self._remember("instructions", instruction, name)
        return name

    def _unused_name(self, name):
        """Return a name derived from ``name`` that no file in the folder has.

        While a cache delta is open (``_begin_cache_files``) the name starts
        with ``CACHE_FILE_PREFIX``, which marks a file that reopening deletes
        when no committed row names it (``_run_archive.restore_caches``).
        """
        self.file(name)
        prefix = CACHE_FILE_PREFIX if self._pending_files is not None else ""
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        candidate, index = prefix + name, 0
        while self.file(candidate).exists():
            index += 1
            candidate = f"{prefix}{index}-{name}"
        return candidate

    def read_instruction(self, name):
        """Load a cached native instruction from its one-operation QPY circuit (``write_instruction``)."""
        if name not in self._read_instructions:
            circuit = self.read_circuit(name)
            self._read_instructions[name] = circuit.data[0].operation
            instruction = self._read_instructions[name]
            self._remember("instructions", instruction, name)
        return self._read_instructions[name]

    def write_state(self, name, state):
        """Save a state input, its preparation and its circuit, and return its description.

        The vector and its normalized direction are stored as given, without
        normalizing them again, so scale and phase survive exactly. When the
        two arrays have the same dtype, shape and bytes (a unit-norm vector is
        its own direction), one array is written and the direction names its
        file. The comparison is one pass over the entries.

        Args:
            name (str): Prefix of the file names.
            state (StateInput): The state input.

        Returns:
            description (dict): JSON description that `read_state` reads.
        """
        import numpy as np
        physical, direction = state._physical, state._direction
        physical_name = self.write_array(name + ".physical.npy", physical)
        if (physical is not None and direction is not None and physical.dtype.str == direction.dtype.str
                and physical.shape == direction.shape and physical.flags.c_contiguous
                and direction.flags.c_contiguous
                # Byte views of the two C-ordered buffers, compared without a copy.
                and memoryview(physical.reshape(-1).view(np.uint8)) == memoryview(direction.reshape(-1).view(np.uint8))):
            direction_name = physical_name
        else:
            direction_name = self.write_array(name + ".direction.npy", direction)
        return dict(manifest=state.manifest.model_dump(mode="json", exclude_computed_fields=True),
            preparation=state.preparation.model_dump(mode="json", exclude_computed_fields=True),
            physical=physical_name, direction=direction_name,
            circuit=self.write_circuit(name + ".qpy", state._native))

    def read_state(self, data):
        """Restore a state input after checking its payload layout against its manifest.

        The expected array shape follows the representation. A vector of dimension
        ``D`` has shape ``(D,)``. A product state on ``n`` qubits has ``(n, 2)`` and
        an occupation string ``(n,)``, and both require ``D = 2**n``. A complex128
        normalized direction must exist exactly when the state is not an occupation
        string and its physical scale is nonzero. Payload bytes must equal the
        declared size. A direction that names the physical file is that one
        payload (``write_state``), read once.
        A supplied circuit must match the declared width and have no classical bits
        or parameters. Values are not rescanned, so a state is not normalized or
        re-ingested on load.
        """
        from nwqlib.problems.inputs import StateInput
        state = StateInput.from_record(dict(format="nwqlib.state_input/2",
            manifest=data["manifest"], preparation=data["preparation"]))
        manifest, spec = state.manifest, state.preparation
        physical, direction = self.read_array(data["physical"]), self.read_array(data["direction"])
        representation, dimension = manifest.reference.representation, manifest.basis.dimension
        if physical is not None or direction is not None:
            if representation not in ("vector", "product", "occupation") or data["circuit"] is not None:
                raise ValueError("saved state payload differs from its representation")
            qubits = dimension.bit_length() - 1
            shape = ((dimension,) if representation == "vector" else
                     (qubits, 2) if representation == "product" else (qubits,))
            if (manifest.shape != shape or spec.physical_scale is None
                    or representation != "vector" and (qubits < 1 or dimension != 1 << qubits)
                    or manifest.dtype not in (("uint8",) if representation == "occupation" else ("float64", "complex128"))):
                raise ValueError("saved state dimensions or encoding differ from its representation")
            _array_layout(physical, shape, manifest.dtype, manifest.byte_order)
            needs_direction = representation != "occupation" and spec.physical_scale.mantissa != 0
            if needs_direction:
                _array_layout(direction, shape, "complex128", manifest.byte_order)
            elif direction is not None:
                raise ValueError("saved state has an undeclared normalized direction")
            if physical.nbytes + (0 if direction is None else direction.nbytes) != manifest.payload_bytes:
                raise ValueError("saved state payload size differs from its declaration")
        if data["circuit"] is not None and (representation != "circuit" or spec.implementation != "qiskit.supplied"):
            raise ValueError("saved circuit differs from its state preparation declaration")
        native = self.read_circuit(data["circuit"])
        if native is not None and (dimension != 1 << native.num_qubits or native.num_clbits or native.num_parameters):
            raise ValueError("saved state circuit width or parameters differ from its declaration")
        return StateInput._from_admitted(manifest, spec, physical, direction, native)

    def write_operator(self, name, operator):
        """Save an operator in its own representation, never a converted one, and return its description.

        Dense, CSR, CSC, Pauli and periodic-stencil inputs keep their arrays or
        parameters, so saving never turns a sparse or Pauli operator into a
        dense matrix. A declaration without data is saved as a declaration.

        Args:
            name (str): Prefix of the file names.
            operator (OperatorInput): The operator.

        Returns:
            description (dict): JSON description that `read_operator` reads.
        """
        representation = operator.manifest.reference.representation
        data = dict(manifest=operator.manifest.model_dump(mode="json", exclude_computed_fields=True),
                    structure=operator.structure)
        if operator._data is None:
            data["declaration_only"] = True
        elif representation == "periodic_stencil":
            from dataclasses import asdict
            data["parameters"] = asdict(operator._data)
        elif representation == "dense":
            data["array"] = self.write_array(name + ".npy", operator._data)
        elif representation in ("csr", "csc"):
            native = operator._data
            data.update(sparse_type=type(native).__name__, shape=native.shape,
                arrays={field: self.write_array(name + "." + field + ".npy", getattr(native, field))
                        for field in ("data", "indices", "indptr")})
        elif representation == "pauli":
            terms = operator.pauli_terms()
            data.update(num_qubits=terms.num_qubits,
                x=self.write_array(name + ".x.npy", terms.x),
                z=self.write_array(name + ".z.npy", terms.z),
                coefficients=self.write_array(name + ".coefficients.npy", terms.coefficients))
        else:
            raise ValueError(f"execution archive has no saved operator format for {representation}")
        return data

    def read_operator(self, data):
        """Restore an operator after checking every array header against its manifest.

        Each representation has a fixed envelope. Dense needs a ``(D, D)``
        float64 or complex128 array. CSR/CSC needs one-dimensional values, signed
        32 or 64-bit indices of the same length and ``D + 1`` index pointers with
        ``indptr[0] == 0`` and ``indptr[-1]`` equal to the value count. Pauli needs
        ``uint64`` x and z masks of ``ceil(n/64)`` words per term and complex128
        coefficients. A periodic stencil restores its parameters and checks its
        declared width. In every case the payload bytes must equal the declared size.
        Individual sparse indices and coefficient values are not scanned. The
        restored object wraps the memory-mapped arrays without copying them: a
        sparse matrix is built empty with the declared shape, and its three
        arrays are then set to the mapped ones, so SciPy does not check or
        convert the saved indices.
        """
        from nwqlib.operators.inputs import OperatorInput
        from nwqlib.operators._pauli import PauliTerms
        declared = OperatorInput.from_record(dict(format="nwqlib.operator_input/2",
            manifest=data["manifest"], structure=data["structure"]))
        manifest = declared.manifest
        representation = manifest.reference.representation
        if "declaration_only" in data:
            return declared
        dimension = manifest.basis.dimension
        if representation == "periodic_stencil":
            from nwqlib.operators.inputs import PeriodicStencil
            # The manifest declares 24 bytes for the three float64 parameters
            # (mass, diffusion, potential) plus the minimal byte length of the
            # integer num_qubits, as operators.inputs.ingest_periodic_stencil
            # writes it.
            native = PeriodicStencil(**data["parameters"])
            if (dimension.bit_length() - 1 != native.num_qubits or dimension & (dimension - 1)
                    or manifest.shape != (4,) or manifest.dtype != "integer+3*float64"
                    or manifest.payload_bytes != 24 + max(1, (native.num_qubits.bit_length() + 7) // 8)):
                raise ValueError("saved periodic parameters differ from their input layout")
        elif representation == "dense":
            native = self.read_array(data["array"])
            if manifest.shape != (dimension, dimension) or manifest.dtype not in ("float64", "complex128"):
                raise ValueError("saved dense operator has invalid dimensions or encoding")
            _array_layout(native, manifest.shape, manifest.dtype, manifest.byte_order)
            if native.nbytes != manifest.payload_bytes:
                raise ValueError("saved dense operator size differs from its declaration")
        elif representation in ("csr", "csc"):
            from scipy import sparse
            classes = {cls.__name__: cls for cls in (sparse.csr_matrix, sparse.csc_matrix,
                                                    sparse.csr_array, sparse.csc_array)}
            if (data["sparse_type"] not in {representation + "_matrix", representation + "_array"}
                    or type(data["arrays"]) is not dict or set(data["arrays"]) != {"data", "indices", "indptr"}
                    or tuple(data["shape"]) != (dimension, dimension) or manifest.shape != (dimension, dimension)
                    or manifest.dtype not in ("float64", "complex128")):
                raise ValueError("saved sparse operator layout differs from its declaration")
            arrays = tuple(self.read_array(data["arrays"][name]) for name in ("data", "indices", "indptr"))
            values, indices, indptr = arrays
            if values is None or values.ndim != 1 or any(a is None or a.dtype.kind != "i" or a.dtype.itemsize not in (4, 8) for a in (indices, indptr)):
                raise ValueError("saved sparse arrays require values and signed integer indices")
            _array_layout(values, values.shape, manifest.dtype, manifest.byte_order)
            _array_layout(indices, values.shape, indices.dtype, manifest.byte_order)
            _array_layout(indptr, (dimension + 1,), indptr.dtype, manifest.byte_order)
            if indptr[0] != 0 or indptr[-1] != values.size or sum(a.nbytes for a in arrays) != manifest.payload_bytes:
                raise ValueError("saved sparse array lengths differ from their declaration")
            # An empty matrix of the declared shape and value dtype receives
            # the three mapped arrays, so SciPy's constructor neither checks
            # nor converts the saved indices.
            native = classes[data["sparse_type"]](tuple(data["shape"]), dtype=values.dtype)
            native.data, native.indices, native.indptr = arrays
        elif representation == "pauli":
            qubits = data["num_qubits"]
            if (type(qubits) is not int or qubits < 1 or dimension.bit_length() - 1 != qubits
                    or dimension & (dimension - 1) or manifest.shape is None or len(manifest.shape) != 1
                    or manifest.dtype != "uint64[x,z]+complex128"):
                raise ValueError("saved Pauli width or encoding differs from its declaration")
            masks_shape = (manifest.shape[0], (qubits + 63) // 64)
            arrays = {name: self.read_array(data[name]) for name in ("x", "z", "coefficients")}
            for name in ("x", "z"):
                _array_layout(arrays[name], masks_shape, "uint64", manifest.byte_order)
            _array_layout(arrays["coefficients"], manifest.shape, "complex128", manifest.byte_order)
            if sum(a.nbytes for a in arrays.values()) != manifest.payload_bytes:
                raise ValueError("saved Pauli storage size differs from its declaration")
            native = object.__new__(PauliTerms)
            object.__setattr__(native, "num_qubits", qubits)
            for name in ("x", "z", "coefficients"):
                object.__setattr__(native, name, arrays[name])
        else:
            raise ValueError(f"execution archive has no saved operator format for {representation}")
        return OperatorInput._from_admitted(manifest, data["structure"], native)

    def write_problem(self, problem):
        """Save a Problem, writing each of its numerical inputs once, and return its description.

        Symbolic expressions are saved as descriptions and never evaluated.

        Args:
            problem (ProblemRecord): The Problem.

        Returns:
            description (dict): JSON description that `read_problem` reads.
        """
        from nwqlib.operators.inputs import OperatorInput
        from nwqlib.problems.inputs import StateInput
        fields = problem.to_record()
        inputs = {}
        for name in type(problem).model_fields:
            value = getattr(problem, name)
            if isinstance(value, OperatorInput):
                inputs[name] = dict(kind="operator", data=self.write_operator("problem-" + name, value))
            elif isinstance(value, StateInput):
                inputs[name] = dict(kind="state", data=self.write_state("problem-" + name, value))
        return dict(fields=fields, inputs=inputs)

    def read_problem(self, data):
        """Rebuild a built-in Problem from its saved description, with its inputs restored.

        The Problem class is chosen from a fixed table of the built-in kinds,
        so saved text never selects an arbitrary class.

        Args:
            data (dict): The description that `write_problem` returned.

        Returns:
            problem (ProblemRecord): The Problem.
        """
        from nwqlib.problems import records
        classes = {cls.model_fields["kind"].default: cls for cls in (
            records.Eigenproblem, records.LinearDynamics, records.LinearSystem,
            records.Expectation, records.SpectralEstimation, records.Optimization)}
        fields = dict(data["fields"])
        for name, item in data["inputs"].items():
            fields[name] = self.read_operator(item["data"]) if item["kind"] == "operator" else self.read_state(item["data"])
        return classes[fields["kind"]].model_validate(fields)

    def write_output(self, output):
        """Save an output's fields, and its observable in its own representation when it has one.

        Args:
            output (OutputRecord): The output.

        Returns:
            description (dict): JSON description that `read_output` reads.
        """
        data = dict(fields=output.model_dump(mode="json", exclude_computed_fields=True))
        if hasattr(output, "observable"):
            data["observable"] = self.write_operator("output-observable", output.observable)
        return data

    def read_output(self, data):
        """Rebuild a built-in output from its saved description.

        The output class is chosen from a fixed table of the built-in kinds,
        as in `read_problem`.

        Args:
            data (dict): The description that `write_output` returned.

        Returns:
            output (OutputRecord): The output.
        """
        from nwqlib.problems import records
        classes = {cls.model_fields["kind"].default: cls for cls in (
            records.Eigenvalue, records.Eigenphase, records.NormalizedExpectation, records.QuadraticForm,
            records.NormSquared, records.Samples, records.Solution, records.StateVector, records.OptimizationCandidate)}
        fields = dict(data["fields"])
        if "observable" in data:
            fields["observable"] = self.read_operator(data["observable"])
        return classes[fields["kind"]].model_validate(fields)

    @staticmethod
    def read_plan(data, *, problem, method, output, reconstruction=None):
        """Rebuild a Plan from its JSON description and the Problem, Method and output already restored.

        Args:
            data (dict): The description returned by `Plan.to_record()`.
            problem (ProblemRecord): The restored Problem.
            method (Method): The restored Method.
            output (OutputRecord): The restored output.
            reconstruction (object | None): The Method's restored
                interpretation data, which replaces the saved one when given.

        Returns:
            plan (Plan): The Plan. A Method that binds circuit blocks binds
                them with `plan._bind(...)` before returning the Plan from
                `load_archive`.
        """
        from nwqlib.core.planning import Plan
        fields = dict(data)
        fields.update(problem=problem, method=method, output=output)
        if reconstruction is not None:
            fields["reconstruction"] = reconstruction
        return Plan.model_validate(fields)


def method_class(name, supplied=None):
    """Return the Method class that a saved name denotes.

    The class is the explicitly supplied one, which must match the name, or a
    built-in Method listed in ``_METHOD_OWNERS``. A saved name outside that
    table is never imported.
    """
    from importlib import import_module
    if supplied is not None:
        cls = supplied if isinstance(supplied, type) else type(supplied)
        if f"{cls.__module__}.{cls.__qualname__}" != name:
            raise ValueError("explicit Method differs from the saved implementation")
        return cls
    from nwqlib.algorithms import _METHOD_OWNERS
    known = {f"nwqlib.algorithms.{owner}.{name}" for name, owner in _METHOD_OWNERS.items()}
    if name not in known:
        raise ValueError("saved external Method requires its explicit implementation")
    module, _, attribute = name.rpartition(".")
    return getattr(import_module(module), attribute)


def save_plan(plan, files):
    """Save selected data through the Method's hook, with its class name."""
    writer = getattr(plan.method, "save_archive", None)
    if not callable(writer):
        raise TypeError(f"{type(plan.method).__name__} requires callable save_archive(plan, files) "
                        "for selected-data persistence; see docs/algorithm_protocol.md")
    return dict(method=f"{type(plan.method).__module__}.{type(plan.method).__qualname__}",
                selected=writer(plan, files))


def load_plan(data, files, *, method=None):
    """Restore the selected Plan through the Method's hook, without planning again.

    The Method restores its selected construction data from the saved files.
    """
    cls = method_class(data["method"], method)
    reader = getattr(cls, "load_archive", None)
    if not callable(reader):
        raise TypeError(f"{cls.__name__} requires callable load_archive(saved, files) "
                        "for selected-data persistence; see docs/algorithm_protocol.md")
    plan = reader(data["selected"], files)
    return plan
