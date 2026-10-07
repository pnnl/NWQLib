"""An explicit snapshot of selected data, durable execution and live caches.

A Run folder holds ``run.json`` (format, the name of the selection file and the
limits in force when the folder was written), the selection file, the journal
``run.sqlite`` with its lock file, the Plan's selected input files, and cache
files written through ``ArchiveFiles`` (QPY for native instructions and circuits,
NPY arrays, and JSON such as a bound noise model with NPY leaves). A Method's own
cache hooks choose among these formats. Cache rows in the journal point to cache
files by name.

Checkpointing a cache follows a write, commit, delete order:

1. Change tracking. ``_CacheEntries`` records which keys of the definition,
   lowered Aer, handle, backend and method dictionaries changed since the last
   commit, and the previous value of each.
2. Write new files. ``cache_record`` serializes only the changed entries, each to a
   name with no existing file (``ArchiveFiles._unused_name``), opened in
   exclusive-create mode. A committed row never points to a file that is being
   rewritten. An object already written in this session reuses its file.
3. Commit rows. The new and deleted cache rows join the caller's journal
   transaction (``write_caches``).
4. Delete files. After the commit, ``finish_caches`` deletes files touched by
   the delta that no row references any more. ``run.json``, the selection file
   and the files the selection references are registered as permanent, and
   pruning skips them.

Interruption between steps 2 and 3 leaves unpublished files, and between steps
3 and 4 can leave superseded files. Reopening deletes unreferenced files in the
reserved cache namespace under the controller lock, before it counts the stored
file bytes. Input files and committed cache dependencies remain protected. The
recovery contract concerns process interruption and does not cover power loss.
A failed commit deletes the new files, refunds their byte charge and restores the
previous dictionary values.

Loading restores these rows as they are, registering a prepared handle's
circuit file and decoding it at the handle's first native use. It does not plan,
lower, transpile, simulate or analyze. Published arrays are read from the journal
at their first use, once per payload.
"""

from contextlib import closing
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sqlite3

from nwqlib._choice_archive import ArchiveFiles


_ABSENT = object()
# The temporary name under which _write_manifest writes run.json before renaming it.
_PARTIAL_MANIFEST = "run.json.partial"
# The journal and its lock and rollback files. max_data_bytes charges the journal's contents, its rows,
# published arrays and payloads, as the Run records them, not the size of these files, so the stored file
# bytes of a Run folder (stored_file_bytes) leave them out.
_JOURNAL_FILES = frozenset({"run.sqlite", "run.sqlite.lock", "run.sqlite-journal"})
# Backend-context keys left out of the frontier row. Aer lowered blocks have
# their own "aer" cache rows and a bound noise model its own JSON file
# (_save_noise). The other keys hold live objects that cannot be saved: IBM's
# service, device and job handles and the local NWQ-Sim child processes. After
# reopening, an adapter works from the saved locator instead.
_TRANSIENT_BACKEND = {"aer_preparations", "noise_model", "service", "device", "jobs", "processes"}


class _CacheEntries(dict):
    """Record changed native-cache keys at their ordinary dictionary mutations.

    Each mutation stores the key in the Run's shared ``changes`` map and, on its
    first change since the last commit, the previous value in ``previous``. A
    checkpoint then serializes only changed keys, without walking the whole cache,
    and a failed commit restores the previous values. Loading clears the change
    records after it restores the rows, so restored entries are not written again.
    Nested dictionaries inside the method context are wrapped too, because a
    Method can keep keyed caches there (for example compiled query gates).

    ``update``, ``setdefault`` and ``pop`` are overridden because CPython's
    ``dict`` implements them without calling ``__setitem__`` or
    ``__delitem__``. Without the overrides those mutations would not be
    recorded. ``clear``, ``popitem`` and ``|=`` are not overridden, so a change
    made through them would not be saved.
    """

    def __init__(self, values=(), *, prefix, changes):
        super().__init__(values)
        self.prefix, self.changes, self.previous = prefix, changes, {}
        if prefix == ("method",):
            for key, value in tuple(self.items()):
                if isinstance(value, dict):
                    dict.__setitem__(self, key, _CacheEntries(value, prefix=(*prefix, key), changes=changes))

    def __setitem__(self, key, value):
        """Store ``value`` and record the key, keeping its value from before the first change.

        Storing the object already held under ``key`` is not a change and is
        not recorded.
        """
        old = dict.get(self, key, _ABSENT)
        if old is value:
            return
        if key not in self.previous:
            self.previous[key] = old
        # A nested method dictionary is a keyed cache (QLS keeps its query gates in
        # one), so its own mutations must be tracked as well.
        if self.prefix == ("method",) and isinstance(value, dict) and not isinstance(value, _CacheEntries):
            value = _CacheEntries(value, prefix=(*self.prefix, key), changes=self.changes)
        dict.__setitem__(self, key, value)
        self.changes[(*self.prefix, key)] = (self, key)

    def __delitem__(self, key):
        old = dict.__getitem__(self, key)
        if key not in self.previous:
            self.previous[key] = old
        dict.__delitem__(self, key)
        self.changes[(*self.prefix, key)] = (self, key)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, default=_ABSENT):
        if key not in self:
            if default is _ABSENT:
                raise KeyError(key)
            return default
        value = self[key]
        del self[key]
        return value


def bind_caches(run):
    """Initialize the Run's cache bookkeeping and wrap its caches in ``_CacheEntries``.

    The definition, handle and backend caches are wrapped so that later
    mutations are recorded for the next checkpoint.
    """
    state = run._state
    changes = state["cache_changes"] = {}
    state["cache_rows"], state["cache_file_refs"], state["cache_keys"] = {}, {}, {}
    state["cache_method_dirty"] = state["cache_frontier_dirty"] = False
    state["cache_next_key"] = 0
    state["cache_row_refs"], state["cache_permanent_files"] = {}, set()
    state["definitions"] = tuple(_CacheEntries(prefix=("definition", group), changes=changes) for group in (0, 1))
    state["handles"] = _CacheEntries(prefix=("handle",), changes=changes)
    state["backend_context"] = _CacheEntries(prefix=("backend",), changes=changes)


@dataclass(frozen=True)
class _CacheDelta:
    """One checkpoint's cache rows and the bookkeeping to commit or undo them.

    ``records`` and ``deleted`` join the journal transaction. ``changes`` are the
    tracked mutations the delta covers, ``keys`` the new row identities,
    ``references`` the files each written row points to, and the two flags say
    whether the method and frontier rows were rewritten.
    """

    records: tuple = ()
    deleted: tuple = ()
    changes: tuple = ()
    keys: tuple = ()
    references: tuple = ()
    method_dirty: bool = False
    frontier_dirty: bool = False


def validate_provenance(plan, forecast, allocation):
    """Check supplied resource provenance without evaluating or selecting it."""
    if allocation is not None:
        from nwqlib.backends.profiles import Allocation
        if type(allocation) is not Allocation:
            raise TypeError("allocation must be the supplied Allocation")
    if forecast is not None:
        from nwqlib.backends.assessment import PlanEstimate
        if type(forecast) is not PlanEstimate:
            raise TypeError("forecast must be the selected PlanEstimate")
        forecast.validate_plan(plan)
        if forecast.allocation != allocation:
            raise ValueError("run allocation differs from its original forecast")


def load_provenance(data):
    """Rebuild the saved PlanEstimate and Allocation records, or None for each absent one.

    Only record validation runs. No profile is evaluated again.
    """
    forecast, allocation = data["forecast"], data["allocation"]
    if forecast is not None:
        from nwqlib.backends.assessment import PlanEstimate
        forecast = PlanEstimate.model_validate(forecast)
    if allocation is not None:
        from nwqlib.backends.profiles import Allocation
        allocation = Allocation.model_validate(allocation)
    return forecast, allocation


def _frontier(run, files):
    """The active native specializations and backend data, never preparation or definition
    history. Each native preparation replaces its specializations, so this row stays small.
    """
    state = run._state
    backend = state["backend_context"]
    from nwqlib.backends.connection import AerBackend
    noise = run.backend._bound_noise_model() if isinstance(run.backend, AerBackend) else None
    return dict(kind="frontier",
        specializations=[(key, files.write_instruction(f"specialization-{index}.qpy", value))
                         for index, (key, value) in enumerate(state["specializations"].items())],
        backend_data={key: value for key, value in backend.items() if key not in _TRANSIENT_BACKEND},
        noise_model=_save_noise(noise, files), noise_model_cached="noise_model" in backend)


def _method_frontier(run, files):
    """The live method context in its own row, rewritten only when the Method's caches change."""
    state, plan = run._state, run.plan
    context = state["method_context"]
    if context is not None:
        writer = getattr(plan.method, "save_run_context", None)
        if not callable(writer):
            raise TypeError("this method has no format for its live run context")
        context = writer(context, files)
    return dict(kind="method", method_context=context)


def _cache_entry(token, value, files, name):
    """Serialize one native cache entry: a definition, a lowered Aer pair or a handle.

    Definitions and lowered Aer sources are single instructions stored as QPY. A
    prepared handle stores its native circuit and backend metadata, which is
    enough to restore it without lowering or transpiling.
    """
    kind = token[0]
    if kind == "definition":
        return dict(kind=kind, group=token[1], key=token[2],
                    file=files.write_instruction(name + ".qpy", value))
    if kind == "aer":
        source, lowered = value
        return dict(kind=kind, target=token[1],
                    source=files.write_instruction(name + "-source.qpy", source),
                    lowered=files.write_instruction(name + "-lowered.qpy", lowered))
    if kind == "handle":
        native = value._native
        return dict(kind=kind, identity=token[1],
                    circuit=files.write_circuit(name + ".qpy", native.circuit), metadata=native.metadata)
    raise ValueError("unknown native cache owner")


def _copy_cache_files(run, files):
    """Copy the files that a durable Run's committed cache rows name into ``files``.

    This applies when every cache change is committed, so the rows in the
    journal describe the live caches. Each file whose name has the reserved
    cache prefix is hard-linked, or copied where a link is not possible,
    under the same name and charged to the new folder's allowance, so the
    rows of the journal copy stay valid unchanged. A row that names a file
    outside the cache namespace needs that file among the ones the new
    folder's selection wrote. Returns False, having copied nothing, when a
    change is uncommitted or such a file is absent. The caller then writes
    the caches again.
    """
    from nwqlib._choice_archive import CACHE_FILE_PREFIX
    state = run._state
    source = state.get("archive_files")
    if (source is None or state["cache_changes"] or state["cache_method_dirty"] or state["cache_frontier_dirty"]
            or not {"frontier", "method"} <= set(state["cache_rows"])):
        return False
    names = sorted({name for refs in state["cache_row_refs"].values() for name in refs})
    cached = [name for name in names if name.startswith(CACHE_FILE_PREFIX)]
    if any(not name.startswith(CACHE_FILE_PREFIX) and name not in files._written_files for name in names):
        return False
    for name in cached:
        origin, target = source.file(name), files.file(name)
        files.reserve(origin.stat().st_size)
        try:
            os.link(origin, target)
        except OSError:
            shutil.copyfile(origin, target)
        files._written_files.add(name)
    return True


def _save_caches(run, files):
    """Serialize the Run's caches once each as cache rows of a Run copy.

    The rows hold the frontier, the method context, the definition and Aer
    caches, and for an Aer backend its handles. ``save`` uses it for an
    in-memory Run, or when the cache files cannot be copied as they are
    (``_copy_cache_files``).
    """
    rows = [("frontier", _frontier(run, files)), ("method", _method_frontier(run, files))]
    for group, cache in enumerate(run._state["definitions"]):
        for key, value in cache.items():
            name = "entry-" + str(len(rows))
            rows.append((name, _cache_entry(("definition", group, key), value, files, name)))
    for target, owner in run._state["backend_context"].get("aer_preparations", {}).items():
        for key, value in owner._blocks.items():
            name = "entry-" + str(len(rows))
            rows.append((name, _cache_entry(("aer", target, key), value, files, name)))
    from nwqlib.backends.connection import AerBackend
    if isinstance(run.backend, AerBackend):
        for key, value in run._state["handles"].items():
            name = "entry-" + str(len(rows))
            if value._native is None:
                value._restore_native()
            rows.append((name, _cache_entry(("handle", key), value, files, name)))
    return tuple(("cache", key, value) for key, value in rows)


def _noise_model_data(model):
    """The saved form of an Aer noise model: ``NoiseModel.to_dict()`` plus its sorted basis gates.

    ``to_dict`` keeps the errors and where they apply but not ``basis_gates``,
    which select the Aer target that a circuit is lowered to, so they are
    stored beside the errors for ``_load_noise_model``. Without them a
    reopened Run would lower its later settings to Aer's default basis plus
    the gates its errors name. When the original model had other basis gates,
    that is a different native context from the settings prepared before the
    save, and an analysis that needs one native context across settings, such
    as Expectation, would reject the Run (docs/run_archives.md).
    """
    return {**model.to_dict(), "basis_gates": sorted(model.basis_gates)}


def _load_noise_model(data):
    """Restore Aer basis gates, error circuits and associations without deprecated from_dict.

    The model starts from the saved basis gates (``_noise_model_data``). Adding
    an error also adds its gate to the basis, as it did when the original
    model received that error, so the rebuilt basis equals the saved one.
    Gate constructors consume the saved parameters. Converting just a gate name
    to a fixed unitary would discard parameterized rotations. No new channel or
    full-system matrix is constructed beyond QuantumError's own admission.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction
    from qiskit.circuit.library import PauliGate, UnitaryGate, get_standard_gate_name_mapping
    from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError

    standards = get_standard_gate_name_mapping()
    model = NoiseModel(basis_gates=data["basis_gates"])
    for saved in data["errors"]:
        kind = saved["type"]
        if kind == "qerror":
            circuits = []
            for branch in saved["instructions"]:
                width = max((max(inst["qubits"], default=-1) for inst in branch), default=-1) + 1
                circuit = QuantumCircuit(width)
                for inst in branch:
                    name, params = inst["name"], inst.get("params", [])
                    if name == "kraus":
                        gate = Instruction(name, len(inst["qubits"]), 0, params)
                    elif name == "unitary":
                        gate = UnitaryGate(*params)
                    elif name == "pauli":
                        gate = PauliGate(*params)
                    elif name in standards and name not in {"measure", "delay"}:
                        template = standards[name]
                        gate = template.base_class(*params)
                    else:
                        raise ValueError(f"unsupported saved noise instruction: {name}")
                    if inst.get("label") is not None:
                        gate = gate.to_mutable()
                        gate.label = inst["label"]
                    circuit.append(gate, inst["qubits"])
                circuits.append(circuit)
            error = QuantumError(list(zip(circuits, saved["probabilities"], strict=True)))
            # Aer exposes the persisted identifier through a read-only property.
            error._id = saved.get("id") or error.id
            operations = (saved["operations"],)
            local, global_error = model.add_quantum_error, model.add_all_qubit_quantum_error
        elif kind == "roerror":
            error = ReadoutError(saved["probabilities"])
            operations = ()
            local, global_error = model.add_readout_error, model.add_all_qubit_readout_error
        else:
            raise ValueError(f"unsupported saved noise error type: {kind}")
        if saved.get("gate_qubits") is None:
            global_error(error, *operations)
        else:
            for qubits in saved["gate_qubits"]:
                local(error, *operations, qubits)
    return model


def _load_caches(run, rows, files):
    """Restore every saved cache row as saved, with no recomputation.

    Definitions, lowered Aer sources, the active frontier and the method
    context come back from their files. A prepared handle comes back with its
    receipt, readout items and setting, and its circuit file is registered
    without being read: the circuit is decoded and the backend's native
    object rebuilt at the handle's first native use (submission or
    inspection, ``PreparedHandle._restore_native``), as after
    ``Run.release_native``. A saved Aer noise model is
    rebuilt and bound to a copy of the backend configuration, which becomes
    ``run.backend``, so the backend object that the caller passed to ``load``
    keeps the model it holds. A missing file rejects at load, and invalid QPY
    rejects when it is read, so the Run is not rebuilt around the gap. The
    change records are cleared at the end, so the next checkpoint writes only
    later changes.
    """
    state, plan = run._state, run.plan
    from nwqlib._prepared_execution import PreparedHandle, _admit_readout
    lazy = []
    for key, data, _ in rows:
        kind = data["kind"]
        token = None
        if kind == "frontier":
            state["specializations"] = {
                (k[0], tuple(tuple(argument) for argument in k[1])): files.read_instruction(name)
                for k, name in data["specializations"]}
            state["backend_context"].update(data["backend_data"])
            if data["noise_model"] is not None:
                # AerBackend binds a caller-owned model without copying it, so the
                # rebuilt model goes to the Run's own copy of the configuration.
                model = _load_noise_model(files.read_numpy_json(data["noise_model"]))
                backend = run.backend.model_copy()
                backend._noise_model = model
                object.__setattr__(run, "backend", backend)
                files._noise_models = {id(model): (model, data["noise_model"])}
                if data["noise_model_cached"]:
                    state["backend_context"]["noise_model"] = model
        elif kind == "method":
            if data["method_context"] is not None:
                reader = getattr(plan.method, "_load_live_run_context", None)
                context = (reader(data["method_context"], files, run.observations) if callable(reader)
                           else plan.method.load_run_context(data["method_context"], files))
                if isinstance(context, dict):
                    context = _CacheEntries(context, prefix=("method",), changes=state["cache_changes"])
                state["method_context"] = context
        elif kind == "definition":
            token = (kind, data["group"], data["key"])
            dict.__setitem__(state["definitions"][data["group"]], data["key"], files.read_instruction(data["file"]))
        elif kind == "aer":
            from nwqlib.backends.qiskit_aer import _AerPreparation
            owners = state["backend_context"].setdefault("aer_preparations", {})
            target = data["target"]
            if target not in owners:
                owner = owners[target] = _AerPreparation()
                owner._blocks = _CacheEntries(prefix=("aer", target), changes=state["cache_changes"])
            source = files.read_instruction(data["source"])
            token = (kind, target, id(source))
            dict.__setitem__(owners[target]._blocks, id(source), (source, files.read_instruction(data["lowered"])))
        elif kind == "handle":
            record = run.prepared_artifact(data["identity"])
            # The circuit file is registered now and decoded at the handle's
            # first native use (PreparedHandle._restore_native).
            files.read_path(data["circuit"])
            # Select the receipt's construction once and resolve the readout
            # from that Experiment and construction, because
            # ``Realization.resolved_observation`` would select it again.
            experiment, construction = record.realization._selected_construction(plan)
            if experiment.batch is None:
                setting, spec = experiment.setting, experiment.observation
            else:
                from nwqlib._prepared_execution import _named_admission_refusal
                from nwqlib.ir.validation import AdmissionStepsExceeded, _Admission
                admission = _Admission(construction.program)
                try:
                    admission.admitted().require_ready()
                    setting, spec = experiment._resolved_observation(admission, construction_id=construction.content_id)
                except AdmissionStepsExceeded as error:
                    raise _named_admission_refusal(plan, error, "restoration") from error
            items = _admit_readout(spec, width=sum(len(reg.bits) for reg in record.quantum_layout),
                classical_width=sum(len(reg.bits) for reg in record.classical_layout), run=run)
            handle = PreparedHandle._make(record, record.realization, None, items, setting,
                                          construction.program.bindings)
            state["handles"][record.content_id] = handle
            lazy.append((handle, data["circuit"], data["metadata"]))
            token = (kind, record.content_id)
        else:
            raise ValueError("unsupported saved cache kind")
        if token is not None:
            state["cache_keys"][token] = key
        state["cache_rows"][key] = data
    # A handle's native object is rebuilt with the Run's final backend
    # configuration, which a saved noise model can replace above.
    for handle, name, metadata in lazy:
        object.__setattr__(handle, "_saved_native", (run.backend, run.directory, name, metadata))
    state["cache_next_key"] = 1 + max((int(key[6:]) for key in state["cache_rows"] if key.startswith("entry-")), default=-1)
    for cache, changed_key in state["cache_changes"].values():
        cache.previous.pop(changed_key, None)
    state["cache_changes"].clear()
    state["cache_method_dirty"] = state["cache_frontier_dirty"] = False



def _save_noise(noise, files):
    """Name of the JSON file that stores ``noise``, writing it on first use only.

    The file is keyed by the model object's identity, so every frontier row of
    one Run points to the same file instead of writing the model again.
    """
    if noise is None:
        return None
    saved = getattr(files, "_noise_models", {})
    if id(noise) not in saved:
        saved[id(noise)] = (noise, files.write_numpy_json("noise-model.json", _noise_model_data(noise)))
        files._noise_models = saved
    return saved[id(noise)][1]


def _write_manifest(run, files):
    """Write the selection to its own file, then the small ``run.json`` that names it.

    The selection holds the Method name and the record returned by its archive
    hook, including the Plan and selected data. The manifest holds the format,
    selection filename and limits. A copy records current limits, while its
    journal header keeps the initial limits and the original Plan identity.
    Return the selection filename and its record for ``archive_selection``.

    ``ArchiveFiles.write_json`` charges each part of a file before writing
    it, so a ``max_data_bytes`` refusal in the middle of a file leaves the
    parts written so far. ``run.json`` is therefore written as
    ``run.json.partial`` and then renamed with ``os.replace``, which
    replaces a file atomically within one filesystem. A refusal or an
    interruption during the write leaves no ``run.json``, and ``load`` then
    reports a Run whose creation stopped before its manifest. The partial
    file stays, as it would after an interruption, and counts among the
    folder's stored file bytes (``stored_file_bytes``).
    """
    from nwqlib._choice_archive import save_plan
    from nwqlib._run_journal import RUN_FORMAT
    selection = save_plan(run.plan, files)
    name = files._unused_name("selection.json")
    files.write_json(name, selection)
    files.write_json(_PARTIAL_MANIFEST, dict(format=RUN_FORMAT, selection=name,
                                             limits=run.limits.model_dump(mode="json")))
    os.replace(files.file(_PARTIAL_MANIFEST), files.file("run.json"))
    files._written_files.discard(_PARTIAL_MANIFEST)
    files._written_files.add("run.json")
    return dict(file=name, selection=selection)


def initialize(run):
    """Save the selected input once before a durable backend can start work.

    The Method's archive hook writes the selected data, the selection file
    (normally ``selection.json``) holds the selection that names it, and
    ``run.json`` names the selection file. All are registered as permanent
    files, which cache pruning skips. Their bytes count toward
    ``max_data_bytes``.
    """
    files = ArchiveFiles(run.directory, None)
    _bind_capacity(run, files)
    run._state["archive_files"] = files
    run._state["archive_selection"] = _write_manifest(run, files)
    _initialize_file_refs(run, files)
    return files


def _bind_capacity(run, files):
    """Charge every archive file write to the Run's stored-data total before it happens.

    The Run folder is opened without an allowance of its own, so
    ``max_data_bytes``, which also counts the journal and the published
    arrays, is the one limit on these writes.
    """
    def reserve(size):
        run.check_capacity(data_bytes=size)
        run._state["external_bytes"] += size
    files._on_reserve = reserve


def _references(value, files):
    """Archive file names a saved row points to, including files those files depend on."""
    names, pending = set(), [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif isinstance(item, str) and item in files._written_files and item not in names:
            names.add(item)
            pending.extend(files._dependencies.get(item, ()))
    return names


def _initialize_file_refs(run, files):
    """Count cache-row references per file and register the selection's files as permanent."""
    state = run._state
    refs = state["cache_file_refs"]
    refs.clear()
    state["cache_row_refs"] = {}
    permanent = _references(state["archive_selection"], files) | {"run.json"}
    state["cache_permanent_files"] = permanent
    for key, row in state["cache_rows"].items():
        names = state["cache_row_refs"][key] = _references(row, files)
        for name in names:
            refs[name] = refs.get(name, 0) + 1


def restore_caches(run, files, selection):
    """Load committed cache owners, then reclaim their unpublished files.

    The caller holds the journal's exclusive lock. Reading the current owners
    first registers all referenced NPY/QPY files, including ndarray leaves
    of SDK JSON, through ArchiveFiles.read_path. A lazy loader must register
    dependencies now, without reading their payloads. Only the reserved cache namespace is collectible. Orphan
    contents are never parsed, and immutable Plan inputs remain protected.
    The sizes of the files left in the folder, apart from the journal, become
    ``external_bytes``, to which later writes add (``_bind_capacity``).
    """
    from nwqlib._choice_archive import CACHE_FILE_PREFIX

    state = run._state
    state["archive_files"], state["archive_selection"] = files, selection
    rows = list(state["journal"].rows("cache"))
    if rows:
        _load_caches(run, rows, files)
    _initialize_file_refs(run, files)
    for path in files.path.iterdir():
        if path.is_file() and path.name.startswith(CACHE_FILE_PREFIX) and path.name not in files._written_files:
            path.unlink()
    state["external_bytes"] = stored_file_bytes(files.path)


def stored_file_bytes(path):
    """Return the total size of the files in the Run folder ``path``, apart from the journal files.

    These are the files that ``ArchiveFiles`` charges to the Run's
    stored-data total when it writes them (``_bind_capacity``).
    ``restore_caches`` counts a reopened Run's stored files this way, and
    ``qhd._durable.closed_trace`` the files of a Run whose creation raised
    before it committed its journal header.
    """
    return sum(item.stat().st_size for item in Path(path).iterdir()
               if item.is_file() and item.name not in _JOURNAL_FILES)


def cache_record(run):
    """Serialize only changed owner keys and the changed active frontier.

    New files are written here, before the journal commit, under names that no
    committed row uses. An in-memory Run has no archive files and returns the
    tracked changes only, so ``finish_caches`` can clear or undo them. If
    serialization fails, ``finish_caches`` deletes the new files and restores
    the dictionaries before the error propagates.
    """
    state, files = run._state, run._state.get("archive_files")
    changes = tuple(state["cache_changes"].items())
    if files is None:
        return _CacheDelta(changes=changes, method_dirty=state["cache_method_dirty"],
                           frontier_dirty=state["cache_frontier_dirty"])
    method = (state["cache_method_dirty"] or "method" not in state["cache_rows"]
              or any(token[0] == "method" for token, _ in changes))
    frontier = (state["cache_frontier_dirty"] or "frontier" not in state["cache_rows"]
                or any(token[0] == "backend" for token, _ in changes))
    if not changes and not frontier and not method:
        return _CacheDelta()
    files._begin_cache_files()
    records, deleted, newkeys = [], [], []
    try:
        if frontier:
            records.append(("cache", "frontier", _frontier(run, files)))
        if method:
            records.append(("cache", "method", _method_frontier(run, files)))
        for token, (cache, key) in changes:
            if token[0] in {"method", "backend"}:
                continue
            if token[0] == "handle":
                from nwqlib.backends.connection import AerBackend
                if not isinstance(run.backend, AerBackend):
                    continue
            identity = state["cache_keys"].get(token)
            if key not in cache:
                if identity is not None:
                    deleted.append(("cache", identity))
                continue
            if identity is None:
                identity = "entry-" + str(state["cache_next_key"] + len(newkeys))
                newkeys.append((token, identity))
            records.append(("cache", identity, _cache_entry(token, cache[key], files, identity)))
        refs = tuple((key, _references(row, files)) for _, key, row in records)
        return _CacheDelta(tuple(records), tuple(deleted), changes, tuple(newkeys), refs, method, frontier)
    except BaseException:
        finish_caches(run, _CacheDelta(changes=changes), committed=False)
        raise


def finish_caches(run, delta, *, committed):
    """Complete a cache delta after its journal transaction succeeded or failed.

    After a failure, the delta's new files are deleted, their byte charge is
    refunded and each changed dictionary entry, including the parameter
    specializations, returns to its value before the delta. After a success, the
    per-file reference counts move to the new rows, and only files touched by this
    delta that no row references any more are deleted. Deleting files only after
    the commit means a process crash can leave extra files but never a row without
    its file.
    """
    state, files = run._state, run._state.get("archive_files")
    newfiles, size = ((), 0) if files is None or files._pending_files is None else files._finish_cache_files(committed=committed)
    if not committed:
        state["external_bytes"] -= size
        for _, (cache, key) in reversed(delta.changes):
            previous = cache.previous.pop(key, _ABSENT)
            if previous is _ABSENT:
                dict.pop(cache, key, None)
            else:
                dict.__setitem__(cache, key, previous)
        state["cache_changes"].clear()
        if "cache_previous_specializations" in state:
            state["specializations"] = state.pop("cache_previous_specializations")
        return
    refs, oldrefs = state["cache_file_refs"], state["cache_row_refs"]
    candidates = set(newfiles)
    for key in {key for _, key in delta.deleted} | {key for _, key, _ in delta.records}:
        for name in oldrefs.pop(key, ()):
            refs[name] -= 1
            candidates.add(name)
        state["cache_rows"].pop(key, None)
    for key, names in delta.references:
        oldrefs[key] = names
        for name in names:
            refs[name] = refs.get(name, 0) + 1
    for _, key, row in delta.records:
        state["cache_rows"][key] = row
    state["cache_keys"].update(delta.keys)
    state["cache_next_key"] += len(delta.keys)
    for token, (cache, key) in delta.changes:
        if key not in cache:
            state["cache_keys"].pop(token, None)
        cache.previous.pop(key, None)
        state["cache_changes"].pop(token, None)
    if delta.method_dirty:
        state["cache_method_dirty"] = False
    if delta.frontier_dirty:
        state["cache_frontier_dirty"] = False
    state.pop("cache_previous_specializations", None)
    # Visit only files touched by this delta. A zero-reference cache file is
    # deleted after its new row pointers committed. Immutable input files stay.
    for name in candidates:
        if refs.get(name, 0) or name in state["cache_permanent_files"]:
            continue
        path = files.file(name)
        file_size = path.stat().st_size
        path.unlink()
        state["external_bytes"] -= file_size
        files._written_files.discard(name)
        files._dependencies.pop(name, None)
        files._forget_objects(name)
        refs.pop(name, None)


def write_caches(run, records=(), *, deleted=(), **kwargs):
    """Commit ``records`` together with the pending cache delta in one Run transition."""
    delta = cache_record(run)
    try:
        encoded = run._write((*records, *delta.records), deleted=(*deleted, *delta.deleted), **kwargs)
    except BaseException:
        finish_caches(run, delta, committed=False)
        raise
    finish_caches(run, delta, committed=True)
    return encoded


def release_native(run):
    """Keep the current frontier. Release only saved, fully collected handles."""
    state = run._state
    files = state.get("archive_files")
    released, kept = [], []
    active, completed = set(), set()
    for event in state["events"]:
        chunk = state["by_attempt"].get(event.attempt)
        # A trajectory attempt is collected when all of its point chunks are.
        chunks = chunk if isinstance(chunk, tuple) else (chunk,)
        if event.status != "completed" or chunk is None or any(item.acquisition_key not in state["chunks"]
                                                               for item in chunks):
            active.add(event.prepared_id)
        else:
            completed.add(event.prepared_id)
    # The active controller may name a native handle for its next operation.
    pending = [run.checkpoint_state] if run.result is None else []
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif isinstance(item, str) and item in state["handles"]:
            active.add(item)
    if state["workflow_current"] is not None or state["publication_batch"] is not None:
        return dict(released=(), kept=tuple((key, "active execution frontier") for key in state["handles"]))
    for identity, handle in state["handles"].items():
        key = state["cache_keys"].get(("handle", identity))
        row = state["cache_rows"].get(key)
        if files is None or row is None or row.get("kind") != "handle":
            kept.append((identity, "no durable QPY snapshot"))
        elif identity in active or identity not in completed:
            kept.append((identity, "active, unsubmitted or uncollected consumer"))
        elif handle._native is None:
            kept.append((identity, "already released"))
        else:
            object.__setattr__(handle, "_saved_native",
                (run.backend, run.directory, row["circuit"], row["metadata"]))
            object.__setattr__(handle, "_native", None)
            files._forget_objects(row["circuit"])
            released.append(identity)
    return dict(released=tuple(released), kept=tuple(kept))



def result_record(result, run):
    """Return the saved row of a Run's Result.

    The row holds the Result, the number of limit amendments its trace holds
    and its stored-data byte count. The trace's amendments and limits must be
    the first amendments of the Run and the limits they produce.
    """
    trace = result.data.trace
    count = len(trace.limit_amendments)
    history = run.limit_amendments
    if count > len(history):
        raise ValueError("Result cap history is not a captured prefix of its Run")
    limits = history[count - 1].new if count else run._state["initial_limits"]
    if trace.limit_amendments != history[:count] or trace.limits != limits:
        raise ValueError("Result cap history is not a captured prefix of its Run")
    return dict(result=result, limit_amendment_count=count, data_bytes=trace.data_bytes)


def save(run, *, path):
    """Save this exact frontier. The copy of an in-memory execution gets its own journal.

    Saving is allowed only at a returned boundary, that is, after a public Run
    call has returned, with no outcome being published and no static item being
    prepared or submitted. A durable Run is copied with SQLite's backup API, which
    gives a consistent snapshot of the live journal. When its cache rows are
    committed, the files they name are copied (hard-linked where possible)
    under the same names and the rows stay as they are, so no cached QPY
    object is decoded or encoded again (``_copy_cache_files``). Otherwise the
    copy's cache rows are rewritten to point to newly written files, and
    released native handles are loaded again so that they can be written. An
    in-memory Run writes its header, limits, amendments, RNG, receipts, events,
    observations with their collection order, checkpoint, arrays and Result into
    the new journal in one transaction. On any failure the new folder is
    removed. The live Run's journal, files and counters are unchanged either
    way.

    Every file of the new folder is charged to a file-byte allowance of
    ``max_data_bytes``. The files that ``ArchiveFiles`` writes are charged before
    they are written. The journal copy of a durable Run is charged by its page
    count before the backup writes those pages. SQLite allocates the pages of a
    commit during the commit, so the pages that the commit of the rewritten
    cache rows adds are charged right after it, and the new journal of an
    in-memory Run is charged by its file size right after its one commit. A
    charge beyond the allowance removes the new folder like any other failure.
    """
    from nwqlib._run_journal import RUN_FORMAT, LocalJournal
    from nwqlib._prepared_execution import Run
    if not isinstance(run, Run):
        raise TypeError("save requires the actual Run")
    with run._lock:
        run._ensure_open()
        state = run._state
        if state["publication_batch"] is not None or state["workflow_current"] is not None:
            raise ValueError("save requires a returned execution frontier")
        files = ArchiveFiles(path, run.limits.max_data_bytes)
        files.path.mkdir()
        try:
            _write_manifest(run, files)
            journal = state["journal"]
            # A durable Run whose cache rows are committed copies the files
            # they name. Its journal copy keeps those rows unchanged.
            copied = journal is not None and _copy_cache_files(run, files)
            files._begin_cache_files()
            caches = () if copied else _save_caches(run, files)
            files._finish_cache_files(committed=True)
            # A durable run copies its existing journal. An in-memory run writes its
            # current frontier once, including the original limit and every amendment.
            if journal is not None:
                pages = journal.connection.execute("PRAGMA page_count").fetchone()[0]
                page_size = journal.connection.execute("PRAGMA page_size").fetchone()[0]
                files.reserve(pages * page_size)
                with closing(sqlite3.connect(files.file("run.sqlite"))) as target:
                    journal.connection.backup(target)
                target = LocalJournal(files.file("run.sqlite"), run.limits.max_data_bytes, create=False)
                try:
                    if copied:
                        target.commit((("limits", "current", run.limits),))
                    else:
                        previous = tuple(("cache", key) for key, _, _ in target.rows("cache"))
                        current = {key for _, key, _ in caches}
                        target.commit((*caches, ("limits", "current", run.limits)),
                                      deleted=tuple(row for row in previous if row[1] not in current))
                    # The pages this commit added are known only now.
                    files.reserve(max(0, target.path.stat().st_size - pages * page_size))
                finally:
                    target.close()
            else:
                target = LocalJournal(files.file("run.sqlite"), run.limits.max_data_bytes, create=True)
                try:
                    rows = [("header", "run", dict(format=RUN_FORMAT, run_id=run.run_id,
                        plan_id=run.plan.content_id,
                        backend=None if run.backend is None else run.backend.model_dump(mode="json"), limits=state["initial_limits"],
                        forecast=run.forecast, allocation=run.allocation)),
                        ("limits", "current", run.limits),
                        ("rng", "current", run.rng.snapshot()), *caches,
                        ("control", "current", dict(cancel_requested=run.cancel_requested,
                            termination_reason=state["termination_reason"]))]
                    for kind, values in (("prepared", state["prepared_artifacts"]),
                            ("preparation", state["preparation_charges"]), ("submission", state["submissions"]),
                            ("remote_preparation", state["remote_preparations"]), ("workflow_item", state["workflow_items"])):
                        rows.extend((kind, key, value) for key, value in values.items())
                    rows.extend(("event", event.attempt, event) for event in state["events"])
                    rows.extend(("chunk", chunk.content_id, chunk) for chunk in state["receipts"].values())
                    rows.extend(("limit_amendment", str(entry.sequence), entry) for entry in run.limit_amendments)
                    fields = state["checkpoint_fields"]
                    if fields is not None:
                        import json
                        rows.append(("checkpoint", "current",
                                     dict(sequence=state["checkpoint_sequence"], fields=list(fields))))
                        rows.extend(("checkpoint_field", name, json.loads(text)) for name, text in fields.items())
                    if run.result is not None:
                        rows.append(("result", "current", result_record(run.result, run)))
                    payloads = {}
                    for handle in run.data.artifacts:
                        manifest = handle.manifest
                        reference = state["artifact_payloads"][manifest.content_id]
                        rows.append(("publication", manifest.content_id, dict(manifest=manifest, payload=reference)))
                        payloads[reference.content_id] = (reference, memoryview(handle.array).cast("B"))
                    rows.extend(("payload", identity, reference) for identity, (reference, _) in payloads.items())
                    target.commit(tuple(rows), collected=tuple(chunk.content_id for chunk in state["chunks"].values()),
                                  payloads=tuple(payloads.values()))
                    # The journal's pages were allocated by this commit.
                    files.reserve(target.path.stat().st_size)
                finally:
                    target.close()
        except BaseException:
            shutil.rmtree(files.path)
            raise
        return files.path


def load(path, *, backend, method=None, progress=None):
    """Open the same durable execution, under its original exclusive lock.

    The Plan is restored through the Method's archive hook and must reproduce the
    saved Plan identity. The journal is then opened by ``Run._restore``, cache rows
    are restored as saved (a prepared handle's circuit is decoded at its first
    native use), published arrays are read from the journal at their first use,
    and a saved Result is attached to the journal's trace with the cap prefix
    and byte counter it captured. The Result keeps its own identity and
    observations, but events and submissions that the Run revised after the
    Result, for example through a later cancellation and its refresh, appear in
    that trace with their later status. Nothing is planned, lowered, acquired or
    analyzed. The folder is opened without a file-byte allowance, because every
    file in it was charged when it was written. After loading, each file write
    is charged to the Run's stored-data total (``_bind_capacity``).
    ``progress`` is the reopened Run's progress callback.
    """
    from nwqlib._choice_archive import load_plan
    from nwqlib._run_journal import RUN_FORMAT, unsupported_run_format
    from nwqlib._prepared_execution import Run
    from nwqlib.execution import ExecutionLimits

    files = ArchiveFiles(path, None)
    try:
        header = files.read_json("run.json")
    except FileNotFoundError as error:
        if files.file("run.sqlite").is_file():
            error.add_note(f"{files.path} holds a Run journal and no run.json. A Run writes run.json whole "
                           "before it commits its journal header, so the creation of this Run stopped before "
                           "any preparation or acquisition, and the folder holds no Run to load")
        raise
    if header.get("format") != RUN_FORMAT:
        raise unsupported_run_format(header.get("format"))
    name = header.get("selection")
    if type(name) is not str:
        raise ValueError("saved run manifest must name its selection file")
    selection = files.read_json(name)
    plan = load_plan(selection, files, method=method)
    run = None
    try:
        run = Run._restore(plan, backend=backend, directory=path, archive_files=files,
                           selection=dict(file=name, selection=selection),
                           saved_limits=ExecutionLimits.model_validate(header["limits"]), progress=progress)
        # Published numerical outputs have one durable payload in SQLite. The
        # Run's store registers each manifest and reads its payload on first
        # use (ArtifactStore.get), once per payload digest. An array read
        # before the Run closes stays available after it closes.
        results = list(run._state["journal"].rows("result"))
        if results:
            cls = plan.method.result_type
            saved = results[0][1]
            count = saved["limit_amendment_count"]
            result = cls.model_validate(saved["result"])
            data = run.data
            from dataclasses import replace
            from nwqlib.execution import ExecutionTrace
            fields = {name: getattr(data.trace, name) for name in ExecutionTrace.model_fields}
            history = run.limit_amendments[:count]
            fields.update(limit_amendments=history, data_bytes=saved["data_bytes"],
                limits=history[-1].new if history else run._state["initial_limits"])
            data = replace(data, trace=ExecutionTrace(**fields))
            result = result._attach(plan, data)
            run._state["result"] = result
        _bind_capacity(run, files)
        return run
    except BaseException:
        if run is not None:
            run.close()
        raise
