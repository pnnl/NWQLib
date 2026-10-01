"""Durable directories of the QHD augmented-Lagrangian layer and of box refinement.

``solve_augmented_lagrangian(..., directory=...)`` and ``refine_box(..., directory=...)`` create every inner
Run durably under one directory, through ``prepare(..., directory=...)``, and rewrite a small outer record
after each completed round or level. ``resume_augmented_lagrangian`` and ``resume_box_refinement`` continue
from it. The directory holds

    controller.json                  the outer record
    controller.json.lock             the directory's controller lock
    problem.pickle                   the live SymPy input, written once
    noise-model.json                 the noise model of a noisy Aer backend, written once, with its NPY arrays
    iterations/<k>/run/              the Run of round k of the augmented-Lagrangian layer
    iterations/<k>/levels/<z>/run/   the Run of level z of round k, with box refinement
    iterations/<k>/levels/<z>/tables.json  table-stage data of level z in augmented-Lagrangian round k
    levels/<z>/run/                  the Run of level z of a standalone refinement
    levels/<z>/tables.json                 table-stage data of standalone refinement level z

Each refinement level also keeps `tables.json` beside its `run/` directory, containing its table-evaluation
count, the refinement's constant C when it is rational and, for the search model, the unscaled support tables;
this applies both to `levels/<z>/` and to `iterations/<k>/levels/<z>/`.

The outer record keeps the run's arguments, which never change, and its committed state. The arguments
include the configuration of the one backend of every inner Run. For the augmented-Lagrangian layer they also include the
constraint preprocessing and the check counts of its setup, which resume reads instead of evaluating the
support tables and summand scans again (``constrained._setup``). The state is the completed rounds, the inner
representation that the round in progress selected (augmented Lagrangian only, committed before its first
inner Run so that resume reuses it), the completed levels of the refinement in progress together with a stop
that its last level already decided, and the final record once the run has terminated. A round or level enters it only after it has completed. Each
inner Run keeps its own lifecycle and journal (``nwqlib._prepared_execution.Run``), so the outer record names
the Runs by identity and repeats none of their counters or observations.

Commit rule. ``Directory.commit`` writes the new record to a temporary file in the directory and moves it
over the previous record with ``os.replace``, which replaces a file atomically within one filesystem, so an
interruption leaves either the previous record or the new one. Nothing is synchronized with ``fsync``, so the
rule covers an interrupted process (an exception, a keyboard interrupt or a killed process) and not a power
loss, the scope that the Run archives have too (docs/prepared_execution.md).

Backend. A backend configuration holds no noise model, and ``AerBackend.from_noise_model`` draws a new
binding identity on every call. In a new process ``load_run`` accepts ``AerBackend(noise_model_id=...)``
with the saved identity, and that backend is unbound, so no public backend object both matches the saved
Runs and carries their model. ``Directory.create`` therefore stores the configuration and writes a bound
model once, in the form in which ``Run.save`` stores it (``_run_archive._noise_model_data``), with its errors
and basis gates. Before any work, ``Directory.bind`` compares the backend given to resume with that
configuration and, for an unbound noisy backend, binds the saved model to its own copy for the Runs that
resume creates, as ``load_run`` binds a Run's saved model to the reopened Run's copy.

Stored data. The files of the outer directory are not charged against ``max_data_bytes``. That limit caps
what the inner Runs record (``nwqlib.execution.ExecutionLimits``), and each Run receives the remainder of the
whole run (``_outer.round_limits``). ``problem.pickle`` and ``noise-model.json`` hold the caller's inputs and
are written once. ``controller.json`` holds the arguments and the records of the completed rounds and levels,
which name their Runs and repeat none of their observations, so its size is that of those records, whose
number of JSON values grows linearly with the completed rounds and levels
(``constrained_records.AugmentedLagrangianRecord``, "Record size"). It is rewritten after each completed
round or level, so the bytes written over a run grow with the square of their number, while the directory
keeps one copy.
The per-level `tables.json` files contain generated table-stage data. Their total stored size is the sum of
their UTF-8 file lengths and grows with the retained levels and their table sizes. They are outer files and
are excluded from the inner Runs' `max_data_bytes`; their save and load memory is admitted against the level
Method's `max_bytes`. There is no cumulative outer-directory disk limit in these fields.
"""

from contextlib import closing, contextmanager
from dataclasses import dataclass
import json
from math import inf
import os
from pathlib import Path
import pickle

from ._outer import HeaderlessRun

RECORD = "controller.json"
PROBLEM = "problem.pickle"
NOISE = "noise-model.json"
# One format per layer. They change with the fields of the outer record, and a directory of another
# format is refused rather than converted, since this unreleased package keeps no development-schema
# compatibility (docs/FRAMEWORK.md, "Package Import Surface").
FORMATS = {"constrained": "qhd.constrained_run/4", "refinement": "qhd.refinement_run/5"}
RESUME = {"constrained": "resume_augmented_lagrangian", "refinement": "resume_box_refinement"}


@contextmanager
def _controller(path):
    """Hold the directory's exclusive controller lock while one call continues it.

    The lock is a nonblocking ``flock`` on ``controller.json.lock``, as a Run holds one on its journal
    (``_run_journal.LocalJournal``). A second process that continues the same directory at the same time
    fails here, before it reads the outer record, instead of rewriting that record after the first. The
    kernel releases the lock when the process ends, so a lock file left by an interruption blocks nothing.
    """
    import fcntl

    descriptor = os.open(path / (RECORD + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            error.add_note(f"{path} already has an active controller. Continue it from one process at a time.")
            raise
        yield
    finally:
        os.close(descriptor)


def require_new(path, kind):
    """Raise FileExistsError before any work when ``path`` exists, naming the function that continues it."""
    if Path(path).exists():
        raise FileExistsError(f"{path} exists. A durable run needs a new directory, and {RESUME[kind]} continues "
                              "an existing one")


def _dump(value):
    """JSON form of a record, a tuple of records, or a plain value."""
    if isinstance(value, tuple):
        return [_dump(item) for item in value]
    dump = getattr(value, "model_dump", None)
    return value if dump is None else dump(mode="json")


def run_backend(backend, execution):
    """Return the backend of every inner Run: ``backend``, or ``AerBackend()`` for quantum execution without one.

    ``Run`` selects local Aer when a quantum Run gets no backend
    (``nwqlib._prepared_execution.Run``), so the resolved backend has the
    configuration that each inner Run records in its journal header.
    """
    if backend is None and execution == "quantum":
        from nwqlib.backends.connection import AerBackend

        return AerBackend()
    return backend


def _noisy(backend):
    """Whether ``backend`` is an ``AerBackend`` that names a noise-model binding."""
    from nwqlib.backends.connection import AerBackend

    return isinstance(backend, AerBackend) and backend.noise_model_id is not None


class Directory:
    """One durable run directory with its fixed settings, the committed state it saved, and its Run folders.

    Attributes:
        path: The directory.
        kind: ``"constrained"`` or ``"refinement"``, which fixes the format.
        settings: The run's arguments as JSON, written once, with the configuration of the inner Runs'
            backend under ``backend``.
        saved: The committed state read by ``open``, as JSON with the keys ``iterations`` and
            ``representation`` (augmented Lagrangian only), ``levels``, ``blocked`` and ``record``. Empty
            for a new directory.
        backend: The backend of the inner Runs that this call creates or continues, set by ``create`` and
            by ``bind``.
        end_at_unfinishable: Whether a reopened Run that cannot finish without new work ends the run
            (``unrecoverable``). A resume call sets it from its argument, and it is False otherwise.
            This option handles an unfinishable continuation after a Run has reopened. A headerless
            folder fails during reopen and is not handled by ``end_at_unfinishable``. Reopen does not
            remove or recreate that folder automatically. Before a completed round or level exists,
            an inner failure still propagates.
    """

    def __init__(self, path, kind, settings, saved):
        self.path, self.kind, self.settings, self.saved = path, kind, settings, saved
        self._iterations = tuple(saved.get("iterations", ()))
        self._representation = saved.get("representation")
        self.backend = None
        self.end_at_unfinishable = False

    @classmethod
    @contextmanager
    def create(cls, path, kind, objects, settings, backend):
        """Create a new directory, write ``objects`` to ``problem.pickle`` and commit the empty state.

        ``objects`` are the live SymPy objects of the problem, which ``problem`` reads back. ``backend`` is
        the caller's backend. Its resolved form (``run_backend``, with ``settings["execution"]``) becomes
        ``Directory.backend`` and its configuration ``settings["backend"]``. The model bound to a noisy
        ``AerBackend`` is written to ``noise-model.json`` as ``_run_archive._noise_model_data`` forms it,
        its errors from ``NoiseModel.to_dict`` with NPY arrays and its basis gates, which select the Aer
        target that a circuit is lowered to. That is the form in which ``Run.save`` stores it and
        ``_run_archive._load_noise_model`` rebuilds it. An unbound noisy backend raises before the
        directory is created. The directory must not exist (``require_new``), as for
        ``prepare(..., directory=...)``, and missing parents are created. The controller lock is held until
        the context ends.
        """
        from nwqlib._choice_archive import ArchiveFiles
        from nwqlib._run_archive import _noise_model_data

        require_new(path, kind)
        backend = run_backend(backend, settings["execution"])
        noise = _noise_model_data(backend._bound_noise_model()) if _noisy(backend) else None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        with _controller(path):
            with (path / PROBLEM).open("xb") as stream:
                pickle.dump(objects, stream, protocol=5)
            if noise is not None:
                ArchiveFiles(path, None).write_numpy_json(NOISE, noise)
            configuration = None if backend is None else backend.model_dump(mode="json")
            directory = cls(path, kind, {**settings, "backend": configuration}, {})
            directory.backend = backend
            directory.commit()
            yield directory

    @classmethod
    @contextmanager
    def open(cls, path, kind):
        """Hold the controller lock of an existing directory and read its outer record.

        A directory of the other layer is refused with the name of its resume function, and a saved result
        archive with the name of its loader.
        """
        from nwqlib._choice_archive import ArchiveFiles

        path = Path(path)
        if not (path / RECORD).is_file():
            # A saved result archive of either layer is opened by its loader, never resumed.
            for name, loader in (("refinement.json", "load_box_refinement"),
                                 ("constrained.json", "load_augmented_lagrangian")):
                if (path / name).is_file():
                    raise FileNotFoundError(f"{path} is a saved result archive containing {name}, not a durable run "
                                            f"directory of {RESUME[kind]}. Open it with {loader}({str(path)!r})")
            raise FileNotFoundError(f"{path} holds no {RECORD}, so it is not a durable run directory of "
                                    f"{RESUME[kind]}. A directory whose first commit was interrupted holds no "
                                    "round yet and can be removed")
        with _controller(path):
            saved = ArchiveFiles(path, None).read_json(RECORD)
            if saved.get("format") != FORMATS[kind]:
                other = [name for name, fmt in FORMATS.items() if fmt == saved.get("format")]
                hint = f", continue it with {RESUME[other[0]]}" if other else ""
                raise ValueError(f"{path} holds format {saved.get('format')!r}, not {FORMATS[kind]!r}{hint}")
            yield cls(path, kind, saved["settings"], saved)

    def bind(self, backend):
        """Set and return ``Directory.backend`` for a resume call, comparing ``backend`` with the stored one first.

        ``backend`` is resolved as ``create`` resolved the original one (``run_backend``), so None stands for
        ``AerBackend()`` under quantum execution. Its configuration must equal ``settings["backend"]``, the
        configuration that every inner Run of the directory records, or ValueError names both before
        anything is reopened, planned or committed. A Run that resume creates on another backend would
        otherwise mix two populations in one run, which the outer record could not show. A noisy
        ``AerBackend`` that still holds its bound model, the original backend in the original process, is
        used as it is. For an unbound one, such as ``AerBackend(noise_model_id=...)`` with the original
        identity in a new process, the model saved by ``create`` is rebuilt with its errors and basis gates
        (``_run_archive._load_noise_model``) and bound to a copy of ``backend``, so the Runs that resume
        creates are lowered to the Aer target of the original Runs. A model that cannot be rebuilt raises
        here, before any work.
        """
        backend = run_backend(backend, self.settings["execution"])
        stored = self.settings["backend"]
        given = None if backend is None else backend.model_dump(mode="json")
        if given != stored:
            hint = ""
            if (stored or {}).get("kind") == "qiskit_aer" and stored.get("noise_model_id") is not None:
                hint = (f". A noisy Aer run continues with AerBackend(noise_model_id={stored['noise_model_id']!r}), "
                        "and resume binds the noise model saved in the directory, while a new from_noise_model "
                        "binding has another identity")
            raise ValueError(f"the inner Runs of {self.path} use the backend configuration {json.dumps(stored)}, and "
                             f"resume was given {json.dumps(given)}. Pass the backend of the original call{hint}")
        if _noisy(backend) and backend._noise_model is None:
            from nwqlib._choice_archive import ArchiveFiles
            from nwqlib._run_archive import _load_noise_model

            model = _load_noise_model(ArchiveFiles(self.path, None).read_numpy_json(NOISE))
            backend = backend.model_copy()
            backend._noise_model = model
        self.backend = backend
        return backend

    def problem(self):
        """Return the live SymPy objects of ``problem.pickle``, read by ``archive._SymbolicReader``.

        The reader resolves SymPy classes only, so no other Python global is loaded.
        """
        from .archive import _SymbolicReader

        with (self.path / PROBLEM).open("rb") as stream:
            return _SymbolicReader(stream).load()

    def commit(self, *, iterations=None, representation=None, levels=(), blocked=None, record=None):
        """Rewrite the outer record with the committed state, by the commit rule of the module docstring.

        ``iterations`` replaces the completed rounds (augmented Lagrangian only), which a commit without
        it keeps, and clears the representation of the round in progress, since the completed round
        holds its own. ``representation`` is the ``InnerRepresentation`` that the round in progress
        selected, committed before its first inner Run so that resume reuses it instead of selecting
        again, and a commit without it keeps the committed one. ``levels`` are the completed levels of
        the refinement in progress and ``blocked`` the ``(termination, failure)`` pair that its last
        level decided, or None. ``record`` is the final record once the run has terminated. It holds
        every round and level, so the other keys are then left empty.
        """
        if iterations is not None:
            self._iterations = tuple(iterations)
            self._representation = None
        if representation is not None:
            self._representation = _dump(representation)
        final = record is not None
        data = dict(format=FORMATS[self.kind], problem=PROBLEM, settings=self.settings)
        if self.kind == "constrained":
            data["iterations"] = [] if final else _dump(self._iterations)
            data["representation"] = None if final else self._representation
        data.update(levels=[] if final else _dump(tuple(levels)),
                    blocked=None if final or blocked is None else list(blocked),
                    record=_dump(record) if final else None)
        target = self.path / RECORD
        temporary = self.path / (RECORD + ".partial")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        os.replace(temporary, target)

    def commit_levels(self, levels, blocked):
        """Commit the completed levels of the refinement in progress, keeping the committed rounds (``Frontier``)."""
        self.commit(levels=levels, blocked=blocked)


@dataclass(frozen=True)
class Frontier:
    """The durable part of one refinement, namely its directory, its level folders and its committed levels.

    The name follows the execution frontier of docs/CODE_TOUR.md, the resumable state of a Run, and
    applies it to a refinement, whose resumable state is its completed levels and the stop that the last
    of them decided.

    ``refinement._refine`` runs level z in ``base/levels/<z>/run``, where ``base`` is the ``outer``
    directory's path for a standalone refinement and its ``iterations/<k>`` folder in round k of the
    augmented-Lagrangian layer. ``levels`` and ``results`` are the levels that a resumed refinement
    already completed and their QHD results, and ``blocked`` the stop that the last of them decided.
    After each completed level the refinement commits through ``outer.commit_levels``.
    """

    outer: "Directory"
    base: Path
    levels: tuple = ()
    results: tuple = ()
    blocked: tuple | None = None

    def folder(self, level):
        """The Run folder of level ``level``, counted from 1."""
        return self.base / "levels" / str(level) / "run"


def live_problem(stored, objects):
    """Return the live problem of a saved run from its stored record and the SymPy objects of its pickle.

    ``stored`` is the portable ``Optimization`` or ``ConstrainedOptimization`` of a saved record, and
    ``objects`` holds the objective, the variables and, for a constrained problem, the equalities and
    inequalities, in that order.
    """
    fields = stored.model_dump(mode="json", exclude_computed_fields=True)
    fields.update(zip(("objective", "variables", "equalities", "inequalities"), objects))
    return type(stored).model_validate(fields)


def header_committed(folder):
    """Whether the Run folder ``folder`` holds a committed journal header, or raise when its journal cannot be read.

    ``Run.__init__`` creates the folder, then its journal ``run.sqlite``, then the selected inputs with
    ``run.json`` (``_run_archive.initialize``), and commits the journal header last. Nothing is prepared or
    acquired before the header commit. A headerless folder can nevertheless contain stored input files
    and data bytes, which ``closed_trace`` accounts for separately. A process stopped after
    ``run.json`` and before the header leaves such a folder, and so does a Run whose creation raised there,
    for example when ``max_data_bytes`` cannot hold its header rows. ``load_run`` refuses it. The journal is
    opened as a Run opens it (``_run_journal.LocalJournal``), which takes its lock and lets SQLite roll back
    an interrupted transaction, and only the presence of a header row is read, so no row size cap applies.

    There are three outcomes. True: the header row is present. False: the folder holds no journal, or the
    journal was read and holds no header row, which includes a journal whose schema was inspected and has
    no tables yet, since ``LocalJournal`` creates the file before the first transaction creates its tables.
    Otherwise the journal could not be read, for example while another connection holds an exclusive SQLite
    lock on it, another controller holds the Run, or the disk reports an I/O error, and the ``sqlite3.Error``
    or ``OSError`` of that read propagates. A failed read is no evidence that the folder holds no work, so
    ``closed_trace`` then records unknown counts and ``reopen`` keeps the folder.
    """
    import sqlite3
    from nwqlib._run_journal import LocalJournal

    path = folder / "run.sqlite"
    if not path.is_file():
        return False
    try:
        journal = LocalJournal(path, inf, create=False)
    except sqlite3.Error:
        # LocalJournal reads the records table as it opens, so a journal without tables raises here. Inspect
        # the schema without waiting for a lock: an empty schema holds no header, and any other state, a
        # failed inspection included, leaves the original error as the outcome.
        try:
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0)) as connection:
                tables = connection.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
        except sqlite3.Error:
            tables = None
        if tables == 0:
            return False
        raise
    try:
        return journal.connection.execute("SELECT 1 FROM records WHERE kind='header'").fetchone() is not None
    finally:
        journal.close()


def reopen(folder, *, backend, progress):
    """Open the Run of an unfinished round or level with ``load_run``, or return None when it has none yet.

    A round or level whose Run exists continues that Run and never gets a new one. The layer created the
    folder for this round or level, so its Run solves the round's or level's problem from its random
    stream, and resume takes the Plan from the Run without planning again. The reopened Run reports to
    ``progress``, the callback of the resume call. When ``load_run`` raises, a note says whether the folder
    holds work, decided by its journal header (``header_committed``). Only a journal that was read and holds
    no header lets the note suggest removing the folder. When the journal cannot be read, the note says so
    and that the folder may hold committed work.
    """
    import sqlite3

    if folder is None or not folder.exists():
        return None
    from nwqlib.scientist import load_run

    try:
        return load_run(folder, backend=backend, progress=progress)
    except Exception as error:
        try:
            committed = header_committed(folder)
        except (sqlite3.Error, OSError) as unread:
            error.add_note(f"The journal of {folder} could not be read ({type(unread).__name__}: {unread}), so the "
                           "folder may hold committed work of an unfinished round or level. Keep it, and resume "
                           "again once its journal can be read")
        else:
            if committed:
                error.add_note(f"{folder} holds the Run of an unfinished round or level, which resume continues "
                               "and never replaces. Keep the folder; its recorded work stays charged")
            else:
                error.add_note(f"{folder} holds no committed journal header, so this "
                               "Run charged no preparation or acquisition. Removing only this inner Run folder "
                               "lets resume create the Run on retry. end_at_unfinishable=True does not handle "
                               "this reopening error")
        raise


def closed_trace(folder, backend):
    """Return ``(trace, reason)`` for the Run of a round or level whose preparation raised.

    ``prepare`` closes its Run when the preparation raises, so the layer never holds that Run object
    (``_outer.run_counts``). ``folder`` is the Run folder of a durable round or level, or None without a
    durable directory, when the layer cannot read the counters and the reason says so. A durable Run keeps
    its counters in its journal, and ``load_run`` reopens the closed folder without planning, preparing or
    acquiring, so its trace gives the work that the failed round or level recorded, attempts of every
    status included. The reason then says that the raised preparation does not establish a built circuit,
    which ``_outer.law_count`` needs. A folder that does not exist means that ``prepare`` raised before
    creating the Run, which spent nothing, so both are None and the counts are zero. A folder whose journal
    was read and holds no committed header (``header_committed``) has no recorded counters, but nothing was
    prepared or acquired before the header, so the counts are known zeros apart from the stored data. The
    result is then an ``_outer.HeaderlessRun`` with the stored bytes of the folder's files apart from the
    journal, counted as a reopened Run counts them (``_run_archive.stored_file_bytes``), and the reason is
    None. When the folder's files cannot be read those bytes are unknown, with the error as their reason.
    When the journal itself cannot be read, the folder may hold a committed header and charged work, so the
    trace is None and the reason names the error, which makes every count unknown (``_outer.run_counts``).
    """
    import sqlite3

    if folder is None:
        return None, "preparation raised and closed its Run before its counters could be read"
    if not folder.exists():
        return None, None
    try:
        committed = header_committed(folder)
    except (sqlite3.Error, OSError) as error:
        return None, (f"preparation raised, and the journal of its Run folder could not be read "
                      f"({type(error).__name__}: {error})")
    if not committed:
        from nwqlib._run_archive import stored_file_bytes

        try:
            return HeaderlessRun(stored_file_bytes(folder)), None
        except OSError as error:
            return HeaderlessRun(None, f"the files of the Run folder, whose creation raised before its journal "
                                       f"header, could not be read: {error}"), None
    from nwqlib.scientist import load_run

    with load_run(folder, backend=backend, progress=False) as run:
        return run.trace, "the preparation raised, and its record does not establish that its circuit was built"


def unrecoverable(error, directory, *, end=False):
    """Whether ``error`` of a reopened Run's continuation must propagate, with a note if so.

    Two outcomes that an interruption leaves in a Run cannot finish without new charged work.
    ``Run.resume`` raises ``RunFailed`` with status ``uncertain`` for an attempt whose outcome no route can
    bring back, such as an interrupted local acquisition or classical evolution, or a lost acknowledgement
    that the backend cannot reconcile (``Run._raise_unrecoverable_outcome``).
    ``_prepared_execution.PreparationNotRebuilt`` reports a local preparation that has a charge but no
    receipt, because it failed or was interrupted (``_prepared_execution._unbuilt``, raised by
    ``execute_static`` before any new submission and by ``_prepare_static_item`` for its own item). Their
    charges stay spent, the Run repeats neither, and resume never gives a round or level a second Run.
    The outer run therefore cannot pass this round or level. By default the error propagates instead of
    ending the run as ``inner_failed``. The outer record keeps its committed rounds and levels unchanged,
    while the Run keeps the step charged and records an interrupted attempt as uncertain, so a later
    resume can still use an original outcome that becomes retrievable, for example a job that the backend
    can reconcile later.

    With ``end``, the ``end_at_unfinishable`` argument of the resume functions, the function returns
    False, and the caller records the error as it records any inner failure (``_outer.inner_failure``).
    After a completed round or level the run then ends with ``inner_failed``, keeps what has completed
    and records the error's type and text as the failure. Before anything has completed the error still
    propagates, as any first failure does. A local process that stopped during an evolution never
    delivers a late outcome, so without this choice the completed rounds could not be returned as a
    result. Ending the run neither repeats nor replaces the step, whose charge stays spent.

    Every other error of a continued Run is one that an uninterrupted run can meet at the same point,
    and the layers record it as they do there. The exception is an interruption between an inner
    failure and the next commit of the outer record, after which the reopened Run can report that
    failure in another form, for example a preparation that raised as ``PreparationNotRebuilt``.

    ``directory`` is the durable directory, which the note names.
    """
    from nwqlib._prepared_execution import PreparationNotRebuilt
    from nwqlib.execution import RunFailed

    if end or not (isinstance(error, RunFailed) and error.status == "uncertain"
                   or isinstance(error, PreparationNotRebuilt)):
        return False
    error.add_note(f"The durable run in {directory} keeps its committed rounds and levels. This Run cannot finish "
                   "without new work, which resume never makes, so the run cannot continue past it. Its work "
                   "stays charged. After a completed round or level, resuming with end_at_unfinishable=True "
                   "ends the run here with inner_failed and keeps what has completed. Before any completed "
                   "outer work the error still propagates")
    return True


def saved_result(folder, *, backend):
    """Return the Result of a committed round or level from its Run folder.

    ``load_run`` opens the Run without planning or acquiring, and it is closed again at once.
    """
    from nwqlib.scientist import load_run

    with load_run(folder, backend=backend) as run:
        return run.result
