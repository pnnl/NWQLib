"""The run log of a saved Run, written in atomic SQLite transactions of bounded size under an exclusive lock.

The run log (``run.sqlite``) is one SQLite file with two tables. ``records(kind,
key, size, collected, payload)`` holds one JSON row per saved fact, keyed by its
kind (header, limits, rng, prepared, preparation, submission, event, chunk,
checkpoint, cache and others) and its identifier, with the UTF-8 byte size of its
JSON text in ``size``. ``collected`` is zero for everything except a chunk
that the Method has collected, which gets the next positive ordinal.
``binary(identity, ordinal, payload)`` holds the payloads of backend circuits and
the saved observation arrays as ordered blocks of at most 1 MiB.

Every change of a Run's state is one ``commit`` call, and SQLite makes it atomic.
Each row is encoded once and its UTF-8 size is checked against the byte limit
before the row is written.
Only cache rows and fields that an adaptive Method removed from its saved
iteration state (the state its iteration code needs to choose the next
experiment) can be deleted. Every other row records work that happened and is
only ever inserted or revised.

One process at a time holds a Run open. ``LocalJournal`` takes a nonblocking
exclusive ``flock`` on the adjacent ``.lock`` file and holds it until ``close``.
The operating system releases that lock when the process exits, so a lock file
left by a crash does not block a later process.
"""

from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Literal
import json
import os
from pathlib import Path

from nwqlib._limits import exceeds
from nwqlib.core.records import ContentID, FrozenArray, Record, Text
from nwqlib.core.planning import Realization, RuntimeOptions
from nwqlib.operators.access import Count


# Format label of a Run folder, written to run.json and the journal header.
# Raise it for changes to the envelope or shared stored records.
# A Method-owned Result field change may instead bump that Method's archive
# format when load_run checks it before restoring the journal and validating
# the concrete Result.
RUN_FORMAT = "nwqlib.run/24"
# Size of the ordered blocks of a saved payload. 1 MiB keeps each value far
# below SQLite's per-value length limit and bounds the buffers of later chunked
# reads, whatever max_data_bytes is. LocalJournal.commit writes these blocks and
# Run._saved_payload passes the size to ArtifactStore._hydrate, which checks it
# against every block.
PAYLOAD_BLOCK_BYTES = 1024**2


def unsupported_run_format(found, where="saved run"):
    """Return the refusal for a Run folder written in another format.

    No compatibility path converts an older development format, so the
    message names the folder's format and the one this source reads, and
    says that the Plan must be run again.
    """
    return ValueError(f"{where} format {found!r} is not {RUN_FORMAT!r}, the only Run format this "
                      "NWQLib reads; older formats are not converted, so run the Plan again in a "
                      "new Run folder (or open this folder with the NWQLib source that wrote it)")


class WorkflowItem(Record):
    """One point of a static Plan, with its random draw and the identifiers of its preparation and acquisition.

    One row per point of the Plan (`StaticPoint`), keyed by the point's key:
    the experiment name, followed by the axis values for an experiment with
    range axes. The runtime seed is
    drawn once when the row is created. The three identifiers are filled in as
    the preparation charge, the preparation record and the attempt commit,
    which lets a restart find the exact work of each experiment instead of
    starting it again.
    """

    realization: Realization
    runtime: RuntimeOptions
    preparation_id: Text | None = None
    prepared_id: ContentID | None = None
    attempt: Text | None = None


class PreparationCharge(Record):
    """One preparation attempt, including a failed one without a preparation record.

    The charge commits before the backend work. ``prepared_id`` stays None for a
    preparation that failed or was interrupted. Even then, a circuit
    preparation counts against ``max_total_circuits``.

    Attributes:
        execution: ``quantum_circuit`` or ``host_kernel``.
        preparation_id: Identifier of this attempt.
        prepared_id: Content hash of the preparation record once the preparation completed, else None.
        checkpoint_sequence: Checkpoint of the iteration state that was current when the charge was made, or None.
        synthesis_work: Exact dense synthesis work that this preparation made, saved in the transaction that first saved its lowering result.
    """

    schema_version: Literal[3] = 3
    execution: Text
    preparation_id: Text
    prepared_id: ContentID | None = None
    checkpoint_sequence: Count | None = None
    synthesis_work: Count = 0


def _portable(value):
    """JSON form of a value the standard encoder cannot write: a Record, datetime or Enum.

    Computed fields such as ``content_id`` are excluded, because they are derived
    from the saved fields and are recomputed on load.
    """
    if isinstance(value, Record):
        return value.model_dump(mode="json", exclude_computed_fields=True)
    if isinstance(value, (datetime, Enum)):
        return value.isoformat() if isinstance(value, datetime) else value.value
    raise TypeError("run data must contain portable records or JSON scalars")


class JSONBytesExceeded(ValueError):
    """The encoded JSON of a value would exceed the byte limit it is checked against.

    ``Run._write`` checks new rows against the space that ``max_data_bytes``
    leaves, and it catches this error to name that limit in its own message.
    """


def _json_bound(value, max_bytes):
    """Bound JSON bytes before encoding, and reject unsupported or deep scalar trees.

    Strings use the encoder's escaped UTF-8 size and integers their decimal
    size. Finite binary64 values have an upper bound of 32 bytes. Reading
    Record fields avoids allocating its portable tree merely to count it. This
    bounds encoded bytes, not Python object storage, CPU time or process memory.
    ``max_bytes=None`` sets no limit and returns the bound.
    """
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("JSON byte limit must be None or a nonnegative integer")
    pending, size = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        # 128 is an untuned recursion boundary, registered in
        # docs/ENGINEERING_CONSTANTS.md ("Local journal scalar admission").
        if depth > 128:
            raise ValueError("run JSON nesting exceeds 128 levels")
        # Container terms: 2 bytes for the brackets or braces and n-1 commas
        # between n members. Each object key adds its escaped text and a colon.
        if isinstance(item, Record):
            # Admit immutable field metadata before model_dump allocates the
            # portable tree. Private native/cache references are never fields.
            fields = type(item).model_fields
            size += 2 + max(0, len(fields)-1)
            for key in fields:
                size += len(json.encoder.encode_basestring(key).encode("utf-8")) + 1
                pending.append((getattr(item, key), depth + 1))
        elif isinstance(item, FrozenArray):
            # FrozenArray fields are counted from their dtype, shape and byte
            # count, including exact padded-base64 length and JSON framing,
            # before any array bytes or base64 text are allocated. For shape
            # (a_1, ..., a_r) with n entries and b = 8 n bytes, the compact
            # {"dtype":"<f8","shape":[...],"data":"..."} has 36 skeleton
            # characters, the dimension digits, r - 1 separators and
            # q = 4 ((b + 2) // 3) base64 characters (RFC 4648, Sec. 4), none
            # of which needs JSON escaping. The object has scalar children at
            # depth + 1 and, for nonempty shape, dimension integers at depth + 2.
            shape = item.array.shape
            if depth + 1 + bool(shape) > 128:
                raise ValueError("run JSON nesting exceeds 128 levels")
            size += (
                36
                + sum(len(str(axis)) for axis in shape)
                + max(0, len(shape) - 1)
                + 4 * ((int(item.array.nbytes) + 2) // 3)
            )
        elif isinstance(item, dict):
            size += 2 + max(0, len(item)-1)
            if any(type(key) is not str for key in item):
                raise TypeError("run JSON dictionaries require string keys")
            for key, child in item.items():
                size += len(json.encoder.encode_basestring(key).encode("utf-8")) + 1
                pending.append((child, depth + 1))
        elif isinstance(item, (list, tuple)):
            size += 2 + max(0, len(item)-1)
            if exceeds(len(item), max_bytes):
                raise JSONBytesExceeded("run data exceeds the byte limit")
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            if exceeds(len(item), max_bytes):
                raise JSONBytesExceeded("run data exceeds the byte limit")
            size += len(json.encoder.encode_basestring(item).encode("utf-8"))
        elif item is None:
            size += 4
        elif type(item) is bool:
            size += 4 if item else 5
        elif type(item) is int:
            # log10(2) > 1/4, so this cheap lower bound rejects an enormous
            # integer before creating its decimal representation.
            if exceeds(item.bit_length()//4, max_bytes):
                raise JSONBytesExceeded("run data exceeds the byte limit")
            size += len(str(item))
        elif type(item) is float:
            # The shortest round-trip repr of a finite binary64 value has at most
            # 17 significant digits, so with sign, point and a three-digit
            # exponent it is at most 24 characters (for example
            # -2.2250738585072014e-308). 32 bytes cover every finite float.
            size += 32
        elif isinstance(item, (datetime, Enum)):
            pending.append((_portable(item), depth + 1))
        else:
            raise TypeError("run data must contain portable records or JSON scalars")
        if exceeds(size, max_bytes):
            raise JSONBytesExceeded("run data exceeds the byte limit")
    return size


def _check_portable(value):
    """Reject a container tree that the run log would not store faithfully, without measuring it.

    Dictionaries need string keys, nesting is limited to 128 levels as in
    ``_json_bound``, and every other item must be a Record, a FrozenArray, a
    string, an exact int, float or bool, None, a datetime or an Enum. The walk
    stops at a Record, whose typed fields its own validation admitted and
    whose JSON form ``_portable`` produces.
    """
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 128:
            raise ValueError("run JSON nesting exceeds 128 levels")
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise TypeError("run JSON dictionaries require string keys")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend((child, depth + 1) for child in item)
        elif not (item is None or type(item) in (bool, int, float)
                  or isinstance(item, (Record, FrozenArray, str, datetime, Enum))):
            raise TypeError("run data must contain portable records or JSON scalars")


def _encode_admitted(value, max_bytes):
    """Encode a value and return its text and UTF-8 byte length, refusing it when that length exceeds ``max_bytes``.

    The compact separators and ``allow_nan=False`` give the spelling that
    ``_json_bound`` counts. Nonfinite floats raise instead of producing invalid
    JSON. The text is UTF-8-encoded once here, and its byte length travels
    with it to the size checks of ``Run._write`` and ``LocalJournal.commit``,
    so no later step encodes it again to measure it.
    """
    text = json.dumps(value, default=_portable, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    size = len(text.encode("utf-8"))
    if max_bytes is not None and size > max_bytes:
        raise JSONBytesExceeded("run data exceeds the byte limit")
    return text, size


def encode(value, max_bytes):
    """Check the size bound of ``value``, then encode it as compact JSON for the run log.

    ``max_bytes=None`` checks only that the run log can store the tree
    (``_check_portable``), as ``encode_records`` does for admitted work.
    """
    if max_bytes is None:
        _check_portable(value)
    else:
        _json_bound(value, max_bytes)
    return _encode_admitted(value, max_bytes)[0]


def _row_value(kind, value):
    """The value a run-log row stores for ``value``.

    A chunk row omits the chunk's ``observation``. The chunk's preparation
    record, the ``prepared`` row named by its ``prepared_id``, stores the same
    readout declaration, and ``Run._write`` checks that they are equal before
    a chunk row is written. Reopening a Run puts the record's observation back
    (``Run._restore_observations``), so the reopened chunk equals the
    original, content hash included. A trajectory point chunk holds only its
    point's one-point readout, which reopening rebuilds from the trajectory
    declaration of the preparation record. A chunk row therefore grows with its JSON
    statistics but not again with the readout's labels or points. A
    probability chunk row holds its array manifests and scalar summaries,
    whose size does not grow with the number of outcomes. The values are
    binary payload blocks. Other rows store their value unchanged.
    """
    from nwqlib.execution import ObservationChunk
    if kind == "chunk" and isinstance(value, ObservationChunk):
        return {name: getattr(value, name) for name in type(value).model_fields if name != "observation"}
    return value


@dataclass(frozen=True)
class _EncodedRecords:
    """The checked JSON rows and their encoded sizes.

    ``sizes`` holds the UTF-8 byte length of each row's text, in row order.
    """

    rows: tuple
    sizes: tuple


def encode_records(records, max_bytes):
    """Encode a complete, finite set of row changes and check their total size before any row is written.

    Each row, in the saved form of ``_row_value``, passes ``_check_portable``
    and is encoded once (``_encode_admitted``). Its measured UTF-8 length is
    checked against the part of ``max_bytes`` that the earlier rows leave, so
    a set whose encoded rows exceed ``max_bytes`` raises JSONBytesExceeded.
    ``max_bytes=None`` sets no bound, for rows of admitted work, whose
    decoding checks bound them, and for the run log's own writes. The text
    of a row exists before its size is checked.
    """
    if not isinstance(records, (tuple, list)):
        raise TypeError("run row transitions require a finite tuple or list")
    records = tuple(records)
    if records and max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("JSON byte limit must be a nonnegative integer or None")
    remaining = max_bytes
    keys = set()
    encoded = []
    for kind, key, value in records:
        if (kind, key) in keys:
            raise ValueError("duplicate row in an atomic run transition")
        keys.add((kind, key))
        value = _row_value(kind, value)
        _check_portable(value)
        text, size = _encode_admitted(value, remaining)
        if remaining is not None:
            remaining -= size
        encoded.append((text, size))
    return _EncodedRecords(tuple((kind, key, text) for (kind, key, _), (text, _) in zip(records, encoded)),
                           tuple(size for _, size in encoded))


def _write_bytes(connection, key, buffer, chunk_bytes):
    """Copy an already checked buffer into the caller's current transaction as ordered blocks."""
    view = memoryview(buffer).cast("B")
    for ordinal, start in enumerate(range(0, len(view), chunk_bytes)):
        connection.execute("INSERT INTO binary VALUES (?,?,?)", (key, ordinal, view[start:start + chunk_bytes]))


class LocalJournal:
    """The run log of one Run. SQLite makes each commit atomic and durable, and a POSIX file lock makes access exclusive.

    The connection uses ``journal_mode=DELETE``, where deleting the rollback
    journal is the commit, and ``synchronous=EXTRA``, which also syncs the
    directory after that deletion. SQLite documents this as added durability when
    a power loss closely follows a commit. Actual durability still depends on the
    filesystem and drive completing their syncs. ``trusted_schema=OFF`` stops
    unaudited SQL functions named in the views, triggers or schema expressions of
    a copied or edited database from running. The folder must stay local and
    writable, because every commit creates and deletes the rollback journal next
    to the database.

    A single JSON row is also bounded by SQLite's maximum string length
    (1,000,000,000 bytes by default), independently of ``max_data_bytes``. Binary
    payloads avoid that per-value limit by being stored in blocks.

    Each ``records`` row stores the UTF-8 byte size of its JSON text in its
    ``size`` column, written with the text in the same transaction. The
    column precedes the text in the row, so a reopened Run restores its
    saved sizes and total from it without reading the text
    (``Run._restore_sizes``). Loading does not check the column against the
    text of the Run's own run log. ``Run._write`` keeps the only byte
    count and is the only admission check of ``max_data_bytes``, so the run
    log neither counts, refuses a commit nor refuses to open because of its
    saved total, which lifecycle rows of admitted attempts can take beyond
    that limit. ``max_bytes`` still bounds each array and binary payload
    read back and sets the block size of new payloads.
    """

    def __init__(self, path, max_bytes, *, create, restore_limits=False):
        """Take the exclusive POSIX lock and open the run log with its byte limit.

        The lock is taken before the database is opened, so a second process
        fails before reading any row. ``create=True`` requires a new file
        (``O_EXCL``), so an existing Run is never initialized twice.
        ``restore_limits`` reads the current limits, whose ``max_data_bytes``
        bounds the arrays and payloads read back. Collection ordinals record
        the order in which the Method collected observations.
        """
        if os.name != "posix":
            raise ValueError("durable run journals require POSIX advisory locking")
        import fcntl
        import sqlite3
        self.path, self.max_bytes = Path(path).resolve(), max_bytes
        self.connection, self.fd = None, None
        self.fd = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                error.add_note(f"Run journal {self.path} already has an active controller; close its open Run handle, such as prepared.run, before reopening.")
                raise
            flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
            database_fd = os.open(self.path, flags, 0o600)
            os.close(database_fd)
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True,
                isolation_level=None, check_same_thread=False)
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=EXTRA")
            self.connection.execute("PRAGMA trusted_schema=OFF")
            if create:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute("CREATE TABLE records (kind TEXT, key TEXT, size INTEGER NOT NULL, "
                    "collected INTEGER NOT NULL DEFAULT 0, payload TEXT NOT NULL, PRIMARY KEY(kind,key))")
                self.connection.execute("CREATE TABLE binary (identity TEXT, ordinal INTEGER, payload BLOB NOT NULL, "
                    "PRIMARY KEY(identity,ordinal))")
                self.connection.execute("COMMIT")
            elif restore_limits:
                row = self.connection.execute(
                    "SELECT payload FROM records WHERE kind='limits' AND key='current'").fetchone()
                if row is not None:
                    from nwqlib.execution import ExecutionLimits
                    self.max_bytes = ExecutionLimits.model_validate_json(row[0]).max_data_bytes
            self._collected_count = self.connection.execute(
                "SELECT count(*) FROM records WHERE kind='chunk' AND collected!=0").fetchone()[0]
        except BaseException:
            self.close()
            raise

    def commit(self, records=(), *, collected=(), payloads=(), deleted=(), encoded=None):
        """Commit all rows of one change of state, or none of them.

        ``BEGIN IMMEDIATE`` takes the write lock at the start, so the payload
        and ordinal reads and the writes see one consistent database.
        ``encoded`` holds the rows already encoded by ``Run._write``, and
        without it the rows are encoded here. Each row's size is written with
        its text. A binary payload whose content hash is already saved is not
        written again, which lets two acquisitions with identical array bytes
        share one payload. Chunks named in ``collected`` get the next
        ordinals, in the given order, only if they have none yet. Any error
        rolls the whole change back, and the in-memory ordinal counter changes
        only after ``COMMIT``. A change with nothing to write opens no
        transaction.
        """
        conn = self.connection
        if conn is None:
            raise ValueError("journal is closed")
        if not records and not collected and not payloads and not deleted:
            return
        if not isinstance(records, (tuple, list)) or not isinstance(deleted, (tuple, list)):
            raise TypeError("run transitions require finite row tuples")
        keys = {(kind, key) for kind, key, _ in records}
        removed = set(deleted)
        if len(keys) != len(records) or len(removed) != len(deleted) or keys & removed:
            raise ValueError("duplicate row or upsert/delete conflict in an atomic transition")
        # Only replaceable cache metadata and fields that a controller removed
        # from its checkpoint can be pruned. Published scientific observations,
        # receipts and payload associations are never deletable.
        if any(kind not in {"cache", "checkpoint_field"} for kind, _ in removed):
            raise ValueError("only cache rows and checkpoint fields can be deleted from a run journal")
        if len({reference.content_id for reference, _ in payloads}) != len(payloads):
            raise ValueError("duplicate binary reference in an atomic transition")
        import sqlite3
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as error:
            error.add_note(f"Cannot begin a transition in Run journal {self.path}; its folder must remain writable.")
            raise
        try:
            new_payloads = []
            for reference, data in payloads:
                if not isinstance(data, (bytes, memoryview)) or len(data) != reference.bytes:
                    raise ValueError("saved binary payload differs from its byte declaration")
                if isinstance(data, memoryview) and (not data.readonly or not data.c_contiguous):
                    raise ValueError("saved binary payload requires read-only contiguous bytes")
                if not conn.execute("SELECT 1 FROM binary WHERE identity=? LIMIT 1", (reference.content_id,)).fetchone():
                    new_payloads.append((reference, data))
            # The same transaction publishes both data and completed observations.
            if encoded is None:
                encoded = encode_records(records, None)
            for reference, data in new_payloads:
                _write_bytes(conn, reference.content_id, data, PAYLOAD_BLOCK_BYTES)
            for kind, key in removed:
                conn.execute("DELETE FROM records WHERE kind=? AND key=?", (kind, key))
            for (kind, key, text), size in zip(encoded.rows, encoded.sizes, strict=True):
                conn.execute("INSERT INTO records(kind,key,size,payload) VALUES(?,?,?,?) ON CONFLICT(kind,key) "
                    "DO UPDATE SET size=excluded.size, payload=excluded.payload", (kind, key, size, text))
            collected_count = self._collected_count
            for key in collected:
                current = conn.execute("SELECT collected FROM records WHERE kind='chunk' AND key=?", (key,)).fetchone()
                if current is None:
                    raise ValueError("collected observation has no durable completed receipt")
                if current[0] == 0:
                    collected_count += 1
                    conn.execute("UPDATE records SET collected=? WHERE kind='chunk' AND key=?", (collected_count, key))
            conn.execute("COMMIT")
            self._collected_count = collected_count
        except BaseException as error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if isinstance(error, sqlite3.Error):
                error.add_note(f"Cannot commit a transition in Run journal {self.path}; its folder must remain writable.")
            raise

    def rows(self, kind):
        """Yield ``(key, parsed JSON, collected)`` for each row of ``kind`` in insertion order.

        ``collected`` is True for a chunk the Method has collected.
        """
        for key, text, collected in self.connection.execute(
                "SELECT key,payload,collected FROM records WHERE kind=? ORDER BY rowid", (kind,)):
            yield key, json.loads(text), bool(collected)

    def payload_chunks(self, reference):
        """Yield the binary blocks of one array payload to its loader, one block alive at a time.

        The list of blocks is streamed, so the iterator's own metadata is
        O(1) instead of O(ceil(B/C)) for B payload bytes in C-byte blocks.
        The loader (``ArtifactStore._hydrate``) checks the block order, each
        block size and the complete total before it exposes any array. A
        yielded block is deleted when the calling code resumes the iterator,
        before the next block is read, so with calling code that also releases
        its block the explicit block memory is one block. SQLite's own caches
        are outside this count. After the run log is closed, the blocks are
        read through a read-only connection held for this iteration. Committed
        payload blocks never change, so an array first read after the Run
        closed is the one it saved.
        """
        import sqlite3
        if exceeds(reference.bytes, self.max_bytes):
            raise ValueError("saved array exceeds max_data_bytes")
        connection = self.connection
        if connection is None:
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
            connection.execute("PRAGMA trusted_schema=OFF")
        try:
            inventory = connection.execute("SELECT ordinal FROM binary WHERE identity=? ORDER BY ordinal",
                                           (reference.content_id,))
            for (ordinal,) in inventory:
                data = connection.execute("SELECT payload FROM binary WHERE identity=? AND ordinal=?",
                                          (reference.content_id, ordinal)).fetchone()[0]
                yield ordinal, data
                del data
        finally:
            if connection is not self.connection:
                connection.close()

    def read_payload(self, reference):
        """Return one saved backend payload as bytes after checking its list of blocks.

        A gap in the block ordinals or a total size different from the reference
        is rejected before the payload is used for a new submission or a remote
        preparation. Block contents are not authenticated.
        """
        if exceeds(reference.bytes, self.max_bytes):
            raise ValueError("saved binary data exceeds max_data_bytes")
        sizes = self.connection.execute("SELECT ordinal,length(payload) FROM binary WHERE identity=? ORDER BY ordinal",
                                        (reference.content_id,)).fetchall()
        if sum(size for _, size in sizes) != reference.bytes or any(i != ordinal for i, (ordinal, _) in enumerate(sizes)):
            raise ValueError("saved binary payload has missing or reordered chunks")
        output = bytearray(reference.bytes)
        offset = 0
        for ordinal, size in sizes:
            data = self.connection.execute("SELECT payload FROM binary WHERE identity=? AND ordinal=?",
                                           (reference.content_id, ordinal)).fetchone()[0]
            output[offset:offset+size] = data
            offset += size
        return bytes(output)

    def close(self):
        """Close the database, then release the Run's exclusive lock by closing its descriptor. Idempotent."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
