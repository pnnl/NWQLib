"""Chunked binary-payload writes inside a run journal transaction."""


def write_bytes(connection, key, buffer, chunk_bytes):
    """Copy an already admitted buffer into the caller's current transaction."""
    view = memoryview(buffer).cast("B")
    for ordinal, start in enumerate(range(0, len(view), chunk_bytes)):
        connection.execute("INSERT INTO binary VALUES (?,?,?)", (key, ordinal, view[start:start + chunk_bytes]))
