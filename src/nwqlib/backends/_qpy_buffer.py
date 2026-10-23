"""Byte-limited in-memory output for prepared backend QPY payloads."""

from io import BytesIO


class QpyBuffer(BytesIO):
    """Check each write's final offset against the limit before BytesIO allocates or copies it."""

    def __init__(self, limit, *, message):
        super().__init__()
        self.limit = limit
        self.message = message

    def write(self, data):
        if self.tell() + memoryview(data).nbytes > self.limit:
            raise ValueError(self.message)
        return super().write(data)
