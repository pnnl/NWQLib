"""File readers and writers.

A quantum SDK is imported only by the functions that take or return its objects.
"""

from nwqlib.io.materialization import QasmMaterialization, QasmMaterializationBudget, materialize_qasm3_file
from nwqlib.io.qasm import export_qasm
from nwqlib.io.streaming import QasmPrefix, QasmWriteBudget, QasmWriteError, QasmWriteReceipt, write_qasm3, write_qasm3_file

__all__ = [
    "export_qasm", "QasmWriteBudget", "QasmWriteReceipt", "QasmPrefix", "QasmWriteError",
    "write_qasm3", "write_qasm3_file", "QasmMaterialization", "QasmMaterializationBudget", "materialize_qasm3_file",
]
