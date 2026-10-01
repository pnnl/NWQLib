"""Shared default allowances for explicitly requested local work and stored data."""

# Decimal GB: permit ordinary research data without a small demonstration cap.
# This bounds known bytes, not process RSS, CPU work or native SDK allocations.
DEFAULT_MAX_BYTES = 10_000_000_000

# 2**16 amplitudes, a 16-qubit direct magnitude/phase state preparation, whose
# synthesis grows as q*2**q. prepare_qiskit, standalone lowering and
# ExecutionLimits share this default. ENGINEERING_CONSTANTS.md, "Explicit input
# operation defaults", gives its reason and revisit condition.
DEFAULT_MAX_DIRECT_AMPLITUDES = 65_536
