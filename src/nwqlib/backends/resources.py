"""Standard gate classes, representative sampled blocks and dense/block-encoding CX laws."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

from nwqlib._validation import integer


@cache
def _standard_gate_classes():
    """Classes of Qiskit's standard gates, built once per process.

    Native circuit intake (``blocks._qiskit_intake``) starts its set of
    library gate classes from this one.
    """
    from qiskit.circuit.library import get_standard_gate_name_mapping
    return frozenset(gate.base_class for gate in get_standard_gate_name_mapping().values())


def dense_unitary_cx_qsd_upper_bound(total_qubits: int) -> int:
    """Return the CX count of the optimized quantum Shannon decomposition of an ``n``-qubit unitary.

    Per Shende, Bullock, and Markov, Synthesis of quantum-logic circuits,
    IEEE TCAD 25, 1000 (2006), doi:10.1109/TCAD.2005.855930, their
    decomposition of an ``n``-qubit unitary takes at most
    ``(23 * 4**n - 72 * 2**n + 64) / 48`` CX gates for ``n >= 2``. This is
    ``(23/48) 4**n - (3/2) 2**n + 4/3`` of Eq. (19) in Appendix A of
    arXiv:quant-ph/0406176v5 (Table 1, row "QSD (l = 2, optimized)"), and the
    numerator is divisible by 48 for every ``n >= 2``, so integer division is
    exact. The explicit base cases exist because block encoding accepts 1x1
    matrices, making routing width ``n = 1`` reachable even though the cited
    recurrence starts at two qubits.

    The count bounds that paper's circuits, not NWQLib's. A dense unitary
    that a backend lowers is synthesized by
    ``_dense_synthesis.dense_unitary_circuit``, which takes
    ``(22/48) 4**n - (3/2) 2**n + 5/3`` CX for a generic input but up to
    ``(25/48) 4**n - (3/2) 2**n + 2/3`` when optimization A.2 keeps three CX
    in its two-qubit blocks (that module's docstring). The dense dilations
    that ``build_block_encoding`` made of matrices near diagonal, scalar,
    rank-one or Kronecker-structured ones took up to 21 CX on three qubits
    and 105 on four, above the 20 and 100 returned here. Routing and the
    per-query law of the dense dilation therefore use this count as an
    estimate.
    """

    if total_qubits < 1:
        raise ValueError("total_qubits must be at least 1")
    if total_qubits == 1:
        return 0
    return (23 * 4**total_qubits - 72 * 2**total_qubits + 64) // 48


def block_encoding_per_query_cx(
    plan_detail: Mapping[str, Any],
    *,
    implementation: str,
    system_qubits: int,
) -> float | None:
    """Per-``U_BE``-query CX count from the implementation's synthesis law.

    The banded and multiplexed-Pauli branches consume the exact structural
    census created by ``BlockEncodingPlan``. Dense dilation uses the count
    of Shende, Bullock and Markov's optimized quantum Shannon decomposition
    that routing uses, an estimate that NWQLib's exact synthesis can exceed
    (:func:`dense_unitary_cx_qsd_upper_bound`). Other implementations have
    unknown synthesis cost. An arbitrary supplied circuit has no width-only
    bound on its length.
    """

    if implementation == "banded":
        counts = plan_detail.get("periodic_gate_counts", plan_detail)
        # One shared QFT/inverse-QFT pair on n system qubits. Each transform has
        # n(n-1)/2 controlled phases at two CX each and n//2 swaps at three CX
        # each. The remaining terms are the multiplexor, diagonal and PREP
        # censuses recorded by the banded construction.
        return (
            2.0 * system_qubits * (system_qubits - 1)
            + 6 * (system_qubits // 2)
            + int(counts["ucrz_multiplexor_gates"])
            * int(counts["ucrz_basis_cx_per_gate"])
            + int(counts["control_diagonal_basis_cx"])
            + int(counts["prep_pair_cx"])
        )
    if implementation in ("pauli_lcu", "multiplexed_pauli"):
        return float(
            int(plan_detail["ucg_core_cx"])
            + int(plan_detail["ucg_completion_diagonal_cx"])
            + int(plan_detail["coefficient_diagonal_cx"])
            + int(plan_detail["prep_pair_cx"])
        )
    if implementation == "dense_dilation":
        return dense_unitary_cx_qsd_upper_bound(system_qubits + 1)
    return None


@dataclass(frozen=True, kw_only=True)
class SampledBlock:
    """One representative circuit block and its multiplicity in the represented work.

    Args:
        name: Human-readable block name.
        circuit: Representative block circuit.
        multiplicity: Nonnegative integer count of this block in the represented work.
    """

    name: str
    circuit: QuantumCircuit
    multiplicity: int

    def __post_init__(self) -> None:
        """Admit ``multiplicity`` as a nonnegative integer."""
        object.__setattr__(self, "multiplicity", integer(self.multiplicity, "multiplicity", 0))
