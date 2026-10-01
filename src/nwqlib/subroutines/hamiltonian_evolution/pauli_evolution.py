"""Qiskit circuit builders for compact Pauli-evolution blocks.

A block with generator G = sum_j c_j P_j and ``time_step`` dt approximates
exp(-i*dt*G) by Qiskit's PauliEvolutionGate with the selected Lie-Trotter or
Suzuki synthesis, applied in term order. The result is exact when the terms
commute. Two block kinds have exact special forms. A
``number_projector`` block applies exp(-i*angle*(prod_q n_q - I/2**s)) on
its support of s qubits, with n_q = (I - Z_q)/2. A ``kinetic`` block made of
an XX and a YY term with equal real coefficient c on the same pair applies
exp(-i*dt*c*(XX + YY)) as one XXPlusYYGate, which is exact because XX and YY
commute. The CX laws below mirror Qiskit's multi-controlled phase synthesis
and NWQLib's diagonal synthesis for the pinned Qiskit version. They have no
paper source and are checked against transpiled circuits by the QHD routing
tests.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
from nwqlib.subroutines.hamiltonian_evolution.pauli_ir import (
    PauliEvolutionBlock,
    coerce_pauli_evolution_block,
    pauli_terms_to_dicts,
)
from nwqlib.subroutines.hamiltonian_evolution.pauli_labels import (
    parse_pauli_label,
    sparse_pauli_op_from_terms,
)

PauliEvolutionSynthesis = Literal["lie_trotter", "suzuki_trotter"]


def _dirty_mcx_cx(num_controls: int) -> int:
    """Return the CX count of one multi-controlled X with ``num_controls`` controls.

    This is Qiskit's ``synth_mcx_n_dirty_i15``, which the MCRZ synthesis
    uses. For one, two and three controls it builds dedicated circuits,
    counted here as 1, 6 and 14 CX. For four or more controls its docstring
    gives at most ``8 k - 6`` CX with ``k - 2`` dirty ancillas. The QHD
    routing tests check the resulting MCPhase counts against transpiled
    circuits.
    """

    if num_controls == 1:
        return 1
    if num_controls == 2:
        return 6
    if num_controls == 3:
        return 14
    return 8 * num_controls - 6


def _mcphase_cx(support_size: int) -> int:
    """Return the CX count of Qiskit's MCPhase gate on ``support_size`` qubits.

    With ``c = support_size - 1 >= 2`` controls, Qiskit's ``MCPhaseGate``
    definition emits MCRZ gates with ``c, c - 1, ..., 1`` controls and one
    final phase gate. The one-control MCRZ is a CRZ with two CX, and a
    one-control MCPhase is a CPhase, also with two CX. An MCRZ with
    ``k >= 2`` controls uses ``_mcsu2_real_diagonal``, which applies two
    multi-controlled X gates on ``ceil(k/2)`` controls and two on
    ``floor(k/2)`` controls. The QHD routing tests compare these counts with
    transpiled circuits of the pinned Qiskit version.
    """

    total = 0
    for num_controls in range(1, support_size):
        if num_controls == 1:
            total += 2
        else:
            total += 2 * _dirty_mcx_cx((num_controls + 1) // 2)
            total += 2 * _dirty_mcx_cx(num_controls // 2)
    return total


def structured_number_projector_provider(support_size: int) -> dict[str, Any]:
    """Choose the cheaper of two exact circuits for a number-projector phase on s qubits, and return its CX count.

    The diagonal provider is NWQLib's phase-diagonal synthesis with one
    nonzero phase, ``2**s - 2`` CX on s qubits by the Theorem 7 recursion of
    Shende, Bullock and Markov, quant-ph/0406176v5 (p. 10), whose
    multiplexed Rz with k select bits costs ``2**k`` CX by the count after
    their Theorem 8 (p. 11). The other provider is Qiskit's MCPhase, whose
    CX count follows the gate's definition in the pinned Qiskit version. The cheaper one is
    selected, and a tie selects MCPhase. The diagonal is cheaper for s = 4
    to 7 (14 against 20 CX at s = 4), and MCPhase is cheaper again from
    s = 8 (220 against 254 CX), because its count grows quadratically in s
    while the diagonal's grows as ``2**s``. QHD omits exact-zero projector
    blocks before routing and resource pricing. The diagonal provider emits
    no gate when its angle is exactly zero.
    """

    if support_size <= 0:
        raise ValueError("number-projector support must be nonempty")
    diagonal_cx = 2**support_size - 2
    mcphase_cx = _mcphase_cx(support_size)
    if diagonal_cx < mcphase_cx:
        return {"provider": "diagonal_synthesis", "cx": diagonal_cx}
    return {"provider": "mcphase", "cx": mcphase_cx}


def resolve_number_projector_lowering(support_size: int) -> dict[str, Any]:
    """Return the structured projector lowering and its circuit-free CX price.

    ``costs[selected]`` is the exact CX count of the provider that
    ``append_number_projector_phase`` realizes for a nonzero angle.
    """

    provider = structured_number_projector_provider(support_size)
    return {
        "selected": "structured",
        "costs": {"structured": provider["cx"]},
        "structured_provider": provider,
    }


# Fixed capacity for each caller-owned wrapped-phase cache. Reuse depends on
# call order. Eviction permits repeated array-shaped wraps while bounding the
# scalar cache independently of the number of compilation steps and angles.
# Registered in docs/ENGINEERING_CONSTANTS.md.
WRAPPED_PHASE_CACHE_ENTRIES = 64


def wrapped_projector_phase(angle: float, support_size: int, cache: dict | None = None) -> float:
    """Return the absolute wrapped phase of a projector on support_size qubits.

    With s=support_size, the calculation uses the same 2**s phase array and
    NumPy operations as the circuit construction of
    `append_number_projector_phase`: phases=(0,...,0,-angle), followed by
    numpy.angle(numpy.exp(1j*phases)). The final entry is reproduced bit for
    bit. Range selection accepts an exactly zero wrapped phase and requires
    a nonzero phase to remain normal after its power-of-two scaling.

    Potential compilation and each rotation-census call own separate
    caches. Within either owner, an equal (float(angle).hex(), int(s)) key
    reuses its cached scalar while that entry is present. The two owners
    use the same wrapped value but do not share cache entries.

    cache, when supplied, is the caller's dictionary. For a cache initialized
    empty and populated only by this helper, at most
    WRAPPED_PHASE_CACHE_ENTRIES entries remain after a call. A miss computes
    the original array-shaped wrap, inserts one Python float and its key,
    and evicts the oldest inserted entry if the capacity is exceeded.
    Insertion can briefly hold one extra entry. Hits do not refresh the
    insertion order. An evicted key is recomputed on its next occurrence,
    so the capacity bounds storage without guaranteeing a hit rate. Each
    miss has work proportional to 2**s. On a miss, the numerical array
    peak is 40*2**s bytes, consisting of the real phase array and simultaneous
    complex input and output of exp. wrapped_phase_workspace_bytes adds the
    qualified fixed allowance for the bounded cache, insertion overlap and
    NumPy bookkeeping. Potential compilation admits this as a successive phase
    and clears its cache after step compilation. The one-hot rotation census
    admits its own cache and miss beside its source records, accumulating
    population and caller-held buffers through the QHD one-hot resource count.
    """
    import numpy as np

    key = (float(angle).hex(), int(support_size))
    if cache is not None and key in cache:
        return cache[key]
    phases = np.zeros(2**support_size)
    phases[-1] = -float(angle)
    wrapped = abs(float(np.angle(np.exp(1.0j * phases))[-1]))
    if cache is not None:
        cache[key] = wrapped
        if len(cache) > WRAPPED_PHASE_CACHE_ENTRIES:
            del cache[next(iter(cache))]
    return wrapped


def wrapped_phase_workspace_bytes(entries):
    """Price one scalar cache and one array-shaped miss, for a reached owner.

    entries is the largest 2**s used by that owner, or zero when no wrap is
    reached. A miss keeps one float64 array and two complex128 arrays at
    its numerical peak, 40*entries bytes. The 65536-byte engineering
    allowance covers the bounded dictionary, at most 65 scalar entries
    during insertion, dictionary resize overlap and fixed NumPy bookkeeping
    on the qualified CPython 3.12.14 / NumPy 2.5.2 stack.

    With E = entries, the lifetimes of ``numpy.angle(numpy.exp(1.0j * phases))``
    are: the product by i holds ``phases`` and one complex128 array (24E),
    exp holds ``phases`` with its complex128 input and output (40E), and
    angle holds ``phases``, the exp output and the float64 result (32E). One
    miss runs at a time, so no factor of the cache capacity multiplies 40E.
    The fixed term is an engineering allowance: a 65-entry inventory of key
    tuples, finite-float hex strings, integers and floats is below 12.3 KB
    before dictionary storage, and fill/churn peaks including miss scratch
    were below 20 KB on that stack.
    """
    return 65536 + 40 * int(entries) if entries else 0


def append_number_projector_phase(
    circuit: QuantumCircuit,
    support: Sequence[int],
    angle: float,
) -> None:
    """Append ``exp(-i angle (prod n_q - I/2**s))`` exactly.

    The explicit global phase ``angle / 2**s`` removes the identity component
    ``I/2**s`` of the projector's Pauli expansion.  The compiler separately
    records the omitted physical identity contribution.
    For the structured diagonal provider, an exact-zero angle emits no gate,
    and QHD omits zero-angle blocks before calling this builder or pricing
    them.
    """

    qubits = tuple(int(qubit) for qubit in support)
    if not qubits:
        raise ValueError("number-projector support must be nonempty")
    circuit.global_phase += float(angle) / (2 ** len(qubits))
    provider = structured_number_projector_provider(len(qubits))["provider"]
    if provider == "diagonal_synthesis":
        phases = [0.0] * (2 ** len(qubits))
        phases[-1] = -float(angle)
        from nwqlib.subroutines._multiplexors import append_control_diagonal_phases

        append_control_diagonal_phases(
            circuit, [circuit.qubits[index] for index in qubits], phases
        )
    elif len(qubits) == 1:
        circuit.p(-float(angle), qubits[0])
    else:
        circuit.mcp(-float(angle), list(qubits[:-1]), qubits[-1])


def _matched_xx_yy(block: PauliEvolutionBlock) -> tuple[int, int, float] | None:
    """Return (q0, q1, c) when a kinetic block is exactly c*(XX + YY) on one pair.

    Only this exact form may use XXPlusYYGate. Coefficients must be equal
    after an imaginary residue of at most 1e-12 is discarded, the same
    admission as the Trotter bound path. Any other block returns None and
    uses the general PauliEvolutionGate.
    """
    if block.kind != "kinetic" or len(block.terms) != 2:
        return None
    parsed = [parse_pauli_label(term.pauli) for term in block.terms]
    if any(len(ops) != 2 for ops in parsed):
        return None
    qubits = tuple(sorted(parsed[0]))
    if tuple(sorted(parsed[1])) != qubits:
        return None
    if {tuple(parsed[0].values()), tuple(parsed[1].values())} != {("X", "X"), ("Y", "Y")}:
        return None
    coefficients = [complex(term.coefficient) for term in block.terms]
    # Absolute imaginary window in the coefficient's units, the same literal
    # as trotterization/error_budget.py. QHD kinetic coefficients are real, so
    # it matters only for caller-built blocks. Registered in
    # docs/ENGINEERING_CONSTANTS.md. Revisit for a relative window or another
    # precision.
    if any(abs(value.imag) > 1.0e-12 for value in coefficients):
        return None
    if coefficients[0].real != coefficients[1].real:
        return None
    return qubits[0], qubits[1], coefficients[0].real


def make_evolution_synthesis(
    method: PauliEvolutionSynthesis = "lie_trotter",
    options: Mapping[str, Any] | None = None,
) -> object:
    """Return a Qiskit synthesis object for ``PauliEvolutionGate``."""

    from qiskit.synthesis import LieTrotter, SuzukiTrotter

    synthesis_options = dict(options or {})
    if method == "lie_trotter":
        return LieTrotter(**synthesis_options)
    if method == "suzuki_trotter":
        return SuzukiTrotter(**synthesis_options)
    raise ValueError(f"unsupported evolution synthesis method: {method}")


def apply_pauli_rotation(
    circuit: QuantumCircuit,
    pauli_label: str,
    angle: float,
    *,
    control: int | None = None,
) -> None:
    """Append the Pauli rotation `exp(-i angle P / 2)` to a circuit, optionally controlled on a separate qubit.

    The general case maps each X to Z with H and each Y to Z with S^dagger
    then H, collects the parity of the support onto its last qubit with a CX
    ladder, applies RZ(angle), and undoes the ladder and the basis change.
    `RZ(angle) = exp(-i*angle*Z/2)` carries no extra global phase, so replacing
    it by CRZ gives exactly the controlled rotation.

    Args:
        circuit (QuantumCircuit): Circuit to append to, in place.
        pauli_label (str): Compact label of `P`, such as `"x0z2"`.
        angle (float): Rotation angle.
        control (int | None): Default `None`. Index of a control qubit that
            is not in the support of `P`.

    Raises:
        ValueError: If the control qubit is in the support or outside the
            circuit.
    """

    ops = parse_pauli_label(pauli_label)
    qubits = sorted(ops)
    if control is not None and (control in ops or not 0 <= control < circuit.num_qubits):
        raise ValueError("Pauli rotation control must be a separate qubit in the circuit")

    if control is None and len(qubits) == 1:
        qubit = qubits[0]
        op = ops[qubit]
        if op == "X":
            circuit.rx(angle, qubit)
        elif op == "Y":
            circuit.ry(angle, qubit)
        elif op == "Z":
            circuit.rz(angle, qubit)
        return

    if control is None and len(qubits) == 2 and ops[qubits[0]] == ops[qubits[1]]:
        q0, q1 = qubits
        op = ops[q0]
        if op == "X":
            circuit.rxx(angle, q0, q1)
            return
        if op == "Y":
            circuit.ryy(angle, q0, q1)
            return
        if op == "Z":
            circuit.rzz(angle, q0, q1)
            return

    for qubit in qubits:
        op = ops[qubit]
        if op == "X":
            circuit.h(qubit)
        elif op == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)

    for parity_control, target in zip(qubits[:-1], qubits[1:]):
        circuit.cx(parity_control, target)
    if control is None:
        circuit.rz(angle, qubits[-1])
    else:
        # The basis/parity circuit cancels in the inactive control branch.
        circuit.crz(angle, control, qubits[-1])
    for parity_control, target in reversed(list(zip(qubits[:-1], qubits[1:]))):
        circuit.cx(parity_control, target)

    for qubit in reversed(qubits):
        op = ops[qubit]
        if op == "X":
            circuit.h(qubit)
        elif op == "Y":
            circuit.h(qubit)
            circuit.s(qubit)


def _hopping_parameter(time_step: float, coefficient: float) -> float:
    """Return the ``XXPlusYYGate`` parameter ``theta = 4 dt c`` of ``exp(-i dt c (XX + YY))``.

    Qiskit defines ``XXPlusYYGate(theta) = exp(-i theta (XX + YY)/4)``. The
    product ``2 dt c`` is formed first, in the evaluation order of the
    stored angle of a QHD hopping block (``kinetic.KineticCompiler``), and
    then doubled, which is exact unless it overflows. A finite theta is
    therefore twice that angle in every range. Forming ``4 dt`` first would
    overflow for a large dt even where theta is finite.

    Raises:
        ValueError: theta is not a finite binary64 number.
    """
    theta = (2.0 * time_step * coefficient) * 2.0
    if not math.isfinite(theta):
        raise ValueError(
            f"the XXPlusYYGate parameter 4*dt*c with dt = {time_step!r} and c = {coefficient!r} is not a finite "
            "binary64 number"
        )
    return theta


def append_pauli_evolution_block(
    circuit: QuantumCircuit,
    block: PauliEvolutionBlock | Mapping[str, Any],
    *,
    evolution_synthesis: PauliEvolutionSynthesis = "lie_trotter",
    evolution_synthesis_options: Mapping[str, Any] | None = None,
) -> None:
    """Append the circuit of one Pauli-evolution block to `circuit`, in place.

    A `"number_projector"` block uses its exact phase circuit, an exact
    `c (XX + YY)` `"kinetic"` block one `XXPlusYYGate`, and any other block
    one Qiskit `PauliEvolutionGate` on all qubits of the circuit.

    Args:
        circuit (QuantumCircuit): Circuit to append to.
        block (PauliEvolutionBlock | Mapping): The block, or a mapping with
            the same fields.
        evolution_synthesis (str): Default `"lie_trotter"`. Product formula
            of the `PauliEvolutionGate`: `"lie_trotter"` (Qiskit
            `LieTrotter`) or `"suzuki_trotter"` (Qiskit `SuzukiTrotter`).
        evolution_synthesis_options (Mapping | None): Default `None`.
            Keyword arguments passed to that Qiskit synthesis class.

    Raises:
        ValueError: If a `"number_projector"` block lacks `support` or
            `angle`, or the synthesis name is not supported.
    """

    from qiskit.circuit.library import PauliEvolutionGate, XXPlusYYGate

    coerced = coerce_pauli_evolution_block(block)
    if coerced.kind == "number_projector":
        if coerced.support is None or coerced.angle is None:
            raise ValueError("number-projector blocks require support and angle")
        append_number_projector_phase(circuit, coerced.support, coerced.angle)
        return

    hopping = _matched_xx_yy(coerced)
    if hopping is not None:
        q0, q1, coefficient = hopping
        circuit.append(XXPlusYYGate(_hopping_parameter(coerced.time_step, coefficient)), [q0, q1])
        return

    operator = sparse_pauli_op_from_terms(
        pauli_terms_to_dicts(coerced.terms),
        num_qubits=circuit.num_qubits,
    )
    gate = PauliEvolutionGate(
        operator,
        time=coerced.time_step,
        synthesis=make_evolution_synthesis(evolution_synthesis, evolution_synthesis_options),
    )
    circuit.append(gate, circuit.qubits)


def build_pauli_evolution_circuit(
    blocks: Sequence[PauliEvolutionBlock | Mapping[str, Any]],
    *,
    num_qubits: int,
    evolution_synthesis: PauliEvolutionSynthesis = "lie_trotter",
    evolution_synthesis_options: Mapping[str, Any] | None = None,
) -> QuantumCircuit:
    """Build a Qiskit circuit that applies Pauli-evolution blocks in order.

    Args:
        blocks (Sequence[PauliEvolutionBlock | Mapping]): Blocks in the
            order they act.
        num_qubits (int): Number of circuit qubits.
        evolution_synthesis (str): Default `"lie_trotter"`. Product formula
            for blocks of the default kind, as for
            `append_pauli_evolution_block`.
        evolution_synthesis_options (Mapping | None): Default `None`.
            Keyword arguments of that Qiskit synthesis class.

    Returns:
        circuit (QuantumCircuit): The circuit on `num_qubits` qubits.

    Examples:
        `0.5 Z_0` and `0.3 X_1` commute, so one Lie-Trotter step of length
        0.7 equals `exp(-0.7i (0.5 Z_0 + 0.3 X_1))`:

        >>> import numpy as np
        >>> from scipy.linalg import expm
        >>> from qiskit.quantum_info import Operator
        >>> from nwqlib.subroutines.hamiltonian_evolution import (
        ...     PauliEvolutionBlock, PauliEvolutionTerm,
        ...     build_pauli_evolution_circuit, sparse_pauli_op_from_terms)
        >>> terms = (PauliEvolutionTerm(pauli="z0", coefficient=0.5),
        ...          PauliEvolutionTerm(pauli="x1", coefficient=0.3))
        >>> block = PauliEvolutionBlock(terms=terms, time_step=0.7)
        >>> circuit = build_pauli_evolution_circuit([block], num_qubits=2)
        >>> H = Operator(sparse_pauli_op_from_terms(
        ...     [{"pauli": "z0", "coefficient": 0.5},
        ...      {"pauli": "x1", "coefficient": 0.3}], 2)).data
        >>> print(np.allclose(Operator(circuit).data, expm(-0.7j * H)))
        True
    """

    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(num_qubits)
    for block in blocks:
        append_pauli_evolution_block(
            circuit,
            block,
            evolution_synthesis=evolution_synthesis,
            evolution_synthesis_options=evolution_synthesis_options,
        )
    return circuit


__all__ = [
    "PauliEvolutionSynthesis",
    "append_number_projector_phase",
    "append_pauli_evolution_block",
    "apply_pauli_rotation",
    "build_pauli_evolution_circuit",
    "make_evolution_synthesis",
    "resolve_number_projector_lowering",
    "structured_number_projector_provider",
]
