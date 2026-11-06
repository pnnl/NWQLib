r"""Factorized coefficient and source preparation for the constant-source LCHS linear combination.

The finite LCHS solution is

    \[u_{\rm fin}=\sum_{a,j}w_a\nu_a e^{st_a}c_j e^{-it_a(k_jL_s+H)}|\psi_a\rangle,\]

with applications a (the initial state at ``t = T`` and the source
nodes at ``t_a = T - r_a``) and kernel nodes j. With the masses
``m_a = w_a nu_a e^{s t_a}`` and centered children ``L_0 = L - c_L I``,
``H_0 = H - c_H I``, the slot coefficient is
``|c_aj| = m_a |c_j|`` with phase
``phi_aj = arg c_j - t_a [k_j (c_L + s) + c_H]``, so the
coefficient magnitude factorizes:

    \[|p\rangle=\Big(\sum_a\sqrt{m_a/B_{\rm app}}|a\rangle\Big)\otimes
      \Big(\sum_j\sqrt{|c_j|/\beta_k}|j\rangle\Big),\qquad
      \gamma=B_{\rm app}\beta_k.\]

This needs O(A + M) data and two state preparations, never the A M slot
table. The phase does not factorize in general. For uniform nodes
``k_j = h j`` with label bits ``j = sum_l v_l b_l`` the coupled phase is
``prod_l exp(-i t_a (c_L + s) h v_l b_l)``: one application-register
diagonal per label bit, controlled on that bit, O(A log M) entries, plus
the sign bit for signed-magnitude labels. Nonuniform nodes use one
O(A M) phase table, admitted against an explicit entry limit. The kernel
phase ``arg c_j`` and the scalar ``-t_a c_H`` are separate O(M) and O(A)
diagonals. Each phase is applied once.

The source LCU circuit is ``P_c``, the kind-controlled state preparation
R, the evolution F on the joint register, the phase diagonal and
``P_c^dagger``; the success block maps ``|0>`` to ``u_fin / (zeta gamma)``.
R is never undone, and one preparation per input kind serves every
application of that kind.

Range-safe magnitudes. The masses are given as binary mantissa and
exponent. With ``E = max_a e_a - 1``, the relative masses
``r_a = ldexp(mu_a, e_a - E)`` lie in (0, 2) and ``Z = sum_a r_a`` in
[1, 2A), and with ``e_a - E = 2 h_a + d_a`` each amplitude is
``ldexp(sqrt(ldexp(mu_a, d_a)/Z), h_a)``, so no probability is formed. The
kernel magnitudes are the lifted ``K_j = 2^k |c_j|`` of the provider owner
(``algorithms.lchs.providers.lifted_magnitudes``) and
``gamma = 2^{E - k} Z beta`` with ``beta = sum_j K_j``. An application
whose amplitude is below the binary64 range can be omitted only within an
explicit physical allowance, with the bound ``sum_dropped m_a beta_raw``;
an exact-zero kernel coefficient is a structural zero, never an omission.
All amplitudes, sums and phase angles are binary64 evaluations, numerical
estimates without an enclosure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from nwqlib.subroutines.block_encoding.nodes import UniformNodeRecipe, uniform_node_labels

@dataclass(frozen=True, kw_only=True)
class FactorizedSourcePreparation:
    """The factorized coefficient state, coupled phases and recovery of one constant-source LCHS sum.

    Registers: the kernel register (``kernel_qubits``, the node labels of
    the joint generator) is low and the application register
    (``application_qubits``) high, matching the system order of
    ``plan_joint_generator`` (node register, application register, physical
    system).

    Attributes:
        application_amplitudes: ``sqrt(m_a / B_app)`` per application label,
            0 on padding and omitted applications.
        kernel_amplitudes: ``sqrt(|c_j| / beta_k)`` per kernel label.
        application_qubits: Width of the application register.
        kernel_qubits: Width of the kernel register.
        application_kinds: ``"initial"``, ``"source"`` or None (padding)
            per application label.
        application_times: Elapsed time ``t_a`` per application label.
        kind_qubit: Index, within the application register, of the bit that
            selects the source direction when both kinds are present, else
            None.
        kind: The one input kind when ``kind_qubit`` is None.
        recovery_mantissa: Mantissa of ``gamma = 2^{E-k} Z beta`` in
            [0.5, 1).
        recovery_exponent: Its binary exponent, an exact integer.
        phase_recipe: ``"none"`` (no coupled phase), ``"twos_complement"``
            or ``"signed_magnitude"`` (bit-conditioned diagonals) or
            ``"table"``.
        kernel_phases: ``arg c_j`` per kernel label.
        application_phases: ``-t_a c_H`` per application label.
        bit_phases: For a uniform recipe, per label bit l, the angles
            ``-t_a (c_L + s) h v_l`` per application label.
        table_nodes: For ``"table"``, the physical node ``k_j`` per node
            label j, from which ``build_coefficient_phase`` forms the angles
            ``-t_a (c_L + s) k_j`` at index ``j + 2**kernel_qubits * a``,
            else None.
        coupled_rate: ``c_L + s``, the scalar of the coupled phases.
        dropped_bound: ``(mantissa, exponent)`` of the omitted physical
            contribution, or None.
        omitted_applications: Input indices of omitted applications.
        data_entries: Number of amplitude and phase entries that building
            the preparation forms: the application and kernel amplitudes,
            the kernel and application phases, the bit phases and, for
            ``"table"``, the ``2**(kernel_qubits + application_qubits)``
            coupled angles, which the record does not store and
            ``build_coefficient_phase`` forms when it builds the diagonal.
            The LCHS construction work counts it.
    """

    application_amplitudes: tuple[float, ...]
    kernel_amplitudes: tuple[float, ...]
    application_qubits: int
    kernel_qubits: int
    application_kinds: tuple[str | None, ...]
    application_times: tuple[float, ...]
    kind_qubit: int | None
    kind: str | None
    recovery_mantissa: float
    recovery_exponent: int
    phase_recipe: str
    kernel_phases: tuple[float, ...]
    application_phases: tuple[float, ...]
    bit_phases: tuple[tuple[float, ...], ...]
    table_nodes: tuple[float, ...] | None
    coupled_rate: float
    dropped_bound: tuple[float, int] | None
    omitted_applications: tuple[int, ...]
    data_entries: int


def _width(count: int) -> int:
    return (count - 1).bit_length() if count > 1 else 0


def _reduced(angle: float) -> float:
    return math.remainder(angle, math.tau)


def plan_factorized_source_preparation(
    *, applications: Sequence[Any], kernel_magnitudes, kernel_lift: int, kernel_arguments, kernel_nodes=None,
    node_recipe: UniformNodeRecipe | None = None, scalar_phase=(0.0, 0.0), drop_allowance: float | None = None,
) -> FactorizedSourcePreparation:
    r"""Compute the factorized PREP amplitudes, coupled phases and recovery without building a circuit.

    Args:
        applications: Records with ``kind`` (``"initial"`` or
            ``"source"``), ``elapsed_time`` and ``mass``, a positive
            PhysicalScale-like value with ``mantissa`` and integer
            ``exponent`` (``algorithms.lchs.source_selection.LCHSApplication``).
        kernel_magnitudes: Lifted magnitudes ``K_j = 2^k |c_j|`` per physical
            node, in node order.
        kernel_lift (int): The lift k.
        kernel_arguments: ``arg c_j`` per physical node.
        kernel_nodes: Physical nodes ``k_j``; required without
            ``node_recipe``, where node j sits on label j.
        node_recipe (UniformNodeRecipe | None): Uniform nodes ``k_j = h j``,
            ``j = -J..J``, whose label mapping places node j and gives the
            bit-conditioned phases.
        scalar_phase: ``(c_L + s, c_H)``, the scalars taken out of the
            evolved children, default ``(0, 0)``.
        drop_allowance (float | None): Absolute physical allowance for
            omitted applications whose amplitude underflows, default None
            (refuse).

    Returns:
        preparation (FactorizedSourcePreparation): Amplitudes, phases,
        layout and recovery.

    Raises:
        ValueError: For no nonzero application or kernel coefficient,
            inconsistent lengths, or an underflowing amplitude without a
            covering allowance.
    """
    magnitudes = np.asarray(kernel_magnitudes, dtype=float)
    arguments = np.asarray(kernel_arguments, dtype=float)
    if magnitudes.ndim != 1 or magnitudes.shape != arguments.shape or not magnitudes.size:
        raise ValueError("kernel magnitudes and arguments must be nonempty one-dimensional and of equal length")
    if np.any(magnitudes < 0) or not np.all(np.isfinite(magnitudes)) or not np.all(np.isfinite(arguments)):
        raise ValueError("kernel magnitudes must be finite and nonnegative, arguments finite")
    lift = integer(kernel_lift, "kernel_lift", 0)
    beta = math.fsum(magnitudes)
    if beta == 0:
        raise ValueError("every kernel coefficient is zero; the solution is exactly zero")
    coupled, hamiltonian = (finite_real(value, "scalar_phase") for value in scalar_phase)
    if node_recipe is not None:
        if magnitudes.size != 2 * node_recipe.half_width + 1:
            raise ValueError("the node recipe and the kernel coefficients have different lengths")
        labels = np.asarray(uniform_node_labels(node_recipe))
        kernel_qubits = node_recipe.address_qubits
        nodes = None
    else:
        if kernel_nodes is None:
            raise ValueError("kernel_nodes is required without a node recipe")
        nodes = np.asarray(kernel_nodes, dtype=float)
        if nodes.shape != magnitudes.shape:
            raise ValueError("kernel_nodes must match the kernel coefficients")
        labels = np.arange(magnitudes.size)
        kernel_qubits = _width(magnitudes.size)
    kernel = np.zeros(1 << kernel_qubits)
    kernel[labels] = np.sqrt(magnitudes) / math.sqrt(beta)
    kernel_phases = np.zeros(1 << kernel_qubits)
    kernel_phases[labels] = np.where(magnitudes > 0, arguments, 0.0)

    records = [item for item in applications if item.mass.mantissa > 0]
    if not records:
        raise ValueError("every application is zero; the solution is exactly zero")
    top = max(item.mass.exponent for item in records) - 1

    def relative(item):
        # Exact integer exponent difference, clamped before any float use.
        return max(item.mass.exponent - top, -2200)

    z = math.fsum(math.ldexp(item.mass.mantissa, relative(item)) for item in records)
    amplitudes = []
    for item in records:
        half, odd = divmod(relative(item), 2)
        amplitudes.append(math.ldexp(math.sqrt(math.ldexp(item.mass.mantissa, odd) / z), half))
    omitted = tuple(index for index, value in enumerate(amplitudes) if value == 0.0)
    dropped = None
    if omitted:
        # Each omitted application contributes at most m_a beta_raw = m_a 2^-k beta (unitary branches).
        terms = [(records[index].mass.mantissa * beta, records[index].mass.exponent - lift) for index in omitted]
        exponent = max(e for _, e in terms)
        mantissa, shift = math.frexp(math.fsum(math.ldexp(m, max(e - exponent, -2200)) for m, e in terms))
        dropped = (mantissa, exponent + shift)
        try:
            upper = math.ldexp(*dropped)
        except OverflowError:
            upper = math.inf
        if drop_allowance is None or upper > finite_real(drop_allowance, "drop_allowance"):
            raise ValueError(f"{len(omitted)} application amplitude(s) are below the binary64 range; their physical "
                             f"bound is {dropped[0]}*2**{dropped[1]}, which needs an explicit drop_allowance")
        records = [item for index, item in enumerate(records) if index not in omitted]
        amplitudes = [value for value in amplitudes if value > 0.0]
        z = math.fsum(math.ldexp(item.mass.mantissa, relative(item)) for item in records)
    initial = [index for index, item in enumerate(records) if item.kind == "initial"]
    sources = [index for index, item in enumerate(records) if item.kind == "source"]
    if len(initial) > 1 or len(initial) + len(sources) != len(records):
        raise ValueError("applications must be at most one initial application and source applications")
    if initial and sources:
        source_bits = _width(len(sources))
        application_qubits = source_bits + 1
        kind_qubit = source_bits
        slots = [0] + [(1 << source_bits) | index for index in range(len(sources))]
        order = initial + sources
        kind = None
    else:
        application_qubits = _width(len(records))
        kind_qubit = None
        slots = list(range(len(records)))
        order = list(range(len(records)))
        kind = records[0].kind
    size = 1 << application_qubits
    application = np.zeros(size)
    kinds: list[str | None] = [None] * size
    times = np.zeros(size)
    for slot, index in zip(slots, order, strict=True):
        application[slot] = amplitudes[index]
        kinds[slot] = records[index].kind
        times[slot] = float(records[index].elapsed_time)
    application_phases = np.array([_reduced(-t * hamiltonian) if kinds[a] else 0.0 for a, t in enumerate(times)])
    bit_phases: tuple[tuple[float, ...], ...] = ()
    table_nodes, table_entries = None, 0
    if coupled == 0.0:
        recipe = "none"
    elif node_recipe is not None:
        recipe = node_recipe.mapping
        h = node_recipe.spacing
        r = node_recipe.address_qubits
        if recipe == "twos_complement":
            weights = [1 << bit for bit in range(r - 1)] + [-(1 << (r - 1))]
        else:
            weights = [1 << bit for bit in range(r - 1)]
        bit_phases = tuple(tuple(_reduced(-t * coupled * h * weight) if kinds[a] else 0.0
                                 for a, t in enumerate(times)) for weight in weights)
    else:
        recipe = "table"
        # build_coefficient_phase forms the 2**(k + a) angles when it builds the diagonal.
        table_nodes = tuple(float(v) for v in nodes)
        table_entries = (1 << kernel_qubits) * size
    mantissa, shift = math.frexp(z * beta)
    entries = size + (1 << kernel_qubits) * 2 + size + sum(len(row) for row in bit_phases) + table_entries
    return FactorizedSourcePreparation(
        application_amplitudes=tuple(float(v) for v in application), kernel_amplitudes=tuple(float(v) for v in kernel),
        application_qubits=application_qubits, kernel_qubits=kernel_qubits, application_kinds=tuple(kinds),
        application_times=tuple(float(v) for v in times), kind_qubit=kind_qubit, kind=kind,
        recovery_mantissa=mantissa, recovery_exponent=top - lift + shift, phase_recipe=recipe,
        kernel_phases=tuple(float(v) for v in kernel_phases),
        application_phases=tuple(float(v) for v in application_phases), bit_phases=bit_phases,
        table_nodes=table_nodes, coupled_rate=coupled, dropped_bound=dropped, omitted_applications=omitted,
        data_entries=entries)


def build_coefficient_preparation(preparation: FactorizedSourcePreparation, *, factors=None,
                                  kernel_preparation=None):
    """Build ``P_c``, the tensor product of the kernel and application state preparations, on [work, kernel, application].

    Args:
        preparation (FactorizedSourcePreparation): The layout.
        factors: Optional ``(kernel, application)`` circuits that prepare the
            two magnitude factors on their own registers, such as selected
            MPS circuits; a None entry, or no pair, uses the direct tree.
            ``P_c`` is their tensor product, whose inverse is the adjoint of
            the same realized circuits.
        kernel_preparation: Optional gate that prepares the kernel factor
            with its own work register, on [work, kernel], such as the
            analytic Low–Somma PREP (``analytic_prep.build_analytic_kernel_query``);
            its work qubits come first in the returned circuit and are left
            in their final state, which a caller projects on zero. It
            replaces the kernel entry of ``factors``.

    Returns:
        circuit (QuantumCircuit): ``P_c |0> = |p>``, on the work qubits of
        ``kernel_preparation`` (none otherwise), then kernel and application.
    """
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.state_preparation import build_qiskit_state_preparation

    k, a = preparation.kernel_qubits, preparation.application_qubits
    work = 0 if kernel_preparation is None else kernel_preparation.num_qubits - k
    circuit = QuantumCircuit(work + k + a, name="factorized_coefficient_prep")
    kernel_factor, application_factor = factors or (None, None)
    if kernel_preparation is not None:
        circuit.append(kernel_preparation, circuit.qubits[:work + k])
    elif k:
        circuit.compose(build_qiskit_state_preparation(np.asarray(preparation.kernel_amplitudes)).circuit
                        if kernel_factor is None else kernel_factor, qubits=circuit.qubits[:k], inplace=True)
    if a:
        circuit.compose(build_qiskit_state_preparation(np.asarray(preparation.application_amplitudes)).circuit
                        if application_factor is None else application_factor, qubits=circuit.qubits[work + k:],
                        inplace=True)
    return circuit


def build_coefficient_phase(preparation: FactorizedSourcePreparation, *, max_bytes: int | None = None):
    r"""Build the slot phase diagonal ``Phi = diag(exp(i phi_aj))`` on [kernel, application].

    For uniform recipes it is the kernel diagonal, the application diagonal
    and one application diagonal per label bit controlled on that bit (with
    the sign bit for signed-magnitude labels, where ``j = (1 - 2 s) sum_l
    2^l b_l``). For a table it is one diagonal over both registers, whose
    ``2**(k + a)`` angles are formed here.

    Args:
        preparation (FactorizedSourcePreparation): The layout.
        max_bytes (int | None): Default None, the physical memory. Limit on
            the angle table of the ``"table"`` recipe, 8 bytes per entry
            for the angles and 16 for their copies in the diagonal.

    Returns:
        circuit (QuantumCircuit): The diagonal.
    """
    from qiskit import QuantumCircuit
    from nwqlib.subroutines._multiplexors import append_control_diagonal_phases

    k, a = preparation.kernel_qubits, preparation.application_qubits
    circuit = QuantumCircuit(k + a, name="coefficient_phase")
    kernel, application = list(circuit.qubits[:k]), list(circuit.qubits[k:])
    append_control_diagonal_phases(circuit, kernel, preparation.kernel_phases)
    append_control_diagonal_phases(circuit, application, preparation.application_phases)
    size = 1 << a
    if preparation.phase_recipe == "twos_complement":
        for bit, angles in enumerate(preparation.bit_phases):
            append_control_diagonal_phases(circuit, [*application, kernel[bit]], [0.0] * size + list(angles))
    elif preparation.phase_recipe == "signed_magnitude":
        sign = kernel[-1]
        for bit, angles in enumerate(preparation.bit_phases):
            negated = [-value for value in angles]
            append_control_diagonal_phases(circuit, [*application, kernel[bit], sign],
                                           [0.0] * size + list(angles) + [0.0] * size + negated)
    elif preparation.phase_recipe == "table":
        entries = 1 << (k + a)
        _check_bytes(24 * entries, max_bytes, "coupled-phase table")
        nodes = np.asarray(preparation.table_nodes)
        labels = np.arange(nodes.size)
        values = np.zeros(entries)
        for slot, (kind, t) in enumerate(zip(preparation.application_kinds, preparation.application_times,
                                             strict=True)):
            if kind:
                values[labels + (slot << k)] = [_reduced(-t * preparation.coupled_rate * node) for node in nodes]
        append_control_diagonal_phases(circuit, [*kernel, *application], values)
    return circuit


def build_kind_state_preparation(preparation: FactorizedSourcePreparation, *, initial=None, source=None):
    """Build R, which prepares the input direction of each application from ``|0>`` by kind, on [application, system].

    With both kinds present the initial preparation is controlled on the
    kind bit being 0 and the source preparation on it being 1; their global
    phases become the relative phase of the two inputs. Each preparation
    must produce the exact complex direction, phase included.

    Args:
        preparation (FactorizedSourcePreparation): The layout.
        initial: Circuit preparing ``u_0 / ||u_0||``, when present.
        source: Circuit preparing ``b / ||b||``, when present.

    Returns:
        circuit (QuantumCircuit): R.
    """
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.qiskit_compat import controlled

    a = preparation.application_qubits
    # An empty circuit, the preparation of |0>, is falsy, so presence is tested against None.
    width = (initial if initial is not None else source).num_qubits
    circuit = QuantumCircuit(a + width, name="kind_state_prep")
    system = circuit.qubits[a:]
    if preparation.kind_qubit is None:
        prep = initial if preparation.kind == "initial" else source
        circuit.compose(prep, qubits=system, inplace=True)
        return circuit
    bit = circuit.qubits[preparation.kind_qubit]
    for value, prep in ((0, initial), (1, source)):
        circuit.append(controlled(prep.to_gate(label=f"prep_{value}"), 1, ctrl_state=value), [bit, *system])
    return circuit


def build_source_lcu_circuit(preparation: FactorizedSourcePreparation, evolution, *, initial=None, source=None,
                             factors=None, kernel_preparation=None):
    r"""Assemble the source LCU circuit: ``P_c``, R, the evolution F, the phase diagonal and ``P_c^dagger``.

    ``evolution`` is a block encoding of ``exp(-i T_e S)`` with recovery
    ``zeta = evolution.alpha``, on the joint system [kernel, application,
    physical]. With all ancillas, the kernel and the application registers
    projected on zero, the circuit maps the physical ``|0>`` to
    ``u_fin / (zeta gamma)``. ``factors`` and ``kernel_preparation`` are the
    optional factor circuits of `build_coefficient_preparation`; ``P_c^dagger``
    is the inverse of the same realized ``P_c``.

    A ``kernel_preparation`` with its own work register W (the analytic
    kernel PREP) keeps W from ``P_c`` to ``P_c^dagger``: R, the evolution and
    the phase diagonal never act on it, and the success projector includes
    it. With ``p_hat = P_c |0>`` on [W, kernel, application] the projected
    block is ``<p_hat| (I_W (x) Phi F R) |p_hat>``, the factorized relation
    above on the enlarged coefficient space (S4, MATH_TRIAGE Q3). The inverse
    is ``qiskit_compat.inverse_realized_gate`` of the realized ``P_c`` gate.

    Returns:
        circuit (QuantumCircuit): On [evolution ancillas, W, kernel,
        application, physical system]; W is empty without a
        ``kernel_preparation``.
    """
    from qiskit import QuantumCircuit

    k, a = preparation.kernel_qubits, preparation.application_qubits
    e = evolution.num_ancillas
    prep = build_coefficient_preparation(preparation, factors=factors, kernel_preparation=kernel_preparation)
    w = prep.num_qubits - k - a
    circuit = QuantumCircuit(e + w + evolution.system_qubits, name="factorized_source_lcu")
    ancillas = list(circuit.qubits[:e])
    work = list(circuit.qubits[e:e + w])
    address = list(circuit.qubits[e + w:e + w + k + a])
    system = list(circuit.qubits[e + w + k + a:])
    if kernel_preparation is None:
        unprepare = prep.inverse()
    else:
        from nwqlib.subroutines.qiskit_compat import inverse_realized_gate
        gate = prep.to_gate(label=prep.name)
        prep = QuantumCircuit(gate.num_qubits, name=gate.label)
        prep.append(gate, prep.qubits)
        unprepare = QuantumCircuit(gate.num_qubits, name=gate.label + "_dg")
        unprepare.append(inverse_realized_gate(gate, native_ucg=False), unprepare.qubits)
    circuit.compose(prep, qubits=[*work, *address], inplace=True)
    circuit.compose(build_kind_state_preparation(preparation, initial=initial, source=source),
                    qubits=[*address[k:], *system], inplace=True)
    circuit.compose(evolution.circuit, qubits=[*ancillas, *address, *system], inplace=True)
    circuit.compose(build_coefficient_phase(preparation), qubits=address, inplace=True)
    circuit.compose(unprepare, qubits=[*work, *address], inplace=True)
    return circuit


__all__ = [
    "FactorizedSourcePreparation",
    "build_coefficient_phase",
    "build_coefficient_preparation",
    "build_kind_state_preparation",
    "build_source_lcu_circuit",
    "plan_factorized_source_preparation",
]
