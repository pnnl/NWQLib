"""SDK-free size laws of the existing direct state-preparation kernels."""

def direct_preparation_cx_bound(num_qubits: int, *, complex_phases: bool = True) -> int:
    """Return the generic magnitude-tree and optional phase-diagonal CX slot bound.

    Each tree has sum(2**k, k=1..n-1) = 2**n-2 CX gates. Exact fast paths
    and zero tables can use fewer slots. This law counts the native
    constructor without synthesis or simulation.

    The magnitude tree is the cascade of uniformly controlled RY rotations
    with k = 0, ..., n-1 controls of Mottonen, Vartiainen, Bergholm and
    Salomaa, arXiv:quant-ph/0407010v1, Sec. III, Eq. (7) (p. 3). Their
    Sec. II (p. 2) gives 2**k CX for a uniformly controlled rotation with
    k >= 1 controls. The k = 0 rotation is a plain rotation with no CX, so a
    tree has sum(2**k, k=1..n-1) CX. The phase diagonal has the same count by
    Shende, Bullock and Markov, arXiv:quant-ph/0406176v5, Theorems 7 and 8
    (pp. 10-11). Mottonen et al., arXiv:quant-ph/0407010v1, also cancel one
    CX in each uniformly
    controlled rotation with at least one control, two per level, by
    mirroring (Sec. III, pp. 3-4), and state that preparation from a basis
    state needs half of their total (Sec. IV, p. 4). That gives
    ``2**(n+1) - 2n - 2`` CX. This law does not assume the cancellation, so
    its complex-case value ``2*max(0, 2**n - 2)`` is never smaller.
    """

    if num_qubits < 0:
        raise ValueError("num_qubits must be nonnegative")
    return (1 + int(complex_phases)) * max(0, (1 << num_qubits) - 2)


def _uniform_superposition_gate_slots(num_qubits: int) -> dict[str, int]:
    """Bound elementary slots in the selected contiguous-uniform fast path.

    For non-power-of-two support, UniformSuperpositionGate has at most n-1
    initial X, n-2 H, one RY, n-1 open CH and n-2 open CRY. CH expands to two
    H, four phase gates and one CX. CRY expands to two RY and two CX. Each
    open control adds two X. Summing gives the returned counts, for example
    ``x = (n-1) + 2(n-1) + 2(n-2) = 5n-7`` and ``cx = (n-1) + 2(n-2) = 3n-5``.
    The resulting envelope also covers the power-of-two H-only path.

    The gate structure is Qiskit's ``UniformSuperpositionGate`` definition,
    which implements Algorithm 1 of Shukla and Vedula, arXiv:2306.11747v2,
    p. 4. For M = sum of 2**l_j over j = 0, ..., k with
    l_0 < ... < l_k <= n-1, their Sec. 2.5 (p. 14) counts k X, l_0 H, one
    RY, l_k - l_0 open controlled H and k - 1 open controlled RY. A
    non-power-of-two M has k >= 1, so these counts are at most n-1, n-2, 1,
    n-1 and n-2. Their Fig. 6 (pp. 15-16) builds a controlled H from one CX
    and a controlled RY from two, the same CX counts as Qiskit's definitions.
    In Qiskit 2.5.2, the lower bound of the declared dependency, the sampled
    support sizes 2**(n-1)+1, 3*2**(n-2)+1 and 2**n-1 for n = 2, ..., 6 stay
    within these top-level counts, and support 2**n-1 reaches the 3n-5 CX
    envelope after lowering. A different Qiskit construction needs a new
    count.
    """
    if num_qubits < 0:
        raise ValueError("num_qubits must be nonnegative")
    if num_qubits < 2:
        return {"h": num_qubits}
    return {"x": 5 * num_qubits - 7, "h": 3 * num_qubits - 4,
            "ry": 2 * num_qubits - 3, "p": 4 * (num_qubits - 1),
            "cx": 3 * num_qubits - 5}


def direct_preparation_controlled_cx_bound(num_qubits: int) -> int:
    """Bound one-control direct PREP using the complete complex tree envelope.

    With D = 2**n, each magnitude or phase tree has at most D-1 rotation
    slots and D-2 CX slots. One extra control costs at most two CX per
    rotation and six per CX, so the two complex trees cost at most
    ``2 * 2(D-1) + 6 * 2(D-2) = 16D - 28`` CX. A controlled global phase is a
    one-qubit phase. The contiguous-uniform fast path is also covered: for
    n>=2 its open CH/CRY decomposition costs at most 41*n-59 CX, below
    16*2**n-28. Basis and full-uniform inputs admit tighter counts at their
    known structural owner. This counts elementary synthesis slots before
    cancellation, not a hardware mapping guarantee.

    The two-CX controlled one-qubit gate is the circuit of Shende, Bullock and
    Markov, arXiv:quant-ph/0406176v5, Sec. 3.1 (p. 9). The six-CX count is
    the textbook Toffoli circuit reproduced by Shende and Markov,
    arXiv:0803.2316v1, Fig. 1 (p. 3). Their Theorem 1 shows that no Toffoli
    circuit of CX and one-qubit gates uses fewer CX. The registered entry is
    "Controlled direct PREP slot bound" in ENGINEERING_CONSTANTS.md.
    """
    if num_qubits < 0:
        raise ValueError("num_qubits must be nonnegative")
    rotations = 2 * max(0, (1 << num_qubits) - 1)
    uniform = _uniform_superposition_gate_slots(num_qubits)
    # One added control turns X into one CX, a one-qubit H, RY or phase gate
    # into a two-CX controlled gate, and CX into a six-CX Toffoli. For n >= 2
    # these weights sum the fast-path slots to 41*n - 59.
    uniform_cx = sum(count * (6 if gate == "cx" else 1 if gate == "x" else 2) for gate, count in uniform.items())
    return max(2 * rotations + 6 * direct_preparation_cx_bound(num_qubits), uniform_cx)

