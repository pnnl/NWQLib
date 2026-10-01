# Supply operators and states

Pass familiar numerical data directly to a scientific Problem. For example, `LinearSystem(A=[[2, 1], [1, 2]], b=[1, 0])` owns a snapshot of that matrix and right-hand side. The method chooses its construction and any necessary embedding. Input admission does not pad, symmetrize, normalize an operator, or change units.

Use input handles explicitly when you want to reuse an admitted input or choose a compact preparation:

```python
from nwqlib.operators import operator_input, ingest_pauli, PeriodicStencil
from nwqlib.problems import state_input, ingest_product, ingest_occupation

A = operator_input([[2, 1], [1, 2]])
b = state_input([3, 4j])
observable = ingest_pauli([("I", 1), ("Z", 1)], num_qubits=1)
basis_state = ingest_occupation("0", num_qubits=1)
product_state = ingest_product([[1, 1j], [1, 0]])
structured = operator_input(PeriodicStencil(100, mass=.1, diffusion=.1, potential=.05))
```

`operator_input` accepts ordinary numerical ndarrays, lists or tuples, canonical SciPy CSR/CSC arrays or matrices, Qiskit `SparsePauliOp`, `PeriodicStencil`, or an existing `OperatorInput`. `state_input` accepts numerical vectors, Qiskit `Statevector` or `QuantumCircuit`, or an existing `StateInput`. Known SDK values load their SDK owner only when supplied. Unknown array protocols, object arrays, unsized numerical streams, ragged rows, bool values and nonfinite data reject. Sparse and compact data are never implicitly densified.

Input admission preserves a representation; it does not promise that every Method can consume it. For ordinary nonzero-time/nonconstant workflows:

| Method | Quantum input access | Explicit classical input access |
| --- | --- | --- |
| ExpectationMethod | Finite Pauli, or dense Hermitian observable with dimension ≤16 | Dense/CSR/CSC/Pauli matvec in the original dimension |
| QLS | Dense, finite Pauli, supported periodic input, or an explicitly original-bound supplied encoding | Dense A and explicit numerical RHS |
| LCHS | Dense A or supported PeriodicStencil | Dense A |
| QPE | Dense Hamiltonian/unitary, or finite Pauli Hamiltonian | Dense spectral input |
| Lanczos / FixedGCIM / ADAPT | Finite Pauli; default small dense conversion, or explicit `input_conversion="dense_pauli"` | Original dense/CSR/CSC/Pauli access, with Method-specific preparation rules |

The eigen Methods' explicit dense-Pauli conversion can accept CSR/CSC, but it has its own work/byte admission and is not a sparse quantum oracle. Default `auto` does not silently select that sparse conversion. Supplying `execution="classical"` does not add a CSR solver to QLS/LCHS/QPE. Initial-condition and algebraic identity shortcuts keep their separate domains. The algorithm guides, which the [home page](index.md) lists by scientific problem, give each Method's output and preparation requirements.

Real and integer data become float64; complex data become complex128. Existing handles are returned unchanged. Supported arrays are copied into immutable byte-backed storage, so changing the caller's arrays cannot change a past input. Hermitian structure means exact equality of the stored matrix and its adjoint. A nearly Hermitian matrix remains general; an eigenproblem can reject it without silently changing the problem.

For a canonical Pauli sum, exact Hermiticity means that every coalesced coefficient is real. `read_xacc` and `parse_xacc` preserve the complex coefficients specified by the fermionic input. If the intended eigenproblem uses the Hermitian part of their result, construct it explicitly with `SparsePauliOp(op.paulis, coeffs=op.coeffs.real)` and pass that operator to `Eigenproblem`. This operation implements `(op+op.adjoint())/2` coefficient by coefficient and keeps every Pauli row, including a row whose new coefficient is zero. A default-tolerance `simplify()` call can also remove small nonzero real terms, so it is not part of this projection. The operator-norm change is at most the sum of the absolute imaginary coefficients. Choosing this Hermitian part is a modeling decision.

Each input operation has `max_bytes=10_000_000_000`, checked before the relevant conversion or snapshot. This bounds the operation's peak live arrays, the bytes of the arrays it keeps alive at the same time, not process RSS or arbitrary SDK work. A 100-qubit product or periodic input uses compact data, so the limit is not a universal qubit cap. A later simulation has its own width, memory, circuit and data limits.

## Coordinates, scale and phase

Coordinate index increases in computational order; qubit 0 is rightmost. Pauli label `IX` applies X to qubit 0. Product row j belongs to qubit j, so the physical tensor order is row q-1 through row 0. Occupation strings are q0-first: `"10"` means qubits `(1,0)`, computational index 1, printed ket `|01>`. No electron count, spin or sector claim is inferred from the syntax.

Physical amplitudes remain available. For `b=[3,4j]`, the numerical norm is 5 and the preparation direction is `[.6,.8j]`. Division uses a positive norm, so negative or complex scaling preserves its physical phase. Vector normalization happens once at admission. Product inputs normalize each two-amplitude factor without forming a full state. `PhysicalScale` stores a binary mantissa/exponent; it can represent a product norm outside float64 range. Its `as_float()` or `squared_as_float()` returns `None` for an unavailable nonzero value, rather than reporting zero. These are numerical values, not certified error bounds.

Zero data are legal physical inputs but have no normalized quantum state. Small non-power-of-two vectors keep their original length and scale; a selected method owns any padded preparation. Metadata does not silently provide that construction.

`prepare_qiskit(state, max_bytes=..., max_direct_amplitudes=65_536)` explicitly constructs a supported unitary preparation from the already normalized direction. It performs no backend run, transpilation or simulation. The amplitude count limits direct vector synthesis only; compact product/occupation preparations and legal input dimensions are unaffected. Planning and estimates accept larger direct inputs, and a Run synthesizes them under its own `ExecutionLimits.max_direct_amplitudes`, which has the same default. A supplied `QuantumCircuit` means its unitary action on the all-zero input. Admission copies stored circuit data, preserves global phase and shared definitions, rejects unsupported/nonunitary structure and free parameters, and does not request lazy definitions. Its unit norm is a supplied unitary premise. `bind_preparation_circuit` additionally accepts an explicit `reference` and matching `basis` when the source has its own identity.

## Explicit classical operations

`OperatorInput.dense_array()`, `entry(...)`, `pauli_terms()` and `sparse_entries(max_bytes=...)` read existing admitted data. They do not invent quantum access. `matvec(vector, max_bytes=..., max_products=1_000_000_000)` performs an explicit classical action. The scalar-product count is derived from the actual representation, not measured CPU time.

Pauli label and mask inputs combine only exact identical words and discard only exact combined zero. Stable summation keeps nonzero cancellation residues. `table.product(other, max_bytes=...)` preserves ordered raw contributions until the selected final coalescing step. `table.group(strategy="qwc" or "commuting", max_comparisons=1_000_000, max_bytes=...)` performs first-fit grouping and charges each candidate tile against the remaining comparison budget before testing it. General commuting groups are algebraic partitions, not automatically single measurement settings.

`fermion_table` and `ingest_fermion` preserve ordered `(mode, action)` strings; action 0 annihilates and 1 creates, with the rightmost operator acting first. `to_pauli(mapping="jw", max_bytes=...)` explicitly maps them with lower-mode parity. `mapping="z_free"` selects qubit ladders without Jordan–Wigner parity. No normal ordering or full-state action is implicit.

`FactorizedOperatorProduct((A, B, ...))` keeps the ordered handles. Its `matvec` acts right to left and checks the total scalar products across all factors. Its explicit `expand` supports homogeneous Pauli, ordered Fermion, or CSR/CSC factors. Sparse candidate multiplications also use `max_products`. A compact product can remain actionable when expansion exceeds its own limit.

`ingest_df(T, factors, constant_energy=E0, orbital_basis=..., energy_unit=..., source=...)` admits the real supplied polynomial

```text
B_l = V_l diag(w_l) V_l^T
H   = E0 I + dΓ(T) + (1/2) sum_l dΓ(B_l)^2
```

T must be exactly symmetric. Signed/zero weights and nonorthogonal V columns are legal. `to_pauli(max_bytes=..., max_products=1_000_000_000)` explicitly performs the contractions and ordered JW conversion. It checks the complete quartic output frontier first. The returned operator, receipt and work fact describe that selected conversion; its numerical error remains unknown.

Here `dΓ(A) = sum_{p,q,σ} A[p,q] a†(p,σ) a(q,σ)`. For an electronic Hamiltonian with one-body integrals `h` and chemists'-notation two-body integrals `g[p,q,r,s] = sum_l B_l[p,q] B_l[r,s]`, T is the shifted matrix `T[p,s] = h[p,s] - (1/2) sum_q g[p,q,q,s]`. The conversion normal-orders each square as `dΓ(B)²/2 = dΓ(B²)/2 + (1/2) sum B[p,q] B[r,s] a†(p,σ) a†(r,τ) a(s,τ) a(q,σ)` and places spin orbital `(p, σ)` on mode `2p+σ`.

`refine_operator_facts(..., refinements=(...))` explicitly acquires Pauli L1 or Hermitian sparse Gershgorin bounds using exact fractions of the stored binary64 values. Empty refinements reuse metadata. Repeating a scan creates a new acquisition receipt; reuse does not relabel prior work as a new zero-cost scan. These bounds describe the admitted operator, not total physical output error.

## Conventions and derivations

The relations below are standard identities or NWQLib conventions. The named owner writes each one out in its docstring or an adjacent comment. [Library-wide conventions](CODE_TOUR.md#library-wide-conventions) gives the qubit, phase, unit and normalization conventions of the whole library.

| Relation | Justification | Owner |
| --- | --- | --- |
| Coordinate index `k = sum_q b_q 2**q`, qubit 0 rightmost | Qiskit little-endian order | `operators/inputs.py` (`ORDER`) |
| Stored row `(x, z)` means `i**popcount(x & z) X**x Z**z`, so both bits set is Y | NWQLib encoding | `operators/_pauli.py` (`PauliTerms`) |
| `SparsePauliOp` coefficient of the canonical label is `coeff * (-i)**phase` | Qiskit phase convention | `operators/inputs.py` (`operator_input`) |
| Dense matrix to Pauli coefficients `c_P = tr(P M)/D` by 2×2 block recursion | Standard identity | `operators/_pauli.py` (`pauli_coefficients`) |
| Pauli product phase and the action of a word on a basis state | Standard Pauli algebra in the stored encoding | `PauliTerms.product`, `apply_terms` |
| Commuting test | Even symplectic product | `PauliTerms.group` |
| Qubit-wise-commuting test | Equal single-qubit Paulis on the shared support | `PauliTerms.group` |
| `a_j† = (prod_{k<j} Z_k)(X_j - iY_j)/2` | Jordan–Wigner transformation | `operators/_fermion.py` (`_ladder_image`) |
| Normal ordering of `dΓ(B)²/2` | Standard anticommutator algebra | `operators/df.py` (`_contract`) |
| Spin orbital `(p, σ)` on mode `2p+σ` | NWQLib convention | `operators/df.py` (`_rows`) |
| Pauli L1 norm and spectral enclosure | Triangle inequality, unit-norm Pauli words | `operators/refinement.py` (`_pauli_values`) |
| Sparse spectral enclosure and norm bound | Gershgorin circle theorem | `operators/refinement.py` (`_sparse_values`) |

## Metadata and saved data

Handles expose `.basis`, `.reference` and `.to_record()`. The last operation projects their manifest and structure/preparation metadata on demand without copying or hashing native data. `.from_record(record)` creates a metadata-only handle: every native accessor explicitly rejects unavailable data, even if the original manifest lists those capabilities. Result/Run archives separately save the actual selected data and native bindings. A descriptive input record alone cannot recreate them.

See [input cost controls](development/input_contracts.md) for the size laws and the boundaries of explicit expansions.
