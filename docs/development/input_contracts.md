# Input cost controls

The functions that accept operator and state inputs check concrete data before conversion, snapshots, hashing or explicit expansion. This page gives the default limits, what they count, and the byte or work formula of each input operation.

## Default and what it counts

The default `max_bytes` is `10_000_000_000`. Each operation compares it with an upper bound on its peak live arrays, the bytes of the arrays alive at the same time at the peak of that operation. The bound is computed before allocation, from what the code knows at that point. Metadata projection performs no numerical action. [Byte and work budgets](../CODE_TOUR.md#byte-and-work-budgets) explains why these budgets are checked before the work they bound.

## What it does not count

The count excludes caller-owned inputs unless the table lists them, SDK and LAPACK internal workspace, Python object overhead and process RSS.

These formulas keep the refusal before the work, but they do not measure SDK allocations or the time spent in a caller's explicit term iterator. A sized term collection that exceeds the limit is rejected before iteration. Unsized Pauli and Fermion constructors keep the accepted entries plus one lookahead. The natural numerical adapters accept only containers of known size.

## Other limits

- Methods and backends set their own limits for later construction and execution.
- `prepare_qiskit(max_direct_amplitudes=65_536)` separately limits the direct synthesis of a vector's magnitudes and phases, whose arithmetic is O(q*2**q). It does not limit compact preparations or the dimensions of a symbolic input or Plan.
- `matvec(max_products=...)`, `PauliTerms.group(max_comparisons=...)` and `expand(max_products=...)` limit work, as [Matrix-vector action](#matrix-vector-action), [Pauli grouping](#pauli-grouping) and [Sparse products](#sparse-products) describe.

## Peak live arrays by operation

| Operation | Peak live arrays checked before work |
| --- | --- |
| Dense input | D² float64 or complex128 entries |
| CSR/CSC input | Converted values plus original indices and indptr bytes |
| Physical vector | Physical float64/complex128 entries plus 16D direction bytes |
| Product preparation input | Physical entries plus 32q direction bytes, with no 2**q vector |
| Occupation input | q bytes plus two dimension-integer slots |
| Periodic stencil | `48 + ceil(bit_length(q)/8) + 2*ceil((q+1)/8)` bytes |
| Pauli labels | `R*(q+4L+S)`, with `L=16*ceil(q/64)+16` and `S=6q+360` |
| Pauli masks | `R*(4L+S)` plus dimension metadata |
| Qiskit `SparsePauliOp` | `R*(5L+S)` plus dimension metadata, because the converted masks and coefficients stay alive through the mask conversion |
| Pauli label output | `M*(q+16)` returned payload plus `t*(q+25)+q+4` bytes for a tile of `t<=min(M,1024)` rows |
| Ordered Pauli product | `2*M*K*L+64*C*W+128*C`, with `W=ceil(q/64)` and blocks of `C=min(max(M,K),1024)` rows of the longer factor |
| First-fit Pauli grouping | `M*L+64*t*W+64*t+32*W`, with `W=ceil(q/64)` and candidate tiles of `t<=min(M,1024)` terms |
| Matrix-vector action | 33D bytes for dense/sparse, `49D+24M+16g+8+max(9M,96*min(D,1024))` for M stored Pauli terms, using the metadata bound `g=min(M,D)` on distinct single-word flip masks |
| Supplied circuit snapshot | Stored arrays and scalars and circuit, gate and reference slots, with a shared payload copied once. The per-object allowances are in [engineering constants](../ENGINEERING_CONSTANTS.md#supplied-circuit-snapshot-allowances) |

The Pauli rows also reserve `S` bytes per term for the content-hash JSON of the Plan records that will hold the term. `Record.content_id` holds two JSON copies beside the record's own data, and the records' own label and coefficient data take at most one more copy. The widest per-term JSON is `J=2q+120` bytes, which holds the Pauli term record of an Expectation Plan with a 24-character binary64 repr, the label and coefficient again in its readout spec and 4 bytes of list brackets, so `S=3J=6q+360`. [Engineering constants](../ENGINEERING_CONSTANTS.md#record-identity-json) gives the measured Plans and the Methods that count their own records.

## Matrix-vector action

`matvec(max_products=1_000_000_000)` limits the scalar products that the representation needs: D² for dense, nnz+D for sparse, and `(M+g)+3M+D(M+g+1)+D+M*D` for M stored Pauli terms with the metadata bound `g=min(M,D)` on distinct single-word flip masks. The Pauli allowance includes preprocessing, grouped action and a possible ordered retry after overflow. Factorized action sums the counts of the chosen factors before reading the vector. A method that repeats an action must also account for its whole number of iterations or queries before the overall computation. A per-action limit does not claim that any number of repeated actions is cheap.

## Pauli grouping

`PauliTerms.group(max_comparisons=1_000_000)` counts each candidate tile against the remaining comparison budget before testing it. The worst case is M(M-1)/2 comparisons, but it is not reserved in advance. Each comparison visits the masks' word width. Grouping returns the original term indices and the number of comparisons made, without building a dense graph. Neither grouping strategy claims optimality. These local controls are never silently increased after a refusal, and they do not limit symbolic resource estimates.

## Fermion input

For raw Fermion input, M strings with P ordered ladder entries use 24M+9P+8 bytes. Six layouts plus dimension metadata cover conversion, snapshots and coalescing. A p-operator string maps to 2**p raw Pauli rows. Before its first product, the complete mapper checks the local two-row ladders, every intermediate ordered product, the final scaling, the destination arrays, coalescing and the share `S` of each mapped row. Count arithmetic is capped before huge powers are formed. No nonzero coefficient is thresholded.

## Sparse products

For a sparse product A*B, the candidate scalar products are `C=sum(row_nnz(B,k) for each stored A(i,k))`, and C bounds both the structural and the final nnz. Later factors use the previous nnz upper bound times the next maximum row nnz. `expand(max_products=1_000_000_000)` checks the cumulative candidates before native multiplication, and `max_bytes` covers normalization copies, multiplication and sorting workspace, and the final immutable storage. No prefix multiplication runs merely to discover a later refusal. The final output uses explicit duplicate and zero coalescing and sorted indices. SciPy's format flags are not trusted, because they can be stale.

## Double-factorized input

Double-factorized (DF) conversion accepts `1+2*n²+4*n⁴` raw Fermion rows and `1+8*n²+64*n⁴` raw Jordan–Wigner candidates when factors are present. The quartic contributions vanish for an empty factor tuple. For r factors and k total columns, its dense contractions use `n²*k+r*n³+r*(n(n+1)/2)²` scalar products. Numerical and mapping arrays are checked before contraction, and no full spin-orbital tensor or reference operator is constructed. The cost fields in its receipt are derived formulas, not CPU measurements.

## Structural refinement

Structural refinement streams exact `Fraction` arithmetic over binary64 input. Every finite value has a denominator dividing 2**1074 and a magnitude below 2**1024. A sum's numerator needs at most `2098+bit_length(number_of_terms)` bits, and endpoint cross comparisons add at most 1074 bits. Before each scan, `refine_operator_facts` checks an allowance of 32 exact-integer slots of `3174+bit_length(number_of_terms)` bits each, which adds two bits for the Gershgorin radius and diagonal offset to the widths above (`operators/refinement.py::_scalar_frontier`, registered in [engineering constants](../ENGINEERING_CONSTANTS.md#numerical-choices-and-representation-sizes)). It publishes the number of input entries and bytes, the derived scalar operations and the output numerator and denominator bytes under a fresh acquisition identity. Reports share the scientific facts of previous scans and never repeat a scan merely to display them.

## Dense matrices that are later synthesized

A dense matrix accepted here can later be synthesized into gates, which costs of order M³ work for an M-square unitary, far more than its intake. That synthesis has its own formula, `_dense_synthesis.dense_synthesis_size`, or `controlled_synthesis_size` for a directly controlled unitary. Qiskit's control of the synthesized gates has its own formula, `gatewise_control_size`. A Method that controls a unitary built from such a matrix, or a supplied circuit that holds one, counts the synthesis and Qiskit's control of it against its own work and byte limits at planning, and a subroutine builder does so before its construction. A backend that lowers an uncontrolled one to a gate basis, and lowering inside a Run that controls a transformed block, reserve the work against the Run's `max_synthesis_work`, a total over all its preparations, before the first synthesis. The backend synthesizes each distinct matrix once per Run object. `lower_qiskit` called outside a Run checks its own `max_synthesis_work` ([Exact dense synthesis](../development/dense_synthesis.md#admission-of-the-exact-synthesis)).

## Stored input arrays

Arrays are immutable byte-backed snapshots. Sparse storage keeps its original CSR/CSC orientation and index dtype. Input hashes bind the stored dtype, shape, ordering and content once. Exact Hermitian equality is a structural input check, and no eigenvalue calculation or symmetrization is performed. Native access on a metadata-only handle is always rejected. The source identity, the preparation phase and the physical scale cannot be rebound independently on a live handle.

## Input invariants and their reasons

These rules are NWQLib's own design. The rules on snapshots and hashes, exact Hermitian structure, exact-zero coalescing, densification, physical scale and metadata-only records are indexed, with their tests, in [Design rationale](design_rationale.md#input-admission). The two rules below protect later code that uses the input.

| Invariant | Failure it prevents | Code |
| --- | --- | --- |
| Size and product formulas are checked before conversion, copying or multiplication | Memory or work limits discovered only after partial computation | `operators.access._check_bytes`, `_check_products`, `operators._factorized._sparse_chain_requirements` |
| Structural refinement publishes exact rationals and a new entry in the report's `receipts` for each scan | A rounded value is reported as a certified bound, or reused facts are relabeled as a new zero-cost scan | `operators.refinement.refine_operator_facts` |
