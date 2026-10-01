# Input cost controls

Input admission functions check concrete data before conversion, snapshots, hashing or explicit expansion. The default `max_bytes` is `10_000_000_000`. Each operation compares it with an upper bound on its peak live arrays, the bytes of the arrays alive at the same time at the peak of that operation, computed before allocating as far as the code knows them. The count excludes caller-owned inputs unless the table lists them, SDK and LAPACK internal workspace, Python object overhead and process RSS. [Byte and work budgets](../CODE_TOUR.md#byte-and-work-budgets) explains why these budgets are checked before the work they bound. Methods and backends own later construction/execution limits. Metadata projection performs no numerical action. `prepare_qiskit(max_direct_amplitudes=65_536)` separately limits the direct vector magnitude/phase synthesis with O(q*2**q) arithmetic. It does not limit compact preparations or symbolic input/Plan dimensions.

| Operation | Peak live arrays checked before work |
| --- | --- |
| Dense input | D² float64 or complex128 entries |
| CSR/CSC input | Converted values plus original indices and indptr bytes |
| Physical vector | Physical float64/complex128 entries plus 16D direction bytes |
| Product preparation input | Physical entries plus 32q direction bytes; no 2**q vector |
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

The Pauli rows also reserve `S` bytes per term for the identity JSON of the Plan records that will hold the term. `Record.content_id` holds two JSON copies beside the record's own data, and the records' own label and coefficient data take at most one more copy. The widest per-term JSON is `J=2q+120` bytes, which holds the Pauli term record of an Expectation Plan with a 24-character binary64 repr, the label and coefficient again in its readout spec and 4 bytes of list brackets, so `S=3J=6q+360`. [Engineering constants](../ENGINEERING_CONSTANTS.md#record-identity-json) gives the measured Plans and the Methods that charge their own records.

These data laws preserve before-work refusal, but do not measure SDK allocations or time spent in a caller's explicit term iterator. Sized term excess rejects before iteration. Unsized Pauli/Fermion constructors keep admitted entries plus one lookahead. Natural numerical adapters accept only known sized containers.

`matvec(max_products=1_000_000_000)` limits the representation's derived scalar products: D² for dense, nnz+D for sparse, and `(M+g)+3M+D(M+g+1)+D+M*D` for M stored Pauli terms with the metadata bound `g=min(M,D)` on distinct single-word flip masks. The Pauli allowance includes preprocessing, grouped action and a possible ordered retry after overflow. Factorized action sums the selected factors' counts before reading the vector. A method that repeats an action must additionally account for its actual iteration or query population before the overall computation; per-action limits are not a claim that any number of repeated actions is cheap.

`PauliTerms.group(max_comparisons=1_000_000)` charges each candidate tile against the remaining comparison budget before testing it. The worst case is M(M-1)/2 comparisons, but it is not reserved in advance. Each comparison visits the actual word width. It produces original term indices and the actual comparison count, without a dense graph. Neither grouping strategy claims optimality. Local controls are never silently increased after refusal and do not limit symbolic resource estimates.

For raw Fermion input, M strings with P ordered ladder entries use 24M+9P+8 bytes. Six layouts plus dimension metadata cover conversion, snapshots and coalescing. A p-operator string maps to 2**p raw Pauli rows. The complete mapper checks local two-row ladders, every intermediate ordered product, final scaling, destination arrays, coalescing and the share `S` of each mapped row before its first product. Count arithmetic is capped before forming huge powers. No nonzero coefficient is thresholded.

For sparse expansion A*B, candidate scalar products are `C=sum(row_nnz(B,k) for each stored A(i,k))`; C bounds structural and final nnz. Later factors use the previous nnz upper bound times the next maximum row nnz. `expand(max_products=1_000_000_000)` checks cumulative candidates before native multiplication, and `max_bytes` covers normalization copies, multiplication and sorting workspace, and final immutable storage. No prefix multiplication runs merely to discover a later refusal. Final output uses explicit duplicate/zero coalescing and sorted indices; stale SciPy format flags are not trusted.

DF conversion admits `1+2*n²+4*n⁴` raw Fermion rows and `1+8*n²+64*n⁴` raw JW candidates when factors are present. The quartic contributions vanish for an empty factor tuple. For r factors and k total columns, its dense contractions use `n²*k+r*n³+r*(n(n+1)/2)²` scalar products. Numerical and mapping arrays are checked before contraction; no full spin-orbital tensor or reference operator is constructed. Receipt cost fields remain derived laws, not CPU measurements.

Structural refinement streams exact `Fraction` arithmetic over binary64 input. Every finite value has a denominator dividing 2**1074 and magnitude below 2**1024. A sum's numerator needs at most `2098+bit_length(number_of_terms)` bits, and endpoint cross comparisons add at most 1074 bits. Before each scan, `refine_operator_facts` checks an allowance of 32 exact-integer slots of `3174+bit_length(number_of_terms)` bits each, which adds two bits for the Gershgorin radius and diagonal offset to the widths above (`operators/refinement.py::_scalar_frontier`, registered in [engineering constants](../ENGINEERING_CONSTANTS.md#numerical-choices-and-representation-sizes)). It publishes the input-entry/byte count, derived scalar operations and output numerator/denominator bytes under a fresh acquisition identity. Reports share the scientific facts of previous scans and never repeat a scan merely to display them.

A dense matrix admitted here can later be synthesized into gates, which costs of order M³ work for an M-square unitary, far more than its intake. That synthesis has its own law, `_dense_synthesis.dense_synthesis_size`, or `controlled_synthesis_size` for a directly controlled unitary. Qiskit's control of the synthesized gates has its own law, `gatewise_control_size`. A Method that controls a unitary built from such a matrix, or a supplied circuit that holds one, charges the synthesis and Qiskit's control of it at planning, and a subroutine builder before its construction, against its own work and byte limits. A backend that lowers an uncontrolled one to a gate basis, and lowering inside a Run that controls a transformed block, reserve the work against the Run's `max_synthesis_work`, a total over all its preparations, before the first synthesis. The backend synthesizes each distinct matrix once per Run object. `lower_qiskit` called outside a Run checks its own `max_synthesis_work` ([selected blocks](../blocks.md#admission-of-the-exact-synthesis)).

Arrays are immutable byte-backed snapshots. Sparse storage keeps its original CSR/CSC orientation and index dtype. Input hashes bind stored dtype, shape, ordering and content once. Exact Hermitian equality is a structural input check. No eigenvalue calculation or symmetrization is performed. Native access on metadata-only handles always rejects. Source identity, preparation phase and physical scale cannot be independently rebound on a live handle.

## Input invariants and their reasons

These rules are NWQLib's own design. Each protects a later consumer from a specific failure.

| Invariant | Failure it prevents | Owner |
| --- | --- | --- |
| Admission copies supported arrays into immutable byte-backed snapshots and hashes dtype, shape, order and content once | A caller's later mutation changes the input of an existing Plan, or a differently stored input shares another's identity | `operators.inputs._freeze_array`, `_digest` |
| Hermitian structure is exact equality of the stored values, with no tolerance or symmetrization | A nearly Hermitian matrix is silently replaced by a different operator | `operators.inputs.ingest_dense`, `ingest_sparse`, `problems.records.Eigenproblem` |
| Admission and conversion remove only sums that are exactly zero. Thresholds on nonzero values are Method controls | Small nonzero coefficients vanish without a record of what was removed | `operators._pauli.combine_terms`, `pauli_coefficients`, `FactorizedOperatorProduct.expand` |
| Input handles never densify sparse, Pauli, Fermion, product or periodic data implicitly | A hidden dense allocation behind a compact input | `OperatorInput.dense_array`, `sparse_entries`, `matvec` |
| Size and product laws are checked before conversion, copying or multiplication | Memory or work limits discovered only after partial computation | `operators.access._check_bytes`, `_check_products`, `operators._factorized._sparse_chain_requirements` |
| A metadata record never restores native access | A description is treated as the data it describes, and work runs on data that was never admitted | `OperatorInput.from_record`, `StateInput.from_record` |
| Physical scale is stored beside the normalized direction, and normalization divides once by a positive norm | Magnitude, sign or complex phase of the scientific input is lost | `problems.inputs.PhysicalScale`, `ingest_vector` |
| Structural refinement publishes exact rationals and a fresh receipt for each scan | A rounded value is reported as a certified bound, or reused facts are relabeled as a new zero-cost scan | `operators.refinement.refine_operator_facts` |
