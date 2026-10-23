# Supply inputs

<a id="supply-operators-and-states"></a>Pass your numerical data directly to a Problem. For example, `LinearSystem(A=[[2, 1], [1, 2]], b=[1, 0])` stores a copy of that matrix and right-hand side, and the Method chooses its circuit construction and any embedding it needs. [Choose a problem and output](problems.md) lists the Problems and their fields, and the [input API](api/inputs.md) documents each function on this page. For a molecule, `build_gcim_chemistry_problem` builds the Hamiltonian from a PySCF geometry and basis and needs the `chemistry` extra ([chemistry inputs](algorithms/gcim.md#chemistry-inputs-and-reference-energies)).

## Inputs each Method accepts

A Problem keeps each input in the representation you supply, and not every Method can use every representation:

| Method | Quantum execution | `execution="classical"` |
| --- | --- | --- |
| ExpectationMethod | Finite Pauli, or dense Hermitian observable with dimension ≤16 | Dense, CSR, CSC or Pauli matrix-vector products in the original dimension |
| QLS | Dense, finite Pauli, supported periodic input, or a supplied `SelectedBlock` encoding bound to the original A | Dense A and an explicit numerical right-hand side |
| LCHS | Dense A or supported `PeriodicStencil` | Dense A |
| QPE | Dense Hamiltonian or unitary, or finite Pauli Hamiltonian | Dense spectral input |
| Lanczos, FixedGCIM, ADAPT | Finite Pauli input is accepted directly, dense matrices of dimension at most 16 are converted automatically, and larger dense matrices or CSR/CSC input require `input_conversion="dense_pauli"`, with all conversions subject to `max_conversion_work` and `max_bytes`. | Original dense, CSR, CSC or Pauli access, with Method-specific preparation rules |

The table describes the general case. LCHS at zero elapsed time, or with a zero initial state and no nonzero source, returns the initial condition without evolution and has its own input domain. Algebraic identity shortcuts, such as an identity-only observable that needs no measurement ([Pauli expectation](algorithms/expectation.md)), also have their own input domains.

The explicit dense-Pauli conversion of Lanczos, FixedGCIM and ADAPT accepts CSR and CSC, with its own work and byte limits, and it is not a sparse quantum oracle. The default `input_conversion="auto"` does not select that conversion for sparse input. `execution="classical"` adds no CSR solver to QLS, LCHS or QPE. The algorithm guides, which the [home page](index.md) lists by scientific problem, give each Method's outputs and preparation requirements.

## What NWQLib keeps as given

- **No padding.** A small non-power-of-two vector keeps its original length and scale. A Method that needs a power-of-two size builds its own padded preparation, and the input metadata does not supply that construction.
- **No symmetrizing.** Hermitian means that the stored matrix equals its adjoint exactly. A nearly Hermitian matrix stays general, and an Eigenproblem rejects it instead of changing the problem. If the Hermitian part of a matrix `A` is your intended problem, pass `(A + A.conj().T) / 2` explicitly.
- **No operator normalization.** An operator keeps its scale. A vector's physical norm stays available beside its unit-length preparation direction ([coordinates, scale and phase](#coordinates-scale-and-phase)).
- **No unit change.** A unit is a label ([units](problems.md#units)).
- **No densifying.** Sparse and compact data are never converted to dense arrays implicitly.

## Input objects {#input-handles}

An input object is the `OperatorInput` or `StateInput` that `operator_input`, `state_input` and functions such as `ingest_pauli` return. Use one to reuse an input without converting it again, or to choose a compact preparation:

```python
from nwqlib.operators import operator_input, ingest_pauli, PeriodicStencil
from nwqlib.problems import state_input, ingest_product, ingest_occupation

A = operator_input([[2, 1], [1, 2]])
b = state_input([3, 4j])
observable = ingest_pauli([("I", 1), ("Z", 1)], num_qubits=1)
basis_state = ingest_occupation("0", num_qubits=1)
product_state = ingest_product([[1, 1j], [1, 0]])
structured = operator_input(
    PeriodicStencil(100, mass=.1, diffusion=.1, potential=.05)
)
print(A.structure, b.preparation.physical_scale.as_float(), b.entry(1))
print(structured.basis.dimension == 2**100)
```

```text
hermitian 5.0 4j
True
```

`operator_input` accepts numerical ndarrays, lists or tuples, canonical SciPy CSR or CSC arrays or matrices, a Qiskit `SparsePauliOp`, a `PeriodicStencil`, or an existing `OperatorInput`. `state_input` accepts numerical vectors, a Qiskit `Statevector` or `QuantumCircuit`, or an existing `StateInput`. SciPy is imported with the input functions. Qiskit is imported when an input is one of its objects. Other array types, object arrays, iterators without a length, ragged rows, bool values and nonfinite data are rejected.

Real and integer data become float64, and complex data become complex128. An existing input object is returned unchanged. Supported arrays are copied, and the copies are immutable, so later changes to your arrays cannot change an input object that already exists.

Each input operation has a byte limit, `max_bytes=10_000_000_000` by default, checked before the conversion or copy. It bounds the operation's peak live arrays, meaning the bytes of the arrays it holds at the same time. It does not bound the memory of the whole Python process or the work done inside SciPy or Qiskit. A 100-qubit product or periodic input uses compact data, so the limit does not cap the number of qubits. A later simulation has its own width, memory, circuit and data limits.

## Hermitian structure of Pauli sums

For a canonical Pauli sum, exact Hermiticity means that every coefficient is real after the terms with identical labels are summed. `read_xacc` and `parse_xacc` keep the complex coefficients specified by the fermionic input. If the intended eigenproblem uses the Hermitian part of their result, construct it explicitly with `SparsePauliOp(op.paulis, coeffs=op.coeffs.real)` and pass that operator to `Eigenproblem`. This operation implements `(op+op.adjoint())/2` coefficient by coefficient and keeps every Pauli row, including a row whose new coefficient is zero. A default-tolerance `simplify()` call can also remove small nonzero real terms, so it is not part of this projection. The operator-norm change is at most the sum of the absolute imaginary coefficients. Choosing this Hermitian part is a modeling decision.

## Coordinates, scale and phase

Coordinate index increases in computational order, and qubit 0 is rightmost. Pauli label `IX` applies X to qubit 0. Product row j belongs to qubit j, so the physical tensor order is row q-1 through row 0. Occupation strings are q0-first, so `"10"` means qubits `(1,0)`, computational index 1, printed ket `|01>`. No electron count, spin or sector is inferred from the syntax.

Physical amplitudes remain available. For `b=[3,4j]`, the numerical norm is 5 and the preparation direction is `[.6,.8j]`. Division uses a positive norm, so negative or complex scaling keeps its physical phase. A vector is normalized once, when its input object is created. Product inputs normalize each two-amplitude factor without forming a full state. `PhysicalScale` stores a binary mantissa and exponent, so it can represent a product norm outside the float64 range. Its `as_float()` or `squared_as_float()` returns `None` when a nonzero value overflows or underflows float64, instead of reporting zero. These are numerical values, not certified error bounds.

Zero data are valid physical inputs but have no normalized quantum state.

`prepare_qiskit(state, max_bytes=..., max_direct_amplitudes=65_536)` explicitly constructs a supported unitary preparation from the normalized direction. It performs no backend run, transpilation or simulation. The amplitude count limits direct vector synthesis only. Compact product and occupation preparations, and the dimensions an input may have, are unaffected. Planning and estimates accept larger direct inputs, and a Run synthesizes them under its own `ExecutionLimits.max_direct_amplitudes`, which has the same default. A supplied `QuantumCircuit` defines the state that its unitary prepares from the all-zero state. `state_input` copies the circuit's data, keeps its global phase and shared gate definitions, rejects unsupported or nonunitary structure and free parameters, and uses only the gate definitions already stored in the circuit, without asking Qiskit to build others. The unit norm of its state rests on your circuit being unitary. `bind_preparation_circuit` additionally takes an explicit `reference` and a matching `basis` when you identify the circuit's source yourself. That reference is your declaration, not a hash of the circuit.

## Explicit classical operations

`OperatorInput.dense_array()`, `entry(...)`, `pauli_terms()` and `sparse_entries(max_bytes=...)` read the data that the input object holds and provide no quantum access. `matvec(vector, max_bytes=..., max_products=1_000_000_000)` computes the matrix-vector product classically. Its count of scalar products is derived from the representation, not from measured CPU time.

Pauli label and mask inputs combine only exactly identical words and drop a term only when its combined coefficient is exactly zero. The combined coefficients are formed by stable summation, so a small nonzero residue of a cancellation is kept. For a `PauliTerms` table `table`, which `pauli_table` and `OperatorInput.pauli_terms()` return, `table.product(other, max_bytes=...)` returns every pairwise product in order without combining identical Pauli words. A later sum, such as the final step of `FactorizedOperatorProduct.expand`, combines them. `table.group(strategy="qwc" or "commuting", max_comparisons=1_000_000, max_bytes=...)` places each term in the first compatible group and, before testing a block of comparisons with the existing groups, checks the block against what remains of `max_comparisons`. General commuting groups are algebraic partitions, not automatically single measurement settings.

`fermion_table` and `ingest_fermion` keep ordered `(mode, action)` strings. Action 0 annihilates and 1 creates, with the rightmost operator acting first. `to_pauli(mapping="jw", max_bytes=...)` maps them to qubits with the Jordan–Wigner parity string on the lower modes. `mapping="z_free"` maps each ladder operator to a qubit raising or lowering operator without that parity string. No normal ordering is applied, and no full state vector is formed.

`FactorizedOperatorProduct((A, B, ...))` keeps the factors, in order, as separate input objects. Its `matvec` acts right to left and checks the total scalar products across all factors. Its explicit `expand` supports homogeneous Pauli, ordered Fermion, or CSR/CSC factors. The scalar multiplications of a sparse expansion are also counted against `max_products`. A compact product can remain usable for `matvec` when expansion exceeds its own limit.

`ingest_df(T, factors, constant_energy=E0, orbital_basis=..., energy_unit=..., source=...)` takes a double-factorized Hamiltonian with real supplied factors

```text
B_l = V_l diag(w_l) V_l^T
H   = E0 I + dΓ(T) + (1/2) sum_l dΓ(B_l)^2
```

T must be exactly symmetric. Signed or zero weights and nonorthogonal V columns are allowed. `to_pauli(max_bytes=..., max_products=1_000_000_000)` performs the contractions and the Jordan–Wigner conversion of the ordered fermionic strings. It checks the size of the complete quartic output before building it. It returns the Pauli `operator`, a `receipt` that records this conversion and its row counts, and `work_fact`, the work of the conversion in NWQLib's work units, a count of scalar operations derived from the sizes by a stated formula rather than a measured time. Its numerical error remains unknown.

Here `dΓ(A) = sum_{p,q,σ} A[p,q] a†(p,σ) a(q,σ)`. For an electronic Hamiltonian with one-body integrals `h` and chemists'-notation two-body integrals `g[p,q,r,s] = sum_l B_l[p,q] B_l[r,s]`, T is the shifted matrix `T[p,s] = h[p,s] - (1/2) sum_q g[p,q,q,s]`. The conversion normal-orders each square as `dΓ(B)²/2 = dΓ(B²)/2 + (1/2) sum B[p,q] B[r,s] a†(p,σ) a†(r,τ) a(s,τ) a(q,σ)` and places spin orbital `(p, σ)` on mode `2p+σ`.

`refine_operator_facts(..., refinements=(...))` computes Pauli L1 or Hermitian sparse Gershgorin bounds on request, using exact fractions of the binary64 values that the input object holds. With empty `refinements`, the call reads none of the operator's entries or terms and uses only metadata and the values held in any `previous` report. Each new pass over the entries or terms is recorded in the report's `receipts`, and values taken from a previous report are never recorded as a new pass of zero cost. These bounds describe the operator as held in binary64, not the total error of a physical output.

## Conventions and derivations

The relations below are standard identities or NWQLib conventions. Each is written out in the docstring or comment of the code that implements it, and the [code tour](CODE_TOUR.md#code-locations-for-inputs-and-search) lists that code. [Conventions](conventions.md) gives the qubit, phase, unit and normalization conventions of the whole library.

| Relation | Justification |
| --- | --- |
| Coordinate index `k = sum_q b_q 2**q`, qubit 0 rightmost | Qiskit little-endian order |
| Stored row `(x, z)` means `i**popcount(x & z) X**x Z**z`, so both bits set is Y | NWQLib encoding |
| `SparsePauliOp` coefficient of the canonical label is `coeff * (-i)**phase` | Qiskit phase convention |
| Dense matrix to Pauli coefficients `c_P = tr(P M)/D` by 2×2 block recursion | Standard identity |
| Pauli product phase and the action of a word on a basis state | Standard Pauli algebra in the stored encoding |
| Commuting test | Even symplectic product |
| Qubit-wise-commuting test | Equal single-qubit Paulis on the shared support |
| `a_j† = (prod_{k<j} Z_k)(X_j - iY_j)/2` | Jordan–Wigner transformation |
| Normal ordering of `dΓ(B)²/2` | Standard anticommutator algebra |
| Spin orbital `(p, σ)` on mode `2p+σ` | NWQLib convention |
| Pauli L1 norm and spectral enclosure | Triangle inequality, unit-norm Pauli words |
| Sparse spectral enclosure and norm bound | Gershgorin circle theorem |

## Metadata and saved data

Input objects have `.basis`, `.reference` and `.to_record()`. `.to_record()` returns a description of the input object with its structure and preparation metadata, without copying or hashing the numerical data. `.from_record(record)` creates an input object that holds only this metadata, so every method that reads data raises an error, even when the original description lists that access. Saved Results and Runs save the numerical data and the Qiskit or SciPy objects themselves ([Save, load and reanalyze results](saved_evidence.md)). A description alone cannot recreate them.

[Input cost controls](development/input_contracts.md) gives the size formulas and the limits of explicit expansions.
