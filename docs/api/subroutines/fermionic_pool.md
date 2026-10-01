# Fermionic pools and generator circuits

Pool records keep their anti-Hermitian coefficients, spin-orbital ordering and generator identities. The [GCiM guide](../../algorithms/gcim.md) describes their use by the configured ADAPT Method.

::: nwqlib.subroutines.fermionic_pool

## XACC Hamiltonian input

`nwqlib.subroutines.xacc.parse_xacc` and `nwqlib.subroutines.xacc.read_xacc` accept two-body XACC input (at most four ladder operators per line) and return a `SparsePauliOp` with its complete identity contribution:

```python
from nwqlib.subroutines.xacc import parse_xacc, read_xacc

# 2 a_0^dagger a_0 - 3 I maps to -2 I - Z.
hamiltonian = parse_xacc("(2,0)0^ 0 +\n(-3,0)", num_modes=1)
# A caller-owned local file can instead be read with read_xacc(path, num_modes=1).
```

Each line's coefficient multiplies the literal ordered operator product; creation is `^` and annihilation has no suffix. Blank lines, whitespace, complex coefficients, scientific notation, and optional trailing `+` are supported. Both entry points strip at most one leading UTF-8 byte-order mark (BOM); two leading BOMs leave the second in the input and raise `ValueError` on line 1. No extra one-half or one-quarter prefactors, symmetry completion, or Hermitian projection are applied. The default `coefficient_cutoff=0.0` keeps every nonzero accumulated coefficient. Normal ordering coalesces equivalent terms before Jordan–Wigner expansion. Both cross-input normal-order sums and cross-monomial Pauli sums use real/imaginary `math.fsum` to keep small residues through large cancellations. Coefficients and individual bounded products use float64; this does not claim exact real arithmetic. Overflow or a nonfinite accumulated component raises `ValueError`; `fsum` also rejects intermediate overflow even if a later term could cancel it.

To deliberately approximate the mapped operator, supply an absolute `coefficient_cutoff` to either entry point. This removes final Pauli coefficients with magnitude at most the cutoff, after all contributions have been combined. It does not discard individual input terms or normal-order terms. For example, `parse_xacc(text, num_modes=24, coefficient_cutoff=1e-12)` requests this approximation explicitly. The removed coefficient magnitudes sum to an upper bound on the operator-norm change, because each Pauli has norm one; the parser does not compute or certify a downstream energy-error bound. For an LCU consumer, the kept label count `L` sets the SELECT index width `ceil(log2(L))`. Deliberate pruning can reduce this width and circuit work; small coefficients are not automatically scientifically irrelevant.

File mode `j` maps directly to qubit `j`. Unlike the spin-adapted pool's interleaved convention, XACC has no implied spin ordering: blocked and interleaved indices remain as supplied. Electron count, spin ordering, and reference occupations must be supplied separately by the consumer. The required positive `num_modes` is both the output width and the caller's admission bound: every written mode, including zero-coefficient terms, must be smaller. Width is never inferred from input indices. Invalid indices fail before JW expansion. Valid wide resource-only models are not subject to local simulation limits. Omitting the required `num_modes` keyword raises Python's ordinary `TypeError`; an invalid supplied bound or malformed XACC syntax raises `ValueError`. The mapped identity includes contributions from number operators and quartic terms, so the file's scalar alone is not the qubit identity shift.

Parsing streams lines and keeps coefficient buckets for stable summation: `O(R)` coefficient storage for `R` raw input terms. Packed real/imaginary parts use 16 payload bytes per contribution, plus array capacity and per-key overhead, while keeping `fsum` accuracy. These buckets are released before mapping, which costs `O(T n)` time and storage for `T` coalesced terms on `n` modes, with at most 16 branches per term. Pauli coefficient buckets keep at most `16 T` contributions during mapping and are released before Qiskit allocates the returned Pauli arrays. No dense matrix, statevector, or backend is used.

The shared matrix-free Pauli action uses NumPy population counts for the Jordan–Wigner parity signs. The Pauli action groups terms by their X/Y flip mask. A group sums its coefficient phases once per coordinate and applies the resulting diagonal to every supplied state column. The output has the same coordinates as the input. The number of state gathers depends on the number of distinct flip masks, while phase formation still visits every Pauli term at every coordinate. For `L` Pauli terms, `G` distinct flip masks and `N=2**n` coordinates the arithmetic costs `O(LN + GN)` for one state. The action keeps one output vector of `N` complex values and tiles of at most 1,024 coordinates. A required complex input conversion is an additional state-sized allocation. It constructs no dense operator or kept `L * 2**n` table.

::: nwqlib.subroutines.xacc

## Compact generator circuits

`build_generator_circuit(generator, theta, ...)` constructs `exp(theta*A)` for an admitted anti-Hermitian generator as a `QuantumCircuit`. An outer control occupies qubit zero and shifts the system modes by one. For inverse construction, the compiler evaluates the complete parameter map at `-theta`, including the even angle functions, to produce `exp(-theta*A)` directly. This avoids re-synthesizing controls through a generic circuit inverse.

::: nwqlib.subroutines.fermionic_circuits.build_generator_circuit

The compiler supports commuting Pauli generators and the default spin-adapted GSD index classes. Pair/split and four-distinct doubles use the closed-form excitation factors of Magoulas and Evangelista, arXiv:2511.13485v2. Pair/split doubles follow Sec. V, Eqs. (15), (21), (22) and (25)–(27). Four-distinct doubles follow Sec. VI, Tables I and II, with the generator mapping of Supplemental Tables SI and SII. The library angle is divided by `sqrt(2)` to give the paper angle, because the paper operators have coefficient norm `sqrt(2)` in the library's normalization. Shared-index doubles use NWQLib's own exact block synthesis instead of those closed forms. They use at most a 64 by 64 real active-mode generator and matrix exponentials of dimension at most five. Fermionic routing includes the signs on occupied spectator modes and cancels when the outer control is zero. Noncommuting custom records must match a supported default family. The compiler does not synthesize them through amplitudes or a generic dense unitary.

These are exact factorization formulas evaluated in floating-point arithmetic. Shared-index two-level synthesis and occupation-conditioned four-distinct rotations can require substantially more gates than small amplitude preparations. Construction is bounded in active support, and no claim of optimal gate count is made. ADAPT composes the selected generator circuits with its actual initial preparation and ordered parameter values.
