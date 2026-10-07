# Fermionic pools {#fermionic-pools-and-generator-circuits}

Enumerate the fermionic excitation pools that ADAPT-GCiM selects from, apply their anti-Hermitian generators to state vectors without forming matrices, build exact circuits of their exponentials, and read XACC fermionic Hamiltonians into Qiskit Pauli operators. Import from `nwqlib.subroutines.fermionic_pool`, `nwqlib.subroutines.fermionic_circuits` and `nwqlib.subroutines.xacc`. The [GCiM guide](../../algorithms/gcim.md) describes how the ADAPT Method uses the pools.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from nwqlib.subroutines.fermionic_circuits import build_generator_circuit
from nwqlib.subroutines.fermionic_pool import (
    apply_generator_exponential,
    enumerate_spin_adapted_gsd_pool,
)

pool = enumerate_spin_adapted_gsd_pool(2)  # 2 spatial orbitals, 4 qubits
for generator in pool:
    print(generator.family, generator.spatial_indices)
state = np.zeros(16, dtype=complex)
state[0b0011] = 1  # spatial orbital 0 doubly occupied
circuit = build_generator_circuit(pool[0], 0.3)
action = apply_generator_exponential(pool[0], state, 0.3)
print(np.allclose(Statevector(state).evolve(circuit).data, action))
```

```text
single (0, 1)
double_singlet (0, 0, 0, 1)
double_singlet (0, 0, 1, 1)
double_singlet (0, 1, 1, 1)
True
```

The circuit of `exp(0.3 A)` for the first pool member agrees with its matrix-free Taylor action.

## Pools and generator actions

::: nwqlib.subroutines.fermionic_pool
    options:
      show_root_heading: false
      heading_level: 3

### Cost of the matrix-free Pauli action

The Pauli action of `apply_generator` groups terms by their X/Y flip mask, and NumPy population counts give the Jordan–Wigner parity signs. A group sums its coefficient phases once per coordinate and applies the resulting diagonal to every supplied state column, and the output has the same coordinates as the input. For `L` Pauli terms, `G` distinct flip masks and `N=2**n` coordinates:

- The number of state gathers depends on the number of distinct flip masks, while phase formation still visits every Pauli term at every coordinate, so the arithmetic costs `O(LN + GN)` for one state.
- The action keeps one output vector of `N` complex values and tiles of at most 1,024 coordinates. A required complex input conversion is an additional state-sized allocation. It constructs no dense operator or stored `L * 2**n` table.

## XACC Hamiltonian input

`parse_xacc` and `read_xacc` accept two-body XACC input (at most four ladder operators per line) and return a `SparsePauliOp` with its complete identity contribution. The `parse_xacc` entry below maps `2 a_0^dagger a_0 - 3 I` to `-2 I - Z`, and `read_xacc(path, num_modes=1)` reads the same text from a local file.

Input format:

- Each line's coefficient multiplies the literal ordered operator product. Creation is `^`, and annihilation has no suffix. Blank lines, whitespace, complex coefficients, scientific notation and an optional trailing `+` are supported.
- Both entry points strip at most one leading UTF-8 byte-order mark (BOM). Two leading BOMs leave the second in the input and raise `ValueError` on line 1.
- No extra one-half or one-quarter prefactors, symmetry completion or Hermitian projection are applied.

Coefficients and the cutoff:

- The default `coefficient_cutoff=0.0` keeps every nonzero accumulated coefficient. Normal ordering merges equivalent terms before the Jordan–Wigner expansion. Both cross-input normal-order sums and cross-monomial Pauli sums use real and imaginary `math.fsum`, to keep small residues through large cancellations. Coefficients and individual bounded products use float64, and the result is not exact real arithmetic.
- Overflow or a nonfinite accumulated component raises `ValueError`. `fsum` also rejects intermediate overflow even if a later term could cancel it.
- To approximate the mapped operator deliberately, supply an absolute `coefficient_cutoff` to either entry point, for example `parse_xacc(text, num_modes=24, coefficient_cutoff=1e-12)`. It removes final Pauli coefficients with magnitude at most the cutoff, after all contributions have been combined, and does not discard individual input terms or normal-order terms.
- The removed coefficient magnitudes sum to an upper bound on the operator-norm change, because each Pauli has norm one. The parser does not compute or certify a downstream energy-error bound. For an LCU, the kept label count `L` sets the SELECT index width `ceil(log2(L))`. Deliberate pruning can reduce this width and circuit work, and small coefficients are not automatically scientifically irrelevant.

Modes and width:

- File mode `j` maps directly to qubit `j`. Unlike the spin-adapted pool's interleaved convention, XACC has no implied spin ordering, and blocked and interleaved indices remain as supplied. Electron count, spin ordering and reference occupations must be supplied separately by the code that uses the operator.
- The required positive `num_modes` is both the output width and the caller's bound on the input: every written mode, including those of zero-coefficient terms, must be smaller. The width is never inferred from input indices, and invalid indices fail before the Jordan–Wigner expansion. Valid wide models used only for resource counts are not subject to local simulation limits.
- Omitting the required `num_modes` keyword raises Python's ordinary `TypeError`. An invalid supplied bound or malformed XACC syntax raises `ValueError`.
- The mapped identity includes contributions from number operators and quartic terms, so the file's scalar alone is not the qubit identity shift.

Cost:

- Parsing streams lines and keeps coefficient buckets for stable summation: `O(R)` coefficient storage for `R` raw input terms. Packed real and imaginary parts use 16 payload bytes per contribution, plus array capacity and per-key overhead, while keeping `fsum` accuracy.
- These buckets are released before mapping, which costs `O(T n)` time and storage for `T` merged terms on `n` modes, with at most 16 branches per term. Pauli coefficient buckets keep at most `16 T` contributions during mapping and are released before Qiskit allocates the returned Pauli arrays. No dense matrix, statevector or backend is used.

::: nwqlib.subroutines.xacc
    options:
      show_root_heading: false
      heading_level: 3

## Compact generator circuits

`build_generator_circuit(generator, theta, ...)` constructs `exp(theta*A)` for a supported anti-Hermitian generator as a `QuantumCircuit`. An outer control occupies qubit zero and shifts the system modes by one. For an inverse, the builder evaluates the complete parameter map at `-theta`, including the even angle functions, to produce `exp(-theta*A)` directly. This avoids synthesizing the controls again through a generic circuit inverse.

The builder supports commuting Pauli generators and the default spin-adapted GSD index classes:

- Pair/split and four-distinct doubles use the closed-form excitation factors of Magoulas and Evangelista, arXiv:2511.13485v2. Pair/split doubles follow Sec. V, Eqs. (15), (21), (22) and (25)–(27). Four-distinct doubles follow Sec. VI, Tables I and II, with the generator mapping of Supplemental Tables SI and SII. The library angle is divided by `sqrt(2)` to give the paper angle, because the paper operators have coefficient norm `sqrt(2)` in the library's normalization.
- Shared-index doubles use NWQLib's own exact block synthesis instead of those closed forms. They use at most a 64 by 64 real active-mode generator and matrix exponentials of dimension at most five.
- Fermionic routing includes the signs on occupied spectator modes and cancels when the outer control is zero.
- Noncommuting custom records must match a supported default family. The builder does not synthesize them through amplitudes or a generic dense unitary.

These are exact factorization formulas evaluated in floating-point arithmetic. Shared-index two-level synthesis and occupation-conditioned four-distinct rotations can require substantially more gates than small amplitude preparations. Construction is bounded in active support, and no claim of optimal gate count is made. ADAPT composes the generator circuits it chose with its initial preparation and ordered parameter values.

::: nwqlib.subroutines.fermionic_circuits.build_generator_circuit
    options:
      heading_level: 3
