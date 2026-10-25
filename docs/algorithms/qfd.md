# QFD

<a id="quantum-filter-diagonalization"></a>`QFD`, quantum filter diagonalization, estimates the smallest eigenvalue of a Hermitian operator $A$ in a basis of time-evolved copies of one reference state $|\psi\rangle$. The basis states are $|\phi_j\rangle=V^{p_j}|\psi\rangle$ for integer powers $p_j$ of one fixed product-formula step $V$ of time $\tau$, and the method solves the projected generalized eigenproblem $Hf=ESf$, where $E$ is the projected eigenvalue, $f$ its coefficient vector, $H_{jk}=\langle\phi_j|A|\phi_k\rangle$ and $S_{jk}=\langle\phi_j|\phi_k\rangle$. This is the Hamiltonian pencil of Parrish and McMahon, arXiv:1909.08925v1, Sec. II.C, Eqs. (5)–(8), with the states of the fixed step $V$ in place of exact time evolution. `result.eigenvalue` is the lowest Ritz value, the smallest projected eigenvalue $E$ after the overlap directions at or below a cutoff are removed, and it does not establish the full-space ground energy.

Use QFD when you can prepare one reference state and want a basis generated from it by time evolution, without choosing basis states or a generator pool. In [GCiM](gcim.md), `FixedGCIM` projects onto basis states that you supply and `ADAPT` grows them from a pool of generators, [Chebyshev Lanczos](lanczos.md) builds its trial space from Chebyshev polynomials of the operator applied to one state, and [QPE](qpe.md) estimates an eigenphase or energy from the signal of controlled time evolution.

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms import QFD
from nwqlib.operators import ingest_pauli
from nwqlib.problems import ingest_occupation

terms = (("ZZ", .5), ("ZI", 1.0), ("IZ", .7), ("XX", .3), ("XI", .2))
problem = Eigenproblem(A=ingest_pauli(terms, num_qubits=2))
method = QFD(initial_state=ingest_occupation((0, 0), num_qubits=2), time_step=0.5, powers=(0, 1, 2))
result = solve(problem, method=method, seed=7)
print(result)
```

```text
Ritz eigenvalue: -1.2678962
Reference energy: 2.2; lowering: 3.4678962
Basis: powers (0, 1, 2) of one fixed V, order 2 with 1 substeps of time step 0.5
projected estimate in the fixed-V trial space; smallest full-space eigenvalue not established
Method: qfd; execution: quantum
first target: aer_statevector; data: 6/6 chunks used; 6 attempts
accuracy not assessed
```

Here $A=0.5\,ZZ+ZI+0.7\,IZ+0.3\,XX+0.2\,XI$ on two qubits, with qubit 0 as the rightmost character of each label, and the reference $|00\rangle$ has energy 2.2. The powers 0, 1 and 2 of one second-order step of time 0.5 give the Ritz value −1.2678962. The lowest eigenvalue of $A$, computed from its 4×4 matrix, is −1.2769679, so this three-state basis gives a value 0.0091 above it. Without `shots`, the matrix elements come from Aer's statevector simulator by exact readout, which involves no shot sampling but is subject to numerical rounding. Replace `terms`, the reference and the powers to treat your own problem. The [API](../api/algorithms/qfd.md) lists every option.

## Inputs you replace {#inputs-you-replace}

- `Eigenproblem(A=...)` holds the Hermitian operator. Pauli input, as from `ingest_pauli`, is used as it is. With the default `input_conversion="auto"`, an explicit dense matrix of dimension at most 16 is converted to Pauli form, and `input_conversion="dense_pauli"` also converts sparse and larger dense matrices within `max_conversion_work` and `max_bytes`. The dimension must be a power of two, because V acts on the qubit register of $A$. Planning raises `ApplicabilityError` for a Problem with a fixed `subspace`, because QFD generates its own basis, and for one with a symmetry `sector`, because QFD does not establish a symmetry of $A$. A product formula of individual Pauli terms need not stay in a sector that $A$ preserves. With the [fermionic-block step](#evolve-with-fermionic-blocks-from-chemistry-integrals), the Plan instead states whether the basis lies in the particle-number sector of the reference.
- `initial_state` is the reference $|\psi\rangle$, in any form that `StateData` accepts. Every basis state starts from its preparation, with its phase. An occupation or product state keeps planning compact on many qubits.
- `time_step` is $\tau$, finite and positive, in inverse units of the Hamiltonian's energy with $\hbar=1$, for example 1/Hartree for a Hamiltonian in Hartree.
- `powers` are distinct integers that include 0, the reference. Their order is the order of the rows and columns of $H$ and $S$, and 0 need not come first. `result.plan.reconstruction.reference_index` is the position of the power 0.
- `steps_per_time_step`, default 1, and `formula_order`, 1 or the default 2, define the step V as below.
- `evolution`, default `"pauli_terms"`, builds V from the Pauli terms of $A$. `"fermionic_blocks"` builds it from an explicit fermionic generator in `evolution_terms`, as in [Evolve with fermionic blocks](#evolve-with-fermionic-blocks-from-chemistry-integrals).
- `overlap_cutoff`, default 1e-12, keeps the overlap directions whose eigenvalues are above it. `result.analyze(overlap_cutoff=...)` solves the same matrix elements at another cutoff without new measurement.

QFD runs on a quantum register only, and `execution="classical"` raises `ApplicabilityError`.

## The fixed step and its powers {#the-fixed-step-and-its-powers}

Write $A=c_II+A_0$, where $c_I$ is the identity coefficient. With the default `evolution="pauli_terms"`, V is built once, from the nonidentity Pauli terms of $A_0$ in the Plan's label order, which can differ from the order of your terms, as printed below, as $r$ substeps of length $\delta t=\tau/r$, where $r$ is `steps_per_time_step`. Each factor is the rotation $R_P(\theta)=\exp(-i\theta P/2)$ about a Pauli word $P$ with coefficient $c$. Order 1 applies every term once with $\theta=2c\,\delta t$. Order 2 applies every term for $\delta t/2$ and then every term again in reverse order, each with $\theta=c\,\delta t$. A negative power applies the exact adjoint $V^\dagger$, the same rotations in reverse order with negated angles, so every basis state is a power of the same unitary. The identity term contributes only a global phase, is left out of V, and the solve adds $c_I$ to the Ritz values afterwards. The Plan stores the factors and angles of one substep:

```python
rec = result.plan.reconstruction
rows = rec.pencil.nonidentity_terms
print([rows[k][0] for k in rec.step_rows.array.tolist()])
print(rec.step_angles.array.tolist())
```

```text
['IZ', 'XI', 'XX', 'ZI', 'ZZ', 'ZZ', 'ZI', 'XX', 'XI', 'IZ']
[0.35, 0.1, 0.15, 0.5, 0.25, 0.25, 0.5, 0.15, 0.1, 0.35]
```

The example has one second-order substep of length 0.5, so each angle is the term's coefficient times 0.5, and the second half repeats the first in reverse order.

$\tau$, $r$ and the order define V and therefore the trial space. Changing any of them changes the basis, and none of them is an accuracy setting. The library does not evaluate the distance of V from $\exp(-i\tau A_0)$, and a product-formula error bound of V is not an error bar on the energy.

$H$ and $S$ are those of the original $A$ on these basis states, and every pair $j\le k$ is measured. Because V is unitary, $S_{jk}=\langle\psi|V^{p_k-p_j}|\psi\rangle$ depends only on the difference of the two powers. $H_{jk}$ does not in general, because $A$ need not commute with V, so no matrix element is taken from another pair.

## Measure H and S {#measure-h-and-s}

A pair $j<k$ prepares the joint state $(|0\rangle|\phi_j\rangle+|1\rangle|\phi_k\rangle)/\sqrt2$ of one ancilla and the system, as `FixedGCIM` does. With the default `acquisition=PencilAcquisition(pair_preparation="prefix")`, the end with the smaller absolute power $b$, the smaller index on a tie, runs the reference and $V^b$ once without control, and only the remaining power difference is controlled on the ancilla, which is valid because integer powers of one V commute. `pair_preparation="controlled"` controls both complete preparations. A diagonal pair prepares its state without control. A prefix pair whose other end has power $q$ uses V or $V^\dagger$ $|b|+|q-b|$ times, a controlled pair $|p_j|+|p_k|$ times and a diagonal pair $|p_j|$ times. `QFD(acquisition=PencilAcquisition(matrix_measurement="termwise", pair_preparation="controlled"))` raises `ValueError`, because QFD measures joint states only. ADAPT's `pair_preparation="vacuum_reference"` and `"vacuum_restore"` need a known vacuum phase of every generator chain ([How matrix elements are obtained](gcim.md#how-matrix-elements-are-obtained)), and QFD records and computes no vacuum phase for V. Planning therefore raises `ApplicabilityError` for either value whenever the Plan measures anything, that is, for every $A$ that is not a multiple of the identity, `powers=(0,)` included, with exact readout and with shots alike. For a multiple of the identity the request is kept unused, with no resolved `pair_preparation`, and no circuit runs.

Without `shots`, each of the $M(M+1)/2$ pairs of $M$ powers is one exact readout, reduced to the entry of $A_0$ and the overlap entry as in [exact FixedGCIM](gcim.md#how-matrix-elements-are-obtained). This needs a target that runs these pair computations, such as Aer's statevector simulator, and the example's three powers took six.

With `shots=N`, each setting measures the system in the common eigenbasis of one of the $G$ qubit-wise commuting groups of the nonidentity terms. A diagonal pair needs $G$ settings and an off-diagonal pair $2G$, with the ancilla measured in the X basis for the real parts and in the Y basis for the imaginary parts, so $M$ powers need $M^2G$ settings and $NM^2G$ shots:

```python
import nwqlib

sampled = nwqlib.plan(problem, method=method, shots=1000, seed=7)
print(len(sampled.reconstruction.pencil.groups), len(sampled.experiments))
print(nwqlib.estimate(sampled).quantity("shots"))
```

```text
3 27
shots: 27000 count [exact]
  basis=selected_logical; lifecycle=planned; population=terminal MeasurementBatch sampled shots
```

The five terms form the three groups $\{IZ,XI\}$, $\{XX\}$ and $\{ZI,ZZ\}$, so the three powers need 27 settings of 1000 shots each. The counts are pooled, with their sample variances and the covariance of the overlap and Hamiltonian estimates, as for [sampled FixedGCIM](gcim.md#how-matrix-elements-are-obtained). Both the real and the imaginary parts are always measured. A real $A$ does not make the evolved states real, because for $A=X$ and $|\psi\rangle=|0\rangle$ one step gives $\langle0|XV|0\rangle=-i\sin\tau$. With shots, `PencilAcquisition(quadratures="real")` therefore makes planning raise `ApplicabilityError`, and with exact readout the requested value has no effect.

`QFD(sampling=PencilSampling(total_shots=..., pilot_fraction=...))`, with `shots=` omitted, spends a budget of `total_shots` shots on the same settings, measured in a uniform pilot stage and again in a main stage whose shots per setting the pilot pencil decides, as described for FixedGCIM in [Pilot and main shot allocation](gcim.md#pilot-and-main-shot-allocation). The two stage budgets sum to `total_shots`. Only the main-stage measurements, made afresh, form $H$, $S$ and the reference energy, the pilot measurements stay in the Run and count toward its totals, and `result.shot_allocation` records the decision. The budget is an upper limit, not the shots spent. When $A$ is a multiple of the identity, the Run spends no shots, and an interrupted Run need not have spent its whole budget. Because the main-stage shots are known only after the pilot stage, `nwqlib.estimate(plan)` reports the Plan's settings and shots as unavailable, and `prepare(plan, settings="all")` is refused.

## Circuit depth and planning limits {#circuit-depth-and-planning-limits}

`max_native_two_qubit_depth=D` requires every circuit, the circuit of each grouped setting or exact readout, to have a native two-qubit depth of at most $D$, with $D$ itself allowed. This is the two-qubit gate depth of the circuit that the backend compiled for execution, with the reference preparation, the uncontrolled and controlled powers of V, the measurement basis changes and the backend's qubit routing. Each circuit is checked just before its submission, and one whose depth exceeds $D$ or is unavailable is not submitted. The run raises `nwqlib.backends.inspection.NativeDepthCapRefusal`, and the settings submitted earlier keep their counts and shots in the Run ([Checks just before submission](../prepared_execution.md#checks-just-before-submission)). The limit changes no circuit, power or term, and under `sampling` it applies to every pilot and main circuit. Only IBM Runtime computes this depth, under the conditions in [IBM Runtime](../ibm.md#native-two-qubit-depth), so on Aer an integer limit refuses the first circuit. The default `None` requests no limit. The prefix preparation controls only the power difference of a pair, so it never needs more controlled uses of V than `"controlled"`, which also controls the reference preparation. For a pair of powers of opposite sign it can need more uses of V in total, for example 4 instead of 3 for the powers −1 and 2.

Each rotation about a Pauli word of weight $w$ costs $2(w-1)$ CX gates without control and $2w$ under control, counted for the circuit that Qiskit 2.6.0 builds from CX and one-qubit gates at optimization level 0, before cancellation, rotation merging and routing. Another Qiskit version can give other counts.

`max_admission_steps`, default 10,000,000, is compared separately with three counts, each before the work it counts:

- the planning work of checking each circuit description that the method builds ([planning work limit](../development/program_checks.md#planning-work-limit)),
- the number of uses of V and $V^\dagger$ in all circuits, from the counts per pair above, with each sampled pair counted once for each of its settings, checked before the circuits are described, and
- the circuit instructions of the distinct evolution blocks that the Plan uses, V, $V^\dagger$ and their controlled forms, each $r$ times the instructions of one substep, checked before any of these circuits is built.

For $A=X+Z$ with order 2 and the powers (0, 1), one substep rotates about X, Z, Z and X, so V has $4r$ instructions and its controlled form $8r$, $12r$ in all. A count above the limit refuses the Plan, and the message names the count. Raise `max_admission_steps` to build a V above the limit, because fewer `steps_per_time_step` define a different V and therefore a different basis. The chemistry inputs of [Evolve with fermionic blocks](#evolve-with-fermionic-blocks-from-chemistry-integrals) stay far below the default. `max_basis_size`, `max_experiments`, `max_bytes`, `max_analysis_work`, `max_classical_products` and `max_conversion_work` have the defaults of `FixedGCIM` and are listed in [Engineering constants](../ENGINEERING_CONSTANTS.md#qfd-defaults-and-limits).

## Evolve with fermionic blocks from chemistry integrals {#evolve-with-fermionic-blocks-from-chemistry-integrals}

With `evolution="fermionic_blocks"`, V comes from an explicit fermionic generator $B=c_BI+B_0$ in `evolution_terms`, while $H$, $S$ and the energies remain those of $A$. Mode $j$ of $B$ acts on qubit $j$ of $A$, with $|1\rangle$ occupied. Build your own $B$ with `ingest_fermion` from a scalar and rows $a_p^\dagger a_q$ and $a_p^\dagger a_q^\dagger a_r a_s$, written with creations left of annihilations, on as many modes as $A$ has qubits and with coefficients in the energy unit of $A$. The two creations, or the two annihilations, may come in either order. After normal ordering and summing equal terms, the coefficients of each term and of its adjoint must be exact complex conjugates, so a self-adjoint term has a real coefficient. Planning refuses other operators. $B$ is split into complete Hermitian blocks, one for the real and one for the imaginary part of each pair of adjoint ladder terms, or one for a self-adjoint term. The supported terms are number, hopping, density, hopping times a spectator number, and double-excitation terms. Under Jordan–Wigner a block maps to commuting Pauli words, so the rotations of its words give its exponential exactly. Order 1 applies every block once, and order 2 applies every block for half the substep and then every block again in reverse order. Each block conserves the particle number, and V applies the exponential of every block whole, so V conserves it as well. The constant $c_B$ is left out of V and enters no energy.

`chemistry_evolution_terms` builds $B$ from the integrals of the data that `build_gcim_chemistry_problem` returns ([Chemistry inputs and reference energies](gcim.md#chemistry-inputs-and-reference-energies)), which needs the `chemistry` extra, `pip install "nwqlib[chemistry]"`. It reads that data and runs no electronic-structure calculation of its own, and it leaves the data and its Hamiltonian, the target $A$, unchanged. It forms the binary64 coefficient table of the normal-ordered terms built from the integrals, with the factor 1/2 of the two-body terms and equal terms summed, and makes it exactly Hermitian by replacing the two coefficients of each pair of distinct adjoint terms with their midpoint, rounded once. A self-adjoint term keeps its real coefficient. It then removes the terms whose coefficients have magnitude at or below `evolution_coefficient_cutoff`, by default 0.0 Hartree, which removes only exact zeros. It compares each pair of distinct adjoint terms once and removes the pair whole, and it compares each self-adjoint term once. The scalar term, the constant energy of the data, is never removed. These coefficients follow the binary64 table, not exact integral arithmetic. The record that `chemistry_evolution_terms` returns with $B$, `QFDEvolutionDiagnostics`, holds the source of $B$, the cutoff, the numbers of changed and removed pairs, upper bounds on the corrections and $c_B$. Passed optionally as `evolution_diagnostics` together with that $B$, the record must have an `evolution_input` equal to the content hash of $B$, and it is saved with the Method. For H2 in STO-3G at 0.74 Å on four qubits:

```python
from nwqlib.algorithms.gcim import build_gcim_chemistry_problem
from nwqlib.algorithms.qfd import chemistry_evolution_terms

data = build_gcim_chemistry_problem("H 0 0 0; H 0 0 0.74", basis="sto-3g", reference_methods=())
terms, diagnostics = chemistry_evolution_terms(data)
print(terms.basis.dimension, diagnostics.dropped_pairs, diagnostics.evolution_coefficient_cutoff)
reference = ingest_occupation(tuple(data.reference_occupations), num_qubits=data.num_qubits)
chemistry = QFD(initial_state=reference, time_step=0.5, powers=(0, 1, 2), evolution="fermionic_blocks",
                evolution_terms=terms, evolution_diagnostics=diagnostics)
chem = solve(data.eigenproblem(), method=chemistry, seed=7)
print(chem.eigenvalue, chem.reference_energy)
print(*chem.plan.assumptions[1:3], sep="\n")
steps = chem.plan.reconstruction
print(len(steps.step_rows.array), len(steps.step_blocks.array))
```

```text
16 0 0.0
-1.1372838344884886 -1.1167593073964248
V is generated by the complete number-conserving fermionic blocks of evolution_terms, B = c_B I + B0 with c_B = -0.09706626816763206, which enters neither V nor an energy. B differs from A, so it is an approximate evolution generator. Upper bounds from exact rational sums rounded up once: delta_gen_0 = sum_{P != I} |a_P - b_P| = 1.1259128367173146e-15 >= ||A0 - B0|| and delta_gen_full = |c_I - c_B| + delta_gen_0 = 1.1259128367173146e-15 >= ||A - B||. These do not bound the distance of V from exp(-i tau A0) and are not energy error bars.
the basis lies in the particle-number sector N = 2 of the reference, because every evolution block is a complete number-conserving fermionic block. With exact H and S, the Ritz value is a Rayleigh-quotient value of A on a state of that sector, and so an upper bound on the smallest eigenvalue of A and, when A also conserves the particle number, which is not checked, on the smallest eigenvalue of A in that sector. Spin projection and total spin are not qualified, since evolution_terms has no spin labels.
116 36
```

The first line gives the dimension 16 of the four-mode space of $B$, no removed pair and the cutoff 0.0. The reference is the occupation $|0011\rangle$ of the two lowest spin orbitals, so the reference energy equals the RHF energy of the data up to rounding. The first of the two lines from `chem.plan.assumptions` states $c_B$ and the upper bounds $\delta_{\rm gen,0}=\sum_{P\neq I}|a_P-b_P|\ge\lVert A_0-B_0\rVert$ and $\delta_{\rm gen,full}=|c_I-c_B|+\delta_{\rm gen,0}\ge\lVert A-B\rVert$, computed from the Pauli coefficients $a_P$ of $A$ and the exact sums $b_P$ of the blocks of $B$. They bound every coefficient difference between $A$ and $B$, whatever its source, and they bound neither the distance of V from $\exp(-i\tau A_0)$ nor the energy error. The second line states that the basis lies in the particle-number sector N = 2 of the reference. The Plan states a sector only for an occupation reference or a product of computational basis states. With exact $H$ and $S$, the Ritz value is then an upper bound on the smallest eigenvalue of $A$ and, when $A$ conserves the particle number, which QFD does not check, on the smallest eigenvalue of $A$ in that sector. The Plan makes no statement about spin projection or total spin, because `evolution_terms` has no spin labels. For a $B$ from `chemistry_evolution_terms`, `diagnostics.spin_statement` reports whether its terms conserve the alpha and beta electron numbers, and hence $S_z$, together with the particle number and $S_z$ of the reference occupations of the data. It says nothing about another `initial_state`, and conservation of the particle number and $S_z$ establishes neither conservation of the total spin nor molecular accuracy. The Ritz value is a projected value of the qubit Hamiltonian $A$ in this basis, and it says nothing about the accuracy of $A$ as a model of the molecule.

`chem.plan.reconstruction.evolution` is the Pauli table of $B$, whose identity row holds $c_B$, and `step_rows` with `step_angles` give the factors of one substep in order. `step_blocks` holds the position in `step_rows` where each complete block starts, so the last line shows the 116 factors and 36 block starts of one second-order substep, the 18 blocks of $B$ applied twice. `chem.plan.method.evolution_diagnostics` returns the `QFDEvolutionDiagnostics` record.

The third count of `max_admission_steps` is the number of circuit instructions of the distinct evolution blocks that the Plan uses, each block counted once. With the default cutoff of `chemistry_evolution_terms`, the occupation reference of the data, the powers (0, 1, 2), time step 0.5, r = 1 and order 2, planning gives:

| Input | H2, STO-3G, 0.74 Å | Linear H4 chain of the [eigenvalue notebook](../examples.md), atoms at z = 0, 2, 4 and 6 Å (adjacent spacing 2.0 Å), STO-3G, active space of 4 electrons in 4 spatial orbitals |
| --- | --- | --- |
| Qubits | 4 | 8 |
| Blocks | 18 | 198 |
| Factors and block starts per substep | 116 and 36 | 2056 and 396 |
| Instructions of V and controlled V | 2144 | 72448 |

These integers are exact counts for these inputs and parameters, and they are the same with exact readout and with 8192 shots per setting. The larger, 72448, is about 0.72% (rounded) of the default 10,000,000. Each sums the instructions of the two evolution blocks of its Plan, V and controlled V, each block counted once, and is not a count of executed circuits, shots, native gates, CPU time or memory. The counts come from planning alone, without running a quantum circuit, although building the chemistry input runs its electronic-structure calculation. They were measured with Python 3.12.14, PySCF 2.14.0, OpenFermion 1.8.1, Qiskit 2.6.0 and Aer 0.18.0 on macOS arm64.

## Read the result {#read-the-result}

`result.eigenvalue` is the lowest kept Ritz value in the unit of the `Eigenproblem`. `result.reference_energy` is $H_{i_0i_0}=\langle\psi|A|\psi\rangle$ from the same matrices, at the position $i_0$ of the power 0, and `result.lowering` is the reference energy minus the Ritz value, negative values included. For $H$ and $S$ computed without sampling or rounding error, the Ritz value is at least the smallest eigenvalue of $A$, and at most the reference energy when no overlap direction is cut. Sampled matrices carry neither guarantee. A lowering is not a recovered fraction of a correlation energy. The basis can also miss the ground state entirely. For $A=\operatorname{diag}(-2,1)$ and $|\psi\rangle=|1\rangle$, every power of V keeps $|1\rangle$, so the Ritz value is 1 while the ground energy is −2.

`result.pencil` holds $H$, $S$ and the solve diagnostics, which have the meaning given in [Read the result](gcim.md#inspect-numerical-diagnostics-and-error-bounds) of the GCiM guide, among them the complete overlap spectrum `overlap_eigenvalues`, the kept rank `kept_rank` and, for sampled matrices, the enclosure diagnostic (`sampled_enclosure`, `sampled_enclosure_violation`), which reports how far the Ritz value lies outside an interval that contains every eigenvalue of $A$, computed from the absolute values of its Pauli coefficients. `result.pairs` holds the exact pair entries with their error bounds, and `result.estimates`, `result.group_moments` and `result.sampled_pair_variances` the sampled means, moments and variances. When planned measurements returned no data, `result.missing` names them, and no pencil is solved. For an operator that is a multiple $cI$ of the identity, no circuit runs, and the eigenvalue and the reference energy are both $c$.

## Save, reanalyze and resume {#save-reanalyze-and-resume}

`result.save(...)` writes the Result with its Plan, the stored factors and angles of V and the measured data, and `nwqlib.load_result` reopens it. Loading rebuilds V and its controlled forms from the stored factors without planning again. `analyze(overlap_cutoff=...)` solves the same $H$ and $S$ at another cutoff without new measurement:

```python
loaded = nwqlib.load_result(result.save("qfd-result"))
print(loaded.eigenvalue == result.eigenvalue, loaded.pencil.overlap_eigenvalues)
again = loaded.analyze(overlap_cutoff=1e-3)
print(again.pencil.kept_rank, again.eigenvalue)
```

```text
True (0.00019849849384574243, 0.028292485736013718, 2.9715090157701405)
2 -1.0929433632414558
```

The smallest overlap eigenvalue, 2.0e-4, shows that the three basis states are nearly linearly dependent. The cutoff 1e-3 removes that direction, and the solve on the two remaining directions gives −1.0929, so the removed direction carried part of the lowering.

A sampled Run saved to a folder and interrupted before all settings are measured continues from that folder. `nwqlib.load_run(directory, backend=...)` reopens it. Its `wait()` keeps the counts already measured, submits only the settings without data and returns the Result. A Run reopened after it has finished submits nothing ([Continue an interrupted run](../run_archives.md)). Under `sampling`, an interrupted shot allocation or final analysis is not retried on its own, and the Run raises an error until `run.resume(reanalyze=True)` is called. After an interrupted allocation, that call makes the allocation decision from the complete pilot data and then continues with any main-stage measurements still missing, which are new measurements. After an interrupted final analysis, with the main stage complete, it repeats the analysis and measures nothing. Completed measurements are not repeated.

## Limits

- The eigenvalue is a projected value in the trial space of the chosen V and powers and does not establish the full-space ground energy.
- V approximates $\exp(-i\tau A_0)$ only through the chosen product formula, and its distance from that evolution is not evaluated.
- No error bound on the energy is computed. The sampled enclosure diagnostic is not a sampling or accuracy bound.
- QFD runs on a quantum register only and refuses a fixed subspace or a symmetry sector.
- With `evolution="fermionic_blocks"`, the Plan states the particle-number sector of the reference, and so of the basis, only for an occupation reference or a product of computational basis states. It states no sector of $A$ and makes no statement about spin projection or total spin. The Ritz value says nothing about the accuracy of $A$ as a model of a molecule.
- With shots, both the real and the imaginary parts are always measured.
- Only IBM Runtime computes the native two-qubit depth that `max_native_two_qubit_depth` limits.

## Source and code map

The Hamiltonian pencil is from Parrish and McMahon, arXiv:1909.08925v1, which builds a nonorthogonal trial space from time-propagated references. QFD uses one reference. Rows marked NWQLib have no paper source of their own, and the Location column names where the derivation or rule is stated. A bare file name in the Code column is in `src/nwqlib/algorithms/qfd/`.

| Step | Source | Location | Code |
| --- | --- | --- | --- |
| Hamiltonian pencil `H f = E S f` in a basis of time-evolved reference states | Parrish and McMahon, arXiv:1909.08925v1 | Sec. II.C, Eqs. (5)–(8) | `method.py`: `QFD.plan`, `QFD.analyze` |
| Fixed product-formula step V of order 1 or 2, its exact adjoint and the stored angles | NWQLib | Factor order and angles in the docstrings of `step_factors` and `evolution_block`, with the order-2 schedule of `rotation_schedule` | `construction.py`: `step_factors`, `evolution_block`, `construct_evolution`<br>`algorithms/qpe/powers.py`: `rotation_schedule` |
| Prefix and controlled pair preparations and the uses of V in each pair circuit | NWQLib | Choice of the end with the smaller absolute power, in the docstrings of `power_pair_recipes` and `macro_uses` | `construction.py`: `power_pair_recipes`, `macro_uses` |
| Joint-state measurement, grouped counts and their pooling, exact pair reduction and the projected solve, shared with FixedGCIM | NWQLib, and the sources in the [GCiM source map](gcim.md#source-and-code-map) | Docstrings of the functions in the last column | `algorithms/gcim/fixed_basis.py`: `sampled_settings`, `read_groups`, `pooled_statistics`, `exact_pair_pencil`, `solve_pencil` |
| Identity offset removed before the projected solve | NWQLib | [Proposition 14](../mathematics.md#r14), the exact effect of an identity offset on the pencil | `algorithms/gcim/fixed_basis.py`: `solve_pencil` |
| Fermionic generator B split into complete number-conserving blocks of commuting Pauli words, the generator distance bounds and the sector statement | Leal-Sanchez et al., arXiv:2607.01560v1, as background for grouping a fragment by an Abelian symmetry<br>NWQLib for the signs, weights and checks of the blocks, for the generator distance bounds, from the triangle inequality over Pauli coefficients, and for the sector statement, from the complete number-conserving blocks and the reference | Fig. 1b<br>Docstrings of `fermionic_blocks`, `fermionic_step` and `generator_difference`, and of the Plan statements in `method.py` | `construction.py`: `fermionic_blocks`, `fermionic_step`, `generator_difference`<br>`method.py`: `QFD.plan` |
| Hermitian generator from chemistry integrals and its diagnostics | NWQLib | Docstrings of `chemistry_evolution_terms` and `QFDEvolutionDiagnostics` | `chemistry.py`: `chemistry_evolution_terms`<br>`records.py`: `QFDEvolutionDiagnostics` |
| Saving and reloading without planning again | NWQLib | Docstrings of `save` and `load` | `archive.py`: `save`, `load` |
