# GCiM options for correlation recovery

GCiM approximates the ground-state energy by diagonalizing a Hamiltonian in a space of prepared states. Improving its recovery of correlation requires both useful trial states and sufficiently accurate matrix elements. This page separates selectable measurement constructions from choices that change the trial space or Hamiltonian. It targets shallow-circuit calculations, including chemistry problems with 20–26 system qubits, without assuming that this size has already been qualified on hardware.

The [GCiM guide](../algorithms/gcim.md) describes the available API, and its [decision table](../algorithms/gcim.md#choose-the-measurement-and-the-basis) says when to choose each current option. Options marked **proposed** below specify intended choices, not callable parameters. Their availability should be updated with their implementation and evidence. Sampled FixedGCIM, ADAPT and QFD measure grouped joint states by default, and ADAPT keeps termwise Hadamard tests as an explicit comparison. Uniform shots per measurement setting are the default, and `PencilSampling` allocates them from a pilot stage instead.

## Scientific objective and comparison basis

For normalized trial states $|\phi_i\rangle$ and the physical Hamiltonian $A$, GCiM forms

$$H_{ij}=\langle\phi_i|A|\phi_j\rangle,\qquad S_{ij}=\langle\phi_i|\phi_j\rangle,\qquad Hc=ESc.$$

The objective is to recover correlation missing from a shallow reference while achieving the accuracy required for the selected Hamiltonian and symmetry sector. A smaller measurement count is useful when it makes that scientific result more accessible. An execution limit on basis size or shots does not establish convergence to the ground state.

For a comparison of measurement constructions, fix $A$, its energy offset and units, the ordered trial states including their relative phases, and the overlap cutoff. These choices define the projected problem. With exact data, constructions that preserve them should give the same H/S and Ritz energies. Their finite-shot covariance and hardware bias can differ.

For a comparison of basis-selection methods, fix the physical Hamiltonian and target sector, then report each basis, reference preparation, and selection cost. A different trial space can give a different exact Ritz energy. This is a change in the approximation, even when the measurement code is shared.

## Choices that preserve the ideal projected problem

### Pair measurement construction

`PencilAcquisition` selects the construction, and planning checks its input conditions before any circuit is generated ([How matrix elements are obtained](../algorithms/gcim.md#how-matrix-elements-are-obtained)).

| Construction | Availability | Required preparation information |
| --- | --- | --- |
| Termwise Hadamard tests | Current in sampled ADAPT as an explicit comparison, `matrix_measurement="termwise"` | Controlled forward and inverse preparation, with consistent phases |
| Grouped joint-state measurement | Current default of sampled FixedGCIM, ADAPT and QFD | Two branch preparations and a grouped Pauli Hamiltonian |
| Vacuum reference with Pauli insertion | Proposed | A reference preparation and known vacuum action of each excitation block |
| Vacuum joint state for the reference row | Current in ADAPT as `pair_preparation="vacuum_reference"` | A reference preparation and a pool whose generators pass ADAPT's vacuum rule |
| Vacuum joint state with restoration for a general pair | Current in ADAPT as `pair_preparation="vacuum_restore"` | A reference preparation and a pool whose generators pass ADAPT's vacuum rule |

Termwise Hadamard tests obtain the real and imaginary components of individual transition elements. Grouped measurements prepare

$$|\chi_{ij}\rangle=\frac{|0\rangle|\phi_i\rangle+|1\rangle|\phi_j\rangle}{\sqrt2}$$

and measure $X_a\otimes P$ or $Y_a\otimes P$ for a Hermitian Pauli string $P$. Their expectations are the real and imaginary parts of $\langle\phi_i|P|\phi_j\rangle$. Qubit-wise commuting (QWC) strings can share the terminal measurement basis. The same shots can estimate an overlap and Hamiltonian fragment, so their covariance belongs in the energy analysis. This joint-state construction and its compatibility with Hamiltonian grouping are described by Huggins et al., arXiv:1909.09114v2, Sec. II.A, Eqs. (4)–(10), [NOVQE](https://arxiv.org/html/1909.09114v2).

For the proposed vacuum-reference construction, write $|\phi_i\rangle=U_iB|\mathrm{vac}\rangle$, where $B$ prepares an occupation reference and $U_i|\mathrm{vac}\rangle=|\mathrm{vac}\rangle$. Starting from $|+\rangle|\mathrm{vac}\rangle$, apply controlled-$B$, uncontrolled $U_j$, controlled-$P$, uncontrolled $U_i^\dagger$, and controlled-$B^\dagger$. Ancilla X/Y expectations give $\langle\phi_i|P|\phi_j\rangle$. Known vacuum phases can be corrected explicitly. The complete $U_i$ block must have known vacuum action, but each elementary factor need not preserve vacuum. Controlled-$P$ can include X/Y factors.

This adaptation follows the controlled-reference mechanism in Yoshioka et al., arXiv:2407.14431v3, Fig. 1b and main text pp. 2–3, [Krylov diagonalization](https://arxiv.org/html/2407.14431v3). It removes controls from the excitation blocks while keeping the forward-plus-inverse sequence. Each inserted $P$ generally needs its own circuit, so this construction does not directly reuse the grouped terminal readout above. Its suitability depends on compiled depth and total shot cost. An arbitrary preparation circuit alone does not establish the required vacuum action.

There are also grouped constructions under the same vacuum conditions. Write $C_1(B)$ and $C_0(B)$ for reference initialization controlled on ancilla states one and zero. For the pair containing $|\phi_0\rangle=B|\mathrm{vac}\rangle$, the sequence $C_1(B)$, $U_j$, $C_0(B)$ prepares $|\chi_{0j}\rangle$ directly. It uses one uncontrolled excitation block and allows the existing terminal grouping. These identities concern the complete preparation blocks, including their phases.

For a general pair, the sequence $C_1(B)$, $U_j$, $U_i^\dagger$, $C_0(B)$, $U_i$ prepares $|\chi_{ij}\rangle$. This algebraic construction removes controlled excitation blocks and allows grouping, at the cost of two executions of the $i$ block, including its inverse. Exchanging the two states can reduce that repeated preparation cost. Before cancellations and routing, the sequence contains preparation depth $d_j+2d_i$ plus the reference controls, where $d_i$ and $d_j$ are the depths of $U_i$ and $U_j$.

ADAPT implements both grouped constructions as `pair_preparation="vacuum_reference"` and `"vacuum_restore"`, with ancilla phase gates that correct known vacuum phases. Planning accepts them only when a sufficient structural rule shows, from the Pauli terms of every pool generator, that its exponential keeps $|\mathrm{vac}\rangle$ up to a phase. The general-pair construction takes the two chains of a pair in their lexicographic order, not in the order of lower cost ([How matrix elements are obtained](../algorithms/gcim.md#how-matrix-elements-are-obtained)). The guide's [comparison](../algorithms/gcim.md#choose-the-measurement-and-the-basis) gives compiled counts for one two-qubit example, and no general compiled saving has been measured.

### Common-prefix factoring

**Current: enabled by the default `pair_preparation="prefix"` and disabled by `"controlled"`.** Let $W_i|0\rangle=|\phi_i\rangle$ and $W_j|0\rangle=|\phi_j\rangle$ denote the complete preparations, including the reference initialization. If $W_i=V_iC$ and $W_j=V_jC$, with the same exact block $C$ executed first, the joint state can be built by applying $C$ once without control and then applying the branch-dependent suffixes. This preserves grouping and the ideal projected matrices. The common block still executes once.

Factoring requires a matching circuit structure and parameters, or an equivalent representation with a known phase relation. Matching ansatz names with different angles is insufficient. ADAPT's nested product states provide some eligible pairs. Pairs of independent single-generator states usually share only the reference preparation. An opaque circuit may lack enough information to identify the common prefix efficiently. The implementation compares exact blocks, the same generators at the same binary64 angles for ADAPT and the same supplied preparation for FixedGCIM, and controls both complete preparations of any other pair. QFD shares the reference and the power of its step with the smaller absolute value instead.

This factoring applies to the joint-state preparation. The sequential vacuum-reference circuit has a different arrangement of forward and inverse blocks, so the same rewrite does not automatically apply there. Likewise, orbital-basis terminal readout belongs to the joint-state construction and is not supplied by selecting termwise vacuum insertion.

Native ADAPT generator synthesis already leaves Pauli basis rotations and parity ladders uncontrolled while controlling the central rotation. Savings from factoring or vacuum references must therefore be compared with that implementation, rather than with a construction that turns every CX into a Toffoli.

### Real or complex quadratures

**Both are current.** The default `quadratures="auto"` of FixedGCIM and ADAPT measures real parts only when a structural proof from the stored inputs holds, `"complex"` measures both parts and `"real"` requires the proof ([Sampled, real-only](../algorithms/gcim.md#how-matrix-elements-are-obtained)). QFD measures both parts. A sufficient condition is $A^*=A$ and $|\phi_i\rangle^*=|\phi_i\rangle$ for every trial state in the same encoded basis and phase convention. Then the ideal H/S entries are real, and imaginary quadratures are unnecessary for their reconstruction.

Real reference amplitudes and exponentials of real skew-symmetric generators satisfy the trial-state part of this condition. The Hamiltonian must independently satisfy $A^*=A$. Real parameters alone do not guarantee real states, as $e^{-i\theta X}|0\rangle$ is generally complex. If states carry different known phases, those phases must be handled consistently before applying the real-only choice. Reality should follow from the input construction, without requiring full-state simulation to establish it.

Hardware errors can violate the ideal reality condition. Omitting imaginary quadratures uses a constraint on the ideal target and does not demonstrate that the noisy preparation is real or that its real-part bias is small.

### Shot allocation

**Both are current.** Uniform allocation is the default, and `sampling=PencilSampling(...)` of FixedGCIM, ADAPT and QFD chooses the main-stage shots from a pilot stage ([Pilot and main shot allocation](../algorithms/gcim.md#pilot-and-main-shot-allocation)). Both estimate the same projected matrices. For a simple Ritz value in a fixed subspace, with $c^\dagger Sc=1$, the first-order relation is

$$\delta E=c^\dagger(\delta H-E\delta S)c.$$

An allocation can use a pilot estimate of these sensitivities and the measured covariance to direct shots toward the selected energy. The pilot and final estimates count toward the same total budget. The fixed subspace here includes the projection selected by the overlap cutoff. Holding the numeric cutoff fixed does not hold that projection fixed when a perturbed S rotates its eigenvectors or changes the number of directions used. Such changes, or a change in the trial basis, require additional analysis beyond this local sensitivity model. Unbiased matrix-element estimates also need not give an unbiased minimum eigenvalue.

The sensitivity weights of reference-row entries include $c_0^*c_j$. That row need not dominate the energy uncertainty, even when its circuits are particularly inexpensive.

Adaptive measurement strategies for quantum subspace methods have direct precedent in Nakamura et al., arXiv:2311.07893v2, Sec. II.2, [Adaptive measurement strategy for quantum subspace methods](https://arxiv.org/html/2311.07893v2). Their reported improvements do not establish a saving for this GCiM implementation or these hardware sizes. Sampling changes statistical error, while trial-space error and hardware bias require separate treatment.

### Classical entries from bounded determinant support

**Proposed choices are quantum acquisition for all entries or classical assembly of eligible entries.** If both states of a pair have explicitly known complex determinant coefficients with bounded support, their overlap and two-body Hamiltonian element can be assembled with Slater–Condon matrix elements. Some occupation-reference and single-excitation pairs have this structure. One eligible state alone does not make its entire H/S row eligible when the other states lack the required coefficient access.

Eligibility requires the complex coefficients, fermionic integrals or equivalent access to determinant matrix elements, and the mapping and occupation conventions. Sample frequencies alone do not supply relative phases. Long excitation products can make the support expensive to enumerate, so the support and assembly work need explicit limits before expansion.

The intended quantities remain the ideal trial-state H/S entries. Combining exact classical entries with noisy quantum estimates still leaves bias in the latter. It does not establish the Gram matrix of one consistently prepared noisy-state family.

## Chemistry measurement choices

**Pauli QWC measurement is current in sampled FixedGCIM, ADAPT and QFD. Double-factorized orbital-basis measurement is proposed.** A factorization of the form $A=\sum_\ell R_\ell D_\ell R_\ell^\dagger$, with $D_\ell$ a diagonal number-operator polynomial, allows simultaneous measurement of many terms after an orbital rotation. On the joint state, apply $R_\ell^\dagger$ to the system register and combine its number measurements with the ancilla outcome.

This requires the fermionic integrals or supplied factors, their orbital and spin conventions, and a compatible fermion-to-qubit mapping. A generic Pauli table need not provide those data. Exact factorization changes the measurement representation. Truncating the factorization used to measure $A$ changes the target Hamiltonian and requires a stated approximation error. Approximation used only in state preparation instead changes the trial states.

Orbital rotations add two-qubit gates and routing cost. Fewer groups do not by themselves imply fewer shots, because group covariance affects the energy variance. See Huggins et al., arXiv:1907.13117v4, Table I and Sec. II.A, [basis-rotation grouping](https://arxiv.org/html/1907.13117v4). Cohn, Motta and Parrish apply factorized readout to off-diagonal QFD matrix elements in [PRX Quantum 2, 040352](https://doi.org/10.1103/PRXQuantum.2.040352), Secs. II.B–D. Ordinary factorization and additional nonlinear compression have different classical costs.

## Choices that change the scientific approximation

| Choice | Effect on the result | Input consequence |
| --- | --- | --- |
| Trial states or generator pool | Changes the variational space and possible correlation recovery | Requires an explicit basis or generators with their reference and symmetry conventions |
| Per-state preparation depth | Restricts the admissible trial states and can limit correlation recovery | Requires a depth limit and a defined gate set, connectivity and compilation convention |
| ADAPT selection rule | Changes which states enter the basis | Requires the data used by that rule and its measurement budget |
| Generator-angle optimization | Changes trial states and adds repeated energy or gradient evaluations | Requires an optimization objective and iteration and evaluation limits. Sampled comparisons also need uncertainty and noise assumptions |
| Overlap cutoff | Changes the subspace used in the solve | Uses the same acquired H/S, but may discard useful or noise-dominated directions |
| Hamiltonian truncation | Changes the physical operator being approximated | Requires a truncation criterion and an error statement for the target quantity |

Supplying a basis or pool and setting `overlap_cutoff` are current capabilities. Selection using the current GCiM Ritz state's residual couplings and predicted energy improvement is implemented as ADAPT's `basis_growth=BoundedRitzGrowth(...)`, which scores each evaluated extension by a lower bound on the decrease that an exact solve of the current kept subspace together with the extension's part outside it would give ([Bounded Ritz growth](../algorithms/gcim.md#bounded-ritz-growth)). ADAPT's default growth instead ranks generators by a product-state commutator gradient. Bounded Ritz growth is therefore a GCiM variant with its own convergence questions, not an equivalent measurement optimization.

Current ADAPT adds generators at fixed `theta` by default, with `optimize_every_m=None`. The optional `optimize_every_m`, `optimize_rounds_n` and `optimize_max_evaluations` settings enable intermittent BFGS angle optimization. Gradient-based generator selection remains an adaptive decision even when angle optimization is disabled, and noisy gradients can change the selected basis. A comparison using optimized states must include the work and uncertainty of that optimization.

A per-state preparation-depth limit is current as `BoundedRitzGrowth(basis_depth_limit=D)`, an upper bound on the CX count of each basis state's preparation before routing, and `max_native_two_qubit_depth` separately limits the native two-qubit depth of every submitted circuit. The first constrains the available approximation space, while the complete pair-measurement circuit can be deeper because it includes controls, repeated preparations or orbital rotations. Both depths matter when selecting an acquisition construction for hardware.

Symmetry conditions apply to both the reference and its preparation. Fixed particle number and $S_z$ do not establish a fixed $S^2$ sector. A cheaper excitation pool can change spin behavior or omit directions needed for correlation recovery. Exact spin-adapted synthesis and compact operator subsets are discussed by Magoulas and Evangelista, arXiv:2511.13485v2, Secs. VI–VIII, [Spin-Adapted Fermionic Unitaries](https://arxiv.org/html/2511.13485v2). Their universality statements do not guarantee convergence at a bounded circuit depth or basis dimension.

The [existing cutoff analysis](../algorithms/gcim.md#inspect-numerical-diagnostics-and-error-bounds) reports overlap eigenvalues, the number of directions used, and conditioning. These quantities explain sensitivity to the projected solve, but do not certify the full-space ground-state energy.

## Comparison results and evidence limits

The following comparison is analytical and describes current constructions or proposed mechanisms. No hardware accuracy or wall-time comparison between the options has been established. Let $L$ be the number of nonidentity Pauli terms and $G>0$ the number of QWC groups.

| Construction for one off-diagonal pair | Measurement settings | Main circuit cost |
| --- | --- | --- |
| Termwise ADAPT, an explicit comparison | $2(L+1)$ | Controlled preparations and term insertion |
| Grouped joint states, the default of FixedGCIM, ADAPT and QFD | $2G$ | Joint-state preparation and terminal QWC readout |
| Prefix factoring of grouped joint states, the default `pair_preparation` | $2G$ | One uncontrolled common prefix and controlled suffixes |
| Proposed vacuum reference with termwise insertion | $2(L+1)$ in the direct construction | Controlled reference operations, uncontrolled forward/inverse blocks and controlled Pauli insertion |
| Grouped vacuum reference row, ADAPT's `"vacuum_reference"` | $2G$ | Two controlled reference operations and one uncontrolled excitation block |
| Grouped vacuum general pair, ADAPT's `"vacuum_restore"` | $2G$ | Two controlled reference operations and three uncontrolled excitation-block executions |
| Real-only grouped joint state, `quadratures="auto"` when the proof holds | $G$ | The corresponding joint-state preparation with only the real quadrature |

For a complete basis of size $b$, grouped measurement of both parts uses $b^2G$ settings, including $G$ per diagonal pair. These are setting counts, not total-shot or accuracy predictions. Scalar-identity Hamiltonians use the existing shortcut described in the GCiM guide.

For a real-only grouped pencil, the count is $b(b+1)G/2$. Only the off-diagonal settings are halved, because diagonal pairs already use a single quadrature.

The single-register interference constructions use $n$ system qubits and one ancilla. Workspace used by preparation or compilation is additional. A double-register NOVQE alternative requires at least $2n+1$ qubits, which is 41–53 for 20–26 system qubits, plus two controlled register swaps. It permits parallel forward preparations but needs a common known vacuum action and a suitable layout. Its larger width and routing cost require separate evaluation before adding it as a selectable construction.

The guide's [decision table](../algorithms/gcim.md#choose-the-measurement-and-the-basis) records small comparisons of the current options on one- and two-qubit inputs. For a measured comparison, record the physical Hamiltonian and its source, mapping, reference and ordered basis, cutoff, backend and calibration, compiler version and layout, pilot and final shots, and classical work. Report ground-state energy error against a reference of stated accuracy when one is available, together with maximum circuit depth, two-qubit gates and elapsed costs. An energy decrease from the initial reference is useful evidence of correlation recovery only when its statistical and systematic uncertainties are resolved. A CCSD reference is application context, not an exact ground-energy certificate.

Hold the basis fixed when attributing a difference to measurement construction. If a pool or selection rule changes, identify the changed trial space and include its construction cost. Without a trusted ground-state reference, report the available error evidence and convergence observations rather than interpreting a projected residual as a bound on missing correlation.

## Device-noise bias

Finite-shot uncertainty does not include gate or readout bias. A reference-energy comparison is an informative first diagnostic, but off-diagonal preparations have different gates, controls and ancilla exposure. Their bias must be considered when interpreting the reconstructed pencil and its energy correction. A small projected residual or a plausible overlap spectrum does not resolve this uncertainty.

A ground-state energy claim requires the uncertainty and bias of the final energy. Attributing additional correlation recovery also requires the uncertainty and bias of the energy correction. Errors common to two estimates can partly cancel, while different acquisition circuits can increase their difference. A large absolute reference bias alone therefore neither qualifies nor rules out a correction. Any mitigation comparison must state the physical quantity estimated, its assumptions and additional quantum and classical costs.
