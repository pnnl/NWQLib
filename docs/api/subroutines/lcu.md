# LCU API

`prepare_lcu_data` records the supplied coefficients and unitaries. The explicit native builders consume that data to construct PREP, SELECT or their composition. Import these operations from `nwqlib.subroutines.lcu`.

PREP uses amplitudes `sqrt(abs(c_j))/sqrt(alpha)`, where `alpha=sum(abs(c_j))`, and SELECT carries the coefficient phases. A nonzero global scale changes alpha while preserving the normalized coefficient direction. The default `coefficient_atol=0` therefore rejects only zero total weight. A supplied positive threshold applies to that total weight and never removes individual coefficients.

The all-zero control block of PREP-SELECT-PREP† is `sum_j c_j U_j / alpha`. SELECT applies `(c_j / abs(c_j)) U_j` on address `j`. The control register precedes the system register, so it occupies the low-order bits of Qiskit's little-endian index. `build_lcu_select` synthesizes each controlled branch exactly from its whole controlled matrix, so the realized SELECT equals this relation to binary64 rounding when every U_j is unitary to rounding. A U_j that passes Qiskit's constructor check with a larger unitarity defect is realized as its unitary polar factor. [Controlled dense unitaries](../../blocks.md#controlled-dense-unitaries) describes the synthesis and its CX count.

`LCUCircuit.preparation_l2_error` reports the phase-sensitive coefficient-state error from the actual preparation owner. Direct preparation reports zero algorithmic error in the ideal-gate model. A layered MPS preparation whose circuit error has not been evaluated reports `None`. This quantity has the same phase convention as coherent PREP, inverse PREP and controlled use.

Coefficient count, finite values and nonzero total weight are checked before unitary conversion. The dense data stores exactly N supplied matrices; the P-N padding addresses act as identity without stored matrices or controlled gates. A supplied zero-weight nonidentity unitary remains a real SELECT branch and undergoes the normal native unitarity admission. Gate-level data has an empty matrix tuple and cannot be passed to the dense SELECT builder.

Intake and native builders accept `max_bytes=10_000_000_000` and `max_work=1_000_000_000`. Before any matrix conversion or synthesis they check the coefficient and PREP tables, the supplied matrices and, for a dense SELECT with a control register, the exact synthesis of each controlled branch, for N terms, P addresses and system dimension D. A branch is synthesized from its PD-square controlled matrix. With M = PD and m = log2(M), it is charged `11 M³ + (m² + 5m + 256) M²` work units, `256 M² + 65536` bytes of working arrays, which one branch at a time uses, and `176 M² + 16384` bytes for the circuit it keeps. The default work limit is a fuse against runaway planning work, sized so that no example or test workload of this release meets it. One branch took about 0.04 s at PD = 128 on an Apple-silicon Mac. A dense SELECT above the default needs an explicitly raised `max_work`. `_admit_lcu`, `controlled_synthesis_size` in `subroutines/_dense_synthesis.py` and the `DEFAULT_MAX_LCU_WORK` entry in [Engineering constants](../../ENGINEERING_CONSTANTS.md) give the derivation and the measurements behind these terms. Pass limits to each separately requested stage. Combined builders forward the same values. Work counts scalar operations, not time. Layered tensor-library workspace remains an uncounted dependency cost.

::: nwqlib.subroutines.lcu.core

::: nwqlib.subroutines.lcu.registry

## Source map

Owners are relative to `nwqlib.subroutines`. Equation and section numbers refer to the listed arXiv versions.

| Scientific step | Source | Location | Code owner |
| --- | --- | --- | --- |
| Two-term PREP-SELECT-PREP† circuit and its extension to general combinations | Childs and Wiebe, arXiv:1202.5822v1 | Lemma 2, Fig. 1 and Theorem 3 | `lcu.core.build_lcu_circuit` |
| Complex weights absorbed as phases of the unitaries | Childs and Wiebe, arXiv:1202.5822v1 | Sec. II, after Eq. (4) | `lcu.core.prepare_lcu_data` |
| Single-register PREP state and `alpha` as the coefficient 1-norm | Low and Chuang, arXiv:1610.06546v3 | Lemma 5 and Eq. (10) | `lcu.data._coefficient_bookkeeping` |
| Subnormalization of a combination of block encodings | Gilyén et al., arXiv:1806.01838v1 | Lemma 52 | Composition rule in the [Framework](../../FRAMEWORK.md#block-encoding-and-qsp-conventions) |
| Branch `j` fires on control value `j` | Qiskit little-endian `ctrl_state` convention | | `lcu.core._controlled_branch_gate` |
| Exact synthesis of each controlled branch, demultiplexed at the top because the controlled matrix is block diagonal in each control qubit | Shende, Bullock and Markov, arXiv:quant-ph/0406176v5, with the block-ZXZ steps of Krol and Al-Ars, arXiv:2403.13692v2 | Theorem 12. [Controlled dense unitaries](../../blocks.md#controlled-dense-unitaries) lists the other locators | `qiskit_compat.controlled`, `_dense_synthesis.controlled_unitary_circuit` |
| Intake and construction admission | NWQLib contract, registered as `DEFAULT_MAX_LCU_WORK` in [Engineering constants](../../ENGINEERING_CONSTANTS.md). The synthesis terms follow the recursion of the exact synthesis | `_dense_synthesis.controlled_synthesis_size` | `lcu.core._admit_lcu` |
