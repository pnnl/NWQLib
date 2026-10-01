# Coherent QPE API

This subroutine constructs a coherent phase-register circuit from the supplied unitary. The statistical estimators and their prepared-population Results are documented in the [QPE Method API](../algorithms/qpe.md).

The m-bit power ladder starts from U and squares the previous matrix once for each further bit, so it forms exactly m-1 matrix products. Phase qubit q controls U^(2**q), and the phase register holds its integer in little-endian order. Each power is a controlled `UnitaryGate` of the full matrix, so a global phase of U becomes a relative phase under control and enters the estimated eigenphase. Its controlled matrix is synthesized as its unitary polar factor to binary64 rounding ([Controlled dense unitaries](../../blocks.md#controlled-dense-unitaries)), so the realized power differs from the computed power by the order of the power's unitarity defect. Each squaring can double the defect of an input that passed the 1e-8 admission window, and Qiskit's own unitarity check of the gate matrix (`numpy.allclose` of U†U with the identity, atol 1e-8) would then reject a higher power, so the powers skip that check. Before the unitarity check, `max_work=1_000_000_000` and `max_bytes` admit the m powers and the exact synthesis of each controlled power (`_dense_synthesis.controlled_synthesis_size`, [selected blocks](../../blocks.md#admission-of-the-exact-synthesis)). An inverse QFT on the phase register follows the ladder. The power matrices exist only as gate data inside the returned circuit, with no separate cache.

::: nwqlib.subroutines.qpe.coherent

## Source map

Owners are relative to `nwqlib.subroutines.qpe`. The source is Cleve, Ekert, Macchiavello and Mosca, "Quantum algorithms revisited", Proc. R. Soc. Lond. A 454, 339 (1998). Section, figure, equation and page numbers refer to [arXiv:quant-ph/9708016v1](https://arxiv.org/abs/quant-ph/9708016v1). The rows assume the system register holds an eigenvector of U with eigenvalue exp(2*pi*i*theta), the paper's exp(2*pi*i*phi). The builder docstring above states the conventions that differ from the paper.

| Scientific step | Source | Location | Code owner |
| --- | --- | --- | --- |
| Hadamards and controlled powers U^(2**q) put the phase register in the state of Eq. (5.1) | Cleve et al., arXiv:quant-ph/9708016v1 | Sec. 5, Fig. 6 and Eq. (5.1), p. 10 | `coherent.build_coherent_qpe_circuit` |
| QFT on the m-qubit phase register and its inverse | Cleve et al., arXiv:quant-ph/9708016v1, and Qiskit's `QFTGate` | Eq. (4.1), p. 8. `QFTGate` includes the output reversal that the network of Fig. 5 leaves out | `coherent.build_coherent_qpe_circuit` |
| Readout of the basis state j when theta = j/2**m modulo 1, otherwise of an integer nearest to 2**m*theta (modulo 2**m) with probability at least 4/pi**2 | Cleve et al., arXiv:quant-ph/9708016v1 | Eqs. (5.2)–(5.4), p. 11 | `coherent.build_coherent_qpe_circuit` |
| Little-endian phase integer, with phase qubit q controlling U^(2**q) | Qiskit bit order. The paper puts the most significant bit on its top qubit, which Fig. 6 labels U^(2^j) with j = m-1, and both orders encode the same integer | Sec. 4, p. 8, and Eq. (5.1), p. 10 | `coherent.build_coherent_qpe_circuit` |
