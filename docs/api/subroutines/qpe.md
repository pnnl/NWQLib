# Coherent QPE {#coherent-qpe-api}

Build the QFT-based phase-estimation circuit of a dense unitary `U`, without measurement: Hadamards on an `m`-qubit phase register, controlled powers `U^(2**q)` and an inverse QFT (Cleve, Ekert, Macchiavello and Mosca, arXiv:quant-ph/9708016v1, Sec. 5). Import it from `nwqlib.subroutines.qpe`. The statistical phase estimators and their Results are in the [QPE Method API](../algorithms/qpe.md).

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from nwqlib.subroutines.qpe import build_coherent_qpe_circuit

U = np.diag([1.0, np.exp(2j * np.pi * 0.25)])
qpe = build_coherent_qpe_circuit(U, num_phase_qubits=2)
circuit = QuantumCircuit(qpe.circuit.num_qubits)
circuit.x(2)  # system qubit 2 holds the eigenvector |1>
circuit.compose(qpe.circuit, inplace=True)
print(Statevector(circuit).probabilities([0, 1]).round(10))
```

```text
[0. 1. 0. 0.]
```

The eigenphase is `theta = 0.25`, and the two phase qubits hold the integer `2**2 * theta = 1` with probability 1.

## Conventions

- Phase qubit `q` controls `U^(2**q)`, and the phase register holds its integer in little-endian order. The register order is `phase` followed by `system`.
- Each power is a controlled `UnitaryGate` of the full matrix, so a global phase of `U` becomes a relative phase under control and enters the estimated eigenphase.
- The builder docstring below states where these conventions differ from the paper's bit order.

## Build the circuit

::: nwqlib.subroutines.qpe.coherent.build_coherent_qpe_circuit
    options:
      heading_level: 3

::: nwqlib.subroutines.qpe.coherent.CoherentQPECircuit
    options:
      heading_level: 3

## Accuracy and limits

- The `m` powers come from repeated squaring, exactly `m - 1` matrix products, and exist only as gate data inside the returned circuit, with no separate cache.
- The builder entry above states the accuracy of the synthesized powers, the unitarity window of the input and the limit checks.

## Source map

Code paths are relative to `nwqlib.subroutines.qpe`. The source is Cleve, Ekert, Macchiavello and Mosca, "Quantum algorithms revisited", Proc. R. Soc. Lond. A 454, 339 (1998). Section, figure, equation and page numbers refer to [arXiv:quant-ph/9708016v1](https://arxiv.org/abs/quant-ph/9708016v1). The rows assume the system register holds an eigenvector of U with eigenvalue exp(2*pi*i*theta), the paper's exp(2*pi*i*phi). The builder docstring above states the conventions that differ from the paper.

| Scientific step | Source | Location | Code |
| --- | --- | --- | --- |
| Hadamards and controlled powers U^(2**q) put the phase register in the state of Eq. (5.1) | Cleve et al., arXiv:quant-ph/9708016v1 | Sec. 5, Fig. 6 and Eq. (5.1), p. 10 | `coherent.build_coherent_qpe_circuit` |
| QFT on the m-qubit phase register and its inverse | Cleve et al., arXiv:quant-ph/9708016v1, and Qiskit's `QFTGate` | Eq. (4.1), p. 8. `QFTGate` includes the output reversal that the network of Fig. 5 leaves out | `coherent.build_coherent_qpe_circuit` |
| Readout of the basis state j when theta = j/2**m modulo 1, otherwise of an integer nearest to 2**m*theta (modulo 2**m) with probability at least 4/pi**2 | Cleve et al., arXiv:quant-ph/9708016v1 | Eqs. (5.2)–(5.4), p. 11 | `coherent.build_coherent_qpe_circuit` |
| Little-endian phase integer, with phase qubit q controlling U^(2**q) | Qiskit bit order. The paper puts the most significant bit on its top qubit, which Fig. 6 labels U^(2^j) with j = m-1, and both orders encode the same integer | Sec. 4, p. 8, and Eq. (5.1), p. 10 | `coherent.build_coherent_qpe_circuit` |
