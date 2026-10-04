# Inputs and input types

Build the matrix, operator and state arguments of a Problem, and look up the types that Problem and record fields accept. The [Supply inputs](../inputs.md) guide says which Method accepts which representation, and [Choose a problem and output](../problems.md) lists the Problems.

```python
from nwqlib.operators import operator_input, ingest_pauli, PeriodicStencil
from nwqlib.problems import state_input
```

A Problem field accepts a NumPy array, a SciPy sparse matrix, a Qiskit `SparsePauliOp`, a `PeriodicStencil` or a state vector directly. The functions on this page make the same conversion explicit and return a handle that keeps an immutable copy of the data in its original representation. Passing one handle to several Problems reuses that copy:

```python
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from nwqlib import LinearSystem
from nwqlib.operators import operator_input
from nwqlib.problems import state_input

H = operator_input(SparsePauliOp(["ZI", "XX"], coeffs=[1.0, 0.5]))
print(H.manifest.reference.representation, H.structure, H.basis.dimension)

A = operator_input(np.array([[2.0, 1.0], [1.0, 2.0]]))
b = state_input([3, 4j])
print(A.matvec(np.array([1.0, 1.0])))
print(b.preparation.physical_scale.as_float())

problem = LinearSystem(A=A, b=b)
print(problem.A is A)
```

```text
pauli hermitian 4
[3. 3.]
5.0
True
```

The Pauli operator stays a sum of two terms on 2 qubits, the dense matrix maps `[1, 1]` to `[3, 3]`, and the state keeps its norm 5 beside the normalized direction `[0.6, 0.8j]` that a circuit prepares.

| You have | Call | You get |
| --- | --- | --- |
| A dense matrix: NumPy array, nested lists | `operator_input(A)` or `ingest_dense(A)` | `OperatorInput`, dense |
| A SciPy CSR or CSC matrix | `operator_input(A)` or `ingest_sparse(A)` | `OperatorInput`, kept sparse |
| A Qiskit `SparsePauliOp` | `operator_input(op)` | `OperatorInput`, Pauli terms |
| Pauli labels and coefficients | `ingest_pauli(terms, num_qubits=q)` | `OperatorInput`, Pauli terms |
| Packed Pauli masks | `ingest_pauli_masks(x, z, coefficients, num_qubits=q)` | `OperatorInput`, Pauli terms |
| A periodic grid operator with mass, diffusion and potential | `PeriodicStencil(q, mass, diffusion, potential)` | Parameters, accepted as a Problem's matrix |
| Fermionic ladder-operator strings | `ingest_fermion(terms, num_modes=q)`, then `.fermion_terms().to_pauli(mapping="jw")` | `OperatorInput`, fermionic, then Pauli |
| A double-factorized electronic Hamiltonian | `ingest_df(T, factors, ...)`, then `.to_pauli()` | `FactorizedHamiltonian`, then `DFConversion` |
| An XACC Hamiltonian file | [`read_xacc` or `parse_xacc`](subroutines/fermionic_pool.md) | Qiskit `SparsePauliOp` |
| A product of operators | `FactorizedOperatorProduct((A, B))` | The factors, kept separate |
| A state vector or Qiskit `Statevector` | `state_input(v)` or `ingest_vector(v)` | `StateInput`, unnormalized |
| A state-preparation circuit | `state_input(circuit)` or `bind_preparation_circuit(...)` | `StateInput` |
| One pair of amplitudes per qubit | `ingest_product(rows)` | `StateInput`, product state |
| A computational basis state | `ingest_occupation(bits, num_qubits=q)` | `StateInput`, at most q X gates |
| Norm and spectral bounds of an operator | `refine_operator_facts(A, unit=..., scope=..., refinements=(...))` | `OperatorFactReport` |

Every function checks the bytes it will keep against `max_bytes`, default 10 GB (decimal, `10_000_000_000`), before it copies data. [Input cost controls](../development/input_contracts.md) gives each size formula. Coordinates follow Qiskit, with qubit 0 the rightmost tensor factor and the rightmost character of a Pauli label.

## Operators

::: nwqlib.operators.inputs.operator_input
    options:
      heading_level: 3

::: nwqlib.operators.inputs.ingest_dense
    options:
      heading_level: 3

::: nwqlib.operators.inputs.ingest_sparse
    options:
      heading_level: 3

::: nwqlib.operators.inputs.ingest_pauli
    options:
      heading_level: 3

::: nwqlib.operators.inputs.pauli_table
    options:
      heading_level: 3

::: nwqlib.operators.inputs.ingest_pauli_masks
    options:
      heading_level: 3

::: nwqlib.operators.inputs.PeriodicStencil
    options:
      heading_level: 3

::: nwqlib.operators.inputs.ingest_periodic_stencil
    options:
      heading_level: 3

::: nwqlib.operators.inputs.OperatorInput
    options:
      heading_level: 3

## Fermionic and factorized operators

::: nwqlib.operators._fermion.ingest_fermion
    options:
      heading_level: 3

::: nwqlib.operators._fermion.fermion_table
    options:
      heading_level: 3

::: nwqlib.operators._fermion.FermionTerms
    options:
      heading_level: 3

::: nwqlib.operators.df.ingest_df
    options:
      heading_level: 3

::: nwqlib.operators.df.FactorizedHamiltonian
    options:
      heading_level: 3

::: nwqlib.operators.df.DFConversion
    options:
      heading_level: 3

::: nwqlib.operators.df.DFConversionReceipt
    options:
      heading_level: 3

::: nwqlib.operators.df.DFManifest
    options:
      heading_level: 3

::: nwqlib.operators._factorized.FactorizedOperatorProduct
    options:
      heading_level: 3

::: nwqlib.operators.access.ProductManifest
    options:
      heading_level: 3

## States

::: nwqlib.problems.inputs.state_input
    options:
      heading_level: 3

::: nwqlib.problems.inputs.ingest_vector
    options:
      heading_level: 3

::: nwqlib.problems.inputs.ingest_product
    options:
      heading_level: 3

::: nwqlib.problems.inputs.ingest_occupation
    options:
      heading_level: 3

::: nwqlib.problems.inputs.bind_preparation_circuit
    options:
      heading_level: 3

::: nwqlib.problems.inputs.StateInput
    options:
      heading_level: 3

::: nwqlib.problems.inputs.StatePreparationSpec
    options:
      heading_level: 3

::: nwqlib.problems.inputs.prepare_qiskit
    options:
      heading_level: 3

::: nwqlib.problems.inputs.QiskitPreparation
    options:
      heading_level: 3

::: nwqlib.problems.inputs.PhysicalScale
    options:
      heading_level: 3
      members:
        - as_float
        - squared_as_float

## Operator bounds

::: nwqlib.operators.refinement.refine_operator_facts
    options:
      heading_level: 3

::: nwqlib.operators.refinement.OperatorFactReport
    options:
      heading_level: 3

::: nwqlib.operators.refinement.RefinementOption
    options:
      heading_level: 3

::: nwqlib.operators.refinement.RefinementReceipt
    options:
      heading_level: 3

## Input metadata

::: nwqlib.operators.access.InputManifest
    options:
      heading_level: 3

## Input types

Problem and record fields use these types. A field typed `OperatorData` or `StateData` accepts the raw forms in the table and stores the handle that [`operator_input`][nwqlib.operators.inputs.operator_input] or [`state_input`][nwqlib.problems.inputs.state_input] returns.

| Type | Accepts |
| --- | --- |
| `OperatorData` | A square NumPy array or nested lists, a canonical SciPy CSR or CSC matrix, a Qiskit `SparsePauliOp`, a `PeriodicStencil` or an `OperatorInput` |
| `StateData` | A vector of numbers, a Qiskit `Statevector` or `QuantumCircuit`, or a `StateInput` |
| `Real` | A finite float. An `int` is converted, and `-0.0` becomes `0.0` |
| `Nonnegative` | A `Real` that is 0 or larger |
| `PositiveInt` | An `int` larger than 0 |
| `Count` | An `int` that is 0 or larger |
| `Text` | A string that is not empty after surrounding whitespace is removed |
| `ContentID` | A string identifying record content. NWQLib generates `"sha256:"` followed by 64 lowercase hexadecimal digits |
| `Unit`, `Scope`, `Source`, `Basis` | Records that label a value's unit, its scope, its source and a coordinate basis |
| `Float64`, `Complex128`, `Rational` | Scalar records: a finite binary64 value, a complex pair of them, an exact integer ratio in lowest terms |
| `Symbol`, `Limit` | A named mathematical symbol, and a limit on one resource in one workflow stage |

None of these types accepts a `bool` where a number is expected.

::: nwqlib.problems.records.OperatorData
    options:
      heading_level: 3
      show_attribute_values: false

::: nwqlib.problems.records.StateData
    options:
      heading_level: 3
      show_attribute_values: false

::: nwqlib.core.records.Real
    options:
      heading_level: 3
      show_attribute_values: false

::: nwqlib.core.records.Nonnegative
    options:
      heading_level: 3
      show_attribute_values: false

::: nwqlib.core.records.PositiveInt
    options:
      heading_level: 3
      show_attribute_values: false

::: nwqlib.operators.access.Count
    options:
      heading_level: 3
      show_attribute_values: false

::: nwqlib.core.records.Text
    options:
      heading_level: 3
      show_attribute_values: false

::: nwqlib.core.records.ContentID
    options:
      heading_level: 3
      show_attribute_values: false

::: nwqlib.core.records.Unit
    options:
      heading_level: 3

::: nwqlib.core.records.Scope
    options:
      heading_level: 3

::: nwqlib.core.records.Source
    options:
      heading_level: 3

::: nwqlib.core.records.Basis
    options:
      heading_level: 3

::: nwqlib.core.records.Float64
    options:
      heading_level: 3

::: nwqlib.core.records.Complex128
    options:
      heading_level: 3

::: nwqlib.core.records.Rational
    options:
      heading_level: 3

::: nwqlib.core.records.Symbol
    options:
      heading_level: 3

::: nwqlib.core.records.Limit
    options:
      heading_level: 3
