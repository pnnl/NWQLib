# Trotterization {#trotterization-api}

Choose the number of Trotter steps for a Pauli Hamiltonian, an evolution time and an operator-norm error budget, evaluate the product-formula error bound behind that choice, and build the product-formula circuit. Import the functions from `nwqlib.subroutines.trotterization`.

```python
from qiskit.quantum_info import SparsePauliOp
from nwqlib.subroutines.trotterization import (
    build_trotter_evolution_circuit,
    select_trotter_step_count,
)

H = SparsePauliOp(["XI", "ZZ"], [0.5, 0.5])
selection = select_trotter_step_count(H, time=2.0, error_budget=0.01, order=2)
print(selection.step_count, round(selection.bound_value, 10))
circuit = build_trotter_evolution_circuit(H, selection)
print(selection.synthesis_method, circuit.num_qubits)
```

```text
8 0.0078125
suzuki_trotter 2
```

The second-order coefficient of this `H` is `1/16`, and 8 is the smallest `r` with `(1/16) * 2**3 / r**2 <= 0.01`.

## The error bound

- The first-order bound is Proposition 9, Eq. (120), and the second-order bound is Proposition 10, Eq. (121), of Childs, Su, Tran, Wiebe and Zhu, Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020. One application of the order-`o` formula obeys `bound(t) = W_up * t**(o + 1)`.
- For an evolution at one supplied time, the step count is the smallest one allowed by the rule of their Sec. V B: with `r` steps the total is at most `r * bound(t / r)`.
- The bound expands each commutator norm over Pauli strings by the triangle inequality, with `||[P, Q]|| = 2` when P and Q anticommute and `||[P, [Q, R]]|| = 4` when both commutators are nonzero. It upper-bounds the theorem's tail-sum norms and is not the actual evolution error. At second order the default `"exact_census"` uses the full Pauli-triangle expression. `bound_variant="relaxed_prefix"` uses pair tests and weighted degrees only, replacing each nested mass by `min(S_i,D_i+D_j-a_i)`, where `S_i` is the later coefficient mass and `D_v` the mass anticommuting with term v. It can select more steps than the full expression. [Proposition 36](../../mathematics.md#r36) gives the proof. Its work is `w*P+2*p+4*E` for packed width w, P possible pairs and E anticommuting pairs. Its contraction scratch is `72*p+96*b+96*ceil(E/b)+65536` bytes at block b, within the complete input and pair-table limit.
- The prefactor `W_up` is evaluated with scaled binary64 upper products and sums followed by rational rescaling, so it bounds the selected expression and can exceed it. The bound and the step inversion use exact rationals formed from `W_up` and the binary64 time and budget, so no intermediate rounds to zero or overflows. The selected count is minimal for `W_up`, and a returned bound is rounded upward to binary64, which keeps a nonzero bound positive and a selected bound at most the budget.

## Functions and records

::: nwqlib.subroutines.trotterization.select_trotter_step_count
    options:
      heading_level: 3

::: nwqlib.subroutines.trotterization.build_trotter_evolution_circuit
    options:
      heading_level: 3

::: nwqlib.subroutines.trotterization.trotter_bound_coefficient
    options:
      heading_level: 3

::: nwqlib.subroutines.trotterization.evaluate_trotter_bound
    options:
      heading_level: 3

::: nwqlib.subroutines.trotterization.dense_trotter_bound_coefficient
    options:
      heading_level: 3

::: nwqlib.subroutines.trotterization.TrotterStepSelection
    options:
      heading_level: 3

## Cost and limits

- A coefficient or fixed-step bound above the largest binary64 number raises `ValueError`.
- The keyword arguments `max_work` (default 1,000,000,000) and `max_bytes` (default 10,000,000,000) limit the tests of anticommuting Pauli pairs and triples. The pair stage is checked before label conversion. The requested expression is checked again with the actual pair count before the nested tests or the contraction.
- The full second-order expression stores three intp indices per surviving triple and can require cubic storage in the number of terms. The contraction block limits coefficient-reduction scratch and does not limit the stored index count. If the requested expression does not fit, the function raises before the stage that would exceed the limits.
- A work- or byte-limit refusal names the failed stage and reports its required work and the current `max_work` and `max_bytes`. A byte-fit failure is reported only when no checked contraction-block candidate fits the current byte limit.
- An order-two refusal of the full expression also names `bound_variant="relaxed_prefix"` as an explicit alternative, which is checked against the same limits.

## Common time grid of QPE powers

A QPE trajectory on a common time grid selects one step for all power positions. At each requested power it checks the circuit prefix, meaning all steps up to that power position, against that power's allowance for the sum of the product-formula, pruning, time-displacement, identity-phase and rotation-angle error bounds. The [QPE implementation map](../../algorithms/qpe.md#implementation-map) lists where controlled product-formula powers use these bounds.

- `val(tau)` denotes the exact real value of the stored binary64 time unit, and a nonnegative integer power `p` targets the time `p * val(tau)`.
- If `h` is the exact real value of the emitted binary64 step and `r` is its cumulative count, the second-order product-formula bound is `r * W * abs(h)**3`. The discrepancy between `r * h` and the target contributes a separate operator bound.
- A complete controlled-power bound also includes pruning and the applicable identity-phase and rotation-angle formation terms. The `common_*` fields of `TrotterStepSelection` record this selection.

## Proposition numbering

Unless marked as arXiv:1912.08854v3, every proposition, equation and section number on this page and in the docstrings follows the Phys. Rev. X version, doi:10.1103/PhysRevX.11.011020. The arXiv:1912.08854v3 preprint, titled "A Theory of Trotter Error", numbers the same results Proposition 15/Eq. (145) (p. 38), Proposition 16/Eq. (152) (p. 39) and Sec. 5.2 (p. 40).
