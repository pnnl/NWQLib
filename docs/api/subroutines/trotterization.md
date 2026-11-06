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

To compare this circuit with a reference matrix, use `qiskit.quantum_info.Operator(circuit.decompose())`, because with Qiskit 2.6.0 `Operator(circuit)` evaluates the circuit's one `PauliEvolutionGate` as the exact exponential of `H`, which agrees with `scipy.linalg.expm` at any step count and hides the product-formula error.

## The error bound

- The first-order bound is Proposition 9, Eq. (120), and the second-order bound is Proposition 10, Eq. (121), of Childs, Su, Tran, Wiebe and Zhu, Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020. One application of the order-`o` formula obeys `bound(t) = W_up * t**(o + 1)`.
- For an evolution at one supplied time, the step count is the smallest one allowed by the rule of their Sec. V B: with `r` steps the total is at most `r * bound(t / r)`.
- The bound expands each commutator norm over Pauli strings by the triangle inequality, with `||[P, Q]|| = 2` when P and Q anticommute and `||[P, [Q, R]]|| = 4` when both commutators are nonzero. It upper-bounds the theorem's tail-sum norms and is not the evolution error itself. At second order `bound_variant="exact_census"` uses the full Pauli-triangle expression. For p ordered terms on q qubits, with `w = ceil(q/64)` mask words and `e_i` the anticommuting partners j > i of term i, that expression tests every k > i, j included, for each anticommuting pair, which makes `w*N(A)` word tests with `N(A) = sum_i e_i*(p-i-1)`. The default `"auto"` counts this from the pair scan, before any nested test, and uses the full expression when the count is at most 1,300,000,000 (`nwqlib.subroutines.trotterization.error_budget.EXACT_CENSUS_WORD_TESTS`) and `"relaxed_prefix"` above it. LCHS contracts K node weightings with the same masks and compares `N(A)*max(w, K)` with the same threshold ([LCHS product-formula planning](../../algorithms/lchs.md#product-formula-planning)). `bound_variant="relaxed_prefix"` uses pair tests and weighted degrees only, replacing each nested mass by `min(S_i,D_i+D_j-a_i)`, where `a_i` is the coefficient magnitude of term i, `S_i` the later coefficient mass and `D_v` the mass anticommuting with term v. It can choose more steps than the full expression. [Proposition 36](../../mathematics.md#r36) gives the proof. Its work is `w*P+2*p+4*E`, with `P = p(p-1)/2` possible pairs and E anticommuting pairs. The full expression costs `w*(P+N(A))` word tests and `E+N(A)` contraction visits and forms its nested masks 32 pairs at a time.
- The prefactor `W_up` is evaluated with scaled binary64 upper products and sums followed by rational rescaling, so it bounds the chosen expression from above and can exceed it. The bound and the step inversion use exact rationals formed from `W_up` and the binary64 time and budget, so no intermediate rounds to zero or overflows. The chosen count is minimal for `W_up`, and a returned bound is rounded upward to binary64, which keeps a nonzero bound positive and the bound at the chosen count at most the budget.

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
- The keyword arguments `max_work` (default `None`, no limit) and `max_bytes` (default `None`, the physical memory) limit the Pauli commutation table, the anticommuting pairs and nested commutation indicators from which the bound is evaluated. The label conversion, `128*p*(q+1)` bytes, is checked against `max_bytes` before the labels are converted, and the pair rows are checked as they grow. With `max_work`, the label conversion and the pair scan are checked before the labels are converted, and the work of the chosen expression after the pair scan and before any nested test. The dominant arrays of the chosen expression, `272*p+16*p*w+8*E` bytes, plus `16*p` for the relaxation or `40*T*w` for the largest tile of nested masks of the full expression, are checked against `max_bytes` at the same point. The largest tile holds `T = max_i min(32, e_i)*(p-i-1)` mask entries, zero when no pair anticommutes.
- The table stores the anticommuting pairs as one row of later partners per term, so its storage grows with the number of anticommuting pairs. The full expression forms its nested masks one tile of pairs at a time.
- A refusal reports the required work or bytes and the limit. For the full second-order expression, chosen or requested, it also states the work or bytes of `bound_variant="relaxed_prefix"`, which can be requested explicitly. A limit never changes the chosen expression.

## Common time grid of QPE powers

A QPE trajectory on a common time grid uses one step for all power positions. At each requested power it checks the circuit prefix, meaning all steps up to that power position, against that power's allowance for the sum of the product-formula, pruning, time-displacement, identity-phase and rotation-angle error bounds. The [QPE implementation map](../../algorithms/qpe.md#implementation-map) lists where controlled product-formula powers use these bounds.

- `val(tau)` denotes the exact real value of the stored binary64 time unit, and a nonnegative integer power `p` targets the time `p * val(tau)`.
- If `h` is the exact real value of the binary64 step that the circuit uses and `r` is its cumulative count, the second-order product-formula bound is `r * W * abs(h)**3`. The discrepancy between `r * h` and the target contributes a separate operator bound.
- A complete controlled-power bound also includes pruning and the applicable identity-phase and rotation-angle formation terms. The `common_*` fields of `TrotterStepSelection` record this choice.
- The first step count inverts only the product-formula and pruning terms, so rounding can push a complete bound above its allowance. The selection then checks at most two larger counts on the same grid and uses a count only when every complete bound fits its allowance ([Commutator count and common step](../../algorithms/qpe.md#commutator-count-and-common-step)). A refusal shows that this rule confirmed no count, not that the target error is unattainable.

## Proposition numbering

Unless marked as arXiv:1912.08854v3, every proposition, equation and section number on this page and in the docstrings follows the Phys. Rev. X version, doi:10.1103/PhysRevX.11.011020. The arXiv:1912.08854v3 preprint, titled "A Theory of Trotter Error", numbers the same results Proposition 15/Eq. (145) (p. 38), Proposition 16/Eq. (152) (p. 39) and Sec. 5.2 (p. 40).
