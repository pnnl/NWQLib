# QFD

<a id="qfd-api"></a>Use `Eigenproblem(A=...)` with `QFD(initial_state=..., time_step=..., powers=...)` to estimate the smallest eigenvalue of A in a basis of time-evolved states `V**p |psi>`. Here psi is your reference state, V is one fixed product-formula step, by default of the nonidentity Pauli terms of A, and p runs over the integer powers you choose. The result reports the lowest projected Ritz value, the energy of the reference state, and the reference energy minus the Ritz value. The Ritz value does not certify the full-space ground energy.

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms import QFD
from nwqlib.algorithms.qfd import chemistry_evolution_terms
```

Every object on this page imports from `nwqlib.algorithms.qfd`, and `QFD` and `QFDResult` also import from `nwqlib.algorithms`.

`QFD` solves the Hamiltonian pencil of Parrish and McMahon, arXiv:1909.08925v1, Sec. II.C, Eqs. (5)–(8), with the states of one fixed product-formula step in place of exact time evolution. It obtains the projected matrices H and S as `FixedGCIM` does, from joint states of an ancilla and the system, so the [GCiM and ADAPT](gcim.md) page documents the objects that the two methods share: `PencilAcquisition`, `PencilSampling`, `ShotAllocation`, `ProjectedPencil` and the matrix-element records that `QFDResult` holds.

## Solve in a basis of evolved states

::: nwqlib.algorithms.qfd.method.QFD
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qfd.records.QFDResult
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qfd.records.QFDReconstruction
    options:
      heading_level: 3
      members:
        - evolution_rows

## Evolve with a generator from chemistry integrals

With `evolution="fermionic_blocks"`, `QFD` builds V from an explicit fermionic generator B, one complete number-conserving Hermitian block of ladder terms at a time, so every basis state from an occupation reference keeps its particle number. H, S and the energies remain those of A. `chemistry_evolution_terms` takes the data from `build_gcim_chemistry_problem` and returns B, built from its integrals, with a `QFDEvolutionDiagnostics` record of how B was derived. Passed to `QFD` as `evolution_diagnostics`, the record is saved with the Method. B and the Hamiltonian A of the same data can differ, and the Plan states upper bounds on their distance.

::: nwqlib.algorithms.qfd.chemistry.chemistry_evolution_terms
    options:
      heading_level: 3

::: nwqlib.algorithms.qfd.records.QFDEvolutionDiagnostics
    options:
      heading_level: 3
      members: false

`solve` calls `plan` and the other hooks of a Method. [Extending NWQLib](../extending.md) describes them.
