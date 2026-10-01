# Analytical test fixture

<a id="executable-contract-specimen"></a>This fixture is a synthetic analytical instance of one complete Chebyshev–Lanczos calculation, from scientific input to result, with every intermediate number stated exactly. CI and the Lanczos tests use it as an independent reference. It claims no circuit execution, sampled measurement, transpilation, device calibration, measured runtime or hardware accuracy. The [Lanczos guide](../algorithms/lanczos.md) describes the method and its sources.

[specimen.json](specimen.json) contains fourteen addressable records. [records.schema.json](records.schema.json) contains version 1 JSON Schema fragments. The checker and generator `docs/scripts/contract_specimen.py` checks the structure, the content hashes and the semantic relations. The [claims register](claims.json) links implementation claims to independent checks and to the scope that remains open. [Plan, compare and solve](../scientist.md) describes the public workflow.

## Check and regenerate the fixture {#run-and-regenerate}

From a development checkout with the dev extra installed:

```bash
python docs/scripts/contract_specimen.py --check
python docs/scripts/contract_specimen.py --generate
pytest tests/test_contract_specimen.py
```

The generator requires only Python's standard library and `jsonschema`. It does not import NWQLib, Qiskit, a simulator, or a compiler. `--path FILE` selects an alternate input/output specimen. Regeneration is explicit, and checking never repairs or rewrites an input. Record the source revision and use the development environment described in [Maintenance](../MAINTENANCE.md).

The checker accepts this fixed Hamiltonian, its two named reference preparations, valid subsets of missing moments, and the presence or absence of the analytical ground evidence. It rejects altered scientific premises and does not validate arbitrary Hamiltonians or programs. Structural schema validation alone is weaker than the complete check. Formatting does not matter, because each content hash covers the canonical JSON of `id`, `kind`, `payload` and the ordered digest references. Each reference includes the target SHA-256. The prepared program also has its own payload digest. A changed prepared object cannot keep its old identity, and rehashing a semantically inconsistent chain does not make it valid. These are content-integrity checks, not signatures or execution attestations.

## Scientific relation

Use four system qubits and Qiskit display labels:

```text
H = X_0 + Z_0 + Z_1 + Z_2 + Z_3
psi = |1110>,  m = 2,  alpha = 5
```

Qubit indexing is little endian: the reference basis index is 14, prepared by X on system qubits 1, 2, 3. The physical logical recipe places the three index qubits at positions 0–2 and the four system qubits at positions 3–6. Ordered SELECT labels are `IIIX, IIIZ, IIZI, IZII, ZIII` for index values 0–4. PREP creates equal positive amplitudes on those five values, with zero padding amplitudes. SELECT acts as identity on padding values 5–7. Odd readout maps padding to zero without postselection. That readout equals physical SELECT on the ideal occupied sector, not on arbitrary padding-contaminated states.

With `U=SELECT` and `R=2|G><G|-I`, the state for degree `k` is `(RU)^floor(k/2)|G,psi>`. Odd degrees read SELECT through the label-controlled Pauli basis transform and signed parity. Even degrees read R by applying PREP inverse and returning +1 for index zero and −1 otherwise. The complete unknown observation set is `k=1,2,3`. `mu_0=1` is algebraically known.

Because the spectator qubits stay fixed, H restricted to basis indices 14,15 is `[[-2,1],[1,-4]]`. Direct multiplication gives raw moments `1,-2,5,-16`. Therefore:

```text
(mu_0, mu_1, mu_2, mu_3) = (1, -2/5, -3/5, 86/125)
```

In basis `{psi, (H/5)psi}`, the physical-energy pencil is

```text
S = [[1, -2/5], [-2/5, 1/5]]
H_proj = [[-2, 1], [1, -16/25]]
det(H_proj - E*S) = (E^2 + 6*E + 7)/25
```

The lowest projected energy is `-3-sqrt(2)`. The tests derive the raw moments and direct basis matrix elements independently of the generator's moment reconstruction, then check the determinant polynomial. All arithmetic in that check is rational. No full-space numerical matrix is needed.

The additional **ground identification** uses separate analytical information: X+Z has eigenvalues `+sqrt(2)` and `-sqrt(2)`. The commuting spectator sum has values −3, −1, 1, 3. Thus the full-space minimum is `-3-sqrt(2)`. Removing that evidence preserves the final projected answer and near-spectrum claim, but removes ground accuracy certification. Every such claim is explicitly confined to the exact analytical fixture. The requested projected energy accuracy is established within this fixture. Actual backend accuracy remains NOT_RUN with an unknown total error bound. There is no finite-shot confidence interval.

## Construction and resource meaning

The Program keeps `MeasurementBatch`, `Sequence`, `Repeat` and shared walk `BlockCall` definitions. The chosen reference realization lowers either three X gates or three HZH sequences. HZH=X exactly, including phase. Per setting the alternative changes the reference inventory from 3 X to 6 H + 3 Z. For the whole three-setting workload:

| Logical primitive | Direct X | HZH |
|---|---:|---:|
| X | 9 | 0 |
| H | 0 | 18 |
| Z | 0 | 9 |
| PREP | 5 | 5 |
| PREP inverse | 3 | 3 |
| SELECT | 2 | 2 |
| Zero reflection | 2 | 2 |
| SELECT basis | 2 | 2 |
| Single-qubit measurement | 17 | 17 |

The basis keeps PREP, SELECT and reflection as logical primitives. It does not claim native gate, T, or physical fault-tolerant counts. Scope is one invocation per required setting, with no shots multiplier. An independent traversal of the events checks the resource total against the lowered recipe. A bounded count-4 `Repeat` check confirms that work multiplies while only one walk body stays stored.

The PreparedArtifact is a **logical fixture recipe** with `submission_ready=false` and `target_compilation=not_run`. The trace records three planned settings, three logical prepared recipes, and zero target-prepared, submitted, or executed settings. Exact fixture observations link to analytical receipt events. They are not backend job receipts. Runtime, cost, device capacity and compiler workspace stay unknown because there is no supplied model. Width alone does not establish feasibility.

The production bridge, `tests/test_lanczos.py::test_committed_analytical_fixture_enters_current_reconstruction`, reads the committed fixture's Hamiltonian and reference, plans the current Lanczos method, decodes the three analytical distribution weights and checks its projected result against the independent sector root. It creates no native circuits or execution receipts. Native construction and signed coefficients have separate small method tests. This fixture does not establish compilation, hardware operation or scaling beyond its analytical instance.

## Partial records and rejected mutations {#legal-partial-records-and-falsifiers}

The generator's `missing` argument can remove one or more required contributions. A consistent partial trace and ObservationView remain valid, with an unavailable energy. The checker rejects promoting that record to a final answer. Its `ground_evidence=False` variant is complete and final for the requested projected energy, with ground accuracy INCONCLUSIVE.

Tests also rehash deliberately inconsistent in-memory records to verify that the semantic checks catch a stale selected realization, changed phase populations, wrong event linkage, invented workspace zero, unsupported statistical interval, and kept ground certificate after evidence removal. Separate mutations check payload and reference digest failures. These tests protect the named relations. They do not turn a schema pass or a successful fixture check into a general scientific guarantee.
