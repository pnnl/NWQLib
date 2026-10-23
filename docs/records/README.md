# Record contracts

<a id="scientific-records"></a>NWQLib's scientific inputs, output definitions, accuracy criteria, error facts and limits are records, frozen Pydantic models derived from `nwqlib.core.records.Record`. This page gives the rules every record follows and the script that checks them. [Choose a problem and output](../problems.md) describes the Problem types, their outputs and units for users, and [Supply inputs](../inputs.md) describes the operator and state input objects.

Importing the scientific input classes, such as `Eigenproblem`, `LinearDynamics` or `Expectation` from `nwqlib`, does not load vendor SDKs or execute algorithms. Record metadata uses Pydantic and standard-library code. Checking and using numerical or symbolic inputs can use the base NumPy, SciPy and SymPy dependencies. The SDKs that backend execution needs are loaded only by the operation that uses them.

## Where each record is defined {#ownership-and-scientific-meaning}

| Code | Defines |
| --- | --- |
| `nwqlib.core` | Immutable records, exact scalar encodings, units, references and limits |
| `nwqlib.evidence` | Facts, error frames and accuracy assessment |
| `nwqlib.problems` | Scientific questions (Problems) and their output definitions |
| The Method that planning chose | Its applicability, numerical choices, execution and analysis |

The Problem table, the output definitions, the unit rules and the residual convention of `ConstrainedOptimization` are in [Choose a problem and output](../problems.md#problem-types).

## Accuracy and facts

`Accuracy` holds three things:

- exactly one tolerance, absolute or relative, positive and finite
- a confidence strictly between zero and one
- the error component it applies to, total or sampling

It holds no achieved-accuracy status. The quantity, unit, scope and conditioning that the tolerance refers to come from the output's error frame, separately from `Accuracy`. A relative tolerance needs supported evidence of the reference scale, and a zero or unknown reference norm never becomes an implicit denominator. [Relative criteria](../verification.md#relative-criteria) describes assessment with an explicit reference and an absolute fallback.

A `Fact` keeps availability, evidence kind, an optional composition role, quantity, unit, scope and assumptions as separate fields. A concrete zero is distinct from unknown. An unknown or not-applicable fact carries a reason and no invented value or evidence basis. A symbolic fact uses `Symbol` and `Source` references, which import and run no code. A concrete value is a `Float64`, `Complex128`, `Rational` or an explicit boolean.

`Evidence` records its source, version, domain and, where applicable, the provenance of the work that produced the value. Witnessed evidence, whose `status` is `"witnessed"`, must also name the receipt that witnesses it, which is the record of the computation that produced the value. It names that record by its identifier (`artifact`), together with the content hash of the witnessed subject and the witnessed scope. Evidence witnessed by a verification receipt also names the content hash of the options record that produced the value. A user assertion recorded with a receipt keeps the kind `user_assertion`, and neither the receipt nor schema validation proves the assertion. A `FramedFact` can carry the failure probability that its evidence supports. This does not turn an unknown fact into a bound or borrow a confidence level from a later `Accuracy` criterion.

## Immutability, content hashes and payloads {#immutability-identity-and-payloads}

### Immutability

Nested record fields are frozen. Typed containers copy the collections that the caller supplies, and the public JSON and dict descriptions are independent copies. `revise(...)` and `model_copy(update=...)` validate the changes and leave the original record untouched. Core and evidence records keep their revision ancestry in `parent_id`. The scientific Problem, output and `Accuracy` inputs leave the revision and schema bookkeeping fields out of their public field set.

### Content hash

A record's content hash, `content_id`, is the SHA-256 digest of a compact, sorted-key UTF-8 JSON object that contains `nwqlib.record/1`, the fully qualified record type and the declared fields. Computed IDs are excluded from the digest input, and ordered collections keep their order. Constructors and mapping or JSON loads validate field data and normalize scalars. An existing instance of the exact declared type is reused, and its content hash is computed when requested. A content hash associates data with content. It does not authenticate a source or prove the content of external data. Unit compatibility compares symbol and dimension independently of revision ancestry.

### Exact scalar encodings

| Encoding | Stores |
| --- | --- |
| `Float64` | One finite binary64 value |
| `Complex128` | Two finite binary64 components |
| `Rational` | Exact integers in reduced form with a positive denominator |

Both binary64 encodings turn signed zero into positive zero. A `Rational` is exact, so `1/3` stays distinct from its binary64 approximation. These encodings reject nonfinite values, boolean numbers and coercion from numeric strings.

### Operator and state payloads

When NWQLib accepts an operator or state, it keeps a read-only copy of the numerical data. The JSON description of such an input keeps its content hash and preparation metadata but cannot recreate access to the numerical data. To save and reopen Plans, Runs and Results with the data they need, use the existing Method archive hooks through [Save, load and reanalyze results](../saved_evidence.md).

`InputRef` declares a representation and an `identity` without creating data access.

### Limit records

A `Limit` names its stage, metric, unit, kind, scope and a nonnegative finite value. Each metric has one unit and one kind, and the kind fixes how the limit is compared:

| Kind | Metrics | Compared as |
| --- | --- | --- |
| `capacity_stock` | Memory and stored bytes | A peak |
| `consumption` | Transfer and materialization bytes, CPU seconds, the count metrics (evaluations, shots, jobs, host invocations, host work) and currency | A sum over the work |
| `deadline` | Wall seconds | An elapsed duration |

Byte and count limits require exact integers, including values beyond binary64's exact-integer range. A `Limit` record does not execute, authorize or enforce work. Operation controls and `ExecutionLimits` are enforced by the code that runs the operation.

## Check the record contracts {#current-validation}

Run `python docs/scripts/check_core_records.py` in the environment that [Maintenance](../MAINTENANCE.md) describes, after checking that `python` belongs to that environment and that `nwqlib` imports from your checkout. The script starts a fresh subprocess that, before NWQLib is imported, installs a blocker that records and rejects import attempts. It then runs, on real inputs, the checks of a scientific input, a two-entry operator action, schema and JSON checks, Program binding, resource estimation of the chosen construction, JSON round trips of the chosen block construction and Pauli readout selection. A forbidden import that the code catches must still fail the script. The script checks dependency isolation and current record behavior without vendor SDKs or algorithm execution. It does not validate the accuracy of a scientific method or authenticate external payloads.
