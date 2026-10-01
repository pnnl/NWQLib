# Command line

The workflow CLI is available as `nwqlib` or `python -m nwqlib`. Scientific execution, reanalysis, search, save/load and verification remain Python operations. Independent native backend build commands keep their own entry points.

## Discover methods

```sh
python -m nwqlib algorithms
python -m nwqlib algorithms --json
python -m nwqlib card finite_pauli_expectation
python -m nwqlib options adapt_gcim
```

`algorithms` and `card` display actual declared method scope, references and limitations. They do not prove applicability, native support or accuracy. `options` shows the configured Method type's JSON schema, including required scientific fields and current defaults. It does not instantiate a Method with invented inputs or plan a workload.

`--version VERSION` selects the exact named version for `card` or `options`. Missing or ambiguous registrations reject without fallback. `--third-party` explicitly includes installed entry-point metadata. It executes no external factory; unloaded external descriptors and schemas remain unavailable. Builtin discovery reads the same configuration owners as Python exports.

## Check an explicitly selected Method

```sh
python -m nwqlib check-method MODULE:case
```

`MODULE` is your importable module and `case` is its factory returning a `MethodCase`. This imports and executes the user-selected trusted factory. Its `MethodCase` declares actual bounded evaluation, an independent scientific oracle and a wrong-result falsifier. The reference Hadamard Method in `tests/_hadamard_method.py` supplies a complete `case`. It performs one two-qubit acquisition, checks the Y expectation of `|+i>`, and exercises the existing Result archive. See [method authoring](algorithm_protocol.md) for the contract and three independent scientific witnesses.

A selected case or archive hook can execute Python code and declared acquisition; this command is not a sandbox. The checker adds no automatic reference study. A failed or unsupported case exits nonzero. Passing establishes the scoped case, rather than general scientific or backend qualification.

## Python discovery

In Python, discovery keeps the actual immutable registration records. A compact display does not replace them with strings:

```python
from nwqlib import methods

registrations = methods()
for registration in registrations:
    print(registration.source.name, registration.source.version)
```

Use the existing `algorithms`, `card` and `options` commands for formatted declarations and configuration schemas. Discovery does not execute or qualify a Method.

## Read a saved report

Save an actual Result in Python, then inspect it without loading its Method or binary data:

```python
result.save("my-result")
```

```sh
python -m nwqlib report my-result
```

`report PATH` reads `result.json` with the archive owner's JSON reader, which refuses nesting deeper than the JSON parser's recursion limit, duplicate keys and nonfinite numbers. The report also refuses a file that parses but is nested too deeply for the indented JSON encoder that writes the report. Its JSON output keeps the saved selection, scientific fields, observations, receipts and trace, plus the `metadata_validation` entry that lists what was checked. Known common schemas and identities, the canonical Plan identity and the associations of the Result with its observations and of the attempts with their forecast are checked. The joins of observations to their receipts and attempts were checked when the Result was saved and are not repeated. Method-specific Result schema/identity and scientific pair checks require the explicitly chosen loader. No array, native circuit or controller cache is hydrated. Binary payload integrity remains unverified, including when data is missing or changed.

A malformed known record or changed canonical Plan description rejects. Method names in the file remain inert data. Reporting does not replan, analyze, execute, refresh a provider or modify the saved directory. Use the Python Result and Run operations for the corresponding explicitly selected work.
