# Use the command line

<a id="command-line"></a>The `nwqlib` command, also available as `python -m nwqlib`, lists the Methods, shows each Method's scope and parameters, checks a Method you wrote, and prints a saved Result. Solving, reanalysis, search, saving, loading and verification are Python operations. The NWQ-Sim runner has its own [build command](nwqsim.md#building-a-selected-nwq-sim-runner).

## Discover methods

```sh
python -m nwqlib algorithms
```

```text
adapt_gcim@4: eigenvalue
chebyshev_lanczos@5: eigenvalue
finite_pauli_expectation@2: normalized_expectation, quadratic_form
fixed_gcim@4: eigenvalue
lchs@3: solution, state_vector, norm_squared, quadratic_form, normalized_expectation, samples
qcels@1: eigenphase, eigenvalue
qhd@2: optimization_candidate
qls@2: solution, state_vector, norm_squared, quadratic_form, normalized_expectation, samples
rfe@2: eigenphase, eigenvalue
rwpe@2: eigenphase, eigenvalue
spe@2: eigenphase, eigenvalue
Declared scope only; discovery does not establish scientific or backend qualification.
```

Each line gives a Method name, its version after `@`, and the outputs it can return ([Choose a problem and output](problems.md#outputs)). `card` prints one Method's declared problems, outputs, input forms, limitations and references as JSON:

```sh
python -m nwqlib card qls
```

```text
{
  ...
  "descriptor": {
    ...
    "problem_families": [
      "linear_system"
    ],
    ...
    "access_families": [
      "dense",
      "pauli",
      "periodic_stencil",
      "bound_block_encoding"
    ],
    ...
    "limitations": [
      "complete propagated physical error and native rounding remain unknown",
      "shortcut supplies a unit direction; classical norm models do not supply physical x"
    ],
    "references": [
      {
        ...
        "reference": "selected inverse Chebyshev QSVT and Dalzell arXiv:2406.12086v2 kernel-reflection methods",
        ...
```

`algorithms --json` prints every card as JSON. `options` prints the JSON schema of a Method's configuration, including required scientific fields and current defaults:

```sh
python -m nwqlib algorithms --json
python -m nwqlib options adapt_gcim
```

These commands show what each Method declares. They do not show that a Method applies to your problem, runs on a given backend or reaches a given accuracy. `options` creates no Method and plans no workload.

`--version VERSION` selects an exact Method version for `card` or `options`, and an unknown or ambiguous name is an error. `--third-party` also lists the Methods that installed packages declare through entry points, without running their code. Their scope, cards and schemas are therefore unavailable from these commands.

## Check a Method you wrote {#check-an-explicitly-selected-method}

```sh
python -m nwqlib check-method MODULE:case
```

`MODULE` is your importable module, and `case` is a function in it that returns a `MethodCase`. The `MethodCase` holds your Method, a Problem for the check and three functions. `evaluate` runs the Method, `accepts` tests the Result against an independently known answer, and `invalid_result` makes a wrong Result for the same Plan. `check-method` then checks the following, as [Add a method](algorithm_protocol.md#author-cases-and-their-limits) defines:

1. The Result that `evaluate` returns satisfies `accepts`, the Method's error model matches its Plan, and the Method names the Result type it returns.
2. The wrong Result from `invalid_result` is rejected by the Method's own check of the Plan and Result, both in memory and when written into a saved Result.
3. The correct Result saves, reloads unchanged and still satisfies `accepts`.

The [reference Hadamard Method](https://github.com/pnnl/NWQLib/blob/main/tests/_hadamard_method.py) supplies a complete `case`. It runs one two-qubit circuit, checks that the Y expectation of `|+i>` is 1, and saves and reloads the Result. On it, `check-method` prints `"status": "CONFORMANT"` with the scope of the check.

`check-method` imports and runs the case you name, including any circuit execution it declares, so it is not a sandbox. It adds no reference study of its own. A failed or unsupported case exits with a nonzero status. A pass establishes this one case only, not general scientific accuracy or backend support.

## Python discovery

In Python, `methods()` returns the registration record of each built-in Method:

```python
from nwqlib import methods

registrations = methods()
for registration in registrations:
    print(registration.source.name, registration.source.version)
```

For formatted declarations and configuration schemas, use the `algorithms`, `card` and `options` commands. Discovery neither runs nor qualifies a Method.

## Read a saved report

Save a Result in Python, then print it without loading its Method or binary data:

```python
result.save("my-result")
```

```sh
python -m nwqlib report my-result
```

The JSON output holds the saved selection, scientific fields, observations, preparation records and trace. `report` checks the shared format and returns this metadata. Method names stay text. Use `nwqlib.load_result(path, method=...)` to reconstruct a Result and access its saved arrays and circuits.
