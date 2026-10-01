# Contributing

## What a good change looks like

- It does one thing: one capability or one fix.
- It adds no new dependency unless the [contribution policy](https://pnnl.github.io/NWQLib/FRAMEWORK/) requires one.
- Its user-facing behavior is visible in typed records, validation results, reports and focused tests.
- It passes the checks in [Maintenance](https://pnnl.github.io/NWQLib/MAINTENANCE/) that apply to it.

The package root `nwqlib` exposes the workflow and finite-search operations. Import other public classes and functions from the subpackage that defines them. A Method that submits your own circuit and returns its counts needs a `descriptor` and two hooks, `plan` and `analyze`, as [Run your own circuit](https://pnnl.github.io/NWQLib/own_circuit/) shows.

## Set up, test and open a pull request

[Set up, test and build](https://pnnl.github.io/NWQLib/development/setup/) gives the steps from cloning to opening a pull request. Before changing algorithm behavior, read the [contribution policy](https://pnnl.github.io/NWQLib/FRAMEWORK/), which states the rules for implementations, execution modes, backends, reports and tests. The [code tour](https://pnnl.github.io/NWQLib/CODE_TOUR/) shows where each part of the library lives.

## Environment lock

`docs/ENVIRONMENT_LOCK.txt` records the Python version and the exact package versions of the validated stable CI environment. Apply it with `pip install -c` (and `--build-constraint`), not `-r`, so that platform-specific packages are not forced onto other systems. It changes only together with `.python-version`, after the full CI checks pass on the new versions.

## Docstrings

Public dataclass docstrings use an `Args:` or `Attributes:` section that names every field exactly once. Literal-valued fields list their accepted spellings. Scientific quantities state their units, normalization and shape when these are not obvious. The [documentation standard](https://pnnl.github.io/NWQLib/FRAMEWORK/#documentation-and-commenting-standard) lists what a docstring of a scientific method contains.

## Example notebooks

The notebooks in `examples/` are generated from their percent-format sources in `examples/generators/`. Edit the source, regenerate the notebook and run the freshness check, as [Notebook generation](https://pnnl.github.io/NWQLib/MAINTENANCE/#notebook-generation) describes. Do not edit the notebook JSON, which would leave two copies of the example logic.

## Figures

A generated figure names the input that reproduces it, or is marked as illustrative. Its caption says whether plotted resources describe a full algorithm circuit, an oracle or a subroutine. A synthesis comparison names the method, the state family, the seed and the approximation parameters.

## Numerical results

A mathematical quantity must satisfy its domain before any evidence about it is assessed. A roundoff adjustment needs an explicit, scale-aware tolerance, keeps the raw value and discloses the adjustment. A signed statistical estimator keeps its meaning, and formatting must not silently turn it into a bounded physical quantity. [Mathematical definitions before evidence](https://pnnl.github.io/NWQLib/FRAMEWORK/#mathematical-definitions-before-evidence) gives the full rule.
