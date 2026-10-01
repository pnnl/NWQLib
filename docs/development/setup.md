# Set up, test and build

These steps give you a development checkout of NWQLib in which the tests, lint and documentation build run as they do in CI. [Maintenance](../MAINTENANCE.md) lists the further checks for each kind of change, and the [contribution policy](../FRAMEWORK.md) states the rules a change must meet.

1. Clone the repository. To open a pull request later, fork it on GitHub first and clone your fork.

    ```bash
    git clone https://github.com/pnnl/NWQLib.git
    cd NWQLib
    ```

2. Create an environment with Python 3.12 or later. The stable CI environment uses the version in `.python-version`. With conda:

    ```bash
    conda create -n nwqlib "python=$(cat .python-version)"
    conda activate nwqlib
    ```

3. Install NWQLib in editable mode with the extras the test suite uses, constrained to the package versions of the stable CI environment in `docs/ENVIRONMENT_LOCK.txt`. This is the install of the stable Full CI job without its coverage plugin. The last argument installs `scikit_tt`, which the MPS state-preparation route needs and which is not on PyPI, at the commit the lock file records.

    ```bash
    python -m pip install --upgrade -c docs/ENVIRONMENT_LOCK.txt pip
    python -m pip install -c docs/ENVIRONMENT_LOCK.txt \
        --build-constraint docs/ENVIRONMENT_LOCK.txt \
        -e ".[dev,aer,qasm,docs,notebook,chemistry,ibm,ionq]" \
        "$(sed -n '/^scikit_tt @ /p' docs/ENVIRONMENT_LOCK.txt)"
    python -m pip check
    ```

    `dev` installs pytest, pytest-xdist, Ruff and jsonschema. `aer` installs Qiskit, which test collection needs, and Qiskit Aer. `qasm`, `docs` and `notebook` add the OpenQASM parser and importer, MkDocs, and the Jupyter tools that the notebook tests use. `chemistry`, `ibm` and `ionq` add PySCF and OpenFermion and the IBM Runtime and IonQ SDKs, whose tests run without credentials. The tests for Nexus, the NWQEC compiler and QDK are skipped when their packages are missing. [Support and external-data validation](../MAINTENANCE.md#support-and-external-data-validation) describes both CI environments.

4. Check that `nwqlib` imports from your checkout. The printed path ends in `src/nwqlib/__init__.py` inside the clone.

    ```bash
    python -c "import nwqlib; print(nwqlib.__file__)"
    ```

5. Run the tests. `-n auto` runs them in parallel workers, and `OMP_NUM_THREADS=1` stops each worker from starting its own BLAS thread pool. The full suite also executes the example notebooks.

    ```bash
    OMP_NUM_THREADS=1 python -m pytest -n auto
    python -m pytest tests/test_shared_contracts.py   # one file
    ```

6. Run the linter.

    ```bash
    ruff check .
    ```

7. Build the documentation and check it. The build writes the site to `site/` and fails on any warning. The policy lint then checks the documentation rules in the sources, and the API entries, links and anchors of the built pages.

    ```bash
    python -m mkdocs build --strict
    python docs/scripts/policy_lint.py --site-dir site
    ```

8. If you changed an example, edit its source in `examples/generators/` and rebuild the notebook. `NAME` is the source file name without `.py`, for example `qls_linear_system_intro`. `--execute` runs the notebook and stores its outputs, and `--check` reports every notebook that differs from its source.

    ```bash
    python examples/generators/build_notebooks.py NAME --execute
    python examples/generators/build_notebooks.py --check
    ```

9. Open a pull request on [GitHub](https://github.com/pnnl/NWQLib). In its description, say what the change does and which checks you ran, with their results. If you propose a check that is expensive or scales poorly, give its cost so that a maintainer can decide whether to add it.
