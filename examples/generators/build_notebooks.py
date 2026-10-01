"""Build the example notebooks from their percent-format sources.

Each ``examples/generators/NAME.py`` becomes ``examples/NAME.ipynb``. A line
``# %% [markdown]`` starts a markdown cell whose ``# ``-prefixed lines become
text, and ``# %%`` starts a code cell. ``# %% jupyter={"source_hidden": true}``
starts a code cell whose source JupyterLab shows collapsed, for display
helpers that a reader need not read. The notebook stores this as the nbformat cell
metadata ``jupyter.source_hidden``. A module docstring before the first marker
is omitted.

A notebook whose cells already match its source keeps its stored outputs.
A changed source replaces the notebook with an output-free version, which
``--execute`` then runs in a Jupyter kernel started from this interpreter.

    python examples/generators/build_notebooks.py                  # write changed English notebooks
    python examples/generators/build_notebooks.py --execute        # also run and store their outputs
    python examples/generators/build_notebooks.py --check          # report notebooks that differ from their sources
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

# Marker line -> (cell type, whether Jupyter shows the source collapsed).
MARKERS = {
    "# %% [markdown]": ("markdown", False),
    "# %%": ("code", False),
    '# %% jupyter={"source_hidden": true}': ("code", True),
}
ASSET_LINK = re.compile(r"\]\((generators/assets/[^)\s]+)\)")


def cells(source):
    """Split percent-format source into ``(cell_type, hidden, text)`` triples."""
    module = ast.parse(source)
    lines = source.splitlines(keepends=True)
    first = module.body[0] if module.body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        lines = lines[first.end_lineno:]
    result, (kind, hidden), buffer = [], MARKERS["# %%"], []

    def flush():
        if kind == "markdown":
            rows = (line.rstrip("\n") for line in buffer)
            text = "\n".join(row[2:] if row.startswith("# ") else "" if row == "#" else row for row in rows)
        else:
            text = "".join(buffer)
        if text.strip():
            result.append((kind, hidden, text.strip("\n")))

    for line in lines:
        marker = MARKERS.get(line.rstrip())
        if marker is None:
            buffer.append(line)
            continue
        flush()
        (kind, hidden), buffer = marker, []
    flush()
    return result


def notebook(source, generated_from, generator):
    """Return the output-free notebook for one source file."""
    notebook_cells = []
    for index, (kind, hidden, text) in enumerate(cells(source.read_text(encoding="utf-8"))):
        cell = {"cell_type": kind, "id": f"cell-{index:03d}",
                "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook_cells.append(cell)
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "nwqlib": {"generated_from": generated_from, "generator": generator},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def without_outputs(document):
    """Keep what the source determines: cell ids, types, collapsed sources, text and notebook identity."""
    return ([(cell["id"], cell["cell_type"], cell["metadata"].get("jupyter"), "".join(cell["source"]))
             for cell in document["cells"]],
            document["metadata"].get("nwqlib"), document["metadata"].get("kernelspec"))


def write(path, document):
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def execute(path):
    """Run every cell in a kernel bound to this interpreter and store the outputs."""
    import nbformat
    from nbclient import NotebookClient

    document = nbformat.read(path, as_version=4)
    client = NotebookClient(document, timeout=1800, record_timing=False,
                            resources={"metadata": {"path": str(path.parent)}})
    manager = client.create_kernel_manager()
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client.execute()
    for cell in document.cells:
        cell.metadata.pop("execution", None)
    write(path, json.loads(nbformat.writes(document)))


def build(output, names, *, check, run):
    """Write, check or execute every notebook in one folder and return the stale paths."""
    generator = Path(os.path.relpath(Path(__file__).resolve(), output.resolve())).as_posix()
    sources = [path for path in sorted((output / "generators").glob("*.py")) if path.name != Path(__file__).name]
    expected = {output / f"{source.stem}.ipynb" for source in sources}
    stale = []
    for source in sources:
        if names and source.stem not in names:
            continue
        for link in ASSET_LINK.findall(source.read_text(encoding="utf-8")):
            if not (output / link).is_file():
                raise FileNotFoundError(f"{source.name} links to missing {output / link}")
        target = output / f"{source.stem}.ipynb"
        fresh = notebook(source, source.relative_to(output).as_posix(), generator)
        current = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
        changed = current is None or without_outputs(current) != without_outputs(fresh)
        if check:
            stale += [target] if changed else []
            continue
        if changed:
            write(target, fresh)
            print("wrote", target)
        if run:
            print("executing", target, flush=True)
            execute(target)
    # Remove only notebooks that this builder generated from a source that no longer exists.
    for path in sorted(set(output.glob("*.ipynb")) - expected):
        if json.loads(path.read_text(encoding="utf-8"))["metadata"].get("nwqlib", {}).get("generator") == generator:
            if check:
                stale.append(path)
            else:
                path.unlink()
                print("removed", path)
    return stale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="notebook names to process, default all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report notebooks that differ from their sources")
    mode.add_argument("--execute", action="store_true", help="run every notebook and store its outputs")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    stale = build(root / "examples", set(args.names), check=args.check, run=args.execute)
    if stale:
        raise SystemExit("Notebooks differ from their sources: " + ", ".join(map(str, stale)))


if __name__ == "__main__":
    main()
