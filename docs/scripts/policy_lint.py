#!/usr/bin/env python3
"""Cheap, stdlib-only checks for repository-wide documentation contracts."""

from __future__ import annotations

import ast
import argparse
from html.parser import HTMLParser
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BACKTICKED = re.compile(r"`([^`]+)`")


def _python_sources(*relative_roots: str) -> list[Path]:
    files: list[Path] = []
    for root in relative_roots:
        files.extend(sorted((REPO_ROOT / root).rglob("*.py")))
    return files


def _markdown_link_targets(text: str) -> list[str]:
    targets: list[str] = []
    position = 0
    while True:
        start = text.find("](", position)
        if start == -1:
            break
        index = start + 2
        depth = 0
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    targets.append(text[start + 2 : index])
                    position = index + 1
                    break
                depth -= 1
            index += 1
        else:
            break
    return targets


def _markdown_link_path(target: str) -> str:
    target = target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(None, 1)[0]


def _decorator_name(decorator: ast.expr) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any(_decorator_name(decorator) == "dataclass" for decorator in node.decorator_list)


def _base_names(node: ast.ClassDef) -> tuple[str, ...]:
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return tuple(names)


def _local_dataclass_fields(node: ast.ClassDef) -> set[str]:
    fields: set[str] = set()
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            annotation = ast.unparse(item.annotation)
            if "ClassVar" not in annotation:
                fields.add(item.target.id)
    return fields


# Reviewed dataclass owners exposed by package exports, API directives, or
# documented public arguments/returns. These are implementation-qualified names,
# including public values implemented in private modules; this is not an export resolver.
PUBLIC_DATACLASS_OWNERS = {
    "nwqlib._prepared_execution": ("Run", "Prepared", "PreparedHandle"),
    "nwqlib.scientist": ("Comparison", "ComparisonRow"),
    "nwqlib.search": ("Candidate", "SearchResult"),
    "nwqlib.algorithms.qhd.constrained": ("ConstrainedQHDResult",),
    "nwqlib.artifacts": ("ArtifactHandle",),
    "nwqlib.core.analysis": ("RunData",),
    "nwqlib.blocks.lowering": ("LogicalCircuit",),
    "nwqlib.blocks.kernels": ("KernelOutput", "BoundKernel"),
    "nwqlib.blocks.selection": ("SelectedBlock",),
    "nwqlib.reporting.records": ("ReportSection",),
    "nwqlib.operators.df": ("DFConversion", "FactorizedHamiltonian"),
    "nwqlib.operators.inputs": ("PeriodicStencil", "OperatorInput"),
    "nwqlib.operators._pauli": ("PauliTerms", "PauliGrouping"),
    "nwqlib.operators._fermion": ("FermionTerms",),
    "nwqlib.operators._factorized": ("FactorizedOperatorProduct",),
    "nwqlib.problems.inputs": ("StateInput", "QiskitPreparation"),
    "nwqlib.evidence.scalar_bound": ("ScalarBoundResult",),
    "nwqlib.evidence.verification": ("ProjectedDiagnostics",),
    "nwqlib.backends.connection": ("NativePreparation", "BackendRefresh"),
    "nwqlib.backends.results": ("BackendRunResult",),
    "nwqlib.backends.resources": ("SampledBlock",),
    "nwqlib.backends.slurm": ("SlurmStatus",),
    "nwqlib.io.materialization": ("QasmMaterialization",),
    "nwqlib.io.streaming": ("QasmPrefix",),
    "nwqlib.algorithms.registry": ("Registration",),
    "nwqlib.algorithms.authoring": ("MethodCase",),
    "nwqlib.subroutines.fermionic_pool": ("FermionicGenerator",),
    "nwqlib.subroutines.pauli_decomposition": ("PauliTerm", "PauliDecomposition"),
    "nwqlib.subroutines.hamiltonian_evolution.pauli_ir": ("PauliEvolutionTerm", "PauliEvolutionBlock"),
    "nwqlib.subroutines.lcu.core": ("LCUData", "LCUCircuit"),
    "nwqlib.subroutines.trotterization.error_budget": ("TrotterStepSelection",),
    "nwqlib.subroutines.state_preparation.mps": ("MPSDecomposition", "MPSCompressionAnalysis"),
    "nwqlib.subroutines.state_preparation.mps_circuit": ("MPSCircuitStatePreparation",),
    "nwqlib.subroutines.state_preparation.direct": ("DirectStatePreparation",),
    "nwqlib.subroutines.qpe.coherent": ("CoherentQPECircuit",),
    "nwqlib.subroutines.block_encoding.core": ("BlockEncoding", "BlockEncodingPlan"),
    "nwqlib.subroutines.block_encoding.banded": ("BandSpecification",),
    "nwqlib.subroutines.qsp.evolution": ("JacobiAngerExpansion",),
    "nwqlib.subroutines.qsp.phases": ("SymmetricQSPPhases",),
    "nwqlib.subroutines.qsp.inverse": ("InverseChebyshevFit",),
    "nwqlib.subroutines.qsp.shortcut": ("KernelReflectionPolynomial",),
    "nwqlib.algorithms.lchs.provider_config": ("ProviderConfig", "ResolvedProviderConfig"),
    "nwqlib.algorithms.lchs.providers": ("LCHSKernelProvider", "LCHSProblemContext", "LCHSQuadraturePlan", "LCHSQuadratureProvider", "LCHSPairCompatibility", "LCHSCoefficientPlan"),
    "nwqlib.algorithms.lchs.time_independent_terms": ("LCHSQuadratureData", "LCHSProductFormulaSelectData", "LCHSProductFormulaNodes"),
    "nwqlib.algorithms.lchs.select_synthesis": ("LCHSProductFormulaSelectPlan",),
    "nwqlib.algorithms.gcim.chemistry": ("GCIMChemistryProblemData",),
}


def _docstring_section_fields(node: ast.ClassDef) -> list[tuple[str, int]]:
    docstring = ast.get_docstring(node)
    if not docstring:
        return []
    names: list[tuple[str, int]] = []
    active = False
    start_line = node.body[0].lineno
    for offset, line in enumerate(docstring.splitlines(), start=1):
        stripped = line.strip()
        if stripped in {"Args:", "Attributes:"}:
            active = True
            continue
        if active and stripped.endswith(":") and not line.startswith((" ", "\t")):
            active = False
            continue
        if active:
            match = re.match(r"\s{4,}([A-Za-z_][A-Za-z0-9_]*):", line)
            if match:
                names.append((match.group(1), start_line + offset))
    return names


def _dataclass_index():
    index = {}
    for path in _python_sources("src"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = ".".join(path.relative_to(REPO_ROOT / "src").with_suffix("").parts)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _is_dataclass(node):
                index[f"{module}.{node.name}"] = (path, node)
    return index


def _inherited_fields(owner, index, seen=None):
    seen = set() if seen is None else seen
    if owner in seen or owner not in index:
        return set()
    seen.add(owner)
    _, node = index[owner]
    module = owner.rpartition(".")[0]
    inherited = set()
    for base in _base_names(node):
        parent = f"{module}.{base}"
        if parent in index:
            inherited.update(_local_dataclass_fields(index[parent][1]))
            inherited.update(_inherited_fields(parent, index, seen))
    return inherited


def check_record_docstring_fields(failures: list[str]) -> None:
    index = _dataclass_index()
    for module, names in PUBLIC_DATACLASS_OWNERS.items():
        for name in names:
            owner = f"{module}.{name}"
            if owner not in index:
                failures.append(f"public dataclass owner {owner} does not resolve")
                continue
            path, node = index[owner]
            relative = path.relative_to(REPO_ROOT)
            documented = _docstring_section_fields(node)
            local = _local_dataclass_fields(node)
            required = {name for name in local if not name.startswith("_")}
            allowed = local | _inherited_fields(owner, index)
            seen = set()
            for field_name, lineno in documented:
                if field_name in seen:
                    failures.append(f"{relative}:{lineno}: {owner} documents duplicate field {field_name!r}")
                seen.add(field_name)
                if field_name not in allowed:
                    failures.append(f"{relative}:{lineno}: {owner} documents unknown field {field_name!r}")
            for field_name in sorted(required - seen):
                failures.append(f"{relative}:{node.lineno}: {owner} field {field_name!r} is missing from its docstring section")


_INACCESSIBLE_REFERENCES = re.compile(
    r"\bJob\s+\d+\b|\bprincipal['’]s\s+committed\b|\bparking[-]lot\s+D6\b|"
    r"\bAdapt[-]GCiM\s+job\s+review\b|\bassigned\s+redesign\b"
)


def check_inaccessible_references(failures: list[str]) -> None:
    """Reject specific inaccessible references without banning Plan or backend job terms."""
    paths = {*_python_sources("src", "tests", "docs/scripts", "examples"),
             *REPO_ROOT.glob("*.md"), *(REPO_ROOT / "docs").rglob("*.md"),
             REPO_ROOT / "pyproject.toml"}
    for path in sorted(paths):
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _INACCESSIBLE_REFERENCES.search(line):
                failures.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: replace inaccessible reference with its technical reason or public source")


def _identifier_corpus() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in _python_sources("src", "docs/scripts", "examples")
    )


def _implementation_map_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    sections = re.findall(
        r"(?ims)^#{2,3} [^\n]*(?:implementation|code)[^\n]*(?:map|owners|witnesses)\n"
        r"(.*?)(?=^#{1,3} |\Z)", text,
    )
    return _strip_fenced_blocks("\n".join(sections))


def _resolves_code_path(token: str, *, algorithm: str | None) -> bool:
    if "*" in token:
        return any(REPO_ROOT.glob(token)) or any((REPO_ROOT / "src" / "nwqlib").glob(token))
    candidates = [REPO_ROOT / token]
    if algorithm is not None:
        candidates.append(REPO_ROOT / "src" / "nwqlib" / "algorithms" / algorithm / token)
    candidates.append(REPO_ROOT / "src" / "nwqlib" / token)
    if token.endswith(".py") and "/" not in token:
        candidates.extend((REPO_ROOT / "src" / "nwqlib").rglob(token))
    return any(candidate.exists() for candidate in candidates)


def _strip_fenced_blocks(text: str) -> str:
    """Fenced blocks break inline backtick-span pairing; drop them before scanning."""
    return re.sub(r"(?ms)^```.*?^```[ \t]*$", "", text)


def _check_backticked_reference(
    failures: list[str],
    *,
    relative: Path,
    token: str,
    corpus: str,
    algorithm: str | None = None,
) -> None:
    if token.strip() != token or any(char.isspace() for char in token):
        return
    if token.startswith("_") and "/" not in token and not token.endswith(".py"):
        return
    if (token.endswith((".py", "/")) or "*" in token
            or token.startswith(("src/", "docs/", "tests/", "examples/"))):
        if not _resolves_code_path(token, algorithm=algorithm):
            failures.append(f"{relative}: backticked path {token!r} does not resolve")
        return
    if _IDENTIFIER.match(token):
        if not re.search(rf"\b{re.escape(token)}\b", corpus):
            failures.append(f"{relative}: backticked identifier {token!r} does not exist in code")


def _defines_name(path: Path, name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    )


def check_backticked_implementation_references(failures: list[str]) -> None:
    corpus = _identifier_corpus()
    mathematics = REPO_ROOT / "docs" / "mathematics.md"
    if mathematics.exists():
        # A location `path.py::Qualified.name` resolves when the file defines its last name component.
        relative = mathematics.relative_to(REPO_ROOT)
        for token in _BACKTICKED.findall(
            _strip_fenced_blocks(mathematics.read_text(encoding="utf-8"))
        ):
            path, separator, qualified = token.partition("::")
            if not separator:
                _check_backticked_reference(failures, relative=relative, token=token, corpus=corpus)
            elif not path.endswith(".py") or not (REPO_ROOT / path).is_file():
                failures.append(f"{relative}: backticked location {token!r} names no Python file")
            elif not _defines_name(REPO_ROOT / path, qualified.rpartition(".")[2]):
                failures.append(f"{relative}: backticked location {token!r} is not defined in its file")
    code_tour = REPO_ROOT / "docs" / "CODE_TOUR.md"
    if code_tour.exists():
        relative = code_tour.relative_to(REPO_ROOT)
        for token in _BACKTICKED.findall(
            _strip_fenced_blocks(code_tour.read_text(encoding="utf-8"))
        ):
            _check_backticked_reference(failures, relative=relative, token=token, corpus=corpus)
    for guide in sorted((REPO_ROOT / "docs" / "algorithms").glob("*.md")):
        relative = guide.relative_to(REPO_ROOT)
        algorithm = guide.stem
        for token in _BACKTICKED.findall(_implementation_map_text(guide)):
            _check_backticked_reference(
                failures,
                relative=relative,
                token=token,
                corpus=corpus,
                algorithm=algorithm,
            )
    constants = REPO_ROOT / "docs" / "ENGINEERING_CONSTANTS.md"
    if constants.exists():
        # A prose mention of a deleted constant is not a current definition.
        declarations = {
            node.id
            for path in _python_sources("src", "examples")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        for lineno, line in enumerate(constants.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("|"):
                continue
            for name in re.findall(r"`([A-Z][A-Z_0-9]*_[A-Z_0-9]+)`", line.split("|")[1]):
                if name not in declarations:
                    failures.append(
                        f"docs/ENGINEERING_CONSTANTS.md:{lineno}: constant {name!r} "
                        "has no current source or example definition"
                    )


def check_internal_markdown_links_resolve(failures: list[str]) -> None:
    paths = sorted((REPO_ROOT / "docs").rglob("*.md"))
    paths.extend(REPO_ROOT / name for name in ("README.md", "CONTRIBUTING.md", "examples/README.md"))
    for path in paths:
        if not path.is_file():
            continue
        text = _strip_fenced_blocks(path.read_text(encoding="utf-8"))
        for target in _markdown_link_targets(text):
            link = urlsplit(_markdown_link_path(target))
            base = unquote(link.path)
            if link.scheme or link.netloc or not base or Path(base).suffix.lower() == ".pdf":
                continue
            resolved = (
                REPO_ROOT / base.removeprefix("/")
                if base.startswith("/")
                else path.parent / base
            )
            if not resolved.exists():
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}: markdown link target "
                    f"{target!r} does not resolve to an existing file"
                )


_MATH_FENCE = re.compile(r"(?ms)^[ \t]*```math[ \t]*\n(.*?)^[ \t]*```[ \t]*$")


def check_math_fences_open_no_html_tags(failures: list[str]) -> None:
    """MkDocs passes ```math content to the page unescaped, so `<` before a letter would start an HTML tag."""
    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in _MATH_FENCE.finditer(text):
            if re.search(r"<[A-Za-z]", match.group(1)):
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}:{text.count(chr(10), 0, match.start()) + 1}: "
                    "```math content has '<' before a letter, which MkDocs passes unescaped as the start "
                    "of an HTML tag. Write \\lt or use a $$ display."
                )


class _RenderedPage(HTMLParser):
    """Read document content, excluding theme navigation and external assets."""

    def __init__(self):
        super().__init__()
        self.in_article = False
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.has_api_entry = False
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.in_article = True
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if self.in_article:
            if tag == "a" and attrs.get("href"):
                self.links.append(attrs["href"])
            if {"doc-class", "doc-function"}.intersection((attrs.get("class") or "").split()):
                self.has_api_entry = True

    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False

    def handle_data(self, data):
        if self.in_article:
            self.text.append(data)


def check_rendered_documentation(failures: list[str], site_dir: Path) -> None:
    """Check the existing MkDocs output without importing or executing package code."""
    site_dir = site_dir.resolve()
    pages = {}
    for path in sorted(site_dir.rglob("*.html")):
        page = _RenderedPage()
        page.feed(path.read_text(encoding="utf-8"))
        pages[path] = page
    if not pages:
        failures.append(f"{site_dir}: no rendered HTML; build the documentation first")
        return
    for path, page in pages.items():
        if _INACCESSIBLE_REFERENCES.search("".join(page.text)):
            failures.append(f"{path.relative_to(site_dir)}: rendered content contains an inaccessible reference")
        for href in sorted(set(page.links)):
            link = urlsplit(href)
            if link.scheme or link.netloc:
                continue
            base = unquote(link.path)
            target = ((site_dir / base.lstrip("/")) if base.startswith("/")
                      else (path.parent / base if base else path)).resolve()
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                failures.append(f"{path.relative_to(site_dir)}: link {href!r} has no rendered target")
            elif (link.fragment and target in pages
                  and unquote(link.fragment) not in pages[target].ids):
                failures.append(f"{path.relative_to(site_dir)}: link {href!r} has no rendered anchor")
    for source in sorted((REPO_ROOT / "docs" / "api").rglob("*.md")):
        directives = re.findall(
            r"(?m)^:::\s+(\S+)", _strip_fenced_blocks(source.read_text(encoding="utf-8"))
        )
        if not directives:
            continue
        relative = source.relative_to(REPO_ROOT / "docs")
        output = site_dir / relative.with_suffix(".html")
        if output not in pages:
            output = site_dir / relative.with_suffix("") / "index.html"
        page = pages.get(output)
        if page is None:
            failures.append(f"{source.relative_to(REPO_ROOT)}: API page was not rendered")
            continue
        for target in directives:
            if target not in page.ids:
                failures.append(f"{source.relative_to(REPO_ROOT)}: API entry {target!r} was not rendered")
        if not page.has_api_entry:
            failures.append(
                f"{source.relative_to(REPO_ROOT)}: API page has no class or function documentation; "
                "point directives at the defining modules or objects"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--site-dir", type=Path, help="also check an already built MkDocs site")
    args = parser.parse_args()
    failures: list[str] = []
    checks = (
        check_record_docstring_fields,
        check_inaccessible_references,
        check_backticked_implementation_references,
        check_internal_markdown_links_resolve,
        check_math_fences_open_no_html_tags,
    )
    print("policy checks: " + ", ".join(check.__name__ for check in checks))
    for check in checks:
        check(failures)
    if args.site_dir is not None:
        print("rendered check: check_rendered_documentation")
        check_rendered_documentation(failures, args.site_dir)
    if failures:
        for failure in failures:
            print(f"POLICY FAIL: {failure}", file=sys.stderr)
        return 1
    print("policy lint: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
