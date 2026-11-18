#!/usr/bin/env python3
"""Cheap, stdlib-only checks for repository-wide documentation contracts."""

from __future__ import annotations

import ast
import argparse
from html.parser import HTMLParser
import re
import sys
import tokenize
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
    "nwqlib.operators._pauli": ("PauliGrouping",),
    "nwqlib.operators._fermion": ("FermionTerms",),
    "nwqlib.operators._factorized": ("FactorizedOperatorProduct",),
    "nwqlib.problems.inputs": ("StateInput", "QiskitPreparation"),
    "nwqlib.evidence.scalar_bound": ("ScalarBoundResult",),
    "nwqlib.evidence.verification": ("ProjectedDiagnostics",),
    "nwqlib.backends.connection": ("NativePreparation", "BackendRefresh"),
    "nwqlib.backends.results": ("BackendRunResult",),
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
    "nwqlib.subroutines.qsp.inverse": ("InverseChebyshevPolynomial",),
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


# A label is one to three capitals and a number, optionally hyphenated, dotted or with a letter suffix.
_LABEL = r"(?:[A-Z]{2,3}-[A-Z]\d+|[A-Z]{1,3}\d+[a-z]?(?:\.\d+)?)"
_LABEL_AT = re.compile(rf"(?<![\w.\-#/$\\])({_LABEL})(?![\w(=]|\s*=[^=])")
# The second number of a range may omit its letters.
_LABEL_RANGE = re.compile(rf"({_LABEL})\s*(?:–|-{{1,2}}|\bto\b)\s*(?:[A-Z]{{1,3}}-?[A-Z]?)?(\d+)")
# A range or a slash pair of labels is written without spaces, which separates it from a difference.
_LABEL_CHAIN = re.compile(rf"(?<![\w.\-#/$\\]){_LABEL}(?:\)?(?:–|-{{1,2}}|/)\(?{_LABEL})+(?![\w(=])")
# A label family with two or more letters, or a dotted number, is rarely a formula symbol.
_DISTINCT_LABEL = re.compile(r"[A-Z]{2,3}-?[A-Z]?\d|[A-Z]\d+\.\d")
_DECLARATION = re.compile(r"\blabels?\s+((?:[\s,]|and\b|–|-|\bto\b|" + _LABEL + r")+)")
_EQUATION_TAG = re.compile(r"\\tag\{([^{}]+)\}|\\text\{\(([^(){}]+)\)\}|\s{2,}\((" + _LABEL + r")\)\s*$", re.M)
# An equation, table, section or page word, then any earlier labels of the same list.
_PAPER_WORD = re.compile(
    r"(?:\bEqs?\.?|\bEquations?|\bTables?|\bFigs?\.?|\bFigures?|\bTheorems?|\bLemmas?|\bPropositions?|"
    r"\bCorollar(?:y|ies)|\bAppendi(?:x|ces)|\bSections?|\bSecs?\.?|\bAlgorithms?|\bpp?\.)\s*\(?\s*"
    r"(?:" + _LABEL + r"\)?\s*(?:,|–|-{1,2}|/|\band\b|\bto\b)\s*\(?\s*)*$")
_PAPER_IDENTIFIER = re.compile(r"arXiv:\s*\S|\bdoi:\s*\S|doi\.org/", re.I)
_PARENTHETICAL = re.compile(r"(?<![\w)\]])\(([^()]*)\)")
_CODE_SPAN = re.compile(r"``.*?``|`[^`\n]*`", re.S)
# An inline code span of a Markdown page. A fenced block opens with three backticks and is kept.
_INLINE_CODE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
# A document is cited by a name in capitals, as a Markdown file is named, that is followed by a numbered
# section or item or by a label, directly or after "of" or "of the", or that stands alone at the start of a
# parenthetical.
_DOCUMENT = re.compile(
    r"(?<![\w./-])([A-Z]{4,}|[A-Z]{3,}(?:_[A-Z0-9]{2,})+)\s+"
    rf"(?:(?:sections?|items?)\s+\d|§\s*\d|\d+(?:\.\d+)+,\s*items?\s+\d"
    rf"|(?:of\s+(?:the\s+)?)?({_LABEL})(?![\w(=]))"
    r"|(?<![\w)\]])\(\s*([A-Z]{3,}(?:_[A-Z0-9]{2,})+)\s*[),:;]")

# Names that match the label pattern and cite nothing. Add a formula symbol or an identifier of
# another library here when the check reports it.
_NOT_CITATIONS = frozenset({
    "PCG64", "RK45", "DOP853",   # NumPy bit generator and SciPy integrators
    "MP2",                       # Moller-Plesset energy
    "NRM2",                      # BLAS routine
    "PRX11",                     # journal volume in a reference that also gives its DOI
})


def _prose_segments(path: Path) -> tuple[set[str], list[tuple[int, str]]]:
    """Return the identifiers of a Python file and its comments and string literals with their first lines."""
    names: set[str] = set()
    segments: list[tuple[int, str]] = []
    run: list[tokenize.TokenInfo] = []

    def close_run():
        if run:
            segments.append((run[0].start[0], "\n".join(token.string.lstrip("#") for token in run)))
            run.clear()

    with tokenize.open(path) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type == tokenize.COMMENT:
                if run and token.start[0] != run[-1].start[0] + 1:
                    close_run()
                if "noqa" not in token.string:
                    run.append(token)
                continue
            if token.type == tokenize.NAME:
                names.add(token.string)
            elif token.type == tokenize.STRING or token.type == getattr(tokenize, "FSTRING_MIDDLE", None):
                close_run()
                segments.append((token.start[0], token.string))
    close_run()
    return names, segments


def _declared_labels(text: str) -> set[str]:
    """Return the labels that a text declares in a sentence about its labels or in an equation tag."""
    declared: set[str] = set()

    def add(fragment: str) -> None:
        for first, last in _LABEL_RANGE.findall(fragment):
            prefix = re.match(r"\D+", first).group()
            start = int(re.search(r"\d+", first).group())
            declared.update(f"{prefix}{number}" for number in range(start, int(last) + 1))
        declared.update(re.findall(_LABEL, fragment))

    for match in _DECLARATION.finditer(text):
        add(match.group(1))
    for match in _EQUATION_TAG.finditer(text):
        add(next(group for group in match.groups() if group))
    return declared


def _cited_labels(text: str) -> list[tuple[int, str, bool]]:
    """Return ``(offset, label, paper)`` for each label that a text cites.

    A distinct label is a citation wherever it stands. A single-letter label
    is one only where the text uses it as a reference: in a parenthetical
    that starts or ends with a label, in a range or a slash pair of labels,
    after a document name, or before a colon at the start of the text.
    ``paper`` says that an equation, table, section or page word introduces
    the label or the list it belongs to.
    """
    masked = _CODE_SPAN.sub(lambda match: " " * len(match.group()), text)
    found: dict[int, tuple[str, bool]] = {}

    def paper(offset: int) -> bool:
        return bool(_PAPER_WORD.search(text[max(0, offset - 80):offset]))

    def add(offset: int, label: str, introduced: bool) -> None:
        if label not in _NOT_CITATIONS:
            found[offset] = (label, introduced or found.get(offset, ("", False))[1])

    for match in _LABEL_AT.finditer(text):
        if _DISTINCT_LABEL.match(match.group(1)):
            add(match.start(1), match.group(1), paper(match.start(1)))
    for match in _LABEL_CHAIN.finditer(masked):
        introduced = paper(match.start())
        for label in re.finditer(_LABEL, match.group()):
            add(match.start() + label.start(), label.group(), introduced)
    for match in _PARENTHETICAL.finditer(masked):
        inner = match.group(1)
        items = [item.strip() for item in inner.split(",")]
        # A tuple of symbols such as (H0, S), or one nested in another tuple, is notation.
        if masked[max(0, match.start() - 1):match.start()] == "(" or masked[match.end():match.end() + 1] == ")":
            continue
        if len(items) > 1 and all(re.fullmatch(r"\S{1,12}", item) for item in items) and not all(
                re.fullmatch(_LABEL, item) for item in items):
            continue
        leads = re.match(rf"\s*{_LABEL}(?=\s*(?:$|[,;:/]|–|-{{1,2}}[A-Z]|\s+[a-z]))", inner)
        closes = re.search(rf"[,;/]\s*{_LABEL}\s*$", inner)
        if leads or closes:
            introduced = paper(match.start())
            for label in _LABEL_AT.finditer(inner):
                add(match.start(1) + label.start(1), label.group(1), introduced)
    leading = re.match(rf"\s*(?:[A-Za-z]{{0,2}}(?:\"\"\"|'''|\"|'))?\s*({_LABEL})\s*:", masked)
    if leading:
        add(leading.start(1), leading.group(1), False)
    for match in _DOCUMENT.finditer(masked):
        if match.group(2):
            add(match.start(2), match.group(2), False)
    return [(offset, label, introduced) for offset, (label, introduced) in sorted(found.items())]


def _check_citations_resolve(failures: list[str], paths: list[Path]) -> None:
    """Require every cited label and document name in Python comments and strings to resolve for a reader."""
    files = {}
    identifiers: set[str] = set()
    declaring: dict[str, set[str]] = {}
    page_declaring: dict[str, set[str]] = {}
    for path in paths:
        names, segments = _prose_segments(path)
        declared = _declared_labels("\n".join(segment for _, segment in segments))
        files[path] = (segments, declared)
        identifiers |= names
        for label in declared:
            declaring.setdefault(label, set()).add(path.stem)
    pages = (*REPO_ROOT.glob("*.md"), *(REPO_ROOT / "docs").rglob("*.md"))
    # A document name resolves to a Markdown page of the repository or to an identifier of its code.
    known = identifiers | _NOT_CITATIONS | {page.stem for page in pages}
    # A page declares labels as a module does. Its inline code spans are skipped, so that a page can show
    # the syntax of a declaration without making one.
    for page in pages:
        for label in _declared_labels(_INLINE_CODE.sub(" ", page.read_text(encoding="utf-8"))):
            page_declaring.setdefault(label, set()).add(page.name)
    for path, (segments, declared) in files.items():
        relative = path.relative_to(REPO_ROOT)
        source = path.read_text(encoding="utf-8")
        has_paper = bool(_PAPER_IDENTIFIER.search(source))
        for first_line, segment in segments:

            def line(offset: int) -> int:
                return first_line + segment.count("\n", 0, offset)

            for offset, label, introduced in _cited_labels(segment):
                base = label if label[-1].isdigit() else label[:-1]
                if {label, base} & declared or (introduced and has_paper):
                    continue
                # A symbol that the file defines by an equation is not a citation.
                if re.search(rf"(?<![\w.]){re.escape(label)}\s*:?=[^=]", source):
                    continue
                # A label declared in another module resolves when this file names that module.
                if any(re.search(rf"\b{re.escape(stem)}\b", source)
                       for stem in declaring.get(label, set()) | declaring.get(base, set())):
                    continue
                # A label declared on a documentation page resolves when this file names that page.
                if any(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", source)
                       for name in page_declaring.get(label, set()) | page_declaring.get(base, set())):
                    continue
                failures.append(
                    f"{relative}:{line(offset)}: label {label!r} cites a numbered statement that a reader of the "
                    "repository cannot open. State the relation here, or declare the label where the repository "
                    "states it ('The labels X1–X9 below name ...', or an equation tag) and name that module or "
                    "documentation page here, as in 'sampled_recipes' or 'lchs_derivations.md', or cite the paper "
                    "with its arXiv version or DOI and its own equation number. If it is a formula symbol or an "
                    "identifier, add it to _NOT_CITATIONS in docs/scripts/policy_lint.py.")
            for match in _DOCUMENT.finditer(_CODE_SPAN.sub(lambda span: " " * len(span.group()), segment)):
                name = match.group(1) or match.group(3)
                if name in known or match.group(2) in _NOT_CITATIONS:
                    continue
                failures.append(
                    f"{relative}:{line(match.start())}: {name!r} names a document that is not in the "
                    "repository. State what it says here, or cite a page under docs/ or a paper with its "
                    "arXiv version or DOI. If it is a name from another library, add it to _NOT_CITATIONS "
                    "in docs/scripts/policy_lint.py.")


def check_inaccessible_references(failures: list[str]) -> None:
    """Reject references that a reader of the repository cannot open.

    Python comments and string literals are checked for labels of numbered
    statements and for document names. A label passes when the repository
    declares it (a sentence that names its labels, or an equation tag) in the
    same file, in a module that the file names, or on a Markdown page of the
    repository that the file names with its ``.md`` extension, when an
    equation, table, section or page word introduces it and the file cites a
    paper by arXiv identifier or DOI, when the file defines it by an
    equation, or when ``_NOT_CITATIONS`` lists it. A document name passes
    when it is a Markdown page of the repository or an identifier of its
    code. A single-letter label in running prose is not examined, because a
    formula symbol has the same form.
    """
    _check_citations_resolve(failures, sorted(_python_sources("src", "tests", "docs/scripts", "examples")))


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


_PYTHON_FENCE = re.compile(r"(?ms)^[ \t]*```python[^\n]*\n(.*?)^[ \t]*```[ \t]*$")
_NWQLIB_IMPORT = re.compile(r"^\s*(?:from\s+(nwqlib(?:\.\w+)*)\s+import\b|import\s+(nwqlib(?:\.\w+)*))", re.M)


def check_user_code_imports_no_private_modules(failures: list[str]) -> None:
    """Reject imports of underscore modules of nwqlib in the code users copy.

    Python blocks of the guides and the notebook generators show users how to
    write their own code, and an underscore module can change without notice.
    The maintainer scripts in docs/scripts are not user code and are not checked.
    """
    paths = [*sorted((REPO_ROOT / "docs").rglob("*.md")), *sorted((REPO_ROOT / "examples" / "generators").rglob("*.py"))]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        blocks = ([(match.start(1), match.group(1)) for match in _PYTHON_FENCE.finditer(text)]
                  if path.suffix == ".md" else [(0, text)])
        for offset, code in blocks:
            for match in _NWQLIB_IMPORT.finditer(code):
                module = match.group(1) or match.group(2)
                if any(part.startswith("_") for part in module.split(".")):
                    line = text.count("\n", 0, offset + match.start()) + 1
                    failures.append(f"{path.relative_to(REPO_ROOT)}:{line}: user code imports the underscore "
                                    f"module {module}; import its public name instead")


class _RenderedPage(HTMLParser):
    """Read the anchors of a rendered page, and the links and API entries of its article, without theme navigation."""

    def __init__(self):
        super().__init__()
        self.in_article = False
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.has_api_entry = False

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
        check_user_code_imports_no_private_modules,
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
