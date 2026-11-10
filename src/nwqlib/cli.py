"""List and describe the Methods, check a Method you wrote, and read the report of a saved Result."""

import argparse
from importlib import import_module
import inspect
import json
import re
from typing import Literal, get_args, get_origin


def _registrations(external):
    from nwqlib.algorithms.registry import (
        AlgorithmRegistry,
        builtin_registrations,
        third_party_registrations,
    )

    rows = builtin_registrations()
    if external:
        rows += third_party_registrations()
    return AlgorithmRegistry(rows).discover()


def _selected(args):
    matches = tuple(
        row
        for row in _registrations(args.third_party)
        if row.source.name == args.method
        and (args.version is None or row.source.version == args.version)
    )
    if len(matches) != 1:
        raise ValueError(
            "select one known method/version from 'algorithms'; no implicit version fallback"
        )
    return matches[0]


def _trusted_factory(reference):
    """Import only the factory, in a Method author's module, that the command-line user names."""
    module, separator, name = reference.partition(":")
    if (
        not separator
        or not all(part.isidentifier() for part in module.split("."))
        or not name.isidentifier()
    ):
        raise ValueError("trusted author case must be an explicit module:factory")
    loaded = import_module(module)
    # Only a missing name is converted here. The selected factory runs later,
    # outside this lookup, so its own AttributeError is not reported as a name.
    try:
        factory = getattr(loaded, name)
    except AttributeError:
        raise LookupError(f"module {module!r} has no author factory {name!r}") from None
    if not callable(factory):
        raise TypeError("the explicitly selected author factory must be callable")
    return factory


def _options_table(registration):
    """Return the options of a registered Method as a table, built from its fields and docstrings.

    Each option has one line: its name, its default as Python source
    (`required` when it has none), the names and option records it accepts,
    and the first sentence of its docstring entry. Under an option that
    takes option records, each record's own options follow, indented. No
    Method is constructed.
    """
    from nwqlib.algorithms.registry import _configured_type

    method = _configured_type(registration)
    rows = [("option", "default", "accepted names and records; description")]
    _option_rows(method, "", rows)
    widths = [max(len(row[column]) for row in rows) for column in (0, 1)]
    source = registration.source
    return "\n".join((f"{method.__name__} options ({source.name}@{source.version}):",
                      *(f"{name:<{widths[0]}}  {default:<{widths[1]}}  {text}".rstrip()
                        for name, default, text in rows)))


def _option_rows(record_type, indent, rows):
    """Append a ``(name, default, text)`` row per option of ``record_type``, and the options of its records."""
    from nwqlib.core.records import _ConstructorDisplay, _argument_text

    descriptions = _field_descriptions(record_type)
    for name, info in record_type.model_fields.items():
        if name in ("schema_version", "parent_id", "kind"):
            continue
        if info.is_required():
            default = "required"
        else:
            value = info.get_default(call_default_factory=True)
            default = _argument_text(value)
        names, records = [], []
        annotations = [info.annotation]
        while annotations:
            annotation = annotations.pop(0)
            if get_origin(annotation) is Literal:
                names.extend(repr(choice) for choice in get_args(annotation))
            elif isinstance(annotation, type) and issubclass(annotation, _ConstructorDisplay):
                records.append(annotation)
            else:
                annotations.extend(get_args(annotation))
        accepted = ", ".join((*names, *(f"{record.__name__}(...)" for record in records)))
        rows.append((indent + name, default,
                     " ".join(filter(None, (accepted and f"Accepts {accepted}.", descriptions.get(name))))))
        for record in records:
            rows.append((f"{indent}  {record.__name__}(...):", "", ""))
            _option_rows(record, indent + "    ", rows)


def _field_descriptions(record_type):
    """Return the first sentence of each field's entry under "Attributes:" in the docstrings of the type and its bases.

    Links and backquotes are removed. Most entries open with the default,
    which the table shows in its own column, so a first sentence that starts
    with "Default" or "Required" is followed by the second, and one that only
    states the default, such as "Default `1`.", is replaced by it.
    """
    found = {}
    for owner in record_type.__mro__:
        lines = inspect.cleandoc(owner.__dict__.get("__doc__") or "").splitlines()
        if "Attributes:" not in lines:
            continue
        entries = {}
        for line in lines[lines.index("Attributes:") + 1:]:
            if line and not line.startswith(" "):
                break
            entry = re.fullmatch(r" {4}(\w+): (.*)", line)
            if entry:
                entries[entry[1]] = [entry[2]]
            elif entries and line.strip():
                entries[list(entries)[-1]].append(line.strip())
        for name, parts in entries.items():
            text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", " ".join(parts)).replace("`", "")
            sentences = re.split(r"(?<=\.)\s+(?=[A-Z\"])", text)
            if re.fullmatch(r"(Default \S+|Required)\.", sentences[0]):
                sentences = sentences[1:] or sentences
            elif sentences[0].startswith(("Default", "Required")):
                sentences = [" ".join(sentences[:2])]
            found.setdefault(name, sentences[0])
    return found


def main(argv=None):
    """Run one `nwqlib` command.

    The commands list or describe the Methods, read a saved report, or run
    the check of a trusted Method that the user names.
    """
    parser = argparse.ArgumentParser(
        prog="nwqlib",
        allow_abbrev=False,
        description=__doc__,
        epilog="check-method executes the explicitly selected trusted case, including any declared acquisition. It is not a sandbox.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("algorithms", help="read declared method inventory", allow_abbrev=False)
    listing.add_argument(
        "--third-party", action="store_true", help="include inert installed entry-point metadata"
    )
    listing.add_argument("--json", action="store_true")
    for name, help_text in (
        ("card", "read a Method's declared scope, references and limitations"),
        ("options", "read a Method's options, their defaults and accepted values, and its required inputs"),
    ):
        command = commands.add_parser(
            name, help=help_text, description=help_text, allow_abbrev=False
        )
        command.add_argument("method")
        command.add_argument("--version")
        command.add_argument("--third-party", action="store_true")
        if name == "options":
            command.add_argument("--json", action="store_true",
                                 help="print the JSON schema of the Method's configuration instead")
    check = commands.add_parser(
        "check-method", help="execute one explicitly selected bounded author case", allow_abbrev=False
    )
    check.add_argument("factory", help="trusted module:factory returning MethodCase")
    report = commands.add_parser(
        "report", help="read saved Result metadata without loading Method code or binary data", allow_abbrev=False
    )
    report.add_argument("path")
    args = parser.parse_args(argv)
    try:
        if args.command == "algorithms":
            from nwqlib.algorithms.registry import algorithm_card

            rows = _registrations(args.third_party)
            if args.json:
                print(json.dumps([algorithm_card(row) for row in rows], allow_nan=False, indent=2))
            else:
                for row in rows:
                    scope = (
                        "external descriptor unavailable until trusted selection"
                        if row.descriptor is None
                        else ", ".join(row.descriptor.output_families)
                    )
                    print(f"{row.source.name}@{row.source.version}: {scope}")
                print(
                    "Declared scope only; discovery does not establish scientific or backend qualification."
                )
        elif args.command == "options" and not args.json:
            print(_options_table(_selected(args)))
        elif args.command in {"card", "options"}:
            from nwqlib.algorithms.registry import algorithm_card, options_schema

            value = (algorithm_card if args.command == "card" else options_schema)(_selected(args))
            print(json.dumps(value, allow_nan=False, indent=2))
        elif args.command == "check-method":
            from nwqlib.algorithms.authoring import check_method

            print(
                json.dumps(
                    check_method(_trusted_factory(args.factory)()), allow_nan=False, indent=2
                )
            )
        else:
            from nwqlib.saved_evidence import read_report

            print(json.dumps(read_report(args.path), allow_nan=False, indent=2))
    except NotImplementedError as error:
        parser.exit(2, f"{error}\n")
    except (ValueError, TypeError, LookupError, OSError, ImportError) as error:
        parser.exit(1, f"{error}\n")
    return 0
