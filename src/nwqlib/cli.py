"""Method discovery, explicit author checks and read-only saved reports."""

import argparse
from importlib import import_module
import json


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
    """Import only the author factory explicitly selected by the CLI user."""
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


def main(argv=None):
    """Dispatch declaration/report reads or one explicitly selected trusted author check."""
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
        ("options", "read the actual Method configuration schema and required inputs"),
    ):
        command = commands.add_parser(
            name, help=help_text, description=help_text, allow_abbrev=False
        )
        command.add_argument("method")
        command.add_argument("--version")
        command.add_argument("--third-party", action="store_true")
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
