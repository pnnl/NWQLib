"""Actionable missing-package errors at explicitly selected operation owners."""

from importlib import import_module


def optional_import(module, *, extra):
    """Import the selected package, preserving dependency-internal failures."""
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if error.name == module.split(".", 1)[0]:
            error.add_note(f"This selected operation requires the '{extra}' extra: pip install 'nwqlib[{extra}]'.")
        raise
