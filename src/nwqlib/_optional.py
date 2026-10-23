"""Missing-package errors that name the extra to install, raised by the operation that needs the package."""

from importlib import import_module


def optional_import(module, *, extra):
    """Import a package and add the install command of ``extra`` when the package itself is missing.

    An import error raised inside the package's own dependencies passes on
    unchanged.
    """
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if error.name == module.split(".", 1)[0]:
            error.add_note(f"This selected operation requires the '{extra}' extra: pip install 'nwqlib[{extra}]'.")
        raise
