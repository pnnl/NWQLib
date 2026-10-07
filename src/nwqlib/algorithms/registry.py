"""Explicit trusted registrations with inert external discovery."""

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points
from typing import TypeVar

from nwqlib.core import Source
from .protocol import Method, AlgorithmDescriptor

T = TypeVar("T")
ENTRY_POINT_GROUP = "nwqlib.algorithms"


@dataclass(frozen=True)
class Registration:
    """An entry that makes a Method findable by name and version: its Source and a trusted factory path.

    Build it with keyword arguments, for example
    `Registration(source=MyMethod.descriptor.source, factory="my_methods:MyMethod")`,
    in a module that does not import the Method itself, and pass it to
    [`AlgorithmRegistry`][nwqlib.algorithms.registry.AlgorithmRegistry].
    `source` and `factory` are required. Building it imports nothing. A saved
    Source string never loads code. Only a registration in the running
    process does. [Explicit trusted registration](../algorithm_protocol.md#explicit-trusted-registration)
    shows the metadata module of the reference Method.

    Args:
        source: Source naming the method and version, as
            `descriptor.source` gives it.
        factory: `"module:attribute"` path of a callable that returns
            the Method, called only by `AlgorithmRegistry.resolve`.
        descriptor: The Method's
            [`AlgorithmDescriptor`][nwqlib.algorithms.protocol.AlgorithmDescriptor]
            when already known, with the same method and version as `source`.
            `None` avoids importing the Method to read it.
        method_type: The Method class when already known. Its
            `descriptor` must equal `descriptor`.

    Raises:
        ValueError: If `factory` is not a `module:attribute` path, or the
            descriptor's method and version differ from `source` or from
            `method_type.descriptor`.
        TypeError: If `method_type` is not a Method subclass.
    """

    source: Source
    factory: str
    descriptor: AlgorithmDescriptor | None = None
    method_type: type[Method] | None = None

    def __post_init__(self):
        """Check the registration without importing anything.

        The factory must be a dotted ``module:attribute`` path of
        identifiers. A known descriptor must name the Source's method and
        version, and a known method_type must be a Method subclass whose own
        descriptor equals it, so discovery and resolution describe the same
        implementation.
        """
        module, separator, attribute = self.factory.partition(":")
        if (
            not separator
            or not all(part.isidentifier() for part in module.split("."))
            or not attribute.isidentifier()
        ):
            raise ValueError("factory must be an explicit module:attribute reference")
        if self.descriptor is not None and (self.descriptor.method, self.descriptor.version) != (
            self.source.name,
            self.source.version,
        ):
            raise ValueError("descriptor and registration method/version differ")
        if self.method_type is not None:
            if not isinstance(self.method_type, type) or not issubclass(self.method_type, Method):
                raise TypeError("registration method_type must be an actual Method class")
            if self.descriptor != self.method_type.descriptor:
                raise ValueError(
                    "registration descriptor differs from its Method configuration owner"
                )


def third_party_registrations() -> tuple[Registration, ...]:
    """Return a registration for each installed `nwqlib.algorithms` entry point, without loading its code.

    Each entry point must be named `method@version`, and its value is the
    factory path. The descriptor and Method class of these registrations stay
    `None` until code is loaded explicitly. `builtin_registrations` does not
    call this function, so installed extensions are found only when
    `third_party_registrations` is called.

    Returns:
        registrations (tuple[Registration, ...]): The registrations, sorted by method, version and factory.

    Raises:
        ValueError: If an entry-point name is not `method@version`.
    """
    rows = []
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        method, separator, version = entry.name.rpartition("@")
        if not separator or not method or not version:
            raise ValueError("algorithm entry-point name must be method@version")
        rows.append(
            Registration(
                source=Source(
                    name=method, version=version, domain=ENTRY_POINT_GROUP, reference=entry.value
                ),
                factory=entry.value,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.source.name, row.source.version, row.factory)))


class AlgorithmRegistry:
    """A list of trusted Method registrations that finds a Method by name and version and builds it on request.

    Build it as `AlgorithmRegistry((registration, ...))`. Without registrations it is
    empty. `discover` lists the registrations without running any
    factory, and `resolve` runs only the factory of the requested Source. A
    registry is not a security sandbox, and a registration does not qualify a
    backend.

    Args:
        registrations (Iterable[Registration]): The registrations,
            at most one per method and version.

    Raises:
        ValueError: If two registrations share a method and version.
    """

    def __init__(self, registrations: Iterable[Registration] = ()):
        self._registrations = {}
        for row in sorted(
            registrations, key=lambda row: (row.source.name, row.source.version, row.factory)
        ):
            key = row.source.name, row.source.version
            if key in self._registrations:
                raise ValueError(f"duplicate algorithm registration: {key[0]}@{key[1]}")
            self._registrations[key] = row

    def discover(self) -> tuple[Registration, ...]:
        """Return the registrations in method and version order, without importing any factory.

        Returns:
            registrations (tuple[Registration, ...]): The registrations.
        """
        return tuple(self._registrations.values())

    def resolve(self, source: Source, *, expected_type: type[T] = Method, **configuration) -> T:
        """Build the registered Method for a Source by calling its factory with the given configuration.

        Only the factory of that Source runs, with exactly the configuration
        supplied, and no earlier configuration is reused. The built Method must match
        the registered method, version, descriptor and class.

        Args:
            source (Source): Source of the method and version, equal to the
                registered one.
            expected_type (type): Class the result must be an
                instance of, which keeps a caller's concrete type.
            **configuration (object): Keyword arguments for the factory, such as Method
                settings.

        Returns:
            method (Method): The configured Method.

        Raises:
            LookupError: If no registration has this method and version.
            ValueError: If `source` differs from the registered Source, or the built
                Method differs from the registration.
            TypeError: If the built object is not a complete Method of
                `expected_type`.
        """
        row = self._registrations.get((source.name, source.version))
        if row is None:
            raise LookupError(
                f"algorithm implementation unavailable: {source.name}@{source.version}"
            )
        if source != row.source:
            raise ValueError("Source does not match registered implementation provenance")
        module, _, attribute = row.factory.partition(":")
        method = direct_method(
            getattr(import_module(module), attribute)(**configuration), expected_type=expected_type
        )
        if (method.descriptor.method, method.descriptor.version) != (source.name, source.version):
            raise ValueError("loaded Method differs from registered method/version")
        if row.descriptor is not None and method.descriptor != row.descriptor:
            raise ValueError("loaded Method descriptor differs from discovery")
        if row.method_type is not None and type(method) is not row.method_type:
            raise ValueError("loaded Method configuration type differs from discovery")
        return method


def direct_method(method: object, *, expected_type: type[T] = Method) -> T:
    """Check that an object is a complete Method and return it, without running a factory.

    A complete Method has a descriptor and implements `plan` and `analyze`.

    Args:
        method (object): The object to check.
        expected_type (type): Class the object must be an
            instance of.

    Returns:
        method (Method): The same object.

    Raises:
        TypeError: If it is not an instance of `expected_type`, has no
            descriptor, or does not implement `plan` and `analyze`.
    """
    if not isinstance(method, expected_type):
        raise TypeError("Method does not implement the caller's expected type")
    if not isinstance(method, Method) or not isinstance(
        getattr(method, "descriptor", None), AlgorithmDescriptor
    ):
        raise TypeError("an actual configured Method and descriptor are required")
    if type(method).plan is Method.plan or type(method).analyze is Method.analyze:
        raise TypeError("Method must implement scientific selection and analysis")
    return method


def builtin_registrations() -> tuple[Registration, ...]:
    """Return a registration for each built-in Method, without constructing or planning one.

    It imports the built-in configuration modules, which can load NumPy, SciPy
    and their dependencies, but no quantum SDK. Installed extensions are not
    included. Their configuration fields, defaults and required fields come from
    each Method class's schema.

    Returns:
        registrations (tuple[Registration, ...]): The registrations, in method and version order.
    """
    from . import _METHOD_OWNERS

    rows = []
    for name, owner in _METHOD_OWNERS.items():
        cls = getattr(import_module("nwqlib.algorithms." + owner), name)
        if not isinstance(cls, type) or not issubclass(cls, Method):
            raise TypeError(f"builtin {name} is not a configured Method")
        rows.append(
            Registration(
                source=cls.descriptor.source,
                factory=f"nwqlib.algorithms.{owner}:{name}",
                descriptor=cls.descriptor,
                method_type=cls,
            )
        )
    return AlgorithmRegistry(rows).discover()


def algorithm_card(registration: Registration) -> dict:
    """Return a JSON-ready dict of a registration's metadata for the command-line listing.

    The keys are `source`, `descriptor`, `options_available` and a fixed
    `qualification` text. `descriptor` is None when the registration does not
    carry it, as for an installed extension found by
    `third_party_registrations`. The card reads only the stored metadata, so
    it loads no code and assesses no backend. `nwqlib card` and
    `nwqlib algorithms --json` print it.
    """
    return {
        "source": registration.source.model_dump(mode="json"),
        "descriptor": None
        if registration.descriptor is None
        else registration.descriptor.model_dump(mode="json"),
        "options_available": registration.method_type is not None,
        "qualification": "not assessed by discovery",
    }


def options_schema(registration: Registration) -> dict:
    """Return the JSON schema of a registered Method's configuration, with its required fields and defaults.

    `nwqlib options METHOD` prints it. The schema comes from the Method class
    (`model_json_schema()`), so no required input is guessed.

    Args:
        registration (Registration): A registration whose `method_type` is known.

    Returns:
        schema (dict): The JSON schema.

    Raises:
        ValueError: If the registration's Method class is not loaded, as for an
            installed extension found by `third_party_registrations`. Select the
            extension's code explicitly in Python first.
    """
    if registration.method_type is None:
        raise ValueError(
            "configuration unavailable: explicitly select trusted external code in Python"
        )
    return registration.method_type.model_json_schema()
