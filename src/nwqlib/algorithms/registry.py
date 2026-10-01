"""Explicit trusted registrations with inert external discovery."""

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import TypeVar

from nwqlib.core import Source
from .protocol import Method, AlgorithmDescriptor

T = TypeVar("T")
ENTRY_POINT_GROUP = "nwqlib.algorithms"


@dataclass(frozen=True)
class Registration:
    """Trusted in-process factory, never loaded from a saved Source string.

    method_type names an already known actual configuration class. External
    inventory may leave it and descriptor unknown until explicit code selection.

    Attributes:
        source: Exact trusted Method/version provenance used for resolution.
        factory: Explicit module:attribute callable selected only by registry resolution.
        descriptor: Already-known applicability metadata; None avoids importing unknown implementation code.
        method_type: Already-known Method configuration class; None for an unloaded external registration.
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


def third_party_registrations(
    entries: Iterable[EntryPoint] | None = None,
) -> tuple[Registration, ...]:
    """Read installed method@version entry-point metadata without loading code."""
    entries = entry_points(group=ENTRY_POINT_GROUP) if entries is None else entries
    rows = []
    for entry in entries:
        if entry.group != ENTRY_POINT_GROUP:
            continue
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
    """An explicit inventory of trusted code; discovery does not execute factories.

    Default inventory is empty. External code runs only when resolve selects its
    exact registration. This is not a security sandbox or backend qualification.
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
        """Return the registrations in name/version order without importing any factory."""
        return tuple(self._registrations.values())

    def resolve(self, source: Source, *, expected_type: type[T] = Method, **configuration) -> T:
        """Invoke only the selected factory with its explicit configuration."""
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
    """Check the actual directly supplied configuration; execute no factory."""
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
    """Read actual builtin configuration types, without constructing or planning.

    Builtin configuration modules are trusted package code. External entry points
    are neither discovered nor loaded implicitly. Scientific fields and defaults
    come from the actual Method class schema, including its required fields.
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
    """Describe a registration from its stored metadata. Discovery qualifies nothing."""
    return {
        "source": registration.source.model_dump(mode="json"),
        "descriptor": None
        if registration.descriptor is None
        else registration.descriptor.model_dump(mode="json"),
        "options_available": registration.method_type is not None,
        "qualification": "not assessed by discovery",
    }


def options_schema(registration: Registration) -> dict:
    """Read the actual Method schema; do not guess required scientific inputs."""
    if registration.method_type is None:
        raise ValueError(
            "configuration unavailable: explicitly select trusted external code in Python"
        )
    return registration.method_type.model_json_schema()
