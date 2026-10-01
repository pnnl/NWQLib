"""Shared request and resolved configuration records for LCHS providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeAlias

from nwqlib.serialization import FieldSerializedRecord

ProviderParameter: TypeAlias = str | int | float | bool | None


def _immutable_parameter_map(
    parameters: Mapping[str, ProviderParameter],
    *,
    owner: str,
) -> Mapping[str, ProviderParameter]:
    if not isinstance(parameters, Mapping):
        raise TypeError(f"{owner}.parameters must be a mapping of JSON scalars")
    normalized: dict[str, ProviderParameter] = {}
    for key, value in sorted(parameters.items()):
        if not isinstance(key, str):
            raise TypeError(f"{owner}.parameters keys must be strings")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"{owner}.parameters[{key!r}] must be a JSON scalar or None")
        normalized[key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True, kw_only=True)
class ProviderConfig(FieldSerializedRecord):
    """A provider request containing only explicitly supplied parameters.

    Attributes:
        implementation: Name to resolve in the selected kernel or quadrature registry.
            Constructing this record alone does not check registry membership.
        parameters: Explicit parameter names and JSON scalar values, stored in sorted
            read-only order. Defaults and provider-specific domain checks are applied
            by the selected resolver, not by this request record.
    """

    implementation: str
    parameters: Mapping[str, ProviderParameter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError("ProviderConfig.implementation must be a non-empty string")
        object.__setattr__(
            self,
            "parameters",
            _immutable_parameter_map(self.parameters, owner="ProviderConfig"),
        )


@dataclass(frozen=True, kw_only=True)
class ResolvedProviderConfig(FieldSerializedRecord):
    """A normalized provider configuration with defaults injected.

    Attributes:
        implementation: Actual resolved implementation name. An algebraic unitary
            reduction can differ from the originally requested quadrature name.
        parameters: Complete resolved scalar parameters in sorted read-only order;
            together with the implementation name they define the configuration identity.
    """

    implementation: str
    parameters: Mapping[str, ProviderParameter]

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError("ResolvedProviderConfig.implementation must be a non-empty string")
        object.__setattr__(
            self,
            "parameters",
            _immutable_parameter_map(
                self.parameters,
                owner="ResolvedProviderConfig",
            ),
        )
