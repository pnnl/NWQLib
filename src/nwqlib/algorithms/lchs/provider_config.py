"""Shared request and resolved configuration records for LCHS providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeAlias

from nwqlib.serialization import FieldSerializedRecord

ProviderParameter: TypeAlias = str | int | float | bool | None
"""Type of one kernel or quadrature parameter value, a JSON scalar or `None`."""


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
    """Choice of an LCHS kernel or k-quadrature rule, with its explicit parameters.

    Build it with keyword arguments and pass it as `lchs_kernel=` or
    `k_quadrature=` of [`LCHS`][nwqlib.algorithms.lchs.method.LCHS] or of
    [`resolve_lchs_coefficient_plan`][nwqlib.algorithms.lchs.providers.resolve_lchs_coefficient_plan],
    for example
    `LCHS(lchs_kernel=ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": 0.7}))`.
    `implementation` is the only required argument. `LCHS` also accepts the
    name alone or a dict with these two keys. The `lchs_kernel` and
    `k_quadrature` rows of `LCHS` list the names and their parameters.

    Attributes:
        implementation: Required. Kernel or quadrature name. Building this
            record does not check the name, and `LCHS` refuses an unknown
            one.
        parameters: Default empty. Parameter names mapped to JSON scalars
            or `None`, stored sorted and read-only. The named rule supplies
            the defaults and range checks when `LCHS` resolves it.
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
    """Kernel or quadrature choice after its defaults and checks are applied.

    It appears in the `resolved_lchs_kernel` and `resolved_k_quadrature`
    fields of an
    [`LCHSCoefficientPlan`][nwqlib.algorithms.lchs.providers.LCHSCoefficientPlan].
    The fields below are read-only.

    Attributes:
        implementation: Name of the rule used. For an exactly zero L the
            algebraic reduction `"unitary_reduction"` replaces the requested
            quadrature name.
        parameters: Every parameter after defaults, stored sorted and
            read-only. Together with `implementation`, they identify the
            configuration.
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
