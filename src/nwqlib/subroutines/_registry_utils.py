"""Small helpers for implementation-selection registries."""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping


@cache
def _installed_version(package_name: str) -> str | None:
    """Return one installed distribution version without repeated metadata reads."""

    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def package_versions(package_names: tuple[str, ...]) -> dict[str, str | None]:
    """Return installed package versions for report metadata.

    importlib.metadata normalizes distribution names, so ``scikit-tt`` and
    ``scikit_tt`` find the same installed distribution.
    """

    return {package_name: _installed_version(package_name) for package_name in package_names}


def implementation_metadata(
    registry: Mapping[str, Mapping[str, Any]],
    name: str,
    *,
    slot: str,
) -> dict[str, Any]:
    """Return a JSON-like implementation metadata copy from a registry."""

    try:
        record = registry[name]
    except KeyError as exc:
        available = "', '".join(sorted(registry))
        raise ValueError(f"{slot} implementation must be one of: '{available}'") from exc

    metadata = dict(record)
    package_names = tuple(str(item) for item in metadata.get("package_names", ()))
    metadata["name"] = name
    metadata["package_names"] = list(package_names)
    metadata["package_versions"] = package_versions(package_names)
    return metadata


__all__ = ["implementation_metadata", "package_versions"]
