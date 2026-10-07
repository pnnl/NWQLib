"""Linear-combination-of-unitaries subroutines."""

from nwqlib.subroutines.lcu.registry import LCU_PREPARATION_IMPLEMENTATIONS

_CORE_EXPORTS = (
    "LCUCircuit",
    "LCUData",
    "LCUPreparationBackend",
    "build_lcu_circuit",
    "build_lcu_prepare",
    "build_lcu_select",
    "prepare_lcu_data",
)
__all__ = list(_CORE_EXPORTS) + ["LCU_PREPARATION_IMPLEMENTATIONS"]


def __getattr__(name):
    if name not in _CORE_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module
    return getattr(import_module(".core", __name__), name)
