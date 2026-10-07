"""Save and load the numerical records, arrays and block bindings of an LCHS Plan.

The record list belongs to this method's actual selected payload. It contains
data only: no provider functions, arbitrary Python objects or SDK gate schema.
"""

from collections.abc import Mapping
from dataclasses import fields
from importlib import import_module
from types import MappingProxyType

import numpy as np

from nwqlib._choice_archive import unsupported_archive_format
from nwqlib.core.records import FrozenArray

from nwqlib.operators.inputs import OperatorInput
from nwqlib.problems.inputs import StateInput
from .method import LCHS
from .primary_records import LCHSReconstruction

# Layout of the saved LCHS manifest: Method, Plan, Problem, output, native data and selected blocks.
FORMAT = "lchs/10"


_RECORD_MODULES = {name: "nwqlib." + module for module, names in (
    ("algorithms.lchs.provider_config", ("ProviderConfig", "ResolvedProviderConfig")),
    ("algorithms.lchs.providers", ("LCHSCoefficientPlan", "LCHSQuadraturePlan", "LCHSPairCompatibility")),
    ("algorithms.lchs.time_independent_terms", ("LCHSQuadratureData", "LCHSProductFormulaSelectData", "_LCHSTrotterNodeRecord",
                                               "LCHSProductFormulaNodes")),
    ("algorithms.lchs.selection", ("LCHSData",)),
    ("algorithms.lchs.periodic", ("_PeriodicPayload",)),
    ("algorithms.lchs.select_synthesis", ("LCHSProductFormulaSelectPlan",)),
    ("algorithms.lchs.source_selection", ("SourceBranchLayout",)),
    ("subroutines.block_encoding.core", ("BlockEncodingPlan",)),
    ("subroutines.block_encoding.banded", ("BandSpecification",)),
    ("subroutines.pauli_decomposition", ("PauliDecomposition", "PauliTerm")),
    ("subroutines.qsp.evolution", ("JacobiAngerExpansion", "QSPPreparedEvolution")),
    ("subroutines.qsp.phases", ("SymmetricQSPPhases",)),
    ("subroutines.state_preparation.mps", ("MPSDecomposition",)),
    ("subroutines.trotterization.error_budget", ("TrotterStepSelection",)),
    ("subroutines._multiplexors", ("AffineAngleTable",)),
) for name in names}


def _record_type(name):
    # Modules come from this method-owned list, never a file's import string.
    # Import only records that occur, so periodic/host archives remain SDK-free.
    return getattr(import_module(_RECORD_MODULES[name]), name)


class _SelectedData:
    """Write and read selected LCHS values as tagged data and array files.

    Only scalars, arrays, ingested inputs, the Plan's own Method and the
    record types listed in _RECORD_MODULES can be saved, so an archive never
    holds pickled code. Reading restores record fields directly, bypassing
    dataclass validation and normalization, so every saved number comes back
    bit for bit.

    With leaves=True an array stays an ndarray leaf of the encoded tree, for
    ArchiveFiles.write_numpy_json to store as its own NPY file and register
    as a dependency of that JSON file. Otherwise each array is written here
    and the tree names its file.
    """

    def __init__(self, files, method, *, leaves=False):
        self.files, self.method, self.next_file, self.leaves = files, method, 0, leaves

    def write(self, value):
        """Return the JSON-ready encoding of value, writing array files as needed.

        JSON scalars (None, str, int, float, bool) are stored as themselves.
        Every other value becomes a dict with a ``kind`` tag: ``method`` for
        the Plan's own Method, ``array``, ``state`` or ``operator`` for a
        file written next to the archive (with ``leaves`` set, ``array``
        holds the ndarray itself), ``frozen_array`` for a ``FrozenArray``
        stored as its array and read back as one, ``complex`` with real and imag
        parts, ``mapping`` and ``tuple`` for containers (lists are saved as
        tuples), or the record class name with its dataclass fields.
        Anything else raises TypeError.
        """
        if isinstance(value, LCHS):
            if value is not self.method:
                raise ValueError("LCHS selected data must use its actual Plan Method")
            return dict(kind="method")
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or type(value) in (str, int, float, bool):
            return value
        if isinstance(value, FrozenArray):
            return dict(kind="frozen_array", value=self.write(value.array)["value"])
        if isinstance(value, np.ndarray) and self.leaves:
            return dict(kind="array", value=value)
        if isinstance(value, (np.ndarray, StateInput, OperatorInput)):
            name = f"selected-{self.next_file}"
            self.next_file += 1
            if isinstance(value, np.ndarray):
                return dict(kind="array", value=self.files.write_array(name+".npy", value))
            if isinstance(value, StateInput):
                return dict(kind="state", value=self.files.write_state(name, value))
            return dict(kind="operator", value=self.files.write_operator(name, value))
        if type(value) is complex:
            return dict(kind="complex", real=value.real, imag=value.imag)
        if isinstance(value, Mapping):
            return dict(kind="mapping", values={key: self.write(item) for key, item in value.items()})
        if isinstance(value, (tuple, list)):
            return dict(kind="tuple", values=[self.write(item) for item in value])
        if type(value).__name__ in _RECORD_MODULES and _record_type(type(value).__name__) is type(value):
            return dict(kind=type(value).__name__, values={field.name: self.write(getattr(value, field.name))
                                                        for field in fields(value)})
        raise TypeError(f"LCHS archive has no selected data format for {type(value).__name__}")

    def read(self, value):
        """Rebuild a value encoded by write from its ``kind`` tag.

        Mappings return as read-only MappingProxyType. Records are rebuilt
        field by field without calling their constructors (see the class
        docstring), and only record classes listed in _RECORD_MODULES are
        imported.
        """
        if not isinstance(value, dict):
            return value
        kind = value["kind"]
        if kind == "method":
            return self.method
        if kind in ("array", "frozen_array"):
            # read_numpy_json already restored an ndarray leaf in place.
            item = value["value"]
            item = item if isinstance(item, np.ndarray) else self.files.read_array(item)
            return FrozenArray(item) if kind == "frozen_array" else item
        if kind == "state":
            return self.files.read_state(value["value"])
        if kind == "operator":
            return self.files.read_operator(value["value"])
        if kind == "complex":
            return complex(value["real"], value["imag"])
        if kind == "tuple":
            return tuple(self.read(item) for item in value["values"])
        if kind == "mapping":
            return MappingProxyType({key: self.read(item) for key, item in value["values"].items()})
        cls = _record_type(kind)
        result = object.__new__(cls)
        # Restore saved fields directly. Dataclass intake can otherwise repeat
        # normalization, numerical validation or copies of selected arrays.
        for field in fields(cls):
            object.__setattr__(result, field.name, self.read(value["values"][field.name]))
        return result


_CONSTRUCTORS = {
    'input': ('nwqlib.blocks.selection','_preparation_circuit'),
    'parity': ('nwqlib.blocks.selection','_primitive_circuit'),
    'tensor_prep': ('nwqlib.algorithms.lchs.native','construct_preparation'),
    'select': ('nwqlib.algorithms.lchs.native','construct_select'),
    'periodic_step': ('nwqlib.algorithms.lchs.periodic','construct_periodic_step'),
    'periodic_phase': ('nwqlib.algorithms.lchs.periodic','construct_periodic_phase'),
}


def save(method,plan,files):
    """Save the Plan, its native numerical data and how each block is rebound.

    Each selected block is saved as its known constructor name, or as the
    index of its base for an inverse or controlled copy. Only a PREP tensor
    block saves its payload. Planning binds the other payloads from the
    Plan's own data, and loading binds them the same way. SELECT and the
    periodic leaves share the Plan's one native payload, SELECT also takes the
    Problem's elapsed time, the input preparation is the Problem's initial
    state (quantum.py, periodic.py and host.py) and a parity block has no
    payload. No circuit is saved. Native circuits are built only when a point
    is prepared.

    The native payload grows with the grid and the schedule (node tables,
    coefficients, host actions and SELECT angle tables), so it is written to
    its own JSON file with NPY array leaves, and the returned record names
    that file in place of the payload. That record is embedded in result.json
    and in a Run's selection file.
    """
    data = _SelectedData(files,method)
    selected = []
    if plan.construction.selections:
        records = [block.record.content_id for block in plan.blocks]
        for block in plan.blocks:
            if block._base is not None:
                selected.append(dict(base=records.index(block._base.record.content_id)))
            else:
                key = next((name for name,(module,function) in _CONSTRUCTORS.items()
                    if block._constructor.__module__==module and block._constructor.__name__==function),None)
                if key is None:
                    raise ValueError('LCHS archive requires its actual known selected constructor')
                if key in {'select','periodic_step','periodic_phase'}:
                    payload = block._payload[0] if key=='select' else block._payload
                    if payload is not plan._native['native_data']:
                        raise ValueError('LCHS native leaf differs from its shared selected data')
                if key=='tensor_prep':
                    selected.append(dict(constructor=key,payload=data.write(block._payload)))
                else:
                    selected.append(dict(constructor=key))
    saved = dict(format=FORMAT,plan=plan.to_record(),problem=files.write_problem(plan.problem),
        output=files.write_output(plan.output),method=method.model_dump(mode='json',exclude_computed_fields=True))
    # Written after the problem, so an input array shared with the native
    # payload keeps its problem file name.
    tree = _SelectedData(files,method,leaves=True)
    native = files.write_numpy_json('lchs-native.json',tree.write(plan._native.get('native_data')))
    return dict(saved,native=native,selected=selected)


def load(saved,files):
    """Rebind the saved LCHS constructors to the saved native data and the Problem.

    The native payload is read back from the JSON file that save named, and
    each PREP tensor payload from its block entry. The other payloads are
    bound as planning bound them (see save). When the Problem has no source,
    the coefficient plan inside the native data is also bound as the Plan's
    coefficient plan, as at planning. A source Plan binds None there, because
    its execution does not use that homogeneous plan.

    Loading checks the format and restores each inverse or controlled copy
    from its saved base index. The periodic Strang payload is used as saved.
    Loading never reselects a grid, fits phases or solves an eigensystem.
    """
    if saved.get('format')!=FORMAT:
        raise unsupported_archive_format("LCHS archive", saved.get('format'), FORMAT)
    method = LCHS.model_validate(saved['method'])
    plan = files.read_plan(saved['plan'],problem=files.read_problem(saved['problem']),method=method,
        output=files.read_output(saved['output']),reconstruction=LCHSReconstruction.model_validate(saved['plan']['reconstruction']))
    data = _SelectedData(files,method)
    selected = data.read(files.read_numpy_json(saved['native']))
    native = {} if selected is None else dict(native_data=selected,
        coefficient_plan=selected.coefficient_plan if plan.problem.source is None else None)
    blocks = []
    if plan.construction.selections:
        from nwqlib.blocks.selection import SelectedBlock
        payloads = dict(select=(selected,plan.problem.elapsed_time),periodic_step=selected,periodic_phase=selected,
            input=plan.problem.initial_state,parity=None)
        for record,item in zip(plan.construction.selections,saved['selected'],strict=True):
            if 'base' in item:
                block = SelectedBlock.bind(record,base=blocks[item['base']])
            else:
                key = item['constructor']
                module,function = _CONSTRUCTORS[key]
                if key=='tensor_prep':
                    payload = data.read(item['payload'])
                else:
                    payload = payloads[key]
                block = SelectedBlock.bind(record,payload=payload,constructor=getattr(import_module(module),function))
            blocks.append(block)
    elif plan.construction.kernels:
        from .host import bind_host
        host_data = None if plan.reconstruction.mode in ('initial','zero') else selected
        blocks = [bind_host(plan,host_data)]
    return plan._bind(blocks=tuple(blocks),**native)
