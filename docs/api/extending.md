# Extending NWQLib

This page lists the classes and functions for adding a method: the `Method` base class and its hooks, the checker and registry for a new method, the `Program` (NWQLib's description of a circuit as named steps), the blocks a Program calls, the readouts it requests and the hooks that save its data. [Add a Method](../algorithm_protocol.md) states every requirement of a Method, and [Run your own circuit](../own_circuit.md) builds the smallest Method step by step.

```python
from nwqlib.algorithms import AlgorithmDescriptor, ApplicabilityError, Method
from nwqlib.algorithms.authoring import MethodCase, check_method
from nwqlib.blocks import SelectedConstruction, lower_qiskit, select_preparation
from nwqlib.ir import (
    Allocate, BlockCall, Definition, Measure, Program, Register, Sequence,
)
```

A new Method ships a `case()` function that returns a `MethodCase`, and `check_method` runs it. With the reference Hadamard Method of `tests/_hadamard_method.py` on the import path:

```python
from nwqlib.algorithms.authoring import check_method
from _hadamard_method import case

report = check_method(case())
print(report["status"], report["method"])
# CONFORMANT example.hadamard_pauli_expectation
```

From a shell, `python -m nwqlib check-method my_methods:case` runs the same check on the `case()` of your module `my_methods`.

| Task | Entries |
| --- | --- |
| Write the Method class | [`Method`](#nwqlib.algorithms.protocol.Method), [`AlgorithmDescriptor`](#nwqlib.algorithms.protocol.AlgorithmDescriptor), [`ApplicabilityError`](#nwqlib.algorithms.protocol.ApplicabilityError) |
| Tie a Plan and a Result to their inputs and data | [`Plan._bind`](#nwqlib.core.planning.Plan._bind), [`Result._attach`](#nwqlib.core.analysis.Result._attach), [`Result.validate_plan`](#nwqlib.core.analysis.Result.validate_plan) |
| Inspect a prepared circuit and its resources | [`PreparedHandle`](#nwqlib._prepared_execution.PreparedHandle) |
| Check a Method against its case | [`MethodCase`](#nwqlib.algorithms.authoring.MethodCase), [`check_method`](#nwqlib.algorithms.authoring.check_method) |
| Register a Method and resolve it by name | [`Registration`](#nwqlib.algorithms.registry.Registration), [`AlgorithmRegistry`](#nwqlib.algorithms.registry.AlgorithmRegistry), [`direct_method`](#nwqlib.algorithms.registry.direct_method) |
| Describe the circuit as a Program | [`Program`](#nwqlib.ir.records.Program), [registers and block signatures](#registers-values-and-block-signatures), [nodes](#nodes) |
| Give a Program named parameters and expressions | [`Parameter`](#nwqlib.ir.expressions.Parameter), [`Expression`](#nwqlib.ir.expressions.Expression) |
| Make the blocks a Program calls and build the Qiskit circuit | [`select_preparation`](#nwqlib.blocks.selection.select_preparation), [`transform_block`](#nwqlib.blocks.selection.transform_block), [`SelectedConstruction`](#nwqlib.blocks.records.SelectedConstruction), [`lower_qiskit`](#nwqlib.blocks.lowering.lower_qiskit) |
| Request readouts and read the observations | [`Experiment`](#nwqlib.core.planning.Experiment), [`ObservationSpec`](#nwqlib.core.planning.ObservationSpec), [`ObservationChunk`](#nwqlib.execution.ObservationChunk), [`Histogram`](#nwqlib.execution.Histogram) |
| Save and reload a Method's data | [`ArchiveFiles`](#nwqlib._choice_archive.ArchiveFiles), [`write_blocks`](#nwqlib.blocks._archive.write_blocks), [`read_blocks`](#nwqlib.blocks._archive.read_blocks) |

## Implement a method

Subclass `Method`, give it a `descriptor`, and implement `plan` and `analyze`. The other hooks have defaults. Plan and Result have four protected hooks that a Method's code calls or overrides: `Plan._bind`, `Result._attach`, `Result._validate_common_plan` and `Result._summary_lines`. They keep each input and its data tied to the Plan and run no extra computation, and [Supported protected extension hooks](../algorithm_protocol.md#supported-protected-extension-hooks) states what each must do. A Method's Result type overrides `validate_plan`, and a new Problem type defines `default_output`.

::: nwqlib.algorithms.protocol.Method
    options:
      heading_level: 3
      members:
        - plan
        - analyze
        - prepare
        - prepare_all_refusal
        - execute
        - error_model
        - verify
        - recover_analysis

::: nwqlib.algorithms.protocol.AlgorithmDescriptor
    options:
      heading_level: 3

::: nwqlib.algorithms.protocol.ApplicabilityError
    options:
      heading_level: 3

::: nwqlib.problems.records.ProblemRecord.default_output
    options:
      heading_level: 3

::: nwqlib.core.planning.Plan._bind
    options:
      heading_level: 3

::: nwqlib.core.planning.Plan.resolve
    options:
      heading_level: 3

::: nwqlib.core.analysis.Result._attach
    options:
      heading_level: 3

::: nwqlib.core.analysis.Result.validate_plan
    options:
      heading_level: 3

::: nwqlib.core.analysis.Result._validate_common_plan
    options:
      heading_level: 3

::: nwqlib.core.analysis.Result._summary_lines
    options:
      heading_level: 3

::: nwqlib._prepared_execution.PreparedHandle
    options:
      heading_level: 3
      members:
        - inspect_circuit
        - inspect_resources

## Check and register a method

::: nwqlib.algorithms.authoring.MethodCase
    options:
      heading_level: 3

::: nwqlib.algorithms.authoring.check_method
    options:
      heading_level: 3

::: nwqlib.algorithms.registry.Registration
    options:
      heading_level: 3

::: nwqlib.algorithms.registry.AlgorithmRegistry
    options:
      heading_level: 3
      members:
        - discover
        - resolve

::: nwqlib.algorithms.registry.builtin_registrations
    options:
      heading_level: 3

::: nwqlib.algorithms.registry.third_party_registrations
    options:
      heading_level: 3

::: nwqlib.algorithms.registry.direct_method
    options:
      heading_level: 3

::: nwqlib.algorithms.registry.options_schema
    options:
      heading_level: 3

## Describe a circuit as a Program

A `Program` lists named nodes: register allocation, block calls, measurements, repeats and their order. It is checked when it is built. [Describe a circuit as a Program](../ir.md) explains the nodes and [Program checks](../development/program_checks.md) the lifecycle rules.

::: nwqlib.ir.records.Program
    options:
      heading_level: 3
      members:
        - check_readiness
        - bind
        - select_experiment
        - iter_definitions

::: nwqlib.ir.records.Definition
    options:
      heading_level: 3
      members: false

::: nwqlib.ir.records.Readiness
    options:
      heading_level: 3
      members:
        - ready
        - require_ready

::: nwqlib.ir.expressions.AdmissionLimits
    options:
      heading_level: 3
      members: false

### Registers, values and block signatures

::: nwqlib.ir.records.Register
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.ClassicalValue
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.BlockSignature
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.QuantumPort
    options:
      heading_level: 4
      members: false

### Nodes

::: nwqlib.ir.records.Sequence
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Parallel
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Repeat
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.BlockCall
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.PortMap
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Argument
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Allocate
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Release
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Measure
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Reset
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.CoherentRegion
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.StateClaim
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Branch
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.ClassicalStage
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.AdaptiveLoop
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.MeasurementBatch
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.Setting
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.RangeAxis
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.records.MetadataRef
    options:
      heading_level: 4
      members: false

### Parameters and expressions

::: nwqlib.ir.expressions.Parameter
    options:
      heading_level: 4
      members:
        - admit

::: nwqlib.ir.expressions.Binding
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.expressions.Expression
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.expressions.Constant
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.expressions.ParameterRef
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.expressions.Binary
    options:
      heading_level: 4
      members: false

::: nwqlib.ir.expressions.ExprRef
    options:
      heading_level: 4
      members: false

## Select and lower blocks

A Program calls blocks by signature name. Each `select_*` function returns a `SelectedBlock` whose record states the block's promised action, recipe and cost rules, and `lower_qiskit` builds the Qiskit circuit from the Program and its blocks. [Compose blocks](../blocks.md) composes a Pauli block encoding from these functions.

::: nwqlib.blocks.selection.select_preparation
    options:
      heading_level: 3

::: nwqlib.blocks.encoding.select_block_encoding
    options:
      heading_level: 3

::: nwqlib.blocks.selection.select_signed_pauli
    options:
      heading_level: 3

::: nwqlib.blocks.selection.select_pauli_preparation
    options:
      heading_level: 3

::: nwqlib.blocks.selection.select_pauli_readout
    options:
      heading_level: 3

::: nwqlib.blocks.selection.select_zero_reflection
    options:
      heading_level: 3

::: nwqlib.blocks.selection.transform_block
    options:
      heading_level: 3

::: nwqlib.blocks.selection.signed_pauli_cx_bound
    options:
      heading_level: 3

::: nwqlib.blocks.selection.pauli_readout_cx_bound
    options:
      heading_level: 3

::: nwqlib.blocks.selection.SelectedBlock
    options:
      heading_level: 3
      members:
        - bind

::: nwqlib.blocks.records.SelectedConstruction
    options:
      heading_level: 3
      members:
        - encoding_semantics

::: nwqlib.blocks.records.PauliEncoding
    options:
      heading_level: 3
      members: false

::: nwqlib.blocks.records.SelectedDefinition
    options:
      heading_level: 3
      members: false

::: nwqlib.blocks.records.Primitive
    options:
      heading_level: 3
      members: false

::: nwqlib.blocks.lowering.lower_qiskit
    options:
      heading_level: 3

::: nwqlib.blocks.lowering.LogicalCircuit
    options:
      heading_level: 3
      members: false

## Request readouts and reduce observations

A Plan's experiments name the readout each circuit returns. Reducers that turn a saved state into statistics during data collection are registered by name in `nwqlib.core.planning.READOUT_REDUCERS`, and the modules of the built-in reducers are listed in `nwqlib.core.planning.BUILTIN_REDUCER_MODULES`. The reduction-allowance row of [Add a Method](../algorithm_protocol.md#configuration-and-selected-science) describes the registration.

::: nwqlib.core.planning.Experiment
    options:
      heading_level: 3
      members: false

::: nwqlib.core.planning.ObservationSpec
    options:
      heading_level: 3
      members:
        - point_observation
        - point_observations
        - point_readout_fields
        - reject_unsupported_schedule
        - unsupported_schedule

::: nwqlib.core.planning.ReductionContext
    options:
      heading_level: 3
      members: false

::: nwqlib.core.planning.registered_reducer
    options:
      heading_level: 3

::: nwqlib.execution.ObservationChunk
    options:
      heading_level: 3
      members:
        - acquisition_key
        - declares_readout_of
        - from_histogram
        - histogram
        - readout
        - validate_unit_bound

::: nwqlib.execution.Histogram
    options:
      heading_level: 3
      members:
        - index_list
        - indices
        - packed_indices

::: nwqlib.execution.PreparedArtifact
    options:
      heading_level: 3
      members:
        - probability_window
        - saved_state_error
        - saved_state_probability_window
        - state_error

## Save and restore a method's data

A Method whose Runs or Results are saved implements `save_archive(plan, files)` and `load_archive(saved, files)` and names its Result class in `result_type`. Both hooks receive an `ArchiveFiles` object, `save_archive` reads the bound blocks from `plan.blocks`, and [When the archive hooks are required](../own_circuit.md#when-the-archive-hooks-are-required) shows them for a small Method.

::: nwqlib._choice_archive.ArchiveFiles
    options:
      heading_level: 3
      members:
        - read_plan
        - read_problem
        - read_output
        - write_state
        - write_operator
        - read_path

::: nwqlib.core.planning.Plan.blocks
    options:
      heading_level: 3

::: nwqlib.blocks._archive.write_blocks
    options:
      heading_level: 3

::: nwqlib.blocks._archive.read_blocks
    options:
      heading_level: 3

::: nwqlib.core.records.InputRef
    options:
      heading_level: 3
      members: false
