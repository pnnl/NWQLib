"""`Program`, NWQLib's description of a circuit as named steps, with bounded parameter binding and structural checks.

The structural check (Program admission) follows the declared lifecycle of
registers and classical values. No quantum SDK is imported.
"""

from .expressions import (
    AdmissionLimits, Binary, Binding, Constant, ExprRef, Expression, Parameter, ParameterRef,
)
from .records import (
    AdaptiveLoop, Allocate, Argument, BlockCall, BlockSignature, Branch, ClassicalStage,
    ClassicalValue, CoherentRegion, Definition, Measure, MeasurementBatch, MetadataRef,
    ObservationKind, Parallel, PortMap, Program, QuantumPort, RangeAxis, Readiness, Register, Release, Repeat,
    Reset, Sequence, Setting, StateClaim,
)

__all__ = [
    "AdmissionLimits", "Binary", "Binding", "Constant", "ExprRef", "Expression", "Parameter",
    "ParameterRef", "AdaptiveLoop", "Allocate", "Argument", "BlockCall", "BlockSignature",
    "Branch", "ClassicalStage", "ClassicalValue", "CoherentRegion", "Definition", "Measure",
    "MeasurementBatch", "MetadataRef", "ObservationKind", "Parallel", "PortMap", "Program", "QuantumPort", "RangeAxis",
    "Readiness", "Register", "Release", "Repeat", "Reset", "Sequence", "Setting", "StateClaim",
]
