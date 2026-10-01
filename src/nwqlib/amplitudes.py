"""Selected simulator amplitudes in original physical coordinates.

Block-encoding convention: An, Childs & Lin, arXiv:2312.03916v2,
Appendix A.2 Definition 23 and A.3 Lemma 24, Eq. (178).
https://arxiv.org/html/2312.03916v2#A3
The method supplies its positive recovery factor; this owner projects the
actual native output and never reconstructs a missing state from counts.
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

import sys
from typing import Literal

from pydantic import StrictBool, model_validator

from nwqlib.artifacts import ArrayOutput, UnavailableOutput
from nwqlib.core.records import ContentID, Record, Source
from nwqlib.operators.access import Count
from nwqlib.problems.inputs import PhysicalScale, compose_recovery
from nwqlib._validation import NUMERICAL_RELATION_RTOL, validate_normalized_mass

AMPLITUDE_MASS_LABELS = ("algorithm_success_mass", "physical_slice_mass")
AMPLITUDE_MASS_UNAVAILABLE = "nonzero selected mass is not representable in binary64"
# A direct helper has a finite known-array envelope. A Run supplies its actual
# selected backend allowance; this is neither a Plan limit nor process RSS.
DEFAULT_AMPLITUDE_BYTES = DEFAULT_MAX_BYTES


def _slice(native, free, fixed, *, size=None):
    """Project fixed bits and an optional original-coordinate prefix.

    Free bits are least significant first in the output index. Contiguous bit
    positions produce a strided view; a noncontiguous projection allocates only
    the requested number of coordinates, never a full-native-width index map.
    """
    import numpy as np

    offset = sum(value << bit for bit, value in fixed)
    size = (1 << len(free)) if size is None else size
    if not free:
        return native[offset:offset + 1]
    if free == tuple(range(free[0], free[0] + len(free))):
        step = 1 << free[0]
        return native[offset:offset + size * step:step]
    indices = np.zeros(size, dtype=np.intp)
    counter = np.arange(size, dtype=np.intp)
    for position, bit in enumerate(free):
        indices |= ((counter >> position) & 1) << bit
    indices |= offset
    return native[indices]


def validate_amplitude_masses(declaration, values):
    """Validate the actual algorithm and narrower physical populations.

    Locally derived from orthogonal coordinate projection: when P_phys is a
    subprojector of P_success, ||P_phys psi||² <= ||P_success psi||². Equality
    requires the same fixed bits AND the whole coordinate register, so a padded
    three-coordinate solution does not inherit four-coordinate mass equality.
    """
    from nwqlib.execution import ScalarValue

    if not declaration.keep_masses:
        if values:
            raise ValueError("amplitude declaration did not select branch masses")
        return
    if (any(type(value) is not ScalarValue or value.frame != "encoded_branch" for value in values)
            or tuple(value.label for value in values) != AMPLITUDE_MASS_LABELS):
        raise ValueError("amplitude masses require the selected ordered scalar labels and frame")
    algorithm, physical = (value.value for value in values)
    for value in (algorithm, physical):
        validate_normalized_mass(value)
    window = NUMERICAL_RELATION_RTOL * max(1.0, algorithm) if algorithm is not None else None
    if algorithm is not None and physical is not None and physical - algorithm > window:
        raise ValueError("physical-coordinate mass exceeds algorithm-success mass")
    if not declaration.has_physical_projection and (
            (algorithm is None) != (physical is None)
            or algorithm is not None and abs(algorithm - physical) > window):
        raise ValueError("the same amplitude event must have the same algorithm and physical mass")


class AmplitudeReadout(Record):
    """One phase-faithful amplitude projection of a selected construction.

    coordinates are least significant first. success is the algorithm event;
    conditions and the original output length select the physical coordinates.
    No execution limit enters this scientific selection. recovery is a positive
    method scale, not an observed norm or a promise about approximation error.
    A physical-frame output requires that recovery.

    Attributes:
        construction_id: Selected construction whose native state is read.
        source: Method source of the projection.
        width: Native circuit width in qubits.
        coordinates: Native qubits that form the output coordinate index, least significant first.
        success: Fixed ``(qubit, bit)`` pairs of the algorithm's success event, such as ancillas at zero.
        conditions: Further fixed ``(qubit, bit)`` pairs that select the physical coordinates.
        output: Requested vector output, its frame and its original dimension.
        recovery: Positive composed scale that maps the selected encoded amplitudes to physical units. None is allowed only for a unit-frame output.
        keep_masses: Whether the reduction also returns the algorithm-success and physical-slice masses.
    """

    schema_version: Literal[2] = 2
    construction_id: ContentID
    source: Source
    width: Count
    coordinates: tuple[Count, ...]
    success: tuple[tuple[Count, Count], ...] = ()
    conditions: tuple[tuple[Count, Count], ...] = ()
    output: ArrayOutput
    recovery: PhysicalScale | None
    keep_masses: StrictBool = False

    @model_validator(mode="after")
    def _projection(self):
        """Require a projection that reads each native qubit exactly once.

        The coordinates and the fixed success and condition qubits must
        partition the native width, fixed values must be 0 or 1, and the
        original vector dimension must fit the coordinate register. The output
        is a complex128 vector. A
        physical-frame output needs a recovery, and any recovery must be
        positive with ``composed_floating_point_recovery`` evidence.
        """
        fixed = self.success + self.conditions
        bits = self.coordinates + tuple(bit for bit, _ in fixed)
        if (len(set(bits)) != len(bits) or len(bits) != self.width
                or any(bit >= self.width for bit in bits)
                or any(value not in (0, 1) for _, value in fixed)):
            raise ValueError("amplitude coordinates and distinct fixed bits must partition the native width")
        dimension = self.output.basis.dimension
        if self.output.dtype != "complex128":
            raise ValueError("amplitude readout returns a complex128 vector")
        if self.output.kind != "vector" or (dimension - 1).bit_length() > len(self.coordinates):
            raise ValueError("original vector dimension must fit the selected coordinate register")
        if self.output.frame == "physical" and self.recovery is None:
            raise ValueError("physical amplitude output requires its selected recovery")
        if self.recovery is not None and (
                self.recovery.mantissa == 0 or self.recovery.evidence != "composed_floating_point_recovery"):
            raise ValueError("amplitude recovery must be positive composed recovery evidence")
        return self

    @property
    def has_physical_projection(self):
        return bool(self.conditions) or self.output.basis.dimension != 1 << len(self.coordinates)

    def admit_materialization(self, *, max_bytes=DEFAULT_AMPLITUDE_BYTES, max_bytes_source=None):
        """Check native return and known projection scratch before allocation.

        The finite bound includes complex native data, the requested gather,
        normalization/physical-scaling scratch, and an optional wider mass
        projection. It excludes undocumented SDK workspace and process overhead.
        A refusal appends ``max_bytes_source``, when given, to name where
        ``max_bytes`` comes from, such as a Run's ``max_data_bytes``.
        """
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("amplitude max_bytes must be a nonnegative integer")
        source = "" if max_bytes_source is None else f", {max_bytes_source}"
        native_limit = min(sys.maxsize // 16, max_bytes // 16)
        # 2**bit_length(m) > m, so this rejects a native vector of more than
        # native_limit complex128 entries before 2**width is formed.
        if self.width >= max(1, native_limit.bit_length()):
            raise ValueError(f"native amplitude return of 2**{self.width} complex128 entries exceeds the "
                             f"known-array allowance max_bytes={max_bytes}{source}")
        native, size = 1 << self.width, self.output.basis.dimension
        # 16 bytes per native complex128 amplitude, plus the registered
        # 192-byte allowance per requested output entry (output.basis.dimension)
        # for the gather, normalization and physical-scaling buffers
        # (ENGINEERING_CONSTANTS.md, "Numerical choices and representation sizes").
        payload = 16 * native + 192 * size
        if self.keep_masses and self.has_physical_projection:
            free = tuple(bit for bit in range(self.width) if bit not in dict(self.success))
            wider = 1 << len(free)
            contiguous = not free or free == tuple(range(free[0], free[0] + len(free)))
            payload += (16 if contiguous else 80) * wider + 8 * len(free)
        if payload > max_bytes:
            raise ValueError(f"amplitude projection needs {payload} known array bytes, more than the allowance "
                             f"max_bytes={max_bytes}{source}")
        return payload

    def select(self, native):
        """Access original coordinates; padded dummy entries are excluded."""
        import numpy as np

        if (type(native) is not np.ndarray or native.dtype != np.dtype("complex128")
                or native.shape != (1 << self.width,)):
            raise ValueError("native amplitude dtype/shape differs from the selected readout")
        return _slice(native, self.coordinates, self.success + self.conditions,
                      size=self.output.basis.dimension)


def reduce_amplitudes(readout, native, *, max_bytes=DEFAULT_AMPLITUDE_BYTES, max_bytes_source=None):
    """Return the requested vector, physical norm, absence and selected masses.

    ``max_bytes`` and ``max_bytes_source`` go to ``admit_materialization``.

    Returns:
        ``(vector, physical_scale, unavailable, masses)``. vector is the
        unit direction or the recovered physical amplitudes of the selected
        coordinates, or None. physical_scale is the recovered physical norm
        as a PhysicalScale, or None without a recovery. unavailable is an
        UnavailableOutput with the reason when vector is None. masses is
        ``(algorithm_success_mass, physical_slice_mass)`` when
        ``keep_masses`` is set, otherwise empty.
    """
    import numpy as np
    from nwqlib.problems.scalars import physical_vector_statistics

    readout.admit_materialization(max_bytes=max_bytes, max_bytes_source=max_bytes_source)
    selected = readout.select(native)
    if not np.isfinite(selected).all():
        raise ValueError("selected native amplitudes must be finite")
    # ACL arXiv:2312.03916v2, Eq. (178): the zero-ancilla block is the desired
    # LCU divided by
    # ||c||_1. Here selected is that encoded amplitude vector after the method's
    # physical-coordinate projection, and recovery includes ||c||_1 plus its
    # input/encoding factors. Multiply by this positive factor, preserving phase;
    # a normalized direction is selected only by an explicit unit-vector output.
    recovery = readout.recovery
    # Only a unit-frame output returns the normalized direction. The norm and
    # its binary scale do not depend on whether the direction is formed, so a
    # physical-frame output makes no normalized copy of the selected vector.
    scale, direction, statistics = physical_vector_statistics(
        selected, need_direction=readout.output.frame == "unit")
    physical_scale = None
    if recovery is not None:
        if scale.mantissa == 0:
            physical_scale = scale
        else:
            combined = compose_recovery(scale, recovery)
            physical_scale = PhysicalScale(mantissa=combined.mantissa, exponent=combined.exponent)
    masses = ()
    if readout.keep_masses:
        physical = statistics[0].value
        algorithm = physical
        if readout.has_physical_projection:
            from math import frexp, isfinite
            from nwqlib._numerics import stable_vector_norm
            free = tuple(bit for bit in range(readout.width) if bit not in dict(readout.success))
            wider = _slice(native, free, readout.success)
            norm = stable_vector_norm(wider)
            if not isfinite(norm):
                raise ValueError("algorithm-success slice norm must be finite")
            mantissa, exponent = frexp(norm)
            algorithm = PhysicalScale(mantissa=mantissa, exponent=exponent).squared_as_float()
        masses = (algorithm, physical)
    if readout.output.frame == "unit":
        if direction is None:
            return None, physical_scale, UnavailableOutput(output=readout.output,
                reason="unit vector is undefined for a zero selected branch"), masses
        return direction, physical_scale, None, masses
    result = recovery.apply_vector(selected)
    if result is None:
        return None, physical_scale, UnavailableOutput(output=readout.output,
            reason="selected physical vector is not representable in complex128"), masses
    return result, physical_scale, None, masses
