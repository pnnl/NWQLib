"""Number-sector diagnostics of actual stored computational-basis counts."""

from pydantic import model_validator

from nwqlib.core.records import Nonnegative, Record, Source, Text, Unit
from nwqlib.operators.access import Count
from ._work import DEFAULT_MAX_INTEGER_BITS, ExactArithmetic
from .error_model import CheckDomain, CheckSpec, ErrorFrame
from .verification import _publish


_SOURCE = Source(name="stored_number_sector", version="1",
    domain="empirical sector leakage in complete computational-register counts",
    reference="nwqlib.evidence.sector.verify_number_sector")


class NumberSectorOptions(Record):
    """Select a Hamming-weight sector of the actual complete measured population.

    The observed fraction outside this sector differs from mean particle number.
    Zero observed leakage does not prove a pure state's sector membership.

    Attributes:
        name: Prefix of the check name.
        particles: Hamming weight N of the selected sector.
        tolerance: Threshold on the observed fraction outside N, in [0,1].
        source: Implementation source of the count reducer.
    """

    name: Text
    particles: Count
    tolerance: Nonnegative
    source: Source = _SOURCE

    @model_validator(mode="after")
    def _criterion(self):
        if self.tolerance > 1 or self.source != _SOURCE:
            raise ValueError("number-sector tolerance is in [0,1] and source is the actual count reducer")
        return self

    def verification_checks(self, result):
        """Return a one-element tuple with the CheckSpec of observed leakage outside Hamming weight N.

        The checked quantity is the fraction of stored complete-register
        shots whose bitstring does not have exactly N ones, on the domain
        [0, 1], against ``tolerance``. The Result must supply its measured
        register width and count observations, and N may not exceed that
        width.
        """
        if any(not callable(getattr(result, name, None)) for name in
               ("number_sector_width", "number_sector_observations")):
            raise ValueError("number-sector verification requires a Result supplying complete "
                             "computational-register counts; parity or ancilla readout is insufficient")
        width = result.number_sector_width()
        if self.particles > width:
            raise ValueError("selected particle number exceeds the actual measured register")
        return (CheckSpec(name=self.name + ".number_sector_leakage", claim_id=result.plan.output.content_id,
            frame=ErrorFrame(quantity="observed_number_sector_leakage", metric="outside_sector_fraction",
                unit=Unit(symbol="1", dimension="dimensionless"), scope=result.plan.problem.evidence_scope,
                conditioning=f"stored complete-register counts outside N={self.particles}; empirical population only"),
            domain=CheckDomain(lower=0., upper=1.), source=self.source, options_id=self.content_id,
            threshold=self.tolerance, prerequisites=(), access=("actual complete-register counts used by this result",),
            experiments=0, classical_work="one bit-count scan over stored bins and exact integer totals",
            reference_work="none; no state preparation, projector action or S-squared calculation",
            data_description="empirical leakage scalar and original result/observation identities"),)


def _exact_count_sum(counts):
    """Return the exact integer sum of nonnegative int64 counts, without int64 wraparound.

    Each count c is below 2**63 and equals (c >> 32) * 2**32 + (c & (2**32 - 1)).
    Each half is below 2**32, so a uint64 sum of fewer than 2**32 halves
    cannot wrap; the two sums are then combined as Python integers. A
    larger population is summed as Python integers.
    """
    import numpy as np

    if counts.size >= 2**32:
        return sum(counts.tolist())
    low = int(np.bitwise_and(counts, 0xFFFFFFFF).sum(dtype=np.uint64))
    high = int(np.right_shift(counts, 32).sum(dtype=np.uint64))
    return (high << 32) + low


def verify_number_sector(result, *, options: NumberSectorOptions, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
    """Read actual selected count bins without rerunning analysis or native work.

    The particle number of an outcome is the set-bit count of its index in
    the observation reader's layout, over all words for a register wider
    than 64 bits. ``total`` and ``outside`` are exact integer sums of the
    counts, and the published value ``outside/total`` uses exact arithmetic.
    """
    import numpy as np

    if type(options) is not NumberSectorOptions:
        raise TypeError("number-sector verification requires concrete selected options")
    checks = options.verification_checks(result)
    chunks = result.number_sector_observations()
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    total = outside = 0
    width = result.number_sector_width()
    for chunk in chunks:
        if chunk.observation.kind != "counts" or chunk.population != "unconditional":
            raise ValueError("number-sector check needs actual unconditional counts")
        histogram = chunk.histogram()
        if histogram.entries and histogram.width != width:
            raise ValueError("number-sector bins require the complete computational register")
        indices = histogram.indices() if histogram.width <= 64 else histogram.packed_indices()
        particles = np.bitwise_count(indices)
        if particles.ndim == 2:
            particles = particles.sum(axis=1, dtype=np.int64)
        chunk_total = _exact_count_sum(histogram.weights)
        if chunk_total != chunk.returned_shots:
            raise ValueError("number-sector total differs from the actual returned shots")
        outside += _exact_count_sum(histogram.weights[particles != options.particles])
        total += chunk_total
    value = None if total == 0 else arithmetic.divide(arithmetic.fraction(outside), arithmetic.fraction(total))
    return _publish(result, options, checks, (value,), max_integer_bits=max_integer_bits)
