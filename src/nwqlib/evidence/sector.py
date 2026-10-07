"""Number-sector diagnostics of actual stored computational-basis counts."""

from typing import ClassVar

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
    """Options of the number-sector check: the fraction of measured shots outside Hamming weight N.

    Build it with keyword arguments, for example
    `NumberSectorOptions(name="number", particles=1, tolerance=0.01)`, and
    pass it to `result.verify(checks=...)` or
    [`verify_number_sector`][nwqlib.evidence.sector.verify_number_sector].
    `name`, `particles` and `tolerance` are required. The checked value is
    the fraction of stored complete-register shots whose bitstring does not
    have exactly N ones, on `[0, 1]`, against `tolerance`. It needs a Result
    that stores complete computational-register counts, such as QHD with the
    one-hot encoding ([verification
    guide](../verification.md#measured-number-sector-leakage)).

    This observed fraction differs from the mean particle number, and zero
    observed leakage does not prove that a pure state lies in the sector.
    The guide gives a counterexample for the mean.

    Attributes:
        name: Required. Prefix of the check name, which is
            `name + ".number_sector_leakage"`.
        particles: Required. Hamming weight N of the sector, at most the
            measured register width.
        tolerance: Required. Threshold on the observed fraction outside N,
            in `[0, 1]`.

    Raises:
        ValueError: If `tolerance` exceeds 1.
    """

    name: Text
    particles: Count
    tolerance: Nonnegative
    source: ClassVar[Source] = _SOURCE

    @model_validator(mode="after")
    def _criterion(self):
        if self.tolerance > 1:
            raise ValueError("number-sector tolerance must lie in [0, 1]")
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
    cannot wrap. The two sums are then combined as Python integers. A
    larger population is summed as Python integers.
    """
    import numpy as np

    if counts.size >= 2**32:
        return sum(counts.tolist())
    low = int(np.bitwise_and(counts, 0xFFFFFFFF).sum(dtype=np.uint64))
    high = int(np.right_shift(counts, 32).sum(dtype=np.uint64))
    return (high << 32) + low


def verify_number_sector(result, *, options: NumberSectorOptions, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
    """Run the number-sector check on a Result's stored counts, running no analysis or circuit again.

    `result.verify(checks=options)` calls this for QHD. The particle number
    of an outcome is the number of set bits of its index, over all words
    for a register wider than 64 bits. The total and the outside count are
    exact integer sums of the counts, and the value `outside/total` is
    exact.

    Args:
        result (Result): A Result that stores complete computational-register
            counts, with its Plan.
        options (NumberSectorOptions): The sector and tolerance.
        max_integer_bits (int): Bit limit of the exact arithmetic. Default
            4096.

    Returns:
        verification (tuple): `(receipt, facts)`: the `VerificationReceipt`
            of this computation, and a one-element tuple with the leakage
            fraction as a [`FramedFact`][nwqlib.evidence.error_model.FramedFact]
            that cites it. The fact is unknown when no shots were counted.

    Raises:
        TypeError: If `options` is not a `NumberSectorOptions`.
        ValueError: If the Result supplies no complete-register counts,
            `particles` exceeds the register width, the stored counts are
            not unconditional counts of the whole register that sum to the
            returned shots, or an exact intermediate value needs more than
            `max_integer_bits` bits.
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
