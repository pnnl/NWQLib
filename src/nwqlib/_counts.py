"""Sampling identities from kept preparation and actual job coordinates.

This records known reuse, not circuit equivalence or statistical independence.
There is no native traversal, payload hash, numerical work or acquisition here.
"""

from dataclasses import dataclass
from hashlib import sha256

from nwqlib.execution import PreparedArtifact
from nwqlib.operators.access import DEFAULT_INPUT_BYTES


@dataclass(frozen=True)
class CountsSource:
    """Identity of one counts population and of its fixed sampling stream, if any.

    ``kind`` is ``fixed_seed`` when the preparation receipt fixes the sampler
    seed, ``unknown`` when sampling semantics are not recorded, or the
    receipt's own sampling kind otherwise.
    """

    identity: str
    stream: str | None
    kind: str


class CountsSources:
    """Bounded sampling identities for one Run's admission or one reduction.

    Likelihood-based estimators multiply the likelihoods of counts
    populations and so assume those populations are independent. A backend
    sampler with a fixed seed returns the same counts for the same circuit
    and context. Feeding that population in twice, or two queries sharing one
    seeded stream, would count one set of samples as independent evidence
    more than once and overstate confidence. This owner derives a sampling
    identity from the preparation receipt (seed, backend configuration,
    target, compiler, environment and the prepared point), refuses a new
    submission whose stream is already used, and refuses a reduction that
    would count a population twice. Re-reading the very same acquisition,
    for example during pending recovery, stays legal. Consumers are the Run's
    submission admission, expectation counts reduction and the QPE likelihood
    controllers.
    """

    def __init__(self, receipt_for, acquisition_for, *, max_bytes=DEFAULT_INPUT_BYTES):
        self.receipt_for, self.acquisition_for = receipt_for, acquisition_for
        self.max_bytes = max_bytes
        self.preparations = {}
        self.counted, self.streams = {}, {}

    def _identity(self, fields):
        """Return "sha256:" plus the hex digest of the _run_journal JSON encoding of fields."""
        from nwqlib._run_journal import encode
        # Bound the portable scalar JSON plus UTF-8 copy before encoding. The
        # Run owns cumulative receipt/data storage; this is no native hash.
        encoded = encode(fields, self.max_bytes // 3)
        return "sha256:" + sha256(encoded.encode("utf-8")).hexdigest()

    def _preparation(self, prepared_id):
        """Return the cached CountsSource that one preparation receipt determines.

        A fixed-seed receipt yields a stream identity (seed, backend
        configuration, target, compiler and environment) and a population
        identity that adds the prepared point, construction, observation,
        native basis, logical-to-native map and logical and native layouts.
        Preparations with equal population identities are one population,
        since the fixed seed reproduces their counts. Without a receipt the
        kind is ``unknown``. Otherwise the receipt's own sampling kind is kept,
        and resolve derives the identity from the actual acquisition. Raises
        ValueError for a receipt that is not a counts preparation.
        """
        cached = self.preparations.get(prepared_id)
        if cached is None:
            receipt = self.receipt_for(prepared_id)
            if receipt is not None and (type(receipt) is not PreparedArtifact or receipt.observation.kind != "counts"):
                raise ValueError("counts source requires its actual counts preparation receipt")
            if receipt is None or receipt.counts_sampling.kind != "fixed_seed":
                cached = CountsSource("", None, "unknown" if receipt is None else receipt.counts_sampling.kind)
            else:
                # Only effective seed and actual producer context determine the
                # stream. Runtime ancestry and fresh preparation UUIDs do not.
                stream = self._identity(("nwqlib.counts-stream/1", receipt.counts_sampling.seed,
                    receipt.backend_configuration_id, receipt.target, receipt.compiler, receipt.environment))
                identity = self._identity(("nwqlib.seeded-counts/1", stream,
                    receipt.realization_id, receipt.construction_id, receipt.observation,
                    receipt.native_basis, receipt.logical_to_native,
                    receipt.native_quantum_layout, receipt.native_classical_layout,
                    receipt.quantum_layout, receipt.classical_layout))
                cached = CountsSource(identity, stream, "fixed_seed")
            self.preparations[prepared_id] = cached
        return cached

    def admit_preparation(self, prepared_id):
        """Reject already known unusable likelihood input before a new submit."""
        source = self._preparation(prepared_id)
        if source.kind == "unknown":
            raise ValueError("independent count likelihoods require known sampling preparation semantics")
        if source.kind == "fixed_seed" and (source.identity in self.counted or source.stream in self.streams):
            raise ValueError("one fixed sampling stream cannot supply another independent likelihood population")

    def reserve_preparation(self, prepared_id):
        """Keep a committed fixed stream, including unused or uncertain attempts.

        Reservation does not claim an observed population. The actual completed
        acquisition is still joined and checked by require_independent.
        """
        source = self._preparation(prepared_id)
        if source.kind == "fixed_seed":
            self.streams.setdefault(source.stream, source.identity)

    def resolve(self, prepared_id, attempt):
        """Return the CountsSource of one attempt of a preparation.

        A fixed-seed preparation is its own source. Any other preparation is
        identified by the provider coordinates of the actual acquisition
        (provider, account, instance, project, region, cluster, host, job,
        child job, item and result key), so that one preparation can supply
        several fresh populations. An attempt without a provider locator is an
        unresolved gap of kind ``unknown``. Raises ValueError when the
        attempt's submission item names another preparation.
        """
        cached = self._preparation(prepared_id)
        if cached.kind == "fixed_seed":
            return cached
        submission, item = self.acquisition_for(attempt)
        if item.prepared_id != prepared_id:
            raise ValueError("counts source differs from its actual submission item")
        # An unresolved intent is a gap, never a fresh observed population.
        if submission.locator is None:
            return CountsSource(self._identity(("nwqlib.unresolved-counts/1", submission.run_id, attempt)), None, "unknown")
        locator = submission.locator
        identity = self._identity(("nwqlib.acquired-counts/1", locator.provider, locator.account,
            locator.instance, locator.project, locator.region, locator.cluster, locator.host,
            locator.job_id, item.child_job, item.item, item.result_key))
        return CountsSource(identity, None, cached.kind)

    def observation(self, chunk):
        """Return the CountsSource of an observed chunk after checking its receipt.

        When a receipt exists, it must record the chunk's Plan, point,
        observation, layouts and population. Raises ValueError otherwise.
        """
        receipt = self.receipt_for(chunk.prepared_id)
        if receipt is not None and (receipt.plan_id != chunk.plan_id or receipt.realization_id != chunk.realization_id
                or receipt.observation != chunk.observation or receipt.quantum_layout != chunk.quantum_layout
                or receipt.classical_layout != chunk.classical_layout or receipt.population != chunk.population):
            raise ValueError("counts preparation differs from its actual observed point/readout")
        return self.resolve(chunk.prepared_id, chunk.attempt)

    def require_independent(self, chunk):
        """Refuse known reused/unknown likelihood data before a caller consumes it.

        Re-reading the very same acquisition is legal for pending recovery. It
        does not authorize a second likelihood update; the controller owns that
        saved transition. Distinct seeds/acquisitions still need model premises.
        """
        source = self.observation(chunk)
        if source.kind == "unknown":
            raise ValueError("independent count likelihoods require actual sampling preparation receipts")
        previous = self.counted.get(source.identity)
        if previous is not None and previous != chunk.acquisition_key:
            raise ValueError("one frozen counts source cannot supply independent likelihood populations")
        if source.stream is not None and self.streams.get(source.stream, source.identity) != source.identity:
            raise ValueError("different queries sharing one fixed sampling stream cannot supply independent likelihood populations")
        self.counted[source.identity] = chunk.acquisition_key
        if source.stream is not None:
            self.streams[source.stream] = source.identity
