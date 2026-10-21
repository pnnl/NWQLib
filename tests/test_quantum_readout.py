"""Independent supplied-bit witnesses; no numerical construction or acquisition."""

import pytest

from nwqlib._quantum_readout import reduce_sample_arrays, reduce_setting, validate_readout_layout
from nwqlib.core import Source
from nwqlib.core.planning import ObservationSpec
from nwqlib.execution import ObservationChunk, RegisterMap
from nwqlib.ir import ClassicalValue

_IDENTITY = "sha256:" + "1" * 64


def _chunk(outcomes, observation, *, classical_layout=(), returned_shots=None):
    """A supplied probabilities or counts chunk; its identity fields are placeholders."""
    counts = observation.kind == "counts"
    return ObservationChunk.from_histogram(outcomes, run_id="r", plan_id=_IDENTITY, realization_id=_IDENTITY,
        prepared_id=_IDENTITY, experiment="e", setting="s", bindings=(), quantum_layout=(),
        classical_layout=classical_layout, attempt="a", job="j", chunk="c", observation=observation,
        population="unconditional", returned_shots=returned_shots, trajectories=None if counts else 1,
        source=Source(name="supplied", version="1", domain="test", reference="supplied outcomes"))


@pytest.mark.parametrize("counts", [False, True])
def test_valued_noncontiguous_flags_keep_algorithm_and_physical_mass(counts):
    success, conditions = ((4,1),(0,0)), ((3,1),)
    # Valued flags in success-then-condition order, then the parity pivot, physical bit 2.
    observed = (4,0,3,2)
    weights = (("0101",3),("1101",1),("1001",2),("1100",4))
    classical = (ClassicalValue(name="flags",dtype="bits",width=2),
                 ClassicalValue(name="result",dtype="bits",width=2))
    layout = (RegisterMap(name="flags",bits=(0,1)), RegisterMap(name="result",bits=(2,3)))
    chunk = _chunk({bits: n if counts else n/10 for bits, n in weights},
        ObservationSpec(kind="counts",shots=10) if counts else
        ObservationSpec(kind="probabilities",qubits=observed,population="unconditional"),
        classical_layout=layout if counts else (), returned_shots=10 if counts else None)
    validate_readout_layout(chunk, observed=observed, classical=classical)
    # Six shots satisfy algorithm flags; only four reach the physical slice,
    # with three positive and one negative parity: (6,4,3-1)/10.
    assert reduce_setting(chunk, observed=observed, success=success, conditions=conditions,
        pivot=2) == pytest.approx((.6,.4,.2), rel=0., abs=2e-16)
    if counts:
        # Classical positions cannot be relabeled as the measured physical bits.
        chunk = chunk.revise(classical_layout=(RegisterMap(name="flags",bits=(4,0)), RegisterMap(name="result",bits=(3,2))))
        with pytest.raises(ValueError, match="classical declarations"):
            validate_readout_layout(chunk, observed=observed, classical=classical)


def test_samples_use_original_coordinates_not_a_high_prefix():
    chunk = _chunk({"10011": 3, "11010": 1, "01011": 2, "11000": 4}, ObservationSpec(kind="counts",shots=12),
        classical_layout=(RegisterMap(name="c",bits=(0,1,2,3,4)),), returned_shots=10)
    # clbits0..4 contain physical bits(1,4,0,2,3), respectively. Dilation flag4=1
    # and flag0=0 give six algorithm shots; physical condition3=1 leaves four.
    indices, counts, algorithm, physical = reduce_sample_arrays(chunk, observed=(1,4,0,2,3), coordinates=(1,2),
        success=((4,1),(0,0)), conditions=((3,1),), dimension=4)
    assert (chunk.observation.shots, chunk.returned_shots, algorithm, physical) == (12,10,6,4)
    # Original coordinate index 1 (bits "01") has three shots and index 2 ("10") one.
    assert (indices.tolist(), counts.tolist()) == ([1,2], [3,1])


def test_padded_coordinate_mass_and_samples_share_original_space():
    chunk=_chunk(dict.fromkeys(("00","01","10","11"), 1), ObservationSpec(kind="counts",shots=4),
        classical_layout=(RegisterMap(name="c",bits=(0,1)),), returned_shots=4)
    mass=reduce_setting(chunk,observed=(0,1),success=(),coordinates=(0,1),dimension=3)
    assert mass==(1.,.75,.75)
    indices,counts,algorithm,physical=reduce_sample_arrays(chunk,observed=(0,1),coordinates=(0,1),success=(),dimension=3)
    assert (algorithm,physical)==(4,3) and indices.tolist()==[0,1,2] and counts.tolist()==[1,1,1]
    with pytest.raises(ValueError,match="basis rotation"):
        reduce_setting(chunk,observed=(0,1),success=(),pivot=1,coordinates=(0,1),dimension=3)


def test_histogram_layout_for_wide_and_single_word_readouts_and_the_count_limit():
    import numpy as np

    # Width 70 needs two index words. The key's rightmost character is classical
    # bit 0, so 1 followed by 64 zeros and then 101 is 2**64 + 5: word 0 holds 5
    # and word 1 holds 1. The order of the stored entries is kept.
    wide = (RegisterMap(name="c", bits=tuple(range(70))),)
    keys = {"0" * 5 + "1" + "0" * 61 + "101": 2, "0" * 68 + "11": 1}
    chunk = _chunk(keys, ObservationSpec(kind="counts", shots=3), classical_layout=wide, returned_shots=3)
    histogram = chunk.histogram()
    assert (histogram.width, histogram.entries) == (70, 2)
    assert histogram.weights.dtype == np.int64 and histogram.weights.tolist() == [2, 1]
    packed = histogram.packed_indices()
    assert packed.dtype == np.uint64 and packed.tolist() == [[5, 1], [3, 0]]
    assert histogram.index_list() == [2**64 + 5, 3]
    with pytest.raises(ValueError, match="width 70"):
        histogram.indices()
    with pytest.raises(ValueError):
        packed[0, 0] = 0
    with pytest.raises(ValueError):
        packed.flags.writeable = True
    with pytest.raises(AttributeError):
        histogram.width = 71
    rebuilt = _chunk((packed, histogram.weights), ObservationSpec(kind="counts", shots=3), classical_layout=wide,
                     returned_shots=3)
    assert rebuilt == chunk and rebuilt.content_id == chunk.content_id
    # A set bit above the width is rejected, also in the padding of the last word.
    with pytest.raises(ValueError, match="exceeds the readout width 70"):
        _chunk((np.array([[0, 1 << 6]], dtype=np.uint64), np.array([3])), ObservationSpec(kind="counts", shots=3),
               classical_layout=wide, returned_shots=3)
    empty = _chunk({}, ObservationSpec(kind="counts", shots=3), classical_layout=wide, returned_shots=0).histogram()
    assert empty.packed_indices().shape == (0, 2) and empty.weights.shape == (0,)
    untyped = _chunk((np.array([]), np.array([])), ObservationSpec(kind="counts", shots=3), classical_layout=wide,
                     returned_shots=0)
    assert untyped.histogram().entries == 0

    # Width 64 fits one word, including its largest index 2**64 - 1.
    single = (RegisterMap(name="c", bits=tuple(range(64))),)
    chunk = _chunk({"1" * 64: 1, "0" * 63 + "1": 1}, ObservationSpec(kind="counts", shots=2), classical_layout=single,
                   returned_shots=2)
    indices = chunk.histogram().indices()
    assert indices.dtype == np.uint64 and indices.tolist() == [2**64 - 1, 1]
    assert chunk.histogram().packed_indices().shape == (2, 1)
    empty = _chunk({}, ObservationSpec(kind="counts", shots=2), classical_layout=single, returned_shots=0).histogram()
    assert empty.indices().shape == (0,) and empty.packed_indices().shape == (0, 1)

    # Probability weights are float64 and bit 0 is the first observed qubit. A
    # marginal with 2*s >= D nonzero values is stored dense, by outcome
    # position with its zeros; one with 2*s < D sparse, in increasing index order.
    probabilities = _chunk({"10": .75, "01": .25}, ObservationSpec(kind="probabilities", qubits=(3, 1))).histogram()
    assert probabilities.weights.dtype == np.float64 and probabilities.weights.tolist() == [0., .25, .75, 0.]
    assert probabilities.indices().tolist() == [0, 1, 2, 3] and probabilities.width == 2
    sparse = _chunk({"110": .75, "001": .25}, ObservationSpec(kind="probabilities", qubits=(3, 1, 0)))
    assert sparse.values[0].encoding == "sparse" and sparse.values[0].entries == sparse.values[0].nonzero == 2
    assert sparse.histogram().indices().tolist() == [1, 6] and sparse.histogram().weights.tolist() == [.25, .75]

    # The count domain: 1 <= requested shots <= 2**63 - 1 at the declaration, and
    # at the chunk every count and the exact total within the requested shots.
    # The int64 maximum 2**63 - 1 is the largest admitted count and shot number.
    one = (RegisterMap(name="c", bits=(0,)),)
    limit = 2**63 - 1
    assert _chunk({"1": limit}, ObservationSpec(kind="counts", shots=limit), classical_layout=one,
                  returned_shots=limit).histogram().weights.tolist() == [limit]
    with pytest.raises(ValueError, match="int64 readout limit"):
        _chunk({"1": limit + 1}, ObservationSpec(kind="counts", shots=limit), classical_layout=one,
               returned_shots=limit + 1)
    with pytest.raises(ValueError, match=f"1 <= shots <= {limit}"):
        ObservationSpec(kind="counts", shots=2**64)
    with pytest.raises(ValueError, match="exact returned total within requested shots"):
        _chunk({"0": limit, "1": limit}, ObservationSpec(kind="counts", shots=limit), classical_layout=one,
               returned_shots=2 * limit)


def test_probability_arrays_check_values_at_publication_and_layout_and_provenance_when_loaded():
    """Publication refuses a negative probability entry."""
    import numpy as np

    marginal = ObservationSpec(kind="probabilities", qubits=(2, 0, 1))
    indices, weights = np.array([1, 6], dtype=np.uint64), np.array([.25, .75])
    weights[0] = -1e-17
    with pytest.raises(ValueError, match="nonnegative"):
        _chunk((indices, weights), marginal)


@pytest.mark.parametrize("success_count, same_mass, contiguous, expected", [
    # Mass passes at 16 visits per complex entry, 8F plus 16A for a success
    # selection, 16d for a separate physical scan, (4n+4)d for a gather and
    # 4w+L declaration checks, written phase by phase for F = 32 and n = 3.
    (8, False, True, 24*32 + 16*8 + 16*6 + 4*5),
    (8, True, False, 24*32 + 16*8 + 16*8 + 4*5),
])
def test_reducer_work_follows_the_executed_mass_and_gather_branches(success_count, same_mass, contiguous, expected):
    """``readout_requirements`` charges the branches ``reduce_scaled`` executes; omitted facts reserve more."""
    from nwqlib._quantum_readout import readout_requirements

    d = 6 if not same_mass and success_count == 8 else 8
    layout = readout_requirements(32, 3, d, 0, 0, has_moment=False, success_count=success_count,
                                  same_mass=same_mass, contiguous=contiguous)
    assert layout[1] == expected
    count_only = readout_requirements(32, 3, d, 0, 0, has_moment=False)
    assert count_only[0] >= layout[0] and count_only[1] >= layout[1]
    moment = readout_requirements(32, 3, d, 4, 128, success_count=success_count, same_mass=same_mass,
                                  contiguous=contiguous)
    full = readout_requirements(32, 3, d, 4, 128)
    assert full[0] >= moment[0] and full[1] >= moment[1]


def test_registry_reduction_of_a_thousand_terms_fits_its_charged_bytes(monkeypatch):
    """The parsed, packed and numerical phases of a 1000-term reduction fit ``projected_requirements``.

    Packing holds the parsed rows, their validated tuples and the packed
    table together. The callback then releases the parsed rows, so when the
    numerical phase starts the newly live memory (the packed table and freed
    tuples that CPython keeps for reuse) is below the parsed-object bytes H
    of the law. The saved state and the parameter text are allocated before
    tracing and added to the measured peak.
    """
    import gc
    import json
    import math
    import sys
    import tracemalloc
    import numpy as np
    import nwqlib._quantum_readout as readout
    from nwqlib.core.planning import ObservationPoint, ReductionContext, execute_reduction

    n, L = 12, 1000
    terms = tuple(("".join("IXYZ"[(j >> (2*k)) & 3] for k in reversed(range(n))), (-1.)**j/(j+1))
                  for j in range(1, L+1))
    point = ObservationPoint(id="end", kind="reduction", reducer=readout.PROJECTED_MOMENTS,
        parameters=readout.projected_parameters(coordinates=range(n), success=((n, 0),), dimension=1 << n, terms=terms))
    state = np.ones(1 << (n+1), dtype=np.complex128)/math.sqrt(1 << (n+1))
    size, _ = readout.projected_requirements(json.loads(point.parameters), n+1)
    execute_reduction(point, state, bindings=(), context=ReductionContext("r", 1e-12, ()))
    entry = []
    reduce_scaled = readout.reduce_scaled

    def numerical_phase(*args, **kwargs):
        entry.append(tracemalloc.get_traced_memory()[0])
        return reduce_scaled(*args, **kwargs)

    monkeypatch.setattr(readout, "reduce_scaled", numerical_phase)
    gc.collect()
    tracemalloc.start()
    try:
        execute_reduction(point, state, bindings=(), context=ReductionContext("r", 1e-12, ()))
        whole = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()
        fields = readout._projected_fields(json.loads(point.parameters))
        table = readout._pauli_table(fields[4], n)
        packing = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(table) == L and len(entry) == 1
    parsed = json.loads(point.parameters)
    rows, validated = parsed["terms"], readout._projected_fields(parsed)[4]
    H = sys.getsizeof(rows) + sum(sys.getsizeof(row) + sys.getsizeof(label) + sys.getsizeof(value)
                                  for row, (label, value) in zip(rows, rows, strict=True))
    H += sys.getsizeof(validated) + sum(sys.getsizeof(row) for row in validated)
    assert entry[0] < H
    assert whole + state.nbytes + len(point.parameters) + 64 <= size
    assert packing + state.nbytes + len(point.parameters) + 64 <= size


@pytest.mark.parametrize("w, n, L, bytes_, work", [
    (20, 18, 2, None, 41_160_944),
    (13, 12, 1000, 1_218_664, 8_818_988),
])
def test_projected_requirements_match_the_derivation_owner_evaluations(w, n, L, bytes_, work):
    """Scalar evaluations of the reduction law by its derivation owner, independent of this implementation.

    Native float coefficients, one row per kept term, contiguous unpadded
    coordinates and w - n success bits. A permuted coordinate order adds the
    gather's (4n + 4)d visits.
    """
    import json
    from nwqlib._quantum_readout import projected_parameters, projected_requirements

    labels = ["Z" * n] * L if L < 1000 else [
        "".join("IXYZ"[(j >> (2*k)) & 3] for k in reversed(range(n))) for j in range(1, L + 1)]
    terms = tuple((label, .5 if L < 1000 else (-1.)**j/(j+1)) for j, label in enumerate(labels, start=1))

    def law(coordinates):
        parameters = projected_parameters(coordinates=coordinates, success=tuple((b, 0) for b in range(n, w)),
                                          dimension=1 << n, terms=terms, moment=bool(L))
        return projected_requirements(json.loads(json.dumps(parameters)), w)

    size, visits = law(range(n))
    assert visits == work and bytes_ in (None, size)
    assert law(tuple(reversed(range(n))))[1] - visits == (4*n + 4) << n
