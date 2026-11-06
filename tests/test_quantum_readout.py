"""Independent supplied-bit witnesses; no numerical construction or acquisition."""

import pytest

from nwqlib._quantum_readout import physical_shot_count, reduce_sample_arrays, validate_readout_layout
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
def test_valued_noncontiguous_flags_keep_layout_and_physical_shots(counts):
    success, conditions = ((4,1),(0,0)), ((3,1),)
    # Valued flags in success-then-condition order, then physical bit 2.
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
    if counts:
        # Six of the ten returned shots satisfy the two valued flags, and four of them also the third.
        assert physical_shot_count(chunk, observed=observed, success=success) == 6
        assert physical_shot_count(chunk, observed=observed, success=success+conditions) == 4
        # Classical positions cannot be relabeled as the measured physical bits.
        chunk = chunk.revise(classical_layout=(RegisterMap(name="flags",bits=(4,0)), RegisterMap(name="result",bits=(3,2))))
        with pytest.raises(ValueError, match="classical declarations"):
            validate_readout_layout(chunk, observed=observed, classical=classical)
    else:
        with pytest.raises(ValueError, match="selected physical bit order"):
            validate_readout_layout(chunk, observed=observed[::-1], classical=classical)


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


def test_padded_coordinate_shots_and_samples_share_original_space():
    chunk=_chunk(dict.fromkeys(("00","01","10","11"), 1), ObservationSpec(kind="counts",shots=4),
        classical_layout=(RegisterMap(name="c",bits=(0,1)),), returned_shots=4)
    # All four shots without the projection. Dimension 3 excludes coordinate 3.
    assert physical_shot_count(chunk,observed=(0,1),success=())==4
    assert physical_shot_count(chunk,observed=(0,1),success=(),coordinates=(0,1),dimension=3)==3
    indices,counts,algorithm,physical=reduce_sample_arrays(chunk,observed=(0,1),coordinates=(0,1),success=(),dimension=3)
    assert (algorithm,physical)==(4,3) and indices.tolist()==[0,1,2] and counts.tolist()==[1,1,1]
    # Coordinate 3 ("011") is inside dimension 5, and its bit reversal, coordinate 6 ("110"), is not.
    chunk=_chunk({"011":1,"110":2}, ObservationSpec(kind="counts",shots=3),
        classical_layout=(RegisterMap(name="c",bits=(0,1,2)),), returned_shots=3)
    assert physical_shot_count(chunk,observed=(7,5,2),success=(),coordinates=(7,5,2),dimension=5)==1


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
    # Only the first entry, with its two shots, has bit 64 set.
    assert physical_shot_count(chunk, observed=tuple(range(70)), success=((64, 1),)) == 2
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


@pytest.mark.parametrize("n, rows, contiguous, step", [(12, 1000, True, 1), (12, 1000, False, 1), (14, 4, False, 1),
                                                       (12, 1000, False, 2654435761)])
def test_projected_reduction_arrays_fit_its_registered_workspace(monkeypatch, n, rows, contiguous, step):
    """The arrays of a projected reduction stay within ``49*2**n + 320*M + 2**17`` bytes.

    tracemalloc follows NumPy's allocations and Python's containers. The
    peak traced memory during ``reduce_scaled`` above its entry, plus the
    packed table that is live through it, stays below the registered
    workspace, for contiguous coordinates and for reversed ones whose
    physical slice is gathered, with many rows and with few rows on more
    coordinates. Row j has the letters of ``j*step`` in base 4. With
    ``step=1`` the 1000 rows share 32 flip masks, and the odd multiplier
    gives most rows a flip mask of their own, so the Pauli action keeps
    hundreds of groups.
    """
    import gc
    import json
    import math
    import tracemalloc
    import numpy as np
    import nwqlib._quantum_readout as readout
    from nwqlib.core.planning import ObservationPoint, ReductionContext, execute_reduction

    terms = tuple(("".join("IXYZ"[(j*step >> (2*k)) & 3] for k in reversed(range(n))), (-1.)**j/(j+1))
                  for j in range(1, rows+1))
    # A letter X or Y flips its qubit.
    flips = {label.translate(str.maketrans("IXYZ", "0110")) for label, _ in terms}
    if rows == 1000:
        assert len(flips) == 32 if step == 1 else len(flips) > rows // 2
    coordinates = tuple(range(n)) if contiguous else tuple(reversed(range(n)))
    point = ObservationPoint(id="end", kind="reduction", reducer=readout.PROJECTED_MOMENTS,
        parameters=readout.projected_parameters(coordinates=coordinates, success=((n, 0),), dimension=1 << n,
                                                terms=terms))
    state = np.ones(1 << (n+1), dtype=np.complex128)/math.sqrt(1 << (n+1))
    execute_reduction(point, state, bindings=(), context=ReductionContext("r", 1e-12, ()))
    measured, reduce_scaled = [], readout.reduce_scaled

    def traced(*args, table, **kwargs):
        entry = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        result = reduce_scaled(*args, table=table, **kwargs)
        packed = table.x.nbytes + table.z.nbytes + table.coefficients.nbytes
        measured.append(tracemalloc.get_traced_memory()[1] - entry + packed)
        return result

    monkeypatch.setattr(readout, "reduce_scaled", traced)
    gc.collect()
    tracemalloc.start()
    try:
        execute_reduction(point, state, bindings=(), context=ReductionContext("r", 1e-12, ()))
    finally:
        tracemalloc.stop()
    workspace = readout._projected_workspace(json.loads(point.parameters), n + 1)
    assert workspace == 49 * (1 << n) + 320 * rows + (1 << 17)
    assert len(measured) == 1 and 0 < measured[0] <= workspace


def test_ordered_action_constants_equal_the_formula_per_dimension_and_term_count():
    """A repeated (N, L) gets the same exact constants, other inputs get their own, and a refused domain still raises.

    The cached values equal the uncached formula for (8, 5), (16, 5),
    (8, 6) and the empty action (8, 0). L = 2**52 has (2L + 4)u >= 1 and is
    refused at every call.
    """
    from fractions import Fraction
    from nwqlib._quantum_readout import ordered_action_constants

    inputs = ((8, 5), (16, 5), (8, 6), (8, 0))
    for n, terms in inputs * 2:
        assert ordered_action_constants(n, terms) == ordered_action_constants.__wrapped__(n, terms)
    assert len({ordered_action_constants(n, terms) for n, terms in inputs}) == len(inputs)
    assert ordered_action_constants(8, 0) == (Fraction(0), Fraction(0))
    for _ in range(2):
        with pytest.raises(ValueError, match="invalid action underflow domain"):
            ordered_action_constants(8, 2**52)
