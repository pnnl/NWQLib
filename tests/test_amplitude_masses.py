"""Selected and wider masses from tiny supplied arrays; no native execution."""

import numpy as np
import pytest

from nwqlib.amplitudes import AmplitudeReadout, reduce_amplitudes
from nwqlib.artifacts import ArrayOutput
from nwqlib.core import Basis, Source
from nwqlib.problems.inputs import compose_recovery

SOURCE=Source(name="supplied_mass_witness",version="1",domain="tiny supplied amplitudes",reference="fixed algebra")


def readout(*,width=5,coordinates=(1,2),success=((0,0),(4,1)),conditions=((3,1),),recovery=None,dimension=None):
    return AmplitudeReadout(construction_id="sha256:"+"0"*64,source=SOURCE,width=width,
        coordinates=coordinates,success=success,conditions=conditions,
        output=ArrayOutput(name="solution",kind="vector",basis=Basis(identity="coordinates",dimension=dimension or 1<<len(coordinates),
            ordering="LSB first"),frame="unit",global_phase="physical"),recovery=recovery,
        keep_masses=True)


def test_wider_and_physical_mass_share_selected_norm(monkeypatch):
    import nwqlib._numerics as numerics
    chosen=readout(recovery=compose_recovery(4.))
    native=np.zeros(32,dtype=np.complex128)
    native[24],native[26]=.3,.4j
    native[16]=.5
    native[1]=np.sqrt(.5)
    calls=[]
    original=numerics.stable_vector_norm
    def counted(value):
        calls.append((value.size,np.shares_memory(value,native)))
        return original(value)
    monkeypatch.setattr(numerics,"stable_vector_norm",counted)
    vector,scale,absent,masses=reduce_amplitudes(chosen,native)
    assert calls==[(4,True),(8,True)]  # Both middle-register selectors are views.
    assert masses==pytest.approx((.5,.25),abs=2e-16,rel=0)
    np.testing.assert_allclose(vector,[.6,.8j,0,0],rtol=0,atol=2e-16)
    assert scale is not None
    assert scale.squared_as_float()==4.


def test_nonfinite_amplitudes_are_rejected_once_and_other_saved_arrays_still_checked():
    from nwqlib.artifacts import ArtifactStore
    chosen=readout(width=2,coordinates=(0,1),success=(),conditions=(),recovery=compose_recovery(4.))
    native=np.array([.5,np.nan,0,0],dtype=np.complex128)
    for frame in ("unit","physical"):
        with pytest.raises(ValueError,match="selected native amplitudes must be finite"):
            reduce_amplitudes(chosen.revise(output=chosen.output.revise(frame=frame)),native)
    # Only the amplitude and probability readouts state that their values are checked.
    with pytest.raises(ValueError,match="artifact payload must be finite"):
        ArtifactStore()._publish(native,output=chosen.output,provenance={})


def test_original_coordinate_order_and_explicit_materialization_limit():
    chosen=readout(coordinates=(2,1))
    native=np.arange(32,dtype=np.complex128)
    np.testing.assert_array_equal(chosen.select(native),native[[24,28,26,30]])
    assert not np.shares_memory(chosen.select(native),native)
    # The explicit array operation owns its limit; selecting a readout is cheap.
    with pytest.raises(ValueError,match="max_bytes=16"):
        chosen.admit_materialization(max_bytes=16)


def test_nonpower_two_projection_excludes_nonzero_dummy_amplitude():
    chosen=readout(width=2,coordinates=(0,1),success=(),conditions=(),
                   dimension=3,recovery=compose_recovery(4.))
    native=np.array([.5,.5j,-.5,.5],dtype=np.complex128)
    vector,scale,absent,masses=reduce_amplitudes(chosen,native)
    np.testing.assert_allclose(vector,np.array([1,1j,-1])/np.sqrt(3),rtol=0,atol=2e-16)
    assert absent is None and masses==pytest.approx((1.,.75),rel=0,abs=3e-16)
    assert scale.squared_as_float()==pytest.approx(12., rel=0, abs=2e-14)
    physical=chosen.revise(output=chosen.output.revise(frame="physical"))
    np.testing.assert_array_equal(reduce_amplitudes(physical,native)[0],[2,2j,-2])


def test_physical_scale_matches_exact_rounding_on_both_routes():
    """apply_vector equals the correctly rounded exact m*2**e*x, or is None when a part overflows or rounds to zero."""
    from fractions import Fraction
    import math
    from nwqlib.problems.inputs import PhysicalScale

    def exact(x, m, e):
        value = Fraction(x) * Fraction(m) * Fraction(2) ** e
        try:
            rounded = float(value)
        except OverflowError:
            return None
        return None if math.isinf(rounded) or (value and not rounded) else rounded

    tiny = math.ldexp(1.0, -60)
    cases = [(.625, 3, [1 + .5j, -.25 + 3j]),             # ordinary scale above one
             (.75, -1000, [tiny + 3 * tiny * 1j]),         # normal scale, subnormal result
             (.5, -1000, [math.ldexp(1.0, -80) + 1j]),     # a nonzero part rounds to zero
             (.75, 1024, [2.0 + 0j]),                      # largest normal scale overflows
             (.5, -1021, [1.0 - 1j]),                      # smallest normal scale
             (.5, 1101, [math.ldexp(1.0, -110) + 1j * math.ldexp(1.0, -120)]),  # decomposition route
             (.5, -1074, [.125j]),
             # Halfway in the rounded mantissa product, just above half the smallest subnormal exactly.
             (math.nextafter(1.0, 0.0), -1074, [complex(math.nextafter(.5, 1.0), -math.nextafter(.5, 1.0))]),
             (math.nextafter(1.0, 0.0), -1074, [.5 + .5j]),
             # Zero input on each route, and a scale beyond the binary64 span.
             (.75, 3, [0j]), (.5, -1000, [0j]), (.5, 1101, [0j]),
             (.5, 5000, [0j, 0j]), (.5, -5000, [0j]), (.5, 5000, [1j]), (.5, -5000, [1 + 0j])]
    for m, e, raw in cases:
        vector = np.array(raw, dtype=np.complex128)
        parts = [(exact(z.real, m, e), exact(z.imag, m, e)) for z in vector]
        expected = None if any(None in pair for pair in parts) else np.array([complex(*pair) for pair in parts])
        result = PhysicalScale(mantissa=m, exponent=e).apply_vector(vector)
        assert (result is None) == (expected is None)
        assert result is None or np.array_equal(result, expected)
    # Scales outside the normal range, one part at a time, with results around the smallest
    # normal for a small scale and around the largest finite value for a large one.
    rng = np.random.default_rng(31)
    for exponent in (-1080, -1074, -1060, -1030, 1030):
        target, spread = (-1024, 60) if exponent < 0 else (1024, 8)
        for _ in range(200):
            m = float(rng.uniform(.5, 1.0))
            x = math.ldexp(float(rng.uniform(-1.0, 1.0)), target - exponent + int(rng.integers(-spread, spread)))
            expected = exact(x, m, exponent)
            result = PhysicalScale(mantissa=m, exponent=exponent).apply_vector(np.array([complex(x, 0.0)]))
            assert (result is None) == (expected is None)
            assert result is None or result[0].real == expected
