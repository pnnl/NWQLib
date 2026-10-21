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
