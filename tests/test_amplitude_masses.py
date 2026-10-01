"""Selected and wider masses from tiny supplied arrays; no native execution."""

import numpy as np
import pytest

from nwqlib.amplitudes import AmplitudeReadout, reduce_amplitudes, validate_amplitude_masses
from nwqlib.artifacts import ArrayOutput
from nwqlib.core import Basis, Source
from nwqlib.execution import ScalarValue
from nwqlib.problems.inputs import compose_recovery

SOURCE=Source(name="supplied_mass_witness",version="1",domain="tiny supplied amplitudes",reference="fixed algebra")


def readout(*,width=5,coordinates=(1,2),success=((0,0),(4,1)),conditions=((3,1),),recovery=None,dimension=None):
    return AmplitudeReadout(construction_id="sha256:"+"0"*64,source=SOURCE,width=width,
        coordinates=coordinates,success=success,conditions=conditions,
        output=ArrayOutput(name="solution",kind="vector",basis=Basis(identity="coordinates",dimension=dimension or 1<<len(coordinates),
            ordering="LSB first"),frame="unit",global_phase="physical"),recovery=recovery,
        keep_masses=True)


@pytest.mark.parametrize("physical_zero",[False,True])
@pytest.mark.parametrize("known_scale",[False,True])
def test_wider_and_physical_mass_share_selected_norm(monkeypatch,physical_zero,known_scale):
    import nwqlib._numerics as numerics
    chosen=readout(recovery=compose_recovery(4.) if known_scale else None)
    native=np.zeros(32,dtype=np.complex128)
    if not physical_zero:
        native[24],native[26]=.3,.4j
    native[16]=.5
    native[1]=np.sqrt(.75 if physical_zero else .5)
    calls=[]
    original=numerics.stable_vector_norm
    def counted(value):
        calls.append((value.size,np.shares_memory(value,native)))
        return original(value)
    monkeypatch.setattr(numerics,"stable_vector_norm",counted)
    vector,scale,absent,masses=reduce_amplitudes(chosen,native)
    assert calls==[(4,True),(8,True)]  # Both middle-register selectors are views.
    assert masses==pytest.approx((.25 if physical_zero else .5,0. if physical_zero else .25),abs=2e-16,rel=0)
    if physical_zero:
        assert vector is None and "zero selected" in absent.reason
    else:
        np.testing.assert_allclose(vector,[.6,.8j,0,0],rtol=0,atol=2e-16)
    assert (scale is not None)==known_scale
    if known_scale:
        assert scale.squared_as_float()==(0. if physical_zero else 4.)


def test_no_conditions_reuse_and_zero_coordinate_view(monkeypatch):
    import nwqlib._numerics as numerics
    chosen=readout(width=2,coordinates=(),success=((0,0),(1,1)),conditions=())
    native=np.array([0,0,.5,np.sqrt(.75)],dtype=np.complex128)
    calls=[]
    original=numerics.stable_vector_norm
    monkeypatch.setattr(numerics,"stable_vector_norm",lambda value:(calls.append(value.size),original(value))[1])
    assert np.shares_memory(chosen.select(native),native)
    vector,_,absent,masses=reduce_amplitudes(chosen,native)
    np.testing.assert_array_equal(vector,[1.])
    assert calls==[1] and masses==(.25,.25) and absent is None
    tiny=np.nextafter(0.,1.)
    _,_,_,masses=reduce_amplitudes(chosen,np.array([0,0,tiny,1],dtype=np.complex128))
    assert masses==(None,None)  # Nonzero square underflow is not zero mass.


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


def test_exact_selected_mass_fields_and_domain():
    chosen=readout()
    def values(algorithm,physical):
        return tuple(ScalarValue(label=label,value=value,frame="encoded_branch",
            unavailable="unrepresentable" if value is None else None) for label,value in
            (("algorithm_success_mass",algorithm),("physical_slice_mass",physical)))
    # Alone, each mass is only nonnegative, and the physical subset stays within the summation
    # window of the algorithm mass. The unit bound follows the executed circuit's receipt.
    for algorithm,physical in ((.5,.25),(1.+5e-13,1.+5e-13),(.5,.5+5e-13),(None,None),
                               (None,1.+5e-13),(1.,None),(1.1,.5),(None,2.),(1.+.75e-12,1.+1.5e-12)):
        validate_amplitude_masses(chosen,values(algorithm,physical))
    for algorithm,physical in ((-.1,0.),(.2,.3),(None,-1e-300)):
        with pytest.raises(ValueError,match="mass"):
            validate_amplitude_masses(chosen,values(algorithm,physical))
    with pytest.raises(ValueError,match="ordered scalar"):
        validate_amplitude_masses(chosen,values(.5,.25)[::-1])
    with pytest.raises(ValueError,match="ordered scalar"):
        validate_amplitude_masses(chosen,(values(.5,.25)[0].revise(frame="unit"),values(.5,.25)[1]))
    with pytest.raises(ValueError,match="did not select"):
        validate_amplitude_masses(chosen.revise(keep_masses=False),values(.5,.25))


def test_amplitude_mass_unit_bound_follows_its_circuit_receipt():
    """The same stored branch mass passes with its executed circuit's roundoff window and fails
    with the window of a shallower receipt.
    """
    import nwqlib
    from nwqlib import LinearSystem
    from nwqlib._validation import NUMERICAL_RELATION_RTOL
    from nwqlib.algorithms.qls import QLS
    from nwqlib.operators import ingest_pauli

    problem = LinearSystem(A=ingest_pauli((("I", 1.), ("X", .1), ("Z", .1)), num_qubits=1), b=[1, .25])
    result = nwqlib.solve(problem, method=QLS(alpha=1.2, kappa=1.4), execution="quantum")
    chunk = next(item for item in result.data.observations.chunks if item.observation.kind == "amplitudes")
    receipt = next(item for item in result.data.receipts if item.content_id == chunk.prepared_id)
    window = receipt.saved_state_probability_window
    assert window > 4 * NUMERICAL_RELATION_RTOL
    def masses(value):
        return tuple(item.revise(value=value) for item in chunk.values)
    chunk.revise(values=masses(1 + window / 2)).validate_unit_bound(receipt)
    with pytest.raises(ValueError, match="roundoff window of its execution"):
        chunk.revise(values=masses(1 + 2 * window)).validate_unit_bound(receipt)
    shallow = receipt.revise(native_operations=0)
    with pytest.raises(ValueError, match="roundoff window of its execution"):
        chunk.revise(values=masses(1 + window / 2), prepared_id=shallow.content_id).validate_unit_bound(shallow)
    with pytest.raises(ValueError, match="its own preparation receipt"):
        chunk.validate_unit_bound(shallow)


def _saved_state_receipt(envelope):
    """A one-qubit amplitude receipt whose saved state carries the host-correction ``envelope``."""
    from nwqlib.core.planning import ObservationSpec, Realization, RuntimeOptions
    from nwqlib.execution import LogicalPreparationReceipt, PreparedArtifact, RegisterMap

    identity="sha256:"+"0"*64
    realization=Realization(plan_id=identity,experiment="mass")
    declaration=readout(width=1,coordinates=(0,),success=(),conditions=(),dimension=2)
    return PreparedArtifact(plan_id=identity,realization_id=realization.content_id,realization=realization,
        construction_id=identity,snapshot="saved-state mass contract",target=SOURCE,compiler=SOURCE,
        runtime=RuntimeOptions(seed=31),observation=ObservationSpec(kind="amplitudes",amplitudes=declaration),
        quantum_layout=(RegisterMap(name="q",bits=(0,)),),classical_layout=(),native_basis=("u","cx"),
        environment=(),preparation_time="2026-09-30T23:44:00Z",construction_work_reserved=0,
        logical=LogicalPreparationReceipt(dynamic_visits=0,reserved_work=0,defined_selections=()),native_operations=0,
        transformation="conditional saved-state phase product",population="unconditional",
        native_quantum_layout=(RegisterMap(name="q",bits=(0,)),),logical_to_native=(0,),
        probability_window_exclusions=("amplitude-derived masses","unitary"),statevector_roundoff=envelope)


def _mass_chunk(receipt,mass):
    from nwqlib.amplitudes import AMPLITUDE_MASS_LABELS
    from nwqlib.artifacts import ArtifactManifest
    from nwqlib.execution import ObservationChunk

    output=receipt.observation.amplitudes.output
    manifest=ArtifactManifest(output=output,digest=receipt.plan_id,data_bytes=output.data_bytes,encoding=output.encoding,
        plan_id=receipt.plan_id,realization_id=receipt.realization_id,construction_id=receipt.construction_id,
        producer_id=receipt.observation.amplitudes.content_id,acquisition=("r","a","job","0"),source=SOURCE)
    return ObservationChunk(run_id="r",plan_id=receipt.plan_id,realization_id=receipt.realization_id,
        prepared_id=receipt.content_id,experiment="mass",setting="mass",bindings=(),
        quantum_layout=receipt.quantum_layout,classical_layout=(),attempt="a",job="job",chunk="0",
        observation=receipt.observation,population="unconditional",returned_shots=None,trajectories=1,
        values=tuple(ScalarValue(label=label,value=mass,frame="encoded_branch") for label in AMPLITUDE_MASS_LABELS),
        source=SOURCE,artifacts=(manifest,),physical_scale_unavailable="physical recovery is unavailable")


def _check_chunk(receipt,chunk,mass):
    chunk.validate_unit_bound(receipt)


def _check_analysis(receipt,chunk,mass):
    from types import SimpleNamespace
    from nwqlib.algorithms.qhd.records import QHDAnalysis

    masses=SimpleNamespace(valid_mass=mass,invalid_mass=0.,observed_mass=mass,
                           pooled_summation_roundoff=QHDAnalysis.pooled_summation_roundoff)
    QHDAnalysis.validate_analysis_masses(masses,SimpleNamespace(receipts=(receipt,),
                                                                observations=SimpleNamespace(chunks=(chunk,))))


@pytest.mark.parametrize("consumer",[_check_chunk,_check_analysis],ids=["chunk","qhd_analysis"])
def test_amplitude_masses_use_the_saved_state_window(consumer):
    """Masses formed from a host-corrected saved state are bounded by the receipt's saved-state window.

    With w the receipt's ``probability_window``, ws its
    ``saved_state_probability_window`` and p zero for the chunk check or QHD's
    pooling allowance for the analysis check, a mass just above 1 + (w + p)
    lies within 1 + (ws + p) and is admitted, and a mass just above
    1 + (ws + p) is refused. An unassessed correction gives no finite upper
    bound, so only nonnegativity is checked. A direct-probability chunk with
    the same envelope keeps ``probability_window``.
    """
    from math import inf,nextafter
    from types import SimpleNamespace
    from nwqlib import _phase_product
    from nwqlib.algorithms.qhd.records import QHDAnalysis
    from nwqlib.execution import ObservationChunk

    receipt=_saved_state_receipt(_phase_product.phase_product_envelope(complex(np.exp(.1j)),2))
    w,ws=receipt.probability_window,receipt.saved_state_probability_window
    p=0. if consumer is _check_chunk else QHDAnalysis.pooled_summation_roundoff((_mass_chunk(receipt,1.),))
    inside,outside=nextafter(1+(w+p),inf),nextafter(1+(ws+p),inf)
    assert inside<=1+(ws+p)
    consumer(receipt,_mass_chunk(receipt,inside),inside)
    with pytest.raises(ValueError,match="roundoff window of its execution"):
        consumer(receipt,_mass_chunk(receipt,outside),outside)
    unassessed=receipt.model_copy(update=dict(statevector_roundoff=None))
    assert unassessed.saved_state_probability_window is None
    for mass in (inside,outside):
        consumer(unassessed,_mass_chunk(unassessed,mass),mass)
    # A stored chunk refuses a negative mass itself, so the chunk check gets
    # the value through a stand-in with the chunk check's fields.
    negative=(SimpleNamespace(prepared_id=unassessed.content_id,point=None,values=(SimpleNamespace(value=-.25),),
                              readout=lambda:SimpleNamespace(kind="amplitudes"))
              if consumer is _check_chunk else _mass_chunk(unassessed,1.))
    with pytest.raises(ValueError,match="nonnegative"):
        (ObservationChunk.validate_unit_bound(negative,unassessed) if consumer is _check_chunk
         else consumer(unassessed,negative,-.25))
    # A direct probability readout of the same receipt keeps the native window.
    direct=SimpleNamespace(prepared_id=receipt.content_id,point=None,observation=SimpleNamespace(kind="probabilities"),
                           applications=(),values=(SimpleNamespace(mass=inside),),
                           readout=lambda:SimpleNamespace(kind="probabilities"),
                           histogram=lambda:SimpleNamespace(entries=len(_mass_chunk(receipt,1.).values)))
    check=(lambda chunk:ObservationChunk.validate_unit_bound(chunk,receipt)) if consumer is _check_chunk else (
        lambda chunk:_check_analysis(receipt,chunk,inside))
    with pytest.raises(ValueError,match="roundoff window"):
        check(direct)
