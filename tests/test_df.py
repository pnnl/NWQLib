"""A one-orbital exact coefficient witness for explicit factor conversion."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from nwqlib.core import Basis, InputRef, Source, Unit
from nwqlib.operators import DFConversionReceipt, ingest_df


def supplied(*, weight=2, **kwargs):
    return ingest_df(np.array([[-.5]]), ((np.array([[1]]), np.array([weight])),),
        constant_energy=.25, orbital_basis=Basis(identity="one orbital", dimension=1, ordering="p=0"),
        energy_unit=Unit(symbol="Ha", dimension="energy"),
        source=InputRef(identity="one-orbital-data", representation="factor data", source=Source(
            name="scalar factor polynomial", version="1", domain="one real orbital", reference="supplied T,V,w")),
        **kwargs)


def test_factor_conversion_has_independent_coefficients_and_no_error_claim():
    data = supplied()
    converted = data.to_pauli(max_products=3)
    # N=n_alpha+n_beta; E0+T*N+B²*N²/2 with T=-1/2, B=2.
    # n=(I-Z)/2 gives 11I/4 - 7(Z0+Z1)/4 + Z0Z1 exactly.
    assert dict(converted.operator.pauli_terms().labels()) == {"II": 2.75, "IZ": -1.75, "ZI": -1.75, "ZZ": 1.}
    assert converted.receipt.raw_fermion_rows == 7 and converted.receipt.raw_jw_candidates == 73
    assert converted.receipt.conversion_error.fact.availability == "unknown"
    assert converted.work_fact.value.numerator == converted.receipt.work
    assert DFConversionReceipt.model_validate_json(converted.receipt.model_dump_json()) == converted.receipt
    assert dict(supplied(weight=-2).to_pauli().operator.pauli_terms().labels()) == dict(converted.operator.pauli_terms().labels())


def test_factor_snapshot_and_before_work_limits(monkeypatch):
    from nwqlib.operators import df
    data = supplied()
    for array in (data._one_body, *data._factors[0]):
        assert array.dtype == np.dtype("float64")
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        data._factors = ()
    monkeypatch.setattr(df, "_contract", lambda *a: pytest.fail("contraction before action admission"))
    with pytest.raises(ValueError, match="max_products"):
        data.to_pauli(max_products=2)
    with pytest.raises(ValueError, match="max_bytes"):
        data.to_pauli(max_bytes=100)
    monkeypatch.setattr(df, "_freeze_array", lambda *a: pytest.fail("snapshot before input admission"))
    with pytest.raises(ValueError, match="max_bytes"):
        supplied(max_bytes=1)
