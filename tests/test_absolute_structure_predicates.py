"""Regression pins for absolute Hermitian and unitary predicates."""

from __future__ import annotations

import numpy as np
import pytest

from nwqlib import Eigenproblem
from nwqlib.subroutines._matrix_checks import is_unitary
from nwqlib.subroutines.qpe.coherent import build_coherent_qpe_circuit


def _scaled_nonhermitian() -> np.ndarray:
    return np.array(
        [[1.0e6, 1.0e6], [1.0e6 + 1.0, 2.0e6]],
        dtype=complex,
    )


def test_all_absolute_hermitian_predicates_reject_scaled_asymmetry() -> None:
    matrix = _scaled_nonhermitian()
    hermitian = np.diag([1.0e6, 2.0e6]).astype(complex)

    with pytest.raises(ValueError, match="Hermitian"):
        Eigenproblem(A=matrix)
    admitted = Eigenproblem(A=hermitian)
    assert admitted.A.manifest.basis.dimension == 2

    from nwqlib import LinearSystem, plan
    from nwqlib.algorithms.qls import QLS
    selected = plan(LinearSystem(A=matrix, b=np.array([1., 0.])),
        method=QLS(), execution="classical", seed=7)
    assert selected.reconstruction.embedding == "hermitian_dilation"
    assert selected.reconstruction.spectral_method == "original_svd"

    # The accepted explicit embedding/original-coordinate relationship has
    # actual selected-host and quantum witnesses in the QLS primary tests.


def test_unitary_predicate_and_coherent_builder_reject_relative_only_match() -> None:
    defective = np.diag([1.0, 1.0 + 1.0e-7]).astype(complex)
    assert not is_unitary(defective)
    assert is_unitary(np.eye(2, dtype=complex))
    with pytest.raises(ValueError, match="unitary must be unitary"):
        build_coherent_qpe_circuit(defective, num_phase_qubits=2)
    assert (
        build_coherent_qpe_circuit(
            np.eye(2, dtype=complex), num_phase_qubits=2
        ).num_system_qubits
        == 1
    )
