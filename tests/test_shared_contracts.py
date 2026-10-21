"""Tests for stable shared public contracts."""

import subprocess
import sys

import numpy as np



def test_dense_pauli_default_preserves_scale_and_relative_pruning_is_covariant() -> None:
    from nwqlib.subroutines.pauli_decomposition import decompose_matrix_to_pauli

    matrix = np.array([[1.0, 1.0e-6], [1.0e-6, -1.0]])
    absolute = decompose_matrix_to_pauli(matrix, atol=1.0e-12)
    np.testing.assert_array_equal(absolute.to_sparse_pauli_op().to_matrix(), matrix)
    relative = decompose_matrix_to_pauli(matrix, atol=1.0e-12, rtol=1.0e-5)
    assert [term.label for term in relative.terms] == ["Z"]
    assert np.linalg.norm(relative.to_sparse_pauli_op().to_matrix() - matrix, 2) == 1.0e-6
    tiny_matrix = np.array([[1.0, 1.0e-100], [1.0e-100, -1.0]])
    exact = decompose_matrix_to_pauli(tiny_matrix)
    np.testing.assert_array_equal(exact.to_sparse_pauli_op().to_matrix(), tiny_matrix)
    for scale in (1e-300, 1e-100, 1e100, 1e300):
        full = decompose_matrix_to_pauli(matrix*scale)
        assert {term.label for term in full.terms} == {"X", "Z"}
        pruned = decompose_matrix_to_pauli(matrix*scale, rtol=1e-5)
        assert [term.label for term in pruned.terms] == ["Z"]
        assert pruned.terms[0].coefficient == scale


def test_root_import_stays_lazy() -> None:
    script = (
        "import sys\n"
        "import warnings\n"
        "import nwqlib\n"
        "qiskit_modules = [name for name in sys.modules if name.startswith('qiskit')]\n"
        "assert not qiskit_modules, qiskit_modules\n"
        "public = [name for name in nwqlib.__all__ if name != '__version__']\n"
        "loaded = [f'nwqlib.{name}' for name in public if f'nwqlib.{name}' in sys.modules]\n"
        "assert not loaded, loaded\n"
        "warnings.simplefilter('error')\n"
        "import nwqlib.backends as backends\n"
        "for name in ('AerBackend', 'NWQSimBackend', 'NWQSimSlurmBackend', 'IBMRuntimeBackend', 'IonQBackend', 'NexusBackend'):\n"
        "    assert getattr(backends, name).__name__ == name\n"
        "optional = ('qiskit', 'qiskit_aer', 'qiskit_ibm_runtime', 'qiskit_ionq', 'qnexus', 'pytket')\n"
        "assert not any(name == prefix or name.startswith(prefix+'.') for name in sys.modules for prefix in optional)\n"
        "for name in public:\n"
        "    module = getattr(nwqlib, name)\n"
        "    if name in nwqlib._SCIENTIST_OPERATIONS:\n"
        "        assert module.__module__ == 'nwqlib.scientist' and module.__name__ == name\n"
        "    elif name in nwqlib._SEARCH_OPERATIONS:\n"
        "        assert module.__module__ == 'nwqlib.search' and module.__name__ == name\n"
        "    elif name in nwqlib._SCIENTIFIC_INPUTS:\n"
        "        assert isinstance(module, type)\n"
        "        assert module.__module__ == 'nwqlib.problems.records' and module.__name__ == name\n"
        "    else:\n"
        "        assert module.__name__ == f'nwqlib.{name}', module.__name__\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
