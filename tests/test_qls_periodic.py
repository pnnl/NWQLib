"""Original periodic spectral facts, actual rounded gap, and compact Plan."""

from fractions import Fraction
from types import SimpleNamespace
import pytest
import nwqlib
from nwqlib.algorithms.qls import QLS, method as owner
from nwqlib.algorithms.qls.primary_records import InversePolynomial
from nwqlib.operators import ingest_pauli, ingest_periodic_stencil
from nwqlib.operators.inputs import OperatorInput
from nwqlib.problems import LinearSystem, NormSquared, QuadraticForm
from nwqlib.problems.inputs import ingest_occupation


def periodic_plan(
    *,
    num_qubits=2,
    mass=1.0,
    diffusion=0.25,
    potential=0.0,
    observable=True,
    method=None,
    execution="quantum",
):
    """Plan a compact periodic A x = |0...0> with a physical scalar output of x = A^-1 b.

    The operator and right-hand side stay compact. The output is the quadratic form of
    I...IZ, or ||x||^2 when observable is false.
    """
    operator = ingest_periodic_stencil(num_qubits, mass, diffusion, potential)
    rhs = ingest_occupation("0" * num_qubits, num_qubits=num_qubits)
    output = (
        QuadraticForm(
            observable=ingest_pauli((("I" * (num_qubits - 1) + "Z", 1.0),), num_qubits=num_qubits)
        )
        if observable
        else NormSquared()
    )
    return nwqlib.plan(
        LinearSystem(A=operator, b=rhs), method=method or QLS(), output=output, execution=execution, seed=7
    )


def forbidden(*args, **kwargs):
    raise AssertionError("compact planning crossed dense, spectral, native or backend boundary")


@pytest.fixture(autouse=True)
def injected_selection(monkeypatch):
    from nwqlib.algorithms.qls import numerical, quantum
    from nwqlib.subroutines.block_encoding import banded
    from nwqlib.subroutines.qsp import phases

    monkeypatch.setattr(OperatorInput, "dense_array", forbidden)
    monkeypatch.setattr(banded, "_detect_banded_structure_with_error", forbidden)
    monkeypatch.setattr(numerical, "_extreme_singular_values", forbidden)
    monkeypatch.setattr(quantum, "_native_encoding", forbidden)
    calls = []

    def polynomial(method, kappa):
        calls.append(kappa)
        return InversePolynomial(
            coefficients=(0.0, 1.0, 0.0, -0.5),
            rescale=2.0,
            candidate_degrees=(1, 3),
            certificate=0.1,
            certificate_grid_points=75,
            certificate_basis="affine_chebyshev_residual_norming",
            lsq_node_count=8,
        )

    monkeypatch.setattr(owner, "select_polynomial", polynomial)
    monkeypatch.setattr(
        phases,
        "solve_symmetric_qsp_phases",
        lambda coefficients, **kw: SimpleNamespace(
            phases=(0.0,) * len(coefficients),
            residual_sup_bound=0.0,
            evaluations=0,
        ),
    )
    return calls


def family_law(mass, diffusion, alpha):
    m, d, c, a = map(Fraction, (mass, diffusion, mass + 2 * diffusion, alpha))
    delta = c - m - 2 * d
    normalization_error = abs(a - (c + 2 * d))
    return abs(delta) + normalization_error, min(m, c - 2 * d - normalization_error)


def encoding(plan):
    return plan._native["encoding"]


def test_exact_periodic_plan_keeps_original_endpoints_and_existing_band_law():
    selected = periodic_plan()
    rec = selected.reconstruction
    child = encoding(selected)
    bands = child._payload.source
    assert bands.offsets == (-1, 0, 1) and bands.coefficients == (-0.25, 1.5, -0.25)
    assert (rec.alpha, rec.kappa_be, rec.encoding_ancillas, rec.width) == (2.0, 2.0, 2, 6)
    assert (rec.sigma_min, rec.sigma_max, rec.spectral_method) == (1.0, 2.0, "analytic_periodic")
    error, lower = family_law(1.0, 0.25, rec.alpha)
    assert error == 0 and lower == 1 and rec.encoding_error == 0.0
    assert child.record.semantics.epsilon == child._payload.error_bound == rec.encoding_error
    report = nwqlib.estimate(selected)
    assert report.construction_id == selected.construction.content_id

def test_periodic_downward_coefficient_rounding_covers_actual_gap():
    mass = 9 * 2.0**-53
    rec = periodic_plan(mass=mass, diffusion=1.0).reconstruction
    assert Fraction(mass + 2.0) - Fraction(mass) - 2 == -Fraction(2.0**-53)
    assert rec.sigma_min == mass and rec.encoding_error == 2.0**-53
    assert rec.alpha == 4 + 2.0**-50
    assert rec.kappa_be == rec.alpha / mass
    assert rec.polynomial_kappa == 2.0**52 + 1
    with pytest.raises(ValueError, match="original spectrum"):
        periodic_plan(mass=mass, diffusion=1.0, method=QLS(kappa=2.0**52))

def test_wide_planning_stays_compact_and_roundtrips_without_native(tmp_path, monkeypatch):
    # This is metadata only; no q65 circuit is built or simulated.
    selected = periodic_plan(num_qubits=65, observable=False)
    assert selected.reconstruction.width == 69 and selected.problem.dimension == 1 << 65
    from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan

    saved = save_plan(selected, ArchiveFiles(tmp_path, 4_000_000))
    monkeypatch.setattr(owner, "select_polynomial", forbidden)
    loaded = load_plan(saved, ArchiveFiles(tmp_path, 4_000_000))
    assert loaded.content_id == selected.content_id
