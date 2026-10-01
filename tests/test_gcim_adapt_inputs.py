"""Exact target admission and explicit processed-generator correction evidence."""

from decimal import Decimal, localcontext
import numpy as np
import pytest
import scipy.sparse as sps
from test_adapt_primary import plan_for
from nwqlib.algorithms.gcim.adapt_inputs import _symmetry_evidence, _project_terms
from nwqlib.operators import ingest_pauli
from nwqlib.problems.records import Eigenproblem


@pytest.mark.parametrize("sparse", (False, True))
def test_original_hermitian_extreme_components_and_no_silent_correction(sparse):
    tiny = np.nextafter(0.0, 1.0)
    raw = np.array([[8e307, tiny], [tiny, -8e307]], complex)
    problem = Eigenproblem(A=sps.csr_matrix(raw) if sparse else raw)
    assert problem.A.basis.dimension == 2
    defective = raw.copy()
    defective[0, 0] += tiny * 1j
    with pytest.raises(ValueError, match="Hermitian"):
        Eigenproblem(A=sps.csr_matrix(defective) if sparse else defective)


def test_explicit_antihermitian_correction_preserves_underflow_evidence():
    tiny = np.nextafter(0.0, 1.0)
    terms, evidence = _project_terms(
        (("Z", complex(tiny, 8e307)), ("X", tiny * 1j)),
        antihermitian=True,
        tolerance=1e-12,
        qubits=1,
    )
    assert terms == (("Z", 8e307j), ("X", tiny * 1j))
    assert evidence["correction_applied"] and evidence["correction_component_max"] == tiny
    assert evidence["relative_frobenius_correction"] is None
    assert evidence["relative_frobenius_correction_status"] == "underflow"
    with pytest.raises(ValueError, match="anti-Hermitian"):
        _project_terms((("Z", complex(tiny, 8e307)),), antihermitian=True, tolerance=0.0, qubits=1)


def test_subnormal_tolerance_compares_full_frobenius_ratio():
    tiny = np.nextafter(0.0, 1.0)
    original = np.array([8e307, *([1e-16j] * 63)])
    with pytest.raises(ValueError, match="Hermitian"):
        _symmetry_evidence(
            original,
            original.real - original,
            tolerance=tiny,
            target="Hermitian",
            exact_symmetry=False,
            pauli_qubits=6,
        )


def test_absolute_pruning_preserves_combination_identity_shift_and_dropped_bound():
    with localcontext() as context:
        context.prec = 80
        remainder = float(sum(map(Decimal.from_float, (0.1, 0.2, -0.3))))
    cutoff = 1e-14
    rows = (
        ("II", 1e6),
        ("XI", 4e-7),
        ("IZ", 1e16),
        ("IZ", 6e-15),
        ("IZ", 6e-15),
        ("IZ", -1e16),
        ("ZI", cutoff),
        ("ZZ", 0.1),
        ("ZZ", 0.2),
        ("ZZ", -0.3),
    )
    expected = float(Decimal.from_float(cutoff) + Decimal.from_float(remainder))
    for shift in (0.0, -1e6, 1e10):
        plan = plan_for(
            A=ingest_pauli((*rows, ("II", shift)), num_qubits=2),
            initial_state=[1, 0, 0, 0],
            pool_rows=((("IY", 1j),),),
            max_iterations=1,
        )
        rec = plan.reconstruction
        assert {t.label: t.coefficient for t in rec.terms if t.label != "II"} == {
            "XI": 4e-7,
            "IZ": 1.2e-14,
        }
        assert rec.hamiltonian_dropped_l1 == pytest.approx(expected, rel=2e-16, abs=0.0)


def _exact_commutator(h_rows, a_rows):
    """[H, A] by exact single-qubit Pauli products with rational coefficients."""
    from fractions import Fraction

    table = {("X", "Y"): (1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "X"): (1j, "Y"),
             ("Y", "X"): (-1j, "Z"), ("Z", "Y"): (-1j, "X"), ("X", "Z"): (-1j, "Y")}

    def product(left, right):
        phase, letters = 1 + 0j, []
        for p, r in zip(left, right):
            if p == "I" or r == "I" or p == r:
                letters.append(r if p == "I" else p if r == "I" else "I")
            else:
                factor, letter = table[p, r]
                phase *= factor
                letters.append(letter)
        return phase, "".join(letters)

    total = {}
    for hp, hc in h_rows:
        for ap, ac in a_rows:
            for sign, (left, right) in ((1, (hp, ap)), (-1, (ap, hp))):
                phase, label = product(left, right)
                c = complex(hc) * complex(ac) * phase * sign
                re, im = total.get(label, (Fraction(0), Fraction(0)))
                total[label] = (re + Fraction(c.real), im + Fraction(c.imag))
    return {label: value for label, value in total.items() if value != (0, 0)}


@pytest.mark.parametrize("case", ("x_iz", "dyadic"))
def test_commutator_from_anticommuting_pairs_equals_product_and_cancel_table(case):
    """The surviving-pair commutator equals the ordered-product table and exact rational [H, A]."""
    from nwqlib.algorithms.gcim.adapt_inputs import _commutator_arrays
    from nwqlib.operators._pauli import combine_terms
    from nwqlib.operators.inputs import pauli_table

    if case == "x_iz":
        h_rows, a_rows = (("X", 1.0),), (("Z", 1j),)
    else:
        h_rows = (("III", 0.75), ("IXZ", 0.5), ("YZI", -0.25), ("ZZI", 0.125), ("XIY", -1.5),
                  ("IIZ", 0.375))
        a_rows = (("IXY", 0.5j), ("YII", -0.25j), ("ZXX", 0.125j), ("IYZ", 1.0j))
    q = len(h_rows[0][0])
    h = pauli_table(h_rows, num_qubits=q)
    a = pauli_table(a_rows, num_qubits=q)
    ledger = [0]
    arrays, removed = _commutator_arrays(h, a, cutoff=0.0, ledger=ledger, held=0,
                                         max_bytes=1 << 30, max_products=1 << 40)
    assert removed == 0.0 and ledger[0] > 0
    ha, ah = h.product(a), a.product(h)
    rows = combine_terms((*ha.labels(), *((p, -c) for p, c in ah.labels())))
    expected = tuple((p, c.real) for p, c in sorted(rows) if c.real != 0)
    assert all(c.imag == 0 for _, c in rows)
    assert arrays.rows() == expected
    exact = _exact_commutator(h_rows, a_rows)
    assert {p: (c, 0) for p, c in arrays.rows()} == exact
    if case == "x_iz":
        assert arrays.rows() == (("Y", 2.0),)
