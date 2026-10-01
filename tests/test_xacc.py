"""Literal XACC semantics checked against occupation-basis fermion action."""

import numpy as np
import pytest

from nwqlib.subroutines.xacc import parse_xacc, read_xacc
from nwqlib.subroutines.fermionic_pool import _fermion_terms_to_pauli, _normal_ordered_operator


def _fermion_matrix(terms, num_modes):
    """Independent right-to-left action using occupation parity, not Pauli algebra."""
    result = np.zeros((2**num_modes, 2**num_modes), dtype=complex)
    for column in range(2**num_modes):
        for coefficient, ops in terms:
            state, amplitude = column, complex(coefficient)
            for mode, creation in reversed(ops):
                occupied = (state >> mode) & 1
                if occupied == creation:
                    amplitude = 0
                    break
                amplitude *= (-1) ** (state & ((1 << mode) - 1)).bit_count()
                state ^= 1 << mode
            result[state, column] += amplitude
    return result


def test_literal_products_match_independent_occupation_action():
    terms = [
        (-2.5, ()),
        (0.4, ((0, 0), (0, 1))),
        (1.2j, ((2, 1), (0, 0))),
        (-1.2j, ((0, 1), (2, 0))),
        (0.7 - 0.3j, ((0, 0), (1, 1), (0, 1), (2, 0))),
        (0.02, ((1, 1), (1, 1))),
        (-0.9, ((0, 1), (2, 1), (2, 0), (0, 0))),
        (0.13j, ((2, 0),)),
        (0.05, ((2, 1), (1, 0), (0, 1))),
        (0.3 - 0.2j, ((3, 1), (0, 0), (2, 1), (3, 0))),
    ]
    text = "\n".join(
        f"({complex(c).real:+.17e}, {complex(c).imag:+.17e}) "
        + " ".join(f"{mode}{'^' if action else ''}" for mode, action in ops)
        + " +"
        for c, ops in terms
    )
    actual = parse_xacc(text, num_modes=4)
    # Machine-precision identity: ten bounded products on four modes; 1e-14
    # allows rounding in the two independent accumulation orders.
    np.testing.assert_allclose(actual.to_matrix(), _fermion_matrix(terms, 4), atol=1e-14, rtol=0)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_complete_identity_and_blocked_mode_labels(tmp_path, encoding):
    text = "\n ( -2.0 , +0e0 ) +\n(6,0) 23^ 23 +\n"
    path = tmp_path / "hamiltonian.xacc"
    path.write_text(text, encoding=encoding)
    op = read_xacc(path, num_modes=24)
    assert op.num_qubits == 24
    assert dict(op.to_list()) == {"I" * 24: 1, "Z" + "I" * 23: -3}
    padded = parse_xacc("(2,0)0^ 0", num_modes=4)
    assert dict(padded.to_list()) == {"IIII": 1, "IIIZ": -1}


def test_zero_cutoff_preserves_small_terms_and_existing_pool_default(tmp_path):
    terms = [(2e-16, ((0, 1), (0, 0)))]
    assert _normal_ordered_operator(terms) == {}
    assert _fermion_terms_to_pauli({terms[0][1]: terms[0][0]}, 1).coeffs[0] == 0
    op = parse_xacc("(2e-16,0)0^ 0", num_modes=1)
    assert dict(op.to_list()) == {"I": 1e-16, "Z": -1e-16}
    # Exact binary cancellation leaves a nonzero coefficient below the pool cutoff.
    residue = 2.0**-45
    text = f"(1,0)0^ 0 +\n({-1 + residue!r},0)0^ 0"
    assert dict(parse_xacc(text, num_modes=1).to_list()) == {"I": residue / 2, "Z": -residue / 2}
    # Pruning acts on the final Pauli coefficients, never on individual inputs.
    # Four 2**-44 scalars sum exactly to 2**-42, above the requested 2**-43.
    text = "\n".join([f"({2.0**-44},0)"] * 4)
    assert dict(parse_xacc(text, num_modes=1, coefficient_cutoff=2.0**-43).to_list()) == {"I": 2.0**-42}
    assert dict(parse_xacc("(2e-16,0)0^ 0", num_modes=1, coefficient_cutoff=1e-15).to_list()) == {"I": 0}
    path = tmp_path / "small.xacc"
    path.write_text("(2e-16,0)0^ 0", encoding="utf-8")
    assert dict(read_xacc(path, num_modes=1).to_list()) == {"I": 1e-16, "Z": -1e-16}
    assert dict(read_xacc(path, num_modes=1, coefficient_cutoff=1e-15).to_list()) == {"I": 0}


def test_constants_zero_and_exact_anticommutation():
    assert dict(parse_xacc("(3,-2)", num_modes=2).to_list()) == {"II": 3 - 2j}
    assert dict(parse_xacc("", num_modes=2).to_list()) == {"II": 0}
    assert dict(parse_xacc("(1,0)2 2^ +\n(1,0)2^ 2", num_modes=3).to_list()) == {"III": 1}
    assert dict(parse_xacc("(1,0)2^ 2^", num_modes=3).to_list()) == {"III": 0}
    assert dict(parse_xacc("\ufeff(3,-2)", num_modes=1).to_list()) == {"I": 3 - 2j}
    # No arithmetic beyond a sum of a singleton: preserve the least subnormal.
    assert dict(parse_xacc("(5e-324,-5e-324)", num_modes=1).to_list()) == {"I": complex(5e-324, -5e-324)}
    # Finite components remain legal even when their complex magnitude overflows.
    assert dict(parse_xacc("(1.7e308,1.7e308)", num_modes=1).to_list()) == {"I": complex(1.7e308, 1.7e308)}


@pytest.mark.parametrize("operators", ["", "0^ 0"])
@pytest.mark.parametrize("cutoff", [0.0, 1e-20])
def test_cross_input_cancellation_keeps_small_real_and_imaginary_residue(operators, cutoff):
    text = "\n".join(f"{coefficient}{operators}" for coefficient in [
        "(1,1)", "(1e-16,2e-16)", "(-1,-1)",
    ])
    expected = {"I": 1e-16 + 2e-16j} if not operators else {
        "I": 5e-17 + 1e-16j, "Z": -5e-17 - 1e-16j,
    }
    assert dict(parse_xacc(text, num_modes=1, coefficient_cutoff=cutoff).to_list()) == expected


@pytest.mark.parametrize("cutoff", [0.0, 1e-20])
def test_jw_cancellation_across_distinct_monomials_preserves_small_identity(cutoff):
    # Three different number operators share an identity contribution only after JW.
    op = parse_xacc("(4,4)0^ 0\n(4e-16,8e-16)1^ 1\n(-4,-4)2^ 2", num_modes=3, coefficient_cutoff=cutoff)
    assert dict(op.to_list()) == {
        "III": 2e-16 + 4e-16j,
        "IIZ": -2 - 2j,
        "IZI": -2e-16 - 4e-16j,
        "ZII": 2 + 2j,
    }


@pytest.mark.parametrize("line", [
    "(nan,0)0^ 0", "(0,inf)0", "(1e999,0)0", "(1,0)-1^ 0",
    "(1,0)0^^ 0", "(1,0)0.5", "(1,0)0^0", "1.0 0^ 0",
    "(1,0)0 + garbage", "(1,0)0 ++", "(1,0)0 1 2 3 4",
    "(1,0)0; __import__('os')",
])
def test_malformed_input_has_actionable_line_error(line):
    with pytest.raises(ValueError, match="XACC line 2:"):
        parse_xacc("(1,0) +\n" + line, num_modes=3)


@pytest.mark.parametrize("width", [True, 0, -1, 2.5, "2"])
def test_width_validation(width):
    with pytest.raises(ValueError, match="num_modes"):
        parse_xacc("(1,0)0", num_modes=width)


def test_missing_and_insufficient_width(monkeypatch):
    import nwqlib.subroutines.xacc as xacc

    def forbidden_mapping(*args, **kwargs):
        pytest.fail("width admission must precede JW expansion")

    monkeypatch.setattr(xacc, "_fermion_terms_to_pauli", forbidden_mapping)
    with pytest.raises(ValueError, match="line 2: mode 3 is outside"):
        parse_xacc("(1,0)\n(0,0)3", num_modes=3)
    with pytest.raises(ValueError, match="line 1: mode 20000 is outside"):
        parse_xacc("(1,0)20000^ 20000", num_modes=24)


@pytest.mark.parametrize("cutoff", [0.0, 1e-20])
def test_coefficient_accumulation_overflow_is_rejected(cutoff):
    with pytest.raises(ValueError, match="accumulation overflowed"):
        parse_xacc("(1e308,0)\n(1e308,0)", num_modes=1, coefficient_cutoff=cutoff)
    # Three distinct number operators contribute 7e307 I each only after JW.
    with pytest.raises(ValueError, match="accumulation overflowed"):
        parse_xacc("(1.4e308,0)0^ 0\n(1.4e308,0)1^ 1\n(1.4e308,0)2^ 2", num_modes=3, coefficient_cutoff=cutoff)


@pytest.mark.parametrize("cutoff", [-1.0, float("nan"), float("inf"), True])
def test_cutoff_validation(cutoff):
    with pytest.raises(ValueError, match="coefficient_cutoff"):
        parse_xacc("(1,0)", num_modes=1, coefficient_cutoff=cutoff)
