"""Four-qubit H2/STO-3G Jordan-Wigner Hamiltonian at R = 0.7414 Angstrom.

``H2_STO3G_JW_TERMS`` is the electronic Hamiltonian in Hartree, stored at
binary64 precision, with the nuclear repulsion excluded from the identity term.
No generating script was recorded for these digits, so they serve as a
regression anchor. The independent reference is O. Kyriienko,
arXiv:1901.09988, doi:10.1038/s41534-019-0239-7, Eq. (M7) on page 9, which
prints the same OpenFermion
STO-3G Jordan-Wigner Hamiltonian at this bond length rounded to six decimals,
with the nuclear repulsion included in xi_0. ``H2_STO3G_M7_TERMS`` keeps those
printed values in that convention.
"""

# Qiskit Pauli labels are big-endian: the qubit-0 term Z0 is "IIIZ".
H2_STO3G_JW_TERMS: tuple[tuple[str, float], ...] = (
    ("IIII", -0.8126179630230784),
    ("IIIZ", 0.1711977490343301),
    ("IIZI", 0.17119774903433005),
    ("IZII", -0.22278593040418368),
    ("ZIII", -0.22278593040418368),
    ("IIZZ", 0.1686221915892094),
    ("IZIZ", 0.12054482205301789),
    ("ZIIZ", 0.16586702410589185),
    ("IZZI", 0.16586702410589185),
    ("ZIZI", 0.12054482205301789),
    ("ZZII", 0.17434844185575626),
    ("XXYY", -0.04532220205287397),
    ("XYYX", 0.04532220205287397),
    ("YXXY", 0.04532220205287397),
    ("YYXX", -0.04532220205287397),
)

# Eq. (M7): H = xi0 + xi1 (Z0 + Z1) - xi2 (Z2 + Z3) + xi3 Z0 Z1
#   + xi4 (Z0 Z2 + Z1 Z3) + xi5 (Z0 Z3 + Z1 Z2) + xi6 Z2 Z3
#   - xi7 (X0 X1 Y2 Y3 - X0 Y1 Y2 X3 - Y0 X1 X2 Y3 + Y0 Y1 X2 X3),
# where subscript j is qubit j, the rightmost character of a Qiskit label.
_M7_XI = (-0.098864, 0.171198, 0.222786, 0.168622, 0.120545, 0.165867, 0.174348, 0.045322)
H2_STO3G_M7_TERMS: tuple[tuple[str, float], ...] = (
    ("IIII", _M7_XI[0]),
    ("IIIZ", _M7_XI[1]),
    ("IIZI", _M7_XI[1]),
    ("IZII", -_M7_XI[2]),
    ("ZIII", -_M7_XI[2]),
    ("IIZZ", _M7_XI[3]),
    ("IZIZ", _M7_XI[4]),
    ("ZIZI", _M7_XI[4]),
    ("ZIIZ", _M7_XI[5]),
    ("IZZI", _M7_XI[5]),
    ("ZZII", _M7_XI[6]),
    ("YYXX", -_M7_XI[7]),
    ("XYYX", _M7_XI[7]),
    ("YXXY", _M7_XI[7]),
    ("XXYY", -_M7_XI[7]),
)
# Printed to six decimals, so each value is within half a unit of the sixth
# decimal of the Hamiltonian the paper computed.
H2_STO3G_M7_ROUNDING = 5.0e-7
# Nuclear repulsion 1/R in Hartree, with R converted using the Bohr radius
# 0.529177210903 Angstrom (CODATA 2018).
H2_STO3G_NUCLEAR_REPULSION = 0.529177210903 / 0.7414
