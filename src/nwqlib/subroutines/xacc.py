"""Bounded XACC fermionic input, mapped literally to Jordan–Wigner Paulis."""

from __future__ import annotations

from io import StringIO
from math import isfinite
from os import PathLike
import re
from typing import Iterable

from qiskit.quantum_info import SparsePauliOp

from nwqlib._validation import finite_real, integer
from nwqlib.subroutines.fermionic_pool import (
    _fermion_terms_to_pauli,
    _normal_ordered_operator,
)

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LINE = re.compile(rf"\s*\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)(.*?)\s*\+?\s*$")
_MODE = re.compile(r"([0-9]+)(\^?)")


def parse_xacc(text: str, *, num_modes: int, coefficient_cutoff: float = 0.0) -> SparsePauliOp:
    """Map XACC text to a qubit Hamiltonian, keeping every nonzero term by default.

    Each nonblank line is ``(real,imag) p^ q ... +``, and the trailing ``+``
    is optional. Operators multiply in their literal written order, ``^``
    denotes creation, and a line without operators is a scalar. Decimal and
    scientific notation are supported. At most four ladder operators per term
    are supported, keeping normal ordering and Jordan-Wigner (JW) expansion
    bounded per term.

    Coefficients already include all prefactors. No symmetry completion,
    Hermitian projection, spin reordering, or extra factors are applied.
    Cross-input sums use real/imaginary ``math.fsum`` at both normal-order and
    Pauli merging stages. Only exactly zero accumulated coefficients are
    removed by default. An explicit nonnegative ``coefficient_cutoff`` removes
    final Pauli coefficients of magnitude at most that absolute cutoff, after
    stable merging, and it never prunes individual input or normal-order terms.
    Individual coefficients and bounded products use float64 arithmetic.
    The identity includes all mapped contributions, not only the file scalar.

    Mode ``j`` maps to qubit ``j`` (rightmost Pauli character for mode zero).
    The required positive ``num_modes`` bounds every written index before JW
    expansion and supplies padding. The caller supplies it as a bound on the
    input, and it is not a local-simulation limit. UTF-8 text may have an
    initial byte-order mark.
    Electron count, spin ordering, and reference occupations are separate
    caller metadata: none can be inferred from this syntax.

    Args:
        text (str): XACC text.
        num_modes (int): Positive number of modes, which is the output
            width. Every written mode index must be smaller.
        coefficient_cutoff (float): Default `0.0`. Nonnegative absolute
            cutoff on the final Pauli coefficients.

    Returns:
        hamiltonian (SparsePauliOp): The Jordan-Wigner image on `num_modes`
            qubits, including its complete identity term.

    Raises:
        TypeError: If `num_modes` is omitted or `text` is not a string.
        ValueError: With a line number, for malformed or nonfinite input,
            an invalid `num_modes` or a mode index outside it.

    Examples:
        `2 a_0^dagger a_0 - 3 I` maps to `-2 I - Z`, because
        `a_0^dagger a_0 = (I - Z)/2`:

        >>> from nwqlib.subroutines.xacc import parse_xacc
        >>> parse_xacc("(2,0)0^ 0 +\\n(-3,0)", num_modes=1).to_list()
        [('I', (-2+0j)), ('Z', (-1+0j))]
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _parse_lines(StringIO(text), num_modes=num_modes, coefficient_cutoff=coefficient_cutoff)


def read_xacc(path: str | PathLike[str], *, num_modes: int, coefficient_cutoff: float = 0.0) -> SparsePauliOp:
    """Read a UTF-8 XACC file and map it to a qubit Hamiltonian, as [`parse_xacc`][nwqlib.subroutines.xacc.parse_xacc] does for text.

    Lines stream into coefficient buckets for stable normal-order summation,
    then the merged polynomial is mapped with Jordan-Wigner and stable Pauli
    summation. Buckets keep O(R) coefficients for R input terms and O(T)
    coefficients for T merged terms (bounded degree four), and no
    statevector or dense operator is constructed.

    Args:
        path (str | os.PathLike[str]): Path of a local UTF-8 XACC file.
        num_modes (int): Positive number of modes, as for `parse_xacc`.
        coefficient_cutoff (float): Default `0.0`. Nonnegative absolute
            cutoff on the final Pauli coefficients, as for `parse_xacc`.

    Returns:
        hamiltonian (SparsePauliOp): The Jordan-Wigner image on `num_modes`
            qubits, including its complete identity term.

    Raises:
        OSError: If the file cannot be opened.
        TypeError: If `num_modes` is omitted.
        ValueError: As for `parse_xacc`, and for a file that is not valid
            UTF-8.
    """
    with open(path, encoding="utf-8") as stream:
        return _parse_lines(stream, num_modes=num_modes, coefficient_cutoff=coefficient_cutoff)


def _parse_lines(lines: Iterable[str], *, num_modes: int, coefficient_cutoff: float) -> SparsePauliOp:
    """Parse XACC lines lazily, normal-order the sum and map it with Jordan-Wigner.

    ``terms`` yields one ``(coefficient, ((mode, action), ...))`` row per
    nonblank line, in written operator order, with action 1 for ``^``.
    ``_normal_ordered_operator`` sums all rows with cutoff zero, so only
    exactly cancelling terms vanish. ``coefficient_cutoff`` applies once, to
    the final Pauli coefficients.
    """
    num_modes = integer(num_modes, "num_modes", 1)
    coefficient_cutoff = finite_real(coefficient_cutoff, "coefficient_cutoff")
    if coefficient_cutoff < 0:
        raise ValueError("coefficient_cutoff must be non-negative")

    def terms():
        """Yield validated ``(coefficient, operators)`` rows, raising with the line number."""
        for line_number, line in enumerate(lines, 1):
            if line_number == 1:
                line = line.removeprefix("\ufeff")
            if not line.strip():
                continue
            prefix = f"XACC line {line_number}: "
            match = _LINE.fullmatch(line)
            if match is None:
                raise ValueError(prefix + "expected finite (real,imag) followed by mode operators")
            real, imag = float(match[1]), float(match[2])
            if not (isfinite(real) and isfinite(imag)):
                raise ValueError(prefix + "coefficient must be finite")
            tokens = match[3].split()
            # Two-body input scope: at most 2**4 JW branches per coalesced term.
            # Revisit only with an explicit higher-body workload and cost contract.
            if len(tokens) > 4:
                raise ValueError(prefix + "at most four ladder operators per term are supported")
            ops = []
            for token in tokens:
                mode_match = _MODE.fullmatch(token)
                if mode_match is None:
                    raise ValueError(prefix + f"invalid mode operator {token!r}; expected 0 or 0^")
                try:
                    mode = int(mode_match[1])
                except ValueError:
                    raise ValueError(prefix + "mode index is too large") from None
                if mode >= num_modes:
                    raise ValueError(prefix + f"mode {mode} is outside num_modes={num_modes}")
                ops.append((mode, int(bool(mode_match[2]))))
            yield complex(real, imag), tuple(ops)

    normal_terms = _normal_ordered_operator(terms(), coefficient_cutoff=0.0)
    return _fermion_terms_to_pauli(normal_terms, num_modes, coefficient_cutoff=coefficient_cutoff)
