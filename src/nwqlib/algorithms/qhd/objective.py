"""Objective preprocessing for the QHD one-hot compiler."""

from __future__ import annotations

from collections import defaultdict
from functools import cached_property

import sympy as sp


def expansion_centers(bounds) -> tuple[float, ...]:
    """Return the point of each box ``[lower, upper]`` nearest the origin.

    ``sp.expand`` writes the objective in monomials about the origin. When a
    box contains the origin, every grid coordinate satisfies
    ``|x| <= upper - lower``, so each monomial is evaluated at arguments no
    larger than the box width. Far from the origin the monomials can be much
    larger than the objective and cancel. For
    ``(x - c)**2`` with ``c = 1e7`` the support table ``x**2 - 2 c x`` and
    the constant ``c**2`` are each near ``1e14``, so binary64 keeps their sum,
    the objective, only to about one ulp of ``1e14``, ``2**-6 ~= 0.016``.
    Expanding in
    ``x - m`` with m the box point nearest the origin bounds every centered
    coordinate by the box width again. A box that contains the origin has
    ``m = 0``, so its expansion is the plain one.
    """

    return tuple(min(max(0.0, float(lower)), float(upper)) for lower, upper in bounds)


def centered_objective(objective: sp.Expr, variables, centers) -> sp.Expr:
    """Return ``objective`` with each variable x replaced by ``x + m``, unexpanded.

    The result is the objective as a function of the centered coordinates
    ``x - m``, written with the original symbols. Each center enters as the
    exact rational value of its binary64 number. Zero centers leave the
    expression unchanged.
    """

    shifts = {
        variable: variable + sp.Rational(center)
        for variable, center in zip(variables, centers, strict=True)
        if center != 0.0
    }
    return objective.xreplace(shifts) if shifts else objective


def uncentered_objective(expression: sp.Expr, variables, centers) -> sp.Expr:
    """Return an expression of the centered coordinates written back in the user's coordinates, for messages.

    Each variable x becomes ``x - m`` with the exact rational value of its
    binary64 center m, the inverse of ``centered_objective``.
    """

    shifts = {
        variable: variable - sp.Rational(center)
        for variable, center in zip(variables, centers, strict=True)
        if center != 0.0
    }
    return expression.xreplace(shifts) if shifts else expression


def expansion_term_charge(node: sp.Expr, admit) -> int:
    """Return the nominal expansion-term charge, admitting every partial count.

    For the grammar of numeric constants, symbols, Add, Mul and nonnegative
    integer Pow, this bounds the formal expanded term count by induction.
    An atom contributes one, Add sums child counts and Mul multiplies them.
    An m-th power of an n-term base has at most C(n+m-1, m) distinct
    products, the number of size-m multisets. Coinciding monomials and
    cancellations can only reduce that count. The loop uses the symmetric
    binomial with j = min(n-1, m) and an exact integer running product.

    Outside this grammar, the current rational/exponent-sum/negative-power
    rules are early-refusal heuristics. In b**e, r is the sum of rational
    exponent terms and the charge uses m = ceil(abs(r)). A negative r
    returns one after admitting its denominator's partial counts. Other
    nodes return one after visiting their arguments. These rules do not
    account for new additions or powers from function rewriting, combined
    denominator populations, or all constructor evaluation during sp.expand.
    Even the polynomial term-count bound does not bound coefficient digits,
    temporary copies, all symbolic work or process memory.

    ``admit(count)`` can refuse each reached partial count before traversal
    continues. QHD assigns an untuned nominal 16-byte charge per term
    (``method.QHD._admit_symbolic_work``); the augmented-Lagrangian layer
    adds the term charge to its counted preprocessing work
    (``constrained._setup``). Neither charge bounds arbitrary SymPy expansion.
    """
    values = [expansion_term_charge(child, admit) for child in node.args]
    if isinstance(node, sp.Add):
        count = sum(values)
    elif isinstance(node, sp.Mul):
        count = 1
        for value in values:
            count *= value
            admit(count)
    elif isinstance(node, sp.Pow):
        r = sum((a for a in sp.Add.make_args(node.exp) if a.is_Rational), sp.S.Zero)
        m, count = int(sp.ceiling(abs(r))), 1
        for j in range(1, min(values[0] - 1, m) + 1):
            count = count * (values[0] + m - j) // j
            admit(count)
        if r < 0:
            count = 1
    else:
        count = 1
    admit(count)
    return count


def node_count(expression: sp.Expr, limit: int) -> int:
    """Return the node count of the tree of ``expression``, or ``limit + 1`` once it exceeds ``limit``.

    Operators and leaves count one each. A subexpression that occurs several
    times counts once per occurrence, because the code that ``sp.lambdify``
    prints evaluates it at every occurrence, and the expansion can repeat a
    denominator in every term. Stopping after ``limit + 1`` nodes keeps the
    count itself within the work that the caller admits.
    """
    count = 0
    for count, _node in enumerate(sp.preorder_traversal(expression), 1):
        if count > limit:
            break
    return count


class ObjectiveDecomposer:
    """Group a SymPy objective by variable support.

    After expansion, each additive term is keyed by the sorted indices of the
    variables it contains. A support S is later evaluated on only ``K**|S|``
    grid tuples and becomes a sum of products of ``|S|`` one-hot number
    operators, instead of a table over all ``K**d`` grid points. Terms with
    empty support are a constant objective, which contributes only a global
    phase to the evolution.

    With ``centers`` the expansion is in the centered coordinates ``x - m``
    of ``centered_objective``, so a support table is evaluated at ``x - m``
    (see ``expansion_centers``). Without them, or with zero centers, it is
    the plain expansion in x.
    """

    def __init__(self, objective: sp.Expr, variables: tuple[sp.Symbol, ...], centers=None) -> None:
        self.variables = tuple(variables)
        self.centers = (0.0,) * len(self.variables) if centers is None else tuple(centers)
        self.objective = sp.expand(centered_objective(objective, self.variables, self.centers))
        self.variable_to_index = {variable: index for index, variable in enumerate(self.variables)}

    @cached_property
    def _groups(self) -> tuple[dict[tuple[int, ...], list[sp.Expr]], list[sp.Expr]]:
        """Return the non-constant terms keyed by sorted variable indices, and the constant terms."""
        grouped: dict[tuple[int, ...], list[sp.Expr]] = defaultdict(list)
        constants: list[sp.Expr] = []
        for term in sp.Add.make_args(self.objective):
            support = tuple(sorted(
                self.variable_to_index[symbol]
                for symbol in term.free_symbols
                if symbol in self.variable_to_index
            ))
            (grouped[support] if support else constants).append(term)
        return dict(grouped), constants

    @cached_property
    def support_expressions(self) -> dict[tuple[int, ...], sp.Expr]:
        """Return the expression that each support table evaluates, keyed by sorted variable indices.

        Each is the expanded sum of its group's terms, formed once and shared
        by the QHD admission, which counts its nodes, and the compiler, which
        evaluates it. ``sp.Add`` flattens the terms in one pass. Python's
        ``sum`` would rebuild and sort the partial sum after every term, work
        quadratic in the group size.
        """
        return {support: sp.expand(sp.Add(*terms)) for support, terms in self._groups[0].items()}

    @cached_property
    def constant(self) -> sp.Expr:
        """Return the sum of the terms with empty support, the constant objective, formed with ``sp.Add``."""
        return sp.Add(*self._groups[1])
