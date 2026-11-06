"""Symbolic objective preprocessing for QHD with one-hot or binary encoding."""

from __future__ import annotations

from collections import defaultdict
from functools import cached_property

import sympy as sp

from nwqlib._limits import exceeds


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
    """Return the nominal expansion-term count, checking every partial count with ``admit``.

    For the grammar of numeric constants, symbols, Add, Mul and nonnegative
    integer Pow, this bounds the formal expanded term count by induction.
    An atom contributes one, Add sums child counts and Mul multiplies them.
    An m-th power of an n-term base has at most C(n+m-1, m) distinct
    products, the number of size-m multisets. Coinciding monomials and
    cancellations can only reduce that count. The loop uses the symmetric
    binomial with j = min(n-1, m) and an exact integer running product.

    Outside this grammar, the current rational/exponent-sum/negative-power
    rules are early-rejection heuristics. In b**e, r is the sum of rational
    exponent terms and the count uses m = ceil(abs(r)). A negative r
    returns one after checking its denominator's partial counts. Other
    nodes return one after visiting their arguments. These rules do not
    account for new additions or powers from function rewriting, the terms
    of combined denominators, or all constructor evaluation during
    sp.expand. Even the polynomial term-count bound does not bound
    coefficient digits, temporary copies, all symbolic work or process
    memory.

    ``admit(count)`` can reject each partial count reached before traversal
    continues. QHD counts an untuned nominal 16 bytes per term
    (``method.QHD._admit_symbolic_work``), and the augmented-Lagrangian layer
    adds the term count to its counted preprocessing work
    (``constrained._setup``). Neither count bounds arbitrary SymPy expansion.
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


def node_count(expression: sp.Expr, limit: int | None) -> int:
    """Return the node count of the tree of ``expression``, or ``limit + 1`` once it exceeds ``limit``.

    ``limit=None`` counts the whole tree.

    Operators and leaves count one each. A subexpression that occurs several
    times counts once per occurrence, because the code that ``sp.lambdify``
    prints evaluates it at every occurrence, and the expansion can repeat a
    denominator in every term. Stopping after ``limit + 1`` nodes keeps the
    count itself within the work limit that the caller checks.
    """
    count = 0
    for count, _node in enumerate(sp.preorder_traversal(expression), 1):
        if exceeds(count, limit):
            break
    return count


def expand_holding_exponentials(expression: sp.Expr) -> sp.Expr:
    """Return ``sp.expand(expression, power_exp=False)`` with each maximal ``exp(...)`` held whole.

    Each maximal ``exp(g(x))`` becomes a fresh symbol, which forms the outer
    expression ``F(x, d_1, ..., d_k)``. Only that outer expression is expanded,
    and the same subtrees are then substituted back simultaneously as
    ``d_j = exp(g_j(x))``. Maximal means that traversal stops at the outer
    ``exp``, which ``xreplace`` does by matching from the root, so a nested
    exponential stays inside it. Repeated identical subtrees share one symbol.
    Support is determined after the restoration (``ObjectiveDecomposer._groups``).

    A symbol carries ``positive=True`` only when ``function.is_positive is True``,
    and is plain otherwise. That truthful answer under the supplied symbol
    assumptions is sufficient for ``exp(g)`` to be a finite strictly positive real
    number at every point of the original domain. ``None`` is not a positivity
    proof, and an overall real objective does not imply it, since ``exp(i x)`` is
    generally not positive. Exact ``exp(g) > 0`` stays true where binary64
    ``exp(g)`` underflows to zero. The permitted transformations are then
    distributivity and the branch-safe identities justified by those true
    assumptions, such as ``Log(d z) = log(d) + Log(z)`` and
    ``(d z)**w = d**w z**w`` for d > 0 and nonzero complex z. They give
    ``F(x, exp(g_1), ...) = expand(F)(x, exp(g_1), ...)`` on the original domain.
    This does not authorize ``force=True`` or arbitrary branch changes. Positivity
    alone does not give ``Log(exp(g)) = g``, which also needs a restriction on the
    imaginary part of g, as ``Log(exp(4i)) = i (4 - 2 pi)`` shows. So
    ``log(y exp(-50 (x - 3.0)**2))`` in real variables becomes
    ``-50 (x - 3.0)**2 + log(y)``, with separate supports and no exponential to
    underflow before its logarithm.

    Those identities qualify the symbolic transformation only, and positivity is a
    semantic premise, not a numerical-stability guarantee. They say nothing about
    binary64 evaluation of the outer operations or about automatic constructor
    rewrites. A log or power reduction can remove a bad intermediate or expose
    different cancellation. Products and powers of exponentials can be combined
    or evaluated again during the restoration, as ``exp(a)**2`` becomes
    ``exp(2*a)``, so the restored expression and the emitted evaluator are what
    a table's values follow.

    The exponent is held because its expansion changes the binary64 evaluation.
    For a > 0 and real x, b, ``-a (x - b)**2 <= 0``, whereas expanding it
    separates ``exp(-a b**2)`` from a potentially enormous second factor. ``exp``
    pulls a ``Float`` constant out of a sum in its argument, and a product of
    such constants can lie outside binary64. The smallest normal number is
    ``2**-1022``, the smallest subnormal ``2**-1074``, and the boundary of
    rounding to zero under ties-to-even is ``exp(-1075 log 2)``. A coefficient
    below it prints as a literal that Python reads as zero, and an evaluation
    through that literal is zero or wrong without any nonfinite value. This
    explains the representation failure and gives no reason for a coefficient
    cutoff.

    Holding the exponentials is no stability theorem for an arbitrary expression.
    An exponential whose exponent is written out with a ``Float`` constant is
    already split: ``exp(-50*x**2 + 300.0*x - 450.0)`` has pulled out
    ``exp(-450.0)`` as a number when SymPy builds it, so the symbol holds only the
    remaining factor. SymPy still holds that number, or a product of such numbers,
    at its own exponent range. For ``T(x) = c v(x) prod_j exp(g_j(x))``, the evaluator
    uses ``T(x) = v(x) sign(c) exp(log|c| + sum_j g_j(x))`` when the product has two or
    more exponential factors or a constant of it leaves the normal binary64 range, and
    it treats a positive constant power as an exponential
    (``validation._combined_product``, with the numerical qualification: relative
    error at most ``expm1(eps_q) + eps_exp exp(eps_q)`` in the
    exponential for a normal finite result, no guarantee of a finite result from a
    representable exponent, no remedy for severe cancellation in the exponent, and
    absolute accuracy for a subnormal result).

    Any other power keeps one power through ``power_exp=False`` and has its exponent
    expanded. A positive non-integer Rational base also stays whole, because expansion
    writes ``(p/q)**g`` as ``p**g*q**(-g)``. Near ``p/q = 1`` the two parts are enormous,
    and their exponents cancel only in binary64. Such a power is held through the
    expansion and its exponent is expanded on its own, so an exponent in monomials on a
    far box still cancels exactly when it is centered.

    The symbols are numbered in sorted order under one name, because ``atoms``
    returns a set in an order that follows the hash seed of the process, and
    SymPy orders Dummies by name and number. The term order of the expansion,
    and so the content identity of a Plan, then cannot follow that seed.
    """
    rational_powers = {power for power in expression.atoms(sp.Pow) if not power.exp.is_number and power.base.is_Rational
                       and not power.base.is_Integer and power.base.is_positive}
    held = {function: sp.Dummy("held", positive=True) if function.is_positive is True else sp.Dummy("held")
            for function in sorted(expression.atoms(sp.exp) | rational_powers, key=sp.default_sort_key)}
    expanded = sp.expand(expression.xreplace(held), power_exp=False)
    return expanded.xreplace({dummy: sp.Pow(function.base, expand_holding_exponentials(function.exp))
                              if function in rational_powers else function for function, dummy in held.items()})


class ObjectiveDecomposer:
    """Group a SymPy objective by variable support.

    After expansion, each additive term is keyed by the sorted indices of the
    variables it contains. A support S is later evaluated on only ``K**|S|``
    grid tuples, instead of a table over all ``K**d`` grid points. On the
    one-hot encoding it becomes a sum of products of ``|S|`` number
    operators, and on the binary encoding a real diagonal operator on the
    ``|S| b`` qubits of its variables (``binary`` module). Terms with
    empty support are a constant objective, which contributes only a global
    phase to the evolution.

    With ``centers`` the expansion is in the centered coordinates ``x - m``
    of ``centered_objective``, so a support table is evaluated at ``x - m``
    (see ``expansion_centers``). Without them, or with zero centers, it is
    the plain expansion in x.

    The expansion multiplies out the outer algebra and holds each exponential
    whole (``expand_holding_exponentials``). With ``f_center(z) = f(z + m)``,
    an exact regrouping ``f_center(z) = c0 + sum_S f_S(z_S)`` over nonempty
    supports gives, after substituting ``z = x - m``, the same-grid-point
    identity ``V(i) = c0 + sum_S T_S(i_S) = f(x_i)`` up to the arithmetic of this
    evaluation path. The original domain remains a premise. A regrouping does
    not justify evaluating at a point where the original expression was
    undefined, and cancellation can conceal that domain, so the checks that
    evaluate the original expression stay (``constrained._domain_check``,
    ``verification``). Comparing minima alone does not check this identity.
    """

    def __init__(self, objective: sp.Expr, variables: tuple[sp.Symbol, ...], centers=None) -> None:
        self.variables = tuple(variables)
        self.centers = (0.0,) * len(self.variables) if centers is None else tuple(centers)
        self.objective = expand_holding_exponentials(centered_objective(objective, self.variables, self.centers))
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
        by the QHD limit check, which counts its nodes, and the compiler, which
        evaluates it. ``sp.Add`` flattens the terms in one pass. Python's
        ``sum`` would rebuild and sort the partial sum after every term, work
        quadratic in the group size.
        """
        return {support: expand_holding_exponentials(sp.Add(*terms)) for support, terms in self._groups[0].items()}

    @cached_property
    def constant(self) -> sp.Expr:
        """Return the sum of the terms with empty support, the constant objective, formed with ``sp.Add``."""
        return sp.Add(*self._groups[1])
