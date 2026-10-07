#!/usr/bin/env python
"""Error-budget figure and table values of the LCHS and QLS guides.

Run from the repository root after ``python -m pip install -e ".[aer,notebook]"``::

    python docs/scripts/error_budget_figures.py

The script writes ``docs/assets/lchs_error_budget.svg`` and prints every
value of the error-budget sections of ``docs/algorithms/lchs.md`` and
``docs/algorithms/qls.md``, followed by the versions that produced them.

LCHS: heat flow ``du/dt = -L u`` with ``L = (2I - S - S†)/4`` on a ring of
``2**q`` sites, a unit point source at site 0 and ``T = 1``, planned with
``approximation_tolerance=0.005`` and ``trotter_synthesis_tolerance=0.005``
at every q of ``HEAT_FLOW_SIZES``. Planning and resource estimation build
no circuit. At ``MEASURED_Q`` the plan is also solved once on the local Aer
statevector and compared with ``scipy.linalg.expm``.

QLS: the first example of the guide, ``A = [[1.1, .1], [.1, .9]]`` and
``b = [1, .25]`` with ``QLS()``, solved once on the local Aer statevector
and checked with ``QLSVerification``. Then the compact ring of the guide's
"Plan and estimate beyond simulation" example is planned and estimated for
every tolerance of ``QLS_TOLERANCES`` at every q of ``QLS_SIZES``.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer
import scipy
from scipy.linalg import expm

from nwqlib import LinearDynamics, LinearSystem, estimate, plan, solve
from nwqlib.algorithms import LCHS, QLS
from nwqlib.algorithms.lchs import resolve_lchs_coefficient_plan
from nwqlib.algorithms.qls import QLSVerification
from nwqlib.operators import PeriodicStencil, operator_input
from nwqlib.problems import ingest_occupation, ingest_product
from nwqlib.resources import ResourceContext

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "docs" / "assets" / "lchs_error_budget.svg"
CX = ResourceContext(basis="cx")
SEED = 7

HEAT_FLOW_SIZES = (3, 10, 20, 40, 100)  # system qubits q of the ring
MEASURED_Q = 3                          # 12 qubits in total, simulated once
TARGET = 0.01                           # sum of the two tolerances of LCHS_METHOD
LCHS_METHOD = LCHS(hamiltonian_evolution_backend="trotter", approximation_tolerance=0.005,
                   trotter_steps=None, trotter_synthesis_tolerance=0.005)

QLS_SIZES = (10, 20, 40)
QLS_TOLERANCES = (0.1, 0.01, 0.001, 0.0001)

# Okabe-Ito colors, distinguishable with the common color-vision deficiencies.
COLORS = {"kernel_approximation": "#56B4E9", "k_quadrature": "#009E73",
          "trotter_synthesis": "#D55E00", "measured": "#000000"}


def heat_flow(q):
    """Return the LCHS guide's heat-flow problem on a ring of 2**q sites."""
    return LinearDynamics(A=PeriodicStencil(q, mass=0.0, diffusion=0.25),
                          initial_state=ingest_occupation("0" * q, num_qubits=q), time=1.0)


def number(value):
    """Return a stored Float64 or Rational value as a float."""
    return float(value.value) if hasattr(value, "value") else value.numerator / value.denominator


def fact_values(item):
    """Map each fact quantity of a Plan or Result to its float value."""
    return {fact.fact.quantity: number(fact.fact.value) for fact in item.facts}


def lchs_values():
    """Plan and estimate the heat flow at every size, and solve it once at MEASURED_Q."""
    rows = []
    for q in HEAT_FLOW_SIZES:
        selected = plan(heat_flow(q), method=LCHS_METHOD, seed=SEED)
        workload = estimate(selected, context=CX)
        bounds = fact_values(selected)
        cx = workload.quantity("cx")
        rows.append(dict(
            q=q, plan=selected, bounds=bounds,
            width=workload.quantity("logical_width", location="logical_device").fact.value.numerator,
            branches=selected.reconstruction.physical_branches,
            slots=selected.reconstruction.padded_branches,
            steps=selected.reconstruction.step_counts[0],
            cx=number(cx.fact.value), cx_kind=cx.interpretation,
        ))
    measured_row = next(row for row in rows if row["q"] == MEASURED_Q)
    result = solve(measured_row["plan"])
    n = 2**MEASURED_Q
    shift = np.roll(np.eye(n), 1, axis=0)  # S e_j = e_(j+1 mod n)
    reference = expm(-(2 * np.eye(n) - shift - shift.T) / 4)[:, 0]  # exp(-L T) e_0 with T = 1
    measured = float(np.linalg.norm(result.solution - reference))
    return rows, measured


def print_lchs(rows, measured):
    """Print the LCHS budget table, the measured error and the scaling table."""
    row = rows[-1]
    coefficients = resolve_lchs_coefficient_plan(row["plan"])
    rule = coefficients.quadrature
    bounds, steps = row["bounds"], row["steps"]
    strang = bounds["trotter_synthesis"]
    total = bounds["kernel_approximation"] + bounds["k_quadrature"] + strang
    print("LCHS heat flow, budget at every q (shown for q = %d)" % row["q"])
    print("  kernel", coefficients.resolved_lchs_kernel.implementation,
          coefficients.resolved_lchs_kernel.parameters)
    print("  cutoff K", f"{rule.range_k:.6g}")
    print("  Gauss rule", 2 * rule.interval_count_each_side, "panels x", rule.node_count, "points")
    print("  branches / address slots", row["branches"], "/", row["slots"])
    print("  coefficient 1-norm", f"{coefficients.coefficient_l1_norm:.6g}")
    for name, value in bounds.items():
        print("  fact", name, repr(value))
    print("  Strang coefficient B = r**2 * bound", f"{steps**2 * strang:.6g}")
    print("  Strang steps r", steps)
    print("  Strang bound at r - 1 and r", f"{strang * steps**2 / (steps - 1)**2:.6g}",
          f"{strang:.6g}")
    print("  sum of the three bounds", f"{total:.6g}", "target", TARGET)
    print(f"  measured L2 error at q = {MEASURED_Q}: {measured!r}")
    print("LCHS heat flow, sizes: q, total qubits, branches (slots), steps, sum of bounds, CX")
    for row in rows:
        b = row["bounds"]
        print("  ", row["q"], row["width"], f"{row['branches']} ({row['slots']})", row["steps"],
              f"{b['kernel_approximation'] + b['k_quadrature'] + b['trotter_synthesis']:.6g}",
              f"{row['cx']:,.0f}", row["cx_kind"])
    return bounds, total, steps


def write_lchs_figure(bounds, total, steps, measured, path):
    """Draw the stacked bounds against the target, with the measured error beside them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "svg.fonttype": "none",                 # keep labels as searchable text
        "svg.hashsalt": "nwqlib-error-budget",  # stable element ids between runs
        "font.family": "sans-serif",
        "font.size": 10,
    })
    fig, ax = plt.subplots(figsize=(8, 2.7), layout="constrained")
    # An opaque white background keeps the figure readable in the site's light and dark schemes.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    handles, left = [], 0.0
    for name, key in (("Cutoff tail", "kernel_approximation"), ("Quadrature", "k_quadrature"),
                      (f"Strang (r = {steps})", "trotter_synthesis")):
        value = bounds[key]
        handles.append(ax.barh(1, value, left=left, height=0.55, color=COLORS[key],
                               label=f"{name} {value:.3g}"))
        left += value
    handles.append(ax.axvline(TARGET, color="black", linestyle="--", linewidth=1.2,
                              label=f"Target {TARGET:g}"))
    handles.extend(ax.plot([measured], [0], linestyle="none", marker="D", markersize=7,
                           color=COLORS["measured"],
                           label=f"Measured at q = {MEASURED_Q}: {measured:.3g}"))
    ax.annotate(f"{measured:.3g}", (measured, 0), textcoords="offset points", xytext=(8, -4))
    ax.set(xlim=(0, 0.0115), ylim=(-0.6, 1.6), yticks=[0, 1],
           yticklabels=[f"Measured, q = {MEASURED_Q}", "Bounds, every q"],
           xlabel="L2 error of the physical vector, unit input",
           title=f"Sum of the three bounds: {total:.4g}")
    # Matplotlib fills legend columns first, so this order puts the three bounds in the first row.
    order = (0, 3, 1, 4, 2)
    fig.legend([handles[i] for i in order], [handles[i].get_label() for i in order],
               ncol=3, fontsize=8.5, loc="outside lower center", frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def qls_problem(q):
    """Return the compact ring of the QLS guide: A = 0.1 I + 0.1 (2I - S - S†), product b."""
    A = operator_input(PeriodicStencil(q, mass=0.1, diffusion=0.1))
    b = ingest_product([[np.cos(0.3 + 0.1 * j), np.sin(0.3 + 0.1 * j)] for j in range(q)])
    return LinearSystem(A=A, b=b)


def print_qls():
    """Print the QLS first-example budget and the tolerance table of the compact ring."""
    result = solve(LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1., .25]), method=QLS(), seed=SEED)
    rec = result.plan.reconstruction
    receipt, _ = result.verify(checks=QLSVerification(comparisons=("inverse_relative_error",)))
    application = receipt.applications[0]
    arguments = {item.parameter: item.value for item in application.arguments}
    norm = float(arguments["encoded_reference_norm"].value)
    print("QLS first example")
    print("  solution", result.x)
    print("  solver", result.plan.method.solver, "degree", rec.degree)
    print("  alpha", rec.alpha, "kappa_be", rec.kappa_be, "polynomial_kappa", rec.polynomial_kappa)
    print("  epsilon_inv", result.plan.method.epsilon_inv, "rescale s", rec.polynomial.rescale)
    print("  stored residual bound (certificate)", repr(rec.polynomial.certificate))
    print("  phase-fit residual bound delta", repr(rec.phase_error))
    print("  phase term kappa_poly s delta / ||y_ref||",
          f"{rec.polynomial_kappa * rec.polynomial.rescale * rec.phase_error / norm:.3g}")
    for item in application.facts:
        print("  fact", item.fact.quantity, repr(item.fact.value.value))
    exact = np.array([25 / 28, 5 / 28])
    print("  relative error against (25/28, 5/28)",
          repr(float(np.linalg.norm(result.x - exact) / np.linalg.norm(exact))))
    print("QLS compact ring: q, epsilon_inv, degree, stored residual bound, delta, qubits, CX")
    for q in QLS_SIZES:
        for tolerance in QLS_TOLERANCES:
            selected = plan(qls_problem(q), method=QLS(epsilon_inv=tolerance), seed=SEED)
            workload = estimate(selected, context=CX)
            rec = selected.reconstruction
            cx = workload.quantity("cx")
            print("  ", q, tolerance, rec.degree, f"{rec.polynomial.certificate:.6g}",
                  f"{rec.phase_error:.2g}",
                  workload.quantity("logical_width", location="logical_device").fact.value.numerator,
                  f"{number(cx.fact.value):,.0f}", cx.interpretation)


def main(argv=None):
    """Write the LCHS figure and print the values of both guides."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", type=Path, default=FIGURE, help="SVG path to write")
    args = parser.parse_args(argv)
    rows, measured = lchs_values()
    bounds, total, steps = print_lchs(rows, measured)
    write_lchs_figure(bounds, total, steps, measured, args.figure)
    print("wrote", args.figure)
    print_qls()
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, "
          f"Qiskit {qiskit.__version__}, Aer {qiskit_aer.__version__}, "
          f"{platform.system()} {platform.machine()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
