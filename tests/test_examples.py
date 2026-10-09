"""Example notebooks: agreement with their sources and execution with scientific checks."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
BUILDER = EXAMPLES_DIR / "generators" / "build_notebooks.py"
NOTEBOOKS = tuple(sorted(path.stem for path in (EXAMPLES_DIR / "generators").glob("*.py") if path != BUILDER))
# A hang guard for one notebook, not a timing qualification.
NOTEBOOK_TIMEOUT_SECONDS = 900

# Each check runs in the executed notebook's namespace and compares a reported
# quantity with a reference that the notebook computes independently of the
# quantum method (FCI, np.linalg.solve, expm, grid evaluation, Euler formula).
CHECKS = {
    "gcim_lanczos_qpe_eigenvalue_intro": """
assert abs(result.eigenvalue - E_FCI) < CHEMICAL_ACCURACY
assert max(abs(lanczos_errors[-1]), abs(qcels_errors[-1])) < MILLIHARTREE * CHEMICAL_ACCURACY
# In the Pauli-sum branch a two-state basis spans the H2 ground state, so both methods reach the
# exact-diagonalization energy.
assert max(abs(my_lanczos.eigenvalue - my_exact), abs(my_adapt.eigenvalue - my_exact)) < 1e-10
# The H2 circuits reproduce the exact evaluation. They are one circuit per pair of the two-state basis,
# whose full-product circuits also give the gradients of both screenings.
assert abs(h2_circuits.eigenvalue - h2_exact.eigenvalue) < 1e-10
assert len({chunk.attempt for chunk in h2_circuits.data.observations.chunks}) == 3
""",
    "qls_linear_system_intro": """
assert np.linalg.norm(x - reference) / np.linalg.norm(reference) < 2 * EPSILON
assert 0 < result.algorithm_success_mass <= 1
""",
    "lchs_linear_dynamics_intro": """
assert relative_error < 0.05
assert abs(h * u.sum() - h * reference.sum()) < 0.05 * h * reference.sum()
# The constant-source branch against the Duhamel formula evaluated with expm.
assert np.linalg.norm(my_u - my_reference) < 0.05 * np.linalg.norm(my_reference)
""",
    "qhd_optimization_intro": """
assert probability[best] > 0.5
assert np.allclose(probability.sum(axis=1), result.marginals.array[0], atol=1e-12, rtol=0)
# The constrained run against an enumeration of its grid x_i = lower + (i + 1) h, h = (upper - lower)/(K + 1),
# written with the operations of grid.OneHotGrid.grid_value and the objective, so the values agree bitwise.
disk_axes = [[lo + (i + 1) * ((hi - lo) / (DISK_POINTS + 1)) for i in range(DISK_POINTS)] for lo, hi in DISK_BOUNDS]
disk_feasible = [(a - 1) ** 2 + (b - 1) ** 2 for a in disk_axes[0] for b in disk_axes[1]
                 if a**2 + b**2 - 1 <= disk.record.feasibility_tolerance]
assert disk_reference.objective == min(disk_feasible)
assert disk.objective == disk_reference.objective
assert disk.best.evaluation.infeasibility <= disk.record.feasibility_tolerance
""",
    "lchs_scientific": """
assert np.isclose(fact_value(verification_facts[0].fact), algorithm_error, rtol=1e-6, atol=0)
""",
    "qls_scientific": """
assert direction_error < no_evolution_error
assert fidelity > 0.99
""",
    # The QHD scientific notebook against its targets, the statements of its text, and two independent computations.
    "qhd_scientific": """
# The settings table of Section 11 tells the reader that the Ackley run uses 10 levels.
assert len(ackley.levels) == 10 and ackley.termination == "level_limit"
assert ackley.objective <= ACKLEY_TARGETS["objective"] and distance(ackley.candidate, SHIFT) <= ACKLEY_TARGETS["distance"]
assert constraint_values(summaries[4]["point"]).max() <= al_options.feasibility_tolerance
# The Z = 4 run, whose selected state Section 3 samples, meets both Rastrigin targets and stops feasible and
# complementary.
assert summaries[4]["gap"] <= RASTRIGIN_TARGETS["gap"] and summaries[4]["distance"] <= RASTRIGIN_TARGETS["distance"]
assert summaries[4]["stop"] == "feasible_complementary"
# The CX law of the 4-qubit binary circuit against its Qiskit compilation at optimization level 0.
assert predicted.quantity("cx").fact.value.numerator == compiled["operations"]["cx"]
# Section 3 samples the kept state of each selected level. Its most probable grid point and that point's
# probability must be the level record's point and point_probability. Owners: refinement.refine_box and
# _outer.grid_point, which read Result.most_probable_indices and most_probable_probability. The index is the
# lexicographically smallest point whose computed probability q = np.abs(z)**2 lies within the Result's tie window
# of the computed maximum (qhd.method._summarize). method._execute_theory reads q before one scalar phase, so the
# kept state gives raw = np.abs(fl(z exp(i phi)))**2 instead. In binary64, u = 2**-53 and
# gamma_n = n u/(1 - n u). With one-ulp phase entries and normal-range arithmetic, the elementwise phase error is
# <= [2u + sqrt(2) gamma_2 (1 + 2u)] |z| < 5u |z|. If complex abs has relative error <= 2u, each abs(z)**2 errs by
# at most gamma_5. Hence |raw - q|/q <= (20u + 25u**2)/(1 - 10u) < 21u, and rtol = 32u leaves room for the
# comparison, whose subtraction is exact for these normal peaks.
# Each raw probability differs from q by <= w q, w = 32u, so the top-two gap changes by <= 2 w max(raw)/(1 - w)
# < 65u max(raw). 128u max(raw) also covers the subtractions and the threshold addition. Above that guard the tie
# set holds only the maximum, and np.argmax of the normalized state finds it.
# A point is reported as x = a + (b - a) i / K rounded once from the exact value, which level_distribution
# reproduces, so the index and coordinate comparisons are exact. The normalization repeats level_distribution's
# operations on the same array, so it is exact as well.
u = 2.0**-53
for level, result, k, p, points in (
        (ackley_level, ackley.results[ackley.best_level - 1], qhd.num_grid_points, p_ackley, points_ackley),
        (rastrigin_level, rastrigin_result, rastrigin_qhd.num_grid_points, p_rastrigin, points_rastrigin)):
    raw = np.abs(np.asarray(result.data.artifact(result.artifact).array)) ** 2
    ordered = np.sort(raw)
    assert ordered[-1] - ordered[-2] > result.most_probable_tie_window + 128 * u * ordered[-1]
    peak = int(np.argmax(p))
    assert divmod(peak, k) == level.point_indices and tuple(points[peak]) == level.point
    assert np.isclose(raw[peak], level.point_probability, rtol=32 * u, atol=0)
    assert p[peak] == raw[peak] / raw.sum()
# No point of the selected Rastrigin grid meets the gap 0.01, so its probability is an empty sum and no shot
# of any draw lands in it.
assert shot_rows[2][2] == 0.0 and shot_rows[2][6] == 0

# Appendix D: the Aer state of the 4-qubit circuit against the dense reference of the notebook, fixed K = 4,
# 8-step physical Ackley product, T = 0.5, gamma = 0.3, binary64, u = 2**-53. The native and readout terms use
# NWQLib's qualified first-order model.
# Owners: _validation.native_state_error, qhd.circuit_errors, and the dense diagonal lowering in
# subroutines._multiplexors. Aer's 613-instruction state budget (G*c/2 + 5)u is 2.3674078121e-10. Angle formation
# adds 8.18e-15, the identity phase 3.83e-14, and the dense lowering and global phase 7.46e-15 + 2.15e-15, from
# exact signed RZ phase sums on all basis addresses and the native phases of this circuit.
# The reference's eight generators X have ||X||_inf < 0.1. A degree-12 Taylor recurrence T_j = T_(j-1) X/j has
# roundoff < 800u and tail < e**0.1 * 0.1**13/13! < 2e-23, each computed expm(X) lies within 1e-13 of it in the
# infinity norm (1.3e-16 observed), and ||M||_2 <= 4||M||_inf for 16-by-16 M, so ||expm(X) - exp(X)||_2 < 1e-12.
# The eight calls with their matrix-vector products and phase and angle formation cost < 1e-11. The different
# formation of the potential (< 3.553e-15) and kinetic matrix (< 5.235e-16) changes the exact product by
# < 2.08e-15, by telescoping and ||exp(-itA) - exp(-itB)|| <= |t| ||A - B|| for Hermitian A, B.
# With 32u for the comparison the total is < 2.5e-10, and the window 1e-9 holds under these premises. The
# physical-phase norm also checks the restored global phase. The circuit without the register transpose gives 0.619.
assert np.linalg.norm(circuit_state - reference_state) <= 1e-9

# The first Ackley level against an independent NumPy split-step loop that reads no NWQLib state: K = 16,
# 2048 midpoint steps, T = 10, gain 8, gamma = 0.3, the uniform start, V = 8 (F - min F)/(max F - min F) of the
# notebook's NumPy Ackley values on the level grid and the spectral eigenvalues 2 pi**2 (q1**2 + q2**2) on the
# unit square, lexicographic order with variable 0 first. The observed-host and readout terms use NWQLib's
# qualified first-order model.
k1, steps1, dt1 = 16, 2048, 10.0 / 2048
axis1 = -5 + 10 * np.arange(k1) / k1
values1 = ackley_value(np.array(list(product(axis1, axis1)))).reshape(k1, k1)
v1 = 8 * (values1 - values1.min()) / (values1.max() - values1.min())
q1 = np.fft.fftfreq(k1, d=1 / k1)
energy1 = 2 * np.pi**2 * (q1[:, None] ** 2 + q1[None, :] ** 2)
psi1 = np.full((k1, k1), 1 / k1, dtype=complex)
for step in range(steps1):
    t = (step + 0.5) * dt1
    a, b = 1 / (1 + 0.3 * t * t), 1 + 0.3 * t * t
    half = np.exp(-0.5j * dt1 * b * v1)
    psi1 = half * np.fft.ifft2(np.exp(-1j * dt1 * a * energy1) * np.fft.fft2(half * psi1))
p_numpy_level1 = (np.abs(psi1) ** 2).ravel()
# Owner of the NWQLib window: qhd.split_step.evolve through method._host_window, 4.59460787775e-11 here. The NumPy
# loop has its own qualification. Each normalized 2-D FFT or inverse FFT costs <= 512u: a 16-point DFT built
# from radicals has matrix error <= 32u, a dense application costs < 216u including the sqrt(2)*gamma_32*4
# inner-product error, two axes cost < 433u, and every transform of this trajectory has residual < 16u against
# that dense calculation (3.7u observed), which covers its 4096 calls. Per step there are two transforms and
# three phase applications of 5u each, and two angle roundings add 2u (sum(dt a) Emax + sum(dt b) 8), with
# sum(dt a) = 2.53816662643, sum(dt b) = 109.999994040 and Emax = 2526.61872668, so delta_B < 2.38e-10.
# The stored step weights agree, and the inputs differ by max|Delta V| < 8.771e-15 and max|Delta E| < 1.881e-13,
# so the two exact products differ by delta_input <= sum(dt a) max|Delta E| + sum(dt b) max|Delta V| < 1.45e-12.
# A probability moves by at most 2 delta + delta**2 for a state at distance delta, so
# |p_A - p_B| <= 4.59e-11 + 2 delta_B + delta_B**2 + 2 delta_input + 32u < 5.3e-10, where 32u covers the
# readout, the kept-state phase multiplication and the comparison. The window 1e-9 holds under these premises
# on raw probabilities. A finite-difference kinetic kernel changes a probability by 0.145.
first = ackley.results[0]
p_level1 = np.abs(np.asarray(first.data.artifact(first.artifact).array)) ** 2
assert np.max(np.abs(p_level1 - p_numpy_level1)) <= 1e-9
# If the NumPy top-two gap exceeds twice that window plus NWQLib's tie window, NWQLib's tie set holds only the
# NumPy maximum. Here the gap is 0.193.
ordered1 = np.sort(p_numpy_level1)
assert ordered1[-1] - ordered1[-2] > 2e-9 + first.most_probable_tie_window
index1 = int(np.argmax(p_numpy_level1))
assert first.most_probable_indices == (index1 // k1, index1 % k1)
""",
    # The resource notebook's references: the analytic Fourier endpoints, elementary operation and CX counts,
    # independent LCHS bounds, exact QLS residuals, an exact-rational commutator census with a commuting
    # counterexample, and the CX counts of the same constructions compiled at 4 to 8 system qubits (basis cx and u,
    # optimization level 0), including every prepared circuit of the two 4-spin QPE plans.
    # The comparison windows use binary64 unit roundoff u = 2**-53. They assume round-to-nearest, ties-to-even basic
    # arithmetic and Fraction-to-float conversion, normal nonzero intermediate values, relative error <= 2u for pow,
    # exp, log, expm1 and complex abs through the platform hypot, and <= 3u for math.fsum of positive terms. The
    # periodic Strang magnitude enclosure also assumes correctly rounded binary64 sqrt and nextafter returning the
    # adjacent representable value. The Strang bound is evaluated exactly after enclosing the stored
    # coefficient magnitudes. QPE uses a scaled outward coefficient checked against an independent exact
    # census within the derived window, then one common step whose exact prefix subtotals are published upward.
    # These windows compare the specified bound evaluations. They do not enclose rounding of the other
    # bound components, coefficient construction or circuit implementation.
    "resource_estimation_at_scale": """
from math import asinh, ceil, isfinite, isqrt, lgamma
from pytest import approx
U = 2.0**-53
assert all((qls[q].reconstruction.sigma_min, qls[q].reconstruction.sigma_max, qls[q].reconstruction.alpha,
            qls[q].reconstruction.kappa_be) == (1.0, 2.0, 2.0, 2.0) for q in SIZES)
# Elementary counts against the folds. Owners: backends/resources.py and algorithms/qls/quantum.py (QLS),
# algorithms/lchs/periodic.py::_construction_law (LCHS) and algorithms/qpe/powers.py::suzuki_step_cx and _make
# (QPE operation and CX laws).
assert all(formula == library for _, formula, library in qls_cx_checks + lchs_cx_checks + qpe_count_checks)
assert len(qpe_count_checks) == 2 * len(qpe)

# The windows below are derived for these LCHS inputs only.
assert (T, L_norm, beta, rule.node_count, rule.interval_count_each_side) == (1.0, 1.0, 0.75, 15, 9)
assert 32 <= K <= 33 and 0.48 < asinh(1.8 / (K / 9)) < 0.49
(_, B_ind, B_lib, _), (_, D_ind, D_lib, _), (_, tail_ind, tail_lib, _), (_, gauss_ind, gauss_lib, _) = lchs_bound_checks
assert all(isfinite(value) and value > 0 for value in (B_ind, B_lib, D_ind, D_lib, tail_ind, tail_lib, gauss_ind, gauss_lib))
# Strang coefficient, recorded by a one-step plan. Owner: algorithms/lchs/periodic.py::select_periodic_parameters,
# which forms B_up = T**3/2 * sum_j M_j (|k_j|/4)**3 exactly with M_j = periodic._magnitude_up(c_j) and records
# B_lib = error_budget._upward_float(B_up), the least binary64 number >= B_up. T = 1, diffusion 1/4 and potential 0
# here at q >= 3, and scaling a node by 1/4 and dividing by 2 is exact.
# Magnitude allowance for periodic._magnitude_up under the premises above.
# Write |c| = s*sqrt(1+t**2), t = min(|Re c|,|Im c|)/s, and M = s*R.
# Set A=(1+u)(1+2u), the upper factor for a normal RN-plus-nextafter step.
# The final s*R is rational. For t<1 and RN(1+square)<2, sum and root
# errors are <=3u absolute. For t<=1/2, square<=A**3*t**2 gives relative
# excess/u <= (d+12)/10+30/11 < 5, d=(A**3-1)/u, using
# sqrt(1+z)>=1+2*z/5 for z=t**2. For t>1/2, ratio-t<=3u/2 gives
# excess/u <= 9/4+30/11+7u < 5. At RN(1+square)=2, t>=1-5u/2
# and the argument is 2+4u, giving excess/u <= (123/28)/(1-5u/4) < 5.
# Unequal binary64 parts have t<=1-u. At t=1 the argument is 2+12u,
# R=0x1.6a09e667f3bd0p+0, and exact squares give R**2<2*(1+5u)**2.
# At t=0, R=1+2u. Outward monotonicity supplies the lower bound.
# Thus |c| <= M < (1+5u)|c| for each nonzero coefficient here.
# The independent sum has factors (1 +/- 2u)^2 (1 +/- u)^2 (1 +/- 3u), from complex abs,
# pow, multiplication, division and positive fsum. The owner forms the
# rational B_up exactly and rounds upward with relative error < 2u.
# With L = (1-2u)^2(1-u)^2(1-3u) and H = (1+2u)^2(1+u)^2(1+3u) the lower/upper products and
# C = (1 + 5u)(1 + 2u), max(H - 1, 1 - L/C) < 16u(1 - u), so rel=16u covers this comparison,
# including the rounding of approx's own tolerance. The two floats are within a factor of two, so their
# subtraction is exact. No factor grows with the 270 positive summands.
assert one_step.reconstruction.step_counts[0] == 1
assert B_ind == approx(B_lib, rel=16 * U, abs=0.0)
# Selected step count. Owners: select_periodic_parameters and error_budget._smallest_step_count.
# Selection inverts the exact B_up, before its upward display conversion.
# With L and H as above, 1/H <= B_up/B_ind <= (1+5u)/L lies inside 1 +/- 15u.
# Both exact endpoints must select r: upper <= e*r**2 and, for r > 1,
# lower > e*(r-1)**2. This makes the comparison of integer counts valid.
B_exact, e, eta = Fraction(B_ind), Fraction(STRANG_ALLOWANCE), Fraction(15, 2**53)
r = 1 + isqrt(max(0, ceil(B_exact / e) - 1))
assert B_exact * (1 + eta) <= e * r**2
if r > 1:
    assert B_exact * (1 - eta) > e * (r - 1) ** 2
assert r == lchs_steps_independent and lchs[100].reconstruction.step_counts
assert all(step == r for step in lchs[100].reconstruction.step_counts)
# Selected bound B/r**2. Owner: select_periodic_parameters. The independent B_ind/r**2 adds
# one rounding factor, since this integer square is exactly representable.
# The owner divides B_up exactly and rounds that result upward (< 2u).
# With the coefficient factors L, H and C above, the relative difference
# is <= max(H*(1+u)-1, 1-L*(1-u)/C) < 17u(1-u), giving rel=17u.
assert 0.0 < D_lib <= STRANG_ALLOWANCE and r**2 <= 2**53
assert D_ind == approx(D_lib, rel=17 * U, abs=0.0)
# Kernel tail, owner algorithms/lchs/providers.py::eq7_tail_bound. beta = 3/4 and 32 <= K <= 33 give x < 5.3.
# With shared rounded C_beta and cosine, the direct and log-domain evaluations have log-error budgets 32u and
# 128u, so their relative difference is <= exp(160u) - 1 < 256u. Qualifying the owner's fixed lgamma call by
# (0, 1) leaves its alternative ACL Lemma 10 bound larger by a factor > 29, so the direct tail is selected.
assert 0.0 < lgamma(3.0) < 1.0
assert tail_ind == approx(tail_lib, rel=256 * U, abs=0.0)
# Gauss quadrature, owner algorithms/lchs/providers.py::_ellipse_rule. Q = 15, 9 panels per half-axis and
# T*||L|| = 1 share rounded C_beta and s = asinh(1.8/h). For 0.48 < s < 0.49, exp(2s)/(exp(2s) - 1) < 1.65.
# Propagating the <= 2u elementary-function errors, including rho**28, gives log-error budgets 96u for the
# direct formula and 80u for the owner, so their relative difference is <= exp(176u) - 1 < 256u.
assert gauss_ind == approx(gauss_lib, rel=256 * U, abs=0.0)

# QLS residual, certificate owner subroutines/qsp/phases.py::chebyshev_norming_sup_bound. Fraction(float)
# represents each stored coefficient and candidate x exactly, and the rational Clenshaw recurrence evaluates
# kappa*x*P(x) - 1 without cancellation roundoff, so the comparison with Fraction(certificate) needs zero slack.
# NumPy's roots only choose the candidates. This checks their residuals, not completeness of the extrema or a
# rounding-enclosed supremum.
assert len(residual_checks) == 4
assert all(candidate <= certificate <= Fraction(tolerance) for tolerance, candidate, certificate in residual_checks)

# Sampled QCELS selects each power independently. Owner: algorithms/qpe/powers.py::_independent_powers, whose
# docstring states the law r_p = max(1, ceil(sqrt(W_up*|t_p|**3/(epsilon_p - |t_p|*d_up)))) and the complete
# bound B_p (docs/algorithms/qpe.md states the per-power selection). The notebook's check_power raised on any
# mismatch, so these assertions restate its results:
# - exact integers: each r_p equals the law evaluated by exact rational inversion from the recorded coefficient
#   W_up and the stored binary64 time, with zero pruning (d_up = 0), as the selector inverts it;
# - each recorded bound equals the upward rounding of B_p evaluated exactly from the stored binary64 parameters,
#   and lies between B_p at the exact census W and at W(1 + delta), both rounded upward, where delta is the derived
#   two-level reduction window of the scaled outward coefficient (notebook census_delta);
# - the bound fits the allowance, which planning requires before it publishes the power.
assert all(selected.reconstruction.common_step is None for selected in (*qpe.values(), commuting))
assert all(power.pruning_error == 0
           for selected in qpe.values() for power in selected.reconstruction.powers[1:])
assert len(qpe_power_checks) == len(qpe) + 1
assert all(len(rows) == len(qpe[key].reconstruction.powers) - 1 if key in qpe else len(rows) == 32
           for key, rows in qpe_power_checks)
assert all(blo <= recorded <= bhi and recorded <= POWER_ALLOWANCE
           for _, rows in qpe_power_checks for p, r, r_exact, recorded, blo, bhi in rows)
# Commuting terms give W = 0, so the law selects the minimum of one step at every power.
assert all(r == 1 for key, rows in qpe_power_checks if key[0] == "Commuting" for p, r, *_ in rows)
assert len(qpe_checks) == len(qpe) + 1
# The recheck charge is NWQLib's own admission value (powers.independent_recheck_work), not an independent
# quantity. Planning succeeded, so each charge fits the Method's max_work.
assert all(row[-1] <= neel_qcels[100].max_work for row in qpe_checks)

# Quantities that the notebook displays as unavailable stay unavailable.
assert all(number(folded.quantity(metric)) is None for folded in (qls_cx[100], lchs_cx[100])
           for metric in ("logical_depth", "t"))
# Ten periodic constructions (QLS and LCHS at five sizes) and four QPE plans (two chains on 4 and 8 spins).
# prepare(plan, settings="all") builds all 66 circuits of each QPE plan without submission, and their compiled count
# weights each circuit by the shots of its setting. The CX laws are exact integers, compared exactly.
assert [row["settings"] for row in compiled_checks if "settings" in row] == [66] * 4
assert [row["q"] for row in compiled_checks] == [*SMALL_SIZES, *SMALL_SIZES, *SMALL_SPINS, *SMALL_SPINS]
assert len(compiled_checks) == 14 and all(row["law"] == row["compiled"] for row in compiled_checks)

# QHD, Section 4. Widths, CX and rotation laws against the construction counts of Appendix A, as exact integers.
# Owners: the compact program's resource laws in algorithms/qhd/method.py, algorithms/qhd/resources.py::rotation_law
# and the binary synthesis selection in algorithms/qhd/binary.py.
assert set(qhd_plans) == {(k, encoding) for k in (8, 16, 32) for encoding in ("one_hot", "binary")}
for (k, encoding), qhd_p in qhd_plans.items():
    m = k.bit_length() - 1
    cx = qhd_cx[k, encoding].quantity("cx")
    rotations = qhd_logical[k, encoding].quantity("arbitrary_rotations")
    width = qhd_logical[k, encoding].quantity("logical_width", location="logical_device")
    assert rotations.fact.value is not None
    assert cx.interpretation == "upper_bound"
    assert width.interpretation == "exact"
    if encoding == "one_hot":
        assert number(width) == 3 * k
        assert number(cx) == 9 * (k - 1) + 6 * k + 4 * (k - 1)**2
        assert rotations.interpretation == "exact"
        assert number(rotations) == 6 * (k - 2) + 6 * k + 3 * (k - 1) + 6 * (k - 1)**2
    else:
        assert number(width) == 3 * m
        assert number(cx) == 6 * m * (m - 1) + 3 * (k - 2) + 3 * m * (m - 1) + 4 * m*m
        assert rotations.interpretation == "upper_bound"
        assert number(rotations) == 9 * m * (m - 1) + 3 * (k - 1) + 3 * m * (m + 1) // 2 + 2 * (m*m + 2*m)
    # Ledger scope, owner algorithms/qhd/resources.py::circuit_resources. Unavailable sources stay None, distinct
    # from zero and from not_applicable.
    sources = {s.name: s for s in qhd_ledgers[k, encoding].error_sources}
    # kinetic_model compares a spectral kinetic operator with the finite-difference stencil, which both encodings
    # apply here.
    assert sources["kinetic_model"].status == "not_applicable" and sources["kinetic_model"].value is None
    assert sources["rotation_pruning"].value == 0
    assert sources["coefficient_rounding"].status == "bound"
    assert sources["aqft"].status == "not_applicable"
    assert sources["compiler"].status == sources["synthesis"].status == "unavailable"
    assert sources["compiler"].value is sources["synthesis"].value is None
    if encoding == "binary":
        assert sources["diagonal_wrap"].status == "unavailable"
        assert sources["diagonal_wrap"].value is None
        assert sources["state_preparation"].status == "bound"
        assert sources["state_preparation"].value == 0
    else:
        assert sources["diagonal_wrap"].status == "not_applicable"
        assert sources["state_preparation"].status == "estimate"

# QHD small circuits of Appendix E, compiled at optimization level 0, owners algorithms/qhd/native.py (emitter) and
# algorithms/qhd/resources.py::circuit_resources. The CX law equals the compiled count, the census of the emitted
# gate definitions agrees with circuit_resources, and the rotation law is exact for one-hot and an upper bound for
# binary.
assert set(qhd_compiled) == {(4, "one_hot"), (4, "binary"), (8, "binary")}
for (k, encoding), row in qhd_compiled.items():
    assert row["width"] == (3*k if encoding == "one_hot" else 3*(k.bit_length() - 1))
    assert row["cx"] == row["cx_law"]
    assert (row["emitted_arbitrary"], row["emitted_exact_t"]) == (row["ledger_arbitrary"], row["ledger_exact_t"])
    assert row["emitted_arbitrary"] <= row["rotation_law"]
    assert row["rotation_interpretation"] == ("exact" if encoding == "one_hot" else "upper_bound")
    if encoding == "one_hot":
        assert row["emitted_arbitrary"] == row["rotation_law"]
assert qhd_compiled[4, "binary"]["diagonal_wrap"] == "not_applicable"
assert qhd_compiled[8, "binary"]["diagonal_wrap"] == "unavailable"
# The stored support tables and constant (formed in algorithms/qhd/method.py) equal the quadratic at every grid tuple
# of the three small plans, in exact rational arithmetic.
assert qhd_objective_matches
# Ideal-evolution bound, owner algorithms/qhd/evolution_bounds.py::evolution_bound. The dyadic tables and their
# differences are exact, each nextafter allowance contributes at most 2u, and at most six later upward conversions of
# positive rational expressions are covered by (1 + 2u)**7 - 1 < 32u, so the outward library value lies in
# [B_ref, B_ref(1 + 32u)]. This checks the evaluation of the formulas, not the elementary-function premises.
u_qhd = Fraction(1, 2**53)
assert len(qhd_bound_checks) == 6
assert all(ref <= Fraction(value) <= ref * (1 + 32*u_qhd) for ref, value in qhd_bound_checks)
# At K = 32 the one-hot splitting term exceeds the cap, so the recorded bound is exactly 2.
assert qhd_ledgers[32, "one_hot"].evolution.evolution == 2

# GCiM, Section 5, owner algorithms/gcim/fixed_basis.py. For m = 2 normalized trial states and G = 2
# qubit-wise-commuting groups (Z...Z and X...X) the grouped plan has 2(M - D)max(1, G) + DG = m**2 G = 8 settings
# (fixed_basis.sampled_settings) and 1024 shots each (Appendix A). The preparation laws give 8q CX per
# off-diagonal setting and zero per diagonal setting, hence 32q*1024 for all eight settings. Exact integers.
from nwqlib.algorithms.gcim.fixed_basis import sampled_settings
assert set(gcim) == set(SIZES)
for q, folded in gcim_cx.items():
    settings = [s[0] for s in sampled_settings(2, gcim[q].reconstruction.groups)]
    assert [g.basis for g in gcim[q].reconstruction.groups] == ["X" * q, "Z" * q]
    assert number(folded.quantity("settings")) == len(gcim[q].experiments) == len(settings) == 2**2 * 2
    assert [e.name for e in gcim[q].experiments] == settings
    assert number(folded.quantity("shots")) == 1024 * len(settings)
    assert number(folded.quantity("logical_width", location="logical_device")) == q + 1
    assert folded.quantity("cx").interpretation == "upper_bound"
    assert number(folded.quantity("cx")) == 32 * q * 1024
# The three-qubit plan's eight prepared circuits, compiled at optimization level 0 and weighted by their shots, stay
# within the bound that estimate reports. Only the four off-diagonal settings compile to CX gates. The text states
# the ratio R = 2 of the bound to this compiled count.
assert gcim_compiled["circuits"] == 8 and sum(cx > 0 for cx in gcim_small_cx) == 4
assert gcim_small_cx == [0, 0, 12, 12, 12, 12, 0, 0]
assert sum(gcim_small_cx) == 16 * 3 == 48
assert gcim_compiled["law"] == 32 * 3 * 1024 == 98304
assert gcim_compiled["compiled"] == 1024 * sum(gcim_small_cx) == 49152
assert gcim_compiled["interpretation"] == "upper_bound" and gcim_compiled["law"] == 2 * gcim_compiled["compiled"]
""",
}


def test_notebooks_match_their_sources():
    completed = subprocess.run([sys.executable, str(BUILDER), "--check"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_notebook_has_a_scientific_check():
    assert set(NOTEBOOKS) == set(CHECKS)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_runs_and_matches_its_references(name):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("matplotlib")
    if name.startswith("gcim"):
        pytest.importorskip("pyscf", reason="the molecular Hamiltonian requires the chemistry extra")
        pytest.importorskip("openfermion", reason="the molecular Hamiltonian requires the chemistry extra")

    document = nbformat.read(EXAMPLES_DIR / f"{name}.ipynb", as_version=4)
    for cell in document.cells:
        if cell.cell_type == "code":
            cell.outputs = []
    # Bind the kernel to this interpreter and checkout, then verify both inside the kernel.
    document.cells.insert(0, nbformat.v4.new_code_cell(
        "import sys, nwqlib\nfrom pathlib import Path\n"
        f"assert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve()\n"
        f"assert Path(nwqlib.__file__).resolve() == Path({str(REPO_ROOT / 'src/nwqlib/__init__.py')!r}).resolve()\n"))
    document.cells.append(nbformat.v4.new_code_cell(CHECKS[name]))
    client = nbclient.NotebookClient(document, timeout=NOTEBOOK_TIMEOUT_SECONDS,
                                     resources={"metadata": {"path": str(EXAMPLES_DIR)}})
    manager = client.create_kernel_manager()
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client.execute()
