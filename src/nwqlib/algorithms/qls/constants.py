"""QLS normalization, spectral-assumption and verification constants."""

# Rescale margin over the Chebyshev norming bound. Canonical Wx synthesis
# (subroutines/qsp/phases.py) needs a target whose norming bound is strictly
# below 1, and dividing the selected polynomial by its norming bound times
# 1 + 1e-3 gives the phase target the norming bound 1/(1 + 1e-3). At this
# margin every QLS target sampled at epsilon_inv = 0.01 (400 inverse values
# of polynomial_kappa from 1.01 to 50, 300 kernel-reflection values from 1.01
# to 80) synthesizes within the 1e-12 residual tolerance. The margin costs
# 0.1% of success amplitude. Registered in docs/ENGINEERING_CONSTANTS.md with
# the sampled targets. Revisit if the norming evaluation or the synthesis cap
# changes.
QLS_TARGET_MARGIN = 1.0e-3
# Engineering ratio c of the default phase target of "qsvt_inverse",
# delta = max(1e-12, c * epsilon_inv / (polynomial_kappa * rescale))
# (quantum._phase_tolerance). When the second argument decides and the actual
# residual meets it, the phase term k*s*delta/||y|| of the relative inverse
# error is at most c*epsilon_inv, because ||y|| >= 1. It is not taken from
# epsilon_inv, which stays the polynomial's own target, so the degree does not
# change. It is neither an optimal constant nor a guarantee on the total error.
# Revisit if a public workload shows a material accuracy or cost change.
QLS_PHASE_RELATIVE_TARGET_SHARE = 0.1
# Polynomial-domain floor, separate from actual alpha/sigma_min. A perfectly
# conditioned input has actual condition 1, while the polynomial constructions
# need a domain parameter strictly above 1. This does not alter the reported
# original spectrum, actual alpha or condition estimate. Revisit if the domain
# requirement of either polynomial construction changes.
QLS_POLYNOMIAL_KAPPA_FLOOR = 1.01

# Relative admission window for already computed singular endpoints, matching
# the relative 1e-12 guard with which _dense_dilation_encoding in
# subroutines/block_encoding/core.py rejects an alpha below ||A||_2. This is a
# numerical consistency tolerance, not a certified spectral or physical-error
# bound. Revisit with a changed precision or spectral estimator.
QLS_SPECTRAL_PREMISE_RTOL = 1e-12

# Explicit verification criteria, separate from record admission
# (docs/ENGINEERING_CONSTANTS.md, "Explicit QLS verification criteria").
# These are numerical or heuristic windows, not spectral or confidence
# theorems.
# - SPECTRAL_COVERAGE_TOLERANCE: dimensionless window on the lower and upper
#   spectral-domain deficits, wide enough for ordinary auto-kappa rounding.
#   Revisit only with an independently justified coverage criterion.
# - EQ17_ABSOLUTE_WINDOW: absolute additive window in automatic Dalzell
#   arXiv:2406.12086v2 Eq. (17) mass comparisons. It keeps the allowance
#   positive when the phase and shot terms vanish. The Eq. (17) shot term has
#   no variance floor. Revisit with a derived roundoff bound for the Eq. (17)
#   endpoints.
# - INVERSE_BINOMIAL_VARIANCE_FLOOR: lower bound on the predicted per-trial
#   Bernoulli variance in the inverse success-mass comparison. With it, a
#   predicted rate of exactly 0 or 1 still has a positive shot allowance.
#   Revisit if the shot term adopts a finite-sample confidence method.
QLS_SPECTRAL_COVERAGE_TOLERANCE = 1e-9
QLS_EQ17_ABSOLUTE_WINDOW = 2e-12
QLS_INVERSE_BINOMIAL_VARIANCE_FLOOR = 1e-12
