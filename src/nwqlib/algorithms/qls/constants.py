"""QLS normalization, spectral-premise and verification constants."""

# Rescale margin over the rigorous Chebyshev norming bound. The phase solver
# demands max|target| <= 1, and dividing a selected polynomial by its norming
# bound times 1 + 1e-3 leaves the phase target with norming bound
# 1/(1 + 1e-3). Every sampled QLS phase target at this margin converges,
# but not always from the L-BFGS start. With the default epsilon_inv = 0.01,
# the sampled targets on which L-BFGS missed the tolerance were all inverse
# targets with polynomial_kappa between 5.01 and 5.72 (degrees 29 to 33),
# and the Newton start of subroutines/qsp/phases.py solved each of them.
# The margin costs 0.1% of success amplitude. Registered in
# docs/ENGINEERING_CONSTANTS.md with the sampled targets. Revisit if a fit
# change makes a QLS phase solve fail at this margin.
QLS_TARGET_MARGIN = 1.0e-3
# Polynomial-domain floor, separate from actual alpha/sigma_min. A perfectly
# conditioned input has actual condition 1, while the polynomial constructions
# need a domain parameter strictly above 1. The 1.01 floor also avoids the
# kernel-reflection (KR) coefficient formula's cancellation near 1. This does
# not alter the reported original spectrum, actual alpha or condition
# estimate. Revisit if the domain requirement of either polynomial
# construction or the kernel-reflection coefficient fit changes.
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
