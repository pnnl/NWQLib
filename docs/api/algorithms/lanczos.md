# Lanczos API

Configure the initial state and Krylov dimension on `Lanczos`, and supply the operator and target through `Eigenproblem`. The Result reports a projected Ritz value, without establishing full-space ground coverage. See the [Lanczos guide](../../algorithms/lanczos.md) for raw Chebyshev moments, cutoff analysis and sensitivity sampling.

The construction follows Kirby, Motta and Mezzacapo, arXiv 2208.00567v4, Sections 2.1–3.1. The shift and rescaling follow Oumarou et al., arXiv 2603.15552v1, whose sensitivity allocation NWQLib adapts, and the thresholded pencil solve follows Epperly, Lin and Nakatsukasa, arXiv 2110.07492v2, Algorithm 1.1. The guide's [source and code map](../../algorithms/lanczos.md#source-and-code-map) gives the equation, page and owning function of each step.

::: nwqlib.algorithms.lanczos.method.Lanczos

::: nwqlib.algorithms.lanczos.method.SensitivitySampling

::: nwqlib.algorithms.lanczos.records.LanczosResult

The Plan fixes the frame `H=cI+alpha*K`, in which the spectrum of K lies in [-1,1]. Its `LanczosReconstruction` stores c as `center`, alpha as `alpha` and the trial dimension m as `krylov_dimension`, so S and the projected matrices are m by m. In `LanczosResult`, `moments` are `mu_k=<psi|T_k(K)|psi>` for the normalized initial state, and `moment_variances` are their empirical sample-mean variances. `overlap` is the raw Chebyshev Gram matrix `S_ij=(mu_(i+j)+mu_|i-j|)/2`, and `hamiltonian` is the physical projected matrix `c*S+alpha*K_proj`, where `K_proj_ij=(mu_(i+j+1)+mu_|i+j-1|+mu_|i-j+1|+mu_|i-j-1|)/4` is K projected onto the trial space. Both use raw Chebyshev coordinates. `eigenvalue` is the lowest physical Ritz value `c+alpha*x` of that pencil after the Gram eigenvalues at or below `cutoff` are discarded, `kept_rank` counts the kept directions and `cutoff_source` names the rule that set the cutoff.

`gram_sampling_bound` is `m*e` with `e=sqrt(2*log(2*r/delta)/n_min)`, where r counts the sampled moments that enter S, n_min is the smallest of their shot counts and delta is `analysis_failure_probability`. With probability at least 1-delta it bounds the spectral norm of the Gram sampling error, provided the shots are independent outcomes in [-1,1] and the acquisitions are unbiased. The bound comes from Hoeffding (1963), doi:10.1080/01621459.1963.10500830, Theorem 2, Eq. (2.6), p. 16, made two-sided as in Eq. (1.4), p. 13, and a union bound over the r moments. `empirical_gram_noise_frobenius_rms` estimates the root-mean-square Frobenius norm of the same error from sample variances, without a coverage claim. Neither quantity bounds the error of `eigenvalue`.
