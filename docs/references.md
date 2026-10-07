# References

Bibliographic references and equation locations for NWQLib's algorithms, subroutines, and research comparisons. Each algorithm guide explains which constructions are implemented and which assumptions apply. Entries marked as background describe related work rather than an available method. Equation numbers refer to the linked paper versions. This page is the public citation catalogue.

## Where each source is used

Each page below gives the paper, equation or section behind the implemented steps. The algorithm guides and the block-encoding, LCU, state-preparation and Pauli-decomposition pages also name the code that computes each step and state where NWQLib departs from the paper.

| Topic | Source map |
| --- | --- |
| Finite Pauli expectation and binary inference | [Expectation guide](algorithms/expectation.md#source-and-code-map) |
| Chebyshev Lanczos | [Lanczos guide](algorithms/lanczos.md#source-and-code-map) |
| LCHS | [LCHS guide](algorithms/lchs.md#source-map) |
| QPE estimators, product-formula powers and coherent QPE | [QPE guide](algorithms/qpe.md#implementation-map), [Trotterization](api/subroutines/trotterization.md), [Coherent QPE](api/subroutines/qpe.md#source-map) |
| QLS, QSP phases and QSVT evolution | [QLS guide](algorithms/qls.md#sources-and-code-map), [QSP and QSVT](api/subroutines/qsp.md) |
| GCiM and ADAPT-GCiM | [GCiM guide](algorithms/gcim.md#source-and-code-map), [Fermionic pools](api/subroutines/fermionic_pool.md#compact-generator-circuits) |
| QHD, its augmented-Lagrangian layer and box refinement | [QHD guide](algorithms/qhd.md#source-map) |
| Block encodings, LCU, state preparation, Pauli decomposition and Pauli evolution | [Block encoding](api/subroutines/block_encoding.md#source-map), [LCU](api/subroutines/lcu.md#source-map), [State preparation](api/subroutines/state_preparation.md#source-map), [Pauli decomposition](api/subroutines/pauli_decomposition.md#source-map), [Hamiltonian evolution](api/subroutines/hamiltonian_evolution.md#source-map) |
| Selected blocks, CX formulas and fault-tolerant resource models | [Compose blocks](blocks.md#source-map), [Exact dense synthesis](development/dense_synthesis.md#source-map), [Estimate resources](resources.md#source-map), [Estimate fault-tolerant resources](fault-tolerant-resources.md#models-and-sources) |
| Bounds, resource formulas and error budgets, with the proof or source of each | [Mathematics](mathematics.md) |
| Operator conventions: Pauli order, Jordan–Wigner images, double factorization | [Inputs](inputs.md#conventions-and-derivations) |
| Scientific example notebooks | [Examples](examples.md#scientific-notebooks) and [Scientific examples](#scientific-examples) below |

The rules for saving, loading, running and backends come from NWQLib's design and have no paper source. [Design rationale](development/design_rationale.md) indexes the failure each one prevents and the page that defines it.

## Expectation and sampling

 - Hoeffding, W. (1963), *Probability Inequalities for Sums of Bounded Random Variables*, Journal of the American Statistical Association 58(301), 13–30. [DOI](https://doi.org/10.1080/01621459.1963.10500830), [Paper](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).
    - The linked scan reproduces the journal pages, and page numbers below are journal pages.
    - Theorem 1, Eq. (2.3), p. 15: P(mean - mu >= t) <= exp(-2 n t^2) for independent 0 <= X_i <= 1 and 0 < t < 1 - mu. The paragraph after Eq. (2.4) on that page replaces t by t/(b-a) when a <= X_i <= b, so outcomes in [-1,1] give exp(-n t^2/2).
    - Theorem 2, Eq. (2.6), p. 16: both bounds directly for every t > 0, with b_i - a_i = 1 or 2.
    - Eq. (1.4), p. 13: the lower tail, which doubles the bound for a two-sided deviation.
    - The Expectation shot selection and binary intervals cite Theorem 1, and the Lanczos Gram sampling bound cites Theorem 2.
    - The count screen that accepts a stall-split valley in QHD box refinement applies Theorem 1 with both tails and a union bound over cells and levels.
    - Eqs. (4.11) and (4.16), p. 22, in the proof of Theorem 2: the exponential bound behind that count screen. The paper notes that this proof also gives a direct derivation of Eq. (2.3).
    - The Hoeffding branch of QHD's selected region coverage ([Proposition 49](mathematics.md#r49)) applies Theorem 1 and Eqs. (1.4), (2.3) with a union bound over contiguous index intervals and levels, using half the total failure budget.
    - NWQLib's binary inference uses the bounded-variable convention X in [-1,1]. Weighted Pauli sums apply the triangle inequality and a union bound over the predeclared setting family.
    - Sampling-model assumptions and binary64 evaluation remain explicit, as described in the [expectation guide](algorithms/expectation.md).

 - Clopper, C. J., & Pearson, E. S. (1934). *The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial*. Biometrika 26(4), 404–413. [DOI](https://doi.org/10.1093/biomet/26.4.404).
    - QHD's selected region coverage uses one-sided binomial-test inversion, with a union bound over all candidate joint boxes and levels. It receives the other half of the failure budget in [Proposition 49](mathematics.md#r49).

## ADAPT-VQE

Gradient measurement references distinguish generator-spectrum assumptions, commuting measurement groups, and shot costs.

 - Grimsley, H. R., Economou, S. E., Barnes, E., & Mayhall, N. J. (2019). An adaptive variational algorithm for exact molecular simulations on a quantum computer. Nature communications, 10(1), 3007. [10.1038/s41467-019-10988-2](https://doi.org/10.1038/s41467-019-10988-2) [arXiv 1812.11173v2](https://arxiv.org/abs/1812.11173v2)
    - arXiv:1812.11173v2, Section II.B: the gradient-screening loop.
 - Anastasiou, P. G., Mayhall, N. J., Barnes, E., & Economou, S. E. (2023). How to really measure operator gradients in ADAPT-VQE. arXiv preprint arXiv:2306.03227. [arXiv 2306.03227v3](https://arxiv.org/abs/2306.03227v3)
    - arXiv:2306.03227v3, Eq. (2): each gradient as the commutator expectation ⟨[H, A]⟩.
    - Sections III.A–III.E: commuting groups and conditional cost comparisons.
 - Wierichs, D., Izaac, J., Wang, C., & Lin, C. Y.-Y. (2022). General parameter-shift rules for quantum gradients. Quantum, 6, 677. [arXiv 2107.12390v3](https://arxiv.org/abs/2107.12390v3)
    - arXiv:2107.12390v3, Sections 3.2–3.5 and Appendix B: finite-frequency derivative rules. A primitive two-shift rule does not automatically apply to an entire spin-adapted unitary.
 - Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., & Killoran, N. (2019). Evaluating analytic gradients on quantum hardware. Physical Review A, 99, 032331. [arXiv 1811.11184v1](https://arxiv.org/abs/1811.11184v1)
    - arXiv:1811.11184v1, Sec. III.A, Theorem 1, Eq. (8) and Eqs. (13)–(14): the two-eigenvalue identity behind the Pauli-insertion derivative.
    - Sec. III.B, Eqs. (16)–(17): the decomposition of a derivative over unitaries, here the Pauli terms of a generator. NWQLib inserts each Pauli factor after the complete generator exponential.
 - Romero, J., Babbush, R., McClean, J. R., Hempel, C., Love, P. J., & Aspuru-Guzik, A. (2019). Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz. Quantum Science and Technology, 4(1), 014008. [arXiv 1701.02691v2](https://arxiv.org/abs/1701.02691v2)
    - arXiv:1701.02691v2, Eqs. (8)–(9) and (17): the UCCSD singles and doubles comparison pool.
 - Yordanov, Y. S., Armaos, V., Barnes, C. H. W., & Arvidsson-Shukur, D. R. M. (2021). Qubit-excitation-based adaptive variational quantum eigensolver. Communications Physics, 4, 228. [arXiv 2011.10540v2](https://arxiv.org/abs/2011.10540v2)
    - arXiv:2011.10540v2, Eqs. (17)–(18): the qubit-excitation comparison pool.
 - Ramôa, M., Anastasiou, P. G., Santos, L. P., Mayhall, N. J., Barnes, E., & Economou, S. E. (2025). Reducing the Resources Required by ADAPT-VQE Using Coupled Exchange Operators and Improved Subroutines. npj Quantum Information, 11, 86. [arXiv 2407.08696v3](https://arxiv.org/abs/2407.08696v3)
    - arXiv:2407.08696v3, Sec. II.B.3, Eqs. (23)–(24): the OVP-CEO comparison pool.

## Generator-Coordinate-Inspired Method (GCIM)

 - Zheng, M., Peng, B., Wiebe, N., Li, A., Yang, X., & Kowalski, K. (2023). Quantum algorithms for generator coordinate methods. Physical Review Research, 5(2), 023200. [10.1103/PhysRevResearch.5.023200](https://doi.org/10.1103/PhysRevResearch.5.023200) [arXiv 2212.09205v1](https://arxiv.org/abs/2212.09205v1)
    - Eq. (13) and Eqs. (14)–(15): the discretized Hill–Wheeler problem and matrix elements. These numbers follow arXiv:2212.09205v1.
 - Zheng, M., Peng, B., Li, A., Yang, X., & Kowalski, K. (2024). Unleashed from constrained optimization: quantum computing for quantum chemistry employing generator coordinate inspired method. npj Quantum Information, 10, 127. [10.1038/s41534-024-00916-8](https://doi.org/10.1038/s41534-024-00916-8) [arXiv 2312.07691v3](https://arxiv.org/abs/2312.07691v3)
    - Numbered as in arXiv:2312.07691v3.
    - METHODS: the adaptive basis growth.
    - Appendix E: the operator pool.
    - Appendix F: positive-eigenvalue truncation.
    - Appendix G: the Hadamard-test estimators.
 - Epperly, E. N., Lin, L., & Nakatsukasa, Y. (2022). A theory of quantum subspace diagonalization. SIAM Journal on Matrix Analysis and Applications, 43(3), 1263-1290. [arXiv 2110.07492v2](https://arxiv.org/abs/2110.07492v2)
    - arXiv:2110.07492v2, Algorithm 1.1: thresholding.
    - Theorem 2.7: the analysis of thresholding for noisy projected matrices.
 - Magoulas, I., & Evangelista, F. A. (2026). Spin-Adapted Fermionic Unitaries: From Lie Algebras to Compact Quantum Circuits. [arXiv 2511.13485v2](https://arxiv.org/abs/2511.13485v2)
    - arXiv:2511.13485v2, Section III, Eqs. (3)–(7): the singlet spin-adapted generators, defined from the spin-orbital excitations of Eqs. (1)–(2).
    - Sections V–VI: the compact fermionic factorizations, with closed-form Wei–Norman parameters in Tables I–II, the Lie-algebra basis elements those parameters multiply in Supplemental Tables SI–SII and gate counts in Table III.
    - The tabulated gate counts do not include every routing and outer-control cost in a library circuit.
 - Jain, P., Izmaylov, A. F., & Kjellgren, E. R. (2026). Exact Factorization of Unitary Transformations with Spin-Adapted Generators. [arXiv 2511.14914v3](https://arxiv.org/abs/2511.14914v3)
    - Sections II.3–II.4 and Table I: alternative factorizations, used as comparison background.
 - Sun, Q., et al. (2018). PySCF: the Python-based simulations of chemistry framework. Wiley Interdisciplinary Reviews: Computational Molecular Science, 8, e1340. [arXiv 1701.08223](https://arxiv.org/abs/1701.08223)
    - Chemistry reference calculations.
 - Møller, C., & Plesset, M. S. (1934). Note on an Approximation Treatment for Many-Electron Systems. Physical Review, 46, 618–622. [10.1103/PhysRev.46.618](https://doi.org/10.1103/PhysRev.46.618)
    - MP2 background.
 - Purvis, G. D., III, & Bartlett, R. J. (1982). A full coupled-cluster singles and doubles model: The inclusion of disconnected triples. Journal of Chemical Physics, 76, 1910–1918. [10.1063/1.443164](https://doi.org/10.1063/1.443164)
    - CCSD background.
 - Abraham, Senapati, Pathak, and Peng (2025). Elucidating many-body effects in molecular core spectra through real-time approaches: Efficient classical approximations and a quantum perspective. [arXiv 2511.17985v1](https://arxiv.org/abs/2511.17985v1)
    - Core-spectra background.
 - Tsuchimochi, T., Mori, Y., & Ten-no, S. L. (2020). Spin-projection for quantum computation: A low-depth approach to strong correlation. Physical Review Research, 2(4), 043142. [10.1103/PhysRevResearch.2.043142](https://doi.org/10.1103/PhysRevResearch.2.043142) [arXiv 2004.12024v2](https://arxiv.org/abs/2004.12024v2)
    - Spin-projection background.
 - Zhu, P., Argentati, M. E., & Knyazev, A. V. (2013). Bounds for the Rayleigh quotient and the spectrum of self-adjoint operators. SIAM Journal on Matrix Analysis and Applications, 34(1), 244–256. [10.1137/120884468](https://doi.org/10.1137/120884468) [arXiv 1207.3240v2](https://arxiv.org/abs/1207.3240v2)
    - Section 2, Eqs. (2.6) and (2.3): the residual inclusion and Temple bounds that [Mathematics, Result 17](mathematics.md#r17) states for the ADAPT residual check.

## Linear Combination of Hamiltonian Simulation (LCHS)

The algorithm guide states the scope of NWQLib's finite quadrature and error information.

 - An, D., Childs, A. M., & Lin, L. (2026). Quantum algorithm for linear non-unitary dynamics with near-optimal dependence on all parameters. Communications in Mathematical Physics, 407(1), 19. [arXiv 2312.03916v2](https://arxiv.org/abs/2312.03916v2)
    - arXiv:2312.03916v2, Eq. (7): the kernel.
    - Eq. (61): the composite quadrature.
    - Eqs. (185)–(186): the pointwise decay. NWQLib integrates that decay and applies `E1(x) <= exp(-x)/x`, then selects the cutoff by scalar bisection.
    - Lemma 11 step rule, Eqs. (64)–(65): assumes T·max‖L‖ ≥ 32/e, so NWQLib selects Gauss panels with Trefethen's ATAP Theorem 19.3.
    - Sec. 2.2 (p. 14): the practical range [0.7, 0.8], in which the default `beta=.75` lies.
 - Trefethen, L. N. *Approximation Theory and Approximation Practice*. SIAM, 2013, ISBN 978-1-61197-239-9.
    - Chapter 19, Theorem 19.3 and Eq. (19.8), as numbered in the [chapter source](https://www.chebfun.org/ATAP/chap19.m).
 - An, D., Liu, J. P., & Lin, L. (2023). Linear combination of hamiltonian simulation for nonunitary dynamics with optimal state preparation cost. Physical Review Letters, 131(15), 150603. [10.1103/PhysRevLett.131.150603](https://doi.org/10.1103/PhysRevLett.131.150603)
    - doi:10.1103/PhysRevLett.131.150603: the original LCHS identity with the Cauchy kernel, which An, Childs, and Lin restate as Eq. (5) and NWQLib offers as `cauchy_density`.
 - Low, G. H., & Somma, R. D. (2025). Optimal quantum simulation of linear non-unitary dynamics. [arXiv 2508.19238v2](https://arxiv.org/abs/2508.19238v2)
    - arXiv:2508.19238v2, Eq. (6) with the parameters of Theorems 2–4: the fixed `j=2, y=1` kernel.
 - NIST Digital Library of Mathematical Functions, [§3.5(v)](https://dlmf.nist.gov/3.5).
    - Eq. 3.5.19: the Gauss quadrature remainder.
    - Eq. 3.5.21: the Gauss–Legendre constants.
    - The constant-source remainder combines DLMF Eqs. 3.5.19 and 3.5.21.
 - Al-Mohy, A. H., & Higham, N. J. (2009). A new scaling and squaring algorithm for the matrix exponential. SIAM Journal on Matrix Analysis and Applications, 31(3), 970–989. [10.1137/09074721X](https://doi.org/10.1137/09074721X).
    - Algorithm 5.1: the algorithm of both SciPy kernels that the `expm` and `closed_form` references use.
    - Code Fragment 2.1: recomputes the diagonal and first superdiagonal of a triangular input during squaring.
 - Higham, N. J. (2008). Functions of Matrices: Theory and Computation. SIAM. [10.1137/1.9780898717778](https://doi.org/10.1137/1.9780898717778).
    - Eq. (10.42), as `_eq_10_42` in SciPy's `scipy/sparse/linalg/_matfuncs.py` cites it: evaluates the first superdiagonal of Al-Mohy and Higham's Code Fragment (above) without cancellation.

Related differential-equation algorithms and simulation constructions:

 - Pocrnic, Johnson, Katabarwa, and Wiebe (2025). Constant-Factor Improvements in Quantum Algorithms for Linear Differential Equations. [arXiv 2506.20760v2](https://arxiv.org/abs/2506.20760v2)
    - arXiv:2506.20760v2, Lemma 3: inverts the weaker bound of Eq. (62) of An, Childs, and Lin with Lambert W, where NWQLib integrates their pointwise decay.
    - Section IV, Eqs. (61)–(65): the effective Hamiltonian used by the QSP route.
 - Aftab, An, and Trivisa (2026). Linear Combination of Hamiltonian Simulation with Commutator Scaling. [arXiv 2606.11475v1](https://arxiv.org/abs/2606.11475v1)
    - Multiproduct-formula and quadrature background.
 - An, Onwunta, and Yang (2024). Fast-forwarding quantum algorithms for linear dissipative differential equations. [arXiv 2410.13189v2](https://arxiv.org/abs/2410.13189v2)
    - Truncated-Dyson background.

Sources listed in other sections:

 - Childs et al. (2021), doi:10.1103/PhysRevX.11.011020, listed under Subroutines: Propositions 9–10 for the product-formula steps.

## Lanczos

 - Kirby, W., Motta, M., & Mezzacapo, A. (2023). Exact and efficient Lanczos method on a quantum computer. Quantum, 7, 1018. [DOI](https://doi.org/10.22331/q-2023-05-23-1018), [arXiv 2208.00567v4](https://arxiv.org/abs/2208.00567v4)
    - arXiv:2208.00567v4, Sections 2.1–3.1, Eqs. (2)–(4), (6)–(7) and (13)–(27): the signed Pauli encoding, the qubitized walk, the Chebyshev pencil and the parity readout. NWQLib uses a coherent SELECT-basis readout variant.
 - Oumarou, O., Ollitrault, P. J., Polla, S., & Gogolin, C. (2026). Optimizing and Comparing Quantum Resources of Statistical Phase Estimation and Krylov Subspace Diagonalization. [arXiv 2603.15552v1](https://arxiv.org/abs/2603.15552v1)
    - arXiv:2603.15552v1, Section 2, Eqs. (1) and (4): the shift and rescaling.
    - Eqs. (14)–(15): the assembly of the projected matrices from moments.
    - Eqs. (16)–(18): reading the moments from walk powers.
    - Section 3.3.2, Eqs. (59)–(60): motivates sensitivity allocation. NWQLib's pilot-dependent allocation and independent main samples have the conditional variance argument described in the [Lanczos guide](algorithms/lanczos.md).
    - Appendix A.3: a real-THC construction, distinct from the signed-Pauli encoding used here.

Sources listed in other sections:

 - Epperly, Lin, and Nakatsukasa (arXiv 2110.07492v2, listed under GCIM), Algorithm 1.1: overlap thresholding. NWQLib's default cutoff uses empirical Gram noise for exploratory regularization.
 - Hoeffding (doi:10.1080/01621459.1963.10500830, listed under Expectation and sampling), Theorem 2: NWQLib's Lanczos separately reports a bound from this theorem and a union bound on the sampled moments entering the Gram matrix, and its explicit confidence policy cuts at twice that bound.

## Quantum Hamiltonian Descent (QHD)

 - Leng, J., Hickman, E., Li, J., & Wu, X. (2023). Quantum Hamiltonian Descent. arXiv preprint arXiv:2303.01471. [10.48550/arXiv.2303.01471](https://doi.org/10.48550/arXiv.2303.01471) [arXiv 2303.01471v1](https://arxiv.org/abs/2303.01471v1)
    - arXiv:2303.01471v1, Eq. (1): the time-dependent Hamiltonian.
    - Eq. (C.4): NWQLib's cubic schedule.
    - Eq. (E.3) and Algorithm 1: first-order product-formula evolution.
    - Eqs. (F.7) and (F.9): the finite-difference stencil and grid potential.
    - Eq. (C.3), p. 32: the first-order pseudo-spectral step with the left-endpoint weights of Eq. (C.2), p. 31. NWQLib's split-step variant applies its symmetric (Strang) form (doi:10.1137/0705041), with which Wu et al., Sec. VI, p. 7, simulate their benchmarks.
    - Algorithm 1, p. 44: meshes the box with the endpoint-exclusive grid of Eq. (E.1), p. 43, which NWQLib's periodic grid uses. Its step 3 names the uniform and a Gaussian state as possible starts.
    - Eq. (F.15), p. 47: the diagonal potential of d variables.
    - Eq. (F.36): the tensor-product Hamming encoding, which NWQLib does not use.
 - Kushnir, S., Leng, J., Peng, Y., Fan, L., & Wu, X. (2025). QHDOPT: A Software for Nonlinear Optimization with Quantum Hamiltonian Descent. INFORMS Journal on Computing, 37(1), 107–124. [10.1287/ijoc.2024.0587](https://doi.org/10.1287/ijoc.2024.0587) [arXiv 2409.03121v1](https://arxiv.org/abs/2409.03121v1)
    - QHDOPT, arXiv:2409.03121v1, Sec. 2.1: NWQLib's default quadratic schedule, given as an example. The paper reports that schedules of this form work well for many test problems.
 - Wu, Z., Li, M., Zheng, M., Wang, M., Liu, J., Stein, S., Li, A., Chen, Y., & Liu, C. (2026). Benchmarking and Resource Analysis for Augmented-Lagrangian Quantum Hamiltonian Descent. [arXiv 2605.12066v1](https://arxiv.org/abs/2605.12066v1)
    - Sec. V, Eqs. (12)–(14), gives the adaptive box-refinement procedure. NWQLib derives its simultaneous selected-region coverage bound from Hoeffding and one-sided Clopper–Pearson in [Proposition 49](mathematics.md#r49).
    - Eq. (15) of arXiv:2605.12066v1, Sec. VI, p. 7: the shifted cubic schedule.
    - arXiv:2605.12066v1, Section IV.A, Eqs. (9)–(10): the one-hot occupation-operator encoding that NWQLib's potential follows.
    - Sec. III.B, Eqs. (6)–(8): the equality terms and multiplier update of the augmented-Lagrangian layer, whose inequality term comes from Rockafellar (below).
    - The paper keeps the penalty fixed or increases it by update rules that it cites from Nocedal and Wright, doi:10.1007/978-0-387-40065-5. The layer's `every_iteration` option reproduces the growth after every round of the authors' code.
    - Sec. V, Eqs. (12)–(13), p. 7: box refinement normalizes each marginal by the valid mass and keeps on each axis an interval that holds a chosen fraction of it. Their Eq. (13) states no tie rule, so the tie rules are NWQLib's.
    - Eq. (14), p. 7: each kept point represents the cell to its right, where NWQLib uses the cell centered on it.
    - The paper links no code, and the experiment code behind its published results has no public location.
 - Liu, C., Wang, M., Li, M., Zheng, M., Stein, S., & Chen, Y. (2026). Encoding Choices and Fault-Tolerant Resource Estimates for Digital Quantum Hamiltonian Descent. [arXiv 2607.16996v1](https://arxiv.org/abs/2607.16996v1)
    - arXiv:2607.16996v1, Eq. (92): the cubic schedule with s = 1, written as their QHD-C schedule. Their code runs the shifted cubic form under that name.
    - arXiv:2607.16996v1, Sec. III, Eq. (12): the one-hot hopping Hamiltonian with the wrap term that NWQLib's periodic grid uses.
    - Eq. (94): the position mean and standard deviation in the final state, which QHD analysis reports conditional on a valid outcome.
    - Sec. IV A, Eqs. (53)–(54): the expansion of a support-local potential block in Pauli-Z strings, which NWQLib's binary encoding offers as Walsh-rotation synthesis beside the exact phase diagonal of Shende, Bullock and Markov, arXiv:quant-ph/0406176v5, Theorem 7.
    - Sec. IV E, Eqs. (89)–(91): the quadratic low-momentum kinetic phase with an unsigned momentum index. NWQLib uses the signed index.
    - The paper links no code and makes additional information available from the corresponding author on request. The QHD guide and docstrings name files of the code of either paper (this one or Wu et al.) only to state which settings it ran.
 - Strang, G. (1968). On the construction and comparison of difference schemes. SIAM Journal on Numerical Analysis, 5(3), 506–517. [10.1137/0705041](https://doi.org/10.1137/0705041)
    - The symmetric splitting of the split-step variant.
 - Coppersmith, D. (2002). An approximate Fourier transform useful in quantum factoring. IBM Research Report RC 19642 (1994). [arXiv quant-ph/0201067v1](https://arxiv.org/abs/quant-ph/0201067v1)
    - arXiv:quant-ph/0201067v1: the approximate quantum Fourier transform that the binary kinetic factor can select.
 - Rockafellar, R. T. (1973). A dual approach to solving nonlinear programming problems by unconstrained optimization. Mathematical Programming, 5, 354–373. [10.1007/BF01580138](https://doi.org/10.1007/BF01580138)
    - The Powell–Hestenes–Rockafellar form of doi:10.1007/BF01580138: the inequality term of the augmented-Lagrangian layer.
 - Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer, Springer Series in Operations Research and Financial Engineering. ISBN 978-0-387-30303-1. [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
    - The source of the penalty update rules that Wu et al. cite (above).
 - Birgin, E. G., & Martínez, J. M. (2014). *Practical Augmented Lagrangian Methods for Constrained Optimization*. SIAM, Fundamentals of Algorithms 10. ISBN 978-1-611973-35-8. [10.1137/1.9781611973365](https://doi.org/10.1137/1.9781611973365)
    - Algorithm 4.1 and Eqs. (4.7)–(4.9) of doi:10.1137/1.9781611973365: the layer's tentative multipliers, default penalty test and multiplier safeguard.
    - Eq. (10.4): the layer's scaled multipliers.
    - Eq. (10.3), Sec. 10.1, p. 114: the layer's effective objective.
    - Sec. 10.2.2, Eqs. (10.6)–(10.8): the layer's stopping test and stationarity diagnostic.
    - The docstrings of the layer also cite the book's Problem 4.6 (p. 38) for placing a bound constraint in the box or among the relaxed constraints, Problem 4.8 (p. 38) for the multiplier update from a squared slack, and Problem 4.9 (p. 39) for the alternative V of the complementarity test.
    - For the conditions of the book's convergence results, which the layer states and does not claim, the docstrings cite Assumption 5.1 (p. 41), Theorems 5.1 and 5.2 (pp. 41–42), Assumption 6.1 (p. 48), Sec. 7.5 with Assumptions 7.6 and 7.7 (p. 64), and Sec. 7.7 with Theorem 7.2 and Assumption 7.9 (p. 70).
 - Li, M., Fan, L., & Han, Z. (2025). Quantum Hamiltonian Descent based Augmented Lagrangian Method for Constrained Nonconvex Nonlinear Optimization. [arXiv 2508.02969v1](https://arxiv.org/abs/2508.02969v1)
    - Augmented-Lagrangian QHD background. The paper (arXiv:2508.02969v1) also combines an augmented Lagrangian with QHD and simulates the QHD dynamics on classical hardware with the simulated bifurcation algorithm.
 - Ross, N. J., & Selinger, P. (2016). Optimal ancilla-free Clifford+T approximation of z-rotations. Quantum Information and Computation, 16(11–12), 901–953. [arXiv 1403.2975v3](https://arxiv.org/abs/1403.2975v3)
    - arXiv:1403.2975v3: the leading term `3 log2(1/eps)` of the typical T count of one z-rotation, which the T estimate of a QHD circuit uses.
 - Al-Mohy, A. H., & Higham, N. J. (2011). Computing the action of the matrix exponential, with an application to exponential integrators. SIAM Journal on Scientific Computing, 33(2), 488–511. [10.1137/100788860](https://doi.org/10.1137/100788860)
    - Algorithm 3.2 (doi:10.1137/100788860): SciPy's `expm_multiply`, which computes the matrix-exponential action in the classical QHD evolution.
    - Eqs. (3.5)–(3.15), Table 3.1, Lemma 4.1 and Eq. (4.7): the error analysis that supports the derived [mass window](glossary.md#mass-window) of that evolution.
 - Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen. Zeitschrift für Angewandte Mathematik und Mechanik, 54(1), 39–51. [10.1002/zamm.19740540106](https://doi.org/10.1002/zamm.19740540106).
    - The compensated summation that the QHD compiler uses for its running phase total and that Python uses in its float `sum` since 3.12.
 - Ogita, T., Rump, S. M., & Oishi, S. (2005). Accurate sum and dot product. SIAM Journal on Scientific Computing, 26(6), 1955–1988. [10.1137/030601818](https://doi.org/10.1137/030601818).
    - Algorithm 4.1, the cascaded summation: the recurrences of Neumaier's compensated summation.
    - Proposition 4.5: the error bound of the equivalent Algorithm 4.4. This gives the error bounds of the QHD running phase total and of the kinetic coefficient's `sum` that the kept-state phase allowance of the QHD circuit routes uses.
 - Higham, N. J. (1993). The accuracy of floating point summation. SIAM Journal on Scientific Computing, 14(4), 783–799. [10.1137/0914050](https://doi.org/10.1137/0914050).
    - Sec. 3, Eq. (3.6), p. 788: the rounding bound that the QHD binary angle formation applies to the butterfly sums of the Walsh–Hadamard transform.

## Quantum Linear Solver (QLS)

 - Childs, A. M., Kothari, R., & Somma, R. D. (2017). Quantum algorithm for systems of linear equations with exponentially improved dependence on precision. SIAM Journal on Computing, 46(6), 1920-1950. [arXiv 1511.02306v2](https://arxiv.org/abs/1511.02306v2)
    - arXiv:1511.02306v2, Lemmas 17–19: inverse-polynomial approximation bounds from which the degree formula d = O(κ log(κ/ε)) follows.
 - Sünderhauf, Nemeth, Walayat, Patterson, and Berntson (2025). Matrix inversion polynomials for the quantum singular value transformation. [arXiv 2507.15537v1](https://arxiv.org/abs/2507.15537v1)
    - arXiv:2507.15537v1, Theorem 1, with Python code in Appendix B: the optimal odd polynomial in closed form.
    - Sec. III, Eq. (26): the grid density.
    - Eq. (25): its equidistant-x statement is not valid.
 - Ehlich, H., and Zeller, K. (1964). Schwankung von Polynomen zwischen Gitterpunkten. *Mathematische Zeitschrift* **86**, 41–44. [10.1007/BF01111276](https://doi.org/10.1007/BF01111276), [Original scans at GDZ](https://gdz.sub.uni-goettingen.de/dms/resolveppn/?PPN=GDZPPN002394847).
    - doi:10.1007/BF01111276, Satz 2, Eqs. (12)–(14), pp. 42–43, on Chebyshev-zero nodes: the norming factor used by NWQLib.
 - Dalzell, A. M. (2024). A shortcut to an optimal quantum linear system solver. arXiv preprint arXiv:2406.12086. [arXiv 2406.12086v2](https://arxiv.org/abs/2406.12086v2)
    - arXiv:2406.12086v2: the shortcut solver construction.
    - The [QLS guide](algorithms/qls.md#sources-and-code-map) records two errata in Dalzell arXiv:2406.12086v2, the elimination direction of Sec. 5.2 and the absolute value in Lemma 3, item 2.
 - Costa, Dalzell, An, and Berry (2026). Constant factor analysis of optimal quantum linear solvers in practice. [arXiv 2604.22185v2](https://arxiv.org/abs/2604.22185v2)
    - arXiv:2604.22185v2: examines the practical constants of the shortcut solver construction of Dalzell.
 - Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for linear systems of equations. Physical review letters, 103(15), 150502. [10.1103/PhysRevLett.103.150502](https://doi.org/10.1103/PhysRevLett.103.150502) [arXiv 0811.3171v3](https://arxiv.org/abs/0811.3171v3)
    - Related background.
 - Low, G. H., & Su, Y. (2026). Quantum linear system algorithm with optimal queries to initial state preparation. Quantum, 10, 2041. [arXiv 2410.18178v2](https://arxiv.org/abs/2410.18178v2)
    - Related background.
 - Costa, P. C. S., An, D., Sanders, Y. R., Su, Y., Babbush, R., & Berry, D. W. (2022). Optimal scaling quantum linear systems solver via discrete adiabatic theorem. PRX Quantum, 3, 040303. [arXiv 2111.08152v1](https://arxiv.org/abs/2111.08152v1)
    - Related background.
 - Gilyén et al. (2019, arXiv:1806.01838v1), listed under Block encoding and quantum signal processing.

## Quantum Phase Estimation (QPE)

The [QPE guide](algorithms/qpe.md#implementation-map) gives the exact versions, the code that computes each step and the RWPE feedback-sign derivation.

 - Ding, Z., & Lin, L. (2023). Even Shorter Quantum Circuit for Phase Estimation on Early Fault-Tolerant Quantum Computers with Applications to Ground-State Energy Estimation. PRX Quantum, 4(2), 020331. [10.1103/PRXQuantum.4.020331](https://doi.org/10.1103/PRXQuantum.4.020331) [arXiv 2211.11973v2](https://arxiv.org/abs/2211.11973v2)
    - arXiv:2211.11973v2, Eq. (2): the quantum complex exponential least-squares (QCELS) objective.
    - Eqs. (11)–(12): the elimination of its amplitude. NWQLib implements one level.
    - Theorems 1–2: assume squared overlap p0 > 0.71.
    - Section IV: Fourier filtering for smaller overlaps.
 - Wan, K., Berta, M., & Campbell, E. T. (2022). Randomized quantum algorithm for statistical phase estimation. Physical Review Letters, 129, 030503. [arXiv 2110.12071v2](https://arxiv.org/abs/2110.12071v2)
    - arXiv:2110.12071v2 PDF Eqs. (A1)–(A2), HTML Eqs. (16)–(17): the Fourier filter of the approximate CDF, Eq. (6), for statistical phase estimation (SPE).
    - Algorithm 1: compares the sampled approximate CDF with the threshold η/2.
    - Theorem 3 with PDF Eqs. (A6)–(A7) and (A12), HTML Eqs. (21)–(22) and (27): the filter domain that NWQLib checks.
    - The SPE controlled-evolution backend has a separate synthesis model from the paper's randomized LCU compiler.
 - Kshirsagar, R., Katabarwa, A., & Johnson, P. D. (2024). On proving the robustness of algorithms for early fault-tolerant quantum computers. Quantum, 8, 1531. [10.22331/q-2024-11-20-1531](https://doi.org/10.22331/q-2024-11-20-1531) [arXiv 2209.11322v3](https://arxiv.org/abs/2209.11322v3)
    - Equation numbers follow arXiv:2209.11322v3, which also carries the sign correction to Eq. (4) and to the output line of Algorithm 1.
    - arXiv:2209.11322v3, Algorithm 1 and Eq. (7): the largest sampled Fourier coefficient for randomized Fourier estimation (RFE).
    - Theorem 2.1: a sample count that the finite default does not meet.
 - Granade, C., & Wiebe, N. (2022). Using random walks for iterative phase estimation. arXiv preprint arXiv:2208.04526. [arXiv 2208.04526v1](https://arxiv.org/abs/2208.04526v1)
    - arXiv:2208.04526v1, Algorithm 1 and Eq. (7): the Gaussian moments and one-bit updates for random-walk phase estimation (RWPE).
    - Algorithm 2: the unwinding, which the QPE guide distinguishes from basic RWPE.
 - Cleve, R., Ekert, A., Macchiavello, C., & Mosca, M. (1998). Quantum algorithms revisited. Proceedings of the Royal Society of London A, 454, 339–354. [10.1098/rspa.1998.0164](https://doi.org/10.1098/rspa.1998.0164) [quant-ph/9708016v1](https://arxiv.org/abs/quant-ph/9708016v1)
    - arXiv:quant-ph/9708016v1 Section 5, Fig. 6 and Eq. (5.1): the phase register that the coherent QPE subroutine prepares with controlled powers of U.
    - Eqs. (5.2)–(5.4): for an eigenvector input, the inverse QFT then yields the nearest phase estimate on the register's grid with probability at least 4/π².
    - The [coherent QPE API](api/subroutines/qpe.md#source-map) maps their bit order to Qiskit's.
 - Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms, 2nd ed. SIAM. [10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027).
    - Lemma 3.1: the constant `gamma_n = n*u/(1-n*u)` for unit roundoff u. The QPE projector roundoff window uses it, and so does the exact-probability window that bounds QPE `raw_mean` and the QLS and QHD reported masses.
    - QHD also cites:
        - Theorem 2.2, p. 38, and Eq. (2.4), p. 40, for the rounding of a result in the normal range.
        - Eq. (2.8), p. 56, for gradual underflow, the basis of the range rule of planning.
        - Theorem 2.5, p. 45 (Sterbenz), for the exact subtraction in the most-probable selection.
        - Sec. 4.2, Eq. (4.6), p. 83, for the pairwise summation bound of the Walsh coefficients.
        - Sec. 4.3, Eq. (4.10), p. 85, for the backward error of compensated summation.

## Subroutines

Equation and figure numbers follow the listed arXiv versions. The exact synthesis of a dense unitary, for a controlled gate, inside a controlled definition, in the MPS preparation or before a backend lowers the circuit, draws on four papers: Shende, Markov and Bullock (2004), Vatan and Williams, Krol and Al-Ars, and Shende, Bullock and Markov (2006).

 - Childs, A. M., & Wiebe, N. (2012). Hamiltonian simulation using linear combinations of unitary operations. Quantum Information and Computation, 12, 901-924. [arXiv 1202.5822v1](https://arxiv.org/abs/1202.5822v1)
    - arXiv:1202.5822v1, Lemma 2: the two-term linear combination of unitaries that the PREP–SELECT–PREP† standard form of Low and Chuang (2019) generalizes.
 - Childs, A. M., Su, Y., Tran, M. C., Wiebe, N., & Zhu, S. (2021). Theory of Trotter error with commutator scaling. Physical Review X, 11, 011020. [10.1103/PhysRevX.11.011020](https://doi.org/10.1103/PhysRevX.11.011020) [arXiv 1912.08854v3](https://arxiv.org/abs/1912.08854v3)
    - doi:10.1103/PhysRevX.11.011020, Proposition 9/Eq. (120) and Proposition 10/Eq. (121): product-formula error bounds.
    - Section V B: the smallest-step-count rule.
    - These numbers follow the Physical Review X version. The arXiv:1912.08854v3 preprint, titled "A Theory of Trotter Error", numbers the same results Proposition 15/Eq. (145), Proposition 16/Eq. (152) and Section 5.2.
 - Shende, V. V., Bullock, S. S., & Markov, I. L. (2006). Synthesis of quantum-logic circuits. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 25, 1000–1010. [10.1109/TCAD.2005.855930](https://doi.org/10.1109/TCAD.2005.855930) [quant-ph/0406176v5](https://arxiv.org/abs/quant-ph/0406176v5)
    - arXiv:quant-ph/0406176v5, Theorems 7–8 and 13, Eq. (19) and Table 1: the multiplexor and dense-unitary CX counts.
    - Theorem 4: the two-CX singly controlled rotation used when an affine angle table replaces a rotation multiplexor.
    - Appendix A, optimization A.2: the diagonal moved between two-qubit blocks.
    - Theorem 12: a controlled unitary is block diagonal in each control qubit, and this theorem demultiplexes it at the top.
 - Shende, V. V., & Markov, I. L. (2009). On the CNOT-cost of TOFFOLI gates. Quantum Information and Computation, 9, 461–486. [arXiv 0803.2316v1](https://arxiv.org/abs/0803.2316v1)
    - arXiv:0803.2316v1, Fig. 1 and Theorem 1: reproduce the textbook six-CX Toffoli circuit of Nielsen and Chuang that the controlled CX formulas use, and prove that no circuit of CX and one-qubit gates implements the Toffoli with fewer.
 - Nielsen, M. A., & Chuang, I. L. (2000). Quantum computation and quantum information. Cambridge University Press. ISBN 978-0-521-63503-5.
    - The textbook source of the six-CX Toffoli circuit, cited here through its reproduction by Shende and Markov.
 - Bergholm, V., Vartiainen, J. J., Möttönen, M., & Salomaa, M. M. (2005). Quantum circuits with uniformly controlled one-qubit gates. Physical Review A, 71, 052330. [quant-ph/0410066v2](https://arxiv.org/abs/quant-ph/0410066v2)
    - arXiv:quant-ph/0410066v2, Section III and Fig. 6(a): the decomposition that Qiskit's `UCGate` implements, which realizes a uniformly controlled one-qubit gate with k controls by 2^k one-qubit gates and 2^k − 1 CX up to a diagonal gate.
 - Shukla, A., & Vedula, P. (2024). An efficient quantum algorithm for preparation of uniform quantum superposition states. Quantum Information Processing, 23, 38. [10.1007/s11128-024-04258-4](https://doi.org/10.1007/s11128-024-04258-4) [arXiv 2306.11747v2](https://arxiv.org/abs/2306.11747v2)
    - arXiv:2306.11747v2, Algorithm 1 and Section 2.5: the uniform-superposition preparation and gate counts that Qiskit's `UniformSuperpositionGate` implements.
 - Möttönen, M., Vartiainen, J. J., Bergholm, V., & Salomaa, M. M. (2005). Transformation of quantum states using uniformly controlled rotations. Quantum Information and Computation, 5, 467. [quant-ph/0407010v1](https://arxiv.org/abs/quant-ph/0407010v1)
    - arXiv:quant-ph/0407010v1, Sections II–III: the uniformly controlled rotations and state-preparation angles.
 - Ran, S.-J. (2020). Encoding of matrix product states into quantum circuits of one- and two-qubit gates. Physical Review A, 101, 032310. [arXiv 1908.07958v2](https://arxiv.org/abs/1908.07958v2)
    - arXiv:1908.07958v2, Section II, Eqs. (6)–(9): one MPS disentangling layer, applied layer by layer in Section III.
 - Oseledets, I. V. (2011). Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5), 2295–2317. [10.1137/090752286](https://doi.org/10.1137/090752286)
    - doi:10.1137/090752286, Algorithm 1 (TT-SVD, p. 2301): produces the MPS cores.
    - Theorem 2.2 (p. 2299): the orthogonal splitting in its proof, which Eq. (2.5) states as a bound, gives the truncation error estimate of the MPS compression analysis. Its norm and fidelity estimates add NWQLib's inner-product step from the same orthogonality.
    - NWQLib drops each singular value at or below a fixed threshold instead of keeping the paper's delta-rank.
 - Shende, V. V., Markov, I. L., & Bullock, S. S. (2004). Minimal universal two-qubit controlled-NOT-based circuits. Physical Review A, 69, 062321. [quant-ph/0308033v3](https://arxiv.org/abs/quant-ph/0308033v3)
    - arXiv:quant-ph/0308033v3, Proposition IV.3 and its proof: the KAK decomposition of a two-qubit unitary in the magic basis.
    - Proposition V.2: a diagonal gate whose product with the unitary needs only two CX.
 - Vatan, F., & Williams, C. (2004). Optimal quantum circuits for general two-qubit gates. Physical Review A, 69, 032315. [quant-ph/0308006v3](https://arxiv.org/abs/quant-ph/0308006v3)
    - arXiv:quant-ph/0308006v3, Sec. V and Fig. 6: the three-CX circuit of exp(i(a XX + b YY + c ZZ)).
 - Krol, A. M., & Al-Ars, Z. (2024). Beyond quantum Shannon: circuit construction for general n-qubit gates based on block ZXZ-decomposition. [arXiv 2403.13692v2](https://arxiv.org/abs/2403.13692v2)
    - arXiv:2403.13692v2, Eqs. (5)–(10): the block-ZXZ form of the quantum Shannon decomposition.
    - Sec. 5.2 and Eq. (11): merge two CX into its central block.
 - Litinski, D. (2019). Magic state distillation: Not as costly as you think. Quantum, 3, 205. [arXiv 1905.06903v3](https://arxiv.org/abs/1905.06903v3)
    - Background for the distillation-factory model that the QDK physical projection applies.
 - Horsman, D., Fowler, A. G., Devitt, S., & Van Meter, R. (2012). Surface code quantum computing by lattice surgery. New Journal of Physics, 14, 123011. [arXiv 1111.4022v3](https://arxiv.org/abs/1111.4022v3)
    - Sec. 7.1: the rotated-lattice data qubits behind the QDK patch size.
    - Sec. 6: the d rounds of error correction per operation.
 - Wang, D. S., Fowler, A. G., & Hollenberg, L. C. L. (2011). Quantum computing with nearest neighbor interactions and error rates over 1%. Physical Review A, 83, 020302(R). [arXiv 1009.3686v1](https://arxiv.org/abs/1009.3686v1)
    - Figs. 1(b) and 2: the four-CNOT syndrome circuit behind the QDK code cycle.
    - The paper reports thresholds of 1.1% to 1.4%.
 - Fowler, A. G., Mariantoni, M., Martinis, J. M., & Cleland, A. N. (2012). Surface codes: Towards practical large-scale quantum computation. Physical Review A, 86, 032324. [arXiv 1208.0928v2](https://arxiv.org/abs/1208.0928v2)
    - Eqs. (10) and (11): the empirical logical error rate whose 0.03 prefactor and exponent the QDK model uses.

Sources listed in other sections:

 - Low and Chuang (2019, arXiv:1610.06546v3, listed under block encoding), Lemma 5: the single-register PREP–SELECT–PREP† standard form.

## Block encoding and quantum signal processing

 - Gilyén, A., Su, Y., Low, G. H., & Wiebe, N. (2019). Quantum singular value transformation and beyond. STOC 2019. [arXiv 1806.01838v1](https://arxiv.org/abs/1806.01838v1)
    - Definition 43 and Corollary 60: the block-encoding definition and Hamiltonian-simulation degree bounds.
    - Corollary 18: QSVT by real polynomials.
    - Theorem 28: robust oblivious amplitude amplification.
    - Lemma 57: the Jacobi–Anger truncation.
    - Lemma 61: the perturbation bound for Hermitian Hamiltonian simulation.
    - These numbers follow arXiv:1806.01838v1.
 - Low and Chuang (2019). Hamiltonian simulation by qubitization. Quantum, 3, 163. [arXiv 1610.06546v3](https://arxiv.org/abs/1610.06546v3)
    - arXiv:1610.06546v3: the qubitized walk and its eigenphase relation.
 - Low, G. H., & Chuang, I. L. (2017). Optimal Hamiltonian simulation by quantum signal processing. Physical Review Letters, 118, 010501. [arXiv 1606.02685v2](https://arxiv.org/abs/1606.02685v2)
    - QSP background.
 - Martyn, Rossi, Tan, and Chuang (2021). Grand unification of quantum algorithms. PRX Quantum, 2, 040203. [arXiv 2105.02859v5](https://arxiv.org/abs/2105.02859v5)
    - arXiv:2105.02859v5: QSP conventions.
 - Dong, Meng, Whaley, and Lin (2021). Efficient phase-factor evaluation in quantum signal processing. PRA, 103, 042419. [arXiv 2002.11649v2](https://arxiv.org/abs/2002.11649v2)
    - arXiv:2002.11649v2: symmetric phase-factor optimization.
 - Dong, Lin, Ni, and Wang (2023). Robust iterative method for symmetric quantum signal processing in all parameter regimes. [arXiv 2307.12468v1](https://arxiv.org/abs/2307.12468v1)
    - arXiv:2307.12468v1, Sec. 3, Eq. (3.1) and Algorithm 3.1: the Newton method that the phase solver runs when L-BFGS misses its tolerance.
    - Sec. 2.2: the convergence limits of L-BFGS near `max|f| = 1`.
 - Haah, J. (2019). Product decomposition of periodic functions in quantum signal processing. Quantum, 3, 190. [arXiv 1806.10236v4](https://arxiv.org/abs/1806.10236v4)
    - Comparison background.
 - Camps, D., & Van Beeumen, R. (2022). FABLE: fast approximate quantum circuits for block-encodings. IEEE Quantum Computing and Engineering. [arXiv 2205.00081v2](https://arxiv.org/abs/2205.00081v2)
    - Comparison background.
 - Camps, D., Lin, L., Van Beeumen, R., & Yang, C. (2024). Explicit quantum circuits for block encodings of certain sparse matrices. SIAM Journal on Matrix Analysis and Applications, 45, 801. [arXiv 2203.10236v4](https://arxiv.org/abs/2203.10236v4)
    - arXiv:2203.10236v4, Theorem 4.1 and Sec. 4.2: the banded-circulant circuit, comparison background. The banded encoding is compared with its normalization.
 - Draper, T. G. (2000). Addition on a quantum computer. [quant-ph/0008033v1](https://arxiv.org/abs/quant-ph/0008033v1)
    - The QFT adder behind the shift operators of the banded block encoding.
 - Motlagh, D., & Wiebe, N. (2024). Generalized Quantum Signal Processing. PRX Quantum, 5(2), 020368. [10.1103/PRXQuantum.5.020368](https://doi.org/10.1103/PRXQuantum.5.020368) [arXiv 2308.01501v2](https://arxiv.org/abs/2308.01501v2)
    - Background for the planned LCHS construction in the [Limitations and open work page](ROADMAP.md#scalable-lchs-select).

## Testing

The metamorphic relations of the [Maintenance page](MAINTENANCE.md#metamorphic-relations-and-their-premises) follow the testing approach of MorphQ, which checks Qiskit by transforming quantum programs in ways whose effect on the output is known. NWQLib's relations and their acceptance limits come from its own design rules.

 - Paltenghi, M., & Pradel, M. (2023). MorphQ: Metamorphic Testing of the Qiskit Quantum Computing Platform. 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), 2413–2424. [10.1109/ICSE48619.2023.00202](https://doi.org/10.1109/ICSE48619.2023.00202) [arXiv 2206.01111v2](https://arxiv.org/abs/2206.01111v2)

## Scientific examples

The three scientific notebooks develop problems from the first two papers below and from Wu et al., arXiv:2605.12066v1, listed under [Quantum Hamiltonian Descent](#quantum-hamiltonian-descent-qhd), and [Examples](examples.md#scientific-notebooks) states the question each one answers.

 - Schleich, P., Kharazi, T., Li, X., Liu, J.-P., Aspuru-Guzik, A., & Wiebe, N. (2025). Arbitrary Boundary Conditions and Constraints in Quantum Algorithms for Differential Equations via Penalty Projections. [arXiv 2506.21751v1](https://arxiv.org/abs/2506.21751v1)
    - Section III.B.1, Eqs. (130)–(134): the two-dimensional heat equation with Dirichlet and Neumann boundaries. Fig. 5 shows its case of vanishing Dirichlet boundaries.
    - Section II.B, Problem 7, Eq. (14): the imaginary penalty −iλP_c.
    - Section IV, Eq. (140): restates the constrained equation that `examples/lchs_scientific.ipynb` solves on a four-point rod.
    - The arXiv HTML version numbers Sections II.B and III.B.1 as II.2 and III.2.1.
 - Li, X., Yin, X., Wiebe, N., Chun, J., Schenter, G. K., Cheung, M. S., & Mülmenstädt, J. (2025). Potential quantum advantage for simulation of fluid dynamics. Physical Review Research, 7, 013036. [DOI](https://doi.org/10.1103/PhysRevResearch.7.013036), [arXiv 2303.16550v3](https://arxiv.org/abs/2303.16550v3)
    - Eq. (3) of the published article: the forward-Euler history-state system with padding steps.
    - The Appendix with Fig. 2 of the published article: the single-node D1Q3 collision at relaxation time τ = 1 and time step τ/10 that `examples/qls_scientific.ipynb` reduces to a 15-dimensional system.
    - arXiv:2303.16550v3 numbers them Eq. (2), Appendix A and Fig. 3.
 - Krovi, H. (2023). Improved quantum algorithms for linear and nonlinear differential equations. Quantum, 7, 913. [DOI](https://doi.org/10.22331/q-2023-02-02-913), [arXiv 2202.01054v4](https://arxiv.org/abs/2202.01054v4)
    - Li et al. use its linear-ODE algorithm in their complexity analysis, which `examples/qls_scientific.ipynb` does not reproduce.
    - The Quantum article's journal page names arXiv v4 as its published version.
