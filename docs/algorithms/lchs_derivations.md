# LCHS derivations {#lchs-derivations}

Use this page to reconstruct what a reported LCHS radius, normalization or component error means and which premises it needs. It derives the relations behind [sampled branch pairs](lchs.md#sampled-branch-pairs), the node and joint block encodings of `qsp_block_encoding`, the constant-source preparation, `lcu_prep="analytic"` and the `apel_berntson` kernel. The [LCHS guide](lchs.md) explains how to request these features.

Three kinds of statement appear below, and they are not interchangeable:

- An **exact-arithmetic relation** holds for the real numbers it names, under the premises written beside it.
- A **numerical estimate** evaluates such a formula in binary64 on the values a Plan holds. It has no arithmetic enclosure, and a number computed this way can fall on the wrong side of the exact value.
- An **unknown component** is one the library reports as unavailable. It is not zero, and no relation on this page supplies it.

None of these relations bounds the total error of a physical output. Gate synthesis, execution, sampling beyond the stated population and floating-point error stay separate unless a section says otherwise.

## Sampled branch pairs {#sampled-branch-pairs}

### Target vector, weights and scores {#sampled-scores}

The sampled route targets the vector

```math
v=\int a(x)\,\chi_x\,d\mu(x),\qquad \lVert\chi_x\rVert=1,
```

where $x$ collects an application and a kernel node. An application is the initial state at elapsed time $T$, a Gauss–Legendre source node at elapsed time $t_a$, or a time drawn from the continuous source interval. With the signed scalar shift $s$ of [the constant-source construction](#source-preparation), the centered parts $L_c=L-c_LI$ and $H_c=H-c_HI$, and the kernel coefficient $c(k)$,

```math
a(x)=m\,c(k)\,e^{i\phi(t,k)},\quad
\phi(t,k)=-t\bigl(k(c_L+s)+c_H\bigr),\quad
\chi_x=e^{-it(kL_c+H_c)}\psi_{\mathrm{kind}}.
```

The mass $m$ is $\nu_0e^{sT}$ for the initial state, $\omega_a\nu_be^{st_a}$ for a discrete source node and the density $\nu_be^{st}$ on a continuous source interval, with $\nu_0,\nu_b$ the input norms and $\omega_a$ the Duhamel weight (An, Childs and Lin, arXiv:2312.03916v2, Eqs. (2), (4), (7)). The physical relative phase of the initial and source states stays in $\psi_{\mathrm{kind}}$.

A proposal $q$ that is positive wherever $a$ is nonzero gives the weight $w=a/q$. Two bounds on it answer different questions: $B\ge\mathbb E_q|w|$ bounds the mean weight and $\lVert v\rVert$, and $W\ge\sup_q|w|$ bounds the score of one shot. Finite laws (applications, kernel nodes, observable terms) are integer tables with exact probabilities $p_j=n_j/2^{52}$ and $n_j\ge1$ on every nonzero entry. The identity $\sum_jp_j(a_j/p_j)\chi_j=\sum_ja_j\chi_j$ needs only division by those actual probabilities. Closeness to the law proportional to $|a_j|$ therefore affects $W$ and the second moments, not the mean. Rounding of the saved weights is a separate discrepancy. A continuous draw, of a source time or a kernel node, uses the midpoint of a uniform 52-bit integer, and its weight uses the ideal density.

For an observable $O=\sum_lo_lP_l$ with real $o_l$ and Pauli strings $P_l$, a row draws $x$, $y$ and a term $l$ with probability $p_l>0$ on every nonzero $o_l$. Each shot prepares

```math
\Psi=\frac{|0\rangle\chi_x+e^{i\theta}|1\rangle\chi_y}{\sqrt2},\qquad \theta=\arg w_y-\arg w_x,
```

and measures the extra qubit in the $X$ basis, with outcome $s_a=\pm1$, and the system in the basis of $P_l$, with parity $s_P=\pm1$. Expanding $\langle\Psi|X\otimes P_l|\Psi\rangle$ gives

```math
\mathbb E[s_as_P\mid x,y,l]=\operatorname{Re}\bigl(e^{i\theta}\langle\chi_x|P_l|\chi_y\rangle\bigr),
```

and $P_l=I$ gives the same relation for $s_a$. The single-shot scores are

```math
Z_Q=\frac{o_l}{p_l}|w_xw_y|\,s_as_P,\qquad Z_N=|w_xw_y|\,s_a.
```

Because $|w_xw_y|e^{i\theta}=\overline{w_x}w_y$, summing over $l$ and integrating over independent $x$ and $y$ gives $\mathbb E Z_Q=v^\dagger Ov$ and $\mathbb E Z_N=\lVert v\rVert^2$. The cross terms with $x\ne y$ are part of these means (An, Liu and Lin, arXiv:2303.01029v2, Supplement IV, and Chakraborty, arXiv:2302.13555v4, Sec. 3.2). The integer laws and the binary scale below are NWQLib's specialization of those constructions.

The scores obey $|Z_Q|\le H_Q=H_OW^2$ and $|Z_N|\le H_N=W^2$, with $H_O=\max_l|o_l|/p_l$. This $H_O$ is different from $C_O=\sum_l|o_l|\ge\lVert O\rVert$, which converts a vector or operator error into an error of $Q$ in [the bias relations](#sampled-bias). The two agree only when $p_l$ equals $|o_l|/C_O$ exactly, which an integer table generally does not. A finite recipe multiplies both scores by the signed multiplier of its member, and the QSP route by its recovery squared, so their ranges use the largest such factor over the whole member table or bank, not over the drawn rows.

Scores in a row table are in a binary scale: each is the physical score divided by $g^2$, with $g$ the field `LCHSPairMoments.scale`. Physical first moments and radii multiply by $g^2$, and physical second moments by $g^4$. The fields `inner_bias` and `model_bias` of `LCHSPairBias` are in this scale too.

### Rows, variance and confidence {#sampled-confidence}

Write $K$ for the number of independent rows and $m$ for `SampledPairs.repeats`, the conditionally independent shots of each row. The [accuracy request](lchs.md#accuracy-requests) writes this row count as $B$. A row holds every random choice that stays fixed during its shots, including the observable term and, on the QSP route, the polynomial members. With $V_b=\operatorname{Var}(\mathbb E[Z\mid\text{row}])$ and $V_w=\mathbb E\operatorname{Var}(Z\mid\text{row})$, the law of total variance gives

```math
\operatorname{Var}(\widehat Z)=\frac{V_b}{K}+\frac{V_w}{Km}.
```

At a fixed total $N=Km$ of shots, the ratio of this variance to the case $m=1$ is $1+(m-1)\rho$, with $\rho=V_b/(V_b+V_w)$ when the denominator is positive. Repeating a row therefore adds little information when the row-to-row variance dominates. This is an identity for the sampling model. A Result does not report estimates of $V_b$ or $V_w$.

Each row mean lies in $[-H,H]$, so Hoeffding's inequality for bounded independent variables (doi:10.1080/01621459.1963.10500830, Theorem 2) gives the sampling radius

```math
r=H\sqrt{\frac{2\log(2/\delta)}{K}},\qquad\text{equivalently}\qquad K=\max\Bigl(1,\Bigl\lceil 2(H/\epsilon)^2\log(2/\delta)\Bigr\rceil\Bigr)\ \text{ for }r\le\epsilon.
```

This radius depends on $K$ and not on $m$. A simultaneous claim about $Q$ and $N$ gives $\delta/2$ to each and uses the union bound, which needs no independence between them. The two scores come from the same shots, so their covariance is not zero in a variance analysis. The statement requires every planned row to be complete with the same $m$, so a partial Run has raw sums and no confidence statement.

A known deterministic bias bound $b$ gives the covering radius $r+b$. With an unknown bias the covering radius is `None`, and $r$ then describes the sampling fluctuation alone.

For a `NormalizedExpectation`, suppose $|Q-\widehat Q|\le r_Q$ and $|N-\widehat N|\le r_N$ hold together and $\widehat N-r_N>0$. On this rectangle $Q/N$ is affine in $Q$ at fixed $N$ and monotone in $N$ at fixed $Q$, in a direction set by the sign of $Q$, so its extremes lie at the four corners. The reported interval compares those corners. When $\widehat N-r_N$ is not positive the ratio has no interval, and the raw estimates and radii remain.

### Bias of the branch evolution and of the model {#sampled-bias}

**Plain product formula.** Suppose the unitary approximate branches satisfy $\lVert\widehat U_x-U_x\rVert\le\delta_x$, and likewise for $y$. Telescoping gives

```math
\bigl|\langle\psi_x|\widehat U_x^\dagger O\widehat U_y-U_x^\dagger OU_y|\psi_y\rangle\bigr|\le\lVert O\rVert(\delta_x+\delta_y),
```

with no $\delta_x\delta_y$ term because the approximate branches are unitary too. A row runs $\max(r_{b(x)},r_{b(y)})$ steps on both branches, so the step count depends on both draws. Each branch nevertheless keeps the bound $\delta_x\le\overline W(k_x)t_x^{p+1}/r_{b(x)}^p$, whichever partner it has, from Childs et al., doi:10.1103/PhysRevX.11.011020, Propositions 9–10. Here $\overline W$ is the Pauli-triangle sum of the node's coefficients and $r_b$ the step count of the bucket holding $|k|$. Multiplying this pointwise bound by $|w_xw_y|$ and integrating over the independent draws factorizes it. With $D\ge\mathbb E[|w|\delta]$, the bias of $N$ is at most $2BD$ and that of $Q$ at most $2C_OBD$. The field `LCHSPairBias.inner_relation` names how $D$ is formed:

| `inner_relation` | Applies to | $D$ |
| --- | --- | --- |
| `"finite_sum"` | Finite applications and discrete nodes, where the probabilities cancel in $\mathbb E[\lvert w\rvert\delta]$. $M_i$ and $t_i$ are the mass and elapsed time of application $i$, and $c_j$ the kernel coefficients | $\bigl(\sum_iM_it_i^{p+1}\bigr)\bigl(\sum_j\lvert c_j\rvert\overline W_j/r_{b(j)}^p\bigr)$ |
| `"bucket_maximum"` | A continuous source time or kernel node, with $T_e$ the largest elapsed time | $B\,T_e^{p+1}\max_b\overline W_b/r_b^p$, with $\overline W_b$ a bound over the whole bucket, so no bucket probability enters |

**Model bias.** If $\Delta_v\ge\lVert u-v\rVert$ for the physical solution $u$ and $\lVert v\rVert\le B$, expanding the quadratic form gives

```math
\bigl|\lVert u\rVert^2-\lVert v\rVert^2\bigr|\le 2B\Delta_v+\Delta_v^2,
```

and the bound for $Q$ multiplies this by $C_O$. $\Delta_v$ counts once each the kernel and k-quadrature bounds times the total application mass (or the kernel and endpoint-guard bounds of a continuous kernel), the Gauss–Legendre bound of a discrete source rule, and the coupling of finite random bits. That coupling follows from the midpoint draw: for an increasing quantile function $f$, integrating over each cell of width $2^{-52}$ gives $\mathbb E|f(u)-f(u_{\mathrm{mid}})|\le2^{-53}(f(1)-f(0))$. It does not enclose the floating-point evaluation of the inverse distribution or of the phases.

### Finite recipes and the QSP bank {#sampled-recipes}

Each route supplies its own bias relation for the pair observable. With $B$ and $C_O$ as above:

| Route | Relation for the bias of $N$ (multiply by $C_O$ for $Q$) | Premises and limits |
| --- | --- | --- |
| Richardson, `SampledPairs(product_formula=Richardson(...))` | $B^2(B_{\mathrm{ext}}+E_{\mathrm{eff}})$, with $B_{\mathrm{ext}}\le4\Lambda\max(1,A)^2(A/s)^4$, $\Lambda=5/3$ | Levels $(s,2s)$ with weights $(-1/3,4/3)$ cancel the shared Strang $h^2$ term. Needs $s>2A$, where $A=(2\mathcal L-1)\,4L_*$ for $\mathcal L$ distinct nonidentity Pauli labels in the union of the labels of $H_c$ and $L_c$ (a label in both counts once) and $L_*=T_e(\lambda_H+K_*\lambda_L)$, with $\lambda_H,\lambda_L$ the coefficient one-norms of $H_c$ and $L_c$, $T_e$ the largest elapsed time and $K_*$ a bound on $|k|$ over the entire kernel support. Below that, a fixed step count leaves the bias unknown. $E_{\mathrm{eff}}=\sum_l\lvert\pi_l\widehat\gamma_l-w_l\rvert$ is the exact discrepancy of the saved multipliers. With at most one distinct nonidentity Pauli label, or $L_*=0$, the evolution is exact, so $B_{\mathrm{ext}}=0$ without $s>2A$, and $E_{\mathrm{eff}}$ remains. One level is shared by both branches, so this extrapolates the joint pair observable (Watson and Watkins, arXiv:2408.14385v2, Eq. (84)) |
| Paired Taylor-series compensation, `PTSC(...)` | $B^2(2\delta+\delta^2)$ | $\delta$ is a uniform recovered-branch bound, the Taylor tail of Zeng, Sun, Jiang and Zhao, arXiv:2212.04566v2, Eqs. (79) and (84), plus NWQLib's perturbation from the finite integer laws. Independent branch corrections give a mean-vector error at most $B\delta$. Only the Taylor and count components are covered |
| Two-member multi-product formula, `"mpf"` | Unknown for two or more distinct nonidentity Pauli labels in the union of the labels of $H_c$ and $L_c$ (a label in both counts once) | $\alpha_1=(3+\sqrt3)/9$, $\alpha_2=(1+\sqrt3)/3$ and $c=(3/2,-1/2)$ obey $\sum c=1$, $\sum c\alpha=1/3$, $\sum c\alpha^2=0$, which cancels the leading $h^2$ term of the pair observable (arXiv:2607.12225v1, Sec. VIII, Eqs. (17)–(24)). Probabilities $(3/4,1/4)$ give multipliers $(2,-2)$ and a second-moment factor 4. With two or more such labels the relation supplies no finite-step remainder, so only raw estimates and sampling radii are reported. With at most one, the evolution remainder is zero |
| QSP phase bank, `SampledPairs.qsp_bank` | $B^2(2\beta+\beta^2)$ | $\beta$ is a support-wide bound on the error of the mean polynomial, derived below |

For the stochastic bank, write the Jacobi–Anger polynomial of a bucket in the paired shells $\mathcal B_n=(z^n+z^{-n})/2$ with coefficients $c_n$. The bank keeps the shells through a cutoff $n_c$ and adds one drawn shell: member $n$ is $R_n=p_{n_c}+\gamma_n\mathcal B_n$, with $\gamma_n=W_cc_n/|c_n|$ and $W_c=\sum_{n_c<n\le d}|c_n|$. With the actual integer probabilities $\pi_n$, the mean member differs from the deterministic polynomial by at most $e_{\mathrm{law}}=\sum_n|\pi_n\gamma_n-c_n|$. Its recovered error is at most $\delta+e_{\mathrm{law}}+\zeta\sum_n\pi_n\kappa_n$ plus the rounding of the normalized coefficients, where $\delta$ is the expansion tail, $\zeta$ the common recovery and $\kappa_n$ the phase residual of member $n$. The two branches draw their members independently, so conditional on the outer pair $(x,y)$ and its shared bucket,

```math
\mathbb E_{\mathrm{members}}\bigl[P_x^\dagger OP_y\mid x,y,\text{bucket}\bigr]=\overline P_x^\dagger O\,\overline P_y,
```

and mean amplitude errors $\beta_x,\beta_y$ give the bound $\lVert O\rVert(\beta_x+\beta_y+\beta_x\beta_y)$. A uniform $\beta$ then gives the table's entry after integrating over the outer draws. The bucket depends on both branches, so this relation is conditional, and the argument does not identify the unconditional estimator with the norm of one mean vector. The bound $\beta$ also adds the child-encoding term $T_e(K_*\delta_L+\delta_H)$ and a branch-rounding term once. A deterministic bank is the one-member case.

Shots on the QSP route score zero when a work qubit does not return to zero, and the row mean divides by all returned shots. Dividing only by the successful shots would be a different estimator. The score and radius arithmetic and the coefficient and phase errors have no enclosure, so the reported radii are numerical estimates of these relations.

## Node encodings {#node-encodings}

Uniform nodes $k_j=hj$, $-J\le j\le J$, $J\ge1$, use a register of $r=\lceil\log_2(2J+1)\rceil$ bits. Put $p=r-1$. Two address maps are available: two's complement stores $j\bmod2^r$, and signed magnitude stores $|j|+2^p[j<0]$, whose negative-zero word $2^p$ is padding. Labels without a node have zero amplitude in the coefficient preparation, so the encoded diagonal may take any bounded value there. This choice is its completion.

The physical radius is $R=hJ$, the kernel cutoff. The encoding normalization is $\alpha=hN$, where $N$ bounds $|j|$ over the completion as well, so $\alpha$ and $R$ can differ. Substituting $b_l=(1-Z_l)/2$ into the signed integer of each map gives the two Pauli forms:

```math
j=-\tfrac12I+2^{r-2}Z_{r-1}-\sum_{l\lt r-1}2^{l-1}Z_l\quad\text{(two's complement)},
```

```math
j=\tfrac{2^p-1}{2}Z_s-\sum_{l\lt p}2^{l-1}Z_sZ_l\quad\text{(signed magnitude)}.
```

Their coefficient one-norms are the values of $N$ in the table.

| `UniformNodeRecipe` | Block on a physical label | $N$ | Completion |
| --- | --- | --- | --- |
| `"rotation_table"` | $\cos(\theta_\nu/2)$ for the saved angle $\theta_\nu=\mathrm{fl}(2\arccos\mathrm{fl}(hj/\alpha))$ | $J$ | Value 0 on padding |
| `"comparator"` | $j/J$ | $J$ | Clipped value $(-1)^{\mathrm{sign}}\min(\lvert j\rvert,J)/J$ |
| `"twos_complement_pauli"` | $j/2^p$ | $2^p$ | Integer value of the word, down to $-2^{p}$ |
| `"signed_magnitude_pauli"` | $j/(2^p-1)$ | $2^p-1$ | Integer value of the word, up to magnitude $2^p-1$, with value 0 on the negative-zero word |

The comparator draws a uniform integer $l\in[0,J)$ and flags $[l<|j|]$, so projecting its signal qubit on zero leaves the probability $\min(|j|,J)/J$ with the sign of $j$. This is the probability mechanism of Low and Somma, arXiv:2508.19238v2, Lemma 18. With $B$ the uniform preparation followed by the comparison, the library conjugates the involution $K=Z_{\mathrm{sign}}C_{f=0}X_z$ as $B^\dagger KB$, which makes the whole unitary a Hermitian involution.

Three error statements apply to these blocks. For ideal comparator or Pauli gates the block is exactly $j/N$, so with the saved $\alpha=\mathrm{fl}(hN)$ the physical diagonal $hj$ is missed by $|j|\,|hN-\alpha|/N\le|hN-\alpha|$. The library computes this with exact rationals and rounds it up. That number says nothing about the gates that implement the ideal block. An angle table has a different statement: its unitary is an involution for every real angle, while its numerical error is the residual $\max_\nu|v_\nu-\lambda\cos(\theta_\nu/2)|$, evaluated in binary64 against the saved table.

## Joint generators {#joint-generators}

The separate children act on the same system, while node and time factors act on their own registers. With $L_0,H_0$ the centered parts, $D_K$ the node diagonal of normalization $\lambda_K$ and $D_Q$ the time diagonal with entries $q_a=t_a/T_e$ and normalization $\lambda_Q$,

```math
S=D_Q\otimes(D_K\otimes L_0+I\otimes H_0),\qquad
\alpha_S=\lambda_Q(\lambda_K\alpha_L+\alpha_H),
```

so $e^{-iT_eS}$ applies $e^{-it_a(k_jL_0+H_0)}$ on address slot $(a,j)$ (adapted from Pocrnic et al., arXiv:2506.20760v2, Sec. IV). A real rotation on one qubit with $\cos^2(\theta/2)=\lambda_K\alpha_L/(\lambda_K\alpha_L+\alpha_H)$ combines the two branches as a two-term LCU. Let $\delta_L,\delta_H,\delta_K,\delta_Q$ be the residuals of the children and diagonals. Use $B_L=\alpha_L+\delta_L$ and $B_H=\alpha_H+\delta_H$, which bound the target norms and also cover the child encoding normalizations. Adding and subtracting each factor gives

```math
\delta_{\mathrm{hom}}\le\lambda_K\delta_L+\delta_KB_L+\delta_H+\delta_{\mathrm{combine}},\qquad
\delta_S\le\lambda_Q\delta_{\mathrm{hom}}+\delta_Q(\lambda_KB_L+B_H).
```

The second inequality uses $\lVert D_Q\rVert\le\lambda_Q$ and the norm of the implemented homogeneous block, at most $\lambda_K\alpha_L+\alpha_H\le\lambda_KB_L+B_H$. The combine term $\delta_{\mathrm{combine}}\le2\alpha_{\mathrm{hom}}|\cos^2(\theta/2)-w_L/\alpha_{\mathrm{hom}}|$, with $w_L=\lambda_K\alpha_L$ and $\alpha_{\mathrm{hom}}=\lambda_K\alpha_L+\alpha_H$, accounts for the saved angle. Residuals of saved angles and of the scalar evaluation stay numerical estimates. If a child's bound is unknown, the combined bound is unknown.

**One encoding of a general $A$.** For an encoding $U_A$ of $A_0=L_0+iH_0$ and a complex coefficient table $c=d_L-id_H$ with $|c|\le r$, put $B=U_C\otimes U_A$ and

```math
U_S=H_h\begin{pmatrix}0&B\\B^\dagger&0\end{pmatrix}H_h.
```

Its good block is the Hermitian part of $cA_0$ over $r\alpha_A$, which is $(d_LL_0+d_HH_0)/(r\alpha_A)$, and $U_S$ is a full involution for every unitary $B$. Taking a Hermitian part does not increase a norm, so the product bound gives $\delta_S\le r\delta_A+\delta_C(\alpha_A+\delta_A)$. The time factor is part of the coefficient table here. An outside control can act on the helper $X$ alone, because $B$ and $B^\dagger$ cancel on the inactive branch.

Three facts stay separate in both constructions: the target is Hermitian, the encoded block is Hermitian, and the full unitary is an involution. For the same principal good block, a full unitary involution implies a Hermitian encoded block, but the converse need not hold. With a nonzero encoding error, a Hermitian target alone does not establish that the encoded block is Hermitian. `AccessSummary` and `JointEncodingPlan` report them in the fields `target_operator_is_hermitian`, `encoded_operator_is_hermitian` and `full_unitary_involution`.

**Imaginary-time contraction.** For $0\le q_a\le1$ and $\alpha\ge\alpha_L+\delta_L\ge\lVert L_0\rVert$, the operator $X_a=(1-q_a)I-q_aL_0/\alpha$ is a Hermitian contraction. With $L=cI+L_0$ and $\beta=T_e\alpha$, direct substitution gives

```math
e^{-t_aL}=e^{-t_a(c-\alpha)}\,e^{-\beta(1-X_a)}.
```

The coefficient error is at most $\max_aq_a(\delta_L/\alpha+|\widehat c_s-c_s|)+2\max_a|\widehat q_a-q_a|$, with $c_s=\alpha_L/\alpha$.

**Products.** Two encodings with independent ancilla registers on one system give the product block $B_1B_2$ in the stated order, with error $e_1N_2+(N_1+e_1)e_2$ for target norm bounds $N_1,N_2$. No commutation is needed for this bound.

## Constant-source preparation {#source-preparation}

Let $\ell\le\lambda_{\min}(L)$ be a known lower bound and $s=-\ell$, so $L_s=L+sI$ is PSD. The identity commutes with $A$, so $e^{-At}=e^{st}e^{-(A+sI)t}$. A positive $\ell$ extracts known physical decay, and a negative $\ell$ supplies the shift that the kernel needs. A negative $\ell$ from structural bounds does not show that $L$ has a negative eigenvalue. The relation needs a valid $\ell$, and a kernel route also needs an upper or norm bound for its processed norm, $u-\ell$ or $\|L\|+|\ell|$ ([stated lower bound and sampling](#stated-lower-bound)).

The finite solution over applications $a$ (the initial state at $t=T$ and the source nodes) and kernel nodes $j$ is

```math
u_{\mathrm{fin}}=\sum_{a,j}\omega_a\nu_ae^{st_a}c_j\,e^{-it_a(k_jL_s+H)}|\psi_a\rangle,
```

with nonnegative weights $\omega_a$ and input norms $\nu_a$. Define $m_a=\omega_a\nu_ae^{st_a}$. Separating the identity parts of $L_s=L_0+(c_L+s)I$ and $H=H_0+c_HI$ gives each slot the magnitude $m_a|c_j|$ and the phase

```math
\phi_{aj}=\arg c_j-t_a\bigl[k_j(c_L+s)+c_H\bigr].
```

The magnitudes factorize into an application state and a kernel state,

```math
|p\rangle=\Bigl(\sum_a\sqrt{m_a/B_{\mathrm{app}}}\,|a\rangle\Bigr)\otimes\Bigl(\sum_j\sqrt{|c_j|/\beta_k}\,|j\rangle\Bigr),\qquad
\gamma=B_{\mathrm{app}}\beta_k,
```

with $B_{\mathrm{app}}=\sum_am_a$ and $\beta_k=\sum_j|c_j|$. The circuit prepares $|p\rangle$, applies the state preparation $R$ selected by the input kind, the evolution, the phase diagonal and the inverse of the coefficient preparation. Expanding the overlaps of the two coefficient states shows that its success block maps $|0\rangle$ to $u_{\mathrm{fin}}/(\zeta\gamma)$, with $\zeta$ the recovery of the evolution polynomial. $R$ is applied once and never undone, and the physical phases of the initial and source states stay in their preparations.

The same address map must serve the coefficient preparation, the node query and the slot phase. Padding has zero ideal amplitude and still counts in every query's normalization and full-space action. Work qubits of an analytic preparation stay through both preparation calls and belong to the success projector, so success is defined on that enlarged space.

For uniform nodes the magnitudes need $O(A+M)$ table entries for $A$ applications and $M$ nodes, and the coupled phase needs $O(A\log M)$ entries through the bit expansion of $j$. A nonuniform grid needs one $O(AM)$ phase table. These are sizes of the construction's tables, not of its run time, its process memory or an arbitrarily lowered circuit.

### Stated lower bound and sampling {#stated-lower-bound}

For combined real Pauli coefficients, each normalized Rayleigh quotient of $L$ is at least $\ell_{\rm struct}=c_I-\sum_{P\ne I}|c_P|$. The maximum of two valid lower bounds is valid, so $\ell=\max(\ell_{\rm struct},\ell_u)$ is a lower bound when the stated $\ell_u$ is correct. An improvement remains conditional on that statement.

The eigenvalues of a Hermitian $L$ of dimension $D$ have the mean $c=\operatorname{tr}(L)/D$. Hence $\lambda_{\min}(L)\le c$, and equality makes every eigenvalue equal to $c$, since each is at least the mean and their sum is $Dc$, so $L=cI$. Every nonidentity Pauli word has trace zero, so for coalesced coefficients $c=c_I$, and a nonzero $c_P$ with $P\ne I$ shows that $L$ is not a multiple of the identity, because distinct words are linearly independent. The same holds for a periodic band with coalesced offsets, whose shifts $S^b$ with $b\ne0$ modulo $D$ have trace zero, with the offset-0 coefficient in place of $c_I$. A stated value above $c$, or equal to $c$ when $L$ is not a multiple of the identity, therefore contradicts the premise, and planning refuses it by comparing stored coefficients exactly. Any upper bound of the spectrum is at least $c$, and above it for such an $L$, so this also covers stated values at or above an upper bound. False values strictly below $c$ stay undetected, so passing this comparison establishes neither the stated bound nor the error bounds conditional on it. A supplied encoding stores no trace, so this comparison does not apply to it.

For a time-independent $A$ and an integrable source, set $w(t)=e^{\ell t}u(t)$. Then $w'=-(A-\ell I)w+e^{\ell t}b(t)$, and multiplying its solution back by $e^{-\ell t}$ gives

```math
u(t)=e^{-\ell t}e^{-(A-\ell I)t}u_0+\int_0^te^{-\ell(t-v)}e^{-(A-\ell I)(t-v)}b(v)\,dv.
```

Pulling one $e^{-\ell t}$ out of the integral while leaving the source unchanged would drop its factor $e^{\ell v}$. The constant source of `LinearDynamics` and its discrete nodes are special cases, in which application $a$ has the mass $m_a$ above with its own duration $t_a$.

Let $\Gamma>0$ be fixed before $N$ independent stationary shots, with $\|u_{\rm approx}\|\le\Gamma$ and $q=\|u_{\rm approx}\|^2$. Each shot gives a Bernoulli $X$ with $p=q/\Gamma^2$, and the estimate is $\widehat q=\Gamma^2\overline X$. Hoeffding's centered bound $\mathbb E\,e^{t(X-p)}\le e^{t^2/8}$, independence, Markov's inequality and minimization over $t\gt0$ give each tail $e^{-2Nr^2}$ for $\overline X-p$ at distance $r$ (doi:10.1080/01621459.1963.10500830). With $r=\epsilon/\Gamma^2$ and both tails,

```math
\Pr(|\widehat q-q|\gt\epsilon)\le2\exp\Bigl(-\frac{2N\epsilon^2}{\Gamma^4}\Bigr),\qquad
N\ge\max\Bigl(1,\Bigl\lceil\frac{\Gamma^4}{2\epsilon^2}\log\frac2\delta\Bigr\rceil\Bigr),
```

for a positive integer $N$. This controls sampling about $q$, not the error against the exact solution. The [LCHS guide](lchs.md#declared-lower-bound) gives the conditions under which it compares routes and shifts.

## Analytic Low–Somma preparation {#analytic-preparation}

With `lcu_prep="analytic"` the kernel factor of $|p\rangle$ is prepared on the trapezoid grid of `low_somma_f2`. Equal weights and the constant kernel prefactor cancel on normalizing, which leaves

```math
f(k)=\frac{e^{-k^2/(8\gamma_{\mathrm{LS}}^2)}}{\sqrt{1+k^2}},\qquad
\psi_f=\frac{\sum_{j=-J}^Jf(hj)|\ell(j)\rangle}{\sqrt{Mp_f}},\qquad
p_f=\frac1M\sum_jf(hj)^2,
```

with $M=2J+1$, $\ell(j)$ the address map of [the node encoding](#node-encodings) and $\gamma_{\mathrm{LS}}$ the kernel width. This is the target in Low and Somma, arXiv:2508.19238v2, Lemma 19, following Eq. (133). The finite parameters below replace that lemma's asymptotic cost, and its ideal unit success is not a claim about this construction.

### Polynomial and aliasing components {#analytic-polynomial}

Let $\lambda>0$ be the normalization of the selected node encoding, completion included, and $F(x)=f(\lambda x)$. Take the Bernstein ellipse with imaginary semiaxis $1/(2\lambda)$, whose parameter is $\rho=\frac1{2\lambda}+\sqrt{1+\frac1{4\lambda^2}}$. On it, writing $\lambda z=x+iy$ with $|y|\le1/2$ gives $\operatorname{Re}(1+(\lambda z)^2)\ge1-y^2\ge3/4$, so the square root is analytic and

```math
|F(z)|\le M_\rho=\frac2{\sqrt3}\,e^{1/(32\gamma_{\mathrm{LS}}^2)}.
```

The Cauchy estimate applied to $F((w+w^{-1})/2)$ on $|w|=\rho$, pairing its positive and negative Laurent coefficients, gives Chebyshev coefficients $|a_m|\le2M_\rho\rho^{-m}$ for $m>0$ (the bound of Trefethen, *Approximation Theory and Approximation Practice*, extended ed., doi:10.1137/1.9781611975949, Theorem 8.1). Summing the geometric tail beyond degree $n$ gives

```math
\epsilon_{\mathrm{tail}}\le\frac{2M_\rho\rho^{-n}}{\rho-1}.
```

The coefficients are computed from $2N$ equispaced angles. Discrete cosine orthogonality gives, for $1\le m\le n<N$,

```math
c_m=a_m+\sum_{l\ge1}\bigl(a_{2Nl+m}+a_{2Nl-m}\bigr),\qquad\text{and separately}\qquad c_0=a_0+\sum_{l\ge1}a_{2Nl}.
```

Bounding each alias by the coefficient estimate and summing over $m=0,\dots,n$ gives the conservative

```math
\epsilon_{\mathrm{alias}}\le\frac{2M_\rho(1+2n)\rho^{-(2N-n)}}{1-\rho^{-2N}}.
```

The two terms share 0.1 times `approx_tol`. Write $p(x)=\sum_{m=0}^n c_mT_m(x)$ for the polynomial formed from the sampled coefficients. With $\Delta=\epsilon_{\mathrm{tail}}+\epsilon_{\mathrm{alias}}$, a margin $0<\eta<1$ and $\mu=(1+\Delta)/(1-\eta)$, the target $q=p/\mu$ has coefficients $q_m=c_m/\mu$ and obeys $\lVert q\rVert_\infty\le1-\eta$, because $|F|\le1$ on $[-1,1]$. On the ideal physical nodes, $x_j=hj/\lambda$. This is an ideal coefficient statement. The binary64 samples, transform and bound evaluation have no enclosure.

### State and amplification relations {#analytic-state}

If both scalar arguments lie in $[-1,1]$, then $T_m'=mU_{m-1}$ and $|U_{m-1}|\le m$ give the Lipschitz constant $L_q=\sum_{m\ge1}m^2|q_m|$, so a node perturbation at most $e_x$ changes each amplitude factor by at most $L_qe_x$. The uniform input is normalized, so no factor $\sqrt M$ appears for the vector. Let $e$ bound the whole discrepancy of the good vector from the target $q(x_j)/\sqrt M$. It contains the phase residual $e_\phi$, the node term $L_qe_x$, an emitter term $e_{\mathrm{emit}}$ and a uniform-support term $e_u$. Write $p_q=M^{-1}\sum_jq(x_j)^2$. The reverse triangle inequality gives $|\sqrt{p_{\mathrm{actual}}}-\sqrt{p_q}|\le e$. Given enclosures $p_q\in[l_q,u_q]$ and $l_f\le p_f$, it follows that

```math
p_-=\bigl(\max(0,\sqrt{l_q}-e)\bigr)^2,\qquad p_+=\min\bigl(1,(\sqrt{u_q}+e)^2\bigr),
```

```math
\delta_{\mathrm{direction}}\le\frac{2\mu E}{\sqrt{l_f}},\qquad E=\frac\Delta\mu+e,\qquad \mu E<\sqrt{l_f}.
```

The last condition keeps the good vector nonzero. The direction inequality adds and subtracts $a/\lVert b\rVert$ for the good vector $a$ and $b=f/(\mu\sqrt M)$, then uses the reverse triangle inequality. It is a phase-sensitive vector distance, not a fidelity after phase alignment.

For a known positive target probability $p_0$, define $K_{\mathrm{AA}}=\max(0,\lceil\pi/(4\arcsin\sqrt{p_0})-1/2\rceil)$, $\theta_*=\pi/(4K_{\mathrm{AA}}+2)$ and the dilution ratio $r_d=\sin^2\theta_*/p_0$. Dilution followed by $(A_dR_0A_d^\dagger R_g)^{K_{\mathrm{AA}}}A_d$, with $R_g=I-2\Pi_{\mathrm{good}}$ and $R_0=2|0\rangle\langle0|-I$ on every input of $A_d$ including the address, rotates the good/bad plane to the angle $(2K_{\mathrm{AA}}+1)\theta$. It therefore reaches the good state when the diluted probability equals $\sin^2\theta_*$. This is the known-probability method of Brassard, Høyer, Mosca and Tapp, arXiv:quant-ph/0005055v1, Sec. 2.1. For the realized ratio $r_d'=\cos^2(\mathrm{angle}/2)$, monotonicity in $p$ gives

```math
\Delta_\theta=\max_{p\in\{p_-,p_+\}}\bigl|\arcsin\sqrt{p\,r_d'}-\theta_*\bigr|,\qquad
\delta_k\le\delta_{\mathrm{direction}}+(2K_{\mathrm{AA}}+1)\Delta_\theta+e_{\mathrm{exec}}.
```

The distance from the rotated state to the good direction is at most the angle error, so $\delta_k$ covers the whole clean state, garbage included. The block discrepancy enters through the probability and direction terms. Being a block error and not a full-unitary error, it is not multiplied by $2K_{\mathrm{AA}}+1$ again. Separately bounded full-gate errors belong to $e_{\mathrm{exec}}$.

### What the Plan reports {#analytic-numerical-scope}

The relations above are conditional, and the current computation does not supply all their terms:

- The probabilities are computed on `k = fl(h j)` and `x = fl(k/lambda)`, so $p_q$ and $p_f$ are binary64 estimates with $l_q=u_q=p_q$ and $l_f=p_f$. They have no interval, and the endpoints $p_\pm$ are not enclosures.
- The node term $e_x=|hN-\lambda|/\lambda$ bounds the normalization of the exact block $j/N$, not every rounding in the probability evaluation. A rounded normalization does not by itself prove that the exact ratio $hj/\lambda$ lies in $[-1,1]$, which the Lipschitz relation assumes.
- The terms $e_{\mathrm{emit}}$, $e_u$ and $e_{\mathrm{exec}}$ have no bound and are recorded as unknown. The numerical state error `kernel_error` leaves them out, so it is a numerical estimate of part of $\delta_k$ and neither a bound on the state nor a certified success probability.

The coefficient state enters the output twice. Comparing the actual and ideal coefficient states around the ideal branch unitary $V$ gives $\lVert\langle\widehat p|V|\widehat p\rangle-\langle p|V|p\rangle\rVert\le2\delta_k$, on the enlarged work space of [the source construction](#source-preparation). With the recovered evolution error counted separately, the physical contribution is $2\gamma\delta_k$, with no extra factor $\zeta$. The Plan gives this product, evaluated from `kernel_error`, as the known part in the reason of its `lcu_coefficient_preparation` entry. The value of that entry stays unknown, because the total error of the coefficient preparation is unknown.

## Apel–Berntson kernel and its radius {#apel-berntson}

### Source relations and the supplied radius {#apel-berntson-sources}

For $R>0$ put $\mu=2\arctan(R)/\pi$, so $0<\mu<1$, and for a continuous allowance $\epsilon_c$ put $\Delta=\log((1-\mu)/\epsilon_c)/\mu$, which must be positive. The kernel on $(-R,R)$ is

```math
g_R(k)=\frac{e^{(1-\mu)\Delta}}{\pi(1+k^2)}\exp\Bigl(-\frac{i\Delta}\pi\log\frac{R+k}{R-k}\Bigr),
```

with continuous coefficient mass $\beta_R=\mu e^{(1-\mu)\Delta}$. All statements below take a Hermitian $H$, a PSD processed part $L_s$ with $\lVert L_s\rVert\le B_{L_s}$ and nonnegative elapsed time at most $T_e$. The relations come from Apel and Berntson, arXiv:2610.03669v1, a preprint, in these separate places:

| Relation | Location | Used for |
| --- | --- | --- |
| Continuous operator error $\epsilon_c$ at a fixed radius | Theorem 4, Eqs. (19)–(20), and Proposition 10, Eqs. (53)–(56) | Every radius |
| Default radius: $\mu$ solves $\log((1-\mu)/\epsilon_c)=\pi\mu^2/\sin(\pi\mu)$, and $R=\tan(\pi\mu/2)$ | Theorem 6, Eqs. (27)–(28) | `radius=None` |
| Tanh–sinh error at that radius | Theorem 8, Eqs. (39), (41), (42) | Default radius |
| Coefficient mass of the finite rule | Corollary 9, Eqs. (44)–(45) | Reported mass discrepancy, not an operator error |

Theorem 8 is stated at the optimal radius. For a supplied radius NWQLib reuses that proof, through the source's Lemmas 14–17, under four conditions on the strip half-width $\eta$:

```math
\eta<\frac\pi6,\qquad \eta<\arcsin(1-\mu),\qquad
D_\eta=1-\frac{\pi^2}4(1+R^2)\sin^2\eta>0,\qquad
\pi\cos\eta-\Delta\sin\eta>0.
```

Those proof steps use no stationarity identity of the optimal radius, so these conditions are what the adaptation needs. It is NWQLib's adaptation, not a statement printed in the paper. Under them the finite rule $k=R\tanh(\frac\pi2\sinh\xi)$ with step $h$ in $\xi$ and $2N+1$ nodes has the two error terms

```math
\frac{2M_\eta}{e^{2\pi\eta/h}-1},\qquad
\frac{4Re^{(1-\mu)\Delta}}\pi\,e^{-\pi\sinh(Nh)},
```

```math
M_\eta=\frac{4Re^{(1-\mu)\Delta+\Delta\sin\eta}}{D_\eta(\pi\cos\eta-\Delta\sin\eta)}
\exp\Bigl(\frac{R\pi\sin\eta}{2-\pi^2\sin^2\eta/2}\,T_eB_{L_s}\Bigr).
```

Planning tries a finite list of $\eta$ strictly inside these conditions and keeps the one with the fewest nodes, which is not a minimum of complete circuit cost. The formulas are evaluated in binary64 without an enclosure.

The continuous error, the two quadrature terms, the mass discrepancy and the final physical-output error are different quantities. The first three are operator-norm components for a unit input, before input, source and PSD recovery.

### The extra radius of the automatic choice {#apel-berntson-radius}

For `Samples` and `accuracy` requests, the [automatic kernel and grid choice](lchs.md#automatic-kernel-and-grid) adds one radius from a leading-cost heuristic. With $\Lambda(\mu)=\log((1-\mu)/\epsilon_c)$, differentiating $\log\beta_R=\log\mu+(1-\mu)\Lambda/\mu$ gives $d\log\beta_R/d\mu=-\Lambda/\mu^2$, and $d\mu/dR=2/(\pi(1+R^2))$. Two leading-cost surrogates of the query normalization of the coherent route then have these stationary equations:

| Generator | Surrogate | Stationary equation |
| --- | --- | --- |
| Separate parts, $a=\alpha_L>0$, $b=\alpha_H\ge0$ | $J_{\mathrm{sep}}(R)=(aR+b)\beta_R^2$ | $\Lambda-\dfrac{\pi\mu^2(1+R^2)a}{4(aR+b)}=0$ |
| One encoding of $A$, normalization $\alpha_A$ | $J_A(R)=\alpha_A\sqrt{1+R^2}\,\beta_R^2$ | $\Lambda-\dfrac\pi4\mu^2R=0$ |

Here $a$, $b$ and $\alpha_A$ are the normalizations of the actual child encodings, not target norms. Each equation has an interior root on $0<\mu<1-\epsilon_c$ by its endpoint signs and continuity. That argument gives neither uniqueness nor a global minimum. The surrogates leave out the finite grid, whose largest node can be below $R$, gate overhead, degree changes, different recoveries and unknown costs. The root found by bisection is therefore one proposal, and planning builds it and ranks it with the other pairs by the ordinary finite-cost comparison.
