# Canonical phase synthesis

The canonical functions `solve_wx_qsp_phases` and `solve_laurent_qsp_phases` in `subroutines/qsp/phases.py` synthesize a real definite-parity Chebyshev polynomial in the Wx convention or a general complex Laurent polynomial on the unit circle. Both use a canonical inverse nonlinear Fourier transform (NLFT). Their output is checked by reconstructing polynomial coefficients from the returned angles. The resulting residual is a numerical uniform-error estimate because the arithmetic of the FFT, transcendental functions and reconstruction has no complete enclosure.

## Canonical complement and Fourier convention

For an analytic polynomial b of degree d with a known cap ||b||∞ <= rho < 1, define the Laurent conjugate f\*(z)=conj(f(1/conj(z))) and

```math
R(\theta)=\frac12\log(1-|b(e^{i\theta})|^2),\qquad
G_-(z)=r_0+2\sum_{k<0}r_kz^k,\qquad a(z)=e^{G_-(z)}.
```

Since R is real, Re G_-=R on the circle. Hence aa\*+bb\*=1. Fejér–Riesz factorization and exterior outer uniqueness give a with powers -d,...,0 and a(infinity)>0. The strict cap excludes unit-circle zeros. This defines the complement without polynomial root finding.

At z_j=exp(2 pi i j/N), coefficients-to-values uses N IFFT and values-to-coefficients uses FFT/N. The implementation evaluates b, then R with log1p, forms the negative-frequency Schwarz part, exponentiates -G_-, and transforms b/a back to its coefficients chi_0,...,chi_d. The Nyquist bin belongs to the negative side. A nonpositive logarithm argument terminates the attempt.

These conventions follow the complex form of the Riemann–Hilbert construction in [Laneve, arXiv:2503.03026v2, Sections 3–5](https://arxiv.org/html/2503.03026v2), with the Fourier discretization estimates of [arXiv:2407.05634v3, Propositions 17–23](https://arxiv.org/html/2407.05634v3#S5.SS1).

## Causal prefixes and the inverse recursion

Let alpha and beta contain the first h coefficients of two analytic series, with nonzero alpha_0. One stripping step is

```math
f=\beta_0/\alpha_0,\qquad \nu=\sqrt{1+|f|^2},\qquad
\alpha'=\frac{\alpha+\bar f\beta}{\nu},\qquad
\beta'=\frac{\beta-f\alpha}{\nu z}.
```

Only the first h-1 new coefficients are needed. A common analytic scaling of alpha and beta does not change the factors. Consequently a recursive prefix need not be normalized and receives no new complement. A common constant phase can make alpha_0 positive without changing f.

The independent displacement formulation uses lower-triangular Toeplitz matrices T(alpha), T(beta):

```math
K=T(\alpha)T(\alpha)^\dagger+T(\beta)T(\beta)^\dagger=L\Delta L^\dagger,
\qquad Ly=\beta/\alpha_0.
```

For positive alpha_0, K is positive definite and its displacement is `K - Z K Z† = [alpha, beta][alpha, beta]†`. A two-column QR recurrence obtains the unit-diagonal L columns and performs the forward solve. The finite-prefix algebra gives the same y as stripping. The leaf output is y itself, with no reversal. This follows the causal argument in [arXiv:2505.12615v1, Lemma 5.2 and Eq. (53)](https://arxiv.org/html/2505.12615v1#S5.SS3). Production uses the stripping form.

Every recursive return includes the normalized transfer pair (A,B)=(a_y\*,b_y) formed from the extracted factors. Split h into m=ceil(h/2) and h-m factors. After solving the first m coefficients, the right prefixes are

```math
\alpha_R=A_L^*\alpha+B_L^*\beta,\qquad
\beta_R=z^{-m}(A_L\beta-B_L\alpha).
```

The merged transfer pair is

```math
A=A_LA_R-z^mB_L^*B_R,\qquad
B=z^mA_L^*B_R+B_LA_R.
```

Here star reverses the Laurent support as well as conjugating coefficients. The shift m counts factors, while a transfer pair for m factors has degree m-1. The equations follow by multiplying M_left D^m M_right D^-m. In the first equation the Laurent shift is -(m-1), so the implementation extracts the convolution slice beginning at m-1. The beta slice begins at m. Every FFT convolution is padded to its complete linear-product length.

The ratio enters this recursion as alpha=(1,0,...,0), beta=conj(reverse(chi)). If y is the output, the original NLFT factors are F=conj(reverse(y)). To see this, the reflected-conjugate sequence corresponds to the normalized pair (a,z^d b\*). Dividing its analytic pair (a\*,z^d b\*) by a\* gives (1,h), where h=z^d(b/a)\* and its first d+1 coefficients are beta. Causality then gives the same factors. Equivalently K=I+T(beta)T(beta)†, which is the ratio Riemann–Hilbert system's Schur complement. This exact equivalence does not supply binary64 error constants for the scaled prefixes.

## Two output conventions

For f(z)=sum f_k z^k with support contained in [-ell,u], including zero in that interval, set p=z^ell f and b=-i p. Define

```math
A(F)=\frac1{\sqrt{1+|F|^2}}\begin{pmatrix}1&F\\-\bar F&1\end{pmatrix},\qquad D(z)=\operatorname{diag}(z,1).
```

The first row of A_0 D ... D A_d is (z^d a,b), so right multiplication by iX gives top-left entry ib=p. Replacing the first ell mathematical signals by diag(1,z^-1)=z^-1 D(z) yields f. With phi=atan|F| and psi=(arg F-pi/2)/2,

```math
A(F)=R_Z(-2\psi)R_X(-2\phi)R_Z(2\psi),\qquad iX=R_X(-\pi).
```

Circuit append order is the reverse of this matrix product. The terminal iX and each negative signal affect controlled and adjoint action and are part of the returned convention.

For a real Chebyshev target f(x)=sum c_k T_k(x) of parity n and degree n, construct

```math
b(z)=i\left[\mathbf1_{n\text{ even}}c_0z^{n/2}+\frac12\sum_{k>0}c_k\left(z^{(n+k)/2}+z^{(n-k)/2}\right)\right].
```

Only coefficients of parity n occur, so the powers are integers and the degree is n. For w=exp(i theta) and x=cos theta, w^-n b(w²)=i f(x). The exact factors are purely imaginary. Let alpha_k=atan(Im F_k), then subtract pi/4 at both endpoints to obtain the Wx phases. For n=0 the sole phase receives both subtractions. The resulting Wx product has Re U_00=f. Numerical projection onto the imaginary factors is covered by reconstruction from the resulting phases.

The Wx-to-reflection conversion uses W=i exp(-i pi Z/4) R exp(-i pi Z/4). Endpoint phases decrease by pi/4, interior phases decrease by pi/2, and the full matrix receives the compensating global phase n\*pi/2. Degree zero needs no conversion. QLS synthesizes its normalized target P/s with `solve_wx_qsp_phases`, where s=B_P(1+QLS_TARGET_MARGIN), and its circuit uses this reflection conversion. For odd degree it keeps the Z on the qubit that averages the +Phi and -Phi passes ([Result 21](../mathematics.md#r21)). Its inverse recovery multiplies by s, the ideal success mass scales as 1/s², and a normalized phase residual delta contributes k\*s\*delta/||y|| to the conditional relative inverse allowance.

## Ideal discretization bounds

For d>=1, eta=1-rho, log r=-log(rho)/(2d), even N>2d and r^N>2, the generic ideal per-coefficient bound is

```math
\Xi_N=\frac{5r^{-N/2}|\log\eta|}{1-r^{-1}},\qquad
E_\chi(N)=4r^{2d-N}\eta^{-1}+\eta^{-1/2}\operatorname{expm1}(\Xi_N).
```

The annulus argument uses analyticity of z^d b and z^d b\*, followed by the maximum-modulus bound, Fourier aliasing and the Schwarz multiplier. It does not require b to have real coefficients or symmetric support. If Delta=(d+1)E_chi < s_c=1-rho², the Riemann–Hilbert resolvent estimate and unitary-product telescoping give

```math
\|\widehat{\mathcal S}-\mathcal S\|_\infty\le
\frac{(d+1)\Delta(1+s_c)}{s_c(s_c-\Delta)}.
```

A ratio-vector one-norm bound can replace Delta directly. The factors satisfy |Fhat-F|<=Delta(1+s_c)/(s_c(s_c-Delta)), and A(F) is 1-Lipschitz, giving the displayed product bound. The implementation uses half the operational allowance for ideal discretization. These formulas assume exact arithmetic after the ratio approximation and the stated norm-cap premise.

For a full complex approximation p to a unit-modulus target with ||p-target||∞<=delta, including coefficient uncertainty, let b=-i z^ell p/sigma, c_flat=1-sigma^-2, and q=(2delta+delta²)/(sigma²-1)<1. For integer J>=1 define

```math
B_J=\frac{\sqrt{2d+1}}{2\sqrt{J+1}}\frac{q^{J+1}}{1-q},\qquad
V_J=\sum_{j=1}^J\frac{\sqrt{2jd+1}}j q^j.
```

Expanding log(1-e), with e=(pp\*-1)/(sigma²-1), bounds the discarded Fourier coefficient one-norm by B_J. The Schwarz multiplier has one-norm at most two. A degree-J exponential model of the finite logarithm has support [-J²d,d], so both Fourier transforms avoid aliasing into coefficients 0,...,d when N>max(2J,J²+1)d. Comparison with that common model gives

```math
\Delta_\chi\le\frac{2\|b\|_{\mathrm{coeff},1}}{\sqrt{c_{\rm flat}}}e^{V_J}
\left[\operatorname{expm1}(2B_J)+\frac{V_J^{J+1}}{(J+1)!}\right].
```

This bounds the complete ratio error and enters the same resolvent relation. The premise applies to the full complex target. An individual cosine or sine target lacks the unit-modulus premise. A tail bound alone does not quantify the rounding of computed Bessel coefficients.

## Real-Wx annulus bound

The real adapter has an additional identity. Write P(x)=sum c_k T_k(x), f(theta)=P(cos theta) and z=exp(2i theta). Then b(z)=i exp(i d theta)f(theta) and bb\*=f(theta)^2. This identity requires real coefficients and definite parity. It does not apply to a general complex target merely because some of its coefficients are real.

Suppose the actual normalized polynomial satisfies ||P||∞<=rho<1. Define

```math
s=(1-\rho)(1+\rho),\qquad h=s/2,\qquad K=\cosh(1/2),
\qquad C_1=\sum_{k=1}^d k|c_k|,\qquad C_2=\sum_{k=1}^d k^2|c_k|.
```

For |v|<=y<=1/(2d), the cosine addition formula and the bounds cosh(ky)-1<=k²y²K/2 and sinh(ky)<=kyK give

```math
|f(\theta+i v)|^2\le\rho^2+y^2C,\qquad
C=\rho K C_2+K^2 C_1^2+\frac{K^2C_2^2}{16d^2}.
```

The real part is compared with the actual real-axis value f(theta), whose modulus is at most rho. This avoids replacing rho by the coefficient one-norm. Choosing

```math
y=\min\left(\frac1{2d},\sqrt{\frac{s}{2C}}\right),\qquad \log r=2y
```

gives |bb\*|<=1-h on r^-1<=|z|<=r. Hence R=(1/2)Log(1-bb\*) is analytic there and its Laurent coefficients satisfy |R_k|<=(1/2)log(1/h)r^-|k|. Negative Schwarz truncation and DFT aliasing, including the negative Nyquist bin, give

```math
\|G_N-G\|_\infty\le\Xi_N=
\frac{5\log(1/h)\,r^{-N/2}}{1-r^{-1}}.
```

The second DFT also aliases the true ratio. The identity chi=b a\*/(1-bb\*) gives |chi|<=rho/h on the inner circle. Its positive Laurent support ends at d. Thus, for even N>2d, the error of each computed coefficient chi_0,...,chi_d is bounded in ideal arithmetic by

```math
E_\chi(N)=\frac{\rho}{h}\frac{r^{d-N}}{1-r^{-N}}
+\frac{\rho}{\sqrt{s}}\operatorname{expm1}(\Xi_N).
```

Both terms are necessary. With n=d+1, the sufficient scalar condition

```math
E_\chi(N)\le\min\left(\frac{s}{2n},
\frac{\epsilon s^2}{4n^2(1+s)}\right)
```

implies the earlier resolvent/product bound is at most epsilon/2. This controls uniform output error in exact arithmetic. It does not guarantee the stricter coefficient-one-norm acceptance test on the numerical angles.

The synthesis doubles N from N_min=next_power_of_two(2d+1) and evaluates these scalar bounds only at the lengths it reaches, so the bounds do not depend on the memory limit, and neither does an accepted result whose length the limit admits. It accepts the first length whose numerical coefficient residual meets the tolerance, even when the conditional ideal bound at that length is loose. A residual miss at a length where the generic, real-Wx or near-flat bound is at most half the tolerance ends the doubling with an error that reports every attempted length and residual. The ideal analysis calls that length sufficient, and a longer transform can still change the floating-point residual. An explicit max_fft, or arrays of the next length above the memory limit, also ends the doubling before allocation. A residual that stops decreasing is recorded and does not end it. The error names the last attempted lengths at which no doubling reduced the residual by more than half, together with what stopped the doubling. Such a plateau shows only the absence of improvement at those lengths. It cannot distinguish an arithmetic floor, a limitation of the residual estimator, conditioning or delayed convergence without more evidence, so it does not show whether more memory or a larger tolerance would help. A failure changes neither the target nor its common normalization.

Constants are identified by exact stored support. A nonzero coefficient can produce a computed C=0 by underflow, so the annulus bound does not apply to a nonconstant target with nonpositive or nonfinite C or y. The code evaluates log r directly and uses expm1 for small denominators. These measures do not supply an arithmetic enclosure. When rho is obtained by numerical norming, all sufficient statements remain conditional on that cap.

## Norming and paired evolution preparation

The canonical real path evaluates its Chebyshev-zero norming grid using a DCT-III. At N=64(d+1), pad x_0=c_0, x_k=c_k/2 for k>0. The transform identity

```math
\operatorname{DCT}_{III}(x)_j=x_0+2\sum_{k=1}^{N-1}x_k
\cos\!\left(\frac{\pi k(j+1/2)}N\right)
=P\!\left(\cos\frac{\pi(j+1/2)}N\right)
```

gives the same grid values in O(d log d) work. The Ehlich–Zeller inequality bounds ||P||∞ by the grid maximum divided by cos(pi d/(2N)). The floating-point evaluation is a numerical bound estimate. Constants use abs(c_0) directly. The public function `chebyshev_polynomial_sup_bound` of `nwqlib.subroutines.qsp.phases` performs this evaluation, and QLS uses it for its rescale.

LCHS chooses one scale max(1,B_cos,B_sin)/0.999 and synthesizes both normalized targets under that common scale. Each target doubles its own transform length and checks its own arrays against the memory limit. Jacobi–Anger coefficient and terminal selection is the preceding scalar stage, and only its Bessel arrays are checked against `max_bytes`.

## Reconstruction, numerical evidence and costs

The verification product is rebuilt from the returned rotations in a balanced tree. General complex synthesis includes the actual terminal RX angle. For arbitrary saved Wx phases, A_j=H exp(i phi_j Z) H and H W H=w^-1 D(w²). If M(z)=A_0 D ... D A_n and c_j is half the sum of its four coefficient entries, then U_00=w^-n sum c_j w^(2j). Its real part has coefficients (c_j+conj(c_(n-j)))/2. This identity requires no symmetry of the numerical phases.

For residual coefficients e_j, sup_circle|sum e_j z^j|<=sum|e_j|. The computed sum R_num controls the operational acceptance test. A floating-point certificate would require

```math
\epsilon_{\rm synth}\le R_{\rm num}+\delta_{\rm sum}+\delta_{\rm rec}+\delta_{\rm coeff}.
```

The unquantified arithmetic terms remain unavailable. A local balanced product error would propagate as e_C<=e_A+e_B+e_A e_B+||R||∞, but this relation does not itself enclose a convolution's arithmetic error.

With fixed leaf coefficient count h, total work per attempt is O(N log N+d log²d+d h), and depth-first live-array storage is O(N+d). The reconstruction has the same asymptotic work as the inverse recursion. Before each attempt the dominant arrays are estimated as max(96N+64h, 2432h) bytes with h=d+1. The estimate covers the simultaneous FFT buffers, the live recursive prefixes and transfers, and the reconstruction. Native-library workspace and process RSS are not included.

Across attempts the target coefficients, degree, scale and factorization remain fixed. Nonfinite factors or a nonpositive logarithm terminate the synthesis immediately. A successful residual check qualifies that numerical result, not every polynomial within its degree and norm range.
