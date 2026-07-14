# Targeted-GDELT News NASCDI NARDL Equations

Let \(P^T_t\) be the terminal-market price, \(P^P_t\) the producer-market price, and \(N_t\) the weekly targeted-GDELT news NASCDI. The reported model uses \(y_t=\log(P^T_t)\) and \(x_t=\log(P^P_t)\).

## NASCDI construction

For each article \(i\) observed in week \(t\), the lexicon score is:

\[
s_i = \sum_{k \in D}w_k\mathbf{1}(k \in i) - \sum_{m \in M}v_m\mathbf{1}(m \in i),
\]

where \(D\) is the disruption lexicon, \(M\) is the mitigation/reopening lexicon, and only articles satisfying the Kashmir/NH-44/apple-market context filter and a minimum material-score threshold enter the index. The raw weekly intensity is:

\[
R_t=\sum_{i \in t}|s_i|.
\]

The normalized index is:

\[
N_t=100+10\left(\frac{R_t-\bar R}{s_R}\right).
\]

## Positive and negative partial sums

\[
\Delta N_t=N_t-N_{t-1},\qquad
N_t^+=\sum_{j=1}^{t}\max(\Delta N_j,0),\qquad
N_t^-=\sum_{j=1}^{t}\max(-\Delta N_j,0).
\]

Here \(N_t^+\) represents disruption intensification and \(N_t^-\) represents disruption easing. Both are non-negative cumulative processes.

## Estimated log NARDL error-correction model

\[
\begin{aligned}
\Delta y_t={}&c+\lambda y_{t-1}+\beta x_{t-1}+\theta^+N^+_{t-1}+\theta^-N^-_{t-1}\\
&+\sum_{i=1}^{4}\phi_i\Delta y_{t-i}
+\sum_{j=0}^{4}\psi_j\Delta x_{t-j}\\
&+\sum_{j=0}^{4}\gamma_j\Delta N^+_{t-j}
+\sum_{j=0}^{4}\delta_j\Delta N^-_{t-j}+\varepsilon_t.
\end{aligned}
\]

The results use Newey-West HAC standard errors with a four-week bandwidth. A stable error-correction relationship requires \(\lambda<0\).

## Long-run effects

Provided \(\lambda\ne0\), the long-run producer-price pass-through and asymmetric NASCDI effects are:

\[
LR_x=-\frac{\beta}{\lambda},\qquad
LR_+=-\frac{\theta^+}{\lambda},\qquad
LR_-=-\frac{\theta^-}{\lambda}.
\]

The long-run asymmetry test is \(H_0:\theta^+=\theta^-\). The short-run asymmetry test is:

\[
H_0:\sum_{j=0}^{4}\gamma_j=\sum_{j=0}^{4}\delta_j.
\]

These are HAC Wald tests. The joint lagged-level F statistic in the output is a diagnostic for the lagged-level block; it is not a replacement for published Pesaran-Shin-Smith bounds critical values.
