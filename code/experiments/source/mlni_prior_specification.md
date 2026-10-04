# Public-information priors for MLNI

The nominal estimator uses the updates in Wang et al., *Online Spatial
Crowdsensing With Expertise-Aware Truth Inference and Task Allocation*, IEEE
JSAC 40(1), 2022, DOI 10.1109/JSAC.2021.3126045, Eqs. (14)–(16). The shared WLS
adapter and public-information hyperparameters are disclosed adaptations; the
MLNI core is not a new SourceBridge contribution.

For public domain T=[L,U], let c=(L+U)/2 and s=U-L. WLS returns source gain
a_i and bias b_i. Normalize x_ji=((Y_ij-b_i)/a_i-c)/s and q_j=(c_j-c)/s.
The latent truth prior is N(0,1/12), the Gaussian moment match of Uniform[L,U]
in normalized units. It is not a truncated or uniform posterior.

For K purchased references, let r_ji=x_ji-q_j, eta_i=mean_j(r_ji), and
V_emp,i=mean_j((r_ji-eta_i)^2). Define

    V_target,i = epsilon_t^2 / (3 a_i^2 s^2)
    V_ref,i = mean_j((epsilon_j^2/a_i^2+h_j^2)/(3s^2))
    v0_i = max(V_emp,i, V_target,i, V_ref,i).

The bias prior is N(eta_i,v0_i/2); the variance prior is IG(2,3v0_i), with
density proportional to v^(-3) exp(-3v0_i/v). Its mode is v0_i and its mean
is 3v0_i. Initialize bias at eta_i, variance at v0_i, and truth at the source
mean. With N=K+1 task rows, the conditional updates are

    z_j = [sum_i (x_ji-h_i)/v_i] / [12+sum_i 1/v_i]
    h_i = [eta_i/(v0_i/2)+sum_j (x_ji-z_j)/v_i]
          / [1/(v0_i/2)+N/v_i]
    v_i = [3v0_i+0.5 sum_j (x_ji-z_j-h_i)^2] / [3+N/2].

The implementation stops when the largest state change is below 1e-6, with
20,000 iterations and a 1e-10 machine guard. The current contract implies
v_i >= 1/109850, so that guard is inactive for valid finite inputs. The public
variance scale is used for every K; K=1 does not identify both affine parameters.
The reference-only empirical prior remains a separate comparator interface.

The nominal independence and plug-in gain assumptions provide a point estimator.
SourceBridge's deterministic radius comes from the interval-depth certificate
and safe projection, not from a posterior variance or Gaussian coverage claim.
Both standalone proper MLNI and SourceBridge use the same nominal prediction.
