# Method

D2ES-Net couples a scene-adaptive ANN-SNN routing branch with a stateful,
evidence-driven spiking representation.

## 1. Scene-aware channel allocation

For a shared feature map \(X \in \mathbb{R}^{C \times H \times W}\), the router
computes a channel-averaged magnitude map and three scattering statistics:

- multiscale local contrast \(D_3\), \(D_5\), and \(D_7\);
- horizontal/vertical directional scattering \(D_{\mathrm{dir}}\).

A compact convolutional encoder produces latent density and difficulty maps.
Spatial means and maxima form a six-dimensional scene descriptor. A two-layer
policy predicts probabilities over the ANN ratios
\(\mathcal{R}=\{0.25,0.50,0.75\}\).

During training, all candidate widths are evaluated and combined by the routing
probabilities. During inference, only the width with the maximum probability is
executed.

## 2. Stateful spiking representation

For the selected SNN channels \(X_s\), multiscale contrast and directional
responses are encoded into \(T\) spatial evidence maps:

\[
(E_1,\ldots,E_T)=\sigma(f_{\mathrm{ev}}(X_s)).
\]

Each internal step follows a soft-reset leaky integrate-and-fire update:

\[
\tilde U_t=\lambda U_{t-1}+X_s\odot E_t,\qquad
S_t=\mathbb{1}[\tilde U_t-\theta\ge 0],\qquad
U_t=\tilde U_t-\theta S_t.
\]

The evidence maps are not repeated copies of the input; they modulate the
spiking path through learned spatial responses.

## 3. Subthreshold recovery

The final post-reset membrane state and all evidence steps determine the
recovery gate:

\[
M_{\mathrm{fired}}=\max_t S_t,\qquad
E_{\max}=\max_t E_t,
\]

\[
R_T=\tanh\left(\frac{\operatorname{ReLU}(U_T)}
{\theta+\epsilon}\right),
\]

\[
G_T=(1-M_{\mathrm{fired}})\odot E_{\max}.
\]

The recovered spiking feature is

\[
F_s=\operatorname{DWConv}\left(\frac{1}{T}\sum_{t=1}^{T}S_t\right)
+\eta G_T\odot R_T.
\]

The continuous and spiking paths are fused through a normalized projection and
a residual connection.

## 4. Optimization

The detection objective is combined with a routing regularizer that balances
path usage while preserving route entropy during training. No extra annotation
is required for density, scene difficulty, or subthreshold responses.

