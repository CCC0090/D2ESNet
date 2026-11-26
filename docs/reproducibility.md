# Reproducibility and measurement protocol

## Deterministic training

- Save the exact data manifest and split checksum.
- Fix Python, PyTorch, CUDA, and GPU driver versions.
- Set the data-loader seed, sampler seed, and PyTorch CUDA seed.
- Record the checkpoint SHA-256 and source commit.
- Report the mean and standard deviation over three independent seeds.

## Accuracy

- Evaluate mAP50, mAP50-95, and AP-small with the COCO bbox protocol.
- Keep confidence, NMS IoU, maximum detections, and input size identical across
  compared methods.

## Static complexity

Static parameter count is obtained by summing model parameters. Static GFLOPs
should be used only as a nominal complexity indicator; it does not model
data-dependent routing.

## Routed efficiency

For each test image:

1. execute the router in evaluation mode;
2. count FLOPs only along the selected ANN-SNN path using a PyTorch FLOP counter;
3. record the selected ANN ratio and SNN channel ratio;
4. average the per-image results over the complete test split.

The SNN channel ratio is

\[
\rho_s=\frac{C-m_{\mathrm{ANN}}}{C}.
\]

Average Routed GFLOPs and Average SNN Channel Ratio should not be mixed with
static model GFLOPs because their execution-path assumptions and operator
coverage differ.

## Latency

For a reproducible runtime protocol:

- use batch size 1, FP32, and 800 x 800 input;
- warm up the model before measurement;
- synchronize GPU execution before and after timing;
- perform at least 100 forward iterations;
- report the same GPU, driver, CUDA, PyTorch, and clock settings for all models.

Absolute latency is hardware-dependent; relative comparisons should be made
under an otherwise idle GPU.

