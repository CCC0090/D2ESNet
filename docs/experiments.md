# Experiments

## Datasets

| Dataset | Images | Input size | Ships | Inshore | Offshore | Small targets |
|---|---:|---:|---:|---:|---:|---:|
| HRSID | 5,604 | 800 x 800 | 16,951 | 18.4% | 81.6% | 54.5% |
| SRSDD-v1.0 | 666 | 1024 x 1024 | 2,884 | 63.1% | 36.9% | 25.14% |

The reported comparisons use identical dataset splits, preprocessing, and
evaluation settings for all methods. Images are resized to 800 x 800 for the
reported training and runtime measurements.

## Training protocol

- 100 epochs, batch size 32, no pretrained weights;
- three independent random seeds;
- default internal spiking steps \(T=3\);
- AdamW-style optimization with automatic learning-rate selection;
- COCO-style detection evaluation for mAP and AP-small.

## Reported results

| Dataset | mAP50 | mAP50-95 | AP-small | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|---:|
| HRSID | 83.94 | 61.09 | 49.77 | 2.0619 | 8.8886 |
| SRSDD-v1.0 | 52.16 | 29.18 | 11.87 | 2.0619 | 8.8886 |

The reported values are the mean of three independent runs. Runtime and
energy measurements use an NVIDIA H100 GPU.

## Ablations

The manuscript reports independent ablations for:

1. dynamic routing versus fixed ANN ratios;
2. direct repetition, local contrast, directional scattering, and complementary
   evidence encoding;
3. spike-only, ungated recovery, residual-potential gating, and evidence-gated
   recovery;
4. internal time steps \(T\in\{1,2,3,4\}\).

The time-step study shows that the largest gains occur between one and three
steps; additional gains at \(T=4\) are smaller and vary across datasets.

