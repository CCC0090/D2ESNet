# Routing algorithm

```text
Input:
    feature map X with C channels
    candidate ANN ratios R = {0.25, 0.50, 0.75}

1. Compute channel magnitude A = mean_c |X_c|.
2. Compute multiscale contrast D3, D5, D7 and direction D_dir.
3. Encode (A, D3, D5, D7, D_dir) into latent density/difficulty maps.
4. Build a six-dimensional scene descriptor.
5. Predict path probabilities p over R.

Training:
    Execute every candidate width.
    Combine candidate outputs using p.
    Add route balance and entropy regularization.

Inference:
    Select r* = argmax_r p_r.
    Execute only the corresponding ANN-SNN width.
    Return the fused residual output and selected route metadata.
```

The latent density and difficulty maps are learned proxies for scene
complexity. They are not presented as measured physical ship-density maps.

