# dgm_eval: gFID

This folder has a smaller copy of [dgm-eval](https://github.com/layer6ai-labs/dgm-eval). The original README is in [`README_native.md`](README_native.md).

## Differences from the original

- Encoders: only Inception and RadioDINO.
- Metrics: only FD, KD and PRDC.
- We add `--avg-all-channels`, which averages the RGB channels of each image. Use it for grayscale medical images.

## Usage

Run this in the repository root:

```bash
python -m dgm_eval <reference_dir> <samples_dir> \
    --model inception --metrics fd kd prdc \
    --batch_size 1024 --nsample 15000 \
    --avg-all-channels --no-load
```

The FID uses all `--nsample` images. PRDC uses at most `--reduced_n` images (default 10,000).
