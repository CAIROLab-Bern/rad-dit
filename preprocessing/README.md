# Preprocessing

Run these commands in the repository root.

## RadImageNet: held-out split

JiT and LDiT need this step. The script makes:

- `data/val_paths.txt`: the held-out images, one path per line. Training excludes them.
- `data/radin_val/`: the same images in one flat folder, as 256×256 PNG files named `<modality>_<bodypart>_<pathology>_<file>.png`. gFID and the analysis scripts read this folder.

```bash
python preprocessing/make_radin_split.py \
    --data_path data/radimagenet --val_paths data/val_paths.txt --out_dir data/radin_val
```

The defaults (`--val_size 15000`, `--seed 42`) give our split.

## BraTS: what the preprocessing does

Both routes share [`brain_dataset.py`](brain_dataset.py), which does these steps:

1. Read the four MRI modalities (T1n, T1c, T2w, T2f) and the segmentation, if it exists.
2. Normalize the intensities and pad and center-crop each volume to 192×192.
3. Keep the axial slices 15 to 154, which is 140 slices per patient.
4. Label each slice `has_tumor` from the segmentation. Validation data has no segmentation, so all its slices get label 0.

Then the routes separate. Both models train from shards: JiT shards hold pixels, and LDiT shards hold latents.

| | JiT | LDiT |
|---|---|---|
| Space | Pixels | Latents |
| Middle output | None | PNG files, 8-bit, 192×192, in `train/0/` and `train/1/`, and `val/0/` for validation data |
| Final output | Pixel shards | Latent shards, see [`../ldit/`](../ldit/) |
| One row in a shard | One slice: all 4 modalities at 256×256, plus segmentation, label and IDs | One slice of one modality: a latent of 32×16×16, plus the label |
| Samples per patient | 140 slices, each with 4 channels | 560 images (140 × 4) |
| Number format | `float16` | PNG is 8-bit. The latents are `float32`. |

For single-channel JiT training, `--brats_flat` shows each modality as a separate sample.

## Why shards

One shard holds many samples, so training reads a few large files and not many small files. The BraTS pixel shards are our addition. For the LDiT loader, see [`../ldit/`](../ldit/).

## BraTS: commands

**JiT:** make the pixel shards. The gFID reference comes from the PNG command below, so also run that command.

```bash
python preprocessing/extract_brats_multichannel_safetensors.py \
    --train-dirs data/brats24/BraTS-GLI/training_data1_v2 data/brats24/BraTS-GLI/training_data_additional \
    --output-dir data/brats24_gli_multichannel_256 --skip-val
```

**LDiT:** make the PNG folders. Then extract the latents, see [`../ldit/`](../ldit/).

```bash
python preprocessing/extract_brats_images.py \
    --train-dirs data/brats24/BraTS-GLI/training_data1_v2 data/brats24/BraTS-GLI/training_data_additional \
    --output-dir data/brats_images
```

This also writes `data/brats_images/val/0/`, the gFID reference for BraTS. The PNG files have 8-bit values, so these latents are not identical to the latents of the paper.
