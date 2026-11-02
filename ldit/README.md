# Rad-DiT: latent diffusion on medical images

This folder has our LightningDiT code for RadImageNet and BraTS. The original README is in [`README_native.md`](README_native.md).

Run all commands in `ldit/`, unless a step says otherwise.

## 1. Download

Download the weights from the original authors:

```bash
hf download hustvl/vavae-imagenet256-f16d32-dinov2 --local-dir pretrained/vavae-imagenet256-f16d32-dinov2

hf download hustvl/lightningdit-xl-imagenet256-800ep --local-dir pretrained/lightningdit-xl-imagenet256-800ep
```

## 2. Extract the latents

Change the paths in the commands and in `configs/data_*.yaml` if your data is in a different place.

**RadImageNet:** first make the held-out split, see [`../preprocessing/`](../preprocessing/). `--exclude_paths` is the list of held-out images, which do not go into the training shards.

```bash
python preprocessing/extract_latents_radimagenet.py \
    --data_path ../data/radimagenet --output_path ../data/latents/radin/train \
    --exclude_paths ../data/val_paths.txt
```

**BraTS:** first make the PNG folders, see [`../preprocessing/`](../preprocessing/).

```bash
python preprocessing/extract_latents_brats.py \
    --data_path ../data/brats_images/train --output_path ../data/latents/brats
```

## 3. Train

Other training configs are in [`configs/`](configs/).

```bash
sbatch slurm/train.sh configs/lightningdit_xl_radin_vavae_region5_256_pretrained_100k.yaml
```

**Without SLURM (local machine or cloud instance such as AWS):** use `bash` in place of `sbatch`, with the same arguments. The scripts use 2 GPUs. For a different number, change `NGPUS` in the script. This also applies to `slurm/calculate_gfid.sh`.

## 4. Eval

Sample and calculate gFID:

```bash
sbatch slurm/calculate_gfid.sh configs/inference_radin_region5_pretrained.yaml \
    <ckpt> <reference_dir> <output_dir>
```

`<ckpt>` is `output/<exp_name>/checkpoints/<step>.pt`. `<reference_dir>` is the folder of held-out PNG images: `../data/radin_val` for RadImageNet, and `../data/brats_images/val/0` for BraTS, see [`../preprocessing/`](../preprocessing/). The script generates and compares 15,000 images. The gFID is the `fd` line in the job log.

For BraTS, use `configs/inference_brats_hastumor.yaml`.

## 5. Analysis

| Script | Question |
|---|---|
| [`analysis/linear_probe.py`](analysis/linear_probe.py) | Do block features predict the pathology? |

The probe reads the flat folder `../data/radin_val`, see [`../preprocessing/`](../preprocessing/).

## 6. Tips

- **Differences from the original:** we use only VA-VAE, and we added a shard-based data loader (`train_shards.py`, `shardloader/`). See "Data loading" below.
- **Class conditioning:** medical classes start at 1001. The CFG row is the last row (`num_classes`), in training and in sampling. RadImageNet: `num_classes: 1006`. BraTS: `num_classes: 1003`.
- **Pretrained start:** as in the official code, `model` weights go into the model and the EMA. The optimizer is new.
- **Resume:** restores the model, the EMA, the optimizer and the step.
- **Latent statistics:** we use the ImageNet VA-VAE statistics, from the VA-VAE download. Training normalizes each latent with these statistics. Sampling reverses this before the VAE decodes: `(z * std) / latent_multiplier + mean`. So training and sampling must use the same statistics. To use your dataset's statistics, set `vavae_stats_path: null` in the training config, and delete `latents_stats.pt` in the shard folder if it exists. Then set `data_path` in the inference config to the folder that has your `latents_stats.pt`.
- **WandB (our addition):** the training configs set `use_wandb: true`. Run `wandb login`, or set `use_wandb: false`. Set `wandb_entity` to your own account.
- **QK-norm:** keep `use_qknorm: true`. Without it, we saw stability problems.

## Data loading (our change)

The official code stores the latents in safetensors files. We added two parts to read these files more efficiently:

- **Window sampler (`shard_window_size`):** shuffles the shards and makes groups. It then shuffles the samples in each group. A batch has samples from one group only, except at group boundaries.
- **LRU cache (`cache_size`):** keeps the most recently used shards open. Keep `cache_size` at or above `shard_window_size`.

## Future work

- Gradient checkpointing
- FSDP
