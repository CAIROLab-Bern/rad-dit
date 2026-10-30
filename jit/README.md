# Rad-JiT: pixel-space diffusion on medical images

This folder has our JiT code for RadImageNet and BraTS. The original README is in [`README_native.md`](README_native.md).

## 1. Data

**RadImageNet:** make the held-out split. `--exclude_paths` is the list of held-out images, which do not go into training. See [`../preprocessing/`](../preprocessing/).

**BraTS:** make the pixel shards, see [`../preprocessing/`](../preprocessing/).

## 2. Train

Run all commands in `jit/`. Download the ImageNet JiT weights, see [`README_native.md`](README_native.md). `<imagenet_ckpt>` is the checkpoint of the same model size. Then use `sbatch slurm/train.sh <output_dir> <flags>`. These are examples. Change the paths to your own. Change `--epochs` for your compute budget: we trained RadImageNet for 80 epochs and BraTS for 40.

**RadIN-B/16**

```bash
sbatch slurm/train.sh output/radin_b16 \
    --model JiT-B/16 --epochs 80 \
    --dataset radimagenet --data_path ../data/radimagenet --exclude_paths ../data/val_paths.txt \
    --label_mode extended --label_offset 1001 --class_num 1005 --cfg_drop_index 1000 --eval_class_num 5 \
    --pretrained_ckpt <imagenet_ckpt>
```

**RadIN-L/16:** as above, with `--model JiT-L/16`.

**Mixed15%-L/16**

```bash
sbatch slurm/train.sh output/mixed15_l16 \
    --model JiT-L/16 --epochs 80 \
    --dataset mixed_radnat --data_path ../data/radimagenet --exclude_paths ../data/val_paths.txt \
    --nat_data_path ../data/imagenet --p_nat 0.15 \
    --label_mode extended --label_offset 1001 --class_num 1005 --cfg_drop_index 1000 --eval_class_num 5 \
    --pretrained_ckpt <imagenet_ckpt>
```

**BraTS-B/16 (single channel)**

```bash
sbatch slurm/train.sh output/brats_b16 \
    --model JiT-B/16 --epochs 20 --warmup_epochs 0 \
    --dataset brats_gli_multichannel --brats_shard_dir ../data/brats24_gli_multichannel_256/train_shards --in_channels 3 --brats_flat \
    --label_mode extended --label_offset 1001 --class_num 1002 --cfg_drop_index 1000 --eval_class_num 2 \
    --pretrained_ckpt <imagenet_ckpt>
```

**Without SLURM (local machine or cloud instance such as AWS):** use `bash` in place of `sbatch`, with the same arguments. The scripts use 2 GPUs. For a different number, change `--nproc_per_node` in the script. This also applies to `slurm/calculate_gfid.sh`.

## 3. Eval

Sample and calculate gFID:

```bash
sbatch slurm/calculate_gfid.sh output/radin_b16 <reference_dir> \
    --model JiT-B/16 \
    --label_mode extended --label_offset 1001 --class_num 1005 --cfg_drop_index 1000 --eval_class_num 5
```

`<reference_dir>` is the folder of held-out PNG images: `../data/radin_val` for RadImageNet, and `../data/brats_images/val/0` for BraTS, see [`../preprocessing/`](../preprocessing/). The script generates and compares 15,000 images. The gFID is the `fd` line in the job log.

For BraTS, use the class flags and the channel flags of the BraTS training example.

## 4. Analysis

| Script | Question |
|---|---|
| [`analysis/linear_probe.py`](analysis/linear_probe.py) | Do block features predict the pathology? |
| [`analysis/compute_lds_c.py`](analysis/compute_lds_c.py) | How spatially structured are the block features? This computes LDS<sub>C</sub>, our metric. |
| [`analysis/representation_similarity.py`](analysis/representation_similarity.py) | Do two images give similar block features? |
| [`../metrics/radial_spectrum.py`](../metrics/radial_spectrum.py) | Do generated images have the real frequency content? |

The probes and the similarity script read the flat folder `../data/radin_val`, see [`../preprocessing/`](../preprocessing/).

**Representation similarity:** download the 25% mixed JiT-L/16 checkpoint (coming soon). Then run:

```bash
python analysis/representation_similarity.py \
    --ckpt <mix25_ckpt> --model JiT-L/16 \
    --radin_val ../data/radin_val --imnet_train ../data/imagenet/train \
    --mode cond --class_num 1005 --cfg_drop_index 1000 \
    --k_per_class 64 --n_per_t 64 --batch_size 8 \
    --out_csv mix25.csv --out_png mix25.png
```

## 5. Tips

- **Differences from the original:** we add RadImageNet, BraTS and mixed datasets, and a pretrained start. We do not compute FID during training. With `--evaluate_gen`, the script loads no training data, so generating images needs only a checkpoint.
- **Class conditioning, ImageNet start:** medical classes start at 1001, and the CFG row is the ImageNet row 1000. Always pass `--cfg_drop_index 1000`. Otherwise the CFG row collides with the last medical class, and the code stops with an error.
- **Class conditioning, scratch:** classes are 0 to N-1, and the CFG row is N. Remove `--pretrained_ckpt` and use `--label_mode plain --label_offset 0 --class_num N`, with no `--cfg_drop_index`.
- **Pretrained start (ours, the official code has none):** EMA weights go into the model and the EMAs. The optimizer is new.
- **WandB (our addition):** add `--use_wandb` to log to Weights & Biases. Run `wandb login` first.
- **Resume:** as in the official code, restores the model, the EMAs, the optimizer and the epoch.

## Future work

- Gradient checkpointing
- FSDP
- Window sampler from LDiT, for the BraTS pixel shards
