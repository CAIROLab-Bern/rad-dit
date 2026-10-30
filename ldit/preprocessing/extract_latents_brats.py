"""
Extract BraTS image latents using VA-VAE tokenizer and save as sharded safetensors.

Reads an ImageFolder-structured dataset (e.g., brats_images/train/) with classes:
    0/  -- no-tumor slices
    1/  -- has-tumor slices

Outputs:
    output_path/
    └── shards/
    │   ├── shard_r00_000.safetensors  (rank 0, shard 0)
    │   ├── shard_r00_001.safetensors  (rank 0, shard 1)
    │   ├── shard_r01_000.safetensors  (rank 1, shard 0)
    │   └── ...

Each safetensors file contains:
    latents:   [N, C, H, W] float32  -- VA-VAE latents
    has_tumor: [N]          int8     -- 0=no tumor, 1=has tumor

Why rank in filename:
    With 8 GPUs, ranks 0-7 run simultaneously and each write their own shards.
    Including rank in the name (shard_r00_*, shard_r01_*, ...) prevents
    write collisions. ShardedLatentDataset discovers all via glob('shard*.safetensors').

Usage (single GPU):
    python preprocessing/extract_latents_brats.py \\
        --data_path ../data/brats_images/train \\
        --output_path ../data/latents/brats \\
        --config tokenizer/configs/vavae-imagenet256-f16d32-dinov2.yaml

Usage (8 GPUs with torchrun):
    torchrun --nproc_per_node=8 preprocessing/extract_latents_brats.py \\
        --data_path ../data/brats_images/train \\
        --output_path ../data/latents/brats \\
        --config tokenizer/configs/vavae-imagenet256-f16d32-dinov2.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from tokenizer.vavae import VA_VAE


def _save_shard(latents_buf, has_tumor_buf, shards_dir, rank, shard_idx):
    """Concatenate buffers and write one safetensors shard to disk."""
    latents = torch.cat(latents_buf, dim=0)      # [N, C, H, W]
    has_tumor = torch.cat(has_tumor_buf, dim=0)  # [N] int64 from ImageFolder
    save_dict = {
        'latents': latents,
        'has_tumor': has_tumor.to(torch.int8),
    }
    save_filename = os.path.join(shards_dir, f'shard_r{rank:02d}_{shard_idx:03d}.safetensors')
    save_file(
        save_dict,
        save_filename,
        metadata={
            'total_size': str(latents.shape[0]),
            'dtype': str(latents.dtype),
        }
    )
    print(f'Saved {save_filename}  ({latents.shape[0]} images)')


def main(args):
    """
    Encode BraTS PNG images to VA-VAE latents and save as sharded safetensors.
    No flip augmentation -- plain image to latent, one pass.
    """
    assert torch.cuda.is_available(), "Latent extraction requires at least one GPU."

    # Setup DDP only under torchrun
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    else:
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup output directories
    shards_dir = os.path.join(args.output_path, 'shards')
    if rank == 0:
        os.makedirs(shards_dir, exist_ok=True)
    # Let all ranks wait until rank 0 has created the directory
    if world_size > 1:
        dist.barrier()

    # Load tokenizer
    tokenizer = VA_VAE(args.config)

    # Single dataset -- no horizontal flip (plain image to latent)
    dataset = ImageFolder(
        args.data_path,
        transform=tokenizer.img_transform(p_hflip=0.0)
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total_data = len(loader.dataset)
    if rank == 0:
        print(f"Total images: {total_data:,}  (this rank sees: {len(loader.sampler):,})")

    run_images = 0
    saved_files = 0
    latents_buf = []
    has_tumor_buf = []

    for batch_idx, (x, y) in enumerate(loader):
        run_images += x.shape[0]
        if rank == 0 and run_images % 1000 == 0:
            print(f'{datetime.now()} processing {run_images} of {len(loader.sampler):,} images (rank 0)')

        with torch.no_grad():
            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

        if batch_idx == 0 and rank == 0:
            print(f'latent shape {z.shape}  dtype {z.dtype}')

        latents_buf.append(z)
        has_tumor_buf.append(y)  # ImageFolder class index: 0=no_tumor, 1=has_tumor

        # Save a shard every ~10k images
        if len(latents_buf) == 10000 // args.batch_size:
            _save_shard(latents_buf, has_tumor_buf, shards_dir, rank, saved_files)
            latents_buf = []
            has_tumor_buf = []
            saved_files += 1

    # Save the remaining images (last partial shard)
    if len(latents_buf) > 0:
        _save_shard(latents_buf, has_tumor_buf, shards_dir, rank, saved_files)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract BraTS latents with VA-VAE tokenizer (no flip, plain image to latent)"
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="ImageFolder root (e.g. ../data/brats_images/train)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output root (shards/ will be created here)")
    parser.add_argument("--config", type=str, default="tokenizer/configs/vavae-imagenet256-f16d32-dinov2.yaml",
                        help="VA-VAE tokenizer config YAML")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size per GPU (default: 20)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers (default: 8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
