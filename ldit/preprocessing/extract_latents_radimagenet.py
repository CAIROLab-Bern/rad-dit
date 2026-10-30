"""
Extract RadImageNet image latents using VA-VAE tokenizer and save as sharded safetensors.

Reads the 3-level hierarchy: root/modality/bodypart/pathology/*.png
Region label assigned from the modality/bodypart combination:
    CT/abd       → 0
    CT/lung      → 1
    MR/hip       → 2
    MR/knee      → 3
    MR/shoulder  → 4

Outputs:
    output_path/
    └── shards/
    │   ├── shard_r00_000.safetensors  (rank 0, shard 0)
    │   ├── shard_r00_001.safetensors  (rank 0, shard 1)
    │   ├── shard_r01_000.safetensors  (rank 1, shard 0)
    │   └── ...

Each safetensors file contains:
    latents:      [N, C, H, W] float32  -- VA-VAE latents
    region_label: [N]          int8     -- 0=CT/abd, 1=CT/lung, 2=MR/hip, 3=MR/knee, 4=MR/shoulder

Usage (single GPU):
    python preprocessing/extract_latents_radimagenet.py \\
        --data_path ../data/radimagenet \\
        --output_path ../data/latents/radin/train \\
        --config tokenizer/configs/vavae-imagenet256-f16d32-dinov2.yaml

Usage (8 GPUs with torchrun):
    torchrun --nproc_per_node=8 preprocessing/extract_latents_radimagenet.py \\
        --data_path ../data/radimagenet \\
        --output_path ../data/latents/radin/train \\
        --config tokenizer/configs/vavae-imagenet256-f16d32-dinov2.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from PIL import Image
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from tokenizer.vavae import VA_VAE


# Maps (modality, bodypart) → region_label int (0-4)
REGION_LABEL_MAP = {
    ('CT', 'abd'):      0,
    ('CT', 'lung'):     1,
    ('MR', 'hip'):      2,
    ('MR', 'knee'):     3,
    ('MR', 'shoulder'): 4,
}


class RadImageNetDataset(Dataset):
    """
    Flat-view of RadImageNet 3-level hierarchy: root/modality/bodypart/pathology/*.png
    Returns (image_tensor, region_label) where region_label is int 0-4.
    """

    def __init__(self, root: str, transform=None, exclude_paths=None):
        self.transform = transform
        self.samples = []  # list of (path, region_label)
        excluded = set()
        if exclude_paths:
            with open(exclude_paths) as f:
                # Compare <modality>/<bodypart>/<pathology>/<file>, not the data root.
                excluded = {'/'.join(Path(line.strip()).parts[-4:]) for line in f if line.strip()}

        root = Path(root)
        for modality_dir in sorted(root.iterdir()):
            if not modality_dir.is_dir():
                continue
            modality = modality_dir.name
            for bodypart_dir in sorted(modality_dir.iterdir()):
                if not bodypart_dir.is_dir():
                    continue
                bodypart = bodypart_dir.name
                key = (modality, bodypart)
                if key not in REGION_LABEL_MAP:
                    print(f"WARNING: Unknown region {modality}/{bodypart} — skipping")
                    continue
                region_label = REGION_LABEL_MAP[key]
                for pathology_dir in sorted(bodypart_dir.iterdir()):
                    if not pathology_dir.is_dir():
                        continue
                    for img_path in sorted(pathology_dir.glob('*.png')):
                        if '/'.join(img_path.parts[-4:]) not in excluded:
                            self.samples.append((img_path, region_label))

        print(f"RadImageNetDataset: {len(self.samples):,} images in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, region_label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, region_label


def _save_shard(latents_buf, region_label_buf, shards_dir, rank, shard_idx):
    """Concatenate buffers and write one safetensors shard to disk."""
    latents = torch.cat(latents_buf, dim=0)
    region_labels = torch.cat(region_label_buf, dim=0)
    save_dict = {
        'latents': latents,
        'region_label': region_labels.to(torch.int8),
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
    Encode RadImageNet PNG images to VA-VAE latents and save as sharded safetensors.
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

    # RadImageNet 3-level hierarchy dataset, no horizontal flip
    dataset = RadImageNetDataset(args.data_path, transform=tokenizer.img_transform(p_hflip=0.0),
                                 exclude_paths=args.exclude_paths)

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
    region_label_buf = []

    for batch_idx, (x, y) in enumerate(loader):
        run_images += x.shape[0]
        if rank == 0 and run_images % 5000 == 0:
            print(f'{datetime.now()} processing {run_images} of {len(loader.sampler):,} images (rank 0)')

        with torch.no_grad():
            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

        if batch_idx == 0 and rank == 0:
            print(f'latent shape {z.shape}  dtype {z.dtype}')
            region_names = {v: f'{k[0]}/{k[1]}' for k, v in REGION_LABEL_MAP.items()}
            for u in y.unique().tolist():
                print(f'  region {u} ({region_names[u]}): {(y==u).sum().item()} in first batch')

        latents_buf.append(z)
        region_label_buf.append(y)

        # Save a shard every ~10k images
        if len(latents_buf) == 10000 // args.batch_size:
            _save_shard(latents_buf, region_label_buf, shards_dir, rank, saved_files)
            latents_buf = []
            region_label_buf = []
            saved_files += 1

    # Save the remaining images (last partial shard)
    if len(latents_buf) > 0:
        _save_shard(latents_buf, region_label_buf, shards_dir, rank, saved_files)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RadImageNet latents with VA-VAE tokenizer (no flip, plain image to latent)"
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="RadImageNet root (e.g. ../data/radimagenet)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output root (shards/ will be created here)")
    parser.add_argument("--config", type=str, default="tokenizer/configs/vavae-imagenet256-f16d32-dinov2.yaml",
                        help="VA-VAE tokenizer config YAML")
    parser.add_argument("--exclude_paths", type=str, default=None,
                        help="Optional text file with validation images to skip, one path per line")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size per GPU (default: 20)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers (default: 8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
