"""
Extract BraTS multichannel slices as sharded safetensors for JiT-L training.

Channel layout (single shard, fields chosen at training time):
    slices:    [N, 4, 256, 256] fp16  in [-1, 1]
                channels = [t1n, t1c, t2w, t2f]
    seg_mask:  [N, 1, 256, 256] fp16  in {-1, -0.5, 0, 0.5, 1}
                via (seg/4)*2 - 1  for BraTS-2024 5-label {0=bg, 1=NETC, 2=SNFH, 3=ET, 4=RC}
                background (label 0) maps to -1, matching MRI background after [-1,1] norm
    subject_id: [N] int32 (offset-encoded)
    slice_idx:  [N] int32
    has_tumor:  [N] uint8 (1 = has tumor in this slice, 0 = no tumor)

Spatial resolution:
    BraTSDataset outputs 192x192 native. We bicubic-upsample 192 -> 256
    at extraction time so training reads 256x256 ready (matches JiT recipe).

Validation split: no seg available -> seg_mask channel is all-(-1) (background label).

Output sharded safetensors (mirrors the LDiT BraTS latent shard schema):
    output_dir/
    ├── train_shards/
    │   ├── shard_00000.safetensors   # 10000 samples per shard
    │   ├── shard_00001.safetensors
    │   └── ...
    └── extraction_metadata.json

Usage:
    python preprocessing/extract_brats_multichannel_safetensors.py \\
        --train-dirs data/brats24/BraTS-GLI/training_data1_v2 \\
                     data/brats24/BraTS-GLI/training_data_additional \\
        --output-dir data/brats24_gli_multichannel_256 \\
        --num-workers 8

Memory: Processes one patient at a time + buffers up to 10k slices before flush.
Time: ~3-4 hours for full BraTS-GLI (~1500 patients, ~211k multichannel slices).
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json

from safetensors.torch import save_file as safetensors_save

from brain_dataset import BraTSDataset


# BraTS 2024 GLI segmentation labels: {0, 1, 2, 3, 4}
#   0 = background, 1 = NETC, 2 = SNFH, 3 = ET, 4 = RC
# Normalization scheme (matches MRI [-1, 1] convention; bg -> -1 like MRI bg):
#   (seg / 4) * 2 - 1  -> {-1.0, -0.5, 0.0, 0.5, 1.0}
SEG_LABEL_MAX = 4.0


def encode_mask_to_norm(seg_tensor):
    """Map BraTS seg {0,1,2,3,4} -> [-1, 1] via (seg/4)*2-1. Returns fp32."""
    return (seg_tensor.float() / SEG_LABEL_MAX) * 2.0 - 1.0


def upsample_192_to_256(t, mode='bicubic'):
    """Upsample a [..., 192, 192] tensor to [..., 256, 256].

    Handles 2D [H, W], 3D [C, H, W], and 4D [N, C, H, W] inputs. Pass
    mode='bicubic' for continuous-intensity images (MRI modalities) and
    mode='nearest' for discrete-label images (segmentation masks) — bicubic
    on a normalized-but-discrete mask produces smoothed boundary values and
    can overshoot the [-1, 1] range (verified empirically), corrupting the
    discrete tissue-class structure.
    """
    if t.dim() == 2:                            # [H, W]
        t4d = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:                          # [C, H, W]
        t4d = t.unsqueeze(0)
    elif t.dim() == 4:                          # [N, C, H, W]
        t4d = t
    else:
        raise ValueError(f"upsample_192_to_256 got rank {t.dim()}")
    kwargs = {'size': (256, 256), 'mode': mode}
    if mode == 'bicubic':
        kwargs['align_corners'] = False
    out = F.interpolate(t4d.float(), **kwargs)
    if t.dim() == 2:
        return out[0, 0]
    if t.dim() == 3:
        return out[0]
    return out


def flush_shard(buffer, shard_path):
    """Stack a buffer of per-slice dicts into a single safetensors shard and write to disk."""
    if not buffer:
        return
    shard_dict = {
        'slices':     torch.stack([b['slices']    for b in buffer]).contiguous().half(),   # [N, 4, 256, 256]
        'seg_mask':   torch.stack([b['seg_mask']  for b in buffer]).contiguous().half(),   # [N, 1, 256, 256]
        'subject_id': torch.tensor([b['subject_id'] for b in buffer], dtype=torch.int32),
        'slice_idx':  torch.tensor([b['slice_idx']  for b in buffer], dtype=torch.int32),
        'has_tumor':  torch.tensor([b['has_tumor']  for b in buffer], dtype=torch.uint8),
    }
    safetensors_save(shard_dict, str(shard_path))


def extract_dataset_to_shards(
    data_dir,
    shard_out_dir,
    split_name,
    num_workers=4,
    seed=42,
    is_validation=False,
    patient_offset=0,
    shard_start_idx=0,
    samples_per_shard=10000,
):
    """
    Extract a BraTS dataset split as 5-channel multichannel slices into sharded safetensors.

    Args:
        data_dir: Path to BraTS data directory (e.g. training_data1_v2/)
        shard_out_dir: Directory for shard_NNNNN.safetensors files (created)
        split_name: 'train' / 'val' for logging
        num_workers: DataLoader workers
        seed: Random seed
        is_validation: If True, no seg available -> seg_mask channel is all -1 (background)
        patient_offset: Starting subject_id offset (to keep IDs unique across multiple dirs)
        shard_start_idx: First shard filename index (for resuming or multi-dir runs)
        samples_per_shard: Slices per safetensors shard (default 10000, matches LDiT convention)

    Returns:
        (stats, next_shard_idx, next_patient_offset)
    """
    print(f"\n{'='*80}")
    print(f"Extracting {split_name.upper()} from: {data_dir}")
    print(f"{'='*80}")

    shard_out_dir = Path(shard_out_dir)
    shard_out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BraTSDataset with monai transforms...")
    dataset = BraTSDataset(
        data_dir=data_dir,
        modalities=['t1n', 't1c', 't2w', 't2f'],
        ref_modality='t1c',
        slice_range=[15, 155],
        image_size=[192, 192, 182],
        crop_size=[192, 192, 182],
        pad_size=[218, 218, 182],
        rotate_90=True,
        return_tumor=(not is_validation),
        repeat_channels=False,
        return_all_slices=True,
        seed=seed,
    )
    print(f"  loaded {len(dataset)} patients")

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )

    modalities = ['t1n', 't1c', 't2w', 't2f']
    stats = {'total_slices': 0, 'no_tumor': 0, 'has_tumor': 0, 'num_patients': 0}

    buffer = []                      # list of per-slice dicts
    next_shard_idx = shard_start_idx

    def maybe_flush_shard():
        nonlocal next_shard_idx
        while len(buffer) >= samples_per_shard:
            chunk, buffer[:] = buffer[:samples_per_shard], buffer[samples_per_shard:]
            shard_path = shard_out_dir / f"shard_{next_shard_idx:05d}.safetensors"
            flush_shard(chunk, shard_path)
            next_shard_idx += 1

    for local_patient_idx, batch in enumerate(tqdm(dataloader, desc=f"  {split_name}")):
        # Modality tensors: each is [1, num_slices, 1, H, W] in [-1, 1]
        # Stack channels first; we'll go slice-by-slice next.
        mod_stack = torch.stack([batch[m].squeeze(0).squeeze(1) for m in modalities], dim=1)
        # mod_stack: [num_slices, 4, H, W] at 192x192, in [-1, 1]
        num_slices = mod_stack.shape[0]

        # has_tumor labels (per slice, training only)
        if not is_validation and 'has_tumor' in batch:
            ht_all = batch['has_tumor'].squeeze(0).to(torch.uint8)   # [num_slices]
        else:
            ht_all = torch.zeros(num_slices, dtype=torch.uint8)

        # Segmentation: [1, num_slices, H, W] integer {0..4}; absent for validation
        if not is_validation and 'seg' in batch:
            seg_all = batch['seg'].squeeze(0)                        # [num_slices, H, W]
            seg_norm = encode_mask_to_norm(seg_all).unsqueeze(1)     # [num_slices, 1, H, W]
        else:
            seg_norm = torch.full((num_slices, 1, mod_stack.shape[-2], mod_stack.shape[-1]),
                                   -1.0, dtype=torch.float32)         # bg = -1, matching MRI bg

        # Upsample 192 -> 256, then split into per-slice rows for the shard buffer.
        # MRI gets bicubic (continuous intensities); SEG gets nearest (discrete
        # labels — bicubic on a discrete mask smears boundaries and overshoots
        # [-1, 1], corrupting the {-1, -0.5, 0, 0.5, 1} structure).
        # Doing it once per patient is cheaper than once per slice.
        mod_256 = upsample_192_to_256(mod_stack, mode='bicubic')       # [num_slices, 4, 256, 256]
        seg_256 = upsample_192_to_256(seg_norm, mode='nearest')        # [num_slices, 1, 256, 256]

        subj_id = patient_offset + local_patient_idx
        for s in range(num_slices):
            ht = int(ht_all[s].item())
            buffer.append({
                'slices':     mod_256[s].clone(),    # [4, 256, 256] fp32
                'seg_mask':   seg_256[s].clone(),    # [1, 256, 256] fp32
                'subject_id': subj_id,
                'slice_idx':  int(s),
                'has_tumor':  ht,
            })
            stats['total_slices'] += 1
            stats['has_tumor']    += ht
            stats['no_tumor']     += (1 - ht)

        # Flush whole shards as we cross the threshold (keeps RAM bounded).
        maybe_flush_shard()

    # Flush any leftover slices as a final, possibly-partial shard.
    if buffer:
        shard_path = shard_out_dir / f"shard_{next_shard_idx:05d}.safetensors"
        flush_shard(buffer, shard_path)
        next_shard_idx += 1
        buffer.clear()

    stats['num_patients'] = len(dataset)
    next_patient_offset = patient_offset + len(dataset)
    return stats, next_shard_idx, next_patient_offset


def main():
    parser = argparse.ArgumentParser(
        description="Extract BraTS 4-modality + seg as sharded safetensors for JiT multichannel training"
    )

    parser.add_argument(
        '--train-dirs', nargs='+',
        default=[
            'data/brats24/BraTS-GLI/training_data1_v2',
            'data/brats24/BraTS-GLI/training_data_additional',
        ],
        help='Paths to BraTS training directories (concatenated; offsets handled)'
    )
    parser.add_argument(
        '--val-dir', type=str,
        default='data/brats24/BraTS-GLI/validation_data',
        help='Path to BraTS validation directory (no seg available -> seg_mask all = -1)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='data/brats24_gli_multichannel_256',
        help='Sharded safetensors output directory'
    )
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers (default 8)')
    parser.add_argument('--samples-per-shard', type=int, default=10000,
                        help='Slices per safetensors shard (default 10000, matches LDiT BraTS schema)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-val', action='store_true',
                        help='Skip validation_data extraction (training shards only)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_shard_dir = output_dir / 'train_shards'
    val_shard_dir = output_dir / 'val_shards'

    print("=" * 80)
    print("BraTS 4-modality + seg multichannel extraction (sharded safetensors)")
    print("=" * 80)
    print(f"Train dirs        : {len(args.train_dirs)}")
    for d in args.train_dirs:
        print(f"  - {d}")
    print(f"Val dir           : {args.val_dir}")
    print(f"Output dir        : {args.output_dir}")
    print(f"Samples/shard     : {args.samples_per_shard}")
    print(f"DataLoader workers: {args.num_workers}")
    print("=" * 80)

    train_stats_total = {'total_slices': 0, 'no_tumor': 0, 'has_tumor': 0, 'num_patients': 0}
    next_shard_idx = 0
    patient_offset = 0

    for train_data_dir in args.train_dirs:
        if not Path(train_data_dir).exists():
            print(f"WARNING: {train_data_dir} does not exist, skipping")
            continue
        stats, next_shard_idx, patient_offset = extract_dataset_to_shards(
            data_dir=train_data_dir,
            shard_out_dir=train_shard_dir,
            split_name='train',
            num_workers=args.num_workers,
            seed=args.seed,
            is_validation=False,
            patient_offset=patient_offset,
            shard_start_idx=next_shard_idx,
            samples_per_shard=args.samples_per_shard,
        )
        for k in ('total_slices', 'no_tumor', 'has_tumor', 'num_patients'):
            train_stats_total[k] += stats[k]

    val_stats = {'total_slices': 0, 'no_tumor': 0, 'has_tumor': 0, 'num_patients': 0}
    if not args.skip_val and Path(args.val_dir).exists():
        val_stats, _, _ = extract_dataset_to_shards(
            data_dir=args.val_dir,
            shard_out_dir=val_shard_dir,
            split_name='val',
            num_workers=args.num_workers,
            seed=args.seed,
            is_validation=True,
            patient_offset=0,
            shard_start_idx=0,
            samples_per_shard=args.samples_per_shard,
        )

    # ------- summary -------
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Training: {train_stats_total['num_patients']} patients -> "
          f"{train_stats_total['total_slices']:,} slices "
          f"(no_tumor {train_stats_total['no_tumor']:,} / has_tumor {train_stats_total['has_tumor']:,})")
    if not args.skip_val:
        print(f"Validation: {val_stats['num_patients']} patients -> {val_stats['total_slices']:,} slices "
              f"(seg_mask = -1, no labels)")
    print(f"Train shards saved to: {train_shard_dir}")
    if not args.skip_val:
        print(f"Val shards saved to: {val_shard_dir}")

    metadata = {
        'description': 'BraTS-GLI 4-modality + seg multichannel slices, sharded safetensors',
        'modalities': ['t1n', 't1c', 't2w', 't2f'],
        'spatial_size': [256, 256],
        'upsample': 'bicubic 192->256 at extraction time (matches LDiT center_crop_arr precedent)',
        'mri_normalization': 'BraTSDataset percentile_clip(1,99) -> [0,1] -> MonaiNormalize(0.5,0.5) -> [-1,1]; bg=0 -> -1',
        'mri_upsample_mode': 'bicubic 192->256 (continuous intensities)',
        'mask_normalization': '(seg/4)*2-1 for BraTS-2024 labels {0,1,2,3,4} -> {-1,-0.5,0,0.5,1}; bg=0 -> -1 (matches MRI bg)',
        'mask_upsample_mode': 'nearest 192->256 (discrete labels; bicubic would smear them)',
        'shard_fields': {
            'slices':     '[N, 4, 256, 256] fp16 (channel order: t1n, t1c, t2w, t2f)',
            'seg_mask':   '[N, 1, 256, 256] fp16 (val shards: all -1)',
            'subject_id': '[N] int32 (offset-encoded across train_dirs)',
            'slice_idx':  '[N] int32',
            'has_tumor':  '[N] uint8 (val shards: all 0)',
        },
        'train_dirs': args.train_dirs,
        'val_dir': args.val_dir,
        'output_dir': str(output_dir),
        'seed': args.seed,
        'samples_per_shard': args.samples_per_shard,
        'train_stats': train_stats_total,
        'val_stats': val_stats,
    }
    metadata_path = output_dir / 'extraction_metadata.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata: {metadata_path}")
    print(f"\nPoint JiT --brats_shard_dir at {train_shard_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
