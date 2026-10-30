"""
Extract BraTS images as PNG folders for VA-VAE latent extraction.
Reuses BraTSDataset with exact same transforms as feature extraction.

Output structure (ImageNet-like):
    output_dir/
    ├── train/
    │   ├── 0/       # No tumor
    │   └── 1/       # Has tumor
    └── val/
        ├── 0/
        └── 1/

Usage:
    python preprocessing/extract_brats_images.py \\
        --train-dirs data/brats24/BraTS-GLI/training_data1_v2 \\
                     data/brats24/BraTS-GLI/training_data_additional \\
        --val-dir data/brats24/BraTS-GLI/validation_data \\
        --output-dir data/brats_images \\
        --num-workers 8

Memory: Processes one patient at a time, very memory efficient
Time: ~2-3 hours for full BraTS dataset
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
import json

from brain_dataset import BraTSDataset


def save_image_worker(img_data, output_dir):
    """Worker function to save a single image (for parallel execution)"""
    img_np, patient_idx, slice_idx, modality, has_tumor = img_data

    # Determine class label: 0=no_tumor, 1=has_tumor
    class_label = '1' if has_tumor else '0'
    class_dir = output_dir / class_label

    # Create filename: patientID_sliceIDX_modality.png
    img_path = class_dir / f"{patient_idx:04d}_{slice_idx:03d}_{modality}.png"

    # Save PNG
    Image.fromarray(img_np).save(img_path)
    return str(img_path)


def extract_images_from_dataset(
    data_dir,
    output_dir,
    split_name,
    num_workers=4,
    num_save_workers=8,
    seed=42,
    is_validation=False,
    patient_offset=0,
):
    """
    Extract all images from a BraTS dataset split.

    Args:
        data_dir: Path to BraTS data directory
        output_dir: Output directory for this split (train or val)
        split_name: 'train' or 'val' for logging
        num_workers: Dataloader workers
        num_save_workers: Parallel PNG save workers
        seed: Random seed
        is_validation: If True, validation split has no labels, save all to class '0'
        patient_offset: Starting patient index (for multi-dir extraction to avoid filename collisions)

    Returns:
        stats: Dictionary with extraction statistics
    """
    print(f"\n{'='*80}")
    print(f"Extracting {split_name.upper()} split from: {data_dir}")
    print(f"{'='*80}")

    # Create class directories
    if is_validation:
        # Validation has no labels, save all to class 0
        (output_dir / '0').mkdir(parents=True, exist_ok=True)
        print("⚠️  Validation split has no tumor labels - saving all images to class 0")
    else:
        # Training has labels, create both classes
        for class_label in ['0', '1']:
            (output_dir / class_label).mkdir(parents=True, exist_ok=True)

    # Setup dataset
    print("Loading BraTS dataset with exact VAE extraction transforms...")
    dataset = BraTSDataset(
        data_dir=data_dir,
        modalities=['t1n', 't1c', 't2w', 't2f'],
        ref_modality='t1c',
        slice_range=[15, 155],  # Only middle slices with content
        image_size=[192, 192, 182],
        crop_size=[192, 192, 182],
        pad_size=[218, 218, 182],
        rotate_90=True,  # Same as VAE extraction
        return_tumor=True if not is_validation else False,  # Validation has no tumor labels
        repeat_channels=False,
        return_all_slices=True,  # Get all slices per patient
        seed=seed,
    )

    print(f"✓ Loaded dataset with {len(dataset)} patients")

    # Create dataloader (batch_size=1 required for return_all_slices=True)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Statistics
    stats = {
        'total_images': 0,
        'no_tumor': 0,
        'has_tumor': 0,
        'by_modality': {'t1n': 0, 't1c': 0, 't2w': 0, 't2f': 0},
    }

    modalities = ['t1n', 't1c', 't2w', 't2f']

    # Extract images with parallel I/O
    print(f"\nExtracting all images from {split_name} split...")
    with ThreadPoolExecutor(max_workers=num_save_workers) as executor:
        save_futures = []
        flush_interval = 50  # Flush every 50 patients to prevent memory buildup

        for patient_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {split_name} patients")):
            # Get tumor labels for this patient's slices (if available)
            if is_validation or 'has_tumor' not in batch:
                # Validation has no labels - treat all as class 0
                # Infer num_slices from modality data
                first_modality = modalities[0]
                num_slices = batch[first_modality].squeeze(0).shape[0]
                has_tumor_labels = None
            else:
                # Training has labels
                # Shape: [1, num_slices]
                has_tumor_labels = batch['has_tumor'].squeeze(0)  # [num_slices]
                num_slices = has_tumor_labels.shape[0]

            # Extract all (slice, modality) combinations for this patient
            for slice_idx in range(num_slices):
                # Get tumor status for this slice
                if has_tumor_labels is not None:
                    has_tumor = bool(has_tumor_labels[slice_idx].item())
                else:
                    has_tumor = False  # Validation - no labels

                for modality in modalities:
                    # Get the specific modality for this slice
                    images = batch[modality].squeeze(0)  # [num_slices, H, W]
                    img_tensor = images[slice_idx].squeeze()  # [H, W]
                    img_np = img_tensor.cpu().numpy()

                    # Denormalize from [-1, 1] to [0, 255]
                    # Using unified method: 127.5 * x + 128.0 (same as SiT, VA-VAE)
                    img_np = np.clip(127.5 * img_np + 128.0, 0, 255).astype(np.uint8)

                    # Convert grayscale to RGB (VA-VAE expects RGB)
                    img_np = np.stack([img_np, img_np, img_np], axis=-1)  # [H, W, 3]

                    # Submit save task to thread pool
                    img_data = (img_np, patient_offset + patient_idx, slice_idx, modality, has_tumor)
                    future = executor.submit(save_image_worker, img_data, output_dir)
                    save_futures.append(future)

                    # Update stats
                    stats['total_images'] += 1
                    if has_tumor:
                        stats['has_tumor'] += 1
                    else:
                        stats['no_tumor'] += 1
                    stats['by_modality'][modality] += 1

            # Periodically flush completed futures to free memory
            if (patient_idx + 1) % flush_interval == 0:
                # Wait for completed futures
                completed = [f for f in save_futures if f.done()]
                for f in completed:
                    f.result()  # Retrieve result to free memory
                # Keep only pending futures
                save_futures = [f for f in save_futures if not f.done()]

        # Wait for all remaining save operations to complete
        print(f"\nWaiting for all save operations to complete...")
        for future in tqdm(save_futures, desc="Finalizing saves"):
            future.result()

    stats['num_patients'] = len(dataset)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract BraTS images for VA-VAE latent extraction (ImageNet-like structure)"
    )

    parser.add_argument(
        '--train-dirs',
        nargs='+',
        default=[
            'data/brats24/BraTS-GLI/training_data1_v2',
            'data/brats24/BraTS-GLI/training_data_additional',
        ],
        help='Paths to BraTS training directories'
    )

    parser.add_argument(
        '--val-dir',
        type=str,
        default='data/brats24/BraTS-GLI/validation_data',
        help='Path to BraTS validation directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for extracted images'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of dataloader workers (default: 8)'
    )

    parser.add_argument(
        '--num-save-workers',
        type=int,
        default=8,
        help='Number of parallel workers for PNG saving (default: 8)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'

    print("="*80)
    print("BraTS Image Extraction")
    print("="*80)
    print(f"Training directories: {len(args.train_dirs)}")
    for d in args.train_dirs:
        print(f"  - {d}")
    print(f"Validation directory: {args.val_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.num_workers} dataloader, {args.num_save_workers} save")
    print("="*80)

    # Check if training already extracted
    train_exists = (train_dir / '0').exists() and (train_dir / '1').exists()
    if train_exists:
        existing_train_count = len(list((train_dir / '0').glob('*.png'))) + len(list((train_dir / '1').glob('*.png')))
        if existing_train_count > 0:
            if not (train_dir / '.complete').exists():
                raise RuntimeError(f"{train_dir} holds a partial extraction. Delete it and run again.")
            print(f"\n⚠️  Training images already exist ({existing_train_count:,} images)")
            print(f"Skipping training extraction. Delete {train_dir} to re-extract.\n")
            train_stats = {
                'total_images': existing_train_count,
                'no_tumor': len(list((train_dir / '0').glob('*.png'))),
                'has_tumor': len(list((train_dir / '1').glob('*.png'))),
                'by_modality': {},
            }
        else:
            train_exists = False

    if not train_exists:
        # Extract training images (combine all training directories)
        train_stats = {
            'total_images': 0,
            'no_tumor': 0,
            'has_tumor': 0,
            'by_modality': {'t1n': 0, 't1c': 0, 't2w': 0, 't2f': 0},
        }

        patient_offset = 0
        for train_data_dir in args.train_dirs:
            if not Path(train_data_dir).exists():
                print(f"⚠️  Warning: {train_data_dir} does not exist, skipping")
                continue

            stats = extract_images_from_dataset(
                data_dir=train_data_dir,
                output_dir=train_dir,
                split_name='train',
                num_workers=args.num_workers,
                num_save_workers=args.num_save_workers,
                seed=args.seed,
                is_validation=False,
                patient_offset=patient_offset,
            )

            # Accumulate stats and advance offset so next dir starts after this dir's patients
            for key in ['total_images', 'no_tumor', 'has_tumor']:
                train_stats[key] += stats[key]
            for modality in ['t1n', 't1c', 't2w', 't2f']:
                train_stats['by_modality'][modality] += stats['by_modality'][modality]
            patient_offset += stats['num_patients']

    if not train_exists:
        (train_dir / '.complete').touch()

    # Check if validation already extracted
    val_exists = (val_dir / '0').exists()
    if val_exists:
        existing_val_count = len(list((val_dir / '0').glob('*.png')))
        if existing_val_count > 0:
            if not (val_dir / '.complete').exists():
                raise RuntimeError(f"{val_dir} holds a partial extraction. Delete it and run again.")
            print(f"\n⚠️  Validation images already exist ({existing_val_count:,} images)")
            print(f"Skipping validation extraction. Delete {val_dir} to re-extract.\n")
            val_stats = {
                'total_images': existing_val_count,
                'no_tumor': existing_val_count,
                'has_tumor': 0,
                'by_modality': {},
            }
        else:
            val_exists = False

    if not val_exists:
        # Extract validation images
        if Path(args.val_dir).exists():
            val_stats = extract_images_from_dataset(
                data_dir=args.val_dir,
                output_dir=val_dir,
                split_name='val',
                num_workers=args.num_workers,
                num_save_workers=args.num_save_workers,
                seed=args.seed,
                is_validation=True,  # Validation has no labels
            )
        else:
            print(f"⚠️  Warning: Validation directory {args.val_dir} does not exist, skipping")
            val_stats = {'total_images': 0, 'no_tumor': 0, 'has_tumor': 0, 'by_modality': {}}

    if not val_exists:
        val_dir.mkdir(parents=True, exist_ok=True)
        (val_dir / '.complete').touch()

    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)

    print("\n📊 Training Set Statistics:")
    print(f"  Total images:     {train_stats['total_images']:,}")
    print(f"  No tumor (class 0): {train_stats['no_tumor']:,} ({train_stats['no_tumor']/train_stats['total_images']*100:.1f}%)")
    print(f"  Has tumor (class 1): {train_stats['has_tumor']:,} ({train_stats['has_tumor']/train_stats['total_images']*100:.1f}%)")
    print(f"  By modality:")
    for mod, count in train_stats['by_modality'].items():
        print(f"    {mod}: {count:,}")

    print("\n📊 Validation Set Statistics:")
    print(f"  Total images:     {val_stats['total_images']:,}")
    if val_stats['total_images'] > 0:
        print(f"  No tumor (class 0): {val_stats['no_tumor']:,} ({val_stats['no_tumor']/val_stats['total_images']*100:.1f}%)")
        print(f"  Has tumor (class 1): {val_stats['has_tumor']:,} ({val_stats['has_tumor']/val_stats['total_images']*100:.1f}%)")
        print(f"  By modality:")
        for mod, count in val_stats['by_modality'].items():
            print(f"    {mod}: {count:,}")

    # Save metadata
    metadata = {
        'train_dirs': args.train_dirs,
        'val_dir': args.val_dir,
        'output_dir': str(args.output_dir),
        'seed': args.seed,
        'train_stats': train_stats,
        'val_stats': val_stats,
    }

    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📁 Output structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── 0/  (no tumor: {train_stats['no_tumor']:,} images)")
    print(f"    │   └── 1/  (has tumor: {train_stats['has_tumor']:,} images)")
    print(f"    ├── val/")
    print(f"    │   ├── 0/  (no tumor: {val_stats['no_tumor']:,} images)")
    print(f"    │   └── 1/  (has tumor: {val_stats['has_tumor']:,} images)")
    print(f"    └── extraction_metadata.json")

    print(f"\nNext step: run ldit/preprocessing/extract_latents_brats.py on {output_dir}/train")
    print("="*80)


if __name__ == '__main__':
    main()
