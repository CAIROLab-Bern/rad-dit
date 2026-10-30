"""
Generic Brain MRI Dataset
Base class for BraTS and other brain MRI datasets
"""

import torch
import torchvision
import numpy as np
import os
import random
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from abc import ABC, abstractmethod

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    ToTensord,
    CenterSpatialCropd,
    SpatialPadd,
    Rotate90d,
)

from monai.transforms import MapTransform, Transform


class MonaiNormalized(Transform):
    """Dictionary wrapper for torchvision Normalize"""
    def __init__(self, keys, mean, std):
        self.keys = keys
        self.normalizer = torchvision.transforms.Normalize(mean, std)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = self.normalizer(d[key])
        return d


class ReplaceValueInSegMaskd(MapTransform):
    """Replace specific value in segmentation masks (e.g., -1 padding → 0)."""

    def __init__(self, keys, old_value=-1, new_value=0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.old_value = old_value
        self.new_value = new_value

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and d[key] is not None:
                # Replace old_value with new_value
                d[key] = torch.where(d[key] == self.old_value, self.new_value, d[key])
        return d


# =============================================================================
# Transform Builders
# =============================================================================

def get_eval_transforms(
    modalities: List[str],
    percentile_clip: Optional[Tuple[float, float]] = (1.0, 99.0),
    crop_size: Optional[List[int]] = None,
    pad_size: Optional[List[int]] = None,
    rotate_90: bool = False,
):
    """
    Build evaluation transforms (no augmentation).

    Args:
        modalities: List of modality names
        image_size: Target 2D size (H, W) - for compatibility
        percentile_clip: Optional intensity percentile clipping (lower, upper)
        crop_size: Optional center crop size [H, W]
        pad_size: Optional padding size [H, W, D]
        rotate_90: Whether to rotate 90 degrees counterclockwise in H,W plane
    """
    keys = modalities + ['seg']
    image_keys = modalities

    # build transformations
    transforms = [
        LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
    ]
    transforms.append(Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True))

    # Add rotation if requested (for correct brain visualization)
    # k=1 means 90 degrees counterclockwise, spatial_axes=(0,1) for H,W in (H,W,D)
    if rotate_90:
        transforms.append(
            Rotate90d(keys=keys, k=1, spatial_axes=(0, 1), allow_missing_keys=True)
        )
    if percentile_clip is not None:
        transforms.append(
            ScaleIntensityRangePercentilesd(
                keys=image_keys,
                lower=percentile_clip[0],
                upper=percentile_clip[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        transforms.append(
            ScaleIntensityd(
                keys=image_keys,
                channel_wise=True,
                minv=.0,
                maxv=1.0
            )
        )
    transforms.append(
        MonaiNormalized(
            keys=image_keys,
            mean=[0.5],
            std=[0.5],
        )
    )

    # Add spatial transforms
    if pad_size is not None:
        transforms.append(
            SpatialPadd(keys=keys, spatial_size=pad_size, mode='constant', value=-1, allow_missing_keys=True)
        )
        # Replace -1 padding with 0 in seg masks for uint8 storage efficiency
        transforms.append(
            ReplaceValueInSegMaskd(keys=['seg'], old_value=-1, new_value=0, allow_missing_keys=True)
        )
    if crop_size is not None:
        transforms.append(
            CenterSpatialCropd(keys=keys, roi_size=crop_size, allow_missing_keys=True)
        )

    transforms.append(ToTensord(keys=keys, allow_missing_keys=True))

    return Compose(transforms)


# =============================================================================
# Base Dataset Class
# =============================================================================

class BrainMRIDataset(Dataset, ABC):
    """
    Generic base class for brain MRI datasets.
    
    Subclasses must implement:
        - build_patient_index()
    """
    
    def __init__(
        self,
        data_dir: str,
        modalities: List[str],
        slice_range: Tuple[int, int],
        image_size: List[int],
        ref_modality: Optional[str] = 't1c',
        transform: Optional[Compose] = None,
        crop_size: Optional[List[int]] = None,
        pad_size: Optional[List[int]] = None,
        rotate_90: bool = False,
        return_tumor: bool = True,
        repeat_channels: bool = False,
        return_all_slices: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Path to data directory
            modalities: List of modality names (e.g., ['t1n', 't1c', 't2w', 't2f', 'seg'])
            slice_range: (start, end) valid slice indices
            image_size: Image 3D size (H, W, D)
            ref_modality: Reference modality for tumor area calculation
            transform: MONAI transform (if None, uses eval transforms)
            crop_size: Optional center crop size [H, W, D]
            pad_size: Optional padding size [H, W, D]
            rotate_90: Whether to rotate 90 degrees counterclockwise
            return_tumor: Whether to calculate and return tumor statistics
            repeat_channels: Whether to repeat single channel (1,H,W) to 3 channels (3,H,W)
            return_all_slices: Whether to return all slices in slice_range (True) or random slice (False)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.modalities = modalities
        self.ref_modality = ref_modality
        self.slice_range = slice_range
        self.image_size = image_size
        self.return_tumor = return_tumor
        self.repeat_channels = repeat_channels
        self.return_all_slices = return_all_slices

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if transform is None:
            self.transform = get_eval_transforms(
                modalities=modalities,
                percentile_clip=None,
                crop_size=crop_size,
                pad_size=pad_size,
                rotate_90=rotate_90,
            )
        else:
            self.transform = transform
        
        self.patient_list = self.build_patient_index()
    
    @abstractmethod
    def build_patient_index(self) -> List[Dict[str, str]]:
        """
        Scan data directory and build patient file paths.
        
        Must return list of dicts, each containing:
            - Keys for each modality (pointing to file paths)
            - Optional 'seg' key if has_tumor=True
            - 'patient_id' key
        
        Example:
            [
                {
                    't1n': '/path/to/patient1-t1n.nii.gz',
                    't1c': '/path/to/patient1-t1c.nii.gz',
                    'seg': '/path/to/patient1-seg.nii.gz',
                    'patient_id': 'patient1',
                },
                ...
            ]
        """
        pass
    
    def __len__(self):
        return len(self.patient_list)
    
    def _get_random_slice(self, volume: torch.Tensor) -> int:
        """Pick random slice from valid range"""
        depth = volume.shape[-1]  # (C, H, W, D) or (H, W, D) 3D images
        start, end = self.slice_range
        start = max(0, start)
        end = min(depth, end)
        return random.randint(start, end - 1)

    def tumor_calculation(
            self,
            seg_3d: torch.Tensor,
            ref_modality_2d: torch.Tensor,
            slice_idx: Optional[int] = None
        ) -> Dict[str, float]:
        """
        Calculate tumor statistics from segmentation mask.

        Override this method in subclasses to customize tumor calculations.

        Args:
            seg_3d: 3D segmentation mask
            ref_modality_2d: 2D reference modality slice for calculating brain area
            slice_idx: Optional slice index to extract from seg_3d (if None, seg_3d is already 2D)

        Returns:
            Dictionary with 'has_tumor' (bool) and 'tumor_area' (float)
        """
        if slice_idx is not None:
            seg_2d = seg_3d[..., slice_idx]
        else:
            seg_2d = seg_3d

        seg_2d = (seg_2d > 0).float()  # Binarize
        tumor_area = np.count_nonzero(seg_2d)
        brain_area = np.count_nonzero(ref_modality_2d)
        rel_tumor_area = float(tumor_area / brain_area) if brain_area > 0 else 0.0

        return {
            'has_tumor': rel_tumor_area > 0,
            'tumor_area': rel_tumor_area
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load patient and return either random 2D slice or all slices.

        Returns:
            dict with modality tensors and metadata
        """
        file_paths = self.patient_list[idx]
        patient_id = file_paths['patient_id']
        data_dict = {k: v for k, v in file_paths.items() if k != 'patient_id'}
        try:
            # Apply transforms (loads and normalizes all modalities)
            volumes = self.transform(data_dict)
        except Exception as e:
            raise RuntimeError(f"Could not load {patient_id}: {data_dict}") from e

        if self.return_all_slices:
            # Return all slices in slice_range
            depth = volumes[self.modalities[0]].shape[-1]
            start, end = self.slice_range
            start = max(0, start)
            end = min(depth, end)
            slice_indices = list(range(start, end))

            result = {}
            for modality in self.modalities:
                vol_3d = volumes[modality]
                slices_2d = []

                for slice_idx in slice_indices:
                    slice_2d = vol_3d[..., slice_idx]  # (C, H, W)

                    # Repeat channels if requested: (1, H, W) -> (3, H, W)
                    if self.repeat_channels and slice_2d.shape[0] == 1:
                        slice_2d = slice_2d.repeat(3, 1, 1)

                    slices_2d.append(slice_2d)

                # Stack all slices: (N, C, H, W) where N = number of slices
                result[modality] = torch.stack(slices_2d, dim=0)

            # Calculate tumor statistics for all slices if requested
            if 'seg' in volumes and self.return_tumor:
                seg_3d = volumes['seg']
                tumor_areas = []
                has_tumors = []
                seg_slices = []

                for i, slice_idx in enumerate(slice_indices):
                    tumor_stats = self.tumor_calculation(seg_3d, result[self.ref_modality][i], slice_idx)
                    has_tumors.append(tumor_stats['has_tumor'])
                    tumor_areas.append(tumor_stats['tumor_area'])

                    # Extract seg mask slice [H, W] - handle different dimensions
                    if seg_3d.dim() == 4:  # [C, H, W, D]
                        seg_slice = seg_3d[0, :, :, slice_idx]
                    elif seg_3d.dim() == 3:  # [H, W, D]
                        seg_slice = seg_3d[:, :, slice_idx]
                    else:
                        raise ValueError(f"Unexpected seg shape: {seg_3d.shape}")
                    seg_slices.append(seg_slice)

                result['has_tumor'] = torch.tensor(has_tumors)
                result['tumor_area'] = torch.tensor(tumor_areas)
                result['seg_path'] = file_paths.get('seg', None)
                result['seg'] = torch.stack(seg_slices, dim=0)  # [N, H, W]

            result['patient_id'] = patient_id
            result['slice_indices'] = slice_indices

        else:
            # Return random slice (original behavior)
            slice_idx = self._get_random_slice(volumes[self.modalities[0]])
            result = {}
            for modality in self.modalities:
                vol_3d = volumes[modality]
                slice_2d = vol_3d[..., slice_idx]  # (C, H, W)

                # Repeat channels if requested: (1, H, W) -> (3, H, W)
                if self.repeat_channels and slice_2d.shape[0] == 1:
                    slice_2d = slice_2d.repeat(3, 1, 1)

                result[modality] = slice_2d

            # get tumor area from segmentation
            if 'seg' in volumes and self.return_tumor:
                seg_3d = volumes['seg']
                tumor_stats = self.tumor_calculation(seg_3d, result[self.ref_modality], slice_idx)
                result['has_tumor'] = tumor_stats['has_tumor']
                result['tumor_area'] = tumor_stats['tumor_area']

                # Extract seg slice - handle different dimensions
                if seg_3d.dim() == 4:  # [C, H, W, D]
                    result['seg'] = seg_3d[0, :, :, slice_idx]  # [H, W]
                elif seg_3d.dim() == 3:  # [H, W, D]
                    result['seg'] = seg_3d[:, :, slice_idx]  # [H, W]
                else:
                    raise ValueError(f"Unexpected seg shape: {seg_3d.shape}")

            result['patient_id'] = patient_id
            result['slice_idx'] = slice_idx

        return result
    


# =============================================================================
# BraTS Dataset
# =============================================================================

class BraTSDataset(BrainMRIDataset):
    """
    BraTS dataset (4 modalities: t1n, t1c, t2w, t2f).
    """
    
    def __init__(
        self,
        data_dir: str,
        modalities: List[str] = ['t1n', 't1c', 't2w', 't2f'],
        ref_modality: str = 't1c',
        slice_range: Tuple[int, int] = (20, 150),
        image_size: List[int] = [192, 192, 182],
        transform: Optional[Compose] = None,
        crop_size: Optional[List[int]] = None,
        pad_size: Optional[List[int]] = None,
        rotate_90: bool = False,
        return_tumor: bool = True,
        repeat_channels: bool = False,
        return_all_slices: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            modalities=modalities,
            slice_range=slice_range,
            image_size=image_size,
            ref_modality=ref_modality,
            transform=transform,
            pad_size=pad_size,
            crop_size=crop_size,
            rotate_90=rotate_90,
            return_tumor=return_tumor,
            repeat_channels=repeat_channels,
            return_all_slices=return_all_slices,
            seed=seed,
        )
    
    def build_patient_index(self) -> List[Dict[str, str]]:
        """Build BraTS patient index"""
        patient_list = []
        skipped = 0
        for root, dirs, files in os.walk(self.data_dir, followlinks=True):
            if not dirs:  # Leaf directory
                files.sort()
                file_paths = {}
                for f in files:
                    if not f.endswith('.nii.gz'):
                        continue
                    # Parse: BraTS-GLI-00000-000-t1n.nii.gz
                    parts = f.split('-')
                    if len(parts) >= 4:
                        seqtype = parts[-1].replace('.nii.gz', '')
                        if seqtype in self.modalities + ['seg']:
                            file_paths[seqtype] = os.path.join(root, f)
                # Verify all modalities present
                required = set(self.modalities)
                if required.issubset(file_paths.keys()):
                    file_paths['patient_id'] = Path(root).name
                    if 'seg' in file_paths:
                        patient_list.append(file_paths)
                        # Could also set a per-patient flag here
                    else:
                        # No seg, but still valid
                        patient_list.append(file_paths)
                else:
                    skipped += 1
                
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} incomplete patients")

        if len(patient_list) == 0:
            raise ValueError(f"No valid patients found in {self.data_dir}")

        # Sort by patient_id for deterministic ordering across machines/filesystems
        # (os.walk returns dirs in arbitrary inode order, which varies across OSes/runs)
        patient_list.sort(key=lambda x: x['patient_id'])

        return patient_list
