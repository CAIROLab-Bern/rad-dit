"""
Safetensors shard dataloader for SiT/DiT training on extracted latents.
Simple loader with lazy loading and LRU caching.
"""
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import json
from pathlib import Path
from safetensors import safe_open
from typing import List, Dict
import psutil
import os
import numpy as np
import shutil

from shardloader.utils import (
    TUMOR_LABEL_MAPPING_EXTENDED,
    REGION_LABEL_MAPPING_EXTENDED,
    SEED_SCALE, EPOCH_SCALE, WINDOW_SCALE,
)


class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def print_stats(self, label=""):
        mem_gb = self.process.memory_info().rss / 1024**3
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[{label}] CPU: {mem_gb:.2f} GB | GPU: {gpu_allocated:.2f}/{gpu_reserved:.2f} GB")
        else:
            print(f"[{label}] CPU: {mem_gb:.2f} GB")
    
    def get_cpu_gb(self):
        return self.process.memory_info().rss / 1024**3


class ShardedLatentDataset(Dataset):
    """
    Simple dataset for loading pre-extracted VAE latents from sharded safetensors.

    Returns all available fields from shards as a dictionary.

    Expected structure:
        features_dir/
        ├── metadata.json (optional)
        └── shards/
            ├── shard000.safetensors
            ├── shard001.safetensors
            └── ...
    """

    def __init__(
        self,
        features_dir: str,
        shard_cache_size: int = 2,
        verbose: bool = False,
        latent_norm: bool = False,
        latent_multiplier: float = 1.0,
        vavae_stats_path: str = None,
        latent_keys: List[str] = None,
        dataset_fraction: float = 1.0,
    ):
        """
        Args:
            features_dir: Path to directory containing shards/
            shard_cache_size: Number of recent shards to keep in RAM
                             0 = no cache (always load from disk)
                             N = keep N most recent shards
                            -1 = load all shards into RAM
            verbose: If True, print detailed memory stats during initialization
            latent_norm: If True, normalize latents using channel-wise mean/std
            latent_multiplier: Multiply latents by this factor after normalization
            vavae_stats_path: Optional path to official VA-VAE stats file to copy if not exists
            latent_keys: Names of tensors in each shard to treat as latents
                         (normalized, multiplied). Defaults to ['latents'].
                         The FIRST key also drives shard-size discovery.
            dataset_fraction: Fraction of shards to keep, in (0, 1].
                              Applied at construction time so the index/window
                              sampler sees a smaller dataset. Default 1.0.
        """
        self.features_dir = Path(features_dir)
        self.shards_dir = self.features_dir / 'shards'
        self.shard_cache_size = shard_cache_size
        self.verbose = verbose
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        self.vavae_stats_path = vavae_stats_path
        self.latent_keys = list(latent_keys) if latent_keys is not None else ['latents']
        if not (0.0 < dataset_fraction <= 1.0):
            raise ValueError(
                f"dataset_fraction must be in (0, 1], got {dataset_fraction}")
        self.dataset_fraction = float(dataset_fraction)
        # Validate directories exist
        self._validate_directories()

        # Load metadata (optional)
        self.metadata = self._load_metadata()

        # Discover shard files
        self.shard_paths = self._discover_shard_paths()

        # Build index mappings for fast lookups
        self._build_index_mappings()

        # Get schema info from first shard
        self._extract_schema_info()

        # Initialize LRU cache
        self._initialize_cache()

        # Optionally preload all shards into RAM
        if self.shard_cache_size == -1:
            self._preload_all_shards()

        # Load latent normalization statistics if needed
        if self.latent_norm:
            self._latent_mean, self._latent_std = self._load_latent_stats()
        else:
            print(f"✓ Indexed {self.total_samples:,} samples (cache={self.shard_cache_size} shards)")

    def _validate_directories(self):
        """Validate that required directories exist."""
        if not self.shards_dir.exists():
            raise ValueError(f"Shards directory not found: {self.shards_dir}")

    def _load_metadata(self) -> dict:
        """Load metadata.json if it exists."""
        metadata_path = self.features_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _discover_shard_paths(self) -> List[Path]:
        """Discover and sort all shard files, then optionally keep a leading fraction."""
        shard_paths = sorted(self.shards_dir.glob('shard*.safetensors'))
        if not shard_paths:
            raise ValueError(f"No shard files found in {self.shards_dir}")
        if self.dataset_fraction < 1.0:
            n_keep = max(1, int(round(len(shard_paths) * self.dataset_fraction)))
            print(
                f"  Subsetting to {n_keep}/{len(shard_paths)} shards "
                f"(dataset_fraction={self.dataset_fraction:.2f})"
            )
            shard_paths = shard_paths[:n_keep]
        return shard_paths

    def _build_index_mappings(self):
        """
        Build index mappings for efficient sample access.

        Creates three data structures:
        1. index_map: List[Tuple[int, int]]
           Maps global_sample_idx -> (shard_idx, local_idx_in_shard)
           Example: index_map[5000] = (2, 150) means sample 5000 is in shard 2 at position 150

        2. shard_sizes: List[int]
           Number of samples in each shard
           Example: shard_sizes[2] = 2048 means shard 2 has 2048 samples

        3. shard_to_samples: Dict[int, List[int]]
           Reverse mapping: shard_idx -> [list of global_sample_indices]
           Example: shard_to_samples[2] = [4000, 4001, ..., 6047]
           Used for O(1) lookup when loading shard windows
        """
        self.index_map: List[tuple] = []
        self.shard_sizes: List[int] = []
        self.shard_to_samples: Dict[int, List[int]] = {}

        if self.verbose:
            print(f"Indexing {len(self.shard_paths)} shards...")

        # `latent_keys[0]` drives shard-size discovery — defaults to 'latents'.
        index_key = self.latent_keys[0]
        for shard_idx, shard_path in enumerate(self.shard_paths):
            with safe_open(shard_path, framework='pt', device='cpu') as f:
                shard_size = f.get_tensor(index_key).shape[0]
            # Track shard size
            self.shard_sizes.append(shard_size)
            # Build forward and reverse mappings
            self.shard_to_samples[shard_idx] = []
            for local_idx in range(shard_size):
                global_idx = len(self.index_map)
                self.index_map.append((shard_idx, local_idx))
                self.shard_to_samples[shard_idx].append(global_idx)

        self.total_samples = len(self.index_map)


    def _extract_schema_info(self):
        """Extract schema information from first shard (keys and shapes)."""
        index_key = self.latent_keys[0]
        with safe_open(self.shard_paths[0], framework='pt', device='cpu') as f:
            self.available_keys = list(f.keys())
            self.latent_shape = tuple(f.get_tensor(index_key)[0].shape)

    def _initialize_cache(self):
        """Initialize LRU cache structures."""
        self.shard_cache: Dict[int, Dict[str, torch.Tensor]] = {}  # Loaded shard data
        self.cache_order: List[int] = []  # LRU tracking: most recent at end

    def _preload_all_shards(self):
        """Preload all shards into RAM (used when shard_cache_size=-1)."""
        if self.verbose:
            print(f"Loading all shards into RAM...")
            tracker = MemoryTracker()

        for shard_idx in range(len(self.shard_paths)):
            self._cache_shard(shard_idx)
            if self.verbose and (shard_idx + 1) % 5 == 0:
                tracker.print_stats(f"Loaded {shard_idx + 1}/{len(self.shard_paths)} shards")

        print(f"✓ Loaded {self.total_samples:,} samples into RAM")

    def _load_latent_stats(self):
        """Load or compute latent normalization statistics."""
        latent_stats_cache_file = self.features_dir / "latents_stats.pt"

        # If stats file exists, load and return
        if latent_stats_cache_file.exists():
            latent_stats = torch.load(latent_stats_cache_file)
            mean = latent_stats['mean'].squeeze(0)  # [1, C, 1, 1] -> [C, 1, 1]
            std = latent_stats['std'].squeeze(0)    # [1, C, 1, 1] -> [C, 1, 1]
            print(f"✓ Loaded latent stats from {latent_stats_cache_file}")
            print(f"  Stats shape: mean={mean.shape}, std={std.shape}")
            if mean.shape[0] == 32:
                print(f"  Channels: {mean.shape[0]} (VA-VAE f16d32)")
            return mean, std

        # If official VA-VAE stats path provided and exists, copy it
        if self.vavae_stats_path and Path(self.vavae_stats_path).exists():
            print(f"Copying official VA-VAE stats from: {self.vavae_stats_path}")
            print(f"  Destination: {latent_stats_cache_file}")
            shutil.copy2(self.vavae_stats_path, latent_stats_cache_file)
            latent_stats = torch.load(latent_stats_cache_file)
            mean = latent_stats['mean'].squeeze(0)  # [1, C, 1, 1] -> [C, 1, 1]
            std = latent_stats['std'].squeeze(0)    # [1, C, 1, 1] -> [C, 1, 1]
            print(f"✓ Copied and loaded official stats")
            print(f"  Stats shape: mean={mean.shape}, std={std.shape}")
            if mean.shape[0] == 32:
                print(f"  Channels: {mean.shape[0]} (VA-VAE f16d32)")
            return mean, std

        # Otherwise compute from scratch
        if self.vavae_stats_path:
            print(f"⚠️  Official stats path provided but not found: {self.vavae_stats_path}")
        print("Computing latent statistics from scratch...")
        latent_stats = self._compute_latent_stats()
        torch.save(latent_stats, latent_stats_cache_file)
        print(f"✓ Saved computed stats to {latent_stats_cache_file}")
        # Stats computed locally already have the right shape [C, 1, 1]
        return latent_stats['mean'], latent_stats['std']

    def _compute_latent_stats(self):
        """Compute channel-wise mean and std for latent normalization.
        Uses the first key in self.latent_keys (the 'index' key)."""
        from tqdm import tqdm
        index_key = self.latent_keys[0]
        num_samples = min(10000, self.total_samples)
        random_indices = np.random.choice(self.total_samples, num_samples, replace=False)
        latents = []
        for idx in tqdm(random_indices, desc="Computing latent stats"):
            shard_idx, local_idx = self.index_map[idx]
            shard_data = self._get_shard(shard_idx)
            latent = shard_data[index_key][local_idx:local_idx+1]
            latents.append(latent)
        latents = torch.cat(latents, dim=0)  # [N, C, H, W]
        # Compute channel-wise mean and std: [N, C, H, W] -> [C, 1, 1]
        mean = latents.mean(dim=[0, 2, 3], keepdim=True).squeeze(0)  # [1, C, 1, 1] -> [C, 1, 1]
        std = latents.std(dim=[0, 2, 3], keepdim=True).squeeze(0)    # [1, C, 1, 1] -> [C, 1, 1]
        latent_stats = {'mean': mean, 'std': std}
        print(f"Computed latent stats shape: mean={mean.shape}, std={std.shape}")
        return latent_stats

    def _load_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Load a shard from disk."""
        shard_data = {}
        with safe_open(self.shard_paths[shard_idx], framework='pt', device='cpu') as f:
            for key in f.keys():
                shard_data[key] = f.get_tensor(key)
        return shard_data

    def _cache_shard(self, shard_idx: int):
        """Load shard into cache with LRU eviction."""
        if shard_idx in self.shard_cache:
            # Update LRU
            self.cache_order.remove(shard_idx)
            self.cache_order.append(shard_idx)
            return

        # Load
        shard_data = self._load_shard(shard_idx)
        self.shard_cache[shard_idx] = shard_data
        self.cache_order.append(shard_idx)

        # Evict if full
        if self.shard_cache_size > 0 and self.shard_cache_size < len(self.shard_cache):
            evict_idx = self.cache_order.pop(0)
            del self.shard_cache[evict_idx]

    def _get_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Get shard (from cache or disk)."""
        if self.shard_cache_size == 0:
            return self._load_shard(shard_idx)
        else:
            self._cache_shard(shard_idx)
            return self.shard_cache[shard_idx]

    def load_shard_window(self, shard_indices: List[int]):
        """
        Load a specific window of shards into cache, clearing others.
        Used by 'ShardWindowSampler' to explicitly manage which shards are in RAM.
        """
        # Clear existing cache
        self.shard_cache.clear()
        self.cache_order.clear()

        # Load requested shards
        for shard_idx in shard_indices:
            self._cache_shard(shard_idx)

    def get_sample_indices_for_shards(self, shard_indices: List[int]) -> List[int]:
        """
        Get all sample indices that belong to the specified shards.

        Returns list of global sample indices from these shards.
        Uses pre-computed reverse mapping for O(1) lookup per shard.
        """
        sample_indices = []
        for shard_idx in shard_indices:
            sample_indices.extend(self.shard_to_samples[shard_idx])
        return sample_indices

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict with ALL available fields from the shard:
            - latents: [C, H, W]
            - modalities: scalar
            - slice_normalized: scalar
            - has_tumor: scalar (if available)
            - tumor_sizes: scalar (if available)
            - seg_masks: [H, W] (if available)
            - slice_indices: scalar (if available)
        """
        shard_idx, local_idx = self.index_map[idx]
        shard_data = self._get_shard(shard_idx)

        # Return all fields for this sample
        sample = {}
        for key in shard_data.keys():
            sample[key] = shard_data[key][local_idx]

        # Apply latent normalization and scaling (from LightningDiT's img_latent_dataset).
        for key in self.latent_keys:
            if key in sample:
                latent = sample[key]

                # Squeeze out any extra singleton dimensions from extraction
                # (latents should be [C, H, W], not [1, C, H, W])
                # if latent.dim() == 4 and latent.shape[0] == 1:
                #     latent = latent.squeeze_(0)  # In-place squeeze

                if self.latent_norm:
                    latent = (latent - self._latent_mean) / self._latent_std
                latent = latent * self.latent_multiplier
                sample[key] = latent

        return sample


class ShardWindowSampler(torch.utils.data.Sampler):
    """
    Sampler that loads shards in windows for memory-efficient training.

    Automatically transitions between shard windows during iteration:
    - Shuffles all shards at epoch start (via set_epoch)
    - Splits shards into windows of size shard_window_size
    - Loads one window at a time, shuffles samples within window
    - Transitions to next window when current exhausted
    - One epoch = one pass through all shards/samples

    Compatible with:
    - Single ShardedLatentDataset
    - ConcatDataset of multiple ShardedLatentDataset instances
    - DistributedDataParallel (DDP)
    """

    def __init__(
        self,
        dataset,
        shard_window_size: int,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset: ShardedLatentDataset or ConcatDataset of ShardedLatentDatasets
            shard_window_size: Number of shards to load in RAM at once (total across all datasets)
            num_replicas: Number of processes (for DDP)
            rank: Rank of current process (for DDP)
            shuffle: Whether to shuffle shards and samples
            seed: Random seed for reproducibility
        """
        # Initialize DDP parameters
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.shard_window_size = shard_window_size

        # Parse dataset (single or concatenated)
        self._parse_dataset(dataset)

        # Validate all datasets are correct type
        self._validate_dataset_types()

        # Build global shard mapping
        self._build_shard_mapping()

        # Validate window size against total shards
        self._validate_window_size()

        # Calculate DDP sample distribution
        self._calculate_sample_distribution()

    def _parse_dataset(self, dataset):
        """
        Parse input dataset into list of datasets.

        Handles both single ShardedLatentDataset and ConcatDataset.
        Sets:
            - self.datasets: List of ShardedLatentDataset instances
            - self.is_concat: Whether input was ConcatDataset
            - self.cumulative_sizes: Cumulative sample counts for offset calculation
        """
        if isinstance(dataset, ConcatDataset):
            self.datasets = list(dataset.datasets)
            self.is_concat = True
            self.cumulative_sizes = dataset.cumulative_sizes
        else:
            self.datasets = [dataset]
            self.is_concat = False
            self.cumulative_sizes = [len(dataset)]

    def _validate_dataset_types(self):
        """Validate that all datasets are ShardedLatentDataset instances."""
        for ds in self.datasets:
            if not isinstance(ds, ShardedLatentDataset):
                raise TypeError("All datasets must be ShardedLatentDataset instances")

    def _build_shard_mapping(self):
        """
        Build unified shard mapping across all datasets.

        Creates: self.shard_to_dataset: List[Tuple[int, int]]
            Maps global_shard_idx -> (dataset_idx, dataset_shard_idx)
            Example: shard_to_dataset[15] = (1, 5) means global shard 15
                        is shard 5 in dataset 1
        """
        self.shard_to_dataset = []
        for ds_idx, ds in enumerate(self.datasets):
            for shard_idx in range(len(ds.shard_paths)):
                self.shard_to_dataset.append((ds_idx, shard_idx))

    def _validate_window_size(self):
        """Auto-cap window size at total shards.

        A window larger than the dataset size is not a semantic error — the
        sampler just uses all available shards. Auto-capping is friendlier
        than crashing for small data fractions.
        """
        total_shards = len(self.shard_to_dataset)
        if self.shard_window_size > total_shards:
            print(
                f"  shard_window_size ({self.shard_window_size}) capped to "
                f"total shards ({total_shards}) — small dataset."
            )
            self.shard_window_size = total_shards

    def _calculate_sample_distribution(self):
        """
        Calculate how samples are distributed across DDP replicas.

        Sets:
            - self.total_samples: Total samples across all datasets
            - self.num_samples: Samples per replica (rounded up)
            - self.total_size: Padded total (divisible by num_replicas)
        """
        self.total_samples = sum(len(ds) for ds in self.datasets)
        self.num_samples = int(np.ceil(self.total_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling (called by training loop)."""
        self.epoch = epoch

    def _get_shuffled_shard_order(self):
        """Shuffle global shards for this epoch. Returns list of global shard indices."""
        if self.shuffle:
            g = torch.Generator()
            # Scale seed and epoch, so that different (seed, epoch) pairs do not
            # give the same shard order. +9_999 keeps this stream disjoint from
            # the window seeds.
            g.manual_seed(self.seed * SEED_SCALE + self.epoch * EPOCH_SCALE + 9_999)
            return torch.randperm(len(self.shard_to_dataset), generator=g).tolist()
        else:
            return list(range(len(self.shard_to_dataset)))

    def _create_shard_windows(self, global_shard_indices):
        """Split global shards into windows of size shard_window_size."""
        windows = []
        for i in range(0, len(global_shard_indices), self.shard_window_size):
            windows.append(global_shard_indices[i:i + self.shard_window_size])
        return windows

    def _get_samples_for_window(self, window_global_shards):
        """
        Get all sample indices for a window WITHOUT loading shards.

        Returns global sample indices (accounting for ConcatDataset offsets).
        Does NOT load shards - that happens lazily in __getitem__() with LRU cache.

        Steps:
        1. Group global shards by their source dataset
        2. Get local sample indices from each dataset
        3. Convert to global indices (add ConcatDataset offset if needed)
        """
        # Step 1: Group shards by dataset
        shards_per_dataset = self._group_shards_by_dataset(window_global_shards)

        # Step 2 & 3: Collect and convert sample indices
        window_samples = []
        for ds_idx, ds_shard_indices in shards_per_dataset.items():
            global_samples = self._get_global_samples_from_dataset(ds_idx, ds_shard_indices)
            window_samples.extend(global_samples)

        return window_samples

    def _group_shards_by_dataset(self, window_global_shards):
        """
        Group global shard indices by their source dataset.

        Args:
            window_global_shards: List of global shard indices

        Returns:
            Dict[int, List[int]]: dataset_idx -> [list of dataset-local shard indices]
        """
        shards_per_dataset = {}
        for global_shard_idx in window_global_shards:
            ds_idx, ds_shard_idx = self.shard_to_dataset[global_shard_idx]
            if ds_idx not in shards_per_dataset:
                shards_per_dataset[ds_idx] = []
            shards_per_dataset[ds_idx].append(ds_shard_idx)
        return shards_per_dataset

    def _get_global_samples_from_dataset(self, ds_idx, ds_shard_indices):
        """
        Get global sample indices from a specific dataset's shards.

        Args:
            ds_idx: Index of dataset
            ds_shard_indices: List of shard indices (local to this dataset)

        Returns:
            List[int]: Global sample indices
        """
        # Get local sample indices for this dataset
        local_samples = self.datasets[ds_idx].get_sample_indices_for_shards(ds_shard_indices)

        # Convert to global indices (add ConcatDataset offset if needed)
        if self.is_concat:
            offset = self.cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0
            global_samples = [idx + offset for idx in local_samples]
        else:
            global_samples = local_samples

        return global_samples

    def _shuffle_window_samples(self, window_samples, window_idx):
        """
        Shuffle samples within a window using window-specific seed.
        All ranks must use the SAME seed to avoid duplicate samples!
        After shuffling identically, each rank takes every Nth sample.

        Seed calculation ensures:
        - Different shuffles across epochs (epoch component)
        - Different shuffles across windows (window_idx component)
        - SAME shuffle across ranks (NO rank component!)
        - No collisions across reasonable epoch/window ranges
        """
        if self.shuffle:
            g = torch.Generator()
            # All ranks shuffle identically.
            # Uses constants from utils.py: SEED_SCALE, EPOCH_SCALE, WINDOW_SCALE
            window_seed = (self.seed * SEED_SCALE +
                          self.epoch * EPOCH_SCALE +
                          window_idx * WINDOW_SCALE)
            g.manual_seed(window_seed)
            indices = torch.randperm(len(window_samples), generator=g).tolist()
            return [window_samples[i] for i in indices]
        return window_samples

    def __iter__(self):
        """
        Main iteration logic:
        1. Shuffle all shards for this epoch (epoch-level randomness)
        2. Split shards into windows
        3. For each window:
           - Get all sample indices from window (NO loading - just indices!)
           - Shuffle samples within window (window-level randomness)
        4. Pad and subsample for DDP

        NOTE: This is called once per worker! We only compute indices here.
        Actual shard loading happens lazily in __getitem__() with LRU cache.

        DDP Behavior (CRITICAL - prevents duplicate samples):
        - All ranks use SAME seed for shard shuffle → same windows
        - All ranks use SAME seed for sample shuffle → identical shuffle
        - Each rank gets every Nth sample (interleaved after shuffle)
        - This ensures: same shards cached, NO duplicate samples across ranks
        """
        # Step 1: Epoch-level shard shuffling (SAME across all ranks)
        global_shard_indices = self._get_shuffled_shard_order()
        # Step 2: Create windows
        windows = self._create_shard_windows(global_shard_indices)
        # Step 3: Process windows sequentially (indices only, no loading)
        all_indices = []
        for window_idx, window_shards in enumerate(windows):
            # Get sample indices for this window (no loading)
            window_samples = self._get_samples_for_window(window_shards)
            # Shuffle samples within window
            window_samples = self._shuffle_window_samples(window_samples, window_idx)
            all_indices.extend(window_samples)

        # Pad to make divisible by num_replicas
        if len(all_indices) < self.total_size:
            all_indices += all_indices[:(self.total_size - len(all_indices))]

        # Subsample for this rank
        all_indices = all_indices[self.rank:self.total_size:self.num_replicas]
        assert len(all_indices) == self.num_samples

        return iter(all_indices)

    def __len__(self):
        return self.num_samples








def collate_has_tumor_extended(batch: List[Dict]) -> tuple:
    """BraTS binary pathology, appended ON TOP of ImageNet (num_classes=1002).

    Same 3-tuple shape as the other collates. tumor_sizes is always None
    (pure binary tumor/no-tumor; no continuous conditioner).

    has_tumor → TUMOR_LABEL_MAPPING_EXTENDED:
        -1 (unknown) → 1000 (preserved ImageNet CFG/uncond token)
         0 (no tumor) → 1001 (NEW class)
         1 (has tumor)→ 1002 (NEW class)
    """
    latents = torch.stack([item['latents'] for item in batch])
    if 'has_tumor' in batch[0]:
        labels = torch.tensor(
            [TUMOR_LABEL_MAPPING_EXTENDED.get(int(item['has_tumor'].item()), 1000)
             for item in batch],
            dtype=torch.int64,
        )
    else:
        labels = torch.full((len(batch),), 1000, dtype=torch.int64)
    return latents, labels, None


def collate_region_extended(batch: List[Dict]) -> tuple:
    """RadImageNet 5-region conditioning, appended ON TOP of ImageNet
    (num_classes=1005). Same 3-tuple shape; tumor_sizes always None.

    region_label (0-4 from the extractor) → REGION_LABEL_MAPPING_EXTENDED
    (1001..1005). Unknown/missing → 1000 (preserved ImageNet CFG token).
    """
    latents = torch.stack([item['latents'] for item in batch])
    if 'region_label' in batch[0]:
        labels = torch.tensor(
            [REGION_LABEL_MAPPING_EXTENDED.get(int(item['region_label'].item()), 1000)
             for item in batch],
            dtype=torch.int64,
        )
    else:
        labels = torch.full((len(batch),), 1000, dtype=torch.int64)
    return latents, labels, None



