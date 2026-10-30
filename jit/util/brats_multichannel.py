"""BraTS-GLI multichannel slice dataset for JiT.

Reads sharded safetensors produced by
    preprocessing/extract_brats_multichannel_safetensors.py

Shard schema (per file):
    slices:     [N, 4, 256, 256] fp16  channels = [t1n, t1c, t2w, t2f]
    seg_mask:   [N, 1, 256, 256] fp16  via (seg/4)*2-1  (val shards: all -1)
    subject_id: [N] int32
    slice_idx:  [N] int32
    has_tumor:  [N] uint8

Dataset yields (image, label) pairs:
    image: [in_channels, 256, 256] fp32 in [-1, 1]
           - in_channels=4: slices                                    (T1n, T1c, T2w, T2f)
           - in_channels=5: cat([slices, seg_mask], dim=0)            (above + mask)
    label: int  -- by default has_tumor (0/1); add label_offset for the extended
           pretraining label scheme if needed.

Design notes:
- All tensors are kept in fp16 on disk and cast to fp32 at __getitem__ time
  so the train loop sees the same precision as ImageNet/RadIN runs.
- Shards are mmap'd from disk via safetensors so we don't pay RAM for the
  whole dataset upfront. Per-shard index lookups are O(1).
- No PIL transforms (data is already pre-normalized to [-1, 1] and pre-
  upsampled to 256x256 at extraction time).
"""
import os
import glob
import torch
from torch.utils.data import Dataset
from safetensors.torch import safe_open


class MultiChannelBraTSDataset(Dataset):
    """BraTS-GLI multichannel slice dataset.

    Class label = has_tumor (per BraTS-2024 ground-truth segmentation).
    BraTS is a 2-CLASS conditioning problem:
        - 0 = no tumor in this slice
        - 1 = has tumor in this slice

    Pair the dataset with:
        scratch run:     --class_num 2   --label_offset 0     (labels 0/1)
        pretrained run:  --class_num 1002 --label_offset 1001 (labels 1001/1002,
                         preserves ImageNet 0..999 + CFG-1000 rows in y_embedder)

    The `cfg_drop_index` should be:
        scratch:     -1 (default → drop = num_classes = 2; last row of 3-row table)
        pretrained:  1000  (re-use the preserved ImageNet CFG token at row 1000)

    Detailed enumeration mirrors the RadIN-extended scheme (which used 5 new
    classes on top of the same 1000+1 ImageNet rows). For BraTS (2 new classes):

        y_embedder.embedding_table  rows  total = num_classes + 1
        ─────────────────────────────────────────────────────────
            0 .. 999    ImageNet classes (preserved from JiT-L 800ep ckpt)
                1000    ImageNet CFG-drop token (preserved; --cfg_drop_index 1000)
                1001    BraTS class 0 = no_tumor   (newly initialised, learnt)
                1002    BraTS class 1 = has_tumor  (newly initialised, learnt)
        ─────────────────────────────────────────────────────────
        --class_num 1002  →  table has 1003 rows (indices 0..1002)
    """

    # BraTS slice-level binary label space (no tumor / has tumor).
    NUM_CLASSES = 2

    # Channel names in `slices` shard schema (index → modality).
    CHANNEL_NAMES = ['t1n', 't1c', 't2w', 't2f']

    def __init__(self, shard_dir, in_channels=4, label_offset=0, select_channel=None,
                 flat_mode=False):
        assert in_channels in (1, 3, 4, 5), \
            f"MultiChannelBraTSDataset only supports in_channels in {{1, 3, 4, 5}}, got {in_channels}"
        if flat_mode:
            # Flat mode: each (slice, modality) pair is a separate sample, so
            # __len__ = base_slices * 4. Used for single-channel BraTS
            # training (matches LDiT's single-modality flat setup): replicates
            # one modality across 3 RGB channels, cycling all 4 modalities
            # across the epoch. Modality is decoded from idx in __getitem__,
            # not fixed at construction.
            assert in_channels == 3, \
                f"flat_mode requires in_channels=3 (got {in_channels})"
            assert select_channel is None, \
                "flat_mode cycles through all modalities; do not pass --select_channel"
            self._select_idx = None
        elif in_channels in (1, 3):
            # Single-modality mode. in_channels=1 keeps shape [1, H, W];
            # in_channels=3 replicates the chosen modality across 3 channels
            # to use the standard RadIN/ImageNet-style 3-ch architecture
            # WITHOUT any channel-grow.
            assert select_channel is not None, \
                f"in_channels={in_channels} requires --select_channel (one of t1n/t1c/t2w/t2f)"
            assert select_channel in self.CHANNEL_NAMES, \
                f"select_channel must be one of {self.CHANNEL_NAMES}, got '{select_channel}'"
            self._select_idx = self.CHANNEL_NAMES.index(select_channel)
        else:
            assert select_channel is None, \
                f"select_channel only valid when in_channels in (1, 3) (got in_channels={in_channels})"
            self._select_idx = None
        self.shard_dir = shard_dir
        self.in_channels = in_channels
        self.select_channel = select_channel
        self.label_offset = int(label_offset)
        self.flat_mode = bool(flat_mode)

        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.safetensors')))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.safetensors found under {shard_dir}")

        # Build (shard_idx, intra_idx) index for O(1) lookup. We compute shard
        # sizes once at __init__ by transient-opening each shard; the persistent
        # handles used during __getitem__ are populated lazily per worker.
        self.shard_sizes = []
        for p in self.shard_paths:
            with safe_open(p, framework='pt') as f:
                n = f.get_slice('slices').get_shape()[0]
            self.shard_sizes.append(n)
        self.cum_sizes = []
        running = 0
        for n in self.shard_sizes:
            running += n
            self.cum_sizes.append(running)

        # Lazy per-worker handle cache: shard_idx -> safe_open object (entered).
        # mmap-shared across workers via the OS page cache; per-process the
        # parsed header lives once instead of being re-parsed on every fetch.
        # Repeated mmap-setup/teardown was ~150 ms/call; this keeps it < 1 ms.
        self._open_shards = {}

        if self.flat_mode:
            sel = ", flat_mode=True (cycles all 4 modalities; __len__ = 4× slices)"
        elif in_channels in (1, 3):
            sel = f", select_channel={select_channel}"
        else:
            sel = ""
        print(f"MultiChannelBraTSDataset: {len(self.shard_paths)} shards, "
              f"{self.cum_sizes[-1]:,} total slices, in_channels={in_channels}{sel}, "
              f"NUM_CLASSES={self.NUM_CLASSES} (has_tumor 0/1), "
              f"label_offset={label_offset} (emitted labels: "
              f"{label_offset}/{label_offset+1})")

    def __len__(self):
        base = self.cum_sizes[-1]
        return base * 4 if self.flat_mode else base

    def _locate(self, idx):
        for shard_idx, cum in enumerate(self.cum_sizes):
            if idx < cum:
                intra = idx - (self.cum_sizes[shard_idx - 1] if shard_idx > 0 else 0)
                return shard_idx, intra
        raise IndexError(idx)

    def _get_shard(self, shard_idx):
        f = self._open_shards.get(shard_idx)
        if f is None:
            # Enter the context manager but never explicitly exit; the underlying
            # mmap is reclaimed when the worker process terminates. PyTorch
            # DataLoader workers persist across batches so the cache is sticky.
            cm = safe_open(self.shard_paths[shard_idx], framework='pt')
            f = cm.__enter__()
            self._open_shards[shard_idx] = f
        return f

    def __getitem__(self, idx):
        if self.flat_mode:
            slice_idx = idx // 4
            mod_idx = idx % 4
        else:
            slice_idx = idx
            mod_idx = self._select_idx                          # None for 4/5 ch
        shard_idx, intra = self._locate(slice_idx)
        f = self._get_shard(shard_idx)
        slices = f.get_slice('slices')[intra]                   # [4, 256, 256] fp16
        if self.in_channels == 5:
            seg = f.get_slice('seg_mask')[intra]                # [1, 256, 256] fp16
            img = torch.cat([slices, seg], dim=0)               # [5, 256, 256]
        elif self.in_channels == 1:
            img = slices[mod_idx:mod_idx + 1]                   # [1, 256, 256]
        elif self.in_channels == 3:
            single = slices[mod_idx:mod_idx + 1]                # [1, 256, 256]
            img = single.expand(3, -1, -1).contiguous()         # [3, 256, 256] — replicated
        else:
            img = slices                                        # [4, 256, 256]
        has_tumor = int(f.get_slice('has_tumor')[intra].item())

        img = img.float()                                       # cast to fp32 for training
        # Clip bicubic-overshoot from the MRI channels back into [-1, 1].
        # (BraTSDataset normalizes MRI to [-1, 1] before upsample, but bicubic
        # ringing at skull boundaries can excursion ~30% beyond. The mask
        # channel is nearest-upsampled and exactly in {-1,-0.5,0,0.5,1}, so the
        # clip is a no-op for it.)
        img = img.clamp_(-1.0, 1.0)
        label = has_tumor + self.label_offset
        return img, label
