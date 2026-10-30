"""MixedRadNat — per-step probabilistic mix of RadImageNet and ImageNet.

Used for the JiT mixed-task training experiment that prevents catastrophic
forgetting of ImageNet during RadIN fine-tuning. Each __getitem__ call,
p_nat probability of drawing from natural images (default 10%) and (1-p_nat)
of drawing from radiology images.

Labels follow the extended-pretraining scheme used elsewhere in this repo:
- ImageNet:    labels 0..999       (natural classes)
- CFG-drop:    row 1000            (shared between domains)
- RadIN:       labels 1001..1005   (regions, region+rad_label_offset)

Pair with:
  --class_num 1005 --cfg_drop_index 1000 --label_mode extended
  --label_offset 1001  (so RadIN labels emit 1001..1005)
  ImageNet samples emit labels 0..999 directly (no offset).

The y_embedder has 1006 rows. Row 1000 (CFG-drop) is shared by both domains.

"""
import os
import random
import torchvision.datasets as datasets
from torch.utils.data import Dataset

from util.radimagenet import RadImageNetDataset


class MixedRadNat(Dataset):
    """Per-step probabilistic mix of RadImageNet and ImageNet.

    Args:
        rad_root: RadImageNet root (3-level hierarchy: modality/bodypart/pathology)
        nat_root: ImageNet ImageFolder root containing class-named subdirs
                  (typically /path/to/imagenet2012/train)
        transform: image transform applied to both domains (same transform —
                   centred crop + flip + PILToTensor)
        p_nat: per-sample probability of drawing from nat (default 0.10)
        rad_label_offset: label offset for RadIN samples (1001 for extended labels)
        rad_exclude_paths: optional exclude-paths file for RadIN val split

    Length is set to len(rad) so an "epoch" sweeps a RadIN-equivalent step
    count. Per-step probability of IN is set by p_nat and is *independent of
    __len__* (each __getitem__ samples a dataset directly by weight, ignoring
    idx). DistributedSampler over this dataset still shuffles index order but
    that order has no effect on which sub-dataset is drawn from.
    """

    def __init__(self, rad_root, nat_root, transform=None,
                 p_nat=0.10, rad_label_offset=1001,
                 rad_exclude_paths=None):
        assert 0.0 < p_nat < 1.0, f"p_nat must be in (0, 1), got {p_nat}"

        self.rad = RadImageNetDataset(
            root=rad_root, transform=transform,
            label_offset=rad_label_offset,
            exclude_paths=rad_exclude_paths,
        )
        self.nat = datasets.ImageFolder(nat_root, transform=transform)
        self.p_nat = float(p_nat)

        # Weights for random.choices — index order matches self._datasets.
        self._datasets = [self.rad, self.nat]
        self._weights = [1.0 - self.p_nat, self.p_nat]

        print(f"MixedRadNat: rad={len(self.rad):,} (label_offset={rad_label_offset}) + "
              f"nat={len(self.nat):,} (label_offset=0), "
              f"p_nat={self.p_nat:.2f} "
              f"(per-step probs: rad={self._weights[0]:.2f}, nat={self._weights[1]:.2f})")
        print(f"  __len__ = len(rad) = {len(self.rad):,} → one 'epoch' = RadIN-equivalent step count")

    def __len__(self):
        return len(self.rad)

    def __getitem__(self, idx):
        # idx is ignored: each call independently samples (dataset, sub-idx)
        # by probability weights. random.choices uses Python's RNG which is
        # safe in DataLoader workers (each worker process has independent RNG
        # state since torch DataLoader seeds workers individually).
        ds = random.choices(self._datasets, weights=self._weights, k=1)[0]
        return ds[random.randrange(len(ds))]
