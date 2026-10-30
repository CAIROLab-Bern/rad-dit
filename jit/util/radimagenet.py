"""RadImageNet pixel-space dataset for JiT.

3-level hierarchy: <root>/<modality>/<bodypart>/<pathology>/*.png
Returns (PIL.Image, region_label) where region_label is one of:
    CT/abd=0, CT/lung=1, MR/hip=2, MR/knee=3, MR/shoulder=4

When `label_offset` is set (e.g. 1001 for the "extended" mode that re-uses an
ImageNet-pretrained label embedding), the returned label is shifted so the
caller can address the freshly-initialised rows 1001..1005 of the embedder
without colliding with the preserved ImageNet 0-999 + CFG-1000 tokens.

When `exclude_paths` points to a text file (one path per line),
images whose `<modality>/<bodypart>/<pathology>/<file.png>` relative path
appears in that file are skipped. Relative-path
matching (not basename) is required because ~25 basenames occur in multiple
pathology subdirs across the full tree.
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


REGION_LABEL_MAP = {
    ('CT', 'abd'):      0,
    ('CT', 'lung'):     1,
    ('MR', 'hip'):      2,
    ('MR', 'knee'):     3,
    ('MR', 'shoulder'): 4,
}
NUM_REGIONS = len(REGION_LABEL_MAP)


def _load_exclude_rel_paths(path):
    """Read newline-separated paths and return the set of their
    <modality>/<bodypart>/<pathology>/<file>.png suffixes. This is the part
    of the path that's stable across data_path prefixes (for example on different machines).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"exclude_paths file not found: {p}")
    rels = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = Path(line).parts
            # Take the LAST 4 components: modality / bodypart / pathology / file.png
            if len(parts) < 4:
                raise ValueError(f"exclude line has too few path components: {line!r}")
            rels.add('/'.join(parts[-4:]))
    return rels


class RadImageNetDataset(Dataset):
    def __init__(self, root, transform=None, label_offset=0, exclude_paths=None):
        self.transform = transform
        self.label_offset = int(label_offset)
        self.samples = []

        if exclude_paths:
            excluded = _load_exclude_rel_paths(exclude_paths)
            print(f"RadImageNetDataset: loaded {len(excluded):,} relative paths to exclude from {exclude_paths}")
        else:
            excluded = set()

        skipped = 0
        root = Path(root)
        for modality_dir in sorted(root.iterdir()):
            if not modality_dir.is_dir():
                continue
            modality = modality_dir.name
            for bodypart_dir in sorted(modality_dir.iterdir()):
                if not bodypart_dir.is_dir():
                    continue
                key = (modality, bodypart_dir.name)
                if key not in REGION_LABEL_MAP:
                    print(f"WARNING: unknown region {modality}/{bodypart_dir.name}, skipping")
                    continue
                region = REGION_LABEL_MAP[key]
                bodypart = bodypart_dir.name
                for pathology_dir in sorted(bodypart_dir.iterdir()):
                    if not pathology_dir.is_dir():
                        continue
                    pathology = pathology_dir.name
                    for img_path in sorted(pathology_dir.glob('*.png')):
                        rel = f"{modality}/{bodypart}/{pathology}/{img_path.name}"
                        if rel in excluded:
                            skipped += 1
                            continue
                        self.samples.append((img_path, region))

        if not self.samples:
            raise RuntimeError(f"RadImageNetDataset found no PNGs under {root}")
        excl_note = f" (excluded {skipped:,} via {exclude_paths})" if excluded else ""
        print(f"RadImageNetDataset: {len(self.samples):,} images in {root} "
              f"(label_offset={self.label_offset}){excl_note}")
        if excluded and skipped != len(excluded):
            print(f"WARNING: expected to exclude {len(excluded):,} but only matched {skipped:,} — "
                  f"check that the exclude_paths file matches this dataset root")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, region = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, region + self.label_offset
