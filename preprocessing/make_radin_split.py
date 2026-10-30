"""Make the RadImageNet held-out split: a path list, and a flat 256x256 PNG folder for gFID and analysis."""
import argparse
import random
from pathlib import Path

from PIL import Image

REGIONS = ('CT/abd', 'CT/lung', 'MR/hip', 'MR/knee', 'MR/shoulder')


def list_images(root):
    paths = []
    for region in REGIONS:
        for pathology in sorted(p for p in (root / region).iterdir() if p.is_dir()):
            paths += sorted(pathology.glob('*.png'))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, required=True, help='folder with CT/ and MR/')
    parser.add_argument('--val_paths', type=Path, required=True, help='output text file')
    parser.add_argument('--out_dir', type=Path, default=None, help='output folder for the flat PNG files')
    parser.add_argument('--val_size', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    paths = list_images(args.data_path)
    perm = list(range(len(paths)))
    random.Random(args.seed).shuffle(perm)
    val = [paths[i] for i in sorted(perm[:args.val_size])]
    args.val_paths.parent.mkdir(parents=True, exist_ok=True)
    args.val_paths.write_text(''.join(f'{p}\n' for p in val))
    print(f'{len(val)} of {len(paths)} images are in {args.val_paths}')

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for p in val:
            name = '_'.join(p.parts[-4:])
            Image.open(p).convert('RGB').resize((256, 256), Image.BICUBIC).save(args.out_dir / name)
        print(f'{len(val)} images are in {args.out_dir}')


if __name__ == '__main__':
    main()
