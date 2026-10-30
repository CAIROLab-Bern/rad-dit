"""LDS_C (our metric) per block and noise level t for a trained JiT model. The CSV also has the standard LDS.

Usage (from jit/):
    python analysis/compute_lds_c.py --ckpt <ckpt> --model JiT-L/16 \
        --data_path <RadImageNet root> --exclude_paths val_paths.txt --out_csv lds_c.csv
"""
import argparse
import csv
import os
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analysis.jit_features import build_model, BlockCapture
from metrics.lds import metric_lds, metric_lds_c
from util.crop import center_crop_arr
from util.radimagenet import RadImageNetDataset

GRID = 16  # 256x256 images, patch size 16


@torch.no_grad()
def main(args):
    device = 'cuda'
    model, _ = build_model(args.ckpt, device, args.model, class_num=args.class_num,
                           cfg_drop_index=args.cfg_drop_index)
    capture = BlockCapture(model)
    # Same data flow as training: same transforms, real labels, shuffled.
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    dataset = RadImageNetDataset(args.data_path, transform=transform, label_offset=args.label_offset,
                                 exclude_paths=args.exclude_paths or None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True)
    torch.manual_seed(args.seed)
    rows = []
    for t_val in args.t_values:
        n_seen = 0
        for x, y in loader:
            if n_seen >= args.n_images:
                break
            x = x.to(device, non_blocking=True).to(torch.float32).div_(255) * 2.0 - 1.0
            y = y.to(device, non_blocking=True).long()
            t = torch.full((x.size(0),), t_val, device=device)
            t_view = t.view(-1, 1, 1, 1)
            z = t_view * x + (1.0 - t_view) * torch.randn_like(x)

            capture.clear()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model.net(z, t, y)

            for block in range(capture.n_blocks):
                # Remove the in-context class tokens, keep the patch tokens.
                act = capture.activations[block].float()[:, -GRID * GRID:, :]
                lds = metric_lds(act, GRID, GRID).cpu()
                lds_c = metric_lds_c(act, GRID, GRID).cpu()
                for b in range(act.size(0)):
                    rows.append(dict(block=block, t=t_val, lds=float(lds[b]), lds_c=float(lds_c[b])))
            n_seen += x.size(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'saved {len(rows)} rows to {args.out_csv}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model', default='JiT-L/16')
    p.add_argument('--data_path', required=True)
    p.add_argument('--exclude_paths', default='')
    p.add_argument('--out_csv', required=True)
    p.add_argument('--t_values', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument('--n_images', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    # Defaults: ImageNet-pretrained model with RadImageNet regions at 1001..1005.
    # For a scratch model use --class_num 5 --label_offset 0 --cfg_drop_index -1.
    p.add_argument('--class_num', type=int, default=1005)
    p.add_argument('--label_offset', type=int, default=1001)
    p.add_argument('--cfg_drop_index', type=int, default=1000)
    main(p.parse_args())
