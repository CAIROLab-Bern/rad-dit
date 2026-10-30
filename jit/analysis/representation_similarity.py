"""Class-aware per-block cosine similarity on a mixed-domain JiT-L/16 (Figure 4 of the paper).

Sample images from specific RadIN regions and ImNet synsets, then assign EACH
image its correct class label during forward (or override to cfg-drop=1000).

8 pair conditions covering 3 groups:
- intra-class within-domain (tabby-tabby, lung-lung)
- inter-class within-domain (tabby-warplane, lung-knee)
- cross-domain (4 combos: {tabby, warplane} x {lung, knee})

Outputs:
  --out_csv
  --out_png   (2x4 grid heatmap)
"""
import argparse
import csv
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.crop import center_crop_arr
from analysis.jit_features import build_model, BlockCapture, forward_capture, cossim_per_layer


# Mixed extended-mode label scheme:
#   ImNet: 0..999 ; cfg-drop: 1000 ; RadIN regions: 1001..1005
#     1001=CT_abd, 1002=CT_lung, 1003=MR_hip, 1004=MR_knee, 1005=MR_shoulder
RADIN_LABELS = {
    'CT_lung':  1002,
    'MR_knee':  1004,
}
IMNET_LABELS = {
    'tabby':    (281, 'n02123045'),
    'warplane': (895, 'n04552348'),
}

PAIR_SPECS = [
    # name,            class_a,    class_b,    group
    ('intra_tabby',    'tabby',    'tabby',    'intra'),
    ('intra_lung',     'CT_lung',  'CT_lung',  'intra'),
    ('inter_imnet',    'tabby',    'warplane', 'inter_same'),
    ('inter_radin',    'CT_lung',  'MR_knee',  'inter_same'),
    ('cross_TL',       'tabby',    'CT_lung',  'cross'),
    ('cross_TK',       'tabby',    'MR_knee',  'cross'),
    ('cross_WL',       'warplane', 'CT_lung',  'cross'),
    ('cross_WK',       'warplane', 'MR_knee',  'cross'),
]


def _preprocess(img):
    img = center_crop_arr(img, 256)
    arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    t = torch.from_numpy(arr).float() / 255.0
    return t * 2.0 - 1.0


def load_radin_class_images(val_root, class_name, k, device='cuda', seed=0):
    """Load K random images from the validation folder whose filename starts with '<class>_'."""
    prefix = class_name + '_'
    pool = sorted(p for p in Path(val_root).glob('*.png') if p.name.startswith(prefix))
    if len(pool) < k:
        raise RuntimeError(f"only {len(pool)} {class_name} images, need {k}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), k, replace=False)
    imgs = [_preprocess(Image.open(pool[int(i)]).convert('RGB')) for i in idx]
    return torch.stack(imgs, dim=0).to(device)


def load_imnet_class_images(imnet_train, synset, k, device='cuda', seed=0):
    """Load K random JPEGs from <imnet_train>/<synset>/."""
    pool = sorted(Path(imnet_train, synset).glob('*.JPEG'))
    if len(pool) < k:
        raise RuntimeError(f"only {len(pool)} {synset} images, need {k}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), k, replace=False)
    imgs = [_preprocess(Image.open(pool[int(i)]).convert('RGB')) for i in idx]
    return torch.stack(imgs, dim=0).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model', default='JiT-L/16')
    p.add_argument('--radin_val', required=True)
    p.add_argument('--imnet_train', required=True)
    p.add_argument('--mode', choices=['cond', 'uncond'], default='cond')
    p.add_argument('--class_num', type=int, default=1005)
    p.add_argument('--cfg_drop_index', type=int, default=1000)
    p.add_argument('--k_per_class', type=int, default=64)
    p.add_argument('--n_per_t', type=int, default=64)
    p.add_argument('--t_grid', nargs='+', type=float,
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--out_png', required=True)
    args = p.parse_args()

    device = 'cuda'
    print(f'[build] {args.model} {args.ckpt} mode={args.mode}')
    model, _ = build_model(args.ckpt, device, args.model,
                           class_num=args.class_num,
                           cfg_drop_index=args.cfg_drop_index)
    capture = BlockCapture(model)
    n_blocks = capture.n_blocks
    print(f'  n_blocks={n_blocks}')

    K = args.k_per_class
    print(f'[input] loading {K} images per class')
    pools = {}
    for name, label in RADIN_LABELS.items():
        pools[name] = (load_radin_class_images(args.radin_val, name, K, device,
                                               seed=zlib.crc32(name.encode()) & 0xffff),
                       label)
        print(f'  {name:10s} {tuple(pools[name][0].shape)} label={label}')
    for name, (label, synset) in IMNET_LABELS.items():
        pools[name] = (load_imnet_class_images(args.imnet_train, synset, K, device,
                                                seed=zlib.crc32(synset.encode()) & 0xffff),
                       label)
        print(f'  {name:10s} {tuple(pools[name][0].shape)} label={label}')

    t_grid = args.t_grid
    N = args.n_per_t
    results = {p[0]: np.zeros((n_blocks, len(t_grid)), dtype=np.float32) for p in PAIR_SPECS}

    torch.manual_seed(0)
    for ti, t in enumerate(t_grid):
        print(f'\n[t={t:.3f}]')
        for pname, a_cls, b_cls, group in PAIR_SPECS:
            x_pool_a, label_a = pools[a_cls]
            x_pool_b, label_b = pools[b_cls]
            if args.mode == 'uncond':
                label_a = label_b = args.cfg_drop_index

            ia = torch.randint(0, x_pool_a.size(0), (N,), device=device)
            ib = torch.randint(0, x_pool_b.size(0), (N,), device=device)
            if a_cls == b_cls:
                same = (ia == ib)
                while same.any():
                    ib = torch.where(same,
                                     torch.randint(0, x_pool_b.size(0), (N,), device=device),
                                     ib)
                    same = (ia == ib)
            x_a = x_pool_a[ia]
            x_b = x_pool_b[ib]
            eps = torch.randn(N, 3, 256, 256, device=device)
            z_a = (1.0 - t) * eps + t * x_a
            z_b = (1.0 - t) * eps + t * x_b
            y_a = torch.full((N,), label_a, device=device, dtype=torch.long)
            y_b = torch.full((N,), label_b, device=device, dtype=torch.long)

            H_a = forward_capture(model, z_a, t, y_a, capture,
                                  batch_size=args.batch_size)
            H_b = forward_capture(model, z_b, t, y_b, capture,
                                  batch_size=args.batch_size)
            sims = cossim_per_layer(H_a, H_b)
            results[pname][:, ti] = sims
            print(f'  {pname:15s} mean={sims.mean():.3f}')

    # CSV
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair', 'group', 'layer', 't', 'cos_sim'])
        for pname, a_cls, b_cls, group in PAIR_SPECS:
            for L in range(n_blocks):
                for ti, tv in enumerate(t_grid):
                    w.writerow([pname, group, L, f'{tv:.4f}', f'{results[pname][L, ti]:.6f}'])
    print(f'\n[csv] {args.out_csv}')

    # 2x4 heatmap grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True,
                              gridspec_kw={'wspace': 0.05, 'hspace': 0.25})
    vmin = min(M.min() for M in results.values())
    vmax = max(M.max() for M in results.values())
    titles = {
        'intra_tabby':  r'intra: tabby$\;$vs$\;$tabby',
        'intra_lung':   r'intra: CT$\_$lung$\;$vs$\;$CT$\_$lung',
        'inter_imnet':  r'inter-IN: tabby$\;$vs$\;$warplane',
        'inter_radin':  r'inter-Rad: CT$\_$lung$\;$vs$\;$MR$\_$knee',
        'cross_TL':     r'cross: tabby$\;$vs$\;$CT$\_$lung',
        'cross_TK':     r'cross: tabby$\;$vs$\;$MR$\_$knee',
        'cross_WL':     r'cross: warplane$\;$vs$\;$CT$\_$lung',
        'cross_WK':     r'cross: warplane$\;$vs$\;$MR$\_$knee',
    }
    last_im = None
    for ax, (pname, a_cls, b_cls, group) in zip(axes.flat, PAIR_SPECS):
        M = results[pname]
        im = ax.imshow(M, aspect='auto', cmap='YlGnBu',
                       vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
        last_im = im
        n_show = 6
        idx = np.linspace(0, len(t_grid)-1, n_show).round().astype(int)
        ax.set_xticks(idx)
        ax.set_xticklabels([f'{t_grid[i]:.2f}' for i in idx], fontsize=8)
        ax.set_xlabel(r'noise $t$', fontsize=9)
        ax.set_title(titles[pname], fontsize=10)
        ax.set_yticks(range(0, n_blocks, 3))
    axes[0, 0].set_ylabel('Block', fontsize=10)
    axes[1, 0].set_ylabel('Block', fontsize=10)

    fig.suptitle(f'Mixed L/16 class-aware cossim ($K={K}$ per class, mode={args.mode}, flatten)',
                 fontsize=12, y=1.00)
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    cbar_ax = fig.add_axes([0.95, 0.15, 0.012, 0.70])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label('cosine similarity', fontsize=10)
    fig.savefig(args.out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[png] {args.out_png}')


if __name__ == '__main__':
    main()
