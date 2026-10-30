"""Linear probe on JiT block activations for RadImageNet pathology classification.

For a checkpoint and a list of noise levels t, for each (block, t) cell:
  - extract mean-pooled features from validation images
  - make a stratified train/test split
  - train one nn.Linear with Adam and report the test accuracy

Output: a CSV with the accuracy per cell, and a heatmap PNG.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# Linear probe trained in pure PyTorch (no sklearn needed)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.crop import center_crop_arr
from analysis.jit_features import build_model, BlockCapture


_ID_PATTERN = re.compile(r'^(.+)_[a-z]+\d+$')


def parse_pathology_label(name):
    """For filename like 'CT_lung_Nodule_lung009666.png', return 'CT_lung_Nodule'.
    Strips the trailing '<bodypart><digits>' ID token. Returns None on no match.
    """
    stem = name.rsplit('.', 1)[0]
    m = _ID_PATTERN.match(stem)
    return m.group(1) if m else None


def load_radin_pathology_balanced(root, k_per_class, img_size=256, device='cuda', seed=0):
    """Walk root, parse pathology label, keep classes with >= k_per_class samples,
    sample k_per_class per class. Returns (X, y, class_names)."""
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    root = Path(root)
    for p in root.glob('*.png'):
        lbl = parse_pathology_label(p.name)
        if lbl is not None:
            buckets[lbl].append(p)
    print(f'  raw pathology classes found: {len(buckets)}')
    # filter classes with at least k_per_class samples
    keep = {k: v for k, v in buckets.items() if len(v) >= k_per_class}
    print(f'  classes with >= {k_per_class} samples: {len(keep)}')
    class_names = sorted(keep.keys())
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    # show histogram
    counts = sorted(((k, len(v)) for k, v in keep.items()),
                    key=lambda x: -x[1])
    print(f'  largest 5: {[(c, n) for c, n in counts[:5]]}')
    print(f'  smallest 5: {[(c, n) for c, n in counts[-5:]]}')

    chosen, labels = [], []
    for cls in class_names:
        pool = keep[cls]
        idx = rng.choice(len(pool), k_per_class, replace=False)
        picked = [pool[i] for i in idx]
        chosen.extend(picked)
        labels.extend([name_to_idx[cls]] * k_per_class)
    print(f'  loaded {len(chosen)} images across {len(class_names)} pathology classes')

    imgs = []
    for p in chosen:
        img = Image.open(p).convert('RGB')
        img = center_crop_arr(img, img_size)
        arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
        t = torch.from_numpy(arr).to(device).float() / 255.0
        t = t * 2.0 - 1.0
        imgs.append(t)
    X = torch.stack(imgs, dim=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names


@torch.no_grad()
def extract_layer_features(model, capture, X_clean, t_scalar, y_uncond_label,
                           batch_size=16, device='cuda'):
    """Forward X_clean with noise level t_scalar, capture per-block features.

    Returns list of (N, D) tensors, one per block, on CPU.
    Features are mean-pooled over image-patch tokens only (in_context stripped).
    """
    N = X_clean.size(0)
    n_blocks = capture.n_blocks
    num_patches = model.net.x_embedder.num_patches
    per_layer_chunks = [[] for _ in range(n_blocks)]
    for i in range(0, N, batch_size):
        x = X_clean[i:i + batch_size]
        nb = x.size(0)
        eps = torch.randn_like(x)
        z = (1.0 - t_scalar) * eps + t_scalar * x
        t_b = torch.full((nb,), t_scalar, device=device, dtype=z.dtype)
        y_b = torch.full((nb,), y_uncond_label, device=device, dtype=torch.long)
        capture.clear()
        _ = model.net(z, t_b, y_b)
        for L in range(n_blocks):
            act = capture.activations[L]            # (B, T, D)
            B, T, D = act.shape
            if T > num_patches:
                act_patches = act[:, T - num_patches:, :]
            else:
                act_patches = act
            pooled = act_patches.mean(dim=1).float().cpu()   # (B, D)
            per_layer_chunks[L].append(pooled)
    return [torch.cat(c, dim=0).numpy() for c in per_layer_chunks]


def stratified_split_idx(y, test_size, seed=0):
    """Return (train_idx, test_idx) with class-balanced test partition."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_test = int(round(test_size * len(idx_c)))
        test_idx.extend(idx_c[:n_test].tolist())
        train_idx.extend(idx_c[n_test:].tolist())
    return np.array(train_idx), np.array(test_idx)


def train_probe(X, y, train_idx, test_idx, n_classes=5,
                lr=1e-3, epochs=200, device='cuda'):
    """Simple Adam-trained linear probe. Returns test top-1 accuracy."""
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).long().to(device)
    W = torch.nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(W(X[train_idx]), y[train_idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (W(X[test_idx]).argmax(1) == y[test_idx]).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model', required=True,
                   choices=['JiT-B/16', 'JiT-L/16', 'JiT-H/16'])
    p.add_argument('--class_num', type=int, default=1005)
    p.add_argument('--cfg_drop_index', type=int, default=1000)
    p.add_argument('--radin_val', required=True,
                   help='validation folder with flat PNG layout')
    p.add_argument('--k_per_class', type=int, default=100,
                   help='Images per class')
    p.add_argument('--t_grid', nargs='+', type=float,
                   default=[0.1, 0.3, 0.5, 0.9])
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--n_depth_points', type=int, default=0,
                   help='If >0: probe only N blocks at evenly-spaced normalised '
                        'depths via linspace(0, n_blocks-1, N). First (0) and '
                        'last (n_blocks-1) ALWAYS included (linspace endpoints). '
                        'Default 0 = all blocks.')
    p.add_argument('--seed', type=int, default=0,
                   help='Seed for ALL RNGs (torch, numpy, python.random) AND '
                        'for the data-sampling + train/test split. Lets multi-seed '
                        'averaging in plot stage filter out the per-cell jitter '
                        '(~3 pp std from probe Linear init + eps noise). '
                        'Default 0 reproduces the legacy single-run behaviour.')
    p.add_argument('--out_csv', required=True)
    p.add_argument('--out_png', required=True)
    args = p.parse_args()

    # Seed every RNG up-front. After this, the probe is reproducible up to GPU
    # non-determinism (cuDNN may still pick different kernels — we don't disable
    # it because the variance from that is tiny vs. the run-level jitter we're
    # averaging out).
    import random as _py_random
    _py_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = 'cuda'
    print(f'[build] {args.model}  class_num={args.class_num}  ckpt={args.ckpt}')
    model, _ = build_model(args.ckpt, device, args.model,
                           class_num=args.class_num,
                           cfg_drop_index=args.cfg_drop_index)
    capture = BlockCapture(model)
    n_blocks = capture.n_blocks
    print(f'  n_blocks={n_blocks}')

    print(f'[input] sampling {args.k_per_class}/class from {args.radin_val}  seed={args.seed}')
    X_clean, y, class_names = load_radin_pathology_balanced(
        args.radin_val, k_per_class=args.k_per_class, device=device, seed=args.seed)
    n_classes = len(class_names)
    print(f'  X={tuple(X_clean.shape)}  n_classes={n_classes}')

    train_idx, test_idx = stratified_split_idx(y, args.test_size, seed=args.seed)
    print(f'  split: {len(train_idx)} train / {len(test_idx)} test')

    # Decide which blocks to probe.
    if args.n_depth_points and args.n_depth_points > 0:
        block_indices = np.round(np.linspace(0, n_blocks - 1, args.n_depth_points)).astype(int)
        block_indices = sorted(set(block_indices.tolist()))   # dedup if N<n_depth
        assert block_indices[0] == 0, f'first block missing: {block_indices}'
        assert block_indices[-1] == n_blocks - 1, f'last block missing: {block_indices}'
        print(f'[depth] subsampling {len(block_indices)} of {n_blocks} blocks: {block_indices}')
    else:
        block_indices = list(range(n_blocks))
    nP = len(block_indices)
    norm_depth = [b / max(n_blocks - 1, 1) for b in block_indices]

    results = np.zeros((nP, len(args.t_grid)), dtype=np.float32)
    for ti, t_scalar in enumerate(args.t_grid):
        print(f'\n[t={t_scalar:.2f}] extracting features...')
        feats = extract_layer_features(model, capture, X_clean, t_scalar,
                                       args.cfg_drop_index,
                                       batch_size=args.batch_size, device=device)
        print(f'  features shape per layer: {feats[0].shape}')
        for pi, L in enumerate(block_indices):
            acc = train_probe(feats[L], y, train_idx, test_idx,
                              n_classes=n_classes, device=device)
            results[pi, ti] = acc
            print(f'  L{L:>2d}  (d={norm_depth[pi]:.2f})  acc={acc:.4f}')

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer', 'norm_depth', 't', 'top1_acc'])
        for pi, L in enumerate(block_indices):
            for ti, t in enumerate(args.t_grid):
                w.writerow([L, f'{norm_depth[pi]:.4f}', f'{t:.3f}',
                            f'{results[pi, ti]:.6f}'])
    print(f'\n[csv] {args.out_csv}')

    # Plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(results, aspect='auto', cmap='viridis',
                   vmin=results.min(), vmax=results.max(), origin='lower')
    ax.set_xticks(range(len(args.t_grid)))
    ax.set_xticklabels([f'{t:.2f}' for t in args.t_grid])
    ax.set_xlabel('noise level t')
    ax.set_ylabel('normalised depth (block / N-1)')
    ax.set_title(f'Linear probe top-1 acc (RadIN pathology, {n_classes} classes)\n'
                 f'{args.model}, K={args.k_per_class}/class')
    ax.set_yticks(range(nP))
    ax.set_yticklabels([f'{d:.2f}' for d in norm_depth])
    plt.colorbar(im, ax=ax, label='top-1 accuracy')
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[png] {args.out_png}')


if __name__ == '__main__':
    main()
