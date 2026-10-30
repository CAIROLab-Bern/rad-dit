"""LDiT-XL/1 linear-probe analogue of jit/analysis/linear_probe.py.

For an LDiT ckpt and noise levels t:
  for each (block, t) cell:
    - encode RGB image with VA-VAE -> latent (B, 32, 16, 16)
    - normalise latent channel-wise (with latents_stats.pt)
    - mix with Gaussian noise via rectified-flow convention:
          z_t = (1 - t) * eps + t * z_0
    - forward through the diffusion model with class label = cfg_drop_index
    - capture per-block hidden state (B, T+sink, D), strip sink tokens, mean-pool
      over T patches -> (B, D)
    - 80/20 stratified split, linear probe (Adam, 200 epochs)
    - report test top-1 accuracy

This mirrors the JiT script semantics exactly. The only LDiT-specific differences are:
  * model = LightningDiT (28 blocks for XL/1) loaded from a training YAML, with `ema`
    state-dict (same key inference.py uses)
  * input is latent (B, 32, 16, 16) not pixel (B, 3, 256, 256)
  * VA-VAE encode + channel-wise normalisation must happen before noising

n_classes is fully determined by (radin_val, k_per_class) -- same auto-discovery
filter as the JiT script -- so passing k_per_class=50 on the validation folder gives 33 classes,
identical to the JiT path.

Output:
  <out_csv>     per-cell accuracy (block, t, top1_acc)
  <out_png>     heatmap (block index x t)
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
from omegaconf import OmegaConf

# LDiT modules
THIS_FILE = os.path.abspath(__file__)
LDIT_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
sys.path.insert(0, LDIT_ROOT)
from models.lightningdit import LightningDiT_models
from tokenizer.vavae import VA_VAE


# ---------------------------- class extraction ----------------------------

_ID_PATTERN = re.compile(r'^(.+)_[a-z]+\d+$')


def parse_pathology_label(name):
    """Same parser as JiT jit/analysis/linear_probe.py — keep IDENTICAL
    so that n_classes(JiT) == n_classes(LDiT) on the same (root, k_per_class)."""
    stem = name.rsplit('.', 1)[0]
    m = _ID_PATTERN.match(stem)
    return m.group(1) if m else None


def load_radin_pathology_balanced(root, k_per_class, img_size=256, device='cuda', seed=0):
    """Walk root, parse pathology label, keep classes with >= k_per_class samples,
    sample k_per_class per class. Returns (X, y, class_names).
    X is float in [-1, 1], shape (N, 3, H, W) — exactly the VA-VAE input format.
    """
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    root = Path(root)
    for p in root.glob('*.png'):
        lbl = parse_pathology_label(p.name)
        if lbl is not None:
            buckets[lbl].append(p)
    print(f'  raw pathology classes found: {len(buckets)}')
    keep = {k: v for k, v in buckets.items() if len(v) >= k_per_class}
    print(f'  classes with >= {k_per_class} samples: {len(keep)}')
    class_names = sorted(keep.keys())
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    counts = sorted(((k, len(v)) for k, v in keep.items()), key=lambda x: -x[1])
    print(f'  largest 5: {[(c, n) for c, n in counts[:5]]}')
    print(f'  smallest 5: {[(c, n) for c, n in counts[-5:]]}')

    chosen, labels = [], []
    for cls in class_names:
        pool = keep[cls]
        idx = rng.choice(len(pool), k_per_class, replace=False)
        chosen.extend([pool[i] for i in idx])
        labels.extend([name_to_idx[cls]] * k_per_class)
    print(f'  loaded {len(chosen)} images across {len(class_names)} pathology classes')

    # Use ADM's center-crop (same as VAE input)
    def _center_crop(pil_img, size):
        while min(*pil_img.size) >= 2 * size:
            pil_img = pil_img.resize(tuple(x // 2 for x in pil_img.size),
                                     resample=Image.BOX)
        scale = size / min(*pil_img.size)
        pil_img = pil_img.resize(tuple(round(x * scale) for x in pil_img.size),
                                 resample=Image.BICUBIC)
        arr = np.array(pil_img)
        crop_y = (arr.shape[0] - size) // 2
        crop_x = (arr.shape[1] - size) // 2
        return arr[crop_y:crop_y + size, crop_x:crop_x + size]

    imgs = []
    for p in chosen:
        img = Image.open(p).convert('RGB')
        arr = _center_crop(img, img_size)
        arr = arr.astype(np.float32).transpose(2, 0, 1) / 255.0
        t = torch.from_numpy(arr).to(device)
        t = t * 2.0 - 1.0      # VA-VAE convention: normalize to [-1, 1]
        imgs.append(t)
    X = torch.stack(imgs, dim=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names


# ---------------------------- model build + capture ----------------------------

def build_ldit(config_yaml, ckpt_path, device):
    """Build LDiT model from a training YAML (same one used at training time);
    load weights from `ema` key of the ckpt — that's what inference.py uses.

    Returns (model, latent_size, cfg_drop_index_for_uncond).
    cfg_drop_index = num_classes - 1 by convention in our YAMLs.
    """
    cfg = OmegaConf.load(config_yaml)
    # latent size
    if 'downsample_ratio' in cfg.vae:
        latent_size = cfg.data.image_size // cfg.vae.downsample_ratio
    else:
        latent_size = cfg.data.image_size // 16

    mcfg = cfg.model
    model = LightningDiT_models[mcfg.model_type](
        input_size=latent_size,
        num_classes=cfg.data.num_classes,
        use_qknorm=mcfg.get('use_qknorm', False),
        use_swiglu=mcfg.get('use_swiglu', False),
        use_rope=mcfg.get('use_rope', False),
        use_rmsnorm=mcfg.get('use_rmsnorm', False),
        wo_shift=mcfg.get('wo_shift', False),
        in_channels=mcfg.get('in_chans', 4),
        learn_sigma=mcfg.get('learn_sigma', False),
    ).to(device).eval()

    print(f'[ckpt] {ckpt_path}')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'ema' in ck:
        sd = ck['ema']
    elif 'model' in ck:
        sd = ck['model']
    else:
        sd = ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  loaded ema state dict. missing={len(missing)} unexpected={len(unexpected)}')
    if len(missing) > 0:
        print(f'  WARN missing[:5]={missing[:5]}')
    if len(unexpected) > 0:
        print(f'  WARN unexpected[:5]={unexpected[:5]}')

    # The CFG row is the last row of the label table, at index num_classes.
    cfg_drop_index = cfg.data.num_classes
    return model, latent_size, cfg_drop_index, cfg


class LDiTBlockCapture:
    """Forward-hook each transformer block; collect post-block hidden states."""
    def __init__(self, model):
        self.activations = {}
        self.handles = []
        self.n_blocks = len(model.blocks)
        for L, blk in enumerate(model.blocks):
            self.handles.append(blk.register_forward_hook(self._make_hook(L)))

    def _make_hook(self, L_idx):
        def hook(_mod, _inp, out):
            self.activations[L_idx] = out.detach()
        return hook

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------- feature extraction ----------------------------

@torch.no_grad()
def extract_layer_features(model, vae, latent_mean, latent_std, capture,
                           X_imgs, t_scalar, y_uncond,
                           batch_size=8, device='cuda'):
    """X_imgs: pixel-space (N, 3, 256, 256) in [-1, 1].
    Returns list[n_blocks] of (N, D) numpy arrays — mean-pooled over T patches,
    sink tokens dropped.
    """
    N = X_imgs.size(0)
    n_blocks = capture.n_blocks
    num_sink = int(getattr(model, 'num_sink_tokens', 0) or 0)
    chunks = [[] for _ in range(n_blocks)]

    for i in range(0, N, batch_size):
        xb = X_imgs[i:i + batch_size]
        nb = xb.size(0)
        # VAE encode -> (nb, 32, 16, 16)
        latent = vae.model.encode(xb).sample()
        # channel-wise normalisation, same convention as training
        z0 = (latent - latent_mean) / latent_std
        # rectified-flow noising: z_t = (1-t)*eps + t*z0   (t=1 clean, t=0 pure noise)
        eps = torch.randn_like(z0)
        zt = (1.0 - t_scalar) * eps + t_scalar * z0
        t_b = torch.full((nb,), t_scalar, device=device, dtype=zt.dtype)
        y_b = torch.full((nb,), y_uncond, device=device, dtype=torch.long)
        capture.clear()
        _ = model(zt, t_b, y_b)
        for L in range(n_blocks):
            act = capture.activations[L]      # (B, T+sink, D)
            T_eff = act.shape[1] - num_sink   # strip sink tokens at the END
            act_patches = act[:, :T_eff]
            pooled = act_patches.mean(dim=1).float().cpu()
            chunks[L].append(pooled)
    return [torch.cat(c, dim=0).numpy() for c in chunks]


# ---------------------------- probe ----------------------------

def stratified_split_idx(y, test_size, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_test = int(round(test_size * len(idx_c)))
        test_idx.extend(idx_c[:n_test].tolist())
        train_idx.extend(idx_c[n_test:].tolist())
    return np.array(train_idx), np.array(test_idx)


def train_probe(X, y, train_idx, test_idx, n_classes,
                lr=1e-3, epochs=200, device='cuda'):
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)
    W = torch.nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(W(X_t[train_idx]), y_t[train_idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (W(X_t[test_idx]).argmax(1) == y_t[test_idx]).float().mean().item()


# ---------------------------- main ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True,
                   help='Trained LDiT ckpt (.pt with `ema` key)')
    p.add_argument('--config', required=True,
                   help='Training YAML matching the ckpt (used to build model + read num_classes / latent_size / vae)')
    p.add_argument('--vae_yaml', default=None,
                   help='Tokenizer YAML override. Default: tokenizer/configs/<train_cfg.vae.model_name>.yaml')
    p.add_argument('--latent_stats',
                   default='pretrained/vavae-imagenet256-f16d32-dinov2/latents_stats.pt',
                   help='latents_stats.pt with per-channel mean & std')
    p.add_argument('--radin_val', required=True,
                   help='validation folder with flat PNG layout')
    p.add_argument('--k_per_class', type=int, default=50,
                   help='Pathology-class sample threshold. 50 -> 33 classes on the validation folder.')
    p.add_argument('--t_grid', nargs='+', type=float, default=[0.1, 0.3, 0.5])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--n_depth_points', type=int, default=0,
                   help='If >0: probe only N blocks at evenly-spaced normalised '
                        'depths via linspace(0, n_blocks-1, N). The first (0) '
                        'and the last (n_blocks-1) block are ALWAYS included '
                        '(linspace endpoints). Default 0 = all blocks.')
    p.add_argument('--cfg_drop_index', type=int, default=None,
                   help='Override the cfg drop slot. Default = num_classes from config; pass num_classes to use the actual drop slot.')
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

    # --- model + VAE
    print(f'[build] config={args.config}')
    model, latent_size, cfg_drop_index_default, train_cfg = build_ldit(
        args.config, args.ckpt, device)
    # CLI override of cfg_drop_index (set to num_classes for the actual drop slot)
    cfg_drop_index = args.cfg_drop_index if args.cfg_drop_index is not None else cfg_drop_index_default
    capture = LDiTBlockCapture(model)
    print(f'  n_blocks={capture.n_blocks}  latent_size={latent_size}  '
          f'num_classes={train_cfg.data.num_classes}  cfg_drop_index={cfg_drop_index}')

    vae_yaml = args.vae_yaml or os.path.join(
        LDIT_ROOT, 'tokenizer/configs', f'{train_cfg.vae.model_name}.yaml')
    print(f'[vae] {vae_yaml}')
    vae = VA_VAE(vae_yaml)
    vae.model.to(device).eval()

    print(f'[stats] {args.latent_stats}')
    ls = torch.load(args.latent_stats, map_location='cpu', weights_only=False)
    # latents_stats.pt has {'mean': (1, C, 1, 1), 'std': (1, C, 1, 1)}
    if isinstance(ls, dict):
        latent_mean = ls.get('mean', ls.get('latent_mean')).to(device).float()
        latent_std = ls.get('std', ls.get('latent_std')).to(device).float()
    else:
        raise RuntimeError(f'unexpected latents_stats format: {type(ls)}')
    print(f'  mean shape={tuple(latent_mean.shape)} std shape={tuple(latent_std.shape)}')

    # --- data
    print(f'[input] sampling {args.k_per_class}/class from {args.radin_val}  seed={args.seed}')
    X_imgs, y, class_names = load_radin_pathology_balanced(
        args.radin_val, k_per_class=args.k_per_class, device=device, seed=args.seed)
    n_classes = len(class_names)
    print(f'  X={tuple(X_imgs.shape)}  n_classes={n_classes}')

    train_idx, test_idx = stratified_split_idx(y, args.test_size, seed=args.seed)
    print(f'  split: {len(train_idx)} train / {len(test_idx)} test')

    # --- decide which blocks to probe
    nB = capture.n_blocks
    if args.n_depth_points and args.n_depth_points > 0:
        block_indices = np.round(np.linspace(0, nB - 1, args.n_depth_points)).astype(int)
        # dedup (only triggers if nB < args.n_depth_points; not the case here)
        block_indices = sorted(set(block_indices.tolist()))
        # Always include first and last (linspace already guarantees this; assert).
        assert block_indices[0] == 0, f'first block missing: {block_indices}'
        assert block_indices[-1] == nB - 1, f'last block missing: {block_indices}'
        print(f'[depth] subsampling {len(block_indices)} of {nB} blocks: {block_indices}')
    else:
        block_indices = list(range(nB))
    nP = len(block_indices)
    norm_depth = [b / max(nB - 1, 1) for b in block_indices]

    # --- per-t feature extraction + per-probed-block linear probe
    results = np.zeros((nP, len(args.t_grid)), dtype=np.float32)
    for ti, t_scalar in enumerate(args.t_grid):
        print(f'\n[t={t_scalar:.2f}] extracting features...')
        feats = extract_layer_features(
            model, vae, latent_mean, latent_std, capture,
            X_imgs, t_scalar, cfg_drop_index,
            batch_size=args.batch_size, device=device)
        print(f'  features shape per layer: {feats[0].shape}')
        for pi, L in enumerate(block_indices):
            acc = train_probe(feats[L], y, train_idx, test_idx,
                              n_classes=n_classes, device=device)
            results[pi, ti] = acc
            print(f'  L{L:>2d}  (d={norm_depth[pi]:.2f})  acc={acc:.4f}')

    # --- save CSV
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer', 'norm_depth', 't', 'top1_acc'])
        for pi, L in enumerate(block_indices):
            for ti, t in enumerate(args.t_grid):
                w.writerow([L, f'{norm_depth[pi]:.4f}', f'{t:.3f}',
                            f'{results[pi, ti]:.6f}'])
    print(f'\n[csv] {args.out_csv}')

    # --- plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(results, aspect='auto', cmap='viridis',
                   vmin=results.min(), vmax=results.max(), origin='lower')
    ax.set_xticks(range(len(args.t_grid)))
    ax.set_xticklabels([f'{t:.2f}' for t in args.t_grid])
    ax.set_xlabel('noise level t')
    ax.set_ylabel('normalised depth (block / N-1)')
    ax.set_title(f'Linear probe top-1 acc (RadIN pathology, {n_classes} classes)\n'
                 f'LDiT-XL/1, K={args.k_per_class}/class')
    ax.set_yticks(range(nP))
    ax.set_yticklabels([f'{d:.2f}' for d in norm_depth])
    plt.colorbar(im, ax=ax, label='top-1 accuracy')
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[png] {args.out_png}')


if __name__ == '__main__':
    main()
