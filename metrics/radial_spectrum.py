"""Radial power spectrum profile across image sources.

For each source directory, computes per-image 2D FFT power spectrum, bins
radially, averages across samples. Plots all sources together with log-y axis.

Usage:
    python metrics/radial_spectrum.py \
        --sources real:<path>:Reference \
                  jit_b:<path>:JiT-B/16 \
                  jit_l:<path>:JiT-L/16 \
                  jit_h:<path>:JiT-H/16 \
        --n_samples 1024 \
        --out_png radial_spectrum.png
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def compute_image_spectrum(img_arr):
    """img_arr: (H, W, 3) or (H, W) uint8 or float. Returns (H, W) power spectrum."""
    if img_arr.ndim == 3:
        if img_arr.shape[0] in (1, 3):
            img_arr = img_arr.transpose(1, 2, 0)
        img = img_arr.mean(axis=-1)
    else:
        img = img_arr
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    img = img - img.mean()
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.abs(F) ** 2


def radial_bin(P, n_bins=20):
    """Return (bin_centers, mean_power_per_bin), normalized frequency in [0, 1]."""
    H, W = P.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = min(cy, cx)
    r_norm = r / max_r
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r_norm >= bin_edges[i]) & (r_norm < bin_edges[i + 1])
        if mask.sum() > 0:
            binned[i] = P[mask].mean()
    return bin_centers, binned


def load_and_spectrum_one(path, n_bins=20):
    img = np.array(Image.open(path).convert('RGB'))
    return radial_bin(compute_image_spectrum(img), n_bins)[1]


def collect_spectra(input_dir, n_samples, n_bins=20):
    """Walk input_dir, sample up to n_samples images, return (N, n_bins)."""
    paths = list(Path(input_dir).glob('**/*.png'))
    if not paths:
        paths = list(Path(input_dir).glob('**/*.jpg'))
    if not paths:
        paths = list(Path(input_dir).glob('**/*.JPEG'))
    rng = np.random.default_rng(0)
    if len(paths) > n_samples:
        idx = rng.choice(len(paths), n_samples, replace=False)
        paths = [paths[i] for i in idx]
    print(f'[{input_dir}] found {len(paths)} images')
    spectra = []
    for i, p in enumerate(paths):
        try:
            s = load_and_spectrum_one(p, n_bins)
            spectra.append(s)
        except Exception as e:
            print(f'  skip {p}: {e}')
        if i % 200 == 0:
            print(f'  {i}/{len(paths)}')
    return np.array(spectra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', nargs='+', required=True,
        help='each: key:dir:label e.g. real:/path/to/radin_val:Reference')
    p.add_argument('--n_samples', type=int, default=1024)
    p.add_argument('--n_bins', type=int, default=20)
    p.add_argument('--out_png', required=True)
    p.add_argument('--out_npz', default=None)
    args = p.parse_args()

    results = {}
    for src in args.sources:
        parts = src.split(':')
        if len(parts) != 3:
            raise ValueError(f"bad --sources entry: {src}; want key:dir:label")
        key, dirp, label = parts
        spectra = collect_spectra(dirp, args.n_samples, args.n_bins)
        if len(spectra) == 0:
            print(f"  skipping {key}: no usable images")
            continue
        freq = np.linspace(0, 1, args.n_bins + 1)
        bin_centers = (freq[:-1] + freq[1:]) / 2
        results[key] = {
            'label': label,
            'dir':   dirp,
            'mean':  spectra.mean(axis=0),
            'std':   spectra.std(axis=0),
            'n':     len(spectra),
            'freq':  bin_centers,
        }

    # Plot: 3-panel
    #   (a) raw log-log spectrum
    #   (b) area-normalized spectrum (per-source SHAPE comparison; same area)
    #   (c) ratio gen/real per frequency band
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ['black', 'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    markers = ['o', 'D', '^', 'v', 's', 'x']
    real_mean = None
    for i, (key, r) in enumerate(results.items()):
        if key in ('real', 'reference') or 'real' in key.lower() or i == 0:
            real_mean = r['mean']
            break
    for i, (key, r) in enumerate(results.items()):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        # (a) raw: LINEAR x, LOG y (PixelGen-style)
        axes[0].semilogy(r['freq'], r['mean'], color=c, marker=m, lw=2, ms=6,
                         label=f"{r['label']} (n={r['n']})")
        # (b) area-normalized
        area = np.trapz(r['mean'], r['freq'])
        axes[1].semilogy(r['freq'], r['mean'] / area, color=c, marker=m, lw=2, ms=6,
                         label=r['label'])
        # (c) ratio gen / real: linear x, linear y
        if real_mean is not None and r['mean'] is not real_mean:
            ratio = r['mean'] / real_mean
            axes[2].plot(r['freq'], ratio, color=c, marker=m, lw=2, ms=6,
                         label=r['label'])
    axes[0].set_xlabel('Normalized radial frequency')
    axes[0].set_ylabel('Radial power')
    axes[0].set_title('(a) Raw radial spectrum (log y)')
    axes[0].legend(loc='lower left', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Normalized radial frequency')
    axes[1].set_ylabel('Radial power / total area')
    axes[1].set_title('(b) Area-normalized (shape comparison)')
    axes[1].legend(loc='lower left', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(1.0, color='black', lw=0.8, ls=':')
    axes[2].set_xlabel('Normalized radial frequency')
    axes[2].set_ylabel('Power(gen) / Power(real)')
    axes[2].set_title('(c) Per-frequency fidelity ratio')
    axes[2].legend(loc='lower left', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.3)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {args.out_png}')

    if args.out_npz:
        np.savez(args.out_npz, **{k: v['mean'] for k, v in results.items()},
                 freq=list(results.values())[0]['freq'])
        print(f'saved {args.out_npz}')


if __name__ == '__main__':
    main()
