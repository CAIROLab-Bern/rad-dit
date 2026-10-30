"""Load a JiT checkpoint and capture the hidden states of its blocks."""
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denoiser import Denoiser
from util.pretrained import load_jit_pretrained


def build_model(ckpt_path, device, model='JiT-L/16',
                class_num=1000, cfg_drop_index=1000, in_channels=3):
    """Build Denoiser, load ckpt, switch to EMA weights for inference.

    class_num / cfg_drop_index defaults match pure ImNet pretrained ckpts.
    For RadIN-finetuned extended-mode ckpts (e.g. mixed models trained with
    --class_num 1005 --label_offset 1001 --cfg_drop_index 1000), pass
    class_num=1005, cfg_drop_index=1000.
    """
    args = types.SimpleNamespace(
        model=model, img_size=256,
        class_num=class_num, label_drop_prob=0.0,
        attn_dropout=0.0, proj_dropout=0.0,
        P_mean=-0.8, P_std=0.8, t_eps=5e-2, noise_scale=1.0,
        ema_decay1=0.9999, ema_decay2=0.9996,
        sampling_method='euler', num_sampling_steps=50, cfg=1.0,
        interval_min=0.1, interval_max=0.9,
        cfg_drop_index=cfg_drop_index, in_channels=in_channels,
    )
    m = Denoiser(args).to(device).eval()
    load_jit_pretrained(m, ckpt_path, args)
    return m, args


class BlockCapture:
    """Register a forward hook on each transformer block; collect post-block hidden states."""
    def __init__(self, model):
        self.activations = {}      # L -> Tensor (B, T, D), on GPU
        self.handles = []
        self.n_blocks = len(model.net.blocks)
        for L, blk in enumerate(model.net.blocks):
            self.handles.append(blk.register_forward_hook(self._make_hook(L)))

    def _make_hook(self, L_idx):
        def hook(mod, inp, out):
            self.activations[L_idx] = out.detach()
        return hook

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def forward_capture(model, z, t_scalar, y, capture, batch_size=16):
    """Forward in mini-batches. Returns list[n_blocks] of CPU tensors (N, T*D)."""
    device = z.device
    N = z.size(0)
    per_layer_chunks = [[] for _ in range(capture.n_blocks)]
    for i in range(0, N, batch_size):
        z_b = z[i:i + batch_size]
        y_b = y[i:i + batch_size]
        t_b = torch.full((z_b.size(0),), t_scalar, device=device, dtype=z_b.dtype)
        capture.clear()
        _ = model.net(z_b, t_b, y_b)
        for L in range(capture.n_blocks):
            per_layer_chunks[L].append(capture.activations[L].flatten(1).float().cpu())
    return [torch.cat(chunks, dim=0) for chunks in per_layer_chunks]


def cossim_per_layer(H_a, H_b):
    """H_a, H_b: list[n_blocks] of (N, K). Cosine over K, mean over N."""
    return np.array([F.cosine_similarity(ha, hb, dim=1).mean().item() for ha, hb in zip(H_a, H_b)])
