"""LDS_C, our spatial structure metric, and the standard LDS it builds on.

metric_lds is the standard Local-vs-Distant Similarity (LDS). It and its helpers are adapted from iREPA
(https://github.com/End2End-Diffusion/iREPA/metrics/spatial_metrics.py).
metric_lds_c computes LDS_C. It is ours, not part of iREPA.

Copyright (c) 2026 Efthymios Georgiou (metric_lds_c). MIT License, see LICENSE
in the repository root.

iREPA license notice for the adapted code:

MIT License

Copyright (c) 2025 Jaskirat Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn.functional as F


class _GridCache:
    """Cache for grid coordinates and Manhattan distances."""

    def __init__(self):
        self._dist = {}
        self._coords = {}

    def coords(self, H: int, W: int, device: torch.device):
        key = (H, W, str(device))
        if key not in self._coords:
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            self._coords[key] = torch.stack([yy.flatten(), xx.flatten()], dim=1)
        return self._coords[key]

    def dist(self, H: int, W: int, device: torch.device):
        key = (H, W, str(device))
        if key not in self._dist:
            c = self.coords(H, W, device)
            self._dist[key] = (c[:, None, :] - c[None, :, :]).abs().sum(-1)
        return self._dist[key]


_GRID = _GridCache()


def _normalize(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(u.to(torch.float32), dim=-1, eps=eps)


def _reshape_hw(u: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return u.view(u.size(0), H, W, u.size(-1))


def _gram_cos(u: torch.Tensor) -> torch.Tensor:
    return torch.einsum("btd,bsd->bts", u, u)


@torch.no_grad()
def metric_lds(u: torch.Tensor, H: int, W: int, far_dist: int = 6, **_) -> torch.Tensor:
    """iREPA Local-vs-Distant Similarity.

    Args:
        u: (B, T, D) raw token features.
        H, W: grid dimensions (H*W == T).
        far_dist: Manhattan distance >= far_dist counts as "far" (default 6).

    Returns:
        (B,) tensor. Higher = more spatial structure.
    """
    B, T, D = u.shape
    u = _normalize(u)
    uHW = _reshape_hw(u, H, W)

    up = (uHW[:, 1:, :, :] * uHW[:, :-1, :, :]).sum(-1)
    left = (uHW[:, :, 1:, :] * uHW[:, :, :-1, :]).sum(-1)
    local = torch.cat([up.flatten(1), left.flatten(1)], dim=1).mean(dim=1)

    G = _gram_cos(u)
    dist = _GRID.dist(H, W, u.device)
    far_mask = (dist >= far_dist) & (dist > 0)
    denom = far_mask.sum().clamp(min=1)
    far = (G.masked_fill(~far_mask, 0.0).sum(dim=(1, 2)) / denom)

    return local - far


@torch.no_grad()
def metric_lds_c(u: torch.Tensor, H: int, W: int, far_dist: int = 6, **_) -> torch.Tensor:
    """LDS_C: LDS after removing the mean token of each image."""
    return metric_lds(u - u.mean(dim=1, keepdim=True), H, W, far_dist=far_dist)
