"""Block SVD: compress b x b tiles independently.

A global SVD has to describe the whole photo with the same few layers. Tiles
let each region pick its own layers, which helps photos with very different
regions (sky vs. leaves). It is a half-way house between global SVD and JPEG,
which also works on 8 x 8 blocks.
"""
from __future__ import annotations

import numpy as np

from .core import to_float


def block_svd(img: np.ndarray, b: int, r: int) -> np.ndarray:
    """Rank-r approximation of every b x b tile, per channel."""
    img = to_float(img)
    H, W, C = img.shape
    Hp, Wp = -(-H // b) * b, -(-W // b) * b
    pad = np.pad(img, ((0, Hp - H), (0, Wp - W), (0, 0)), mode="edge")
    tiles = pad.reshape(Hp // b, b, Wp // b, b, C).transpose(0, 2, 4, 1, 3)  # (ty, tx, c, b, b)
    U, s, Vt = np.linalg.svd(tiles, full_matrices=False)
    rec = (U[..., :r] * s[..., None, :r]) @ Vt[..., :r, :]
    out = rec.transpose(0, 3, 1, 4, 2).reshape(Hp, Wp, C)[:H, :W]
    return np.clip(out, 0.0, 1.0)


def block_numbers(m: int, n: int, b: int, r: int, channels: int = 3) -> int:
    tiles = (-(-m // b)) * (-(-n // b))
    return channels * tiles * r * (2 * b + 1)
