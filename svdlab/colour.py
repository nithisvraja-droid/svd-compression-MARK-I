"""Three ways to handle colour, compared at the same storage budget.

* ``per_channel`` - SVD of R, G and B separately (the textbook approach).
* ``stacked``     - one SVD of the wide matrix [R | G | B] (m x 3n), so the
                    three channels share the same left singular vectors.
* ``ycbcr``       - convert to brightness (Y) + two colour-difference channels
                    and give brightness most of the layers. Human vision is far
                    more sensitive to brightness detail than to colour detail,
                    which is the same trick JPEG uses (chroma subsampling).
"""
from __future__ import annotations

import numpy as np

from .core import decompose, numbers_per_layer, reconstruct, to_float

# ITU-R BT.601 (full range), the matrix JPEG uses
_RGB2YCC = np.array([[0.299, 0.587, 0.114],
                     [-0.168736, -0.331264, 0.5],
                     [0.5, -0.418688, -0.081312]])
_YCC2RGB = np.linalg.inv(_RGB2YCC)


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    out = to_float(img) @ _RGB2YCC.T
    out[:, :, 1:] += 0.5
    return out


def ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    tmp = img.copy()
    tmp[:, :, 1:] -= 0.5
    return np.clip(tmp @ _YCC2RGB.T, 0.0, 1.0)


def budget_numbers(m: int, n: int, k: int) -> int:
    """Numbers used by the per-channel method at rank k (the shared budget)."""
    return 3 * k * numbers_per_layer(m, n)


def per_channel(img: np.ndarray, k: int) -> np.ndarray:
    return reconstruct(decompose(img), k)


def stacked(img: np.ndarray, k: int) -> np.ndarray:
    """Rank-k' SVD of [R | G | B] with k' chosen to match the per-channel budget."""
    img = to_float(img)
    m, n, _ = img.shape
    kk = max(1, budget_numbers(m, n, k) // (m + 3 * n + 1))
    wide = np.concatenate([img[:, :, c] for c in range(3)], axis=1)
    U, s, Vt = np.linalg.svd(wide, full_matrices=False)
    approx = (U[:, :kk] * s[:kk]) @ Vt[:kk]
    return np.clip(np.stack(np.split(approx, 3, axis=1), axis=2), 0.0, 1.0)


def ycbcr(img: np.ndarray, k: int, luma_share: float = 0.6) -> np.ndarray:
    """Same total budget (3k layers) split unevenly: Y gets ``luma_share`` of it."""
    ycc = rgb_to_ycbcr(img)
    total = 3 * k
    ky = max(1, round(total * luma_share))
    kc = max(1, (total - ky) // 2)
    rec = reconstruct(decompose(ycc), [ky, kc, kc])
    return ycbcr_to_rgb(rec)


METHODS = {"per_channel": per_channel, "stacked": stacked, "ycbcr": ycbcr}
