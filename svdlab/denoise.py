"""Denoising by truncation.

Random noise spreads its energy thinly over *all* singular values, while the
picture lives in the first few. Dropping the small singular values therefore
removes more noise than picture.

``optimal_rank`` picks the cut-off automatically with the optimal hard
threshold of Gavish & Donoho (2014) for unknown noise level:
    tau = omega(beta) * median(singular values),  beta = m/n (<= 1)
"""
from __future__ import annotations

import numpy as np

from .core import decompose, reconstruct, to_float


def add_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(to_float(img) + rng.normal(0.0, sigma, to_float(img).shape), 0.0, 1.0)


def omega(beta: float) -> float:
    """Gavish-Donoho approximation of the optimal threshold coefficient."""
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def optimal_rank(s: np.ndarray, m: int, n: int) -> int:
    beta = min(m, n) / max(m, n)
    tau = omega(beta) * np.median(s)
    return max(1, int(np.sum(s > tau)))


def denoise(noisy: np.ndarray, k: int | None = None) -> tuple[np.ndarray, list[int]]:
    """Truncate each channel; with k=None the rank is chosen per channel."""
    svds = decompose(noisy)
    if k is None:
        ks = [optimal_rank(d.s, *d.shape) for d in svds]
    else:
        ks = [k] * len(svds)
    return reconstruct(svds, ks), ks
