"""Core truncated-SVD image compression.

An image channel is an m x n matrix A. Its SVD is A = U diag(s) V^T with the
singular values s sorted from largest to smallest. Keeping the first k terms

    A_k = s_1 u_1 v_1^T + ... + s_k u_k v_k^T

gives the best rank-k approximation of A in both the Frobenius and the
spectral norm (Eckart-Young-Mirsky theorem).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelSVD:
    """Thin SVD of one channel: A = U @ diag(s) @ Vt."""

    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.U.shape[0], self.Vt.shape[1]

    def reconstruct(self, k: int) -> np.ndarray:
        k = int(np.clip(k, 0, len(self.s)))
        return (self.U[:, :k] * self.s[:k]) @ self.Vt[:k]


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 or float image -> float64 in [0, 1], always H x W x C."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float64) / 255.0
    else:
        a = a.astype(np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] == 4:  # drop alpha
        a = a[:, :, :3]
    return a


def to_uint8(img: np.ndarray) -> np.ndarray:
    a = np.clip(img, 0.0, 1.0)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return (a * 255.0 + 0.5).astype(np.uint8)


def decompose(img: np.ndarray) -> list[ChannelSVD]:
    """Thin SVD of every colour channel of a float image."""
    img = to_float(img)
    out = []
    for c in range(img.shape[2]):
        U, s, Vt = np.linalg.svd(img[:, :, c], full_matrices=False)
        out.append(ChannelSVD(U, s, Vt))
    return out


def reconstruct(svds: list[ChannelSVD], k: int | list[int]) -> np.ndarray:
    """Rebuild an H x W x C image from per-channel SVDs, keeping k layers.

    ``k`` may be one int for every channel or a list with one k per channel.
    """
    ks = [k] * len(svds) if np.isscalar(k) else list(k)
    return np.clip(np.stack([d.reconstruct(kk) for d, kk in zip(svds, ks)], axis=2), 0.0, 1.0)


def compress(img: np.ndarray, k: int) -> np.ndarray:
    """One-call helper: rank-k reconstruction of an image (float, [0, 1])."""
    return reconstruct(decompose(img), k)


# ---------------------------------------------------------------- storage

def numbers_per_layer(m: int, n: int) -> int:
    """One layer = one column of U (m), one row of V^T (n), one sigma."""
    return m + n + 1


def storage_numbers(m: int, n: int, k: int, channels: int = 1) -> int:
    return channels * k * numbers_per_layer(m, n)


def storage_ratio(m: int, n: int, k: int) -> float:
    """Numbers stored for rank k divided by numbers in the raw channel."""
    return k * numbers_per_layer(m, n) / (m * n)


def break_even_rank(m: int, n: int, bytes_per_number: float = 1.0,
                    bytes_per_pixel: float = 1.0) -> int:
    """Largest k for which the SVD form is still smaller than the raw image.

    With the textbook count (every number costs the same) this is
    floor(mn / (m + n + 1)). Real files store pixels as 1-byte integers, so a
    float32 SVD (4 bytes per number) breaks even four times sooner.
    """
    return int((m * n * bytes_per_pixel) // (numbers_per_layer(m, n) * bytes_per_number))


# ---------------------------------------------------------------- energy / theory

def energy(s: np.ndarray) -> np.ndarray:
    """Cumulative energy captured: (s_1^2 + ... + s_k^2) / sum(s^2), k = 1..r."""
    e = np.cumsum(np.asarray(s, dtype=np.float64) ** 2)
    return e / e[-1]


def rank_for_energy(s: np.ndarray, target: float) -> int:
    """Smallest k whose layers capture at least ``target`` of the energy."""
    return int(np.searchsorted(energy(s), target) + 1)


def eckart_young_error(s: np.ndarray, k: int) -> tuple[float, float]:
    """Predicted errors of the best rank-k approximation.

    Returns (Frobenius, spectral) = (sqrt(s_{k+1}^2 + ... + s_r^2), s_{k+1}).
    """
    tail = np.asarray(s[k:], dtype=np.float64)
    return float(np.sqrt(np.sum(tail ** 2))), float(tail[0]) if tail.size else 0.0


# ---------------------------------------------------------------- randomized SVD

def randomized_svd(A: np.ndarray, k: int, oversample: int = 10, power_iters: int = 2,
                   seed: int | None = 0) -> ChannelSVD:
    """Top-k SVD by random projection (Halko, Martinsson & Tropp, 2011).

    Much faster than a full SVD when only the first few layers are needed.
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    p = min(k + oversample, min(m, n))
    Y = A @ rng.standard_normal((n, p))
    for _ in range(power_iters):  # sharpen the spectrum
        Y, _ = np.linalg.qr(Y)
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    Ub, s, Vt = np.linalg.svd(Q.T @ A, full_matrices=False)
    return ChannelSVD((Q @ Ub)[:, :k], s[:k], Vt[:k])
