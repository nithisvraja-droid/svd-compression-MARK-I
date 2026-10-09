"""Image quality metrics."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .core import to_float


def rel_error(ref: np.ndarray, img: np.ndarray) -> float:
    """Relative Frobenius error ||ref - img|| / ||ref||."""
    ref, img = to_float(ref), to_float(img)
    return float(np.linalg.norm(ref - img) / np.linalg.norm(ref))


def psnr(ref: np.ndarray, img: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (images in [0, 1])."""
    ref, img = to_float(ref), to_float(img)
    mse = float(np.mean((ref - img) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def ssim(ref: np.ndarray, img: np.ndarray) -> float:
    """Structural similarity (1 = identical)."""
    ref, img = to_float(ref), to_float(img)
    return float(structural_similarity(ref, img, channel_axis=2, data_range=1.0))


def all_metrics(ref: np.ndarray, img: np.ndarray) -> dict[str, float]:
    return {"rel_err": rel_error(ref, img), "psnr": psnr(ref, img), "ssim": ssim(ref, img)}
