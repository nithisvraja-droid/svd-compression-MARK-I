"""A real file format for SVD-compressed images (.svdz).

Counting "numbers" hides an awkward fact: pixels are stored as 1-byte
integers, while numpy's SVD returns 8-byte floats. A rank-k SVD kept as
float64 is therefore 8x bigger than the number count suggests. To get a fair
file size we quantise U and V^T to 8-bit integers, one scale per column/row,
and keep the singular values as float32.

Layout per channel and per layer: m int8 + n int8 + 3 float32
(sigma, scale of u, scale of v). Everything is then zlib-compressed.
"""
from __future__ import annotations

import io
import zlib

import numpy as np

from .core import ChannelSVD, decompose, to_float


def _quant(x: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(x).max(axis=axis, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(x / scale).astype(np.int8)
    return q, scale.astype(np.float32).squeeze(axis)


def quantise(svd: ChannelSVD, k: int) -> dict[str, np.ndarray]:
    qu, su = _quant(svd.U[:, :k], axis=0)
    qv, sv = _quant(svd.Vt[:k], axis=1)
    return {"U": qu, "su": su, "s": svd.s[:k].astype(np.float32), "V": qv, "sv": sv}


def dequantise(q: dict[str, np.ndarray]) -> np.ndarray:
    U = q["U"].astype(np.float32) * q["su"][None, :]
    Vt = q["V"].astype(np.float32) * q["sv"][:, None]
    return (U * q["s"]) @ Vt


def raw_bytes(m: int, n: int, k: int, channels: int) -> int:
    """Uncompressed payload size of the quantised format."""
    return channels * k * (m + n + 3 * 4)


def encode(img: np.ndarray, k: int) -> bytes:
    """Compress an image to .svdz bytes."""
    svds = decompose(img)
    buf = io.BytesIO()
    arrays = {}
    for c, d in enumerate(svds):
        for key, val in quantise(d, k).items():
            arrays[f"{c}_{key}"] = val
    np.savez(buf, **arrays)
    return zlib.compress(buf.getvalue(), level=9)


def decode(data: bytes) -> np.ndarray:
    """Rebuild a float image in [0, 1] from .svdz bytes."""
    z = np.load(io.BytesIO(zlib.decompress(data)))
    chans = sorted({key.split("_")[0] for key in z.files}, key=int)
    out = [dequantise({f: z[f"{c}_{f}"] for f in ("U", "su", "s", "V", "sv")}) for c in chans]
    return np.clip(np.stack(out, axis=2), 0.0, 1.0)


def quantised_reconstruct(img: np.ndarray, k: int) -> np.ndarray:
    """Round-trip without touching disk (for experiments)."""
    return decode(encode(to_float(img), k))
