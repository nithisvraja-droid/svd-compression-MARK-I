"""JPEG baseline: find the JPEG quality whose file is closest to a byte budget."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .core import to_float, to_uint8


def jpeg_bytes(img: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(to_uint8(img)).save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def jpeg_decode(data: bytes) -> np.ndarray:
    return to_float(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))


def jpeg_at_size(img: np.ndarray, target_bytes: int) -> tuple[np.ndarray, int, int]:
    """Largest-quality JPEG that fits in ``target_bytes``.

    Returns (decoded image, quality used, actual bytes). If even quality 1
    does not fit, quality 1 is returned.
    """
    lo, hi, best = 1, 100, None
    while lo <= hi:
        q = (lo + hi) // 2
        data = jpeg_bytes(img, q)
        if len(data) <= target_bytes:
            best, lo = (q, data), q + 1
        else:
            hi = q - 1
    if best is None:
        best = (1, jpeg_bytes(img, 1))
    q, data = best
    return jpeg_decode(data), q, len(data)
