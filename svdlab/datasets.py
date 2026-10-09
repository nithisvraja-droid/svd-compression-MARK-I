"""Test images. All ship with scikit-image, so nothing has to be downloaded.

They were picked to cover very different kinds of picture, because how well
SVD works depends far more on the picture than on the code.
"""
from __future__ import annotations

import numpy as np
from skimage import data
from skimage.color import gray2rgb

from .core import to_float

# name -> (loader, what kind of picture it is)
IMAGES = {
    "astronaut":  (data.astronaut,  "portrait (face)"),
    "coffee":     (data.coffee,     "smooth photo"),
    "chelsea":    (data.chelsea,    "fur texture"),
    "rocket":     (data.rocket,     "sky and structure"),
    "text":       (data.text,       "printed text"),
    "gravel":     (data.gravel,     "pure texture"),
    "colorwheel": (data.colorwheel, "flat synthetic"),
}


def load(name: str) -> np.ndarray:
    """Float RGB image in [0, 1]."""
    img = IMAGES[name][0]()
    if img.ndim == 2:
        img = gray2rgb(img)
    return to_float(img)[:, :, :3]


def all_images() -> dict[str, np.ndarray]:
    return {name: load(name) for name in IMAGES}
