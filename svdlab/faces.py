"""Machine "eyes" for the human-vs-machine benchmark.

* Detection: SCRFD-500M face detector         (InsightFace ``det_500m.onnx``)
* Identity : ArcFace MobileFaceNet, WebFace600K (InsightFace ``w600k_mbf.onnx``)

Both come from InsightFace's ``buffalo_sc`` model pack, which is released for
non-commercial research use. Run ``python scripts/fetch_models.py`` once to
download it into ``models/``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models" / "buffalo_sc"

# 5-point template for 112 x 112 ArcFace crops (eyes, nose, mouth corners)
_REF = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                 [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


class FaceModels:
    def __init__(self, model_dir: Path | str = MODELS, det_size: int = 640):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        opts = {"providers": ["CPUExecutionProvider"]}
        self.det = ort.InferenceSession(str(model_dir / "det_500m.onnx"), **opts)
        self.rec = ort.InferenceSession(str(model_dir / "w600k_mbf.onnx"), **opts)
        self.size = det_size

    # -- detection -----------------------------------------------------------
    def detect(self, rgb: np.ndarray) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        """Best face in an RGB image (0-255 floats or uint8).

        Returns (confidence, box [x1 y1 x2 y2], 5 landmarks).
        """
        rgb = np.asarray(rgb, np.float32)
        h, w = rgb.shape[:2]
        sc = self.size / max(h, w)
        rs = cv2.resize(rgb, (int(w * sc), int(h * sc)))
        pad = np.zeros((self.size, self.size, 3), np.float32)
        pad[: rs.shape[0], : rs.shape[1]] = rs
        blob = ((pad - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.det.run(None, {self.det.get_inputs()[0].name: blob})
        best = (0.0, None, None)
        for i, stride in enumerate((8, 16, 32)):
            scores = outs[i][:, 0]
            boxes, kps = outs[i + 3] * stride, outs[i + 6] * stride
            g = self.size // stride
            cy, cx = np.mgrid[:g, :g]
            anchors = np.stack([cx, cy], -1).reshape(-1, 2).repeat(2, 0) * stride
            j = int(np.argmax(scores))
            if scores[j] > best[0]:
                a = anchors[j]
                box = np.r_[a - boxes[j, :2], a + boxes[j, 2:]] / sc
                best = (float(scores[j]), box, (a + kps[j].reshape(5, 2)) / sc)
        return best

    # -- identity ------------------------------------------------------------
    def embed(self, rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Unit-length 512-d ArcFace embedding of the face at ``landmarks``."""
        M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), _REF)
        face = cv2.warpAffine(np.asarray(rgb, np.float32), M, (112, 112), borderValue=0)
        blob = ((face - 127.5) / 127.5).transpose(2, 0, 1)[None].astype(np.float32)
        e = self.rec.run(None, {self.rec.get_inputs()[0].name: blob})[0][0]
        return e / np.linalg.norm(e)
