"""Machine side of the human-vs-machine benchmark.

For every rank k the photo is rebuilt and three "machine eyes" look at it:

  1. Is there a face?     SCRFD detector confidence           (threshold 0.50)
  2. Who is it?           ArcFace cosine similarity to the    (threshold 0.40)
                          original face, aligned with the
                          original's landmarks so this tests
                          recognition alone
  3. Same as original?    SSIM against the original           (threshold 0.95)

    python scripts/fetch_models.py
    python scripts/machine_benchmark.py                 # astronaut test image
    python scripts/machine_benchmark.py my_photo.jpg    # your own photo

Writes results/11_machine_<name>.csv, figures/11_machine_<name>.png and
docs/benchmark/machine.js (the data the human test page compares against).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svdlab.core import decompose, reconstruct, storage_ratio, to_float  # noqa: E402
from svdlab.datasets import load  # noqa: E402
from svdlab.faces import FaceModels  # noqa: E402
from svdlab.metrics import psnr, rel_error, ssim  # noqa: E402

TH = {"face": 0.50, "identity": 0.40, "ssim": 0.95}


def main(path: str | None):
    if path:
        name = Path(path).stem
        img = to_float(np.asarray(Image.open(path).convert("RGB")))
    else:
        name, img = "astronaut", load("astronaut")
    m, n, _ = img.shape
    fm = FaceModels()
    svds = decompose(img)
    o255 = img * 255
    s0, _, kp0 = fm.detect(o255)
    if kp0 is None or s0 < TH["face"]:
        sys.exit("no clear face in the original photo")
    e0 = fm.embed(o255, kp0)

    ks = list(range(1, 101)) + [120, 150, 200, 250, 300]
    ks = [k for k in ks if k <= min(m, n)]
    rows = []
    for k in ks:
        r = reconstruct(svds, k)
        r255 = np.round(r * 255)
        score, _, _ = fm.detect(r255)
        rows.append({"k": k, "storage_pct": 100 * storage_ratio(m, n, k), "rel_err": rel_error(img, r),
                     "psnr": psnr(img, r), "ssim": ssim(img, r), "face_score": score,
                     "identity_cos": float(e0 @ fm.embed(r255, kp0))})

    first = lambda key, t: next((r["k"] for r in rows if r[key] >= t), None)  # noqa: E731
    kf, ki, ks_ = first("face_score", TH["face"]), first("identity_cos", TH["identity"]), first("ssim", TH["ssim"])
    print(f"{name}: face at k={kf}, identity at k={ki}, SSIM>=0.95 at k={ks_}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"11_machine_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()} for r in rows])

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for a, key, t, kk, title in [(ax[0], "face_score", TH["face"], kf, "1 · Is there a face?  (detector)"),
                                 (ax[1], "identity_cos", TH["identity"], ki, "2 · Who is it?  (ArcFace)"),
                                 (ax[2], "ssim", TH["ssim"], ks_, "3 · Same as original?  (SSIM)")]:
        a.plot([r["k"] for r in rows], [r[key] for r in rows], color="#eb6834")
        a.axhline(t, color="#7a8294", ls=":", lw=1)
        if kk:
            a.axvline(kk, color="#7a8294", ls="--", lw=1)
            a.text(kk * 1.08, 0.04, f"k = {kk}", fontsize=8, color="#4a5263")
        a.set(xscale="log", ylim=(0, 1.02), xlabel="rank k (log)", title=title)
    fig.savefig(ROOT / "figures" / f"11_machine_{name}.png", dpi=130, bbox_inches="tight")

    if not path:  # feed the human test page
        out = ROOT / "docs" / "benchmark" / "machine.js"
        out.parent.mkdir(parents=True, exist_ok=True)
        compact = [[r["k"], round(r["storage_pct"], 2), round(100 * r["rel_err"], 2), round(r["psnr"], 2),
                    round(r["ssim"], 4), round(r["face_score"], 3), round(r["identity_cos"], 3)] for r in rows]
        out.write_text(f"const IMAGE={json.dumps({'name': name, 'h': m, 'w': n})};\n"
                       f"const MACHINE={json.dumps(compact)};\n")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
