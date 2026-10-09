"""Run every experiment and write results/*.csv and figures/*.png.

    python scripts/run_experiments.py            # everything (~3-5 min on a laptop)
    python scripts/run_experiments.py --quick    # fewer ranks, for CI
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svdlab import codec, colour  # noqa: E402
from svdlab.blocks import block_numbers, block_svd  # noqa: E402
from svdlab.core import (break_even_rank, decompose, eckart_young_error, energy,  # noqa: E402
                         randomized_svd, rank_for_energy, reconstruct, storage_ratio, to_uint8)
from svdlab.datasets import IMAGES, all_images  # noqa: E402
from svdlab.denoise import add_noise, denoise  # noqa: E402
from svdlab.jpeg import jpeg_at_size  # noqa: E402
from svdlab.metrics import all_metrics, psnr, ssim  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# one consistent look for every figure
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e3e6ec", "grid.linewidth": 0.8, "axes.edgecolor": "#9aa1ad",
    "axes.prop_cycle": plt.cycler(color=PALETTE), "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def write_csv(name: str, rows: list[dict]):
    with open(RES / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  results/{name}.csv")


def save(fig, name: str):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  figures/{name}.png")


def main(quick: bool):
    t0 = time.time()
    imgs = all_images()
    svds = {n: decompose(im) for n, im in imgs.items()}
    KS = [1, 2, 5, 10, 20, 50, 100] if quick else [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

    # ---------------------------------------------------------------- 1 spectrum + energy
    print("1. singular values and energy")
    rows = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in svds.items():
        s = np.sqrt(sum(c.s ** 2 for c in d))  # combined over R, G, B
        a1.semilogy(np.arange(1, len(s) + 1), s / s[0], label=name)
        e = energy(s)
        a2.plot(np.arange(1, len(e) + 1), e, label=name)
        m, n, _ = imgs[name].shape
        rows.append({"image": name, "kind": IMAGES[name][1], "height": m, "width": n,
                     "k_90": rank_for_energy(s, .90), "k_95": rank_for_energy(s, .95),
                     "k_99": rank_for_energy(s, .99), "k_999": rank_for_energy(s, .999),
                     "break_even_k": break_even_rank(m, n)})
    a1.set(title="Singular values fall off fast", xlabel="layer i", ylabel="σᵢ / σ₁  (log scale)", xlim=(1, 300), ylim=(1e-5, 1.5))
    a2.set(title="Energy captured by the first k layers", xlabel="k", ylabel="energy", xlim=(1, 100), ylim=(0.7, 1.0))
    a2.legend(ncol=2, fontsize=8, loc="lower right")
    save(fig, "01_spectrum_energy")
    write_csv("01_energy", rows)

    # ---------------------------------------------------------------- 2 Eckart-Young check
    print("2. Eckart-Young check")
    rows = []
    A = imgs["astronaut"][:, :, 0]
    d = svds["astronaut"][0]
    for k in [1, 5, 10, 20, 50, 100, 200]:
        R = A - d.reconstruct(k)
        fro_pred, spec_pred = eckart_young_error(d.s, k)
        rows.append({"k": k, "frobenius_measured": float(np.linalg.norm(R)), "frobenius_predicted": fro_pred,
                     "spectral_measured": float(np.linalg.norm(R, 2)), "spectral_predicted": spec_pred})
    write_csv("02_eckart_young", rows)
    worst = max(abs(r["frobenius_measured"] - r["frobenius_predicted"]) for r in rows)
    print(f"  worst |measured - predicted| Frobenius gap: {worst:.2e}")

    # ---------------------------------------------------------------- 3 gallery, layers, residual
    print("3. gallery, layers, residual")
    img, d = imgs["astronaut"], svds["astronaut"]
    m, n, _ = img.shape
    shows = [1, 5, 20, 50]
    fig, ax = plt.subplots(1, len(shows) + 1, figsize=(15, 3.4))
    for a, k in zip(ax, shows):
        rec = reconstruct(d, k)
        q = all_metrics(img, rec)
        a.imshow(rec)
        a.set_title(f"k = {k}\n{storage_ratio(m, n, k):.1%} storage · PSNR {q['psnr']:.1f} dB", fontsize=9)
    ax[-1].imshow(img)
    ax[-1].set_title("original\n100%", fontsize=9)
    for a in ax:
        a.axis("off")
    save(fig, "03_gallery")

    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    run = np.zeros_like(img)
    for i in range(4):
        layer = np.stack([(c.U[:, i:i + 1] * c.s[i]) @ c.Vt[i:i + 1] for c in d], 2)
        run = run + layer
        lv = layer - layer.min()
        ax[0, i].imshow(lv / max(lv.max(), 1e-9))
        ax[0, i].set_title(f"layer {i + 1} alone  (σ ≈ {np.mean([c.s[i] for c in d]):.0f})", fontsize=9)
        ax[1, i].imshow(np.clip(run, 0, 1))
        ax[1, i].set_title("layer 1" if i == 0 else f"sum of layers 1–{i + 1}", fontsize=9)
    for a in ax.flat:
        a.axis("off")
    save(fig, "04_layers")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    rec = reconstruct(d, 20)
    resid = img - rec
    ax[0].imshow(img); ax[0].set_title("original")
    ax[1].imshow(rec); ax[1].set_title("k = 20 (kept)")
    ax[2].imshow(np.clip(0.5 + 3 * resid, 0, 1)); ax[2].set_title("residual × 3 (thrown away)")
    for a in ax:
        a.axis("off")
    save(fig, "05_residual")

    # ---------------------------------------------------------------- 4 quality vs storage, all images
    print("4. quality vs storage for every image")
    rows = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, im in imgs.items():
        m, n, _ = im.shape
        xs, ps, ss = [], [], []
        for k in KS:
            if k > min(m, n):
                continue
            rec = reconstruct(svds[name], k)
            q = all_metrics(im, rec)
            xs.append(100 * storage_ratio(m, n, k)); ps.append(q["psnr"]); ss.append(q["ssim"])
            rows.append({"image": name, "k": k, "storage_pct": xs[-1], **q})
        a1.plot(xs, ps, marker="o", ms=3, label=name)
        a2.plot(xs, ss, marker="o", ms=3, label=name)
    for a in (a1, a2):
        a.axvline(100, color="#7a8294", ls="--", lw=1)
        a.text(101, a.get_ylim()[0], " break-even", color="#7a8294", fontsize=8, va="bottom")
        a.set_xlabel("storage (% of raw numbers)")
        a.set_xscale("log")
    a1.set(title="PSNR vs storage", ylabel="PSNR (dB)")
    a2.set(title="SSIM vs storage", ylabel="SSIM")
    a2.legend(fontsize=8, ncol=2)
    save(fig, "06_quality_vs_storage")
    write_csv("04_quality", rows)

    # ---------------------------------------------------------------- 5 colour handling
    print("5. colour handling at equal budget")
    rows = []
    ck = [5, 10, 20, 40] if quick else [2, 5, 10, 20, 40, 80]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, name in zip(ax, ["astronaut", "coffee", "rocket"]):
        im = imgs[name]
        for meth, fn in colour.METHODS.items():
            vals = []
            for k in ck:
                q = all_metrics(im, fn(im, k))
                vals.append(q["psnr"])
                rows.append({"image": name, "k": k, "method": meth, **q})
            a.plot(ck, vals, marker="o", ms=3, label=meth.replace("_", "-"))
        a.set(title=name, xlabel="per-channel k (same storage for all three)", xscale="log")
        a.set_xticks(ck, [str(k) for k in ck])
        a.minorticks_off()
    ax[0].set_ylabel("PSNR (dB)")
    ax[0].legend()
    save(fig, "07_colour_modes")
    write_csv("05_colour", rows)

    # ---------------------------------------------------------------- 6 quantisation
    print("6. 8-bit quantisation")
    rows = []
    for name in ["astronaut", "coffee", "chelsea"]:
        im = imgs[name]
        m, n, c = im.shape
        for k in [10, 20, 50, 100]:
            exact = reconstruct(svds[name], k)
            data = codec.encode(im, k)
            rec = codec.decode(data)
            rows.append({"image": name, "k": k,
                         "float64_bytes": c * k * (m + n + 1) * 8, "float32_bytes": c * k * (m + n + 1) * 4,
                         "svdz_bytes": len(data), "raw_pixels_bytes": m * n * c,
                         "psnr_float": psnr(im, exact), "psnr_8bit": psnr(im, rec),
                         "psnr_loss_db": psnr(im, exact) - psnr(im, rec)})
    write_csv("06_quantisation", rows)

    # ---------------------------------------------------------------- 7 block SVD
    print("7. block SVD vs global SVD")
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for a, name in zip(ax, ["astronaut", "rocket"]):
        im = imgs[name]
        m, n, _ = im.shape
        g_x, g_y = [], []
        for k in KS:
            if k > min(m, n):
                continue
            g_x.append(100 * storage_ratio(m, n, k)); g_y.append(psnr(im, reconstruct(svds[name], k)))
        a.plot(g_x, g_y, marker="o", ms=3, label="global SVD")
        for b in (8, 16, 32):
            bx, by = [], []
            for r in range(1, min(b - 1, 12) + 1):  # r = b would be lossless
                num = block_numbers(m, n, b, r)
                bx.append(100 * num / (3 * m * n)); by.append(psnr(im, block_svd(im, b, r)))
                rows.append({"image": name, "block": b, "rank": r, "storage_pct": bx[-1], "psnr": by[-1]})
            a.plot(bx, by, marker="o", ms=3, label=f"{b}×{b} tiles")
        a.set(title=name, xlabel="storage (% of raw numbers)", ylabel="PSNR (dB)", xscale="log", xlim=(0.4, 120), ylim=(None, 50))
        a.legend(fontsize=8)
    save(fig, "08_block_svd")
    write_csv("07_block_svd", rows)

    # ---------------------------------------------------------------- 8 SVD vs JPEG
    print("8. SVD (.svdz) vs JPEG at the same file size")
    rows = []
    jk = [5, 20, 50] if quick else [5, 10, 20, 50, 100]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.4))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    for a, name in zip(ax.flat, imgs):
        im = imgs[name]
        sx, sy, jx, jy = [], [], [], []
        for k in jk:
            data = codec.encode(im, k)
            rec = codec.decode(data)
            jrec, quality, jb = jpeg_at_size(im, len(data))
            rows.append({"image": name, "k": k, "bytes": len(data), "svd_psnr": psnr(im, rec),
                         "svd_ssim": ssim(im, rec), "jpeg_quality": quality, "jpeg_bytes": jb,
                         "jpeg_psnr": psnr(im, jrec), "jpeg_ssim": ssim(im, jrec)})
            sx.append(len(data) / 1024); sy.append(rows[-1]["svd_psnr"])
            jx.append(jb / 1024); jy.append(rows[-1]["jpeg_psnr"])
        a.plot(sx, sy, marker="o", ms=3, label="SVD (.svdz)")
        a.plot(jx, jy, marker="s", ms=3, label="JPEG")
        a.set(title=f"{name} · {IMAGES[name][1]}", xlabel="file size (KB)", ylabel="PSNR (dB)")
    ax.flat[0].legend()
    ax.flat[-1].axis("off")
    save(fig, "09_svd_vs_jpeg")
    write_csv("08_svd_vs_jpeg", rows)

    # ---------------------------------------------------------------- 9 denoising
    print("9. denoising")
    rows = []
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    im = imgs["astronaut"]
    noisy = add_noise(im, 0.10)
    nd = decompose(noisy)
    kr = list(range(1, 151, 2)) if not quick else [5, 20, 40, 80]
    pv = [psnr(im, reconstruct(nd, k)) for k in kr]
    best_k = kr[int(np.argmax(pv))]
    auto, auto_ks = denoise(noisy)
    for k, p in zip(kr, pv):
        rows.append({"k": k, "psnr_vs_clean": p})
    rows.append({"k": f"auto {auto_ks}", "psnr_vs_clean": psnr(im, auto)})
    write_csv("09_denoise", rows)
    ax[0].imshow(noisy); ax[0].set_title(f"noisy (σ = 0.10)\nPSNR {psnr(im, noisy):.1f} dB")
    ax[1].imshow(auto); ax[1].set_title(f"auto rank {auto_ks[0]} (Gavish–Donoho)\nPSNR {psnr(im, auto):.1f} dB")
    ax[2].imshow(im); ax[2].set_title("clean original")
    for a in ax[:3]:
        a.axis("off")
    ax[3].plot(kr, pv)
    ax[3].axvline(best_k, color="#7a8294", ls="--", lw=1)
    ax[3].axvline(auto_ks[0], color=PALETTE[1], ls=":", lw=1.5)
    ax[3].text(best_k + 2, min(pv), f" best k = {best_k}", fontsize=8, color="#4a5263")
    ax[3].set(title="PSNR against the clean image", xlabel="k", ylabel="PSNR (dB)")
    save(fig, "10_denoise")

    # ---------------------------------------------------------------- 10 randomized SVD
    print("10. randomized SVD speed")
    rows = []
    from skimage import data as skdata
    big = skdata.hubble_deep_field()[:, :, 0].astype(np.float64) / 255
    t = time.perf_counter(); U, s, Vt = np.linalg.svd(big, full_matrices=False); t_full = time.perf_counter() - t
    for k in [10, 50, 100]:
        t = time.perf_counter(); r = randomized_svd(big, k); t_r = time.perf_counter() - t
        err_full = np.linalg.norm(big - (U[:, :k] * s[:k]) @ Vt[:k])
        err_r = np.linalg.norm(big - r.reconstruct(k))
        rows.append({"matrix": f"{big.shape[0]}x{big.shape[1]}", "k": k, "full_svd_s": t_full,
                     "randomized_s": t_r, "speedup": t_full / t_r, "error_ratio": err_r / err_full})
    write_csv("10_randomized_svd", rows)

    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
