"""Command line tool.

    python -m svdlab info       photo.jpg
    python -m svdlab compress   photo.jpg --k 50          -> photo_k50.png + photo_k50.svdz
    python -m svdlab compress   photo.jpg --energy 0.99
    python -m svdlab compress   photo.jpg --psnr 30
    python -m svdlab decompress photo_k50.svdz            -> photo_k50.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import codec
from .core import break_even_rank, decompose, energy, rank_for_energy, reconstruct, to_float, to_uint8
from .metrics import all_metrics


def _load(path: str) -> np.ndarray:
    return to_float(np.asarray(Image.open(path).convert("RGB")))


def _rank_for_psnr(img, svds, target: float) -> int:
    lo, hi = 1, len(svds[0].s)
    while lo < hi:  # PSNR grows with k, so binary search works
        mid = (lo + hi) // 2
        if all_metrics(img, reconstruct(svds, mid))["psnr"] >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def cmd_info(a):
    img = _load(a.image)
    m, n, _ = img.shape
    svds = decompose(img)
    s = np.sqrt(sum(d.s ** 2 for d in svds))  # combined spectrum
    print(f"{a.image}: {n} x {m} px, rank {len(s)}")
    print(f"break-even rank (counting numbers): k = {break_even_rank(m, n)}")
    print(f"break-even rank (float32 arrays):    k = {break_even_rank(m, n, 4.0)}")
    print(f"break-even rank (8-bit .svdz file):  k = {(m * n) // (m + n + 12)}")
    for t in (0.90, 0.95, 0.99, 0.999):
        print(f"  {t:6.1%} energy at k = {rank_for_energy(s, t)}")


def cmd_compress(a):
    img = _load(a.image)
    m, n, c = img.shape
    svds = decompose(img)
    if a.k:
        k = a.k
    elif a.energy:
        k = rank_for_energy(np.sqrt(sum(d.s ** 2 for d in svds)), a.energy)
    else:
        k = _rank_for_psnr(img, svds, a.psnr)
    data = codec.encode(img, k)
    rec = codec.decode(data)
    stem = Path(a.out or Path(a.image).with_suffix("")).as_posix() + f"_k{k}"
    Path(stem + ".svdz").write_bytes(data)
    Image.fromarray(to_uint8(rec)).save(stem + ".png")
    q = all_metrics(img, rec)
    print(f"k = {k}  ({energy(svds[0].s)[k - 1]:.1%} energy in channel 1)")
    print(f"raw pixels {m * n * c:,} B -> .svdz {len(data):,} B  ({len(data) / (m * n * c):.1%})")
    print(f"PSNR {q['psnr']:.2f} dB   SSIM {q['ssim']:.4f}   error {q['rel_err']:.2%}")
    print(f"wrote {stem}.svdz and {stem}.png")


def cmd_decompress(a):
    rec = codec.decode(Path(a.file).read_bytes())
    out = a.out or os.path.splitext(a.file)[0] + ".png"
    Image.fromarray(to_uint8(rec)).save(out)
    print(f"wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="svdlab", description="SVD image compression")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="singular-value summary of an image")
    s.add_argument("image")
    s.set_defaults(fn=cmd_info)
    s = sub.add_parser("compress", help="compress to .svdz (+ preview .png)")
    s.add_argument("image")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--k", type=int, help="number of layers to keep")
    g.add_argument("--energy", type=float, help="keep this fraction of the energy, e.g. 0.99")
    g.add_argument("--psnr", type=float, help="smallest k reaching this PSNR (dB)")
    s.add_argument("--out", help="output path stem")
    s.set_defaults(fn=cmd_compress)
    s = sub.add_parser("decompress", help="decode a .svdz file to .png")
    s.add_argument("file")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_decompress)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
