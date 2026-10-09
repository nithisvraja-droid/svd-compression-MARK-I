"""Analyse human test results and compare them with the machine.

Every finished test on docs/benchmark/ ends with a result code. Collect the
codes in a text file (one per line) and run:

    python scripts/analyse_human.py codes.txt

Prints the human thresholds next to the machine ones and writes
figures/12_human_vs_machine.png.
"""
from __future__ import annotations

import base64
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LADDER = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50]
AFC = [10, 20, 30, 50, 75, 100, 150, 200]


def load_runs(path: str) -> list[dict]:
    runs = []
    for line in Path(path).read_text().split():
        try:
            r = json.loads(base64.b64decode(line))
            if "afc" in r and "kFace" in r:
                runs.append(r)
        except Exception:  # noqa: BLE001
            print(f"skipped unreadable code: {line[:20]}...")
    return runs


def machine(name: str = "astronaut") -> list[dict]:
    with open(ROOT / "results" / f"11_machine_{name}.csv") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main(path: str, include_seen: bool = False):
    runs = load_runs(path)
    if not include_seen:
        runs = [r for r in runs if not r.get("seen")]
    if not runs:
        sys.exit("no usable runs")
    n = len(runs)
    inf = float("inf")
    k_face = statistics.median(r["kFace"] if r["kFace"] is not None else inf for r in runs)
    k_rec = statistics.median(r["kRec"] if r["kRec"] is not None else inf for r in runs)
    acc = {}
    for k in AFC:
        trials = [t for r in runs for t in r["afc"] if t[0] == k]
        acc[k] = sum(t[1] for t in trials) / len(trials) if trials else None
    k_same = next((k for k in AFC if acc[k] is not None and acc[k] <= 0.75), inf)

    M = machine()
    first = lambda key, t: next((int(r["k"]) for r in M if r[key] >= t), None)  # noqa: E731
    print(f"{n} participants")
    print(f"{'task':28s}{'humans':>10s}{'machine':>10s}")
    print(f"{'is there a face?':28s}{k_face:>10}{first('face_score', .5):>10}")
    print(f"{'who is it?':28s}{k_rec:>10}{first('identity_cos', .4):>10}")
    print(f"{'same as the original?':28s}{k_same:>10}{first('ssim', .95):>10}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    face_curve = [sum(r["kFace"] is not None and r["kFace"] <= k for r in runs) / n for k in LADDER]
    rec_curve = [sum(r["kRec"] is not None and r["kRec"] <= k for r in runs) / n for k in LADDER]
    mk = [r["k"] for r in M]
    for a, hx, hy, key, title in [(ax[0], LADDER, face_curve, "face_score", "Is there a face?"),
                                  (ax[1], LADDER, rec_curve, "identity_cos", "Who is it?"),
                                  (ax[2], AFC, [acc[k] for k in AFC], "ssim", "Spot the original")]:
        a.plot(hx, hy, "o-", color="#2a78d6", label="humans")
        a.plot(mk, [r[key] for r in M], color="#eb6834", label="machine")
        a.set(xscale="log", ylim=(0, 1.02), title=title, xlabel="rank k (log)")
    ax[2].axhline(0.5, color="#7a8294", ls=":", lw=1)
    ax[0].legend(frameon=False)
    fig.savefig(ROOT / "figures" / "12_human_vs_machine.png", dpi=130, bbox_inches="tight")
    print("wrote figures/12_human_vs_machine.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], "--include-seen" in sys.argv)
