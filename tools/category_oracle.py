"""Upper bound of noise-type adaptive parameters (development only).

For a small grid of settings, report per noise category which setting is best, and the
'oracle' PESQ if every category could use its own best setting. If the oracle is close to
the best single setting, a noise-type classifier/rule is not worth building.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denoise  # noqa: E402
import evaluate as ev  # noqa: E402


def main():
    rows = ev.load_manifest("train")
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), 400, replace=False))]
    grid = list(product([1.2, 1.6, 2.2], [-28.0, -35.0, -42.0]))
    table = {}
    with ProcessPoolExecutor() as pool:
        for over, xm in grid:
            p = replace(denoise.DEFAULT, noise_overest=over, xi_min_db=xm)
            res = list(pool.map(ev._score_one, [(r, p, "proposed") for r in rows], chunksize=4))
            table[(over, xm)] = np.array([x["pesq"] for x in res])
            print(f"over {over} xi_min {xm}: PESQ {table[(over, xm)].mean():.4f}", flush=True)
    cats = np.array([r["noise_category"] for r in rows])
    best_single = max(v.mean() for v in table.values())
    total = 0.0
    print("\nPer category best setting:")
    for c in sorted(set(cats), key=lambda c: -(cats == c).sum()):
        m = cats == c
        scores = {k: v[m].mean() for k, v in table.items()}
        k = max(scores, key=scores.get)
        total += scores[k] * m.sum()
        ref = table[(1.6, -35.0)][m].mean()
        print(f"  {c:22s} n={m.sum():3d} best over={k[0]} xi_min={k[1]}: {scores[k]:.3f} "
              f"(default {ref:.3f}, {scores[k] - ref:+.3f})")
    print(f"\nbest single setting PESQ = {best_single:.4f}")
    print(f"oracle per-category PESQ = {total / len(rows):.4f}")


if __name__ == "__main__":
    main()
