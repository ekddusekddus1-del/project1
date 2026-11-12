"""Coordinate-wise parameter search on a train subset (development only).

python tools/tune.py --subset 400 --grid "gain_floor_db=[-10,-15,-20,-25]" --grid "dd_alpha=[0.95,0.98]"
Each --grid line is searched one at a time (coordinate descent), keeping the best value.
"""
from __future__ import annotations

import argparse
import ast
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denoise  # noqa: E402
import evaluate as ev  # noqa: E402


def run(pool, rows, params):
    res = list(pool.map(ev._score_one, [(r, params, "proposed") for r in rows], chunksize=4))
    return ev.summarize(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="append", required=True)
    ap.add_argument("--set", action="append")
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()
    rows = ev.load_manifest(a.split)
    if a.subset and a.subset < len(rows):
        rng = np.random.default_rng(a.seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), a.subset, replace=False))]
    params = ev.parse_sets(a.set)
    grids = []
    for g in a.grid:
        k, v = g.split("=", 1)
        grids.append((k, ast.literal_eval(v)))
    with ProcessPoolExecutor() as pool:
        best = run(pool, rows, params)
        print(f"start PESQ={best['pesq']:.4f} SNR={best['snr']:.2f}", flush=True)
        for _ in range(a.rounds):
            for k, values in grids:
                for v in values:
                    if getattr(params, k) == v:
                        continue
                    cand = replace(params, **{k: v})
                    s = run(pool, rows, cand)
                    mark = ""
                    if s["pesq"] > best["pesq"]:
                        best, params, mark = s, cand, "  *"
                    print(f"  {k}={v!r:10} PESQ={s['pesq']:.4f} SNR={s['snr']:.2f}{mark}", flush=True)
                print(f"best {k}={getattr(params, k)!r}", flush=True)
    print("FINAL", {k: getattr(params, k) for k, _ in grids}, f"PESQ={best['pesq']:.4f}")


if __name__ == "__main__":
    main()
