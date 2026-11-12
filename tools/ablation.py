"""Ablation of the final pipeline: remove one component at a time (development only).

For validation and the full train split, reports mean PESQ / SNR, the change against the full
system, and robustness counts (files worse than the noisy input; files worse than the
full system by more than 0.1 PESQ).
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denoise  # noqa: E402
import evaluate as ev  # noqa: E402

D = denoise.DEFAULT
VARIANTS = {
    "full system": {},
    "- multi-resolution fusion": dict(mr_mode="off"),
    "- modulation post-filter": dict(mod_mode="off"),
    "- multi-band parameters": dict(band_over_mid=1.0, band_over_hi=1.0,
                                    band_ximin_lo_db=0.0, band_ximin_mid_db=0.0),
    "- HRNR": dict(hrnr=False),
    "- quantile noise estimate": dict(quantile_win_s=0.0),
    "- noise over-estimation (x1.0)": dict(noise_overest=1.0),
}


def run(pool, rows, params, method="proposed"):
    res = list(pool.map(ev._score_one, [(r, params, method) for r in rows], chunksize=4))
    return np.array([x["pesq"] for x in res]), np.array([x["snr"] for x in res])


def main():
    with ProcessPoolExecutor() as pool:
        for split in ("validation", "train"):
            rows = ev.load_manifest(split)
            noisy_pq, _ = run(pool, rows, D, "noisy")
            full_pq = None
            print(f"\n== {split} (n={len(rows)}) | noisy PESQ {noisy_pq.mean():.4f}", flush=True)
            for name, kw in VARIANTS.items():
                pq, sn = run(pool, rows, replace(D, **kw))
                if full_pq is None:
                    full_pq = pq
                worse_in = int((pq < noisy_pq).sum())
                worse_full = int((pq < full_pq - 0.1).sum())
                print(f"  {name:32s} PESQ {pq.mean():.4f} ({pq.mean() - full_pq.mean():+.4f})"
                      f"  SNR {sn.mean():6.2f}  worse-than-noisy {worse_in:3d}"
                      f"  >0.1 below full {worse_full:3d}", flush=True)


if __name__ == "__main__":
    main()
