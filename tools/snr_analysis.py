"""Does the best (gain floor, over-estimation) depend on the input SNR? (development only)

Evaluates a small grid on a train subset and prints, per target-SNR group, the PESQ of each
setting plus the correlation between the blind input-SNR estimate and the true input SNR.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denoise  # noqa: E402
import evaluate as ev  # noqa: E402


def main():
    rows = ev.load_manifest("train")
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), 400, replace=False))]
    floors, overs = [-14, -18, -24, -30], [1.0, 1.6, 2.5]
    with ProcessPoolExecutor() as pool:
        est = np.array(list(pool.map(ev.input_snr_pair, rows, chunksize=8)))
        # manual Pearson r (np.corrcoef crashed in the BLAS call on this machine)
        a, b = est[:, 0] - est[:, 0].mean(), est[:, 1] - est[:, 1].mean()
        r = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
        print(f"corr(est, true input SNR) = {r:.3f}")
        for snr in sorted({r['target_snr_db'] for r in rows}, key=float):
            m = np.array([r["target_snr_db"] == snr for r in rows])
            print(f"  target {snr:>5}: est mean {est[m, 0].mean():6.2f}  true mean {est[m, 1].mean():6.2f}")
        table = {}
        for f, o in product(floors, overs):
            p = replace(denoise.DEFAULT, gain_floor_db=f, noise_overest=o)
            res = list(pool.map(ev._score_one, [(r, p, "proposed") for r in rows], chunksize=4))
            table[(f, o)] = np.array([x["pesq"] for x in res])
            print(f"floor {f:4} over {o:3}: PESQ {table[(f, o)].mean():.4f}", flush=True)
    print("\nPer target SNR (best marked *):")
    for snr in sorted({r['target_snr_db'] for r in rows}, key=float):
        m = np.array([r["target_snr_db"] == snr for r in rows])
        scores = {k: v[m].mean() for k, v in table.items()}
        best = max(scores, key=scores.get)
        line = "  ".join(f"{k[0]}/{k[1]}:{s:.3f}{'*' if k == best else ''}" for k, s in scores.items())
        print(f" {snr:>5} dB (n={m.sum()}): {line}")
    # oracle upper bound: best setting per SNR group
    oracle = sum(max(v[np.array([r['target_snr_db'] == s for r in rows])].sum() for v in table.values())
                 for s in {r['target_snr_db'] for r in rows}) / len(rows)
    print(f"\noracle per-SNR-group PESQ = {oracle:.4f} (upper bound of SNR-adaptive rule)")


if __name__ == "__main__":
    main()
