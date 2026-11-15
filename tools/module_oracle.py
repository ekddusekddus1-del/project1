"""Which noise-type module helps which noise? Oracle upper bounds (development only).

Runs every config on the same train subset, stores the per-file PESQ matrix, then reports
  * mean PESQ of each config applied to all files,
  * per noise category: best config and its gain over the default,
  * oracle PESQ when each category / each feature group / each file picks its best config.
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denoise  # noqa: E402
import evaluate as ev  # noqa: E402

D = denoise.DEFAULT
MB = dict(band_over_mid=1.25, band_over_hi=1.8, band_ximin_lo_db=-5.0, band_ximin_mid_db=-5.0)
CONFIGS = {
    "default": {},
    "multiband": MB,
    "tone_1.0": dict(tone_scale=1.0),
    "tone_0.5": dict(tone_scale=0.5),
    "tone_1.0_th15": dict(tone_scale=1.0, tone_thresh_db=15.0),
    "tone_1.0_0.25s": dict(tone_scale=1.0, tone_min_s=0.25),
    "lf_0.3": dict(fast_scale=0.3),
    "lf_0.6": dict(fast_scale=0.6),
    "lf_1.0": dict(fast_scale=1.0),
    "fast_0.15": dict(fast_scale=0.15, fast_fmax_hz=8000.0),
    "fast_0.3": dict(fast_scale=0.3, fast_fmax_hz=8000.0),
    "fast_0.6": dict(fast_scale=0.6, fast_fmax_hz=8000.0),
    "trans_0.25": dict(trans_scale=0.25),
    "trans_0.5": dict(trans_scale=0.5),
    "trans_1.0": dict(trans_scale=1.0),
    "wiener": dict(gain="wiener", dd_alpha=0.9),
    "ss": dict(gain="ss", noise_overest=4.0, hrnr=False, xi_min_db=-20.0, gain_floor_db=-21.0),
    "milder": dict(xi_min_db=-25.0, gain_floor_db=-25.0),
    "stronger": dict(noise_overest=2.2, xi_min_db=-42.0),
}
GROUPS = {
    "A_stationary_broadband": ["aircraft_cockpit", "broadband_noise", "hf_channel", "destroyer_operations"],
    "B_low_frequency": ["machine", "wind", "traffic_vehicle", "military_vehicle", "factory", "vehicle_interior"],
    "C_tonal": ["alarm_siren", "phone_dialing", "destroyer_engine"],
    "D_impulsive": ["footsteps", "clap", "click", "door_moving", "machine_gun", "animal", "bell"],
    "E_nonstationary_broadband": ["water", "shower", "tooth_brushing"],
}


def main():
    rows = ev.load_manifest("train")
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), 400, replace=False))]
    cats = np.array([r["noise_category"] for r in rows])
    names = list(CONFIGS)
    mat = np.zeros((len(names), len(rows)))
    with ProcessPoolExecutor() as pool:
        for i, n in enumerate(names):
            p = replace(D, **CONFIGS[n])
            res = list(pool.map(ev._score_one, [(r, p, "proposed") for r in rows], chunksize=4))
            mat[i] = [x["pesq"] for x in res]
            print(f"{n:16s} all-files PESQ {mat[i].mean():.4f} ({mat[i].mean() - mat[0].mean():+.4f})",
                  flush=True)
    out = ev.ROOT / "results" / "tuning_round7"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "module_matrix.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "noise_category", *names])
        for j, r in enumerate(rows):
            w.writerow([r["segment_id"], r["noise_category"], *np.round(mat[:, j], 4)])

    print("\nPer category: best config (gain over default) | top-3")
    cat_total = 0.0
    for c in sorted(set(cats), key=lambda c: -(cats == c).sum()):
        m = cats == c
        means = mat[:, m].mean(axis=1)
        order = np.argsort(-means)
        cat_total += means[order[0]] * m.sum()
        top = ", ".join(f"{names[k]} {means[k] - means[0]:+.3f}" for k in order[:3])
        print(f"  {c:22s} n={m.sum():3d} default {means[0]:.3f} | {top}")

    print("\nPer group: best config")
    grp_total = 0.0
    for g, members in GROUPS.items():
        m = np.isin(cats, members)
        means = mat[:, m].mean(axis=1)
        k = int(np.argmax(means))
        grp_total += means[k] * m.sum()
        print(f"  {g:28s} n={m.sum():3d} default {means[0]:.3f} best {names[k]} "
              f"{means[k]:.3f} ({means[k] - means[0]:+.3f})")
    n = len(rows)
    print(f"\ndefault (all files)        : {mat[0].mean():.4f}")
    print(f"best single config         : {mat.mean(axis=1).max():.4f} ({names[int(np.argmax(mat.mean(axis=1)))]})")
    print(f"oracle per GROUP (5 rules) : {grp_total / n:.4f}")
    print(f"oracle per CATEGORY (23)   : {cat_total / n:.4f}")
    print(f"oracle per FILE            : {mat.max(axis=0).mean():.4f}")


if __name__ == "__main__":
    main()
