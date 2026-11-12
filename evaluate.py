"""Development-only evaluation: PESQ-WB and SNR on train/validation.

Usage:
    python evaluate.py --split validation
    python evaluate.py --split train --subset 300
    python evaluate.py --split validation --set gain=wiener --set gain_floor_db=-15

Requires requirements-eval.txt (pesq). The clean files are used ONLY here.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import soundfile as sf

import denoise

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "noise_reduction_challenge"


def snr_db(clean, est):
    err = np.sum((clean - est) ** 2)
    return float(10 * np.log10(np.sum(clean ** 2) / max(err, 1e-20)))


def load_manifest(split):
    with open(DATA_ROOT / split / "samples.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _baseline_wiener(noisy, sr):
    """Course baseline (fixed noise PSD + DD Wiener) for reference."""
    p = denoise.Params(frame_ms=32, hop_ratio=0.25, track_noise=False, init_fraction=0.2,
                       quantile_win_s=0, gain="wiener", dd_alpha=0.98, noise_overest=1.0,
                       gain_floor_db=20 * np.log10(0.05), xi_min_db=-100)
    return denoise.enhance(noisy, sr, p)


def _score_one(args):
    row, params, method = args
    from pesq import pesq
    clean, sr = sf.read(DATA_ROOT / row["clean_path"], dtype="float64")
    noisy, _ = sf.read(DATA_ROOT / row["noisy_path"], dtype="float64")
    if method == "noisy":
        out = noisy
    elif method == "baseline":
        out = _baseline_wiener(noisy, sr)
    else:
        out = denoise.enhance(noisy, sr, params)
    # Simulate 16-bit WAV output quantisation, as the submission writes PCM_16.
    out = np.round(np.clip(out, -1, 1 - 2 ** -15) * 32768) / 32768
    try:
        pq = float(pesq(sr, clean, out, "wb"))
    except Exception:
        pq = float("nan")
    return {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
            "target_snr_db": row["target_snr_db"], "pesq": pq, "snr": snr_db(clean, out)}


def input_snr_pair(row):
    """(blind input-SNR estimate, true input SNR) for one manifest row."""
    noisy, sr = sf.read(DATA_ROOT / row["noisy_path"], dtype="float64")
    clean, _ = sf.read(DATA_ROOT / row["clean_path"], dtype="float64")
    true = 10 * np.log10(np.sum(clean ** 2) / np.sum((noisy - clean) ** 2))
    return float(denoise.estimate_input_snr(noisy, sr)), float(true)


def evaluate(split="validation", params=denoise.DEFAULT, method="proposed",
             subset=None, seed=0, workers=None):
    rows = load_manifest(split)
    if subset and subset < len(rows):
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), subset, replace=False))]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_score_one, [(r, params, method) for r in rows], chunksize=4))
    return res


def summarize(res):
    pq = np.array([r["pesq"] for r in res])
    sn = np.array([r["snr"] for r in res])
    return {"n": len(res), "pesq": float(np.nanmean(pq)), "snr": float(np.mean(sn)),
            "pesq_fail": int(np.isnan(pq).sum())}


def by_group(res, key):
    groups = {}
    for r in res:
        groups.setdefault(r[key], []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def parse_sets(items):
    names = {f.name: f.type for f in fields(denoise.Params)}
    kw = {}
    for item in items or []:
        k, v = item.split("=", 1)
        if k not in names:
            raise SystemExit(f"unknown param {k}")
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    return replace(denoise.DEFAULT, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--method", default="proposed", choices=["proposed", "noisy", "baseline"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--set", action="append", help="override Params field, e.g. gain=wiener")
    ap.add_argument("--groups", action="store_true", help="print per-category table")
    ap.add_argument("--save", type=str, default=None, help="save per-file CSV + summary JSON")
    a = ap.parse_args()
    params = parse_sets(a.set)
    res = evaluate(a.split, params, a.method, a.subset)
    s = summarize(res)
    print(f"[{a.split}/{a.method}] n={s['n']} PESQ={s['pesq']:.4f} SNR={s['snr']:.3f} dB "
          f"fails={s['pesq_fail']}")
    if a.groups:
        for key in ("noise_category", "target_snr_db"):
            print(f"-- by {key}")
            for k, v in by_group(res, key).items():
                print(f"  {k:24s} n={v['n']:4d} PESQ={v['pesq']:.3f} SNR={v['snr']:.2f}")
    if a.save:
        out = Path(a.save)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{a.split}_{a.method}_per_file.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)
        summary = {"split": a.split, "method": a.method, "params": asdict(params), **s,
                   "by_category": by_group(res, "noise_category"),
                   "by_snr": by_group(res, "target_snr_db")}
        (out / f"{a.split}_{a.method}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
