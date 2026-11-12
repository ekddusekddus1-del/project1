"""Characterise each noise category from the additive noise d = noisy - clean (analysis only).

Features (per file, then averaged per category):
  nonstat  : std over frames of frame energy [dB]           -> time variation
  impulse  : 95th-percentile minus median frame energy [dB]  -> short bursts
  flatness : spectral flatness of the mean noise PSD (0..1)  -> broadband vs coloured
  tonal    : max peak-to-local-median ratio of mean PSD [dB] -> narrowband tones
  low/high : energy share below 500 Hz / above 4 kHz
  sweep    : std of the per-frame peak frequency [Hz] (only for tonal noise) -> moving tones
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402

N_FFT, HOP, SR = 512, 128, 16000


def features(d: np.ndarray) -> dict:
    _, _, z = signal.stft(d, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")
    p = np.abs(z[:, 2:-2]) ** 2 + 1e-14
    f = np.arange(p.shape[0]) * SR / N_FFT
    e = 10 * np.log10(p.sum(axis=0))
    mean_psd = p.mean(axis=1)
    flat = float(np.exp(np.mean(np.log(mean_psd))) / np.mean(mean_psd))
    peak = 10 * np.log10(mean_psd / median_filter(mean_psd, size=15, mode="nearest"))
    band = (f > 150) & (f < 6000)
    tonal = float(peak[band].max())
    pk = f[band][np.argmax(p[band], axis=0)]
    return {"nonstat": float(e.std()), "impulse": float(np.percentile(e, 95) - np.median(e)),
            "flatness": flat, "tonal": tonal,
            "low": float(mean_psd[f < 500].sum() / mean_psd.sum()),
            "high": float(mean_psd[f > 4000].sum() / mean_psd.sum()),
            "sweep": float(pk.std())}


def main():
    rows = ev.load_manifest("train")
    per_file = {}
    for r in rows:
        x, _ = sf.read(ev.DATA_ROOT / r["noisy_path"], dtype="float64")
        s, _ = sf.read(ev.DATA_ROOT / r["clean_path"], dtype="float64")
        per_file[r["segment_id"]] = (r["noise_category"], features(x - s))
    pesq = {r["segment_id"]: float(r["pesq"]) for r in
            csv.DictReader(open(ev.ROOT / "results/train_proposed_per_file.csv", encoding="utf-8-sig"))}
    groups = defaultdict(list)
    for sid, (cat, ft) in per_file.items():
        groups[cat].append((ft, pesq.get(sid, np.nan)))
    keys = ["nonstat", "impulse", "flatness", "tonal", "low", "high", "sweep"]
    print(f"{'category':22s} {'n':>4} " + " ".join(f"{k:>8}" for k in keys) + "    PESQ")
    for cat, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        m = {k: np.median([ft[k] for ft, _ in items]) for k in keys}
        pq = np.nanmean([q for _, q in items])
        print(f"{cat:22s} {len(items):4d} " + " ".join(f"{m[k]:8.2f}" for k in keys) + f"  {pq:6.3f}")
    out = ev.ROOT / "results" / "noise_features_train.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "noise_category", *keys, "pesq"])
        for sid, (cat, ft) in per_file.items():
            w.writerow([sid, cat, *[round(ft[k], 4) for k in keys], pesq.get(sid, "")])
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
