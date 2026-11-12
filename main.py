"""Submission entry point: denoise every WAV in an input folder.

    python main.py                       # ./input  -> ./output
    python main.py --input IN --output OUT

For each input file an output WAV with the same file name (and relative sub-path),
sample rate, channel count, sample count and sample format is written.
Only the noisy input is used; no clean reference is needed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from denoise import DEFAULT, enhance_multichannel

FLOAT_SUBTYPES = {"FLOAT", "DOUBLE"}
PCM_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32", "PCM_U8", "PCM_S8"}


def process_file(src: Path, dst: Path) -> None:
    info = sf.info(str(src))
    x, sr = sf.read(str(src), dtype="float64", always_2d=True)  # (n, channels)
    x = np.nan_to_num(x)
    y = enhance_multichannel(x, sr, DEFAULT)
    y = np.nan_to_num(y)
    subtype = info.subtype if info.subtype in FLOAT_SUBTYPES | PCM_SUBTYPES else "PCM_16"
    if subtype not in FLOAT_SUBTYPES:
        y = np.clip(y, -1.0, 1.0)
    if y.shape != x.shape:
        raise RuntimeError(f"shape mismatch for {src}: {x.shape} -> {y.shape}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = y[:, 0] if info.channels == 1 else y
    sf.write(str(dst), out, sr, subtype=subtype, format="WAV")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Traditional DSP speech denoiser")
    ap.add_argument("--input", default="input", help="input folder (default: ./input)")
    ap.add_argument("--output", default="output", help="output folder (default: ./output)")
    a = ap.parse_args(argv)
    in_dir, out_dir = Path(a.input), Path(a.output)
    if not in_dir.is_dir():
        print(f"input folder not found: {in_dir.resolve()}", file=sys.stderr)
        return 1
    files = sorted(p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        print(f"no WAV files in {in_dir.resolve()}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    failed = 0
    for i, src in enumerate(files, 1):
        dst = out_dir / src.relative_to(in_dir)
        try:
            process_file(src, dst)
        except Exception as e:  # keep going; report at the end
            failed += 1
            print(f"[{i}/{len(files)}] FAILED {src}: {e}", file=sys.stderr)
            continue
        if i % 50 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"done: {len(files) - failed}/{len(files)} files -> {out_dir.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
