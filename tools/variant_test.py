"""Grid over Params overrides (development only).

Reports PESQ on (a) the four low-voiced train speakers, (b) all other train files and
(c) validation, each as the change against the current default.
python tools/variant_test.py "mr_mode='geo'" "mr_mode='time',mr_weight=0.4" ...
"""
from __future__ import annotations

import ast
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denoise  # noqa: E402
import evaluate as ev  # noqa: E402

BAD = ("speaker026", "speaker030", "speaker036", "speaker005")


def parse(spec):
    kw = {}
    for item in filter(None, spec.split(",")):
        k, v = item.split("=")
        kw[k] = ast.literal_eval(v)
    return kw


def run(pool, rows, params):
    res = list(pool.map(ev._score_one, [(r, params, "proposed") for r in rows], chunksize=4))
    return np.array([x["pesq"] for x in res])


def main():
    train = ev.load_manifest("train")
    sets = {
        "bad-4": [r for r in train if r["segment_id"].startswith(BAD)],
        "train-other": [r for r in train if not r["segment_id"].startswith(BAD)],
        "validation": ev.load_manifest("validation"),
    }
    only = sys.argv[1:]
    with ProcessPoolExecutor() as pool:
        base = {k: run(pool, v, denoise.DEFAULT) for k, v in sets.items()}
        print("default  " + "  ".join(f"{k} {v.mean():.4f}" for k, v in base.items()), flush=True)
        for spec in only:
            p = replace(denoise.DEFAULT, **parse(spec))
            out, d = [], {}
            for k, v in sets.items():
                d[k] = run(pool, v, p) - base[k]
                out.append(f"{k} {d[k].mean():+.4f}")
            d_tr = np.concatenate([d["bad-4"], d["train-other"]]).mean()
            print(f"{spec:45s} " + "  ".join(out) + f"  | train-all {d_tr:+.4f}", flush=True)


if __name__ == "__main__":
    main()
