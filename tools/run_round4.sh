#!/usr/bin/env bash
# Round 4 experiments (train 400 subset): pre-filter, 2-stage, fair gain comparison, window retune.
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=${OUT:-results/tuning_round4}
mkdir -p "$OUT"
T="$PY tools/tune.py --subset 400"

$T --grid "hp_hz=[50,80,120,150]" > "$OUT/1_highpass.log" 2>&1
$T --grid "refine_th=[0.1,0.2,0.35,0.5]" --grid "noise_overest=[1.2,1.4,1.8,2.0]" \
   --grid "gain_floor_db=[-18,-24]" > "$OUT/2_two_stage.log" 2>&1
$T --set "gain='ss'" --grid "noise_overest=[1.0,1.3,2.0,2.5,3.0]" \
   --grid "gain_floor_db=[-10,-14,-17,-24]" --grid "frame_ms=[32,48,64,96]" > "$OUT/3_spectral_subtraction.log" 2>&1
$T --set "gain='wiener'" --grid "dd_alpha=[0.9,0.98]" --grid "noise_overest=[1.0,1.3,2.0]" \
   --grid "gain_floor_db=[-14,-17,-24]" --grid "frame_ms=[48,64,96]" --grid "xi_min_db=[-15,-25]" \
   > "$OUT/4_wiener.log" 2>&1
for w in blackman kaiser8; do
  $T --set "window='$w'" --set frame_ms=96 --grid "gain_floor_db=[-18,-24]" \
     --grid "noise_overest=[1.3,2.0]" --grid "dd_alpha=[0.93,0.97]" --grid "hop_ratio=[0.0625,0.1875]" \
     --grid "quantile_q=[0.3,0.5]" --grid "init_fraction=[0.1,0.2]" > "$OUT/5_window_$w.log" 2>&1
done
echo ALL_DONE
