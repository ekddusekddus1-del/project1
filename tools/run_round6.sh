#!/usr/bin/env bash
# Round 6 (train 400 subset): HRNR variants, multi-band, cepstral smoothing, noise-type oracle.
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=${OUT:-results/tuning_round6}
mkdir -p "$OUT"
T="$PY -u tools/tune.py --subset 400"

$T --grid "hrnr_iters=[2,3]" --grid "hrnr_nl=['full']" --grid "hrnr_rho_pow=[0.5,2.0]" \
   --grid "hrnr_mix=[0.5,0.7]" --grid "hrnr_scale=[1.0,1.5]" > "$OUT/1_hrnr_variants.log" 2>&1
$T --grid "band_over_lo=[0.7,1.3,1.8]" --grid "band_over_mid=[0.8,1.25]" --grid "band_over_hi=[0.7,1.3,1.8]" \
   --grid "band_ximin_lo_db=[-5.0,5.0]" --grid "band_ximin_mid_db=[-5.0,5.0]" \
   --grid "band_ximin_hi_db=[-10.0,-5.0,5.0]" > "$OUT/2_multiband.log" 2>&1
$T --set cep_smooth=True --grid "cep_alpha_rest=[0.5,0.7,0.95]" --grid "cep_mix=[0.3,0.6]" \
   --grid "cep_env_ms=[0.5,2.0]" --grid "cep_alpha_pitch=[0.1,0.7]" --grid "cep_alpha_env=[0.0,0.5]" \
   > "$OUT/3_cepstral.log" 2>&1
$PY -u tools/category_oracle.py > "$OUT/4_category_oracle.log" 2>&1
echo ALL_DONE
