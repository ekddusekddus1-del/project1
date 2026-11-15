# 2026 신호 및 시스템 1차 챌린지 — 전통적 신호 처리 기반 음성 잡음 제거

- **팀명:** 늙크크
- **제출 회차:** 1차 (최초 제출)

딥러닝·머신러닝을 사용하지 않는 STFT 기반 단일 채널 음성 향상 알고리즘입니다.
실행에는 `numpy`, `scipy`, `soundfile`만 필요합니다.

## 1. 설치

Python 3.10 이상 (3.11, 3.12에서 확인).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 실행 방법

저장소 루트에 `input/` 폴더를 만들고 테스트 WAV 파일을 넣은 뒤 실행합니다.

```bash
python main.py                                   # ./input  -> ./output
python main.py --input <입력 폴더> --output <출력 폴더>   # 폴더 직접 지정
```

- `input/`의 모든 `.wav` 파일(하위 폴더 포함)을 처리하여 `output/`에 **같은 파일명**으로 저장합니다.
- 출력은 입력과 **같은 샘플링 주파수, 채널 수, 샘플 수, 샘플 형식(예: PCM_16)** 을 유지합니다.
  다채널 입력은 채널별로 독립 처리합니다.
- 경로는 모두 상대 경로 또는 인자로 지정하며, 개인 PC 절대 경로에 의존하지 않습니다.
- 처리 시간: 3초 파일당 약 2.6초 (일반 노트북 CPU, 단일 프로세스).
- 알고리즘 파라미터는 모두 [denoise.py](denoise.py)의 `Params` 기본값에 포함되어 있으며, 추가 설정 파일은 없습니다.

## 3. 알고리즘

```
[아래 STFT 처리를 두 해상도로 각각 수행: 긴 창 96 ms (1536 samples), 짧은 창 12 ms (192 samples)]

noisy ─► STFT (Blackman, hop = 창 길이의 1/8, 87.5 % overlap)
       ─► 잡음 PSD 추정
            ① 저에너지 프레임(하위 10 %) 평균        ─┐
            ② 0.5 s 슬라이딩 40 % 분위수 (bin별)     ─┴─ min(①, ②)  × 과대추정
       ─► 대역별 파라미터        < 1 kHz     1–4 kHz     > 4 kHz
            과대추정            1.6         2.0         2.88      (1.6 × 1.0 / 1.25 / 1.8)
            ξ_min              −40 dB      −40 dB      −35 dB
       ─► Decision-Directed 사전 SNR 추정 (α = 0.97)
       ─► MMSE Log-Spectral Amplitude 이득 G_DD
       ─► HRNR: ŝ = ISTFT(G_DD·Y) → 반파 정류 max(ŝ, 0) → STFT = S_harmo
                ξ_HRNR = 1.25 · [ρ|G_DD·Y|² + (1−ρ)|S_harmo|²] / λ_d,  ρ = G_DD
                G = 0.6 · LSA(ξ_HRNR) + 0.4 · G_DD
       ─► ISTFT (noisy 위상 사용) ─► y_long (96 ms) / y_short (12 ms)

다중 해상도 결합: 16 ms Hann STFT에서 bin별로
            |Y₁| = 0.6 · |Y_long| + 0.4 · min(|Y_long|, |Y_short|),  위상은 Y_long ─► y₁
변조 영역 후처리: y₁의 STFT 크기 궤적(32 ms, hop 2 ms)을 bin마다 다시 STFT(256 ms)
            → 변조 영역 DD + LSA (ξ_min −10 dB) → 역변환 → ISTFT = y₂
enhanced = 0.75 · y₁ + 0.25 · y₂
```

| 구성 요소 | 내용 | 참고 문헌 |
| :--- | :--- | :--- |
| 이득 함수 | MMSE-LSA: $G=\frac{\xi}{1+\xi}\exp\left(\tfrac12 E_1(v)\right)$, $v=\frac{\xi\gamma}{1+\xi}$ | Ephraim & Malah (1985) |
| 사전 SNR | Decision-directed: $\xi_t=\alpha\frac{\lvert\hat S_{t-1}\rvert^2}{\lambda_d}+(1-\alpha)\max(\gamma_t-1,0)$ | Ephraim & Malah (1984) |
| 잡음 PSD ① | 파일 내 저에너지 프레임 평균 (과제 베이스라인 `estimate_noise_power` 방식) | Boll (1979) |
| 잡음 PSD ② | bin별 시간축 분위수, 지수분포 편향 보정 $\div(-\ln(1-q))$ | Stahl, Fischer & Bippus (2000) |
| 하모닉 재생 | HRNR: 반파 정류로 억제된 음성 하모닉을 재생해 사전 SNR에 반영 | Plapous, Marro & Scalart (2006) |
| 대역별 억제 | 저·중·고역별 과대추정과 ξ_min (고역은 더 강하게 억제) | Kamath & Loizou (2002) |
| 다중 해상도 결합 | 긴 창(주파수 분해능, 하모닉)과 짧은 창(시간 분해능, onset·자음) 결과 중 bin별로 더 억제된 쪽을 40 % 반영. 긴 창의 시간 번짐과 잔류 잡음 감소 | 자체 설계 (창 길이 trade-off 실험에서 착안) |
| 변조 영역 후처리 | 크기 궤적의 변조 스펙트럼에서 MMSE 추정, 잔여 잡음의 시간 변동 억제 | Paliwal, Schwerin & Wójcicki (2012) |

HRNR은 이득으로 깎인 유성음 하모닉을 되살리기 때문에, 음성 왜곡 없이 더 깊게 억제(ξ_min −20 → −35 dB)할 수 있게 해 주었습니다.

`min(①, ②)` 결합은 정상 잡음에서는 ①이, 시간에 따라 잡음이 줄어드는 구간에서는 ②가 과대추정을 막아 줍니다.
②를 단독으로 쓰면 연속 발화 구간에서 음성이 잡음으로 추정되어 PESQ가 크게 떨어졌습니다.

## 4. Validation 결과 (제출 버전)

평가: PESQ-WB (16 kHz, `pesq==0.0.4`), SNR = $10\log_{10}\frac{\sum s^2}{\sum (s-\hat s)^2}$, 파일별 단순 평균.
`main.py`로 16-bit PCM WAV를 저장한 뒤 그 파일로 측정한 값과 일치함을 확인했습니다.

| 방법 | Validation PESQ | Validation SNR (dB) | Train 전체 PESQ |
| :--- | ---: | ---: | ---: |
| Noisy 입력 | 1.395 | 9.63 | — |
| 과제 베이스라인 Wiener (기본값) | 1.628 | 13.29 | — |
| DD + LSA (HRNR 없음) | 1.893 | 14.31 | 1.885 |
| DD + LSA + HRNR | 2.017 | 14.53 | 1.968 |
| + 대역별 파라미터 | 2.031 | 14.52 | 1.979 |
| + 변조 영역 후처리 | 2.041 | 14.30 | 1.984 |
| **제안 방법 (+ 다중 해상도 결합, 제출)** | **2.051** | **14.13** | **2.003** |

Validation 상세 결과는 [results/validation_proposed_summary.json](results/validation_proposed_summary.json)에 있습니다.

상세 결과(파일별, 잡음 종류별): [results/](results/) · 튜닝 로그: [results/tuning_round4/](results/tuning_round4/), [results/tuning_round5/](results/tuning_round5/), [results/tuning_round6/](results/tuning_round6/), [results/tuning_round7/](results/tuning_round7/), [results/tuning_round8/](results/tuning_round8/), [results/tuning_round9/](results/tuning_round9/)

## 5. 실험 과정

파라미터는 **Train에서 무작위로 고른 400개(seed 0)** 에서만 좌표 탐색([tools/tune.py](tools/tune.py))으로 선택했고,
Validation과 Train 전체는 최종 확인에만 사용했습니다.

### 5.1 단계별 개선 (Train-400 PESQ)

| 단계 | 변경 | PESQ |
| :--- | :--- | ---: |
| 0 | SPP 잡음 추적 + LSA, Hann 32 ms, 하한 −20 dB | 1.643 |
| 1 | 이득 하한 −14 dB, DD α 0.95 | 1.682 |
| 2 | **SPP 잡음 추적 제거 → 고정 잡음 추정** | 1.764 |
| 3 | 잡음 과대추정 1.3, 64 ms 창, hop 1/8 | 1.806 |
| 4 | 80 ms 창, 하한 −17 dB, 과대추정 1.6 | 1.848 |
| 5 | 슬라이딩 분위수 잡음 추정과 min 결합 | 1.857 |
| 6 | 하한 −21 dB, 분위수 q 0.4 | 1.882 |
| 7 | **Blackman 96 ms**, 하한 −24 dB, 저에너지 비율 0.1 | 1.884 |
| 8 | **HRNR 추가** (혼합 0.75, 하한 −28 dB) | 1.928 |
| 9 | HRNR에 맞춰 재튜닝: ξ_min −35 dB, 하한 −40 dB, DD α 0.97, HRNR 혼합 0.6·스케일 1.25 | 1.974 |
| 10 | **대역별 과대추정·ξ_min** (중역 ×1.25, 고역 ×1.8, 저·중역 ξ_min −5 dB) | 1.991 |
| 11 | **변조 영역 MMSE 후처리** (25 % 혼합, ξ_min −10 dB) | 1.998 |
| 12 | **다중 해상도 결합** (12 ms 짧은 창, 16 ms 결합창, min 결합 40 %) | Train 전체 1.984 → **2.003** |

12단계는 Train 전체(1296개)로 선택했습니다 (짧은 창 12–32 ms, 결합 방식 time / geo / min, 가중치 0.2–0.6 비교, [results/tuning_round9/](results/tuning_round9/)).

### 5.2 이득 함수 비교 (HRNR 적용 전, 같은 잡음 추정, 각자 파라미터 튜닝)

| 이득 함수 | 최적 파라미터 | Val PESQ | Val SNR | Train 전체 PESQ |
| :--- | :--- | ---: | ---: | ---: |
| Spectral Subtraction | α = 4.0, β = −21 dB, 96 ms | 1.807 | 14.74 | 1.813 |
| Wiener (DD) | a = 0.90, g_min = −21 dB, 96 ms | 1.863 | 14.54 | 1.852 |
| MMSE-LSA, Hann 80 ms | a = 0.95, 하한 −24 dB | 1.885 | 14.36 | 1.884 |
| **MMSE-LSA, Blackman 96 ms** | a = 0.95, 하한 −24 dB | **1.893** | 14.31 | **1.885** |

SS는 SNR은 가장 높지만 음성 왜곡·musical noise로 PESQ가 가장 낮았습니다.

### 5.3 창 함수 비교 (Train-400, 창별 최적 길이)

| Window | Blackman | Kaiser (β=8) | Hann | Hamming | Bartlett | sqrt-Hann |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 최적 길이 | 96 ms | 96 ms | 80 ms | 80 ms | 80 ms | 64 ms |
| PESQ | 1.884* | 1.884* | 1.882 | 1.879 | 1.876 | 1.871 |

\* 주요 파라미터 6종을 창에 맞게 재튜닝한 값. 창 종류의 영향(≤0.013)은 창 길이의 영향(≤0.2)보다 훨씬 작았습니다.

### 5.4 효과가 없었던 시도 (Train-400, 기준 대비)

| 시도 | 변화 |
| :--- | :--- |
| SPP 기반 잡음 추적 (Gerkmann & Hendriks 2012) | −0.08 |
| OM-LSA (Cohen & Berdugo 2001) | −0.08 |
| 분위수 잡음 추정 단독 사용 | −0.21 |
| Butterworth 고역통과 전처리 50–150 Hz | −0.015 ~ 0 (SNR은 최대 −6.5 dB) |
| 2단 처리 (1단 이득으로 무음 bin 판정 후 잡음 재추정) | −0.007 ~ −0.06 |
| 이득 융합 (LSA와 Wiener / SS 이득의 가중 기하평균) | LSA 가중치를 높일수록 개선 → LSA 단독이 최적 |
| TSNR (Plapous et al. 2006, 2단계 사전 SNR) | +0.004 (DD 50 % 혼합), HRNR과 함께 쓰면 효과 없음 → 미채택 |
| 입력 SNR 적응 파라미터 (블라인드 SNR 추정 → 하한·과대추정 선택) | 0/5/10/15 dB 모두 최적 설정이 같음, SNR별 oracle 상한 = 단일 설정 최고값 → 미채택 |
| 시간·주파수 이득 평활 | −0.002 ~ −0.04 |
| HRNR 변형 (2–3회 반복, 전파 정류, ρ 지수) | −0.022 ~ 0 |
| Cepstral smoothing of gain (Breithaupt et al. 2008) | +0.003 단독, 대역별 파라미터와 결합 시 효과 없음 |
| 청각 마스킹 기반 적응 (Virag 1999, Johnston 1988 임계값) | Validation −0.001 / Train +0.002 (방향 불일치) → 미채택 |
| β-order MMSE (You et al. 2005) | 최적 β = 0.25에서 +0.001 (β → 0 이 LSA이므로 LSA가 사실상 최적) |
| Super-Gaussian MAP (Lotter & Vary 2005) | 튜닝 후에도 −0.06 (저SNR bin 과억제) |
| 유성음 위상 복원 (Krawczyk & Gerkmann 2014) | −0.001 ~ −0.011. 깨끗한 음성에 적용해도 PESQ 4.64 → 3.63으로, 하모닉 모델 위상의 오차가 noisy 위상 오차보다 큼 |
| 변조 영역 MMSE 단독 사용 | 1.809 (현재 파이프라인 1.991보다 낮음) → 후처리로만 사용 |
| IMCRA 잡음 추정 (Cohen 2003, 오프라인 구현) | 대체 시 −0.20 ~ −0.23, 기존 추정과 min 결합 시 −0.06 ~ −0.08 (Train 전체) |
| Consistent Wiener 재합성 (Le Roux & Vincent 2013, 사후 분산 가중 최소제곱 ISTFT) | −0.003 ~ −0.009 (Train 전체), Validation −0.005 ~ −0.016 |
| 다중 해상도 결합의 다른 방식 | 파형 가중 평균 −0.005 ~ +0.003, 크기 기하평균 −0.007 ~ +0.012 (min 결합 +0.020보다 낮음) |
| 저역 유성음 하모닉 보호 (f0 추정 → 300–800 Hz 아래 하모닉 bin의 이득 하한 상향, `lhp_*`) | Train 전체 +0.001 ~ +0.003, Validation +0.004 ~ +0.008. 목표였던 저역 화자 4명은 모든 설정에서 −0.002 ~ −0.008 → 이득 대비 복잡도가 커서 미채택 (`results/tuning_round9/`) |

### 5.5 잡음 종류별 처리(앙상블) 검토

잡음 성분(noisy − clean)의 특징(시간 변동, 충격성, 평탄도, 톤 피크, 저역 비율, 피크 이동)을 측정해
23종을 5개 그룹(정상 광대역 / 저역 집중 / 톤 / 충격성 / 비정상 광대역)으로 나누고 ([tools/noise_features.py](tools/noise_features.py)),
그룹별 전용 모듈(톤 추적, 저역 빠른 추적, 전 대역 빠른 추적, HPSS 충격음 분리)을 구현해 oracle 상한을 측정했습니다
([tools/module_oracle.py](tools/module_oracle.py), Train-400).

| 방식 | PESQ |
| :--- | ---: |
| 대역별 파라미터를 모든 파일에 일괄 적용 | 1.991 |
| 5개 그룹마다 최적 설정 선택 (분류 100 % 정확 가정) | 1.998 |
| 23개 종류마다 최적 설정 선택 (분류 100 % 정확 가정) | 2.001 |

- 전용 모듈은 해당 잡음에서도 효과가 없었습니다. 음성 하모닉·자음 onset이 톤·충격음과 같은 특징을 가져 음성까지 억제했고,
  모든 파일에 적용하면 PESQ가 0.13–0.74 떨어졌습니다.
- 완벽한 분류를 가정해도 일괄 적용 대비 +0.006에 그치고, 오분류 시 손실이 커서 **잡음별 분기 없이 하나의 알고리즘으로 일반화**했습니다.

### 5.6 구성 요소 제거 실험 (ablation)

최종 시스템에서 구성 요소를 하나씩 뺀 결과입니다 ([tools/ablation.py](tools/ablation.py), [results/ablation.log](results/ablation.log)).
모든 구성 요소가 Validation과 Train 양쪽에서 기여하며, 단계를 추가해서 오히려 떨어지는 경우는 없었습니다.

| 제거한 구성 요소 | Validation PESQ | Train 전체 PESQ |
| :--- | ---: | ---: |
| (없음, 최종 시스템) | 2.051 | 2.003 |
| 다중 해상도 결합 | 2.041 (−0.010) | 1.984 (−0.020) |
| 변조 영역 후처리 | 2.041 (−0.010) | 1.998 (−0.006) |
| 대역별 파라미터 | 2.042 (−0.010) | 1.994 (−0.009) |
| HRNR | 1.923 (−0.129) | 1.894 (−0.109) |
| 분위수 잡음 추정 | 2.023 (−0.028) | 1.977 (−0.026) |
| 잡음 과대추정 (×1.0) | 2.012 (−0.039) | 1.991 (−0.012) |

### 5.7 남은 한계

Train에서 입력보다 PESQ가 낮아지는 파일 166개 중 98개가 4명의 화자(speaker005/026/030/036)에 몰려 있습니다.
이 화자들은 깨끗한 음성 에너지의 64–85 %가 300 Hz 아래에 있고(다른 화자 약 50 %), 그 저역 에너지가 발화 구간에만 나타나
녹음 잡음이 아닌 목소리 자체의 특성(낮은 음성·근접 효과)으로 보입니다. 데이터에 저역 잡음(기계음, 바람 등)이 많아
저역을 강하게 억제하도록 튜닝되었기 때문에 이 화자들은 음성도 함께 깎입니다 (입력 대비 개선 폭 +0.05, 다른 화자 +0.63).
저역 하모닉 보호를 시도했지만 해결되지 않았습니다 (5.4).

## 6. 저장소 구성

| 파일 | 용도 |
| :--- | :--- |
| [main.py](main.py) | **제출 실행 파일**: `input/` → `output/` |
| [denoise.py](denoise.py) | 알고리즘 구현과 파라미터 (`Params`) |
| [evaluate.py](evaluate.py) | (개발용) Train/Validation PESQ·SNR 평가 |
| [tools/tune.py](tools/tune.py) | (개발용) 좌표 탐색 기반 파라미터 튜닝 |
| [tools/run_round4.sh](tools/run_round4.sh), [tools/run_round6.sh](tools/run_round6.sh) | (개발용) 조합·비교 실험 일괄 실행 |
| [tools/snr_analysis.py](tools/snr_analysis.py), [tools/category_oracle.py](tools/category_oracle.py), [tools/module_oracle.py](tools/module_oracle.py) | (개발용) SNR·잡음 종류별 적응의 oracle 상한 분석 |
| [tools/noise_features.py](tools/noise_features.py) | (개발용) 잡음 종류별 신호 특징 측정 |
| [tools/ablation.py](tools/ablation.py) | (개발용) 구성 요소 제거 실험 |
| [tools/variant_test.py](tools/variant_test.py), [tools/lhp_test.py](tools/lhp_test.py) | (개발용) 설정 비교 (저역 화자 / 나머지 Train / Validation) |
| `requirements.txt` | 실행용 패키지 |
| `requirements-eval.txt` | 평가용 패키지 (`pesq` 포함) |

### 평가 재현 (선택)

`pesq`는 C 확장이라 Windows에서는 conda-forge 설치를 권장합니다.

```bash
conda create -n denoise -c conda-forge python=3.12 numpy scipy pesq=0.0.4
conda activate denoise
pip install soundfile
# 데이터: 저장소 루트에 noise_reduction_challenge/{train,validation}/ 배치
python evaluate.py --split validation --groups
python evaluate.py --split validation --set "gain='wiener'" --set dd_alpha=0.9   # 파라미터 변경 비교
```

## 7. 참고 문헌

1. S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Trans. ASSP*, vol. 27, no. 2, pp. 113–120, 1979. doi:10.1109/TASSP.1979.1163209
2. Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," *IEEE Trans. ASSP*, vol. 32, no. 6, pp. 1109–1121, 1984. doi:10.1109/TASSP.1984.1164453
3. Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," *IEEE Trans. ASSP*, vol. 33, no. 2, pp. 443–445, 1985. doi:10.1109/TASSP.1985.1164550
4. V. Stahl, A. Fischer, and R. Bippus, "Quantile based noise estimation for spectral subtraction and Wiener filtering," *Proc. IEEE ICASSP*, 2000, pp. 1875–1878. doi:10.1109/ICASSP.2000.862122
5. C. Plapous, C. Marro, and P. Scalart, "Improved signal-to-noise ratio estimation for speech enhancement," *IEEE Trans. ASLP*, vol. 14, no. 6, pp. 2098–2108, 2006. doi:10.1109/TASL.2006.872621 (HRNR 채택, TSNR 비교)
6. T. Gerkmann and R. C. Hendriks, "Unbiased MMSE-based noise power estimation with low complexity and low tracking delay," *IEEE Trans. ASLP*, vol. 20, no. 4, pp. 1383–1393, 2012. doi:10.1109/TASL.2011.2180896 (비교 실험, 미채택)
7. I. Cohen and B. Berdugo, "Speech enhancement for non-stationary noise environments," *Signal Processing*, vol. 81, no. 11, pp. 2403–2418, 2001. doi:10.1016/S0165-1684(01)00128-1 (비교 실험, 미채택)
8. S. D. Kamath and P. C. Loizou, "A multi-band spectral subtraction method for enhancing speech corrupted by colored noise," *Proc. IEEE ICASSP*, 2002. (대역별 파라미터 착안)
9. N. Virag, "Single channel speech enhancement based on masking properties of the human auditory system," *IEEE Trans. Speech Audio Process.*, vol. 7, no. 2, pp. 126–137, 1999. doi:10.1109/89.748118 (비교 실험, 미채택)
10. J. D. Johnston, "Transform coding of audio signals using perceptual noise criteria," *IEEE J. Sel. Areas Commun.*, vol. 6, no. 2, pp. 314–323, 1988. (마스킹 임계값 계산)
11. D. Fitzgerald, "Harmonic/percussive separation using median filtering," *Proc. DAFx*, 2010. (충격음 모듈, 미채택)
12. C. Breithaupt, T. Gerkmann, and R. Martin, "A novel a priori SNR estimation approach based on selective cepstro-temporal smoothing," *Proc. IEEE ICASSP*, 2008. (비교 실험, 미채택)
13. Y. Hu and P. C. Loizou, "Subjective comparison and evaluation of speech enhancement algorithms," *Speech Communication*, vol. 49, pp. 588–601, 2007. doi:10.1016/j.specom.2006.12.006 (전통 알고리즘 비교 참고)
14. C. H. You, S. N. Koh, and S. Rahardja, "β-order MMSE spectral amplitude estimation for speech enhancement," *IEEE Trans. Speech Audio Process.*, vol. 13, no. 4, pp. 475–486, 2005. (비교 실험, 미채택)
15. T. Lotter and P. Vary, "Speech enhancement by MAP spectral amplitude estimation using a super-Gaussian speech model," *EURASIP J. Applied Signal Processing*, vol. 2005, no. 7, pp. 1110–1126, 2005. (비교 실험, 미채택)
16. M. Krawczyk and T. Gerkmann, "STFT phase reconstruction in voiced speech for an improved single-channel speech enhancement," *IEEE/ACM Trans. ASLP*, vol. 22, no. 12, pp. 1931–1940, 2014. (비교 실험, 미채택)
17. K. Paliwal, B. Schwerin, and K. Wójcicki, "Speech enhancement using a minimum mean-square error short-time spectral modulation magnitude estimator," *Speech Communication*, vol. 54, no. 2, pp. 282–305, 2012. (변조 영역 후처리 채택)
18. I. Cohen, "Noise spectrum estimation in adverse environments: improved minima controlled recursive averaging," *IEEE Trans. Speech Audio Process.*, vol. 11, no. 5, pp. 466–475, 2003. doi:10.1109/TSA.2003.811544 (비교 실험, 미채택)
19. J. Le Roux and E. Vincent, "Consistent Wiener filtering for audio source separation," *IEEE Signal Process. Lett.*, vol. 20, no. 3, pp. 217–220, 2013. doi:10.1109/LSP.2012.2225617 (비교 실험, 미채택)
