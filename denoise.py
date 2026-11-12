"""Traditional (non-learning) single-channel speech enhancement.

Pipeline (per channel):
    STFT -> SPP-based MMSE noise PSD tracking -> decision-directed a-priori SNR
         -> MMSE log-spectral amplitude (LSA) gain (optionally OM-LSA) -> ISTFT

Only numpy / scipy are used. No clean reference, no training.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import signal
from scipy.ndimage import (convolve1d, maximum_filter, median_filter, minimum_filter1d,
                           percentile_filter, uniform_filter, uniform_filter1d)
from scipy.special import exp1, hyp1f1
from scipy.special import gamma as gamma_fn


@dataclass(frozen=True)
class Params:
    # Defaults = submission configuration, tuned on a 400-file TRAIN subset
    # (coordinate search, tools/tune.py) and checked on validation.
    # --- STFT ---
    window: str = "blackman"        # hann | hamming | blackman | sqrt_hann | bartlett | kaiser8
    frame_ms: float = 96.0          # analysis window length (1536 samples @16 kHz)
    hop_ratio: float = 0.125        # hop = hop_ratio * window (12 ms)
    # --- noise PSD initialisation (from the lowest-energy frames) ---
    init_fraction: float = 0.10     # fraction of lowest-energy frames to average
    # --- non-causal sliding quantile noise estimate (Stahl et al., 2000) ---
    quantile_win_s: float = 0.5     # window length in seconds (0 = off)
    quantile_q: float = 0.4         # quantile of the per-bin power
    quantile_mix: str = "min"       # combine with base estimate: "min" | "replace" | "max"
    # --- SPP-based noise tracking (Gerkmann & Hendriks, 2012); off: hurt PESQ ---
    track_noise: bool = False
    # IMCRA (Cohen 2003), offline variant: "off" | "replace" (instead of the initial-frame
    # estimate) | "min" (element-wise minimum with it)
    imcra: str = "off"
    imcra_win_s: float = 1.0        # minimum-search window (centred)
    imcra_alpha_s: float = 0.9      # recursive time smoothing of the periodogram
    imcra_alpha_d: float = 0.85     # noise update smoothing
    imcra_beta: float = 1.47        # bias compensation of the noise estimate
    imcra_bmin: float = 1.66        # bias of the minimum
    imcra_gamma0: float = 4.6
    imcra_gamma1: float = 3.0
    imcra_zeta0: float = 1.67
    spp_xi_h1_db: float = 15.0      # fixed a-priori SNR under speech presence
    spp_prior: float = 0.5          # P(H1)
    noise_alpha: float = 0.8        # smoothing of the noise PSD update
    two_pass: bool = True           # re-run tracking starting from pass-1 final estimate
    # --- gain ---
    gain: str = "lsa"               # "wiener" | "lsa" | "omlsa"
    dd_alpha: float = 0.97          # decision-directed smoothing
    xi_min_db: float = -35.0        # a-priori SNR floor (effective suppression limit)
    gain_floor_db: float = -40.0    # minimum gain
    noise_overest: float = 1.6      # multiply noise PSD used by the gain
    mix_w: float = 0.5              # LSA weight for gain fusion ("lsa_wiener" / "lsa_ss")
    # "beta": beta-order MMSE (You et al. 2005); beta -> 0 = LSA, beta = 1 = STSA
    beta_order: float = 0.5
    # "map_sg": super-Gaussian joint MAP amplitude estimator (Lotter & Vary 2005)
    sg_mu: float = 1.74
    sg_nu: float = 0.126
    # --- post-processing of the gain (reduces musical noise) ---
    gain_smooth_t: int = 1          # median/mean smoothing length over frames (1 = off)
    gain_smooth_f: int = 1          # smoothing length over frequency bins (1 = off)
    noise_smooth_t: int = 1         # moving-average length of noise PSD over frames
    # --- TSNR / HRNR (Plapous, Marro & Scalart 2006) ---
    tsnr: bool = False
    tsnr_gain: str = "lsa"          # gain rule applied to the TSNR a-priori SNR
    tsnr_mix: float = 1.0           # 1 = pure TSNR gain, 0 = DD gain
    hrnr: bool = True
    hrnr_gain: str = "lsa"
    hrnr_mix: float = 0.6           # final gain = 0.6 * G_HRNR + 0.4 * G_DD
    hrnr_scale: float = 1.25        # scale of the HRNR a-priori SNR
    hrnr_iters: int = 1             # repeat the HRNR step
    hrnr_nl: str = "half"           # non-linearity: "half" max(s,0) | "full" |s|
    hrnr_rho_pow: float = 1.0       # rho = G_DD ** pow (weight of the TSNR/DD part)
    # --- multi-band over-estimation / SNR floor (Kamath & Loizou 2002 idea) ---
    band_edges_hz: tuple = (1000.0, 4000.0)   # low | mid | high
    band_over_lo: float = 1.0       # multiplies noise_overest in the low band
    band_over_mid: float = 1.25
    band_over_hi: float = 1.8
    band_ximin_lo_db: float = -5.0  # added to xi_min_db in the low band
    band_ximin_mid_db: float = -5.0
    band_ximin_hi_db: float = 0.0
    # --- selective cepstro-temporal smoothing of the gain (Breithaupt et al. 2008) ---
    cep_smooth: bool = False
    cep_env_ms: float = 1.0         # quefrencies below this: envelope, alpha_env
    cep_alpha_env: float = 0.2
    cep_alpha_pitch: float = 0.4    # around the detected pitch quefrency
    cep_alpha_rest: float = 0.9     # all other quefrencies: strong temporal smoothing
    cep_mix: float = 1.0
    # --- input-SNR adaptive parameters (rule based, no learning) ---
    adapt: bool = False
    adapt_snr_lo: float = 0.0       # estimated input SNR (dB) mapped to the *_lo values
    adapt_snr_hi: float = 20.0      # ... and to the *_hi values (linear in between)
    adapt_floor_lo_db: float = -24.0
    adapt_floor_hi_db: float = -24.0
    adapt_over_lo: float = 1.6
    adapt_over_hi: float = 1.6
    # --- noise-type specific modules (all off by default; enabled by rules) ---
    # tone: persistent narrowband peaks (stationary or sweeping) are treated as noise
    tone_scale: float = 0.0         # 0 = off; noise >= tone_scale * |Y|^2 on tonal bins
    tone_thresh_db: float = 12.0    # peak height above the local spectral median
    tone_min_s: float = 0.4         # minimum duration of a tonal track
    # second (fast) tracker: short-window quantile, combined with max() below f_max
    fast_scale: float = 0.0         # 0 = off
    fast_win_s: float = 0.15
    fast_q: float = 0.3
    fast_fmax_hz: float = 500.0     # 500 Hz = low band only (wind); 8000 = full band
    # transient: percussive part (HPSS median filtering) is added to the noise estimate
    trans_scale: float = 0.0        # 0 = off
    trans_kernel: int = 17          # median length (bins / frames)
    # --- auditory masking based adaptation (Virag 1999; threshold: Johnston 1988) ---
    mask_on: bool = False
    mask_lo_db: float = -15.0       # mask-to-noise ratio (T/N) at which suppression is strongest
    mask_hi_db: float = 5.0         # ... and at which it is weakest (noise is masked)
    mask_over_strong: float = 1.3   # over-estimation multiplier where noise is audible
    mask_over_weak: float = 0.8     # ... where noise is masked by speech
    mask_ximin_strong_db: float = -5.0   # added to xi_min where noise is audible
    mask_ximin_weak_db: float = 5.0      # ... where noise is masked
    # --- phase reconstruction of voiced speech (Krawczyk & Gerkmann 2014) ---
    phase_on: bool = False
    pr_amount: float = 1.0          # max weight of the reconstructed phase
    pr_snr0_db: float = 0.0         # local SNR at which the weight is halved
    pr_fmax_hz: float = 4000.0      # harmonics above this keep the noisy phase
    pr_vth: float = 0.5             # voicing threshold on the normalised autocorrelation
    pr_frame_ms: float = 32.0       # separate (short) STFT for the harmonic model
    pr_lobe: int = 1                # bins on each side of a harmonic that are reconstructed
    # --- modulation-domain MMSE (Paliwal, Schwerin & Wojcicki 2012) ---
    mod_mode: str = "post"          # "off" | "only" (replace pipeline) | "post" (after pipeline)
    mod_ac_ms: float = 32.0         # acoustic frame
    mod_ac_hop_ms: float = 2.0      # acoustic hop (modulation sampling period)
    mod_win_ms: float = 256.0       # modulation frame
    mod_hop_ratio: float = 0.125
    mod_dd_alpha: float = 0.95
    mod_xi_min_db: float = -10.0
    mod_overest: float = 1.0
    mod_init_fraction: float = 0.15
    mod_mix: float = 0.25           # for "post": weight of the modulation-domain result
    # --- optional pre-filter / two-stage refinement ---
    hp_hz: float = 0.0              # Butterworth high-pass cut-off before STFT (0 = off)
    hp_order: int = 4
    # --- low-frequency harmonic protection (voiced speech harmonics below lhp_fmax_hz) ---
    # --- multi-resolution fusion: a 2nd run with a short frame, combined with the long one ---
    mr_mode: str = "min"            # "off" | "time" (waveform blend) | "geo" | "min" (magnitudes)
    mr_frame_ms: float = 12.0       # short-frame length of the 2nd run
    mr_weight: float = 0.4          # weight of the short-frame result
    mr_an_ms: float = 16.0          # analysis frame used to combine magnitudes
    # --- consistent Wiener-type resynthesis (Le Roux & Vincent 2013): weighted LS ISTFT ---
    cw_iters: int = 0               # conjugate-gradient iterations (0 = plain ISTFT)
    cw_pow: float = 1.0             # weight = (posterior variance) ** -cw_pow
    lhp_on: bool = False
    lhp_fmax_hz: float = 300.0      # protect harmonics up to this frequency
    lhp_floor_db: float = -6.0      # gain lower bound on protected bins
    lhp_lobe_hz: float = 15.0       # half width around each harmonic
    lhp_gamma_db: float = 3.0       # only where the noisy power exceeds the noise estimate by this
    lhp_vth: float = 0.6            # voicing threshold (normalised autocorrelation)
    lhp_frame_ms: float = 40.0      # pitch analysis frame (same hop as the main STFT)
    refine_th: float = 0.0          # 2nd stage: bins with stage-1 gain <= th are noise (0 = off)
    refine_min_frames: int = 10     # minimum noise-only frames per bin to trust re-estimate


DEFAULT = Params()


# ----------------------------------------------------------------------------
# STFT helpers
# ----------------------------------------------------------------------------
def _stft_sizes(sr: int, p: Params) -> tuple[int, int]:
    n_fft = 2 * int(round(p.frame_ms * 1e-3 * sr / 2))  # even length
    hop = max(1, int(round(n_fft * p.hop_ratio)))
    return n_fft, hop


def _window(name: str, n: int) -> np.ndarray:
    if name == "sqrt_hann":
        return np.sqrt(signal.get_window("hann", n))
    if name.startswith("kaiser"):          # e.g. "kaiser8"
        return signal.get_window(("kaiser", float(name[6:] or 8)), n)
    return signal.get_window(name, n)


def _stft(x, sr, n_fft, hop, window="hann"):
    _, _, z = signal.stft(x, fs=sr, window=_window(window, n_fft), nperseg=n_fft,
                          noverlap=n_fft - hop, nfft=n_fft, boundary="zeros", padded=True)
    return z


def _istft(z, sr, n_fft, hop, length, window="hann"):
    _, y = signal.istft(z, fs=sr, window=_window(window, n_fft), nperseg=n_fft,
                        noverlap=n_fft - hop,
                        nfft=n_fft, input_onesided=True, boundary=True)
    y = y[:length]
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y


# ----------------------------------------------------------------------------
# Noise PSD estimation
# ----------------------------------------------------------------------------
def _initial_noise(power: np.ndarray, p: Params) -> np.ndarray:
    """Mean of the lowest-energy frames (ignoring padded edge frames)."""
    n_frames = power.shape[1]
    inner = np.arange(2, n_frames - 2) if n_frames > 8 else np.arange(n_frames)
    energy = power[:, inner].mean(axis=0)
    count = max(1, int(np.ceil(p.init_fraction * len(inner))))
    sel = inner[np.argsort(energy, kind="stable")[:count]]
    return np.maximum(power[:, sel].mean(axis=1), 1e-12)


def _track_noise(power: np.ndarray, noise0: np.ndarray, p: Params) -> np.ndarray:
    """Unbiased MMSE noise PSD tracking with speech presence probability."""
    xi_h1 = 10 ** (p.spp_xi_h1_db / 10)
    prior_ratio = (1 - p.spp_prior) / p.spp_prior
    c = xi_h1 / (1 + xi_h1)
    n_bins, n_frames = power.shape
    noise = np.empty_like(power)
    n = noise0.copy()
    p_smooth = np.zeros(n_bins)
    for t in range(n_frames):
        y2 = power[:, t]
        gamma = y2 / n
        spp = 1.0 / (1.0 + prior_ratio * (1 + xi_h1) * np.exp(np.minimum(-gamma * c, 0) ))
        # avoid stagnation
        p_smooth = 0.9 * p_smooth + 0.1 * spp
        spp = np.where(p_smooth > 0.99, np.minimum(spp, 0.99), spp)
        est = (1 - spp) * y2 + spp * n
        n = p.noise_alpha * n + (1 - p.noise_alpha) * est
        n = np.maximum(n, 1e-12)
        noise[:, t] = n
    return noise


def _imcra_noise(power: np.ndarray, p: Params, hop_s: float) -> np.ndarray:
    """Improved minima-controlled recursive averaging (Cohen 2003), offline.

    Two iterations of smoothing + minimum tracking give a speech-absence probability q;
    a speech-presence probability then controls the recursive noise update.
    Minima are taken over a centred window (the whole file is available).
    """
    n_bins, n_fr = power.shape
    win = max(3, int(round(p.imcra_win_s / hop_s)) | 1)
    b = np.array([0.25, 0.5, 0.25])                       # frequency smoothing (w = 1)

    def smooth(pw, ind=None):
        if ind is None:
            sf = convolve1d(pw, b, axis=0, mode="nearest")
        else:
            num = convolve1d(pw * ind, b, axis=0, mode="nearest")
            den = convolve1d(ind.astype(float), b, axis=0, mode="nearest")
            sf = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        out = np.empty_like(pw)
        prev = np.where(np.isnan(sf[:, 0]), pw[:, 0], sf[:, 0])
        for t in range(n_fr):
            cur = sf[:, t]
            prev = np.where(np.isnan(cur), prev, p.imcra_alpha_s * prev + (1 - p.imcra_alpha_s) * np.nan_to_num(cur))
            out[:, t] = prev
        return out

    S = smooth(power)
    smin = minimum_filter1d(S, win, axis=1, mode="nearest")
    ind = (power / (p.imcra_bmin * smin) < p.imcra_gamma0) & (S / (p.imcra_bmin * smin) < p.imcra_zeta0)
    S2 = smooth(power, ind)
    smin2 = np.maximum(minimum_filter1d(S2, win, axis=1, mode="nearest"), 1e-14)
    gmin = power / (p.imcra_bmin * smin2)
    zeta = S / (p.imcra_bmin * smin2)
    g1 = p.imcra_gamma1
    q = np.where(gmin <= 1, 1.0, np.clip((g1 - gmin) / (g1 - 1), 0, 1))
    q = np.where(zeta < p.imcra_zeta0, q, 0.0)
    q = np.minimum(q, 0.99)
    noise = np.empty_like(power)
    lam = smin2[:, 0] * p.imcra_bmin
    a_prev = None
    for t in range(n_fr):
        gamma = np.minimum(power[:, t] / lam, 1e4)
        ml = np.maximum(gamma - 1, 0)
        xi = ml if a_prev is None else 0.92 * a_prev / lam + 0.08 * ml
        xi = np.maximum(xi, 10 ** (-2.5))
        v = np.minimum(gamma * xi / (1 + xi), 500)
        spp = 1 / (1 + q[:, t] / (1 - q[:, t]) * (1 + xi) * np.exp(-v))
        a_prev = _lsa_gain(xi, gamma) ** 2 * power[:, t]
        ad = p.imcra_alpha_d + (1 - p.imcra_alpha_d) * spp
        lam = ad * lam + (1 - ad) * power[:, t]
        noise[:, t] = lam
    return np.maximum(p.imcra_beta * noise, 1e-12)


def _quantile_noise(power: np.ndarray, win: int, q: float) -> np.ndarray:
    """Per-bin sliding q-quantile over frames, bias-corrected for an exponential PSD."""
    win = max(3, win | 1)
    qn = percentile_filter(power, percentile=100 * q, size=(1, win), mode="reflect")
    return np.maximum(qn / -np.log1p(-q), 1e-12)


def _tone_mask(power: np.ndarray, p: Params, hop_s: float) -> np.ndarray:
    """Bins belonging to long, narrowband peaks (tracks may move by +-2 bins/frame)."""
    ldb = 10 * np.log10(power)
    peak = ldb - median_filter(ldb, size=(31, 1), mode="nearest") > p.tone_thresh_db
    peak &= (ldb >= np.roll(ldb, 1, axis=0)) & (ldb >= np.roll(ldb, -1, axis=0))
    near = maximum_filter(peak, size=(5, 1))          # allow small frequency drift
    # duration: count of consecutive frames where a peak is nearby
    min_len = max(2, int(round(p.tone_min_s / hop_s)))
    run = np.zeros_like(near, dtype=np.int32)
    for t in range(near.shape[1]):
        run[:, t] = np.where(near[:, t], (run[:, t - 1] if t else 0) + 1, 0)
    # backward pass so the whole run (not only its tail) is marked
    long_ = np.zeros_like(near)
    for t in range(near.shape[1] - 1, -1, -1):
        nxt = long_[:, t + 1] if t + 1 < near.shape[1] else np.zeros(near.shape[0], bool)
        long_[:, t] = near[:, t] & ((run[:, t] >= min_len) | nxt)
    return maximum_filter(long_ & peak, size=(3, 1))  # main-lobe neighbours


def _percussive_power(power: np.ndarray, k: int) -> np.ndarray:
    """HPSS (Fitzgerald 2010): Wiener-like soft mask of the percussive component."""
    h = median_filter(power, size=(1, k), mode="nearest")   # along time  -> harmonic
    v = median_filter(power, size=(k, 1), mode="nearest")   # along freq  -> percussive
    return power * v ** 2 / (v ** 2 + h ** 2 + 1e-30)


def apply_noise_modules(power, noise, p: Params, hop_s: float, sr: int):
    if p.fast_scale > 0:
        qn = _quantile_noise(power, int(round(p.fast_win_s / hop_s)), p.fast_q)
        f = np.arange(power.shape[0]) * sr / (2 * (power.shape[0] - 1))
        band = (f <= p.fast_fmax_hz)[:, None]
        noise = np.where(band, np.maximum(noise, p.fast_scale * qn), noise)
    if p.tone_scale > 0:
        mask = _tone_mask(power, p, hop_s)
        noise = np.where(mask, np.maximum(noise, p.tone_scale * power), noise)
    if p.trans_scale > 0:
        noise = noise + p.trans_scale * _percussive_power(power, p.trans_kernel)
    return noise


def estimate_noise(power: np.ndarray, p: Params, hop_s: float = 0.008,
                   noise0: np.ndarray | None = None) -> np.ndarray:
    if noise0 is None:
        noise0 = _initial_noise(power, p)
    if not p.track_noise:
        noise = np.repeat(noise0[:, None], power.shape[1], axis=1)
    else:
        noise = _track_noise(power, noise0, p)
        if p.two_pass:
            # second pass starts from a better-informed initial estimate
            start = np.minimum(noise0, np.median(noise, axis=1))
            noise = _track_noise(power, np.maximum(start, 1e-12), p)
    if p.imcra != "off":
        im = _imcra_noise(power, p, hop_s)
        noise = im if p.imcra == "replace" else np.minimum(noise, im)
    if p.quantile_win_s > 0:
        qn = _quantile_noise(power, int(round(p.quantile_win_s / hop_s)), p.quantile_q)
        if p.quantile_mix == "replace":
            noise = qn
        elif p.quantile_mix == "max":
            noise = np.maximum(noise, qn)
        else:
            noise = np.minimum(noise, qn)
    return noise


# ----------------------------------------------------------------------------
# Gain computation
# ----------------------------------------------------------------------------
def _lsa_gain(xi, gamma):
    v = np.maximum(xi * gamma / (1 + xi), 1e-10)
    return xi / (1 + xi) * np.exp(0.5 * exp1(v))


def _beta_gain(xi, gamma, beta):
    """beta-order MMSE spectral amplitude estimator (You, Koh & Rahardja 2005).

    G = sqrt(v)/gamma * [Gamma(beta/2 + 1) * M(-beta/2, 1; -v)]^(1/beta);
    for large v the estimator tends to the Wiener gain xi / (1 + xi).
    """
    if abs(beta) < 1e-3:
        return _lsa_gain(xi, gamma)
    v = np.maximum(xi * gamma / (1 + xi), 1e-10)
    small = v < 300
    vs = np.where(small, v, 1.0)
    m = hyp1f1(-beta / 2, 1.0, -vs)
    g = np.sqrt(vs) / gamma * np.maximum(gamma_fn(beta / 2 + 1) * m, 1e-30) ** (1 / beta)
    return np.where(small, g, xi / (1 + xi))


def _map_sg_gain(xi, gamma, mu, nu):
    """Joint MAP amplitude estimator with a super-Gaussian prior (Lotter & Vary 2005)."""
    u = 0.5 - mu / (4 * np.sqrt(np.maximum(gamma * xi, 1e-10)))
    return u + np.sqrt(u ** 2 + nu / (2 * np.maximum(gamma, 1e-10)))


def _gain_fn(name: str, xi, gamma, p=None):
    if name == "wiener":
        return xi / (1 + xi)
    if name == "beta":
        return _beta_gain(xi, gamma, p.beta_order)
    if name == "map_sg":
        return _map_sg_gain(xi, gamma, p.sg_mu, p.sg_nu)
    return _lsa_gain(xi, gamma)


def _adapt_params(power: np.ndarray, noise: np.ndarray, p: Params) -> Params:
    """Rule-based parameter selection from the estimated input (global) SNR.

    snr_est = 10 log10( sum max(|Y|^2 - N, 0) / sum N ); floor and over-estimation are
    linearly interpolated between their low-SNR and high-SNR values.
    """
    speech = np.maximum(power - noise, 0).sum()
    snr = 10 * np.log10(max(speech, 1e-20) / max(noise.sum(), 1e-20))
    w = float(np.clip((snr - p.adapt_snr_lo) / (p.adapt_snr_hi - p.adapt_snr_lo), 0, 1))
    return replace(p,
                   gain_floor_db=(1 - w) * p.adapt_floor_lo_db + w * p.adapt_floor_hi_db,
                   noise_overest=(1 - w) * p.adapt_over_lo + w * p.adapt_over_hi)


def estimate_input_snr(x: np.ndarray, sr: int, p: Params = None) -> float:
    """Diagnostic helper: the global SNR estimate used by the adaptive rule."""
    p = p or DEFAULT
    n_fft, hop = _stft_sizes(sr, p)
    xp = np.pad(x, (0, max(0, n_fft - len(x))))
    power = np.abs(_stft(xp, sr, n_fft, hop, p.window)) ** 2 + 1e-14
    noise = estimate_noise(power, p, hop / sr)
    return 10 * np.log10(max(np.maximum(power - noise, 0).sum(), 1e-20) / noise.sum())


def _band_vec(values, edges, n_bins, sr):
    f = np.arange(n_bins) * sr / (2 * (n_bins - 1))
    idx = np.searchsorted(np.asarray(edges, dtype=float), f, side="right")
    return np.asarray(values, dtype=float)[idx]


def compute_gain(power: np.ndarray, noise: np.ndarray, p: Params,
                 over=None, xi_min=None) -> np.ndarray:
    over = p.noise_overest if over is None else over
    noise = noise * (over[:, None] if np.ndim(over) == 1 else over)
    gamma = np.minimum(power / noise, 1e4)
    xi_min = 10 ** (p.xi_min_db / 10) if xi_min is None else xi_min
    xi_min = np.broadcast_to(xi_min[:, None] if np.ndim(xi_min) == 1 else xi_min, power.shape)
    g_min = 10 ** (p.gain_floor_db / 20)
    n_bins, n_frames = power.shape
    gain = np.empty_like(power)
    prev_amp2 = None
    xi_h1 = 10 ** (p.spp_xi_h1_db / 10)
    for t in range(n_frames):
        g_t = gamma[:, t]
        ml = np.maximum(g_t - 1, 0)
        if prev_amp2 is None:
            xi = np.maximum(ml, xi_min[:, t])
        else:
            xi = np.maximum(p.dd_alpha * prev_amp2 / noise[:, t] + (1 - p.dd_alpha) * ml,
                            xi_min[:, t])
        if p.gain in ("lsa_wiener", "lsa_ss"):
            # gain fusion: weighted geometric mean of LSA and Wiener / SS gains
            other = (xi / (1 + xi) if p.gain == "lsa_wiener" else
                     np.sqrt(np.maximum(1 - 1 / np.maximum(g_t, 1e-10), g_min ** 2)))
            g = _lsa_gain(xi, g_t) ** p.mix_w * np.maximum(other, 1e-6) ** (1 - p.mix_w)
        elif p.gain == "ss":
            # power spectral subtraction (Boll 1979): alpha = noise_overest, beta = g_min^2
            g = np.sqrt(np.maximum(1 - 1 / np.maximum(g_t, 1e-10), g_min ** 2))
        elif p.gain == "wiener":
            g = xi / (1 + xi)
        elif p.gain in ("beta", "map_sg"):
            g = _gain_fn(p.gain, xi, g_t, p)
        else:
            g = _lsa_gain(xi, g_t)
            if p.gain == "omlsa":
                v = xi * g_t / (1 + xi)
                q = 0.5  # a-priori speech absence probability
                spp = 1 / (1 + q / (1 - q) * (1 + xi) * np.exp(-np.minimum(v, 50)))
                g = g ** spp * g_min ** (1 - spp)
        g = np.clip(g, g_min, 1.0)
        gain[:, t] = g
        prev_amp2 = (g ** 2) * power[:, t]
    return gain


def _bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def masking_threshold(speech_pow: np.ndarray, sr: int) -> np.ndarray:
    """Johnston (1988) simultaneous-masking threshold per STFT bin.

    speech_pow: (bins, frames) estimate of the clean speech power.
    Steps: Bark band energies -> spreading function (Schroeder) -> tonality offset from the
    spectral flatness -> renormalisation -> spread back to bins.
    Implemented with element-wise products only (no BLAS calls).
    """
    n_bins = speech_pow.shape[0]
    f = np.arange(n_bins) * sr / (2 * (n_bins - 1))
    band = np.floor(_bark(f)).astype(int)
    n_b = band.max() + 1
    onehot = band[None, :] == np.arange(n_b)[:, None]                   # (bands, bins)
    counts = onehot.sum(axis=1).astype(float)
    energy = (onehot[:, :, None] * speech_pow[None, :, :]).sum(axis=1)  # (bands, frames)
    dz = np.arange(n_b)[:, None] - np.arange(n_b)[None, :]              # masker j -> maskee i
    sf_db = 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1 + (dz + 0.474) ** 2)
    sf = 10 ** (sf_db / 10)                                             # (bands, bands)
    spread = (sf[:, :, None] * energy[None, :, :]).sum(axis=1)          # (bands, frames)
    spread /= sf.sum(axis=1)[:, None]                                   # renormalise
    # tonality from the spectral flatness of each frame (Johnston: SFM_max = -60 dB)
    sp = np.maximum(speech_pow, 1e-20)
    sfm_db = 10 * np.log10(np.exp(np.mean(np.log(sp), axis=0)) / np.mean(sp, axis=0))
    tonal = np.minimum(sfm_db / -60.0, 1.0)                             # (frames,)
    idx = np.arange(n_b)[:, None]
    offset_db = tonal[None, :] * (14.5 + idx) + (1 - tonal[None, :]) * 5.5
    thr_band = spread * 10 ** (-offset_db / 10)
    return thr_band[band] / counts[band][:, None]                       # per bin


def _cepstral_smooth(gain: np.ndarray, n_fft: int, sr: int, p: Params) -> np.ndarray:
    """Selective cepstro-temporal smoothing of the log gain.

    Each frame's log gain is taken to the cepstral (quefrency) domain. Low quefrencies
    (spectral envelope) and the pitch peak are smoothed only lightly over time; all other
    quefrencies (fine-structure fluctuations = musical noise) are smoothed strongly.
    """
    lg = np.log(np.maximum(gain, 1e-6))
    cep = np.fft.irfft(lg, n=n_fft, axis=0)           # (n_fft, frames), real, symmetric
    n_q = n_fft // 2 + 1
    q_env = max(1, int(round(p.cep_env_ms * 1e-3 * sr)))
    q_lo, q_hi = int(sr / 400), min(int(sr / 70), n_q - 2)   # pitch 70..400 Hz
    out = np.empty_like(cep)
    state = cep[:, 0].copy()
    for t in range(cep.shape[1]):
        alpha = np.full(n_q, p.cep_alpha_rest)
        alpha[:q_env] = p.cep_alpha_env
        if q_hi > q_lo:
            k = q_lo + int(np.argmax(cep[q_lo:q_hi, t]))
            alpha[max(k - 2, 0):k + 3] = p.cep_alpha_pitch
        a = np.concatenate([alpha, alpha[1:n_fft - n_q + 1][::-1]])   # mirror to full length
        state = a * state + (1 - a) * cep[:, t]
        out[:, t] = state
    return np.exp(np.fft.rfft(out, axis=0).real)


def estimate_f0(x: np.ndarray, sr: int, n: int, hop: int, vth: float):
    """Frame-wise f0 [Hz] (0 = unvoiced) by normalised autocorrelation, frames aligned with
    scipy.signal.stft(boundary='zeros'). Octave errors are reduced by taking the smallest
    lag whose correlation is within 90 % of the maximum."""
    xp = np.pad(x, (n // 2, n // 2 + n))
    n_fr = 1 + (len(x) + n - n) // hop + 1
    idx = np.arange(n)[None, :] + hop * np.arange(n_fr)[:, None]
    idx = np.minimum(idx, len(xp) - 1)
    fr = xp[idx] * np.hanning(n)[None, :]
    spec = np.fft.rfft(fr, 2 * n, axis=1)
    r = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    r = r / (np.maximum(r[:, :1], 1e-20))
    lag_lo, lag_hi = int(sr / 400), int(sr / 70)
    seg = r[:, lag_lo:lag_hi]
    peak = seg.max(axis=1)
    cand = seg >= 0.9 * peak[:, None]
    loc = (seg >= np.roll(seg, 1, axis=1)) & (seg >= np.roll(seg, -1, axis=1))
    first = np.argmax(cand & loc, axis=1)
    k = np.clip(first, 1, seg.shape[1] - 2)
    a, b, c = seg[np.arange(len(k)), k - 1], seg[np.arange(len(k)), k], seg[np.arange(len(k)), k + 1]
    den = a - 2 * b + c
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    lag = lag_lo + k + np.clip(delta, -1, 1)
    energy = 10 * np.log10(np.maximum((fr ** 2).sum(axis=1), 1e-20))
    voiced = (b >= vth) & (energy > energy.max() - 40)
    f0 = np.where(voiced, sr / lag, 0.0)
    f0s = median_filter(f0, size=5, mode="nearest")
    return np.where(voiced, np.where(f0s > 0, f0s, f0), 0.0)


def _protect_low_harmonics(gain, z, power, noise, sr, n_fft, hop, length, p: Params):
    """Raise the gain to a floor on voiced low-frequency harmonics.

    f0 is estimated on the first-pass enhanced waveform (frames centred like the main STFT).
    Only bins near h * f0 (h >= 1, below lhp_fmax_hz) whose noisy power exceeds the noise
    estimate by lhp_gamma_db are protected, so stationary hum between harmonics stays suppressed.
    """
    s_hat = _istft(gain * z, sr, n_fft, hop, length, p.window)
    n = 2 * int(round(p.lhp_frame_ms * 1e-3 * sr / 2))
    n_frames = gain.shape[1]
    f0 = estimate_f0(s_hat, sr, n, hop, p.lhp_vth)[:n_frames]
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    fk = np.arange(gain.shape[0]) * sr / n_fft
    k_max = int(np.searchsorted(fk, p.lhp_fmax_hz, side="right"))
    f = fk[:k_max, None]
    f0s = np.where(f0 > 0, f0, 1e9)[None, :]          # unvoiced -> h = 0 (not protected)
    h = np.round(f / f0s)
    near = (h >= 1) & (np.abs(f - h * f0s) <= p.lhp_lobe_hz)
    strong = power[:k_max] > noise[:k_max] * 10 ** (p.lhp_gamma_db / 10)
    prot = near & strong & (f0[None, :] > 0)
    out = gain.copy()
    out[:k_max] = np.where(prot, np.maximum(gain[:k_max], 10 ** (p.lhp_floor_db / 20)), gain[:k_max])
    return out


def reconstruct_phase(y: np.ndarray, noisy: np.ndarray, sr: int, p: Params) -> np.ndarray:
    """Replace the (noisy) phase of the enhanced signal around voiced harmonics.

    Harmonic phases are advanced frame-to-frame by omega_h * hop (Krawczyk & Gerkmann 2014);
    bins within the main lobe follow the linear phase of the symmetric analysis window.
    The reconstructed phase is blended in more strongly where the local SNR is low.
    """
    n = 2 * int(round(p.pr_frame_ms * 1e-3 * sr / 2))
    hop = n // 4
    if len(y) < 2 * n:
        return y
    z = _stft(y, sr, n, hop, "hann")
    zn = _stft(noisy, sr, n, hop, "hann")
    n_bins, n_frames = z.shape
    noise = _initial_noise(np.abs(zn) ** 2 + 1e-14, p)
    f0 = estimate_f0(y, sr, n, hop, p.pr_vth)[:n_frames]
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    phase = np.angle(z)
    rec = phase.copy()
    touched = np.zeros(z.shape, dtype=bool)
    psi = {}
    prev_f0 = 0.0
    wk = 2 * np.pi * np.arange(n_bins) / n
    for t in range(n_frames):
        if f0[t] <= 0:
            psi, prev_f0 = {}, 0.0
            continue
        n_h = int(min(p.pr_fmax_hz, sr / 2 - 2 * sr / n) // f0[t])
        new = {}
        for h in range(1, n_h + 1):
            om = 2 * np.pi * h * f0[t] / sr
            kh = int(round(om * n / (2 * np.pi)))
            if h in psi and prev_f0 > 0:
                ps = psi[h] + 0.5 * (om + 2 * np.pi * h * prev_f0 / sr) * hop
            else:   # onset: initialise from the observed phase
                ps = phase[kh, t] + (wk[kh] - om) * n / 2
            new[h] = ps
            lo, hi = max(kh - p.pr_lobe, 0), min(kh + p.pr_lobe, n_bins - 1)
            ks = np.arange(lo, hi + 1)
            rec[ks, t] = ps - (wk[ks] - om) * n / 2
            touched[ks, t] = True
        psi, prev_f0 = new, f0[t]
    snr = np.abs(z) ** 2 / noise[:, None]
    w = p.pr_amount / (1 + snr / 10 ** (p.pr_snr0_db / 10))
    w = np.where(touched, w, 0.0)
    mix = (1 - w) * np.exp(1j * phase) + w * np.exp(1j * rec)
    z_new = np.abs(z) * np.exp(1j * np.angle(mix))
    return _istft(z_new, sr, n, hop, len(y), "hann")


def modulation_enhance(x: np.ndarray, sr: int, p: Params) -> np.ndarray:
    """MMSE (log-spectral) estimation of the short-time *modulation* magnitude.

    acoustic STFT |X(k, t)| -> per-bin modulation STFT -> DD + LSA gain over modulation
    frames -> inverse modulation STFT -> enhanced acoustic magnitude + noisy acoustic phase.
    """
    n = 2 * int(round(p.mod_ac_ms * 1e-3 * sr / 2))
    hop = max(1, int(round(p.mod_ac_hop_ms * 1e-3 * sr)))
    length = len(x)
    xp = np.pad(x, (0, max(0, n - length)))
    z = _stft(xp, sr, n, hop, "hamming")
    mag, ph = np.abs(z), np.angle(z)
    fs_mod = sr / hop                                   # modulation sampling rate
    nm = 2 * int(round(p.mod_win_ms * 1e-3 * fs_mod / 2))
    hm = max(1, int(round(nm * p.mod_hop_ratio)))
    T = mag.shape[1]
    if T < 2 * nm:
        return x
    _, _, zm = signal.stft(mag, fs=fs_mod, window="hamming", nperseg=nm, noverlap=nm - hm,
                           nfft=nm, boundary="zeros", padded=True, axis=-1)   # (k, m, frames)
    pw = np.abs(zm) ** 2 + 1e-14
    k_ac, n_mod, n_fr = pw.shape
    flat = pw.reshape(k_ac * n_mod, n_fr)
    q = replace(p, init_fraction=p.mod_init_fraction)
    noise = _initial_noise(flat, q)[:, None] * p.mod_overest
    gamma = np.minimum(flat / noise, 1e4)
    xi_min = 10 ** (p.mod_xi_min_db / 10)
    gain = np.empty_like(flat)
    prev = None
    for t in range(n_fr):
        ml = np.maximum(gamma[:, t] - 1, 0)
        xi = ml if prev is None else p.mod_dd_alpha * prev / noise[:, 0] + (1 - p.mod_dd_alpha) * ml
        xi = np.maximum(xi, xi_min)
        g = np.clip(_lsa_gain(xi, gamma[:, t]), np.sqrt(xi_min), 1.0)
        gain[:, t] = g
        prev = g ** 2 * flat[:, t]
    zm_hat = zm * gain.reshape(k_ac, n_mod, n_fr)
    _, mag_hat = signal.istft(zm_hat, fs=fs_mod, window="hamming", nperseg=nm, noverlap=nm - hm,
                              nfft=nm, input_onesided=True, boundary=True,
                              time_axis=-1, freq_axis=-2)
    mag_hat = np.maximum(mag_hat[:, :T], 0)
    if mag_hat.shape[1] < T:
        mag_hat = np.pad(mag_hat, ((0, 0), (0, T - mag_hat.shape[1])))
    return _istft(mag_hat * np.exp(1j * ph), sr, n, hop, length, "hamming")


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def enhance(x: np.ndarray, sr: int, p: Params = DEFAULT) -> np.ndarray:
    """Enhance a mono float waveform; returns the same length."""
    x = np.asarray(x, dtype=np.float64)
    if p.mod_mode == "only":
        return modulation_enhance(x, sr, p)
    y = _enhance_stft(x, sr, p)
    if p.mr_mode != "off":
        y = _fuse_resolutions(y, _enhance_stft(x, sr, replace(p, frame_ms=p.mr_frame_ms)), sr, p)
    if p.mod_mode == "post":
        y = p.mod_mix * modulation_enhance(y, sr, p) + (1 - p.mod_mix) * y
    return y


def _fuse_resolutions(y_long, y_short, sr, p: Params):
    """Combine the long- and short-frame outputs (waveform blend, or per-bin magnitudes with
    the long-frame phase)."""
    w = p.mr_weight
    if p.mr_mode == "time":
        return (1 - w) * y_long + w * y_short
    n = 2 * int(round(p.mr_an_ms * 1e-3 * sr / 2))
    hop = n // 4
    length = len(y_long)
    if length < n:
        return (1 - w) * y_long + w * y_short
    zl, zs = _stft(y_long, sr, n, hop), _stft(y_short, sr, n, hop)
    al, as_ = np.abs(zl) + 1e-12, np.abs(zs) + 1e-12
    if p.mr_mode == "geo":
        mag = al ** (1 - w) * as_ ** w
    elif p.mr_mode == "min":
        mag = (1 - w) * al + w * np.minimum(al, as_)
    else:
        raise ValueError(p.mr_mode)
    return _istft(mag * np.exp(1j * np.angle(zl)), sr, n, hop, length)


def _consistent_istft(target, post_var, sr, n_fft, hop, length, p: Params):
    """Time signal whose STFT is closest to `target` in a weighted LS sense.

    Minimises sum_tf w_tf |STFT(s)_tf - target_tf|^2 with w = posterior variance ** -cw_pow
    (Le Roux & Vincent 2013: consistent Wiener filtering), solved by conjugate gradients on
    the normal equations and started from the plain ISTFT. Frames follow scipy.signal.stft
    (boundary='zeros', padded=True); w = 1 for every bin gives the ordinary ISTFT solution.
    """
    win = _window(p.window, n_fft)
    n_fr = target.shape[1]
    L = n_fft + hop * (n_fr - 1)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_fr)[:, None]
    tgt = (target * win.sum()).T                                   # (frames, bins), unscaled
    w = np.maximum(post_var, 1e-14).T ** -p.cw_pow
    w = w / np.mean(w)

    def fwd(u):
        return np.fft.rfft(u[idx] * win[None, :], axis=1)

    def adj(Y):
        # Re(R^H D Y) per frame (D: one-sided bin multiplicity) == irfft(Y) * n, then OLA
        out = np.zeros(L)
        np.add.at(out, idx, np.fft.irfft(Y, n=n_fft, axis=1) * n_fft * win[None, :])
        return out

    def normal(u):
        return adj(w * fwd(u))

    y0 = _istft(target, sr, n_fft, hop, max(length, 1), p.window)
    u = np.zeros(L)
    off = n_fft // 2
    seg = min(len(y0), L - off)
    u[off:off + seg] = y0[:seg]
    b = adj(w * tgt)
    r = b - normal(u)
    q = r.copy()
    rr = r @ r
    for _ in range(p.cw_iters):
        Aq = normal(q)
        alpha = rr / max(q @ Aq, 1e-30)
        u += alpha * q
        r -= alpha * Aq
        rr_new = r @ r
        q = r + rr_new / max(rr, 1e-30) * q
        rr = rr_new
    return u[off:off + length]


def _enhance_stft(x: np.ndarray, sr: int, p: Params) -> np.ndarray:
    """Single-resolution STFT enhancement (noise estimate -> DD/LSA -> HRNR -> ISTFT)."""
    length = len(x)
    n_fft, hop = _stft_sizes(sr, p)
    if length < n_fft:
        xp = np.pad(x, (0, n_fft - length))
    else:
        xp = x
    if p.hp_hz > 0:
        # Butterworth high-pass pre-filter, zero phase (offline)
        sos = signal.butter(p.hp_order, p.hp_hz, btype="highpass", fs=sr, output="sos")
        xp = signal.sosfiltfilt(sos, xp, padlen=min(3 * (2 * len(sos) + 1), len(xp) - 1))
    z = _stft(xp, sr, n_fft, hop, p.window)
    power = np.abs(z) ** 2 + 1e-14
    noise = estimate_noise(power, p, hop / sr)
    noise = apply_noise_modules(power, noise, p, hop / sr, sr)
    if p.noise_smooth_t > 1:
        noise = uniform_filter1d(noise, p.noise_smooth_t, axis=1, mode="nearest")
    if p.adapt:
        p = _adapt_params(power, noise, p)
    n_bins = power.shape[0]
    bo = (p.band_over_lo, p.band_over_mid, p.band_over_hi)
    bx = (p.band_ximin_lo_db, p.band_ximin_mid_db, p.band_ximin_hi_db)
    over = p.noise_overest * _band_vec(bo, p.band_edges_hz, n_bins, sr)
    xi_min = 10 ** ((p.xi_min_db + _band_vec(bx, p.band_edges_hz, n_bins, sr)) / 10)
    gain = compute_gain(power, noise, p, over, xi_min)
    over, xi_min = over[:, None], xi_min[:, None]
    if p.mask_on:
        # masking threshold from the first-pass speech estimate; suppression is relaxed
        # where the (over-estimated) noise lies below the threshold, strengthened elsewhere
        thr = masking_threshold(gain ** 2 * power, sr)
        mnr = 10 * np.log10(thr / (noise * over))
        w = np.clip((mnr - p.mask_lo_db) / (p.mask_hi_db - p.mask_lo_db), 0, 1)
        over = over * ((1 - w) * p.mask_over_strong + w * p.mask_over_weak)
        xi_min = xi_min * 10 ** (((1 - w) * p.mask_ximin_strong_db + w * p.mask_ximin_weak_db) / 10)
        gain = compute_gain(power, noise, p, over, xi_min)
    if p.tsnr or p.hrnr:
        nz = noise * over
        gamma = np.minimum(power / nz, 1e4)
        g_min = 10 ** (p.gain_floor_db / 20)
        if p.tsnr:
            # TSNR (Plapous et al. 2006): a-priori SNR re-estimated from the DD output,
            # removing the one-frame delay of the decision-directed estimate.
            xi2 = np.maximum(gain ** 2 * gamma, xi_min)
            g2 = _gain_fn(p.tsnr_gain, xi2, gamma, p)
            gain = p.tsnr_mix * np.clip(g2, g_min, 1.0) + (1 - p.tsnr_mix) * gain
        if p.hrnr:
            # HRNR: half-wave rectification of the enhanced waveform regenerates the
            # harmonics suppressed by the gain; they enter a new a-priori SNR estimate.
            for _ in range(p.hrnr_iters):
                s_hat = _istft(gain * z, sr, n_fft, hop, len(xp), p.window)
                s_nl = np.abs(s_hat) if p.hrnr_nl == "full" else np.maximum(s_hat, 0.0)
                harmo = _stft(s_nl, sr, n_fft, hop, p.window)
                rho = gain ** p.hrnr_rho_pow
                xi3 = (rho * gain ** 2 * power + (1 - rho) * np.abs(harmo) ** 2) / nz
                xi3 = np.maximum(p.hrnr_scale * xi3, xi_min)
                g3 = np.clip(_gain_fn(p.hrnr_gain, xi3, gamma, p), g_min, 1.0)
                gain = p.hrnr_mix * g3 + (1 - p.hrnr_mix) * gain
    if p.lhp_on:
        gain = _protect_low_harmonics(gain, z, power, noise * over, sr, n_fft, hop, len(xp), p)
    if p.cep_smooth:
        g_min = 10 ** (p.gain_floor_db / 20)
        gs = _cepstral_smooth(gain, n_fft, sr, p)
        gain = np.clip(p.cep_mix * gs + (1 - p.cep_mix) * gain, g_min, 1.0)
    if p.refine_th > 0:
        # two-stage: re-estimate the noise PSD from bins the 1st stage judged speech-absent
        absent = gain <= p.refine_th
        count = absent.sum(axis=1)
        ref = (power * absent).sum(axis=1) / np.maximum(count, 1)
        base = _initial_noise(power, p)
        noise0 = np.where(count >= p.refine_min_frames, ref, base)
        noise = estimate_noise(power, p, hop / sr, noise0=np.maximum(noise0, 1e-12))
        gain = compute_gain(power, noise, p)
    if p.gain_smooth_t > 1 or p.gain_smooth_f > 1:
        g_min = 10 ** (p.gain_floor_db / 20)
        # smooth in the log domain, zero-phase (offline) window
        lg = np.log(np.maximum(gain, 1e-6))
        lg = uniform_filter(lg, size=(p.gain_smooth_f, p.gain_smooth_t), mode="nearest")
        gain = np.clip(np.exp(lg), g_min, 1.0)
    if p.cw_iters > 0:
        y = _consistent_istft(gain * z, gain * noise * over, sr, n_fft, hop, len(xp), p)[:length]
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
    else:
        y = _istft(gain * z, sr, n_fft, hop, length, p.window)
    if p.phase_on:
        y = reconstruct_phase(y, x, sr, p)
    return y


def enhance_multichannel(x: np.ndarray, sr: int, p: Params = DEFAULT) -> np.ndarray:
    """x: (n,) or (n, channels). Each channel is processed independently."""
    if x.ndim == 1:
        return enhance(x, sr, p)
    return np.stack([enhance(x[:, c], sr, p) for c in range(x.shape[1])], axis=1)


__all__ = ["Params", "DEFAULT", "enhance", "enhance_multichannel", "replace"]
