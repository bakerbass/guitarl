"""Quick ground-truth test of the spectral score on reference recordings.

Positive class: GB_NH harmonic clips (pitches69=fret7, pitches74=fret5, pitches78=fret4)
Negative class: dead_note clips, general_note clips at the same pitch
"""
import sys
import numpy as np
from scipy.signal import welch, resample_poly
from scipy.io import wavfile
from pathlib import Path
from math import gcd

D_STRING_OPEN_FREQ = 146.83
FRET_TO_HARMONIC   = {4: 5, 5: 4, 7: 3}
PITCH_TO_FRET      = {69: 7, 74: 5, 78: 4}
BANDWIDTH          = 0.03
NOISE_FLOOR        = 80.0
SR                 = 22050
HOP                = 512
THRESHOLD          = 0.65

BASE = Path(__file__).parent.parent / "HarmonicsClassifier" / "note_clips"


def load_wav(path, target_sr=SR):
    """Load WAV with scipy (no libsndfile needed), resample to target_sr."""
    file_sr, data = wavfile.read(str(path))
    # Convert to float32 mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    # Resample if needed
    if file_sr != target_sr:
        g = gcd(file_sr, target_sr)
        data = resample_poly(data, target_sr // g, file_sr // g).astype(np.float32)
    return data


def onset_detect_simple(y, sr, hop):
    """Simple energy-based onset: first frame above 10% of max RMS."""
    frame_len = hop
    n_frames = len(y) // frame_len
    rms = np.array([np.sqrt(np.mean(y[i*frame_len:(i+1)*frame_len]**2))
                    for i in range(n_frames)])
    thresh = 0.10 * rms.max()
    hits = np.where(rms > thresh)[0]
    if len(hits) == 0:
        return 0
    onset_sample = hits[0] * frame_len
    return min(onset_sample, sr)   # cap at 1 s


def spectral_score(wav_path, target_fret):
    y = load_wav(wav_path, target_sr=SR)

    # Onset-align
    onset = onset_detect_simple(y, SR, HOP)
    y = y[onset:]

    # Skip 300 ms transient, take up to 2 s steady-state
    skip = int(0.3 * SR)
    y = y[skip: skip + int(2.0 * SR)]
    if len(y) < int(0.25 * SR):
        return None, {}

    nperseg = min(4096, len(y))
    freqs, psd = welch(y, fs=SR, nperseg=nperseg, noverlap=nperseg // 2)

    harmonic_n = FRET_TO_HARMONIC[target_fret]
    harm_fund  = harmonic_n * D_STRING_OPEN_FREQ
    f0         = D_STRING_OPEN_FREQ

    # Energy at desired partials
    desired = 0.0
    partial = harm_fund
    while partial < SR / 2:
        lo, hi = partial * (1 - BANDWIDTH), partial * (1 + BANDWIDTH)
        desired += float(psd[(freqs >= lo) & (freqs <= hi)].sum())
        partial += harm_fund

    # Energy at open-string fundamental (should be suppressed)
    f0_e = float(psd[(freqs >= f0 * (1-BANDWIDTH)) & (freqs <= f0 * (1+BANDWIDTH))].sum())

    total = float(psd[freqs >= NOISE_FLOOR].sum()) + 1e-12

    her       = desired / total
    fund_sup  = 1.0 - float(np.clip(f0_e / total * 5.0, 0, 1))
    sig       = float(np.clip(total / 1e-4, 0, 1))
    score     = float(np.clip(0.60*her + 0.25*fund_sup + 0.15*sig, 0, 1))

    return score, dict(HER=her, fund_sup=fund_sup, sig=sig, total_e=total)


def row(tag, label, fname, fret, s, b):
    ok = "✓" if tag else "✗"
    print(f"  {ok} {label:<14} {fname[-38:]:<38} f{fret}  {s:.3f}  HER={b['HER']:.3f}  FSupp={b['fund_sup']:.3f}")


print("=" * 78)
print(f"{'':3}{'LABEL':<15} {'FILE':<38} {'FRT':>3}  {'SCORE':>5}  {'HER':>8}  {'FundSupp':>8}")
print("=" * 78)

correct = total_tested = 0

# ── Harmonics (expect score >= THRESHOLD) ──────────────────────────────
for pitch, fret in sorted(PITCH_TO_FRET.items()):
    wavs = sorted((BASE / "harmonic").glob(f"GB_NH*pitches{pitch}*.wav"))[:5]
    for wav in wavs:
        s, b = spectral_score(wav, fret)
        if s is None:
            continue
        ok = s >= THRESHOLD
        correct += ok
        total_tested += 1
        row(ok, f"HARM-f{fret}", wav.name, fret, s, b)

print()

# ── Dead notes (expect score < THRESHOLD, tested against fret 7) ───────
for wav in sorted((BASE / "dead_note").glob("*.wav"))[:8]:
    s, b = spectral_score(wav, 7)
    if s is None:
        continue
    ok = s < THRESHOLD
    correct += ok
    total_tested += 1
    row(ok, "DEAD_NOTE", wav.name, 7, s, b)

print()

# ── General notes at A4 (pitch 69) — same pitch as fret-7, wrong timbre ─
gen_69 = sorted((BASE / "general_note").glob("*pitches69*.wav"))[:4]
for wav in gen_69:
    s, b = spectral_score(wav, 7)
    if s is None:
        continue
    ok = s < THRESHOLD
    correct += ok
    total_tested += 1
    row(ok, "GEN@A4", wav.name, 7, s, b)

print()
# ── General notes at fret-5 pitch (74) ─────────────────────────────────
gen_74 = sorted((BASE / "general_note").glob("*pitches74*.wav"))[:4]
for wav in gen_74:
    s, b = spectral_score(wav, 5)
    if s is None:
        continue
    ok = s < THRESHOLD
    correct += ok
    total_tested += 1
    row(ok, "GEN@E5", wav.name, 5, s, b)

print("=" * 78)
acc = correct / total_tested if total_tested else 0
print(f"  Accuracy: {correct}/{total_tested} = {acc:.1%}  (threshold={THRESHOLD})")
print("  ✓ = correctly classified   ✗ = wrong")
