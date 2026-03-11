"""Three-way correlation: CNN classifier  ×  cosine-similarity  ×  spectral score.

CNN harmonic_prob is read directly from the JSON sidecars saved alongside each
success WAV — no model re-inference needed.  Cosine-sim and spectral scores are
computed fresh from the audio.

Usage:
    python correlate_scores.py
    python correlate_scores.py --runs-dir "runs copy" --save-csv out.csv
"""

import argparse
import json
import re
import sys
import numpy as np
import librosa
from pathlib import Path
from math import gcd
from scipy.signal import welch, resample_poly
from scipy.io import wavfile
from scipy.stats import pearsonr, spearmanr

# ── Constants ─────────────────────────────────────────────────────────────────
SR          = 22050
N_FFT       = 4096
HOP         = 512
N_MELS      = 128
FMIN        = 80
FMAX        = 10000
MAX_ONSET   = 1.0

D_STRING_OPEN_FREQ = 146.83
FRET_TO_HARMONIC_N = {4: 5, 5: 4, 7: 3, 9: 4}
BANDWIDTH          = 0.03
NOISE_FLOOR_HZ     = 80.0

CNN_THRESHOLD  = 0.80
COS_THRESHOLD  = 0.80
SPEC_THRESHOLD = 0.65

HARMONIC_FRETS = sorted(FRET_TO_HARMONIC_N.keys())
REF_BASE       = Path(__file__).parent.parent / "HarmonicsClassifier" / "note_clips" / "harmonic"
FRET_TO_PITCH  = {4: 78, 5: 74, 7: 69}

SUCCESS_RE = re.compile(
    r'\d+_\d{8}_\d{6}_str\d+_fret(?P<fret>[\d.]+)_torque(?P<torque>[\d.]+)\.wav',
    re.IGNORECASE,
)


def nearest_harmonic_fret(f):
    return min(HARMONIC_FRETS, key=lambda x: abs(x - f))


# ── Audio I/O ─────────────────────────────────────────────────────────────────
def load_wav(path):
    file_sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if file_sr != SR:
        g = gcd(file_sr, SR)
        data = resample_poly(data, SR // g, file_sr // g).astype(np.float32)
    return data


def onset_align(y):
    frames = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP)
    if len(frames) > 0:
        c = int(librosa.frames_to_samples(frames[0], hop_length=HOP))
        if c <= int(MAX_ONSET * SR):
            return y[c:]
    return y


# ── Mel helpers ───────────────────────────────────────────────────────────────
def compute_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max)


def normalize_01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


# ── Reference mel cache ───────────────────────────────────────────────────────
def load_ref_mels(ref_base, fret_to_pitch):
    ref_mels = {}
    for fret, pitch in fret_to_pitch.items():
        wavs = sorted(ref_base.glob(f"GB_NH*pitches{pitch}*.wav"))
        mels = []
        for w in wavs:
            try:
                y = load_wav(w)
                y = onset_align(y)
                if len(y) >= int(0.5 * SR):
                    mels.append(normalize_01(compute_mel(y)))
            except Exception as e:
                print(f"  [ref skip] {w.name}: {e}")
        ref_mels[fret] = mels
        print(f"  Loaded {len(mels)} reference mels for fret {fret} (pitch {pitch})")
    return ref_mels


# ── Score functions ───────────────────────────────────────────────────────────
def cosine_score(y, target_fret, ref_mels):
    refs = ref_mels.get(target_fret)
    if not refs:
        return float('nan')
    y = onset_align(y)
    if len(y) < int(0.5 * SR):
        return 0.0
    mel = normalize_01(compute_mel(y))
    best = 0.0
    for ref in refs:
        min_t = min(mel.shape[1], ref.shape[1])
        v1 = mel[:, :min_t].ravel()
        v2 = ref[:, :min_t].ravel()
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        best = max(best, sim)
    return float(np.clip(best, 0.0, 1.0))


def spectral_score(y, target_fret):
    y = onset_align(y)
    skip = int(0.3 * SR)
    y = y[skip: skip + int(2.0 * SR)]
    if len(y) < int(0.25 * SR):
        return float('nan'), float('nan')

    nperseg = min(4096, len(y))
    freqs, psd = welch(y, fs=SR, nperseg=nperseg, noverlap=nperseg // 2)

    harm_fund = FRET_TO_HARMONIC_N[target_fret] * D_STRING_OPEN_FREQ
    f0        = D_STRING_OPEN_FREQ

    desired = 0.0
    partial = harm_fund
    while partial < SR / 2:
        lo, hi = partial * (1 - BANDWIDTH), partial * (1 + BANDWIDTH)
        desired += float(psd[(freqs >= lo) & (freqs <= hi)].sum())
        partial += harm_fund

    f0_e  = float(psd[(freqs >= f0*(1-BANDWIDTH)) & (freqs <= f0*(1+BANDWIDTH))].sum())
    total = float(psd[freqs >= NOISE_FLOOR_HZ].sum()) + 1e-12

    her      = desired / total
    fund_sup = 1.0 - float(np.clip(f0_e / total * 5.0, 0, 1))
    sig      = float(np.clip(total / 1e-4, 0, 1))
    score    = float(np.clip(0.60*her + 0.25*fund_sup + 0.15*sig, 0, 1))
    return score, her


# ── Correlation helpers ───────────────────────────────────────────────────────
def corr_row(label_a, label_b, a, b):
    pr, pp = pearsonr(a, b)
    sr, sp = spearmanr(a, b)
    return f"  {label_a:<10} × {label_b:<10}  Pearson r={pr:+.4f} (p={pp:.1e})  Spearman r={sr:+.4f} (p={sp:.1e})"


def ascii_scatter(x_arr, y_arr, x_label, y_label, w=40, h=20):
    grid = np.zeros((h, w), dtype=int)
    for x, y in zip(x_arr, y_arr):
        xi = min(int(x * w), w - 1)
        yi = min(int(y * h), h - 1)
        grid[yi, xi] += 1
    print(f"\n  SCATTER  x={x_label}  y={y_label}")
    print('  ' + '─' * (w + 2))
    for row in reversed(range(h)):
        cells = ''.join(
            ' ' if grid[row, col] == 0 else
            '·' if grid[row, col] == 1 else
            str(min(grid[row, col], 9)) if grid[row, col] < 10 else '#'
            for col in range(w)
        )
        print(f"  |{cells}| {(row+1)/h:.2f}")
    print('  ' + '─' * (w + 2))
    print('   0' + ' ' * (w // 2 - 3) + '0.5' + ' ' * (w // 2 - 3) + '1.0')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default='runs copy')
    ap.add_argument('--ref-dir', default=str(REF_BASE))
    ap.add_argument('--top-n', type=int, default=12)
    ap.add_argument('--save-csv', default=None)
    args = ap.parse_args()

    runs_root = Path(args.runs_dir)
    ref_base  = Path(args.ref_dir)
    if not runs_root.is_dir(): print(f"ERROR: {runs_root} not found"); sys.exit(1)
    if not ref_base.is_dir():  print(f"ERROR: {ref_base} not found");  sys.exit(1)

    print(f"\nLoading reference mels ...")
    ref_mels = load_ref_mels(ref_base, FRET_TO_PITCH)

    # Only success clips — they have JSON sidecars with CNN scores
    all_wavs = [w for w in sorted(runs_root.rglob('*.wav'))
                if 'successes' in w.parts]
    print(f"\nScoring {len(all_wavs)} success clips ...\n")

    records = []
    no_json = 0
    for i, wav in enumerate(all_wavs, 1):
        m = SUCCESS_RE.match(wav.name)
        if not m:
            continue
        fret_raw = float(m.group('fret'))
        torque   = float(m.group('torque'))
        target   = nearest_harmonic_fret(fret_raw)
        if target not in ref_mels:
            continue

        # Load CNN score from JSON sidecar
        json_path = wav.with_suffix('.json')
        if not json_path.exists():
            no_json += 1
            continue
        with open(json_path) as f:
            meta = json.load(f)
        cnn_prob = float(meta.get('harmonic_prob', meta.get('confidence', 0.0)))

        try:
            y = load_wav(wav)
        except Exception as e:
            print(f"  [load err] {wav.name}: {e}")
            continue

        cos          = cosine_score(y, target, ref_mels)
        spec, her    = spectral_score(y, target)

        if any(np.isnan(v) for v in (cos, spec)):
            continue

        run = wav.parts[-3]  # runs copy / <run_name> / successes / file.wav
        records.append(dict(
            run=run,
            file=wav.name,
            fret_raw=fret_raw,
            fret=target,
            torque=torque,
            cnn=cnn_prob,
            cosine=cos,
            spectral=spec,
            her=her,
            predicted_label=meta.get('predicted_label', '?'),
        ))

        if i % 50 == 0:
            print(f"  {i}/{len(all_wavs)} ...", flush=True)

    if not records:
        print("No scoreable clips found."); return

    cnn_arr  = np.array([r['cnn']      for r in records])
    cos_arr  = np.array([r['cosine']   for r in records])
    spec_arr = np.array([r['spectral'] for r in records])
    her_arr  = np.array([r['her']      for r in records])

    W = 82
    print()
    print('=' * W)
    print(f"  THREE-WAY CORRELATION   (n={len(records)}  |  {no_json} clips had no JSON, skipped)")
    print('=' * W)
    print(corr_row('CNN',    'cosine',   cnn_arr,  cos_arr))
    print(corr_row('CNN',    'spectral', cnn_arr,  spec_arr))
    print(corr_row('cosine', 'spectral', cos_arr,  spec_arr))

    # Pass rates per metric
    cnn_pass  = cnn_arr  >= CNN_THRESHOLD
    cos_pass  = cos_arr  >= COS_THRESHOLD
    spec_pass = spec_arr >= SPEC_THRESHOLD
    print()
    print(f"  {'METRIC':<12}  {'mean':>6}  {'std':>5}  {'pass%':>6}  threshold")
    print('  ' + '-' * 40)
    print(f"  {'CNN':<12}  {cnn_arr.mean():>6.3f}  {cnn_arr.std():>5.3f}  "
          f"{100*cnn_pass.mean():>5.1f}%  ≥{CNN_THRESHOLD}")
    print(f"  {'cosine_sim':<12}  {cos_arr.mean():>6.3f}  {cos_arr.std():>5.3f}  "
          f"{100*cos_pass.mean():>5.1f}%  ≥{COS_THRESHOLD}")
    print(f"  {'spectral':<12}  {spec_arr.mean():>6.3f}  {spec_arr.std():>5.3f}  "
          f"{100*spec_pass.mean():>5.1f}%  ≥{SPEC_THRESHOLD}")

    # Agreement matrix
    print()
    print(f"  AGREEMENT (both pass OR both fail)")
    print(f"  CNN × cosine:   {100*np.mean(cnn_pass == cos_pass):.1f}%")
    print(f"  CNN × spectral: {100*np.mean(cnn_pass == spec_pass):.1f}%")
    print(f"  cos × spectral: {100*np.mean(cos_pass == spec_pass):.1f}%")
    print(f"  All three agree:{100*np.mean(cnn_pass & cos_pass & spec_pass | ~cnn_pass & ~cos_pass & ~spec_pass):.1f}%")

    # ── Per-fret breakdown ────────────────────────────────────────────────────
    print()
    print(f"  {'FRET':<6} {'N':>4}  "
          f"{'CNN×cos r':>10}  {'CNN×spec r':>11}  {'cos×spec r':>11}  "
          f"{'CNN%':>5}  {'cos%':>5}  {'spec%':>6}")
    print('  ' + '-' * 68)
    for fret in sorted(FRET_TO_HARMONIC_N.keys()):
        sub = [r for r in records if r['fret'] == fret]
        if len(sub) < 5:
            continue
        c  = np.array([r['cnn']      for r in sub])
        co = np.array([r['cosine']   for r in sub])
        s  = np.array([r['spectral'] for r in sub])
        r1, _ = pearsonr(c,  co)
        r2, _ = pearsonr(c,  s)
        r3, _ = pearsonr(co, s)
        print(f"  fret {fret:<2} {len(sub):>4}  "
              f"{r1:>+10.4f}  {r2:>+11.4f}  {r3:>+11.4f}  "
              f"{100*np.mean(c>=CNN_THRESHOLD):>4.0f}%  "
              f"{100*np.mean(co>=COS_THRESHOLD):>4.0f}%  "
              f"{100*np.mean(s>=SPEC_THRESHOLD):>5.0f}%")

    # ── Biggest CNN↑ / spectral↓ false positives ─────────────────────────────
    diff_cnn_spec = cnn_arr - spec_arr
    idx = np.argsort(diff_cnn_spec)[::-1][:args.top_n]
    print(f"\n  TOP {args.top_n} CNN-HIGH / SPECTRAL-LOW  (CNN says harmonic, spectral disagrees)")
    print('  ' + '-' * (W - 2))
    print(f"  {'CNN':>6}  {'cos':>6}  {'spec':>6}  {'HER':>6}  {'fret':>5}  {'torque':>6}  run / file")
    for i in idx:
        r = records[i]
        print(f"  {r['cnn']:>6.3f}  {r['cosine']:>6.3f}  {r['spectral']:>6.3f}  "
              f"{r['her']:>6.3f}  f{r['fret']}({r['fret_raw']:.1f})  "
              f"{r['torque']:>6.0f}  {r['run'][:20]}  {r['file'][-30:]}")

    # ── Biggest spectral↑ / CNN↓  ─────────────────────────────────────────────
    diff_spec_cnn = spec_arr - cnn_arr
    idx2 = np.argsort(diff_spec_cnn)[::-1][:args.top_n // 2]
    print(f"\n  TOP {args.top_n//2} SPECTRAL-HIGH / CNN-LOW  (spectral says harmonic, CNN disagrees)")
    print('  ' + '-' * (W - 2))
    for i in idx2:
        r = records[i]
        print(f"  {r['cnn']:>6.3f}  {r['cosine']:>6.3f}  {r['spectral']:>6.3f}  "
              f"{r['her']:>6.3f}  f{r['fret']}({r['fret_raw']:.1f})  "
              f"{r['torque']:>6.0f}  {r['run'][:20]}  {r['file'][-30:]}")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    ascii_scatter(cnn_arr,  spec_arr, 'CNN',    'spectral')
    ascii_scatter(cos_arr,  spec_arr, 'cosine', 'spectral')
    ascii_scatter(cnn_arr,  cos_arr,  'CNN',    'cosine')

    # ── Optional CSV ─────────────────────────────────────────────────────────
    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\n  Saved {len(records)} rows → {args.save_csv}")


if __name__ == '__main__':
    main()
