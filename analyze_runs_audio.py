"""Spectral analysis of all audio in './runs' (successes + audio dumps).

Parses metadata from filenames and paths, scores each clip, and reports
per-run summaries plus a global leaderboard.

Filename formats:
  successes/   000001_20260228_124534_str2_fret6.94_torque56.wav
  audio_dumps/ dump_YYYYMMDD_HHMMSS/001_str2_fret6.94_torque56.wav
"""

import re
import sys
import argparse
import numpy as np
from math import gcd
from pathlib import Path
from collections import defaultdict
from scipy.signal import welch, resample_poly
from scipy.io import wavfile

# ── Spectral constants (mirrors utils/reward.py) ──────────────────────────────
D_STRING_OPEN_FREQ   = 146.83        # D3, MIDI 50
FRET_TO_HARMONIC_N   = {4: 5, 5: 4, 7: 3, 9: 4}
BANDWIDTH            = 0.03          # ±3 % around each partial
NOISE_FLOOR_HZ       = 80.0
THRESHOLD            = 0.65
SR                   = 22050
HOP                  = 512

# Harmonic frets we recognise; anything else gets nearest-neighbour assignment
HARMONIC_FRETS = sorted(FRET_TO_HARMONIC_N.keys())  # [4, 5, 7, 9]


# ── Helpers ───────────────────────────────────────────────────────────────────
def nearest_harmonic_fret(fret_val: float) -> int:
    """Return the closest canonical harmonic fret to fret_val."""
    return min(HARMONIC_FRETS, key=lambda f: abs(f - fret_val))


def load_wav(path: Path) -> np.ndarray:
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


def onset_align(y: np.ndarray) -> np.ndarray:
    """Return audio starting at first frame above 10 % of peak RMS (cap 1 s)."""
    n_frames = len(y) // HOP
    rms = np.array([np.sqrt(np.mean(y[i * HOP:(i + 1) * HOP] ** 2))
                    for i in range(n_frames)])
    if rms.max() == 0:
        return y
    hits = np.where(rms > 0.10 * rms.max())[0]
    onset = hits[0] * HOP if len(hits) else 0
    return y[min(onset, SR):]


def spectral_score(y: np.ndarray, target_fret: int) -> tuple[float, dict]:
    """Return (score, breakdown) for audio y given a canonical harmonic fret."""
    y = onset_align(y)
    skip = int(0.3 * SR)
    y = y[skip: skip + int(2.0 * SR)]
    if len(y) < int(0.25 * SR):
        return float('nan'), {}

    nperseg = min(4096, len(y))
    freqs, psd = welch(y, fs=SR, nperseg=nperseg, noverlap=nperseg // 2)

    harmonic_n  = FRET_TO_HARMONIC_N[target_fret]
    harm_fund   = harmonic_n * D_STRING_OPEN_FREQ
    f0          = D_STRING_OPEN_FREQ

    desired = 0.0
    partial = harm_fund
    while partial < SR / 2:
        lo, hi = partial * (1 - BANDWIDTH), partial * (1 + BANDWIDTH)
        desired += float(psd[(freqs >= lo) & (freqs <= hi)].sum())
        partial += harm_fund

    f0_e  = float(psd[(freqs >= f0 * (1 - BANDWIDTH)) & (freqs <= f0 * (1 + BANDWIDTH))].sum())
    total = float(psd[freqs >= NOISE_FLOOR_HZ].sum()) + 1e-12

    her      = desired / total
    fund_sup = 1.0 - float(np.clip(f0_e / total * 5.0, 0, 1))
    sig      = float(np.clip(total / 1e-4, 0, 1))
    score    = float(np.clip(0.60 * her + 0.25 * fund_sup + 0.15 * sig, 0, 1))

    return score, dict(HER=her, fund_sup=fund_sup, sig=sig, total_e=total)


# ── Filename / path parsers ───────────────────────────────────────────────────
SUCCESS_RE = re.compile(
    r'(?P<seq>\d+)_(?P<ts>\d{8}_\d{6})_str(?P<string>\d+)_fret(?P<fret>[\d.]+)_torque(?P<torque>[\d.]+)\.wav',
    re.IGNORECASE,
)
DUMP_RE = re.compile(
    r'(?P<seq>\d+)_str(?P<string>\d+)_fret(?P<fret>[\d.]+)_torque(?P<torque>[\d.]+)\.wav',
    re.IGNORECASE,
)


def parse_metadata(wav_path: Path, runs_root: Path):
    """Extract run name, clip type, fret, torque, etc. from path + filename."""
    rel = wav_path.relative_to(runs_root)
    parts = rel.parts  # (run_name, ..., filename)

    run_name = parts[0]
    fname    = wav_path.name

    meta = dict(run=run_name, path=wav_path, clip_type=None,
                seq=None, ts=None, string=None,
                fret_raw=None, fret_harmonic=None, torque=None,
                dump_episode=None)

    if 'successes' in parts:
        meta['clip_type'] = 'success'
        m = SUCCESS_RE.match(fname)
        if m:
            meta['seq']    = int(m.group('seq'))
            meta['ts']     = m.group('ts')
            meta['string'] = int(m.group('string'))
            meta['fret_raw']  = float(m.group('fret'))
            meta['torque']    = float(m.group('torque'))
    elif 'audio_dumps' in parts:
        meta['clip_type'] = 'dump'
        # dump episode is the folder name inside audio_dumps
        dump_idx = parts.index('audio_dumps')
        if dump_idx + 1 < len(parts):
            meta['dump_episode'] = parts[dump_idx + 1]
        m = DUMP_RE.match(fname)
        if m:
            meta['seq']    = int(m.group('seq'))
            meta['string'] = int(m.group('string'))
            meta['fret_raw']  = float(m.group('fret'))
            meta['torque']    = float(m.group('torque'))
    else:
        return None  # unknown layout; skip

    if meta['fret_raw'] is not None:
        meta['fret_harmonic'] = nearest_harmonic_fret(meta['fret_raw'])

    return meta


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Spectral analysis of runs audio')
    ap.add_argument('--runs-dir', type=str,
                    default=str(Path(__file__).parent / 'runs'),
                    help='Root directory of runs (default: ./runs)')
    ap.add_argument('--run', type=str, default=None,
                    help='Analyse only runs matching this substring')
    ap.add_argument('--type', choices=['success', 'dump', 'all'], default='all',
                    help='Clip type to include (default: all)')
    ap.add_argument('--min-fret', type=float, default=None,
                    help='Only include clips where fret_raw >= this value')
    ap.add_argument('--max-fret', type=float, default=None,
                    help='Only include clips where fret_raw <= this value')
    ap.add_argument('--verbose', '-v', action='store_true',
                    help='Print every clip (not just summaries)')
    ap.add_argument('--worst', action='store_true',
                    help='Report the lowest-scoring clips instead of the highest')
    ap.add_argument('--threshold', type=float, default=THRESHOLD,
                    help=f'Spectral pass threshold (default: {THRESHOLD})')
    args = ap.parse_args()

    runs_root = Path(args.runs_dir)
    if not runs_root.is_dir():
        print(f"ERROR: directory not found: {runs_root}")
        sys.exit(1)

    threshold = args.threshold

    # ── Collect all WAV files ─────────────────────────────────────────────────
    all_wavs = sorted(runs_root.rglob('*.wav'))
    print(f"Found {len(all_wavs)} WAV files under {runs_root.name}/\n")

    # ── Score each clip ───────────────────────────────────────────────────────
    results_by_run = defaultdict(list)
    skipped = 0

    for wav in all_wavs:
        meta = parse_metadata(wav, runs_root)
        if meta is None:
            skipped += 1
            continue
        if args.run and args.run.lower() not in meta['run'].lower():
            continue
        if args.type != 'all' and meta['clip_type'] != args.type:
            continue
        if args.min_fret is not None and (meta['fret_raw'] is None or meta['fret_raw'] < args.min_fret):
            continue
        if args.max_fret is not None and (meta['fret_raw'] is None or meta['fret_raw'] > args.max_fret):
            continue

        try:
            y = load_wav(wav)
        except Exception as e:
            print(f"  [LOAD ERROR] {wav.name}: {e}")
            skipped += 1
            continue

        target_fret = meta['fret_harmonic'] if meta['fret_harmonic'] else 7
        score, breakdown = spectral_score(y, target_fret)

        meta['score']     = score
        meta['breakdown'] = breakdown
        meta['pass']      = (not np.isnan(score)) and score >= threshold

        results_by_run[meta['run']].append(meta)

    # ── Per-run summary ───────────────────────────────────────────────────────
    COL = 78
    print('=' * COL)
    print(f"  {'RUN':<35} {'TYPE':<8} {'N':>4}  {'PASS':>5}  {'PASS%':>6}  {'AVG_SCORE':>9}  {'AVG_HER':>7}")
    print('=' * COL)

    global_rows = []

    for run_name in sorted(results_by_run.keys()):
        rows = results_by_run[run_name]

        for clip_type in ('success', 'dump'):
            sub = [r for r in rows if r['clip_type'] == clip_type]
            if not sub:
                continue
            valid = [r for r in sub if not np.isnan(r['score'])]
            n_pass = sum(1 for r in valid if r['pass'])
            avg_score = np.mean([r['score'] for r in valid]) if valid else float('nan')
            avg_her   = np.mean([r['breakdown'].get('HER', float('nan')) for r in valid if r['breakdown']]) if valid else float('nan')
            pct       = 100 * n_pass / len(valid) if valid else 0.0

            rname_display = run_name[:35]
            print(f"  {rname_display:<35} {clip_type:<8} {len(sub):>4}  {n_pass:>5}  {pct:>5.1f}%  {avg_score:>9.3f}  {avg_her:>7.3f}")

            if args.verbose:
                for r in sorted(sub, key=lambda x: (x['seq'] or 0)):
                    ok   = '✓' if r['pass'] else '✗'
                    s    = r['score']
                    bd   = r['breakdown']
                    ep   = r.get('dump_episode') or ''
                    ep   = f" ep={ep[-6:]}" if ep else ''
                    her  = bd.get('HER', float('nan'))
                    fsup = bd.get('fund_sup', float('nan'))
                    fret = r['fret_raw']
                    torq = r['torque']
                    rel = r['path'].relative_to(Path.cwd()) if r['path'].is_relative_to(Path.cwd()) else r['path']
                    print(f"      {ok} seq={r['seq']:>4}{ep}  f{r['fret_harmonic']}({fret:.2f})  "
                          f"t={torq:>5.0f}  score={s:.3f}  HER={her:.3f}  Fsup={fsup:.3f}  {rel}")
                print()

            global_rows.extend(valid)

    # ── Global summary ────────────────────────────────────────────────────────
    print('=' * COL)
    n_total = len(global_rows)
    n_pass  = sum(1 for r in global_rows if r['pass'])
    avg_s   = np.mean([r['score'] for r in global_rows]) if global_rows else float('nan')
    avg_h   = np.mean([r['breakdown'].get('HER', float('nan')) for r in global_rows if r['breakdown']]) if global_rows else float('nan')

    print(f"\n  GLOBAL  {n_total} clips  |  {n_pass} pass ({100*n_pass/n_total:.1f}% of valid)  |  "
          f"avg score={avg_s:.3f}  avg HER={avg_h:.3f}")
    print(f"  threshold={threshold}  |  skipped={skipped} (unrecognised layout)")

    # ── Per-fret breakdown ────────────────────────────────────────────────────
    print(f"\n  {'FRET':<6} {'N':>4}  {'PASS':>5}  {'PASS%':>6}  {'AVG_SCORE':>9}  {'AVG_HER':>7}")
    print('  ' + '-' * 44)
    for fret in sorted(FRET_TO_HARMONIC_N.keys()):
        sub = [r for r in global_rows if r.get('fret_harmonic') == fret]
        if not sub:
            continue
        n_p   = sum(1 for r in sub if r['pass'])
        a_s   = np.mean([r['score'] for r in sub])
        a_h   = np.mean([r['breakdown'].get('HER', float('nan')) for r in sub if r['breakdown']])
        print(f"  fret {fret:<2} {len(sub):>4}  {n_p:>5}  {100*n_p/len(sub):>5.1f}%  {a_s:>9.3f}  {a_h:>7.3f}")

    # ── Top 10 best / worst spectral scores (success clips only) ────────────
    successes_only = [r for r in global_rows if r['clip_type'] == 'success']
    if successes_only:
        if args.worst:
            ranked = sorted(successes_only, key=lambda r: r['score'])[:10]
            print(f"\n  BOTTOM 10 SUCCESS CLIPS BY SPECTRAL SCORE")
        else:
            ranked = sorted(successes_only, key=lambda r: r['score'], reverse=True)[:10]
            print(f"\n  TOP 10 SUCCESS CLIPS BY SPECTRAL SCORE")
        print('  ' + '-' * 72)
        for r in ranked:
            bd  = r['breakdown']
            rel = r['path'].relative_to(Path.cwd()) if r['path'].is_relative_to(Path.cwd()) else r['path']
            print(f"    score={r['score']:.3f}  HER={bd.get('HER',0):.3f}  "
                  f"Fsup={bd.get('fund_sup',0):.3f}  "
                  f"f{r['fret_harmonic']}({r['fret_raw']:.2f})  "
                  f"t={r['torque']:>5.0f}  "
                  f"{rel}")

    print()


if __name__ == '__main__':
    main()
