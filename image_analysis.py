"""
image_analysis.py — Spectrogram comparison of RL agent audio vs dataset reference.

Compare onset-aligned mel spectrograms side-by-side with quantitative similarity
metrics (cosine similarity, MSE, SSIM, Pearson r).

Usage:
    # Single pair
    conda activate guitaRL
    python image_analysis.py
    python image_analysis.py --rl <rl.wav> --ref <ref.wav> --output comparison.png
    python image_analysis.py --play-audio

    # Batch: all RL successes vs all GB_NH pitch-69 references
    python image_analysis.py --batch
    python image_analysis.py --batch --top-n 3 --rank-by ssim --play-audio
    python image_analysis.py --batch --rl-dir runs/my_run --top-n 5
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

# ── Parameters matching utils/audio_reward.py and HarmonicsClassifier/train_cnn.py ──
SR = 22050
N_FFT = 4096
HOP_LENGTH = 512
N_MELS = 128
FMIN = 80
FMAX = 10000

_THIS_DIR = Path(__file__).parent
_DEFAULT_RL = _THIS_DIR / "runs/good_run_final_audio/successes/000025_20260225_151318_str2_fret7.14_torque287.wav"
_DEFAULT_REF = _THIS_DIR / "../HarmonicsClassifier/note_clips/harmonic/GB_NH_harmonic_n2_pitches69_rep07.wav"
_DEFAULT_RL_DIR      = _THIS_DIR / "runs/good_run_final_audio"
_DEFAULT_REF_DIR     = _THIS_DIR / "../HarmonicsClassifier/note_clips/harmonic"
_DEFAULT_NEG_CTRL_DIR = _THIS_DIR / "../HarmonicsClassifier/note_clips/dead_note"

# Metrics where higher = better (all except mse)
_HIGHER_IS_BETTER = {"cosine_sim", "ssim", "pearson_r"}

# Onset alignment guards
MAX_ONSET_SEC = 1    # ignore any detected onset later than this (fall back to 0)
MIN_ALIGNED_SEC = 0.5  # skip pairs where the cropped duration is shorter than this


# ── Core audio / feature functions ───────────────────────────────────────────

def load_and_align(path: Union[str, Path], sr: int = SR) -> np.ndarray:
    """Load audio and trim to the first detected onset.

    If no onset is found within the first MAX_ONSET_SEC of the clip the raw
    audio is returned unchanged, preventing onset detection from latching onto
    a late resonance event and leaving a uselessly short tail.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_sample = 0
    if len(onset_frames) > 0:
        candidate = int(librosa.frames_to_samples(onset_frames[0], hop_length=HOP_LENGTH))
        if candidate <= int(MAX_ONSET_SEC * sr):
            onset_sample = candidate
        # else: onset found too late in the clip — likely a resonance artefact; ignore it
    return y[onset_sample:]


def compute_mel_db(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute mel spectrogram in dB using project-standard parameters."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max)


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _peak_normalize(y: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.max(np.abs(y))
    return y / (peak + 1e-8)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_similarity_metrics(mel1: np.ndarray, mel2: np.ndarray) -> Dict[str, float]:
    """Compute similarity metrics between two same-shape mel spectrograms.

    Inputs are normalized to [0, 1] so dB scale offsets don't dominate.

    Returns:
        cosine_sim  — cosine similarity of flattened vectors  (1.0 = identical)
        mse         — mean squared error on normalized arrays  (0.0 = identical)
        ssim        — structural similarity index              (1.0 = identical)
        pearson_r   — Pearson correlation coefficient          (1.0 = identical)
    """
    n1_2d = _normalize_01(mel1)
    n2_2d = _normalize_01(mel2)
    n1 = n1_2d.ravel()
    n2 = n2_2d.ravel()

    return {
        "cosine_sim": float(cosine_similarity(n1.reshape(1, -1), n2.reshape(1, -1))[0, 0]),
        "mse":        float(mean_squared_error(n1, n2)),
        "ssim":       float(ssim(n1_2d, n2_2d, data_range=1.0)),
        "pearson_r":  float(np.corrcoef(n1, n2)[0, 1]),
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\n── Similarity Metrics ─────────────────────────────")
    print(f"  Cosine similarity : {metrics['cosine_sim']:.4f}  (1.0 = identical)")
    print(f"  MSE (normalised)  : {metrics['mse']:.4f}  (0.0 = identical)")
    print(f"  SSIM              : {metrics['ssim']:.4f}  (1.0 = identical)")
    print(f"  Pearson r         : {metrics['pearson_r']:.4f}  (1.0 = identical)")
    print("────────────────────────────────────────────────────\n")


# ── Audio playback ────────────────────────────────────────────────────────────

def play_audio_pair(
    y_rl: np.ndarray,
    y_ref: np.ndarray,
    label_rl: str,
    label_ref: str,
    sr: int = SR,
) -> None:
    """Play RL audio then reference audio, peak-normalized, with console labels."""
    for y, label in [(y_rl, label_rl), (y_ref, label_ref)]:
        print(f"  ▶ Playing: {label}")
        sd.play(_peak_normalize(y), samplerate=sr)
        sd.wait()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(
    mel1: np.ndarray,
    mel2: np.ndarray,
    label1: str,
    label2: str,
    duration_sec: float,
    metrics: Optional[Dict[str, float]] = None,
    rank: Optional[int] = None,
    sr: int = SR,
    output: Optional[Union[str, Path]] = None,
    y1: Optional[np.ndarray] = None,
    y2: Optional[np.ndarray] = None,
    play_audio: bool = False,
) -> None:
    """Plot two mel spectrograms side by side with optional metrics and playback."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Human-readable Hz ticks that fall within [FMIN, FMAX]
    _hz_ticks = [100, 200, 500, 1000, 2000, 5000, 10000]
    hz_ticks = [t for t in _hz_ticks if FMIN <= t <= FMAX]

    for ax, mel, label in zip(axes, [mel1, mel2], [label1, label2]):
        img = librosa.display.specshow(
            mel, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel",
            fmin=FMIN, fmax=FMAX,
            ax=ax, cmap="magma",
        )
        ax.set_yticks(hz_ticks)
        ax.set_yticklabels([f"{t:,}" for t in hz_ticks])
        ax.set_ylabel("Hz")
        ax.set_title(label, fontsize=9, wrap=True)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    rank_str = f"  |  Rank #{rank}" if rank is not None else ""
    plt.suptitle(
        f"Mel Spectrogram Comparison  |  aligned: {duration_sec:.3f}s{rank_str}",
        fontsize=12, fontweight="bold",
    )

    if metrics is not None:
        # Primary stats for the paper: cosine similarity and MSE
        primary_text = (
            f"Cosine Similarity = {metrics['cosine_sim']:.4f}"
            f"        MSE = {metrics['mse']:.4f}"
        )
        fig.text(
            0.5, 0.055, primary_text,
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#888888", linewidth=0.8),
        )
        # Secondary stats: SSIM and Pearson r
        secondary_text = (
            f"SSIM = {metrics['ssim']:.4f}"
            f"    Pearson r = {metrics['pearson_r']:.4f}"
        )
        fig.text(
            0.5, 0.01, secondary_text,
            ha="center", va="bottom", fontsize=9, color="#555555",
        )

    plt.tight_layout(rect=[0, 0.12, 1, 1] if metrics else [0, 0, 1, 1])

    if output:
        plt.savefig(str(output), dpi=150, bbox_inches="tight")
        print(f"  Saved plot: {output}")

    if play_audio and y1 is not None and y2 is not None:
        play_audio_pair(y1, y2, label1, label2, sr=sr)

    plt.show()


# ── Pair-level helpers ────────────────────────────────────────────────────────

def _load_pair(
    rl_path: Path, ref_path: Path
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Load, align, and crop a pair.

    Returns (mel_rl, mel_ref, duration_sec, y_rl, y_ref).
    Raises ValueError if the aligned duration is shorter than MIN_ALIGNED_SEC.
    """
    y_rl = load_and_align(rl_path)
    y_ref = load_and_align(ref_path)
    min_len = min(len(y_rl), len(y_ref))
    duration_sec = min_len / SR
    if duration_sec < MIN_ALIGNED_SEC:
        raise ValueError(
            f"aligned duration {duration_sec:.3f}s < MIN_ALIGNED_SEC ({MIN_ALIGNED_SEC}s)"
        )
    y_rl_crop = y_rl[:min_len]
    y_ref_crop = y_ref[:min_len]
    return compute_mel_db(y_rl_crop), compute_mel_db(y_ref_crop), duration_sec, y_rl_crop, y_ref_crop


def compute_baseline(ref_wavs: List[Path]) -> List[Dict[str, float]]:
    """Compute all pairwise ref-vs-ref similarities to establish a same-class ceiling.

    Uses every unique ordered pair (i < j) of reference files — i.e. two independent
    recordings of the same pitch/technique.  Returns a list of metric dicts in the
    same format as compute_similarity_metrics().
    """
    pairs = [(ref_wavs[i], ref_wavs[j])
             for i in range(len(ref_wavs))
             for j in range(i + 1, len(ref_wavs))]

    print(f"Baseline: computing {len(pairs)} ref-vs-ref pairs...")
    baseline = []
    for a, b in pairs:
        try:
            mel_a, mel_b, _, _, _ = _load_pair(a, b)
            baseline.append(compute_similarity_metrics(mel_a, mel_b))
        except Exception as exc:
            print(f"  [baseline skip] {a.name} vs {b.name}: {exc}")
    print(f"  Done ({len(baseline)} valid pairs)\n")
    return baseline


def compute_neg_ctrl(
    rl_wavs: List[Path],
    neg_ctrl_dir: Union[str, Path],
) -> List[Dict]:
    """Compare every RL WAV against all WAVs in neg_ctrl_dir.

    Returns a list of result dicts identical in structure to the main batch
    results (keys: rl, ref, metrics) so the same aggregation helpers work.
    Only metrics are stored — audio arrays are discarded to save memory.
    """
    neg_ctrl_wavs = sorted(Path(neg_ctrl_dir).glob("*.wav"))
    if not neg_ctrl_wavs:
        print(f"  [neg ctrl] No WAVs found in {neg_ctrl_dir}")
        return []

    n_pairs = len(rl_wavs) * len(neg_ctrl_wavs)
    print(f"Neg ctrl: {len(rl_wavs)} RL × {len(neg_ctrl_wavs)} dead-note files "
          f"= {n_pairs} pairs...")
    results = []
    for i, rl_wav in enumerate(rl_wavs):
        for nc_wav in neg_ctrl_wavs:
            try:
                mel_rl, mel_nc, _, _, _ = _load_pair(rl_wav, nc_wav)
                results.append({
                    "rl":      rl_wav,
                    "ref":     nc_wav,
                    "metrics": compute_similarity_metrics(mel_rl, mel_nc),
                })
            except Exception:
                pass  # silently skip short/broken clips
        print(f"  {(i+1)*len(neg_ctrl_wavs)}/{n_pairs} pairs...", end="\r")
    print(f"  Done ({len(results)} valid pairs)\n")
    return results


def print_neg_ctrl_report(
    harmonic_results: List[Dict],
    neg_ctrl_results: List[Dict],
) -> None:
    """Paired statistical comparison: RL-vs-harmonic vs RL-vs-neg-ctrl.

    Because both sets use the same RL files, a paired t-test is the correct
    choice — it cancels out per-recording recording-condition variance and
    isolates the effect of harmonic vs non-harmonic content.

    A significant positive Cohen's d on cosine similarity (RL is more similar
    to harmonics than to dead notes) is direct evidence of task success.
    """
    from collections import defaultdict
    from scipy import stats

    def _per_file_means(results, metric):
        groups = defaultdict(list)
        for r in results:
            groups[r["rl"]].append(r["metrics"][metric])
        # Return in sorted-key order so both arrays align to the same RL files
        keys = sorted(groups.keys())
        return keys, np.array([np.mean(groups[k]) for k in keys])

    h_keys, h_cos = _per_file_means(harmonic_results,  "cosine_sim")
    n_keys, n_cos = _per_file_means(neg_ctrl_results,   "cosine_sim")
    _,      h_mse = _per_file_means(harmonic_results,  "mse")
    _,      n_mse = _per_file_means(neg_ctrl_results,   "mse")

    # Keep only RL files that appear in both sets (intersection, sorted)
    common = sorted(set(h_keys) & set(n_keys))
    h_idx  = [h_keys.index(k) for k in common]
    n_idx  = [n_keys.index(k) for k in common]
    h_cos, n_cos = h_cos[h_idx], n_cos[n_idx]
    h_mse, n_mse = h_mse[h_idx], n_mse[n_idx]

    # Paired t-test (same RL files on both sides)
    # scipy stubs leave TtestResult elements typed as object; cast explicitly
    _r_cos = stats.ttest_rel(h_cos, n_cos)
    t_cos: float = _r_cos.statistic  # type: ignore[assignment]
    p_cos: float = _r_cos.pvalue     # type: ignore[assignment]
    _r_mse = stats.ttest_rel(h_mse, n_mse)
    t_mse: float = _r_mse.statistic  # type: ignore[assignment]
    p_mse: float = _r_mse.pvalue     # type: ignore[assignment]

    # Effect size on the paired differences
    d_cos = _cohens_d(h_cos, n_cos)
    d_mse = _cohens_d(h_mse, n_mse)

    sep = "═" * 60
    print(f"\n{sep}")
    print("  NEGATIVE CONTROL  (RL-vs-harmonic  vs  RL-vs-dead-note)")
    print(f"  n = {len(common)} matched RL files  |  paired t-test")
    print(sep)
    print(f"  {'':14} {'RL→harmonic':>14} {'RL→dead-note':>14}")
    print(f"  {'Cosine sim':14} {h_cos.mean():>13.4f}   {n_cos.mean():>13.4f}")
    print(f"  {'MSE':14} {h_mse.mean():>13.4f}   {n_mse.mean():>13.4f}")

    print(f"\n  {'Metric':<14} {'paired t':>10} {'p':>10}  {'Cohen d':>9}  effect  interpretation")
    print(f"  {'-'*14} {'-'*10} {'-'*10}  {'-'*9}  ------  --------------")
    for label, t, p, d, higher_better in [
        ("Cosine sim", t_cos, p_cos, d_cos, True),
        ("MSE",        t_mse, p_mse, d_mse, False),
    ]:
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        # Positive result: harmonic > dead-note for cosine sim, harmonic < dead-note for MSE
        favours = (d > 0) == higher_better
        interp = "favours harmonic" if favours else "favours dead-note"
        print(f"  {label:<14} {t:>+10.3f} {p:>9.4f}{sig:<3} {d:>+9.3f}  {_effect_label(d):<6}  {interp}")

    print(f"\n  Significance: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant")
    print(sep + "\n")


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two independent samples."""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / (pooled_std + 1e-12))


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:   return "negligible"
    if ad < 0.5:   return "small"
    if ad < 0.8:   return "medium"
    return "large"


def print_success_report(
    rl_results: List[Dict],
    baseline_metrics: List[Dict[str, float]],
) -> None:
    """Print statistical comparison of RL-vs-ref against the ref-vs-ref baseline.

    Uses per-RL-file means as the independent observations (one value per
    recording) to avoid inflating sample size across the N×M pairs.

    Reports:
        - Descriptive stats for both distributions
        - Welch's two-sample t-test  (parametric)
        - Mann-Whitney U test        (non-parametric; no normality assumption)
        - Cohen's d effect size
    """
    from collections import defaultdict
    from scipy import stats

    # One independent observation per RL file
    groups: Dict[Path, List] = defaultdict(list)
    for r in rl_results:
        groups[r["rl"]].append(r["metrics"])

    rl_cos = np.array([np.mean([m["cosine_sim"] for m in v]) for v in groups.values()])
    rl_mse = np.array([np.mean([m["mse"]        for m in v]) for v in groups.values()])

    b_cos = np.array([m["cosine_sim"] for m in baseline_metrics])
    b_mse = np.array([m["mse"]        for m in baseline_metrics])

    # Welch's t-test (does not assume equal variance)
    t_cos, p_t_cos = stats.ttest_ind(rl_cos, b_cos, equal_var=False)
    t_mse, p_t_mse = stats.ttest_ind(rl_mse, b_mse, equal_var=False)

    # Mann-Whitney U (non-parametric, no normality assumption)
    u_cos, p_mw_cos = stats.mannwhitneyu(rl_cos, b_cos, alternative="two-sided")
    u_mse, p_mw_mse = stats.mannwhitneyu(rl_mse, b_mse, alternative="two-sided")

    d_cos = _cohens_d(rl_cos, b_cos)
    d_mse = _cohens_d(rl_mse, b_mse)

    sep = "═" * 60
    print(f"\n{sep}")
    print("  BASELINE  (ref-vs-ref, same pitch)")
    print(sep)
    print(f"  Pairs   : {len(baseline_metrics)}")
    print(f"  Cosine sim  : {b_cos.mean():.4f} ± {b_cos.std(ddof=1):.4f}"
          f"   median {np.median(b_cos):.4f}")
    print(f"  MSE         : {b_mse.mean():.4f} ± {b_mse.std(ddof=1):.4f}"
          f"   median {np.median(b_mse):.4f}")

    print(f"\n  RL vs ref  (n={len(groups)} RL files, per-file means)")
    print(sep)
    print(f"  Cosine sim  : {rl_cos.mean():.4f} ± {rl_cos.std(ddof=1):.4f}"
          f"   median {np.median(rl_cos):.4f}")
    print(f"  MSE         : {rl_mse.mean():.4f} ± {rl_mse.std(ddof=1):.4f}"
          f"   median {np.median(rl_mse):.4f}")

    print(f"\n  STATISTICAL TESTS  (RL per-file means vs ref-vs-ref pairs)")
    print(sep)
    print(f"  {'Metric':<14} {'Welch t':>10} {'p (t)':>10}  "
          f"{'MW U':>10} {'p (MW)':>10}  {'Cohen d':>9}  effect")
    print(f"  {'-'*14} {'-'*10} {'-'*10}  {'-'*10} {'-'*10}  {'-'*9}  ------")
    for label, t, p_t, u, p_mw, d in [
        ("Cosine sim",  t_cos, p_t_cos, u_cos, p_mw_cos, d_cos),
        ("MSE",         t_mse, p_t_mse, u_mse, p_mw_mse, d_mse),
    ]:
        sig_t  = "***" if p_t  < 0.001 else ("**" if p_t  < 0.01 else ("*" if p_t  < 0.05 else "ns"))
        sig_mw = "***" if p_mw < 0.001 else ("**" if p_mw < 0.01 else ("*" if p_mw < 0.05 else "ns"))
        print(f"  {label:<14} {t:>+10.3f} {p_t:>9.4f}{sig_t:<2}"
              f"  {u:>10.1f} {p_mw:>9.4f}{sig_mw:<2}"
              f"  {d:>+9.3f}  {_effect_label(d)}")

    print(f"\n  Significance: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant")
    print(sep + "\n")


def compare_pair(
    rl_path: Union[str, Path],
    ref_path: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    play_audio: bool = False,
) -> Dict[str, float]:
    """Onset-align, compute metrics, and plot a single RL/reference pair."""
    rl_path = Path(rl_path)
    ref_path = Path(ref_path)

    print(f"RL   : {rl_path.name}")
    print(f"Ref  : {ref_path.name}")

    mel_rl, mel_ref, duration_sec, y_rl, y_ref = _load_pair(rl_path, ref_path)
    print(f"Aligned duration: {duration_sec:.3f}s")

    metrics = compute_similarity_metrics(mel_rl, mel_ref)
    print_metrics(metrics)

    plot_comparison(
        mel_rl, mel_ref,
        label1=f"RL agent: {rl_path.stem}",
        label2=f"Reference: {ref_path.stem}",
        duration_sec=duration_sec,
        metrics=metrics,
        output=output,
        y1=y_rl, y2=y_ref,
        play_audio=play_audio,
    )
    return metrics


# ── Batch comparison ──────────────────────────────────────────────────────────

def batch_compare(
    rl_dir: Union[str, Path],
    ref_dir: Union[str, Path],
    top_n: int = 1,
    rank_by: str = "ssim",
    output_dir: Optional[Union[str, Path]] = None,
    play_audio: bool = False,
    baseline: bool = False,
    neg_ctrl_dir: Optional[Union[str, Path]] = None,
) -> List[Dict]:
    """Compare all RL WAVs in rl_dir against all GB_NH pitch-69 reference WAVs.

    Prints a ranked summary table, then plots (and optionally saves/plays) the top_n pairs.

    Args:
        rl_dir:       Directory searched recursively for *.wav files.
        ref_dir:      Directory searched for GB_NH*pitches69*.wav files.
        top_n:        Number of best pairs to plot.
        rank_by:      Metric used for ranking: cosine_sim | mse | ssim | pearson_r.
        output_dir:   If given, saves each plotted figure as a PNG here.
        play_audio:   If True, play RL then reference audio for each plotted pair.
        baseline:     If True, compute ref-vs-ref similarity ceiling and report stats.
        neg_ctrl_dir: If given, compare RL against dead-note clips as a negative control.

    Returns:
        Sorted list of result dicts (best first).
    """
    rl_dir = Path(rl_dir)
    ref_dir = Path(ref_dir)

    rl_wavs = sorted(rl_dir.rglob("*.wav"))
    ref_wavs = sorted(ref_dir.glob("GB_NH*pitches69*.wav"))

    if not rl_wavs:
        print(f"No WAV files found under {rl_dir}")
        return []
    if not ref_wavs:
        print(f"No GB_NH pitches69 WAVs found under {ref_dir}")
        return []

    print(f"Batch: {len(rl_wavs)} RL files × {len(ref_wavs)} reference files "
          f"= {len(rl_wavs) * len(ref_wavs)} pairs")
    print(f"Ranking by: {rank_by}  |  plotting top {top_n}\n")

    results = []
    n_pairs = len(rl_wavs) * len(ref_wavs)
    for i, rl_wav in enumerate(rl_wavs):
        for ref_wav in ref_wavs:
            try:
                mel_rl, mel_ref, duration_sec, y_rl, y_ref = _load_pair(rl_wav, ref_wav)
                metrics = compute_similarity_metrics(mel_rl, mel_ref)
                results.append({
                    "rl":           rl_wav,
                    "ref":          ref_wav,
                    "mel_rl":       mel_rl,
                    "mel_ref":      mel_ref,
                    "y_rl":         y_rl,
                    "y_ref":        y_ref,
                    "duration_sec": duration_sec,
                    "metrics":      metrics,
                    "score":        metrics[rank_by],
                })
            except Exception as exc:
                print(f"  [skip] {rl_wav.name} vs {ref_wav.name}: {exc}")

        done = (i + 1) * len(ref_wavs)
        print(f"  {done}/{n_pairs} pairs processed...", end="\r")

    print()

    # Sort: higher is better except for mse
    reverse = rank_by in _HIGHER_IS_BETTER
    results.sort(key=lambda r: r["score"], reverse=reverse)

    _print_batch_table(results, rank_by)

    # Plot (and optionally play) top N
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for rank, r in enumerate(results[:top_n], start=1):
        out_path = None
        if output_dir is not None:
            out_path = Path(output_dir) / f"rank{rank:02d}_{r['rl'].stem}_vs_{r['ref'].stem}.png"

        print(f"\n── Plotting rank #{rank} ─────────────────────────────")
        print(f"  RL  : {r['rl'].name}")
        print(f"  Ref : {r['ref'].name}")
        print_metrics(r["metrics"])

        plot_comparison(
            r["mel_rl"], r["mel_ref"],
            label1=f"RL agent: {r['rl'].stem}",
            label2=f"Reference: {r['ref'].stem}",
            duration_sec=r["duration_sec"],
            metrics=r["metrics"],
            rank=rank,
            output=out_path,
            y1=r["y_rl"], y2=r["y_ref"],
            play_audio=play_audio,
        )

    # Optional ref-vs-ref baseline
    baseline_metrics: List[Dict[str, float]] = []
    if baseline:
        baseline_metrics = compute_baseline(ref_wavs)
        print_success_report(results, baseline_metrics)

    # Optional negative control: RL vs dead-note clips
    neg_ctrl_results: List[Dict] = []
    if neg_ctrl_dir is not None:
        neg_ctrl_results = compute_neg_ctrl(rl_wavs, neg_ctrl_dir)
        if neg_ctrl_results:
            print_neg_ctrl_report(results, neg_ctrl_results)

    # Summary plot over the full batch
    plot_batch_summary(
        results,
        baseline_metrics=baseline_metrics or None,
        neg_ctrl_results=neg_ctrl_results or None,
        output=Path(output_dir) / "batch_summary.png" if output_dir else None,
    )

    return results


def _rl_short_label(rl_path: Path) -> str:
    """Extract a compact label from an RL success filename.

    e.g. '000025_20260225_151318_str2_fret7.14_torque287' → '#25 fret7.14 τ287'
    """
    parts = rl_path.stem.split("_")
    idx = str(int(parts[0])) if parts[0].isdigit() else parts[0]
    fret  = next((p for p in parts if p.startswith("fret")),   "")
    torq  = next((p for p in parts if p.startswith("torque")), "")
    return f"#{idx} {fret} τ{torq.replace('torque', '')}"


def plot_batch_summary(
    results: List[Dict],
    baseline_metrics: Optional[List[Dict[str, float]]] = None,
    neg_ctrl_results: Optional[List[Dict]] = None,
    output: Optional[Union[str, Path]] = None,
) -> None:
    """Horizontal error-bar plot of cosine similarity and MSE per RL file.

    For each RL file the mean and std are computed across all reference files
    it was compared against.  Files are sorted by mean cosine similarity
    (best at the top) so the ranking is immediately readable.

    If baseline_metrics is provided (ref-vs-ref pairs), a shaded band and
    median line are overlaid so the RL distribution can be read against the
    same-class ceiling.

    If neg_ctrl_results is provided (RL-vs-dead-note pairs), per-file means
    are overlaid as scatter markers so harmonic vs non-harmonic similarity
    can be read at a glance.
    """
    if not results:
        return

    from collections import defaultdict
    groups: Dict[Path, List[Dict]] = defaultdict(list)
    for r in results:
        groups[r["rl"]].append(r["metrics"])

    rl_paths = list(groups.keys())
    cos_means = np.array([np.mean([m["cosine_sim"] for m in v]) for v in groups.values()])
    cos_stds  = np.array([np.std( [m["cosine_sim"] for m in v]) for v in groups.values()])
    mse_means = np.array([np.mean([m["mse"]        for m in v]) for v in groups.values()])
    mse_stds  = np.array([np.std( [m["mse"]        for m in v]) for v in groups.values()])

    # Sort by cosine similarity descending (best at top of horizontal chart)
    order  = np.argsort(cos_means)          # ascending → bottom-to-top on hbar
    labels = [_rl_short_label(rl_paths[i]) for i in order]
    n_refs = len(next(iter(groups.values())))

    y = np.arange(len(order))
    _, axes = plt.subplots(1, 2, figsize=(14, max(5, len(order) * 0.35 + 1.5)),
                           sharey=True)

    # Pre-compute baseline arrays once if provided
    b_cos = np.array([m["cosine_sim"] for m in baseline_metrics]) if baseline_metrics else None
    b_mse = np.array([m["mse"]        for m in baseline_metrics]) if baseline_metrics else None

    # Pre-compute neg ctrl per-file means aligned to rl_paths
    nc_cos: Optional[np.ndarray] = None
    nc_mse: Optional[np.ndarray] = None
    if neg_ctrl_results:
        nc_groups: Dict[Path, List[Dict]] = defaultdict(list)
        for r in neg_ctrl_results:
            nc_groups[r["rl"]].append(r["metrics"])
        nc_cos = np.array([
            np.mean([m["cosine_sim"] for m in nc_groups[p]]) if p in nc_groups else np.nan
            for p in rl_paths
        ])
        nc_mse = np.array([
            np.mean([m["mse"] for m in nc_groups[p]]) if p in nc_groups else np.nan
            for p in rl_paths
        ])

    # ── Cosine Similarity ────────────────────────────────────────────────────
    ax = axes[0]
    ax.barh(y, cos_means[order], xerr=cos_stds[order],
            capsize=3, color="#4C72B0", alpha=0.85, error_kw={"elinewidth": 1.2})
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_title("Cosine Similarity\n(mean ± std across refs)", fontsize=11)
    ax.set_xlim(left=max(0, cos_means.min() - 3 * cos_stds.max()))
    ax.axvline(cos_means.mean(), color="black", linestyle="--", linewidth=0.8,
               label=f"RL batch mean = {cos_means.mean():.4f}")
    if b_cos is not None:
        ax.axvspan(b_cos.mean() - b_cos.std(), b_cos.mean() + b_cos.std(),
                   alpha=0.15, color="#2ca02c", label="ref-vs-ref mean ± std")
        ax.axvline(np.median(b_cos), color="#2ca02c", linestyle=":", linewidth=1.4,
                   label=f"ref-vs-ref median = {np.median(b_cos):.4f}")
    if nc_cos is not None:
        ax.scatter(nc_cos[order], y, marker="D", color="#d62728", s=20, zorder=3,
                   label="neg ctrl (dead-note) mean")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)

    # ── MSE ─────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.barh(y, mse_means[order], xerr=mse_stds[order],
            capsize=3, color="#DD8452", alpha=0.85, error_kw={"elinewidth": 1.2})
    ax.set_xlabel("MSE (normalised)", fontsize=11)
    ax.set_title("MSE\n(mean ± std across refs)", fontsize=11)
    ax.axvline(mse_means.mean(), color="black", linestyle="--", linewidth=0.8,
               label=f"RL batch mean = {mse_means.mean():.4f}")
    if b_mse is not None:
        ax.axvspan(b_mse.mean() - b_mse.std(), b_mse.mean() + b_mse.std(),
                   alpha=0.15, color="#2ca02c", label="ref-vs-ref mean ± std")
        ax.axvline(np.median(b_mse), color="#2ca02c", linestyle=":", linewidth=1.4,
                   label=f"ref-vs-ref median = {np.median(b_mse):.4f}")
    if nc_mse is not None:
        ax.scatter(nc_mse[order], y, marker="D", color="#d62728", s=20, zorder=3,
                   label="neg ctrl (dead-note) mean")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    n_rl = len(rl_paths)
    plt.suptitle(
        f"Batch Similarity Summary  |  {n_rl} RL files × {n_refs} references",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if output:
        plt.savefig(str(output), dpi=150, bbox_inches="tight")
        print(f"  Saved summary plot: {output}")

    plt.show()


def _print_batch_table(results: List[Dict], rank_by: str) -> None:
    """Print a ranked summary table of all batch results."""
    col = 42
    # Mark the active ranking column with *
    labels = {
        "cosine_sim": "cosine",
        "mse":        "mse",
        "ssim":       "ssim",
        "pearson_r":  "pearson",
    }
    cols = {k: f"*{v}*" if k == rank_by else v for k, v in labels.items()}
    header = (
        f"{'Rank':<5}{'RL file':<{col}}{'Reference file':<{col}}"
        f"{cols['cosine_sim']:>8}{cols['mse']:>8}{cols['ssim']:>8}{cols['pearson_r']:>8}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for rank, r in enumerate(results, start=1):
        m = r["metrics"]
        print(
            f"{rank:<5}{r['rl'].name:<{col}}{r['ref'].name:<{col}}"
            f"{m['cosine_sim']:>8.4f}{m['mse']:>8.4f}"
            f"{m['ssim']:>8.4f}{m['pearson_r']:>8.4f}"
        )
    print(sep + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare onset-aligned mel spectrograms of RL audio vs dataset reference."
    )

    # Single-pair mode
    parser.add_argument("--rl", type=str, default=str(_DEFAULT_RL),
                        help="(single mode) Path to RL agent WAV file")
    parser.add_argument("--ref", type=str, default=str(_DEFAULT_REF),
                        help="(single mode) Path to reference dataset WAV file")
    parser.add_argument("--output", type=str, default=None,
                        help="(single mode) Path to save the comparison PNG")

    # Batch mode
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: compare all RL WAVs vs all GB_NH pitch-69 refs")
    parser.add_argument("--rl-dir", type=str, default=str(_DEFAULT_RL_DIR),
                        help="(batch) Directory to search recursively for RL WAVs")
    parser.add_argument("--ref-dir", type=str, default=str(_DEFAULT_REF_DIR),
                        help="(batch) Directory containing GB_NH pitches69 WAVs")
    parser.add_argument("--top-n", type=int, default=1,
                        help="(batch) Number of best pairs to plot (default: 1)")
    parser.add_argument("--rank-by", type=str, default="ssim",
                        choices=["cosine_sim", "mse", "ssim", "pearson_r"],
                        help="(batch) Metric used for ranking (default: ssim)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="(batch) Directory to save plotted PNGs")

    # Shared
    parser.add_argument("--play-audio", action="store_true",
                        help="Play RL audio then reference audio for each plotted pair")
    parser.add_argument("--baseline", action="store_true",
                        help="(batch) Compute ref-vs-ref similarity ceiling and report success rate")
    parser.add_argument("--neg-ctrl-dir", type=str, default=None, metavar="DIR",
                        help="(batch) Directory of dead-note WAVs for negative control comparison "
                             f"(default: {_DEFAULT_NEG_CTRL_DIR})")

    args = parser.parse_args()

    if args.batch:
        batch_compare(
            rl_dir=args.rl_dir,
            ref_dir=args.ref_dir,
            top_n=args.top_n,
            rank_by=args.rank_by,
            output_dir=args.output_dir,
            play_audio=args.play_audio,
            baseline=args.baseline,
            neg_ctrl_dir=args.neg_ctrl_dir,
        )
    else:
        compare_pair(args.rl, args.ref, args.output, play_audio=args.play_audio)


if __name__ == "__main__":
    main()
