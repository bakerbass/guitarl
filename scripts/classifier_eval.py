"""
Evaluate the HarmonicsClassifier against robot-collected audio clips.

Scans a successes/ directory for WAV + JSON sidecar pairs, runs classifier
inference on each clip, and compares the prediction to the human-reviewed or
auto-suggested label.  Reports accuracy, per-class precision/recall/F1, and
a confusion matrix heatmap.

Can also be run on the HarmonicsClassifier's own test set by pointing
--clips at the test audio directory.

Usage:
    # Evaluate against reviewed robot clips
    python scripts/classifier_eval.py \\
        --model ../HarmonicsClassifier/models/best_model.pt \\
        --clips runs/harmonic_sac_TIMESTAMP/successes/ \\
        --reviewed-only

    # Evaluate against all clips (reviewed or not)
    python scripts/classifier_eval.py \\
        --model ../HarmonicsClassifier/models/best_model.pt \\
        --clips runs/harmonic_sac_TIMESTAMP/successes/

    # Run across multiple success directories (e.g. classifier iterations 1-4)
    python scripts/classifier_eval.py \\
        --model ../HarmonicsClassifier/models/best_model_r4.pt \\
        --clips runs/iter1/successes runs/iter2/successes runs/iter3/successes
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Add project root so utils/ is importable
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Add HarmonicsClassifier path
_hc_path = _root.parent / "HarmonicsClassifier"
if str(_hc_path) not in sys.path:
    sys.path.insert(0, str(_hc_path))

from osc_realtime_classifier import HarmonicsCNN  # noqa: E402


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Must match HarmonicsClassifier label_map
CLASS_NAMES = ['harmonic', 'dead_note', 'general_note']
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Mel spectrogram parameters (must match training)
MODEL_SR = 22050
CAPTURE_DURATION = 3.0   # seconds — classifier was trained on 3-second clips
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 80
FMAX = 8000


def load_model(model_path: Path, device: torch.device) -> HarmonicsCNN:
    checkpoint = torch.load(model_path, map_location=device)
    model = HarmonicsCNN(num_classes=3, dropout=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess(audio: np.ndarray, device_sr: int, device: torch.device) -> torch.Tensor:
    """Replicate the preprocessing pipeline used during training."""
    # Resample if needed
    if device_sr != MODEL_SR:
        audio = librosa.resample(audio, orig_sr=device_sr, target_sr=MODEL_SR)

    # Trim silence
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    if len(trimmed) < MODEL_SR * 0.1:
        trimmed = audio

    # Pad or trim to fixed length
    target_len = int(MODEL_SR * CAPTURE_DURATION)
    if len(trimmed) < target_len:
        trimmed = np.pad(trimmed, (0, target_len - len(trimmed)))
    else:
        trimmed = trimmed[:target_len]

    # Mel spectrogram → log dB → normalise
    mel = librosa.feature.melspectrogram(
        y=trimmed, sr=MODEL_SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(device)


def classify(model: HarmonicsCNN, tensor: torch.Tensor) -> dict:
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        'predicted_idx': pred_idx,
        'predicted_label': CLASS_NAMES[pred_idx],
        'harmonic_prob': float(probs[0]),
        'dead_prob': float(probs[1]),
        'general_prob': float(probs[2]),
    }


def load_clip(wav_path: Path, json_path: Path) -> dict | None:
    """Load a WAV clip and its JSON sidecar. Returns None if invalid."""
    try:
        audio, sr = sf.read(str(wav_path), dtype='float32')
        if audio.ndim > 1:
            audio = audio[:, 0]  # take first channel if stereo
    except Exception as e:
        logger.warning(f"Failed to read {wav_path}: {e}")
        return None

    try:
        with open(json_path) as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read {json_path}: {e}")
        return None

    return {'audio': audio, 'sr': sr, 'meta': meta, 'wav_path': wav_path}


def gather_clips(clips_dirs: list, reviewed_only: bool) -> list:
    """Gather all valid (wav, json) pairs from one or more directories."""
    clips = []
    for clips_dir in clips_dirs:
        clips_dir = Path(clips_dir)
        if not clips_dir.exists():
            logger.warning(f"Clips directory not found: {clips_dir}")
            continue
        for wav_path in sorted(clips_dir.glob("*.wav")):
            json_path = wav_path.with_suffix('.json')
            if not json_path.exists():
                continue
            data = load_clip(wav_path, json_path)
            if data is None:
                continue
            if reviewed_only and not data['meta'].get('reviewed', False):
                continue
            clips.append(data)
    return clips


def plot_confusion_matrix(cm: np.ndarray, output_path: Path = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix — Classifier on Robot Audio')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate HarmonicsClassifier on robot-collected audio clips')
    parser.add_argument('--model', required=True,
                        help='Path to HarmonicsClassifier .pt model')
    parser.add_argument('--clips', nargs='+', required=True, metavar='DIR',
                        help='One or more directories containing WAV + JSON clip pairs')
    parser.add_argument('--reviewed-only', action='store_true', default=False,
                        help='Only include clips where reviewed=true in the JSON sidecar')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results JSON + confusion matrix PNG '
                             '(default: first clips directory)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='PyTorch device')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    model = load_model(model_path, device)
    logger.info(f"Loaded classifier from {model_path}")

    # Gather clips
    clips = gather_clips(args.clips, reviewed_only=args.reviewed_only)
    if not clips:
        logger.error("No clips found (check --clips and --reviewed-only flags).")
        sys.exit(1)
    logger.info(f"Loaded {len(clips)} clips")

    # Run inference
    true_labels = []
    pred_labels = []
    details = []

    for clip in clips:
        # Ground-truth label comes from the JSON sidecar
        meta = clip['meta']
        gt_label = meta.get('suggested_label', 'harmonic')
        if gt_label not in LABEL_TO_IDX:
            logger.warning(f"Unknown label '{gt_label}' in {clip['wav_path']} — skipping")
            continue

        tensor = preprocess(clip['audio'], clip['sr'], device)
        result = classify(model, tensor)

        true_labels.append(gt_label)
        pred_labels.append(result['predicted_label'])

        correct = result['predicted_label'] == gt_label
        details.append({
            'wav': str(clip['wav_path']),
            'true_label': gt_label,
            'predicted_label': result['predicted_label'],
            'correct': correct,
            'harmonic_prob': result['harmonic_prob'],
            'dead_prob': result['dead_prob'],
            'general_prob': result['general_prob'],
        })

        if not correct:
            logger.debug(
                f"MISMATCH  true={gt_label:<12} pred={result['predicted_label']:<12}  "
                f"H={result['harmonic_prob']:.3f}  {clip['wav_path'].name}"
            )

    if not true_labels:
        logger.error("No clips could be classified.")
        sys.exit(1)

    # Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(
        true_labels, pred_labels,
        target_names=CLASS_NAMES,
        labels=CLASS_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=CLASS_NAMES)

    print(f"\n{'=' * 60}")
    print(f"CLASSIFIER EVALUATION — {len(true_labels)} clips")
    print(f"Model: {model_path.name}")
    print(f"Reviewed only: {args.reviewed_only}")
    print(f"{'=' * 60}")
    print(f"\nOverall accuracy: {accuracy:.4f} ({accuracy:.1%})")
    print(f"\nPer-class report:\n{report}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    header = "         " + "  ".join(f"{n:>12}" for n in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row = f"{row_name:>12}  " + "  ".join(f"{v:>12}" for v in cm[i])
        print(row)
    print(f"{'=' * 60}\n")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.clips[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    results = {
        'model': str(model_path),
        'n_clips': len(true_labels),
        'reviewed_only': args.reviewed_only,
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'class_names': CLASS_NAMES,
        'details': details,
    }
    results_path = output_dir / "classifier_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Confusion matrix plot
    cm_path = output_dir / "classifier_eval_confusion_matrix.png"
    plot_confusion_matrix(cm, output_path=cm_path)


if __name__ == '__main__':
    main()
