#!/usr/bin/env python3
"""
Export a self-contained augmented dataset zip for HarmonicsClassifier retraining.

Workflow:
  1. Scan runs/*/successes/ and runs/*/audio_dumps/dump_*/ for reviewed RL clips
     (reviewed=true in JSON, suggested_label set by review_successes.py).
  2. Stage a copy of the HarmonicsClassifier note_clips/ directory.
  3. Add the new RL clips into the appropriate label subdirectory.
  4. Write a manifest.json with per-label counts and provenance.
  5. Zip the staged directory.
  6. Delete the unzipped staging directory.

Usage:
    python scripts/export_augmented_dataset.py
    python scripts/export_augmented_dataset.py --runs-dir ./runs --output-dir ~/Desktop
    python scripts/export_augmented_dataset.py --include-unreviewed   # warn but include
    python scripts/export_augmented_dataset.py --no-original          # RL clips only
    python scripts/export_augmented_dataset.py --no-dumps             # skip audio dump clips
"""

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GUITARL_ROOT      = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR  = GUITARL_ROOT / "runs"
DEFAULT_CLASSIF   = GUITARL_ROOT.parent / "HarmonicsClassifier"
VALID_LABELS      = {"harmonic", "dead_note", "general_note"}

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
DIM    = "\033[2m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_wavs(directory: Path) -> int:
    return sum(1 for _ in directory.rglob("*.wav"))


def _collect_rl_clips(
    runs_dir: Path,
    include_unreviewed: bool,
) -> Tuple[List[Tuple[Path, dict, str, str]], int]:
    """Scan runs/*/successes/ for reviewed success clips.

    Returns (clips, skipped_unreviewed_count).
    Each clip is (wav_path, meta, run_label, source_type) where
    source_type is 'success'.
    """
    clips = []
    skipped = 0
    for json_path in sorted(runs_dir.glob("*/successes/*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"{YELLOW}  Warning: could not parse {json_path.name}: {exc}{RESET}")
            continue

        reviewed = meta.get("reviewed", False)
        label    = meta.get("suggested_label", "")

        if not reviewed:
            if include_unreviewed:
                print(f"{YELLOW}  Including unreviewed: {json_path.name}{RESET}")
            else:
                skipped += 1
                continue

        if label not in VALID_LABELS:
            print(f"{YELLOW}  Skipping unknown label '{label}': {json_path.name}{RESET}")
            continue

        wav_name = meta.get("wav_file", json_path.stem + ".wav")
        wav_path = json_path.parent / wav_name
        if not wav_path.exists():
            print(f"{YELLOW}  WAV not found, skipping: {wav_path.name}{RESET}")
            continue

        run_label = json_path.parent.parent.name  # harmonic_sac_TIMESTAMP
        clips.append((wav_path, meta, run_label, "success"))

    return clips, skipped


def _collect_dump_rl_clips(
    runs_dir: Path,
    include_unreviewed: bool,
) -> Tuple[List[Tuple[Path, dict, str, str]], int]:
    """Scan runs/*/audio_dumps/dump_*/ for reviewed audio-dump clips.

    Returns (clips, skipped_unreviewed_count).
    Each clip is (wav_path, meta, run_label, source_type) where
    run_label is '<run_name>_dump_<dump_ts>' and source_type is 'dump'.
    """
    clips = []
    skipped = 0
    for json_path in sorted(runs_dir.glob("*/audio_dumps/dump_*/*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"{YELLOW}  Warning: could not parse {json_path.name}: {exc}{RESET}")
            continue

        reviewed = meta.get("reviewed", False)
        # Dump clips use suggested_label after review; fall back to predicted_label
        label = meta.get("suggested_label", meta.get("predicted_label", ""))

        if not reviewed:
            if include_unreviewed:
                print(f"{YELLOW}  Including unreviewed dump: {json_path.name}{RESET}")
                # For unreviewed dumps use the CNN prediction
                label = meta.get("predicted_label", "")
            else:
                skipped += 1
                continue

        if label not in VALID_LABELS:
            print(f"{YELLOW}  Skipping unknown label '{label}': {json_path.name}{RESET}")
            continue

        wav_name = meta.get("wav_file", json_path.stem + ".wav")
        wav_path = json_path.parent / wav_name
        if not wav_path.exists():
            print(f"{YELLOW}  WAV not found, skipping: {wav_path.name}{RESET}")
            continue

        run_name  = json_path.parent.parent.parent.name  # harmonic_sac_TIMESTAMP
        dump_name = json_path.parent.name                # dump_YYYYMMDD_HHMMSS
        run_label = f"{run_name}_{dump_name}"
        # store the resolved label back so downstream code can read it uniformly
        meta = dict(meta)  # shallow copy — don't mutate the on-disk version
        meta["suggested_label"] = label
        clips.append((wav_path, meta, run_label, "dump"))

    return clips, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package an augmented HarmonicsClassifier dataset as a zip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        metavar="DIR",
        help=f"Root of RL training runs (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=DEFAULT_CLASSIF,
        metavar="DIR",
        help=f"HarmonicsClassifier repo root (default: {DEFAULT_CLASSIF})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Directory to write the final zip file into (default: current directory).",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include clips that have not been reviewed yet (suggested_label used as-is). "
             "Off by default — unreviewed clips may have incorrect labels.",
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Skip copying the original HarmonicsClassifier note_clips/. "
             "Useful if you only want to ship the new RL clips.",
    )
    parser.add_argument(
        "--no-dumps",
        action="store_true",
        help="Skip audio dump clips (runs/*/audio_dumps/dump_*/) — include only "
             "success clips from runs/*/successes/.",
    )
    args = parser.parse_args()

    runs_dir       = args.runs_dir.resolve()
    classifier_dir = args.classifier_dir.resolve()
    output_dir     = args.output_dir.resolve()
    note_clips_src = classifier_dir / "note_clips"

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    if not runs_dir.exists():
        print(f"{RED}Runs directory not found: {runs_dir}{RESET}", file=sys.stderr)
        sys.exit(1)

    if not args.no_original and not note_clips_src.exists():
        print(
            f"{RED}HarmonicsClassifier note_clips/ not found: {note_clips_src}\n"
            f"Pass --no-original to skip copying the existing dataset, or\n"
            f"pass --classifier-dir <path> to point to the correct location.{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect RL clips
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{CYAN}=== Collecting reviewed RL clips ==={RESET}")

    success_clips, skipped_s = _collect_rl_clips(runs_dir, args.include_unreviewed)
    if skipped_s:
        print(
            f"{YELLOW}  Skipped {skipped_s} unreviewed success clip(s). "
            f"Run with --all-runs or pass --include-unreviewed.{RESET}"
        )
    print(f"  Success clips found : {len(success_clips)}")

    dump_clips: list = []
    skipped_d = 0
    if not args.no_dumps:
        dump_clips, skipped_d = _collect_dump_rl_clips(runs_dir, args.include_unreviewed)
        if skipped_d:
            print(
                f"{YELLOW}  Skipped {skipped_d} unreviewed dump clip(s). "
                f"Run with --run-dumps / --all-dumps or pass --include-unreviewed.{RESET}"
            )
        print(f"  Dump clips found    : {len(dump_clips)}")
    else:
        print(f"  Dump clips          : skipped (--no-dumps)")

    rl_clips = success_clips + dump_clips
    skipped  = skipped_s + skipped_d

    if not rl_clips and not args.no_original:
        print(f"{YELLOW}  No reviewed RL clips found — exporting original dataset only.{RESET}")
    elif not rl_clips and args.no_original:
        print(f"{RED}No reviewed RL clips found and --no-original set. Nothing to export.{RESET}")
        sys.exit(1)

    rl_by_label: dict = defaultdict(list)
    for wav_path, meta, run_label, source_type in rl_clips:
        rl_by_label[meta["suggested_label"]].append((wav_path, meta, run_label, source_type))

    print(f"  RL clips to include:")
    for label in VALID_LABELS:
        n = len(rl_by_label[label])
        print(f"    {label:<15}: {n}")
    print(f"  Total RL clips: {len(rl_clips)}")

    # ------------------------------------------------------------------
    # Original dataset counts
    # ------------------------------------------------------------------
    orig_counts: dict = {}
    if not args.no_original:
        print(f"\n{BOLD}{CYAN}=== Original dataset ==={RESET}")
        for label in VALID_LABELS:
            label_dir = note_clips_src / label
            n = _count_wavs(label_dir) if label_dir.exists() else 0
            orig_counts[label] = n
            print(f"    {label:<15}: {n}")
        print(f"  Total original: {sum(orig_counts.values())}")

    # ------------------------------------------------------------------
    # Confirm
    # ------------------------------------------------------------------
    total_new = sum(orig_counts.get(l, 0) + len(rl_by_label[l]) for l in VALID_LABELS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"harmonics_dataset_augmented_{timestamp}"
    zip_path = output_dir / f"{archive_name}.zip"

    print(f"\n{BOLD}Output zip  : {zip_path}{RESET}")
    print(f"Total clips : {total_new}")
    print()
    print("Continue? [y/n]: ", end="", flush=True)

    try:
        import tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        ch = input("").strip()[:1]
    print(ch)

    if ch.lower() != "y":
        print("Aborted.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Stage directory
    # ------------------------------------------------------------------
    stage_dir = output_dir / archive_name
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_clips = stage_dir / "note_clips"
    for label in VALID_LABELS:
        (stage_clips / label).mkdir(parents=True)

    # Copy original note_clips
    if not args.no_original:
        print(f"\n{BOLD}Copying original dataset...{RESET}")
        for label in VALID_LABELS:
            src_label_dir = note_clips_src / label
            dst_label_dir = stage_clips / label
            if not src_label_dir.exists():
                continue
            copied = 0
            for wav in sorted(src_label_dir.glob("*.wav")):
                shutil.copy2(wav, dst_label_dir / wav.name)
                copied += 1
            print(f"  {label:<15}: {copied} file(s) copied")

    # Copy RL clips — prefix with RL_ to avoid name collisions
    print(f"\n{BOLD}Adding RL clips...{RESET}")
    run_clip_idx: Counter = Counter()
    manifest_entries = []

    for label in VALID_LABELS:
        for wav_path, meta, run_label, source_type in rl_by_label[label]:
            run_clip_idx[run_label] += 1
            idx = run_clip_idx[run_label]
            dest_name = f"RL_{run_label}_{idx:04d}_{wav_path.name}"
            dest      = stage_clips / label / dest_name
            shutil.copy2(wav_path, dest)
            manifest_entries.append({
                "dest_file":     f"note_clips/{label}/{dest_name}",
                "source_type":   source_type,
                "source_run":    run_label,
                "source_wav":    wav_path.name,
                "label":         label,
                "reviewed":      meta.get("reviewed", False),
                "harmonic_prob": meta.get("harmonic_prob"),
                "fret_position": meta.get("fret_position"),
                "torque":        meta.get("torque"),
                "string_index":  meta.get("string_index"),
                "episode":       meta.get("episode"),
                "dump_ts":       meta.get("dump_ts"),
                "buffer_position": meta.get("buffer_position"),
            })
        print(f"  {label:<15}: {len(rl_by_label[label])} RL clip(s) added")

    # Write manifest
    success_count_by_label = {l: sum(1 for e in manifest_entries if e["label"] == l and e["source_type"] == "success") for l in VALID_LABELS}
    dump_count_by_label    = {l: sum(1 for e in manifest_entries if e["label"] == l and e["source_type"] == "dump")    for l in VALID_LABELS}
    manifest = {
        "created":                datetime.now().isoformat(),
        "original_counts":        orig_counts,
        "rl_counts":              {l: len(rl_by_label[l]) for l in VALID_LABELS},
        "rl_success_counts":      success_count_by_label,
        "rl_dump_counts":         dump_count_by_label,
        "total_counts":           {
            l: orig_counts.get(l, 0) + len(rl_by_label[l]) for l in VALID_LABELS
        },
        "skipped_unreviewed":     skipped,
        "rl_clips":               manifest_entries,
    }
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  Manifest written: manifest.json")

    # Final staged counts
    print(f"\n{BOLD}Staged dataset counts:{RESET}")
    for label in VALID_LABELS:
        n = _count_wavs(stage_clips / label)
        print(f"  {label:<15}: {n}")

    # ------------------------------------------------------------------
    # Zip
    # ------------------------------------------------------------------
    print(f"\n{BOLD}Zipping → {zip_path.name} ...{RESET}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(stage_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(stage_dir))

    zip_mb = zip_path.stat().st_size / 1_048_576
    print(f"  Done — {zip_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Clean up staging directory
    # ------------------------------------------------------------------
    print(f"{BOLD}Removing staging directory...{RESET}")
    shutil.rmtree(stage_dir)

    print(f"\n{GREEN}{BOLD}✓ Export complete:{RESET} {zip_path}\n")
    print(
        "  To use on another machine:\n"
        f"    unzip {zip_path.name}\n"
        "    # Move note_clips/ into HarmonicsClassifier/, then run:\n"
        "    python run_build_dataset.py\n"
        "    python train_cnn.py\n"
    )


if __name__ == "__main__":
    main()
