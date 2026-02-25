#!/usr/bin/env python3
"""
Review script for clips recorded during RL training.

Supports two clip sources:

  SUCCESS CLIPS  — saved by --record-successes during training
    python scripts/review_successes.py runs/RUN/successes/
    python scripts/review_successes.py --all-runs [RUNS_DIR]
    python scripts/review_successes.py --all-runs ./runs --unreviewed-only

  AUDIO DUMP CLIPS  — saved by pressing 'a' during --audio-history training
    python scripts/review_successes.py --dump runs/RUN/audio_dumps/dump_TS/
    python scripts/review_successes.py --run-dumps runs/RUN/
    python scripts/review_successes.py --run-dumps runs/RUN/ --unreviewed-only
    python scripts/review_successes.py --all-dumps [RUNS_DIR]

Common options:
    --unreviewed-only   skip clips already marked reviewed=true
    --no-audio          disable playback (headless machines)

Controls (during review):
    h  — label as harmonic   (approve if already harmonic)
    d  — label as dead_note
    g  — label as general_note
    r  — replay audio
    s  — skip (leave as-is, do NOT set reviewed=True)
    q  — quit and save summary
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
DIM    = "\033[2m"

LABEL_STYLES = {
    "harmonic":     f"{GREEN}harmonic{RESET}",
    "dead_note":    f"{RED}dead_note{RESET}",
    "general_note": f"{YELLOW}general_note{RESET}",
}

VALID_LABELS = {"harmonic", "dead_note", "general_note"}


def _fmt_label(label: str) -> str:
    return LABEL_STYLES.get(label, label)


def _clear_line() -> None:
    print("\r\033[K", end="", flush=True)


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------

def _try_import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        return None


def play_audio(wav_path: Path, sd, device=None) -> None:
    """Play a WAV file to the chosen output device, blocking until done."""
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    sd.play(audio, samplerate=sr, device=device)
    sd.wait()


def play_with_replay(wav_path: Path, sd, device=None) -> None:
    """Play once; caller can trigger replay via 'r' prompt key."""
    play_audio(wav_path, sd, device=device)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _prompt(prompt_str: str) -> str:
    """Read a single character without requiring Enter (Unix only)."""
    import tty
    import termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _prompt_fallback(prompt_str: str) -> str:
    """Line-buffered fallback for environments without tty support."""
    raw = input(prompt_str + " ").strip().lower()
    return raw[:1] if raw else ""


def _prompt_output_device(sd) -> Optional[int]:
    """List output-capable devices and let the user choose one.

    Returns the sounddevice device ID, or None to use the system default.
    """
    devices = sd.query_devices()
    output_devices = [
        (idx, dev)
        for idx, dev in enumerate(devices)
        if dev["max_output_channels"] > 0
    ]

    if not output_devices:
        print(f"{YELLOW}No output devices found — audio will be disabled.{RESET}")
        return None

    print(f"\n{BOLD}=== Select Playback Output Device ==={RESET}")
    for i, (idx, dev) in enumerate(output_devices):
        marker = " (system default)" if idx == sd.default.device[1] else ""
        print(
            f"  [{i}]  ID={idx:2d}  ch={dev['max_output_channels']:2d}  "
            f"SR={int(dev['default_samplerate'])}  {dev['name']}{marker}"
        )
    print(f"  [d]  Use system default")
    print()

    try:
        prompt_fn = _prompt
        import tty, termios  # noqa: F401
    except ImportError:
        prompt_fn = _prompt_fallback

    while True:
        print("  Select device number (or 'd' for default): ", end="", flush=True)
        ch = prompt_fn("")
        print(ch)

        if ch.lower() == "d":
            print(f"  Using system default output device.\n")
            return None

        try:
            choice = int(ch)
            if 0 <= choice < len(output_devices):
                dev_id, dev = output_devices[choice]
                print(f"  Using: {dev['name']} (ID: {dev_id})\n")
                return dev_id
            else:
                print(f"  Enter a number between 0 and {len(output_devices)-1}, or 'd'.")
        except ValueError:
            print(f"  Enter a number or 'd'.")


# ---------------------------------------------------------------------------
# Core review loop
# ---------------------------------------------------------------------------

def review_clip(wav_path: Path, meta: dict, index: int, total: int, sd, no_audio: bool,
                device=None, run_label: str = "") -> str:
    """
    Display clip metadata, (optionally) play audio, prompt for label.

    Returns:
        'harmonic' | 'dead_note' | 'general_note' — if user labelled
        's'                                        — skip (no change)
        'q'                                        — quit
    """
    current_label = meta.get("suggested_label", "harmonic")
    reviewed      = meta.get("reviewed", False)
    fret          = meta.get("fret_position", "?")
    torque        = meta.get("torque", "?")
    string_idx    = meta.get("string_index", "?")
    harm_prob     = meta.get("harmonic_prob",    meta.get("harmonic_probability", "?"))
    dead_prob     = meta.get("dead_prob",        "?")
    gen_prob      = meta.get("general_prob",     "?")
    reward        = meta.get("reward",           "?")

    reviewed_tag = f" {DIM}[reviewed]{RESET}" if reviewed else ""
    run_tag = f" {DIM}[{run_label}]{RESET}" if run_label else ""

    print()
    print(f"{BOLD}─── [{index}/{total}] {wav_path.name} ───{RESET}{reviewed_tag}{run_tag}")
    print(f"  String: {string_idx}   Fret: {fret}   Torque: {torque}")

    # Format probabilities if numeric
    try:
        probs = (
            f"  H={float(harm_prob):.2f}  D={float(dead_prob):.2f}  G={float(gen_prob):.2f}"
        )
    except (TypeError, ValueError):
        probs = f"  probs unavailable"
    print(probs + f"   reward={reward}")
    print(f"  Suggested label: {_fmt_label(current_label)}")

    if not no_audio:
        print(f"  {DIM}Playing...{RESET}", end="", flush=True)
        try:
            play_audio(wav_path, sd, device=device)
        except Exception as exc:
            print(f"\n  {YELLOW}Audio playback failed: {exc}{RESET}")
        _clear_line()

    try:
        prompt_fn = _prompt
        # Quick self-test: will raise if not a real tty
        import tty, termios  # noqa: F401
    except ImportError:
        prompt_fn = _prompt_fallback

    while True:
        key_hint = (
            f"  {BOLD}[h]{RESET}armonic  "
            f"{BOLD}[d]{RESET}ead_note  "
            f"{BOLD}[g]{RESET}eneral_note  "
            f"{BOLD}[r]{RESET}eplay  "
            f"{BOLD}[s]{RESET}kip  "
            f"{BOLD}[q]{RESET}uit"
        )
        print(key_hint, end=" › ", flush=True)

        ch = prompt_fn("")
        print(ch)  # echo

        if ch == "h":
            return "harmonic"
        elif ch == "d":
            return "dead_note"
        elif ch == "g":
            return "general_note"
        elif ch == "r":
            if no_audio:
                print(f"  {YELLOW}(--no-audio, replay skipped){RESET}")
            else:
                print(f"  {DIM}Replaying...{RESET}", end="", flush=True)
                try:
                    play_audio(wav_path, sd, device=device)
                except Exception as exc:
                    print(f"\n  {YELLOW}Replay failed: {exc}{RESET}")
                _clear_line()
        elif ch == "s":
            return "s"
        elif ch == "q":
            return "q"
        else:
            print(f"  {DIM}Unrecognized key '{ch}'. Use h/d/g/r/s/q.{RESET}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _collect_clips(dirs: list[Path], unreviewed_only: bool) -> list[tuple[Path, Path, dict, str]]:
    """Return a flat sorted list of (wav, json, meta, run_label) tuples."""
    clips = []
    for d in dirs:
        run_label = d.parent.name  # e.g. harmonic_sac_20260220_141411
        for jp in sorted(d.glob("*.json")):
            try:
                meta = json.loads(jp.read_text())
            except Exception:
                continue
            wav_name = meta.get("wav_file", jp.stem + ".wav")
            wp = d / wav_name
            if not wp.exists():
                print(f"{DIM}  Skipping {jp.name} (no matching WAV){RESET}")
                continue
            if unreviewed_only and meta.get("reviewed", False):
                continue
            clips.append((wp, jp, meta, run_label))
    return clips


def _collect_dump_clips(
    dump_dirs: list[Path], unreviewed_only: bool
) -> list[tuple[Path, Path, dict, str]]:
    """Return a flat sorted list of (wav, json, meta, dump_label) for audio dump clips.

    dump_label is '<run_name>/<dump_dirname>', e.g.
    'harmonic_sac_20260220_170023/dump_20260220_171026'
    """
    clips = []
    for d in dump_dirs:
        run_name  = d.parent.parent.name   # …/runs/<run_name>/audio_dumps/<dump_ts>/
        dump_name = d.name                 # dump_YYYYMMDD_HHMMSS
        dump_label = f"{run_name}/{dump_name}"
        for jp in sorted(d.glob("*.json")):
            try:
                meta = json.loads(jp.read_text())
            except Exception:
                continue
            wav_name = meta.get("wav_file", jp.stem + ".wav")
            wp = d / wav_name
            if not wp.exists():
                print(f"{DIM}  Skipping {jp.name} (no matching WAV){RESET}")
                continue
            if unreviewed_only and meta.get("reviewed", False):
                continue
            clips.append((wp, jp, meta, dump_label))
    return clips


def review_dump_clip(
    wav_path: Path,
    meta: dict,
    index: int,
    total: int,
    sd,
    no_audio: bool,
    device=None,
    dump_label: str = "",
) -> str:
    """
    Display audio-dump clip metadata, optionally play, prompt for label.

    Returns:
        'harmonic' | 'dead_note' | 'general_note' — if user labelled
        's'                                        — skip
        'q'                                        — quit
    """
    buf_pos     = meta.get("buffer_position", "?")
    buf_size    = meta.get("buffer_size",     "?")
    dump_ts     = meta.get("dump_ts",         "?")
    string_idx  = meta.get("string_index",    "?")
    fret        = meta.get("fret_position",   "?")
    torque      = meta.get("torque",          "?")
    harm_prob   = meta.get("harmonic_prob",   "?")
    dead_prob   = meta.get("dead_prob",       "?")
    gen_prob    = meta.get("general_prob",    "?")
    pred_label  = meta.get("predicted_label", "?")
    total_rew   = meta.get("total_reward",    "?")
    audio_rew   = meta.get("audio_reward",    "?")
    fret_rew    = meta.get("fret_reward",     "?")
    torque_rew  = meta.get("torque_reward",   "?")
    filtered    = meta.get("filtered",        "?")
    rl_action   = meta.get("rl_action",       "?")
    reviewed    = meta.get("reviewed",        False)
    # current label: relabelled value if already reviewed, else CNN prediction
    current_label = meta.get("suggested_label", pred_label)

    reviewed_tag  = f" {DIM}[reviewed]{RESET}" if reviewed else ""
    dump_tag      = f" {DIM}[{dump_label}]{RESET}" if dump_label else ""
    filtered_tag  = f" {YELLOW}[filtered]{RESET}" if filtered else ""

    print()
    print(
        f"{BOLD}─── [{index}/{total}] {wav_path.name} ───{RESET}"
        f"{reviewed_tag}{dump_tag}"
    )
    print(f"  Dump     : {dump_ts}  buf {buf_pos}/{buf_size}{filtered_tag}")
    print(f"  String: {string_idx}   Fret: {fret}   Torque: {torque}")

    try:
        probs = (
            f"  H={float(harm_prob):.3f}  D={float(dead_prob):.3f}  G={float(gen_prob):.3f}"
        )
    except (TypeError, ValueError):
        probs = "  probs unavailable"
    print(probs + f"   CNN→ {_fmt_label(pred_label)}")

    try:
        rew_line = (
            f"  rew: total={float(total_rew):.3f}  "
            f"audio={float(audio_rew):.3f}  "
            f"fret={float(fret_rew):.3f}  "
            f"torque={float(torque_rew):.3f}"
        )
    except (TypeError, ValueError):
        rew_line = f"  reward: {total_rew}"
    print(rew_line)
    print(f"  Action   : {rl_action}")
    print(f"  Label now: {_fmt_label(str(current_label))}")

    if not no_audio:
        print(f"  {DIM}Playing...{RESET}", end="", flush=True)
        try:
            play_audio(wav_path, sd, device=device)
        except Exception as exc:
            print(f"\n  {YELLOW}Audio playback failed: {exc}{RESET}")
        _clear_line()

    try:
        prompt_fn = _prompt
        import tty, termios  # noqa: F401
    except ImportError:
        prompt_fn = _prompt_fallback

    while True:
        key_hint = (
            f"  {BOLD}[h]{RESET}armonic  "
            f"{BOLD}[d]{RESET}ead_note  "
            f"{BOLD}[g]{RESET}eneral_note  "
            f"{BOLD}[r]{RESET}eplay  "
            f"{BOLD}[s]{RESET}kip  "
            f"{BOLD}[q]{RESET}uit"
        )
        print(key_hint, end=" › ", flush=True)
        ch = prompt_fn("")
        print(ch)

        if ch == "h":
            return "harmonic"
        elif ch == "d":
            return "dead_note"
        elif ch == "g":
            return "general_note"
        elif ch == "r":
            if no_audio:
                print(f"  {YELLOW}(--no-audio, replay skipped){RESET}")
            else:
                print(f"  {DIM}Replaying...{RESET}", end="", flush=True)
                try:
                    play_audio(wav_path, sd, device=device)
                except Exception as exc:
                    print(f"\n  {YELLOW}Replay failed: {exc}{RESET}")
                _clear_line()
        elif ch == "s":
            return "s"
        elif ch == "q":
            return "q"
        else:
            print(f"  {DIM}Unrecognized key '{ch}'. Use h/d/g/r/s/q.{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively review and relabel recorded harmonic clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # -- success clip args --------------------------------------------------
    parser.add_argument(
        "successes_dir",
        type=Path,
        nargs="?",
        help="Path to a successes/ directory produced by --record-successes. "
             "Not required when --all-runs is set.",
    )
    parser.add_argument(
        "--all-runs",
        metavar="RUNS_DIR",
        nargs="?",
        const="./runs",
        default=None,
        help="Scan every */successes/ directory under RUNS_DIR (default: ./runs).",
    )
    # -- audio dump args ----------------------------------------------------
    parser.add_argument(
        "--dump",
        metavar="DUMP_DIR",
        type=Path,
        default=None,
        help="Review a single audio_dumps/dump_TS/ directory.",
    )
    parser.add_argument(
        "--run-dumps",
        metavar="RUN_DIR",
        nargs="?",
        const=".",
        default=None,
        help="Review all dump_TS/ directories under RUN_DIR/audio_dumps/ "
             "(default: current directory).",
    )
    parser.add_argument(
        "--all-dumps",
        metavar="RUNS_DIR",
        nargs="?",
        const="./runs",
        default=None,
        help="Review every */audio_dumps/dump_TS/ directory under RUNS_DIR "
             "(default: ./runs).",
    )
    # -- shared args --------------------------------------------------------
    parser.add_argument(
        "--unreviewed-only",
        action="store_true",
        help="Skip clips that have already been reviewed (reviewed=true in JSON).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio playback (useful on headless machines).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Determine mode: dumps vs successes
    # -----------------------------------------------------------------------
    dump_mode = args.dump is not None or args.run_dumps is not None or args.all_dumps is not None

    if dump_mode:
        # -------------------------------------------------------------------
        # Resolve dump directories
        # -------------------------------------------------------------------
        if args.dump is not None:
            dump_dirs = [args.dump.resolve()]
            if not dump_dirs[0].exists():
                print(f"{RED}Directory not found: {dump_dirs[0]}{RESET}", file=sys.stderr)
                sys.exit(1)
        elif args.run_dumps is not None:
            run_dir = Path(args.run_dumps).resolve()
            if not run_dir.exists():
                print(f"{RED}Run directory not found: {run_dir}{RESET}", file=sys.stderr)
                sys.exit(1)
            dump_dirs = sorted(
                p for p in (run_dir / "audio_dumps").glob("dump_*")
                if p.is_dir()
            )
            if not dump_dirs:
                print(f"{YELLOW}No dump_*/ directories found under {run_dir / 'audio_dumps'}{RESET}")
                sys.exit(0)
        else:  # --all-dumps
            runs_root = Path(args.all_dumps).resolve()
            if not runs_root.exists():
                print(f"{RED}Runs directory not found: {runs_root}{RESET}", file=sys.stderr)
                sys.exit(1)
            dump_dirs = sorted(
                p for p in runs_root.glob("*/audio_dumps/dump_*")
                if p.is_dir()
            )
            if not dump_dirs:
                print(f"{YELLOW}No dump_*/ directories found under {runs_root}{RESET}")
                sys.exit(0)

        clips = _collect_dump_clips(dump_dirs, args.unreviewed_only)
        clip_review_fn  = review_dump_clip
        label_key       = "dump_label"  # informational only
        source_dirs     = dump_dirs
        mode_name       = "Audio Dump"
        location_label  = (
            Path(args.all_dumps).resolve() if args.all_dumps
            else (Path(args.run_dumps).resolve() if args.run_dumps
                  else dump_dirs[0])
        )

    else:
        # -------------------------------------------------------------------
        # Resolve success directories (original behaviour)
        # -------------------------------------------------------------------
        if args.all_runs is not None:
            runs_root = Path(args.all_runs).resolve()
            if not runs_root.exists():
                print(f"{RED}Runs directory not found: {runs_root}{RESET}", file=sys.stderr)
                sys.exit(1)
            success_dirs = sorted(
                p for p in runs_root.glob("*/successes")
                if p.is_dir()
            )
            if not success_dirs:
                print(f"{YELLOW}No successes/ directories found under {runs_root}{RESET}")
                sys.exit(0)
        elif args.successes_dir is not None:
            success_dirs = [args.successes_dir.resolve()]
            if not success_dirs[0].exists():
                print(f"{RED}Directory not found: {success_dirs[0]}{RESET}", file=sys.stderr)
                sys.exit(1)
        else:
            print(
                f"{RED}Error: provide a successes/ directory, use --all-runs,\n"
                f"       or use --dump / --run-dumps / --all-dumps for audio dump review.{RESET}\n"
                f"  Examples:\n"
                f"    python scripts/review_successes.py runs/RUN/successes/\n"
                f"    python scripts/review_successes.py --all-runs\n"
                f"    python scripts/review_successes.py --dump runs/RUN/audio_dumps/dump_TS/\n"
                f"    python scripts/review_successes.py --run-dumps runs/RUN/\n"
                f"    python scripts/review_successes.py --all-dumps",
                file=sys.stderr,
            )
            sys.exit(1)

        clips = _collect_clips(success_dirs, args.unreviewed_only)
        clip_review_fn  = review_clip
        source_dirs     = success_dirs
        mode_name       = "Harmonic Clip"
        location_label  = (
            Path(args.all_runs).resolve() if args.all_runs
            else success_dirs[0]
        )

    # -----------------------------------------------------------------------
    # Collect clips and show pre-flight summary
    # -----------------------------------------------------------------------

    if not clips:
        msg = "No unreviewed clips found." if args.unreviewed_only else "No clips found."
        print(f"{YELLOW}{msg}{RESET}")
        sys.exit(0)

    # Per-source breakdown
    from collections import Counter
    src_counts = Counter(label for _, _, _, label in clips)

    print(f"\n{BOLD}{CYAN}=== Pre-flight Summary ({mode_name} mode) ==={RESET}")
    if len(source_dirs) > 1:
        print(f"  Sources:")
        for label, count in sorted(src_counts.items()):
            print(f"    {label}: {count} clip(s)")
    else:
        print(f"  Directory : {source_dirs[0]}")
    qualifier = " unreviewed" if args.unreviewed_only else ""
    print(f"  {BOLD}Total{qualifier} clips : {len(clips)}{RESET}")
    print()

    try:
        prompt_fn = _prompt
        import tty, termios  # noqa: F401
    except ImportError:
        prompt_fn = _prompt_fallback

    print("  Continue? [y/n]: ", end="", flush=True)
    ch = prompt_fn("")
    print(ch)
    if ch.lower() != "y":
        print("Aborted.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Audio backend + output device selection
    # -----------------------------------------------------------------------
    sd = None
    output_device = None
    if not args.no_audio:
        sd = _try_import_sounddevice()
        if sd is None:
            print(
                f"{YELLOW}sounddevice not installed — audio playback disabled.{RESET}\n"
                f"  Install with: pip install sounddevice\n"
            )
            args.no_audio = True
        else:
            output_device = _prompt_output_device(sd)

    # -----------------------------------------------------------------------
    # Review loop
    # -----------------------------------------------------------------------
    total = len(clips)
    counts = {"harmonic": 0, "dead_note": 0, "general_note": 0, "skipped": 0}

    print(
        f"\n{BOLD}{CYAN}=== {mode_name} Review ==={RESET}\n"
        f"  Clips : {total}"
        + (" (unreviewed only)" if args.unreviewed_only else "")
        + f"\n  Audio : {'disabled (--no-audio)' if args.no_audio else 'enabled'}\n"
    )

    for i, (wav_path, json_path, meta, src_label) in enumerate(clips, start=1):
        multi_src = len(source_dirs) > 1
        result = clip_review_fn(
            wav_path, meta, i, total, sd, args.no_audio,
            device=output_device,
            **({"run_label": src_label if multi_src else ""} if not dump_mode
               else {"dump_label": src_label if multi_src else ""}),
        )

        if result == "q":
            print(f"\n{YELLOW}Quit.{RESET}")
            break

        if result == "s":
            counts["skipped"] += 1
            continue

        # Update JSON in place
        old_label = meta.get("suggested_label", "harmonic")
        meta["suggested_label"] = result
        meta["reviewed"] = True
        json_path.write_text(json.dumps(meta, indent=2, default=str))

        change_note = (
            f"  {DIM}(unchanged){RESET}"
            if result == old_label
            else f"  {DIM}{old_label} → {result}{RESET}"
        )
        print(f"  {GREEN}✓ Saved{RESET} {_fmt_label(result)}{change_note}")
        counts[result] += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    reviewed_total = counts["harmonic"] + counts["dead_note"] + counts["general_note"]
    print(
        f"\n{BOLD}─── Session Summary ───{RESET}\n"
        f"  Reviewed  : {reviewed_total}\n"
        f"  {_fmt_label('harmonic'):<30s}: {counts['harmonic']}\n"
        f"  {_fmt_label('dead_note'):<30s}: {counts['dead_note']}\n"
        f"  {_fmt_label('general_note'):<30s}: {counts['general_note']}\n"
        f"  Skipped   : {counts['skipped']}\n"
    )

    # Check for remaining unreviewed clips across all source directories
    remaining = sum(
        1
        for d in source_dirs
        for jp in d.glob("*.json")
        if not json.loads(jp.read_text()).get("reviewed", False)
    )
    if remaining:
        print(f"  {YELLOW}{remaining} clip(s) still unreviewed in {location_label}{RESET}")
    else:
        print(f"  {GREEN}All clips in this source are reviewed.{RESET}")
    print()


if __name__ == "__main__":
    main()
