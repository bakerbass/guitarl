#!/usr/bin/env python3
"""
Review script for successful harmonics recorded during RL training.

Iterates through each audio clip in a successes/ directory, plays it back,
and prompts you to approve or correct the suggested label.

Usage:
    python scripts/review_successes.py runs/RUN/successes/
    python scripts/review_successes.py runs/RUN/successes/ --unreviewed-only
    python scripts/review_successes.py runs/RUN/successes/ --no-audio

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


def play_audio(wav_path: Path, sd) -> None:
    """Play a WAV file to the default output device, blocking until done."""
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    sd.play(audio, samplerate=sr)
    sd.wait()


def play_with_replay(wav_path: Path, sd) -> None:
    """Play once; caller can trigger replay via 'r' prompt key."""
    play_audio(wav_path, sd)


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


# ---------------------------------------------------------------------------
# Core review loop
# ---------------------------------------------------------------------------

def review_clip(wav_path: Path, meta: dict, index: int, total: int, sd, no_audio: bool) -> str:
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

    print()
    print(f"{BOLD}─── [{index}/{total}] {wav_path.name} ───{RESET}{reviewed_tag}")
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
            play_audio(wav_path, sd)
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
                    play_audio(wav_path, sd)
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively review and relabel recorded harmonic clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "successes_dir",
        type=Path,
        help="Path to a successes/ directory produced by --record-successes.",
    )
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

    successes_dir: Path = args.successes_dir.resolve()

    if not successes_dir.exists():
        print(f"{RED}Directory not found: {successes_dir}{RESET}", file=sys.stderr)
        sys.exit(1)

    # Collect all paired (wav, json) clips, sorted by stem
    json_files = sorted(successes_dir.glob("*.json"))
    if not json_files:
        print(f"{YELLOW}No JSON files found in {successes_dir}{RESET}")
        sys.exit(0)

    clips = []
    for jp in json_files:
        meta = json.loads(jp.read_text())
        wav_name = meta.get("wav_file", jp.stem + ".wav")
        wp = successes_dir / wav_name
        if not wp.exists():
            print(f"{DIM}  Skipping {jp.name} (no matching WAV){RESET}")
            continue
        if args.unreviewed_only and meta.get("reviewed", False):
            continue
        clips.append((wp, jp, meta))

    if not clips:
        msg = "No unreviewed clips found." if args.unreviewed_only else "No clips found."
        print(f"{YELLOW}{msg}{RESET}")
        sys.exit(0)

    # Audio backend
    sd = None
    if not args.no_audio:
        sd = _try_import_sounddevice()
        if sd is None:
            print(
                f"{YELLOW}sounddevice not installed — audio playback disabled.{RESET}\n"
                f"  Install with: pip install sounddevice\n"
            )
            args.no_audio = True

    # -----------------------------------------------------------------------
    # Review loop
    # -----------------------------------------------------------------------
    total = len(clips)
    counts = {"harmonic": 0, "dead_note": 0, "general_note": 0, "skipped": 0}

    print(
        f"\n{BOLD}{CYAN}=== Harmonic Clip Review ==={RESET}\n"
        f"  Directory : {successes_dir}\n"
        f"  Clips     : {total}"
        + (" (unreviewed only)" if args.unreviewed_only else "")
        + f"\n  Audio     : {'disabled (--no-audio)' if args.no_audio else 'enabled'}\n"
    )

    for i, (wav_path, json_path, meta) in enumerate(clips, start=1):
        result = review_clip(wav_path, meta, i, total, sd, args.no_audio)

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

    # Check for remaining unreviewed clips in the directory
    remaining = sum(
        1
        for jp in successes_dir.glob("*.json")
        if not json.loads(jp.read_text()).get("reviewed", False)
    )
    if remaining:
        print(f"  {YELLOW}{remaining} clip(s) still unreviewed in {successes_dir}{RESET}")
    else:
        print(f"  {GREEN}All clips in this directory are reviewed.{RESET}")
    print()


if __name__ == "__main__":
    main()
