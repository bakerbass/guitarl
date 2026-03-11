"""
Inspect training runs one at a time, plotting reward over episodes.

Default behavior (no flags):
- Goes through runs in reverse chronological order (latest first)
- Skips short runs (false starts with <10 episodes logged)
- Shows reward vs episodes plot for each run
- Press 'n' for next, 'p' for previous, 'q' to quit
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_runs_sorted(runs_dir: Path, reverse: bool = True) -> List[Path]:
    """Get all run directories sorted by modification time."""
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    # Sort by modification time
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=reverse)
    return run_dirs


def load_run_data(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load progress.csv from a run directory."""
    csv_path = run_dir / "logs" / "progress.csv"
    if not csv_path.exists():
        logger.debug(f"No progress.csv found in {run_dir.name}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.warning(f"Failed to read {csv_path}: {e}")
        return None


def is_valid_run(df: pd.DataFrame, min_rows: int = 10) -> bool:
    """Check if run has enough data (filter out false starts)."""
    if df is None or len(df) < min_rows:
        return False
    return True


def load_parent_run(run_dir: Path) -> Optional[pd.DataFrame]:
    """If this run was resumed from another, load that run's progress.csv."""
    marker = run_dir / 'resumed_from.txt'
    if not marker.exists():
        return None
    try:
        parent_path = Path(marker.read_text().strip())
        return load_run_data(parent_path)
    except Exception as e:
        logger.debug(f"Could not load parent run: {e}")
        return None


def _extract_reward_series(df: pd.DataFrame):
    """Return (episode_array, reward_array, len_array_or_None) from a progress df."""
    episode_col, reward_col, len_col = None, None, None

    if 'time/episodes' in df.columns:
        episode_col = 'time/episodes'
    elif 'episodes' in df.columns:
        episode_col = 'episodes'

    if 'rollout/ep_rew_mean' in df.columns:
        reward_col = 'rollout/ep_rew_mean'
    elif 'ep_rew_mean' in df.columns:
        reward_col = 'ep_rew_mean'
    elif 'reward' in df.columns:
        reward_col = 'reward'

    if 'rollout/ep_len_mean' in df.columns:
        len_col = 'rollout/ep_len_mean'
    elif 'ep_len_mean' in df.columns:
        len_col = 'ep_len_mean'

    if episode_col is None or reward_col is None:
        return None, None, None

    needed = [episode_col, reward_col] + ([len_col] if len_col else [])
    sub = df[needed].dropna(subset=[episode_col, reward_col])

    lens = sub[len_col].values if len_col else None
    return sub[episode_col].values, sub[reward_col].values, lens


def plot_run(run_dir: Path, df: pd.DataFrame, run_index: int, total_runs: int):
    """Plot reward-per-step (reward normalised by episode length) over training."""
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(13, 6)
    
    # Extract run name and timestamp
    run_name = run_dir.name
    timestamp_str = ""
    if "harmonic_sac_" in run_name:
        timestamp_part = run_name.replace("harmonic_sac_", "")
        try:
            dt = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            timestamp_str = timestamp_part

    # ── Load current + optional parent run data ────────────────────────────
    cur_eps, cur_rew, cur_len = _extract_reward_series(df)

    if cur_eps is None:
        logger.error(f"Could not find episode or reward columns in {run_name}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return

    parent_df = load_parent_run(run_dir)
    par_eps, par_rew, par_len = _extract_reward_series(parent_df) if parent_df is not None else (None, None, None)

    # ── Stitch into a single sequential x-axis ────────────────────────────
    if par_eps is not None and len(par_eps) > 0:
        n_parent = len(par_eps)
        x = list(range(n_parent)) + list(range(n_parent, n_parent + len(cur_eps)))
        rewards = list(par_rew) + list(cur_rew)

        # lengths: combine if both present, else use only current
        if par_len is not None and cur_len is not None:
            lengths = list(par_len) + list(cur_len)
        elif cur_len is not None:
            lengths = [None] * n_parent + list(cur_len)
        elif par_len is not None:
            lengths = list(par_len) + [None] * len(cur_eps)
        else:
            lengths = None

        phase_boundary = n_parent  # index where phase 2 begins
    else:
        x = list(range(len(cur_eps)))
        rewards = list(cur_rew)
        lengths = list(cur_len) if cur_len is not None else None
        phase_boundary = None

    import numpy as np
    x = np.array(x, dtype=float)
    rewards = np.array(rewards, dtype=float)

    # ── Normalise by episode length ───────────────────────────────────────
    if lengths is not None:
        lens_arr = np.array([l if l is not None else np.nan for l in lengths], dtype=float)
        # forward-fill NaNs so gaps in the parent don't blow up
        lens_ser = pd.Series(lens_arr).fillna(method='ffill').fillna(1.0).values
        lens_arr = np.where(lens_ser > 0, lens_ser, 1.0)
        y = rewards / lens_arr
        y_label = 'Reward per Step  (ep_rew_mean / ep_len_mean)'
    else:
        y = rewards
        y_label = 'Mean Episode Reward  (no length data — unnormalised)'

    # ── Figure ────────────────────────────────────────────────────────────
    fig.suptitle(
        f'Run {run_index + 1}/{total_runs}:  {run_name}'
        + (f'  ({timestamp_str})' if timestamp_str else ''),
        fontsize=12, fontweight='bold', y=0.98
    )

    ax = fig.add_subplot(1, 1, 1)

    if phase_boundary is not None:
        # Phase 1 (parent run)
        ax.plot(x[:phase_boundary], y[:phase_boundary],
                linewidth=2, color='#E84855', alpha=0.75, label='phase 1 (pretrain)')
        # Phase 2 (this run)
        ax.plot(x[phase_boundary:], y[phase_boundary:],
                linewidth=2, color='#2E86AB', alpha=0.85, label='phase 2')
        ax.axvline(phase_boundary, color='orange', linewidth=1.5, linestyle='--',
                   label='phase 1 → 2')
        ax.text(phase_boundary + 0.5, 0.97, 'phase 2 start',
                transform=ax.get_xaxis_transform(),
                color='orange', fontsize=8, verticalalignment='top', rotation=90)
    else:
        ax.plot(x, y, linewidth=2, color='#2E86AB', alpha=0.85)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xlabel('Episodes (sequential index)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    if phase_boundary is not None:
        ax.legend(loc='upper left', fontsize=9)

    # Stats annotation (phase 2 only)
    disp = y[phase_boundary:] if phase_boundary is not None else y
    if len(disp):
        stats_text = (
            f"final: {disp[-1]:.3f}\n"
            f"max:  {float(pd.Series(disp).max()):.3f}\n"
            f"mean: {float(pd.Series(disp).mean()):.3f}\n"
            f"std:  {float(pd.Series(disp).std()):.3f}"
        )
        ax.text(
            0.99, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )

    # Navigation hint
    plt.figtext(
        0.5, 0.005,
        "n / → : next     p / ← : previous     q / Esc : quit",
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.draw()
    plt.pause(0.01)


def inspect_runs(runs_dir: Path, min_rows: int = 10, reverse: bool = True, list_only: bool = False, last_n: int = None):
    """Main inspection loop."""
    # Get all runs sorted
    all_runs = get_runs_sorted(runs_dir, reverse=reverse)
    logger.info(f"Found {len(all_runs)} total run directories")
    
    # Filter valid runs
    valid_runs = []
    for run_dir in all_runs:
        df = load_run_data(run_dir)
        if is_valid_run(df, min_rows):
            valid_runs.append((run_dir, df))
        else:
            logger.debug(f"Skipping {run_dir.name} (insufficient data)")
    
    logger.info(f"Found {len(valid_runs)} valid runs (>{min_rows} logged episodes)")

    if last_n is not None:
        valid_runs = valid_runs[:last_n]
        logger.info(f"Limiting to last {last_n} runs")
    
    # If list-only mode, just print runs and exit
    if list_only:
        print(f"\n{'='*80}")
        print(f"Valid runs ({len(valid_runs)} total):")
        print(f"{'='*80}\n")
        for idx, (run_dir, df) in enumerate(valid_runs, 1):
            run_name = run_dir.name
            timestamp_str = ""
            if "harmonic_sac_" in run_name:
                timestamp_part = run_name.replace("harmonic_sac_", "")
                try:
                    dt = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    timestamp_str = timestamp_part
            
            # Get reward column
            reward_col = None
            if 'rollout/ep_rew_mean' in df.columns:
                reward_col = 'rollout/ep_rew_mean'
            elif 'ep_rew_mean' in df.columns:
                reward_col = 'ep_rew_mean'
            elif 'reward' in df.columns:
                reward_col = 'reward'
            
            if reward_col:
                series = df[reward_col].dropna()
                if series.empty:
                    print(f"{idx:3d}. {run_name}")
                    print(f"     Time: {timestamp_str}")
                    print(f"     Episodes logged: {len(df)}   |   (no valid reward values)")
                    print()
                else:
                    final_reward = series.iloc[-1]
                    max_reward = float(series.max())
                    print(f"{idx:3d}. {run_name}")
                    print(f"     Time: {timestamp_str}")
                    print(f"     Episodes logged: {len(df)}   |   Final reward: {final_reward:.2f}   |   Max reward: {max_reward:.2f}")
                    print()
        return
    
    if not valid_runs:
        logger.error("No valid runs found!")
        return
    
    # Interactive viewing
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    
    current_idx = 0
    
    def show_current():
        run_dir, df = valid_runs[current_idx]
        plot_run(run_dir, df, current_idx, len(valid_runs))
    
    def on_key(event):
        nonlocal current_idx
        
        if event.key == 'n' or event.key == 'right':
            # Next run
            current_idx = (current_idx + 1) % len(valid_runs)
            show_current()
        elif event.key == 'p' or event.key == 'left':
            # Previous run
            current_idx = (current_idx - 1) % len(valid_runs)
            show_current()
        elif event.key == 'q' or event.key == 'escape':
            # Quit
            plt.close('all')
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show first run
    show_current()
    
    logger.info(f"Showing run 1/{len(valid_runs)}")
    logger.info("Use 'n'/'p' to navigate, 'q' to quit")
    
    # Keep window open
    plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(
        description='Inspect training runs one at a time, plotting reward over episodes.'
    )
    parser.add_argument(
        '--runs-dir',
        type=Path,
        default=Path(__file__).parent / 'runs',
        help='Directory containing run folders (default: ./runs)'
    )
    parser.add_argument(
        '--min-rows',
        type=int,
        default=10,
        help='Minimum number of logged episodes to consider a run valid (default: 10)'
    )
    parser.add_argument(
        '--oldest-first',
        action='store_true',
        help='Show runs in chronological order (oldest first) instead of reverse'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Just list valid runs without opening the GUI'
    )
    parser.add_argument(
        '--last-n',
        type=int,
        default=None,
        metavar='N',
        help='Only inspect the last N runs (most recent unless --oldest-first)'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.runs_dir.exists():
        logger.error(f"Runs directory not found: {args.runs_dir}")
        return
    
    inspect_runs(
        runs_dir=args.runs_dir,
        min_rows=args.min_rows,
        reverse=not args.oldest_first,
        list_only=args.list_only,
        last_n=args.last_n,
    )


if __name__ == '__main__':
    main()
