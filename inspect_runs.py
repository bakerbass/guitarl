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


def plot_run(run_dir: Path, df: pd.DataFrame, run_index: int, total_runs: int):
    """Plot reward and episode length over training for a single run."""
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(13, 7)
    
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
    
    # Detect columns
    episode_col = None
    reward_col = None
    len_col = None
    
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
        logger.error(f"Could not find episode or reward columns in {run_name}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return
    
    # Drop rows where episodes or rewards are NaN
    plot_df = df[[episode_col, reward_col] + ([len_col] if len_col else [])].dropna(
        subset=[episode_col, reward_col]
    )
    
    episodes = plot_df[episode_col].values
    rewards = plot_df[reward_col].values
    
    # ── Figure layout ──────────────────────────────────────────────────────
    n_plots = 2 if len_col else 1
    axes = fig.subplots(n_plots, 1, sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    # Title
    fig.suptitle(
        f'Run {run_index + 1}/{total_runs}:  {run_name}'
        + (f'  ({timestamp_str})' if timestamp_str else ''),
        fontsize=12, fontweight='bold', y=0.98
    )
    
    # ── Reward subplot ─────────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(episodes, rewards, linewidth=2, color='#2E86AB', alpha=0.85, label='ep_rew_mean')
    ax0.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax0.set_ylabel('Mean Episode Reward', fontsize=11)
    ax0.grid(True, alpha=0.3, linestyle='--')
    ax0.legend(loc='upper left', fontsize=9)
    
    # Stats annotation
    stats_text = (
        f"final: {rewards[-1]:.2f}\n"
        f"max:  {float(pd.Series(rewards).max()):.2f}\n"
        f"mean: {float(pd.Series(rewards).mean()):.2f}\n"
        f"std:  {float(pd.Series(rewards).std()):.2f}"
    )
    ax0.text(
        0.99, 0.97, stats_text,
        transform=ax0.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )
    
    # ── Episode length subplot ─────────────────────────────────────────────
    if len_col and n_plots == 2:
        ep_lens = plot_df[len_col].dropna().values
        ep_len_episodes = plot_df.dropna(subset=[len_col])[episode_col].values
        ax1 = axes[1]
        ax1.plot(ep_len_episodes, ep_lens, linewidth=2, color='#E84855', alpha=0.85, label='ep_len_mean')
        ax1.set_ylabel('Mean Episode Length (steps)', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=9)
    
    axes[-1].set_xlabel('Episodes', fontsize=11)
    
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


def inspect_runs(runs_dir: Path, min_rows: int = 10, reverse: bool = True, list_only: bool = False):
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
        list_only=args.list_only
    )


if __name__ == '__main__':
    main()
