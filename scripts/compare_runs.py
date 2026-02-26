"""
Compare multiple guitaRL training runs and produce a side-by-side summary table
and learning-curve plot.  Useful for reporting ablation study results.

Usage:
    python scripts/compare_runs.py \\
        --runs runs/full_run runs/no_filtration_run runs/no_audio_run \\
        --labels "Full" "No Filtration" "No Audio" \\
        --output ablation_comparison.png

    # Two runs, auto-labelled from directory name
    python scripts/compare_runs.py --runs runs/run_A runs/run_B
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_progress_csv(run_dir: Path) -> pd.DataFrame:
    """Load SB3 progress.csv from a run directory."""
    csv_path = run_dir / "logs" / "progress.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"progress.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_eval_json(run_dir: Path) -> dict:
    """Load results JSON written by evaluate.py, if present."""
    for pattern in ("eval_results/results_*.json", "results_*.json"):
        matches = list(run_dir.glob(pattern))
        if matches:
            with open(matches[0]) as f:
                return json.load(f)
    return {}


def summarise_run(df: pd.DataFrame, eval_data: dict, label: str) -> dict:
    """Compute scalar summary statistics from a run's CSV + optional eval JSON."""
    summary = {'label': label}

    # ── From progress.csv ────────────────────────────────────────────
    ts_col = 'time/total_timesteps'
    rew_col = 'rollout/ep_rew_mean'
    eval_col = 'eval/mean_reward'

    if ts_col not in df.columns:
        # SB3 sometimes uses different column names
        ts_col = next((c for c in df.columns if 'timestep' in c.lower()), None)

    if rew_col in df.columns and ts_col and ts_col in df.columns:
        df_clean = df.dropna(subset=[rew_col])
        if not df_clean.empty:
            summary['final_ep_rew_mean'] = float(df_clean[rew_col].iloc[-10:].mean())
            summary['peak_ep_rew_mean'] = float(df_clean[rew_col].max())
            summary['total_timesteps'] = int(df_clean[ts_col].iloc[-1])

    if eval_col in df.columns:
        df_eval_clean = df.dropna(subset=[eval_col])
        if not df_eval_clean.empty:
            summary['best_eval_reward'] = float(df_eval_clean[eval_col].max())
            summary['final_eval_reward'] = float(df_eval_clean[eval_col].iloc[-5:].mean())

    # ── Sample efficiency: timesteps to first eval reward above threshold ──
    SAMPLE_EFF_THRESHOLD = 0.5  # mean eval reward > this
    if eval_col in df.columns and ts_col and ts_col in df.columns:
        above = df.dropna(subset=[eval_col, ts_col])
        above = above[above[eval_col] > SAMPLE_EFF_THRESHOLD]
        if not above.empty:
            summary['steps_to_threshold'] = int(above[ts_col].iloc[0])
        else:
            summary['steps_to_threshold'] = None

    # ── From evaluate.py JSON ────────────────────────────────────────
    if eval_data:
        summary['eval_success_rate'] = eval_data.get('success_rate')
        summary['eval_mean_harmonic_prob'] = eval_data.get('mean_harmonic_prob')
        summary['eval_mean_reward'] = eval_data.get('mean_reward')
        summary['eval_n_episodes'] = eval_data.get('n_episodes')

    return summary


def print_table(summaries: list):
    """Print a markdown-formatted comparison table."""
    keys = [
        ('total_timesteps',       'Total timesteps'),
        ('final_ep_rew_mean',     'Final ep_rew_mean (last 10)'),
        ('peak_ep_rew_mean',      'Peak ep_rew_mean'),
        ('best_eval_reward',      'Best eval reward'),
        ('steps_to_threshold',    'Steps to eval_reward > 0.5'),
        ('eval_success_rate',     'Eval success rate'),
        ('eval_mean_harmonic_prob', 'Eval mean H-prob'),
    ]

    labels = [s['label'] for s in summaries]
    col_w = max(30, max(len(l) for l in labels) + 2)
    header_row = f"{'Metric':<40}" + "".join(f"{l:>{col_w}}" for l in labels)
    sep = "-" * len(header_row)

    print("\n" + sep)
    print(header_row)
    print(sep)
    for key, display in keys:
        vals = []
        for s in summaries:
            v = s.get(key)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        row = f"{display:<40}" + "".join(f"{v:>{col_w}}" for v in vals)
        print(row)
    print(sep + "\n")


def plot_learning_curves(run_dfs: list, labels: list, output_path: Path):
    """Plot rollout/ep_rew_mean learning curves for all runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ts_col = 'time/total_timesteps'
    rew_col = 'rollout/ep_rew_mean'
    eval_col = 'eval/mean_reward'

    for i, (df, label) in enumerate(zip(run_dfs, labels)):
        c = colors[i % len(colors)]

        # Episode reward
        ax = axes[0]
        if rew_col in df.columns:
            df_plot = df.dropna(subset=[rew_col])
            if ts_col in df_plot.columns:
                ax.plot(df_plot[ts_col], df_plot[rew_col], label=label, color=c, alpha=0.8)
            else:
                ax.plot(df_plot[rew_col], label=label, color=c, alpha=0.8)

        # Eval reward
        ax2 = axes[1]
        if eval_col in df.columns:
            df_eval = df.dropna(subset=[eval_col])
            if ts_col in df_eval.columns:
                ax2.plot(df_eval[ts_col], df_eval[eval_col], label=label, color=c,
                         marker='o', markersize=4, linestyle='-')
            else:
                ax2.plot(df_eval[eval_col], label=label, color=c, marker='o', markersize=4)

    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Episode Reward (mean)')
    axes[0].set_title('Training Reward (rollout/ep_rew_mean)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Eval Mean Reward')
    axes[1].set_title('Evaluation Reward (EvalCallback)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

    plt.suptitle('Run Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Compare multiple guitaRL training runs')
    parser.add_argument('--runs', nargs='+', required=True, metavar='RUN_DIR',
                        help='Paths to run directories (each must contain logs/progress.csv)')
    parser.add_argument('--labels', nargs='+', default=None, metavar='LABEL',
                        help='Display labels for each run (default: directory name)')
    parser.add_argument('--output', type=str, default=None, metavar='PNG',
                        help='Path to save the comparison plot PNG')
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    labels = args.labels if args.labels else [d.name for d in run_dirs]

    if len(labels) != len(run_dirs):
        print(f"Error: --labels must have the same count as --runs "
              f"({len(args.runs)} runs, {len(labels)} labels)")
        sys.exit(1)

    # Load data
    run_dfs = []
    summaries = []
    for run_dir, label in zip(run_dirs, labels):
        if not run_dir.exists():
            print(f"Warning: run directory not found: {run_dir}")
            run_dfs.append(pd.DataFrame())
            summaries.append({'label': label})
            continue

        try:
            df = load_progress_csv(run_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            df = pd.DataFrame()

        eval_data = load_eval_json(run_dir)
        summary = summarise_run(df, eval_data, label)
        run_dfs.append(df)
        summaries.append(summary)
        print(f"Loaded: {run_dir}  ({len(df)} rows)")

    # Print comparison table
    print_table(summaries)

    # Save table as CSV
    table_rows = []
    for s in summaries:
        table_rows.append(s)
    summary_df = pd.DataFrame(table_rows).set_index('label')
    print("Summary DataFrame:")
    print(summary_df.to_string())

    # Plot
    output_path = Path(args.output) if args.output else None
    valid_dfs = [(df, lbl) for df, lbl in zip(run_dfs, labels) if not df.empty]
    if valid_dfs:
        plot_learning_curves(
            [df for df, _ in valid_dfs],
            [lbl for _, lbl in valid_dfs],
            output_path,
        )
    else:
        print("No valid run data to plot.")


if __name__ == '__main__':
    main()
