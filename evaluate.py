"""
Evaluation script for trained harmonic RL agent.

Usage:
    python evaluate.py --model runs/harmonic_sac_20250101/best_model/best_model.zip --episodes 20
    python evaluate.py --model runs/harmonic_sac_20250101/best_model/best_model.zip --target-fret 7 --visualize
"""

import argparse
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import sys

from stable_baselines3 import SAC
from env.harmonic_env import HarmonicEnv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_policy(model, env, n_episodes=10, deterministic=True, render=False):
    """
    Evaluate trained policy.
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_successes = []
    episode_positions = []
    episode_forces = []
    episode_steps = []
    episode_harmonic_probs = []
    
    all_trajectories = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        trajectory = {
            'target_fret': info['target_fret'],
            'positions': [],
            'forces': [],
            'rewards': [],
            'harmonic_probs': [],
        }
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Record trajectory
            trajectory['positions'].append(info['position_mm'])
            trajectory['forces'].append(info['force'])
            trajectory['rewards'].append(reward)
            
            if info['classification'] is not None:
                harmonic_prob = info['classification']['harmonic_prob']
                trajectory['harmonic_probs'].append(harmonic_prob)
            
            if render:
                env.render()
        
        # Episode metrics
        final_classification = info['classification']
        if final_classification is not None:
            final_harmonic_prob = final_classification['harmonic_prob']
            success = final_harmonic_prob > 0.8
        else:
            final_harmonic_prob = 0.0
            success = False
        
        episode_rewards.append(episode_reward)
        episode_successes.append(float(success))
        episode_steps.append(step_count)
        episode_harmonic_probs.append(final_harmonic_prob)
        episode_positions.append(np.mean(trajectory['positions']))
        episode_forces.append(np.mean(trajectory['forces']))
        
        all_trajectories.append(trajectory)
        
        logger.info(
            f"Episode {episode + 1}/{n_episodes}: "
            f"Fret={info['target_fret']}, "
            f"Reward={episode_reward:.3f}, "
            f"Success={success}, "
            f"Harmonic prob={final_harmonic_prob:.3f}, "
            f"Steps={step_count}"
        )
    
    # Aggregate metrics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean(episode_successes),
        'mean_steps': np.mean(episode_steps),
        'mean_harmonic_prob': np.mean(episode_harmonic_probs),
        'mean_position': np.mean(episode_positions),
        'mean_force': np.mean(episode_forces),
        'episode_rewards': episode_rewards,
        'episode_successes': episode_successes,
        'trajectories': all_trajectories,
    }
    
    return results


def visualize_results(results: Dict, output_path: Path = None):
    """Visualize evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Rewards over episodes
    ax = axes[0, 0]
    ax.plot(results['episode_rewards'], marker='o')
    ax.axhline(y=np.mean(results['episode_rewards']), color='r', linestyle='--', label='Mean')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success rate
    ax = axes[0, 1]
    window_size = 5
    successes = np.array(results['episode_successes'])
    if len(successes) >= window_size:
        smoothed = np.convolve(successes, np.ones(window_size)/window_size, mode='valid')
        ax.plot(range(window_size-1, len(successes)), smoothed, marker='o')
    else:
        ax.plot(successes, marker='o')
    ax.axhline(y=results['success_rate'], color='r', linestyle='--', label=f'Mean: {results["success_rate"]:.3f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success (5-ep moving avg)')
    ax.set_title('Success Rate Over Episodes')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Position/Force trajectories (first 3 episodes)
    ax = axes[1, 0]
    for i, traj in enumerate(results['trajectories'][:3]):
        steps = range(len(traj['positions']))
        ax.plot(steps, traj['positions'], marker='o', label=f"Ep {i+1} (Fret {traj['target_fret']})")
        
        # Mark target position
        target_positions = {4: 112.0, 5: 139.0, 7: 187.0}
        target_pos = target_positions.get(traj['target_fret'], 0)
        ax.axhline(y=target_pos, linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Position (mm)')
    ax.set_title('Position Trajectories (First 3 Episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Harmonic probability over steps
    ax = axes[1, 1]
    for i, traj in enumerate(results['trajectories'][:3]):
        if traj['harmonic_probs']:
            steps = range(len(traj['harmonic_probs']))
            ax.plot(steps, traj['harmonic_probs'], marker='o', label=f"Ep {i+1} (Fret {traj['target_fret']})")
    
    ax.axhline(y=0.8, color='r', linestyle='--', label='Success threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Harmonic Probability')
    ax.set_title('Harmonic Quality Over Steps (First 3 Episodes)')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved visualization to {output_path}")
    
    plt.show()


def print_summary(results: Dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mean Reward:         {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Success Rate:        {results['success_rate']:.1%}")
    print(f"Mean Harmonic Prob:  {results['mean_harmonic_prob']:.3f}")
    print(f"Mean Steps/Episode:  {results['mean_steps']:.1f}")
    print(f"Mean Position:       {results['mean_position']:.1f} mm")
    print(f"Mean Force:          {results['mean_force']:.3f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate harmonic RL agent')
    parser.add_argument('--model', required=True, help='Path to trained model (.zip)')
    parser.add_argument('--model-classifier', default='../HarmonicsClassifier/models/best_model.pt',
                        help='Path to HarmonicsClassifier model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--string-index', type=int, default=3, help='String to evaluate on')
    parser.add_argument('--target-fret', type=int, choices=[4, 5, 7], default=None,
                        help='Specific fret to test (default: random)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--output-dir', type=str, default='./eval_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Check paths
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    classifier_path = Path(args.model_classifier)
    if not classifier_path.exists():
        logger.error(f"Classifier model not found: {classifier_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = SAC.load(model_path)
    
    # Create environment
    curriculum_mode = 'random' if args.target_fret is None else 'fixed_fret'
    env = HarmonicEnv(
        model_path=str(classifier_path),
        string_index=args.string_index,
        curriculum_mode=curriculum_mode,
        max_steps=10
    )
    
    # If specific fret requested, override
    if args.target_fret is not None:
        env.HARMONIC_FRETS = [args.target_fret]
    
    # Evaluate
    logger.info(f"Evaluating for {args.episodes} episodes...")
    results = evaluate_policy(
        model,
        env,
        n_episodes=args.episodes,
        deterministic=args.deterministic
    )
    
    # Print summary
    print_summary(results)
    
    # Visualize
    if args.visualize:
        viz_path = output_dir / f"evaluation_{model_path.stem}.png"
        visualize_results(results, output_path=viz_path)
    
    # Save results
    import json
    results_path = output_dir / f"results_{model_path.stem}.json"
    
    # Convert numpy types for JSON serialization
    json_results = {
        k: (v.tolist() if isinstance(v, np.ndarray) else 
            float(v) if isinstance(v, (np.float32, np.float64)) else 
            v)
        for k, v in results.items()
        if k != 'trajectories'  # Skip trajectories for JSON
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    env.close()


if __name__ == '__main__':
    main()
