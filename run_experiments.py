"""
Run experiments and compare different agents
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json
import time
import os

from config import SoccerEnvironmentConfig, TrainingConfig, MADDPGConfig
from trainers import IndependentLearningTrainer, MADDPGTrainer, create_trainer

def run_baseline_experiment(num_episodes: int = 1000):
    """Run baseline experiment with random agents"""
    print("Running baseline experiment with random agents...")

    env_config = SoccerEnvironmentConfig()
    training_config = TrainingConfig()

    trainer = create_trainer(
        "independent",
        env_config,
        training_config,
        agent_type="random"
    )

    results = trainer.train(num_episodes)

    # Evaluate performance
    eval_results = trainer.evaluate(num_episodes=50)

    return {
        'agent_type': 'random',
        'training_results': results,
        'evaluation': eval_results
    }

def run_dqn_experiment(num_episodes: int = 2000):
    """Run DQN independent learning experiment"""
    print("Running DQN independent learning experiment...")

    env_config = SoccerEnvironmentConfig()
    training_config = TrainingConfig()

    dqn_config = {
        'hidden_dims': (256, 128),
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'buffer_size': 10000,
        'batch_size': 64
    }

    trainer = create_trainer(
        "independent",
        env_config,
        training_config,
        agent_type="dqn",
        agent_configs=dqn_config
    )

    results = trainer.train(num_episodes)

    # Evaluate performance
    eval_results = trainer.evaluate(num_episodes=50)

    return {
        'agent_type': 'dqn',
        'training_results': results,
        'evaluation': eval_results
    }

def run_maddpg_experiment(num_episodes: int = 2000):
    """Run MADDPG experiment"""
    print("Running MADDPG experiment...")

    env_config = SoccerEnvironmentConfig()
    training_config = TrainingConfig()
    maddpg_config = MADDPGConfig()

    trainer = create_trainer(
        "maddpg",
        env_config,
        training_config,
        maddpg_config=maddpg_config
    )

    results = trainer.train(num_episodes)

    # Evaluate performance
    eval_results = trainer.evaluate(num_episodes=50)

    return {
        'agent_type': 'maddpg',
        'training_results': results,
        'evaluation': eval_results
    }

def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and visualize experimental results"""
    print("Analyzing experimental results...")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Agent Soccer Training Results Comparison', fontsize=16)

    # Learning curves
    ax = axes[0, 0]
    for result in results:
        agent_type = result['agent_type']
        rewards = result['training_results']['episode_rewards']

        # Smooth the curve using moving average
        window = 100
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=f'{agent_type.upper()}', alpha=0.8)
        else:
            ax.plot(rewards, label=f'{agent_type.upper()}', alpha=0.8)

    ax.set_title('Learning Curves (Episode Rewards)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode lengths
    ax = axes[0, 1]
    for result in results:
        agent_type = result['agent_type']
        lengths = result['training_results']['episode_lengths']

        # Smooth the curve
        window = 100
        if len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=f'{agent_type.upper()}', alpha=0.8)
        else:
            ax.plot(lengths, label=f'{agent_type.upper()}', alpha=0.8)

    ax.set_title('Episode Lengths')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Win rates comparison
    ax = axes[1, 0]
    agent_types = []
    team_0_win_rates = []
    team_1_win_rates = []

    for result in results:
        agent_types.append(result['agent_type'].upper())
        eval_results = result['evaluation']
        team_0_win_rates.append(eval_results['win_rate_team_0'])
        team_1_win_rates.append(eval_results['win_rate_team_1'])

    x = np.arange(len(agent_types))
    width = 0.35

    ax.bar(x - width/2, team_0_win_rates, width, label='Team 0 (Blue)', alpha=0.8)
    ax.bar(x + width/2, team_1_win_rates, width, label='Team 1 (Red)', alpha=0.8)

    ax.set_title('Team Win Rates (Evaluation)')
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Win Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Performance summary
    ax = axes[1, 1]
    metrics = ['avg_reward', 'avg_length', 'team_0_avg_score', 'team_1_avg_score']
    metric_names = ['Avg Reward', 'Avg Length', 'Team 0 Score', 'Team 1 Score']

    data = []
    for result in results:
        eval_results = result['evaluation']
        data.append([eval_results[metric] for metric in metrics])

    data = np.array(data)

    # Normalize data for better visualization
    data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

    im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto')

    # Add text annotations
    for i in range(len(agent_types)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Performance Summary (Normalized)')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_names, rotation=45)
    ax.set_yticks(range(len(agent_types)))
    ax.set_yticklabels(agent_types)

    plt.tight_layout()
    plt.savefig('/Users/ohtaniryouichi/DRL/experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)

    for result in results:
        agent_type = result['agent_type']
        eval_results = result['evaluation']
        print(f"\n{agent_type.upper()} Agent Results:")
        print(f"  Average Reward: {eval_results['avg_reward']:.2f}")
        print(f"  Average Episode Length: {eval_results['avg_length']:.1f}")
        print(f"  Team 0 Win Rate: {eval_results['win_rate_team_0']:.2%}")
        print(f"  Team 1 Win Rate: {eval_results['win_rate_team_1']:.2%}")
        print(f"  Team 0 Avg Score: {eval_results['team_0_avg_score']:.2f}")
        print(f"  Team 1 Avg Score: {eval_results['team_1_avg_score']:.2f}")

def save_results(results: List[Dict[str, Any]], filename: str = "experiment_results.json"):
    """Save results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = {
            'agent_type': result['agent_type'],
            'evaluation': result['evaluation'],
            'training_summary': {
                'total_episodes': len(result['training_results']['episode_rewards']),
                'final_avg_reward': np.mean(result['training_results']['episode_rewards'][-100:])
                                   if len(result['training_results']['episode_rewards']) >= 100
                                   else np.mean(result['training_results']['episode_rewards']),
                'final_avg_length': np.mean(result['training_results']['episode_lengths'][-100:])
                                   if len(result['training_results']['episode_lengths']) >= 100
                                   else np.mean(result['training_results']['episode_lengths'])
            }
        }
        serializable_results.append(serializable_result)

    filepath = f"/Users/ohtaniryouichi/DRL/{filename}"
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {filepath}")

def run_quick_comparison():
    """Run a quick comparison of different agents with fewer episodes"""
    print("Running quick comparison experiment...")

    experiments = [
        ("Random Baseline", lambda: run_baseline_experiment(num_episodes=500)),
        ("DQN Independent", lambda: run_dqn_experiment(num_episodes=1000))
    ]

    results = []

    for name, experiment_func in experiments:
        print(f"\n{'='*20} {name} {'='*20}")
        start_time = time.time()

        try:
            result = experiment_func()
            elapsed_time = time.time() - start_time
            result['training_time'] = elapsed_time
            results.append(result)
            print(f"Completed in {elapsed_time:.1f} seconds")
        except Exception as e:
            print(f"Failed: {e}")
            continue

    if len(results) >= 2:
        analyze_results(results)
        save_results(results, "quick_comparison_results.json")
    else:
        print("Not enough successful experiments for comparison.")

    return results

def run_full_comparison():
    """Run full comparison of all implemented agents"""
    print("Running full comparison experiment...")

    experiments = [
        ("Random Baseline", lambda: run_baseline_experiment(num_episodes=1000)),
        ("DQN Independent", lambda: run_dqn_experiment(num_episodes=3000)),
        ("MADDPG", lambda: run_maddpg_experiment(num_episodes=3000))
    ]

    results = []

    for name, experiment_func in experiments:
        print(f"\n{'='*20} {name} {'='*20}")
        start_time = time.time()

        try:
            result = experiment_func()
            elapsed_time = time.time() - start_time
            result['training_time'] = elapsed_time
            results.append(result)
            print(f"Completed in {elapsed_time:.1f} seconds")
        except Exception as e:
            print(f"Failed: {e}")
            continue

    if len(results) >= 2:
        analyze_results(results)
        save_results(results, "full_comparison_results.json")
    else:
        print("Not enough successful experiments for comparison.")

    return results

if __name__ == "__main__":
    # Run quick comparison by default
    results = run_quick_comparison()
    print("\nQuick comparison completed!")