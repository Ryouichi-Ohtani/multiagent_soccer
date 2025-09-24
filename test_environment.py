"""
Test script for soccer environment with random agents
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import time

from config import SoccerEnvironmentConfig, TrainingConfig
from soccer_env import make_soccer_env
from agents import RandomAgent

def test_basic_environment():
    """Test basic environment functionality"""
    print("Testing basic environment functionality...")

    # Create environment
    config = SoccerEnvironmentConfig()
    env = make_soccer_env(config, render_mode=None, action_type="continuous")

    print(f"Environment created with {len(env.agents)} agents")
    print(f"Agents: {env.agents}")
    print(f"Observation space: {env.observation_spaces[env.agents[0]]}")
    print(f"Action space: {env.action_spaces[env.agents[0]]}")

    # Test reset
    observations = env.reset()
    print(f"Reset successful, observations shape: {[obs.shape for obs in observations.values()]}")

    # Test step
    for agent in env.agents:
        action = env.action_spaces[agent].sample()
        print(f"Agent {agent} taking action: {action}")
        env.step(action)

    print("Basic environment test completed successfully!")
    return True

def test_random_agents_episode():
    """Test full episode with random agents"""
    print("Testing full episode with random agents...")

    # Create environment and agents
    config = SoccerEnvironmentConfig()
    env = make_soccer_env(config, render_mode=None, action_type="continuous")

    # Create random agents
    agents = {}
    for i, agent_name in enumerate(env.agents):
        agents[agent_name] = RandomAgent(i, action_space_size=5, action_type="continuous")

    # Run episode
    observations = env.reset()
    episode_rewards = {agent: 0 for agent in env.agents}
    episode_length = 0

    print("Running episode...")
    start_time = time.time()

    while not all(env.terminations.values()) and not all(env.truncations.values()):
        for agent in env.agents:
            if not env.terminations.get(agent, False) and not env.truncations.get(agent, False):
                # Get action from agent
                obs = env.observe(agent)
                action = agents[agent].select_action(obs, training=False)

                # Take step
                env.step(action)

                # Accumulate reward
                episode_rewards[agent] += env.rewards.get(agent, 0)

                episode_length += 1

                # Break if episode is done
                if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                    break

    elapsed_time = time.time() - start_time

    print(f"Episode completed in {elapsed_time:.2f} seconds")
    print(f"Episode length: {episode_length} steps")
    print(f"Final scores: {env.scores}")
    print(f"Episode rewards: {episode_rewards}")

    env.close()
    return True

def test_multiple_episodes(num_episodes: int = 5):
    """Test multiple episodes and collect statistics"""
    print(f"Testing {num_episodes} episodes for performance analysis...")

    config = SoccerEnvironmentConfig()
    env = make_soccer_env(config, render_mode=None, action_type="continuous")

    # Create random agents
    agents = {}
    for i, agent_name in enumerate(env.agents):
        agents[agent_name] = RandomAgent(i, action_space_size=5, action_type="continuous")

    # Statistics
    episode_lengths = []
    episode_rewards = []
    final_scores = []

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        observations = env.reset()
        episode_reward = {agent: 0 for agent in env.agents}
        steps = 0

        while not all(env.terminations.values()) and not all(env.truncations.values()):
            for agent in env.agents:
                if not env.terminations.get(agent, False) and not env.truncations.get(agent, False):
                    obs = env.observe(agent)
                    action = agents[agent].select_action(obs, training=False)
                    env.step(action)
                    episode_reward[agent] += env.rewards.get(agent, 0)
                    steps += 1

                    if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                        break

        episode_lengths.append(steps)
        episode_rewards.append(episode_reward)
        final_scores.append(env.scores.copy())

        print(f"  Steps: {steps}, Scores: {env.scores}, Avg Reward: {np.mean(list(episode_reward.values())):.2f}")

    # Print statistics
    print(f"\n=== Statistics over {num_episodes} episodes ===")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")

    # Team scores
    team_0_scores = [score[0] for score in final_scores]
    team_1_scores = [score[1] for score in final_scores]

    print(f"Team 0 (Blue) average score: {np.mean(team_0_scores):.2f} ± {np.std(team_0_scores):.2f}")
    print(f"Team 1 (Red) average score: {np.mean(team_1_scores):.2f} ± {np.std(team_1_scores):.2f}")

    # Average rewards per agent
    for agent in env.agents:
        agent_rewards = [ep_reward[agent] for ep_reward in episode_rewards]
        print(f"{agent} average reward: {np.mean(agent_rewards):.2f} ± {np.std(agent_rewards):.2f}")

    env.close()
    return True

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Multi-Agent Soccer Environment Test Suite")
    print("=" * 60)

    tests = [
        ("Basic Environment", test_basic_environment),
        ("Random Agents Episode", test_random_agents_episode),
        ("Multiple Episodes", lambda: test_multiple_episodes(3))
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append(result)
            print(f"✓ {test_name} PASSED")
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for i, (test_name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_name}: {status}")

    success_rate = sum(results) / len(results)
    print(f"\nSuccess Rate: {success_rate:.1%} ({sum(results)}/{len(results)})")

    return success_rate == 1.0

if __name__ == "__main__":
    run_all_tests()