"""
Training frameworks for multi-agent soccer environment
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import random
import time
from abc import ABC, abstractmethod

from config import SoccerEnvironmentConfig, TrainingConfig, MADDPGConfig
from soccer_env import make_soccer_env
from agents import BaseAgent, RandomAgent, DQNAgent, MADDPGAgent

class ReplayBuffer:
    """Experience replay buffer for multi-agent learning"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Tuple):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch from buffer"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class BaseTrainer(ABC):
    """Base class for all training frameworks"""

    def __init__(self, env_config: SoccerEnvironmentConfig,
                 training_config: TrainingConfig):
        self.env_config = env_config
        self.training_config = training_config
        self.env = make_soccer_env(env_config, render_mode=None)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.scores_history = []
        self.training_metrics = defaultdict(list)

    @abstractmethod
    def train(self, num_episodes: int) -> Dict[str, Any]:
        """Train agents for specified number of episodes"""
        pass

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current agent performance"""
        total_rewards = []
        total_lengths = []
        team_scores = [[], []]

        for episode in range(num_episodes):
            observations = self.env.reset()
            episode_reward = 0
            steps = 0

            while not all(self.env.terminations.values()) and not all(self.env.truncations.values()):
                actions = {}
                for agent in self.env.agents:
                    if not self.env.terminations.get(agent, False) and not self.env.truncations.get(agent, False):
                        obs = self.env.observe(agent)
                        actions[agent] = self._get_agent_action(agent, obs, training=False)

                for agent, action in actions.items():
                    self.env.step(action)
                    episode_reward += self.env.rewards.get(agent, 0)
                    steps += 1

                    if self.env.terminations.get(agent, False) or self.env.truncations.get(agent, False):
                        break

            total_rewards.append(episode_reward)
            total_lengths.append(steps)
            team_scores[0].append(self.env.scores[0])
            team_scores[1].append(self.env.scores[1])

        return {
            'avg_reward': np.mean(total_rewards),
            'avg_length': np.mean(total_lengths),
            'team_0_avg_score': np.mean(team_scores[0]),
            'team_1_avg_score': np.mean(team_scores[1]),
            'win_rate_team_0': sum(1 for i in range(num_episodes) if team_scores[0][i] > team_scores[1][i]) / num_episodes,
            'win_rate_team_1': sum(1 for i in range(num_episodes) if team_scores[1][i] > team_scores[0][i]) / num_episodes,
        }

    @abstractmethod
    def _get_agent_action(self, agent: str, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """Get action from agent"""
        pass

class IndependentLearningTrainer(BaseTrainer):
    """Independent learning trainer where each agent learns separately"""

    def __init__(self, env_config: SoccerEnvironmentConfig,
                 training_config: TrainingConfig,
                 agent_type: str = "dqn",
                 agent_configs: Dict = None):
        super().__init__(env_config, training_config)

        self.agent_type = agent_type
        self.agent_configs = agent_configs or {}

        # Create agents
        self.agents = {}
        for i, agent_name in enumerate(self.env.agents):
            if agent_type == "dqn":
                self.agents[agent_name] = DQNAgent(
                    agent_id=i,
                    obs_dim=28,  # From observation space
                    **self.agent_configs
                )
            elif agent_type == "random":
                self.agents[agent_name] = RandomAgent(i)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

    def train(self, num_episodes: int) -> Dict[str, Any]:
        """Train agents independently"""
        print(f"Starting independent learning training with {self.agent_type} agents")

        for episode in range(num_episodes):
            observations = self.env.reset()
            episode_rewards = {agent: 0 for agent in self.env.agents}
            episode_length = 0

            # Store previous observations for experience replay
            prev_observations = {}

            while not all(self.env.terminations.values()) and not all(self.env.truncations.values()):
                actions = {}

                # Get actions from all agents
                for agent in self.env.agents:
                    if not self.env.terminations.get(agent, False) and not self.env.truncations.get(agent, False):
                        obs = self.env.observe(agent)
                        action = self.agents[agent].select_action(obs, training=True)
                        actions[agent] = action
                        prev_observations[agent] = obs

                # Execute actions
                for agent, action in actions.items():
                    self.env.step(action)
                    reward = self.env.rewards.get(agent, 0)
                    episode_rewards[agent] += reward
                    episode_length += 1

                    # Store experience for DQN agents
                    if self.agent_type == "dqn" and agent in prev_observations:
                        next_obs = self.env.observe(agent)
                        done = self.env.terminations.get(agent, False) or self.env.truncations.get(agent, False)

                        if isinstance(action, np.ndarray):
                            action = int(action[0]) if len(action) > 0 else 0

                        self.agents[agent].store_experience(
                            prev_observations[agent], action, reward, next_obs, done
                        )

                        # Learn from experience
                        metrics = self.agents[agent].learn()
                        if metrics and metrics['loss'] > 0:
                            self.training_metrics[f'{agent}_loss'].append(metrics['loss'])

                    if self.env.terminations.get(agent, False) or self.env.truncations.get(agent, False):
                        break

            # Update target networks for DQN agents
            if self.agent_type == "dqn" and episode % 100 == 0:
                for agent_name, agent in self.agents.items():
                    agent.update_target_network()

            # Record episode statistics
            total_reward = sum(episode_rewards.values())
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            self.scores_history.append(self.env.scores.copy())

            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                print(f"Episode {episode}: Avg Reward (last 100): {avg_reward:.2f}, Scores: {self.env.scores}")

                # Evaluate current performance
                if episode % 500 == 0 and episode > 0:
                    eval_metrics = self.evaluate(num_episodes=10)
                    print(f"Evaluation: {eval_metrics}")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'scores_history': self.scores_history,
            'training_metrics': dict(self.training_metrics)
        }

    def _get_agent_action(self, agent: str, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """Get action from specific agent"""
        if self.agent_type == "dqn":
            discrete_action = self.agents[agent].select_action(observation, training)
            # Convert discrete action to continuous for environment
            from spaces import ActionSpace
            action_space = ActionSpace("discrete")
            return action_space.convert_discrete_to_continuous(discrete_action)
        else:
            return self.agents[agent].select_action(observation, training)

class MADDPGTrainer(BaseTrainer):
    """MADDPG trainer with centralized critic"""

    def __init__(self, env_config: SoccerEnvironmentConfig,
                 training_config: TrainingConfig,
                 maddpg_config: MADDPGConfig):
        super().__init__(env_config, training_config)
        self.maddpg_config = maddpg_config

        # Create MADDPG agents
        self.agents = {}
        for i, agent_name in enumerate(self.env.agents):
            self.agents[agent_name] = MADDPGAgent(i, maddpg_config)

        # Shared replay buffer
        self.replay_buffer = ReplayBuffer(maddpg_config.buffer_size)

    def train(self, num_episodes: int) -> Dict[str, Any]:
        """Train MADDPG agents"""
        print("Starting MADDPG training")

        for episode in range(num_episodes):
            observations = self.env.reset()
            episode_rewards = {agent: 0 for agent in self.env.agents}
            episode_length = 0

            # Episode experience
            episode_experiences = []

            while not all(self.env.terminations.values()) and not all(self.env.truncations.values()):
                # Get global observation and actions
                global_obs = []
                actions = {}

                for agent in self.env.agents:
                    if not self.env.terminations.get(agent, False) and not self.env.truncations.get(agent, False):
                        obs = self.env.observe(agent)
                        action = self.agents[agent].select_action(obs, training=True)
                        actions[agent] = action
                        global_obs.append(obs)

                # Execute actions and collect rewards
                global_actions = list(actions.values())
                step_experience = {
                    'global_obs': np.concatenate(global_obs),
                    'actions': actions.copy(),
                    'global_actions': np.concatenate(global_actions),
                    'rewards': {},
                    'next_global_obs': None,
                    'dones': {}
                }

                for agent, action in actions.items():
                    self.env.step(action)
                    reward = self.env.rewards.get(agent, 0)
                    episode_rewards[agent] += reward
                    step_experience['rewards'][agent] = reward
                    step_experience['dones'][agent] = self.env.terminations.get(agent, False) or self.env.truncations.get(agent, False)
                    episode_length += 1

                    if step_experience['dones'][agent]:
                        break

                # Get next global observation
                next_global_obs = []
                for agent in self.env.agents:
                    next_obs = self.env.observe(agent)
                    next_global_obs.append(next_obs)
                step_experience['next_global_obs'] = np.concatenate(next_global_obs)

                episode_experiences.append(step_experience)

            # Store experiences in replay buffer
            for exp in episode_experiences:
                self.replay_buffer.push(exp)

            # Train agents if enough experiences
            if len(self.replay_buffer) > self.maddpg_config.batch_size:
                self._train_maddpg_step()

            # Record statistics
            total_reward = sum(episode_rewards.values())
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            self.scores_history.append(self.env.scores.copy())

            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, Scores: {self.env.scores}")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'scores_history': self.scores_history,
            'training_metrics': dict(self.training_metrics)
        }

    def _train_maddpg_step(self):
        """Perform one MADDPG training step"""
        batch = self.replay_buffer.sample(self.maddpg_config.batch_size)

        for i, (agent_name, agent) in enumerate(self.agents.items()):
            # Extract data for this agent
            states = torch.FloatTensor([exp['global_obs'] for exp in batch]).to(agent.device)
            actions = torch.FloatTensor([exp['global_actions'] for exp in batch]).to(agent.device)
            rewards = torch.FloatTensor([exp['rewards'][agent_name] for exp in batch]).to(agent.device)
            next_states = torch.FloatTensor([exp['next_global_obs'] for exp in batch]).to(agent.device)
            dones = torch.BoolTensor([exp['dones'][agent_name] for exp in batch]).to(agent.device)

            # Get agent-specific observations
            agent_obs = states[:, i*28:(i+1)*28]  # 28D observation per agent
            next_agent_obs = next_states[:, i*28:(i+1)*28]

            # Update critic
            with torch.no_grad():
                next_actions = torch.cat([
                    self.agents[list(self.agents.keys())[j]].target_actor(next_states[:, j*28:(j+1)*28])
                    for j in range(len(self.agents))
                ], dim=1)
                target_q = agent.target_critic(next_states, next_actions)
                target_q = rewards + (self.maddpg_config.gamma * target_q * ~dones)

            current_q = agent.critic(states, actions)
            critic_loss = F.mse_loss(current_q.squeeze(), target_q.squeeze())

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)
            agent.critic_optimizer.step()

            # Update actor
            agent_actions = agent.actor(agent_obs)
            full_actions = actions.clone()
            full_actions[:, i*5:(i+1)*5] = agent_actions  # 5D action per agent

            actor_loss = -agent.critic(states, full_actions).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.5)
            agent.actor_optimizer.step()

            # Soft update target networks
            agent.soft_update(agent.actor, agent.target_actor)
            agent.soft_update(agent.critic, agent.target_critic)

            # Record metrics
            self.training_metrics[f'{agent_name}_critic_loss'].append(critic_loss.item())
            self.training_metrics[f'{agent_name}_actor_loss'].append(actor_loss.item())

    def _get_agent_action(self, agent: str, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """Get action from MADDPG agent"""
        return self.agents[agent].select_action(observation, training)

def create_trainer(trainer_type: str, env_config: SoccerEnvironmentConfig,
                  training_config: TrainingConfig, **kwargs) -> BaseTrainer:
    """Factory function to create trainers"""
    if trainer_type == "independent":
        return IndependentLearningTrainer(env_config, training_config, **kwargs)
    elif trainer_type == "maddpg":
        maddpg_config = kwargs.get('maddpg_config', MADDPGConfig())
        return MADDPGTrainer(env_config, training_config, maddpg_config)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")