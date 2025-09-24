"""
Agent implementations for soccer environment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import random

from config import SoccerEnvironmentConfig, MADDPGConfig

class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, agent_id: int, action_space_size: int):
        self.agent_id = agent_id
        self.action_space_size = action_space_size

    @abstractmethod
    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action based on observation"""
        pass

    @abstractmethod
    def learn(self, experiences: List) -> Dict[str, float]:
        """Learn from experiences"""
        pass

    def save(self, filepath: str):
        """Save agent model"""
        pass

    def load(self, filepath: str):
        """Load agent model"""
        pass

class RandomAgent(BaseAgent):
    """Random agent for baseline and testing"""

    def __init__(self, agent_id: int, action_space_size: int = 5,
                 action_type: str = "continuous"):
        super().__init__(agent_id, action_space_size)
        self.action_type = action_type

    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """Select random action"""
        if self.action_type == "continuous":
            # Continuous action: [move_x, move_y, kick_power, kick_dir_x, kick_dir_y]
            action = np.array([
                np.random.uniform(-1, 1),  # move_x
                np.random.uniform(-1, 1),  # move_y
                np.random.uniform(0, 1),   # kick_power
                np.random.uniform(-1, 1),  # kick_dir_x
                np.random.uniform(-1, 1),  # kick_dir_y
            ], dtype=np.float32)
        else:
            # Discrete action
            action = np.random.randint(0, 9)  # 9 possible actions

        return action

    def learn(self, experiences: List) -> Dict[str, float]:
        """Random agent doesn't learn"""
        return {"loss": 0.0}

class MLPNetwork(nn.Module):
    """Multi-layer perceptron network"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DQNAgent(BaseAgent):
    """Deep Q-Network agent"""

    def __init__(self, agent_id: int, obs_dim: int, action_dim: int = 9,
                 hidden_dims: Tuple[int, ...] = (256, 128),
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000,
                 batch_size: int = 64):
        super().__init__(agent_id, action_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = MLPNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = MLPNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Copy weights to target network
        self.update_target_network()

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, experiences: List = None) -> Dict[str, float]:
        """Learn from experiences in replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update epsilon
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

class Actor(nn.Module):
    """Actor network for MADDPG"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.network = MLPNetwork(obs_dim, action_dim, hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(obs))

class Critic(nn.Module):
    """Critic network for MADDPG"""

    def __init__(self, global_obs_dim: int, global_action_dim: int,
                 hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.network = MLPNetwork(global_obs_dim + global_action_dim, 1, hidden_dims)

    def forward(self, global_obs: torch.Tensor, global_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([global_obs, global_actions], dim=1)
        return self.network(x)

class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration"""

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset noise"""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """Sample noise"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class MADDPGAgent(BaseAgent):
    """Multi-Agent Deep Deterministic Policy Gradient agent"""

    def __init__(self, agent_id: int, config: MADDPGConfig):
        super().__init__(agent_id, config.action_dim)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = Actor(config.obs_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.critic = Critic(config.global_obs_dim, config.global_action_dim,
                           config.hidden_dims).to(self.device)
        self.target_actor = Actor(config.obs_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.target_critic = Critic(config.global_obs_dim, config.global_action_dim,
                                  config.hidden_dims).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Noise for exploration
        self.noise = OUNoise(config.action_dim, sigma=config.noise_scale)

    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using actor network"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor).cpu().data.numpy().flatten()

        if training:
            action += self.noise.sample()
            action = np.clip(action, -1, 1)

        return action

    def learn(self, experiences: Dict) -> Dict[str, float]:
        """Learn from experiences (implemented in trainer)"""
        # This will be implemented in the MADDPG trainer
        return {"actor_loss": 0.0, "critic_loss": 0.0}

    def soft_update(self, local_model: nn.Module, target_model: nn.Module):
        """Soft update of target network"""
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(
                self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

def create_agent(agent_type: str, agent_id: int, config: Dict) -> BaseAgent:
    """Factory function to create agents"""
    if agent_type == "random":
        return RandomAgent(agent_id)
    elif agent_type == "dqn":
        return DQNAgent(agent_id, **config)
    elif agent_type == "maddpg":
        return MADDPGAgent(agent_id, **config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")