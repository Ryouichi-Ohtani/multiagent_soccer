"""
Main Soccer Environment - PettingZoo compatible multi-agent environment
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from config import SoccerEnvironmentConfig, TrainingConfig
from physics import PhysicsEngine
from renderer import SoccerRenderer
from spaces import ObservationSpace, ActionSpace
from rewards import RewardCalculator, RewardShaper

class SoccerEnvironment(AECEnv):
    """
    PettingZoo-compatible soccer environment for multi-agent reinforcement learning
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "soccer_v1"
    }

    def __init__(self, config: SoccerEnvironmentConfig = None,
                 training_config: TrainingConfig = None,
                 render_mode: str = None,
                 action_type: str = "continuous"):
        """
        Initialize soccer environment

        Args:
            config: Environment configuration
            training_config: Training configuration
            render_mode: Rendering mode ("human", "rgb_array", or None)
            action_type: Action space type ("continuous" or "discrete")
        """
        super().__init__()

        self.config = config or SoccerEnvironmentConfig()
        self.training_config = training_config or TrainingConfig()
        self.render_mode = render_mode
        self.action_type = action_type

        # Initialize components
        self.physics = PhysicsEngine(self.config)
        self.reward_calculator = RewardCalculator(self.config)
        self.reward_shaper = RewardShaper(self.config)

        # Initialize spaces
        self.observation_space_handler = ObservationSpace(self.config)
        self.action_space_handler = ActionSpace(action_type)

        # Agent setup
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agents = self.possible_agents[:]

        # Observation and action spaces
        self.observation_spaces = {
            agent: self.observation_space_handler.gym_space
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space_handler.gym_space
            for agent in self.possible_agents
        }

        # Agent selector for turn-based execution
        self._agent_selector = agent_selector(self.agents)

        # Initialize renderer if needed
        self.renderer = None
        if self.render_mode == "human":
            self.renderer = SoccerRenderer(self.config)

        # Game state
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)

        # Reset physics
        self.physics.reset()

        # Reset agents
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Reset game state
        self.step_count = 0
        self.scores = [0, 0]  # [team_0, team_1]
        self.episode_terminated = False
        self.episode_truncated = False

        # Reset rewards and info
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Store previous state for reward calculation
        self.prev_state = None
        self.ball_possession = -1  # -1: no possession, 0-3: player id
        self.last_touch = -1

        return self._get_observations()

    def step(self, action: Any):
        """Execute one step in the environment"""
        if self.episode_terminated or self.episode_truncated:
            return self._was_dead_step(action)

        agent_id = int(self.agent_selection.split('_')[1])

        # Store previous state
        if self.prev_state is None:
            self.prev_state = self.physics.get_state()

        # Convert discrete action to continuous if needed
        if self.action_type == "discrete" and isinstance(action, (int, np.integer)):
            action = self.action_space_handler.convert_discrete_to_continuous(action)

        # Execute action in physics
        actions = {self.agent_selection: action}
        goal_result, ball_touched_by = self.physics.step(actions)

        # Update ball possession and last touch
        if ball_touched_by is not None:
            self.ball_possession = ball_touched_by
            self.last_touch = ball_touched_by

        # Handle goal scoring
        if goal_result:
            if goal_result == "goal_left":
                self.scores[1] += 1  # Red team scored
            elif goal_result == "goal_right":
                self.scores[0] += 1  # Blue team scored

        # Get current state
        current_state = self.physics.get_state()

        # Calculate rewards
        self._calculate_rewards(agent_id, action, self.prev_state, current_state,
                              goal_result, ball_touched_by)

        # Update step count
        self.step_count += 1

        # Check termination conditions
        self._check_termination()

        # Move to next agent
        self.agent_selection = self._agent_selector.next()

        # Update previous state
        self.prev_state = current_state

    def _calculate_rewards(self, agent_id: int, action: np.ndarray,
                          prev_state: Dict, current_state: Dict,
                          goal_result: Optional[str], ball_touched_by: Optional[int]):
        """Calculate rewards for all agents"""
        # Reset rewards for this step
        self.rewards = {agent: 0.0 for agent in self.agents}

        # Calculate rewards for each agent
        for i, agent in enumerate(self.agents):
            reward = self.reward_shaper.shaped_reward(
                i, action if i == agent_id else np.zeros(5),
                prev_state, current_state,
                goal_scored=goal_result,
                ball_touched_by=ball_touched_by,
                scores=tuple(self.scores)
            )
            self.rewards[agent] = reward

    def _check_termination(self):
        """Check if episode should terminate"""
        # Game ends if max steps reached
        if self.step_count >= self.config.MAX_STEPS:
            self.episode_truncated = True

        # Game ends if goal difference is too large (optional)
        goal_diff = abs(self.scores[0] - self.scores[1])
        if goal_diff >= 5:  # End early if one team is dominating
            self.episode_terminated = True

        # Update termination/truncation for all agents
        if self.episode_terminated or self.episode_truncated:
            for agent in self.agents:
                self.terminations[agent] = self.episode_terminated
                self.truncations[agent] = self.episode_truncated

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        current_state = self.physics.get_state()
        observations = {}

        for i, agent in enumerate(self.agents):
            obs = self.observation_space_handler.create_observation(
                i, current_state, tuple(self.scores), self.step_count,
                self.config.MAX_STEPS, self.ball_possession, self.last_touch
            )
            observations[agent] = obs

        return observations

    def observe(self, agent: str) -> np.ndarray:
        """Get observation for specific agent"""
        agent_id = int(agent.split('_')[1])
        current_state = self.physics.get_state()

        return self.observation_space_handler.create_observation(
            agent_id, current_state, tuple(self.scores), self.step_count,
            self.config.MAX_STEPS, self.ball_possession, self.last_touch
        )

    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and self.renderer:
            current_state = self.physics.get_state()
            return self.renderer.render(current_state, tuple(self.scores), self.step_count)
        elif self.render_mode == "rgb_array":
            # Return RGB array for recording
            if not self.renderer:
                self.renderer = SoccerRenderer(self.config)
            current_state = self.physics.get_state()
            self.renderer.render(current_state, tuple(self.scores), self.step_count)
            # Convert pygame surface to numpy array
            import pygame
            rgb_array = pygame.surfarray.array3d(self.renderer.screen)
            return np.transpose(rgb_array, (1, 0, 2))

    def close(self):
        """Clean up resources"""
        if self.renderer:
            self.renderer.close()

    def state(self) -> np.ndarray:
        """Get global state (concatenated observations)"""
        observations = self._get_observations()
        return np.concatenate([observations[agent] for agent in self.agents])

    def _was_dead_step(self, action):
        """Handle action taken when episode is over"""
        # This method is required by PettingZoo but not used in our implementation
        pass

# Wrapper functions for easier usage

def make_soccer_env(config: SoccerEnvironmentConfig = None,
                   render_mode: str = None,
                   action_type: str = "continuous") -> SoccerEnvironment:
    """Create soccer environment with default settings"""
    return SoccerEnvironment(config, render_mode=render_mode, action_type=action_type)

def make_parallel_soccer_env(config: SoccerEnvironmentConfig = None,
                            render_mode: str = None,
                            action_type: str = "continuous"):
    """Create parallel version of soccer environment"""
    from pettingzoo.utils import parallel_to_aec
    env = make_soccer_env(config, render_mode, action_type)
    return parallel_to_aec(env)

# Compatibility with stable-baselines3
class SB3SoccerEnv:
    """Stable-Baselines3 compatible wrapper"""
    def __init__(self, config: SoccerEnvironmentConfig = None,
                 action_type: str = "continuous"):
        self.env = make_soccer_env(config, action_type=action_type)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        # For SB3 compatibility
        self.observation_space = self.env.observation_spaces[self.agents[0]]
        self.action_space = self.env.action_spaces[self.agents[0]]

    def reset(self):
        observations = self.env.reset()
        return np.array([observations[agent] for agent in self.agents])

    def step(self, actions):
        # Execute actions for all agents simultaneously
        rewards = []
        done = False
        infos = []

        for i, agent in enumerate(self.agents):
            if not done:
                self.env.step(actions[i])
                rewards.append(self.env.rewards[agent])
                done = self.env.terminations[agent] or self.env.truncations[agent]
                infos.append(self.env.infos[agent])

        obs = [self.env.observe(agent) for agent in self.agents]
        return np.array(obs), np.array(rewards), done, infos