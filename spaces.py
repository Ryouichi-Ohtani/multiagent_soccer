"""
Observation and action space definitions for soccer environment
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Union
from config import SoccerEnvironmentConfig

class ObservationSpace:
    """
    Observation space for each agent (28 dimensions total)
    """
    def __init__(self, config: SoccerEnvironmentConfig):
        self.config = config

        # Observation space bounds
        field_width, field_height = config.FIELD_SIZE
        max_velocity = config.PLAYER_SPEED * 2  # Max possible velocity
        max_distance = np.sqrt(field_width**2 + field_height**2)  # Diagonal distance

        # Define observation bounds
        obs_low = np.array([
            # Self state (4 dims)
            0, 0,           # position (normalized)
            -max_velocity, -max_velocity,  # velocity

            # Ball state (4 dims)
            0, 0,           # position (normalized)
            -max_velocity, -max_velocity,  # velocity

            # Teammate state (4 dims)
            0, 0,           # position (normalized)
            -max_velocity, -max_velocity,  # velocity

            # Opponent 1 state (4 dims)
            0, 0,           # position (normalized)
            -max_velocity, -max_velocity,  # velocity

            # Opponent 2 state (4 dims)
            0, 0,           # position (normalized)
            -max_velocity, -max_velocity,  # velocity

            # Goal information (4 dims)
            0,              # own goal distance
            0,              # enemy goal distance
            -np.pi,         # own goal angle
            -np.pi,         # enemy goal angle

            # Context information (4 dims)
            -1,             # ball possession (-1: none, 0-3: player id)
            0,              # time remaining (normalized)
            -10,            # score difference
            -1,             # last touch player id
        ], dtype=np.float32)

        obs_high = np.array([
            # Self state (4 dims)
            1, 1,           # position (normalized)
            max_velocity, max_velocity,  # velocity

            # Ball state (4 dims)
            1, 1,           # position (normalized)
            max_velocity, max_velocity,  # velocity

            # Teammate state (4 dims)
            1, 1,           # position (normalized)
            max_velocity, max_velocity,  # velocity

            # Opponent 1 state (4 dims)
            1, 1,           # position (normalized)
            max_velocity, max_velocity,  # velocity

            # Opponent 2 state (4 dims)
            1, 1,           # position (normalized)
            max_velocity, max_velocity,  # velocity

            # Goal information (4 dims)
            max_distance,   # own goal distance
            max_distance,   # enemy goal distance
            np.pi,          # own goal angle
            np.pi,          # enemy goal angle

            # Context information (4 dims)
            3,              # ball possession (player 0-3)
            1,              # time remaining (normalized)
            10,             # score difference
            3,              # last touch player id
        ], dtype=np.float32)

        self.gym_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def create_observation(self, agent_id: int, state: Dict,
                         scores: Tuple[int, int], step: int,
                         max_steps: int, ball_possession: int = -1,
                         last_touch: int = -1) -> np.ndarray:
        """Create observation for specific agent"""

        field_width, field_height = self.config.FIELD_SIZE
        players = state['players']
        ball = state['ball']

        # Get agent info
        agent = players[agent_id]
        agent_team = agent['team']
        agent_pos = agent['pos'] / np.array([field_width, field_height])  # Normalize
        agent_vel = agent['vel'] / self.config.PLAYER_SPEED  # Normalize

        # Get ball info
        ball_pos = ball['pos'] / np.array([field_width, field_height])  # Normalize
        ball_vel = ball['vel'] / self.config.PLAYER_SPEED  # Normalize

        # Get teammate and opponents
        teammates = [p for i, p in enumerate(players)
                    if p['team'] == agent_team and i != agent_id]
        opponents = [p for p in players if p['team'] != agent_team]

        teammate = teammates[0] if teammates else agent  # fallback
        teammate_pos = teammate['pos'] / np.array([field_width, field_height])
        teammate_vel = teammate['vel'] / self.config.PLAYER_SPEED

        # Opponents
        opp1 = opponents[0] if len(opponents) > 0 else agent
        opp2 = opponents[1] if len(opponents) > 1 else agent

        opp1_pos = opp1['pos'] / np.array([field_width, field_height])
        opp1_vel = opp1['vel'] / self.config.PLAYER_SPEED

        opp2_pos = opp2['pos'] / np.array([field_width, field_height])
        opp2_vel = opp2['vel'] / self.config.PLAYER_SPEED

        # Goal information
        if agent_team == 0:  # Blue team (left side)
            own_goal_pos = np.array([0, 0.5])
            enemy_goal_pos = np.array([1, 0.5])
        else:  # Red team (right side)
            own_goal_pos = np.array([1, 0.5])
            enemy_goal_pos = np.array([0, 0.5])

        own_goal_dist = np.linalg.norm(agent_pos - own_goal_pos)
        enemy_goal_dist = np.linalg.norm(agent_pos - enemy_goal_pos)

        # Goal angles
        own_goal_vec = own_goal_pos - agent_pos
        enemy_goal_vec = enemy_goal_pos - agent_pos

        own_goal_angle = np.arctan2(own_goal_vec[1], own_goal_vec[0])
        enemy_goal_angle = np.arctan2(enemy_goal_vec[1], enemy_goal_vec[0])

        # Context information
        time_remaining = (max_steps - step) / max_steps
        score_diff = scores[agent_team] - scores[1 - agent_team]

        # Construct observation
        observation = np.concatenate([
            # Self state
            agent_pos, agent_vel,

            # Ball state
            ball_pos, ball_vel,

            # Teammate state
            teammate_pos, teammate_vel,

            # Opponent states
            opp1_pos, opp1_vel,
            opp2_pos, opp2_vel,

            # Goal information
            [own_goal_dist, enemy_goal_dist, own_goal_angle, enemy_goal_angle],

            # Context information
            [ball_possession, time_remaining, score_diff, last_touch]
        ]).astype(np.float32)

        return observation

class ActionSpace:
    """
    Action space for each agent
    """
    def __init__(self, action_type: str = "continuous"):
        self.action_type = action_type

        if action_type == "continuous":
            # 5-dimensional continuous action space
            # [move_x, move_y, kick_power, kick_dir_x, kick_dir_y]
            self.gym_space = spaces.Box(
                low=np.array([-1, -1, 0, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # 9-dimensional discrete action space
            # [NOOP, UP, DOWN, LEFT, RIGHT, KICK_UP, KICK_DOWN, KICK_LEFT, KICK_RIGHT]
            self.gym_space = spaces.Discrete(9)

        self.action_meanings = {
            0: "NOOP",
            1: "UP",
            2: "DOWN",
            3: "LEFT",
            4: "RIGHT",
            5: "KICK_UP",
            6: "KICK_DOWN",
            7: "KICK_LEFT",
            8: "KICK_RIGHT"
        }

    def sample(self) -> Union[np.ndarray, int]:
        """Sample a random action"""
        return self.gym_space.sample()

    def convert_discrete_to_continuous(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous action format"""
        action_map = {
            0: np.array([0, 0, 0, 0, 0]),        # NOOP
            1: np.array([0, -1, 0, 0, 0]),       # UP
            2: np.array([0, 1, 0, 0, 0]),        # DOWN
            3: np.array([-1, 0, 0, 0, 0]),       # LEFT
            4: np.array([1, 0, 0, 0, 0]),        # RIGHT
            5: np.array([0, 0, 0.5, 0, -1]),     # KICK_UP
            6: np.array([0, 0, 0.5, 0, 1]),      # KICK_DOWN
            7: np.array([0, 0, 0.5, -1, 0]),     # KICK_LEFT
            8: np.array([0, 0, 0.5, 1, 0]),      # KICK_RIGHT
        }

        return action_map.get(action, action_map[0]).astype(np.float32)

def create_spaces(config: SoccerEnvironmentConfig, action_type: str = "continuous"):
    """Create observation and action spaces"""
    obs_space = ObservationSpace(config)
    action_space = ActionSpace(action_type)

    return obs_space, action_space