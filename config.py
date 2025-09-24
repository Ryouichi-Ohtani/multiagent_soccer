"""
Configuration file for Multi-Agent Soccer Game
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class SoccerEnvironmentConfig:
    """Environment configuration for soccer game"""
    FIELD_SIZE: Tuple[int, int] = (800, 600)
    GOAL_SIZE: Tuple[int, int] = (20, 200)
    BALL_RADIUS: int = 10
    PLAYER_RADIUS: int = 20
    MAX_STEPS: int = 1000

    NUM_PLAYERS_PER_TEAM: int = 2
    TEAM_COLORS: Tuple[str, str] = ('blue', 'red')
    PLAYER_SPEED: float = 5.0
    BALL_SPEED_MULTIPLIER: float = 1.5

    FRICTION: float = 0.95
    BALL_DECAY: float = 0.98
    COLLISION_THRESHOLD: float = 30.0

@dataclass
class MADDPGConfig:
    """MADDPG algorithm configuration"""
    obs_dim: int = 28
    action_dim: int = 5
    global_obs_dim: int = 112  # 28 * 4 agents
    global_action_dim: int = 20  # 5 * 4 agents
    hidden_dims: Tuple[int, ...] = (256, 128)

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    batch_size: int = 256
    buffer_size: int = int(1e6)
    noise_scale: float = 0.1
    noise_decay: float = 0.9999

@dataclass
class TrainingConfig:
    """Training configuration"""
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    save_freq: int = 1000
    eval_freq: int = 500
    log_freq: int = 100

    # Reproducibility
    random_seed: int = 42

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_name: str = "soccer_multiagent"
    log_dir: str = "logs"
    save_dir: str = "saved_models"
    video_dir: str = "videos"

    # Algorithms to run
    algorithms: Tuple[str, ...] = ("random", "dqn", "ppo", "maddpg")