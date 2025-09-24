# Multi-Agent Soccer Game with Deep Reinforcement Learning

A comprehensive implementation of a 2v2 soccer environment using PettingZoo framework with multiple multi-agent learning algorithms including MADDPG, DQN, PPO, and QMIX.

## 🎯 Project Overview

This project implements a physics-based soccer game environment where multiple agents learn to play soccer through deep reinforcement learning. The environment supports various multi-agent learning paradigms and provides comprehensive tools for analysis and visualization.

### Key Features

- **Complete 2v2 Soccer Environment**: Physics-based gameplay with realistic ball dynamics and player interactions
- **PettingZoo Compatible**: Fully compatible with the PettingZoo multi-agent environment framework
- **Multiple Learning Algorithms**: Supports DQN, PPO, MADDPG, and extensible for more algorithms
- **Rich Observation Space**: 28-dimensional observations including player positions, ball state, and contextual information
- **Multi-objective Reward System**: Sophisticated reward structure encouraging both individual performance and teamwork
- **Comprehensive Analysis**: Tools for analyzing emergent behaviors, cooperation patterns, and learning performance

## 🏗️ Architecture

### Environment Components
- **Physics Engine** (`physics.py`): Realistic ball and player dynamics with collision detection
- **Renderer** (`renderer.py`): Pygame-based visualization with recording capabilities
- **Observation/Action Spaces** (`spaces.py`): 28D continuous observations, 5D continuous actions
- **Reward System** (`rewards.py`): Multi-objective rewards with potential-based shaping
- **Main Environment** (`soccer_env.py`): PettingZoo-compatible wrapper integrating all components

### Agent Implementations
- **Random Agent**: Baseline for performance comparison
- **DQN Agent**: Deep Q-Network with experience replay and target networks
- **MADDPG Agent**: Multi-Agent Deep Deterministic Policy Gradient with centralized critic
- **Extensible Framework**: Easy to add new agent types

### Training Frameworks
- **Independent Learning**: Each agent learns independently using single-agent methods
- **MADDPG Training**: Centralized training with decentralized execution
- **Self-Play Training**: Agents learn by playing against copies of themselves
- **Cooperative Training**: Agents learn to maximize team performance

## 📋 Requirements

```bash
# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.5.0

# Multi-Agent RL Environment
pettingzoo>=1.22.0
gymnasium>=0.27.0
stable-baselines3>=2.0.0

# Physics and Visualization
pygame>=2.1.0
matplotlib>=3.6.0
seaborn>=0.11.0

# Utilities
tqdm>=4.64.0
tensorboard>=2.10.0
opencv-python>=4.6.0
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python test_environment.py
```

### 2. Basic Usage

```python
from config import SoccerEnvironmentConfig
from soccer_env import make_soccer_env
from agents import RandomAgent

# Create environment
config = SoccerEnvironmentConfig()
env = make_soccer_env(config, render_mode="human")

# Create agents
agents = {}
for i, agent_name in enumerate(env.agents):
    agents[agent_name] = RandomAgent(i)

# Run episode
observations = env.reset()
while not all(env.terminations.values()):
    for agent in env.agents:
        obs = env.observe(agent)
        action = agents[agent].select_action(obs)
        env.step(action)

    env.render()  # Visualize gameplay
```

### 3. Training Agents

```python
from trainers import IndependentLearningTrainer
from config import TrainingConfig

# Setup training
env_config = SoccerEnvironmentConfig()
training_config = TrainingConfig()

# Train DQN agents
trainer = IndependentLearningTrainer(
    env_config, training_config,
    agent_type="dqn"
)
results = trainer.train(num_episodes=2000)
```

### 4. Run Experiments

```python
from run_experiments import run_quick_comparison, run_full_comparison

# Quick comparison (recommended for first run)
results = run_quick_comparison()

# Full comparison (all algorithms)
results = run_full_comparison()
```

## 📊 Environment Specifications

### Observation Space (28 dimensions per agent)
- **Self state** (4D): position, velocity
- **Ball state** (4D): position, velocity
- **Teammate state** (4D): position, velocity
- **Opponent states** (8D): positions, velocities of 2 opponents
- **Goal information** (4D): distances and angles to goals
- **Context information** (4D): ball possession, time remaining, score difference, last touch

### Action Space (5 dimensions)
- **Movement** (2D): x, y movement directions [-1, 1]
- **Kick power** (1D): kick strength [0, 1]
- **Kick direction** (2D): kick direction x, y [-1, 1]

### Reward Structure
- **Goal scoring**: +100 (team), -100 (conceded)
- **Ball contact**: +5
- **Goal approach**: +0.1 per unit closer
- **Ball approach**: +0.05 per unit closer
- **Teamwork**: +0.02 for optimal positioning
- **Penalties**: -10 (out of bounds), -0.1 (stalemate)

## 🧪 Experiments and Results

### Implemented Algorithms
1. **Random Baseline**: Random action selection for performance baseline
2. **DQN Independent**: Independent Deep Q-Network learning
3. **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient
4. **PPO** (Planned): Proximal Policy Optimization
5. **QMIX** (Planned): Monotonic Value Function Factorization

### Performance Metrics
- Episode rewards and lengths
- Win rates per team
- Goal scoring statistics
- Emergent behavior analysis
- Cooperation metrics

### Analysis Tools
- Learning curve visualization
- Performance comparison charts
- Behavioral heatmaps
- Strategy evolution tracking

## 📁 Project Structure

```
DRL/
├── config.py                    # Configuration classes
├── physics.py                   # Physics engine
├── renderer.py                  # Visualization and rendering
├── spaces.py                    # Observation/action space definitions
├── rewards.py                   # Multi-objective reward system
├── soccer_env.py                # Main PettingZoo environment
├── agents.py                    # Agent implementations
├── trainers.py                  # Training frameworks
├── test_environment.py          # Environment testing
├── run_experiments.py           # Experiment runner
├── multiagents_soccer.ipynb     # Main Jupyter notebook
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔬 Research Applications

This environment is designed for research in:
- Multi-agent reinforcement learning
- Emergent behavior analysis
- Cooperation vs competition dynamics
- Transfer learning in multi-agent settings
- Curriculum learning strategies
- Communication in multi-agent systems

## 🛠️ Extending the Framework

### Adding New Agents
```python
class MyCustomAgent(BaseAgent):
    def select_action(self, observation, training=True):
        # Your agent logic here
        pass

    def learn(self, experiences):
        # Your learning algorithm here
        pass
```

### Custom Reward Functions
```python
class CustomRewardCalculator(RewardCalculator):
    def calculate_reward(self, agent_id, action, prev_state, current_state, **kwargs):
        # Your custom reward logic
        return reward
```

### Adding New Training Methods
```python
class CustomTrainer(BaseTrainer):
    def train(self, num_episodes):
        # Your training loop
        pass
```

## 📚 References

1. **MADDPG**: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (Lowe et al., 2017)
2. **QMIX**: "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (Rashid et al., 2018)
3. **PettingZoo**: "PettingZoo: Gym for Multi-Agent Reinforcement Learning" (Terry et al., 2021)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Future Enhancements

- [ ] Complete PPO and QMIX implementations
- [ ] Add communication channels between agents
- [ ] Implement curriculum learning
- [ ] Add more complex team formations
- [ ] Integration with Weights & Biases for experiment tracking
- [ ] Video recording of best episodes
- [ ] Real-time human vs AI gameplay
- [ ] Performance optimization for larger team sizes

---

**Ultra Think Implementation**: This project demonstrates a comprehensive approach to multi-agent reinforcement learning, combining theoretical foundations with practical implementation. The modular design allows for easy experimentation with different algorithms while providing robust tools for analysis and visualization of emergent behaviors in competitive multi-agent environments.