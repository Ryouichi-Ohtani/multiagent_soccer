#!/usr/bin/env python3
"""
Create a fully integrated Google Colab notebook for Multi-Agent Soccer Game
This script combines all .py files into a single executable notebook
"""

import json
import os

def create_integrated_notebook():
    """Create the complete integrated notebook"""
    
    # Read all Python files
    python_files = {
        'config.py': open('/home/user/webapp/config.py', 'r').read(),
        'physics.py': open('/home/user/webapp/physics.py', 'r').read(),
        'spaces.py': open('/home/user/webapp/spaces.py', 'r').read(),
        'rewards.py': open('/home/user/webapp/rewards.py', 'r').read(),
        'renderer.py': open('/home/user/webapp/renderer.py', 'r').read(),
        'soccer_env.py': open('/home/user/webapp/soccer_env.py', 'r').read(),
        'agents.py': open('/home/user/webapp/agents.py', 'r').read(),
        'trainers.py': open('/home/user/webapp/trainers.py', 'r').read(),
        'test_environment.py': open('/home/user/webapp/test_environment.py', 'r').read()
    }
    
    # Create notebook structure
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "collapsed_sections": []
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": []
    }
    
    # Add title and overview
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "title_cell"},
        "source": [
            "# 🎮 Multi-Agent Soccer Game with Deep Reinforcement Learning\n",
            "## Google Colab 完全統合版\n\n",
            "このノートブックは深層強化学習を用いたマルチエージェントサッカーゲームの完全統合版です。\n",
            "すべての必要なコードが単一のノートブック内に含まれているため、Google Colabで直接実行できます。\n\n",
            "### 📋 実装内容\n",
            "- **環境**: 2v2サッカーゲーム（PettingZoo互換）\n",
            "- **物理エンジン**: リアルタイムな衝突検出とボール物理\n",
            "- **エージェント**: Random, DQN, MADDPG\n",
            "- **観測空間**: 28次元（プレイヤー位置、ボール状態、ゲーム情報）\n",
            "- **行動空間**: 5次元連続行動（移動+キック）\n",
            "- **報酬システム**: 多目的報酬関数\n\n",
            "### ✨ 特徴\n",
            "1. 完全な物理シミュレーション\n",
            "2. 複数の学習アルゴリズム実装\n",
            "3. 創発的行動の分析\n",
            "4. チームワーク評価メトリクス\n\n",
            "### 📊 実装ドキュメント\n",
            "詳細な実装内容は「マルチエージェントサッカーゲーム実装ドキュメント.pdf」に記載されています。"
        ]
    })
    
    # Add dependency installation
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "install_header"},
        "source": ["## 📦 1. 必要なライブラリのインストール"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "install_deps"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Google Colab用のライブラリインストール\n",
            "!pip install -q gymnasium\n",
            "!pip install -q pettingzoo\n",
            "!pip install -q pygame\n",
            "!pip install -q torch torchvision\n",
            "!pip install -q matplotlib seaborn\n",
            "!pip install -q numpy\n",
            "!pip install -q opencv-python\n\n",
            "print(\"✅ All dependencies installed successfully!\")"
        ]
    })
    
    # Add imports
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "import_header"},
        "source": ["## 🔧 2. 基本インポートとセットアップ"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "imports"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 基本ライブラリのインポート\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "import torch.nn.functional as F\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from typing import Dict, List, Tuple, Optional, Any, Union\n",
            "import json\n",
            "from collections import defaultdict, deque\n",
            "import random\n",
            "from dataclasses import dataclass\n",
            "from abc import ABC, abstractmethod\n",
            "import time\n",
            "import os\n\n",
            "# Gymnasium and PettingZoo\n",
            "import gymnasium as gym\n",
            "from gymnasium import spaces\n",
            "from pettingzoo import AECEnv\n",
            "from pettingzoo.utils import agent_selector, wrappers\n\n",
            "# Pygame (optional for rendering)\n",
            "try:\n",
            "    import pygame\n",
            "    PYGAME_AVAILABLE = True\n",
            "except ImportError:\n",
            "    PYGAME_AVAILABLE = False\n",
            "    print(\"⚠️ Pygame not available. Rendering disabled.\")\n\n",
            "# Set random seeds\n",
            "np.random.seed(42)\n",
            "torch.manual_seed(42)\n",
            "if torch.cuda.is_available():\n",
            "    torch.cuda.manual_seed(42)\n\n",
            "# Set matplotlib style\n",
            "plt.style.use('seaborn-v0_8-darkgrid')\n",
            "sns.set_palette(\"husl\")\n\n",
            "print(f\"✅ PyTorch version: {torch.__version__}\")\n",
            "print(f\"✅ CUDA available: {torch.cuda.is_available()}\")\n",
            "print(f\"✅ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\")"
        ]
    })
    
    # Add configuration code
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "config_header"},
        "source": ["## ⚙️ 3. 設定クラス (Configuration)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "config_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['config.py'] + "\n\nprint(\"✅ Configuration classes defined successfully!\")"
    })
    
    # Add physics engine
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "physics_header"},
        "source": ["## 🎯 4. 物理エンジン (Physics Engine)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "physics_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['physics.py'] + "\n\nprint(\"✅ Physics engine implemented successfully!\")"
    })
    
    # Add observation and action spaces
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "spaces_header"},
        "source": ["## 🎮 5. 観測・行動空間 (Observation and Action Spaces)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "spaces_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['spaces.py'] + "\n\nprint(\"✅ Observation and action spaces defined successfully!\")"
    })
    
    # Add reward system
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "rewards_header"},
        "source": ["## 🏆 6. 報酬システム (Reward System)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "rewards_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['rewards.py'] + "\n\nprint(\"✅ Reward system implemented successfully!\")"
    })
    
    # Add renderer
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "renderer_header"},
        "source": ["## 🎨 7. レンダラー (Renderer)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "renderer_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['renderer.py'] + "\n\nprint(\"✅ Renderer implemented successfully!\")"
    })
    
    # Add main environment
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "env_header"},
        "source": ["## 🌍 8. メイン環境 (Main Soccer Environment)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "env_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['soccer_env.py'] + "\n\nprint(\"✅ Soccer environment implemented successfully!\")"
    })
    
    # Add agents
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "agents_header"},
        "source": ["## 🤖 9. エージェント実装 (Agents)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "agents_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['agents.py'] + "\n\nprint(\"✅ Agents implemented successfully!\")"
    })
    
    # Add trainers
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "trainers_header"},
        "source": ["## 📚 10. トレーニングフレームワーク (Training Frameworks)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "trainers_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['trainers.py'] + "\n\nprint(\"✅ Training frameworks implemented successfully!\")"
    })
    
    # Add test functions
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "test_header"},
        "source": ["## 🧪 11. テスト関数 (Test Functions)"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "test_code"},
        "execution_count": None,
        "outputs": [],
        "source": python_files['test_environment.py'] + "\n\nprint(\"✅ Test functions implemented successfully!\")"
    })
    
    # Add execution section
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "execution_header"},
        "source": [
            "## 🚀 12. 実行セクション (Execution Section)\n\n",
            "以下のセルで環境をテストし、エージェントを訓練できます。"
        ]
    })
    
    # Test environment
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "test_env_header"},
        "source": ["### 12.1 環境テスト"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "test_env"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 環境のテストを実行\n",
            "print(\"🧪 Running environment tests...\")\n",
            "print(\"=\" * 60)\n\n",
            "test_passed = run_all_tests()\n\n",
            "if test_passed:\n",
            "    print(\"\\n✅ All environment tests passed! Ready to proceed with training.\")\n",
            "else:\n",
            "    print(\"\\n❌ Some tests failed. Please check the implementation.\")"
        ]
    })
    
    # Create environment and run baseline
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "baseline_header"},
        "source": ["### 12.2 環境作成とベースライン実行"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "baseline_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 環境の作成\n",
            "config = SoccerEnvironmentConfig()\n",
            "env = make_soccer_env(config, render_mode=None, action_type=\"continuous\")\n\n",
            "print(\"📊 Environment Configuration:\")\n",
            "print(f\"  Field Size: {config.FIELD_SIZE}\")\n",
            "print(f\"  Max Steps: {config.MAX_STEPS}\")\n",
            "print(f\"  Players per Team: {config.NUM_PLAYERS_PER_TEAM}\")\n",
            "print(f\"  Agents: {env.agents}\")\n",
            "print(f\"  Observation space: {env.observation_spaces[env.agents[0]].shape}\")\n",
            "print(f\"  Action space: {env.action_spaces[env.agents[0]].shape}\")\n\n",
            "# ランダムエージェントでベースライン実行\n",
            "print(\"\\n🎲 Running baseline with random agents...\")\n",
            "print(\"=\" * 60)\n\n",
            "random_agents = {}\n",
            "for i, agent_name in enumerate(env.agents):\n",
            "    random_agents[agent_name] = RandomAgent(i, action_space_size=5, action_type=\"continuous\")\n\n",
            "# 5エピソード実行\n",
            "episode_stats = []\n",
            "for episode in range(5):\n",
            "    env.reset()\n",
            "    episode_reward = {agent: 0 for agent in env.agents}\n",
            "    steps = 0\n",
            "    \n",
            "    while not all(env.terminations.values()) and not all(env.truncations.values()):\n",
            "        for agent in env.agents:\n",
            "            if not env.terminations.get(agent, False) and not env.truncations.get(agent, False):\n",
            "                obs = env.observe(agent)\n",
            "                action = random_agents[agent].select_action(obs, training=False)\n",
            "                env.step(action)\n",
            "                episode_reward[agent] += env.rewards.get(agent, 0)\n",
            "                steps += 1\n",
            "                \n",
            "                if env.terminations.get(agent, False) or env.truncations.get(agent, False):\n",
            "                    break\n",
            "    \n",
            "    print(f\"Episode {episode + 1}: {steps} steps, Scores: {env.scores}, \"\n",
            "          f\"Total Reward: {sum(episode_reward.values()):.2f}\")\n",
            "    \n",
            "    episode_stats.append({\n",
            "        'episode': episode,\n",
            "        'steps': steps,\n",
            "        'scores': env.scores.copy(),\n",
            "        'total_reward': sum(episode_reward.values())\n",
            "    })\n\n",
            "# 統計表示\n",
            "avg_steps = np.mean([stat['steps'] for stat in episode_stats])\n",
            "avg_reward = np.mean([stat['total_reward'] for stat in episode_stats])\n",
            "team_0_wins = sum(1 for stat in episode_stats if stat['scores'][0] > stat['scores'][1])\n",
            "team_1_wins = sum(1 for stat in episode_stats if stat['scores'][1] > stat['scores'][0])\n",
            "draws = len(episode_stats) - team_0_wins - team_1_wins\n\n",
            "print(f\"\\n📈 Baseline Statistics (Random Agents):\")\n",
            "print(f\"  Average Steps per Episode: {avg_steps:.1f}\")\n",
            "print(f\"  Average Total Reward: {avg_reward:.2f}\")\n",
            "print(f\"  Team 0 (Blue) Wins: {team_0_wins}\")\n",
            "print(f\"  Team 1 (Red) Wins: {team_1_wins}\")\n",
            "print(f\"  Draws: {draws}\")"
        ]
    })
    
    # DQN Training example
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "dqn_header"},
        "source": ["### 12.3 DQNエージェントの訓練例"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "dqn_training"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# DQNエージェントの訓練例（短縮版）\n",
            "print(\"🤖 Training DQN agents (demo version - 100 episodes)...\")\n",
            "print(\"=\" * 60)\n\n",
            "# トレーナーの作成\n",
            "training_config = TrainingConfig()\n",
            "agent_configs = {\n",
            "    'obs_dim': 28,\n",
            "    'action_dim': 9,  # 離散行動\n",
            "    'hidden_dims': (256, 128),\n",
            "    'lr': 1e-3,\n",
            "    'gamma': 0.99,\n",
            "    'epsilon': 1.0,\n",
            "    'epsilon_decay': 0.995,\n",
            "    'epsilon_min': 0.01,\n",
            "    'buffer_size': 10000,\n",
            "    'batch_size': 64\n",
            "}\n\n",
            "trainer = IndependentLearningTrainer(\n",
            "    env_config=config,\n",
            "    training_config=training_config,\n",
            "    agent_type=\"dqn\",\n",
            "    agent_configs=agent_configs\n",
            ")\n\n",
            "# 100エピソードの訓練を実行（デモ用）\n",
            "results = trainer.train(num_episodes=100)\n\n",
            "# 結果の可視化\n",
            "plt.figure(figsize=(12, 4))\n\n",
            "# Episode rewards\n",
            "plt.subplot(1, 3, 1)\n",
            "plt.plot(results['episode_rewards'])\n",
            "plt.title('Episode Rewards')\n",
            "plt.xlabel('Episode')\n",
            "plt.ylabel('Total Reward')\n",
            "plt.grid(True)\n\n",
            "# Episode lengths\n",
            "plt.subplot(1, 3, 2)\n",
            "plt.plot(results['episode_lengths'])\n",
            "plt.title('Episode Lengths')\n",
            "plt.xlabel('Episode')\n",
            "plt.ylabel('Steps')\n",
            "plt.grid(True)\n\n",
            "# Team scores\n",
            "plt.subplot(1, 3, 3)\n",
            "team_0_scores = [score[0] for score in results['scores_history']]\n",
            "team_1_scores = [score[1] for score in results['scores_history']]\n",
            "plt.plot(team_0_scores, label='Team 0 (Blue)', alpha=0.7)\n",
            "plt.plot(team_1_scores, label='Team 1 (Red)', alpha=0.7)\n",
            "plt.title('Team Scores')\n",
            "plt.xlabel('Episode')\n",
            "plt.ylabel('Score')\n",
            "plt.legend()\n",
            "plt.grid(True)\n\n",
            "plt.tight_layout()\n",
            "plt.show()\n\n",
            "print(\"\\n✅ Training completed!\")"
        ]
    })
    
    # MADDPG Training example
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "maddpg_header"},
        "source": ["### 12.4 MADDPGエージェントの訓練例"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "maddpg_training"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# MADDPGエージェントの訓練例（短縮版）\n",
            "print(\"🤖 Training MADDPG agents (demo version - 100 episodes)...\")\n",
            "print(\"=\" * 60)\n\n",
            "# MADDPG設定\n",
            "maddpg_config = MADDPGConfig()\n\n",
            "# MADDPGトレーナーの作成\n",
            "maddpg_trainer = MADDPGTrainer(\n",
            "    env_config=config,\n",
            "    training_config=training_config,\n",
            "    maddpg_config=maddpg_config\n",
            ")\n\n",
            "# 100エピソードの訓練を実行（デモ用）\n",
            "maddpg_results = maddpg_trainer.train(num_episodes=100)\n\n",
            "# 結果の可視化\n",
            "plt.figure(figsize=(10, 4))\n\n",
            "# Episode rewards\n",
            "plt.subplot(1, 2, 1)\n",
            "plt.plot(maddpg_results['episode_rewards'])\n",
            "plt.title('MADDPG Episode Rewards')\n",
            "plt.xlabel('Episode')\n",
            "plt.ylabel('Total Reward')\n",
            "plt.grid(True)\n\n",
            "# Team cooperation\n",
            "plt.subplot(1, 2, 2)\n",
            "team_0_scores = [score[0] for score in maddpg_results['scores_history']]\n",
            "team_1_scores = [score[1] for score in maddpg_results['scores_history']]\n",
            "cooperation_metric = [abs(s0 - s1) for s0, s1 in zip(team_0_scores, team_1_scores)]\n",
            "plt.plot(cooperation_metric)\n",
            "plt.title('Team Balance (Lower is Better)')\n",
            "plt.xlabel('Episode')\n",
            "plt.ylabel('Score Difference')\n",
            "plt.grid(True)\n\n",
            "plt.tight_layout()\n",
            "plt.show()\n\n",
            "print(\"\\n✅ MADDPG training completed!\")"
        ]
    })
    
    # Summary
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "summary"},
        "source": [
            "## 📝 まとめ\n\n",
            "このノートブックでは、以下を実装しました：\n\n",
            "### ✅ 実装済み機能\n",
            "1. **完全な物理エンジン**: 衝突検出、ボール物理、プレイヤー移動\n",
            "2. **PettingZoo互換環境**: マルチエージェント強化学習の標準インターフェース\n",
            "3. **複数のエージェント実装**:\n",
            "   - Random Agent (ベースライン)\n",
            "   - DQN Agent (独立学習)\n",
            "   - MADDPG Agent (協調学習)\n",
            "4. **豊富な観測空間**: 28次元の観測（位置、速度、ゲーム状態）\n",
            "5. **多目的報酬システム**: ゴール、ボール制御、チームワーク\n",
            "6. **訓練フレームワーク**: 独立学習とMADDPG\n\n",
            "### 🚀 今後の拡張\n",
            "1. PPOエージェントの実装\n",
            "2. QMIXアルゴリズムの追加\n",
            "3. セルフプレイ訓練の実装\n",
            "4. より詳細な創発的行動の分析\n",
            "5. リアルタイムビジュアライゼーション\n\n",
            "### 📚 参考文献\n",
            "- [PettingZoo Documentation](https://pettingzoo.farama.org/)\n",
            "- [MADDPG Paper](https://arxiv.org/abs/1706.02275)\n",
            "- [DQN Paper](https://arxiv.org/abs/1312.5602)\n\n",
            "### 🎯 実験のヒント\n",
            "- エピソード数を増やすことで学習性能が向上します\n",
            "- ハイパーパラメータの調整で異なる戦略が生まれます\n",
            "- 報酬重みの調整でエージェントの行動が変化します\n\n",
            "**Happy Training! 🎮**"
        ]
    })
    
    return notebook

# Create and save the notebook
if __name__ == "__main__":
    notebook = create_integrated_notebook()
    
    # Save the notebook
    with open("/home/user/webapp/multiagents_soccer_colab.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Integrated notebook created successfully!")
    print("📁 Saved as: multiagents_soccer_colab.ipynb")