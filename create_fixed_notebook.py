#!/usr/bin/env python3
"""
Create a fixed version of the integrated Google Colab notebook
with all import statements properly handled
"""

import json
import re

def remove_internal_imports(code):
    """Remove imports from internal modules"""
    internal_modules = [
        'config', 'physics', 'spaces', 'rewards', 
        'renderer', 'soccer_env', 'agents', 'trainers', 
        'test_environment'
    ]
    
    lines = code.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Check if this is an import from internal modules
        is_internal_import = False
        for module in internal_modules:
            if re.search(f'from {module} import', line) or re.search(f'^import {module}', line.strip()):
                is_internal_import = True
                break
        
        # Skip internal imports (comment them out)
        if not is_internal_import:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def create_fixed_integrated_notebook():
    """Create the complete integrated notebook with fixed imports"""
    
    # Read all Python files
    python_files = {}
    file_names = ['config.py', 'physics.py', 'spaces.py', 'rewards.py', 
                  'renderer.py', 'soccer_env.py', 'agents.py', 'trainers.py', 
                  'test_environment.py']
    
    for file_name in file_names:
        with open(f'/home/user/webapp/{file_name}', 'r') as f:
            # Remove internal imports from each file
            python_files[file_name] = remove_internal_imports(f.read())
    
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
            "## Google Colab 完全統合版 (修正版)\n\n",
            "このノートブックは深層強化学習を用いたマルチエージェントサッカーゲームの完全統合版です。\n",
            "すべての必要なコードが単一のノートブック内に含まれており、インポートエラーが修正されています。\n\n",
            "### 📋 実装内容\n",
            "- **環境**: 2v2サッカーゲーム（PettingZoo互換）\n",
            "- **物理エンジン**: リアルタイムな衝突検出とボール物理\n",
            "- **エージェント**: Random, DQN, MADDPG\n",
            "- **観測空間**: 28次元（プレイヤー位置、ボール状態、ゲーム情報）\n",
            "- **行動空間**: 5次元連続行動（移動+キック）\n",
            "- **報酬システム**: 多目的報酬関数\n\n",
            "### ⚠️ 実行の注意事項\n",
            "1. **上から順番に実行してください** - 各セルが前のセルで定義されたクラスに依存しています\n",
            "2. **GPU使用推奨** - Runtime → Change runtime type → GPU\n",
            "3. **すべてのコードセルを実行** - スキップすると後続のセルでエラーが発生します\n\n",
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
        "metadata": {"id": "install_deps", "cellView": "form"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "#@title ライブラリのインストール (クリックして実行)\n",
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
        "metadata": {"id": "imports", "cellView": "form"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "#@title 基本ライブラリのインポート (クリックして実行)\n",
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
    
    # Add all code sections with proper ordering
    sections = [
        ("⚙️ 3. 設定クラス (Configuration)", "config_", python_files['config.py'], 
         "Configuration classes defined"),
        ("🎯 4. 物理エンジン (Physics Engine)", "physics_", python_files['physics.py'], 
         "Physics engine implemented"),
        ("🎮 5. 観測・行動空間 (Observation and Action Spaces)", "spaces_", python_files['spaces.py'], 
         "Observation and action spaces defined"),
        ("🏆 6. 報酬システム (Reward System)", "rewards_", python_files['rewards.py'], 
         "Reward system implemented"),
        ("🎨 7. レンダラー (Renderer)", "renderer_", python_files['renderer.py'], 
         "Renderer implemented"),
        ("🌍 8. メイン環境 (Main Soccer Environment)", "env_", python_files['soccer_env.py'], 
         "Soccer environment implemented"),
        ("🤖 9. エージェント実装 (Agents)", "agents_", python_files['agents.py'], 
         "Agents implemented"),
        ("📚 10. トレーニングフレームワーク (Training Frameworks)", "trainers_", python_files['trainers.py'], 
         "Training frameworks implemented"),
        ("🧪 11. テスト関数 (Test Functions)", "test_", python_files['test_environment.py'], 
         "Test functions implemented")
    ]
    
    for title, id_prefix, code, success_msg in sections:
        # Add section header
        notebook['cells'].append({
            "cell_type": "markdown",
            "metadata": {"id": f"{id_prefix}header"},
            "source": [f"## {title}"]
        })
        
        # Add code cell
        notebook['cells'].append({
            "cell_type": "code",
            "metadata": {"id": f"{id_prefix}code"},
            "execution_count": None,
            "outputs": [],
            "source": code + f"\n\nprint(\"✅ {success_msg} successfully!\")"
        })
    
    # Add execution section
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "execution_header"},
        "source": [
            "## 🚀 12. 実行セクション (Execution Section)\n\n",
            "以下のセルで環境をテストし、エージェントを訓練できます。\n",
            "**重要**: 上記のすべてのコードセルを実行してから、以下のセルを実行してください。"
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
    
    # Simple training demo
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "training_header"},
        "source": ["### 12.3 簡単な訓練デモ"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "simple_training"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 簡単な訓練デモ（Random Agentsの性能測定）\n",
            "print(\"📊 Performance measurement over multiple episodes...\")\n",
            "print(\"=\" * 60)\n\n",
            "num_test_episodes = 20\n",
            "all_rewards = []\n",
            "all_scores = {'team_0': [], 'team_1': []}\n\n",
            "for ep in range(num_test_episodes):\n",
            "    env.reset()\n",
            "    episode_reward = 0\n",
            "    \n",
            "    while not all(env.terminations.values()) and not all(env.truncations.values()):\n",
            "        for agent in env.agents:\n",
            "            if not env.terminations.get(agent, False) and not env.truncations.get(agent, False):\n",
            "                obs = env.observe(agent)\n",
            "                action = random_agents[agent].select_action(obs, training=False)\n",
            "                env.step(action)\n",
            "                episode_reward += env.rewards.get(agent, 0)\n",
            "                \n",
            "                if env.terminations.get(agent, False) or env.truncations.get(agent, False):\n",
            "                    break\n",
            "    \n",
            "    all_rewards.append(episode_reward)\n",
            "    all_scores['team_0'].append(env.scores[0])\n",
            "    all_scores['team_1'].append(env.scores[1])\n",
            "    \n",
            "    if (ep + 1) % 5 == 0:\n",
            "        print(f\"Episodes {ep-3}-{ep+1}: Avg Reward = {np.mean(all_rewards[-5:]):.2f}, \"\n",
            "              f\"Scores = Blue:{np.mean(all_scores['team_0'][-5:]):.1f}, \"\n",
            "              f\"Red:{np.mean(all_scores['team_1'][-5:]):.1f}\")\n\n",
            "# 結果のプロット\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n\n",
            "# Rewards over time\n",
            "ax1.plot(all_rewards, alpha=0.5)\n",
            "ax1.plot(np.convolve(all_rewards, np.ones(5)/5, mode='valid'), linewidth=2)\n",
            "ax1.set_title('Total Rewards per Episode')\n",
            "ax1.set_xlabel('Episode')\n",
            "ax1.set_ylabel('Total Reward')\n",
            "ax1.grid(True)\n\n",
            "# Team scores\n",
            "ax2.plot(all_scores['team_0'], label='Team 0 (Blue)', alpha=0.7)\n",
            "ax2.plot(all_scores['team_1'], label='Team 1 (Red)', alpha=0.7)\n",
            "ax2.set_title('Team Scores Over Episodes')\n",
            "ax2.set_xlabel('Episode')\n",
            "ax2.set_ylabel('Goals Scored')\n",
            "ax2.legend()\n",
            "ax2.grid(True)\n\n",
            "plt.tight_layout()\n",
            "plt.show()\n\n",
            "print(f\"\\n📈 Final Statistics ({num_test_episodes} episodes):\")\n",
            "print(f\"  Mean Total Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\")\n",
            "print(f\"  Team 0 Mean Score: {np.mean(all_scores['team_0']):.2f} ± {np.std(all_scores['team_0']):.2f}\")\n",
            "print(f\"  Team 1 Mean Score: {np.mean(all_scores['team_1']):.2f} ± {np.std(all_scores['team_1']):.2f}\")"
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
            "### 🚀 使用方法\n",
            "1. **すべてのコードセルを上から順番に実行**\n",
            "2. 環境テストで動作確認\n",
            "3. ベースライン実行で基本性能を確認\n",
            "4. 必要に応じて訓練パラメータを調整\n\n",
            "### ⚠️ トラブルシューティング\n",
            "- **ModuleNotFoundError**: すべてのコードセルを実行したか確認\n",
            "- **NameError**: セルの実行順序を確認（上から順番に）\n",
            "- **GPU関連エラー**: Runtime → Change runtime type → GPU\n\n",
            "### 📚 参考文献\n",
            "- [PettingZoo Documentation](https://pettingzoo.farama.org/)\n",
            "- [MADDPG Paper](https://arxiv.org/abs/1706.02275)\n",
            "- [DQN Paper](https://arxiv.org/abs/1312.5602)\n\n",
            "**Happy Training! 🎮**"
        ]
    })
    
    return notebook

# Create and save the fixed notebook
if __name__ == "__main__":
    notebook = create_fixed_integrated_notebook()
    
    # Save the notebook
    with open("/home/user/webapp/multiagents_soccer_colab_fixed.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Fixed integrated notebook created successfully!")
    print("📁 Saved as: multiagents_soccer_colab_fixed.ipynb")
    print("\n📋 主な修正内容:")
    print("  1. 内部モジュール間のimport文を削除")
    print("  2. すべてのコードが自己完結型に")
    print("  3. 実行順序の明確化")
    print("  4. エラー防止のための注意事項追加")