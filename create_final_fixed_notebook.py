#!/usr/bin/env python3
"""
Create the final fixed version with proper agent_selector handling
"""

import json
import re

def fix_pettingzoo_imports(code):
    """Fix PettingZoo imports and usage"""
    # Replace the import line
    code = code.replace(
        'from pettingzoo.utils import agent_selector, wrappers',
        'from pettingzoo.utils import AgentSelector, wrappers'
    )
    
    # Fix all usages of agent_selector
    code = code.replace(
        'self._agent_selector = agent_selector(self.agents)',
        'self._agent_selector = AgentSelector(self.agents)'
    )
    
    code = code.replace(
        'self._agent_selector = agent_selector(self.agents)',
        'self._agent_selector = AgentSelector(self.agents)'
    )
    
    return code

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
        
        # Skip internal imports
        if not is_internal_import:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def create_final_fixed_notebook():
    """Create the final fixed notebook"""
    
    # Read all Python files and clean them
    python_files = {}
    file_names = ['config.py', 'physics.py', 'spaces.py', 'rewards.py', 
                  'renderer.py', 'soccer_env.py', 'agents.py', 'trainers.py', 
                  'test_environment.py']
    
    for file_name in file_names:
        with open(f'/home/user/webapp/{file_name}', 'r') as f:
            content = f.read()
            # Remove internal imports
            content = remove_internal_imports(content)
            # Fix PettingZoo imports specifically for soccer_env.py
            if file_name == 'soccer_env.py':
                content = fix_pettingzoo_imports(content)
            python_files[file_name] = content
    
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
    
    # Title and overview
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "title_cell"},
        "source": [
            "# 🎮 Multi-Agent Soccer Game with Deep Reinforcement Learning\n",
            "## Google Colab 完全統合版 (最終修正版)\n\n",
            "このノートブックは深層強化学習を用いたマルチエージェントサッカーゲームの完全統合版です。\n\n",
            "### ⚠️ 重要: 実行手順\n",
            "1. **Runtime → Change runtime type → GPU を選択**\n",
            "2. **Runtime → Run all または上から順番に実行**\n",
            "3. **すべてのコードセルを実行してから実行セクションを使用**\n\n",
            "### 📋 修正内容\n",
            "- ✅ ModuleNotFoundError 修正済み\n",
            "- ✅ TypeError (agent_selector) 修正済み\n",
            "- ✅ すべての内部import削除済み\n\n",
            "### 🎯 実装内容\n",
            "- 2v2サッカーゲーム（PettingZoo互換）\n",
            "- 物理エンジン・報酬システム完備\n",
            "- Random, DQN, MADDPG エージェント実装"
        ]
    })
    
    # Dependencies installation
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "install_header"},
        "source": ["## 📦 Step 1: 必要なライブラリのインストール"]
    })
    
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "install_deps"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 必要なライブラリをインストール\n",
            "!pip install -q gymnasium\n",
            "!pip install -q pettingzoo\n",
            "!pip install -q pygame\n",
            "!pip install -q torch torchvision\n",
            "!pip install -q matplotlib seaborn\n",
            "!pip install -q numpy\n\n",
            "print(\"✅ All dependencies installed successfully!\")"
        ]
    })
    
    # Basic imports with fixed PettingZoo import
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "import_header"},
        "source": ["## 🔧 Step 2: 基本インポート"]
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
            "# Gymnasium and PettingZoo (修正済み)\n",
            "import gymnasium as gym\n",
            "from gymnasium import spaces\n",
            "from pettingzoo import AECEnv\n",
            "from pettingzoo.utils import AgentSelector, wrappers  # Fixed: AgentSelector instead of agent_selector\n\n",
            "# Pygame (optional)\n",
            "try:\n",
            "    import pygame\n",
            "    PYGAME_AVAILABLE = True\n",
            "except ImportError:\n",
            "    PYGAME_AVAILABLE = False\n",
            "    print(\"⚠️ Pygame not available. Rendering disabled.\")\n\n",
            "# Set seeds for reproducibility\n",
            "np.random.seed(42)\n",
            "torch.manual_seed(42)\n",
            "if torch.cuda.is_available():\n",
            "    torch.cuda.manual_seed(42)\n\n",
            "# Matplotlib settings\n",
            "plt.style.use('seaborn-v0_8-darkgrid')\n",
            "sns.set_palette(\"husl\")\n\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f\"✅ PyTorch version: {torch.__version__}\")\n",
            "print(f\"✅ Using device: {device}\")"
        ]
    })
    
    # Add all code sections
    sections = [
        ("⚙️ Step 3: 設定クラス", "config_", python_files['config.py']),
        ("🎯 Step 4: 物理エンジン", "physics_", python_files['physics.py']),
        ("🎮 Step 5: 観測・行動空間", "spaces_", python_files['spaces.py']),
        ("🏆 Step 6: 報酬システム", "rewards_", python_files['rewards.py']),
        ("🎨 Step 7: レンダラー", "renderer_", python_files['renderer.py']),
        ("🌍 Step 8: メイン環境", "env_", python_files['soccer_env.py']),
        ("🤖 Step 9: エージェント", "agents_", python_files['agents.py']),
        ("📚 Step 10: トレーニング", "trainers_", python_files['trainers.py']),
        ("🧪 Step 11: テスト関数", "test_", python_files['test_environment.py'])
    ]
    
    for title, id_prefix, code in sections:
        notebook['cells'].append({
            "cell_type": "markdown",
            "metadata": {"id": f"{id_prefix}header"},
            "source": [f"## {title}"]
        })
        
        notebook['cells'].append({
            "cell_type": "code",
            "metadata": {"id": f"{id_prefix}code"},
            "execution_count": None,
            "outputs": [],
            "source": code + f"\n\nprint(\"✅ Section completed: {title}\")"
        })
    
    # Execution section
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "exec_header"},
        "source": [
            "## 🚀 Step 12: 実行セクション\n\n",
            "**注意**: 上記のすべてのコードセルを実行してから以下を実行してください。"
        ]
    })
    
    # Test environment
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "test_env"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 環境のテスト\n",
            "print(\"🧪 Testing the environment...\")\n",
            "print(\"=\" * 60)\n\n",
            "try:\n",
            "    # 簡単なテスト\n",
            "    test_config = SoccerEnvironmentConfig()\n",
            "    test_env = make_soccer_env(test_config, render_mode=None, action_type=\"continuous\")\n",
            "    print(f\"✅ Environment created successfully!\")\n",
            "    print(f\"   Agents: {test_env.agents}\")\n",
            "    print(f\"   Observation space: {test_env.observation_spaces[test_env.agents[0]].shape}\")\n",
            "    print(f\"   Action space: {test_env.action_spaces[test_env.agents[0]].shape}\")\n",
            "    \n",
            "    # Reset test\n",
            "    test_env.reset()\n",
            "    print(f\"✅ Environment reset successful!\")\n",
            "    \n",
            "    # Step test\n",
            "    for agent in test_env.agents:\n",
            "        action = test_env.action_spaces[agent].sample()\n",
            "        test_env.step(action)\n",
            "        break  # Just test one step\n",
            "    print(f\"✅ Environment step successful!\")\n",
            "    \n",
            "    test_env.close()\n",
            "    print(f\"\\n✅ All basic tests passed!\")\n",
            "    \n",
            "except Exception as e:\n",
            "    print(f\"❌ Error during testing: {e}\")\n",
            "    print(\"Please make sure all previous cells have been executed.\")"
        ]
    })
    
    # Run baseline
    notebook['cells'].append({
        "cell_type": "code",
        "metadata": {"id": "run_baseline"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# ベースライン実行（ランダムエージェント）\n",
            "print(\"🎮 Running baseline with random agents...\")\n",
            "print(\"=\" * 60)\n\n",
            "# 環境とエージェントの作成\n",
            "config = SoccerEnvironmentConfig()\n",
            "env = make_soccer_env(config, render_mode=None, action_type=\"continuous\")\n\n",
            "# ランダムエージェントの作成\n",
            "random_agents = {}\n",
            "for i, agent_name in enumerate(env.agents):\n",
            "    random_agents[agent_name] = RandomAgent(i, action_space_size=5, action_type=\"continuous\")\n\n",
            "# 5エピソード実行\n",
            "episode_results = []\n",
            "for episode in range(5):\n",
            "    env.reset()\n",
            "    episode_reward = 0\n",
            "    steps = 0\n",
            "    \n",
            "    while not all(env.terminations.values()) and not all(env.truncations.values()):\n",
            "        for agent in env.agents:\n",
            "            if not env.terminations.get(agent, False) and not env.truncations.get(agent, False):\n",
            "                obs = env.observe(agent)\n",
            "                action = random_agents[agent].select_action(obs, training=False)\n",
            "                env.step(action)\n",
            "                episode_reward += env.rewards.get(agent, 0)\n",
            "                steps += 1\n",
            "                \n",
            "                if env.terminations.get(agent, False) or env.truncations.get(agent, False):\n",
            "                    break\n",
            "    \n",
            "    episode_results.append({\n",
            "        'steps': steps,\n",
            "        'reward': episode_reward,\n",
            "        'scores': env.scores.copy()\n",
            "    })\n",
            "    \n",
            "    print(f\"Episode {episode + 1}: Steps={steps}, Scores={env.scores}, Reward={episode_reward:.2f}\")\n\n",
            "# 統計\n",
            "avg_steps = np.mean([r['steps'] for r in episode_results])\n",
            "avg_reward = np.mean([r['reward'] for r in episode_results])\n",
            "print(f\"\\n📊 Statistics:\")\n",
            "print(f\"   Average steps: {avg_steps:.1f}\")\n",
            "print(f\"   Average reward: {avg_reward:.2f}\")\n\n",
            "env.close()\n",
            "print(\"\\n✅ Baseline completed successfully!\")"
        ]
    })
    
    # Summary
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "summary"},
        "source": [
            "## 📝 まとめ\n\n",
            "### ✅ 実装完了\n",
            "- 完全な物理エンジンとPettingZoo互換環境\n",
            "- Random, DQN, MADDPG エージェント\n",
            "- 訓練フレームワーク\n\n",
            "### 🔧 修正済みの問題\n",
            "- ModuleNotFoundError: 内部import削除\n",
            "- TypeError (agent_selector): AgentSelectorに修正\n",
            "- 実行順序の依存関係\n\n",
            "### 📚 次のステップ\n",
            "1. より長いエピソードで訓練\n",
            "2. ハイパーパラメータの調整\n",
            "3. 学習曲線の分析\n\n",
            "**Happy Training! 🎮**"
        ]
    })
    
    return notebook

# Create and save
if __name__ == "__main__":
    notebook = create_final_fixed_notebook()
    
    with open("/home/user/webapp/multiagents_soccer_colab_final.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Final fixed notebook created!")
    print("📁 Saved as: multiagents_soccer_colab_final.ipynb")
    print("\n🔧 主な修正:")
    print("  1. agent_selector → AgentSelector に変更")
    print("  2. すべての内部import削除")
    print("  3. 実行順序を明確化")
    print("  4. エラーハンドリング追加")