#!/usr/bin/env python3
"""
Add Behavioral Cloning learning to multiagents_soccer_20sec_improved_physics_final.ipynb
20秒エピソードで100回の学習を追加
"""

import json

def add_bc_learning():
    """Add BC learning to the notebook"""
    
    # Read the notebook
    with open('multiagents_soccer_20sec_improved_physics_final.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the last training/execution cell and add BC learning after it
    insertion_index = -1
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'run_match' in source or 'trainer.run_match' in source or 'match_stats' in source:
                insertion_index = i + 1
    
    if insertion_index == -1:
        insertion_index = len(notebook['cells'])
    
    # Create BC learning cells
    bc_cells = []
    
    # Add BC header
    bc_cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "bc_learning_header"},
        "source": [
            "## 🎓 Behavioral Cloning (BC) Learning from Expert\n",
            "## エキスパートからの模倣学習\n\n",
            "エキスパートエージェントのプレイデータを収集し、Behavioral Cloningで学習します。\n",
            "20秒のエピソードを100回実行して学習データを収集します。"
        ]
    })
    
    # Add BC Agent implementation
    bc_cells.append({
        "cell_type": "code",
        "metadata": {"id": "bc_agent_implementation"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 🤖 Behavioral Cloning Agent\n",
            "class BCAgent(BaseAgent):\n",
            "    \"\"\"Agent that learns from expert demonstrations using Behavioral Cloning\"\"\"\n",
            "    \n",
            "    def __init__(self, agent_id: int, action_dim: int, observation_dim: int = 28,\n",
            "                 hidden_dim: int = 128, lr: float = 0.001):\n",
            "        super().__init__(agent_id, action_dim)\n",
            "        self.observation_dim = observation_dim\n",
            "        self.hidden_dim = hidden_dim\n",
            "        self.lr = lr\n",
            "        \n",
            "        # Build neural network for behavior cloning\n",
            "        self.policy_net = nn.Sequential(\n",
            "            nn.Linear(observation_dim, hidden_dim),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(hidden_dim, hidden_dim),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(hidden_dim, action_dim),\n",
            "            nn.Tanh()  # Actions are in [-1, 1]\n",
            "        )\n",
            "        \n",
            "        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)\n",
            "        self.loss_fn = nn.MSELoss()\n",
            "        \n",
            "        # Storage for demonstrations\n",
            "        self.demonstrations = []\n",
            "        \n",
            "    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:\n",
            "        \"\"\"Select action using learned policy\"\"\"\n",
            "        with torch.no_grad():\n",
            "            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)\n",
            "            action = self.policy_net(obs_tensor).squeeze(0).numpy()\n",
            "        \n",
            "        # Add small noise during training for exploration\n",
            "        if training:\n",
            "            action += np.random.randn(self.action_dim) * 0.1\n",
            "        \n",
            "        return np.clip(action, -1, 1)\n",
            "    \n",
            "    def add_demonstration(self, observation: np.ndarray, action: np.ndarray):\n",
            "        \"\"\"Add expert demonstration to buffer\"\"\"\n",
            "        self.demonstrations.append((observation, action))\n",
            "    \n",
            "    def train_on_demonstrations(self, batch_size: int = 64, epochs: int = 10):\n",
            "        \"\"\"Train on collected demonstrations\"\"\"\n",
            "        if len(self.demonstrations) < batch_size:\n",
            "            print(f\"Not enough demonstrations: {len(self.demonstrations)} < {batch_size}\")\n",
            "            return {}\n",
            "        \n",
            "        dataset_size = len(self.demonstrations)\n",
            "        losses = []\n",
            "        \n",
            "        for epoch in range(epochs):\n",
            "            epoch_loss = 0.0\n",
            "            num_batches = 0\n",
            "            \n",
            "            # Shuffle demonstrations\n",
            "            indices = np.random.permutation(dataset_size)\n",
            "            \n",
            "            for i in range(0, dataset_size - batch_size, batch_size):\n",
            "                batch_indices = indices[i:i+batch_size]\n",
            "                \n",
            "                # Prepare batch\n",
            "                obs_batch = []\n",
            "                action_batch = []\n",
            "                for idx in batch_indices:\n",
            "                    obs, act = self.demonstrations[idx]\n",
            "                    obs_batch.append(obs)\n",
            "                    action_batch.append(act)\n",
            "                \n",
            "                obs_tensor = torch.FloatTensor(np.array(obs_batch))\n",
            "                action_tensor = torch.FloatTensor(np.array(action_batch))\n",
            "                \n",
            "                # Forward pass\n",
            "                predicted_actions = self.policy_net(obs_tensor)\n",
            "                loss = self.loss_fn(predicted_actions, action_tensor)\n",
            "                \n",
            "                # Backward pass\n",
            "                self.optimizer.zero_grad()\n",
            "                loss.backward()\n",
            "                self.optimizer.step()\n",
            "                \n",
            "                epoch_loss += loss.item()\n",
            "                num_batches += 1\n",
            "            \n",
            "            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0\n",
            "            losses.append(avg_loss)\n",
            "            \n",
            "            if epoch % 2 == 0:\n",
            "                print(f\"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}\")\n",
            "        \n",
            "        return {\n",
            "            'final_loss': losses[-1] if losses else 0,\n",
            "            'avg_loss': np.mean(losses) if losses else 0,\n",
            "            'num_demonstrations': len(self.demonstrations)\n",
            "        }\n",
            "    \n",
            "    def learn(self, *args, **kwargs):\n",
            "        \"\"\"Compatibility method\"\"\"\n",
            "        return self.train_on_demonstrations()\n",
            "    \n",
            "    def save(self, path: str):\n",
            "        \"\"\"Save model\"\"\"\n",
            "        torch.save({\n",
            "            'policy_net': self.policy_net.state_dict(),\n",
            "            'optimizer': self.optimizer.state_dict(),\n",
            "            'demonstrations': self.demonstrations\n",
            "        }, path)\n",
            "    \n",
            "    def load(self, path: str):\n",
            "        \"\"\"Load model\"\"\"\n",
            "        checkpoint = torch.load(path)\n",
            "        self.policy_net.load_state_dict(checkpoint['policy_net'])\n",
            "        self.optimizer.load_state_dict(checkpoint['optimizer'])\n",
            "        self.demonstrations = checkpoint['demonstrations']\n",
            "\n",
            "print('✅ Behavioral Cloning Agent implemented')"
        ]
    })
    
    # Add data collection function
    bc_cells.append({
        "cell_type": "code",
        "metadata": {"id": "collect_demonstrations"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 📊 Collect Expert Demonstrations\n",
            "def collect_expert_demonstrations(num_episodes: int = 100, episode_length: int = 600):\n",
            "    \"\"\"Collect demonstrations from expert agents for BC learning\n",
            "    \n",
            "    Args:\n",
            "        num_episodes: Number of 20-second episodes to collect (default 100)\n",
            "        episode_length: Steps per episode (600 = 20 seconds)\n",
            "    \"\"\"\n",
            "    print(f\"📊 Collecting expert demonstrations...\")\n",
            "    print(f\"   Episodes: {num_episodes}\")\n",
            "    print(f\"   Episode length: {episode_length} steps (20 seconds)\\n\")\n",
            "    \n",
            "    # Create environment with 20-second episodes\n",
            "    env = SoccerEnvironment(extended_config, render_mode='rgb_array')\n",
            "    \n",
            "    # Create expert agents\n",
            "    expert_agents = {}\n",
            "    for i, agent in enumerate(env.possible_agents):\n",
            "        team = 0 if 'team_0' in agent else 1\n",
            "        expert_agents[agent] = EnhancedExpertAgent(i, team, extended_config)\n",
            "    \n",
            "    # Create BC agents to collect demonstrations\n",
            "    bc_agents = {}\n",
            "    for i, agent in enumerate(env.possible_agents):\n",
            "        bc_agents[agent] = BCAgent(i, action_dim=5, observation_dim=28)\n",
            "    \n",
            "    # Collect demonstrations\n",
            "    total_steps = 0\n",
            "    for episode in tqdm(range(num_episodes), desc=\"Collecting demonstrations\"):\n",
            "        observations, _ = env.reset()\n",
            "        \n",
            "        episode_steps = 0\n",
            "        while env.agents and episode_steps < episode_length:\n",
            "            agent_id = env.agent_selection\n",
            "            obs = observations[agent_id] if isinstance(observations, dict) else observations\n",
            "            \n",
            "            # Get expert action\n",
            "            expert_action = expert_agents[agent_id].select_action(obs, training=False)\n",
            "            \n",
            "            # Store demonstration\n",
            "            bc_agents[agent_id].add_demonstration(obs, expert_action)\n",
            "            \n",
            "            # Step environment\n",
            "            observations, rewards, terminations, truncations, infos = env.step(expert_action)\n",
            "            \n",
            "            episode_steps += 1\n",
            "            total_steps += 1\n",
            "        \n",
            "        if (episode + 1) % 20 == 0:\n",
            "            print(f\"  Episode {episode + 1}/{num_episodes} completed\")\n",
            "    \n",
            "    env.close()\n",
            "    \n",
            "    print(f\"\\n✅ Data collection complete!\")\n",
            "    print(f\"   Total steps: {total_steps:,}\")\n",
            "    print(f\"   Demonstrations per agent:\")\n",
            "    for agent_name, bc_agent in bc_agents.items():\n",
            "        print(f\"     {agent_name}: {len(bc_agent.demonstrations):,}\")\n",
            "    \n",
            "    return bc_agents\n",
            "\n",
            "print('✅ Demonstration collection function ready')"
        ]
    })
    
    # Add BC training execution
    bc_cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "bc_training_header"},
        "source": [
            "### 🎯 Execute BC Learning (100 episodes × 20 seconds)"
        ]
    })
    
    bc_cells.append({
        "cell_type": "code",
        "metadata": {"id": "execute_bc_learning"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 🎓 Execute Behavioral Cloning Learning\n",
            "print(\"=\"*60)\n",
            "print(\"🎓 BEHAVIORAL CLONING LEARNING\")\n",
            "print(\"=\"*60)\n",
            "print(\"\\n📚 Phase 1: Collect Expert Demonstrations\\n\")\n",
            "\n",
            "# Collect demonstrations from 100 episodes of 20 seconds each\n",
            "bc_agents = collect_expert_demonstrations(num_episodes=100, episode_length=600)\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"📖 Phase 2: Train BC Agents\\n\")\n",
            "\n",
            "# Train each BC agent on collected demonstrations\n",
            "training_results = {}\n",
            "for agent_name, bc_agent in bc_agents.items():\n",
            "    print(f\"\\n🤖 Training {agent_name}...\")\n",
            "    results = bc_agent.train_on_demonstrations(batch_size=64, epochs=10)\n",
            "    training_results[agent_name] = results\n",
            "    print(f\"  ✅ Training complete! Final loss: {results['final_loss']:.4f}\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"📊 TRAINING SUMMARY\")\n",
            "print(\"=\"*60)\n",
            "for agent_name, results in training_results.items():\n",
            "    print(f\"\\n{agent_name}:\")\n",
            "    print(f\"  Demonstrations: {results['num_demonstrations']:,}\")\n",
            "    print(f\"  Final Loss: {results['final_loss']:.4f}\")\n",
            "    print(f\"  Average Loss: {results['avg_loss']:.4f}\")\n",
            "\n",
            "print(\"\\n✅ Behavioral Cloning learning complete!\")\n",
            "print(\"   All agents have learned from 100 episodes of expert play.\")\n",
            "print(\"   Each episode was 20 seconds (600 steps).\")"
        ]
    })
    
    # Add evaluation
    bc_cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "bc_evaluation_header"},
        "source": [
            "### 🏆 Evaluate BC Agents vs Experts"
        ]
    })
    
    bc_cells.append({
        "cell_type": "code",
        "metadata": {"id": "evaluate_bc_agents"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 🏆 Evaluate learned BC agents\n",
            "def evaluate_bc_agents(bc_agents: Dict, num_episodes: int = 5):\n",
            "    \"\"\"Evaluate BC agents in 20-second matches\"\"\"\n",
            "    print(f\"\\n🏆 Evaluating BC Agents ({num_episodes} matches)\\n\")\n",
            "    \n",
            "    env = SoccerEnvironment(extended_config, render_mode='rgb_array')\n",
            "    \n",
            "    total_goals = [0, 0]\n",
            "    total_rewards = defaultdict(float)\n",
            "    \n",
            "    for episode in range(num_episodes):\n",
            "        print(f\"Match {episode + 1}/{num_episodes}\")\n",
            "        observations, _ = env.reset()\n",
            "        \n",
            "        episode_rewards = defaultdict(float)\n",
            "        frames = []\n",
            "        \n",
            "        step_count = 0\n",
            "        while env.agents and step_count < 600:  # 20 seconds\n",
            "            agent_id = env.agent_selection\n",
            "            obs = observations[agent_id] if isinstance(observations, dict) else observations\n",
            "            \n",
            "            # Get BC agent action\n",
            "            action = bc_agents[agent_id].select_action(obs, training=False)\n",
            "            \n",
            "            # Step environment\n",
            "            observations, rewards, terminations, truncations, infos = env.step(action)\n",
            "            \n",
            "            # Track rewards\n",
            "            for agent, reward in rewards.items():\n",
            "                episode_rewards[agent] += reward\n",
            "            \n",
            "            # Capture frame periodically\n",
            "            if step_count % 30 == 0:  # Every second\n",
            "                frames.append(env.render())\n",
            "            \n",
            "            step_count += 1\n",
            "        \n",
            "        # Get final score\n",
            "        final_score = env.score if hasattr(env, 'score') else [0, 0]\n",
            "        total_goals[0] += final_score[0]\n",
            "        total_goals[1] += final_score[1]\n",
            "        \n",
            "        print(f\"  Score: Team 0: {final_score[0]} - Team 1: {final_score[1]}\")\n",
            "        \n",
            "        # Accumulate rewards\n",
            "        for agent, reward in episode_rewards.items():\n",
            "            total_rewards[agent] += reward\n",
            "        \n",
            "        # Save video for first and last episode\n",
            "        if episode == 0 or episode == num_episodes - 1:\n",
            "            video_name = f'bc_match_{episode + 1}.mp4'\n",
            "            if frames:\n",
            "                save_video(frames, video_name, fps=1)  # 1 fps since we capture every second\n",
            "                print(f\"  Video saved: {video_name}\")\n",
            "    \n",
            "    env.close()\n",
            "    \n",
            "    print(\"\\n\" + \"=\"*60)\n",
            "    print(\"📊 EVALUATION RESULTS\")\n",
            "    print(\"=\"*60)\n",
            "    print(f\"\\nTotal Goals:\")\n",
            "    print(f\"  Team 0: {total_goals[0]}\")\n",
            "    print(f\"  Team 1: {total_goals[1]}\")\n",
            "    print(f\"\\nAverage Goals per Match:\")\n",
            "    print(f\"  Team 0: {total_goals[0] / num_episodes:.2f}\")\n",
            "    print(f\"  Team 1: {total_goals[1] / num_episodes:.2f}\")\n",
            "    print(f\"\\nAverage Rewards:\")\n",
            "    for agent in sorted(total_rewards.keys()):\n",
            "        avg_reward = total_rewards[agent] / num_episodes\n",
            "        print(f\"  {agent}: {avg_reward:.2f}\")\n",
            "    \n",
            "    return total_goals, total_rewards\n",
            "\n",
            "# Run evaluation\n",
            "print(\"🎮 Starting BC agent evaluation...\")\n",
            "bc_goals, bc_rewards = evaluate_bc_agents(bc_agents, num_episodes=5)\n",
            "print(\"\\n✅ Evaluation complete! BC agents have been trained and tested.\")"
        ]
    })
    
    # Insert BC cells into notebook
    for cell in bc_cells:
        notebook['cells'].insert(insertion_index, cell)
        insertion_index += 1
    
    # Save the updated notebook
    output_file = 'multiagents_soccer_20sec_bc_learning.ipynb'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created notebook with BC learning: {output_file}")
    print("\n📋 Added features:")
    print("  1. 🤖 BCAgent class with neural network policy")
    print("  2. 📊 Expert demonstration collection (100 episodes × 20 seconds)")
    print("  3. 🎓 Behavioral Cloning training with MSE loss")
    print("  4. 🏆 Evaluation of learned agents")
    print("  5. 📹 Video recording of BC agent matches")
    
    return output_file

if __name__ == "__main__":
    add_bc_learning()