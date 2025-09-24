#!/usr/bin/env python3
"""
Create extended training and visualization notebook sections
"""

import json

def create_training_visualization_cells():
    """Create notebook cells for extended training and visualization"""
    
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "extended_training_header"},
        "source": [
            "# 🚀 Extended Training and Visualization\n",
            "## より長いエピソードでの訓練と動画可視化\n\n",
            "このセクションでは、より長いエピソードでエージェントを訓練し、結果を動画として可視化します。"
        ]
    })
    
    # Install additional dependencies for video
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "video_deps_header"},
        "source": ["### 📦 動画作成用の追加ライブラリ"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "install_video_deps"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 動画作成用ライブラリのインストール\n",
            "!apt-get update -qq\n",
            "!apt-get install -qq xvfb\n",
            "!pip install -q imageio imageio-ffmpeg\n",
            "!pip install -q pyvirtualdisplay\n",
            "\n",
            "import imageio\n",
            "from IPython.display import HTML, display\n",
            "import base64\n",
            "\n",
            "# Virtual display for rendering\n",
            "from pyvirtualdisplay import Display\n",
            "display_virtual = Display(visible=0, size=(1400, 900))\n",
            "display_virtual.start()\n",
            "\n",
            "print(\"✅ Video dependencies installed successfully!\")"
        ]
    })
    
    # Extended training implementation
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "extended_trainer_header"},
        "source": ["### 📚 拡張訓練クラスの実装"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "extended_trainer_code"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "class ExtendedTrainer:\n",
            "    \"\"\"Extended trainer with video recording capabilities\"\"\"\n",
            "    \n",
            "    def __init__(self, env_config, agent_type=\"dqn\"):\n",
            "        self.env_config = env_config\n",
            "        self.agent_type = agent_type\n",
            "        self.episode_rewards = []\n",
            "        self.episode_lengths = []\n",
            "        self.scores_history = []\n",
            "        self.video_frames = []\n",
            "        \n",
            "    def create_agents(self, env):\n",
            "        \"\"\"Create agents based on type\"\"\"\n",
            "        agents = {}\n",
            "        \n",
            "        if self.agent_type == \"random\":\n",
            "            for i, agent_name in enumerate(env.agents):\n",
            "                agents[agent_name] = RandomAgent(i, action_space_size=5, action_type=\"continuous\")\n",
            "        \n",
            "        elif self.agent_type == \"dqn\":\n",
            "            for i, agent_name in enumerate(env.agents):\n",
            "                agents[agent_name] = DQNAgent(\n",
            "                    agent_id=i,\n",
            "                    obs_dim=28,\n",
            "                    action_dim=9,\n",
            "                    hidden_dims=(256, 128),\n",
            "                    lr=1e-3,\n",
            "                    gamma=0.99,\n",
            "                    epsilon=1.0,\n",
            "                    epsilon_decay=0.995,\n",
            "                    epsilon_min=0.01,\n",
            "                    buffer_size=10000,\n",
            "                    batch_size=64\n",
            "                )\n",
            "        \n",
            "        elif self.agent_type == \"maddpg\":\n",
            "            maddpg_config = MADDPGConfig()\n",
            "            for i, agent_name in enumerate(env.agents):\n",
            "                agents[agent_name] = MADDPGAgent(i, maddpg_config)\n",
            "        \n",
            "        return agents\n",
            "    \n",
            "    def train(self, num_episodes=100, record_video_every=20, max_video_episodes=5):\n",
            "        \"\"\"Train agents and record videos\"\"\"\n",
            "        print(f\"🎮 Starting extended training with {self.agent_type} agents\")\n",
            "        print(f\"   Episodes: {num_episodes}\")\n",
            "        print(f\"   Recording video every {record_video_every} episodes\")\n",
            "        print(\"=\" * 60)\n",
            "        \n",
            "        # Create environments\n",
            "        env = make_soccer_env(self.env_config, render_mode=None, action_type=\"continuous\")\n",
            "        render_env = make_soccer_env(self.env_config, render_mode=\"rgb_array\", action_type=\"continuous\")\n",
            "        \n",
            "        # Create agents\n",
            "        agents = self.create_agents(env)\n",
            "        \n",
            "        videos = []  # Store video data\n",
            "        \n",
            "        for episode in range(num_episodes):\n",
            "            # Determine if we should record this episode\n",
            "            record_this_episode = (episode % record_video_every == 0) and (len(videos) < max_video_episodes)\n",
            "            \n",
            "            # Use render environment if recording\n",
            "            current_env = render_env if record_this_episode else env\n",
            "            current_env.reset()\n",
            "            \n",
            "            episode_reward = {agent: 0 for agent in current_env.agents}\n",
            "            episode_frames = []\n",
            "            steps = 0\n",
            "            \n",
            "            # Store experiences for learning (DQN)\n",
            "            episode_experiences = {agent: [] for agent in current_env.agents}\n",
            "            \n",
            "            while not all(current_env.terminations.values()) and not all(current_env.truncations.values()):\n",
            "                # Record frame if needed\n",
            "                if record_this_episode:\n",
            "                    frame = current_env.render()\n",
            "                    if frame is not None:\n",
            "                        episode_frames.append(frame)\n",
            "                \n",
            "                for agent_name in current_env.agents:\n",
            "                    if not current_env.terminations.get(agent_name, False) and not current_env.truncations.get(agent_name, False):\n",
            "                        # Get observation and action\n",
            "                        obs = current_env.observe(agent_name)\n",
            "                        \n",
            "                        if self.agent_type == \"dqn\":\n",
            "                            action = agents[agent_name].select_action(obs, training=True)\n",
            "                            # Convert discrete to continuous\n",
            "                            action_space = ActionSpace(\"discrete\")\n",
            "                            action_continuous = action_space.convert_discrete_to_continuous(action)\n",
            "                            current_env.step(action_continuous)\n",
            "                            \n",
            "                            # Store experience\n",
            "                            next_obs = current_env.observe(agent_name)\n",
            "                            reward = current_env.rewards.get(agent_name, 0)\n",
            "                            done = current_env.terminations.get(agent_name, False) or current_env.truncations.get(agent_name, False)\n",
            "                            \n",
            "                            agents[agent_name].store_experience(obs, action, reward, next_obs, done)\n",
            "                            \n",
            "                            # Learn from experience\n",
            "                            if len(agents[agent_name].replay_buffer) > agents[agent_name].batch_size:\n",
            "                                agents[agent_name].learn()\n",
            "                        else:\n",
            "                            action = agents[agent_name].select_action(obs, training=True)\n",
            "                            current_env.step(action)\n",
            "                            reward = current_env.rewards.get(agent_name, 0)\n",
            "                        \n",
            "                        episode_reward[agent_name] += reward\n",
            "                        steps += 1\n",
            "                        \n",
            "                        if current_env.terminations.get(agent_name, False) or current_env.truncations.get(agent_name, False):\n",
            "                            break\n",
            "            \n",
            "            # Save video if recorded\n",
            "            if record_this_episode and episode_frames:\n",
            "                videos.append({\n",
            "                    'episode': episode,\n",
            "                    'frames': episode_frames,\n",
            "                    'scores': current_env.scores.copy(),\n",
            "                    'reward': sum(episode_reward.values())\n",
            "                })\n",
            "                print(f\"📹 Recorded video for episode {episode}\")\n",
            "            \n",
            "            # Update target networks for DQN\n",
            "            if self.agent_type == \"dqn\" and episode % 10 == 0:\n",
            "                for agent_name in agents:\n",
            "                    agents[agent_name].update_target_network()\n",
            "            \n",
            "            # Store metrics\n",
            "            self.episode_rewards.append(sum(episode_reward.values()))\n",
            "            self.episode_lengths.append(steps)\n",
            "            self.scores_history.append(current_env.scores.copy())\n",
            "            \n",
            "            # Print progress\n",
            "            if episode % 10 == 0:\n",
            "                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)\n",
            "                print(f\"Episode {episode}: Avg Reward (last 10): {avg_reward:.2f}, Scores: {current_env.scores}\")\n",
            "        \n",
            "        env.close()\n",
            "        render_env.close()\n",
            "        \n",
            "        return videos\n",
            "\n",
            "print(\"✅ Extended trainer class defined!\")"
        ]
    })
    
    # Run extended training
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "run_extended_header"},
        "source": ["### 🎮 拡張訓練の実行"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "run_extended_training"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 拡張訓練の実行\n",
            "print(\"🚀 Starting extended training...\")\n",
            "print(\"This will take a few minutes. Please be patient.\")\n",
            "print(\"=\" * 60)\n",
            "\n",
            "# Configuration\n",
            "config = SoccerEnvironmentConfig()\n",
            "config.MAX_STEPS = 500  # Shorter episodes for faster training\n",
            "\n",
            "# Create trainer\n",
            "trainer = ExtendedTrainer(config, agent_type=\"random\")  # Start with random for quick results\n",
            "\n",
            "# Train and record videos\n",
            "videos = trainer.train(\n",
            "    num_episodes=50,      # Total episodes\n",
            "    record_video_every=10, # Record every 10 episodes\n",
            "    max_video_episodes=5   # Maximum 5 videos\n",
            ")\n",
            "\n",
            "print(f\"\\n✅ Training completed!\")\n",
            "print(f\"   Total episodes: {len(trainer.episode_rewards)}\")\n",
            "print(f\"   Videos recorded: {len(videos)}\")\n",
            "print(f\"   Average reward: {np.mean(trainer.episode_rewards):.2f}\")\n",
            "print(f\"   Average episode length: {np.mean(trainer.episode_lengths):.1f}\")"
        ]
    })
    
    # Visualize training results
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "viz_results_header"},
        "source": ["### 📊 訓練結果の可視化"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "visualize_results"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 訓練結果の可視化\n",
            "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
            "\n",
            "# Episode rewards\n",
            "ax = axes[0, 0]\n",
            "ax.plot(trainer.episode_rewards, alpha=0.3, label='Raw')\n",
            "if len(trainer.episode_rewards) > 10:\n",
            "    smoothed = np.convolve(trainer.episode_rewards, np.ones(10)/10, mode='valid')\n",
            "    ax.plot(range(9, len(trainer.episode_rewards)), smoothed, linewidth=2, label='Smoothed (10-ep)')\n",
            "ax.set_title('Episode Rewards Over Time')\n",
            "ax.set_xlabel('Episode')\n",
            "ax.set_ylabel('Total Reward')\n",
            "ax.grid(True, alpha=0.3)\n",
            "ax.legend()\n",
            "\n",
            "# Episode lengths\n",
            "ax = axes[0, 1]\n",
            "ax.plot(trainer.episode_lengths, alpha=0.5, color='orange')\n",
            "ax.set_title('Episode Lengths')\n",
            "ax.set_xlabel('Episode')\n",
            "ax.set_ylabel('Steps')\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "# Team scores over time\n",
            "ax = axes[1, 0]\n",
            "team_0_scores = [score[0] for score in trainer.scores_history]\n",
            "team_1_scores = [score[1] for score in trainer.scores_history]\n",
            "ax.plot(team_0_scores, label='Team 0 (Blue)', alpha=0.7, color='blue')\n",
            "ax.plot(team_1_scores, label='Team 1 (Red)', alpha=0.7, color='red')\n",
            "ax.set_title('Team Scores Over Episodes')\n",
            "ax.set_xlabel('Episode')\n",
            "ax.set_ylabel('Goals Scored')\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "# Win rate analysis\n",
            "ax = axes[1, 1]\n",
            "wins_0 = sum(1 for s in trainer.scores_history if s[0] > s[1])\n",
            "wins_1 = sum(1 for s in trainer.scores_history if s[1] > s[0])\n",
            "draws = len(trainer.scores_history) - wins_0 - wins_1\n",
            "\n",
            "labels = ['Team 0 Wins', 'Team 1 Wins', 'Draws']\n",
            "sizes = [wins_0, wins_1, draws]\n",
            "colors = ['#3498db', '#e74c3c', '#95a5a6']\n",
            "ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)\n",
            "ax.set_title('Win Rate Distribution')\n",
            "\n",
            "plt.suptitle(f'Training Results - {trainer.agent_type.upper()} Agents', fontsize=16, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"\\n📈 Statistics Summary:\")\n",
            "print(f\"   Team 0 wins: {wins_0} ({wins_0/len(trainer.scores_history)*100:.1f}%)\")\n",
            "print(f\"   Team 1 wins: {wins_1} ({wins_1/len(trainer.scores_history)*100:.1f}%)\")\n",
            "print(f\"   Draws: {draws} ({draws/len(trainer.scores_history)*100:.1f}%)\")\n",
            "print(f\"   Max reward: {max(trainer.episode_rewards):.2f}\")\n",
            "print(f\"   Min reward: {min(trainer.episode_rewards):.2f}\")"
        ]
    })
    
    # Create and display videos
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "create_videos_header"},
        "source": ["### 🎬 動画の作成と表示"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "create_videos"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "def create_video_from_frames(frames, output_path, fps=30):\n",
            "    \"\"\"Create video from frames\"\"\"\n",
            "    if not frames:\n",
            "        print(\"No frames to create video\")\n",
            "        return None\n",
            "    \n",
            "    # Convert frames to proper format\n",
            "    processed_frames = []\n",
            "    for frame in frames:\n",
            "        if frame.dtype != np.uint8:\n",
            "            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)\n",
            "        processed_frames.append(frame)\n",
            "    \n",
            "    # Create video\n",
            "    imageio.mimsave(output_path, processed_frames, fps=fps)\n",
            "    return output_path\n",
            "\n",
            "def display_video(video_path):\n",
            "    \"\"\"Display video in Colab\"\"\"\n",
            "    video = open(video_path, 'rb').read()\n",
            "    encoded = base64.b64encode(video).decode('ascii')\n",
            "    html_code = f'''\n",
            "    <video width=\"800\" height=\"600\" controls>\n",
            "        <source src=\"data:video/mp4;base64,{encoded}\" type=\"video/mp4\">\n",
            "    </video>\n",
            "    '''\n",
            "    return HTML(html_code)\n",
            "\n",
            "# Create and display videos\n",
            "print(\"🎬 Creating videos from recorded episodes...\")\n",
            "print(\"=\" * 60)\n",
            "\n",
            "video_paths = []\n",
            "for i, video_data in enumerate(videos):\n",
            "    output_path = f'/tmp/soccer_episode_{video_data[\"episode\"]}.mp4'\n",
            "    \n",
            "    # Create video\n",
            "    create_video_from_frames(video_data['frames'], output_path, fps=30)\n",
            "    video_paths.append(output_path)\n",
            "    \n",
            "    print(f\"✅ Created video {i+1}: Episode {video_data['episode']}\")\n",
            "    print(f\"   Scores: {video_data['scores']}\")\n",
            "    print(f\"   Total Reward: {video_data['reward']:.2f}\")\n",
            "    print()\n",
            "\n",
            "print(f\"\\n🎥 Videos saved to: /tmp/\")\n",
            "print(\"Use the next cell to display videos\")"
        ]
    })
    
    # Display videos
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "display_videos"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 動画の表示\n",
            "if videos and len(video_paths) > 0:\n",
            "    print(f\"📺 Displaying video from Episode {videos[0]['episode']}\")\n",
            "    print(f\"   Scores: Blue {videos[0]['scores'][0]} - Red {videos[0]['scores'][1]}\")\n",
            "    display(display_video(video_paths[0]))\n",
            "else:\n",
            "    print(\"No videos available to display.\")\n",
            "    print(\"Please run the training cell first.\")"
        ]
    })
    
    # Advanced training with DQN
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "advanced_training_header"},
        "source": ["### 🧠 DQNエージェントでの高度な訓練"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "advanced_dqn_training"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# DQNエージェントでのより長い訓練\n",
            "print(\"🧠 Starting advanced DQN training...\")\n",
            "print(\"This will take longer but show learning progress.\")\n",
            "print(\"=\" * 60)\n",
            "\n",
            "# Configuration for longer training\n",
            "config_dqn = SoccerEnvironmentConfig()\n",
            "config_dqn.MAX_STEPS = 300  # Balanced episode length\n",
            "\n",
            "# Create DQN trainer\n",
            "dqn_trainer = ExtendedTrainer(config_dqn, agent_type=\"dqn\")\n",
            "\n",
            "# Train with DQN agents\n",
            "dqn_videos = dqn_trainer.train(\n",
            "    num_episodes=200,      # More episodes for learning\n",
            "    record_video_every=40, # Record every 40 episodes to see progress\n",
            "    max_video_episodes=5   # Record 5 videos total\n",
            ")\n",
            "\n",
            "print(f\"\\n✅ DQN Training completed!\")\n",
            "print(f\"   Total episodes: {len(dqn_trainer.episode_rewards)}\")\n",
            "print(f\"   Videos recorded: {len(dqn_videos)}\")\n",
            "print(f\"   Final avg reward (last 20): {np.mean(dqn_trainer.episode_rewards[-20:]):.2f}\")\n",
            "print(f\"   Initial avg reward (first 20): {np.mean(dqn_trainer.episode_rewards[:20]):.2f}\")\n",
            "print(f\"   Improvement: {np.mean(dqn_trainer.episode_rewards[-20:]) - np.mean(dqn_trainer.episode_rewards[:20]):.2f}\")"
        ]
    })
    
    # Compare results
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "compare_results"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 訓練結果の比較\n",
            "if 'dqn_trainer' in globals():\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
            "    \n",
            "    # Rewards comparison\n",
            "    ax = axes[0]\n",
            "    \n",
            "    # Random agent rewards\n",
            "    random_rewards_smooth = np.convolve(trainer.episode_rewards, np.ones(10)/10, mode='valid')\n",
            "    ax.plot(range(9, len(trainer.episode_rewards)), random_rewards_smooth, \n",
            "            label='Random Agents', alpha=0.7, color='gray')\n",
            "    \n",
            "    # DQN agent rewards\n",
            "    dqn_rewards_smooth = np.convolve(dqn_trainer.episode_rewards, np.ones(10)/10, mode='valid')\n",
            "    ax.plot(range(9, len(dqn_trainer.episode_rewards)), dqn_rewards_smooth, \n",
            "            label='DQN Agents', linewidth=2, color='green')\n",
            "    \n",
            "    ax.set_title('Learning Progress Comparison')\n",
            "    ax.set_xlabel('Episode')\n",
            "    ax.set_ylabel('Average Reward (10-ep smoothed)')\n",
            "    ax.legend()\n",
            "    ax.grid(True, alpha=0.3)\n",
            "    \n",
            "    # Score distribution\n",
            "    ax = axes[1]\n",
            "    \n",
            "    # Calculate average scores for last 20 episodes\n",
            "    random_scores = trainer.scores_history[-20:] if len(trainer.scores_history) >= 20 else trainer.scores_history\n",
            "    dqn_scores = dqn_trainer.scores_history[-20:] if len(dqn_trainer.scores_history) >= 20 else dqn_trainer.scores_history\n",
            "    \n",
            "    random_avg = [np.mean([s[0] for s in random_scores]), np.mean([s[1] for s in random_scores])]\n",
            "    dqn_avg = [np.mean([s[0] for s in dqn_scores]), np.mean([s[1] for s in dqn_scores])]\n",
            "    \n",
            "    x = np.arange(2)\n",
            "    width = 0.35\n",
            "    \n",
            "    ax.bar(x - width/2, random_avg, width, label='Random', color='gray', alpha=0.7)\n",
            "    ax.bar(x + width/2, dqn_avg, width, label='DQN', color='green', alpha=0.7)\n",
            "    \n",
            "    ax.set_title('Average Goals Scored (Last 20 Episodes)')\n",
            "    ax.set_xticks(x)\n",
            "    ax.set_xticklabels(['Team 0 (Blue)', 'Team 1 (Red)'])\n",
            "    ax.set_ylabel('Average Goals')\n",
            "    ax.legend()\n",
            "    ax.grid(True, alpha=0.3, axis='y')\n",
            "    \n",
            "    plt.suptitle('Random vs DQN Agent Performance', fontsize=16, fontweight='bold')\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    print(\"\\n📊 Performance Comparison:\")\n",
            "    print(f\"Random Agents - Avg Reward: {np.mean(trainer.episode_rewards):.2f} ± {np.std(trainer.episode_rewards):.2f}\")\n",
            "    print(f\"DQN Agents - Avg Reward: {np.mean(dqn_trainer.episode_rewards):.2f} ± {np.std(dqn_trainer.episode_rewards):.2f}\")\n",
            "else:\n",
            "    print(\"Please run DQN training first to see comparison.\")"
        ]
    })
    
    # Display DQN videos
    cells.append({
        "cell_type": "code",
        "metadata": {"id": "display_dqn_videos"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# DQN訓練の動画表示\n",
            "if 'dqn_videos' in globals() and dqn_videos:\n",
            "    print(\"🎬 Creating DQN training videos...\")\n",
            "    \n",
            "    dqn_video_paths = []\n",
            "    for i, video_data in enumerate(dqn_videos):\n",
            "        output_path = f'/tmp/dqn_episode_{video_data[\"episode\"]}.mp4'\n",
            "        create_video_from_frames(video_data['frames'], output_path, fps=30)\n",
            "        dqn_video_paths.append(output_path)\n",
            "        print(f\"✅ Created DQN video {i+1}: Episode {video_data['episode']}\")\n",
            "    \n",
            "    # Display comparison: early vs late training\n",
            "    print(\"\\n📺 Early Training (Episode {}):\".format(dqn_videos[0]['episode']))\n",
            "    display(display_video(dqn_video_paths[0]))\n",
            "    \n",
            "    if len(dqn_video_paths) > 1:\n",
            "        print(\"\\n📺 Later Training (Episode {}):\".format(dqn_videos[-1]['episode']))\n",
            "        display(display_video(dqn_video_paths[-1]))\n",
            "else:\n",
            "    print(\"No DQN videos available. Please run DQN training first.\")"
        ]
    })
    
    return cells

# Create the extended notebook
if __name__ == "__main__":
    # Load the existing final notebook
    with open("/home/user/webapp/multiagents_soccer_colab_final.ipynb", "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Add new cells
    new_cells = create_training_visualization_cells()
    notebook['cells'].extend(new_cells)
    
    # Save as new extended notebook
    with open("/home/user/webapp/multiagents_soccer_extended.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Extended training and visualization notebook created!")
    print("📁 Saved as: multiagents_soccer_extended.ipynb")
    print("\n📋 新機能:")
    print("  1. より長いエピソードでの訓練（50-200エピソード）")
    print("  2. 動画記録機能（訓練中の gameplay を記録）")
    print("  3. 訓練結果の詳細な可視化")
    print("  4. Random vs DQN の性能比較")
    print("  5. 学習曲線と統計分析")