#!/usr/bin/env python3
"""
Fix the DQNAgent training error in the notebook
"""

import json

def create_fixed_dqn_agent_code():
    """Create fixed DQNAgent code"""
    
    fixed_code = '''class DQNAgent(BaseAgent):
    """Deep Q-Network agent - Fixed version"""

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

    def learn(self, experiences: List = None, training: bool = True) -> Dict[str, float]:
        """Learn from experiences in replay buffer - FIXED"""
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

        # Update epsilon (only if training)
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

# Replace the DQNAgent class with the fixed version
print("✅ DQNAgent class fixed - 'training' parameter added to learn method")'''
    
    return fixed_code

def create_fixed_extended_notebook():
    """Create notebook with all fixes"""
    
    # Load the extended notebook
    with open("/home/user/webapp/multiagents_soccer_extended.ipynb", "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Find and fix the agents cell
    for i, cell in enumerate(notebook['cells']):
        if cell.get('metadata', {}).get('id') == 'agents_code':
            # Get the original agents code
            original_code = cell['source']
            
            # Find the DQNAgent class and replace it
            if 'class DQNAgent(BaseAgent):' in original_code:
                # Split the code at DQNAgent class
                before_dqn = original_code.split('class DQNAgent(BaseAgent):')[0]
                
                # Find where MADDPGAgent class starts (or end of DQNAgent)
                if 'class MADDPGAgent(BaseAgent):' in original_code:
                    after_dqn_start = original_code.find('class MADDPGAgent(BaseAgent):')
                    after_dqn = original_code[after_dqn_start:]
                else:
                    # If no MADDPG, find the create_agent function
                    if 'def create_agent(' in original_code:
                        after_dqn_start = original_code.find('def create_agent(')
                        after_dqn = original_code[after_dqn_start:]
                    else:
                        after_dqn = '\n\nprint("✅ Agents implemented successfully!")'
                
                # Create the fixed DQNAgent code
                fixed_dqn = create_fixed_dqn_agent_code()
                
                # Combine the parts
                cell['source'] = before_dqn + fixed_dqn + '\n\n' + after_dqn
                print(f"Fixed agents cell at index {i}")
                break
    
    # Also add a simpler fix cell that can be run separately
    fix_cell = {
        "cell_type": "markdown",
        "metadata": {"id": "dqn_fix_header"},
        "source": ["### 🔧 DQNAgent Fix (Run if you encounter 'training' error)"]
    }
    
    fix_code_cell = {
        "cell_type": "code",
        "metadata": {"id": "dqn_fix_code"},
        "execution_count": None,
        "outputs": [],
        "source": create_fixed_dqn_agent_code()
    }
    
    # Insert the fix cells before the extended training section
    insert_index = None
    for i, cell in enumerate(notebook['cells']):
        if cell.get('metadata', {}).get('id') == 'extended_training_header':
            insert_index = i
            break
    
    if insert_index:
        notebook['cells'].insert(insert_index, fix_cell)
        notebook['cells'].insert(insert_index + 1, fix_code_cell)
    
    # Save the fixed notebook
    with open("/home/user/webapp/multiagents_soccer_extended_fixed.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Fixed notebook created: multiagents_soccer_extended_fixed.ipynb")
    
    # Also create a standalone fix cell that can be copy-pasted
    standalone_fix = '''# Quick Fix for DQNAgent 'training' error
# Run this cell if you encounter: NameError: name 'training' is not defined

# Fixed learn method for DQNAgent
def fixed_learn(self, experiences=None, training=True):
    """Learn from experiences - FIXED VERSION"""
    if len(self.replay_buffer) < self.batch_size:
        return {"loss": 0.0}
    
    batch = random.sample(self.replay_buffer, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states).to(self.device)
    actions = torch.LongTensor(actions).to(self.device)
    rewards = torch.FloatTensor(rewards).to(self.device)
    next_states = torch.FloatTensor(next_states).to(self.device)
    dones = torch.BoolTensor(dones).to(self.device)
    
    current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
    
    with torch.no_grad():
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
    
    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
    
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
    self.optimizer.step()
    
    if training:  # Fixed: now training is a parameter
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    return {"loss": loss.item(), "epsilon": self.epsilon}

# Apply the fix to DQNAgent class
DQNAgent.learn = fixed_learn

print("✅ DQNAgent.learn method has been fixed!")
print("   You can now run the DQN training without errors.")'''
    
    with open("/home/user/webapp/dqn_quick_fix.py", "w") as f:
        f.write(standalone_fix)
    
    print("\n📝 Also created: dqn_quick_fix.py (standalone fix)")
    
    return True

if __name__ == "__main__":
    create_fixed_extended_notebook()
    
    print("\n🔧 修正内容:")
    print("  1. DQNAgent.learn()メソッドに'training'パラメータを追加")
    print("  2. デフォルト値をtraining=Trueに設定")
    print("  3. epsilon更新をtrainingフラグで制御")
    print("\n📋 使用方法:")
    print("  1. multiagents_soccer_extended_fixed.ipynb を使用")
    print("  2. または現在のノートブックでdqn_quick_fix.pyの内容を実行")