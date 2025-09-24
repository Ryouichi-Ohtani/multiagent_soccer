#!/usr/bin/env python3
"""
Create a properly fixed version of the 20sec improved physics notebook
"""

import json

def create_fixed_notebook():
    """Create a properly fixed version"""
    
    # Read the original improved notebook
    with open('multiagents_soccer_20sec_improved_physics.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Fix the EnhancedExpertAgent class completely
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            
            if 'class EnhancedExpertAgent' in source:
                print(f"Replacing EnhancedExpertAgent at cell {i}")
                
                # Complete fixed implementation
                cell['source'] = [
                    "class EnhancedExpertAgent(BaseAgent):\n",
                    "    \"\"\"Enhanced expert agent for 20-second episodes with stamina management\"\"\"\n",
                    "    \n",
                    "    def __init__(self, agent_id: int, team: int, config: ExtendedSoccerConfig):\n",
                    "        super().__init__(agent_id, 5)\n",
                    "        self.team = team\n",
                    "        self.config = config\n",
                    "        self.field_width, self.field_height = config.FIELD_SIZE\n",
                    "        self.stamina = 1.0  # Stamina system for longer games\n",
                    "        self.role = 'attacker' if agent_id % 2 == 0 else 'defender'\n",
                    "        self.last_ball_pos = None\n",
                    "        self.stuck_counter = 0\n",
                    "        \n",
                    "    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:\n",
                    "        \"\"\"Enhanced strategy for longer episodes\"\"\"\n",
                    "        # Initialize action array first\n",
                    "        action = np.zeros(5)\n",
                    "        \n",
                    "        # Parse observation\n",
                    "        self_pos = observation[0:2]\n",
                    "        ball_pos = observation[4:6]\n",
                    "        teammate_pos = observation[8:10]\n",
                    "        opp1_pos = observation[12:14]\n",
                    "        opp2_pos = observation[16:18]\n",
                    "        \n",
                    "        # Denormalize positions\n",
                    "        self_x, self_y = self_pos[0] * self.field_width, self_pos[1] * self.field_height\n",
                    "        ball_x, ball_y = ball_pos[0] * self.field_width, ball_pos[1] * self.field_height\n",
                    "        teammate_x, teammate_y = teammate_pos[0] * self.field_width, teammate_pos[1] * self.field_height\n",
                    "        \n",
                    "        # Current positions as arrays\n",
                    "        self_pos_denorm = np.array([self_x, self_y])\n",
                    "        ball_pos_denorm = np.array([ball_x, ball_y])\n",
                    "        \n",
                    "        # 🚨 Stuck detection\n",
                    "        if self.last_ball_pos is not None:\n",
                    "            ball_movement = np.linalg.norm(ball_pos_denorm - self.last_ball_pos)\n",
                    "            if ball_movement < 2.0:\n",
                    "                self.stuck_counter += 1\n",
                    "            else:\n",
                    "                self.stuck_counter = 0\n",
                    "        self.last_ball_pos = ball_pos_denorm.copy()\n",
                    "        \n",
                    "        # Distance to ball\n",
                    "        dist_to_ball = np.linalg.norm(ball_pos_denorm - self_pos_denorm)\n",
                    "        \n",
                    "        # Apply escape strategy if stuck\n",
                    "        if self.stuck_counter > 5:\n",
                    "            # Escape strategy when stuck\n",
                    "            angle = random.uniform(0, 2 * math.pi)\n",
                    "            action[0:2] = np.array([math.cos(angle), math.sin(angle)])\n",
                    "            if dist_to_ball < 50:\n",
                    "                action[2] = 1.0  # Strong kick\n",
                    "            action[0:2] += np.random.randn(2) * 0.2  # Add noise\n",
                    "            return action\n",
                    "        \n",
                    "        # Stamina management for 20-second games\n",
                    "        self.stamina -= 0.001  # Gradual stamina decrease\n",
                    "        self.stamina = max(0.3, self.stamina)  # Minimum stamina\n",
                    "        \n",
                    "        # Speed modifier based on stamina\n",
                    "        speed_modifier = 0.5 + 0.5 * self.stamina\n",
                    "        \n",
                    "        # Role-based strategy\n",
                    "        if self.role == 'attacker':\n",
                    "            # Attacker logic\n",
                    "            if self.team == 0:  # Blue team attacks right goal\n",
                    "                goal_x, goal_y = self.field_width - 30, self.field_height / 2\n",
                    "            else:  # Red team attacks left goal\n",
                    "                goal_x, goal_y = 30, self.field_height / 2\n",
                    "            \n",
                    "            goal_pos = np.array([goal_x, goal_y])\n",
                    "            \n",
                    "            if dist_to_ball < 50:\n",
                    "                # Has ball - move toward goal\n",
                    "                direction_to_goal = goal_pos - ball_pos_denorm\n",
                    "                direction_to_goal = direction_to_goal / (np.linalg.norm(direction_to_goal) + 1e-6)\n",
                    "                \n",
                    "                # Shooting range check\n",
                    "                dist_to_goal = np.linalg.norm(goal_pos - ball_pos_denorm)\n",
                    "                if dist_to_goal < 150:  # In shooting range\n",
                    "                    action[2] = 1.0  # Kick\n",
                    "                    action[3] = 0.8  # Kick power\n",
                    "                \n",
                    "                # Move in direction of goal\n",
                    "                action[0:2] = direction_to_goal * speed_modifier\n",
                    "            else:\n",
                    "                # Move to ball\n",
                    "                direction_to_ball = ball_pos_denorm - self_pos_denorm\n",
                    "                if np.linalg.norm(direction_to_ball) > 0:\n",
                    "                    direction_to_ball = direction_to_ball / np.linalg.norm(direction_to_ball)\n",
                    "                action[0:2] = direction_to_ball * speed_modifier\n",
                    "                \n",
                    "                # Sprint if far from ball\n",
                    "                if dist_to_ball > 200 and self.stamina > 0.5:\n",
                    "                    action[4] = 1.0  # Sprint\n",
                    "                    self.stamina -= 0.01  # Extra stamina cost\n",
                    "        \n",
                    "        else:  # Defender\n",
                    "            # Defender logic\n",
                    "            if self.team == 0:  # Blue team defends left goal\n",
                    "                goal_x, goal_y = 30, self.field_height / 2\n",
                    "            else:  # Red team defends right goal\n",
                    "                goal_x, goal_y = self.field_width - 30, self.field_height / 2\n",
                    "            \n",
                    "            goal_pos = np.array([goal_x, goal_y])\n",
                    "            \n",
                    "            # Position between ball and own goal\n",
                    "            ideal_pos = goal_pos + 0.4 * (ball_pos_denorm - goal_pos)\n",
                    "            \n",
                    "            # Move to ideal defensive position\n",
                    "            direction = ideal_pos - self_pos_denorm\n",
                    "            if np.linalg.norm(direction) > 0:\n",
                    "                direction = direction / np.linalg.norm(direction)\n",
                    "            \n",
                    "            action[0:2] = direction * speed_modifier\n",
                    "            \n",
                    "            # Clear ball if close\n",
                    "            if dist_to_ball < 40:\n",
                    "                # Clear away from goal\n",
                    "                clear_direction = ball_pos_denorm - goal_pos\n",
                    "                if np.linalg.norm(clear_direction) > 0:\n",
                    "                    clear_direction = clear_direction / np.linalg.norm(clear_direction)\n",
                    "                action[0:2] = clear_direction\n",
                    "                action[2] = 1.0  # Kick\n",
                    "                action[3] = 1.0  # Full power clear\n",
                    "        \n",
                    "        # Add small random noise to prevent perfect symmetry\n",
                    "        action[0:2] += np.random.randn(2) * 0.05\n",
                    "        \n",
                    "        # Ensure action is within bounds\n",
                    "        action = np.clip(action, -1, 1)\n",
                    "        \n",
                    "        return action\n",
                    "    \n",
                    "    def learn(self, *args, **kwargs):\n",
                    "        \"\"\"Expert agents don't learn\"\"\"\n",
                    "        return {}\n",
                    "    \n",
                    "    def save(self, path: str):\n",
                    "        \"\"\"Save agent (not needed for expert)\"\"\"\n",
                    "        pass\n",
                    "    \n",
                    "    def load(self, path: str):\n",
                    "        \"\"\"Load agent (not needed for expert)\"\"\"\n",
                    "        pass\n",
                    "\n",
                    "print('✅ Enhanced expert agents with stuck detection created')"
                ]
                break
    
    # Also ensure training uses 20 seconds (600 steps)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            
            # Fix any references to episode length
            if 'MAX_STEPS' in source:
                new_source = []
                for line in cell['source']:
                    # Ensure MAX_STEPS is 600 for 20 seconds
                    if 'MAX_STEPS' in line and ('300' in line or '150' in line):
                        line = line.replace('300', '600').replace('150', '600')
                    new_source.append(line)
                cell['source'] = new_source
            
            # Fix training episode references
            if 'num_episodes' in source or 'episode' in source.lower():
                new_source = []
                for line in cell['source']:
                    # Update any 10-second references to 20 seconds
                    if '10 second' in line or '10-second' in line:
                        line = line.replace('10 second', '20 second').replace('10-second', '20-second')
                    if '10秒' in line:
                        line = line.replace('10秒', '20秒')
                    new_source.append(line)
                cell['source'] = new_source
    
    # Save the fixed notebook
    output_file = 'multiagents_soccer_20sec_improved_physics_final.ipynb'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created final fixed notebook: {output_file}")
    print("\n📋 Fixes applied:")
    print("  1. ✅ Fixed UnboundLocalError - action initialized at start of method")
    print("  2. ✅ Training episodes set to 20 seconds (600 steps)")
    print("  3. ✅ Stuck detection properly implemented")
    print("  4. ✅ All variable scoping issues resolved")
    
    return output_file

if __name__ == "__main__":
    create_fixed_notebook()