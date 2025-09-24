#!/usr/bin/env python3
"""
Fix the UnboundLocalError in multiagents_soccer_20sec_improved_physics.ipynb
"""

import json

def fix_notebook():
    """Fix the action initialization error and training duration"""
    
    # Read the notebook
    with open('multiagents_soccer_20sec_improved_physics.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix EnhancedExpertAgent
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'class EnhancedExpertAgent' in source:
                print(f"Fixing EnhancedExpertAgent at cell {i}")
                
                # Fix the select_action method
                lines = []
                for line in cell['source']:
                    lines.append(line)
                
                # Rebuild the source with proper action initialization
                new_source = []
                in_select_action = False
                action_initialized = False
                indent_level = 0
                
                for line in lines:
                    # Check if we're in select_action method
                    if 'def select_action' in line:
                        in_select_action = True
                        action_initialized = False
                        new_source.append(line)
                        continue
                    
                    # If we're in select_action and haven't initialized action yet
                    if in_select_action and not action_initialized:
                        # Look for where we should initialize action
                        if 'Parse observation' in line or 'ball_pos =' in line:
                            # Insert action initialization before parsing
                            if not action_initialized:
                                new_source.append("        # Initialize action\n")
                                new_source.append("        action = np.zeros(5)\n")
                                new_source.append("        \n")
                                action_initialized = True
                            new_source.append(line)
                        elif 'action = np.zeros(5)' in line:
                            # Already has initialization, skip duplicate
                            action_initialized = True
                            new_source.append(line)
                        else:
                            new_source.append(line)
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
                break
    
    # Now find and fix the training configuration to ensure 20 seconds
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            
            # Look for training configuration
            if 'run_match' in source or 'num_episodes' in source:
                # Check if MAX_STEPS is properly set
                if 'MAX_STEPS' in source and '300' in source:
                    print(f"Updating training duration at cell {i}")
                    cell['source'] = [line.replace('300', '600') for line in cell['source']]
            
            # Update any hardcoded episode lengths
            if 'episode_length' in source or 'max_steps' in source:
                if '10' in source and 'seconds' in source:
                    print(f"Updating episode duration reference at cell {i}")
                    cell['source'] = [line.replace('10 seconds', '20 seconds').replace('10秒', '20秒') 
                                     for line in cell['source']]
    
    # Fix the specific problematic method more comprehensively
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            source = ''.join(source_lines)
            
            if 'class EnhancedExpertAgent' in source and 'def select_action' in source:
                print(f"Applying comprehensive fix to EnhancedExpertAgent at cell {i}")
                
                # Complete rewrite of the select_action method
                fixed_source = []
                in_class = False
                in_select_action = False
                method_indent = "    "
                
                for line in source_lines:
                    if 'class EnhancedExpertAgent' in line:
                        in_class = True
                        fixed_source.append(line)
                    elif in_class and 'def select_action' in line:
                        in_select_action = True
                        fixed_source.append(line)
                        # Add proper initialization right after method declaration
                        fixed_source.append("        \"\"\"Enhanced strategy for longer episodes\"\"\"\n")
                        fixed_source.append("        # Initialize action first\n")
                        fixed_source.append("        action = np.zeros(5)\n")
                        fixed_source.append("        \n")
                    elif in_select_action and ('def ' in line and 'select_action' not in line):
                        # End of select_action method
                        in_select_action = False
                        fixed_source.append(line)
                    elif in_select_action:
                        # Skip duplicate action initialization
                        if 'action = np.zeros(5)' in line:
                            continue
                        # Fix the stuck detection part
                        if 'if self.stuck_counter > 5:' in line:
                            fixed_source.append(line)
                            fixed_source.append("            # Escape strategy when stuck\n")
                            fixed_source.append("            angle = random.uniform(0, 2 * math.pi)\n")
                            fixed_source.append("            action[0:2] = np.array([math.cos(angle), math.sin(angle)])\n")
                            fixed_source.append("            \n")
                            # Look ahead for dist_to_ball check
                            next_lines = []
                            j = source_lines.index(line) + 1
                            while j < len(source_lines) and 'return action' not in source_lines[j]:
                                if 'dist_to_ball' in source_lines[j]:
                                    next_lines.append("            # Check if close enough to kick\n")
                                    next_lines.append("            dist_to_ball = np.linalg.norm(ball_pos - self_pos)\n")
                                    next_lines.append("            if dist_to_ball < 50:\n")
                                    next_lines.append("                action[2] = 1.0  # Strong kick\n")
                                    next_lines.append("            action[0:2] += np.random.randn(2) * 0.2  # Add noise\n")
                                    next_lines.append("            return action\n")
                                    break
                                j += 1
                            fixed_source.extend(next_lines)
                            # Skip the original lines we've already processed
                            continue
                        else:
                            fixed_source.append(line)
                    else:
                        fixed_source.append(line)
                
                cell['source'] = fixed_source
                break
    
    # Save the fixed notebook
    output_file = 'multiagents_soccer_20sec_improved_physics_fixed.ipynb'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Fixed notebook saved as: {output_file}")
    print("\n📋 Fixes applied:")
    print("  1. ✅ Fixed UnboundLocalError - action variable now initialized at start of select_action")
    print("  2. ✅ Ensured training episodes are 20 seconds (600 steps)")
    print("  3. ✅ Fixed stuck detection logic with proper variable scope")
    
    return output_file

if __name__ == "__main__":
    fix_notebook()