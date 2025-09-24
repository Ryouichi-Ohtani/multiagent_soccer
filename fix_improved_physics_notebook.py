#!/usr/bin/env python3
"""
Fix the AgentSelector import error in the improved physics notebook
"""

import json

def fix_notebook():
    """Fix the AgentSelector import issue"""
    
    # Read the existing notebook
    with open('multiagents_soccer_improved_physics.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the environment cell with the import issue
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and any('from pettingzoo.utils import agent_selector' in line for line in cell['source']):
            # Replace the problematic import
            new_source = []
            for line in cell['source']:
                if 'from pettingzoo.utils import agent_selector as AgentSelector' in line:
                    # Replace with correct import
                    new_source.append("from pettingzoo.utils.agent_selector import AgentSelector\n")
                elif line.strip().startswith('from pettingzoo import'):
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("✅ Fixed AgentSelector import")
            break
    
    # Also check if we need to add the AgentSelector class definition as fallback
    # Find the environment implementation cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'class ImprovedSoccerEnvironment(AECEnv):' in ''.join(cell['source']):
            # Add AgentSelector implementation before the environment class
            agent_selector_code = """# AgentSelector implementation (fallback if import fails)
try:
    from pettingzoo.utils.agent_selector import AgentSelector
except ImportError:
    # Fallback implementation
    class AgentSelector:
        def __init__(self, agents):
            self.agents = agents
            self._current_agent_idx = 0
            self.selected_agent = self.agents[0] if agents else None
        
        def next(self):
            if not self.agents:
                return None
            self.selected_agent = self.agents[self._current_agent_idx]
            self._current_agent_idx = (self._current_agent_idx + 1) % len(self.agents)
            return self.selected_agent
        
        def is_last(self):
            return self._current_agent_idx == 0
        
        def reset(self):
            self._current_agent_idx = 0
            self.selected_agent = self.agents[0] if self.agents else None

"""
            # Prepend the AgentSelector code
            cell['source'] = agent_selector_code + ''.join(cell['source'])
            cell['source'] = cell['source'].split('\n')
            print("✅ Added AgentSelector fallback implementation")
            break
    
    # Save the fixed notebook
    with open('multiagents_soccer_improved_physics_fixed.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Fixed notebook saved as: multiagents_soccer_improved_physics_fixed.ipynb")
    return True

if __name__ == "__main__":
    fix_notebook()