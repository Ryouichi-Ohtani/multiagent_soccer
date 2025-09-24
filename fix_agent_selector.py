#!/usr/bin/env python3
"""
Fix the agent_selector import issue in the notebook
"""

import json

def fix_agent_selector_import():
    """Fix the agent_selector import and usage"""
    
    # Load the fixed notebook
    with open('/home/user/webapp/multiagents_soccer_colab_fixed.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the imports cell
    for cell in notebook['cells']:
        if cell.get('source') and isinstance(cell['source'], str):
            # Fix the import statement
            if 'from pettingzoo.utils import agent_selector, wrappers' in cell['source']:
                cell['source'] = cell['source'].replace(
                    'from pettingzoo.utils import agent_selector, wrappers',
                    'from pettingzoo.utils import agent_selector as AgentSelector, wrappers'
                )
                print("Fixed import statement")
            
            # Fix the usage of agent_selector
            if 'self._agent_selector = agent_selector(self.agents)' in cell['source']:
                cell['source'] = cell['source'].replace(
                    'self._agent_selector = agent_selector(self.agents)',
                    'self._agent_selector = AgentSelector(self.agents)'
                )
                print("Fixed agent_selector usage")
            
            if 'self._agent_selector = agent_selector(self.agents)' in cell['source']:
                cell['source'] = cell['source'].replace(
                    'self._agent_selector = agent_selector(self.agents)',
                    'self._agent_selector = AgentSelector(self.agents)'
                )
    
    # Save the fixed notebook
    with open('/home/user/webapp/multiagents_soccer_colab_fixed2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Notebook fixed and saved as multiagents_soccer_colab_fixed2.ipynb")

if __name__ == "__main__":
    fix_agent_selector_import()