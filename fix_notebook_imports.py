#!/usr/bin/env python3
"""
Fix import statements in the integrated notebook
Remove module imports since all code is in the same notebook
"""

import re

def remove_internal_imports(code):
    """Remove imports from internal modules"""
    # List of internal module imports to remove
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
            if f'from {module} import' in line or f'import {module}' in line:
                is_internal_import = True
                break
        
        # Keep the line if it's not an internal import
        if not is_internal_import:
            filtered_lines.append(line)
        else:
            # Add a comment showing what was removed
            filtered_lines.append(f"# {line}  # Removed: internal import")
    
    return '\n'.join(filtered_lines)

# Test with physics.py content
physics_code = open('/home/user/webapp/physics.py', 'r').read()
cleaned_physics = remove_internal_imports(physics_code)

print("Sample cleaned code (first 500 chars):")
print(cleaned_physics[:500])
print("\n✅ Import cleaning function created!")