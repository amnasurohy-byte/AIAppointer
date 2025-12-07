"""
Script to update all analysis/verification scripts to use config.py
Run this after changing the dataset in config.py
"""
import os
import re

# Files to update
files_to_update = [
    'analyze_constraints.py',
    'analyze_ranks.py',
    'verify_billet_lookup.py',
    'verify_inference.py',
    'debug_rank.py',
    'debug_parser.py',
    'debug_features.py',
    'check_history_depth.py',
    'data_analysis.py'
]

for filename in files_to_update:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
        
        # Add import if not present
        if 'from config import DATASET_PATH' not in content:
            # Add after other imports
            lines = content.split('\n')
            import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_idx = i + 1
            lines.insert(import_idx, 'from config import DATASET_PATH')
            content = '\n'.join(lines)
        
        # Replace hardcoded CSV paths
        content = re.sub(
            r"['\"]hr_star_trek_v4c_modernized_clean.*?\.csv['\"]",
            "DATASET_PATH",
            content
        )
        
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"Updated: {filename}")

print("\nAll files updated to use config.py!")
print("To change the dataset in the future, just edit config.py")
