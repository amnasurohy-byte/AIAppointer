# Global Configuration for AI Appointer Assist System

# Dataset Configuration
DATASET_PATH = 'data/hr_star_trek_v4c_modernized_clean_modified_v4.csv'

# Model Configuration
MODELS_DIR = 'models'

# Constraint Configuration
# Rank Flexibility: How many ranks up/down to allow from exact match
# 0 = Strict (exact rank only)
# 1 = Allow ±1 rank (e.g., Commander can get Lt.Cdr or Captain roles)
# 2 = Allow ±2 ranks (more creative/flexible)
DEFAULT_RANK_FLEXIBILITY = 0  # Strict by default

# UI Configuration
UI_PAGE_TITLE = "AI Appointer Assist"
UI_LAYOUT = "wide"
