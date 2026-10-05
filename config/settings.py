"""
Central configuration module for project.

This module stores all constants shared across the project.

No logic is executed here — this module only defines static values that
other modules (scripts, src/, notebooks, app/) import from.
"""

# import pandas as pd
from pathlib import Path

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PROJECT-RELATIVE DIRECTORIES
EXPORT_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_FIG = PROJECT_ROOT / 'results' / 'figures'
RESULTS_TAB = PROJECT_ROOT / 'results' / 'tables'