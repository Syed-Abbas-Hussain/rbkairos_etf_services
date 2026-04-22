"""Configure pytest so that the simulation/ directory is on sys.path.

This allows test files to import fixtures directly via `from fixtures import ...`.
"""

import sys
import os

# Add this directory to sys.path so 'fixtures' is importable
sys.path.insert(0, os.path.dirname(__file__))
