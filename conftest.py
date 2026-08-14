import sys
from pathlib import Path

# Automatically append VYOMAAV package root to Python module path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
