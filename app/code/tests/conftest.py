"""Make the app modules importable when pytest collects this folder."""
import sys
from pathlib import Path

# app/code is the parent of this tests/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
