"""Make the repo-root modules importable when pytest collects this folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
