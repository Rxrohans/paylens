import subprocess
import sys
from pathlib import Path

# Add src to path and run the real app
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app import *