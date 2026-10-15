import sys
from pathlib import Path

# Add project root directory to sys.path so app can be imported
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from app import app
