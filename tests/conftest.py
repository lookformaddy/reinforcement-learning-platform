"""Put src/ on sys.path so plain `pytest` (no install step) passes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
