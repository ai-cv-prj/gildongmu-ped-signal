"""Command-line entry point for the traffic-light inference benchmark."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parts.traffic_light.runtime.pipeline import main


if __name__ == "__main__":
    main()
