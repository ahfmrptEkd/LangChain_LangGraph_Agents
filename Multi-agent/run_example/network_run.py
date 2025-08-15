"""Quick runner for the Network template.

Usage:
    python Multi-agent/run_example/network_run.py "Your topic"
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env")
# Allow importing modules from the templates directory despite hyphen in parent path
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
sys.path.insert(0, str(TEMPLATES_DIR))

import network  # type: ignore


def main():
    if len(sys.argv) < 2:
        print("Usage: python Multi-agent/run_example/network_run.py 'your topic'")
        sys.exit(1)
    text = sys.argv[1]
    result = network.run(text)
    print(result)


if __name__ == "__main__":
    main()


