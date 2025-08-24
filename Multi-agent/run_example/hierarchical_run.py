"""Quick runner for the Hierarchical template.

Usage:
    python Multi-agent/run_example/hierarchical_run.py "Your task"
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env")
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
sys.path.insert(0, str(TEMPLATES_DIR))

import hierarchical  # type: ignore


def main():
    if len(sys.argv) < 2:
        print("Usage: python Multi-agent/run_example/hierarchical_run.py 'your task'")
        sys.exit(1)
    text = sys.argv[1]
    result = hierarchical.run(text)
    print(result)


if __name__ == "__main__":
    main()


