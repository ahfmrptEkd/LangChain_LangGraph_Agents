"""Quick runner for the Tool-calling Supervisor template.

Usage:
    python Multi-agent/run_example/supervisor_tool_run.py "Your task"
"""

import sys
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
sys.path.insert(0, str(TEMPLATES_DIR))

import supervisor_tool  # type: ignore


def main():
    if len(sys.argv) < 2:
        print("Usage: python Multi-agent/run_example/supervisor_tool_run.py 'your task'")
        sys.exit(1)
    text = sys.argv[1]
    result = supervisor_tool.run(text)
    print(result)


if __name__ == "__main__":
    main()


