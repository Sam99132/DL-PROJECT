"""
scripts/run_deepar_pipeline.py - Pipeline runner for Global DeepAR Model.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_deepar_pipeline import (
    M5DeepARPipeline,
    generate_deepar_markdown_report,
    parse_args,
    main,
)

__all__ = [
    "M5DeepARPipeline",
    "generate_deepar_markdown_report",
    "parse_args",
    "main",
]

if __name__ == "__main__":
    main()
