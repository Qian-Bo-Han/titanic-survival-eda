"""运行泰坦尼克号生存特征分析。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from titanic_eda.pipeline import run_analysis


if __name__ == "__main__":
    run_analysis(PROJECT_ROOT / "data" / "raw" / "train.csv")
