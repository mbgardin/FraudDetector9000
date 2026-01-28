from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / data
    reports_dir: Path = project_root / reports

@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

@dataclass(frozen=True)
class DataConfig:
    filename: str = creditcard.csv
    label_col: str = Class
    time_col: str = Time

@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
