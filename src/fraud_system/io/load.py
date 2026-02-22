from pathlib import Path
import pandas as pd


def load_file(path: str) -> pd.DataFrame:
    path = Path(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    elif path.suffix == ".parquet":
        return pd.read_parquet(path)

    elif path.suffix in [".json", ".jsonl"]:
        return pd.read_json(path, lines=True)

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")