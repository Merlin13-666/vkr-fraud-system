from pathlib import Path
import yaml
import pandas as pd


class Schema:
    def __init__(self, schema_path: str):
        self.schema_path = Path(schema_path)
        self.config = self._load_schema()

    def _load_schema(self):
        with open(self.schema_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_config = self.config.get("columns", {})

        required_keys = ["transaction_id", "time", "target"]
        for key in required_keys:
            if key not in columns_config:
                raise ValueError(f"Schema is missing required mapping: {key}")

        col_map = {
            columns_config["transaction_id"]: "transaction_id",
            columns_config["time"]: "time",
            columns_config["target"]: "target",
        }

        df = df.rename(columns=col_map)
        return df

    def validate(self, df: pd.DataFrame):
        required = ["transaction_id", "time", "target"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after schema mapping: {missing}")