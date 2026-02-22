from pathlib import Path
import yaml
import pandas as pd


class Schema:
    def __init__(self, schema_path: str):
        self.schema_path = Path(schema_path).resolve()
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

        # Проверяем, что исходные столбцы реально есть во входном df
        src_cols = [
            columns_config["transaction_id"],
            columns_config["time"],
            columns_config["target"],
        ]
        missing_src = [c for c in src_cols if c not in df.columns]
        if missing_src:
            raise ValueError(f"Input dataframe is missing columns from schema mapping: {missing_src}")

        col_map = {
            columns_config["transaction_id"]: "transaction_id",
            columns_config["time"]: "time",
            columns_config["target"]: "target",
        }
        return df.rename(columns=col_map)

    def validate(self, df: pd.DataFrame):
        required = ["transaction_id", "time", "target"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after schema mapping: {missing}")

    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df["transaction_id"] = pd.to_numeric(df["transaction_id"], errors="raise").astype("int64")
        df["time"] = pd.to_numeric(df["time"], errors="raise").astype("int64")
        df["target"] = pd.to_numeric(df["target"], errors="raise").astype("int8")
        return df

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.rename_columns(df)
        self.validate(df)
        df = self.cast_types(df)
        return df