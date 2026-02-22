from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from fraud_system.io.load import load_file
from fraud_system.io.schema import Schema
from fraud_system.data.split import time_based_split


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # --- Config ---
    base_cfg = _read_yaml("configs/base.yaml")
    schema_path = base_cfg.get("schema_path", "configs/schema_ieee_cis.yaml")

    raw_path = Path(base_cfg["data"]["raw_path"])
    processed_path = Path(base_cfg["data"]["processed_path"])
    splits_path = Path("data/splits")

    _ensure_dir(processed_path)
    _ensure_dir(splits_path)

    # --- Load raw IEEE-CIS (train only for A2) ---
    train_tr_path = raw_path / "train_transaction.csv"
    train_id_path = raw_path / "train_identity.csv"

    if not train_tr_path.exists():
        raise FileNotFoundError(f"Missing file: {train_tr_path}")
    if not train_id_path.exists():
        raise FileNotFoundError(f"Missing file: {train_id_path}")

    print(f"[A2] Loading: {train_tr_path}")
    df_tr = load_file(str(train_tr_path))

    print(f"[A2] Loading: {train_id_path}")
    df_id = load_file(str(train_id_path))

    # --- Merge identity into transaction ---
    # IEEE-CIS: join by TransactionID
    if "TransactionID" not in df_tr.columns or "TransactionID" not in df_id.columns:
        raise ValueError("Expected 'TransactionID' column in both train_transaction and train_identity")

    n_tr_before = len(df_tr)
    df = df_tr.merge(df_id, on="TransactionID", how="left")
    print(f"[A2] Merge train_transaction + train_identity: {n_tr_before} -> {len(df)} rows (should be same)")

    # --- Apply schema mapping (canonical columns + validation + type casting) ---
    schema = Schema(schema_path)
    df = schema.apply(df)

    # --- Basic sanity checks ---
    if df["transaction_id"].isna().any():
        raise ValueError("Found NaN in transaction_id after schema.apply()")

    n_dups = int(df["transaction_id"].duplicated().sum())
    if n_dups > 0:
        raise ValueError(f"transaction_id is not unique: duplicated={n_dups}. Check merge logic.")

    print(f"[A2] Data after schema: rows={len(df)}, cols={df.shape[1]}")
    print(f"[A2] time range: {df['time'].min()} .. {df['time'].max()}")
    print(f"[A2] fraud rate total: {df['target'].mean():.6f}")

    # --- Time-based split ---
    split_cfg = base_cfg.get("split", {})
    ratios = (
        float(split_cfg.get("train_ratio", 0.6)),
        float(split_cfg.get("val_ratio", 0.2)),
        float(split_cfg.get("test_ratio", 0.2)),
    )

    train_df, val_df, test_df, split_info = time_based_split(
        df=df,
        time_col="time",
        ratios=ratios,
        target_col="target",
    )

    print(f"[A2] Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"[A2] Fraud rate: train={train_df['target'].mean():.6f}, "
          f"val={val_df['target'].mean():.6f}, test={test_df['target'].mean():.6f}")

    # --- Save processed parquet ---
    train_out = processed_path / "train.parquet"
    val_out = processed_path / "val.parquet"
    test_out = processed_path / "test.parquet"

    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)
    test_df.to_parquet(test_out, index=False)

    print(f"[A2] Saved: {train_out}")
    print(f"[A2] Saved: {val_out}")
    print(f"[A2] Saved: {test_out}")

    # --- Save split info + columns snapshot ---
    split_info_path = splits_path / "split_info.json"
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    columns_snapshot_path = Path("artifacts/evaluation")
    _ensure_dir(columns_snapshot_path)
    with open(columns_snapshot_path / "columns.json", "w", encoding="utf-8") as f:
        json.dump({"columns": list(df.columns)}, f, ensure_ascii=False, indent=2)

    print(f"[A2] Saved: {split_info_path}")
    print(f"[A2] Saved: {columns_snapshot_path / 'columns.json'}")

    print("[A2] Done.")


if __name__ == "__main__":
    main()