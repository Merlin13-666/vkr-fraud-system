from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_system.inference.tabular_predict import (
    load_model,
    load_thresholds,
    load_feature_spec,
    predict_with_policy,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Batch predict with tabular model + policy")
    parser.add_argument("--input", required=True, help="Input csv or parquet file")
    parser.add_argument("--model", required=True, help="Path to model.pkl")
    parser.add_argument("--thresholds", required=True, help="Path to thresholds json")
    parser.add_argument("--feature-spec", required=True, help="Path to tabular_feature_spec.json")
    parser.add_argument("--out", required=True, help="Output parquet path")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    _ensure_dir(out_path.parent)

    print(f"[A8] Loading input: {input_path}")

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Only csv or parquet supported")

    model = load_model(Path(args.model))
    thresholds = load_thresholds(Path(args.thresholds))
    feature_spec = load_feature_spec(Path(args.feature_spec))

    preds, summary = predict_with_policy(
        df=df,
        model=model,
        thresholds=thresholds,
        feature_spec=feature_spec,
    )

    preds.to_parquet(out_path, index=False)

    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[A8] Saved predictions: {out_path}")
    print(f"[A8] Saved summary: {summary_path}")
    print("[A8] Done.")


if __name__ == "__main__":
    main()