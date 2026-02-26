from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


SCORE_CANDIDATES = [
    "risk_score", "p_tabular", "score", "proba", "prob", "p",
    "y_proba", "y_pred_proba", "pred_proba", "fraud_proba", "fraud_score",
]
DECISION_CANDIDATES = ["decision", "action", "label", "pred", "prediction", "y_pred"]
ID_CANDIDATES = ["transaction_id", "TransactionID", "id", "row_id"]


def pick_col(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], kind: str) -> str:
    if preferred:
        if preferred in df.columns:
            return preferred
        raise ValueError(f"{kind}: column '{preferred}' not found. Available: {df.columns.tolist()}")
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"{kind}: no suitable column found. Tried {candidates}. Available: {df.columns.tolist()}")


def _to_float_array(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").astype(float).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", required=True, help="offline parquet")
    ap.add_argument("--api", required=True, help="api parquet")

    ap.add_argument("--id-col", default=None, help="join key (auto if omitted)")
    ap.add_argument("--score-col", default=None, help="score column (auto if omitted)")
    ap.add_argument("--decision-col", default=None, help="decision column (auto if omitted)")

    ap.add_argument("--join-mode", default="auto", choices=["auto", "id", "position"],
                    help="auto: try id-join then fallback to position; id: only by id; position: only by row order")
    ap.add_argument("--tol", type=float, default=1e-9, help="absolute tolerance for score diff")
    args = ap.parse_args()

    off_p = Path(args.offline)
    api_p = Path(args.api)
    if not off_p.exists():
        raise FileNotFoundError(off_p)
    if not api_p.exists():
        raise FileNotFoundError(api_p)

    df_off = pd.read_parquet(off_p)
    df_api = pd.read_parquet(api_p)

    # pick score/decision
    score_off = pick_col(df_off, args.score_col, SCORE_CANDIDATES, "offline score")
    score_api = pick_col(df_api, args.score_col, SCORE_CANDIDATES, "api score")
    dec_off = pick_col(df_off, args.decision_col, DECISION_CANDIDATES, "offline decision")
    dec_api = pick_col(df_api, args.decision_col, DECISION_CANDIDATES, "api decision")

    def compare_from_merged(m: pd.DataFrame, id_key: str) -> None:
        s_off = _to_float_array(m[f"{score_off}_offline"])
        s_api = _to_float_array(m[f"{score_api}_api"])
        diff = np.abs(s_off - s_api)

        dec_eq = (m[f"{dec_off}_offline"].astype(str) == m[f"{dec_api}_api"].astype(str)).to_numpy()

        print(f"Rows compared: {len(m)}")
        print(f"Using columns: id={id_key}, offline_score={score_off}, api_score={score_api}, "
              f"offline_dec={dec_off}, api_dec={dec_api}")

        print("\nScore diff:")
        print(f"  max   = {diff.max():.12g}")
        print(f"  mean  = {diff.mean():.12g}")
        print(f"  p99   = {np.quantile(diff, 0.99):.12g}")
        print(f"  > tol ({args.tol}) = {(diff > args.tol).sum()}")

        print("\nDecision match:")
        print(f"  equal = {dec_eq.sum()} / {len(dec_eq)}")
        print(f"  diff  = {(~dec_eq).sum()}")

        bad = m.loc[
            diff > args.tol,
            [id_key, f"{score_off}_offline", f"{score_api}_api", f"{dec_off}_offline", f"{dec_api}_api"],
        ].head(20)
        if not bad.empty:
            print("\nExamples where score differs > tol (first 20):")
            print(bad.to_string(index=False))

        bad_dec = m.loc[
            ~dec_eq,
            [id_key, f"{score_off}_offline", f"{score_api}_api", f"{dec_off}_offline", f"{dec_api}_api"],
        ].head(20)
        if not bad_dec.empty:
            print("\nExamples where decision differs (first 20):")
            print(bad_dec.to_string(index=False))

    def position_join() -> None:
        n = min(len(df_off), len(df_api))
        if n == 0:
            raise RuntimeError("One of the files is empty.")
        if len(df_off) != len(df_api):
            print(f"[WARN] Different lengths: offline={len(df_off)}, api={len(df_api)}. Comparing first {n} rows.")

        m = pd.DataFrame({
            "__pos__": np.arange(1, n + 1),
            f"{score_off}_offline": df_off[score_off].iloc[:n].reset_index(drop=True),
            f"{score_api}_api": df_api[score_api].iloc[:n].reset_index(drop=True),
            f"{dec_off}_offline": df_off[dec_off].iloc[:n].reset_index(drop=True),
            f"{dec_api}_api": df_api[dec_api].iloc[:n].reset_index(drop=True),
        })
        compare_from_merged(m, "__pos__")

    def id_join() -> bool:
        # pick id columns
        id_off = pick_col(df_off, args.id_col, ID_CANDIDATES, "offline id")
        id_api = pick_col(df_api, args.id_col, ID_CANDIDATES, "api id")

        off = df_off[[id_off, score_off, dec_off]].copy()
        api = df_api[[id_api, score_api, dec_api]].copy()

        # unify names
        off = off.rename(columns={id_off: "__id__", score_off: f"{score_off}_offline", dec_off: f"{dec_off}_offline"})
        api = api.rename(columns={id_api: "__id__", score_api: f"{score_api}_api", dec_api: f"{dec_api}_api"})

        # CRITICAL: cast both to string to avoid int64 vs str merge error
        off["__id__"] = off["__id__"].astype(str)
        api["__id__"] = api["__id__"].astype(str)

        m = off.merge(api, on="__id__", how="inner")
        if m.empty:
            return False

        compare_from_merged(m, "__id__")
        return True

    if args.join_mode == "position":
        position_join()
        return

    if args.join_mode == "id":
        ok = id_join()
        if not ok:
            raise RuntimeError(
                "ID-join produced 0 matched rows. Скорее всего offline id=TransactionID, а api id=row_1..row_N.\n"
                "Используй --join-mode position ИЛИ сделай так, чтобы API возвращал TransactionID (canonical endpoint)."
            )
        return

    # auto mode
    ok = id_join()
    if not ok:
        print("[WARN] ID-join produced 0 matched rows. Falling back to position-join.")
        position_join()


if __name__ == "__main__":
    main()