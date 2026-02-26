from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def wait_api(base_url: str, timeout_s: int = 10) -> None:
    url = base_url.rstrip("/") + "/health"
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"/health returned {r.status_code}: {r.text}")
    except Exception as e:
        raise RuntimeError(
            f"API недоступно по {base_url}. "
            f"Сначала запусти: python -m scripts.13_serve_api. "
            f"Детали: {type(e).__name__}: {e}"
        )


def post_predict(
    base_url: str,
    api_key: Optional[str],
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int = 60,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + endpoint
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Не удалось подключиться к API: {url}. "
            f"Проверь что сервер запущен и порт верный. "
            f"Ошибка: {e}"
        )

    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def read_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)
    if suf == ".json":
        # ожидаем либо list[dict], либо {"rows":[...]}
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "rows" in obj:
            return pd.DataFrame(obj["rows"])
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        raise ValueError("Unsupported json format: expected list[dict] or {'rows':[...]}")

    raise ValueError(f"Unsupported file type: {p.suffix}")


def to_payload_rows(df: pd.DataFrame, include_gnn: bool) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        d = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
        if not include_gnn:
            d.pop("gnn_score", None)
        rows.append(d)
    return {"rows": rows}


def save_output(items: List[Dict[str, Any]], out_path: str) -> None:
    df = pd.json_normalize(items)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    suf = p.suffix.lower()
    if suf == ".parquet":
        df.to_parquet(p, index=False)
    elif suf == ".csv":
        df.to_csv(p, index=False)
    else:
        p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(items: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(items)
    if df.empty:
        print("No items")
        return
    print("Counts by decision:")
    print(df["decision"].value_counts(dropna=False))
    print("\nRisk score summary:")
    print(df["risk_score"].describe())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input file: .parquet/.csv/.json")
    ap.add_argument("--out", required=True, help="Output file: .parquet/.csv/.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base url")
    ap.add_argument("--endpoint", default="/predict/rows", help="API endpoint (/predict/rows or /predict/canonical)")
    ap.add_argument(
        "--api-key",
        default=os.getenv("FRAUD_API_API_KEY", ""),
        help="API key (or env FRAUD_API_API_KEY). ВАЖНО: env должен быть в этом терминале.",
    )
    ap.add_argument("--batch-size", type=int, default=500, help="Batch size")
    ap.add_argument("--model", choices=["tabular", "fusion_external"], default="tabular", help="Model for rows endpoint")
    args = ap.parse_args()

    # Ранний фейл с понятной подсказкой (чтобы не ловить 401 внутри цикла)
    if not args.api_key:
        raise RuntimeError(
            "Missing API key. Передай --api-key <key> или установи env FRAUD_API_API_KEY "
            "в этом терминале (например: $env:FRAUD_API_API_KEY='super-secret')."
        )

    wait_api(args.base_url)

    df = read_input(args.input)

    include_gnn = (args.model == "fusion_external")
    if include_gnn and "gnn_score" not in df.columns:
        raise ValueError("model=fusion_external требует колонку gnn_score в input файле")

    all_items: List[Dict[str, Any]] = []
    n = len(df)

    for start in range(0, n, args.batch_size):
        batch = df.iloc[start : start + args.batch_size].copy()

        payload = to_payload_rows(batch, include_gnn=include_gnn)
        payload["options"] = {"model": args.model, "with_reasons": False, "reasons_topk": 5}

        res = post_predict(
            base_url=args.base_url,
            api_key=args.api_key,
            endpoint=args.endpoint,
            payload=payload,
        )
        items = res.get("items", [])
        all_items.extend(items)
        print(f"batch {start}:{start+len(batch)} OK -> {len(items)} items")

    save_output(all_items, args.out)
    print_summary(all_items)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()