# VKR Fraud System

**Тема ВКР:** *Система обнаружения мошеннических банковских транзакций*  

Проект — воспроизводимая «система» (pipeline + артефакты + отчёт + политика решений), реализующая **гибридный таблично-графовый подход**:

- **Tabular scorer:** LightGBM (табличная модель по engineered-признакам)
- **Graph scorer:** GNN на гетерографе (Hetero GraphSAGE, PyTorch Geometric)
- **Fusion:** логистическая регрессия поверх скореров  
  **честный режим:** обучение мета-модели на `external VAL`, тест на `external TEST`
- **Policy:** пороги `ALLOW / REVIEW / DENY` + таблицы зон + доли зон + простая экономика (cost)
- **Graph Viz:** интерактивная визуализация ego-графа (PyVis HTML) для расследования кейсов
- **Auto-report:** HTML-отчёт (таблицы + изображения) пригодный для РПЗ/презентации
- **One-click:** полный запуск одной командой (`scripts.run_all`, A12)

---

## Contents

- [0) Requirements](#0-requirements)
- [1) Installation](#1-installation)
- [2) Dataset](#2-dataset)
- [3) Quick start (A12)](#3-quick-start-a12)
- [4) Project structure](#4-project-structure)
- [5) Pipeline overview](#5-pipeline-overview)
- [6) Step-by-step run (A2…A11)](#6-step-by-step-run-a2a11)
- [7) Outputs and artifacts](#7-outputs-and-artifacts)
- [8) Batch inference (predict)](#8-batch-inference-predict)
- [9) Reproducibility / Git](#9-reproducibility--git)
- [10) Troubleshooting](#10-troubleshooting)
- [11) What to show for VKR defense](#11-what-to-show-for-vkr-defense)
- [12) (Optional) Model self-diagnostics — idea](#12-optional-model-self-diagnostics--idea)

---

## 0) Requirements

### 0.1 Software
- **OS:** Windows 10/11 (PowerShell examples below)
- **Python:** **3.10+**
- **pip:** актуальный (желательно `pip>=23`)
- (Опционально) **Git**, если фиксируешь версии кода

### 0.2 Hardware (ориентиры)
- CPU-режим возможен, но GNN может считаться заметно дольше.
- Для комфортной работы: **RAM 16+ GB** (лучше 32 GB).

---

## 1) Installation

### 1.1 Create venv (Windows PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
````

### 1.2 Install dependencies

```bash
pip install -r requirements_torch_cpu.txt
pip install -r requirements_pyg_cpu.txt
pip install -r requirements.txt
```

> Примечание: конкретные колёса torch/pyg зависят от версии Python/OS.
> Если установка PyG падает — смотри раздел [Troubleshooting](#10-troubleshooting).

---

## 2) Dataset

Используется **IEEE-CIS Fraud Detection** (Kaggle). Минимально нужны:

* `train_transaction.csv`
* `train_identity.csv`

### 2.1 Expected location

Положи файлы сюда:

```text
data/raw/ieee-cis/
  train_transaction.csv
  train_identity.csv
```

> Если у тебя есть `test_transaction.csv` / `test_identity.csv` и проект поддерживает их — можно хранить там же.
> Текущий пайплайн ориентирован на time-split внутри train.

---

## 3) Quick start (A12)

Полный прогон пайплайна (шаги пропускаются, если артефакты уже существуют):

```bash
python -m scripts.run_all
```

Полезные режимы:

```bash
# пересчитать всё (заново прогнать шаги, даже если артефакты уже есть)
python -m scripts.run_all --force

# пропустить внешнюю честную часть (A9/A10)
python -m scripts.run_all --skip-external

# прогнать только оценку + отчёт (если модели уже готовы)
python -m scripts.run_all --from-step A7_evaluate --to-step A11_auto_report
```

**Результат:** итоговый отчёт: `reports/report.html`

---

## 4) Project structure

```text
vkr_fraud_system/
  configs/
    base.yaml
    schema_ieee_cis.yaml

  data/
    raw/                # сырые csv/parquet (обычно не коммитятся)
      ieee-cis/
    processed/          # подготовленные parquet (обычно не коммитятся)
    splits/             # split_info.json

  artifacts/
    tabular/            # model.pkl
    graph/              # node_map, edges, graph_data.pt, gnn_model.pt, tx_index.parquet
    fusion/             # fusion.pkl, fusion_external.pkl
    thresholds/         # thresholds_*.json
    evaluation/         # метрики, кривые, предикты, таблицы зон

  reports/
    report.html
    assets/             # png + graph.html
    tables/             # csv/json таблицы

  src/
    fraud_system/
      io/
      data/
      features/
      models/
      graph/
      evaluation/
      api/

  scripts/
    00_prepare_data.py
    01_train_tabular.py
    02_build_graph.py
    03_make_graph_data.py
    04_train_gnn.py
    05_train_fusion.py
    06_evaluate.py
    07_predict_tabular.py
    08_predict_gnn_external.py
    09_calibrate_gnn.py
    10_train_fusion_external.py
    11_auto_report.py
    12_shap_tabular.py
    13_serve_api.py
    16_build_graph_viz.py
    run_all.py
```

---

## 5) Pipeline overview

### 5.1 Train / experiment path (offline)

```text
raw (CSV)
  → schema + merge
  → time split (train/val/test)
  → processed (parquet)
  → tabular train + preds
  → graph build (node_map + edges + tx_index)
  → PyG data (graph_data.pt)
  → GNN train (internal ablation)
  → external inductive preds + calibration
  → fusion_external (honest)
  → evaluate (thresholds + zones + cost)
  → auto_report (HTML)
  → graph_viz (PyVis HTML)
```

### 5.2 Inference path (system logic)

```text
raw
  → schema/features
  → tabular score + (gnn inductive score)
  → fusion_external score
  → thresholds/policy → decision: ALLOW / REVIEW / DENY
  → (optional) reasons/explainability + graph investigation
```

---

## 6) Step-by-step run (A2…A11)

Ниже — шаги пайплайна и их назначение. Обычно запускать вручную не нужно, если используешь A12 (`scripts.run_all`).

---

### A2) Prepare data (merge + schema + time split)

```bash
python -m scripts.00_prepare_data
```

Что делает:

1. Загружает IEEE-CIS из `data/raw/ieee-cis/`
2. Объединяет transaction/identity по `TransactionID`
3. Применяет schema mapping (`configs/schema_ieee_cis.yaml`):

   * канонические поля: `transaction_id`, `time`, `target`
4. Делает **time-based split** (например, 60/20/20) без утечки по времени
5. Сохраняет:

   * `data/processed/train.parquet`
   * `data/processed/val.parquet`
   * `data/processed/test.parquet`
   * `data/splits/split_info.json`
   * `artifacts/evaluation/columns.json`

---

### A3) Tabular baseline (LightGBM)

```bash
python -m scripts.01_train_tabular
```

Артефакты:

* `artifacts/tabular/model.pkl`
* `artifacts/evaluation/tabular_metrics.json`
* `artifacts/evaluation/pr_curve_tabular.png`
* `artifacts/evaluation/val_pred_tabular.parquet`
* `artifacts/evaluation/test_pred_tabular.parquet`
* `artifacts/evaluation/tabular_feature_spec.json`

---

### A4) Build heterogeneous graph

#### A4.1) Graph artifacts (node_map + edges)

```bash
python -m scripts.02_build_graph
```

Артефакты:

* `artifacts/graph/node_map.parquet`
* `artifacts/graph/edges.parquet`
* `artifacts/graph/graph_info.json`

#### A4.2) Convert to PyG HeteroData

```bash
python -m scripts.03_make_graph_data
```

Артефакты:

* `artifacts/graph/graph_data.pt`
* `artifacts/graph/tx_index.parquet` (mapping `transaction_id → tx_node_id`)

---

### A5) GNN scorer (internal split, ablation/debug)

```bash
python -m scripts.04_train_gnn
```

Артефакты:

* `artifacts/graph/gnn_model.pt`
* `artifacts/graph/tx_scaler.json`
* `artifacts/evaluation/gnn_metrics.json`
* `artifacts/evaluation/pr_curve_gnn.png`
* `artifacts/evaluation/val_pred_gnn.parquet`
* `artifacts/evaluation/test_pred_gnn.parquet`

> Internal режим нужен для отладки/абляции. Главный режим ВКР — external честный.

---

### A6) Fusion internal (ablation)

```bash
python -m scripts.05_train_fusion
```

Артефакты:

* `artifacts/fusion/fusion.pkl`
* `artifacts/evaluation/fusion_metrics_internal.json`
* `artifacts/evaluation/pr_curve_fusion_internal.png`
* `artifacts/evaluation/val_pred_fusion_internal.parquet`
* `artifacts/evaluation/test_pred_fusion_internal.parquet`

---

### A9) Inductive GNN predict (external honest for graph)

```bash
python -m scripts.08_predict_gnn_external --split val
python -m scripts.08_predict_gnn_external --split test
```

Артефакты:

* `artifacts/evaluation/val_pred_gnn_external.parquet`
* `artifacts/evaluation/test_pred_gnn_external.parquet`
* `artifacts/evaluation/gnn_external_metrics_val.json`
* `artifacts/evaluation/gnn_external_metrics_test.json`

---

### A9.3) Calibration (temperature scaling)

```bash
python -m scripts.09_calibrate_gnn
```

Артефакты:

* `artifacts/graph/gnn_temperature.json`
* `artifacts/evaluation/val_pred_gnn_external_calibrated.parquet`
* `artifacts/evaluation/test_pred_gnn_external_calibrated.parquet`

---

### A10) Fusion external (MAIN / honest mode)

**Основной режим ВКР:**

* обучение мета-модели на **external VAL**
* тест на **external TEST**

```bash
python -m scripts.10_train_fusion_external
```

Артефакты:

* `artifacts/fusion/fusion_external.pkl`
* `artifacts/evaluation/fusion_metrics_external.json`
* `artifacts/evaluation/pr_curve_fusion_external.png`
* `artifacts/evaluation/val_pred_fusion_external.parquet`
* `artifacts/evaluation/test_pred_fusion_external.parquet`

---

### A7) Evaluate + thresholds + decisions + cost

```bash
python -m scripts.06_evaluate
```

Что делает:

* считает метрики (logloss / PR-AUC / ROC-AUC)
* подбирает пороги `T_review`, `T_deny` по VAL
* строит таблицы зон и доли зон на VAL/TEST
* считает простую модель экономических потерь (cost)

Выходные файлы (tabular и fusion_external):

* `artifacts/thresholds/thresholds_tabular.json`
* `artifacts/thresholds/thresholds_fusion_external.json`
* `artifacts/evaluation/decision_zones_*.csv`
* `artifacts/evaluation/zone_share_*.png`
* `artifacts/evaluation/cost_*_test.json`

---

### A11) Graph visualization (PyVis HTML)

Интерактивная визуализация ego-графа (узлы/связи вокруг транзакции).

```bash
python -m scripts.16_build_graph_viz --auto-pick --pred-path artifacts/evaluation/val_pred_fusion_external.parquet
```

Выход:

* `reports/assets/graph.html`

Открыть в браузере:

```text
reports/assets/graph.html
```

---

### A11) Auto-report (HTML)

```bash
python -m scripts.11_auto_report
```

Выход:

* `reports/report.html`
* `reports/tables/model_comparison.csv`
* `reports/assets/*.png` (кривые, доли зон, SHAP и т.д.)

---

## 7) Outputs and artifacts

### 7.1 Main result (for VKR)

Главный результат системы — **fusion_external**:

* честная оценка на future split
* пороговая политика `ALLOW/REVIEW/DENY`
* отчёт `reports/report.html`

### 7.2 Where to look

* **Метрики:** `artifacts/evaluation/*metrics*.json`
* **Предикты:** `artifacts/evaluation/*pred*.parquet`
* **Пороги:** `artifacts/thresholds/*.json`
* **Зоны:** `artifacts/evaluation/decision_zones_*.csv`
* **Экономика:** `artifacts/evaluation/cost_*_test.json`
* **Отчёт:** `reports/report.html`

---

## 8) Batch inference (predict)

### 8.1 Tabular predict (example)

```bash
python -m scripts.07_predict_tabular `
  --input data/processed/test.parquet `
  --model artifacts/tabular/model.pkl `
  --thresholds artifacts/thresholds/thresholds_tabular.json `
  --feature-spec artifacts/evaluation/tabular_feature_spec.json `
  --out artifacts/predict/test_tabular.parquet
```

> На Windows PowerShell удобно переносить строки через backtick: `

---

## 9) Reproducibility / Git

### 9.1 What to commit

Коммитим:

* `src/`
* `scripts/`
* `configs/`
* `README.md`
* (опционально) `requirements*.txt`

Обычно **не коммитим** (генерируемое/большое):

* `data/raw/`
* `data/processed/`
* `artifacts/` (кроме мелких json/csv по необходимости)
* большие модели (`*.pt`, `*.pkl`)
* большие предикты (`*.parquet`)

### 9.2 Recommended workflow

После ключевых шагов делай коммит:

* `A2: prepare data + time split`
* `A3: train tabular + preds`
* `A4: build heterograph artifacts`
* `A9/A10: external honest mode`
* `A7/A11: thresholds + report`

---

## 10) Troubleshooting

### 10.1 PyG / torch install issues

Если `torch_geometric` не ставится или падает импорт — чаще всего проблема в несовпадении версий
(torch, python, cpu/cuda). Лечится установкой соответствующих колёс для твоей версии torch.

### 10.2 Graph viz ошибки (tx not in tx_index)

Если ты используешь `--auto-pick` и получаешь ошибку вида
`transaction_id=... not found in tx_index`, это значит что:

* auto-pick выбрал tx из предиктов, которых нет в текущем граф-индексе

Решение:

* пересобрать граф артефакты шагами `A4_build_graph` + `A4_make_graph_data`
* или явно указать tx-id из `tx_index.parquet`:

  ```bash
  python -m scripts.16_build_graph_viz --tx-id <ID>
  ```

> В актуальной версии проекта auto-pick уже выбирает tx из пересечения preds ∩ tx_index.

### 10.3 Missing artifacts

Если auto-report пишет `Not found`, значит соответствующий шаг не был запущен.
Запусти пайплайн до нужного шага:

```bash
python -m scripts.run_all --from-step A3_tabular --to-step A11_auto_report
```

---

## 11) What to show for VKR defense

Минимальный «набор для комиссии»:

1. **fusion_external (A10)** — честная оценка на future split
2. **evaluate (A7)** — пороги / зоны / экономика
3. **report (A11)** — `reports/report.html` (таблицы + картинки)
4. **one-click (A12)** — `python -m scripts.run_all`
5. (опционально) **graph viz** — `reports/assets/graph.html`

---

