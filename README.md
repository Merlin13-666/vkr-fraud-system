# VKR Fraud System — гибридная таблично-графовая система антифрода

**Тема ВКР:** *Система обнаружения мошеннических банковских транзакций*  
**Цель проекта:** воспроизводимая “система” (pipeline → артефакты → отчёт → policy), которая объединяет
табличную модель и GNN на гетерографе, и выдаёт финальное решение **ALLOW / REVIEW / DENY**.

---

## Что внутри (коротко)

- **Tabular scorer:** LightGBM по табличным признакам.
- **Graph scorer:** GNN на гетерографе (Hetero GraphSAGE / PyTorch Geometric).
- **Fusion (main):** логистическая регрессия поверх скореров.
  - **честный режим:** обучение мета-модели на `external VAL`, тест на `external TEST` (future split).
- **Policy:** подбор порогов `t_review / t_deny` + таблицы decision zones + доли зон + простая экономика (cost).
- **Graph Viz:** интерактивная визуализация ego-графа для расследования кейса (PyVis HTML). :contentReference[oaicite:3]{index=3}
- **Auto-report:** HTML-отчёт “под РПЗ/защиту” (таблицы + графики + ссылки на артефакты).
- **One-click (A12):** полный прогон одной командой (`python -m scripts.run_all`). :contentReference[oaicite:4]{index=4}

---

## Содержание

- [1. Быстрый старт](#1-быстрый-старт)
- [2. Установка](#2-установка)
- [3. Данные](#3-данные)
- [4. Структура проекта](#4-структура-проекта)
- [5. Пайплайн (A2…A11)](#5-пайплайн-a2a11)
- [6. One-click runner (A12)](#6-one-click-runner-a12)
- [7. Артефакты и результаты](#7-артефакты-и-результаты)
- [8. Запуск каждого шага вручную](#8-запуск-каждого-шага-вручную)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Что показывать на защите](#10-что-показывать-на-защите)

---

## 1. Быстрый старт

### 1.1 Полный прогон пайплайна (A12)
```bash
python -m scripts.run_all
````

**Результат:** `reports/report.html` + `reports/assets/*` + `reports/tables/*`

### 1.2 Пересчитать всё заново

```bash
python -m scripts.run_all --force
```

> Логика run_all: шаг запускается, если `--force` или если не существуют ожидаемые outputs. 

---

## 2. Установка

### 2.1 Python / окружение

* Python **3.10+**
* Windows PowerShell примеры ниже

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2.2 Зависимости

> Важно: torch/pyg ставятся в зависимости от платформы. Если PyG падает при установке — см. Troubleshooting.

```bash
pip install -r requirements_torch_cpu.txt
pip install -r requirements_pyg_cpu.txt
pip install -r requirements.txt
```

---

## 3. Данные

Используется датасет **IEEE-CIS Fraud Detection**.

Минимально нужны:

* `train_transaction.csv`
* `train_identity.csv`

Ожидаемое расположение:

```text
data/raw/ieee-cis/
  train_transaction.csv
  train_identity.csv
```

---

## 4. Структура проекта

```text
vkr_fraud_system/
  configs/
  data/
    raw/ieee-cis/
    processed/
    splits/
  artifacts/
    tabular/
    graph/
    fusion/
    thresholds/
    evaluation/
  reports/
    report.html
    assets/
    tables/
  src/fraud_system/
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
    17_graph_stats.py
    18_train_graph_metrics_baseline.py
    19_ablation_gnn.py
    20_graph_robustness.py
    run_all.py
```

---

## 5. Пайплайн (A2…A11)

Ниже — “идеальная цепочка” для полного эксперимента и отчёта:

1. **A2** Prepare data → `data/processed/*.parquet`, `data/splits/split_info.json`
2. **A3** Train tabular → `artifacts/tabular/*`, PR-кривая, предикты
3. **A4.1** Build graph → `artifacts/graph/node_map.parquet`, `edges.parquet`
4. **A4.2** Make PyG data → `graph_data.pt`, `tx_index.parquet`
5. **A4.3** Graph stats → графики топологии
6. **A6.1** Graph-metrics baseline → контрольная модель “только топология”
7. **A5** Train GNN (internal) → отладка/абляция
8. **A6** Fusion internal → отладка/абляция
9. **A9** Predict GNN external (val/test) → inductive предикты (честный режим для графа) 
10. **A9.3** Calibrate GNN → calibrated предикты
11. **A10** Fusion external (MAIN) → честные метрики/предикты
12. **A7** Evaluate → thresholds/zones/cost
13. **A11** Graph viz (PyVis) → `reports/assets/graph.html` 
14. **A11** Auto-report → `reports/report.html`

---

## 6. One-click runner (A12)

### 6.1 Запуск

```bash
python -m scripts.run_all
```

### 6.2 Пересчёт всего

```bash
python -m scripts.run_all --force
```

### 6.3 Запуск только отчёта (если артефакты уже есть)

У run_all есть режим “only report” (граф-виз + авто-отчёт).
(Если у тебя этот флаг включён в текущей версии run_all.)

Пример:

```bash
python -m scripts.run_all --only-report
```

### 6.4 Запуск подграфа / от-до шагов

Если у тебя включены флаги `--from-step` / `--to-step` — удобно пересчитывать кусок пайплайна, например:

```bash
python -m scripts.run_all --from-step A7_evaluate --to-step A11_auto_report
```

> Важно: run_all запускает шаги по факту наличия outputs (кэширование шагов). 

---

## 7. Артефакты и результаты

### 7.1 Главный результат для ВКР

**Основной режим:** `fusion_external` (future split, честная оценка)

Смотри:

* `artifacts/evaluation/fusion_metrics_external.json`
* `reports/report.html`

### 7.2 Где смотреть что именно

* Метрики: `artifacts/evaluation/*metrics*.json`
* Предикты: `artifacts/evaluation/*pred*.parquet`
* Пороги: `artifacts/thresholds/*.json`
* Decision zones: `artifacts/evaluation/decision_zones_*.csv`
* Доли зон: `artifacts/evaluation/zone_share_*.png`
* Экономика: `artifacts/evaluation/cost_*_test.json`
* Итоговый отчёт: `reports/report.html`
* Визуализация графа: `reports/assets/graph.html`

---

## 8. Запуск каждого шага вручную

Ниже — команды (как в отчёте/презентации удобно показывать “по блокам”).

### A2) Prepare data

```bash
python -m scripts.00_prepare_data
```

**Что делает:** загрузка CSV → merge → schema mapping → time split → parquet + split_info.json

---

### A3) Train Tabular (LightGBM)

```bash
python -m scripts.01_train_tabular
```

**Выходы:** модель, PR-кривая, метрики, предикты `val/test`

---

### A4.1) Build graph artifacts

```bash
python -m scripts.02_build_graph
```

**Выходы:** `node_map.parquet`, `edges.parquet`, `graph_info.json`

---

### A4.2) Convert to PyG HeteroData

```bash
python -m scripts.03_make_graph_data
```

**Выходы:** `graph_data.pt`, `tx_index.parquet`

---

### A4.3) Graph topology stats

```bash
python -m scripts.17_graph_stats
```

**Выходы:** графики степени, компонент, распределения рёбер и т.д.

---

### A6.1) Graph-metrics baseline (без GNN)

```bash
python -m scripts.18_train_graph_metrics_baseline
```

**Смысл:** показать вклад “одной топологии” без нейросети (контрольная модель).

---

### A5) Train GNN (internal)

```bash
python -m scripts.04_train_gnn --device cpu
```

**Смысл:** отладка GNN в internal-режиме (внутри train-графа), не выдаётся за честный production.

---

### A6) Fusion internal

```bash
python -m scripts.05_train_fusion
```

**Смысл:** абляция/отладка fusion на internal предиктах.

---

### A9) Inductive GNN predict (external val/test)

```bash
python -m scripts.08_predict_gnn_external --split val
python -m scripts.08_predict_gnn_external --split test
```

Параметр:

* `--split {val|test}` 

---

### A9.3) Calibration (temperature scaling)

```bash
python -m scripts.09_calibrate_gnn
```

**Смысл:** калибровать вероятности external GNN (чтобы fusion/thresholds работали стабильнее).

---

### A10) Fusion external (MAIN)

```bash
python -m scripts.10_train_fusion_external
```

**Смысл:** честная мета-модель: обучается на external VAL, тестируется на external TEST.

---

### A7) Evaluate (thresholds / zones / cost)

```bash
python -m scripts.06_evaluate
```

**Выходы:** thresholds + decision zones + доли зон + cost.

---

### A11) Graph visualization (PyVis)

```bash
# Автовыбор “самой рискованной” транзакции из pred parquet:
python -m scripts.16_build_graph_viz --auto-pick --pred-path artifacts/evaluation/val_pred_fusion_external.parquet

# Или явно по transaction_id:
python -m scripts.16_build_graph_viz --mode ego_tx --tx-id 123456789

# Или ego-граф вокруг сущности:
python -m scripts.16_build_graph_viz --mode ego_entity --entity-type card --entity-value 1234
```

Параметры (основные): 

* `--mode {ego_tx|ego_entity}`
* `--tx-id <int>` (для ego_tx)
* `--auto-pick` + `--pred-path <parquet>` (для ego_tx без ручного id)
* `--entity-type <str>` + `--entity-value <str>` (для ego_entity)
* `--hops <int>` — радиус обхода
* `--max-nodes <int>`, `--max-edges <int>` — ограничения на размер
* `--show-physics` — включить “физику” в визуализации

Выход: `reports/assets/graph.html`

---

### A11) Auto-report

```bash
python -m scripts.11_auto_report
```

Выход: `reports/report.html`

---

## 9. Troubleshooting

### 9.1 “Данные и разбиение” пустые / train/val/test = None

Это почти всегда значит одно из двух:

1. **Файл `data/splits/split_info.json` не создан** → не запускался A2, или упал.
2. **В split_info.json другие ключи**, чем ожидает отчёт (например, `train_rows` вместо `train_size`).

**Что сделать:**

* Перезапусти A2:

  ```bash
  python -m scripts.00_prepare_data
  ```
* Открой `data/splits/split_info.json` и проверь поля:

  * ожидаются `train_size`, `val_size`, `test_size` и `*_fraud_rate`.

---

### 9.2 “PR-AUC vs layers” и “PR-AUC vs edge drop%” не видно в отчёте

Самая частая причина — **картинки не копируются в `reports/assets/`**, или в HTML вставлен путь не из `reports/assets`, а напрямую из `artifacts/evaluation/...`.

**Правильно:** отчёт должен ссылаться на файлы, которые реально лежат в `reports/assets/`.
Проверь наличие:

* `reports/assets/gnn_ablation_pr_auc_vs_layers.png`
* `reports/assets/graph_robustness_pr_auc_vs_drop.png`

Если их нет — запусти соответствующие шаги:

```bash
python -m scripts.19_ablation_gnn
python -m scripts.20_graph_robustness
python -m scripts.11_auto_report
```

---

### 9.3 GraphViz ругается “tx not found in tx_index”

Это значит: транзакция из pred-файла отсутствует в текущем граф-индексе.

Решение:

* пересобрать граф:

  ```bash
  python -m scripts.02_build_graph
  python -m scripts.03_make_graph_data
  ```
* или указывать `--tx-id` вручную (из `artifacts/graph/tx_index.parquet`).

---

## 10. Что показывать на защите

Минимальный “комиссионный пакет”:

1. **Главный режим (A10)**: `fusion_external` метрики + PR-кривая
2. **Policy (A7)**: thresholds + зоны + доли зон + cost
3. **Auto-report (A11)**: `reports/report.html`
4. **One-click (A12)**: `python -m scripts.run_all`
5. (опционально) **GraphViz (A11)**: `reports/assets/graph.html`

---

## Лицензия / дисклеймер

Проект учебно-исследовательский. Не является банковской production-системой.
Датасет IEEE-CIS используется в рамках условий Kaggle/источника.

```
