# VKR Fraud System

**Тема ВКР:** *Система обнаружения мошеннических банковских транзакций*  
Реализована воспроизводимая система на основе **гибридного подхода**:

- **Tabular scorer:** LightGBM (табличная модель)
- **Graph scorer:** GNN на гетерографе (Hetero GraphSAGE, PyTorch Geometric)
- **Fusion:** логистическая регрессия по выходам скореров (**честный режим** = обучение на external VAL, тест на external TEST)
- **Policy:** пороги `ALLOW/REVIEW/DENY` + таблицы зон + простая экономика (cost)
- **Auto-report:** HTML отчёт (таблицы + картинки) пригодный для РПЗ/презентации
- **One-click:** полный запуск одной командой (A12)

---

## 0) Требования

- Python **3.10+**
- Датасет: **IEEE-CIS Fraud Detection** (Kaggle), файлы:
  - `train_transaction.csv`
  - `train_identity.csv`

---

## 1) Установка

### 1.1 Виртуальное окружение (Windows PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\activate
````

### 1.2 Зависимости

Базово:

```bash
pip install -r requirements.txt
```

Если используешь CPU torch/pyg (как у тебя):

* `requirements_torch_cpu.txt`
* `requirements_pyg_cpu.txt`

---

## 2) Быстрый старт (одна кнопка, A12)

Полный прогон пайплайна (если артефакты уже есть — шаги будут пропускаться):

```bash
python -m scripts.run_all
```

Полезные режимы:

```bash
# пересчитать всё (если хочешь “с нуля”)
python -m scripts.run_all --force

# пропустить внешнюю часть (A9/A10)
python -m scripts.run_all --skip-external

# прогнать только оценку+отчёт (если модели уже готовы)
python -m scripts.run_all --from-step A7_evaluate --to-step A11_auto_report
```

**Результат:** итоговый отчёт: `reports/report.html`

---

## 3) Структура проекта

```text
vkr_fraud_system/
  configs/
    base.yaml
    schema_ieee_cis.yaml
  data/
    raw/                # сырые csv/parquet (не коммитятся)
    processed/          # подготовленные parquet (не коммитятся)
    splits/             # split_info.json
  artifacts/
    tabular/            # model.pkl (не коммитится)
    graph/              # node_map, edges, graph_data.pt, gnn_model.pt (не коммитится)
    fusion/             # fusion.pkl, fusion_external.pkl (не коммитится)
    thresholds/         # thresholds_*.json
    evaluation/         # метрики, кривые, предикты (не коммитится)
  reports/
    report.html
    assets/
    tables/
  src/
    fraud_system/
      io/
      data/
      features/
      models/
      graph/
      evaluation/
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
    run_all.py
```

---

## 4) Архитектура пайплайна

**train path:**

raw → schema → merge → time split → processed →
(tabular train) + (graph build → gnn train) → (fusion) → evaluate → report

**inference path (система):**

raw → schema → features → tabular + (gnn inductive) → fusion_external → thresholds → decision

---

## 5) Пошаговый запуск (если без A12)

### A2. Подготовка данных (prepare + time split)

```bash
python -m scripts.00_prepare_data
```

Что делает:

1. Загружает IEEE-CIS:

   * `data/raw/ieee-cis/train_transaction.csv`
   * `data/raw/ieee-cis/train_identity.csv`
2. Merge по TransactionID
3. Schema mapping (`configs/schema_ieee_cis.yaml`):

   * канонические поля: `transaction_id`, `time`, `target`
4. Time-based split (60/20/20) без утечки по времени
5. Сохраняет:

   * `data/processed/train.parquet`
   * `data/processed/val.parquet`
   * `data/processed/test.parquet`
   * `data/splits/split_info.json`
   * `artifacts/evaluation/columns.json`

---

### A3. Tabular baseline (LightGBM)

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

### A4. Построение гетерографа

#### A4.1 Build graph artifacts

```bash
python -m scripts.02_build_graph
```

Артефакты:

* `artifacts/graph/node_map.parquet`
* `artifacts/graph/edges.parquet`
* `artifacts/graph/graph_info.json`

#### A4.2 Convert to PyG HeteroData

```bash
python -m scripts.03_make_graph_data
```

Артефакты:

* `artifacts/graph/graph_data.pt` (PyG HeteroData)
* `artifacts/graph/tx_index.parquet` (transaction_id → tx_node_id)

---

### A5. GNN scorer (Hetero GraphSAGE, internal)

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

Примечание: internal режим используется для абляции/отладки (не является “честной” оценкой на будущих данных).

---

### A6. Fusion internal (ablation)

```bash
python -m scripts.05_train_fusion
```

Артефакты:

* `artifacts/fusion/fusion.pkl`
* `artifacts/evaluation/fusion_metrics_internal.json`
* `artifacts/evaluation/pr_curve_fusion_internal.png`

---

### A9. Inductive GNN predict (external, честный режим для графа)

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

### A9.3 Calibration (temperature scaling)

```bash
python -m scripts.09_calibrate_gnn
```

Артефакты:

* `artifacts/graph/gnn_temperature.json`
* `artifacts/evaluation/val_pred_gnn_external_calibrated.parquet`
* `artifacts/evaluation/test_pred_gnn_external_calibrated.parquet`

---

### A10. Fusion external (главный режим системы)

**Это основной “честный” режим ВКР:**

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

### A7. Evaluate + thresholds + decisions (ALLOW/REVIEW/DENY) + cost

```bash
python -m scripts.06_evaluate
```

Что делает:

* считает метрики (logloss / PR-AUC / ROC-AUC)
* подбирает пороги `T_review`, `T_deny` на VAL
* строит таблицы зон и доли зон на VAL/TEST
* считает простую модель экономических потерь (cost)

Выходные файлы (tabular и fusion_external):

* `artifacts/thresholds/thresholds_tabular.json`
* `artifacts/thresholds/thresholds_fusion_external.json`
* `artifacts/evaluation/decision_zones_*.csv`
* `artifacts/evaluation/decision_binary_*.csv`
* `artifacts/evaluation/zone_share_*.png`
* `artifacts/evaluation/cost_*_test.json`

---

### A11. Auto-report (таблицы + картинки “для РПЗ/презы”)

```bash
python -m scripts.11_auto_report
```

Артефакты:

* `reports/report.html`
* `reports/tables/model_comparison.csv`
* `reports/assets/*.png`

---

## 6) Batch Predict (Tabular + Policy)

Пакетный инференс для табличной модели (пример):

```bash
python -m scripts.07_predict_tabular \
  --input data/processed/test.parquet \
  --model artifacts/tabular/model.pkl \
  --thresholds artifacts/thresholds/thresholds_tabular.json \
  --feature-spec artifacts/evaluation/tabular_feature_spec.json \
  --out artifacts/predict/test_tabular.parquet
```

---

## 7) Git / воспроизводимость

Рекомендации:

* `data/processed/` и `artifacts/` — генерируемые, обычно **не коммитятся**
* Коммитим: `src/`, `scripts/`, `configs/`, `README.md`
* После каждого шага (A2..A12) делаем commit с понятным сообщением вида:

  * `A10: train fusion external (honest mode)`
  * `A11: auto report for RПЗ`

---

## 8) Главный режим ВКР (что показывать комиссии)

1. `fusion_external` (A10) — честная оценка на future split
2. `scripts.06_evaluate` (A7) — пороги/зоны/экономика
3. `reports/report.html` (A11) — пакет результатов для РПЗ/презы
4. `python -m scripts.run_all` (A12) — “система одной командой”

````
