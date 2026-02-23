# VKR Fraud System

Система обнаружения мошеннических банковских транзакций на основе гибридного подхода:
- **tabular scorer** (LightGBM)
- **graph scorer** (GNN на гетерографе: Hetero GraphSAGE)
- далее (следующие шаги): **fusion**, пороги решений allow/review/deny, explainability, predict из raw.

Проект предназначен для воспроизводимого запуска пайплайна:
**prepare → train → graph → gnn → (fusion) → evaluate → predict**

---

## Требования

- Python 3.10+
- PyCharm Community (опционально)
- Датасет: IEEE-CIS Fraud Detection (Kaggle)

---

## Установка

1) Создать виртуальное окружение (пример для Windows PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```
2) Установить зависимости:
```bash
pip install -r requirements.txt
```
Для CPU-версий torch/pyg можно использовать дополнительные файлы:
- requirements_torch_cpu.txt 
- requirements_pyg_cpu.txt

## Быстрая проверка окружения (A0)
```bash
python -m scripts.run_all
```
## Структура проекта
```commandline
vkr_fraud_system/
  configs/
    base.yaml
    schema_ieee_cis.yaml
  data/
    raw/                # сырые csv/parquet (не коммитятся)
    processed/          # подготовленные parquet (не коммитятся)
    splits/             # split_info.json (можно хранить)
  artifacts/
    tabular/            # model.pkl (не коммитится)
    graph/              # node_map, edges, graph_data.pt (не коммитится)
    evaluation/         # метрики, кривые, предикты (не коммитится)
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
```
## Архитектура пайплайна
raw → schema → merge → time split → processed → (tabular train / graph build / gnn train) → evaluation → (fusion) → predict

## A2. Подготовка данных (prepare + time split)
Запуск:
```bash
python -m scripts.00_prepare_data
```
#### Что делает скрипт:
1) Загружает сырые данные IEEE-CIS из:
   - data/raw/ieee-cis/train_transaction.csv 
   - data/raw/ieee-cis/train_identity.csv
2) Объединяет transaction + identity по TransactionID 
3) Применяет schema-мэппинг (configs/schema_ieee_cis.yaml):
    - приводит к каноническим колонкам: transaction_id, time, target 
    - выполняет валидацию и приведение типов
4) Делает time-based split (60% / 20% / 20%) без утечки по времени 
   5) Сохраняет:
    - data/processed/train.parquet 
    - data/processed/val.parquet 
    - data/processed/test.parquet 
    - data/splits/split_info.json 
    - artifacts/evaluation/columns.json

После выполнения этого шага данные считаются “единой истиной” для всех моделей.
## A3. Tabular baseline (LightGBM)
Запуск:
```bash
python -m scripts.01_train_tabular
```
Выходные артефакты:
   - модель: artifacts/tabular/model.pkl 
   - метрики: artifacts/evaluation/tabular_metrics.json 
   - PR-кривая: artifacts/evaluation/pr_curve_tabular.png 
   - предсказания:
     - artifacts/evaluation/val_pred_tabular.parquet 
     - artifacts/evaluation/test_pred_tabular.parquet

## A4. Построение гетерографа (node_map + edges + graph_data)
### A4.1 Build graph artifacts
Запуск:
```bash
python -m scripts.02_build_graph
```
Что сохраняется:
   - artifacts/graph/node_map.parquet 
   - artifacts/graph/edges.parquet 
   - artifacts/graph/graph_info.json

Особенности:
   - сущности строятся с фильтром редких значений min_freq 
   - значения сущностей префиксуются (например DeviceInfo::...), чтобы избежать смешения доменов

### A4.2 Convert to PyG HeteroData
Запуск:
```bash
python -m scripts.03_make_graph_data
```
Что сохраняется:
   - artifacts/graph/graph_data.pt (PyG HeteroData)
   - artifacts/graph/tx_index.parquet (transaction_id → tx_node_id)
## A5. GNN scorer (Hetero GraphSAGE, CPU)

Постановка:
   - node classification на узлах transaction 
   - граф построен на train-only периоде 
   - оценка проводится на time-based holdout внутри train-graph 
   - индуктивный режим для новых транзакций будет реализован на этапе predict (следующие шаги)

Запуск:
```bash
python -m scripts.04_train_gnn
```
Выходные артефакты:
   - модель: artifacts/graph/gnn_model.pt 
   - метрики: artifacts/evaluation/gnn_metrics.json 
   - PR-кривая: artifacts/evaluation/pr_curve_gnn.png 
   - предсказания:
     - artifacts/evaluation/val_pred_gnn.parquet 
     - artifacts/evaluation/test_pred_gnn.parquet 
     - artifacts/evaluation/train_pred_gnn.parquet (если включено)




## Примечания

Папки data/processed/ и artifacts/ являются генерируемыми и обычно не коммитятся в git.

Все параметры путей и настроек лежат в configs/base.yaml.

## A6. Fusion (Stacking, train-only graph mode)
В текущей версии GNN построен на train-периоде и использует внутренний time-split.

Fusion обучается в режиме:
   - tabular(train)\
   - gnn(train / val_internal / test_internal)

⚠️ Результаты fusion в этом режиме являются внутренними (internal) и используются для анализа взаимодействия скореров.

Для финальной оценки обобщающей способности требуется индуктивный режим GNN (см. A7).

Запуск:
```bash
python -m scripts.05_train_fusion
```
Выход:
   - artifacts/fusion/fusion.pkl 
   - artifacts/evaluation/fusion_metrics_internal.json 
   - artifacts/evaluation/*_fusion_internal.parquet 
   - artifacts/evaluation/pr_curve_fusion_internal.png

## A7. Evaluate + thresholds + decisions (allow/review/deny)

На этапе A7 подбираются пороги решений для системы на основе вероятности мошенничества.

Политика решений:
- `p >= T_deny`  → **DENY**
- `T_review <= p < T_deny` → **REVIEW**
- `p < T_review` → **ALLOW**

Пороги подбираются на **валидационной выборке** (VAL) и фиксируются,
после чего оцениваются на **тестовой выборке** (TEST).

По умолчанию используются ограничения:
- максимальный FPR для зоны DENY: `max_fpr_deny = 0.01`
- максимальная доля операций в зоне REVIEW: `max_review_share = 0.10`

Запуск:

```bash
python -m scripts.06_evaluate
```
Выходные файлы:
- `artifacts/thresholds/thresholds_tabular.json` — выбранные пороги `T_review`, `T_deny`
- `artifacts/evaluation/decision_zones_tabular_val.csv` — таблица зон на VAL
- `artifacts/evaluation/decision_zones_tabular_test.csv` — таблица зон на TEST
- `artifacts/evaluation/decision_binary_tabular_val.csv` — метрики deny-порога на VAL (Precision/Recall/FPR)
- `artifacts/evaluation/decision_binary_tabular_test.csv` — метрики deny-порога на TEST
- `artifacts/evaluation/evaluate_summary.json` — сводка метрик + политика порогов

Пороги подбираются на VAL и применяются на TEST без переобучения.

### A7.2 (дополнительно)
Скрипт также строит:
- графики долей операций по зонам (VAL/TEST)
- простую модель экономических потерь (cost) на TEST

Выход:
- `artifacts/evaluation/zone_share_tabular_val.png`
- `artifacts/evaluation/zone_share_tabular_test.png`
- `artifacts/evaluation/cost_tabular_test.json`