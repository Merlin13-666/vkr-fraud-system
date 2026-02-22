# VKR Fraud System

Система обнаружения мошеннических банковских транзакций (табличная модель + графовая модель + гибридная fusion).
Проект предназначен для воспроизводимого запуска пайплайна: prepare → train → evaluate → predict.

## Установка
1) Создать venv
2) Установить зависимости:
   pip install -r requirements.txt

## Запуск
Проверка окружения:
python -m scripts.run_all

(Далее будут добавлены скрипты пайплайна: 00_prepare_data.py, 01_train_tabular.py, ...)

## Подготовка данных

Перед обучением моделей необходимо выполнить подготовку данных:

```bash
python -m scripts.00_prepare_data
```
### Что делает скрипт:

1) Загружает сырые данные IEEE-CIS из:
   - data/raw/ieee-cis/train_transaction.csv 
   - data/raw/ieee-cis/train_identity.csv 
2) Объединяет transaction + identity по TransactionID 
3) Применяет schema-мэппинг (configs/schema_ieee_cis.yaml):
   - приводит к каноническим колонкам: transaction_id, time, target 
   - выполняет валидацию 
   - приводит типы
4) Выполняет time-based split (60% / 20% / 20%)
5) Сохраняет:
   - data/processed/train.parquet 
   - data/processed/val.parquet 
   - data/processed/test.parquet 
   - data/splits/split_info.json 
   - artifacts/evaluation/columns.json

После выполнения этого шага данные считаются "единой истиной"
для всех моделей (tabular, graph, fusion).
---
## Структура проекта

```markdown
vkr_fraud_system/
  configs/
  data/
    raw/
    processed/
    splits/
  artifacts/
    evaluation/
  src/
    fraud_system/
      io/
      data/
  scripts/
    00_prepare_data.py
```
## Архитектура пайплайна

raw → schema → merge → time split → processed → models → evaluation → predict