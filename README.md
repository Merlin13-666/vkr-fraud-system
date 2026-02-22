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