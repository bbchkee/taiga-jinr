
# iact-tools

Набор утилит для анализа данных TAIGA-IACT (Cherenkov). Включает загрузку,
реконструкцию энергии, базовые каты, построение спектров и распределения `theta^2`.
Для корректной работы необходим python 3.9 или выше.

## Структура
- `iact_tools/` — код пакета
  - `data_loading.py` — загрузка CSV и быстрые гистограммы
  - `models.py` — мат. модели и `theta^2`
  - `reconstruction.py` — реконструкция энергии и каты
  - `analysis.py` — графики спектров и `theta^2`
  - `regressor.py` — Torch-регрессор энергии
  - `utils.py` — вспомогательное
- `scripts/run_pipeline.py` — CLI-скрипт
- `docs/` — документация
- `tests/` — место под тесты

## Быстрый старт
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m scripts.run_pipeline   --model-path /model/csv/folder   --exp-path exp/csv/folder --exp-pattern "*.csv" --sums edist/file/path  --throw-radius 1000 --eff-bins 20  --unfold --tau-auto

```


## Лицензия
GNU GPL v3
