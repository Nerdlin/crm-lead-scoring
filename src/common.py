"""Общие пути, константы и утилиты проекта (аналитическая часть РК-1)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FEATURES = DATA / "features"
CANDIDATES = DATA / "candidates"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

for _p in (RAW, PROCESSED, FEATURES, CANDIDATES, REPORTS, FIG):
    _p.mkdir(parents=True, exist_ok=True)

# Первоисточник: UCI Machine Learning Repository (на Kaggle — зеркало
# henriqueyamahata/bank-marketing). Файл скачивается автоматически,
# целостность проверяется по SHA-256.
SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
SOURCE_SHA256 = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
KAGGLE_URL = "https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing"
RAW_ZIP = RAW / "bank_marketing.zip"
RAW_CSV = RAW / "bank-additional-full.csv"

# Кандидат, который проверялся и был отклонён (синтетический CRM-датасет).
MAVEN_BASE = "https://raw.githubusercontent.com/ChitikaneniVarsha/Maven-Sales-Challenge/master/"
MAVEN_FILES = ["accounts.csv", "products.csv", "sales_pipeline.csv", "sales_teams.csv"]

SEED = 42
TEST_SHARE = 0.20          # последние 20% контактов по времени — тест
TOP_K = 0.20               # бизнес-метрика: recall в первых 20% списка обзвона

# Допущения о цене ошибок (тенге), согласовать с заказчиком.
COST_FN = 150_000          # упущенная запись на программу (маржа с программы)
COST_FP = 5_000            # лишний звонок: время оператора + риск отписки/жалобы

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Поля, которые становятся известны только ПОСЛЕ звонка — утечка.
LEAKAGE_COLS = ["duration"]
MACRO_COLS = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, name: str) -> None:
    with open(REPORTS / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def load_json(name: str):
    with open(REPORTS / name, encoding="utf-8") as f:
        return json.load(f)
