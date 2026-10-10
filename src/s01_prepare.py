"""Шаг 1. Загрузка raw, проверка SHA-256, слои processed и features."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
import io

import numpy as np
import pandas as pd

from common import (CANDIDATES, FEATURES, MONTHS, PROCESSED, RAW_CSV, RAW_ZIP,
                    SOURCE_SHA256, SOURCE_URL, save_json, sha256)


def download_raw() -> None:
    if not RAW_ZIP.exists():
        local = CANDIDATES / "bank_marketing.zip"
        if local.exists():
            shutil.copy(local, RAW_ZIP)
        else:
            print("Скачиваю", SOURCE_URL)
            urllib.request.urlretrieve(SOURCE_URL, RAW_ZIP)
    digest = sha256(RAW_ZIP)
    assert digest == SOURCE_SHA256, f"SHA-256 не совпадает: {digest}"
    if not RAW_CSV.exists():
        with zipfile.ZipFile(RAW_ZIP) as outer:
            inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
            RAW_CSV.write_bytes(inner.read("bank-additional/bank-additional-full.csv"))


def reconstruct_time(df: pd.DataFrame) -> pd.DataFrame:
    """Года в выгрузке нет. Строки упорядочены по дате звонка (май 2008 –
    ноябрь 2010), поэтому год увеличивается, когда номер месяца «откатывается»
    назад более чем на 2 позиции."""
    m = df["month"].map({k: i for i, k in enumerate(MONTHS, 1)}).to_numpy()
    year = np.empty(len(m), dtype=int)
    year[0] = 2008
    for i in range(1, len(m)):
        year[i] = year[i - 1] + int(m[i - 1] - m[i] > 2)
    df["year"] = year
    df["month_num"] = m
    df["period"] = pd.to_datetime(dict(year=year, month=m, day=1))
    return df


def main() -> dict:
    download_raw()
    raw = pd.read_csv(RAW_CSV, sep=";")
    facts = {"raw_rows": len(raw), "raw_cols": raw.shape[1],
             "raw_mb": RAW_CSV.stat().st_size / 2**20, "sha256": SOURCE_SHA256}

    df = raw.copy()
    df.columns = [c.replace(".", "_") for c in df.columns]
    df.insert(0, "contact_seq", np.arange(len(df)))      # порядок звонков
    df = reconstruct_time(df)
    df["y"] = (df["y"] == "yes").astype(int)

    # скрытые пропуски -> NaN (исходные значения сохраняются в raw)
    for c in ["job", "marital", "education", "default", "housing", "loan"]:
        df[c] = df[c].replace("unknown", np.nan)
    # pdays = 999 противоречит previous > 0 в 4110 строках (см. чек-лист, п. 5),
    # поэтому флаг строится по previous
    df["never_contacted"] = (df["previous"] == 0).astype(int)
    df["pdays"] = df["pdays"].replace(999, np.nan)

    # полные дубли (все 21 исходное поле совпадают, включая длительность
    # звонка и макропоказатели) — технические повторы, удаляем
    dup = raw.duplicated()
    facts["dups_removed"] = int(dup.sum())
    df = df.loc[~dup.to_numpy()].reset_index(drop=True)
    df.to_parquet(PROCESSED / "contacts.parquet", index=False)
    facts["processed_rows"] = len(df)

    # features: только то, что известно ДО звонка
    feat = df[["contact_seq", "year", "month_num", "period", "y",
               "age", "job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "campaign", "pdays", "previous",
               "poutcome", "never_contacted", "emp_var_rate", "cons_price_idx",
               "cons_conf_idx", "euribor3m", "nr_employed", "duration"]].copy()
    feat["campaign_clip"] = feat["campaign"].clip(upper=feat["campaign"].quantile(0.99))
    feat["prev_success"] = (feat["poutcome"] == "success").astype(int)
    feat.to_parquet(FEATURES / "contacts_features.parquet", index=False)
    save_json(facts, "prepare_facts.json")
    print(facts)
    return facts


if __name__ == "__main__":
    main()
