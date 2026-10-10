"""Шаг 0. Сравнение датасетов-кандидатов: есть ли в данных сигнал.

Проверяется синтетический CRM-датасет Maven Analytics (CRM Sales
Opportunities): можно ли по данным на момент начала сделки предсказать её
исход. Результат обосновывает выбор основного источника.
"""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import CANDIDATES, MAVEN_BASE, MAVEN_FILES, save_json


def main() -> dict:
    for f in MAVEN_FILES:
        p = CANDIDATES / f
        if not p.exists():
            urllib.request.urlretrieve(MAVEN_BASE + f, p)
    pl = pd.read_csv(CANDIDATES / "sales_pipeline.csv", parse_dates=["engage_date", "close_date"])
    acc = pd.read_csv(CANDIDATES / "accounts.csv")
    team = pd.read_csv(CANDIDATES / "sales_teams.csv")
    prod = pd.read_csv(CANDIDATES / "products.csv")
    c = pl[pl.deal_stage.isin(["Won", "Lost"])].copy()
    c["y"] = (c.deal_stage == "Won").astype(int)
    c = c.merge(acc, on="account", how="left").merge(team, on="sales_agent", how="left")
    c = c.sort_values("engage_date")
    X = pd.get_dummies(c[["sales_agent", "product", "sector", "regional_office", "office_location"]]
                       .fillna("NA")).astype(float)
    X["log_rev"] = np.log1p(c.revenue.fillna(0)); X["log_emp"] = np.log1p(c.employees.fillna(0))
    n = int(len(c) * 0.8)
    m = LogisticRegression(max_iter=3000, C=0.3).fit(X[:n], c.y[:n])
    auc = roc_auc_score(c.y[n:], m.predict_proba(X[n:])[:, 1])
    win_by_agent = c.groupby("sales_agent").y.mean()
    res = {
        "maven_rows": len(pl), "maven_closed": len(c), "maven_win_rate": float(c.y.mean()),
        "maven_auc": float(auc),
        "maven_agent_win_min": float(win_by_agent.min()), "maven_agent_win_max": float(win_by_agent.max()),
        "maven_product_mismatch": sorted(set(pl["product"]) - set(prod["product"])),
        "maven_sector_typo": "technolgy" in set(acc.sector),
        "maven_account_na": int(pl.account.isna().sum()),
        "maven_period": [str(pl.engage_date.min().date()), str(pl.close_date.max().date())],
    }
    save_json(res, "candidates.json")
    print(res)
    return res


if __name__ == "__main__":
    main()
