"""Шаг 4. Baseline: B0 константа, B1 правило, B2 логрегрессия, B2+ и L — утечка.
Стилизованные графики потерь, прироста маржи и кривой обзвона.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (COST_FN, COST_FP, FEATURES, FIG, REPORTS, SEED, TEST_SHARE, TOP_K,
                    save_json)
from viz_theme import apply_theme, TEAL_DARK, TEAL_MID, CORAL, RUST, DARK_SLATE

apply_theme()

CAT = ["job", "marital", "education", "default", "housing", "loan", "contact",
       "day_of_week", "poutcome"]
NUM = ["age", "campaign_clip", "previous", "never_contacted"]
CAT_PLUS = CAT + ["month"]
MACRO = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]

RNG = np.random.default_rng(SEED)

def make_lr(cat, num, C=0.5):
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=5000))])

def rule_scores(tr, te, keys, alpha=20):
    prior = tr.y.mean()
    g = tr.groupby(keys, dropna=False).y.agg(["sum", "count"])
    sm = (g["sum"] + alpha * prior) / (g["count"] + alpha)
    return te[keys].merge(sm.rename("s").reset_index(), on=keys, how="left")["s"].fillna(prior).to_numpy()

_TIE = None

def order(p):
    return np.lexsort((_TIE[: len(p)], -p))

def at_k(y, p, k=TOP_K):
    idx = order(p)[: int(np.ceil(len(y) * k))]
    return y[idx].mean(), y[idx].sum() / y.sum()

def wave_effect(y, p, k=TOP_K):
    idx = order(p)[: int(np.ceil(len(y) * k))]
    return (y[idx].sum() * COST_FN - len(idx) * COST_FP) / 1e6

def main() -> dict:
    global _TIE
    df = pd.read_parquet(FEATURES / "contacts_features.parquet").sort_values("contact_seq")
    for c in CAT_PLUS:
        df[c] = df[c].astype(object)
    n_tr = int(len(df) * (1 - TEST_SHARE))
    tr, te = df.iloc[:n_tr], df.iloc[n_tr:]
    y_tr, y_te = tr.y.to_numpy(), te.y.to_numpy()
    _TIE = RNG.random(len(te))

    models = {}
    models["B0: Случайный порядок (Константа)"] = np.full(len(te), y_tr.mean())
    models["B1: Экспертное правило (Канал × Опыт)"] = rule_scores(tr, te, ["poutcome", "contact"])
    b2 = make_lr(CAT, NUM).fit(tr, y_tr)
    models["B2: Логистическая регрессия (Baseline)"] = b2.predict_proba(te)[:, 1]
    b2m = make_lr(CAT_PLUS, NUM + MACRO).fit(tr, y_tr)
    models["B2+: Baseline + Макроконтекст"] = b2m.predict_proba(te)[:, 1]
    lk = make_lr(CAT_PLUS, NUM + MACRO + ["duration"]).fit(tr, y_tr)
    models["L: Модель с утечкой (+ Длительность*)"] = lk.predict_proba(te)[:, 1]

    rows = []
    for name, p in models.items():
        prec, rec = at_k(y_te, p)
        auc = roc_auc_score(y_te, p) if np.unique(p).size > 1 else 0.5
        rows.append({"Модель": name, "ROC-AUC": auc, "PR-AUC": average_precision_score(y_te, p),
                     "Precision@20%": prec, "Recall@20%": rec, "Lift@20%": rec / TOP_K,
                     "Эффект волны 20%, млн ₸": wave_effect(y_te, p)})
    met = pd.DataFrame(rows)
    met.to_csv(REPORTS / "baseline_metrics.csv", index=False, encoding="utf-8-sig")

    # Контроль на случайном разбиении
    a, b = train_test_split(df, test_size=TEST_SHARE, random_state=SEED, stratify=df.y)
    auc_rand = roc_auc_score(b.y, make_lr(CAT, NUM).fit(a, a.y).predict_proba(b)[:, 1])
    auc_rand_m = roc_auc_score(b.y, make_lr(CAT_PLUS, NUM + MACRO).fit(a, a.y).predict_proba(b)[:, 1])

    names = b2.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(b2.named_steps["lr"].coef_[0], index=names).sort_values()
    coef.to_csv(REPORTS / "b2_coefficients.csv", encoding="utf-8-sig")

    # Рисунок 10: Стильное оформление кривой добавочной маржи и cumulative gain
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    model_colors = ["#CED4DA", TEAL_MID, TEAL_DARK, "#5A189A", RUST]
    ks = np.arange(0.05, 1.0001, 0.05)
    base_rate = y_te.mean()
    inc = {}

    for (name, p), c in zip(models.items(), model_colors):
        o = order(p)
        vals = [(y_te[o[: int(np.ceil(len(o) * k))]].sum() - base_rate * int(np.ceil(len(o) * k))) * COST_FN / 1e6
                for k in ks]
        inc[name] = dict(zip([f"{k:.2f}" for k in ks], vals))
        is_leak = name.startswith("L")
        short_label = name.split(":")[0]
        ax[0].plot(ks * 100, vals, color=c, marker="o" if not is_leak else "x", markersize=3.5,
                   linewidth=2.0 if "B2:" in name else 1.4, linestyle="--" if is_leak else "-",
                   label=short_label, zorder=5 if "B2:" in name else 3)

    ax[0].axvline(TOP_K * 100, color="#495057", linestyle=":", linewidth=1.2, label=f"Рабочая квота 20% (N={int(len(y_te)*TOP_K):,})".replace(",", " "))
    ax[0].set_xlabel("Ёмкость операторов в 1-й волне (доля базы), %")
    ax[0].set_ylabel("Добавочная маржа к случайному обзвону, млн ₸")
    ax[0].set_title("(а) Добавочная ценность приоритизации")
    ax[0].legend(loc="upper left", frameon=True, fontsize=8)

    gain = {}
    for (name, p), c in zip(models.items(), model_colors):
        o = order(p)
        cum = np.cumsum(y_te[o]) / y_te.sum()
        xs = np.arange(1, len(o) + 1) / len(o)
        is_leak = name.startswith("L")
        short_label = name.split(":")[0]
        ax[1].plot(xs * 100, cum * 100, color=c, linewidth=2.0 if "B2:" in name else 1.3,
                   linestyle="--" if is_leak else "-", label=short_label, zorder=5 if "B2:" in name else 3)
        gain[name] = {str(k): float(cum[int(len(o) * k / 100) - 1]) for k in (10, 20, 30, 50)}

    ax[1].plot([0, 100], [0, 100], color="#ADB5BD", linestyle=":", linewidth=1, label="Случайный порядок")
    ax[1].plot([0, base_rate * 100, 100], [0, 100, 100], color=CORAL, linestyle="-.", linewidth=1, label="Теоретический оптимум")
    ax[1].axvline(TOP_K * 100, color="#495057", linestyle=":", linewidth=1.2)
    ax[1].set_xlabel("Охваченная доля базы лидов, %")
    ax[1].set_ylabel("Доля найденных покупателей, %")
    ax[1].set_title("(б) Кумулятивная кривая отклика (Gain Chart)")
    ax[1].legend(loc="lower right", frameon=True, fontsize=7.8)

    fig.tight_layout()
    fig.savefig(FIG / "10_baseline.png", bbox_inches="tight")
    plt.close(fig)

    extra = {
        "n_train": int(n_tr), "n_test": int(len(te)),
        "conv_train": float(y_tr.mean()), "conv_test": float(y_te.mean()),
        "train_period": [str(tr.period.min().date()), str(tr.period.max().date())],
        "test_period": [str(te.period.min().date()), str(te.period.max().date())],
        "auc_random_split": float(auc_rand), "auc_random_split_plus": float(auc_rand_m),
        "coef_top_pos": {k: float(v) for k, v in coef.tail(7)[::-1].items()},
        "coef_top_neg": {k: float(v) for k, v in coef.head(7).items()},
        "gain": gain, "incremental_margin": inc,
        "max_recall_at_k": float(min(1, TOP_K / y_te.mean()))
    }
    save_json(extra, "baseline_extra.json")
    print("Baseline redesign completed.")
    return extra

if __name__ == "__main__":
    main()
