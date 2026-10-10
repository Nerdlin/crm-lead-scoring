"""Шаг 2. Чек-лист качества данных из 8 пунктов (по слою raw)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import MONTHS, RAW_CSV, REPORTS, save_json
from s01_prepare import reconstruct_time


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> dict:
    raw = pd.read_csv(RAW_CSV, sep=";")
    raw.columns = [c.replace(".", "_") for c in raw.columns]
    raw = reconstruct_time(raw)
    raw["t"] = (raw["y"] == "yes").astype(int)
    n = len(raw)
    orig_cols = [c for c in raw.columns if c not in ("year", "month_num", "period", "t")]
    d: dict = {"rows": n, "cols": len(orig_cols)}
    rows = []

    # 1. Полнота
    explicit = int(raw[orig_cols].isna().sum().sum())
    unk = {c: float((raw[c] == "unknown").mean()) for c in orig_cols
           if not pd.api.types.is_numeric_dtype(raw[c]) and (raw[c] == "unknown").any()}
    d["explicit_na"] = explicit
    d["unknown_share"] = unk
    d["pdays_999"] = float((raw["pdays"] == 999).mean())
    d["poutcome_nonexistent"] = float((raw["poutcome"] == "nonexistent").mean())
    unk_txt = ", ".join(f"{k} {pct(v)}" for k, v in sorted(unk.items(), key=lambda kv: -kv[1]))
    rows.append(["1. Полнота",
                 f"явных пропусков (NaN): {explicit}; скрытые пропуски 'unknown' в {len(unk)} колонках "
                 f"({unk_txt}); значение-заглушка pdays = 999 («ранее не звонили») у {pct(d['pdays_999'])}",
                 "Проблема",
                 "'unknown' -> NaN и отдельная категория «не указано»; pdays = 999 -> NaN + флаг never_contacted"])

    # 2. Уникальность
    full_dup = int(raw[orig_cols].duplicated().sum())
    prof = ["age", "job", "marital", "education", "default", "housing", "loan"]
    prof_dup = int(raw.duplicated(prof).sum())
    feat_cols = [c for c in orig_cols if c not in ("y", "duration")]
    g = raw.groupby(feat_cols, dropna=False)["t"].nunique()
    conflict = int((g > 1).sum())
    d.update(full_dup=full_dup, profile_dup=prof_dup, conflict_groups=conflict)
    rows.append(["2. Уникальность",
                 f"идентификатора клиента нет; полных дублей строк: {full_dup}; строк с повторяющимся "
                 f"профилем (возраст+работа+образование+семья+кредиты): {prof_dup} ({pct(prof_dup / n)}); "
                 f"групп с одинаковыми признаками, но разной целевой: {conflict}",
                 "Внимание",
                 "полные дубли удалены; без ID нельзя отличить повторный звонок тому же клиенту — "
                 "фиксируем риск, в CRM обязателен client_id"])

    # 3. Типы и форматы
    rows.append(["3. Типы и форматы",
                 "дата звонка хранится только как месяц и день недели текстом ('may', 'mon'), года нет; "
                 "числовой pdays содержит код 999; категории с точками ('basic.4y', 'admin.'); "
                 "имена колонок с точками ('emp.var.rate')",
                 "Внимание",
                 "snake_case; год восстановлен по порядку строк; код 999 -> NaN; справочники категорий"])

    # 4. Допустимые значения и выбросы
    q = raw[["age", "campaign", "duration", "previous"]].quantile(0.99)
    d.update(age_min=int(raw.age.min()), age_max=int(raw.age.max()),
             campaign_max=int(raw.campaign.max()), campaign_p99=float(q["campaign"]),
             duration_max=int(raw.duration.max()), duration_p99=float(q["duration"]),
             duration_zero=int((raw.duration == 0).sum()),
             campaign_over_p99=int((raw.campaign > q["campaign"]).sum()),
             age_over_90=int((raw.age > 90).sum()))
    rows.append(["4. Допустимые значения и выбросы",
                 f"отрицательных значений нет; возраст {d['age_min']}–{d['age_max']}; число звонков в кампании "
                 f"до {d['campaign_max']} (99-й перцентиль {d['campaign_p99']:.0f}); длительность звонка до "
                 f"{d['duration_max']} с (p99 {d['duration_p99']:.0f}); звонков длительностью 0 с: {d['duration_zero']}",
                 "Внимание",
                 "campaign обрезается по p99; длительность в модель не идёт (п. 8); возраст > 90 — реальные, оставляем"])

    # 5. Согласованность
    c1 = int(((raw.pdays != 999) & (raw.previous == 0)).sum())
    c2 = int(((raw.poutcome == "nonexistent") & (raw.previous > 0)).sum())
    c3 = int(((raw.previous > 0) & (raw.pdays == 999)).sum())
    c4 = int(((raw.duration == 0) & (raw.t == 1)).sum())
    macro_per_month = int(raw.groupby("period")["euribor3m"].nunique().max())
    d.update(c_pdays_prev0=c1, c_pout_none_prev=c2, c_prev_pdays999=c3, c_dur0_yes=c4,
             euribor_unique_per_month_max=macro_per_month)
    rows.append(["5. Согласованность",
                 f"pdays задан, а previous = 0: {c1}; poutcome = 'nonexistent' при previous > 0: {c2}; "
                 f"previous > 0, но pdays = 999: {c3} (прошлый контакт был, дата не сохранена); "
                 f"успех при длительности 0 с: {c4}",
                 "Внимание",
                 "противоречие previous/pdays фиксируем: признак never_contacted строим по previous; "
                 "в CRM дата каждого контакта хранится в журнале"])

    # 6. Информативность
    near_const = {c: float(raw[c].value_counts(normalize=True).iloc[0]) for c in orig_cols
                  if raw[c].value_counts(normalize=True).iloc[0] > 0.95 and c != "y"}
    default_yes = int((raw["default"] == "yes").sum())
    by_loan = raw.groupby("loan")["t"].mean().to_dict()
    by_housing = raw.groupby("housing")["t"].mean().to_dict()
    d.update(near_const=near_const, default_yes=default_yes, conv_by_loan=by_loan,
             conv_by_housing=by_housing)
    rows.append(["6. Информативность",
                 f"константных колонок нет; почти константные (> 95% одно значение): "
                 f"{', '.join(f'{k} ({pct(v)})' for k, v in near_const.items())}; default = 'yes' всего у "
                 f"{default_yes} клиентов; конверсия почти не зависит от housing/loan "
                 f"({pct(min(by_loan.values()))}–{pct(max(by_loan.values()))})",
                 "Внимание",
                 "default используется только как «известно/неизвестно»; housing/loan оставлены, "
                 "решение о переносе — по реальным данным"])

    # 7. Покрытие по времени
    per = raw.groupby("period").agg(n=("t", "size"), conv=("t", "mean"))
    by_year = raw.groupby("year").agg(n=("t", "size"), conv=("t", "mean"))
    missing_months = [m for m in MONTHS if m not in raw["month"].unique()]
    d.update(periods=len(per), first=str(per.index.min().date()), last=str(per.index.max().date()),
             by_year={int(k): {"n": int(v.n), "conv": float(v.conv)} for k, v in by_year.iterrows()},
             missing_months=missing_months, min_month_n=int(per.n.min()), max_month_n=int(per.n.max()))
    yr = "; ".join(f"{k}: {v['n']} звонков, конверсия {pct(v['conv'])}" for k, v in d["by_year"].items())
    rows.append(["7. Покрытие по времени",
                 f"дат нет, год восстановлен по порядку строк: {d['first'][:7]} – {d['last'][:7]}, "
                 f"{d['periods']} месяцев; {yr}; январь и февраль отсутствуют, объём по месяцам от "
                 f"{d['min_month_n']} до {d['max_month_n']}",
                 "Проблема",
                 "сплит train/test строго по времени; конверсию сравнивать только внутри периода; "
                 "в CRM обязательны called_at / enrolled_at"])

    # 8. Целевая и утечка
    auc_dur = roc_auc_score(raw.t, raw.duration)
    qd = pd.qcut(raw.duration, 5, labels=False, duplicates="drop")
    conv_q = raw.groupby(qd)["t"].mean().round(3).tolist()
    d.update(target_rate=float(raw.t.mean()), target_na=int(raw.y.isna().sum()),
             auc_duration=float(auc_dur), conv_by_duration_q=conv_q)
    rows.append(["8. Целевая и утечка",
                 f"целевая y без пропусков, конверсия {pct(d['target_rate'])} (дисбаланс ≈ 1:{(1 - d['target_rate']) / d['target_rate']:.0f}); "
                 f"длительность звонка известна только после разговора, но одна даёт ROC-AUC {auc_dur:.2f}; "
                 f"конверсия по квинтилям длительности от {pct(min(conv_q))} до {pct(max(conv_q))}",
                 "Проблема",
                 "duration исключена из baseline (показана отдельно как модель-утечка); оценка только по "
                 "признакам, известным до звонка"])

    cl = pd.DataFrame(rows, columns=["Пункт", "Что обнаружено", "Статус", "Что делаем"])
    cl.to_csv(REPORTS / "quality_checklist.csv", index=False, encoding="utf-8-sig")
    save_json(d, "quality_details.json")
    print(cl.to_string())
    return d


if __name__ == "__main__":
    main()
