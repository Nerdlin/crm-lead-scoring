"""Шаг 3. Разведочный анализ (EDA): уникальная стилистика графиков, альтернативные акценты."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import FIG, PROCESSED, RAW_CSV, TEST_SHARE, save_json
from viz_theme import (apply_theme, TEAL_DARK, TEAL_MID, TEAL_LIGHT, AMBER_WARM,
                       CORAL, RUST, DARK_SLATE, GREY_BORDER)

apply_theme()

JOB_RU = {
    "admin.": "Административный персонал",
    "blue-collar": "Рабочие специальности",
    "technician": "Технические специалисты",
    "services": "Сфера обслуживания",
    "management": "Руководители и топ-менеджмент",
    "retired": "Пенсионеры",
    "entrepreneur": "Предприниматели / бизнес",
    "self-employed": "Самозанятые / фриланс",
    "housemaid": "Домашний персонал",
    "unemployed": "Временно неработающие",
    "student": "Студенты вузов",
    "unknown": "Не указано в анкете"
}

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)

def main() -> dict:
    df = pd.read_parquet(PROCESSED / "contacts.parquet")
    raw = pd.read_csv(RAW_CSV, sep=";")
    f: dict = {}
    mean_conv = df.y.mean()
    f["conv"] = float(mean_conv)

    # -------------------------------------------------------------
    # 01. Целевая переменная (Донат-диаграмма + тренд по годам со стеками)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    
    # Левая часть: Donut chart вместо обычного bar chart
    cnt = df.y.value_counts().sort_index()
    labels = ["Не заключили договор", "Оплатили курс"]
    colors = ["#CED4DA", CORAL]
    wedges, texts, autotexts = ax[0].pie(
        cnt.values, labels=labels, autopct="%1.1f%%", startangle=140,
        colors=colors, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    plt.setp(autotexts, size=8.5, weight="bold", color=DARK_SLATE)
    plt.setp(texts, size=8, color=DARK_SLATE)
    ax[0].set_title("Соотношение классов (N = 41 176)", pad=10)

    # Правая часть: группированная динамика по годам с числом лидов
    by = df.groupby("year").y.agg(["mean", "size", "sum"])
    x_pos = np.arange(len(by))
    bars = ax[1].bar(x_pos, by["mean"] * 100, width=0.48, color=TEAL_DARK, edgecolor="none", zorder=3)
    ax[1].axhline(mean_conv * 100, color=CORAL, linestyle="--", linewidth=1.2, label=f"Средняя: {mean_conv:.1%}", zorder=4)
    
    for i, (m, s, sm) in enumerate(zip(by["mean"], by["size"], by["sum"])):
        ax[1].text(i, m * 100 + 1.5, f"{m:.1%}\n({int(sm):,} из {int(s):,})".replace(",", " "),
                   ha="center", va="bottom", fontsize=7.5, color=DARK_SLATE, weight="bold")
    ax[1].set_xticks(x_pos)
    ax[1].set_xticklabels([f"{yr} г." for yr in by.index])
    ax[1].set_ylim(0, by["mean"].max() * 125)
    ax[1].set_ylabel("Доля конверсии, %")
    ax[1].set_title("Конверсия по годам сдвига кампании")
    ax[1].legend(loc="upper left", frameon=True, fontsize=8)
    save(fig, "01_target.png")
    f["conv_by_year"] = {str(k): float(v) for k, v in by["mean"].items()}

    # -------------------------------------------------------------
    # 02. Скрытые пропуски (Горизонтальные lollipop-диаграммы)
    # -------------------------------------------------------------
    miss = {}
    for c in ["default", "education", "housing", "loan", "job", "marital"]:
        miss[f"Анкета: {c}"] = (raw[c] == "unknown").mean() * 100
    miss["Заглушка: pdays=999"] = (raw.pdays == 999).mean() * 100
    miss["Заглушка: poutcome='nonexistent'"] = (raw.poutcome == "nonexistent").mean() * 100
    s = pd.Series(miss).sort_values()

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    y_pos = np.arange(len(s))
    
    # Lollipop style
    for y_idx, (name, val) in enumerate(s.items()):
        col = RUST if "Заглушка" in name else TEAL_MID
        ax.hlines(y=y_idx, xmin=0, xmax=val, color=col, alpha=0.7, linewidth=2.2, zorder=3)
        ax.scatter(val, y_idx, color=col, s=65, zorder=4, edgecolor="white", linewidth=1)
        ax.text(val + 1.5, y_idx, f"{val:.1f}%", va="center", fontsize=8, weight="bold", color=DARK_SLATE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(s.index)
    ax.axvline(40, color="#ADB5BD", linestyle="--", linewidth=1, label="Порог фильтрации фичей (40%)", zorder=2)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Доля записей в исходной выгрузке raw, %")
    ax.set_title("Карта скрытых пропусков ('unknown') и технических кодов-заглушек")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    save(fig, "02_missing.png")
    f["unknown_default"] = float(miss["Анкета: default"])

    # -------------------------------------------------------------
    # 03. Канал коммуникации и род занятий (Двухпанельный анализ)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 2.3]})
    
    # Канал
    ch = df.groupby("contact").y.agg(["mean", "size"])
    ch_labels = ["Мобильный (GSM)", "Городской телефон"]
    ch_vals = [ch.loc["cellular", "mean"] * 100, ch.loc["telephone", "mean"] * 100]
    ax[0].bar(ch_labels, ch_vals, color=[TEAL_DARK, "#ADB5BD"], width=0.5, zorder=3)
    ax[0].axhline(mean_conv * 100, color=CORAL, linestyle=":", linewidth=1.2, label="Базовая")
    for idx, v in enumerate(ch_vals):
        n_cnt = ch.loc["cellular" if idx == 0 else "telephone", "size"]
        ax[0].text(idx, v + 0.6, f"{v:.1f}%\n(n={n_cnt:,})".replace(",", " "), ha="center", va="bottom", fontsize=8, weight="bold")
    ax[0].set_ylim(0, 22)
    ax[0].set_ylabel("Конверсия в оплату, %")
    ax[0].set_title("Канал связи")

    # Профессии
    jb = df.assign(job=df.job.fillna("unknown")).groupby("job").y.agg(["mean", "size"]).sort_values("mean")
    bar_cols = [AMBER_WARM if j in ("student", "retired") else TEAL_MID for j in jb.index]
    y_idx = np.arange(len(jb))
    ax[1].barh(y_idx, jb["mean"] * 100, color=bar_cols, height=0.62, zorder=3)
    ax[1].axvline(mean_conv * 100, color=CORAL, linestyle="--", linewidth=1.2, label=f"Средняя ({mean_conv:.1%})", zorder=4)
    
    for i, (m, n) in enumerate(zip(jb["mean"], jb["size"])):
        ax[1].text(m * 100 + 0.4, i, f"{m:.1%}  (n={n:,})".replace(",", " "), va="center", fontsize=7.8)
    
    ax[1].set_yticks(y_idx)
    ax[1].set_yticklabels([JOB_RU.get(j, j) for j in jb.index])
    ax[1].set_xlim(0, 39)
    ax[1].set_xlabel("Конверсия в оплату программы, %")
    ax[1].set_title("Конверсия по сферам занятости слушателей")
    ax[1].legend(loc="lower right", frameon=True, fontsize=8)
    save(fig, "03_channel_job.png")
    f["conv_cell"] = float(ch.loc["cellular", "mean"])
    f["conv_tel"] = float(ch.loc["telephone", "mean"])
    f["conv_job"] = {k: float(v) for k, v in jb["mean"].items()}

    # -------------------------------------------------------------
    # 04. История прошлых кампаний и закон затухания отдачи звонков
    # -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    
    # poutcome
    pout_order = ["nonexistent", "failure", "success"]
    pout_ru = ["Новый клиент\n(ранее не обращался)", "Прошлый контакт:\nотказ", "Прошлый контакт:\nуспешная оплата"]
    po = df.groupby("poutcome").y.agg(["mean", "size"]).loc[pout_order]
    bars_po = ax[0].bar(pout_ru, po["mean"] * 100, color=["#ADB5BD", TEAL_MID, CORAL], width=0.55, zorder=3)
    for i, (m, n) in enumerate(zip(po["mean"], po["size"])):
        ax[0].text(i, m * 100 + 1.8, f"{m:.1%}\n(N={n:,})".replace(",", " "), ha="center", va="bottom", fontsize=8, weight="bold")
    ax[0].set_ylim(0, 85)
    ax[0].set_ylabel("Конверсия в оплату, %")
    ax[0].set_title("Влияние предыдущего опыта клиента")

    # campaign: step line + fill
    cg = df.assign(cb=df.campaign.clip(upper=10)).groupby("cb").y.agg(["mean", "size"])
    xs = np.arange(1, len(cg) + 1)
    ax[1].plot(xs, cg["mean"] * 100, color=RUST, marker="o", markersize=5, linewidth=1.8, label="Конверсия звонка, %", zorder=4)
    ax[1].fill_between(xs, cg["mean"] * 100, color=RUST, alpha=0.1, zorder=2)
    
    ax_twin = ax[1].twinx()
    ax_twin.bar(xs, cg["size"], color=TEAL_LIGHT, alpha=0.45, width=0.5, label="Пул клиентов (объём)", zorder=1)
    ax_twin.set_ylabel("Количество клиентов в пуле", color=TEAL_DARK)
    ax_twin.grid(False)

    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([str(i) if i < 10 else "10+" for i in range(1, len(cg) + 1)])
    ax[1].set_xlabel("Порядковый номер контакта с клиентом в кампании")
    ax[1].set_ylabel("Конверсия, %", color=RUST)
    ax[1].set_title("Эффект затухания отдачи при назойливых звонках")
    
    # Combined legend
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=8)
    save(fig, "04_history_campaign.png")
    f["conv_pout"] = {k: float(v) for k, v in po["mean"].items()}
    f["share_campaign_gt3"] = float((df.campaign > 3).mean())

    # -------------------------------------------------------------
    # 05. Возрастной срез слушателей (KDE + половозрастные когорты)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    
    # KDE
    sns.kdeplot(data=df[df.y == 0], x="age", color="#ADB5BD", fill=True, alpha=0.3, label="Отказ / нет оплаты", ax=ax[0], linewidth=1.2)
    sns.kdeplot(data=df[df.y == 1], x="age", color=TEAL_DARK, fill=True, alpha=0.35, label="Оплата курса", ax=ax[0], linewidth=1.5)
    ax[0].set_xlabel("Возраст слушателя (полных лет)")
    ax[0].set_ylabel("Плотность распределения")
    ax[0].set_title("Сравнение плотности возраста по исходу")
    ax[0].legend(loc="upper right", frameon=True, fontsize=8)

    # Категории
    bins = [16, 25, 30, 40, 50, 60, 100]
    lab = ["17–25", "26–30", "31–40", "41–50", "51–60", "60+"]
    ag = df.groupby(pd.cut(df.age, bins, labels=lab), observed=True).y.agg(["mean", "size"])
    bar_cols = [CORAL if a in ("17–25", "60+") else TEAL_DARK for a in ag.index]
    ax[1].bar(ag.index.astype(str), ag["mean"] * 100, color=bar_cols, width=0.55, zorder=3)
    ax[1].axhline(mean_conv * 100, color="#6C757D", linestyle=":", linewidth=1, label="Средняя")
    for i, (m, n) in enumerate(zip(ag["mean"], ag["size"])):
        ax[1].text(i, m * 100 + 1.2, f"{m:.1%}\n({n:,})".replace(",", " "), ha="center", va="bottom", fontsize=7.8)
    ax[1].set_ylim(0, ag["mean"].max() * 125)
    ax[1].set_ylabel("Конверсия, %")
    ax[1].set_xlabel("Возрастная категория")
    ax[1].set_title("U-образная зависимость конверсии от возраста")
    save(fig, "05_age.png")
    f["conv_age"] = {str(k): float(v) for k, v in ag["mean"].items()}
    f["age_median_yes"] = float(df.loc[df.y == 1, "age"].median())
    f["age_median_no"] = float(df.loc[df.y == 0, "age"].median())

    # -------------------------------------------------------------
    # 06. Выбросы (Скрипичные диаграммы с точками перцентилей)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for a_idx, (col_name, title_txt) in enumerate([("campaign", "Интенсивность звонков (campaign)"), ("duration", "Длительность звонка (duration, с)")]):
        ax[a_idx].violinplot(df[col_name], orientation="horizontal", showmedians=True, showextrema=True)
        p99 = df[col_name].quantile(0.99)
        ax[a_idx].axvline(p99, color=CORAL, linestyle="--", linewidth=1.2)
        ax[a_idx].text(p99, 1.15, f" 99% перцентиль\n ({p99:.0f})", color=CORAL, fontsize=8, weight="bold")
        ax[a_idx].set_title(title_txt)
        ax[a_idx].set_yticks([])
    save(fig, "06_outliers.png")
    f["campaign_p99"] = float(df.campaign.quantile(0.99))
    f["campaign_clipped"] = int((df.campaign > f["campaign_p99"]).sum())

    # -------------------------------------------------------------
    # 07. Матрица корреляций (Компактный чистый тепловой стиль)
    # -------------------------------------------------------------
    corr_cols = ["age", "campaign", "previous", "duration", "emp_var_rate", "cons_price_idx",
                 "cons_conf_idx", "euribor3m", "nr_employed", "y"]
    corr_ru = ["Возраст", "Кампания", "Пред.контакт", "Длительность*", "Инд.занятости", "Инд.цен",
               "Инд.довер.", "Euribor 3M", "Число раб.", "Целевая (y)"]
    cr = df[corr_cols].corr("spearman")
    cr.columns = corr_ru
    cr.index = corr_ru

    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.triu(np.ones_like(cr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(cr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.75},
                annot=True, fmt=".2f", annot_kws={"size": 7.5}, ax=ax)
    ax.set_title("Матрица ранговых корреляций Спирмена (*duration — признак утечки)")
    save(fig, "07_corr.png")
    f["rho_y"] = {k: float(v) for k, v in df[corr_cols].corr("spearman")["y"].drop("y").items()}
    f["rho_euribor_nr"] = float(df[corr_cols].corr("spearman").loc["euribor3m", "nr_employed"])

    # -------------------------------------------------------------
    # 08. Временной дрейф макросреды и конверсии
    # -------------------------------------------------------------
    pm = df.groupby("period").agg(n=("y", "size"), conv=("y", "mean"), eur=("euribor3m", "mean"))
    cut_date = df.period.iloc[int(len(df) * (1 - TEST_SHARE))]

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    # Объем как область
    ax.fill_between(pm.index, pm.n, color=TEAL_LIGHT, alpha=0.35, label="Объём звонков в месяц (шкала log)")
    ax.set_yscale("log")
    ax.set_ylabel("Месячный поток заявок (log-шкала)", color=TEAL_DARK)

    ax2 = ax.twinx()
    ax2.plot(pm.index, pm.conv * 100, color=CORAL, marker="o", markersize=4, linewidth=1.8, label="Фактическая конверсия, %")
    ax2.plot(pm.index, pm.eur * 10, color=DARK_SLATE, linestyle=":", linewidth=1.4, label="Ставка Euribor 3M (×10)")
    ax2.set_ylabel("Конверсия в оплату, % / Euribor × 10", color=DARK_SLATE)
    ax2.grid(False)

    ax.axvline(cut_date, color="#212529", linestyle="--", linewidth=1.2)
    ax.text(cut_date, ax.get_ylim()[1] * 0.55, "  ← Train (80%) | Test (20%) →", color="#212529", fontsize=8.5, weight="bold")

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, fontsize=8)
    ax.set_title("Динамика воронки и макроэкономический сдвиг (май 2008 – ноябрь 2010)")
    save(fig, "08_drift.png")
    f["test_start"] = str(cut_date.date())
    f["conv_month_min"] = float(pm.conv.min())
    f["conv_month_max"] = float(pm.conv.max())
    f["corr_conv_eur_month"] = float(pm[["conv", "eur"]].corr("spearman").iloc[0, 1])

    # -------------------------------------------------------------
    # 09. Утечка данных через duration и календарная сезонность
    # -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    
    # duration утечка
    db = pd.cut(df.duration, [-1, 60, 120, 180, 300, 600, 5000],
                labels=["до 1 мин", "1–2 мин", "2–3 мин", "3–5 мин", "5–10 мин", "> 10 мин"])
    dg = df.groupby(db, observed=True).y.agg(["mean", "size"])
    ax[0].bar(dg.index.astype(str), dg["mean"] * 100, color=CORAL, width=0.55, zorder=3)
    for i, (m, n) in enumerate(zip(dg["mean"], dg["size"])):
        ax[0].text(i, m * 100 + 1.2, f"{m:.0%}", ha="center", va="bottom", fontsize=8, weight="bold")
    ax[0].set_ylim(0, 72)
    ax[0].set_ylabel("Конверсия в оплату, %")
    ax[0].set_title("(а) Природа Data Leakage: длительность беседы")

    # Месяцы
    mo = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    mo_ru = ["Март", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
    mg = df.groupby("month").y.agg(["mean", "size"]).loc[mo]
    bar_m_cols = [CORAL if m in ("mar", "sep", "oct", "dec") else TEAL_DARK for m in mo]
    ax[1].bar(mo_ru, mg["mean"] * 100, color=bar_m_cols, width=0.55, zorder=3)
    ax[1].set_ylabel("Конверсия, %")
    ax[1].set_title("(б) Сезонность конверсии по месяцам")
    save(fig, "09_leakage_month.png")
    f["conv_dur"] = {str(k): float(v) for k, v in dg["mean"].items()}
    f["conv_month"] = {str(k): float(v) for k, v in mg["mean"].items()}

    save_json(f, "eda_facts.json")
    print("EDA updated with new theme: 30 facts")
    return f

if __name__ == "__main__":
    main()
