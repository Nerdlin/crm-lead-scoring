"""Шаг 6. Генерация процессных схем: BPMN-схема (to-be) и Dataflow-диаграмма.
Стилизованный дизайн: современная карточная инфографика с акцентными плашками.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import numpy as np

from common import FIG
from viz_theme import TEAL_DARK, TEAL_MID, TEAL_LIGHT, AMBER_WARM, CORAL, RUST, DARK_SLATE

def draw_bpmn():
    fig, ax = plt.subplots(figsize=(13.5, 6.4), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # Стилизованные дорожки с мягким контрастом
    ax.fill_between([0, 100], 33.5, 49.5, color="#F4F8F9", ec="#D0E1E5", lw=1.2)
    ax.fill_between([0, 100], 17.0, 33.5, color="#F8FAF9", ec="#D0E1E5", lw=1.2)
    ax.fill_between([0, 100], 0.8, 17.0, color="#FFFFFF", ec="#D0E1E5", lw=1.2)

    # Заголовки дорожек (с бейджами)
    def lane_badge(y_c, text, bg_col):
        p = FancyBboxPatch((0.8, y_c - 4.5), 7.2, 9, boxstyle="round,pad=0.2,rounding_size=0.6",
                           fc=bg_col, ec="none")
        ax.add_patch(p)
        ax.text(4.4, y_c, text, va="center", ha="center", weight="bold", color="white", fontsize=7.2)

    lane_badge(41.5, "ВХОДЯЩИЙ\nПОТОК ЛИДОВ\n(Сайт/1C)", TEAL_DARK)
    lane_badge(25.2, "AI-МОДУЛЬ\nСКОРИНГА\n(CRM Core)", TEAL_MID)
    lane_badge(8.9, "ОТДЕЛ\nПРОДАЖ\n(Операторы)", CORAL)

    def draw_card(x, y, w, h, title, text="", bg="#FFFFFF", border="#B0BEC5"):
        # Карточка с тенью
        p_shadow = FancyBboxPatch((x + 0.3, y - 0.3), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
                                 fc="#E0E0E0", ec="none", alpha=0.5)
        ax.add_patch(p_shadow)
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
                           fc=bg, ec=border, lw=1.3)
        ax.add_patch(p)
        full_text = f"{title}\n{text}" if text else title
        ax.text(x + w/2, y + h/2, full_text, ha="center", va="center", fontsize=7.5, color=DARK_SLATE)

    def arrow(x1, y1, x2, y2, label="", color="#455A64", style="-|>"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                            fc=color, ec=color, lw=1.3)
        ax.add_patch(a)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 1.1, label, ha="center", va="bottom", fontsize=6.8,
                    color=color, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CFD8DC", lw=0.6))

    # Старт процесса
    c_start = Circle((11.5, 41.5), 2.2, fc="#E8F5E9", ec="#2E7D32", lw=2)
    ax.add_patch(c_start)
    ax.text(11.5, 41.5, "Вход", ha="center", va="center", fontsize=7.5, weight="bold", color="#1B5E20")

    # Шаг 1: Заявка
    draw_card(16, 38.5, 12, 6, "1. Поступление заявки", "Форма / звонок / рекомендация", bg="#E0F2F1", border=TEAL_MID)
    arrow(13.7, 41.5, 16, 41.5)

    # Переход в скоринг
    arrow(22, 38.5, 22, 28.5)
    draw_card(16, 22.5, 12, 6, "2. Снимок t₀", "Фиксация атрибутов до звонка", bg="#E8F5E9", border="#43A047")

    arrow(28, 25.5, 31.5, 25.5)
    draw_card(31.5, 22.5, 13.5, 6, "3. ML-скоринг P", "Логистическая регрессия B2", bg="#E8F5E9", border="#43A047")

    # Шлюз решения
    gx, gy = 49, 22.5
    poly = Polygon([[gx+3, gy+3], [gx+6, gy+6], [gx+9, gy+3], [gx+6, gy]], closed=True,
                   fc="#FFF8E1", ec=AMBER_WARM, lw=1.6)
    ax.add_patch(poly)
    ax.text(gx+6, gy+3, "P ≥ θ ?", ha="center", va="center", weight="bold", fontsize=7.6, color=DARK_SLATE)
    arrow(45, 25.5, gx+3, 25.5)

    # Ветка: Горячая очередь
    arrow(gx+6, gy+6, 55, 38.5)
    ax.text(50.5, 32, "P ≥ 0.20\n(Топ-20%)", fontsize=6.8, color=RUST, weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#FFCDD2", lw=0.6))
    draw_card(47, 38.5, 16, 6, "4а. Горячая очередь", "SLA дозвона ≤ 15 мин", bg="#FFEBEE", border="#E53935")

    # Ветка: Стандартная очередь
    arrow(gx+9, gy+3, 60, gy+3)
    ax.text(56.5, 27.5, "Остальные", fontsize=6.8, color="#616161", weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CFD8DC", lw=0.6))
    draw_card(60, 22.5, 14, 6, "4б. Стандартная очередь", "SLA ≤ 24 ч + e-mail цепочка", bg="#ECEFF1", border="#78909C")

    # Спуск в отдел продаж
    arrow(55, 38.5, 55, 12)
    arrow(67, 22.5, 67, 12)

    draw_card(46, 6, 29, 6, "5. Консультация слушателя", "Звонок менеджера, фиксация возражений и статуса", bg="#FFF3E0", border=CORAL)

    # Исход сделки
    ox, oy = 78, 6
    poly2 = Polygon([[ox+2.5, oy+3], [ox+5.5, oy+6], [ox+8.5, oy+3], [ox+5.5, oy]], closed=True,
                    fc="#FFF8E1", ec=AMBER_WARM, lw=1.6)
    ax.add_patch(poly2)
    ax.text(ox+5.5, oy+3, "Оплата ?", ha="center", va="center", weight="bold", fontsize=7.2, color=DARK_SLATE)
    arrow(75, 9, ox+2.5, 9)

    # Успех
    arrow(ox+5.5, oy+6, 85, 38.5)
    ax.text(80.5, 23, "Да (факт 1С)", fontsize=6.8, color="#2E7D32", weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#C8E6C9", lw=0.6))
    draw_card(78, 38.5, 14, 6, "6а. Оплата курса (y=1)", "Проводка в учётной системе", bg="#E8F5E9", border="#2E7D32")

    # Отказ
    arrow(ox+8.5, 9, 89, 9)
    ax.text(87, 10.5, "Отказ", fontsize=6.5, color=RUST, weight="bold")
    draw_card(89, 6, 9.5, 6, "6б. Архив (y=0)", "Причина отказа", bg="#FFEBEE", border="#EF5350")

    # Обратная связь
    arrow(85, 38.5, 85, 28.5)
    draw_card(76, 22.5, 18, 6, "7. Аудит & Дообучение", "Логирование (t₀, y) для retraining", bg="#EDE7F6", border="#7E57C2")

    # Финиш
    c_end = Circle((96.5, 41.5), 2.2, fc="#FFEBEE", ec="#C62828", lw=2)
    ax.add_patch(c_end)
    ax.text(96.5, 41.5, "Финиш", ha="center", va="center", fontsize=7.2, weight="bold", color="#B71C1C")
    arrow(92, 41.5, 94.3, 41.5)

    fig.tight_layout()
    fig.savefig(FIG / "s1_bpmn.png", bbox_inches="tight")
    plt.close(fig)
    print("BPMN diagram redesigned.")

def draw_dataflow():
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def card(x, y, w, h, title, items, bg="#FFFFFF", border=TEAL_MID):
        p_shadow = FancyBboxPatch((x + 0.3, y - 0.3), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
                                 fc="#E0E0E0", ec="none", alpha=0.5)
        ax.add_patch(p_shadow)
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
                           fc=bg, ec=border, lw=1.3)
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 2.0, title, ha="center", va="center", fontsize=8.2, weight="bold", color=DARK_SLATE)
        txt = "\n".join([f"• {it}" for it in items])
        ax.text(x + w/2, y + (h - 2.5)/2, txt, ha="center", va="center", fontsize=7.2, color="#37474F")

    def arrow(x1, y1, x2, y2, color="#546E7A"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                            fc=color, ec=color, lw=1.2)
        ax.add_patch(a)

    # 4 колонки слоев
    ax.text(12, 47, "1. СЫРЫЕ ИСТОЧНИКИ", ha="center", va="center", weight="bold", color=TEAL_DARK, fontsize=8.8)
    ax.text(37, 47, "2. ХРАНИЛИЩЕ RAW & LAKE", ha="center", va="center", weight="bold", color="#00796B", fontsize=8.8)
    ax.text(62, 47, "3. ML ВИСТАВКА & МОДЕЛЬ", ha="center", va="center", weight="bold", color="#2E7D32", fontsize=8.8)
    ax.text(87, 47, "4. БИЗНЕС-ПРИЛОЖЕНИЯ", ha="center", va="center", weight="bold", color=CORAL, fontsize=8.8)

    ax.axvline(24.5, c="#CFD8DC", ls=":", lw=1)
    ax.axvline(49.5, c="#CFD8DC", ls=":", lw=1)
    ax.axvline(74.5, c="#CFD8DC", ls=":", lw=1)

    # Источники
    card(2, 35, 20.5, 9, "Формы веб-сайта", ["Лендинги программ (ACCA, CPA)", "UTM-метки и промокоды", "Контактные e-mail и телефоны"], bg="#E0F7FA", border=TEAL_MID)
    card(2, 23, 20.5, 9, "Телефония & Мессенджеры", ["Логи входящих звонков", "Канал: GSM / WhatsApp", "Таймстемпы первого обращения"], bg="#E0F7FA", border=TEAL_MID)
    card(2, 11, 20.5, 9, "Учётная система 1C / Банк", ["Номера договоров обучения", "Факт банковской оплаты (y=1)", "Выставленные счета"], bg="#E0F7FA", border=TEAL_MID)
    card(2, 0, 20.5, 8.5, "Внешний контекст", ["Календарь сессий экзаменов", "Сезонность набора групп"], bg="#E0F7FA", border=TEAL_MID)

    # Lake
    card(27.5, 27, 19.5, 14, "Слой RAW (Data Lake)", ["Неизменяемые CSV выгрузки", "Контроль целостности SHA-256", "Каталогизация без потери истории", "Скрытые пропуски 'unknown' -> NaN"], bg="#E0F2F1", border="#00897B")
    card(27.5, 6, 19.5, 14, "Слой PROCESSED (Parquet)", ["Нормализация номеров телефонов", "Единый client_id контрагента", "Удаление технических дубликатов", "Выравнивание таймстемпов t₀"], bg="#E0F2F1", border="#00897B")

    arrow(22.5, 39.5, 27.5, 35)
    arrow(22.5, 27.5, 27.5, 31)
    arrow(22.5, 15.5, 27.5, 13)
    arrow(22.5, 4.5, 27.5, 9)
    arrow(37.2, 27, 37.2, 20)

    # Features & ML
    card(52.5, 27, 19.5, 14, "Витрина FEATURES", ["Атрибуты строго до звонка", "One-Hot кодирование категорий", "Масштабирование StandardScaler", "Флаг never_contacted"], bg="#E8F5E9", border="#43A047")
    card(52.5, 6, 19.5, 14, "ML-Сервис Скоринга", ["Логистическая регрессия B2", "Расчёт скора P (оплата)", "Извлечение топ-весов факторов", "Мониторинг дрейфа (PSI)"], bg="#E8F5E9", border="#43A047")

    arrow(47, 34, 52.5, 34)
    arrow(47, 13, 52.5, 13)
    arrow(62.2, 27, 62.2, 20)

    # Приложения CRM
    card(77.5, 33, 20, 10.5, "Горячая очередь звонков", ["Лиды с высоким скором (топ-20%)", "Норматив SLA ≤ 15 минут", "Подсказка ключевых факторов скора"], bg="#FFEBEE", border="#E53935")
    card(77.5, 19, 20, 10.5, "Стандартная очередь", ["Лиды стандартного приоритета", "Регулярный контакт менеджера", "Автоматические email-напоминания"], bg="#FFF3E0", border=CORAL)
    card(77.5, 3, 20, 12, "Дашборд РОП и Аудита", ["Контроль соблюдения SLA менеджерами", "Мониторинг конверсии по источникам", "Регулярный аудит утечек данных", "Пул разметки для дообучения"], bg="#EDE7F6", border="#7E57C2")

    arrow(72, 14, 77.5, 38)
    arrow(72, 13, 77.5, 24)
    arrow(72, 10, 77.5, 9)

    fig.tight_layout()
    fig.savefig(FIG / "s2_dataflow.png", bbox_inches="tight")
    plt.close(fig)
    print("Dataflow diagram redesigned.")

def main():
    draw_bpmn()
    draw_dataflow()

if __name__ == "__main__":
    main()
