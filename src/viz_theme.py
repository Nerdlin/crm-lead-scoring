"""Визуальный стиль графиков проекта:
- Современная корпоративная палитра: глубокий изумруд/teal (#005F73, #0A9396),
  мягкий кораллово-янтарный акцент (#EE9B00, #CA6702, #AE2012),
  нейтральный серо-стальной (#E9D8A6, #94D2BD, #6C757D).
- Шрифт: Arial / DejaVu Sans.
- Стилистика: тонкие вспомогательные оси, чистые подписи значений на столбцах,
  мягкие скругления, полупрозрачные области, горизонтальные разделители.
- Отсутствие шаблонного сине-оранжевого дефолтного вида matplotlib.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# Палитра "Modern Emerald & Amber"
TEAL_DARK = "#005F73"    # основной темный
TEAL_MID = "#0A9396"     # основной цветной
TEAL_LIGHT = "#94D2BD"   # светлый акцент
AMBER_LIGHT = "#E9D8A6"  # песочно-желтый
AMBER_WARM = "#EE9B00"   # теплый янтарь
CORAL = "#CA6702"        # оранжево-коралловый
RUST = "#BB3E03"         # насыщенный акцент/негатив
DARK_SLATE = "#1D2D44"   # глубокий графит для текста
GREY_BG = "#F8F9FA"      # фон подложек
GREY_BORDER = "#DEE2E6"  # рамки

def apply_theme():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": DARK_SLATE,
        "axes.labelsize": 8.5,
        "axes.labelcolor": DARK_SLATE,
        "axes.edgecolor": "#CED4DA",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E9ECEF",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "xtick.color": "#495057",
        "ytick.color": "#495057",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF"
    })
