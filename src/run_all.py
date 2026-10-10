"""Главный оркестратор воспроизведения всей аналитической части РК-1."""
from __future__ import annotations

import time
import s00_candidates
import s01_prepare
import s02_quality
import s03_eda
import s04_baseline
import s05_benchmark
import s06_diagrams
import s07_report

def main():
    print("=" * 60)
    print("ЗАПУСК АНАЛИТИЧЕСКОГО КОНВЕЙЕРА РК-1 (ABDB)")
    print("=" * 60)
    t0 = time.time()

    print("\n[Шаг 0/7] Аудит кандидатов и обоснование выбора...")
    s00_candidates.main()

    print("\n[Шаг 1/7] Загрузка данных, проверка SHA-256, слои raw/processed/features...")
    s01_prepare.main()

    print("\n[Шаг 2/7] Аудит качества данных по чек-листу (8 пунктов)...")
    s02_quality.main()

    print("\n[Шаг 3/7] Разведочный анализ (EDA) и генерация графиков...")
    s03_eda.main()

    print("\n[Шаг 4/7] Обучение базовой модели (Baseline) и расчёт потерь...")
    s04_baseline.main()

    print("\n[Шаг 5/7] Бенчмарк инструментов (CSV vs Parquet, pandas vs DuckDB)...")
    s05_benchmark.main()

    print("\n[Шаг 6/7] Генерация архитектурных и процессных схем (BPMN / Dataflow)...")
    s06_diagrams.main()

    print("\n[Шаг 7/7] Компиляция итогового академического отчёта в Word (.docx)...")
    doc_path = s07_report.build_report()

    print("\n" + "=" * 60)
    print(f"Конвейер успешно завершён за {time.time() - t0:.1f} с!")
    print(f"Итоговый документ готов: {doc_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
