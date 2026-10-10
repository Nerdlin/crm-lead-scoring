"""Шаг 5. Замеры: CSV vs Parquet, pandas vs DuckDB (x1 и x25)."""
from __future__ import annotations

import time
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import psutil

from common import PROCESSED, REPORTS, save_json

REPEATS = 5
SCALES = [1, 25]


def timeit(fn):
    ts = []
    for _ in range(REPEATS):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def agg_pandas(path):
    d = pd.read_csv(path)
    return d.groupby(["year", "month", "job"]).agg(n=("y", "size"), conv=("y", "mean")).reset_index()


def main() -> dict:
    base = pd.read_parquet(PROCESSED / "contacts.parquet").drop(columns=["period"])
    out = {"ram_gb": psutil.virtual_memory().total / 2**30}
    tmp = Path(tempfile.mkdtemp())
    for s in SCALES:
        d = pd.concat([base] * s, ignore_index=True)
        csv, pq = tmp / f"x{s}.csv", tmp / f"x{s}.parquet"
        d.to_csv(csv, index=False); d.to_parquet(pq, index=False)
        con = duckdb.connect()
        q = (f"SELECT year, month, job, count(*) n, avg(y) conv FROM read_parquet('{pq.as_posix()}') "
             "GROUP BY 1,2,3")
        out[f"x{s}"] = {
            "rows": len(d),
            "csv_mb": csv.stat().st_size / 2**20,
            "parquet_mb": pq.stat().st_size / 2**20,
            "read_csv_s": timeit(lambda: pd.read_csv(csv)),
            "read_parquet_s": timeit(lambda: pd.read_parquet(pq)),
            "agg_pandas_csv_s": timeit(lambda: agg_pandas(csv)),
            "agg_duckdb_parquet_s": timeit(lambda: con.execute(q).fetchdf()),
            "mem_mb": pd.read_csv(csv).memory_usage(deep=True).sum() / 2**20,
        }
        con.close()
        print(s, out[f"x{s}"])
    pd.DataFrame({k: v for k, v in out.items() if k.startswith("x")}).to_csv(
        REPORTS / "benchmark.csv", encoding="utf-8-sig")
    save_json(out, "benchmark.json")
    return out


if __name__ == "__main__":
    main()
