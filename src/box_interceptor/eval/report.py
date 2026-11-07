"""Сохранение таблиц с результатами: CSV (для кода) + Markdown (для людей и docs/)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from box_interceptor import paths

METRICS_DIR = paths.OUTPUTS / "metrics"


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v: object) -> str:
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def save_table(df: pd.DataFrame, name: str) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_DIR / f"{name}.csv", index=False)
    md = METRICS_DIR / f"{name}.md"
    md.write_text(to_markdown(df) + "\n")
    return md
