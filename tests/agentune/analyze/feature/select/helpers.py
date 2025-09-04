from pathlib import Path

import polars as pl
import pytest


def load_and_clean_csv(file_path: str, dataset_name: str, *, cast_text_to_float: bool = False) -> pl.DataFrame:
    """Load and clean CSV data for tests.

    - Skips the test if the file does not exist.
    - Reads the CSV with Polars defaults.
    - For string columns, strips both double and single quotes.
    - If cast_text_to_float is True, attempts to cast string columns to Float64 (non-strict).
    """
    if not Path(file_path).exists():
        pytest.skip(f'{dataset_name} dataset not found at {file_path}')

    df = pl.read_csv(file_path)

    # Clean string columns by removing surrounding quotes
    text_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    if text_cols:
        df = df.with_columns([
            pl.col(col)
            .str.strip_chars('"')
            .str.strip_chars("'")
            .alias(col)
            for col in text_cols
        ])

        if cast_text_to_float:
            df = df.with_columns([
                pl.col(col).cast(pl.Float64, strict=False).alias(col)
                for col in text_cols
            ])

    return df
