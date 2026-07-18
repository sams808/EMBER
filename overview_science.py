"""
overview_science.py — dataset-level audit queries (framework-agnostic).

Ported from the old app's HanfordDataModel.overview/units_audit/
missing_audit/phase_audit/type_audit/farm_audit/top_analytes/top_elements,
restructured as plain functions over a HanfordDataset instead of methods on
a god-class.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import polars as pl

from data_model import HanfordDataset


def overview(dataset: HanfordDataset) -> pd.DataFrame:
    df = dataset.require_df()
    out = {
        "rows": df.height,
        "columns": df.width,
        "n_tanks": df.get_column("WasteSiteId").n_unique(),
        "n_analytes": df.get_column("Analyte").n_unique(),
        "n_primary_elements": df.filter(pl.col("Element").is_not_null()).get_column("Element").n_unique(),
        "n_waste_phases": df.get_column("WastePhase").n_unique(),
        "n_waste_types": df.get_column("WasteType").n_unique(),
        "n_units": df.get_column("Units").n_unique(),
    }
    report = dataset.report
    if report is not None:
        out.update({
            "source_file": str(report.source_path),
            "cache_used": report.cache_used,
            "load_seconds": round(report.load_seconds, 3),
            "estimated_size_mb": round(report.estimated_size_mb or 0, 3),
            "tank_attributes_file": str(report.attributes_path) if report.attributes_path else "",
            "tank_attributes_rows": report.attributes_rows,
        })
    return pd.DataFrame([out])


def units_audit(dataset: HanfordDataset) -> pd.DataFrame:
    df = dataset.require_df()
    return (
        df.group_by("Units")
        .agg([
            pl.len().alias("N_rows"),
            pl.col("Analyte").n_unique().alias("N_analytes"),
            pl.col("Element").n_unique().alias("N_elements"),
            pl.col("WasteSiteId").n_unique().alias("N_tanks"),
            pl.col("Inventory").sum().alias("TotalInventory"),
            pl.col("Inventory").min().alias("MinInventory"),
            pl.col("Inventory").max().alias("MaxInventory"),
        ])
        .sort("TotalInventory", descending=True)
        .to_pandas()
    )


def missing_audit(dataset: HanfordDataset) -> pd.DataFrame:
    df = dataset.require_df()
    rows = df.height
    records = []
    for col in df.columns:
        n_null = df.select(pl.col(col).is_null().sum()).item()
        records.append({
            "column": col,
            "n_null": int(n_null),
            "pct_null": 100.0 * float(n_null) / max(rows, 1),
            "dtype": str(df.schema[col]),
        })
    return pd.DataFrame(records).sort_values(["pct_null", "column"], ascending=[False, True])


def category_audit(dataset: HanfordDataset, category_col: str) -> pd.DataFrame:
    df = dataset.require_df()
    return (
        df.group_by(category_col)
        .agg([
            pl.len().alias("N_rows"),
            pl.col("WasteSiteId").n_unique().alias("N_tanks"),
            pl.col("Analyte").n_unique().alias("N_analytes"),
            pl.col("Element").n_unique().alias("N_elements"),
            pl.col("Inventory").sum().alias("TotalInventory"),
        ])
        .sort("TotalInventory", descending=True)
        .to_pandas()
    )


def phase_audit(dataset: HanfordDataset) -> pd.DataFrame:
    return category_audit(dataset, "WastePhase")


def type_audit(dataset: HanfordDataset) -> pd.DataFrame:
    return category_audit(dataset, "WasteType")


def farm_audit(dataset: HanfordDataset) -> pd.DataFrame:
    return category_audit(dataset, "TankFarm")


def top_analytes(dataset: HanfordDataset, unit: Optional[str] = None, top_n: int = 50) -> pd.DataFrame:
    df = dataset.require_df()
    if unit and unit != "All":
        df = df.filter(pl.col("Units") == unit)
    return (
        df.group_by(["Units", "Analyte", "Element", "ElementList", "AnalyteClass"])
        .agg([
            pl.col("Inventory").sum().alias("TotalInventory"),
            pl.col("WasteSiteId").n_unique().alias("N_tanks"),
            pl.len().alias("N_rows"),
        ])
        .sort("TotalInventory", descending=True)
        .head(top_n)
        .to_pandas()
    )


def top_elements(dataset: HanfordDataset, unit: Optional[str] = None, top_n: int = 50) -> pd.DataFrame:
    df = dataset.require_df().filter(pl.col("Element").is_not_null())
    if unit and unit != "All":
        df = df.filter(pl.col("Units") == unit)
    return (
        df.group_by(["Units", "Element"])
        .agg([
            pl.col("Inventory").sum().alias("TotalInventory"),
            pl.col("WasteSiteId").n_unique().alias("N_tanks"),
            pl.col("Analyte").n_unique().alias("N_analytes"),
            pl.len().alias("N_rows"),
        ])
        .sort("TotalInventory", descending=True)
        .head(top_n)
        .to_pandas()
    )
