"""
BloomCast data pipeline
-----------------------
Fuses two real, public data sources into a single daily time series for the
Sarasota / Venice nearshore zone on the West Florida Shelf:

  1. FWC Historic HAB database -> Karenia brevis cell counts (cells/L)  [GROUND TRUTH]
  2. NDBC buoy 42013 (offshore shelf) -> winds + water temperature       [DRIVERS]

The key engineered driver is the UPWELLING-FAVORABLE WIND INDEX. On the West
Florida Shelf (coastline ~N-S, water to the west), northerly winds drive offshore
surface Ekman transport and shoreward BOTTOM transport, carrying subsurface
K. brevis cells onshore -- the physical mechanism behind nearshore bloom
intensification. Index = WSPD * cos(WDIR). Positive = upwelling-favorable.

Output: data/processed/daily.csv  (one row per day, 2015-2023)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "processed"
OUT.mkdir(exist_ok=True)

# Sarasota/Venice nearshore box (dense FWC sampling, heart of the 2018 bloom)
LAT_MIN, LAT_MAX = 27.0, 27.5
LON_MIN, LON_MAX = -82.75, -82.30

# ----------------------------------------------------------------------------
# 1. GROUND TRUTH: FWC Karenia brevis cell counts
# ----------------------------------------------------------------------------
def load_cell_counts():
    df = pd.read_csv(DATA / "hab_2015_2023.csv")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df["date"] = pd.to_datetime(df["SAMPLE_DATE"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df["cells"] = pd.to_numeric(df["COUNT_"], errors="coerce")
    df = df.dropna(subset=["date", "cells", "LATITUDE", "LONGITUDE"])
    # spatial filter to the zone
    z = df[
        df.LATITUDE.between(LAT_MIN, LAT_MAX) & df.LONGITUDE.between(LON_MIN, LON_MAX)
    ].copy()
    # daily aggregate: use the 90th-percentile sample as the day's bloom level
    # (robust to single outliers, still captures intensity)
    daily = (
        z.groupby("date")["cells"]
        .agg(cells_p90=lambda s: np.percentile(s, 90), cells_max="max", n_samples="count")
        .reset_index()
    )
    return daily


# ----------------------------------------------------------------------------
# 2. DRIVERS: NDBC buoy 42013 winds + water temp
# ----------------------------------------------------------------------------
NDBC_MISSING = {"WDIR": 999, "WSPD": 99.0, "WTMP": 999.0}

def load_buoy():
    frames = []
    for f in sorted(DATA.glob("42013_*.txt")):
        # whitespace-delimited; both header lines start with '#'
        d = pd.read_csv(f, sep=r"\s+", comment="#", header=None,
                        names=["YY","MM","DD","hh","mm","WDIR","WSPD","GST","WVHT",
                               "DPD","APD","MWD","PRES","ATMP","WTMP","DEWP","VIS","TIDE"])
        frames.append(d)
    b = pd.concat(frames, ignore_index=True)
    b["date"] = pd.to_datetime(dict(year=b.YY, month=b.MM, day=b.DD), errors="coerce")
    for col, miss in NDBC_MISSING.items():
        b[col] = pd.to_numeric(b[col], errors="coerce").replace(miss, np.nan)
    b = b.dropna(subset=["date"])

    # Upwelling-favorable wind index (per observation), then daily mean.
    # WDIR is the direction the wind comes FROM (deg true). Northerly (WDIR~0)
    # => cos ~ +1 => upwelling-favorable. Southerly (WDIR~180) => downwelling.
    b["upwell"] = b["WSPD"] * np.cos(np.radians(b["WDIR"]))

    daily = (
        b.groupby(b.date.dt.normalize())
        .agg(upwell=("upwell", "mean"),
             wspd=("WSPD", "mean"),
             wtmp=("WTMP", "mean"))
        .reset_index()
        .rename(columns={"date": "date"})
    )
    return daily


# ----------------------------------------------------------------------------
# 3. FUSE + ENGINEER FEATURES on a continuous daily calendar
# ----------------------------------------------------------------------------
def build():
    cells = load_cell_counts()
    buoy = load_buoy()

    cal = pd.DataFrame({"date": pd.date_range("2015-01-01", "2023-12-31", freq="D")})
    df = cal.merge(cells, on="date", how="left").merge(buoy, on="date", how="left")

    # Carry buoy gaps forward briefly (sensors drop out); short, defensible.
    df[["upwell", "wspd", "wtmp"]] = df[["upwell", "wspd", "wtmp"]].ffill(limit=5)

    # Water-temp anomaly vs. day-of-year climatology (warm anomalies favor blooms)
    df["doy"] = df.date.dt.dayofyear
    clim = df.groupby("doy")["wtmp"].transform("mean")
    df["wtmp_anom"] = df["wtmp"] - clim

    # Rolling upwelling exposure (cumulative onshore forcing over recent weeks)
    df["upwell_7d"] = df["upwell"].rolling(7, min_periods=3).mean()
    df["upwell_14d"] = df["upwell"].rolling(14, min_periods=5).mean()

    # Recent observed bloom momentum (log scale; sampling is intermittent)
    df["cells_obs"] = df["cells_p90"].copy()
    df["log_cells"] = np.log10(df["cells_p90"].fillna(0) + 1)
    df["cells_recent"] = df["log_cells"].rolling(14, min_periods=1).max()

    # ---- TARGET: does a HIGH bloom occur in the NEXT 1-7 days? ----
    # FWC "high" category = >1,000,000 cells/L. We predict that threshold.
    BLOOM = 1_000_000
    fut = df["cells_max"].fillna(0).iloc[::-1].rolling(7, min_periods=1).max().iloc[::-1]
    df["future7_max"] = fut.shift(-1)  # strictly future (next 1-7 days)
    df["bloom_next7"] = (df["future7_max"] >= BLOOM).astype(int)

    df["year"] = df.date.dt.year
    df.to_csv(OUT / "daily.csv", index=False)
    return df


if __name__ == "__main__":
    df = build()
    print("Rows:", len(df), "| date range:", df.date.min().date(), "->", df.date.max().date())
    print("Days with cell-count obs:", df["cells_obs"].notna().sum())
    print("Days with buoy data:", df["upwell"].notna().sum())
    print("Positive labels (bloom_next7=1):", int(df["bloom_next7"].sum()))
    print("\n=== 2018 monthly: mean upwelling index, mean wtmp_anom, days w/ high bloom ===")
    d18 = df[df.year == 2018].copy()
    d18["ym"] = d18.date.dt.to_period("M")
    summary = d18.groupby("ym").agg(
        upwell_mean=("upwell", "mean"),
        wtmp_anom=("wtmp_anom", "mean"),
        bloom_days=("bloom_next7", "sum"),
        max_cells=("cells_max", "max"),
    )
    print(summary.to_string())
