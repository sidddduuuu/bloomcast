"""
Prepare pre-computed demo data for the BloomCast dashboard.
Runs the model ONCE, scores all of 2018, and writes small CSVs the app reads.
The app makes ZERO live calls -> instant + unbreakable during a demo.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "processed"
APP = Path(__file__).resolve().parent.parent / "app" / "demo_data"
APP.mkdir(parents=True, exist_ok=True)

# ---- 1. Daily risk series for 2018 (engine output) ----
df = pd.read_csv(OUT / "daily.csv", parse_dates=["date"])
df["sin_doy"] = np.sin(2*np.pi*df.doy/365.25)
df["cos_doy"] = np.cos(2*np.pi*df.doy/365.25)
FEATS = ["wtmp_anom","upwell_7d","upwell_14d","sin_doy","cos_doy","cells_recent"]
d = df.dropna(subset=["wtmp_anom","upwell_7d","upwell_14d"]).copy()
tr, te = d[d.year != 2018], d[d.year == 2018].sort_values("date")

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
clf.fit(tr[FEATS], tr["bloom_next7"])
te = te.copy()
te["risk"] = (clf.predict_proba(te[FEATS])[:,1] * 100).round(1)
auc = roc_auc_score(te["bloom_next7"], te["risk"])
print(f"Held-out 2018 nowcast AUC: {auc:.3f}")

# standardized driver contributions per day (for the explainability panel)
lr = clf.named_steps["logisticregression"]; sc = clf.named_steps["standardscaler"]
Xs = sc.transform(te[FEATS])
contrib = Xs * lr.coef_[0]                      # signed contribution per feature/day
nice = {"wtmp_anom":"Water-temp anomaly","upwell_7d":"Upwelling winds (7d)",
        "upwell_14d":"Upwelling winds (14d)","sin_doy":"Season","cos_doy":"Season ",
        "cells_recent":"Recent cell levels"}
for i,f in enumerate(FEATS):
    te[f"c_{f}"] = contrib[:,i].round(3)

cols = ["date","risk","cells_obs","cells_max","wtmp_anom","upwell_7d","wtmp",
        "cells_recent","bloom_next7"] + [f"c_{f}" for f in FEATS]
te[cols].to_csv(APP / "daily_risk_2018.csv", index=False)
print("wrote daily_risk_2018.csv:", len(te), "days")

# ---- 2. Map points: observed cell-count samples across SW Florida, 2018 ----
raw = pd.read_csv(DATA / "hab_2015_2023.csv")
raw.columns = [c.strip().lstrip("\ufeff") for c in raw.columns]
raw["date"] = pd.to_datetime(raw["SAMPLE_DATE"], errors="coerce").dt.tz_localize(None).dt.normalize()
raw["cells"] = pd.to_numeric(raw["COUNT_"], errors="coerce")
m = raw[(raw.date>="2018-01-01") & (raw.date<="2018-12-31")
        & raw.LATITUDE.between(25.8,27.7) & raw.LONGITUDE.between(-82.95,-81.6)].copy()
m = m.dropna(subset=["cells","LATITUDE","LONGITUDE"])

def cat(c):
    if c < 1_000: return "Background"
    if c < 10_000: return "Very low"
    if c < 100_000: return "Low"
    if c < 1_000_000: return "Medium"
    return "High (closure)"
m["category"] = m["cells"].apply(cat)
m[["date","LATITUDE","LONGITUDE","LOCATION","cells","category"]].rename(
    columns={"LATITUDE":"lat","LONGITUDE":"lon"}).to_csv(APP / "map_points_2018.csv", index=False)
print("wrote map_points_2018.csv:", len(m), "samples")

# ---- 3. Headline metrics for the app ----
import json
peak = te.loc[te["cells_max"].idxmax()]
json.dump({"auc": round(float(auc),3),
           "peak_date": str(peak["date"].date()),
           "peak_cells": int(peak["cells_max"]),
           "n_train_days": int(len(tr)),
           "n_sources": 2},
          open(APP / "meta.json","w"), indent=2)
print("wrote meta.json")
