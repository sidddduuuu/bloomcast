"""
BloomCast risk model
--------------------
Predicts P(high Karenia brevis bloom in the next 1-7 days) for the Sarasota zone.

Honest validation design:
  - Train on all bloom seasons EXCEPT 2018.
  - Test on 2018 (held out) -- the model never sees the famous bloom in training.
  - Report AUC, and crucially the LEAD TIME at bloom onset.

Two models are compared to be transparent about where the signal comes from:
  A) PHYSICAL + SEASON  -> drivers only (temp anomaly, wind transport, seasonality).
                           Tests whether physics forecasts ONSET before cells spike.
  B) OPERATIONAL        -> adds recent observed cell momentum (what an operator
                           actually has). This is the deployed model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
df = pd.read_csv(OUT / "daily.csv", parse_dates=["date"])

# Seasonality (blooms are strongly seasonal: late summer / fall)
df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)

PHYS = ["wtmp_anom", "upwell_7d", "upwell_14d", "sin_doy", "cos_doy"]
OPER = PHYS + ["cells_recent"]

# Need valid drivers + a label
base = df.dropna(subset=["wtmp_anom", "upwell_7d", "upwell_14d"]).copy()
train = base[base.year != 2018]
test = base[base.year == 2018].sort_values("date")


def fit_eval(feats, name):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(train[feats], train["bloom_next7"])
    p = clf.predict_proba(test[feats])[:, 1]
    auc = roc_auc_score(test["bloom_next7"], p)
    # logistic coefficients (standardized) for explainability
    lr = clf.named_steps["logisticregression"]
    coefs = dict(zip(feats, lr.coef_[0].round(3)))
    return clf, p, auc, coefs


print("Train days:", len(train), "| Test (2018) days:", len(test))
print("2018 positive rate:", round(test["bloom_next7"].mean(), 3))
print()

for feats, name in [(PHYS, "A) PHYSICAL + SEASON"), (OPER, "B) OPERATIONAL")]:
    clf, p, auc, coefs = fit_eval(feats, name)
    print(f"--- {name} ---")
    print(f"  Held-out 2018 AUC: {auc:.3f}")
    print(f"  Coefficients (standardized): {coefs}")
    # save operational model's 2018 risk series for the demo
    if name.startswith("B"):
        out = test[["date", "cells_obs", "cells_max", "bloom_next7",
                    "wtmp_anom", "upwell_7d", "cells_recent"]].copy()
        out["risk"] = (p * 100).round(1)
        out.to_csv(OUT / "backtest_2018.csv", index=False)
    print()

# ---- LEAD TIME: when does risk cross threshold vs. first real high bloom? ----
bt = pd.read_csv(OUT / "backtest_2018.csv", parse_dates=["date"])
# First day the actual bloom hit "high" (>=1M cells/L observed)
first_bloom = bt[bt["cells_max"] >= 1_000_000]["date"].min()
# First day risk crosses 50 sustained (3+ of prior 5 days), before that bloom
bt["hi"] = bt["risk"] >= 50
bt["sustained"] = bt["hi"].rolling(5, min_periods=3).sum() >= 3
warn = bt[(bt["sustained"]) & (bt["date"] <= first_bloom)]["date"].min()
print("=== LEAD TIME (operational model, 2018 held out) ===")
print("First observed HIGH bloom (>=1M cells/L):", first_bloom.date() if pd.notna(first_bloom) else "n/a")
print("First sustained risk>=50 warning:        ", warn.date() if pd.notna(warn) else "n/a")
if pd.notna(warn) and pd.notna(first_bloom):
    print("LEAD TIME:", (first_bloom - warn).days, "days early")
