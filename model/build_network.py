"""
BloomCast network engine — scores 6 SW Florida operator sites across 2015-2023.
Pools all sites to train one logistic-regression risk model, with 2018 held out
for the honest headline metric. Exports a compact JSON for the web dashboard.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SITES = [
    ("tampa",     "Tampa Bay",                27.75, -82.60, (27.55,27.95,-82.75,-82.45)),
    ("pinellas",  "Pinellas / Clearwater",    27.97, -82.83, (27.75,28.20,-82.90,-82.70)),
    ("sarasota",  "Sarasota Bay",             27.33, -82.58, (27.15,27.45,-82.70,-82.40)),
    ("charlotte", "Charlotte Harbor / Venice",26.95, -82.25, (26.75,27.15,-82.45,-82.00)),
    ("ftmyers",   "Fort Myers / Sanibel",     26.48, -82.05, (26.30,26.65,-82.25,-81.85)),
    ("naples",    "Naples / Collier",         26.08, -81.78, (25.90,26.25,-81.95,-81.60)),
]
HIGH = 1_000_000

# ---- buoy drivers (shared regional forcing) ----
def load_buoy():
    miss={"WDIR":999,"WSPD":99.0,"WTMP":999.0}; frames=[]
    for f in sorted(DATA.glob("42013_*.txt")):
        d=pd.read_csv(f,sep=r"\s+",comment="#",header=None,
            names=["YY","MM","DD","hh","mm","WDIR","WSPD","GST","WVHT","DPD","APD",
                   "MWD","PRES","ATMP","WTMP","DEWP","VIS","TIDE"])
        frames.append(d)
    b=pd.concat(frames,ignore_index=True)
    b["date"]=pd.to_datetime(dict(year=b.YY,month=b.MM,day=b.DD),errors="coerce")
    for c,m in miss.items(): b[c]=pd.to_numeric(b[c],errors="coerce").replace(m,np.nan)
    b=b.dropna(subset=["date"])
    b["upwell"]=b.WSPD*np.cos(np.radians(b.WDIR))
    g=b.groupby(b.date.dt.normalize()).agg(upwell=("upwell","mean"),wtmp=("WTMP","mean")).reset_index()
    return g.rename(columns={"date":"date"})

# ---- FWC cell counts ----
raw=pd.read_csv(DATA/"hab_2015_2023.csv"); raw.columns=[c.strip().lstrip("\ufeff") for c in raw.columns]
raw["date"]=pd.to_datetime(raw["SAMPLE_DATE"],errors="coerce").dt.tz_localize(None).dt.normalize()
raw["cells"]=pd.to_numeric(raw["COUNT_"],errors="coerce")
raw=raw.dropna(subset=["cells","LATITUDE","LONGITUDE","date"])

buoy=load_buoy()
cal=pd.DataFrame({"date":pd.date_range("2015-01-01","2023-12-31",freq="D")})

def site_frame(box):
    la0,la1,lo0,lo1=box
    z=raw[raw.LATITUDE.between(la0,la1)&raw.LONGITUDE.between(lo0,lo1)]
    daily=z.groupby("date")["cells"].agg(
        cells_p90=lambda s:np.percentile(s,90),cells_max="max",n="count").reset_index()
    df=cal.merge(daily,on="date",how="left").merge(buoy,on="date",how="left")
    df[["upwell","wtmp"]]=df[["upwell","wtmp"]].ffill(limit=5)
    df["doy"]=df.date.dt.dayofyear
    df["wtmp_anom"]=df["wtmp"]-df.groupby("doy")["wtmp"].transform("mean")
    df["upwell_7d"]=df.upwell.rolling(7,min_periods=3).mean()
    df["upwell_14d"]=df.upwell.rolling(14,min_periods=5).mean()
    df["log_cells"]=np.log10(df.cells_p90.fillna(0)+1)
    df["cells_recent"]=df.log_cells.rolling(14,min_periods=1).max()
    df["sin_doy"]=np.sin(2*np.pi*df.doy/365.25); df["cos_doy"]=np.cos(2*np.pi*df.doy/365.25)
    fut=df.cells_max.fillna(0).iloc[::-1].rolling(7,min_periods=1).max().iloc[::-1].shift(-1)
    df["bloom_next7"]=(fut>=HIGH).astype(int)
    df["year"]=df.date.dt.year
    return df

FEATS=["wtmp_anom","upwell_7d","upwell_14d","sin_doy","cos_doy","cells_recent"]
frames={sid:site_frame(box) for sid,_,_,_,box in SITES}

# pooled training, 2018 held out
alld=pd.concat([f.assign(site=sid) for sid,f in frames.items()],ignore_index=True)
alld=alld.dropna(subset=["wtmp_anom","upwell_7d","upwell_14d"])
tr=alld[alld.year!=2018]
clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
clf.fit(tr[FEATS],tr.bloom_next7)
te=alld[alld.year==2018]
auc=roc_auc_score(te.bloom_next7,clf.predict_proba(te[FEATS])[:,1])
print(f"Pooled held-out 2018 AUC across {len(SITES)} sites: {auc:.3f}")
lr=clf.named_steps["logisticregression"]; sc=clf.named_steps["standardscaler"]

# ---- score every site/day + export ----
DR_KEYS=["Recent cell levels","Water-temp anomaly","Upwelling winds (7d)","Upwelling winds (14d)","Season"]
def driver_vec(Xs_row):
    c=Xs_row*lr.coef_[0]
    return [round(float(c[5]),2),round(float(c[0]),2),round(float(c[1]),2),
            round(float(c[2]),2),round(float(c[3]+c[4]),2)]

series={}; samples={}
for sid,_,_,_,box in SITES:
    f=frames[sid].copy()
    valid=f.dropna(subset=["wtmp_anom","upwell_7d","upwell_14d"]).index
    f["risk"]=np.nan
    Xs=sc.transform(f.loc[valid,FEATS])
    f.loc[valid,"risk"]=(clf.predict_proba(f.loc[valid,FEATS])[:,1]*100).round(1)
    series[sid]={}
    for yr in range(2015,2024):
        fy=f[f.year==yr]
        Xsy=sc.transform(fy[FEATS].fillna(0))
        recs=[]
        for i,(_,r) in enumerate(fy.iterrows()):
            recs.append({
                "doy":int(r.doy),"d":r.date.strftime("%m-%d"),
                "risk":None if pd.isna(r.risk) else float(r.risk),
                "obs":None if pd.isna(r.cells_p90) else int(r.cells_p90),
                "max":None if pd.isna(r.cells_max) else int(r.cells_max),
                "wtmp":None if pd.isna(r.wtmp) else round(float(r.wtmp),1),
                "dr":driver_vec(Xsy[i]) if not pd.isna(r.risk) else None,
            })
        series[sid][yr]=recs

# map points + recent-sample tables, per year, per site
points={}; 
for yr in range(2015,2024):
    pts=[]
    for si,(sid,_,_,_,box) in enumerate(SITES):
        la0,la1,lo0,lo1=box
        z=raw[(raw.date.dt.year==yr)&raw.LATITUDE.between(la0,la1)&raw.LONGITUDE.between(lo0,lo1)]
        for t,la,lo,c in zip(z.date,z.LATITUDE,z.LONGITUDE,z.cells):
            pts.append([si,int(t.dayofyear),round(float(la),4),round(float(lo),4),int(c)])
    points[yr]=pts

payload={
  "sites":[{"id":sid,"name":nm,"lat":la,"lon":lo} for sid,nm,la,lo,_ in SITES],
  "years":list(range(2015,2024)),
  "driverKeys":DR_KEYS,
  "series":series,"points":points,
  "meta":{"auc":round(float(auc),3),"n_sites":len(SITES),"n_sources":2,
          "n_samples":int(len(raw)),"n_train_days":int(len(tr))}
}
out=ROOT/"app"/"bloomcast_network.json"
out.write_text(json.dumps(payload,separators=(",",":")))
import os
print("wrote",out.name,":",round(os.path.getsize(out)/1024/1024,2),"MB")
print("sites:",len(SITES),"| years:",9,"| total map points:",sum(len(v) for v in points.values()))
