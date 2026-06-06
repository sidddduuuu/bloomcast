# 🌊 BloomCast

**One live, location-specific risk signal for harmful algal blooms (HABs) — built from fragmented public ocean data.**

Coastal operators (shellfish farms, finfish aquaculture, desalination intakes) currently track red-tide risk across five disconnected sources: NOAA respiratory bulletins, state agency PDFs, spreadsheets, their own water samples, and eyeballed satellite imagery. None of it is unified, location-specific, or current. BloomCast pulls these together into a single, continuous, site-specific risk score.

---

## What it does

- **Site-specific risk score (0–100)** for a coastal location, updated daily.
- **Unifies fragmented public data** — FWC *Karenia brevis* cell counts + NDBC shelf-buoy winds & water temperature — into one signal.
- **Fills the gaps** between sparse, lagging lab samples with a model-based nowcast.
- **Explainable** — shows which drivers are pushing risk up or down.
- **Map view** of observed bloom intensity across SW Florida.
- **2018 replay** — scrub through the worst red tide on record and watch the risk track it.

## Validation (the honest version)

The risk engine is a logistic-regression model trained on **7 bloom seasons (2015–2023, excluding 2018)** and tested on **2018 held entirely out of training**.

- **Held-out 2018 AUC: 0.88**, against a bloom that peaked at **90,000,000 cells/L on 2018-08-13**.
- It is a **risk-monitoring / nowcasting** tool, not a long-range forecaster. Predicting bloom onset from clean water weeks ahead is the open frontier — and it requires the integrated, ground-truthed dataset BloomCast is built to accumulate. **That dataset is the moat.**

## Data sources (public, no auth)

| Source | Data | Role |
|---|---|---|
| FWC Historic HAB Database (2015–2023) | *K. brevis* cell counts | ground truth |
| NDBC buoy 42013 (West Florida Shelf) | wind direction/speed, water temp | physical drivers |

## Run locally

```bash
make install     # install dependencies
make fetch       # download raw public data
make data        # build the fused daily dataset
make demo        # precompute dashboard data
make run         # launch the dashboard
```

The dashboard reads only the small precomputed files in `app/demo_data/`, so it runs instantly and makes **zero live API calls** during a demo.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. At share.streamlit.io, point to `app/streamlit_app.py`.
3. `app/demo_data/` is committed, so the deployed app is fully self-contained.

## Structure

```
bloomcast/
├── pipeline/build_dataset.py     # fuse FWC + buoy into daily series
├── model/risk_model.py           # train + held-out 2018 evaluation
├── model/prepare_demo_data.py    # precompute dashboard data
├── app/streamlit_app.py          # the dashboard
├── app/demo_data/                # precomputed (committed)
├── fetch_data.sh                 # re-download raw data
└── Makefile
```

## Roadmap

- Email/SMS alerts on risk thresholds
- Satellite chlorophyll-a integration (offshore initiation signal)
- More regions and bloom species beyond Florida *K. brevis*
- Operator data flywheel → the dataset that unlocks true forecasting
