"""
BloomCast — unified HAB risk monitoring dashboard.
Loads pre-computed demo data only. No live calls. Built for the 2018 SW Florida replay.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

DD = Path(__file__).resolve().parent / "demo_data"

st.set_page_config(page_title="BloomCast", page_icon="🌊", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- styling ----------
st.markdown("""
<style>
:root { --ink:#0b2a3a; --teal:#0a8a8a; }
.block-container { padding-top: 2rem; max-width: 1300px; }
h1,h2,h3 { color: var(--ink); }
.bc-title { font-size: 2.1rem; font-weight: 800; color:#0b2a3a; letter-spacing:-.5px; margin-bottom:0;}
.bc-sub { color:#5b7886; font-size:1.02rem; margin-top:.15rem; }
.metric-card { background:linear-gradient(135deg,#f3fafb,#e8f4f6); border:1px solid #d4e8ec;
  border-radius:14px; padding:14px 18px; }
.metric-card .v { font-size:1.6rem; font-weight:800; color:#0a8a8a; }
.metric-card .l { font-size:.8rem; color:#5b7886; text-transform:uppercase; letter-spacing:.5px;}
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600;}
small.cap { color:#7a93a0; }
</style>
""", unsafe_allow_html=True)

# ---------- data ----------
@st.cache_data
def load():
    daily = pd.read_csv(DD / "daily_risk_2018.csv", parse_dates=["date"])
    pts = pd.read_csv(DD / "map_points_2018.csv", parse_dates=["date"])
    meta = json.load(open(DD / "meta.json"))
    return daily, pts, meta

daily, pts, meta = load()

# ---------- header ----------
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown('<div class="bc-title">🌊 BloomCast</div>', unsafe_allow_html=True)
    st.markdown('<div class="bc-sub">One live, location-specific risk signal for harmful algal blooms — '
                'built from fragmented public ocean data.</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div style="display:flex; gap:10px; justify-content:flex-end;">
      <div class="metric-card"><div class="v">{meta['auc']:.2f}</div><div class="l">AUC · 2018 held out</div></div>
      <div class="metric-card"><div class="v">{meta['n_sources']}</div><div class="l">Data sources unified</div></div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ---------- sidebar: site + replay scrubber ----------
st.sidebar.header("Monitoring site")
st.sidebar.selectbox("Operator location", ["Sarasota Bay, FL (oyster lease)"], index=0)
st.sidebar.caption("Anchored on the West Florida Shelf — the 2018 red tide epicenter.")

st.sidebar.header("2018 replay")
dates = daily["date"].dt.date.tolist()
default = dates.index(pd.Timestamp("2018-08-01").date())
sel = st.sidebar.select_slider("Scrub through the bloom", options=dates, value=dates[default])
sel = pd.Timestamp(sel)
st.sidebar.caption("Drag to watch the unified risk picture track the real bloom — "
                   "on data the model never trained on.")

row = daily[daily["date"] == sel].iloc[0]
risk = float(row["risk"])

# ---------- top row: gauge + map ----------
left, right = st.columns([1, 1.6])

with left:
    st.subheader(sel.strftime("%B %d, 2018"))
    color = "#1a9850" if risk < 40 else ("#f0a800" if risk < 70 else "#d73027")
    band = "LOW" if risk < 40 else ("ELEVATED" if risk < 70 else "HIGH")
    g = go.Figure(go.Indicator(
        mode="gauge+number", value=risk,
        number={"suffix": "", "font": {"size": 44, "color": color}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color, "thickness": 0.3},
               "steps": [{"range": [0, 40], "color": "#e8f6e9"},
                         {"range": [40, 70], "color": "#fdf2da"},
                         {"range": [70, 100], "color": "#fde3e0"}],
               "threshold": {"line": {"color": color, "width": 4}, "value": risk}}))
    g.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=0))
    st.plotly_chart(g, use_container_width=True)
    st.markdown(f"<div style='text-align:center;margin-top:-12px'>"
                f"<span class='pill' style='background:{color}22;color:{color}'>RISK: {band}</span></div>",
                unsafe_allow_html=True)
    obs = row["cells_obs"]
    if pd.notna(obs):
        st.markdown(f"<small class='cap'>Latest lab sample near this date: "
                    f"<b>{int(obs):,} cells/L</b></small>", unsafe_allow_html=True)
    else:
        st.markdown("<small class='cap'>No lab sample this day — risk is the model's "
                    "gap-filling nowcast.</small>", unsafe_allow_html=True)

with right:
    st.subheader("Observed bloom — SW Florida")
    window = pts[(pts["date"] >= sel - pd.Timedelta(days=7)) & (pts["date"] <= sel)].copy()
    cmap = {"Background": [120,160,170], "Very low": [90,180,160], "Low": [240,200,60],
            "Medium": [240,140,40], "High (closure)": [200,40,40]}
    if len(window):
        window["color"] = window["category"].map(cmap)
        window["r"] = (np.log10(window["cells"].clip(lower=1) + 1) * 900).clip(800, 9000)
    else:
        window = pts.head(1).assign(color=[[120,160,170]], r=800)
    layer = pdk.Layer("ScatterplotLayer", data=window,
                      get_position=["lon", "lat"], get_fill_color="color",
                      get_radius="r", opacity=0.7, pickable=True)
    view = pdk.ViewState(latitude=26.8, longitude=-82.2, zoom=7.2)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                             map_style="light",
                             tooltip={"text": "{LOCATION}\n{cells} cells/L ({category})"}),
                    use_container_width=True)
    st.markdown("<small class='cap'>Samples from the prior 7 days · "
                "<span style='color:#1a9850'>●</span> low "
                "<span style='color:#f0a800'>●</span> medium "
                "<span style='color:#d73027'>●</span> high/closure</small>", unsafe_allow_html=True)

# ---------- risk timeline ----------
st.subheader("Risk signal vs. the real bloom — full 2018")
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily["date"], y=daily["risk"], name="BloomCast risk",
                         line=dict(color="#0a8a8a", width=2.5), fill="tozeroy",
                         fillcolor="rgba(10,138,138,.12)"))
obsd = daily[daily["cells_max"].notna()]
fig.add_trace(go.Scatter(x=obsd["date"], y=(np.log10(obsd["cells_max"]+1)/8*100),
                         name="Observed cells (log, scaled)", mode="markers",
                         marker=dict(color="#d73027", size=5, opacity=.55)))
fig.add_vline(x=sel, line=dict(color="#0b2a3a", width=2, dash="dash"))
fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=1.12), yaxis_title="Risk (0–100)",
                  plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

# ---------- driver explainability ----------
st.subheader("Why the risk is at this level")
drivers = {"Recent cell levels": row["c_cells_recent"],
           "Water-temp anomaly": row["c_wtmp_anom"],
           "Upwelling winds (7d)": row["c_upwell_7d"],
           "Upwelling winds (14d)": row["c_upwell_14d"],
           "Season": row["c_sin_doy"] + row["c_cos_doy"]}
dd = pd.DataFrame({"driver": list(drivers), "contribution": list(drivers.values())})
dd = dd.sort_values("contribution")
fig2 = go.Figure(go.Bar(x=dd["contribution"], y=dd["driver"], orientation="h",
                        marker_color=["#d73027" if v > 0 else "#4575b4" for v in dd["contribution"]]))
fig2.update_layout(height=240, margin=dict(l=10, r=10, t=6, b=10),
                   xaxis_title="← lowers risk    raises risk →", plot_bgcolor="white")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("What BloomCast is — and isn't"):
    st.markdown(f"""
**Is:** a unified, location-specific HAB **risk monitoring** engine. It fuses {meta['n_sources']}
fragmented public sources (FWC *Karenia brevis* cell counts + NDBC shelf buoy winds/temperature)
into one continuous daily risk signal, filling the gaps between sparse, lagging lab samples.

**Validated:** trained on {meta['n_train_days']} days across 7 bloom seasons, then tested on **2018
held entirely out of training** — AUC **{meta['auc']:.2f}** against the worst red tide on record
(peak {meta['peak_cells']:,} cells/L on {meta['peak_date']}).

**Isn't (yet):** a long-range forecaster that predicts blooms from clean water weeks out. That's the
frontier — and it requires the integrated, ground-truthed dataset BloomCast is built to accumulate.
That dataset is the moat.
""")
