"""FreightTiger AI Analyst - Streamlit Chat Interface (Phase 7).

Natural-language Q&A over the freight-cost anomaly detection pipeline.
All numeric claims are grounded in pipeline DataFrames - zero hallucination.
"""

import sys
import re
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -- Project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SHIPMENT_RECORDS_FILE,
    CONTEXT_NOTES_FILE,
    FINAL_OUTPUT_FILE,
    DEFAULT_HISTORY_THRESHOLD,
    DEFAULT_PEER_THRESHOLD,
)
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations
from src.pipeline.anomaly_detector import detect_candidate_anomalies

st.set_page_config(page_title='FreightTiger AI Analyst', page_icon='🐯', layout='wide', initial_sidebar_state='expanded')

# ===== CSS =====
CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [data-testid='stAppViewContainer'] { font-family: 'Inter', sans-serif; background: #080c14; color: #e2e8f0; }
[data-testid='stSidebar'] { background: #0a0f1c; border-right: 1px solid #1e293b; }
.hero { background: linear-gradient(135deg,#0d1b2a 0%,#1a2744 45%,#1e3a5f 100%); border: 1px solid #1e3a5f; border-radius: 20px; padding: 2.2rem 2.8rem; margin-bottom: 1.6rem; position: relative; overflow: hidden; }
.hero::after { content: ''; position: absolute; top: -60px; right: -60px; width: 300px; height: 300px; background: radial-gradient(circle, rgba(56,189,248,0.10) 0%, transparent 70%); pointer-events: none; }
.hero h1 { font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.3rem 0; }
.hero p { color: #94a3b8; font-size: 0.96rem; margin: 0; }
.note-card { background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: .85rem 1rem; margin-bottom: .7rem; font-size: .82rem; }
.note-nid { font-family: 'JetBrains Mono', monospace; color: #38bdf8; font-weight: 700; font-size: .78rem; }
.note-meta { color: #64748b; font-size: .71rem; margin: .2rem 0 .4rem 0; }
.note-body { color: #94a3b8; line-height: 1.45; }
.badge-j { display:inline-block; background:rgba(52,211,153,.12); color:#34d399; border:1px solid #34d39940; border-radius:6px; font-size:.68rem; padding:.1rem .5rem; margin-top:.4rem; }
.stChatInput > div { background:#111827 !important; border:1px solid #1e293b !important; border-radius:12px !important; }
.stChatInput input { color:#e2e8f0 !important; background:transparent !important; }
div[data-testid='stMetric'] { background:#111827; border:1px solid #1e293b; border-radius:12px; padding:.9rem 1.1rem; }
div[data-testid='stMetric'] label { color:#64748b !important; font-size:.72rem !important; }
div[data-testid='stMetricValue'] { color:#f1f5f9 !important; }
.stButton > button { background:linear-gradient(135deg,#1d4ed8,#2563eb) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-size:.8rem !important; font-weight:500 !important; transition:all .2s !important; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 4px 14px rgba(37,99,235,.4) !important; }
h2, h3 { color:#f1f5f9 !important; }
.stTabs [data-baseweb='tab'] { color:#64748b !important; }
.stTabs [aria-selected='true'] { color:#38bdf8 !important; }
hr { border-color:#1e293b !important; }
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)


# ===== Data Loading =====
@st.cache_data(show_spinner='Loading pipeline data...')
def load_all_data():
    raw_df, _ = load_and_validate_shipments(SHIPMENT_RECORDS_FILE, strict=True)
    weekly_df = aggregate_weekly_shipments(raw_df)
    enriched  = compute_baselines_and_deviations(weekly_df)
    enriched  = detect_candidate_anomalies(enriched,
                    history_threshold=DEFAULT_HISTORY_THRESHOLD,
                    peer_threshold=DEFAULT_PEER_THRESHOLD)
    report_df = pd.read_csv(FINAL_OUTPUT_FILE) if FINAL_OUTPUT_FILE.is_file() else None
    notes_df  = pd.read_csv(CONTEXT_NOTES_FILE)
    return raw_df, weekly_df, enriched, report_df, notes_df


# ===== Q&A Engine =====
def _find_route(query, routes):
    q = query.lower()
    for r in routes:
        if r.lower() in q:
            return r
        parts = r.split("-")
        if len(parts) == 2 and parts[0].lower() in q and parts[1].lower() in q:
            return r
    return None

def _find_note(query, note_ids):
    qu = query.upper()
    for nid in note_ids:
        if nid in qu:
            return nid
    return None

def answer_query(query, raw_df, weekly_df, enriched, report_df, notes_df):
    """Deterministic Q&A grounded in pipeline data. Returns (markdown, [sources])."""
    q      = query.strip().lower()
    routes = enriched["route"].unique().tolist()
    nids   = notes_df["note_id"].tolist()

    # 1. Anomaly counts
    if any(k in q for k in ["how many anomal","total anomal","anomaly count","how many flagged","number of anomal"]):
        cands    = enriched[enriched["is_anomaly_candidate"]]
        n        = len(cands)
        affected = cands["route"].unique()
        if report_df is not None:
            nj = int((report_df["flagged"] == "No (justified)").sum())
            nu = int((report_df["flagged"] == "Yes").sum())
            parts = [
                f"The pipeline found **{n} anomaly candidates** across **{len(affected)} routes**.",
                "",
                f"- 🟢 **{nj} Justified** - a context note explains the cost rise",
                f"- 🔴 **{nu} Unexplained** - flagged for human review",
                "",
                "Routes: " + ", ".join(f"`{r}`" for r in sorted(affected))
            ]
            txt = "\n".join(parts)
        else:
            txt = f"Found **{n} anomaly candidates** across **{len(affected)} routes**."
        return txt, ["anomaly_detector.py", "freight_anomaly_report.csv"]

    # 2. Most expensive
    if any(k in q for k in ["most expensive","highest cost","costliest"]):
        avg = weekly_df.groupby("route")["cost_per_tonne_km"].mean().sort_values(ascending=False)
        tr, tv = avg.index[0], avg.iloc[0]
        parts = [f"**`{tr}`** has the highest average cost at **₹{tv:.2f}/tonne-km**.", "", "**All routes:**"]
        for r, v in avg.items():
            parts.append(f"- `{r}`: ₹{v:.2f}/t-km")
        return "\n".join(parts), ["weekly_aggregator.py"]

    # 3. Cheapest
    if any(k in q for k in ["cheapest","lowest cost","least expensive"]):
        avg = weekly_df.groupby("route")["cost_per_tonne_km"].mean().sort_values()
        tr, tv = avg.index[0], avg.iloc[0]
        parts = [f"**`{tr}`** has the lowest average cost at **₹{tv:.2f}/tonne-km**.", "", "**All routes:**"]
        for r, v in avg.items():
            parts.append(f"- `{r}`: ₹{v:.2f}/t-km")
        return "\n".join(parts), ["weekly_aggregator.py"]

    # 4. Specific route
    mr = _find_route(query, routes)
    if mr:
        rd  = weekly_df[weekly_df["route"] == mr]
        re_ = enriched[enriched["route"] == mr]
        ac  = int(re_["is_anomaly_candidate"].sum())
        rt  = rd["route_type"].iloc[0] if len(rd) else "Unknown"
        parts = [
            f"### Route: `{mr}` ({rt})", "",
            "| Metric | Value |", "|---|---|",
            f"| Avg cost | ₹{rd['cost_per_tonne_km'].mean():.2f}/t-km |",
            f"| Max cost | ₹{rd['cost_per_tonne_km'].max():.2f}/t-km |",
            f"| Min cost | ₹{rd['cost_per_tonne_km'].min():.2f}/t-km |",
            f"| Weeks of data | {len(rd)} |",
            f"| Anomaly candidates | {ac} |",
            "",
        ]
        if report_df is not None:
            rr = report_df[report_df["route"] == mr]
            if len(rr):
                parts.append("**Anomaly weeks:**")
                for _, row in rr.iterrows():
                    badge = "🟢 Justified" if row["flagged"] == "No (justified)" else "🔴 Flagged"
                    rs = str(row["reason"])
                    short = rs[:110] + ("..." if len(rs)>110 else "")
                    parts.append(f"- **{row['week_of']}** | {row['cost_per_tonne_km']} ₹/t-km | {badge}")
                    parts.append(f"  > *{short}*")
            else:
                parts.append("No anomaly candidates for this route under current thresholds.")
        return "\n".join(parts), ["weekly_aggregator.py", "freight_anomaly_report.csv"]

    # 5. Context note lookup
    mn = _find_note(query, nids)
    if mn or any(k in q for k in ["context note","what notes","all notes","list notes"]):
        if mn:
            row = notes_df[notes_df["note_id"] == mn].iloc[0]
            parts = [
                f"### Note `{mn}`", "",
                f"- **Date**: {row['date']}",
                f"- **Applies to**: {row['applies_to']}",
                f"- **Text**: {row['note']}",
                "",
            ]
            if report_df is not None:
                m = report_df[report_df["matched_note_id"] == mn]
                if len(m):
                    parts.append(f"**Justifies {len(m)} anomaly candidate(s):**")
                    for _, r in m.iterrows():
                        parts.append(f"- `{r['route']}` - week of {r['week_of']}")
                else:
                    parts.append("*This note was not selected to justify any anomaly candidate.*")
            return "\n".join(parts), ["context_notes.csv","guardrails.py"]
        parts = ["### All Context Notes", ""]
        for _, r in notes_df.iterrows():
            parts.append(f"**`{r['note_id']}`** ({r['date']}) - *{r['applies_to']}*")
            parts.append(f"> {r['note']}")
            parts.append("")
        return "\n".join(parts), ["context_notes.csv"]

    # 6. Date query
    dm = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    if dm:
        dt = dm.group(1)
        wk = enriched[enriched["week_of"] == dt]
        if len(wk):
            parts = [f"### Data for week `{dt}`", ""]
            for _, row in wk.iterrows():
                fl = "🔴 Anomaly" if row["is_anomaly_candidate"] else "✅ Normal"
                parts.extend([
                    f"**`{row['route']}`** ({row['route_type']})",
                    f"- Cost: ₹{row['cost_per_tonne_km']:.2f}/t-km",
                    f"- vs History: {row['vs_own_history']}",
                    f"- vs Peers: {row['vs_similar_routes']}",
                    f"- Status: {fl}",
                    "",
                ])
        else:
            parts = [f"No records for `{dt}`. Dataset covers {enriched['week_of'].min()} to {enriched['week_of'].max()}."]
        return "\n".join(parts), ["weekly_aggregator.py","anomaly_detector.py"]

    # 7. Methodology
    if any(k in q for k in ["threshold","methodology","how is anomaly","how does it work","how are anomal","algorithm"]):
        parts = [
            "### Anomaly Detection Methodology", "",
            "A route-week is an **anomaly candidate** when either:", "",
            f"1. **Historical surge** >= **{DEFAULT_HISTORY_THRESHOLD:.0f}%** above the route's trailing 8-week average (current week excluded)",
            f"2. **Peer deviation** >= **{DEFAULT_PEER_THRESHOLD:.0f}%** above same-route-type peer average for that week", "",
            "Either condition alone is sufficient (OR logic). Missing baselines are **never treated as zero**.", "",
            "**Cost formula:**",
            "```",
            "cost_per_tonne_km = sum(freight_cost_inr) / sum(quantity_tonnes x distance_km)",
            "```", "",
            "**Post-detection guardrails:** Each candidate is evaluated against context notes.",
            "Only notes with a genuine causal mechanism within the correct temporal window are accepted.",
        ]
        return "\n".join(parts), ["anomaly_detector.py","baseline_calculator.py","guardrails.py"]

    # 8. Overview
    if any(k in q for k in ["overview","summary","tell me about","describe the data","give me a summary","dataset"]):
        rt = weekly_df.groupby("route_type")["route"].nunique().to_dict()
        parts = [
            "### Dataset Overview", "",
            "| Field | Value |", "|---|---|",
            f"| Date range | {enriched['week_of'].min()} to {enriched['week_of'].max()} |",
            f"| Routes | {enriched['route'].nunique()} ({rt.get('Short',0)} Short, {rt.get('Medium',0)} Medium, {rt.get('Long',0)} Long) |",
            f"| Route-week records | {len(enriched)} |",
            f"| Shipment records | {len(raw_df)} |",
            f"| Context notes | {len(notes_df)} |",
            f"| Anomaly candidates | {int(enriched['is_anomaly_candidate'].sum())} |",
            "",
            "**Routes:**",
        ]
        parts += [f"- `{r}`" for r in sorted(enriched["route"].unique())]
        return "\n".join(parts), ["shipment_records.csv","anomaly_detector.py"]

    # 9. Explanations
    if any(k in q for k in ["explain","why is","why was","reason for","what caused","justify"]):
        if report_df is not None:
            cands = report_df[report_df["flagged"].isin(["Yes","No (justified)"])]
            if len(cands):
                parts = ["### Anomaly Explanations (first 8)", ""]
                for _, row in cands.head(8).iterrows():
                    badge = "🟢" if row["flagged"] == "No (justified)" else "🔴"
                    ntag  = f" `{row['matched_note_id']}`" if row.get("matched_note_id") else ""
                    rs    = str(row["reason"])
                    parts.append(f"{badge} **`{row['route']}`** - {row['week_of']}{ntag}")
                    parts.append(f"> {rs[:150] + ('...' if len(rs)>150 else '')}")
                    parts.append("")
                if len(cands) > 8:
                    parts.append(f"*...{len(cands)-8} more. Ask about a specific route for full details.*")
                return "\n".join(parts), ["freight_anomaly_report.csv","explainer.py"]

    # 10. Route comparison
    if any(k in q for k in ["compare","comparison","all routes by cost","route comparison","which route is"]):
        avg = weekly_df.groupby("route")["cost_per_tonne_km"].agg(["mean","std","count"]).reset_index()
        avg.columns = ["route","avg","std","weeks"]
        avg = avg.sort_values("avg", ascending=False)
        parts = ["### Route Cost Comparison", "", "| Route | Avg (Rs/t-km) | Std Dev | Weeks |", "|---|---|---|---|"]
        for _, r in avg.iterrows():
            parts.append(f"| `{r['route']}` | {r['avg']:.2f} | {r['std']:.2f} | {int(r['weeks'])} |")
        return "\n".join(parts), ["weekly_aggregator.py"]

    # 11. Specific events
    if "flood" in q or ("chennai" in q and "bangalore" in q):
        nr   = notes_df[notes_df["note_id"]=="N001"].iloc[0]
        parts = [f"### Chennai-Bangalore Flooding (N001)", "", f"**{nr['note']}**", "", "Valid for weeks **`2025-02-24`** and **`2025-03-03`** only.", ""]
        if report_df is not None:
            for _, r in report_df[report_df["matched_note_id"]=="N001"].iterrows():
                parts.append(f"- `{r['route']}` week {r['week_of']}: ₹{r['cost_per_tonne_km']}/t-km")
        return "\n".join(parts), ["context_notes.csv","guardrails.py"]

    if "festival" in q or ("ahmedabad" in q and "mumbai" in q):
        nr   = notes_df[notes_df["note_id"]=="N002"].iloc[0]
        parts = [f"### Festival Surcharge (N002)", "", f"**{nr['note']}**", "", "Valid only for week **`2025-01-20`** on Ahmedabad-Mumbai.", ""]
        if report_df is not None:
            for _, r in report_df[report_df["matched_note_id"]=="N002"].iterrows():
                parts.append(f"- `{r['route']}` week {r['week_of']}: ₹{r['cost_per_tonne_km']}/t-km")
        return "\n".join(parts), ["context_notes.csv","guardrails.py"]

    if "diesel" in q:
        nr   = notes_df[notes_df["note_id"]=="N003"].iloc[0]
        parts = [f"### Diesel Price Rise (N003)", "", f"**{nr['note']}**", "", "Applies to **all routes** from `2025-05-05` (validated within 3 weeks).", ""]
        if report_df is not None:
            m = report_df[report_df["matched_note_id"]=="N003"]
            if len(m):
                for _, r in m.iterrows():
                    parts.append(f"- `{r['route']}` week {r['week_of']}")
            else:
                parts.append("*No anomaly candidates matched N003 - the +5-7% diesel effect may be below the 20% threshold.*")
        return "\n".join(parts), ["context_notes.csv","guardrails.py"]

    # 12. Help
    if any(k in q for k in ["help","what can you","capabilities","what questions"]):
        parts = [
            "### What Can I Help With?", "",
            "**Cost and Routes**",
            "- Average / max / min cost for any route",
            "- Compare all routes",
            "- Most / least expensive route", "",
            "**Anomalies**",
            "- Total anomaly count and affected routes",
            "- Specific anomaly weeks and explanations",
            "- Whether a specific week was flagged", "",
            "**Context Notes**",
            "- Lookup any note N001-N010",
            "- Which anomalies a note justifies",
            "- Flooding / festival / diesel events", "",
            "**Methodology**",
            "- How the 20% threshold works",
            "- Historical vs peer baseline",
            "- Anti-hallucination guardrails", "",
            "*Try: 'Tell me about Mumbai-Pune' or 'How many anomalies?'*",
        ]
        return "\n".join(parts), []

    # Fallback
    parts = [
        "I could not match that to pipeline data. Try:", "",
        "- 'How many anomalies were detected?'",
        "- 'Tell me about the Delhi-Jaipur route'",
        "- 'What is note N002?'",
        "- 'How does anomaly detection work?'",
        "- 'What is the most expensive route?'",
    ]
    return "\n".join(parts), []


# ===== Chart helpers =====
_BG   = '#0a0f1c'
_GRID = '#1e293b'

def _base(height=330, margin=None):
    m = margin or dict(l=50, r=20, t=40, b=50)
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(gridcolor=_GRID, linecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, linecolor=_GRID),
        legend=dict(bgcolor="#111827", bordercolor=_GRID),
        height=height, margin=m,
    )

def cost_trend_chart(enr, route, report_df):
    rd = enr[enr["route"]==route].copy()
    aw = []
    if report_df is not None:
        aw = report_df[report_df["route"]==route]["week_of"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rd["week_of"], y=rd["cost_per_tonne_km"],
        mode="lines+markers", name="Weekly Cost",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=6, color="#38bdf8"),
        hovertemplate="Week: %{x}<br>Rs %{y:.2f}/t-km<extra></extra>",
    ))
    if "trailing_history_avg" in rd.columns:
        fig.add_trace(go.Scatter(
            x=rd["week_of"], y=rd["trailing_history_avg"],
            mode="lines", name="8-wk Trailing Avg",
            line=dict(color="#818cf8", width=1.5, dash="dot"),
            hovertemplate="Trailing avg: Rs %{y:.2f}<extra></extra>",
        ))
    ad = rd[rd["week_of"].isin(aw)]
    if len(ad):
        fig.add_trace(go.Scatter(
            x=ad["week_of"], y=ad["cost_per_tonne_km"],
            mode="markers", name="Anomaly",
            marker=dict(size=13, color="#f87171", symbol="diamond", line=dict(color="#fca5a5",width=1.5)),
            hovertemplate="Anomaly: Rs %{y:.2f}/t-km<extra></extra>",
        ))
    fig.update_layout(
        **_base(height=330),
        title=dict(text=f"Cost Trend - {route}", font=dict(color="#f1f5f9", size=14)),
        xaxis_title="Week of (Monday)", yaxis_title="Rs/tonne-km",
    )
    return fig

def route_bar_chart(weekly_df):
    avg = weekly_df.groupby("route")["cost_per_tonne_km"].mean().sort_values(ascending=True).reset_index()
    fig = go.Figure(go.Bar(
        x=avg["cost_per_tonne_km"], y=avg["route"], orientation="h",
        marker_color="#38bdf8",
        hovertemplate="%{y}: Rs %{x:.2f}/t-km<extra></extra>",
    ))
    fig.update_layout(
        **_base(height=300, margin=dict(l=180,r=20,t=40,b=50)),
        title=dict(text="Average Cost by Route", font=dict(color="#f1f5f9", size=14)),
        xaxis_title="Rs/tonne-km",
    )
    return fig

def anomaly_map_chart(enr, report_df):
    if report_df is None or len(report_df)==0:
        return None
    df = report_df.copy()
    df["color"] = df["flagged"].map({"Yes":"#f87171","No (justified)":"#34d399"})
    df["tip"] = df.apply(lambda r: f"{r['route']} | {r['week_of']} | Rs {r['cost_per_tonne_km']}/t-km | {r['flagged']}", axis=1)
    fig = go.Figure(go.Scatter(
        x=df["week_of"], y=df["route"],
        mode="markers",
        marker=dict(size=16, color=df["color"], symbol="square", opacity=.85),
        text=df["tip"], hoverinfo="text",
    ))
    fig.update_layout(
        **_base(height=280, margin=dict(l=180,r=20,t=40,b=50)),
        title=dict(text="Anomaly Map  (green=Justified  red=Flagged)", font=dict(color="#f1f5f9",size=14)),
        xaxis_title="Week",
    )
    return fig


# ===== UI Layout =====
EXAMPLES = [
    "How many anomalies were detected?",
    "Tell me about the Mumbai-Pune route",
    "What is the most expensive route?",
    "Explain the Chennai-Bangalore flooding",
    "What does note N002 say?",
    "How does anomaly detection work?",
    "Compare all routes by cost",
    "What happened in week 2025-01-20?",
]

def render_sidebar(enr, report_df, notes_df):
    with st.sidebar:
        st.markdown("### 🐯 FreightTiger Analyst")
        routes = ["All Routes"] + sorted(enr["route"].unique())
        sel = st.selectbox("Focus route", routes, key="route_sel")
        st.markdown("---")
        st.markdown("### 📝 Context Notes")
        for _, n in notes_df.iterrows():
            nid     = n["note_id"]
            matched = 0
            if report_df is not None:
                matched = int((report_df["matched_note_id"]==nid).sum())
            badge = f'<span class="badge-j">Justifies {matched}</span>' if matched > 0 else ""
            note_txt = str(n["note"])
            short    = note_txt[:95] + ("..." if len(note_txt)>95 else "")
            st.markdown(
                f'<div class="note-card">'
                f'<div class="note-nid">{nid}</div>'
                f'<div class="note-meta">📅 {n["date"]} · {n["applies_to"]}</div>'
                f'<div class="note-body">{short}</div>'
                f'{badge}</div>',
                unsafe_allow_html=True,
            )
        st.caption("Deterministic pipeline | Phase 7 | No hallucination")
    return sel

def render_metrics(raw_df, enr, report_df):
    n_cand = int(enr["is_anomaly_candidate"].sum())
    nj     = int((report_df["flagged"]=="No (justified)").sum()) if report_df is not None else 0
    nu     = n_cand - nj
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📦 Shipments", f"{len(raw_df):,}")
    with c2: st.metric("🗺 Routes", enr["route"].nunique())
    with c3: st.metric("⚠️ Anomaly Candidates", n_cand, delta=f"{nu} unexplained", delta_color="inverse")
    with c4: st.metric("✅ Justified", nj, delta=f"{nj}/{n_cand} resolved", delta_color="normal")

def render_chat_tab(raw_df, weekly_df, enr, report_df, notes_df):
    st.markdown("**💡 Try asking:**")
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        with cols[i%4]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.messages.append({"role":"user","content":ex})
                ans, srcs = answer_query(ex, raw_df, weekly_df, enr, report_df, notes_df)
                st.session_state.messages.append({"role":"assistant","content":ans,"sources":srcs})
                st.rerun()
    st.markdown("---")
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🐯"):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        st.caption("📎 Sources: " + " | ".join(f"`{s}`" for s in msg["sources"]))
        if st.button("🗑 Clear conversation", key="clr"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("👋 Click an example above or type your question below.")
    if prompt := st.chat_input("Ask about freight costs, anomalies, or context notes...", key="chat_input"):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.spinner("Analysing pipeline data..."):
            ans, srcs = answer_query(prompt, raw_df, weekly_df, enr, report_df, notes_df)
        st.session_state.messages.append({"role":"assistant","content":ans,"sources":srcs})
        st.rerun()

def render_charts_tab(enr, report_df, weekly_df, sel):
    if sel != "All Routes":
        st.plotly_chart(cost_trend_chart(enr, sel, report_df), use_container_width=True)
        rd = weekly_df[weekly_df["route"]==sel]
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg", f"Rs {rd['cost_per_tonne_km'].mean():.2f}/t-km")
        c2.metric("Max", f"Rs {rd['cost_per_tonne_km'].max():.2f}/t-km")
        c3.metric("Min", f"Rs {rd['cost_per_tonne_km'].min():.2f}/t-km")
    else:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(route_bar_chart(weekly_df), use_container_width=True)
        with col2:
            fig = anomaly_map_chart(enr, report_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Run the pipeline first to see the anomaly map.")
        st.markdown("#### Cost Trends by Route")
        route_list = sorted(enr["route"].unique())
        route_tabs = st.tabs(route_list)
        for tab, route in zip(route_tabs, route_list):
            with tab:
                st.plotly_chart(cost_trend_chart(enr, route, report_df), use_container_width=True)

def render_data_tab(report_df, sel):
    if report_df is None:
        st.info("Run `python run_pipeline.py` to generate the anomaly report first.")
        return
    data = report_df if sel=="All Routes" else report_df[report_df["route"]==sel]
    if len(data)==0:
        st.info(f"No anomaly records for `{sel}`.")
        return
    def style_row(row):
        if row.get("flagged")=="Yes":
            return ["background-color:rgba(248,113,113,0.08)"]*len(row)
        elif row.get("flagged")=="No (justified)":
            return ["background-color:rgba(52,211,153,0.08)"]*len(row)
        return [""]*len(row)
    st.dataframe(data.style.apply(style_row, axis=1), use_container_width=True, height=440, hide_index=True)
    st.caption(f"{len(data)} rows | green=Justified | red=Unexplained flagged")


# ===== Main =====
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    try:
        raw_df, weekly_df, enr, report_df, notes_df = load_all_data()
    except Exception as e:
        st.error(f"Failed to load pipeline data: {e}")
        st.info("Ensure data/raw/shipment_records.csv and data/raw/context_notes.csv are present.")
        st.stop()
    st.markdown('''
    <div class="hero">
        <h1>🐯 FreightTiger AI Analyst</h1>
        <p>Ask anything about freight costs, anomalies, and context notes - powered by your deterministic pipeline, not guesswork.</p>
    </div>''', unsafe_allow_html=True)
    sel = render_sidebar(enr, report_df, notes_df)
    render_metrics(raw_df, enr, report_df)
    st.markdown("---")
    tab_chat, tab_charts, tab_data = st.tabs([
        "💬 AI Analyst Chat",
        "📈 Visual Explorer",
        "📋 Anomaly Data",
    ])
    with tab_chat:   render_chat_tab(raw_df, weekly_df, enr, report_df, notes_df)
    with tab_charts: render_charts_tab(enr, report_df, weekly_df, sel)
    with tab_data:   render_data_tab(report_df, sel)

if __name__ == "__main__":
    main()