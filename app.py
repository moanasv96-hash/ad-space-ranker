"""
Ad Space Ranking System — Streamlit UI  (Maroon Theme)
Run with:  streamlit run app.py
Place advertising_inventory_dataset.csv in the same folder.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Space Ranker",
    page_icon="📢",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
TARGET       = "match_score"
DATA_PATH    = "advertising_inventory_dataset.csv"

# ─────────────────────────────────────────────
# COLOUR PALETTE  (maroon-based)
# ─────────────────────────────────────────────
M_DARK    = "#4A0000"   # deep maroon – sidebar / hero dark end
M_MID     = "#800020"   # true maroon – primary
M_LIGHT   = "#B03060"   # rose-maroon – accent / hover
M_ROSE    = "#FFE4E8"   # blush – very light tint for card backgrounds
M_GOLD    = "#C9A84C"   # warm gold – success / rank #1
M_SILVER  = "#8A9BAE"   # steel – rank #2
M_BRONZE  = "#A0673A"   # bronze – rank #3
M_CREAM   = "#FFF8F8"   # near-white with warm tint – page bg
M_INK     = "#1A0A0A"   # near-black text on light bg
M_MUTED   = "#6B4050"   # muted maroon text
DANGER    = "#E53E3E"
WARNING   = "#D97706"
SUCCESS   = "#2D6A4F"

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* ── page background ── */
.stApp {{ background-color: {M_CREAM}; }}

/* ── hide default streamlit chrome ── */
#MainMenu, footer {{ visibility: hidden; }}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(175deg, {M_DARK} 0%, {M_MID} 70%, {M_LIGHT} 100%);
    border-right: 3px solid {M_GOLD};
}}
[data-testid="stSidebar"] * {{ color: #FFEEF0 !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: #FFD6DA !important; font-weight: 700 !important; }}
[data-testid="stSidebar"] label {{ color: #FFCDD2 !important; font-size: 0.8rem !important; font-weight:600 !important; }}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input {{
    background: rgba(255,255,255,0.12) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"] {{
    background: {M_GOLD} !important;
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.2) !important; }}

/* ── sidebar button ── */
[data-testid="stSidebar"] .stButton > button {{
    background: {M_GOLD} !important;
    color: {M_DARK} !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em;
    box-shadow: 0 3px 10px rgba(0,0,0,0.3);
    transition: all .2s;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: #E5C35A !important;
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(0,0,0,0.35);
}}

/* ── hero banner ── */
.hero {{
    background: linear-gradient(130deg, {M_DARK} 0%, {M_MID} 55%, {M_LIGHT} 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    border: 1px solid {M_GOLD}55;
    box-shadow: 0 4px 20px rgba(74,0,0,0.25);
}}
.hero h1 {{ font-size: 1.9rem; font-weight: 800; margin: 0 0 0.3rem; color: #fff; }}
.hero p  {{ font-size: 0.95rem; opacity: 0.88; margin: 0; color: #FFD6DA; }}
.hero-accent {{ color: {M_GOLD}; font-weight: 700; }}

/* ── section headers ── */
.sec-header {{
    font-size: 1rem;
    font-weight: 700;
    color: {M_DARK};
    border-left: 4px solid {M_MID};
    padding: 0.1rem 0 0.1rem 0.65rem;
    margin: 1.5rem 0 0.75rem;
    background: linear-gradient(90deg, {M_ROSE} 0%, transparent 100%);
    border-radius: 0 6px 6px 0;
}}

/* ── info / tip box ── */
.info-box {{
    background: {M_ROSE};
    border: 1px solid #F5C6CB;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.84rem;
    color: {M_DARK};
    margin: 0.5rem 0 1rem;
    line-height: 1.5;
}}
.info-box strong {{ color: {M_MID}; }}

/* ── metric cards overrides ── */
[data-testid="metric-container"] {{
    background: white;
    border: 1px solid #F5C6CB;
    border-radius: 12px;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 1px 6px rgba(128,0,32,0.08);
    border-top: 3px solid {M_MID};
}}
[data-testid="metric-container"] label {{
    color: {M_MUTED} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="metric-container"] [data-testid="metric-value"] {{
    color: {M_DARK} !important;
    font-weight: 800 !important;
    font-size: 1.4rem !important;
}}
[data-testid="metric-container"] [data-testid="metric-delta"] {{
    color: {SUCCESS} !important;
}}

[data-testid="stMetricLabel"] {{
    color: #64748B !important; /* Cool grey text */
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

[data-testid="stMetricValue"] {{
    color: #1E293B !important; /* Dark slate text */
    font-weight: 800 !important;
    font-size: 1.5rem !important;
}}

[data-testid="column"] {{
    background: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
    border-top: 4px solid #4F46E5; /* Adds an Indigo accent top border */
}}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: white;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #F5C6CB;
    box-shadow: 0 1px 4px rgba(128,0,32,0.07);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.88rem;
    color: {M_MUTED};
    background: transparent;
    border: none;
    padding: 0.4rem 1.2rem;
    transition: all .15s;
}}
.stTabs [aria-selected="true"] {{
    background: {M_MID} !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(128,0,32,0.3);
}}

/* ── rank table ── */
.rank-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.82rem;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(128,0,32,0.1);
}}
.rank-table thead tr {{
    background: linear-gradient(90deg, {M_DARK}, {M_MID});
}}
.rank-table th {{
    color: #FFEEF0;
    font-weight: 700;
    padding: 0.65rem 0.85rem;
    text-align: left;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    white-space: nowrap;
}}
.rank-table tbody tr:nth-child(even) td {{ background: #FFF0F3; }}
.rank-table tbody tr:nth-child(odd)  td {{ background: white; }}
.rank-table tbody tr:hover td {{
    background: #FFE0E6 !important;
    transition: background .1s;
}}
.rank-table td {{
    padding: 0.5rem 0.85rem;
    color: {M_INK};
    border-bottom: 1px solid #FFD6DA;
    vertical-align: middle;
}}
.rank-table td small {{ color: {M_MUTED}; }}

/* ── rank badges ── */
.rank-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: {M_MID};
    color: white;
    font-weight: 800;
    font-size: 0.78rem;
    box-shadow: 0 2px 6px rgba(128,0,32,0.4);
}}
.rank-badge.gold   {{ background: linear-gradient(135deg,#C9A84C,#E8C96B); color:{M_DARK}; }}
.rank-badge.silver {{ background: linear-gradient(135deg,#8A9BAE,#B0BEC5); color:white; }}
.rank-badge.bronze {{ background: linear-gradient(135deg,#A0673A,#C68B5A); color:white; }}

/* ── channel pill ── */
.ch-pill {{
    display: inline-block;
    background: {M_ROSE};
    color: {M_MID};
    border: 1px solid #F5C6CB;
    border-radius: 99px;
    padding: 2px 9px;
    font-size: 0.73rem;
    font-weight: 700;
    white-space: nowrap;
}}

/* ── score bar ── */
.score-wrap {{ display:flex; align-items:center; gap:0.5rem; }}
.score-bar-bg {{
    background: #FFD6DA;
    border-radius: 99px;
    height: 8px;
    flex: 1;
    min-width: 60px;
}}
.score-bar-fill {{
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, {M_MID}, {M_GOLD});
}}
.score-val {{ font-weight: 700; font-size: 0.82rem; min-width: 38px; }}

/* ── best match hero card ── */
.best-card {{
    background: linear-gradient(130deg, {M_DARK} 0%, {M_MID} 100%);
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    color: white;
    border: 1px solid {M_GOLD}44;
    box-shadow: 0 4px 18px rgba(74,0,0,0.3);
}}
.best-card .label {{font-size:.75rem;opacity:.75;font-weight:600;text-transform:uppercase;letter-spacing:.07em;}}
.best-card .name  {{font-size:1.5rem;font-weight:800;margin:.25rem 0 .1rem;}}
.best-card .sub   {{font-size:.88rem;color:#FFD6DA;}}

/* ── download button ── */
.stDownloadButton > button {{
    background: {M_MID} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
}}
.stDownloadButton > button:hover {{
    background: {M_DARK} !important;
}}

/* ── expander ── */
.streamlit-expanderHeader {{
    background: {M_ROSE} !important;
    color: {M_DARK} !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}}

/* ── dataframe ── */
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ML HELPERS
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["audience_aligned"] = (df["target_demographic"] == df["primary_audience"]).astype(int)
    df["budget_ratio"] = (df["campaign_budget_lkr"] / df["base_cost_lkr"].replace(0, np.nan)).fillna(0).clip(upper=10)
    df["log_reach_to_cost"] = np.log1p(df["reach_to_cost_ratio"])
    for col in ["historical_ctr", "historical_roi", "historical_conversion_rate"]:
        mn, mx = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - mn) / (mx - mn + 1e-9)
    df["hist_quality"] = df[["historical_ctr_norm", "historical_roi_norm",
                               "historical_conversion_rate_norm"]].mean(axis=1)
    df.drop(columns=["historical_ctr_norm", "historical_roi_norm",
                      "historical_conversion_rate_norm"], inplace=True)
    df["log_bookings"] = np.log1p(df["previous_bookings"])
    df.drop(columns=["reach_to_cost_ratio", "historical_ctr",
                      "historical_roi", "historical_conversion_rate"],
            inplace=True, errors="ignore")
    return df


def build_preprocessor(cat_cols, num_cols):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ], remainder="drop")


@st.cache_resource(show_spinner=False)
def train_and_cache(data_path: str):
    df_raw = pd.read_csv(data_path)
    df_eng = engineer_features(df_raw)
    FEATURES_TO_DROP = [TARGET, "target_location", "primary_audience"]
    X = df_eng.drop(columns=FEATURES_TO_DROP, errors="ignore")
    y = df_eng[TARGET]
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    models = {
        "Ridge":             Ridge(alpha=10.0),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=8,
                                                    random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=5, subsample=0.8,
                                                        random_state=RANDOM_STATE),
        "XGBoost":           xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                               max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                               eval_metric="rmse", random_state=RANDOM_STATE,
                                               verbosity=0),
    }
    results, pipes = {}, {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        cv   = cross_val_score(pipe, X_train, y_train,
                               cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
                               scoring="r2", n_jobs=-1)
        pipe.fit(X_train, y_train)
        yp = pipe.predict(X_test)
        results[name] = {
            "CV R²": cv.mean(), "CV std": cv.std(),
            "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
            "MAE":  mean_absolute_error(y_test, yp),
            "R²":   r2_score(y_test, yp),
        }
        pipes[name] = pipe
    best_name = max(results, key=lambda k: results[k]["R²"])
    best_pipe = pipes[best_name]
    try:
        m = best_pipe.named_steps["model"]
        p = best_pipe.named_steps["prep"]
        feat_names = (list(p.named_transformers_["cat"].get_feature_names_out()) +
                      list(p.named_transformers_["num"].get_feature_names_out()))
        fi = pd.DataFrame({"feature": feat_names, "importance": m.feature_importances_})
        fi = fi.sort_values("importance", ascending=False).head(15).reset_index(drop=True)
    except Exception:
        fi = pd.DataFrame()
    return best_pipe, results, best_name, X_test, y_test, fi, df_raw


def rank_ad_spaces(model_pipe, df_inventory, advertiser, top_n=10):
    df = df_inventory.copy()
    df["audience_aligned"]    = (df["target_demographic"] == df["primary_audience"]).astype(int)
    df["business_type"]       = advertiser["business_type"]
    df["target_demographic"]  = advertiser["target_demographic"]
    df["target_location"]     = advertiser["target_location"]
    df["campaign_budget_lkr"] = advertiser["campaign_budget_lkr"]
    df["campaign_goal"]       = advertiser["campaign_goal"]
    df["budget_gap_lkr"]      = df["campaign_budget_lkr"] - df["base_cost_lkr"]
    df["budget_ratio"]        = (df["campaign_budget_lkr"] / df["base_cost_lkr"].replace(0, np.nan)).fillna(0).clip(upper=10)
    df["log_reach_to_cost"]   = np.log1p(df["reach_to_cost_ratio"])
    df["log_bookings"]        = np.log1p(df["previous_bookings"])
    for col in ["historical_ctr", "historical_roi", "historical_conversion_rate"]:
        mn, mx = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - mn) / (mx - mn + 1e-9)
    df["hist_quality"] = df[["historical_ctr_norm", "historical_roi_norm",
                               "historical_conversion_rate_norm"]].mean(axis=1)
    df.drop(columns=["historical_ctr_norm", "historical_roi_norm", "historical_conversion_rate_norm",
                      "reach_to_cost_ratio", "historical_ctr", "historical_roi",
                      "historical_conversion_rate"], inplace=True, errors="ignore")
    X_infer = df.drop(columns=[TARGET, "target_location", "primary_audience"], errors="ignore")
    df["predicted_match_score"] = model_pipe.predict(X_infer).clip(0, 1)
    top = df.sort_values("predicted_match_score", ascending=False).head(top_n).reset_index(drop=True)
    top.index += 1
    return top


# ─────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────

def rank_badge(rank: int) -> str:
    cls = {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "")
    return f'<span class="rank-badge {cls}">{rank}</span>'


def score_bar_html(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.80:   col = M_GOLD
    elif score >= 0.65: col = WARNING
    else:               col = DANGER
    return (f'<div class="score-wrap">'
            f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
            f'<span class="score-val" style="color:{col}">{score:.3f}</span>'
            f'</div>')


def plotly_maroon_layout(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="#FFF8F8",
        font=dict(family="Inter", color=M_INK),
        margin=dict(l=10, r=15, t=40, b=10),
        title_font=dict(size=13, color=M_DARK, family="Inter"),
    )
    fig.update_xaxes(gridcolor="#FFE0E6", linecolor="#FFD6DA", tickfont=dict(color=M_MUTED))
    fig.update_yaxes(gridcolor="#FFE0E6", linecolor="#FFD6DA", tickfont=dict(color=M_MUTED))
    return fig

MAROON_SEQ = [M_ROSE, "#E8A0AA", M_LIGHT, M_MID, M_DARK]
MAROON_DIV = [M_ROSE, M_LIGHT, M_MID, M_DARK, "#1A0000"]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📢 Ad Space Ranker")
    st.markdown("*Find your perfect ad placement*")
    st.divider()
    st.markdown("### 🏢 Business Profile")

    if os.path.exists(DATA_PATH):
        _df = pd.read_csv(DATA_PATH)
        biz_types    = sorted(_df["business_type"].unique())
        demographics = sorted(_df["target_demographic"].unique())
        locations    = sorted(_df["target_location"].unique())
        goals        = sorted(_df["campaign_goal"].unique())
    else:
        biz_types    = ["E-commerce", "Education", "Healthcare", "FMCG"]
        demographics = ["18-24", "25-34", "35-44", "45-54", "55+",
                        "Students", "Affluent", "Young Professionals"]
        locations    = ["Colombo", "Battaramulla", "Kadawatha"]
        goals        = ["Brand Awareness", "Sales Conversion",
                        "Lead Generation", "Website Traffic"]

    business_type      = st.selectbox("Business Type",        biz_types)
    target_demographic = st.selectbox("Target Demographic",   demographics)
    target_location    = st.selectbox("Target Location",      locations)
    campaign_goal      = st.selectbox("Campaign Goal",        goals)

    st.markdown("### 💰 Campaign Budget")
    campaign_budget = st.number_input(
        "Budget (LKR)", min_value=30_000, max_value=800_000,
        value=200_000, step=10_000, format="%d")
    st.markdown(
        f"<small style='color:#FFCDD2;'>≈ USD&nbsp;{campaign_budget/300:,.0f} "
        f"&nbsp;|&nbsp; ≈ EUR&nbsp;{campaign_budget/330:,.0f}</small>",
        unsafe_allow_html=True)

    st.markdown("### 🎯 Results")
    top_n = st.slider("Top-N recommendations", 5, 20, 10)

    st.divider()
    run_btn = st.button("🚀  Find Best Ad Spaces", use_container_width=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📢 Ad Space Ranking System</h1>
  <p>AI-powered placement recommendations — find the best ad spaces for your
     <span class="hero-accent">budget</span>,
     <span class="hero-accent">audience</span> &amp;
     <span class="hero-accent">campaign goals</span>.</p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: `{DATA_PATH}` — place the CSV in the same folder.")
    st.stop()

with st.spinner("⏳ Training model on first load (~15 s) …"):
    best_pipe, model_results, best_name, X_test, y_test, fi_df, df_raw = \
        train_and_cache(DATA_PATH)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_rec, tab_eda, tab_model = st.tabs(
    ["🎯  Recommendations", "📊  Data Explorer", "🧠  Model Insights"])


# ═══════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ═══════════════════════════════════════════════
with tab_rec:

    if not run_btn:
        st.markdown(f"""
        <div class="info-box">
            👈 Complete your <strong>business profile</strong> in the sidebar, then click
            <strong>Find Best Ad Spaces</strong> to get AI-ranked placement recommendations
            tailored to your budget, audience, and campaign goal.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header">Inventory Snapshot</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Ad Spaces",  f"{len(df_raw):,}")
        c2.metric("Placement Types",  df_raw["placement_name"].nunique())
        c3.metric("Channel Types",    df_raw["channel_type"].nunique())
        c4.metric("Districts",        df_raw["district"].nunique())

    else:
        advertiser = {
            "business_type":       business_type,
            "target_demographic":  target_demographic,
            "target_location":     target_location,
            "campaign_budget_lkr": campaign_budget,
            "campaign_goal":       campaign_goal,
        }
        with st.spinner("Scoring all ad spaces …"):
            ranked = rank_ad_spaces(best_pipe, df_raw, advertiser, top_n=top_n)

        # ── Campaign profile strip ────────────────────────────
        st.markdown('<div class="sec-header">Your Campaign Profile</div>', unsafe_allow_html=True)
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        for col, lbl, val in zip([pc1, pc2, pc3, pc4, pc5],
            ["Business", "Demographic", "Location", "Budget (LKR)", "Goal"],
            [business_type, target_demographic, target_location,
             f"{campaign_budget:,}", campaign_goal]):
            col.metric(lbl, val)

        st.divider()

        # ── Best match hero ───────────────────────────────────
        best_score   = ranked["predicted_match_score"].iloc[0]
        best_place   = ranked["placement_name"].iloc[0]
        best_channel = ranked["channel_type"].iloc[0]
        best_cost    = ranked["base_cost_lkr"].iloc[0]
        best_reach   = ranked["estimated_reach"].iloc[0]

        h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
        with h1:
            st.markdown(f"""
            <div class="best-card">
              <div class="label">🏆 #1 Best Match</div>
              <div class="name">{best_place}</div>
              <div class="sub">{best_channel} channel · {ranked["listing_type"].iloc[0]}</div>
            </div>""", unsafe_allow_html=True)
        h2.metric("Match Score",     f"{best_score:.3f}", delta="Top pick")
        h3.metric("Base Cost (LKR)", f"{int(best_cost):,}")
        h4.metric("Est. Reach",      f"{int(best_reach):,}")

        # ── Rank table ────────────────────────────────────────
        st.markdown('<div class="sec-header">Top Recommendations</div>', unsafe_allow_html=True)

        rows = []
        for rank, row in ranked.iterrows():
            budget_ok  = "✅" if row["base_cost_lkr"] <= campaign_budget else "⚠️"
            rows.append(
                f"<tr>"
                f"<td>{rank_badge(rank)}</td>"
                f"<td><strong style='color:{M_DARK}'>{row['placement_name']}</strong>"
                f"<br><small>{row['listing_type']}</small></td>"
                f"<td><span class='ch-pill'>{row['channel_type']}</span></td>"
                f"<td style='color:{M_INK}'>{row['district']}</td>"
                f"<td>{budget_ok}&nbsp;<strong>{int(row['base_cost_lkr']):,}</strong></td>"
                f"<td>{int(row['estimated_reach']):,}</td>"
                f"<td>{row['audience_match_score']*100:.0f}%</td>"
                f"<td>{row['location_match_score']*100:.0f}%</td>"
                f"<td>{row['channel_match_score']*100:.0f}%</td>"
                f"<td>{score_bar_html(row['predicted_match_score'])}</td>"
                f"</tr>"
            )

        st.markdown(
            f"""<table class="rank-table">
              <thead><tr>
                <th>#</th><th>Placement</th><th>Channel</th><th>District</th>
                <th>Cost (LKR)</th><th>Est. Reach</th>
                <th>Audience</th><th>Location</th><th>Channel</th>
                <th>Match Score</th>
              </tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>""",
            unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────
        st.markdown('<div class="sec-header">Visual Breakdown</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        with ch1:
            fig_bar = px.bar(
                ranked.reset_index().rename(columns={"index": "Rank"}),
                x="predicted_match_score", y="placement_name",
                orientation="h", color="predicted_match_score",
                color_continuous_scale=MAROON_SEQ,
                text=ranked["predicted_match_score"].map(lambda x: f"{x:.3f}").values,
                labels={"predicted_match_score": "Match Score", "placement_name": "Placement"},
                title=f"Top-{top_n} Match Scores",
            )
            fig_bar.update_traces(textposition="outside", textfont_color=M_DARK)
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(plotly_maroon_layout(fig_bar, 420), use_container_width=True, theme=None)

        with ch2:
            top1 = ranked.iloc[0]
            cats = ["Audience", "Location", "Channel", "Budget Fit", "Hist. Quality"]
            vals = [
                top1["audience_match_score"],
                top1["location_match_score"],
                top1["channel_match_score"],
                min(campaign_budget / max(top1["base_cost_lkr"], 1), 1),
                top1.get("hist_quality", 0.5),
            ]
            fig_rad = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself",
                line=dict(color=M_MID, width=2),
                fillcolor=f"rgba(128,0,32,0.18)",
                name=top1["placement_name"],
            ))
            fig_rad.update_layout(
                polar=dict(
                    bgcolor="#FFF8F8",
                    radialaxis=dict(visible=True, range=[0,1],
                                    tickfont=dict(color=M_MUTED, size=9),
                                    gridcolor="#FFD6DA"),
                    angularaxis=dict(tickfont=dict(color=M_DARK, size=10),
                                     gridcolor="#FFD6DA"),
                ),
                title=dict(text=f"#1 Score Dimensions — {top1['placement_name'][:28]}",
                           font=dict(color=M_DARK, size=12)),
                paper_bgcolor="white",
                font=dict(family="Inter"),
                height=420,
                margin=dict(l=30, r=30, t=50, b=10),
            )
            st.plotly_chart(fig_rad, use_container_width=True, theme=None)

        ch3, ch4 = st.columns(2)
        with ch3:
            dist_ch = ranked["channel_type"].value_counts().reset_index()
            dist_ch.columns = ["Channel", "Count"]
            fig_pie = px.pie(dist_ch, names="Channel", values="Count",
                             color_discrete_sequence=[M_MID, M_LIGHT, M_GOLD,
                                                      M_DARK, "#D4A0A8", M_SILVER, M_BRONZE],
                             title="Channel Mix in Top Picks", hole=0.42)
            fig_pie.update_traces(textfont_color="white", textfont_size=11)
            fig_pie.update_layout(paper_bgcolor="white", font_family="Inter",
                                  title_font_color=M_DARK, title_font_size=13,
                                  height=300, margin=dict(l=0,r=0,t=40,b=0),
                                  legend_font_color=M_INK)
            st.plotly_chart(fig_pie, use_container_width=True, theme=None)

        with ch4:
            fig_sc = px.scatter(
                ranked.reset_index(), x="base_cost_lkr", y="estimated_reach",
                color="predicted_match_score", size="predicted_match_score",
                hover_name="placement_name",
                color_continuous_scale=MAROON_SEQ,
                labels={"base_cost_lkr": "Base Cost (LKR)",
                        "estimated_reach": "Estimated Reach"},
                title="Cost vs Reach (colour = match score)",
            )
            fig_sc.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plotly_maroon_layout(fig_sc, 300), use_container_width=True, theme=None)

        # ── Export ────────────────────────────────────────────
        st.divider()
        exp_cols = ["placement_name", "listing_type", "channel_type", "district",
                    "base_cost_lkr", "estimated_reach", "audience_match_score",
                    "location_match_score", "channel_match_score", "predicted_match_score"]
        export_df = ranked[[c for c in exp_cols if c in ranked.columns]]
        export_df.index.name = "Rank"
        st.download_button(
            "⬇️  Download Results as CSV",
            export_df.to_csv().encode(),
            file_name="top_ad_spaces.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ═══════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="sec-header">Dataset Overview</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records",   f"{len(df_raw):,}")
    m2.metric("Avg Match Score", f"{df_raw[TARGET].mean():.3f}")
    m3.metric("Placement Types", df_raw["placement_name"].nunique())
    m4.metric("Business Types",  df_raw["business_type"].nunique())

    st.markdown('<div class="sec-header">Match Score Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(df_raw, x=TARGET, nbins=35,
                             color_discrete_sequence=[M_MID],
                             labels={TARGET: "Match Score"},
                             title="Distribution of match_score across all inventory")
    fig_hist.update_traces(marker_line_color=M_DARK, marker_line_width=0.5)
    st.plotly_chart(plotly_maroon_layout(fig_hist, 260), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-header">Avg Score by Channel</div>', unsafe_allow_html=True)
        ch_avg = df_raw.groupby("channel_type")[TARGET].mean().sort_values().reset_index()
        fig_ch = px.bar(ch_avg, x=TARGET, y="channel_type", orientation="h",
                        color=TARGET, color_continuous_scale=MAROON_SEQ,
                        text=ch_avg[TARGET].map(lambda x: f"{x:.3f}"),
                        labels={TARGET: "Avg Match Score", "channel_type": "Channel"})
        fig_ch.update_traces(textposition="outside", textfont_color=M_INK)
        fig_ch.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plotly_maroon_layout(fig_ch, 300), use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-header">Avg Score by Listing Type</div>', unsafe_allow_html=True)
        lt_avg = df_raw.groupby("listing_type")[TARGET].mean().sort_values().reset_index()
        fig_lt = px.bar(lt_avg, x=TARGET, y="listing_type", orientation="h",
                        color=TARGET, color_continuous_scale=MAROON_SEQ,
                        text=lt_avg[TARGET].map(lambda x: f"{x:.3f}"),
                        labels={TARGET: "Avg Match Score", "listing_type": "Listing Type"})
        fig_lt.update_traces(textposition="outside", textfont_color=M_INK)
        fig_lt.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plotly_maroon_layout(fig_lt, 360), use_container_width=True)

    st.markdown('<div class="sec-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_df = df_raw.select_dtypes(include="number")
    fig_hm = px.imshow(num_df.corr(),
                        color_continuous_scale=["#FFE8EC", M_LIGHT, M_MID, M_DARK],
                        zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                        title="Pearson Correlation Matrix")
    fig_hm.update_layout(paper_bgcolor="white", font_family="Inter",
                          title_font_color=M_DARK, height=500,
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown('<div class="sec-header">Business Type × Channel</div>', unsafe_allow_html=True)
    pivot = df_raw.pivot_table(values=TARGET, index="business_type",
                                columns="channel_type", aggfunc="mean")
    fig_pv = px.imshow(pivot, color_continuous_scale=MAROON_SEQ,
                        text_auto=".2f", aspect="auto",
                        title="Avg Match Score: Business Type vs Channel")
    fig_pv.update_layout(paper_bgcolor="white", font_family="Inter",
                          title_font_color=M_DARK, height=420,
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_pv, use_container_width=True)

    with st.expander("🗂️ Raw Data Preview (first 50 rows)"):
        st.dataframe(df_raw.head(50), use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="sec-header">Model Comparison</div>', unsafe_allow_html=True)

    res_df = pd.DataFrame(model_results).T.sort_values("R²", ascending=False)
    disp   = res_df.copy()
    for c in ["CV R²", "CV std", "RMSE", "MAE", "R²"]:
        disp[c] = disp[c].map(lambda x: f"{x:.4f}")
    disp.index.name = "Model"

    def highlight_best(row):
        is_best = row.name == best_name
        style   = f"background-color:{M_ROSE};font-weight:700;color:{M_DARK}"
        return [style if is_best else "" for _ in row]

    st.dataframe(disp.style.apply(highlight_best, axis=1), use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
        ✔ <strong>Best model: {best_name}</strong> &nbsp;—&nbsp;
        Test R² = {model_results[best_name]['R²']:.4f} &nbsp;|&nbsp;
        RMSE = {model_results[best_name]['RMSE']:.4f} &nbsp;|&nbsp;
        MAE = {model_results[best_name]['MAE']:.4f}
    </div>""", unsafe_allow_html=True)

    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown('<div class="sec-header">R² by Model</div>', unsafe_allow_html=True)
        fig_r2 = px.bar(res_df.reset_index(), x="index", y="R²",
                        color="R²", color_continuous_scale=MAROON_SEQ,
                        text=res_df["R²"].map(lambda x: f"{x:.3f}").values,
                        labels={"index": "Model"})
        fig_r2.update_traces(textposition="outside", textfont_color=M_INK)
        fig_r2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plotly_maroon_layout(fig_r2, 310), use_container_width=True)

    with cm2:
        st.markdown('<div class="sec-header">RMSE vs MAE</div>', unsafe_allow_html=True)
        err_df = res_df[["RMSE", "MAE"]].reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Value")
        fig_err = px.bar(err_df, x="index", y="Value", color="Metric", barmode="group",
                         labels={"index": "Model"},
                         color_discrete_sequence=[M_MID, M_GOLD])
        st.plotly_chart(plotly_maroon_layout(fig_err, 310), use_container_width=True)

    if not fi_df.empty:
        st.markdown('<div class="sec-header">Top-15 Feature Importances</div>', unsafe_allow_html=True)
        fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale=MAROON_SEQ,
                        text=fi_df["importance"].map(lambda x: f"{x:.3f}"),
                        labels={"importance": "Importance", "feature": "Feature"})
        fig_fi.update_traces(textposition="outside", textfont_color=M_INK)
        fig_fi.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(plotly_maroon_layout(fig_fi, 500), use_container_width=True)

    st.markdown('<div class="sec-header">Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test)
    mn, mx = float(y_test.min()), float(y_test.max())
    fig_av = go.Figure()
    fig_av.add_trace(go.Scatter(
        x=y_test.values, y=y_pred, mode="markers",
        marker=dict(color=M_MID, opacity=0.5, size=5, line=dict(width=0)),
        name="Predictions"))
    fig_av.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color=M_GOLD, dash="dash", width=2), name="Perfect fit"))
    fig_av.update_layout(
        xaxis_title="Actual match_score", yaxis_title="Predicted match_score")
    st.plotly_chart(plotly_maroon_layout(fig_av, 380), use_container_width=True)

    residuals = y_test.values - y_pred
    fig_res = px.histogram(x=residuals, nbins=35,
                            color_discrete_sequence=[M_LIGHT],
                            labels={"x": "Residual (Actual − Predicted)"},
                            title="Residual Distribution")
    fig_res.update_traces(marker_line_color=M_DARK, marker_line_width=0.5)
    st.plotly_chart(plotly_maroon_layout(fig_res, 260), use_container_width=True, theme=None)