"""
Football Match Prediction App
================================
Streamlit app for predicting match outcomes and goal totals.
Based on pre-match features: Elo ratings, form, historical xG, betting odds.

Usage:
    pip install streamlit pandas numpy scikit-learn xgboost joblib
    streamlit run football_predictor_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark tactical board aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --pitch-dark:   #0a0e0c;
    --pitch-mid:    #111a14;
    --pitch-light:  #18261c;
    --grass-line:   #1e3523;
    --neon-green:   #39ff75;
    --neon-amber:   #ffbf00;
    --neon-red:     #ff3b5c;
    --text-primary: #e8f0e9;
    --text-muted:   #6b8a70;
    --card-bg:      rgba(24, 38, 28, 0.85);
    --border:       rgba(57, 255, 117, 0.18);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--pitch-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace;
}

[data-testid="stSidebar"] {
    background: var(--pitch-mid) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}

h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-primary) !important;
}

.stButton > button {
    background: var(--neon-green) !important;
    color: var(--pitch-dark) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.55rem 2rem !important;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #fff !important;
    transform: translateY(-1px);
    box-shadow: 0 0 18px rgba(57,255,117,0.35);
}

.stFileUploader {
    border: 1px dashed var(--border) !important;
    border-radius: 4px;
    background: var(--pitch-light) !important;
    padding: 0.5rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: var(--neon-green) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.8rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--pitch-mid) !important;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.6rem 1.4rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: var(--neon-green) !important;
    border-bottom: 2px solid var(--neon-green) !important;
    background: rgba(57,255,117,0.06) !important;
}

/* Selectbox / inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: var(--pitch-light) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}

/* Dataframes */
.stDataFrame { border: 1px solid var(--border); border-radius: 4px; }

/* Section divider */
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-left: 3px solid var(--neon-green);
    padding-left: 0.5rem;
    margin: 0.5rem 0 0.8rem 0;
}

.pred-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}

.prob-bar-wrap { margin: 0.4rem 0; }
.prob-label {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 2px;
}
.prob-bar-outer {
    background: rgba(255,255,255,0.06);
    border-radius: 2px; height: 8px; overflow: hidden;
}
.prob-bar-inner { height: 100%; border-radius: 2px; }

.info-box {
    background: rgba(57,255,117,0.05);
    border: 1px solid rgba(57,255,117,0.2);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (identical logic to the notebook)
# ─────────────────────────────────────────────────────────────────────────────

def cap(series, lo=None, hi=None):
    if lo is not None:
        series = series.clip(lower=lo)
    if hi is not None:
        series = series.clip(upper=hi)
    return series


def expanding_team_mean(df, team_col, value_col, default):
    result = pd.Series(index=df.index, dtype=float)
    for team, grp in df.groupby(team_col, sort=False):
        vals = grp[value_col].copy()
        hist = vals.expanding().mean().shift(1)
        hist.iloc[0] = default
        result.loc[grp.index] = hist.values
    return result.fillna(default)


def build_features(df):
    df = df.copy()

    df["HomeElo"] = cap(df["HomeElo"], lo=1000, hi=3000)
    df["AwayElo"] = cap(df["AwayElo"], lo=1000, hi=3000)
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    df["EloRatio"] = df["HomeElo"] / df["AwayElo"].replace(0, np.nan).fillna(1500)

    for col in ["Form3Home", "Form5Home", "Form3Away", "Form5Away"]:
        df[col] = cap(df[col], lo=0, hi=15)

    df["Form3Diff"] = df["Form3Home"] - df["Form3Away"]
    df["Form5Diff"] = df["Form5Home"] - df["Form5Away"]
    df["Form5Ratio"] = (df["Form5Home"] + 0.5) / (df["Form5Away"] + 0.5)
    df["Form5Ratio"] = cap(df["Form5Ratio"], lo=0.1, hi=10)

    df["HomeXG"] = expanding_team_mean(df, "HomeTeam", "FTHome", default=1.5)
    df["AwayXG"] = expanding_team_mean(df, "AwayTeam", "FTAway", default=1.2)
    df["XGDiff"] = df["HomeXG"] - df["AwayXG"]

    df["HomeAttack"] = cap(df["HomeXG"] * df["Form5Home"] / 7.5, lo=0, hi=5)
    df["AwayAttack"] = cap(df["AwayXG"] * df["Form5Away"] / 7.5, lo=0, hi=5)

    for col in ["OddHome", "OddDraw", "OddAway"]:
        df[col] = np.where((df[col] > 1) & (df[col] < 1000), df[col], np.nan)

    df["ProbHome"] = (1 / df["OddHome"]).fillna(0.45)
    df["ProbDraw"] = (1 / df["OddDraw"]).fillna(0.28)
    df["ProbAway"] = (1 / df["OddAway"]).fillna(0.27)
    row_sum = df[["ProbHome", "ProbDraw", "ProbAway"]].sum(axis=1).replace(0, 1)
    df["ProbHome"] /= row_sum
    df["ProbDraw"] /= row_sum
    df["ProbAway"] /= row_sum
    df["OddsEdge"] = df["ProbHome"] - df["ProbAway"]

    df["HandiSize"] = df["HandiSize"].fillna(0)

    cluster_cols = ["C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB"]
    for col in cluster_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median()).fillna(0)

    df["MatchMonth"] = df["MatchDate"].dt.month
    df["MatchDayOfWeek"] = df["MatchDate"].dt.dayofweek

    feature_cols = [
        "HomeElo", "AwayElo", "EloDiff", "EloRatio",
        "Form3Diff", "Form5Diff", "Form5Ratio",
        "Form3Home", "Form5Home", "Form3Away", "Form5Away",
        "HomeXG", "AwayXG", "XGDiff",
        "HomeAttack", "AwayAttack",
        "ProbHome", "ProbDraw", "ProbAway", "OddsEdge",
        "HandiSize",
        "MatchMonth", "MatchDayOfWeek",
    ] + [c for c in cluster_cols if c in df.columns]

    df[feature_cols] = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return df, feature_cols


@st.cache_resource(show_spinner=False)
def train_models(df_bytes: bytes):
    """Train all models and return them + metadata. Cached by file content."""
    data = pd.read_csv(io.BytesIO(df_bytes), low_memory=False)
    data["MatchDate"] = pd.to_datetime(data["MatchDate"], errors="coerce")
    data = data.dropna(subset=["MatchDate", "FTResult"]).sort_values("MatchDate").reset_index(drop=True)
    data = data[data["FTResult"].isin(["H", "D", "A"])]

    processed, FEATURE_COLS = build_features(data)

    le = LabelEncoder()
    processed["Result_encoded"] = le.fit_transform(processed["FTResult"])

    X = processed[FEATURE_COLS]
    y_outcome = processed["Result_encoded"]
    y_home = processed["FTHome"]
    y_away = processed["FTAway"]
    y_total = y_home + y_away

    split_idx = int(len(processed) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_out_train, y_out_test = y_outcome.iloc[:split_idx], y_outcome.iloc[split_idx:]
    y_home_train, y_home_test = y_home.iloc[:split_idx], y_home.iloc[split_idx:]
    y_away_train, y_away_test = y_away.iloc[:split_idx], y_away.iloc[split_idx:]
    y_tot_train, y_tot_test = y_total.iloc[:split_idx], y_total.iloc[split_idx:]

    sw_train = compute_sample_weight("balanced", y_out_train)

    # Outcome models
    outcome_models = {
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000,
                C=0.5, random_state=42, n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ),
            method="sigmoid", cv=3,
        ),
    }

    outcome_results = {}
    for name, model in outcome_models.items():
        if name == "XGBoost":
            model.fit(X_train, y_out_train, sample_weight=sw_train)
        else:
            model.fit(X_train, y_out_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        acc = accuracy_score(y_out_test, y_pred)
        ll = log_loss(y_out_test, y_proba)
        brier = np.mean([
            brier_score_loss((y_out_test == c).astype(int), y_proba[:, i])
            for i, c in enumerate(np.unique(y_out_train))
        ])
        outcome_results[name] = dict(
            model=model, accuracy=acc, log_loss=ll, brier=brier,
            y_pred=y_pred, y_proba=y_proba,
        )

    best_outcome_name = min(outcome_results, key=lambda k: outcome_results[k]["log_loss"])

    # Goals models
    xgb_home = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="count:poisson", random_state=42, n_jobs=-1,
    )
    xgb_away = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="count:poisson", random_state=42, n_jobs=-1,
    )
    poi_home = Pipeline([("scaler", StandardScaler()), ("reg", PoissonRegressor(alpha=0.1, max_iter=500))])
    poi_away = Pipeline([("scaler", StandardScaler()), ("reg", PoissonRegressor(alpha=0.1, max_iter=500))])

    xgb_home.fit(X_train, y_home_train)
    xgb_away.fit(X_train, y_away_train)
    poi_home.fit(X_train, y_home_train)
    poi_away.fit(X_train, y_away_train)

    xgb_pred_h = xgb_home.predict(X_test).clip(0)
    xgb_pred_a = xgb_away.predict(X_test).clip(0)
    poi_pred_h = poi_home.predict(X_test).clip(0)
    poi_pred_a = poi_away.predict(X_test).clip(0)

    rmse_xgb = np.sqrt(mean_squared_error(y_tot_test, xgb_pred_h + xgb_pred_a))
    rmse_poi = np.sqrt(mean_squared_error(y_tot_test, poi_pred_h + poi_pred_a))
    best_goals = "XGBoost" if rmse_xgb <= rmse_poi else "Poisson"

    return {
        "data": data,
        "processed": processed,
        "FEATURE_COLS": FEATURE_COLS,
        "le": le,
        "X_train": X_train, "X_test": X_test,
        "y_out_train": y_out_train, "y_out_test": y_out_test,
        "y_home_test": y_home_test, "y_away_test": y_away_test,
        "y_tot_test": y_tot_test,
        "outcome_results": outcome_results,
        "best_outcome_name": best_outcome_name,
        "xgb_home": xgb_home, "xgb_away": xgb_away,
        "poi_home": poi_home, "poi_away": poi_away,
        "xgb_pred_h": xgb_pred_h, "xgb_pred_a": xgb_pred_a,
        "best_goals": best_goals,
        "rmse_xgb": rmse_xgb, "rmse_poi": rmse_poi,
        "split_idx": split_idx,
    }


def build_single_match_row(inputs: dict, processed_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Build a one-row feature DataFrame for a single match prediction."""
    home_team = inputs["home_team"]
    away_team = inputs["away_team"]
    match_date = inputs["match_date"]

    # Historical xG from processed data
    home_hist = processed_df[processed_df["HomeTeam"] == home_team]["FTHome"]
    away_hist = processed_df[processed_df["AwayTeam"] == away_team]["FTAway"]
    home_xg = home_hist.mean() if len(home_hist) else 1.5
    away_xg = away_hist.mean() if len(away_hist) else 1.2

    home_elo = inputs["home_elo"]
    away_elo = inputs["away_elo"]
    elo_diff = home_elo - away_elo
    elo_ratio = home_elo / max(away_elo, 1)

    f3h = inputs["form3home"]; f5h = inputs["form5home"]
    f3a = inputs["form3away"]; f5a = inputs["form5away"]

    xg_diff = home_xg - away_xg
    home_attack = min((home_xg * f5h / 7.5), 5)
    away_attack = min((away_xg * f5a / 7.5), 5)

    odd_h = inputs["odd_home"]; odd_d = inputs["odd_draw"]; odd_a = inputs["odd_away"]
    if odd_h > 1: ph = 1 / odd_h
    else: ph = 0.45
    if odd_d > 1: pd_ = 1 / odd_d
    else: pd_ = 0.28
    if odd_a > 1: pa = 1 / odd_a
    else: pa = 0.27
    rs = ph + pd_ + pa
    ph /= rs; pd_ /= rs; pa /= rs

    row = {
        "HomeElo": home_elo, "AwayElo": away_elo,
        "EloDiff": elo_diff, "EloRatio": elo_ratio,
        "Form3Diff": f3h - f3a, "Form5Diff": f5h - f5a,
        "Form5Ratio": (f5h + 0.5) / (f5a + 0.5),
        "Form3Home": f3h, "Form5Home": f5h,
        "Form3Away": f3a, "Form5Away": f5a,
        "HomeXG": home_xg, "AwayXG": away_xg, "XGDiff": xg_diff,
        "HomeAttack": home_attack, "AwayAttack": away_attack,
        "ProbHome": ph, "ProbDraw": pd_, "ProbAway": pa,
        "OddsEdge": ph - pa,
        "HandiSize": inputs.get("handi_size", 0),
        "MatchMonth": match_date.month,
        "MatchDayOfWeek": match_date.weekday(),
    }

    for c in ["C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB"]:
        if c in feature_cols:
            row[c] = 0.0

    df_row = pd.DataFrame([row])[feature_cols]
    return df_row


def prob_bar_html(label, value, color):
    pct = f"{value*100:.1f}%"
    return f"""
<div class="prob-bar-wrap">
  <div class="prob-label"><span>{label}</span><span>{pct}</span></div>
  <div class="prob-bar-outer">
    <div class="prob-bar-inner" style="width:{value*100:.1f}%; background:{color};"></div>
  </div>
</div>"""


def get_importances(model):
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        return model.calibrated_classifiers_[0].estimator.feature_importances_
    elif hasattr(model, "named_steps"):
        final = model.named_steps.get("clf", model[-1])
        if hasattr(final, "calibrated_classifiers_") and final.calibrated_classifiers_:
            return final.calibrated_classifiers_[0].estimator.feature_importances_
        return getattr(final, "feature_importances_", None)
    return getattr(model, "feature_importances_", None)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Match Predictor")
    st.markdown('<div class="section-title">01 — Data Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Matches.csv", type=["csv"])

    st.markdown('<div class="section-title">02 — About</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
Pre-match prediction engine using:<br>
• <b>Elo ratings</b> &amp; form points<br>
• <b>Historical xG</b> per team<br>
• <b>Betting odds</b> implied probabilities<br>
• <b>XGBoost</b> + Poisson regression<br><br>
No in-game data — zero leakage.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# Football Match Predictor")
st.markdown("#### Pre-match outcome & goals prediction · zero data leakage")

if uploaded_file is None:
    st.markdown("""
<div class="info-box" style="margin-top:2rem;">
    👆 Upload your <b>Matches.csv</b> in the sidebar to begin.<br><br>
    Required columns: <code>MatchDate, HomeTeam, AwayTeam, FTHome, FTAway, FTResult,
    HomeElo, AwayElo, Form3Home, Form5Home, Form3Away, Form5Away,
    OddHome, OddDraw, OddAway, HandiSize</code>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Train / load from cache ──────────────────────────────────────────────────
with st.spinner("Training models on your data…"):
    state = train_models(uploaded_file.getvalue())

data         = state["data"]
processed    = state["processed"]
FEATURE_COLS = state["FEATURE_COLS"]
le           = state["le"]
outcome_res  = state["outcome_results"]
best_name    = state["best_outcome_name"]
xgb_home     = state["xgb_home"]
xgb_away     = state["xgb_away"]

st.success(f"✓ Loaded {len(data):,} matches  ·  {data['MatchDate'].min().date()} → {data['MatchDate'].max().date()}", icon="✅")

teams = sorted(set(data["HomeTeam"].dropna().unique()) | set(data["AwayTeam"].dropna().unique()))

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_predict, tab_eval, tab_explore = st.tabs(["🔮 Predict", "📊 Model Eval", "🗃 Data Explorer"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.markdown("### Single Match Prediction")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Home Team</div>', unsafe_allow_html=True)
        home_team  = st.selectbox("Home Team", teams, key="home_team")
        home_elo   = st.number_input("Home Elo", value=1500, min_value=1000, max_value=3000, step=10)
        form3home  = st.slider("Form3 Home (pts last 3 games)", 0, 9, 5)
        form5home  = st.slider("Form5 Home (pts last 5 games)", 0, 15, 8)

    with col_right:
        st.markdown('<div class="section-title">Away Team</div>', unsafe_allow_html=True)
        away_team  = st.selectbox("Away Team", [t for t in teams if t != home_team], key="away_team")
        away_elo   = st.number_input("Away Elo", value=1450, min_value=1000, max_value=3000, step=10)
        form3away  = st.slider("Form3 Away (pts last 3 games)", 0, 9, 4)
        form5away  = st.slider("Form5 Away (pts last 5 games)", 0, 15, 6)

    st.markdown('<div class="section-title">Betting Odds & Match Info</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: odd_home = st.number_input("Odd Home", value=2.10, min_value=1.01, max_value=100.0, step=0.05)
    with c2: odd_draw = st.number_input("Odd Draw", value=3.40, min_value=1.01, max_value=100.0, step=0.05)
    with c3: odd_away = st.number_input("Odd Away", value=3.20, min_value=1.01, max_value=100.0, step=0.05)
    with c4: handi    = st.number_input("Handicap Size", value=0.0, min_value=-5.0, max_value=5.0, step=0.25)
    with c5: match_date = st.date_input("Match Date", value=pd.Timestamp.today())

    predict_btn = st.button("⚡  Generate Prediction", use_container_width=True)

    if predict_btn:
        inputs = dict(
            home_team=home_team, away_team=away_team,
            match_date=pd.Timestamp(match_date),
            home_elo=home_elo, away_elo=away_elo,
            form3home=form3home, form5home=form5home,
            form3away=form3away, form5away=form5away,
            odd_home=odd_home, odd_draw=odd_draw, odd_away=odd_away,
            handi_size=handi,
        )
        X_single = build_single_match_row(inputs, processed, FEATURE_COLS)
        best_model = outcome_res[best_name]["model"]

        proba = best_model.predict_proba(X_single)[0]
        pred_class = best_model.predict(X_single)[0]
        pred_label = le.inverse_transform([pred_class])[0]

        # Map A/D/H probabilities (le sorts alphabetically: A=0, D=1, H=2)
        class_order = list(le.classes_)   # e.g. ['A','D','H']
        prob_dict = {c: proba[i] for i, c in enumerate(class_order)}

        pred_goals_h = max(xgb_home.predict(X_single)[0], 0)
        pred_goals_a = max(xgb_away.predict(X_single)[0], 0)
        pred_total   = pred_goals_h + pred_goals_a

        label_map = {"H": f"Home Win ({home_team})", "D": "Draw", "A": f"Away Win ({away_team})"}
        result_label = label_map.get(pred_label, pred_label)

        # ── Output card ──────────────────────────────────────────────────────
        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Result", result_label)
        r2.metric("Home Win %",  f"{prob_dict.get('H', 0)*100:.1f}%")
        r3.metric("Draw %",      f"{prob_dict.get('D', 0)*100:.1f}%")
        r4.metric("Away Win %",  f"{prob_dict.get('A', 0)*100:.1f}%")

        g1, g2, g3 = st.columns(3)
        g1.metric("Predicted Home Goals", f"{pred_goals_h:.2f}")
        g2.metric("Predicted Away Goals", f"{pred_goals_a:.2f}")
        g3.metric("Predicted Total Goals", f"{pred_total:.2f}")

        colors = {"H": "#39ff75", "D": "#ffbf00", "A": "#ff3b5c"}
        bars_html = "".join([
            prob_bar_html(label_map.get(k, k), v, colors.get(k, "#888"))
            for k, v in sorted(prob_dict.items(), key=lambda x: -x[1])
        ])
        st.markdown(f'<div class="pred-card">{bars_html}</div>', unsafe_allow_html=True)

        over25 = 1 - np.exp(-pred_total) * sum(
            np.exp(-pred_total) * pred_total**k / np.math.factorial(k) for k in range(3)
        )
        st.markdown(f"""
<div class="pred-card" style="display:flex;gap:2rem;font-family:'IBM Plex Mono';font-size:0.82rem;color:var(--text-muted);">
  <span>🏆 Best model: <b style="color:var(--neon-green)">{best_name}</b></span>
  <span>📈 Over 2.5 goals est.: <b style="color:var(--neon-amber)">{over25*100:.1f}%</b></span>
  <span>📅 Match date: <b>{match_date}</b></span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.markdown("### Model Performance on Hold-out Test Set")
    st.markdown(f"*Time-based split — last 20% of matches used for evaluation*")

    # ── Metrics summary ──────────────────────────────────────────────────────
    cols = st.columns(len(outcome_res))
    for col, (name, res) in zip(cols, outcome_res.items()):
        short = name.split("(")[0].strip()
        star = " ★" if name == best_name else ""
        col.metric(f"{short}{star}", f"{res['accuracy']:.4f}", f"log-loss {res['log_loss']:.4f}")

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.patch.set_facecolor("#0a0e0c")
    for ax in axes.flat:
        ax.set_facecolor("#111a14")
        ax.tick_params(colors="#6b8a70")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3523")

    names_short = [n.split("(")[0].strip() for n in outcome_res]
    accs = [outcome_res[n]["accuracy"] for n in outcome_res]
    lls  = [outcome_res[n]["log_loss"] for n in outcome_res]
    palette = ["#39ff75", "#ffbf00", "#ff3b5c"]

    # Accuracy
    ax = axes[0, 0]
    bars = ax.bar(names_short, accs, color=palette)
    ax.set_ylim(0.40, max(accs) + 0.08)
    ax.set_title("Outcome Accuracy", color="#e8f0e9", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", color="#6b8a70")
    ax.tick_params(axis="x", rotation=15, labelcolor="#e8f0e9", labelsize=8)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", color="#e8f0e9", fontsize=9)

    # Log-loss
    ax = axes[0, 1]
    bars = ax.bar(names_short, lls, color=palette)
    ax.set_title("Log-loss (lower = better)", color="#e8f0e9", fontsize=11, fontweight="bold")
    ax.set_ylabel("Log-loss", color="#6b8a70")
    ax.tick_params(axis="x", rotation=15, labelcolor="#e8f0e9", labelsize=8)
    for b, v in zip(bars, lls):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", color="#e8f0e9", fontsize=9)

    # Confusion matrix
    ax = axes[0, 2]
    best_res = outcome_res[best_name]
    cm = confusion_matrix(state["y_out_test"], best_res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d",
                cmap=sns.light_palette("#39ff75", as_cmap=True),
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Confusion — {best_name.split('(')[0].strip()}", color="#e8f0e9", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", color="#6b8a70")
    ax.set_ylabel("Actual", color="#6b8a70")
    ax.tick_params(colors="#e8f0e9")

    # Actual vs Predicted goals
    ax = axes[1, 0]
    xgb_pred_h = state["xgb_pred_h"]; xgb_pred_a = state["xgb_pred_a"]
    pred_total_xgb = xgb_pred_h + xgb_pred_a
    y_tot_test = state["y_tot_test"]
    ax.scatter(y_tot_test, pred_total_xgb, alpha=0.25, s=5, color="#39ff75")
    lims = [0, max(y_tot_test.max(), pred_total_xgb.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    r2 = r2_score(y_tot_test, pred_total_xgb)
    ax.set_title(f"XGBoost Goals  (R²={r2:.3f})", color="#e8f0e9", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual", color="#6b8a70"); ax.set_ylabel("Predicted", color="#6b8a70")

    # Error distribution
    ax = axes[1, 1]
    errors = pred_total_xgb - y_tot_test
    ax.hist(errors, bins=25, color="#39ff75", edgecolor="#0a0e0c", alpha=0.85)
    ax.axvline(0, color="#ff3b5c", linestyle="--", linewidth=1.2)
    ax.set_title("Goals Prediction Error", color="#e8f0e9", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted − Actual", color="#6b8a70")
    ax.set_ylabel("Count", color="#6b8a70")

    # Feature importances
    ax = axes[1, 2]
    importances = get_importances(best_res["model"])
    if importances is not None:
        fi = pd.Series(importances, index=FEATURE_COLS).nlargest(12).sort_values()
        colors_fi = ["#39ff75" if v == fi.max() else "#1D9E75" for v in fi.values]
        fi.plot.barh(ax=ax, color=colors_fi)
        ax.set_title("Top 12 Feature Importances", color="#e8f0e9", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance", color="#6b8a70")
        ax.tick_params(axis="y", labelcolor="#e8f0e9", labelsize=8)
    else:
        ax.text(0.5, 0.5, "Not available\n(Logistic Regression)", transform=ax.transAxes,
                ha="center", va="center", color="#6b8a70")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Goals RMSE comparison
    st.markdown("---")
    st.markdown("**Goals Regression (RMSE on test set)**")
    g1, g2 = st.columns(2)
    g1.metric("XGBoost Poisson RMSE", f"{state['rmse_xgb']:.4f}",
              delta="Best" if state["best_goals"] == "XGBoost" else None)
    g2.metric("Poisson GLM RMSE", f"{state['rmse_poi']:.4f}",
              delta="Best" if state["best_goals"] == "Poisson" else None)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

with tab_explore:
    st.markdown("### Data Explorer")

    c1, c2 = st.columns(2)
    with c1:
        filter_team = st.selectbox("Filter by team", ["All"] + teams)
    with c2:
        filter_result = st.selectbox("Filter by FT result", ["All", "H", "D", "A"])

    display_df = data[["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult",
                        "HomeElo", "AwayElo", "OddHome", "OddDraw", "OddAway"]].copy()

    if filter_team != "All":
        display_df = display_df[(display_df["HomeTeam"] == filter_team) | (display_df["AwayTeam"] == filter_team)]
    if filter_result != "All":
        display_df = display_df[display_df["FTResult"] == filter_result]

    st.markdown(f"*Showing {len(display_df):,} matches*")
    st.dataframe(display_df.tail(200).sort_values("MatchDate", ascending=False), use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("**Result distribution**")
    dist = data["FTResult"].value_counts().rename({"H": "Home Win", "D": "Draw", "A": "Away Win"})
    fig2, ax2 = plt.subplots(figsize=(5, 2.5))
    fig2.patch.set_facecolor("#0a0e0c")
    ax2.set_facecolor("#111a14")
    ax2.bar(dist.index, dist.values, color=["#39ff75", "#ffbf00", "#ff3b5c"])
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e3523")
    ax2.tick_params(colors="#e8f0e9")
    ax2.set_ylabel("Count", color="#6b8a70")
    st.pyplot(fig2, use_container_width=False)
    plt.close(fig2)
