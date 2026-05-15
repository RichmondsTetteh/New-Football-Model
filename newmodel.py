# app.py
# Streamlit app for Football Match Outcome & Goals Prediction
# -----------------------------------------------------------
# This app uses the trained models saved from your notebook:
# - outcome_model.joblib
# - goals_home_model.joblib
# - goals_away_model.joblib
# - label_encoder.joblib
# - feature_cols.joblib
#
# Run:
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import re
import requests
import zipfile
import warnings
from math import exp, factorial
from datetime import datetime
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------

st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Match Prediction App")
st.markdown("""
Predict:
- Match Outcome (Home / Draw / Away)
- Expected Goals
- Total Goals
- Scoreline Probabilities

Powered by:
- XGBoost
- Poisson Goal Modelling
- Historical Form & Elo Features
""")

st.info(
    "⏳ **First-time tip:** The first click on **Analyze Teams** may take a while — the app needs to "
    "download match files from football-data.co.uk. Once cached, subsequent analyses will be much faster."
)

# -----------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------

@st.cache_resource
def load_models():
    outcome_model = joblib.load("outcome_model.joblib")
    goals_home_model = joblib.load("goals_home_model.joblib")
    goals_away_model = joblib.load("goals_away_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    feature_cols = joblib.load("feature_cols.joblib")

    return (
        outcome_model,
        goals_home_model,
        goals_away_model,
        label_encoder,
        feature_cols
    )

(
    outcome_model,
    goals_home_model,
    goals_away_model,
    label_encoder,
    FEATURE_COLS
) = load_models()

# -----------------------------------------------------------
# HELPER FUNCTIONS (original app logic — unchanged)
# -----------------------------------------------------------

def poisson_prob(k, lam):
    return (exp(-lam) * (lam ** k)) / factorial(k)

def score_matrix(home_xg, away_xg, max_goals=5):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = poisson_prob(i, home_xg) * poisson_prob(j, away_xg)
    return matrix

def outcome_from_score_matrix(matrix):
    home_win = np.tril(matrix, -1).sum()
    draw     = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    return home_win, draw, away_win

# -----------------------------------------------------------
# AUTO-POPULATION: Football-Data scraping helpers
# -----------------------------------------------------------

DEFAULT_ELO = 1500.0

@st.cache_data(show_spinner=False)
def download_and_merge_football_data():
    """Downloads and merges football data from football-data.co.uk."""
    base_url      = "https://www.football-data.co.uk/"
    data_page_url = base_url + "data.php"

    all_dfs         = []
    unique_csv_urls = set()
    visited_pages   = set()

    try:
        resp = requests.get(data_page_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        html_pages = {data_page_url}
        for link in soup.find_all("a", href=True):
            href    = link["href"]
            abs_url = urljoin(base_url, href)
            if abs_url.endswith((".htm", ".php")) and abs_url.startswith(base_url) and abs_url != data_page_url:
                html_pages.add(abs_url)
            elif abs_url.endswith(".csv") and abs_url.startswith(base_url):
                unique_csv_urls.add(abs_url)
            elif abs_url.endswith(".zip") and abs_url.startswith(base_url):
                if "data.zip" in abs_url or re.search(r"\b[A-Z0-9]{1,2}\.zip$", abs_url):
                    try:
                        zr = requests.get(abs_url, stream=True, timeout=15)
                        zr.raise_for_status()
                        with zipfile.ZipFile(io.BytesIO(zr.content)) as z:
                            for fn in z.namelist():
                                if fn.endswith(".csv") and not fn.startswith("__MACOSX"):
                                    try:
                                        with z.open(fn) as cf:
                                            df_s = pd.read_csv(cf, encoding="ISO-8859-1", on_bad_lines="skip")
                                            df_s["Source_File"] = fn
                                            all_dfs.append(df_s)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

        for page_url in list(html_pages):
            if page_url in visited_pages:
                continue
            visited_pages.add(page_url)
            try:
                pr = requests.get(page_url, timeout=15)
                pr.raise_for_status()
                ps = BeautifulSoup(pr.content, "html.parser")
                for link in ps.find_all("a", href=True):
                    href = link["href"]
                    if href.endswith(".csv"):
                        full_url = urljoin(page_url, href)
                        if full_url.startswith(base_url):
                            unique_csv_urls.add(full_url)
                    elif href.endswith(".zip"):
                        if "data.zip" in href or re.search(r"\b[A-Z0-9]{1,2}\.zip$", href):
                            fzu = urljoin(page_url, href)
                            if fzu.startswith(base_url):
                                try:
                                    zr = requests.get(fzu, stream=True, timeout=15)
                                    zr.raise_for_status()
                                    with zipfile.ZipFile(io.BytesIO(zr.content)) as z:
                                        for fn in z.namelist():
                                            if fn.endswith(".csv") and not fn.startswith("__MACOSX"):
                                                try:
                                                    with z.open(fn) as cf:
                                                        df_s = pd.read_csv(cf, encoding="ISO-8859-1", on_bad_lines="skip")
                                                        df_s["Source_File"] = fn
                                                        all_dfs.append(df_s)
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
            except Exception:
                pass

        for csv_url in unique_csv_urls:
            try:
                df_s = pd.read_csv(csv_url, encoding="ISO-8859-1", on_bad_lines="skip")
                df_s["Source_File"] = csv_url.split("/")[-1]
                all_dfs.append(df_s)
            except Exception:
                pass

        if not all_dfs:
            return pd.DataFrame()

        df_full = pd.concat(all_dfs, ignore_index=True, sort=False)
        df_full["Date"] = pd.to_datetime(df_full["Date"], format="%d/%m/%Y", errors="coerce")
        df_full.dropna(subset=["Date"], inplace=True)
        df_full.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], inplace=True)
        df_full = df_full.sort_values(by="Date").reset_index(drop=True)
        for col in ["HS", "AS", "HST", "AST"]:
            if col in df_full.columns:
                df_full[col] = df_full[col].fillna(0)
        return df_full

    except Exception as e:
        st.error(f"Error loading football data: {e}")
        return pd.DataFrame()


def calculate_team_form(team_name, df_data, num_matches):
    team_matches = df_data[
        (df_data["HomeTeam"] == team_name) | (df_data["AwayTeam"] == team_name)
    ].copy()
    if team_matches.empty:
        return 0, "No matches found", pd.DataFrame()
    recent = team_matches.sort_values("Date", ascending=False).head(num_matches)
    points, results = 0, []
    for _, row in recent.iterrows():
        if row["HomeTeam"] == team_name:
            if row["FTR"] == "H": points += 3; results.append("W")
            elif row["FTR"] == "D": points += 1; results.append("D")
            else: results.append("L")
        else:
            if row["FTR"] == "A": points += 3; results.append("W")
            elif row["FTR"] == "D": points += 1; results.append("D")
            else: results.append("L")
    results.reverse()
    return points, " ".join(results), recent


def calculate_average_stats(team_name, recent_matches_df):
    if recent_matches_df.empty:
        return {"Shots For": 0, "Shots Against": 0, "Shots On Target For": 0, "Shots On Target Against": 0}
    sf = sa = stf = sta = 0
    for _, row in recent_matches_df.iterrows():
        if row["HomeTeam"] == team_name:
            sf += row.get("HS", 0); sa += row.get("AS", 0)
            stf += row.get("HST", 0); sta += row.get("AST", 0)
        else:
            sf += row.get("AS", 0); sa += row.get("HS", 0)
            stf += row.get("AST", 0); sta += row.get("HST", 0)
    n = len(recent_matches_df)
    return {
        "Shots For": sf / n, "Shots Against": sa / n,
        "Shots On Target For": stf / n, "Shots On Target Against": sta / n,
    }


def calculate_season_average_stats(team_name, df_data):
    tm = df_data[
        (df_data["HomeTeam"] == team_name) | (df_data["AwayTeam"] == team_name)
    ].copy()
    if tm.empty:
        return {"Shots For": 0, "Shots Against": 0, "Shots On Target For": 0, "Shots On Target Against": 0}
    sf = sa = stf = sta = 0
    for _, row in tm.iterrows():
        if row["HomeTeam"] == team_name:
            sf += row.get("HS", 0); sa += row.get("AS", 0)
            stf += row.get("HST", 0); sta += row.get("AST", 0)
        else:
            sf += row.get("AS", 0); sa += row.get("HS", 0)
            stf += row.get("AST", 0); sta += row.get("HST", 0)
    n = len(tm)
    return {
        "Shots For": sf / n, "Shots Against": sa / n,
        "Shots On Target For": stf / n, "Shots On Target Against": sta / n,
    }


@st.cache_data(show_spinner="Fetching ELO ratings...", ttl=1800)
def load_elo_ratings():
    for delta in [0, 1]:
        try:
            date_str = (datetime.now() - pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
            resp = requests.get(f"http://api.clubelo.com/{date_str}", timeout=15)
            resp.raise_for_status()
            df_elo = pd.read_csv(io.StringIO(resp.text))
            df_elo["Clean_Club"] = df_elo["Club"].astype(str).str.strip()
            return df_elo
        except Exception:
            continue
    st.warning("Could not load ELO ratings. Defaulting to 1500.")
    return pd.DataFrame(columns=["Club", "Elo", "Clean_Club"])


def get_elo_for_team(team_name, df_elo):
    if df_elo.empty or not team_name.strip():
        return DEFAULT_ELO, None
    name_lower = team_name.strip().lower()
    mask    = df_elo["Clean_Club"].str.lower().str.contains(name_lower, na=False, regex=False)
    matches = df_elo[mask]
    if matches.empty:
        return DEFAULT_ELO, f'No ELO match for "{team_name}". Using default {DEFAULT_ELO}.'
    exact = matches[matches["Clean_Club"].str.lower() == name_lower]
    if len(exact) == 1:
        return float(exact["Elo"].iloc[0]), None
    if len(matches) > 1:
        candidates = ", ".join(matches["Clean_Club"].tolist()[:5])
        return DEFAULT_ELO, f'"{team_name}" matched multiple ({candidates}). Using default.'
    return float(matches["Elo"].iloc[0]), None

# -----------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------

DEFAULT_INPUTS = {
    "home_elo": DEFAULT_ELO,
    "away_elo": DEFAULT_ELO,
    "form3_home": 7,
    "form5_home": 10,
    "form3_away": 6,
    "form5_away": 9,
    "home_xg": 1.8,
    "away_xg": 1.2,
    "odd_home": 2.00,
    "odd_draw": 3.50,
    "odd_away": 3.80,
    "handi_size": 0.0,
    "c_lth": 0.0,
    "c_lta": 0.0,
    "c_vhd": 0.0,
    "c_vad": 0.0,
    "c_htb": 0.0,
    "c_phb": 0.0,
}

for k, v in DEFAULT_INPUTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "fd_analysis_done" not in st.session_state:
    st.session_state.fd_analysis_done = False
if "fd_results" not in st.session_state:
    st.session_state.fd_results = {}

# -----------------------------------------------------------
# ELO LOADING
# -----------------------------------------------------------

df_elo = load_elo_ratings()

# -----------------------------------------------------------
# TEAM INPUTS & ANALYZE BUTTON
# -----------------------------------------------------------

col_t1, col_t2 = st.columns(2)
with col_t1:
    home_team = st.text_input("🏠 Home Team", "Manchester City")
with col_t2:
    away_team = st.text_input("🏟️ Away Team", "Liverpool")

home_elo_scraped, home_warn = get_elo_for_team(home_team, df_elo)
away_elo_scraped, away_warn = get_elo_for_team(away_team, df_elo)

if home_warn:
    st.warning(f"🏠 {home_warn}")
if away_warn:
    st.warning(f"🏟️ {away_warn}")

# Flag stale analysis when teams change
teams_changed = False
if "prev_home" not in st.session_state or st.session_state.prev_home != home_team:
    st.session_state.home_elo = home_elo_scraped
    st.session_state.prev_home = home_team
    if st.session_state.fd_analysis_done:
        st.session_state.fd_analysis_done = False
        teams_changed = True

if "prev_away" not in st.session_state or st.session_state.prev_away != away_team:
    st.session_state.away_elo = away_elo_scraped
    st.session_state.prev_away = away_team
    if st.session_state.fd_analysis_done:
        st.session_state.fd_analysis_done = False
        teams_changed = True

btn_col, status_col = st.columns([2, 3])
with btn_col:
    analyze_clicked = st.button(
        "📊 Analyze Teams",
        type="primary",
        use_container_width=True,
        help="Loads historical data from football-data.co.uk and calculates form + shot stats",
    )
with status_col:
    if teams_changed:
        st.warning("⚠️ Team names changed — click **Analyze Teams** to refresh stats.")
    elif st.session_state.fd_analysis_done:
        st.success("✅ Historical stats loaded — inputs updated. Ready to predict.")

# -----------------------------------------------------------
# ANALYZE TEAMS
# -----------------------------------------------------------

if analyze_clicked:
    with st.spinner("Loading and merging football data from football-data.co.uk..."):
        df_full = download_and_merge_football_data()

    if df_full.empty:
        st.error("Could not load football data. Check internet connection or data source.")
    else:
        pts3_home, form3_str_home, matches3_home = calculate_team_form(home_team, df_full, 3)
        pts5_home, form5_str_home, matches5_home = calculate_team_form(home_team, df_full, 5)
        pts3_away, form3_str_away, matches3_away = calculate_team_form(away_team, df_full, 3)
        pts5_away, form5_str_away, matches5_away = calculate_team_form(away_team, df_full, 5)

        stats5_home    = calculate_average_stats(home_team, matches5_home)
        stats5_away    = calculate_average_stats(away_team, matches5_away)
        season_home    = calculate_season_average_stats(home_team, df_full)
        season_away    = calculate_season_average_stats(away_team, df_full)

        w_season, w_last5 = 0.70, 0.30
        weighted_home = {k: (w_season * season_home[k]) + (w_last5 * stats5_home[k]) for k in stats5_home}
        weighted_away = {k: (w_season * season_away[k]) + (w_last5 * stats5_away[k]) for k in stats5_away}

        # Derive xG estimate from shots on target
        home_xg_est = round(weighted_home["Shots On Target For"] / 3.0, 2)
        away_xg_est = round(weighted_away["Shots On Target For"] / 3.0, 2)

        # Push values into session state so sidebar widgets pick them up
        st.session_state.form3_home = pts3_home
        st.session_state.form5_home = pts5_home
        st.session_state.form3_away = pts3_away
        st.session_state.form5_away = pts5_away
        st.session_state.home_xg    = max(0.1, home_xg_est)
        st.session_state.away_xg    = max(0.1, away_xg_est)

        st.session_state.fd_results = {
            "home_team": home_team, "away_team": away_team,
            "pts3_home": pts3_home, "form3_str_home": form3_str_home, "matches3_home": matches3_home,
            "pts5_home": pts5_home, "form5_str_home": form5_str_home, "matches5_home": matches5_home,
            "pts3_away": pts3_away, "form3_str_away": form3_str_away, "matches3_away": matches3_away,
            "pts5_away": pts5_away, "form5_str_away": form5_str_away, "matches5_away": matches5_away,
            "weighted_home": weighted_home, "weighted_away": weighted_away,
        }
        st.session_state.fd_analysis_done = True
        st.rerun()

# -----------------------------------------------------------
# DISPLAY HISTORICAL ANALYSIS
# -----------------------------------------------------------

if st.session_state.fd_analysis_done and st.session_state.fd_results:
    r = st.session_state.fd_results
    if r.get("home_team") == home_team and r.get("away_team") == away_team:
        st.markdown("---")
        st.header(f"📋 Historical Analysis: {home_team} vs {away_team}")
        left, right = st.columns(2)

        with left:
            st.subheader(f"🏠 {home_team}")
            st.write(f"**Last 3:** {r['form3_str_home']} — Points: **{r['pts3_home']}**")
            if not r["matches3_home"].empty:
                cols_show = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] if c in r["matches3_home"].columns]
                st.dataframe(r["matches3_home"][cols_show], use_container_width=True)
            st.write(f"**Last 5:** {r['form5_str_home']} — Points: **{r['pts5_home']}**")
            if not r["matches5_home"].empty:
                cols_show = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] if c in r["matches5_home"].columns]
                st.dataframe(r["matches5_home"][cols_show], use_container_width=True)
            st.markdown("**Weighted Shot Stats (70% season / 30% last 5):**")
            for k, v in r["weighted_home"].items():
                st.write(f"- {k}: **{v:.2f}**")

        with right:
            st.subheader(f"🏟️ {away_team}")
            st.write(f"**Last 3:** {r['form3_str_away']} — Points: **{r['pts3_away']}**")
            if not r["matches3_away"].empty:
                cols_show = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] if c in r["matches3_away"].columns]
                st.dataframe(r["matches3_away"][cols_show], use_container_width=True)
            st.write(f"**Last 5:** {r['form5_str_away']} — Points: **{r['pts5_away']}**")
            if not r["matches5_away"].empty:
                cols_show = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] if c in r["matches5_away"].columns]
                st.dataframe(r["matches5_away"][cols_show], use_container_width=True)
            st.markdown("**Weighted Shot Stats (70% season / 30% last 5):**")
            for k, v in r["weighted_away"].items():
                st.write(f"- {k}: **{v:.2f}**")

        st.info(
            "✅ Form points and xG have been automatically applied to the sidebar inputs. "
            "Review and adjust as needed, then click **🔮 Predict Match**."
        )

# -----------------------------------------------------------
# SIDEBAR INPUTS  (reads from session_state so Analyze auto-fills them)
# -----------------------------------------------------------

st.sidebar.header("📊 Match Inputs")

home_team_display = st.sidebar.text_input("Home Team", home_team, key="sidebar_home", disabled=True)
away_team_display = st.sidebar.text_input("Away Team", away_team, key="sidebar_away", disabled=True)

st.sidebar.subheader("Elo Ratings")
home_elo = st.sidebar.number_input(
    "Home Elo", min_value=1000, max_value=3000,
    value=int(st.session_state.home_elo), key="input_home_elo"
)
away_elo = st.sidebar.number_input(
    "Away Elo", min_value=1000, max_value=3000,
    value=int(st.session_state.away_elo), key="input_away_elo"
)

st.sidebar.subheader("Recent Form")
form3_home = st.sidebar.slider("Form3 Home", 0, 15, int(st.session_state.form3_home), key="input_form3_home")
form5_home = st.sidebar.slider("Form5 Home", 0, 15, int(st.session_state.form5_home), key="input_form5_home")
form3_away = st.sidebar.slider("Form3 Away", 0, 15, int(st.session_state.form3_away), key="input_form3_away")
form5_away = st.sidebar.slider("Form5 Away", 0, 15, int(st.session_state.form5_away), key="input_form5_away")

st.sidebar.subheader("Expected Goals")
home_xg = st.sidebar.slider("Historical Home XG", 0.0, 5.0, float(st.session_state.home_xg), 0.1, key="input_home_xg")
away_xg = st.sidebar.slider("Historical Away XG", 0.0, 5.0, float(st.session_state.away_xg), 0.1, key="input_away_xg")

st.sidebar.subheader("Betting Odds")
odd_home = st.sidebar.number_input("Home Win Odds", 1.01, 50.0, float(st.session_state.odd_home), key="input_odd_home")
odd_draw = st.sidebar.number_input("Draw Odds",      1.01, 50.0, float(st.session_state.odd_draw), key="input_odd_draw")
odd_away = st.sidebar.number_input("Away Win Odds",  1.01, 50.0, float(st.session_state.odd_away), key="input_odd_away")

st.sidebar.subheader("Handicap")
handi_size = st.sidebar.slider("Handicap Size", -5.0, 5.0, float(st.session_state.handi_size), 0.25, key="input_handi")

st.sidebar.subheader("Cluster Features")
c_lth = st.sidebar.number_input("C_LTH", value=float(st.session_state.c_lth), key="input_c_lth")
c_lta = st.sidebar.number_input("C_LTA", value=float(st.session_state.c_lta), key="input_c_lta")
c_vhd = st.sidebar.number_input("C_VHD", value=float(st.session_state.c_vhd), key="input_c_vhd")
c_vad = st.sidebar.number_input("C_VAD", value=float(st.session_state.c_vad), key="input_c_vad")
c_htb = st.sidebar.number_input("C_HTB", value=float(st.session_state.c_htb), key="input_c_htb")
c_phb = st.sidebar.number_input("C_PHB", value=float(st.session_state.c_phb), key="input_c_phb")

# -----------------------------------------------------------
# FEATURE ENGINEERING  (original logic — unchanged)
# -----------------------------------------------------------

elo_diff   = home_elo - away_elo
elo_ratio  = home_elo / away_elo

form3_diff  = form3_home - form3_away
form5_diff  = form5_home - form5_away
form5_ratio = (form5_home + 0.5) / (form5_away + 0.5)

xg_diff = home_xg - away_xg

home_attack = home_xg * form5_home / 7.5
away_attack = away_xg * form5_away / 7.5

prob_home = 1 / odd_home
prob_draw = 1 / odd_draw
prob_away = 1 / odd_away

prob_sum   = prob_home + prob_draw + prob_away
prob_home /= prob_sum
prob_draw /= prob_sum
prob_away /= prob_sum

odds_edge = prob_home - prob_away

month        = 5
day_of_week  = 4

input_dict = {
    "HomeElo":       home_elo,
    "AwayElo":       away_elo,
    "EloDiff":       elo_diff,
    "EloRatio":      elo_ratio,

    "Form3Diff":     form3_diff,
    "Form5Diff":     form5_diff,
    "Form5Ratio":    form5_ratio,

    "Form3Home":     form3_home,
    "Form5Home":     form5_home,
    "Form3Away":     form3_away,
    "Form5Away":     form5_away,

    "HomeXG":        home_xg,
    "AwayXG":        away_xg,
    "XGDiff":        xg_diff,

    "HomeAttack":    home_attack,
    "AwayAttack":    away_attack,

    "ProbHome":      prob_home,
    "ProbDraw":      prob_draw,
    "ProbAway":      prob_away,
    "OddsEdge":      odds_edge,

    "HandiSize":     handi_size,

    "MatchMonth":    month,
    "MatchDayOfWeek": day_of_week,

    "C_LTH":         c_lth,
    "C_LTA":         c_lta,
    "C_VHD":         c_vhd,
    "C_VAD":         c_vad,
    "C_HTB":         c_htb,
    "C_PHB":         c_phb,
}

# Ensure all feature columns exist
for col in FEATURE_COLS:
    if col not in input_dict:
        input_dict[col] = 0

X_input = pd.DataFrame([input_dict])[FEATURE_COLS]

# -----------------------------------------------------------
# PREDICTION  (original logic — unchanged)
# -----------------------------------------------------------

if st.sidebar.button("🔮 Predict Match", type="primary", use_container_width=True):

    # Outcome prediction
    outcome_probs = outcome_model.predict_proba(X_input)[0]
    outcome_pred  = outcome_model.predict(X_input)[0]
    outcome_label = label_encoder.inverse_transform([outcome_pred])[0]

    # Goals prediction
    pred_home_goals = float(goals_home_model.predict(X_input)[0])
    pred_away_goals = float(goals_away_model.predict(X_input)[0])
    pred_home_goals = max(0, pred_home_goals)
    pred_away_goals = max(0, pred_away_goals)
    total_goals     = pred_home_goals + pred_away_goals

    # Score probabilities
    matrix = score_matrix(pred_home_goals, pred_away_goals)
    home_win_prob, draw_prob, away_win_prob = outcome_from_score_matrix(matrix)

    # -------------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------------

    st.header(f"🏟️ {home_team} vs {away_team}")
    if st.session_state.fd_analysis_done:
        st.caption("ℹ️ Stats sourced from football-data.co.uk historical data.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏠 Expected Home Goals", f"{pred_home_goals:.2f}")
    with col2:
        st.metric("⚽ Expected Total Goals", f"{total_goals:.2f}")
    with col3:
        st.metric("🚗 Expected Away Goals", f"{pred_away_goals:.2f}")

    st.divider()

    # Outcome Probabilities
    st.subheader("📈 Match Outcome Probabilities")
    outcome_df = pd.DataFrame({
        "Outcome":     ["Home Win", "Draw", "Away Win"],
        "Probability": [outcome_probs[0], outcome_probs[1], outcome_probs[2]],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=outcome_df["Outcome"], y=outcome_df["Probability"]))
    fig.update_layout(height=400, yaxis_title="Probability", xaxis_title="Outcome")
    st.plotly_chart(fig, use_container_width=True)

    # Final Prediction
    st.subheader("🎯 Final Prediction")
    if outcome_label == "H":
        final_result = f"{home_team} WIN"
    elif outcome_label == "D":
        final_result = "DRAW"
    else:
        final_result = f"{away_team} WIN"

    st.success(final_result)
    st.write(f"""
    ### Model Summary
    - Home Win Probability: **{outcome_probs[0]:.2%}**
    - Draw Probability: **{outcome_probs[1]:.2%}**
    - Away Win Probability: **{outcome_probs[2]:.2%}**
    """)

    # Score Matrix
    st.subheader("📋 Most Likely Scorelines")
    score_probs = []
    for i in range(6):
        for j in range(6):
            score_probs.append({"Score": f"{i} - {j}", "Probability": matrix[i, j]})
    score_df = pd.DataFrame(score_probs).sort_values(by="Probability", ascending=False).head(10)
    score_df["Probability"] = (score_df["Probability"] * 100).round(2)
    st.dataframe(score_df, use_container_width=True)

    # Poisson Model
    st.subheader("📊 Poisson Score Model")
    poisson_df = pd.DataFrame({
        "Result":      ["Home Win", "Draw", "Away Win"],
        "Probability": [round(home_win_prob * 100, 2), round(draw_prob * 100, 2), round(away_win_prob * 100, 2)],
    })
    st.dataframe(poisson_df, use_container_width=True)

    # Engineered Features
    with st.expander("🧠 Engineered Features Used"):
        feature_view = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Value":   [input_dict[c] for c in FEATURE_COLS],
        })
        st.dataframe(feature_view, use_container_width=True)

else:
    if not st.session_state.fd_analysis_done:
        st.info("👈 Click **Analyze Teams** to load historical stats, then click **🔮 Predict Match** in the sidebar.")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------

st.divider()
st.markdown(
    """
### 📦 Required Files

Place these files in the same folder as `app.py`:

- outcome_model.joblib
- goals_home_model.joblib
- goals_away_model.joblib
- label_encoder.joblib
- feature_cols.joblib

### ▶️ Run App

```
streamlit run app.py
```
"""
)
