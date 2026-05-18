import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import exp, factorial
import plotly.graph_objects as go
import requests
import io
import zipfile
import warnings
from datetime import datetime
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

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

# =============================================================================
# CONSTANTS
# =============================================================================
# Model filenames
OUTCOME_MODEL_FILENAME = 'outcome_model.joblib'
GOALS_HOME_MODEL_FILENAME = 'goals_home_model.joblib'
GOALS_AWAY_MODEL_FILENAME = 'goals_away_model.joblib'
LABEL_ENCODER_FILENAME = 'label_encoder.joblib'
FEATURE_COLS_FILENAME = 'feature_cols.joblib'

DEFAULT_ELO = 1500.0

# DEFAULTS for input features (used for initializing session state and reset)
APP_DEFAULTS = {
    'HomeElo': 1800.0,
    'AwayElo': 1750.0,
    'Form3Home': 7,
    'Form5Home': 10,
    'Form3Away': 6,
    'Form5Away': 9,
    'HomeExpectedGoals': 1.8,
    'AwayExpectedGoals': 1.2,
    'OddHome': 2.0,
    'OddDraw': 3.5,
    'OddAway': 3.8,
    'HandiSize': 0.0,
    'C_LTH': 0.0, 'C_LTA': 0.0, 'C_VHD': 0.0, 'C_VAD': 0.0, 'C_HTB': 0.0, 'C_PHB': 0.0
}

# -----------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------

@st.cache_resource
def load_models():
    try:
        outcome_model = joblib.load(OUTCOME_MODEL_FILENAME)
        goals_home_model = joblib.load(GOALS_HOME_MODEL_FILENAME)
        goals_away_model = joblib.load(GOALS_AWAY_MODEL_FILENAME)
        label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        feature_cols = joblib.load(FEATURE_COLS_FILENAME)
        return (
            outcome_model,
            goals_home_model,
            goals_away_model,
            label_encoder,
            feature_cols
        )
    except Exception as e:
        st.error(f"Failed to load models. Ensure '{OUTCOME_MODEL_FILENAME}', etc. are in the same directory. Error: {e}")
        st.stop()

(
    outcome_model,
    goals_home_model,
    goals_away_model,
    label_encoder,
    FEATURE_COLS
) = load_models()

# -----------------------------------------------------------
# HELPER FUNCTIONS (for prediction)
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
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()

    return home_win, draw, away_win

# =============================================================================
# FOOTBALL-DATA.CO.UK DATA LOADING (from second app)
# =============================================================================
@st.cache_data(show_spinner="Fetching ELO ratings...", ttl=3600) # Cache for 1 hour
def download_and_merge_football_data():
    """Downloads and merges football data from football-data.co.uk."""
    base_url = "https://www.football-data.co.uk/"
    data_page_url = base_url + "data.php"

    all_dfs = []
    unique_csv_urls = set()
    visited_html_pages = set()

    try:
        response = requests.get(data_page_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        html_pages_to_scrape = {data_page_url}

        # Find all direct CSV and ZIP links on the main page
        for link in soup.find_all('a', href=True):
            href = link['href']
            abs_url = urljoin(base_url, href)
            if abs_url.endswith(('.htm', '.php')) and abs_url.startswith(base_url) and abs_url != data_page_url:
                html_pages_to_scrape.add(abs_url)
            elif abs_url.endswith('.csv') and abs_url.startswith(base_url):
                unique_csv_urls.add(abs_url)
            elif abs_url.endswith('.zip') and abs_url.startswith(base_url):
                if 'data.zip' in abs_url or re.search(r'\b[A-Z0-9]{1,2}\.zip$', abs_url):
                    try:
                        zr = requests.get(abs_url, stream=True, timeout=15)
                        zr.raise_for_status()
                        with zipfile.ZipFile(io.BytesIO(zr.content)) as z:
                            for fn in z.namelist():
                                if fn.endswith('.csv') and not fn.startswith('__MACOSX'):
                                    try:
                                        with z.open(fn) as cf:
                                            df_s = pd.read_csv(cf, encoding='ISO-8859-1', on_bad_lines='skip')
                                            df_s['Source_File'] = fn
                                            all_dfs.append(df_s)
                                    except Exception: # Catch errors for specific CSVs inside ZIP
                                        pass
                    except Exception: # Catch errors for ZIP download/open
                        pass

        # Recursively scrape other HTML pages for more CSV/ZIP links
        for page_url in list(html_pages_to_scrape):
            if page_url in visited_html_pages:
                continue
            visited_html_pages.add(page_url)
            try:
                pr = requests.get(page_url, timeout=15)
                pr.raise_for_status()
                ps = BeautifulSoup(pr.content, 'html.parser')
                for link in ps.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.csv'):
                        full_url = urljoin(page_url, href)
                        if full_url.startswith(base_url):
                            unique_csv_urls.add(full_url)
                    elif href.endswith('.zip'):
                        if 'data.zip' in href or re.search(r'\b[A-Z0-9]{1,2}\.zip$', href):
                            fzu = urljoin(page_url, href)
                            if fzu.startswith(base_url):
                                try:
                                    zr = requests.get(fzu, stream=True, timeout=15)
                                    zr.raise_for_status()
                                    with zipfile.ZipFile(io.BytesIO(zr.content)) as z:
                                        for fn in z.namelist():
                                            if fn.endswith('.csv') and not fn.startswith('__MACOSX'):
                                                try:
                                                    with z.open(fn) as cf:
                                                        df_s = pd.read_csv(cf, encoding='ISO-8859-1', on_bad_lines='skip')
                                                        df_s['Source_File'] = fn
                                                        all_dfs.append(df_s)
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
            except Exception: # Catch errors for page request/parsing
                pass

        # Download all found CSVs
        for csv_url in unique_csv_urls:
            try:
                df_s = pd.read_csv(csv_url, encoding='ISO-8859-1', on_bad_lines='skip')
                df_s['Source_File'] = csv_url.split('/')[-1]
                all_dfs.append(df_s)
            except Exception: # Catch errors for CSV download
                pass

        if not all_dfs:
            return pd.DataFrame()

        df_full = pd.concat(all_dfs, ignore_index=True, sort=False)

        # Standardize date column and filter for valid dates
        date_cols = ['Date', 'MatchDate']
        for col in date_cols:
            if col in df_full.columns:
                df_full[col] = pd.to_datetime(df_full[col], format='%d/%m/%Y', errors='coerce')
                df_full.rename(columns={col: 'Date'}, inplace=True)
                break # Only need one valid date column
        df_full.dropna(subset=['Date'], inplace=True)

        df_full.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)
        df_full = df_full.sort_values(by='Date').reset_index(drop=True)

        # Fill missing stats with 0 for calculation purposes
        for col in ['HS', 'AS', 'HST', 'AST']:
            if col in df_full.columns:
                df_full[col] = df_full[col].fillna(0)

        return df_full

    except Exception as e:
        st.error(f"Error loading football data: {e}")
        return pd.DataFrame()


# =============================================================================
# FORM & STATS HELPERS (from second app)
# =============================================================================
def calculate_team_form(team_name, df_data, num_matches):
    team_matches = df_data[(df_data['HomeTeam'] == team_name) | (df_data['AwayTeam'] == team_name)].copy()
    if team_matches.empty:
        return 0, "No matches found", pd.DataFrame()
    recent = team_matches.sort_values('Date', ascending=False).head(num_matches)
    points = 0
    results = []
    for _, row in recent.iterrows():
        # Ensure 'FTR' column exists before trying to access it
        if 'FTR' not in row.index:
            st.warning(f"'FTR' column missing for match involving {team_name}. Cannot calculate form.")
            return 0, "N/A", pd.DataFrame()
        if row['HomeTeam'] == team_name:
            if row['FTR'] == 'H': points += 3; results.append('W')
            elif row['FTR'] == 'D': points += 1; results.append('D')
            else: results.append('L')
        else:
            if row['FTR'] == 'A': points += 3; results.append('W')
            elif row['FTR'] == 'D': points += 1; results.append('D')
            else: results.append('L')
    results.reverse()
    return points, " ".join(results), recent

def calculate_average_stats(team_name, recent_matches_df):
    if recent_matches_df.empty:
        return {'Shots For': 0, 'Shots Against': 0, 'Shots On Target For': 0, 'Shots On Target Against': 0}
    sf = sa = stf = sta = 0
    n_matches = 0
    for _, row in recent_matches_df.iterrows():
        if 'HS' in row.index and 'AS' in row.index and 'HST' in row.index and 'AST' in row.index:
            n_matches += 1
            if row['HomeTeam'] == team_name:
                sf += row.get('HS', 0); sa += row.get('AS', 0)
                stf += row.get('HST', 0); sta += row.get('AST', 0)
            else:
                sf += row.get('AS', 0); sa += row.get('HS', 0)
                stf += row.get('AST', 0); sta += row.get('HST', 0)
    if n_matches == 0:
        return {'Shots For': 0, 'Shots Against': 0, 'Shots On Target For': 0, 'Shots On Target Against': 0}
    return {
        'Shots For': sf / n_matches, 'Shots Against': sa / n_matches,
        'Shots On Target For': stf / n_matches, 'Shots On Target Against': sta / n_matches
    }

def calculate_season_average_stats(team_name, df_data):
    tm = df_data[(df_data['HomeTeam'] == team_name) | (df_data['AwayTeam'] == team_name)].copy()
    if tm.empty:
        return {'Shots For': 0, 'Shots Against': 0, 'Shots On Target For': 0, 'Shots On Target Against': 0}
    sf = sa = stf = sta = 0
    n_matches = 0
    for _, row in tm.iterrows():
        if 'HS' in row.index and 'AS' in row.index and 'HST' in row.index and 'AST' in row.index:
            n_matches += 1
            if row['HomeTeam'] == team_name:
                sf += row.get('HS', 0); sa += row.get('AS', 0)
                stf += row.get('HST', 0); sta += row.get('AST', 0)
            else:
                sf += row.get('AS', 0); sa += row.get('HS', 0)
                stf += row.get('AST', 0); sta += row.get('HST', 0)
    if n_matches == 0:
        return {'Shots For': 0, 'Shots Against': 0, 'Shots On Target For': 0, 'Shots On Target Against': 0}
    return {
        'Shots For': sf / n_matches, 'Shots Against': sa / n_matches,
        'Shots On Target For': stf / n_matches, 'Shots On Target Against': sta / n_matches
    }


# =============================================================================
# ELO (from second app)
# =============================================================================
@st.cache_data(show_spinner="Fetching ELO ratings...", ttl=1800)
def load_elo_ratings():
    for delta in [0, 1]: # Try today and yesterday's ELO data
        try:
            date_str = (datetime.now() - pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
            resp = requests.get(f"http://api.clubelo.com/{date_str}", timeout=15)
            resp.raise_for_status()
            df_elo = pd.read_csv(io.StringIO(resp.text))
            df_elo['Clean_Club'] = df_elo['Club'].astype(str).str.strip()
            return df_elo
        except Exception:
            continue
    st.warning("Could not load ELO ratings. Defaulting to 1500.")
    return pd.DataFrame(columns=['Club', 'Elo', 'Clean_Club'])

def get_elo_for_team(team_name, df_elo):
    if df_elo.empty or not team_name.strip():
        return DEFAULT_ELO, None
    name_lower = team_name.strip().lower()
    mask = df_elo['Clean_Club'].str.lower().str.contains(name_lower, na=False, regex=False)
    matches = df_elo[mask]
    if matches.empty:
        return DEFAULT_ELO, f'No ELO match for "{team_name}". Using default {DEFAULT_ELO}.'
    exact = matches[matches['Clean_Club'].str.lower() == name_lower]
    if len(exact) == 1:
        return float(exact['Elo'].iloc[0]), None
    if len(matches) > 1:
        candidates = ', '.join(matches['Clean_Club'].tolist()[:5])
        return DEFAULT_ELO, f'"{team_name}" matched multiple ({candidates}). Using default.'
    return float(matches['Elo'].iloc[0]), None


# =============================================================================
# SESSION STATE INIT (from second app)
# =============================================================================
if 'inputs' not in st.session_state:
    st.session_state.inputs = {**APP_DEFAULTS}
if 'fd_analysis_done' not in st.session_state:
    st.session_state.fd_analysis_done = False
if 'fd_results' not in st.session_state:
    st.session_state.fd_results = {}


# Load resources for auto-population
df_elo = load_elo_ratings()

# -----------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------
st.sidebar.header("📊 Match Inputs")

# Team name inputs with ELO auto-update
home_team_default = st.session_state.get('prev_home_team', 'Manchester City')
away_team_default = st.session_state.get('prev_away_team', 'Liverpool')
home_team = st.sidebar.text_input("Home Team", home_team_default)
away_team = st.sidebar.text_input("Away Team", away_team_default)

# Check for team name changes to trigger ELO lookup and analysis invalidation
teams_changed = False
if st.session_state.get('prev_home_team') != home_team:
    home_elo_scraped, home_warn = get_elo_for_team(home_team, df_elo)
    st.session_state.inputs['HomeElo'] = home_elo_scraped
    st.session_state.prev_home_team = home_team
    if st.session_state.fd_analysis_done:
        st.session_state.fd_analysis_done = False
        teams_changed = True

if st.session_state.get('prev_away_team') != away_team:
    away_elo_scraped, away_warn = get_elo_for_team(away_team, df_elo)
    st.session_state.inputs['AwayElo'] = away_elo_scraped
    st.session_state.prev_away_team = away_team
    if st.session_state.fd_analysis_done:
        st.session_state.fd_analysis_done = False
        teams_changed = True

# Display ELO warnings if any
if 'home_warn' in locals() and home_warn: st.sidebar.warning(f"🏠 {home_warn}")
if 'away_warn' in locals() and away_warn: st.sidebar.warning(f"🏟️ {away_warn}")

# Analyze Teams button and status
btn_col, status_col = st.sidebar.columns([2, 3])
with btn_col:
    analyze_clicked = st.button(
        "📊 Analyze Teams",
        type="primary",
        use_container_width=True,
        help="Loads historical data from football-data.co.uk and calculates form + shot stats",
        key='analyze_teams_btn'
    )

with status_col:
    if teams_changed:
        st.warning("⚠️ Team names changed — click **Analyze Teams** to refresh stats.")
    elif st.session_state.fd_analysis_done:
        st.success("✅ Historical stats loaded — inputs updated. Ready to predict.")

# Historical data analysis logic
if analyze_clicked:
    with st.spinner('Loading and merging football data from football-data.co.uk... This may take a moment.'):
        df_full = download_and_merge_football_data()

    if df_full.empty:
        st.error("Could not load football data. Check internet connection or data source.")
        st.session_state.fd_analysis_done = False
    else:
        pts3_home, form3_str_home, matches3_home = calculate_team_form(home_team, df_full, 3)
        pts5_home, form5_str_home, matches5_home = calculate_team_form(home_team, df_full, 5)
        pts3_away, form3_str_away, matches3_away = calculate_team_form(away_team, df_full, 3)
        pts5_away, form5_str_away, matches5_away = calculate_team_form(away_team, df_full, 5)

        # Note: The first app's model does not directly use 'shots' or 'target' as features.
        # These are calculated for potential display/info, but XG is the direct input.
        stats5_home = calculate_average_stats(home_team, matches5_home)
        stats5_away = calculate_average_stats(away_team, matches5_away)
        season_stats_home = calculate_season_average_stats(home_team, df_full)
        season_stats_away = calculate_season_average_stats(away_team, df_full)

        w_season, w_last5 = 0.70, 0.30 # Weighted average for shots/target
        weighted_home_stats = {
            k: (w_season * season_stats_home[k]) + (w_last5 * stats5_home[k])
            for k in stats5_home
        }
        weighted_away_stats = {
            k: (w_season * season_stats_away[k]) + (w_last5 * stats5_away[k])
            for k in stats5_away
        }

        home_xg_est = round(weighted_home_stats['Shots On Target For'] / 3.0, 2)
        away_xg_est = round(weighted_away_stats['Shots On Target For'] / 3.0, 2)

        st.session_state.inputs.update({
            'Form3Home': pts3_home,
            'Form5Home': pts5_home,
            'Form3Away': pts3_away,
            'Form5Away': pts5_away,
            'HomeExpectedGoals': max(0.1, home_xg_est),
            'AwayExpectedGoals': max(0.1, away_xg_est),
            # Store raw shot stats, though not direct model inputs for THIS app
            'HomeShots': round(weighted_home_stats['Shots For'], 1),
            'AwayShots': round(weighted_away_stats['Shots For'], 1),
            'HomeTarget': round(weighted_home_stats['Shots On Target For'], 1),
            'AwayTarget': round(weighted_away_stats['Shots On Target For'], 1),
        })

        st.session_state.fd_results = {
            'home_team': home_team, 'away_team': away_team,
            'pts3_home': pts3_home, 'form3_str_home': form3_str_home, 'matches3_home': matches3_home,
            'pts5_home': pts5_home, 'form5_str_home': form5_str_home, 'matches5_home': matches5_home,
            'pts3_away': pts3_away, 'form3_str_away': form3_str_away, 'matches3_away': matches3_away,
            'pts5_away': pts5_away, 'form5_str_away': form5_str_away, 'matches5_away': matches5_away,
            'weighted_home': weighted_home_stats, 'weighted_away': weighted_away_stats,
        }
        st.session_state.fd_analysis_done = True
        st.rerun()

# Display historical analysis results if done
if st.session_state.fd_analysis_done and st.session_state.fd_results and \
   st.session_state.fd_results.get('home_team') == home_team and st.session_state.fd_results.get('away_team') == away_team:
    r = st.session_state.fd_results
    st.markdown("---")
    st.header(f"📋 Historical Analysis: {home_team} vs {away_team}")

    left, right = st.columns(2)

    with left:
        st.subheader(f"🏠 {home_team}")
        st.write(f"**Last 3:** {r['form3_str_home']} — Points: **{r['pts3_home']}**")
        if not r['matches3_home'].empty:
            cols_to_show = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                            if c in r['matches3_home'].columns]
            st.dataframe(r['matches3_home'][cols_to_show], use_container_width=True)

        st.write(f"**Last 5:** {r['form5_str_home']} — Points: **{r['pts5_home']}**")
        if not r['matches5_home'].empty:
            cols_to_show = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                            if c in r['matches5_home'].columns]
            st.dataframe(r['matches5_home'][cols_to_show], use_container_width=True)

        st.markdown("**Weighted Shot Stats (70% season / 30% last 5):**")
        for k, v in r['weighted_home'].items():
            st.write(f"- {k}: **{v:.2f}**")

    with right:
        st.subheader(f"🏟️ {away_team}")
        st.write(f"**Last 3:** {r['form3_str_away']} — Points: **{r['pts3_away']}**")
        if not r['matches3_away'].empty:
            cols_to_show = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                            if c in r['matches3_away'].columns]
            st.dataframe(r['matches3_away'][cols_to_show], use_container_width=True)

        st.write(f"**Last 5:** {r['form5_str_away']} — Points: **{r['pts5_away']}**")
        if not r['matches5_away'].empty:
            cols_to_show = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                            if c in r['matches5_away'].columns]
            st.dataframe(r['matches5_away'][cols_to_show], use_container_width=True)

        st.markdown("**Weighted Shot Stats (70% season / 30% last 5):**")
        for k, v in r['weighted_away'].items():
            st.write(f"- {k}: **{v:.2f}**")

    st.info(
        "✅ Form points and shot stats have been automatically applied to the prediction inputs "
        "in the sidebar. Review and adjust as needed, then click **Predict Match**."
    )


# Actual Sidebar Input Fields (now using session_state for default values)

st.sidebar.subheader("Elo Ratings")
home_elo = st.sidebar.number_input("Home Elo", min_value=1000.0, max_value=3000.0,
                                   value=float(st.session_state.inputs['HomeElo'] if 'HomeElo' in st.session_state.inputs else APP_DEFAULTS['HomeElo']))
away_elo = st.sidebar.number_input("Away Elo", min_value=1000.0, max_value=3000.0,
                                   value=float(st.session_state.inputs['AwayElo'] if 'AwayElo' in st.session_state.inputs else APP_DEFAULTS['AwayElo']))
st.session_state.inputs['HomeElo'] = home_elo # Update session state on widget change
st.session_state.inputs['AwayElo'] = away_elo

st.sidebar.subheader("Recent Form")
form3_home = st.sidebar.slider("Form3 Home", 0, 15,
                               value=int(st.session_state.inputs['Form3Home'] if 'Form3Home' in st.session_state.inputs else APP_DEFAULTS['Form3Home']))
form5_home = st.sidebar.slider("Form5 Home", 0, 15,
                               value=int(st.session_state.inputs['Form5Home'] if 'Form5Home' in st.session_state.inputs else APP_DEFAULTS['Form5Home']))
form3_away = st.sidebar.slider("Form3 Away", 0, 15,
                               value=int(st.session_state.inputs['Form3Away'] if 'Form3Away' in st.session_state.inputs else APP_DEFAULTS['Form3Away']))
form5_away = st.sidebar.slider("Form5 Away", 0, 15,
                               value=int(st.session_state.inputs['Form5Away'] if 'Form5Away' in st.session_state.inputs else APP_DEFAULTS['Form5Away']))
st.session_state.inputs['Form3Home'] = form3_home
st.session_state.inputs['Form5Home'] = form5_home
st.session_state.inputs['Form3Away'] = form3_away
st.session_state.inputs['Form5Away'] = form5_away

st.sidebar.subheader("Expected Goals")
home_xg = st.sidebar.slider("Historical Home XG", 0.0, 5.0, step=0.1,
                            value=float(st.session_state.inputs['HomeExpectedGoals'] if 'HomeExpectedGoals' in st.session_state.inputs else APP_DEFAULTS['HomeExpectedGoals']))
away_xg = st.sidebar.slider("Historical Away XG", 0.0, 5.0, step=0.1,
                            value=float(st.session_state.inputs['AwayExpectedGoals'] if 'AwayExpectedGoals' in st.session_state.inputs else APP_DEFAULTS['AwayExpectedGoals']))
st.session_state.inputs['HomeExpectedGoals'] = home_xg
st.session_state.inputs['AwayExpectedGoals'] = away_xg

st.sidebar.subheader("Betting Odds")
odd_home = st.sidebar.number_input("Home Win Odds", 1.01, 50.0, step=0.05,
                                   value=float(st.session_state.inputs['OddHome'] if 'OddHome' in st.session_state.inputs else APP_DEFAULTS['OddHome']))
odd_draw = st.sidebar.number_input("Draw Odds", 1.01, 50.0, step=0.05,
                                   value=float(st.session_state.inputs['OddDraw'] if 'OddDraw' in st.session_state.inputs else APP_DEFAULTS['OddDraw']))
odd_away = st.sidebar.number_input("Away Win Odds", 1.01, 50.0, step=0.05,
                                   value=float(st.session_state.inputs['OddAway'] if 'OddAway' in st.session_state.inputs else APP_DEFAULTS['OddAway']))
st.session_state.inputs['OddHome'] = odd_home
st.session_state.inputs['OddDraw'] = odd_draw
st.session_state.inputs['OddAway'] = odd_away

st.sidebar.subheader("Handicap")
handi_size = st.sidebar.slider("Handicap Size", -5.0, 5.0, step=0.25,
                               value=float(st.session_state.inputs['HandiSize'] if 'HandiSize' in st.session_state.inputs else APP_DEFAULTS['HandiSize']))
st.session_state.inputs['HandiSize'] = handi_size

st.sidebar.subheader("Cluster Features")
c_lth = st.sidebar.number_input("C_LTH",
                                value=float(st.session_state.inputs['C_LTH'] if 'C_LTH' in st.session_state.inputs else APP_DEFAULTS['C_LTH']))
c_lta = st.sidebar.number_input("C_LTA",
                                value=float(st.session_state.inputs['C_LTA'] if 'C_LTA' in st.session_state.inputs else APP_DEFAULTS['C_LTA']))
c_vhd = st.sidebar.number_input("C_VHD",
                                value=float(st.session_state.inputs['C_VHD'] if 'C_VHD' in st.session_state.inputs else APP_DEFAULTS['C_VHD']))
c_vad = st.sidebar.number_input("C_VAD",
                                value=float(st.session_state.inputs['C_VAD'] if 'C_VAD' in st.session_state.inputs else APP_DEFAULTS['C_VAD']))
c_htb = st.sidebar.number_input("C_HTB",
                                value=float(st.session_state.inputs['C_HTB'] if 'C_HTB' in st.session_state.inputs else APP_DEFAULTS['C_HTB']))
c_phb = st.sidebar.number_input("C_PHB",
                                value=float(st.session_state.inputs['C_PHB'] if 'C_PHB' in st.session_state.inputs else APP_DEFAULTS['C_PHB']))
st.session_state.inputs['C_LTH'] = c_lth
st.session_state.inputs['C_LTA'] = c_lta
st.session_state.inputs['C_VHD'] = c_vhd
st.session_state.inputs['C_VAD'] = c_vad
st.session_state.inputs['C_HTB'] = c_htb
st.session_state.inputs['C_PHB'] = c_phb

# Reset to Defaults button
if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True, key='reset_btn'):
    # Reset session state inputs to initial APP_DEFAULTS, but keep scraped ELOs if they exist
    reset_elo_home = get_elo_for_team(home_team, df_elo)[0] if 'home_team' in locals() else APP_DEFAULTS['HomeElo']
    reset_elo_away = get_elo_for_team(away_team, df_elo)[0] if 'away_team' in locals() else APP_DEFAULTS['AwayElo']

    st.session_state.inputs = {**APP_DEFAULTS, 'HomeElo': reset_elo_home, 'AwayElo': reset_elo_away}
    st.session_state.fd_analysis_done = False
    st.session_state.fd_results = {}
    st.rerun()

predict_clicked = st.sidebar.button("🚀 Predict Match", type="primary", use_container_width=True, key='predict_btn')

# -----------------------------------------------------------
# FEATURE ENGINEERING (Uses values from st.session_state.inputs)
# -----------------------------------------------------------
# Fetch latest values from session state for feature engineering
current_inputs = st.session_state.inputs

elo_diff = current_inputs['HomeElo'] - current_inputs['AwayElo']
elo_ratio = current_inputs['HomeElo'] / current_inputs['AwayElo'] if current_inputs['AwayElo'] != 0 else 1.0

form3_diff = current_inputs['Form3Home'] - current_inputs['Form3Away']
form5_diff = current_inputs['Form5Home'] - current_inputs['Form5Away']
form5_ratio = (current_inputs['Form5Home'] + 0.5) / (current_inputs['Form5Away'] + 0.5)

xg_diff = current_inputs['HomeExpectedGoals'] - current_inputs['AwayExpectedGoals']

home_attack = current_inputs['HomeExpectedGoals'] * current_inputs['Form5Home'] / 7.5
away_attack = current_inputs['AwayExpectedGoals'] * current_inputs['Form5Away'] / 7.5

prob_home = 1 / current_inputs['OddHome'] if current_inputs['OddHome'] != 0 else 0
prob_draw = 1 / current_inputs['OddDraw'] if current_inputs['OddDraw'] != 0 else 0
prob_away = 1 / current_inputs['OddAway'] if current_inputs['OddAway'] != 0 else 0

prob_sum = prob_home + prob_draw + prob_away

# Normalize probabilities, handle zero sum to avoid division by zero
if prob_sum > 0:
    prob_home /= prob_sum
    prob_draw /= prob_sum
    prob_away /= prob_sum
else:
    prob_home, prob_draw, prob_away = 0.45, 0.28, 0.27 # Default if sum is zero

odds_edge = prob_home - prob_away

# Hardcoded temporal features as in the original first app
month = 5
day_of_week = 4

input_dict = {
    "HomeElo": current_inputs['HomeElo'],
    "AwayElo": current_inputs['AwayElo'],
    "EloDiff": elo_diff,
    "EloRatio": elo_ratio,

    "Form3Diff": form3_diff,
    "Form5Diff": form5_diff,
    "Form5Ratio": form5_ratio,

    "Form3Home": current_inputs['Form3Home'],
    "Form5Home": current_inputs['Form5Home'],
    "Form3Away": current_inputs['Form3Away'],
    "Form5Away": current_inputs['Form5Away'],

    "HomeXG": current_inputs['HomeExpectedGoals'], # Mapped from HomeExpectedGoals
    "AwayXG": current_inputs['AwayExpectedGoals'], # Mapped from AwayExpectedGoals
    "XGDiff": xg_diff,

    "HomeAttack": home_attack,
    "AwayAttack": away_attack,

    "ProbHome": prob_home,
    "ProbDraw": prob_draw,
    "ProbAway": prob_away,
    "OddsEdge": odds_edge,

    "HandiSize": current_inputs['HandiSize'],

    "MatchMonth": month,
    "MatchDayOfWeek": day_of_week,

    "C_LTH": current_inputs['C_LTH'],
    "C_LTA": current_inputs['C_LTA'],
    "C_VHD": current_inputs['C_VHD'],
    "C_VAD": current_inputs['C_VAD'],
    "C_HTB": current_inputs['C_HTB'],
    "C_PHB": current_inputs['C_PHB']
}

# Ensure all feature columns exist, fill with 0 if missing from input_dict
# This handles cases where FEATURE_COLS might contain items not directly from current_inputs or derived
final_input_dict = {col: input_dict.get(col, 0.0) for col in FEATURE_COLS}
X_input = pd.DataFrame([final_input_dict])[FEATURE_COLS]

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------

if predict_clicked:

    # Outcome prediction
    outcome_probs = outcome_model.predict_proba(X_input)[0]
    outcome_pred = outcome_model.predict(X_input)[0]
    outcome_label = label_encoder.inverse_transform([outcome_pred])[0]

    # Goals prediction
    pred_home_goals = float(goals_home_model.predict(X_input)[0])
    pred_away_goals = float(goals_away_model.predict(X_input)[0])

    pred_home_goals = max(0, pred_home_goals)
    pred_away_goals = max(0, pred_away_goals)

    total_goals = pred_home_goals + pred_away_goals

    # Score probabilities
    matrix = score_matrix(pred_home_goals, pred_away_goals)

    home_win_prob, draw_prob, away_win_prob = outcome_from_score_matrix(matrix)

    # -------------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------------

    st.header(f"🏟️ {home_team} vs {away_team}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🏠 Expected Home Goals",
            f"{pred_home_goals:.2f}"
        )

    with col2:
        st.metric(
            "⚽ Expected Total Goals",
            f"{total_goals:.2f}"
        )

    with col3:
        st.metric(
            "🚗 Expected Away Goals",
            f"{pred_away_goals:.2f}"
        )

    st.divider()

    # -------------------------------------------------------
    # OUTCOME PROBABILITIES
    # -------------------------------------------------------

    st.subheader("📈 Match Outcome Probabilities")

    # Ensure outcome_probs has 3 elements for H, D, A
    # Assuming label_encoder.classes_ maps to the order of outcome_probs
    outcome_map = {cls: prob for cls, prob in zip(label_encoder.classes_, outcome_probs)}

    outcome_df = pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Probability": [
            outcome_map.get('H', 0.0),
            outcome_map.get('D', 0.0),
            outcome_map.get('A', 0.0)
        ]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=outcome_df["Outcome"],
        y=outcome_df["Probability"]
    ))

    fig.update_layout(
        height=400,
        yaxis_title="Probability",
        xaxis_title="Outcome"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------
    # FINAL PREDICTION
    # -------------------------------------------------------

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
    - Home Win Probability: **{outcome_map.get('H', 0.0):.2%}**
    - Draw Probability: **{outcome_map.get('D', 0.0):.2%}**
    - Away Win Probability: **{outcome_map.get('A', 0.0):.2%}**
    """)

    # -------------------------------------------------------
    # SCORE MATRIX
    # -------------------------------------------------------

    st.subheader("📋 Most Likely Scorelines")

    score_probs = []

    for i in range(6):
        for j in range(6):
            score_probs.append({
                "Score": f"{i} - {j}",
                "Probability": matrix[i, j]
            })

    score_df = pd.DataFrame(score_probs)
    score_df = score_df.sort_values(
        by="Probability",
        ascending=False
    ).head(10)

    score_df["Probability"] = (
        score_df["Probability"] * 100
    ).round(2)

    st.dataframe(
        score_df,
        use_container_width=True
    )

    # -------------------------------------------------------
    # IMPLIED RESULT FROM POISSON MATRIX
    # -------------------------------------------------------

    st.subheader("📊 Poisson Score Model")

    poisson_df = pd.DataFrame({
        "Result": ["Home Win", "Draw", "Away Win"],
        "Probability": [
            home_win_prob,
            draw_prob,
            away_win_prob
        ]
    })

    poisson_df["Probability"] = (
        poisson_df["Probability"] * 100
    ).round(2)

    st.dataframe(poisson_df, use_container_width=True)

    # -------------------------------------------------------
    # FEATURE VIEW
    # -------------------------------------------------------

    with st.expander("🧠 Engineered Features Used"):

        feature_view = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Value": [final_input_dict[c] for c in FEATURE_COLS]
        })

        st.dataframe(feature_view, use_container_width=True)

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------

st.divider()

st.markdown(
    
### 📦 Required Files

- outcome_model.joblib
- goals_home_model.joblib
- goals_away_model.joblib
- label_encoder.joblib
- feature_cols.joblib

)

# Initial info message if no analysis has been done
if not st.session_state.fd_analysis_done and not predict_clicked:
    st.info("👈 Click **Analyze Teams** to load historical stats, then click **Predict Match** in the sidebar.")
