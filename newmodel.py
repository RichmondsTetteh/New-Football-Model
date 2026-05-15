# app.py
# Advanced Football Match Prediction App
# --------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from math import exp, factorial
import plotly.graph_objects as go

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

- Match Winner
- Draw Probability
- Expected Goals
- Most Likely Scorelines
- Poisson Probabilities

Features are AUTO-generated from:
- football-data.co.uk
- ClubElo ratings
- Historical form
- Historical goals
""")

# -----------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------

DEFAULT_ELO = 1500

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
# LOAD FOOTBALL DATA
# -----------------------------------------------------------

@st.cache_data(show_spinner="Loading football data...")
def download_football_data():

    base_url = "https://www.football-data.co.uk/"
    page_url = base_url + "data.php"

    dfs = []

    try:

        response = requests.get(page_url, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        csv_urls = set()

        for link in soup.find_all("a", href=True):

            href = link["href"]

            if href.endswith(".csv"):

                csv_urls.add(urljoin(base_url, href))

        for csv_url in csv_urls:

            try:

                df = pd.read_csv(
                    csv_url,
                    encoding="ISO-8859-1",
                    on_bad_lines="skip"
                )

                dfs.append(df)

            except Exception:
                pass

        if not dfs:
            return pd.DataFrame()

        data = pd.concat(dfs, ignore_index=True)

        data["Date"] = pd.to_datetime(
            data["Date"],
            dayfirst=True,
            errors="coerce"
        )

        data = data.dropna(subset=["Date"])

        data = data.sort_values("Date")

        return data

    except Exception as e:

        st.error(f"Failed loading football data: {e}")

        return pd.DataFrame()

# -----------------------------------------------------------
# LOAD ELO
# -----------------------------------------------------------

@st.cache_data(show_spinner="Loading ClubElo ratings...")
def load_elo_ratings():

    try:

        date_str = datetime.now().strftime("%Y-%m-%d")

        url = f"http://api.clubelo.com/{date_str}"

        elo_df = pd.read_csv(url)

        elo_df["Clean_Club"] = (
            elo_df["Club"]
            .astype(str)
            .str.strip()
        )

        return elo_df

    except Exception:

        return pd.DataFrame()

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def get_elo(team_name, elo_df):

    if elo_df.empty:
        return DEFAULT_ELO

    matches = elo_df[
        elo_df["Clean_Club"]
        .str.lower()
        .str.contains(team_name.lower(), na=False)
    ]

    if matches.empty:
        return DEFAULT_ELO

    return float(matches["Elo"].iloc[0])

def calculate_form(team_name, data, n=5):

    matches = data[
        (data["HomeTeam"] == team_name) |
        (data["AwayTeam"] == team_name)
    ].copy()

    if matches.empty:
        return 0

    matches = matches.sort_values(
        "Date",
        ascending=False
    ).head(n)

    points = 0

    for _, row in matches.iterrows():

        if row["HomeTeam"] == team_name:

            if row["FTR"] == "H":
                points += 3

            elif row["FTR"] == "D":
                points += 1

        else:

            if row["FTR"] == "A":
                points += 3

            elif row["FTR"] == "D":
                points += 1

    return points

def get_goal_stats(team_name, data):

    matches = data[
        (data["HomeTeam"] == team_name) |
        (data["AwayTeam"] == team_name)
    ].copy()

    if matches.empty:
        return 1.5

    goals = []

    for _, row in matches.iterrows():

        if row["HomeTeam"] == team_name:
            goals.append(row.get("FTHG", 0))
        else:
            goals.append(row.get("FTAG", 0))

    return np.mean(goals)

def poisson_prob(k, lam):

    return (exp(-lam) * (lam ** k)) / factorial(k)

def score_matrix(home_xg, away_xg, max_goals=5):

    matrix = np.zeros((max_goals + 1, max_goals + 1))

    for i in range(max_goals + 1):

        for j in range(max_goals + 1):

            matrix[i, j] = (
                poisson_prob(i, home_xg) *
                poisson_prob(j, away_xg)
            )

    return matrix

def outcome_from_matrix(matrix):

    home = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away = np.triu(matrix, 1).sum()

    return home, draw, away

# -----------------------------------------------------------
# LOAD DATASETS
# -----------------------------------------------------------

data = download_football_data()

elo_df = load_elo_ratings()

if data.empty:

    st.error("No football data loaded.")
    st.stop()

all_teams = sorted(
    pd.concat([
        data["HomeTeam"],
        data["AwayTeam"]
    ]).dropna().unique()
)

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------

st.sidebar.header("⚽ Match Selection")

home_team = st.sidebar.selectbox(
    "Home Team",
    all_teams
)

away_team = st.sidebar.selectbox(
    "Away Team",
    all_teams,
    index=1
)

# -----------------------------------------------------------
# AUTO FEATURES
# -----------------------------------------------------------

home_elo = get_elo(home_team, elo_df)
away_elo = get_elo(away_team, elo_df)

form3_home = calculate_form(home_team, data, 3)
form5_home = calculate_form(home_team, data, 5)

form3_away = calculate_form(away_team, data, 3)
form5_away = calculate_form(away_team, data, 5)

home_xg = get_goal_stats(home_team, data)
away_xg = get_goal_stats(away_team, data)

# -----------------------------------------------------------
# OPTIONAL USER INPUTS
# -----------------------------------------------------------

st.sidebar.header("📈 Market Inputs")

odd_home = st.sidebar.number_input(
    "Home Odds",
    min_value=1.01,
    max_value=50.0,
    value=2.00
)

odd_draw = st.sidebar.number_input(
    "Draw Odds",
    min_value=1.01,
    max_value=50.0,
    value=3.40
)

odd_away = st.sidebar.number_input(
    "Away Odds",
    min_value=1.01,
    max_value=50.0,
    value=3.60
)

handi_size = st.sidebar.slider(
    "Handicap",
    -5.0,
    5.0,
    0.0,
    0.25
)

# -----------------------------------------------------------
# AUTO FEATURE DISPLAY
# -----------------------------------------------------------

st.subheader("📊 Auto-Generated Features")

feature_preview = pd.DataFrame({
    "Feature": [
        "Home Elo",
        "Away Elo",
        "Home Form (5)",
        "Away Form (5)",
        "Home XG",
        "Away XG"
    ],
    "Value": [
        round(home_elo, 2),
        round(away_elo, 2),
        form5_home,
        form5_away,
        round(home_xg, 2),
        round(away_xg, 2)
    ]
})

st.dataframe(
    feature_preview,
    use_container_width=True
)

# -----------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------

elo_diff = home_elo - away_elo
elo_ratio = home_elo / away_elo

form3_diff = form3_home - form3_away
form5_diff = form5_home - form5_away

form5_ratio = (
    (form5_home + 0.5) /
    (form5_away + 0.5)
)

xg_diff = home_xg - away_xg

home_attack = (
    home_xg * form5_home / 7.5
)

away_attack = (
    away_xg * form5_away / 7.5
)

prob_home = 1 / odd_home
prob_draw = 1 / odd_draw
prob_away = 1 / odd_away

prob_sum = (
    prob_home +
    prob_draw +
    prob_away
)

prob_home /= prob_sum
prob_draw /= prob_sum
prob_away /= prob_sum

odds_edge = prob_home - prob_away

month = datetime.now().month
day_of_week = datetime.now().weekday()

input_dict = {
    "HomeElo": home_elo,
    "AwayElo": away_elo,
    "EloDiff": elo_diff,
    "EloRatio": elo_ratio,

    "Form3Diff": form3_diff,
    "Form5Diff": form5_diff,
    "Form5Ratio": form5_ratio,

    "Form3Home": form3_home,
    "Form5Home": form5_home,

    "Form3Away": form3_away,
    "Form5Away": form5_away,

    "HomeXG": home_xg,
    "AwayXG": away_xg,

    "XGDiff": xg_diff,

    "HomeAttack": home_attack,
    "AwayAttack": away_attack,

    "ProbHome": prob_home,
    "ProbDraw": prob_draw,
    "ProbAway": prob_away,

    "OddsEdge": odds_edge,

    "HandiSize": handi_size,

    "MatchMonth": month,
    "MatchDayOfWeek": day_of_week,

    "C_LTH": 0,
    "C_LTA": 0,
    "C_VHD": 0,
    "C_VAD": 0,
    "C_HTB": 0,
    "C_PHB": 0
}

for col in FEATURE_COLS:

    if col not in input_dict:
        input_dict[col] = 0

X_input = pd.DataFrame(
    [input_dict]
)[FEATURE_COLS]

# -----------------------------------------------------------
# PREDICT
# -----------------------------------------------------------

if st.button("🔮 Predict Match"):

    outcome_probs = (
        outcome_model
        .predict_proba(X_input)[0]
    )

    outcome_pred = (
        outcome_model
        .predict(X_input)[0]
    )

    outcome_label = (
        label_encoder
        .inverse_transform([outcome_pred])[0]
    )

    pred_home_goals = max(
        0,
        float(goals_home_model.predict(X_input)[0])
    )

    pred_away_goals = max(
        0,
        float(goals_away_model.predict(X_input)[0])
    )

    total_goals = (
        pred_home_goals +
        pred_away_goals
    )

    matrix = score_matrix(
        pred_home_goals,
        pred_away_goals
    )

    home_prob, draw_prob, away_prob = (
        outcome_from_matrix(matrix)
    )

    # -------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------

    st.header(
        f"🏟️ {home_team} vs {away_team}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🏠 Home Goals",
        f"{pred_home_goals:.2f}"
    )

    c2.metric(
        "⚽ Total Goals",
        f"{total_goals:.2f}"
    )

    c3.metric(
        "🚗 Away Goals",
        f"{pred_away_goals:.2f}"
    )

    st.divider()

    # -------------------------------------------------------
    # OUTCOME CHART
    # -------------------------------------------------------

    outcome_df = pd.DataFrame({
        "Outcome": [
            "Home Win",
            "Draw",
            "Away Win"
        ],
        "Probability": outcome_probs
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=outcome_df["Outcome"],
        y=outcome_df["Probability"]
    ))

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------

    if outcome_label == "H":

        final_result = (
            f"{home_team} WIN"
        )

    elif outcome_label == "D":

        final_result = "DRAW"

    else:

        final_result = (
            f"{away_team} WIN"
        )

    st.success(final_result)

    # -------------------------------------------------------
    # SCORELINES
    # -------------------------------------------------------

    st.subheader(
        "📋 Most Likely Scorelines"
    )

    scores = []

    for i in range(6):

        for j in range(6):

            scores.append({
                "Score": f"{i}-{j}",
                "Probability": matrix[i, j]
            })

    score_df = pd.DataFrame(scores)

    score_df = (
        score_df
        .sort_values(
            "Probability",
            ascending=False
        )
        .head(10)
    )

    score_df["Probability"] = (
        score_df["Probability"] * 100
    ).round(2)

    st.dataframe(
        score_df,
        use_container_width=True
    )

    # -------------------------------------------------------
    # FEATURE TABLE
    # -------------------------------------------------------

    with st.expander(
        "🧠 Features Used"
    ):

        features_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Value": [
                input_dict[c]
                for c in FEATURE_COLS
            ]
        })

        st.dataframe(
            features_df,
            use_container_width=True
        )

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------

st.divider()

st.markdown("""
### 📦 Required Files

Place these files beside app.py:

- outcome_model.joblib
- goals_home_model.joblib
- goals_away_model.joblib
- label_encoder.joblib
- feature_cols.joblib

### ▶️ Run

streamlit run app.py
""")
