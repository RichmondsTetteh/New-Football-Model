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
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
- Match Outcome (Home / Draw / Away)
- Expected Goals
- Total Goals
- Scoreline Probabilities

Powered by:
- XGBoost
- Poisson Goal Modelling
- Historical Form & Elo Features
""")

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
# HELPER FUNCTIONS
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

# -----------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------

st.sidebar.header("📊 Match Inputs")

home_team = st.sidebar.text_input("Home Team", "Manchester City")
away_team = st.sidebar.text_input("Away Team", "Liverpool")

st.sidebar.subheader("Elo Ratings")

home_elo = st.sidebar.number_input(
    "Home Elo",
    min_value=1000,
    max_value=3000,
    value=1800
)

away_elo = st.sidebar.number_input(
    "Away Elo",
    min_value=1000,
    max_value=3000,
    value=1750
)

st.sidebar.subheader("Recent Form")

form3_home = st.sidebar.slider("Form3 Home", 0, 15, 7)
form5_home = st.sidebar.slider("Form5 Home", 0, 15, 10)

form3_away = st.sidebar.slider("Form3 Away", 0, 15, 6)
form5_away = st.sidebar.slider("Form5 Away", 0, 15, 9)

st.sidebar.subheader("Expected Goals")

home_xg = st.sidebar.slider("Historical Home XG", 0.0, 5.0, 1.8, 0.1)
away_xg = st.sidebar.slider("Historical Away XG", 0.0, 5.0, 1.2, 0.1)

st.sidebar.subheader("Betting Odds")

odd_home = st.sidebar.number_input("Home Win Odds", 1.01, 50.0, 2.00)
odd_draw = st.sidebar.number_input("Draw Odds", 1.01, 50.0, 3.50)
odd_away = st.sidebar.number_input("Away Win Odds", 1.01, 50.0, 3.80)

st.sidebar.subheader("Handicap")

handi_size = st.sidebar.slider("Handicap Size", -5.0, 5.0, 0.0, 0.25)

st.sidebar.subheader("Cluster Features")

c_lth = st.sidebar.number_input("C_LTH", value=0.0)
c_lta = st.sidebar.number_input("C_LTA", value=0.0)
c_vhd = st.sidebar.number_input("C_VHD", value=0.0)
c_vad = st.sidebar.number_input("C_VAD", value=0.0)
c_htb = st.sidebar.number_input("C_HTB", value=0.0)
c_phb = st.sidebar.number_input("C_PHB", value=0.0)

# -----------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------

elo_diff = home_elo - away_elo
elo_ratio = home_elo / away_elo

form3_diff = form3_home - form3_away
form5_diff = form5_home - form5_away
form5_ratio = (form5_home + 0.5) / (form5_away + 0.5)

xg_diff = home_xg - away_xg

home_attack = home_xg * form5_home / 7.5
away_attack = away_xg * form5_away / 7.5

prob_home = 1 / odd_home
prob_draw = 1 / odd_draw
prob_away = 1 / odd_away

prob_sum = prob_home + prob_draw + prob_away

prob_home /= prob_sum
prob_draw /= prob_sum
prob_away /= prob_sum

odds_edge = prob_home - prob_away

month = 5
day_of_week = 4

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

    "C_LTH": c_lth,
    "C_LTA": c_lta,
    "C_VHD": c_vhd,
    "C_VAD": c_vad,
    "C_HTB": c_htb,
    "C_PHB": c_phb
}

# Ensure all feature columns exist
for col in FEATURE_COLS:
    if col not in input_dict:
        input_dict[col] = 0

X_input = pd.DataFrame([input_dict])[FEATURE_COLS]

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------

if st.button("🔮 Predict Match"):

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

    outcome_df = pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Probability": [
            outcome_probs[0],
            outcome_probs[1],
            outcome_probs[2]
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
    - Home Win Probability: **{outcome_probs[0]:.2%}**
    - Draw Probability: **{outcome_probs[1]:.2%}**
    - Away Win Probability: **{outcome_probs[2]:.2%}**
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
            "Value": [input_dict[c] for c in FEATURE_COLS]
        })

        st.dataframe(feature_view, use_container_width=True)

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------

st.divider()

st.markdown("""
### 📦 Required Files

Place these files in the same folder as `app.py`:

- outcome_model.joblib
- goals_home_model.joblib
- goals_away_model.joblib
- label_encoder.joblib
- feature_cols.joblib

### ▶️ Run App

```bash
streamlit run app.py
