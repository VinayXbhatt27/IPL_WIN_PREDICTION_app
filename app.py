import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.title('🏏 IPL Win Predictor 🏆')

# Historical team performance data (win rates from recent seasons)
team_strength = {
    'Mumbai Indians': 0.65,
    'Chennai Super Kings': 0.63,
    'Delhi Capitals': 0.58,
    'Royal Challengers Bangalore': 0.55,
    'Kolkata Knight Riders': 0.54,
    'Sunrisers Hyderabad': 0.53,
    'Kings XI Punjab': 0.52,
    'Rajasthan Royals': 0.51
}

# Head to head winning percentages (simplified)
head_to_head = {
    ('Mumbai Indians', 'Chennai Super Kings'): 0.55,
    ('Chennai Super Kings', 'Mumbai Indians'): 0.45,
    # Add more combinations as needed
}

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# Dictionary mapping cities to venues
venues = {
    'Mumbai': 'Wankhede Stadium',
    'Kolkata': 'Eden Gardens',
    'Chennai': 'M. A. Chidambaram Stadium',
    'Delhi': 'Arun Jaitley Stadium',
    'Bangalore': 'M. Chinnaswamy Stadium',
    'Hyderabad': 'Rajiv Gandhi International Stadium',
    'Punjab': 'Punjab Cricket Association Stadium',
    'Jaipur': 'Sawai Mansingh Stadium'
}

# Enhanced venue characteristics
venue_characteristics = {
    'Wankhede Stadium': {'avg_score': 165, 'chase_success': 0.55, 'pitch_rating': 8},
    'Eden Gardens': {'avg_score': 160, 'chase_success': 0.52, 'pitch_rating': 7},
    'M. A. Chidambaram Stadium': {'avg_score': 155, 'chase_success': 0.48, 'pitch_rating': 6},
    'Arun Jaitley Stadium': {'avg_score': 162, 'chase_success': 0.51, 'pitch_rating': 7},
    'M. Chinnaswamy Stadium': {'avg_score': 170, 'chase_success': 0.58, 'pitch_rating': 9},
    'Rajiv Gandhi International Stadium': {'avg_score': 157, 'chase_success': 0.49, 'pitch_rating': 7},
    'Punjab Cricket Association Stadium': {'avg_score': 163, 'chase_success': 0.52, 'pitch_rating': 7},
    'Sawai Mansingh Stadium': {'avg_score': 158, 'chase_success': 0.50, 'pitch_rating': 6}
}

# Load the RandomForest model from main directory
model = None
model_path = os.path.join(os.path.dirname(__file__), 'ra_pipe.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    st.error(f"Model file '{model_path}' not found. Please ensure the file exists in the correct directory.")

# Add a warning/error state to the session state
if 'show_error' not in st.session_state:
    st.session_state.show_error = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams), key='batting_team')
with col2:
    # Filter out the batting team from bowling team options
    bowling_teams = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox('Select the bowling team', sorted(bowling_teams), key='bowling_team')

selected_city = st.selectbox('Select host city', sorted(cities))
selected_venue = venues.get(selected_city, "Other Venue")
venue_avg = venue_characteristics.get(selected_venue, {'avg_score': 160})['avg_score']

target = st.number_input('Target', min_value=0, step=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Input validation function
def validate_inputs():
    if target == 0:
        return False, "Please enter a valid target score"
    if score > target:
        return False, "Current score cannot be greater than target"
    if score == target and overs < 20:
        return False, "Target already achieved!"
    if target > 300:
        return False, "Invalid target score (too high)"
    if overs == 20 and score < target:
        return False, "Match already lost - all overs completed"
    if wickets == 10:
        return False, "Match already lost - all wickets down"
    if overs > 0 and score/overs > 36:  # Max possible run rate per over (6 sixes)
        return False, "Invalid current run rate (too high)"
    return True, ""

if st.button('Predict Probability'):
    # Validate inputs
    is_valid, error_message = validate_inputs()
    
    if not is_valid:
        st.error(error_message)
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        
        # Calculate run rates
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 999

        # Create input dataframe matching the RandomForest model's expected format
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr],
            'total_resources': [(wickets_left * 10 + balls_left/12)],
            'match_pressure': [(runs_left * wickets_left) / (balls_left + 1)],
            'chase_difficulty': [(target - score) / ((balls_left / 6) + 0.1)]
        })

        # Make prediction using the RandomForest model
        result = model.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]
        
        # Display predictions with enhanced UI
        st.header(f"{batting_team}- {str(round(win_prob * 100))}%")
        st.header(f"{bowling_team}- {str(round(loss_prob * 100))}%")
        
        # Display match situation analysis
        st.subheader("Match Situation Analysis:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Required Run Rate: {rrr:.2f}")
            st.write(f"Current Run Rate: {crr:.2f}")
            st.write(f"Resources Remaining: {(wickets_left * 10 + balls_left/12):.1f}%")
        
        with col2:
            st.write(f"Balls Remaining: {balls_left}")
            st.write(f"Wickets in Hand: {wickets_left}")
            st.write(f"Runs Needed per Over: {(runs_left * 6 / balls_left):.2f}" if balls_left > 0 else "N/A")
            
        # Provide detailed commentary
        st.subheader("Match Situation:")
        if win_prob > 0.8:
            st.success("Batting team is in a commanding position! Victory highly likely.")
        elif win_prob > 0.6:
            st.info("Batting team has the upper hand, but needs to maintain momentum.")
        elif win_prob > 0.4:
            st.warning("Match is evenly poised - could go either way!")
        elif win_prob > 0.2:
            st.warning("Bowling team has the advantage, batting team needs something special.")
        else:
            st.error("Bowling team is in control, batting team needs a miracle!")