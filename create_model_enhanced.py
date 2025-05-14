import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
# Load the data
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

print("Processing data...")
# Data preprocessing
match_df = match.copy()
delivery_df = delivery.copy()

# Calculate total runs per match
total_score_df = delivery_df.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]['total_runs'].values

# Add target column to matches
match_df['target'] = total_score_df + 1

# Calculate venue statistics
venue_stats = delivery_df[delivery_df['inning'] == 1].groupby('match_id').agg({
    'total_runs': 'sum'
}).reset_index()
venue_stats = pd.merge(venue_stats, match_df[['id', 'venue']], left_on='match_id', right_on='id')
venue_avg_scores = venue_stats.groupby('venue')['total_runs'].mean().reset_index()
venue_avg_scores.columns = ['venue', 'venue_avg_score']

print("Creating enhanced features...")
# Create ball by ball match data with enhanced features
match_ids = delivery_df['match_id'].unique()
ball_data = []

for match_id in match_ids:
    match_info = match_df[match_df['id'] == match_id].iloc[0]
    innings2 = delivery_df[delivery_df['match_id'] == match_id][delivery_df['inning'] == 2]
    
    current_score = 0
    wickets = 0
    balls = 0
    target = match_info['target']
    venue_avg = venue_avg_scores[venue_avg_scores['venue'] == match_info['venue']]['venue_avg_score'].values[0]
    
    for ball in innings2.itertuples():
        balls += 1
        current_score += ball.total_runs
        if pd.notna(ball.player_dismissed):
            wickets += 1
            
        runs_left = target - current_score
        balls_left = 120 - balls
        wickets_left = 10 - wickets
        overs = balls // 6
        current_over = overs + (balls % 6) / 10  # Convert to decimal overs
        
        # Enhanced features
        crr = (current_score * 6) / balls if balls > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 999
        
        # New and improved features
        is_powerplay = 1 if overs <= 6 else 0
        is_death_overs = 1 if overs >= 16 else 0
        nrr_diff = crr - rrr
        
        # Resource-based features
        wicket_resources = (wickets_left * 100) / 10  # Percentage of wickets remaining
        balls_resources = (balls_left * 100) / 120    # Percentage of balls remaining
        total_resources = (wicket_resources + balls_resources) / 2
        
        # Pressure and difficulty metrics
        match_pressure = (runs_left * wickets) / (balls_left + 1)  # Higher value = more pressure
        chase_difficulty = (target - current_score) / (balls_left / 6)  # Runs needed per over
        venue_difficulty = target / venue_avg
        
        # Run rate required per wicket
        runs_per_wicket_needed = runs_left / (wickets_left + 1)
        
        ball_data.append({
            'match_id': match_id,
            'batting_team': match_info['team2'],
            'bowling_team': match_info['team1'],
            'city': match_info['city'],
            'venue': match_info['venue'],
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets': wickets_left,
            'target': target,
            'current_score': current_score,
            'current_over': current_over,
            'crr': crr,
            'rrr': rrr,
            'is_powerplay': is_powerplay,
            'is_death_overs': is_death_overs,
            'nrr_diff': nrr_diff,
            'wicket_resources': wicket_resources,
            'balls_resources': balls_resources,
            'total_resources': total_resources,
            'match_pressure': match_pressure,
            'chase_difficulty': chase_difficulty,
            'venue_difficulty': venue_difficulty,
            'runs_per_wicket_needed': runs_per_wicket_needed,
            'venue_avg_score': venue_avg,
            'result': 1 if match_info['winner'] == match_info['team2'] else 0
        })

ball_df = pd.DataFrame(ball_data)

# Remove extreme cases where prediction would be trivial
ball_df = ball_df[ball_df['balls_left'] > 0]
ball_df = ball_df[ball_df['runs_left'] > 0]

print("Preparing features...")
# Prepare features and target
categorical_features = ['batting_team', 'bowling_team', 'city']  # Removed 'venue' to reduce complexity
numerical_features = [
    'runs_left', 'balls_left', 'wickets', 'target', 
    'crr', 'rrr', 'total_resources',
    'match_pressure', 'chase_difficulty'
]  # Selected most important features only

X = ball_df[categorical_features + numerical_features]
y = ball_df['result']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Creating and training model...")
# Create preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Create base classifier with reduced complexity
base_clf = RandomForestClassifier(
    n_estimators=100,  # Reduced from 200
    max_depth=10,      # Reduced from 15
    min_samples_split=20,  # Increased from 10
    min_samples_leaf=10,   # Increased from 5
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# Create calibrated classifier with fewer CV folds
calibrated_clf = CalibratedClassifierCV(base_clf, cv=3, method='sigmoid')  # Reduced CV folds from 5 to 3

# Create the full pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', calibrated_clf)
])

# Train the model
pipe.fit(X_train, y_train)

print("Evaluating model...")
# Make predictions
y_pred = pipe.predict(X_test)
y_pred_proba = pipe.predict_proba(X_test)

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print("=============================")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nSaving model...")
# Save the model
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Done! Model saved as pipe.pkl") 