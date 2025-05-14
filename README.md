# IPL Win Predictor API

This Flask API serves predictions for IPL match outcomes based on the current match situation.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Get Win Prediction
**Endpoint:** `/predict`
**Method:** POST

**Request Body:**
```json
{
    "batting_team": "Mumbai Indians",
    "bowling_team": "Chennai Super Kings",
    "city": "Mumbai",
    "target": 180,
    "score": 100,
    "overs": 12.4,
    "wickets": 3
}
```

**Response:**
```json
{
    "loss_probability": 0.3,
    "win_probability": 0.7,
    "message": "Mumbai Indians has a 70.0% chance of winning!"
}
```

### 2. Get Valid Teams
**Endpoint:** `/teams`
**Method:** GET

**Response:**
```json
{
    "teams": [
        "Sunrisers Hyderabad",
        "Mumbai Indians",
        "Royal Challengers Bangalore",
        "Kolkata Knight Riders",
        "Kings XI Punjab",
        "Chennai Super Kings",
        "Rajasthan Royals",
        "Delhi Capitals"
    ]
}
```

## Example Usage with cURL

1. Get prediction:
```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "batting_team": "Mumbai Indians",
    "bowling_team": "Chennai Super Kings",
    "city": "Mumbai",
    "target": 180,
    "score": 100,
    "overs": 12.4,
    "wickets": 3
}'
```

2. Get teams list:
```bash
curl http://localhost:5000/teams
``` 