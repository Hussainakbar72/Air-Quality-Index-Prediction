# Air Quality CO Predictor

This project trains a Random Forest model on the UCI Air Quality dataset to predict CO (carbon monoxide) concentration and serves the predictions through a small Flask web UI.

## Prerequisites
- Python 3.10+ (project tested with the bundled `venv`)
- Air Quality dataset (`AirQualityUCI.csv`) in the project root

## Setup
```bash
# 1. (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install project dependencies
pip install -r requirements.txt
```

## Train the model
```bash
python app.py
```
The script cleans the CSV, trains a Random Forest, reports RMSE/R², and writes the trained pipeline plus feature list to `models/rf_co_model.joblib`.

## Run the Flask app
```bash
python air-quality-flask.py
```
Then open the printed URL (default `http://127.0.0.1:5000`) to interact with the predictor UI located in `template/index.html`.

## Project structure
- `app.py` – data prep + training pipeline
- `air-quality-flask.py` – Flask server that loads the saved model
- `models/` – serialized model artifacts
- `template/` – frontend assets
- `AirQualityUCI.csv` – dataset (not included in version control)

