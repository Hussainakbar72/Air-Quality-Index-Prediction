import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Paths
CSV_PATH = "AirQualityUCI.csv"
TARGET = "CO(GT)"
SAVE_PATH = "models/rf_co_model.joblib"
RANDOM_STATE = 42

# Load CSV
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset file not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH, sep=';')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Fix comma decimals
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(',', '.', regex=False).str.strip()

# Convert numeric columns
for col in df.columns:
    if col not in ["Date", "Time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing target CO(GT)
df = df[~df[TARGET].isna()].copy()

# --- FIXED DATETIME PARSING (VERY IMPORTANT) ---
df["datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    format="%d/%m/%Y %H.%M.%S",   # Correct format for AirQuality UCI
    errors="coerce"
)

# Extract fixed features
df["hour"] = df["datetime"].dt.hour.fillna(12)
df["dayofweek"] = df["datetime"].dt.dayofweek.fillna(3)
df["month"] = df["datetime"].dt.month.fillna(6)

df = df.drop(columns=["Date", "Time", "datetime"], errors="ignore")

# Select features
X = df.drop(columns=[TARGET], errors="ignore").select_dtypes(include=[np.number])
y = df[TARGET]

if X.empty:
    raise ValueError("X is empty — datetime parsing failed!")

numeric_cols = X.columns.tolist()

# Model Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluation
preds = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(
    {"pipeline": pipeline, "features": numeric_cols},
    SAVE_PATH
)

print("Model saved to", SAVE_PATH)

# ----- EXAMPLE PREDICTION -----
print("\n--- Example Prediction ---")
sample = {
    'PT08.S1(CO)': 933.0,
    'NMHC(GT)': -200.0,
    'C6H6(GT)': 6.4,
    'PT08.S2(NMHC)': 831.0,
    'NOx(GT)': 105.0,
    'PT08.S3(NOx)': 888.0,
    'NO2(GT)': 72.0,
    'PT08.S4(NO2)': 1514.0,
    'PT08.S5(O3)': 710.0,
    'T': 26.0,
    'RH': 34.9,
    'AH': 1.1509,
    'hour': 12,        # No more NaN
    'dayofweek': 3,
    'month': 6
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample], columns=numeric_cols)

prediction = pipeline.predict(sample_df)[0]
print("Input:", sample)
print("Prediction:", prediction)
print("Actual: 1.3")
