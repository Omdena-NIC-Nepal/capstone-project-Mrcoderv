from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import pandas as pd

# Define paths
PROCESSED_DATA_PATH = Path("../data/processed")
MODEL_PATH = Path("../models")
MODEL_PATH.mkdir(exist_ok=True)

def train_model():
    """Train Random Forest classifier."""
    df = pd.read_csv(PROCESSED_DATA_PATH / "features.csv")
    
    # Assume climate_zone is the target (adjust as needed)
    X = df[["temperature", "precipitation", "SPI", "temp_lag1", "precip_lag1", "monsoon"]]
    y = df["climate_zone"]  # Replace with actual target column
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    joblib.dump(rf, MODEL_PATH / "rf_classifier.joblib")
    print("Model saved to models/rf_classifier.joblib")

if __name__ == "__main__":
    train_model()