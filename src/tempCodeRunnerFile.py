# Removed unused import
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Removed unused mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class ClimateModels:
    def __init__(self, data_path='data/processed/merged_climate_data.csv'):
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.scaler = StandardScaler()
        self._preprocess_data()

    def _preprocess_data(self):
        """Prepare data for modeling"""
        # Handle missing values
        if self.df.empty:
            raise ValueError("The dataset is empty. Please provide a valid dataset.")
        self.df = self.df.dropna().copy()
        if self.df.empty:
            raise ValueError("The dataset is empty after dropping missing values.")
        if self.df.empty:
            raise ValueError("The dataset is empty after dropping missing values.")
        
        # Create temporal features
        self.df['Year'] = pd.to_numeric(self.df['Year'])
        self.df = self.df.sort_values('Year')
        
        # Normalize features
        self.features = [col for col in self.df.columns 
                         if col not in ['Year', 'Temperature', 'AnnualFireCount']]
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

    def train_temperature_model(self, test_size=0.2):
        """Train Random Forest model for temperature prediction"""
        X = self.df[self.features + ['Year']].copy()
        y = self.df['Temperature']
        
        # Add lag features
        for lag in [1, 2, 3]:
            X[f'Temp_Lag_{lag}'] = y.shift(lag)
        
        X = X.dropna().copy()
        y = y.iloc[3:].copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )  # Variables are used later
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'r2': r2_score(y_test, preds)
        }
        
        self.models['temperature'] = model
        return model, metrics

    def train_fire_model(self, threshold=100):
        """Train classifier for extreme fire years"""
        X = self.df[self.features + ['Temperature']]
        y = (self.df['AnnualFireCount'] > threshold).astype(int)
        
        model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X, y)
        
        self.models['fire_risk'] = model
        return model

    def arima_temperature_forecast(self, steps=5):
        """ARIMA model for temperature forecasting"""
        ts_data = self.df.set_index('Year')['Temperature']
        model = ARIMA(ts_data, order=(2,1,1))
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=steps)
        return forecast

    def save_model(self, model_name, path):
        """Save trained model to disk"""
        if model_name in self.models:
            joblib.dump(self.models[model_name], path)
        else:
            raise ValueError(f"Model {model_name} not found")

    @staticmethod
    def load_model(path):
        """Load pre-trained model"""
        return joblib.load(path)

    def predict_temperature(self, features):
        """Make temperature prediction"""
        if 'temperature' not in self.models:
            raise ValueError("Train temperature model first")
            
        scaled_features = self.scaler.transform([features])
        return self.models['temperature'].predict(scaled_features)[0]
    # Removed misplaced line
def main():
    # Removed misplaced line
    cm = ClimateModels()
    _, metrics = cm.train_temperature_model()
    # Train temperature model
    temp_model, metrics = cm.train_temperature_model()
    print(f"Temperature Model Metrics: {metrics}")
    
    # Save model
    cm.save_model('temperature', 'models/temperature_model.joblib')
    
    # Load and use
    loaded_model = ClimateModels.load_model('models/temperature_model.joblib')
    sample_features = cm.df.iloc[-1][cm.features].tolist()
    prediction = loaded_model.predict([sample_features])
    print(f"Next year temperature prediction: {prediction[0]:.2f}°C")

if __name__ == "__main__":
    main()