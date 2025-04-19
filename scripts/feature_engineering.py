import pandas as pd
import numpy as np
from scipy.stats import gamma
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROCESSED_DATA_PATH = Path("../data/processed")

def calculate_spi(precip, window=12):
    """Calculate Standardized Precipitation Index with validation."""
    precip_rolling = precip.rolling(window, min_periods=1).sum()
    precip_rolling = precip_rolling.dropna()
    
    if len(precip_rolling) < 2:
        return pd.Series(np.nan, index=precip.index)
    
    try:
        params = gamma.fit(precip_rolling, floc=0)
        spi = gamma.cdf(precip_rolling, *params)
        # Normalize to standard normal distribution
        spi = np.where(spi > 0.5, np.log(spi / (1 - spi)), -np.log((1 - spi) / spi))
        return pd.Series(spi, index=precip.index)
    except ValueError:
        return pd.Series(np.nan, index=precip.index)

def engineer_features():
    """Create features with proper error handling."""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH / "climate_fires.csv")
        df["date"] = pd.to_datetime(df["date"])
        
        # Feature engineering
        df["monsoon"] = df["date"].dt.month.isin([6, 7, 8, 9]).astype(int)
        df["temp_lag1"] = df.groupby("region")["temperature"].shift(1)
        df["precip_lag1"] = df.groupby("region")["precipitation"].shift(1)
        
        # SPI calculation
        df["SPI"] = df.groupby("region")["precipitation"].transform(calculate_spi)
        
        # Save features
        df.to_csv(PROCESSED_DATA_PATH / "features.csv", index=False)
        print("Features successfully saved.")
        
    except Exception as e:
        print(f"Feature engineering failed: {str(e)}")

if __name__ == "__main__":
    engineer_features()