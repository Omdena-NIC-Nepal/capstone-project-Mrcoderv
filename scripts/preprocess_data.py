import pandas as pd
import geopandas as gpd
from pathlib import Path

# Define paths
RAW_DATA_PATH = Path("../data/raw")
PROCESSED_DATA_PATH = Path("../data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

def load_climate_data():
    """Load and clean climate datasets."""
    files = [
        "observed_annual-average-mean-temp.csv",
        "observed_annual-average-largest-1-day-precipitation.csv",
        "observed_annual-relative-humidity.csv"
    ]    
    dfs = []  # Initialize list to store DataFrames
    for file in files:
        file_path = RAW_DATA_PATH / "climate" / file
        if not file_path.exists():
            print(f"ERROR: Missing file: {file_path}")
            raise FileNotFoundError(f"File {file} not found in raw data!")
        
        # Read and clean data
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        dfs.append(df)  # Add to list
    
    # Merge data
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["date", "region"], how="outer")
    return merged_df

def load_environment_data():
    """Load and clean environmental datasets."""
    fires_df = pd.read_csv(RAW_DATA_PATH / "environment" / "monthly-fire-counts-in-n.csv")
    fires_df["date"] = pd.to_datetime(fires_df["date"], errors="coerce")
    
    # Load shapefile
    fires_gdf = gpd.read_file(RAW_DATA_PATH / "environment" / "forest_fires_nepal.shp")
    fires_gdf = fires_gdf.to_crs("EPSG:4326")
    return fires_df, fires_gdf

def merge_datasets(climate_df, fires_df, fires_gdf):
    """Merge climate and environmental data."""
    merged_df = climate_df.merge(fires_df, on=["date", "region"], how="left")
    merged_gdf = gpd.GeoDataFrame(merged_df.merge(fires_gdf[["region", "geometry"]], on="region", how="left"))
    return merged_gdf

def save_processed_data(merged_gdf):
    """Save processed data."""
    merged_gdf.drop(columns=["geometry"]).to_csv(PROCESSED_DATA_PATH / "climate_fires.csv", index=False)
    merged_gdf.to_file(PROCESSED_DATA_PATH / "geospatial_data.gpkg", driver="GPKG")

if __name__ == "__main__":
    climate_df = load_climate_data()
    fires_df, fires_gdf = load_environment_data()
    merged_gdf = merge_datasets(climate_df, fires_df, fires_gdf)
    save_processed_data(merged_gdf)
    print("Data preprocessing complete. Output saved to data/processed/")