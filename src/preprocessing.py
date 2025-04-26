import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data():
    try:
        # Load datasets
        logging.info("Loading datasets...")
        temp_df = pd.read_csv('data/raw/observed-annual-average_temp.csv')
        socio_df = pd.read_csv(
            'data/raw/dem_socio_economic_data_npl.csv',
            comment='#',
            header=0
        )
        fire_df = pd.read_csv('data/raw/monthly-fire-counts-in-n.csv')

        # Process temperature data
        logging.info("Processing temperature data...")
        temp_df = temp_df.rename(columns={'Category': 'Year', 'Annual Mean': 'Temperature'})
        temp_df = temp_df[(temp_df['Year'] >= 1970) & (temp_df['Year'] <= 2023)]
        temp_df['Year'] = temp_df['Year'].astype(int)

        # Process fire data
        logging.info("Processing fire data...")
        fire_df[['Year', 'Month']] = fire_df['Category'].str.split('-', expand=True)
        fire_df['Year'] = fire_df['Year'].astype(int)
        annual_fires = fire_df.groupby('Year')['FireChart'].sum().reset_index()
        annual_fires = annual_fires.rename(columns={'FireChart': 'AnnualFireCount'})

        # Process socio-economic data
        logging.info("Processing socio-economic data...")
        socio_clean = socio_df.pivot_table(
            index=['year'],
            columns='indicator_id',
            values='value',
            aggfunc='mean'
        ).reset_index().rename(columns={'year': 'Year'})

        # Clean and deduplicate columns
        socio_clean.columns = [
            f"IND_{col}_MEAN" if str(col).isdigit() else col
            for col in socio_clean.columns
        ]
        socio_clean = socio_clean.loc[:, ~socio_clean.columns.duplicated()]

        # Merge datasets
        logging.info("Merging datasets...")
        merged_df = temp_df.merge(socio_clean, on='Year', how='inner')
        merged_df = merged_df.merge(annual_fires, on='Year', how='inner')

        # Drop mostly empty columns (less than 80% non-null)
        min_valid = int(0.8 * len(merged_df))
        merged_df = merged_df.dropna(axis=1, thresh=min_valid)

        # Drop rows with any remaining NaNs
        merged_df = merged_df.dropna()

        # Save cleaned data
        output_path = 'data/processed/merged_climate_data.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")

        return merged_df

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()
