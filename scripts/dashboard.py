import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

PROCESSED_DATA_PATH = Path("../data/processed")

@st.cache_data
def load_data():
    """Load data with validation."""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH / "climate_fires.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        st.error("Processed data not found. Run preprocessing first.")
        st.stop()

def main():
    st.title("Nepal Climate Change Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    regions = st.sidebar.multiselect("Regions", df["region"].unique(), default=df["region"].unique())
    date_range = st.sidebar.date_input("Date Range", [df["date"].min(), df["date"].max()])
    
    # Filter data
    filtered_df = df[
        (df["region"].isin(regions)) &
        (df["date"] >= pd.to_datetime(date_range[0])) &
        (df["date"] <= pd.to_datetime(date_range[1]))
    ]
    
    # Temperature visualization
    st.subheader("Temperature Trends")
    fig = px.line(filtered_df, x="date", y="temperature", color="region")
    st.plotly_chart(fig, use_container_width=True)
    
    # SPI Visualization
    st.subheader("Drought Indicators (SPI)")
    spi_fig = px.bar(filtered_df, x="date", y="SPI", color="region", barmode="group")
    st.plotly_chart(spi_fig, use_container_width=True)
    
    # Data download
    st.download_button(
        "Download Filtered Data",
        filtered_df.to_csv(index=False),
        "climate_data.csv",
        "text/csv"
    )

if __name__ == "__main__":
    main()