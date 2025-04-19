import pandas as pd
import plotly.express as px
from pathlib import Path

PROCESSED_DATA_PATH = Path("../data/processed")
VIS_PATH = Path("../visualizations")
VIS_PATH.mkdir(exist_ok=True)

def load_data():
    """Load processed data with validation."""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH / "climate_fires.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Processed data not found. Run preprocessing first.")

def plot_temperature_trend(df):
    """Plot interactive temperature trends with improved styling."""
    fig = px.line(
        df,
        x="date",
        y="temperature",
        color="region",
        title="Temperature Trends in Nepal",
        labels={"date": "Year", "temperature": "Mean Temperature (°C)"},
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis=dict(
            title="Year",
            dtick="M12",
            tickformat="%Y",
            showgrid=False,
            showline=True,
            linecolor="black",
            mirror=True
        ),
        yaxis=dict(
            title="Mean Temperature (°C)",
            showgrid=False,
            showline=True,
            linecolor="black",
            mirror=True,
            zeroline=False
        ),
        hovermode="x unified",
        legend=dict(
            title="Region",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Arial", size=12),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor="white"
    )
    
    # Save visualization
    fig.write_html(VIS_PATH / "temperature_trends.html")
    print("Temperature visualization saved to visualizations/")

if __name__ == "__main__":
    df = load_data()
    plot_temperature_trend(df)