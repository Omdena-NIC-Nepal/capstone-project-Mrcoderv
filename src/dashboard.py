import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from sklearn.preprocessing import StandardScaler

def load_data():
    try:
        df = pd.read_csv('data/processed/merged_climate_data.csv')
        required_cols = ['Year', 'Temperature', 'AnnualFireCount']
        if not all(col in df.columns for col in required_cols):
            st.error("Missing critical columns in dataset")
            return None
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
        df['AnnualFireCount'] = pd.to_numeric(df['AnnualFireCount'], errors='coerce')
        return df.dropna(subset=required_cols)
    except FileNotFoundError:
        st.error("Processed data file not found. Run preprocessing first.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def predict_next_temperature(df):
    try:
        model = joblib.load('models/temperature_model.joblib')
        scaler = StandardScaler()

        # Features used during training
        feature_cols = [col for col in df.columns if col not in ['Year', 'Temperature', 'AnnualFireCount']]
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

        # Prepare latest row
        latest = df_scaled.iloc[-1:].copy()
        for lag in [1, 2, 3]:
            latest[f'Temp_Lag_{lag}'] = df['Temperature'].shift(lag).iloc[-1]

        latest = latest.dropna()
        if latest.empty:
            return None

        model_input = latest[feature_cols + ['Year', 'Temp_Lag_1', 'Temp_Lag_2', 'Temp_Lag_3']]
        prediction = model.predict(model_input)[0]
        return prediction
    except Exception as e:
        st.warning(f"Prediction not available: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Nepal Climate Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    df = load_data()
    if df is None:
        return

    st.title("🇳🇵 Nepal Climate Impact Analysis Dashboard")

    # Predict next year temperature
    predicted_temp = predict_next_temperature(df)

    # Section 1: Temperature Trends
    with st.expander("Temperature Analysis", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.line(
                df, x='Year', y='Temperature',
                title="Annual Temperature Trends",
                labels={'Temperature': 'Temperature (°C)'},
                markers=True
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric(
                "Latest Temperature", 
                f"{df['Temperature'].iloc[-1]:.1f}°C",
                delta=f"{df['Temperature'].iloc[-1] - df['Temperature'].iloc[-2]:.1f}°C YoY"
            )
            if predicted_temp is not None:
                st.success(f"📈 Predicted Next Year Temperature: **{predicted_temp:.2f}°C**")
            else:
                st.warning("Prediction unavailable")

    # Section 2: Fire Analysis
    with st.expander("Wildfire Analysis"):
        tab1, tab2 = st.tabs(["Annual Trends", "Monthly Patterns"])
        with tab1:
            fig = px.bar(
                df, x='Year', y='AnnualFireCount',
                title="Annual Wildfire Counts",
                labels={'AnnualFireCount': 'Number of Fires'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fire_df = pd.read_csv('data/raw/monthly-fire-counts-in-n.csv')
            fire_df[['Year', 'Month']] = fire_df['Category'].str.split('-', expand=True)
            heatmap_df = fire_df.pivot_table(
                index='Year', columns='Month', values='FireChart', aggfunc='sum'
            )
            fig = px.imshow(
                heatmap_df,
                labels=dict(x="Month", y="Year", color="Fires"),
                title="Monthly Fire Patterns (2013-2023)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Section 3: Socio-Economic Correlations
    with st.expander("Climate-Socioeconomic Relationships"):
        feature_cols = [col for col in df.columns if col not in ['Year', 'Temperature', 'AnnualFireCount']]
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X-axis Variable", feature_cols)
        with col2:
            y_var = st.selectbox("Y-axis Variable", feature_cols, index=1)

        try:
            fig = px.scatter(
                df, x=x_var, y=y_var, color='Temperature',
                hover_data=['Year'], trendline="ols",
                labels={
                    'Temperature': 'Temp (°C)',
                    x_var: x_var.replace('_', ' '),
                    y_var: y_var.replace('_', ' ')
                },
                title=f"{x_var} vs {y_var} with Temperature Overlay"
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.error("Trendline requires `statsmodels`. Install with: `pip install statsmodels`")

if __name__ == "__main__":
    main()
